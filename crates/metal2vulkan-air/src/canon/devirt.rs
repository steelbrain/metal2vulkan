//! Devirtualization of the inlined entry.
//!
//! Inlining takes direct calls. A call through a function pointer is devirtualized here, over the
//! one inlined body, and inlining then runs again on the direct calls it made
//! ([`super::inline_rounds`]):
//!
//! - **A constant function address.** A call whose callee is the constant address of a defined
//!   function (or of a visible-function reference linking resolved), at that function's own
//!   type, is a direct call to it: the shape a function pointer argument has once the helper that
//!   calls it is inlined. A function an explicit call stack calls is called through its enter
//!   marker instead (`Cx::enters`). At another type it stays, and the post-check refuses it.
//! - **Visible-function tables.** The entry's `visible_function_table` parameters are read as
//!   buffers of `u32` words (`pipeline::table`): `size()` loads word 1,
//!   `is_null_visible_function_table` compares word 0 with 0, and entry `i` is word `2 + i`, the
//!   function pointer's value. A function pointer so made (through `phi`, `select`, `freeze`, and
//!   no-op casts whose other operands are words too) is carried as that word: an equality
//!   comparison of two words (or a word and null, 0) compares words, a `ptrtoint` is the word,
//!   and a call through it that passes no word as an argument is a **switch at the call** over
//!   the functions the pipeline can place in a table ([`Candidate`]s) whose type is the call's:
//!   each case calls one directly, and the default (a null entry, or a word no candidate of this
//!   type has; both undefined in Metal, see `pipeline::table`) calls nothing and yields zero. The
//!   switch dispatches on the call's own operand, so its divergence is the source call's: the rule
//!   for switches the translation introduces over one use does not apply to it.
//!
//! Nothing here refuses a call SCCP may still remove. A candidate that cannot be imported, of
//! the call's type or of a type unknown because its module did not decode, is called through a
//! declaration that the post-check refuses with that candidate's reason; a table call no
//! candidate can answer is the empty switch, whose default answers, because a table entry holds a
//! handle and a handle exists only for a linked function. A word used any other
//! way (stored, passed as an argument, returned, compared with something that is not one, or
//! merged with a pointer from memory) is an escape ([`Escapes`]): the use takes `poison` in its
//! place, after a call the post-check refuses as `DynamicCall` if it survives.
//!
//! **Tables loaded from memory**. A table held in memory, an argument buffer's
//! `visible_function_table` member, is the 8 bytes of its resource id (`gpuResourceID`, the
//! member's AIR type is a pointer of that size; an array member is an `i64` that `inttoptr`
//! makes the table). A table value made of such loads ([`ids`]: a `load` of it, an `inttoptr`
//! of an `i64`, and `phi`, `select`, `freeze`, and no-op casts of those and of null, `undef`, or
//! `poison`) is carried as its `i64` id: the load reads the id (`load i64` from the same
//! address), and a table operation on it is word `w` of the table of that id
//! ([`FunctionTable::Word`], which the access family reads through the stage's function-table
//! directory and set-1 heap): `size()` is word 1, `is_null_visible_function_table` word 0 = 0,
//! and entry `i` word `2 + i`, a function-pointer word as above. A store of one stores its id,
//! the 8 bytes Metal stores (a table copied into memory). An id used any other way is an
//! escape as a word is. A table operation on anything else (a table merged with an entry
//! parameter, or made otherwise) is left as it is, and a call through it is `DynamicCall`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::budget::Meter;
use crate::error::{DiagName, Error, Feature, MissingInput, Unsupported};
use crate::intrinsics::{AirIntrinsic, FunctionTable};
use crate::ir::TailKind;
use crate::ir::{AddrSpace, Align, Block, BlockId, Body, Call, Callee, CastOp, Const};
use crate::ir::{BinOp, IntFlags, IntPredicate, ModuleParts, Param, Type, TypeId, Value};
use crate::ir::{ConstExpr, ConstId, GlobalId};
use crate::ir::{Constant, FastMath, FuncId, Function, Idx, Inst, InstId, InstKind};

use super::fold::Folder;
use super::function_table::{FIRST_ENTRY_WORD, SIZE_WORD};
use super::link::{self, Candidate, Stub, Stubs};
use super::renumber::isomorphism;
use super::rewrite::{filled, invariant, scoped};
use super::sccp::{self, Fact, FcState};

/// An entry parameter that is a visible-function table.
#[derive(Clone, Copy, Debug)]
pub(super) struct TableParam {
    /// Its LLVM parameter index.
    pub arg: u32,
}

/// What a pass devirtualizes against.
pub(super) struct Cx<'a> {
    /// The entry's table parameters.
    pub tables: &'a [TableParam],
    /// The functions a table can hold.
    pub candidates: &'a [Candidate],
    /// The linked modules, whose types an unimported candidate's are.
    pub linked: &'a [link::LinkedModule],
    /// The entry's parameter types.
    pub params: &'a [TypeId],
    /// The functions an explicit call stack calls (recursion), each with its group's marker: a case
    /// of one of them is no direct call but a call of the marker with the switch's word before the
    /// call's arguments, one case block for all of a group's words (`super::recursion`).
    pub groups: &'a BTreeMap<FuncId, FuncId>,
    /// The functions an explicit call stack calls, each with its enter marker: a call through
    /// one's constant address calls the marker, as every direct call to it does.
    pub enters: &'a BTreeMap<FuncId, FuncId>,
    /// The word each function whose address is taken is carried as.
    pub words: &'a FunctionWords,
    /// The function constants as SCCP sees them, and the entry's name: what the constants bound
    /// so far decide of a call through a pointer is decided before the call becomes a switch
    /// ([`sccp::facts`]).
    pub fcs: &'a HashMap<GlobalId, FcState>,
    pub name: &'a DiagName,
}

/// The word a function's address is carried as.
///
/// A function a table can hold (a [`Candidate`]) has its handle's word, the one a table entry
/// holding it has, so a table's entry and the function's address compare as Metal compares them
/// (a table's entry holding a function equals that function's address). Every other function whose
/// address a body the entry reaches takes as a value has a word of the stage's own, past every
/// handle the pipeline numbers, ascending in function order: one only `private_functions` links,
/// one the module defines that is not visible, and a visible-function reference no linked function
/// resolves (whose every surviving use is refused as a call to it is). Metal's own value of a
/// function pointer is a GPU address that differs between pipelines of one descriptor, so no
/// program can depend on it; only equality, and the function called, are Metal's answers.
#[derive(Debug, Default)]
pub(super) struct FunctionWords {
    /// Each function's word: a definition's, or a stub declaration's that is not resolved.
    of: BTreeMap<FuncId, u32>,
    /// Per function type, the defined functions that have a word of the stage's own.
    own: BTreeMap<TypeId, Vec<(u32, FuncId)>>,
}

impl FunctionWords {
    /// The words of `candidates` and of `addressed`, the defined functions and unresolved stubs
    /// whose address a reached body takes, those no candidate is numbered from `first_own` in
    /// function order.
    pub(super) fn new(
        parts: &ModuleParts,
        candidates: &[Candidate],
        addressed: &BTreeSet<FuncId>,
        first_own: u32,
        m: &mut Meter,
    ) -> Result<FunctionWords, Error> {
        let mut words = FunctionWords::default();
        let entry = std::mem::size_of::<(FuncId, u32)>() * 2;
        for c in candidates {
            m.tick(1)?;
            if let Ok(g) = c.def {
                m.charge(entry)?;
                words.of.insert(g, c.word);
            }
        }
        let mut next = first_own;
        for &f in addressed {
            m.tick(1)?;
            if words.of.contains_key(&f) {
                continue;
            }
            let func = parts
                .funcs
                .get(f)
                .ok_or_else(|| invariant("an addressed function out of range"))?;
            m.charge(entry * 2)?;
            words.of.insert(f, next);
            if func.body.is_some() {
                words.own.entry(func.ty).or_default().push((next, f));
            }
            next = next
                .checked_add(1)
                .ok_or_else(|| invariant("function words past the word space"))?;
        }
        Ok(words)
    }

    /// The word of `f`, a definition or an unresolved stub, when it has one.
    pub(super) fn word(&self, f: FuncId) -> Option<u32> {
        self.of.get(&f).copied()
    }

    /// The defined functions of type `ty` that have a word of the stage's own.
    fn own(&self, ty: TypeId) -> &[(u32, FuncId)] {
        self.own.get(&ty).map_or(&[], Vec::as_slice)
    }
}

/// What one pass made of the body.
pub(super) struct Pass {
    pub body: Body,
    /// Whether it made a direct call to a defined function (inlining runs again).
    pub calls: bool,
}

/// The table a table operation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableRef {
    /// An entry parameter, read as a buffer of words.
    Param(Value),
    /// A table loaded from memory, by its resource id ([`ids`]): an id-web value, or null, `undef`,
    /// or `poison`.
    Id(Value),
}

/// How one instruction is rewritten.
#[derive(Clone, Copy, Debug)]
enum Plan {
    /// A function-pointer `phi`, `select`, or `freeze`, carried as its `i32` word.
    Retype,
    /// Replaced by this operand (a no-op cast of a function pointer or a table, a `ptrtoint` to
    /// `i32`, an `inttoptr` that makes a table of its id).
    Forward(Value),
    /// `size()`: word 1 of the table.
    Size(TableRef),
    /// `is_null_visible_function_table`: word 0 of the table is 0.
    IsNull(TableRef),
    /// Entry `index` of the table: word `2 + index`.
    Lookup { table: TableRef, index: Value },
    /// An equality comparison of function pointers, as words.
    Cmp,
    /// `ptrtoint` of a function pointer to another width than 32.
    Word(CastOp),
    /// A call made direct to this function.
    Direct(FuncId),
    /// A call through a table: the switch of this site.
    Site(usize),
    /// A table loaded from memory: the load of its resource id, `load i64` from its address.
    IdLoad,
    /// A table `phi`, `select`, or `freeze`, carried as its `i64` resource id.
    IdRetype,
    /// A table stored to memory: the store of its resource id, the 8 bytes Metal stores.
    IdStore,
    /// A function pointer loaded from memory: `load i64` from its address, the word in its low
    /// half.
    WordLoad,
    /// A function pointer made of an integer (`inttoptr`) other than 32 bits wide: that integer
    /// cast to the word's 32 bits by this.
    FromInt(CastOp),
    /// A function pointer stored to memory: its word, zero-extended to the pointer's 8 bytes.
    WordStore,
    /// A call through a word the function constants settle ([`Fact::Settled`]): a call of this
    /// refused declaration with the word, so SCCP deletes it or refuses the constant where the call
    /// stood, and the post-check refuses it if it survives, as the call through the pointer was.
    Unfolded(FuncId),
}

/// The cases of one call through a table: each case's word and its arm, the arms in block order,
/// and the zero of the call's type the default yields (`None` for a `void` call).
struct Site {
    cases: Vec<(u32, usize)>,
    arms: Vec<Arm>,
    zero: Option<ConstId>,
}

/// One case block of a [`Site`]: it calls `callee` with the call's arguments, or, for a group
/// (`Cx::groups`), its marker with the switch's word first.
#[derive(Clone, Copy, Debug)]
struct Arm {
    callee: FuncId,
    grouped: bool,
}

impl Site {
    /// The site of `words`, each a word and the function a table holding it calls: a function of
    /// a group ([`Cx::groups`]) shares its group's block, and another function an explicit call
    /// stack calls is called through its enter marker ([`Cx::enters`]).
    fn new(words: Vec<(u32, FuncId)>, cx: &Cx<'_>, zero: Option<ConstId>) -> Site {
        let mut cases = Vec::with_capacity(words.len());
        let mut arms: Vec<Arm> = Vec::with_capacity(words.len());
        for (word, f) in words {
            let arm = match cx.groups.get(&f) {
                Some(&marker) => {
                    let found = arms.iter().position(|a| a.grouped && a.callee == marker);
                    found.unwrap_or_else(|| {
                        arms.push(Arm {
                            callee: marker,
                            grouped: true,
                        });
                        arms.len() - 1
                    })
                }
                None => {
                    arms.push(Arm {
                        callee: cx.enters.get(&f).copied().unwrap_or(f),
                        grouped: false,
                    });
                    arms.len() - 1
                }
            };
            cases.push((word, arm));
        }
        Site { cases, arms, zero }
    }

    /// The heap bytes the site holds.
    fn bytes(&self) -> usize {
        self.cases.len() * std::mem::size_of::<(u32, usize)>()
            + self.arms.len() * std::mem::size_of::<Arm>()
    }
}

/// Devirtualize `body`, the entry's inlined body: `None` when nothing is to be done.
pub(super) fn pass(
    body: &Body,
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    cx: &Cx<'_>,
    m: &mut Meter,
) -> Result<Option<Pass>, Error> {
    scoped(m, |m, held| {
        let Some(analysis) = analyze(body, parts, stubs, cx, m, held)? else {
            return Ok(None);
        };
        let calls = analysis.calls;
        let body = emit(body, parts, &analysis, cx, m, held)?;
        Ok(Some(Pass { body, calls }))
    })
}

/// The plans of one pass.
struct Analysis {
    plans: BTreeMap<InstId, Plan>,
    sites: Vec<Site>,
    calls: bool,
    /// The function-pointer words (the web), which the rewrite carries as `i32`, and the tables
    /// carried as their `i64` resource ids ([`ids`]).
    words: BTreeSet<InstId>,
    /// The uses of a word in no position a word can take ([`Escapes`]).
    escapes: Escapes,
    /// The declaration of [`FunctionTable::Word`], when a table operation reads a table by its
    /// resource id.
    word: Option<FuncId>,
    /// Each function address's word as an `i32` constant, for the word operands.
    addrs: BTreeMap<ConstId, ConstId>,
    /// Each constant converting a function's address to an integer, as its word at that width:
    /// replaced in every operand.
    consts: BTreeMap<ConstId, ConstId>,
}

/// The uses of words that are no word operations (a word stored, passed as an argument, or
/// compared with something that is not one). Each such instruction takes `poison` of the
/// pointer's type in place of the word, and a call to a declaration refused as `DynamicCall`
/// precedes it (or, for a `phi`, its block's first other instruction): the post-check refuses
/// that call if SCCP keeps it, so the `poison` is never observed, and a use SCCP removes costs
/// nothing.
///
/// A use of the word of a visible-function reference no linked function resolves is refused
/// the same way, with the reference's own refusal (`NeedsInput(LinkedFunction)`), as Metal
/// refuses to make a pipeline whose function takes such an address ("unresolved
/// visible function reference"), but only where the use survives, as a call to it is.
#[derive(Default)]
struct Escapes {
    /// The instructions that take `poison` for their word operands.
    users: BTreeSet<InstId>,
    /// The refused declarations called before each instruction they precede.
    markers: BTreeMap<InstId, Vec<FuncId>>,
    /// The refused declaration of an escape (`void ()`).
    decl: Option<FuncId>,
    /// The refused declaration (`void ()`) of each unresolved stub whose word is used.
    missing: BTreeMap<FuncId, FuncId>,
    /// `poison` of each pointer type a word had.
    poison: BTreeMap<TypeId, ConstId>,
}

/// Who uses each instruction's result: a compressed table over the body.
struct Users {
    start: Vec<u32>,
    list: Vec<InstId>,
}

impl Users {
    fn build(body: &Body, m: &mut Meter, held: &mut usize) -> Result<Users, Error> {
        let n = body.insts.len();
        let mut start = filled(m, held, n + 1, 0u32)?;
        for inst in body.insts.values() {
            m.tick(1)?;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(i) = v {
                    if let Some(s) = start.get_mut(i.index() + 1) {
                        *s += 1;
                    }
                }
            });
        }
        for k in 1..start.len() {
            let prev = start.get(k - 1).copied().unwrap_or(0);
            if let Some(s) = start.get_mut(k) {
                *s += prev;
            }
        }
        let total = start.last().copied().unwrap_or(0) as usize;
        // A user is listed once per operand naming the value; every consumer is idempotent.
        let mut list = filled(m, held, total, InstId::new(0))?;
        let mut next = filled(m, held, n, 0u32)?;
        next.copy_from_slice(start.get(..n).unwrap_or(&[]));
        for (u, inst) in body.insts.iter() {
            m.tick(1)?;
            inst.kind.for_each_value(|v| {
                let Value::Inst(i) = v else { return };
                if let Some(slot) = next.get_mut(i.index()) {
                    if let Some(l) = list.get_mut(*slot as usize) {
                        *l = u;
                    }
                    *slot += 1;
                }
            });
        }
        Ok(Users { start, list })
    }

    fn of(&self, i: InstId) -> &[InstId] {
        let s = self.start.get(i.index()).copied().unwrap_or(0) as usize;
        let e = self.start.get(i.index() + 1).copied().unwrap_or(0) as usize;
        self.list.get(s..e).unwrap_or(&[])
    }
}

/// The block holding instruction `i` (blocks tile the arena in order).
fn block_of(body: &Body, i: InstId) -> Option<BlockId> {
    let mut lo = 0usize;
    let mut hi = body.blocks.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let b = BlockId::from_index(mid)?;
        let block = body.blocks.get(b)?;
        if i < block.start {
            hi = mid;
        } else if i >= block.end {
            lo = mid + 1;
        } else {
            return Some(b);
        }
    }
    None
}

impl Escapes {
    /// Record `u`, an instruction of `body` using a word of `words` in no position a word can
    /// take.
    fn record(
        &mut self,
        (body, words): (&Body, &dyn Fn(InstId) -> bool),
        (parts, stubs): (&mut ModuleParts, &mut Stubs),
        u: InstId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        if self.users.contains(&u) {
            return Ok(());
        }
        let inst = body
            .insts
            .get(u)
            .ok_or_else(|| invariant("a user out of range"))?;
        let charge = std::mem::size_of::<InstId>() * 4;
        m.charge(charge)?;
        *held += charge;
        self.users.insert(u);
        let decl = match self.decl {
            Some(d) => d,
            None => {
                let e = Unsupported::not_implemented(Feature::DynamicCall).into();
                let d = refused_marker(parts, stubs, b"function pointer", e, m)?;
                self.decl = Some(d);
                d
            }
        };
        self.mark(body, u, decl, m, held)?;
        let mut tys = Vec::new();
        inst.kind.for_each_value(|v| {
            if let Value::Inst(i) = v {
                if words(i) {
                    tys.push(body.insts.get(i).map(|w| w.ty));
                }
            }
        });
        for ty in tys {
            let ty = ty.ok_or_else(|| invariant("a word out of range"))?;
            if let std::collections::btree_map::Entry::Vacant(slot) = self.poison.entry(ty) {
                let poison = Constant {
                    ty,
                    value: Const::Poison,
                };
                let c = parts.consts.intern(poison, m)?;
                m.charge(std::mem::size_of::<(TypeId, ConstId)>() * 2)?;
                *held += std::mem::size_of::<(TypeId, ConstId)>() * 2;
                slot.insert(c);
            }
        }
        Ok(())
    }
}

impl Escapes {
    /// Call `decl`, a refused declaration, before `u`, or before a `phi`'s block's first other
    /// instruction.
    fn mark(
        &mut self,
        body: &Body,
        u: InstId,
        decl: FuncId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let inst = body
            .insts
            .get(u)
            .ok_or_else(|| invariant("a user out of range"))?;
        let at = match inst.kind {
            InstKind::Phi { .. } => {
                let block = block_of(body, u)
                    .and_then(|b| body.blocks.get(b))
                    .ok_or_else(|| invariant("a phi outside the blocks"))?;
                block
                    .inst_ids()
                    .find(|&i| {
                        !matches!(
                            body.insts.get(i).map(|i| &i.kind),
                            Some(InstKind::Phi { .. })
                        )
                    })
                    .ok_or_else(|| invariant("a block without a terminator"))?
            }
            _ => u,
        };
        let decls = self.markers.entry(at).or_default();
        if !decls.contains(&decl) {
            let charge = std::mem::size_of::<(InstId, FuncId)>() * 2;
            m.charge(charge)?;
            *held += charge;
            decls.push(decl);
        }
        Ok(())
    }

    /// Refuse `u`, which uses the word of `stub`, an unresolved stub, as a surviving call to it
    /// is refused (see the type's docs).
    fn missing(
        &mut self,
        (body, parts, stubs): (&Body, &mut ModuleParts, &mut Stubs),
        (u, stub): (InstId, FuncId),
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let decl = match self.missing.get(&stub) {
            Some(&d) => d,
            None => {
                let e = match stubs.get(stub) {
                    Some(Stub::Missing(s)) => Error::NeedsInput(MissingInput::LinkedFunction {
                        symbol: DiagName::new(String::from_utf8_lossy(&s.0)),
                    }),
                    Some(Stub::Refused(e)) => e.clone(),
                    Some(Stub::Resolved(_)) | None => {
                        return Err(invariant("a word of a stub that is not unresolved"))
                    }
                };
                let d = refused_marker(parts, stubs, b"visible function reference", e, m)?;
                m.charge(std::mem::size_of::<(FuncId, FuncId)>() * 2)?;
                *held += std::mem::size_of::<(FuncId, FuncId)>() * 2;
                self.missing.insert(stub, d);
                d
            }
        };
        self.mark(body, u, decl, m, held)
    }
}

/// The refused declaration of type `ret (i32)` a settled call through a word returning `ret`
/// calls with its word ([`Plan::Unfolded`]): `DynamicCall` if a call to it survives, as a call
/// through a pointer that is no constant is.
fn unfolded_decl(
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    ret: TypeId,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let i32t = parts.types.intern(Type::Int(32), m)?;
    let ty = Type::Function {
        ret,
        params: Box::new([i32t]),
        vararg: false,
    };
    let ty = parts.types.intern(ty, m)?;
    let e = Unsupported::not_implemented(Feature::DynamicCall).into();
    refused(parts, stubs, (b"function pointer", ty), e, m)
}

/// A refused declaration of type `void ()` named `name`, whose surviving call the post-check
/// refuses with `e`.
fn refused_marker(
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    name: &[u8],
    e: Error,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let void = parts.types.intern(Type::Void, m)?;
    let ty = Type::Function {
        ret: void,
        params: Box::new([]),
        vararg: false,
    };
    let ty = parts.types.intern(ty, m)?;
    refused(parts, stubs, (name, ty), e, m)
}

/// The defined function `callee` calls through its constant address at its own type, if any.
fn constant_target(
    parts: &ModuleParts,
    stubs: &Stubs,
    callee: Value,
    fn_ty: TypeId,
) -> Option<FuncId> {
    let Value::Const(c) = callee else {
        return None;
    };
    let Some(Const::FuncAddr(f)) = parts.consts.get(c).map(|k| &k.value) else {
        return None;
    };
    let target = match stubs.get(*f) {
        Some(link::Stub::Resolved(def)) => *def,
        _ => *f,
    };
    let func = parts.funcs.get(target)?;
    (func.body.is_some() && func.ty == fn_ty).then_some(target)
}

/// Whether `v` is a constant a function-pointer word can take the place of: null (word 0),
/// `undef`, or `poison`.
fn word_constant(parts: &ModuleParts, v: Value) -> bool {
    match v {
        Value::Const(c) => matches!(
            parts.consts.get(c).map(|k| &k.value),
            Some(Const::Null | Const::Undef | Const::Poison)
        ),
        Value::Arg(_) | Value::Inst(_) => false,
    }
}

fn is_ptr(parts: &ModuleParts, ty: TypeId) -> bool {
    matches!(parts.types.get(ty), Some(Type::Ptr { .. }))
}

/// A declaration whose surviving call the post-check refuses with `e`, named `name`, of type
/// `ty`.
fn refused(
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    (name, ty): (&[u8], TypeId),
    e: Error,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let id = declare(parts, (name, ty), m)?;
    stubs.refuse(id, e, m)?;
    Ok(id)
}

/// The declaration of [`FunctionTable::Word`], `i32 (i64, i32)`: the module's own when it
/// already has one of that name and type (an earlier pass made it), else a new one. The name is
/// no AIR intrinsic's; the call's callee states the operation.
fn word_decl(parts: &mut ModuleParts, m: &mut Meter) -> Result<FuncId, Error> {
    const NAME: &[u8] = b"visible_function_table word";
    let i32t = parts.types.intern(Type::Int(32), m)?;
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let ty = Type::Function {
        ret: i32t,
        params: Box::new([i64t, i32t]),
        vararg: false,
    };
    let ty = parts.types.intern(ty, m)?;
    let found = parts
        .funcs
        .iter()
        .find(|(_, f)| f.body.is_none() && f.ty == ty && parts.symbols.get(f.name) == Some(NAME));
    match found {
        Some((id, _)) => Ok(id),
        None => declare(parts, (NAME, ty), m),
    }
}

/// A new declaration named `name`, of function type `ty`.
pub(super) fn declare(
    parts: &mut ModuleParts,
    (name, ty): (&[u8], TypeId),
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let params = match parts.types.get(ty) {
        Some(Type::Function { params, .. }) => params.len(),
        _ => return Err(invariant("a call's type is not a function type")),
    };
    let sym = parts.symbols.intern(name, m)?;
    let mut ps = Vec::new();
    m.reserve(&mut ps, params)?;
    ps.resize(params, Param::default());
    let f = Function {
        name: sym,
        ty,
        linkage: crate::ir::Linkage::External,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        cconv: crate::ir::CallConv::C,
        attrs: Default::default(),
        ret_attrs: Default::default(),
        params: ps.into_boxed_slice(),
        body: None,
        section: None,
        align: None,
    };
    parts.funcs.push(f, m)
}

/// The type of value `v` of `body`.
fn value_ty(body: &Body, parts: &ModuleParts, cx: &Cx<'_>, v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Arg(a) => cx.params.get(a.index()).copied(),
        Value::Const(c) => parts.consts.ty(c),
    }
}

/// How table value `inst` is carried as its resource id, if it can be: a plain `load` reads the id
/// from the same address, an `inttoptr` of an `i64` is that `i64`, a no-op cast is its operand's
/// id, and a `phi`, `select`, or `freeze` merges ids. The operands whose ids it carries go to
/// `carried`.
fn id_plan(
    (body, parts, cx): (&Body, &ModuleParts, &Cx<'_>),
    inst: &Inst,
    carried: &mut Vec<Value>,
) -> Option<Plan> {
    if !is_ptr(parts, inst.ty) {
        return None;
    }
    match &inst.kind {
        InstKind::Load {
            volatile: false,
            atomic: None,
            ..
        } => Some(Plan::IdLoad),
        InstKind::Cast {
            op: CastOp::IntToPtr,
            v,
        } => {
            let bits = value_ty(body, parts, cx, *v).and_then(|t| parts.types.int_bits(t));
            (bits == Some(64)).then_some(Plan::Forward(*v))
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        } => {
            carried.push(*v);
            Some(Plan::Forward(*v))
        }
        InstKind::Freeze(v) => {
            carried.push(*v);
            Some(Plan::IdRetype)
        }
        InstKind::Select { t, f, .. } => {
            carried.extend([*t, *f]);
            Some(Plan::IdRetype)
        }
        InstKind::Phi { incoming, .. } => {
            carried.extend(incoming.iter().map(|&(x, _)| x));
            Some(Plan::IdRetype)
        }
        _ => None,
    }
}

/// The id web: the table values carried as their `i64` resource ids, found from `seeds` (the table
/// operands of table operations on no entry parameter) through each member's carried operands and
/// its `phi`, `select`, `freeze`, and no-op cast users. A value is a member when it has an
/// [`id_plan`] and every operand it carries is a member or null, `undef`, or `poison`; the others
/// are pruned to a fixed point, and a table operation on one is left as it is.
fn ids(
    (body, parts, cx): (&Body, &ModuleParts, &Cx<'_>),
    seeds: &[Value],
    users: &Users,
    m: &mut Meter,
    held: &mut usize,
) -> Result<BTreeMap<InstId, Plan>, Error> {
    let entry = std::mem::size_of::<(InstId, Plan)>() * 2;
    let one = std::mem::size_of::<InstId>() * 2;
    let mut members: BTreeMap<InstId, Plan> = BTreeMap::new();
    let mut seen: BTreeSet<InstId> = BTreeSet::new();
    let mut work: Vec<InstId> = Vec::new();
    let mut carried = Vec::new();
    let push = |work: &mut Vec<InstId>, i: InstId, m: &mut Meter, held: &mut usize| {
        m.charge(std::mem::size_of::<InstId>())?;
        *held += std::mem::size_of::<InstId>();
        work.push(i);
        Ok::<(), Error>(())
    };
    for &v in seeds {
        if let Value::Inst(i) = v {
            push(&mut work, i, m, held)?;
        }
    }
    while let Some(v) = work.pop() {
        m.tick(1)?;
        if seen.contains(&v) {
            continue;
        }
        m.charge(one)?;
        *held += one;
        seen.insert(v);
        let inst = body
            .insts
            .get(v)
            .ok_or_else(|| invariant("a table value out of range"))?;
        carried.clear();
        let Some(plan) = id_plan((body, parts, cx), inst, &mut carried) else {
            continue;
        };
        m.charge(entry)?;
        *held += entry;
        members.insert(v, plan);
        for &x in &carried {
            if let Value::Inst(i) = x {
                if !seen.contains(&i) {
                    push(&mut work, i, m, held)?;
                }
            }
        }
        for &u in users.of(v) {
            if seen.contains(&u) {
                continue;
            }
            let user = body
                .insts
                .get(u)
                .ok_or_else(|| invariant("a user out of range"))?;
            let carries = match &user.kind {
                InstKind::Phi { .. } | InstKind::Freeze(_) => true,
                InstKind::Select { c, .. } => *c != Value::Inst(v),
                InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ..
                } => true,
                _ => false,
            };
            if carries && is_ptr(parts, user.ty) {
                push(&mut work, u, m, held)?;
            }
        }
    }
    let mut check: Vec<InstId> = members.keys().copied().collect();
    m.charge(check.len() * std::mem::size_of::<InstId>())?;
    *held += check.len() * std::mem::size_of::<InstId>();
    while let Some(v) = check.pop() {
        m.tick(1)?;
        if !members.contains_key(&v) {
            continue;
        }
        let inst = body
            .insts
            .get(v)
            .ok_or_else(|| invariant("a table value out of range"))?;
        carried.clear();
        id_plan((body, parts, cx), inst, &mut carried);
        let fine = carried.iter().all(|&x| match x {
            Value::Inst(i) => members.contains_key(&i),
            other => word_constant(parts, other),
        });
        if !fine {
            members.remove(&v);
            for &u in users.of(v) {
                if members.contains_key(&u) {
                    m.charge(std::mem::size_of::<InstId>())?;
                    *held += std::mem::size_of::<InstId>();
                    check.push(u);
                }
            }
        }
    }
    Ok(members)
}

/// A table operation on no entry parameter: the operation, its table operand, and an entry lookup's
/// index.
struct IdOp {
    at: InstId,
    op: FunctionTable,
    table: Value,
    index: Option<Value>,
}

/// The function constant `v` holds the address of, a stub's linked definition for a resolved
/// stub's, with its word when it has one.
fn addr_word(parts: &ModuleParts, stubs: &Stubs, cx: &Cx<'_>, v: Value) -> Option<(FuncId, u32)> {
    let Value::Const(c) = v else {
        return None;
    };
    let Some(Const::FuncAddr(f)) = parts.consts.get(c).map(|k| &k.value) else {
        return None;
    };
    let target = match stubs.get(*f) {
        Some(Stub::Resolved(def)) => *def,
        _ => *f,
    };
    cx.words.word(target).map(|w| (target, w))
}

/// Whether `ty` is a function pointer's type: a pointer into the thread space, as a function's
/// address is.
fn is_fn_ptr(parts: &ModuleParts, ty: TypeId) -> bool {
    matches!(
        parts.types.get(ty),
        Some(Type::Ptr {
            space: AddrSpace::Thread
        })
    )
}

/// Whether `ptr`, an address, points into memory a function pointer can live in: a thread's or
/// its threadgroup's. Metal refuses a function pointer as a buffer's pointee, so one
/// in device or constant memory has no Metal meaning to keep.
fn holds_fn_ptrs(body: &Body, parts: &ModuleParts, cx: &Cx<'_>, ptr: Value) -> bool {
    matches!(
        value_ty(body, parts, cx, ptr).and_then(|t| parts.types.get(t)),
        Some(Type::Ptr {
            space: AddrSpace::Thread | AddrSpace::Threadgroup
        })
    )
}

/// Push to `out` the operands `kind` carries when it is a carrier of a function pointer: a
/// `phi`, a `select` (its condition aside), a `freeze`, or a no-op cast. Whether it is one.
fn carried(kind: &InstKind, out: &mut Vec<Value>) -> bool {
    match kind {
        InstKind::Phi { incoming, .. } => {
            out.extend(incoming.iter().map(|&(x, _)| x));
            true
        }
        InstKind::Select { t, f, .. } => {
            out.extend([*t, *f]);
            true
        }
        InstKind::Freeze(x)
        | InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v: x,
        } => {
            out.push(*x);
            true
        }
        _ => false,
    }
}

/// What a call through a word may call ([`reach`]).
#[derive(Default)]
struct Reach {
    /// The functions whose constant addresses flow into the word.
    funcs: BTreeSet<FuncId>,
    /// Whether a word that may be any function's flows into it too: a table's entry, a pointer
    /// loaded from memory, or one made of an integer.
    any: bool,
    /// Whether a table's entry flows into it.
    table: bool,
}

/// What the word `callee`, a member of `web`, may hold: the carriers' operands back to the
/// constant addresses and the words that may be any function's.
fn reach(
    (body, parts, stubs, cx): (&Body, &ModuleParts, &Stubs, &Cx<'_>),
    plans: &BTreeMap<InstId, Plan>,
    callee: Value,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Reach, Error> {
    let mut r = Reach::default();
    let mut stack = vec![callee];
    let mut seen = BTreeSet::new();
    let mut ops = Vec::new();
    while let Some(v) = stack.pop() {
        m.tick(1)?;
        let Value::Inst(i) = v else {
            if let Some((f, _)) = addr_word(parts, stubs, cx, v) {
                m.charge(std::mem::size_of::<FuncId>() * 2)?;
                *held += std::mem::size_of::<FuncId>() * 2;
                r.funcs.insert(f);
            }
            continue;
        };
        if !seen.insert(i) {
            continue;
        }
        m.charge(std::mem::size_of::<InstId>() * 3)?;
        *held += std::mem::size_of::<InstId>() * 3;
        let inst = body
            .insts
            .get(i)
            .ok_or_else(|| invariant("a word out of range"))?;
        ops.clear();
        if carried(&inst.kind, &mut ops) {
            m.charge(ops.len() * std::mem::size_of::<Value>())?;
            *held += ops.len() * std::mem::size_of::<Value>();
            stack.extend(ops.iter().copied());
            continue;
        }
        r.any = true;
        if matches!(plans.get(&i), Some(Plan::Lookup { .. })) {
            r.table = true;
        }
    }
    Ok(r)
}

fn analyze(
    body: &Body,
    parts: &mut ModuleParts,
    stubs: &mut Stubs,
    cx: &Cx<'_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Option<Analysis>, Error> {
    let mut plans: BTreeMap<InstId, Plan> = BTreeMap::new();
    let mut calls = false;
    // The function-pointer words (the web).
    let mut web: BTreeSet<InstId> = BTreeSet::new();
    let mut seeds = Vec::new();
    let mut id_ops: Vec<IdOp> = Vec::new();
    // The values a use needs as words: a call's callee that is no constant, a carrier of a
    // function's address, and what a function's address is compared with.
    let mut demand: Vec<InstId> = Vec::new();
    // The calls through a pointer that is no constant: each call, its callee, and its type.
    let mut indirect: Vec<(InstId, InstId, TypeId)> = Vec::new();
    // The stores of a function's address.
    let mut addr_stores: Vec<InstId> = Vec::new();
    // Each function address's word as an `i32` constant, and each constant converting a
    // function's address to an integer (`ptrtoint`) as its word at that width.
    let mut addrs: BTreeMap<ConstId, ConstId> = BTreeMap::new();
    let mut consts: BTreeMap<ConstId, ConstId> = BTreeMap::new();
    // The instructions whose `ptrtoint` constants are words.
    let mut const_users: Vec<InstId> = Vec::new();
    let mut read: BTreeSet<ConstId> = BTreeSet::new();
    let charge = std::mem::size_of::<(InstId, Plan)>() * 2;
    let mut operands: Vec<Value> = Vec::new();
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        // The functions' addresses among the operands (a call's callee is no operand here).
        operands.clear();
        match &inst.kind {
            InstKind::Call(call) => operands.extend(call.args.iter().copied()),
            kind => kind.for_each_value(|v| operands.push(v)),
        }
        let mut converts = false;
        for &v in &operands {
            let Value::Const(c) = v else {
                continue;
            };
            if let Some((_, w)) = addr_word(parts, stubs, cx, v) {
                if read.insert(c) {
                    m.charge(std::mem::size_of::<(ConstId, ConstId)>() * 3)?;
                    *held += std::mem::size_of::<(ConstId, ConstId)>() * 3;
                    let i32t = parts.types.intern(Type::Int(32), m)?;
                    let word = Constant {
                        ty: i32t,
                        value: Const::Int(u64::from(w)),
                    };
                    let word = parts.consts.intern(word, m)?;
                    addrs.insert(c, word);
                }
                continue;
            }
            let Some(k) = parts.consts.get(c) else {
                continue;
            };
            let Const::Expr(ConstExpr::Cast {
                op: CastOp::PtrToInt,
                ops: [a],
            }) = &k.value
            else {
                continue;
            };
            let (a, ty) = (*a, k.ty);
            let Some((_, w)) = addr_word(parts, stubs, cx, Value::Const(a)) else {
                continue;
            };
            converts = true;
            if read.insert(c) {
                let bits = parts.types.int_bits(ty).unwrap_or(64);
                let w = if bits >= 32 {
                    u64::from(w)
                } else {
                    u64::from(w) & ((1u64 << bits) - 1)
                };
                m.charge(std::mem::size_of::<(ConstId, ConstId)>() * 3)?;
                *held += std::mem::size_of::<(ConstId, ConstId)>() * 3;
                let word = parts.consts.intern(
                    Constant {
                        ty,
                        value: Const::Int(w),
                    },
                    m,
                )?;
                consts.insert(c, word);
            }
        }
        if converts {
            m.charge(std::mem::size_of::<InstId>())?;
            *held += std::mem::size_of::<InstId>();
            const_users.push(i);
        }
        let is_addr = |v: Value| addr_word(parts, stubs, cx, v).is_some();
        match &inst.kind {
            InstKind::ICmp {
                pred: IntPredicate::Eq | IntPredicate::Ne,
                a,
                b,
            } => {
                for (x, y) in [(*a, *b), (*b, *a)] {
                    if let (true, Value::Inst(y)) = (is_addr(x), y) {
                        m.charge(std::mem::size_of::<InstId>())?;
                        *held += std::mem::size_of::<InstId>();
                        demand.push(y);
                    }
                }
            }
            InstKind::Store { val, ptr, .. }
                if is_addr(*val) && !is_addr(*ptr) && holds_fn_ptrs(body, parts, cx, *ptr) =>
            {
                m.charge(std::mem::size_of::<InstId>())?;
                *held += std::mem::size_of::<InstId>();
                addr_stores.push(i);
            }
            kind if is_ptr(parts, inst.ty) => {
                operands.clear();
                if carried(kind, &mut operands) && operands.iter().any(|&v| is_addr(v)) {
                    m.charge(std::mem::size_of::<InstId>())?;
                    *held += std::mem::size_of::<InstId>();
                    demand.push(i);
                }
            }
            _ => {}
        }
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        match &call.callee {
            Callee::Air {
                op: AirIntrinsic::FunctionTable(op),
                ..
            } => {
                if !matches!(
                    op,
                    FunctionTable::Size | FunctionTable::IsNull | FunctionTable::GetFunctionPointer
                ) {
                    continue;
                }
                let Some(&table) = call.args.first() else {
                    continue;
                };
                let index = match op {
                    FunctionTable::GetFunctionPointer => Some(
                        *call
                            .args
                            .get(1)
                            .ok_or_else(|| invariant("a table lookup without an index"))?,
                    ),
                    _ => None,
                };
                let Value::Arg(a) = table else {
                    m.charge(std::mem::size_of::<IdOp>())?;
                    *held += std::mem::size_of::<IdOp>();
                    id_ops.push(IdOp {
                        at: i,
                        op: *op,
                        table,
                        index,
                    });
                    continue;
                };
                if !cx.tables.iter().any(|t| t.arg == a.raw()) {
                    continue;
                }
                let plan = match index {
                    None if *op == FunctionTable::Size => Plan::Size(TableRef::Param(table)),
                    None => Plan::IsNull(TableRef::Param(table)),
                    Some(index) => {
                        m.charge(charge)?;
                        *held += charge;
                        web.insert(i);
                        seeds.push(i);
                        Plan::Lookup {
                            table: TableRef::Param(table),
                            index,
                        }
                    }
                };
                m.charge(charge)?;
                *held += charge;
                plans.insert(i, plan);
            }
            Callee::Indirect { callee, fn_ty } => {
                if let Some(target) = constant_target(parts, stubs, *callee, *fn_ty) {
                    m.charge(charge)?;
                    *held += charge;
                    match cx.enters.get(&target) {
                        Some(&enter) => {
                            plans.insert(i, Plan::Direct(enter));
                        }
                        None => {
                            plans.insert(i, Plan::Direct(target));
                            calls = true;
                        }
                    }
                } else if let Value::Inst(c) = callee {
                    m.charge(std::mem::size_of::<(InstId, InstId, TypeId)>())?;
                    *held += std::mem::size_of::<(InstId, InstId, TypeId)>();
                    indirect.push((i, *c, *fn_ty));
                }
            }
            _ => {}
        }
    }
    // What the function constants bound so far decide of the values a use needs as words, by
    // SCCP's fixpoint over the body as it stands (`sccp::facts`). A call whose callee they make one
    // function's address is a direct call to it. A call whose callee is computed from an unsupplied
    // constant is settled: it is left as it was before item 27, so SCCP refuses the constant where
    // the call stands (or deletes the call), and no switch is built over every function its pointer
    // might be. A value they settle is no word.
    let mut settled: BTreeSet<InstId> = BTreeSet::new();
    if !indirect.is_empty() || !demand.is_empty() {
        scoped(m, |m, scratch| {
            let facts = {
                let mut scx = sccp::Context {
                    folder: Folder {
                        types: &parts.types,
                        consts: &mut parts.consts,
                        held: 0,
                    },
                    globals: &parts.globals,
                    fcs: cx.fcs,
                    name: cx.name.clone(),
                };
                sccp::facts(body, &mut scx, m, scratch)?
            };
            demand.retain(|&d| facts.of(Value::Inst(d)) != Fact::Settled);
            for &(call, callee, fn_ty) in &indirect {
                m.tick(1)?;
                let fact = facts.of(Value::Inst(callee));
                let target = match fact {
                    Fact::Const(k) => constant_target(parts, stubs, Value::Const(k), fn_ty),
                    Fact::Settled | Fact::Varies => None,
                };
                match (fact, target) {
                    (Fact::Settled, _) => {
                        m.charge(std::mem::size_of::<InstId>() * 3)?;
                        *held += std::mem::size_of::<InstId>() * 3;
                        settled.insert(call);
                    }
                    (_, Some(target)) => {
                        m.charge(charge)?;
                        *held += charge;
                        match cx.enters.get(&target) {
                            Some(&enter) => {
                                plans.insert(call, Plan::Direct(enter));
                            }
                            None => {
                                plans.insert(call, Plan::Direct(target));
                                calls = true;
                            }
                        }
                    }
                    (_, None) => {
                        m.charge(std::mem::size_of::<InstId>())?;
                        *held += std::mem::size_of::<InstId>();
                        demand.push(callee);
                    }
                }
            }
            Ok(())
        })?;
    }
    let users = match seeds.is_empty() && id_ops.is_empty() && demand.is_empty() {
        true => None,
        false => Some(Users::build(body, m, held)?),
    };
    // The tables loaded from memory: each operation on one carried as its resource id reads its
    // words by that id.
    let mut idweb: BTreeMap<InstId, Plan> = BTreeMap::new();
    let mut word = None;
    if let (Some(users), false) = (&users, id_ops.is_empty()) {
        let operands: Vec<Value> = id_ops.iter().map(|o| o.table).collect();
        m.charge(operands.len() * std::mem::size_of::<Value>())?;
        *held += operands.len() * std::mem::size_of::<Value>();
        idweb = ids((body, parts, cx), &operands, users, m, held)?;
        for o in &id_ops {
            let read = match o.table {
                Value::Inst(t) => idweb.contains_key(&t),
                other => word_constant(parts, other),
            };
            if !read {
                continue;
            }
            let table = TableRef::Id(o.table);
            let plan = match (o.op, o.index) {
                (FunctionTable::Size, _) => Plan::Size(table),
                (FunctionTable::IsNull, _) => Plan::IsNull(table),
                (_, Some(index)) => {
                    m.charge(charge)?;
                    *held += charge;
                    web.insert(o.at);
                    seeds.push(o.at);
                    Plan::Lookup { table, index }
                }
                (_, None) => return Err(invariant("a table lookup without an index")),
            };
            m.charge(charge)?;
            *held += charge;
            plans.insert(o.at, plan);
            if word.is_none() {
                word = Some(word_decl(parts, m)?);
            }
        }
        for (&v, &plan) in &idweb {
            m.charge(charge)?;
            *held += charge;
            plans.insert(v, plan);
        }
    }
    let mut sites = Vec::new();
    let mut escapes = Escapes::default();
    if !seeds.is_empty() || !demand.is_empty() {
        let users = users
            .as_ref()
            .ok_or_else(|| invariant("function-pointer words without their users"))?;
        // The web: the table lookups, and every value connected to a word's use through the
        // carriers and the equality comparisons that is a carrier, a pointer loaded from memory, or
        // one made of an integer. A carrier is followed back to its operands and forward to its
        // carrier users; a comparison of two pointers makes both words when one is.
        let mut work: Vec<InstId> = seeds
            .iter()
            .copied()
            .chain(demand.iter().copied())
            .collect();
        m.charge(work.len() * std::mem::size_of::<InstId>())?;
        *held += work.len() * std::mem::size_of::<InstId>();
        let mut tried: BTreeSet<InstId> = BTreeSet::new();
        let mut ops = Vec::new();
        while let Some(v) = work.pop() {
            m.tick(1)?;
            if !tried.insert(v) {
                continue;
            }
            m.charge(std::mem::size_of::<InstId>() * 3)?;
            *held += std::mem::size_of::<InstId>() * 3;
            let inst = body
                .insts
                .get(v)
                .ok_or_else(|| invariant("a word out of range"))?;
            ops.clear();
            // A lookup is already a member.
            let member = if web.contains(&v) {
                true
            } else if is_ptr(parts, inst.ty) && carried(&inst.kind, &mut ops) {
                m.charge(ops.len() * std::mem::size_of::<InstId>() + charge)?;
                *held += ops.len() * std::mem::size_of::<InstId>() + charge;
                work.extend(ops.iter().filter_map(|&x| match x {
                    Value::Inst(x) => Some(x),
                    _ => None,
                }));
                web.insert(v);
                true
            } else if is_fn_ptr(parts, inst.ty) {
                let plan = match &inst.kind {
                    InstKind::Load {
                        ptr,
                        volatile: false,
                        atomic: None,
                        ..
                    } if holds_fn_ptrs(body, parts, cx, *ptr) => Some(Plan::WordLoad),
                    InstKind::Cast {
                        op: CastOp::IntToPtr,
                        v: x,
                    } => {
                        match value_ty(body, parts, cx, *x).and_then(|t| parts.types.int_bits(t)) {
                            Some(32) => Some(Plan::Forward(*x)),
                            Some(bits) if bits > 32 => Some(Plan::FromInt(CastOp::Trunc)),
                            Some(_) => Some(Plan::FromInt(CastOp::ZExt)),
                            None => None,
                        }
                    }
                    _ => None,
                };
                match plan {
                    Some(plan) => {
                        m.charge(charge * 2)?;
                        *held += charge * 2;
                        plans.insert(v, plan);
                        web.insert(v);
                        true
                    }
                    None => false,
                }
            } else {
                false
            };
            if !member {
                continue;
            }
            for &u in users.of(v) {
                let user = body
                    .insts
                    .get(u)
                    .ok_or_else(|| invariant("a user out of range"))?;
                let carries = match &user.kind {
                    InstKind::Phi { .. } | InstKind::Freeze(_) => true,
                    InstKind::Select { c, .. } => *c != Value::Inst(v),
                    InstKind::Cast {
                        op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                        ..
                    } => true,
                    InstKind::ICmp {
                        pred: IntPredicate::Eq | IntPredicate::Ne,
                        a,
                        b,
                    } => {
                        for x in [*a, *b] {
                            if let Value::Inst(x) = x {
                                if x != v {
                                    m.charge(std::mem::size_of::<InstId>())?;
                                    *held += std::mem::size_of::<InstId>();
                                    work.push(x);
                                }
                            }
                        }
                        false
                    }
                    _ => false,
                };
                if carries && is_ptr(parts, user.ty) {
                    m.charge(std::mem::size_of::<InstId>())?;
                    *held += std::mem::size_of::<InstId>();
                    work.push(u);
                }
            }
        }
        // A carrier any other operand of which is no word (a `phi` or `select` of a word and a
        // pointer from an argument) is no word either, and neither is what it carries on: it
        // stays a pointer, and the words it takes are escapes.
        let mut check: Vec<InstId> = web.iter().copied().collect();
        m.charge(check.len() * std::mem::size_of::<InstId>())?;
        *held += check.len() * std::mem::size_of::<InstId>();
        while let Some(v) = check.pop() {
            m.tick(1)?;
            if !web.contains(&v) {
                continue;
            }
            let inst = body
                .insts
                .get(v)
                .ok_or_else(|| invariant("a word out of range"))?;
            ops.clear();
            // A lookup, a load, or an integer's pointer is a word whatever its operands.
            if !carried(&inst.kind, &mut ops) {
                continue;
            }
            let fine = ops.iter().all(|&x| match x {
                Value::Inst(i) => web.contains(&i),
                other => {
                    word_constant(parts, other) || addr_word(parts, stubs, cx, other).is_some()
                }
            });
            if !fine {
                web.remove(&v);
                for &u in users.of(v) {
                    if web.contains(&u) {
                        m.charge(std::mem::size_of::<InstId>())?;
                        *held += std::mem::size_of::<InstId>();
                        check.push(u);
                    }
                }
            }
        }
        // Every use of every word.
        let mut refusals: BTreeMap<(Option<usize>, TypeId), FuncId> = BTreeMap::new();
        // The refused declaration of each return type a settled call has.
        let mut unfolded: BTreeMap<TypeId, FuncId> = BTreeMap::new();
        let members: Vec<InstId> = web.iter().copied().collect();
        m.charge(members.len() * std::mem::size_of::<InstId>())?;
        *held += members.len() * std::mem::size_of::<InstId>();
        for &v in &members {
            let inst = body
                .insts
                .get(v)
                .ok_or_else(|| invariant("a word out of range"))?;
            if let std::collections::btree_map::Entry::Vacant(slot) = plans.entry(v) {
                let plan = match &inst.kind {
                    InstKind::Cast { v: x, .. } => Plan::Forward(*x),
                    _ => Plan::Retype,
                };
                m.charge(charge)?;
                *held += charge;
                slot.insert(plan);
            }
            let word = Value::Inst(v);
            for &u in users.of(v) {
                m.tick(1)?;
                if web.contains(&u) {
                    continue;
                }
                let user = body
                    .insts
                    .get(u)
                    .ok_or_else(|| invariant("a user out of range"))?;
                let is_word = |x: Value| match x {
                    Value::Inst(i) => web.contains(&i),
                    Value::Const(c) => {
                        matches!(parts.consts.get(c).map(|k| &k.value), Some(Const::Null))
                            || addr_word(parts, stubs, cx, x).is_some()
                    }
                    Value::Arg(_) => false,
                };
                // Each user is decided by its own operands, whichever word reaches it first.
                let plan = match &user.kind {
                    InstKind::ICmp {
                        pred: IntPredicate::Eq | IntPredicate::Ne,
                        a,
                        b,
                    } if is_word(*a) && is_word(*b) => Some(Plan::Cmp),
                    InstKind::Cast {
                        op: CastOp::PtrToInt,
                        v: x,
                    } => match parts.types.int_bits(user.ty) {
                        Some(32) => Some(Plan::Forward(*x)),
                        Some(bits) if bits > 32 => Some(Plan::Word(CastOp::ZExt)),
                        Some(_) => Some(Plan::Word(CastOp::Trunc)),
                        None => None,
                    },
                    // A word stored to memory a function pointer can live in: its 8 bytes.
                    InstKind::Store { val, ptr, .. }
                        if *val == word && *ptr != word && holds_fn_ptrs(body, parts, cx, *ptr) =>
                    {
                        Some(Plan::WordStore)
                    }
                    // A call through the word that does not also pass a word: a switch over the
                    // functions it may hold. A word passed as an argument is an escape, whatever
                    // the call.
                    InstKind::Call(call)
                        if !call
                            .args
                            .iter()
                            .any(|x| matches!(x, Value::Inst(i) if web.contains(i))) =>
                    {
                        match call.callee {
                            Callee::Indirect { callee, fn_ty } if callee == word => {
                                if plans.contains_key(&u) {
                                    continue;
                                }
                                if settled.contains(&u) {
                                    let decl = match unfolded.get(&user.ty) {
                                        Some(&d) => d,
                                        None => {
                                            let d = unfolded_decl(parts, stubs, user.ty, m)?;
                                            m.charge(std::mem::size_of::<(TypeId, FuncId)>() * 2)?;
                                            *held += std::mem::size_of::<(TypeId, FuncId)>() * 2;
                                            unfolded.insert(user.ty, d);
                                            d
                                        }
                                    };
                                    Some(Plan::Unfolded(decl))
                                } else {
                                    let r = reach((body, parts, stubs, cx), &plans, word, m, held)?;
                                    let site = site(
                                        (parts, stubs),
                                        cx,
                                        (fn_ty, user.ty),
                                        &r,
                                        &mut refusals,
                                        m,
                                    )?;
                                    Some(match site {
                                        Ok(s) => {
                                            calls |= s.arms.iter().any(|a| {
                                                parts
                                                    .funcs
                                                    .get(a.callee)
                                                    .is_some_and(|f| f.body.is_some())
                                            });
                                            let bytes = s.bytes();
                                            m.charge(bytes)?;
                                            *held += bytes;
                                            sites.push(s);
                                            Plan::Site(sites.len() - 1)
                                        }
                                        Err(none) => Plan::Direct(none),
                                    })
                                }
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match plan {
                    Some(plan) => {
                        if let std::collections::btree_map::Entry::Vacant(slot) = plans.entry(u) {
                            m.charge(charge)?;
                            *held += charge;
                            slot.insert(plan);
                        }
                    }
                    None => {
                        let words = |i: InstId| web.contains(&i) || idweb.contains_key(&i);
                        escapes.record((body, &words), (parts, stubs), u, m, held)?
                    }
                }
            }
        }
    }
    // A function's address stored to memory: its word's 8 bytes.
    for &u in &addr_stores {
        if let std::collections::btree_map::Entry::Vacant(slot) = plans.entry(u) {
            m.charge(charge)?;
            *held += charge;
            slot.insert(Plan::WordStore);
        }
    }
    // Every use of every resource id: a table operation reading it, a member carrying it on,
    // or a store of it (the id is the table's representation in memory), else an escape.
    if let Some(users) = &users {
        let mut carried = Vec::new();
        for &v in idweb.keys() {
            for &u in users.of(v) {
                m.tick(1)?;
                let user = body
                    .insts
                    .get(u)
                    .ok_or_else(|| invariant("a user out of range"))?;
                if let InstKind::Store { val, ptr, .. } = &user.kind {
                    if *val == Value::Inst(v) && *ptr != Value::Inst(v) {
                        if let std::collections::btree_map::Entry::Vacant(slot) = plans.entry(u) {
                            m.charge(charge)?;
                            *held += charge;
                            slot.insert(Plan::IdStore);
                        }
                        continue;
                    }
                }
                let fine = match plans.get(&u) {
                    Some(
                        Plan::Size(TableRef::Id(t))
                        | Plan::IsNull(TableRef::Id(t))
                        | Plan::Lookup {
                            table: TableRef::Id(t),
                            ..
                        },
                    ) => {
                        // The table operand alone (an entry index is no table).
                        *t == Value::Inst(v)
                            && match &user.kind {
                                InstKind::Call(call) => {
                                    call.args.iter().skip(1).all(|&x| x != Value::Inst(v))
                                }
                                _ => false,
                            }
                    }
                    _ if idweb.contains_key(&u) => {
                        carried.clear();
                        id_plan((body, parts, cx), user, &mut carried);
                        let mut operands = 0usize;
                        user.kind
                            .for_each_value(|x| operands += usize::from(x == Value::Inst(v)));
                        let as_carried = carried.iter().filter(|&&x| x == Value::Inst(v)).count();
                        operands == as_carried
                    }
                    _ => false,
                };
                if !fine {
                    let words = |i: InstId| web.contains(&i) || idweb.contains_key(&i);
                    escapes.record((body, &words), (parts, stubs), u, m, held)?;
                }
            }
        }
    }
    // A use of an unresolved stub's word (a word operand the rewrite replaces, or a `ptrtoint`
    // of its address) is refused where it survives ([`Escapes`]).
    let word_users = plans
        .iter()
        .filter(|(_, p)| matches!(p, Plan::Retype | Plan::Cmp | Plan::WordStore))
        .map(|(&u, _)| u)
        .chain(const_users.iter().copied())
        .collect::<Vec<_>>();
    m.charge(word_users.len() * std::mem::size_of::<InstId>())?;
    *held += word_users.len() * std::mem::size_of::<InstId>();
    for u in word_users {
        m.tick(1)?;
        let user = body
            .insts
            .get(u)
            .ok_or_else(|| invariant("a user out of range"))?;
        let mut stubs_used = Vec::new();
        user.kind.for_each_value(|x| {
            let Value::Const(c) = x else { return };
            let c = match parts.consts.get(c).map(|k| &k.value) {
                Some(Const::Expr(ConstExpr::Cast {
                    op: CastOp::PtrToInt,
                    ops: [a],
                })) => *a,
                _ => c,
            };
            if let Some((f, _)) = addr_word(parts, stubs, cx, Value::Const(c)) {
                if parts.funcs.get(f).is_some_and(|f| f.body.is_none()) {
                    stubs_used.push(f);
                }
            }
        });
        for f in stubs_used {
            escapes.missing((body, parts, stubs), (u, f), m, held)?;
        }
    }
    if plans.is_empty() && consts.is_empty() && escapes.markers.is_empty() {
        return Ok(None);
    }
    let words: BTreeSet<InstId> = web.iter().chain(idweb.keys()).copied().collect();
    m.charge(words.len() * std::mem::size_of::<InstId>() * 2)?;
    *held += words.len() * std::mem::size_of::<InstId>() * 2;
    Ok(Some(Analysis {
        plans,
        sites,
        calls,
        words,
        escapes,
        word,
        addrs,
        consts,
    }))
}

/// The switch of a call of type `fn_ty` (returning `ret`) through a word that may hold what
/// `reach` says: its cases, or (`Err`) the refused declaration the call becomes when nothing can
/// answer it.
///
/// The cases are the functions of the call's type the word may hold. A word that may be any
/// function's (a table's entry, a pointer from memory) may hold each function a table can hold
/// (the pipeline's [`Candidate`]s) and each function of the call's type whose address is taken; one
/// made only of constant addresses holds just those. A function of another type reaches the
/// default: calling it through this type is undefined in Metal.
fn site(
    (parts, stubs): (&mut ModuleParts, &mut Stubs),
    cx: &Cx<'_>,
    (fn_ty, ret): (TypeId, TypeId),
    reach: &Reach,
    refusals: &mut BTreeMap<(Option<usize>, TypeId), FuncId>,
    m: &mut Meter,
) -> Result<Result<Site, FuncId>, Error> {
    let mut arms: Vec<(u32, FuncId)> = Vec::new();
    for (k, c) in cx.candidates.iter().enumerate() {
        m.tick(1)?;
        let named = matches!(c.def, Ok(g) if reach.funcs.contains(&g));
        if !reach.any && !named {
            continue;
        }
        let refusal = match &c.def {
            Ok(g) => {
                let gty = parts
                    .funcs
                    .get(*g)
                    .map(|f| f.ty)
                    .ok_or_else(|| invariant("a candidate out of range"))?;
                if gty == fn_ty {
                    arms.push((c.word, *g));
                    continue;
                }
                let types = (&parts.types, &parts.types);
                if isomorphism(types, (gty, fn_ty), |_| None, m)?.is_none() {
                    // Another type: a table entry holding it reaches the default (undefined in
                    // Metal).
                    continue;
                }
                // The call's type up to struct identity: it would need its values converted.
                link::not_implemented(&c.name, "type")
            }
            // Not imported: a surviving call that may call it (its type is the call's, or
            // unknown because its module did not decode) is refused with the reason.
            Err(u) => {
                if let Some((l, ty)) = u.ty {
                    let from = cx
                        .linked
                        .get(l)
                        .and_then(|l| l.code.as_ref().ok())
                        .ok_or_else(|| invariant("a candidate's module out of range"))?;
                    let types = (&from.module.parts().types, &parts.types);
                    if isomorphism(types, (ty, fn_ty), |_| None, m)?.is_none() {
                        continue;
                    }
                }
                u.error.clone()
            }
        };
        let key = (Some(k), fn_ty);
        let decl = match refusals.get(&key) {
            Some(&d) => d,
            None => {
                let d = refused(parts, stubs, (&c.name, fn_ty), refusal, m)?;
                refusals.insert(key, d);
                d
            }
        };
        arms.push((c.word, decl));
    }
    // The functions no table holds: every one of the call's type whose address is taken, or
    // those whose addresses flow into the word. An unresolved stub's arm calls it, which the
    // post-check refuses as a direct call to it.
    let own = cx.words.own(fn_ty);
    if reach.any {
        m.charge(std::mem::size_of_val(own))?;
        arms.extend_from_slice(own);
    }
    for &f in &reach.funcs {
        m.tick(1)?;
        let Some(w) = cx.words.word(f) else {
            continue;
        };
        if arms.iter().any(|&(aw, _)| aw == w) {
            continue;
        }
        let ty = parts
            .funcs
            .get(f)
            .map(|f| f.ty)
            .ok_or_else(|| invariant("an addressed function out of range"))?;
        if ty == fn_ty {
            m.charge(std::mem::size_of::<(u32, FuncId)>())?;
            arms.push((w, f));
        }
    }
    // A table's entry takes this site with no arms too: an entry holds a function handle, and a
    // handle exists only for a function the pipeline links, so where nothing of the call's type is
    // linked every entry's word is null and the switch is unreachable by construction. What runs
    // is the shader's own guards on the table (`is_null_visible_function_table`, `size()`,
    // `f == nullptr`), which is what Metal runs for the same program.
    if !arms.is_empty() || !reach.any || reach.table {
        let zero = match parts.types.get(ret) {
            Some(Type::Void) => None,
            _ => {
                let zero = Constant {
                    ty: ret,
                    value: Const::Null,
                };
                Some(parts.consts.intern(zero, m)?)
            }
        };
        return Ok(Ok(Site::new(arms, cx, zero)));
    }
    // A word that may be any function's with no table behind it, and nothing of its type to call:
    // a pointer from memory or one made of an integer, holding no function of this stage.
    let key = (None, fn_ty);
    let decl = match refusals.get(&key) {
        Some(&d) => d,
        None => {
            let e = Unsupported::not_implemented(Feature::DynamicCall).into();
            let d = refused(parts, stubs, (b"dynamic call", fn_ty), e, m)?;
            refusals.insert(key, d);
            d
        }
    };
    Ok(Err(decl))
}

/// How many output instructions and blocks `plan` takes in place of one instruction of type
/// `ty` (a call's switch opens `blocks` new blocks after its own).
fn size_of(plan: Option<&Plan>, sites: &[Site], void: bool) -> Result<(u32, u32), Error> {
    Ok(match plan {
        None
        | Some(
            Plan::Retype
            | Plan::Cmp
            | Plan::Word(_)
            | Plan::Direct(_)
            | Plan::IdLoad
            | Plan::IdRetype
            | Plan::IdStore
            | Plan::FromInt(_)
            | Plan::Unfolded(_),
        ) => (1, 0),
        Some(Plan::Forward(_)) => (0, 0),
        // The 8 bytes' load and the word's truncation; the word's extension and the store.
        Some(Plan::WordLoad | Plan::WordStore) => (2, 0),
        // Through the parameter: the word's address and load. By the id: the word's read.
        Some(Plan::Size(TableRef::Param(_))) => (2, 0),
        Some(Plan::Size(TableRef::Id(_))) => (1, 0),
        Some(Plan::IsNull(_)) => (2, 0),
        Some(Plan::Lookup {
            table: TableRef::Param(_),
            ..
        }) => (4, 0),
        Some(Plan::Lookup {
            table: TableRef::Id(_),
            ..
        }) => (2, 0),
        Some(Plan::Site(s)) => {
            let n = u32::try_from(sites.get(*s).map_or(0, |s| s.arms.len()))
                .map_err(|_| invariant("cases"))?;
            // The switch, two per arm, the default's branch, and the continuation's phi.
            (1 + 2 * n + 1 + u32::from(!void), n + 2)
        }
    })
}

/// Where the output puts each instruction's result and each block.
struct Layout {
    /// Per instruction: its result's output id (`u32::MAX` for none).
    result: Vec<u32>,
    /// Per block: its first output block and the one holding its terminator.
    first: Vec<u32>,
    last: Vec<u32>,
}

fn layout(
    body: &Body,
    parts: &ModuleParts,
    a: &Analysis,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Layout, Error> {
    let mut result = filled(m, held, body.insts.len(), u32::MAX)?;
    let mut first = filled(m, held, body.blocks.len(), 0u32)?;
    let mut last = filled(m, held, body.blocks.len(), 0u32)?;
    let (mut ni, mut nb) = (0u32, 0u32);
    let bump = |x: u32, by: u32| x.checked_add(by).ok_or_else(|| invariant("ids"));
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        if let Some(f) = first.get_mut(b.index()) {
            *f = nb;
        }
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let void = matches!(parts.types.get(inst.ty), Some(Type::Void));
            if let Some(decls) = a.escapes.markers.get(&i) {
                // The refused calls before it.
                let n = u32::try_from(decls.len()).map_err(|_| invariant("markers"))?;
                ni = bump(ni, n)?;
            }
            let plan = a.plans.get(&i);
            let (insts, blocks) = size_of(plan, &a.sites, void)?;
            let at = match plan {
                Some(Plan::Forward(_)) => None,
                Some(Plan::Site(_)) => (!void).then(|| ni + insts - 1),
                Some(Plan::Size(_) | Plan::IsNull(_) | Plan::Lookup { .. } | Plan::WordLoad) => {
                    Some(ni + insts - 1)
                }
                _ => Some(ni),
            };
            if let (Some(at), Some(slot)) = (at, result.get_mut(i.index())) {
                *slot = at;
            }
            ni = bump(ni, insts)?;
            nb = bump(nb, blocks)?;
        }
        if let Some(l) = last.get_mut(b.index()) {
            *l = nb;
        }
        nb = bump(nb, 1)?;
    }
    Ok(Layout {
        result,
        first,
        last,
    })
}

/// The constants and types the rewrite uses.
struct Words {
    i32t: TypeId,
    i64t: TypeId,
    zero: ConstId,
    size_word: ConstId,
    first_entry: ConstId,
    undef: ConstId,
    poison: ConstId,
    /// The `i32` word numbers a table read by its id takes.
    size_word32: ConstId,
    first_entry32: ConstId,
    /// The `i64` resource ids of a null, `undef`, and `poison` table.
    zero64: ConstId,
    undef64: ConstId,
    poison64: ConstId,
}

impl Words {
    fn new(parts: &mut ModuleParts, m: &mut Meter) -> Result<Words, Error> {
        let i32t = parts.types.intern(Type::Int(32), m)?;
        let i64t = parts.types.intern(Type::Int(64), m)?;
        let mut c = |ty, value| parts.consts.intern(Constant { ty, value }, m);
        Ok(Words {
            i32t,
            i64t,
            zero: c(i32t, Const::Int(0))?,
            size_word: c(i64t, Const::Int(u64::from(SIZE_WORD)))?,
            first_entry: c(i64t, Const::Int(u64::from(FIRST_ENTRY_WORD)))?,
            undef: c(i32t, Const::Undef)?,
            poison: c(i32t, Const::Poison)?,
            size_word32: c(i32t, Const::Int(u64::from(SIZE_WORD)))?,
            first_entry32: c(i32t, Const::Int(u64::from(FIRST_ENTRY_WORD)))?,
            zero64: c(i64t, Const::Int(0))?,
            undef64: c(i64t, Const::Undef)?,
            poison64: c(i64t, Const::Poison)?,
        })
    }
}

struct Emit<'a> {
    body: &'a Body,
    a: &'a Analysis,
    lay: &'a Layout,
    words: Words,
    out: Body,
    open: Option<(InstId, Option<crate::ir::Sym>)>,
}

impl Emit<'_> {
    fn next_inst(&self) -> Result<u32, Error> {
        u32::try_from(self.out.insts.len()).map_err(|_| invariant("instruction ids"))
    }

    fn next_block(&self) -> Result<u32, Error> {
        u32::try_from(self.out.blocks.len()).map_err(|_| invariant("block ids"))
    }

    fn start(&mut self, name: Option<crate::ir::Sym>) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(invariant("a segment is already open"));
        }
        self.open = Some((InstId::new(self.next_inst()?), name));
        Ok(())
    }

    fn close(&mut self, m: &mut Meter) -> Result<(), Error> {
        let (start, name) = self
            .open
            .take()
            .ok_or_else(|| invariant("no open segment"))?;
        let end = InstId::new(self.next_inst()?);
        self.out.blocks.push(Block { start, end, name }, m)?;
        Ok(())
    }

    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<Value, Error> {
        let id = self.out.push_inst(Inst { ty, kind }, m)?;
        Ok(Value::Inst(id))
    }

    /// The output value of operand `v`.
    fn map(&self, v: Value) -> Result<Value, Error> {
        let mut v = v;
        for _ in 0..=self.body.insts.len() {
            match v {
                Value::Inst(i) => match self.a.plans.get(&i) {
                    Some(Plan::Forward(next)) => v = *next,
                    _ => {
                        let id = self
                            .lay
                            .result
                            .get(i.index())
                            .copied()
                            .filter(|&r| r != u32::MAX)
                            .ok_or_else(|| invariant("a use of a value without a result"))?;
                        return Ok(Value::Inst(InstId::new(id)));
                    }
                },
                Value::Const(c) => {
                    return Ok(Value::Const(self.a.consts.get(&c).copied().unwrap_or(c)));
                }
                Value::Arg(_) => return Ok(v),
            }
        }
        Err(invariant("forwarded values form a cycle"))
    }

    /// The output value of operand `v` of an instruction that takes no word: a word's operand
    /// (only an escape has one) is `poison` of the word's pointer type ([`Escapes`]).
    fn plain(&self, v: Value) -> Result<Value, Error> {
        match v {
            Value::Inst(i) if self.a.words.contains(&i) => {
                let ty = self
                    .body
                    .insts
                    .get(i)
                    .map(|w| w.ty)
                    .ok_or_else(|| invariant("a word out of range"))?;
                self.a
                    .escapes
                    .poison
                    .get(&ty)
                    .map(|&c| Value::Const(c))
                    .ok_or_else(|| invariant("a word used by no escape"))
            }
            other => self.map(other),
        }
    }

    /// The output word of function-pointer operand `v`: a word, or null, `undef`, `poison`, or a
    /// function's address (its word).
    fn word(&self, parts: &ModuleParts, v: Value) -> Result<Value, Error> {
        match v {
            Value::Const(c) => Ok(Value::Const(match parts.consts.get(c).map(|k| &k.value) {
                Some(Const::Null) => self.words.zero,
                Some(Const::Undef) => self.words.undef,
                Some(Const::Poison) => self.words.poison,
                Some(Const::FuncAddr(_)) => *self
                    .a
                    .addrs
                    .get(&c)
                    .ok_or_else(|| invariant("a function's address without its word"))?,
                _ => return Err(invariant("a function-pointer constant that is no word")),
            })),
            other => self.map(other),
        }
    }

    /// The output resource id of table operand `v`: an id-web value's, an `i64` an `inttoptr` made
    /// the table of, or the id of null (0), `undef`, or `poison`.
    fn id(&self, parts: &ModuleParts, v: Value) -> Result<Value, Error> {
        let Value::Const(c) = self.map(v)? else {
            return self.map(v);
        };
        let k = parts
            .consts
            .get(c)
            .ok_or_else(|| invariant("a constant out of range"))?;
        if parts.types.int_bits(k.ty) == Some(64) {
            return Ok(Value::Const(c));
        }
        Ok(Value::Const(match k.value {
            Const::Null => self.words.zero64,
            Const::Undef => self.words.undef64,
            Const::Poison => self.words.poison64,
            _ => return Err(invariant("a table constant that is no resource id")),
        }))
    }

    /// Word `k` of the table whose output resource id is `id` ([`FunctionTable::Word`]).
    fn table_word(&mut self, id: Value, k: Value, m: &mut Meter) -> Result<Value, Error> {
        let decl = self
            .a
            .word
            .ok_or_else(|| invariant("a table read by its id without its declaration"))?;
        let call = Call {
            callee: Callee::Air {
                decl,
                op: AirIntrinsic::FunctionTable(FunctionTable::Word),
            },
            args: Box::new([id, k]),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        };
        self.push(self.words.i32t, InstKind::Call(Box::new(call)), m)
    }

    fn first(&self, b: BlockId) -> Result<BlockId, Error> {
        self.lay
            .first
            .get(b.index())
            .map(|&x| BlockId::new(x))
            .ok_or_else(|| invariant("block out of range"))
    }

    fn last(&self, b: BlockId) -> Result<BlockId, Error> {
        self.lay
            .last
            .get(b.index())
            .map(|&x| BlockId::new(x))
            .ok_or_else(|| invariant("block out of range"))
    }

    /// `kind` with its operands mapped by `f` and its block operands to their output blocks.
    fn rewrite(
        &self,
        kind: &InstKind,
        f: &dyn Fn(&Self, Value) -> Result<Value, Error>,
    ) -> Result<InstKind, Error> {
        let mut kind = kind.clone();
        let mut failed = None;
        kind.for_each_value_mut(|v| match f(self, *v) {
            Ok(n) => *v = n,
            Err(e) => {
                failed.get_or_insert(e);
            }
        });
        if let Some(e) = failed {
            return Err(e);
        }
        match &mut kind {
            InstKind::Br { dest, .. } => *dest = self.first(*dest)?,
            InstKind::CondBr { t, f, .. } => {
                *t = self.first(*t)?;
                *f = self.first(*f)?;
            }
            InstKind::Switch { default, cases, .. } => {
                *default = self.first(*default)?;
                for (_, b) in cases.iter_mut() {
                    *b = self.first(*b)?;
                }
            }
            InstKind::Phi { incoming, .. } => {
                for (_, p) in incoming.iter_mut() {
                    *p = self.last(*p)?;
                }
            }
            _ => {}
        }
        Ok(kind)
    }
}

fn emit(
    body: &Body,
    parts: &mut ModuleParts,
    a: &Analysis,
    cx: &Cx<'_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Body, Error> {
    let lay = layout(body, parts, a, m, held)?;
    let words = Words::new(parts, m)?;
    let void = parts
        .types
        .lookup(&Type::Void)
        .ok_or_else(|| invariant("a body without a void type"))?;
    let align = Align::from_bytes(4);
    let parts: &ModuleParts = parts;
    let mut e = Emit {
        body,
        a,
        lay: &lay,
        words,
        out: Body::default(),
        open: None,
    };
    let table_ty = |table: Value| -> Result<TypeId, Error> {
        match table {
            Value::Arg(a) => cx
                .params
                .get(a.index())
                .copied()
                .ok_or_else(|| invariant("a table parameter out of range")),
            _ => Err(invariant("a table that is not a parameter")),
        }
    };
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        if e.first(b)?.raw() != e.next_block()? {
            return Err(invariant("devirtualized block offsets disagree"));
        }
        e.start(block.name)?;
        for i in block.inst_ids() {
            m.tick(1)?;
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let load = |ptr: Value| InstKind::Load {
                ptr,
                align,
                volatile: false,
                atomic: None,
                range: None,
            };
            for &decl in a.escapes.markers.get(&i).into_iter().flatten() {
                let marker = Call {
                    callee: Callee::Direct(decl),
                    args: Box::new([]),
                    fmf: FastMath::NONE,
                    tail: TailKind::None,
                    attrs: None,
                    range: None,
                };
                e.push(void, InstKind::Call(Box::new(marker)), m)?;
            }
            match a.plans.get(&i).copied() {
                None => {
                    let kind = e.rewrite(&inst.kind, &|e, v| e.plain(v))?;
                    e.push(inst.ty, kind, m)?;
                }
                Some(Plan::Forward(_)) => {}
                Some(Plan::Retype) => {
                    let kind = e.rewrite(&inst.kind, &|e, v| match (&inst.kind, v) {
                        // A `select`'s condition is no word.
                        (InstKind::Select { c, .. }, v) if v == *c => e.map(v),
                        _ => e.word(parts, v),
                    })?;
                    e.push(e.words.i32t, kind, m)?;
                }
                Some(Plan::Cmp) => {
                    let kind = e.rewrite(&inst.kind, &|e, v| e.word(parts, v))?;
                    e.push(inst.ty, kind, m)?;
                }
                Some(Plan::Word(op)) => {
                    let InstKind::Cast { v, .. } = inst.kind else {
                        return Err(invariant("a pointer word that is no cast"));
                    };
                    let v = e.map(v)?;
                    e.push(inst.ty, InstKind::Cast { op, v }, m)?;
                }
                Some(Plan::Size(TableRef::Id(table))) => {
                    let id = e.id(parts, table)?;
                    e.table_word(id, Value::Const(e.words.size_word32), m)?;
                }
                Some(Plan::IsNull(TableRef::Id(table))) => {
                    let id = e.id(parts, table)?;
                    let w = e.table_word(id, Value::Const(e.words.zero), m)?;
                    let cmp = InstKind::ICmp {
                        pred: IntPredicate::Eq,
                        a: w,
                        b: Value::Const(e.words.zero),
                    };
                    e.push(inst.ty, cmp, m)?;
                }
                Some(Plan::Lookup {
                    table: TableRef::Id(table),
                    index,
                }) => {
                    let id = e.id(parts, table)?;
                    let k = InstKind::BinOp {
                        op: BinOp::Add,
                        flags: IntFlags::NONE,
                        fmf: FastMath::NONE,
                        a: e.map(index)?,
                        b: Value::Const(e.words.first_entry32),
                    };
                    let k = e.push(e.words.i32t, k, m)?;
                    e.table_word(id, k, m)?;
                }
                Some(Plan::IdLoad) => {
                    let InstKind::Load { ptr, align, .. } = inst.kind else {
                        return Err(invariant("a table load that is no load"));
                    };
                    let kind = InstKind::Load {
                        ptr: e.plain(ptr)?,
                        align,
                        volatile: false,
                        atomic: None,
                        range: None,
                    };
                    e.push(e.words.i64t, kind, m)?;
                }
                Some(Plan::IdRetype) => {
                    let kind = e.rewrite(&inst.kind, &|e, v| match (&inst.kind, v) {
                        // A `select`'s condition is no id.
                        (InstKind::Select { c, .. }, v) if v == *c => e.map(v),
                        _ => e.id(parts, v),
                    })?;
                    e.push(e.words.i64t, kind, m)?;
                }
                Some(Plan::WordLoad) => {
                    let InstKind::Load { ptr, align, .. } = inst.kind else {
                        return Err(invariant("a function pointer's load that is no load"));
                    };
                    let kind = InstKind::Load {
                        ptr: e.plain(ptr)?,
                        align,
                        volatile: false,
                        atomic: None,
                        range: None,
                    };
                    let wide = e.push(e.words.i64t, kind, m)?;
                    let cast = InstKind::Cast {
                        op: CastOp::Trunc,
                        v: wide,
                    };
                    e.push(e.words.i32t, cast, m)?;
                }
                Some(Plan::FromInt(op)) => {
                    let InstKind::Cast { v, .. } = inst.kind else {
                        return Err(invariant("an integer's pointer that is no cast"));
                    };
                    let v = e.map(v)?;
                    e.push(e.words.i32t, InstKind::Cast { op, v }, m)?;
                }
                Some(Plan::WordStore) => {
                    let InstKind::Store {
                        val,
                        ptr,
                        align,
                        volatile,
                        atomic,
                    } = inst.kind
                    else {
                        return Err(invariant("a function pointer's store that is no store"));
                    };
                    let cast = InstKind::Cast {
                        op: CastOp::ZExt,
                        v: e.word(parts, val)?,
                    };
                    let wide = e.push(e.words.i64t, cast, m)?;
                    let kind = InstKind::Store {
                        val: wide,
                        ptr: e.plain(ptr)?,
                        align,
                        volatile,
                        atomic,
                    };
                    e.push(inst.ty, kind, m)?;
                }
                Some(Plan::IdStore) => {
                    let InstKind::Store {
                        val,
                        ptr,
                        align,
                        volatile,
                        atomic,
                    } = inst.kind
                    else {
                        return Err(invariant("a table store that is no store"));
                    };
                    let kind = InstKind::Store {
                        val: e.id(parts, val)?,
                        ptr: e.plain(ptr)?,
                        align,
                        volatile,
                        atomic,
                    };
                    e.push(inst.ty, kind, m)?;
                }
                Some(Plan::Size(TableRef::Param(table))) => {
                    let ptr = table_ty(table)?;
                    let gep = InstKind::Gep {
                        src_ty: e.words.i32t,
                        base: e.map(table)?,
                        indices: Box::new([Value::Const(e.words.size_word)]),
                        inbounds: true,
                    };
                    let at = e.push(ptr, gep, m)?;
                    e.push(inst.ty, load(at), m)?;
                }
                Some(Plan::IsNull(TableRef::Param(table))) => {
                    let w = e.push(e.words.i32t, load(e.map(table)?), m)?;
                    let cmp = InstKind::ICmp {
                        pred: IntPredicate::Eq,
                        a: w,
                        b: Value::Const(e.words.zero),
                    };
                    e.push(inst.ty, cmp, m)?;
                }
                Some(Plan::Lookup {
                    table: TableRef::Param(table),
                    index,
                }) => {
                    let ptr = table_ty(table)?;
                    let wide = InstKind::Cast {
                        op: CastOp::ZExt,
                        v: e.map(index)?,
                    };
                    let wide = e.push(e.words.i64t, wide, m)?;
                    let word = InstKind::BinOp {
                        op: BinOp::Add,
                        flags: IntFlags::NONE,
                        fmf: FastMath::NONE,
                        a: wide,
                        b: Value::Const(e.words.first_entry),
                    };
                    let word = e.push(e.words.i64t, word, m)?;
                    let gep = InstKind::Gep {
                        src_ty: e.words.i32t,
                        base: e.map(table)?,
                        indices: Box::new([word]),
                        inbounds: true,
                    };
                    let at = e.push(ptr, gep, m)?;
                    e.push(e.words.i32t, load(at), m)?;
                }
                Some(Plan::Direct(f)) => {
                    let InstKind::Call(call) = &inst.kind else {
                        return Err(invariant("a devirtualized call that is no call"));
                    };
                    let mut call = call.clone();
                    call.callee = Callee::Direct(f);
                    let kind = e.rewrite(&InstKind::Call(call), &|e, v| e.plain(v))?;
                    e.push(inst.ty, kind, m)?;
                }
                Some(Plan::Unfolded(decl)) => {
                    let InstKind::Call(call) = &inst.kind else {
                        return Err(invariant("a settled call that is no call"));
                    };
                    let Callee::Indirect { callee, .. } = call.callee else {
                        return Err(invariant("a settled call that is not indirect"));
                    };
                    let kind = Call {
                        callee: Callee::Direct(decl),
                        args: Box::new([e.map(callee)?]),
                        fmf: FastMath::NONE,
                        tail: TailKind::None,
                        attrs: None,
                        range: None,
                    };
                    e.push(inst.ty, InstKind::Call(Box::new(kind)), m)?;
                }
                Some(Plan::Site(s)) => {
                    let InstKind::Call(call) = &inst.kind else {
                        return Err(invariant("a table call that is no call"));
                    };
                    let Callee::Indirect { callee, .. } = call.callee else {
                        return Err(invariant("a table call that is not indirect"));
                    };
                    let site = a
                        .sites
                        .get(s)
                        .ok_or_else(|| invariant("a site out of range"))?;
                    let n = u32::try_from(site.arms.len()).map_err(|_| invariant("cases"))?;
                    let arm0 = e.next_block()? + 1;
                    let default = BlockId::new(arm0 + n);
                    let cont = BlockId::new(arm0 + n + 1);
                    let to = |arm: usize| {
                        u32::try_from(arm)
                            .map(|a| BlockId::new(arm0 + a))
                            .map_err(|_| invariant("cases"))
                    };
                    let cases: Box<[(u64, BlockId)]> = site
                        .cases
                        .iter()
                        .map(|&(w, arm)| Ok((u64::from(w), to(arm)?)))
                        .collect::<Result<_, Error>>()?;
                    let word = e.map(callee)?;
                    let switch = InstKind::Switch {
                        v: word,
                        default,
                        cases,
                        hints: None,
                    };
                    e.push(void, switch, m)?;
                    e.close(m)?;
                    let returns = inst.ty != void;
                    // The join's PHI, when the call returns: its incoming list is its operand
                    // list, charged once, by `push_inst` as the PHI is pushed. A `void` call has
                    // none.
                    let mut incoming =
                        Vec::with_capacity(if returns { site.arms.len() + 1 } else { 0 });
                    for (a, arm) in site.arms.iter().zip(arm0..) {
                        e.start(None)?;
                        let mut direct: Box<Call> = call.clone();
                        direct.callee = Callee::Direct(a.callee);
                        let mut kind = e.rewrite(&InstKind::Call(direct), &|e, v| e.plain(v))?;
                        // A group's marker takes the word before the call's arguments.
                        if let (true, InstKind::Call(c)) = (a.grouped, &mut kind) {
                            let mut args = Vec::with_capacity(c.args.len() + 1);
                            args.push(word);
                            args.extend(c.args.iter().copied());
                            c.args = args.into_boxed_slice();
                        }
                        let r = e.push(inst.ty, kind, m)?;
                        e.push(
                            void,
                            InstKind::Br {
                                dest: cont,
                                hints: None,
                            },
                            m,
                        )?;
                        e.close(m)?;
                        if returns {
                            incoming.push((r, BlockId::new(arm)));
                        }
                    }
                    e.start(None)?;
                    e.push(
                        void,
                        InstKind::Br {
                            dest: cont,
                            hints: None,
                        },
                        m,
                    )?;
                    e.close(m)?;
                    e.start(None)?;
                    if returns {
                        // The default calls nothing: its result is zero.
                        let zero = site
                            .zero
                            .ok_or_else(|| invariant("a call without a zero"))?;
                        incoming.push((Value::Const(zero), default));
                        let phi = InstKind::Phi {
                            incoming: incoming.into_boxed_slice(),
                            fmf: FastMath::NONE,
                        };
                        e.push(inst.ty, phi, m)?;
                    }
                }
            }
        }
        e.close(m)?;
        if e.last(b)?.raw() + 1 != e.next_block()? {
            return Err(invariant("devirtualized block offsets disagree"));
        }
    }
    for (i, &r) in lay.result.iter().enumerate() {
        if r != u32::MAX && r as usize >= e.out.insts.len() {
            return Err(invariant(&format!("result {i} past the body")));
        }
    }
    Ok(e.out)
}
