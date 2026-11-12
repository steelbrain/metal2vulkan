//! Static-initializer evaluation.
//!
//! Metal runs a module's `air.static_init` functions once per pipeline, before any entry, so
//! their results are pipeline constants: predicate globals that gate arguments, integer globals
//! that `air.location_index` names, and copies of function constants into internal globals. This
//! pass interprets them over constants and folds their stores into the stored globals'
//! initializers, which then become constant. It runs after function-constant binding: a bound
//! constant's global already holds its value. Initializers run in `llvm.global_ctors` order; a
//! module linked from several translation units, each with its own initializer, may carry no
//! list. Its order is then unstated, so its initializers are accepted only when they commute: no
//! global one writes is touched by another.
//!
//! The interpreter executes, it does not analyze: each function runs from its entry block along
//! the one path the constants decide, a callee runs in a new frame, and every value is a
//! constant, a pointer into a global (`global + byte offset`), or **undefined from an unsupplied
//! function constant** (the SCCP lattice's `Fc`). What it evaluates: integer arithmetic,
//! comparisons, and casts ([`Folder`]), `fcmp` on constants with no subnormal operand, `select`,
//! element and member access,
//! shuffles, `air.convert` of an integer or `bool`, a scalar or each lane of a vector, to another
//! integer (as C++ converts it) or to `bool` (`x != 0`), and of a scalar one to a float (rounded
//! to nearest even, as `sitofp` and `uitofp` are), unless it names `.sat` or a rounding to a float
//! other than `rte` (a compute pipeline of OpenCL mode honors those), scalar `fadd`, `fsub`, `fmul`,
//! `fdiv`, and `fneg` where every IEEE 754 evaluation agrees (finite normal operands, and a result
//! that is an exact zero or normal: [`Folder::static_float_binop`]), GEPs with constant indices,
//! non-volatile loads and stores of globals, `llvm.{s,u}{max,min}` and `llvm.ctpop`, and the two
//! function-constant intrinsics (`air.normalize_function_constant_predicate` maps an undefined
//! operand to 0). Anything else (a loop, a load of an externally initialized global, a store
//! outside the constant address space, an overlapping access of another type, an alloca, other
//! float arithmetic, a call to a declaration) is `StaticInitNotEvaluable`. A branch on an
//! unsupplied constant takes its false way, a `select` on one is its false arm, and an `and` with
//! zero, an `or` with all ones and a `mul` by zero are that constant whatever the other operand
//! holds, as SCCP's are; a branch on `undef` is `UndefCondition`, and a call cycle is
//! `Unsupported{Recursion}`. A scalar integer division or remainder by zero, whose result the IR
//! leaves undefined, is undefined the same way ([`Undefined::Ir`]), naming no constant. Metal makes
//! the pipeline and drops its kernel's stores of the quotient and of a value computed from it,
//! and every store of a kernel whose loop a quotient computed from it bounds (an Apple M3 Max
//! under macOS 27.0.1, `static_init_division.m`).
//!
//! [`super::static_values`] evaluates under [`Unevaluable::Opaque`]: a pure operation the
//! interpreter does not evaluate (float arithmetic outside the agreement, a conversion from a
//! float or of a vector to floats) yields an opaque value instead of refusing, which taints what it flows into
//! like an undefined value and leaves the globals it reaches unknown. Everything else still
//! refuses, as does a branch on, an address from, or a store through an opaque value.
//!
//! Memory is modeled per global as its **leaves** (scalars, pointers, and vectors at byte
//! offsets, flattened from the value type), materialized from the initializer when the
//! initializer first touches the global. An access must name exactly one leaf of its type, or
//! exactly the leaves of an aggregate type, or read or write bytes of known value: a load or
//! store of an integer, a float, or a vector of either whose lanes fill whole bytes, every byte
//! of which lies in such a leaf holding a constant (a wide load over a table's fields, a narrow
//! one inside a field, or a copy of a `float`, `half` or `short2` through an integer of its size),
//! in the module's little-endian byte order. A global with a leaf left undefined by an unsupplied
//! constant gets an `undef` initializer and is reported as **derived** from that constant (the
//! lowest index), so that SCCP taints every read of it.

use std::collections::{BTreeMap, HashMap};

use crate::budget::Meter;
use crate::error::{CanonError, DiagName, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature, Unsupported};
use crate::intrinsics::{
    AirIntrinsic, Convert, FunctionConstant, LlvmIntrinsic, NumKind, Rounding,
};
use crate::ir::layout::Endian;
use crate::ir::ConstId;
use crate::ir::{AddrSpace, Arena, BinOp, BlockId, Body, Callee, CastOp, Const, ConstExpr};
use crate::ir::{Constant, FuncId, Function, Global, GlobalId, Idx, InstId, InstKind};
use crate::ir::{Linkage, ModuleParts, Section, Type, TypeId, TypeLayouts, TypeTable, Value};

use super::fold::Folder;
use super::rewrite::invariant;
use super::sccp::{FcState, Undefined};

/// The most leaves one global may flatten into. A larger global touched by a static initializer
/// is refused as not evaluable (a precision limit, never a semantic one); the ledger bounds the
/// total.
const LEAF_LIMIT: u64 = 1 << 16;

/// Globals whose value is now undefined, with what leaves it so: an unsupplied function constant
/// (the lowest index), or an operation the IR leaves undefined.
pub(super) type Derived = Vec<(GlobalId, Undefined)>;

/// The new initializer of each written global.
type Inits = Vec<(GlobalId, ConstId)>;

/// An interpreted value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SVal {
    C(ConstId),
    /// Undefined: computed from the unsupplied constant with this index, or by an operation the
    /// IR leaves undefined.
    Fc(Undefined),
    /// The address `global + offset` (bytes).
    Ptr(GlobalId, u64),
    /// Not evaluated: computed by a pure operation the interpreter does not evaluate (float
    /// arithmetic outside the agreement). Only [`Unevaluable::Opaque`] evaluation produces it.
    Opaque,
}

/// What evaluation does with a pure operation it does not evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unevaluable {
    /// Refuse the module (`StaticInitNotEvaluable`): specialization bakes every stored global.
    Refuse,
    /// Carry the result as an opaque value. A global it reaches is reported opaque, and globals
    /// computed without it still resolve. A branch on it, an address computed from it, or a
    /// store through it still refuses: the path, or what memory it changes, is then unknown.
    /// Only [`super::static_values`] (validation's case checker) evaluates this way.
    Opaque,
}

/// The outcome of evaluating the static initializers.
pub(super) struct Evaluated {
    /// Globals whose value is now undefined, with what leaves it so.
    pub derived: Derived,
    /// Globals holding an opaque value: stored from an operation the interpreter does not
    /// evaluate. Their initializers are left as they were. Always empty under
    /// [`Unevaluable::Refuse`].
    pub opaque: Vec<GlobalId>,
}

/// Whether conversion `c` of an integer names a suffix that a compute pipeline of OpenCL mode
/// honors and no other pipeline does: `.sat`, or a rounding to a float other than to nearest
/// even. Its answer depends on the pipeline, so it is not evaluated here.
fn honored_in_opencl_mode(c: &Convert) -> bool {
    let directed = matches!(
        c.rounding,
        Some(Rounding::Zero | Rounding::Up | Rounding::Down)
    );
    c.saturate || (c.dst_kind == NumKind::Float && directed)
}

fn not_evaluable(at: Location) -> Error {
    InputError::new(
        InputErrorKind::Canon(CanonError::StaticInitNotEvaluable),
        at,
    )
    .into()
}

/// Evaluate the static initializers of `parts` and fold their stores into initializers.
/// `fcs` states every declared function constant. Returns the globals derived from an unsupplied
/// constant, and those left opaque (only under [`Unevaluable::Opaque`]).
pub(super) fn evaluate(
    parts: &mut ModuleParts,
    fcs: &HashMap<GlobalId, FcState>,
    unevaluable: Unevaluable,
    m: &mut Meter,
) -> Result<Evaluated, Error> {
    let (order, listed) = initializers(parts)?;
    if order.is_empty() {
        return Ok(Evaluated {
            derived: Vec::new(),
            opaque: Vec::new(),
        });
    }
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let charged = parts.types.len() * std::mem::size_of::<Option<crate::ir::Layout>>();
    let mut ev = Eval {
        types: &parts.types,
        layouts: &layouts,
        folder: Folder {
            types: &parts.types,
            consts: &mut parts.consts,
            held: 0,
        },
        globals: &parts.globals,
        funcs: &parts.funcs,
        symbols: &parts.symbols,
        fcs,
        unevaluable,
        little: parts.data_layout.endian == Endian::Little,
        mem: BTreeMap::new(),
        current: 0,
        held: charged,
    };
    let result = order
        .iter()
        .enumerate()
        .try_for_each(|(i, &f)| {
            ev.current = i;
            ev.run(f, m)
        })
        .and_then(|()| {
            if listed || ev.commute() {
                ev.fold(m)
            } else {
                Err(not_evaluable(Location::module()))
            }
        });
    let held = ev.held;
    m.release(held);
    let (inits, evaluated) = result?;
    for (g, init) in inits {
        let global = parts
            .globals
            .get_mut(g)
            .ok_or_else(|| invariant("global out of range"))?;
        global.init = Some(init);
        global.constant = true;
        global.externally_initialized = false;
    }
    Ok(evaluated)
}

/// The static initializers in the order Metal runs them, and whether that order is stated:
/// `llvm.global_ctors` order (ascending priority, stable), which must list exactly the
/// `air.static_init` functions. A lone initializer needs no list. Several initializers without a
/// list come in definition order, unstated: evaluation then accepts them only when no order could
/// change what they compute ([`Eval::commute`]).
fn initializers(parts: &ModuleParts) -> Result<(Vec<FuncId>, bool), Error> {
    let section: Vec<FuncId> = parts
        .funcs
        .iter()
        .filter(|(_, f)| f.section == Some(Section::StaticInit) && f.body.is_some())
        .map(|(id, _)| id)
        .collect();
    if section.is_empty() {
        return Ok((section, true));
    }
    let ctors = parts.globals.values().find(|g| {
        g.linkage == Linkage::Appending
            && parts.symbols.get(g.name) == Some(b"llvm.global_ctors".as_slice())
    });
    let Some(ctors) = ctors else {
        let lone = section.len() == 1;
        return Ok((section, lone));
    };
    let value = |c: ConstId| parts.consts.get(c).map(|k| &k.value);
    let Some(Const::Aggregate(entries)) = ctors.init.and_then(value) else {
        return Err(not_evaluable(Location::module()));
    };
    let mut listed = Vec::with_capacity(entries.len());
    for &e in entries.iter() {
        let Some(Const::Aggregate(fields)) = value(e) else {
            return Err(not_evaluable(Location::module()));
        };
        let (Some(&prio), Some(&func)) = (fields.first(), fields.get(1)) else {
            return Err(not_evaluable(Location::module()));
        };
        let prio = match value(prio) {
            Some(Const::Int(p)) => *p,
            _ => return Err(not_evaluable(Location::module())),
        };
        let mut func = func;
        let f = loop {
            match value(func) {
                Some(Const::FuncAddr(f)) => break *f,
                Some(Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ops: [v],
                })) => func = *v,
                _ => return Err(not_evaluable(Location::module())),
            }
        };
        listed.push((prio, f));
    }
    listed.sort_by_key(|&(p, _)| p);
    let mut order: Vec<FuncId> = listed.into_iter().map(|(_, f)| f).collect();
    let mut sorted = order.clone();
    sorted.sort();
    sorted.dedup();
    if sorted != section || sorted.len() != order.len() {
        return Err(not_evaluable(Location::module()));
    }
    order.shrink_to_fit();
    Ok((order, true))
}

/// One global's leaves: offset to (leaf type, value).
type Cells = BTreeMap<u64, (TypeId, SVal)>;

/// A global the initializers touched.
struct Touched {
    cells: Cells,
    /// Whether an initializer stored to it.
    written: bool,
    /// The run-order index of the first initializer that touched it.
    first: usize,
    /// Whether an initializer other than `first` touched it too.
    shared: bool,
}

struct Eval<'a> {
    types: &'a TypeTable,
    layouts: &'a TypeLayouts,
    folder: Folder<'a>,
    globals: &'a Arena<GlobalId, Global>,
    funcs: &'a Arena<FuncId, Function>,
    symbols: &'a crate::ir::Symbols,
    fcs: &'a HashMap<GlobalId, FcState>,
    unevaluable: Unevaluable,
    /// Whether the module's data layout is little-endian (`e`), the byte order an access of
    /// another type than its leaves reads and writes them in (`read_bytes`, `write_bytes`).
    little: bool,
    /// Every global the initializers touched (read or written).
    mem: BTreeMap<GlobalId, Touched>,
    /// The run-order index of the initializer running.
    current: usize,
    /// Ledger bytes to return when evaluation ends.
    held: usize,
}

/// What executing one instruction did.
enum Step {
    /// Defined the instruction's value (`None` for a store) and moves to the next instruction.
    Next(Option<SVal>),
    /// Moved to another block.
    Jumped,
    /// Calls a defined function with these arguments.
    Call(FuncId, Vec<SVal>, Location),
    /// Returns from the frame.
    Return(Option<SVal>),
}

fn frame_bytes(fr: &Frame<'_>) -> usize {
    fr.vals.len() * std::mem::size_of::<Option<SVal>>() + fr.visited.len()
}

/// One activation of an interpreted function.
struct Frame<'b> {
    func: FuncId,
    body: &'b Body,
    args: Vec<SVal>,
    vals: Vec<Option<SVal>>,
    visited: Vec<bool>,
    block: BlockId,
    prev: Option<BlockId>,
    /// The next instruction to execute.
    at: InstId,
    /// The call in the caller that receives this frame's return value.
    ret_to: Option<InstId>,
}

impl<'a> Eval<'a> {
    fn charge(&mut self, m: &mut Meter, bytes: usize) -> Result<(), Error> {
        m.charge(bytes)?;
        self.held = self.held.saturating_add(bytes);
        Ok(())
    }

    /// The value of a pure operation the interpreter does not evaluate at `at`.
    fn unevaluated(&self, at: &Location) -> Result<SVal, Error> {
        match self.unevaluable {
            Unevaluable::Refuse => Err(not_evaluable(at.clone())),
            Unevaluable::Opaque => Ok(SVal::Opaque),
        }
    }

    fn name(&self, f: FuncId) -> DiagName {
        let bytes = self
            .funcs
            .get(f)
            .and_then(|f| self.symbols.get(f.name))
            .unwrap_or(b"?");
        DiagName::new(String::from_utf8_lossy(bytes))
    }

    fn frame<'b>(
        &mut self,
        func: FuncId,
        args: Vec<SVal>,
        ret_to: Option<InstId>,
        m: &mut Meter,
    ) -> Result<Frame<'b>, Error>
    where
        'a: 'b,
    {
        let f = self
            .funcs
            .get(func)
            .ok_or_else(|| invariant("function out of range"))?;
        let Some(body) = f.body.as_ref() else {
            return Err(not_evaluable(Location::function(
                self.name(func),
                None,
                None,
            )));
        };
        let n = body.insts.len();
        let bytes = n * std::mem::size_of::<Option<SVal>>() + body.blocks.len();
        self.charge(m, bytes)?;
        let start = body
            .blocks
            .get(BlockId::new(0))
            .ok_or_else(|| invariant("a body without blocks"))?
            .start;
        let mut visited = vec![false; body.blocks.len()];
        if let Some(v) = visited.first_mut() {
            *v = true;
        }
        Ok(Frame {
            func,
            body,
            args,
            vals: vec![None; n],
            visited,
            block: BlockId::new(0),
            prev: None,
            at: start,
            ret_to,
        })
    }

    /// Run the static initializer `f` to completion.
    fn run(&mut self, f: FuncId, m: &mut Meter) -> Result<(), Error> {
        let first = self.frame(f, Vec::new(), None, m)?;
        let mut stack = vec![first];
        loop {
            m.tick(1)?;
            let Some(top) = stack.last_mut() else {
                return Ok(());
            };
            match self.step(top, m)? {
                Step::Next(v) => {
                    if let Some(slot) = top.vals.get_mut(top.at.index()) {
                        *slot = v;
                    }
                    top.at = InstId::new(top.at.raw() + 1);
                }
                Step::Jumped => {}
                Step::Call(g, args, at) => {
                    let call = top.at;
                    if stack.iter().any(|fr| fr.func == g) {
                        return Err(Unsupported::not_implemented(Feature::Recursion)
                            .at(at)
                            .into());
                    }
                    let callee = self.frame(g, args, Some(call), m)?;
                    stack.push(callee);
                }
                Step::Return(r) => {
                    let ret_to = top.ret_to;
                    if let Some(done) = stack.pop() {
                        let bytes = frame_bytes(&done);
                        m.release(bytes);
                        self.held = self.held.saturating_sub(bytes);
                    }
                    if let (Some(caller), Some(call)) = (stack.last_mut(), ret_to) {
                        if let Some(slot) = caller.vals.get_mut(call.index()) {
                            *slot = r;
                        }
                        caller.at = InstId::new(call.raw() + 1);
                    }
                }
            }
        }
    }

    /// Execute the instruction `fr` is at.
    fn step(&mut self, fr: &mut Frame<'_>, m: &mut Meter) -> Result<Step, Error> {
        let (func, i, body) = (fr.func, fr.at, fr.body);
        let at = Location::function(self.name(func), Some(fr.block.raw()), Some(i.raw()));
        let inst = body
            .insts
            .get(i)
            .ok_or_else(|| invariant("instruction out of range"))?;
        let ty = inst.ty;
        let args = &fr.args;
        let vals = &fr.vals;
        let val = |v: Value| -> Result<SVal, Error> {
            match v {
                Value::Const(c) => Ok(SVal::C(c)),
                Value::Arg(a) => args
                    .get(a.index())
                    .copied()
                    .ok_or_else(|| invariant("argument out of range")),
                Value::Inst(d) => vals
                    .get(d.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| invariant("a use before its definition")),
            }
        };
        let v = match &inst.kind {
            InstKind::Br { dest, .. } => return self.jump(fr, *dest, &at),
            InstKind::CondBr { c, t, f, .. } => {
                let dest = match self.decide(val(*c)?, &at)? {
                    0 => *f,
                    _ => *t,
                };
                return self.jump(fr, dest, &at);
            }
            InstKind::Switch {
                v, default, cases, ..
            } => {
                let x = self.decide(val(*v)?, &at)?;
                let dest = cases
                    .iter()
                    .find(|&&(k, _)| k == x)
                    .map_or(*default, |&(_, d)| d);
                return self.jump(fr, dest, &at);
            }
            InstKind::Ret(v) => return Ok(Step::Return(v.map(val).transpose()?)),
            InstKind::Phi { incoming, .. } => {
                let prev = fr
                    .prev
                    .ok_or_else(|| invariant("a PHI in an entry block"))?;
                let v = incoming
                    .iter()
                    .find(|&&(_, p)| p == prev)
                    .map(|&(v, _)| v)
                    .ok_or_else(|| invariant("a PHI without its edge"))?;
                val(v)?
            }
            InstKind::Call(call) => match call.callee {
                Callee::Direct(g) => {
                    let args = call
                        .args
                        .iter()
                        .map(|&a| val(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(Step::Call(g, args, at));
                }
                Callee::Air {
                    op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
                    ..
                } => {
                    let g = match call.args.first().map(|&a| val(a)).transpose()? {
                        Some(SVal::C(c)) => self.ptr_of(c, &at)?,
                        Some(SVal::Ptr(g, 0)) => Some((g, 0)),
                        _ => None,
                    };
                    let state = g.and_then(|(g, _)| self.fcs.get(&g)).copied();
                    let Some(state) = state.filter(|s| s.decl) else {
                        return Err(not_evaluable(at));
                    };
                    SVal::C(self.int(ty, u64::from(state.supplied), m)?)
                }
                Callee::Air {
                    op: AirIntrinsic::FunctionConstant(FunctionConstant::NormalizePredicate(_)),
                    ..
                } => {
                    let x = match call.args.first().map(|&a| val(a)).transpose()? {
                        Some(SVal::C(c)) => Some(c),
                        Some(SVal::Fc(_)) => None,
                        Some(SVal::Opaque) => return Ok(Step::Next(Some(SVal::Opaque))),
                        _ => return Err(not_evaluable(at)),
                    };
                    let c = self.folder.normalize_predicate(ty, x, m)?;
                    SVal::C(c.ok_or_else(|| not_evaluable(at.clone()))?)
                }
                Callee::Llvm {
                    op: LlvmIntrinsic::Int { op, .. } | LlvmIntrinsic::NarrowInt { op, .. },
                    ..
                } => match &call.args[..] {
                    [a, b] => {
                        let (a, b) = (val(*a)?, val(*b)?);
                        self.pure2(a, b, &at, m, |f, x, y, m| f.int_minmax(op, ty, x, y, m))?
                    }
                    [x] => {
                        let x = val(*x)?;
                        self.pure2(x, x, &at, m, |f, x, _, m| f.int_unary(op, ty, x, m))?
                    }
                    _ => return Err(not_evaluable(at)),
                },
                // An integer or `bool` source, a scalar or each lane of a vector: to `bool` as
                // `x != 0`, to another integer as C++ converts it; a scalar to a float rounded to
                // nearest even.
                Callee::Air {
                    op: AirIntrinsic::Convert(c),
                    ..
                } if c.src_kind != NumKind::Float
                    && c.src_fp8.is_none()
                    && (c.dst_kind != NumKind::Float || !c.src.is_vector())
                    && !honored_in_opencl_mode(&c) =>
                {
                    let signed = c.src_kind == NumKind::Signed;
                    let x = match call.args.first() {
                        Some(&x) => val(x)?,
                        None => return Err(not_evaluable(at)),
                    };
                    self.pure2(x, x, &at, m, |f, x, _, m| f.convert_int(x, signed, ty, m))?
                }
                // Any other conversion is a pure function of its operand.
                Callee::Air {
                    op: AirIntrinsic::Convert(_),
                    ..
                } => self.unevaluated(&at)?,
                _ => return Err(not_evaluable(at)),
            },
            InstKind::BinOp { op, a, b, .. } if op.is_float() => {
                let (op, a, b) = (*op, val(*a)?, val(*b)?);
                self.pure2(a, b, &at, m, |f, x, y, m| {
                    f.static_float_binop(op, ty, x, y, m)
                })?
            }
            InstKind::BinOp { op, a, b, .. } => {
                let (op, a, b) = (*op, val(*a)?, val(*b)?);
                match (a, b) {
                    (SVal::Fc(_), SVal::C(k)) | (SVal::C(k), SVal::Fc(_))
                        if self.folder.absorbs(op, k) =>
                    {
                        SVal::C(k)
                    }
                    (SVal::C(_), SVal::C(d))
                        if matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem)
                            && self.folder.int(d) == Some(0) =>
                    {
                        SVal::Fc(Undefined::Ir)
                    }
                    _ => self.pure2(a, b, &at, m, |f, x, y, m| f.binop(op, ty, x, y, m))?,
                }
            }
            InstKind::FNeg { v, .. } => {
                let x = val(*v)?;
                self.pure2(x, x, &at, m, |f, x, _, m| f.static_float_neg(ty, x, m))?
            }
            InstKind::ICmp { pred, a, b } => {
                let (p, a, b) = (*pred, val(*a)?, val(*b)?);
                self.pure2(a, b, &at, m, |f, x, y, m| f.icmp(p, ty, x, y, m))?
            }
            InstKind::FCmp { pred, a, b, .. } => {
                let (p, a, b) = (*pred, val(*a)?, val(*b)?);
                self.pure2(a, b, &at, m, |f, x, y, m| f.fcmp(p, ty, x, y, m))?
            }
            InstKind::Cast {
                op: op @ (CastOp::SIToFP | CastOp::UIToFP),
                v,
            } => {
                let signed = *op == CastOp::SIToFP;
                let x = val(*v)?;
                self.pure2(x, x, &at, m, |f, x, _, m| f.int_to_float(x, signed, ty, m))?
            }
            InstKind::Cast { op, v } => {
                let (op, v) = (*op, val(*v)?);
                let reinterpret = matches!(op, CastOp::Bitcast | CastOp::AddrSpaceCast);
                match v {
                    SVal::Ptr(..) if reinterpret => v,
                    SVal::Fc(i) => SVal::Fc(i),
                    SVal::Opaque => SVal::Opaque,
                    SVal::C(c) => match self.ptr_of(c, &at)? {
                        Some((g, o)) if reinterpret => SVal::Ptr(g, o),
                        _ => match self.folder.cast(op, c, ty, m)? {
                            Some(c) => SVal::C(c),
                            None => self.unevaluated(&at)?,
                        },
                    },
                    SVal::Ptr(..) => return Err(not_evaluable(at)),
                }
            }
            InstKind::Freeze(v) => match val(*v)? {
                SVal::C(c) => match self.folder.freeze(c, m)? {
                    Some(c) => SVal::C(c),
                    None => self.unevaluated(&at)?,
                },
                other => other,
            },
            InstKind::Select { c, t, f, .. } => {
                let (c, t, f) = (val(*c)?, val(*t)?, val(*f)?);
                match c {
                    SVal::Fc(_) => f,
                    SVal::Opaque => SVal::Opaque,
                    SVal::C(k) => match self.folder.int(k) {
                        Some(0) => f,
                        Some(_) => t,
                        None => return Err(not_evaluable(at)),
                    },
                    SVal::Ptr(..) => return Err(not_evaluable(at)),
                }
            }
            InstKind::ExtractElement { vec, idx } => {
                let (v, x) = (val(*vec)?, val(*idx)?);
                self.pure2(v, x, &at, m, |f, v, x, m| f.extract_element(v, x, m))?
            }
            InstKind::InsertElement { vec, elt, idx } => {
                let (v, e, x) = (val(*vec)?, val(*elt)?, val(*idx)?);
                match (taint(&[v, e, x]), v, e, x) {
                    (Some(t), ..) => t,
                    (None, SVal::C(v), SVal::C(e), SVal::C(x)) => {
                        match self.folder.insert_element(ty, v, e, x, m)? {
                            Some(c) => SVal::C(c),
                            None => self.unevaluated(&at)?,
                        }
                    }
                    _ => return Err(not_evaluable(at)),
                }
            }
            InstKind::ExtractValue { agg, indices } => match val(*agg)? {
                SVal::C(c) => match self.folder.extract_value(c, indices, m)? {
                    Some(c) => SVal::C(c),
                    None => self.unevaluated(&at)?,
                },
                SVal::Fc(i) => SVal::Fc(i),
                SVal::Opaque => SVal::Opaque,
                SVal::Ptr(..) => return Err(not_evaluable(at)),
            },
            InstKind::ShuffleVector { a, b, mask } => {
                let (a, b) = (val(*a)?, val(*b)?);
                self.pure2(a, b, &at, m, |f, a, b, m| f.shuffle(ty, a, b, mask, m))?
            }
            InstKind::InsertValue {
                agg,
                val: v,
                indices,
            } => {
                let (a, v) = (val(*agg)?, val(*v)?);
                self.pure2(a, v, &at, m, |f, a, v, m| f.insert_value(a, v, indices, m))?
            }
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                let base = val(*base)?;
                let idx = indices
                    .iter()
                    .map(|&x| val(x))
                    .collect::<Result<Vec<_>, _>>()?;
                self.gep(*src_ty, base, &idx, &at)?
            }
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => {
                let p = val(*ptr)?;
                self.load(ty, p, &at, m)?
            }
            InstKind::Store {
                val: v,
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => {
                let vty = self.value_ty(fr, *v);
                let (v, p) = (val(*v)?, val(*ptr)?);
                self.store(vty, v, p, &at, m)?;
                return Ok(Step::Next(None));
            }
            _ => return Err(not_evaluable(at)),
        };
        Ok(Step::Next(Some(v)))
    }

    /// The IR type of an operand of `fr`'s function.
    fn value_ty(&self, fr: &Frame<'_>, v: Value) -> Option<TypeId> {
        match v {
            Value::Const(c) => self.folder.consts.ty(c),
            Value::Inst(i) => fr.body.insts.get(i).map(|i| i.ty),
            Value::Arg(a) => match self.types.get(self.funcs.get(fr.func)?.ty)? {
                Type::Function { params, .. } => params.get(a.index()).copied(),
                _ => None,
            },
        }
    }

    /// Move `fr` to `dest`. A second visit of a block is a loop, which the evaluator refuses.
    fn jump(&self, fr: &mut Frame<'_>, dest: BlockId, at: &Location) -> Result<Step, Error> {
        let seen = fr
            .visited
            .get_mut(dest.index())
            .ok_or_else(|| invariant("block out of range"))?;
        if *seen {
            return Err(not_evaluable(at.clone()));
        }
        *seen = true;
        fr.prev = Some(fr.block);
        fr.block = dest;
        fr.at = fr
            .body
            .blocks
            .get(dest)
            .ok_or_else(|| invariant("block out of range"))?
            .start;
        Ok(Step::Jumped)
    }

    /// The integer a branch condition holds: an undefined one's 0, its false way.
    fn decide(&self, v: SVal, at: &Location) -> Result<u64, Error> {
        match v {
            SVal::Fc(_) => Ok(0),
            SVal::C(c) => match self.folder.int(c) {
                Some(x) => Ok(x),
                None => match self.folder.consts.get(c).map(|k| &k.value) {
                    Some(Const::Undef | Const::Poison) => {
                        Err(Error::InvalidInput(InputError::new(
                            InputErrorKind::Canon(CanonError::UndefCondition),
                            at.clone(),
                        )))
                    }
                    _ => Err(not_evaluable(at.clone())),
                },
            },
            SVal::Ptr(..) | SVal::Opaque => Err(not_evaluable(at.clone())),
        }
    }

    fn int(&mut self, ty: TypeId, v: u64, m: &mut Meter) -> Result<ConstId, Error> {
        self.folder.consts.intern(
            Constant {
                ty,
                value: Const::Int(v),
            },
            m,
        )
    }

    /// A two-operand pure operation: undefined if either operand is, `fold` over constants.
    fn pure2(
        &mut self,
        a: SVal,
        b: SVal,
        at: &Location,
        m: &mut Meter,
        fold: impl FnOnce(
            &mut Folder<'a>,
            ConstId,
            ConstId,
            &mut Meter,
        ) -> Result<Option<ConstId>, Error>,
    ) -> Result<SVal, Error> {
        if let Some(t) = taint(&[a, b]) {
            return Ok(t);
        }
        match (a, b) {
            (SVal::C(x), SVal::C(y)) => match fold(&mut self.folder, x, y, m)? {
                Some(c) => Ok(SVal::C(c)),
                None => self.unevaluated(at),
            },
            _ => Err(not_evaluable(at.clone())),
        }
    }

    /// The global and byte offset a pointer constant addresses; `None` for a constant that is
    /// not a pointer into a global (an integer, `null`).
    fn ptr_of(&self, c: ConstId, at: &Location) -> Result<Option<(GlobalId, u64)>, Error> {
        let mut c = c;
        let mut extra = 0u64;
        loop {
            match self.folder.consts.get(c).map(|k| &k.value) {
                Some(Const::GlobalAddr(g)) => return Ok(Some((*g, extra))),
                Some(Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ops: [v],
                })) => c = *v,
                Some(Const::Expr(ConstExpr::Gep { src_ty, ops, .. })) => {
                    let (base, idx) = ops.split_first().ok_or_else(|| not_evaluable(at.clone()))?;
                    let idx: Vec<SVal> = idx.iter().map(|&k| SVal::C(k)).collect();
                    let off = self.offset(*src_ty, &idx, at)?;
                    extra = extra
                        .checked_add(off)
                        .ok_or_else(|| not_evaluable(at.clone()))?;
                    c = *base;
                }
                _ => return Ok(None),
            }
        }
    }

    /// The byte offset of `getelementptr src_ty, _, idx...` (constant indices only).
    fn offset(&self, src_ty: TypeId, idx: &[SVal], at: &Location) -> Result<u64, Error> {
        let err = || not_evaluable(at.clone());
        let index = |v: &SVal| -> Result<i64, Error> {
            let SVal::C(c) = *v else { return Err(err()) };
            let bits = self
                .folder
                .consts
                .ty(c)
                .and_then(|t| self.types.int_bits(t))
                .ok_or_else(err)?;
            let x = self.folder.int(c).ok_or_else(err)?;
            let shift = 64 - u32::from(bits.clamp(1, 64));
            Ok(((x << shift) as i64) >> shift)
        };
        let Some((first, rest)) = idx.split_first() else {
            return Ok(0);
        };
        let size = self.layouts.alloc_size(src_ty).ok_or_else(err)?;
        let mut off = index(first)?
            .checked_mul(i64::try_from(size).map_err(|_| err())?)
            .ok_or_else(err)?;
        let mut ty = src_ty;
        for x in rest {
            let i = index(x)?;
            let (step, next) = match self.types.get(ty) {
                Some(Type::Struct { fields, .. }) => {
                    let k = usize::try_from(i).map_err(|_| err())?;
                    let offs = self.layouts.field_offsets(ty).ok_or_else(err)?;
                    let o = *offs.get(k).ok_or_else(err)?;
                    (
                        i64::try_from(o).map_err(|_| err())?,
                        *fields.get(k).ok_or_else(err)?,
                    )
                }
                // A vector steps by its element's allocation size (LLVM's GEP rule), taken only
                // for byte-sized elements, where that is also where the element is stored.
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                    let l = self.layouts.get(*elem).ok_or_else(err)?;
                    let vector = matches!(self.types.get(ty), Some(Type::Vector { .. }));
                    if vector && l.alloc_size != l.store_size {
                        return Err(err());
                    }
                    (
                        i.checked_mul(i64::try_from(l.alloc_size).map_err(|_| err())?)
                            .ok_or_else(err)?,
                        *elem,
                    )
                }
                _ => return Err(err()),
            };
            off = off.checked_add(step).ok_or_else(err)?;
            ty = next;
        }
        u64::try_from(off).map_err(|_| err())
    }

    fn gep(&self, src_ty: TypeId, base: SVal, idx: &[SVal], at: &Location) -> Result<SVal, Error> {
        if let Some(t) = taint(idx).or(taint(&[base])) {
            return Ok(t);
        }
        let (g, o) = match base {
            SVal::Ptr(g, o) => (g, o),
            SVal::C(c) => self
                .ptr_of(c, at)?
                .ok_or_else(|| not_evaluable(at.clone()))?,
            SVal::Fc(_) | SVal::Opaque => return Ok(base),
        };
        let off = self.offset(src_ty, idx, at)?;
        Ok(SVal::Ptr(
            g,
            o.checked_add(off)
                .ok_or_else(|| not_evaluable(at.clone()))?,
        ))
    }

    fn pointer(&self, p: SVal, at: &Location) -> Result<(GlobalId, u64), Error> {
        match p {
            SVal::Ptr(g, o) => Ok((g, o)),
            SVal::C(c) => self.ptr_of(c, at)?.ok_or_else(|| not_evaluable(at.clone())),
            SVal::Fc(_) | SVal::Opaque => Err(not_evaluable(at.clone())),
        }
    }

    /// The leaves of `g`, materialized from its initializer on first touch; records that the
    /// running initializer touched `g`.
    fn cells(&mut self, g: GlobalId, at: &Location, m: &mut Meter) -> Result<(), Error> {
        let current = self.current;
        if let Some(t) = self.mem.get_mut(&g) {
            t.shared |= t.first != current;
            return Ok(());
        }
        let global = self
            .globals
            .get(g)
            .ok_or_else(|| invariant("global out of range"))?;
        if global.externally_initialized {
            return Err(not_evaluable(at.clone()));
        }
        let init = global.init.ok_or_else(|| not_evaluable(at.clone()))?;
        let mut cells = Cells::new();
        let mut work = vec![(global.value_ty, init, 0u64)];
        while let Some((ty, c, off)) = work.pop() {
            m.tick(1)?;
            if leaf(self.types, ty) {
                if cells.len() as u64 >= LEAF_LIMIT {
                    return Err(not_evaluable(at.clone()));
                }
                self.charge(m, 4 * std::mem::size_of::<(u64, (TypeId, SVal))>())?;
                let v = match self.ptr_of(c, at)? {
                    Some((g, o)) if o != 0 => SVal::Ptr(g, o),
                    _ => SVal::C(c),
                };
                cells.insert(off, (ty, v));
                continue;
            }
            let members = self.members(ty, c, at, m)?;
            for (t, k, o) in members {
                work.push((
                    t,
                    k,
                    off.checked_add(o)
                        .ok_or_else(|| not_evaluable(at.clone()))?,
                ));
            }
        }
        self.mem.insert(
            g,
            Touched {
                cells,
                written: false,
                first: current,
                shared: false,
            },
        );
        Ok(())
    }

    /// The members of aggregate constant `c` of type `ty`, with their byte offsets.
    fn members(
        &mut self,
        ty: TypeId,
        c: ConstId,
        at: &Location,
        m: &mut Meter,
    ) -> Result<Vec<(TypeId, ConstId, u64)>, Error> {
        let err = || not_evaluable(at.clone());
        let (tys, offs): (Vec<TypeId>, Vec<u64>) = match self.types.get(ty) {
            Some(Type::Struct { fields, .. }) => {
                let offs = self.layouts.field_offsets(ty).ok_or_else(err)?;
                (fields.to_vec(), offs.to_vec())
            }
            Some(Type::Array { elem, len }) => {
                if *len > LEAF_LIMIT {
                    return Err(err());
                }
                let s = self.layouts.alloc_size(*elem).ok_or_else(err)?;
                let n = usize::try_from(*len).map_err(|_| err())?;
                self.charge(m, n * 16)?;
                (vec![*elem; n], (0..*len).map(|i| i * s).collect())
            }
            _ => return Err(err()),
        };
        let fill = match self.folder.consts.get(c).map(|k| k.value.clone()) {
            Some(Const::Aggregate(elems)) => {
                if elems.len() != tys.len() {
                    return Err(invariant("an aggregate constant of the wrong arity"));
                }
                return Ok(tys
                    .into_iter()
                    .zip(elems.iter().copied())
                    .zip(offs)
                    .map(|((t, k), o)| (t, k, o))
                    .collect());
            }
            Some(v @ (Const::Null | Const::Undef | Const::Poison)) => v,
            _ => return Err(err()),
        };
        let mut out = Vec::with_capacity(tys.len());
        for (t, o) in tys.into_iter().zip(offs) {
            let k = self.folder.consts.intern(
                Constant {
                    ty: t,
                    value: fill.clone(),
                },
                m,
            )?;
            out.push((t, k, o));
        }
        Ok(out)
    }

    /// The leaves an access of type `ty` at `off` covers: `(offset, leaf type)`.
    fn leaves(
        &self,
        ty: TypeId,
        off: u64,
        at: &Location,
        out: &mut Vec<(u64, TypeId)>,
    ) -> Result<(), Error> {
        let err = || not_evaluable(at.clone());
        let mut work = vec![(ty, off)];
        while let Some((t, o)) = work.pop() {
            if leaf(self.types, t) {
                if out.len() as u64 >= LEAF_LIMIT {
                    return Err(err());
                }
                out.push((o, t));
                continue;
            }
            match self.types.get(t) {
                Some(Type::Struct { fields, .. }) => {
                    let offs = self.layouts.field_offsets(t).ok_or_else(err)?;
                    for (&f, &fo) in fields.iter().zip(offs.iter()).rev() {
                        work.push((f, o.checked_add(fo).ok_or_else(err)?));
                    }
                }
                Some(Type::Array { elem, len }) => {
                    if *len > LEAF_LIMIT {
                        return Err(err());
                    }
                    let s = self.layouts.alloc_size(*elem).ok_or_else(err)?;
                    for k in (0..*len).rev() {
                        work.push((*elem, o.checked_add(k * s).ok_or_else(err)?));
                    }
                }
                _ => return Err(err()),
            }
        }
        out.sort_by_key(|&(o, _)| o);
        Ok(())
    }

    fn load(&mut self, ty: TypeId, p: SVal, at: &Location, m: &mut Meter) -> Result<SVal, Error> {
        let (g, off) = self.pointer(p, at)?;
        if let Some(state) = self.fcs.get(&g).filter(|s| s.decl && !s.supplied) {
            return Ok(SVal::Fc(state.index));
        }
        self.cells(g, at, m)?;
        let cells = &self
            .mem
            .get(&g)
            .ok_or_else(|| invariant("cells not materialized"))?
            .cells;
        if let Some(&(t, v)) = cells.get(&off) {
            if t == ty {
                return Ok(v);
            }
        }
        let mut want = Vec::new();
        self.leaves(ty, off, at, &mut want)?;
        let matched = want.len() > 1
            && want
                .iter()
                .all(|(o, t)| cells.get(o).is_some_and(|&(ct, _)| ct == *t));
        if !matched {
            // An access that does not match the leaves: a union read through another member, an
            // integer load over several integer fields of a table (or inside one), or a copy of
            // a float or vector through an integer of its size. It reads undefined bytes as
            // undefined, and bytes of known value as the value they spell (`read_bytes`);
            // anything else is not evaluable.
            if let Some(t) = self.overlap_taint(cells, ty, off) {
                return Ok(t);
            }
            let bytes = self
                .read_bytes(cells, ty, off, m)?
                .ok_or_else(|| not_evaluable(at.clone()))?;
            let value = self.bytes_const(ty, &bytes, m);
            m.release(bytes.len());
            return Ok(SVal::C(value?.ok_or_else(|| not_evaluable(at.clone()))?));
        }
        let mut vals = Vec::with_capacity(want.len());
        for (o, _) in &want {
            if let Some(&(_, v)) = cells.get(o) {
                vals.push(v);
            }
        }
        if let Some(t) = taint(&vals) {
            return Ok(t);
        }
        let consts: Vec<ConstId> = vals
            .iter()
            .map(|v| match v {
                SVal::C(c) => Ok(*c),
                _ => Err(not_evaluable(at.clone())),
            })
            .collect::<Result<_, _>>()?;
        let mut next = consts.into_iter();
        Ok(SVal::C(self.build(ty, &mut next, at, m)?))
    }

    /// The lanes of a type each of whose bytes in memory is a byte of one lane's value: an
    /// integer of whole bytes, a float, or a vector of either (lane `i` at byte `i · size`, its
    /// elements filling whole bytes). Returns the lane size in bytes, the lane count, and whether
    /// the lanes are floats. `None` for any other type: a pointer (its address is no constant of
    /// known bytes), an integer that does not fill whole bytes (an `i1`'s unused bits), and an
    /// aggregate (its padding).
    fn byte_lanes(&self, t: TypeId) -> Option<(u64, u64, bool)> {
        let scalar = |t: &Type| match *t {
            Type::Int(w) if w % 8 == 0 && (8..=64).contains(&w) => Some((u64::from(w / 8), false)),
            ref t => t.float_bits().map(|b| (u64::from(b / 8), true)),
        };
        match self.types.get(t)? {
            Type::Vector { elem, len } => {
                let (size, float) = scalar(self.types.get(*elem)?)?;
                Some((size, u64::from(*len), float))
            }
            t => scalar(t).map(|(size, float)| (size, 1, float)),
        }
    }

    /// Byte `k` of leaf constant `c` of type `t` in memory, in the module's little-endian order:
    /// byte `b` of a lane holds its bits `8b..8b+8` (a float's are its IEEE 754 encoding), and
    /// lane `i` starts at byte `i · size`. `None` unless `t` has byte lanes (`byte_lanes`) and
    /// the lane is a constant: an `undef` or `poison` lane has no byte to read.
    fn const_byte(&self, t: TypeId, c: ConstId, k: u64) -> Option<u8> {
        let (size, lanes, _) = self.byte_lanes(t)?;
        let (lane, byte) = (k / size, k % size);
        if lane >= lanes {
            return None;
        }
        let bits = |k: ConstId| match self.folder.consts.get(k).map(|k| &k.value)? {
            Const::Int(x) | Const::Float(x) => Some(*x),
            Const::Null => Some(0),
            _ => None,
        };
        let x = match self.folder.consts.get(c).map(|k| &k.value)? {
            Const::Null => 0,
            Const::Aggregate(elems) if elems.len() as u64 == lanes => {
                bits(*elems.get(usize::try_from(lane).ok()?)?)?
            }
            Const::Int(x) | Const::Float(x) if lanes == 1 => *x,
            _ => return None,
        };
        Some((x >> (8 * byte)) as u8)
    }

    /// The constant of type `t` (with byte lanes, `byte_lanes`) whose bytes in memory are
    /// `bytes`, the inverse of `const_byte`. `None` for any other type, or a byte count other
    /// than its lanes'.
    fn bytes_const(
        &mut self,
        t: TypeId,
        bytes: &[u8],
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let Some((size, lanes, float)) = self.byte_lanes(t) else {
            return Ok(None);
        };
        if size.checked_mul(lanes) != u64::try_from(bytes.len()).ok() {
            return Ok(None);
        }
        let (Ok(size), Ok(lanes)) = (usize::try_from(size), usize::try_from(lanes)) else {
            return Ok(None);
        };
        let vector = match self.types.get(t) {
            Some(Type::Vector { elem, .. }) => Some(*elem),
            _ => None,
        };
        let mut elems = Vec::new();
        m.reserve(&mut elems, lanes)?;
        for lane in bytes.chunks_exact(size) {
            let mut word = [0u8; 8];
            if let Some(dst) = word.get_mut(..size) {
                dst.copy_from_slice(lane);
            }
            let x = u64::from_le_bytes(word);
            let value = if float {
                Const::Float(x)
            } else {
                Const::Int(x)
            };
            let ty = vector.unwrap_or(t);
            elems.push(self.folder.consts.intern(Constant { ty, value }, m)?);
        }
        let c = match vector {
            Some(_) => self.folder.consts.intern(
                Constant {
                    ty: t,
                    value: Const::Aggregate(elems.into_boxed_slice()),
                },
                m,
            )?,
            None => match elems.first() {
                Some(&c) => c,
                None => return Ok(None),
            },
        };
        Ok(Some(c))
    }

    /// The bytes of a load of type `ty` (with byte lanes, `byte_lanes`) at byte `off` that no
    /// leaf of type `ty` at `off` holds: a wide load over several fields of a table (an `i64` at
    /// byte 4 of a `{ i64, [8 x i8] }` union), a narrow one inside a field, or a copy of a float or
    /// a vector through an integer of its size (initializers copy `float`, `half` and `short2`
    /// function constants as `i32` and `i16`). Each byte holds the same value whatever type wrote
    /// it (`const_byte`), so the load is exact. `None` (not evaluable) unless the module is
    /// little-endian and every byte the load reads belongs to a leaf with byte lanes holding a
    /// constant: a byte of padding, of a pointer, of an `undef` leaf, past the global, or an `i1`'s
    /// unused bits stay unread. The bytes are charged to the meter, and the caller releases them.
    fn read_bytes(
        &self,
        cells: &Cells,
        ty: TypeId,
        off: u64,
        m: &mut Meter,
    ) -> Result<Option<Vec<u8>>, Error> {
        let Some(size) = self.byte_lanes(ty).and(self.layouts.store_size(ty)) else {
            return Ok(None);
        };
        let (true, Some(end), Ok(n)) = (self.little, off.checked_add(size), usize::try_from(size))
        else {
            return Ok(None);
        };
        m.charge(n)?;
        let mut bytes = Vec::with_capacity(n);
        let first = cells.range(..=off).next_back().map_or(off, |(&o, _)| o);
        for (&o, &(t, v)) in cells.range(first..end) {
            let next = off + bytes.len() as u64;
            let cell_end = self.layouts.store_size(t).and_then(|s| o.checked_add(s));
            match cell_end {
                Some(e) if e <= next => continue,
                Some(_) if o <= next => {}
                _ => break,
            }
            let SVal::C(c) = v else {
                break;
            };
            let stop = cell_end.map_or(next, |e| e.min(end));
            for b in next..stop {
                m.tick(1)?;
                match self.const_byte(t, c, b - o) {
                    Some(x) => bytes.push(x),
                    None => break,
                }
            }
            if off + (bytes.len() as u64) < stop {
                break;
            }
        }
        if bytes.len() == n {
            Ok(Some(bytes))
        } else {
            m.release(n);
            Ok(None)
        }
    }

    /// Store constant `c` of type `ty` (with byte lanes, `byte_lanes`) at byte `off` of `g` over
    /// leaves of other types (an initializer's copy of a `float` or `short2` function constant
    /// through an `i32`): each leaf the store overlaps takes the stored bytes (`const_byte`), keeps
    /// its others, and is rebuilt from its bytes (`bytes_const`), which is how memory holds them.
    /// Exact when every stored byte lands in a leaf with byte lanes and a leaf the store covers
    /// only in part holds a constant; otherwise not evaluable: a byte stored into padding, a
    /// pointer, or beside an `undef` leaf's other bytes would not be held exactly.
    fn write_bytes(
        &mut self,
        g: GlobalId,
        ty: TypeId,
        c: ConstId,
        off: u64,
        at: &Location,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let err = || not_evaluable(at.clone());
        let size = self
            .byte_lanes(ty)
            .and(self.layouts.store_size(ty))
            .ok_or_else(err)?;
        let end = off.checked_add(size).ok_or_else(err)?;
        if !self.little {
            return Err(err());
        }
        // Each overlapped leaf's new bytes, computed before any leaf changes.
        let mut images: Vec<(u64, TypeId, Vec<u8>)> = Vec::new();
        let mut charged = 0usize;
        let r = (|| {
            let cells = &self
                .mem
                .get(&g)
                .ok_or_else(|| invariant("cells not materialized"))?
                .cells;
            let first = cells.range(..=off).next_back().map_or(off, |(&o, _)| o);
            let mut next = off;
            for (&o, &(t, v)) in cells.range(first..end) {
                let cell_end = self
                    .layouts
                    .store_size(t)
                    .and_then(|s| o.checked_add(s))
                    .ok_or_else(err)?;
                if cell_end <= off {
                    continue;
                }
                // A stored byte no leaf holds (padding between leaves) would be lost.
                if o > next || self.byte_lanes(t).is_none() {
                    return Err(err());
                }
                let n = usize::try_from(cell_end - o).map_err(|_| err())?;
                m.charge(n)?;
                charged += n;
                let mut image = Vec::with_capacity(n);
                for b in o..cell_end {
                    m.tick(1)?;
                    let byte = if (off..end).contains(&b) {
                        self.const_byte(ty, c, b - off)
                    } else {
                        match v {
                            SVal::C(k) => self.const_byte(t, k, b - o),
                            _ => None,
                        }
                    };
                    image.push(byte.ok_or_else(err)?);
                }
                next = cell_end;
                images.push((o, t, image));
            }
            if next < end {
                return Err(err());
            }
            Ok(())
        })();
        let rebuilt = r.and_then(|()| {
            let mut out = Vec::with_capacity(images.len());
            for (o, t, image) in &images {
                let k = self.bytes_const(*t, image, m)?.ok_or_else(err)?;
                out.push((*o, SVal::C(k)));
            }
            Ok(out)
        });
        m.release(charged);
        let rebuilt = rebuilt?;
        let entry = self
            .mem
            .get_mut(&g)
            .ok_or_else(|| invariant("cells not materialized"))?;
        for (o, v) in rebuilt {
            if let Some(cell) = entry.cells.get_mut(&o) {
                cell.1 = v;
            }
        }
        Ok(())
    }

    /// Rebuild a constant of type `ty` from its leaves in offset order.
    ///
    /// Nothing caps how deep aggregate types nest, so the walk is post-order over an explicit
    /// stack of partly built aggregates: its host stack use is the same at every depth.
    fn build(
        &mut self,
        ty: TypeId,
        leaves: &mut impl Iterator<Item = ConstId>,
        at: &Location,
        m: &mut Meter,
    ) -> Result<ConstId, Error> {
        /// An aggregate being built: its type, member types, and the members built so far.
        struct Partial {
            ty: TypeId,
            members: Vec<TypeId>,
            elems: Vec<ConstId>,
        }
        let mut stack: Vec<Partial> = Vec::new();
        let mut next = ty;
        loop {
            let mut built = if leaf(self.types, next) {
                Some(leaves.next().ok_or_else(|| invariant("too few leaves"))?)
            } else {
                let members: Vec<TypeId> = match self.types.get(next) {
                    Some(Type::Struct { fields, .. }) => fields.to_vec(),
                    Some(Type::Array { elem, len }) => {
                        vec![*elem; usize::try_from(*len).map_err(|_| not_evaluable(at.clone()))?]
                    }
                    _ => return Err(not_evaluable(at.clone())),
                };
                stack.push(Partial {
                    ty: next,
                    elems: Vec::with_capacity(members.len()),
                    members,
                });
                None
            };
            // Hand each finished value to its parent, finishing parents whose members are all
            // built, until one has a member left to build.
            loop {
                let Some(top) = stack.last_mut() else {
                    return built.ok_or_else(|| invariant("build finished no value"));
                };
                if let Some(c) = built.take() {
                    top.elems.push(c);
                }
                if let Some(&t) = top.members.get(top.elems.len()) {
                    m.tick(1)?;
                    next = t;
                    break;
                }
                let Some(done) = stack.pop() else {
                    return Err(invariant("build stack emptied"));
                };
                built = Some(self.folder.consts.intern(
                    Constant {
                        ty: done.ty,
                        value: Const::Aggregate(done.elems.into_boxed_slice()),
                    },
                    m,
                )?);
            }
        }
    }

    fn store(
        &mut self,
        ty: Option<TypeId>,
        v: SVal,
        p: SVal,
        at: &Location,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let ty = ty.ok_or_else(|| not_evaluable(at.clone()))?;
        let (g, off) = self.pointer(p, at)?;
        let global = self
            .globals
            .get(g)
            .ok_or_else(|| invariant("global out of range"))?;
        if global.space != AddrSpace::Constant || self.fcs.get(&g).is_some_and(|s| s.decl) {
            return Err(not_evaluable(at.clone()));
        }
        self.cells(g, at, m)?;
        let mut want = Vec::new();
        self.leaves(ty, off, at, &mut want)?;
        let parts: Vec<SVal> = match v {
            _ if want.len() == 1 => vec![v],
            SVal::Fc(_) | SVal::Opaque => vec![v; want.len()],
            SVal::C(c) => {
                let mut out = Vec::with_capacity(want.len());
                let mut work = vec![(ty, c)];
                while let Some((t, k)) = work.pop() {
                    if leaf(self.types, t) {
                        out.push(SVal::C(k));
                        continue;
                    }
                    let members = self.members(t, k, at, m)?;
                    work.extend(members.into_iter().rev().map(|(t, k, _)| (t, k)));
                }
                out
            }
            SVal::Ptr(..) => return Err(not_evaluable(at.clone())),
        };
        let size = self
            .layouts
            .store_size(ty)
            .ok_or_else(|| not_evaluable(at.clone()))?;
        let layouts = self.layouts;
        let entry = self
            .mem
            .get_mut(&g)
            .ok_or_else(|| invariant("cells not materialized"))?;
        let exact = want
            .iter()
            .all(|(o, t)| entry.cells.get(o).is_some_and(|cell| cell.0 == *t));
        entry.written = true;
        if exact {
            for ((o, _), part) in want.iter().zip(parts) {
                if let Some(cell) = entry.cells.get_mut(o) {
                    cell.1 = part;
                }
            }
            return Ok(());
        }
        // A store through another member of a union, or a copy of a float or vector through an
        // integer of its size: a constant's bytes replace those of the leaves they overlap
        // (`write_bytes`), and undefined (or opaque) bytes make every leaf they overlap
        // undefined (or opaque).
        match v {
            SVal::Fc(_) | SVal::Opaque => {}
            SVal::C(c) => return self.write_bytes(g, ty, c, off, at, m),
            SVal::Ptr(..) => return Err(not_evaluable(at.clone())),
        }
        let end = off
            .checked_add(size)
            .ok_or_else(|| not_evaluable(at.clone()))?;
        let mut any = false;
        for (&o, cell) in entry.cells.range_mut(..end) {
            let cell_end = o.saturating_add(layouts.store_size(cell.0).unwrap_or(0));
            if cell_end > off {
                cell.1 = v;
                any = true;
            }
        }
        if any {
            Ok(())
        } else {
            Err(not_evaluable(at.clone()))
        }
    }

    /// The undefined value among the leaves a mismatched access of type `ty` at `off` overlaps.
    fn overlap_taint(&self, cells: &Cells, ty: TypeId, off: u64) -> Option<SVal> {
        let end = off.checked_add(self.layouts.store_size(ty)?)?;
        let overlapping: Vec<SVal> = cells
            .range(..end)
            .filter(|&(&o, &(t, _))| {
                o.saturating_add(self.layouts.store_size(t).unwrap_or(0)) > off
            })
            .map(|(_, &(_, v))| v)
            .collect();
        taint(&overlapping)
    }

    /// Whether the initializers that ran compute the same in every order: no global one of them
    /// wrote was touched by another. Each initializer then read only initial values or its own
    /// stores, so in any order each would take the path it took here and store what it stored.
    fn commute(&self) -> bool {
        !self.mem.values().any(|t| t.written && t.shared)
    }

    /// The new initializer of every written global, and the globals derived from an unsupplied
    /// constant or left opaque.
    fn fold(&mut self, m: &mut Meter) -> Result<(Inits, Evaluated), Error> {
        let written: Vec<(GlobalId, Vec<SVal>)> = self
            .mem
            .iter()
            .filter(|(_, t)| t.written)
            .map(|(&g, t)| (g, t.cells.values().map(|&(_, v)| v).collect()))
            .collect();
        let mut inits = Vec::with_capacity(written.len());
        let mut derived = Vec::new();
        let mut opaque = Vec::new();
        for (g, vals) in written {
            let at = Location::module();
            let ty = self
                .globals
                .get(g)
                .ok_or_else(|| invariant("global out of range"))?
                .value_ty;
            match taint(&vals) {
                Some(SVal::Fc(i)) => {
                    let undef = self.folder.consts.intern(
                        Constant {
                            ty,
                            value: Const::Undef,
                        },
                        m,
                    )?;
                    inits.push((g, undef));
                    derived.push((g, i));
                    continue;
                }
                Some(_) => {
                    opaque.push(g);
                    continue;
                }
                None => {}
            }
            let mut consts = Vec::with_capacity(vals.len());
            for v in vals {
                consts.push(match v {
                    SVal::C(c) => c,
                    SVal::Ptr(p, 0) => self.folder.consts.intern(
                        Constant {
                            ty: self
                                .global_ptr_ty(p)
                                .ok_or_else(|| not_evaluable(at.clone()))?,
                            value: Const::GlobalAddr(p),
                        },
                        m,
                    )?,
                    _ => return Err(not_evaluable(at)),
                });
            }
            let mut next = consts.into_iter();
            inits.push((g, self.build(ty, &mut next, &at, m)?));
        }
        Ok((inits, Evaluated { derived, opaque }))
    }

    /// The pointer type of global `g`'s address.
    fn global_ptr_ty(&self, g: GlobalId) -> Option<TypeId> {
        let space = self.globals.get(g)?.space;
        self.types.lookup(&Type::Ptr { space })
    }
}

/// Whether memory holds a value of type `ty` as one leaf.
fn leaf(types: &TypeTable, ty: TypeId) -> bool {
    matches!(
        types.get(ty),
        Some(
            Type::Int(_)
                | Type::Half
                | Type::BFloat
                | Type::Float
                | Type::Double
                | Type::Ptr { .. }
                | Type::Vector { .. }
        )
    )
}

/// The undefined value (lowest index) among `vals`, else an opaque one, if any.
fn taint(vals: &[SVal]) -> Option<SVal> {
    vals.iter()
        .filter_map(|v| match v {
            SVal::Fc(i) => Some(*i),
            _ => None,
        })
        .min()
        .map(SVal::Fc)
        .or_else(|| vals.contains(&SVal::Opaque).then_some(SVal::Opaque))
}
