//! Pointer cells: pointers stored into and loaded back from forwarding allocas.
//!
//! When every access to an alloca is visible (`memory::provenance`'s cells), a pointer loaded
//! from it is one of the pointers stored at its slot: its roots are theirs, so its representation
//! is chosen like any other pointer's, and the cell holds each pointer in that lowered
//! representation, in the private arena at the slot's offset:
//!
//! - a `u32` word ([`Encoding::Word`]): an arena offset (threadgroup, private, a handle array, a
//!   buffer table) or a handle index (the null handle is 0);
//! - two `u32` words ([`Encoding::Pair`]): a narrow fat pointer `(root, offset)` (root 0 is null).
//!
//! The representations agree between a store and every load of its slot: a load's roots contain
//! the stored pointer's, in one address space and one class (provenance joins them), and each
//! encoding numbers its roots globally (arena offsets, slot roots, handle candidates). A wide
//! pointer keeps its 8-byte guest address (`access::dynamic`): a stored pointer is wide exactly
//! when its load is, since both root sets are in one component. A null or undefined pointer stored
//! writes zeros to the slot's eight bytes, which read back as null in every encoding (and as guest
//! address 0). A slot never written reads as whatever the arena holds; every encoding's uses are
//! safe on any bits (the null arm of a handle or fat-pointer switch, the arena's containment).
//!
//! A volatile cell access is the same slot access whose words carry `Volatile`: the slot is the
//! invocation's own, so a volatile load still reads the pointer last stored there.
//!
//! **Slot groups**. The stores and loads that meet at a slot form one group (`CellFacts::groups`),
//! and a store writes the encoding the group's loads read, checked before any store is selected
//! ([`plan_reads`]). A pointer into the constant blob is its `u32` blob offset (`access::blob`), so
//! a slot whose loads are all blob pointers holds that word; a load whose roots also hold a buffer
//! slot or null is a static fat pointer, which names the blob `BLOB_ROOT`, so a blob pointer stored
//! where such loads read it is written as `(BLOB_ROOT, offset)`. A store whose group's loads read
//! an encoding other than the one it writes is refused.
//!
//! Not implemented (`Unsupported{Lowering(Access), NotImplemented}`): a heap texture handle, a
//! constant pointer other than null, a threadgroup global's address, or a blob-form constant
//! global's address, a store whose slot group's loads read two encodings, and atomic cell accesses.

use std::collections::BTreeMap;

use crate::air::ir::{Align, CastOp, Const, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;
use crate::lower::arith::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::arena::Arena;
use crate::memory::leaf::{self, k32, Leaf, Path};
use crate::memory::provenance::{PrivForm, Repr, Root};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;

use super::emit::side;
use super::local::{self, Home, LocalAccess, LocalPtr};
use super::{blob, is_ptr_ty, operand, refuse, PtrSide};

/// How a cell slot holds a pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// A `u32`: an arena offset or a handle index.
    Word,
    /// A narrow fat pointer `(root, offset)`.
    Pair,
}

/// What a cell store writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CellValue {
    /// A pointer value, in its lowered representation.
    Value(ValueId),
    /// A threadgroup global's address (its arena offset) or a blob-form constant global's (its
    /// blob offset).
    Offset(u32),
    /// A pointer into the constant blob stored where static fat pointers read it:
    /// `(BLOB_ROOT, offset)`.
    Blob(LocalPtr),
    /// The null or an undefined pointer: zeros.
    Zero,
}

/// The encodings the loads of one slot group read ([`plan_reads`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Reads {
    /// Some load reads a `u32` word.
    word: bool,
    /// Some load reads a narrow fat pointer.
    pair: bool,
    /// Some load reads no cell encoding: a wide pointer's guest address, or a representation no
    /// cell holds (that load is refused).
    other: bool,
}

impl Reads {
    /// Record one load's encoding: a refused representation (an `Unsupported` refusal, which
    /// the load's own selection raises again) reads no cell encoding; any other error, an
    /// internal invariant above all, is the planning's own failure and propagates.
    fn add(&mut self, enc: Result<Option<Encoding>, Error>) -> Result<(), Error> {
        match enc {
            Ok(Some(Encoding::Word)) => self.word = true,
            Ok(Some(Encoding::Pair)) => self.pair = true,
            Ok(None) | Err(Error::Unsupported(_)) => self.other = true,
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// Whether every load reads `enc`.
    fn only(self, enc: Encoding) -> bool {
        !self.other
            && match enc {
                Encoding::Word => !self.pair,
                Encoding::Pair => !self.word,
            }
    }
}

/// A load or store of one cell slot: the private-arena access of its word or words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellAccess {
    pub access: LocalAccess,
    /// The stored pointer (stores only).
    pub value: Option<CellValue>,
}

/// The pointer-cell strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CellStrategy {
    Load(CellAccess),
    Store(CellAccess),
    /// A `ptrtoint` whose bits go only into forwarding cells: nothing; its stores store the
    /// pointer.
    Bits,
    /// An `inttoptr` of a pointer word read from a forwarding cell: the word, which holds the
    /// pointer's lowered representation.
    Word(ValueId),
}

impl CellStrategy {
    /// The volatile form (`access::volatile_form`): the slot's words carry `Volatile`.
    pub(super) fn volatile(self) -> CellStrategy {
        let v = |a: CellAccess| CellAccess {
            access: LocalAccess {
                volatile: true,
                ..a.access
            },
            ..a
        };
        match self {
            CellStrategy::Load(a) => CellStrategy::Load(v(a)),
            CellStrategy::Store(a) => CellStrategy::Store(v(a)),
            CellStrategy::Bits => CellStrategy::Bits,
            CellStrategy::Word(w) => CellStrategy::Word(w),
        }
    }

    /// The census name.
    pub fn name(&self) -> &'static str {
        match self {
            CellStrategy::Load(_) => "access.cell.load",
            CellStrategy::Store(_) => "access.cell.store",
            CellStrategy::Bits => "access.cell.bits",
            CellStrategy::Word(_) => "access.cell.word",
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            CellStrategy::Load(a) | CellStrategy::Store(a) => a.access.words(),
            CellStrategy::Bits | CellStrategy::Word(_) => 0,
        }
    }
}

/// The encoding of pointer value `v` in a cell, `None` for a wide pointer (its guest address,
/// `access::dynamic`), or the refusal of a representation no cell holds yet.
fn encoding(cx: &SelectCx<'_>, v: Value) -> Result<Option<Encoding>, Error> {
    let r = cx
        .memory
        .repr(v)
        .ok_or_else(|| cx.invariant("a cell pointer without a representation"))?;
    Ok(Some(match r {
        Repr::BufPtr { wide: true, .. } => return Ok(None),
        // A blob offset (`access::blob`).
        Repr::BufPtr { wide: false, .. } => match blob::is_blob(cx, v)? {
            true => Encoding::Word,
            false => Encoding::Pair,
        },
        Repr::Handle { cands, .. }
            if cx.memory.roots(cands).binary_search(&Root::Dynamic).is_ok() =>
        {
            return Err(refuse())
        }
        Repr::Handle { .. }
        | Repr::TgPtr
        | Repr::HandleArray { .. }
        | Repr::BufferTable { .. }
        | Repr::PrivPtr(PrivForm::Arena) => Encoding::Word,
        // A pointer stored into a cell takes its object out of the typed form, and a loaded
        // pointer's roots are stored pointers' roots. A payload pointer in a cell is not
        // implemented.
        Repr::PrivPtr(PrivForm::Typed(_))
        | Repr::TgTyped(_)
        | Repr::ConstPtr(_)
        | Repr::PayloadPtr => return Err(refuse()),
    }))
}

/// The leaf of an encoding at a slot aligned to `align` bytes.
fn leaf(enc: Encoding, align: Option<Align>) -> Leaf {
    let lanes = match enc {
        Encoding::Word => 1,
        Encoding::Pair => 2,
    };
    Leaf {
        form: Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes,
        },
        bytes: 4 * u32::from(lanes),
        path: if align.map_or(1, |a| a.bytes()) >= 4 {
            Path::Aligned
        } else {
            Path::Bytes
        },
    }
}

/// The private-arena access of the slot `ptr` addresses.
fn slot(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    enc: Encoding,
    align: Option<Align>,
) -> Result<LocalAccess, Error> {
    let home = Home::Arena(Arena::Private);
    if local::home_of(cx, ptr) != Some(home) {
        return Err(cx.invariant("a forwarding cell outside the private arena"));
    }
    local::use_arena(cx, Arena::Private)?;
    Ok(LocalAccess {
        arena: Arena::Private,
        ptr: local::local_ptr(cx, ptr, home)?,
        value: None,
        leaf: leaf(enc, align),
        contained: true,
        volatile: false,
    })
}

/// A constant pointer stored into a cell: the null or an undefined pointer (zeros, whatever the
/// slot's encoding), a threadgroup global's address, or a blob-form constant global's address.
fn constant(cx: &SelectCx<'_>, v: Value) -> Result<(Encoding, CellValue), Error> {
    let Value::Const(c) = v else {
        return Err(cx.invariant("a cell constant that is not a constant"));
    };
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::Null | Const::Undef | Const::Poison) => Ok((Encoding::Pair, CellValue::Zero)),
        _ if local::home_of(cx, v) == Some(Home::Arena(Arena::Threadgroup)) => {
            match local::local_ptr(cx, v, Home::Arena(Arena::Threadgroup))? {
                LocalPtr::Offset(o) => Ok((Encoding::Word, CellValue::Offset(o))),
                LocalPtr::Value(_) => Err(refuse()),
            }
        }
        Some(Const::GlobalAddr(_)) if blob::is_blob(cx, v)? => match blob::blob_ptr(cx, v)? {
            LocalPtr::Offset(o) => Ok((Encoding::Word, CellValue::Offset(o))),
            LocalPtr::Value(_) => Err(refuse()),
        },
        _ => Err(refuse()),
    }
}

/// Whether a cell holds pointer `v` in an encoding of its own: not a wide pointer, whose cell bytes
/// are its guest address, so its bits and the integer are one; a refused representation (an
/// `Unsupported` refusal) holds none either, and any other error propagates.
fn encoded(cx: &SelectCx<'_>, v: Value) -> Result<bool, Error> {
    match encoding(cx, v) {
        Ok(Some(_)) => Ok(true),
        Ok(None) | Err(Error::Unsupported(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Plan the encodings the loads of each slot group read, before any store is selected:
/// `AccessPlan::cell_reads`, by group; and the pointer bits that move through forwarding cells as
/// their pointer: the `ptrtoint`s that emit nothing (`AccessPlan::cell_bits`) and the pointer-word
/// loads (`AccessPlan::cell_words`), each one whose pointer a cell holds in its own encoding
/// (`encoded`). Any other pointer's bits take the integer path: a wide pointer's guest address is
/// the integer, stored and loaded as data, and a refused one is refused where its cell is read.
pub fn plan_reads(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let cells = &cx.memory.cells;
    let mut bits = std::collections::BTreeSet::new();
    for (&c, &p) in &cells.bits {
        m.tick(1)?;
        if encoded(cx, p)? {
            bits.insert(c);
        }
    }
    cx.families.access.cell_bits = bits;
    let mut words = std::collections::BTreeSet::new();
    for (&l, &c) in &cells.words {
        m.tick(1)?;
        if encoded(cx, Value::Inst(c))? {
            words.insert(l);
        }
    }
    cx.families.access.cell_words = words;
    if cells.accesses.is_empty() {
        return Ok(());
    }
    let entry = tree_bytes(1, std::mem::size_of::<(InstId, Reads)>());
    let mut reads: BTreeMap<InstId, Reads> = BTreeMap::new();
    let mut charged = 0usize;
    let mut plan = || -> Result<(), Error> {
        for &at in &cells.accesses {
            m.tick(1)?;
            let Some(inst) = cx.body.insts.get(at) else {
                continue;
            };
            if !matches!(inst.kind, InstKind::Load { .. }) {
                continue;
            }
            // A pointer word reads its `inttoptr`'s pointer.
            let read = match cells.words.get(&at) {
                Some(c) => Value::Inst(*c),
                None if is_ptr_ty(cx, inst.ty) => Value::Inst(at),
                None => continue,
            };
            let Some(group) = cells.group(at) else {
                return Err(cx.invariant("a cell load without a slot group"));
            };
            if !reads.contains_key(&group) {
                m.charge(entry)?;
                charged += entry;
            }
            reads.entry(group).or_default().add(encoding(cx, read))?;
        }
        Ok(())
    };
    let planned = plan();
    m.release(charged);
    planned?;
    cx.families.access.cell_reads = reads;
    Ok(())
}

/// The encoding and value store `at` writes: its pointer's own (`own`) when every load of its
/// slot group reads that, a blob pointer as `(BLOB_ROOT, offset)` when they all read a fat
/// pointer, and zeros whatever they read.
fn agree(
    cx: &SelectCx<'_>,
    at: InstId,
    val: Value,
    own: (Encoding, CellValue),
) -> Result<(Encoding, CellValue), Error> {
    let group = cx
        .memory
        .cells
        .group(at)
        .ok_or_else(|| cx.invariant("a cell store without a slot group"))?;
    let Some(reads) = cx.families.access.cell_reads.get(&group).copied() else {
        // No load reads the slot.
        return Ok(own);
    };
    let (enc, value) = own;
    if value == CellValue::Zero || reads.only(enc) {
        return Ok(own);
    }
    if enc == Encoding::Word && reads.only(Encoding::Pair) && blob::is_blob(cx, val)? {
        return Ok((Encoding::Pair, CellValue::Blob(blob::blob_ptr(cx, val)?)));
    }
    Err(refuse())
}

/// Select the cell strategy of `inst`: a plain load or store of a pointer that provenance
/// forwards through its cells (`MemoryPlan::cells`), unless the pointer is wide. `None` for any
/// other instruction.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
) -> Result<Option<CellStrategy>, Error> {
    if cx.families.access.cell_bits.contains(&at) {
        return Ok(Some(CellStrategy::Bits));
    }
    if let InstKind::Cast {
        op: CastOp::IntToPtr,
        v: Value::Inst(l),
    } = &inst.kind
    {
        if cx.families.access.cell_words.contains(l) {
            return Ok(Some(CellStrategy::Word(operand(cx, Value::Inst(*l))?)));
        }
    }
    if !cx.memory.cells.accesses.contains(&at) {
        return Ok(None);
    }
    // A pointer word whose pointer takes the integer path (a wide pointer's guest address) is
    // loaded as data.
    let word = cx.families.access.cell_words.contains(&at);
    if cx.memory.cells.words.contains_key(&at) && !word {
        return Ok(None);
    }
    Ok(Some(match &inst.kind {
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if is_ptr_ty(cx, inst.ty) || word => {
            // A pointer word holds its `inttoptr`'s pointer.
            let read = cx
                .memory
                .cells
                .words
                .get(&at)
                .map_or(Value::Inst(at), |c| Value::Inst(*c));
            let Some(enc) = encoding(cx, read)? else {
                return Ok(None);
            };
            CellStrategy::Load(CellAccess {
                access: slot(cx, *ptr, enc, *align)?,
                value: None,
            })
        }
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } => {
            // A store of pointer bits stores the pointer, unless the bits take the integer path (a
            // wide pointer's guest address): then it stores them as data.
            let val = match val {
                Value::Inst(c) if cx.memory.cells.bits.contains_key(c) => {
                    if !cx.families.access.cell_bits.contains(c) {
                        return Ok(None);
                    }
                    cx.memory.cells.bits_of(*val).unwrap_or(*val)
                }
                v => *v,
            };
            let own = match val {
                Value::Const(_) => constant(cx, val)?,
                v => match encoding(cx, v)? {
                    Some(enc) => (enc, CellValue::Value(operand(cx, v)?)),
                    None => return Ok(None),
                },
            };
            let (enc, value) = agree(cx, at, val, own)?;
            CellStrategy::Store(CellAccess {
                access: slot(cx, *ptr, enc, *align)?,
                value: Some(value),
            })
        }
        // An atomic cell access (a volatile one is selected as the plain one).
        _ => return Err(refuse()),
    }))
}

/// Emit cell strategy `s`, defining value `vid` for a load.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &CellStrategy,
) -> BResult<()> {
    match s {
        CellStrategy::Load(a) => {
            let Some(v) = local::emit_access(cx, fb, &a.access, None)? else {
                return Err(cx.fail("a cell load without a value"));
            };
            cx.define(vid, v);
        }
        CellStrategy::Store(a) => {
            let v = match a.value {
                Some(CellValue::Value(x)) => cx.operand(fb, x)?,
                Some(CellValue::Offset(o)) => k32(fb, o)?,
                Some(CellValue::Blob(p)) => side(cx, fb, PtrSide::Blob(p))?,
                Some(CellValue::Zero) => leaf::zero(fb, &a.access.leaf)?,
                None => return Err(cx.fail("a cell store without a value")),
            };
            local::emit_access(cx, fb, &a.access, Some(v))?;
        }
        CellStrategy::Bits => {}
        CellStrategy::Word(w) => {
            let x = cx.operand(fb, *w)?;
            cx.define(vid, x);
        }
    }
    Ok(())
}

/// The `inttoptr` whose pointer the pointer word `v` holds, when `v` is one that moves as its
/// pointer (`AccessPlan::cell_words`): `select` holds the word as that pointer's lowered
/// representation.
pub fn word_pointer(cx: &SelectCx<'_>, v: Value) -> Option<Value> {
    match v {
        Value::Inst(l) if cx.families.access.cell_words.contains(&l) => {
            cx.memory.cells.words.get(&l).map(|c| Value::Inst(*c))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{InternalError, InternalKind};

    /// A load whose representation no cell holds (an `Unsupported` refusal) reads no cell
    /// encoding; an internal invariant failure is the planning's own and propagates (it is not
    /// folded into a refusal of the group's stores).
    #[test]
    fn a_refused_load_reads_other_and_an_internal_error_propagates() {
        let mut r = Reads::default();
        assert!(r.add(Ok(Some(Encoding::Word))).is_ok());
        assert!(r.only(Encoding::Word));
        assert!(r.add(Err(refuse())).is_ok());
        assert!(
            !r.only(Encoding::Word),
            "a refused load reads no cell encoding"
        );
        let internal = Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "a cell pointer without a representation",
        ));
        let mut fresh = Reads::default();
        assert!(matches!(fresh.add(Err(internal)), Err(Error::Internal(_))));
        assert_eq!(fresh, Reads::default(), "a failed add records nothing");
    }
}
