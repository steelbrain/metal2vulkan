//! Private arrays allocated as far as their accesses reach.
//!
//! A thread's array that an index reaches at run time is private memory a driver backs with
//! scratch, held to the device's Function storage at its allocation size. A kernel written for
//! its largest configuration declares its arrays at that size and indexes them by counters whose
//! trip counts its function constants decide: a `float acc[32][32][4]` whose loops run 2, 4 and
//! 4 times reaches 32 of its 4096 elements. Each such alloca is allocated as the nest of arrays
//! its accesses reach: each level of it as long as the largest index an access takes there, plus
//! one. An alloca qualifies when:
//!
//! - it allocates one object (the constant count 1) of an array type, whose arrays nested from the
//!   outside in, down to the first element that is not an array, are its **levels**;
//! - each use of it is the base of a `getelementptr` of its own type whose first index is the
//!   constant 0 (an **address**), or the address of an access at its own address; each use of an
//!   address is the address of an access. An **access** is a plain load, or a plain store of
//!   another value, of the type its address names, or of the type at that type's offset 0 (its
//!   element 0 so many levels down, as where trailing zero indices were left out), which reaches
//!   element 0 of those levels;
//! - every index of every address after the first is **bounded** within the length of the array
//!   it indexes (a struct's member index is a constant): a constant; a counter of a loop whose
//!   header runs the bound constants decide, followed from its preheader's constants run by run as
//!   unrolling follows them, within [`RUNS`] runs (`unroll::counter_bounds`), over every run its
//!   header makes; a PHI elsewhere of bounded values; or `zext`, `sext`, `trunc`, `add`, `sub`,
//!   `mul`, `shl`, `lshr`, `ashr`, `udiv`, `sdiv`, `urem`, `srem`, `and`, `or`, `xor` and `select`
//!   of bounded values, where each bound stays among the non-negative values of its type, so no
//!   step wraps.
//!
//! An access of fewer indices than the levels (counting the zeros of the levels it passes down)
//! reaches every element of the levels below, so those keep their length. Every access keeps its indices and its element; only
//! the elements no access can reach are gone, so the program computes what it computed. An alloca
//! that does not qualify keeps its type: an index with no bound may reach any element of it, and
//! a use of another kind (a pointer stored, passed, merged or compared, a byte offset, a copy)
//! reads its bytes by an offset of the declared layout.

use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, LoopForest, LoopId};
use crate::ir::{Arena, BinOp, Body, CastOp, Const, FuncId, Global, GlobalId, Idx, InstId};
use crate::ir::{InstKind, ModuleParts, Type, TypeId, TypeTable, Value};

use super::fold::Folder;
use super::rewrite::{filled, invariant, scoped};
use super::unroll::{self, Cx};

/// The most header runs a loop's counter is followed for.
pub(super) const RUNS: u32 = 1024;

/// The deepest index expression bounded.
const DEPTH: u32 = 32;

/// No entry in the role table.
const NONE: u32 = u32::MAX;

/// The role table's mark of an address (the alloca's index beside it).
const ADDRESS: u32 = 1 << 31;

/// One candidate alloca.
struct Array {
    inst: InstId,
    ty: TypeId,
    /// Each level's length, outermost first.
    lens: Vec<u64>,
    /// The type below the last level.
    elem: TypeId,
    /// The length each level's accesses reach.
    reach: Vec<u64>,
    /// Its addresses.
    addresses: Vec<InstId>,
    /// The fewest levels a load or store at the alloca's own address passes through to the type
    /// it accesses (its element 0 at that depth); `None` when none does.
    direct: Option<usize>,
    qualifies: bool,
}

/// The depth at which `x` is the type at offset 0 of `ty`: `ty` itself at 0, then each array's
/// element and each struct's first member in turn; `None` when it is none of them.
fn zero_depth(types: &TypeTable, ty: TypeId, x: TypeId) -> Option<usize> {
    let mut t = ty;
    for d in 0.. {
        if t == x {
            return Some(d);
        }
        t = match types.get(t)? {
            Type::Array { elem, .. } => *elem,
            Type::Struct { fields, .. } => *fields.first()?,
            _ => return None,
        };
    }
    None
}

/// `body` with each alloca that qualifies allocated as far as its accesses reach (module docs).
/// `func` is the entry whose body it is.
pub(super) fn narrow(
    mut body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<Body, Error> {
    let edits = scoped(m, |m, held| plan(&body, parts, func, m, held))?;
    for &(i, new) in &edits {
        m.tick(1)?;
        match body.insts.get_mut(i).map(|inst| &mut inst.kind) {
            Some(InstKind::Alloca { ty, .. }) => *ty = new,
            Some(InstKind::Gep { src_ty, .. }) => *src_ty = new,
            _ => return Err(invariant("a narrowed instruction is no alloca or address")),
        }
    }
    m.release(edits.capacity() * std::mem::size_of::<(InstId, TypeId)>());
    Ok(body)
}

/// The levels of `ty`, outermost first, and the type below them; `None` when `ty` is no array.
fn levels(types: &TypeTable, mut ty: TypeId) -> Option<(Vec<u64>, TypeId)> {
    let mut lens = Vec::new();
    while let Some(Type::Array { elem, len }) = types.get(ty) {
        lens.push(*len);
        ty = *elem;
    }
    (!lens.is_empty()).then_some((lens, ty))
}

/// Whether `v` is the integer constant `x`.
fn is_const(parts: &ModuleParts, v: Value, x: u64) -> bool {
    let Value::Const(c) = v else {
        return false;
    };
    let Some(k) = parts.consts.get(c) else {
        return false;
    };
    matches!(parts.types.get(k.ty), Some(Type::Int(_))) && k.value == Const::Int(x)
}

/// The type an address of `src_ty` with `indices` names: arrays' elements and structs' members
/// (whose indices are constants) after the first index; `None` through anything else.
fn named(parts: &ModuleParts, src_ty: TypeId, indices: &[Value]) -> Option<TypeId> {
    let mut t = src_ty;
    for &i in indices.get(1..)? {
        t = match parts.types.get(t)? {
            Type::Array { elem, .. } => *elem,
            Type::Struct { fields, .. } => {
                let Value::Const(c) = i else {
                    return None;
                };
                let Const::Int(f) = parts.consts.get(c)?.value else {
                    return None;
                };
                *fields.get(usize::try_from(f).ok()?)?
            }
            _ => return None,
        };
    }
    Some(t)
}

/// The type of `v` in the entry `func`'s `body`.
fn value_ty(body: &Body, parts: &ModuleParts, func: FuncId, v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Const(c) => parts.consts.ty(c),
        Value::Arg(a) => {
            let f = parts.funcs.get(func)?;
            match parts.types.get(f.ty)? {
                Type::Function { params, .. } => params.get(a.index()).copied(),
                _ => None,
            }
        }
    }
}

/// The edits of `body`: each alloca that qualifies and its addresses, with the type it narrows to.
fn plan(
    body: &Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Vec<(InstId, TypeId)>, Error> {
    let n = body.insts.len();
    let mut role = filled(m, held, n, NONE)?;
    let mut arrays: Vec<Array> = Vec::new();
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Alloca { ty, count, .. } = &inst.kind else {
            continue;
        };
        if !is_const(parts, *count, 1) {
            continue;
        }
        let Some((lens, elem)) = levels(&parts.types, *ty) else {
            continue;
        };
        let k = u32::try_from(arrays.len()).map_err(|_| invariant("allocas"))?;
        if let Some(slot) = role.get_mut(i.index()) {
            *slot = k;
        }
        let entry = 2 * lens.len() * std::mem::size_of::<u64>() + std::mem::size_of::<Array>();
        m.charge(entry)?;
        *held = held.saturating_add(entry);
        arrays.push(Array {
            inst: i,
            ty: *ty,
            reach: vec![0; lens.len()],
            lens,
            elem,
            addresses: Vec::new(),
            direct: None,
            qualifies: true,
        });
    }
    if arrays.is_empty() {
        return Ok(Vec::new());
    }
    // The addresses.
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Gep {
            src_ty,
            base: Value::Inst(a),
            indices,
            ..
        } = &inst.kind
        else {
            continue;
        };
        let k = role.get(a.index()).copied().unwrap_or(NONE);
        let Some(array) = arrays.get_mut(k as usize) else {
            continue;
        };
        let address = *src_ty == array.ty
            && indices.first().is_some_and(|&z| is_const(parts, z, 0))
            && !indices.contains(&Value::Inst(*a));
        if address {
            if let Some(slot) = role.get_mut(i.index()) {
                *slot = k | ADDRESS;
            }
            m.grow(&mut array.addresses, 1)?;
            *held = held.saturating_add(std::mem::size_of::<InstId>());
            array.addresses.push(i);
        }
    }
    // Every use: an address's base, or an access through an address of the type it names or of
    // the type at its offset 0 (`zero`, per address: the fewest levels down).
    let mut zero = filled(m, held, n, NONE)?;
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let mine = role.get(i.index()).copied().unwrap_or(NONE);
        inst.kind.for_each_value(|v| {
            let Value::Inst(j) = v else {
                return;
            };
            let r = role.get(j.index()).copied().unwrap_or(NONE);
            if r == NONE {
                return;
            }
            let k = r & !ADDRESS;
            let Some(array) = arrays.get_mut(k as usize) else {
                return;
            };
            // The type a plain load or store accesses through `v`.
            let accessed = match &inst.kind {
                InstKind::Load {
                    ptr,
                    volatile: false,
                    atomic: None,
                    ..
                } if *ptr == v => Some(inst.ty),
                InstKind::Store {
                    val,
                    ptr,
                    volatile: false,
                    atomic: None,
                    ..
                } if *ptr == v && *val != v => value_ty(body, parts, func, *val),
                _ => None,
            };
            let allowed = if r & ADDRESS == 0 {
                let address = mine == r | ADDRESS
                    && matches!(&inst.kind, InstKind::Gep { base, .. } if *base == v);
                let direct = accessed.and_then(|x| zero_depth(&parts.types, array.ty, x));
                if let Some(d) = direct {
                    array.direct = Some(array.direct.map_or(d, |e| e.min(d)));
                }
                address || direct.is_some()
            } else {
                let at = match body.insts.get(j).map(|a| &a.kind) {
                    Some(InstKind::Gep {
                        src_ty, indices, ..
                    }) => named(parts, *src_ty, indices),
                    _ => None,
                };
                // The access's type at offset 0 of the type the address names: its element 0
                // that many levels down.
                let z = at
                    .zip(accessed)
                    .and_then(|(t, x)| zero_depth(&parts.types, t, x));
                if let (Some(z), Some(slot)) = (z, zero.get_mut(j.index())) {
                    *slot = (*slot).min(u32::try_from(z).unwrap_or(NONE));
                }
                z.is_some()
            };
            if !allowed {
                array.qualifies = false;
            }
        });
    }
    if !arrays
        .iter()
        .any(|a| a.qualifies && !a.addresses.is_empty())
    {
        return Ok(Vec::new());
    }
    // The bounds of the addresses' indices, and the reach of each level.
    let cfg = Cfg::build(body, m)?;
    let reached = LoopForest::compute(&cfg, m).and_then(|forest| {
        let r = (|| {
            let cx = Cx::new(body, (&cfg, &forest), m, held)?;
            let (types, consts, globals) = (&parts.types, &mut parts.consts, &parts.globals);
            let mut bounds = Bounds {
                cx: &cx,
                folder: Folder {
                    types,
                    consts,
                    held: 0,
                },
                globals,
                state: filled(m, held, n, Range::Unseen)?,
                counted: filled(m, held, forest.len(), false)?,
                held: 0,
            };
            let r = reach(body, (&mut arrays, &zero), &mut bounds, m);
            *held = held.saturating_add(bounds.held);
            r
        })();
        m.release(forest.ledger_bytes());
        r
    });
    m.release(cfg.ledger_bytes());
    reached?;
    // The narrowed types.
    let mut edits: Vec<(InstId, TypeId)> = Vec::new();
    for array in arrays
        .iter()
        .filter(|a| a.qualifies && !a.addresses.is_empty())
    {
        m.tick(1)?;
        if array.reach == array.lens {
            continue;
        }
        let mut t = array.elem;
        for &len in array.reach.iter().rev() {
            t = parts.types.intern(Type::Array { elem: t, len }, m)?;
        }
        m.grow(&mut edits, 1 + array.addresses.len())?;
        edits.push((array.inst, t));
        edits.extend(array.addresses.iter().map(|&g| (g, t)));
    }
    Ok(edits)
}

/// Each level's reach over `arrays`' addresses, each accessed `zero` levels below the type it
/// names at the fewest (`NONE` for an address nothing accesses); an array an index of which has no
/// bound within its array no longer qualifies.
fn reach(
    body: &Body,
    (arrays, zero): (&mut [Array], &[u32]),
    bounds: &mut Bounds<'_, '_>,
    m: &mut Meter,
) -> Result<(), Error> {
    for array in arrays.iter_mut().filter(|a| a.qualifies) {
        for &g in &array.addresses {
            m.tick(1)?;
            let Some(InstKind::Gep { indices, .. }) = body.insts.get(g).map(|i| &i.kind) else {
                return Err(invariant("an address is no getelementptr"));
            };
            let depth = array.lens.len();
            let types = bounds.folder.types;
            let mut t = array.ty;
            for (level, &idx) in indices.iter().skip(1).enumerate() {
                let bound = bounds.of(idx, 0, m)?;
                // The type the index steps into, when its bound lies within it: an array's
                // element, or a struct's member at a constant index.
                let into = match (types.get(t), bound) {
                    (Some(Type::Array { elem, len }), Some((_, hi))) if hi < *len => {
                        if let Some(r) = array.reach.get_mut(level) {
                            *r = (*r).max(hi + 1);
                        }
                        Some(*elem)
                    }
                    (Some(Type::Struct { fields, .. }), Some((f, g))) if f == g => {
                        usize::try_from(f).ok().and_then(|f| fields.get(f)).copied()
                    }
                    _ => None,
                };
                let Some(into) = into else {
                    array.qualifies = false;
                    break;
                };
                t = into;
            }
            if !array.qualifies {
                break;
            }
            // An access element 0 of the levels below the ones the address indexes, and the
            // levels below those whole.
            let indexed = indices.len().saturating_sub(1).min(depth);
            let below = match zero.get(g.index()).copied() {
                Some(z) if z != NONE => usize::try_from(z).unwrap_or(usize::MAX),
                _ => 0,
            };
            let whole = indexed.saturating_add(below);
            for (level, (r, &len)) in array.reach.iter_mut().zip(&array.lens).enumerate() {
                if level >= whole {
                    *r = len;
                } else if level >= indexed {
                    *r = (*r).max(1);
                }
            }
        }
        // An access at the alloca's own address reaches element 0 of the levels it passes
        // through, and the levels below them whole.
        if let Some(d) = array.direct {
            for (level, (r, &len)) in array.reach.iter_mut().zip(&array.lens).enumerate() {
                *r = if level < d { (*r).max(1) } else { len };
            }
        }
    }
    Ok(())
}

/// What is known of an instruction's value ([`Bounds::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Range {
    Unseen,
    /// Being bounded: a cycle back to it has no bound.
    Open,
    Known(u64, u64),
    Unknown,
}

/// The bounds of integer values (module docs).
struct Bounds<'a, 'f> {
    cx: &'a Cx<'a>,
    folder: Folder<'f>,
    globals: &'f Arena<GlobalId, Global>,
    state: Vec<Range>,
    /// Per loop, whether its counters were followed.
    counted: Vec<bool>,
    /// Ledger bytes of the counters' lists.
    held: usize,
}

/// Whether `x` is a non-negative value of a `bits`-wide integer (1 to 64 bits).
fn fits(x: u64, bits: u8) -> bool {
    (1..=64).contains(&bits) && x < 1u64 << (bits - 1)
}

impl Bounds<'_, '_> {
    /// The least and the greatest value `v` holds, when bounded (module docs).
    fn of(&mut self, v: Value, depth: u32, m: &mut Meter) -> Result<Option<(u64, u64)>, Error> {
        let i = match v {
            Value::Arg(_) => return Ok(None),
            Value::Const(c) => {
                let bits = match self.folder.consts.ty(c).map(|t| self.folder.types.get(t)) {
                    Some(Some(Type::Int(b))) => *b,
                    _ => return Ok(None),
                };
                return Ok(self
                    .folder
                    .int(c)
                    .filter(|&x| fits(x, bits))
                    .map(|x| (x, x)));
            }
            Value::Inst(i) => i,
        };
        match self.state.get(i.index()).copied() {
            Some(Range::Known(lo, hi)) => return Ok(Some((lo, hi))),
            Some(Range::Unseen) => {}
            _ => return Ok(None),
        }
        m.tick(1)?;
        if depth > DEPTH {
            self.set(i, Range::Unknown);
            return Ok(None);
        }
        self.set(i, Range::Open);
        let r = self.compute(i, depth, m)?;
        if self.state.get(i.index()) == Some(&Range::Open) {
            self.set(i, r.map_or(Range::Unknown, |(lo, hi)| Range::Known(lo, hi)));
        }
        Ok(r)
    }

    fn set(&mut self, i: InstId, r: Range) {
        if let Some(slot) = self.state.get_mut(i.index()) {
            *slot = r;
        }
    }

    fn compute(
        &mut self,
        i: InstId,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<(u64, u64)>, Error> {
        let body = self.cx.body();
        let inst = body
            .insts
            .get(i)
            .ok_or_else(|| invariant("instruction out of range"))?;
        let bits = match self.folder.types.get(inst.ty) {
            Some(Type::Int(b)) => *b,
            _ => return Ok(None),
        };
        let d = depth + 1;
        let r = match &inst.kind {
            InstKind::Phi { incoming, .. } => {
                let block = self.cx.block_of(i)?;
                if let Some(l) = self.cx.forest().loop_of_header(block) {
                    return self.counter(i, l, m);
                }
                let mut joined: Option<(u64, u64)> = None;
                for &(v, _) in incoming.iter() {
                    let Some((lo, hi)) = self.of(v, d, m)? else {
                        return Ok(None);
                    };
                    joined = Some(joined.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
                }
                joined
            }
            InstKind::Cast {
                op: CastOp::ZExt | CastOp::SExt | CastOp::Trunc,
                v,
            } => self.of(*v, d, m)?,
            InstKind::Select { t, f, .. } => match (self.of(*t, d, m)?, self.of(*f, d, m)?) {
                (Some((a, b)), Some((c, e))) => Some((a.min(c), b.max(e))),
                _ => None,
            },
            InstKind::BinOp { op, a, b, .. } => {
                let (ra, rb) = (self.of(*a, d, m)?, self.of(*b, d, m)?);
                // Bounded by one operand alone: `and` by its bounded operand, and an unsigned
                // remainder by its divisor, whatever the other holds.
                match (op, ra, rb) {
                    (BinOp::And, Some((_, x)), None) | (BinOp::And, None, Some((_, x))) => {
                        return Ok(Some((0, x)));
                    }
                    (BinOp::URem, None, Some((b0, b1))) if b0 >= 1 => {
                        return Ok(Some((0, b1 - 1)));
                    }
                    _ => {}
                }
                let (Some((a0, a1)), Some((b0, b1))) = (ra, rb) else {
                    return Ok(None);
                };
                // The greatest value of the bits of `x`.
                let ones = |x: u64| u64::MAX.checked_shr(x.leading_zeros()).unwrap_or(0);
                match op {
                    BinOp::Add => a0.checked_add(b0).zip(a1.checked_add(b1)),
                    BinOp::Sub => a0.checked_sub(b1).zip(a1.checked_sub(b0)),
                    BinOp::Mul => a0.checked_mul(b0).zip(a1.checked_mul(b1)),
                    BinOp::Shl if b0 == b1 && b0 < u64::from(bits) => {
                        let s = u32::try_from(b0).unwrap_or(u32::MAX);
                        a0.checked_shl(s)
                            .zip(a1.checked_shl(s))
                            .filter(|&(_, hi)| hi >> s == a1)
                    }
                    BinOp::LShr | BinOp::AShr if b1 < u64::from(bits) => Some((a0 >> b1, a1 >> b0)),
                    BinOp::UDiv | BinOp::SDiv if b0 >= 1 => Some((a0 / b1, a1 / b0)),
                    BinOp::URem | BinOp::SRem if b0 >= 1 => {
                        if a1 < b0 {
                            Some((a0, a1))
                        } else {
                            Some((0, a1.min(b1 - 1)))
                        }
                    }
                    BinOp::And => Some((0, a1.min(b1))),
                    BinOp::Or => Some((a0.max(b0), ones(a1.max(b1)))),
                    BinOp::Xor => Some((0, ones(a1.max(b1)))),
                    _ => None,
                }
            }
            _ => None,
        };
        Ok(r.filter(|&(lo, hi)| lo <= hi && fits(hi, bits)))
    }

    /// The bound of header PHI `i` of loop `l`: its loop's counters followed, once per loop.
    fn counter(
        &mut self,
        i: InstId,
        l: LoopId,
        m: &mut Meter,
    ) -> Result<Option<(u64, u64)>, Error> {
        let followed = self.counted.get(l.index()).copied().unwrap_or(true);
        if !followed {
            if let Some(slot) = self.counted.get_mut(l.index()) {
                *slot = true;
            }
            let mut held = 0;
            let bounds = unroll::counter_bounds(
                self.cx,
                l,
                (&mut self.folder, self.globals),
                RUNS,
                m,
                &mut held,
            )?;
            self.held = self.held.saturating_add(held);
            for (p, b) in bounds.into_iter().flatten() {
                let bits = self.cx.body().insts.get(p).and_then(|inst| {
                    match self.folder.types.get(inst.ty) {
                        Some(Type::Int(b)) => Some(*b),
                        _ => None,
                    }
                });
                let r = match (b, bits) {
                    (Some((lo, hi)), Some(bits)) if lo <= hi && fits(hi, bits) => {
                        Range::Known(lo, hi)
                    }
                    _ => Range::Unknown,
                };
                self.set(p, r);
            }
        }
        Ok(match self.state.get(i.index()).copied() {
            Some(Range::Known(lo, hi)) => Some((lo, hi)),
            _ => None,
        })
    }
}
