//! Thread-memory fields one store writes, read as the value it stored.
//!
//! A C++ lambda that captures by reference, or a struct passed to a function by pointer, leaves
//! pointers in thread memory: a closure is an `alloca` of pointers to the captured locals, each
//! stored once where the closure is made and loaded back in its call operator's body. Once
//! inlining has made the entry one body, each such load reads a field of an `alloca` that one
//! store wrote, earlier on every path. Lowering keeps an object whose address is stored, or that
//! holds pointers, in the private arena, and reaches memory through a pointer loaded back from it
//! at an offset only the run knows, which a driver backs with scratch memory. This pass reads each
//! such load as the value its store stored, as LLVM's single-store promotion does, so an access
//! through it is an access at a constant offset of the object it points into, and removes an
//! object no read is left of, with its stores.
//!
//! An `alloca` is an **object** here when it allocates a constant count of a sized type, at most
//! [`MAX_BYTES`], and every use of its address, and of every address derived from it by
//! `getelementptr`s at constant indices and by bitcasts, is one of those derivations, the address
//! of a plain (non-volatile, non-atomic) load or store, or the destination or source of a plain
//! `llvm.memset`, `llvm.memcpy` or `llvm.memmove`. Its address is never stored, passed to another
//! call, merged by a `phi` or `select`, compared or made an integer, so no other pointer reaches
//! its bytes: they are written by those stores and calls alone. A load of an object at a constant
//! offset is read as the value of a store when:
//!
//! - that store is the one write of the object (a store or a memory call) that covers any of the
//!   load's bytes, and it writes exactly them, a value of the load's type, which is no aggregate
//!   holding a pointer, and not the null pointer of a space other than device or constant; and
//! - the store dominates the load (it is earlier in the load's block, or its block dominates the
//!   load's).
//!
//! Every path to the load runs the store, and no other write reaches those bytes, so the load
//! reads what the store's latest run stored: the value of its operand at that run. The operand
//! dominates the store, and no run of the operand's definition comes between the store's latest
//! run and the load (one would make a path from the definition to the load that avoids the store,
//! and the definition dominates the store), so that value is the one SSA names at the load. A load
//! the rule does not cover (a field two writes reach, a part of a field, a field no store wrote, a
//! load before its store) stays as it is.
//!
//! The null pointer of a thread or threadgroup space has no representation as a value in lowering:
//! its null is an arena offset the memory plan sets aside only for a pointer it holds in memory. So
//! a load that would read such a null stays a load, and its object stays. An aggregate that holds
//! a pointer is never a value in lowering either: each `extractvalue` of one is the pointer or value
//! its insertion chain inserted, or a member loaded from memory, and such a null is among the
//! members an insertion chain can carry. So a load of such an aggregate stays too.
//!
//! A load read as the bits of a pointer (`ptrtoint p` of an integer as wide as `p`) that an
//! `inttoptr` turns back into a pointer of `p`'s own type is the round trip through memory the
//! memory plan reads as the pointer the slot holds; the `inttoptr` is read as `p` itself, so the
//! round trip stays the pointer once its memory goes.
//!
//! An object that no load and no memory call reads after that is read by nothing: its stores, the
//! memory calls that write it, its derived addresses and the `alloca` go. A stored address that
//! was its own object's only use outside its accesses no longer escapes, so that object is an
//! object in the next round: the closure's fields are read as the captured locals' addresses in
//! one round, the closure goes, and the locals are plain objects of their own, which the memory
//! plan holds typed, at the offsets the accesses name. At most [`ROUNDS`] rounds run, each over
//! the body the last rebuilt.

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::{Cfg, DomTree};
use crate::intrinsics::LlvmIntrinsic;
use crate::ir::{AddrSpace, Const, Type, TypeId, TypeLayouts, Value};
use crate::ir::{BlockId, Body, Callee, CastOp, Idx, Inst, InstId, InstKind, Layout, ModuleParts};

use super::pieces::{gep_offset, int};
use super::rewrite::{apply, filled, scoped, BodyEdit};

/// The most rounds: a round removes the objects whose stores held the next round's addresses.
pub(super) const ROUNDS: usize = 4;

/// The largest object whose bytes are followed.
pub(super) const MAX_BYTES: u64 = 1 << 16;

/// No write covers the byte.
const NONE: u32 = u32::MAX;

/// More than one write covers the byte.
const MANY: u32 = u32::MAX - 1;

/// Where an address points: into which object, and at which byte offset (`None`: an offset only
/// the run knows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    object: u32,
    at: Option<u64>,
}

/// One write of an object.
#[derive(Clone, Copy, Debug)]
struct Write {
    at: u64,
    len: u64,
    /// The store, with its value; `None` for a memory call.
    store: Option<(InstId, Value)>,
}

/// One `alloca` and what reaches it.
#[derive(Debug)]
struct Object {
    inst: InstId,
    /// Its bytes; `None` when it is no object (module docs).
    bytes: Option<u64>,
    /// Its address escapes, or an access of it is at an offset only the run knows.
    out: bool,
    /// A memory call reads it.
    copied: bool,
    writes: Vec<Write>,
    /// Its loads: the load, its offset and its type.
    loads: Vec<(InstId, u64, TypeId)>,
    /// Its derived addresses, its stores and the memory calls that write it.
    owned: Vec<InstId>,
}

/// `body` with every load of thread memory that one store wrote read as its value, and every
/// object nothing reads removed (module docs); `func`'s parameters type its arguments. Says
/// whether anything changed; returns `body` itself when nothing did.
pub(super) fn forward(
    body: Body,
    parts: &ModuleParts,
    params: &[TypeId],
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let mut body = body;
    let mut changed = false;
    for _ in 0..ROUNDS {
        let Some(next) = round(&body, parts, params, m)? else {
            break;
        };
        body.release(m);
        body = next;
        changed = true;
    }
    Ok((body, changed))
}

/// What a round reads: the body, the module's tables, the type layouts, and the entry's parameter
/// types.
#[derive(Clone, Copy)]
struct Cx<'a> {
    body: &'a Body,
    parts: &'a ModuleParts,
    layouts: &'a TypeLayouts,
    params: &'a [TypeId],
}

impl Cx<'_> {
    /// The type of `v`.
    fn ty_of(&self, v: Value) -> Option<TypeId> {
        match v {
            Value::Inst(i) => self.body.insts.get(i).map(|n| n.ty),
            Value::Arg(n) => self.params.get(n.index()).copied(),
            Value::Const(k) => self.parts.consts.ty(k),
        }
    }
}

/// One round over `body`: the rebuilt body, or `None` when the round changes nothing.
fn round(
    body: &Body,
    parts: &ModuleParts,
    params: &[TypeId],
    m: &mut Meter,
) -> Result<Option<Body>, Error> {
    if !body
        .insts
        .iter()
        .any(|(_, i)| matches!(i.kind, InstKind::Alloca { .. }))
    {
        return Ok(None);
    }
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let layout_bytes = parts.types.len() * std::mem::size_of::<Option<Layout>>();
    let cx = Cx {
        body,
        parts,
        layouts: &layouts,
        params,
    };
    let r = scoped(m, |m, held| plan(cx, m, held));
    m.release(layout_bytes);
    r
}

/// Push `x` onto `v`, its growth charged to `m` and counted in `held` (`Meter::grow`).
fn push<T>(v: &mut Vec<T>, x: T, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
    let added = if v.len() < v.capacity() {
        0
    } else {
        (v.len() + 1).max(v.capacity() * 2).max(4) - v.capacity()
    };
    m.grow(v, 1)?;
    *held += added * std::mem::size_of::<T>();
    v.push(x);
    Ok(())
}

/// Find the objects of `body`, read its loads, remove what nothing reads, and rebuild it.
fn plan(cx: Cx<'_>, m: &mut Meter, held: &mut usize) -> Result<Option<Body>, Error> {
    let Cx {
        body,
        parts,
        layouts,
        ..
    } = cx;
    let n = body.insts.len();
    // Each instruction's place: an object's `alloca`, or an address derived from one.
    let mut objects: Vec<Object> = Vec::new();
    let mut places: Vec<Option<Place>> = filled(m, held, n, None)?;
    let mut done: Vec<bool> = filled(m, held, n, false)?;
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        if let InstKind::Alloca { ty, count, .. } = inst.kind {
            let bytes = int(parts, count)
                .and_then(|c| layouts.alloc_size(ty)?.checked_mul(c))
                .filter(|&b| b <= MAX_BYTES);
            let object = u32::try_from(objects.len()).map_err(|_| past_index())?;
            let o = Object {
                inst: i,
                bytes,
                out: bytes.is_none(),
                copied: false,
                writes: Vec::new(),
                loads: Vec::new(),
                owned: Vec::new(),
            };
            push(&mut objects, o, m, held)?;
            if let Some(slot) = places.get_mut(i.index()) {
                *slot = Some(Place {
                    object,
                    at: Some(0),
                });
            }
            if let Some(d) = done.get_mut(i.index()) {
                *d = true;
            }
        }
    }
    if objects.is_empty() {
        return Ok(None);
    }
    let mut chain: Vec<InstId> = Vec::new();
    for (i, _) in body.insts.iter() {
        m.tick(1)?;
        place_of(cx, i, (&mut places, &mut done), (&mut chain, held), m)?;
    }

    // Each use of an object's address.
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        uses(cx, (i, inst), &places, &mut objects, m, held)?;
    }

    // The loads read as their store's value, and the objects nothing reads.
    let mut edit = BodyEdit::keep_all(body, m, held)?;
    let mut starts: Vec<InstId> = Vec::new();
    m.reserve(&mut starts, body.blocks.len())?;
    *held += body.blocks.len() * std::mem::size_of::<InstId>();
    starts.extend(body.blocks.iter().map(|(_, b)| b.start));
    let mut dom: Option<DomTree<BlockId>> = None;
    let mut changed = false;
    let r = (|| -> Result<(), Error> {
        for o in &objects {
            m.tick(1)?;
            changed |= decide(cx, o, &starts, &mut dom, &mut edit, m)?;
        }
        Ok(())
    })();
    if let Some(d) = dom {
        m.release(d.ledger_bytes());
    }
    r?;
    if !changed {
        return Ok(None);
    }
    round_trips(cx, &mut edit, m)?;
    apply(body, &edit, &|c| c, m).map(Some)
}

/// Read each `inttoptr` of a load `edit` reads as a pointer's bits, at the pointer's width and to
/// its type, as that pointer (module docs).
fn round_trips(cx: Cx<'_>, edit: &mut BodyEdit, m: &mut Meter) -> Result<(), Error> {
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let InstKind::Cast {
            op: CastOp::IntToPtr,
            v: Value::Inst(load),
        } = inst.kind
        else {
            continue;
        };
        // What the load is read as, through loads read as other loads' values.
        let mut at = load;
        let mut steps = 0usize;
        while let Some(Some(Value::Inst(next))) = edit.replace.get(at.index()).copied() {
            m.tick(1)?;
            at = next;
            steps += 1;
            if steps > cx.body.insts.len() {
                return Err(past_index());
            }
        }
        if at == load {
            continue;
        }
        let Some(InstKind::Cast {
            op: CastOp::PtrToInt,
            v: ptr,
        }) = cx.body.insts.get(at).map(|n| &n.kind)
        else {
            continue;
        };
        let bits = cx.body.insts.get(load).map(|n| n.ty);
        let wide = bits.and_then(|t| cx.layouts.store_size(t));
        if cx.ty_of(*ptr) == Some(inst.ty)
            && wide.is_some()
            && wide == cx.layouts.store_size(inst.ty)
        {
            edit.replace(i, *ptr);
        }
    }
    Ok(())
}

/// Record the uses instruction `inst` (`i`) makes of the objects' addresses (module docs).
fn uses(
    cx: Cx<'_>,
    (i, inst): (InstId, &Inst),
    places: &[Option<Place>],
    objects: &mut [Object],
    m: &mut Meter,
    held: &mut usize,
) -> Result<(), Error> {
    let place = |v: Value| match v {
        Value::Inst(i) => places.get(i.index()).copied().flatten(),
        _ => None,
    };
    let escape = |v: Value, objects: &mut [Object]| {
        if let Some(o) = place(v).and_then(|p| objects.get_mut(p.object as usize)) {
            o.out = true;
        }
    };
    let own = place(Value::Inst(i));
    match &inst.kind {
        // A derivation: its own uses are its object's.
        InstKind::Gep { base, .. } if own.is_some() => {
            if let Some(o) = place(*base).and_then(|p| objects.get_mut(p.object as usize)) {
                push(&mut o.owned, i, m, held)?;
            }
        }
        InstKind::Cast {
            op: CastOp::Bitcast,
            v,
        } if own.is_some() => {
            if let Some(o) = place(*v).and_then(|p| objects.get_mut(p.object as usize)) {
                push(&mut o.owned, i, m, held)?;
            }
        }
        InstKind::Load {
            ptr,
            volatile: false,
            atomic: None,
            ..
        } => {
            if let Some(p) = place(*ptr) {
                if let Some(o) = objects.get_mut(p.object as usize) {
                    match p.at {
                        Some(at) => push(&mut o.loads, (i, at, inst.ty), m, held)?,
                        None => o.out = true,
                    }
                }
            }
        }
        InstKind::Store {
            val,
            ptr,
            volatile: false,
            atomic: None,
            ..
        } => {
            escape(*val, objects);
            if let Some(p) = place(*ptr) {
                let len = cx.ty_of(*val).and_then(|t| cx.layouts.store_size(t));
                if let Some(o) = objects.get_mut(p.object as usize) {
                    match (p.at, len) {
                        (Some(at), Some(len)) => {
                            let w = Write {
                                at,
                                len,
                                store: Some((i, *val)),
                            };
                            push(&mut o.writes, w, m, held)?;
                            push(&mut o.owned, i, m, held)?;
                        }
                        _ => o.out = true,
                    }
                }
            }
        }
        InstKind::Call(c) => match (memory_call(&c.callee), &*c.args) {
            (Some(copies), &[dst, src, len, ..]) => {
                for &v in c.args.iter().skip(2) {
                    escape(v, objects);
                }
                if let Some(p) = place(dst) {
                    if let Some(o) = objects.get_mut(p.object as usize) {
                        match p.at {
                            // A write of a length only the run knows covers every byte from its
                            // start.
                            Some(at) => {
                                let len = int(cx.parts, len)
                                    .unwrap_or_else(|| o.bytes.map_or(0, |b| b.saturating_sub(at)));
                                let w = Write {
                                    at,
                                    len,
                                    store: None,
                                };
                                push(&mut o.writes, w, m, held)?;
                                push(&mut o.owned, i, m, held)?;
                            }
                            None => o.out = true,
                        }
                    }
                }
                if copies {
                    if let Some(o) = place(src).and_then(|p| objects.get_mut(p.object as usize)) {
                        o.copied = true;
                    }
                } else {
                    escape(src, objects);
                }
            }
            _ => inst.kind.for_each_value(|v| escape(v, objects)),
        },
        // Every other use of an address lets it out.
        other => other.for_each_value(|v| escape(v, objects)),
    }
    Ok(())
}

/// Read each load of object `o` that one dominating store wrote as its value, and remove `o`
/// when no read of it is left (module docs). `starts` holds each block's first instruction; the
/// dominator tree is built the first time a store and its load are in different blocks. Says
/// whether `edit` changed.
fn decide(
    cx: Cx<'_>,
    o: &Object,
    starts: &[InstId],
    dom: &mut Option<DomTree<BlockId>>,
    edit: &mut BodyEdit,
    m: &mut Meter,
) -> Result<bool, Error> {
    let (false, Some(bytes)) = (o.out, o.bytes) else {
        return Ok(false);
    };
    let len = usize::try_from(bytes).map_err(|_| past_index())?;
    // The one write covering each byte: `NONE` for none, `MANY` for more than one.
    let mut owner: Vec<u32> = Vec::new();
    m.reserve(&mut owner, len)?;
    owner.resize(len, NONE);
    let r = (|| -> Result<bool, Error> {
        let mut read_left = o.copied;
        for (k, w) in o.writes.iter().enumerate() {
            m.tick(1 + u32::try_from(w.len >> 10).unwrap_or(u32::MAX))?;
            let k = u32::try_from(k).map_err(|_| past_index())?;
            let span =
                w.at.checked_add(w.len)
                    .filter(|&e| e <= bytes)
                    .and_then(|e| owner.get_mut(w.at as usize..e as usize));
            let Some(span) = span else {
                // A write past the object's end: its bytes are not followed.
                return Ok(false);
            };
            for b in span {
                *b = if *b == NONE { k } else { MANY };
            }
        }
        let block_of = |i: InstId| {
            let k = starts.partition_point(|s| s.raw() <= i.raw());
            BlockId::new(u32::try_from(k.saturating_sub(1)).unwrap_or(0))
        };
        let mut changed = false;
        for &(load, at, ty) in &o.loads {
            m.tick(1)?;
            let store = cx.layouts.store_size(ty).and_then(|n| {
                let span = owner.get(at as usize..usize::try_from(at.checked_add(n)?).ok()?)?;
                let &k = span.first()?;
                if k == NONE || k == MANY || span.iter().any(|&x| x != k) {
                    return None;
                }
                let w = o.writes.get(k as usize)?;
                let (store, val) = w.store?;
                let value = (!cx.parts.types.holds_pointer(ty)
                    || matches!(cx.parts.types.get(ty), Some(Type::Ptr { .. })))
                    && !memory_null(cx, val);
                (value && w.at == at && w.len == n && cx.ty_of(val)? == ty).then_some((store, val))
            });
            let Some((store, val)) = store else {
                read_left = true;
                continue;
            };
            let (sb, lb) = (block_of(store), block_of(load));
            let dominates = if sb == lb {
                store.raw() < load.raw()
            } else {
                if dom.is_none() {
                    let cfg = Cfg::build(cx.body, m)?;
                    let built = DomTree::compute(&cfg, m);
                    m.release(cfg.ledger_bytes());
                    *dom = Some(built?);
                }
                dom.as_ref().is_some_and(|d| d.dominates(sb, lb))
            };
            if dominates {
                edit.replace(load, val);
                changed = true;
            } else {
                read_left = true;
            }
        }
        if !read_left {
            for &i in &o.owned {
                edit.delete(i);
            }
            edit.delete(o.inst);
            changed = true;
        }
        Ok(changed)
    })();
    m.release(len * std::mem::size_of::<u32>());
    r
}

/// Whether `v` is the null pointer of a space other than device or constant, which lowering holds
/// only in memory (module docs).
fn memory_null(cx: Cx<'_>, v: Value) -> bool {
    let Value::Const(k) = v else {
        return false;
    };
    let space = match cx.parts.consts.ty(k).and_then(|t| cx.parts.types.get(t)) {
        Some(Type::Ptr { space }) => *space,
        _ => return false,
    };
    !matches!(space, AddrSpace::Device | AddrSpace::Constant)
        && matches!(cx.parts.consts.get(k).map(|c| &c.value), Some(Const::Null))
}

/// Whether `callee` is a plain memory call, and whether it reads its second argument: `memcpy`
/// and `memmove` copy from it, `memset` fills with it. `None` for any other call.
fn memory_call(callee: &Callee) -> Option<bool> {
    match callee {
        Callee::Llvm {
            op:
                LlvmIntrinsic::Memcpy {
                    volatile: false, ..
                }
                | LlvmIntrinsic::Memmove {
                    volatile: false, ..
                },
            ..
        } => Some(true),
        Callee::Llvm {
            op: LlvmIntrinsic::Memset {
                volatile: false, ..
            },
            ..
        } => Some(false),
        _ => None,
    }
}

/// Set the place of instruction `i` and of the derivations it is reached through: a
/// `getelementptr` or bitcast of a placed address is placed in its object, at the base's offset
/// plus its constant indices' (an offset only the run knows when one is not constant).
fn place_of(
    cx: Cx<'_>,
    i: InstId,
    (places, done): (&mut [Option<Place>], &mut [bool]),
    (chain, held): (&mut Vec<InstId>, &mut usize),
    m: &mut Meter,
) -> Result<(), Error> {
    // Walk down to a decided instruction, then decide the chain back up.
    chain.clear();
    let mut at = i;
    loop {
        if done.get(at.index()).copied().unwrap_or(true) {
            break;
        }
        m.tick(1)?;
        push(chain, at, m, held)?;
        let base = match cx.body.insts.get(at).map(|n| &n.kind) {
            Some(InstKind::Gep { base, .. })
            | Some(InstKind::Cast {
                op: CastOp::Bitcast,
                v: base,
            }) => *base,
            _ => break,
        };
        match base {
            Value::Inst(b) => at = b,
            _ => break,
        }
    }
    while let Some(c) = chain.pop() {
        let place = match cx.body.insts.get(c).map(|n| &n.kind) {
            Some(InstKind::Gep {
                src_ty,
                base: Value::Inst(b),
                indices,
                ..
            }) => places.get(b.index()).copied().flatten().map(|p| Place {
                object: p.object,
                at: p.at.and_then(|a| {
                    a.checked_add(gep_offset(cx.parts, cx.layouts, *src_ty, indices)?)
                }),
            }),
            Some(InstKind::Cast {
                op: CastOp::Bitcast,
                v: Value::Inst(b),
            }) => places.get(b.index()).copied().flatten(),
            _ => None,
        };
        if let Some(slot) = places.get_mut(c.index()) {
            *slot = place;
        }
        if let Some(d) = done.get_mut(c.index()) {
            *d = true;
        }
    }
    Ok(())
}

/// An object or byte index past its type: the internal invariant's refusal.
fn past_index() -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        "thread-memory forwarding: an index past its type",
    ))
}
