//! The bytes of an argument buffer's indirect-command objects.
//!
//! Metal's encoder writes an object's resource id into a command-buffer or compute or render
//! pipeline-state member of an argument buffer. Under the identity domains a command buffer's
//! field holds its word, the `gpuResourceID` the program or the host wrote there: the device
//! address of its records (`lower::access::icb`). Metal lays a Tier 2 argument buffer out as the
//! equivalent C structure, whose contents the GPU and the CPU may both modify, so an access to
//! those bytes is a plain access of the buffer's memory, as an access to any other member's is.
//! Any other object's field, and a command buffer's under the resident-table domains, holds no
//! value a translation defines. Only the handle load reads it: a pointer load of exactly one such
//! field, at a translation-time constant offset in the buffer's first element, which
//! `Walk::seed_loaded` classes as the object's handle (`lower::access::object`). Any other access
//! that may reach such an object's bytes is `ArgumentBufferObjects`, not implemented:
//!
//! - **The offsets an access may take** are its shape's (a constant, or a constant plus multiples
//!   of its strides) within its range. An access at least as wide as its step touches one
//!   contiguous range of bytes, which is tested once; a narrower one is tested at each offset, each
//!   a unit of work, at most `OBJECT_OFFSETS`.
//! - **The bytes a buffer holds.** A reference holds every byte of its `air.buffer_size` (or of
//!   its layout, when that is larger), a partial last element's too. A pointer with no stated size
//!   (`constant T* ab [[buffer(0)]]`, no `air.buffer_size`) may be indexed past its first element
//!   without bound. Each byte is judged by the layout at its offset modulo `air.arg_type_size` when
//!   that stride covers the layout; an access whose offsets are too many to list is then checked
//!   once per residue modulo the stride. Without such a stride, only the bytes below both the
//!   layout's end and `air.arg_type_size` are judged by the layout, and every byte at or past the
//!   smaller of the two counts as an object's.
//! - **The layouts** are a graph by metadata node, which may share a layout between many members.
//!   Which layouts hold an object, at any nesting depth, is found once per entry (`ObjectLayouts`)
//!   by one pass up the nesting edges; an overlap test then descends only into the elements an
//!   access covers partly (one or two per level when members do not overlap), each visit a unit
//!   of work (`overlaps`), so malformed overlapping members end at the work cap.
//! - **Run-time pointers.** The rule sees an access through the buffer's own parameter (or an
//!   element of a buffer table of them). A pointer whose root is a run-time address (`Dynamic`)
//!   reaches the buffer only when its address became data: the buffer's address escapes (a
//!   `ptrtoint`, a store into memory that does not forward it), or a read may load a pointer
//!   member whose pointee layout holds an object, or holds such a pointer
//!   (`ObjectLayouts::behind`). The analysis then refuses the entry's first run-time dereference
//!   (`analyze`).

use crate::air::abi::{
    AbiHandle, Member, MemberKind, ParamBinding, ParamRole, StructInfo, StructRef,
};
use crate::budget::{Meter, LAYOUT_NESTING};
use crate::error::Error;

use super::{OffsetRange, PtrFact, Reach, Shape};

/// Whether a handle is an indirect-command object: a command buffer or a compute or render
/// pipeline state.
pub(super) fn is_object(h: AbiHandle) -> bool {
    matches!(
        h,
        AbiHandle::CommandBuffer | AbiHandle::ComputePipelineState | AbiHandle::RenderPipelineState
    )
}

/// Whether the field of an object of handle `h` holds no value a translation defines, so that
/// only its handle load reads it: a pipeline state's, and a command buffer's unless its field
/// holds its word (`words`, the identity domains).
fn opaque(h: AbiHandle, words: bool) -> bool {
    is_object(h) && !(words && matches!(h, AbiHandle::CommandBuffer))
}

/// The most offsets (or residues) one access is checked at; an access that may take more is
/// refused.
const OBJECT_OFFSETS: u32 = 4096;

/// The layout a pointer member points to, when it is a buffer or argument buffer with one.
fn pointee(mem: &Member) -> Option<StructRef> {
    match &mem.kind {
        MemberKind::Indirect(arg) => match &arg.role {
            ParamRole::Buffer(b) | ParamRole::ArgumentBuffer(b) => b.layout,
            _ => None,
        },
        MemberKind::Plain | MemberKind::IndirectStruct { .. } => None,
    }
}

/// The layout a member nests by value, when it is a struct (or an array of them).
fn nested(mem: &Member) -> Option<StructRef> {
    match (mem.layout, &mem.kind) {
        (Some(inner), MemberKind::Plain | MemberKind::IndirectStruct { .. }) => Some(inner),
        _ => None,
    }
}

/// The reverse edge member `mem` makes in a graph of `n` layouts: the layout it nests by value
/// (`false`) or points to (`true`).
fn edge(mem: &Member, n: usize) -> Option<(u32, bool)> {
    nested(mem)
        .map(|t| (t.0, false))
        .or_else(|| pointee(mem).map(|t| (t.0, true)))
        .filter(|(t, _)| (*t as usize) < n)
}

/// Which layouts hold an indirect-command object whose field is opaque.
#[derive(Debug, Default)]
pub(super) struct ObjectLayouts {
    /// A command buffer's field holds its word, plain data (the identity domains).
    words: bool,
    /// By value, at any nesting depth.
    holds: Vec<bool>,
    /// Behind a pointer member at any nesting depth, whose pointee holds an object or is itself
    /// `behind`: a read of that member loads the address of an object's memory.
    behind: Vec<bool>,
}

impl ObjectLayouts {
    /// Both tables for `structs`, with a command buffer's field its word where `words`: one pass
    /// up the reverse nesting and pointer edges from the layouts that hold an object directly.
    /// Each member and edge is a unit of work, and the edge table is charged to the ledger.
    pub(super) fn of(
        structs: &[StructInfo],
        words: bool,
        m: &mut Meter,
    ) -> Result<ObjectLayouts, Error> {
        let n = structs.len();
        let work = |units: usize, m: &mut Meter| m.tick(u32::try_from(units).unwrap_or(u32::MAX));
        // Reverse edges, sorted by the layout they lead to: `(child or pointee, the layout that
        // holds it, through a pointer)`.
        let mut count = 0usize;
        for info in structs {
            work(info.members.len(), m)?;
            count += info
                .members
                .iter()
                .filter(|mem| edge(mem, n).is_some())
                .count();
        }
        let mut edges: Vec<(u32, u32, bool)> = Vec::new();
        m.reserve(&mut edges, count)?;
        for (s, info) in (0u32..).zip(structs) {
            edges.extend(
                info.members
                    .iter()
                    .filter_map(|mem| edge(mem, n))
                    .map(|(t, pointer)| (t, s, pointer)),
            );
        }
        edges.sort_unstable();
        work(count, m)?;
        let parents = |t: u32| {
            let lo = edges.partition_point(|e| e.0 < t);
            let hi = edges.partition_point(|e| e.0 <= t);
            edges.get(lo..hi).unwrap_or(&[])
        };

        let mut holds: Vec<bool> = Vec::new();
        m.reserve(&mut holds, n)?;
        holds.extend(structs.iter().map(|info| {
            info.members
                .iter()
                .any(|mem| mem.handle().is_some_and(|h| opaque(h, words)))
        }));
        let mut queue: Vec<u32> = Vec::new();
        m.reserve(&mut queue, n)?;
        queue.extend((0u32..).zip(&holds).filter(|(_, h)| **h).map(|(s, _)| s));
        while let Some(t) = queue.pop() {
            let up = parents(t);
            work(up.len(), m)?;
            for &(_, s, pointer) in up {
                if let Some(h) = holds.get_mut(s as usize).filter(|h| !pointer && !**h) {
                    *h = true;
                    queue.push(s);
                }
            }
        }

        let mut behind: Vec<bool> = Vec::new();
        m.reserve(&mut behind, n)?;
        behind.resize(n, false);
        for t in (0u32..).zip(&holds).filter(|(_, h)| **h).map(|(t, _)| t) {
            let up = parents(t);
            work(up.len(), m)?;
            for &(_, s, pointer) in up {
                if let Some(b) = behind.get_mut(s as usize).filter(|b| pointer && !**b) {
                    *b = true;
                    queue.push(s);
                }
            }
        }
        while let Some(t) = queue.pop() {
            let up = parents(t);
            work(up.len(), m)?;
            for &(_, s, _) in up {
                if let Some(b) = behind.get_mut(s as usize).filter(|b| !**b) {
                    *b = true;
                    queue.push(s);
                }
            }
        }
        // The edge table and the worklist are dropped here; the two tables stay charged.
        m.release(count * std::mem::size_of::<(u32, u32, bool)>() + n * std::mem::size_of::<u32>());
        Ok(ObjectLayouts {
            words,
            holds,
            behind,
        })
    }

    fn holds(&self, s: StructRef) -> bool {
        self.holds.get(s.0 as usize).copied().unwrap_or(false)
    }

    fn behind(&self, s: StructRef) -> bool {
        self.behind.get(s.0 as usize).copied().unwrap_or(false)
    }

    /// Whether member `mem` is an object whose field is opaque.
    fn opaque(&self, mem: &Member) -> bool {
        mem.handle().is_some_and(|h| opaque(h, self.words))
    }

    /// Whether a pointer member loads the address of an object's memory.
    fn object_pointer(&self, mem: &Member) -> bool {
        pointee(mem).is_some_and(|t| self.holds(t) || self.behind(t))
    }
}

/// The byte size of layout `s`: the end of its last member (a struct member's size is its whole
/// layout's).
fn layout_end(structs: &[StructInfo], s: StructRef) -> u64 {
    structs.get(s.0 as usize).map_or(0, |info| {
        info.members
            .iter()
            .map(|mem| {
                u64::from(mem.offset).saturating_add(
                    u64::from(mem.size).saturating_mul(u64::from(mem.array_len.max(1))),
                )
            })
            .max()
            .unwrap_or(0)
    })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Whether bytes `[lo, hi)` of layout `s` overlap a member `leaf` accepts, following nested
/// struct members and member arrays below `depth` (deeper is assumed to). `contains` says which
/// layouts hold such a member at any depth: a wholly covered element is answered by it, and only
/// the one or two elements an access covers partly are descended into. Each visit charges a unit
/// of work per member.
fn overlaps(
    structs: &[StructInfo],
    contains: &dyn Fn(StructRef) -> bool,
    leaf: &dyn Fn(&Member) -> bool,
    s: StructRef,
    (lo, hi): (u64, u64),
    depth: u64,
    m: &mut Meter,
) -> Result<bool, Error> {
    if depth >= LAYOUT_NESTING {
        return Ok(true);
    }
    let Some(info) = structs.get(s.0 as usize) else {
        return Ok(false);
    };
    m.tick(u32::try_from(info.members.len()).unwrap_or(u32::MAX))?;
    for mem in &info.members {
        let start = u64::from(mem.offset);
        let size = u64::from(mem.size);
        let end = start.saturating_add(size.saturating_mul(u64::from(mem.array_len.max(1))));
        if hi <= start || end <= lo {
            continue;
        }
        let hit = match nested(mem) {
            Some(inner) if size > 0 => {
                if !contains(inner) {
                    false
                } else if lo <= start && end <= hi {
                    true
                } else {
                    let first = (lo.max(start) - start) / size;
                    let last = (hi.min(end) - 1 - start) / size;
                    // A middle element is wholly covered.
                    if last > first + 1 {
                        true
                    } else {
                        let mut any = false;
                        for k in first..=last {
                            let base = start + k * size;
                            let within = (lo.saturating_sub(base), hi.min(base + size) - base);
                            if within.0 == 0 && within.1 == size {
                                any = true;
                                break;
                            }
                            if overlaps(structs, contains, leaf, inner, within, depth + 1, m)? {
                                any = true;
                                break;
                            }
                        }
                        any
                    }
                }
            }
            _ => leaf(mem),
        };
        if hit {
            return Ok(true);
        }
    }
    Ok(false)
}

/// An argument buffer whose layout holds an indirect-command object by value, or the address of
/// memory that does.
#[derive(Clone, Copy, Debug)]
pub(super) struct ObjectBuffer {
    /// The entry parameter (for a buffer table, the table).
    pub(super) param: u32,
    layout: StructRef,
    /// The bytes of an element that the layout alone describes: the layout's end, or
    /// `air.arg_type_size` when that is smaller. Without a stride, every byte at or past them may
    /// be a later element's, so it counts as an object's.
    first: u64,
    /// The bytes between consecutive elements: `air.arg_type_size` when it covers the layout,
    /// for a parameter that is not a buffer table; `None` when no stride is known.
    stride: Option<u64>,
    /// The bytes `[0, held)` the buffer holds: for a reference (`air.buffer_size`), its size, or
    /// the layout's end when that is larger; `None` (unbounded) for a pointer with no stated size
    /// or a buffer table.
    held: Option<u64>,
    holds: bool,
    behind: bool,
}

impl ObjectBuffer {
    /// The object buffer parameter `b` is, if it is an argument buffer whose layout holds an
    /// object or its address; `table` says whether it is a buffer table, whose elements' strides
    /// and sizes are not stated.
    pub(super) fn of(
        b: &ParamBinding,
        table: bool,
        structs: &[StructInfo],
        layouts: &ObjectLayouts,
    ) -> Option<ObjectBuffer> {
        let ParamRole::ArgumentBuffer(arg) = &b.role else {
            return None;
        };
        let layout = arg.layout?;
        let (holds, behind) = (layouts.holds(layout), layouts.behind(layout));
        if !holds && !behind {
            return None;
        }
        let end = layout_end(structs, layout);
        let size = b.ty.size.map(u64::from).filter(|_| !table);
        Some(ObjectBuffer {
            param: b.param,
            layout,
            first: size.map_or(end, |s| s.min(end)),
            stride: size.filter(|&s| s > 0 && s >= end),
            held: arg
                .buffer_size
                .filter(|_| !table)
                .map(|s| u64::from(s).max(end)),
            holds,
            behind,
        })
    }

    /// Whether this buffer's address, once it is data, may lead a run-time pointer to an
    /// object's bytes.
    pub(super) fn reaches_objects(&self) -> bool {
        self.holds || self.behind
    }

    /// Whether an access through a pointer of `fact` into this buffer, reaching `reach` from it,
    /// may touch an object member's bytes. The caller exempts the handle load.
    pub(super) fn touches_object(
        &self,
        structs: &[StructInfo],
        layouts: &ObjectLayouts,
        fact: PtrFact,
        reach: Option<Reach>,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        if !self.holds {
            return Ok(false);
        }
        let leaf = |mem: &Member| layouts.opaque(mem);
        self.touches(structs, &|s| layouts.holds(s), &leaf, fact, reach, m)
    }

    /// Whether a read through a pointer of `fact`, reaching `reach`, may load a pointer member
    /// whose pointee holds an object or another such pointer.
    pub(super) fn reads_object_address(
        &self,
        structs: &[StructInfo],
        layouts: &ObjectLayouts,
        fact: PtrFact,
        reach: Option<Reach>,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        if !self.behind {
            return Ok(false);
        }
        let leaf = |mem: &Member| layouts.object_pointer(mem);
        self.touches(structs, &|s| layouts.behind(s), &leaf, fact, reach, m)
    }

    /// Whether an access through a pointer of `fact` reaching `reach` may touch a member `leaf`
    /// accepts (`contains` says which layouts hold one).
    fn touches(
        &self,
        structs: &[StructInfo],
        contains: &dyn Fn(StructRef) -> bool,
        leaf: &dyn Fn(&Member) -> bool,
        fact: PtrFact,
        reach: Option<Reach>,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let Some(reach) = reach else {
            return Ok(true);
        };
        let (Ok(start), Ok(end)) = (i64::try_from(reach.start), i64::try_from(reach.end)) else {
            return Ok(true);
        };
        // The bytes `[0, limit)` the buffer holds, `None` when unbounded.
        let limit = match self.held.map(i64::try_from) {
            None => None,
            Some(Ok(l)) => Some(l),
            Some(Err(_)) => return Ok(true),
        };
        // Only an access starting in `[1 - end, limit - start)` meets those bytes; the offsets it
        // may start at are its range's, and its shape's: `base` plus multiples of `step`. `lo`
        // and `hi` are inclusive.
        let (mut lo, mut hi) = (
            1 - end,
            limit.map_or(i64::MAX, |l| l.saturating_sub(start).saturating_sub(1)),
        );
        match fact.range {
            OffsetRange::Empty => return Ok(false),
            OffsetRange::Span { lo: l, hi: h } => (lo, hi) = (lo.max(l), hi.min(h)),
            OffsetRange::Top => {}
        }
        if hi < lo {
            return Ok(false);
        }
        let (base, step) = match fact.shape {
            Shape::Empty => return Ok(false),
            Shape::Exact(o) => (o, 0),
            Shape::Strided { base, strides } => match i64::try_from(strides.iter().fold(0, gcd)) {
                Ok(step) if step > 0 => (base, step),
                _ => return Ok(true),
            },
            Shape::Top => (lo, 1),
        };
        let bytes = |(a, b): (i64, i64), m: &mut Meter| {
            self.buffer_bytes(structs, contains, leaf, (a, b), limit, m)
        };
        if step == 0 {
            if base < lo || base > hi {
                return Ok(false);
            }
            m.tick(1)?;
            return bytes((base.saturating_add(start), base.saturating_add(end)), m);
        }
        // The first offset at or above `lo` congruent to `base`, and how many follow it.
        let first = i128::from(lo) + (i128::from(base) - i128::from(lo)).rem_euclid(step.into());
        if i128::from(hi) < first {
            return Ok(false);
        }
        let count = (i128::from(hi) - first) / i128::from(step) + 1;
        let last = first + (count - 1) * i128::from(step);
        let (Ok(first), Ok(last)) = (i64::try_from(first), i64::try_from(last)) else {
            return Ok(true);
        };
        if i128::from(end) - i128::from(start) >= i128::from(step) {
            // At least as wide as the step: consecutive accesses meet, so together they touch
            // exactly the bytes from the first's start to the last's end.
            m.tick(1)?;
            return bytes((first.saturating_add(start), last.saturating_add(end)), m);
        }
        if count <= i128::from(OBJECT_OFFSETS) {
            let Ok(count) = u32::try_from(count) else {
                return Ok(true);
            };
            m.tick(count)?;
            for k in 0..i64::from(count) {
                let at = first + k * step;
                if bytes((at.saturating_add(start), at.saturating_add(end)), m)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        // Too many offsets to list: in a buffer of repeated elements, the offsets' residues
        // modulo the stride, each an access in the periodic layout.
        let Some(Ok(s)) = self.stride.map(i64::try_from) else {
            return Ok(true);
        };
        let g = i64::try_from(gcd(step.unsigned_abs(), s.unsigned_abs())).unwrap_or(1);
        let Ok(residues) = u32::try_from(s / g) else {
            return Ok(true);
        };
        if residues > OBJECT_OFFSETS {
            return Ok(true);
        }
        m.tick(residues)?;
        let first = first.rem_euclid(g);
        for j in 0..i64::from(residues) {
            let at = first + j * g;
            let (a, b) = (at.saturating_add(start), at.saturating_add(end));
            if self.buffer_bytes(structs, contains, leaf, (a, b), None, m)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether bytes `[a, b)` of the buffer, clipped to `[0, limit)`, overlap a member `leaf`
    /// accepts. With a stride, each byte is judged by the layout at its offset modulo the
    /// stride; without one, a byte at or past `first` may be a later element's, so it counts as
    /// such a member.
    fn buffer_bytes(
        &self,
        structs: &[StructInfo],
        contains: &dyn Fn(StructRef) -> bool,
        leaf: &dyn Fn(&Member) -> bool,
        (a, b): (i64, i64),
        limit: Option<i64>,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let (a, b) = (a.max(0), limit.map_or(b, |l| b.min(l)));
        if b <= a {
            return Ok(false);
        }
        let (Ok(a), Ok(b)) = (u64::try_from(a), u64::try_from(b)) else {
            return Ok(true);
        };
        let Some(s) = self.stride else {
            if b > self.first {
                return Ok(true);
            }
            return overlaps(structs, contains, leaf, self.layout, (a, b), 0, m);
        };
        let (k0, k1) = (a / s, (b - 1) / s);
        if k1 > k0 + 1 {
            // A whole element.
            return Ok(contains(self.layout));
        }
        for k in k0..=k1 {
            let base = k * s;
            let within = (a.saturating_sub(base), b.min(base + s) - base);
            if overlaps(structs, contains, leaf, self.layout, within, 0, m)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests;
