//! The status word of a texel result.
//!
//! `read`, `sample`, and `gather` return `{T, i8}` where their declaration states a residency
//! status (`intrinsics::Texture::status`, which AIR 2.2 added); an AIR 2.1 declaration returns
//! the texel alone, so such a call has no word, nothing here follows it, and lowering builds no
//! status composite for it (`emit::with_word` hands back the texel). The status word is Metal's
//! residency status:
//! `sparse_read`, `sparse_sample`, and `sparse_gather` compile to the same intrinsics as `read`,
//! `sample`, and `gather`, and `sparse_color::resident()` is `(status & 1) == 0` (bit 0 set: the
//! access touched a non-resident region). No MSL function reads another bit: `metal_stdlib`'s
//! `sparse_color` keeps only `from_resident_pages : 1`, built from bit 0, so the other bits are
//! not established, and a function whose effects depend on one of them is refused.
//!
//! **Carriers**. The word travels in values before it is taken: a carrier is a value with the
//! member path at which it holds a call's word (`[1]` in the call's own `{T, i8}`, empty in the
//! word itself). A copy (a `phi` merging reads from different blocks, a `select`, a `freeze`)
//! carries it at the same path. An `insertvalue` into a carrier carries it on unless it writes the
//! word's own member (Apple's GL code generator writes a read's swizzled texel back into member 0,
//! and the word goes on in member 1); one that inserts a carrier carries it below the member it
//! writes (a word re-inserted in a rebuilt `{T, i8}`, a read nested in a larger aggregate). An
//! `extractvalue` of a member that holds the word carries it below that member; of another member
//! (the texel), it observes nothing; of the word's own member, it takes the word. Every other use
//! of a carrier (a store, a call, a return, ...) exposes the whole word, which is refused.
//!
//! **Demanded bits**. Which bits of a word the function's effects depend on is a backward
//! demanded-bits analysis over the word's forward slice: `and` with a constant keeps the constant's
//! bits, `or`, `xor`, `phi`, `freeze`, and a `select`'s two values pass the bits demanded of
//! their result, shifts by a constant move them, `zext`, `sext`, and `trunc` re-width them, and
//! `add`, `sub`, and `mul` demand every bit at or below the highest bit demanded of their result
//! (carries only rise). A word inserted whole into an aggregate demands nothing there: its bits are
//! demanded where a carrier takes it again. Every other use (a store, a call, a comparison, a
//! branch, a conversion, a vector insertion, ...) demands every bit of the value it uses.
//! Front-end spellings of `resident()` (`and 1` then `xor 1` or `icmp eq 0`, `trunc` to `i1`, and
//! the optimizer's `shl k` / `and (1 << k)` / `or` packing of several words) demand bit 0 alone.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{AirIntrinsic, TexOp};
use crate::air::ir::{BinOp, Body, Callee, CastOp, InstId, InstKind, Type, Value};
use crate::lower::select::SelectCx;
use crate::lower::users::Users;

use super::const_int;

/// How a texel call's status word is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusUse {
    /// The function's effects depend on bit 0 of the word (`!resident()`) and on no other bit.
    Residency,
    /// An effect depends on another bit of the word, or on the whole `{T, i8}` value: not
    /// established.
    Other,
}

/// Whether `kind` is a texture call returning a texel with a status word: one whose declaration
/// states a residency status, so its result is `{T, i8}`. An AIR 2.1 declaration states none and
/// returns the texel itself, which carries no word to observe.
fn texel_call(kind: &InstKind) -> bool {
    let InstKind::Call(c) = kind else {
        return false;
    };
    matches!(
        &c.callee,
        Callee::Air { op: AirIntrinsic::Texture(t), .. } if t.status && matches!(
            t.op,
            TexOp::Read | TexOp::Sample { .. } | TexOp::SampleGrad { .. } | TexOp::Gather { .. }
        )
    )
}

/// The operands a copy passes through whole: a `phi`'s incoming values, a `select`'s two
/// values, and a `freeze`'s operand.
fn copied(kind: &InstKind) -> Option<Vec<Value>> {
    Some(match kind {
        InstKind::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
        InstKind::Select { t, f, .. } => vec![*t, *f],
        InstKind::Freeze(v) => vec![*v],
        _ => return None,
    })
}

/// Where a carrier holds a word: the value, and the member path of the word in it (empty: the
/// value is the word).
type Place = (InstId, Box<[u32]>);

/// The paths at which user `kind` of the carrier `v` holding a word at `path` holds it in its
/// result, or `None` where the use exposes the word. The word itself (`path` empty) is carried
/// only by an `insertvalue` of it; its other uses are the demanded-bits slice's.
fn carried(kind: &InstKind, v: Value, path: &[u32]) -> Option<Vec<Box<[u32]>>> {
    let mut out = Vec::new();
    match kind {
        InstKind::InsertValue { agg, val, indices } if *agg == v || *val == v => {
            if *agg == v {
                if indices.starts_with(path) && indices.len() > path.len() {
                    // Into the word's own member: a member no word has.
                    return None;
                }
                if !path.starts_with(indices) {
                    out.push(path.into());
                }
            }
            if *val == v {
                out.push(indices.iter().chain(path).copied().collect());
            }
        }
        _ if path.is_empty() => {}
        _ if copied(kind).is_some_and(|p| p.contains(&v)) => out.push(path.into()),
        InstKind::ExtractValue { agg, indices } if *agg == v => {
            if let Some(below) = path.strip_prefix(&**indices) {
                out.push(below.into());
            } else if indices.starts_with(path) {
                return None;
            }
        }
        _ => return None,
    }
    Some(out)
}

/// The low `w` bits.
fn mask(w: u32) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

/// Every bit at or below the highest bit of `d` (0 for none).
fn at_or_below(d: u64) -> u64 {
    match d {
        0 => 0,
        _ => mask(64 - d.leading_zeros()),
    }
}

/// The integer width of instruction `i`'s scalar result, if it is one.
fn width(cx: &SelectCx<'_>, body: &Body, i: InstId) -> Option<u32> {
    let ty = body.insts.get(i)?.ty;
    match cx.entry.module.types().get(ty)? {
        Type::Int(w) => Some(u32::from(*w)),
        _ => None,
    }
}

/// The bits of operand `v` (of width `wv`) that user `u` (of width `wu`, when an integer
/// scalar) demands, given the bits `d` demanded of `u`'s result; `None` where `u` passes no
/// bits through (it demands every bit of `v`).
fn transfer(
    cx: &SelectCx<'_>,
    kind: &InstKind,
    v: Value,
    wv: u32,
    wu: Option<u32>,
    d: u64,
) -> Option<u64> {
    let constant = |x: Value| const_int(cx, x);
    match kind {
        InstKind::Phi { .. } | InstKind::Freeze(_) => Some(d),
        InstKind::Select { t, f, .. } if *t == v || *f == v => Some(d),
        InstKind::BinOp { op, a, b, .. } => {
            wu?;
            let other = if *a == v { *b } else { *a };
            match op {
                BinOp::And => Some(match constant(other) {
                    Some(c) => d & c,
                    None => d,
                }),
                BinOp::Or | BinOp::Xor => Some(d),
                BinOp::Add | BinOp::Sub | BinOp::Mul => Some(at_or_below(d)),
                BinOp::Shl | BinOp::LShr | BinOp::AShr if *a == v && *b != v => {
                    let k = constant(*b).filter(|k| *k < u64::from(wv))?;
                    let k = u32::try_from(k).ok()?;
                    Some(match op {
                        BinOp::Shl => d >> k,
                        BinOp::LShr => (d << k) & mask(wv),
                        _ => {
                            let shifted = (d << k) & mask(wv);
                            // The top `k` bits of the result copy the sign bit.
                            let top = d & !(mask(wv) >> k) & mask(wv);
                            if top != 0 {
                                shifted | 1u64 << (wv - 1)
                            } else {
                                shifted
                            }
                        }
                    })
                }
                _ => None,
            }
        }
        InstKind::Cast { op, .. } => {
            let wu = wu?;
            match op {
                CastOp::ZExt => Some(d & mask(wv)),
                CastOp::SExt => {
                    let sign = if d & !mask(wv) != 0 {
                        1u64 << (wv - 1)
                    } else {
                        0
                    };
                    Some((d & mask(wv)) | sign)
                }
                CastOp::Trunc => Some(d & mask(wu)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The texel calls of `body` whose status word is observed, by how.
pub(super) fn observed(cx: &SelectCx<'_>, body: &Body) -> BTreeMap<InstId, StatusUse> {
    // Every instruction that uses each instruction.
    let users = Users::of(body);
    // The carriers, each with the calls whose word it holds there: the texel calls (the word at
    // member 1) and every value a carrier's use carries one into.
    let mut carries: BTreeMap<Place, BTreeSet<InstId>> = body
        .insts
        .iter()
        .filter(|(_, i)| texel_call(&i.kind))
        .map(|(at, _)| ((at, Box::from([1u32])), BTreeSet::from([at])))
        .collect();
    let mut out: BTreeMap<InstId, StatusUse> = BTreeMap::new();
    let mut work: Vec<Place> = carries.keys().cloned().collect();
    while let Some(place) = work.pop() {
        let from = carries.get(&place).cloned().unwrap_or_default();
        let (c, path) = &place;
        for &u in users.get(*c) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let Some(paths) = carried(&inst.kind, Value::Inst(*c), path) else {
                for call in &from {
                    out.insert(*call, StatusUse::Other);
                }
                continue;
            };
            for p in paths {
                let set = carries.entry((u, p.clone())).or_default();
                let before = set.len();
                set.extend(from.iter().copied());
                if set.len() != before {
                    work.push((u, p));
                }
            }
        }
    }
    // The words taken, each with the calls it is the word of.
    let words: Vec<(InstId, &BTreeSet<InstId>)> = carries
        .iter()
        .filter(|((_, path), _)| path.is_empty())
        .map(|((w, _), calls)| (*w, calls))
        .collect();
    let taken: BTreeSet<InstId> = words.iter().map(|(w, _)| *w).collect();
    // Demanded bits over the words' forward slices: a value joins the slice through a use that
    // passes bits; every other use demands all of its bits.
    let mut slice: BTreeSet<InstId> = taken.clone();
    let mut work: Vec<InstId> = slice.iter().copied().collect();
    while let Some(v) = work.pop() {
        let Some(wv) = width(cx, body, v) else {
            continue;
        };
        for &u in users.get(v) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let wu = width(cx, body, u);
            if transfer(cx, &inst.kind, Value::Inst(v), wv, wu, 1).is_some()
                && wu.is_some()
                && slice.insert(u)
            {
                work.push(u);
            }
        }
    }
    let mut demanded: BTreeMap<InstId, u64> = BTreeMap::new();
    let mut work: Vec<InstId> = slice.iter().copied().collect();
    while let Some(v) = work.pop() {
        let wv = width(cx, body, v).unwrap_or(64);
        let mut d = 0u64;
        for &u in users.get(v) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let passed = if slice.contains(&u) {
                let du = demanded.get(&u).copied().unwrap_or(0);
                transfer(cx, &inst.kind, Value::Inst(v), wv, width(cx, body, u), du)
            } else if taken.contains(&v)
                && matches!(&inst.kind, InstKind::InsertValue { val, .. } if *val == Value::Inst(v))
            {
                // A word inserted whole: demanded where it is taken again.
                Some(0)
            } else {
                None
            };
            d |= passed.unwrap_or(u64::MAX) & mask(wv);
        }
        if demanded.get(&v).copied().unwrap_or(0) != d {
            demanded.insert(v, d);
            // The operands in the slice depend on this value's demand.
            if let Some(inst) = body.insts.get(v) {
                inst.kind.for_each_value(|x| {
                    if let Value::Inst(o) = x {
                        if slice.contains(&o) {
                            work.push(o);
                        }
                    }
                });
            }
        }
    }
    for (w, calls) in words {
        let d = demanded.get(&w).copied().unwrap_or(0) & mask(8);
        let used = match d {
            0 => continue,
            1 => StatusUse::Residency,
            _ => StatusUse::Other,
        };
        for call in calls {
            let e = out.entry(*call).or_insert(used);
            if used == StatusUse::Other {
                *e = StatusUse::Other;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{at_or_below, mask};

    #[test]
    fn masks_cover_their_widths() {
        assert_eq!(mask(1), 1);
        assert_eq!(mask(8), 0xFF);
        assert_eq!(mask(64), u64::MAX);
        assert_eq!(at_or_below(0), 0);
        assert_eq!(at_or_below(1), 1);
        assert_eq!(at_or_below(0x10), 0x1F);
        assert_eq!(at_or_below(1 << 63), u64::MAX);
    }
}
