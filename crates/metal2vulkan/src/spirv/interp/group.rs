//! Subgroup operations over the active tangle of one SIMT step.
//!
//! The tangle is the set of lanes executing the instruction together (see `sched`). Reading a
//! value from a lane outside it, or outside the subgroup, is undefined in SPIR-V and traps here
//! ([`TrapKind::InactiveLane`]); so does a non-uniform `Broadcast` lane id. Reductions and scans
//! combine values in increasing lane order, rounding each float operation.

use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, GroupArith, GroupInst, GroupOperation, QuadDirection, ShuffleOp};

use super::eval::{float_op, int_op, map2};
use super::float;
use super::sched::{Lane, Prog};
use super::value::{mask, sext, Val};
use super::{malformed, IResult, TrapKind};

/// Execute `op` for `lanes`, the tangle in scheduling order. Returns one result per lane, in the
/// same order.
pub(super) fn exec(p: &Prog<'_>, op: &GroupInst, size: u32, lanes: &[&Lane]) -> IResult<Vec<Val>> {
    // Operand values per lane, keyed by subgroup lane id.
    let operand = |v: Value| -> IResult<Vec<(u32, Val)>> {
        lanes
            .iter()
            .map(|l| Ok((l.sg_lane, l.get(p, v)?)))
            .collect()
    };
    let active: Vec<u32> = lanes.iter().map(|l| l.sg_lane).collect();
    let from = |vals: &[(u32, Val)], lane: i64| -> IResult<Val> {
        u32::try_from(lane)
            .ok()
            .and_then(|k| vals.iter().find(|(id, _)| *id == k))
            .map(|(_, v)| v.clone())
            .ok_or_else(|| TrapKind::InactiveLane { lane }.err())
    };
    let lowest = active
        .iter()
        .copied()
        .min()
        .ok_or_else(|| malformed("empty tangle"))?;
    let per_lane = |f: &dyn Fn(u32) -> IResult<Val>| -> IResult<Vec<Val>> {
        active.iter().map(|id| f(*id)).collect()
    };
    let all_bools = |vals: &[(u32, Val)]| -> IResult<Vec<bool>> {
        vals.iter().map(|(_, v)| Ok(v.s()? != 0)).collect()
    };
    let ballot_of = |vals: &[(u32, Val)]| -> IResult<Val> {
        let mut words = [0u64; 4];
        for (id, v) in vals {
            if v.s()? != 0 {
                if let Some(w) = words.get_mut((*id / 32) as usize) {
                    *w |= 1 << (id % 32);
                }
            }
        }
        Ok(Val::from_comps(words.to_vec()))
    };
    // A ballot operand's bits below the subgroup size, as a u64.
    let ballot_bits = |v: &Val| -> IResult<u64> {
        let c = v.comps()?;
        let lo = c.first().copied().unwrap_or(0) & mask(32);
        let hi = c.get(1).copied().unwrap_or(0) & mask(32);
        Ok((lo | (hi << 32)) & mask(size as u8))
    };
    match op {
        GroupInst::Elect => per_lane(&|id| Ok(Val::S(u64::from(id == lowest)))),
        GroupInst::All(v) | GroupInst::Any(v) => {
            let bs = all_bools(&operand(*v)?)?;
            let r = if matches!(op, GroupInst::All(_)) {
                bs.iter().all(|b| *b)
            } else {
                bs.iter().any(|b| *b)
            };
            per_lane(&|_| Ok(Val::S(u64::from(r))))
        }
        GroupInst::BroadcastFirst(v) => {
            let vals = operand(*v)?;
            let src = from(&vals, i64::from(lowest))?;
            per_lane(&|_| Ok(src.clone()))
        }
        GroupInst::Ballot(v) => {
            let b = ballot_of(&operand(*v)?)?;
            per_lane(&|_| Ok(b.clone()))
        }
        GroupInst::BallotBitExtract { ballot, index } => {
            let bs = operand(*ballot)?;
            let is = operand(*index)?;
            per_lane(&|id| {
                let b = ballot_bits(&from(&bs, i64::from(id))?)?;
                let i = from(&is, i64::from(id))?.s()?;
                if i >= u64::from(size) {
                    return Err(TrapKind::Undefined("ballot bit index beyond the subgroup").err());
                }
                Ok(Val::S((b >> i) & 1))
            })
        }
        GroupInst::Shuffle { op: sop, value, x } => {
            let vals = operand(*value)?;
            let xs = operand(*x)?;
            per_lane(&|id| {
                let x = from(&xs, i64::from(id))?.s()? as i64;
                let src = match sop {
                    ShuffleOp::Shuffle => x,
                    ShuffleOp::Xor => i64::from(id) ^ x,
                };
                if src >= i64::from(size) {
                    return Err(TrapKind::InactiveLane { lane: src }.err());
                }
                from(&vals, src)
            })
        }
        GroupInst::Arith {
            op: aop,
            group,
            value,
        } => {
            let vals = operand(*value)?;
            let n = p.t.num_of(*value)?;
            let combine = |a: &Val, b: &Val| -> IResult<Val> { arith(*aop, n.width, a, b) };
            let fold = |pred: &dyn Fn(u32) -> bool| -> IResult<Option<Val>> {
                let mut sorted: Vec<&(u32, Val)> = vals.iter().filter(|(k, _)| pred(*k)).collect();
                sorted.sort_by_key(|(k, _)| *k);
                let mut acc: Option<Val> = None;
                for (_, v) in sorted {
                    acc = Some(match acc {
                        None => v.clone(),
                        Some(a) => combine(&a, v)?,
                    });
                }
                Ok(acc)
            };
            per_lane(&|id| {
                let r = match group {
                    GroupOperation::Reduce => fold(&|_| true)?,
                    GroupOperation::InclusiveScan => fold(&|k| k <= id)?,
                    GroupOperation::ExclusiveScan => fold(&|k| k < id)?,
                    GroupOperation::ClusteredReduce { size: c, .. } => {
                        let c = (*c).max(1);
                        fold(&|k| k / c == id / c)?
                    }
                };
                match r {
                    Some(v) => Ok(v),
                    None => identity(*aop, n.width, &from(&vals, i64::from(id))?),
                }
            })
        }
        GroupInst::QuadBroadcast { value, index } => {
            let vals = operand(*value)?;
            let is = operand(*index)?;
            per_lane(&|id| {
                let i = from(&is, i64::from(id))?.s()?;
                if i >= 4 {
                    return Err(TrapKind::InactiveLane { lane: i as i64 }.err());
                }
                from(&vals, i64::from(id & !3) + i as i64)
            })
        }
        GroupInst::QuadSwap { value, dir, .. } => {
            let vals = operand(*value)?;
            let x = match dir {
                QuadDirection::Horizontal => 1,
                QuadDirection::Vertical => 2,
                QuadDirection::Diagonal => 3,
            };
            per_lane(&|id| from(&vals, i64::from(id ^ x)))
        }
        GroupInst::QuadAny(v) => {
            let vals = operand(*v)?;
            per_lane(&|id| {
                let mut any = false;
                for (k, b) in &vals {
                    any |= (*k & !3) == (id & !3) && b.s()? != 0;
                }
                Ok(Val::S(u64::from(any)))
            })
        }
    }
}

/// Combine two lane values with a reduction operator.
fn arith(op: GroupArith, w: u8, a: &Val, b: &Val) -> IResult<Val> {
    let int = |bop: BinOp| map2(a, b, |x, y| int_op(bop, w, x, y));
    let flt = |bop: BinOp| map2(a, b, |x, y| float_op(bop, w, x, y));
    match op {
        GroupArith::IAdd => int(BinOp::IAdd),
        GroupArith::BitwiseAnd | GroupArith::LogicalAnd => int(BinOp::BitwiseAnd),
        GroupArith::BitwiseOr | GroupArith::LogicalOr => int(BinOp::BitwiseOr),
        GroupArith::BitwiseXor | GroupArith::LogicalXor => int(BinOp::BitwiseXor),
        GroupArith::FAdd => flt(BinOp::FAdd),
        GroupArith::IMul => int(BinOp::IMul),
        GroupArith::FMul => flt(BinOp::FMul),
        GroupArith::UMin => map2(a, b, |x, y| Ok(x.min(y))),
        GroupArith::UMax => map2(a, b, |x, y| Ok(x.max(y))),
        GroupArith::SMin => map2(a, b, |x, y| {
            Ok(if sext(x, w) <= sext(y, w) { x } else { y })
        }),
        GroupArith::SMax => map2(a, b, |x, y| {
            Ok(if sext(x, w) >= sext(y, w) { x } else { y })
        }),
        GroupArith::FMin => map2(a, b, |x, y| float::bin(w, x, y, f64::min)),
        GroupArith::FMax => map2(a, b, |x, y| float::bin(w, x, y, f64::max)),
    }
}

/// The identity of a reduction operator, shaped like `like` (the result of an empty exclusive
/// scan).
fn identity(op: GroupArith, w: u8, like: &Val) -> IResult<Val> {
    let id = match op {
        GroupArith::IAdd
        | GroupArith::BitwiseOr
        | GroupArith::BitwiseXor
        | GroupArith::LogicalOr
        | GroupArith::LogicalXor
        | GroupArith::UMax => 0,
        GroupArith::BitwiseAnd | GroupArith::UMin => mask(w),
        GroupArith::LogicalAnd | GroupArith::IMul => 1,
        GroupArith::SMin => mask(w) >> 1,
        GroupArith::SMax => 1u64 << (w - 1),
        GroupArith::FAdd => 0,
        GroupArith::FMul => float::from_f64(1.0, w)?,
        GroupArith::FMin => float::from_f64(f64::INFINITY, w)?,
        GroupArith::FMax => float::from_f64(f64::NEG_INFINITY, w)?,
    };
    let n = like.comps()?.len();
    Ok(Val::from_comps(vec![id; n]))
}
