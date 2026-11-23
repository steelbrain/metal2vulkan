//! The AGX lane intrinsics `llvm.agx{2,3}.*`, ported from the old engine's device-measured
//! lowerings (`passes/air_calls/agx_emask.rs`, `passes/air_calls/matrix.rs`,
//! `native/emitter/ops/air_calls.rs`), and the bit interleave measured on an M3 Max. Every other
//! AGX operation (`Agx(Unmodeled)`: the `wide` emask forms, `igemm`, …) is refused as
//! `Unsupported{AgxUnmodeled, NotImplemented}`, by the memory analysis where it takes a pointer;
//! `llvm.agx3.yield`, a delay with no data effect, never reaches lowering: decode drops it
//! (`air::intrinsics::Dropped::AgxYield`).
//!
//! - **`llvm.agx3.edgecheck(base, low, count)`** is the four-lane activity mask of the emask
//!   accesses: bit `i` is set exactly when `base + i >= 0` (signed) and
//!   `(unsigned)(base + i - low) < count`. Per-thread arithmetic, no subgroup operation.
//! - **`llvm.agx3.{load,store}.with.emask.{global,local}.<T>`** access the lanes of a 2- or
//!   4-lane vector of 8-, 16-, or 32-bit integers at `ptr + i · sizeof(elem)`. Lane `i` is
//!   active exactly when bit `i` of the first mask is set; an inactive lane is never accessed,
//!   and an inactive lane of a load reads zero. The second mask is the shape constant
//!   `(1 << lanes) - 1` (the backend accepts no other value; another is refused), and the
//!   stride operand has no effect. The "lanes" are the vector's elements, not simdgroup lanes:
//!   no subgroup operation. Each active lane is one contained leaf access through the access
//!   family's endpoints (`lower::access::copy`), inside a selection on its mask bit.
//! - **`llvm.agx2.cluster.num`** names the physical GPU cluster the threadgroup runs on, uniform
//!   over the threadgroup and chosen by the hardware (0..=36 measured on an M3 Max). Vulkan has
//!   no cluster; the constant 0 is a value the hardware itself gives (one cluster), and it keeps
//!   the value uniform.
//! - **`llvm.agx2.shfl(a, b)`** interleaves the bits of its two `i16` operands into an `i32`: bit
//!   `2i` is bit `i` of `a`, bit `2i + 1` bit `i` of `b` (measured on an M3 Max over all 2^32
//!   pairs; MSL documents the same contract for `interleave(ushort, ushort)`, which compiles to
//!   another intrinsic). Each operand, held zero-extended in its container, is spread to the even
//!   bits by four mask-and-shift steps, and `b`'s is shifted up by one. Per-thread arithmetic,
//!   exact on every device, no subgroup operation.
//! - **`llvm.agx2.{f16,f32}matmad{4x4,8x8}.v2{f16,f32}(a, b, c)`** is `c + a·b` over an 8x8
//!   tile distributed two cells per lane over the 32 lanes of the simdgroup (device-measured
//!   storage mapping: lane `L` owns row `((L >> 1) & 3) | ((L >> 4) << 2)`, columns
//!   `((L >> 3) & 1) · 4 + (L & 1) · 2` and the next one); the 4x4 form multiplies each of the
//!   tile's four 4x4 quadrants independently. Each lane reads the `a` and `b` cells of step `k`
//!   from their owners with absolute shuffles inside its own simdgroup (`base | owner` in
//!   `Cluster32`) and accumulates with `Fma` in increasing `k`. It raises the class to
//!   `CrossLane(SHUFFLE)` in both tiers. The storage mapping is measured; the rounding of the
//!   accumulation is not, so it records `ReductionOrder` and `DeviceArithmetic`, and a
//!   `half` form, whose fma rounds through `f32`, also `F16FmaDoubleRounding`; each step's rounding
//!   to `half` keeps half subnormals.

use std::borrow::Cow;

use crate::air::intrinsics::{AgxOp, Elem};
use crate::air::ir::{Call, Inst, InstId, Value, ValueId};
use crate::budget::Meter;
use crate::device::subgroup::OpBits;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::access::copy::{self, Delta, Endpoint};
use crate::lower::access::AccessPlan;
use crate::lower::arith::legal::{form_on, FloatFmt, Form, Held, LoweredConst};
use crate::lower::arith::{self, narrow, ops};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::{Leaf, Path};
use crate::pipeline::artifact::PrecisionNote;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::check_operands;
use super::group::{args, bin_k, k, lane, nth};
use super::matrix::{tile_mad, TileMad};

/// The refusal of an AGX operation without a modeled contract.
pub fn unmodeled() -> Error {
    Unsupported::not_implemented(Feature::AgxUnmodeled).into()
}

/// A selected AGX lane intrinsic.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgxStrategy {
    /// `llvm.agx3.edgecheck`, with the form of its `i16` result.
    EdgeCheck(Form),
    /// `llvm.agx2.cluster.num`: the constant 0 in its `i32` form.
    ClusterNum(Form),
    /// `llvm.agx2.shfl`: the bit interleave of two `i16`s.
    Shfl,
    /// `llvm.agx2.*matmad*`.
    MatMad(MatMad),
    /// `llvm.agx3.{load,store}.with.emask.*`.
    Emask(Emask),
}

/// A distributed-register matrix multiply-add.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMad {
    /// The `<2 x half>` or `<2 x float>` form of the operands and result.
    pub form: Form,
    /// 4 (four independent quadrants) or 8.
    pub dim: u8,
}

/// A lane-masked load or store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emask {
    /// Where the lanes live (a store's endpoint is written).
    pub end: Endpoint,
    /// The stored vector (stores only).
    pub value: Option<ValueId>,
    /// The `i16` lane mask.
    pub mask: ValueId,
    /// The form of the whole vector.
    pub form: Form,
    /// One lane's leaf.
    pub leaf: Leaf,
}

impl AgxStrategy {
    /// The census name in the finished plan `slots`: a lane-masked load or store qualified by what
    /// its endpoint reaches (`access::SideReach`).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            AgxStrategy::EdgeCheck(_) => "agx.edgecheck".into(),
            AgxStrategy::ClusterNum(_) => "agx.cluster_num".into(),
            AgxStrategy::Shfl => "agx.shfl".into(),
            AgxStrategy::MatMad(MatMad { dim: 4, .. }) => "agx.matmad4x4".into(),
            AgxStrategy::MatMad(_) => "agx.matmad8x8".into(),
            AgxStrategy::Emask(e) if e.value.is_some() => {
                e.end.reach(slots).qualify("agx.emask_store")
            }
            AgxStrategy::Emask(e) => e.end.reach(slots).qualify("agx.emask_load"),
        }
    }

    /// An upper estimate of the SPIR-V words one use emits: about six words per instruction with
    /// their constants, per lane the access with its containment and candidate switch, and per
    /// multiply-add step its owner arithmetic, three shuffles, and two fused multiply-adds with
    /// their roundings.
    pub fn words(&self) -> u32 {
        match self {
            AgxStrategy::EdgeCheck(_) => 4 * 48 + 32,
            AgxStrategy::ClusterNum(_) => 12,
            AgxStrategy::Shfl => 26 * 6 + 32,
            AgxStrategy::MatMad(m) => {
                let round = narrow::round_words(m.form);
                96 + u32::from(m.dim) * (72 + 2 * round)
            }
            AgxStrategy::Emask(e) => {
                let per = e.leaf.cost().saturating_add(64);
                let side = e.end.access_words(per).saturating_add(e.end.base_words());
                u32::from(e.form.lanes)
                    .saturating_mul(side.saturating_add(48))
                    .saturating_add(48)
            }
        }
    }
}

fn value_id(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("an AGX operand out of the value range"))
}

fn arg(cx: &SelectCx<'_>, call: &Call, i: usize) -> Result<Value, Error> {
    call.args
        .get(i)
        .copied()
        .ok_or_else(|| cx.invariant("an AGX call without its operands"))
}

/// Select the strategy of AGX call `inst` (`call`, operation `op`).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &Call,
    op: &AgxOp,
    m: &mut Meter,
) -> Result<AgxStrategy, Error> {
    if matches!(op, AgxOp::Unmodeled(_)) {
        return Err(unmodeled());
    }
    check_operands(cx, &call.args)?;
    match *op {
        AgxOp::EdgeCheck | AgxOp::ClusterNum | AgxOp::Shfl => {
            let form = form_on(cx.entry, cx.profile, inst.ty)?;
            if !form.is_int() || form.lanes != 1 || form.container() != 32 {
                return Err(cx.invariant("an AGX integer result outside its signature"));
            }
            Ok(match op {
                AgxOp::EdgeCheck => AgxStrategy::EdgeCheck(form),
                AgxOp::ClusterNum => AgxStrategy::ClusterNum(form),
                _ => AgxStrategy::Shfl,
            })
        }
        AgxOp::MatMad { elem, dim } => select_matmad(cx, at, inst, call, elem, dim),
        AgxOp::LoadEmask { ty, .. } | AgxOp::StoreEmask { ty, .. } => {
            let load = matches!(op, AgxOp::LoadEmask { .. });
            select_emask(cx, inst, call, load, ty.lanes, ty.elem.bits() / 8, m)
        }
        AgxOp::Unmodeled(_) => Err(unmodeled()),
    }
}

fn select_matmad(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &Call,
    elem: Elem,
    dim: u8,
) -> Result<AgxStrategy, Error> {
    let form = form_on(cx.entry, cx.profile, inst.ty)?;
    let fmt = match elem {
        Elem::F16 => FloatFmt::Half,
        _ => FloatFmt::F32,
    };
    if form.float_fmt() != Some(fmt) || form.lanes != 2 {
        return Err(cx.invariant("an AGX matrix multiply-add outside its signature"));
    }
    let id = cx
        .values
        .inst(at)
        .ok_or_else(|| cx.invariant("an instruction out of the value range"))?;
    cx.precision.insert(PrecisionNote::ReductionOrder);
    cx.precision.insert(PrecisionNote::DeviceArithmetic);
    // The steps are `f32` `Fma`s, as the matrix form's.
    arith::fp_env(cx, call.fmf, form.with(Held::Float(FloatFmt::F32)));
    if fmt == FloatFmt::Half {
        cx.precision.insert(PrecisionNote::F16FmaDoubleRounding);
    }
    let simd = &mut cx.families.simd;
    simd.direct_ops = simd.direct_ops.union(OpBits::SHUFFLE);
    simd.cluster_ops = simd.cluster_ops.union(OpBits::SHUFFLE);
    simd.groups = true;
    simd.observes_lanes = true;
    simd.simdgroup_members = true;
    cx.tangled.insert(id);
    Ok(AgxStrategy::MatMad(MatMad { form, dim }))
}

fn select_emask(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &Call,
    load: bool,
    lanes: u32,
    bytes: u32,
    m: &mut Meter,
) -> Result<AgxStrategy, Error> {
    let ptr = arg(cx, call, 0)?;
    let (value, mask, shape) = if load {
        (None, arg(cx, call, 1)?, arg(cx, call, 2)?)
    } else {
        (
            Some(arg(cx, call, 1)?),
            arg(cx, call, 2)?,
            arg(cx, call, 3)?,
        )
    };
    // The shape mask is the constant of the vector's shape; the backend rejects any other.
    let shape_ok = matches!(
        cx.constant(shape),
        Some(LoweredConst::Scalar { bits, .. }) if bits == (1u64 << lanes) - 1
    );
    if !shape_ok {
        return Err(unmodeled());
    }
    let ty = match value {
        None => inst.ty,
        Some(v) => cx
            .value_ty(v)
            .ok_or_else(|| cx.invariant("an emask value without a type"))?,
    };
    let form = form_on(cx.entry, cx.profile, ty)?;
    if !form.is_int() || u32::from(form.lanes) != lanes || u32::from(form.bits()) != 8 * bytes {
        return Err(cx.invariant("an emask value outside its signature"));
    }
    // Lane `i` is at `ptr + i · bytes`: each lane is as aligned as the pointer, at most to its
    // own size.
    let align = copy::arg_align(call, 0).min(u64::from(bytes));
    let path = if bytes == 4 && align >= 4 {
        Path::Aligned
    } else if bytes <= 2 && align >= u64::from(bytes) {
        Path::OneWord
    } else {
        Path::Bytes
    };
    let leaf = Leaf {
        form: Form { lanes: 1, ..form },
        bytes,
        path,
    };
    let end = copy::endpoint(cx, ptr, !load, m)?;
    Ok(AgxStrategy::Emask(Emask {
        end,
        value: value.map(|v| value_id(cx, v)).transpose()?,
        mask: value_id(cx, mask)?,
        form,
        leaf,
    }))
}

/// Emit strategy `s` of call `at` and bind its result.
pub(super) fn emit<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &AgxStrategy,
) -> BResult<()> {
    let result = cx.plan.values.inst(at);
    let x = match s {
        AgxStrategy::EdgeCheck(_) => {
            let a = args(cx, fb, at)?;
            edgecheck(fb, nth(&a, 0)?, nth(&a, 1)?, nth(&a, 2)?)?
        }
        AgxStrategy::ClusterNum(form) => ops::splat(fb, *form, 0)?,
        AgxStrategy::Shfl => {
            let a = args(cx, fb, at)?;
            let even = spread(fb, nth(&a, 0)?)?;
            let odd = spread(fb, nth(&a, 1)?)?;
            let odd = bin_k(fb, BinOp::ShiftLeftLogical, odd, 1)?;
            fb.binary(BinOp::BitwiseOr, even, odd)?
        }
        AgxStrategy::MatMad(m) => {
            let a = args(cx, fb, at)?;
            matmad(cx, fb, at, m, &a)?
        }
        AgxStrategy::Emask(e) => match emask(cx, fb, e)? {
            Some(x) => x,
            None => return Ok(()),
        },
    };
    let Some(result) = result else {
        return Err(cx.fail("an AGX result out of the value range"));
    };
    cx.define(result, x);
    Ok(())
}

/// `edgecheck(base, low, count)`: bit `i` for `i < 4` when `base + i >= 0` (signed; written
/// `base >= -i`, which cannot overflow) and `(base + i - low) < count` (unsigned).
fn edgecheck(
    fb: &mut FunctionBuilder<'_>,
    base: SValue,
    low: SValue,
    count: SValue,
) -> BResult<SValue> {
    let zero = k(fb, 0)?;
    let mut acc = zero;
    for i in 0..4u32 {
        let lane = if i == 0 {
            base
        } else {
            bin_k(fb, BinOp::IAdd, base, i)?
        };
        let neg = k(fb, 0u32.wrapping_sub(i))?;
        let nonneg = fb.binary(BinOp::SGreaterThanEqual, base, neg)?;
        let rel = fb.binary(BinOp::ISub, lane, low)?;
        let within = fb.binary(BinOp::ULessThan, rel, count)?;
        let on = fb.binary(BinOp::LogicalAnd, nonneg, within)?;
        let bit = k(fb, 1 << i)?;
        let bits = fb.select(on, bit, zero)?;
        acc = if i == 0 {
            bits
        } else {
            fb.binary(BinOp::BitwiseOr, acc, bits)?
        };
    }
    Ok(acc)
}

/// The 16 bits of `x` (an `i16` held zero-extended) spread to the even bits of a word, bit `i` to
/// bit `2i`: each step moves the upper half of every group of bits up by the step, and the mask
/// keeps the moved halves and the lower ones.
fn spread(fb: &mut FunctionBuilder<'_>, mut x: SValue) -> BResult<SValue> {
    for (step, mask) in [
        (8, 0x00FF_00FF),
        (4, 0x0F0F_0F0F),
        (2, 0x3333_3333),
        (1, 0x5555_5555),
    ] {
        let up = bin_k(fb, BinOp::ShiftLeftLogical, x, step)?;
        let both = fb.binary(BinOp::BitwiseOr, x, up)?;
        x = bin_k(fb, BinOp::BitwiseAnd, both, mask)?;
    }
    Ok(x)
}

/// `c + a·b` of the distributed tile ([`super::matrix::tile_mad`]), each step rounded to the
/// cell format.
fn matmad(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    m: &MatMad,
    a: &[SValue],
) -> BResult<SValue> {
    let (x, y, z) = (nth(a, 0)?, nth(a, 1)?, nth(a, 2)?);
    let ln = lane(cx, fb)?;
    let parts = |fb: &mut FunctionBuilder<'_>, v| -> BResult<[SValue; 2]> {
        Ok([
            fb.composite_extract(v, &[0])?,
            fb.composite_extract(v, &[1])?,
        ])
    };
    let (xs, ys, zs) = (parts(fb, x)?, parts(fb, y)?, parts(fb, z)?);
    let how = TileMad {
        dim: u32::from(m.dim),
        step: Form { lanes: 1, ..m.form },
        divergent: super::partly_live_at(cx, at),
    };
    let acc = tile_mad(fb, &ln, &how, xs, ys, zs)?;
    let ty = ops::ty(fb, m.form)?;
    fb.composite_construct(ty, &acc)
}

/// A lane-masked access: each lane in a selection on its mask bit. A load's value (an inactive
/// lane reads zero), or `None` for a store.
fn emask<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    e: &Emask,
) -> BResult<Option<SValue>> {
    let mask = cx.operand(fb, e.mask)?;
    let stored = match e.value {
        Some(v) => Some(cx.operand(fb, v)?),
        None => None,
    };
    let base = copy::base(cx, fb, &e.end)?;
    let zero = ops::splat(fb, e.leaf.form, 0)?;
    let none = k(fb, 0)?;
    let mut parts = Vec::with_capacity(usize::from(e.form.lanes));
    for i in 0..u32::from(e.form.lanes) {
        let bit = bin_k(fb, BinOp::BitwiseAnd, mask, 1 << i)?;
        let on = fb.binary(BinOp::INotEqual, bit, none)?;
        let d = Delta::Const(i * e.leaf.bytes);
        let leaf = e.leaf;
        match stored {
            Some(v) => {
                let x = fb.composite_extract(v, &[i])?;
                let t: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
                    copy::write(cx, fb, base, d, leaf, x)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                fb.if_(cx, on, t, None)?;
            }
            None => {
                let t: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
                    let x = copy::read(cx, fb, base, d, leaf)?;
                    Ok(ArmEnd::FallThrough(vec![x]))
                });
                let f: ArmFn<'_, EmitCx<'c>> =
                    Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![zero])));
                let Merged(m) = fb.if_(cx, on, t, Some(f))?;
                parts.push(nth(&m, 0)?);
            }
        }
    }
    if stored.is_some() {
        return Ok(None);
    }
    let ty = ops::ty(fb, e.form)?;
    fb.composite_construct(ty, &parts).map(Some)
}
