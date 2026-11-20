//! A count over a divisor as an `f32`, correctly rounded with integer operations alone, and the
//! per-lane constants it reads.
//!
//! The math intrinsics' normalized-integer conversions, the vertex fetch's normalized formats,
//! and the sRGB decode steps emit it at compile time. It is a file of its own so that it is among
//! the runtime's inputs (validation's `build.rs`) without the math intrinsics, which only the
//! compile emits.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{elem_ty, ones, splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp};

/// A constant of form `f` whose lane `i` is `bits[i]`: one value for every lane is a splat
/// ([`splat`]).
pub(super) fn per_lane(fb: &mut FunctionBuilder<'_>, f: Form, bits: &[u64]) -> BResult<Value> {
    match bits {
        [first, rest @ ..] if rest.iter().all(|b| b == first) => return splat(fb, f, *first),
        _ => {}
    }
    if bits.len() != usize::from(f.lanes) {
        return Err(BuildError::Operand {
            op: crate::spv::Op::ExtInst,
            rule: OperandRule::Arity,
        });
    }
    let e = elem_ty(fb, f.held)?;
    let mut parts = Vec::with_capacity(bits.len());
    for b in bits {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(b & ones(f.container())),
        })?);
    }
    let t = ty(fb, f)?;
    let c = fb.module().constant(Constant::Composite {
        ty: t,
        parts: parts.into(),
    })?;
    fb.module().const_value(c)
}

/// `k / d` correctly rounded to `f32`, for integers `0 <= k <= d` (`u32` lanes of form `u`) and
/// a divisor `1 <= d < 2^16`, with integer operations only, so it is exact on every device and
/// under every fast-math mode: Metal's normalized-integer conversions (`c / 255`, correctly rounded
/// on every byte of A1's `srgb-decode-of-every-byte`, bcb3682b, where a device division is one ULP
/// off on about half of them) and its sRGB decode steps (`k / 4095`).
///
/// `k = 0` is `+0`. Otherwise `k` is shifted left until `a = k * 2^e` lies in `[d, 2d)`
/// (`e >= 0`: `k <= d`), so `k / d = (a / d) * 2^-e` with `a / d` in `[1, 2)`. The significand
/// `q = floor(a * 2^23 / d)` (24 bits, its leading one included) is two exact integer divisions
/// (`a * 2^12 < 2^29`, then the remainder times `2^11`), and the last remainder rounds it to
/// nearest even. The biased exponent is `127 - e`, so the bits are `((126 - e) << 23) + q` plus
/// the rounding, which carries into the exponent when `q` is all ones (`k = d` gives exactly
/// `1.0`).
pub fn div_exact(fb: &mut FunctionBuilder<'_>, u: Form, k: Value, d: u32) -> BResult<Value> {
    div_exact_lanes(fb, u, k, &[d])
}

/// [`div_exact`] with lane `i` of `k` divided by `d[i]` (one divisor for every lane when `d` has
/// one): the sequence is lane-wise, its divisor and the divisor's top bit per-lane constants, so
/// each lane is its own [`div_exact`].
pub fn div_exact_lanes(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    k: Value,
    d: &[u32],
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let dl: Vec<u64> = d.iter().map(|d| u64::from(*d)).collect();
    let dk = per_lane(fb, u, &dl)?;
    let zero = fb.binary(BinOp::IEqual, k, z)?;
    // Any k >= 1 in its place for 0, whose answer is selected below.
    let k = fb.glsl(GlslOp::UMax, &[k, one])?;
    let m = fb.glsl(GlslOp::FindUMsb, &[k])?;
    let top: Vec<u64> = d
        .iter()
        .map(|d| u64::from(31 - d.leading_zeros()))
        .collect();
    let top = per_lane(fb, u, &top)?;
    let e = fb.binary(BinOp::ISub, top, m)?;
    let a = fb.binary(BinOp::ShiftLeftLogical, k, e)?;
    let short = fb.binary(BinOp::ULessThan, a, dk)?;
    let a2 = fb.binary(BinOp::ShiftLeftLogical, a, one)?;
    let a = fb.select(short, a2, a)?;
    let e1 = fb.binary(BinOp::IAdd, e, one)?;
    let e = fb.select(short, e1, e)?;
    let (k11, k12) = (c(fb, 11)?, c(fb, 12)?);
    let hi = fb.binary(BinOp::ShiftLeftLogical, a, k12)?;
    let q1 = fb.binary(BinOp::UDiv, hi, dk)?;
    let t = fb.binary(BinOp::IMul, q1, dk)?;
    let r1 = fb.binary(BinOp::ISub, hi, t)?;
    let lo = fb.binary(BinOp::ShiftLeftLogical, r1, k11)?;
    let q2 = fb.binary(BinOp::UDiv, lo, dk)?;
    let t = fb.binary(BinOp::IMul, q2, dk)?;
    let r2 = fb.binary(BinOp::ISub, lo, t)?;
    let q = fb.binary(BinOp::ShiftLeftLogical, q1, k11)?;
    let q = fb.binary(BinOp::BitwiseOr, q, q2)?;
    // Round to nearest even on the remainder.
    let twice = fb.binary(BinOp::ShiftLeftLogical, r2, one)?;
    let above = fb.binary(BinOp::UGreaterThan, twice, dk)?;
    let tie = fb.binary(BinOp::IEqual, twice, dk)?;
    let lsb = fb.binary(BinOp::BitwiseAnd, q, one)?;
    let odd = fb.binary(BinOp::INotEqual, lsb, z)?;
    let tie_up = fb.binary(BinOp::LogicalAnd, tie, odd)?;
    let up = fb.binary(BinOp::LogicalOr, above, tie_up)?;
    let inc = fb.select(up, one, z)?;
    let k126 = c(fb, 126)?;
    let be = fb.binary(BinOp::ISub, k126, e)?;
    let k23 = c(fb, 23)?;
    let be = fb.binary(BinOp::ShiftLeftLogical, be, k23)?;
    let bits = fb.binary(BinOp::IAdd, be, q)?;
    let bits = fb.binary(BinOp::IAdd, bits, inc)?;
    let bits = fb.select(zero, z, bits)?;
    let ft = ty(fb, u.with(Held::Float(FloatFmt::F32)))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}
