//! Metal's precise division, by the device's `OpFDiv`.
//!
//! Metal's `float` division without `arcp` (a flagless `fdiv`, `precise::divide`, and the plain
//! operator under `-fno-fast-math`) answers the quotient rounded once to nearest even (a probe
//! cases 914cc4d2 and 711cc190, every lane), and its `half` division under every flag set the
//! quotient rounded once from the exact one (4c363561, 54f0fe8e). Vulkan bounds `OpFDiv` at 2.5
//! ulp for a divisor in `[2^-126, 2^126]` and not at all beyond, and devices use that latitude:
//! RADV answers `3 / 3` with `0x3f7fffff` and a divisor past `2^126` with a zero. Every such
//! division takes the device's `OpFDiv` ([`native`]), its divisor kept in Vulkan's range, within
//! Vulkan's bound (`PrecisionNote::NativeDivision`): a `half` one the 16-bit instruction where the
//! module computes in 16 bits under the 16-bit float controls, the `float` one rounded otherwise.

use crate::air::ir::{BinOp as AirBin, FastMath};
use crate::lower::arith::legal::{FloatFmt, Form};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{f16_splat, float_ty, fsplat};
use crate::spirv::builder::{FunctionBuilder, QuantizeForm};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp};

/// Whether the float binary operator `op` with fast-math flags `fmf` on format `fmt` is Metal's
/// precise division: an `fdiv` on `float` without `arcp`, or on `half` under every flag set (the
/// module's comment).
pub fn precise(op: AirBin, fmt: FloatFmt, fmf: FastMath) -> bool {
    op == AirBin::FDiv
        && match fmt {
            FloatFmt::F32 => !fmf.contains(FastMath::ARCP),
            FloatFmt::Half => true,
            FloatFmt::BFloat | FloatFmt::F64 => false,
        }
}

/// The census name of a precise division ([`native`]).
pub const NATIVE: &str = "arith.fdiv.native";

/// An upper estimate of the SPIR-V words of one [`native`] division on form `f`: the divisor's
/// magnitude, its tests, the scale's selects, two products, the division, and a `half` result's
/// rounding, or the 16-bit form's conversions and constants.
pub fn native_words(f: Form) -> u32 {
    96 + 16 * u32::from(f.lanes)
}

/// Metal's precise division `x / y` on `float` or `half` lanes (form `f`): the device's `OpFDiv`
/// (`PrecisionNote::NativeDivision`).
///
/// On `float`, within Vulkan's 2.5 ulp for a divisor in `[2^-126, 2^126]`, the operands of a
/// divisor past `2^126` first scaled by `2^-2` into that range (exactly, but where the dividend
/// becomes a denormal, whose quotient by such a divisor Metal flushes to zero anyway).
///
/// On `half`, in a module whose `half` roundings take the 16-bit float controls
/// ([`QuantizeForm::Controlled`]), the device's 16-bit `OpFDiv` ([`native_16`]); otherwise the
/// `float` quotient of the widened operands, as on `float`, rounded to `half`.
pub fn native(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    match f.float_fmt() {
        Some(FloatFmt::Half) if fb.module().quantize_form() == QuantizeForm::Controlled => {
            native_16(fb, f, x, y)
        }
        Some(FloatFmt::F32) => scaled(fb, f, x, y),
        Some(FloatFmt::Half) => {
            let q = scaled(fb, f, x, y)?;
            narrow::round(fb, f, q)
        }
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// `float` `x / y` on form `f`'s container, the operands of a divisor past `2^126` scaled by
/// `2^-2` ([`native`]).
fn scaled(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let ay = fb.glsl(GlslOp::FAbs, &[y])?;
    let limit = fsplat(fb, f, 2f64.powi(126))?;
    let past = fb.binary(BinOp::FOrdGreaterThan, ay, limit)?;
    let (quarter, one) = (fsplat(fb, f, 0.25)?, fsplat(fb, f, 1.0)?);
    let scale = fb.select(past, quarter, one)?;
    let x = fb.binary(BinOp::FMul, x, scale)?;
    let y = fb.binary(BinOp::FMul, y, scale)?;
    fb.binary(BinOp::FDiv, x, y)
}

/// `half` `x / y` (`half` values held in `float` lanes of form `f`) by the device's 16-bit
/// `OpFDiv`: each operand narrowed to 16 bits, exactly since it is a `half`, the quotient, and its
/// widening, exactly. Vulkan bounds the 16-bit `OpFDiv` at 2.5 ulp for a divisor in
/// `[2^-14, 2^14]` (or zero), so the operands of a divisor past `2^14` are first scaled by `2^-2`
/// and those of a denormal divisor by `2^10`, into that range: exactly, but where the dividend
/// scaled down becomes an inexact denormal (`|x| < 2^-12`), whose quotient by such a divisor is
/// below `2^-26` and rounds to the zero of its sign, or the dividend scaled up overflows
/// (`|x| >= 64`), whose quotient by such a divisor overflows too. The module computes in 16 bits
/// ([`narrow::computes_in_16_bits`]), so its entry declares the 16-bit float controls, under which
/// denormal operands and results are kept and zeros' signs, infinities and NaNs are IEEE 754's.
fn native_16(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let (h, w) = (float_ty(fb, f.lanes, 16)?, float_ty(fb, f.lanes, 32)?);
    let x = fb.convert(ConvOp::FConvert, x, h)?;
    let y = fb.convert(ConvOp::FConvert, y, h)?;
    let ay = fb.glsl(GlslOp::FAbs, &[y])?;
    // 2^14, 2^-14, 2^-2, 2^10 and 1.
    let (upper, lower) = (
        f16_splat(fb, f.lanes, 0x7400)?,
        f16_splat(fb, f.lanes, 0x0400)?,
    );
    let (down, up, one) = (
        f16_splat(fb, f.lanes, 0x3400)?,
        f16_splat(fb, f.lanes, 0x6400)?,
        f16_splat(fb, f.lanes, 0x3c00)?,
    );
    let past = fb.binary(BinOp::FOrdGreaterThan, ay, upper)?;
    let below = fb.binary(BinOp::FOrdLessThan, ay, lower)?;
    let scale = fb.select(below, up, one)?;
    let scale = fb.select(past, down, scale)?;
    let x = fb.binary(BinOp::FMul, x, scale)?;
    let y = fb.binary(BinOp::FMul, y, scale)?;
    let q = fb.binary(BinOp::FDiv, x, y)?;
    fb.convert(ConvOp::FConvert, q, w)
}
