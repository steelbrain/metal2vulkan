//! Fused multiply-add on `float` and `half`.
//!
//! Metal's `fma` (`air.fma`, `llvm.fma`) rounds the exact `a * b + c` once. Vulkan's
//! `GLSL.std.450 Fma` does not promise that (its precision is inherited from an `OpFMul`
//! followed by an `OpFAdd`). A `float` fused multiply-add is the device's `GLSL.std.450 Fma`
//! ([`Route::Native`]) on every profile, within Vulkan's precision rather than rounded once
//! (`PrecisionNote::NativeFma`).
//!
//! A `half` one takes [`Route::F16Native`] where the profile computes `half` under the 16-bit
//! float controls
//! ([`narrow::half_controls`]): the device's `Fma` on 16-bit floats between the same exact
//! conversions, which the module's 16-bit holding then removes (`spirv::half_hold`), so the
//! operands and the result stay 16-bit floats. Its precision is Vulkan's for `Fma`, inherited from
//! an `OpFMul` followed by an `OpFAdd`, each correctly rounded to `half` under the controls'
//! `RoundingModeRTE`, where Metal's rounds once: noted `PrecisionNote::NativeFma`, as
//! [`Route::Native`] is. Every other `half` one takes [`Route::Native`]: the device's `float`
//! `Fma` of the operands held in `float`, whose product is exact there, rounded to `half`.
//!
//! An extended-range blend computes its own correctly rounded multiply-adds ([`half_sum`],
//! [`float_fma_exact`]): the blend the device cannot do, in plain arithmetic.

use crate::device::DeviceProfile;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow::{self, u32_form};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp, Type};

/// How a fused multiply-add computes its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Route {
    /// A `half` fused multiply-add by the device's `Fma` on 16-bit floats under the 16-bit float
    /// controls: within Vulkan's precision rather than rounded once (`PrecisionNote::NativeFma`)
    /// ([`half_route`]).
    F16Native,
    /// The device's `GLSL.std.450 Fma` on the `float` lanes (a `half`'s held in `float`, its
    /// result rounded to `half`), within Vulkan's precision rather than rounded once
    /// (`PrecisionNote::NativeFma`).
    Native,
}

impl Route {
    /// Whether the route is the device's own `float` instruction (its `Fma`), at the device's
    /// denormals and NaN encoding.
    pub fn device_float(self) -> bool {
        self == Route::Native
    }
}

/// The census names of the fused multiply-add strategies (one per [`Route`]).
pub const F16_NATIVE: &str = "arith.fma.f16.native";
/// See [`F16_NATIVE`].
pub const NATIVE: &str = "arith.fma.native";

/// The `half` route profile `p` offers: [`Route::F16Native`] where the profile computes under the
/// 16-bit float controls (`narrow::half_controls`), else the device's `float` `Fma` rounded to
/// `half` ([`Route::Native`]).
pub fn half_route(p: &DeviceProfile) -> Route {
    match narrow::half_controls(p) {
        true => Route::F16Native,
        false => Route::Native,
    }
}

/// The routes a profile's fused multiply-adds take, by format ([`routes`]): a `float` one's is
/// [`Route::Native`] on every profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Routes {
    /// A `half` one's: [`half_route`].
    pub half: Route,
}

/// The routes profile `p` offers ([`half_route`]).
pub fn routes(p: &DeviceProfile) -> Routes {
    Routes {
        half: half_route(p),
    }
}

/// A fused multiply-add on floats of `form` (`float` or `half` lanes) by `route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fma {
    pub form: Form,
    pub route: Route,
}

impl Fma {
    /// The fused multiply-add on `form` by `routes` (a profile's [`routes`]): `half` by its
    /// `half` route, `float` by the device's `Fma`. `None` for any other form.
    pub fn by(routes: Routes, form: Form) -> Option<Fma> {
        let route = match form.float_fmt()? {
            FloatFmt::Half => routes.half,
            FloatFmt::F32 => Route::Native,
            FloatFmt::BFloat | FloatFmt::F64 => return None,
        };
        Some(Fma { form, route })
    }

    /// The census name.
    pub fn name(self) -> &'static str {
        match self.route {
            Route::F16Native => F16_NATIVE,
            Route::Native => NATIVE,
        }
    }

    /// An upper estimate of the SPIR-V words per use: the instruction; on [`Route::F16Native`] the
    /// three conversions, the instruction and the widening.
    pub fn words(self) -> u32 {
        match self.route {
            Route::F16Native => 32,
            // The instruction, and a `half` result's rounding.
            Route::Native => 64,
        }
    }
}

/// Emit `fma` on `args` (three floats of `fma.form`). The result is a value of the form's format.
pub fn emit(fb: &mut FunctionBuilder<'_>, fma: Fma, args: &[Value]) -> BResult<Value> {
    let [a, b, c] = *args else {
        return Err(BuildError::Type(TypeRule::FunctionSignature));
    };
    match (fma.route, fma.form.float_fmt()) {
        (Route::F16Native, Some(FloatFmt::Half)) => f16(fb, fma.form.lanes, [a, b, c]),
        // The device's `Fma`; on `half` operands held in `float` the product is exact, so only
        // the sum's rounding and the rounding to `half` differ from Metal's one rounding.
        (Route::Native, Some(FloatFmt::F32)) => fb.glsl(GlslOp::Fma, &[a, b, c]),
        (Route::Native, Some(FloatFmt::Half)) => {
            let r = fb.glsl(GlslOp::Fma, &[a, b, c])?;
            narrow::round(fb, fma.form, r)
        }
        _ => Err(BuildError::Type(TypeRule::FunctionSignature)),
    }
}

/// The `half` fused multiply-add of `a`, `b`, `c` (`half` values held in `float` lanes of
/// `lanes` lanes) by the device's `Fma` on 16-bit floats ([`Route::F16Native`]): each operand
/// converted to 16 bits, exactly since it is a `half`, the instruction, and its result widened,
/// exactly, within Vulkan's precision for `Fma`, noted.
fn f16(fb: &mut FunctionBuilder<'_>, lanes: u8, [a, b, c]: [Value; 3]) -> BResult<Value> {
    let lanes_of = |fb: &mut FunctionBuilder<'_>, width: u8| -> BResult<_> {
        let e = fb.module().ty(Type::Float { width })?;
        match lanes {
            1 => Ok(e),
            len => fb.module().ty(Type::Vector { elem: e, len }),
        }
    };
    let (h, w) = (lanes_of(fb, 16)?, lanes_of(fb, 32)?);
    let narrow = |fb: &mut FunctionBuilder<'_>, x: Value| fb.convert(ConvOp::FConvert, x, h);
    let (a, b, c) = (narrow(fb, a)?, narrow(fb, b)?, narrow(fb, c)?);
    let r = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    fb.convert(ConvOp::FConvert, r, w)
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// Float arithmetic decorated `NoContraction`: rounded on its own, never fused or regrouped.
fn exact_op(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary_with(op, a, b, Contraction::Forbidden)
}

/// `s = x + y` (floats of form `f`) and its rounding error `e`, exactly ([`sum_error`]), for
/// finite `x` and `y` whose sum does not overflow.
fn two_sum(fb: &mut FunctionBuilder<'_>, x: Value, y: Value) -> BResult<(Value, Value)> {
    let s = exact_op(fb, BinOp::FAdd, x, y)?;
    let e = sum_error(fb, s, x, y)?;
    Ok((s, e))
}

/// The rounding error of `s`, the sum of `x` and `y` rounded to nearest, exactly (FastTwoSum on
/// the operands ordered by magnitude: `t = s - big` is exact, and so is `small - t`), for finite
/// `x` and `y` whose sum does not overflow.
fn sum_error(fb: &mut FunctionBuilder<'_>, s: Value, x: Value, y: Value) -> BResult<Value> {
    let ax = fb.glsl(GlslOp::FAbs, &[x])?;
    let ay = fb.glsl(GlslOp::FAbs, &[y])?;
    let x_big = bin(fb, BinOp::FOrdGreaterThanEqual, ax, ay)?;
    let big = fb.select(x_big, x, y)?;
    let small = fb.select(x_big, y, x)?;
    let t = exact_op(fb, BinOp::FSub, s, big)?;
    exact_op(fb, BinOp::FSub, small, t)
}

/// `s` (floats of form `f`, bits of integer form `u`) moved to round to odd given its exact
/// rounding error `e`: where `e` is nonzero and `s`'s lowest significand bit is clear, the
/// neighbour on `e`'s side, whose lowest bit is set (consecutive encodings alternate it). `s`
/// is then never zero (an inexact sum is not), so the step stays within its sign. A NaN `e`
/// (from an infinite or NaN `s`) leaves `s`.
fn to_odd(fb: &mut FunctionBuilder<'_>, f: Form, u: Form, s: Value, e: Value) -> BResult<Value> {
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let sb = fb.convert(ConvOp::Bitcast, s, ut)?;
    let (zu, one) = (splat(fb, u, 0)?, splat(fb, u, 1)?);
    let low = bin(fb, BinOp::BitwiseAnd, sb, one)?;
    let even = bin(fb, BinOp::IEqual, low, zu)?;
    let zf = fsplat(fb, f, 0.0)?;
    let inexact = bin(fb, BinOp::FOrdNotEqual, e, zf)?;
    let s_neg = bin(fb, BinOp::FOrdLessThan, s, zf)?;
    let e_neg = bin(fb, BinOp::FOrdLessThan, e, zf)?;
    // The exact value lies beyond `s` (away from zero) when `e` has `s`'s sign.
    let grow = bin(fb, BinOp::LogicalEqual, s_neg, e_neg)?;
    let up = bin(fb, BinOp::IAdd, sb, one)?;
    let down = bin(fb, BinOp::ISub, sb, one)?;
    let moved = fb.select(grow, up, down)?;
    let fix = bin(fb, BinOp::LogicalAnd, inexact, even)?;
    let r = fb.select(fix, moved, sb)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The exact `a * b + c` of `half` values held in `float` (form `f`), rounded to odd in
/// `float`. The product `p` (two 11-bit significands, between `2^-48` and `2^32` in magnitude)
/// is exact in `float`, so the device's `Fma` is the exact sum rounded once, `s`, whether it
/// fuses or not; `s`'s exact error is FastTwoSum's on `p` and `c` ([`sum_error`]), and
/// [`to_odd`] keeps the rounding direction. An infinite or NaN `s` (only from such an operand)
/// is `Fma`'s, unmoved, decided on its bits. Rounded to `half` to nearest even, it is the `half`
/// fused multiply-add correctly rounded (an extended-range blend's).
///
/// Both choices keep the result off two renderings of SPIRV-Cross, which MoltenVK uses (`f7850ba7`
/// lanes 2, 3, and 7): a `NoContraction` `OpFMul` becomes `fma(l, r, 0)`, which answers `+0` for a
/// `-0` product, so `s`'s signed zeros are `Fma`'s, never `p`'s (a zero `p` leaves the error zero);
/// and `OpFOrdNotEqual` becomes the unordered `!=`, true for the NaN error of an infinite `s`, so
/// [`to_odd`]'s move is not taken there.
pub(crate) fn half_sum(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let wide = f.with(Held::Float(FloatFmt::F32));
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let p = exact_op(fb, BinOp::FMul, a, b)?;
    let e = sum_error(fb, s, p, c)?;
    let odd = to_odd(fb, wide, u32_form(f), s, e)?;
    let finite = all_finite(fb, wide, &[s])?;
    fb.select(finite, odd, s)
}

/// `a * b + c` of finite `float`s (lanes of form `f`) rounded once to `float`, nearest even, in
/// plain `float` arithmetic, whatever the device's `Fma` fuses: Dekker's exact product
/// `a * b = ph + pl` (each factor split at 12 bits by Veltkamp's constant, every step
/// `NoContraction`), then Boldo and Melquiond's emulation of a fused multiply-add through rounding
/// to odd (IEEE TC 57(4), 2008): `(th, tl) = TwoSum(c, ph)`, `v = RO(tl + pl)` ([`half_sum`]'s
/// odd sum, its product `tl · 1` exact), and `RN(th + v)`. Exact where no part underflows: an
/// extended-range blend of a `float4` output, whose operands are at most 2 in magnitude and whose
/// result is rounded to `half` next, far above a part that small. An infinite or NaN result (an
/// infinite or NaN operand's: the operands lie far below `float`'s overflow) is the device's
/// `Fma`'s, decided on its bits, as [`half_sum`]'s is: an infinity of the exact sum's sign, or a
/// NaN, whose encoding the caller chooses.
///
/// A zero result has the sign IEEE 754-2008 (6.3) gives a fused multiply-add's exact zero, put on
/// the bits as an integer ([`zero_sign`]). The emulation's own zero has no defined sign: `th + v`
/// adds a `+0` `v` to a `-0` `th` (`-0 · 1 + -0`), whose sum is `+0` under roundTiesToEven;
/// SPIRV-Cross renders the `NoContraction` product `ph` as `fma(a, b, 0)`, a `+0` for a `-0`
/// product ([`half_sum`]); and a module without `SignedZeroInfNanPreserve` lets the device ignore
/// the sign of a zero in any float instruction (Vulkan's Precision and Operation of SPIR-V
/// Instructions). `OpBitcast` keeps the bits, and the integer instructions are exact.
pub(crate) fn float_fma_exact(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let (ah, al) = veltkamp(fb, f, a)?;
    let (bh, bl) = veltkamp(fb, f, b)?;
    let ph = exact_op(fb, BinOp::FMul, a, b)?;
    let hh = exact_op(fb, BinOp::FMul, ah, bh)?;
    let t = exact_op(fb, BinOp::FSub, hh, ph)?;
    let hl = exact_op(fb, BinOp::FMul, ah, bl)?;
    let t = exact_op(fb, BinOp::FAdd, t, hl)?;
    let lh = exact_op(fb, BinOp::FMul, al, bh)?;
    let t = exact_op(fb, BinOp::FAdd, t, lh)?;
    let ll = exact_op(fb, BinOp::FMul, al, bl)?;
    let pl = exact_op(fb, BinOp::FAdd, t, ll)?;
    let (th, tl) = two_sum(fb, c, ph)?;
    let one = fsplat(fb, f, 1.0)?;
    let v = half_sum(fb, f.with(Held::Float(FloatFmt::Half)), tl, one, pl)?;
    let r = exact_op(fb, BinOp::FAdd, th, v)?;
    let r = zero_sign(fb, f, [a, b, c], r)?;
    // An infinite or NaN result, only from such an operand, is the device's `Fma`'s, which IEEE 754
    // gives exactly: the emulation's own steps subtract an infinity from itself there, a NaN.
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let finite = all_finite(fb, f, &[s])?;
    fb.select(finite, r, s)
}

/// `r`, a multiply-add of `a`, `b`, `c` (`float` lanes of form `f`) rounded once, with a zero
/// result given IEEE 754-2008's sign (6.3): where the exact `a · b + c` is zero, a fused
/// multiply-add's zero is the sign of that sum, the product signed by the rules for
/// multiplication, so under roundTiesToEven `-0` only where the product's sign (`a`'s and `b`'s
/// differing) and `c`'s are both negative, and `+0` otherwise: opposite zeros, or a product that
/// `c` cancels, whose signs differ. The test and the sign are on the bits, as integers
/// (`OpBitcast`, `OpBitwiseAnd`, `OpBitwiseXor`, `OpIEqual`, and `OpSelect` of integers), so the
/// sign is the same whatever the device's float modes.
fn zero_sign(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [a, b, c]: [Value; 3],
    r: Value,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let [ab, bb, cb, rb] = [a, b, c, r].map(|x| fb.convert(ConvOp::Bitcast, x, ut));
    let (ab, bb, cb, rb) = (ab?, bb?, cb?, rb?);
    let (mag, sign, z) = (
        splat(fb, u, 0x7FFF_FFFF)?,
        splat(fb, u, 0x8000_0000)?,
        splat(fb, u, 0)?,
    );
    let m = bin(fb, BinOp::BitwiseAnd, rb, mag)?;
    let zero = bin(fb, BinOp::IEqual, m, z)?;
    let p = bin(fb, BinOp::BitwiseXor, ab, bb)?;
    let both = bin(fb, BinOp::BitwiseAnd, p, cb)?;
    let signed = bin(fb, BinOp::BitwiseAnd, both, sign)?;
    let w = fb.select(zero, signed, rb)?;
    fb.convert(ConvOp::Bitcast, w, ft)
}

/// `x`'s high 12 significand bits and the rest, `x = hi + lo` exactly (Veltkamp's split by
/// `2^12 + 1`), for finite `x` far below `float`'s overflow.
fn veltkamp(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<(Value, Value)> {
    let k = fsplat(fb, f, 4097.0)?;
    let t = exact_op(fb, BinOp::FMul, k, x)?;
    let u = exact_op(fb, BinOp::FSub, t, x)?;
    let hi = exact_op(fb, BinOp::FSub, t, u)?;
    let lo = exact_op(fb, BinOp::FSub, x, hi)?;
    Ok((hi, lo))
}

/// Whether every lane of `float` values `xs` (form `f`) is finite, from their bits (exact
/// whatever the device does with infinities and NaNs).
fn all_finite(fb: &mut FunctionBuilder<'_>, f: Form, xs: &[Value]) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let mut all: Option<Value> = None;
    for x in xs {
        let b = fb.convert(ConvOp::Bitcast, *x, ut)?;
        let e = bin(fb, BinOp::BitwiseAnd, b, exp)?;
        let fin = bin(fb, BinOp::INotEqual, e, exp)?;
        all = Some(match all {
            Some(v) => bin(fb, BinOp::LogicalAnd, v, fin)?,
            None => fin,
        });
    }
    all.ok_or(BuildError::Type(TypeRule::FunctionSignature))
}
