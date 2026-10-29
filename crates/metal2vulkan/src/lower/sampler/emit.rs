//! Emission of the sampling strategies.

use crate::lower::emit::EmitCx;
use crate::lower::texture::half::TexelOrigin;
use crate::lower::texture::{
    half_class, heap_element, held_texel, live_or_zero, resident_access, sparse_spec, tile_map,
    tiled_word, with_word, ImageRef,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::ops::image::ArmedOffset;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, DerivOp, GlslOp, ImageOperands, ImageQuery, MemAccess,
    Type as SType, UnOp,
};

use crate::air::ir::ValueId;
use crate::lower::texture::TexelClass;
use crate::memory::handles::{entry_field, EntryField};
use crate::metal::sampler::SamplerWord;

use super::{
    BiasReach, Binding, CombinedImage, CubeFootprint, Hoist, Lod, MinLevel, Offset, Pixel,
    SampleOp, Sampling, Source, Unmapped,
};

impl Binding {
    /// The sampled image of the binding, made in the open block (where it must be consumed): the
    /// combined image sampler, or `OpSampledImage` of the separate texture and sampler.
    pub fn sampled_image(self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        match self {
            Binding::Combined(g) => {
                let ptr = fb.module().global_value(g)?;
                fb.load(ptr, MemAccess::default())
            }
            Binding::Separate { image, sampler } => {
                let image = fb.module().global_value(image)?;
                let image = fb.load(image, MemAccess::default())?;
                let sampler = fb.module().global_value(sampler)?;
                let sampler = fb.load(sampler, MemAccess::default())?;
                fb.sampled_image(image, sampler)
            }
        }
    }

    /// The image of the binding: `OpImage` of the combined image sampler, or the separate
    /// texture.
    pub fn image(self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        match self {
            Binding::Combined(_) => {
                let si = self.sampled_image(fb)?;
                fb.image(si)
            }
            Binding::Separate { image, .. } => {
                let image = fb.module().global_value(image)?;
                fb.load(image, MemAccess::default())
            }
        }
    }
}

/// The lowest exponent [`bias_gradients`] scales by: `2^-150` rounds to zero (or the smallest
/// denormal), so every scaled component is at most `2^-149` and the level is below 0.
const BIAS_EXPONENT_MIN: f32 = -150.0;

/// The highest exponent [`bias_gradients`] scales by: the scaled pair's largest component is then
/// in `[2^125, 2^126)`, finite, a level of at least 125, past any texture's top level.
const BIAS_EXPONENT_MAX: f32 = 126.0;

fn f32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

/// A vector of `n` elements of `elem`, or `elem` itself for `n = 1`.
fn vec(fb: &mut FunctionBuilder<'_>, elem: TypeId, n: u8) -> BResult<TypeId> {
    if n == 1 {
        Ok(elem)
    } else {
        fb.module().ty(SType::Vector { elem, len: n })
    }
}

/// The signed 32-bit constant offset `k` (a scalar or vector).
pub(super) fn const_offset(fb: &mut FunctionBuilder<'_>, k: &[i32]) -> BResult<SValue> {
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let mut parts = Vec::with_capacity(k.len());
    for x in k {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: i32_,
            bits: ConstBits(u64::from(*x as u32)),
        })?);
    }
    let c = match parts.as_slice() {
        [one] => *one,
        _ => {
            let n = u8::try_from(parts.len()).unwrap_or(u8::MAX);
            let ty = vec(fb, i32_, n)?;
            fb.module().constant(Constant::Composite {
                ty,
                parts: parts.into(),
            })?
        }
    };
    fb.module().const_value(c)
}

/// `x / size`, lane-wise, for the plane coordinate (or gradient) `x` of `lanes` components and the
/// level-0 size `size` (a `uvec` whose first `lanes` components are the plane's).
fn scale(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    size: SValue,
    lanes: u8,
    size_lanes: u8,
) -> BResult<SValue> {
    let plane = match (lanes, size_lanes) {
        (l, s) if l == s => size,
        (1, _) => fb.composite_extract(size, &[0])?,
        (l, _) => {
            let idx: Vec<u32> = (0..u32::from(l)).collect();
            fb.vector_shuffle(size, size, &idx)?
        }
    };
    let f = f32_ty(fb)?;
    let ty = vec(fb, f, lanes)?;
    let fsize = fb.convert(ConvOp::ConvertUToF, plane, ty)?;
    fb.binary(BinOp::FDiv, x, fsize)
}

/// The gradients of a rule-2 sample (`Hoist`): the coordinate's derivatives in the open block,
/// scaled by `2^B` with Metal's whole reading of the bias (`metal_bias`, `bias_gradients`). The
/// arms' samplers differ, so an anisotropic one's minor axis is not told apart (the isotropic
/// route, `PrecisionNote::GradFromBias`).
pub fn hoisted_gradients(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    h: &Hoist,
) -> BResult<(SValue, SValue)> {
    let coord = cx.operand(fb, h.coord)?;
    let dx = fb.derivative(DerivOp::DPdx, coord)?;
    let dy = fb.derivative(DerivOp::DPdy, coord)?;
    let Some(bias) = h.bias else {
        return Ok((dx, dy));
    };
    let b = cx.operand(fb, bias)?;
    let b = metal_bias(fb, b)?;
    bias_gradients(fb, dx, dy, b, h.lanes)
}

/// The derivatives `(dx, dy)` of a `lanes`-float coordinate scaled by `2^bias`, the whole bias,
/// as Metal applies it (Vulkan's `Bias` operand stops at `maxSamplerLodBias`), finite for every
/// bias:
///
/// 1. `e`, the exponent of the pair's largest magnitude (`Frexp`, so `max |d| = m · 2^e` with
///    `m` in `[1/2, 1)`; 0 for a zero pair), and the pair brought to that range exactly
///    (`Ldexp(d, -e)`);
/// 2. `t = clamp(bias + e, -150, 126)`, and the normalized pair times `2^t` (`Exp2`).
///
/// Where `bias + e` is in range this is `(dx, dy) · 2^bias` (exactly for an integer bias, which
/// only moves exponents; `Exp2`'s precision otherwise), so the level is the derivatives' level
/// plus the bias. Past the range the level saturates where Metal's whole bias puts it, and the
/// pair holds no infinity (so no `0 · ∞` NaN in a zero component): at `t = 126` its largest
/// component is in `[2^125, 2^126)`, a level of at least 125, past any texture's top level, where
/// the unclamped level is too; at `t = -150` every component is zero or the smallest denormal, a
/// level below 0 (magnified), as the unclamped level is. A zero pair stays zero (magnified, as
/// `log2 0 + bias` is). A non-finite derivative is the device's, as the implicit sample's is.
pub fn bias_gradients(
    fb: &mut FunctionBuilder<'_>,
    dx: SValue,
    dy: SValue,
    bias: SValue,
    lanes: u8,
) -> BResult<(SValue, SValue)> {
    let f = f32_ty(fb)?;
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let ax = fb.glsl(GlslOp::FAbs, &[dx])?;
    let ay = fb.glsl(GlslOp::FAbs, &[dy])?;
    let pair = fb.glsl(GlslOp::FMax, &[ax, ay])?;
    let mut largest = if lanes == 1 {
        pair
    } else {
        fb.composite_extract(pair, &[0])?
    };
    for lane in 1..u32::from(lanes) {
        let x = fb.composite_extract(pair, &[lane])?;
        largest = fb.glsl(GlslOp::FMax, &[largest, x])?;
    }
    let parts = fb.glsl(GlslOp::FrexpStruct, &[largest])?;
    let e = fb.composite_extract(parts, &[1])?;
    let down = fb.unary(UnOp::SNegate, e)?;
    let down = if lanes == 1 {
        down
    } else {
        let ty = vec(fb, i32_, lanes)?;
        fb.composite_construct(ty, &vec![down; usize::from(lanes)])?
    };
    let nx = fb.glsl(GlslOp::Ldexp, &[dx, down])?;
    let ny = fb.glsl(GlslOp::Ldexp, &[dy, down])?;
    let ef = fb.convert(ConvOp::ConvertSToF, e, f)?;
    let t = fb.binary(BinOp::FAdd, bias, ef)?;
    let lo = f32_const(fb, BIAS_EXPONENT_MIN.to_bits())?;
    let hi = f32_const(fb, BIAS_EXPONENT_MAX.to_bits())?;
    let t = fb.glsl(GlslOp::FClamp, &[t, lo, hi])?;
    let scale = fb.glsl(GlslOp::Exp2, &[t])?;
    let scale = if lanes == 1 {
        scale
    } else {
        let ty = vec(fb, f, lanes)?;
        fb.composite_construct(ty, &vec![scale; usize::from(lanes)])?
    };
    Ok((
        fb.binary(BinOp::FMul, nx, scale)?,
        fb.binary(BinOp::FMul, ny, scale)?,
    ))
}

/// The largest magnitude [`metal_bias`] reads a bias at: past it Metal's reading is clamped to
/// its range whatever the bias, so the rounding below never leaves the `half` range.
const METAL_BIAS_REACH: f32 = 64.0;

/// Metal's reading `B` of fragment bias `b`, on the CPU: `b` rounded to the nearest `half` (ties to
/// even), floored to a multiple of `1/64`, and clamped into `[-32, 31.984375]`; a NaN stays a NaN
/// (the device's reading, unmeasured). [`metal_bias`] emits the same steps. Measured on an Apple
/// M4: every quad of five samplers' 32,768 samples biased from -40 to 40 at levels from
/// -1.5 to 11.5 reads level `lod + B`.
pub(super) fn metal_bias_of(b: f32) -> f32 {
    if b.is_nan() {
        return b;
    }
    let x = b.clamp(-METAL_BIAS_REACH, METAL_BIAS_REACH);
    let h = if x.abs() < HALF_MIN_NORMAL {
        // A `half` denormal: below `1/64` in magnitude, so only its sign matters, and it is
        // negative where `x` is below half the smallest one (a tie rounds to the even zero).
        if x < -HALF_MIN_DENORMAL / 2.0 {
            -HALF_MIN_DENORMAL
        } else {
            0.0
        }
    } else {
        f32::from_bits(half_rounded_bits(x.to_bits()))
    };
    ((h * 64.0).floor() / 64.0).clamp(METAL_BIAS_MIN, METAL_BIAS_MAX)
}

/// Metal's least and greatest bias.
const METAL_BIAS_MIN: f32 = -32.0;
const METAL_BIAS_MAX: f32 = 31.984375;

/// The smallest normal `half`, `2^-14`, and the smallest denormal one, `2^-24`.
const HALF_MIN_NORMAL: f32 = 6.103_515_6e-5;
const HALF_MIN_DENORMAL: f32 = 5.960_464_5e-8;

/// The bits of the `f32` of bits `bits`, a normal number within the `half` range, rounded to a
/// `half`'s precision, to nearest with ties to even: its 13 low mantissa bits dropped, rounding
/// up past half of them and on a tie where the kept bits are odd (a carry moves the exponent).
fn half_rounded_bits(bits: u32) -> u32 {
    let magnitude = bits & 0x7fff_ffff;
    let odd = (magnitude >> 13) & 1;
    ((magnitude + 0xfff + odd) & 0xffff_e000) | (bits & 0x8000_0000)
}

/// Emits [`metal_bias_of`] of the `f32` bias `b`, step for step, in integer and exact float
/// operations (no `OpQuantizeToF16`, whose rounding Vulkan leaves to the device): the clamp into
/// `±64`, the rounding of the bits to a `half`'s precision, the denormal range's sign, the floor at
/// `1/64` (a product by a power of two, exact), and the clamp into Metal's range; a NaN bias
/// selects itself.
pub(super) fn metal_bias(fb: &mut FunctionBuilder<'_>, b: SValue) -> BResult<SValue> {
    let f = f32_ty(fb)?;
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let fc = |fb: &mut FunctionBuilder<'_>, x: f32| f32_const(fb, x.to_bits());
    let uc = |fb: &mut FunctionBuilder<'_>, x: u32| -> BResult<SValue> {
        let c = fb.u32_const(x)?;
        fb.module().const_value(c)
    };
    let (lo, hi) = (fc(fb, -METAL_BIAS_REACH)?, fc(fb, METAL_BIAS_REACH)?);
    let x = fb.glsl(GlslOp::FClamp, &[b, lo, hi])?;
    // The rounding of the bits ([`half_rounded_bits`]).
    let bits = fb.convert(ConvOp::Bitcast, x, u)?;
    let mask = uc(fb, 0x7fff_ffff)?;
    let magnitude = fb.binary(BinOp::BitwiseAnd, bits, mask)?;
    let sign_mask = uc(fb, 0x8000_0000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_mask)?;
    let thirteen = uc(fb, 13)?;
    let kept = fb.binary(BinOp::ShiftRightLogical, magnitude, thirteen)?;
    let one = uc(fb, 1)?;
    let odd = fb.binary(BinOp::BitwiseAnd, kept, one)?;
    let below_half = uc(fb, 0xfff)?;
    let up = fb.binary(BinOp::IAdd, magnitude, below_half)?;
    let up = fb.binary(BinOp::IAdd, up, odd)?;
    let keep = uc(fb, 0xffff_e000)?;
    let rounded = fb.binary(BinOp::BitwiseAnd, up, keep)?;
    let rounded = fb.binary(BinOp::BitwiseOr, rounded, sign)?;
    let h = fb.convert(ConvOp::Bitcast, rounded, f)?;
    // The `half` denormal range: its sign alone.
    let magnitude_f = fb.glsl(GlslOp::FAbs, &[x])?;
    let min_normal = fc(fb, HALF_MIN_NORMAL)?;
    let small = fb.binary(BinOp::FOrdLessThan, magnitude_f, min_normal)?;
    let tie = fc(fb, -HALF_MIN_DENORMAL / 2.0)?;
    let negative = fb.binary(BinOp::FOrdLessThan, x, tie)?;
    let (denormal, zero) = (fc(fb, -HALF_MIN_DENORMAL)?, fc(fb, 0.0)?);
    let tiny = fb.select(negative, denormal, zero)?;
    let h = fb.select(small, tiny, h)?;
    // The floor at `1/64` and Metal's range.
    let (sixty_four, sixty_fourth) = (fc(fb, 64.0)?, fc(fb, 1.0 / 64.0)?);
    let steps = fb.binary(BinOp::FMul, h, sixty_four)?;
    let steps = fb.glsl(GlslOp::Floor, &[steps])?;
    let reading = fb.binary(BinOp::FMul, steps, sixty_fourth)?;
    let (min, max) = (fc(fb, METAL_BIAS_MIN)?, fc(fb, METAL_BIAS_MAX)?);
    let reading = fb.glsl(GlslOp::FClamp, &[reading, min, max])?;
    let nan = fb.unary(UnOp::IsNan, b)?;
    fb.select(nan, b, reading)
}

/// The gradients through which a device's anisotropic filter reads Metal's level of a fragment
/// sample of a 2D plane biased by Metal's `bias` (`B`, [`metal_bias`]) through an anisotropic
/// sampler: the coordinate's derivatives `dx`, `dy` (2 floats each) with the shorter, in texels of
/// the level-0 size `size` (a `uvec` whose first two components are the plane's), scaled by `2^B`
/// but held at least at a sixteenth of the longer, and the longer kept, raised to the shorter's new
/// length where that passes it.
///
/// Metal's level of such a sample is `max(log2 minor + B, log2 major - log2 maxAnisotropy)`, the
/// bias shifting the footprint's minor axis alone (an Apple M4: every one of 32,768 quads of a 16x
/// anisotropic trilinear sampler within 0.06 of a level, biases from -40 to 40, footprints to 32 to
/// 1; scaling both axes by `2^B` misses 15,791). A device's anisotropic level of the pair is
/// `max(log2 shorter, log2 longer - log2 maxAnisotropy)` with its own anisotropy, so the pair gives
/// Metal's level where `minor + B` stays below the major axis, and the isotropic `minor + B` past
/// it, as Metal's rule does. The floor at a sixteenth of the longer, Metal's greatest
/// `maxAnisotropy`, moves no level a device clamping the pair's ratio at its anisotropy (at
/// most 16) reads, and keeps the ratio within 16 for a device that mishandles a greater one: x86-64
/// lavapipe read levels up to 10 below Metal's on 1,343 of 32,768 quads, every one of a ratio past
/// 4,564, where its explicit-gradient samples of ratios up to 32 are within 0.08 of Metal's. A
/// shorter that vanishes there (at most `2^-100` of the floor) is replaced by the longer's
/// perpendicular in texels at the floor's length. `Exp2`'s precision and the device's level
/// arithmetic are the device's (`PrecisionNote::GradFromBias`).
pub(super) fn axis_gradients(
    fb: &mut FunctionBuilder<'_>,
    (dx, dy): (SValue, SValue),
    bias: SValue,
    size: SValue,
    size_lanes: u8,
) -> BResult<(SValue, SValue)> {
    scaled_axis_gradients(
        fb,
        (dx, dy),
        |fb| fb.glsl(GlslOp::Exp2, &[bias]),
        size,
        size_lanes,
    )
}

/// [`axis_gradients`] with the bias's scale `2^B` emitted by `scale` where the sequence first
/// reads it: the device's `Exp2` there, and in the interpreter's test, which leaves `Exp2`
/// unmodelled (its precision is the device's), an exact `2^B` read from the test's input.
fn scaled_axis_gradients(
    fb: &mut FunctionBuilder<'_>,
    (dx, dy): (SValue, SValue),
    scale: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<SValue>,
    size: SValue,
    size_lanes: u8,
) -> BResult<(SValue, SValue)> {
    let f = f32_ty(fb)?;
    let v2 = vec(fb, f, 2)?;
    let plane = match size_lanes {
        2 => size,
        _ => fb.vector_shuffle(size, size, &[0, 1])?,
    };
    let texels = fb.convert(ConvOp::ConvertUToF, plane, v2)?;
    let tx = fb.binary(BinOp::FMul, dx, texels)?;
    let ty = fb.binary(BinOp::FMul, dy, texels)?;
    let lx = fb.glsl(GlslOp::Length, &[tx])?;
    let ly = fb.glsl(GlslOp::Length, &[ty])?;
    let x_shorter = fb.binary(BinOp::FOrdLessThanEqual, lx, ly)?;
    let shorter_len = fb.glsl(GlslOp::FMin, &[lx, ly])?;
    let longer_len = fb.glsl(GlslOp::FMax, &[lx, ly])?;
    let bool_ = fb.module().ty(SType::Bool)?;
    let b2 = vec(fb, bool_, 2)?;
    let pick = fb.composite_construct(b2, &[x_shorter, x_shorter])?;
    let shorter = fb.select(pick, dx, dy)?;
    let longer = fb.select(pick, dy, dx)?;
    let scale = scale(fb)?;
    let scaled = fb.binary(BinOp::VectorTimesScalar, shorter, scale)?;
    let reach = fb.binary(BinOp::FMul, shorter_len, scale)?;
    // The floor: a sixteenth of the longer, where the scaled shorter falls below it, as the
    // unscaled shorter brought to that length, or, where it vanishes, the longer's perpendicular
    // in texels, `(-l_y · h / w, l_x · w / h) / 16`, whose texel length is the floor too.
    let sixteenth = f32_const(fb, (1.0f32 / 16.0).to_bits())?;
    let floor_len = fb.binary(BinOp::FMul, longer_len, sixteenth)?;
    let below = fb.binary(BinOp::FOrdLessThan, reach, floor_len)?;
    let tiny = f32_const(fb, 2f32.powi(-100).to_bits())?;
    let least = fb.binary(BinOp::FMul, floor_len, tiny)?;
    let vanishes = fb.binary(BinOp::FOrdLessThanEqual, shorter_len, least)?;
    let lift = fb.binary(BinOp::FDiv, floor_len, shorter_len)?;
    let lifted = fb.binary(BinOp::VectorTimesScalar, shorter, lift)?;
    let (w, h) = (
        fb.composite_extract(texels, &[0])?,
        fb.composite_extract(texels, &[1])?,
    );
    let (lx_, ly_) = (
        fb.composite_extract(longer, &[0])?,
        fb.composite_extract(longer, &[1])?,
    );
    let h_w = fb.binary(BinOp::FDiv, h, w)?;
    let w_h = fb.binary(BinOp::FDiv, w, h)?;
    let px = fb.binary(BinOp::FMul, ly_, h_w)?;
    let px = fb.unary(UnOp::FNegate, px)?;
    let py = fb.binary(BinOp::FMul, lx_, w_h)?;
    let perpendicular = fb.composite_construct(v2, &[px, py])?;
    let perpendicular = fb.binary(BinOp::VectorTimesScalar, perpendicular, sixteenth)?;
    let lifted = fb.select(vanishes, perpendicular, lifted)?;
    let shorter = fb.select(below, lifted, scaled)?;
    // The longer's raise: `max(1, shorter · 2^B / longer)`, 1 for a zero pair.
    let ratio = fb.binary(BinOp::FDiv, reach, longer_len)?;
    let one = f32_const(fb, 1f32.to_bits())?;
    let raise = fb.glsl(GlslOp::FMax, &[ratio, one])?;
    let zero = f32_const(fb, 0)?;
    let some = fb.binary(BinOp::FOrdGreaterThan, longer_len, zero)?;
    let raise = fb.select(some, raise, one)?;
    let longer = fb.binary(BinOp::VectorTimesScalar, longer, raise)?;
    Ok((
        fb.select(pick, shorter, longer)?,
        fb.select(pick, longer, shorter)?,
    ))
}

/// Metal's footprint of the explicit gradients `dx`, `dy` of cube direction `p`, in the form
/// `form` names: the gradients from which the device's rule reads it.
///
/// **Metal's rule** (an Apple M3 Max under macOS 27.0.1, the oracle; the Apple M4 under macOS 27
/// read alike, its earlier probes): Metal drops each gradient's component on the direction's major
/// axis `a` (the face normal), its share of the projection's quotient included, and reads the
/// face's two tangent components `g[s]` and `g[t]`, on the axes of the face's `s_c` and `t_c`
/// (Vulkan's "Cube Map Face Selection" table, which Metal's faces follow), over `2 |m_a|`. A
/// probe of 12,288 lanes a sampler (random directions on every face at `|m_a|` of 0.5, 1 and 2,
/// random gradients; isotropic, anisotropic of degree 2 to 16, nearest and linear mip filters, a
/// 64x64 `RGBA8Unorm` cube of seven levels) found the cube's sample, on every lane whose filter
/// stays inside the face, equal to the sample of a 2D array holding the faces at the face
/// coordinate with the face gradients `(σ_s g[s], σ_t g[t]) / (2 |m_a|)` (`σ` the signs of the
/// table's `s_c` and `t_c`) through an isotropic sampler, and through an anisotropic one with
/// `dPdx`'s `t` component and `dPdy`'s `s` component times `ρ = σ_s σ_t` (-1 on the `-x`, `-y` and
/// `+z` faces, whose ellipse Metal mirrors); every lane's sample with the normal component zeroed
/// equal to its own.
///
/// - [`CubeFootprint::Length`] (an isotropic sampler): each gradient's two tangent components
///   combined by Metal's length, `max(M + 3m/16, 49M/64 + 21m/32)` of the larger and smaller
///   magnitude (the anisotropic emulation's, measured on 2D textures), on the tangent `(a + 2) mod
///   3` alone. Vulkan's rule ("Cube Map Derivative Transformation") reads a gradient whose normal
///   component is zero as `∂s_c / (2 |m_a|)` and `∂t_c / (2 |m_a|)`, so the device reads Metal's
///   length whatever its approximation of a vector's (one component is nonzero); its level is its
///   own logarithm of it, where Metal's piecewise-linear one minifies only past a footprint of
///   `1 + 3/2048` (`PrecisionNote::CubeGradientFootprint`). Every known route reads the one
///   component alike, so a route not measured takes it too (but for a `y` face through
///   [`CubeFootprint::Projected`]'s fixup, which reads it on both tangents).
/// - [`CubeFootprint::Face`] (an anisotropic sampler): each gradient with its normal component
///   zeroed and Metal's mirroring applied, `(0, ρ g_y, g_z)` and `(0, g_y, ρ g_z)` on an `x` face,
///   `(g_x, 0, ρ g_z)` and `(ρ g_x, 0, g_z)` on a `y` face, `(g_x, ρ g_y, 0)` and `(ρ g_x, g_y, 0)`
///   on a `z` face (`dPdx`'s and `dPdy`'s), from which Vulkan's rule reads Metal's face gradients;
///   the device filters that ellipse by its own anisotropic rule.
/// - [`CubeFootprint::Projected`] (MoltenVK's graphics route under macOS 27,
///   `CubeGradientRoute::AppleProjected`): SPIRV-Cross's `spvGradientCube` (MoltenVK 1.4.1's
///   `adec7acb`) projects each gradient onto the face and writes the two values where Apple's
///   earlier reading took them: `(d_y, d_y, d_z)` on an `x` face, `(d_x, d_z, d_x)` on a `y` face
///   and `(d_x, d_y, d_x)` on a `z` face. Metal under macOS 27 reads the tangents of those, so on
///   an `x` or `z` face the gradient with its normal component zeroed reaches Metal as its own
///   (the probe: every lane of every sampler equal), and is handed so. On a `y` face Metal reads
///   `d_x` on both tangents, the `z` tangent lost: the function hands `(64/91 L, 0, 0)`, `L`
///   Metal's length of `(g_x, g_z)`, whose length on both tangents is `L` again, so an isotropic
///   sample keeps Metal's footprint up to that division's rounding (the probe: all 6,144 lanes of
///   each of three isotropic samplers equal, every face); an anisotropic one's ellipse lies on the
///   diagonal, 922 to 932 of a `y` face's 1,024 lanes differing
///   (`PrecisionNote::CubeGradientFootprint`).
/// - [`CubeFootprint::Metal`]: the gradients as the call states them, for a driver that hands them
///   to Metal's own `gradientcube` unchanged (`CubeGradientRoute::Metal`): exact, no note.
///
/// The major axis is `z` over `y` over `x` at equal magnitudes, which on a direction whose ties are
/// broken as Metal breaks them (`face::metal_ties`) is Metal's, there being none left. A footprint
/// its sampler's word chooses (`CubeFootprint::Word`) is [`word_footprint`]'s; it has no form of
/// its own here.
pub(super) fn cube_footprint(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    grads: (SValue, SValue),
    form: CubeFootprint,
) -> BResult<(SValue, SValue)> {
    match form {
        CubeFootprint::Metal => Ok(grads),
        CubeFootprint::Word => Err(BuildError::Operand {
            op: crate::spv::Op::Select,
            rule: OperandRule::Arity,
        }),
        CubeFootprint::Face => face_gradients(fb, p, grads),
        CubeFootprint::Length => tangent_length(fb, p, grads, false),
        CubeFootprint::Projected => tangent_length(fb, p, grads, true),
    }
}

/// The major axis of cube direction `p`, as `bool`s: `z` where `|z|` is at least `|x|` and `|y|`,
/// else `y` where `|y|` is at least `|x|`, else `x`.
pub(super) struct Major {
    pub(super) y: SValue,
    pub(super) z: SValue,
    pub(super) y_or_z: SValue,
}

impl Major {
    pub(super) fn of(fb: &mut FunctionBuilder<'_>, p: SValue) -> BResult<Major> {
        let a = fb.glsl(GlslOp::FAbs, &[p])?;
        let [ax, ay, az] = components(fb, a)?;
        let z_x = fb.binary(BinOp::FOrdGreaterThanEqual, az, ax)?;
        let z_y = fb.binary(BinOp::FOrdGreaterThanEqual, az, ay)?;
        let z = fb.binary(BinOp::LogicalAnd, z_x, z_y)?;
        let not_z = fb.unary(UnOp::LogicalNot, z)?;
        let y_x = fb.binary(BinOp::FOrdGreaterThanEqual, ay, ax)?;
        let y = fb.binary(BinOp::LogicalAnd, not_z, y_x)?;
        let y_or_z = fb.binary(BinOp::LogicalOr, y, z)?;
        Ok(Major { y, z, y_or_z })
    }
}

/// The three components of the `vec3` `g`.
pub(super) fn components(fb: &mut FunctionBuilder<'_>, g: SValue) -> BResult<[SValue; 3]> {
    Ok([
        fb.composite_extract(g, &[0])?,
        fb.composite_extract(g, &[1])?,
        fb.composite_extract(g, &[2])?,
    ])
}

/// [`CubeFootprint::Face`]: each gradient without its normal component, `dPdx`'s `t` component
/// and `dPdy`'s `s` component times `ρ`.
fn face_gradients(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    (dx, dy): (SValue, SValue),
) -> BResult<(SValue, SValue)> {
    let f = f32_ty(fb)?;
    let triple = vec(fb, f, 3)?;
    let m = Major::of(fb, p)?;
    let zero = f32_const(fb, 0)?;
    // `ρ` is -1 where `m_a` is negative on an `x` or `y` face, positive on a `z` face.
    let [px, py, pz] = components(fb, p)?;
    let pa = fb.select(m.y, py, px)?;
    let pa = fb.select(m.z, pz, pa)?;
    let negative = fb.binary(BinOp::FOrdLessThan, pa, zero)?;
    let flip = fb.binary(BinOp::LogicalNotEqual, negative, m.z)?;
    // `dx`'s `t` component takes `ρ`: `y` on an `x` or `z` face, `z` on a `y` face. `dy`'s `s`
    // component does: `z` on an `x` face, `x` on a `y` or `z` face.
    let face = |fb: &mut FunctionBuilder<'_>, g: SValue, is_dx: bool| -> BResult<SValue> {
        let [gx, gy, gz] = components(fb, g)?;
        let rho = |fb: &mut FunctionBuilder<'_>, c: SValue| -> BResult<SValue> {
            let n = fb.unary(UnOp::FNegate, c)?;
            fb.select(flip, n, c)
        };
        let rgz = rho(fb, gz)?;
        let (x_kept, y_kept, z_on_x, z_on_y) = match is_dx {
            true => (gx, rho(fb, gy)?, gz, rgz),
            false => (rho(fb, gx)?, gy, rgz, gz),
        };
        let on_x = fb.select(m.y_or_z, x_kept, zero)?;
        let on_y = fb.select(m.y, zero, y_kept)?;
        let on_z = fb.select(m.y, z_on_y, z_on_x)?;
        let on_z = fb.select(m.z, zero, on_z)?;
        fb.composite_construct(triple, &[on_x, on_y, on_z])
    };
    Ok((face(fb, dx, true)?, face(fb, dy, false)?))
}

/// [`CubeFootprint::Length`]: each gradient's two tangent components' Metal length on the
/// tangent `(a + 2) mod 3`; or, `projected` ([`CubeFootprint::Projected`]), an `x` or `z` face's
/// gradient without its normal component and a `y` face's length times `64/91` on `x`.
fn tangent_length(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    (dx, dy): (SValue, SValue),
    projected: bool,
) -> BResult<(SValue, SValue)> {
    let f = f32_ty(fb)?;
    let triple = vec(fb, f, 3)?;
    let m = Major::of(fb, p)?;
    let zero = f32_const(fb, 0)?;
    // Metal's length of a pair of equal components `c` is `91/64 c`.
    let diagonal = f32_const(fb, (64.0f32 / 91.0).to_bits())?;
    let one = |fb: &mut FunctionBuilder<'_>, g: SValue| -> BResult<SValue> {
        let [gx, gy, gz] = components(fb, g)?;
        // The face's two tangents: `y` and `z` on an `x` face, `x` and `z` on a `y` face, `x`
        // and `y` on a `z` face.
        let first = fb.select(m.y_or_z, gx, gy)?;
        let second = fb.select(m.z, gy, gz)?;
        let length = super::anisotropic::metal_length(fb, first, second)?;
        if projected {
            let on_y_face = fb.binary(BinOp::FMul, length, diagonal)?;
            let on_x = fb.select(m.y, on_y_face, gx)?;
            let on_x = fb.select(m.y_or_z, on_x, zero)?;
            let on_y = fb.select(m.y, zero, gy)?;
            let on_z = fb.select(m.y_or_z, zero, gz)?;
            return fb.composite_construct(triple, &[on_x, on_y, on_z]);
        }
        // The tangent `(a + 2) mod 3`: `z` for an `x` face, `x` for a `y` face, `y` for a `z`
        // face.
        let on_x = fb.select(m.y, length, zero)?;
        let on_y = fb.select(m.z, length, zero)?;
        let on_z = fb.select(m.y_or_z, zero, length)?;
        fb.composite_construct(triple, &[on_x, on_y, on_z])
    };
    Ok((one(fb, dx)?, one(fb, dy)?))
}

/// The footprint of a cube's explicit gradients at direction `p` through a runtime or argument
/// buffer's sampler of word `word` (a `u32`), by the word (`CubeFootprint::Word`): where it states
/// `SamplerWord::ANISOTROPIC`, the form an anisotropic constexpr sampler takes on the route, else
/// the isotropic one, each [`cube_footprint`]'s: [`CubeFootprint::Face`] and
/// [`CubeFootprint::Length`]. The device's sampler filters by the bound sampler's own anisotropy,
/// as it does a constexpr sampler's.
pub(super) fn word_footprint(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    grads: (SValue, SValue),
    word: SValue,
) -> BResult<(SValue, SValue)> {
    let anisotropic = cube_footprint(fb, p, grads, CubeFootprint::Face)?;
    let isotropic = cube_footprint(fb, p, grads, CubeFootprint::Length)?;
    by_anisotropy(fb, word, anisotropic, isotropic)
}

/// Per gradient, `anisotropic`'s where sampler word `word` (a `u32`) states
/// `SamplerWord::ANISOTROPIC`, else `isotropic`'s ([`word_footprint`]).
pub(super) fn by_anisotropy(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (ax, ay): (SValue, SValue),
    (ix, iy): (SValue, SValue),
) -> BResult<(SValue, SValue)> {
    let bit = fb.u32_const(SamplerWord::ANISOTROPIC)?;
    let bit = fb.module().const_value(bit)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, bit)?;
    let anisotropic = fb.binary(BinOp::INotEqual, have, zero)?;
    Ok((
        fb.select(anisotropic, ax, ix)?,
        fb.select(anisotropic, ay, iy)?,
    ))
}

/// The sampled image of `source`, its image facts, and for a heap element whether its handle is
/// live (not the null element): a combined binding's, or a heap element's image (`NonUniform`) with
/// its heap sampler.
pub(super) fn sampled_image(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    source: Source,
) -> BResult<(SValue, CombinedImage, Option<SValue>)> {
    match source {
        Source::Combined(key) => {
            let Some(&(binding, image)) = cx.decl.families.sampler.combined.get(&key) else {
                return Err(cx.fail(format!("combined binding {key:?} is not declared")));
            };
            Ok((binding.sampled_image(fb)?, image, None))
        }
        Source::Heap { handle, key } => {
            let Some(g) = cx.decl.families.sampler.heap.get(&key).copied() else {
                return Err(cx.fail(format!("heap sampler {key:?} is not declared")));
            };
            let element = heap_element(cx, fb, (key.group, !key.compare), handle)?;
            let ptr = fb.module().global_value(g)?;
            let sampler = fb.load(ptr, MemAccess::default())?;
            let si = fb.sampled_image(element.img, sampler)?;
            let image = CombinedImage {
                plane: super::plane_of(element.decl.dim),
                arrayed: element.decl.arrayed,
                texel: element.decl.texel,
            };
            Ok((si, image, Some(element.live)))
        }
        Source::HeapSampler {
            handle,
            sampler,
            key,
        } => {
            let Some(g) = cx.decl.families.sampler.heap_samplers.get(&key).copied() else {
                return Err(cx.fail(format!("argument sampler array {key:?} is not declared")));
            };
            let element = heap_element(cx, fb, (key.group, !key.compare), handle)?;
            // The element of the sampler handle's table entry; a missed or null handle is the null
            // entry's element 0, the neutral sampler.
            let index = heap_sampler_field(cx, fb, sampler, EntryField::Element)?;
            let base = fb.module().global_value(g)?;
            let p = fb.access_chain_non_uniform(base, index)?;
            let sampler = fb.load(p, MemAccess::default())?;
            let si = fb.sampled_image(element.img, sampler)?;
            let image = CombinedImage {
                plane: super::plane_of(element.decl.dim),
                arrayed: element.decl.arrayed,
                texel: element.decl.texel,
            };
            Ok((si, image, Some(element.live)))
        }
    }
}

/// Where a sampling operation observes its residency on a slot with a sparse arm: the slot's
/// `SparseTexture` specialization, and the combined binding, whose sampled image each arm makes (a
/// sampled image is consumed in the block that makes it).
fn sparse_arm(cx: &mut EmitCx<'_>, s: &Sampling) -> BResult<Option<(SValue, Binding)>> {
    let Source::Combined(key) = s.source else {
        return Ok(None);
    };
    let Some(spec) = sparse_spec(cx, key.texture, s.residency) else {
        return Ok(None);
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    Ok(Some((spec, binding)))
}

/// The value of sampling strategy `s` in the open block; `grads` are the gradients of a rule-2
/// sample (`Lod::Hoisted`), computed by its switch-at-use. A heap element's sample is zero for the
/// null element. An operation through an argument buffer's sampler then tests the sampler's word
/// for a cubic filter where the stage marks it ([`cubic_reach`]).
pub fn sampling_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
) -> BResult<SValue> {
    let value = match s.swizzle {
        true => swizzled_value(cx, fb, s, grads)?,
        false => sampled_value(cx, fb, s, grads)?,
    };
    // A texture the heap holds by its raw texels alone: Metal's sample or gather of it, after
    // the device's access, which stays where it was.
    let value = match &s.heap_raw {
        Some(raw) => super::heap_raw::emit(cx, fb, s, raw, value)?,
        None => value,
    };
    // A texture of another class: its own result's words handed back as declared.
    let value = match &s.declared {
        Some(d) => declared_value(cx, fb, s, d, value)?,
        None => value,
    };
    if let Source::HeapSampler { sampler, .. } = s.source {
        if cx.plan.families.sampler.cubic_reach {
            cubic_reach(cx, fb, sampler)?;
        }
    }
    Ok(value)
}

/// [`sampled_value`] of sampling `s` of a packed normalized slot of
/// `LoweringInputs::swizzled_storage` ([`Sampling::swizzle`]), whose view takes no swizzle: the
/// unswizzled texture's held texel, which its class converts, swizzled by the slot's
/// `TextureSwizzle` specialization as plain Metal swizzles a sampled texture's texels; a gather
/// of a component made of the unswizzled texture's gathers of each component, the one the
/// swizzle names taken (0 or 1 at every tap for its `Zero` or `One`); then the lanes Metal reads
/// as constants, which the slot's `ConstantLanes` specialization states through the swizzle
/// ([`held_constants`], which the unswizzled accesses skip). A gather's status word is its
/// red gather's: each of the four reads the same texels.
fn swizzled_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
) -> BResult<SValue> {
    let Source::Combined(key) = s.source else {
        return Err(cx.fail("a swizzled sampling of a heap element"));
    };
    let Some(word) = cx.decl.families.texture.swizzle.get(&key.texture).copied() else {
        return Err(cx.fail(format!(
            "texture slot {} has no swizzle constant",
            key.texture
        )));
    };
    let (texel, status, gathered) = match &s.op {
        SampleOp::Gather {
            component,
            dref,
            offset,
        } => {
            let mut texels = Vec::with_capacity(4);
            let mut status = None;
            for c in 0..4u32 {
                let mut g = s.clone();
                g.op = SampleOp::Gather {
                    component: c,
                    dref: *dref,
                    offset: offset.clone(),
                };
                let value = sampled_value(cx, fb, &g, grads)?;
                let (x, w) = texel_and_word(cx, fb, &g, value)?;
                texels.push(x);
                if c == 0 {
                    status = w;
                }
            }
            let Ok(taps) = <[SValue; 4]>::try_from(texels) else {
                return Err(cx.fail("a swizzled gather without four components"));
            };
            let x = crate::lower::texture::swizzled_gather(cx, fb, taps, word, *component)?;
            (x, status, Some(*component))
        }
        _ => {
            let value = sampled_value(cx, fb, s, grads)?;
            let (x, w) = texel_and_word(cx, fb, s, value)?;
            (
                crate::lower::texture::swizzled_held(cx, fb, x, word)?,
                w,
                None,
            )
        }
    };
    let texel = match s.unmapped {
        Unmapped::Device => texel,
        Unmapped::Constants | Unmapped::Named => {
            crate::lower::texture::constant_lanes(cx, fb, key.texture, texel, gathered)?
        }
    };
    with_word(cx, fb, &s.result, texel, status)
}

/// The held texel of `value`, a sampling's `{T, i8}` or `T` result ([`Sampling::result`]), and
/// its status word where it has one.
fn texel_and_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    value: SValue,
) -> BResult<(SValue, Option<SValue>)> {
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), &s.result)? else {
        return Err(cx.fail("a sampling result without a lowered type"));
    };
    match matches!(fb.module().types().get(ty)?, SType::Struct { .. }) {
        true => Ok((
            fb.composite_extract(value, &[0])?,
            Some(fb.composite_extract(value, &[1])?),
        )),
        false => Ok((value, None)),
    }
}

/// Sampling `s`'s value `value` of its texture's own class (`Sampling::result`, a `{T, i8}` or
/// `T` of the native texel `Sampling::texel`) as its declared result `d`: the texel's words
/// handed back as the declared type (`texture::reinterpret`), the status word kept.
fn declared_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    d: &super::DeclaredResult,
    value: SValue,
) -> BResult<SValue> {
    let Some(native) = s.texel else {
        return Err(cx.fail("a declared sampling result without its texel"));
    };
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), &s.result)? else {
        return Err(cx.fail("a sampling result without a lowered type"));
    };
    let (x, word) = match matches!(fb.module().types().get(ty)?, SType::Struct { .. }) {
        true => (
            fb.composite_extract(value, &[0])?,
            Some(fb.composite_extract(value, &[1])?),
        ),
        false => (value, None),
    };
    let x = crate::lower::texture::reinterpret(fb, native, d.conv, x)?;
    with_word(cx, fb, &d.result, x, word)
}

/// Where the argument buffer's sampler of handle `sampler` is cubic (its table entry's word holds
/// `SamplerWord::CUBIC`), set `StatusBits::CUBIC_ARGUMENT_SAMPLER` in the command buffer's
/// `ExecutionStatus` word (`RuntimeFact::CubicArgumentSampler`): the operation sampled through
/// the Vulkan sampler's nearest placeholder, not Metal's bicubic filter, so its command buffer
/// fails. The mark is on that path only: the operation itself, implicit derivatives and all, ran
/// before the test, outside the selection. An operation through the same handle after an earlier
/// one of its block runs no test of its own: wherever it runs, that test ran on the same sampler
/// (16 samples of one handle in a block test once).
fn cubic_reach<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    sampler: ValueId,
) -> BResult<()> {
    if cx.cubic_test_dominates(fb, sampler) {
        return Ok(());
    }
    // `CUBIC` is the word's highest bit (`SamplerWord::FIELDS`).
    let word = heap_sampler_field(cx, fb, sampler, EntryField::Flags)?;
    let bit = fb.u32_const(SamplerWord::CUBIC)?;
    let bit = fb.module().const_value(bit)?;
    let cubic = fb.binary(BinOp::UGreaterThanEqual, word, bit)?;
    let mark: ArmFn<'_, EmitCx<'c>> = Box::new(|fb, cx| {
        crate::lower::access::absent::mark_bits(
            cx,
            fb,
            crate::bind::status::StatusBits::CUBIC_ARGUMENT_SAMPLER,
        )?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(cx, cubic, mark, None)?;
    cx.cubic_tested(fb, sampler, word);
    Ok(())
}

/// [`sampling_value`] before an argument buffer's sampler's cubic test.
pub(super) fn sampled_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
) -> BResult<SValue> {
    // Metal's anisotropic filter emulated by the device's samples at explicit levels; a runtime
    // sampler's where its word states anisotropy, the rest of this path elsewhere.
    if let Some(a) = s.anisotropic.as_ref() {
        return super::anisotropic::sample(cx, fb, s, a, grads);
    }
    let (si, image, live) = sampled_image(cx, fb, s.source)?;
    if let SampleOp::QueryLevel { min } = &s.op {
        let x = query_level(cx, fb, si, *min)?;
        return live_or_zero(fb, live, x);
    }
    let arm = sparse_arm(cx, s)?;
    // A tile map answers a pixel sampler's fetch and a sample or gather with Metal's footprint:
    // selection gives a slot whose observed samples or gathers it does not answer no tile-map arm.
    let tiled = match s.source {
        Source::Combined(key) => tile_map(cx, key.texture),
        Source::Heap { .. } | Source::HeapSampler { .. } => None,
    };
    let fetched = matches!(s.op, SampleOp::PixelFetch { .. });
    if arm.is_some() && tiled.is_some() && !fetched && s.footprint.is_none() {
        return Err(cx.fail("a filtered sample's residency on a slot with a tile map"));
    }
    // Metal's footprint: the device's plain sample or gather, and the word of exactly the texels
    // Metal counts, through the slot's sparse arm or tile map.
    let exact = match (arm, s.footprint, tiled) {
        (Some((spec, binding)), Some(fp), Some(map)) => Some((spec, binding, fp, map)),
        (Some(_), Some(_), None) => {
            return Err(cx.fail("Metal's footprint on a slot without a tile map"))
        }
        _ => None,
    };
    let arm = if exact.is_some() { None } else { arm };
    if let SampleOp::PixelFetch { offset } = &s.op {
        let (x, word) = pixel_fetch(cx, fb, s, si, image, offset.as_ref(), arm)?;
        let x = unmapped_value(cx, fb, s, x, word)?;
        let x = live_or_zero(fb, live, x)?;
        return texel_value(cx, fb, s, image, (x, TexelOrigin::Image), word);
    }
    if let SampleOp::Bicubic(b) = &s.op {
        // No sparse form: the slot takes no sparse arm, and the word is 0. The routines return the
        // held texel, a `half` result already rounded.
        let x = super::bicubic::emit(cx, fb, s, si, image, b)?;
        let x = live_or_zero(fb, live, x)?;
        return with_word(cx, fb, &s.result, x, None);
    }
    // An argument buffer's sampler whose flags select the exact fetch: the fetch, made here from
    // the sampled image the sample reads too, and chosen after the sample by the handle's flags.
    // Both run, so no branch puts the sample in divergent flow.
    let flagged_fetch = match (&s.pixel, &s.op) {
        (
            Pixel::Heap(handle),
            SampleOp::Sample {
                dref: None, offset, ..
            },
        ) => {
            let (x, word) = pixel_fetch(cx, fb, s, si, image, offset.as_ref(), arm)?;
            Some((*handle, x, word))
        }
        _ => None,
    };
    let (mut coord, mut grad) = plane_and_gradients(cx, fb, s, grads)?;
    // Pixel coordinates: the plane coordinate (and any gradient) over the level-0 size, always
    // or behind the runtime sampler's word.
    if let Some(spec) = pixel_coordinates(cx, fb, s)? {
        let img = fb.image(si)?;
        let zero = fb.u32_const(0)?;
        let zero = fb.module().const_value(zero)?;
        let size = fb.query(ImageQuery::SizeLod(zero), img)?;
        let size_lanes = image.plane + u8::from(image.arrayed);
        (coord, grad) = scaled(fb, s.lanes, (size, size_lanes), spec, (coord, grad))?;
    }
    // The plane coordinate as the sample reads it (pixel coordinates scaled), whose derivatives a
    // bias past the device's limit scales, and from which Metal's footprint of an observed access
    // follows.
    let plane = coord;
    let (coord, layer) = layered(cx, fb, s, coord)?;
    // The access, from the sampled image `si`; under a sparse arm each arm loads its own.
    // The access, from the sampled image `si` (loaded in the open block); under a sparse arm each
    // arm loads its own.
    let observe = |cx: &mut EmitCx<'_>,
                   fb: &mut FunctionBuilder<'_>,
                   si: SValue,
                   access: &dyn Fn(&mut FunctionBuilder<'_>, SValue, bool) -> BResult<SValue>|
     -> BResult<(SValue, Option<SValue>)> {
        match arm {
            Some((spec, binding)) => {
                let (x, word) = resident_access(cx, fb, spec, |fb, sparse| {
                    let si = binding.sampled_image(fb)?;
                    access(fb, si, sparse)
                })?;
                Ok((x, Some(word)))
            }
            None => Ok((access(fb, si, false)?, None)),
        }
    };
    let (texel, word) = match &s.op {
        SampleOp::Sample {
            lod,
            dref,
            min_lod,
            offset,
        } => {
            let (ops, dref, beyond) = sample_operands(
                cx,
                fb,
                s,
                (plane, grad, true),
                (lod, *min_lod, offset, *dref),
            )?;
            let (x, word) = observe(cx, fb, si, &sample_access(coord, dref, ops))?;
            // Past the device's limit the sample with the scaled derivatives replaces the
            // implicit one; the implicit sample stays where it was, unconditional.
            let (x, word) = match beyond {
                None => (x, word),
                Some((test, grads)) => {
                    let mut grad_ops = ops;
                    grad_ops.bias = None;
                    grad_ops.grad = Some(grads);
                    let kept: Vec<SValue> = std::iter::once(x).chain(word).collect();
                    let past: ArmFn<'_, EmitCx<'_>> = Box::new(|fb, cx| {
                        // A sampled image is consumed in the block that makes it.
                        let (si, _, _) = sampled_image(cx, fb, s.source)?;
                        let access = sample_access(coord, dref, grad_ops);
                        let (x, word) = observe(cx, fb, si, &access)?;
                        Ok(ArmEnd::FallThrough(
                            std::iter::once(x).chain(word).collect(),
                        ))
                    });
                    let within: ArmFn<'_, EmitCx<'_>> =
                        Box::new(move |_, _| Ok(ArmEnd::FallThrough(kept)));
                    let Merged(merged) = fb.if_(cx, test, past, Some(within))?;
                    match merged[..] {
                        [x] => (x, None),
                        [x, w] => (x, Some(w)),
                        _ => return Err(cx.fail("a biased sample without its merged texel")),
                    }
                }
            };
            // A depth sample without comparison reads the first component.
            let x = if s.depth && dref.is_none() {
                fb.composite_extract(x, &[0])?
            } else {
                x
            };
            match flagged_fetch {
                None => (x, word),
                // The fetch where the sampler's word states what the image needs: pixel
                // coordinates, one isotropic level clamped to the edge on each plane axis, and
                // nearest filters unless the texture is integer.
                Some((handle, fetched, fetched_word)) => {
                    let integer = image.texel != TexelClass::Float;
                    let mask = SamplerWord::exact_fetch(image.plane, integer);
                    let state = heap_sampler_field(cx, fb, handle, EntryField::Flags)?;
                    let take = word_states(fb, state, mask)?;
                    let x = fb.select(take, fetched, x)?;
                    let word = match (fetched_word, word) {
                        (Some(f), Some(w)) => Some(fb.select(take, f, w)?),
                        (None, None) => None,
                        _ => return Err(cx.fail("a fetch and a sample of different residency")),
                    };
                    (x, word)
                }
            }
        }
        SampleOp::Gather {
            component,
            dref,
            offset,
        } => {
            let mut ops = ImageOperands::default();
            offset_operands(cx, fb, offset, s.lanes, &mut ops)?;
            let component = *component;
            let dref = match dref {
                Some(r) => Some(reference(cx, fb, s, *r)?),
                None => None,
            };
            observe(cx, fb, si, &|fb, si, sparse| match (dref, sparse) {
                (Some(r), true) => fb.sparse_dref_gather(si, coord, r, &ops),
                (Some(r), false) => fb.dref_gather(si, coord, r, &ops),
                (None, true) => fb.sparse_gather(si, coord, component, &ops),
                (None, false) => fb.gather(si, coord, component, &ops),
            })?
        }
        SampleOp::Lod { clamped } => {
            // `x` is the level the sampler would access (clamped), `y` the level relative to the
            // base (unclamped).
            let l = fb.query_lod(si, coord)?;
            let l = fb.composite_extract(l, &[if *clamped { 0 } else { 1 }])?;
            return live_or_zero(fb, live, l);
        }
        SampleOp::PixelFetch { .. } | SampleOp::Bicubic(_) | SampleOp::QueryLevel { .. } => {
            return Err(cx.fail("a fetching sample or level query past its own path"))
        }
    };
    let word = match exact {
        Some((spec, binding, fp, map)) => {
            let site = super::footprint::Site {
                binding,
                coord: plane,
                lanes: s.lanes,
                layer,
            };
            Some(super::footprint::word(cx, fb, fp, spec, map, site)?)
        }
        None => word,
    };
    let texel = live_or_zero(fb, live, texel)?;
    texel_value(cx, fb, s, image, (texel, TexelOrigin::Image), word)
}

/// The plane coordinate of `s`, read from the AIR operand as given: the device's sampler, and an
/// emulation's arithmetic, read a subnormal coordinate as the device does.
pub(super) fn plane_coord(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
) -> BResult<SValue> {
    cx.operand(fb, s.coord)
}

/// The plane coordinate of `s` and its explicit gradients, read from the AIR operands: a cube's
/// direction with its ties broken as Metal breaks them and its gradients as Metal's footprint on
/// that face, and a rule-2 sample's the gradients `grads` its switch-at-use hoisted.
pub(super) fn plane_and_gradients(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
) -> BResult<(SValue, Option<(SValue, SValue)>)> {
    let coord = plane_coord(cx, fb, s)?;
    let coord = if s.cube {
        super::face::metal_ties(fb, coord)?
    } else {
        coord
    };
    let grad = match &s.op {
        SampleOp::Sample {
            lod: Lod::Grad(dx, dy),
            ..
        } => Some((cx.operand(fb, *dx)?, cx.operand(fb, *dy)?)),
        SampleOp::Sample {
            lod: Lod::CubeGrad(dx, dy, CubeFootprint::Word),
            ..
        } => {
            let (dx, dy) = (cx.operand(fb, *dx)?, cx.operand(fb, *dy)?);
            let Some(pixel) = word_pixel(s) else {
                return Err(cx.fail("a cube footprint by a word without its sampler's word"));
            };
            let word = sampler_word(cx, fb, pixel)?;
            Some(word_footprint(fb, coord, (dx, dy), word)?)
        }
        SampleOp::Sample {
            lod: Lod::CubeGrad(dx, dy, form),
            ..
        } => {
            let (dx, dy) = (cx.operand(fb, *dx)?, cx.operand(fb, *dy)?);
            Some(cube_footprint(fb, coord, (dx, dy), *form)?)
        }
        SampleOp::Sample {
            lod: Lod::Hoisted, ..
        } => match grads {
            Some(g) => Some(g),
            None => return Err(cx.fail("a hoisted-gradient sample without its gradients")),
        },
        _ => None,
    };
    Ok((coord, grad))
}

/// Whether `s`'s plane coordinate is in pixels: `None` when normalized, `Some(None)` always, and
/// `Some(Some(c))` where `bool` `c` says so (a runtime sampler's word, its slot's `SamplerState`
/// specialization or its argument buffer table entry's).
pub(super) fn pixel_coordinates(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
) -> BResult<Option<Option<SValue>>> {
    Ok(match s.pixel {
        Pixel::None => None,
        Pixel::Always => Some(None),
        Pixel::Spec(_) | Pixel::Heap(_) => {
            let word = sampler_word(cx, fb, s.pixel)?;
            Some(Some(word_states(fb, word, SamplerWord::PIXEL_COORDINATES)?))
        }
    })
}

/// The word of the runtime sampler `pixel` names: a slot's `SamplerState` specialization (masked by
/// the fields the function reads), or an argument buffer's sampler table entry's.
pub(super) fn sampler_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    pixel: Pixel,
) -> BResult<SValue> {
    match pixel {
        Pixel::Spec(n) => match cx.decl.families.sampler.words.get(&n).copied() {
            Some(w) => Ok(w),
            None => Err(cx.fail(format!("sampler slot {n} has no state word"))),
        },
        Pixel::Heap(handle) => heap_sampler_field(cx, fb, handle, EntryField::Flags),
        Pixel::None | Pixel::Always => Err(cx.fail("a constexpr sampler's word")),
    }
}

/// Field `field` of the sampler table entry an argument buffer's sampler handle `handle` names: the
/// handle is the entry's index, the null entry's 0 for a missed or null id.
fn heap_sampler_field(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    handle: ValueId,
    field: EntryField,
) -> BResult<SValue> {
    let Some(table) = cx.decl.families.access.sampler_table else {
        return Err(cx.fail("an argument buffer's sampler without its sampler table"));
    };
    let k = cx.operand(fb, handle)?;
    entry_field(fb, table.table, k, field)
}

/// Whether sampler word `word` masked by `mask` is exactly [`SamplerWord::PIXEL_COORDINATES`] (a
/// `bool`): every predicate a sampling tests of a runtime sampler is a pixel sampler's, with the
/// other fields of `mask` at their zero values ([`SamplerWord::exact_fetch`]).
fn word_states(fb: &mut FunctionBuilder<'_>, word: SValue, mask: u32) -> BResult<SValue> {
    let m = fb.u32_const(mask)?;
    let m = fb.module().const_value(m)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, m)?;
    let pixel = fb.u32_const(SamplerWord::PIXEL_COORDINATES)?;
    let pixel = fb.module().const_value(pixel)?;
    fb.binary(BinOp::IEqual, have, pixel)
}

/// The plane coordinate `coord` (`lanes` floats) and gradients `grad` over the level-0 size
/// `size` (`size_lanes` components): always, or where `spec` holds.
pub(super) fn scaled(
    fb: &mut FunctionBuilder<'_>,
    lanes: u8,
    (size, size_lanes): (SValue, u8),
    spec: Option<SValue>,
    (coord, grad): (SValue, Option<(SValue, SValue)>),
) -> BResult<(SValue, Option<(SValue, SValue)>)> {
    let pick = |fb: &mut FunctionBuilder<'_>, x: SValue| -> BResult<SValue> {
        let scaled = scale(fb, x, size, lanes, size_lanes)?;
        match spec {
            Some(c) => fb.select(c, scaled, x),
            None => Ok(scaled),
        }
    };
    let coord = pick(fb, coord)?;
    let grad = match grad {
        Some((dx, dy)) => Some((pick(fb, dx)?, pick(fb, dy)?)),
        None => None,
    };
    Ok((coord, grad))
}

/// The coordinate `coord` of `s` with the array layer as its last component, as a float, and
/// the layer.
pub(super) fn layered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    coord: SValue,
) -> BResult<(SValue, Option<SValue>)> {
    let layer = match s.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let Some(l) = layer else {
        return Ok((coord, None));
    };
    let f = f32_ty(fb)?;
    let lf = fb.convert(ConvOp::ConvertUToF, l, f)?;
    let ty = vec(fb, f, s.lanes + 1)?;
    Ok((fb.composite_construct(ty, &[coord, lf])?, layer))
}

/// The `SpecKey::OffsetBaseAxes` tests of `s`'s slot (`texture::OffsetBase`), by which its
/// emulation places a texel offset (`position::offset_dropped`), where `s` reads them
/// ([`Sampling::offset_base`]); `None` for every other access, a heap element's among them.
pub(super) fn offset_base(
    cx: &mut EmitCx<'_>,
    s: &Sampling,
) -> BResult<Option<crate::lower::texture::OffsetBase>> {
    match s.source {
        Source::Combined(key) if s.offset_base => {
            match cx.decl.families.texture.offset_base.get(&key.texture) {
                Some(&base) => Ok(Some(base)),
                None => Err(cx.fail(format!(
                    "texture slot {} has no OffsetBaseAxes specialization",
                    key.texture
                ))),
            }
        }
        _ => Ok(None),
    }
}

/// The texel offset operand of `o` on a `lanes`-component plane: a constant's `ConstOffset`, a
/// dynamic one's `Offset` as Metal reads it, which an armed offset arms
/// (`FunctionBuilder::arm_offset`, [`arm_selector`]): every access it is the operand of becomes
/// the switch over its values.
pub(super) fn offset_operands(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    o: &Option<Offset>,
    lanes: u8,
    ops: &mut ImageOperands,
) -> BResult<()> {
    match o {
        Some(Offset::Const(k)) => ops.const_offset = Some(const_offset(fb, k)?),
        Some(Offset::Dynamic(v, arms)) => {
            let value = metal_offset(cx, fb, *v, lanes)?;
            if let Some(a) = arms {
                let selector = arm_selector(fb, value, a, lanes)?;
                fb.arm_offset(
                    value,
                    ArmedOffset {
                        selector,
                        lanes: a.lanes.clone(),
                    },
                )?;
            }
            ops.offset = Some(value);
        }
        None => {}
    }
    Ok(())
}

/// The selector of armed offset `arms` whose value Metal reads is `value` (`lanes` signed lanes,
/// each in `[-8, 7]`): a `u32` whose bits `4j .. 4j + 4` hold the `j`-th dynamic lane plus 8.
fn arm_selector(
    fb: &mut FunctionBuilder<'_>,
    value: SValue,
    arms: &super::OffsetArms,
    lanes: u8,
) -> BResult<SValue> {
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let eight = fb.module().constant(Constant::Scalar {
        ty: i32_,
        bits: ConstBits(8),
    })?;
    let eight = fb.module().const_value(eight)?;
    let mut selector = None;
    let dynamic =
        (0..u32::from(lanes)).filter(|k| arms.lanes.get(*k as usize).is_some_and(Option::is_none));
    for (j, k) in dynamic.enumerate() {
        let x = if lanes == 1 {
            value
        } else {
            fb.composite_extract(value, &[k])?
        };
        let x = fb.binary(BinOp::IAdd, x, eight)?;
        let x = fb.convert(ConvOp::Bitcast, x, u32_)?;
        let x = match j {
            0 => x,
            _ => {
                let shift = fb.u32_const(4 * j as u32)?;
                let shift = fb.module().const_value(shift)?;
                fb.binary(BinOp::ShiftLeftLogical, x, shift)?
            }
        };
        selector = Some(match selector {
            None => x,
            Some(s) => fb.binary(BinOp::BitwiseOr, s, x)?,
        });
    }
    selector.ok_or(BuildError::Operand {
        op: crate::spv::Op::Switch,
        rule: OperandRule::Arity,
    })
}

/// A dynamic bias's test of the device's limit and the scaled derivatives a sample takes where
/// it passes.
pub(super) type Beyond = Option<(SValue, (SValue, SValue))>;

/// The operands of sample `s` (level `lod`, `min_lod` clamp, `offset`, comparison `dref`) at
/// plane coordinate `plane` with gradients `grad`: its image operands, its comparison reference,
/// and for a bias decided per invocation the test and the gradients that replace the `Bias`
/// operand where it holds, both in the implicit sample's own position. An `unshared`
/// sample's axis gradients read its texture's level-0 size and its sampler's word; a shared one (a
/// switch-at-use's emulation, whose arms' samplers differ) takes the derivatives scaled by `2^B`
/// through every sampler, the isotropic route.
pub(super) fn sample_operands(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (plane, grad, unshared): (SValue, Option<(SValue, SValue)>, bool),
    (lod, min_lod, offset, dref): (&Lod, Option<ValueId>, &Option<Offset>, Option<ValueId>),
) -> BResult<(ImageOperands, Option<SValue>, Beyond)> {
    let mut ops = ImageOperands::default();
    let mut beyond = None;
    match lod {
        Lod::Implicit { bias, reach } => {
            if let Some(b) = bias {
                let b = cx.operand(fb, *b)?;
                // Metal's reading of the bias, but where the device's sampler reads it.
                let b = match reach {
                    BiasReach::Metal => b,
                    _ => metal_bias(fb, b)?,
                };
                // The level-0 size the axis gradients read, where they are taken.
                let axes = match reach {
                    BiasReach::Axes if unshared => {
                        // A sampled image is consumed in the block that makes it.
                        let (si, image, _) = sampled_image(cx, fb, s.source)?;
                        let img = fb.image(si)?;
                        let zero = fb.u32_const(0)?;
                        let zero = fb.module().const_value(zero)?;
                        let size = fb.query(ImageQuery::SizeLod(zero), img)?;
                        Some((size, image.plane + u8::from(image.arrayed)))
                    }
                    _ => None,
                };
                match *reach {
                    BiasReach::Metal | BiasReach::Within => ops.bias = Some(b),
                    BiasReach::Beyond => {
                        ops.grad = Some(plane_bias_gradients(fb, plane, b, s.lanes)?);
                    }
                    BiasReach::Axes => {
                        ops.grad = Some(match axes {
                            Some(size) => plane_axis_gradients(fb, plane, b, size)?,
                            None => plane_bias_gradients(fb, plane, b, s.lanes)?,
                        });
                    }
                    BiasReach::Either {
                        max_bias_bits,
                        sampler_bias_bits,
                    } => {
                        // The derivatives in the implicit sample's own position, before any
                        // branch: both need the fragment's uniform control flow.
                        let scaled = plane_bias_gradients(fb, plane, b, s.lanes)?;
                        let past = past_limit(fb, b, max_bias_bits, sampler_bias_bits)?;
                        ops.bias = Some(b);
                        beyond = Some((past, scaled));
                    }
                }
            }
        }
        Lod::Level(Some(l)) => ops.lod = Some(cx.operand(fb, *l)?),
        Lod::Level(None) => ops.lod = Some(f32_const(fb, 0)?),
        Lod::Base { bias, min } => ops.lod = Some(base_level(cx, fb, *bias, *min)?),
        Lod::Grad(..) | Lod::CubeGrad(..) | Lod::Hoisted => ops.grad = grad,
    }
    if let Some(m) = min_lod {
        ops.min_lod = Some(cx.operand(fb, m)?);
    }
    // An armed offset's switch holds no implicit level: the sample takes the gradients its bias
    // takes past the device's limit (the derivatives scaled by `2^B`, or the axis gradients
    // through an anisotropic word), or the plain derivatives without a bias, all in its own
    // position (`PrecisionNote::GradFromBias`).
    if matches!(offset, Some(Offset::Dynamic(_, Some(_))))
        && ops.lod.is_none()
        && ops.grad.is_none()
    {
        let grads = match (beyond.take(), ops.bias.take(), lod) {
            (Some((_, grads)), _, _) => grads,
            (
                None,
                Some(b),
                Lod::Implicit {
                    reach: BiasReach::Metal,
                    ..
                },
            ) => {
                let b = metal_bias(fb, b)?;
                plane_bias_gradients(fb, plane, b, s.lanes)?
            }
            (None, Some(b), _) => plane_bias_gradients(fb, plane, b, s.lanes)?,
            (None, None, _) => (
                fb.derivative(DerivOp::DPdx, plane)?,
                fb.derivative(DerivOp::DPdy, plane)?,
            ),
        };
        ops.grad = Some(grads);
    }
    offset_operands(cx, fb, offset, s.lanes, &mut ops)?;
    let dref = match dref {
        Some(r) => Some(reference(cx, fb, s, r)?),
        None => None,
    };
    Ok((ops, dref, beyond))
}

/// The reference `r` of sampling `s`'s depth comparison as the comparison reads it: clamped to
/// `[0, 1]`, a NaN taken as 0, where the pipeline's `fragmentDepthCompareClampMask` clamps it
/// ([`Sampling::clamp_reference`]), else as computed.
pub(super) fn reference(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    r: ValueId,
) -> BResult<SValue> {
    let r = cx.operand(fb, r)?;
    if !s.clamp_reference {
        return Ok(r);
    }
    let zero = f32_const(fb, 0)?;
    let one = f32_const(fb, 1.0f32.to_bits())?;
    let is_nan = fb.unary(UnOp::IsNan, r)?;
    let r = fb.select(is_nan, zero, r)?;
    fb.glsl(GlslOp::NClamp, &[r, zero, one])
}

/// The access of a sample at `coord` with comparison `dref` and operands `ops`, from the sampled
/// image it is handed, in its sparse form where asked.
pub(super) fn sample_access(
    coord: SValue,
    dref: Option<SValue>,
    ops: ImageOperands,
) -> impl Fn(&mut FunctionBuilder<'_>, SValue, bool) -> BResult<SValue> {
    move |fb, si, sparse| {
        if sparse {
            fb.sparse_sample(si, coord, dref, &ops)
        } else {
            fb.sample(si, coord, dref, false, &ops)
        }
    }
}

/// The derivatives of 2D plane coordinate `plane` in the open block as the axis gradients of
/// Metal's bias `bias` on a texture of level-0 size `size` ([`axis_gradients`]).
pub(super) fn plane_axis_gradients(
    fb: &mut FunctionBuilder<'_>,
    plane: SValue,
    bias: SValue,
    (size, size_lanes): (SValue, u8),
) -> BResult<(SValue, SValue)> {
    let dx = fb.derivative(DerivOp::DPdx, plane)?;
    let dy = fb.derivative(DerivOp::DPdy, plane)?;
    axis_gradients(fb, (dx, dy), bias, size, size_lanes)
}

/// Whether sampler word `word` (a `u32`) has any bit of `bits`.
pub(super) fn word_states_any(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    bits: u32,
) -> BResult<SValue> {
    let m = fb.u32_const(bits)?;
    let m = fb.module().const_value(m)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, m)?;
    fb.binary(BinOp::INotEqual, have, zero)
}

/// The derivatives of plane coordinate `plane` (`lanes` floats) in the open block, scaled by
/// `2^bias` ([`bias_gradients`]).
fn plane_bias_gradients(
    fb: &mut FunctionBuilder<'_>,
    plane: SValue,
    bias: SValue,
    lanes: u8,
) -> BResult<(SValue, SValue)> {
    let dx = fb.derivative(DerivOp::DPdx, plane)?;
    let dy = fb.derivative(DerivOp::DPdy, plane)?;
    bias_gradients(fb, dx, dy, bias, lanes)
}

/// Whether bias `b`, with the sampler's own bias (`f32` bits), passes the device's
/// `maxSamplerLodBias` (`f32` bits), where Vulkan clamps the sum: `|b + own| > max`, false for a
/// NaN sum (which keeps the `Bias` operand).
fn past_limit(
    fb: &mut FunctionBuilder<'_>,
    b: SValue,
    max_bias_bits: u32,
    sampler_bias_bits: u32,
) -> BResult<SValue> {
    let sum = if sampler_bias_bits & 0x7fff_ffff == 0 {
        b
    } else {
        let own = f32_const(fb, sampler_bias_bits)?;
        fb.binary(BinOp::FAdd, b, own)?
    };
    let magnitude = fb.glsl(GlslOp::FAbs, &[sum])?;
    let max = f32_const(fb, max_bias_bits)?;
    fb.binary(BinOp::FOrdGreaterThan, magnitude, max)
}

/// A dynamic texel offset of `lanes` components as Metal reads it: each signed component's low
/// [`super::METAL_TEXEL_OFFSET_BITS`] bits sign-extended (`OpBitFieldSExtract(v, 0, 4)`,
/// [`super::metal_offset`]). The value always lies in Metal's range [`super::METAL_TEXEL_OFFSET`],
/// so as the `Offset` operand it lies in the device's range, which selection checked contains
/// Metal's (VUID-RuntimeSpirv-OpImageSample-06435/06436, OpImage-06376/06377); a pixel fetch adds
/// it to the texel coordinate.
fn metal_offset(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: ValueId,
    lanes: u8,
) -> BResult<SValue> {
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let ivec = vec(fb, i32_, lanes)?;
    let v = cx.operand(fb, v)?;
    // The operand is a signed vector; a lowered `i32` lane may be held unsigned.
    let v = if fb.value_ty(v)? == ivec {
        v
    } else {
        fb.convert(ConvOp::Bitcast, v, ivec)?
    };
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let bits = fb.u32_const(super::METAL_TEXEL_OFFSET_BITS)?;
    let bits = fb.module().const_value(bits)?;
    fb.bit_field_extract(true, v, zero, bits)
}

/// Metal's answer to a LOD query outside a fragment function: 0 for the unclamped level (`min`
/// `None`), and for the clamped one the sampler's minimum level in sixty-fourths bounded by the
/// last level of `si`'s image.
fn query_level(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    si: SValue,
    min: Option<MinLevel>,
) -> BResult<SValue> {
    let min = match min {
        None | Some(MinLevel::Const(0)) => return f32_const(fb, 0),
        Some(MinLevel::Const(v)) => f32_const(fb, (v as f32 / 64.0).to_bits())?,
        Some(MinLevel::Slot(n)) => {
            let Some(v) = cx.decl.families.sampler.min_levels.get(&n).copied() else {
                return Err(cx.fail(format!("sampler slot {n} has no minimum level")));
            };
            let f = f32_ty(fb)?;
            let v = fb.convert(ConvOp::ConvertUToF, v, f)?;
            let scale = f32_const(fb, (1.0f32 / 64.0).to_bits())?;
            fb.binary(BinOp::FMul, v, scale)?
        }
    };
    let img = fb.image(si)?;
    let levels = fb.query(ImageQuery::Levels, img)?;
    let one = fb.u32_const(1)?;
    let one = fb.module().const_value(one)?;
    let last = fb.binary(BinOp::ISub, levels, one)?;
    let f = f32_ty(fb)?;
    let last = fb.convert(ConvOp::ConvertUToF, last, f)?;
    fb.glsl(GlslOp::FMin, &[min, last])
}

/// The `f32` constant with bits `bits`.
fn f32_const(fb: &mut FunctionBuilder<'_>, bits: u32) -> BResult<SValue> {
    let f = f32_ty(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: f,
        bits: ConstBits(u64::from(bits)),
    })?;
    fb.module().const_value(c)
}

/// The explicit level of a sample outside a fragment function (`Lod::Base`): Metal's default level
/// 0 shifted by the whole bias (Metal does not clamp it to `maxSamplerLodBias`, and an explicit
/// level is not clamped there either), then bounded below by the minimum-LOD clamp.
fn base_level(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    bias: Option<ValueId>,
    min: Option<ValueId>,
) -> BResult<SValue> {
    let mut level = match bias {
        Some(b) => cx.operand(fb, b)?,
        None => f32_const(fb, 0)?,
    };
    if let Some(m) = min {
        let m = cx.operand(fb, m)?;
        level = fb.glsl(GlslOp::FMax, &[level, m])?;
    }
    Ok(level)
}

/// Where the word of sample `s`'s runtime or argument buffer's sampler is read: its pixel
/// handling's, else (a cube's) its source's sampler slot or sampler table entry.
pub(super) fn word_pixel(s: &Sampling) -> Option<Pixel> {
    match (s.pixel, s.source) {
        (Pixel::Spec(_) | Pixel::Heap(_), _) => Some(s.pixel),
        (
            _,
            Source::Combined(super::BindingKey {
                sampler: super::SamplerRef::Slot(n),
                ..
            })
            | Source::Heap {
                key:
                    super::HeapKey {
                        sampler: super::SamplerRef::Slot(n),
                        ..
                    },
                ..
            },
        ) => Some(Pixel::Spec(n)),
        (_, Source::HeapSampler { sampler, .. }) => Some(Pixel::Heap(sampler)),
        _ => None,
    }
}

/// The texel a nearest, clamp-to-edge pixel sampler returns, exactly: `OpImageFetch` of the base
/// level at `clamp(floor(coord) + offset, 0, size - 1)`, the layer clamped into the array as
/// sampling clamps it. A depth texel is its first component. Under a sparse arm the fetch's status
/// word too.
fn pixel_fetch(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    si: SValue,
    image: super::CombinedImage,
    offset: Option<&Offset>,
    arm: Option<(SValue, Binding)>,
) -> BResult<(SValue, Option<SValue>)> {
    let img = fb.image(si)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })?;
    let lanes = s.lanes;
    let ivec = vec(fb, i32_, lanes)?;
    let uvec = vec(fb, u32_, lanes)?;
    // The plane components of the size, and of the constant 1.
    let size_lanes = image.plane + u8::from(image.arrayed);
    let plane = match (lanes, size_lanes) {
        (l, n) if l == n => size,
        (1, _) => fb.composite_extract(size, &[0])?,
        (l, _) => {
            let idx: Vec<u32> = (0..u32::from(l)).collect();
            fb.vector_shuffle(size, size, &idx)?
        }
    };
    let one = |fb: &mut FunctionBuilder<'_>, ty: crate::spirv::ids::TypeId, n: u8| {
        let c = fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(1),
        })?;
        let c = if n == 1 {
            c
        } else {
            let vt = vec(fb, ty, n)?;
            fb.module().constant(Constant::Composite {
                ty: vt,
                parts: vec![c; usize::from(n)].into(),
            })?
        };
        fb.module().const_value(c)
    };
    let coord = plane_coord(cx, fb, s)?;
    let floor = fb.glsl(GlslOp::Floor, &[coord])?;
    let mut texel = fb.convert(ConvOp::ConvertFToS, floor, ivec)?;
    match offset {
        Some(Offset::Const(k)) => {
            let k = const_offset(fb, k)?;
            texel = fb.binary(BinOp::IAdd, texel, k)?;
        }
        Some(Offset::Dynamic(v, _)) => {
            let v = metal_offset(cx, fb, *v, lanes)?;
            texel = fb.binary(BinOp::IAdd, texel, v)?;
        }
        None => {}
    }
    // Clamp to edge: into `[0, size - 1]` per axis (a level-0 size is at least 1).
    let one_u = one(fb, u32_, lanes)?;
    let top = fb.binary(BinOp::ISub, plane, one_u)?;
    let top = fb.convert(ConvOp::Bitcast, top, ivec)?;
    let lo = fb.module().constant(Constant::Null { ty: ivec })?;
    let lo = fb.module().const_value(lo)?;
    let texel = fb.glsl(GlslOp::SClamp, &[texel, lo, top])?;
    let texel = fb.convert(ConvOp::Bitcast, texel, uvec)?;
    // The layer, which sampling clamps into the array.
    let layer = match s.layer {
        Some(l) => {
            let l = cx.operand(fb, l)?;
            let layers = fb.composite_extract(size, &[u32::from(lanes)])?;
            let one = one(fb, u32_, 1)?;
            let last = fb.binary(BinOp::ISub, layers, one)?;
            Some(fb.glsl(GlslOp::UMin, &[l, last])?)
        }
        None => None,
    };
    let at = match layer {
        Some(l) => {
            let ty = vec(fb, u32_, lanes + 1)?;
            fb.composite_construct(ty, &[texel, l])?
        }
        None => texel,
    };
    let ops = ImageOperands {
        lod: Some(zero),
        ..ImageOperands::default()
    };
    let fetch = move |fb: &mut FunctionBuilder<'_>, sparse: bool| -> BResult<SValue> {
        if sparse {
            fb.sparse_fetch(img, at, &ops)
        } else {
            fb.fetch(img, at, &ops)
        }
    };
    let tiled = match s.source {
        Source::Combined(key) => tile_map(cx, key.texture),
        _ => None,
    };
    let (x, word) = match (arm, tiled) {
        // A tile map: the plain fetch, and the residency of the fetched texel's tile at level 0.
        (Some((spec, _)), Some(map)) => {
            let x = fetch(fb, false)?;
            let zero = fb.u32_const(0)?;
            let zero = fb.module().const_value(zero)?;
            let tx = if lanes == 1 {
                texel
            } else {
                fb.composite_extract(texel, &[0])?
            };
            let ty = if lanes >= 2 {
                fb.composite_extract(texel, &[1])?
            } else {
                zero
            };
            let slice = layer.unwrap_or(zero);
            let word = tiled_word(cx, fb, spec, map, [tx, ty, zero], zero, slice)?;
            (x, Some(word))
        }
        (Some((spec, _)), None) => {
            let (x, word) = resident_access(cx, fb, spec, fetch)?;
            (x, Some(word))
        }
        (None, _) => (fetch(fb, false)?, None),
    };
    let x = if s.depth {
        fb.composite_extract(x, &[0])?
    } else {
        x
    };
    Ok((x, word))
}

/// `x`, the device's four-lane value of pixel fetch `s`, zero where its status word `word` (a
/// `u32`) says its texel is unmapped under [`Unmapped::Named`]: Metal reads zero memory there,
/// where a block of a sparse image that holds no mapped tile reads the device's value. The lanes
/// the format lacks take their constants after the held texel's conversion ([`held_constants`]).
fn unmapped_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    x: SValue,
    word: Option<SValue>,
) -> BResult<SValue> {
    if s.unmapped != Unmapped::Named {
        return Ok(x);
    }
    let Some(word) = word else {
        return Err(cx.fail("an access that names its texel without its residency"));
    };
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let kept = fb.binary(BinOp::IEqual, word, zero)?;
    let ty = fb.value_ty(x)?;
    let null = fb.module().constant(Constant::Null { ty })?;
    let null = fb.module().const_value(null)?;
    fb.select(kept, x, null)
}

/// `held`, the held texel of access `s`, with the lanes Metal reads as constants set to them
/// where the access takes them ([`Unmapped::Constants`] and [`Unmapped::Named`]): the slot's
/// `ConstantLanes` specialization names them, a gather's by its component; a swizzled slot's
/// sampling sets them after its swizzle instead ([`swizzled_value`]).
pub(super) fn held_constants(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    held: SValue,
) -> BResult<SValue> {
    let Source::Combined(key) = s.source else {
        return Ok(held);
    };
    // A swizzled slot's sampling sets them after its swizzle ([`swizzled_value`]).
    if s.unmapped == Unmapped::Device || s.swizzle {
        return Ok(held);
    }
    let gathered = match s.op {
        SampleOp::Gather { component, .. } => Some(component),
        _ => None,
    };
    crate::lower::texture::constant_lanes(cx, fb, key.texture, held, gathered)
}

/// The sampling result of image texel `texel`: the held texel ([`held_value`]) with the lanes
/// Metal reads as constants set ([`held_constants`]), as `{T, i8}` where the result is one, with
/// status word `word` (the sparse arm's; 0 where none is given, Metal's for a resident access).
pub(super) fn texel_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    image: super::CombinedImage,
    (texel, origin): (SValue, TexelOrigin),
    word: Option<SValue>,
) -> BResult<SValue> {
    let held = held_value(cx, fb, s, image, (texel, origin))?;
    let held = held_constants(cx, fb, s, held)?;
    with_word(cx, fb, &s.result, held, word)
}

/// The held texel of sampling `s`'s image texel `texel`: an 8-bit sRGB texel's raw code decoded
/// by the device's transfer, a slot's `half` or `float` texel converted by the bound format's class,
/// else the texel held as the result's type holds it; before the lanes Metal reads as constants
/// are set ([`held_constants`]).
pub(super) fn held_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    image: super::CombinedImage,
    (texel, origin): (SValue, TexelOrigin),
) -> BResult<SValue> {
    let Some(form) = s.texel else {
        return Err(cx.fail("a sample without a texel form"));
    };
    // An 8-bit sRGB texel's raw code, or the device's filter of the codes, decoded by the
    // device's transfer: a gather's lanes are one channel's, all alpha or all color.
    if s.raw {
        let color = match s.op {
            SampleOp::Gather { component, .. } => [component != 3; 4],
            _ => crate::lower::texture::SRGB_TEXEL,
        };
        return crate::lower::texture::srgb_texel(fb, form, texel, color);
    }
    // A `half` texel of a slot converts by the bound format's class: a gather's texel as a read's,
    // a sample's (nearest, linear, or fetched) filtered value as a sample's, each with the access's
    // lanes. Not where the sample's texel is the device's (`Sampling::half_texel`): it is held.
    let half = match s.source {
        Source::Combined(key) if !s.depth && s.half_texel => {
            half_class(cx, &ImageRef::Slot(key.texture), form)
        }
        Source::Heap { handle, .. } | Source::HeapSampler { handle, .. }
            if !s.depth && s.half_texel =>
        {
            crate::lower::texture::heap_half_class(cx, fb, handle, form)?
        }
        _ => None,
    };
    Ok(match half {
        Some(h) => {
            let lanes = sample_lanes(cx, fb, s)?;
            h.convert(fb, texel, lanes, origin)?
        }
        None => held_texel(fb, image.texel, form, texel)?,
    })
}

/// Whether sampler word `word` states a sampler that reads one texel of one level: nearest min and
/// mag filters, no anisotropy, and no linear mip filter (a `bool`).
fn word_reads_one(fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
    let filters =
        fb.u32_const(SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR | SamplerWord::ANISOTROPIC)?;
    let filters = fb.module().const_value(filters)?;
    let filters = fb.binary(BinOp::BitwiseAnd, word, filters)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let nearest = fb.binary(BinOp::IEqual, filters, zero)?;
    let mip = fb.u32_const(SamplerWord::MIP_FILTER)?;
    let mip = fb.module().const_value(mip)?;
    let mip = fb.binary(BinOp::BitwiseAnd, word, mip)?;
    let linear_mip = fb.u32_const(2 << SamplerWord::MIP_FILTER_SHIFT)?;
    let linear_mip = fb.module().const_value(linear_mip)?;
    let one_level = fb.binary(BinOp::INotEqual, mip, linear_mip)?;
    fb.binary(BinOp::LogicalAnd, nearest, one_level)
}

/// The lanes of sampling `s`'s texel for the class's conversion (`half::TexelLanes::code`): a 2D
/// gather's channel, one texel's lanes, or a value that may be filtered, as [`Sampling::one_texel`]
/// says; where a runtime or argument buffer's sampler's word decides, one texel's where the word
/// states nearest min and mag filters, no linear mip filter, and no anisotropy, else filtered.
pub(super) fn sample_lanes(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
) -> BResult<SValue> {
    use crate::lower::texture::half::TexelLanes;
    use crate::lower::texture::OneTexel;
    let word = match s.one_texel {
        OneTexel::Always => {
            let lanes = match s.op {
                SampleOp::Gather { component, .. } => TexelLanes::Gathered(component),
                _ => TexelLanes::Texel,
            };
            return lanes.value(fb);
        }
        OneTexel::Never => return TexelLanes::Filtered.value(fb),
        OneTexel::Word(n) => sampler_word(cx, fb, Pixel::Spec(n))?,
        OneTexel::Heap(h) => sampler_word(cx, fb, Pixel::Heap(h))?,
    };
    let one = word_reads_one(fb, word)?;
    let texel = TexelLanes::Texel.value(fb)?;
    let filtered = TexelLanes::Filtered.value(fb)?;
    fb.select(one, texel, filtered)
}

#[cfg(test)]
mod axis_tests;

#[cfg(test)]
mod tests {
    use super::{metal_bias_of, BIAS_EXPONENT_MAX, BIAS_EXPONENT_MIN};

    /// `Frexp`'s exponent of finite `x`: `x = m · 2^e` with `|m|` in `[1/2, 1)`; 0 for zero.
    fn frexp_exponent(x: f32) -> i32 {
        if x == 0.0 {
            return 0;
        }
        let bits = x.abs().to_bits();
        let biased = (bits >> 23) as i32;
        if biased == 0 {
            // A denormal: its exponent from the leading one of the mantissa.
            -126 - ((bits << 9).leading_zeros() as i32)
        } else {
            biased - 126
        }
    }

    /// `x · 2^e`, rounded once to `f32` (`Ldexp`).
    fn ldexp(x: f32, e: i32) -> f32 {
        (f64::from(x) * 2f64.powi(e)) as f32
    }

    /// `bias_gradients` on the CPU, with `Exp2` exact (rounded once to `f32`): the model of the
    /// emitted sequence, step for step.
    fn scaled(dx: [f32; 2], dy: [f32; 2], bias: f32) -> ([f32; 2], [f32; 2]) {
        let largest = dx.iter().chain(&dy).map(|x| x.abs()).fold(0.0f32, f32::max);
        let e = frexp_exponent(largest);
        let t = (bias + e as f32).clamp(BIAS_EXPONENT_MIN, BIAS_EXPONENT_MAX);
        let s = 2f64.powf(f64::from(t)) as f32;
        let m = |d: [f32; 2]| [ldexp(d[0], -e) * s, ldexp(d[1], -e) * s];
        (m(dx), m(dy))
    }

    /// The level of a scaled pair on a 16-texel plane (the larger gradient length, in texels).
    fn level(dx: [f32; 2], dy: [f32; 2]) -> f64 {
        let len = |d: [f32; 2]| (f64::from(d[0]).powi(2) + f64::from(d[1]).powi(2)).sqrt();
        (len(dx).max(len(dy)) * 16.0).log2()
    }

    /// The scaled derivatives are finite for every bias, never `0 · ∞` = NaN in a zero component,
    /// and give the level Metal's whole bias gives: the derivatives' level plus the bias wherever
    /// that is in range, at least 125 (past any texture's top level) wherever it is at least 125,
    /// and below 0 (magnified) wherever it is far below 0. An integer bias in range only moves
    /// exponents, so its pair is exactly the derivatives times `2^bias`; a zero pair stays zero
    /// (magnified).
    #[test]
    fn bias_gradients_are_finite_and_saturate_as_the_whole_bias() {
        // `a2_fragment_bias_clamp`'s 2^13 normalized units per pixel (level 17), an ordinary one,
        // and a denormal one.
        let derivatives = [8192.0f32, 0.75, 2f32.powi(-140)];
        let biases = [
            -17.5f32,
            -17.0,
            3.0,
            99.5,
            126.0,
            200.0,
            1000.0,
            f32::MAX,
            -100.0,
            -1000.0,
            f32::MIN,
        ];
        for d in derivatives {
            for bias in biases {
                let (x, y) = scaled([d, 0.0], [0.0, d], bias);
                assert!(
                    x.iter().chain(&y).all(|v| v.is_finite()),
                    "{d} {bias}: {x:?} {y:?}"
                );
                assert_eq!((x[1], y[0]), (0.0, 0.0), "{d} {bias}: a zero stays zero");
                let exact = (f64::from(d) * 16.0).log2() + f64::from(bias);
                let got = level(x, y);
                if exact >= 125.0 {
                    assert!(got >= 125.0, "{d} {bias}: {got} for {exact}");
                } else if exact < -100.0 {
                    assert!(got < 0.0, "{d} {bias}: {got} for {exact}");
                } else {
                    assert!((got - exact).abs() < 1e-3, "{d} {bias}: {got} for {exact}");
                }
                if bias.fract() == 0.0 && exact.abs() < 100.0 {
                    let times = |v: f32| (f64::from(v) * 2f64.powf(f64::from(bias))) as f32;
                    assert_eq!((x[0], y[1]), (times(d), times(d)), "{d} {bias}: exact");
                }
            }
        }
        assert_eq!(scaled([0.0; 2], [0.0; 2], 1000.0), ([0.0; 2], [0.0; 2]));
    }

    /// Every finite non-negative `half`, as `f32`, in increasing order (its bits' order): decoded
    /// from the bits alone, a reference independent of the rounding under test.
    fn halves() -> Vec<f32> {
        (0u32..0x7c00)
            .map(|bits| {
                let (e, m) = ((bits >> 10) as i32, f64::from(bits & 0x3ff));
                let x = if e == 0 {
                    m * 2f64.powi(-24)
                } else {
                    (1.0 + m / 1024.0) * 2f64.powi(e - 15)
                };
                x as f32
            })
            .collect()
    }

    /// `x` (finite, at most 65504 in magnitude) rounded to the nearest `half`, ties to the even
    /// one, by search in `table` ([`halves`]).
    fn to_half(table: &[f32], x: f32) -> f32 {
        let a = x.abs();
        let i = table.partition_point(|h| *h < a);
        let h = match (table.get(i.wrapping_sub(1)), table.get(i)) {
            (_, Some(&hi)) if hi == a => hi,
            (Some(&lo), Some(&hi)) => {
                let (dl, dh) = (f64::from(a) - f64::from(lo), f64::from(hi) - f64::from(a));
                if dl < dh || (dl == dh && (i - 1) % 2 == 0) {
                    lo
                } else {
                    hi
                }
            }
            (None, Some(&hi)) => hi,
            (Some(&lo), None) => lo,
            (None, None) => unreachable!("an empty table"),
        };
        h.copysign(x)
    }

    /// Metal's reading of bias `b`, by [`to_half`]: the rule as measured.
    fn reference(table: &[f32], b: f32) -> f32 {
        let h = to_half(table, b.clamp(-64.0, 64.0));
        ((h * 64.0).floor() / 64.0).clamp(-32.0, 31.984375)
    }

    /// The module's reading of a bias (`metal_bias_of`, which `metal_bias` emits step for step) is
    /// Metal's measured rule, `half` rounding to nearest even, the floor at `1/64` and the clamp,
    /// at every `half` and the neighbours and midpoints of every one up to 64 (each rounding's tie
    /// and its two sides), at every `f32` bias within `±2^-12` of a multiple of `1/64` in `[-40,
    /// 40]` sampled by a stride, and at the ends: a NaN stays a NaN, and a bias past Metal's range
    /// reads its end.
    #[test]
    fn a_bias_reads_as_metals_fixed_point_reading() {
        let table = halves();
        let mut points: Vec<f32> = Vec::new();
        for (i, &h) in table.iter().enumerate().filter(|(_, h)| **h <= 64.0) {
            points.push(h);
            if let Some(&next) = table.get(i + 1) {
                let mid = ((f64::from(h) + f64::from(next)) / 2.0) as f32;
                points.extend([mid, f32::from_bits(mid.to_bits() + 1)]);
                if mid.to_bits() > 0 {
                    points.push(f32::from_bits(mid.to_bits() - 1));
                }
            }
        }
        for k in -2560i32..=2560 {
            let edge = k as f32 / 64.0;
            for step in (-4096i32..=4096).step_by(37) {
                points.push(edge + step as f32 * 2f32.powi(-24));
            }
        }
        let mut checked = 0;
        for &p in &points {
            for b in [p, -p] {
                // Either zero reads level `lod` alike.
                assert_eq!(
                    (metal_bias_of(b) + 0.0).to_bits(),
                    (reference(&table, b) + 0.0).to_bits(),
                    "bias {b:e} ({:#010x})",
                    b.to_bits()
                );
                checked += 1;
            }
        }
        assert!(checked > 100_000, "{checked} biases");
        assert!(metal_bias_of(f32::NAN).is_nan());
        for (b, want) in [
            (40.0, 31.984375),
            (32.0, 31.984375),
            (31.99, 31.984375),
            (1e30, 31.984375),
            (f32::INFINITY, 31.984375),
            (-32.0, -32.0),
            (-40.0, -32.0),
            (f32::NEG_INFINITY, -32.0),
            (-1e-6, -1.0 / 64.0),
            (-2f32.powi(-25), 0.0),
            (1e-6, 0.0),
            (0.015624, 0.015625),
            (-0.0, 0.0),
        ] {
            assert_eq!(metal_bias_of(b), want, "bias {b:e}");
        }
    }
}
