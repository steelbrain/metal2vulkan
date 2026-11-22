//! Metal's texel-space position of a normalized coordinate, which the filters the device has no
//! form for read their footprints from (`bicubic`, `extended`, `heap_raw`, and the residency
//! `footprint`), and Metal's placement of a texel offset on a repeating or mirrored axis.
//!
//! **The position** (Apple M3 Max, macOS 26.7, and Apple M4, macOS 27.0.1, byte-identical; the
//! tie, gather, and bicubic probes). Per axis of size `N` and normalized coordinate `u`, Apple's
//! texture unit first wraps `u` into one period, in `f32`, where the axis repeats: `u' = u -
//! floor(u)` under `repeat`, `u' = u - 2 floor(u / 2)` under `mirrored_repeat` (a period of two),
//! the subtraction rounded toward positive infinity (`wrap`; it rounds at all only where `-1 < u <
//! 0`), or exactly where `N` is a power of two (below). Then it scales: `X = u' N` rounded to `f32`
//! with ties away from zero. A clamping axis scales `u` itself. Every filter reads its footprint
//! from `X`: the linear and bicubic filters and a gather the quantized `q = floor(256 X + 1/2) -
//! 128`, a nearest filter `floor(X)`, and the taps then wrap modulo `N` or mirror with period `2N`.
//! The probes measured the wrap on the linear, nearest, gather, and bicubic filters (every lane,
//! both chips), and the ties on the linear, gather, and bicubic filters under `clamp_to_zero`,
//! where a product exactly half a unit in the last place from two `f32`s decides the footprint. The
//! wrap's rounding comes from the position cases (`wrg_sampler_positions` at 13x11, 7x5, and 255x3
//! on the M3, whose outputs the M4's equal byte for byte): rounded up, every linear, nearest,
//! gather, and bicubic result of their repeating and mirrored axes is Metal's; rounded to nearest
//! even, with ties away, or not at all, it misses lanes with `-1 < u < 0` whose wrapped `u'` needs
//! more bits than an `f32` holds.
//!
//! **A power-of-two size wraps exactly**. On a repeating or mirrored axis whose size is a power of
//! two, Metal reads the footprints of the exact `u'`, per axis: the linear-filter probes (M4 and
//! M3, byte-identical) at 2D 16x8 and 4x32 and 3D 16x16x16, 8x4x2 and 16x8x4, and on the 4-texel
//! axis of a 9x4x11 texture whose other axes round up, and the position cases at 16x8, 4x32 and
//! 16x11 (linear, nearest, and gathers, `gather_compare` included; M3 and M4). `wrap` rounds the
//! subtraction down there, which reads the same footprints: `u' N` is then exact, and every step a
//! footprint reads (an integer, or `i + 1/2` plus a multiple of `1/512`) lies on the `1/512` grid,
//! which an `f32` holds exactly below `2^15`, so rounding `u'` down never carries `X` across one.
//! Rounded up, it reads one `1/256` step, sometimes a texel, high where `-1 < u < 0` lands just
//! below a step.
//!
//! Multiplying first and wrapping the taps after (as a Vulkan device does) reads another footprint wherever `u` lies outside `[0, 1)` on a size that is not a
//! power of two and `u N` lands near a step: the product rounds at the magnitude of `u N`, not of
//! `u' N`. At a power-of-two size `u N` is exact, and multiplying first reads Metal's footprints.
//!
//! **Ties away from zero, in integers** ([`position`]). `p = u' N` rounded to nearest even is a tie
//! rounded toward zero exactly when the bits of the exact product below `p`'s last place are
//! `100…0` and `p`'s significand is even; then `X` is the next `f32` away from zero, `p`'s bits
//! plus one. The exact product's low bits are those of `bits(u') * N` (an `OpIMul`, modulo `2^32`):
//! `u'`'s exponent and implicit bit sit at bit 23 and up, so they never reach the bits the test
//! reads. Those are bits `0` to `s`, where `s` is `p`'s biased exponent less `u'`'s (the sign bits
//! cancel, `p` having `u'`'s sign); a shift left by `31 - s` puts them at the top of the word,
//! where the tie reads `0x4000_0000`. This is exact for every finite normal `u'` and every
//! `N < 2^22`, where `s <= 22` and the bits stay below bit 23; Metal's sizes are at most `2^14`. A
//! denormal `u'` may be adjusted wrongly, by one unit in the last place of an `X` below `2^-100`,
//! which moves no footprint. A tie moves a footprint only at a negative position, or at one of
//! `2^14` or more. Where the two neighbours of a tie lie on either side of a footprint's step (an
//! integer, or a gather's `i + 1/2 - 1/512`), the upper neighbour is the step, whose significand is
//! even below `2^14`: the even rounding picks it, as ties away from zero does for a positive
//! position, while at a negative one ties away from zero picks the lower. That is why the probes'
//! deciding lanes were clamped axes' negative coordinates, and a repeating or mirrored axis, whose
//! position is never negative, differs only at sizes of at least `2^13` (mirrored) or `2^14`
//! (repeating). A product past `f32`'s range gives an unspecified `X`, as before (the footprints
//! read no non-finite position: `bicubic`). The test is 12 integer instructions on every device.
//!
//! **A texel offset on a repeating or mirrored axis** ([`offset_dropped`]). Metal adds a constant
//! texel offset to the tap and then wraps it, as Vulkan places it, but on a repeating or mirrored
//! axis whose level size `n` lies in `[2, 7]` and whose texture's base-level size on that axis is
//! not a power of two: there it drops the offset on that axis. At the base level that is a size of
//! 3, 5, 6, or 7. The offset probes:
//! - round 1, Apple M4 on macOS 27.0.1 and Apple M3 Max on macOS 26.7, byte-identical, each run
//!   with and without Metal's validation layer: every 2D size from 1 to 17 per axis and eight
//!   larger controls, offsets -8 to 7 per axis, nearest and linear samples, gathers of each
//!   component, and a depth texture's gathers; a clamping axis places the offset as Vulkan does
//!   on every size;
//! - round 2, the M4: base levels up to 16384, 2D arrays and 3D textures, per axis;
//! - round 3, the M4: every level of 18 bases at an explicit level, nearest and linear samples
//!   (level sizes 2 and 4 drop it under a base that is not a power of two and keep it under one
//!   that is; sizes 3, 5 and 6 drop it, and 8 and above keep it, under either), and each level
//!   above the base through a view of that one level, which reads exactly as the level does: the
//!   view's parent's base-level size decides.
//!
//! An emulated footprint applies the rule to its taps (`bicubic::axis`). The base-level size the
//! rule reads is a view's parent's, which a shader
//! cannot query: a view whose first level lies above its parent's base reads the bound view's
//! own, so on an axis of size 2 or 4 under a parent base that is not a power of two (where the
//! view's base is) the rule would keep the offset Metal drops. The binder states those axes in the
//! slot's `SpecKey::OffsetBaseAxes` specialization (0 for every texture that is not such a view),
//! whose tests (`texture::OffsetBase`) the rule reads; a heap element's access reads none, and
//! records `PrecisionNote::ViewBaseTexelOffset` (the residue there).
//!
//! **A runtime sampler's position** ([`position_by_mode`]): an emulation of a runtime or argument
//! buffer's sampler reads the address mode from the sampler's word in the function. On a
//! `MirrorClampToEdge` axis (which no constexpr sampler states) Metal's position is a clamping
//! axis's, `u N` unwrapped, its taps mirrored once and clamped as Vulkan's `MIRROR_CLAMP_TO_EDGE`
//! mirrors and clamps them, and a texel offset is dropped by the sizes that drop it on a repeating
//! axis (every gather of 13x11, 7x5 and 16x8 textures through a slot's and an argument buffer's
//! runtime sampler, with offsets of 2 and -3, on an Apple M3 Max and an Apple M4 alike), a pixel
//! coordinate's as a normalized one's.

use crate::metal::SamplerAddressMode;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, Type as SType};

use super::bicubic::TapAddress;

/// The census words of one axis's [`position`]: the conversion of the size, the wrap (a mirrored
/// axis's four instructions, and the rounding's thirteen, which choose up or down by the size), the
/// product, and the ties' twelve integer instructions, with their constants (186 words on a
/// mirrored axis). Every estimate that emits a position adds it per axis.
pub const POSITION_WORDS: u32 = 200;

fn f32_const(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn u32_const(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

/// The census words of [`offset_dropped`] and of a footprint's placement of the offset by it: the
/// subtractions, the comparisons, the mask and their conjunction, the select, and their constants,
/// in the leveled form (the larger).
pub const OFFSET_RULE_WORDS: u32 = 64;

/// Whether Metal drops a constant texel offset on a repeating or mirrored axis whose level size is
/// `n` (a `u32`), a `bool` (module docs): `n` lies in `[2, 7]` and the axis's base-level size
/// `base` (a `u32`; `None`: `n` is the base level's) is not a power of two, `n - 2 <= 5` unsigned
/// and `base & (base - 1) != 0`. At the base level that is `n` of 3, 5, 6, or 7, `n - 3 <= 4`
/// unsigned and `n != 4`. Every other size, and every clamping axis, takes the offset. The
/// base-level size Metal reads is a view's parent's: where `parent` (a `bool`, the axis's
/// `texture::OffsetBase` test) holds, it is not a power of two though the bound view's is, so the
/// offset drops at every `n` in `[2, 7]`.
pub fn offset_dropped(
    fb: &mut FunctionBuilder<'_>,
    n: SValue,
    base: Option<SValue>,
    parent: Option<SValue>,
) -> BResult<SValue> {
    let (base, parent) = match (base, parent) {
        (None, None) => {
            let three = u32_const(fb, 3)?;
            let four = u32_const(fb, 4)?;
            let t = fb.binary(BinOp::ISub, n, three)?;
            let small = fb.binary(BinOp::ULessThanEqual, t, four)?;
            let odd_one = fb.binary(BinOp::INotEqual, n, four)?;
            return fb.binary(BinOp::LogicalAnd, small, odd_one);
        }
        (base, parent) => (base.unwrap_or(n), parent),
    };
    let two = u32_const(fb, 2)?;
    let five = u32_const(fb, 5)?;
    let t = fb.binary(BinOp::ISub, n, two)?;
    let small = fb.binary(BinOp::ULessThanEqual, t, five)?;
    let one = u32_const(fb, 1)?;
    let below = fb.binary(BinOp::ISub, base, one)?;
    let mask = fb.binary(BinOp::BitwiseAnd, base, below)?;
    let zero = u32_const(fb, 0)?;
    let uneven = fb.binary(BinOp::INotEqual, mask, zero)?;
    let uneven = match parent {
        Some(p) => fb.binary(BinOp::LogicalOr, uneven, p)?,
        None => uneven,
    };
    fb.binary(BinOp::LogicalAnd, small, uneven)
}

/// The census words a slot's `SpecKey::OffsetBaseAxes` test adds to [`offset_dropped`] on each
/// plane axis an offset moves, per footprint that places it: the general rule's
/// words where they replace the base level's shorter ones, the disjunction, and their constants,
/// and a share of the slot's tests' declaration (a specialization constant and three
/// `OpSpecConstantOp` tests per axis, `texture::OffsetBase`).
pub const PARENT_RULE_WORDS: u32 = 64;

/// Metal's position `X` (an `f32`, module docs) of normalized coordinate `u` (an `f32`) on an axis
/// of size `n` (a `u32`, below `2^22`) addressed by `address`, with the size as an `f32`.
pub fn position(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    address: TapAddress,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    // The wrap into one period, in `f32`, its subtraction rounded up, or down at a power-of-two
    // size.
    let u = match address {
        TapAddress::Edge | TapAddress::Border => u,
        TapAddress::Repeat => {
            let fl = fb.glsl(GlslOp::Floor, &[u])?;
            wrap(fb, u, fl, n, (one, zero))?
        }
        TapAddress::Mirror => {
            let twice = mirror_start(fb, u)?;
            wrap(fb, u, twice, n, (one, zero))?
        }
    };
    scale(fb, u, n, (one, zero))
}

/// The census words [`position_by_mode`] adds to [`position`]'s: both periods' starts, the
/// mode's two tests and selects, and their constants.
pub const MODE_POSITION_WORDS: u32 = 64;

/// [`position`] on an axis whose address mode is `mode`, a `u32` holding `MTLSamplerAddressMode`'s
/// value (a runtime sampler's, read from its `SamplerWord` in the function): the period starts at
/// `floor(u)` under `Repeat` and at `2 floor(u / 2)` under `MirrorRepeat`, and at 0 under every
/// other mode, where the wrap is `u` itself (`u - 0` and its sum with 0 are `u`, so the rounding's
/// step is 0) and the product scales `u`, as a clamping axis's does.
pub fn position_by_mode(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    mode: SValue,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    let fl = fb.glsl(GlslOp::Floor, &[u])?;
    let twice = mirror_start(fb, u)?;
    let none = f32_const(fb, 0.0)?;
    let repeat = u32_const(fb, SamplerAddressMode::Repeat.as_raw())?;
    let mirror = u32_const(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
    let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
    let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
    let k = fb.select(mirrors, twice, none)?;
    let k = fb.select(repeats, fl, k)?;
    let u = wrap(fb, u, k, n, (one, zero))?;
    scale(fb, u, n, (one, zero))
}

/// The start `2 floor(u / 2)` of a mirrored axis's period of two around `u` (an `f32`).
fn mirror_start(fb: &mut FunctionBuilder<'_>, u: SValue) -> BResult<SValue> {
    let half = f32_const(fb, 0.5)?;
    let h = fb.binary(BinOp::FMul, u, half)?;
    let fl = fb.glsl(GlslOp::Floor, &[h])?;
    fb.binary(BinOp::FAdd, fl, fl)
}

/// Metal's position `X` of the wrapped coordinate `u` (an `f32`) on an axis of size `n` (a `u32`):
/// the product rounded with ties away from zero (module docs), with the size as an `f32`;
/// `(one, zero)` are `u32`s.
fn scale(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<(SValue, SValue)> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
    // The product rounded to nearest even, on its own (in the profile's form, never fused into what
    // follows).
    let p = fb.binary_no_contraction(BinOp::FMul, u, nf)?;
    // A tie rounded toward zero: bits `0..=s` of `bits(u) * n` are `0100…0` (module docs).
    let bu = fb.convert(ConvOp::Bitcast, u, w)?;
    let bp = fb.convert(ConvOp::Bitcast, p, w)?;
    let lo = fb.binary(BinOp::IMul, bu, n)?;
    let k23 = u32_const(fb, 23)?;
    let eu = fb.binary(BinOp::ShiftRightLogical, bu, k23)?;
    let ep = fb.binary(BinOp::ShiftRightLogical, bp, k23)?;
    let s = fb.binary(BinOp::ISub, ep, eu)?;
    let k31 = u32_const(fb, 31)?;
    let sh = fb.binary(BinOp::ISub, k31, s)?;
    let top = fb.binary(BinOp::ShiftLeftLogical, lo, sh)?;
    let tie_bits = u32_const(fb, 0x4000_0000)?;
    let tie = fb.binary(BinOp::IEqual, top, tie_bits)?;
    let up = fb.select(tie, one, zero)?;
    let bx = fb.binary(BinOp::IAdd, bp, up)?;
    Ok((fb.convert(ConvOp::Bitcast, bx, f)?, nf))
}

/// `u - k` rounded toward positive infinity, or toward negative infinity where the size `n` (a
/// `u32`) is a power of two, for an `f32` coordinate `u` and the integer `k` (an `f32`) of its
/// period's start (`floor(u)`, or `2 floor(u / 2)` mirrored), with `(1, 0)` as `u32`s (module
/// docs). The difference rounded to nearest even, `s`, is inexact only where `-1 < u < 0` (`k` is
/// `-1`, or `-2` mirrored, and `s` lies in `[1/2, 1]`, or `[1, 2]`), and there `z = s + k` is
/// exact by Sterbenz's lemma; elsewhere `s` is exact and `z` is `u`. So the exact difference
/// `u - k` exceeds `s` exactly when `u > z`, and then the next `f32` up from `s` (which is not
/// negative) is its bits plus one; it lies below `s` exactly when `u < z`, and then the next `f32`
/// down is its bits minus one. The three float operations are `NoContraction`: a device rounds each
/// on its own and never folds `(u - k) + k` to `u`.
fn wrap(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    k: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let s = fb.binary_no_contraction(BinOp::FSub, u, k)?;
    let z = fb.binary_no_contraction(BinOp::FAdd, s, k)?;
    let below = fb.binary(BinOp::FOrdLessThan, z, u)?;
    let above = fb.binary(BinOp::FOrdLessThan, u, z)?;
    let up = fb.select(below, one, zero)?;
    let minus_one = u32_const(fb, u32::MAX)?;
    let down = fb.select(above, minus_one, zero)?;
    // A power of two, `n & (n - 1) == 0`, wraps exactly (module docs).
    let m = fb.binary(BinOp::ISub, n, one)?;
    let m = fb.binary(BinOp::BitwiseAnd, n, m)?;
    let power_of_two = fb.binary(BinOp::IEqual, m, zero)?;
    let step = fb.select(power_of_two, down, up)?;
    let bs = fb.convert(ConvOp::Bitcast, s, w)?;
    let bs = fb.binary(BinOp::IAdd, bs, step)?;
    fb.convert(ConvOp::Bitcast, bs, f)
}

#[cfg(test)]
mod tests;
