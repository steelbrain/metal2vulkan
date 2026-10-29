//! Metal's bicubic filter (`filter::bicubic`), emulated by fetches.
//!
//! **The filter Metal applies** (Apple M3 Max, macOS 26.7; authored `bcu_*` cases and probes). Per
//! plane axis of a sample through a constexpr bicubic sampler, the texel-space coordinate `x`
//! (Metal's position of the normalized coordinate on the level-0 size, wrapped into one period
//! first on a repeating axis and scaled with ties away from zero, `position`; or the pixel
//! coordinate) is read as `q = floor(256 (x - 1/2) + 1/2)`: the footprint's base texel `i = q >> 8`
//! and the fraction `f = (q & 255) / 256`, a fraction rounded half up to 1/256. The four taps `i -
//! 1 .. i + 2` (plus any constant texel offset, which a repeating axis of 3, 5, 6, or 7 texels
//! drops as Metal's other filters do: `placed_offset`; the round-3 offset probe, the M3 Max and an
//! M4 byte-identical) take the Catmull-Rom weights of `f` (`a = -1/2`), the 2D filter is their
//! product, and each tap is addressed on its own: `clamp_to_edge` clamps its index into the level,
//! `repeat` wraps it, and `clamp_to_zero` and `clamp_to_border` replace an out-of-range tap by the
//! sampler's border color seen through the bound format's channels (`metal::bicubic`;
//! `clamp_to_zero` reads the border color too). The level is always the base level (Metal's bicubic
//! sampler has no mip filter, so the level, bias, gradients, and fragment derivatives select
//! nothing), a 2D array's layer is clamped into the array, and what is done with the sum depends on
//! the bound format: the [`SpecKey::BicubicTexels`] specialization of the texture slot selects the
//! bilinear inner 2x2 instead (32-bit float channels), clamps the sum into `[0, 1]` or `[-1, 1]`
//! (normalized channels), or rounds it to `half` (half and small-float channels). A 1D texture is
//! filtered bilinearly over taps `i` and `i + 1` whatever its format, with the same fraction.
//!
//! **Small levels and mirrored axes** (Apple M3 Max, macOS 26.7, the primary Metal reference; the
//! probes probes, every lane of each run with and without Metal's validation layer,
//! `r8Unorm`, `rgba8Unorm`, and `rgba16Float` on random texels at dyadic coordinates from 2.5
//! periods below to 3.5 above):
//! - A 2D level below four texels on either axis (1x1 to 13x4, 15 sizes) is filtered by Metal's
//!   linear filter under the sampler's address modes: its sample equals a linear sampler's with
//!   the same modes on all 4,096 lanes of each run, under `clamp_to_edge`, `repeat`,
//!   `mirrored_repeat`, `clamp_to_zero`, and one mirrored axis beside a clamped one. The
//!   emulation takes the linear 2D sample there ([`routines::Quad`]) over the footprint's inner
//!   taps, which are the linear filter's (the bicubic and linear footprints share `q`).
//! - On a level of at least four texels each way, a mirrored axis of a format Metal filters
//!   bicubically reads texel 1 at fraction 0 whatever the coordinate: with both axes mirrored
//!   every lane is texel `(1, 1)` exactly, and with one the sample varies with the other axis
//!   alone and equals a clamping sampler's at texel 1's centre (17 sizes from 4x4 to 64x64, odd and
//!   even, square and not, private and shared storage). The emulation pins the axis's base tap to
//!   1 and its fraction to 0. A 32-bit float format, whose bicubic filter is bilinear, mirrors as
//!   the linear filter does (its sample equals a linear mirrored sampler's on every lane at every
//!   size): there the axis keeps the mirrored position and its taps reflect.
//! - The M4 (macOS 27.0.1, the secondary, whose bicubic family its claim excludes) answers the
//!   small levels and the 32-bit float formats alike, but on larger levels its mirrored axis reads
//!   no texel 1: a power-of-two extent clamps as `clamp_to_edge` would, and any other wraps the
//!   footprint's first tap into `[0, 2n)` and clamps the rest at the far edge. The translation
//!   follows the primary.
//! - A 1D texture, pixel coordinates (which Metal's front end admits only with clamping modes), a
//!   constant offset on a mirrored axis, and a gather through a mirrored bicubic sampler are not
//!   measured, and are refused (below).
//!
//! **The arithmetic after the fetches** is the device's, in module routines every sample calls
//! ([`routines`]): the Catmull-Rom weights computed in `f32` from the fraction, the sixteen taps
//! summed by the device's multiply and add, the bilinear inner taps by its `FMix` where Metal's rule
//! filters bilinearly (a 32-bit float format, a level under four texels), the normalized formats'
//! clamps, and a `half` result the device's narrowing of the `f32` sum (`arith::narrow::round`).
//! Metal's own datapath (its fixed-point weight table, exact integer sums rounded half up onto the
//! format's grid, its float accumulators and its `half` rounding of a half format's sum) is not
//! reproduced: the sample records `PrecisionNote::BicubicFilterArithmetic`. An 8-bit sRGB
//! texture held as its UNORM twin reads the raw codes through a UNORM view under its
//! `SrgbTexture` variant ([`Bicubic::raw`]) and decodes each tap by the device's transfer
//! (`PrecisionNote::SrgbTexelDecode`) before the sum.
//!
//! Everything else above is Metal's reading, except inferences:
//! - The signed clamp's lower bound comes from the format's range (no reading went below `-1`).
//! - The small-level and mirrored-axis rules of the formats our probes did not read
//!   (all but `R8Unorm`, `RGBA8Unorm`, `RGBA16Float`, and `R32Float`) are their families'.
//! - `RG8UnormSRGB`, never probed, takes its family's treatment. So do `RG8Unorm`, `RG8Snorm`,
//!   `RG16Unorm`, `BGRA8Unorm`, and `A8Unorm`, which take the exact rule no case reads on them.
//! - `Depth32Float` is filtered as the 32-bit float formats are, by family: no case reads a
//!   bicubic sum of it.
//!
//! **Refused** (`NotImplemented(SamplerBicubicFilter)`): `min_lod_clamp` (Apple's device compiler
//! fails to build a pipeline that samples a bicubic sampler with it, so Metal gives no reference);
//! `mirrored_repeat` on a 1D texture, with pixel coordinates, beside a constant offset on the
//! mirrored axis, or in a gather (not measured); a dynamic texel offset, a depth comparison, an LOD query, a 3D or
//! cube texture, a depth or integer texture, a heap texture (no slot specializes its format for a
//! sample, and no reading covers a gather through a bicubic sampler on one), anisotropy above 1, and one sampler clamping to zero on one axis and to a
//! non-transparent-black border on another: none is measured. A gather through a bicubic sampler
//! on a texture slot is an ordinary gather (measured), through the immutable sampler
//! [`constexpr_desc`] makes.
//!
//! [`SpecKey::BicubicTexels`]: crate::pipeline::sym::SpecKey::BicubicTexels
//! [`constexpr_desc`]: super::constexpr_desc

use crate::air::abi::{AddressMode, BorderColor, Filter, SamplerState};
use crate::air::intrinsics::ImageDim;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::texture::{OffsetBase, TexelConv};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::sampler::SamplerWord;
use crate::metal::{SamplerAddressMode, SamplerBorderColor};
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{ConstId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageOperands, ImageQuery, Type as SType, UnOp,
};

use super::{CombinedImage, Sampling};

pub mod routines;

fn refuse() -> Error {
    Unsupported::not_implemented(Feature::SamplerBicubicFilter).into()
}

/// How a tap outside the level is addressed on one plane axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TapAddress {
    /// `clamp_to_edge`: the index clamped into the level.
    Edge,
    /// `repeat`: the index wrapped into the level.
    Repeat,
    /// `clamp_to_zero` or `clamp_to_border`: the tap replaced by the border color.
    Border,
    /// `mirrored_repeat`: the index reflected into the level, `-1 - i` for a negative `i`, then
    /// taken modulo twice the extent `n` and reflected back from `n` on (Metal's linear filter; the
    /// bicubic filter's where it filters linearly, and texel 1 elsewhere: the module docs).
    Mirror,
}

/// A sample through a bicubic sampler: the fetches, weights, and per-format treatment the module
/// docs state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bicubic {
    /// The texture slot whose `BicubicTexels` specialization states the bound format's treatment.
    pub texture: u32,
    /// The addressing of each plane axis (`s`, then `t` for a 2D texture).
    pub address: Vec<TapAddress>,
    /// The sampler's border color, which every axis clamping to zero or to the border reads.
    pub border: BorderColor,
    /// Pixel coordinates: the plane coordinate is already in texels.
    pub pixel: bool,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The sample reads an 8-bit sRGB texture's raw codes, its slot in the `SrgbTexture`
    /// variant: each tap's color lanes become the device's decode of their codes (`srgb_raw_tap`)
    /// before the sum.
    pub raw: bool,
}

impl Bicubic {
    /// A 1D texture, filtered bilinearly whatever its format.
    fn linear_only(&self) -> bool {
        self.address.len() == 1
    }

    /// The census words: the footprint (each axis's position, which a pixel coordinate does not
    /// take), per tap its addressing and fetch, and the calls of the filter's routines, whose own
    /// words the module counts once ([`routines::Routine::words`]), and a raw sample's tap
    /// conversions; for a 2D texture the small level's test and linear sample ([`SMALL_WORDS`], its
    /// four taps' conversions on a raw sample) and each mirrored axis's pinned footprint
    /// ([`PINNED_WORDS`]), and Metal's offset rule on each axis a constant offset moves
    /// (`position::OFFSET_RULE_WORDS`). The estimate bounds every public bicubic fixture
    /// (`the_estimate_bounds_the_bicubic_fixtures`), every raw form
    /// (`the_srgb_estimate_bounds_its_emission`), and the mirrored forms (`mirrored_axes`).
    pub(super) fn words(&self) -> u32 {
        let (base, taps) = if self.linear_only() {
            (448, 2)
        } else {
            (768 + SMALL_WORDS, 20)
        };
        let position = if self.pixel {
            0
        } else {
            self.address.len() as u32 * super::position::POSITION_WORDS
        };
        let mirrored = self
            .address
            .iter()
            .filter(|a| **a == TapAddress::Mirror)
            .count() as u32;
        // Metal's offset rule on each axis the offset moves.
        let offset = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32)
            * super::position::OFFSET_RULE_WORDS;
        base + position
            + mirrored * PINNED_WORDS
            + offset
            + if self.raw {
                taps * crate::lower::texture::SRGB_TAP_WORDS
            } else {
                0
            }
    }
}

/// The census words of a 2D sample's small-level path ([`Bicubic::words`]): the level's test, the
/// four inner taps' fetches, the merged fractions, the linear routines' calls, and the selects.
const SMALL_WORDS: u32 = 256;

/// The census words of a mirrored axis's pinned footprint ([`Bicubic::words`]): the format's and
/// the level's tests and the selects of the fraction and the four taps.
const PINNED_WORDS: u32 = 96;

/// The decoded state of sampler `r` when it is a constexpr sampler with Metal's bicubic filter.
pub(super) fn state(r: SamplerRef) -> Result<Option<SamplerState>, Error> {
    let SamplerRef::Constexpr(words) = r else {
        return Ok(None);
    };
    let s = super::state_of(words)?;
    Ok((s.min_filter == Filter::Bicubic || s.mag_filter == Filter::Bicubic).then_some(s))
}

/// The state of a runtime sampler whose min or mag filter is Metal's `Cubic`, from its word's
/// `SamplerWord::CUBIC_STATE` bits (`LoweringInputs::cubic_samplers`): Metal samples through such
/// a sampler as through a constexpr bicubic sampler of the same address modes, border color, and
/// coordinates, whatever its other filter and the level (990 runs of 8,192
/// lanes, nine formats, every address mode and border color, normalized and pixel coordinates,
/// min/mag pairs `(2, 2)`, `(2, 1)`, `(1, 2)`, `(2, 0)` and `(0, 2)`, implicit, explicit and
/// gradient levels, bit for bit; through an argument buffer alike). `clamp_to_zero` reads the
/// sampler's border color, as a runtime sampler's does. The state is both filters bicubic, without
/// a mip filter, isotropic, under the weighted-average reduction: a word that states anything else
/// is not measured (a mip filter, anisotropy, another reduction, an unnamed border color, or a
/// `MirrorClampToEdge` axis, which no bicubic reading covers), and is refused here by the rule the
/// binder refuses it by before it names the variant (`metal::bicubic::runtime_cubic_measured`).
pub(super) fn cubic_state(word: u32) -> Result<SamplerState, Error> {
    use crate::air::abi::{CompareFunction, Coordinates, Half, MipFilter, Reduction};
    if !crate::metal::bicubic::runtime_cubic_measured(word) {
        return Err(refuse());
    }
    let field = |shift: u32, mask: u32| (word & mask) >> shift;
    let address = |shift: u32, mask: u32| -> Result<AddressMode, Error> {
        match SamplerAddressMode::try_from(field(shift, mask)) {
            Ok(SamplerAddressMode::ClampToEdge) => Ok(AddressMode::ClampToEdge),
            Ok(SamplerAddressMode::Repeat) => Ok(AddressMode::Repeat),
            Ok(SamplerAddressMode::MirrorRepeat) => Ok(AddressMode::MirroredRepeat),
            Ok(SamplerAddressMode::ClampToZero) => Ok(AddressMode::ClampToZero),
            Ok(SamplerAddressMode::ClampToBorderColor) => Ok(AddressMode::ClampToBorder),
            _ => Err(refuse()),
        }
    };
    let border_color =
        match SamplerBorderColor::try_from(field(SamplerWord::BORDER_SHIFT, SamplerWord::BORDER)) {
            Ok(SamplerBorderColor::TransparentBlack) => BorderColor::TransparentBlack,
            Ok(SamplerBorderColor::OpaqueBlack) => BorderColor::OpaqueBlack,
            Ok(SamplerBorderColor::OpaqueWhite) => BorderColor::OpaqueWhite,
            _ => return Err(refuse()),
        };
    let [s_shift, t_shift, r_shift] = SamplerWord::ADDRESS_SHIFT;
    let [s_mask, t_mask, r_mask] = SamplerWord::ADDRESS;
    Ok(SamplerState {
        address_s: address(s_shift, s_mask)?,
        address_t: address(t_shift, t_mask)?,
        address_r: address(r_shift, r_mask)?,
        mag_filter: Filter::Bicubic,
        min_filter: Filter::Bicubic,
        mip_filter: MipFilter::None,
        coordinates: if word & SamplerWord::PIXEL_COORDINATES != 0 {
            Coordinates::Pixel
        } else {
            Coordinates::Normalized
        },
        compare: CompareFunction::Never,
        max_anisotropy: 1,
        lod_min_clamp: Half(0),
        lod_max_clamp: Half(0),
        border_color,
        reduction: Reduction::WeightedAverage,
        lod_bias: Half(0),
    })
}

/// The checks every operation through bicubic sampler `s` passes, on a texture of dimension
/// `dim` (`depth`, `integer`: its kind; `heap`: an argument-buffer texture), the operation a
/// filtered sample where `sample` (not a gather or an LOD query): what the module docs refuse of
/// the sampler, the texture, and the operation.
pub(super) fn check(
    s: &SamplerState,
    dim: ImageDim,
    (depth, integer, heap): (bool, bool, bool),
    sample: bool,
) -> Result<(), Error> {
    let axes = match dim {
        ImageDim::D1 => 1,
        ImageDim::D2 => 2,
        ImageDim::D3 | ImageDim::Cube | ImageDim::Buffer => return Err(refuse()),
    };
    // Metal's front end admits a bicubic sampler only with both filters bicubic and no mip
    // filter; anything else is not the filter measured.
    let bicubic_only = s.min_filter == Filter::Bicubic
        && s.mag_filter == Filter::Bicubic
        && s.mip_filter == crate::air::abi::MipFilter::None;
    if !bicubic_only || depth || integer || heap || s.max_anisotropy > 1 {
        return Err(refuse());
    }
    let modes = plane_modes(s, axes);
    // A mirrored axis is measured for a 2D texture's sample at normalized coordinates (module
    // docs).
    let pixel = s.coordinates == crate::air::abi::Coordinates::Pixel;
    if modes.contains(&AddressMode::MirroredRepeat) && (axes == 1 || pixel || !sample) {
        return Err(refuse());
    }
    Ok(())
}

/// The check a bicubic sample with constant texel offset `offset` passes beyond [`check`]'s:
/// no offset on a mirrored axis, whose footprint with an offset Metal was not measured on.
fn check_offset(s: &SamplerState, offset: Option<&[i32]>) -> Result<(), Error> {
    let modes = [s.address_s, s.address_t];
    let shifted = offset
        .unwrap_or(&[])
        .iter()
        .zip(modes)
        .any(|(o, m)| *o != 0 && m == AddressMode::MirroredRepeat);
    if shifted {
        return Err(refuse());
    }
    Ok(())
}

/// The address modes of the first `axes` plane axes (`s`, then `t`).
fn plane_modes(s: &SamplerState, axes: usize) -> Vec<AddressMode> {
    [s.address_s, s.address_t].into_iter().take(axes).collect()
}

/// The bicubic sample of texture slot `texture` (dimension `dim`) through sampler `s`, which
/// passed [`check`], with constant texel offset `offset`.
pub(super) fn plan(
    s: &SamplerState,
    texture: u32,
    dim: ImageDim,
    offset: Option<Vec<i32>>,
) -> Result<Bicubic, Error> {
    let axes = if dim == ImageDim::D1 { 1 } else { 2 };
    let modes = plane_modes(s, axes);
    let address = modes
        .iter()
        .map(|m| match m {
            AddressMode::ClampToEdge => Ok(TapAddress::Edge),
            AddressMode::Repeat => Ok(TapAddress::Repeat),
            AddressMode::ClampToZero | AddressMode::ClampToBorder => Ok(TapAddress::Border),
            AddressMode::MirroredRepeat => Ok(TapAddress::Mirror),
        })
        .collect::<Result<Vec<_>, Error>>()?;
    check_offset(s, offset.as_deref())?;
    Ok(Bicubic {
        texture,
        address,
        border: s.border_color,
        pixel: s.coordinates == crate::air::abi::Coordinates::Pixel,
        offset,
        raw: false,
    })
}

/// Scalar and vector types and constants the emission uses.
pub(super) struct Tys {
    pub(super) f32_: TypeId,
    i32_: TypeId,
    pub(super) u32_: TypeId,
    pub(super) bool_: TypeId,
    pub(super) vec4: TypeId,
}

impl Tys {
    pub(super) fn new(fb: &mut FunctionBuilder<'_>) -> BResult<Tys> {
        let m = fb.module();
        let f32_ = m.ty(SType::Float { width: 32 })?;
        let i32_ = m.ty(SType::Int {
            width: 32,
            signed: true,
        })?;
        let u32_ = m.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let bool_ = m.ty(SType::Bool)?;
        let vec4 = m.ty(SType::Vector { elem: f32_, len: 4 })?;
        Ok(Tys {
            f32_,
            i32_,
            u32_,
            bool_,
            vec4,
        })
    }
}

pub(super) fn scalar(fb: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<ConstId> {
    fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })
}

pub(super) fn value(fb: &mut FunctionBuilder<'_>, c: ConstId) -> BResult<SValue> {
    fb.module().const_value(c)
}

fn f32_value(fb: &mut FunctionBuilder<'_>, t: &Tys, x: f32) -> BResult<SValue> {
    let c = scalar(fb, t.f32_, u64::from(x.to_bits()))?;
    value(fb, c)
}

fn i32_value(fb: &mut FunctionBuilder<'_>, t: &Tys, x: i32) -> BResult<SValue> {
    let c = scalar(fb, t.i32_, u64::from(x as u32))?;
    value(fb, c)
}

/// Whether bit `bit` of the specialization `spec` is set: `0 < (spec & bit)`, computed in the
/// function, since a function reads a texel format key's value as data.
pub(super) fn flag(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    spec: ConstId,
    bit: u32,
) -> BResult<SValue> {
    let v = value(fb, spec)?;
    Bits::Value(v).test(fb, t, bit)
}

/// A texture slot's format bits as an emulation reads them: the slot's specialization constant, or,
/// in the shared emulation of a switch-at-use (SEMANTICS 1012), the bits of the candidate its
/// handle names, which the switch's arms yield.
#[derive(Clone, Copy)]
pub(super) enum Bits {
    Spec(ConstId),
    Value(SValue),
}

impl Bits {
    /// The bits as a `u32` value.
    pub(super) fn value(self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        match self {
            Bits::Spec(c) => value(fb, c),
            Bits::Value(v) => Ok(v),
        }
    }

    /// Whether any bit of `bit` is set: a specialization constant's [`flag`], or the test of the
    /// value in the function.
    pub(super) fn test(self, fb: &mut FunctionBuilder<'_>, t: &Tys, bit: u32) -> BResult<SValue> {
        match self {
            Bits::Spec(c) => flag(fb, t, c, bit),
            Bits::Value(v) => {
                let mask = scalar(fb, t.u32_, u64::from(bit))?;
                let mask = value(fb, mask)?;
                let zero = scalar(fb, t.u32_, 0)?;
                let zero = value(fb, zero)?;
                let and = fb.binary(BinOp::BitwiseAnd, v, mask)?;
                fb.binary(BinOp::ULessThan, zero, and)
            }
        }
    }
}

/// One plane axis of the footprint: the base tap's index (offset applied), the fraction's
/// 1/256 steps `k`, and the level's extent, as `int`, `uint`, and `int`.
struct Axis {
    base: SValue,
    k: SValue,
    extent: SValue,
}

/// The footprint of axis coordinate `c` on an extent of `n` texels (`u32`): Metal's 1/256
/// fraction rounded half up, `q = floor(256 x - 127.5)`, exact for every finite `x` whose `q`
/// fits an `i32` (`-2^23 + 1/2 <= x <= 2^23`).
///
/// - `y = 256 x` is exact (a power of two), and so is `floor(y)`. `r = y - floor(y)` is exact
///   too, except for `y` in `(-1/2, 0)`. There `r` rounds into `[1/2, 1]` while its real value
///   lies in `(1/2, 1)`; rounding is monotonic and `1/2` is an `f32`, so `r >= 1/2` holds either
///   way. Then `q = floor(y) - 128 + (r >= 1/2)` is computed in integers after the conversion.
/// - `floor(y)` is clamped into `±(2^31 - 128)`, both ends `f32` values, before `OpConvertFToS`.
///   Inside the range above, `floor(y)` lies within the clamp except at `x = 2^23`, where
///   `floor(y) = 2^31`: a separate `floor(y) > 2^31 - 128` test adds back the 128 the clamp took
///   (there `y` is a whole number, so `r = 0` and nothing else is added). No integer step
///   overflows: `q` stays in `[-2^31, 2^31 - 128]`.
/// - Past the range, a coordinate whose `256 x` is finite (`|x| < 2^120`) saturates `q` to
///   `-2^31` or `2^31 - 128`: the translation's choice, and Metal's value there is not measured.
/// - A non-finite `x`, or one whose `256 x` overflows, gives an unspecified footprint. Vulkan
///   lets a device assume no float operand or result is an infinity or a NaN unless
///   `SignedZeroInfNanPreserve` is requested, which this lowering does not request. `NClamp`,
///   the comparisons, and `OpConvertFToS` of an unrepresentable value then give undefined
///   values. Whatever `i32` `q` holds, each tap is clamped (`SClamp`) or wrapped (`SMod` by the
///   positive extent) into the level, and integer adds wrap, so every fetch stays in bounds.
/// - A normalized coordinate's `x` is Metal's position on an axis addressed by `address`
///   ([`super::position`]): `c` wrapped into one period first where the axis repeats or mirrors,
///   then times the extent, rounded to `f32` with ties away from zero. A repeating axis's `x` lies
///   in `[0, n]` and a mirrored one's in `[0, 2n]`, so only a clamping axis reaches the bounds
///   above. A runtime sampler's axis takes the position under the mode its word states, and the
///   coordinate itself where the word states pixel coordinates.
///
/// A constant texel `offset` moves the base tap as Metal places it ([`placed_offset`]), on a
/// level of size `n` whose base level's size is `base` (`None`: `n` is the base level's), by the
/// axis's `texture::OffsetBase` test `parent`; a sampling routine's offset is its call's value,
/// placed by the same rule, so a 0 adds nothing.
fn axis(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    (c, n, base): (SValue, SValue, Option<SValue>),
    (address, from): (AxisAddress, TapBase),
    (offset, parent): (Option<TapOffset>, Option<SValue>),
) -> BResult<Axis> {
    let x = match address {
        AxisAddress::Mode { pixel: true, .. } => c,
        AxisAddress::Mode { address, .. } => super::position::position(fb, c, n, address)?.0,
        AxisAddress::Word { mode, pixel } => {
            let x = super::position::position_by_mode(fb, c, n, mode)?.0;
            fb.select(pixel, c, x)?
        }
    };
    let offset = match offset {
        None | Some(TapOffset::Const(0)) => None,
        Some(o) => Some(placed_offset(fb, t, o, (address, n, base), parent)?),
    };
    let (base, k) = match from {
        TapBase::Quantized => quantize(fb, t, x, offset)?,
        TapBase::Floor => {
            let zero = scalar(fb, t.u32_, 0)?;
            (floor_index(fb, t, x, offset)?, value(fb, zero)?)
        }
    };
    let extent = fb.convert(ConvOp::Bitcast, n, t.i32_)?;
    Ok(Axis { base, k, extent })
}

/// The texel offset `o` an axis's base tap moves by (an `int`), as Metal places it, on an axis
/// addressed by `address` of level size `n` (a `u32`) and base-level size `base` (`None`: `n`):
/// `o` itself, added before the taps are addressed, but 0 on a repeating or mirrored axis whose
/// sizes drop it (`position::offset_dropped`; the linear and bicubic filters, gathers and nearest
/// samples alike), a runtime sampler's axis by the mode its word states, under which a
/// `MirrorClampToEdge` axis's sizes drop it too, its pixel coordinates' as its normalized ones'; by
/// the parent's base-level size where `parent` holds.
fn placed_offset(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    o: TapOffset,
    (address, n, base): (AxisAddress, SValue, Option<SValue>),
    parent: Option<SValue>,
) -> BResult<SValue> {
    let v = match o {
        TapOffset::Const(o) => i32_value(fb, t, o)?,
        TapOffset::Value(v) => v,
    };
    let wraps = match address {
        AxisAddress::Mode {
            address: TapAddress::Repeat | TapAddress::Mirror,
            ..
        } => None,
        AxisAddress::Mode { .. } => return Ok(v),
        AxisAddress::Word { mode, .. } => {
            let repeat = scalar(fb, t.u32_, u64::from(SamplerAddressMode::Repeat.as_raw()))?;
            let repeat = value(fb, repeat)?;
            let mirror = scalar(fb, t.u32_, MIRROR_REPEAT)?;
            let mirror = value(fb, mirror)?;
            let mirror_clamp = scalar(fb, t.u32_, MIRROR_CLAMP_TO_EDGE)?;
            let mirror_clamp = value(fb, mirror_clamp)?;
            let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
            let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
            let clamps_once = fb.binary(BinOp::IEqual, mode, mirror_clamp)?;
            let wraps = fb.binary(BinOp::LogicalOr, repeats, mirrors)?;
            Some(fb.binary(BinOp::LogicalOr, wraps, clamps_once)?)
        }
    };
    let dropped = super::position::offset_dropped(fb, n, base, parent)?;
    let dropped = match wraps {
        Some(w) => fb.binary(BinOp::LogicalAnd, w, dropped)?,
        None => dropped,
    };
    let zero = i32_value(fb, t, 0)?;
    fb.select(dropped, zero, v)
}

/// A nearest filter's texel `floor(x)` plus `offset` (an `int` value) of texel-space position
/// `x` (an `f32`), its floor clamped into `±(2^31 - 128)` before the conversion as [`quantize`]'s
/// is (a position past it, or a non-finite one, reads an unspecified texel within the level).
fn floor_index(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    x: SValue,
    offset: Option<SValue>,
) -> BResult<SValue> {
    let fl = fb.glsl(GlslOp::Floor, &[x])?;
    let lo = f32_value(fb, t, -2_147_483_520.0)?;
    let hi = f32_value(fb, t, 2_147_483_520.0)?;
    let fl = fb.glsl(GlslOp::NClamp, &[fl, lo, hi])?;
    let mut i = fb.convert(ConvOp::ConvertFToS, fl, t.i32_)?;
    if let Some(o) = offset {
        i = fb.binary(BinOp::IAdd, i, o)?;
    }
    Ok(i)
}

/// The base tap `i = q >> 8` plus `offset` (an `int` value) and the fraction's 1/256 steps
/// `k = q & 255` (a `uint`) of texel-space position `x` (an `f32`), `q = floor(256 x - 127.5)`
/// ([`axis`]).
pub(super) fn quantize(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    x: SValue,
    offset: Option<SValue>,
) -> BResult<(SValue, SValue)> {
    let k256 = f32_value(fb, t, 256.0)?;
    let y = fb.binary_no_contraction(BinOp::FMul, x, k256)?;
    let fl = fb.glsl(GlslOp::Floor, &[y])?;
    let r = fb.binary_no_contraction(BinOp::FSub, y, fl)?;
    let half = f32_value(fb, t, 0.5)?;
    let up = fb.binary(BinOp::FOrdGreaterThanEqual, r, half)?;
    // `2^31 - 128`, the largest `f32` below `2^31`.
    let lo = f32_value(fb, t, -2_147_483_520.0)?;
    let hi = f32_value(fb, t, 2_147_483_520.0)?;
    let over = fb.binary(BinOp::FOrdGreaterThan, fl, hi)?;
    let fl = fb.glsl(GlslOp::NClamp, &[fl, lo, hi])?;
    let fl = fb.convert(ConvOp::ConvertFToS, fl, t.i32_)?;
    let one = i32_value(fb, t, 1)?;
    let zero = i32_value(fb, t, 0)?;
    let inc = fb.select(up, one, zero)?;
    // Past the clamp's top `r` is 0 (or NaN), so `inc` is 0 and the clamped 128 is the whole
    // adjustment.
    let k128 = i32_value(fb, t, 128)?;
    let inc = fb.select(over, k128, inc)?;
    let m128 = i32_value(fb, t, -128)?;
    let q = fb.binary(BinOp::IAdd, fl, m128)?;
    let q = fb.binary(BinOp::IAdd, q, inc)?;
    let eight = i32_value(fb, t, 8)?;
    let mut base = fb.binary(BinOp::ShiftRightArithmetic, q, eight)?;
    if let Some(o) = offset {
        base = fb.binary(BinOp::IAdd, base, o)?;
    }
    let mask = i32_value(fb, t, 255)?;
    let k = fb.binary(BinOp::BitwiseAnd, q, mask)?;
    let k = fb.convert(ConvOp::Bitcast, k, t.u32_)?;
    Ok((base, k))
}

/// The index of tap `i + d` on `axis` under `address`, as `uint` within the level, and for a
/// border axis (a runtime sampler's axis included, whose mode may be one) whether the tap lies
/// inside it.
fn tap(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    axis: &Axis,
    d: i32,
    address: AxisAddress,
) -> BResult<(SValue, Option<SValue>)> {
    let i = if d == 0 {
        axis.base
    } else {
        let d = i32_value(fb, t, d)?;
        fb.binary(BinOp::IAdd, axis.base, d)?
    };
    let zero = i32_value(fb, t, 0)?;
    let one = i32_value(fb, t, 1)?;
    let last = fb.binary(BinOp::ISub, axis.extent, one)?;
    let inside = |fb: &mut FunctionBuilder<'_>| {
        let lo = fb.binary(BinOp::SGreaterThanEqual, i, zero)?;
        let hi = fb.binary(BinOp::SLessThan, i, axis.extent)?;
        fb.binary(BinOp::LogicalAnd, lo, hi)
    };
    let (at, inside) = match address {
        AxisAddress::Mode {
            address: TapAddress::Edge,
            ..
        } => (fb.glsl(GlslOp::SClamp, &[i, zero, last])?, None),
        // The extent is positive, so `SMod` lands in `[0, extent)`.
        AxisAddress::Mode {
            address: TapAddress::Repeat,
            ..
        } => (fb.binary(BinOp::SMod, i, axis.extent)?, None),
        AxisAddress::Mode {
            address: TapAddress::Border,
            ..
        } => (
            fb.glsl(GlslOp::SClamp, &[i, zero, last])?,
            Some(inside(fb)?),
        ),
        AxisAddress::Mode {
            address: TapAddress::Mirror,
            ..
        } => {
            let m = reflected_once(fb, t, i)?;
            (mirrored(fb, t, m, axis.extent)?, None)
        }
        // Each mode's index, selected by the word's mode; a border mode's tap outside the level
        // reads the border. A `MirrorClampToEdge` axis (a runtime sampler's alone) reflects the
        // index once, `-1 - i` for a negative `i`, then clamps it into the level, as Metal
        // addresses each tap of a pixel or normalized sampler.
        AxisAddress::Word { mode, .. } => {
            let edge = fb.glsl(GlslOp::SClamp, &[i, zero, last])?;
            let wrapped = fb.binary(BinOp::SMod, i, axis.extent)?;
            let m = reflected_once(fb, t, i)?;
            let reflected = mirrored(fb, t, m, axis.extent)?;
            let once = fb.glsl(GlslOp::SMin, &[m, last])?;
            let repeat = scalar(fb, t.u32_, u64::from(SamplerAddressMode::Repeat.as_raw()))?;
            let repeat = value(fb, repeat)?;
            let mirror = scalar(fb, t.u32_, MIRROR_REPEAT)?;
            let mirror = value(fb, mirror)?;
            let mirror_clamp = scalar(fb, t.u32_, MIRROR_CLAMP_TO_EDGE)?;
            let mirror_clamp = value(fb, mirror_clamp)?;
            let border = scalar(fb, t.u32_, CLAMP_TO_ZERO)?;
            let border = value(fb, border)?;
            let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
            let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
            let clamps_once = fb.binary(BinOp::IEqual, mode, mirror_clamp)?;
            let at = fb.select(mirrors, reflected, edge)?;
            let at = fb.select(clamps_once, once, at)?;
            let at = fb.select(repeats, wrapped, at)?;
            let within = inside(fb)?;
            let unbordered = fb.binary(BinOp::ULessThan, mode, border)?;
            (at, Some(fb.binary(BinOp::LogicalOr, within, unbordered)?))
        }
    };
    Ok((fb.convert(ConvOp::Bitcast, at, t.u32_)?, inside))
}

/// `MTLSamplerAddressModeMirrorRepeat`'s value.
pub(super) const MIRROR_REPEAT: u64 = SamplerAddressMode::MirrorRepeat.as_raw() as u64;
/// `MTLSamplerAddressModeMirrorClampToEdge`'s value.
const MIRROR_CLAMP_TO_EDGE: u64 = SamplerAddressMode::MirrorClampToEdge.as_raw() as u64;
/// `MTLSamplerAddressModeClampToZero`'s value, the least of the two border modes
/// (`ClampToBorderColor` is the next).
const CLAMP_TO_ZERO: u64 = SamplerAddressMode::ClampToZero.as_raw() as u64;

/// Index `i` (an `int`) reflected once about the level's start: `i`, or `-1 - i` (`!i`, never
/// overflowing) for a negative `i`, so never negative.
fn reflected_once(fb: &mut FunctionBuilder<'_>, t: &Tys, i: SValue) -> BResult<SValue> {
    let zero = i32_value(fb, t, 0)?;
    let neg = fb.binary(BinOp::SLessThan, i, zero)?;
    let flipped = fb.unary(UnOp::Not, i)?;
    fb.select(neg, flipped, i)
}

/// An index reflected once ([`reflected_once`]), `m` (a non-negative `int`), reflected into a
/// level of `extent` texels (an `int`) as Metal's mirrored repeat reflects it: `r = m mod 2n`, then
/// `r` or `2n - 1 - r`. The extent is positive and below `2^30`, so `2n` is too, and `SMod` of the
/// non-negative `m` lands in `[0, 2n)`.
fn mirrored(fb: &mut FunctionBuilder<'_>, t: &Tys, m: SValue, extent: SValue) -> BResult<SValue> {
    let one = i32_value(fb, t, 1)?;
    let two_n = fb.binary(BinOp::IAdd, extent, extent)?;
    let r = fb.binary(BinOp::SMod, m, two_n)?;
    let low = fb.binary(BinOp::SLessThan, r, extent)?;
    let top = fb.binary(BinOp::ISub, two_n, one)?;
    let back = fb.binary(BinOp::ISub, top, r)?;
    fb.select(low, r, back)
}

/// Metal's texel at the border for color `b` of the bound texture, whose treatment is `bits`:
/// channel `c` is 1 where its [`BicubicTexels::BORDER`] bit for `b` is set, else 0 (the format's
/// channels and the texture's swizzle, which the binder states).
fn border_texel(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    bits: Bits,
    b: BorderColor,
) -> BResult<SValue> {
    let column = border_column(b);
    let one = f32_value(fb, t, 1.0)?;
    let zero = f32_value(fb, t, 0.0)?;
    let mut parts = Vec::with_capacity(4);
    for c in 0..4 {
        let set = bits.test(fb, t, 1 << (BicubicTexels::BORDER + 3 * c + column))?;
        parts.push(fb.select(set, one, zero)?);
    }
    fb.composite_construct(t.vec4, &parts)
}

/// The column of border color `b` among [`BicubicTexels::BORDER`]'s texels:
/// `MTLSamplerBorderColor`'s value, which a runtime sampler's word holds.
fn border_column(b: BorderColor) -> u32 {
    match b {
        BorderColor::TransparentBlack => SamplerBorderColor::TransparentBlack,
        BorderColor::OpaqueBlack => SamplerBorderColor::OpaqueBlack,
        BorderColor::OpaqueWhite => SamplerBorderColor::OpaqueWhite,
    }
    .as_raw()
}

/// The border texel a runtime sampler's taps read outside a border axis, as [`border_texel`] reads
/// a constant color's: that of the color word `word` states, whose value is the color's column,
/// under `ClampToZero` as under `ClampToBorderColor`.
fn word_border_texel(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    bits: Bits,
    word: SValue,
) -> BResult<SValue> {
    let column = word_field(
        fb,
        t,
        word,
        SamplerWord::BORDER_SHIFT,
        SamplerWord::BORDER >> SamplerWord::BORDER_SHIFT,
    )?;
    let all = bits.value(fb)?;
    let low = scalar(fb, t.u32_, 1)?;
    let low = value(fb, low)?;
    let one = f32_value(fb, t, 1.0)?;
    let nil = f32_value(fb, t, 0.0)?;
    let mut parts = Vec::with_capacity(4);
    for c in 0..4 {
        let first = scalar(fb, t.u32_, u64::from(BicubicTexels::BORDER + 3 * c))?;
        let first = value(fb, first)?;
        let shift = fb.binary(BinOp::IAdd, first, column)?;
        let bit = fb.binary(BinOp::ShiftRightLogical, all, shift)?;
        let bit = fb.binary(BinOp::BitwiseAnd, bit, low)?;
        let set = fb.binary(BinOp::IEqual, bit, low)?;
        parts.push(fb.select(set, one, nil)?);
    }
    fb.composite_construct(t.vec4, &parts)
}

/// How a filter emulated by fetches addresses its taps.
#[derive(Clone, Copy)]
pub(super) enum Addressing<'a> {
    /// A constexpr sampler's: per plane axis (`s`, then `t`, then `r`) the addressing of an index
    /// outside the level, the border color the [`TapAddress::Border`] axes read, and whether the
    /// coordinate is in pixels.
    Modes {
        address: &'a [TapAddress],
        border: BorderColor,
        pixel: bool,
    },
    /// A runtime sampler's [`SamplerWord`] (a `u32`) on `axes` plane axes: each axis's address
    /// mode, the border color, and the coordinates, read from it in the function.
    Word { word: SValue, axes: usize },
}

impl Addressing<'_> {
    /// The plane axes.
    pub(super) fn axes(&self) -> usize {
        match self {
            Addressing::Modes { address, .. } => address.len(),
            Addressing::Word { axes, .. } => *axes,
        }
    }
}

/// How one plane axis reads its position and addresses its taps ([`Addressing`]).
#[derive(Clone, Copy)]
enum AxisAddress {
    /// A constexpr sampler's: the axis's addressing, and whether the coordinate is in pixels.
    Mode { address: TapAddress, pixel: bool },
    /// A runtime sampler's: the axis's address mode (a `u32`, `MTLSamplerAddressMode`'s value)
    /// and whether the word states pixel coordinates (a `bool`).
    Word { mode: SValue, pixel: SValue },
}

/// The taps a filter emulated by fetches reads: their [`Addressing`], the constant texel offset
/// Metal applies per plane axis (placed as [`placed_offset`] says), the offsets `d` of the taps
/// `i + d` from the footprint's base texel `i`, how `i` follows from Metal's position, and the
/// level they are fetched from (a `u32`; `None`: the base level).
pub(super) struct Taps<'a> {
    pub(super) addressing: Addressing<'a>,
    pub(super) offset: Option<&'a [i32]>,
    pub(super) span: &'a [i32],
    pub(super) base: TapBase,
    pub(super) level: Option<SValue>,
}

/// How a footprint's base texel `i` follows from Metal's texel-space position `x` on an axis
/// ([`super::position`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TapBase {
    /// The linear and bicubic filters' and a gather's: `q >> 8` of the quantized
    /// `q = floor(256 x + 1/2) - 128` ([`quantize`]).
    Quantized,
    /// A nearest filter's: `floor(x)`.
    Floor,
}

/// One tap on one plane axis: its index within the level (`uint`), and for a border axis whether
/// it lies inside the level.
pub(super) type Tap = (SValue, Option<SValue>);

/// The footprint of a sample emulated by fetches (module docs): each plane axis's fraction steps
/// `k` (`uint`) and taps, and what a fetch of the base level at a tap reads.
pub(super) struct Footprint {
    layer: Option<SValue>,
    coord_ty: TypeId,
    border: Option<SValue>,
    lod: ImageOperands,
    /// Per plane axis, the fraction's 1/256 steps and the taps of [`Taps::span`].
    pub(super) axes: Vec<(SValue, Vec<Tap>)>,
}

impl Footprint {
    /// The texel at `site` (a tap of each plane axis, `s` first) of the base level of `img`, the
    /// layer's, a tap outside a border axis replaced by the border color.
    pub(super) fn fetch(
        &self,
        fb: &mut FunctionBuilder<'_>,
        img: SValue,
        site: &[Tap],
    ) -> BResult<SValue> {
        let (at, inside) = self.site(fb, site)?;
        let texel = self.read(fb, img, at)?;
        self.border(fb, texel, inside)
    }

    /// The image coordinate of `site` (a tap of each plane axis, `s` first), the layer's, and on
    /// a border axis whether each of its taps lies inside the level.
    pub(super) fn site(
        &self,
        fb: &mut FunctionBuilder<'_>,
        site: &[Tap],
    ) -> BResult<(SValue, Option<SValue>)> {
        let mut parts = Vec::with_capacity(site.len() + 1);
        let mut inside: Option<SValue> = None;
        for &(at, tin) in site {
            parts.push(at);
            inside = match (inside, tin) {
                (Some(a), Some(b)) => Some(fb.binary(BinOp::LogicalAnd, a, b)?),
                (a, b) => a.or(b),
            };
        }
        if let Some(l) = self.layer {
            parts.push(l);
        }
        let at = match parts[..] {
            [x] => x,
            [] => return Err(missing_tap()),
            _ => fb.composite_construct(self.coord_ty, &parts)?,
        };
        Ok((at, inside))
    }

    /// The texel of `img`'s base level at image coordinate `at`.
    pub(super) fn read(
        &self,
        fb: &mut FunctionBuilder<'_>,
        img: SValue,
        at: SValue,
    ) -> BResult<SValue> {
        fb.fetch(img, at, &self.lod)
    }

    /// `texel`, or the border color where its tap lies outside a border axis (`inside`).
    pub(super) fn border(
        &self,
        fb: &mut FunctionBuilder<'_>,
        texel: SValue,
        inside: Option<SValue>,
    ) -> BResult<SValue> {
        match (inside, self.border) {
            (Some(c), Some(bt)) => fb.select(c, texel, bt),
            _ => Ok(texel),
        }
    }
}

/// Where an emulation reads its taps: the image of its own binding, or, in the shared emulation of
/// a switch-at-use, each candidate's image in its own arm, or an extended-range
/// texture's raw texels, each decoded (`super::extended`).
#[derive(Clone, Copy)]
pub(super) enum Texels<'a> {
    Own(SValue),
    Candidates(&'a super::Candidates<'a>),
    Extended(&'a super::extended::Texture),
}

impl Texels<'_> {
    /// The texels at `sites` (each a tap of every plane axis, `s` first), each passed through
    /// `post`: from the own image, a fetch per site in turn; from the candidates, one dispatch
    /// whose arms each fetch every site from their image (the null arm zeros), then each site's
    /// border color and `post` in turn; from an extended-range texture, each site's raw texel
    /// decoded to its `float` value, or the border texel, in turn.
    pub(super) fn fetch<S: AsRef<[Tap]>>(
        self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        (t, fp): (&Tys, &Footprint),
        sites: &[S],
        post: &dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
    ) -> BResult<Vec<SValue>> {
        let mut out = Vec::with_capacity(sites.len());
        match self {
            Texels::Own(img) => {
                for site in sites {
                    let v = fp.fetch(fb, img, site.as_ref())?;
                    out.push(post(cx, fb, v)?);
                }
            }
            Texels::Extended(tex) => {
                for site in sites {
                    let v = super::extended::texel(cx, fb, fp, tex, site.as_ref(), false)?;
                    out.push(post(cx, fb, v)?);
                }
            }
            Texels::Candidates(c) => {
                let mut at = Vec::with_capacity(sites.len());
                for site in sites {
                    at.push(fp.site(fb, site.as_ref())?);
                }
                let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, s: &Sampling| {
                    let img = c.image(cx, fb, s)?;
                    let mut v = Vec::with_capacity(at.len());
                    for (a, _) in &at {
                        v.push(fp.read(fb, img, *a)?);
                    }
                    Ok(v)
                };
                let vec4 = t.vec4;
                let n = at.len();
                let null = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
                    let z = fb.module().constant(Constant::Null { ty: vec4 })?;
                    Ok(vec![value(fb, z)?; n])
                };
                let texels = c.each(cx, fb, &read, &null)?;
                if texels.len() != n {
                    return Err(cx.fail("a shared emulation's taps without their texels"));
                }
                for (v, (_, inside)) in texels.into_iter().zip(at) {
                    let v = fp.border(fb, v, inside)?;
                    out.push(post(cx, fb, v)?);
                }
            }
        }
        Ok(out)
    }
}

/// The plain data a footprint reads of its sample: what a sampling routine's call passes, or the
/// open function's own values of the sample.
#[derive(Clone, Copy)]
pub(super) struct At {
    /// The plane coordinate as the sample states it (`lanes` floats).
    pub(super) coord: SValue,
    /// The array layer (an integer).
    pub(super) layer: Option<SValue>,
    /// The slot's `SpecKey::OffsetBaseAxes` tests, by which an offset is placed, where the access
    /// reads them (`position::offset_dropped`).
    pub(super) parent: Option<OffsetBase>,
    /// A sampling routine's texel offset, its call's value (an `int` or `ivecN` of `N` plane axes),
    /// which the routine places in place of its taps' constant one (`None`: the taps' own).
    pub(super) offset: Option<(SValue, u8)>,
}

/// The texel offset one plane axis's base tap moves by, before [`placed_offset`] places it: a
/// constant, or a sampling routine's call's lane (an `int`).
#[derive(Clone, Copy)]
enum TapOffset {
    Const(i32),
    Value(SValue),
}

impl At {
    /// The plain data of sample `s` in the open block: its coordinate's and layer's values, and
    /// its slot's offset tests where it reads them.
    pub(super) fn of(
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        s: &Sampling,
    ) -> BResult<At> {
        let coord = super::emit::plane_coord(cx, fb, s)?;
        let layer = match s.layer {
            Some(l) => Some(cx.operand(fb, l)?),
            None => None,
        };
        let parent = super::emit::offset_base(cx, s)?;
        Ok(At {
            coord,
            layer,
            parent,
            offset: None,
        })
    }
}

/// The footprint over `taps` of a sample at plane coordinate `coord` (and array layer `layer`)
/// of image `img` (image facts `image`) at the taps' level, whose taps are fetched from `img`, the
/// border color seen through the channels of the format's treatment `bits`: an extended-range
/// texture's emulated sample (`super::extended`), which reads its slot's own image, its offset
/// (its taps' constant one, or a sampling routine's call's value `offset`, as [`At::offset`])
/// placed by the slot's `SpecKey::OffsetBaseAxes` tests `parent` where it reads them (a level's
/// taps read its size, and the offset rule the base level's too).
pub(super) fn footprint_of_image(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    (coord, layer, offset): (SValue, Option<SValue>, Option<(SValue, u8)>),
    (img, image): (SValue, CombinedImage),
    bits: Bits,
    (taps, parent): (&Taps<'_>, Option<OffsetBase>),
) -> BResult<Footprint> {
    let at = At {
        coord,
        layer,
        parent,
        offset,
    };
    let zero_u = fb.u32_const(0)?;
    let zero_u = value(fb, zero_u)?;
    let size = fb.query(ImageQuery::SizeLod(taps.level.unwrap_or(zero_u)), img)?;
    let offset = at.offset.is_some() || taps.offset.is_some_and(|o| o.iter().any(|o| *o != 0));
    let base = match (taps.level, offset) {
        (Some(_), true) => Some(fb.query(ImageQuery::SizeLod(zero_u), img)?),
        _ => None,
    };
    footprint_at(fb, t, &at, (size, base, image), bits, taps)
}

/// [`footprint_of_image`] of an image whose base-level size `size` (a `uvec2`) the caller states:
/// an in-place texture's (`super::in_place`), whose taps read the base level of its one level.
pub(super) fn footprint_sized(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    (coord, layer, offset): (SValue, Option<SValue>, Option<(SValue, u8)>),
    (size, image): (SValue, CombinedImage),
    bits: Bits,
    (taps, parent): (&Taps<'_>, Option<OffsetBase>),
) -> BResult<Footprint> {
    let at = At {
        coord,
        layer,
        parent,
        offset,
    };
    footprint_at(fb, t, &at, (size, None, image), bits, taps)
}

/// The footprint of sample `s` over `taps` on an image of level-0 size `size` (image facts
/// `image`), the border color seen through the channels of the bound format's treatment `bits`.
pub(super) fn footprint_on(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    (size, image): (SValue, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Footprint> {
    let at = At::of(cx, fb, s)?;
    footprint_at(fb, t, &at, (size, None, image), bits, taps)
}

/// The footprint over `taps` of a sample whose plain data is `at` (its plane coordinate, array
/// layer, and the slot's offset tests) on a level of size `size` of an image whose level-0 size is
/// `base` (`None`: `size` is the level-0 size; image facts `image`), the border color seen through
/// the channels of the bound format's treatment `bits`.
fn footprint_at(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At,
    (size, base, image): (SValue, Option<SValue>, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Footprint> {
    let zero_u = fb.u32_const(0)?;
    let zero_u = value(fb, zero_u)?;
    let size_lanes = image.plane + u8::from(image.arrayed);
    let (coord, parent) = (at.coord, at.parent);
    let axes = taps.addressing.axes();
    // A runtime sampler's coordinates and each plane axis's address mode, from its word.
    let (pixel, modes) = match taps.addressing {
        Addressing::Modes { .. } => (None, Vec::new()),
        Addressing::Word { word, .. } => {
            let mut modes = Vec::with_capacity(axes);
            for shift in SamplerWord::ADDRESS_SHIFT.into_iter().take(axes) {
                modes.push(word_field(fb, t, word, shift, 0b111)?);
            }
            let bit = scalar(fb, t.u32_, u64::from(SamplerWord::PIXEL_COORDINATES))?;
            let bit = value(fb, bit)?;
            let p = fb.binary(BinOp::BitwiseAnd, word, bit)?;
            (Some(fb.binary(BinOp::INotEqual, p, zero_u)?), modes)
        }
    };
    // Each plane axis's footprint (`s`, then `t`), with its addressing.
    let mut ax = Vec::with_capacity(axes);
    for a in 0..axes {
        let lane = a as u32;
        let c = if axes == 1 {
            coord
        } else {
            fb.composite_extract(coord, &[lane])?
        };
        let n = if size_lanes == 1 {
            size
        } else {
            fb.composite_extract(size, &[lane])?
        };
        let nb = match base {
            Some(b) if size_lanes == 1 => Some(b),
            Some(b) => Some(fb.composite_extract(b, &[lane])?),
            None => None,
        };
        let address = match (taps.addressing, pixel) {
            (Addressing::Modes { address, pixel, .. }, _) => match address.get(a) {
                Some(&address) => AxisAddress::Mode { address, pixel },
                None => return Err(missing_tap()),
            },
            (Addressing::Word { .. }, Some(pixel)) => match modes.get(a) {
                Some(&mode) => AxisAddress::Word { mode, pixel },
                None => return Err(missing_tap()),
            },
            (Addressing::Word { .. }, None) => return Err(missing_tap()),
        };
        let off = match at.offset {
            Some((v, 1)) if a == 0 => Some(TapOffset::Value(v)),
            Some((v, lanes)) if a < usize::from(lanes) => {
                Some(TapOffset::Value(fb.composite_extract(v, &[lane])?))
            }
            Some(_) => None,
            None => taps
                .offset
                .and_then(|o| o.get(a).copied())
                .map(TapOffset::Const),
        };
        let read = (address, taps.base);
        let off = (off, OffsetBase::axis(parent, a as u32));
        ax.push((axis(fb, t, (c, n, nb), read, off)?, address));
    }
    // The layer, clamped into the array as sampling clamps it.
    let layer = match at.layer {
        Some(l) => {
            let layers = fb.composite_extract(size, &[axes as u32])?;
            let one = fb.u32_const(1)?;
            let one = value(fb, one)?;
            let last = fb.binary(BinOp::ISub, layers, one)?;
            Some(fb.glsl(GlslOp::UMin, &[l, last])?)
        }
        None => None,
    };
    let coord_len = axes + usize::from(layer.is_some());
    let coord_ty = if coord_len == 1 {
        t.u32_
    } else {
        fb.module().ty(SType::Vector {
            elem: t.u32_,
            len: coord_len as u8,
        })?
    };
    let border = match taps.addressing {
        Addressing::Modes {
            address, border, ..
        } if address.contains(&TapAddress::Border) => Some(border_texel(fb, t, bits, border)?),
        Addressing::Modes { .. } => None,
        Addressing::Word { word, .. } => Some(word_border_texel(fb, t, bits, word)?),
    };
    let lod = ImageOperands {
        lod: Some(taps.level.unwrap_or(zero_u)),
        ..ImageOperands::default()
    };
    // The taps `i + d` of each axis, with their in-range conditions.
    let mut out = Vec::with_capacity(axes);
    for (axis, address) in &ax {
        let mut row = Vec::with_capacity(taps.span.len());
        for d in taps.span {
            row.push(tap(fb, t, axis, *d, *address)?);
        }
        out.push((axis.k, row));
    }
    Ok(Footprint {
        layer,
        coord_ty,
        border,
        lod,
        axes: out,
    })
}

/// The field of `word` (a `u32`) at bit `shift`, `mask` wide once shifted down.
pub(super) fn word_field(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    word: SValue,
    shift: u32,
    mask: u32,
) -> BResult<SValue> {
    let sh = scalar(fb, t.u32_, u64::from(shift))?;
    let sh = value(fb, sh)?;
    let m = scalar(fb, t.u32_, u64::from(mask))?;
    let m = value(fb, m)?;
    let f = fb.binary(BinOp::ShiftRightLogical, word, sh)?;
    fb.binary(BinOp::BitwiseAnd, f, m)
}

/// The `vec4` texel the bicubic sample `s` of sampled image `si` (image facts `image`) reads
/// (module docs), its `half` result the device's narrowing of the `f32` sum.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    si: SValue,
    image: CombinedImage,
    b: &Bicubic,
) -> BResult<SValue> {
    let Some(&spec) = cx.decl.families.sampler.bicubic.get(&b.texture) else {
        return Err(cx.fail(format!(
            "texture slot {} has no bicubic specialization",
            b.texture
        )));
    };
    let half = half_result(cx, s)?;
    let t = Tys::new(fb)?;
    let span = b.span();
    let taps = b.taps(span);
    let img = fb.image(si)?;
    let zero_u = fb.u32_const(0)?;
    let zero_u = value(fb, zero_u)?;
    let size = fb.query(ImageQuery::SizeLod(zero_u), img)?;
    let bits = Bits::Spec(spec);
    let fp = footprint_on(cx, fb, &t, s, (size, image), bits, &taps)?;
    let x = filter(cx, fb, (&t, &fp, size), b, bits, Texels::Own(img))?;
    narrowed(fb, x, half)
}

/// The `vec4` texel of the bicubic sample `s` shared by the arms of a switch-at-use (SEMANTICS
/// 1012): the footprint on the named candidate's level-0 size `size` (image facts `image`), its
/// format's bits `bits`, and its taps read by `c`'s arms.
pub(super) fn emit_shared(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (size, image): (SValue, CombinedImage),
    b: &Bicubic,
    bits: SValue,
    c: &super::Candidates<'_>,
) -> BResult<SValue> {
    let half = half_result(cx, s)?;
    let t = Tys::new(fb)?;
    let span = b.span();
    let taps = b.taps(span);
    let bits = Bits::Value(bits);
    let fp = footprint_on(cx, fb, &t, s, (size, image), bits, &taps)?;
    let x = filter(cx, fb, (&t, &fp, size), b, bits, Texels::Candidates(c))?;
    narrowed(fb, x, half)
}

/// Whether the bicubic sample `s` has a `half` result.
fn half_result(cx: &mut EmitCx<'_>, s: &Sampling) -> BResult<bool> {
    match s.texel.map(|f| f.conv) {
        Some(TexelConv::F32) => Ok(false),
        Some(TexelConv::F16) => Ok(true),
        _ => Err(cx.fail("a bicubic sample without a float texel form")),
    }
}

/// Texel `x` (a `vec4` of `f32`) as a sample's result holds it: the device's narrowing to `half`
/// where `half` (`arith::narrow::round`), else as it is.
pub(super) fn narrowed(fb: &mut FunctionBuilder<'_>, x: SValue, half: bool) -> BResult<SValue> {
    use crate::lower::arith::legal::{FloatFmt, Form, Held};
    if !half {
        return Ok(x);
    }
    let form = Form {
        held: Held::Float(FloatFmt::Half),
        lanes: 4,
    };
    crate::lower::arith::narrow::round(fb, form, x)
}

impl Bicubic {
    /// The taps `i + d` of each axis: `i .. i + 1` for a 1D texture's bilinear filter,
    /// `i - 1 .. i + 2` otherwise.
    pub(super) fn span(&self) -> &'static [i32] {
        if self.linear_only() {
            &[0, 1]
        } else {
            &[-1, 0, 1, 2]
        }
    }

    /// The taps the filter reads over `span`.
    pub(super) fn taps<'a>(&'a self, span: &'a [i32]) -> Taps<'a> {
        Taps {
            addressing: Addressing::Modes {
                address: &self.address,
                border: self.border,
                pixel: self.pixel,
            },
            offset: self.offset.as_deref(),
            span,
            base: TapBase::Quantized,
            level: None,
        }
    }
}

/// The filter's `f32` texel over footprint `fp` (types `t`) on a base level of `size` (the `uint`
/// lanes `ImageQuerySizeLod` gives), under the bound format's treatment `bits`, its taps read from
/// `texels` (module docs): the routines' sum in the device's arithmetic ([`routines`]).
pub(super) fn filter(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (t, fp, size): (&Tys, &Footprint, SValue),
    b: &Bicubic,
    bits: Bits,
    texels: Texels<'_>,
) -> BResult<SValue> {
    let mut axes = fp.axes.iter();
    let Some((kx, xtaps)) = axes.next() else {
        return Err(cx.fail("a bicubic sample without a plane axis"));
    };
    let kx = *kx;
    let vertical = axes.next().map(|(k, row)| (*k, row));
    // A raw sample's taps are the device's decode of their codes.
    let post = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: SValue| {
        if b.raw {
            crate::lower::texture::srgb_raw_tap(fb, v)
        } else {
            Ok(v)
        }
    };
    let flags = bits.value(fb)?;
    let call = |fb: &mut FunctionBuilder<'_>, f, args: &[SValue]| -> BResult<SValue> {
        fb.call(f, args)?.ok_or_else(missing_routine)
    };
    let Some((ky, ytaps)) = vertical else {
        // A 1D texture: bilinear over taps `i` and `i + 1`, whatever the format.
        let Some(line) = cx.decl.families.sampler.bicubic_line else {
            return Err(cx.fail("a 1D bicubic sample without its routines"));
        };
        let sites: Vec<[Tap; 1]> = xtaps.iter().map(|x| [*x]).collect();
        let mut args = texels.fetch(cx, fb, (t, fp), &sites, &post)?;
        args.extend([kx, flags]);
        return call(fb, line.sum, &args);
    };
    let Some(plane) = cx.decl.families.sampler.bicubic_plane else {
        return Err(cx.fail("a 2D bicubic sample without its routines"));
    };
    let (&[x0, x1, x2, x3], &[y0, y1, y2, y3]) = (xtaps.as_slice(), ytaps.as_slice()) else {
        return Err(cx.fail("a 2D bicubic sample's footprint is not four taps square"));
    };
    // A level below four texels on either plane axis takes Metal's linear filter, and so does a
    // 32-bit float format (module docs).
    let w = fb.composite_extract(size, &[0])?;
    let h = fb.composite_extract(size, &[1])?;
    let least = fb.glsl(GlslOp::UMin, &[w, h])?;
    let four = scalar(fb, t.u32_, 4)?;
    let four = value(fb, four)?;
    let small = fb.binary(BinOp::ULessThan, least, four)?;
    let is_linear = bits.test(fb, t, BicubicTexels::LINEAR)?;
    let bilinear = fb.binary(BinOp::LogicalOr, small, is_linear)?;
    // A mirrored axis reads texel 1 at fraction 0 where Metal takes its bicubic sum (a level of
    // at least four texels each way, a format not filtered bilinearly), else the linear filter's
    // mirrored taps (module docs).
    let pinned = if b.address.contains(&TapAddress::Mirror) {
        Some(fb.unary(UnOp::LogicalNot, bilinear)?)
    } else {
        None
    };
    let pin = |fb: &mut FunctionBuilder<'_>, axis: usize, k: SValue, taps: [Tap; 4]| {
        let (Some(p), Some(TapAddress::Mirror)) = (pinned, b.address.get(axis)) else {
            return Ok((k, taps));
        };
        let zero = scalar(fb, t.u32_, 0)?;
        let zero = value(fb, zero)?;
        let k = fb.select(p, zero, k)?;
        let mut out = taps;
        for (index, tap) in (0u64..).zip(out.iter_mut()) {
            let at = scalar(fb, t.u32_, index)?;
            let at = value(fb, at)?;
            tap.0 = fb.select(p, at, tap.0)?;
        }
        BResult::Ok((k, out))
    };
    let (kx, [x0, x1, x2, x3]) = pin(fb, 0, kx, [x0, x1, x2, x3])?;
    let (ky, [y0, y1, y2, y3]) = pin(fb, 1, ky, [y0, y1, y2, y3])?;
    // The footprint row by row (the `t` axis), each by column (the `s` axis).
    let mut sites = Vec::with_capacity(16);
    for y in [y0, y1, y2, y3] {
        for x in [x0, x1, x2, x3] {
            sites.push([x, y]);
        }
    }
    let mut args = texels.fetch(cx, fb, (t, fp), &sites, &post)?;
    // The bilinear filter's fractions: a small level's merged ones (an axis whose two inner taps
    // read one texel takes fraction 0), else the sample's own.
    let mkx = super::linear::merged(fb, t, kx, [x1, x2])?;
    let mky = super::linear::merged(fb, t, ky, [y1, y2])?;
    let lkx = fb.select(small, mkx, kx)?;
    let lky = fb.select(small, mky, ky)?;
    args.extend([kx, ky, lkx, lky, bilinear, flags]);
    call(fb, plane.sum, &args)
}

/// A routine call without a result: every routine returns its texel.
pub(super) fn missing_routine() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::FunctionCall,
        rule: OperandRule::Callee,
    }
}

/// A fetch at a site without a tap: every site has one per plane axis.
pub(super) fn missing_tap() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ImageFetch,
        rule: OperandRule::Arity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One axis clamping to zero beside one clamping to a colored border reads the sampler's one
    /// border color on both, as Metal does. AIR cannot spell the state (an axis coded 0 takes the
    /// sampler's one border color), so it is built here from a decoded bicubic sampler.
    #[test]
    fn zero_beside_a_colored_border_reads_the_border_color() {
        // Bicubic min and mag, no mip filter, normalized, clamp to zero on every axis.
        let zero = SamplerState::from_words([0x007b_ff00_0008_1400, 0]).expect("decodes");
        assert_eq!(zero.address_s, AddressMode::ClampToZero);
        assert!(check(&zero, ImageDim::D2, (false, false, false), true).is_ok());
        for border in [BorderColor::OpaqueBlack, BorderColor::OpaqueWhite] {
            let colored = SamplerState {
                address_t: AddressMode::ClampToBorder,
                border_color: border,
                ..zero
            };
            assert!(check(&colored, ImageDim::D2, (false, false, false), true).is_ok());
            let planned = plan(&colored, 0, ImageDim::D2, None).expect("planned");
            assert_eq!(planned.border, border);
            assert_eq!(planned.address, vec![TapAddress::Border; 2]);
        }
    }
}
