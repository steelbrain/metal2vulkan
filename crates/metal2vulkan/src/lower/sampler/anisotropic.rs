//! Metal's anisotropic filter, emulated by the device's samples at explicit levels.
//!
//! **The rule** (an Apple M4 on macOS 27.0.1, seven probe rounds, our own
//! synthetic textures and gradients; the M3 Max on macOS 26.7 answered every compared footprint
//! identically). For a sample of a 2D plane with gradients `gx`, `gy` in level-0
//! texels through a sampler of `maxAnisotropy` rounded up to a power of two, `a = 2^la`
//! (`metal::sampler::vk_sampler`):
//!
//! - each gradient's length is `max(M + 3m/16, 49M/64 + 21m/32)` of its larger and smaller
//!   component magnitudes `M`, `m`; the major axis is the longer, and the taps step along it
//!   divided by that length (not its Euclidean length);
//! - Metal takes logarithms by a four-segment piecewise-linear `log2` of its own; the emulation
//!   takes the device's `Log2` (and `Exp2` for the powers): the determinant `|gx.x gy.y - gx.y
//!   gy.x|` of the gradients gives the minor axis's;
//! - the level is `max(log2(det) - max(log2(major), 0), log2(major) - la)`: the minor axis held
//!   at a texel of the major or more, and the major over `a` (Metal rounds it to 256ths);
//! - the tap count is the first of 1, 2, 3, 4, 6, 8, 10, 12, 14 and 16 whose `ceil(64 log2 n) /
//!   64` the log ratio `log2(major) - max(log2(det) - log2(major), 0)` does not pass, at most `a`
//!   (11,090 of 11,096 footprints under Metal's logarithm);
//! - tap `j` of `n` weighs `floor(256 / n)` 256ths, the remainder one each to the middle taps,
//!   and sits `(j - (n - 1) / 2) s0` texels from the centre along the major axis, with the
//!   spacing `s0 = major · g · S`: `r'` the log ratio's power (at most `a`), `g = (r' - 1) / 2`
//!   for two taps and `(1 - κ_n (1 - r'/n)) / n` past two (`κ_n` fitted per count), `S` the power
//!   `2^-λ` of the level past the texture's levels (the level past 0 under no mip filter); Metal
//!   rounds `r'` to `1/256`, `g · S` to `1/4096` and `S` to a 6-bit mantissa of sixty-fourths;
//! - each tap reads the level by the sampler's mip filter: level 0 without one, the level
//!   `L + 1` past half a level under a nearest one, and under a linear one the fractional level,
//!   whose two levels the device's own linear mip filter blends (Metal blends them by `256 - q`
//!   and `q` 256ths).
//!
//! **Exactness** (the emulation's residue against the M4's answers): the length, the tap count's
//! thresholds, the weights and the direction are Metal's; the logarithms, the powers, the level's
//! fraction and the levels' blend are the device's (`PrecisionNote::AnisotropicFilter`), so the
//! level, a tap count at a threshold and the spacing may differ from Metal's by their rounding.
//! The spacing past two taps below the cap is an approximation (`κ_n`). Each tap is the device's
//! own filter at the level handed to it, so its footprint and sum are the device's. The lane
//! `runtime_lane::anisotropic_filter` measures it on `rgba8Unorm` at `maxAnisotropy` 2 to 16.
//!
//! **The form.** Each tap is one device sample at an explicit level: under a linear mip filter the
//! fractional level, which the device's linear mip filter blends: the device's blend, which need
//! not be Metal's (an RTX 3090's is 1.03 codes off without anisotropy). A level of
//! 0 where Metal minifies (its level, before the texture's levels hold it, above 0, so on a
//! texture of one level too) is handed as `2^-12`, which keeps the device's minification filter
//! and blends nothing a device's mip precision keeps. A nearest mip filter's level is handed
//! whole: the 3090's anisotropic sampler blends the next level at a fractional explicit level
//! whatever its mipmap mode. Under a nearest mip
//! filter or none, level 0 where Metal minifies is handed as `0.25` where the device's sampler of
//! the binding is isotropic (`metal::sampler::device_anisotropy`), where Vulkan reads level 0 by
//! the min filter at it on every device, and which a device that rounds `2^-12` to 0 (the 780M's
//! RADV) still minifies at; as `2^-12` where it is anisotropic, whose level and taps Vulkan leaves
//! to the implementation (an RTX 5080's read past level 0 at `0.25`).
//!
//! **Cubes** ([`cube_face`]). Metal under macOS 27 samples a cube with explicit gradients as it
//! samples the 2D plane of the direction's face with Metal's face gradients, each gradient without
//! its normal component (the Apple M3 Max, every inner lane of each probed sampler equal to a 2D
//! array sample of the face; `emit::cube_footprint`'s `Face` form). The emulation reads those, in
//! the face's texels, as the plane's gradients, and moves the direction across the face by each
//! tap's step, which the device projects onto the face again (a tap past the face's edge onto the
//! next face, as the device projects it). Through an isotropic constexpr sampler the module takes
//! Metal's level alone, its major axis's (an isotropic cube's measured so: a footprint past
//! `1 + 3/2048` minifies, `unk_cube_grad_table`'s turning points), and one tap, where no exact
//! path of the sample's own answers it (a fetched coordinate): its levels are Metal's, each the
//! device's filter.
//!
//! **The loop**. Tap 0 is written out, and taps 1 to `n - 1` are one loop's body, which asks the
//! driver not to unroll it (`LoopControl` `DontUnroll`): each tap's arithmetic and the sum's order
//! are those of the taps written out one by one, so the answer is the same (the lane's statistics,
//! every sampler alike), and a driver compiles one tap's code instead of sixteen. A guest
//! fragment's variant of 135 emulated samples took 1.49M words instead of 1.79M, and its pipeline
//! 29.0 s to make on an RTX 3090 instead of 38.9 s.
//!
//! **The routine** (`routine::Filter`). The emulation (the level-0 size and level count, the
//! degree and mip filter, the footprint and the taps' sum) is one routine per shape: the binding,
//! the sampler's degree, mip filter and device anisotropy, a cube, an array, a `min_lod_clamp` and
//! a texel offset. Every sample of the shape calls it with what its site computes (the
//! coordinate, the gradients, a cube's face steps, the layer and the clamp), and converts the sum
//! as its texel. Where the profile admits the `Offset` operand on a sample (`maintenance8` and
//! `shaderImageGatherExtended`), the call passes the texel offset, constant or dynamic, and the
//! taps' samples take it as that operand, which adds the same offset to the texel coordinates at
//! each level as `ConstOffset` does (the specification's texel coordinate offset), so the samples
//! at every offset call one routine; elsewhere a constant offset is the shape's `ConstOffset`. Each
//! answer is the one the code written at the site gave, and the module holds one copy of the code
//! a driver compiles, where a sample costs a call
//! (`anisotropic_samples_call_one_routine`). Written at each site, the emulation was about 2,100
//! words a sample, each compiled again: a guest fragment's variant of an anisotropic runtime
//! sampler under the exact forms took 1,113,991 words where the plain function took 572,061, and
//! 33.6 to 40.7 s to make on the RTX 5080 Laptop GPU; as the routine, 659,590 words, made in 20.9
//! to 26.1 s on a loaded host whose plain function took 16.7 to 22.3 s, its calls kept calls there
//! (`DriverWorkaround::DontInlineKeepsCalls`). A sample at a
//! dynamic texel offset on a profile without the `Offset` operand on a sample, whose arms the
//! taps' samples read, keeps the code at its site.
//!
//! **Reach** (`select`): a sample without comparison of a 2D texture or 2D array of `float` or
//! `half` texels, at the fragment's implicit level (with a bias, through Metal's biased minor
//! axis) or with explicit gradients, or of a cube or cube array with explicit gradients, through a
//! constexpr sampler whose `maxAnisotropy` is above 1 (a cube's isotropic one too) or a runtime
//! sampler slot whose word states anisotropy, on a device without `samplerAnisotropy`: every
//! Vulkan sampler is isotropic there (`metal::sampler::device_anisotropy`), so the emulation goes
//! where that feature's refusal was, the device's own sampling unchanged where the feature is. A
//! runtime slot's degree is its `SpecKey::SamplerAnisotropy` specialization; where its word states no
//! anisotropy, the sample is the plain one. Elsewhere (a cube's implicit level, a 3D texture, a
//! comparison, an observed residency, an argument buffer's sampler) the device filters, recorded
//! as `PrecisionNote::DeviceAnisotropicFilter` where the sampler's anisotropy is known (on Metal's
//! own sampler, Metal's filter, unrecorded); without `samplerAnisotropy` isotropically, and an
//! argument buffer's anisotropic sampler, which the function cannot record, is refused there
//! (`metal::sampler::check_argument_sampler`).
//!
//! **The variant**. A runtime slot's emulation, at every sample it reaches, made a guest's
//! fragment function of 337 such samples twice its size (2.35M words from 1.18M, the emulation
//! written at each site), whatever samplers its draws bind, and every making of its pipeline hands
//! the driver the whole module. Such a sample states
//! `RuntimeFact::AnisotropicSampler` for its slot and takes the emulation only under the variant
//! naming the slot (`LoweringInputs::anisotropic_samplers`), which the binder asks for wherever
//! the bound sampler's word states anisotropy; elsewhere it is the plain sample, which is
//! Metal's for every word the function is then bound with. Under the variant the sample is the
//! form above, its word's test included, so a slot named or not, each word's answer is the one
//! before.

use crate::air::abi::{Coordinates, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::{NoteOn, NoteTest, SelectCx};
use crate::lower::texture::ImageRef;
use crate::metal::sampler::{device_anisotropy_by_filters, SamplerUse, SamplerWord};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec, Merged};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, Contraction, ConvOp, DerivOp, GlslOp, ImageOperands, ImageQuery,
    MemAccess, Type as SType, UnOp,
};

use super::routine::{Filter, Shape};
use super::{BindingKey, Lod, Offset, SampleOp, Sampling, Source};

/// A sample that takes Metal's anisotropic filter, emulated (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Anisotropic {
    /// The sampler's degree.
    pub degree: Degree,
    /// The sampler's mip filter.
    pub mip: Mip,
    /// Whether the device's sampler of the sample's binding filters anisotropically, which
    /// decides the level handed for level 0 where Metal minifies ([`read_levels`]).
    pub device: DeviceAnisotropy,
    /// Whether the profile admits the `Offset` operand on a sample (`maintenance8` and
    /// `shaderImageGatherExtended`), so the routine's call passes the sample's texel offset
    /// (`routine::Filter::passed_offset`).
    pub offset_operand: bool,
}

/// Whether the device's sampler of an emulated sample's binding filters anisotropically
/// (`anisotropyEnable`), as `metal::sampler::vk_sampler` makes it
/// (`metal::sampler::device_anisotropy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceAnisotropy {
    /// A constexpr sampler's: its Vulkan sampler's own flag.
    Const(bool),
    /// A runtime sampler slot's, by the filters its word states: `(linear, nearest)`, the flag
    /// where both filters are linear and where one is nearest
    /// (`metal::sampler::device_anisotropy_by_filters`). A word with a cubic filter, whose other
    /// filter it does not state, is taken as anisotropic.
    Filters(bool, bool),
}

/// The degree a sampler filters with, as `log2` of its `maxAnisotropy` rounded up to a power of
/// two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Degree {
    /// A constexpr sampler's: 1 to 4.
    Const(u32),
    /// Runtime sampler slot `n`'s: its `SpecKey::SamplerAnisotropy` specialization (0 for an
    /// isotropic sampler, whose sample is the plain one).
    Slot(u32),
}

/// The mip filter a sample reads its levels by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mip {
    /// A constexpr sampler's.
    Const(MipFilter),
    /// A runtime sampler slot's word's `SamplerWord::MIP_FILTER` field.
    Word,
}

/// The most taps Metal's filter takes (at `maxAnisotropy` 16); a degree past it (a runtime
/// sampler's 32) takes 16, its tap counts past 16 not being measured.
pub const MOST_TAPS: u32 = 16;

/// `log2` of [`MOST_TAPS`].
const MOST_LOG: i32 = 4;

/// The tap counts, and the log ratio (in sixty-fourths of a level) each holds up to:
/// `ceil(64 log2 n)`.
const COUNTS: [(i32, i32); 10] = [
    (1, 0),
    (2, 64),
    (3, 102),
    (4, 128),
    (6, 166),
    (8, 192),
    (10, 213),
    (12, 230),
    (14, 244),
    (16, 256),
];

/// The spacing's correction past two taps, `κ_n`, fitted per tap count to the rounds 5 and 6 (an
/// approximation below the ratio cap, exact at it).
const KAPPA: [(i32, f32); 8] = [
    (3, 0.0600),
    (4, 0.0348),
    (6, 0.0255),
    (8, 0.0159),
    (10, 0.0210),
    (12, 0.0392),
    (14, 0.0153),
    (16, 0.0178),
];

/// The level handed for level 0 where Metal's level is above 0 under a linear mip filter, and under
/// a nearest one or none where the device's sampler filters anisotropically: the device minifies
/// there, and blends nothing of level 1 at any mip precision up to 11 bits. Vulkan leaves an
/// anisotropic sampler's level and taps to the implementation ("Texel Anisotropic Filtering": "The
/// particular scheme for anisotropic texture filtering is implementation-dependent"), and an RTX
/// 5080's reads past level 0 at a fractional level whatever its mipmap mode (the lane
/// `runtime_lane::anisotropic_filter` at `MINIFIED_NEAREST`: a linear filter under a nearest mip
/// filter at `maxAnisotropy` 8 off by up to 147 codes).
const MINIFIED_ZERO: f32 = 1.0 / 4096.0;

/// The level handed for level 0 where Metal's level is above 0 under a nearest mip filter or
/// none where the device's sampler is isotropic (`anisotropyEnable` false), where Vulkan defines
/// the level and the filter ("Image Level(s) Selection": under `VK_SAMPLER_MIPMAP_MODE_NEAREST`
/// the level is `nearest(d')`, `ceil(d' + 0.5) - 1` or `floor(d' + 0.5)`, with `d' = level_base +
/// clamp(λ, 0, q)`, level 0 for `λ` up to `0.5`; "Texel Filtering": "If λ is less than or equal to
/// zero, the texture is said to be magnified, and the filter mode within a mip level is selected
/// by the magFilter in the sampler. If λ is greater than zero, the texture is said to be minified,
/// and the filter mode within a mip level is selected by the minFilter in the sampler."). It is a
/// multiple of `2^-4`, the coarsest `mipLodPrecisionBits` Vulkan allows. A sampler without
/// mipmapping holds its levels at `metal::sampler::NOT_MIPMAPPED_MAX_LOD`, this level. Handed
/// `MINIFIED_ZERO` there, the 780M's RADV magnified: every lane of `unk_cube_grad_table`'s probes
/// that Metal minifies read the nearest texel of a sampler that minifies linearly and magnifies
/// nearest.
const MINIFIED_NEAREST: f32 = 0.25;

/// Whether constexpr sampler `words` used as `use_` is one that Metal's own sampler on `profile`
/// (`DriverDeviation::MetalSampler`) filters anisotropically other than as Metal's `NotMipmapped`:
/// handed anisotropy, without a mip filter, its two filters different on the device. Vulkan has no
/// sampler without mipmapping, and no level clamp the device's sampler takes reads Metal's
/// `NotMipmapped` footprint at `maxAnisotropy` 2 to 16 where the filters differ
/// (`metal::sampler::vk_sampler`).
fn own_not_mipmapped(
    words: [u64; 2],
    use_: SamplerUse,
    profile: &crate::device::DeviceProfile,
) -> Result<bool, Error> {
    if !profile.takes_driver_deviation(crate::error::DriverDeviation::MetalSampler) {
        return Ok(false);
    }
    let state = super::constexpr::state_of(words)?;
    let device = super::constexpr::constexpr_desc(words, use_, profile)?;
    Ok(state.mip_filter == MipFilter::None
        && device.max_anisotropy.is_some()
        && device.min_filter != device.mag_filter)
}

/// Whether sample `op` of a texture of `shape` (`image`) through static sampler `sampler` (`None`
/// for an argument buffer's) takes Metal's anisotropic filter emulated (the module's reach), with
/// its note recorded: `AnisotropicFilter` where it does, `DeviceAnisotropicFilter` where the
/// sampler's anisotropy is known and the sample, filtered at an implicit level or with
/// gradients, is out of reach; a runtime sampler slot's under its word's anisotropy alone. A
/// device whose own sampler answers the sample, anisotropy included, takes neither: one that
/// samples natively or through Metal's own sampler, with `samplerAnisotropy`.
/// `integer`: the texel class is an integer one (never filtered); `observed`: the function
/// reads the sample's residency; `other_class`: the slot's variant binds a texture of another
/// class (`LoweringInputs::cross_class`); `swizzled`: the slot swizzles its converted texels
/// (`LoweringInputs::swizzled_storage`). `use_` is the sample's binding's use of its sampler,
/// which with the sampler decides the device's sampler.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    (sampler, image, shape): (Option<SamplerRef>, ImageRef, ImageShape),
    (integer, observed, other_class, swizzled): (bool, bool, bool, bool),
    use_: SamplerUse,
    m: &mut Meter,
) -> Result<Option<Anisotropic>, Error> {
    // Without `samplerAnisotropy` every Vulkan sampler is isotropic
    // (`metal::sampler::device_anisotropy`): the emulation goes where it reaches an anisotropic
    // sampler's sample there on every device, and the device's isotropic filter is recorded
    // elsewhere.
    let anisotropy = cx.profile.fields().images.sampler_anisotropy;
    let SampleOp::Sample { lod, dref, .. } = op else {
        return Ok(None);
    };
    let derived = matches!(
        lod,
        Lod::Implicit { .. } | Lod::Grad(..) | Lod::Hoisted | Lod::CubeGrad(..)
    );
    if integer || !derived {
        return Ok(None);
    }
    let cube = matches!(lod, Lod::CubeGrad(..));
    if anisotropy {
        cx.families.sampler.device_sampling = true;
        if let Some(SamplerRef::Constexpr(words)) = sampler {
            if own_not_mipmapped(words, use_, cx.profile)? {
                cx.precision
                    .insert(PrecisionNote::NotMipmappedAnisotropicFilter);
            }
        }
        return Ok(None);
    }
    let (degree, mip, device) = match sampler {
        Some(SamplerRef::Constexpr(words)) => {
            let state = super::constexpr::state_of(words)?;
            if state.max_anisotropy <= 1 || state.coordinates == Coordinates::Pixel {
                return Ok(None);
            }
            let la = state.max_anisotropy.next_power_of_two().trailing_zeros();
            let device = super::constexpr::constexpr_desc(words, use_, cx.profile)?
                .max_anisotropy
                .is_some();
            (
                Degree::Const(la),
                Mip::Const(state.mip_filter),
                DeviceAnisotropy::Const(device),
            )
        }
        Some(SamplerRef::Slot(n)) => {
            let (linear, nearest) = device_anisotropy_by_filters(use_, cx.profile);
            (
                Degree::Slot(n),
                Mip::Word,
                DeviceAnisotropy::Filters(linear, nearest),
            )
        }
        None => return Ok(None),
    };
    let reached = dref.is_none()
        && (shape.dim == ImageDim::D2 || cube)
        && !shape.ms
        && !shape.depth
        && !observed
        && !other_class
        && !swizzled
        && matches!(image, ImageRef::Slot(_));
    // Out of reach on Metal's own sampler, the device's filter is Metal's, at the footprint the
    // cube's form hands it (`PrecisionNote::CubeGradientFootprint`).
    if anisotropy && !reached {
        return Ok(None);
    }
    let note = match reached {
        true => PrecisionNote::AnisotropicFilter,
        false => PrecisionNote::DeviceAnisotropicFilter,
    };
    match degree {
        Degree::Const(_) => {
            cx.precision.insert(note);
        }
        Degree::Slot(n) => {
            // A reached sample states its slot's fact, and takes the emulation only under the
            // variant naming the slot, which the binder asks for wherever the bound word states
            // anisotropy. Elsewhere it is the plain sample, with no emulation in the module:
            // Metal's sample wherever the word states no anisotropy, the only words a function
            // compiled without the slot is bound with there.
            if reached {
                let slot = super::select::sampler_slot(cx.stage, n);
                cx.facts
                    .insert(crate::pipeline::facts::RuntimeFact::AnisotropicSampler { slot });
                if !cx.inputs.anisotropic_samplers.contains(&slot) {
                    return Ok(None);
                }
            }
            if let Some(s) = cx.families.sampler.slots.get_mut(&n) {
                match reached {
                    true => s.emulated = true,
                    false => s.anisotropy = true,
                }
            }
            let anisotropic = NoteTest::differ(NoteOn::Sampler(n), SamplerWord::ANISOTROPIC, 0);
            cx.note_when(note, vec![[anisotropic].into()], m)?;
        }
    }
    let images = &cx.profile.fields().images;
    Ok(reached.then_some(Anisotropic {
        degree,
        mip,
        device,
        offset_operand: images.dynamic_sample_offset && images.gather_extended,
    }))
}

/// Sample `s` through Metal's anisotropic filter (the module docs), in its result's form: for a
/// runtime sampler slot, where its word states anisotropy, else the plain sample. `grads` are a
/// switch-at-use's hoisted gradients (`Lod::Hoisted`).
pub(super) fn sample(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    a: &Anisotropic,
    grads: Option<(SValue, SValue)>,
) -> BResult<SValue> {
    let SampleOp::Sample {
        lod,
        dref: None,
        min_lod,
        offset,
    } = &s.op
    else {
        return Err(cx.fail("an anisotropic emulation of another access than a plain sample"));
    };
    let Source::Combined(key) = s.source else {
        return Err(cx.fail("an anisotropic emulation of a heap element"));
    };
    let Some(&(_, image)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    let coord = super::emit::plane_coord(cx, fb, s)?;
    // A cube's direction with its ties broken as Metal breaks them, whose face Metal reads.
    let coord = match s.cube {
        true => super::face::metal_ties(fb, coord)?,
        false => coord,
    };
    // The gradients, in the sample's own position.
    let (dx, dy) = match lod {
        Lod::Grad(dx, dy) | Lod::CubeGrad(dx, dy, _) => {
            (cx.operand(fb, *dx)?, cx.operand(fb, *dy)?)
        }
        Lod::Hoisted => match grads {
            Some(g) => g,
            None => return Err(cx.fail("a hoisted anisotropic sample without its gradients")),
        },
        Lod::Implicit { bias: None, .. } => (
            fb.derivative(DerivOp::DPdx, coord)?,
            fb.derivative(DerivOp::DPdy, coord)?,
        ),
        // Metal's bias shifts the footprint's minor axis alone, in the level-0 size's texels.
        Lod::Implicit { bias: Some(b), .. } => {
            let b = cx.operand(fb, *b)?;
            let b = super::emit::metal_bias(fb, b)?;
            let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
            let img = fb.image(si)?;
            let zero = u32c(fb, 0)?;
            let size = fb.query(ImageQuery::SizeLod(zero), img)?;
            let size_lanes = image.plane + u8::from(image.arrayed);
            super::emit::plane_axis_gradients(fb, coord, b, (size, size_lanes))?
        }
        _ => return Err(cx.fail("an anisotropic emulation of a sample at an explicit level")),
    };
    let min_lod = match min_lod {
        Some(m) => Some(cx.operand(fb, *m)?),
        None => None,
    };
    let layer = match s.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    // A cube's face, and Metal's gradients on it.
    let (face, on_face) = match lod {
        Lod::CubeGrad(..) => {
            let (face, on_face) = cube_face(fb, coord, (dx, dy))?;
            (Some(face), on_face)
        }
        _ => (None, (dx, dy)),
    };
    let site = Site {
        coord,
        grads: on_face,
        face,
        layer,
        min_lod,
    };
    let a = *a;
    match a.degree {
        Degree::Const(_) => {
            let x = filter(cx, fb, s, (key, a), &site, offset)?;
            super::emit::texel_value(
                cx,
                fb,
                s,
                image,
                (x, crate::lower::texture::half::TexelOrigin::Computed),
                None,
            )
        }
        Degree::Slot(n) => {
            // The plain sample where the bound sampler's word states no anisotropy.
            let word = super::emit::sampler_word(cx, fb, super::Pixel::Spec(n))?;
            let anisotropic = super::emit::word_states_any(fb, word, SamplerWord::ANISOTROPIC)?;
            let plain = Sampling {
                anisotropic: None,
                ..s.clone()
            };
            let emulation: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = filter(cx, fb, s, (key, a), &site, offset)?;
                let x = super::emit::texel_value(
                    cx,
                    fb,
                    s,
                    image,
                    (x, crate::lower::texture::half::TexelOrigin::Computed),
                    None,
                )?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let device: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = super::emit::sampled_value(cx, fb, &plain, grads)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, anisotropic, emulation, Some(device))?;
            match merged[..] {
                [x] => Ok(x),
                _ => Err(cx.fail("an anisotropic sample without its merged result")),
            }
        }
    }
}

/// What a sample's emulation reads that its site computes: the coordinate and its gradients, a
/// cube's face, the array layer and the `min_lod_clamp`.
#[derive(Clone, Copy)]
struct Site {
    /// The plane coordinate (normalized); a cube's direction, its ties broken as Metal breaks
    /// them.
    coord: SValue,
    /// The plane coordinate's gradients (normalized); a cube's, Metal's gradients on its face.
    grads: (SValue, SValue),
    /// A cube's face, across which its taps step.
    face: Option<Face>,
    /// The array layer (an integer), for an array.
    layer: Option<SValue>,
    /// The sample's `min_lod_clamp`.
    min_lod: Option<SValue>,
}

/// The emulation's sum of the taps of sample `s` through binding `key` (the device's texels, before
/// the texel's conversion), as filter `a` takes it: its shape's routine's call
/// (`routine::Filter`), or its code in the open block where the texel offset is dynamic and takes
/// arms, which the site's samples read.
fn filter(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (key, a): (BindingKey, Anisotropic),
    site: &Site,
    offset: &Option<Offset>,
) -> BResult<SValue> {
    // Where the profile admits the `Offset` operand on a sample, the call passes the offset, a
    // constant one too, and the taps' samples read it; else a constant offset is the shape's
    // `ConstOffset`, and a dynamic one's arms are the site's.
    let (offset, passed) = match offset {
        Some(Offset::Dynamic(_, Some(_))) => {
            let mut ops = ImageOperands::default();
            super::emit::offset_operands(cx, fb, offset, s.lanes, &mut ops)?;
            return emulate(cx, fb, (s.source, s.lanes), a, site, ops);
        }
        Some(Offset::Dynamic(_, None)) => {
            let mut ops = ImageOperands::default();
            super::emit::offset_operands(cx, fb, offset, s.lanes, &mut ops)?;
            match ops.offset {
                Some(v) => (None, Some(v)),
                None => return Err(cx.fail("a dynamic texel offset without its operand")),
            }
        }
        Some(Offset::Const(k)) if a.offset_operand => {
            (None, Some(super::emit::const_offset(fb, k)?))
        }
        Some(Offset::Const(k)) => (Some(k.clone()), None),
        None => (None, None),
    };
    let mut args = vec![site.coord, site.grads.0, site.grads.1];
    if let Some(face) = site.face {
        args.extend([face.s, face.t]);
    }
    args.extend(site.layer);
    args.extend(site.min_lod);
    args.extend(passed);
    let mut tys = Vec::with_capacity(args.len());
    for &v in &args {
        tys.push(fb.value_ty(v)?);
    }
    let shape = Shape::Anisotropic(Filter {
        binding: key,
        filter: a,
        lanes: s.lanes,
        cube: site.face.is_some(),
        layer: site.layer.is_some(),
        min_lod: site.min_lod.is_some(),
        offset,
        passed_offset: passed.is_some(),
    });
    let f = match cx.sampling.get(&shape, &tys) {
        Some(f) => f,
        None => {
            let Shape::Anisotropic(r) = &shape else {
                return Err(cx.fail("an anisotropic routine of another shape"));
            };
            let mut rcx = cx.for_new_function();
            let made = define(&mut rcx, fb.module(), r, &tys);
            cx.adopt(rcx);
            let f = made?;
            cx.sampling.insert(shape, tys, f);
            f
        }
    };
    match fb.call(f, &args)? {
        Some(x) => Ok(x),
        None => Err(cx.fail("an anisotropic routine without its result")),
    }
}

/// Define the routine of emulation `r` (`routine::Filter`) in `mb`, its parameters of types `tys`
/// in [`filter`]'s order: the sum the emulation's code gave at the site, read from `r`'s binding
/// and its sampler's specializations.
fn define(
    cx: &mut EmitCx<'_>,
    mb: &mut ModuleBuilder,
    r: &Filter,
    tys: &[TypeId],
) -> BResult<FuncId> {
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let ret = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let mut fb = mb.sampling_routine(ret)?;
    let mut params = Vec::with_capacity(tys.len());
    for &ty in tys {
        params.push(fb.param(ty)?);
    }
    let want = 3
        + 2 * usize::from(r.cube)
        + usize::from(r.layer)
        + usize::from(r.min_lod)
        + usize::from(r.passed_offset);
    if params.len() != want {
        return Err(cx.fail("an anisotropic routine without its parameters"));
    }
    let mut params = params.into_iter();
    let (Some(coord), Some(dx), Some(dy)) = (params.next(), params.next(), params.next()) else {
        return Err(cx.fail("an anisotropic routine without its coordinate"));
    };
    let face = match r.cube {
        true => match (params.next(), params.next()) {
            (Some(s), Some(t)) => Some(Face { s, t }),
            _ => return Err(cx.fail("an anisotropic routine without its face")),
        },
        false => None,
    };
    let layer = r.layer.then(|| params.next()).flatten();
    let min_lod = r.min_lod.then(|| params.next()).flatten();
    let passed = r.passed_offset.then(|| params.next()).flatten();
    let site = Site {
        coord,
        grads: (dx, dy),
        face,
        layer,
        min_lod,
    };
    let mut ops = ImageOperands::default();
    if let Some(k) = &r.offset {
        ops.const_offset = Some(super::emit::const_offset(&mut fb, k)?);
    }
    ops.offset = passed;
    let x = emulate(
        cx,
        &mut fb,
        (Source::Combined(r.binding), r.lanes),
        r.filter,
        &site,
        ops,
    )?;
    fb.ret(Some(x))?;
    fb.finish()
}

/// The emulation's sum in the open block for the sample of `source` (its coordinate of `lanes`
/// plane lanes) at `site` with offset operands `ops`: the level-0 size and the level count, the
/// sampler's degree and mip filter, then [`filtered`].
fn emulate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (source, lanes): (Source, u8),
    a: Anisotropic,
    site: &Site,
    ops: ImageOperands,
) -> BResult<SValue> {
    let (si, _, _) = super::emit::sampled_image(cx, fb, source)?;
    let img = fb.image(si)?;
    let zero = u32c(fb, 0)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    let levels = fb.query(ImageQuery::Levels, img)?;
    let at = At {
        coord: site.coord,
        grads: site.grads,
        size,
        levels,
        min_lod: site.min_lod,
        ops,
        layer: site.layer,
        face: site.face,
    };
    let (la, mip, minified) = match (a.degree, a.mip, a.device) {
        (Degree::Const(la), Mip::Const(mip), DeviceAnisotropy::Const(device)) => {
            let la = i32c(fb, la.min(MOST_LOG as u32) as i32)?;
            let mip = u32c(fb, mip_code(mip))?;
            let minified = f32c(fb, minified_level(device))?;
            (la, mip, minified)
        }
        (Degree::Slot(n), Mip::Word, DeviceAnisotropy::Filters(linear, nearest)) => {
            let Some(&spec) = cx.decl.families.sampler.degrees.get(&n) else {
                return Err(cx.fail(format!("sampler slot {n} has no anisotropy degree")));
            };
            let word = super::emit::sampler_word(cx, fb, super::Pixel::Spec(n))?;
            let field = u32c(fb, SamplerWord::MIP_FILTER_SHIFT)?;
            let mip = fb.binary(BinOp::ShiftRightLogical, word, field)?;
            let three = u32c(fb, 3)?;
            let mip = fb.binary(BinOp::BitwiseAnd, mip, three)?;
            let i = int(fb)?;
            let la = fb.convert(ConvOp::Bitcast, spec, i)?;
            let most = i32c(fb, MOST_LOG)?;
            let la = fb.glsl(GlslOp::SMin, &[la, most])?;
            let minified = word_minified_level(fb, word, (linear, nearest))?;
            (la, mip, minified)
        }
        _ => return Err(cx.fail("an anisotropic emulation of a mixed degree and mip filter")),
    };
    let most = match a.degree {
        Degree::Const(la) => 1 << la.min(MOST_LOG as u32),
        Degree::Slot(_) => MOST_TAPS,
    };
    filtered(cx, fb, (source, lanes), &at, (la, mip, minified), most)
}

/// What a sample's emulation reads, taken in the sample's own position.
#[derive(Clone, Copy)]
struct At {
    /// The plane coordinate (normalized); a cube's direction, its ties broken as Metal breaks
    /// them.
    coord: SValue,
    /// The plane coordinate's gradients (normalized); a cube's, Metal's gradients on its face.
    grads: (SValue, SValue),
    /// The level-0 size (`u32` components, the layer count last for an array).
    size: SValue,
    /// The level count (`u32`).
    levels: SValue,
    /// The sample's `min_lod_clamp`.
    min_lod: Option<SValue>,
    /// The sample's offset operands.
    ops: ImageOperands,
    /// The array layer (an integer), for an array.
    layer: Option<SValue>,
    /// A cube's face, across which its taps step.
    face: Option<Face>,
}

/// A cube sample's face: the steps of its direction that move the face's normalized `s` and `t`
/// by one, `2 |m_a| σ_s e_s` and `2 |m_a| σ_t e_t` (`vec3`s), in Vulkan's face table ("Cube
/// Map Face Selection"), whose faces and their axes are Metal's.
#[derive(Clone, Copy)]
pub(super) struct Face {
    pub(super) s: SValue,
    pub(super) t: SValue,
}

/// The face of cube direction `p` (its ties broken as Metal breaks them, so the major axis is
/// `emit::Major`'s), and Metal's gradients on it (`vec2`s, normalized) of the direction's `dx` and
/// `dy`: each without its normal component, `σ_s (g_s, g_t) / (2 |m_a|)` for `dx` and `σ_t (g_s,
/// g_t) / (2 |m_a|)` for `dy`, the face's two tangents `s` and `t` and their signs `σ` as Vulkan's
/// face table states them. That is Vulkan's face gradients of the two, `dx`'s `t` and `dy`'s `s`
/// mirrored where `σ_s σ_t` is `-1` (the `-x`, `-y` and `+z` faces): Metal under macOS 27 samples
/// a cube as it samples the face's 2D plane with these gradients (the Apple M3 Max, every inner
/// lane of each probed sampler equal, `emit::cube_footprint`'s `Face` form).
pub(super) fn cube_face(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    (dx, dy): (SValue, SValue),
) -> BResult<(Face, (SValue, SValue))> {
    let f = float(fb)?;
    let v2 = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let v3 = fb.module().ty(SType::Vector { elem: f, len: 3 })?;
    let m = super::emit::Major::of(fb, p)?;
    let [px, py, pz] = super::emit::components(fb, p)?;
    let ma = fb.select(m.y, py, px)?;
    let ma = fb.select(m.z, pz, ma)?;
    let zf = f32c(fb, 0.0)?;
    let negative = fb.binary(BinOp::FOrdLessThan, ma, zf)?;
    let onef = f32c(fb, 1.0)?;
    let minus = f32c(fb, -1.0)?;
    // `σ_s`: `-sign(m_a)` on an `x` face, 1 on a `y` face, `sign(m_a)` on a `z` face. `σ_t`:
    // `sign(m_a)` on a `y` face, -1 on the others.
    let on_x = fb.select(negative, onef, minus)?;
    let on_z = fb.select(negative, minus, onef)?;
    let sigma_s = fb.select(m.z, on_z, on_x)?;
    let sigma_s = fb.select(m.y, onef, sigma_s)?;
    let on_y = fb.select(negative, minus, onef)?;
    let sigma_t = fb.select(m.y, on_y, minus)?;
    let mag = fb.glsl(GlslOp::FAbs, &[ma])?;
    let two = f32c(fb, 2.0)?;
    let span = fb.binary(BinOp::FMul, mag, two)?;
    // The steps: `s` lies on `z` on an `x` face and on `x` on the others; `t` on `z` on a `y`
    // face and on `y` on the others.
    let s_len = fb.binary(BinOp::FMul, sigma_s, span)?;
    let t_len = fb.binary(BinOp::FMul, sigma_t, span)?;
    let s_on_x = fb.select(m.y_or_z, s_len, zf)?;
    let s_on_z = fb.select(m.y_or_z, zf, s_len)?;
    let s = fb.composite_construct(v3, &[s_on_x, zf, s_on_z])?;
    let t_on_y = fb.select(m.y, zf, t_len)?;
    let t_on_z = fb.select(m.y, t_len, zf)?;
    let t = fb.composite_construct(v3, &[zf, t_on_y, t_on_z])?;
    // Metal's gradients on the face.
    let gradient = |fb: &mut FunctionBuilder<'_>, g: SValue, sigma: SValue| -> BResult<SValue> {
        let [gx, gy, gz] = super::emit::components(fb, g)?;
        let gs = fb.select(m.y_or_z, gx, gz)?;
        let gt = fb.select(m.y, gz, gy)?;
        let gs = fb.binary(BinOp::FMul, gs, sigma)?;
        let gt = fb.binary(BinOp::FMul, gt, sigma)?;
        let gs = fb.binary(BinOp::FDiv, gs, span)?;
        let gt = fb.binary(BinOp::FDiv, gt, span)?;
        fb.composite_construct(v2, &[gs, gt])
    };
    let gx = gradient(fb, dx, sigma_s)?;
    let gy = gradient(fb, dy, sigma_t)?;
    Ok((Face { s, t }, (gx, gy)))
}

/// Metal's mip filter code in the sampler word's field: 0 none, 1 nearest, 2 linear.
fn mip_code(mip: MipFilter) -> u32 {
    match mip {
        MipFilter::None => 0,
        MipFilter::Nearest => 1,
        MipFilter::Linear => 2,
    }
}

/// The level handed for level 0 where Metal minifies under a nearest mip filter or none, where the
/// device's sampler filters anisotropically (`device`) or not: [`MINIFIED_ZERO`] or
/// [`MINIFIED_NEAREST`].
fn minified_level(device: bool) -> f32 {
    match device {
        true => MINIFIED_ZERO,
        false => MINIFIED_NEAREST,
    }
}

/// [`minified_level`] of the device's sampler of a runtime sampler slot bound with sampler word
/// `word`, its anisotropy `(linear, nearest)` by the filters the word states
/// ([`DeviceAnisotropy::Filters`]); a word with a cubic filter takes the anisotropic one's.
fn word_minified_level(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (linear, nearest): (bool, bool),
) -> BResult<SValue> {
    if linear == nearest {
        return f32c(fb, minified_level(linear));
    }
    let both = u32c(fb, SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR)?;
    let filters = fb.binary(BinOp::BitwiseAnd, word, both)?;
    let both_linear = fb.binary(BinOp::IEqual, filters, both)?;
    let cubic = u32c(fb, SamplerWord::CUBIC)?;
    let cubic = fb.binary(BinOp::BitwiseAnd, word, cubic)?;
    let zero = u32c(fb, 0)?;
    let cubic = fb.binary(BinOp::INotEqual, cubic, zero)?;
    // The flags differ: the device's sampler is anisotropic where the word's filters are the
    // anisotropic case's (both linear, or one nearest), or where it states a cubic filter.
    let device = match linear {
        true => both_linear,
        false => fb.unary(UnOp::LogicalNot, both_linear)?,
    };
    let device = fb.binary(BinOp::LogicalOr, device, cubic)?;
    let zero_level = f32c(fb, MINIFIED_ZERO)?;
    let nearest_level = f32c(fb, MINIFIED_NEAREST)?;
    fb.select(device, zero_level, nearest_level)
}

/// The emulation's sum of the device's texels of `source` (its coordinate of `lanes` plane lanes):
/// the footprint, then up to `most` taps.
/// `minified` is the level handed for level 0 where Metal minifies under a nearest mip filter or
/// none ([`minified_level`]).
fn filtered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    src: (Source, u8),
    at: &At,
    (la, mip, minified): (SValue, SValue, SValue),
    most: u32,
) -> BResult<SValue> {
    let f = float(fb)?;
    let v2 = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let plane = fb.vector_shuffle(at.size, at.size, &[0, 1])?;
    let texels = fb.convert(ConvOp::ConvertUToF, plane, v2)?;
    let gx = fb.binary(BinOp::FMul, at.grads.0, texels)?;
    let gy = fb.binary(BinOp::FMul, at.grads.1, texels)?;
    let foot = footprint(fb, (gx, gy), (la, mip), at.levels)?;
    let levels = read_levels(fb, &foot, at, (mip, minified))?;
    // The taps' weights: `floor(256 / n)` each, the remainder one each to the middle taps.
    let n = foot.n;
    let full = i32c(fb, 256)?;
    let each = fb.binary(BinOp::SDiv, full, n)?;
    let used = fb.binary(BinOp::IMul, each, n)?;
    let rem = fb.binary(BinOp::ISub, full, used)?;
    let lo = fb.binary(BinOp::ISub, n, rem)?;
    let one = i32c(fb, 1)?;
    let lo = fb.binary(BinOp::ShiftRightArithmetic, lo, one)?;
    let hi = fb.binary(BinOp::IAdd, lo, rem)?;
    // The first tap's offset in texels along the major axis: `-(n - 1) s0 / 2`.
    let nf = fb.convert(ConvOp::ConvertSToF, n, f)?;
    let onef = f32c(fb, 1.0)?;
    let span = fb.binary(BinOp::FSub, nf, onef)?;
    let half = f32c(fb, 0.5)?;
    let span = fb.binary(BinOp::FMul, span, half)?;
    let taps = Taps {
        at: *at,
        u: foot.u,
        s0: foot.s0,
        span,
        texels,
        levels,
        weights: (each, lo, hi),
    };
    // Tap 0, then taps 1 to `n - 1` (at most `most`) in a loop of one tap's code that asks the
    // driver not to unroll it: each tap's arithmetic and the sum's order are those of the taps
    // written out one by one, so the answer is the same, and the driver compiles one tap instead
    // of `most`.
    let zero = i32c(fb, 0)?;
    let first = tap(cx, fb, src, &taps, zero)?;
    if most == 1 {
        return Ok(first);
    }
    let i = int(fb)?;
    let sum_ty = fb.value_ty(first)?;
    let sum = fb.local_var(sum_ty, None)?;
    let sum = fb.var_ptr(sum)?;
    fb.store(sum, first, MemAccess::default())?;
    let index = fb.local_var(i, None)?;
    let index = fb.var_ptr(index)?;
    let one = i32c(fb, 1)?;
    fb.store(index, one, MemAccess::default())?;
    let most = i32c(fb, most as i32)?;
    let body: LoopFn<'_, EmitCx<'_>> = Box::new(move |fb, cx, lc| {
        let j = fb.load(index, MemAccess::default())?;
        let past_n = fb.binary(BinOp::SGreaterThanEqual, j, n)?;
        let past_most = fb.binary(BinOp::SGreaterThanEqual, j, most)?;
        let done = fb.binary(BinOp::LogicalOr, past_n, past_most)?;
        let exit: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let t = tap(cx, fb, src, &taps, j)?;
        let kept = fb.load(sum, MemAccess::default())?;
        let total = fb.binary(BinOp::FAdd, kept, t)?;
        fb.store(sum, total, MemAccess::default())?;
        let one = i32c(fb, 1)?;
        let next = fb.binary(BinOp::IAdd, j, one)?;
        fb.store(index, next, MemAccess::default())
    });
    let spec = LoopSpec {
        exit_sel: None,
        dont_unroll: true,
    };
    fb.loop_(cx, spec, body)?;
    fb.load(sum, MemAccess::default())
}

/// Metal's footprint of one sample (the module docs).
struct Footprint {
    /// The level, held within the texture's levels.
    lambda: SValue,
    /// The level before it is held within the texture's levels: above 0 where Metal minifies.
    raw: SValue,
    /// The tap count (`i32`).
    n: SValue,
    /// The step direction in texels (`vec2`).
    u: SValue,
    /// The spacing in level-0 texels.
    s0: SValue,
}

/// Metal's footprint of gradients `gx`, `gy` (level-0 texels) through a sampler of degree `2^la`
/// (`i32`) and mip filter code `mip` (`u32`) on a texture of `levels` (`u32`) levels, its
/// logarithms and powers the device's `Log2` and `Exp2` (module docs).
fn footprint(
    fb: &mut FunctionBuilder<'_>,
    (gx, gy): (SValue, SValue),
    (la, mip): (SValue, SValue),
    levels: SValue,
) -> BResult<Footprint> {
    let f = float(fb)?;
    let i = int(fb)?;
    let v2 = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let lx = length(fb, gx)?;
    let ly = length(fb, gy)?;
    let major = fb.glsl(GlslOp::FMax, &[lx, ly])?;
    let x_major = fb.binary(BinOp::FOrdGreaterThanEqual, lx, ly)?;
    let b2 = bool2(fb)?;
    let pick = fb.composite_construct(b2, &[x_major, x_major])?;
    let g = fb.select(pick, gx, gy)?;
    let zf = f32c(fb, 0.0)?;
    let some = fb.binary(BinOp::FOrdGreaterThan, major, zf)?;
    let ux = fb.composite_extract(g, &[0])?;
    let uy = fb.composite_extract(g, &[1])?;
    let ux = fb.binary(BinOp::FDiv, ux, major)?;
    let uy = fb.binary(BinOp::FDiv, uy, major)?;
    let onef = f32c(fb, 1.0)?;
    let ux = fb.select(some, ux, onef)?;
    let uy = fb.select(some, uy, zf)?;
    let u = fb.composite_construct(v2, &[ux, uy])?;
    // A logarithm of no length: far below every level.
    let none = f32c(fb, -1.0e9)?;
    let lmaj = fb.glsl(GlslOp::Log2, &[major])?;
    let lmaj = fb.select(some, lmaj, none)?;
    let one = i32c(fb, 1)?;
    // The determinant of the gradients.
    let component = |fb: &mut FunctionBuilder<'_>, v: SValue, c: u32| -> BResult<SValue> {
        fb.composite_extract(v, &[c])
    };
    let (gxx, gyy) = (component(fb, gx, 0)?, component(fb, gy, 1)?);
    let (gxy, gyx) = (component(fb, gx, 1)?, component(fb, gy, 0)?);
    let p = fb.binary(BinOp::FMul, gxx, gyy)?;
    let q = fb.binary(BinOp::FMul, gxy, gyx)?;
    let det = fb.binary(BinOp::FSub, p, q)?;
    let det = fb.glsl(GlslOp::FAbs, &[det])?;
    let has_det = fb.binary(BinOp::FOrdGreaterThan, det, zf)?;
    let ldet = fb.glsl(GlslOp::Log2, &[det])?;
    // The minor axis's level, the major held at a texel or more; the major's past the degree.
    let held = fb.glsl(GlslOp::FMax, &[lmaj, zf])?;
    let lmin = fb.binary(BinOp::FSub, ldet, held)?;
    let lmin = fb.select(has_det, lmin, none)?;
    let laf = fb.convert(ConvOp::ConvertSToF, la, f)?;
    let capped = fb.binary(BinOp::FSub, lmaj, laf)?;
    let lv = fb.glsl(GlslOp::FMax, &[lmin, capped])?;
    // The log ratio, the minor at a texel or more.
    let excess = fb.binary(BinOp::FSub, ldet, lmaj)?;
    let excess = fb.glsl(GlslOp::FMax, &[excess, zf])?;
    let excess = fb.select(has_det, excess, zf)?;
    let lr = fb.binary(BinOp::FSub, lmaj, excess)?;
    // The tap count: the first count whose threshold the ratio does not pass, at most `2^la`.
    let degree = fb.binary(BinOp::ShiftLeftLogical, one, la)?;
    let mut n = degree;
    for &(count, th) in COUNTS.iter().rev() {
        let th = f32c(fb, th as f32 / 64.0)?;
        let within = fb.binary(BinOp::FOrdLessThanEqual, lr, th)?;
        let count = i32c(fb, count)?;
        n = fb.select(within, count, n)?;
    }
    let n = fb.glsl(GlslOp::SMin, &[n, degree])?;
    // The spacing: `major · g · S`.
    let r = fb.glsl(GlslOp::Exp2, &[lr])?;
    let degf = fb.convert(ConvOp::ConvertSToF, degree, f)?;
    let rp = fb.glsl(GlslOp::FMin, &[r, degf])?;
    let nf = fb.convert(ConvOp::ConvertSToF, n, f)?;
    let half = f32c(fb, 0.5)?;
    let two_g = fb.binary(BinOp::FSub, rp, onef)?;
    let two_g = fb.binary(BinOp::FMul, two_g, half)?;
    let mut kappa = f32c(fb, KAPPA.last().map_or(0.0, |&(_, k)| k))?;
    for &(count, k) in KAPPA.iter().rev().skip(1) {
        let c = i32c(fb, count)?;
        let is = fb.binary(BinOp::IEqual, n, c)?;
        let k = f32c(fb, k)?;
        kappa = fb.select(is, k, kappa)?;
    }
    let share = fb.binary(BinOp::FDiv, rp, nf)?;
    let short = fb.binary(BinOp::FSub, onef, share)?;
    let short = fb.binary(BinOp::FMul, kappa, short)?;
    let many_g = fb.binary(BinOp::FSub, onef, short)?;
    let many_g = fb.binary(BinOp::FDiv, many_g, nf)?;
    let two = i32c(fb, 2)?;
    let is_two = fb.binary(BinOp::IEqual, n, two)?;
    let g = fb.select(is_two, two_g, many_g)?;
    // `S = 2^-past`: the level past 0, less the level past the texture's last under a mip filter.
    let lvs = fb.glsl(GlslOp::FMax, &[lv, zf])?;
    let zi = i32c(fb, 0)?;
    let last = fb.convert(ConvOp::Bitcast, levels, i)?;
    let last = fb.binary(BinOp::ISub, last, one)?;
    let last = fb.glsl(GlslOp::SMax, &[last, zi])?;
    let top = fb.convert(ConvOp::ConvertSToF, last, f)?;
    let within = fb.glsl(GlslOp::FMin, &[lvs, top])?;
    let zu = u32c(fb, 0)?;
    let mipped = fb.binary(BinOp::INotEqual, mip, zu)?;
    let lvc = fb.select(mipped, within, zf)?;
    let past = fb.binary(BinOp::FSub, lvc, lvs)?;
    let sc = fb.glsl(GlslOp::Exp2, &[past])?;
    let gs = fb.binary(BinOp::FMul, g, sc)?;
    let s0 = fb.binary(BinOp::FMul, major, gs)?;
    let several = fb.binary(BinOp::SGreaterThan, n, one)?;
    let s0 = fb.select(several, s0, zf)?;
    // The level within the texture's levels.
    let lambda = fb.glsl(GlslOp::FClamp, &[lv, zf, top])?;
    Ok(Footprint {
        lambda,
        raw: lv,
        n,
        u,
        s0,
    })
}

/// Metal's length of gradient `v` (a `vec2`): `max(M + 3m/16, 49M/64 + 21m/32)`.
fn length(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
    let x = fb.composite_extract(v, &[0])?;
    let y = fb.composite_extract(v, &[1])?;
    metal_length(fb, x, y)
}

/// Metal's length of the gradient of components `x` and `y` (`f32`s): `max(M + 3m/16, 49M/64 +
/// 21m/32)` of their larger and smaller magnitude `M` and `m`, each product and sum rounded as
/// written (no contraction).
pub(super) fn metal_length(fb: &mut FunctionBuilder<'_>, x: SValue, y: SValue) -> BResult<SValue> {
    let x = fb.glsl(GlslOp::FAbs, &[x])?;
    let y = fb.glsl(GlslOp::FAbs, &[y])?;
    let big = fb.glsl(GlslOp::FMax, &[x, y])?;
    let small = fb.glsl(GlslOp::FMin, &[x, y])?;
    let term = |fb: &mut FunctionBuilder<'_>, p: f32, q: f32| -> BResult<SValue> {
        let pc = f32c(fb, p)?;
        let qc = f32c(fb, q)?;
        let a = fb.binary_with(BinOp::FMul, big, pc, Contraction::Forbidden)?;
        let b = fb.binary_with(BinOp::FMul, small, qc, Contraction::Forbidden)?;
        fb.binary_with(BinOp::FAdd, a, b, Contraction::Forbidden)
    };
    let first = term(fb, 1.0, 0.1875)?;
    let second = term(fb, 0.765_625, 0.656_25)?;
    fb.glsl(GlslOp::FMax, &[first, second])
}

/// The level each tap reads ([`read_levels`]).
#[derive(Clone, Copy)]
struct Levels {
    first: SValue,
}

/// The level a tap reads at Metal's level `foot.lambda`, raised to the sample's `min_lod_clamp`,
/// by mip filter code `mip`: the level itself without a mip filter (so the device chooses its
/// minification or magnification filter by its sign), the nearer whole level under a nearest one
/// (`L + 1` past half a level), and the fractional level under a linear one, whose two levels the
/// device's own linear mip filter blends. Level 0 where Metal minifies is handed as
/// `MINIFIED_ZERO` under a linear mip filter and as `minified` under the others
/// ([`minified_level`], by the device's sampler's anisotropy): where its level before the
/// texture's levels hold it (`foot.raw`, with the clamp) is above 0, so a texture of one level
/// minifies as Metal does (an isotropic cube's threshold, `1 + 3/2048`, measured on a cube of one
/// level).
fn read_levels(
    fb: &mut FunctionBuilder<'_>,
    foot: &Footprint,
    at: &At,
    (mip, minified_nearest): (SValue, SValue),
) -> BResult<Levels> {
    let level = |fb: &mut FunctionBuilder<'_>, l: SValue| -> BResult<SValue> {
        match at.min_lod {
            Some(m) => fb.glsl(GlslOp::FMax, &[l, m]),
            None => Ok(l),
        }
    };
    let lf = level(fb, foot.lambda)?;
    let raw = level(fb, foot.raw)?;
    let zf = f32c(fb, 0.0)?;
    let above = fb.binary(BinOp::FOrdGreaterThan, raw, zf)?;
    let whole = fb.glsl(GlslOp::Floor, &[lf])?;
    let q = fb.binary(BinOp::FSub, lf, whole)?;
    let half = f32c(fb, 0.5)?;
    let up = fb.binary(BinOp::FOrdGreaterThan, q, half)?;
    let onef = f32c(fb, 1.0)?;
    let next = fb.binary(BinOp::FAdd, whole, onef)?;
    let nearest = fb.select(up, next, whole)?;
    // Level 0 above Metal's 0 is handed as `MINIFIED_ZERO` or `minified_nearest`, which keep the
    // device's minification filter.
    let minified = f32c(fb, MINIFIED_ZERO)?;
    let nearest_at_zero = fb.binary(BinOp::FOrdEqual, nearest, zf)?;
    let lift = fb.binary(BinOp::LogicalAnd, nearest_at_zero, above)?;
    let nearest = fb.select(lift, minified_nearest, nearest)?;
    let at_zero = fb.binary(BinOp::FOrdEqual, lf, zf)?;
    let lift = fb.binary(BinOp::LogicalAnd, at_zero, above)?;
    let linear = fb.select(lift, minified, lf)?;
    let one = u32c(fb, 1)?;
    let two = u32c(fb, 2)?;
    let is_nearest = fb.binary(BinOp::IEqual, mip, one)?;
    let is_linear = fb.binary(BinOp::IEqual, mip, two)?;
    let unfiltered = fb.select(lift, minified_nearest, lf)?;
    let first = fb.select(is_nearest, nearest, unfiltered)?;
    let first = fb.select(is_linear, linear, first)?;
    Ok(Levels { first })
}

/// Everything a tap reads.
#[derive(Clone, Copy)]
struct Taps {
    at: At,
    u: SValue,
    s0: SValue,
    /// `(n - 1) / 2`.
    span: SValue,
    /// The plane's level-0 size as floats.
    texels: SValue,
    levels: Levels,
    /// `floor(256 / n)`, and the taps from `lo` below `hi` that take one more 256th.
    weights: (SValue, SValue, SValue),
}

/// Tap `j`'s weighted texel (`vec4`), `j` an `int`, its samples made in the open block (and in
/// an arm of its own for the second level).
fn tap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (source, lanes): (Source, u8),
    t: &Taps,
    j: SValue,
) -> BResult<SValue> {
    let f = float(fb)?;
    let jf = fb.convert(ConvOp::ConvertSToF, j, f)?;
    let k = fb.binary(BinOp::FSub, jf, t.span)?;
    let off = fb.binary(BinOp::FMul, k, t.s0)?;
    let step = fb.binary(BinOp::VectorTimesScalar, t.u, off)?;
    let step = fb.binary(BinOp::FDiv, step, t.texels)?;
    // A cube's tap: the direction moved across its face by the step in the face's `s` and `t`,
    // which the device projects back onto that face (or, past its edge, onto the next).
    let step = match t.at.face {
        Some(face) => {
            let ds = fb.composite_extract(step, &[0])?;
            let dt = fb.composite_extract(step, &[1])?;
            let along_s = fb.binary(BinOp::VectorTimesScalar, face.s, ds)?;
            let along_t = fb.binary(BinOp::VectorTimesScalar, face.t, dt)?;
            fb.binary(BinOp::FAdd, along_s, along_t)?
        }
        None => step,
    };
    let p = fb.binary(BinOp::FAdd, t.at.coord, step)?;
    let p = match t.at.layer {
        Some(l) => {
            let lf = fb.convert(ConvOp::ConvertUToF, l, f)?;
            let with_layer = fb.module().ty(SType::Vector {
                elem: f,
                len: lanes + 1,
            })?;
            fb.composite_construct(with_layer, &[p, lf])?
        }
        None => p,
    };
    let (each, lo, hi) = t.weights;
    let from = fb.binary(BinOp::SGreaterThanEqual, j, lo)?;
    let below = fb.binary(BinOp::SLessThan, j, hi)?;
    let middle = fb.binary(BinOp::LogicalAnd, from, below)?;
    let one = i32c(fb, 1)?;
    let zi = i32c(fb, 0)?;
    let extra = fb.select(middle, one, zi)?;
    let w = fb.binary(BinOp::IAdd, each, extra)?;
    let w = fb.convert(ConvOp::ConvertSToF, w, f)?;
    let inv = f32c(fb, 1.0 / 256.0)?;
    let w = fb.binary(BinOp::FMul, w, inv)?;
    let ops = t.at.ops;
    let x = read(cx, fb, source, (p, ops), t.levels.first)?;
    fb.binary(BinOp::VectorTimesScalar, x, w)
}

/// The device's sample of `source` at `p` with offset operands `ops` at explicit level `lod`, from
/// a sampled image made in the open block.
fn read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    source: Source,
    (p, mut ops): (SValue, ImageOperands),
    lod: SValue,
) -> BResult<SValue> {
    let (si, _, _) = super::emit::sampled_image(cx, fb, source)?;
    ops.lod = Some(lod);
    fb.sample(si, p, None, false, &ops)
}

fn float(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

fn int(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })
}

fn bool2(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    let b = fb.module().ty(SType::Bool)?;
    fb.module().ty(SType::Vector { elem: b, len: 2 })
}

fn f32c(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let f = float(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: f,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn i32c(fb: &mut FunctionBuilder<'_>, x: i32) -> BResult<SValue> {
    let i = int(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: i,
        bits: ConstBits(u64::from(x as u32)),
    })?;
    fb.module().const_value(c)
}

fn u32c(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

#[cfg(test)]
mod tests;
