//! Metal's private fixed-function pipeline state of OpenGL on Metal (Apple SPI of
//! `MTLRenderPipelineDescriptor`): which properties the translation builds, and the refusal of the
//! rest.
//!
//! Measured on an Apple M4 (macOS 27.0.1; owned Objective-C and MSL probes setting each property
//! by key-value coding):
//!
//! - `sampleMask` ANDs the coverage of a pipeline with a fragment function, the bits past the
//!   pipeline's sample count ignored, after the function runs: the function runs for every
//!   rasterized sample, masked or not, its `[[sample_mask]]` reads the rasterized coverage, and the
//!   color, depth and stencil writes and the visibility count then keep only the mask's samples
//!   (ANDed with the function's own `[[sample_mask]]` output and alpha to coverage); under
//!   `[[early_fragment_tests]]` only the colors. A pipeline without a fragment function ignores the
//!   mask: its depth and stencil writes and its visibility count keep every sample. Vulkan's
//!   `pSampleMask` ([`GraphicsState::sample_mask`]) masks before the function runs, so it is
//!   Metal's only where the function cannot tell; elsewhere the function applies the mask itself in
//!   its `SampleMask` output (`LoweringInputs::sample_mask`, [`shader_sample_mask`]), which Vulkan
//!   applies after it runs, and after early tests to the colors alone ([`sample_mask`]).
//! - `clipDistanceEnableMask`, `pointCoordLowerLeft`, `pointSizeOutputVS`, `twoSideEnabled`,
//!   `vertexDepthCompareClampMask`, `depthStencilWriteDisabled`, `sampleCoverageInvert` and
//!   `needsCustomBorderColorSamplers` (render and compute) change nothing Metal draws, at any
//!   value tried: they are kept and read by nothing. A sampler's private custom border color
//!   (`borderColorSPI` 3 with its four values) samples transparent black, the values ignored,
//!   with the flag or without: the raw border color 3 the runtime already
//!   samples so (`sampler_border_color`).
//! - `alphaTestEnabled` tests each fragment after its function runs (probes on an Apple M3 Max,
//!   macOS 27.0.1): color 0's alpha and the encoder's reference (`setAlphaTestReferenceValue:`, 1
//!   until set in each encoder) are each clamped to `[0, 1]`, a NaN taken as 0, and converted to an
//!   8-bit unorm code as an 8-bit unorm attachment stores a float, whatever color 0's format; the
//!   fragment passes where `alphaTestFunction` holds between the alpha's code and the
//!   reference's, and one that fails writes nothing and counts no visibility. The fragment
//!   function runs the test itself ([`alpha_test`], `LoweringInputs::alpha_test`); `Always`
//!   changes nothing. Metal's compiler fails a function that writes no color 0 under the test,
//!   and a function past `Always`, a pipeline without a fragment function and a color 0 other
//!   than four `float` or `half` lanes are refused, their answers unmeasured.
//! - `fragmentDepthCompareClampMask` clamps the reference of a fragment function's depth
//!   comparison (`sample_compare` and `gather_compare`) to `[0, 1]`, a NaN taken as 0: bit `s`
//!   through `[[sampler(s)]]`, whichever texture slot it compares, and bit 31 through any
//!   constexpr sampler (probes on an Apple M3 Max, macOS 27.0.1). The fragment function clamps
//!   them itself (`LoweringInputs::depth_compare_clamp_mask`, [`depth_compares_clamped`]); a
//!   comparison through an argument buffer's sampler, which no bit names, is refused under a mask
//!   other than 0.
//! - `vertexAmplificationMode` 0 and 1 draw alike, as the encoder's amplification count and view
//!   mappings say, at counts 1 and 2 (probes on an Apple M3 Max); Metal's descriptor asserts at 2
//!   or more (`vertexAmplificationMode value 2 is invalid`), refused as an unknown value.
//!   `explicitVisibilityGroupID` changes nothing in a pipeline whose `maxVertexAmplificationCount`
//!   is 1; where the pipeline amplifies, a group other than the default draws one copy's color and
//!   counts every copy's samples on Metal, a rule not built: refused
//!   (`PrivateRenderField::ExplicitVisibilityGroup`).
//! - `logicOperationEnabled` applies `logicOperation` (its low four bits, in Direct3D's order:
//!   clear, set, copy, copy inverted, no-op, invert, and, nand, or, nor, xor, equivalent, and
//!   reverse, and inverted, or reverse, or inverted) to the codes of each normalized, sRGB and
//!   integer color attachment, the fragment's code against the stored one, blending ignored there,
//!   and the write mask kept after it; a 16- or 32-bit float attachment ignores it and blends
//!   (probes on an Apple M3 Max, macOS 27.0.1: `RGBA8Unorm`, `RGBA8Snorm`, `RGBA8Unorm_sRGB`,
//!   `BGRA8Unorm`, `RGB10A2Unorm`, `R16Unorm`, `RGBA8Uint`, `RGBA8Sint`, `R32Uint`, `RGBA16Float`,
//!   every operation, blended and not). [`logic_operation`] draws copy and no-op on every device
//!   by the attachments' blends and masks, and the rest through Vulkan's `logicOp`
//!   ([`GraphicsState::logic_op`]), which applies the same operation to the same codes of a
//!   normalized or integer attachment, but for a 10-bit format's, which Metal widens to 16 bits
//!   first: Vulkan's operation is the device's native form there, recorded
//!   (`PrecisionNote::NativeLogicOperation`, [`note_vulkan_codes`]). A device without it that
//!   orders framebuffer fetch (the
//!   pixel interlock, or rasterization-order input attachments) has the fragment function compute
//!   the operation itself, over the texel its first output's write reads (`lower::fragment`'s
//!   `logic`; probe `logic_operation` on an Apple M3 Max, macOS 27.0.1: every operation into
//!   `R8Unorm`, `A8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`, `R16Unorm`, `RGBA16Unorm`, `RGB10A2Unorm`,
//!   `BGR10A2Unorm`, `RG8Snorm`,
//!   `RGBA8Snorm`, `RG16Snorm`, `RGBA16Snorm`, `RGBA8Uint`, `R16Uint`, `R32Uint`, `RGBA8Sint`,
//!   `RG16Sint` and `R32Sint`, some under write masks, at 4 samples and over overlapping
//!   fragments): a NaN takes the channel's low end (0, or -1 signed), a signed
//!   normalized channel's lowest code reads and stores as the code above it, and an integer output
//!   saturates to the channel's range before the operation; `RGB10A2Unorm` and `BGR10A2Unorm`
//!   operate on their codes widened to 16 bits, the result narrowed back. It refuses formats
//!   outside that rule (`RGB10A2Uint` and the packed 16-bit formats, whose operation Metal
//!   computes otherwise), color 0 under alpha to coverage, and a function that writes the attachment
//!   through a shadow stencil, an imageblock or a merged word ([`logic_operated`]). A device that
//!   orders fetch in neither form refuses the operation naming the features. So is an operation
//!   other than copy and no-op refused where Vulkan's form is not Metal's: an sRGB attachment
//!   (Vulkan applies none there), a float attachment that blends under Vulkan's operation (which
//!   turns every blend off), a format whose operation Metal was not measured with
//!   (extended-range, packed small-float, YCbCr), or a function that writes an attachment itself
//!   ([`logic_operated`]).
//! - `sampleCoverage` keeps a fraction of the samples, dithered over each 2x2 quad of window
//!   pixels, and is applied as `sampleMask` is, after the function runs: the value clamped to
//!   `[0, 1]` (a NaN taken as 0) and converted to an 8-bit unorm code `c`; at one sample a pixel
//!   is covered where the value is above 0.5; at 2 and 4 samples the quad keeps `floor((c + s / 2)
//!   / s)` samples, `s = 64 / n`, handed out one at a time to its pixels in the order `(0, 0)`,
//!   `(1, 1)`, `(1, 0)`, `(0, 1)` of the window pixel's coordinates modulo 2 (the viewport's
//!   origin plays no part), and each pixel's filled in the sample order 0, 3, 1, 2 at 4 samples
//!   and 0, 1 at 2 (probes `sc` to `sc4` on an Apple M3 Max, macOS 27.0.1: every code at 1, 2
//!   and 4 samples, ties, NaN and infinities, under a viewport offset, with the sample mask and
//!   the function's own `[[sample_mask]]` output). The coverage is ANDed with both masks and
//!   alpha to coverage, and `sampleCoverageInvert` changes nothing. As with the sample mask, the
//!   function runs for every rasterized sample and its `[[sample_mask]]` reads the rasterized
//!   coverage; colors, depth, stencil and visibility counts keep the covered samples, under
//!   `[[early_fragment_tests]]` only the colors, and a pipeline without a fragment function
//!   ignores it. [`applied_masks`] gives the two masks together by the pixel's place in its quad;
//!   a pipeline whose masks differ between places has its fragment function apply them itself
//!   (`LoweringInputs::sample_mask`), from the window pixel's coordinates. A value other than 1
//!   at a sample count Metal was not measured at (8) is refused `NotImplemented`
//!   ([`Feature::PrivateRenderState`]) naming it.
//! - `openGLModeEnabled` changes nothing Metal draws or reflects (probe
//!   `conformance/probes/opengl_mode.m` on an Apple M3 Max, macOS 27.0.1, each draw made with the
//!   mode off and on and its records compared byte for byte: depth bands at clip z 0.25, -0.5 and
//!   1.5, the window origin, facing under `twoSideEnabled` 0 and 1, the flat input of triangle and
//!   line lists and strips, points of size 1 and 3 with their `[[point_coord]]` under
//!   `pointCoordLowerLeft` and `pointSizeOutputVS` 0 and 1, lines, a clip distance under
//!   `clipDistanceEnableMask` 0xff and 0, the sample positions at 4 samples, `sampleCoverage` 0.5
//!   inverted and not, `sampleMask`, a logic operation, `depthStencilWriteDisabled` 0 and 1, and
//!   the bindings a pipeline reflects): the properties above that change nothing change nothing
//!   with it either. It makes two states Metal otherwise refuses, which the runtime answers: a
//!   fragment function writing depth without a depth attachment, whose colors draw as without the
//!   output, and a rasterizing state whose vertex function returns `void`, whose draws run the
//!   vertex function and rasterize nothing. The other refusals tried stand: fragment inputs no
//!   vertex output writes or of another type, a `float` output to an integer attachment, and
//!   stage-in attributes without a vertex descriptor or missing from it. The compile reads it not.
//! - `pointSmoothEnabled` changes what Metal draws; until the translation builds it, on is
//!   refused `NotImplemented` naming it.
//! - `textureWriteRoundingMode` 2 rounds a fragment function's texture writes to half-float and
//!   `RG11B10Float` textures to nearest even, which the translation has not built: such a write is
//!   refused naming it (`PrivateRenderField::TextureWriteRounding`), where the function's slot
//!   declares such a format, and at the draw where one is bound (`lower::texture`'s
//!   `mode_rounded_write`). Its color writes are built.
//!
//! [`GraphicsState::sample_mask`]: crate::pipeline::GraphicsState::sample_mask

use std::borrow::Cow;

use crate::device::requirements::VkFeature;
use crate::device::DeviceProfile;
use crate::error::{
    Attachment, Capability, DescriptorError, Error, Feature, PrivateRenderField, RuntimeInput,
    Unsupported,
};
use crate::library::FunctionArtifact;
use crate::metal::format::{write_conversion, WriteConversion};
use crate::metal::{ColorWriteMask, CompareFunction, PixelFormat};
use crate::pipeline::artifact::{PrecisionNote, StageModule};
use crate::pipeline::descriptor::{PrivateRenderState, RenderPipelineDescriptor};
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::pipeline::stage_io::{DepthCompares, StageBuiltin};
use crate::pipeline::sym::SampledType;
use crate::pipeline::GraphicsState;
use crate::vk;

/// The refusal of `field` at a value the translation has not built.
fn refused(field: PrivateRenderField) -> Error {
    Error::Unsupported(Unsupported::not_implemented(Feature::PrivateRenderState(
        field,
    )))
}

/// `Ok` where every property of the private state of `d` is Metal's default or built (the module
/// docs), else the refusal naming the first that is not.
pub(crate) fn check_render(d: &RenderPipelineDescriptor) -> Result<(), Error> {
    use PrivateRenderField as F;
    let s = &d.private;
    alpha_test(s)?;
    // Metal's descriptor asserts at a mode of 2 or more.
    if s.vertex_amplification_mode > 1 {
        return Err(Error::InvalidDescriptor(
            DescriptorError::UnknownEnumValue {
                ty: "vertexAmplificationMode",
                value: u32::try_from(s.vertex_amplification_mode).unwrap_or(u32::MAX),
            },
        ));
    }
    if s.explicit_visibility_group_id != u32::MAX && d.max_vertex_amplification_count > 1 {
        return Err(refused(F::ExplicitVisibilityGroup));
    }
    if s.sample_coverage != 1.0
        && coverage_masks(s.sample_coverage, d.raster_sample_count).is_none()
    {
        return Err(refused(F::SampleCoverage));
    }
    match s.point_smooth_enabled {
        true => Err(refused(F::PointSmooth)),
        false => Ok(()),
    }
}

/// The samples of a pixel at `samples` samples, as a mask.
fn all_samples(samples: u32) -> u32 {
    match samples {
        n if n >= 32 => u32::MAX,
        n => (1u32 << n) - 1,
    }
}

/// The masks Metal's `sampleCoverage` of `coverage` keeps at `samples` samples, by the window
/// pixel's place in its 2x2 quad, index `(x & 1) | (y & 1) << 1` (the module docs); `None` at a
/// sample count Metal was not measured at.
pub(crate) fn coverage_masks(coverage: f32, samples: u32) -> Option<[u32; 4]> {
    /// Each place's turn in taking the quad's samples: `(0, 0)`, `(1, 1)`, `(1, 0)`, `(0, 1)`.
    const TURN: [u32; 4] = [0, 2, 3, 1];
    let order: &[u32] = match samples {
        1 => return Some([u32::from(coverage > 0.5); 4]),
        2 => &[0, 1],
        4 => &[0, 3, 1, 2],
        _ => return None,
    };
    let value = if coverage.is_nan() {
        0.0
    } else {
        coverage.clamp(0.0, 1.0)
    };
    // The exact product rounded to nearest; its one tie, 0.5, rounds to 128 either way.
    let code = (f64::from(value) * 255.0).round_ties_even() as u32;
    let step = 64 / samples;
    let kept = (code + step / 2) / step;
    Some(TURN.map(|turn| {
        let taken = (kept + 3 - turn) / 4;
        order[..taken as usize]
            .iter()
            .fold(0, |mask, sample| mask | 1 << sample)
    }))
}

/// Whether `masks` keeps the same samples at every place of the quad.
pub(crate) fn uniform(masks: &[u32; 4]) -> bool {
    masks.iter().all(|&m| m == masks[0])
}

/// The alpha test of the private state `s` its fragment function runs
/// (`LoweringInputs::alpha_test`): the comparison where the test is on, `None` where it is off or
/// `Always`, which keeps every fragment; a function past `Always` is refused
/// (`PrivateRenderField::AlphaTest`), its answer unmeasured.
pub(crate) fn alpha_test(s: &PrivateRenderState) -> Result<Option<CompareFunction>, Error> {
    if !s.alpha_test_enabled {
        return Ok(None);
    }
    let function = u32::try_from(s.alpha_test_function)
        .ok()
        .and_then(|raw| CompareFunction::try_from(raw).ok())
        .ok_or_else(|| refused(PrivateRenderField::AlphaTest))?;
    Ok((function != CompareFunction::Always).then_some(function))
}

/// `Ok` where the fragment function `fragment` of the pipeline `d` runs the pipeline's alpha
/// test ([`alpha_test`]) as compiled: a pipeline with the test on and no fragment function is
/// refused (`PrivateRenderField::AlphaTest`, Metal's answer unmeasured), and a function compiled
/// without the test, or for another comparison, is compiled for the pipeline again
/// (`NeedsInput(AlphaTest)`).
pub(crate) fn alpha_tested(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
) -> Result<(), Error> {
    let wanted = alpha_test(&d.private)?;
    match fragment {
        None if wanted.is_some() => Err(refused(PrivateRenderField::AlphaTest)),
        None => Ok(()),
        Some(f) if f.interface.io.alpha_test != wanted => {
            Err(Error::NeedsInput(RuntimeInput::AlphaTest))
        }
        Some(_) => Ok(()),
    }
}

/// `Ok` where the fragment function `fragment` of the pipeline `d` clamps the references of its
/// depth comparisons as the pipeline's `fragmentDepthCompareClampMask` says
/// (`LoweringInputs::depth_compare_clamp_mask`): one through an argument buffer's sampler under a
/// mask other than 0 is refused (`PrivateRenderField::FragmentDepthCompareClamp`, Metal's answer
/// unmeasured), and one compiled for another mask over its samplers is compiled for the pipeline
/// again (`NeedsInput(DepthCompareClamp)`).
pub(crate) fn depth_compares_clamped(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
) -> Result<(), Error> {
    let mask = d.private.fragment_depth_compare_clamp_mask;
    let Some(c) = fragment.map(|f| f.interface.io.depth_compares) else {
        return Ok(());
    };
    if mask != 0 && c.samplers & DepthCompares::ARGUMENT != 0 {
        return Err(refused(PrivateRenderField::FragmentDepthCompareClamp));
    }
    match c.clamped == mask & c.samplers & !DepthCompares::ARGUMENT {
        true => Ok(()),
        false => Err(Error::NeedsInput(RuntimeInput::DepthCompareClamp)),
    }
}

/// The masks Metal applies to the coverage of the pipeline `d` after its fragment function runs,
/// by the window pixel's place in its 2x2 quad ([`coverage_masks`]): the private sample mask's
/// bits below the sample count ANDed with the sample coverage's, where `d` has a fragment function
/// under enabled rasterization; `None` where they keep every sample at every place, or `d` has no
/// fragment stage, which ignores both.
pub(crate) fn applied_masks(d: &RenderPipelineDescriptor) -> Option<[u32; 4]> {
    if d.fragment.is_none() || !d.rasterization_enabled {
        return None;
    }
    let all = all_samples(d.raster_sample_count);
    // Metal ignores the bits past the sample count, so the high word is dropped.
    let mask = (d.private.sample_mask as u32) & all;
    let coverage = match d.private.sample_coverage == 1.0 {
        true => [all; 4],
        // `check_render` refuses a sample count without Metal's masks.
        false => coverage_masks(d.private.sample_coverage, d.raster_sample_count)?,
    };
    let masks = coverage.map(|c| c & mask);
    (masks != [all; 4]).then_some(masks)
}

/// Whether the fragment function `a` could tell Vulkan's sample mask, applied before it runs,
/// from Metal's, applied after: it writes memory (Metal runs it for the masked samples too), reads
/// `[[sample_mask]]` (Metal's is the rasterized coverage), or declares `[[early_fragment_tests]]`
/// (Metal's mask then keeps every sample's depth, stencil and visibility count, and only the
/// colors' masked samples unwritten).
fn observes_sample_mask(a: &FunctionArtifact) -> bool {
    let io = &a.interface.io;
    a.interface.writes_memory
        || io.builtins.contains(&StageBuiltin::SampleMaskIn)
        || io.early_fragment_tests
}

/// The private masks the fragment function of `d`, compiled without them as `a`, applies itself
/// (`LoweringInputs::sample_mask`): the masks Metal applies ([`applied_masks`]), where they differ
/// between the places of a quad, which Vulkan's one mask cannot say, or where `a` could tell
/// Vulkan's mask from them ([`observes_sample_mask`]); `None` elsewhere.
pub(crate) fn shader_sample_mask(
    d: &RenderPipelineDescriptor,
    a: &FunctionArtifact,
) -> Option<[u32; 4]> {
    applied_masks(d).filter(|m| !uniform(m) || observes_sample_mask(a))
}

/// [`GraphicsState::sample_mask`] for the pipeline `d`, its fragment function `fragment` and
/// its graphics state `g`, or the refusal where neither Vulkan's mask nor the function's own
/// (`StageIo::sample_mask`) is Metal's. A function that could tell Vulkan's mask from Metal's,
/// or of a pipeline whose masks differ between the places of a quad, applies the masks itself,
/// and the pipeline has no Vulkan mask; one compiled without them, or for others, is compiled for
/// the pipeline again (`NeedsInput(SampleMask)`). A pipeline whose fragment function writes an
/// attachment itself (a storage attachment, a shadow stencil or an imageblock, none measured
/// under the masks) is refused, naming the sample coverage where it is not 1.
pub(crate) fn sample_mask(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
    g: &GraphicsState,
) -> Result<Option<u32>, Error> {
    let applied = applied_masks(d);
    let own = fragment.and_then(|f| f.interface.io.sample_mask);
    if own.is_some() && own != applied {
        return Err(Error::NeedsInput(RuntimeInput::SampleMask));
    }
    let Some(masks) = applied else {
        return Ok(None);
    };
    let writes_itself = !g.storage_attachments.is_empty()
        || g.shadow_stencil.is_some()
        || g.fragment_imageblock.is_some();
    if writes_itself {
        return Err(refused(match d.private.sample_coverage == 1.0 {
            true => PrivateRenderField::SampleMask,
            false => PrivateRenderField::SampleCoverage,
        }));
    }
    match (own, fragment) {
        (Some(_), _) => Ok(None),
        (None, Some(f)) if observes_sample_mask(f) || !uniform(&masks) => {
            Err(Error::NeedsInput(RuntimeInput::SampleMask))
        }
        (None, _) => Ok(Some(masks[0])),
    }
}

/// Metal's logic operations by the low four bits of `logicOperation` (Direct3D's order), as
/// Vulkan's.
const LOGIC_OPS: [vk::LogicOp; 16] = [
    vk::LogicOp::CLEAR,
    vk::LogicOp::SET,
    vk::LogicOp::COPY,
    vk::LogicOp::COPY_INVERTED,
    vk::LogicOp::NO_OP,
    vk::LogicOp::INVERT,
    vk::LogicOp::AND,
    vk::LogicOp::NAND,
    vk::LogicOp::OR,
    vk::LogicOp::NOR,
    vk::LogicOp::XOR,
    vk::LogicOp::EQUIVALENT,
    vk::LogicOp::AND_REVERSE,
    vk::LogicOp::AND_INVERTED,
    vk::LogicOp::OR_REVERSE,
    vk::LogicOp::OR_INVERTED,
];

/// How Metal's logic operation takes a color attachment of a format (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogicClass {
    /// A normalized or integer format: the operation on its codes, blending ignored.
    Codes,
    /// An 8-bit sRGB format: the operation on its encoded codes, blending ignored.
    Srgb,
    /// A 16- or 32-bit float format: no operation, the attachment's blend.
    Float,
    /// A format whose operation Metal was not measured with.
    Unmeasured,
}

/// The class of logic operation an attachment of `pf` takes (the module docs): by the
/// conversion of its channels (`metal::format::write_conversion`).
fn logic_class(pf: PixelFormat) -> LogicClass {
    let half = ChannelWrite::FloatTowardZero {
        mantissa: 10,
        signed: true,
    };
    match write_conversion(pf) {
        Some(WriteConversion::Exact) => match crate::metal::format::sampled_type(pf) {
            Some(SampledType::Uint | SampledType::Sint) => LogicClass::Codes,
            _ => LogicClass::Unmeasured,
        },
        Some(WriteConversion::Float) => LogicClass::Float,
        Some(WriteConversion::Rules(rules)) => {
            let rules = rules.iter().flatten();
            if rules.clone().any(|r| *r == ChannelWrite::Srgb) {
                LogicClass::Srgb
            } else if rules.clone().all(|r| {
                matches!(
                    r,
                    ChannelWrite::Unorm { .. }
                        | ChannelWrite::Snorm { .. }
                        | ChannelWrite::HalfUnorm { .. }
                )
            }) {
                LogicClass::Codes
            } else if rules.clone().all(|r| *r == half) {
                LogicClass::Float
            } else {
                LogicClass::Unmeasured
            }
        }
        _ => LogicClass::Unmeasured,
    }
}

/// How a pipeline draws its private logic operation other than copy and no-op
/// ([`logic_operation`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LogicDraw {
    /// Through Vulkan's `logicOp` ([`GraphicsState::logic_op`]).
    Vulkan(vk::LogicOp),
    /// By the fragment function, over the texel its write reads through framebuffer fetch
    /// (`LoweringInputs::logic_operation`, `lower::fragment`'s `logic`), on a device without
    /// Vulkan's `logicOp` that orders the fetch.
    Shader(vk::LogicOp),
}

impl LogicDraw {
    /// The Vulkan logic operation the pipeline draws with, if any.
    pub(crate) fn vulkan(draw: Option<LogicDraw>) -> Option<vk::LogicOp> {
        match draw {
            Some(LogicDraw::Vulkan(op)) => Some(op),
            _ => None,
        }
    }

    /// The logic operation the fragment function computes, if any.
    pub(crate) fn shader(draw: Option<LogicDraw>) -> Option<vk::LogicOp> {
        match draw {
            Some(LogicDraw::Shader(op)) => Some(op),
            _ => None,
        }
    }
}

/// The pipeline `d` draws as on `p` under its private logic operation (the module docs), and how
/// it draws the operation: each normalized, sRGB and integer attachment unblended, under a no-op
/// also unwritten, a float attachment as it is. Copy and no-op are then Metal's on every device
/// and need no operation. Every other operation takes Vulkan's `logicOp`
/// ([`LogicDraw::Vulkan`]); on a device without it that orders framebuffer fetch, the fragment
/// function computes it ([`LogicDraw::Shader`]), refused `NotImplemented`
/// (`PrivateRenderField::LogicOperation`) at a normalized or integer attachment whose format it
/// does not serve (`lower::fragment`'s `logic`); on a device that has neither, it is refused naming
/// them. Every operation but copy and no-op is refused `NotImplemented` where an attachment is
/// sRGB, where a float attachment blends under Vulkan's operation, and where an attachment's
/// format was not measured. A pipeline without a fragment function writes no color, and takes no
/// operation.
pub(crate) fn logic_operation<'a>(
    d: &'a RenderPipelineDescriptor,
    p: &DeviceProfile,
) -> Result<(Cow<'a, RenderPipelineDescriptor>, Option<LogicDraw>), Error> {
    if !d.private.logic_operation_enabled || d.fragment.is_none() {
        return Ok((Cow::Borrowed(d), None));
    }
    let op = LOGIC_OPS[(d.private.logic_operation & 15) as usize];
    let blendless = matches!(op, vk::LogicOp::COPY | vk::LogicOp::NO_OP);
    let shader = !blendless && !p.fields().enabled.features.contains(&VkFeature::LogicOp);
    if shader
        && !(crate::lower::fragment::interlock_orders(p) || crate::lower::fragment::inputs_order(p))
    {
        return Err(Unsupported::needs_any_of(
            Feature::PrivateRenderState(PrivateRenderField::LogicOperation),
            vec![
                vec![Capability::Feature(VkFeature::LogicOp)],
                vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)],
                vec![Capability::Feature(
                    VkFeature::RasterizationOrderColorAttachmentAccess,
                )],
            ],
        )
        .into());
    }
    let mut out = d.clone();
    for (index, slot) in out.color_attachments.iter_mut().enumerate() {
        let Some(a) = slot
            .as_mut()
            .filter(|a| a.pixel_format != PixelFormat::Invalid)
        else {
            continue;
        };
        let refused = || {
            refused(PrivateRenderField::LogicOperation)
                .at_attachment(Attachment::Color(index as u32))
        };
        match (logic_class(a.pixel_format), blendless) {
            (LogicClass::Unmeasured, _) => return Err(refused()),
            (LogicClass::Float, true) => {}
            // The function leaves a float attachment to the device's blend.
            (LogicClass::Float, false) if a.blending_enabled && !shader => return Err(refused()),
            (LogicClass::Float, false) => {}
            (LogicClass::Srgb, false) => return Err(refused()),
            // A format the function's operation does not serve, and color 0 under alpha to
            // coverage, whose alpha the coverage reads before Metal's operation.
            (LogicClass::Codes, false)
                if shader
                    && (crate::lower::fragment::logic_channels(a.pixel_format).is_none()
                        || (index == 0 && d.alpha_to_coverage)) =>
            {
                return Err(refused());
            }
            (LogicClass::Codes, _) | (LogicClass::Srgb, true) => {
                a.blending_enabled = false;
                if op == vk::LogicOp::NO_OP {
                    a.write_mask = ColorWriteMask::empty();
                }
            }
        }
    }
    let draw = match (blendless, shader) {
        (true, _) => None,
        (false, true) => Some(LogicDraw::Shader(op)),
        (false, false) => Some(LogicDraw::Vulkan(op)),
    };
    Ok((Cow::Owned(out), draw))
}

/// `Ok` where the linked pipeline `g` draws its logic operation `draw` ([`logic_operation`]) as
/// Metal's. Vulkan's operation applies to the color attachments of the subpass alone, so a
/// pipeline whose fragment function writes an attachment itself (a storage attachment, a shadow
/// stencil, an imageblock, or a word it merges with its input attachment's) is refused
/// (`PrivateRenderField::LogicOperation`), Metal's answer there unmeasured. The operation the
/// fragment function computes must be the one it compiled under (`StageIo::logic_operation`,
/// else `NeedsInput(LogicOperation)`), and is refused beside a shadow stencil, an imageblock or a
/// merged word, which it does not compute over.
pub(crate) fn logic_operated(
    draw: Option<LogicDraw>,
    fragment: Option<&FunctionArtifact>,
    g: &GraphicsState,
) -> Result<(), Error> {
    let compiled = fragment.and_then(|f| f.interface.io.logic_operation);
    let Some(draw) = draw else {
        return match compiled {
            Some(_) => Err(Error::NeedsInput(RuntimeInput::LogicOperation)),
            None => Ok(()),
        };
    };
    let merges = fragment.is_some_and(|f| {
        f.facts.iter().any(|fact| {
            matches!(
                fact,
                crate::pipeline::facts::RuntimeFact::WordInputMerge { .. }
            )
        })
    });
    let beside = g.shadow_stencil.is_some() || g.fragment_imageblock.is_some() || merges;
    match draw {
        LogicDraw::Vulkan(_) if beside || !g.storage_attachments.is_empty() => {
            Err(refused(PrivateRenderField::LogicOperation))
        }
        LogicDraw::Vulkan(_) => match compiled {
            Some(_) => Err(Error::NeedsInput(RuntimeInput::LogicOperation)),
            None => Ok(()),
        },
        LogicDraw::Shader(_) if beside => Err(refused(PrivateRenderField::LogicOperation)),
        LogicDraw::Shader(op) => match compiled == Some(op) {
            true => Ok(()),
            false => Err(Error::NeedsInput(RuntimeInput::LogicOperation)),
        },
    }
}

/// Note Vulkan's operation (`PrecisionNote::NativeLogicOperation`) on the fragment stage among
/// `modules` of the linked pipeline `g`, of descriptor `d` as it draws its logic operation
/// ([`logic_operation`]), where Vulkan's `logicOp` ([`GraphicsState::logic_op`]) operates on an
/// attachment whose codes Metal's operation widens (`RGB10A2Unorm` and `BGR10A2Unorm`, on 16-bit
/// codes: `lower::fragment`'s `logic`): one a first output writes (`GraphicsState::color_outputs`)
/// with a channel its write mask keeps. Vulkan's operation on the 10-bit codes themselves is the
/// device's native form, and may store another code than Metal's.
pub(crate) fn note_vulkan_codes(
    d: &RenderPipelineDescriptor,
    g: &GraphicsState,
    modules: &mut [StageModule],
) {
    if g.logic_op.is_none() {
        return;
    }
    let widened = d.color_attachments.iter().enumerate().any(|(i, a)| {
        let written = g.color_outputs & 1u32.checked_shl(i as u32).unwrap_or(0) != 0
            && g.color_attachments
                .get(i)
                .and_then(Option::as_ref)
                .is_some_and(|s| !s.write_mask.is_empty());
        written
            && a.as_ref()
                .is_some_and(|a| crate::lower::fragment::logic_widens(a.pixel_format))
    });
    if !widened {
        return;
    }
    let note = PrecisionNote::NativeLogicOperation;
    for m in modules
        .iter_mut()
        .filter(|m| m.stage == vk::ShaderStage::FRAGMENT)
    {
        if !m.precision_notes.contains(&note) {
            m.precision_notes.push(note);
            m.precision_notes.sort_unstable();
        }
    }
}

/// The text the key of the render pipeline `d` appends for its private state: empty at Metal's
/// defaults and wherever only properties that change nothing differ, so every other pipeline
/// keeps its key. A logic operation's attachments are already in `d` as it draws them
/// ([`logic_operation`]), so only the operation is added, Vulkan's or the fragment function's:
/// copy and no-op key as the pipeline of those attachments.
pub(crate) fn render_key_fields(d: &RenderPipelineDescriptor) -> String {
    let mut fields = match applied_masks(d) {
        Some(m) if uniform(&m) => format!(" sample_mask {:#x}", m[0]),
        Some(m) => format!(
            " sample_masks {:#x} {:#x} {:#x} {:#x}",
            m[0], m[1], m[2], m[3]
        ),
        None => String::new(),
    };
    if let Ok(Some(function)) = alpha_test(&d.private) {
        fields.push_str(&format!(" alpha_test {}", function.as_raw()));
    }
    let op = LOGIC_OPS[(d.private.logic_operation & 15) as usize];
    let vulkan = !matches!(op, vk::LogicOp::COPY | vk::LogicOp::NO_OP);
    if d.private.logic_operation_enabled && d.fragment.is_some() && vulkan {
        fields.push_str(&format!(" logic_op {}", op.as_raw()));
    }
    fields
}
