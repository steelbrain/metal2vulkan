//! `GraphicsState` of a render pipeline.
//!
//! Every value is mapped from the descriptor through `metal::{format, state}`:
//!
//! - **Topology** is the list topology of the descriptor's topology class (`Unspecified` is the
//!   triangle class). With extended dynamic state it is dynamic within the class; a draw of
//!   another class (possible only for `Unspecified`) is a consumer-side pipeline variant.
//! - **Color attachments**: the format mapping for `ColorAttachment` (plus
//!   `ColorAttachmentBlend` when blending), the write mask, and the blend state; at most
//!   `maxColorAttachments` of them. A blend of factors one and zero, added, for color and alpha,
//!   of an attachment the fragment function rounds to Metal's codes (`StageIo::rounded_colors`), or
//!   writes as a half's bits (`StageIo::half_bit_colors`), takes no Vulkan blend: the function
//!   writes what Metal's blend stores, which a device's blend could still change (x86 lavapipe adds
//!   the clear's `+0` to a `-0` source of an `R16Float` attachment, Metal keeps `-0`). An
//!   attachment that no first (index 0) output of the fragment function names (every attachment of
//!   a pipeline without a fragment function) gets an empty write mask, so the draw leaves it
//!   unmodified. That is what Metal does where measured: an unblended attachment the function reads
//!   through framebuffer fetch but has no output for, which a draw's later fragments read as
//!   loaded; an attachment without an output whose blend would change it (`Zero`/`Zero`,
//!   `One`/`One`, a blend-color factor), and one named only by a dual-source second output
//!   `[[color(0), index(1)]]`, unblended or blended without a `Source1` factor. Apple documents
//!   that a nil fragment function disables rasterization into the color attachments. Under the
//!   descriptor's mask, Vulkan leaves such an attachment's values undefined in the older text of
//!   the fragment output interface (the current text says they must not be modified when no output
//!   is declared at the location, but a second output alone is declared there and leaves the first
//!   input of the blend undefined), and with no fragment shader undefined values may be written to
//!   every color attachment.
//! - **Identical attachment states without `independentBlend`**: Vulkan then requires every element
//!   of `pAttachments` to be identical
//!   (`VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605`). Where the states above
//!   differ, link gives every attachment one state when that is exact, and otherwise refuses
//!   `Unsupported{IndependentBlending, Feature(IndependentBlend)}`. The draw must keep the
//!   state of each attachment the fragment function has an output for whose mask is not empty,
//!   so those must agree, and that state is the one. An attachment the function has no output
//!   for, of either index (with a fragment function), is not modified whatever its state, under
//!   the current text
//!   of Vulkan's fragment output interface ("must not be modified", Vulkan 1.4.348), so it takes
//!   that state where its format can blend as the state does. An attachment with an empty mask
//!   writes nothing whatever its blend, so it takes that state only if the state's mask is empty
//!   too. With no attachment of the first kind, every attachment takes an empty mask without
//!   blending. On a device with `independentBlend` the states stay as above, and the pipeline's
//!   requirements name the feature when they differ.
//! - **Depth and stencil** formats for `DepthStencilAttachment`.
//! - **Dynamic states**: what Metal sets per encoder rather than per pipeline. Always viewport,
//!   scissor, depth bias, blend constants, and the stencil reference and masks; with extended
//!   dynamic state also cull mode, front face, topology, the depth/stencil test state, and the
//!   vertex strides of `PerBind` layouts.
//! - **Storage attachments**: an attachment the fragment stage reads in the interlocked storage
//!   form (`ResourceKind::TileAttachment`) has no color attachment state: the subpass leaves it
//!   out, and `link::fetch` states how the consumer binds it.
//! - **Input attachments**: one per framebuffer-fetch binding of the fragment stage. Metal orders
//!   each fetch after the earlier fragments of the draw that cover the pixel; Vulkan does only
//!   under rasterization-order attachment access, so a pipeline with input attachments on a profile
//!   with `rasterizationOrderColorAttachmentAccess` states `rasterization_order_attachment_access`
//!   (and the compiled pipeline requires the feature). Without the feature no fragment stage reads
//!   an input attachment (`lower::fragment::color` refuses it). In a multisampled pipeline
//!   (`rasterSampleCount` above 1) Metal runs a function that reads a color attachment once per
//!   covered sample, each invocation reading its own sample (measured), so the stage, compiled for
//!   the pipeline (`LoweringInputs::multisampled`), reads a multisampled `SubpassData` image (or,
//!   in the storage form, a multisampled storage image) at its `SampleId`, whose input runs it once
//!   per covered sample: it is a per-sample function under the sample invocations below. A fetching
//!   stage compiled for the other kind of pipeline (one sample against more, or the reverse)
//!   declares an image of the wrong sample count: `NeedsInput(Multisampled)`. A color input the
//!   function declares but never reads changes nothing at link: MSL says such a function *may* run
//!   per sample, and an Apple M3 Max ran a function with a device atomic once per covered sample
//!   under 4x multisampling with or without one.
//! - **Fragment invocations under multisampling**. With `rasterSampleCount` above 1 and
//!   rasterization on, Metal (Apple M3 Max, 4x, measured) runs a fragment function that writes
//!   memory once per covered sample, whatever its inputs, and every invocation reads `[[position]]`
//!   (x, y, z, and w), its center and centroid varyings, `[[point_coord]]`, and
//!   `[[barycentric_coord]]` as the pixel's, and `[[sample_mask]]` as the pixel's coverage; only
//!   `[[sample_id]]` and sample-qualified varyings are the sample's. Vulkan runs a fragment shader
//!   once per covered sample only under sample shading, which a `SampleId` input turns on and a
//!   `Sample` input may, and sample shading moves `FragCoord` to the sample, lets an undecorated or
//!   `Centroid` input interpolate anywhere in the sample's area, may move `PointCoord` there, and
//!   leaves in `SampleMask` only the invocation's samples. A fragment compiled for the pipeline
//!   (`LoweringInputs::multisampled`) reads `[[position]]`'s x and y as `floor(FragCoord.xy) + 0.5`
//!   and, where the device has interpolation functions whose `InterpolateAtOffset` reads from the
//!   pixel center, its center varyings through that instruction at offset 0, or, where the device
//!   reads them at the pixel center itself as Metal does (`VkProperty::MetalCenterVaryings`,
//!   MoltenVK), as they are: both the pixel center whatever the invocation shades
//!   (`StageIo::{multisampled, varyings_at_pixel, position_depth}`). A function that writes memory
//!   (`StageInterface::writes_memory`) runs under the pipeline's sample shading at a minimum of 1.0
//!   (`GraphicsState::sample_shading`, needing `sampleRateShading`), as one that reads
//!   `[[sample_id]]` does: once per covered sample, where a Vulkan stage without sample shading
//!   runs at least once per pixel. A function that runs per sample (one of those, or one that
//!   reads a sample-qualified varying) beside `[[sample_mask]]` cannot run under sample shading:
//!   Vulkan gives an invocation under sample shading only the bits of the samples it shades, a quad
//!   holds one sample index, and no subgroup is promised to hold a pixel's other sample
//!   invocations, so no form gathers the pixel's coverage. Compiled for the pipeline, such a
//!   function runs once per pixel instead, without sample shading, and its body once per covered
//!   sample from a loop over the coverage (`StageIo::coverage_loop`), its outputs the last pass's
//!   (`PrecisionNote::LoopedSampleOutputs`); compiled without `multisampled` it is
//!   `NeedsInput(Multisampled)`, and one the loop does not fit is refused for the reason its
//!   interface records (`CoverageLoopRefusal`). On a driver that runs a
//!   memory-writing function once per covered sample itself (`CoverageLoop::DriverRunsPerSample`,
//!   MoltenVK on Apple's GPU) the function runs as written, without sample shading, and its other
//!   reads are checked as below. A per-sample function reads the pixel values Vulkan has no pixel
//!   form for where Vulkan evaluates them, as a pixel-rate function does: `[[position]]`'s z and w
//!   (the interpolated depth and `1/w`, at a position Vulkan does not state), `[[point_coord]]`
//!   ("the fragment center, or the location of a sample"), `[[barycentric_coord]]`, and a centroid
//!   varying (at the invocation's sample under sample shading), each within half a pixel of the
//!   pixel center Metal reads it at; the compile layer notes `PrecisionNote::MultisamplePosition`
//!   for them, as for a pixel-rate function's. Link refuses a per-sample function's center varying
//!   on a device without the varyings' pixel form, naming what the device lacks:
//!   `VkProperty::InterpolateAtOffsetCentered` (a device whose `InterpolateAtOffset` does not read
//!   from the pixel center, as MoltenVK's does not, has no such form, and has a pixel form only
//!   where it reads the varying at the pixel center itself) and
//!   `shaderSampleRateInterpolationFunctions`; and a function that needs sample shading, or reads
//!   `[[position]]` or a center varying per sample, on a device without `sampleRateShading`. Such a
//!   function whose pixel reads all have forms on the device, but which was compiled without
//!   `multisampled`, is `NeedsInput(Multisampled)`: compiled for the pipeline it links. A function
//!   compiled for another device's forms is `Unsupported{SampleRateShading, NotImplemented}`. A
//!   function that runs per
//!   pixel, or a pipeline of one sample, has no sample shading: `FragCoord`'s x and y are the
//!   pixel's center and `SampleMask` the fragment's coverage. The sample mask a function returns
//!   keeps, under either API, the samples of the invocation whose bits it sets (measured with
//!   `[[sample_id]]` and with a device atomic).
//! - **Split pixel fragments**. A driver that rasterizes a triangle it clips as several
//!   (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`: lavapipe) runs a pixel-rate function
//!   once per piece in each pixel on the pieces' shared edges, with that piece's coverage, where
//!   Metal runs it once with the pixel's: its `[[sample_mask]]` holds part of the samples and a
//!   centroid varying moves to a covered sample. There link refuses a pipeline of more than one
//!   sample whose fragment function reads either (`Unsupported{PixelCoverage}` or
//!   `Unsupported{CentroidVarying}`, `Needs::DriverDefect`).
//! - **Fragment outputs**: a color output to an attachment the descriptor sets must have the
//!   attachment's numeric class (`float` to a float, normalized, or sRGB format, `uint` to an
//!   unsigned-integer one, `int` to a signed-integer one); an output of another class is Metal's
//!   pipeline error or a fragment compiled for other attachments (`ColorAttachment`). An integer
//!   output is declared `int` exactly when the fragment compiled under its attachment in
//!   `LoweringInputs::signed_color_attachments`, so a `uint` output to a signed-integer attachment
//!   is a stage compiled for another pipeline. An output to an absent attachment is discarded, as
//!   in Metal. An output (of either index) with fewer components than the attachment's format has
//!   channels (`metal::format::color_output_components`) is Metal's pipeline error, where Vulkan
//!   would write undefined values to the missing ones (`ColorAttachment`).
//! - **Dual-source blending**: an attachment whose blend reads a `Source1` factor needs the
//!   fragment function's `[[color(n), index(1)]]` output and its first output (Metal refuses the
//!   blend without the first: `ColorAttachment`), an attachment index below
//!   `maxFragmentDualSrcAttachments`, and at most that many attachments written.
//! - **Alpha to coverage** reads the alpha of the first output at attachment 0. Without a
//!   four-component float output there (none, fewer components, or an integer one) Metal's
//!   coverage stays the rasterizer's, measured at 1 and 4 samples, while Vulkan requires an alpha
//!   output at location 0 (`VUID-VkGraphicsPipelineCreateInfo-alphaToCoverageEnable-08891`), so
//!   link turns it off. Over an integer attachment 0 it is Metal's pipeline error
//!   (`ColorAttachment`). Over an `A8Unorm` attachment 0 whose image holds Metal's alpha in red
//!   (`R8_UNORM`, no `A8_UNORM`), the output holds Metal's alpha in its alpha lane too
//!   (`StageIo::alpha_in_red_colors`), so Vulkan's coverage, which reads the output's alpha
//!   whatever the format keeps (the specification's multisample coverage section: "All alpha
//!   values in this section refer only to the alpha component of the fragment shader output that
//!   has a `Location` and `Index` decoration of zero"), reads Metal's alpha;
//!   over an `ARGB8Unorm` one, whose image's alpha is Metal's blue, it is
//!   `NotImplemented(Format)`.
//! - **Alpha to one** is `VkPipelineMultisampleStateCreateInfo::alphaToOneEnable`
//!   (`GraphicsState::alpha_to_one`), the exact counterpart of Metal's `alphaToOneEnabled`: both
//!   replace the fragment's alpha with one before the blend reads it and before the write mask
//!   decides what is stored (measured on an Apple M3 Max: under source-alpha factors the stored
//!   color is the source alone, and an attachment mask without alpha still keeps the one out of
//!   memory while the blend reads it), and both leave alpha to coverage reading the alpha the
//!   function wrote. Vulkan gates it behind the `alphaToOne` device feature
//!   (`VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785`), so the pipeline's
//!   requirements name the feature (`GraphicsState::multisample_features`) and a device without
//!   it is refused `Unsupported{AlphaToOne, Feature(AlphaToOne)}`: a device gap, never a refusal
//!   of the field itself.
//! - **Vertex amplification**: `maxVertexAmplificationCount` above 1 makes a pipeline whose draws
//!   may be multiview, which needs `multiview`; its draws of more amplifications than the device's
//!   `maxMultiviewViewCount` are draws of one amplification each. A count of 0 is an invalid
//!   descriptor. `GraphicsState::declares_layer` states whether a stage declares `Layer` (a vertex
//!   function that chooses its render target slice, or a fragment function that reads it), which a
//!   multiview pipeline cannot (`VUID-VkGraphicsPipelineCreateInfo-renderPass-06050`, `-06059`):
//!   the binder refuses such a pipeline's amplified draws, and its draws at a count of 1 are
//!   ordinary layered rendering.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::requirements::{FormatUsage, VkFeature, VkLimit, VkProperty};
use crate::device::{DeviceProfile, DriverWorkaround};
use crate::error::{
    Attachment, Capability, DescriptorError, Error, Feature, RuntimeInput, Unsupported,
};
use crate::metal;
use crate::metal::format::{
    color_output_components, sampled_type, vk_format, ExtendedRange, FormatEmulation,
};
use crate::metal::state::{
    blend_factors, check_constant_alpha, color_write_mask, constant_alpha_withheld,
};
use crate::metal::PrimitiveTopologyClass;
use crate::pipeline::color_writes;
use crate::pipeline::graphics::{
    ColorAttachmentState, GraphicsState, InputAttachmentUse, TessellationState,
};
use crate::pipeline::interface::StageInterface;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::rotated_blend::{alpha_in_red_blend, rotated_blend, RotatedBlend};
use crate::pipeline::stage_io::{
    CoverageLoop, Interpolation, IoNumeric, IoRole, Sampling, StageBuiltin, StageIo,
};
use crate::pipeline::sym::{ResourceKind, SampledType, Stage};
use crate::pipeline::vertex::{VertexInputState, VertexStride};
use crate::vk;

/// The sample-count flag of a raster sample count.
fn samples(count: u32) -> Result<vk::SampleCountFlags, Error> {
    use vk::SampleCountFlags as S;
    Ok(match count {
        1 => S::COUNT_1,
        2 => S::COUNT_2,
        4 => S::COUNT_4,
        8 => S::COUNT_8,
        16 => S::COUNT_16,
        32 => S::COUNT_32,
        64 => S::COUNT_64,
        value => {
            return Err(Error::InvalidDescriptor(
                DescriptorError::UnknownEnumValue {
                    ty: "rasterSampleCount",
                    value,
                },
            ))
        }
    })
}

fn topology(class: PrimitiveTopologyClass) -> vk::PrimitiveTopology {
    match class {
        PrimitiveTopologyClass::Point => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopologyClass::Line => vk::PrimitiveTopology::LINE_LIST,
        _ => vk::PrimitiveTopology::TRIANGLE_LIST,
    }
}

/// Check the color group of render inputs `r` (`RenderLinkInputs::color_group`) with vertex stage
/// `vs`, fragment stage `fs`, a tessellation pipeline where `tessellated`, on `p`: the fragment
/// stage compiled for the variant's own group (`NeedsInput(ColorWrites)` for one compiled for
/// another, or for none), and, where the pipeline takes more than one group, neither stage writes
/// memory and the pipeline is no tessellation pipeline, which each group's draw would run again
/// (the limit refusal, `compile::color_group`).
fn color_group(
    r: &RenderLinkInputs,
    (vs, fs): (&StageInterface, Option<&StageInterface>),
    tessellated: bool,
    p: &DeviceProfile,
) -> Result<(), Error> {
    if fs.is_some_and(|fs| fs.io.color_group != r.color_group) {
        return Err(Error::NeedsInput(RuntimeInput::ColorWrites));
    }
    let Some(g) = r.color_group.as_ref().filter(|g| g.groups() > 1) else {
        return Ok(());
    };
    if tessellated || vs.writes_memory || fs.is_some_and(|fs| fs.writes_memory) {
        return Err(crate::pipeline::compile::color_group::refusal(g, p));
    }
    Ok(())
}

/// The vertex amplification count of render inputs `r` (see the module docs).
fn vertex_amplification(r: &RenderLinkInputs, p: &DeviceProfile) -> Result<u32, Error> {
    let count = r.max_vertex_amplification_count;
    if count == 0 {
        return Err(Error::InvalidDescriptor(
            DescriptorError::VertexAmplificationCount { count },
        ));
    }
    if count == 1 {
        return Ok(count);
    }
    let f = p.fields();
    if !f.enabled.features.contains(&VkFeature::Multiview) {
        return Err(Unsupported::needs(
            Feature::VertexAmplification,
            Capability::Feature(VkFeature::Multiview),
        )
        .into());
    }
    // More amplifications than the device has views draw one amplification at a time.
    Ok(count)
}

/// The graphics state of render inputs `r` with vertex stage `vs`, fragment stage `fs`, vertex
/// input `vi`, and, for a tessellation pipeline, its tessellation state `tessellation`: its
/// topology is `PATCH_LIST` and never dynamic state, as its draws draw patches only.
/// `input_blends`: the extended-range attachments whose blend the fragment function computes over
/// their input attachments, and the channels each write keeps (`link::fetch::input_blends`).
pub fn graphics_state(
    r: &RenderLinkInputs,
    (vs, fs): (&StageInterface, Option<&StageInterface>),
    vi: &VertexInputState,
    tessellation: Option<TessellationState>,
    input_blends: &BTreeMap<u32, vk::ColorComponentFlags>,
    p: &DeviceProfile,
) -> Result<GraphicsState, Error> {
    if r.alpha_to_one && !p.fields().enabled.features.contains(&VkFeature::AlphaToOne) {
        return Err(Unsupported::needs(
            Feature::AlphaToOne,
            Capability::Feature(VkFeature::AlphaToOne),
        )
        .into());
    }
    color_group(r, (vs, fs), tessellation.is_some(), p)?;
    let max_vertex_amplification_count = vertex_amplification(r, p)?;
    // A vertex function that selects its slice through the synthesized geometry stage
    // (`IoRole::SelectedLayer`) declares `Layer` there, and a fragment function that reads it as 0
    // (`StageIo::zero_indices`) reads it as Metal does only where no amplification moves it.
    let declares_layer = std::iter::once(vs).chain(fs).any(|s| {
        s.io.builtins.contains(&StageBuiltin::Layer)
            || s.io.zero_indices.contains(&StageBuiltin::Layer)
            || s.io.outputs.iter().any(|o| o.role == IoRole::SelectedLayer)
    });
    let limits = &p.fields().limits;
    let used = r
        .color_attachments
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |i| i + 1) as u64;
    if used > u64::from(limits.max_color_attachments) {
        return Err(Unsupported::needs(
            Feature::InterfaceComponents,
            Capability::Limit {
                limit: VkLimit::MaxColorAttachments,
                required: used,
                available: u64::from(limits.max_color_attachments),
            },
        )
        .into());
    }
    // The attachments the fragment function writes, through a first (index 0) output; the
    // others keep their contents (see the module docs). `declared` adds those a dual-source second
    // output alone names, which Vulkan's fragment output interface counts as declared.
    let colors = |second: bool| -> BTreeSet<u32> {
        fs.into_iter()
            .flat_map(|f| &f.io.outputs)
            .filter_map(|o| match o.role {
                IoRole::Color { attachment, index } if second || index == 0 => Some(attachment),
                _ => None,
            })
            .collect()
    };
    let written = colors(false);
    let declared = colors(true);
    // An attachment the function reads in the interlocked storage form is no color attachment of
    // the subpass (`link::fetch`).
    let storage: BTreeSet<u32> = fs
        .into_iter()
        .flat_map(|f| &f.descriptors)
        .filter_map(|d| match d.slot.kind {
            ResourceKind::TileAttachment(color) => Some(color),
            _ => None,
        })
        .collect();
    // The rule word of each attachment whose rules the fragment function selects at the draw.
    let color_write_rules = match fs {
        Some(f) => color_writes::rule_words(
            &f.io,
            &r.color_attachments,
            (r.alpha_to_coverage, r.texture_write_rounding_mode),
            p,
        )?,
        None => Vec::new(),
    };
    let selected_rounds = |index: u32| {
        let word = color_write_rules.iter().find(|(a, _)| *a == index);
        fs.and_then(|f| f.io.selected_colors.get(&index))
            .zip(word)
            .is_some_and(|(sel, (_, word))| color_writes::selected_rounds(sel, *word))
    };
    let mut color_attachments = Vec::with_capacity(used as usize);
    for (index, a) in r.color_attachments.iter().take(used as usize).enumerate() {
        color_attachments.push(match a {
            None => None,
            Some(_) if storage.contains(&(index as u32)) => None,
            Some(a) => {
                let state = || -> Result<ColorAttachmentState, Error> {
                    // A word attachment renders through the view of its word, which the function
                    // writes whole and unblended (`color_writes::word_color`): its one component,
                    // where the function writes the attachment and the mask keeps some of the
                    // format's channels (all of them, or some where the function merges the word
                    // into its input attachment's, `fetch::word_merges`).
                    if let Some(word) =
                        color_writes::word_color(index as u32, a, r.alpha_to_coverage, p)
                    {
                        let channels = [
                            vk::ColorComponentFlags::R,
                            vk::ColorComponentFlags::G,
                            vk::ColorComponentFlags::B,
                            vk::ColorComponentFlags::A,
                        ]
                        .into_iter()
                        .zip(word.fields())
                        .filter(|(_, f)| f.is_some())
                        .fold(vk::ColorComponentFlags::empty(), |m, (c, _)| m | c);
                        let kept = color_write_mask(a.write_mask)? & channels;
                        let write_mask = match written.contains(&(index as u32)) && !kept.is_empty()
                        {
                            true => vk::ColorComponentFlags::R,
                            false => vk::ColorComponentFlags::empty(),
                        };
                        return Ok(ColorAttachmentState {
                            format: word.word_format(),
                            write_mask,
                            blend: None,
                        });
                    }
                    // The fragment function writes what Metal's blend of one and zero stores,
                    // so the attachment takes no Vulkan blend, nor the format's blend support:
                    // one it rounds, and one it writes as a half's bits. But not a 16-bit float
                    // attachment's from a module without fast math, whose blend Metal computes
                    // keeping the destination's product by zero: that one blends, announced
                    // (`color_writes::keeps_zero`).
                    let rounded = fs.is_some_and(|f| {
                        f.io.rounded_colors.contains(&(index as u32))
                            || f.io.half_bit_colors.contains(&(index as u32))
                    }) || selected_rounds(index as u32);
                    let keeps_zero =
                        fs.is_some_and(|f| color_writes::keeps_zero(&f.io, index as u32, a));
                    let replaced = rounded && color_writes::replaces(a) && !keeps_zero;
                    // An extended-range attachment whose blend the function computes over its
                    // input attachment takes no Vulkan blend, and the function's write stores
                    // every raw component (none where it keeps every channel).
                    let input_blend = input_blends.get(&(index as u32)).copied();
                    // An attachment held with its channels rotated takes the function's rotated
                    // output, the descriptor's mask rotated, and the blend `rotated_blend`
                    // finds Metal's in Vulkan's factors; one holding Metal's alpha in red
                    // (`A8Unorm` in `R8_UNORM`) the rotated output too, whose red is Metal's
                    // alpha, the mask's alpha bit in red, and Metal's alpha blend in red.
                    let alpha_red = metal::format::alpha_in_red(a.pixel_format, p);
                    let rotated = metal::format::rotated_channels(a.pixel_format) || alpha_red;
                    let blended = a.blending_enabled && !replaced && input_blend.is_none();
                    let mut usages = BTreeSet::from([FormatUsage::ColorAttachment]);
                    if blended {
                        usages.insert(FormatUsage::ColorAttachmentBlend);
                    }
                    let descriptor_mask = color_write_mask(a.write_mask)?;
                    let descriptor_mask = match (alpha_red, rotated) {
                        (true, _) => rotated_mask(descriptor_mask) & vk::ColorComponentFlags::R,
                        (false, true) => rotated_mask(descriptor_mask),
                        (false, false) => descriptor_mask,
                    };
                    // An extended-range attachment holds its raw texels, whose components the
                    // function writes whole.
                    let descriptor_mask = match (ExtendedRange::of(a.pixel_format), input_blend) {
                        (Some(form), Some(kept)) => {
                            let stored = match kept == vk::ColorComponentFlags::known_bits() {
                                true => vk::ColorComponentFlags::empty(),
                                false => vk::ColorComponentFlags::known_bits(),
                            };
                            extended_write_mask(form, stored)?
                        }
                        (Some(form), None) => extended_write_mask(form, descriptor_mask)?,
                        (None, _) => descriptor_mask,
                    };
                    let write_mask = if written.contains(&(index as u32)) {
                        descriptor_mask
                    } else {
                        vk::ColorComponentFlags::empty()
                    };
                    let blend = match (replaced || input_blend.is_some(), rotated) {
                        (true, _) => None,
                        (false, false) => blend_factors(a, p)?,
                        (false, true) if alpha_red => alpha_in_red_blend(a, p)?,
                        // The second output the rotation needs is the function's only beside
                        // a first output to the attachment; without one the draw writes nothing
                        // there whatever the blend (its mask is empty).
                        (false, true) => match rotated_blend(a, p)? {
                            Some(RotatedBlend::Direct(state)) => Some(state),
                            Some(RotatedBlend::Synthesized(state, _))
                                if written.contains(&(index as u32)) =>
                            {
                                Some(state)
                            }
                            Some(RotatedBlend::Synthesized(..)) | None => None,
                        },
                    };
                    let mapping = vk_format(a.pixel_format, &usages, p)?;
                    // The format holds Metal's alpha in red for these usages exactly where the
                    // function wrote it there (`alpha_in_red`, of the attachment usage alone).
                    if (mapping.emulation == FormatEmulation::AlphaInRed) != alpha_red {
                        return Err(Unsupported::not_implemented(Feature::Format).into());
                    }
                    let format = mapping.format;
                    Ok(ColorAttachmentState {
                        format,
                        write_mask,
                        blend,
                    })
                };
                Some(state().map_err(|e| e.at_attachment(Attachment::Color(index as u32)))?)
            }
        });
    }
    if let Some(fs) = fs {
        check_outputs(r, fs, &color_attachments, p)?;
    }
    // Under an empty mask the blend writes nothing, so a constant-alpha factor the portability
    // subset withholds is dropped, not refused: after the descriptor checks, which read the blend
    // Metal was given.
    for s in color_attachments.iter_mut().flatten() {
        if s.write_mask.is_empty() && s.blend.is_some_and(|b| constant_alpha_withheld(&b, p)) {
            s.blend = None;
        }
    }
    if !p.fields().fragment.independent_blend {
        uniform_blend(&mut color_attachments, &declared, fs.is_some(), p)?;
    }
    for (index, s) in color_attachments.iter().enumerate() {
        if let Some(b) = s.as_ref().and_then(|s| s.blend.as_ref()) {
            check_constant_alpha(b, p)
                .map_err(|e| e.at_attachment(Attachment::Color(index as u32)))?;
        }
    }
    let alpha_to_coverage = reads_alpha_to_coverage(r, fs)?;
    let ds = BTreeSet::from([FormatUsage::DepthStencilAttachment]);
    let depth_format = match r.depth_format {
        Some(pf) => Some(
            vk_format(pf, &ds, p)
                .map_err(|e| e.at_attachment(Attachment::Depth))?
                .format,
        ),
        None => None,
    };
    let stencil_format = match r.stencil_format {
        Some(pf) => Some(
            vk_format(pf, &ds, p)
                .map_err(|e| e.at_attachment(Attachment::Stencil))?
                .format,
        ),
        None => None,
    };
    let mut input_attachments = Vec::new();
    for d in fs.map(|f| f.descriptors.as_slice()).unwrap_or_default() {
        if let ResourceKind::InputAttachment(color) = d.slot.kind {
            let attachment = r
                .color_attachments
                .get(color as usize)
                .copied()
                .flatten()
                .ok_or(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                    index: color,
                }))?;
            // An input attachment of an image holding Metal's channels rotated would read them
            // so: not modelled. (One holding Metal's alpha in red is read as Metal's alpha by a
            // stage compiled for it, `StageIo::alpha_in_red_colors`, which
            // `color_writes::check` holds to the pipeline's.)
            // Nor is a two-plane YCbCr attachment's, whose texels Metal holds as halves and the
            // attachment as the function's floats.
            if metal::format::rotated_channels(attachment.pixel_format)
                || metal::format::ycbcr_renders(attachment.pixel_format)
                || metal::format::ycbcr_pass_refused(attachment.pixel_format)
            {
                let refused = Unsupported::not_implemented(Feature::Format);
                return Err(refused.at_attachment(Attachment::Color(color)).into());
            }
            // An extended-range attachment's input reads its raw texels, the raw format's
            // `uint`s, which a stage compiled for the attachment decodes (`color_writes::check`);
            // a word attachment's reads its word, which a stage merges (`fetch::word_merges`).
            let word =
                color_writes::word_color(color, &attachment, r.alpha_to_coverage, p).is_some();
            let format_class = match ExtendedRange::of(attachment.pixel_format) {
                Some(_) => SampledType::Uint,
                None if word => SampledType::Uint,
                None => sampled_type(attachment.pixel_format).unwrap_or(SampledType::Float),
            };
            input_attachments.push(InputAttachmentUse {
                color_index: color,
                input_attachment_index: color,
                format_class,
            });
        }
    }
    let rasterization_samples = samples(r.raster_sample_count)?;
    // A fetched attachment is read (and, in the storage form, written) at the invocation's own
    // sample by a stage compiled for a multisampled pipeline, and at the pixel by one compiled
    // for one sample (see the module docs): a stage compiled for the other kind declares an image
    // of the wrong sample count.
    if let Some(fs) = fs {
        let fetches = !input_attachments.is_empty() || !storage.is_empty();
        if fetches && fs.io.multisampled != (r.raster_sample_count > 1) {
            return Err(Error::NeedsInput(RuntimeInput::Multisampled));
        }
    }
    let sample_shading = sample_invocations(r, fs, p)?;
    use vk::DynamicState as D;
    let mut dynamic_states = vec![
        D::VIEWPORT,
        D::SCISSOR,
        D::DEPTH_BIAS,
        D::BLEND_CONSTANTS,
        D::STENCIL_COMPARE_MASK,
        D::STENCIL_WRITE_MASK,
        D::STENCIL_REFERENCE,
    ];
    // `dynamic_vertex_stride` is extended dynamic state (Vulkan 1.3 or the EXT feature), which
    // brings every state below.
    if p.fields().dynamic_vertex_stride {
        // The topology stays static where the restart cannot follow it: Metal restarts every strip
        // at the all-ones index and never a list, so a pipeline whose topology a draw could switch
        // between a list and a strip needs its `primitiveRestartEnable` to follow, which a static
        // input assembly cannot. Where the device also has `extendedDynamicState2` (core in Vulkan
        // 1.3 without a feature, else `VK_EXT_extended_dynamic_state2`'s), which makes the restart
        // dynamic, a vertex pipeline makes both dynamic and serves every topology of its class: the
        // consumer creates it with its class's list topology and records each draw's topology
        // and restart (`metal::state::input_assembly`). Elsewhere the consumer creates the
        // pipeline with the draw's topology and restart. A mesh pipeline has no input assembly,
        // and a tessellation pipeline's patch list is a class of its own.
        if p.fields()
            .enabled
            .features
            .contains(&VkFeature::ExtendedDynamicState2)
            && vs.stage == Stage::Vertex
            && tessellation.is_none()
        {
            dynamic_states.extend([D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE]);
        }
        dynamic_states.extend([
            D::CULL_MODE,
            D::FRONT_FACE,
            D::DEPTH_TEST_ENABLE,
            D::DEPTH_WRITE_ENABLE,
            D::DEPTH_COMPARE_OP,
            D::STENCIL_TEST_ENABLE,
            D::STENCIL_OP,
        ]);
        let strides = vi.bindings.iter().map(|b| match b.stride {
            VertexStride::Static(s) => Some(s),
            VertexStride::PerBind => None,
        });
        if super::vertex::strides_dynamic(p, strides) {
            dynamic_states.push(D::VERTEX_INPUT_BINDING_STRIDE);
        }
    }
    // A fragment stage that binds a snapshot reads through it, drawn twice with its color write
    // masks as dynamic state (`GraphicsState::snapshot_reads`).
    let snapshot_reads = fs
        .into_iter()
        .flat_map(|f| &f.descriptors)
        .any(|d| matches!(d.slot.kind, ResourceKind::SnapshotTexels(_)));
    if snapshot_reads {
        dynamic_states.push(D::COLOR_WRITE_MASK_EXT);
    }
    let state = GraphicsState {
        topology: match tessellation {
            Some(_) => vk::PrimitiveTopology::PATCH_LIST,
            None => topology(r.input_primitive_topology),
        },
        color_attachments,
        depth_format,
        stencil_format,
        rasterization_samples,
        alpha_to_coverage,
        alpha_to_one: r.alpha_to_one,
        rasterizer_discard: !r.rasterization_enabled,
        dynamic_states,
        // Ordered wherever the device can order the reads.
        rasterization_order_attachment_access: !input_attachments.is_empty()
            && p.fields().fragment.rasterization_order_color_access,
        input_attachments,
        max_vertex_amplification_count,
        declares_layer,
        tessellation,
        // The fragment stage's facts decide it (`link::imageblock`).
        fragment_imageblock: None,
        // The fragment stage's facts decide them (`link::fetch`).
        storage_attachments: Vec::new(),
        shadow_stencil: None,
        sample_shading,
        vertex_writes_memory: vs.writes_memory,
        // Stored as the builtin, through the synthesized geometry stages, or (on a device with
        // one viewport) nowhere: the raw varying is written in every case.
        writes_viewport_index: vs.io.builtins.contains(&StageBuiltin::ViewportIndex)
            || vs
                .io
                .outputs
                .iter()
                .any(|o| matches!(o.role, IoRole::SelectedViewport | IoRole::ViewportIndex)),
        early_fragment_tests: fs.is_some_and(|fs| fs.io.early_fragment_tests),
        // An emulated mesh function's pipeline states its records (`link`).
        mesh_records: None,
        emulated_tessellation: None,
        writes_point_size: vs.writes_point_size(),
        flat_fragment_inputs: fs.is_some_and(|fs| reads_flat_inputs(&fs.io)),
        clip_distances: vs
            .io
            .builtins
            .iter()
            .find_map(|b| match b {
                StageBuiltin::ClipDistance { count } => Some(*count),
                _ => None,
            })
            .unwrap_or(0),
        writes_depth: fs.is_some_and(|fs| {
            fs.io
                .builtins
                .iter()
                .any(|b| matches!(b, StageBuiltin::FragDepth(_)))
        }),
        reads_barycentric_coord: fs.is_some_and(|fs| reads_barycentric_coord(&fs.io)),
        reads_primitive_id: fs.is_some_and(|fs| reads_primitive_id(&fs.io)),
        reads_position_depth: fs.is_some_and(|fs| {
            fs.io.builtins.contains(&StageBuiltin::FragCoord) && fs.io.position_depth
        }),
        fragment_writes_memory: fs.is_some_and(|fs| fs.writes_memory),
        // The storage form's writes join the outputs' once `link::fetch` decided them.
        color_outputs: 0,
        sample_mask: None,
        // The fragment stage's facts decide it (`link::shader_tests`).
        shader_tests: None,
        color_write_rules,
        snapshot_reads,
        color_group: r.color_group.clone(),
        // The pipeline's private logic operation, once compile decided it
        // (`private_state::logic_operation`).
        logic_op: None,
    };
    // The sample count against the attachments' sample counts.
    state.sample_count_properties(&p.fields().enabled.properties)?;
    split_pixel_fragments(r, fs, p, state.sample_shading)?;
    Ok(state)
}

/// Whether a fragment stage of interface `io` reads `[[barycentric_coord]]`
/// (`GraphicsState::reads_barycentric_coord`): either `BaryCoord` builtin, or either form's
/// varying (`IoRole::Barycentric`), perspective-correct or not.
fn reads_barycentric_coord(io: &StageIo) -> bool {
    io.builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::BaryCoord { .. }))
        || io
            .inputs
            .iter()
            .any(|v| matches!(v.role, IoRole::Barycentric { .. }))
}

/// Whether a fragment stage of interface `io` reads `[[primitive_id]]`
/// (`GraphicsState::reads_primitive_id`): the `PrimitiveId` builtin, or its varying
/// (`IoRole::PrimitiveId`).
fn reads_primitive_id(io: &StageIo) -> bool {
    io.builtins.contains(&StageBuiltin::PrimitiveId)
        || io.inputs.iter().any(|v| v.role == IoRole::PrimitiveId)
}

/// Whether a fragment stage of interface `io` reads a flat input
/// (`GraphicsState::flat_fragment_inputs`): a flat user varying, the render target array or
/// viewport index a vertex function wrote, as an input of its own role or as its builtin, or the
/// primitive ID's varying.
fn reads_flat_inputs(io: &StageIo) -> bool {
    let flat = io.inputs.iter().any(|v| {
        matches!(
            v.role,
            IoRole::Varying {
                interpolation: Some(Interpolation::Flat),
                ..
            } | IoRole::ViewportIndex
                | IoRole::RenderTargetArrayIndex
                | IoRole::PrimitiveId
        )
    });
    flat || io
        .builtins
        .iter()
        .any(|b| matches!(b, StageBuiltin::Layer | StageBuiltin::ViewportIndex))
}

/// Refuse what a fragment stage `fs` of a multisampled pipeline reads of its coverage on a
/// driver that rasterizes a clipped triangle as several
/// ([`DriverWorkaround::ClippedTrianglesSplitPixelFragments`]): `[[sample_mask]]` and a centroid
/// varying, each of which such a split changes in the pixels on the pieces' shared edges.
/// `[[sample_mask]]` is read once per pixel here: `sample_invocations` refused it beside a function
/// that runs per sample under sample shading. A function sample shading surely runs per sample
/// (the pipeline's, `sample_shading`, or the one a `SampleId` input enables; a `Sample` input only
/// may enable it) reads a centroid varying at its own sample, which lies in one piece: no split
/// changes it.
fn split_pixel_fragments(
    r: &RenderLinkInputs,
    fs: Option<&StageInterface>,
    p: &DeviceProfile,
    sample_shading: bool,
) -> Result<(), Error> {
    let defect = DriverWorkaround::ClippedTrianglesSplitPixelFragments;
    let Some(fs) = fs else {
        return Ok(());
    };
    if r.raster_sample_count <= 1 || !p.fields().workarounds.contains(&defect) {
        return Ok(());
    }
    let io = &fs.io;
    if io.builtins.contains(&StageBuiltin::SampleMaskIn) {
        return Err(Unsupported::driver_defect(Feature::PixelCoverage, defect).into());
    }
    if sample_shading || io.builtins.contains(&StageBuiltin::SampleId) {
        return Ok(());
    }
    let centroid = io.inputs.iter().any(|v| {
        matches!(
            v.role,
            IoRole::Varying {
                interpolation: Some(Interpolation::Smooth {
                    sampling: Sampling::Centroid,
                    ..
                }),
                ..
            }
        )
    });
    if centroid {
        return Err(Unsupported::driver_defect(Feature::CentroidVarying, defect).into());
    }
    Ok(())
}

/// Check how the fragment stage `fs` runs and reads under multisampling (see the module docs),
/// returning whether the pipeline enables sample shading at a minimum of 1.0
/// (`GraphicsState::sample_shading`). With more than one sample and rasterization on, a function
/// that writes memory runs per covered sample under sample shading, which needs
/// `sampleRateShading`; a function that runs per sample (that one, or one that reads
/// `[[sample_id]]`, as a per-sample framebuffer fetch's stage does, or a sample-qualified varying)
/// reads `[[position]]`'s x and y and its center varyings at the pixel through their pixel forms,
/// is refused for a center varying where the device has no pixel form, and needs `multisampled`
/// where it has one but was compiled without it. Its other pixel reads (`[[position]]`'s z and w,
/// `[[point_coord]]`, `[[barycentric_coord]]`, a centroid varying) read where Vulkan evaluates
/// them, noted by the compile layer.
fn sample_invocations(
    r: &RenderLinkInputs,
    fs: Option<&StageInterface>,
    p: &DeviceProfile,
) -> Result<bool, Error> {
    // A pipeline has a fragment stage only with rasterization on.
    let Some(fs) = fs else {
        return Ok(false);
    };
    if r.raster_sample_count <= 1 {
        return Ok(false);
    }
    let io = &fs.io;
    // The barycentric weights' varyings are center varyings.
    let interpolated = |at: Sampling| {
        io.inputs.iter().any(|v| match v.role {
            IoRole::Varying {
                interpolation: Some(Interpolation::Smooth { sampling, .. }),
                ..
            } => sampling == at,
            IoRole::Barycentric { .. } => at == Sampling::Center,
            _ => false,
        })
    };
    let rate = p.fields().fragment.sample_rate_shading;
    let refused = || -> Error {
        if rate {
            Unsupported::not_implemented(Feature::SampleRateShading)
        } else {
            Unsupported::needs(
                Feature::SampleRateShading,
                Capability::Feature(VkFeature::SampleRateShading),
            )
        }
        .into()
    };
    // Metal repeats the side effects per covered sample; Vulkan does under sample shading at a
    // minimum of 1.0, which the pipeline enables (a `SampleId` input enables it too).
    let sample_shading = fs.writes_memory;
    let per_sample = sample_shading
        || io.builtins.contains(&StageBuiltin::SampleId)
        || interpolated(Sampling::Sample);
    if !per_sample {
        return Ok(false);
    }
    // Metal hands each of the pixel's sample invocations the pixel's coverage. A Vulkan
    // invocation under sample shading sees only its own samples' bits, and nothing gathers a
    // pixel's sample invocations: a quad holds one sample index, and no subgroup is promised to
    // hold the rest. The function runs once per pixel instead, compiled for the pipeline, and its
    // body once per covered sample, without sample shading; one the loop does not fit is refused
    // for the reason its interface records. A driver that runs a
    // memory-writing function once per covered sample whatever the pipeline's sample shading hands
    // each run the pixel's coverage itself, as Metal does: the function's other reads are checked
    // as any per-sample function's, and the pipeline enables no sample shading, under which that
    // driver would mask the coverage.
    if io.builtins.contains(&StageBuiltin::SampleMaskIn) {
        match io.coverage_loop {
            CoverageLoop::Looped => return Ok(false),
            CoverageLoop::Unlooped => return Err(Error::NeedsInput(RuntimeInput::Multisampled)),
            CoverageLoop::Refused(r) => return Err(r.error()),
            CoverageLoop::NotNeeded => {
                return Err(
                    Unsupported::not_representable(Feature::PixelCoverageAtSampleRate).into(),
                )
            }
            CoverageLoop::DriverRunsPerSample => {}
        }
    }
    let sample_shading = sample_shading && io.coverage_loop != CoverageLoop::DriverRunsPerSample;
    if sample_shading && !rate {
        return Err(refused());
    }
    // Compiled for a multisampled pipeline, the function reads `[[position]]`'s x and y and
    // (with the pixel form) its center varyings at the pixel. `[[position]]`'s z and w,
    // `[[point_coord]]`, `[[barycentric_coord]]` and a centroid varying have no pixel form under
    // sample shading: they read where Vulkan evaluates them, within the pixel, as a pixel-rate
    // function's do, and the compile layer notes `PrecisionNote::MultisamplePosition` for them.
    let position = io.builtins.contains(&StageBuiltin::FragCoord);
    let center = interpolated(Sampling::Center);
    if !position && !center {
        return Ok(sample_shading);
    }
    if !rate {
        return Err(refused());
    }
    let enabled = &p.fields().enabled;
    // A device that reads center varyings at the pixel center itself, as Metal does, has their
    // pixel form: their read as it is.
    if center
        && !io.varyings_at_pixel
        && !enabled
            .properties
            .contains(&VkProperty::MetalCenterVaryings)
    {
        // Beside `sampleRateShading`, the varyings' pixel form needs an `InterpolateAtOffset` that
        // reads from the pixel center and the interpolation functions a portability-subset device
        // withholds.
        let mut needs = Vec::new();
        if !enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
        {
            needs.push(Capability::Property(
                VkProperty::InterpolateAtOffsetCentered,
            ));
        }
        let feature = VkFeature::ShaderSampleRateInterpolationFunctions;
        if enabled.portability_withholds(feature) {
            needs.push(Capability::Feature(feature));
        }
        if !needs.is_empty() {
            return Err(Unsupported::needs_any_of(Feature::SampleRateShading, vec![needs]).into());
        }
    }
    // The device has every pixel form the function needs; a function compiled without
    // `multisampled` uses none of them.
    if !io.multisampled {
        return Err(Error::NeedsInput(RuntimeInput::Multisampled));
    }
    if center && !io.varyings_at_pixel {
        // Compiled for another device.
        return Err(refused());
    }
    Ok(sample_shading)
}

/// Whether the pipeline's alpha to coverage stays enabled (see the module docs): only over a
/// fragment output whose alpha it reads, a four-component float first output at attachment 0.
/// Without one, Metal's coverage is the rasterizer's, as with alpha to coverage off. A
/// descriptor that enables it over an integer attachment 0 is Metal's pipeline error; that was
/// measured with rasterization on, and is refused with it off as well (unmeasured).
fn reads_alpha_to_coverage(
    r: &RenderLinkInputs,
    fs: Option<&StageInterface>,
) -> Result<bool, Error> {
    if !r.alpha_to_coverage {
        return Ok(false);
    }
    let integer = r.color_attachments[0].is_some_and(|a| {
        matches!(
            sampled_type(a.pixel_format),
            Some(SampledType::Uint | SampledType::Sint)
        )
    });
    if integer {
        return Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 0,
        }));
    }
    // An extended-range attachment 0 is written as raw texels, whose alpha the coverage cannot
    // read: what Metal's coverage is there is unmeasured.
    if r.color_attachments[0].is_some_and(|a| ExtendedRange::of(a.pixel_format).is_some()) {
        let refused = Unsupported::not_implemented(Feature::Format);
        return Err(refused.at_attachment(Attachment::Color(0)).into());
    }
    let reads = fs.into_iter().flat_map(|f| &f.io.outputs).any(|o| {
        o.role
            == IoRole::Color {
                attachment: 0,
                index: 0,
            }
            && o.ty.numeric == IoNumeric::Float
            && o.ty.components == 4
    });
    // An attachment 0 held with its channels rotated takes Metal's alpha in its red, where
    // Vulkan's coverage reads its alpha (Metal's blue). One holding Metal's alpha in red takes
    // that alpha in its alpha lane too (`StageIo::alpha_in_red_colors`, which
    // `color_writes::check` holds to the pipeline's), so the coverage reads Metal's alpha, as
    // Metal's does through an `A8Unorm` attachment 0 (probe `a8`'s uses: the masks of every alpha
    // in 1/32 steps at four samples are `RGBA8Unorm`'s).
    if reads
        && r.color_attachments[0].is_some_and(|a| metal::format::rotated_channels(a.pixel_format))
    {
        let refused = Unsupported::not_implemented(Feature::Format);
        return Err(refused.at_attachment(Attachment::Color(0)).into());
    }
    Ok(reads)
}

/// The color attachment states `states` made identical, on a profile without `independentBlend`,
/// where identical states leave every attachment as the descriptor's would; otherwise
/// `Unsupported{IndependentBlending, Feature(IndependentBlend)}` (see the module docs).
/// `declared` holds the attachments an output of the fragment function names (either
/// dual-source index), and `fragment` says whether the pipeline has a fragment function.
fn uniform_blend(
    states: &mut [Option<ColorAttachmentState>],
    declared: &BTreeSet<u32>,
    fragment: bool,
    p: &DeviceProfile,
) -> Result<(), Error> {
    // An attachment's element of `pAttachments`; its format is not part of it.
    let element = |s: &ColorAttachmentState| (s.write_mask, s.blend);
    let mut present = states.iter().flatten().map(element);
    let Some(first) = present.next() else {
        return Ok(());
    };
    if present.all(|e| e == first) {
        return Ok(());
    }
    let refused = |index: usize| -> Error {
        Unsupported::needs(
            Feature::IndependentBlending,
            Capability::Feature(VkFeature::IndependentBlend),
        )
        .at_attachment(Attachment::Color(index as u32))
        .into()
    };
    // No output at its location: the draw does not modify it under any state. An attachment a
    // dual-source second output alone names has an output there, and its empty mask (from
    // `graphics_state`) must stay.
    let untouched = |index: usize| fragment && !declared.contains(&(index as u32));
    // The element of the attachments the draw writes: one, which the others then take.
    let mut kept = None;
    for (index, s) in states.iter().enumerate() {
        let Some(s) = s else { continue };
        if untouched(index) || s.write_mask.is_empty() {
            continue;
        }
        match kept {
            Some(k) if k != element(s) => return Err(refused(index)),
            _ => kept = Some(element(s)),
        }
    }
    let (write_mask, blend) = kept.unwrap_or((vk::ColorComponentFlags::empty(), None));
    for (index, s) in states.iter_mut().enumerate() {
        let Some(s) = s else { continue };
        if untouched(index) {
            // Any state leaves it as it was, but a blend must be valid for its format
            // (`VUID-VkGraphicsPipelineCreateInfo-renderPass-06041`, `-06062` under dynamic
            // rendering), and it has no second output.
            if let Some(b) = blend {
                if b.reads_second_output()
                    || !p.supports_format(s.format, FormatUsage::ColorAttachmentBlend)
                {
                    return Err(refused(index));
                }
            }
        } else if s.write_mask.is_empty() && !write_mask.is_empty() {
            // Written under an empty mask: its blend writes nothing, but its mask must stay empty.
            return Err(refused(index));
        }
        s.write_mask = write_mask;
        s.blend = blend;
    }
    Ok(())
}

/// The fragment outputs of `fs` against the color attachments (see the module docs).
/// The write mask of an attachment held in the raw texels of extended-range `form` whose Metal
/// mask is `mask` (as Vulkan's channels): every raw component where `mask` keeps every channel the
/// format has, none where it keeps none of them. A mask of some of its channels is
/// `NotImplemented(Format)`: Metal keeps a masked channel's code but rewrites the texel's other
/// bits (`BGRA10_XR`'s alpha clamped even where masked, its low bits and `BGR10_XR`'s top bits
/// made canonical), which a mask of the raw components cannot express. A padded form's channels are
/// its raw words, one each, and the mask is theirs: a loading pass that draws keeps a masked
/// channel's code and zeroes its word's low six bits, every word of each tile it stores alike (an
/// Apple M3 Max and an M4, byte-identical), which a consumer's load pass does before the pass (the
/// runtime's `render/unused_bits_load.rs`, `bind::unused_bits_module`), so a channel the mask
/// leaves out keeps the word Metal's pass leaves. A bfloat form's channels are its own (red; red
/// and green; all four), each a raw component; a mask of some of them is unmeasured and refused as
/// above.
fn extended_write_mask(
    form: ExtendedRange,
    mask: vk::ColorComponentFlags,
) -> Result<vk::ColorComponentFlags, Error> {
    use vk::ColorComponentFlags as C;
    let channels = match form {
        ExtendedRange::Bgr10 { .. } => C::R | C::G | C::B,
        ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => C::R,
        ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => C::R | C::G,
        ExtendedRange::R16Bfloat => C::R,
        ExtendedRange::Rg16Bfloat => C::R | C::G,
        _ => C::R | C::G | C::B | C::A,
    };
    let kept = mask & channels;
    if kept.is_empty() {
        return Ok(C::empty());
    }
    // `RGB10A2Unorm_sRGB`'s channels share one word: a mask of some of them is refused as above.
    if form.padded() && form.raw_components() == channels.as_raw().count_ones() {
        return Ok(kept);
    }
    if kept != channels {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    Ok(match form.raw_components() {
        1 => C::R,
        2 => C::R | C::G,
        _ => channels,
    })
}

fn check_outputs(
    r: &RenderLinkInputs,
    fs: &StageInterface,
    states: &[Option<ColorAttachmentState>],
    p: &DeviceProfile,
) -> Result<(), Error> {
    let invalid = |index: u32| Error::InvalidDescriptor(DescriptorError::ColorAttachment { index });
    let mut written = BTreeSet::new();
    let mut second = BTreeSet::new();
    for o in &fs.io.outputs {
        let IoRole::Color { attachment, index } = o.role else {
            continue;
        };
        if index == 0 {
            written.insert(attachment);
        } else {
            second.insert(attachment);
        }
        let Some(a) = r
            .color_attachments
            .get(attachment as usize)
            .copied()
            .flatten()
        else {
            continue;
        };
        match (o.ty.numeric, sampled_type(a.pixel_format)) {
            (IoNumeric::Float, Some(SampledType::Float))
            | (IoNumeric::Uint, Some(SampledType::Uint))
            | (IoNumeric::Sint, Some(SampledType::Sint)) => {}
            _ => return Err(invalid(attachment)),
        }
        // Fewer components than the format's channels: Metal's pipeline error, where Vulkan would
        // write undefined values to the missing ones.
        if color_output_components(a.pixel_format).is_some_and(|n| o.ty.components < n) {
            return Err(invalid(attachment));
        }
    }
    let dual: Vec<u32> = states
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref()?.blend?.reads_second_output().then_some(i as u32))
        .collect();
    if dual.is_empty() {
        return Ok(());
    }
    let max = p.fields().limits.max_fragment_dual_src_attachments;
    let over = |required: u64| -> Error {
        Unsupported::needs(
            Feature::DualSourceBlending,
            Capability::Limit {
                limit: VkLimit::MaxFragmentDualSrcAttachments,
                required,
                available: u64::from(max),
            },
        )
        .into()
    };
    for &a in &dual {
        // Metal blends a second output only with the first ("does not write to render target
        // color(0), index(0) that is required for blending").
        if !second.contains(&a) || !written.contains(&a) {
            return Err(invalid(a));
        }
        if a >= max {
            return Err(over(u64::from(a) + 1).at_attachment(Attachment::Color(a)));
        }
    }
    // VUID-RuntimeSpirv-Fragment-06427: with dual-source blending, the attachments written stay
    // within the dual-source limit.
    if written.len() as u64 > u64::from(max) {
        return Err(over(written.len() as u64));
    }
    Ok(())
}

/// The write mask `mask` of Metal's channels, of an image holding them rotated
/// (`metal::format::to_rotated_image`): Metal's alpha, red, green and blue are the image's red,
/// green, blue and alpha.
fn rotated_mask(mask: vk::ColorComponentFlags) -> vk::ColorComponentFlags {
    use vk::ColorComponentFlags as C;
    let channels = [C::R, C::G, C::B, C::A].map(|c| mask.contains(c));
    let image = metal::format::to_rotated_image(channels);
    [C::R, C::G, C::B, C::A]
        .into_iter()
        .zip(image)
        .filter(|&(_, on)| on)
        .fold(C::empty(), |out, (c, _)| out | c)
}
