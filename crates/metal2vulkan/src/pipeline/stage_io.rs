//! The stage-IO interface of one vertex or fragment function: the plain-data contract between
//! `lower::stage_io` and link.
//!
//! A stage module declares its user interface variables with relocatable `Location` literals, each
//! keyed by a [`VaryingKey`]. This module says what every key *is*, so link numbers them without
//! reading AIR or the module's words:
//!
//! - [`IoRole::Attribute`]: a vertex input (`[[attribute(n)]]`), fetched by fixed-function vertex
//!   input. Link gives it a location and describes it in `VertexInputState`.
//! - [`IoRole::Varying`]: a vertex output or fragment input, matched across the two stages by its
//!   AIR user key (the Metal linking rule). Both stages use the same [`VaryingKey`] for one user
//!   key, so one `LinkTable.locations` entry numbers both sides.
//! - [`IoRole::Color`]: a fragment color output (`[[color(n), index(i)]]`). Its location must be
//!   the attachment `n` (Vulkan's fragment output location is the color attachment index).
//! - [`IoRole::ViewportIndex`]: the `[[viewport_array_index]]` a vertex function writes, as
//!   written, carried to a fragment function that reads it. A flat `uint` varying for link, one key
//!   ([`VaryingKey::viewport_index`]) on both sides.
//! - [`IoRole::RenderTargetArrayIndex`]: the `[[render_target_array_index]]` a vertex function
//!   writes, as written, carried to a fragment function that reads it. A flat `uint` varying for
//!   link, one key ([`VaryingKey::render_target_array_index`]) on both sides.
//! - [`IoRole::WindowDepth`]: a vertex function's window depth, carried to a fragment function
//!   that runs its draw's depth test itself at more than one sample. A no-perspective `float`
//!   varying for link, one key ([`VaryingKey::window_depth`]) on both sides.
//! - [`IoRole::Barycentric`]: the first two barycentric weights a fragment function reads as
//!   `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric`, which the pipeline's
//!   synthesized geometry stage writes. A `float2` varying, perspective-correct or not, one key
//!   per form ([`VaryingKey::barycentric`]).
//! - [`IoRole::SelectedLayer`] and [`IoRole::SelectedViewport`]: the `Layer` and `ViewportIndex` a
//!   vertex function selects its primitive's slice and viewport with on a device that cannot store
//!   the builtin from a vertex function, which the pipeline's synthesized geometry stage stores
//!   from the primitive's first vertex. A flat `uint` vertex output, one key each
//!   ([`VaryingKey::selected_layer`], [`VaryingKey::selected_viewport`]), that no fragment
//!   function reads.
//! - [`IoRole::PrimitiveId`]: Metal's `[[primitive_id]]` on a device that cannot declare a
//!   fragment `PrimitiveId`, which the pipeline's vertex function computes for the primitive each
//!   vertex provokes, and behind a post-tessellation vertex function emulated on a device without
//!   `tessellationShader`, which writes its patch's position in the draw. A flat `uint` varying
//!   for link, one key ([`VaryingKey::primitive_id`]) on both sides.
//!
//! The roles live in disjoint key namespaces ([`VaryingKey::user`], [`VaryingKey::attribute`],
//! [`VaryingKey::color`], [`VaryingKey::viewport_index`],
//! [`VaryingKey::render_target_array_index`], [`VaryingKey::window_depth`],
//! [`VaryingKey::barycentric`], [`VaryingKey::selected_layer`],
//! [`VaryingKey::selected_viewport`], [`VaryingKey::primitive_id`]): the key bytes start with a
//! namespace tag, so no AIR user key can collide with another role's key. Keys stay opaque to
//! link; it reads the role here, never the bytes.
//!
//! Every interface variable is 32-bit (narrow varyings are widened at the interface, so no varying
//! needs `storageInputOutput16`) but a color output written as the bits of the half Metal stores,
//! 16-bit ([`IoType::bits`]); [`IoType::source_bits`] keeps the AIR element width for format
//! checks. An attribute or color takes one whole location; a user varying takes one location at
//! component 0 unless link packs varyings into shared locations, for which the module decorates
//! every varying with a relocatable `Component`.
//!
//! A vertex attribute the function fetches in the shader ([`StageIo::pulled`]) has no `Location`;
//! it is listed with the fetch it was compiled under, so link can check the pipeline's vertex
//! descriptor projects to the same fetch. A kernel's `[[stage_in]]` attributes
//! ([`StageIo::stage_in`]) are listed the same way with the stage-input descriptor's fetch, which
//! the compute pipeline checks.

use std::collections::BTreeMap;

use crate::error::{DescriptorError, Error};
use crate::metal;
use crate::pipeline::descriptor::StageInputDescriptor;
use crate::pipeline::lower_inputs::{ChannelWrite, VertexFetch};
use crate::pipeline::sym::{AirUserKey, VaryingKey};

/// Key namespace tags (the first key byte).
const USER: u8 = 0;
const ATTRIBUTE: u8 = 1;
const COLOR: u8 = 2;
const VIEWPORT_INDEX: u8 = 3;
const RENDER_TARGET_ARRAY_INDEX: u8 = 4;
const WINDOW_DEPTH: u8 = 5;
const BARYCENTRIC: u8 = 6;
const SELECTED_LAYER: u8 = 7;
const SELECTED_VIEWPORT: u8 = 8;
const PRIMITIVE_ID: u8 = 9;

impl VaryingKey {
    /// The key of the user varying with AIR user key `k` (vertex output and fragment input).
    pub fn user(k: &[u8]) -> VaryingKey {
        let mut b = Vec::with_capacity(k.len() + 1);
        b.push(USER);
        b.extend_from_slice(k);
        VaryingKey(AirUserKey(b.into()))
    }

    /// The key of vertex attribute `index` (`[[attribute(index)]]`).
    pub fn attribute(index: u32) -> VaryingKey {
        let mut b = vec![ATTRIBUTE];
        b.extend_from_slice(&index.to_le_bytes());
        VaryingKey(AirUserKey(b.into()))
    }

    /// The key of fragment color output `attachment` with dual-source index `index`.
    pub fn color(attachment: u32, index: u32) -> VaryingKey {
        let mut b = vec![COLOR];
        b.extend_from_slice(&attachment.to_le_bytes());
        b.extend_from_slice(&index.to_le_bytes());
        VaryingKey(AirUserKey(b.into()))
    }

    /// The key of the raw viewport index a vertex function writes and a fragment function reads
    /// (`IoRole::ViewportIndex`).
    pub fn viewport_index() -> VaryingKey {
        VaryingKey(AirUserKey(vec![VIEWPORT_INDEX].into()))
    }

    /// The key of the raw render target slice index a vertex function writes and a fragment
    /// function reads (`IoRole::RenderTargetArrayIndex`).
    pub fn render_target_array_index() -> VaryingKey {
        VaryingKey(AirUserKey(vec![RENDER_TARGET_ARRAY_INDEX].into()))
    }

    /// The key of the window depth a vertex function writes and a fragment function that runs
    /// its draw's depth test itself at more than one sample reads (`IoRole::WindowDepth`).
    pub fn window_depth() -> VaryingKey {
        VaryingKey(AirUserKey(vec![WINDOW_DEPTH].into()))
    }

    /// The key of the barycentric weights a pipeline's synthesized geometry stage writes and a
    /// fragment function reads as `[[barycentric_coord]]` (`IoRole::Barycentric`), in the form
    /// `perspective` names.
    pub fn barycentric(perspective: bool) -> VaryingKey {
        VaryingKey(AirUserKey(vec![BARYCENTRIC, u8::from(perspective)].into()))
    }

    /// The key of the `Layer` a vertex function selects for the pipeline's synthesized geometry
    /// stage to store (`IoRole::SelectedLayer`).
    pub fn selected_layer() -> VaryingKey {
        VaryingKey(AirUserKey(vec![SELECTED_LAYER].into()))
    }

    /// The key of the `ViewportIndex` a vertex function selects for the pipeline's synthesized
    /// geometry stage to store (`IoRole::SelectedViewport`).
    pub fn selected_viewport() -> VaryingKey {
        VaryingKey(AirUserKey(vec![SELECTED_VIEWPORT].into()))
    }

    /// The key of the primitive ID a vertex function writes and a fragment function reads as
    /// `[[primitive_id]]` (`IoRole::PrimitiveId`).
    pub fn primitive_id() -> VaryingKey {
        VaryingKey(AirUserKey(vec![PRIMITIVE_ID].into()))
    }
}

/// The stage-IO interface of one function (empty for kernels).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageIo {
    /// Declared user input variables (vertex attributes or fragment varyings), in parameter
    /// order.
    pub inputs: Vec<IoVariable>,
    /// Declared user output variables (vertex varyings or fragment colors), in the order of the
    /// entry's outputs.
    pub outputs: Vec<IoVariable>,
    /// The builtins the module declares, sorted.
    pub builtins: Vec<StageBuiltin>,
    /// The vertex attributes the function fetches in the shader (`VertexFetch::Pulled`), sorted by
    /// attribute index. They have no `Location`: each reads the storage buffer
    /// `ResourceSlot{Vertex, Buffer, buffer_index}` its fetch names.
    pub pulled: Vec<PulledInput>,
    /// The kernel `[[stage_in]]` attributes the function fetches in the shader, sorted by attribute
    /// index, each with the fetch it was compiled under; they read the kernel buffer slots their
    /// fetches name.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stage_in: Vec<StageInInput>,
    /// The fragment function was compiled for a render pipeline that rasterizes more than one
    /// sample (`LoweringInputs::multisampled`): it reads `[[position]]`'s x and y as
    /// `floor(FragCoord.xy) + 0.5`, the pixel center Metal reads in every invocation, whatever
    /// samples the invocation shades.
    #[cfg_attr(feature = "serde", serde(default))]
    pub multisampled: bool,
    /// Under `multisampled`, the function reads its center-sampled varyings at the pixel center,
    /// as Metal does in every invocation: through `InterpolateAtOffset` at offset 0, the pixel
    /// center (GLSL.std.450), or as they are on a device that reads them there itself
    /// (`VkProperty::MetalCenterVaryings`). Without it they read wherever the device evaluates an
    /// undecorated input.
    #[cfg_attr(feature = "serde", serde(default))]
    pub varyings_at_pixel: bool,
    /// The fragment function may read `[[position]]`'s z or w: anything but its x and y lanes.
    /// Metal reads them at the pixel center in every invocation, and Vulkan evaluates them where
    /// it shades, a sample under sample shading.
    #[cfg_attr(feature = "serde", serde(default))]
    pub position_depth: bool,
    /// How a fragment function that reads `[[sample_mask]]` and that Metal runs once per covered
    /// sample reads the pixel's coverage in each of those runs ([`CoverageLoop`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub coverage_loop: CoverageLoop,
    /// How the fragment function writes each color attachment whose format converts its float
    /// output (`LoweringInputs::color_writes`), for every attachment it writes, through an output
    /// variable or in the storage form: link refuses a stage compiled for another map than the
    /// pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_writes: BTreeMap<u32, [Option<ChannelWrite>; 4]>,
    /// The attachments among `color_writes` whose float writes the function rounds to Metal's
    /// codes in the shader, sorted: those with a channel it rounds by Metal's rule rather than
    /// leaving to the device (`StageModule::rounded_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rounded_colors: Vec<u32>,
    /// The attachments among `color_writes` whose output the function writes as the bits of the
    /// half Metal stores, through a 16-bit output, rounding none of its channels (a `half`
    /// output to a half-float attachment), sorted: the device's conversion of a 32-bit output
    /// would not keep a NaN's payload (`StageModule::half_bit_colors`). Link gives these, as
    /// `rounded_colors`, no Vulkan blend of one and zero.
    #[cfg_attr(feature = "serde", serde(default))]
    pub half_bit_colors: Vec<u32>,
    /// The extended-range attachments the fragment function writes a `float` or `half` output
    /// to through output variables of dual-source index 0
    /// (`LoweringInputs::extended_range_colors`), by color index: each output variable there is
    /// declared as the raw format's unsigned components and holds the raw texel Metal stores
    /// ([`StageIo::declared_type`]; the variable's `ty` stays the Metal output's, which link checks
    /// against the attachment's format), and link gives the attachment the raw format and refuses a
    /// stage compiled for another map than the pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_range_colors: BTreeMap<u32, crate::pipeline::lower_inputs::ExtendedRangeColor>,
    /// The word attachments the fragment function writes a `float` or `half` output to through
    /// output variables of dual-source index 0 (`LoweringInputs::word_colors`), by color index:
    /// each output variable there is declared as the word's unsigned integer and holds the word
    /// Metal stores ([`StageIo::declared_type`]; the variable's `ty` stays the Metal output's,
    /// which link checks against the attachment's format), and link gives the attachment the
    /// word's format and refuses a stage compiled for another map than the pipeline's
    /// (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub word_colors: BTreeMap<u32, crate::metal::format::WordAttachment>,
    /// The input attachments the fragment function reads through Metal's decode of their bytes
    /// (`LoweringInputs::srgb_fetches`): each read snaps the device's sRGB decode back to its
    /// byte and decodes that by Metal's table. Link refuses a stage compiled for another set
    /// than the pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_fetches: std::collections::BTreeSet<u32>,
    /// The attachments held with their channels rotated (`LoweringInputs::rotated_colors`) that
    /// the fragment function writes through an output variable of either dual-source index, by
    /// color index: each such output stores Metal's channels rotated
    /// (`metal::format::to_rotated_image`), so link gives the attachment its image's format, its
    /// write mask rotated, and a blend only where the rotation leaves the blend Metal's, and
    /// refuses a stage compiled for another set than the pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_colors: std::collections::BTreeSet<u32>,
    /// The attachments whose image holds Metal's alpha in red
    /// (`LoweringInputs::alpha_in_red_colors`) that the fragment function writes through an output
    /// variable of either dual-source index, or reads (`[[color(n)]]`) through an input attachment
    /// or the interlocked storage form, by color index: each such output holds Metal's alpha in
    /// the image's alpha lane too, which alpha to coverage reads, and each read takes the image's
    /// red as Metal's alpha. Link refuses a stage compiled for another set than the pipeline's
    /// (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha_in_red_colors: std::collections::BTreeSet<u32>,
    /// The second outputs the fragment function writes for attachments held with their channels
    /// rotated (`LoweringInputs::rotated_factors`), by color index: those of the attachments it
    /// writes through a first output, each with what it holds in each channel of the image. Link
    /// blends the attachment through Vulkan's `SRC1` factors from it and refuses a stage compiled
    /// for another map than the pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_factors: BTreeMap<u32, [crate::pipeline::lower_inputs::FactorLane; 4]>,
    /// The forms of the blends of attachments held with their channels rotated the function
    /// compiled under (`LoweringInputs::rotated_blends`), by color index: those of the
    /// attachments it writes through a first output. Link refuses a stage compiled for another
    /// map than the pipeline's (`NeedsInput(ColorWrites)`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_blends: BTreeMap<u32, crate::pipeline::lower_inputs::RotatedBlendForm>,
    /// The inputs the fragment function reads per primitive (`PerPrimitiveEXT`), sorted: those
    /// of the keys its mesh render pipeline's mesh function writes per primitive
    /// (`LoweringInputs::primitive_varyings`). Link refuses a fragment function whose list is not
    /// the pipeline's (`NeedsInput(PrimitiveVaryings)`); empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_inputs: Vec<VaryingKey>,
    /// The fragment function declares `[[early_fragment_tests]]`: Metal runs the depth and
    /// stencil tests before it, and runs it only for the fragments that pass them (under a depth
    /// compare of `Never` it never runs, a probe). What the function declares, not what the module
    /// does (a framebuffer fetch's form may add early tests, `lower::fragment`); false for every
    /// other function. Link states it as `GraphicsState::early_fragment_tests`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub early_fragment_tests: bool,
    /// The colors the fragment function writes and stores nowhere, its render pipeline having
    /// no attachment there (`LoweringInputs::unattached_colors`), sorted: no output variable at
    /// each. Link refuses the function in a pipeline with an attachment at one of them
    /// (`NeedsInput(ColorWrites)`); empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unattached_colors: Vec<u32>,
    /// The render pipeline's private masks the fragment function applies itself, by the window
    /// pixel's place in its 2x2 quad (`LoweringInputs::sample_mask`): its returns AND the
    /// pixel's into the `SampleMask` output, which [`builtins`](StageIo::builtins) then lists as
    /// `SampleMaskOut`. `None` for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sample_mask: Option<[u32; 4]>,
    /// The render pipeline's private alpha test the fragment function runs itself
    /// (`LoweringInputs::alpha_test`): its returns kill an invocation whose color 0's alpha fails
    /// the comparison against `PushField::AlphaTestReference`. `None` for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha_test: Option<crate::metal::CompareFunction>,
    /// The samplers a fragment function makes depth comparisons through, and those whose
    /// references it clamps (`LoweringInputs::depth_compare_clamp_mask`). Link refuses a function
    /// whose clamped samplers are not the pipeline's mask over its samplers
    /// (`NeedsInput(DepthCompareClamp)`). Empty for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_compares: DepthCompares,
    /// The channels taking a `half` output at `float` precision the function compiled under
    /// (`LoweringInputs::half_output_lanes`), by color index: those of the attachments it writes
    /// a `half` output to. Link refuses a stage one of whose entries is not the pipeline's
    /// (`NeedsInput(ColorWrites)`); an attachment without one, every attachment of a function
    /// compiled without the map among them, takes the `half` the function rounds, under
    /// `PrecisionNote::HalfOutputPrecision` (under fast math `HalfOutputFastMathBlend`, by
    /// [`half_fast_colors`](Self::half_fast_colors)), and links.
    /// Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub half_output_lanes:
        BTreeMap<u32, [Option<crate::pipeline::lower_inputs::HalfOutputLane>; 4]>,
    /// The `textureWriteRoundingMode` under which a fragment function's texture writes that
    /// Apple's mode 2 stores otherwise were compiled (`LoweringInputs::texture_write_rounding`),
    /// where the function writes a `float` texel that may reach a half-float or packed
    /// small-float texture (`RG11B10Float`, `RGB9E5Float`) or a `half` texel that may reach a
    /// packed small-float one; `None` where it writes none (and for every other function). Link
    /// answers a stage whose mode is not the pipeline's `NeedsInput(ColorWrites)`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_write_rounding: Option<crate::metal::TextureWriteRoundingMode>,
    /// The varyings a vertex function writes the same value at every vertex of a primitive,
    /// sorted: computed only from constants, `constant` memory read through uniform pointers, the
    /// draw's instance, base vertex, base instance, amplification and patch, and the attributes
    /// it fetches per instance or at a constant step, never from `[[vertex_id]]` or another
    /// per-vertex input. A fragment function of its pipeline reads those of its user varyings it
    /// interpolates flat ([`uniform_inputs`](StageIo::uniform_inputs)). The raw varyings of the
    /// slice and viewport indices it writes are among them so
    /// (`VaryingKey::render_target_array_index`, `VaryingKey::viewport_index`): Vulkan takes
    /// `Layer` and `ViewportIndex` from a vertex function only where every vertex of a primitive
    /// stores the same, so it stores an index that is not to its selected index instead
    /// (`IoRole::SelectedLayer`, `IoRole::SelectedViewport`). Empty for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub uniform_outputs: Vec<VaryingKey>,
    /// The `float` and `half` varyings the fragment function reads flat although it states an
    /// interpolation, because its render pipeline's vertex function writes them the same at every
    /// vertex of a primitive (`LoweringInputs::uniform_varyings`), sorted. Link refuses the
    /// function behind a vertex function whose
    /// [`uniform_outputs`](StageIo::uniform_outputs) lack one of them
    /// (`NeedsInput(UniformVaryings)`). Empty for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub uniform_inputs: Vec<VaryingKey>,
    /// The fixed-function attributes a vertex function was compiled to take as stepped per
    /// instance (`LoweringInputs::instanced_attributes`), sorted: its
    /// [`uniform_outputs`](StageIo::uniform_outputs) hold only while they are, so link refuses it
    /// behind a vertex descriptor that steps one of them per vertex
    /// (`InvalidDescriptor(VertexAttribute)`). Empty for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub instanced_attributes: Vec<u32>,
    /// The attachments among `color_writes` whose rules the draw selects
    /// (`ChannelWrite::Selected`), by color index, with what the function's write does under each
    /// code of the attachment's rule word (`PushField::ColorWriteRule`): link derives the word
    /// from the pipeline's attachment (`GraphicsState::color_write_rules`), and from it whether
    /// the write rounds in the shader and the notes it keeps.
    #[cfg_attr(feature = "serde", serde(default))]
    pub selected_colors: BTreeMap<u32, SelectedWrite>,
    /// The attachments a `half` color output of a module compiled without fast math
    /// (`air.compile.fast_math_disable`, safe or relaxed math) reaches, by color index, sorted.
    /// Metal stores there the `half` the function returns, blended into a 16-bit float format with
    /// each factor, product and sum rounded to `half` and into a 32-bit float one as a `float`
    /// multiply-add of it, so the function stores Metal's value wherever the attachment does not
    /// blend; the pipeline's stage records `PrecisionNote::HalfOutputPrecision` only where one of
    /// these attachments blends (`color_writes::blends_output`). A module with fast math lists its
    /// attachments in [`half_fast_colors`](Self::half_fast_colors); one naming no math mode records
    /// its note itself and lists none. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub half_blend_colors: Vec<u32>,
    /// The attachments a `float` color output (a scalar or vector of AIR's `float`) of a module
    /// compiled without fast math (`air.compile.fast_math_disable`) reaches as color attachments,
    /// by color index, sorted. Where such an attachment is a 16-bit float format that blends with
    /// factors one and zero, added, Metal keeps the destination's product by zero in the blend (a
    /// NaN of an infinite or NaN texel, `+0` of a `-0` output over a texel of `+0` or above), which
    /// no write of the output alone stores: the pipeline blends it there and
    /// its stage records `PrecisionNote::HalfOutputPrecision` (`link::graphics`). Empty for any
    /// other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub float_blend_colors: Vec<u32>,
    /// The attachments a `half` color output of a module compiled with fast math
    /// (`air.compile.fast_math_enable`) reaches as color attachments, by color index, sorted (not
    /// one the function writes in the storage form, nor one whose channels take the output at
    /// `float` precision where Metal computes it so, [`half_output_lanes`](Self::half_output_lanes)).
    /// Metal's compiler folds the attachment's blend and conversion into the function under fast
    /// math, so the pipeline's stage records `PrecisionNote::HalfOutputFastMathBlend` where one of
    /// these attachments stores some channel (`color_writes::stores_output`). One whose write mask
    /// stores none of its channels, or that the pipeline lacks, stores nothing, so no value of the
    /// output can differ there. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub half_fast_colors: Vec<u32>,
    /// The index builtins, `Layer` and `ViewportIndex`, a fragment function reads as 0 on a device
    /// that cannot declare them (`Layer` needs `geometryShader` or `shaderOutputLayer`,
    /// `ViewportIndex` `multiViewport` or `shaderOutputViewportIndex`), sorted: compiled for a
    /// pipeline where no stage writes the index (neither under
    /// `LoweringInputs::render_target_array_index_varying` or
    /// `LoweringInputs::viewport_index_varying` nor under `LoweringInputs::amplification_draws`),
    /// whose builtin would hold 0, which Metal's `[[render_target_array_index]]` and
    /// `[[viewport_array_index]]` read there too. Link counts each as a read of its builtin: one
    /// beside a vertex function that writes the index is `NeedsInput`, and `Layer` here makes the
    /// pipeline's [`GraphicsState::declares_layer`](crate::pipeline::graphics::GraphicsState::declares_layer).
    /// Empty for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zero_indices: Vec<StageBuiltin>,
    /// The group of its render pipeline's color attachments the fragment function writes
    /// (`LoweringInputs::color_group`): its colors at locations `0..`. Link refuses the function
    /// in a pipeline variant of another group (`NeedsInput(ColorWrites)`). `None` for a function
    /// compiled with each color at its own index, and for every other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_group: Option<crate::pipeline::lower_inputs::ColorGroup>,
}

/// What a fragment function's write of a color attachment whose rules the draw selects does
/// under each code of the attachment's rule word (`StageIo::selected_colors`): bit `c` of a mask
/// is code `c` (`pipeline::color_writes`'s codes: no rule, the device's conversion, an 8-bit
/// unorm channel's, a half-float channel's).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectedWrite {
    /// The lanes of the output the rules take, the first `lanes` channels' codes.
    pub lanes: u8,
    /// A lane of code `c` rounds the output to Metal's code in the shader.
    pub rounds: u8,
    /// A lane of code `c` stores a NaN as the device encodes it
    /// (`PrecisionNote::DeviceNanEncoding`).
    pub nans: u8,
}

/// The samplers a fragment function makes depth comparisons through ([`StageIo::depth_compares`]),
/// as bits of Metal's `fragmentDepthCompareClampMask`: bit `s` for `[[sampler(s)]]` (`s` below
/// 16, Metal's sampler slots), bit 31 for a constexpr sampler, and bit 30
/// ([`DepthCompares::ARGUMENT`]) for an argument buffer's sampler, which no mask bit names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthCompares {
    /// Every sampler the function compares through.
    pub samplers: u32,
    /// Those whose references the function clamps: the mask it compiled under over `samplers`.
    pub clamped: u32,
}

impl DepthCompares {
    /// The bit of a comparison through an argument buffer's sampler.
    pub const ARGUMENT: u32 = 1 << 30;
    /// The bit of a comparison through a constexpr sampler (Metal's mask bit 31, measured).
    pub const CONSTEXPR: u32 = 1 << 31;
}

/// How a fragment function reads `[[sample_mask]]` where Metal runs it once per covered sample: it
/// reads `[[sample_id]]` or a sample-qualified varying, or writes memory. Metal hands each of those
/// runs the pixel's whole coverage, and a Vulkan invocation under sample shading sees only its own
/// sample's bit, so such a function runs once per pixel instead, reads the coverage there, and runs
/// its body once per covered sample in a loop over the coverage's bits, each pass reading its
/// sample's index and its sample-qualified varyings at its sample. One pixel-rate invocation cannot
/// write each sample's outputs, so the loop is only for a function whose side effects are its
/// memory writes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoverageLoop {
    /// The function does not both read `[[sample_mask]]` and run once per sample on Metal.
    #[default]
    NotNeeded,
    /// Compiled for a multisampled pipeline (`LoweringInputs::multisampled`), the function runs
    /// once per pixel, without sample shading, and runs its body once per covered sample. Its
    /// outputs are the last pass's, written to each covered sample whose pass's sample mask keeps
    /// it, where Metal writes each sample's from that sample's run
    /// (`PrecisionNote::LoopedSampleOutputs`).
    Looped,
    /// The function loops over its covered samples when compiled for a multisampled pipeline,
    /// and was compiled without `multisampled`: link asks for it
    /// (`NeedsInput(Multisampled)`) in a pipeline of more than one sample.
    Unlooped,
    /// The function cannot loop over its covered samples: link refuses it in a pipeline of more
    /// than one sample for this reason.
    Refused(CoverageLoopRefusal),
    /// The driver runs the function, which writes memory, once per covered sample whatever the
    /// pipeline's sample shading, each run reading the pixel's coverage, as Metal does
    /// (`DriverWorkaround::MemoryWritingFragmentsRunPerSample`): it is compiled as written, without
    /// the loop, its `[[sample_id]]` and sample-qualified varyings read per sample, and link
    /// enables no sample shading, under which the driver would hand each run only its own sample's
    /// bit of the coverage.
    DriverRunsPerSample,
}

/// Why a fragment function that reads `[[sample_mask]]` and runs once per sample on Metal has no
/// loop over its covered samples ([`CoverageLoop::Refused`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoverageLoopRefusal {
    /// It runs under `[[early_fragment_tests]]`: Metal then runs only the samples that pass the
    /// tests, each reading the coverage from before them, and a Vulkan invocation reads either
    /// coverage, never both. `Unsupported{PixelCoverageAtSampleRate, NotRepresentable}`.
    EarlyFragmentTests,
    /// It discards: a sample's discard would have to end only that sample's pass.
    /// `Unsupported{PixelCoverageAtSampleRate, NotImplemented}`.
    Discard,
    /// It computes a derivative, samples a texture at an implicit level of detail, or runs a
    /// quad or simdgroup operation or a barrier: the passes' counts differ between the
    /// invocations of a quad. `Unsupported{PixelCoverageAtSampleRate, NotImplemented}`.
    QuadOperations,
    /// It reads a color attachment or the imageblock, or names a raster order group.
    /// `Unsupported{PixelCoverageAtSampleRate, NotImplemented}`.
    Attachments,
    /// It reads a sample-qualified varying on a device without interpolation functions
    /// (`shaderSampleRateInterpolationFunctions`, which a portability-subset device may
    /// withhold), with which each pass reads it at its sample.
    /// `Unsupported{PixelCoverageAtSampleRate}` needing that feature.
    InterpolationFunctions,
}

impl CoverageLoopRefusal {
    /// The refusal link and the compile layer return for it.
    pub(crate) fn error(self) -> Error {
        use crate::device::requirements::VkFeature;
        use crate::error::{Capability, Feature, Unsupported};
        let f = Feature::PixelCoverageAtSampleRate;
        match self {
            CoverageLoopRefusal::EarlyFragmentTests => Unsupported::not_representable(f),
            CoverageLoopRefusal::Discard
            | CoverageLoopRefusal::QuadOperations
            | CoverageLoopRefusal::Attachments => Unsupported::not_implemented(f),
            CoverageLoopRefusal::InterpolationFunctions => Unsupported::needs(
                f,
                Capability::Feature(VkFeature::ShaderSampleRateInterpolationFunctions),
            ),
        }
        .into()
    }
}

impl StageIo {
    /// Whether the function has no stage interface.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
            && self.outputs.is_empty()
            && self.builtins.is_empty()
            && self.pulled.is_empty()
            && self.stage_in.is_empty()
    }

    /// Every variable, inputs first.
    pub fn variables(&self) -> impl Iterator<Item = &IoVariable> {
        self.inputs.iter().chain(&self.outputs)
    }

    /// Every vertex attribute the function reads, by attribute index, with its declared numeric
    /// class and the fetch it was compiled under: fixed function for an `Attribute` input
    /// variable, the listed fetch for a pulled one.
    pub fn attributes(&self) -> BTreeMap<u32, (IoNumeric, VertexFetch)> {
        let mut out = BTreeMap::new();
        for v in &self.inputs {
            if let IoRole::Attribute {
                index,
                narrow_float,
            } = v.role
            {
                out.insert(
                    index,
                    (v.ty.numeric, VertexFetch::FixedFunction { narrow_float }),
                );
            }
        }
        for p in &self.pulled {
            out.insert(p.index, (p.ty.numeric, p.fetch));
        }
        out
    }

    /// The type variable `v` is declared with: its `ty`, but for a color output of dual-source
    /// index 0 to an attachment among `extended_range_colors` the raw texel's, the raw format's
    /// unsigned 32-bit components, and to one among `word_colors` the word's, one unsigned 32-bit
    /// component.
    pub(crate) fn declared_type(&self, v: &IoVariable) -> IoType {
        let IoRole::Color {
            attachment,
            index: 0,
        } = v.role
        else {
            return v.ty;
        };
        if self.word_colors.contains_key(&attachment) {
            return IoType {
                numeric: IoNumeric::Uint,
                components: 1,
                source_bits: 32,
                bits: 32,
            };
        }
        match self.extended_range_colors.get(&attachment) {
            Some(e) => IoType {
                numeric: IoNumeric::Uint,
                components: e.form.raw_components() as u8,
                source_bits: 32,
                bits: 32,
            },
            None => v.ty,
        }
    }
}

/// A vertex attribute fetched in the shader: it has no interface variable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PulledInput {
    /// The Metal attribute index.
    pub index: u32,
    /// The type the shader reads the attribute as.
    pub ty: IoType,
    /// The projection entry the function was compiled under (`VertexFetch::Pulled`).
    pub fetch: VertexFetch,
}

impl PulledInput {
    /// Attribute `index` of type `ty`, fetched by `fetch`.
    pub fn new(index: u32, ty: IoType, fetch: VertexFetch) -> PulledInput {
        PulledInput { index, ty, fetch }
    }
}

/// A kernel `[[stage_in]]` attribute: fetched in the shader, like a pulled vertex attribute.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInInput {
    /// The Metal attribute index.
    pub index: u32,
    /// The type the shader reads the attribute as.
    pub ty: IoType,
    /// The stage-input descriptor's entries the function was compiled under.
    pub fetch: StageInFetch,
}

impl StageInInput {
    /// Attribute `index` of type `ty`, fetched by `fetch`.
    pub fn new(index: u32, ty: IoType, fetch: StageInFetch) -> StageInInput {
        StageInInput { index, ty, fetch }
    }
}

/// How one kernel `[[stage_in]]` attribute is fetched: its entries of the stage-input descriptor,
/// canonical as [`StageInputDescriptor::canonical`] writes them. Everything that changes the fetch
/// code is here; built with [`StageInFetch::of`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageInFetch {
    pub format: metal::AttributeFormat,
    /// The Metal buffer index (in the kernel's buffer argument table) the attribute's layout
    /// reads.
    pub buffer_index: u32,
    /// The attribute's byte offset in an element.
    pub offset: u32,
    /// The layout's stride: `None` for one of `MTLBufferLayoutStrideDynamic`, which the kernel
    /// reads per dispatch (`PushField::StageInStride`); `Some(0)` for a constant step, which
    /// reads no element.
    pub stride: Option<u32>,
    pub step_function: metal::StepFunction,
    /// The layout's step rate; 0 for a constant step.
    pub step_rate: u32,
    /// The index type and the Metal buffer index of the index buffer of an indexed step; `None`
    /// for any other step.
    pub index: Option<(metal::IndexType, u32)>,
}

impl StageInFetch {
    /// The fetch of attribute `attribute` under descriptor `d`: an attribute the descriptor lacks
    /// is `InvalidDescriptor(StageInputAttribute)`, one whose layout it lacks
    /// `InvalidDescriptor(StageInputLayout)`, and one whose layout steps by a vertex stage's step
    /// function `InvalidDescriptor(StageInputStepFunction)`, as Metal's pipeline creation stops
    /// it.
    pub fn of(d: &StageInputDescriptor, attribute: u32) -> Result<StageInFetch, Error> {
        use metal::StepFunction as S;
        let a = d
            .attributes
            .get(&attribute)
            .ok_or(Error::InvalidDescriptor(
                DescriptorError::StageInputAttribute { attribute },
            ))?;
        let l = d
            .layouts
            .get(&a.buffer_index)
            .ok_or(Error::InvalidDescriptor(
                DescriptorError::StageInputLayout {
                    layout: a.buffer_index,
                },
            ))?;
        if crate::pipeline::descriptor::vertex_step(l.step_function) {
            return Err(Error::InvalidDescriptor(
                DescriptorError::StageInputStepFunction {
                    layout: a.buffer_index,
                    step_function: l.step_function,
                },
            ));
        }
        let constant = l.step_function == S::Constant;
        let indexed = matches!(
            l.step_function,
            S::ThreadPositionInGridXIndexed | S::ThreadPositionInGridYIndexed
        );
        Ok(StageInFetch {
            format: a.format,
            buffer_index: a.buffer_index,
            offset: a.offset,
            stride: if constant { Some(0) } else { l.stride },
            step_function: l.step_function,
            step_rate: if constant { 0 } else { l.step_rate },
            index: indexed.then_some((d.index_type, d.index_buffer_index)),
        })
    }
}

/// One user interface variable.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IoVariable {
    /// The relocation key of its `Location`.
    pub key: VaryingKey,
    pub role: IoRole,
    pub ty: IoType,
}

/// What an interface variable is.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IoRole {
    /// Vertex attribute `index`, fetched by fixed-function vertex input. `narrow_float` is the
    /// `VertexFetch::FixedFunction` bit the function was compiled under: the Metal format's
    /// components are floats narrower than 32 bits, whose widening the function canonicalizes
    /// where it reads the attribute as `float`.
    Attribute { index: u32, narrow_float: bool },
    /// A vertex output or fragment input matched by its AIR user key.
    Varying {
        user: AirUserKey,
        /// The fragment input's interpolation; `None` on a vertex output (Vulkan takes the
        /// fragment side's decorations).
        interpolation: Option<Interpolation>,
    },
    /// Fragment color output: location `attachment`, dual-source `Index` `index` (0 or 1).
    Color { attachment: u32, index: u32 },
    /// The `[[viewport_array_index]]` a vertex function writes, as written: a vertex output, and
    /// the flat input of a fragment function compiled under
    /// `LoweringInputs::viewport_index_varying` that reads `[[viewport_array_index]]`. The vertex
    /// function's `ViewportIndex` holds the index only below the draw's viewport count (0 at or
    /// past it, where Metal draws through viewport 0), and Metal's fragment function reads the
    /// index as written.
    ViewportIndex,
    /// The `[[render_target_array_index]]` a vertex function writes, as written: a vertex output
    /// in every variant, and the flat input of a fragment function compiled under
    /// `LoweringInputs::render_target_array_index_varying` that reads
    /// `[[render_target_array_index]]`. The vertex function's `Layer` holds the index only below
    /// the render pass's `renderTargetArrayLength` (0 at or past it, where Metal renders the
    /// pass's first layer), and its variant for a pass of one layer writes no `Layer`; Metal's
    /// fragment function reads the index as written in both.
    RenderTargetArrayIndex,
    /// A mesh function's per-primitive output (`air.mesh_primitive_data`), matched with a
    /// fragment input by its AIR user key like a [`IoRole::Varying`]: the module declares it one
    /// element per primitive, decorated `PerPrimitiveEXT`. The fragment input that reads it is a
    /// flat [`IoRole::Varying`] its function reads per primitive
    /// ([`StageIo::primitive_inputs`]).
    PrimitiveVarying { user: AirUserKey },
    /// The window depth `z / w` of the position a vertex function compiled under
    /// `LoweringInputs::window_depth_varying` stores: a vertex output, and the no-perspective
    /// input of a fragment function that runs its draw's depth test itself at more than one
    /// sample, which reads it at its sample and at the pixel centre to compute each sample's
    /// depth (`lower::fragment`'s `depth_test`).
    WindowDepth,
    /// The first two barycentric weights a fragment function compiled for a device without
    /// `fragmentShaderBarycentric` reads as its `[[barycentric_coord]]`: a center-sampled `float2`
    /// input, perspective-correct when `perspective`, which no Metal function writes. Link
    /// synthesizes the pipeline's geometry stage, which writes `(1, 0)`, `(0, 1)` and `(0, 0)` at
    /// a triangle's vertices in their order (`(1, 0)` and `(0, 1)` at a line's, `(1, 0)` at a
    /// point), and passes the fragment stage's other inputs through.
    Barycentric { perspective: bool },
    /// The `Layer` a vertex function compiled for a device without `shaderOutputLayer` selects its
    /// primitive's render target slice with: the value it would store to `Layer` (the index it
    /// writes, plus the amplification's offset when drawn one amplification at a time, held below
    /// the render pass's length and 0 at or past it), as a flat `uint` vertex output. No fragment
    /// function reads it: the pipeline's synthesized geometry stage stores it to `Layer` at every
    /// vertex it emits, from the primitive's first vertex, whose index Metal renders the primitive
    /// with.
    SelectedLayer,
    /// The `ViewportIndex` a vertex function compiled for a device with `multiViewport` but
    /// without `shaderOutputViewportIndex` selects its primitive's viewport with, as
    /// [`IoRole::SelectedLayer`] is for its slice: held below the draw's viewport count and 0 at
    /// or past it.
    SelectedViewport,
    /// Metal's `[[primitive_id]]` carried from a vertex function to its fragment function: the
    /// flat `uint` output of a vertex function compiled under
    /// `LoweringInputs::primitive_id_varying`, and the input a fragment function reads as its
    /// `[[primitive_id]]` on a device that cannot declare a fragment `PrimitiveId` (neither
    /// `geometryShader` nor `tessellationShader`) or under that field. A plain vertex function
    /// writes at each vertex the index of the primitive the vertex provokes: a flat input reads the
    /// primitive's provoking vertex, its first in a pipeline of Vulkan's default provoking vertex
    /// (the Vulkan specification's primitive topologies),
    /// whose stream position is the primitive's own times its vertex count in a list and the
    /// primitive's own in a strip, so the vertex writes its stream position over the draw's vertex
    /// count per primitive (`PushField::VerticesPerPrimitive`): Metal's count of the draw's
    /// primitives in each instance (Metal probes on the M3 Max, macOS 27.0.1). A post-tessellation
    /// vertex function emulated on a device without `tessellationShader`
    /// (`GraphicsState::emulated_tessellation`) writes at every vertex of a patch the patch's
    /// position in its draw (`patchStart + i`), Metal's `[[primitive_id]]` behind the tessellator.
    PrimitiveId,
}

impl IoRole {
    /// Whether the role is a varying: a user varying, a raw index a vertex function writes for
    /// its fragment function, its window depth, the barycentric weights a synthesized geometry
    /// stage writes, a selected index the geometry stage reads, or the primitive ID. Link numbers
    /// these, and packs those a vertex function writes.
    pub(crate) fn is_varying(&self) -> bool {
        matches!(
            self,
            IoRole::Varying { .. }
                | IoRole::ViewportIndex
                | IoRole::RenderTargetArrayIndex
                | IoRole::WindowDepth
                | IoRole::Barycentric { .. }
                | IoRole::SelectedLayer
                | IoRole::SelectedViewport
                | IoRole::PrimitiveId
        )
    }
}

/// Interpolation of a fragment input.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Interpolation {
    Flat,
    Smooth {
        sampling: Sampling,
        perspective: bool,
    },
}

/// Where a smooth fragment input is sampled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
}

/// The declared type of an interface variable: `components` lanes (1–4) of `numeric`, each of
/// `bits` bits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IoType {
    pub numeric: IoNumeric,
    pub components: u8,
    /// The AIR element width (8, 16, or 32; 16 for `half`; 1 for `bool`, carried as `uint` 0 or
    /// 1): narrower values are widened at the interface.
    pub source_bits: u8,
    /// The declared lane width: 32, or 16 for a `float` color output written as the bits of the
    /// half Metal stores into a half-float attachment (`storageInputOutput16`).
    pub bits: u8,
}

/// The numeric class the variable is declared with.
///
/// Integers are declared unsigned (the engine holds every integer zero-extended in an unsigned
/// container). A fixed-function attribute of a signed-integer format therefore feeds a `uint`
/// variable, which Vulkan's vertex-input rule does not match exactly; link decides. The one signed
/// variable is a color output to a signed-integer attachment, which Vulkan writes only from a
/// signed output (`LoweringInputs::signed_color_attachments`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IoNumeric {
    Float,
    Uint,
    Sint,
}

/// A builtin a stage module declares, with what it tells the pipeline.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StageBuiltin {
    // ---- Vertex ----
    VertexIndex,
    InstanceIndex,
    /// `Position`; `invariant`: decorated `Invariant` (Metal `[[invariant]]`).
    Position {
        invariant: bool,
    },
    /// `PointSize`; `default`: the function has no point-size output, so the module writes 1.0.
    PointSize {
        default: bool,
    },
    /// `ClipDistance` with this many elements.
    ClipDistance {
        count: u32,
    },
    /// `Layer` (a vertex output, or a fragment input). A vertex function stores it as 0 for an
    /// index at or past the render pass's `renderTargetArrayLength`
    /// (`PushField::RenderTargetArrayLength`).
    Layer,
    /// `ViewportIndex` (a vertex output, or a fragment input). A vertex function stores it as 0
    /// for an index at or past the draw's viewport count (`RuntimeFact::ViewportCount`).
    ViewportIndex,
    // ---- Fragment ----
    FragCoord,
    FrontFacing,
    PointCoord,
    SampleId,
    SampleMaskIn,
    PrimitiveId,
    /// `FragDepth` with its `DepthReplacing` mode.
    FragDepth(DepthMode),
    StencilRef,
    SampleMaskOut,
    /// `BaseInstance` (`shaderDrawParameters`): a pulled per-instance attribute with a step rate
    /// above 1 reads it, and a vertex function's `[[base_instance]]` is it.
    BaseInstance,
    /// `BaryCoordKHR` (`perspective`) or `BaryCoordNoPerspKHR`: Metal's `[[barycentric_coord]]`
    /// (`fragmentShaderBarycentric`).
    BaryCoord {
        perspective: bool,
    },
    /// `ViewIndex` (`multiview`): Metal's `[[amplification_id]]` in a vertex or fragment function.
    ViewIndex,
    /// `BaseVertex` (`shaderDrawParameters`): a vertex function's `[[base_vertex]]`, the draw's
    /// first vertex without an index buffer and its vertex offset with one.
    BaseVertex,
}

/// How a fragment function replaces depth.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthMode {
    Any,
    Greater,
    Less,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No user key can equal an attribute or color key: the namespaces differ in the first byte.
    #[test]
    fn key_namespaces_are_disjoint() {
        let attr = VaryingKey::attribute(3);
        let color = VaryingKey::color(3, 0);
        let viewport = VaryingKey::viewport_index();
        let layer = VaryingKey::render_target_array_index();
        let depth = VaryingKey::window_depth();
        let perspective = VaryingKey::barycentric(true);
        let linear = VaryingKey::barycentric(false);
        let tagged = [
            &attr,
            &color,
            &viewport,
            &layer,
            &depth,
            &perspective,
            &linear,
        ];
        for (i, a) in tagged.iter().enumerate() {
            for b in &tagged[i + 1..] {
                assert_ne!(a, b);
            }
        }
        for k in [
            &attr.0 .0[..],
            &color.0 .0[..],
            &viewport.0 .0[..],
            &layer.0 .0[..],
            b"",
            b"\x01\x03\0\0\0",
        ] {
            let u = VaryingKey::user(k);
            for t in tagged {
                assert_ne!(&u, t);
            }
            assert_eq!(u.0 .0[0], USER);
        }
        assert_ne!(VaryingKey::color(1, 0), VaryingKey::color(1, 1));
        assert_eq!(VaryingKey::user(b"uv"), VaryingKey::user(b"uv"));
    }
}
