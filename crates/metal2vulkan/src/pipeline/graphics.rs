//! `GraphicsState`: the shader-coupled fixed-function state of a render pipeline.
//!
//! It carries the Metal-to-Vulkan mapping of the descriptor's blend, sample, and topology state as
//! plain `vk::` values, the states that must be dynamic on the profile, and the framebuffer-fetch
//! render-pass contract. Metal's depth/stencil state is a separate encoder object and is mapped by
//! `metal`, not here.

use std::collections::BTreeSet;

use crate::device::requirements::{VkFeature, VkProperty};
use crate::error::{
    Attachment, Capability, Error, Feature, InternalError, InternalKind, Unsupported,
};
use crate::metal;
use crate::pipeline::interface::PatchDomain;
use crate::pipeline::sym::SampledType;
use crate::vk;

/// Fixed-function state of a render pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphicsState {
    /// The list topology of the descriptor's topology class (`PATCH_LIST` for tessellation). The
    /// topology is never dynamic state: the consumer creates the pipeline with the draw's topology
    /// within the class and the primitive restart Metal applies to it, both from
    /// [`metal::state::input_assembly`].
    pub topology: vk::PrimitiveTopology,
    /// Color attachments by Metal index; `None` for an absent attachment. Build
    /// `VkPipelineColorBlendStateCreateInfo::pAttachments` from
    /// [`color_blend_attachments`](GraphicsState::color_blend_attachments), which also states
    /// the elements of absent attachments.
    pub color_attachments: Vec<Option<ColorAttachmentState>>,
    pub depth_format: Option<vk::Format>,
    pub stencil_format: Option<vk::Format>,
    pub rasterization_samples: vk::SampleCountFlags,
    /// The descriptor's `alphaToCoverageEnabled`, kept only where the fragment function has a
    /// four-component float first output at attachment 0 whose alpha it reads: without one Metal
    /// leaves the coverage as rasterized, and Vulkan requires that output. The device covers by
    /// its own algorithm: where it is not measured to cover as Metal does, the fragment stage
    /// records `PrecisionNote::DeviceAlphaCoverage`.
    pub alpha_to_coverage: bool,
    /// The descriptor's `alphaToOneEnabled`, as
    /// `VkPipelineMultisampleStateCreateInfo::alphaToOneEnable`: both replace the fragment's
    /// alpha with one before the blend reads it and before the write mask decides what is stored,
    /// and neither changes the alpha [`alpha_to_coverage`](GraphicsState::alpha_to_coverage)
    /// reads. The pipeline's requirements name `alphaToOne`.
    pub alpha_to_one: bool,
    /// `rasterizationEnabled = NO`: the pipeline has no fragment stage and discards primitives.
    pub rasterizer_discard: bool,
    /// State the consumer must make dynamic (or turn into its own pipeline variants).
    pub dynamic_states: Vec<vk::DynamicState>,
    /// Framebuffer-fetch inputs. In a pipeline of more than one sample the fragment function runs
    /// once per covered sample and reads its own sample of each.
    pub input_attachments: Vec<InputAttachmentUse>,
    /// Metal's `maxVertexAmplificationCount` (1 without vertex amplification): a draw with this
    /// pipeline amplifies each vertex by a count in `[1, max_vertex_amplification_count]`, or,
    /// above 1, by a larger count too, as plain Metal draws one. Above 1 the pipeline is a
    /// multiview pipeline: the consumer creates it, and begins its render pass instance, with the
    /// view mask its render plan states (`BindingPlan::views`).
    pub max_vertex_amplification_count: u32,
    /// A stage declares `Layer`: the vertex function chooses its render target slice, or the
    /// fragment function reads it. Such a pipeline cannot be multiview
    /// (`VUID-VkGraphicsPipelineCreateInfo-renderPass-06050`, `-06059`), so the binder refuses its
    /// draws at an amplification count above 1.
    pub declares_layer: bool,
    /// The tessellation state of a pipeline whose vertex function is a post-tessellation vertex
    /// function; `None` otherwise. Such a pipeline's `topology` is `PATCH_LIST` (never dynamic) and
    /// its stages are the synthesized vertex and tessellation control stages, then the function's
    /// evaluation stage, then the fragment stage.
    pub tessellation: Option<TessellationState>,
    /// The vertex function (a post-tessellation vertex function included) writes memory
    /// (`StageInterface::writes_memory`): each run of the pipeline's vertex stage repeats its
    /// side effects, so a rate-mapped pass drawn region by region, which records each draw once
    /// per region, cannot draw such a pipeline in more than one region.
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_writes_memory: bool,
    /// The vertex function (a post-tessellation vertex function included) writes `ViewportIndex`
    /// (`[[viewport_array_index]]`): each primitive selects its viewport and scissor from the
    /// encoder's arrays, so a rate-mapped pass drawn region by region, which sets one viewport per
    /// region, cannot draw such a pipeline.
    /// Set too on a device without `multiViewport`, where the function stores no `ViewportIndex`
    /// (every index selects viewport 0 of a draw's one, as Metal's does at or past the count):
    /// there a draw of the pipeline while the encoder holds more than one viewport cannot select
    /// one, and the runtime refuses it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_viewport_index: bool,
    /// A render pipeline variant that draws its lines as bands of triangles
    /// (`VariantKey::line_band`, `pipeline::link::band`): its `topology` is `PATCH_LIST` (never
    /// dynamic), its stages the vertex stage, the band's synthesized tessellation control and
    /// evaluation stages, then the fragment stage. `None` for every other pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub line_band: Option<LineBandState>,
    /// The vertex function writes `[[point_size]]` (as its compiled function's
    /// `writes_point_size` answers; a post-tessellation vertex function's is dropped and
    /// answers `false`). Metal sizes a point
    /// drawn by a pipeline whose function writes none 1 only under the point input primitive
    /// topology class; under any other class it rasterizes nothing, the vertex function run all
    /// the same (Metal probes, an M4). The module writes 1.0 for such a function, so a consumer
    /// drawing points with it under another class keeps them from rasterizing (the runtime draws
    /// through empty scissors).
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_point_size: bool,
    /// Whether the fragment function writes its depth (`[[depth(...)]]`). A draw in depth clamp
    /// mode through a range above 0 runs such a function's `LoweringInputs::depth_clamp_range`
    /// variant where it can, as Metal stores a negative NaN depth as 0 past the range.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_depth: bool,
    /// Whether the fragment function reads `[[barycentric_coord]]`, perspective-correct or not
    /// and of any width (`float`, `float2`, `float3`), from the translated fragment stage's
    /// builtins (`BaryCoordKHR`, `BaryCoordNoPerspKHR`). Metal reads the weights at the pixel center (MSL 5.2.3.4), under a `centroid` or
    /// `sample` qualifier too, and so does the stage. `false` for a pipeline without a fragment
    /// function. A primitive's weights follow the order its vertices are drawn in, so
    /// a consumer that draws them in another order can change what a draw shows where this is
    /// set, as it can where the fragment function reads a flat input.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reads_barycentric_coord: bool,
    /// Whether the fragment function reads `[[primitive_id]]`: Vulkan's `PrimitiveId`, or the
    /// varying its vertex function computes it into on a device that cannot declare it
    /// (`IoRole::PrimitiveId`). `false` for a pipeline without a fragment function. A consumer
    /// that draws one Metal primitive as several Vulkan ones (Metal's quads, one primitive ID a
    /// quad, as two triangles) changes what a draw shows only where this is set.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reads_primitive_id: bool,
    /// The fragment function writes memory (`StageInterface::writes_memory`): it stores to a
    /// buffer or a texture, or updates one atomically. `false` for a pipeline without a fragment
    /// function. Metal never runs a function that declares `[[early_fragment_tests]]` for a
    /// fragment its depth or stencil test fails; a consumer that stands for those tests by a
    /// state every fragment passes, its color writes masked, runs such a function only where
    /// this is `false` (and the function writes no attachment as a storage image), since the
    /// run then leaves nothing another invocation or a later command reads.
    #[cfg_attr(feature = "serde", serde(default))]
    pub fragment_writes_memory: bool,
    /// The color attachments the fragment function may write, as a mask: bit `n` for color
    /// attachment `n`. Each attachment that a first (`[[color(n)]]`, index 0) output of the
    /// function names is set. It is decided from the stage's outputs, which come from the AIR's output metadata of the entry its
    /// function constants specialized, never from names.
    ///
    /// An attachment not set is one the function never writes:
    /// - no output names it, or a function constant gates its output off;
    /// - a dual-source second output (`[[color(n), index(1)]]`) alone names it, which writes
    ///   nothing by itself;
    /// - the pass the variant was compiled for lacks it (`LoweringInputs::dropped_colors`),
    ///   where plain Metal drops the write;
    /// - the pipeline has no attachment there and the function was compiled for it
    ///   (`LoweringInputs::unattached_colors`), where Metal discards the write.
    ///
    /// One set may still go unwritten, since an output the function leaves undefined is written
    /// all the same. An index the pipeline has no attachment for is set where a first output
    /// names it in a function compiled without that input, though Metal discards that write;
    /// `Device::compile_render_pipeline` compiles every `Function` fragment stage with it, so only
    /// color 0 under alpha to coverage, whose alpha the coverage reads, stays set there. An
    /// attachment the variant zeroes for a pass the pipeline lacks
    /// (`LoweringInputs::zeroed_colors`) is set, since the stage writes zero to it. 0 for a
    /// pipeline without a fragment function. The attachment's write mask is separate:
    /// [`color_attachments`](GraphicsState::color_attachments) gives an attachment that no first
    /// output names an empty mask.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_outputs: u32,
    /// `VkPipelineMultisampleStateCreateInfo::pSampleMask`'s one word: the descriptor's private
    /// sample mask ([`PrivateRenderState::sample_mask`](crate::pipeline::PrivateRenderState::sample_mask))
    /// at the pipeline's samples, which Metal ANDs with the coverage of a pipeline with a
    /// fragment function, ignoring the bits past the sample count. `None` where it keeps every
    /// sample, where the pipeline has no fragment function, and where the fragment function applies
    /// the mask itself (`StageIo::sample_mask`), as one that writes memory, reads `[[sample_mask]]`
    /// or declares `[[early_fragment_tests]]` does; the consumer then passes no mask.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sample_mask: Option<u32>,
    /// `VkPipelineColorBlendStateCreateInfo::logicOp` with `logicOpEnable`: the descriptor's
    /// private `logicOperation`
    /// ([`PrivateRenderState::logic_operation`](crate::pipeline::PrivateRenderState::logic_operation)),
    /// which Metal applies to the codes of each normalized and integer attachment as Vulkan
    /// does, in a pipeline whose float attachments do not blend and that has no sRGB
    /// attachment; the pipeline's requirements name `logicOp`. Copy and no-op need none: the
    /// attachments' states already draw them. `None` with the logic operation off, and for those.
    #[cfg_attr(feature = "serde", serde(default))]
    pub logic_op: Option<vk::LogicOp>,
}

/// What a consumer and the binder need of a tessellation pipeline.
///
/// Metal draws `patchCount` patches of `instanceCount` instances from `patchStart` and
/// `baseInstance`; the pipeline draws each patch as one vertex of a `PATCH_LIST` of
/// `patch_control_points` (1) control point: its vertex stage passes Vulkan's vertex index (the
/// patch's position, `patchStart` included) and instance index (the base instance included) to
/// its control stage, which reads the patch's tessellation factors at the position from the
/// encoder's factor buffer (`SyntheticBinding::TessellationFactors`) and hands Metal's patch
/// index (the position, or the draw's patch index buffer's word there,
/// `SyntheticBinding::PatchIndices`), the instance index, and each control point's element (the
/// patch index times `control_points` plus the point's, or the control-point index buffer's
/// index there, `SyntheticBinding::ControlPointIndices`) to the evaluation stage.
///
/// Create the pipeline in Vulkan's default tessellation domain origin, `UPPER_LEFT`: chain no
/// `VkPipelineTessellationDomainOriginStateCreateInfo` other than one naming it. The control
/// stage's vertex order and the evaluation stage's `[[position_in_patch]]` map Metal's winding
/// and patch coordinates under that origin; `LOWER_LEFT` reverses the winding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationState {
    /// `VkPipelineTessellationStateCreateInfo::patchControlPoints`.
    pub patch_control_points: u32,
    /// The post-tessellation vertex function's patch type.
    pub domain: PatchDomain,
    /// Its `[[patch(_, N)]]` control points per patch (0 when unstated): a patch draw's
    /// `numberOfPatchControlPoints` must equal a stated count.
    pub control_points: u32,
    /// The descriptor's `tessellationFactorStepFunction`: which of the draw's patch and instance
    /// indices address a patch's factors.
    pub factor_step_function: metal::TessellationFactorStepFunction,
    /// Byte offset of the control stage's push range (`PipelineLayout::push_constants`, stage
    /// `TESSELLATION_CONTROL`): five words, the factor buffer's residual bind offset, its
    /// instance stride, the factor scale, the patch index buffer's residual bind offset, and
    /// whether the draw binds one; and a sixth, the control-point index buffer's residual bind
    /// offset, for a pipeline with a control-point index type. Under the identity domains the
    /// control stage binds no buffer and reads the factors through their address: its fields
    /// (`PushField::TessellationFactorAddress`, and where read the instance stride and the scale)
    /// are a vertex-stage placement of `PipelineInterface::push` from this offset, and it reads
    /// no patch index buffer.
    pub push_offset: u32,
    /// The descriptor's `tessellationControlPointIndexType`: other than `None`, the pipeline
    /// draws indexed patches only (`DrawCall::IndexedPatches`), whose control-point index buffer
    /// (`SyntheticBinding::ControlPointIndices`) its control stage reads; `None`, plain patches
    /// only.
    pub control_point_index_type: metal::TessellationControlPointIndexType,
}

/// What each patch of a render pipeline variant that draws bands is (`VariantKey::line_band`,
/// `LineBandState::primitive`): Apple's `setLineWidth:` at a width the device's own lines do not
/// draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BandPrimitive {
    /// A line: its two vertices, drawn as the parallelogram Metal draws for it.
    Lines,
    /// A triangle Metal draws in fill mode `Lines`: its three vertices, whose three edges are
    /// each drawn as the parallelogram Metal draws for it.
    Triangles,
}

impl BandPrimitive {
    /// `VkPipelineTessellationStateCreateInfo::patchControlPoints`: the primitive's vertices.
    pub fn patch_control_points(self) -> u32 {
        match self {
            BandPrimitive::Lines => 2,
            BandPrimitive::Triangles => 3,
        }
    }
}

/// What a consumer and the binder need of a render pipeline variant that draws its lines, or its
/// triangles' edges, as bands (`GraphicsState::line_band`): Apple's `setLineWidth:` at a width the
/// device's own lines do not draw.
///
/// Draw each primitive as one patch of [`BandPrimitive::patch_control_points`] control points, its
/// vertices in order, with primitive restart off: a line list or a triangle list as it is; a strip,
/// a loop, a fan, Metal's private triangle types or a list under Apple's last provoking vertex as
/// the list of its primitives the runtime writes. Draw in fill mode. Lines
/// ([`BandPrimitive::Lines`]): the band's two triangles cover the parallelogram Metal draws for the
/// line; draw them with no culling and with front face [`FRONT_FACE`](Self::FRONT_FACE), so they
/// are front-facing as a line is, and without depth bias, which Metal does not apply to lines.
/// Triangles ([`BandPrimitive::Triangles`]): the band's triangles cover the three parallelograms
/// Metal draws for the edges, and face as the triangle does; draw them with the draw's own cull
/// mode, front face and depth bias, which Metal applies to the triangle (its slope and largest
/// depth's exponent; the device biases each band triangle by its own). Push the band's
/// `[sx, sy, h, f]` (`f32`s; `PushField::LineBand`) at [`push_offset`](Self::push_offset) for
/// stage `TESSELLATION_EVALUATION`: the half extent of the draw's Vulkan viewport 0, its height
/// negative as a Y-flipped viewport's is, half the width in framebuffer pixels, and for triangles
/// 1 where the draw's front face is `CLOCKWISE` and -1 where it is `COUNTER_CLOCKWISE` (0 for
/// lines; the binder writes it from `BindState::line_band`). Create the pipeline in Vulkan's
/// default tessellation domain origin, `UPPER_LEFT`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineBandState {
    /// Byte offset of the evaluation stage's push range (`PipelineLayout::push_constants`, stage
    /// `TESSELLATION_EVALUATION`, [`PUSH_BYTES`](Self::PUSH_BYTES)).
    pub push_offset: u32,
    /// What each patch is.
    pub primitive: BandPrimitive,
}

impl LineBandState {
    /// The push range's bytes: one `vec4`.
    pub const PUSH_BYTES: u32 = crate::pipeline::link::band::LineBand::PUSH_BYTES;
    /// The front face a draw of a band of lines sets, under which its triangles are front-facing.
    pub const FRONT_FACE: vk::FrontFace = crate::pipeline::link::band::LineBand::FRONT_FACE;
}

impl GraphicsState {
    /// A state that draws `topology` at `rasterization_samples` and nothing else: no attachment,
    /// dynamic state, tessellation, imageblock, fetch or amplification (one view). A consumer
    /// that builds a state itself, for a test or a pipeline it describes, assigns the rest.
    pub fn new(
        topology: vk::PrimitiveTopology,
        rasterization_samples: vk::SampleCountFlags,
    ) -> GraphicsState {
        GraphicsState {
            topology,
            color_attachments: Vec::new(),
            depth_format: None,
            stencil_format: None,
            rasterization_samples,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterizer_discard: false,
            dynamic_states: Vec::new(),
            input_attachments: Vec::new(),
            max_vertex_amplification_count: 1,
            declares_layer: false,
            tessellation: None,
            vertex_writes_memory: false,
            writes_viewport_index: false,
            line_band: None,
            writes_point_size: false,
            writes_depth: false,
            reads_barycentric_coord: false,
            reads_primitive_id: false,
            fragment_writes_memory: false,
            color_outputs: 0,
            sample_mask: None,
            logic_op: None,
        }
    }

    /// The features the pipeline's dynamic states need: `extendedDynamicState` for any state
    /// `VK_EXT_extended_dynamic_state` defines, and `extendedDynamicState2` for
    /// `VK_EXT_extended_dynamic_state2`'s primitive restart, which Vulkan 1.3 makes core without a
    /// feature member (`device::Exposure::Implied`), so a 1.3 device has them and a 1.2 device
    /// needs the extension and the feature enabled
    /// (`VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03378`, `-04868`). The Vulkan 1.0
    /// states need nothing. A state outside these is one link never makes: an invariant error,
    /// never a guessed need.
    pub(crate) fn dynamic_state_features(&self) -> Result<BTreeSet<VkFeature>, Error> {
        let mut features = BTreeSet::new();
        for &state in &self.dynamic_states {
            if let Some(feature) = dynamic_state_feature(state)? {
                features.insert(feature);
            }
        }
        Ok(features)
    }

    /// The write mask and blend state of each element of
    /// `VkPipelineColorBlendStateCreateInfo::pAttachments`, by Metal index, one per entry of
    /// `color_attachments`: a present attachment's own, and an absent attachment's (its subpass
    /// reference is `VK_ATTACHMENT_UNUSED`, so the element writes nothing) the first present
    /// attachment's. On a device without `independentBlend` link makes every present
    /// attachment's identical, so every element is identical
    /// (`VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605`); the pipeline's requirements
    /// name `independentBlend` exactly when the elements differ.
    pub fn color_blend_attachments(&self) -> Vec<(vk::ColorComponentFlags, Option<BlendState>)> {
        let present = self.color_attachments.iter().flatten().next();
        self.color_attachments
            .iter()
            .map(|a| match a.as_ref().or(present) {
                Some(a) => (a.write_mask, a.blend),
                None => (vk::ColorComponentFlags::empty(), None),
            })
            .collect()
    }

    /// The features the pipeline's color blend state needs: `independentBlend` when the elements of
    /// [`color_blend_attachments`](GraphicsState::color_blend_attachments) differ (core 1.0,
    /// required to be supported only from 1.4, and enabled either way), and `dualSrcBlend` when a
    /// blend reads the fragment's second output
    /// (`VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-00608` and its siblings).
    pub(crate) fn color_blend_features(&self) -> BTreeSet<VkFeature> {
        let mut features = BTreeSet::new();
        let elements = self.color_blend_attachments();
        if elements.windows(2).any(|pair| pair[0] != pair[1]) {
            features.insert(VkFeature::IndependentBlend);
        }
        let dual = self
            .color_attachments
            .iter()
            .flatten()
            .filter_map(|a| a.blend)
            .any(|b| b.reads_second_output());
        if dual {
            features.insert(VkFeature::DualSrcBlend);
        }
        features
    }

    /// The features the pipeline's multisample state needs: `alphaToOne` for
    /// [`alpha_to_one`](GraphicsState::alpha_to_one)
    /// (`VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785`).
    pub(crate) fn multisample_features(&self) -> BTreeSet<VkFeature> {
        let mut features = BTreeSet::new();
        if self.alpha_to_one {
            features.insert(VkFeature::AlphaToOne);
        }
        features
    }

    /// The sample-count properties the pipeline's rasterization samples need on a device whose
    /// enabled properties are `enabled`. With more than one sample and rasterization on:
    /// - each color attachment needs a count every color attachment of its kind takes,
    ///   `framebufferColorSampleCounts` (float and fixed-point) or
    ///   `framebufferIntegerColorSampleCounts`, or else one its own format takes
    ///   ([`VkProperty::ColorAttachmentSampleCount`]): a device may report more for one format
    ///   than for every format of its kind;
    /// - a depth or stencil format needs `framebufferDepthSampleCounts` or
    ///   `framebufferStencilSampleCounts` by its components;
    /// - a pipeline without attachments needs `framebufferNoAttachmentsSampleCounts`.
    ///
    /// One sample needs nothing: every format supports it. Each need names the first of its
    /// alternatives `enabled` has; one it has none of is refused as
    /// `Unsupported{RasterSampleCount}` naming every alternative.
    pub(crate) fn sample_count_properties(
        &self,
        enabled: &BTreeSet<VkProperty>,
    ) -> Result<BTreeSet<VkProperty>, Unsupported> {
        let count = self.rasterization_samples.as_raw();
        if count <= 1 || self.rasterizer_discard {
            return Ok(BTreeSet::new());
        }
        let mut needs: Vec<(Option<Attachment>, Vec<VkProperty>)> = Vec::new();
        for (index, a) in self.color_attachments.iter().enumerate() {
            if let Some(a) = a {
                let alternatives = color_sample_count_alternatives(a.format, count).to_vec();
                needs.push((Some(Attachment::Color(index as u32)), alternatives));
            }
        }
        for format in [self.depth_format, self.stencil_format]
            .into_iter()
            .flatten()
        {
            let block = format.block();
            if block.is_some_and(|b| b.depth.is_some()) {
                let depth = VkProperty::FramebufferDepthSampleCount(count);
                needs.push((Some(Attachment::Depth), vec![depth]));
            }
            if block.is_some_and(|b| b.stencil.is_some()) {
                let stencil = VkProperty::FramebufferStencilSampleCount(count);
                needs.push((Some(Attachment::Stencil), vec![stencil]));
            }
        }
        if needs.is_empty() {
            let none = VkProperty::FramebufferNoAttachmentsSampleCount(count);
            needs.push((None, vec![none]));
        }
        let mut properties = BTreeSet::new();
        for (attachment, alternatives) in needs {
            match alternatives.iter().find(|p| enabled.contains(p)) {
                Some(&p) => {
                    properties.insert(p);
                }
                None => {
                    let refused = Unsupported::needs_any_of(
                        Feature::RasterSampleCount,
                        alternatives
                            .into_iter()
                            .map(|p| vec![Capability::Property(p)])
                            .collect(),
                    );
                    return Err(match attachment {
                        Some(attachment) => refused.at_attachment(attachment),
                        None => refused,
                    });
                }
            }
        }
        Ok(properties)
    }
}

/// The properties of which a color attachment of `format` at `count` samples needs one, in the
/// order a need names them: the count every color attachment of its kind takes
/// (`framebufferColorSampleCounts` for float and fixed-point formats,
/// `framebufferIntegerColorSampleCounts` for integer ones), then its own format's.
pub(crate) fn color_sample_count_alternatives(format: vk::Format, count: u32) -> [VkProperty; 2] {
    let integer = matches!(
        format.numeric_type(),
        Some(vk::NumericType::Uint | vk::NumericType::Sint)
    );
    let kind = if integer {
        VkProperty::FramebufferIntegerColorSampleCount(count)
    } else {
        VkProperty::FramebufferColorSampleCount(count)
    };
    [kind, VkProperty::ColorAttachmentSampleCount(format, count)]
}

/// The feature `state` needs as dynamic state (see [`GraphicsState::dynamic_state_features`]).
fn dynamic_state_feature(state: vk::DynamicState) -> Result<Option<VkFeature>, Error> {
    use vk::DynamicState as D;
    Ok(match state {
        D::VIEWPORT
        | D::SCISSOR
        | D::LINE_WIDTH
        | D::DEPTH_BIAS
        | D::BLEND_CONSTANTS
        | D::DEPTH_BOUNDS
        | D::STENCIL_COMPARE_MASK
        | D::STENCIL_WRITE_MASK
        | D::STENCIL_REFERENCE => None,
        D::CULL_MODE
        | D::FRONT_FACE
        | D::PRIMITIVE_TOPOLOGY
        | D::VIEWPORT_WITH_COUNT
        | D::SCISSOR_WITH_COUNT
        | D::VERTEX_INPUT_BINDING_STRIDE
        | D::DEPTH_TEST_ENABLE
        | D::DEPTH_WRITE_ENABLE
        | D::DEPTH_COMPARE_OP
        | D::DEPTH_BOUNDS_TEST_ENABLE
        | D::STENCIL_TEST_ENABLE
        | D::STENCIL_OP => Some(VkFeature::ExtendedDynamicState),
        // `VK_EXT_extended_dynamic_state2`'s first feature (core in Vulkan 1.3 without one), made
        // dynamic only where the profile has it.
        D::PRIMITIVE_RESTART_ENABLE => Some(VkFeature::ExtendedDynamicState2),
        other => {
            return Err(Error::Internal(InternalError::new(
                InternalKind::Invariant,
                format!(
                    "dynamic state {} has no modelled requirement",
                    other.as_raw()
                ),
            )))
        }
    })
}

/// One color attachment's format, write mask, and blend state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorAttachmentState {
    pub format: vk::Format,
    /// The descriptor's write mask, or empty when no first (index 0) output of the fragment
    /// function names the attachment, so that the draw leaves it unmodified. On a device without
    /// `independentBlend` every attachment has one mask and blend state: such an attachment then
    /// takes the mask of the attachments the function writes, which leaves it unmodified all the
    /// same, as no output is declared at its location.
    pub write_mask: vk::ColorComponentFlags,
    /// `None` when blending is disabled.
    pub blend: Option<BlendState>,
}

impl ColorAttachmentState {
    /// A color attachment of `format` that writes the channels of `write_mask`, blended by
    /// `blend` (`None`: written as the fragment returns it).
    pub fn new(
        format: vk::Format,
        write_mask: vk::ColorComponentFlags,
        blend: Option<BlendState>,
    ) -> ColorAttachmentState {
        ColorAttachmentState {
            format,
            write_mask,
            blend,
        }
    }
}

/// Vulkan blend factors and operations of one attachment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendState {
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
}

impl BlendState {
    /// Whether a factor reads the fragment's second output (`[[color(n), index(1)]]`): dual-source
    /// blending.
    pub fn reads_second_output(&self) -> bool {
        [
            self.src_color,
            self.dst_color,
            self.src_alpha,
            self.dst_alpha,
        ]
        .into_iter()
        .any(|f| {
            matches!(
                f,
                vk::BlendFactor::SRC1_COLOR
                    | vk::BlendFactor::ONE_MINUS_SRC1_COLOR
                    | vk::BlendFactor::SRC1_ALPHA
                    | vk::BlendFactor::ONE_MINUS_SRC1_ALPHA
            )
        })
    }

    /// Whether a color factor is the blend constant's alpha (`CONSTANT_ALPHA` or
    /// `ONE_MINUS_CONSTANT_ALPHA`, Metal's `BlendAlpha` factors), which the portability subset
    /// gates on `constantAlphaColorBlendFactors`. The alpha factors are not gated.
    pub(crate) fn constant_alpha_color(&self) -> bool {
        [self.src_color, self.dst_color].into_iter().any(|f| {
            matches!(
                f,
                vk::BlendFactor::CONSTANT_ALPHA | vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA
            )
        })
    }
}

/// A framebuffer-fetch input: color attachment `color_index` read as input attachment
/// `input_attachment_index`. The consumer's render pass provides the self-dependency (or
/// `dynamicRenderingLocalRead` mapping) that the contract names, and the barriers between draws
/// it states (`PrecisionNote::UnorderedFetchWithinDraw`). In a pipeline of more than one
/// sample the attachment is the pass's multisampled one, of `GraphicsState::rasterization_samples`
/// samples, which the function reads at each invocation's own sample.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InputAttachmentUse {
    pub color_index: u32,
    pub input_attachment_index: u32,
    /// The numeric type the shader reads.
    pub format_class: SampledType,
}

impl InputAttachmentUse {
    /// Color attachment `color_index` read as input attachment `input_attachment_index`, whose
    /// sampled type is `format_class`.
    pub fn new(
        color_index: u32,
        input_attachment_index: u32,
        format_class: SampledType,
    ) -> InputAttachmentUse {
        InputAttachmentUse {
            color_index,
            input_attachment_index,
            format_class,
        }
    }
}
