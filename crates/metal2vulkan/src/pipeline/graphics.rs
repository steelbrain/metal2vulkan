//! `GraphicsState`: the shader-coupled fixed-function state of a render pipeline.
//!
//! It carries the Metal-to-Vulkan mapping of the descriptor's blend, sample, and topology state as
//! plain `vk::` values, the states that must be dynamic on the profile, and the framebuffer-fetch
//! render-pass contract. Metal's depth/stencil state is a separate encoder object and is mapped by
//! `metal`, not here.

use std::collections::BTreeSet;

use crate::air::container::Hash256;
use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{
    Attachment, Capability, Error, Feature, InternalError, InternalKind, Unsupported,
};
use crate::metal;
use crate::pipeline::descriptor::TessellationDescriptor;
use crate::pipeline::interface::{MeshRecords, PatchDomain};
use crate::pipeline::sym::SampledType;
use crate::vk;

/// Fixed-function state of a render pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphicsState {
    /// The list topology of the descriptor's topology class (`PATCH_LIST` for tessellation).
    /// The topology is never dynamic state: the consumer creates the pipeline with the draw's
    /// topology within the class and the primitive restart Metal applies to it, both from
    /// [`metal::state::input_assembly`](crate::metal::state::input_assembly).
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
    /// leaves the coverage as rasterized, and Vulkan requires that output.
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
    /// The pipeline and its subpass use rasterization-order attachment access: set when the
    /// pipeline has input attachments and the device has `rasterizationOrderColorAttachmentAccess`,
    /// so each fetch is ordered after the earlier fragments of the draw that cover its pixel, as
    /// Metal orders it. The consumer creates the subpass with
    /// `VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_EXT` and the
    /// pipeline's color blend state with
    /// `VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT`, and
    /// the pipeline's requirements name the feature. A device without the feature takes no input
    /// attachment: a fetch there is the interlocked storage form or refused.
    pub rasterization_order_attachment_access: bool,
    /// Metal's `maxVertexAmplificationCount` (1 without vertex amplification): a draw with this
    /// pipeline amplifies each vertex by a count in `[1, max_vertex_amplification_count]`, or,
    /// above 1, by a larger count too, as plain Metal draws one. Above 1 the pipeline is a
    /// multiview pipeline: the consumer creates it, and begins its render pass instance, with the
    /// view mask its render plan states (`BindingPlan::views`).
    pub max_vertex_amplification_count: u32,
    /// A stage declares `Layer`: the vertex function chooses its render target slice, or the
    /// fragment function reads it. Such a pipeline cannot be multiview
    /// (`VUID-VkGraphicsPipelineCreateInfo-renderPass-06050`, `-06059`), so the binder refuses its
    /// draws at an amplification count above 1. A vertex function compiled under
    /// `VariantKey::dropped_vertex_layer` (a pass of one layer) declares none; the binder refuses
    /// its amplified draws all the same.
    pub declares_layer: bool,
    /// The tessellation state of a pipeline whose vertex function is a post-tessellation vertex
    /// function; `None` otherwise. Such a pipeline's `topology` is `PATCH_LIST` (never dynamic) and
    /// its stages are the synthesized vertex and tessellation control stages, then the function's
    /// evaluation stage, then the fragment stage.
    pub tessellation: Option<TessellationState>,
    /// The render pass's fragment imageblock, when the fragment function reads or writes it: the
    /// consumer binds the pass's imageblock storage to every draw of the pass
    /// (`WriteSource::Imageblock`).
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// `VkPipelineMultisampleStateCreateInfo::sampleShadingEnable`, with `minSampleShading` 1.0:
    /// the pipeline rasterizes more than one sample and its fragment function writes memory
    /// (`StageInterface::writes_memory`), which Metal runs once per covered sample. The pipeline's
    /// requirements name `sampleRateShading`
    /// (`VUID-VkPipelineMultisampleStateCreateInfo-sampleShadingEnable-00784`). Without it the
    /// consumer leaves sample shading disabled.
    pub sample_shading: bool,
    /// The vertex function (a post-tessellation vertex function included) writes memory
    /// (`StageInterface::writes_memory`): each run of the pipeline's vertex stage repeats its
    /// side effects. A rate-mapped pass drawn region by region records each draw once per region,
    /// so [`RateMapRegions::admits`](crate::bind::RateMapRegions::admits) refuses such a pipeline
    /// in a pass of more than one region.
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_writes_memory: bool,
    /// The vertex function (a post-tessellation vertex function included) writes `ViewportIndex`
    /// (`[[viewport_array_index]]`): each primitive selects its viewport and scissor from the
    /// encoder's arrays. A rate-mapped pass drawn region by region sets one viewport per region, so
    /// [`RateMapRegions::admits`](crate::bind::RateMapRegions::admits) refuses such a pipeline.
    /// Set too on a device without `multiViewport`, where the function stores no `ViewportIndex`
    /// (every index selects viewport 0 of a draw's one, as Metal's does at or past the count):
    /// there a draw of the pipeline while the encoder holds more than one viewport cannot select
    /// one, and the runtime refuses it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_viewport_index: bool,
    /// The color attachments the fragment function reads, and may write, as storage images
    /// inside the ordered pixel interlock (framebuffer fetch on a device without
    /// `rasterizationOrderColorAttachmentAccess`): each such attachment's entry of
    /// `color_attachments` is `None`, its subpass color reference is `VK_ATTACHMENT_UNUSED`, and
    /// the consumer binds a 2D view of the attachment's image as a storage image
    /// (`WriteSource::TileAttachment`) in `GENERAL` layout, over the pixels of the render pass; in
    /// a pipeline of more than one sample the image has `rasterization_samples` samples, or, where
    /// the attachment's `view` is [`StorageAttachmentView::SamplePlanes`], the consumer binds a
    /// 2D-array view of a single-sample image of one layer per sample in its place. See
    /// [`StorageAttachmentUse`] for the consumer contract.
    #[cfg_attr(feature = "serde", serde(default))]
    pub storage_attachments: Vec<StorageAttachmentUse>,
    /// The fragment function's `[[stencil]]` output is a shader-side stencil over a shadow of
    /// the pass's stencil attachment: on a device without `VK_EXT_shader_stencil_export`, the
    /// function applies Metal's stencil test and operations itself, inside the ordered pixel
    /// interlock. `None` for every other pipeline, and for one whose pass has no stencil
    /// attachment (the function then tests nothing, as Metal's output reaches no attachment).
    /// See [`ShadowStencilUse`] for the consumer contract.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shadow_stencil: Option<ShadowStencilUse>,
    /// The fragment function declares `[[early_fragment_tests]]` (its stage interface's
    /// `io.early_fragment_tests`): Metal runs it only for the fragments that pass the depth and
    /// stencil tests. A consumer
    /// that emulates a test Vulkan does not run (a depth compare of `Never` without a depth
    /// attachment, which Metal fails for every fragment) keeps such a function from running.
    #[cfg_attr(feature = "serde", serde(default))]
    pub early_fragment_tests: bool,
    /// A mesh render pipeline on a device without `VK_EXT_mesh_shader`: its mesh function is
    /// the pipeline's `COMPUTE` module, which writes these records, and a synthesized `VERTEX`
    /// module draws them as a list of [`MeshRecords::topology`](MeshRecords::topology)
    /// (the binder's `DrawCommands::mesh_dispatch` states the consumer contract). `None` for
    /// every other pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_records: Option<MeshRecords>,
    /// A render pipeline whose vertex function is a post-tessellation vertex function, on a
    /// device without `tessellationShader`: the function is the pipeline's `VERTEX` module, which
    /// draws the tessellator's triangles itself as a `TRIANGLE_LIST` from an emulated draw's
    /// records (the binder's `DrawCommands::tessellation_dispatch` states the consumer contract),
    /// and the pipeline has no tessellation stages (`tessellation` is `None`). `None` for every
    /// other pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated_tessellation: Option<EmulatedTessellation>,
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
    /// The fragment function reads a flat input, which every fragment of a primitive reads from
    /// one vertex of it (Vulkan's provoking vertex): a user varying it declares `[[flat]]` (a
    /// mesh function's per-primitive output read as one included), or the
    /// `[[render_target_array_index]]` or `[[viewport_array_index]]` a vertex function wrote,
    /// read as a varying or as the `Layer` or `ViewportIndex` builtin, or the `[[primitive_id]]`
    /// its vertex function computes (`IoRole::PrimitiveId`). `false` for a pipeline
    /// without a fragment function, and for one whose fragment function reads only smooth
    /// inputs, perspective-correct or not, or none. A consumer that draws a primitive's vertices
    /// in another order can change what a draw shows only where this is set or the fragment
    /// function reads `[[barycentric_coord]]`, which is no varying input and is not counted here
    /// ([`reads_barycentric_coord`](GraphicsState::reads_barycentric_coord) states it); smooth
    /// inputs interpolate alike in every order. What the vertex function writes to
    /// `Layer` or `ViewportIndex` itself is [`declares_layer`](GraphicsState::declares_layer) and
    /// [`writes_viewport_index`](GraphicsState::writes_viewport_index).
    #[cfg_attr(feature = "serde", serde(default))]
    pub flat_fragment_inputs: bool,
    /// The elements of the vertex function's own `[[clip_distance]]` output, 0 for none (a mesh
    /// pipeline's too). A draw of points compiled under `ViewportCompensation::point_guard`
    /// writes four more after them, and two more again under `ViewportCompensation::depth`,
    /// within `maxClipDistances` and `maxCombinedClipAndCullDistances`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub clip_distances: u32,
    /// Whether the fragment function writes its depth (`[[depth(...)]]`). A draw in depth clamp
    /// mode through a range above 0 runs such a function's `LoweringInputs::depth_clamp_range`
    /// variant where it can, as Metal stores a negative NaN depth as 0 past the range.
    #[cfg_attr(feature = "serde", serde(default))]
    pub writes_depth: bool,
    /// Whether the fragment function reads `[[barycentric_coord]]`, perspective-correct or not
    /// and of any width (`float`, `float2`, `float3`), from the translated fragment stage's
    /// builtins (`BaryCoordKHR`, `BaryCoordNoPerspKHR`) or, on a device without
    /// `fragmentShaderBarycentric`, its varyings, which the pipeline's synthesized geometry stages
    /// write. Metal reads the weights at the pixel center (MSL 5.2.3.4), under a `centroid` or
    /// `sample` qualifier too, and so does the stage. `false` for a pipeline without a fragment
    /// function. A primitive's weights follow the order its vertices are drawn in, so
    /// a consumer that draws them in another order can change what a draw shows only where this
    /// or [`flat_fragment_inputs`](GraphicsState::flat_fragment_inputs) is set.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reads_barycentric_coord: bool,
    /// Whether the fragment function reads `[[primitive_id]]`: Vulkan's `PrimitiveId`, or the
    /// varying its vertex function computes it into on a device that cannot declare it
    /// (`IoRole::PrimitiveId`). `false` for a pipeline without a fragment function. A consumer
    /// that draws one Metal primitive as several Vulkan ones (Metal's quads, one primitive ID a
    /// quad, as two triangles) changes what a draw shows only where this is set.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reads_primitive_id: bool,
    /// Whether the fragment function may read its `[[position]]`'s z or w lanes, which Metal
    /// gives as the viewport's depth `n + (f - n) z` unclamped and the clip-space w (the
    /// fragment stage's `StageIo::position_depth` with its `FragCoord` builtin). `false` for a
    /// pipeline without a fragment function, and for one that reads only the position's x and y
    /// or not at all. A consumer that draws through a Vulkan viewport depth range other than
    /// Metal's, where only the stored depth must be Metal's (one depth in Metal's depth clamp
    /// mode), keeps what such a function reads only where this is `false`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub reads_position_depth: bool,
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
    /// function names is set, whether the function writes it through an output variable or, for
    /// an attachment it reads in the interlocked storage form, through its storage image. It is
    /// decided from the stage's outputs, which come from the AIR's output metadata of the entry its
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
    /// The fragment function runs the draw's depth test itself, inside the ordered pixel
    /// interlock, against a stand-in of the pass's depth attachment: a function that stores a color
    /// through the interlocked storage form before Vulkan's tests, compiled for a draw whose tests
    /// may drop fragments (`VariantKey::shader_tests`). `None` for every other pipeline. See
    /// [`ShaderTestsUse`] for the consumer contract.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_tests: Option<ShaderTestsUse>,
    /// The rule word of each color attachment whose channels' rules the fragment function
    /// selects at the draw (`StageIo::selected_colors`), by color index, sorted: each channel's
    /// code of the rule this pipeline's attachment takes (`pipeline::color_writes::rule_word`),
    /// which the binder writes to the function's `PushField::ColorWriteRule` of the attachment.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_write_rules: Vec<(u32, u32)>,
    /// The fragment function reads textures through snapshots (`VariantKey::snapshot_textures`):
    /// each draw of the pipeline is drawn twice in its subpass, the texture's attachment in
    /// `VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`, first with every color write mask 0,
    /// then with the pipeline's own (`PushHole::SnapshotPhase`), which it sets for every color
    /// blend element: `dynamic_states` holds `VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT`. The consumer
    /// creates the pipeline with `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`. Its
    /// requirements name `attachmentFeedbackLoopLayout` and `extendedDynamicState3ColorWriteMask`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub snapshot_reads: bool,
    /// The group of the pipeline's color attachments this variant draws
    /// (`VariantKey::color_group`), where the device draws them in groups
    /// ([`ColorGroup`](crate::pipeline::ColorGroup)): every color-indexed part of the state
    /// (`color_attachments`, `input_attachments`, `storage_attachments`, `color_write_rules`
    /// among them) indexes the group's attachments by location, Metal's attachment
    /// `colors()[l]` at location `l`, and the fragment stage writes those alone. Each draw of the
    /// pipeline is a draw of every group's variant, in a pass of that group's attachments; the
    /// vertex stage is the same module in every variant, and the fragment stage's `FragDepth`,
    /// `SampleMask` and `FragStencilRefEXT` outputs are decorated `Invariant` where there is more
    /// than one group. `None` where each attachment is at its own index.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_group: Option<crate::pipeline::lower_inputs::ColorGroup>,
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
    /// offset, for a pipeline with a control-point index type.
    pub push_offset: u32,
    /// The descriptor's `tessellationControlPointIndexType`: other than `None`, the pipeline
    /// draws indexed patches only (`DrawCall::IndexedPatches`), whose control-point index buffer
    /// (`SyntheticBinding::ControlPointIndices`) its control stage reads; `None`, plain patches
    /// only.
    pub control_point_index_type: metal::TessellationControlPointIndexType,
}

/// What a consumer and the binder need of a tessellation pipeline on a device without
/// `tessellationShader` (`GraphicsState::emulated_tessellation`).
///
/// Metal draws `patchCount` patches of `instanceCount` instances from `patchStart` and
/// `baseInstance`. Each draw of patches runs the tessellation scan
/// ([`TessellationScanModule`](crate::bind::TessellationScanModule) of [`Self::scan_kernel`]),
/// which reads each patch's factors from the encoder's factor buffer and writes the draw's
/// records, one per instance and patch, and its chunks' `VkDrawIndirectCommand`s; each chunk is
/// a draw of the pipeline, whose vertex function decodes and places the tessellator's point for
/// the vertex it draws and reads Metal's patch index, instance index and control points' elements
/// as the native control stage hands them. The pipeline's topology is `TRIANGLE_LIST`, not
/// dynamic state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmulatedTessellation {
    /// The post-tessellation vertex function's patch type.
    pub domain: PatchDomain,
    /// Its `[[patch(_, N)]]` control points per patch (0 when unstated): a patch draw's
    /// `numberOfPatchControlPoints` must equal a stated count.
    pub control_points: u32,
    /// The descriptor's tessellation state, which the vertex function bakes in.
    pub descriptor: TessellationDescriptor,
}

impl EmulatedTessellation {
    /// The tessellation scan the pipeline's draws run.
    pub fn scan_kernel(&self) -> crate::bind::TessellationScanKernel {
        crate::bind::TessellationScanKernel {
            domain: self.domain,
            partition_mode: self.descriptor.partition_mode,
            output_winding_order: self.descriptor.output_winding_order,
            max_tessellation_factor: self.descriptor.max_tessellation_factor,
            factor_step_function: self.descriptor.factor_step_function,
            factor_scale_enabled: self.descriptor.factor_scale_enabled,
        }
    }
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
            rasterization_order_attachment_access: false,
            max_vertex_amplification_count: 1,
            declares_layer: false,
            tessellation: None,
            fragment_imageblock: None,
            sample_shading: false,
            vertex_writes_memory: false,
            writes_viewport_index: false,
            storage_attachments: Vec::new(),
            shadow_stencil: None,
            early_fragment_tests: false,
            mesh_records: None,
            emulated_tessellation: None,
            writes_point_size: false,
            flat_fragment_inputs: false,
            clip_distances: 0,
            writes_depth: false,
            reads_barycentric_coord: false,
            reads_primitive_id: false,
            reads_position_depth: false,
            fragment_writes_memory: false,
            color_outputs: 0,
            sample_mask: None,
            shader_tests: None,
            color_write_rules: Vec::new(),
            snapshot_reads: false,
            color_group: None,
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

    /// The features the pipeline's multisample state needs: `sampleRateShading` for
    /// [`sample_shading`](GraphicsState::sample_shading)
    /// (`VUID-VkPipelineMultisampleStateCreateInfo-sampleShadingEnable-00784`), and `alphaToOne`
    /// for [`alpha_to_one`](GraphicsState::alpha_to_one)
    /// (`VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785`).
    pub(crate) fn multisample_features(&self) -> BTreeSet<VkFeature> {
        let mut features = BTreeSet::new();
        if self.sample_shading {
            features.insert(VkFeature::SampleRateShading);
        }
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
        // `VK_EXT_extended_dynamic_state3`'s color write masks, dynamic only in a pipeline that
        // reads through snapshots (`GraphicsState::snapshot_reads`).
        D::COLOR_WRITE_MASK_EXT => Some(VkFeature::ExtendedDynamicState3ColorWriteMask),
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

/// How a render pipeline's fragment function reaches the render pass's imageblock.
///
/// Metal keeps a fragment imageblock in tile memory: per pixel, for the whole render pass, its
/// contents undefined when the pass begins, and every pipeline of the pass that declares
/// `[[imageblock_data]]` sees the same bytes at its layout's offsets.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblock {
    /// The representation the profile's capabilities chose.
    pub mechanism: ImageblockMechanism,
    /// The 32-bit words of each pixel's cell the function reaches: the pass's storage needs at
    /// least this many, and the largest over the pass's pipelines.
    pub words: u32,
    /// The fingerprint of the master layout the fragment function's views view: SHA-256 over
    /// its data size and the layout's Merkle digest (SHA-256 over each member's offset, size,
    /// array length, user key, and nested layout's digest). Equal fingerprints are structurally
    /// equal masters. A view of another master of the same `data_size` and `alignment` in the
    /// same pass reaches the same bytes at its own offsets, as each layout's words in the storage
    /// hold them (measured on an M3 Max and an M4 without Metal's API validation, which stops
    /// such a pass); which bytes one of another size or alignment reaches is not read in full, so a
    /// consumer refuses such a pass (the translator sees one pipeline at a time and cannot).
    pub master: Hash256,
    /// The bytes of a cell (`air.imageblock_data_size`).
    pub data_size: u32,
    /// The alignment Metal states for the fragment function's views (`air.arg_type_align_size`:
    /// their master layout's). With `data_size` it gives Metal's `imageblockSampleLength` of the
    /// pipeline (measured on an M3 Max): `data_size` plus the bytes its color attachments pack to,
    /// rounded up to 8 and then to `alignment`.
    pub alignment: u32,
}

/// The representation of a fragment imageblock.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageblockMechanism {
    /// One coherent `R32Uint` 2D-array storage image per render pass instance: layer `w` holds
    /// word `w` of the cell of pixel `(x, y)` at texel `(x, y)`, the pixel's framebuffer
    /// coordinate (`FragCoord`). The fragment function reads its input view and writes its
    /// output view inside the ordered pixel-interlock critical section
    /// (`fragmentShaderPixelInterlock`), which orders overlapping fragments of the pass in
    /// primitive order across draws, as Metal's raster order groups do. The consumer creates the
    /// image with `STORAGE` usage, an extent covering the pass's render area's offset plus its
    /// extent (texel `(x, y)` is framebuffer pixel `(x, y)`, not relative to the area), and at
    /// least `words` layers, keeps it in `GENERAL` layout for the whole pass, binds it to every
    /// draw of the pass, and draws only pipelines of one `master` in the pass.
    StorageImageInterlock,
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

/// A color attachment a render pipeline's fragment function reads, and may write, as a storage
/// image inside the ordered pixel interlock (`GraphicsState::storage_attachments`).
///
/// The draw orders every fragment's read, body, and write of a pixel after the earlier fragments
/// of the pixel in primitive order, across draws and instances of the subpass, as Metal orders
/// framebuffer fetch. The consumer:
///
/// - leaves the attachment out of the color attachments of every subpass a pipeline that names
///   it draws in (its color reference there is `VK_ATTACHMENT_UNUSED`): an image a subpass
///   writes as an attachment is no storage image of the same subpass;
/// - binds the image `view` names, in `format` with the identity mapping, as the plan's
///   `WriteSource::TileAttachment { color_index, format, view }` (which carries the same `view`),
///   in `GENERAL` layout, texel `(x, y)` being framebuffer pixel `(x, y)` (so the image covers
///   the render area's offset plus its extent):
///   - [`StorageAttachmentView::Attachment`]: a 2D view of the attachment's own image, of the
///     render pass's layer and level, the image created with `STORAGE` usage (beside its own).
///     In a pipeline of more than one sample that image is the pass's multisampled attachment,
///     of the pipeline's `GraphicsState::rasterization_samples`, the one sample count of the
///     pipeline and its storage images, and the fragment function runs once per covered sample,
///     reading and writing its own sample; the pipeline's requirements name
///     `shaderStorageImageMultisample` and the count's `storageImageSampleCounts` bit;
///   - [`StorageAttachmentView::SamplePlanes`] `{ layers }` (a pipeline of more than one sample
///     on a device without multisampled storage images): a 2D-array view
///     (`VK_IMAGE_VIEW_TYPE_2D_ARRAY`) of all `layers` layers of the attachment's sample planes,
///     a single-sample 2D-array image the consumer keeps beside the attachment, of `format`, with
///     `STORAGE` usage and `layers` (the pipeline's `rasterization_samples`) layers, never a 2D
///     view and never the multisampled attachment itself. The fragment function runs once per
///     covered sample and reads and writes sample `s` of pixel `(x, y)` at texel `(x, y)` of
///     layer `s`, so the planes need no multisampled storage image and no
///     `storageImageSampleCounts` bit. The consumer fills layer `s` from sample `s` of the
///     attachment's image before the draws (no copy changes a sample count, so a per-sample
///     shader read does), and after them writes layer `s` back into sample `s` (a sample-rate
///     pass that writes only its own sample), or, where the pass resolves the attachment or the
///     attachment is memoryless, resolves from the planes as Metal resolves the samples;
/// - makes earlier writes of the image visible to the fragment shader's storage reads before
///   the draws (a color attachment write of an earlier subpass or pass, a transfer, a load
///   operation), and the fragment shader's storage writes visible to later uses after them.
///
/// A pass of more than one layer, and an amplified draw, are not implemented: the binder refuses
/// them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StorageAttachmentUse {
    pub color_index: u32,
    /// The attachment's format, which the storage view takes.
    pub format: vk::Format,
    /// The fragment function writes the attachment (otherwise it only reads it).
    pub writes: bool,
    /// The function declares the storage image without a format (`Unknown`), so the view's
    /// format ([`view_format`](Self::view_format)) supports storage reads (and, when `writes`,
    /// writes) without one (`VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT` and its write
    /// twin); otherwise the declaration names the view's format's SPIR-V image format.
    pub formatless: bool,
    /// The image the consumer binds and how it views it: the attachment's own image, or its
    /// sample planes (see the type's docs). The plan's `WriteSource::TileAttachment` carries the
    /// same value.
    #[cfg_attr(feature = "serde", serde(default))]
    pub view: StorageAttachmentView,
    /// Where `Some`, the storage view is in this format, not in `format`, which stays the
    /// attachment's own ([`view_alias_of`](Self::view_alias_of) decides it from `format` and the
    /// device):
    ///
    /// - an 8-bit sRGB attachment's unsigned-integer format of its texel size (`R32_UINT` for a
    ///   4-byte texel): its format has no storage access, and the function unpacks the texels in
    ///   `format`'s byte order and decodes and encodes them itself, by Metal's measured
    ///   conversions;
    /// - a BGRA-order attachment's RGBA-order twin (`R8G8B8A8_UNORM` for `B8G8R8A8_UNORM`,
    ///   `A2B10G10R10_UNORM_PACK32` for `A2R10G10B10_UNORM_PACK32`) on a device that neither
    ///   reads nor writes `format` without a format (SPIR-V names no image format for it): the
    ///   function exchanges red and blue on every read and write, and the device converts each
    ///   channel as it converts `format`'s.
    ///
    /// The consumer binds such a view of an image made with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT`
    /// (and `VK_IMAGE_CREATE_EXTENDED_USAGE_BIT` where `format` has no storage access), whose
    /// `VkImageFormatListCreateInfo` lists `format` and this format, with `STORAGE` usage (the
    /// sample planes of [`StorageAttachmentView::SamplePlanes`] in this format); its draws that
    /// do not fetch render the image through a view in `format`. `None` for any other format, and
    /// on a device that accesses a BGRA-order format without one: the view is in `format`.
    #[cfg_attr(feature = "serde", serde(default, alias = "uint_view"))]
    pub view_alias: Option<vk::Format>,
    /// The channels the function's write keeps from the texel it reads first, by the pipeline's
    /// write mask: empty where it does not write the attachment or the mask covers every channel of
    /// the format, all four where it covers none of them (the write stores nothing), else those of
    /// the format the mask leaves out. One module serves every write mask that keeps some of a
    /// format's channels (a blended 8-bit unorm attachment with alpha: each of alpha alone, one
    /// color channel, and more): the binder writes these into each draw's push bytes, where the
    /// function reads them. Nothing for the consumer to do.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept: vk::ColorComponentFlags,
}

/// The image a storage view of a color attachment binds, and how it is viewed
/// (`StorageAttachmentUse::view`, `WriteSource::TileAttachment`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageAttachmentView {
    /// A 2D view (`VK_IMAGE_VIEW_TYPE_2D`) of the attachment's own image, of the pass's layer
    /// and level: single-sample in a pass of one sample, of the pass's samples in a multisampled
    /// pass.
    #[default]
    Attachment,
    /// A 2D-array view (`VK_IMAGE_VIEW_TYPE_2D_ARRAY`) of every layer of the attachment's sample
    /// planes: a single-sample 2D-array image of `layers` layers (the pipeline's
    /// `rasterization_samples`), layer `s` holding sample `s` of each pixel, which the consumer
    /// fills from the multisampled attachment before the draws and writes back (or resolves)
    /// after them.
    SamplePlanes { layers: u32 },
}

impl StorageAttachmentUse {
    /// Color attachment `color_index` of `format` bound as a storage image through `view`, written
    /// where `writes`, declared without a format where `formatless`.
    pub fn new(
        color_index: u32,
        format: vk::Format,
        writes: bool,
        formatless: bool,
        view: StorageAttachmentView,
    ) -> StorageAttachmentUse {
        StorageAttachmentUse {
            color_index,
            format,
            writes,
            formatless,
            view,
            view_alias: None,
            kept: vk::ColorComponentFlags::empty(),
        }
    }

    /// The use whose write keeps `kept` from the texel ([`kept`](Self::kept)).
    pub fn with_kept(self, kept: vk::ColorComponentFlags) -> StorageAttachmentUse {
        StorageAttachmentUse { kept, ..self }
    }

    /// The use with its storage view in `view_alias` ([`view_alias`](Self::view_alias)).
    pub fn with_view_alias(self, view_alias: vk::Format) -> StorageAttachmentUse {
        StorageAttachmentUse {
            view_alias: Some(view_alias),
            ..self
        }
    }

    /// The format of the storage view: [`view_alias`](Self::view_alias) where it is set, else
    /// `format`.
    pub fn view_format(&self) -> vk::Format {
        self.view_alias.unwrap_or(self.format)
    }

    /// The format other than `format` a fetching draw views a color attachment of `format` in,
    /// where a fragment function reads it in the storage form on `profile`
    /// ([`view_alias`](Self::view_alias)): `R32_UINT` for `B8G8R8A8_SRGB` and `R8G8B8A8_SRGB`,
    /// `R16_UINT` for `R8G8_SRGB` and `R8_UINT` for `R8_SRGB`, on every device; `R8G8B8A8_UNORM`
    /// for `B8G8R8A8_UNORM` and `A2B10G10R10_UNORM_PACK32` for `A2R10G10B10_UNORM_PACK32` where
    /// the device neither reads nor writes the attachment's format without a format, and stores
    /// the twin and declares it (without a format, or by its SPIR-V image format,
    /// `shaderStorageImageExtendedFormats` for an extended one); `None` for any other format and
    /// device. The compile layer and a consumer that makes an attachment's image before it knows
    /// the pipeline (a memoryless attachment's backing, a render target's free usage) both take
    /// it from here, so the image is made mutable to the view the pipeline binds. The compile
    /// layer sets an sRGB view only where it builds the form (today the four-byte formats).
    pub fn view_alias_of(format: vk::Format, profile: &DeviceProfile) -> Option<vk::Format> {
        crate::metal::format::fetch_view_alias(format, profile)
    }

    /// The format usages the consumer's image and view need: `StorageImage` of the view's
    /// format, and the without-format accesses of a formatless declaration.
    pub(crate) fn format_usages(
        &self,
    ) -> Vec<(vk::Format, crate::device::requirements::FormatUsage)> {
        use crate::device::requirements::FormatUsage as U;
        let view = self.view_format();
        let mut out = vec![(view, U::StorageImage)];
        if self.formatless {
            out.push((view, U::StorageImageReadWithoutFormat));
            if self.writes {
                out.push((view, U::StorageImageWriteWithoutFormat));
            }
        }
        out
    }
}

/// A render pipeline whose fragment function's `[[stencil]]` output is a shader-side stencil
/// (`GraphicsState::shadow_stencil`).
///
/// Metal tests and writes the stencil with the function's output as the reference of both faces
/// (its low 8 bits): `(r & readMask) cmp (s & readMask)`, then `s' = (s & ~writeMask) |
/// (op(s, r) & writeMask)`, before the depth test, per sample in primitive order. Without
/// `VK_EXT_shader_stencil_export` no fixed-function state takes a per-fragment reference, so the
/// function does it: inside the ordered pixel interlock it reads the pixel's value from a shadow
/// of the stencil, applies the face's test and operation from the encoder's state
/// (`BindState::depth_stencil`, written as `PushField::ShadowStencilState`), writes the shadow,
/// and kills a fragment that fails. A discarded fragment changes nothing. The depth test stays
/// fixed-function, after the function. In a pipeline of more than one sample the shadow holds
/// each sample, the function tests each sample of its coverage, and its `SampleMask` output keeps
/// the samples that pass.
///
/// The consumer, for each draw with such a pipeline:
///
/// - creates the pipeline with the stencil test disabled (`stencilTestEnable` false, or
///   `vkCmdSetStencilTestEnable(false)` where the pipeline makes it dynamic): the function is the
///   test, and the attachment's stencil is not written during the draw;
/// - keeps the shadow: a 2D image of `format` and `samples` samples (the pipeline's) with
///   `STORAGE` and transfer usage, of the stencil attachment's extent, whose texel `(x, y)` (each
///   sample) is framebuffer pixel `(x, y)` (that sample), bound in `GENERAL` layout as the plan's
///   `WriteSource::ShadowStencil`;
/// - copies the stencil into the shadow before the draw and back after it, outside the render
///   pass: of one sample, `vkCmdCopyImageToBuffer` of the attachment's stencil aspect (one byte
///   per texel) then `vkCmdCopyBufferToImage` into the shadow, and the reverse afterwards; of more,
///   `vkCmdCopyImage` between the stencil aspect and the shadow, both ways, which
///   `VK_KHR_maintenance8`'s depth/stencil-to-color copies allow (no buffer copy takes a
///   multisampled image; the pipeline requires `maintenance8` then). A render pass instance
///   that holds such a draw is therefore split around it: end the pass, copy in, begin it again
///   loading every attachment, draw, end it, copy out, and begin it again. Consecutive such draws
///   share one copy in and one copy out while nothing else writes the stencil between them. Each
///   copy needs its barriers: attachment writes before the copy out of the stencil, the copy into
///   the shadow before the fragment shader's storage reads and writes, those before the copy
///   back, and that copy before the next attachment access;
/// - sets `BindState::depth_stencil` to the encoder's `MTLDepthStencilDescriptor` for the draw.
///   The binder refuses a state whose stencil write could depend on the depth test, typed as
///   needing the extension: a depth attachment, a depth comparison other than `Always`, and a
///   face whose depth-failure operation differs from its pass operation. The stencil references
///   are unused (the function's output is the reference of both faces), and the front face is the
///   pipeline's, from the encoder's winding.
///
/// The split costs a render pass instance per such draw (or run of draws). Synchronizing the
/// shadow inside the pass (an input-attachment read of the stencil under a feedback-loop layout,
/// and an in-pass write-back) would avoid it; that is a possible later optimisation, not part of
/// this contract.
///
/// An amplified draw and a pass of more than one layer are not implemented: the binder refuses
/// them. Alpha to coverage, which changes a fragment's samples before Metal's stencil test, is
/// refused at link as needing the extension.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShadowStencilUse {
    /// The shadow's format, `R8_UINT`.
    pub format: vk::Format,
    /// The function declares the shadow without a format (`Unknown`), so the format supports
    /// storage reads and writes without one (`VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`
    /// and its write twin); otherwise the declaration names `R8ui`.
    pub formatless: bool,
    /// The shadow's samples: the pipeline's rasterization samples.
    pub samples: u32,
}

impl ShadowStencilUse {
    /// The format usages the consumer's shadow image and view need: `StorageImage`, and the
    /// without-format accesses of a formatless declaration.
    pub(crate) fn format_usages(
        &self,
    ) -> Vec<(vk::Format, crate::device::requirements::FormatUsage)> {
        use crate::device::requirements::FormatUsage as U;
        let mut out = vec![(self.format, U::StorageImage)];
        if self.formatless {
            out.push((self.format, U::StorageImageReadWithoutFormat));
            out.push((self.format, U::StorageImageWriteWithoutFormat));
        }
        out
    }
}

/// A render pipeline whose fragment function runs the draw's depth test itself
/// (`GraphicsState::shader_tests`).
///
/// A function that stores a color through the interlocked storage form stores before Vulkan's
/// tests, where Metal stores only the fragments that pass them. For a draw whose tests may drop
/// fragments, the function tests itself: at each return, inside the ordered pixel interlock, it
/// reads the pixel's depth (its sample's, in a pipeline of more than one sample) from a stand-in of
/// the depth attachment, compares the fragment's depth with it by the encoder's compare function,
/// writes the stand-in where the test passes and depth writes are enabled, and stores its colors
/// only where it passes; a failing fragment is killed after the critical section, so the pass's
/// other color attachments drop it too. A discarded fragment changes nothing. The fragment's depth
/// is the device's `FragCoord.z` (Metal's `[[position]].z`), clamped as Metal clamps it: to `[0,
/// 1]` in depth clip mode, to the viewport's depth range in clamp mode.
///
/// The consumer, for each draw with such a pipeline:
///
/// - keeps the pass's depth/stencil attachment in the draw's subpass, with the depth test
///   enabled under `VK_COMPARE_OP_ALWAYS`, depth writes, the stencil test, and depth bounds off,
///   whatever the encoder's depth/stencil state: the function is the test, and the attachment is
///   not written during the draw;
/// - keeps the stand-in ([`StandInUse`]): an image of the depth attachment's extent with
///   `STORAGE` usage, in `GENERAL` layout, bound as the plan's `WriteSource::DepthStandIn`,
///   whose texel `(x, y)` holds pixel `(x, y)`'s depth as an `f32` (each sample's, as the
///   stand-in's view says);
/// - fills the stand-in from the depth aspect before the draw (or sets it to the pass's clear
///   depth where the draw's render pass instance clears the attachment) and writes it back to the
///   depth aspect after it, outside the render pass, with the barriers each copy needs.
///   Consecutive such draws share one fill and one write-back while nothing else uses the depth
///   attachment between them; a draw of another pipeline in between ends the instance;
/// - sets `BindState::depth_stencil`, `BindState::depth_range` and
///   `BindState::depth_clip_mode_clamp` to the encoder's state for the draw.
///
/// Where the draw's stencil test can drop a fragment or write (`stencil`), the function runs it
/// too, first, as Metal does: the face's (`FrontFacing`) comparison of its reference with the
/// stand-in's value under the read mask, then its stencil-failure, depth-failure or pass operation
/// under the write mask, and the depth test and color stores only where the stencil test passes.
/// The consumer then also (the stencil test staying off, as above):
///
/// - keeps the stencil's stand-in: an `R8_UINT` image of the stencil attachment's extent and the
///   pipeline's samples (multisampled at more than one) with `STORAGE` usage, in `GENERAL`
///   layout, bound as the plan's `WriteSource::StencilStandIn`, whose texel `(x, y)` (each
///   sample) holds pixel `(x, y)`'s stencil (that sample's);
/// - fills it from the stencil aspect before the draw (or sets it to the pass's clear value
///   where the draw's render pass instance clears the stencil) and writes it back to the stencil
///   aspect after it, as for the depth: of one sample through a buffer, of more by
///   `vkCmdCopyImage` between the stencil aspect and the stand-in, which `VK_KHR_maintenance8`
///   allows (the pipeline requires it then), as for the shader-side stencil's shadow
///   ([`ShadowStencilUse`]);
/// - sets `BindState::stencil_references` to the encoder's front and back references.
///
/// The binder refuses a counted draw (a visibility result mode is set), an amplified draw, and a
/// pass of more than one layer; it asks for the function only for a pass with a depth
/// attachment, and for the stencil's stand-in only for a pass with both.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderTestsUse {
    /// The depth attachment's stand-in.
    pub depth: StandInUse,
    /// The stencil attachment's stand-in, where the function runs the stencil test too
    /// (`RuntimeFact::ShaderStencilTest`): `R8_UINT`, viewed as an image of the pipeline's
    /// samples. `None` where the draws it serves have a stencil test that does nothing.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stencil: Option<StandInUse>,
}

/// A stand-in of a depth or stencil attachment's plane, which a fragment function that runs the
/// draw's tests itself reads and writes as a storage image (`GraphicsState::shader_tests`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StandInUse {
    /// The stand-in's format: `R32_SFLOAT` for depth, `R8_UINT` for stencil.
    pub format: vk::Format,
    /// The function declares the stand-in without a format (`Unknown`), so the format supports
    /// storage reads and writes without one; otherwise the declaration names it.
    pub formatless: bool,
    /// The pipeline's rasterization samples.
    pub samples: u32,
    /// How the image holds the samples.
    pub view: StandInView,
}

impl StandInUse {
    /// The format usages the consumer's stand-in image and view need: `StorageImage`, and the
    /// without-format accesses of a formatless declaration.
    pub(crate) fn format_usages(
        &self,
    ) -> Vec<(vk::Format, crate::device::requirements::FormatUsage)> {
        use crate::device::requirements::FormatUsage as U;
        let mut out = vec![(self.format, U::StorageImage)];
        if self.formatless {
            out.push((self.format, U::StorageImageReadWithoutFormat));
            out.push((self.format, U::StorageImageWriteWithoutFormat));
        }
        out
    }
}

/// How a stand-in holds a pipeline's samples (as the storage form's attachments do).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StandInView {
    /// A 2D image of the pipeline's samples: single-sample at one sample, multisampled at more
    /// (on a device with multisampled storage images), sample `s` of pixel `(x, y)` at texel
    /// `(x, y)`, sample `s`.
    Image,
    /// A single-sample 2D-array image of `layers` layers (the pipeline's samples), bound through
    /// a 2D-array view: layer `s` holds sample `s` of each pixel, on a device without multisampled
    /// storage images.
    SamplePlanes { layers: u32 },
}

/// A framebuffer-fetch input: color attachment `color_index` read as input attachment
/// `input_attachment_index`. The consumer's render pass provides the self-dependency (or
/// `dynamicRenderingLocalRead` mapping) that the contract names. In a pipeline of more than one
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
