//! `BindingPlan`: the shape-level plan of a pipeline.
//!
//! A plan is a pure value of `(pipeline, BindShape)`, memoized by the consumer. It writes every
//! layout binding: a guest resource, a synthetic buffer, or a neutral stand-in of the exact
//! descriptor and view type. Compute planning is `ComputePipeline::plan` (`compute`); graphics
//! planning is `bind::plan::graphics`.

pub(crate) mod compute;
pub(crate) mod resources;
pub(crate) mod slots;

use crate::bind::heap::HeapSource;
use crate::bind::neutral::NeutralResource;
use crate::cache::{SpecializationKey, VariantKey};
use crate::metal::sampler::OpaqueBlackBorder;
use crate::pipeline::artifact::SpecConstant;
use crate::pipeline::graphics::StorageAttachmentView;
use crate::pipeline::interface::{AccessSet, BufferForm, PlacedField, Window};

use crate::pipeline::sym::{
    AccelerationStructureLevel, HandleClass, PushField, SlotRef, SpecKey, Stage,
};
use crate::pipeline::vertex::VertexStride;
use crate::vk;

/// The shape-level binding plan of a pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindingPlan {
    /// The variant the shape requires: the storage-image choices, `BufferRangeLimit` device-address
    /// slots, vertex buffers fixed-function input cannot fetch as bound, a vertex function's
    /// dropped `Layer` output in a pass of one layer, and the read-only slots Profile F's per-stage
    /// storage-buffer fallback must keep as storage views where the shape binds more than a uniform
    /// view may reach. Empty on the common path; otherwise the consumer compiles the pipeline with
    /// this variant (`Device::compile_render_variant` for a render pipeline) and plans that
    /// pipeline, whose plan for the same shape states the same key. The descriptors are those of
    /// the pipeline planned; a draw of a plan whose variant pulls a vertex buffer the pipeline
    /// fetches fixed function is an internal error, and one whose variant drops (or keeps) the
    /// vertex `Layer` output the pipeline keeps (or drops) is `BindError::NeedsVariant`.
    pub variant: VariantKey,
    /// Specialization of the pipeline for this shape: the constants the shape decides
    /// (threadgroup lengths, slot presence). Each compute `DispatchRegion` carries the complete
    /// key: these, plus its local size and `MultiRegion`.
    pub specialization: SpecializationKey,
    /// One write per layout binding and array element.
    pub descriptors: Vec<DescriptorWrite>,
    pub vertex_buffers: Vec<VertexBufferBind>,
    /// Staging, writeback, and hazard facts.
    pub uses: Vec<ResourceUse>,
    /// Which push fields instantiation fills, and where.
    pub push: PushTemplate,
    pub diagnostics: Vec<PlanDiagnostic>,
    /// What `instantiate_dispatch` needs of the pipeline and the device (compute plans only).
    pub compute: Option<ComputeTemplate>,
    /// What `instantiate_draw` needs of the pipeline and the device (render plans only).
    pub graphics: Option<GraphicsTemplate>,
    /// The views of a render plan's draws under the shape's vertex amplification; the default, no
    /// multiview, for a compute plan.
    pub views: ViewSetup,
}

impl BindingPlan {
    /// The value this plan's bound resources give `key` where a stage reads it as data
    /// (`PushField::Specialized`), the value instantiation pushes, or where a stage's specialized
    /// note alone tests it (`SpecKey::BoundSamplerWord`, the bound sampler's word), which nothing
    /// pushes; `None` for any other key. A pipeline's key carries no such value, so a runtime
    /// reads a stage's notes that test it through `StageModule::precision_notes_bound` with this.
    pub fn specialized_value(&self, key: SpecKey) -> Option<u64> {
        let resources = match (&self.compute, &self.graphics) {
            (Some(c), _) => &c.resources,
            (None, Some(g)) => &g.resources,
            (None, None) => return None,
        };
        resources
            .specialized
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| u64::from(*v))
    }
}

// Adding a field to a plan type: a plan's content digest omits a field whose value is zero (`0`,
// `false`, `None`, empty), so a new field must hold its zero value exactly when the plan means what
// it meant before the field existed. If its zero would mean new behaviour for existing plans, the
// addition must also bump the runtime version or the validation executor's ABI (ask the
// orchestrator): the digest cannot see it (`cache::digest`, `docs/VALIDATION.md`).
#[cfg(feature = "serde")]
impl BindingPlan {
    /// The plan's content digest: a SHA-256 of everything the plan states, by the rules of
    /// `ComputePipeline::content_digest` (fields and enum variants by name, a field at its zero
    /// value omitted), taken over its `serde` serialization. A plan names no cache key and no
    /// version, so two builds that plan a pipeline for a shape alike give the same digest; it
    /// changes whenever a value the plan states changes. Like the pipelines', it identifies content
    /// and authenticates nothing. Fails only when a `Serialize` impl reports an error, which none
    /// of the plan's types does.
    pub fn content_digest(&self) -> Result<crate::cache::Hash256, String> {
        crate::cache::digest::node_of(self)
            .map(|node| node.root("binding plan"))
            .map_err(|error| error.to_string())
    }
}

/// How the consumer sets up the views of a render plan's draws. Vertex amplification is multiview:
/// amplification `a` of a draw is view `a`, which reads `a` as its `[[amplification_id]]` and
/// renders into layer `a` of each attachment view. A draw renders into the pass's layers from
/// `base_layer`: `view_mask`'s views of them, or without multiview the one layer there; the
/// default, both 0, renders the pass's every layer as a pass without amplification does.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewSetup {
    /// The multiview view mask: `(1 << count) - 1` for an amplification count above 1, and 0
    /// (no multiview) otherwise. The consumer creates the pipeline with it
    /// (`VkPipelineRenderingCreateInfo::viewMask`, or its subpass's) and records the draw in a
    /// render pass instance of the same view mask.
    pub view_mask: u32,
    /// The layer of the pass, counted from its first, that view 0 renders into (the first view
    /// mapping's `renderTargetArrayIndexOffset` where it is below the pass's
    /// `renderTargetArrayLength`, and 0 where it is not): the consumer's attachment views start
    /// there, and view `a` renders into layer `base_layer + a`. Without multiview a draw whose
    /// `base_layer` is not 0 renders into that one layer, through attachment views of it alone.
    pub base_layer: u32,
}

/// The pipeline and device facts a compute plan instantiates with: per buffer slot its form,
/// window, and descriptor; the specialization layout; the local-size rules; the push placement; and
/// the root-table layout. Opaque to the consumer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeTemplate {
    /// Every statically used buffer slot, by slot.
    pub(crate) buffers: Vec<PlannedBuffer>,
    /// The stage's specialization constants, in `SpecId` order.
    pub(crate) spec: Vec<SpecConstant>,
    /// The values the shape decides, by key.
    pub(crate) fixed: Vec<(SpecKey, u64)>,
    pub(crate) local: LocalRules,
    /// Every placed push field, in placement order.
    pub(crate) fields: Vec<PlacedField>,
    /// `(offset, size)` of the push-constant range.
    pub(crate) push_range: Option<(u32, u32)>,
    /// `(set, binding, size)` of the stage's `ArgumentTable` uniform buffer.
    pub(crate) table: Option<(u32, u32, u32)>,
    /// The first root of the resident set (`R0`), when the function reads the root table.
    pub(crate) root_first: Option<u32>,
    /// `minStorageBufferOffsetAlignment` (the indirect clamp's source range).
    pub(crate) storage_alignment: u64,
    /// The texel buffers the dispatch binds.
    pub(crate) resources: ResourceTemplate,
    /// Whether the kernel reads the stage-in region (`RuntimeFact::StageInRegion`): a dispatch
    /// without a region is `BindError::MissingStageInRegion`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) stage_in_region: bool,
    /// The kernel's `[[stage_in]]` fetches of a thread-position step, whose elements decide each
    /// region's `SpecKey::StageInReach` (`dispatch::stage_in_reach`); empty where the kernel
    /// declares no such key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) stage_in_grid: Vec<crate::pipeline::stage_io::StageInFetch>,
}

/// The pipeline and device facts a render plan instantiates with: the buffer slots of both stages,
/// each stage's push placement and tables, the fixed-function vertex buffers, and the texel
/// buffers. Opaque to the consumer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphicsTemplate {
    /// Every statically used buffer slot of both stages, by slot.
    pub(crate) buffers: Vec<PlannedBuffer>,
    /// Each stage, in pipeline order.
    pub(crate) stages: Vec<StageTemplate>,
    /// Every fixed-function vertex buffer, in binding order.
    pub(crate) vertex_buffers: Vec<PlannedVertexBuffer>,
    /// `(layout, stride)` of every pulled layout with a static stride, by layout.
    pub(crate) pulled_strides: Vec<(u32, u32)>,
    pub(crate) resources: ResourceTemplate,
    /// What a per-bind vertex stride must meet.
    pub(crate) vertex_strides: StrideRules,
    /// The device's `maxMultiviewInstanceIndex` when the plan's draws are multiview
    /// (`ViewSetup::view_mask` is not 0); `None` otherwise.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) multiview_instance_limit: Option<u32>,
    /// The pipeline planned is not the variant its vertex function's render target slice asks
    /// for at the shape's render target array length: it writes `Layer` in a pass of one layer, or
    /// does not in a pass of more; or not the variant the shape's vertex amplification asks for,
    /// drawn as views or one amplification at a time. A draw is `BindError::NeedsVariant` with the
    /// plan's `variant`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) other_layer_variant: bool,
    /// A plan whose draws draw one amplification each (`VariantKey::amplification_draws`): the
    /// `(render target slice offset, viewport offset)` of each amplification, in order. Each draw
    /// is recorded once per entry, after the push writes that make it that amplification's
    /// (`DrawCommands::amplifications`). Empty for a plan whose draws are one draw each. A plan
    /// whose pipeline is not the variant its amplification asks for sets `other_layer_variant`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) amplifications: Vec<[u32; 2]>,
    /// A tessellation pipeline's, native or emulated: its draws draw patches.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) tessellation: Option<FactorTemplate>,
    /// The device lacks `drawIndirectFirstInstance`: an indirect draw's arguments are read on
    /// the CPU (`DrawCommands::indirect_needs_arguments`), since Metal's may carry a base
    /// instance.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) indirect_first_instance_missing: bool,
    /// The device enables `robustBufferAccess2`, under which Vulkan bounds an index fetch past
    /// the bound index buffer range (`VUID-vkCmdDrawIndexed-robustBufferAccess2-08798` applies
    /// only without it): a direct indexed draw past its index buffer is drawn, not clamped.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) robust_index_fetch: bool,
    /// A fragment function's shader-side stencil: its draws write the encoder's stencil state as
    /// `PushField::ShadowStencilState`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) shadow_stencil: Option<ShadowTemplate>,
    /// A mesh render pipeline's mesh stage: its draws are mesh draws, each of its own
    /// specialization.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) mesh: Option<MeshTemplate>,
    /// A fragment function that runs the draw's depth test itself: its draws write the encoder's
    /// depth state as `PushField::ShaderTestsState`, and, where it runs the stencil test too, the
    /// encoder's stencil state and references as `PushField::ShaderStencilState`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) shader_tests: Option<ShaderTestsTemplate>,
    /// A pipeline compiled in the primitive ID's de-indexed form of this index type
    /// (`VariantKey::primitive_id_indices`): its indexed draws write `PushField::CaptureLayout`
    /// and run without indices.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) deindexed: Option<crate::metal::IndexType>,
    /// A pipeline compiled in the private list form (`VariantKey::private_list`): its draws are
    /// the consumer's lists of Metal's private primitive types, run without indices, of any
    /// draw call.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) private_list: bool,
    /// `PushField::StorageKeptChannels`: each storage-form color attachment's kept channels
    /// (`StorageAttachmentUse::kept`), bits `4c` to `4c + 3` for color `c`, which its draws
    /// write.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) storage_kept: u32,
    /// A band variant's (`GraphicsState::line_band`): the byte offset of its evaluation stage's
    /// push range, where its draws push `BindState::line_band` (`PushField::LineBand`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) line_band: Option<u32>,
}

/// What a mesh draw of a mesh render pipeline instantiates: the pipeline's specialization layout
/// with the values the shape decides, which each draw completes with its threadgroups' sizes, and
/// the device's threadgroup limits of each stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct MeshTemplate {
    /// Every stage's specialization constants, in `SpecId` order.
    pub spec: Vec<SpecConstant>,
    /// The values the shape decides, by key.
    pub fixed: Vec<(SpecKey, u64)>,
    /// Every `LocalSizeConstraint{multiple}` of the pipeline: a threadgroup of either stage must
    /// be whole subgroups of the flattened tier's size.
    pub multiples: Vec<u32>,
    /// The mesh stage's limits (`maxMeshWorkGroup*`; an emulated mesh function's are the
    /// compute stage's, `maxComputeWorkGroup*`).
    pub mesh: GroupLimits,
    /// The task stage's limits (`maxTaskWorkGroup*`), of a pipeline with an object function.
    pub object: Option<GroupLimits>,
    /// The object function is compiled folded (`RuntimeFact::FoldedGroups`,
    /// `lower::mesh::folds_object_grids`): each draw runs its object grid in pieces within the task
    /// stage's limits (`bind::mesh_arguments`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub folded: bool,
    /// The object function runs a partial threadgroup as Metal does
    /// (`RuntimeFact::PartialObjectThreadgroups`): a `drawMeshThreads` whose threads leave one
    /// draws it, its region flagged `lower::dispatch::NONUNIFORM`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub partial_groups: bool,
    /// An emulated mesh function's records: each draw dispatches its mesh threadgroups and
    /// draws their primitives (`DrawCommands::mesh_dispatch`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub records: Option<RecordsTemplate>,
}

/// A stage's threadgroup and grid limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct GroupLimits {
    /// `max*WorkGroupSize`.
    pub size: [u32; 3],
    /// `max*WorkGroupInvocations`.
    pub invocations: u32,
    /// `max*WorkGroupCount`.
    pub count: [u32; 3],
    /// `max*WorkGroupTotalCount`; `None` for the compute stage, which has none.
    pub total: Option<u32>,
    /// The stage the limits are of, which names them.
    pub stage: GroupStage,
}

/// The Vulkan stage a mesh draw's threadgroups run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum GroupStage {
    /// An object function's (`TASK_EXT`).
    Task,
    /// A mesh function's (`MESH_EXT`).
    Mesh,
    /// An emulated mesh function's (`COMPUTE`).
    Compute,
}

/// What a draw of an emulated mesh function's pipeline sizes from its grid
/// (`DrawCommands::mesh_dispatch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct RecordsTemplate {
    /// The words of one threadgroup's records (`MeshRecords::group_words`).
    pub group_words: u64,
    /// The vertices of one threadgroup's primitives in the draw: `NP · k`.
    pub group_vertices: u32,
    /// The device's `maxStorageBufferRange`, past which a draw's records or work buffer cannot
    /// be bound.
    pub max_range: u32,
    /// The words of an object threadgroup's slot in the work buffer (`MeshRecords::slot_words`).
    pub slot_words: u32,
    /// `maxComputeWorkGroupCount[0]`, the mesh dispatch's rows.
    pub row: u32,
    /// `maxComputeWorkGroupCount[1]`, its most rows.
    pub rows: u32,
    /// `minStorageBufferOffsetAlignment`, to which the scan's arguments range is aligned.
    pub alignment: u64,
}

/// What a draw of a pipeline whose fragment function runs the draw's depth test itself
/// instantiates: the encoder's depth state, which the plan reads from the draw's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ShaderTestsTemplate;

/// What a draw of a pipeline with a shader-side stencil instantiates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ShadowTemplate {
    /// The pass has a stencil attachment, so the plan binds the shadow
    /// (`GraphicsState::shadow_stencil`); otherwise a neutral stands in and nothing is tested.
    pub attached: bool,
    /// The pass has a depth attachment, whose test could decide the stencil write.
    pub depth_attachment: bool,
}

/// What a patch draw of a tessellation pipeline instantiates: the push fields of the stage that
/// reads the patch and control-point index buffers, their descriptor ranges, and the factor
/// buffer's range, read by the control stage of a native pipeline or the scan of an emulated one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct FactorTemplate {
    pub step: crate::metal::TessellationFactorStepFunction,
    /// Bytes of one patch's factors.
    pub patch_bytes: u32,
    /// `minStorageBufferOffsetAlignment`.
    pub alignment: u64,
    /// `maxStorageBufferRange`.
    pub max_range: u64,
    /// The `N` of the function's `[[patch(_, N)]]` (0 when unstated).
    pub control_points: u32,
    /// `(set, binding)` of the patch index buffer, where a stage reads it: always a native
    /// pipeline's control stage, an emulated function where it reads its patch index or its
    /// control points.
    pub patch_indices: Option<(u32, u32)>,
    /// For a pipeline with a control-point index type, the bytes of one index (2 or 4), and
    /// `(set, binding)` of the control-point index buffer where a stage reads it (a native
    /// pipeline's control stage, an emulated function that reads its control points); `None` for
    /// a pipeline without one.
    pub control_point_indices: Option<(Option<(u32, u32)>, u64)>,
    /// The stage that reads the factors.
    pub site: FactorSite,
}

/// The stage that reads a tessellation pipeline's factors ([`FactorTemplate`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum FactorSite {
    /// A native pipeline's control stage, whose push range holds the draw's factor and index
    /// buffer fields.
    Control {
        /// The byte offset of the control stage's push range (`TessellationState::push_offset`).
        push_offset: u32,
        /// `(set, binding)` of the factor buffer when the plan writes it (a factor buffer is set);
        /// `None` when the zero buffer stands in.
        binding: Option<(u32, u32)>,
    },
    /// An emulated pipeline's tessellation scan (`DrawCommands::tessellation_dispatch`), whose
    /// draws' vertex stage reads the index buffer fields from its own push window.
    Scan {
        /// The scan the pipeline's draws run.
        kernel: crate::bind::TessellationScanKernel,
        /// Whether the shape sets a factor buffer; without one the scan reads the zero buffer.
        factors: bool,
    },
}

/// The device's and pipeline's rules for a per-bind vertex stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct StrideRules {
    /// `maxVertexInputBindingStride`.
    pub max: u32,
    /// The portability subset's `minVertexInputBindingStrideAlignment`, when the device has one.
    pub portability_alignment: Option<u32>,
    /// The strides are dynamic state (`VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE`): a nonzero
    /// one must cover an element's attributes (`VUID-vkCmdBindVertexBuffers2-pStrides-06209`),
    /// and a draw lists every binding's stride.
    pub dynamic: bool,
    /// The portability subset withholds `vertexAttributeAccessBeyondStride`: a stride, zero
    /// included, must cover an element's attributes.
    pub within_stride: bool,
}

/// One stage of a render plan.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct StageTemplate {
    pub stage: Stage,
    pub flags: vk::ShaderStageFlags,
    /// Every placed push field, in placement order.
    pub fields: Vec<PlacedField>,
    /// `(offset, size)` of the stage's push-constant range.
    pub push_range: Option<(u32, u32)>,
    /// `(set, binding)` of the stage's `ArgumentTable` uniform buffer.
    pub table: Option<(u32, u32)>,
    /// Bytes of the stage's `ArgumentTable` (0 without one).
    pub table_size: u32,
    /// The first root of the resident set (`R0`), when the stage reads its root table.
    pub root_first: Option<u32>,
}

/// One fixed-function vertex buffer of a render plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PlannedVertexBuffer {
    /// The Vulkan vertex binding.
    pub binding: u32,
    /// The Metal vertex-stage buffer slot.
    pub slot: SlotRef,
    /// The pipeline's stride, or `PerBind` (the bind's `attributeStride`, dynamic state).
    pub stride: VertexStride,
    pub input_rate: vk::VertexInputRate,
    /// The instance divisor (1 when not divided).
    pub divisor: u32,
    /// The largest Vulkan address alignment of the binding's attribute formats: the bind offset
    /// and a per-bind stride must be multiples of it.
    pub alignment: u32,
    /// The bytes of one element the attributes read: the smallest attribute offset and the
    /// largest `offset + format size` (`[start, end)`).
    pub reads: (u32, u32),
}

/// The texture resources a draw or dispatch instantiates: texel-buffer ranges, and the heap members
/// and handle tables.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ResourceTemplate {
    pub texel_buffers: Vec<PlannedTexel>,
    /// Every heap member, in element order: `heap_members[i]` is element `i + 1`.
    pub heap_members: Vec<crate::bind::heap::HeapMember>,
    /// Each stage's handle table.
    pub handle_tables: Vec<crate::bind::heap::PlannedHandleTable>,
    /// The level count the binder pushes for each slot whose level count a function reads
    /// (`PushField::TextureLevels`): the bound texture's, 0 where a neutral stands in.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_levels: Vec<(SlotRef, u32)>,
    /// The integer channel widths of the bound texture, which the binder pushes for each slot
    /// whose integer writes read them (`PushField::TextureIntegerChannels`):
    /// `metal::format::integer_channels_word` of its format, 0 where a neutral stands in.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_integer_channels: Vec<(SlotRef, u32)>,
    /// The value the binder pushes for each key a function reads as a push field
    /// (`PushField::Specialized`): its specialization constant's, from the bound resources
    /// (`slots::bound_value`); and the value of each key a specialized note's test alone reads
    /// (`SpecKey::BoundSamplerWord`), which nothing pushes (`slots::record_noted`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub specialized: Vec<(crate::pipeline::sym::SpecKey, u32)>,
    /// Each stage's sampler table.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sampler_tables: Vec<crate::bind::heap::PlannedHandleTable>,
    /// The argument samplers' shape the sampler arrays and tables were planned for, when the
    /// plan has one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub argument_samplers: Option<crate::bind::state::ArgumentSamplersShape>,
    /// What the sampler arrays and tables admit of a sampler state of a shape
    /// `argument_samplers` does not list, as a command buffer lists the states again
    /// (`BindingPlan::relisted_sampler_tables`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub sampler_admission: crate::bind::heap::SamplerAdmission,
    /// Each stage's function-table directory.
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_table_directories: Vec<crate::bind::heap::PlannedHandleTable>,
    /// The tables' shape the function-table array and directories were planned for, when the
    /// plan has one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_tables: Option<crate::bind::state::FunctionTablesShape>,
    /// The directories place each table in the function-table snapshot, not at an element of the
    /// function-table array.
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_table_snapshot: bool,
    /// The directories hold each table's device address, not an element or a snapshot offset: the
    /// pipeline reads the tables' words there.
    #[cfg_attr(feature = "serde", serde(default))]
    pub function_table_addresses: bool,
}

/// One bound texel buffer of a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PlannedTexel {
    pub set: u32,
    pub binding: u32,
    pub slot: SlotRef,
    /// `minTexelBufferOffsetAlignment`.
    pub alignment: u64,
    /// `maxTexelBufferElements`.
    pub max_elements: u64,
    /// The view format's texel block: bytes and texels per block (`vk::Format::block`).
    pub block_size: u64,
    pub texels_per_block: u64,
}

/// One statically used buffer slot of a compute plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct PlannedBuffer {
    pub slot: SlotRef,
    pub window: Window,
    pub form: BufferForm,
    pub access: AccessSet,
    /// `(set, binding)` of a descriptor-form slot.
    pub binding: Option<(u32, u32)>,
    /// The descriptor base alignment: the view's `min*BufferOffsetAlignment`, at least 16
    /// (`views::base_alignment`).
    pub alignment: u64,
    /// The view's `max*BufferRange`.
    pub max_range: u64,
    /// What the shape binds at the slot.
    pub bound: BoundKind,
    /// A `device` slot read through a uniform view: no written slot may be bound overlapping
    /// its window (`RuntimeFact::UniformViewAliasing`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub disjoint: bool,
}

/// What a shape binds at a buffer slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum BoundKind {
    Unbound,
    Buffer {
        device_address: bool,
        /// The allocation's `BindShape` length class.
        length_class: u8,
    },
    Bytes,
}

/// The local-size rules every dispatch region satisfies.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct LocalRules {
    /// The flattened tiers' `(sn, 1, 1)` workgroup (`SpecKey::FlatLocalSize`).
    pub flat: bool,
    /// Every `LocalSizeConstraint{multiple}` of the pipeline.
    pub multiples: Vec<u32>,
    pub max_size: [u32; 3],
    pub max_invocations: u32,
    /// `maxComputeWorkgroupSubgroups`.
    pub max_subgroups: u32,
    pub max_count: [u32; 3],
    pub required_subgroup_size: Option<u32>,
    pub stage_flags: vk::PipelineShaderStageCreateFlags,
    /// The emulated simdgroups' scratch (`RuntimeFact::SimdgroupEmulation`).
    pub scratch: Option<ScratchRule>,
    /// A region of one row needs no whole subgroups (`RuntimeFact::RowLockstep`).
    pub rows: bool,
    /// The pipeline is a tile function (`RuntimeFact::TileFunction`): its threads take Metal's
    /// tile-dispatch layout, built for the threadgroups `dispatch::tile_layout_measured` takes,
    /// dispatched by threadgroups.
    pub tile: bool,
    /// The threadgroup Metal answers a tile function whose threadgroup is the tile
    /// (`RuntimeFact::TileThreadgroup` with `tile_sized`): the shape's tile, its imageblock size,
    /// one deep. Every region of its dispatches carries it as the nominal threadgroup
    /// (`MultiRegion` 1) and the grid of its tiles, whatever threads per tile run.
    pub tile_threadgroup: Option<[u32; 3]>,
}

/// A region of `N` threads takes `fixed + ⌈N / 32⌉ · per_simdgroup` bytes of threadgroup memory,
/// at most `available` (`maxComputeSharedMemorySize`), when the kernel's simdgroups are emulated.
/// `fixed` is the declared arena with its `[[threadgroup(n)]]` regions
/// (`lower::simd::workgroup::arena_bytes`: whole words and at least `MIN_ARENA_BYTES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ScratchRule {
    pub per_simdgroup: u32,
    pub fixed: u64,
    pub available: u32,
}

/// One descriptor write.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorWrite {
    pub set: u32,
    pub binding: u32,
    pub array_element: u32,
    pub ty: vk::DescriptorType,
    pub source: WriteSource,
}

/// What a descriptor write points at.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WriteSource {
    /// The buffer bound at `slot`. The range is derived from the slot's window and bind offset: the
    /// consumer binds exactly that range, never more.
    Buffer { slot: SlotRef },
    /// `set*Bytes` data uploaded by the consumer.
    InlineBytes { slot: SlotRef },
    /// The texture bound at `slot`, through this view.
    Texture { slot: SlotRef, view: ImageViewDesc },
    /// A combined image sampler: its image and its sampler.
    CombinedImageSampler {
        image: ImageSource,
        sampler: SamplerSource,
    },
    /// The planes of the two-plane YCbCr texture bound at `slot`, read in place
    /// (`TextureFacts::ycbcr_in_place`): a uniform texel buffer over the planes' bytes in
    /// `YCbCrInPlace::format`, from the first byte of the first plane for
    /// `YCbCrInPlace::view_bytes`. Only a slot under its in-place variant binds it
    /// (`VariantKey::ycbcr_in_place`).
    YCbCrInPlace { slot: SlotRef },
    /// The planes of the two-plane YCbCr texture bound at `slot`, held as an image the device's
    /// own sampler Y′CbCr conversion samples (`TextureShape::Texture::ycbcr_planes_sampler`): a
    /// combined image sampler whose image is that image viewed with the conversion
    /// (`bind::YCbCrPlanesSampler`) and whose sampler is the binding's immutable one, in
    /// `GENERAL`. Only a slot under its planes-image variant binds it
    /// (`VariantKey::ycbcr_planes_sampled`, `BindingSource::YCbCrPlanes`).
    YCbCrPlanesImage { slot: SlotRef },
    /// The texture buffer bound at `slot`, as a texel buffer view.
    TexelBuffer {
        slot: SlotRef,
        format: vk::Format,
        window: Window,
    },
    /// A `VkSampler` for the sampler bound at `slot`, made for this use: `compare` enables depth
    /// comparison and `integer` selects integer border colors (`metal::sampler::SamplerUse`,
    /// `bind::sampler_desc`). The plan checks what the shape states
    /// (reduction); the rest of the descriptor is per bind. `opaque_black` says how an opaque-black
    /// border is made to read Metal's texel through every texture the sampler meets
    /// (`metal::sampler::OpaqueBlackBorder`).
    Sampler {
        slot: SlotRef,
        compare: bool,
        integer: bool,
        opaque_black: OpaqueBlackBorder,
    },
    /// The stage's argument table; instantiation fills the bytes.
    ArgumentTable { stage: Stage, bytes_len: u32 },
    /// A handle table of `stage` (its texture table, or its sampler table), a read-only storage
    /// buffer of `bytes_len` bytes whose contents instantiation gives (`DescriptorBytes`). The
    /// consumer binds exactly `bytes_len` bytes: the shader counts the table's entries from the
    /// bound range.
    HandleTable { stage: Stage, bytes_len: u32 },
    /// The constant blob of `stage`: its `StageModule::constant_blob`, padded to 16 bytes by the
    /// consumer.
    ConstantBlob { stage: Stage },
    /// Element `element` of the set-1 heap array of `class`: the resident texture `source`,
    /// through this view. Every write of one texture has one layout: `GENERAL` when some storage
    /// array holds it or, for a texture bound at a slot, the slot's own binding is a storage image;
    /// `SHADER_READ_ONLY_OPTIMAL` otherwise.
    HeapEntry {
        class: HandleClass,
        element: u32,
        source: HeapSource,
        view: ImageViewDesc,
    },
    /// Color attachment `color_index`, as an input attachment.
    Attachment { color_index: u32 },
    /// A neutral stand-in: statically used, nothing (compatible) bound.
    Neutral(NeutralResource),
    /// The encoder's tessellation factor buffer (`BindState::tessellation_factors`), a
    /// read-only storage buffer of the control stage over the draw's `DescriptorRange`.
    TessellationFactors,
    /// The top-level structure of the acceleration structure bound at buffer slot `slot`
    /// (`setAccelerationStructure:atBufferIndex:`): the bound instance structure itself when
    /// `level` is `Instance`, and, when `level` is `Primitive`, the top-level structure of one
    /// instance over the bound primitive structure
    /// ([`primitive_wrap`](crate::bind::acceleration::primitive_wrap)). A slot the guest left
    /// unbound takes a top-level structure of no instances, which every ray misses.
    AccelerationStructure {
        slot: SlotRef,
        level: AccelerationStructureLevel,
    },
    /// The render pass's fragment imageblock storage (`GraphicsState::fragment_imageblock`): the
    /// one storage image the consumer creates for the pass instance, in `GENERAL` layout, bound to
    /// every draw of the pass, and to every tile dispatch whose function's explicit imageblock
    /// stands alone, which stages its tile's cells from it and back.
    Imageblock,
    /// A render pass's color attachment `color_index` as a storage image: the view `view` names,
    /// in `format`, with the identity component mapping, in `GENERAL` layout. For
    /// [`StorageAttachmentView::Attachment`] that is a 2D view of the attachment's own image, of
    /// the pass's layer and level; for [`StorageAttachmentView::SamplePlanes`] `{ layers }` a
    /// 2D-array view of all `layers` layers of the attachment's sample planes, the single-sample
    /// image the consumer keeps beside a multisampled attachment, never a 2D view.
    ///
    /// - A tile dispatch's: the image's format is the tile pipeline descriptor's
    ///   (`TileRenderPipelineDescriptor::color_attachment_formats`), as Metal requires of the
    ///   pass, and its extent at least the render area (`BindState::render_area`). `format` is
    ///   the image's own for an integer format, and for a float format the unsigned-integer
    ///   format of the same channels and widths (`R16_UINT` for `R16_SFLOAT`), so the dispatch
    ///   moves the texels' bits: the consumer creates such an image with `MUTABLE_FORMAT` (and
    ///   `EXTENDED_USAGE` when its own format is not storage-capable).
    ///   `view` is always `Attachment`.
    /// - A draw's whose fragment function reads the attachment in the interlocked storage form
    ///   (`GraphicsState::storage_attachments`): `format` and `view` are the pipeline's
    ///   [`StorageAttachmentUse`](crate::pipeline::StorageAttachmentUse)'s, and the image has
    ///   `STORAGE` usage and is no color attachment of the subpass.
    TileAttachment {
        color_index: u32,
        format: vk::Format,
        view: StorageAttachmentView,
        /// A draw's: the pipeline's `StorageAttachmentUse::view_alias`, where the view is in that
        /// format, of an image the consumer made mutable to it (an sRGB attachment's
        /// unsigned-integer format of its texel size, a BGRA-order one's RGBA-order twin on a
        /// device that accesses its format without one in neither direction); `None` elsewhere,
        /// and for a tile dispatch's, whose view is `format`.
        #[cfg_attr(feature = "serde", serde(alias = "uint_view"))]
        view_alias: Option<vk::Format>,
    },
    /// The tile map of the sparse texture bound at texture slot `slot` (`bind::SparseTileMap`),
    /// a read-only storage buffer of [`SparseTileMap::byte_len`](crate::bind::SparseTileMap)
    /// bytes, from which a function answers the texture's residency where the device has no
    /// shader residency. The consumer binds the map it keeps for that texture, holding the mappings
    /// Metal's would hold when the command runs. A slot whose bound texture is not sparse takes the
    /// zero buffer instead: the module reads no tile map there.
    TileMap { slot: SlotRef },
    /// Element `index + 1` of a set-1 sampler array: a `VkSampler` for
    /// `BindState::argument_samplers[index]`, made for this use as `Sampler` is
    /// (`bind::sampler_desc` with `compare` and `integer`), or the neutral sampler
    /// (`NeutralResource::Sampler { compare }`) where the state lists fewer samplers: no handle
    /// names that element then. The plan writes every element of its shape's capacity class
    /// (`ArgumentSamplersShape::entries`), so the list may grow or shrink within the class
    /// without a new plan; the consumer resolves each write against the state it binds.
    ArgumentSampler {
        index: u32,
        compare: bool,
        integer: bool,
    },
    /// The shadow of the render pass's stencil attachment that a fragment function's
    /// shader-side stencil tests and writes (`GraphicsState::shadow_stencil`): the consumer's
    /// `R8_UINT` 2D storage image of the attachment's extent, in `GENERAL` layout
    /// ([`ShadowStencilUse`](crate::pipeline::ShadowStencilUse)).
    ShadowStencil,
    /// Element `index + 1` of the set-1 function-table array: the words of
    /// `BindState::function_tables[index]` (`pipeline::table`'s layout, as
    /// [`FunctionTableWords`](crate::bind::FunctionTableWords) encodes them), a read-only
    /// storage buffer the consumer binds whole, or the zero buffer
    /// (`NeutralResource::ZeroBuffer`) where the state lists fewer tables: no directory entry
    /// names that element then. The plan writes every element of its shape's capacity class
    /// (`FunctionTablesShape::entries`), so the list may grow or shrink within the class without
    /// a new plan; the consumer resolves each write against the state it binds.
    FunctionTable { index: u32 },
    /// A patch draw's patch index buffer (`DrawCall::Patches::patch_index`), a read-only storage
    /// buffer of the control stage (an emulated tessellation pipeline's vertex stage) over the
    /// draw's `DescriptorRange`. A draw without one has no range for it, and the consumer binds
    /// the zero buffer of `min_bytes` there (`NeutralResource::ZeroBuffer`'s), whose words the
    /// stage discards: the draw pushes `PushField::PatchIndexBound` 0.
    PatchIndices { min_bytes: u32 },
    /// An indexed patch draw's control-point index buffer
    /// (`DrawCall::Patches::control_point_index`), a read-only storage buffer of the control
    /// stage (an emulated tessellation pipeline's vertex stage) of a pipeline with a control-point
    /// index type over the draw's `DescriptorRange`. Every draw of such a pipeline has one.
    ControlPointIndices,
    /// The render pass's tile threadgroup memory (`SyntheticBinding::TileMemory`): a storage
    /// buffer the consumer keeps for the pass, of at least `bytes` bytes (every tile of the
    /// render area, each `SpecKey::TileThreadgroupLength` bytes), zeroed when the pass begins and
    /// kept across its tile dispatches, bound whole.
    TileMemory { bytes: u64 },
    /// The function-table snapshot (`SyntheticBinding::FunctionTableSnapshot`) of a device
    /// without the set-1 function-table array's features: a read-only storage buffer of the
    /// words of `BindState::function_tables`, laid out as
    /// [`FunctionTableSnapshot`](crate::bind::FunctionTableSnapshot) of the listed sizes gives
    /// them, bound whole. The consumer writes it as the command's command buffer commits: the
    /// command reads each table's entries as they stand then, where Metal reads them as they
    /// stand when the command runs, which differs only for a CPU write racing the GPU. The
    /// stage's directory gives each listed id its record's offset, by the same sizes.
    FunctionTableSnapshot,
    /// An emulated mesh function's records (`SyntheticBinding::MeshRecords`): the draw's
    /// `MeshDispatch::records`, a region of its own or the command buffer's records pool, which
    /// the consumer binds whole for the mesh dispatch and the draw alike
    /// ([`MeshDispatch`](crate::bind::MeshDispatch)).
    MeshRecords,
    /// An emulated mesh draw's work buffer (`SyntheticBinding::MeshWork`, the object and mesh
    /// stages' alike): the draw's region of `MeshDispatch::work_bytes`, holding
    /// `MeshDispatch::work`, bound whole ([`MeshDispatch`](crate::bind::MeshDispatch)).
    MeshWork,
    /// An emulated tessellation draw's work buffer (`SyntheticBinding::TessellationWork`, the
    /// vertex stage's): the draw's region holding
    /// [`TessellationDispatch::work`](crate::bind::TessellationDispatch::work), bound whole
    /// ([`TessellationDispatch`](crate::bind::TessellationDispatch)).
    TessellationWork,
    /// An emulated tessellation draw's records (`SyntheticBinding::TessellationRecords`, the
    /// vertex stage's): the draw's `TessellationDispatch::records`, a region of its own or the
    /// command buffer's records pool, bound whole for the scan and the draw alike
    /// ([`TessellationDispatch`](crate::bind::TessellationDispatch)).
    TessellationRecords,
    /// The command buffer's execution status (`SyntheticBinding::ExecutionStatus`) of a stage on
    /// a device that does not address buffers: a storage buffer of the
    /// [`ExecutionStatus`](crate::bind::ExecutionStatus) words, the same buffer every command of
    /// the command buffer writes (where the device addresses buffers, the one whose address
    /// `PushHole::ExecutionStatusAddress` takes), its
    /// [`ExecutionStatus::BYTES`](crate::bind::ExecutionStatus::BYTES) bytes bound, starting all
    /// zero ([`ExecutionStatus::initial`](crate::bind::ExecutionStatus::initial)). The consumer
    /// reads it once the command buffer completes.
    ExecutionStatus,
    /// The stand-in of the render pass's depth attachment that a fragment function that runs
    /// the draw's depth test itself tests and writes (`GraphicsState::shader_tests`): the
    /// consumer's `R32_SFLOAT` storage image of the attachment's extent, holding its samples as
    /// the stand-in's view says, in `GENERAL` layout
    /// ([`StandInUse`](crate::pipeline::StandInUse)).
    DepthStandIn,
    /// The `half` shadow of the render pass's extended-range color attachment `color_index`
    /// (`BindingSource::ExtendedShadow`): the consumer's `R16G16B16A16_SFLOAT` single-sample 2D
    /// storage image of the attachment's extent, in `GENERAL` layout, holding Metal's tile value
    /// of each texel of the pass (`VariantKey::extended_shadows`).
    ExtendedShadow { color_index: u32 },
    /// The snapshot of the texture the draw binds at fragment texture slot `slot`
    /// (`BindingSource::SnapshotTexels`, `VariantKey::snapshot_textures`): the consumer's
    /// single-sample 2D `R32G32B32A32_UINT` storage image covering the draw's render area, in
    /// `GENERAL` layout, which the draw's fragments write at their pixels in the draw that takes
    /// the snapshot and read in the draw that reads it (`PushHole::SnapshotPhase`).
    SnapshotTexels { slot: u32 },
    /// The stand-in of the render pass's stencil attachment that a fragment function that runs
    /// the draw's stencil test itself tests and writes (`ShaderTestsUse::stencil`): the
    /// consumer's `R8_UINT` storage image of the attachment's extent and the pipeline's samples,
    /// in `GENERAL` layout ([`StandInUse`](crate::pipeline::StandInUse)).
    StencilStandIn,
}

/// The image of a combined image-sampler write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageSource {
    /// The texture bound at `slot`, through this view.
    Texture { slot: SlotRef, view: ImageViewDesc },
    /// A neutral image of the declared type.
    Neutral(NeutralResource),
}

/// The sampler of a combined image-sampler write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerSource {
    /// A `VkSampler` for the runtime sampler bound at `slot`, made for this use (as
    /// `WriteSource::Sampler`).
    Slot {
        slot: SlotRef,
        compare: bool,
        integer: bool,
        opaque_black: OpaqueBlackBorder,
    },
    /// The layout's immutable (constexpr) sampler: the write carries only the image.
    Immutable,
    /// The neutral sampler: the runtime sampler slot is unbound.
    Neutral(NeutralResource),
}

/// The view the consumer creates for a texture write.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageViewDesc {
    pub view_type: vk::ImageViewType,
    /// The view format: the texture's own (an 8-bit sRGB texture's UNORM twin where the device
    /// holds one), or the storage alias the variant chose.
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    /// The layout the texture is in while the draw or dispatch runs. Every write of one slot's
    /// texture or one heap member has the same layout, `GENERAL` when one of them is a storage
    /// image; the plan cannot see that two slots or members hold one texture.
    pub layout: vk::ImageLayout,
    pub components: vk::ComponentMapping,
    /// The levels the view holds: `None` every level of the texture, `Some(i)` level `i` alone,
    /// an element of a storage slot's level views (`RuntimeFact::StorageTextureLevelViews`). Levels
    /// count from the bound texture's own first level: level `i` of a Metal texture view whose
    /// first level is its parent's level `b` is the parent's level `b + i`. Every layer either way.
    #[cfg_attr(feature = "serde", serde(default))]
    pub level: Option<u32>,
    /// The function applies the bound texture's swizzle (`MTLTexture.swizzle`) to the texels it
    /// reads and writes through the view, so the view takes none: its component mapping is
    /// `components` alone. Set for a storage view of a swizzled texture at a slot of
    /// `VariantKey::swizzled_storage`, which a Vulkan storage view could not swizzle, and
    /// for the sampled view of a swizzled extended-range texture at a slot of it, whose raw codes
    /// the view's swizzle would move before the function decodes them, and of a swizzled packed
    /// normalized texture, whose codes the function converts by each channel's own scale. Set too,
    /// under no variant, for the storage view of a swizzled texture at a slot the function only
    /// queries: it reads and writes no texel, and no swizzle reaches what a query answers. `false`
    /// for every other view, on which the consumer composes the texture's swizzle with
    /// `components` (and a swizzled storage view is the consumer's refusal).
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_swizzle: bool,
}

impl ImageViewDesc {
    /// A view of type `view_type` and format `format` over the aspect `aspect` of every level, in
    /// layout `layout`, with the component mapping `components`, no twin, and the texture's
    /// swizzle for the consumer to compose.
    pub fn new(
        view_type: vk::ImageViewType,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
        components: vk::ComponentMapping,
    ) -> ImageViewDesc {
        ImageViewDesc {
            view_type,
            format,
            aspect,
            layout,
            components,
            level: None,
            shader_swizzle: false,
        }
    }
}

/// A Vulkan vertex buffer binding fed by a Metal vertex-stage buffer slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexBufferBind {
    pub binding: u32,
    /// The Metal buffer index in the vertex stage.
    pub buffer_index: u32,
    /// The stride comes from the bind (`attributeStride`), as dynamic state.
    pub per_bind_stride: bool,
}

/// Which push fields instantiation fills.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushTemplate {
    /// In placement order per stage.
    pub fields: Vec<PushFieldPlacement>,
}

/// Where one push field of one stage lives.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushFieldPlacement {
    pub stage: Stage,
    pub field: PushField,
    pub location: FieldLocation,
    pub size: u32,
}

/// A push field's location: the stage's push window, or its spill to the argument table.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FieldLocation {
    /// Byte offset in the pipeline's push-constant block.
    PushWindow { offset: u32 },
    /// Byte offset in the stage's argument table.
    ArgumentTable { offset: u32 },
}

/// Something the plan did that the consumer may want to log.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlanDiagnostic {
    /// The resource bound at `slot` does not match the shader's declared type; a neutral of the
    /// declared type is written instead (guest undefined behavior).
    IncompatibleBinding {
        slot: SlotRef,
        substituted: NeutralResource,
    },
    /// The resident texture `source` matches the heap array of `class` in everything but its
    /// numeric type, and the array's numeric type is only the default of a heap group that is
    /// only queried (`RuntimeFact::HeapNumericType`): element `element` of the array is the
    /// neutral, so a query through a handle to `source` answers the neutral's.
    IncompatibleHeapEntry {
        class: HandleClass,
        element: u32,
        source: HeapSource,
        substituted: NeutralResource,
    },
    /// The texture bound at `slot` matches the shader's declared type in everything but its
    /// numeric type (its format's class, float or signed or unsigned integer, is not the declared
    /// one), and the plan has no exact answer for it: plain Metal reads such a texture in its own
    /// class and hands back the words as the declared type (and stores a write's lanes' bits as
    /// its own class), which the slot's variant (`VariantKey::cross_class`) gives for a color
    /// texture where the function's compile states `RuntimeFact::CrossClassReads` (storage slots,
    /// texture buffers, multisampled and sparse textures, writes, `short` and `ushort` results and
    /// queries among them). A depth, stencil, extended-range or YCbCr texture, a depth slot, an
    /// atomic or `ulong` access, a written slot over a swizzled texture and an integer write to an
    /// `RGB9E5Float` texture are not measured. A neutral of the declared type is written instead,
    /// which the runtime announces (`cross-class-texture-neutral`).
    CrossClassNeutral {
        slot: SlotRef,
        substituted: NeutralResource,
    },
    /// The resident texture `source`, which the `stage` function reaches, matches the heap array
    /// of `class` in everything but its numeric type, and the stage reads that class by an access
    /// with no answer for it (`RuntimeFact::HeapCrossClassNeutral`: a sample, gather or LOD
    /// query, a write or atomic, a storage or multisampled group's read, a `short`, `ushort` or
    /// `ulong` read; or any access, for a depth, stencil, extended-range or two-plane texture):
    /// element `element` of the array is the neutral, which such an access reads, where plain
    /// Metal reads the texture in its own class. The runtime announces it
    /// (`cross-class-texture-neutral`).
    CrossClassHeapNeutral {
        stage: Stage,
        class: HandleClass,
        element: u32,
        source: HeapSource,
    },
    /// The texture bound at storage slot `slot` (a function's write, or read and write) is an
    /// ASTC texture, whose blocks no storage view stores into: plain Metal completes the
    /// function's writes, storing bytes no format defines into the blocks, and reads the blocks'
    /// decode through a slot that also reads them. A neutral of the declared type is written
    /// instead (its writes dropped, its reads zero, its size queries the neutral's), which leaves
    /// the texture's blocks as they were; the runtime announces it (`astc-storage-neutral`).
    AstcStorageNeutral {
        slot: SlotRef,
        substituted: NeutralResource,
    },
}

/// How a draw or dispatch uses one resource: staging, writeback, and hazard facts. A plan's `uses`
/// are the shape's (bound slots, used resources); `DrawCommands.uses` and `DispatchCommands.uses`
/// add each command's index buffer and indirect arguments.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceUse {
    pub resource: UseTarget,
    /// Proven from lowering; unknown access is `READ_WRITE`.
    pub access: AccessSet,
    pub class: AccessClass,
    pub stages: vk::PipelineStageFlags2,
    /// `None`: the whole resource from the bound offset. The bound offset of a slot is its bind
    /// offset; of the index buffer, the draw's index offset; of the indirect buffer, the
    /// arguments' offset.
    pub range: Option<ByteRange>,
}

/// The resource a [`ResourceUse`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UseTarget {
    /// The resource bound at a slot.
    Slot(SlotRef),
    /// A used resource, by consumer token.
    Used(u32),
    /// The draw's index buffer.
    IndexBuffer,
    /// The draw's or dispatch's indirect buffer.
    Indirect,
    /// The encoder's tessellation factor buffer, from its bind offset.
    TessellationFactors,
    /// The tile map of the sparse texture bound at a texture slot, whole: the consumer's update of
    /// the map before the command (a mapping change) needs a barrier to this read.
    TileMap(SlotRef),
    /// A patch draw's patch index buffer, from the draw's offset into it.
    PatchIndices,
    /// An indexed patch draw's control-point index buffer, from the draw's offset into it.
    ControlPointIndices,
}

/// How a resource is accessed, for barrier stage/access selection.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AccessClass {
    Storage,
    Uniform,
    Sampled,
    StorageImage,
    VertexInput,
    Index,
    Indirect,
    Attachment,
    /// A two-plane YCbCr texture's planes, read in place through the slot's uniform texel buffer
    /// (`WriteSource::YCbCrInPlace`) or the image of them the device's conversion samples
    /// (`WriteSource::YCbCrPlanesImage`): a read of the planes' bytes, not of the texture's image.
    YCbCrPlanes,
}

/// A byte range relative to the bound offset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}
