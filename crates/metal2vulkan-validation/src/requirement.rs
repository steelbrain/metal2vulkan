use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// An executor of authored cases: the Metal oracle that qualifies a case, and the Vulkan candidate
/// whose observations are compared against it. The old engine's candidate was retired with the old
/// engine, so a closed authoring-capability census needs every executor to run every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Executor {
    MetalOracle,
    /// The engine's candidate (`candidate::engine`), which runs the shipping product.
    EngineCandidate,
}

impl Executor {
    pub const ALL: [Self; 2] = [Self::MetalOracle, Self::EngineCandidate];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MetalOracle => "metal_oracle",
            Self::EngineCandidate => "engine_candidate",
        }
    }
}

impl fmt::Display for Executor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Whose limit a [`ToolingRequirement`] records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Limit {
    /// This crate's tooling does not yet express, run, or observe the structure: work to model.
    /// Its rows keep the census open.
    Tooling,
    /// Metal itself cannot build a pipeline for the structure: Apple's compiler fails, as measured
    /// on the reference device, so no Metal answer exists for any tooling to produce. It stops
    /// every executor (no case can be qualified), and its rows are classified and closed without a
    /// reference: the census counts them apart and never as unresolved.
    Metal,
    /// `metal2vulkan-runtime` answers the structure with a typed refusal, or offers no call for it
    /// and the consumer refuses it itself (its README's known gaps): the Metal oracle runs it, the
    /// engine's candidate refuses it, typed, and its rows are classified and closed without a
    /// candidate run, counted apart and never as unresolved. A device test holds each such
    /// requirement to the refusal both ways on lavapipe where the runtime has a call for it: the
    /// shape the requirement's check accepts gets the named refusal, and the same shape without it
    /// is made. The day the runtime builds the shape that test fails, and the limit goes back to
    /// [`Limit::Tooling`] while the executor's support is built.
    Runtime,
}

impl Limit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tooling => "tooling",
            Self::Metal => "metal",
            Self::Runtime => "runtime",
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A concrete gap between harvested AIR structure and the authored execution contract.
///
/// This is the shared vocabulary for classification, schema/executor capability checks, cached
/// census rows, and focused audits. Keeping it typed prevents one component from silently
/// inventing a requirement name that another component cannot select or resolve. Each
/// requirement names the executors it stops ([`ToolingRequirement::stops`]): an authoring
/// requirement (no case can express or observe the structure) stops both, and an executor
/// requirement only the executor that does not implement the family. Each also names whose limit it
/// is ([`ToolingRequirement::limit`]): this crate's tooling, or Metal's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolingRequirement {
    FunctionConstantLiteral,
    VertexAttributeLiteral,
    TessellationAttributeLiteral,
    TessellationSystemInputLiteral,
    FragmentImageblockMemberLiteral,
    KernelStageInputLiteral,
    /// A texture buffer that is no entry parameter of its own: one an argument buffer holds, or an
    /// element of a texture-handle array (`executor_contract`). The engine's candidate runs
    /// neither. The runtime makes no texture buffer (`metal2vulkan_runtime::Gap::TextureBuffer`
    /// at `new_texture`), and its identity domains hold no texture buffer among an argument
    /// buffer's textures, so it refuses such a function's pipeline at its make, typed
    /// (`Unsupported{IdentityDomains}`, or, on a device whose answer lacks the features an
    /// argument buffer's texel buffers are read through, `Feature::ArgumentBufferTextures` at
    /// that query first): the limit is the runtime's ([`Limit::Runtime`]), held both ways by the
    /// device tests
    /// `candidate::tests::engine_compute::a_texture_buffer_is_refused_where_a_2d_texture_is_made`
    /// and `..::a_texture_buffer_member_is_refused_at_the_make_where_a_2d_member_is_made`. The
    /// Metal oracle makes and binds both, so a case of either stays Metal-qualified
    /// ([`Self::qualifies_on_metal`]).
    TextureBufferMember,
    StorageTextureFormatLiteral,
    ImplicitImageblockLiteral,
    IndirectCommandBuffer,
    VertexSideEffectObservation,
    FragmentVaryingObservationType,
    FragmentOutputObservationType,
    /// An intersection-function table held in an argument buffer.
    ArgumentBufferIntersectionFunctionTable,
    /// An intersection-function table entry that links an intersection function.
    LinkedIntersectionFunction,
    /// A device-buffer array bound to a vertex or fragment function.
    RenderDeviceBufferArray,
    /// An acceleration structure or intersection-function table bound to a vertex or fragment
    /// function.
    RenderAccelerationStructure,
    /// A depth/stencil attachment of a vertex case.
    VertexDepthStencilAttachment,
    /// Render targets of a rasterization-disabled vertex case.
    RasterizationDisabledRenderTarget,
    /// A coordinate-mapping call reads its rasterization-rate map from somewhere other than
    /// offset 0 of an entry buffer parameter (`executor_contract::rate_map`): the
    /// `rasterization_rate_map` literal places a map only there, so the map stays authored bytes,
    /// which the engine's module decodes as this crate's parameter data.
    RasterizationRateMapPlacement,
    /// A compute kernel that samples a texture at an implicit level with a positive (or NaN)
    /// constant or run-time `min_lod_clamp`, in a form measured to crash Apple's compiler service
    /// building its compute pipeline (`executor_contract::min_lod_clamp`), so no Metal reference
    /// exists for it ([`Limit::Metal`]).
    KernelMinLodClamp,
    /// A stage of a mesh render pipeline that its row or case needs and nothing supplies
    /// (`executor_contract::mesh`): the mesh stage of an object function whose payload the
    /// generated launch observer cannot read, the object stage of a mesh function whose payload no
    /// generated object function writes, or a row's fragment stage where its mesh function's
    /// observable effect is its rasterized output. A case supplies one by naming a partner of its
    /// own module (`MeshDraw`, whose rasterizing draw must name its fragment function), or by
    /// stating the observation or object launch whose generated stages take the partners' places
    /// (`companion::MeshStages`).
    MeshCompanion,
    /// A case gives a 32-bit float depth image a depth outside `[0, 1]`
    /// (`executor_contract::depth_range`): a `Depth32Float` texture's texels (a texture-array
    /// element's or an argument buffer's texture's among them) or a depth attachment's initial
    /// depth holds a NaN, a negative value other than `-0.0`, or a value above 1.0, which Metal's
    /// texel keeps. The runtime takes no upload of a 32-bit float depth on any device
    /// (`metal2vulkan_runtime::Gap::FloatDepthUpload`, which its own texture and blit tests hold
    /// both ways), and a draw, the one way a guest fills such a texture through it, writes depths
    /// clamped to the viewport's `[0, 1]`: no call of the runtime puts such a value where Metal
    /// holds it, so the limit is the runtime's ([`Limit::Runtime`]) and the engine's candidate
    /// does not run the case. The case checker raises it from the case's literals on every device
    /// (`executor_contract::case_requirements`), never a source's structure, so no census row
    /// carries it. The Metal oracle runs it, so a case of it stays Metal-qualified
    /// ([`Self::qualifies_on_metal`]).
    DepthOutsideUnitRange,
    /// An object or mesh function's row or case (`executor_contract::mesh`), or a case that runs a
    /// pipeline template of one: the engine's candidate draws no mesh render pipeline. The runtime
    /// has no mesh call: it makes no mesh render pipeline state and records no mesh draw, and the
    /// consumer refuses mesh pipelines itself (`crates/metal2vulkan-runtime/README.md`, its known
    /// gaps), so the limit is the runtime's ([`Limit::Runtime`]) with no call to hold a device
    /// test to; that README sentence is its guard. When a guest uses a mesh pipeline it is built
    /// natively with its call, and the limit goes back to [`Limit::Tooling`] then. The Metal
    /// oracle draws it, so a case of it stays Metal-qualified ([`Self::qualifies_on_metal`]), and
    /// the authored compile census still compiles an object or mesh case's pipeline
    /// (`engine::prepare_compile`).
    MeshRenderPipeline,
    /// A replayed pipeline template the engine's candidate does not replay
    /// (`candidate::engine::replay_templates`): a compute template that binds more than buffers
    /// and threadgroup memory, or a render template that is not a rasterization-disabled vertex
    /// function binding buffers only (a template of a family the runtime does not build is refused
    /// as that family first). A case's templates raise it, never a source's structure, so no census
    /// row carries it.
    ReplayedTemplateResources,
    /// A sequentially consistent fence its entry's pipeline compiles
    /// (`executor_contract::seq_cst_fence`): an `atomic_thread_fence` of `memory_order_seq_cst`
    /// that orders a memory beyond its thread, or a `seq_cst` barrier of the MSL 4.1 form whose
    /// memory scope is the device. The translator orders such a fence with the command buffer's
    /// other ones through an atomic on its execution status word, which the runtime's identity
    /// domains do not reach: the runtime refuses the pipeline (`Unsupported{IdentityDomains}`,
    /// naming the fence, "a sequentially consistent fence"), so the limit is the runtime's
    /// ([`Limit::Runtime`]) and the engine's candidate does not run the case. The device tests
    /// `candidate::tests::engine_compute::a_seq_cst_fence_is_refused_through_the_runtime_as_its_requirement_says`
    /// and `..::a_fence_of_another_order_is_made_through_the_runtime` hold it both ways. The Metal
    /// oracle runs it, so a case of it stays Metal-qualified ([`Self::qualifies_on_metal`]).
    SeqCstFence,
    /// A visible function table bound to a vertex or fragment function (`executor_contract`'s
    /// `engine_source_requirements` and `case_requirements`): function pointers called from a
    /// render pipeline, which Metal lets a program ask for through
    /// `supportsFunctionPointersFromRender`, an Apple6-family feature. The runtime's render
    /// pipeline state reads no function table (its make is
    /// `metal2vulkan_runtime::Gap::RenderPipeline` of `DrawInput`, a known gap its README states,
    /// which is its guard), so the limit is the runtime's ([`Limit::Runtime`]) and the engine's
    /// candidate does not run the case. The Metal oracle draws it, so a case of it stays
    /// Metal-qualified ([`Self::qualifies_on_metal`]).
    RenderFunctionTable,
    /// An indexed draw that reads past its index buffer: a case's out-of-bounds target of a draw's
    /// index buffer (`executor_contract::case_requirements`). The runtime refuses a draw whose
    /// indices run past the buffer, as Metal's validation stops it
    /// (`metal2vulkan_runtime::Refusal::IndexBufferOffset`, a refusal its README states, which is
    /// its guard), so the limit is the runtime's ([`Limit::Runtime`]) and the engine's candidate
    /// does not run the case. The case checker raises it from the case's literals, never a
    /// source's structure, so no census row carries it. The Metal oracle draws it, so a case of it
    /// stays Metal-qualified ([`Self::qualifies_on_metal`]).
    IndexBufferPastEnd,
    /// A 64-bit texture atomic its module calls (`executor_contract::unsupported_air_requirements`):
    /// an `atomic_min` or `atomic_max` on a Metal `ulong` texture, an Apple8-family feature (Metal's
    /// feature set tables: "GPU devices in the Apple8 family support 64-bit atomic minimum and
    /// maximum using ulong, on both buffers and textures, only on macOS"). The translator lowers it
    /// to a 64-bit image atomic, which needs `shaderImageInt64Atomics`, and the runtime enables no
    /// 64-bit image atomic on any device (its floor's features), so it refuses the pipeline on
    /// every device as `Unsupported{ImageAtomics}` needing that feature
    /// (`device_capability_missing`), a known gap its README states, which is its guard: the limit
    /// is the runtime's ([`Limit::Runtime`]) and the engine's candidate does not run the case. The
    /// Metal oracle runs it, so a case of it stays Metal-qualified ([`Self::qualifies_on_metal`]).
    UlongTextureAtomic,
    /// A depth/stencil attachment of both aspects (`executor_contract::case_requirements`), which
    /// the Metal oracle makes as Metal's separate `Depth32Float` and `Stencil8` textures, both
    /// attached to one pass: plain Metal on every Mac. The runtime does not build that form yet: it
    /// renders into no separate depth and stencil textures in one pass
    /// (`metal2vulkan_runtime::Gap::RenderPass` of `DepthStencilViews`, and its pipeline state
    /// `Gap::RenderPipeline` of `DepthStencilFormats`, known gaps its README states, which are its
    /// guard), so the limit is the runtime's ([`Limit::Runtime`]) and the engine's candidate does
    /// not run the case. The case checker raises it from the case's literals, never a source's
    /// structure, so no census row carries it. The Metal oracle draws it, so a case of it stays
    /// Metal-qualified ([`Self::qualifies_on_metal`]).
    SeparateDepthStencil,
    /// A visible function table bound to a compute kernel (`executor_contract::case_requirements`;
    /// a render function's is [`Self::RenderFunctionTable`]). The runtime's compute encoder has no
    /// `setVisibleFunctionTable:atBufferIndex:`, and its make of a pipeline state that reads a
    /// table is `metal2vulkan_runtime::Gap::KernelInput` of `KernelInputKind::Slot`, a known gap
    /// its README states, held on lavapipe by the runtime's device test
    /// `tests::kernel_input::a_visible_function_table_kernel_is_refused_as_a_slot`.
    KernelFunctionTable,
    /// A compute kernel's explicit imageblock (`executor_contract::case_requirements`), a tile
    /// pass's tile size included. The runtime's compute encoder has no
    /// `setImageblockWidth:height:`, and its make of a pipeline state that reaches an imageblock is
    /// `metal2vulkan_runtime::Gap::KernelInput` of `KernelInputKind::RuntimeFact` naming
    /// `ImageblockSize`, held on lavapipe by the runtime's device test
    /// `tests::kernel_input::an_explicit_imageblock_kernel_is_refused_as_its_named_runtime_fact`;
    /// it builds no tile function. Both are known gaps its README states.
    KernelImageblock,
    /// A texture buffer a case binds to an entry parameter (`executor_contract::case_requirements`;
    /// one an argument buffer holds is [`Self::TextureBufferMember`]). The runtime makes no texture
    /// buffer (`metal2vulkan_runtime::Gap::TextureBuffer` at `new_texture`), a known gap its README
    /// states, held both ways on lavapipe by the device test
    /// `candidate::tests::engine_compute::a_texture_buffer_is_refused_where_a_2d_texture_is_made`.
    TextureBuffer,
    /// A fragment function's imageblock, the case's cells of it
    /// (`executor_contract::case_requirements`). The runtime refuses a fragment function with an
    /// `[[imageblock_data]]` input or output at creation on every device (`Error::Translation`
    /// naming `Imageblocks`, `NotImplemented`), a known gap its README states, which is its guard.
    FragmentImageblock,
    /// A sparse texture a case binds (`executor_contract::case_requirements`). The runtime makes
    /// no sparse texture: its texture descriptor has no sparse form, so no call of it makes one.
    SparseTexture,
    /// An acceleration structure or an intersection-function table bound to a compute kernel
    /// (`executor_contract::case_requirements`; a render function's is
    /// [`Self::RenderAccelerationStructure`]). The runtime makes no acceleration structure, and its
    /// compute encoder binds neither, so no call of it runs such a kernel.
    KernelAccelerationStructure,
    /// A compute kernel's `[[stage_in]]` inputs, or its dispatch's stage-in region
    /// (`executor_contract::case_requirements`). The runtime's make of a pipeline state that
    /// fetches stage inputs is `metal2vulkan_runtime::Gap::KernelInput` of
    /// `KernelInputKind::StageIn`, a known gap its README states, held on lavapipe by the runtime's
    /// device test `tests::kernel_input::a_stage_in_kernel_is_refused_as_stage_in`.
    KernelStageInput,
    /// A rasterization-rate map (`executor_contract::case_requirements`): a render pass's, or one
    /// whose parameter data a buffer or an argument buffer's buffer holds. The runtime makes no
    /// rate map, and its render pass descriptor has no field for one, so no call of it makes one.
    RasterizationRateMap,
    /// A vertex buffer bound at the stride its draw sets, a dynamic layout stride
    /// (`executor_contract::case_requirements`). The runtime's render encoder does not build
    /// `setVertexBuffer:offset:attributeStride:atIndex:`, a known gap its README states, which is
    /// its guard.
    DynamicVertexStride,
    /// An indirect command buffer of compute commands a case holds
    /// (`executor_contract::case_requirements`). The runtime builds no compute command:
    /// `Device::new_indirect_command_buffer` makes indexed draw commands alone
    /// (`metal2vulkan_runtime::Gap::IndirectCommandBuffers` of `IndirectCommandsKind::CommandTypes`
    /// for any other type), and a compute pipeline state made for indirect command buffers is
    /// `IndirectCommandsKind::ComputeCommands`, known gaps its README states, which are its guard.
    IndirectComputeCommands,
    /// An indirect command buffer of render commands whose descriptor allows draws other than
    /// indexed draws alone (`executor_contract::case_requirements`): draws of primitives, patch or
    /// indexed patch draws, or mesh draws of threadgroups or threads. The runtime's indirect
    /// command buffer holds indexed draws alone
    /// (`metal2vulkan_runtime::Gap::IndirectCommandBuffers` of `IndirectCommandsKind::CommandTypes`
    /// for any other type), and the translator refuses a kernel's encoding of any draw but an
    /// indexed triangle draw (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`), known
    /// gaps its README states, which are its guard.
    IndirectDrawTypes,
    /// An indirect command buffer of render commands that set their own pipeline state
    /// (`inheritPipelineState` false) or draw with the executing encoder's buffers
    /// (`inheritBuffers` true) (`executor_contract::case_requirements`). The runtime builds commands
    /// that inherit the pipeline state and bind their own buffers alone
    /// (`metal2vulkan_runtime::Gap::IndirectCommandBuffers` of
    /// `IndirectCommandsKind::CommandPipelines` or `IndirectCommandsKind::InheritedBuffers`),
    /// known gaps its README states, which are its guard.
    IndirectCommandInheritance,
    /// A command the host encodes into an indirect command buffer before its producer runs
    /// (`executor_contract::case_requirements`, the command buffer's `commands`). The runtime has
    /// no `indirectRenderCommandAtIndex:` or `indirectComputeCommandAtIndex:`: the CPU encodes no
    /// command into its Private records, a known gap its README states, which is its guard.
    HostEncodedIndirectCommands,
    /// A render or compute pipeline state an argument buffer holds
    /// (`executor_contract::case_requirements`, the case's `pipeline_states`). The runtime's
    /// pipeline states answer no `gpuResourceID`, so no call of it gives the word a program writes
    /// into an argument buffer's pipeline-state field, and its indirect commands set no pipeline
    /// state of their own (`IndirectCommandsKind::CommandPipelines`).
    ArgumentBufferPipelineState,
}

/// Every executor (the requirement leaves no case to author, or none to observe).
const AUTHORING: &[Executor] = &Executor::ALL;
const ENGINE: &[Executor] = &[Executor::EngineCandidate];

impl ToolingRequirement {
    pub const ALL: [Self; 44] = [
        Self::FunctionConstantLiteral,
        Self::VertexAttributeLiteral,
        Self::TessellationAttributeLiteral,
        Self::TessellationSystemInputLiteral,
        Self::FragmentImageblockMemberLiteral,
        Self::KernelStageInputLiteral,
        Self::TextureBufferMember,
        Self::StorageTextureFormatLiteral,
        Self::ImplicitImageblockLiteral,
        Self::IndirectCommandBuffer,
        Self::VertexSideEffectObservation,
        Self::FragmentVaryingObservationType,
        Self::FragmentOutputObservationType,
        Self::ArgumentBufferIntersectionFunctionTable,
        Self::LinkedIntersectionFunction,
        Self::RenderDeviceBufferArray,
        Self::RenderAccelerationStructure,
        Self::VertexDepthStencilAttachment,
        Self::RasterizationDisabledRenderTarget,
        Self::RasterizationRateMapPlacement,
        Self::KernelMinLodClamp,
        Self::MeshCompanion,
        Self::DepthOutsideUnitRange,
        Self::MeshRenderPipeline,
        Self::ReplayedTemplateResources,
        Self::SeqCstFence,
        Self::RenderFunctionTable,
        Self::IndexBufferPastEnd,
        Self::UlongTextureAtomic,
        Self::SeparateDepthStencil,
        Self::KernelFunctionTable,
        Self::KernelImageblock,
        Self::TextureBuffer,
        Self::FragmentImageblock,
        Self::SparseTexture,
        Self::KernelAccelerationStructure,
        Self::KernelStageInput,
        Self::RasterizationRateMap,
        Self::DynamicVertexStride,
        Self::IndirectComputeCommands,
        Self::IndirectDrawTypes,
        Self::IndirectCommandInheritance,
        Self::HostEncodedIndirectCommands,
        Self::ArgumentBufferPipelineState,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FunctionConstantLiteral => "function_constant_literal",
            Self::VertexAttributeLiteral => "vertex_attribute_literal",
            Self::TessellationAttributeLiteral => "tessellation_attribute_literal",
            Self::TessellationSystemInputLiteral => "tessellation_system_input_literal",
            Self::FragmentImageblockMemberLiteral => "fragment_imageblock_member_literal",
            Self::KernelStageInputLiteral => "kernel_stage_input_literal",
            Self::TextureBufferMember => "texture_buffer_member",
            Self::StorageTextureFormatLiteral => "storage_texture_format_literal",
            Self::ImplicitImageblockLiteral => "implicit_imageblock_literal",
            Self::IndirectCommandBuffer => "indirect_command_buffer",
            Self::VertexSideEffectObservation => "vertex_side_effect_observation",
            Self::FragmentVaryingObservationType => "fragment_varying_observation_type",
            Self::FragmentOutputObservationType => "fragment_output_observation_type",
            Self::ArgumentBufferIntersectionFunctionTable => {
                "argument_buffer_intersection_function_table"
            }
            Self::LinkedIntersectionFunction => "linked_intersection_function",
            Self::RenderDeviceBufferArray => "render_device_buffer_array",
            Self::RenderAccelerationStructure => "render_acceleration_structure",
            Self::VertexDepthStencilAttachment => "vertex_depth_stencil_attachment",
            Self::RasterizationDisabledRenderTarget => "rasterization_disabled_render_target",
            Self::RasterizationRateMapPlacement => "rasterization_rate_map_placement",
            Self::KernelMinLodClamp => "kernel_min_lod_clamp",
            Self::MeshCompanion => "mesh_companion",
            Self::DepthOutsideUnitRange => "depth_outside_unit_range",
            Self::MeshRenderPipeline => "mesh_render_pipeline",
            Self::ReplayedTemplateResources => "replayed_template_resources",
            Self::SeqCstFence => "seq_cst_fence",
            Self::RenderFunctionTable => "render_function_table",
            Self::IndexBufferPastEnd => "index_buffer_past_end",
            Self::UlongTextureAtomic => "ulong_texture_atomic",
            Self::SeparateDepthStencil => "separate_depth_stencil",
            Self::KernelFunctionTable => "kernel_function_table",
            Self::KernelImageblock => "kernel_imageblock",
            Self::TextureBuffer => "texture_buffer",
            Self::FragmentImageblock => "fragment_imageblock",
            Self::SparseTexture => "sparse_texture",
            Self::KernelAccelerationStructure => "kernel_acceleration_structure",
            Self::KernelStageInput => "kernel_stage_input",
            Self::RasterizationRateMap => "rasterization_rate_map",
            Self::DynamicVertexStride => "dynamic_vertex_stride",
            Self::IndirectComputeCommands => "indirect_compute_commands",
            Self::IndirectDrawTypes => "indirect_draw_types",
            Self::IndirectCommandInheritance => "indirect_command_inheritance",
            Self::HostEncodedIndirectCommands => "host_encoded_indirect_commands",
            Self::ArgumentBufferPipelineState => "argument_buffer_pipeline_state",
        }
    }

    /// The executors this requirement stops.
    ///
    /// - Every executor: the case schema cannot express the structure (the `*_literal`
    ///   requirements), no declared observation captures the stage's effect (the observation
    ///   requirements), the only effect is encoded indirect commands no case executes
    ///   ([`ToolingRequirement::IndirectCommandBuffer`]), or the Metal oracle cannot build the
    ///   pipeline, so no candidate has a reference to be compared with
    ///   ([`ToolingRequirement::KernelMinLodClamp`]), or a mesh render pipeline lacks a stage
    ///   nothing supplies ([`ToolingRequirement::MeshCompanion`]).
    /// - The engine's candidate: the families its case mapping and render executor do not
    ///   implement yet (`candidate::engine`), each of which the Metal oracle runs, and
    ///   rasterization-rate map data a case can author only as bytes, which hold one
    ///   implementation's private layout where the engine's module decodes this crate's, and a
    ///   depth outside `[0, 1]` a case gives a depth image, which the runtime puts nowhere
    ///   ([`ToolingRequirement::DepthOutsideUnitRange`]), and a mesh render pipeline or a replayed
    ///   template it does not draw or replay ([`ToolingRequirement::MeshRenderPipeline`],
    ///   [`ToolingRequirement::ReplayedTemplateResources`]), and a texture buffer an argument
    ///   buffer or a texture-handle array holds ([`ToolingRequirement::TextureBufferMember`]),
    ///   and a sequentially consistent fence, a render function's function table, an indexed draw
    ///   past its index buffer, a 64-bit texture atomic and separate depth and stencil attachments,
    ///   which the runtime refuses ([`ToolingRequirement::SeqCstFence`],
    ///   [`ToolingRequirement::RenderFunctionTable`], [`ToolingRequirement::IndexBufferPastEnd`],
    ///   [`ToolingRequirement::UlongTextureAtomic`], [`ToolingRequirement::SeparateDepthStencil`]).
    ///
    /// The requirements that stopped only the old engine's candidate were retired with it.
    pub const fn stops(self) -> &'static [Executor] {
        match self {
            Self::FunctionConstantLiteral
            | Self::VertexAttributeLiteral
            | Self::TessellationAttributeLiteral
            | Self::TessellationSystemInputLiteral
            | Self::FragmentImageblockMemberLiteral
            | Self::KernelStageInputLiteral
            | Self::StorageTextureFormatLiteral
            | Self::ImplicitImageblockLiteral
            | Self::IndirectCommandBuffer
            | Self::VertexSideEffectObservation
            | Self::FragmentVaryingObservationType
            | Self::FragmentOutputObservationType
            | Self::KernelMinLodClamp
            | Self::MeshCompanion => AUTHORING,
            Self::ArgumentBufferIntersectionFunctionTable
            | Self::LinkedIntersectionFunction
            | Self::RenderDeviceBufferArray
            | Self::RenderAccelerationStructure
            | Self::VertexDepthStencilAttachment
            | Self::RasterizationDisabledRenderTarget
            | Self::RasterizationRateMapPlacement
            | Self::DepthOutsideUnitRange
            | Self::MeshRenderPipeline
            | Self::TextureBufferMember
            | Self::ReplayedTemplateResources
            | Self::SeqCstFence
            | Self::RenderFunctionTable
            | Self::IndexBufferPastEnd
            | Self::UlongTextureAtomic
            | Self::SeparateDepthStencil
            | Self::KernelFunctionTable
            | Self::KernelImageblock
            | Self::TextureBuffer
            | Self::FragmentImageblock
            | Self::SparseTexture
            | Self::KernelAccelerationStructure
            | Self::KernelStageInput
            | Self::RasterizationRateMap
            | Self::DynamicVertexStride
            | Self::IndirectComputeCommands
            | Self::IndirectDrawTypes
            | Self::IndirectCommandInheritance
            | Self::HostEncodedIndirectCommands
            | Self::ArgumentBufferPipelineState => ENGINE,
        }
    }

    /// Whether a case carrying this requirement is still a case to keep: the Metal oracle runs it
    /// and its observation is evidence for a candidate that runs the family again, so the checker
    /// accepts it with no candidate (`executor_contract::evidence_gap`), and the engine's candidate
    /// refuses it, typed. Exactly the runtime's limits ([`Limit::Runtime`]): their cases are
    /// qualified on Metal.
    pub const fn qualifies_on_metal(self) -> bool {
        matches!(self.limit(), Limit::Runtime)
    }

    /// Whose limit this requirement records. Every requirement is this crate's tooling's except
    /// [`ToolingRequirement::KernelMinLodClamp`], which Apple's compiler cannot build a pipeline
    /// for, and [`ToolingRequirement::SeqCstFence`], [`ToolingRequirement::TextureBufferMember`],
    /// [`ToolingRequirement::DepthOutsideUnitRange`], [`ToolingRequirement::MeshRenderPipeline`],
    /// [`ToolingRequirement::RenderFunctionTable`], [`ToolingRequirement::IndexBufferPastEnd`],
    /// [`ToolingRequirement::UlongTextureAtomic`], [`ToolingRequirement::SeparateDepthStencil`],
    /// [`ToolingRequirement::KernelFunctionTable`], [`ToolingRequirement::KernelImageblock`],
    /// [`ToolingRequirement::TextureBuffer`], [`ToolingRequirement::FragmentImageblock`],
    /// [`ToolingRequirement::SparseTexture`], [`ToolingRequirement::KernelAccelerationStructure`],
    /// [`ToolingRequirement::KernelStageInput`], [`ToolingRequirement::RasterizationRateMap`],
    /// [`ToolingRequirement::DynamicVertexStride`], [`ToolingRequirement::IndirectComputeCommands`],
    /// [`ToolingRequirement::IndirectDrawTypes`], [`ToolingRequirement::IndirectCommandInheritance`],
    /// [`ToolingRequirement::HostEncodedIndirectCommands`] and
    /// [`ToolingRequirement::ArgumentBufferPipelineState`], which the runtime refuses or does not
    /// build.
    pub const fn limit(self) -> Limit {
        match self {
            Self::KernelMinLodClamp => Limit::Metal,
            Self::SeqCstFence
            | Self::TextureBufferMember
            | Self::DepthOutsideUnitRange
            | Self::MeshRenderPipeline
            | Self::RenderFunctionTable
            | Self::IndexBufferPastEnd
            | Self::UlongTextureAtomic
            | Self::SeparateDepthStencil
            | Self::KernelFunctionTable
            | Self::KernelImageblock
            | Self::TextureBuffer
            | Self::FragmentImageblock
            | Self::SparseTexture
            | Self::KernelAccelerationStructure
            | Self::KernelStageInput
            | Self::RasterizationRateMap
            | Self::DynamicVertexStride
            | Self::IndirectComputeCommands
            | Self::IndirectDrawTypes
            | Self::IndirectCommandInheritance
            | Self::HostEncodedIndirectCommands
            | Self::ArgumentBufferPipelineState => Limit::Runtime,
            _ => Limit::Tooling,
        }
    }

    /// Whether this requirement stops `executor`.
    pub fn stops_executor(self, executor: Executor) -> bool {
        self.stops().contains(&executor)
    }

    /// Whether a row carrying this requirement keeps the authoring-capability census open: it is
    /// a tooling gap ([`Limit::Tooling`]), and every requirement stops an executor the census
    /// closes on. A [`Limit::Metal`] requirement closes its row without a reference: no tooling
    /// can give Metal an answer its compiler cannot build. A [`Limit::Runtime`] requirement closes
    /// its row without a candidate run: the product refuses the structure, typed.
    pub fn blocks_closure(self) -> bool {
        self.limit() == Limit::Tooling
    }

    /// The requirements of `requirements` that stop `executor`.
    pub fn stopping(
        requirements: impl IntoIterator<Item = Self>,
        executor: Executor,
    ) -> std::collections::BTreeSet<Self> {
        requirements
            .into_iter()
            .filter(|requirement| requirement.stops_executor(executor))
            .collect()
    }
}

impl FromStr for ToolingRequirement {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|requirement| requirement.as_str() == value)
            .ok_or_else(|| format!("unknown tooling requirement {value:?}"))
    }
}

impl fmt::Display for ToolingRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An effect both executors run exactly but no declared observation can capture.
///
/// Descriptive, never blocking: it is persisted and reported beside [`ToolingRequirement`], never
/// in it. A row carrying one is authorable, and its cases evidence the shader's other outputs only.
/// No report may count the semantics of an unobserved effect as verified: symbol coverage leaves
/// its calls out, and the authoring census prints each effect with [`UnobservedEffect::reason`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnobservedEffect {
    /// The commands a shader encodes into an indirect command buffer. Both executors supply the
    /// buffer under one descriptor policy and fail a run whose encoding faulted, but Metal has no
    /// host readback of encoded commands (its indirect command objects expose setters only). The
    /// one observation is executing the commands, which a case does and a fact of the source alone
    /// cannot claim.
    IndirectCommandContents,
}

impl UnobservedEffect {
    pub const ALL: [Self; 1] = [Self::IndirectCommandContents];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndirectCommandContents => "indirect_command_contents",
        }
    }

    /// Why the effect is not observed and what evidence is owed, for reports.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::IndirectCommandContents => {
                "ICB contents unobserved: no Metal readback of encoded commands; only a case \
                 that executes them observes them"
            }
        }
    }
}

impl FromStr for UnobservedEffect {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|effect| effect.as_str() == value)
            .ok_or_else(|| format!("unknown unobserved effect {value:?}"))
    }
}

impl fmt::Display for UnobservedEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_json_use_the_same_stable_requirement_name() {
        for requirement in ToolingRequirement::ALL {
            assert_eq!(
                serde_json::to_string(&requirement).unwrap(),
                format!("\"{requirement}\"")
            );
            assert_eq!(requirement.to_string().parse(), Ok(requirement));
        }
        for effect in UnobservedEffect::ALL {
            assert_eq!(
                serde_json::to_string(&effect).unwrap(),
                format!("\"{effect}\"")
            );
            assert_eq!(effect.to_string().parse(), Ok(effect));
            assert!(ToolingRequirement::ALL
                .iter()
                .all(|requirement| requirement.as_str() != effect.as_str()));
        }
    }

    /// The mesh render pipeline's requirements: a stage nothing generates stops both executors
    /// (`mesh_companion`), this crate's gap, which keeps the census open; and the engine's
    /// candidate draws none (`mesh_render_pipeline`), whose cases stay Metal-qualified, the
    /// runtime's limit (it has no mesh call), which closes its rows.
    #[test]
    fn mesh_requirements_stop_the_executors_that_do_not_draw_them_yet() {
        let requirement = ToolingRequirement::MeshCompanion;
        assert_eq!(requirement.stops(), Executor::ALL);
        assert_eq!(requirement.limit(), Limit::Tooling, "{requirement}");
        assert!(requirement.blocks_closure(), "{requirement}");
        assert!(!requirement.qualifies_on_metal(), "{requirement}");
        let requirement = ToolingRequirement::MeshRenderPipeline;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
    }

    /// A texture buffer an argument buffer or a texture-handle array holds: a case expresses it
    /// and the Metal oracle runs it, so its cases stay Metal-qualified; the runtime refuses it, so
    /// the engine's candidate runs none, the runtime's limit, which closes its rows.
    #[test]
    fn a_texture_buffer_member_stops_only_the_candidate() {
        let requirement = ToolingRequirement::TextureBufferMember;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert!("texture_buffer_literal"
            .parse::<ToolingRequirement>()
            .is_err());
    }

    /// A sequentially consistent fence: the Metal oracle runs it, so its cases stay
    /// Metal-qualified; the runtime refuses it, so the engine's candidate runs none, the runtime's
    /// limit, which closes its rows.
    #[test]
    fn a_seq_cst_fence_stops_only_the_candidate() {
        let requirement = ToolingRequirement::SeqCstFence;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert_eq!("seq_cst_fence".parse(), Ok(requirement));
    }

    /// A render function's function table: the Metal oracle draws it, so its cases stay
    /// Metal-qualified; the runtime's render pipeline state reads none, so the engine's candidate
    /// runs none, the runtime's limit, which closes its rows.
    #[test]
    fn a_render_function_table_stops_only_the_candidate() {
        let requirement = ToolingRequirement::RenderFunctionTable;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert_eq!("render_function_table".parse(), Ok(requirement));
    }

    /// An indexed draw past its index buffer: the Metal oracle draws it, so its cases stay
    /// Metal-qualified; the runtime refuses the draw as Metal's validation does, so the engine's
    /// candidate runs none, the runtime's limit, which closes its rows.
    #[test]
    fn an_index_buffer_past_its_end_stops_only_the_candidate() {
        let requirement = ToolingRequirement::IndexBufferPastEnd;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert_eq!("index_buffer_past_end".parse(), Ok(requirement));
    }

    /// A 64-bit texture atomic: the Metal oracle runs it, so its cases stay Metal-qualified; the
    /// runtime enables no 64-bit image atomic, so the engine's candidate runs none, the runtime's
    /// limit, which closes its rows. The retired `ulong_texture` name stays retired.
    #[test]
    fn a_ulong_texture_atomic_stops_only_the_candidate() {
        let requirement = ToolingRequirement::UlongTextureAtomic;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert_eq!("ulong_texture_atomic".parse(), Ok(requirement));
    }

    /// Separate depth and stencil attachments: the Metal oracle draws them, so their cases stay
    /// Metal-qualified; the runtime does not build that pass yet, so the engine's candidate runs
    /// none, the runtime's limit, which closes its rows.
    #[test]
    fn a_separate_depth_stencil_attachment_stops_only_the_candidate() {
        let requirement = ToolingRequirement::SeparateDepthStencil;
        assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
        assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
        assert!(!requirement.blocks_closure(), "{requirement}");
        assert!(requirement.qualifies_on_metal(), "{requirement}");
        assert_eq!("separate_depth_stencil".parse(), Ok(requirement));
    }

    /// The families the runtime builds no call for, or refuses at creation on every device: the
    /// Metal oracle runs each, so its cases stay Metal-qualified; the engine's candidate runs none,
    /// the runtime's limit, which closes their rows.
    #[test]
    fn what_the_runtime_does_not_build_stops_only_the_candidate() {
        for (requirement, name) in [
            (
                ToolingRequirement::KernelFunctionTable,
                "kernel_function_table",
            ),
            (ToolingRequirement::KernelImageblock, "kernel_imageblock"),
            (ToolingRequirement::TextureBuffer, "texture_buffer"),
            (
                ToolingRequirement::FragmentImageblock,
                "fragment_imageblock",
            ),
            (ToolingRequirement::SparseTexture, "sparse_texture"),
            (
                ToolingRequirement::KernelAccelerationStructure,
                "kernel_acceleration_structure",
            ),
            (ToolingRequirement::KernelStageInput, "kernel_stage_input"),
            (
                ToolingRequirement::RasterizationRateMap,
                "rasterization_rate_map",
            ),
            (
                ToolingRequirement::DynamicVertexStride,
                "dynamic_vertex_stride",
            ),
            (
                ToolingRequirement::IndirectComputeCommands,
                "indirect_compute_commands",
            ),
            (ToolingRequirement::IndirectDrawTypes, "indirect_draw_types"),
            (
                ToolingRequirement::IndirectCommandInheritance,
                "indirect_command_inheritance",
            ),
            (
                ToolingRequirement::HostEncodedIndirectCommands,
                "host_encoded_indirect_commands",
            ),
            (
                ToolingRequirement::ArgumentBufferPipelineState,
                "argument_buffer_pipeline_state",
            ),
        ] {
            assert_eq!(requirement.stops(), [Executor::EngineCandidate]);
            assert_eq!(requirement.limit(), Limit::Runtime, "{requirement}");
            assert!(!requirement.blocks_closure(), "{requirement}");
            assert!(requirement.qualifies_on_metal(), "{requirement}");
            assert_eq!(name.parse(), Ok(requirement));
        }
    }

    /// Every requirement stops some executor, and an executor's name is as stable as a
    /// requirement's. A requirement that stops the Metal oracle stops every executor: when Metal
    /// cannot run a case, no candidate has an observation to be compared with. Only a tooling gap
    /// keeps the census open; the runtime's limit stops the candidate alone.
    #[test]
    fn every_requirement_stops_an_executor_and_a_metal_stop_is_an_authoring_stop() {
        for requirement in ToolingRequirement::ALL {
            let stops = requirement.stops();
            assert!(!stops.is_empty(), "{requirement}");
            assert!(
                stops.windows(2).all(|pair| pair[0] < pair[1]),
                "{requirement}"
            );
            if requirement.stops_executor(Executor::MetalOracle) {
                assert_eq!(stops, Executor::ALL, "{requirement}");
            }
            // Metal's own limit stops every executor and closes its row; every tooling gap keeps
            // the census open.
            match requirement.limit() {
                Limit::Metal => {
                    assert_eq!(stops, Executor::ALL, "{requirement}");
                    assert!(!requirement.blocks_closure(), "{requirement}");
                }
                Limit::Runtime => {
                    assert_eq!(stops, [Executor::EngineCandidate], "{requirement}");
                    assert!(requirement.qualifies_on_metal(), "{requirement}");
                    assert!(!requirement.blocks_closure(), "{requirement}");
                }
                Limit::Tooling => assert!(requirement.blocks_closure(), "{requirement}"),
            }
        }
        assert_eq!(
            ToolingRequirement::ALL
                .into_iter()
                .filter(|requirement| requirement.limit() == Limit::Metal)
                .collect::<Vec<_>>(),
            [ToolingRequirement::KernelMinLodClamp]
        );
        assert_eq!(
            ToolingRequirement::ALL
                .into_iter()
                .filter(|requirement| requirement.limit() == Limit::Runtime)
                .collect::<Vec<_>>(),
            [
                ToolingRequirement::TextureBufferMember,
                ToolingRequirement::DepthOutsideUnitRange,
                ToolingRequirement::MeshRenderPipeline,
                ToolingRequirement::SeqCstFence,
                ToolingRequirement::RenderFunctionTable,
                ToolingRequirement::IndexBufferPastEnd,
                ToolingRequirement::UlongTextureAtomic,
                ToolingRequirement::SeparateDepthStencil,
                ToolingRequirement::KernelFunctionTable,
                ToolingRequirement::KernelImageblock,
                ToolingRequirement::TextureBuffer,
                ToolingRequirement::FragmentImageblock,
                ToolingRequirement::SparseTexture,
                ToolingRequirement::KernelAccelerationStructure,
                ToolingRequirement::KernelStageInput,
                ToolingRequirement::RasterizationRateMap,
                ToolingRequirement::DynamicVertexStride,
                ToolingRequirement::IndirectComputeCommands,
                ToolingRequirement::IndirectDrawTypes,
                ToolingRequirement::IndirectCommandInheritance,
                ToolingRequirement::HostEncodedIndirectCommands,
                ToolingRequirement::ArgumentBufferPipelineState,
            ]
        );
        for executor in Executor::ALL {
            assert_eq!(
                serde_json::to_string(&executor).unwrap(),
                format!("\"{executor}\"")
            );
        }
        // The old engine's candidate and the requirements that stopped only it are retired: neither
        // name parses.
        assert!(serde_json::from_str::<Executor>("\"old_engine_candidate\"").is_err());
        for retired in ["static_sampler_pixel_address", "tile_grid", "ulong_texture"] {
            assert!(retired.parse::<ToolingRequirement>().is_err(), "{retired}");
        }
    }
}
