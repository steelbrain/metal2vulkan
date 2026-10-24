//! The runtime side of the case vocabulary: `metal2vulkan_runtime` on whatever Vulkan device the
//! host has.
//!
//! Every call answers the runtime's typed error where it refuses, so a case records what it got
//! rather than panicking, and the comparison can tell our refusal from the venue's honest gap.
//!
//! No wait here can hang: a command buffer is waited for with a bound, and a wait that runs out
//! answers `NeverReturned` as a recorded fact rather than blocking the suite.
use crate::archives::{
    self, Added, Bytes, Created, DynamicSource, LibraryAt, Observation, Program, ProgramLibrary,
    Scratch, Step,
};
use crate::argument_encoders::{ArgumentDescriptorSpec, ArgumentWrite, EncoderFacts};
use crate::case::{
    Answer, ArgumentSlot, Bindings, CommandDraw, CoordinateProbe, Failure, Fill, FragmentFill,
    FragmentFilled, Handle, IndirectReadBack, MappedCoordinates, MeshCommand, MeshFunctions,
    MeshReflectionSpec, MeshStep, PassOutcome, PlaneTexture, RateMapFacts, RateMapForm,
    RateMapRequest, SamplerSpec, ShaderSource, Side, StageInRegion, StageInput, TextureRole,
    TextureSpec, TileFacts, TilePipelineSpec, TileStep, Venue,
};
use crate::command_buffer_facts::CommandBufferFacts;
use crate::compute_descriptors::{
    ComputeDescriptorFacts, DescriptorSpec as ComputeDescriptorSpec,
    SuppliedLists as ComputeSuppliedLists, AWAY as COMPUTE_AWAY,
};
use crate::compute_pass::{
    ComputePass, ComputePassOutcome, ComputeStep, Dispatching, PassResource,
};
use crate::device_queries::{Box6, HeapBufferQuery, RegionQuery};
use crate::function_bindings::{
    ArrayFacts, BindingFacts, BufferFacts, MemberFacts, PayloadFacts, PointerFacts, ReferenceFacts,
    StageBindings, StageReflection, StructFacts, TextureFacts, ThreadgroupFacts, TypeFacts, DEPTH,
};
use crate::function_facts::{
    AttributeFacts, ConstantAt, ConstantFacts, ConstantValue, FunctionFacts,
};
use crate::heap_query::{HeapBufferSpec, HeapOwnFacts, HeapSpec, QueriedResource, ResourceFacts};
use crate::indirect_commands::{
    CommandsMade, ComputeCommandStep, EncodedCommand, IcbDescriptor, RenderCommandStep,
};
use crate::library_facts::{FunctionDescriptorSpec, LibraryFacts};
use crate::mesh_descriptors::{
    DescriptorSpec as MeshDescriptorSpec, MeshDescriptorFacts, SuppliedLists as MeshSuppliedLists,
    AWAY as MESH_AWAY,
};
use crate::pass_descriptors::{
    AttachmentFacts, Indices, Numbers, PassDescriptorFacts, PassSpec, Reads, Row, Shape,
    ASSIGNED_COLOR, DEPTH_AT, MUTATED_AFTER, READ_COLORS as PASS_COLORS, STENCIL_AT, SWEPT_COLOR,
};
use crate::pipeline_facts::{ComputeStateFacts, ComputeStateSpec, RenderStateFacts};
use crate::pipeline_functions::{FunctionsDescriptorFacts, FunctionsDescriptorSpec, Write};
use crate::pipeline_handlers::{
    BothPaths, Delivered, Form as PipelineForm, Handled, StateFacts,
    IMAGEBLOCK as HANDLER_IMAGEBLOCK,
};
use crate::pipeline_option_reflection::UnderOptions;
use crate::pipeline_options::{OptionOutcome, FRAGMENT_STAGE, KERNEL_STAGE, VERTEX_STAGE};
use crate::queue_slots::QueueSlots;
use crate::render::{
    ColorState, LodClamp, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction,
    StepResource, TargetSpec,
};
use crate::render_descriptors::{
    AttributeFacts as DescriptorAttributeFacts, ColorFacts, DescriptorSpec, LayoutFacts,
    LinkedCounts, RenderDescriptorFacts, SuppliedLists, VertexDescriptorFacts, AWAY, READ_BUFFERS,
    READ_COLORS, READ_VERTEX_SLOTS,
};
use crate::sparse_mapping::{
    MappingMode, MappingStep, SparseMapped, SparseMappingProgram, TileRect,
};
use crate::sparse_query::{
    HeapFacts, QueryPage, SparseHeapSpec, SparseTextureFacts, SparseTextureSpec, SparseTileQuery,
    TileSize,
};
use crate::texture_query::{HeapTextureQuery, SizeAndAlign, Storage};
use crate::texture_views::{SharedHandleFacts, ViewCall, ViewFacts, ViewRequest};
use crate::tile_descriptors::{
    DescriptorSpec as TileDescriptorSpec, SuppliedLists as TileSuppliedLists, TileDescriptorFacts,
    AWAY as TILE_AWAY,
};
use crate::transcript::{Negative, Value};
use crate::visible_tables::{
    Asked, Binding, Entry, HandleFacts, LinkedList as SpecLinkedList, Made, Maker, TableFacts,
    TableOutcome, TableSpec, Writes,
};
use metal2vulkan::bind::{
    ArgumentDescriptor, RateMapDescriptor, RateMapLayerDescriptor, SamplerDescriptor, TileMapping,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, BindingAccess, DispatchType, IndexType, IndirectCommandType, Mutability,
    PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, LinkedFunctions, MeshPipelineDescriptor,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, TessellationDescriptor, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_runtime::{
    ArgumentEncoder, BinaryArchive, BinaryArchiveDescriptor, BlitOption, Buffer, ClearColor,
    CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Coordinate2D,
    DepthStencilState, Device, DrawPrimitiveType, DynamicLibrary, Fence, Heap, HeapDescriptor,
    HeapType, HostMemory, IndirectCommand, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectDraw, IoSurfacePlane, IoSurfaceYCbCr, Library, LoadAction,
    MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, Origin, PipelineOption,
    PipelineOptions, PurgeableState, RasterizationRateMap, Region,
    RenderPassColorAttachmentDescriptor, RenderPassDepthAttachmentDescriptor, RenderPassDescriptor,
    RenderPassStencilAttachmentDescriptor, RenderPipelineFunctionsDescriptor, RenderPipelineState,
    ResourceOptions, ResourceRef, ResourceStateCommandEncoder, SamplePosition, SamplerState,
    SamplerStateOptions, ScissorRect, Size, SparseTextureRegionAlignmentMode, StoreAction,
    StoreActionOptions, Texture, TextureDescriptor, Viewport, VisibleFunctionTable,
    VisibleFunctionTableDescriptor,
};
use std::time::Duration;
/// The `MTLResourceUsage` the raw bits a case states name, refused where they name a bit Metal has
/// no name for.
///
/// A case writes the bits because the vocabulary names neither API's type, so the conversion is
/// where an unknown bit has to be answered for; it reads as a typed refusal of ours, at the
/// residency command's own site.
fn resource_usage(bits: u64) -> Answer<ResourceUsage> {
    u32::try_from(bits)
        .ok()
        .and_then(|bits| ResourceUsage::try_from(bits).ok())
        .ok_or_else(|| Failure::named("NoSuchResourceUsage", format!("{bits:#x}")))
}

/// The `MTLBarrierScope` the raw bits a case states name, refused as [`resource_usage`] is.
fn barrier_scope(bits: u64) -> Answer<BarrierScope> {
    u32::try_from(bits)
        .ok()
        .and_then(|bits| BarrierScope::try_from(bits).ok())
        .ok_or_else(|| Failure::named("NoSuchBarrierScope", format!("{bits:#x}")))
}

/// The longest the side waits for a command buffer. A wait that runs out is recorded, not retried.
const WAIT: Duration = Duration::from_secs(60);

/// The runtime's region for a case's six numbers.
fn our_region(held: &Box6) -> Region {
    Region::new_3d(held[0], held[1], held[2], held[3], held[4], held[5])
}

/// A case's six numbers for the runtime's region.
fn their_region(held: &Region) -> Box6 {
    [
        held.origin.x,
        held.origin.y,
        held.origin.z,
        held.size.width,
        held.size.height,
        held.size.depth,
    ]
}

/// The runtime's size for a case's three axes.
fn our_size(axes: [u64; 3]) -> Size {
    Size::new(axes[0], axes[1], axes[2])
}

/// Set every field [`AWAY`] states on `descriptor` and `options`, each away from Metal's own
/// default.
///
/// The label and the binary archives are the **creation's** options rather than the descriptor's
/// fields, which is where the runtime takes `MTLRenderPipelineDescriptor.label` and
/// `binaryArchives` from, so a reset has to put both back.
fn state_descriptor(descriptor: &mut RenderPipelineDescriptor, options: &mut PipelineOptions) {
    options.label = Some(AWAY.label.to_owned());
    // Metal's deprecated `sampleCount` and its `rasterSampleCount` are one field of the runtime's,
    // which is the claim the case's two sites check.
    descriptor.raster_sample_count = AWAY.sample_count;
    descriptor.alpha_to_coverage = AWAY.alpha_to_coverage;
    descriptor.alpha_to_one = AWAY.alpha_to_one;
    descriptor.rasterization_enabled = AWAY.rasterization_enabled;
    descriptor.max_vertex_amplification_count = AWAY.max_vertex_amplification_count;
    descriptor.depth_format = Some(AWAY.depth_format);
    descriptor.stencil_format = Some(AWAY.stencil_format);
    descriptor.input_primitive_topology = AWAY.input_primitive_topology;
    let mut tessellation = TessellationDescriptor::default();
    tessellation.max_tessellation_factor = AWAY.max_tessellation_factor;
    tessellation.partition_mode = AWAY.tessellation_partition_mode;
    tessellation.factor_format = AWAY.tessellation_factor_format;
    tessellation.factor_step_function = AWAY.tessellation_factor_step_function;
    tessellation.output_winding_order = AWAY.tessellation_output_winding_order;
    tessellation.control_point_index_type = AWAY.tessellation_control_point_index_type;
    tessellation.factor_scale_enabled = AWAY.tessellation_factor_scale_enabled;
    descriptor.tessellation = Some(tessellation);
    descriptor.support_indirect_command_buffers = AWAY.support_indirect_command_buffers;
    descriptor.support_adding_vertex_binary_functions = AWAY.support_adding_vertex_binary_functions;
    descriptor.support_adding_fragment_binary_functions =
        AWAY.support_adding_fragment_binary_functions;
    descriptor.max_vertex_call_stack_depth = AWAY.max_vertex_call_stack_depth;
    descriptor.max_fragment_call_stack_depth = AWAY.max_fragment_call_stack_depth;
    for color in AWAY.colors {
        let mut attachment = ColorAttachmentDescriptor::new(color.pixel_format);
        attachment.blending_enabled = color.blending_enabled;
        attachment.source_rgb_blend_factor = color.source_rgb;
        attachment.destination_rgb_blend_factor = color.destination_rgb;
        attachment.rgb_blend_operation = color.rgb_operation;
        attachment.source_alpha_blend_factor = color.source_alpha;
        attachment.destination_alpha_blend_factor = color.destination_alpha;
        attachment.alpha_blend_operation = color.alpha_operation;
        attachment.write_mask = color.write_mask;
        if let Some(slot) = descriptor.color_attachments.get_mut(color.index as usize) {
            *slot = Some(attachment);
        }
    }
    for (index, mutability) in AWAY.vertex_buffers {
        descriptor.vertex_buffers.insert(*index, *mutability);
    }
    for (index, mutability) in AWAY.fragment_buffers {
        descriptor.fragment_buffers.insert(*index, *mutability);
    }
    let mut layout = VertexDescriptor::new();
    for attribute in AWAY.attributes {
        layout.attributes.insert(
            attribute.index,
            VertexAttributeDescriptor::new(
                attribute.format,
                attribute.offset,
                attribute.buffer_index,
            ),
        );
    }
    for stated in AWAY.layouts {
        let mut held = VertexBufferLayoutDescriptor::new(Some(stated.stride));
        held.step_function = stated.step_function;
        held.step_rate = stated.step_rate;
        layout.layouts.insert(stated.index, held);
    }
    descriptor.vertex_descriptor = Some(layout);
}

/// What `descriptor` and `options` answer of themselves, the runtime's side of every getter of
/// `MTLRenderPipelineDescriptor`'s own surface.
///
/// The slots Metal's arrays answer for an index nothing set are the **runtime's claim about
/// Metal's defaults**, written here rather than taken from the case: an attachment the descriptor
/// holds no entry for is Metal's `MTLPixelFormatInvalid` with blending off and every channel
/// written, a buffer slot nothing set is `MTLMutabilityDefault`, and an attribute slot nothing set
/// is format `Invalid` at offset 0 in buffer 0 with a layout of stride 0 stepping per vertex at a
/// rate of 1. Each of those is a claim the Metal leg's recording checks.
fn read_descriptor(
    descriptor: &RenderPipelineDescriptor,
    options: &PipelineOptions,
    supplied: bool,
) -> RenderDescriptorFacts {
    let (vertex_libraries, vertex_linked) = stage_lists(Some(&descriptor.vertex));
    let (fragment_libraries, fragment_linked) = stage_lists(descriptor.fragment.as_ref());
    let colors = (0..READ_COLORS)
        .map(|index| {
            let held = descriptor
                .color_attachments
                .get(index as usize)
                .and_then(Option::as_ref)
                .copied()
                .unwrap_or_else(|| ColorAttachmentDescriptor::new(PixelFormat::Invalid));
            ColorFacts {
                pixel_format: held.pixel_format.as_raw(),
                blending_enabled: held.blending_enabled,
                source_rgb: held.source_rgb_blend_factor.as_raw(),
                destination_rgb: held.destination_rgb_blend_factor.as_raw(),
                rgb_operation: held.rgb_blend_operation.as_raw(),
                source_alpha: held.source_alpha_blend_factor.as_raw(),
                destination_alpha: held.destination_alpha_blend_factor.as_raw(),
                alpha_operation: held.alpha_blend_operation.as_raw(),
                write_mask: held.write_mask.as_raw(),
            }
        })
        .collect();
    let tessellation = descriptor.tessellation.unwrap_or_default();
    RenderDescriptorFacts {
        label: options.label.clone(),
        vertex_function: Some(stage_entry(&descriptor.vertex)),
        fragment_function: descriptor.fragment.as_ref().map(stage_entry),
        vertex_descriptor: descriptor.vertex_descriptor.as_ref().map_or_else(
            || VertexDescriptorFacts::empty(READ_VERTEX_SLOTS),
            read_vertex_descriptor,
        ),
        sample_count: u64::from(descriptor.raster_sample_count),
        raster_sample_count: u64::from(descriptor.raster_sample_count),
        alpha_to_coverage_enabled: descriptor.alpha_to_coverage,
        alpha_to_one_enabled: descriptor.alpha_to_one,
        rasterization_enabled: descriptor.rasterization_enabled,
        max_vertex_amplification_count: u64::from(descriptor.max_vertex_amplification_count),
        colors,
        depth_attachment_pixel_format: descriptor
            .depth_format
            .unwrap_or(PixelFormat::Invalid)
            .as_raw(),
        stencil_attachment_pixel_format: descriptor
            .stencil_format
            .unwrap_or(PixelFormat::Invalid)
            .as_raw(),
        input_primitive_topology: descriptor.input_primitive_topology.as_raw(),
        tessellation_partition_mode: tessellation.partition_mode.as_raw(),
        max_tessellation_factor: u64::from(tessellation.max_tessellation_factor),
        tessellation_factor_scale_enabled: tessellation.factor_scale_enabled,
        tessellation_factor_format: tessellation.factor_format.as_raw(),
        tessellation_control_point_index_type: tessellation.control_point_index_type.as_raw(),
        tessellation_factor_step_function: tessellation.factor_step_function.as_raw(),
        tessellation_output_winding_order: tessellation.output_winding_order.as_raw(),
        vertex_buffers: read_mutability(&descriptor.vertex_buffers),
        fragment_buffers: read_mutability(&descriptor.fragment_buffers),
        support_indirect_command_buffers: descriptor.support_indirect_command_buffers,
        binary_archives: options.binary_archives.len() as u64,
        supplied: supplied.then_some(SuppliedLists {
            vertex_preloaded_libraries: vertex_libraries,
            fragment_preloaded_libraries: fragment_libraries,
            vertex_linked_functions: vertex_linked,
            fragment_linked_functions: fragment_linked,
        }),
        support_adding_vertex_binary_functions: descriptor.support_adding_vertex_binary_functions,
        support_adding_fragment_binary_functions: descriptor
            .support_adding_fragment_binary_functions,
        max_vertex_call_stack_depth: u64::from(descriptor.max_vertex_call_stack_depth),
        max_fragment_call_stack_depth: u64::from(descriptor.max_fragment_call_stack_depth),
    }
}

/// The mutability of a buffer-index map's first [`READ_BUFFERS`] slots, the runtime's side of
/// Metal's pipeline buffer array.
///
/// A slot the map holds no entry for is `MTLMutabilityDefault`, which is the **runtime's claim
/// about Metal's default** and what each family's Metal recording checks.
fn read_mutability(stated: &std::collections::BTreeMap<u32, Mutability>) -> Vec<u32> {
    (0..READ_BUFFERS)
        .map(|index| {
            stated
                .get(&index)
                .copied()
                .unwrap_or(Mutability::Default)
                .as_raw()
        })
        .collect()
}

/// How many objects one stage's supplied lists hold: its preloaded libraries, and the four lengths
/// of its linked functions.
///
/// One function for every stage of every pipeline descriptor the suite reads, because the runtime
/// takes all of them from the stage's own
/// [`FunctionDescriptor`](metal2vulkan::pipeline::FunctionDescriptor) rather than from the
/// descriptor. A stage made from an already-compiled function carries none of either.
fn stage_lists(source: Option<&StageSource>) -> (u64, LinkedCounts) {
    match source {
        Some(StageSource::Function(held)) => (
            held.libraries.preloaded.len() as u64,
            LinkedCounts {
                functions: held.linked.functions.len() as u64,
                private_functions: held.linked.private_functions.len() as u64,
                binary_functions: held.linked.binary_functions.len() as u64,
                groups: held.linked.groups.len() as u64,
            },
        ),
        _ => (0, LinkedCounts::default()),
    }
}

/// The name a stage made from an already-compiled function records. No case of this family states
/// one; a side that did would otherwise record the compiled stage as a function named nothing.
const COMPILED: &str = "compiled";

/// A stage's entry name, as a descriptor family's site records it.
fn stage_entry(source: &StageSource) -> String {
    match source {
        StageSource::Function(held) => held.function.info().name.to_owned(),
        StageSource::Compiled(_) => COMPILED.to_owned(),
    }
}

/// Set every field [`COMPUTE_AWAY`] states on a compute pipeline descriptor and its creation
/// options, each away from Metal's own default.
///
/// The label and the binary archives are the **creation's** options rather than the descriptor's
/// fields, which is where the runtime takes `MTLComputePipelineDescriptor.label` and
/// `binaryArchives` from, so a reset has to put both back.
fn state_compute_descriptor(
    descriptor: &mut metal2vulkan::pipeline::ComputePipelineDescriptor,
    options: &mut PipelineOptions,
) {
    options.label = Some(COMPUTE_AWAY.label.to_owned());
    descriptor.threadgroup_size_is_multiple_of_execution_width =
        COMPUTE_AWAY.threadgroup_size_is_multiple_of_execution_width;
    descriptor.max_total_threads_per_threadgroup =
        Some(COMPUTE_AWAY.max_total_threads_per_threadgroup);
    descriptor.support_indirect_command_buffers = COMPUTE_AWAY.support_indirect_command_buffers;
    descriptor.support_adding_binary_functions = COMPUTE_AWAY.support_adding_binary_functions;
    descriptor.max_call_stack_depth = COMPUTE_AWAY.max_call_stack_depth;
    descriptor.required_threads_per_threadgroup =
        Some(COMPUTE_AWAY.required_threads_per_threadgroup);
    for (index, mutability) in COMPUTE_AWAY.buffers {
        descriptor.buffers.insert(*index, *mutability);
    }
}

/// What a compute pipeline descriptor and its options answer of themselves, the runtime's side of
/// every getter of `MTLComputePipelineDescriptor`'s compared surface.
///
/// Metal's zero thread limit and zero required threadgroup size are the runtime's `None`, which is
/// the claim the Metal leg's recording checks.
fn read_compute_descriptor(
    descriptor: &metal2vulkan::pipeline::ComputePipelineDescriptor,
    options: &PipelineOptions,
    supplied: bool,
) -> ComputeDescriptorFacts {
    let (preloaded, linked) = stage_lists(Some(&descriptor.kernel));
    let required = descriptor
        .required_threads_per_threadgroup
        .unwrap_or([0; 3]);
    ComputeDescriptorFacts {
        label: options.label.clone(),
        compute_function: Some(stage_entry(&descriptor.kernel)),
        threadgroup_size_is_multiple_of_execution_width: descriptor
            .threadgroup_size_is_multiple_of_execution_width,
        max_total_threads_per_threadgroup: u64::from(
            descriptor.max_total_threads_per_threadgroup.unwrap_or(0),
        ),
        buffers: read_mutability(&descriptor.buffers),
        support_indirect_command_buffers: descriptor.support_indirect_command_buffers,
        binary_archives: options.binary_archives.len() as u64,
        supplied: supplied.then_some(ComputeSuppliedLists {
            preloaded_libraries: preloaded,
            // Metal's deprecated `insertLibraries` and its `preloadedLibraries` are one field of
            // the runtime's, which is the claim the case's two sites check.
            insert_libraries: preloaded,
            linked_functions: linked,
        }),
        support_adding_binary_functions: descriptor.support_adding_binary_functions,
        max_call_stack_depth: u64::from(descriptor.max_call_stack_depth),
        required_threads_per_threadgroup: [
            u64::from(required[0]),
            u64::from(required[1]),
            u64::from(required[2]),
        ],
    }
}

/// Set every field [`TILE_AWAY`] states on a tile render pipeline descriptor and its options.
///
/// `resets` says whether the caller will replace the descriptor afterwards, and the buffer
/// mutability is stated only where it will not -- matching the Metal leg, which cannot state it on
/// a descriptor it is about to reset
/// ([`TileDescriptorFacts::buffers`](crate::tile_descriptors::TileDescriptorFacts::buffers)).
fn state_tile_descriptor(
    descriptor: &mut TileRenderPipelineDescriptor,
    options: &mut PipelineOptions,
    resets: bool,
) {
    options.label = Some(TILE_AWAY.label.to_owned());
    descriptor.raster_sample_count = TILE_AWAY.raster_sample_count;
    descriptor.threadgroup_size_matches_tile_size = TILE_AWAY.threadgroup_size_matches_tile_size;
    descriptor.max_total_threads_per_threadgroup =
        Some(TILE_AWAY.max_total_threads_per_threadgroup);
    descriptor.support_adding_binary_functions = TILE_AWAY.support_adding_binary_functions;
    descriptor.max_call_stack_depth = TILE_AWAY.max_call_stack_depth;
    descriptor.required_threads_per_threadgroup = Some(TILE_AWAY.required_threads_per_threadgroup);
    for (index, format) in TILE_AWAY.colors {
        if let Some(slot) = descriptor.color_attachment_formats.get_mut(*index as usize) {
            *slot = Some(*format);
        }
    }
    if !resets {
        for (index, mutability) in TILE_AWAY.buffers {
            descriptor.buffers.insert(*index, *mutability);
        }
    }
}

/// What a tile render pipeline descriptor and its options answer of themselves.
///
/// A colour slot the descriptor holds no format for is Metal's `MTLPixelFormatInvalid`, which is
/// the runtime's claim about Metal's default for a tile attachment.
fn read_tile_descriptor(
    descriptor: &TileRenderPipelineDescriptor,
    options: &PipelineOptions,
    supplied: bool,
) -> TileDescriptorFacts {
    let (preloaded, linked) = stage_lists(Some(&descriptor.tile));
    let required = descriptor
        .required_threads_per_threadgroup
        .unwrap_or([0; 3]);
    TileDescriptorFacts {
        label: options.label.clone(),
        tile_function: stage_entry(&descriptor.tile),
        raster_sample_count: u64::from(descriptor.raster_sample_count),
        colors: (0..READ_COLORS)
            .map(|index| {
                descriptor
                    .color_attachment_formats
                    .get(index as usize)
                    .and_then(|slot| *slot)
                    .unwrap_or(PixelFormat::Invalid)
                    .as_raw()
            })
            .collect(),
        threadgroup_size_matches_tile_size: descriptor.threadgroup_size_matches_tile_size,
        // Read only where the Metal leg can read Metal's, which is every shape but the reset
        // ([`TileDescriptorFacts::buffers`](crate::tile_descriptors::TileDescriptorFacts::buffers)).
        buffers: supplied.then(|| read_mutability(&descriptor.buffers)),
        max_total_threads_per_threadgroup: u64::from(
            descriptor.max_total_threads_per_threadgroup.unwrap_or(0),
        ),
        binary_archives: options.binary_archives.len() as u64,
        supplied: supplied.then_some(TileSuppliedLists {
            preloaded_libraries: preloaded,
            linked_functions: linked,
        }),
        support_adding_binary_functions: descriptor.support_adding_binary_functions,
        max_call_stack_depth: u64::from(descriptor.max_call_stack_depth),
        required_threads_per_threadgroup: [
            u64::from(required[0]),
            u64::from(required[1]),
            u64::from(required[2]),
        ],
    }
}

/// Set every field [`MESH_AWAY`] states on a mesh render pipeline descriptor and its options.
fn state_mesh_descriptor(descriptor: &mut MeshPipelineDescriptor, options: &mut PipelineOptions) {
    options.label = Some(MESH_AWAY.label.to_owned());
    descriptor.raster_sample_count = MESH_AWAY.raster_sample_count;
    descriptor.alpha_to_coverage = MESH_AWAY.alpha_to_coverage;
    descriptor.alpha_to_one = MESH_AWAY.alpha_to_one;
    descriptor.rasterization_enabled = MESH_AWAY.rasterization_enabled;
    descriptor.depth_format = Some(MESH_AWAY.depth_format);
    descriptor.stencil_format = Some(MESH_AWAY.stencil_format);
    descriptor.support_indirect_command_buffers = MESH_AWAY.support_indirect_command_buffers;
    for color in MESH_AWAY.colors {
        let mut attachment = ColorAttachmentDescriptor::new(color.pixel_format);
        attachment.blending_enabled = color.blending_enabled;
        attachment.source_rgb_blend_factor = color.source_rgb;
        attachment.destination_rgb_blend_factor = color.destination_rgb;
        attachment.rgb_blend_operation = color.rgb_operation;
        attachment.source_alpha_blend_factor = color.source_alpha;
        attachment.destination_alpha_blend_factor = color.destination_alpha;
        attachment.alpha_blend_operation = color.alpha_operation;
        attachment.write_mask = color.write_mask;
        if let Some(slot) = descriptor.color_attachments.get_mut(color.index as usize) {
            *slot = Some(attachment);
        }
    }
}

/// What a mesh render pipeline descriptor and its options answer of their compared surface.
///
/// An attachment the descriptor holds no entry for is Metal's `MTLPixelFormatInvalid` with blending
/// off and every channel written, which is the runtime's claim about Metal's default and what the
/// Metal leg's recording checks.
fn read_mesh_descriptor(
    descriptor: &MeshPipelineDescriptor,
    options: &PipelineOptions,
    supplied: bool,
) -> MeshDescriptorFacts {
    let (_, object_linked) = stage_lists(descriptor.object.as_ref());
    let (_, mesh_linked) = stage_lists(Some(&descriptor.mesh));
    let (_, fragment_linked) = stage_lists(descriptor.fragment.as_ref());
    MeshDescriptorFacts {
        label: options.label.clone(),
        object_function: descriptor.object.as_ref().map(stage_entry),
        mesh_function: Some(stage_entry(&descriptor.mesh)),
        fragment_function: descriptor.fragment.as_ref().map(stage_entry),
        raster_sample_count: u64::from(descriptor.raster_sample_count),
        alpha_to_coverage_enabled: descriptor.alpha_to_coverage,
        alpha_to_one_enabled: descriptor.alpha_to_one,
        rasterization_enabled: descriptor.rasterization_enabled,
        colors: (0..READ_COLORS)
            .map(|index| {
                let held = descriptor
                    .color_attachments
                    .get(index as usize)
                    .and_then(Option::as_ref)
                    .copied()
                    .unwrap_or_else(|| ColorAttachmentDescriptor::new(PixelFormat::Invalid));
                ColorFacts {
                    pixel_format: held.pixel_format.as_raw(),
                    blending_enabled: held.blending_enabled,
                    source_rgb: held.source_rgb_blend_factor.as_raw(),
                    destination_rgb: held.destination_rgb_blend_factor.as_raw(),
                    rgb_operation: held.rgb_blend_operation.as_raw(),
                    source_alpha: held.source_alpha_blend_factor.as_raw(),
                    destination_alpha: held.destination_alpha_blend_factor.as_raw(),
                    alpha_operation: held.alpha_blend_operation.as_raw(),
                    write_mask: held.write_mask.as_raw(),
                }
            })
            .collect(),
        depth_attachment_pixel_format: descriptor
            .depth_format
            .unwrap_or(PixelFormat::Invalid)
            .as_raw(),
        stencil_attachment_pixel_format: descriptor
            .stencil_format
            .unwrap_or(PixelFormat::Invalid)
            .as_raw(),
        support_indirect_command_buffers: descriptor.support_indirect_command_buffers,
        binary_archives: options.binary_archives.len() as u64,
        supplied: supplied.then_some(MeshSuppliedLists {
            object_linked_functions: object_linked,
            mesh_linked_functions: mesh_linked,
            fragment_linked_functions: fragment_linked,
        }),
    }
}

/// What a vertex descriptor's first [`READ_VERTEX_SLOTS`] attribute and layout slots answer, the
/// runtime's side of Metal's two arrays.
fn read_vertex_descriptor(layout: &VertexDescriptor) -> VertexDescriptorFacts {
    VertexDescriptorFacts {
        attributes: (0..READ_VERTEX_SLOTS)
            .map(|index| {
                let held = layout.attributes.get(&index).copied().unwrap_or_else(|| {
                    VertexAttributeDescriptor::new(metal2vulkan::metal::VertexFormat::Invalid, 0, 0)
                });
                DescriptorAttributeFacts {
                    format: held.format.as_raw(),
                    offset: u64::from(held.offset),
                    buffer_index: u64::from(held.buffer_index),
                }
            })
            .collect(),
        layouts: (0..READ_VERTEX_SLOTS)
            .map(|index| {
                let held = layout
                    .layouts
                    .get(&index)
                    .copied()
                    .unwrap_or_else(|| VertexBufferLayoutDescriptor::new(Some(0)));
                LayoutFacts {
                    // `MTLBufferLayoutStrideDynamic` is the runtime's `None`, and Metal answers
                    // the sentinel itself (`NSUIntegerMax`); a slot nothing set has a stride of 0
                    // either way, so no shape of this family reaches the sentinel.
                    stride: held.stride.map_or(u64::MAX, u64::from),
                    step_function: held.step_function.as_raw(),
                    step_rate: u64::from(held.step_rate),
                }
            })
            .collect(),
    }
}

/// The bytes the handler-wait probe's command buffer fills, so it has real work to be scheduled and
/// completed for rather than being an empty commit.
const HANDLER_FILL: u64 = 64;

/// The runtime side: the device, the queue the case's command buffers come from, and the objects
/// the case has made, by handle.
pub struct RuntimeSide<'a> {
    device: &'a Device,
    queue: Option<CommandQueue>,
    libraries: Vec<Library>,
    pipelines: Vec<ComputePipelineState>,
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    /// The sampler states the case has made. Its own pool: a sampler state is its own runtime
    /// object, not another name for one of the pools above.
    samplers: Vec<SamplerState>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// runtime object, not another name for one of the pools above.
    rate_maps: Vec<RasterizationRateMap>,
    /// For each argument buffer the case has made, the textures it names: a resource a kernel
    /// reaches through an argument buffer rather than through a slot is resident only where the
    /// encoder was told so, so binding the buffer makes them resident. Not a pool — the buffer
    /// itself lives in `buffers`.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    command_buffers: Vec<CommandBuffer>,
    /// The render pipeline states the case has made, each with whether its creation was asked
    /// for a reflection. One pool, because a mesh state, a tile state and a graphics state are
    /// one runtime object (`RenderPipelineState`) made from three descriptors.
    ///
    /// The runtime answers `texture_access` off the state itself, so nothing is kept beside it
    /// here; the Metal side has to hold the reflection its creation answered, because Metal
    /// answers one only then.
    render_pipelines: Vec<RenderPipelineState>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<IndirectCommandBuffer>,
    /// The depth and stencil states the case has made. Its own pool: a `DepthStencilState` is its
    /// own runtime object, not another name for one of the pools above.
    depth_stencil_states: Vec<DepthStencilState>,
    /// The heaps the case has made. Its own pool: a `Heap` is its own runtime object, not another
    /// name for one of the pools above, and not another name for the depth and stencil states
    /// either.
    heaps: Vec<Heap>,
    /// The argument encoders the case has made, each with the argument buffer it writes into. Its
    /// own pool: an `ArgumentEncoder` is its own runtime object, not another name for one of the
    /// pools above.
    argument_encoders: Vec<EncoderAt>,
    /// Which buffer stands for the surface behind a texture handle that lies over an IOSurface
    /// plane.
    ///
    /// **A lookup, not a pool, and not another name for `heaps` or `buffers`.** The buffer it
    /// names is a case's own and already lives in `buffers`; what this adds is the way back from
    /// the texture handle to it, which a rotated target's plane read and write need and nothing
    /// else does. There is no surface on this side at all — the stand-in buffer is the whole point
    /// of the family, and is why these cases run where there is no IOSurface.
    planes: Vec<(Handle, Buffer)>,
    /// The host pages this side mapped for a no-copy buffer ([`Side::host_buffer`]), one per such
    /// buffer, in the order they were made.
    ///
    /// **Declared after every resource pool on purpose**, because fields drop in declaration
    /// order: a no-copy buffer's pages have to stay mapped for as long as the buffer, and
    /// `buffers` above is dropped before this is. Not a pool a handle names — the buffer itself
    /// lives in `buffers`, and what this adds is the mapping under it.
    host_pages: Vec<crate::host_pages::HostPages>,
    /// This case's open notice scope, which `announced` reads: every degrade notice the device
    /// delivers while this side lives is this case's and no other's.
    ///
    /// **Declared last on purpose**, after every resource pool, for the same drop-order reason:
    /// letting go of a resource may itself announce a notice, and the scope has to still be open
    /// to hear it. Once the scope closes such a notice is unattributed
    /// ([`crate::device::Notices::unattributed`]) rather than the next case's, which is the whole
    /// point of the type.
    notices: crate::device::CaseNotices<'a>,
}

impl<'a> RuntimeSide<'a> {
    /// The side on `device`, reading the notices its case announced from `notices`, the scope
    /// [`crate::device::Notices::open`] handed over.
    ///
    /// The scope is a value and not a closure over a shared list on purpose: a side cannot be
    /// made without one, and the one it holds belongs to one case, so there is no way to build a
    /// side that reads another case's notices.
    pub fn new(device: &'a Device, notices: crate::device::CaseNotices<'a>) -> Self {
        Self {
            device,
            queue: None,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            rate_maps: Vec::new(),
            argument_residents: Vec::new(),
            command_buffers: Vec::new(),
            render_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            depth_stencil_states: Vec::new(),
            heaps: Vec::new(),
            argument_encoders: Vec::new(),
            planes: Vec::new(),
            host_pages: Vec::new(),
            notices,
        }
    }

    /// The device the side runs on, for a case that reads an answer no portable call covers yet.
    pub fn device(&self) -> &Device {
        self.device
    }

    /// The queue the case's command buffers come from, made on first use.
    fn queue(&mut self) -> Answer<&CommandQueue> {
        if self.queue.is_none() {
            self.queue = Some(
                self.device
                    .new_command_queue()
                    .map_err(|error| Failure::of(&error))?,
            );
        }
        Ok(self.queue.as_ref().expect("the queue was just made"))
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&Buffer> {
        self.buffers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// Level 0 of `texture` read back through a blit into a shared buffer of its own, at
    /// `row_pitch` bytes a row: what a private texture's bytes are, since the host reaches none of
    /// them directly.
    fn blit_out(&mut self, texture: &Texture, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let length = u64::from(row_pitch) * u64::from(size[1]);
        let back = self
            .device
            .new_buffer(length.max(1), ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(size[0].into(), size[1].into(), 1),
            &back,
            0,
            row_pitch.into(),
            length,
            BlitOption::None,
        )
        .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Err(Failure::named(
                "ReadBackNeverReturned",
                "the blit of the target into a buffer did not complete",
            ));
        }
        let handle = keep(&mut self.buffers, back);
        keep(&mut self.command_buffers, command_buffer);
        self.read_buffer(handle, 0, length as usize)
    }

    /// The descriptor of a mesh render pipeline of `library`'s `functions`.
    fn mesh_descriptor(
        &self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<MeshPipelineDescriptor> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = MeshPipelineDescriptor::new(stage(functions.mesh)?);
        descriptor.object = functions.object.map(stage).transpose()?;
        descriptor.fragment = Some(stage(functions.fragment)?);
        for attachment in &mut descriptor.color_attachments[..functions.attachments as usize] {
            *attachment = Some(ColorAttachmentDescriptor::new(TARGET_PIXEL_FORMAT));
        }
        Ok(descriptor)
    }

    /// The texture `handle` names.
    fn held_texture(&self, handle: Handle) -> Answer<&Texture> {
        self.textures
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The indirect command buffer `handle` names.
    fn held_commands(&self, handle: Handle) -> Answer<&IndirectCommandBuffer> {
        self.command_buffers_indirect
            .get(handle.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("indirect commands {}", handle.0))
            })
    }

    /// Encode one step onto an indirect compute command: `Ok(None)` where the setter answered,
    /// `Ok(Some(failure))` where it refused, and `Err` where the case named no such object.
    fn encode_compute_command(
        &self,
        encoder: &metal2vulkan_runtime::IndirectComputeCommand,
        step: &ComputeCommandStep,
    ) -> Answer<Option<Failure>> {
        let answered = match step {
            ComputeCommandStep::Pipeline(state) => {
                let state = self.pipelines.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("compute pipeline {}", state.0))
                })?;
                encoder.set_compute_pipeline_state(state)
            }
            ComputeCommandStep::KernelBuffer {
                buffer,
                offset,
                index,
            } => encoder.set_kernel_buffer(Some(self.held(*buffer)?), *offset, *index),
            ComputeCommandStep::KernelBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => encoder.set_kernel_buffer_with_stride(
                Some(self.held(*buffer)?),
                *offset,
                *stride,
                *index,
            ),
            ComputeCommandStep::ThreadgroupMemoryLength { length, index } => {
                encoder.set_threadgroup_memory_length(*length, *index)
            }
            ComputeCommandStep::StageInRegion { origin, size } => {
                encoder.set_stage_in_region(region_of(*origin, *size))
            }
            ComputeCommandStep::ImageblockSize { width, height } => {
                encoder.set_imageblock_size(u64::from(*width), u64::from(*height))
            }
            ComputeCommandStep::DispatchThreadgroups { groups, threads } => {
                encoder.concurrent_dispatch_threadgroups(size(*groups), size(*threads))
            }
            ComputeCommandStep::DispatchThreads { threads, per_group } => {
                encoder.concurrent_dispatch_threads(size(*threads), size(*per_group))
            }
            ComputeCommandStep::Barrier => encoder.set_barrier(),
            ComputeCommandStep::ClearBarrier => encoder.clear_barrier(),
            ComputeCommandStep::Reset => encoder.reset(),
        };
        Ok(answered.err().map(|error| Failure::of(&error)))
    }

    /// Encode one step onto an indirect render command, as
    /// [`encode_compute_command`](Self::encode_compute_command) encodes a dispatch's.
    fn encode_render_command(
        &self,
        encoder: &metal2vulkan_runtime::IndirectRenderCommand,
        step: &RenderCommandStep,
    ) -> Answer<Option<Failure>> {
        let answered = match step {
            RenderCommandStep::Pipeline(state) => {
                let state = self.render_pipelines.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("render pipeline {}", state.0))
                })?;
                encoder.set_render_pipeline_state(state)
            }
            RenderCommandStep::VertexBuffer {
                buffer,
                offset,
                index,
            } => encoder.set_vertex_buffer(Some(self.held(*buffer)?), *offset, *index),
            RenderCommandStep::VertexBufferWithStride {
                buffer,
                offset,
                stride,
                index,
            } => encoder.set_vertex_buffer_with_stride(
                Some(self.held(*buffer)?),
                *offset,
                *stride,
                *index,
            ),
            RenderCommandStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => encoder.set_fragment_buffer(Some(self.held(*buffer)?), *offset, *index),
            RenderCommandStep::ObjectBuffer {
                buffer,
                offset,
                index,
            } => encoder.set_object_buffer(Some(self.held(*buffer)?), *offset, *index),
            RenderCommandStep::MeshBuffer {
                buffer,
                offset,
                index,
            } => encoder.set_mesh_buffer(Some(self.held(*buffer)?), *offset, *index),
            RenderCommandStep::ObjectThreadgroupMemoryLength { length, index } => {
                encoder.set_object_threadgroup_memory_length(*length, *index)
            }
            RenderCommandStep::Draw {
                primitive,
                first,
                count,
                instances,
                base_instance,
            } => encoder.draw_primitives(
                *primitive,
                u64::from(*first),
                u64::from(*count),
                u64::from(*instances),
                u64::from(*base_instance),
            ),
            RenderCommandStep::DrawIndexed {
                primitive,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => encoder.draw_indexed_primitives(
                *primitive,
                u64::from(*count),
                *index_type,
                self.held(*indices)?,
                *offset,
                u64::from(*instances),
                i64::from(*base_vertex),
                u64::from(*base_instance),
            ),
            RenderCommandStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
                factors,
                factor_offset,
                factor_instance_stride,
            } => {
                let indices = match patch_indices {
                    Some(handle) => Some(self.held(*handle)?),
                    None => None,
                };
                encoder.draw_patches(
                    u64::from(*control_points),
                    u64::from(*first),
                    u64::from(*count),
                    indices,
                    *patch_index_offset,
                    u64::from(*instances),
                    u64::from(*base_instance),
                    self.held(*factors)?,
                    *factor_offset,
                    *factor_instance_stride,
                )
            }
            RenderCommandStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_indices,
                control_point_index_offset,
                instances,
                base_instance,
                factors,
                factor_offset,
                factor_instance_stride,
            } => {
                let indices = match patch_indices {
                    Some(handle) => Some(self.held(*handle)?),
                    None => None,
                };
                encoder.draw_indexed_patches(
                    u64::from(*control_points),
                    u64::from(*first),
                    u64::from(*count),
                    indices,
                    *patch_index_offset,
                    Some(self.held(*control_point_indices)?),
                    *control_point_index_offset,
                    u64::from(*instances),
                    u64::from(*base_instance),
                    self.held(*factors)?,
                    *factor_offset,
                    *factor_instance_stride,
                )
            }
            RenderCommandStep::DrawMeshThreadgroups {
                groups,
                object,
                mesh,
            } => encoder.draw_mesh_threadgroups(size(*groups), size(*object), size(*mesh)),
            RenderCommandStep::DrawMeshThreads {
                threads,
                object,
                mesh,
            } => encoder.draw_mesh_threads(size(*threads), size(*object), size(*mesh)),
            RenderCommandStep::Barrier => encoder.set_barrier(),
            RenderCommandStep::ClearBarrier => encoder.clear_barrier(),
            RenderCommandStep::Reset => encoder.reset(),
        };
        Ok(answered.err().map(|error| Failure::of(&error)))
    }

    /// The heap `handle` names.
    fn held_heap(&self, handle: Handle) -> Answer<&Heap> {
        self.heaps
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    /// The sampler state `handle` names.
    fn held_sampler(&self, handle: Handle) -> Answer<&SamplerState> {
        self.samplers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// The argument encoder `handle` names, with the argument buffer it writes into.
    fn held_encoder(&self, handle: Handle) -> Answer<&EncoderAt> {
        self.argument_encoders
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("argument encoder {}", handle.0)))
    }

    /// The rasterization-rate map `handle` names.
    fn held_rate_map(&self, handle: Handle) -> Answer<&RasterizationRateMap> {
        self.rate_maps
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("rate map {}", handle.0)))
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up,
    /// each of `stage_in` set in order, `dispatch` encoded, and `commit`. The two dispatch calls
    /// differ in that one line alone, so they share this body and a case cannot pick up a
    /// different binding order with one of them.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        dispatch: impl FnOnce(
            &mut metal2vulkan_runtime::ComputeCommandEncoder,
        ) -> Result<(), metal2vulkan_runtime::Error>,
    ) -> Answer<Handle> {
        let bound: Vec<Buffer> = bindings
            .buffers
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let sampled: Vec<Texture> = bindings
            .textures
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<SamplerState> = bindings
            .samplers
            .iter()
            .map(|handle| self.held_sampler(*handle).cloned())
            .collect::<Answer<Vec<SamplerState>>>()?;
        // The textures the bound argument buffers name: Metal keeps a resource reached through an
        // argument buffer resident only where the encoder was told so.
        let resident: Vec<Texture> = self
            .argument_residents
            .iter()
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.clone())
            .map(|handle| self.held_texture(handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(Failure::from_error)?;
        encoder
            .set_compute_pipeline_state(&state)
            .map_err(Failure::from_error)?;
        for (at, buffer) in bound.iter().enumerate() {
            encoder
                .set_buffer(Some(buffer), 0, at as u32)
                .map_err(Failure::from_error)?;
        }
        for (at, texture) in sampled.iter().enumerate() {
            encoder
                .set_texture(Some(texture), at as u32)
                .map_err(Failure::from_error)?;
        }
        for (at, sampler) in states.iter().enumerate() {
            encoder
                .set_sampler_state(Some(sampler), at as u32)
                .map_err(Failure::from_error)?;
        }
        for texture in &resident {
            encoder
                .use_resource(ResourceRef::Texture(texture), ResourceUsage::READ)
                .map_err(Failure::from_error)?;
        }
        for region in stage_in {
            match region {
                StageInRegion::Direct { origin, size } => encoder
                    .set_stage_in_region(Region::new_3d(
                        origin[0].into(),
                        origin[1].into(),
                        origin[2].into(),
                        size[0].into(),
                        size[1].into(),
                        size[2].into(),
                    ))
                    .map_err(Failure::from_error)?,
                StageInRegion::Indirect { buffer, offset } => encoder
                    .set_stage_in_region_indirect(self.held(*buffer)?, *offset)
                    .map_err(Failure::from_error)?,
            }
        }
        dispatch(&mut encoder).map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// The usage a texture's role is.
fn usage_of(role: TextureRole) -> TextureUsage {
    match role {
        TextureRole::Read => TextureUsage::SHADER_READ,
        TextureRole::Write => TextureUsage::SHADER_WRITE,
        TextureRole::ReadWrite => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureRole::RenderTarget => TextureUsage::RENDER_TARGET,
    }
}

/// Three axes as the runtime's size.
fn size(axes: [u32; 3]) -> Size {
    Size::new(axes[0].into(), axes[1].into(), axes[2].into())
}

/// Every field of a case's [`RenderPipelineSpec`] **except its two stages**, as the runtime side
/// states them on the pipeline descriptor.
///
/// It exists to be testable without a device. A whole `RenderPipelineDescriptor` cannot be built
/// without one -- its `new` takes a `StageSource`, and neither a `Function` nor a
/// `CompiledFunction` can be made without a device -- so the stage-independent fields are mapped
/// into this value first and copied onto the descriptor after. That is not a mock: it is the value
/// the real path builds and uses.
///
/// The guard over it is `tests::the_runtime_side_forwards_every_field_of_a_pipeline_spec`, and the
/// reason it is wanted is that this side has now twice silently failed to state a field the Metal
/// side states: `write_mask`, and then `depth_format` and `stencil_format`.
pub(crate) struct PipelineFields {
    pub color_attachments: [Option<ColorAttachmentDescriptor>; 8],
    pub depth_format: Option<metal2vulkan::metal::PixelFormat>,
    pub stencil_format: Option<metal2vulkan::metal::PixelFormat>,
    pub raster_sample_count: u32,
    pub rasterization_enabled: bool,
    pub vertex_descriptor: Option<VertexDescriptor>,
    pub support_indirect_command_buffers: bool,
    /// The label the creation's options carry, which is where the runtime takes
    /// `MTLRenderPipelineDescriptor.label` from rather than from the descriptor itself.
    pub label: Option<String>,
    /// The tessellation fields, where the case states them.
    pub tessellation: Option<metal2vulkan::pipeline::TessellationDescriptor>,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    pub input_primitive_topology: metal2vulkan::metal::PrimitiveTopologyClass,
}

/// [`PipelineFields`] of `spec`: `Err` where the spec names a colour attachment past the eight a
/// Metal pipeline holds.
pub(crate) fn pipeline_fields(spec: &RenderPipelineSpec<'_>) -> Answer<PipelineFields> {
    let mut color_attachments: [Option<ColorAttachmentDescriptor>; 8] = Default::default();
    for (at, color) in spec.colors.iter().enumerate() {
        let Some(color) = color else { continue };
        let attachment = color_attachments
            .get_mut(at)
            .ok_or_else(|| Failure::named("NoSuchAttachment", format!("color {at}")))?;
        *attachment = Some(color_descriptor(color));
    }
    Ok(PipelineFields {
        color_attachments,
        depth_format: spec.depth_format,
        stencil_format: spec.stencil_format,
        raster_sample_count: spec.samples,
        rasterization_enabled: spec.rasterization,
        vertex_descriptor: spec.vertex_layout.as_ref().map(vertex_descriptor),
        support_indirect_command_buffers: spec.support_indirect_command_buffers,
        label: spec.label.map(str::to_owned),
        tessellation: spec.tessellation.as_ref().map(tessellation_descriptor),
        alpha_to_coverage: spec.alpha_to_coverage,
        alpha_to_one: spec.alpha_to_one,
        input_primitive_topology: spec.input_primitive_topology,
    })
}

/// The runtime's `TessellationDescriptor` for `state`: every field the case names, since the
/// vocabulary is Metal's own and there is no second spelling to keep in step.
fn tessellation_descriptor(
    state: &crate::render::TessellationState,
) -> metal2vulkan::pipeline::TessellationDescriptor {
    let mut descriptor = metal2vulkan::pipeline::TessellationDescriptor::default();
    descriptor.max_tessellation_factor = state.max_factor;
    descriptor.partition_mode = state.partition;
    descriptor.factor_format = state.factor_format;
    descriptor.factor_step_function = state.factor_step;
    descriptor.output_winding_order = state.winding;
    descriptor.control_point_index_type = state.control_point_index_type;
    descriptor.factor_scale_enabled = state.factor_scale_enabled;
    descriptor
}

/// The runtime's vertex descriptor for `layout`: the same attribute indices, buffer indices and
/// step functions the case states, so both sides fetch the same way.
fn vertex_descriptor(layout: &crate::render::VertexLayout<'_>) -> VertexDescriptor {
    let mut descriptor = VertexDescriptor::new();
    for attribute in layout.attributes {
        descriptor.attributes.insert(
            attribute.index,
            VertexAttributeDescriptor::new(attribute.format, attribute.offset, attribute.buffer),
        );
    }
    for buffer in layout.layouts {
        let mut held = VertexBufferLayoutDescriptor::new(Some(buffer.stride));
        held.step_function = buffer.step;
        held.step_rate = buffer.rate;
        descriptor.layouts.insert(buffer.index, held);
    }
    descriptor
}

/// The runtime's stage-input descriptor for `stage_input`: the same attribute indices, buffer
/// indices, formats, strides, steps and rates, since the case's vocabulary is Metal's own.
fn stage_input_descriptor(stage_input: &StageInput) -> StageInputDescriptor {
    let mut descriptor = StageInputDescriptor::new();
    for attribute in &stage_input.attributes {
        descriptor.attributes.insert(
            attribute.index,
            StageInputAttributeDescriptor::new(
                attribute.format,
                attribute.offset,
                attribute.buffer_index,
            ),
        );
    }
    for layout in &stage_input.layouts {
        let mut written = StageInputLayoutDescriptor::new(layout.stride);
        written.step_function = layout.step_function;
        written.step_rate = layout.step_rate;
        descriptor.layouts.insert(layout.buffer_index, written);
    }
    descriptor.index_type = stage_input.index_type;
    descriptor.index_buffer_index = stage_input.index_buffer_index;
    descriptor
}

/// `device`'s library of AIR bitcode: the translator parses it, and the device takes the parsed
/// library. The runtime's own `new_library_with_data` takes a metallib alone, as Metal's
/// `newLibraryWithData:` does, and the suite's shaders are AIR.
fn new_library_with_air(
    device: &Device,
    bytes: &[u8],
) -> Result<Library, metal2vulkan_runtime::Error> {
    let library = metal2vulkan::library::Library::parse(bytes).map_err(|error| match error {
        metal2vulkan::error::Error::Unsupported(unsupported) => {
            metal2vulkan_runtime::Error::Unsupported(unsupported)
        }
        other => metal2vulkan_runtime::Error::Compile(other),
    })?;
    Ok(device.new_library_with_compile_library(library))
}

/// `slots`'s handle at `at` pushed onto `kept`, answering its index.
fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// One argument encoder of the case's, with where its arguments go.
///
/// The target is kept beside the encoder because `constantDataAtIndex:` answers an **address**,
/// and the fact a case compares is its offset inside the arguments: to take that offset the side
/// needs the argument buffer the encoder was given and where in it the arguments start.
struct EncoderAt {
    encoder: ArgumentEncoder,
    /// The argument buffer and the offset of the arguments in it; `None` until one is set.
    target: Option<(Handle, u64)>,
}

/// One stage-input attribute of ours, in the vocabulary both sides answer in.
fn our_attribute(attribute: &metal2vulkan::reflection::Attribute) -> AttributeFacts {
    AttributeFacts {
        name: attribute.name().to_owned(),
        attribute_index: attribute.attribute_index(),
        attribute_type: attribute.attribute_type(),
        active: attribute.is_active(),
        patch_data: attribute.is_patch_data(),
        patch_control_point_data: attribute.is_patch_control_point_data(),
    }
}

/// The constant values the case states, as the runtime's `MTLFunctionConstantValues`.
///
/// The runtime holds values by index and by name, which is the two single setters; a range is
/// **expanded** into one value an index from its first, which is what
/// `setConstantValues:type:withRange:` means. The bytes are split by the type's own size, so a
/// range of three `bool`s is three one-byte values at three indices.
fn our_constants(values: &[ConstantValue]) -> metal2vulkan::library::FunctionConstantValues {
    let mut built = metal2vulkan::library::FunctionConstantValues::defaults();
    for value in values {
        let one = |bytes: &[u8]| {
            metal2vulkan::library::FunctionConstantValue::new(value.data_type, bytes.to_vec())
        };
        match &value.at {
            ConstantAt::Index(index) => built = built.with_index(*index, one(&value.bytes)),
            ConstantAt::Name(name) => built = built.with_name(name.clone(), one(&value.bytes)),
            ConstantAt::Range { first, count } => {
                let count = *count as usize;
                let each = match count {
                    0 => 0,
                    count => value.bytes.len() / count,
                };
                for step in 0..count {
                    let at = step * each;
                    built = built.with_index(
                        first + step as u32,
                        one(&value.bytes[at..at.saturating_add(each).min(value.bytes.len())]),
                    );
                }
            }
        }
    }
    built
}

/// Every `MTLFunction` getter the vocabulary reads, off a function however it was made.
///
/// Shared by `newFunctionWithName:`, `newFunctionWithName:constantValues:` and
/// `newFunctionWithDescriptor:`, so the three ways of making a function are read through one walk
/// and a case comparing them against each other compares the same getters.
fn our_function_facts(
    function: &metal2vulkan_runtime::Function,
    registry: u64,
) -> Answer<FunctionFacts> {
    let dictionary = function
        .function_constants_dictionary()
        .map_err(|error| Failure::of(&error))?;
    Ok(FunctionFacts {
        name: function.name().to_owned(),
        function_type: function
            .function_type()
            .map_err(|error| Failure::of(&error))?,
        patch_type: function.patch_type().map_err(|error| Failure::of(&error))?,
        patch_control_point_count: function
            .patch_control_point_count()
            .map_err(|error| Failure::of(&error))?,
        vertex_attributes: function
            .vertex_attributes()
            .map_err(|error| Failure::of(&error))?
            .map(|held| held.iter().map(our_attribute).collect()),
        stage_input_attributes: function
            .stage_input_attributes()
            .map_err(|error| Failure::of(&error))?
            .map(|held| held.iter().map(our_attribute).collect()),
        // A `BTreeMap`, so the names are already in the order the Metal side sorts its own into:
        // the comparison is of two sorted lists of a dictionary's entries.
        constants: dictionary
            .iter()
            .map(|(name, constant)| ConstantFacts {
                name: name.clone(),
                data_type: constant.data_type(),
                index: constant.index(),
                required: constant.is_required(),
            })
            .collect(),
        options: function.options(),
        // The registry ID is each host's own number, so only the **comparison** is portable: the
        // two sides agree that the function came from the case's device, never on the number that
        // says so.
        device_is_the_cases: function.device().registry_id() == registry,
    })
}

/// One reflected binding of ours, in the vocabulary both sides answer in.
///
/// The kind-specific fields come from the subprotocol the binding conforms to, which the runtime
/// states as an `Option` per kind — the same decision Metal's side makes from `type`.
fn our_binding(binding: &metal2vulkan::reflection::Binding) -> BindingFacts {
    BindingFacts {
        name: binding.name().to_owned(),
        binding_type: binding.binding_type(),
        access: binding.access(),
        index: binding.index(),
        used: binding.is_used(),
        argument: binding.is_argument(),
        buffer: binding.buffer().map(|buffer| BufferFacts {
            alignment: buffer.buffer_alignment(),
            data_size: buffer.buffer_data_size(),
            data_type: buffer.buffer_data_type(),
            struct_type: buffer.buffer_struct_type().map(our_struct),
            pointer_type: buffer
                .buffer_pointer_type()
                .map(|pointer| our_pointer(pointer, DEPTH)),
        }),
        threadgroup: binding.threadgroup().map(|threadgroup| ThreadgroupFacts {
            alignment: threadgroup.threadgroup_memory_alignment(),
            data_size: threadgroup.threadgroup_memory_data_size(),
        }),
        texture: binding.texture().map(|texture| TextureFacts {
            texture_type: texture.texture_type(),
            data_type: texture.texture_data_type(),
            depth: texture.is_depth_texture(),
            array_length: texture.array_length(),
        }),
        payload: binding.object_payload().map(|payload| PayloadFacts {
            alignment: payload.object_payload_alignment(),
            data_size: payload.object_payload_data_size(),
        }),
    }
}

/// All five stage lists of one reflection of ours, as the Metal side reads Metal's five.
fn our_stage_bindings(
    reflection: &metal2vulkan::reflection::RenderPipelineReflection,
) -> StageBindings {
    let read = |bindings: &[metal2vulkan::reflection::Binding]| -> Vec<BindingFacts> {
        bindings.iter().map(our_binding).collect()
    };
    StageBindings {
        vertex: read(reflection.vertex_bindings()),
        fragment: read(reflection.fragment_bindings()),
        tile: read(reflection.tile_bindings()),
        object: read(reflection.object_bindings()),
        mesh: read(reflection.mesh_bindings()),
    }
}

/// A struct type of ours: its members, and whether `member_by_name` finds each of them.
fn our_struct(held: &metal2vulkan::reflection::StructType) -> StructFacts {
    our_struct_within(held, DEPTH)
}

fn our_struct_within(held: &metal2vulkan::reflection::StructType, depth: u32) -> StructFacts {
    StructFacts {
        members: held
            .members()
            .iter()
            .map(|member| our_member(member, depth))
            .collect(),
        found_by_name: held.members().iter().all(|member| {
            held.member_by_name(member.name())
                .is_some_and(|found| found.name() == member.name())
        }),
    }
}

fn our_member(member: &metal2vulkan::reflection::StructMember, depth: u32) -> MemberFacts {
    MemberFacts {
        name: member.name().to_owned(),
        offset: member.offset(),
        data_type: member.data_type(),
        argument_index: member.argument_index(),
        kind: match depth {
            0 => TypeFacts::Deeper,
            _ => our_detail(
                member.struct_type(),
                member.array_type(),
                member.texture_reference_type(),
                member.pointer_type(),
                depth - 1,
            ),
        },
    }
}

/// Which of the four type getters answered, in the order Metal declares them.
fn our_detail(
    held_struct: Option<&metal2vulkan::reflection::StructType>,
    held_array: Option<&metal2vulkan::reflection::ArrayType>,
    held_reference: Option<&metal2vulkan::reflection::TextureReferenceType>,
    held_pointer: Option<&metal2vulkan::reflection::PointerType>,
    depth: u32,
) -> TypeFacts {
    if let Some(held) = held_struct {
        return TypeFacts::Struct(our_struct_within(held, depth));
    }
    if let Some(held) = held_array {
        return TypeFacts::Array(Box::new(our_array(held, depth)));
    }
    if let Some(held) = held_reference {
        return TypeFacts::TextureReference(ReferenceFacts {
            data_type: held.texture_data_type(),
            texture_type: held.texture_type(),
            access: held.access(),
            depth: held.is_depth_texture(),
        });
    }
    if let Some(held) = held_pointer {
        return TypeFacts::Pointer(Box::new(our_pointer(held, depth)));
    }
    TypeFacts::Plain
}

fn our_array(held: &metal2vulkan::reflection::ArrayType, depth: u32) -> ArrayFacts {
    ArrayFacts {
        element_type: held.element_type(),
        array_length: held.array_length(),
        stride: held.stride(),
        argument_index_stride: held.argument_index_stride(),
        element: match depth {
            0 => TypeFacts::Deeper,
            _ => our_detail(
                held.element_struct_type(),
                held.element_array_type(),
                held.element_texture_reference_type(),
                held.element_pointer_type(),
                depth - 1,
            ),
        },
    }
}

fn our_pointer(held: &metal2vulkan::reflection::PointerType, depth: u32) -> PointerFacts {
    PointerFacts {
        element_type: held.element_type(),
        access: held.access(),
        alignment: held.alignment(),
        data_size: held.data_size(),
        element_is_argument_buffer: held.element_is_argument_buffer(),
        element: match depth {
            0 => TypeFacts::Deeper,
            // A pointer answers no texture reference of its own, which is what the nil says.
            _ => our_detail(
                held.element_struct_type(),
                held.element_array_type(),
                None,
                None,
                depth - 1,
            ),
        },
    }
}

/// The variant a command buffer's status records as.
/// Bytes the command buffer whose timings [`Side::command_buffer_facts`] reads fills: a few, so
/// the GPU ran and nothing else did.
const TIMED_BYTES: usize = 256;

/// The labels the two queues of [`Side::command_buffer_facts`] take: distinct, so a command
/// buffer that answered the other queue reads as the other label.
const QUEUE_LABELS: [&str; 2] = ["conformance-queue-a", "conformance-queue-b"];

/// The byte a read-back buffer of a sparse mapping program is made holding, so a byte no copy
/// wrote is a value the side put there rather than whatever the allocator left.
const READ_GUARD: u8 = 0x7e;

/// Whether a step of a sparse mapping program closed its command buffer.
enum Committed {
    No,
    Yes,
}

/// A sparse mapping program's command buffer and the one encoder of it that is open.
///
/// Two `Option`s rather than one enum, because the runtime's `end_encoding` consumes its encoder,
/// which is what [`Option::take`] hands over. Only one is ever `Some`:
/// [`Mapping::blit`] and [`Mapping::state`] close the other first, as the runtime refuses a second
/// encoder of one command buffer.
struct Mapping {
    queue: CommandQueue,
    command_buffer: Option<CommandBuffer>,
    blit: Option<metal2vulkan_runtime::BlitCommandEncoder>,
    state: Option<ResourceStateCommandEncoder>,
}

impl Mapping {
    /// The command buffer the next command goes into, made on first use.
    fn command_buffer(&mut self) -> Answer<CommandBuffer> {
        if self.command_buffer.is_none() {
            self.command_buffer = Some(self.queue.command_buffer().map_err(Failure::from_error)?);
        }
        Ok(self
            .command_buffer
            .clone()
            .expect("the command buffer was just made"))
    }

    /// End whichever encoder is open, leaving none.
    fn close_encoders(&mut self) -> Answer<()> {
        if let Some(encoder) = self.blit.take() {
            encoder.end_encoding().map_err(Failure::from_error)?;
        }
        if let Some(encoder) = self.state.take() {
            encoder.end_encoding().map_err(Failure::from_error)?;
        }
        Ok(())
    }

    /// The program's blit encoder, ending a resource state encoder first where one is open.
    fn blit(&mut self) -> Answer<&mut metal2vulkan_runtime::BlitCommandEncoder> {
        if self.blit.is_none() {
            self.close_encoders()?;
            let buffer = self.command_buffer()?;
            self.blit = Some(buffer.blit_command_encoder().map_err(Failure::from_error)?);
        }
        Ok(self.blit.as_mut().expect("the encoder was just opened"))
    }

    /// The program's resource state encoder, ending a blit encoder first where one is open.
    fn state(&mut self) -> Answer<&mut ResourceStateCommandEncoder> {
        if self.state.is_none() {
            self.close_encoders()?;
            let buffer = self.command_buffer()?;
            self.state = Some(
                buffer
                    .resource_state_command_encoder()
                    .map_err(Failure::from_error)?,
            );
        }
        Ok(self.state.as_mut().expect("the encoder was just opened"))
    }
}

/// The runtime's `TileMapping` for a case's mapping mode, by its raw value, so the two spellings
/// cannot drift apart.
fn tile_mapping(mode: MappingMode) -> TileMapping {
    match mode {
        MappingMode::Map => TileMapping::Map,
        MappingMode::Unmap => TileMapping::Unmap,
    }
}

/// Three axes as the runtime's size.
fn long_size(axes: [u64; 3]) -> Size {
    Size {
        width: axes[0],
        height: axes[1],
        depth: axes[2],
    }
}

/// Three axes as the runtime's origin.
fn long_origin(axes: [u64; 3]) -> Origin {
    Origin {
        x: axes[0],
        y: axes[1],
        z: axes[2],
    }
}

/// A region of tiles as the `Region` the mapping calls take: a mapping region is stated in
/// **tiles**, not texels, so the numbers pass through unchanged.
fn tile_region(region: &TileRect) -> Region {
    Region {
        origin: long_origin(region.origin),
        size: long_size(region.size),
    }
}

/// `GPUStartTime` and `GPUEndTime`, or the runtime's typed refusal of them: a device whose queue
/// family writes no timestamps has no answer, and that is a fact the facts record as a refusal
/// rather than as two zeros.
fn gpu_times(command_buffer: &CommandBuffer) -> Result<[f64; 2], Failure> {
    let start = command_buffer
        .gpu_start_time()
        .map_err(|error| Failure::of(&error))?;
    let end = command_buffer
        .gpu_end_time()
        .map_err(|error| Failure::of(&error))?;
    Ok([start, end])
}

fn status_value(status: CommandBufferStatus) -> Value {
    Value::unit_variant(format!("{status:?}"))
}

/// `length` bytes from `offset` of a shared buffer the probe made itself, for an event probe whose
/// buffers never enter the case's handle pool.
fn read_shared(buffer: &Buffer, offset: u64, length: usize) -> Result<Vec<u8>, Failure> {
    let contents = buffer
        .contents()
        .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
    // SAFETY: the probe made the buffer at least `offset + length` bytes long, the host reaches
    // them, and the caller waited for the command buffers that wrote them.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
    Ok(bytes.to_vec())
}

/// The names a command buffer's handlers logged, in the order they were called, with each name
/// also sent so the probe can wait for one rather than guess when the log is whole.
///
/// The channel is what makes the log readable: the probe waits for the name it cares about to
/// arrive instead of resting on when a wait of its own returns relative to a handler, which is one
/// of the very orders a case over this family is there to measure.
struct HandlerLog {
    called: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    sender: std::sync::mpsc::Sender<String>,
    arrived: std::sync::mpsc::Receiver<String>,
}

impl HandlerLog {
    /// An empty log.
    fn new() -> Self {
        let (sender, arrived) = std::sync::mpsc::channel();
        Self {
            called: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            sender,
            arrived,
        }
    }

    /// A handler that logs itself as `name`.
    fn handler(&self, name: &'static str) -> metal2vulkan_runtime::Handler {
        let (called, sender) = (std::sync::Arc::clone(&self.called), self.sender.clone());
        Box::new(move |_: &CommandBuffer| {
            if let Ok(mut called) = called.lock() {
                called.push(name.to_owned());
            }
            let _ = sender.send(name.to_owned());
        })
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: Duration) {
        let deadline = std::time::Instant::now() + bound;
        loop {
            if self
                .called
                .lock()
                .is_ok_and(|called| called.iter().any(|called| called == name))
            {
                return;
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() || self.arrived.recv_timeout(left).is_err() {
                return;
            }
        }
    }

    /// The names logged so far, in order.
    fn names(&self) -> Vec<String> {
        self.called
            .lock()
            .map(|called| called.clone())
            .unwrap_or_default()
    }
}

impl Side for RuntimeSide<'_> {
    fn venue(&self) -> Venue {
        let properties = self.device.properties();
        Venue {
            device: properties.name.clone(),
            driver: format!(
                "vendor {:#06x} device {:#06x}",
                properties.vendor_id, properties.device_id
            ),
            api: format!(
                "{}.{}",
                properties.api_version.major, properties.api_version.minor
            ),
            metal_level: properties.metal_level.macos().to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let library = new_library_with_air(self.device, source.bitcode)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn visible_function_table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &TableSpec<'_>,
    ) -> Answer<TableOutcome> {
        let caller = self
            .libraries
            .get(caller.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", caller.0)))?
            .clone();
        let defining = self
            .libraries
            .get(functions.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", functions.0)))?
            .clone();
        let registry = self.device.registry_id();
        let named = |name: &str| {
            defining
                .new_function_with_name(name)
                .map_err(|error| Failure::of(&error))
        };
        // What the stage links, in the order the spec states: a handle's word is the function's
        // position among the callable ones, so the order is the question.
        let mut linked = LinkedFunctions::new();
        for (name, list) in spec.linked.functions {
            let function = named(name)?.linked_function();
            linked = match list {
                SpecLinkedList::Functions => linked.with_function(function),
                SpecLinkedList::PrivateFunctions => linked.with_private_function(function),
            };
        }
        let entry = caller
            .new_function_with_name(spec.entry)
            .map_err(|error| Failure::of(&error))?;
        let descriptor = entry.function_descriptor().with_linked(linked);
        // The two makers the vocabulary has: a compute state, or a drawing state whose vertex
        // function is the case's entry and which has no fragment function, so one stage of it
        // links and the rest do not.
        let maker = match spec.maker {
            Maker::Compute => {
                let state = self
                    .device
                    .new_compute_pipeline_state(
                        &ComputePipelineDescriptor::new(StageSource::Function(descriptor)),
                        &PipelineOptions::default(),
                    )
                    .map_err(|error| Failure::of(&error))?;
                RuntimeMaker::Compute(state)
            }
            Maker::DrawingStage(stage) | Maker::TileStage(stage) => {
                let mut pipeline = RenderPipelineDescriptor::new(StageSource::Function(descriptor));
                pipeline.rasterization_enabled = false;
                let state = self
                    .device
                    .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
                    .map_err(|error| Failure::of(&error))?;
                RuntimeMaker::Render(state, stage.stages())
            }
        };
        // Every asked function's handle and its own `functionType`, in the order asked.
        let mut handles = Vec::new();
        let mut function_types = Vec::new();
        let mut held = Vec::new();
        for name in spec.handles {
            let function = named(name)?;
            function_types.push(
                function
                    .function_type()
                    .map_err(|error| Failure::of(&error))?,
            );
            let answer = maker.function_handle(&function).map(|handle| {
                let facts = handle.as_ref().map(|handle| HandleFacts {
                    name: handle.name().to_owned(),
                    device_is_the_cases: handle.device().registry_id() == registry,
                });
                held.push((*name, handle));
                facts
            });
            handles.push(Asked::of(answer));
        }
        let handle_of = |name: &str| {
            held.iter()
                .find(|(of, _)| *of == name)
                .and_then(|(_, handle)| handle.as_ref())
        };
        // The table, and a second one of the same descriptor where the case asks, so it can say
        // whether two tables of one state answer different resource IDs.
        let wanted = VisibleFunctionTableDescriptor::new(u64::from(spec.function_count));
        let made = maker.new_visible_function_table(&wanted);
        let second = spec
            .second_table
            .then(|| maker.new_visible_function_table(&wanted));
        let table = match &made {
            Ok(table) => Some(table.clone()),
            Err(_) => None,
        };
        let facts = made.map(|table| {
            Some(TableFacts {
                resource_id_is_non_zero: table.gpu_resource_id().0 != 0,
                a_second_tables_id_differs: second.map(|second| match second {
                    Ok(second) => second.gpu_resource_id() != table.gpu_resource_id(),
                    // A second table the state would not make: not "the same ID", which is what
                    // `false` would say, so the case reads the creation's refusal at the site
                    // above and this stays the honest answer for a table there is not.
                    Err(_) => false,
                }),
                device_is_the_cases: table.device().registry_id() == registry,
            })
        });
        // The entries, by whichever setter the case names.
        let mut entries_written = Ok(());
        if let Some(table) = &table {
            let mut taken = Vec::new();
            for entry in spec.entries {
                taken.push(match entry {
                    Entry::Function(name) => handle_of(name),
                    Entry::Nil => None,
                });
            }
            entries_written = match spec.writes {
                Writes::OneAtATime => {
                    let mut answer = Ok(());
                    for (at, handle) in taken.iter().enumerate() {
                        answer = table.set_function(*handle, at as u32);
                        if answer.is_err() {
                            break;
                        }
                    }
                    answer
                }
                Writes::WholeRange if taken.is_empty() => Ok(()),
                Writes::WholeRange => table.set_functions(&taken, 0),
            }
            .map_err(|error| Failure::of(&error));
        }
        let Some(threads) = spec.threads else {
            let mut outcome = TableOutcome::read_only(
                handles,
                wanted.function_count,
                function_types,
                Made::of(facts),
            );
            outcome.entries_written = entries_written;
            return Ok(outcome);
        };
        let RuntimeMaker::Compute(state) = &maker else {
            return Err(Failure::named(
                "NoDispatch",
                "a render stage's table has no dispatch in this family".to_owned(),
            ));
        };
        // One command buffer, one compute encoder: the output at buffer 0, the table at its own
        // index where the case binds it, and one dispatch of `threads` single-thread threadgroups.
        let out = self
            .device
            .new_buffer(u64::from(threads) * 4, ResourceOptions::default())
            .map_err(|error| Failure::of(&error))?;
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(Failure::from_error)?;
        encoder
            .set_compute_pipeline_state(state)
            .map_err(Failure::from_error)?;
        encoder
            .set_buffer(Some(&out), 0, 0)
            .map_err(Failure::from_error)?;
        if let (Binding::Bound, Some(table)) = (spec.binding, &table) {
            encoder
                .set_visible_function_table(Some(table), TABLE_INDEX)
                .map_err(Failure::from_error)?;
        }
        encoder
            .dispatch_threadgroups(size([threads, 1, 1]), size([1, 1, 1]))
            .map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        let status = match command_buffer.wait_until_completed_for(WAIT) {
            Ok(()) => status_value(command_buffer.status()),
            Err(_) => Value::Nothing(Negative::NeverReturned),
        };
        let contents = out
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: the buffer is `threads * 4` bytes long, the host reaches them, and the wait
        // above returned before they are read.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), threads as usize * 4) };
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Ok(TableOutcome {
            handles,
            descriptor_function_count: wanted.function_count,
            function_types,
            table: Made::of(facts),
            entries_written,
            words,
            status,
        })
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            metal2vulkan::pipeline::StageSource::Function(function.function_descriptor()),
        );
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?;
        Ok(u64::from(state.static_threadgroup_memory_length()))
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        let contents = buffer
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes, the host reaches
        // them, and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len());
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .new_buffer(length.max(1) as u64, ResourceOptions::shared())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn host_buffer(&mut self, spec: &crate::host_pages::HostBufferSpec) -> Answer<Handle> {
        if let Some(why) = spec.unmappable() {
            return Err(Failure::named("UnmappableHostBuffer", why));
        }
        let length = usize::try_from(spec.bytes)
            .map_err(|_| Failure::named("UnmappableHostBuffer", "more bytes than a usize"))?;
        let pages = crate::host_pages::HostPages::of(length, spec.pattern)
            .ok_or_else(|| Failure::named("MmapFailed", format!("{length} bytes")))?;
        let raw = if spec.read_only {
            ResourceOptions::shared().raw() | ResourceOptions::PRIVATE_BIT
        } else {
            ResourceOptions::shared().raw()
        };
        let options = ResourceOptions::from_raw(raw).map_err(|error| Failure::of(&error))?;
        // SAFETY: the pages are this side's own mapping of `length` writable bytes, and they stay
        // mapped for as long as the buffer: `host_pages` is declared after `buffers`, so the
        // buffer is released first, and nothing remaps or frees them meanwhile. No deallocator, so
        // the runtime frees nothing of ours.
        let buffer = unsafe {
            let memory = HostMemory::new(pages.pointer());
            self.device
                .new_buffer_with_bytes_no_copy(memory, spec.bytes, options, None)
        }
        .map_err(|error| Failure::of(&error))?;
        self.host_pages.push(pages);
        Ok(keep(&mut self.buffers, buffer))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        let mut descriptor =
            TextureDescriptor::texture_2d(format, size[0].into(), size[1].into(), false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(|error| Failure::of(&error))?;
        texture
            .replace_region(
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
                bytes,
                row_pitch.into(),
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        texture
            .get_bytes(
                &mut bytes,
                row_pitch.into(),
                0,
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(bytes)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            spec.size[0].into(),
            spec.size[1].into(),
            false,
        );
        descriptor.usage = usage_of(spec.role);
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(|error| Failure::of(&error))?;
        if !bytes.is_empty() {
            texture
                .replace_region(
                    Region::new_2d(0, 0, spec.size[0].into(), spec.size[1].into()),
                    0,
                    0,
                    bytes,
                    spec.row_pitch.into(),
                    0,
                )
                .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        let texture = self.held_texture(texture)?;
        texture
            .replace_region(
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
                bytes,
                row_pitch.into(),
                0,
            )
            .map_err(Failure::from_error)
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let format = PixelFormat::try_from(request.format).map_err(|problem| {
            Failure::named(
                "NoSuchPixelFormat",
                format!("{}: {problem:?}", request.format),
            )
        })?;
        let descriptor = texture_descriptor(request, format);
        // One of the three Metal calls that answers a texture of this descriptor, as the request
        // says: the device's own, a buffer's, or a parent texture's view.
        let texture = match request.over {
            None => self.device.new_texture(&descriptor),
            Some(crate::texture::Over::Buffer {
                buffer,
                offset,
                bytes_per_row,
            }) => self
                .held(buffer)?
                .new_texture(&descriptor, offset, bytes_per_row),
            Some(crate::texture::Over::View { texture }) => {
                self.held_texture(texture)?.new_texture_view_with_swizzle(
                    format,
                    request.texture_type,
                    0..u64::from(request.levels.max(1)),
                    0..u64::from(request.slices.max(1)),
                    request.swizzle,
                )
            }
        }
        .map_err(|error| Failure::of(&error))?;
        for upload in request.uploads {
            texture
                .replace_region(
                    Region {
                        origin: Origin::new(
                            upload.origin[0].into(),
                            upload.origin[1].into(),
                            upload.origin[2].into(),
                        ),
                        size: Size::new(
                            upload.size[0].into(),
                            upload.size[1].into(),
                            upload.size[2].max(1).into(),
                        ),
                    },
                    upload.level.into(),
                    upload.slice.into(),
                    upload.bytes,
                    upload.row_pitch.into(),
                    upload.image_pitch.into(),
                )
                .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let mut descriptor = SamplerDescriptor::new();
        descriptor.min_filter = spec.min_filter;
        descriptor.mag_filter = spec.mag_filter;
        descriptor.mip_filter = spec.mip_filter;
        descriptor.s_address_mode = spec.s_address_mode;
        descriptor.t_address_mode = spec.t_address_mode;
        descriptor.r_address_mode = spec.r_address_mode;
        descriptor.border_color = spec.border_color;
        descriptor.normalized_coordinates = spec.normalized_coordinates;
        descriptor.lod_min_clamp = spec.lod_min_clamp;
        descriptor.lod_max_clamp = spec.lod_max_clamp;
        descriptor.lod_average = spec.lod_average;
        descriptor.max_anisotropy = spec.max_anisotropy;
        descriptor.compare_function = spec.compare_function;
        descriptor.reduction_mode = spec.reduction_mode;
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = spec.support_argument_buffers;
        let sampler = self
            .device
            .new_sampler_state_with_options(&descriptor, options)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpu_resource_id().0
                }
                ArgumentSlot::Sampler(handle) => self.held_sampler(handle)?.gpu_resource_id().0,
            };
            words.extend(id.to_le_bytes());
        }
        let buffer = self.buffer(&words)?;
        self.argument_residents.push((buffer, named));
        Ok(buffer)
    }

    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let encoder = function
            .new_argument_encoder_with_buffer_index(buffer_index)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(
            &mut self.argument_encoders,
            EncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let descriptors: Vec<ArgumentDescriptor> = arguments
            .iter()
            .map(|spec| {
                let mut descriptor = ArgumentDescriptor::new(spec.data_type, spec.index);
                descriptor.array_length = spec.array_length;
                descriptor.access = spec.access;
                descriptor.texture_type = spec.texture_type;
                descriptor.constant_block_alignment = spec.constant_block_alignment;
                descriptor
            })
            .collect();
        let encoder = self
            .device
            .new_argument_encoder_with_arguments(&descriptors)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(
            &mut self.argument_encoders,
            EncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_facts(&mut self, encoder: Handle, label: &str) -> Answer<EncoderFacts> {
        let registry = self.device.registry_id();
        let held = self
            .argument_encoders
            .get_mut(encoder.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("argument encoder {}", encoder.0))
            })?;
        held.encoder.set_label(label);
        Ok(EncoderFacts {
            encoded_length: held.encoder.encoded_length(),
            alignment: held.encoder.alignment(),
            label: held.encoder.label().map(str::to_owned),
            // The registry ID is each host's own number, so only the **comparison** is portable:
            // the two sides agree that the encoder came from the case's device, never on the
            // number that says so.
            device_is_the_cases: held.encoder.device().registry_id() == registry,
        })
    }

    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let held = self.held(buffer)?.clone();
        let at = self
            .argument_encoders
            .get_mut(encoder.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("argument encoder {}", encoder.0))
            })?;
        let arguments = offset + element.unwrap_or(0) * at.encoder.encoded_length();
        match element {
            None => at.encoder.set_argument_buffer(Some(&held), offset),
            Some(element) => at.encoder.set_argument_buffer_start_offset_array_element(
                Some(&held),
                offset,
                element,
            ),
        }
        .map_err(|error| Failure::of(&error))?;
        at.target = Some((buffer, arguments));
        Ok(())
    }

    fn nested_argument_encoder(&mut self, encoder: Handle, index: u32) -> Answer<Option<Handle>> {
        let held = self.held_encoder(encoder)?.encoder.clone();
        match held.new_argument_encoder_for_buffer_at_index(index) {
            Ok(nested) => Ok(Some(keep(
                &mut self.argument_encoders,
                EncoderAt {
                    encoder: nested,
                    target: None,
                },
            ))),
            // The runtime's typed `Nil` is the same fact as Metal's nil object, so both sides
            // answer `None` for it; every other error stays an error, which the case records as
            // the refusal it is.
            Err(metal2vulkan_runtime::Error::Nil(_)) => Ok(None),
            Err(error) => Err(Failure::of(&error)),
        }
    }

    fn encode_argument(&mut self, encoder: Handle, write: &ArgumentWrite) -> Answer<()> {
        // The encoder is cloned so the resources it writes can be looked up in the same pools
        // while it is in hand; a clone shares the layout and the argument buffer it was given, so
        // it writes the same bytes in the same place.
        let held = self.held_encoder(encoder)?.encoder.clone();
        let target = self.held_encoder(encoder)?.target.map(|(buffer, _)| buffer);
        let textures = |handles: &[Option<Handle>]| -> Answer<Vec<Option<Texture>>> {
            handles
                .iter()
                .map(|handle| handle.map(|h| self.held_texture(h).cloned()).transpose())
                .collect()
        };
        let samplers = |handles: &[Option<Handle>]| -> Answer<Vec<Option<SamplerState>>> {
            handles
                .iter()
                .map(|handle| handle.map(|h| self.held_sampler(h).cloned()).transpose())
                .collect()
        };
        let buffers = |handles: &[Option<Handle>]| -> Answer<Vec<Option<Buffer>>> {
            handles
                .iter()
                .map(|handle| handle.map(|h| self.held(h).cloned()).transpose())
                .collect()
        };
        let computes = |handles: &[Option<Handle>]| -> Answer<Vec<Option<ComputePipelineState>>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.pipelines.get(h.0 as usize).cloned().ok_or_else(|| {
                                Failure::named("NoSuchHandle", format!("pipeline {}", h.0))
                            })
                        })
                        .transpose()
                })
                .collect()
        };
        let renders = |handles: &[Option<Handle>]| -> Answer<Vec<Option<RenderPipelineState>>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.render_pipelines
                                .get(h.0 as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("render pipeline {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let commands = |handles: &[Option<Handle>]| -> Answer<Vec<Option<IndirectCommandBuffer>>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.command_buffers_indirect
                                .get(h.0 as usize)
                                .cloned()
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("indirect command buffer {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let answer = match write {
            ArgumentWrite::Texture { texture, index } => {
                let held_textures = textures(&[*texture])?;
                held.set_texture(held_textures[0].as_ref(), *index)
            }
            ArgumentWrite::Textures {
                textures: ts,
                first,
            } => {
                let held_textures = textures(ts)?;
                let borrowed: Vec<Option<&Texture>> =
                    held_textures.iter().map(Option::as_ref).collect();
                held.set_textures(&borrowed, *first)
            }
            ArgumentWrite::Sampler { sampler, index } => {
                let held_samplers = samplers(&[*sampler])?;
                held.set_sampler_state(held_samplers[0].as_ref(), *index)
            }
            ArgumentWrite::Samplers {
                samplers: ss,
                first,
            } => {
                let held_samplers = samplers(ss)?;
                let borrowed: Vec<Option<&SamplerState>> =
                    held_samplers.iter().map(Option::as_ref).collect();
                held.set_sampler_states(&borrowed, *first)
            }
            ArgumentWrite::Buffer {
                buffer,
                offset,
                index,
            } => {
                let held_buffers = buffers(&[*buffer])?;
                held.set_buffer(held_buffers[0].as_ref(), *offset, *index)
            }
            ArgumentWrite::Buffers { buffers: bs, first } => {
                let handles: Vec<Option<Handle>> = bs.iter().map(|(handle, _)| *handle).collect();
                let held_buffers = buffers(&handles)?;
                let borrowed: Vec<Option<&Buffer>> =
                    held_buffers.iter().map(Option::as_ref).collect();
                let offsets: Vec<u64> = bs.iter().map(|(_, offset)| *offset).collect();
                held.set_buffers(&borrowed, &offsets, *first)
            }
            ArgumentWrite::ComputePipelineState { state, index } => {
                let held_states = computes(&[*state])?;
                held.set_compute_pipeline_state(held_states[0].as_ref(), *index)
            }
            ArgumentWrite::ComputePipelineStates { states, first } => {
                let held_states = computes(states)?;
                let borrowed: Vec<Option<&ComputePipelineState>> =
                    held_states.iter().map(Option::as_ref).collect();
                held.set_compute_pipeline_states(&borrowed, *first)
            }
            ArgumentWrite::RenderPipelineState { state, index } => {
                let held_states = renders(&[*state])?;
                held.set_render_pipeline_state(held_states[0].as_ref(), *index)
            }
            ArgumentWrite::RenderPipelineStates { states, first } => {
                let held_states = renders(states)?;
                let borrowed: Vec<Option<&RenderPipelineState>> =
                    held_states.iter().map(Option::as_ref).collect();
                held.set_render_pipeline_states(&borrowed, *first)
            }
            ArgumentWrite::IndirectCommandBuffer { commands: c, index } => {
                let held_commands = commands(&[*c])?;
                held.set_indirect_command_buffer(held_commands[0].as_ref(), *index)
            }
            ArgumentWrite::IndirectCommandBuffers { commands: c, first } => {
                let held_commands = commands(c)?;
                let borrowed: Vec<Option<&IndirectCommandBuffer>> =
                    held_commands.iter().map(Option::as_ref).collect();
                held.set_indirect_command_buffers(&borrowed, *first)
            }
            ArgumentWrite::VisibleFunctionTable { table, index } => match table {
                None => held.set_visible_function_table(None, *index),
                // No case states one, and neither side holds a pool of them, so both answer the
                // same "no such call" rather than one of them inventing an object.
                Some(_) => return Err(Failure::unavailable("visible_function_table_handle")),
            },
            ArgumentWrite::VisibleFunctionTables { tables, first } => {
                if tables.iter().any(Option::is_some) {
                    return Err(Failure::unavailable("visible_function_table_handle"));
                }
                let empty: Vec<Option<&metal2vulkan_runtime::VisibleFunctionTable>> =
                    tables.iter().map(|_| None).collect();
                held.set_visible_function_tables(&empty, *first)
            }
            ArgumentWrite::Constant { index, bytes } => {
                let at = held
                    .constant_data(*index)
                    .map_err(|error| Failure::of(&error))?;
                // SAFETY: `constant_data` answers the constant's address inside the argument
                // buffer's mapping, and the case states a constant no longer than the argument it
                // names, which its own layout facts compare.
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len());
                }
                Ok(())
            }
        };
        answer.map_err(|error| Failure::of(&error))?;
        // A texture a function reaches through an argument buffer is resident only where the
        // encoder was told so, so binding the buffer makes it resident.
        if let Some(buffer) = target {
            let named = write.textures();
            if !named.is_empty() {
                self.argument_residents.push((buffer, named));
            }
        }
        Ok(())
    }

    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<BindingFacts>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let mut options = PipelineOptions::default();
        options.options = PipelineOption::BINDING_INFO;
        let (_, reflection) = self
            .device
            .new_compute_pipeline_state_with_function_and_reflection(&function, &options)
            .map_err(|error| Failure::of(&error))?;
        let reflection = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        Ok(reflection.bindings().iter().map(our_binding).collect())
    }

    fn compute_pipeline_reflection_under(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<UnderOptions> {
        let function = self.our_function(library, entry)?;
        let mut built = PipelineOptions::default();
        built.options = PipelineOption::from_raw(options);
        let made = self
            .device
            .new_compute_pipeline_state_with_function_and_reflection(&function, &built);
        let Ok((state, reflection)) = made else {
            return Ok(UnderOptions::refused());
        };
        let found = reflection.map(|reflection| {
            let bindings = reflection.bindings();
            bindings.iter().map(our_binding).collect()
        });
        // The state is kept so it outlives the reflection its creation answered, as the Metal
        // side's is.
        keep(&mut self.pipelines, state);
        Ok(UnderOptions {
            refused: false,
            bindings: found,
        })
    }

    fn archives(&mut self, program: &Program<'_>) -> Answer<Vec<Observation>> {
        // The scratch directory lives exactly as long as the program: its files are the program's
        // own and nothing outside the call reads them, so they are gone when the call returns.
        let scratch = Scratch::new().map_err(|why| Failure::named("NoScratchDirectory", why))?;
        let mut built = Vec::with_capacity(program.libraries.len());
        for entry in program.libraries {
            built.push(
                new_library_with_air(self.device, entry.source.bitcode)
                    .map_err(Failure::from_error)?,
            );
        }
        let registry = self.device.registry_id();
        let mut archives: Vec<Option<BinaryArchive>> = Vec::new();
        let mut from_data: Vec<Option<Library>> = Vec::new();
        let mut dynamics: Vec<Option<DynamicLibrary>> = Vec::new();
        let mut out = Vec::with_capacity(program.steps.len());
        for (at, (_, step)) in program.steps.iter().enumerate() {
            out.push(
                runtime_step(
                    self.device,
                    &scratch,
                    program,
                    &built,
                    registry,
                    &mut archives,
                    &mut from_data,
                    &mut dynamics,
                    step,
                )
                .map_err(|failure| failure.at_step(at))?,
            );
        }
        Ok(out)
    }

    fn compute_pipeline_under_options(
        &mut self,
        library: Handle,
        entry: &str,
        options: u64,
    ) -> Answer<OptionOutcome> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        // The case's raw option word, whole: the runtime keeps every bit and acts on the ones
        // Metal defines, which is the claim the sweep holds to Metal.
        let options = PipelineOptions::new(PipelineOption::from_raw(options));
        let (_state, reflection) = self
            .device
            .new_compute_pipeline_state_with_function_and_reflection(&function, &options)
            .map_err(|error| Failure::of(&error))?;
        Ok(match reflection {
            Some(reflection) => OptionOutcome {
                reflected: true,
                stages: vec![(
                    KERNEL_STAGE,
                    Some(reflection.bindings().iter().map(our_binding).collect()),
                )],
            },
            None => OptionOutcome::without_reflection(&[KERNEL_STAGE]),
        })
    }

    fn render_pipeline_under_options(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
        options: u64,
    ) -> Answer<OptionOutcome> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage(vertex)?);
        descriptor.fragment = Some(stage(fragment)?);
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
        let options = PipelineOptions::new(PipelineOption::from_raw(options));
        let (_state, reflection) = self
            .device
            .new_render_pipeline_state_with_reflection(&descriptor, &options)
            .map_err(Failure::from_error)?;
        Ok(match reflection {
            Some(reflection) => OptionOutcome {
                reflected: true,
                stages: vec![
                    (
                        VERTEX_STAGE,
                        Some(
                            reflection
                                .vertex_bindings()
                                .iter()
                                .map(our_binding)
                                .collect(),
                        ),
                    ),
                    (
                        FRAGMENT_STAGE,
                        Some(
                            reflection
                                .fragment_bindings()
                                .iter()
                                .map(our_binding)
                                .collect(),
                        ),
                    ),
                ],
            },
            None => OptionOutcome::without_reflection(&[VERTEX_STAGE, FRAGMENT_STAGE]),
        })
    }

    fn render_pipeline_functions_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &FunctionsDescriptorSpec<'_>,
    ) -> Answer<FunctionsDescriptorFacts> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let mut made = Vec::new();
        for name in spec.names() {
            let function = library
                .new_function_with_name(name)
                .map_err(|error| Failure::of(&error))?;
            made.push((name, function));
        }
        let held = |name: &str| -> Answer<metal2vulkan_runtime::Function> {
            made.iter()
                .find(|(held, _)| *held == name)
                .map(|(_, function)| function.clone())
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        // The runtime has no descriptor object: a program hands it this value, so the case's
        // writes are writes to its three fields, the last of each winning as a property's would.
        let mut descriptor = RenderPipelineFunctionsDescriptor::new();
        for (stage, writes) in [spec.vertex, spec.fragment, spec.tile]
            .into_iter()
            .enumerate()
        {
            for write in writes {
                let mut functions = Vec::new();
                if let Write::Functions(names) = write {
                    for name in *names {
                        functions.push(held(name)?);
                    }
                }
                match stage {
                    0 => descriptor.vertex_additional_binary_functions = functions,
                    1 => descriptor.fragment_additional_binary_functions = functions,
                    _ => descriptor.tile_additional_binary_functions = functions,
                }
            }
        }
        let names = |functions: &[metal2vulkan_runtime::Function]| -> Vec<String> {
            functions
                .iter()
                .map(|function| function.name().to_owned())
                .collect()
        };
        Ok(FunctionsDescriptorFacts {
            vertex: names(&descriptor.vertex_additional_binary_functions),
            fragment: names(&descriptor.fragment_additional_binary_functions),
            tile: names(&descriptor.tile_additional_binary_functions),
        })
    }

    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let (descriptor, options) = self.our_compute_descriptor(library, spec)?;
        let state = self
            .device
            .new_compute_pipeline_state(&descriptor, &options)
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<ComputeStateFacts> {
        let registry = self.device.registry_id();
        let state = self
            .pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?;
        Ok(our_compute_state_facts(state, registry, imageblock))
    }

    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<RenderStateFacts> {
        let registry = self.device.registry_id();
        let state = self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?;
        Ok(our_render_state_facts(state, registry, imageblock))
    }

    fn render_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &DescriptorSpec<'_>,
    ) -> Answer<RenderDescriptorFacts> {
        let held = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |entry: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                held.new_function_with_name(entry)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        // The runtime has no mutable descriptor object, so Metal's `init` is
        // `RenderPipelineDescriptor::new` and Metal's `reset` is the same call again. The stating
        // still runs for a reset shape — the point of the shape is that what the reset undoes was
        // really set — and then the whole value is replaced, which is the only reset our side has.
        let mut descriptor = RenderPipelineDescriptor::new(stage(spec.vertex)?);
        let mut options = PipelineOptions::default();
        if let Some(fragment) = spec.fragment {
            descriptor.fragment = Some(stage(fragment)?);
        }
        if spec.shape.sets() {
            state_descriptor(&mut descriptor, &mut options);
        }
        if spec.shape.resets() {
            descriptor = RenderPipelineDescriptor::new(stage(spec.vertex)?);
            options = PipelineOptions::default();
            if let Some(fragment) = spec.fragment {
                descriptor.fragment = Some(stage(fragment)?);
            }
        }
        Ok(read_descriptor(&descriptor, &options, !spec.shape.resets()))
    }

    fn compute_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &ComputeDescriptorSpec<'_>,
    ) -> Answer<ComputeDescriptorFacts> {
        let held = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = || -> Answer<StageSource> {
            Ok(StageSource::Function(
                held.new_function_with_name(spec.kernel)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        // The runtime has no mutable descriptor object, so Metal's `init` is
        // `ComputePipelineDescriptor::new` and Metal's `reset` is the same call again. The stating
        // still runs for a reset shape — the point of the shape is that what the reset undoes was
        // really set — and then the whole value is replaced, which is the only reset our side has.
        let mut descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(stage()?);
        let mut options = PipelineOptions::default();
        if spec.shape.sets() {
            state_compute_descriptor(&mut descriptor, &mut options);
        }
        if spec.shape.resets() {
            descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(stage()?);
            options = PipelineOptions::default();
        }
        Ok(read_compute_descriptor(
            &descriptor,
            &options,
            !spec.shape.resets(),
        ))
    }

    fn tile_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &TileDescriptorSpec<'_>,
    ) -> Answer<TileDescriptorFacts> {
        let held = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = || -> Answer<StageSource> {
            Ok(StageSource::Function(
                held.new_function_with_name(spec.tile)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = TileRenderPipelineDescriptor::new(stage()?);
        let mut options = PipelineOptions::default();
        if spec.shape.sets() {
            state_tile_descriptor(&mut descriptor, &mut options, spec.shape.resets());
        }
        if spec.shape.resets() {
            descriptor = TileRenderPipelineDescriptor::new(stage()?);
            options = PipelineOptions::default();
        }
        Ok(read_tile_descriptor(
            &descriptor,
            &options,
            !spec.shape.resets(),
        ))
    }

    fn mesh_descriptor_facts(
        &mut self,
        library: Handle,
        spec: &MeshDescriptorSpec<'_>,
    ) -> Answer<MeshDescriptorFacts> {
        let held = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |entry: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                held.new_function_with_name(entry)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let fresh = |descriptor: &mut MeshPipelineDescriptor| -> Answer<()> {
            descriptor.fragment = Some(stage(spec.fragment)?);
            descriptor.object = match spec.object {
                Some(object) => Some(stage(object)?),
                None => None,
            };
            Ok(())
        };
        let mut descriptor = MeshPipelineDescriptor::new(stage(spec.mesh)?);
        let mut options = PipelineOptions::default();
        fresh(&mut descriptor)?;
        if spec.shape.sets() {
            state_mesh_descriptor(&mut descriptor, &mut options);
        }
        if spec.shape.resets() {
            descriptor = MeshPipelineDescriptor::new(stage(spec.mesh)?);
            options = PipelineOptions::default();
            fresh(&mut descriptor)?;
        }
        Ok(read_mesh_descriptor(
            &descriptor,
            &options,
            !spec.shape.resets(),
        ))
    }

    fn pass_descriptor_facts(&mut self, spec: &PassSpec) -> Answer<PassDescriptorFacts> {
        // The runtime has no descriptor object: Metal's `renderPassDescriptor` is this call, and
        // an attachment Metal's program sets to nil is this type's `Default` again.
        let mut descriptor = RenderPassDescriptor::new();
        if spec.shape.sets() {
            for index in 0..PASS_COLORS {
                let stated = crate::pass_descriptors::stated_row(index);
                state_color_attachment(&mut descriptor.color_attachments[index as usize], &stated)?;
            }
            state_depth_attachment(
                &mut descriptor.depth_attachment,
                &crate::pass_descriptors::stated_row(DEPTH_AT),
            )?;
            state_stencil_attachment(
                &mut descriptor.stencil_attachment,
                &crate::pass_descriptors::stated_row(STENCIL_AT),
            )?;
        }
        if spec.shape == Shape::Assigned {
            // Metal's `setObject:atIndexedSubscript:` and its two attachment setters, as the
            // runtime spells them: an assignment into the descriptor's own value. A value assigned
            // in Rust is a copy by construction, so what this side answers is the assigned values
            // and never the later mutation; whether Metal agrees is the case's measurement.
            let assigned = crate::pass_descriptors::stated_row(ASSIGNED_COLOR);
            let after = crate::pass_descriptors::stated_row(MUTATED_AFTER);
            let mut color = RenderPassColorAttachmentDescriptor::default();
            state_color_attachment(&mut color, &assigned)?;
            descriptor.color_attachments[ASSIGNED_COLOR as usize] = color.clone();
            state_color_attachment(&mut color, &after)?;

            let mut depth = RenderPassDepthAttachmentDescriptor::default();
            state_depth_attachment(&mut depth, &crate::pass_descriptors::stated_row(DEPTH_AT))?;
            descriptor.depth_attachment = depth.clone();
            state_depth_attachment(&mut depth, &after)?;
            depth.clear_depth = crate::pass_descriptors::BASE.clear_depth;

            let mut stencil = RenderPassStencilAttachmentDescriptor::default();
            state_stencil_attachment(
                &mut stencil,
                &crate::pass_descriptors::stated_row(STENCIL_AT),
            )?;
            descriptor.stencil_attachment = stencil.clone();
            state_stencil_attachment(&mut stencil, &after)?;
            stencil.clear_stencil = crate::pass_descriptors::BASE.clear_stencil;
        }
        if let Some(row) = spec.row {
            let shifted = |shift: u64| match spec.reads.rotates() {
                true => row.rotated(shift),
                false => row,
            };
            state_color_attachment(
                &mut descriptor.color_attachments[SWEPT_COLOR as usize],
                &shifted(0),
            )?;
            state_depth_attachment(&mut descriptor.depth_attachment, &shifted(1))?;
            state_stencil_attachment(&mut descriptor.stencil_attachment, &shifted(2))?;
        }
        if let Some(numbers) = spec.numbers {
            descriptor.render_target_array_length = numbers.0[0];
            descriptor.render_target_width = numbers.0[1];
            descriptor.render_target_height = numbers.0[2];
            descriptor.default_raster_sample_count = numbers.0[3];
            descriptor.tile_width = numbers.0[4];
            descriptor.tile_height = numbers.0[5];
            descriptor.imageblock_sample_length = numbers.0[6];
            descriptor.threadgroup_memory_length = numbers.0[7];
        }
        if let Some(stated) = spec.positions {
            descriptor.sample_positions = stated
                .iter()
                .map(|(x, y)| SamplePosition::new(*x, *y))
                .collect();
        }
        if spec.shape == Shape::Cleared {
            for index in 0..PASS_COLORS {
                descriptor.color_attachments[index as usize] = Default::default();
            }
            descriptor.depth_attachment = Default::default();
            descriptor.stencil_attachment = Default::default();
        }
        Ok(read_pass_descriptor(&descriptor, spec.reads))
    }

    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[ConstantValue],
    ) -> Answer<FunctionFacts> {
        let registry = self.device.registry_id();
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = match constants.is_empty() {
            true => library.new_function_with_name(entry),
            false => {
                library.new_function_with_name_constant_values(entry, &our_constants(constants))
            }
        }
        .map_err(|error| Failure::of(&error))?;
        our_function_facts(&function, registry)
    }

    fn library_facts(&mut self, library: Handle) -> Answer<LibraryFacts> {
        let registry = self.device.registry_id();
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        Ok(LibraryFacts {
            function_names: library.function_names(),
            library_type: library.library_type(),
            install_name: library.install_name().map(str::to_owned),
            // The registry ID is each host's own number, so only the **comparison** is portable,
            // as it is for a function's own device.
            device_is_the_cases: library.device().registry_id() == registry,
        })
    }

    fn function_facts_from_descriptor(
        &mut self,
        library: Handle,
        descriptor: &FunctionDescriptorSpec,
    ) -> Answer<FunctionFacts> {
        let registry = self.device.registry_id();
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let mut made = metal2vulkan_runtime::FunctionDescriptor::new(&descriptor.name);
        if !descriptor.constants.is_empty() {
            made.constant_values = Some(our_constants(&descriptor.constants));
        }
        made.options = descriptor.options;
        let function = library
            .new_function_with_descriptor(&made)
            .map_err(|error| Failure::of(&error))?;
        our_function_facts(&function, registry)
    }

    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let held = self.held_encoder(encoder)?;
        let (buffer, arguments) = held.target.ok_or_else(|| {
            Failure::named("NoArgumentBuffer", "no argument buffer is set".to_owned())
        })?;
        let at = held
            .encoder
            .constant_data(index)
            .map_err(|error| Failure::of(&error))?;
        let contents = self
            .held(buffer)?
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        let base = contents.as_ptr() as usize + arguments as usize;
        (at.as_ptr() as usize)
            .checked_sub(base)
            .map(|offset| offset as u64)
            .ok_or_else(|| {
                Failure::named(
                    "ConstantBeforeTheArguments",
                    "constantDataAtIndex: answered an address below the arguments".to_owned(),
                )
            })
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, &[], |encoder| {
            encoder.dispatch_threadgroups(size(threadgroups), size(threads))
        })
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let mut descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            metal2vulkan::pipeline::StageSource::Function(function.function_descriptor()),
        );
        descriptor.stage_input = Some(stage_input_descriptor(stage_input));
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, stage_in, |encoder| {
            encoder.dispatch_threads(size(grid), size(threads))
        })
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let command_buffer = self
            .command_buffers
            .get(command_buffer.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("command buffer {}", command_buffer.0),
                )
            })?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        Ok(status_value(command_buffer.status()))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length());
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let contents = buffer
            .contents()
            .ok_or_else(|| Failure::named("NoContents", "a buffer the host does not reach"))?;
        // SAFETY: `[offset, offset + length)` is inside the buffer's bytes, the host reaches
        // them, and the caller waited for the command buffer that wrote them.
        let bytes =
            unsafe { std::slice::from_raw_parts(contents.as_ptr().add(offset as usize), length) };
        Ok(bytes.to_vec())
    }

    fn announced(&mut self) -> Vec<String> {
        self.notices.take()
    }

    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let mut descriptor = metal2vulkan_runtime::IndirectCommandBufferDescriptor::new();
        descriptor.command_types =
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let made = self
            .device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command = made
            .indirect_render_command_at(0)
            .map_err(Failure::from_error)?;
        let primitive = DrawPrimitiveType::from_raw(primitive_type).map_err(|problem| {
            Failure::named(
                "NoSuchPrimitiveType",
                format!("{primitive_type}: {problem:?}"),
            )
        })?;
        if indexed {
            let indices = self
                .device
                .new_buffer(4096, ResourceOptions::shared())
                .map_err(Failure::from_error)?;
            command
                .draw_indexed_primitives(primitive, 6, IndexType::UInt16, &indices, 0, 1, 0, 0)
                .map_err(Failure::from_error)?;
        } else {
            command
                .draw_primitives(primitive, 0, 6, 1, 0)
                .map_err(Failure::from_error)?;
        }
        let answered = command
            .draw()
            .map_err(Failure::from_error)?
            .map(drawn_primitive_type);
        let recorded = match made
            .commands(0..1)
            .map_err(Failure::from_error)?
            .into_iter()
            .next()
        {
            Some(IndirectCommand::Render(render)) => render.draw.map(drawn_primitive_type),
            _ => None,
        };
        Ok(IndirectReadBack {
            command_type: command
                .command_type()
                .map_err(Failure::from_error)?
                .as_raw()
                .into(),
            answered_primitive_type: answered,
            recorded_primitive_type: recorded,
        })
    }

    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let queue = self.queue()?.clone();
        crate::runtime_work::run(self.device, &queue, work)
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let state = {
            let library = self.libraries.get(fill.library.0 as usize).ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("library {}", fill.library.0))
            })?;
            let function = |name: &str| -> Answer<StageSource> {
                Ok(StageSource::Function(
                    library
                        .new_function_with_name(name)
                        .map_err(|error| Failure::of(&error))?
                        .function_descriptor(),
                ))
            };
            let mut descriptor = RenderPipelineDescriptor::new(function(fill.vertex)?);
            descriptor.fragment = Some(function(fill.fragment)?);
            descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(fill.format));
            self.device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map_err(Failure::from_error)?
        };
        let mut target = TextureDescriptor::texture_2d(
            fill.format,
            u64::from(fill.size[0]),
            u64::from(fill.size[1]),
            false,
        );
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        let target = self
            .device
            .new_texture(&target)
            .map_err(Failure::from_error)?;
        let bound: Vec<Buffer> = fill
            .buffers
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::DontCare;
        attachment.store_action = StoreAction::Store;
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        encoder
            .set_render_pipeline_state(&state)
            .map_err(Failure::from_error)?;
        for (index, buffer) in bound.iter().enumerate() {
            encoder
                .set_fragment_buffer(Some(buffer), 0, index as u32)
                .map_err(Failure::from_error)?;
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, fill.vertices, 1, 0)
            .map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        let status = if command_buffer.wait_until_completed_for(WAIT).is_err() {
            Value::Nothing(Negative::NeverReturned)
        } else {
            status_value(command_buffer.status())
        };
        keep(&mut self.command_buffers, command_buffer);
        let bytes = self.blit_out(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        let state = self
            .device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn mesh_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
        options: u64,
    ) -> Answer<StageReflection> {
        let descriptor = self.mesh_reflection_descriptor(library, spec)?;
        let mut built = PipelineOptions::default();
        built.options = PipelineOption::from_raw(options);
        let made = self
            .device
            .new_mesh_render_pipeline_state_with_reflection(&descriptor, &built);
        let Ok((state, reflection)) = made else {
            return Ok(StageReflection::refused());
        };
        let found = match reflection {
            Some(reflection) => StageReflection::reflected(our_stage_bindings(&reflection)),
            None => StageReflection::unreflected(),
        };
        // The state is kept so it outlives the reflection its creation answered.
        keep(&mut self.render_pipelines, state);
        Ok(found)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let mut descriptor = TextureDescriptor::texture_2d(
            TARGET_PIXEL_FORMAT,
            u64::from(width),
            u64::from(height),
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<RenderPipelineState> = self.render_pipelines.clone();
        let buffers: Vec<Buffer> = self.buffers.clone();
        let held = |handle: Handle| -> Answer<&Buffer> {
            buffers
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
        };
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(steps.len());
        for step in steps {
            let answered = match step {
                MeshStep::Pipeline(state) => states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                    })
                    .and_then(|state| {
                        encoder
                            .set_render_pipeline_state(state)
                            .map_err(Failure::from_error)
                    }),
                MeshStep::ObjectBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_object_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::ObjectBytes { bytes, index } => encoder
                    .set_object_bytes(bytes, *index)
                    .map_err(Failure::from_error),
                MeshStep::MeshBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_mesh_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::FragmentBuffer {
                    buffer,
                    offset,
                    index,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .set_fragment_buffer(Some(buffer), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                MeshStep::DrawThreadgroups {
                    groups,
                    object,
                    mesh,
                } => encoder
                    .draw_mesh_threadgroups(size(*groups), size(*object), size(*mesh))
                    .map_err(Failure::from_error),
                MeshStep::DrawThreads {
                    threads,
                    object,
                    mesh,
                } => encoder
                    .draw_mesh_threads(size(*threads), size(*object), size(*mesh))
                    .map_err(Failure::from_error),
                MeshStep::DrawIndirect {
                    buffer,
                    offset,
                    object,
                    mesh,
                } => held(*buffer).and_then(|buffer| {
                    encoder
                        .draw_mesh_threadgroups_indirect(
                            buffer,
                            *offset,
                            size(*object),
                            size(*mesh),
                        )
                        .map_err(Failure::from_error)
                }),
                MeshStep::BarrierScope {
                    scope,
                    after,
                    before,
                } => scope_bits(*scope).and_then(|scope| {
                    let (after, before) = (stages(*after)?, stages(*before)?);
                    encoder.memory_barrier_with_scope(scope, after, before);
                    Ok(())
                }),
                MeshStep::BarrierResources {
                    buffers: over,
                    after,
                    before,
                } => over
                    .iter()
                    .map(|handle| held(*handle))
                    .collect::<Answer<Vec<&Buffer>>>()
                    .and_then(|over| {
                        let resources: Vec<ResourceRef<'_>> =
                            over.into_iter().map(ResourceRef::Buffer).collect();
                        encoder
                            .memory_barrier_with_resources(
                                &resources,
                                stages(*after)?,
                                stages(*before)?,
                            )
                            .map_err(Failure::from_error)
                    }),
            };
            refused.push(answered.err());
        }
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let mut descriptor = self.mesh_descriptor(library, functions)?;
        descriptor.support_indirect_command_buffers = true;
        let state = self
            .device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_MESH_THREADGROUPS
            .union(IndirectCommandType::DRAW_MESH_THREADS);
        descriptor.inherit_pipeline_state = false;
        descriptor.inherit_buffers = false;
        descriptor.max_object_buffer_bind_count = 1;
        descriptor.max_mesh_buffer_bind_count = 1;
        let made = self
            .device
            .new_indirect_command_buffer(
                &descriptor,
                commands.len() as u64,
                ResourceOptions::shared(),
            )
            .map_err(Failure::from_error)?;
        for (at, command) in commands.iter().enumerate() {
            let encoded = made
                .indirect_render_command_at(at as u64)
                .map_err(Failure::from_error)?;
            let state = self
                .render_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?
                .clone();
            encoded
                .set_render_pipeline_state(&state)
                .map_err(Failure::from_error)?;
            for (buffer, index) in &command.object_buffers {
                let buffer = self.held(*buffer)?.clone();
                encoded
                    .set_object_buffer(Some(&buffer), 0, *index)
                    .map_err(Failure::from_error)?;
            }
            for (buffer, index) in &command.mesh_buffers {
                let buffer = self.held(*buffer)?.clone();
                encoded
                    .set_mesh_buffer(Some(&buffer), 0, *index)
                    .map_err(Failure::from_error)?;
            }
            match command.draw {
                CommandDraw::Threadgroups {
                    groups,
                    object,
                    mesh,
                } => encoded.draw_mesh_threadgroups(size(groups), size(object), size(mesh)),
                CommandDraw::Threads {
                    threads,
                    object,
                    mesh,
                } => encoded.draw_mesh_threads(size(threads), size(object), size(mesh)),
            }
            .map_err(Failure::from_error)?;
        }
        Ok(keep(&mut self.command_buffers_indirect, made))
    }

    fn compute_pipeline_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(entry)
            .map_err(|error| Failure::of(&error))?;
        let mut descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            metal2vulkan::pipeline::StageSource::Function(function.function_descriptor()),
        );
        descriptor.support_indirect_command_buffers = true;
        descriptor.stage_input = stage_input.map(stage_input_descriptor);
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|error| Failure::of(&error))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn indirect_command_buffer(
        &mut self,
        spec: &IcbDescriptor,
        commands: &[EncodedCommand],
    ) -> Answer<CommandsMade> {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = spec.command_types;
        descriptor.inherit_pipeline_state = spec.inherit_pipeline_state;
        descriptor.inherit_buffers = spec.inherit_buffers;
        descriptor.max_vertex_buffer_bind_count = spec.max_vertex_buffer_bind_count;
        descriptor.max_fragment_buffer_bind_count = spec.max_fragment_buffer_bind_count;
        descriptor.max_kernel_buffer_bind_count = spec.max_kernel_buffer_bind_count;
        descriptor.max_kernel_threadgroup_memory_bind_count =
            spec.max_kernel_threadgroup_memory_bind_count;
        descriptor.max_object_buffer_bind_count = spec.max_object_buffer_bind_count;
        descriptor.max_mesh_buffer_bind_count = spec.max_mesh_buffer_bind_count;
        descriptor.max_object_threadgroup_memory_bind_count =
            spec.max_object_threadgroup_memory_bind_count;
        descriptor.support_dynamic_attribute_stride = spec.support_dynamic_attribute_stride;
        descriptor.support_color_attachment_mapping = spec.support_color_attachment_mapping;
        descriptor.inherit_depth_stencil_state = spec.inherit_depth_stencil_state;
        descriptor.inherit_depth_bias = spec.inherit_depth_bias;
        descriptor.inherit_depth_clip_mode = spec.inherit_depth_clip_mode;
        descriptor.inherit_cull_mode = spec.inherit_cull_mode;
        descriptor.inherit_front_facing_winding = spec.inherit_front_facing_winding;
        descriptor.inherit_triangle_fill_mode = spec.inherit_triangle_fill_mode;
        let count = commands.len().max(1) as u64;
        let made = self
            .device
            .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(commands.len());
        for (at, command) in commands.iter().enumerate() {
            let at = at as u64;
            refused.push(match command {
                EncodedCommand::Compute(steps) => {
                    let encoded = made
                        .indirect_compute_command_at(at)
                        .map_err(Failure::from_error)?;
                    steps
                        .iter()
                        .map(|step| self.encode_compute_command(&encoded, step))
                        .collect::<Answer<Vec<Option<Failure>>>>()?
                }
                EncodedCommand::Render(steps) => {
                    let encoded = made
                        .indirect_render_command_at(at)
                        .map_err(Failure::from_error)?;
                    steps
                        .iter()
                        .map(|step| self.encode_render_command(&encoded, step))
                        .collect::<Answer<Vec<Option<Failure>>>>()?
                }
            });
        }
        Ok(CommandsMade {
            commands: keep(&mut self.command_buffers_indirect, made),
            refused,
        })
    }

    fn indirect_commands_size(&mut self, commands: Handle) -> Answer<u64> {
        Ok(self.held_commands(commands)?.size())
    }

    fn reset_indirect_commands(&mut self, commands: Handle, first: u64, count: u64) -> Answer<()> {
        let icb = self.held_commands(commands)?;
        icb.reset_with_range(first..first + count)
            .map_err(Failure::from_error)
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let resident: Vec<Buffer> = used
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let icb = self
            .command_buffers_indirect
            .get(commands.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("command buffer {}", commands.0))
            })?
            .clone();
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let stages = RenderStages::OBJECT
            .union(RenderStages::MESH)
            .union(RenderStages::FRAGMENT);
        let mut refused = Vec::new();
        for buffer in &resident {
            refused.push(
                encoder
                    .use_resource(
                        ResourceRef::Buffer(buffer),
                        ResourceUsage::READ.union(ResourceUsage::WRITE),
                        stages,
                    )
                    .err()
                    .map(|error| Failure::of(&error)),
            );
        }
        refused.push(
            encoder
                .execute_commands_in_buffer(&icb, 0..icb.size())
                .err()
                .map(|error| Failure::of(&error)),
        );
        encoder.end_encoding().map_err(Failure::from_error)?;
        let committed = command_buffer
            .commit()
            .err()
            .map(|error| Failure::of(&error));
        if committed.is_some() {
            refused.push(committed);
            return Ok(PassOutcome {
                refused,
                status: status_value(command_buffer.status()),
                error: None,
            });
        }
        refused.push(None);
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        let format = PixelFormat::try_from(over.format).map_err(|problem| {
            Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
        })?;
        let [width, height] = over.size;
        let mut descriptor =
            TextureDescriptor::texture_2d(format, u64::from(width), u64::from(height), false);
        descriptor.resource_options = ResourceOptions::shared();
        let buffer = self.held(buffer)?.clone();
        let texture = match over.ostype {
            Some(ostype) => {
                descriptor.usage = TextureUsage::SHADER_READ;
                let (offset, row) = over.planes[over.plane as usize];
                let (chroma_offset, chroma_row) = over.planes[1];
                let facts = IoSurfaceYCbCr::new(
                    u32::from_be_bytes(ostype),
                    u64::from(over.plane),
                    [u64::from(width), u64::from(height)],
                    chroma_offset,
                    chroma_row,
                );
                let plane = IoSurfacePlane::new(offset, row, true).with_ycbcr(facts);
                buffer.new_texture_with_iosurface_plane(&descriptor, plane)
            }
            None => {
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                buffer.new_texture(&descriptor, over.planes[0].0, over.planes[0].1)
            }
        }
        .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        let texture = self.held_texture(texture)?.clone();
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        let region = |[x, y, width, height]: [u32; 4]| {
            Region::new_2d(
                u64::from(x),
                u64::from(y),
                u64::from(width),
                u64::from(height),
            )
        };
        let filled = match fill {
            Fill::Pattern {
                level,
                slice,
                region: at,
                pattern,
            } => blit.fill_texture(
                &texture,
                u64::from(*level),
                u64::from(*slice),
                region(*at),
                pattern,
            ),
            Fill::Color {
                level,
                slice,
                region: at,
                color,
                packed_into,
            } => PixelFormat::try_from(*packed_into)
                .map_err(metal2vulkan_runtime::Error::Compile)
                .and_then(|packed_into| {
                    blit.fill_texture_color(
                        &texture,
                        u64::from(*level),
                        u64::from(*slice),
                        region(*at),
                        ClearColor::new(color[0], color[1], color[2], color[3]),
                        Some(packed_into),
                    )
                }),
        };
        blit.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        // The fill's own answer is the case's subject, so a refusal is reported as the call's
        // typed error rather than aborting the encoding: the command buffer still ends and is
        // waited for, as the probe's did, so the bytes it did or did not write are readable.
        let refused = filled.err().map(|error| Failure::of(&error));
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        match refused {
            Some(failure) => Err(failure),
            None => Ok(status_value(command_buffer.status())),
        }
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let mut descriptor = RateMapDescriptor::new(request.screen);
        descriptor.label = request.label.map(str::to_owned);
        // The form is how **Metal's** descriptor object is built, and the runtime has no
        // descriptor object: its `RateMapDescriptor` is a plain struct, so every form but one
        // reaches it the same way. That is the comparison — four Metal construction paths against
        // one struct, judged on what the compiled map derives.
        //
        // `Holed` is the exception, and it is the one claim this side makes rather than echoes:
        // Metal's `layerCount` counts layers from index 0 contiguously, so a descriptor with a
        // layer at index 1 and nil at index 0 answers 0 and compiles a map of **no** layers. The
        // runtime's `layers` is a `Vec`, which cannot hold that hole, so a consumer cannot reach
        // the shape at all — and the honest statement of what the runtime would do with it is an
        // empty `Vec`. A case of this form is the suite asking whether those two are the same map.
        descriptor.layers = match request.form {
            RateMapForm::Holed => Vec::new(),
            _ => request
                .layers
                .iter()
                .map(|layer| {
                    RateMapLayerDescriptor::new(layer.horizontal.clone(), layer.vertical.clone())
                })
                .collect(),
        };
        let map = self
            .device
            .new_rasterization_rate_map(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.rate_maps, map))
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let map = self.held_rate_map(map)?;
        let size = |size: Size| [size.width, size.height, size.depth];
        let sized = map.parameter_buffer_size_and_align();
        Ok(RateMapFacts {
            label: map.label().map(str::to_owned),
            screen_size: size(map.screen_size()),
            physical_granularity: size(map.physical_granularity()),
            layer_count: map.layer_count(),
            physical_sizes: layers
                .iter()
                .map(|layer| size(map.physical_size_for_layer(*layer)))
                .collect(),
            parameter_size: sized.size,
            parameter_align: sized.align,
        })
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let buffer = self.held(into)?.clone();
        let map = self.held_rate_map(map)?;
        map.copy_parameter_data_to_buffer(&buffer, offset)
            .map_err(Failure::from_error)?;
        Ok(())
    }

    fn rate_map_coordinates(
        &mut self,
        map: Handle,
        probes: &[CoordinateProbe],
    ) -> Answer<Vec<MappedCoordinates>> {
        let map = self.held_rate_map(map)?.clone();
        let mut out = Vec::with_capacity(probes.len());
        for probe in probes {
            // A refusal here is reported as one, per site, and is never worked around: the
            // runtime refuses the screen-to-physical map over a map of no layers and over a map of
            // a zero screen extent, which are the two shapes Metal's own call is measured to end
            // its process over. A case that reaches either has asked for a shape the suite says is
            // not a case.
            let physical = map
                .map_screen_to_physical_coordinates(
                    Coordinate2D::new(probe.screen[0], probe.screen[1]),
                    probe.layer,
                )
                .map_err(Failure::from_error)?;
            let back = map
                .map_physical_to_screen_coordinates(
                    Coordinate2D::new(probe.physical[0], probe.physical[1]),
                    probe.layer,
                )
                .map_err(Failure::from_error)?;
            out.push(MappedCoordinates {
                screen_to_physical: [physical.x.to_bits(), physical.y.to_bits()],
                physical_to_screen: [back.x.to_bits(), back.y.to_bits()],
            });
        }
        Ok(out)
    }

    fn rate_map_device_is_the_sides(&mut self, map: Handle) -> Answer<bool> {
        let map = self.held_rate_map(map)?;
        // `registry_id` is the public identity of a device; `Device::same` is the runtime's own
        // crate-private one. Two handles of one device answer one registry ID, which is the whole
        // of what `device` promises.
        Ok(map.device().registry_id() == self.device.registry_id())
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let wanted = row_pitch as usize * size[1] as usize;
        if before.len() != wanted {
            return Err(Failure::named(
                "TooFewBytes",
                format!(
                    "{} bytes before the read, and it reads {wanted}",
                    before.len()
                ),
            ));
        }
        let texture = self.held_texture(texture)?;
        let mut bytes = before.to_vec();
        texture
            .get_bytes(
                &mut bytes,
                row_pitch.into(),
                0,
                Region::new_2d(0, 0, size[0].into(), size[1].into()),
                0,
                0,
            )
            .map_err(Failure::from_error)?;
        Ok(bytes)
    }

    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage(vertex)?);
        descriptor.fragment = Some(stage(fragment)?);
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
        let options = PipelineOptions::new(PipelineOption::BINDING_INFO);
        // The reflection out-parameter is asked for, as the case's Metal side asks for it, and
        // dropped here: what the case reads is `texture_access`, the state's own answer, and the
        // runtime's rule is that asking for a reflection may change what the creation computes.
        let (state, _) = self
            .device
            .new_render_pipeline_state_with_reflection(&descriptor, &options)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let state = self
            .render_pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        indices
            .iter()
            .map(|index| {
                state
                    .texture_access(stage, *index)
                    .map_err(Failure::from_error)
            })
            .collect()
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let descriptor = self.our_tile_descriptor(library, spec)?;
        let state = self
            .device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn tile_pipeline_bindings(
        &mut self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
        options: u64,
    ) -> Answer<StageReflection> {
        let descriptor = self.our_tile_descriptor(library, spec)?;
        let mut built = PipelineOptions::default();
        built.options = PipelineOption::from_raw(options);
        let made = self
            .device
            .new_tile_render_pipeline_state_with_reflection(&descriptor, &built);
        let Ok((state, reflection)) = made else {
            return Ok(StageReflection::refused());
        };
        let found = match reflection {
            Some(reflection) => StageReflection::reflected(our_stage_bindings(&reflection)),
            None => StageReflection::unreflected(),
        };
        // The state is kept so it outlives the reflection its creation answered, as the Metal
        // side's is.
        keep(&mut self.render_pipelines, state);
        Ok(found)
    }

    fn target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        if let Some(plane) = &spec.surface {
            return self.surface_target(spec, plane);
        }
        if let Some(over) = &spec.over_buffer {
            return self.buffer_backed_target(spec, over);
        }
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        descriptor.resource_options = match spec.shared {
            true => ResourceOptions::shared(),
            false => ResourceOptions::private(),
        };
        // The shape before the samples, so a multisampled array target keeps the array type the
        // shape states rather than the plain multisample type the arm below would give it.
        if let Some(shape) = &spec.shape {
            descriptor.texture_type = shape.texture_type;
            descriptor.mipmap_level_count = u64::from(shape.levels);
            descriptor.array_length = u64::from(shape.array_length);
            descriptor.depth = u64::from(shape.depth);
        }
        if spec.samples > 1 {
            if spec.shape.is_none() {
                descriptor.texture_type = TextureType::Type2DMultisample;
            }
            descriptor.sample_count = u64::from(spec.samples);
        }
        let texture = self
            .device
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let (descriptor, options) = self.our_render_descriptor(library, spec)?;
        let state = self
            .device
            .new_render_pipeline_state(&descriptor, &options)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.render_pipelines, state))
    }

    fn pipeline_both_paths(
        &mut self,
        library: Handle,
        form: &PipelineForm<'_>,
    ) -> Answer<BothPaths> {
        self.our_both_paths(library, form)
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let state = self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0)))?;
        Ok(our_tile_state_facts(state))
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let attachments: Vec<Texture> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).cloned())
            .collect::<Answer<Vec<Texture>>>()?;
        let states: Vec<RenderPipelineState> = self.render_pipelines.clone();
        let buffers: Vec<Buffer> = self.buffers.clone();
        let textures: Vec<Texture> = self.textures.clone();
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&attachments) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
        }
        pass.tile_width = u64::from(tile[0]);
        pass.tile_height = u64::from(tile[1]);
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .render_command_encoder(&pass)
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(steps.len());
        for step in steps {
            let answered = match step {
                TileStep::Pipeline(state) => states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0))
                    })
                    .and_then(|state| {
                        encoder
                            .set_render_pipeline_state(state)
                            .map_err(Failure::from_error)
                    }),
                TileStep::TileBuffer {
                    buffer,
                    offset,
                    index,
                } => buffers
                    .get(buffer.0 as usize)
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", buffer.0)))
                    .and_then(|buffer| {
                        encoder
                            .set_tile_buffer(Some(buffer), *offset, *index)
                            .map_err(Failure::from_error)
                    }),
                TileStep::TileBytes { bytes, index } => encoder
                    .set_tile_bytes(bytes, *index)
                    .map_err(Failure::from_error),
                TileStep::TileTexture { texture, index } => textures
                    .get(texture.0 as usize)
                    .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", texture.0)))
                    .and_then(|texture| {
                        encoder
                            .set_tile_texture(Some(texture), *index)
                            .map_err(Failure::from_error)
                    }),
                TileStep::DispatchThreadsPerTile(threads) => encoder
                    .dispatch_threads_per_tile(size(*threads))
                    .map_err(Failure::from_error),
            };
            refused.push(answered.err());
        }
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn compute_pass(&mut self, pass: &ComputePass<'_>) -> Answer<ComputePassOutcome> {
        // The pools are cloned once, so every command can reach the case's objects while the
        // command buffer is borrowed.
        let buffers: Vec<Buffer> = self.buffers.clone();
        let textures: Vec<Texture> = self.textures.clone();
        let samplers: Vec<SamplerState> = self.samplers.clone();
        let states: Vec<ComputePipelineState> = self.pipelines.clone();
        let held_heaps: Vec<Heap> = self.heaps.clone();
        let heap = |handle: Handle| -> Answer<&Heap> {
            held_heaps
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
        };
        let buffer = |handle: Handle| -> Answer<&Buffer> {
            buffers
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
        };
        let texture = |handle: Handle| -> Answer<&Texture> {
            textures
                .get(handle.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
        };
        let sampler = |handle: Handle| -> Answer<Option<&SamplerState>> {
            samplers
                .get(handle.0 as usize)
                .map(Some)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
        };
        let resource = |named: &PassResource| -> Answer<ResourceRef<'_>> {
            Ok(match named {
                PassResource::Buffer(handle) => ResourceRef::Buffer(buffer(*handle)?),
                PassResource::Texture(handle) => ResourceRef::Texture(texture(*handle)?),
            })
        };
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut encoder = command_buffer
            .compute_command_encoder_with_dispatch_type(match pass.dispatching {
                Dispatching::Serial => DispatchType::Serial,
                Dispatching::Concurrent => DispatchType::Concurrent,
            })
            .map_err(Failure::from_error)?;
        let dispatch_type = Value::unit_variant(match encoder.dispatch_type() {
            DispatchType::Concurrent => "Concurrent",
            _ => "Serial",
        });
        let mut refused = Vec::with_capacity(pass.steps.len());
        for step in pass.steps {
            let answered = match step {
                ComputeStep::Pipeline(state) => states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("compute pipeline {}", state.0))
                    })
                    .and_then(|state| {
                        encoder
                            .set_compute_pipeline_state(state)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::Buffer {
                    buffer: handle,
                    offset,
                    index,
                } => buffer(*handle).and_then(|held| {
                    encoder
                        .set_buffer(Some(held), *offset, *index)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::BufferOffset { offset, index } => encoder
                    .set_buffer_offset(*offset, *index)
                    .map_err(Failure::from_error),
                ComputeStep::Buffers {
                    buffers: named,
                    first,
                } => named
                    .iter()
                    .map(|(handle, _)| buffer(*handle).map(Some))
                    .collect::<Answer<Vec<Option<&Buffer>>>>()
                    .and_then(|held| {
                        let offsets: Vec<u64> = named.iter().map(|(_, at)| *at).collect();
                        encoder
                            .set_buffers(&held, &offsets, *first)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::Bytes { bytes, index } => encoder
                    .set_bytes(bytes, *index)
                    .map_err(Failure::from_error),
                ComputeStep::BufferWithStride {
                    buffer: handle,
                    offset,
                    stride,
                    index,
                } => buffer(*handle).and_then(|held| {
                    encoder
                        .set_buffer_with_stride(Some(held), *offset, *stride, *index)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::BuffersWithStrides {
                    buffers: named,
                    first,
                } => named
                    .iter()
                    .map(|(handle, _, _)| buffer(*handle).map(Some))
                    .collect::<Answer<Vec<Option<&Buffer>>>>()
                    .and_then(|held| {
                        let offsets: Vec<u64> = named.iter().map(|(_, at, _)| *at).collect();
                        let strides: Vec<u64> = named.iter().map(|(_, _, by)| *by).collect();
                        encoder
                            .set_buffers_with_strides(&held, &offsets, &strides, *first)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::BufferOffsetWithStride {
                    offset,
                    stride,
                    index,
                } => encoder
                    .set_buffer_offset_with_stride(*offset, *stride, *index)
                    .map_err(Failure::from_error),
                ComputeStep::BytesWithStride {
                    bytes,
                    stride,
                    index,
                } => encoder
                    .set_bytes_with_stride(bytes, *stride, *index)
                    .map_err(Failure::from_error),
                ComputeStep::StageInRegion { origin, size } => encoder
                    .set_stage_in_region(Region::new_3d(
                        origin[0].into(),
                        origin[1].into(),
                        origin[2].into(),
                        size[0].into(),
                        size[1].into(),
                        size[2].into(),
                    ))
                    .map_err(Failure::from_error),
                ComputeStep::Texture {
                    texture: handle,
                    index,
                } => texture(*handle).and_then(|held| {
                    encoder
                        .set_texture(Some(held), *index)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::Textures {
                    textures: named,
                    first,
                } => named
                    .iter()
                    .map(|handle| texture(*handle).map(Some))
                    .collect::<Answer<Vec<Option<&Texture>>>>()
                    .and_then(|held| {
                        encoder
                            .set_textures(&held, *first)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::Sampler {
                    sampler: handle,
                    index,
                } => sampler(*handle).and_then(|held| {
                    encoder
                        .set_sampler_state(held, *index)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::Samplers {
                    samplers: named,
                    first,
                } => named
                    .iter()
                    .map(|handle| sampler(*handle))
                    .collect::<Answer<Vec<Option<&SamplerState>>>>()
                    .and_then(|held| {
                        encoder
                            .set_sampler_states(&held, *first)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::ClampedSampler {
                    sampler: handle,
                    lod_min,
                    lod_max,
                    index,
                } => sampler(*handle).and_then(|held| {
                    encoder
                        .set_sampler_state_with_lod_clamps(held, *lod_min, *lod_max, *index)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::ClampedSamplers {
                    samplers: named,
                    first,
                } => named
                    .iter()
                    .map(|(handle, _, _)| sampler(*handle))
                    .collect::<Answer<Vec<Option<&SamplerState>>>>()
                    .and_then(|held| {
                        let mins: Vec<f32> = named.iter().map(|(_, min, _)| *min).collect();
                        let maxes: Vec<f32> = named.iter().map(|(_, _, max)| *max).collect();
                        encoder
                            .set_sampler_states_with_lod_clamps(&held, &mins, &maxes, *first)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::ThreadgroupMemory { length, index } => encoder
                    .set_threadgroup_memory_length(*length, *index)
                    .map_err(Failure::from_error),
                ComputeStep::UseResource {
                    resource: named,
                    usage,
                } => resource(named).and_then(|held| {
                    encoder
                        .use_resource(held, resource_usage(*usage)?)
                        .map_err(Failure::from_error)
                }),
                ComputeStep::UseResources {
                    resources: named,
                    usage,
                } => named
                    .iter()
                    .map(resource)
                    .collect::<Answer<Vec<ResourceRef<'_>>>>()
                    .and_then(|held| {
                        encoder
                            .use_resources(&held, resource_usage(*usage)?)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::UseHeap { heap: handle } => heap(*handle)
                    .and_then(|held| encoder.use_heap(held).map_err(Failure::from_error)),
                ComputeStep::UseHeaps { heaps: named } => named
                    .iter()
                    .map(|handle| heap(*handle))
                    .collect::<Answer<Vec<&Heap>>>()
                    .and_then(|held| encoder.use_heaps(&held).map_err(Failure::from_error)),
                ComputeStep::BarrierScope { scope } => barrier_scope(*scope).map(|scope| {
                    encoder.memory_barrier_with_scope(scope);
                }),
                ComputeStep::BarrierResources { resources: named } => named
                    .iter()
                    .map(resource)
                    .collect::<Answer<Vec<ResourceRef<'_>>>>()
                    .and_then(|held| {
                        encoder
                            .memory_barrier_with_resources(&held)
                            .map_err(Failure::from_error)
                    }),
                ComputeStep::DispatchThreadgroups { groups, threads } => encoder
                    .dispatch_threadgroups(size(*groups), size(*threads))
                    .map_err(Failure::from_error),
                ComputeStep::DispatchThreadgroupsIndirect {
                    buffer: handle,
                    offset,
                    threads,
                } => buffer(*handle).and_then(|held| {
                    encoder
                        .dispatch_threadgroups_indirect(held, *offset, size(*threads))
                        .map_err(Failure::from_error)
                }),
            };
            refused.push(answered.err());
        }
        encoder.end_encoding().map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(ComputePassOutcome {
                refused,
                dispatch_type,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(ComputePassOutcome {
            refused,
            dispatch_type,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let state = self
            .device
            .new_depth_stencil_state(spec)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.depth_stencil_states, state))
    }

    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let buffer = self.held_plane(target)?.clone();
        if bytes.len() != buffer.length() as usize {
            return Err(Failure::named(
                "PlaneLength",
                format!("{} bytes for a plane of {}", bytes.len(), buffer.length()),
            ));
        }
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer is of shared storage and at least `bytes.len()` bytes long, checked
        // just above, and no command uses it now: a case writes the plane before it opens its pass.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let buffer = self.held_plane(target)?.clone();
        let length = buffer.length() as usize;
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer's own `length` bytes of shared storage, and the caller waited for the
        // command buffer that wrote them.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) };
        Ok(bytes.to_vec())
    }

    fn render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        // The pools are cloned once, so each piece of the program can reach the case's objects
        // while the command buffer is borrowed.
        let pools = Pools {
            textures: self.textures.clone(),
            buffers: self.buffers.clone(),
            states: self.render_pipelines.clone(),
            depth_stencil: self.depth_stencil_states.clone(),
            samplers: self.samplers.clone(),
            computes: self.pipelines.clone(),
            indirect: self.command_buffers_indirect.clone(),
            heaps: self.heaps.clone(),
        };
        let texture = |handle: Handle| pools.texture(handle);
        let buffer = |handle: Handle| pools.buffer(handle);
        let command_buffer = self
            .queue()?
            .command_buffer()
            .map_err(Failure::from_error)?;
        let mut refused = Vec::with_capacity(work.len());
        for piece in work {
            let answered = match piece {
                RenderWork::Pass(pass) => encode_pass(&command_buffer, pass, &pools),
                RenderWork::ReadTexture {
                    texture: from,
                    into,
                    offset,
                    row_pitch,
                    size: extent,
                    plane,
                    at,
                } => texture(*from).and_then(|from| {
                    let into = buffer(*into)?;
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.copy_from_texture_to_buffer(
                        from,
                        u64::from(at.slice),
                        u64::from(at.level),
                        Origin::new(0, 0, u64::from(at.depth_plane)),
                        Size::new(u64::from(extent[0]), u64::from(extent[1]), 1),
                        into,
                        *offset,
                        u64::from(*row_pitch),
                        u64::from(*row_pitch) * u64::from(extent[1]),
                        crate::runtime_work::option_of(*plane),
                    )
                    .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::CopyBuffer {
                    from,
                    from_offset,
                    into,
                    into_offset,
                    length,
                } => buffer(*from).and_then(|from| {
                    let into = buffer(*into)?;
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.copy_from_buffer_to_buffer(
                        from,
                        *from_offset,
                        into,
                        *into_offset,
                        *length,
                    )
                    .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::Dispatch(dispatch) => pools
                    .computes
                    .get(dispatch.pipeline.0 as usize)
                    .ok_or_else(|| {
                        Failure::named(
                            "NoSuchHandle",
                            format!("compute pipeline {}", dispatch.pipeline.0),
                        )
                    })
                    .and_then(|state| {
                        let mut encoder = command_buffer
                            .compute_command_encoder()
                            .map_err(Failure::from_error)?;
                        encoder
                            .set_compute_pipeline_state(state)
                            .map_err(Failure::from_error)?;
                        for (at, (held, offset)) in dispatch.buffers.iter().enumerate() {
                            encoder
                                .set_buffer(Some(buffer(*held)?), *offset, at as u32)
                                .map_err(Failure::from_error)?;
                        }
                        for (at, held) in dispatch.textures.iter().enumerate() {
                            encoder
                                .set_texture(Some(texture(*held)?), at as u32)
                                .map_err(Failure::from_error)?;
                        }
                        encoder
                            .dispatch_threads(size(dispatch.grid), size(dispatch.threads))
                            .map_err(Failure::from_error)?;
                        encoder.end_encoding().map_err(Failure::from_error)
                    }),
                RenderWork::ComputeCommands {
                    commands,
                    first,
                    count,
                    used,
                } => pools.indirect(*commands).and_then(|icb| {
                    let mut encoder = command_buffer
                        .compute_command_encoder()
                        .map_err(Failure::from_error)?;
                    for (held, write) in used {
                        encoder
                            .use_resource(
                                ResourceRef::Buffer(buffer(*held)?),
                                commands_usage(*write),
                            )
                            .map_err(Failure::from_error)?;
                    }
                    encoder
                        .execute_commands_in_buffer(icb, *first..*first + *count)
                        .map_err(Failure::from_error)?;
                    encoder.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::ResetCommands {
                    commands,
                    first,
                    count,
                } => pools.indirect(*commands).and_then(|icb| {
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.reset_commands_in_buffer(icb, *first..*first + *count)
                        .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::CopyCommands {
                    from,
                    first,
                    count,
                    into,
                    into_index,
                } => pools.indirect(*from).and_then(|source| {
                    let destination = pools.indirect(*into)?;
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.copy_indirect_command_buffer(
                        source,
                        *first..*first + *count,
                        destination,
                        *into_index,
                    )
                    .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
                RenderWork::OptimizeCommands {
                    commands,
                    first,
                    count,
                } => pools.indirect(*commands).and_then(|icb| {
                    let mut blit = command_buffer
                        .blit_command_encoder()
                        .map_err(Failure::from_error)?;
                    blit.optimize_indirect_command_buffer(icb, *first..*first + *count)
                        .map_err(Failure::from_error)?;
                    blit.end_encoding().map_err(Failure::from_error)
                }),
            };
            refused.push(answered.err());
        }
        command_buffer.commit().map_err(Failure::from_error)?;
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(PassOutcome {
                refused,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        }
        Ok(PassOutcome {
            refused,
            status: status_value(command_buffer.status()),
            error: command_buffer
                .error()
                .map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }

    fn heap_texture_size_and_align(&mut self, query: &HeapTextureQuery) -> Answer<SizeAndAlign> {
        let answered = self
            .device
            .heap_texture_size_and_align(&queried_descriptor(query))
            .map_err(Failure::from_error)?;
        Ok(SizeAndAlign {
            size: answered.size,
            align: answered.align,
        })
    }

    fn heap_buffer_size_and_align(&mut self, query: &HeapBufferQuery) -> Answer<SizeAndAlign> {
        let options = ResourceOptions::from_raw(query.options).map_err(Failure::from_error)?;
        let answered = self
            .device
            .heap_buffer_size_and_align(query.length, options);
        Ok(SizeAndAlign {
            size: answered.size,
            align: answered.align,
        })
    }

    fn supports_vertex_amplification_count(&mut self, count: u64) -> Answer<bool> {
        self.device
            .properties()
            .supports_vertex_amplification_count(count)
            .map_err(|gap| Failure::named("RuntimeGap", format!("{gap:?}")))
    }

    fn supports_rasterization_rate_map_layers(&mut self, layers: u64) -> Answer<bool> {
        self.device
            .properties()
            .supports_rasterization_rate_map_with_layer_count(layers)
            .map_err(|gap| Failure::named("RuntimeGap", format!("{gap:?}")))
    }

    fn minimum_linear_texture_alignment(&mut self, format: PixelFormat) -> Answer<u64> {
        self.device
            .properties()
            .minimum_linear_texture_alignment_for_pixel_format(format)
            .map_err(Failure::from_error)
    }

    fn convert_sparse_pixel_regions(&mut self, query: &RegionQuery) -> Answer<Vec<Box6>> {
        let pixels: Vec<Region> = query.regions.iter().map(our_region).collect();
        let mut tiles: Vec<Region> = pixels.clone();
        let mode = match query.inward {
            true => SparseTextureRegionAlignmentMode::Inward,
            false => SparseTextureRegionAlignmentMode::Outward,
        };
        self.device
            .convert_sparse_pixel_regions(&pixels, &mut tiles, our_size(query.tile), mode)
            .map_err(Failure::from_error)?;
        Ok(tiles.iter().map(their_region).collect())
    }

    fn convert_sparse_tile_regions(&mut self, query: &RegionQuery) -> Answer<Vec<Box6>> {
        let tiles: Vec<Region> = query.regions.iter().map(our_region).collect();
        let pixels = self
            .device
            .convert_sparse_tile_regions(&tiles, our_size(query.tile));
        Ok(pixels.iter().map(their_region).collect())
    }

    fn queue_slots(&mut self, max: Option<u32>, cap: u32) -> Answer<QueueSlots> {
        let queue = match max {
            None => self.device.new_command_queue(),
            Some(max) => self
                .device
                .new_command_queue_with_max_command_buffer_count(max as usize),
        }
        .map_err(Failure::from_error)?;
        let other = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let making = queue.clone();
        crate::queue_slots::probe(
            cap,
            move || making.command_buffer().map_err(Failure::from_error),
            || other.command_buffer().is_ok(),
        )
    }

    fn handler_wait(&mut self, wait: crate::handler_waits::OwnWait) -> Answer<Value> {
        use crate::handler_waits::{HandlerKind, WaitKind};
        use std::sync::mpsc;
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let buffer = self
            .device
            .new_buffer(HANDLER_FILL, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.fill_buffer(&buffer, 0..HANDLER_FILL, 0x11)
            .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        let (sender, answer) = mpsc::channel();
        let kind = wait.wait;
        let handler: metal2vulkan_runtime::Handler = Box::new(move |own: &CommandBuffer| {
            let answered = match kind {
                WaitKind::UntilCompleted => own.wait_until_completed(),
                WaitKind::UntilScheduled => own.wait_until_scheduled(),
            };
            let _ = sender.send(answered.err().map(|error| Failure::of(&error)));
        });
        match wait.from {
            HandlerKind::Scheduled => command_buffer.add_scheduled_handler(handler),
            HandlerKind::Completed => command_buffer.add_completed_handler(handler),
        }
        .map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        // Bounded, so a runtime that did not refuse the handler's wait is recorded rather than
        // hanging the suite: the handler would then still be inside its wait and the command
        // buffer would never complete.
        if command_buffer.wait_until_completed_for(WAIT).is_err() {
            return Ok(Value::Nothing(Negative::NeverReturned));
        }
        match answer.recv_timeout(WAIT) {
            Ok(None) => Ok(Value::ok(Value::Absent)),
            Ok(Some(failure)) => Err(failure),
            Err(_) => Ok(Value::Nothing(Negative::NeverReturned)),
        }
    }

    fn detach_in_a_completed_handler(
        &mut self,
        buffer: Handle,
        after: Duration,
    ) -> Answer<crate::host_detach::DetachedInHandler> {
        use crate::event_orders::Status;
        use crate::host_detach::{DetachedInHandler, FILL_BYTE, FILL_BYTES};
        use std::sync::mpsc;
        let detaching = self.held(buffer)?.clone();
        let length = detaching.length().min(FILL_BYTES);
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        // The command buffer has to *use* the buffer: a submission that named it is the only thing
        // that can hold it, and a detach asked about a buffer nothing took is a different
        // question.
        blit.fill_buffer(&detaching, 0..length, FILL_BYTE)
            .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        let (sender, answered) = mpsc::channel();
        let handler: metal2vulkan_runtime::Handler = Box::new(move |_: &CommandBuffer| {
            // Past the race between the completion's own release and this handler's start: the
            // region where Metal's answer is deterministic (`crate::host_detach`).
            std::thread::sleep(after);
            let _ = sender.send(detaching.detach_backing());
        });
        command_buffer
            .add_completed_handler(handler)
            .map_err(Failure::from_error)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        // The handler's own channel, with a bound: an answer that did not arrive is the negative
        // fact, and this thread never waits for the command buffer itself, which is the one wait
        // Metal has no bounded form of (`crate::host_detach`).
        let (detached, status) = match answered.recv_timeout(crate::host_detach::WAIT) {
            Ok(Ok(detached)) => (
                Some(detached),
                Status::read(format!("{:?}", command_buffer.status())),
            ),
            Ok(Err(error)) => return Err(Failure::of(&error)),
            Err(_) => (None, Status::NeverSettled),
        };
        Ok(DetachedInHandler { detached, status })
    }

    fn shared_event_handle_round_trip(
        &mut self,
        label: &str,
    ) -> Answer<crate::shared_events::EventHandleRoundTrip> {
        use crate::shared_events::{
            EventHandleRoundTrip, HANDLE_FIRST_VALUE, HANDLE_SECOND_VALUE, HANDLE_UNREACHED_VALUE,
            WAIT_MS,
        };
        let bound = Duration::from_millis(WAIT_MS);
        let registry = self.device.registry_id();
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        event.set_label(label);
        let read_label = event.label();
        let device_is_the_cases = event
            .device()
            .is_some_and(|device| device.registry_id() == registry);
        let value_at_first = event.signaled_value();
        event.set_signaled_value(HANDLE_FIRST_VALUE);
        let value_after_the_host_set_it = event.signaled_value();
        let handle = event.new_shared_event_handle();
        let handle_label = handle.label();
        let reopened = self
            .device
            .new_shared_event_with_handle(&handle)
            .map_err(Failure::from_error)?;
        let reopened_label = reopened.label();
        let reopened_device_is_the_cases = reopened
            .device()
            .is_some_and(|device| device.registry_id() == registry);
        let reopened_value = reopened.signaled_value();
        reopened.set_signaled_value(HANDLE_SECOND_VALUE);
        let value_on_the_first_after_the_second_was_set = event.signaled_value();
        let reopened_waits = [
            reopened.wait_until_signaled_value(HANDLE_SECOND_VALUE, bound),
            reopened.wait_until_signaled_value(HANDLE_UNREACHED_VALUE, bound),
        ];
        Ok(EventHandleRoundTrip {
            label: read_label,
            device_is_the_cases,
            value_at_first,
            value_after_the_host_set_it,
            handle_label,
            reopened_label,
            reopened_device_is_the_cases,
            reopened_value,
            value_on_the_first_after_the_second_was_set,
            reopened_waits,
        })
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        use crate::shared_events::{EventRise, WAIT_MS};
        let bound = Duration::from_millis(WAIT_MS);
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        event.set_label(label);
        let read_label = event.label();
        let initial = event.signaled_value();
        // The host's sets: a lower value leaves the event where it was.
        event.set_signaled_value(5);
        event.set_signaled_value(3);
        event.set_signaled_value(5);
        let after_host_sets = event.signaled_value();
        let signal = |value: u64| -> Answer<u64> {
            let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), value)
                .map_err(Failure::from_error)?;
            command_buffer.commit().map_err(Failure::from_error)?;
            if command_buffer.wait_until_completed_for(WAIT).is_err() {
                return Err(Failure::named(
                    "NeverCompleted",
                    format!("the command buffer signalling {value}"),
                ));
            }
            Ok(event.signaled_value())
        };
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [
            event.wait_until_signaled_value(3, bound),
            event.wait_until_signaled_value(5, bound),
            event.wait_until_signaled_value(6, bound),
        ];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = event.wait_until_signaled_value(u64::MAX, bound);
        Ok(EventRise {
            label: read_label,
            initial,
            after_host_sets,
            after_gpu_signals,
            waits,
            after_max_signal,
            waited_for_max,
        })
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        use crate::event_orders::{
            CrossQueueWait, Status, HALF, SCRATCH, SIGNALLING_BYTE, WAITING_BYTE,
        };
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let (waiting_queue, signalling_queue) = (queue()?, queue()?);
        let event = self.device.new_event().map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let fill = |command_buffer: &CommandBuffer,
                    range: std::ops::Range<u64>,
                    byte: u8|
         -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, range, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let waiting = waiting_queue
            .command_buffer()
            .map_err(Failure::from_error)?;
        waiting
            .encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        fill(&waiting, 0..HALF, WAITING_BYTE)?;
        let signalling = signalling_queue
            .command_buffer()
            .map_err(Failure::from_error)?;
        fill(&signalling, HALF..SCRATCH, SIGNALLING_BYTE)?;
        signalling
            .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        // The waiting one first, which is the whole point: its signal is not committed yet, so a
        // runtime that submitted in commit order onto one shared queue would deadlock here.
        waiting.commit().map_err(Failure::from_error)?;
        let waiting_after_its_commit = Status::read(format!("{:?}", waiting.status()));
        signalling.commit().map_err(Failure::from_error)?;
        let settled = |command_buffer: &CommandBuffer| {
            if command_buffer
                .wait_until_completed_for(crate::event_orders::COMPLETION)
                .is_err()
            {
                Status::NeverSettled
            } else {
                Status::read(format!("{:?}", command_buffer.status()))
            }
        };
        let signalling_at_the_end = settled(&signalling);
        let waiting_at_the_end = settled(&waiting);
        let bytes = read_shared(&scratch, 0, SCRATCH as usize)?;
        Ok(CrossQueueWait {
            waiting_after_its_commit,
            waiting_at_the_end,
            signalling_at_the_end,
            bytes,
        })
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        use crate::event_orders::{COMPLETION, SCRATCH};
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let queues = [queue()?, queue()?];
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let mut committed = Vec::new();
        for (queue, &value) in queues.iter().zip(&values) {
            let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, value as u8)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)?;
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(&event), value)
                .map_err(Failure::from_error)?;
            command_buffer.commit().map_err(Failure::from_error)?;
            committed.push(command_buffer);
        }
        for command_buffer in &committed {
            if command_buffer.wait_until_completed_for(COMPLETION).is_err() {
                return Err(Failure::named(
                    "NeverCompleted",
                    "a signalling command buffer",
                ));
            }
        }
        Ok(event.signaled_value())
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        use crate::event_orders::{
            HeldCommandBuffer, Status, COMPLETED, COMPLETION, HELD_FOR, SCHEDULED, SCRATCH,
        };
        let queue = || self.device.new_command_queue().map_err(Failure::from_error);
        let (held_queue, other_queue) = (queue()?, queue()?);
        let event = self
            .device
            .new_shared_event()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let fill = |command_buffer: &CommandBuffer, byte: u8| -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let log = HandlerLog::new();
        let held = held_queue.command_buffer().map_err(Failure::from_error)?;
        held.encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&event), 1)
            .map_err(Failure::from_error)?;
        fill(&held, 1)?;
        held.add_scheduled_handler(log.handler(SCHEDULED))
            .map_err(Failure::from_error)?;
        held.add_completed_handler(log.handler(COMPLETED))
            .map_err(Failure::from_error)?;
        held.commit().map_err(Failure::from_error)?;
        let behind = held_queue.command_buffer().map_err(Failure::from_error)?;
        fill(&behind, 2)?;
        behind.commit().map_err(Failure::from_error)?;
        let other = other_queue.command_buffer().map_err(Failure::from_error)?;
        fill(&other, 3)?;
        other.commit().map_err(Failure::from_error)?;
        let other_queue_status = if other.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", other.status()))
        };
        // The window the negative facts are false in: a submission that was going to happen has
        // had its chance by now.
        std::thread::sleep(HELD_FOR);
        let held_while_waiting = Status::read(format!("{:?}", held.status()));
        let behind_while_waiting = Status::read(format!("{:?}", behind.status()));
        let handlers_while_waiting = log.names();
        event.set_signaled_value(1);
        // The held command buffer's own completed handler is the bound: it is the last of its
        // handlers to run, so once it has, the log is whole and the status is settled.
        log.wait_for(COMPLETED, COMPLETION);
        let handlers_at_the_end = log.names();
        let held_at_the_end = if handlers_at_the_end.iter().any(|name| name == COMPLETED) {
            Status::read(format!("{:?}", held.status()))
        } else {
            Status::NeverSettled
        };
        let behind_at_the_end = if behind.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", behind.status()))
        };
        Ok(HeldCommandBuffer {
            other_queue: other_queue_status,
            held_while_waiting,
            behind_while_waiting,
            handlers_while_waiting,
            handlers_at_the_end,
            held_at_the_end,
            behind_at_the_end,
        })
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        use crate::event_orders::{Status, COMPLETION, HELD_FOR, SCRATCH, SIGNAL_WAIT};
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let event = || self.device.new_shared_event().map_err(Failure::from_error);
        let (before, gate, after) = (event()?, event()?, event()?);
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let fill = |byte: u8| -> Result<(), Failure> {
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(Failure::from_error)?;
            blit.fill_buffer(&scratch, 0..SCRATCH, byte)
                .map_err(Failure::from_error)?;
            blit.end_encoding().map_err(Failure::from_error)
        };
        let signal = |event: &metal2vulkan_runtime::SharedEvent| {
            command_buffer
                .encode_signal_event(metal2vulkan_runtime::EventRef::from(event), 1)
                .map_err(Failure::from_error)
        };
        fill(1)?;
        signal(&before)?;
        command_buffer
            .encode_wait_for_event(metal2vulkan_runtime::EventRef::from(&gate), 1)
            .map_err(Failure::from_error)?;
        fill(2)?;
        signal(&after)?;
        command_buffer.commit().map_err(Failure::from_error)?;
        let before_landed = before.wait_until_signaled_value(1, SIGNAL_WAIT);
        // The window the two negative facts are false in: the signal behind the wait has not
        // landed, and the command buffer is no further along than committed.
        std::thread::sleep(HELD_FOR);
        let after_while_pending = after.signaled_value();
        let status_while_pending = Status::read(format!("{:?}", command_buffer.status()));
        gate.set_signaled_value(1);
        let status_at_the_end = if command_buffer.wait_until_completed_for(COMPLETION).is_err() {
            Status::NeverSettled
        } else {
            Status::read(format!("{:?}", command_buffer.status()))
        };
        Ok(crate::event_orders::SignalBeforeWait {
            before_landed,
            after_while_pending,
            status_while_pending,
            after_at_the_end: after.signaled_value(),
            status_at_the_end,
        })
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        use crate::event_orders::{Status, COMPLETED, SCHEDULED, SCRATCH};
        use crate::scheduled_handlers::{
            Returned, ScheduledHandlerOrder, ARRIVAL, HOLD_LIMIT, WINDOW,
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{mpsc, Arc, Mutex};
        let queue = self
            .device
            .new_command_queue()
            .map_err(Failure::from_error)?;
        let scratch = self
            .device
            .new_buffer(SCRATCH, ResourceOptions::private())
            .map_err(Failure::from_error)?;
        let command_buffer = queue.command_buffer().map_err(Failure::from_error)?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(Failure::from_error)?;
        blit.fill_buffer(&scratch, 0..SCRATCH, 1)
            .map_err(Failure::from_error)?;
        blit.end_encoding().map_err(Failure::from_error)?;
        let order: Arc<Mutex<Vec<String>>> = Arc::default();
        let (entered, entering) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        {
            let order = Arc::clone(&order);
            command_buffer
                .add_scheduled_handler(Box::new(move |own: &CommandBuffer| {
                    if let Ok(mut order) = order.lock() {
                        order.push(SCHEDULED.to_owned());
                    }
                    let _ = entered.send(format!("{:?}", own.status()));
                    // Held, so the probe can read the status and the wait while this handler is
                    // inside itself. Bounded, so the handler leaves whatever the probe does.
                    let _ = released.recv_timeout(HOLD_LIMIT);
                }))
                .map_err(Failure::from_error)?;
        }
        let (done, completing) = mpsc::channel();
        {
            let order = Arc::clone(&order);
            command_buffer
                .add_completed_handler(Box::new(move |own: &CommandBuffer| {
                    if let Ok(mut order) = order.lock() {
                        order.push(COMPLETED.to_owned());
                    }
                    let _ = done.send(format!("{:?}", own.status()));
                }))
                .map_err(Failure::from_error)?;
        }
        command_buffer.commit().map_err(Failure::from_error)?;
        let inside_the_handler = match entering.recv_timeout(ARRIVAL) {
            Ok(word) => Status::read(word),
            Err(_) => Status::NeverSettled,
        };
        let while_the_handler_ran = Status::read(format!("{:?}", command_buffer.status()));
        let returned = Arc::new(AtomicBool::new(false));
        let (answered, answer) = mpsc::channel();
        {
            let (waited, returned) = (command_buffer.clone(), Arc::clone(&returned));
            // The wait runs on a thread of its own because the fact under test is that it does
            // **not** return; the thread is left to finish rather than joined, so a wait that
            // never came back leaves that thread and not the suite.
            std::thread::spawn(move || {
                let answer = waited.wait_until_scheduled();
                returned.store(true, Ordering::SeqCst);
                let _ = answered.send(answer.err().map(|error| Failure::of(&error)));
            });
        }
        // The window the negative fact is false in.
        std::thread::sleep(WINDOW);
        let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
        let _ = release.send(());
        let wait_answered = match answer.recv_timeout(ARRIVAL) {
            Ok(None) => Returned::Ok,
            Ok(Some(failure)) => Returned::Err(failure.recorded()),
            Err(_) => Returned::NotWithinTheWindow,
        };
        let inside_the_completed_handler = match completing.recv_timeout(ARRIVAL) {
            Ok(word) => Status::read(word),
            Err(_) => Status::NeverSettled,
        };
        let handler_order = order.lock().map(|order| order.clone()).unwrap_or_default();
        Ok(ScheduledHandlerOrder {
            inside_the_handler,
            while_the_handler_ran,
            wait_returned_while_the_handler_ran,
            wait_answered,
            inside_the_completed_handler,
            handler_order,
        })
    }

    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        let buffer = self
            .device
            .new_buffer(
                request.buffer_length,
                resource_options(request.buffer_storage),
            )
            .map_err(Failure::from_error)?;
        let texture = buffer
            .new_texture(
                &queried_descriptor(&request.descriptor),
                request.offset,
                request.bytes_per_row,
            )
            .map_err(Failure::from_error)?;
        let facts = crate::buffer_texture::BufferTextureFacts {
            texture_type: u64::from(texture.texture_type().as_raw()),
            pixel_format: u64::from(texture.pixel_format().as_raw()),
            width: texture.width(),
            height: texture.height(),
            depth: texture.depth(),
            mipmap_level_count: texture.mipmap_level_count(),
            array_length: texture.array_length(),
            sample_count: texture.sample_count(),
            usage: u64::from(texture.usage().as_raw()),
            storage_mode: texture.storage_mode().raw(),
            cpu_cache_mode: texture.cpu_cache_mode().raw(),
            hazard_tracking_mode: texture.hazard_tracking_mode().raw(),
            gpu_optimized: texture.allow_gpu_optimized_contents(),
        };
        // Both are kept, as the Metal side keeps them: a texture over a buffer the case dropped
        // would be a texture over freed memory, and a later row could read it.
        self.buffers.push(buffer);
        self.textures.push(texture);
        Ok(facts)
    }

    fn queried_texture_facts(
        &mut self,
        query: &HeapTextureQuery,
    ) -> Answer<Option<crate::buffer_texture::BufferTextureFacts>> {
        let texture = match self.device.new_texture(&queried_descriptor(query)) {
            Ok(texture) => texture,
            // Plain Metal answers nil for a descriptor it describes no texture of, and the runtime
            // answers the typed `Nil` for the same descriptor: the same fact, so both sides answer
            // the absent texture. Anything else is our own refusal and is recorded as one.
            Err(metal2vulkan_runtime::Error::Nil(_)) => return Ok(None),
            Err(error) => return Err(Failure::of(&error)),
        };
        let facts = crate::buffer_texture::BufferTextureFacts {
            texture_type: u64::from(texture.texture_type().as_raw()),
            pixel_format: u64::from(texture.pixel_format().as_raw()),
            width: texture.width(),
            height: texture.height(),
            depth: texture.depth(),
            mipmap_level_count: texture.mipmap_level_count(),
            array_length: texture.array_length(),
            sample_count: texture.sample_count(),
            usage: u64::from(texture.usage().as_raw()),
            storage_mode: texture.storage_mode().raw(),
            cpu_cache_mode: texture.cpu_cache_mode().raw(),
            hazard_tracking_mode: texture.hazard_tracking_mode().raw(),
            gpu_optimized: texture.allow_gpu_optimized_contents(),
        };
        self.textures.push(texture);
        Ok(Some(facts))
    }

    fn makes_texture(&mut self, query: &HeapTextureQuery) -> Answer<bool> {
        match self.device.new_texture(&queried_descriptor(query)) {
            Ok(_) => Ok(true),
            // Metal answers nil for a descriptor it describes no texture of, and the runtime
            // answers the typed `Nil` for the same descriptor: the same fact, so both sides answer
            // `false`. Anything else is our own refusal and is recorded as one.
            Err(metal2vulkan_runtime::Error::Nil(_)) => Ok(false),
            Err(error) => Err(Failure::of(&error)),
        }
    }

    fn sparse_tile_size(&mut self, query: &SparseTileQuery) -> Answer<TileSize> {
        let samples = u64::from(query.samples);
        let size = match query.page {
            None => self
                .device
                .sparse_tile_size(query.texture_type, query.format, samples),
            Some(page) => self.device.sparse_tile_size_for_page(
                query.texture_type,
                query.format,
                samples,
                page,
            ),
        };
        Ok(TileSize {
            width: size.width,
            height: size.height,
            depth: size.depth,
        })
    }

    fn sparse_tile_size_in_bytes(&mut self, page: QueryPage) -> Answer<u64> {
        Ok(match page {
            None => self.device.sparse_tile_size_in_bytes(),
            Some(page) => self.device.sparse_tile_size_in_bytes_for_page(page),
        })
    }

    fn sparse_heap(&mut self, spec: &SparseHeapSpec) -> Answer<Handle> {
        let mut descriptor =
            HeapDescriptor::new(HeapType::Sparse, spec.size, ResourceOptions::private());
        descriptor.sparse_page_size = spec.page;
        let heap = self
            .device
            .new_heap(&descriptor)
            .map_err(Failure::from_error)?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<HeapFacts> {
        let heap = self.held_heap(heap)?;
        Ok(HeapFacts {
            heap_type: heap.heap_type().raw(),
            size: heap.size(),
            used_size: heap.used_size(),
        })
    }

    fn heap_texture(&mut self, heap: Handle, spec: &SparseTextureSpec) -> Answer<Handle> {
        let descriptor = sparse_texture_descriptor(spec);
        let texture = self
            .held_heap(heap)?
            .new_texture(&descriptor)
            .map_err(Failure::from_error)?;
        self.textures.push(texture);
        Ok(Handle(self.textures.len() as u32 - 1))
    }

    fn sparse_texture_facts(&mut self, texture: Handle) -> Answer<SparseTextureFacts> {
        let texture = self.held_texture(texture)?;
        Ok(SparseTextureFacts {
            is_sparse: texture.is_sparse(),
            // Metal answers both getters of every texture; the runtime answers `None` for one
            // that is not sparse, which is the same fact in the two APIs' spellings, and the
            // lane's own reading of it. A case asks them only of a sparse texture.
            first_mipmap_in_tail: texture.first_mipmap_in_tail().unwrap_or(0),
            tail_size_in_bytes: texture.tail_size_in_bytes().unwrap_or(0),
        })
    }

    fn heap(&mut self, spec: &HeapSpec) -> Answer<Handle> {
        if let Some(why) = spec.is_well_formed() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let descriptor = HeapDescriptor::new(
            spec.heap_type,
            spec.size,
            ResourceOptions::new(storage_mode_of(spec.storage), spec.cache, spec.hazard),
        );
        let heap = self
            .device
            .new_heap(&descriptor)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.heaps, heap))
    }

    fn heap_own_facts(&mut self, heap: Handle) -> Answer<HeapOwnFacts> {
        let registry = self.device.registry_id();
        let heap = self.held_heap(heap)?;
        Ok(HeapOwnFacts {
            heap_type: heap.heap_type().raw(),
            size: heap.size(),
            used_size: heap.used_size(),
            current_allocated_size: heap.current_allocated_size(),
            storage_mode: heap.storage_mode().raw(),
            cpu_cache_mode: heap.cpu_cache_mode().raw(),
            hazard_tracking_mode: heap.hazard_tracking_mode().raw(),
            resource_options: heap.resource_options().raw(),
            label: heap.label(),
            own_device: heap.device().registry_id() == registry,
        })
    }

    fn heap_max_available_size(&mut self, heap: Handle, alignment: u64) -> Answer<u64> {
        self.held_heap(heap)?
            .max_available_size(alignment)
            .map_err(Failure::from_error)
    }

    fn set_heap_label(&mut self, heap: Handle, label: &str) -> Answer<()> {
        self.held_heap(heap)?.set_label(label);
        Ok(())
    }

    fn heap_purgeable_state(&mut self, heap: Handle, state: PurgeableState) -> Answer<u64> {
        self.held_heap(heap)?
            .set_purgeable_state(state)
            .map(|was| was.raw())
            .map_err(Failure::from_error)
    }

    fn heap_buffer(&mut self, heap: Handle, spec: &HeapBufferSpec) -> Answer<Handle> {
        let held = self.held_heap(heap)?;
        // The options are the heap's own, which both APIs require: a buffer placed with any other
        // is refused `Invalid`, which is this tree's reading of "Metal's validation stops it".
        let options = held.resource_options();
        let buffer = match spec.offset {
            None => held.new_buffer(spec.length, options),
            Some(offset) => held.new_buffer_at_offset(spec.length, options, offset),
        }
        .map_err(Failure::from_error)?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn resource_facts(
        &mut self,
        of: QueriedResource,
        heap: Option<Handle>,
    ) -> Answer<ResourceFacts> {
        let registry = self.device.registry_id();
        // The heap the case placed the resource in, resolved before the resource is borrowed: the
        // comparison is a relation between two objects the case named, not an address.
        //
        // By `object_id` and not by pointer. The runtime's `Heap` is a handle each resource keeps a
        // clone of, so a pointer comparison answers `false` for a buffer that did answer the heap
        // it was placed in -- which is what the first run of this family recorded. `object_id` is
        // the runtime's own identity for exactly this question; the Metal side compares its two
        // objects as Metal makes them, one object per heap.
        let placed_in: Option<metal2vulkan_runtime::ObjectId> = match heap {
            None => None,
            Some(heap) => Some(self.held_heap(heap)?.object_id()),
        };
        // Each of the four modes is read through **its own getter**, never derived from
        // `resourceOptions`: they are four Metal properties, and the suite's business is what each
        // one answers. Deriving three of them from the fourth was this family's own first bug --
        // it read the runtime's unresolved option word where Metal's leg read the resolved
        // `hazardTrackingMode`, and reported a product difference at a site where there was none.
        match of {
            QueriedResource::Buffer(handle) => {
                let buffer = self.held(handle)?;
                let answered = buffer.heap().map(Heap::object_id);
                Ok(ResourceFacts {
                    label: buffer.label(),
                    own_device: buffer.device().registry_id() == registry,
                    storage_mode: buffer.storage_mode().raw(),
                    cpu_cache_mode: buffer.cpu_cache_mode().raw(),
                    hazard_tracking_mode: buffer.hazard_tracking_mode().raw(),
                    resource_options: buffer.resource_options().raw(),
                    in_a_heap: answered.is_some(),
                    expected_heap: answered.is_some() && answered == placed_in,
                    heap_offset: buffer.heap_offset(),
                    aliasable: buffer.is_aliasable(),
                })
            }
            QueriedResource::Texture(handle) => {
                let texture = self.held_texture(handle)?;
                let answered = texture.heap().map(Heap::object_id);
                Ok(ResourceFacts {
                    label: texture.label(),
                    own_device: texture.device().registry_id() == registry,
                    storage_mode: texture.storage_mode().raw(),
                    cpu_cache_mode: texture.cpu_cache_mode().raw(),
                    hazard_tracking_mode: texture.hazard_tracking_mode().raw(),
                    resource_options: texture.resource_options().raw(),
                    in_a_heap: answered.is_some(),
                    expected_heap: answered.is_some() && answered == placed_in,
                    heap_offset: texture.heap_offset(),
                    aliasable: texture.is_aliasable(),
                })
            }
        }
    }

    fn set_resource_label(&mut self, of: QueriedResource, label: &str) -> Answer<()> {
        match of {
            QueriedResource::Buffer(handle) => self.held(handle)?.set_label(label),
            QueriedResource::Texture(handle) => self.held_texture(handle)?.set_label(label),
        }
        Ok(())
    }

    fn make_resource_aliasable(&mut self, of: QueriedResource) -> Answer<()> {
        match of {
            QueriedResource::Buffer(handle) => self.held(handle)?.make_aliasable(),
            QueriedResource::Texture(handle) => self.held_texture(handle)?.make_aliasable(),
        }
        .map_err(Failure::from_error)
    }

    fn resource_purgeable_state(
        &mut self,
        of: QueriedResource,
        state: PurgeableState,
    ) -> Answer<u64> {
        match of {
            QueriedResource::Buffer(handle) => self.held(handle)?.set_purgeable_state(state),
            QueriedResource::Texture(handle) => {
                self.held_texture(handle)?.set_purgeable_state(state)
            }
        }
        .map(|was| was.raw())
        .map_err(Failure::from_error)
    }

    fn texture_view(&mut self, of: Handle, request: &ViewRequest) -> Answer<Handle> {
        let texture = self.held_texture(of)?;
        let levels = request.first_level..request.first_level + request.level_count;
        let slices = request.first_slice..request.first_slice + request.slice_count;
        let view = match request.call {
            ViewCall::PixelFormat => texture.new_texture_view(request.format),
            ViewCall::LevelsSlices => texture.new_texture_view_with_levels_slices(
                request.format,
                request.texture_type,
                levels,
                slices,
            ),
            // The runtime has one call for the two Metal selectors that state a swizzle: an
            // `MTLTextureViewDescriptor` holds exactly the five values the five-argument selector
            // takes, so the runtime's leg answers the same for both and the Metal leg is where the
            // two selectors part.
            ViewCall::Swizzled | ViewCall::Descriptor => texture.new_texture_view_with_swizzle(
                request.format,
                request.texture_type,
                levels,
                slices,
                request.swizzle,
            ),
        }
        .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, view))
    }

    fn texture_view_facts(&mut self, texture: Handle, parent: Option<Handle>) -> Answer<ViewFacts> {
        // By `object_id` and not by pointer, for the reason [`Side::resource_facts`]'s heap
        // comparison gives: the runtime's `Texture` is a handle a view keeps a clone of.
        let of_parent: Option<metal2vulkan_runtime::ObjectId> = match parent {
            None => None,
            Some(parent) => Some(self.held_texture(parent)?.object_id()),
        };
        let texture = self.held_texture(texture)?;
        let answered = texture.parent_texture().map(Texture::object_id);
        Ok(ViewFacts {
            has_parent: answered.is_some(),
            expected_parent: answered.is_some() && answered == of_parent,
            parent_relative_level: texture.parent_relative_level(),
            parent_relative_slice: texture.parent_relative_slice(),
            swizzle: crate::texture_views::swizzle_raw(texture.swizzle()),
            shareable: texture.is_shareable(),
            texture_type: u64::from(texture.texture_type().as_raw()),
            pixel_format: u64::from(texture.pixel_format().as_raw()),
            mipmap_level_count: texture.mipmap_level_count(),
            array_length: texture.array_length(),
        })
    }

    fn shared_texture(&mut self, query: &HeapTextureQuery) -> Answer<Handle> {
        let texture = self
            .device
            .new_shared_texture(&queried_descriptor(query))
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    fn shared_texture_handle(&mut self, texture: Handle) -> Answer<SharedHandleFacts> {
        let device = self.device;
        let registry = device.registry_id();
        let handle = self
            .held_texture(texture)?
            .new_shared_texture_handle()
            .map_err(Failure::from_error)?;
        let own_device = handle.device().registry_id() == registry;
        let opened = device
            .new_shared_texture_with_handle(&handle, metal2vulkan_runtime::IdPlacement::Any)
            .map_err(Failure::from_error)?;
        let facts = SharedHandleFacts {
            own_device,
            opens: true,
            opened_shareable: opened.is_shareable(),
            opened_width: opened.width(),
            opened_height: opened.height(),
            opened_levels: opened.mipmap_level_count(),
        };
        // Kept as the Metal side keeps it: a texture opened from a handle and dropped here would
        // be a texture over memory a later row could still be asked about.
        keep(&mut self.textures, opened);
        Ok(facts)
    }

    fn sparse_mapping(&mut self, program: &SparseMappingProgram) -> Answer<SparseMapped> {
        // The venue's own tile size, against the grid the program's regions are stated in. A
        // disagreement stops the program **before anything is encoded**: a region in tiles means
        // nothing without the grid it was written for.
        for spec in &program.textures {
            let query = SparseTileQuery {
                texture_type: spec.texture_type,
                format: spec.format,
                samples: spec.samples,
                page: Some(program.heap.page),
            };
            let held = self.sparse_tile_size(&query)?;
            let stated = program.tile.tile;
            if [held.width, held.height, held.depth] != stated {
                return Err(Failure::named(
                    "TileSize",
                    format!(
                        "{} tiles {}x{}x{} where the program states {}x{}x{}",
                        spec.site(),
                        held.width,
                        held.height,
                        held.depth,
                        stated[0],
                        stated[1],
                        stated[2]
                    ),
                ));
            }
        }
        let heap_handle = self.sparse_heap(&program.heap)?;
        let heap = self.held_heap(heap_handle)?.clone();
        let mut textures = Vec::new();
        for spec in &program.textures {
            let made = self.heap_texture(heap_handle, spec)?;
            textures.push(self.held_texture(made)?.clone());
        }
        let mut buffers = Vec::new();
        for bytes in &program.buffers {
            let made = self.buffer(bytes)?;
            buffers.push(self.held(made)?.clone());
        }
        let fences: Vec<Fence> = (0..program.fences)
            .map(|_| self.device.new_fence())
            .collect();
        let tails: Vec<(u64, u64)> = textures
            .iter()
            .map(|texture| {
                (
                    texture.first_mipmap_in_tail().unwrap_or(0),
                    texture.tail_size_in_bytes().unwrap_or(0),
                )
            })
            .collect();
        // A scratch buffer for each fill and each read, in the order the steps name them, so a
        // read's bytes are the side's own and never one of the program's.
        let mut scratch: Vec<Handle> = Vec::new();
        for step in &program.steps {
            let bytes = match step {
                MappingStep::FillRegion { region, value, .. } => {
                    let (_, length) = program.tile.bytes_of(region);
                    vec![*value; length as usize]
                }
                MappingStep::ReadRegion { region, .. } => {
                    let (_, length) = program.tile.bytes_of(region);
                    vec![READ_GUARD; length as usize]
                }
                _ => continue,
            };
            scratch.push(self.buffer(&bytes)?);
        }
        let held_scratch: Vec<Buffer> = scratch
            .iter()
            .map(|handle| self.held(*handle).cloned())
            .collect::<Answer<Vec<Buffer>>>()?;
        let queue = self.queue()?.clone();
        let mut run = Mapping {
            queue,
            command_buffer: None,
            blit: None,
            state: None,
        };
        let texture_at = |at: usize| -> Answer<&Texture> {
            textures
                .get(at)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {at}")))
        };
        let mut used_size = vec![heap.used_size()];
        let mut refused: Vec<Option<Failure>> = Vec::with_capacity(program.steps.len());
        let mut statuses = Vec::new();
        let mut reads = Vec::new();
        let mut pending: Vec<(usize, usize)> = Vec::new();
        let mut scratch_at = 0usize;
        for step in &program.steps {
            let answered: Answer<Committed> =
                (|| match step {
                    MappingStep::Commit => {
                        run.close_encoders()?;
                        match run.command_buffer.is_some() {
                            true => Ok(Committed::Yes),
                            false => Err(Failure::named("NoCommandBuffer", "a commit of nothing")),
                        }
                    }
                    MappingStep::NextEncoder => {
                        run.close_encoders()?;
                        Ok(Committed::No)
                    }
                    MappingStep::FillRegion {
                        texture,
                        level,
                        slice,
                        region,
                        ..
                    } => {
                        let held = texture_at(*texture)?.clone();
                        let source = held_scratch[scratch_at].clone();
                        let (per_row, bytes) = program.tile.bytes_of(region);
                        let (origin, extent) = program.tile.texels_of(region);
                        scratch_at += 1;
                        run.blit()?
                            .copy_from_buffer_to_texture(
                                &source,
                                0,
                                per_row,
                                bytes,
                                long_size(extent),
                                &held,
                                *slice,
                                *level,
                                long_origin(origin),
                                BlitOption::None,
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::ReadRegion {
                        texture,
                        level,
                        slice,
                        region,
                    } => {
                        let held = texture_at(*texture)?.clone();
                        let into = held_scratch[scratch_at].clone();
                        let (per_row, bytes) = program.tile.bytes_of(region);
                        let (origin, extent) = program.tile.texels_of(region);
                        pending.push((scratch_at, bytes as usize));
                        scratch_at += 1;
                        run.blit()?
                            .copy_from_texture_to_buffer(
                                &held,
                                *slice,
                                *level,
                                long_origin(origin),
                                long_size(extent),
                                &into,
                                0,
                                per_row,
                                bytes,
                                BlitOption::None,
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::One {
                        texture,
                        mode,
                        region,
                        level,
                        slice,
                    } => {
                        let held = texture_at(*texture)?.clone();
                        run.state()?
                            .update_texture_mapping(
                                &held,
                                tile_mapping(*mode),
                                tile_region(region),
                                *level,
                                *slice,
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::Many {
                        texture,
                        mode,
                        regions,
                    } => {
                        let held = texture_at(*texture)?.clone();
                        let rects: Vec<Region> = regions
                            .iter()
                            .map(|(region, _, _)| tile_region(region))
                            .collect();
                        let levels: Vec<u64> = regions.iter().map(|(_, level, _)| *level).collect();
                        let slices: Vec<u64> = regions.iter().map(|(_, _, slice)| *slice).collect();
                        run.state()?
                            .update_texture_mappings(
                                &held,
                                tile_mapping(*mode),
                                &rects,
                                &levels,
                                &slices,
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::Indirect {
                        texture,
                        mode,
                        buffer,
                        offset,
                    } => {
                        let held = texture_at(*texture)?.clone();
                        let arguments = buffers.get(*buffer).cloned().ok_or_else(|| {
                            Failure::named("NoSuchHandle", format!("buffer {buffer}"))
                        })?;
                        run.state()?
                            .update_texture_mapping_indirect(
                                &held,
                                tile_mapping(*mode),
                                &arguments,
                                *offset,
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::Move {
                        from,
                        from_slice,
                        from_level,
                        from_origin,
                        size,
                        to,
                        to_slice,
                        to_level,
                        to_origin,
                    } => {
                        let source = texture_at(*from)?.clone();
                        let destination = texture_at(*to)?.clone();
                        run.state()?
                            .move_texture_mappings(
                                &source,
                                *from_slice,
                                *from_level,
                                long_origin(*from_origin),
                                long_size(*size),
                                &destination,
                                *to_slice,
                                *to_level,
                                long_origin(*to_origin),
                            )
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::UpdateFence { fence } => {
                        let held = fences.get(*fence).cloned().ok_or_else(|| {
                            Failure::named("NoSuchHandle", format!("fence {fence}"))
                        })?;
                        run.state()?
                            .update_fence(&held)
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                    MappingStep::WaitForFence { fence } => {
                        let held = fences.get(*fence).cloned().ok_or_else(|| {
                            Failure::named("NoSuchHandle", format!("fence {fence}"))
                        })?;
                        run.state()?
                            .wait_for_fence(&held)
                            .map_err(|error| Failure::of(&error))?;
                        Ok(Committed::No)
                    }
                })();
            let committed = matches!(answered, Ok(Committed::Yes));
            refused.push(answered.err());
            if !committed {
                continue;
            }
            let status = match run.command_buffer.take() {
                Some(buffer) => {
                    buffer.commit().map_err(|error| Failure::of(&error))?;
                    match buffer.wait_until_completed_for(WAIT) {
                        Ok(()) => status_value(buffer.status()),
                        Err(_) => Value::Nothing(Negative::NeverReturned),
                    }
                }
                None => Value::Absent,
            };
            statuses.push(status);
            used_size.push(heap.used_size());
            for &(slot, length) in &pending {
                reads.push(self.read_buffer(scratch[slot], 0, length)?);
            }
            pending.clear();
        }
        // A program that left an encoder open closes it: its steps' answers stand, and the case's
        // own guard holds every program to ending on a commit.
        let _ = run.close_encoders();
        Ok(SparseMapped {
            refused,
            used_size,
            tails,
            statuses,
            reads,
        })
    }

    fn command_buffer_facts(&mut self) -> Answer<CommandBufferFacts> {
        let scratch = self.buffer(&[0u8; TIMED_BYTES])?;
        let target = self.held(scratch)?.clone();
        let queue = self.queue()?.clone();
        // Two queues with distinct labels, a command buffer on each: whether each command
        // buffer's `command_queue` answers **its own** queue's label. Neither API compares two
        // queues for identity publicly, so this is the strongest fact both legs compute alike.
        let mut queue_label_is_its_own = [false; 2];
        for (at, label) in QUEUE_LABELS.iter().enumerate() {
            let labelled: CommandQueue = self
                .device
                .new_command_queue()
                .map_err(Failure::from_error)?;
            labelled.set_label(label);
            let buffer: CommandBuffer = labelled.command_buffer().map_err(Failure::from_error)?;
            let answered = buffer.command_queue();
            queue_label_is_its_own[at] = answered.label().as_deref() == Some(*label);
        }
        let held: CommandBuffer = queue.command_buffer().map_err(Failure::from_error)?;
        let device_is_the_queues = held.device().registry_id() == queue.device().registry_id();
        let kernel_before = [held.kernel_start_time(), held.kernel_end_time()];
        let gpu_before = gpu_times(&held);
        // One blit of a few bytes of the side's own buffer: enough work that the GPU ran, and no
        // shader, draw or dispatch at all.
        let mut encoder = held.blit_command_encoder().map_err(Failure::from_error)?;
        encoder
            .fill_buffer(&target, 0..TIMED_BYTES as u64, 0xa5)
            .map_err(Failure::from_error)?;
        encoder.end_encoding().map_err(Failure::from_error)?;
        held.commit().map_err(Failure::from_error)?;
        let status = match held.wait_until_completed_for(WAIT) {
            Ok(()) => status_value(held.status()),
            Err(_) => Value::Nothing(Negative::NeverReturned),
        };
        let kernel_after = [held.kernel_start_time(), held.kernel_end_time()];
        let gpu_after = gpu_times(&held);

        // A command buffer of its own for the encoder, so the timings above are of a command
        // buffer that only blitted and this one only opens an encoder.
        let state_buffer: CommandBuffer = queue.command_buffer().map_err(Failure::from_error)?;
        let opened = state_buffer.resource_state_command_encoder();
        let resource_state_encoder_opened = opened.is_ok();
        // Ended with nothing encoded into it: it maps nothing and takes no heap memory.
        if let Ok(encoder) = opened {
            encoder.end_encoding().map_err(Failure::from_error)?;
        }
        state_buffer.commit().map_err(Failure::from_error)?;
        let state_encoder_status = match state_buffer.wait_until_completed_for(WAIT) {
            Ok(()) => status_value(state_buffer.status()),
            Err(_) => Value::Nothing(Negative::NeverReturned),
        };
        let gpu = match (gpu_before, gpu_after) {
            (Ok(before), Ok(after)) => Ok((before, after)),
            (Err(failure), _) | (_, Err(failure)) => Err(failure),
        };
        Ok(CommandBufferFacts::new(
            device_is_the_queues,
            queue_label_is_its_own,
            kernel_before,
            kernel_after,
            gpu,
            resource_state_encoder_opened,
            status,
            state_encoder_status,
        ))
    }
}

/// The storage mode a [`Storage`] names, as the runtime's own enumeration spells it.
///
/// [`resource_options`] already maps the four to whole option words; a heap states its cache and
/// hazard modes too, so it needs the storage field alone.
fn storage_mode_of(storage: Storage) -> metal2vulkan_runtime::StorageMode {
    match storage {
        Storage::Shared => metal2vulkan_runtime::StorageMode::Shared,
        Storage::Managed => metal2vulkan_runtime::StorageMode::Managed,
        Storage::Private => metal2vulkan_runtime::StorageMode::Private,
        Storage::Memoryless => metal2vulkan_runtime::StorageMode::Memoryless,
    }
}

/// The runtime's descriptor for a case's sparse texture: the case's vocabulary is Metal's own, so
/// every field reaches the descriptor unchanged.
/// The runtime's texture descriptor for the one a case states: every field of
/// [`TextureRequest`](crate::texture::TextureRequest) reaches it.
///
/// A function of its own, and not inlined in the call, so that
/// `both_sides_state_every_field_of_a_texture_request` can hold it to that field by field with no
/// device — the swizzle most of all. A swizzle this side dropped while the Metal side set it would
/// leave every swizzle case passing for the wrong reason: both legs would be reading the texture's
/// unswizzled texels, and the comparison would be of two answers to a question neither leg was
/// asked.
///
/// [`crate::texture::Over`] is **not** here: it is which object the call is sent to, not a field of
/// the descriptor, and both sides send it in their own API's spelling.
pub(crate) fn texture_descriptor(
    request: &crate::texture::TextureRequest<'_>,
    format: PixelFormat,
) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(
        format,
        request.size[0].into(),
        request.size[1].into(),
        false,
    );
    descriptor.texture_type = request.texture_type;
    descriptor.depth = request.size[2].max(1).into();
    descriptor.mipmap_level_count = request.levels.max(1).into();
    descriptor.array_length = request.slices.max(1).into();
    descriptor.usage = request.usage;
    descriptor.resource_options = if request.private {
        ResourceOptions::private()
    } else {
        ResourceOptions::shared()
    };
    descriptor.allow_gpu_optimized_contents = request.gpu_optimized;
    descriptor.swizzle = request.swizzle;
    descriptor
}

fn sparse_texture_descriptor(spec: &SparseTextureSpec) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = spec.texture_type;
    descriptor.pixel_format = spec.format;
    descriptor.width = u64::from(spec.size[0]);
    descriptor.height = u64::from(spec.size[1]);
    descriptor.depth = u64::from(spec.size[2]);
    descriptor.mipmap_level_count = u64::from(spec.levels);
    descriptor.sample_count = u64::from(spec.samples);
    descriptor.array_length = u64::from(spec.array_length);
    descriptor.usage = if spec.render_target {
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
    } else {
        TextureUsage::SHADER_READ
    };
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// The runtime's color attachment descriptor for one the case states: every field of
/// [`ColorState`] reaches it, because the case's vocabulary is Metal's own and the two sides must
/// give the two implementations the same descriptor.
///
/// It is a function of its own, and not inlined in the call, so that
/// `the_runtime_side_forwards_every_field_of_a_color_state` can hold it to that field by field
/// with no device. A field the Metal side sets and this one drops is a harness bug that reads as
/// our implementation's: the write mask was dropped here, which made the tile-store family's
/// masked draw write every channel on our side while Metal's wrote none.
pub(crate) fn color_descriptor(color: &ColorState) -> ColorAttachmentDescriptor {
    let mut made = ColorAttachmentDescriptor::new(color.format);
    made.write_mask = color.write_mask;
    if let Some(blend) = &color.blend {
        made.blending_enabled = true;
        made.source_rgb_blend_factor = blend.source_rgb;
        made.destination_rgb_blend_factor = blend.destination_rgb;
        made.rgb_blend_operation = blend.rgb;
        made.source_alpha_blend_factor = blend.source_alpha;
        made.destination_alpha_blend_factor = blend.destination_alpha;
        made.alpha_blend_operation = blend.alpha;
    }
    made
}

/// The runtime's descriptor for a queried one: the case's vocabulary is Metal's own, so every
/// field reaches the descriptor unchanged.
fn queried_descriptor(query: &HeapTextureQuery) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = query.texture_type;
    descriptor.pixel_format = query.format;
    descriptor.width = u64::from(query.size[0]);
    descriptor.height = u64::from(query.size[1]);
    descriptor.depth = u64::from(query.size[2]);
    descriptor.mipmap_level_count = u64::from(query.levels);
    descriptor.sample_count = u64::from(query.samples);
    descriptor.array_length = u64::from(query.array_length);
    descriptor.usage = query.usage;
    descriptor.resource_options = resource_options(query.storage);
    descriptor.allow_gpu_optimized_contents = query.gpu_optimized;
    descriptor
}

/// The resource options a [`Storage`] names: the runtime side's one spelling of the three modes,
/// which [`crate::runtime_work`] reads too, so a case's storage mode cannot mean one thing to a
/// queried descriptor and another to a program's buffer.
pub fn resource_options(storage: Storage) -> ResourceOptions {
    match storage {
        Storage::Shared => ResourceOptions::shared(),
        Storage::Private => ResourceOptions::private(),
        Storage::Managed => ResourceOptions::new(
            metal2vulkan_runtime::StorageMode::Managed,
            metal2vulkan_runtime::CpuCacheMode::DefaultCache,
            metal2vulkan_runtime::HazardTrackingMode::Default,
        ),
        Storage::Memoryless => ResourceOptions::new(
            metal2vulkan_runtime::StorageMode::Memoryless,
            metal2vulkan_runtime::CpuCacheMode::DefaultCache,
            metal2vulkan_runtime::HazardTrackingMode::Default,
        ),
    }
}

/// Encode one render pass of a program into `command_buffer`, answering the first refusal it met.
impl RuntimeSide<'_> {
    /// The `RenderPipelineDescriptor` and the creation options `spec` describes.
    ///
    /// Its own method rather than part of a creation, because the same descriptor is handed to two
    /// creations: a case of [`crate::pipeline_handlers`] takes one descriptor through the
    /// counterpart of Metal's `completionHandler:` call and through its synchronous sibling, and
    /// the two are only comparable where the value they are given is the same. The label travels
    /// in the options because that is where the runtime takes
    /// `MTLRenderPipelineDescriptor.label` from.
    fn our_render_descriptor(
        &self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<(RenderPipelineDescriptor, PipelineOptions)> {
        let stage = |stage: &StageFunction<'_>| -> Answer<StageSource> {
            let named = stage.library.unwrap_or(library);
            let held = self
                .libraries
                .get(named.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", named.0)))?;
            Ok(StageSource::Function(
                held.new_function_with_name(stage.entry)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage(&spec.vertex)?);
        descriptor.fragment = spec.fragment.as_ref().map(&stage).transpose()?;
        let apart = pipeline_fields(spec)?;
        descriptor.color_attachments = apart.color_attachments;
        descriptor.depth_format = apart.depth_format;
        descriptor.stencil_format = apart.stencil_format;
        descriptor.raster_sample_count = apart.raster_sample_count;
        descriptor.rasterization_enabled = apart.rasterization_enabled;
        descriptor.vertex_descriptor = apart.vertex_descriptor;
        descriptor.support_indirect_command_buffers = apart.support_indirect_command_buffers;
        descriptor.tessellation = apart.tessellation;
        descriptor.alpha_to_coverage = apart.alpha_to_coverage;
        descriptor.alpha_to_one = apart.alpha_to_one;
        descriptor.input_primitive_topology = apart.input_primitive_topology;
        let mut options = PipelineOptions::default();
        options.label = apart.label;
        Ok((descriptor, options))
    }

    /// The `ComputePipelineDescriptor` and the creation options `spec` describes.
    ///
    /// Its own method for the reason [`RuntimeSide::our_render_descriptor`]'s is.
    fn our_compute_descriptor(
        &self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<(
        metal2vulkan::pipeline::ComputePipelineDescriptor,
        PipelineOptions,
    )> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(spec.entry)
            .map_err(|error| Failure::of(&error))?;
        let mut descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(
            StageSource::Function(function.function_descriptor()),
        );
        descriptor.support_indirect_command_buffers = spec.support_indirect_command_buffers;
        descriptor.required_threads_per_threadgroup = spec.required_threads_per_threadgroup;
        let mut options = PipelineOptions::default();
        options.label = spec.label.map(str::to_owned);
        Ok((descriptor, options))
    }

    /// The `TileRenderPipelineDescriptor` `spec` describes.
    ///
    /// Its own method for the reason [`RuntimeSide::our_render_descriptor`]'s is. A tile
    /// descriptor carries no label, so there are no options to build beside it.
    /// The mesh descriptor `spec` describes, whose attachments are the formats the case names.
    fn mesh_reflection_descriptor(
        &self,
        library: Handle,
        spec: &MeshReflectionSpec<'_>,
    ) -> Answer<MeshPipelineDescriptor> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let stage = |name: &str| -> Answer<StageSource> {
            Ok(StageSource::Function(
                library
                    .new_function_with_name(name)
                    .map_err(Failure::from_error)?
                    .function_descriptor(),
            ))
        };
        let mut descriptor = MeshPipelineDescriptor::new(stage(spec.mesh)?);
        descriptor.object = spec.object.map(stage).transpose()?;
        descriptor.fragment = Some(stage(spec.fragment)?);
        for (slot, format) in descriptor
            .color_attachments
            .iter_mut()
            .zip(spec.attachments)
        {
            *slot = Some(ColorAttachmentDescriptor::new(*format));
        }
        Ok(descriptor)
    }

    fn our_tile_descriptor(
        &self,
        library: Handle,
        spec: &TilePipelineSpec<'_>,
    ) -> Answer<TileRenderPipelineDescriptor> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .new_function_with_name(spec.entry)
            .map_err(Failure::from_error)?;
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
            function.function_descriptor(),
        ));
        for (slot, format) in descriptor
            .color_attachment_formats
            .iter_mut()
            .zip(spec.attachments)
        {
            *slot = Some(*format);
        }
        descriptor.max_total_threads_per_threadgroup = spec.max_total_threads_per_threadgroup;
        descriptor.threadgroup_size_matches_tile_size = spec.threadgroup_size_matches_tile_size;
        Ok(descriptor)
    }

    /// One pipeline descriptor through the counterpart of Metal's `completionHandler:` form and
    /// through that form's synchronous sibling ([`crate::pipeline_handlers`]).
    ///
    /// **The runtime has no asynchronous creation form.** The coverage map names each handler
    /// form's counterpart as the synchronous call "the handler receives what the call returns", so
    /// this side takes that call for both paths and answers
    /// [`Arrival::Answered`](crate::pipeline_handlers::Arrival::Answered): a call that returned is
    /// a delivery. The module's prose says so outright, and the absence of an asynchronous form is
    /// a reported gap rather than a comparison. Nothing here waits on anything.
    fn our_both_paths(&mut self, library: Handle, form: &PipelineForm<'_>) -> Answer<BothPaths> {
        let registry = self.device.registry_id();
        // Both paths are the one call, taken twice over the one descriptor. Written as a closure
        // the two calls below share, so neither path can drift from the other inside this side.
        let delivered = match form {
            PipelineForm::Render {
                spec,
                options: None,
            } => {
                let (descriptor, options) = self.our_render_descriptor(library, spec)?;
                let made = || match self.device.new_render_pipeline_state(&descriptor, &options) {
                    Ok(state) => Delivered::made(StateFacts::Render(our_render_state_facts(
                        &state,
                        registry,
                        HANDLER_IMAGEBLOCK,
                    ))),
                    Err(_) => Delivered::refused(false),
                };
                (made(), made())
            }
            PipelineForm::Render {
                spec,
                options: Some(bits),
            } => {
                let (descriptor, mut options) = self.our_render_descriptor(library, spec)?;
                options.options = PipelineOption::from_raw(*bits);
                let made = || match self
                    .device
                    .new_render_pipeline_state_with_reflection(&descriptor, &options)
                {
                    Ok((state, reflection)) => Delivered::reflected(
                        StateFacts::Render(our_render_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (made(), made())
            }
            PipelineForm::ComputeFunction {
                entry,
                options: None,
            } => {
                let function = self.our_function(library, entry)?;
                let options = PipelineOptions::default();
                let made = || match self
                    .device
                    .new_compute_pipeline_state_with_function(&function, &options)
                {
                    Ok(state) => Delivered::made(StateFacts::Compute(our_compute_state_facts(
                        &state,
                        registry,
                        HANDLER_IMAGEBLOCK,
                    ))),
                    Err(_) => Delivered::refused(false),
                };
                (made(), made())
            }
            PipelineForm::ComputeFunction {
                entry,
                options: Some(bits),
            } => {
                let function = self.our_function(library, entry)?;
                let mut options = PipelineOptions::default();
                options.options = PipelineOption::from_raw(*bits);
                let made = || match self
                    .device
                    .new_compute_pipeline_state_with_function_and_reflection(&function, &options)
                {
                    Ok((state, reflection)) => Delivered::reflected(
                        StateFacts::Compute(our_compute_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (made(), made())
            }
            PipelineForm::ComputeDescriptor { spec, options } => {
                let (descriptor, mut built) = self.our_compute_descriptor(library, spec)?;
                built.options = PipelineOption::from_raw(*options);
                let made = || match self
                    .device
                    .new_compute_pipeline_state_with_reflection(&descriptor, &built)
                {
                    Ok((state, reflection)) => Delivered::reflected(
                        StateFacts::Compute(our_compute_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (made(), made())
            }
            PipelineForm::Tile { spec, options } => {
                let descriptor = self.our_tile_descriptor(library, spec)?;
                let mut built = PipelineOptions::default();
                built.options = PipelineOption::from_raw(*options);
                let made = || match self
                    .device
                    .new_tile_render_pipeline_state_with_reflection(&descriptor, &built)
                {
                    Ok((state, reflection)) => Delivered::reflected(
                        StateFacts::Tile(our_tile_state_facts(&state)),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (made(), made())
            }
            PipelineForm::Mesh { functions, options } => {
                let descriptor = self.mesh_descriptor(library, functions)?;
                let mut built = PipelineOptions::default();
                built.options = PipelineOption::from_raw(*options);
                let made = || match self
                    .device
                    .new_mesh_render_pipeline_state_with_reflection(&descriptor, &built)
                {
                    Ok((state, reflection)) => Delivered::reflected(
                        StateFacts::Render(our_render_state_facts(
                            &state,
                            registry,
                            HANDLER_IMAGEBLOCK,
                        )),
                        reflection.is_some(),
                    ),
                    Err(_) => Delivered::refused(true),
                };
                (made(), made())
            }
        };
        let (through_the_counterpart, synchronous) = delivered;
        Ok(BothPaths::new(
            Handled::answered(through_the_counterpart),
            synchronous,
        ))
    }

    /// `newFunctionWithName:` of `entry` in the library `library` names.
    fn our_function(&self, library: Handle, entry: &str) -> Answer<metal2vulkan_runtime::Function> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        library
            .new_function_with_name(entry)
            .map_err(Failure::from_error)
    }

    /// The buffer standing for the surface behind the texture `target`.
    fn held_plane(&self, target: Handle) -> Answer<&Buffer> {
        self.planes
            .iter()
            .find(|(held, _)| *held == target)
            .map(|(_, buffer)| buffer)
            .ok_or_else(|| {
                Failure::named(
                    "NotOverASurface",
                    format!("texture {} lies over no surface plane", target.0),
                )
            })
    }

    /// A target as a texture over a buffer the case made, through the runtime's own
    /// `newTextureWithDescriptor:offset:bytesPerRow:`.
    ///
    /// The stride is refused before the call for the same reason the Metal side refuses it: a
    /// `bytesPerRow` below the target's own row ends Metal's process, so neither side may make
    /// such a texture and both must refuse the same shape, or the comparison would be of two
    /// different questions.
    fn buffer_backed_target(
        &mut self,
        spec: &TargetSpec,
        over: &crate::render::OverBuffer,
    ) -> Answer<Handle> {
        if let Some(why) = over.unsafe_stride() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        // A texture over a buffer holds the buffer's own bytes, so its storage is the buffer's.
        descriptor.resource_options = ResourceOptions::shared();
        let buffer = self.held(over.buffer)?.clone();
        let texture = buffer
            .new_texture(&descriptor, over.offset, over.bytes_per_row)
            .map_err(Failure::from_error)?;
        Ok(keep(&mut self.textures, texture))
    }

    /// A framebuffer-only rotated texture over a buffer standing for an IOSurface plane
    /// (`Buffer::new_texture_with_iosurface_plane`), whose bytes start as the plane's pad byte.
    ///
    /// The runtime's stand-in is a buffer rather than a surface, which is the whole reason a case
    /// of this family runs where there is no IOSurface at all. The **stride rule Metal asserts on**
    /// is checked here too ([`crate::surface::PlaneSpec::unsafe_stride`]): not because the runtime
    /// would end the process, but so the two sides refuse the same planes and a case cannot pass on
    /// one side by asking a question the other was never allowed to be asked.
    fn surface_target(
        &mut self,
        spec: &TargetSpec,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Handle> {
        if plane.unsafe_stride(spec.size) {
            return Err(Failure::named(
                "WouldEndMetalsProcess",
                format!(
                    "a plane of {} bytes a row behind a {} texel row",
                    plane.row_pitch,
                    spec.size[0] * plane.texel
                ),
            ));
        }
        let length = plane.length() as u64;
        let buffer = self
            .device
            .new_buffer(length, ResourceOptions::shared())
            .map_err(Failure::from_error)?;
        let contents = buffer.contents().ok_or_else(|| {
            Failure::named("NoContents", "the plane's buffer is not host-visible")
        })?;
        // SAFETY: the buffer was just made with `length` bytes of shared storage and no command
        // uses it yet.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), plane.pad, length as usize) };
        let mut descriptor = TextureDescriptor::texture_2d(
            spec.format,
            u64::from(spec.size[0]),
            u64::from(spec.size[1]),
            false,
        );
        let mut usage = TextureUsage::RENDER_TARGET;
        if spec.shader_read {
            usage |= TextureUsage::SHADER_READ;
        }
        if spec.shader_write {
            usage |= TextureUsage::SHADER_WRITE;
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::shared();
        descriptor.framebuffer_only = true;
        descriptor.rotation = u64::from(plane.rotation);
        let texture = buffer
            .new_texture_with_iosurface_plane(
                &descriptor,
                metal2vulkan_runtime::IoSurfacePlane::new(0, u64::from(plane.row_pitch), false),
            )
            .map_err(Failure::from_error)?;
        let handle = keep(&mut self.textures, texture);
        self.planes.push((handle, buffer));
        Ok(handle)
    }
}

/// The objects a render program's pieces reach, cloned out of the side's pools once so each piece
/// can find the case's objects while the command buffer is borrowed.
///
/// One value rather than one slice per pool: a program's pieces each need a different handful of
/// them, and a pool the vocabulary gains later would otherwise widen every signature on the way.
struct Pools {
    textures: Vec<Texture>,
    buffers: Vec<Buffer>,
    states: Vec<RenderPipelineState>,
    depth_stencil: Vec<DepthStencilState>,
    samplers: Vec<SamplerState>,
    computes: Vec<ComputePipelineState>,
    indirect: Vec<IndirectCommandBuffer>,
    heaps: Vec<Heap>,
}

impl Pools {
    /// The texture `handle` names.
    fn texture(&self, handle: Handle) -> Answer<&Texture> {
        self.textures
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The buffer `handle` names.
    fn buffer(&self, handle: Handle) -> Answer<&Buffer> {
        self.buffers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// The indirect command buffer `handle` names.
    fn indirect(&self, handle: Handle) -> Answer<&IndirectCommandBuffer> {
        self.indirect.get(handle.0 as usize).ok_or_else(|| {
            Failure::named("NoSuchHandle", format!("indirect commands {}", handle.0))
        })
    }
}

fn encode_pass(command_buffer: &CommandBuffer, pass: &RenderPass, pools: &Pools) -> Answer<()> {
    let Pools {
        textures,
        buffers,
        states,
        depth_stencil,
        samplers,
        indirect,
        heaps,
        ..
    } = pools;
    let sampler = |handle: Handle| -> Answer<&SamplerState> {
        samplers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    };
    let texture = |handle: Handle| -> Answer<&Texture> {
        textures
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    };
    let buffer = |handle: Handle| -> Answer<&Buffer> {
        buffers
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    };
    let heap = |handle: Handle| -> Answer<&Heap> {
        heaps
            .get(handle.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    };
    // A residency or a barrier step names an `MTLResource`, which is a buffer or a texture of the
    // case's own pools.
    let resource = |held: StepResource| -> Answer<ResourceRef<'_>> {
        Ok(match held {
            StepResource::Buffer(handle) => ResourceRef::Buffer(buffer(handle)?),
            StepResource::Texture(handle) => ResourceRef::Texture(texture(handle)?),
        })
    };
    let mut descriptor = RenderPassDescriptor::new();
    if let Some(size) = pass.size {
        descriptor.render_target_width = u64::from(size[0]);
        descriptor.render_target_height = u64::from(size[1]);
        descriptor.default_raster_sample_count = 1;
    }
    for (at, color) in pass.colors.iter().enumerate() {
        let Some(held) = color.texture else { continue };
        let attachment = descriptor
            .color_attachments
            .get_mut(at)
            .ok_or_else(|| Failure::named("NoSuchAttachment", format!("color {at}")))?;
        attachment.texture = Some(texture(held)?.clone());
        attachment.level = u64::from(color.place.level);
        attachment.slice = u64::from(color.place.slice);
        attachment.depth_plane = u64::from(color.place.depth_plane);
        attachment.resolve_level = u64::from(color.place.resolve_level);
        attachment.resolve_slice = u64::from(color.place.resolve_slice);
        attachment.resolve_depth_plane = u64::from(color.place.resolve_depth_plane);
        attachment.load_action = color.load;
        attachment.store_action = color.store;
        attachment.clear_color = ClearColor::new(
            color.clear[0],
            color.clear[1],
            color.clear[2],
            color.clear[3],
        );
        attachment.store_action_options = color.options;
        if let Some(resolve) = color.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(depth) = &pass.depth {
        let attachment = &mut descriptor.depth_attachment;
        attachment.texture = Some(texture(depth.texture)?.clone());
        attachment.level = u64::from(depth.place.level);
        attachment.slice = u64::from(depth.place.slice);
        attachment.depth_plane = u64::from(depth.place.depth_plane);
        attachment.resolve_level = u64::from(depth.place.resolve_level);
        attachment.resolve_slice = u64::from(depth.place.resolve_slice);
        attachment.resolve_depth_plane = u64::from(depth.place.resolve_depth_plane);
        attachment.load_action = depth.load;
        attachment.store_action = depth.store;
        attachment.clear_depth = depth.clear;
        attachment.depth_resolve_filter = depth.filter;
        attachment.store_action_options = depth.options;
        if let Some(resolve) = depth.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(stencil) = &pass.stencil {
        let attachment = &mut descriptor.stencil_attachment;
        attachment.texture = Some(texture(stencil.texture)?.clone());
        attachment.level = u64::from(stencil.place.level);
        attachment.slice = u64::from(stencil.place.slice);
        attachment.depth_plane = u64::from(stencil.place.depth_plane);
        attachment.resolve_level = u64::from(stencil.place.resolve_level);
        attachment.resolve_slice = u64::from(stencil.place.resolve_slice);
        attachment.resolve_depth_plane = u64::from(stencil.place.resolve_depth_plane);
        attachment.load_action = stencil.load;
        attachment.store_action = stencil.store;
        attachment.clear_stencil = stencil.clear;
        attachment.stencil_resolve_filter = stencil.filter;
        attachment.store_action_options = stencil.options;
        if let Some(resolve) = stencil.resolve {
            attachment.resolve_texture = Some(texture(resolve)?.clone());
        }
    }
    if let Some(held) = pass.visibility {
        descriptor.visibility_result_buffer = Some(buffer(held)?.clone());
    }
    // Written whatever it is, since 0 is Metal's own default and the field takes it.
    descriptor.render_target_array_length = u64::from(pass.array_length);
    // The runtime takes Metal's own positions where the case states none, and refuses what Metal
    // refuses, typed: no guard of the suite's stands between a case and that refusal, because the
    // refusal is what the case compares.
    descriptor.sample_positions = pass.positions.clone();
    let mut encoder = command_buffer
        .render_command_encoder(&descriptor)
        .map_err(Failure::from_error)?;
    for step in &pass.steps {
        match step {
            RenderStep::Pipeline(state) => {
                let state = states.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("render pipeline {}", state.0))
                })?;
                encoder
                    .set_render_pipeline_state(state)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::DepthStencilState(state) => {
                let state = depth_stencil.get(state.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                })?;
                encoder
                    .set_depth_stencil_state(state)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::StencilReference(value) => encoder.set_stencil_reference_value(*value),
            RenderStep::VertexBuffer {
                buffer: held,
                offset,
                index,
            } => encoder
                .set_vertex_buffer(Some(buffer(*held)?), *offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentBuffer {
                buffer: held,
                offset,
                index,
            } => encoder
                .set_fragment_buffer(Some(buffer(*held)?), *offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentTexture {
                texture: held,
                index,
            } => encoder
                .set_fragment_texture(Some(texture(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentSampler {
                sampler: held,
                index,
            } => encoder
                .set_fragment_sampler_state(Some(sampler(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::VertexBytes { bytes, index } => encoder
                .set_vertex_bytes(bytes, *index)
                .map_err(Failure::from_error)?,
            RenderStep::Viewport(rect) => encoder
                .set_viewport(Viewport::new(
                    rect.origin[0],
                    rect.origin[1],
                    rect.size[0],
                    rect.size[1],
                    rect.depth[0],
                    rect.depth[1],
                ))
                .map_err(Failure::from_error)?,
            RenderStep::VisibilityResultMode { mode, offset } => encoder
                .set_visibility_result_mode(*mode, *offset)
                .map_err(Failure::from_error)?,
            RenderStep::BlendColor(channels) => {
                let [red, green, blue, alpha] = channels.0;
                encoder.set_blend_color(red, green, blue, alpha);
            }
            RenderStep::Draw {
                primitive,
                first,
                count,
                instances,
            } => encoder
                .draw_primitives(*primitive, *first, *count, *instances, 0)
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndexed {
                primitive,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    *index_type,
                    buffer(*indices)?,
                    *offset,
                    *instances,
                    *base_vertex,
                    *base_instance,
                )
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndirect {
                primitive,
                arguments,
                offset,
            } => encoder
                .draw_primitives_indirect(*primitive, buffer(*arguments)?, *offset)
                .map_err(Failure::from_error)?,
            // Metal's short draw selectors are the long ones with its own documented defaults, and
            // the runtime's API has one entry point a family rather than five. So our leg reaches
            // these four through that entry point with the defaults written out, which is what
            // makes the comparison a reading of Metal's defaults: a short form that behaved as
            // something other than one instance from base 0 would differ here.
            RenderStep::DrawPlain {
                primitive,
                first,
                count,
            } => encoder
                .draw_primitives(*primitive, *first, *count, 1, 0)
                .map_err(Failure::from_error)?,
            RenderStep::DrawInstanced {
                primitive,
                first,
                count,
                instances,
            } => encoder
                .draw_primitives(*primitive, *first, *count, *instances, 0)
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndexedPlain {
                primitive,
                count,
                index_type,
                indices,
                offset,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    *index_type,
                    buffer(*indices)?,
                    *offset,
                    1,
                    0,
                    0,
                )
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndexedInstanced {
                primitive,
                count,
                index_type,
                indices,
                offset,
                instances,
            } => encoder
                .draw_indexed_primitives(
                    *primitive,
                    *count,
                    *index_type,
                    buffer(*indices)?,
                    *offset,
                    *instances,
                    0,
                    0,
                )
                .map_err(Failure::from_error)?,
            RenderStep::DrawIndexedIndirect {
                primitive,
                index_type,
                indices,
                index_offset,
                arguments,
                offset,
            } => encoder
                .draw_indexed_primitives_indirect(
                    *primitive,
                    *index_type,
                    buffer(*indices)?,
                    *index_offset,
                    buffer(*arguments)?,
                    *offset,
                )
                .map_err(Failure::from_error)?,
            RenderStep::UseBuffer {
                buffer: held,
                write,
                stages: bits,
            } => encoder
                .use_resource(
                    ResourceRef::Buffer(buffer(*held)?),
                    commands_usage(*write),
                    stages(*bits)?,
                )
                .map_err(Failure::from_error)?,
            RenderStep::ExecuteCommands {
                commands,
                first,
                count,
            } => {
                let icb = indirect.get(commands.0 as usize).ok_or_else(|| {
                    Failure::named("NoSuchHandle", format!("indirect commands {}", commands.0))
                })?;
                encoder
                    .execute_commands_in_buffer(icb, *first..*first + *count)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::Scissor(rect) => encoder
                .set_scissor_rect(scissor(rect))
                .map_err(Failure::from_error)?,
            RenderStep::Scissors(rects) => {
                let held: Vec<ScissorRect> = rects.iter().map(scissor).collect();
                encoder
                    .set_scissor_rects(&held)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::Viewports(rects) => {
                let held: Vec<Viewport> = rects.iter().map(viewport_of).collect();
                encoder.set_viewports(&held).map_err(Failure::from_error)?;
            }
            RenderStep::Cull(mode) => encoder.set_cull_mode(*mode),
            RenderStep::FrontFacing(winding) => encoder.set_front_facing_winding(*winding),
            RenderStep::FillMode(mode) => encoder
                .set_triangle_fill_mode(*mode)
                .map_err(Failure::from_error)?,
            RenderStep::DepthClip(mode) => encoder
                .set_depth_clip_mode(*mode)
                .map_err(Failure::from_error)?,
            RenderStep::DepthBias(values) => encoder
                .set_depth_bias(values.bias, values.slope_scale, values.clamp)
                .map_err(Failure::from_error)?,
            RenderStep::StencilReferences { front, back } => {
                encoder.set_stencil_reference_values(*front, *back);
            }
            RenderStep::VertexBufferOffset { offset, index } => encoder
                .set_vertex_buffer_offset(*offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentBufferOffset { offset, index } => encoder
                .set_fragment_buffer_offset(*offset, *index)
                .map_err(Failure::from_error)?,
            RenderStep::FragmentBytes { bytes, index } => encoder
                .set_fragment_bytes(bytes, *index)
                .map_err(Failure::from_error)?,
            RenderStep::VertexBuffers {
                buffers: range,
                first,
            } => {
                let (held, offsets) = buffer_range(&buffer, range)?;
                let bound: Vec<Option<&Buffer>> = held.iter().copied().map(Some).collect();
                encoder
                    .set_vertex_buffers(&bound, &offsets, *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::FragmentBuffers {
                buffers: range,
                first,
            } => {
                let (held, offsets) = buffer_range(&buffer, range)?;
                let bound: Vec<Option<&Buffer>> = held.iter().copied().map(Some).collect();
                encoder
                    .set_fragment_buffers(&bound, &offsets, *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::VertexTexture {
                texture: held,
                index,
            } => encoder
                .set_vertex_texture(Some(texture(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::VertexTextures {
                textures: range,
                first,
            } => {
                let bound = texture_range(&texture, range)?;
                encoder
                    .set_vertex_textures(&bound, *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::FragmentTextures {
                textures: range,
                first,
            } => {
                let bound = texture_range(&texture, range)?;
                encoder
                    .set_fragment_textures(&bound, *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::VertexSampler {
                sampler: held,
                index,
            } => encoder
                .set_vertex_sampler_state(Some(sampler(*held)?), *index)
                .map_err(Failure::from_error)?,
            RenderStep::VertexSamplers {
                samplers: range,
                first,
            } => {
                let bound = sampler_range(&sampler, range)?;
                encoder
                    .set_vertex_sampler_states(&bound, None, *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::FragmentSamplers {
                samplers: range,
                first,
            } => {
                let bound = sampler_range(&sampler, range)?;
                encoder
                    .set_fragment_sampler_states(&bound, None, *first)
                    .map_err(Failure::from_error)?;
            }
            // Metal's one-slot clamped setter, which the runtime offers as a range of one with
            // clamps (`docs/METAL-COVERAGE.md`): the same call, stated at one index.
            RenderStep::VertexSamplerClamped {
                sampler: held,
                clamp,
                index,
            } => encoder
                .set_vertex_sampler_states(
                    &[Some(sampler(*held)?)],
                    Some((&[clamp.min], &[clamp.max])),
                    *index,
                )
                .map_err(Failure::from_error)?,
            RenderStep::FragmentSamplerClamped {
                sampler: held,
                clamp,
                index,
            } => encoder
                .set_fragment_sampler_states(
                    &[Some(sampler(*held)?)],
                    Some((&[clamp.min], &[clamp.max])),
                    *index,
                )
                .map_err(Failure::from_error)?,
            RenderStep::VertexSamplersClamped {
                samplers: range,
                first,
            } => {
                let (bound, minimums, maximums) = clamped_sampler_range(&sampler, range)?;
                encoder
                    .set_vertex_sampler_states(&bound, Some((&minimums, &maximums)), *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::FragmentSamplersClamped {
                samplers: range,
                first,
            } => {
                let (bound, minimums, maximums) = clamped_sampler_range(&sampler, range)?;
                encoder
                    .set_fragment_sampler_states(&bound, Some((&minimums, &maximums)), *first)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::BarrierScope {
                scope,
                after,
                before,
            } => encoder.memory_barrier_with_scope(*scope, *after, *before),
            RenderStep::BarrierResources {
                resources,
                after,
                before,
            } => {
                let over = resource_range(&resource, resources)?;
                encoder
                    .memory_barrier_with_resources(&over, *after, *before)
                    .map_err(Failure::from_error)?;
            }
            // Metal's deprecated residency calls name every render stage, which is how the
            // runtime implements them (`docs/METAL-COVERAGE.md`).
            RenderStep::UseResource {
                resource: held,
                usage,
            } => encoder
                .use_resource(resource(*held)?, *usage, RenderStages::ALL_KNOWN)
                .map_err(Failure::from_error)?,
            RenderStep::UseResourceInStages {
                resource: held,
                usage,
                stages,
            } => encoder
                .use_resource(resource(*held)?, *usage, *stages)
                .map_err(Failure::from_error)?,
            RenderStep::UseResources {
                resources: held,
                usage,
            } => {
                let over = resource_range(&resource, held)?;
                encoder
                    .use_resources(&over, *usage, RenderStages::ALL_KNOWN)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::UseResourcesInStages {
                resources: held,
                usage,
                stages,
            } => {
                let over = resource_range(&resource, held)?;
                encoder
                    .use_resources(&over, *usage, *stages)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::UseHeap(held) => encoder
                .use_heap(heap(*held)?, RenderStages::ALL_KNOWN)
                .map_err(Failure::from_error)?,
            RenderStep::UseHeapInStages { heap: held, stages } => encoder
                .use_heap(heap(*held)?, *stages)
                .map_err(Failure::from_error)?,
            RenderStep::UseHeaps(held) => {
                let over = heap_range(&heap, held)?;
                encoder
                    .use_heaps(&over, RenderStages::ALL_KNOWN)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::UseHeapsInStages {
                heaps: held,
                stages,
            } => {
                let over = heap_range(&heap, held)?;
                encoder
                    .use_heaps(&over, *stages)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::TessellationFactorBuffer {
                buffer: held,
                offset,
                instance_stride,
            } => {
                let factors = held.map(&buffer).transpose()?;
                encoder
                    .set_tessellation_factor_buffer(factors, *offset, *instance_stride)
                    .map_err(Failure::from_error)?;
            }
            RenderStep::TessellationFactorScale { bits } => {
                encoder.set_tessellation_factor_scale(f32::from_bits(*bits));
            }
            RenderStep::DrawPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                instances,
                base_instance,
            } => {
                let indices = patch_indices.map(&buffer).transpose()?;
                encoder
                    .draw_patches(
                        *control_points,
                        *first,
                        *count,
                        indices,
                        *patch_index_offset,
                        *instances,
                        *base_instance,
                    )
                    .map_err(Failure::from_error)?;
            }
            RenderStep::DrawIndexedPatches {
                control_points,
                first,
                count,
                patch_indices,
                patch_index_offset,
                control_point_indices,
                control_point_index_offset,
                instances,
                base_instance,
            } => {
                let indices = patch_indices.map(&buffer).transpose()?;
                encoder
                    .draw_indexed_patches(
                        *control_points,
                        *first,
                        *count,
                        indices,
                        *patch_index_offset,
                        buffer(*control_point_indices)?,
                        *control_point_index_offset,
                        *instances,
                        *base_instance,
                    )
                    .map_err(Failure::from_error)?;
            }
        }
    }
    encoder.end_encoding().map_err(Failure::from_error)
}

/// The usage an indirect command buffer's residency declaration states: `MTLResourceUsageRead`,
/// and `MTLResourceUsageWrite` beside it where `write`.
///
/// Distinct from [`resource_usage`], which takes the raw bits a residency *command* of a pass
/// states and can refuse them: the choice a command-buffer step offers is only whether the
/// commands' functions write what they are handed, so there are no unknown bits to answer for.
fn commands_usage(write: bool) -> ResourceUsage {
    if write {
        ResourceUsage::READ.union(ResourceUsage::WRITE)
    } else {
        ResourceUsage::READ
    }
}

/// The runtime's `ScissorRect` a case's rectangle is.
fn scissor(rect: &crate::render::ScissorBox) -> ScissorRect {
    ScissorRect::new(rect.x, rect.y, rect.width, rect.height)
}

/// The runtime's `Viewport` a case's viewport is.
fn viewport_of(rect: &crate::render::ViewportRect) -> Viewport {
    Viewport::new(
        rect.origin[0],
        rect.origin[1],
        rect.size[0],
        rect.size[1],
        rect.depth[0],
        rect.depth[1],
    )
}

/// The buffers and offsets a range step names.
fn buffer_range<'a>(
    held: &impl Fn(Handle) -> Answer<&'a Buffer>,
    range: &[(Handle, u64)],
) -> Answer<(Vec<&'a Buffer>, Vec<u64>)> {
    let buffers = range
        .iter()
        .map(|(handle, _)| held(*handle))
        .collect::<Answer<Vec<_>>>()?;
    Ok((buffers, range.iter().map(|(_, offset)| *offset).collect()))
}

/// The textures a range step names, as the call binds them.
fn texture_range<'a>(
    held: &impl Fn(Handle) -> Answer<&'a Texture>,
    range: &[Handle],
) -> Answer<Vec<Option<&'a Texture>>> {
    range.iter().map(|handle| held(*handle).map(Some)).collect()
}

/// The sampler states a range step names, as the call binds them.
fn sampler_range<'a>(
    held: &impl Fn(Handle) -> Answer<&'a SamplerState>,
    range: &[Handle],
) -> Answer<Vec<Option<&'a SamplerState>>> {
    range.iter().map(|handle| held(*handle).map(Some)).collect()
}

/// The sampler states a clamped range step names, with its two clamp arrays.
#[allow(clippy::type_complexity)]
fn clamped_sampler_range<'a>(
    held: &impl Fn(Handle) -> Answer<&'a SamplerState>,
    range: &[(Handle, LodClamp)],
) -> Answer<(Vec<Option<&'a SamplerState>>, Vec<f32>, Vec<f32>)> {
    let samplers = range
        .iter()
        .map(|(handle, _)| held(*handle).map(Some))
        .collect::<Answer<Vec<_>>>()?;
    Ok((
        samplers,
        range.iter().map(|(_, clamp)| clamp.min).collect(),
        range.iter().map(|(_, clamp)| clamp.max).collect(),
    ))
}

/// The resources a residency or barrier step names.
fn resource_range<'a>(
    held: &impl Fn(StepResource) -> Answer<ResourceRef<'a>>,
    range: &[StepResource],
) -> Answer<Vec<ResourceRef<'a>>> {
    range.iter().map(|resource| held(*resource)).collect()
}

/// The heaps a range step names.
fn heap_range<'a>(
    held: &impl Fn(Handle) -> Answer<&'a Heap>,
    range: &[Handle],
) -> Answer<Vec<&'a Heap>> {
    range.iter().map(|handle| held(*handle)).collect()
}

/// The pixel format [`crate::case::TARGET_FORMAT`] names, as the runtime spells it.
const TARGET_PIXEL_FORMAT: PixelFormat = PixelFormat::RGBA32Uint;

/// The region `origin` and `size` name, as the runtime takes one.
fn region_of(origin: [u32; 3], size: [u32; 3]) -> Region {
    Region::new_3d(
        u64::from(origin[0]),
        u64::from(origin[1]),
        u64::from(origin[2]),
        u64::from(size[0]),
        u64::from(size[1]),
        u64::from(size[2]),
    )
}

/// The raw `MTLRenderStages` bits as the runtime takes them.
fn stages(bits: u64) -> Answer<RenderStages> {
    RenderStages::try_from(bits as u32)
        .map_err(|problem| Failure::named("NoSuchRenderStages", format!("{bits:#x}: {problem}")))
}

/// The raw `MTLBarrierScope` bits as the runtime takes them.
fn scope_bits(bits: u64) -> Answer<BarrierScope> {
    BarrierScope::try_from(bits as u32)
        .map_err(|problem| Failure::named("NoSuchBarrierScope", format!("{bits:#x}: {problem}")))
}

impl Failure {
    /// The failure a typed runtime error is, for a call that returns one by value.
    fn from_error(error: metal2vulkan_runtime::Error) -> Self {
        Failure::of(&error)
    }
}

/// The raw primitive type a draw holds; a draw with no primitive type (patches, mesh) reads as
/// points, as Metal's own getters answer such a record.
fn drawn_primitive_type(draw: IndirectDraw) -> u64 {
    match draw {
        IndirectDraw::Primitives { primitive_type, .. }
        | IndirectDraw::IndexedPrimitives { primitive_type, .. } => primitive_type.raw(),
        _ => DrawPrimitiveType::from(PrimitiveType::Point).raw(),
    }
}

/// State one colour attachment of the runtime's descriptor from `row`.
///
/// A raw value the runtime does not define is its typed error, which the case records at the row
/// rather than turning into a cast.
fn state_color_attachment(
    attachment: &mut RenderPassColorAttachmentDescriptor,
    row: &Row,
) -> Answer<()> {
    attachment.load_action = LoadAction::from_raw(row.actions.load).map_err(Failure::from_error)?;
    attachment.store_action =
        StoreAction::from_raw(row.actions.store).map_err(Failure::from_error)?;
    attachment.store_action_options =
        StoreActionOptions::from_raw(row.actions.options).map_err(Failure::from_error)?;
    attachment.level = row.indices.0[0];
    attachment.slice = row.indices.0[1];
    attachment.depth_plane = row.indices.0[2];
    attachment.resolve_level = row.indices.0[3];
    attachment.resolve_slice = row.indices.0[4];
    attachment.resolve_depth_plane = row.indices.0[5];
    attachment.clear_color = ClearColor::new(
        row.clear_color[0],
        row.clear_color[1],
        row.clear_color[2],
        row.clear_color[3],
    );
    Ok(())
}

/// State the depth attachment of the runtime's descriptor from `row`.
fn state_depth_attachment(
    attachment: &mut RenderPassDepthAttachmentDescriptor,
    row: &Row,
) -> Answer<()> {
    attachment.load_action = LoadAction::from_raw(row.actions.load).map_err(Failure::from_error)?;
    attachment.store_action =
        StoreAction::from_raw(row.actions.store).map_err(Failure::from_error)?;
    attachment.store_action_options =
        StoreActionOptions::from_raw(row.actions.options).map_err(Failure::from_error)?;
    attachment.level = row.indices.0[0];
    attachment.slice = row.indices.0[1];
    attachment.depth_plane = row.indices.0[2];
    attachment.resolve_level = row.indices.0[3];
    attachment.resolve_slice = row.indices.0[4];
    attachment.resolve_depth_plane = row.indices.0[5];
    attachment.clear_depth = row.clear_depth;
    attachment.depth_resolve_filter =
        MultisampleDepthResolveFilter::from_raw(row.filters.depth).map_err(Failure::from_error)?;
    Ok(())
}

/// State the stencil attachment of the runtime's descriptor from `row`.
fn state_stencil_attachment(
    attachment: &mut RenderPassStencilAttachmentDescriptor,
    row: &Row,
) -> Answer<()> {
    attachment.load_action = LoadAction::from_raw(row.actions.load).map_err(Failure::from_error)?;
    attachment.store_action =
        StoreAction::from_raw(row.actions.store).map_err(Failure::from_error)?;
    attachment.store_action_options =
        StoreActionOptions::from_raw(row.actions.options).map_err(Failure::from_error)?;
    attachment.level = row.indices.0[0];
    attachment.slice = row.indices.0[1];
    attachment.depth_plane = row.indices.0[2];
    attachment.resolve_level = row.indices.0[3];
    attachment.resolve_slice = row.indices.0[4];
    attachment.resolve_depth_plane = row.indices.0[5];
    attachment.clear_stencil = row.clear_stencil;
    attachment.stencil_resolve_filter =
        MultisampleStencilResolveFilter::from_raw(row.filters.stencil)
            .map_err(Failure::from_error)?;
    Ok(())
}

/// What one colour attachment of the runtime's descriptor answers.
fn read_color_attachment(attachment: &RenderPassColorAttachmentDescriptor) -> AttachmentFacts {
    AttachmentFacts {
        has_texture: attachment.texture.is_some(),
        has_resolve_texture: attachment.resolve_texture.is_some(),
        indices: Indices([
            attachment.level,
            attachment.slice,
            attachment.depth_plane,
            attachment.resolve_level,
            attachment.resolve_slice,
            attachment.resolve_depth_plane,
        ]),
        actions: crate::pass_descriptors::Actions {
            load: attachment.load_action.raw(),
            store: attachment.store_action.raw(),
            options: attachment.store_action_options.raw(),
        },
    }
}

/// What the depth attachment of the runtime's descriptor answers.
fn read_depth_attachment(attachment: &RenderPassDepthAttachmentDescriptor) -> AttachmentFacts {
    AttachmentFacts {
        has_texture: attachment.texture.is_some(),
        has_resolve_texture: attachment.resolve_texture.is_some(),
        indices: Indices([
            attachment.level,
            attachment.slice,
            attachment.depth_plane,
            attachment.resolve_level,
            attachment.resolve_slice,
            attachment.resolve_depth_plane,
        ]),
        actions: crate::pass_descriptors::Actions {
            load: attachment.load_action.raw(),
            store: attachment.store_action.raw(),
            options: attachment.store_action_options.raw(),
        },
    }
}

/// What the stencil attachment of the runtime's descriptor answers.
fn read_stencil_attachment(attachment: &RenderPassStencilAttachmentDescriptor) -> AttachmentFacts {
    AttachmentFacts {
        has_texture: attachment.texture.is_some(),
        has_resolve_texture: attachment.resolve_texture.is_some(),
        indices: Indices([
            attachment.level,
            attachment.slice,
            attachment.depth_plane,
            attachment.resolve_level,
            attachment.resolve_slice,
            attachment.resolve_depth_plane,
        ]),
        actions: crate::pass_descriptors::Actions {
            load: attachment.load_action.raw(),
            store: attachment.store_action.raw(),
            options: attachment.store_action_options.raw(),
        },
    }
}

/// What the runtime's descriptor answers under `reads`.
fn read_pass_descriptor(descriptor: &RenderPassDescriptor, reads: Reads) -> PassDescriptorFacts {
    let mut facts = PassDescriptorFacts::empty();
    for index in 0..reads.colors() {
        let color = &descriptor.color_attachments[index as usize];
        if matches!(reads, Reads::Whole | Reads::Attachments) {
            facts.colors.push(read_color_attachment(color));
        }
        let clear = color.clear_color;
        facts.color_clears.push([
            clear.red.to_bits(),
            clear.green.to_bits(),
            clear.blue.to_bits(),
            clear.alpha.to_bits(),
        ]);
    }
    if matches!(reads, Reads::Whole | Reads::Attachments | Reads::ClearDepth) {
        let depth = &descriptor.depth_attachment;
        facts.clear_depth = Some(depth.clear_depth.to_bits());
        if reads != Reads::ClearDepth {
            facts.depth = Some(read_depth_attachment(depth));
            facts.depth_resolve_filter = Some(depth.depth_resolve_filter.raw());
        }
    }
    if matches!(
        reads,
        Reads::Whole | Reads::Attachments | Reads::ClearStencil
    ) {
        let stencil = &descriptor.stencil_attachment;
        facts.clear_stencil = Some(stencil.clear_stencil);
        if reads != Reads::ClearStencil {
            facts.stencil = Some(read_stencil_attachment(stencil));
            facts.stencil_resolve_filter = Some(stencil.stencil_resolve_filter.raw());
        }
    }
    if matches!(reads, Reads::Whole | Reads::Numbers) {
        facts.numbers = Some(Numbers([
            descriptor.render_target_array_length,
            descriptor.render_target_width,
            descriptor.render_target_height,
            descriptor.default_raster_sample_count,
            descriptor.tile_width,
            descriptor.tile_height,
            descriptor.imageblock_sample_length,
            descriptor.threadgroup_memory_length,
        ]));
        facts.has_visibility_buffer = Some(descriptor.visibility_result_buffer.is_some());
        facts.has_rate_map = Some(descriptor.rasterization_rate_map.is_some());
    }
    if matches!(reads, Reads::Whole | Reads::Positions) {
        facts.position_count = Some(descriptor.sample_positions.len() as u64);
        facts.positions = Some(
            descriptor
                .sample_positions
                .iter()
                .map(|position| (position.x.to_bits(), position.y.to_bits()))
                .collect(),
        );
    }
    facts
}

/// What a compute pipeline state answers of itself, read off the state rather than off a handle.
///
/// A free function because [`crate::pipeline_handlers`]' case reads these getters off a state it
/// makes and spends inside one call, with no handle to look up.
/// [`Side::compute_pipeline_facts`] is the same reading through the case's pool, so the two can
/// never drift.
fn our_compute_state_facts(
    state: &ComputePipelineState,
    registry: u64,
    imageblock: [u32; 3],
) -> ComputeStateFacts {
    let required = state.required_threads_per_threadgroup();
    ComputeStateFacts {
        label: state.label().map(str::to_owned),
        // The registry ID is each host's own number, so only the **comparison** is portable.
        device_is_the_cases: state.device().registry_id() == registry,
        static_threadgroup_memory_length: u64::from(state.static_threadgroup_memory_length()),
        support_indirect_command_buffers: state.support_indirect_command_buffers(),
        imageblock_memory_length: state.imageblock_memory_length_for_dimensions(size(imageblock)),
        required_threads_per_threadgroup: [required.width, required.height, required.depth],
    }
}

/// What a drawing render pipeline state answers of itself, read off the state.
///
/// A free function for the reason [`our_compute_state_facts`] gives.
fn our_render_state_facts(
    state: &RenderPipelineState,
    registry: u64,
    imageblock: [u32; 3],
) -> RenderStateFacts {
    let required = state.required_threads_per_tile_threadgroup();
    RenderStateFacts {
        label: state.label().map(str::to_owned),
        device_is_the_cases: state.device().registry_id() == registry,
        max_total_threads_per_threadgroup: u64::from(state.max_total_threads_per_threadgroup()),
        threadgroup_size_matches_tile_size: state.threadgroup_size_matches_tile_size(),
        imageblock_sample_length: state.imageblock_sample_length().ok(),
        imageblock_memory_length: state
            .imageblock_memory_length_for_dimensions(size(imageblock))
            .ok(),
        support_indirect_command_buffers: state.support_indirect_command_buffers(),
        required_threads_per_tile_threadgroup: [required.width, required.height, required.depth],
    }
}

/// The three getters a **tile** state answers, read off the state.
///
/// A free function for the reason [`our_compute_state_facts`] gives.
fn our_tile_state_facts(state: &RenderPipelineState) -> TileFacts {
    TileFacts {
        max_total_threads_per_threadgroup: u64::from(state.max_total_threads_per_threadgroup()),
        threadgroup_size_matches_tile_size: state.threadgroup_size_matches_tile_size(),
        imageblock_sample_length: state.imageblock_sample_length().ok(),
    }
}

/// The bytes a step names, built with the suite's own container writer.
///
/// Our whole of the two packagers Metal's side reaches through `xcrun`: the runtime takes a
/// metallib's bytes and has no writer of its own, so [`crate::archives::container`] stands in for
/// `xcrun metallib` and for `xcrun metal -dynamiclib`. The container is written around the bitcode
/// `llvm-as` assembled, which is the dialect this side parses, exactly as every case already hands
/// the two sides two dialects of one shader.
fn runtime_bytes(program: &Program<'_>, bytes: Bytes) -> Answer<Vec<u8>> {
    let held = |at: usize| -> Answer<&ProgramLibrary<'_>> {
        program
            .libraries
            .get(at)
            .ok_or_else(|| Failure::named("NoSuchProgramLibrary", at.to_string()))
    };
    match bytes {
        Bytes::Metallib { library } => {
            let held = held(library)?;
            Ok(archives::container(held.source.bitcode, held.entries, None))
        }
        Bytes::DynamicLibraryFile {
            library,
            install_name,
        } => {
            let held = held(library)?;
            Ok(archives::container(
                held.source.bitcode,
                held.entries,
                Some(install_name),
            ))
        }
        Bytes::Text(text) => Ok(text.as_bytes().to_vec()),
        Bytes::Empty => Ok(Vec::new()),
    }
}

/// A stage of one of a program's libraries, as a descriptor takes it.
fn runtime_stage(built: &[Library], at: usize, name: &str) -> Answer<StageSource> {
    let library = built
        .get(at)
        .ok_or_else(|| Failure::named("NoSuchProgramLibrary", at.to_string()))?;
    Ok(StageSource::Function(
        library
            .new_function_with_name(name)
            .map_err(Failure::from_error)?
            .function_descriptor(),
    ))
}

/// The render pipeline descriptor a step names, over `attachments` colour attachments of the
/// program's format.
fn runtime_render_descriptor(
    program: &Program<'_>,
    built: &[Library],
    at: usize,
    vertex: &str,
    fragment: &str,
    attachments: usize,
) -> Answer<RenderPipelineDescriptor> {
    let mut descriptor = RenderPipelineDescriptor::new(runtime_stage(built, at, vertex)?);
    descriptor.fragment = Some(runtime_stage(built, at, fragment)?);
    for index in 0..attachments {
        let slot = descriptor
            .color_attachments
            .get_mut(index)
            .ok_or_else(|| Failure::named("NoSuchColorAttachment", index.to_string()))?;
        *slot = Some(ColorAttachmentDescriptor::new(program.attachment));
    }
    Ok(descriptor)
}

/// One step of a program on the runtime.
#[allow(clippy::too_many_arguments)]
fn runtime_step(
    device: &Device,
    scratch: &Scratch,
    program: &Program<'_>,
    built: &[Library],
    registry: u64,
    archives_made: &mut Vec<Option<BinaryArchive>>,
    from_data: &mut Vec<Option<Library>>,
    dynamics: &mut Vec<Option<DynamicLibrary>>,
    step: &Step,
) -> Answer<Observation> {
    match *step {
        Step::NewArchive { from } => {
            let descriptor =
                BinaryArchiveDescriptor::new(from.map(|source| scratch.source(source)));
            let made = device.new_binary_archive(&descriptor).ok();
            let refused = made.is_none();
            archives_made.push(made);
            Ok(Observation::Made { refused })
        }
        Step::ArchiveLabel { archive, set } => {
            let Some(Some(archive)) = archives_made.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            if let Some(label) = set {
                archive.set_label(label);
            }
            Ok(Observation::Label(archive.label()))
        }
        Step::ArchiveDevice { archive } => {
            let Some(Some(archive)) = archives_made.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            Ok(Observation::Flag(
                archive.device().registry_id() == registry,
            ))
        }
        Step::Add { archive, what } => {
            let Some(Some(archive)) = archives_made.get(archive).cloned() else {
                return Ok(Observation::refused_for(step));
            };
            let refused = match what {
                Added::Compute { library, kernel } => archive
                    .add_compute_pipeline_functions(&ComputePipelineDescriptor::new(runtime_stage(
                        built, library, kernel,
                    )?))
                    .is_err(),
                Added::Render {
                    library,
                    vertex,
                    fragment,
                    attachments,
                } => archive
                    .add_render_pipeline_functions(&runtime_render_descriptor(
                        program,
                        built,
                        library,
                        vertex,
                        fragment,
                        attachments,
                    )?)
                    .is_err(),
                Added::Tile { library, tile } => {
                    let mut descriptor =
                        TileRenderPipelineDescriptor::new(runtime_stage(built, library, tile)?);
                    descriptor.color_attachment_formats[0] = Some(program.attachment);
                    archive
                        .add_tile_render_pipeline_functions(&descriptor)
                        .is_err()
                }
                // `addMeshRenderPipelineFunctionsWithDescriptor:error:` has no counterpart on this
                // side: `BinaryArchive` has three adders and Metal has four. Answered as the
                // refusal it is, so the case reads `Refused` with the call named, rather than as a
                // bool that differed.
                Added::Mesh { .. } => {
                    return Err(Failure::unavailable(
                        "addMeshRenderPipelineFunctionsWithDescriptor",
                    ))
                }
            };
            Ok(Observation::Made { refused })
        }
        Step::Serialize { archive, to } => {
            let Some(Some(archive)) = archives_made.get(archive) else {
                return Ok(Observation::refused_for(step));
            };
            let refused = archive.serialize_to_url(&scratch.path(to)).is_err();
            Ok(Observation::Made { refused })
        }
        Step::Create {
            archives,
            options,
            what,
        } => {
            let mut listed = Vec::with_capacity(archives.len());
            for index in archives {
                match archives_made.get(*index).cloned().flatten() {
                    Some(archive) => listed.push(archive),
                    None => return Ok(Observation::refused_for(step)),
                }
            }
            let mut options = PipelineOptions::new(PipelineOption::from_raw(options));
            options.binary_archives = listed;
            let refused = match what {
                Created::Compute { library, kernel } => device
                    .new_compute_pipeline_state(
                        &ComputePipelineDescriptor::new(runtime_stage(built, library, kernel)?),
                        &options,
                    )
                    .is_err(),
                Created::Render {
                    library,
                    vertex,
                    fragment,
                    attachments,
                } => device
                    .new_render_pipeline_state(
                        &runtime_render_descriptor(
                            program,
                            built,
                            library,
                            vertex,
                            fragment,
                            attachments,
                        )?,
                        &options,
                    )
                    .is_err(),
            };
            Ok(Observation::Made { refused })
        }
        Step::FileState { file } => Ok(scratch.state(file)),
        Step::Write { file, bytes } => {
            let written = runtime_bytes(program, bytes)?;
            let refused = std::fs::write(scratch.path(file), &written).is_err();
            Ok(Observation::Made { refused })
        }
        Step::NewLibraryWithData { bytes } => {
            let data = runtime_bytes(program, bytes)?;
            let made = device.new_library_with_data(&data).ok();
            let facts = made.as_ref().map(|library| LibraryFacts {
                function_names: library.function_names(),
                library_type: library.library_type(),
                install_name: library.install_name().map(str::to_owned),
                device_is_the_cases: library.device().registry_id() == registry,
            });
            let refused = made.is_none();
            from_data.push(made);
            Ok(Observation::Library { refused, facts })
        }
        Step::NewDynamicLibrary { from } => {
            let made = match from {
                DynamicSource::Library(at) => {
                    let library = match at {
                        LibraryAt::Program(index) => {
                            built.get(index).cloned().ok_or_else(|| {
                                Failure::named("NoSuchProgramLibrary", index.to_string())
                            })?
                        }
                        LibraryAt::FromData(index) => {
                            match from_data.get(index).cloned().flatten() {
                                Some(library) => library,
                                None => {
                                    dynamics.push(None);
                                    return Ok(Observation::refused_for(step));
                                }
                            }
                        }
                    };
                    device.new_dynamic_library(&library).ok()
                }
                DynamicSource::Url(source) => device
                    .new_dynamic_library_with_url(&scratch.source(source))
                    .ok(),
            };
            let install_name = made
                .as_ref()
                .map(|dynamic| dynamic.install_name().to_owned());
            let refused = made.is_none();
            dynamics.push(made);
            Ok(Observation::Dynamic {
                refused,
                install_name,
            })
        }
        Step::DynamicLabel { dynamic, set } => {
            let Some(Some(dynamic)) = dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            if let Some(label) = set {
                dynamic.set_label(label);
            }
            Ok(Observation::Label(dynamic.label()))
        }
        Step::DynamicDevice { dynamic } => {
            let Some(Some(dynamic)) = dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            Ok(Observation::Flag(
                dynamic.device().registry_id() == registry,
            ))
        }
        Step::SerializeDynamic { dynamic, to } => {
            let Some(Some(dynamic)) = dynamics.get(dynamic) else {
                return Ok(Observation::refused_for(step));
            };
            let refused = dynamic.serialize_to_url(&scratch.path(to)).is_err();
            Ok(Observation::Made { refused })
        }
    }
}

/// The Metal buffer index the owned callers declare their `visible_function_table` at, which is
/// the index the encoder binds the case's table at. It is the fixtures' own `[[buffer(1)]]` and
/// not a choice of this side's.
const TABLE_INDEX: u32 = 1;

/// The pipeline state a case's handles and table come from, on this side: a compute state, or a
/// render state and the one stage the case asks about.
///
/// It exists because the two states' selectors are two different calls of the runtime's — a
/// kernel's take no stage and a render state's take one — and the case's answer must not depend
/// on which, so both go through one pair of methods here.
enum RuntimeMaker {
    Compute(ComputePipelineState),
    Render(RenderPipelineState, RenderStages),
}

impl RuntimeMaker {
    /// `functionHandleWithFunction:`, or its `:stage:` spelling.
    fn function_handle(
        &self,
        function: &metal2vulkan_runtime::Function,
    ) -> Result<Option<metal2vulkan_runtime::FunctionHandle>, metal2vulkan_runtime::Error> {
        match self {
            RuntimeMaker::Compute(state) => state.function_handle(function),
            RuntimeMaker::Render(state, stage) => state.function_handle(function, *stage),
        }
    }

    /// `newVisibleFunctionTableWithDescriptor:`, or its `:stage:` spelling.
    fn new_visible_function_table(
        &self,
        descriptor: &VisibleFunctionTableDescriptor,
    ) -> Result<VisibleFunctionTable, metal2vulkan_runtime::Error> {
        match self {
            RuntimeMaker::Compute(state) => state.new_visible_function_table(descriptor),
            RuntimeMaker::Render(state, stage) => {
                state.new_visible_function_table(descriptor, *stage)
            }
        }
    }
}
