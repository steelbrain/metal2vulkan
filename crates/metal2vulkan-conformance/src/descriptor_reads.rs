//! The translator's pipeline descriptors as a side states and reads them for the descriptor
//! families ([`crate::render_descriptors`], [`crate::compute_descriptors`]), and the stage lists
//! and buffer mutability every pipeline descriptor family reads.
//!
//! Both runtimes take Metal's pipeline descriptors as the translator's
//! ([`RenderPipelineDescriptor`], [`ComputePipelineDescriptor`]), so the stating and the reading
//! are one walk for both sides. What neither descriptor holds, its `label` and its
//! `binaryArchives`, each side answers itself as a [`Field`].

use crate::case::Field;
use crate::compute_descriptors::{
    ComputeDescriptorFacts, SuppliedLists as ComputeSuppliedLists, AWAY as COMPUTE_AWAY,
};
use crate::render_descriptors::{
    AttributeFacts as DescriptorAttributeFacts, ColorFacts, LayoutFacts, LinkedCounts,
    RenderDescriptorFacts, SuppliedLists, VertexDescriptorFacts, AWAY, READ_BUFFERS, READ_COLORS,
    READ_VERTEX_SLOTS,
};
use metal2vulkan::metal::{Mutability, PixelFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
    TessellationDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};

/// Set every field of the translator's descriptor that [`AWAY`] states, each away from Metal's
/// own default.
///
/// The label is no field of the translator's descriptor: a side states it where its runtime
/// takes it, or answers its read-back as a refused [`Field`].
pub(crate) fn state_render(descriptor: &mut RenderPipelineDescriptor) {
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

/// What the translator's `descriptor` answers of itself, beside the `label` and `binary_archives`
/// the side answers, a side's answer to every getter of `MTLRenderPipelineDescriptor`'s own
/// surface.
///
/// The slots Metal's arrays answer for an index nothing set are the **runtime's claim about
/// Metal's defaults**, written here rather than taken from the case: an attachment the descriptor
/// holds no entry for is Metal's `MTLPixelFormatInvalid` with blending off and every channel
/// written, a buffer slot nothing set is `MTLMutabilityDefault`, and an attribute slot nothing set
/// is format `Invalid` at offset 0 in buffer 0 with a layout of stride 0 stepping per vertex at a
/// rate of 1. Each of those is a claim the Metal leg's recording checks.
pub(crate) fn read_render(
    descriptor: &RenderPipelineDescriptor,
    label: Field<Option<String>>,
    binary_archives: Field<u64>,
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
        label,
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
        binary_archives,
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
pub(crate) fn read_mutability(stated: &std::collections::BTreeMap<u32, Mutability>) -> Vec<u32> {
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
pub(crate) fn stage_lists(source: Option<&StageSource>) -> (u64, LinkedCounts) {
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
pub(crate) fn stage_entry(source: &StageSource) -> String {
    match source {
        StageSource::Function(held) => held.function.info().name.to_owned(),
        StageSource::Compiled(_) => COMPILED.to_owned(),
    }
}

/// Set every field of the translator's compute pipeline descriptor that [`COMPUTE_AWAY`] states,
/// each away from Metal's own default.
///
/// The label is no field of the translator's descriptor, as [`state_render`] says.
pub(crate) fn state_compute(descriptor: &mut ComputePipelineDescriptor) {
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

/// What the translator's compute pipeline `descriptor` answers of itself, beside the `label` and
/// `binary_archives` the side answers, a side's answer to every getter of
/// `MTLComputePipelineDescriptor`'s compared surface.
///
/// Metal's zero thread limit and zero required threadgroup size are the runtime's `None`, which is
/// the claim the Metal leg's recording checks.
pub(crate) fn read_compute(
    descriptor: &ComputePipelineDescriptor,
    label: Field<Option<String>>,
    binary_archives: Field<u64>,
    supplied: bool,
) -> ComputeDescriptorFacts {
    let (preloaded, linked) = stage_lists(Some(&descriptor.kernel));
    let required = descriptor
        .required_threads_per_threadgroup
        .unwrap_or([0; 3]);
    ComputeDescriptorFacts {
        label,
        compute_function: Some(stage_entry(&descriptor.kernel)),
        threadgroup_size_is_multiple_of_execution_width: descriptor
            .threadgroup_size_is_multiple_of_execution_width,
        max_total_threads_per_threadgroup: u64::from(
            descriptor.max_total_threads_per_threadgroup.unwrap_or(0),
        ),
        buffers: read_mutability(&descriptor.buffers),
        support_indirect_command_buffers: descriptor.support_indirect_command_buffers,
        binary_archives,
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
