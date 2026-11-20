//! The case vocabulary's pipeline descriptors as the translator's types: the fields this side
//! states on the runtime's descriptors, mapped apart from any device so a test can hold them field
//! by field (`tests::the_runtime_side_forwards_every_field_of_a_pipeline_spec`).

use crate::case::{Answer, Failure, StageInput, TextureRole};
use crate::render::{ColorState, RenderPipelineSpec, VertexStride};
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};

/// The usage a texture's role is.
pub(crate) fn usage_of(role: TextureRole) -> TextureUsage {
    match role {
        TextureRole::Read => TextureUsage::SHADER_READ,
        TextureRole::Write => TextureUsage::SHADER_WRITE,
        TextureRole::ReadWrite => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureRole::RenderTarget => TextureUsage::RENDER_TARGET,
    }
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
        // `None` is the runtime's spelling of `MTLBufferLayoutStrideDynamic`, which is the
        // sentinel the suite's vocabulary states as its own variant.
        let mut held = VertexBufferLayoutDescriptor::new(match buffer.stride {
            VertexStride::Bytes(stride) => Some(stride),
            VertexStride::Dynamic => None,
        });
        held.step_function = buffer.step;
        held.step_rate = buffer.rate;
        descriptor.layouts.insert(buffer.index, held);
    }
    descriptor
}

/// The runtime's stage-input descriptor for `stage_input`: the same attribute indices, buffer
/// indices, formats, strides, steps and rates, since the case's vocabulary is Metal's own.
pub(crate) fn stage_input_descriptor(stage_input: &StageInput) -> StageInputDescriptor {
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
