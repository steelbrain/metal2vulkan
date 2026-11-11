//! The Vulkan graphics pipeline of an engine `RenderPipeline`: one builder that L4 driver
//! acceptance ([`super::acceptance`]) and the render executor ([`super::render`]) share, so a
//! pipeline the sweep creates is the pipeline the executor draws with.
//!
//! Everything comes from the artifact's graphics state and vertex input: the render pass
//! ([`create_render_pass`]: every color attachment at its Metal index, each framebuffer-fetch
//! input as an input attachment of its color attachment with the by-region self-dependency the
//! contract names, rasterization-order color access where the artifact states it, and the
//! depth/stencil attachment), the vertex input (static strides, and zero for a stride the pipeline
//! makes dynamic state; instance divisors through the divisor state), the multisample, blend
//! (rasterization-order access with the subpass's), and dynamic states. The caller supplies what the artifact leaves to the
//! consumer: each stage's specialization, the draw's topology within the compiled class with the
//! primitive restart Metal applies to it, and the depth-stencil state and front face (Metal's
//! `MTLDepthStencilState` and front-facing winding, which are encoder state, not pipeline state).
//! What this builder does not create is [`GraphicsError::Unmodelled`], never a guess.

use crate::executor_contract::DrawDepthStencil;
use ash::vk;
use metal2vulkan::device::DriverFact;
use metal2vulkan::pipeline::{GraphicsState, RenderPipeline, VertexStride};
use std::collections::BTreeSet;
use std::ffi::CString;
use std::time::Duration;

use super::METAL_DEFAULT_FRONT_FACE;

/// Why no graphics pipeline (or render pass) was created.
#[derive(Debug)]
pub(super) enum GraphicsError {
    /// A state this builder does not create (a per-bind stride the pipeline keeps static, a
    /// depth and a stencil format no one attachment holds, a framebuffer-fetch input without a
    /// color attachment): a modelled limit of the tool.
    Unmodelled(String),
    /// The device does not provide what the pipeline needs: the format of its depth/stencil
    /// attachment as a depth/stencil attachment ([`attachable_depth_stencil_format`]).
    NotSupported(String),
    /// The driver refused an object.
    Refused(String),
}

impl std::fmt::Display for GraphicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphicsError::Unmodelled(why)
            | GraphicsError::NotSupported(why)
            | GraphicsError::Refused(why) => f.write_str(why),
        }
    }
}

/// The one depth/stencil attachment format of `graphics`: its depth or its stencil format, or the
/// format both name. Metal's separate `Depth32Float` and `Stencil8` attachments are one Vulkan
/// attachment; the only such pair this builder combines is `D32_SFLOAT` with `S8_UINT`.
pub(super) fn depth_stencil_format(
    graphics: &GraphicsState,
) -> Result<Option<vk::Format>, GraphicsError> {
    let depth = graphics.depth_format.map(vk::Format::from);
    let stencil = graphics.stencil_format.map(vk::Format::from);
    Ok(match (depth, stencil) {
        (None, None) => None,
        (Some(format), None) | (None, Some(format)) => Some(format),
        (Some(depth), Some(stencil)) if depth == stencil => Some(depth),
        (Some(vk::Format::D32_SFLOAT), Some(vk::Format::S8_UINT)) => {
            Some(vk::Format::D32_SFLOAT_S8_UINT)
        }
        (Some(depth), Some(stencil)) => {
            return Err(GraphicsError::Unmodelled(format!(
                "a depth attachment of format {} and a stencil attachment of format {} in one \
                 render pass",
                depth.as_raw(),
                stencil.as_raw()
            )))
        }
    })
}

/// [`depth_stencil_format`] of `graphics`, when the device of `physical` can attach it
/// ([`attachable`]).
pub(super) fn attachable_depth_stencil_format(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    graphics: &GraphicsState,
) -> Result<Option<vk::Format>, GraphicsError> {
    depth_stencil_format(graphics)?
        .map(|format| {
            // SAFETY: `physical` is a live physical device of `instance`.
            let properties =
                unsafe { instance.get_physical_device_format_properties(physical, format) };
            attachable(format, properties)
        })
        .transpose()
}

/// `format` when a device whose `properties` for it are these supports it as a depth/stencil
/// attachment with optimal tiling, else [`GraphicsError::NotSupported`]. Vulkan requires
/// `D32_SFLOAT` but neither `S8_UINT` nor `D32_SFLOAT_S8_UINT` (only one of it and
/// `D24_UNORM_S8_UINT`, which holds less than Metal's `Depth32Float`), so a device may lack the
/// attachment Metal's separate depth and stencil ones need: typed, never a failed run.
pub(super) fn attachable(
    format: vk::Format,
    properties: vk::FormatProperties,
) -> Result<vk::Format, GraphicsError> {
    if properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    {
        Ok(format)
    } else {
        let name = match format {
            vk::Format::D32_SFLOAT_S8_UINT => "D32_SFLOAT_S8_UINT".to_string(),
            vk::Format::D32_SFLOAT => "D32_SFLOAT".to_string(),
            vk::Format::S8_UINT => "S8_UINT".to_string(),
            other => format!("VkFormat {}", other.as_raw()),
        };
        Err(GraphicsError::NotSupported(format!(
            "depth/stencil attachment format {name}"
        )))
    }
}

/// Whether `format` has a stencil aspect.
pub(super) fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Whether `format` has a depth aspect.
pub(super) fn has_depth(format: vk::Format) -> bool {
    format != vk::Format::S8_UINT
}

/// The render pass of `graphics`: one subpass writing every color attachment at its Metal index
/// (`VK_ATTACHMENT_UNUSED` for an absent one), in `COLOR_ATTACHMENT_OPTIMAL` (`GENERAL` for an
/// attachment a framebuffer-fetch input also reads, as the input attachment of the same index,
/// with the by-region self-dependency of [`FETCH_SELF_DEPENDENCY`]), then the depth/stencil
/// attachment ([`depth_stencil_format`]) in `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`.
/// The subpass has rasterization-order color access when the artifact states it
/// (`GraphicsState::rasterization_order_attachment_access`), so each fetch sees the earlier
/// fragments of the draw that cover its pixel, as in Metal; where it does not, each draw records
/// after a barrier of that dependency ([`fetch_between_draws`]). Every attachment is loaded before and
/// stored after the subpass. The framebuffer's views are the present color attachments in index
/// order, then the depth/stencil one.
pub(super) fn create_render_pass(
    device: &ash::Device,
    graphics: &GraphicsState,
) -> Result<vk::RenderPass, GraphicsError> {
    let samples: vk::SampleCountFlags = graphics.rasterization_samples.into();
    let fetched = |index: u32| {
        graphics
            .input_attachments
            .iter()
            .any(|input| input.color_index == index)
    };
    let mut attachments = Vec::new();
    let mut colors = Vec::new();
    let mut numbers = Vec::new();
    for (index, state) in graphics.color_attachments.iter().enumerate() {
        let Some(state) = state else {
            colors.push(
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED),
            );
            numbers.push(None);
            continue;
        };
        let layout = if fetched(index as u32) {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let number = attachments.len() as u32;
        colors.push(
            vk::AttachmentReference::default()
                .attachment(number)
                .layout(layout),
        );
        numbers.push(Some((number, layout)));
        attachments.push(
            vk::AttachmentDescription::default()
                .format(state.format.into())
                .samples(samples)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(layout)
                .final_layout(layout),
        );
    }
    let mut inputs = Vec::new();
    for input in &graphics.input_attachments {
        let at = input.input_attachment_index as usize;
        if inputs.len() <= at {
            inputs.resize(
                at + 1,
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED),
            );
        }
        let (number, layout) = numbers
            .get(input.color_index as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                GraphicsError::Unmodelled(format!(
                    "framebuffer-fetch input {} has no color attachment",
                    input.color_index
                ))
            })?;
        inputs[at] = vk::AttachmentReference::default()
            .attachment(number)
            .layout(layout);
    }
    let depth_stencil = depth_stencil_format(graphics)?.map(|format| {
        let number = attachments.len() as u32;
        let (depth_op, stencil_op) = (
            if has_depth(format) {
                (vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE)
            } else {
                (
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::DONT_CARE,
                )
            },
            if has_stencil(format) {
                (vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE)
            } else {
                (
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::DONT_CARE,
                )
            },
        );
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(samples)
                .load_op(depth_op.0)
                .store_op(depth_op.1)
                .stencil_load_op(stencil_op.0)
                .stencil_store_op(stencil_op.1)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        vk::AttachmentReference::default()
            .attachment(number)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    });
    let mut subpass = vk::SubpassDescription::default()
        .flags(subpass_flags(graphics))
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)
        .input_attachments(&inputs);
    if let Some(reference) = depth_stencil.as_ref() {
        subpass = subpass.depth_stencil_attachment(reference);
    }
    let subpasses = [subpass];
    // Framebuffer fetch reads the attachment the same subpass writes: the contract's by-region
    // self-dependency.
    let scopes = FETCH_SELF_DEPENDENCY;
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(0)
        .src_stage_mask(scopes.0)
        .dst_stage_mask(scopes.0)
        .src_access_mask(scopes.1)
        .dst_access_mask(scopes.2)
        .dependency_flags(vk::DependencyFlags::BY_REGION)];
    let mut info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    if !graphics.input_attachments.is_empty() {
        info = info.dependencies(&dependencies);
    }
    // SAFETY: `info` and every array it points at are live for the call.
    unsafe { device.create_render_pass(&info, None) }
        .map_err(|error| GraphicsError::Refused(format!("create render pass: {error}")))
}

/// The stages and the source and destination accesses of the self-dependency of a subpass whose
/// colors a framebuffer-fetch input reads ([`create_render_pass`]), by region: from the color
/// writes to the input attachment reads (a fetch after an earlier draw's writes, or the load
/// operations'), and from the fragment shader's reads to the color reads and writes (a write
/// after an earlier draw's fetch).
pub(super) const FETCH_SELF_DEPENDENCY: (vk::PipelineStageFlags, vk::AccessFlags, vk::AccessFlags) = (
    vk::PipelineStageFlags::from_raw(
        vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw(),
    ),
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    vk::AccessFlags::from_raw(
        vk::AccessFlags::INPUT_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw(),
    ),
);

/// Whether `graphics` reads input attachments ordered between draws alone: on a device with
/// neither rasterization-order color attachment access nor the pixel interlock, which states no
/// rasterization-order access (`GraphicsState::rasterization_order_attachment_access`). The
/// contract has the consumer order each draw of it after the color writes of the draws before,
/// and each draw that writes a color after their reads, by a barrier of the subpass's dependency
/// on itself; the executor records one before each of its draws ([`fetch_between_draws`]), after
/// the load operations too, as the runtime does.
pub(super) fn fetches_between_draws(graphics: &GraphicsState) -> bool {
    !graphics.input_attachments.is_empty() && !graphics.rasterization_order_attachment_access
}

/// Record, into `command`, inside a render pass of [`create_render_pass`] whose subpass reads
/// input attachments, the barrier of its [`FETCH_SELF_DEPENDENCY`]: the draws after it read and
/// write the colors after the draws before it wrote and read them.
///
/// # Safety
///
/// `command` is recording inside such a render pass instance, on `device`.
pub(super) unsafe fn fetch_between_draws(device: &ash::Device, command: vk::CommandBuffer) {
    let (stages, src, dst) = FETCH_SELF_DEPENDENCY;
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst);
    // SAFETY: the caller's contract; the subpass depends on itself by region with these scopes.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            stages,
            stages,
            vk::DependencyFlags::BY_REGION,
            &[barrier],
            &[],
            &[],
        );
    }
}

/// The subpass flags of `graphics`: rasterization-order color access where it states it.
pub(super) fn subpass_flags(graphics: &GraphicsState) -> vk::SubpassDescriptionFlags {
    if graphics.rasterization_order_attachment_access {
        vk::SubpassDescriptionFlags::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_EXT
    } else {
        vk::SubpassDescriptionFlags::empty()
    }
}

/// The color blend state flags of `graphics`, matching its subpass's ([`subpass_flags`]).
pub(super) fn color_blend_flags(
    graphics: &GraphicsState,
) -> vk::PipelineColorBlendStateCreateFlags {
    if graphics.rasterization_order_attachment_access {
        vk::PipelineColorBlendStateCreateFlags::RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXT
    } else {
        vk::PipelineColorBlendStateCreateFlags::empty()
    }
}

/// One stage's specialization: its map entries and packed data.
pub(super) type Specialization = (Vec<vk::SpecializationMapEntry>, Vec<u8>);

/// What the consumer decides for one graphics pipeline of a `RenderPipeline`.
pub(super) struct Consumer<'a> {
    /// Each stage's module, in the artifact's stage order.
    pub modules: &'a [vk::ShaderModule],
    /// Each stage's specialization, in the same order.
    pub specializations: &'a [Specialization],
    pub layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    /// The draw's topology, within the compiled topology class.
    pub topology: vk::PrimitiveTopology,
    /// Whether the input assembly restarts primitives: Metal restarts every strip at the all-ones
    /// index and never a list (`metal::state::input_assembly`).
    pub primitive_restart: bool,
    /// The encoder's depth-stencil state ([`metal_depth_stencil`]).
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    /// The encoder's cull mode.
    pub cull_mode: vk::CullModeFlags,
    /// The encoder's front-facing winding ([`front_face`]).
    pub front_face: vk::FrontFace,
}

/// Metal's default `MTLDepthStencilState`: no depth test (compare `Always`, writes off), no
/// stencil test.
pub(super) fn metal_default_depth_stencil() -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    metal_depth_stencil(&DrawDepthStencil::default())
}

/// The depth-stencil state of `state` as the Metal oracle sets it: the product's mapping of its
/// `MTLDepthStencilDescriptor` (`metal::state::depth_stencil_state`), with each face's stencil
/// reference. A `[[stencil]]` output replaces the reference on both faces.
pub(super) fn metal_depth_stencil(
    state: &DrawDepthStencil,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let mapped = state.vulkan();
    let face = |ops: metal2vulkan::metal::state::StencilOpState, reference: u32| {
        vk::StencilOpState::default()
            .fail_op(ops.fail_op.into())
            .pass_op(ops.pass_op.into())
            .depth_fail_op(ops.depth_fail_op.into())
            .compare_op(ops.compare_op.into())
            .compare_mask(ops.compare_mask)
            .write_mask(ops.write_mask)
            .reference(reference)
    };
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(mapped.depth_test_enable)
        .depth_write_enable(mapped.depth_write_enable)
        .depth_compare_op(mapped.depth_compare_op.into())
        .stencil_test_enable(mapped.stencil_test_enable)
        .front(face(mapped.front, state.front_reference))
        .back(face(mapped.back, state.back_reference))
}

/// The front face of `state`'s front-facing winding: Metal's clockwise default is
/// [`METAL_DEFAULT_FRONT_FACE`] under the executor's flipped viewport, and its counter-clockwise
/// winding the other face.
pub(super) fn front_face(state: &DrawDepthStencil) -> vk::FrontFace {
    if state.winding == metal2vulkan::metal::Winding::CounterClockwise {
        vk::FrontFace::COUNTER_CLOCKWISE
    } else {
        METAL_DEFAULT_FRONT_FACE
    }
}

/// Create the graphics pipeline of `pipeline` as `consumer` decides it: the pipeline and the
/// duration of the `vkCreateGraphicsPipelines` call alone. The caller owns (and destroys) the
/// pipeline. The diagnostics the executing driver's `facts` measure it to report as it makes
/// the pipeline (`DriverFact::pipeline_creation_diagnostics`) fail no row from inside that call;
/// those of its color attachments the driver cannot blend fail none anywhere
/// (`MessageSink::making`).
pub(super) fn create_graphics_pipeline(
    device: &ash::Device,
    facts: &BTreeSet<DriverFact>,
    pipeline: &RenderPipeline,
    consumer: &Consumer<'_>,
) -> Result<(vk::Pipeline, Duration), GraphicsError> {
    let graphics = &pipeline.graphics;
    let dynamic: Vec<vk::DynamicState> = graphics
        .dynamic_states
        .iter()
        .map(|&state| state.into())
        .collect();
    let dynamic_strides = dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
    let input = &pipeline.vertex_input;
    let bindings = input
        .bindings
        .iter()
        .map(|b| {
            let stride = match b.stride {
                VertexStride::Static(stride) => stride,
                // The bind's stride is dynamic state: the static one is ignored.
                _ if dynamic_strides => 0,
                _ => {
                    return Err(GraphicsError::Unmodelled(format!(
                        "vertex binding {} has a per-bind stride the pipeline keeps static \
                         (a pipeline per stride)",
                        b.binding
                    )))
                }
            };
            Ok(vk::VertexInputBindingDescription::default()
                .binding(b.binding)
                .stride(stride)
                .input_rate(b.input_rate.into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR> = input
        .bindings
        .iter()
        .filter(|b| b.divisor != 1)
        .map(|b| vk::VertexInputBindingDivisorDescriptionKHR {
            binding: b.binding,
            divisor: b.divisor,
        })
        .collect();
    let attributes: Vec<vk::VertexInputAttributeDescription> = input
        .attributes
        .iter()
        .map(|a| {
            vk::VertexInputAttributeDescription::default()
                .location(a.location)
                .binding(a.binding)
                .format(a.format.into())
                .offset(a.offset)
        })
        .collect();
    let entries = pipeline
        .stages
        .iter()
        .map(|stage| CString::new(stage.entry_point.as_str()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GraphicsError::Unmodelled("an entry point name contains NUL".into()))?;
    if consumer.modules.len() != pipeline.stages.len()
        || consumer.specializations.len() != pipeline.stages.len()
    {
        return Err(GraphicsError::Unmodelled(
            "a module and a specialization per stage".into(),
        ));
    }
    let spec_infos: Vec<vk::SpecializationInfo<'_>> = consumer
        .specializations
        .iter()
        .map(|(map, data)| {
            vk::SpecializationInfo::default()
                .map_entries(map)
                .data(data)
        })
        .collect();
    let mut subgroups: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>> = pipeline
        .stages
        .iter()
        .map(|stage| {
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(stage.required_subgroup_size.unwrap_or(0))
        })
        .collect();
    let mut stages = Vec::with_capacity(pipeline.stages.len());
    let topology = metal2vulkan::vk::PrimitiveTopology::from_raw(consumer.topology.as_raw());
    // Of a pipeline's barycentric geometry stages, only the draw's class's runs.
    for (at, (((stage, module), (entry, spec)), subgroup)) in pipeline
        .stages
        .iter()
        .zip(consumer.modules)
        .zip(entries.iter().zip(&spec_infos))
        .zip(subgroups.iter_mut())
        .enumerate()
    {
        if !pipeline.runs_stage(at, topology) {
            continue;
        }
        let mut info = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage.stage.into())
            .module(*module)
            .name(entry)
            .flags(stage.stage_flags.into());
        if spec.map_entry_count != 0 {
            info = info.specialization_info(spec);
        }
        if stage.required_subgroup_size.is_some() {
            info = info.push_next(subgroup);
        }
        stages.push(info);
    }
    let mut divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
        .vertex_binding_divisors(&divisors);
    let mut vertex = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    if !divisors.is_empty() {
        vertex = vertex.push_next(&mut divisor_state);
    }
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(consumer.topology)
        .primitive_restart_enable(consumer.primitive_restart);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .rasterizer_discard_enable(graphics.rasterizer_discard)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(consumer.cull_mode)
        .front_face(consumer.front_face)
        .line_width(1.0);
    let mut multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(graphics.rasterization_samples.into())
        // Metal's once-per-covered-sample invocations of a function that writes memory; the minimum
        // is ignored without sample shading, and stays Vulkan's default there.
        .sample_shading_enable(graphics.sample_shading)
        .min_sample_shading(if graphics.sample_shading { 1.0 } else { 0.0 })
        .alpha_to_coverage_enable(graphics.alpha_to_coverage)
        .alpha_to_one_enable(graphics.alpha_to_one);
    // The descriptor's private sample mask, one word.
    let sample_mask = graphics.sample_mask.map(|mask| [mask]);
    if let Some(mask) = &sample_mask {
        multisample = multisample.sample_mask(mask);
    }
    // An absent attachment's element is the artifact's too: identical to the others on a device
    // without `independentBlend`.
    let blends: Vec<vk::PipelineColorBlendAttachmentState> = graphics
        .color_blend_attachments()
        .into_iter()
        .map(|(write_mask, state)| {
            let blend = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(write_mask.into());
            match state {
                None => blend,
                Some(b) => blend
                    .blend_enable(true)
                    .src_color_blend_factor(b.src_color.into())
                    .dst_color_blend_factor(b.dst_color.into())
                    .color_blend_op(b.color_op.into())
                    .src_alpha_blend_factor(b.src_alpha.into())
                    .dst_alpha_blend_factor(b.dst_alpha.into())
                    .alpha_blend_op(b.alpha_op.into()),
            }
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .flags(color_blend_flags(graphics))
        .attachments(&blends);
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    // A tessellation pipeline's patches are its control points.
    let tessellation = graphics.tessellation.map(|t| {
        vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(t.patch_control_points)
    });
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&consumer.depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic_state)
        .layout(consumer.layout)
        .render_pass(consumer.render_pass)
        .subpass(0);
    if let Some(tessellation) = &tessellation {
        info = info.tessellation_state(tessellation);
    }
    let expected = facts
        .iter()
        .flat_map(|fact| {
            fact.pipeline_creation_diagnostics(&graphics.dynamic_states, topology, &[])
        })
        .collect();
    let clock = crate::budgets::CreationClock::start();
    let created = super::device::expecting_diagnostics(expected, || {
        // SAFETY: every pointer in `info` outlives the call; the caller destroys the result.
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
    });
    let elapsed = clock.read();
    match created {
        Ok(pipelines) => Ok((pipelines.first().copied().unwrap_or_default(), elapsed)),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                // SAFETY: a pipeline the failed call still returned is destroyed once, unused.
                unsafe { device.destroy_pipeline(pipeline, None) };
            }
            Err(GraphicsError::Refused(format!(
                "the driver refused the graphics pipeline: {error}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depth/stencil format the device does not attach is refused, typed, as not supported;
    /// one it attaches is used.
    #[test]
    fn a_depth_stencil_format_the_device_does_not_attach_is_not_supported() {
        let attaches = vk::FormatProperties::default()
            .optimal_tiling_features(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
        assert!(attachable(vk::Format::D32_SFLOAT_S8_UINT, attaches)
            .is_ok_and(|format| format == vk::Format::D32_SFLOAT_S8_UINT));
        let sampled_only = vk::FormatProperties::default()
            .optimal_tiling_features(vk::FormatFeatureFlags::SAMPLED_IMAGE);
        match attachable(vk::Format::D32_SFLOAT_S8_UINT, sampled_only) {
            Err(GraphicsError::NotSupported(why)) => {
                assert!(why.contains("D32_SFLOAT_S8_UINT"), "{why}")
            }
            Err(other) => panic!("{other}"),
            Ok(_) => panic!("a format the device does not attach was used"),
        }
    }

    /// The derived states map to the Vulkan state the executor created before an authored state
    /// existed: a `[[stencil]]` output always passes and replaces every bit under reference 0, a
    /// depth output tests with its comparison and writes, and no output tests nothing, all under
    /// clockwise front faces. An authored state keeps each face's ops, masks, and reference, and
    /// its winding. (`ash` types are compared by raw value.)
    #[test]
    fn the_depth_stencil_state_is_the_products_mapping_of_the_oracles() {
        use crate::case::{
            CompareFunctionLiteral as C, DepthStencilStateLiteral, StencilFaceLiteral,
            StencilOperationLiteral as Op, WindingLiteral,
        };
        use crate::executor_contract::{DepthCompare, DepthStencilTest};
        // (depth test, depth write, depth compare, stencil test)
        let tests = |s: &vk::PipelineDepthStencilStateCreateInfo| {
            (
                s.depth_test_enable,
                s.depth_write_enable,
                s.depth_compare_op.as_raw(),
                s.stencil_test_enable,
            )
        };
        // (compare, fail, pass, depth fail, compare mask, write mask, reference)
        let face = |f: vk::StencilOpState| {
            (
                f.compare_op.as_raw(),
                f.fail_op.as_raw(),
                f.pass_op.as_raw(),
                f.depth_fail_op.as_raw(),
                f.compare_mask,
                f.write_mask,
                f.reference,
            )
        };
        let derived =
            |depth, stencil| DrawDepthStencil::derived(DepthStencilTest { depth, stencil });
        let both = derived(Some(DepthCompare::Less), true);
        let state = metal_depth_stencil(&both);
        assert_eq!(
            tests(&state),
            (vk::TRUE, vk::TRUE, vk::CompareOp::LESS.as_raw(), vk::TRUE)
        );
        let replace = (
            vk::CompareOp::ALWAYS.as_raw(),
            vk::StencilOp::KEEP.as_raw(),
            vk::StencilOp::REPLACE.as_raw(),
            vk::StencilOp::KEEP.as_raw(),
            u32::MAX,
            u32::MAX,
            0,
        );
        assert_eq!(face(state.front), replace);
        assert_eq!(face(state.back), replace);
        assert!(front_face(&both) == METAL_DEFAULT_FRONT_FACE);
        assert_eq!(
            tests(&metal_depth_stencil(&derived(
                Some(DepthCompare::Always),
                false
            ))),
            (
                vk::TRUE,
                vk::TRUE,
                vk::CompareOp::ALWAYS.as_raw(),
                vk::FALSE
            )
        );
        assert_eq!(
            tests(&metal_default_depth_stencil()),
            (
                vk::FALSE,
                vk::FALSE,
                vk::CompareOp::ALWAYS.as_raw(),
                vk::FALSE
            )
        );

        let literal = |compare, pass, read_mask, write_mask| StencilFaceLiteral {
            compare,
            stencil_failure: Op::Zero,
            depth_failure: Op::Keep,
            depth_stencil_pass: pass,
            read_mask,
            write_mask,
        };
        let authored = DrawDepthStencil::authored(&DepthStencilStateLiteral {
            depth_compare: C::Always,
            depth_write: false,
            front: literal(C::NotEqual, Op::DecrementClamp, 0x3C, 0x0F),
            back: literal(C::Less, Op::Invert, 0xFF, 0x81),
            front_reference: 7,
            back_reference: 0xC0,
            front_facing_winding: WindingLiteral::CounterClockwise,
        });
        let state = metal_depth_stencil(&authored);
        assert_eq!(
            tests(&state),
            (
                vk::FALSE,
                vk::FALSE,
                vk::CompareOp::ALWAYS.as_raw(),
                vk::TRUE
            )
        );
        assert_eq!(
            face(state.front),
            (
                vk::CompareOp::NOT_EQUAL.as_raw(),
                vk::StencilOp::ZERO.as_raw(),
                vk::StencilOp::DECREMENT_AND_CLAMP.as_raw(),
                vk::StencilOp::KEEP.as_raw(),
                0x3C,
                0x0F,
                7
            )
        );
        assert_eq!(
            face(state.back),
            (
                vk::CompareOp::LESS.as_raw(),
                vk::StencilOp::ZERO.as_raw(),
                vk::StencilOp::INVERT.as_raw(),
                vk::StencilOp::KEEP.as_raw(),
                0xFF,
                0x81,
                0xC0
            )
        );
        assert!(front_face(&authored) == vk::FrontFace::COUNTER_CLOCKWISE);
    }
}
