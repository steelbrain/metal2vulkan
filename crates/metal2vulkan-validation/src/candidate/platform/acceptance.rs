//! Driver acceptance (L4) of a render row: create its pipeline from the binder layout its
//! artifact carries, time `vkCreateGraphicsPipelines`, and destroy everything. Nothing is bound,
//! recorded, or submitted, so no shader executes. The context runs under
//! `VK_LAYER_KHRONOS_validation`; any message it reports makes the row
//! [`Acceptance::InvalidUsage`]. Its layouts are [`LayoutObjects`], what the device lacks is
//! [`unmet_requirements`], and a render pipeline (a rasterization-disabled vertex function or a
//! rasterizing draw) is created by [`super::graphics_pipeline`]: its render pass from the
//! graphics state (color, framebuffer-fetch, and depth/stencil attachments), its vertex input
//! (divisors, dynamic strides), and every dynamic state it declares, at each stage's default
//! specialization, the compiled topology, and Metal's default depth-stencil state. A kernel row
//! is made through the runtime instead (`runtime_acceptance`).

use super::device::{with_validated_context, VulkanContext};
use super::graphics_pipeline::{self, GraphicsError};
use crate::acceptance::{Acceptance, AcceptanceError};
use crate::gpu::DeviceType;
use ash::vk;
use metal2vulkan::device::Requirements;
use metal2vulkan::pipeline::{PipelineLayout, RenderPipeline, StageModule};

/// The descriptor-set layouts (with their immutable samplers) and the pipeline layout of one
/// binder `PipelineLayout`, destroyed on every exit path.
pub(super) struct LayoutObjects<'a> {
    device: &'a ash::Device,
    samplers: Vec<vk::Sampler>,
    pub(super) set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(super) pipeline_layout: vk::PipelineLayout,
}

impl Drop for LayoutObjects<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this attempt, is destroyed once, and
        // no pipeline or submission that uses it outlives the owner (callers drop their
        // pipelines and wait for their submissions first).
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for layout in &self.set_layouts {
                self.device.destroy_descriptor_set_layout(*layout, None);
            }
            for sampler in &self.samplers {
                self.device.destroy_sampler(*sampler, None);
            }
        }
    }
}

/// Every object one acceptance attempt creates beyond the layout, destroyed on every exit path
/// (before the layout, a field declared after them).
struct Objects<'a> {
    device: &'a ash::Device,
    modules: Vec<vk::ShaderModule>,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    layout: LayoutObjects<'a>,
}

impl Drop for Objects<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this attempt, is destroyed once, and
        // nothing was submitted that could still use it.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for module in &self.modules {
                self.device.destroy_shader_module(*module, None);
            }
        }
    }
}

/// Create `pipeline` on `device` from its binder layout and report what the driver did.
///
/// Only opening or using the context (the implementation, the hang tripwire, a missing
/// validation layer) is [`AcceptanceError::DeviceUnavailable`]: `create` never fails
/// through `with_validated_context`'s error.
pub fn create_acceptance_pipeline(
    device: DeviceType,
    pipeline: &RenderPipeline,
) -> Result<Acceptance, AcceptanceError> {
    with_validated_context(device, |context| {
        let created = create(context, pipeline);
        // Every object of the attempt is destroyed by now, so destruction reports here too.
        let messages = context.take_messages();
        Ok(match created {
            Ok(_) if !messages.is_empty() => Ok(Acceptance::InvalidUsage(messages)),
            other => other,
        })
    })
    .map_err(AcceptanceError::DeviceUnavailable)?
}

/// The `Device` the context `device_type` opens, as `corpus-pipeline` compiles a row for a
/// committed profile: its queried description restricted to what it enabled
/// (`DeviceDescription::with_enabled`), under the default policy. What a device test compiles a
/// render row for, so its profile is the opened device's, never a committed one.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub fn opened_worker_device(
    device_type: DeviceType,
) -> Result<crate::worker::compile::WorkerDevice, crate::check::admission::ExecutorError> {
    use crate::check::admission::ExecutorError;
    with_validated_context(device_type, |context| {
        Ok(context
            .opened_description
            .clone()
            .map_err(ExecutorError::Failed)
            .and_then(|description| {
                crate::worker::compile::WorkerDevice::new(
                    description,
                    metal2vulkan::device::DevicePolicy::default(),
                )
                .map_err(|error| crate::candidate::engine::product_refusal(&error))
            }))
    })
    .map_err(ExecutorError::Failed)?
}

/// What of `requirements` the context did not enable or its device does not provide, one name
/// each: the API version, features, and extensions against what the context enabled
/// (`acceptance::unenabled` on [`VulkanContext::opened_description`]); properties, limits, formats,
/// stage flags, and float-controls independence against the opened device's queried profile, never
/// against a committed profile; and every robustness feature the device's description states that
/// the context could not enable (`VulkanContext::unmet_robustness`). The requirements are the
/// product's own, which it derives from the module it emits (the old executor's SPIR-V scan, which
/// read the module's declarations again in the old engine's vocabulary, is retired).
pub(super) fn unmet_requirements(
    context: &VulkanContext,
    requirements: &Requirements,
) -> Vec<String> {
    let mut missing = crate::acceptance::unenabled(&context.opened_description, requirements);
    // The robust encodings compile only where the device enables what its description states; a
    // context that could not enable one runs nothing.
    missing.extend(
        context.unmet_robustness.iter().map(|feature| {
            format!("{feature:?} (stated by the device's description, not enabled)")
        }),
    );
    // A stage flag's feature is both an enable and a device fact: named once.
    for unmet in crate::acceptance::unmet_device_facts(&context.opened, requirements) {
        if !missing.contains(&unmet) {
            missing.push(unmet);
        }
    }
    missing
}

fn create(
    context: &VulkanContext,
    pipeline: &RenderPipeline,
) -> Result<Acceptance, AcceptanceError> {
    let stages = &pipeline.stages;
    let layout = &pipeline.layout;
    let missing = unmet_requirements(context, &pipeline.requirements);
    if !missing.is_empty() {
        return Ok(Acceptance::NotEnabled(missing));
    }
    let layout = match LayoutObjects::create(&context.device, layout) {
        Ok(layout) => layout,
        Err(LayoutError::Shape(why)) => return Err(AcceptanceError::Tooling(why)),
        Err(LayoutError::Refused(refused)) => return Ok(Acceptance::Refused(refused)),
    };
    let mut objects = Objects {
        device: &context.device,
        modules: Vec::new(),
        render_pass: vk::RenderPass::null(),
        pipeline: vk::Pipeline::null(),
        layout,
    };
    for stage in stages {
        let info = vk::ShaderModuleCreateInfo::default().code(&stage.spirv);
        // SAFETY: `info` points at live SPIR-V words for the duration of the call.
        match unsafe { context.device.create_shader_module(&info, None) } {
            Ok(module) => objects.modules.push(module),
            Err(error) => {
                return Ok(Acceptance::Refused(format!(
                    "create shader module: {error}"
                )))
            }
        }
    }
    let specializations: Vec<_> = stages.iter().map(default_specialization).collect();
    let unmodelled_or_refused = |error: GraphicsError| match error {
        GraphicsError::Unmodelled(why) => Err(AcceptanceError::Unmodelled(why)),
        GraphicsError::NotSupported(why) => Ok(Acceptance::NotEnabled(vec![why])),
        GraphicsError::Refused(why) => Ok(Acceptance::Refused(why)),
    };
    if let Err(error) = graphics_pipeline::attachable_depth_stencil_format(
        &context.instance,
        context.physical,
        &pipeline.graphics,
    ) {
        return unmodelled_or_refused(error);
    }
    objects.render_pass =
        match graphics_pipeline::create_render_pass(&context.device, &pipeline.graphics) {
            Ok(render_pass) => render_pass,
            Err(error) => return unmodelled_or_refused(error),
        };
    let created = graphics_pipeline::create_graphics_pipeline(
        &context.device,
        context.executes_moltenvk(),
        pipeline,
        &graphics_pipeline::Consumer {
            modules: &objects.modules,
            specializations: &specializations,
            layout: objects.layout.pipeline_layout,
            render_pass: objects.render_pass,
            topology: pipeline.graphics.topology.into(),
            // The class's list topology, which Metal never restarts.
            primitive_restart: false,
            depth_stencil: graphics_pipeline::metal_default_depth_stencil(),
            // The encoder's defaults: no culling, clockwise front faces.
            cull_mode: vk::CullModeFlags::NONE,
            front_face: super::METAL_DEFAULT_FRONT_FACE,
        },
    );
    match created {
        Ok((created, create)) => {
            objects.pipeline = created;
            Ok(Acceptance::Created { create })
        }
        Err(error) => unmodelled_or_refused(error),
    }
}

/// A render stage's specialization at its defaults: its map and data, every constant at its
/// default value, packed in id order.
fn default_specialization(stage: &StageModule) -> (Vec<vk::SpecializationMapEntry>, Vec<u8>) {
    let mut map = Vec::new();
    let mut data = Vec::new();
    for constant in &stage.specialization.constants {
        let size = constant.size.min(8) as usize;
        map.push(
            vk::SpecializationMapEntry::default()
                .constant_id(constant.id)
                .offset(data.len() as u32)
                .size(size),
        );
        data.extend_from_slice(&constant.default.to_le_bytes()[..size]);
    }
    (map, data)
}

/// Why [`LayoutObjects::create`] made no layout.
pub(super) enum LayoutError {
    /// The binder layout's sets are not at their own indices: a layout-shape defect of the tool
    /// or binder, not something the driver refused.
    Shape(String),
    /// The driver refused an object.
    Refused(String),
}

impl<'a> LayoutObjects<'a> {
    /// The descriptor-set layouts (with immutable samplers and binding flags) and the pipeline
    /// layout of `layout`.
    pub(super) fn create(
        device: &'a ash::Device,
        layout: &PipelineLayout,
    ) -> Result<LayoutObjects<'a>, LayoutError> {
        // Vulkan numbers sets by position.
        if let Some((index, set)) = layout
            .sets
            .iter()
            .enumerate()
            .find(|(index, set)| set.set as usize != *index)
        {
            return Err(LayoutError::Shape(format!(
                "layout set {} is not at index {index}",
                set.set
            )));
        }
        let mut objects = LayoutObjects {
            device,
            samplers: Vec::new(),
            set_layouts: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
        };
        create_layouts(&mut objects, layout).map_err(LayoutError::Refused)?;
        Ok(objects)
    }
}

/// Fill `objects` with the layouts of `layout`, whose sets are at their own indices. An error is
/// the driver refusing an object.
fn create_layouts(objects: &mut LayoutObjects<'_>, layout: &PipelineLayout) -> Result<(), String> {
    for set in &layout.sets {
        let mut immutable: Vec<Vec<vk::Sampler>> = Vec::with_capacity(set.bindings.len());
        for binding in &set.bindings {
            let mut samplers = Vec::new();
            if let Some(desc) = &binding.immutable_sampler {
                let mut reduction = vk::SamplerReductionModeCreateInfo::default();
                let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
                let info = metal2vulkan::ash_interop::sampler_create_info(
                    desc,
                    &mut reduction,
                    &mut border,
                );
                // SAFETY: `info` and its chain are live for the call.
                let sampler = unsafe { objects.device.create_sampler(&info, None) }
                    .map_err(|error| format!("create immutable sampler: {error}"))?;
                objects.samplers.push(sampler);
                samplers = vec![sampler; binding.count as usize];
            }
            immutable.push(samplers);
        }
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = set
            .bindings
            .iter()
            .zip(&immutable)
            .map(|(binding, samplers)| {
                let info = vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(binding.ty.into())
                    .descriptor_count(binding.count)
                    .stage_flags(binding.stages.into());
                if samplers.is_empty() {
                    info
                } else {
                    info.immutable_samplers(samplers)
                }
            })
            .collect();
        let flags: Vec<vk::DescriptorBindingFlags> =
            set.bindings.iter().map(|b| b.flags.into()).collect();
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let mut info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(set.flags.into())
            .bindings(&bindings);
        if flags.iter().any(|f| !f.is_empty()) {
            info = info.push_next(&mut flags_info);
        }
        // SAFETY: `info` and every array it points at are live for the call.
        let created = unsafe { objects.device.create_descriptor_set_layout(&info, None) }
            .map_err(|error| format!("create descriptor set layout {}: {error}", set.set))?;
        objects.set_layouts.push(created);
    }
    let push: Vec<vk::PushConstantRange> = layout
        .push_constants
        .iter()
        .map(|range| {
            vk::PushConstantRange::default()
                .stage_flags(range.stages.into())
                .offset(range.offset)
                .size(range.size)
        })
        .collect();
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&objects.set_layouts)
        .push_constant_ranges(&push);
    // SAFETY: `info` and its arrays are live for the call.
    objects.pipeline_layout = unsafe { objects.device.create_pipeline_layout(&info, None) }
        .map_err(|error| format!("create pipeline layout: {error}"))?;
    Ok(())
}
