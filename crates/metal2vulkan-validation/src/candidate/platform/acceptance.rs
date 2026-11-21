//! Driver acceptance (L4): create an engine pipeline from the binder layout its artifact
//! carries, time `vkCreate*Pipelines`, and destroy everything. Nothing is bound, recorded, or
//! submitted, so no shader executes. The context runs under `VK_LAYER_KHRONOS_validation`; any
//! message it reports makes the row [`Acceptance::InvalidUsage`]. The layout half
//! ([`LayoutObjects`], [`unmet_requirements`]) is shared with the engine executors, and a
//! render pipeline (a rasterization-disabled vertex function or a rasterizing draw) is created by
//! the render executor's own builder ([`super::graphics_pipeline`]): its render pass from the
//! graphics state (color, framebuffer-fetch, and depth/stencil attachments), its vertex input
//! (divisors, dynamic strides), and every dynamic state it declares, at each stage's default
//! specialization, the compiled topology, and Metal's default depth-stencil state. A compute
//! pipeline is created from the dispatch region the binder plans for the row's representative
//! dispatch (its specialization, stage flags, and required subgroup size), as the executors create
//! each region's pipeline, never at the stage's placeholder defaults.

use super::device::{with_validated_context, VulkanContext};
use super::graphics_pipeline::{self, GraphicsError};
use crate::acceptance::{Acceptance, AcceptanceError, RowPipeline};
use crate::gpu::DeviceType;
use ash::vk;
use metal2vulkan::bind::DispatchRegion;
use metal2vulkan::device::Requirements;
use metal2vulkan::pipeline::{PipelineLayout, StageModule};
use std::ffi::CString;

/// The descriptor-set layouts (with their immutable samplers) and the pipeline layout of one
/// binder `PipelineLayout`, destroyed on every exit path, and what a pool that allocates one set
/// of each layout needs ([`LayoutObjects::pool_sizes`], [`LayoutObjects::pool_flags`]).
pub(super) struct LayoutObjects<'a> {
    device: &'a ash::Device,
    samplers: Vec<vk::Sampler>,
    pub(super) set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(super) pipeline_layout: vk::PipelineLayout,
    /// The pool sizes and flags of [`pool_requirements`].
    pub(super) pool_sizes: Vec<vk::DescriptorPoolSize>,
    pub(super) pool_flags: vk::DescriptorPoolCreateFlags,
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
    pipeline: &RowPipeline,
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

fn create(context: &VulkanContext, pipeline: &RowPipeline) -> Result<Acceptance, AcceptanceError> {
    let stages = pipeline.stages();
    let layout = match pipeline {
        RowPipeline::Compute { pipeline, .. } => &pipeline.layout,
        RowPipeline::Render { pipeline, .. } => &pipeline.layout,
    };
    let missing = unmet_requirements(context, pipeline.requirements());
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
    let entries = stages
        .iter()
        .map(|stage| {
            CString::new(stage.entry_point.as_str())
                .map_err(|_| AcceptanceError::Tooling("entry point name contains NUL".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let created = stage_creations(pipeline);
    let specializations: Vec<(Vec<vk::SpecializationMapEntry>, Vec<u8>)> = created
        .iter()
        .map(|stage| (stage.map.clone(), stage.data.clone()))
        .collect();
    let spec_infos: Vec<vk::SpecializationInfo<'_>> = specializations
        .iter()
        .map(|(map, data)| {
            vk::SpecializationInfo::default()
                .map_entries(map)
                .data(data)
        })
        .collect();
    let mut subgroup_sizes: Vec<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'_>> =
        created
            .iter()
            .map(|stage| {
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                    .required_subgroup_size(stage.required_subgroup_size.unwrap_or(0))
            })
            .collect();
    let mut stage_infos = Vec::with_capacity(stages.len());
    for ((((stage, module), entry), (spec, subgroup)), creation) in stages
        .iter()
        .zip(&objects.modules)
        .zip(&entries)
        .zip(spec_infos.iter().zip(subgroup_sizes.iter_mut()))
        .zip(&created)
    {
        let mut info = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage.stage.into())
            .module(*module)
            .name(entry)
            .flags(creation.flags);
        if !creation.map.is_empty() {
            info = info.specialization_info(spec);
        }
        if creation.required_subgroup_size.is_some() {
            info = info.push_next(subgroup);
        }
        stage_infos.push(info);
    }
    let result = match pipeline {
        RowPipeline::Compute { .. } => {
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_infos[0])
                .layout(objects.layout.pipeline_layout);
            let clock = crate::budgets::CreationClock::start();
            // SAFETY: every handle in `info` is live; the result is destroyed with `objects`.
            let created = unsafe {
                context
                    .device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            };
            (created.map_err(|(_, error)| error), clock.read())
        }
        RowPipeline::Render { pipeline: p, .. } => {
            let unmodelled_or_refused = |error: GraphicsError| match error {
                GraphicsError::Unmodelled(why) => Err(AcceptanceError::Unmodelled(why)),
                GraphicsError::NotSupported(why) => Ok(Acceptance::NotEnabled(vec![why])),
                GraphicsError::Refused(why) => Ok(Acceptance::Refused(why)),
            };
            if let Err(error) = graphics_pipeline::attachable_depth_stencil_format(
                &context.instance,
                context.physical,
                &p.graphics,
            ) {
                return unmodelled_or_refused(error);
            }
            objects.render_pass =
                match graphics_pipeline::create_render_pass(&context.device, &p.graphics) {
                    Ok(render_pass) => render_pass,
                    Err(error) => return unmodelled_or_refused(error),
                };
            let created = graphics_pipeline::create_graphics_pipeline(
                &context.device,
                context.executes_moltenvk(),
                p,
                &graphics_pipeline::Consumer {
                    modules: &objects.modules,
                    specializations: &specializations,
                    layout: objects.layout.pipeline_layout,
                    render_pass: objects.render_pass,
                    topology: p.graphics.topology.into(),
                    // The class's list topology, which Metal never restarts.
                    primitive_restart: false,
                    depth_stencil: graphics_pipeline::metal_default_depth_stencil(),
                    // The encoder's defaults: no culling, clockwise front faces.
                    cull_mode: vk::CullModeFlags::NONE,
                    front_face: super::METAL_DEFAULT_FRONT_FACE,
                },
            );
            return match created {
                Ok((created, create)) => {
                    objects.pipeline = created;
                    Ok(Acceptance::Created { create })
                }
                Err(error) => unmodelled_or_refused(error),
            };
        }
    };
    match result {
        (Ok(pipelines), create) => {
            objects.pipeline = pipelines.first().copied().unwrap_or_default();
            Ok(Acceptance::Created { create })
        }
        (Err(error), _) => Ok(Acceptance::Refused(format!("create pipeline: {error}"))),
    }
}

/// How each stage of `pipeline` is created: a compute row's one stage from the region the binder
/// plans for its representative dispatch, as the executors create each region's pipeline, never at
/// the stage's placeholder defaults (with which a stage whose simd tier requires full subgroups has
/// a local size of 1, `VUID-VkPipelineShaderStageCreateInfo-flags-02759`); a render row's stages at
/// their defaults.
fn stage_creations(pipeline: &RowPipeline) -> Vec<StageCreation> {
    match pipeline {
        RowPipeline::Compute { creation, .. } => vec![StageCreation::of_region(&creation.region)],
        RowPipeline::Render { .. } => pipeline
            .stages()
            .iter()
            .map(StageCreation::defaults)
            .collect(),
    }
}

/// How one stage is created: its specialization map and data, its create flags, and the subgroup
/// size it requires.
struct StageCreation {
    map: Vec<vk::SpecializationMapEntry>,
    data: Vec<u8>,
    flags: vk::PipelineShaderStageCreateFlags,
    required_subgroup_size: Option<u32>,
}

impl StageCreation {
    /// A compute stage as the binder's dispatch `region` creates it: the region's specialization
    /// (the plan's constants, the local size, `MultiRegion`), stage flags, and required subgroup
    /// size.
    fn of_region(region: &DispatchRegion) -> StageCreation {
        StageCreation {
            map: region
                .specialization
                .entries
                .iter()
                .map(|&entry| entry.into())
                .collect(),
            data: region.specialization.data.clone(),
            flags: region.stage_flags.into(),
            required_subgroup_size: region.required_subgroup_size,
        }
    }

    /// A render stage at its defaults: every constant at its default value, packed in id order,
    /// with the stage's own flags and subgroup size.
    fn defaults(stage: &StageModule) -> StageCreation {
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
        StageCreation {
            map,
            data,
            flags: stage.stage_flags.into(),
            required_subgroup_size: stage.required_subgroup_size,
        }
    }
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
        let (pool_sizes, pool_flags) = pool_requirements(layout).map_err(LayoutError::Shape)?;
        let mut objects = LayoutObjects {
            device,
            samplers: Vec::new(),
            set_layouts: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
            pool_sizes,
            pool_flags,
        };
        create_layouts(&mut objects, layout).map_err(LayoutError::Refused)?;
        Ok(objects)
    }
}

/// What a descriptor pool that allocates one set of each of `layout`'s sets needs: every
/// binding's descriptor count summed per descriptor type (types in ascending raw order, none
/// empty), since a set consumes each binding's whole count whether the plan writes it or not (a
/// heap array's capacity, an immutable heap sampler); and `UPDATE_AFTER_BIND` when some set layout
/// is `UPDATE_AFTER_BIND_POOL`, which only such a pool allocates
/// (VUID-VkDescriptorSetAllocateInfo-pSetLayouts-03044). A count that overflows is an error.
pub(super) fn pool_requirements(
    layout: &PipelineLayout,
) -> Result<(Vec<vk::DescriptorPoolSize>, vk::DescriptorPoolCreateFlags), String> {
    let mut counts: std::collections::BTreeMap<i32, u32> = std::collections::BTreeMap::new();
    let mut flags = vk::DescriptorPoolCreateFlags::empty();
    for set in &layout.sets {
        let set_flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
        if set_flags.contains(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL) {
            flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
        }
        for binding in &set.bindings {
            let ty: vk::DescriptorType = binding.ty.into();
            let count = counts.entry(ty.as_raw()).or_default();
            *count = count
                .checked_add(binding.count)
                .ok_or("the layout's descriptor count overflows")?;
        }
    }
    let sizes = counts
        .into_iter()
        .filter(|&(_, count)| count > 0)
        .map(|(ty, count)| {
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::from_raw(ty))
                .descriptor_count(count)
        })
        .collect();
    Ok((sizes, flags))
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

#[cfg(test)]
mod tests {
    use super::{pool_requirements, stage_creations, StageCreation};
    use crate::acceptance::{compile_row, Projection, RowCompile, RowPipeline, RowStage};
    use crate::candidate::engine::tests::{assemble, counting_kernel};
    use ash::vk;
    use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{
        BindingSource, ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, HandleClass,
        ImageAccess, SampledType, StageSource,
    };
    use metal2vulkan::Budget;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    fn compile(device: &Device, bitcode: &[u8], entry: &str) -> ComputePipeline {
        let function = Library::from_air(bitcode)
            .and_then(|library| library.function(entry))
            .unwrap();
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        device
            .compile_compute_pipeline(&descriptor, &Budget::standard())
            .unwrap()
    }

    /// A pool sized from the plan's writes cannot allocate a heap pipeline's sets (set 1's arrays
    /// take the layout's whole capacity, most elements unwritten, and only an update-after-bind
    /// pool allocates it). The pool is the layout's: on `moltenvk-m3`, `argument_buffer_read` of
    /// the owned `handles` fixture needs the sum of every binding's count by type, from an
    /// update-after-bind pool, its set-1 arrays each the heap capacity: the sampled float group's
    /// and its unsigned integer twin's (the class a float read reaches a raw member through) alike.
    /// A kernel without a heap needs a plain pool. The heap arrays are the resident-table
    /// domains' handle table, which the executor's own context compiles under.
    #[test]
    fn descriptor_pools_hold_every_binding_of_the_layout() {
        let path = crate::engine::profiles_dir().join("moltenvk-m3.json");
        let text = std::fs::read_to_string(path).unwrap();
        let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
        let device = Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).unwrap();
        let handles = match llvm_assemble(&engine_dir().join("handles.ll")) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected handles.ll: {why}"),
        };
        let heap = compile(&device, &handles, "argument_buffer_read");
        let (sizes, flags) = pool_requirements(&heap.layout).unwrap();
        assert!(flags == vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let mut expected = std::collections::BTreeMap::<i32, u32>::new();
        for binding in heap.layout.sets.iter().flat_map(|set| &set.bindings) {
            let ty: vk::DescriptorType = binding.ty.into();
            *expected.entry(ty.as_raw()).or_default() += binding.count;
        }
        let set1 = &heap.layout.sets[1].bindings;
        let capacity = set1[0].count;
        assert!(capacity > 1, "{capacity}");
        // Every heap array is the capacity: a sampled float group's twin is its group's.
        assert!(set1.iter().all(|b| b.count == capacity), "{set1:?}");
        let sampled = |ty: SampledType| {
            set1.iter().find(|b| match b.source {
                BindingSource::Heap(HandleClass::Image(c)) => {
                    c.access == ImageAccess::Sampled && !c.multisampled && c.sampled_type == ty
                }
                _ => false,
            })
        };
        let (Some(float), Some(twin)) = (sampled(SampledType::Float), sampled(SampledType::Uint))
        else {
            panic!("no sampled float group with its unsigned twin: {set1:?}");
        };
        let shape = |b: &metal2vulkan::pipeline::DescriptorBinding| match b.source {
            BindingSource::Heap(HandleClass::Image(c)) => (c.dim, c.arrayed),
            _ => unreachable!(),
        };
        assert_eq!(shape(float), shape(twin));
        assert_eq!(float.count, twin.count);
        // The sampled images are set 0's and the heap's arrays of them, each the capacity.
        let set0: u32 = heap.layout.sets[0]
            .bindings
            .iter()
            .filter(|b| vk::DescriptorType::from(b.ty) == vk::DescriptorType::SAMPLED_IMAGE)
            .map(|b| b.count)
            .sum();
        let arrays = set1
            .iter()
            .filter(|b| vk::DescriptorType::from(b.ty) == vk::DescriptorType::SAMPLED_IMAGE)
            .count() as u32;
        assert_eq!(
            expected.get(&vk::DescriptorType::SAMPLED_IMAGE.as_raw()),
            Some(&(set0 + arrays * capacity))
        );
        assert_eq!(
            sizes
                .iter()
                .map(|size| (size.ty.as_raw(), size.descriptor_count))
                .collect::<Vec<_>>(),
            expected.into_iter().collect::<Vec<_>>()
        );

        let Some(plain) = assemble(&counting_kernel()) else {
            return;
        };
        let plain = compile(&device, &plain, "count");
        let (sizes, flags) = pool_requirements(&plain.layout).unwrap();
        assert!(flags.is_empty());
        assert!(!sizes.is_empty());
        assert!(sizes.iter().all(|size| size.descriptor_count > 0));
    }

    /// A kernel row's one stage is created with exactly the specialization map and data, stage
    /// flags, and required subgroup size of the region the binder plans for its representative
    /// dispatch, which are not the stage's defaults: on `moltenvk-m3`, `simd_shuffle_k`'s simd tier
    /// requires full subgroups, which its defaults' local size of 1 violates
    /// (`VUID-VkPipelineShaderStageCreateInfo-flags-02759`), and `a4_imageblock_extent`'s region
    /// also sets its imageblock's extent. Printed per profile: whether the tier requires full
    /// subgroups.
    #[test]
    fn a_kernel_rows_stage_is_created_from_its_planned_region_not_its_defaults() {
        let full = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
        for profile in ["moltenvk-m3", "lavapipe"] {
            let device = crate::engine::load_device(profile).unwrap();
            for (file, entry) in [
                ("simd_ops.ll", "simd_shuffle_k"),
                ("a4_imageblock_extent.ll", "a4_imageblock_extent"),
            ] {
                let bitcode = match llvm_assemble(&engine_dir().join(file)) {
                    Ok(bitcode) => bitcode,
                    Err(AssembleError::Missing(why)) => {
                        eprintln!("SKIPPED: {why}");
                        return;
                    }
                    Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {file}: {why}"),
                };
                let compiled = compile_row(
                    &bitcode,
                    entry,
                    RowStage::Kernel,
                    Projection::Derived,
                    &device,
                    None,
                );
                let RowCompile::Compiled(row) = compiled else {
                    panic!("{profile} {entry}: {compiled:?}");
                };
                let created = stage_creations(&row);
                let RowPipeline::Compute { pipeline, creation } = &*row else {
                    panic!("{profile} {entry}: a kernel row is a compute pipeline");
                };
                let [stage] = created.as_slice() else {
                    panic!("{profile} {entry}: {} stages", created.len());
                };
                let region = &creation.region;
                eprintln!(
                    "{profile} {entry}: flags {:#x}, required subgroup size {:?}",
                    stage.flags.as_raw(),
                    stage.required_subgroup_size
                );
                assert_eq!(
                    stage
                        .map
                        .iter()
                        .map(|entry| (entry.constant_id, entry.offset, entry.size))
                        .collect::<Vec<_>>(),
                    region
                        .specialization
                        .entries
                        .iter()
                        .map(|&entry| {
                            let entry: vk::SpecializationMapEntry = entry.into();
                            (entry.constant_id, entry.offset, entry.size)
                        })
                        .collect::<Vec<_>>(),
                    "{profile} {entry}"
                );
                assert!(!stage.map.is_empty(), "{profile} {entry}");
                assert_eq!(stage.data, region.specialization.data, "{profile} {entry}");
                assert_eq!(
                    stage.flags.as_raw(),
                    vk::PipelineShaderStageCreateFlags::from(region.stage_flags).as_raw(),
                    "{profile} {entry}"
                );
                assert_eq!(
                    stage.required_subgroup_size, region.required_subgroup_size,
                    "{profile} {entry}"
                );
                // The defaults leave the local size (and the imageblock's extent) at their
                // placeholders, which the region specializes.
                let defaults = StageCreation::defaults(&pipeline.stage);
                assert_ne!(stage.data, defaults.data, "{profile} {entry}");
                if (profile, entry) == ("moltenvk-m3", "simd_shuffle_k") {
                    assert!(stage.flags.contains(full), "{:#x}", stage.flags.as_raw());
                    assert!(
                        defaults.flags.contains(full),
                        "{:#x}",
                        defaults.flags.as_raw()
                    );
                }
            }
        }
    }
}
