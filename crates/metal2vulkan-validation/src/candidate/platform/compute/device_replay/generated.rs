//! The device replay's generated-commands path in the engine compute executor: the consumer
//! side of the product's `replay_preprocess_generated` (`docs/API.md`, "With
//! `VK_EXT_device_generated_commands`").
//!
//! - every variant's pipeline is created with the product's shared pipeline layout
//!   (`GeneratedReplay::pipeline_layout`) and `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`,
//!   with its one region's specialization; the variants' descriptor sets are the first's (the
//!   product refuses variants that bind other descriptors), written once;
//! - an indirect execution set holds the pipelines in variant order (the first is its initial
//!   pipeline), and an indirect commands layout holds the product's three tokens at their offsets
//!   with its stride;
//! - each command of the range has preprocess memory of its own, sized by
//!   `vkGetGeneratedCommandsMemoryRequirementsEXT` for the most sequences its execution runs;
//! - after the preprocess and its barrier, the descriptor sets are bound once and, per command, a
//!   barrier (as K14c's between commands), the initial pipeline, zeroed push constants (the tokens
//!   write every byte), and `vkCmdExecuteGeneratedCommandsEXT` over its sequences with the count
//!   the preprocess wrote ([`record_executions`]);
//! - the commands layout is made for explicit preprocessing, every command's execution is
//!   preprocessed (`vkCmdPreprocessGeneratedCommandsEXT`) before the first runs, a barrier orders
//!   those writes before the executions, and each runs preprocessed: no preprocessing follows an
//!   execution, a form that runs right on every driver.

use super::*;
use crate::candidate::platform::generated_commands::{
    Commands, Execution, ExecutionSetCreateInfo, ExecutionSetPipelineInfo, ExecutionSetToken,
    GeneratedCommandsInfo, Handle, LayoutCreateInfo, LayoutToken, MemoryRequirementsInfo,
    PushConstantToken, WriteExecutionSetPipeline, EXECUTION_SET_PIPELINES, INDIRECT_BINDABLE,
    PREPROCESS_BUFFER, PREPROCESS_WRITE, TOKEN_DISPATCH, TOKEN_EXECUTION_SET, TOKEN_PUSH_CONSTANT,
};
pub(super) use crate::candidate::platform::generated_commands::{
    PREPROCESS_READ, PREPROCESS_STAGE,
};
use metal2vulkan::bind::GeneratedReplay;
use std::collections::btree_map::Entry;
use std::time::Duration;

/// The generated-commands path's objects for one command buffer's range. Its own handles (the
/// execution set and commands layout before the pipelines they name, then the modules and the
/// preprocess memory) are destroyed first, then the shared layout and the plan's resources
/// (fields), as a K14c variant's are.
pub(super) struct Generated<'c> {
    device: &'c ash::Device,
    commands: Commands,
    modules: Vec<vk::ShaderModule>,
    /// Each variant's pipeline, in variant order: the execution set's indices.
    pipelines: Vec<vk::Pipeline>,
    execution_set: Handle,
    commands_layout: Handle,
    /// The preprocess memory's buffer and allocation ([`Generated::reserve_preprocess`]).
    preprocess: Option<(vk::Buffer, vk::DeviceMemory)>,
    /// Each command's preprocess memory, by slot: its address and size (`(0, 0)` where the
    /// device needs none).
    reserved: Vec<(vk::DeviceAddress, vk::DeviceSize)>,
    sets: Vec<vk::DescriptorSet>,
    layout: LayoutObjects<'c>,
    pub(super) resources: PlanResources<'c>,
}

impl Drop for Generated<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this replay and is destroyed once; the
        // replay waited for its submission (under the hang protocol) before returning normally.
        unsafe {
            if self.execution_set != 0 {
                self.commands.destroy_execution_set(self.execution_set);
            }
            if self.commands_layout != 0 {
                self.commands.destroy_commands_layout(self.commands_layout);
            }
            for pipeline in &self.pipelines {
                self.device.destroy_pipeline(*pipeline, None);
            }
            for module in &self.modules {
                self.device.destroy_shader_module(*module, None);
            }
            if let Some((buffer, memory)) = self.preprocess {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
    }
}

fn fail(why: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::Failed(format!("the generated-commands replay: {why}"))
}

impl<'c> Generated<'c> {
    /// Set up the generated-commands path of the `prepared` variants (in their order) as the
    /// product's `replay` and `layout` state it: the shared layout, the first variant's
    /// descriptor sets, each variant's indirect-bindable pipeline, the execution set, and the
    /// indirect commands layout. The preprocess memory waits for
    /// [`reserve_preprocess`](Generated::reserve_preprocess).
    pub(super) fn create(
        context: &'c VulkanContext,
        device: &Device,
        commands: Commands,
        prepared: &[Prepared],
        replay: &GeneratedReplay,
        layout: &GeneratedLayout,
    ) -> Result<Generated<'c>, ExecutorError> {
        let first = prepared.first().ok_or_else(|| fail("no variant"))?;
        let shared = match LayoutObjects::create(&context.device, &replay.pipeline_layout) {
            Ok(shared) => shared,
            Err(LayoutError::Shape(why) | LayoutError::Refused(why)) => return Err(fail(why)),
        };
        // The variants bind alike (the product refuses others): the first's resources and sets
        // are every variant's.
        let resources = first.resources(context, device)?;
        let mut generated = Generated {
            device: &context.device,
            commands,
            modules: Vec::with_capacity(prepared.len()),
            pipelines: Vec::with_capacity(prepared.len()),
            execution_set: 0,
            commands_layout: 0,
            preprocess: None,
            reserved: Vec::new(),
            sets: Vec::new(),
            layout: shared,
            resources,
        };
        generated.sets = generated.resources.write_sets(context, &generated.layout)?;
        for variant in prepared {
            let [region] = variant.commands.regions.as_slice() else {
                return Err(variant.fail("not one replayed region".into()));
            };
            let module_info =
                vk::ShaderModuleCreateInfo::default().code(&variant.pipeline.stage.spirv);
            // SAFETY: `module_info` points at live SPIR-V words for the call.
            let module = unsafe { context.device.create_shader_module(&module_info, None) }
                .map_err(|error| format!("create shader module: {error}"))?;
            generated.modules.push(module);
            let (pipeline, took) = bindable_pipeline(
                context,
                module,
                generated.layout.pipeline_layout,
                &variant.pipeline,
                region,
            )?;
            generated.pipelines.push(pipeline);
            crate::budgets::creation_within_ceiling("the compute pipeline", took)?;
        }
        let count = u32::try_from(generated.pipelines.len()).map_err(fail)?;
        let pipeline_info = ExecutionSetPipelineInfo::new(generated.pipelines[0], count);
        // SAFETY: `pipeline_info` and the initial pipeline are live for the call.
        generated.execution_set =
            unsafe { commands.create_execution_set(&ExecutionSetCreateInfo::new(&pipeline_info)) }
                .map_err(|error| fail(format!("create the indirect execution set: {error}")))?;
        // The initial pipeline is index 0; the others go at their variant's index.
        let writes: Vec<WriteExecutionSetPipeline> = (1..count)
            .map(|index| WriteExecutionSetPipeline::new(index, generated.pipelines[index as usize]))
            .collect();
        if !writes.is_empty() {
            // SAFETY: nothing has used the set yet; every written pipeline is live.
            unsafe { commands.update_execution_set(generated.execution_set, &writes) };
        }
        let set_token = ExecutionSetToken {
            kind: EXECUTION_SET_PIPELINES,
            shader_stages: vk::ShaderStageFlags::COMPUTE,
        };
        let push_token = PushConstantToken {
            update_range: vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: layout.push_bytes,
            },
        };
        let tokens = [
            LayoutToken::new(
                TOKEN_EXECUTION_SET,
                std::ptr::from_ref(&set_token).cast(),
                GeneratedLayout::INDEX_OFFSET,
            ),
            LayoutToken::new(
                TOKEN_PUSH_CONSTANT,
                std::ptr::from_ref(&push_token).cast(),
                GeneratedLayout::PUSH_OFFSET,
            ),
            LayoutToken::new(TOKEN_DISPATCH, std::ptr::null(), layout.dispatch_offset()),
        ];
        let info = LayoutCreateInfo::new(layout.stride, generated.layout.pipeline_layout, &tokens);
        // SAFETY: `info`, its tokens, and their data are live for the call.
        generated.commands_layout = unsafe { commands.create_commands_layout(&info) }
            .map_err(|error| fail(format!("create the indirect commands layout: {error}")))?;
        Ok(generated)
    }

    /// Reserve each command's preprocess memory: `vkGetGeneratedCommandsMemoryRequirementsEXT`
    /// for the most sequences its execution runs ([`GeneratedLayout::max_sequences`], asked once
    /// per distinct count), each at its alignment in one buffer of preprocess usage. No two
    /// executions share any of it.
    pub(super) fn reserve_preprocess(
        &mut self,
        context: &VulkanContext,
        layout: &GeneratedLayout,
    ) -> Result<(), ExecutorError> {
        let mut asked: BTreeMap<u32, vk::MemoryRequirements> = BTreeMap::new();
        let mut placed = Vec::with_capacity(layout.slots as usize);
        let mut total = 0u64;
        let mut alignment = 1u64;
        let mut type_bits = u32::MAX;
        for slot in 0..layout.slots {
            let sequences = layout.max_sequences(slot);
            let requirement = match asked.entry(sequences) {
                Entry::Occupied(known) => *known.get(),
                Entry::Vacant(new) => {
                    let info = MemoryRequirementsInfo::new(
                        self.execution_set,
                        self.commands_layout,
                        sequences,
                    );
                    // SAFETY: the execution set and commands layout are live.
                    *new.insert(unsafe { self.commands.memory_requirements(&info) })
                }
            };
            let align = requirement.alignment.max(1);
            let at = total.next_multiple_of(align);
            placed.push((at, requirement.size));
            total = at + requirement.size;
            alignment = alignment.max(align);
            if requirement.size != 0 {
                type_bits &= requirement.memory_type_bits;
            }
        }
        if total == 0 {
            self.reserved = vec![(0, 0); placed.len()];
            return Ok(());
        }
        let mut usage = vk::BufferUsageFlags2CreateInfoKHR::default()
            .usage(PREPROCESS_BUFFER | vk::BufferUsageFlags2KHR::SHADER_DEVICE_ADDRESS);
        let info = vk::BufferCreateInfo::default()
            .size(total)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut usage);
        let device = self.device;
        // SAFETY: `info` and its chain are live for the call.
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(|error| fail(format!("create the preprocess buffer: {error}")))?;
        self.preprocess = Some((buffer, vk::DeviceMemory::null()));
        // SAFETY: the buffer is live.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = context
            .memory_type(
                requirements.memory_type_bits & type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
            .map_err(fail)?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut flags);
        // SAFETY: `allocation` and its chain are live for the call.
        let memory = unsafe { device.allocate_memory(&allocation, None) }
            .map_err(|error| fail(format!("allocate the preprocess memory: {error}")))?;
        self.preprocess = Some((buffer, memory));
        // SAFETY: the buffer and memory are live, and the memory is of a type the buffer allows.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|error| fail(format!("bind the preprocess memory: {error}")))?;
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: the buffer was created with device-address usage and its memory is bound.
        let address = unsafe { device.get_buffer_device_address(&info) };
        if address % alignment != 0 {
            return Err(fail(format!(
                "the preprocess buffer's address {address:#x} is not {alignment}-aligned"
            )));
        }
        self.reserved = placed
            .into_iter()
            .map(|(at, size)| {
                if size == 0 {
                    (0, 0)
                } else {
                    (address + at, size)
                }
            })
            .collect();
        Ok(())
    }
}

/// `pipeline`'s compute pipeline for `region` in `module`, created with the shared `layout` and
/// `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`, and how long its creation took.
fn bindable_pipeline(
    context: &VulkanContext,
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    pipeline: &ComputePipeline,
    region: &DispatchRegion,
) -> Result<(vk::Pipeline, Duration), ExecutorError> {
    let entry = CString::new(pipeline.stage.entry_point.as_str())
        .map_err(|_| "entry point name contains NUL")?;
    let map: Vec<vk::SpecializationMapEntry> = region
        .specialization
        .entries
        .iter()
        .map(|&entry| entry.into())
        .collect();
    let spec = vk::SpecializationInfo::default()
        .map_entries(&map)
        .data(&region.specialization.data);
    let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
        .required_subgroup_size(region.required_subgroup_size.unwrap_or(0));
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry)
        .flags(region.stage_flags.into());
    if !map.is_empty() {
        stage = stage.specialization_info(&spec);
    }
    if region.required_subgroup_size.is_some() {
        stage = stage.push_next(&mut subgroup);
    }
    let mut flags = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(INDIRECT_BINDABLE);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout)
        .push_next(&mut flags);
    let clock = crate::budgets::CreationClock::start();
    // SAFETY: every pointer in `info` outlives the call; the caller destroys the result.
    let created = unsafe {
        context
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, error)| {
        fail(format!(
            "the driver refused an indirect-bindable compute pipeline: {error}"
        ))
    })?;
    let pipeline = created
        .first()
        .copied()
        .ok_or_else(|| fail("the driver created no pipeline"))?;
    Ok((pipeline, clock.read()))
}

/// Record, after the preprocess and its barrier, the descriptor sets once and per command of
/// `layout` in range order: `between` (as K14c's between commands, from the second on), the
/// initial pipeline, zeroed push constants (the tokens write every byte of them), and the
/// command's `vkCmdExecuteGeneratedCommandsEXT` over its sequences in the scratch at
/// `scratch_address`, with the count the preprocess wrote and its own preprocess memory. A
/// command that starts no segment has the count 0: its execution runs nothing. The layout is made
/// for explicit preprocessing: the initial pipeline and zeroed push constants are bound first,
/// every command's execution is preprocessed
/// (`vkCmdPreprocessGeneratedCommandsEXT`) with that state, a barrier orders the preprocessing's
/// writes before the executions' preprocess and indirect reads, and each execution runs
/// preprocessed. A range whose preprocess memory was not reserved for every command records
/// nothing and fails.
///
/// # Safety
/// `command` is recording on `device`, every object of `generated` is live until the submission
/// completes, and the scratch at `scratch_address` holds `layout`'s sections.
pub(super) unsafe fn record_executions(
    device: &ash::Device,
    command: vk::CommandBuffer,
    generated: &Generated<'_>,
    layout: &GeneratedLayout,
    scratch_address: vk::DeviceAddress,
    between: &vk::MemoryBarrier<'_>,
) -> Result<(), ExecutorError> {
    if generated.reserved.len() != layout.slots as usize {
        return Err(fail(format!(
            "preprocess memory for {} of {} commands",
            generated.reserved.len(),
            layout.slots
        )));
    }
    let pipeline_layout = generated.layout.pipeline_layout;
    let zeros = vec![0u8; layout.push_bytes as usize];
    // The state every preprocessing and execution takes: the execution set's initial pipeline and
    // zeroed push constants (the tokens write every byte). An execution leaves the bound pipeline
    // and push constants undefined.
    let bind = || {
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                generated.pipelines[0],
            );
            device.cmd_push_constants(
                command,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &zeros,
            );
        }
    };
    // Each command's execution: its sequences, the count the preprocess wrote, and its own
    // preprocess memory.
    let execution = |slot: u32| {
        let sequences = layout.max_sequences(slot);
        Execution {
            indirect: (
                scratch_address + layout.sequence_offset(slot, 0),
                u64::from(layout.stride) * u64::from(sequences),
            ),
            preprocess: generated.reserved[slot as usize],
            max_sequences: sequences,
            count_address: scratch_address + layout.count_offset(slot),
        }
    };
    // SAFETY: the caller's contract; each command's preprocess memory is its own.
    unsafe {
        if !generated.sets.is_empty() {
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &generated.sets,
                &[],
            );
        }
        bind();
        for slot in 0..layout.slots {
            let execution = execution(slot);
            let info = GeneratedCommandsInfo::new(
                generated.execution_set,
                generated.commands_layout,
                &execution,
            );
            generated.commands.preprocess(command, &info);
        }
        let preprocessed = vk::MemoryBarrier::default()
            .src_access_mask(PREPROCESS_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ | PREPROCESS_READ);
        device.cmd_pipeline_barrier(
            command,
            PREPROCESS_STAGE,
            vk::PipelineStageFlags::DRAW_INDIRECT | PREPROCESS_STAGE,
            vk::DependencyFlags::empty(),
            &[preprocessed],
            &[],
            &[],
        );
        for slot in 0..layout.slots {
            if slot > 0 {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(between),
                    &[],
                    &[],
                );
            }
            bind();
            let execution = execution(slot);
            let info = GeneratedCommandsInfo::new(
                generated.execution_set,
                generated.commands_layout,
                &execution,
            );
            generated.commands.execute_preprocessed(command, &info);
        }
    }
    Ok(())
}
