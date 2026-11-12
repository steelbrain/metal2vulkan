//! The generated-commands recording of an execution on the GPU alone (`gpu_execute.rs`), where
//! the device enabled `VK_EXT_device_generated_commands` and the product sets its path up for the
//! variants (`bind::replay_preprocess_generated`).
//!
//! - **What runs.** After the replay's preprocess and its barrier, per command of the range one
//!   `vkCmdExecuteGeneratedCommandsEXT` of the sequences the preprocess counted for it, a compute
//!   barrier before each but the first. Where the driver's own preprocessing of an execution after
//!   another goes wrong (`generated_commands::preprocess_apart`), the commands layout is made for
//!   explicit preprocessing: every execution of the range is preprocessed
//!   (`vkCmdPreprocessGeneratedCommandsEXT`) after the barrier, then a barrier, then the executions
//!   run preprocessed, so no preprocessing follows an execution.
//! - **Objects.** Per execution: each variant's indirect-bindable pipeline under the pipeline
//!   layout the variants share, the indirect execution set and commands layout, and the preprocess
//!   memory; the command buffer holds them until it completes.

use super::buffer::Buffer;
use super::degradation::CompiledPipeline;
use super::encoder::EncoderCore;
use super::error::{internal, vulkan, Error};
use super::generated_commands::{
    preprocess_apart, Commands, Execution as GeneratedExecution, ExecutionSetCreateInfo,
    ExecutionSetPipelineInfo, ExecutionSetToken, GeneratedCommandsInfo, Handle, LayoutCreateInfo,
    LayoutToken, MemoryRequirementsInfo, PushConstantToken, WriteExecutionSetPipeline,
    EXECUTION_SET_PIPELINES, TOKEN_DISPATCH, TOKEN_EXECUTION_SET, TOKEN_PUSH_CONSTANT,
};
use super::gpu_execute::{Prepared, Scratch};
use super::pipeline::{RawPipeline, VariantObjects};
use super::release::{self, Generated};
use super::resource::ResourceKey;
use super::tracker::{Barrier, Use};
use super::Device;
use ash::vk;
use metal2vulkan::bind::{GeneratedLayout, ReplayPreprocess};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// Record into `core`, after the replay's preprocess `pre` and its barrier, the generated-commands
/// executions of the commands of `prepared`'s variants through `commands` (the module docs),
/// reading `scratch`, with `uses`, what the variants' plans reach. Each variant's objects and its
/// pipeline's creation time, in the variants' order. Refused as the device refuses an object it
/// makes.
pub(super) fn record(
    core: &mut EncoderCore,
    (commands, pre, prepared): (&Commands, &ReplayPreprocess, &[Prepared]),
    scratch: &Scratch,
    uses: &[Use<ResourceKey>],
) -> Result<Vec<(Arc<VariantObjects>, Duration)>, Error> {
    let device = core.device().clone();
    let (Some(replay), Some(layout)) = (&pre.generated, pre.layout.generated) else {
        return Err(internal("a replay without the generated-commands path"));
    };
    // Each variant's indirect-bindable pipeline, under a layout defined as the shared one.
    let mut objects = Vec::with_capacity(prepared.len());
    let mut pipelines = Vec::with_capacity(prepared.len());
    for p in prepared {
        let [region] = p.commands.regions.as_slice() else {
            return Err(internal("a replay variant of other than one region"));
        };
        let mut made = VariantObjects::create(
            &device,
            &replay.pipeline_layout,
            std::slice::from_ref(&p.variant.pipeline().stage),
            CompiledPipeline::Kernel(p.variant.pipeline()),
            None,
        )?;
        made.name_pipelines(|| p.variant.objects().pipeline_name(None));
        let made = Arc::new(made);
        pipelines.push(made.indirect_bindable_pipeline(region)?);
        objects.push(made);
    }
    let pipeline_layout = objects
        .first()
        .ok_or_else(|| internal("a replay of no variant"))?
        .layout();
    // Where the driver's own preprocessing of an execution after another runs the wrong shader,
    // every execution is preprocessed apart before the first runs.
    let explicit = preprocess_apart(&device.properties().description);
    let generated =
        GeneratedObjects::create(&device, &pipelines, pipeline_layout, &layout, explicit)?;
    let (memory, reserved) = generated.preprocess_memory(&device, &layout)?;
    let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let zeros = vec![0u8; layout.push_bytes as usize];
    let initial = pipelines[0].raw;
    let (execution_set, commands_layout) = (generated.execution_set, generated.commands_layout);
    let scratch = scratch.address;
    // Each command's execution: its sequences, its count and its own preprocess memory.
    let execution = |slot: u32| {
        let sequences = layout.max_sequences(slot);
        GeneratedExecution {
            indirect: (
                scratch + layout.sequence_offset(slot, 0),
                u64::from(layout.stride) * u64::from(sequences),
            ),
            preprocess: reserved[slot as usize],
            max_sequences: sequences,
            count_address: scratch + layout.count_offset(slot),
        }
    };
    core.command(compute, uses, |raw, command_buffer| {
        // The state every preprocessing and execution takes: the execution set's initial
        // pipeline and zeroed push constants (the tokens write every byte). An execution leaves
        // the bound pipeline and push constants undefined.
        let bind = || {
            // SAFETY: the segment's command buffer is recording; the pipeline and its layout are
            // held until the command buffer completes.
            unsafe {
                raw.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, initial);
                raw.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &zeros,
                );
            }
        };
        // SAFETY: as above; every object the executions name is held until the command buffer
        // completes, and each command's preprocess memory is its own. Where `explicit`, the
        // layout was made for explicit preprocessing, every execution is preprocessed with the
        // state it then runs with, and a barrier orders those writes before every execution.
        unsafe {
            if explicit {
                bind();
                for slot in 0..layout.slots {
                    let info = GeneratedCommandsInfo::new(
                        execution_set,
                        commands_layout,
                        &execution(slot),
                    );
                    commands.preprocess(command_buffer, &info);
                }
                Barrier::everything().record(raw, command_buffer);
            }
            for slot in 0..layout.slots {
                if slot > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                bind();
                let info =
                    GeneratedCommandsInfo::new(execution_set, commands_layout, &execution(slot));
                match explicit {
                    true => commands.execute_preprocessed(command_buffer, &info),
                    false => commands.execute(command_buffer, &info),
                }
            }
        }
    });
    let made = objects
        .iter()
        .zip(&pipelines)
        .map(|(objects, pipeline)| (Arc::clone(objects), pipeline.created_in))
        .collect();
    core.retain(generated);
    core.retain(memory);
    core.retain(pipelines);
    core.retain(objects);
    Ok(made)
}

/// An execution's indirect execution set and indirect commands layout, released through the
/// release door (`release.rs`) when the command buffer that holds them completes.
struct GeneratedObjects {
    /// The device whose extension commands made them: held, so they are released while it lives
    /// whichever of their holders goes last.
    device: Device,
    execution_set: Handle,
    commands_layout: Handle,
}

impl Drop for GeneratedObjects {
    /// Release both through the release door (`release.rs`): once the process is exiting, they
    /// are left to its end.
    fn drop(&mut self) {
        // Made only by a device that has the commands.
        let Some(commands) = self.device.shared().generated_commands else {
            return;
        };
        // SAFETY: no pending command uses the objects (a command buffer holds them until it
        // completes); each is this execution's, released once, by its device's commands.
        unsafe {
            if self.execution_set != 0 {
                release::generated(&commands, Generated::ExecutionSet(self.execution_set));
            }
            if self.commands_layout != 0 {
                release::generated(&commands, Generated::CommandsLayout(self.commands_layout));
            }
        }
    }
}

impl GeneratedObjects {
    /// The execution set of `pipelines` (the first its initial pipeline, each at its variant's
    /// index) and the commands layout of `layout`'s three tokens under `pipeline_layout`, which
    /// every pipeline's layout is defined as, made for explicit preprocessing where `explicit`.
    fn create(
        device: &Device,
        pipelines: &[RawPipeline],
        pipeline_layout: vk::PipelineLayout,
        layout: &GeneratedLayout,
        explicit: bool,
    ) -> Result<GeneratedObjects, Error> {
        let commands = device
            .shared()
            .generated_commands
            .ok_or_else(|| internal("generated objects on a device without the commands"))?;
        let mut objects = GeneratedObjects {
            device: device.clone(),
            execution_set: 0,
            commands_layout: 0,
        };
        let (first, rest) = pipelines
            .split_first()
            .ok_or_else(|| internal("an execution set of no pipeline"))?;
        let count = u32::try_from(pipelines.len()).map_err(|_| internal("too many variants"))?;
        let info = ExecutionSetPipelineInfo::new(first.raw, count);
        // SAFETY: `info` and the initial pipeline are live for the call; dropping `objects`
        // releases what was made on a later failure.
        objects.execution_set =
            unsafe { commands.create_execution_set(&ExecutionSetCreateInfo::new(&info)) }
                .map_err(vulkan)?;
        let writes: Vec<WriteExecutionSetPipeline> = (1..)
            .zip(rest)
            .map(|(index, pipeline)| WriteExecutionSetPipeline::new(index, pipeline.raw))
            .collect();
        if !writes.is_empty() {
            // SAFETY: nothing has used the set yet; every written pipeline is live.
            unsafe { commands.update_execution_set(objects.execution_set, &writes) };
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
        let info = LayoutCreateInfo::new(layout.stride, pipeline_layout, &tokens);
        let info = match explicit {
            true => info.explicitly_preprocessed(),
            false => info,
        };
        // SAFETY: `info`, its tokens, and their data are live for the call.
        objects.commands_layout =
            unsafe { commands.create_commands_layout(&info) }.map_err(vulkan)?;
        Ok(objects)
    }

    /// Each command's preprocess memory, by slot (its address and size; `(0, 0)` where the
    /// device needs none), in one buffer: `vkGetGeneratedCommandsMemoryRequirementsEXT` for the
    /// most sequences its execution runs (asked once per distinct count), each at its alignment.
    /// No two executions share any of it.
    fn preprocess_memory(
        &self,
        device: &Device,
        layout: &GeneratedLayout,
    ) -> Result<(Option<Buffer>, Vec<(u64, u64)>), Error> {
        let commands = self
            .device
            .shared()
            .generated_commands
            .ok_or_else(|| internal("generated objects on a device without the commands"))?;
        let mut asked: BTreeMap<u32, vk::MemoryRequirements> = BTreeMap::new();
        let mut placed = Vec::with_capacity(layout.slots as usize);
        let mut total = 0u64;
        let mut alignment = 1u64;
        let mut memory_type_bits = u32::MAX;
        for slot in 0..layout.slots {
            let sequences = layout.max_sequences(slot);
            let requirement = *asked.entry(sequences).or_insert_with(|| {
                let info = MemoryRequirementsInfo::new(
                    self.execution_set,
                    self.commands_layout,
                    sequences,
                );
                // SAFETY: the execution set and commands layout are live, made by these commands.
                unsafe { commands.memory_requirements(&info) }
            });
            let align = requirement.alignment.max(1);
            let at = total.next_multiple_of(align);
            placed.push((at, requirement.size));
            total = at + requirement.size;
            alignment = alignment.max(align);
            if requirement.size != 0 {
                memory_type_bits &= requirement.memory_type_bits;
            }
        }
        if total == 0 {
            return Ok((None, vec![(0, 0); placed.len()]));
        }
        let buffer = device.preprocess_buffer(vk::MemoryRequirements {
            size: total,
            alignment,
            memory_type_bits,
        })?;
        let address = buffer
            .device_address()
            .ok_or_else(|| internal("preprocess memory without a device address"))?;
        if address % alignment != 0 {
            return Err(internal("preprocess memory below its alignment"));
        }
        let reserved = placed
            .into_iter()
            .map(|(at, size)| {
                if size == 0 {
                    (0, 0)
                } else {
                    (address + at, size)
                }
            })
            .collect();
        Ok((Some(buffer), reserved))
    }
}
