//! The rt side's compute pass: one encoder of a case's commands ([`ComputeStep`]), committed and
//! waited for.
//!
//! A step the runtime has no counterpart of answers `Unavailable` with its Metal call, and the
//! pass goes on with the steps after it, as Metal's encoder keeps the commands around one it
//! drops. A range setter is its slots set one by one, which is what Metal's range call is. Both
//! barrier forms are the runtime's one barrier (`memory_barrier`), and the encoder's
//! `dispatchType` is the consumer's own: the runtime takes it as the encoder's argument and keeps
//! no getter.

use super::{waited_for, RtSide, WAIT};
use crate::case::{Answer, Failure, Handle};
use crate::compute_pass::{
    ComputePass, ComputePassOutcome, ComputeStep, Dispatching, PassResource,
};
use crate::transcript::{Negative, Value};
use metal2vulkan_rt as rt;

/// A typed error of the runtime as the failure it records.
fn of_rt(error: rt::Error) -> Failure {
    Failure::of_rt(&error)
}

/// The runtime's usage for the raw bits a case states: `MTLResourceUsageRead`, `Write` and
/// `Sample`, a refusal of any other bit, as the old runtime's side refuses it, and `Unavailable` for
/// no bit, a usage the runtime's type does not state.
fn usage(bits: u64) -> Answer<rt::ResourceUsage> {
    if bits & !7 != 0 {
        return Err(Failure::named("NoSuchResourceUsage", format!("{bits:#x}")));
    }
    [
        (1, rt::ResourceUsage::READ),
        (2, rt::ResourceUsage::WRITE),
        (4, rt::ResourceUsage::SAMPLE),
    ]
    .into_iter()
    .filter(|(bit, _)| bits & bit != 0)
    .map(|(_, usage)| usage)
    .reduce(|held, usage| held | usage)
    .ok_or_else(|| Failure::unavailable("useResource with no usage"))
}

/// A case's `MTLBarrierScope` bits, refused past `Buffers`, `Textures` and `RenderTargets`.
fn scope(bits: u64) -> Answer<()> {
    match bits & !7 {
        0 => Ok(()),
        _ => Err(Failure::named("NoSuchBarrierScope", format!("{bits:#x}"))),
    }
}

impl RtSide<'_> {
    /// The resource a step names.
    fn resource(&self, named: &PassResource) -> Answer<rt::Resource<'_>> {
        Ok(match named {
            PassResource::Buffer(handle) => rt::Resource::Buffer(self.held(*handle)?),
            PassResource::Texture(handle) => rt::Resource::Texture(self.held_texture(*handle)?),
        })
    }

    fn held_sampler(&self, handle: Handle) -> Answer<&rt::SamplerState> {
        (self.samplers.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    fn held_pipeline(&self, handle: Handle) -> Answer<&rt::ComputePipelineState> {
        (self.pipelines.get(handle.0 as usize))
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("compute pipeline {}", handle.0)))
    }

    /// One step encoded into `encoder`.
    fn step(&self, encoder: &mut rt::ComputeCommandEncoder, step: &ComputeStep) -> Answer<()> {
        match step {
            ComputeStep::Pipeline(state) => {
                (encoder.set_compute_pipeline_state(self.held_pipeline(*state)?)).map_err(of_rt)
            }
            ComputeStep::Buffer {
                buffer,
                offset,
                index,
            } => (encoder.set_buffer(Some(self.held(*buffer)?), *offset, *index)).map_err(of_rt),
            ComputeStep::BufferOffset { offset, index } => {
                encoder.set_buffer_offset(*offset, *index).map_err(of_rt)
            }
            ComputeStep::Buffers { buffers, first } => {
                for (at, (buffer, offset)) in (*first..).zip(buffers) {
                    (encoder.set_buffer(Some(self.held(*buffer)?), *offset, at)).map_err(of_rt)?;
                }
                Ok(())
            }
            ComputeStep::Bytes { bytes, index } => encoder.set_bytes(bytes, *index).map_err(of_rt),
            ComputeStep::Texture { texture, index } => {
                (encoder.set_texture(Some(self.held_texture(*texture)?), *index)).map_err(of_rt)
            }
            ComputeStep::Textures { textures, first } => {
                for (at, texture) in (*first..).zip(textures) {
                    (encoder.set_texture(Some(self.held_texture(*texture)?), at)).map_err(of_rt)?;
                }
                Ok(())
            }
            ComputeStep::Sampler { sampler, index } => (encoder
                .set_sampler_state(Some(self.held_sampler(*sampler)?), *index))
            .map_err(of_rt),
            ComputeStep::Samplers { samplers, first } => {
                for (at, sampler) in (*first..).zip(samplers) {
                    (encoder.set_sampler_state(Some(self.held_sampler(*sampler)?), at))
                        .map_err(of_rt)?;
                }
                Ok(())
            }
            ComputeStep::ThreadgroupMemory { length, index } => {
                (encoder.set_threadgroup_memory_length(*length, *index)).map_err(of_rt)
            }
            ComputeStep::UseResource {
                resource,
                usage: bits,
            } => (encoder.use_resource(self.resource(resource)?, usage(*bits)?)).map_err(of_rt),
            ComputeStep::UseResources {
                resources,
                usage: bits,
            } => {
                let usage = usage(*bits)?;
                for resource in resources {
                    (encoder.use_resource(self.resource(resource)?, usage)).map_err(of_rt)?;
                }
                Ok(())
            }
            ComputeStep::BarrierScope { scope: bits } => scope(*bits).map(|()| {
                encoder.memory_barrier();
            }),
            ComputeStep::BarrierResources { resources } => {
                for resource in resources {
                    self.resource(resource)?;
                }
                encoder.memory_barrier();
                Ok(())
            }
            ComputeStep::DispatchThreadgroups { groups, threads } => encoder
                .dispatch_threadgroups(*groups, *threads)
                .map_err(of_rt),
            ComputeStep::DispatchThreadgroupsIndirect {
                buffer,
                offset,
                threads,
            } => (encoder.dispatch_threadgroups_with_indirect_buffer(
                self.held(*buffer)?,
                *offset,
                *threads,
            ))
            .map_err(of_rt),
            // The attribute-stride setters, the stage-in region, the LOD-clamped samplers and
            // heaps: the runtime has no counterpart yet.
            ComputeStep::BufferWithStride { .. }
            | ComputeStep::BuffersWithStrides { .. }
            | ComputeStep::BufferOffsetWithStride { .. }
            | ComputeStep::BytesWithStride { .. }
            | ComputeStep::StageInRegion { .. }
            | ComputeStep::ClampedSampler { .. }
            | ComputeStep::ClampedSamplers { .. }
            | ComputeStep::UseHeap { .. }
            | ComputeStep::UseHeaps { .. } => Err(Failure::unavailable(step.call())),
        }
    }

    /// One compute pass of `pass`'s steps in one command buffer, committed and waited for within
    /// [`WAIT`].
    pub(super) fn run_compute_pass(
        &mut self,
        pass: &ComputePass<'_>,
    ) -> Answer<ComputePassOutcome> {
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut encoder = (command_buffer.compute_command_encoder(match pass.dispatching {
            Dispatching::Serial => rt::DispatchType::Serial,
            Dispatching::Concurrent => rt::DispatchType::Concurrent,
        }))
        .map_err(of_rt)?;
        let mut refused = Vec::with_capacity(pass.steps.len());
        let mut set = Vec::new();
        for step in pass.steps {
            let answered = self.step(&mut encoder, step);
            if let (Ok(()), ComputeStep::Pipeline(state)) = (&answered, step) {
                set.push(state.0 as usize);
            }
            refused.push(answered.err());
        }
        encoder.end_encoding();
        command_buffer.commit().map_err(of_rt)?;
        self.dispatched.extend(set);
        let dispatch_type = Failure::named(
            "TheConsumers",
            "an encoder's dispatchType is the argument it was made with: metal2vulkan-rt keeps no \
             getter and leaves it to its consumer",
        )
        .recorded();
        let Some(word) = waited_for(&command_buffer, WAIT) else {
            return Ok(ComputePassOutcome {
                refused,
                dispatch_type,
                status: Value::Nothing(Negative::NeverReturned),
                error: None,
            });
        };
        Ok(ComputePassOutcome {
            refused,
            dispatch_type,
            status: Value::unit_variant(word),
            error: (command_buffer.error()).map(|error| Value::unit_variant(format!("{error:?}"))),
        })
    }
}
