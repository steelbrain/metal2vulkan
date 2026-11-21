//! The rt side's state and target calls that the runtime has the counterpart of: a compute state's
//! limit under its descriptor, a compute state for indirect commands (the runtime's typed refusal),
//! a compute state of a whole descriptor, a command queue's slots, a fragment fill and a render
//! target of the mesh family's format.

use super::{keep, waited_for, RtSide, QUEUE_COMMAND_BUFFERS, WAIT};
use crate::case::{Answer, FragmentFill, FragmentFilled, Handle, Side, StageInput};
use crate::pipeline_facts::ComputeStateSpec;
use crate::pipeline_limits::{floored, ComputeLimitsSpec, ComputeStateLimits, Creation};
use crate::transcript::{Negative, Value};
use metal2vulkan_runtime as rt;

/// A typed error of the runtime as the failure it records.
fn of_rt(error: rt::Error) -> crate::case::Failure {
    crate::case::Failure::of_rt(&error)
}

/// Whether the runtime's error is Metal's own refusal of the descriptor: its validation's
/// (`Refused`) or the translator's refusal of a descriptor Metal refuses (`InvalidDescriptor`).
fn refuses_as_metal(error: &rt::Error) -> bool {
    matches!(
        error,
        rt::Error::Refused(_)
            | rt::Error::Translation {
                error: metal2vulkan::error::Error::InvalidDescriptor(_),
                ..
            }
    )
}

impl RtSide<'_> {
    /// The library's `entry` as a stage of a descriptor.
    pub(super) fn kernel(&self, library: Handle, entry: &str) -> Answer<rt::StageSource> {
        let function = (self.held_library(library)?.function(entry))
            .map_err(|error| self.on.translation_failure(&error))?;
        Ok(rt::StageSource::Function(rt::FunctionDescriptor::new(
            function,
        )))
    }

    /// [`crate::case::Side::compute_pipeline_limits`]: the runtime's state of the descriptor, and
    /// its `maxTotalThreadsPerThreadgroup`, or Metal's refusal where the runtime's error is one.
    pub(super) fn compute_limits(
        &mut self,
        library: Handle,
        spec: &ComputeLimitsSpec<'_>,
    ) -> Answer<Creation<ComputeStateLimits>> {
        let mut descriptor = rt::ComputePipelineDescriptor::new(self.kernel(library, spec.entry)?);
        descriptor.max_total_threads_per_threadgroup =
            (spec.max_total_threads != 0).then_some(spec.max_total_threads);
        let on = self.on;
        match (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
        {
            Ok(state) => Ok(Creation::made(ComputeStateLimits {
                max_total_threads_per_threadgroup: floored(u64::from(
                    state.max_total_threads_per_threadgroup(),
                )),
            })),
            Err(error) if refuses_as_metal(&error) => Ok(Creation::refused()),
            Err(error) => Err(on.failure(&error)),
        }
    }

    /// [`crate::case::Side::compute_pipeline_for_commands`]: the runtime's state of the kernel
    /// with `supportIndirectCommandBuffers` set, under the stage input where the case names one.
    pub(super) fn compute_state_for_commands(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: Option<&StageInput>,
    ) -> Answer<Handle> {
        let mut descriptor = rt::ComputePipelineDescriptor::new(self.kernel(library, entry)?);
        descriptor.support_indirect_command_buffers = true;
        descriptor.stage_input = stage_input.map(super::fields::stage_input_descriptor);
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    /// [`crate::case::Side::compute_pipeline_with_descriptor`]: the runtime's state of the
    /// descriptor's kernel, `supportIndirectCommandBuffers` and `requiredThreadsPerThreadgroup`.
    /// The runtime takes no label: a state's label is its consumer's.
    pub(super) fn described_compute(
        &mut self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let mut descriptor = rt::ComputePipelineDescriptor::new(self.kernel(library, spec.entry)?);
        descriptor.support_indirect_command_buffers = spec.support_indirect_command_buffers;
        descriptor.required_threads_per_threadgroup = spec.required_threads_per_threadgroup;
        let on = self.on;
        let state = (on.device)
            .new_compute_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        Ok(keep(&mut self.pipelines, state))
    }

    /// [`crate::case::Side::queue_slots`]: the probe both sides run, on a queue of the runtime's
    /// made with `max` (`newCommandQueue`'s count where none is named) and another queue.
    pub(super) fn slots(
        &mut self,
        max: Option<u32>,
        cap: u32,
    ) -> Answer<crate::queue_slots::QueueSlots> {
        let device = &self.on.device;
        let queue =
            (device.new_command_queue(max.unwrap_or(QUEUE_COMMAND_BUFFERS))).map_err(of_rt)?;
        let other = (device.new_command_queue(QUEUE_COMMAND_BUFFERS)).map_err(of_rt)?;
        crate::queue_slots::probe(
            cap,
            move || queue.command_buffer().map_err(of_rt),
            || other.command_buffer().is_ok(),
        )
    }

    /// [`crate::case::Side::fragment_fill`]: the runtime's state of the two functions over one
    /// colour attachment of the fill's format, a Private target, one pass that loads nothing and
    /// stores, the buffers at fragment indices 0 and up, one `drawPrimitives:` of triangles, and
    /// then a blit of level 0 into a Shared buffer, each command buffer waited for within
    /// [`WAIT`].
    pub(super) fn fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let function = |name: &str| -> Answer<rt::StageSource> {
            let function = (self.held_library(fill.library)?.function(name))
                .map_err(|error| self.on.translation_failure(&error))?;
            Ok(rt::StageSource::Function(rt::FunctionDescriptor::new(
                function,
            )))
        };
        let mut descriptor = rt::RenderPipelineDescriptor::new(function(fill.vertex)?);
        descriptor.fragment = Some(function(fill.fragment)?);
        descriptor.color_attachments[0] = Some(rt::ColorAttachmentDescriptor::new(fill.format));
        let on = self.on;
        let state = (on.device)
            .new_render_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
            .map_err(|error| on.failure(&error))?;
        let [width, height] = fill.size;
        let mut target =
            rt::TextureDescriptor::new(rt::TextureType::Type2D, fill.format, width, height);
        target.usage = rt::TextureUsage::RENDER_TARGET;
        target.storage_mode = rt::StorageMode::Private;
        let index = on.texture_index(target.pixel_format, target.usage);
        let target = on.device.new_texture(&target, index).map_err(of_rt)?;
        let bound = (fill.buffers.iter())
            .map(|&handle| self.held(handle).cloned())
            .collect::<Answer<Vec<rt::Buffer>>>()?;
        let mut pass = rt::RenderPassDescriptor::default();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = rt::LoadAction::DontCare;
        attachment.store_action = rt::StoreAction::Store;
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut encoder = (command_buffer.render_command_encoder(&pass)).map_err(of_rt)?;
        let encoded = (|| {
            encoder.set_render_pipeline_state(&state).map_err(of_rt)?;
            for (index, buffer) in bound.iter().enumerate() {
                (encoder.set_fragment_buffer(Some(buffer), 0, index as u32)).map_err(of_rt)?;
            }
            (encoder.draw_primitives(rt::PrimitiveType::Triangle, 0, fill.vertices, 1, 0))
                .map_err(of_rt)
        })();
        encoder.end_encoding();
        encoded?;
        command_buffer.commit().map_err(of_rt)?;
        let status = waited_for(&command_buffer, WAIT)
            .map_or(Value::Nothing(Negative::NeverReturned), Value::unit_variant);
        let at = keep(&mut self.render_pipelines, state);
        self.drawn.insert(at.0 as usize);
        let bytes = self.read_level(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    /// Level 0 of `texture`, `size` texels, blitted into a Shared buffer at `row_pitch` bytes a
    /// row and read once that command buffer completed within [`WAIT`].
    fn read_level(
        &mut self,
        texture: &rt::Texture,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let length = u64::from(row_pitch) * u64::from(size[1]);
        let back = self.zeroed_buffer(length as usize)?;
        let into = self.held(back)?.clone();
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut blit = command_buffer.blit_command_encoder().map_err(of_rt)?;
        let place = rt::TexturePlace {
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
        };
        let layout = rt::BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(row_pitch),
            bytes_per_image: length,
        };
        let copied =
            blit.copy_from_texture_to_buffer(texture, place, [size[0], size[1], 1], &into, layout);
        blit.end_encoding();
        copied.map_err(of_rt)?;
        command_buffer.commit().map_err(of_rt)?;
        if waited_for(&command_buffer, WAIT).is_none() {
            return Err(crate::case::Failure::named(
                "ReadBackNeverReturned",
                "the blit of the target into a buffer did not complete",
            ));
        }
        self.read_buffer(back, 0, length as usize)
    }

    /// [`crate::case::Side::render_target`]: a Shared 2D render target of the mesh family's
    /// format, at an index as the runtime takes one for a colour render target.
    pub(super) fn mesh_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        let format = rt::PixelFormat::RGBA32Uint;
        let mut descriptor =
            rt::TextureDescriptor::new(rt::TextureType::Type2D, format, width, height);
        descriptor.usage = rt::TextureUsage::RENDER_TARGET;
        descriptor.storage_mode = rt::StorageMode::Shared;
        let index = (self.on).texture_index(descriptor.pixel_format, descriptor.usage);
        let texture = self
            .on
            .device
            .new_texture(&descriptor, index)
            .map_err(of_rt)?;
        Ok(keep(&mut self.textures, texture))
    }
}
