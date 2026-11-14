//! The runtime's side of the parity harness: [`Api`] on `metal2vulkan_runtime`, on the device
//! the lane's cfg selects, under the validation layer (`runtime_lane::validated_device_with`), at
//! its queue plan (`runtime_lane::at_plan`): a fixture's command queues are its own, as many as
//! it asks for, and the lane's queue limits would change the second one's placement.

use super::api::{Answer, Api, Completion, Fixture, Layout, Refused, Region, Times};
use super::record::Refusal;
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::runtime_lane::{at_plan, validated_device_with, Messages, ScratchCache};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputeCommandEncoder, ComputePipelineState, Device, Error, EventRef, Fence, PipelineOptions,
    ResourceOptions, SharedEvent, Size, Texture, TextureDescriptor,
};
use std::ops::Range;
use std::time::Duration;

/// The longest a command buffer of a scenario may take.
const WAIT: Duration = Duration::from_secs(60);

/// A runtime device, its validation messages and its shader cache, for one case. The device is
/// dropped before the cache directory is removed.
pub(crate) struct RuntimeApi {
    device: Device,
    pub(crate) messages: Messages,
    _cache: ScratchCache,
}

impl RuntimeApi {
    pub(crate) fn open() -> Result<RuntimeApi, String> {
        let cache = ScratchCache::new().map_err(|error| error.to_string())?;
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan)?;
        Ok(RuntimeApi {
            device,
            messages,
            _cache: cache,
        })
    }
}

/// The record of a runtime refusal: what Metal's API validation stops is `Invalid`; what the
/// runtime or the device lacks is a gap; anything else failed.
pub(super) fn refused(error: Error) -> Refused {
    let refusal = match error {
        Error::Invalid(_) => Refusal::Stopped,
        Error::Unsupported(_) | Error::NotImplemented(_) | Error::AppleOnly(_) => Refusal::Gap,
        _ => Refusal::Failed,
    };
    Refused {
        refusal,
        detail: error.to_string(),
    }
}

impl Api for RuntimeApi {
    type Buffer = Buffer;
    type Texture = Texture;
    type Pipeline = ComputePipelineState;
    type Queue = CommandQueue;
    type CommandBuffer = CommandBuffer;
    type Event = SharedEvent;
    type Fence = Fence;
    type Blit = BlitCommandEncoder;
    type Compute = ComputeCommandEncoder;

    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Buffer> {
        self.device.new_buffer(length, options).map_err(refused)
    }

    fn write(&self, buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("a Shared buffer");
        assert!(bytes.len() as u64 <= buffer.length());
        // SAFETY: the mapping covers the buffer's length, which holds `bytes`; no command
        // buffer uses the buffer while the scenario writes it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    }

    fn read(&self, buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        let length = usize::try_from(buffer.length()).expect("a buffer the host maps");
        // SAFETY: the mapping covers the buffer's length; the command buffers that wrote it
        // completed.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
    }

    fn texture(&self, descriptor: &TextureDescriptor) -> Answer<Texture> {
        self.device.new_texture(descriptor).map_err(refused)
    }

    fn pipeline(&self, fixture: &Fixture) -> Answer<ComputePipelineState> {
        let library = self
            .device
            .new_library_with_air(&fixture.bitcode)
            .map_err(refused)?;
        let function = library
            .new_function_with_name(&fixture.row.entry)
            .map_err(refused)?;
        self.device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .map_err(refused)
    }

    fn queue(&self) -> Answer<CommandQueue> {
        self.device.new_command_queue().map_err(refused)
    }

    fn command_buffer(&self, queue: &CommandQueue) -> Answer<CommandBuffer> {
        queue.command_buffer().map_err(refused)
    }

    fn event(&self) -> Answer<SharedEvent> {
        self.device.new_shared_event().map_err(refused)
    }

    fn fence(&self) -> Answer<Fence> {
        Ok(self.device.new_fence())
    }

    fn signal(&self, cb: &CommandBuffer, event: &SharedEvent, value: u64) -> Answer<()> {
        cb.encode_signal_event(EventRef::from(event), value)
            .map_err(refused)
    }

    fn wait(&self, cb: &CommandBuffer, event: &SharedEvent, value: u64) -> Answer<()> {
        cb.encode_wait_for_event(EventRef::from(event), value)
            .map_err(refused)
    }

    fn blit(&self, cb: &CommandBuffer) -> Answer<BlitCommandEncoder> {
        cb.blit_command_encoder().map_err(refused)
    }

    fn copy_buffer(
        &self,
        blit: &mut BlitCommandEncoder,
        (from, from_offset): (&Buffer, u64),
        (to, to_offset): (&Buffer, u64),
        size: u64,
    ) -> Answer<()> {
        blit.copy_from_buffer_to_buffer(from, from_offset, to, to_offset, size)
            .map_err(refused)
    }

    fn fill(
        &self,
        blit: &mut BlitCommandEncoder,
        buffer: &Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Answer<()> {
        blit.fill_buffer(buffer, range, value).map_err(refused)
    }

    fn copy_to_texture(
        &self,
        blit: &mut BlitCommandEncoder,
        (from, layout): (&Buffer, Layout),
        (to, region): (&Texture, Region),
        options: BlitOption,
    ) -> Answer<()> {
        blit.copy_from_buffer_to_texture(
            from,
            layout.offset,
            layout.bytes_per_row,
            layout.bytes_per_image,
            region.size,
            to,
            region.slice,
            region.level,
            region.origin,
            options,
        )
        .map_err(refused)
    }

    fn copy_from_texture(
        &self,
        blit: &mut BlitCommandEncoder,
        (from, region): (&Texture, Region),
        (to, layout): (&Buffer, Layout),
        options: BlitOption,
    ) -> Answer<()> {
        blit.copy_from_texture_to_buffer(
            from,
            region.slice,
            region.level,
            region.origin,
            region.size,
            to,
            layout.offset,
            layout.bytes_per_row,
            layout.bytes_per_image,
            options,
        )
        .map_err(refused)
    }

    fn blit_update_fence(&self, blit: &mut BlitCommandEncoder, fence: &Fence) -> Answer<()> {
        blit.update_fence(fence).map_err(refused)
    }

    fn blit_wait_fence(&self, blit: &mut BlitCommandEncoder, fence: &Fence) -> Answer<()> {
        blit.wait_for_fence(fence).map_err(refused)
    }

    fn end_blit(&self, blit: BlitCommandEncoder) -> Answer<()> {
        blit.end_encoding().map_err(refused)
    }

    fn compute(&self, cb: &CommandBuffer) -> Answer<ComputeCommandEncoder> {
        cb.compute_command_encoder().map_err(refused)
    }

    fn dispatch(
        &self,
        compute: &mut ComputeCommandEncoder,
        pipeline: &ComputePipelineState,
        buffers: &[(&Buffer, u64)],
        threads: Size,
    ) -> Answer<()> {
        compute
            .set_compute_pipeline_state(pipeline)
            .map_err(refused)?;
        for &(buffer, index) in buffers {
            let index = u32::try_from(index).expect("a buffer index");
            compute
                .set_buffer(Some(buffer), 0, index)
                .map_err(refused)?;
        }
        compute
            .dispatch_threads(threads, Size::new(1, 1, 1))
            .map_err(refused)
    }

    fn compute_update_fence(
        &self,
        compute: &mut ComputeCommandEncoder,
        fence: &Fence,
    ) -> Answer<()> {
        compute.update_fence(fence).map_err(refused)
    }

    fn compute_wait_fence(&self, compute: &mut ComputeCommandEncoder, fence: &Fence) -> Answer<()> {
        compute.wait_for_fence(fence).map_err(refused)
    }

    fn end_compute(&self, compute: ComputeCommandEncoder) -> Answer<()> {
        compute.end_encoding().map_err(refused)
    }

    fn commit(&self, cb: &CommandBuffer) -> Answer<()> {
        cb.commit().map_err(refused)
    }

    fn completion(&self, cb: CommandBuffer) -> Result<Completion, String> {
        cb.wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        let status = match cb.status() {
            CommandBufferStatus::Completed => "completed",
            CommandBufferStatus::Error => "error",
            other => return Err(format!("a completed command buffer is {other:?}")),
        };
        Ok(Completion {
            status: status.to_owned(),
            times: self.times(&cb).ok(),
        })
    }

    fn times(&self, cb: &CommandBuffer) -> Answer<Times> {
        Ok(Times {
            gpu_start: cb.gpu_start_time().map_err(refused)?,
            gpu_end: cb.gpu_end_time().map_err(refused)?,
            kernel_start: cb.kernel_start_time(),
            kernel_end: cb.kernel_end_time(),
        })
    }

    fn host_time(&self) -> f64 {
        metal2vulkan_runtime::host_time()
    }
}
