//! The old runtime (`metal2vulkan-runtime`) behind the harness's traits, each method its call of
//! the same name. The cutover deletes this file, its `mod` line, the `--runtime old` arm of the
//! command and the crate's dependency on the old runtime.

use crate::{Code, Result, Size, Storage};
use metal2vulkan_runtime as runtime;
use runtime::{CommandBufferStatus, PipelineOptions, ResourceOptions};
use std::ptr::NonNull;

/// The old runtime's `MTLCreateSystemDefaultDevice`: the preferred Vulkan physical device
/// (discrete, then integrated, virtual, CPU) it can open, with its default options.
pub fn system_default() -> Result<runtime::Device> {
    Ok(runtime::Device::system_default()?)
}

fn size(size: Size) -> runtime::Size {
    runtime::Size::new(size[0], size[1], size[2])
}

impl crate::Device for runtime::Device {
    type Buffer = runtime::Buffer;
    type Library = runtime::Library;
    type Function = runtime::Function;
    type ComputePipelineState = runtime::ComputePipelineState;
    type CommandQueue = runtime::CommandQueue;
    type CommandBuffer = runtime::CommandBuffer;
    type ComputeCommandEncoder = runtime::ComputeCommandEncoder;

    fn name(&self) -> String {
        runtime::Device::name(self).to_owned()
    }

    fn new_buffer(&self, length: u64, storage: Storage) -> Result<runtime::Buffer> {
        let options = match storage {
            Storage::Shared => ResourceOptions::shared(),
            Storage::Private => ResourceOptions::private(),
        };
        Ok(runtime::Device::new_buffer(self, length, options)?)
    }

    fn new_library(&self, code: Code<'_>) -> Result<runtime::Library> {
        match code {
            Code::Metallib(bytes) => Ok(self.new_library_with_data(bytes)?),
            Code::Air(bytes) => {
                let library = metal2vulkan::library::Library::from_air(bytes)?;
                Ok(self.new_library_with_compile_library(library))
            }
        }
    }

    fn new_compute_pipeline_state(
        &self,
        function: &runtime::Function,
    ) -> Result<runtime::ComputePipelineState> {
        Ok(self.new_compute_pipeline_state_with_function(function, &PipelineOptions::default())?)
    }

    fn new_command_queue(&self) -> Result<runtime::CommandQueue> {
        Ok(runtime::Device::new_command_queue(self)?)
    }

    /// Its pooled and dedicated `VkDeviceMemory` blocks with a placement in them, and its empty
    /// kept ones (`WorkCounts`).
    fn gauges(&self) -> Vec<(&'static str, u64)> {
        let counts = self.work_counts();
        vec![
            ("memory_blocks", counts.memory_blocks_live),
            ("dedicated_blocks", counts.dedicated_memory_blocks_live),
            ("empty_blocks", counts.empty_memory_blocks_live),
        ]
    }
}

impl crate::Buffer for runtime::Buffer {
    fn contents(&self) -> Option<NonNull<u8>> {
        runtime::Buffer::contents(self)
    }
}

impl crate::Library<runtime::Device> for runtime::Library {
    fn new_function(&self, name: &str) -> Result<runtime::Function> {
        Ok(self.new_function_with_name(name)?)
    }
}

impl crate::CommandQueue<runtime::Device> for runtime::CommandQueue {
    fn command_buffer(&self) -> Result<runtime::CommandBuffer> {
        Ok(runtime::CommandQueue::command_buffer(self)?)
    }
}

impl crate::CommandBuffer<runtime::Device> for runtime::CommandBuffer {
    fn compute_command_encoder(&self) -> Result<runtime::ComputeCommandEncoder> {
        Ok(runtime::CommandBuffer::compute_command_encoder(self)?)
    }

    fn commit(&self) -> Result<()> {
        Ok(runtime::CommandBuffer::commit(self)?)
    }

    fn wait_until_completed(&self) -> Result<()> {
        runtime::CommandBuffer::wait_until_completed(self)?;
        match self.status() {
            CommandBufferStatus::Completed => Ok(()),
            status => {
                Err(format!("the command buffer ended {status:?}: {:?}", self.error()).into())
            }
        }
    }

    fn gpu_start_time(&self) -> Result<f64> {
        Ok(runtime::CommandBuffer::gpu_start_time(self)?)
    }

    fn gpu_end_time(&self) -> Result<f64> {
        Ok(runtime::CommandBuffer::gpu_end_time(self)?)
    }
}

impl crate::ComputeCommandEncoder<runtime::Device> for runtime::ComputeCommandEncoder {
    fn set_compute_pipeline_state(&mut self, state: &runtime::ComputePipelineState) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_compute_pipeline_state(
            self, state,
        )?)
    }

    fn set_buffer(&mut self, buffer: &runtime::Buffer, offset: u64, index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_buffer(
            self,
            Some(buffer),
            offset,
            index,
        )?)
    }

    fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_bytes(
            self, bytes, index,
        )?)
    }

    fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::dispatch_threadgroups(
            self,
            size(groups),
            size(threads),
        )?)
    }

    fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::dispatch_threads(
            self,
            size(threads),
            size(per_group),
        )?)
    }

    fn end_encoding(self) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::end_encoding(self)?)
    }
}
