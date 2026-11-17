//! The runtimes' compute harness: one set of Metal compute workloads run through a runtime's
//! Metal-shaped API on one GPU, with no guest. Each workload's result is checked against a CPU
//! reference before any time is reported. It is a regression gate between the runtimes, not a
//! measure against native Metal.
//!
//! The traits below are the Metal compute subset the workloads call, one method per Metal call.
//! A runtime implements them once: the old runtime in [`old`], which the cutover deletes with its
//! file.
//!
//! A workload is a [`plan::Plan`]: its buffers and their starting contents, the dispatches of one
//! run, and the contents its buffers must hold after the first run. [`run::Harness::run`] makes its
//! objects, checks the first run, then times the rest. The authored workloads ([`authored`]) are
//! MSL kernels committed as public fixtures (`crates/metal2vulkan-validation/fixtures/public/
//! harness/workloads.metal`, compiled by `scripts/regen-air-fixtures --dir harness`); their
//! library is the fixture's LLVM text, assembled by `llvm-as` from `PATH`.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-harness -- --runtime old [--runs 200] [--warmup 5] \
//!     [--only bandwidth,alu]
//! ```
//!
//! The command prints, per workload: the GPU time of a run (`GPUEndTime - GPUStartTime`) at the
//! 10th, 50th and 90th percentiles, the encoding thread's wall time per dispatch from the command
//! buffer's creation through `commit`'s return, and the process's CPU time per dispatch from
//! there to the wait's return (unix only), each the median of the timed runs; and the runtime's
//! gauges after the 100th and 200th timed runs. For each workload it also prints a line
//! `M2V-MATRIX time harness-<runtime>/<workload> <median GPU us>` for the matrix's runner.

use std::ptr::NonNull;

pub mod authored;
mod clocks;
pub mod old;
pub mod plan;
pub mod run;

#[cfg(test)]
mod tests;

/// A harness error: a runtime's own, boxed, or the harness's words.
pub type Error = Box<dyn std::error::Error>;

/// A harness result.
pub type Result<T> = std::result::Result<T, Error>;

/// An `MTLSize`: width, height and depth.
pub type Size = [u64; 3];

/// The storage mode of a buffer the workloads make (`MTLStorageMode`); the CPU cache mode and
/// hazard tracking are Metal's defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `MTLStorageModeShared`: the CPU reads and writes it through `contents`.
    Shared,
    /// `MTLStorageModePrivate`: the GPU's alone.
    Private,
}

/// A library's code.
#[derive(Clone, Copy, Debug)]
pub enum Code<'a> {
    /// A metallib, as `newLibraryWithData:error:` takes it.
    Metallib(&'a [u8]),
    /// AIR bitcode, which Metal takes in no call: the translator's library of it.
    Air(&'a [u8]),
}

/// `MTLDevice`: the device a runtime opened, and the types of the objects it makes.
pub trait Device: Sized {
    /// `MTLBuffer`.
    type Buffer: Buffer;
    /// `MTLLibrary`.
    type Library: Library<Self>;
    /// `MTLFunction`.
    type Function;
    /// `MTLComputePipelineState`.
    type ComputePipelineState;
    /// `MTLCommandQueue`.
    type CommandQueue: CommandQueue<Self>;
    /// `MTLCommandBuffer`.
    type CommandBuffer: CommandBuffer<Self>;
    /// `MTLComputeCommandEncoder`.
    type ComputeCommandEncoder: ComputeCommandEncoder<Self>;

    /// `name`.
    fn name(&self) -> String;

    /// `newBufferWithLength:options:` of `storage`.
    fn new_buffer(&self, length: u64, storage: Storage) -> Result<Self::Buffer>;

    /// `newLibraryWithData:error:` of a metallib; of AIR bitcode, the translator's library.
    fn new_library(&self, code: Code<'_>) -> Result<Self::Library>;

    /// `newComputePipelineStateWithFunction:error:`.
    fn new_compute_pipeline_state(
        &self,
        function: &Self::Function,
    ) -> Result<Self::ComputePipelineState>;

    /// `newCommandQueue`.
    fn new_command_queue(&self) -> Result<Self::CommandQueue>;

    /// The runtime's live gauges by name: how many of each thing it holds now. No Metal
    /// counterpart.
    fn gauges(&self) -> Vec<(&'static str, u64)>;
}

/// `MTLBuffer`.
pub trait Buffer {
    /// `contents`: the CPU address of the buffer's bytes; `None` for `Private` storage.
    fn contents(&self) -> Option<NonNull<u8>>;
}

/// `MTLLibrary`.
pub trait Library<D: Device> {
    /// `newFunctionWithName:`.
    fn new_function(&self, name: &str) -> Result<D::Function>;
}

/// `MTLCommandQueue`.
pub trait CommandQueue<D: Device> {
    /// `commandBuffer`.
    fn command_buffer(&self) -> Result<D::CommandBuffer>;
}

/// `MTLCommandBuffer`.
pub trait CommandBuffer<D: Device> {
    /// `computeCommandEncoder`: a serial compute encoder.
    fn compute_command_encoder(&self) -> Result<D::ComputeCommandEncoder>;

    /// `commit`.
    fn commit(&self) -> Result<()>;

    /// `waitUntilCompleted`; an error unless the command buffer's `status` is then
    /// `MTLCommandBufferStatusCompleted`.
    fn wait_until_completed(&self) -> Result<()>;

    /// `GPUStartTime`, in seconds.
    fn gpu_start_time(&self) -> Result<f64>;

    /// `GPUEndTime`, in seconds.
    fn gpu_end_time(&self) -> Result<f64>;
}

/// `MTLComputeCommandEncoder`.
pub trait ComputeCommandEncoder<D: Device> {
    /// `setComputePipelineState:`.
    fn set_compute_pipeline_state(&mut self, state: &D::ComputePipelineState) -> Result<()>;

    /// `setBuffer:offset:atIndex:`.
    fn set_buffer(&mut self, buffer: &D::Buffer, offset: u64, index: u32) -> Result<()>;

    /// `setBytes:length:atIndex:`.
    fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<()>;

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<()>;

    /// `dispatchThreads:threadsPerThreadgroup:`.
    fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<()>;

    /// `endEncoding`.
    fn end_encoding(self) -> Result<()>;
}
