//! The runtimes' compute harness: one set of Metal compute workloads run through a runtime's
//! Metal-shaped API on one GPU, with no guest. Each workload's result is checked against a CPU
//! reference before any time is reported. It is a regression gate between the runtimes, not a
//! measure against native Metal.
//!
//! The traits below are the Metal compute subset the workloads call, one method per Metal call,
//! with the blit encoder's copies and fill.
//! A runtime implements them once: the old runtime in [`old`], which the cutover deletes with its
//! file, and the new runtime in [`rt`]. A workload a runtime answers with a typed refusal (a
//! known gap or a device limit, [`Refused`]) is reported by name as refused, not as a failure.
//!
//! A workload is a [`plan::Plan`]: its buffers and their starting contents, the dispatches of one
//! run and how many go in each command buffer, the contents its buffers must hold after the first
//! run, and the buffers runtimes compare. Most workloads put a run in one command buffer; `frame`
//! and `frame_5cb` are frames of one and of five command buffers of four small dispatches each,
//! the `burst_` workloads bursts of 160 and 2,000, and `long_4cb` four long command buffers.
//! `constant_managed` reads a `Managed` table of constants whose first 256 bytes the CPU writes
//! again before each run (`plan::BufferPlan::modified`), and `constant_private` the same reads
//! from a `Private` table, its reference.
//! A [`run::Session`] makes its objects, checks the first run, then times the rest. The authored
//! workloads ([`authored`]) are MSL kernels committed as public fixtures
//! (`crates/metal2vulkan-validation/fixtures/public/harness/workloads.metal`, compiled by
//! `scripts/regen-air-fixtures --dir harness`), each with a CPU reference; their library is the
//! fixture's LLVM text, assembled by `llvm-as` from `PATH`, which also holds the harness's own
//! copy kernel. Described workloads ([`description`]) are kernels of metallibs in a directory,
//! named by a description file; both are the command's arguments, and nothing in the repository
//! names them. They have no CPU reference: each runtime's outputs must match the first
//! runtime's. The creation workloads ([`authored::CREATES`]) run with the authored ones: `Shared`
//! buffers of 4 KiB, 1 MiB and 64 MiB, each made and released in turn over the range the last
//! one left ([`run::Session::create`]), each checked to start zero; and compute pipeline states of
//! one content (`create_state`), the first of a content the device has not seen timed apart
//! with one dispatch of it through to completion (`first`: a runtime may compile its pipeline in
//! the creation or at the first dispatch, and the sum holds both; the row also shows the creation
//! alone), then 1,000 of a content it has, each made and released in turn.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-harness -- --runtime old|new[,old|new...] [--runs 200] \
//!     [--warmup 5] [--no-settle] [--only NAME,NAME...] [--workloads DIRECTORY --description FILE]
//! ```
//!
//! After a workload's first run, two of its runs are in flight: each run is committed while the
//! GPU runs the one before it, so the GPU does not idle between runs, and a run is waited for
//! once, at its last command buffer, as a program waits for a burst it committed. Before the timed
//! runs, the command warms the workload up: at least `--warmup` runs and half a second, then
//! until the medians of its last two windows of 10 runs are within 3% of each other, for at most
//! five seconds, so that an integrated GPU's clock has risen before it is timed. `--no-settle`
//! runs `--warmup` runs alone, for a run that counts rather than times (its pipelines).
//!
//! The command prints, per workload and runtime: the warm-up's runs (marked `~` where the time
//! did not settle), the GPU time of a run (from its first command buffer's `GPUStartTime` to its
//! last's `GPUEndTime`) at the 10th, 50th and
//! 90th percentiles of the timed runs, the encoding thread's wall time per dispatch from the
//! command buffer's creation through `commit`'s return, and the process's CPU time per dispatch
//! from there to the return of the wait for the run before (unix only), each the median of the
//! timed runs; where the workload asks for it ([`plan::Plan::commit_to_complete`]), the time of
//! each timed command buffer from just before its `commit` until its completed handler ran, at
//! the 50th and 99th percentiles of all of them, and, for a run of several command buffers, on a
//! line of its own at those percentiles by the command buffer's place in its run; and the
//! runtime's gauges after the 100th and 200th timed runs. Per creation
//! workload and runtime it prints the warm-up, the wall time of `newBufferWithLength:options:` at
//! the same percentiles, and the gauges. Beside each row it prints lines for the matrix's runner
//! (docs/VALIDATION.md, "The matrix"):
//!
//! ```text
//! M2V-MATRIX time harness-<runtime>/<workload> <steady us>
//! M2V-MATRIX gauge harness-<runtime>/<workload>/<gauge>@100 <value>
//! M2V-MATRIX gauge harness-<runtime>/<workload>/<gauge>@200 <value>
//! ```
//!
//! where the steady time ([`run::steady`], the table's `steady` column) is the lowest median of
//! the timed runs' consecutive windows of 10, with at least 5 timed runs. A workload the device
//! sizes ([`plan::Scaled`]) prints the size it ran at on a line of its own, and its matrix time
//! is that time scaled to its base size, so its row keeps one scale whatever size it ran at. After
//! the plan
//! workloads, per runtime and set (`authored` or `described`), it prints how many `VkPipeline`s
//! they created and how many workloads it refused, and where the runtime counts them, its
//! first-use waits ([`Waits`]): the count, and their total and longest time in microseconds.
//! Each plan workload also prints the runtime's own counts of its work ([`Device::counts`]) over
//! one run after the timed ones, run alone; the table shows that run's submissions
//! ([`Device::submissions`]), which timing decides, so they are no matrix metric:
//!
//! ```text
//! M2V-MATRIX metric harness-<runtime>/<set>/pipelines_created <n>
//! M2V-MATRIX metric harness-<runtime>/<set>/refused <n>
//! M2V-MATRIX metric harness-<runtime>/<set>/first_use_waits <n>
//! M2V-MATRIX time harness-<runtime>/<set>/first_use_wait_total <us>
//! M2V-MATRIX time harness-<runtime>/<set>/first_use_wait_longest <us>
//! M2V-MATRIX metric harness-<runtime>/<workload>/<count>_per_run <n>
//! ```
//!
//! and once every workload has run, where the runtime's gauges outlive its device,
//! `M2V-MATRIX gauge harness-<runtime>/device/<gauge>@drop <value>` read after the device dropped.
//! A workload that fails prints `M2V-FAILED harness-<runtime>/<workload>` beside its `FAILED`
//! line, for the runtime it failed on ([`Failed`]), or for each runtime where it names none.

use std::ptr::NonNull;

pub mod authored;
mod clocks;
pub mod description;
pub mod old;
pub mod plan;
pub mod rt;
pub mod run;

#[cfg(test)]
mod tests;

/// A harness error: a runtime's own, boxed, or the harness's words.
pub type Error = Box<dyn std::error::Error>;

/// A runtime's typed refusal of a workload: a known gap or one of the device's limits, which the
/// runtime answers instead of running it. The command reports the workload by name as refused,
/// apart from its failures.
#[derive(Debug)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// A workload's failure on one runtime, other than a refusal: the runtime's name and why.
#[derive(Debug)]
pub struct Failed {
    /// The runtime's name (`old` or `new`).
    pub runtime: String,
    /// Why it failed.
    pub why: Error,
}

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.runtime, self.why)
    }
}

impl std::error::Error for Failed {}

/// A runtime's first-use waits so far: encoding calls that waited while the `VkPipeline` they
/// needed was made ([`Device::first_use_waits`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Waits {
    /// How many.
    pub count: u64,
    /// Their total time, in nanoseconds.
    pub nanos: u64,
    /// The longest since the device was made, in nanoseconds.
    pub longest_nanos: u64,
}

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
    /// `MTLStorageModeManaged`: the CPU writes it through `contents` and calls
    /// `didModifyRange:`, which brings the GPU's copy up to date. The new runtime has no such
    /// mode; its harness keeps the GPU's copy in a `Private` buffer (`rt`).
    Managed,
}

/// A library's code.
#[derive(Clone, Copy, Debug)]
pub enum Code<'a> {
    /// A metallib, as `newLibraryWithData:error:` takes it.
    Metallib(&'a [u8]),
    /// AIR bitcode, which Metal takes in no call: the translator's library of it.
    Air(&'a [u8]),
}

/// `MTLDevice`: the device a runtime opened, and its buffers.
pub trait Device: Sized {
    /// `MTLBuffer`.
    type Buffer: Buffer;

    /// `name`.
    fn name(&self) -> String;

    /// `newBufferWithLength:options:` of `storage`.
    fn new_buffer(&self, length: u64, storage: Storage) -> Result<Self::Buffer>;

    /// The runtime's live gauges by name: how many of each thing it holds now. No Metal
    /// counterpart.
    fn gauges(&self) -> Vec<(&'static str, u64)>;

    /// What reads the gauges once the device is dropped, where they outlive it; `None` where
    /// they end with it. No Metal counterpart.
    fn gauges_after_drop(&self) -> Option<GaugeReader>;

    /// How many `VkPipeline`s the runtime has created on the device so far, by its own count. No
    /// Metal counterpart.
    fn pipelines_created(&self) -> u64;

    /// The runtime's own counts of its work on the device so far, by name: those it keeps one
    /// total of among encoders, barriers and bytes copied, which a run's commands alone decide.
    /// No Metal counterpart.
    fn counts(&self) -> Vec<(&'static str, u64)>;

    /// The runtime's `vkQueueSubmit` calls on the device so far. A run's share depends on how
    /// many of its command buffers the runtime found ready together, which timing decides. No
    /// Metal counterpart.
    fn submissions(&self) -> u64;

    /// The runtime's first-use waits so far; `None` where it does not count them. No Metal
    /// counterpart.
    fn first_use_waits(&self) -> Option<Waits>;
}

/// A reader of a runtime's gauges that outlives its device ([`Device::gauges_after_drop`]).
pub type GaugeReader = Box<dyn Fn() -> Vec<(&'static str, u64)>>;

/// The compute calls of `MTLDevice`: its libraries, pipelines and queues, and the types of the
/// objects they make.
pub trait ComputeDevice: Device {
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
    /// `MTLBlitCommandEncoder`.
    type BlitCommandEncoder: BlitCommandEncoder<Self>;
    /// `MTLTexture`.
    type Texture: Texture;
    /// `MTLSamplerState`.
    type SamplerState;

    /// `newLibraryWithData:error:` of a metallib; of AIR bitcode, the translator's library.
    fn new_library(&self, code: Code<'_>) -> Result<Self::Library>;

    /// `newComputePipelineStateWithFunction:error:`.
    fn new_compute_pipeline_state(
        &self,
        function: &Self::Function,
        private: &[Self::Function],
    ) -> Result<Self::ComputePipelineState>;

    /// `newCommandQueue`.
    fn new_command_queue(&self) -> Result<Self::CommandQueue>;

    /// `texture`'s texture, made as its [`TextureSource`](plan::TextureSource) says; a runtime
    /// that makes no such texture refuses it ([`Refused`]).
    fn new_texture(&self, texture: &plan::TexturePlan) -> Result<Self::Texture>;

    /// `newTextureViewWithPixelFormat:textureType:levels:slices:` of `parent`: `texture`'s pixel
    /// format and type, its one level and `slices` (its [`TextureSource::View`](plan::TextureSource::View)).
    fn new_texture_view(
        &self,
        parent: &Self::Texture,
        texture: &plan::TexturePlan,
        slices: std::ops::Range<u32>,
    ) -> Result<Self::Texture>;

    /// `newSamplerStateWithDescriptor:` of `sampler`.
    fn new_sampler_state(&self, sampler: plan::SamplerPlan) -> Result<Self::SamplerState>;
}

/// `MTLTexture`.
pub trait Texture {
    /// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:` of all of level 0
    /// of `slice`, `width` by `height` texels, rows `bytes_per_row` apart in `bytes`.
    fn replace_region(
        &self,
        slice: u32,
        size: [u32; 2],
        bytes: &[u8],
        bytes_per_row: u64,
    ) -> Result<()>;
}

/// `MTLBuffer`.
pub trait Buffer {
    /// `contents`: the CPU address of the buffer's bytes; `None` for `Private` storage.
    fn contents(&self) -> Option<NonNull<u8>>;

    /// `didModifyRange:` of the `length` bytes at `offset`: the CPU wrote them. Nothing for
    /// storage other than `Managed`.
    fn did_modify_range(&self, offset: u64, length: u64) -> Result<()>;
}

/// `MTLLibrary`.
pub trait Library<D: ComputeDevice> {
    /// `newFunctionWithName:`, or with `constants` `newFunctionWithName:constantValues:error:`.
    fn new_function(&self, name: &str, constants: &[plan::Constant]) -> Result<D::Function>;
}

/// `MTLCommandQueue`.
pub trait CommandQueue<D: ComputeDevice> {
    /// `commandBuffer`.
    fn command_buffer(&self) -> Result<D::CommandBuffer>;
}

/// `MTLCommandBuffer`.
pub trait CommandBuffer<D: ComputeDevice> {
    /// `computeCommandEncoder`: a serial compute encoder.
    fn compute_command_encoder(&self) -> Result<D::ComputeCommandEncoder>;

    /// `blitCommandEncoder`.
    fn blit_command_encoder(&self) -> Result<D::BlitCommandEncoder>;

    /// `commit`.
    fn commit(&self) -> Result<()>;

    /// `waitUntilCompleted`; an error unless the command buffer's `status` is then
    /// `MTLCommandBufferStatusCompleted`.
    fn wait_until_completed(&self) -> Result<()>;

    /// `GPUStartTime`, in seconds.
    fn gpu_start_time(&self) -> Result<f64>;

    /// `GPUEndTime`, in seconds.
    fn gpu_end_time(&self) -> Result<f64>;

    /// `addCompletedHandler:`, with a handler that reads nothing of the command buffer.
    fn add_completed_handler(&self, handler: Box<dyn FnOnce() + Send>) -> Result<()>;
}

/// `MTLBlitCommandEncoder`.
pub trait BlitCommandEncoder<D: ComputeDevice> {
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:`
    /// `destinationSlice:destinationLevel:destinationOrigin:` of `copy`, from `source` to
    /// `destination`, the textures it names.
    fn copy_from_texture(
        &mut self,
        copy: &plan::TextureCopy,
        source: &D::Texture,
        destination: &D::Texture,
    ) -> Result<()>;

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:`
    /// `toTexture:destinationSlice:destinationLevel:destinationOrigin:` of `copy`, from `buffer`
    /// into `texture`, the two it names.
    fn copy_from_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        buffer: &D::Buffer,
        texture: &D::Texture,
    ) -> Result<()>;

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:`
    /// `destinationOffset:destinationBytesPerRow:destinationBytesPerImage:` of `copy`, from
    /// `texture` into `buffer`, the two it names.
    fn copy_from_texture_to_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        texture: &D::Texture,
        buffer: &D::Buffer,
    ) -> Result<()>;

    /// `fillBuffer:range:value:`.
    fn fill_buffer(
        &mut self,
        buffer: &D::Buffer,
        range: std::ops::Range<u64>,
        value: u8,
    ) -> Result<()>;

    /// `synchronizeResource:` of a buffer.
    fn synchronize_resource(&mut self, buffer: &D::Buffer) -> Result<()>;

    /// `endEncoding`.
    fn end_encoding(self) -> Result<()>;
}

/// `MTLComputeCommandEncoder`.
pub trait ComputeCommandEncoder<D: ComputeDevice> {
    /// `setComputePipelineState:`.
    fn set_compute_pipeline_state(&mut self, state: &D::ComputePipelineState) -> Result<()>;

    /// `setBuffer:offset:atIndex:`.
    fn set_buffer(&mut self, buffer: &D::Buffer, offset: u64, index: u32) -> Result<()>;

    /// `setBytes:length:atIndex:`.
    fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<()>;

    /// `setTexture:atIndex:`.
    fn set_texture(&mut self, texture: &D::Texture, index: u32) -> Result<()>;

    /// `setSamplerState:atIndex:`.
    fn set_sampler_state(&mut self, sampler: &D::SamplerState, index: u32) -> Result<()>;

    /// `setThreadgroupMemoryLength:atIndex:`.
    fn set_threadgroup_memory_length(&mut self, length: u32, index: u32) -> Result<()>;

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<()>;

    /// `dispatchThreads:threadsPerThreadgroup:`.
    fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<()>;

    /// `endEncoding`.
    fn end_encoding(self) -> Result<()>;
}
