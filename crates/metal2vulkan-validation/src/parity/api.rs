//! The Metal API a scenario is written against: Metal's calls, shaped as
//! the runtime's own vocabulary (`ResourceOptions`, `TextureDescriptor`, `Size`, `Origin`,
//! `BlitOption`), which Metal's side maps onto `objc2-metal` and the runtime's side passes on.
//!
//! Every call that Metal can refuse answers an [`Answer`]. On Metal a refused call is one its
//! API validation stops: the process ends there (the recorder runs each case in a child process
//! of its own), so a scenario ends a case at its first refusal on both sides.

use super::record::{Refusal, Value};
use metal2vulkan_runtime::{BlitOption, Origin, ResourceOptions, Size, TextureDescriptor};
use std::io::Write as _;
use std::ops::Range;
use std::time::{Duration, Instant};

/// A refused call: how, and the refusing side's own words, which are not compared.
#[derive(Debug)]
pub(crate) struct Refused {
    pub(crate) refusal: Refusal,
    pub(crate) detail: String,
}

/// What a call answers.
pub(crate) type Answer<T> = Result<T, Refused>;

/// Where a texture's bytes lie in a buffer: `sourceOffset`, `sourceBytesPerRow` and
/// `sourceBytesPerImage` (or the destination's).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) offset: u64,
    pub(crate) bytes_per_row: u64,
    pub(crate) bytes_per_image: u64,
}

/// The texels of a texture a copy reads or writes: `size` texels at `origin` of a slice and level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Region {
    pub(crate) slice: u64,
    pub(crate) level: u64,
    pub(crate) origin: Origin,
    pub(crate) size: Size,
}

/// A command buffer's times, in seconds of the host's time base, which Metal's are in
/// (`mach_absolute_time`'s, `CLOCK_UPTIME_RAW` on macOS): `GPUStartTime`, `GPUEndTime`,
/// `kernelStartTime` and `kernelEndTime`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Times {
    pub(crate) gpu_start: f64,
    pub(crate) gpu_end: f64,
    pub(crate) kernel_start: f64,
    pub(crate) kernel_end: f64,
}

/// A command buffer once it completed: its status record, and its times when the API tells
/// them (the runtime refuses GPU times on a device that writes no timestamps).
pub(crate) struct Completion {
    pub(crate) status: String,
    pub(crate) times: Option<Times>,
}

/// A committed public kernel fixture a scenario dispatches: its source row (its entry, and what
/// Metal compiles it from) and the AIR bitcode the runtime takes, as the runtime lane makes it.
pub(crate) struct Fixture {
    pub(crate) row: crate::source::SourceRow,
    pub(crate) bitcode: Vec<u8>,
}

/// Metal's API, as the scenarios use it. Each implementation is one device.
pub(crate) trait Api {
    type Buffer;
    type Texture;
    type Pipeline;
    type Queue;
    type CommandBuffer;
    type Event;
    type Fence;
    type Blit;
    type Compute;

    /// `newBufferWithLength:options:`.
    fn buffer(&self, length: u64, options: ResourceOptions) -> Answer<Self::Buffer>;
    /// Write `bytes` at the start of a `Shared` buffer's `contents`.
    fn write(&self, buffer: &Self::Buffer, bytes: &[u8]);
    /// A `Shared` buffer's `contents`, every byte.
    fn read(&self, buffer: &Self::Buffer) -> Vec<u8>;
    /// `newTextureWithDescriptor:`.
    fn texture(&self, descriptor: &TextureDescriptor) -> Answer<Self::Texture>;
    /// A compute pipeline state of the fixture's entry, from a library of its AIR.
    fn pipeline(&self, fixture: &Fixture) -> Answer<Self::Pipeline>;
    /// `newCommandQueue`.
    fn queue(&self) -> Answer<Self::Queue>;
    /// `commandBuffer`.
    fn command_buffer(&self, queue: &Self::Queue) -> Answer<Self::CommandBuffer>;
    /// `newSharedEvent`.
    fn event(&self) -> Answer<Self::Event>;
    /// `newFence`.
    fn fence(&self) -> Answer<Self::Fence>;
    /// `encodeSignalEvent:value:`.
    fn signal(&self, cb: &Self::CommandBuffer, event: &Self::Event, value: u64) -> Answer<()>;
    /// `encodeWaitForEvent:value:`.
    fn wait(&self, cb: &Self::CommandBuffer, event: &Self::Event, value: u64) -> Answer<()>;
    /// `blitCommandEncoder`.
    fn blit(&self, cb: &Self::CommandBuffer) -> Answer<Self::Blit>;
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    fn copy_buffer(
        &self,
        blit: &mut Self::Blit,
        from: (&Self::Buffer, u64),
        to: (&Self::Buffer, u64),
        size: u64,
    ) -> Answer<()>;
    /// `fillBuffer:range:value:`.
    fn fill(
        &self,
        blit: &mut Self::Blit,
        buffer: &Self::Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Answer<()>;
    /// `copyFromBuffer:...toTexture:...options:`.
    fn copy_to_texture(
        &self,
        blit: &mut Self::Blit,
        from: (&Self::Buffer, Layout),
        to: (&Self::Texture, Region),
        options: BlitOption,
    ) -> Answer<()>;
    /// `copyFromTexture:...toBuffer:...options:`.
    fn copy_from_texture(
        &self,
        blit: &mut Self::Blit,
        from: (&Self::Texture, Region),
        to: (&Self::Buffer, Layout),
        options: BlitOption,
    ) -> Answer<()>;
    /// `updateFence:` on a blit encoder.
    fn blit_update_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()>;
    /// `waitForFence:` on a blit encoder.
    fn blit_wait_fence(&self, blit: &mut Self::Blit, fence: &Self::Fence) -> Answer<()>;
    /// `endEncoding` of a blit encoder.
    fn end_blit(&self, blit: Self::Blit) -> Answer<()>;
    /// `computeCommandEncoder`.
    fn compute(&self, cb: &Self::CommandBuffer) -> Answer<Self::Compute>;
    /// `setComputePipelineState:`, each buffer at offset 0 at its index, and
    /// `dispatchThreads:threadsPerThreadgroup:` of `threads` in groups of one thread.
    fn dispatch(
        &self,
        compute: &mut Self::Compute,
        pipeline: &Self::Pipeline,
        buffers: &[(&Self::Buffer, u64)],
        threads: Size,
    ) -> Answer<()>;
    /// `updateFence:` on a compute encoder.
    fn compute_update_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()>;
    /// `waitForFence:` on a compute encoder.
    fn compute_wait_fence(&self, compute: &mut Self::Compute, fence: &Self::Fence) -> Answer<()>;
    /// `endEncoding` of a compute encoder.
    fn end_compute(&self, compute: Self::Compute) -> Answer<()>;
    /// `commit`.
    fn commit(&self, cb: &Self::CommandBuffer) -> Answer<()>;
    /// Wait for the command buffer to complete (on Metal under the hang protocol of
    /// `gpu::wait`), then its status and times.
    fn completion(&self, cb: Self::CommandBuffer) -> Result<Completion, String>;
    /// `GPUStartTime`, `GPUEndTime`, `kernelStartTime` and `kernelEndTime` now.
    fn times(&self, cb: &Self::CommandBuffer) -> Answer<Times>;
    /// The host's time now, in seconds of the time base command buffers' times count.
    fn host_time(&self) -> f64;
}

/// The records of one case, made as the scenario runs it.
pub(crate) struct Observation {
    case: &'static str,
    /// In a recorder's child process: each record and each call about to be made is printed
    /// too, so the parent knows the call Metal's API validation stopped.
    child: bool,
    records: Vec<(String, Value)>,
    started: Instant,
    gpu: Option<Duration>,
}

/// The line before each call a child makes ([`Observation::call`]).
pub(crate) const AT: &str = "PARITY-AT ";
/// The line of each record a child makes.
pub(crate) const RECORD: &str = "PARITY-RECORD ";

impl Observation {
    pub(crate) fn new(case: &'static str, child: bool) -> Observation {
        Observation {
            case,
            child,
            records: Vec::new(),
            started: Instant::now(),
            gpu: Some(Duration::ZERO),
        }
    }

    /// Make the call `op`, recording that it was taken or how it was refused: its value, or
    /// `None` when refused, which ends the case.
    pub(crate) fn call<T>(&mut self, op: &str, call: impl FnOnce() -> Answer<T>) -> Option<T> {
        let name = format!("{}/{op}", self.case);
        if self.child {
            println!("{AT}{name}");
            let _ = std::io::stdout().flush();
        }
        match call() {
            Ok(value) => {
                self.push(name, Value::Done);
                Some(value)
            }
            Err(refused) => {
                self.push(
                    name,
                    Value::Refused {
                        refusal: refused.refusal,
                        detail: refused.detail,
                    },
                );
                None
            }
        }
    }

    /// Record `bytes` read back as `op`.
    pub(crate) fn bytes(&mut self, op: &str, bytes: &[u8]) {
        let name = format!("{}/{op}", self.case);
        self.push(name, Value::bytes(bytes));
    }

    /// Record a command buffer's completion as `op`, and count its GPU time (`GPUStartTime` to
    /// `GPUEndTime`).
    pub(crate) fn completed(&mut self, op: &str, completion: Completion) {
        let gpu = completion
            .times
            .map(|times| Duration::from_secs_f64((times.gpu_end - times.gpu_start).max(0.0)));
        self.gpu = match (self.gpu, gpu) {
            (Some(total), Some(gpu)) => Some(total + gpu),
            _ => None,
        };
        let name = format!("{}/{op}", self.case);
        self.push(
            name,
            Value::Completed {
                status: completion.status,
            },
        );
    }

    fn push(&mut self, name: String, value: Value) {
        if self.child {
            let line = serde_json::to_string(&(&name, &value)).expect("a record serializes");
            println!("{RECORD}{line}");
            let _ = std::io::stdout().flush();
        }
        self.records.push((name, value));
    }

    /// The case's records, and its timing line (report only: wall time beside GPU time, where
    /// the API told it for every command buffer).
    pub(crate) fn finish(self, scenario: &str) -> Vec<(String, Value)> {
        let gpu = self.gpu.map_or("unknown".to_owned(), |gpu| {
            format!("{} us", gpu.as_micros())
        });
        eprintln!(
            "parity time: {scenario}/{} wall {} us, gpu {gpu}",
            self.case,
            self.started.elapsed().as_micros()
        );
        self.records
    }
}
