//! A new resource's first GPU work: a texture's layout move, the zeros it or a buffer reads. The
//! device's next submission does it (`runtime/src/texture/first_move.rs`), so making one waits
//! for no GPU work, on the device the cfg selects, under the validation layer.
//!
//! - **Making waits for nothing** (the consumer's PS ask: Metal made textures in 102 µs and 18 µs
//!   with a 2 s kernel still scheduled, where the runtime blocked 2.07 s). GPU busy work is
//!   committed that spins until the test releases it, at most about 2 s (in command buffers of a
//!   quarter of a second each: a driver may end a longer submission), and writes an end mark when
//!   it ends. Then textures are made down every path (a texture of its own, `Private` and
//!   `Shared`; over a buffer; over a no-copy buffer; in a heap; over detached bytes, a stand-in), a
//!   heap's buffer and new buffers are made (a `Private` one with bytes among them, and a `Managed`
//!   one), a `Managed` buffer's `did_modify_range` is called (on a device without unified memory
//!   each copies to the buffer's own bytes by the next submission), and a no-copy buffer with a
//!   texture over it is rebound. The end mark reads 0 as each call returns. Afterwards every
//!   texture and buffer holds what Metal's holds: zero where new, the bytes it was made with, the
//!   bytes it lies over where it aliases, what a blit wrote.
//! - **Making waits for no submission's replay.** A command buffer whose deferred command buffers
//!   take long to replay (`recorder`) and to submit is committed, and a `Private` texture made
//!   while its submission replays and submits them waits for no submission's place
//!   (`WorkCounts::registrations_waited` stays 0): a submission replays them before it takes the
//!   new resources' first work, and from its take until its `vkQueueSubmit` returned (MoltenVK
//!   encodes the Metal commands inside it) it holds the texture slots' writes, not the texture's
//!   registration (`WorkCounts::heap_slot_writes_deferred`), or, on a device whose heap sets take
//!   no writes while pending, the registrations, which then wait only for that span. The making's
//!   time is printed, not bounded: it holds the host's and the driver's waits. On lavapipe under
//!   the validation layer, beside 128 busy threads on 32 CPUs, its `vkCreateImage` slept up to
//!   6 ms on the layer's lock, which the replay's own Vulkan calls on the recorder's thread held
//!   while the host ran another thread, and once waited 8 ms for a CPU. It makes a
//!   `Private` texture: its move waits for a submission, so its making registers it, where a new
//!   `Shared` texture's move is made on the host on a device with host image copies
//!   (`texture/host_copy.rs`), registering nothing, whatever the order. One is made before,
//!   untimed: a device's first texture making took 4 to 9 ms on RADV
//!   and the Iris Xe, 32 ms on a loaded Iris Xe, waiting for nothing.
//! - **The zeros come before every use, on every queue** (the coordinator's conditions). A new
//!   `Private` buffer whose memory held other bytes is read through its address unnamed, and
//!   through its heap made resident, by a command buffer on another queue, after a command
//!   buffer on the first queue took its fill: zero. `Shared` and `Managed` bytes read zero
//!   through `contents` before any submission. A submission that carries the fill and fails
//!   (`commit_failed`, a failing submission check) still runs it: a later read is zero. Bytes a
//!   command buffer writes through a resource placed in a new heap, and through one aliasing it,
//!   stay: the heap's fill comes before its first command.
//! - **A heap texture's `replace_region` waits for nothing**: its copy is staged for the device's
//!   next submission; a later command buffer reads the later of two calls' bytes, and `get_bytes`
//!   the staged bytes. A new heap texture's staged copy lands where that submission also carries
//!   the texture's first move, a command buffer's or `get_bytes`'s, in one `vkQueueSubmit`.
//! - **Neutrals and sparse textures wait for nothing**: kernels whose texture and texel-buffer
//!   slots nothing is bound to are encoded and committed, and a sparse texture is made, while the
//!   busy work runs, with no internal submission; the kernels then complete cleanly and the sparse
//!   texture copies out.
//! - **A `Shared` or `Managed` texture's CPU copies wait for nothing** (the consumer's gamma-table
//!   upload and frame-target read). Where the device copies on the host, `replace_region` and
//!   `get_bytes` return while the busy work runs and make no submission; on every device their
//!   bytes are ordered with the command buffers before and after.
//! - **A display-sized texture copies texel exact** (the consumer's 1080p frame-target read took
//!   1.0 to 2.55 s on an RTX 5080, its texture copying on the host from memory the host does not
//!   cache). A 1920 x 1080 `BGRA8Unorm` `Shared` and `Managed` texture's `get_bytes` reads a
//!   completed blit's texels exactly, and a later command buffer reads `replace_region`'s, each
//!   call timed and its path printed (`FRAME-COPY`); with a 2 s kernel running, its first read
//!   returns while the kernel runs where it copies on the host or the staging queue, and the line
//!   gives the best, median and worst of the reads that did.
//! - **A CPU copy never waits behind queued work** (the consumer's guest hang). With a command
//!   buffer on each of four queues waiting on an event the calling thread signals only afterwards,
//!   `get_bytes` and `replace_region` return where the texture copies on the host or the staging
//!   queue (`STAGING-HANG`).
//! - **A multisample `Shared` texture starts by a clear** (RADV refused a 4x `BGRA8Unorm` render
//!   target, its zeros copied into it on the host, which Vulkan forbids for a multisampled image).
//!   Such textures are made, started neither on the host nor on the staging queue, and resolve to
//!   zero through a pass that loads them.

use super::commit_work::relay_marks;
use super::degrade_notices::{
    assert_once_allowing, forced, made_compute_kinds, shared_in_system_memory,
};
use super::gpu_times::{disagreeable_kinds, read_kinds};
use super::host_memory::{no_copy, not_importable, recording, Mapping, Seen, GUEST_PAGE};
use super::rebind_reach::{node, walk, Outputs, Uses, A, POINTER_WALKER};
use super::rebinds::rebind;
use super::sparse_textures::{lacks_sparse, SparseFacts};
use super::{
    buffer_with, case_bound_value_kinds, case_making_kinds, encode_case, pipeline_state,
    public_case, read_buffer, CaseResources, Messages, WAIT,
};
use crate::air_library::AirLibrary;
use crate::runtime_lane::host_memory::on_device;
use metal2vulkan::bind::{StatusReport, TileMapping};
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    host_time, BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus,
    CommandQueue, CommitFailure, ComputePipelineState, CpuCacheMode, DebugMessage, Degradation,
    Device, Error, HazardTrackingMode, HeapDescriptor, HeapType, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, Size, StorageMode, StoreAction,
    Submission, SubmissionCheck, Texture, TextureDescriptor, TimedOut,
};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The most GPU time the busy work takes before it gives up, where nothing released it
/// ([`release`]): far longer than every call a test makes while it runs.
const BUSY: Duration = Duration::from_secs(2);

/// The command buffers the busy work is committed as, one after another on its queue, each a spin
/// of at most `BUSY / LINKS` at the kernel's measured rate: a driver may end any one submission
/// that runs too long. Linux 7.0's amdgpu ends a job after 2 s by default (`lockup_timeout` unset),
/// resetting the ring and losing the submitting context's device: the whole of [`BUSY`] in one
/// command buffer did so on an RX 7900 XTX and a Radeon 780M (Mesa 26.0.8, kernel 7.0.0-38).
const LINKS: u32 = 8;

/// The least GPU time the busy kernel's rate is measured over, at that rate.
pub(super) const MEASURED: Duration = Duration::from_millis(50);

/// The most runs at one cap the busy kernel's rate is measured by.
pub(super) const RUNS: usize = 3;

/// The textures' side, format and bytes per row: one guest page of texels.
const SIDE: u64 = 64;
const PF: PixelFormat = PixelFormat::RGBA8Unorm;
const ROW: u64 = 4 * SIDE;
const BYTES: usize = (ROW * SIDE) as usize;

/// The end mark the busy work's last command writes: four bytes of 1.
pub(super) const ENDED: u32 = 0x0101_0101;

/// Public cases whose kernels read texture and texture-buffer slots that nothing is bound to here:
/// a sampled texture's, a storage texture's and a storage texel buffer's (`kernel_textures`,
/// `buffer_textures`).
const UNBOUND_CASES: [&str; 3] = [
    "sample-each-of-three-textures-through-one-dynamic-array-index",
    "write-through-a-storage-texture-selected-per-thread",
    "write-four-bytes-into-a-storage-texel-buffer",
];

/// The busy kernel: the owned fixture `rfs_agx_yield`'s `agx_yield_lookback` as one threadgroup
/// that takes ticket 1 and spins for the flag of ticket 0 (the busy work's first word) until the
/// test raises it ([`release`]) or its cap of loads runs out, then gives up (as
/// `framebuffer_fetch::storage_forms` holds a copy), and its cap for about `BUSY / LINKS`: the
/// busy work is [`LINKS`] such spins, a command buffer each, about [`BUSY`] in all. Shared with
/// `compressed_decode`'s lane of writes behind work in flight.
///
/// Its cap, not the flag, ended it on an RTX 3090 (NVIDIA 615.71.09) when it was one spin of
/// [`BUSY`]: alone, each of six busy works released 0 to 50 ms into its spin ran its whole cap,
/// 2.0 s on the GPU, and published nothing.
pub(super) struct Busy {
    state: ComputePipelineState,
    /// The loads each spin makes at most.
    cap: u32,
    /// The kinds its pipeline state's making announced where the device had not
    /// (`made_compute_kinds`), and its measurement's reads of GPU times (`gpu_times::read_kinds`).
    pub(super) made: BTreeSet<Degradation>,
}

impl Busy {
    /// The kernel, its cap the loads of `BUSY / LINKS` at its rate: the fastest of every measured run,
    /// in loads per second of the run's time on the GPU where the device answers it
    /// ([`gpu_took`]). The cap grows sixteenfold until a run takes [`MEASURED`] at that rate, and
    /// a cap whose run takes it is run again, up to [`RUNS`] times. The GPU's other work only
    /// slows a run, so the fastest is the kernel's own rate unless every run was slowed; and a
    /// run's wall time held that work too: on an RTX 3090 (NVIDIA 615.71.09) beside the lane's
    /// other tests, which run such kernels too, a run of 4096 loads, under a millisecond on the
    /// GPU, took about 200 ms from its commit, as long as another process's run, and the cap
    /// scaled from that wall time, 37559 loads, spun for about 7 ms, less than a call took there
    /// (7.6 ms); and three runs of 65536 loads, about 12 ms alone, each took 200 to 420 ms from
    /// their GPU start to their GPU end. The first run is not measured: it holds the
    /// device's first use of the pipeline and of a command buffer, which on a loaded host took
    /// past [`MEASURED`] by themselves, and a cap scaled from it spun for microseconds (lavapipe in
    /// master's gate at 409546f2: a test of a 0.5 s busy work took 0.35 s whole, and its calls
    /// saw the end mark).
    pub(super) fn new(device: &Device, queue: &CommandQueue) -> Busy {
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rfs_agx_yield.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("agx_yield_lookback")
            .unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut made = made_compute_kinds(device, function.function_descriptor());
        made.extend(read_kinds(device));
        let mut busy = Busy {
            state,
            cap: 1 << 12,
            made,
        };
        busy.run(device, queue);
        // The fastest rate of every run, in loads per second, and each run's loads, time on the
        // GPU and time from commit to completion, for the printed line.
        let mut rate = 0.0_f64;
        let mut runs = Vec::new();
        let at_rate = |cap: u32, rate: f64| f64::from(cap) / rate >= MEASURED.as_secs_f64();
        loop {
            for _ in 0..RUNS {
                let (took, wall) = busy.run(device, queue);
                rate = rate.max(f64::from(busy.cap) / took.as_secs_f64());
                runs.push((busy.cap, took, wall));
                if !at_rate(busy.cap, rate) {
                    break;
                }
            }
            if at_rate(busy.cap, rate) || busy.cap >= 1 << 28 {
                break;
            }
            busy.cap <<= 4;
        }
        let link = BUSY / LINKS;
        busy.cap = (rate * link.as_secs_f64()).clamp(1.0, f64::from(u32::MAX)) as u32;
        let runs: Vec<String> = (runs.iter())
            .map(|(cap, took, wall)| format!("{cap} in {took:.2?} ({wall:.2?})"))
            .collect();
        println!(
            "BUSY-CAP on {}: {LINKS} command buffers of {} loads ({link:?} each), at the fastest \
             rate of the runs of loads in GPU time (from commit to completion): {}",
            device.name(),
            busy.cap,
            runs.join(", ")
        );
        busy
    }

    /// Run one spin of the current cap, released by nothing, and its end mark: the time it took
    /// on the GPU ([`gpu_took`]), and from its commit to its completion.
    fn run(&self, device: &Device, queue: &CommandQueue) -> (Duration, Duration) {
        let at = Instant::now();
        let words = buffer_with(device, &[0; 8]).unwrap();
        let cap = buffer_with(device, &self.cap.to_le_bytes()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        self.spin(device, &command_buffer, &words, &cap);
        end_mark(&command_buffer, &words);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let wall = at.elapsed();
        assert_eq!(command_buffer.error(), None, "the busy work");
        assert_eq!(read_mark(&words), ENDED, "the busy work's end mark");
        (gpu_took(&command_buffer, wall), wall)
    }

    /// Commit the busy work on `queue`: [`LINKS`] command buffers, each a spin on the same flag,
    /// the last then a blit that writes the end mark into the word after the flag, ordered after
    /// its spin by the buffer both use. Once the flag is raised every spin still to run ends at
    /// its first load. The command buffers, and the buffer of the flag and the mark.
    pub(super) fn commit(&self, device: &Device, queue: &CommandQueue) -> (BusyWork, Buffer) {
        let words = buffer_with(device, &[0; 8]).unwrap();
        let cap = buffer_with(device, &self.cap.to_le_bytes()).unwrap();
        let links = (0..LINKS)
            .map(|link| {
                let command_buffer = queue.command_buffer().unwrap();
                self.spin(device, &command_buffer, &words, &cap);
                if link + 1 == LINKS {
                    end_mark(&command_buffer, &words);
                }
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        (BusyWork { links }, words)
    }

    /// Encode one spin into `command_buffer`: one threadgroup with a ticket of its own (the kernel
    /// takes the ticket's value and raises it), spinning on `words`' flag for at most `cap`'s loads.
    fn spin(&self, device: &Device, command_buffer: &CommandBuffer, words: &Buffer, cap: &Buffer) {
        let ticket = buffer_with(device, &1u32.to_le_bytes()).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&ticket), 0, 0).unwrap();
        encoder.set_buffer(Some(words), 0, 1).unwrap();
        encoder.set_buffer(Some(cap), 0, 2).unwrap();
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threadgroups(one, one).unwrap();
        encoder.end_encoding().unwrap();
    }
}

/// Encode the blit that writes [`ENDED`] into the word after `words`' flag.
fn end_mark(command_buffer: &CommandBuffer, words: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(words, 4..8, 1).unwrap();
    blit.end_encoding().unwrap();
}

/// The busy work in flight ([`Busy::commit`]): its command buffers, in commit order on one queue.
pub(super) struct BusyWork {
    links: Vec<CommandBuffer>,
}

impl BusyWork {
    /// The last command buffer, which writes the end mark.
    fn last(&self) -> &CommandBuffer {
        self.links.last().expect("the busy work's command buffers")
    }

    /// Wait until every command buffer of the busy work completed.
    pub(super) fn wait_until_completed_for(&self, timeout: Duration) -> Result<(), TimedOut> {
        (self.links.iter()).try_for_each(|link| link.wait_until_completed_for(timeout))
    }

    /// The first error of its command buffers, in commit order.
    pub(super) fn error(&self) -> Option<CommandBufferError> {
        self.links.iter().find_map(CommandBuffer::error)
    }

    /// The status of its first command buffer not completed, else `Completed`.
    pub(super) fn status(&self) -> CommandBufferStatus {
        (self.links.iter())
            .map(CommandBuffer::status)
            .find(|status| *status != CommandBufferStatus::Completed)
            .unwrap_or(CommandBufferStatus::Completed)
    }

    /// Wait until every command buffer's batches are submitted: the last, committed last on the
    /// queue, is submitted after the others.
    pub(super) fn scheduled(&self) {
        scheduled(self.last());
    }

    /// When its first command buffer started on the GPU.
    fn gpu_start_time(&self) -> Result<f64, Error> {
        self.links[0].gpu_start_time()
    }

    /// When its last command buffer ended on the GPU.
    fn gpu_end_time(&self) -> Result<f64, Error> {
        self.last().gpu_end_time()
    }
}

/// The time `command_buffer`, completed in `wall` from its commit, took on the GPU: its GPU end
/// less its GPU start, where the device answers them (`CommandBuffer::gpu_start_time`) apart,
/// else `wall`, which holds it.
pub(super) fn gpu_took(command_buffer: &CommandBuffer, wall: Duration) -> Duration {
    match (
        command_buffer.gpu_start_time(),
        command_buffer.gpu_end_time(),
    ) {
        (Ok(start), Ok(end)) if end > start => Duration::from_secs_f64(end - start).min(wall),
        _ => wall,
    }
}

/// Raise the flag the busy work's spin waits for: it ends at its next load (or, where the device
/// does not show it the host's store while it runs, at its cap).
pub(super) fn release(words: &Buffer) {
    let contents = words.contents().expect("a Shared buffer");
    // SAFETY: the buffer's first word, `Shared` and 4-aligned, which the GPU only reads.
    let flag = unsafe { AtomicU32::from_ptr(contents.as_ptr().cast::<u32>()) };
    flag.store(1, Ordering::SeqCst);
}

/// The end mark: the word after the spin's flag, read atomically while the GPU may write it.
pub(super) fn read_mark(words: &Buffer) -> u32 {
    let contents = words.contents().expect("a Shared buffer");
    // SAFETY: the buffer's second word, `Shared` and 4-aligned, which the host only reads.
    let mark = unsafe { AtomicU32::from_ptr(contents.as_ptr().cast::<u32>().add(1)) };
    mark.load(Ordering::SeqCst)
}

/// Wait until `command_buffer`'s batches are all submitted.
pub(super) fn scheduled(command_buffer: &CommandBuffer) {
    let deadline = Instant::now() + WAIT;
    while matches!(
        command_buffer.status(),
        CommandBufferStatus::NotEnqueued
            | CommandBufferStatus::Enqueued
            | CommandBufferStatus::Committed
    ) {
        assert!(Instant::now() < deadline, "never scheduled");
        std::thread::sleep(Duration::from_micros(100));
    }
}

/// A `SIDE`² texture descriptor of `options`.
fn descriptor(options: ResourceOptions) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PF, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = options;
    d
}

/// `Private` storage with tracked hazards: a heap's resources are ordered by the runtime.
fn tracked_private() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Private,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    )
}

/// Bytes `(i * 7 + salt) % 251`.
fn pattern(salt: usize) -> Vec<u8> {
    (0..BYTES).map(|i| ((i * 7 + salt) % 251) as u8).collect()
}

/// A no-copy buffer over a fresh mapping of the texels' bytes holding `bytes`, which its
/// deallocator releases; `None`, printed, where the device imports no host memory.
fn over_pages(device: &Device, bytes: &[u8], seen: &Arc<Seen>) -> Option<Buffer> {
    let len = BYTES.next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    mapping.write(0, bytes);
    let pointer = mapping.pointer();
    // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            len,
            ResourceOptions::shared(),
            Some(recording(seen, mapping)),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            println!("no host memory on {}: {error:?}", device.name());
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// `texture`'s texels, copied by a blit into a new `Shared` buffer, in `command_buffer`.
fn copy_out(command_buffer: &CommandBuffer, texture: &Texture) -> Buffer {
    copy_out_sized(command_buffer, texture, SIDE, SIDE, ROW / SIDE)
}

/// `bytes` copied into `texture` by a blit, in `command_buffer`.
fn copy_in(command_buffer: &CommandBuffer, texture: &Texture, bytes: &[u8]) {
    let source = buffer_with(command_buffer.device(), bytes).unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        ROW,
        ROW * SIDE,
        Size::new(SIDE, SIDE, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

/// `buffer`'s first `len` bytes, copied by a blit into a new `Shared` buffer, in
/// `command_buffer`.
fn buffer_out(command_buffer: &CommandBuffer, buffer: &Buffer, len: u64) -> Buffer {
    let out = (command_buffer.device())
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, len)
        .unwrap();
    blit.end_encoding().unwrap();
    out
}

/// Whether a host pointer's `len` bytes are all zero.
fn zero_contents(buffer: &Buffer, len: u64) -> bool {
    read_buffer(buffer, 0, len).unwrap().iter().all(|&b| b == 0)
}

/// The calls a test times on `device`: what each made, and how long it took.
///
/// On lavapipe (`MESA_LLVMPIPE`) making an image view, a buffer view or a sampler, and compiling
/// a shader, take the lock its queue holds while it runs a submission's command buffers (Mesa
/// 26.0.8: `lvp_CreateImageView` and `lvp_CreateBufferView` in `lvp_image.c`, `lvp_CreateSampler`
/// and `lvp_queue_submit` in `lvp_device.c`, `lvp_shader_compile` in `lvp_pipeline.c`), so a call
/// that makes one waits there for the busy work whatever the runtime does. The first call that
/// returns only once the busy work ended, with no internal submission, is noted as that wait
/// (`Calls::queue_lock_wait`); the calls after it run with the busy work ended, and only the
/// test's count of internal submissions still checks them.
///
/// A call timed by [`time_announcing`](Calls::time_announcing) may return once the busy work
/// ended, with no internal submission, where it announced the kind of degrade notice it names:
/// the runtime announces such a wait where the driver is measured to make it (on ANV, the
/// device's first sparse texture, `DriverFact::FirstSparseResourceWaitsForQueues`). It is noted
/// (`Calls::announced_wait`) and must be a test's last timed call.
struct Calls<'a> {
    device: &'a Device,
    /// The busy work, and the buffer of its end mark.
    busy: &'a BusyWork,
    mark: Buffer,
    took: Vec<(String, Duration)>,
    /// Whether the device's driver is lavapipe, whose object making waits for its queue's work.
    queue_lock: bool,
    /// The call that waited for lavapipe's queue lock, and how long.
    queue_lock_wait: Option<(String, Duration)>,
    /// The call that waited for the busy work where it announced the wait, how long, and the
    /// kind it announced.
    announced_wait: Option<(String, Duration, Degradation)>,
    /// The degrade notices taken from the device's messages by [`Calls::time_announcing`].
    notices: Vec<DebugMessage>,
}

impl<'a> Calls<'a> {
    fn new(device: &'a Device, busy: &'a BusyWork, mark: Buffer) -> Self {
        let driver = device.properties().description.driver_id;
        Calls {
            device,
            busy,
            mark,
            took: Vec::new(),
            queue_lock: driver == Some(DriverId::MESA_LLVMPIPE),
            queue_lock_wait: None,
            announced_wait: None,
            notices: Vec::new(),
        }
    }

    /// Run `call`, check the busy work had not ended as it returned, and note its time. A call
    /// that waited names the internal submissions it made. On lavapipe a call that waited with
    /// no internal submission is the queue lock's wait (the type's docs).
    fn time<T>(&mut self, what: impl Into<String>, call: impl FnOnce() -> T) -> T {
        self.time_waiting(what.into(), None, call)
    }

    /// [`time`](Self::time) of a call that may wait for the busy work, with no internal
    /// submission, where it announces `kind` among `messages`' degrade notices (the type's docs).
    fn time_announcing<T>(
        &mut self,
        what: impl Into<String>,
        announcing: (&Messages, Degradation),
        call: impl FnOnce() -> T,
    ) -> T {
        self.time_waiting(what.into(), Some(announcing), call)
    }

    /// [`time`](Self::time), accepting the wait `announcing`'s kind announces where it is given.
    fn time_waiting<T>(
        &mut self,
        what: String,
        announcing: Option<(&Messages, Degradation)>,
        call: impl FnOnce() -> T,
    ) -> T {
        if let Some((messages, _)) = announcing {
            self.notices.extend(messages.take_notices());
        }
        let before = self.device.work_counts().internal_submissions;
        let (at, begun) = (Instant::now(), host_time());
        let made = call();
        let (took, returned) = (at.elapsed(), host_time());
        let internal = self.device.work_counts().internal_submissions - before;
        let announced = announcing.and_then(|(messages, kind)| {
            let notices = messages.take_notices();
            let announced = (notices.iter()).any(|message| {
                (message.degradation.as_ref()).is_some_and(|n| n.degradation == kind)
            });
            self.notices.extend(notices);
            announced.then_some(kind)
        });
        if let (Some(kind), 0, true) = (announced, internal, read_mark(&self.mark) != 0) {
            println!(
                "FIRST-USE on {}: {what} waited {took:?} for the busy work where it announced \
                 `{}`, with no internal submission",
                self.device.name(),
                kind.code()
            );
            self.announced_wait = Some((what.clone(), took, kind));
            self.took.push((what, took));
            return made;
        }
        if self.queue_lock && internal == 0 && read_mark(&self.mark) != 0 {
            if self.queue_lock_wait.is_none() {
                println!(
                    "FIRST-USE on {}: {what} waited {took:?} for lavapipe's queue lock, with no \
                     internal submission; the calls after it run with the busy work ended",
                    self.device.name()
                );
                self.queue_lock_wait = Some((what.clone(), took));
            }
            self.took.push((what, took));
            return made;
        }
        if read_mark(&self.mark) != 0 {
            panic!(
                "{what} returned only once the GPU's busy work ended ({took:?}, {internal} \
                 internal submissions) on {}: it waited for it, where Metal's returns at once; {}",
                self.device.name(),
                self.span(begun, returned)
            );
        }
        self.took.push((what, took));
        made
    }

    /// Where the busy work ran on the GPU against a call from host time `begun` to `returned`,
    /// once it completed, in milliseconds from the call's start: a failure's report.
    fn span(&self, begun: f64, returned: f64) -> String {
        self.busy.wait_until_completed_for(WAIT).unwrap();
        let ms = |at: f64| (at - begun) * 1e3;
        match (self.busy.gpu_start_time(), self.busy.gpu_end_time()) {
            (Ok(start), Ok(end)) => format!(
                "the busy work ran on the GPU from {:.1} ms to {:.1} ms ({:.1} ms), and the call \
                 returned at {:.1} ms, from the call's start",
                ms(start),
                ms(end),
                (end - start) * 1e3,
                ms(returned)
            ),
            (start, _) => format!("the busy work's GPU times: {start:?}"),
        }
    }

    /// How the calls returned against the busy work, for the test's summary line.
    fn waits(&self) -> String {
        match (&self.queue_lock_wait, &self.announced_wait) {
            (Some((what, took)), _) => format!("{what} waited {took:?} for lavapipe's queue lock"),
            (None, Some((what, took, kind))) => {
                format!(
                    "{what} waited {took:?} where it announced `{}`",
                    kind.code()
                )
            }
            (None, None) => {
                format!("every call returned with the busy work (at most {BUSY:?}) running")
            }
        }
    }

    /// Every degrade notice of the device's `messages` the test met, each kind announced once and
    /// exactly `kinds`: those [`time_announcing`](Self::time_announcing) took and those since;
    /// and each of `allowed`, the kinds the dispatches' pushed values decide
    /// (`degrade_notices::bound_value_kinds`), once or not.
    fn announced(
        &self,
        messages: &Messages,
        kinds: &[Degradation],
        allowed: &BTreeSet<Degradation>,
    ) -> BTreeMap<Degradation, usize> {
        let notices: Vec<DebugMessage> = (self.notices.iter().cloned())
            .chain(messages.take_notices())
            .collect();
        assert_once_allowing(&notices, kinds, allowed)
    }
}

/// Textures made down every path, buffers, a heap's resources and a rebind, each while the GPU's
/// busy work runs: each call returns before it ends, and every resource then holds
/// Metal's bytes (the module docs).
#[test]
fn new_resources_wait_for_no_gpu_work() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let busy = Busy::new(device, &queue);
        let seen = Arc::new(Seen::default());
        // Before the busy work: the bytes the textures over buffers lie over, a no-copy buffer to
        // detach, and one with a texture over it, used once, to rebind.
        let linear_bytes = pattern(1);
        let linear = buffer_with(device, &linear_bytes).unwrap();
        let pages_bytes = pattern(2);
        let pages = over_pages(device, &pages_bytes, &seen);
        let detached = over_pages(device, &pattern(3), &seen);
        let rebound = over_pages(device, &pattern(4), &seen).map(|buffer| {
            let d = descriptor(buffer.resource_options());
            let texture = buffer.new_texture(&d, 0, ROW).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let out = copy_out(&command_buffer, &texture);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(read_buffer(&out, 0, BYTES as u64).unwrap(), pattern(4));
            (buffer, texture)
        });
        if let Some(buffer) = &detached {
            assert!(buffer.detach_backing().unwrap(), "nothing holds its pages");
        }
        let rebind_bytes = pattern(5);
        let rebind_to = Mapping::anonymous(BYTES.next_multiple_of(GUEST_PAGE));
        rebind_to.write(0, &rebind_bytes);
        // A `Managed` buffer, written through `contents` once the busy work runs.
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let modified = device.new_buffer(BYTES as u64, managed).unwrap();
        let modified_bytes = pattern(9);

        let (busy_work, mark) = busy.commit(device, &queue);
        busy_work.scheduled();
        let mut calls = Calls::new(device, &busy_work, mark);
        assert_eq!(
            read_mark(&calls.mark),
            0,
            "the busy work ended before the calls"
        );
        let private = calls.time("new_texture (Private)", || {
            device
                .new_texture(&descriptor(ResourceOptions::private()))
                .unwrap()
        });
        let shared = calls.time("new_texture (Shared)", || {
            device
                .new_texture(&descriptor(ResourceOptions::shared()))
                .unwrap()
        });
        let over_buffer = calls.time("new_texture over a buffer", || {
            linear
                .new_texture(&descriptor(linear.resource_options()), 0, ROW)
                .unwrap()
        });
        let over_no_copy = pages.as_ref().map(|buffer| {
            calls.time("new_texture over a no-copy buffer", || {
                buffer
                    .new_texture(&descriptor(buffer.resource_options()), 0, ROW)
                    .unwrap()
            })
        });
        let heap = calls.time("new_heap (Private)", || {
            let d = HeapDescriptor::new(HeapType::Automatic, 1 << 20, tracked_private());
            device.new_heap(&d).unwrap()
        });
        let in_heap = calls.time("new_texture in a heap", || {
            heap.new_texture(&descriptor(ResourceOptions::private()))
                .unwrap()
        });
        let heap_buffer = calls.time("new_buffer in a heap", || {
            heap.new_buffer(256, ResourceOptions::private()).unwrap()
        });
        let stand_in = detached.as_ref().map(|buffer| {
            calls.time("new_texture over detached bytes", || {
                buffer
                    .new_texture(&descriptor(buffer.resource_options()), 0, ROW)
                    .unwrap()
            })
        });
        let private_buffer = calls.time("new_buffer (Private)", || {
            device
                .new_buffer(BYTES as u64, ResourceOptions::private())
                .unwrap()
        });
        let bytes_buffer_bytes = pattern(7);
        let bytes_buffer = calls.time("new_buffer_with_bytes (Private)", || {
            device
                .new_buffer_with_bytes(&bytes_buffer_bytes, ResourceOptions::private())
                .unwrap()
        });
        // On a device without unified memory each copies to the buffer's own bytes.
        let managed_bytes = pattern(8);
        let managed_buffer = calls.time("new_buffer_with_bytes (Managed)", || {
            device
                .new_buffer_with_bytes(&managed_bytes, managed)
                .unwrap()
        });
        let contents = modified.contents().expect("a Managed buffer's contents");
        // SAFETY: the buffer holds `BYTES` bytes at `contents`, which no command uses.
        unsafe { std::ptr::copy_nonoverlapping(modified_bytes.as_ptr(), contents.as_ptr(), BYTES) };
        calls.time("did_modify_range", || {
            modified.did_modify_range(0..BYTES as u64).unwrap()
        });
        let shared_buffer = calls.time("new_buffer (Shared)", || {
            device
                .new_buffer(BYTES as u64, ResourceOptions::shared())
                .unwrap()
        });
        assert!(
            zero_contents(&shared_buffer, BYTES as u64),
            "a new Shared buffer reads zero at once"
        );
        if let Some((buffer, _)) = &rebound {
            calls.time(
                "rebind_bytes_no_copy of a buffer with a texture over it",
                || rebind(buffer, rebind_to, &seen).unwrap(),
            );
        }
        let took: Vec<String> = (calls.took.iter())
            .map(|(what, took)| format!("{what} {took:?}"))
            .collect();
        println!(
            "FIRST-USE on {name}: {}: {}",
            calls.waits(),
            took.join(", ")
        );
        release(&calls.mark);
        busy_work.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(busy_work.error(), None);
        assert_eq!(read_mark(&calls.mark), ENDED, "the busy work's end mark");

        // What each holds, as Metal's does.
        let written = pattern(6);
        let command_buffer = queue.command_buffer().unwrap();
        copy_in(&command_buffer, &private, &written);
        copy_in(&command_buffer, &in_heap, &written);
        let outs = [
            (
                "the Private texture, written",
                copy_out(&command_buffer, &private),
                written.clone(),
            ),
            (
                "the Shared texture",
                copy_out(&command_buffer, &shared),
                vec![0; BYTES],
            ),
            (
                "the texture over a buffer",
                copy_out(&command_buffer, &over_buffer),
                linear_bytes,
            ),
            (
                "the heap's texture, written",
                copy_out(&command_buffer, &in_heap),
                written,
            ),
            (
                "the Private buffer",
                buffer_out(&command_buffer, &private_buffer, BYTES as u64),
                vec![0; BYTES],
            ),
            (
                "the heap's buffer",
                buffer_out(&command_buffer, &heap_buffer, 256),
                vec![0; 256],
            ),
            (
                "the Private buffer made with bytes",
                buffer_out(&command_buffer, &bytes_buffer, BYTES as u64),
                bytes_buffer_bytes,
            ),
            (
                "the Managed buffer made with bytes",
                buffer_out(&command_buffer, &managed_buffer, BYTES as u64),
                managed_bytes,
            ),
            (
                "the Managed buffer after did_modify_range",
                buffer_out(&command_buffer, &modified, BYTES as u64),
                modified_bytes,
            ),
        ];
        let no_copy = over_no_copy
            .as_ref()
            .map(|texture| copy_out(&command_buffer, texture));
        let rebound_out = (rebound.as_ref()).map(|(_, texture)| copy_out(&command_buffer, texture));
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {name}");
        for (what, out, expected) in outs {
            let len = expected.len() as u64;
            assert!(
                read_buffer(&out, 0, len).unwrap() == expected,
                "{what} on {name} reads other bytes than Metal's"
            );
        }
        if let Some(out) = no_copy {
            assert_eq!(
                read_buffer(&out, 0, BYTES as u64).unwrap(),
                pages_bytes,
                "the texture over a no-copy buffer on {name}"
            );
        }
        if let Some(out) = rebound_out {
            assert_eq!(
                read_buffer(&out, 0, BYTES as u64).unwrap(),
                rebind_bytes,
                "the texture over a rebound buffer on {name}"
            );
        }
        drop((
            stand_in,
            over_no_copy,
            private,
            shared,
            over_buffer,
            in_heap,
            heap,
        ));
        drop((rebound, detached, pages));
    });
}

/// The fills the long replay's command buffer encodes at first
/// ([`a_new_texture_waits_for_no_submission_s_replay`]), doubled until its submission comes at
/// least [`LONG_REPLAY`] after its commit, up to [`MOST_REPLAY_FILLS`].
const REPLAY_FILLS: usize = 1 << 12;
const MOST_REPLAY_FILLS: usize = 1 << 20;

/// The least time from the long replay's commit until its command buffer is scheduled.
const LONG_REPLAY: Duration = Duration::from_millis(40);

/// How long after that commit the texture is made: past the scheduler's first look at the
/// command buffer, well inside its replay.
const AFTER_COMMIT: Duration = Duration::from_millis(2);

/// Making a texture waits for no submission's replay of deferred command buffers, nor for its
/// `vkQueueSubmit` (Metal's makes wait for no GPU work, nor for a submission). One blit encoder of
/// fills of distinct words, whose replay into a `VkCommandBuffer` (`recorder`) and submission take
/// at least [`LONG_REPLAY`] (its fills doubled until they do), is committed, and a `Private`
/// texture, whose making registers its move and writes its slot (the module docs), is made
/// [`AFTER_COMMIT`] later, after one made untimed before: its registration waits for no
/// submission's place (`WorkCounts::registrations_waited`), and its time is printed. A device that
/// records as it encodes (no recorder) replays nothing, printed.
#[test]
fn a_new_texture_waits_for_no_submission_s_replay() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let first = device.new_texture(&descriptor(tracked_private())).unwrap();
        let mut fills = REPLAY_FILLS;
        loop {
            let target = (device.new_buffer(4 * fills as u64, ResourceOptions::private())).unwrap();
            let deferred = device.work_counts().command_buffers_deferred;
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            for at in 0..fills as u64 {
                blit.fill_buffer(&target, 4 * at..4 * at + 4, at as u8)
                    .unwrap();
            }
            blit.end_encoding().unwrap();
            if device.work_counts().command_buffers_deferred == deferred {
                eprintln!("FIRST-USE {name}: records as it encodes, so no submission replays");
                return;
            }
            let committed = Instant::now();
            command_buffer.commit().unwrap();
            std::thread::sleep(AFTER_COMMIT);
            let waited = device.work_counts().registrations_waited;
            let making = Instant::now();
            let texture = device.new_texture(&descriptor(tracked_private())).unwrap();
            let made = making.elapsed();
            let waited = device.work_counts().registrations_waited - waited;
            scheduled(&command_buffer);
            let submitted = committed.elapsed();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            drop(texture);
            eprintln!(
                "FIRST-USE {name}: a Private texture made {:.1} ms after the commit of {fills} \
                 fills took {:.3} ms, its registration waiting {waited} times for a \
                 submission's place; the command buffer was scheduled {:.1} ms after its commit",
                AFTER_COMMIT.as_secs_f64() * 1e3,
                made.as_secs_f64() * 1e3,
                submitted.as_secs_f64() * 1e3,
            );
            if submitted < LONG_REPLAY && fills < MOST_REPLAY_FILLS {
                fills *= 2;
                continue;
            }
            assert!(
                submitted >= LONG_REPLAY,
                "{MOST_REPLAY_FILLS} fills were scheduled {submitted:?} after their commit, \
                 short of {LONG_REPLAY:?}"
            );
            assert_eq!(
                waited, 0,
                "a texture made {AFTER_COMMIT:?} after the commit, {submitted:?} before its \
                 command buffer was scheduled, waited for a submission's place"
            );
            break;
        }
        drop(first);
    });
}

/// A `Private` buffer of `len` bytes made where a buffer that held `node(A)` was just released
/// (most likely its memory): a fill that did not run would leave those bytes. Whether the
/// memory was the released one's, printed.
fn over_dirty_memory(device: &Device, queue: &CommandQueue, len: u64) -> Buffer {
    let dirty = device.new_buffer(len, ResourceOptions::private()).unwrap();
    let source = buffer_with(device, &node(A)).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&source, 0, &dirty, 0, 16)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let address = dirty.gpu_address();
    drop((command_buffer, dirty));
    let made = device.new_buffer(len, ResourceOptions::private()).unwrap();
    println!(
        "FIRST-USE on {}: a new Private buffer {} the released one's memory",
        device.name(),
        if made.gpu_address() == address {
            "took"
        } else {
            "did not take"
        }
    );
    made
}

/// A command buffer on `queue` that does nothing of the resources, committed with `failure`
/// where given, and submitted: the first submission after them, which takes their first work.
fn first_after(device: &Device, queue: &CommandQueue, failure: Option<&str>) -> CommandBuffer {
    let unrelated = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&unrelated, 0..16, 7).unwrap();
    blit.end_encoding().unwrap();
    match failure {
        None => command_buffer.commit().unwrap(),
        Some("commit_failed") => command_buffer.commit_failed(CommitFailure::new(8)).unwrap(),
        Some(_) => {
            let check: SubmissionCheck = Arc::new(|_: &Submission<'_>| NonZeroU64::new(8));
            command_buffer.set_submission_check(check).unwrap();
            command_buffer.commit().unwrap();
        }
    }
    scheduled(&command_buffer);
    command_buffer
}

/// A new resource's zeros come before every use of it on every queue (the module docs): reads
/// through its address unnamed and through its heap made resident, on another queue than the
/// one whose submission took the fill, after a submission that took it failed, and writes
/// through resources placed in a new heap, aliased, which its fill never overwrites.
#[test]
fn a_new_resource_s_zeros_come_before_every_use_on_every_queue() {
    on_device(|device| {
        let name = device.name();
        let first = device.new_command_queue().unwrap();
        let second = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &second, &walker);
        let unnamed = |member: &Buffer, uses: Uses<'_>, what: &str| {
            let walked = walk(device, &second, &walker, member, uses, false, || {});
            assert_eq!(walked.command_buffer.error(), None, "{what} on {name}");
            assert!(
                walked.produced == outputs.zero,
                "{what} on {name} read other bytes than zero (the dirty node's: {})",
                walked.produced == outputs.a
            );
        };

        // Through its address, unnamed, on another queue than the fill's.
        let member = over_dirty_memory(device, &first, 16);
        let took = first_after(device, &first, None);
        unnamed(
            &member,
            Uses::Unnamed,
            "a new Private buffer read through its address",
        );
        took.wait_until_completed_for(WAIT).unwrap();

        // Through its heap, made resident, on another queue than the fill's.
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                tracked_private(),
            ))
            .unwrap();
        let placed = heap.new_buffer(16, ResourceOptions::private()).unwrap();
        let took = first_after(device, &first, None);
        unnamed(
            &placed,
            Uses::Heap(&heap),
            "a new heap's buffer read through use_heap",
        );
        took.wait_until_completed_for(WAIT).unwrap();

        // After the submission that carried the fill failed.
        for failure in ["commit_failed", "a failing submission check"] {
            let member = over_dirty_memory(device, &first, 16);
            let failed = first_after(device, &first, Some(failure));
            failed.wait_until_completed_for(WAIT).unwrap();
            assert!(failed.error().is_some(), "{failure} fails it on {name}");
            unnamed(&member, Uses::Unnamed, failure);
        }

        // Writes through resources placed in a new heap, one aliasing the other, in the first
        // command buffer after it, read through the other on the other queue: the heap's fill
        // came before them, and nothing after overwrote them; the rest reads the new heap's zero.
        let d = HeapDescriptor::new(HeapType::Placement, 1 << 20, tracked_private());
        let heap = device.new_heap(&d).unwrap();
        let buffer = (heap.new_buffer_at_offset(512, ResourceOptions::private(), 0)).unwrap();
        let alias = (heap.new_buffer_at_offset(256, ResourceOptions::private(), 0)).unwrap();
        let command_buffer = first.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&alias, 0..16, 0xa5).unwrap();
        blit.fill_buffer(&buffer, 256..512, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {name}");
        let command_buffer = second.command_buffer().unwrap();
        let out = buffer_out(&command_buffer, &buffer, 512);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "on {name}");
        let mut expected = vec![0; 512];
        expected[..16].fill(0xa5);
        expected[256..].fill(0x5a);
        assert_eq!(
            read_buffer(&out, 0, 512).unwrap(),
            expected,
            "bytes written through a new heap's placed buffers, one aliasing the other, on {name}"
        );
        drop((alias, buffer, heap));

        // `Shared` and `Managed` bytes read zero before any submission.
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        for options in [ResourceOptions::shared(), managed] {
            let made = device.new_buffer(BYTES as u64, options);
            match made {
                Ok(buffer) => assert!(
                    zero_contents(&buffer, BYTES as u64),
                    "a new {options:?} buffer's contents on {name}"
                ),
                Err(Error::Unsupported(_)) => println!("no {options:?} buffer on {name}"),
                Err(error) => panic!("new_buffer {options:?}: {error:?}"),
            }
        }
    });
}

/// The kind [`neutrals_and_sparse_textures_wait_for_no_gpu_work`] takes where the driver's first
/// sparse resource waits for every queue's work.
pub(super) const FIRST_SPARSE: [Degradation; 1] = [Degradation::FirstSparseResourceWait];

/// Kernels whose texture and texture-buffer slots nothing is bound to, encoded and committed (the
/// device's neutral images and texel buffers made as they are), and a sparse texture made, while
/// the GPU's busy work runs: each returns before it ends and makes no internal submission
/// (`WorkCounts::internal_submissions` unchanged). The kernels' encodings and commits allocate
/// no device memory (`WorkCounts::device_memory_allocated` unchanged): their pipeline states'
/// creation made the neutrals they bind and kept room in the memory the encodings place their own
/// resources in. Then the kernels' command buffers complete with a clean execution report, the
/// sparse texture, its first tile mapped, copies out, and the validation layer reports nothing. A
/// device without sparse textures prints its refusal.
///
/// Where the driver's first sparse resource waits for every queue's work
/// (`DriverFact::FirstSparseResourceWaitsForQueues`, ANV over i915's TR-TT), the sparse texture,
/// the device's first, may return once the busy work ended where its making announced
/// [`FIRST_SPARSE`], still with no internal submission; the notice is announced once, exactly where
/// the fact holds and the texture is a sparse image (not emulated).
#[test]
fn neutrals_and_sparse_textures_wait_for_no_gpu_work() {
    super::degrade_notices::on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let busy = Busy::new(device, &queue);
        let cases: Vec<_> = UNBOUND_CASES
            .iter()
            .map(|&case_name| {
                let case = public_case(case_name).unwrap();
                let pipeline = pipeline_state(device, &case).unwrap();
                let buffers: BTreeMap<u32, Buffer> = (case.compute.buffers.iter())
                    .map(|(&index, bytes)| (index, buffer_with(device, bytes).unwrap()))
                    .collect();
                (case_name, case, pipeline, buffers)
            })
            .collect();
        let sparse_heap =
            HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let sparse_heap = match device.new_heap(&sparse_heap) {
            Ok(heap) => Some(heap),
            Err(error) if lacks_sparse(&error) => {
                println!("FIRST-USE-SPARSE {name}: refused {error:?}");
                None
            }
            Err(error) => panic!("a sparse heap on {name}: {error:?}"),
        };
        let before = device.work_counts();

        let (busy_work, mark) = busy.commit(device, &queue);
        busy_work.scheduled();
        let mut calls = Calls::new(device, &busy_work, mark);
        assert_eq!(
            read_mark(&calls.mark),
            0,
            "the busy work ended before the calls"
        );
        let allocated = device.work_counts().device_memory_allocated;
        let unbound: Vec<(&str, CommandBuffer)> = (cases.iter())
            .map(|(case_name, case, pipeline, buffers)| {
                let command_buffer = calls.time(format!("{case_name}: its encoding"), || {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    let resources = CaseResources::default();
                    encode_case(&mut encoder, pipeline, &case.compute, buffers, &resources)
                        .unwrap();
                    encoder.end_encoding().unwrap();
                    command_buffer
                });
                calls.time(format!("{case_name}: its commit"), || {
                    command_buffer.commit().unwrap()
                });
                (*case_name, command_buffer)
            })
            .collect();
        let encoded = device.work_counts().device_memory_allocated;
        println!(
            "FIRST-USE-NEUTRALS-MEMORY on {name}: {} bytes allocated by the encodings and commits",
            encoded - allocated
        );
        assert_eq!(
            encoded, allocated,
            "the encodings and commits of kernels with unbound slots on {name} allocated device memory"
        );
        let sparse = sparse_heap.as_ref().map(|heap| {
            let announcing = (messages, FIRST_SPARSE[0]);
            calls.time_announcing("new_texture in a sparse heap", announcing, || {
                let mut d = TextureDescriptor::texture_2d(PF, 256, 256, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::private();
                heap.new_texture(&d).unwrap()
            })
        });
        let after = device.work_counts();
        let took: Vec<String> = (calls.took.iter())
            .map(|(what, took)| format!("{what} {took:?}"))
            .collect();
        println!(
            "FIRST-USE-NEUTRALS on {name}: {}, with no internal submission: {}",
            calls.waits(),
            took.join(", ")
        );
        assert_eq!(
            after.internal_submissions, before.internal_submissions,
            "a neutral's or a sparse texture's making on {name} made a submission"
        );
        // What the steps force, each from the device's facts: the pipeline states' makings, the
        // sparse texture's (`SparseFacts`), and the `Shared` buffers'.
        let sparse_facts = sparse.as_ref().map(|_| SparseFacts::of(device, PF));
        let mut kinds = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        )]);
        kinds.extend(&busy.made);
        // No kernel here uses a simdgroup operation (the busy kernel's ticket, spin and publish
        // run behind `if (lid == 0)`, which lockstep orders nothing for), so no dispatch runs
        // emulated simdgroups, a threadgroup of one thread included, and none announces it.
        // The dispatches bind nothing at the kernels' texture slots, so the values they push
        // there (a format class the function reads as data) decide the notes their stages keep by
        // such values, which a making does not announce: each is allowed.
        // A calibrating device may yet place one of these command buffers' GPU times outside its
        // life, which it announces at the read (`gpu_times::disagreeable_kinds`).
        let mut allowed = disagreeable_kinds(device);
        for (_, case, _, _) in &cases {
            kinds.extend(case_making_kinds(device, case));
            allowed.extend(case_bound_value_kinds(device, case));
        }
        if let Some(sparse_facts) = &sparse_facts {
            kinds.extend(sparse_facts.made(device));
        }
        let kinds: Vec<Degradation> = kinds.into_iter().collect();
        let announced = calls.announced(messages, &kinds, &allowed);
        let image = sparse_facts.is_some_and(|facts| facts.image);
        let facts = &device.properties().driver_facts;
        let waits = image && facts.contains(&DriverFact::FirstSparseResourceWaitsForQueues);
        for kind in FIRST_SPARSE {
            assert_eq!(
                announced.contains_key(&kind),
                waits,
                "{kind:?} on {name} exactly where its first sparse image waits: {announced:?}"
            );
        }
        release(&calls.mark);
        busy_work.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(busy_work.error(), None);
        assert_eq!(read_mark(&calls.mark), ENDED, "the busy work's end mark");
        for (case_name, command_buffer) in unbound {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{case_name} on {name}"
            );
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default()),
                "{case_name} on {name}"
            );
        }
        if let Some(texture) = &sparse {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
            encoder
                .update_texture_mapping(texture, TileMapping::Map, Region::new_2d(0, 0, 1, 1), 0, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            let out = copy_out(&command_buffer, texture);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "the sparse texture on {name}");
            let zero = read_buffer(&out, 0, BYTES as u64).unwrap() == vec![0; BYTES];
            println!("FIRST-USE-SPARSE {name}: its first tile mapped reads zero: {zero}");
        }
        drop((sparse, sparse_heap, cases));
    })
}

/// A `Shared` and a `Managed` texture's CPU copies while the GPU's busy work runs
/// (`texture/host_copy.rs`). Where the device copies on the host (its new image is counted
/// `WorkCounts::images_started_on_the_host`): the texture reads zero, and `get_bytes` and
/// `replace_region` return before the busy work ends and make no submission
/// (`WorkCounts::internal_submissions` unchanged), each counted a host copy. On every device: the
/// bytes written read back; a command buffer committed after `replace_region` reads them; a blit's
/// bytes read back through `get_bytes` once it completed. The making of a 4096 x 4096
/// `RGBA8Unorm` `Shared` texture, which writes its zeros on the host there, is timed
/// (`HOST-COPY` lines), and its last row reads zero.
#[test]
fn a_heap_texture_s_replace_region_waits_for_no_gpu_work() {
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let d = descriptor(ResourceOptions::shared());
        let size = device.heap_texture_size_and_align(&d).unwrap().size;
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                size,
                ResourceOptions::shared(),
            ))
            .unwrap();
        let texture = heap.new_texture_at_offset(&d, 0).unwrap();
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        let (first, second) = (pattern(8), pattern(9));
        let (busy_work, mark) = busy.commit(device, &queue);
        busy_work.scheduled();
        assert_eq!(read_mark(&mark), 0, "the busy work ended before the calls");
        let before = device.work_counts();
        let at = Instant::now();
        texture
            .replace_region(region, 0, 0, &first, ROW, 0)
            .unwrap();
        texture
            .replace_region(region, 0, 0, &second, ROW, 0)
            .unwrap();
        let (took, mark_then) = (at.elapsed(), read_mark(&mark));
        let after = device.work_counts();
        println!("STAGED-COPY two replace_region calls on {name}: {took:?} while the GPU was busy");
        assert_eq!(
            mark_then, 0,
            "replace_region of a heap texture on {name} waited for the busy work ({took:?})"
        );
        assert_eq!(
            after.internal_submissions, before.internal_submissions,
            "replace_region of a heap texture on {name} made a submission"
        );
        release(&mark);
        busy_work.wait_until_completed_for(WAIT).unwrap();
        // A command buffer committed after the calls reads the bytes of the later one.
        let command_buffer = queue.command_buffer().unwrap();
        let out = copy_out(&command_buffer, &texture);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            read_buffer(&out, 0, BYTES as u64).unwrap(),
            second,
            "a later command buffer on {name} reads the later call's bytes"
        );
        // And get_bytes reads a staged copy's bytes.
        texture
            .replace_region(region, 0, 0, &first, ROW, 0)
            .unwrap();
        let mut read = vec![0xee; BYTES];
        texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
        assert_eq!(read, first, "get_bytes on {name} reads the staged bytes");
        drop((texture, heap));
    });
}

/// A new heap texture's staged copy goes in the same `VkSubmitInfo` as its first move, and
/// lands: with no submission between the texture's making and `replace_region`, the copy is
/// staged (no host copy, no internal or staging submission), and the device's next submission
/// carries the move, in a command buffer first in its work's `VkSubmitInfo`, and the copy after
/// it, at the work's start (`texture/first_move.rs`). MoltenVK runs each `VkSubmitInfo` as a
/// Metal command buffer of its own, so the move, the copy and the work run in one, as the
/// texture's first barrier names no host read (`image_barrier.rs`). Down both submissions that
/// take first work: a committed command buffer's blit (`command_buffer/submit.rs`) and
/// `get_bytes`'s internal submission (`device/submit.rs`) each read the copy's bytes, in one
/// `vkQueueSubmit` of one `VkSubmitInfo` (the internal one followed by the relay's call of no
/// work where the runtime relays the driver's failures, `commit_work::relay_marks`: after the
/// runtime's own submission a fence alone, of no `VkSubmitInfo`). Then an empty command buffer committed after a new texture is made carries its
/// first move in its batch's one `VkSubmitInfo`.
#[test]
fn a_new_heap_texture_s_staged_copy_lands_in_its_first_move_s_submission() {
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let d = descriptor(ResourceOptions::shared());
        let size = device.heap_texture_size_and_align(&d).unwrap().size;
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        let marks = relay_marks(device, &queue);
        for (salt, by_get_bytes) in [(10, false), (11, true)] {
            let options = ResourceOptions::shared();
            let heap = (device.new_heap(&HeapDescriptor::new(HeapType::Placement, size, options)))
                .unwrap();
            let before = device.work_counts();
            let texture = heap.new_texture_at_offset(&d, 0).unwrap();
            let written = pattern(salt);
            texture
                .replace_region(region, 0, 0, &written, ROW, 0)
                .unwrap();
            let staged = device.work_counts();
            let path = if by_get_bytes { "get_bytes" } else { "a blit" };
            assert_eq!(
                (
                    staged.queue_submissions,
                    staged.host_texel_copies,
                    staged.staging_submissions
                ),
                (
                    before.queue_submissions,
                    before.host_texel_copies,
                    before.staging_submissions
                ),
                "{path} on {name}: the texture's making or replace_region submitted or copied"
            );
            let read = if by_get_bytes {
                let mut read = vec![0xee; BYTES];
                texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
                read
            } else {
                let command_buffer = queue.command_buffer().unwrap();
                let out = copy_out(&command_buffer, &texture);
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{path} on {name}");
                read_buffer(&out, 0, BYTES as u64).unwrap()
            };
            let after = device.work_counts();
            let wrong = read.iter().zip(&written).filter(|(r, w)| r != w).count();
            assert_eq!(
                wrong,
                0,
                "{path} on {name}: bytes other than the copy's (the first texel read {:?}, \
                 written {:?})",
                &read[..4],
                &written[..4]
            );
            assert_eq!(
                after.host_texel_copies, before.host_texel_copies,
                "{path} on {name}: a host copy"
            );
            assert_eq!(
                after.internal_submissions - before.internal_submissions,
                u64::from(by_get_bytes),
                "{path} on {name}: internal submissions"
            );
            let calls = 1 + if by_get_bytes { marks } else { 0 };
            assert_eq!(
                (
                    after.queue_submissions - before.queue_submissions,
                    after.submit_infos - before.submit_infos,
                ),
                (calls, 1),
                "{path} on {name}: the move, the copy and the work in one VkSubmitInfo"
            );
            drop((texture, heap));
        }
        // A batch with no work: its first move rides in its one `VkSubmitInfo`, its end's.
        let options = ResourceOptions::shared();
        let heap =
            (device.new_heap(&HeapDescriptor::new(HeapType::Placement, size, options))).unwrap();
        let before = device.work_counts();
        let texture = heap.new_texture_at_offset(&d, 0).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.error(),
            None,
            "an empty command buffer on {name}"
        );
        let after = device.work_counts();
        assert_eq!(
            (
                after.command_buffers_begun_batch_work - before.command_buffers_begun_batch_work,
                after.queue_submissions - before.queue_submissions,
                after.submit_infos - before.submit_infos,
            ),
            (1, 1, 1),
            "an empty command buffer on {name}: its first work, in its one VkSubmitInfo"
        );
        drop((texture, heap));
    });
}

#[test]
fn shared_textures_copy_on_the_host_while_the_gpu_is_busy() {
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        for options in [ResourceOptions::shared(), managed] {
            let mode = options.storage_mode();
            let before = device.work_counts();
            let texture = device.new_texture(&descriptor(options)).unwrap();
            let made = device.work_counts();
            let hosted = made.images_started_on_the_host > before.images_started_on_the_host;
            let staged = made.staging_submissions > before.staging_submissions;
            println!(
                "HOST-COPY {mode:?} texture on {name}: {}",
                match (hosted, staged) {
                    (true, _) => "moved and zeroed on the host",
                    (false, true) => "moved and zeroed on the staging queue",
                    (false, false) => "copied through submissions",
                }
            );
            let (busy_work, mark) = busy.commit(device, &queue);
            busy_work.scheduled();
            assert_eq!(read_mark(&mark), 0, "the busy work ended before the calls");
            let mut read = vec![0xee; BYTES];
            let at = Instant::now();
            texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
            let (read_took, read_mark_then) = (at.elapsed(), read_mark(&mark));
            assert!(
                read.iter().all(|&b| b == 0),
                "{mode:?}: a new texture reads zero"
            );
            let written = pattern(6);
            let at = Instant::now();
            texture
                .replace_region(region, 0, 0, &written, ROW, 0)
                .unwrap();
            let (write_took, write_mark_then) = (at.elapsed(), read_mark(&mark));
            texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
            assert_eq!(read, written, "{mode:?}: the bytes written read back");
            let copied = device.work_counts();
            println!(
                "HOST-COPY {mode:?} on {name}: get_bytes {read_took:?}, replace_region \
                 {write_took:?} while the GPU was busy"
            );
            if hosted {
                assert_eq!(
                    (read_mark_then, write_mark_then),
                    (0, 0),
                    "{mode:?} on {name}: a host copy waited for the busy work (get_bytes \
                     {read_took:?}, replace_region {write_took:?})"
                );
                assert_eq!(
                    copied.internal_submissions, before.internal_submissions,
                    "{mode:?} on {name}: a host copy made a submission"
                );
                assert_eq!(copied.host_texel_copies - before.host_texel_copies, 3);
            }
            // On the staging queue each copy waits for its own submission alone.
            if staged {
                assert_eq!(
                    (read_mark_then, write_mark_then),
                    (0, 0),
                    "{mode:?} on {name}: a staging copy waited for the busy work (get_bytes \
                     {read_took:?}, replace_region {write_took:?})"
                );
                assert_eq!(
                    copied.internal_submissions, before.internal_submissions,
                    "{mode:?} on {name}: a staging copy made an internal submission"
                );
                assert_eq!(copied.staging_submissions - made.staging_submissions, 3);
            }
            release(&mark);
            busy_work.wait_until_completed_for(WAIT).unwrap();
            // A command buffer committed after the call reads the bytes it wrote.
            let command_buffer = queue.command_buffer().unwrap();
            let out = copy_out(&command_buffer, &texture);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                read_buffer(&out, 0, BYTES as u64).unwrap(),
                written,
                "{mode:?}: a later command buffer reads the host's bytes"
            );
            // And the host reads what a completed command buffer wrote.
            let blitted = pattern(7);
            let command_buffer = queue.command_buffer().unwrap();
            copy_in(&command_buffer, &texture, &blitted);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
            assert_eq!(read, blitted, "{mode:?}: get_bytes reads the blit's bytes");
        }
        // The cost of a large texture's zeros, written by the making.
        let side = 4096;
        let mut large = TextureDescriptor::texture_2d(PF, side, side, false);
        large.usage = TextureUsage::SHADER_READ;
        large.resource_options = ResourceOptions::shared();
        let before = device.work_counts();
        let at = Instant::now();
        let texture = device.new_texture(&large).unwrap();
        let took = at.elapsed();
        let made = device.work_counts();
        let hosted = made.images_started_on_the_host > before.images_started_on_the_host;
        let staged = made.staging_submissions > before.staging_submissions;
        println!(
            "HOST-COPY a {side}x{side} RGBA8Unorm Shared texture made in {took:?} on {name} ({})",
            match (hosted, staged) {
                (true, _) => "its zeros written on the host",
                (false, true) => "its zeros written on the staging queue",
                (false, false) => "its zeros left to the next submission",
            }
        );
        let mut row = vec![0xee; (4 * side) as usize];
        texture
            .get_bytes(
                &mut row,
                4 * side,
                0,
                Region::new_2d(0, side - 1, side, 1),
                0,
                0,
            )
            .unwrap();
        assert!(row.iter().all(|&b| b == 0), "the large texture's last row");
    });
}

/// A display-sized `Shared` and `Managed` `BGRA8Unorm` texture a render pass may draw into, as a
/// frame target is: `get_bytes` reads what a completed blit wrote, texel for texel, each of
/// `FRAME_READS` times, and a command buffer committed after `replace_region` reads the bytes it
/// wrote. Prints each call's time (`FRAME-COPY` lines). With about 2 s of GPU work running, it
/// reads `replace_region`'s texels up to `LOADED_READS` times, until the work ends; where it copies
/// on the host or the staging queue, the first read returns while the work runs, and the line gives
/// the distribution of those that did.
#[test]
fn frame_sized_textures_copy_texel_exact() {
    const WIDTH: u64 = 1920;
    const HEIGHT: u64 = 1080;
    const FRAME_ROW: u64 = 4 * WIDTH;
    const FRAME_BYTES: usize = (FRAME_ROW * HEIGHT) as usize;
    const FRAME_READS: usize = 5;
    const LOADED_READS: usize = 40;
    let frame_pattern = |salt: usize| -> Vec<u8> {
        (0..FRAME_BYTES)
            .map(|i| ((i * 7 + salt) % 251) as u8)
            .collect()
    };
    let millis = |took: Duration| took.as_secs_f64() * 1000.0;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let busy = Busy::new(device, &queue);
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
        let size = Size::new(WIDTH, HEIGHT, 1);
        for options in [ResourceOptions::shared(), managed] {
            let mode = options.storage_mode();
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
            descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            descriptor.resource_options = options;
            let before = device.work_counts();
            let texture = device.new_texture(&descriptor).unwrap();
            let made = device.work_counts();
            let hosted = made.images_started_on_the_host > before.images_started_on_the_host;
            let staged = made.staging_submissions > before.staging_submissions;
            let cached = takes_host_cached_memory(device);
            if has_a_copy_engine(device) && !cached {
                assert!(
                    staged,
                    "{mode:?} on {name}: a device with a copy engine and no memory the host caches \
                     for a texture copies on its staging queue"
                );
            }
            let uncached = made.uncached_host_images > before.uncached_host_images;
            assert!(
                !(hosted && cached && uncached),
                "{mode:?} on {name}: a texture that copies on the host lies in memory the host \
                 caches where the device has it"
            );
            // The GPU writes it, and the host reads that once the command buffer completed.
            let drawn = frame_pattern(11);
            let source = buffer_with(device, &drawn).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                FRAME_ROW,
                FRAME_ROW * HEIGHT,
                size,
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let mut reads = Vec::new();
            for _ in 0..FRAME_READS {
                let mut read = vec![0xee; FRAME_BYTES];
                let at = Instant::now();
                texture
                    .get_bytes(&mut read, FRAME_ROW, 0, region, 0, 0)
                    .unwrap();
                reads.push(millis(at.elapsed()));
                assert!(
                    read == drawn,
                    "{mode:?} on {name}: get_bytes reads the completed blit's texels"
                );
            }
            // The host writes it, and a command buffer committed after reads that.
            let written = frame_pattern(12);
            let at = Instant::now();
            texture
                .replace_region(region, 0, 0, &written, FRAME_ROW, 0)
                .unwrap();
            let write_took = millis(at.elapsed());
            let out = device
                .new_buffer(FRAME_BYTES as u64, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &out,
                0,
                FRAME_ROW,
                FRAME_ROW * HEIGHT,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert!(
                read_buffer(&out, 0, FRAME_BYTES as u64).unwrap() == written,
                "{mode:?} on {name}: a later command buffer reads replace_region's texels"
            );
            // With about 2 s of GPU work running: on the host or the staging queue, `get_bytes`
            // returns while it runs (an absolute time is not portable), still texel exact. The
            // reads go on until it ends, at most `LOADED_READS`; the times of those that
            // returned while it ran are their distribution under load.
            let (busy_work, mark) = busy.commit(device, &queue);
            busy_work.scheduled();
            let mut loaded = Vec::new();
            let mut ended_after = None;
            for _ in 0..LOADED_READS {
                let mut read = vec![0xee; FRAME_BYTES];
                let at = Instant::now();
                texture
                    .get_bytes(&mut read, FRAME_ROW, 0, region, 0, 0)
                    .unwrap();
                let took = millis(at.elapsed());
                let running = read_mark(&mark) == 0;
                assert!(
                    read == written,
                    "{mode:?} on {name}: get_bytes under load reads replace_region's texels"
                );
                if !running {
                    ended_after = Some(took);
                    break;
                }
                loaded.push(took);
            }
            release(&mark);
            busy_work.wait_until_completed_for(WAIT).unwrap();
            if hosted || staged {
                assert!(
                    !loaded.is_empty(),
                    "{mode:?} on {name}: get_bytes waited for the busy work ({ended_after:.2?} ms)"
                );
            }
            loaded.sort_by(f64::total_cmp);
            let under_load = match (loaded.first(), loaded.last()) {
                (Some(best), Some(worst)) => format!(
                    "{} reads, best {best:.2} ms, median {:.2} ms, worst {worst:.2} ms",
                    loaded.len(),
                    loaded[loaded.len() / 2]
                ),
                _ => format!("none returned while it ran (the first after {ended_after:.2?} ms)"),
            };
            let best = reads.iter().copied().fold(f64::MAX, f64::min);
            println!(
                "FRAME-COPY {mode:?} {WIDTH}x{HEIGHT} BGRA8Unorm on {name} ({}): get_bytes best \
                 {best:.2} ms of {reads:.2?}, replace_region {write_took:.2} ms; with 2 s of GPU \
                 work running: {under_load}; texel-exact",
                match (hosted, staged) {
                    (true, _) if uncached => "copied on the host from uncached memory",
                    (true, _) => "copied on the host",
                    (false, true) => "copied on the staging queue",
                    (false, false) => "copied through submissions",
                },
            );
        }
    });
}

/// Whether `device` lists a copy engine the runtime stages on, as Vulkan itself reports the
/// physical device, so a runtime that took no staging queue is caught: a family of transfers
/// alone (sparse binding and protected memory aside) whose copies take any texel.
fn has_a_copy_engine(device: &Device) -> bool {
    use ash::vk;
    // SAFETY: the handles are used only here, while the device lives, to query the physical
    // device.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let families = unsafe {
        raw.instance
            .get_physical_device_queue_family_properties(raw.physical_device)
    };
    let unit = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };
    let besides =
        vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING | vk::QueueFlags::PROTECTED;
    families.iter().any(|family| {
        family.queue_count > 0
            && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && (family.queue_flags & !besides).is_empty()
            && family.min_image_transfer_granularity == unit
    })
}

/// Whether `device` has memory the host caches that a texture's host-transfer image takes, as
/// Vulkan itself reports the physical device: a host-visible, coherent, host-cached type the
/// runtime allocates, device-local, or any on a device with unified memory (a Radeon 780M's
/// system memory). A frame-sized `BGRA8Unorm` image may lie in it on every
/// device measured.
fn takes_host_cached_memory(device: &Device) -> bool {
    use ash::vk;
    // SAFETY: as in `has_a_copy_engine`.
    let raw = unsafe { device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    use vk::MemoryPropertyFlags as P;
    let cached = P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED;
    let unusable = P::DEVICE_COHERENT_AMD | P::DEVICE_UNCACHED_AMD | P::LAZILY_ALLOCATED;
    let unified = device.properties().has_unified_memory;
    (memory.memory_types[..memory.memory_type_count as usize].iter()).any(|ty| {
        let flags = ty.property_flags;
        flags.contains(cached)
            && !flags.intersects(unusable)
            && (unified || flags.contains(P::DEVICE_LOCAL))
    })
}

/// `get_bytes` and `replace_region` of a `Shared` and a `Managed` texture return while a command
/// buffer on each of four queues (every `VkQueue` of the round robin) waits on an event the
/// calling thread signals only after the calls (the consumer's guest hang: its FIFO thread called
/// `get_bytes` before it signalled what its queued work waited on), where they copy on the host
/// or on the staging queue: Metal's copies wait for no GPU work. The calls run on another thread,
/// bounded: one that waits fails the test once the event is signalled, so nothing stays blocked.
/// Prints the route (`STAGING-HANG` lines).
#[test]
fn cpu_copies_return_while_queued_work_waits_on_a_later_event() {
    on_device(|device| {
        let name = device.name();
        let managed = ResourceOptions::new(
            StorageMode::Managed,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        let before = device.work_counts();
        let textures: Vec<Texture> = [ResourceOptions::shared(), managed]
            .into_iter()
            .map(|options| device.new_texture(&descriptor(options)).unwrap())
            .collect();
        let made = device.work_counts();
        let hosted = made.images_started_on_the_host - before.images_started_on_the_host == 2;
        let staged = made.staging_submissions - before.staging_submissions == 2;
        // The queued work reads both textures once the event is signalled.
        let queues: Vec<CommandQueue> = (0..4)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        let event = device.new_shared_event().unwrap();
        let waiting: Vec<(CommandBuffer, Vec<Buffer>)> = queues
            .iter()
            .map(|queue| {
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .encode_wait_for_event((&event).into(), 1)
                    .unwrap();
                let outs = (textures.iter())
                    .map(|texture| copy_out(&command_buffer, texture))
                    .collect();
                command_buffer.commit().unwrap();
                (command_buffer, outs)
            })
            .collect();
        let written = pattern(9);
        let (done, finished) = std::sync::mpsc::channel();
        let returned = std::thread::scope(|scope| {
            scope.spawn(|| {
                for texture in &textures {
                    let mut read = vec![0xee; BYTES];
                    texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
                    assert!(read.iter().all(|&b| b == 0), "a new texture reads zero");
                    texture
                        .replace_region(region, 0, 0, &written, ROW, 0)
                        .unwrap();
                    texture.get_bytes(&mut read, ROW, 0, region, 0, 0).unwrap();
                    assert_eq!(read, written, "replace_region's bytes read back");
                }
                done.send(()).unwrap();
            });
            let returned = finished.recv_timeout(Duration::from_secs(20)).is_ok();
            event.set_signaled_value(1);
            returned
        });
        for (command_buffer, outs) in &waiting {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
            for out in outs {
                assert_eq!(
                    read_buffer(out, 0, BYTES as u64).unwrap(),
                    written,
                    "{name}: the queued work reads replace_region's bytes"
                );
            }
        }
        let route = match (hosted, staged) {
            (true, _) => "on the host",
            (false, true) => "on the staging queue",
            (false, false) => "through submissions",
        };
        println!(
            "STAGING-HANG on {name}: copies {route} with four queues waiting on an event: {}",
            match returned {
                true => "returned",
                false => "waited for the event",
            }
        );
        if hosted || staged {
            assert!(
                returned,
                "{name}: a copy {route} waited for queued work waiting on a later event"
            );
        }
    });
}

/// A new `Shared` 4x texture, as Metal makes one, of `BGRA8Unorm` and `RGBA16Float`, a render
/// target with and without `ShaderRead`: made under the validation layer without a message, started
/// neither on the host nor on the staging queue (Vulkan copies nothing into a multisampled image),
/// and resolved to zero by a pass that loads it and draws nothing (a new `Shared` texture reads
/// zero).
#[test]
fn multisampled_shared_textures_start_by_a_clear() {
    const W: u64 = 1024;
    const H: u64 = 64;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        for (format, bpt) in [(PixelFormat::BGRA8Unorm, 4), (PixelFormat::RGBA16Float, 8)] {
            for usage in [
                TextureUsage::RENDER_TARGET,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ] {
                let mut d = TextureDescriptor::texture_2d(format, W, H, false);
                d.texture_type = TextureType::Type2DMultisample;
                d.sample_count = 4;
                d.usage = usage;
                d.resource_options = ResourceOptions::shared();
                let before = device.work_counts();
                let texture = device.new_texture(&d).unwrap_or_else(|error| {
                    panic!("{name}: a 4x {format:?} {usage:?} texture: {error:?}")
                });
                let made = device.work_counts();
                assert_eq!(
                    (made.images_started_on_the_host, made.staging_submissions),
                    (
                        before.images_started_on_the_host,
                        before.staging_submissions
                    ),
                    "{name}: a 4x {format:?} texture started by a copy"
                );
                let mut r = TextureDescriptor::texture_2d(format, W, H, false);
                r.usage = TextureUsage::RENDER_TARGET;
                let resolved = device.new_texture(&r).unwrap();
                let mut pass = RenderPassDescriptor::new();
                let target = &mut pass.color_attachments[0];
                target.texture = Some(texture.clone());
                target.load_action = LoadAction::Load;
                target.store_action = StoreAction::MultisampleResolve;
                target.resolve_texture = Some(resolved.clone());
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                let out = copy_out_sized(&command_buffer, &resolved, W, H, bpt);
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
                assert!(
                    zero_contents(&out, W * H * bpt),
                    "{name}: a new 4x {format:?} texture resolves to zero"
                );
                println!("MS-START {name}: a 4x {format:?} {usage:?} texture starts by a clear");
            }
        }
    });
}

/// `texture`'s `w` x `h` texels of `bpt` bytes, copied by a blit into a new `Shared` buffer, in
/// `command_buffer`.
fn copy_out_sized(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    w: u64,
    h: u64,
    bpt: u64,
) -> Buffer {
    let device = command_buffer.device();
    let out = device
        .new_buffer(w * h * bpt, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &out,
        0,
        w * bpt,
        w * h * bpt,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    out
}
