//! The device's check of whether its GPU idles when consecutive work comes from another `VkQueue`
//! (`queue_share.rs`), on a thread of its own, once per driver, release and runtime.
//!
//! - **What it measures.** Two `VkQueue`s of the device's primary family that the round robin
//!   lends ([`pair`]), and batches of one command buffer each that write a timestamp at the top of
//!   the pipe and one at the bottom, nothing between. Each round submits two such batches, both
//!   waiting for one value of a gate timeline, so the GPU holds both before it runs either: on the
//!   two `VkQueue`s (the alternation), or both on one (its control). The gap between the end of
//!   the batch the GPU ran first and the start of the other is a switch's where the two were
//!   apart, a submission's where they were not, and the switch cost is the median gap of the first
//!   less the median of the second. The rounds alternate between the two, [`ROUNDS`] each, so work
//!   of the program's own meanwhile reaches both alike. On the RTX 5080 Laptop GPU (NVIDIA
//!   615.71.09) it measured 49.0 to 51.3 us in six runs, where the GPU idles 56 to 62 us between
//!   consecutive command buffers of different queues, and a chain of batches each waiting on a
//!   timeline for the one before it measured 53.2 to 59.3 us.
//! - **The gate opens on the GPU.** A round's first submission, on its first `VkQueue`, is its
//!   opening: fills of a scratch buffer of the check's ([`DELAY_BYTES`] each) that signal the gate
//!   as they complete, made before the two batches that wait for it. So no batch of the check ever
//!   waits for a signal the host has yet to make: every wait is for a signal already submitted,
//!   and the device's work in flight completes with no further call of the check's thread. A
//!   driver call on another thread that waits for the GPU's work in flight, holding the driver's
//!   lock that the check's next call takes, therefore waits for work that completes. Seen
//!   otherwise, when the host signalled the gate once both batches were submitted: on an RTX 5080
//!   Laptop GPU (NVIDIA 615.71.09) beside the runtime suite, another device's `vkDestroyDevice`
//!   never returned while the check sat in the `vkQueueSubmit` of its round's second batch, the
//!   first waiting on the GPU for that signal (`tests::queue_switch_gate`). A round whose gate
//!   had opened as its last batch's submission returned (the host slower than the fills) may have
//!   run its first batch before the second was submitted, so its gap counts for nothing, and the
//!   later rounds' openings repeat the fill twice as many times, up to [`MOST_REPEATS`].
//! - **Which timestamps.** Timestamps of two `VkQueue`s compare only where the device calibrates
//!   them: "An implementation supporting `VK_KHR_calibrated_timestamps` or
//!   `VK_EXT_calibrated_timestamps` will use the same time domain for all its `VkQueue` so that
//!   timestamp values reported for `VK_TIME_DOMAIN_DEVICE_KHR` can be matched to any timestamp
//!   captured through `vkCmdWriteTimestamp`" (the Vulkan specification, Time Domains). A device
//!   without calibration, or whose primary family writes no timestamps, measures nothing: its
//!   verdict is `Unchecked`.
//! - **Its verdict**: `Idles` where the switch cost is above `queue_share::SWITCH_IDLES` (20 us),
//!   else `DoesNotIdle`; `Unchecked` where a run failed (a Vulkan call refused, or too few rounds
//!   whose four timestamps were written). Until it answers the device shares no `VkQueue`.
//! - **Its cost**: [`ROUNDS`] times two rounds of two batches, each with no work but its two
//!   timestamps: about 2 ms of the GPU's time where the switch costs 60 us; each round's opening,
//!   fills of [`DELAY_BYTES`] repeated until they outlast the host's two submissions; a scratch
//!   buffer of that size in device-local memory while the check runs; and the host's waits for
//!   each round.
//! - **Kept.** A measured verdict is written as one record in the shader cache's content directory
//!   (`shader_cache/store.rs`), named by the driver's identity, the check's version and the
//!   runtime's (`queue_share::record_name`): a device of the same driver and release reads it as
//!   it is made and runs no check. A failed run is kept nowhere.
//! - **Out of the device's way.** Before each round the thread steps aside for the device's
//!   allocations and frees on other threads (`step_aside.rs`): it goes on once none is in
//!   progress and none ended for `step_aside::QUIET`, at once while a call waits for its verdict
//!   (`Device::settle_queue_switch_check`). Its submissions are counted as the checks'
//!   (`WorkCounts::device_checks`), not among the device's (`queue_submissions`).
//! - **Its thread.** Each reach of the device runs inside a section of the exit guard
//!   (`exit_guard.rs`), as every runtime thread's does: the objects' making, the submissions, each
//!   wait of at most [`WAIT_SLICE`], and the releases. Once the process is exiting it ends,
//!   unrecorded, and leaves its objects to the process's end. A refused opening is the round's
//!   last submission, so no batch submitted waits for good.
//! - **Its queues held through its submissions.** A round takes the lock of each of its
//!   `VkQueue`s, in index order, before its opening, and lets them go once its last batch was
//!   submitted (`QueueUses::use_queues_kept`), so its three submissions go in their order with no
//!   other thread's between them. Every batch there completes with no more work of the host's, so
//!   a wait for a queue to idle under its lock (`Device::drain_queues`, after a failed wait; a
//!   submission check's after a refused call) waits for work that completes.
//! - **Holding nothing of the device.** The check holds only the parts of the device it works
//!   with ([`Parts`]), never a handle: a check that held one for its run would hold the device's
//!   last once the program dropped its own while the check ran (the first moments of a device
//!   that makes a command queue and goes), and the device would go on the check's thread after
//!   the program's last drop returned, where a consumer that adopted its `VkDevice` destroys it
//!   then ("Host access to `device` must be externally synchronized", and every object made from
//!   it is destroyed before it: VUID-vkDestroyDevice-device-05137). So the program's last drop
//!   tears the device down before it returns. The teardown first stops the check
//!   (`QueueSwitch::stop`): the check reads that between its steps, before its objects' making,
//!   before each round and before it reads the timestamps, so the teardown waits for at most one
//!   step (a round's two batches, which complete as soon as the GPU reaches them, or the making),
//!   then for the check to let its objects go, unrecorded and its verdict `Unchecked`, before the
//!   check's thread drops its shares of the device's parts and ends.

use crate::allocations::Allocations;
use crate::command_buffer::fences::Fences;
use crate::device::functions::{Functions, Vk};
use crate::device::relay::{By, Mark, QueueUses};
use crate::device::submit_stalls::Uses;
use crate::device::{lock, Device, QueueSlot};
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::freed_memory::Parking;
use crate::in_place::DriverIdentity;
use crate::loader_word::{DeviceWord, Held};
use crate::queue_share::{encode, record_name, QueueSwitch, Verdict, SWITCH_IDLES};
use crate::release;
use crate::shader_cache::store::Store;
use crate::work_counts::Counters;
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The rounds of each kind, the two `VkQueue`s' and the one's.
pub(crate) const ROUNDS: usize = 24;

/// How long one wait for a round's batches lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The bytes of the check's scratch buffer, which each fill of a round's opening writes whole
/// (the module docs).
const DELAY_BYTES: u64 = 4 << 20;

/// The most times a round's opening repeats its fill (the module docs).
const MOST_REPEATS: u32 = 1 << 10;

/// The fewest rounds of each kind whose timestamps were all written that a verdict needs.
const FEWEST: usize = ROUNDS / 2;

/// The two `VkQueue`s the check measures, by index among the device's: the first two the round
/// robin lends of the primary family; `None` where it lends fewer.
pub(crate) fn pair(device: &Device) -> Option<(usize, usize)> {
    let shared = device.shared();
    let mut same = (0..shared.round_robin())
        .filter(|&index| shared.queues[index].family == shared.queue_family);
    Some((same.next()?, same.next()?))
}

/// Whether the device's command buffers write timestamps the host reads, and it calibrates them:
/// the check compares two `VkQueue`s' timestamps, and a queue is known long only by its command
/// buffers' own (`queue_share.rs`). Not where it answers GPU times from the host
/// (`gpu_times_from_host`, or a driver whose timestamps wait for every queue's).
pub(crate) fn timed(device: &Device) -> bool {
    let timing = &device.shared().timing;
    !timing.answers_from_host()
        && timing
            .clock()
            .is_some_and(|clock| clock.calibration.is_some())
}

/// Start the check on a thread of its own: whether it started. The thread holds `device`'s
/// [`Parts`], never a handle (the module docs).
pub(crate) fn start(device: &Device) -> bool {
    let parts = Parts::of(device);
    // Its command buffers, made here, on the thread that starts it (`Made`): one per batch, and
    // the openings' fill.
    let made = parts
        .pair
        .map(|_| Made::make(device, parts.family, 4 * ROUNDS + 1));
    let pool = match &made {
        Some(Ok(made)) => made.pool,
        _ => vk::CommandPool::null(),
    };
    let switch = Arc::clone(&parts.switch);
    switch.began();
    let builder = std::thread::Builder::new().name("m2v-queue-switch-check".to_owned());
    let spawned = exit_guard::spawn(builder, move || {
        let switch = Arc::clone(&parts.switch);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        crate::loader_word::on_a_check_s_thread();
        crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::Apart);
        run(&parts, made);
        // Its shares go before the teardown that waits for it goes on.
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let dropping = Instant::now();
        drop(parts);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        switch.timed_step(1, dropping);
        switch.ended();
    });
    if spawned.is_err() {
        if pool != vk::CommandPool::null() {
            // SAFETY: the pool is the check's and nothing of it was used.
            unsafe { release::object(device.shared().vk(), pool) };
        }
        switch.ended();
    }
    spawned.is_ok()
}

/// The check's command pool, of the primary family, and its command buffers, one per batch and,
/// last, the openings' one, made as it starts on the thread that starts it (the program's, making its first command queue of
/// the round robin): their `vkAllocateCommandBuffers` is the one call of the check's that the
/// loader serves, which reads the device's loader word (`loader_word.rs`), so the check's thread
/// makes none beside the program's work, whose messages MoltenVK writes it with. The check's
/// thread records them ([`Objects::make`]).
struct Made {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl Made {
    /// `device`'s, for `batches` batches of family `family`; `Err` where a call was refused (the
    /// pool then released).
    fn make(device: &Device, family: u32, batches: usize) -> Result<Made, Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(family);
        // SAFETY: the device is live.
        let pool = unsafe { raw.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        exit_guard::called(Call::CreateCommandPool);
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(batches as u32);
        // The loader serves it; the device's loader word is put back first.
        match device.allocate_command_buffers(&allocate) {
            Ok(buffers) => Ok(Made { pool, buffers }),
            Err(error) => {
                // SAFETY: the pool is the check's, none of its command buffers made.
                unsafe { release::object(raw, pool) };
                Err(error)
            }
        }
    }

    /// Let the pool go unused, inside a section (left to the process's end once it is exiting).
    fn release(self, parts: &Parts) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: the pool is the check's, none of its command buffers submitted.
        unsafe { release::object(parts.vk(), self.pool) };
    }
}

/// The parts of a device its check works with, none of which holds the device (the module docs):
/// shares of its functions, its `VkQueue`s, its parking, its counts, its fences, its loader word,
/// its sharing and its shader cache's store, and what it reads of the device as the check
/// starts.
struct Parts {
    functions: Arc<Functions>,
    queues: Arc<[QueueSlot]>,
    parking: Option<Arc<Parking>>,
    counts: Arc<Counters>,
    fences: Arc<Fences>,
    loader_word: Arc<DeviceWord>,
    switch: Arc<QueueSwitch>,
    /// The device's allocations, whose work in progress the check steps aside for
    /// (`step_aside.rs`).
    allocations: Arc<Allocations>,
    store: Arc<Store>,
    /// The primary family, its pool's.
    family: u32,
    /// How many of the device's `VkQueue`s, the first, the round robin lends.
    round_robin: usize,
    /// The two `VkQueue`s it measures ([`pair`]), where it measures.
    pair: Option<(usize, usize)>,
    /// Nanoseconds per timestamp tick, and the bits a timestamp holds.
    clock: (f64, u32),
    identity: DriverIdentity,
    /// Each memory type's flags, by index: the scratch buffer's memory takes a device-local one
    /// where its buffer does.
    type_flags: Vec<vk::MemoryPropertyFlags>,
}

impl Parts {
    /// `device`'s.
    fn of(device: &Device) -> Parts {
        let shared = device.shared();
        let clock = shared.timing.clock();
        let type_flags = {
            let memory = lock(&shared.memory);
            (0..vk::MAX_MEMORY_TYPES as u32)
                .map_while(|index| memory.type_flags(index))
                .collect()
        };
        Parts {
            functions: shared.functions(),
            queues: Arc::clone(&shared.queues),
            parking: shared.parking.get().cloned(),
            counts: Arc::clone(&shared.counts),
            fences: Arc::clone(&shared.fences),
            loader_word: Arc::clone(&shared.loader_word),
            switch: Arc::clone(&shared.queue_switch),
            allocations: Arc::clone(&shared.allocations.allocations),
            store: shared.shader_cache.content_store_share(),
            family: shared.queue_family,
            round_robin: shared.round_robin(),
            pair: pair(device).filter(|_| timed(device)),
            clock: clock.map_or((0.0, 64), |clock| (clock.period, clock.valid_bits)),
            identity: device.driver_identity(),
            type_flags,
        }
    }

    /// The device's functions.
    fn vk(&self) -> &Vk {
        self.functions.get()
    }

    /// Step aside for the device's allocations and frees on other threads before a round
    /// (`step_aside.rs`): not while the check stops or a call waits for its verdict. Counted where
    /// it waited.
    fn step(&self) {
        let switch = &*self.switch;
        let activity = self.allocations.activity();
        if let Some(waited) = activity.step_aside(|| switch.stopping() || switch.awaited()) {
            self.counts.check_stepped_aside(waited);
        }
    }

    /// What a use of one of the device's `VkQueue`s reaches.
    fn queue_uses(&self) -> QueueUses<'_> {
        QueueUses {
            queues: &self.queues,
            parking: self.parking.as_deref(),
            counts: &self.counts,
            fences: &self.fences,
            functions: &self.functions,
        }
    }

    /// A hold of the device's loader word across a step, where the word was found overwritten
    /// (`loader_word.rs`): none elsewhere.
    fn step_held(&self) -> Option<Held<'_>> {
        // SAFETY: the device is live while its check runs (its teardown stops it first).
        unsafe { self.loader_word.step_hold(None) }
    }

    /// Whether the device's loader word was found overwritten (`loader_word.rs`): a round
    /// started then is held whole.
    fn overwritten(&self) -> bool {
        // SAFETY: the device is live while its check runs (its teardown stops it first).
        unsafe { self.loader_word.overwritten() }
    }

    /// Settle the device's verdict (`Device::settle_queue_switch`).
    fn settle(&self, verdict: Verdict, cost: i64, took: Option<Duration>) {
        let slots = &self.queues[..self.round_robin];
        self.switch.settle_over(slots, verdict, cost, took);
    }
}

/// The check's thread: the measurement, the verdict, the record. Stopped by the device's
/// teardown, it settles `Unchecked` and records nothing.
fn run(parts: &Parts, made: Option<Result<Made, Error>>) {
    let started = Instant::now();
    let (Some(pair), Some(made)) = (parts.pair, made) else {
        parts.settle(Verdict::Unchecked, 0, None);
        return;
    };
    let measured = made.and_then(|made| measure(parts, pair, made));
    parts.counts.queue_switch_check_ran();
    let took = Some(started.elapsed());
    let Ok(Some((alternated, alone))) = measured else {
        parts.settle(Verdict::Unchecked, 0, took);
        return;
    };
    let cost = alternated - alone;
    let verdict = match cost > SWITCH_IDLES {
        true => Verdict::Idles,
        false => Verdict::DoesNotIdle,
    };
    let identity = &parts.identity;
    // A failed write leaves the verdict to the device's life alone.
    let _ =
        (parts.store).write_record_soon(&record_name(identity), encode(identity, verdict, cost));
    parts.settle(verdict, cost, took);
}

/// The median gap, in nanoseconds, between batches on the two `VkQueue`s of `pair` and between
/// batches on one: `Ok(None)` where too few rounds' timestamps were written, the process is
/// exiting or the device's teardown stopped the check, `Err` where a Vulkan call refused.
fn measure(parts: &Parts, pair: (usize, usize), made: Made) -> Result<Option<(i64, i64)>, Error> {
    if parts.switch.stopping() {
        made.release(parts);
        return Ok(None);
    }
    let Some(mut objects) = Objects::make(parts, made)? else {
        return Ok(None);
    };
    let ran = objects.run(parts, pair);
    let gaps = match ran {
        Ok(true) if !parts.switch.stopping() => objects.gaps(parts),
        Ok(_) => Ok(None),
        Err(error) => Err(error),
    };
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    let releasing = Instant::now();
    objects.release(parts);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    parts.switch.timed_step(0, releasing);
    gaps
}

/// The check's Vulkan objects: its command pool and command buffers ([`Made`]), one per batch and
/// the openings' one, a timestamp pool of two queries per batch, its scratch buffer and its
/// memory, its timelines (the gate, and one per `VkQueue` its batches signal), and a fence for
/// each of a round's three submissions, which a held round's submissions signal.
struct Objects {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    /// The openings' command buffer: a barrier on the fills before it, then a fill of the
    /// scratch buffer, recorded for simultaneous use, so an opening submits it [`repeats`]
    /// times.
    ///
    /// [`repeats`]: Objects::repeats
    opening: vk::CommandBuffer,
    scratch: vk::Buffer,
    scratch_memory: vk::DeviceMemory,
    queries: vk::QueryPool,
    gate: vk::Semaphore,
    done: [vk::Semaphore; 2],
    fences: [vk::Fence; 3],
    /// How many times the next round's opening runs its fill: 1 at first, twice as many after a
    /// late round, at most [`MOST_REPEATS`].
    repeats: u32,
    /// Each round's: whether its gate had opened by the time its last batch's submission returned
    /// (its gap then counts for nothing).
    late: Vec<bool>,
    /// Whether every batch submitted completed (or none was submitted): only then are they
    /// released (`false` once the process is exiting mid-run).
    settled: bool,
}

impl Objects {
    /// The objects of `made`, one batch a command buffer, each recorded: its two queries reset,
    /// then a timestamp at the top of the pipe and one at the bottom; and the openings' fill.
    /// `Ok(None)` where the process is exiting (`made` then left to its end).
    fn make(parts: &Parts, made: Made) -> Result<Option<Objects>, Error> {
        let _held = parts.step_held();
        let Some(_busy) = exit_guard::enter() else {
            return Ok(None);
        };
        let mut buffers = made.buffers;
        let opening = buffers.pop().unwrap_or_default();
        let rounds = buffers.len() / 2;
        let mut objects = Objects {
            pool: made.pool,
            buffers,
            opening,
            scratch: vk::Buffer::null(),
            scratch_memory: vk::DeviceMemory::null(),
            queries: vk::QueryPool::null(),
            gate: vk::Semaphore::null(),
            done: [vk::Semaphore::null(); 2],
            fences: [vk::Fence::null(); 3],
            repeats: 1,
            late: vec![false; rounds],
            settled: true,
        };
        let filled = objects.fill(parts);
        drop(_busy);
        match filled {
            Ok(()) => Ok(Some(objects)),
            Err(error) => {
                objects.release(parts);
                Err(error)
            }
        }
    }

    /// Make the scratch buffer and its memory, the query pool, the timelines and the fences, and
    /// record the command buffers, inside the caller's section.
    fn fill(&mut self, parts: &Parts) -> Result<(), Error> {
        let raw = parts.vk();
        self.make_scratch(parts)?;
        let batches = self.buffers.len();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * batches as u32);
        // SAFETY: the device is live, inside a section.
        self.queries = unsafe { raw.create_query_pool(&info, None) }.map_err(vulkan)?;
        exit_guard::called(Call::CreateQueryPool);
        let [first, second] = &mut self.done;
        for semaphore in [&mut self.gate, first, second] {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: the device is live and enables `timelineSemaphore`, inside a section.
            *semaphore = unsafe { raw.create_semaphore(&info, None) }.map_err(vulkan)?;
            exit_guard::called(Call::CreateSemaphore);
        }
        for fence in &mut self.fences {
            // SAFETY: the device is live, inside a section.
            *fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vulkan)?;
            exit_guard::called(Call::CreateFence);
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        for (batch, &buffer) in self.buffers.iter().enumerate() {
            let first = 2 * batch as u32;
            // SAFETY: a new command buffer of the pool, recorded once on this thread, its queries
            // reset in it before they are written; inside a section.
            unsafe {
                raw.begin_command_buffer(buffer, &begin).map_err(vulkan)?;
                exit_guard::called(Call::BeginCommandBuffer);
                raw.cmd_reset_query_pool(buffer, self.queries, first, 2);
                exit_guard::called(Call::CmdResetQueryPool);
                let top = vk::PipelineStageFlags::TOP_OF_PIPE;
                raw.cmd_write_timestamp(buffer, top, self.queries, first);
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, self.queries, first + 1);
                exit_guard::called(Call::CmdWriteTimestamp);
                raw.end_command_buffer(buffer).map_err(vulkan)?;
                exit_guard::called(Call::EndCommandBuffer);
            }
        }
        // The openings' fill, each after the fills before it on its `VkQueue` (a barrier's first
        // scope holds every command earlier in submission order), so repeats add up.
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
        let written = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
        let transfer = vk::PipelineStageFlags::TRANSFER;
        // SAFETY: a new command buffer of the pool, recorded once on this thread; the scratch
        // buffer is the check's, bound, of `DELAY_BYTES` bytes and made for transfers into it;
        // inside a section.
        unsafe {
            raw.begin_command_buffer(self.opening, &begin)
                .map_err(vulkan)?;
            exit_guard::called(Call::BeginCommandBuffer);
            let dependency = vk::DependencyFlags::empty();
            raw.cmd_pipeline_barrier(
                self.opening,
                transfer,
                transfer,
                dependency,
                &written,
                &[],
                &[],
            );
            exit_guard::called(Call::CmdPipelineBarrier);
            raw.cmd_fill_buffer(self.opening, self.scratch, 0, DELAY_BYTES, 0);
            exit_guard::called(Call::CmdFillBuffer);
            raw.end_command_buffer(self.opening).map_err(vulkan)?;
            exit_guard::called(Call::EndCommandBuffer);
        }
        Ok(())
    }

    /// Make the scratch buffer, of [`DELAY_BYTES`], and bind it to memory of its own, of a
    /// device-local type where it takes one, inside the caller's section.
    fn make_scratch(&mut self, parts: &Parts) -> Result<(), Error> {
        let raw = parts.vk();
        let info = vk::BufferCreateInfo::default()
            .size(DELAY_BYTES)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device is live, inside a section.
        let made = unsafe { crate::creation::buffer(raw, &info) };
        exit_guard::called(Call::CreateBuffer);
        self.scratch = made.map_err(vulkan)?;
        // SAFETY: the check's buffer, inside a section.
        let needs = unsafe { raw.get_buffer_memory_requirements(self.scratch) };
        exit_guard::called(Call::GetBufferMemoryRequirements);
        let takes = |index: &u32| needs.memory_type_bits & (1 << index) != 0;
        let types = (0..parts.type_flags.len() as u32).filter(takes);
        let local = |&index: &u32| {
            parts.type_flags[index as usize].contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        };
        let Some(type_index) = types.clone().find(local).or_else(|| types.clone().next()) else {
            return Err(vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        };
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(type_index);
        // SAFETY: the device is live, inside a section; a type the buffer takes.
        let allocated = unsafe { crate::creation::allocate_memory(raw, &info) };
        exit_guard::called(Call::AllocateMemory);
        self.scratch_memory = allocated.map_err(vulkan)?;
        // SAFETY: the check's unbound buffer, and its memory of a type it takes, of its size.
        let bound =
            unsafe { crate::creation::bind_buffer(raw, self.scratch, self.scratch_memory, 0) };
        exit_guard::called(Call::BindBufferMemory);
        bound.map_err(vulkan)
    }

    /// Submit every round's opening and batches on `pair`, and wait for each round's: whether
    /// every batch completed (false once the process is exiting, the objects then left to its
    /// end, and once the device's teardown stopped the check before a round, every batch
    /// submitted completed).
    fn run(&mut self, parts: &Parts, pair: (usize, usize)) -> Result<bool, Error> {
        let mut done = [0u64; 2];
        for round in 0..2 * ROUNDS {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            parts.switch.held.pass();
            if parts.switch.stopping() {
                return Ok(false);
            }
            parts.step();
            let queues = match round % 2 {
                0 => [pair.0, pair.1],
                _ => [pair.0, pair.0],
            };
            let gate = (self.gate, round as u64 + 1);
            // The opening, on the round's first `VkQueue`, opens the gate as its fills complete.
            let mut batches = vec![Batch {
                buffers: vec![self.opening; self.repeats as usize],
                queue: queues[0],
                wait: None,
                signal: gate,
            }];
            for (place, &queue) in queues.iter().enumerate() {
                let lane = usize::from(queue != pair.0);
                done[lane] += 1;
                batches.push(Batch {
                    buffers: vec![self.buffers[2 * round + place]],
                    queue,
                    wait: Some(gate),
                    signal: (self.done[lane], done[lane]),
                });
            }
            let Some(late) = self.round(parts, &batches)? else {
                return Ok(false);
            };
            self.late[round] = late;
            if late {
                self.repeats = (2 * self.repeats).min(MOST_REPEATS);
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                parts.switch.late_round();
            }
        }
        Ok(true)
    }

    /// Submit `batches`, the opening first, then wait for each timeline they signal to reach the
    /// last value a submitted one signals: whether the round was late (its gate open as its last
    /// submission returned), `None` once the process is exiting. Where a submission is refused,
    /// those made before it are waited for, the refusal answered after, and the objects kept
    /// where they could not be.
    fn round(&mut self, parts: &Parts, batches: &[Batch]) -> Result<Option<bool>, Error> {
        self.settled = false;
        // Where the device's loader word was found overwritten as the round starts, held from
        // before its first submission through the wait for its batches, for the check's own
        // fences of them and for the relay's marks after them (`loader_word.rs`): MoltenVK names
        // the device as each completes, after the timelines are signalled. It signals each fence
        // as the batch's Metal command buffer completes, as the submission checks' own fences end
        // their steps; where the queue's failures are not relayed (an adopted device, whose
        // instance has no relay), no mark follows them. Taken inside the round's queues' locks,
        // as each batch's submission takes the word inside its queue's, and kept once they go.
        let hold = parts.overwritten();
        let Some(submitted) = self.submit(parts, batches, hold) else {
            return Ok(None);
        };
        // Each timeline's last value a submitted batch signals: each rises in the order its
        // batches run.
        let mut reached: Vec<(vk::Semaphore, u64)> = Vec::with_capacity(batches.len());
        for &(semaphore, value) in batches[..submitted.count].iter().map(|batch| &batch.signal) {
            match reached.iter_mut().find(|(known, _)| *known == semaphore) {
                Some(known) => known.1 = known.1.max(value),
                None => reached.push((semaphore, value)),
            }
        }
        let fenced = match hold {
            true => &self.fences[..submitted.count],
            false => &[],
        };
        let waited = self.wait(parts, &reached)? && self.wait_fences(parts, fenced)?;
        parts.loader_word.end_step(submitted.step, submitted.marks);
        if !waited {
            return Ok(None);
        }
        self.settled = true;
        match submitted.refused {
            Some(error) => Err(vulkan(error)),
            None => Ok(Some(submitted.late)),
        }
    }

    /// Submit `batches`, each alone on its `VkQueue`, in order until one is refused, inside one
    /// section, with each of the round's `VkQueue`s under its lock from before the first
    /// submission until after the last (the module docs), then read whether the gate had opened:
    /// what was submitted ([`Submitted`]), `None` where the process is exiting.
    fn submit<'p>(&self, parts: &'p Parts, batches: &[Batch], hold: bool) -> Option<Submitted<'p>> {
        let _busy = exit_guard::enter()?;
        let raw = parts.vk();
        let uses = parts.queue_uses();
        let mut queues: Vec<usize> = batches.iter().map(|batch| batch.queue).collect();
        queues.sort_unstable();
        queues.dedup();
        let (count, refused, step, marks) = uses.use_queues_kept(&queues, By::Runtime, |held| {
            let step = match hold {
                true => parts.step_held(),
                false => None,
            };
            let mut submitted = 0;
            let mut refused = None;
            let mut marks = Vec::new();
            for (batch, &own) in batches.iter().zip(&self.fences) {
                // The crate's device tests' hold of a round on one `VkQueue` between its two
                // batches' submissions, its opening and first batch submitted
                // (`tests::device_recovery`, `tests::queue_switch_gate`).
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                if batches
                    .get(2)
                    .is_some_and(|second| std::ptr::eq(batch, second))
                    && batch.queue == batches[1].queue
                {
                    parts.switch.between.pass();
                }
                let fence = match hold {
                    true => own,
                    false => vk::Fence::null(),
                };
                let waits = batch.wait.map(|(semaphore, _)| semaphore);
                let wait_values = batch.wait.map(|(_, value)| value);
                let stages = batch.wait.map(|_| vk::PipelineStageFlags::ALL_COMMANDS);
                let signals = [batch.signal.0];
                let signal_values = [batch.signal.1];
                let mut values = vk::TimelineSemaphoreSubmitInfo::default()
                    .wait_semaphore_values(wait_values.as_slice())
                    .signal_semaphore_values(&signal_values);
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(waits.as_slice())
                    .wait_dst_stage_mask(stages.as_slice())
                    .command_buffers(&batch.buffers)
                    .signal_semaphores(&signals)
                    .push_next(&mut values);
                // SAFETY: the queue is used under its lock, inside a section; each command buffer
                // is recorded and ended, a batch's in no other submission, the opening's recorded
                // for simultaneous use, so it may be pending and repeat; each timeline's signals
                // rise in the order they run (the gate's openings run in their rounds' order on
                // their first `VkQueue`, each lane's batches in its queue's order); a batch's
                // wait is for the gate's value the round's opening signals, submitted before it,
                // so no wait is for a signal yet to be submitted; the fence, where there is one,
                // is unsignaled (made so, or reset once the round before signalled it) and in no
                // pending submission. The device's loader word is held across the call and the
                // relay's mark after it (MoltenVK names the device as it logs each,
                // `loader_word.rs`), taken inside the queue's lock; where it was found
                // overwritten, the round holds it from before its first submission (`step`), so
                // no submission completes while it is let go.
                let ((word, made), mark) =
                    held.use_queue_kept(batch.queue, Uses::OneSignalling, true, |queue| unsafe {
                        let word = parts.loader_word.held();
                        parts.counts.queue_submitted(1, batch.buffers.len());
                        (word, raw.queue_submit(queue, &[submit], fence))
                    });
                drop(word);
                marks.extend(mark);
                exit_guard::called(Call::QueueSubmit);
                if let Err(error) = made {
                    refused = Some(error);
                    break;
                }
                submitted += 1;
            }
            (submitted, refused, step, marks)
        });
        // Late where the gate opened before the last submission returned: the batch the GPU ran
        // first may then have run before the other reached it.
        let late = match count == batches.len() {
            true => {
                // SAFETY: the check's timeline, inside a section.
                let value = unsafe { raw.get_semaphore_counter_value(self.gate) };
                exit_guard::called(Call::GetSemaphoreCounterValue);
                value.map(|value| value >= batches[0].signal.1)
            }
            false => Ok(false),
        };
        let (late, refused) = match late {
            Ok(late) => (late, refused),
            Err(error) => (false, refused.or(Some(error))),
        };
        Some(Submitted {
            count,
            refused,
            step,
            marks,
            late,
        })
    }

    /// Wait for each timeline of `reached` to reach its value, a slice at a time, each a section:
    /// whether they did (false once the process is exiting).
    fn wait(&self, parts: &Parts, reached: &[(vk::Semaphore, u64)]) -> Result<bool, Error> {
        let (semaphores, values): (Vec<vk::Semaphore>, Vec<u64>) = reached.iter().copied().unzip();
        if semaphores.is_empty() {
            return Ok(true);
        }
        loop {
            let Some(_busy) = exit_guard::enter() else {
                return Ok(false);
            };
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            // SAFETY: the timelines are the check's, each signalled to its value by a submission
            // made.
            let waited = unsafe {
                parts
                    .vk()
                    .wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64)
            };
            exit_guard::called(Call::WaitSemaphores);
            match waited {
                Ok(()) => return Ok(true),
                Err(vk::Result::TIMEOUT) => continue,
                Err(error) => return Err(vulkan(error)),
            }
        }
    }

    /// Wait for `fences`, the check's own of a held round's batches submitted, a slice at a time,
    /// each a section, then reset them for the next round: whether they were signalled (false
    /// once the process is exiting).
    fn wait_fences(&self, parts: &Parts, fences: &[vk::Fence]) -> Result<bool, Error> {
        if fences.is_empty() {
            return Ok(true);
        }
        loop {
            let Some(_busy) = exit_guard::enter() else {
                return Ok(false);
            };
            // SAFETY: the check's fences, each in a submission made.
            let waited =
                unsafe { (parts.vk()).wait_for_fences(fences, true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(error) => return Err(vulkan(error)),
            }
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        parts.switch.waited_fences(fences.len());
        let Some(_busy) = exit_guard::enter() else {
            return Ok(false);
        };
        // SAFETY: each is signalled, its submission completed.
        let reset = unsafe { parts.vk().reset_fences(fences) };
        exit_guard::called(Call::ResetFences);
        reset.map_err(vulkan)?;
        Ok(true)
    }

    /// The median gap between batches on the two `VkQueue`s and on one, in nanoseconds, from the
    /// timestamps of the rounds that were not late (`Ok(None)` where fewer than [`FEWEST`] of
    /// either kind were written).
    fn gaps(&self, parts: &Parts) -> Result<Option<(i64, i64)>, Error> {
        let _held = parts.step_held();
        let Some(_busy) = exit_guard::enter() else {
            return Ok(None);
        };
        let (period, valid_bits) = parts.clock;
        let count = 2 * self.buffers.len();
        let mut stamps = vec![[0u64; 2]; count];
        // SAFETY: the pool's queries, each written by a batch that completed; 64-bit values each
        // with its availability.
        let read = unsafe {
            parts.vk().get_query_pool_results(
                self.queries,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        exit_guard::called(Call::GetQueryPoolResults);
        match read {
            Ok(()) | Err(vk::Result::NOT_READY) => {}
            Err(error) => return Err(vulkan(error)),
        }
        let mask = match valid_bits {
            64.. => u64::MAX,
            bits => (1u64 << bits) - 1,
        };
        let at = |query: usize| {
            let [value, available] = stamps[query];
            (available != 0).then_some(value & mask)
        };
        // Ticks from `from` to `to`, in nanoseconds, either way.
        let span = |from: u64, to: u64| {
            let ticks = match to >= from {
                true => (to - from) as f64,
                false => -((from - to) as f64),
            };
            (ticks * period) as i64
        };
        // The batch of query pair `batch`: its start and end.
        let batch = |batch: usize| Some((at(2 * batch)?, at(2 * batch + 1)?));
        let mut apart = Vec::new();
        let mut together = Vec::new();
        for (round, &late) in self.late.iter().enumerate() {
            let (false, Some(one), Some(two)) = (late, batch(2 * round), batch(2 * round + 1))
            else {
                continue;
            };
            let (first, second) = match one.0 <= two.0 {
                true => (one, two),
                false => (two, one),
            };
            let gap = span(first.1, second.0);
            match round % 2 {
                0 => apart.push(gap),
                _ => together.push(gap),
            }
        }
        if apart.len() < FEWEST || together.len() < FEWEST {
            return Ok(None);
        }
        Ok(Some((median(&mut apart), median(&mut together))))
    }

    /// Let the objects go, inside a section, where every batch submitted completed; else leave
    /// them to the device's end.
    fn release(&self, parts: &Parts) {
        if !self.settled {
            return;
        }
        let _held = parts.step_held();
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = parts.vk();
        // SAFETY: no submission that names them is pending (each completed, or none was made),
        // inside a section; a null handle is skipped.
        unsafe {
            for semaphore in [self.gate, self.done[0], self.done[1]] {
                if semaphore != vk::Semaphore::null() {
                    release::object(raw, semaphore);
                }
            }
            for fence in self.fences {
                if fence != vk::Fence::null() {
                    release::object(raw, fence);
                }
            }
            if self.queries != vk::QueryPool::null() {
                release::object(raw, self.queries);
            }
            if self.scratch != vk::Buffer::null() {
                release::object(raw, self.scratch);
            }
            if self.scratch_memory != vk::DeviceMemory::null() {
                release::object(raw, self.scratch_memory);
            }
            release::object(raw, self.pool);
        }
    }
}

/// One submission of a round: its command buffers, its `VkQueue`, the timeline value it waits
/// for, if any, and the one it signals.
struct Batch {
    buffers: Vec<vk::CommandBuffer>,
    queue: usize,
    wait: Option<(vk::Semaphore, u64)>,
    signal: (vk::Semaphore, u64),
}

/// What a round's submission answers ([`Objects::submit`]).
struct Submitted<'p> {
    /// How many of its batches were submitted.
    count: usize,
    /// The refusal of a submission, or of the read of the gate after them.
    refused: Option<vk::Result>,
    /// The hold of the device's loader word where it was found overwritten.
    step: Option<Held<'p>>,
    /// The relay's marks after the batches, kept.
    marks: Vec<Mark<'p>>,
    /// Whether the gate had opened as the last submission returned ([`Objects::late`]).
    late: bool,
}

/// The median of `values` (sorted here).
fn median(values: &mut [i64]) -> i64 {
    values.sort_unstable();
    values[values.len() / 2]
}
