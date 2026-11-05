//! The device's check of whether its GPU idles when consecutive work comes from another `VkQueue`
//! (`queue_share.rs`), on a thread of its own, once per driver, release and runtime.
//!
//! - **What it measures.** Two `VkQueue`s of the device's primary family that the round robin
//!   lends ([`pair`]), and batches of one command buffer each that write a timestamp at the top of
//!   the pipe and one at the bottom, nothing between. Each round submits two such batches, both
//!   waiting for one value of a gate timeline that the host signals once both are submitted, so
//!   the GPU holds both before it runs either: on the two `VkQueue`s (the alternation), or both
//!   on one (its control). The gap between the end of the batch the GPU ran
//!   first and the start of the other is a switch's where the two were apart, a submission's where
//!   they were not, and the switch cost is the median gap of the first less the median of the
//!   second. The rounds alternate between the two, [`ROUNDS`] each, so work of the program's own
//!   meanwhile reaches both alike. On the RTX 5080 Laptop GPU (NVIDIA 615.71.09) it measured 49.0
//!   to 51.3 us in six runs, where the GPU idles 56 to 62 us between consecutive command buffers
//!   of different queues, and a chain of batches each waiting on a timeline for the one before
//!   it measured 53.2 to 59.3 us.
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
//!   timestamps: about 2 ms of the GPU's time where the switch costs 60 us, and the host's waits
//!   for each round.
//! - **Kept.** A measured verdict is written as one record in the shader cache's content directory
//!   (`shader_cache/store.rs`), named by the driver's identity, the check's version and the
//!   runtime's (`queue_share::record_name`): a device of the same driver and release reads it as
//!   it is made and runs no check. A failed run is kept nowhere.
//! - **Its thread.** Each reach of the device runs inside a section of the exit guard
//!   (`exit_guard.rs`), as every runtime thread's does: the objects' making, the submissions, each
//!   wait of at most [`WAIT_SLICE`], and the releases. Once the process is exiting it ends,
//!   unrecorded, and leaves its objects to the process's end. A submission refused still opens its
//!   round's gate, so no batch submitted waits for good.
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

use crate::command_buffer::fences::Fences;
use crate::device::functions::{Functions, Vk};
use crate::device::relay::{By, QueueUses};
use crate::device::submit_stalls::Uses;
use crate::device::{Device, QueueSlot};
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::freed_memory::Parking;
use crate::in_place::DriverIdentity;
use crate::loader_word::LoaderWord;
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
    let switch = Arc::clone(&parts.switch);
    switch.began();
    let builder = std::thread::Builder::new().name("m2v-queue-switch-check".to_owned());
    let spawned = exit_guard::spawn(builder, move || {
        let switch = Arc::clone(&parts.switch);
        run(&parts);
        // Its shares go before the teardown that waits for it goes on.
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let dropping = Instant::now();
        drop(parts);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        switch.timed_step(1, dropping);
        switch.ended();
    });
    if spawned.is_err() {
        switch.ended();
    }
    spawned.is_ok()
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
    loader_word: Arc<LoaderWord>,
    switch: Arc<QueueSwitch>,
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
}

impl Parts {
    /// `device`'s.
    fn of(device: &Device) -> Parts {
        let shared = device.shared();
        let clock = shared.timing.clock();
        Parts {
            functions: shared.functions(),
            queues: Arc::clone(&shared.queues),
            parking: shared.parking.get().cloned(),
            counts: Arc::clone(&shared.counts),
            fences: Arc::clone(&shared.fences),
            loader_word: Arc::clone(&shared.loader_word),
            switch: Arc::clone(&shared.queue_switch),
            store: shared.shader_cache.content_store_share(),
            family: shared.queue_family,
            round_robin: shared.round_robin(),
            pair: pair(device).filter(|_| timed(device)),
            clock: clock.map_or((0.0, 64), |clock| (clock.period, clock.valid_bits)),
            identity: device.driver_identity(),
        }
    }

    /// The device's functions.
    fn vk(&self) -> &Vk {
        self.functions.get()
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

    /// Settle the device's verdict (`Device::settle_queue_switch`).
    fn settle(&self, verdict: Verdict, cost: i64, took: Option<Duration>) {
        let slots = &self.queues[..self.round_robin];
        self.switch.settle_over(slots, verdict, cost, took);
    }
}

/// The check's thread: the measurement, the verdict, the record. Stopped by the device's
/// teardown, it settles `Unchecked` and records nothing.
fn run(parts: &Parts) {
    let started = Instant::now();
    let Some(pair) = parts.pair else {
        parts.settle(Verdict::Unchecked, 0, None);
        return;
    };
    let measured = measure(parts, pair);
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
fn measure(parts: &Parts, pair: (usize, usize)) -> Result<Option<(i64, i64)>, Error> {
    if parts.switch.stopping() {
        return Ok(None);
    }
    let Some(mut objects) = Objects::make(parts, 4 * ROUNDS)? else {
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

/// The check's Vulkan objects: a command pool of the primary family and its command buffers, one
/// per batch, a timestamp pool of two queries per batch, and its timelines: the gate, and one per
/// `VkQueue` its batches signal.
struct Objects {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    queries: vk::QueryPool,
    gate: vk::Semaphore,
    done: [vk::Semaphore; 2],
    /// Whether every batch submitted completed (or none was submitted): only then are they
    /// released (`false` once the process is exiting mid-run).
    settled: bool,
}

impl Objects {
    /// The objects for `batches` batches, each command buffer recorded: its two queries reset,
    /// then a timestamp at the top of the pipe and one at the bottom. `Ok(None)` where the process
    /// is exiting.
    fn make(parts: &Parts, batches: usize) -> Result<Option<Objects>, Error> {
        let Some(_busy) = exit_guard::enter() else {
            return Ok(None);
        };
        let raw = parts.vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(parts.family);
        // SAFETY: the device is live, inside a section.
        let pool = unsafe { raw.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        exit_guard::called(Call::CreateCommandPool);
        let mut objects = Objects {
            pool,
            buffers: Vec::new(),
            queries: vk::QueryPool::null(),
            gate: vk::Semaphore::null(),
            done: [vk::Semaphore::null(); 2],
            settled: true,
        };
        let made = objects.fill(parts, batches);
        drop(_busy);
        match made {
            Ok(()) => Ok(Some(objects)),
            Err(error) => {
                objects.release(parts);
                Err(error)
            }
        }
    }

    /// Make the pool's command buffers, the query pool and the timelines, and record the command
    /// buffers, inside the caller's section.
    fn fill(&mut self, parts: &Parts, batches: usize) -> Result<(), Error> {
        let raw = parts.vk();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(batches as u32);
        // SAFETY: the device is live (its teardown waits for the check), and the pool is the
        // check's, which this thread alone uses; its loader word is put back first
        // (`loader_word.rs`).
        let allocated = unsafe {
            parts.loader_word.restore();
            raw.allocate_command_buffers(&allocate)
        };
        exit_guard::called(Call::AllocateCommandBuffers);
        self.buffers = allocated.map_err(vulkan)?;
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
        Ok(())
    }

    /// Submit every round's batches on `pair`, and wait for each round's: whether every batch
    /// completed (false once the process is exiting, the objects then left to its end, and once
    /// the device's teardown stopped the check before a round, every batch submitted completed).
    fn run(&mut self, parts: &Parts, pair: (usize, usize)) -> Result<bool, Error> {
        let mut done = [0u64; 2];
        for round in 0..2 * ROUNDS {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            parts.switch.held.pass();
            if parts.switch.stopping() {
                return Ok(false);
            }
            let queues = match round % 2 {
                0 => [pair.0, pair.1],
                _ => [pair.0, pair.0],
            };
            let gate = round as u64 + 1;
            let mut batches = Vec::with_capacity(2);
            for (place, &queue) in queues.iter().enumerate() {
                let lane = usize::from(queue != pair.0);
                done[lane] += 1;
                batches.push(Batch {
                    buffer: self.buffers[2 * round + place],
                    queue,
                    wait: (self.gate, gate),
                    signal: (self.done[lane], done[lane]),
                });
            }
            if !self.round(parts, &batches, gate, done)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Submit `batches`, then wait for each `VkQueue`'s timeline to reach its value of `done`:
    /// whether they completed (false once the process is exiting). Where a submission is refused,
    /// those made before it are waited for, the refusal answered after, and the objects kept
    /// where they could not be.
    fn round(
        &mut self,
        parts: &Parts,
        batches: &[Batch],
        gate: u64,
        done: [u64; 2],
    ) -> Result<bool, Error> {
        self.settled = false;
        let (submitted, refused) = match self.submit(parts, batches, gate) {
            Some(answer) => answer,
            None => return Ok(false),
        };
        let mut reached = done;
        for batch in &batches[submitted..] {
            let lane = usize::from(batch.signal.0 != self.done[0]);
            reached[lane] = reached[lane].min(batch.signal.1 - 1);
        }
        if !self.wait(parts, reached)? {
            return Ok(false);
        }
        self.settled = true;
        match refused {
            Some(error) => Err(vulkan(error)),
            None => Ok(true),
        }
    }

    /// Submit `batches`, each alone on its `VkQueue`, in order until one is refused, then signal
    /// the gate to `gate` from the host, inside one section: how many were submitted, and the
    /// refusal; `None` where the process is exiting.
    fn submit(
        &self,
        parts: &Parts,
        batches: &[Batch],
        gate: u64,
    ) -> Option<(usize, Option<vk::Result>)> {
        let _busy = exit_guard::enter()?;
        let raw = parts.vk();
        let uses = parts.queue_uses();
        let mut submitted = 0;
        let mut refused = None;
        for batch in batches {
            let buffers = [batch.buffer];
            let waits = [batch.wait.0];
            let wait_values = [batch.wait.1];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let signals = [batch.signal.0];
            let signal_values = [batch.signal.1];
            let mut values = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&buffers)
                .signal_semaphores(&signals)
                .push_next(&mut values);
            // SAFETY: the queue is used under its lock, inside a section; the command buffer is
            // recorded, ended and in no other submission; each timeline's signals rise in the order
            // they run (each lane's batches run in its queue's order); the gate's wait is
            // signalled below.
            let made = uses.use_queue(
                batch.queue,
                By::Runtime,
                Uses::OneSignalling,
                |queue| unsafe {
                    parts.counts.queue_submitted(1, 1);
                    raw.queue_submit(queue, &[submit], vk::Fence::null())
                },
            );
            exit_guard::called(Call::QueueSubmit);
            if let Err(error) = made {
                refused = Some(error);
                break;
            }
            submitted += 1;
        }
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.gate)
            .value(gate);
        // SAFETY: the gate's value rises with each signal, its only signaller the host.
        let signalled = unsafe { raw.signal_semaphore(&signal) };
        exit_guard::called(Call::SignalSemaphore);
        Some((submitted, refused.or(signalled.err())))
    }

    /// Wait for each `VkQueue`'s timeline to reach its value of `done`, a slice at a time, each a
    /// section: whether they did (false once the process is exiting).
    fn wait(&self, parts: &Parts, done: [u64; 2]) -> Result<bool, Error> {
        let (semaphores, values): (Vec<vk::Semaphore>, Vec<u64>) = (self.done.iter())
            .zip(done)
            .filter(|&(_, value)| value > 0)
            .map(|(&semaphore, value)| (semaphore, value))
            .unzip();
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

    /// The median gap between batches on the two `VkQueue`s and on one, in nanoseconds, from the
    /// timestamps (`Ok(None)` where fewer than [`FEWEST`] of either kind were written).
    fn gaps(&self, parts: &Parts) -> Result<Option<(i64, i64)>, Error> {
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
        for round in 0..2 * ROUNDS {
            let (Some(one), Some(two)) = (batch(2 * round), batch(2 * round + 1)) else {
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
            if self.queries != vk::QueryPool::null() {
                release::object(raw, self.queries);
            }
            release::object(raw, self.pool);
        }
    }
}

/// One batch: its command buffer, its `VkQueue`, the timeline value it waits for and the one it
/// signals.
struct Batch {
    buffer: vk::CommandBuffer,
    queue: usize,
    wait: (vk::Semaphore, u64),
    signal: (vk::Semaphore, u64),
}

/// The median of `values` (sorted here).
fn median(values: &mut [i64]) -> i64 {
    values.sort_unstable();
    values[values.len() / 2]
}
