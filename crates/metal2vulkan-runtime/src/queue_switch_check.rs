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
//!   wait of at most [`WAIT_SLICE`], and the releases. The check holds its device for its run, a
//!   few milliseconds, so its objects never outlive it; once the process is exiting it ends,
//!   unrecorded, and leaves them to the process's end. A submission refused still opens its
//!   round's gate, so no batch submitted waits for good.

use crate::device::relay::By;
use crate::device::submit_stalls::Uses;
use crate::device::Device;
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::queue_share::{encode, record_name, Verdict, SWITCH_IDLES};
use crate::release;
use ash::vk;
use std::time::{Duration, Instant};

/// The rounds of each kind, the two `VkQueue`s' and the one's.
pub(crate) const ROUNDS: usize = 24;

/// How long one wait for a round's batches lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The fewest rounds of each kind whose timestamps were all written that a verdict needs.
const FEWEST: usize = ROUNDS / 2;

/// The two `VkQueue`s the check measures, by index among the device's: the first two the round
/// robin lends of the primary family, where the device calibrates its timestamps and that family
/// writes them; `None` elsewhere.
pub(crate) fn pair(device: &Device) -> Option<(usize, usize)> {
    let shared = device.shared();
    shared.timing.clock()?.calibration?;
    let mut same = (0..shared.round_robin())
        .filter(|&index| shared.queues[index].family == shared.queue_family);
    Some((same.next()?, same.next()?))
}

/// Start the check on a thread of its own: whether it started.
pub(crate) fn start(device: &Device) -> bool {
    let weak = device.downgrade();
    let builder = std::thread::Builder::new().name("m2v-queue-switch-check".to_owned());
    exit_guard::spawn(builder, move || run(weak)).is_ok()
}

/// The check's thread: the measurement, the verdict, the record.
fn run(weak: crate::device::WeakDevice) {
    let started = Instant::now();
    let Some(device) = weak.upgrade() else {
        return;
    };
    let Some(pair) = pair(&device) else {
        device.settle_queue_switch(Verdict::Unchecked, 0, None);
        return;
    };
    let measured = measure(&device, pair);
    device.shared().counts.queue_switch_check_ran();
    let took = Some(started.elapsed());
    let Ok(Some((alternated, alone))) = measured else {
        device.settle_queue_switch(Verdict::Unchecked, 0, took);
        return;
    };
    let cost = alternated - alone;
    let verdict = match cost > SWITCH_IDLES {
        true => Verdict::Idles,
        false => Verdict::DoesNotIdle,
    };
    let identity = device.driver_identity();
    // A failed write leaves the verdict to the device's life alone.
    let _ = (device.shared().shader_cache.content_store())
        .write_record_soon(&record_name(&identity), encode(&identity, verdict, cost));
    device.settle_queue_switch(verdict, cost, took);
}

/// The median gap, in nanoseconds, between batches on the two `VkQueue`s of `pair` and between
/// batches on one: `Ok(None)` where too few rounds' timestamps were written or the process is
/// exiting, `Err` where a Vulkan call refused.
fn measure(device: &Device, pair: (usize, usize)) -> Result<Option<(i64, i64)>, Error> {
    let Some(mut objects) = Objects::make(device, 4 * ROUNDS)? else {
        return Ok(None);
    };
    let ran = objects.run(device, pair);
    let gaps = match ran {
        Ok(true) => objects.gaps(device),
        Ok(false) => Ok(None),
        Err(error) => Err(error),
    };
    objects.release(device);
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
    fn make(device: &Device, batches: usize) -> Result<Option<Objects>, Error> {
        let Some(_busy) = exit_guard::enter() else {
            return Ok(None);
        };
        let shared = device.shared();
        let raw = shared.vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(shared.queue_family);
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
        let made = objects.fill(device, batches);
        drop(_busy);
        match made {
            Ok(()) => Ok(Some(objects)),
            Err(error) => {
                objects.release(device);
                Err(error)
            }
        }
    }

    /// Make the pool's command buffers, the query pool and the timelines, and record the command
    /// buffers, inside the caller's section.
    fn fill(&mut self, device: &Device, batches: usize) -> Result<(), Error> {
        let raw = device.shared().vk();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(batches as u32);
        self.buffers = device.allocate_command_buffers(&allocate)?;
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
    /// completed (false once the process is exiting, the objects then left to its end).
    fn run(&mut self, device: &Device, pair: (usize, usize)) -> Result<bool, Error> {
        let mut done = [0u64; 2];
        for round in 0..2 * ROUNDS {
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
            if !self.round(device, &batches, gate, done)? {
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
        device: &Device,
        batches: &[Batch],
        gate: u64,
        done: [u64; 2],
    ) -> Result<bool, Error> {
        self.settled = false;
        let (submitted, refused) = match self.submit(device, batches, gate) {
            Some(answer) => answer,
            None => return Ok(false),
        };
        let mut reached = done;
        for batch in &batches[submitted..] {
            let lane = usize::from(batch.signal.0 != self.done[0]);
            reached[lane] = reached[lane].min(batch.signal.1 - 1);
        }
        if !self.wait(device, reached)? {
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
        device: &Device,
        batches: &[Batch],
        gate: u64,
    ) -> Option<(usize, Option<vk::Result>)> {
        let _busy = exit_guard::enter()?;
        let shared = device.shared();
        let raw = shared.vk();
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
            let made = shared.use_queue(
                batch.queue,
                By::Runtime,
                Uses::OneSignalling,
                |queue| unsafe {
                    shared.counts.queue_submitted(1, 1);
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
    fn wait(&self, device: &Device, done: [u64; 2]) -> Result<bool, Error> {
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
                device
                    .shared()
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
    fn gaps(&self, device: &Device) -> Result<Option<(i64, i64)>, Error> {
        let Some(_busy) = exit_guard::enter() else {
            return Ok(None);
        };
        let shared = device.shared();
        let Some(clock) = shared.timing.clock() else {
            return Ok(None);
        };
        let count = 2 * self.buffers.len();
        let mut stamps = vec![[0u64; 2]; count];
        // SAFETY: the pool's queries, each written by a batch that completed; 64-bit values each
        // with its availability.
        let read = unsafe {
            shared.vk().get_query_pool_results(
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
        let mask = match clock.valid_bits {
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
            (ticks * clock.period) as i64
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
    fn release(&self, device: &Device) {
        if !self.settled {
            return;
        }
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = device.shared().vk();
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
