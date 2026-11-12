//! The device's staging queue (`create::queue_plan`): the first queue of a family of transfers
//! alone, a copy engine; else, on a device where no memory the host caches holds a texture's image
//! (`memory::host_cached_placement`), the first queue of a family of graphics or compute work none
//! of the device's own queues are of (an RX 7900 XTX's compute family). There a `Shared` or
//! `Managed` texture's CPU copies run on a device that copies none on the host from memory the
//! host caches (`texture/host_copy.rs`). An image that copies here is concurrent among the
//! device's families and this one ([`Device::staging_sharing`]), so no image changes owner; every
//! other resource keeps the device's sharing (`sharing.rs`).
//!
//! - **Why not the host, where its memory is uncached.** The host reads uncached memory slowly
//!   (`memory::host_cached`). On the XTX (RADV, Mesa 26.2.6), which has no device-local type the
//!   host caches, a texture's host-transfer image lay in its device-local host-visible type
//!   (resizable BAR), and a `get_bytes` copied on the host out of it took 33 to 41 ms at
//!   256 x 256, 1.26 to 1.71 s in most calls at 1920 x 1080 (3.2 and 5.4 ms at best), 36 to 49 ms
//!   at 4096 x 4096, and past the end of 1.5 s of GPU fills when they ran; copied here, 0.10 to
//!   0.13 ms, 2.0 to 2.2 ms (1.9 to 2.1 ms with the fills running) and 17 ms
//!   (`tests::cpu_copy_times`).
//! - **Why a queue of its own.** A copy on the device's first `VkQueue` waits behind every command
//!   buffer submitted there before it: behind a 2 s kernel, a 1080p `get_bytes` returned after
//!   1.99 s on an RTX 3090. It never waits for a wait no submitted signal meets: the scheduler
//!   holds a command buffer whose event wait only a later commit's signal meets on the host,
//!   unsubmitted (`scheduler/mod.rs`), so a copy beside it returns. Through an RX 550's first
//!   `VkQueue` (RADV, Mesa 26.2.6, which has no host copies), a read, 4097 one-texel writes and a
//!   read beside such a command buffer took 0.15 s, for an event and for a shared event
//!   (`tests::copies_beside_a_held_wait`). A plain event's wait for a
//!   signal the caller reserved (`Event::reserve_vk_signal`) goes to its `VkQueue` before that
//!   signal is submitted, and there the same copies waited until it was. Metal's `getBytes` and
//!   `replaceRegion` of shared storage are CPU copies that wait for no GPU work. Another queue of
//!   the graphics family is not enough: on the 3090 (615.71.09) a readback there waited 1.88 s for
//!   a 2 s kernel running on the first, where the copy engine's took 7.8 to 13.5 ms with the
//!   kernel still running, and 2.65 ms idle (raw Vulkan). Here nothing but these
//!   copies runs, so a copy waits for itself and the staging work submitted before it alone. A
//!   queue of compute work shares the GPU's compute units with the program's work, so a copy
//!   there may still wait behind a kernel that fills them (up to 1.05 s on a Radeon 780M):
//!   which is why it stages only where the host's copies would read uncached memory.
//! - **What it waits for.** Each submission waits, on the GPU, for the value through which each
//!   of the device's other `VkQueue`s completed as the host saw it: work that already completed,
//!   so the wait costs nothing, and the writes of every command buffer the program saw complete
//!   are visible to the copy. It never waits for work in flight: a command in flight meanwhile
//!   is the program's race, as on Metal.
//! - **What waits for it.** Each submission signals the staging queue's own timeline. A host call
//!   that needs what it did on the CPU (`get_bytes`, a texture's `VkImage` handed out, the
//!   program's own use of a `VkQueue`) waits for it on the host. Every other submission of the
//!   device's, a batch or an internal one on any `VkQueue`, waits for it on the GPU until the host
//!   saw it complete ([`Device::staging_wait`]), but a batch of a `VkQueue` where an earlier
//!   batch waited for it (`command_buffer/submit.rs`): the copies it made are visible to the
//!   commands after that wait, in submission order there, which never waits for work of the
//!   program's. A submission is not waited for
//!   before the next one is recorded: its command buffer is one of up to [`BUFFERS`], each reused
//!   once the host saw the value it signalled reached.
//! - **Who uses it.** A texture made to copy through it (`texture/staging_copy.rs`) has its first
//!   move and zeros, and its `replace_region` copies, queued for the next submission here, which
//!   carries them all ([`StagingLock::submit`]): the next batch or internal submission of the
//!   device's makes it before its own, and a `get_bytes` carries them before its read. So none
//!   of its work waits in `texture/first_move.rs` for a batch, and neither its making nor its
//!   writes wait for a submission. A buffer's `VkBuffer` handed out, or its memory exported,
//!   has its own fills still waiting (a new `Private` buffer's zeros or staged bytes) made here,
//!   in a submission of their own the call waits for (`Device::settle_mine`): every buffer of the
//!   device's is shared with this queue's family (`sharing.rs`).
//!
//! The device's fill queue, where it has one (`create::queue_plan`): the staging queue's family's
//! second queue, where that family is of transfers alone (NVIDIA's copy engine). A batch's new
//! buffers' fills (`texture/first_move.rs`: a new `Private` buffer's or a `Managed` buffer's
//! device copy's zeros, a buffer's staged bytes, a `Managed` buffer's copies in after
//! `did_modify_range`, a rebind's or a making's) run there, in a submission of their own the
//! batch waits for ([`Device::submit_fills`]), rather than at the start of the batch's own work
//! on its `VkQueue`.
//!
//! - **Why.** A copy through a queue of the graphics family runs on the engine the program's
//!   kernels run on. On an RTX 3090 (615.71.09), a 64 MiB `Managed` buffer copied in each frame
//!   that way left another queue's kernels 20% of their rate alone, and at 256 MiB 11%
//!   (`tests::managed_copies_bench`); in raw Vulkan on an RTX 5080 Laptop GPU (615.71.09), buffer
//!   copies through the graphics family beside such kernels moved 13.5 GB/s and left them 52 to
//!   56% of their rate, and through the copy engine 28.5 GB/s and 92 to 97%
//!   (`examples/copy_engine.rs`).
//! - **Why not the staging queue.** A fill comes after the batches before its batch on that
//!   batch's `VkQueue` (below); on the staging queue, which waits for no work in flight, every
//!   staging copy behind it would wait for those batches too.
//! - **What it waits for.** The batches before the batch on its `VkQueue` (that queue's timeline at
//!   the value before the batch's), what the batch waits for (events, semaphores, the earlier
//!   work of other queues its plan orders it after, earlier first work and staging copies), and
//!   each timeline's completed value, for visibility: what the fills waited for at the start of
//!   the batch's work. A batch that a group of batches held (`command_buffer/group.rs`), whose
//!   values are not submitted yet, waits for a binary semaphore (which its one wait consumes), or
//!   took an acquire from an external agent keeps its fills at its start; so does one that finds
//!   every command buffer of the fill queue's pending ([`Device::fill_queue_has_room`]), as a wait
//!   on the host for one may wait for work that waits for the caller.
//! - **What waits for it.** The batch, before any of its commands; and every other submission of
//!   the device's, a batch or an internal one on any `VkQueue`, on the GPU until the host saw it
//!   complete, as for the staging queue ([`Device::fill_queue_wait`]): any of their commands may
//!   reach the buffers unnamed. The fills' writes are available once the submission's signal is
//!   ("The first access scope includes all memory access performed by the device", the Vulkan
//!   specification, 7.4.1 Semaphore Signaling) and visible to the commands after the wait. Every
//!   buffer is concurrent among the device's families and this one (`sharing.rs`), so none
//!   changes owner: "concurrent access to any range or image subresource of the object from
//!   multiple queue families is supported" (`VK_SHARING_MODE_CONCURRENT`), where an exclusive one
//!   would need "their ownership explicitly transferred" (7.7.6 Queue Family Ownership Transfer).
//! - **What it holds.** Each buffer's memory and each fill's staging buffer, until the batch that
//!   waited for the submission completed ([`FillSubmission`]).
//! - **Not moved.** A texture's first move and zeros, host calls' staged texel copies and an
//!   internal submission's fills stay where they were; encoders' copies of host shadows run in
//!   the program's command buffers.

use super::submit::OwnTimeline;
use super::{lock, Device, LoaderWord};
use crate::error::{internal, vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::sharing::Sharing;
use crate::texture::first_move::Fills;
use crate::work_counts::Begun;
use ash::vk;
use ash::vk::Handle;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// How long one wait for a staging submission lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The most command buffers the staging queue records into: past them, a submission waits on the
/// host for the oldest one's before it records (the module docs).
pub(crate) const BUFFERS: usize = 8;

/// What a queue of this module's is for (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// The staging queue: copies that wait for no work in flight.
    Staging,
    /// The fill queue: a batch's fills, after the work before it on its `VkQueue`.
    Fills,
}

/// The staging queue, or the fill queue, and its objects.
pub(super) struct StagingQueue {
    role: Role,
    /// The queue's family: one of transfers alone, or of graphics or compute work none of the
    /// device's own queues are of (the module docs).
    family: u32,
    /// The family's queue flags, which the barriers recorded for it follow (`image_barrier.rs`).
    flags: vk::QueueFlags,
    /// The device's families, then the queue's: those an image that copies here is shared
    /// among.
    families: Vec<u32>,
    /// The queue's loader word, put back before each use (`relay.rs`).
    word: LoaderWord,
    /// The queue, and the objects its submissions use once made: under one lock, held for a
    /// submission's recording and its `vkQueueSubmit`, never for a wait.
    objects: Mutex<Objects>,
    /// The timeline every submission signals, once made: read with no lock
    /// ([`Device::staging_wait`]).
    timeline: OnceLock<vk::Semaphore>,
    /// The value the last submission made signals, stored under `objects`' lock once it was made.
    submitted: AtomicU64,
    /// The timeline as the host saw it complete.
    pub(super) done: OwnTimeline,
    /// The tests' hold on the submissions ([`Device::hold_staging_submissions`]): a timeline and
    /// a value each waits for on the GPU besides.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    held: Mutex<Option<(vk::Semaphore, u64)>>,
}

struct Objects {
    queue: vk::Queue,
    made: Option<Made>,
}

/// The command pool, command buffers and timeline of the staging submissions, made on first use.
struct Made {
    pool: vk::CommandPool,
    /// Each command buffer, and the value of the last submission of it: free once the host saw
    /// the timeline reach it (0: never submitted).
    buffers: Vec<(vk::CommandBuffer, u64)>,
    timeline: vk::Semaphore,
    /// The value the last submission signalled.
    signalled: u64,
}

impl StagingQueue {
    /// The queue of `role`, `queue` of `family`, whose queue flags are `flags`, on a device that
    /// submits to `families`.
    ///
    /// # Safety
    ///
    /// `queue` was just returned by the device: its word is the loader's.
    pub(super) unsafe fn new(
        role: Role,
        family: u32,
        flags: vk::QueueFlags,
        queue: vk::Queue,
        families: &[u32],
    ) -> StagingQueue {
        StagingQueue {
            role,
            family,
            flags,
            families: families.iter().copied().chain([family]).collect(),
            // SAFETY: the caller's contract.
            word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
            objects: Mutex::new(Objects { queue, made: None }),
            timeline: OnceLock::new(),
            submitted: AtomicU64::new(0),
            done: OwnTimeline::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held: Mutex::new(None),
        }
    }

    /// Destroy the objects; the device is idle.
    pub(super) fn destroy(&self, device: &ash::Device) {
        if let Some(made) = lock(&self.objects).made.take() {
            // SAFETY: the device is idle and nothing else holds these objects.
            unsafe {
                release::object(device, made.timeline);
                release::object(device, made.pool);
            }
        }
    }
}

impl Made {
    fn new(device: &Device, family: u32) -> Result<Made, Error> {
        let raw = device.shared().vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(family);
        // SAFETY: the device is live and `pool_info` valid.
        let pool = unsafe { raw.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the device is live and enables `timelineSemaphore`.
        match unsafe { raw.create_semaphore(&info, None) } {
            Ok(timeline) => Ok(Made {
                pool,
                buffers: Vec::new(),
                timeline,
                signalled: 0,
            }),
            Err(error) => {
                // SAFETY: nothing allocated from the pool is in use.
                unsafe { release::object(raw, pool) };
                Err(vulkan(error))
            }
        }
    }
}

/// The staging queue's lock, or the fill queue's, held to record and make a submission (the
/// module docs).
pub(crate) struct StagingLock<'a> {
    device: &'a Device,
    staging: &'a StagingQueue,
    objects: MutexGuard<'a, Objects>,
}

impl StagingLock<'_> {
    /// Record `record` into a free command buffer of the staging queue and submit it there,
    /// waiting on the GPU for what the device's other `VkQueue`s completed as the host saw (the
    /// module docs); the value it signals, not waited for ([`Device::wait_staging`]). `record`
    /// runs outside a render pass, for the staging queue's family (of transfers alone, or of
    /// graphics or compute work), whose queue flags it is given: it records copies, fills and
    /// barriers, of color aspects alone.
    pub(crate) fn submit(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer, vk::QueueFlags),
    ) -> Result<u64, Error> {
        self.submit_after(&[], record)
    }

    /// As [`submit`](Self::submit), waiting on the GPU besides for each of `after`, timelines and
    /// values whose signals were submitted: the fill queue's submission, after the work before
    /// a batch on its `VkQueue` and what the batch waits for (the module docs).
    pub(crate) fn submit_after(
        &mut self,
        after: &[(vk::Semaphore, u64)],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer, vk::QueueFlags),
    ) -> Result<u64, Error> {
        let device = self.device;
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let shared = device.shared();
        let raw = shared.vk();
        let staging = self.staging;
        let objects = &mut *self.objects;
        if objects.made.is_none() {
            let made = Made::new(device, staging.family)?;
            let _ = staging.timeline.set(made.timeline);
            staging.done.made(made.timeline);
            objects.made = Some(made);
        }
        let queue = objects.queue;
        let Some(made) = objects.made.as_mut() else {
            unreachable!("made above");
        };
        let at = free_buffer(device, staging, made)?;
        let buffer = made.buffers[at].0;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is not pending (the host saw the value of its last
        // submission reached) and the staging lock is held; beginning it resets it (the pool
        // resets command buffers).
        unsafe { raw.begin_command_buffer(buffer, &begin) }.map_err(vulkan)?;
        shared.counts.command_buffer_begun(Begun::Internal);
        record(raw, buffer, staging.flags);
        // SAFETY: as above.
        unsafe { raw.end_command_buffer(buffer) }.map_err(vulkan)?;
        // The completed work of every other queue: a wait that costs nothing, for visibility;
        // then `after`, one wait a timeline, at the highest value asked.
        let mut completed: Vec<(vk::Semaphore, u64)> = Vec::new();
        let asked = (shared.queues.iter())
            .map(|slot| (slot.timeline, slot.completed_value()))
            .chain(after.iter().copied())
            .filter(|&(semaphore, value)| value > 0 && semaphore != made.timeline);
        for (semaphore, value) in asked {
            match completed.iter_mut().find(|(at, _)| *at == semaphore) {
                Some(wait) => wait.1 = wait.1.max(value),
                None => completed.push((semaphore, value)),
            }
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let completed: Vec<(vk::Semaphore, u64)> =
            completed.into_iter().chain(*lock(&staging.held)).collect();
        let wait_semaphores: Vec<vk::Semaphore> = completed.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = completed.iter().map(|w| w.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; completed.len()];
        // It signals its own timeline, which a host call or the device's next submissions wait
        // for.
        let value = made.signalled + 1;
        let signal_semaphores = [made.timeline];
        let signal_values = [value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let buffers = [buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        // What the submission carries, to its value: the device's pending audit's
        // (`pending_audit.rs`), where it is on.
        let carried = (made.timeline, value);
        if let Some(audit) = raw.pending_audit() {
            audit.submitted(raw, carried, &[&buffers[..]]);
        }
        // SAFETY: the queue is used under the staging lock alone; the command buffer is
        // executable; every value waited for was signalled (the host saw it); the signal of the
        // timeline is above its value.
        shared.counts.queue_submitted(1, buffers.len());
        shared.queue_quiet.began();
        let submitted = unsafe {
            staging.word.restore();
            raw.queue_submit(queue, &[submit], vk::Fence::null())
        };
        shared.queue_quiet.ended();
        exit_guard::called(Call::QueueSubmit);
        if let (Err(_), Some(audit)) = (&submitted, raw.pending_audit()) {
            audit.withdrawn(carried, &[&buffers[..]]);
        }
        submitted.map_err(|error| device.failed(error))?;
        made.signalled = value;
        made.buffers[at].1 = value;
        staging.submitted.store(value, Ordering::Release);
        match staging.role {
            Role::Staging => shared.counts.staging_submission(),
            Role::Fills => shared.counts.fill_queue_submission(),
        }
        Ok(value)
    }
}

/// The index of a command buffer of `made`, `staging`'s, no submission holds: one whose last
/// submission the host saw complete, else a new one while there are fewer than [`BUFFERS`], else
/// the oldest, once a wait on the host saw its value reached.
fn free_buffer(device: &Device, staging: &StagingQueue, made: &mut Made) -> Result<usize, Error> {
    let seen = staging.done.value();
    let oldest = (made.buffers.iter().enumerate())
        .min_by_key(|(_, &(_, value))| value)
        .map(|(at, &(_, value))| (at, value));
    if let Some((at, value)) = oldest {
        if value <= seen || reached(device, staging, value, 0)? {
            return Ok(at);
        }
    }
    if made.buffers.len() < BUFFERS {
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(made.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffer = device.allocate_command_buffers(&allocate)?[0];
        made.buffers.push((buffer, 0));
        return Ok(made.buffers.len() - 1);
    }
    let Some((at, value)) = oldest else {
        unreachable!("BUFFERS command buffers are made");
    };
    wait_for(device, staging, value)?;
    Ok(at)
}

/// Whether the host saw `staging`'s timeline, made, reach `value` within `timeout` nanoseconds
/// (0: at once, "If timeout is zero, then vkWaitSemaphores does not wait"); its completion noted
/// where it did. A wait, not a read: Vulkan orders host operations after a signal only through a
/// wait (`device/submit.rs`, `binding_seen`).
fn reached(
    device: &Device,
    staging: &StagingQueue,
    value: u64,
    timeout: u64,
) -> Result<bool, Error> {
    let shared = device.shared();
    let timeline = *(staging.timeline.get())
        .ok_or_else(|| internal("a wait for a staging or fill-queue submission before any"))?;
    let semaphores = [timeline];
    let values = [value];
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    // SAFETY: the timeline is the device's, live while the device is.
    let waited = unsafe { shared.vk().wait_semaphores(&info, timeout) };
    exit_guard::called(Call::WaitSemaphores);
    match waited {
        Ok(()) => {
            staging.done.reached(value);
            Ok(true)
        }
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(error) => Err(device.failed(error)),
    }
}

/// Wait on the host until `staging`'s timeline reached `value`, a slice at a time.
fn wait_for(device: &Device, staging: &StagingQueue, value: u64) -> Result<(), Error> {
    let slice = WAIT_SLICE.as_nanos() as u64;
    while !reached(device, staging, value, slice)? {
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
    }
    Ok(())
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Device {
    /// Have each staging submission made from now on wait on the GPU, besides, for `hold`'s
    /// timeline to reach its value (`None`: no more), which the test signals from the host: the
    /// tests' way to see batches submitted while a staging submission has not completed, whatever
    /// the host's timing. No-op on a device without a staging queue.
    pub(crate) fn hold_staging_submissions(&self, hold: Option<(vk::Semaphore, u64)>) {
        if let Some(staging) = self.shared().staging.as_ref() {
            *lock(&staging.held) = hold;
        }
    }

    /// The fill queue's family and that family's queue flags, where the device has one.
    pub(crate) fn fill_queue_family(&self) -> Option<(u32, vk::QueueFlags)> {
        (self.shared().fill_queue.as_ref()).map(|queue| (queue.family, queue.flags))
    }
}

impl Device {
    /// Whether the device reserved a staging queue (the module docs): a device it created with a
    /// family of transfers alone whose copies take any texel, or, where no memory the host caches
    /// holds a texture's image, with such a family of graphics or compute work beside its own.
    pub(crate) fn has_staging_queue(&self) -> bool {
        self.shared().staging.is_some()
    }

    /// Whether the device has a fill queue (the module docs) with a command buffer a submission
    /// may record into now, with no wait on the host: one whose last submission the host saw
    /// complete, or room for another of [`BUFFERS`]. A submission there may wait for work that
    /// waits for the caller (a signal it reserved, `Event::reserve_vk_signal`), so the batch that
    /// finds none keeps its fills at its start rather than wait under the scheduler's lock.
    pub(crate) fn fill_queue_has_room(&self) -> bool {
        let Some(fill_queue) = self.shared().fill_queue.as_ref() else {
            return false;
        };
        let objects = lock(&fill_queue.objects);
        let Some(made) = objects.made.as_ref() else {
            return true;
        };
        let seen = fill_queue.done.value();
        made.buffers.len() < BUFFERS
            || (made.buffers.iter()).any(|&(_, value)| {
                value <= seen || reached(self, fill_queue, value, 0).unwrap_or(false)
            })
    }

    /// The sharing of an image that copies on the staging queue: concurrent among the device's
    /// families and the staging queue's; `None` where the device has none.
    pub(crate) fn staging_sharing(&self) -> Option<Sharing<'_>> {
        (self.shared().staging.as_ref()).map(|staging| Sharing::new(&staging.families))
    }

    /// The staging queue's lock, for a submission there ([`StagingLock::submit`]).
    pub(crate) fn staging_lock(&self) -> Result<StagingLock<'_>, Error> {
        let staging = (self.shared().staging.as_ref())
            .ok_or_else(|| internal("a staging copy on a device without a staging queue"))?;
        Ok(StagingLock {
            device: self,
            staging,
            objects: lock(&staging.objects),
        })
    }

    /// The fill queue's lock, for a submission there ([`StagingLock::submit_after`]); `None`
    /// where the device has no fill queue (the module docs).
    pub(crate) fn fill_queue_lock(&self) -> Option<StagingLock<'_>> {
        let fill_queue = self.shared().fill_queue.as_ref()?;
        Some(StagingLock {
            device: self,
            staging: fill_queue,
            objects: lock(&fill_queue.objects),
        })
    }

    /// Wait on the host for the staging submission that signals `value`.
    pub(crate) fn wait_staging(&self, value: u64) -> Result<(), Error> {
        let staging = (self.shared().staging.as_ref())
            .ok_or_else(|| internal("a staging wait on a device without a staging queue"))?;
        wait_on(self, staging, value)
    }

    /// Wait on the host for the fill-queue submission that signals `value`.
    pub(crate) fn wait_fill_queue(&self, value: u64) -> Result<(), Error> {
        let fill_queue = (self.shared().fill_queue.as_ref())
            .ok_or_else(|| internal("a fill-queue wait on a device without a fill queue"))?;
        wait_on(self, fill_queue, value)
    }

    /// The staging queue's last submission while the host has not seen it complete: what every
    /// other submission of the device's waits for on the GPU, or follows on its `VkQueue` a wait
    /// for (the module docs), as what it copied may be among what it uses. The host sees it
    /// complete once a wait of no time returned for it.
    pub(crate) fn staging_wait(&self) -> Option<(vk::Semaphore, u64)> {
        pending(self, self.shared().staging.as_ref()?)
    }

    /// The fill queue's last submission while the host has not seen it complete, as
    /// [`staging_wait`](Self::staging_wait)'s: what it filled may be among what a submission uses.
    pub(crate) fn fill_queue_wait(&self) -> Option<(vk::Semaphore, u64)> {
        pending(self, self.shared().fill_queue.as_ref()?)
    }
}

/// A batch's fills submitted on the fill queue ([`Device::submit_fills`]): the timeline and value
/// the batch waits for, and what the submission holds until it completed (each buffer's memory,
/// a fill's staging buffer). A batch holds it until it completed; its drop waits on the host for
/// the submission where the host has not seen it complete (a batch that held it completed after
/// it, so that wait returns at once; one whose own submission failed did not).
pub(crate) struct FillSubmission {
    device: Device,
    pub(crate) timeline: vk::Semaphore,
    pub(crate) value: u64,
    _held: Vec<Box<dyn Any + Send + Sync>>,
}

impl Drop for FillSubmission {
    fn drop(&mut self) {
        // A section of the exit guard, wherever the last holder lets go: while the process exits,
        // nothing is waited for (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // A failed wait (a lost device) leaves nothing running that uses what it holds.
        let _ = self.device.wait_fill_queue(self.value);
    }
}

impl Device {
    /// Submit `fills`, a batch's, on the fill queue, waiting on the GPU for each of `after` (the
    /// module docs); `None` where the device has no fill queue, the fills given back.
    pub(crate) fn submit_fills(
        &self,
        fills: Fills<'_>,
        after: &[(vk::Semaphore, u64)],
    ) -> Result<Option<FillSubmission>, Error> {
        let Some(mut fill_queue) = self.fill_queue_lock() else {
            return Ok(None);
        };
        let shared = self.shared();
        let raw = shared.vk();
        let value = fill_queue.submit_after(after, |_, command_buffer, _| {
            // SAFETY: recording for the fill queue's family, outside a render pass: fills of
            // buffers alone, each shared with that family, as their sources are (`sharing.rs`);
            // the batch that uses them waits for the submission.
            unsafe { fills.record((raw, &shared.counts), command_buffer) }
        })?;
        let timeline = *(fill_queue.staging.timeline.get())
            .ok_or_else(|| internal("a fill-queue submission made no timeline"))?;
        drop(fill_queue);
        Ok(Some(FillSubmission {
            device: self.clone(),
            timeline,
            value,
            _held: fills.submitted(),
        }))
    }
}

/// Wait on the host for `staging`'s submission that signals `value`.
fn wait_on(device: &Device, staging: &StagingQueue, value: u64) -> Result<(), Error> {
    if staging.done.value() >= value {
        return Ok(());
    }
    wait_for(device, staging, value)
}

/// `staging`'s last submission, its timeline and value, while the host has not seen it complete.
fn pending(device: &Device, staging: &StagingQueue) -> Option<(vk::Semaphore, u64)> {
    let submitted = staging.submitted.load(Ordering::Acquire);
    if submitted <= staging.done.value() {
        return None;
    }
    let timeline = *staging.timeline.get()?;
    // A failed wait (a lost device) leaves the submission waited for.
    match reached(device, staging, submitted, 0) {
        Ok(true) => None,
        _ => Some((timeline, submitted)),
    }
}
