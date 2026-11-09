//! The device's staging queue: the first queue of a family of transfers alone, a copy engine
//! (`create::queue_plan`), where a `Shared` or `Managed` texture's CPU copies run on a device that
//! copies none on the host from memory the host caches (`texture/host_copy.rs`). An image that
//! copies here is concurrent among the device's families and this one ([`Device::staging_sharing`]),
//! so no image changes owner; every other resource keeps the device's sharing (`sharing.rs`).
//!
//! - **Why a copy engine.** A copy on the device's first `VkQueue` waits behind every command
//!   buffer submitted there before it: behind a 2 s kernel, a 1080p `get_bytes` returned after
//!   1.99 s on an RTX 3090, and forever behind a command buffer waiting on an event only a later
//!   command of the calling thread signals (the consumer's guest hang). Metal's `getBytes` and
//!   `replaceRegion` of shared storage are CPU copies that wait for no GPU work. Another queue of
//!   the graphics family is not enough: on the 3090 (615.71.09) a readback there waited 1.88 s for
//!   a 2 s kernel running on the first, where the copy engine's took 7.8 to 13.5 ms with the
//!   kernel still running, and 2.65 ms idle (raw Vulkan). Here nothing but these
//!   copies runs, so a copy waits for itself and the staging work submitted before it alone.
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

use super::{lock, Device, LoaderWord};
use crate::error::{internal, vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::sharing::Sharing;
use crate::work_counts::Begun;
use ash::vk;
use ash::vk::Handle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// How long one wait for a staging submission lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The most command buffers the staging queue records into: past them, a submission waits on the
/// host for the oldest one's before it records (the module docs).
pub(crate) const BUFFERS: usize = 8;

/// The staging queue and its objects.
pub(super) struct StagingQueue {
    /// The queue's family: one of transfers alone.
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
    /// The staging queue `queue` of `family`, whose queue flags are `flags`, on a device that
    /// submits to `families`.
    ///
    /// # Safety
    ///
    /// `queue` was just returned by the device: its word is the loader's.
    pub(super) unsafe fn new(
        family: u32,
        flags: vk::QueueFlags,
        queue: vk::Queue,
        families: &[u32],
    ) -> StagingQueue {
        StagingQueue {
            family,
            flags,
            families: families.iter().copied().chain([family]).collect(),
            // SAFETY: the caller's contract.
            word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
            objects: Mutex::new(Objects { queue, made: None }),
            timeline: OnceLock::new(),
            submitted: AtomicU64::new(0),
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

/// The staging queue's lock, held to record and make a submission (the module docs).
pub(crate) struct StagingLock<'a> {
    device: &'a Device,
    staging: &'a StagingQueue,
    objects: MutexGuard<'a, Objects>,
}

impl StagingLock<'_> {
    /// Record `record` into a free command buffer of the staging queue and submit it there,
    /// waiting on the GPU for what the device's other `VkQueue`s completed as the host saw (the
    /// module docs); the value it signals, not waited for ([`Device::wait_staging`]). `record`
    /// runs outside a render pass, for a family of transfers alone, whose queue flags it is
    /// given: it records copies, fills and barriers, of color aspects alone.
    pub(crate) fn submit(
        &mut self,
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
            shared.staging_done.made(made.timeline);
            objects.made = Some(made);
        }
        let queue = objects.queue;
        let Some(made) = objects.made.as_mut() else {
            unreachable!("made above");
        };
        let at = free_buffer(device, made)?;
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
        // The completed work of every other queue: a wait that costs nothing, for visibility.
        let completed: Vec<(vk::Semaphore, u64)> = (shared.queues.iter())
            .map(|slot| (slot.timeline, slot.completed_value()))
            .filter(|&(_, value)| value > 0)
            .collect();
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
        shared.counts.staging_submission();
        Ok(value)
    }
}

/// The index of a command buffer of `made` no submission holds: one whose last submission the
/// host saw complete, else a new one while there are fewer than [`BUFFERS`], else the oldest,
/// once a wait on the host saw its value reached.
fn free_buffer(device: &Device, made: &mut Made) -> Result<usize, Error> {
    let shared = device.shared();
    let seen = shared.staging_done.value();
    let oldest = (made.buffers.iter().enumerate())
        .min_by_key(|(_, &(_, value))| value)
        .map(|(at, &(_, value))| (at, value));
    if let Some((at, value)) = oldest {
        if value <= seen || reached(device, made.timeline, value, 0)? {
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
    wait_for(device, made.timeline, value)?;
    Ok(at)
}

/// Whether the host saw `timeline` reach `value` within `timeout` nanoseconds (0: at once, "If
/// timeout is zero, then vkWaitSemaphores does not wait"); the staging timeline's completion
/// noted where it did. A wait, not a read: Vulkan orders host operations after a signal only
/// through a wait (`device/submit.rs`, `binding_seen`).
fn reached(
    device: &Device,
    timeline: vk::Semaphore,
    value: u64,
    timeout: u64,
) -> Result<bool, Error> {
    let shared = device.shared();
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
            shared.staging_done.reached(value);
            Ok(true)
        }
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(error) => Err(device.failed(error)),
    }
}

/// Wait on the host until `timeline` reached `value`, a slice at a time.
fn wait_for(device: &Device, timeline: vk::Semaphore, value: u64) -> Result<(), Error> {
    let slice = WAIT_SLICE.as_nanos() as u64;
    while !reached(device, timeline, value, slice)? {
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
}

impl Device {
    /// Whether the device reserved a staging queue (the module docs): a device it created with a
    /// family of transfers alone whose copies take any texel.
    pub(crate) fn has_staging_queue(&self) -> bool {
        self.shared().staging.is_some()
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

    /// Wait on the host for the staging submission that signals `value`.
    pub(crate) fn wait_staging(&self, value: u64) -> Result<(), Error> {
        if self.shared().staging_done.value() >= value {
            return Ok(());
        }
        let staging = (self.shared().staging.as_ref())
            .ok_or_else(|| internal("a staging wait on a device without a staging queue"))?;
        let timeline = *(staging.timeline.get())
            .ok_or_else(|| internal("a staging wait before any staging submission"))?;
        wait_for(self, timeline, value)
    }

    /// The staging queue's last submission while the host has not seen it complete: what every
    /// other submission of the device's waits for on the GPU, or follows on its `VkQueue` a wait
    /// for (the module docs), as what it copied may be among what it uses. The host sees it
    /// complete once a wait of no time returned for it.
    pub(crate) fn staging_wait(&self) -> Option<(vk::Semaphore, u64)> {
        let shared = self.shared();
        let staging = shared.staging.as_ref()?;
        let submitted = staging.submitted.load(Ordering::Acquire);
        if submitted <= shared.staging_done.value() {
            return None;
        }
        let timeline = *staging.timeline.get()?;
        // A failed wait (a lost device) leaves the submission waited for.
        match reached(self, timeline, submitted, 0) {
            Ok(true) => None,
            _ => Some((timeline, submitted)),
        }
    }
}
