//! MoltenVK's command-buffer failures, relayed to the command buffer that failed.
//!
//! **Metal.** A GPU fault fails the command buffer it struck, and the queue runs on: our probe (an
//! M4 with macOS 27.0.1, plain Metal) measured a blit into a buffer over
//! read-only pages failing with `MTLCommandBufferErrorInternal` (1, "Discarded (victim of GPU
//! error/recovery)"), as did the command buffer committed after it while it ran, the later ones
//! completing; an execution range past its buffer failing with `PageFault` (3), the next command
//! buffer completing. The consumer measured the blit's `PageFault` on an M3 Max with 26.7.
//!
//! **MoltenVK 1.4.1** runs each `VkSubmitInfo` as a Metal command buffer, in order on one Metal
//! queue per `VkQueue`. When one fails, its completed handler reports it to the instance's
//! debug-utils messengers, naming the `VkQueue` (`"<result>: Lost|Resumed VkDevice after
//! MTLCommandBuffer \"<label>\" execution failed (code <n>): <description>"`,
//! `handleMTLCommandBufferError` in `MVKQueue.mm`), before the handler that signals the
//! submission's fence and inline semaphores. By default it then marks the device lost, so every
//! later submission fails, where Metal fails that command buffer alone: an instance the runtime
//! creates with the relay asks it not to (`MVK_CONFIG_RESUME_LOST_DEVICE`, through
//! `VK_EXT_layer_settings` ahead of the consumer's own settings, which win; [`setting`]). It
//! marks the device lost still for the codes that lose it (`Blacklisted`, `NotPermitted`,
//! `DeviceRemoved`), and the runtime sees the loss as before.
//!
//! **The relay.** Each `VkQueue` of a device on MoltenVK whose instance has the relay's messenger
//! keeps its submissions in flight in submission order ([`Flights`]), each with a fence its
//! submission signals: a batch of a command buffer's, which there always takes one
//! (`command_buffer/submit.rs`), and, after any other use of the queue (the program's through
//! `CommandQueue::with_vk_queue`, or the runtime's own outside a batch), a submission of no work
//! with one, which ends where that use's submissions end. Metal runs one queue's completed
//! handlers in order, on one thread (the probe's `handlers` and `handlers_fault`: 600
//! handlers, failures among them, none out of order), so when MoltenVK reports a failure every
//! submission before the failed one has its fence signaled, and the failed one does not: the
//! relay gives the first submission whose fence is unsignaled the failure's code, with Metal's
//! description of it, the message's text after the code (`localizedDescription`; the consumer's
//! failures came with the code alone). A command buffer whose batch it was finishes with that error
//! once its fences are signaled (`CommandBufferShared::complete`: `PageFault` for 3, `DeviceLost`
//! for 11, else `CommandBufferError::Execution`), its description kept
//! (`CommandBuffer::error_description`); a failure of the program's own submission is the program's
//! to read (its own messenger hears it); and one of the runtime's own, or one no submission in
//! flight explains (a Metal command buffer MoltenVK made for itself), loses the device: every
//! batch in flight then takes the loss's code (`DeviceRemoved`, 11), and no later one is
//! submitted. So does every batch where no fence can be read, the device lost before the report
//! (MoltenVK under the program's own `MVK_CONFIG_RESUME_LOST_DEVICE` off). Where the device's
//! line recovers (`device/line.rs`), its command buffers answer as the recovery does: where the
//! loss failed a wait for their fences, the command buffer of the oldest incomplete batch as the
//! hang (`GpuHang`) and the rest as its victims (`command_buffer/completion.rs`), the loss's code
//! leaving that answer as it is; where their fences signaled, each as a victim
//! (`GpuRecoveryVictim`).
//!
//! **The loader word.** MoltenVK writes `ICD_LOADER_MAGIC` over the first word of an object it
//! names in a message (`loader_word.rs`), the queue's here, where a layer between the loader and
//! the driver reads the queue's dispatch table (the validation layer then ends the process at
//! the next submission). The queue's word is put back before each use of
//! the queue (`QueueSlot::with`) and as the relay hears a message naming it; a submission on
//! another thread in between still reads the magic.
//!
//! **Not relayed.** An adopted device (its instance is the consumer's, with MoltenVK's default:
//! the device is lost), and a device whose instance lacks `VK_EXT_debug_utils` (the same).

use super::functions::Functions;
use super::submit_stalls::{Caller, Uses};
use super::{lock, SlotHold};
use crate::exit_guard::{self, Call};
use crate::loader_word::LoaderWord;
use ash::vk::{self, Handle};
use std::collections::VecDeque;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The longest wait for a kept mark's fence inside one section of the exit guard.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// MoltenVK's setting that keeps a device after a command buffer fails (`MVKConfigMembers.def`:
/// `resumeLostDevice`, a `VkBool32`).
const RESUME: &CStr = c"MVK_CONFIG_RESUME_LOST_DEVICE";

/// The layer setting an instance with the relay asks for, ahead of the consumer's own (the
/// module docs).
pub(super) fn setting() -> super::create::LayerSetting {
    super::create::LayerSetting::new(
        super::argument_buffers::LAYER.to_owned(),
        RESUME.to_owned(),
        super::create::LayerSettingValue::Bool(true),
    )
}

/// `MTLCommandBufferErrorDeviceRemoved`: the code a device lost by the relay gives its command
/// buffers in flight.
pub(crate) const DEVICE_REMOVED: u64 = 11;

/// The relay's messenger on an instance the runtime created, and the relay it reports to.
pub(super) struct Messenger {
    loader: ash::ext::debug_utils::Instance,
    handle: vk::DebugUtilsMessengerEXT,
    /// The callback's user data; it outlives the messenger.
    relay: Box<Relay>,
}

impl Messenger {
    /// The relay's messenger on `instance`, of warnings and errors.
    ///
    /// # Safety
    ///
    /// `instance` is live, from `entry`, and enabled `VK_EXT_debug_utils`.
    pub(super) unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Messenger, vk::Result> {
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        use vk::DebugUtilsMessageTypeFlagsEXT as T;
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let relay = Box::<Relay>::default();
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(S::WARNING | S::ERROR)
            .message_type(T::GENERAL | T::VALIDATION | T::PERFORMANCE)
            .pfn_user_callback(Some(deliver))
            .user_data(std::ptr::from_ref::<Relay>(&relay).cast_mut().cast());
        // SAFETY: the caller's; the relay lives in the messenger.
        let handle = unsafe { loader.create_debug_utils_messenger(&info, None) }?;
        Ok(Messenger {
            loader,
            handle,
            relay,
        })
    }

    /// The relay it reports to.
    pub(super) fn relay(&self) -> &Relay {
        &self.relay
    }

    /// Destroy the messenger; the instance is destroyed next.
    pub(super) fn destroy(&self) {
        // SAFETY: the messenger belongs to the live instance and is destroyed once.
        unsafe { crate::release::messenger(&self.loader, self.handle) };
    }
}

/// What an instance's relay messenger reports to: the queues of the instance's devices whose
/// failures it relays.
#[derive(Default)]
pub(crate) struct Relay {
    queues: Mutex<Vec<Arc<Flights>>>,
}

impl Relay {
    /// Relay the failures of `flights`' queue.
    pub(super) fn watch(&self, flights: Arc<Flights>) {
        lock(&self.queues).push(flights);
    }

    /// Stop relaying the failures of the device whose loss `lost` records: it is torn down, and
    /// nothing of it is in flight.
    pub(super) fn forget(&self, lost: &Arc<AtomicBool>) {
        lock(&self.queues).retain(|flights| !Arc::ptr_eq(&flights.lost, lost));
    }

    /// A message the driver reported naming `queue`, of `text`.
    fn heard(&self, queue: vk::Queue, text: &str) {
        let queues = lock(&self.queues);
        let Some(flights) = queues.iter().find(|flights| flights.queue == queue) else {
            return;
        };
        // SAFETY: the queue is live while its device is relayed (`forget` runs first at its
        // teardown, under this lock).
        unsafe { flights.word.restore() };
        let Some((code, description)) = failure(text) else {
            return;
        };
        // Once the process is exiting, the device is left alone.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        match flights.first_pending() {
            Some(By::Batch(failed)) => failed.fail(code, Some(description)),
            Some(By::Program) => {}
            Some(By::Runtime | By::Checks) | None => {
                flights.lost.store(true, Ordering::Release);
                for each in queues
                    .iter()
                    .filter(|f| Arc::ptr_eq(&f.lost, &flights.lost))
                {
                    each.lose_all();
                }
            }
        }
    }
}

/// The `MTLCommandBufferError` code of a MoltenVK message reporting a failed Metal command buffer
/// (the module docs' format), with Metal's description of it, if it is one.
pub(crate) fn failure(text: &str) -> Option<(u64, &str)> {
    let (_, after) = text.split_once("VkDevice after MTLCommandBuffer ")?;
    let (_, code) = after.split_once(" execution failed (code ")?;
    let (digits, rest) = code.split_once(')')?;
    let code = digits.parse().ok().filter(|&code| code != 0)?;
    let description = rest.strip_prefix(':').unwrap_or(rest).trim();
    Some((code, description))
}

/// A command buffer's failure the driver relayed for one of its batches: Metal's
/// `MTLCommandBufferError` code and its description, the first one kept.
#[derive(Default)]
pub(crate) struct Relayed(Mutex<Option<(u64, Option<String>)>>);

impl Relayed {
    /// Keep `code` (not 0) and `description`, unless a failure was kept already.
    fn fail(&self, code: u64, description: Option<&str>) {
        let mut failure = lock(&self.0);
        if failure.is_none() {
            *failure = Some((code, description.map(str::to_owned)));
        }
    }

    /// The device was lost: its loss's code (`DeviceRemoved`), unless a failure was kept already
    /// (the module docs).
    pub(crate) fn lost(&self) {
        self.fail(DEVICE_REMOVED, None);
    }

    /// The code and the description kept, if a failure was relayed.
    pub(crate) fn failure(&self) -> Option<(u64, Option<String>)> {
        lock(&self.0).clone()
    }
}

/// Whose use of a queue a submission in flight is.
#[derive(Clone)]
pub(crate) enum By {
    /// A batch of a command buffer's: its failure goes to the cell, empty until one.
    Batch(Arc<Relayed>),
    /// The program's own use (`CommandQueue::with_vk_queue`).
    Program,
    /// The runtime's own use outside a batch.
    Runtime,
    /// The device's submission checks' use (`submit_check.rs`): the runtime's own, its mark
    /// counted with the checks' calls (`WorkCounts::submit_check_calls`), not the device's.
    Checks,
}

/// A submission in flight: the fence it signals, whose it is, whether its call returned, and
/// whether its use's caller keeps it.
struct Flight {
    fence: vk::Fence,
    by: By,
    /// Whether the `vkQueueSubmit` that signals the fence returned: until it did, the fence is the
    /// submitting thread's (a submission's fence is externally synchronized, and a driver may
    /// signal it before the call returns), so no other thread reads, resets or gives it back.
    made: bool,
    /// Whether the mark is kept for the caller of its use, which may wait for its fence
    /// ([`Mark`]): until it lets go, no other thread resets or gives the fence back.
    kept: bool,
}

/// One `VkQueue`'s submissions in flight, in submission order (the module docs).
pub(crate) struct Flights {
    queue: vk::Queue,
    /// The queue's loader word (the module docs).
    word: LoaderWord,
    /// The device's functions, for the fences' state, through their checked door
    /// (`functions.rs`): live while the device is relayed.
    functions: Functions,
    /// The device's loss, shared with it.
    lost: Arc<AtomicBool>,
    log: Mutex<VecDeque<Flight>>,
}

impl Flights {
    /// The flights of `queue` of `device`, whose loss `lost` records.
    ///
    /// # Safety
    ///
    /// `queue` is a live queue of `device` whose first word is still the loader's.
    pub(super) unsafe fn new(
        queue: vk::Queue,
        functions: Functions,
        lost: Arc<AtomicBool>,
    ) -> Flights {
        Flights {
            queue,
            // SAFETY: the caller's.
            word: unsafe { LoaderWord::of_handle(queue.as_raw()) },
            functions,
            lost,
            log: Mutex::default(),
        }
    }

    /// A submission of `fence`, `by`'s, is made next, under the queue's lock: until it is
    /// [`made`](Self::made), the log holds it pending and reads nothing of its fence.
    pub(crate) fn push(&self, fence: vk::Fence, by: By) {
        lock(&self.log).push_back(Flight {
            fence,
            by,
            made: false,
            kept: false,
        });
    }

    /// The mark of `fence` pushed last was [`made`](Self::made), and is kept for its use's
    /// caller ([`Mark`]) from the same moment, so no other use takes it in between.
    fn made_kept(&self, fence: vk::Fence) {
        let mut log = lock(&self.log);
        if let Some(flight) = log.iter_mut().rev().find(|flight| flight.fence == fence) {
            flight.made = true;
            flight.kept = true;
        }
    }

    /// The caller of the kept mark of `fence` lets go of it: the next use takes it once done.
    fn let_go_of_kept(&self, fence: vk::Fence) {
        let mut log = lock(&self.log);
        if let Some(flight) = log.iter_mut().find(|flight| flight.fence == fence) {
            flight.kept = false;
        }
    }

    /// The submission of `fence` pushed last was made: its call returned, and its fence may be
    /// read.
    pub(crate) fn made(&self, fence: vk::Fence) {
        let mut log = lock(&self.log);
        if let Some(flight) = log.iter_mut().rev().find(|flight| flight.fence == fence) {
            flight.made = true;
        }
    }

    /// The submission of `fence` pushed last was not made.
    pub(super) fn unpush(&self, fence: vk::Fence) {
        let mut log = lock(&self.log);
        if log.back().is_some_and(|flight| flight.fence == fence) {
            log.pop_back();
        }
    }

    /// The submissions through the one of the last of `fences` completed (their fences are
    /// signaled, and every earlier submission's): they leave the log, but for the marks kept for
    /// their uses' callers ([`Mark`]), which stay at its front, signaled, until let go. The fences
    /// of the others not of a batch go onto `marks`, for the caller to reset and give back (a
    /// batch's goes back with it).
    pub(crate) fn settled(&self, fences: &[vk::Fence], marks: &mut Vec<vk::Fence>) {
        let mut log = lock(&self.log);
        let Some(through) = log
            .iter()
            .rposition(|flight| fences.contains(&flight.fence))
        else {
            return;
        };
        let mut kept = Vec::new();
        for flight in log.drain(..=through) {
            match (&flight.by, flight.kept) {
                (_, true) => kept.push(flight),
                (By::Batch(_), false) => {}
                (_, false) => marks.push(flight.fence),
            }
        }
        for flight in kept.into_iter().rev() {
            log.push_front(flight);
        }
    }

    /// How many of the queue's submissions are in the log: made, and not yet seen complete.
    pub(crate) fn in_flight(&self) -> usize {
        lock(&self.log).len()
    }

    /// The batch of `fence` is let go of after a failed wait: it leaves the log, so its fence,
    /// which may be released, is never read.
    pub(crate) fn let_go(&self, fence: vk::Fence) {
        lock(&self.log).retain(|flight| flight.fence != fence);
    }

    /// The first submissions not of a batch whose calls returned, which no use's caller keeps and
    /// whose fences are signaled, each before any other in the log: they leave it, and their
    /// fences are for the caller to reset and give back. Inside a section of the exit guard.
    pub(crate) fn done_marks(&self) -> Vec<vk::Fence> {
        let mut log = lock(&self.log);
        let mut done = Vec::new();
        while let Some(front) = log.front() {
            if matches!(front.by, By::Batch(_))
                || !front.made
                || front.kept
                || !self.signaled(front.fence)
            {
                break;
            }
            done.extend(log.pop_front().map(|flight| flight.fence));
        }
        done
    }

    /// Every fence of the log, which leaves it: the device is torn down, idle.
    pub(super) fn drain(&self) -> Vec<vk::Fence> {
        lock(&self.log)
            .drain(..)
            .map(|flight| flight.fence)
            .collect()
    }

    /// Whose the first submission is whose fence is unsignaled, or whose call has not returned
    /// (its fence unread: pending); `None` when every fence is signaled or one cannot be read.
    /// Inside a section of the exit guard.
    fn first_pending(&self) -> Option<By> {
        let log = lock(&self.log);
        for flight in log.iter() {
            if !flight.made {
                return Some(flight.by.clone());
            }
            // SAFETY: the fence is the device's and in the log, so not yet released.
            let status = unsafe {
                self.functions
                    .get()
                    .wait_for_fences(&[flight.fence], true, 0)
            };
            exit_guard::called(Call::WaitForFences);
            match status {
                Ok(()) => continue,
                Err(vk::Result::TIMEOUT) => return Some(flight.by.clone()),
                Err(_) => return None,
            }
        }
        None
    }

    /// Whether `fence` of the log is signaled. Inside a section of the exit guard.
    fn signaled(&self, fence: vk::Fence) -> bool {
        // SAFETY: as `first_pending`'s.
        let status = unsafe { self.functions.get().wait_for_fences(&[fence], true, 0) };
        exit_guard::called(Call::WaitForFences);
        status.is_ok()
    }

    /// Every batch in flight takes the device's loss ([`Relayed::lost`]), unless it failed
    /// already.
    fn lose_all(&self) {
        for flight in lock(&self.log).iter() {
            if let By::Batch(failed) = &flight.by {
                failed.lost();
            }
        }
    }
}

impl super::DeviceShared {
    /// The submissions in flight of `VkQueue` `queue`, where its failures are relayed.
    pub(crate) fn flights(&self, queue: usize) -> Option<&Arc<Flights>> {
        self.queues[queue].flights.get()
    }

    /// Each `VkQueue` whose failures are relayed, with its submissions in flight.
    pub(crate) fn relayed_queues(&self) -> impl Iterator<Item = (usize, &Arc<Flights>)> {
        self.queues
            .iter()
            .enumerate()
            .filter_map(|(queue, slot)| slot.flights.get().map(|flights| (queue, flights)))
    }

    /// Submit a batch of a command buffer's to `VkQueue` `queue` with `f`, under the queue's
    /// lock: where its failures are relayed, the submission of `fence` is logged first, its
    /// failure for `failed` ([`By::Batch`]), made once `f` returns ([`Flights::made`]), and
    /// dropped from the log again when `f` fails. A
    /// batch that signals the queue's timeline to `value` notes it once submitted (the exit
    /// hook's value, and where freed memory waits for the batches in flight, theirs, under the
    /// gate's read lock: `freed_memory.rs`). The time `f` takes, the driver's, is counted
    /// (`WorkCounts::batch_submit_nanos`), and the call counted in the queue's place in its
    /// driver's period of stalls (`submit_stalls.rs`): `f` makes one signalling call.
    pub(crate) fn submit_logged(
        &self,
        (queue, value): (usize, Option<u64>),
        fence: Option<vk::Fence>,
        failed: &Arc<Relayed>,
        f: impl FnOnce(vk::Queue) -> Result<(), vk::Result>,
    ) -> Result<(), vk::Result> {
        let slot = &self.queues[queue];
        let logged = fence.zip(slot.flights.get());
        let parking = self.parking.get();
        let (submitted, behind) = slot.with_counted(Caller::Work, |vk_queue, period| {
            let _gate = parking.map(|parking| parking.submitting());
            if let Some((fence, flights)) = logged {
                flights.push(fence, By::Batch(Arc::clone(failed)));
            }
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            crate::command_buffer::completion::trace::submitting("submit called");
            let cpu = period.cpu_now();
            let entered = crate::gpu_time::host_nanos();
            let submitted = f(vk_queue);
            let end = crate::gpu_time::host_nanos();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            crate::command_buffer::completion::trace::submitting("submit returned");
            let nanos = end.saturating_sub(entered);
            (self.counts).batch_submitted_in(nanos);
            let took = period.took(nanos, cpu);
            if submitted.is_ok() && period.called(Caller::Work, took, end) {
                self.counts.submit_stall_on_work();
            }
            match (&submitted, logged) {
                (Err(_), Some((fence, flights))) => flights.unpush(fence),
                (Ok(()), Some((fence, flights))) => flights.made(fence),
                _ => {}
            }
            if let (Ok(()), Some(value)) = (&submitted, value) {
                slot.submitted(value);
                if let Some(parking) = parking {
                    parking.submitted(queue, value);
                }
            }
            submitted
        });
        if let Some(waited) = behind {
            self.counts.submit_behind_idle_stall(waited);
        }
        submitted
    }

    /// `f` with `VkQueue` `queue` under its lock: a use outside a batch, `by`'s. Where the
    /// queue's failures are relayed, a submission of no work after `f`'s, under the same lock,
    /// signals a fence logged as `by`'s, which ends where `f`'s submissions end (the module
    /// docs), made once its call returns ([`Flights::made`]); the marks before it whose fences
    /// are signaled go back first. The mark is counted among the device's submissions
    /// (`WorkCounts::queue_submissions`), or the submission checks' for theirs
    /// ([`By::Checks`]). Where freed memory
    /// waits for the work in flight, the program's use takes a value of the queue's timeline of
    /// such marks before `f` runs, which that submission signals too (`freed_memory.rs`); should
    /// the submission fail, the value counts as reached once the queue is idle. Each reach of the
    /// device is a section of the exit guard of its own (`f` runs outside one: it may be the
    /// program's); once the process is exiting no mark is made. A use of `uses`
    /// [`Uses::OneSignalling`] is counted in the queue's place in its driver's period of stalls
    /// (`submit_stalls.rs`); any other, or one with a mark, forgets the place.
    pub(crate) fn use_queue<R>(
        &self,
        queue: usize,
        by: By,
        uses: Uses,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> R {
        self.queue_uses().use_queue(queue, by, uses, f)
    }

    /// What its uses of a queue outside a batch reach ([`QueueUses`]), borrowed from it.
    pub(crate) fn queue_uses(&self) -> QueueUses<'_> {
        QueueUses {
            queues: &self.queues,
            parking: self.parking.get().map(|parking| &**parking),
            counts: &self.counts,
            fences: &self.fences,
            functions: &self.functions,
        }
    }
}

/// What a use of a device's `VkQueue` outside a batch reaches ([`QueueUses::use_queue`]): its
/// queues, its parking, its counts, its fences and its functions. Borrowed from the device
/// ([`DeviceShared::queue_uses`](super::DeviceShared::queue_uses)), or from the shares of them
/// the queue-switch check holds, which hold no device (`queue_switch_check.rs`).
pub(crate) struct QueueUses<'a> {
    pub(crate) queues: &'a [super::QueueSlot],
    pub(crate) parking: Option<&'a crate::freed_memory::Parking>,
    pub(crate) counts: &'a crate::work_counts::Counters,
    pub(crate) fences: &'a crate::command_buffer::fences::Fences,
    pub(crate) functions: &'a Functions,
}

impl<'a> QueueUses<'a> {
    /// [`DeviceShared::use_queue`](super::DeviceShared::use_queue), over what it reaches.
    pub(crate) fn use_queue<R>(
        &self,
        queue: usize,
        by: By,
        uses: Uses,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> R {
        self.use_queue_kept(queue, by, uses, false, f).0
    }

    /// [`use_queue`](Self::use_queue), its mark, where `keep` asks and one was made, kept for
    /// the caller until the [`Mark`] drops: a caller that holds the device's loader word through
    /// its use's completion waits for the mark too (`loader_word.rs`), the use's last submission
    /// on the queue, whose Metal command buffer MoltenVK names the device for as it completes.
    pub(crate) fn use_queue_kept<R>(
        &self,
        queue: usize,
        by: By,
        uses: Uses,
        keep: bool,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> (R, Option<Mark<'a>>) {
        let fence = self.mark_fence(queue);
        let mut hold = self.queues[queue].hold();
        self.use_held(&mut hold, (queue, by), (uses, keep), fence, f)
    }

    /// [`use_queue_kept`](Self::use_queue_kept) for each use `f` makes through the [`Together`]
    /// it is given, each `by`'s, with each of `queues` (each once, in index order) under its lock
    /// from before `f` begins until after it returns: no other use of those queues comes between
    /// two of `f`'s, nor between its last and what it does after it. The queue-switch check
    /// submits each round so, its batches waiting for a gate it then signals from the host
    /// (`queue_switch_check.rs`).
    pub(crate) fn use_queues_kept<T>(
        &self,
        queues: &[usize],
        by: By,
        f: impl FnOnce(&mut Together<'_, 'a>) -> T,
    ) -> T {
        debug_assert!(
            queues.windows(2).all(|pair| pair[0] < pair[1]),
            "the queues are held each once, in index order"
        );
        let queue_slots: &'a [super::QueueSlot] = self.queues;
        let holds = (queues.iter())
            .map(|&queue| (queue, queue_slots[queue].hold()))
            .collect();
        f(&mut Together {
            uses: self,
            by,
            holds,
        })
    }

    /// The fence of the relay's mark after a use of `queue`, where its failures are relayed
    /// (none elsewhere, or once the process is exiting), taken before the use takes the queue's
    /// lock: the marks before it whose fences are signaled go back first.
    fn mark_fence(&self, queue: usize) -> Option<vk::Fence> {
        let flights = self.queues[queue].flights.get();
        flights.and_then(|flights| {
            exit_guard::enter().and_then(|_busy| {
                let raw = self.functions.get();
                let mut done = flights.done_marks();
                if !done.is_empty() {
                    // SAFETY: each is signaled, its submission completed, and out of the log.
                    let reset = unsafe { raw.reset_fences(&done) };
                    exit_guard::called(Call::ResetFences);
                    // Kept idle where reset; the rest, and all where the reset failed, released.
                    if reset.is_ok() {
                        self.fences.keep(&mut done);
                    }
                    for fence in done {
                        // SAFETY: as above.
                        unsafe { crate::release::object(raw, fence) };
                    }
                }
                // SAFETY: the device's fences, inside a section.
                unsafe { self.fences.take(raw) }.ok()
            })
        })
    }

    /// [`use_queue_kept`](Self::use_queue_kept) of `queue`, `by`'s, under `hold`, the queue's
    /// lock, `fence` the relay's mark's ([`mark_fence`](Self::mark_fence)).
    fn use_held<R>(
        &self,
        hold: &mut SlotHold<'_>,
        (queue, by): (usize, By),
        (uses, keep): (Uses, bool),
        fence: Option<vk::Fence>,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> (R, Option<Mark<'a>>) {
        let queues: &'a [super::QueueSlot] = self.queues;
        let slot = &queues[queue];
        let flights = slot.flights.get();
        let parking = self.parking.filter(|_| matches!(by, By::Program));
        let checks = matches!(by, By::Checks);
        if flights.is_none() && parking.is_none() {
            let used = match uses {
                Uses::Unknown => hold.with(f),
                Uses::OneSignalling => {
                    let (used, behind) = hold.with_counted(Caller::Work, |vk_queue, period| {
                        let cpu = period.cpu_now();
                        let entered = crate::gpu_time::host_nanos();
                        let used = f(vk_queue);
                        let end = crate::gpu_time::host_nanos();
                        let took = period.took(end.saturating_sub(entered), cpu);
                        if period.called(Caller::Work, took, end) {
                            self.counts.submit_stall_on_work();
                        }
                        used
                    });
                    if let Some(waited) = behind {
                        self.counts.submit_behind_idle_stall(waited);
                    }
                    used
                }
            };
            return (used, None);
        }
        hold.with(|vk_queue| {
            // Taken before `f` submits: a block that goes meanwhile waits for the mark.
            let mark = parking.and_then(|parking| {
                let _busy = exit_guard::enter()?;
                parking.program_use(queue, self.functions.get())
            });
            let used = f(vk_queue);
            let Some(_busy) = exit_guard::enter() else {
                return (used, None);
            };
            let raw = self.functions.get();
            let logged = fence.zip(flights);
            if let Some((fence, flights)) = logged {
                flights.push(fence, by);
            }
            if fence.is_some() || mark.is_some() {
                let signal = mark.map(|(timeline, value)| ([timeline], [value]));
                let mut values = vk::TimelineSemaphoreSubmitInfo::default();
                let mut submit = vk::SubmitInfo::default();
                if let Some((timelines, signalled)) = &signal {
                    values = values.signal_semaphore_values(signalled);
                    submit = submit.signal_semaphores(timelines).push_next(&mut values);
                }
                // A mark alone needs a submission of its own; a fence alone none.
                let submits = match signal.is_some() {
                    true => std::slice::from_ref(&submit),
                    false => &[],
                };
                match checks {
                    true => self.counts.submit_check_called(1),
                    false => self.counts.queue_submitted(submits.len(), 0),
                }
                // SAFETY: a submission of no work, with an unsignaled fence in no pending
                // submission where it has one, signalling the timeline of marks above every
                // value signalled there before where it has one; the queue is used under its
                // lock.
                let submitted =
                    unsafe { raw.queue_submit(vk_queue, submits, fence.unwrap_or_default()) };
                exit_guard::called(Call::QueueSubmit);
                if let (Ok(()), Some((fence, flights))) = (&submitted, logged) {
                    if keep {
                        flights.made_kept(fence);
                        return (used, Some(Mark { flights, fence }));
                    }
                    flights.made(fence);
                }
                if submitted.is_err() {
                    if let Some((fence, flights)) = logged {
                        flights.unpush(fence);
                        // SAFETY: the failed submission is not pending (as a failed batch's
                        // fence).
                        unsafe { crate::release::object(raw, fence) };
                    }
                    if let (Some(parking), Some((_, value))) = (parking, mark) {
                        // The use's submissions completed once the queue is idle (or the device
                        // is lost); the answer changes nothing.
                        // SAFETY: the queue is used under its lock.
                        let _ = unsafe { raw.queue_wait_idle(vk_queue) };
                        exit_guard::called(Call::QueueWaitIdle);
                        parking.program_completed(queue, value);
                    }
                }
            }
            (used, None)
        })
    }
}

/// The queues a use of several holds, each under its lock ([`QueueUses::use_queues_kept`]).
pub(crate) struct Together<'u, 'a> {
    uses: &'u QueueUses<'a>,
    by: By,
    holds: Vec<(usize, SlotHold<'a>)>,
}

impl<'a> Together<'_, 'a> {
    /// [`QueueUses::use_queue_kept`] of `queue`, one of the queues held, under its hold.
    pub(crate) fn use_queue_kept<R>(
        &mut self,
        queue: usize,
        uses: Uses,
        keep: bool,
        f: impl FnOnce(vk::Queue) -> R,
    ) -> (R, Option<Mark<'a>>) {
        let fence = self.uses.mark_fence(queue);
        let (_, hold) = (self.holds.iter_mut())
            .find(|(held, _)| *held == queue)
            .expect("a use of a queue the use of several holds");
        (self.uses).use_held(hold, (queue, self.by.clone()), (uses, keep), fence, f)
    }
}

/// The relay's mark after a use of a queue, kept for the use's caller
/// ([`QueueUses::use_queue_kept`]): while it lives, no other use takes it from the log or resets
/// its fence, so the caller may wait for it. Let go as it drops, for the next use to take.
pub(crate) struct Mark<'a> {
    flights: &'a Flights,
    fence: vk::Fence,
}

impl Mark<'_> {
    /// The mark's fence, for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn fence(&self) -> vk::Fence {
        self.fence
    }

    /// Wait for the mark's submission to complete, a slice of [`WAIT_SLICE`] at a time, each
    /// inside a section of the exit guard: whether it did (`false` once the process is exiting,
    /// or where the wait failed).
    pub(crate) fn wait(&self) -> bool {
        loop {
            let Some(_busy) = exit_guard::enter() else {
                return false;
            };
            // SAFETY: the fence is the device's, in the log, its call made and the mark kept, so
            // no thread resets or releases it.
            let waited = unsafe {
                (self.flights.functions.get()).wait_for_fences(
                    &[self.fence],
                    true,
                    WAIT_SLICE.as_nanos() as u64,
                )
            };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => return true,
                Err(vk::Result::TIMEOUT) => continue,
                Err(_) => return false,
            }
        }
    }
}

impl Drop for Mark<'_> {
    fn drop(&mut self) {
        self.flights.let_go_of_kept(self.fence);
    }
}

/// The relay messenger's callback: the [`Relay`] at `user` hears each message naming a queue.
pub(super) unsafe extern "system" fn deliver(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes valid callback data, or none.
    let Some(data) = (unsafe { data.as_ref() }) else {
        return vk::FALSE;
    };
    if data.p_message.is_null() || data.p_objects.is_null() || user.is_null() {
        return vk::FALSE;
    }
    // SAFETY: the message is a NUL-terminated string the driver owns for the call.
    let text = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();
    // SAFETY: the callback data holds `object_count` objects.
    let objects = unsafe { std::slice::from_raw_parts(data.p_objects, data.object_count as usize) };
    // SAFETY: `user` is the relay its messenger was made with, which outlives it.
    let relay = unsafe { &*user.cast::<Relay>() };
    // No program code runs here (the exit guard's `program` is for that), and nothing here
    // panics but a broken invariant, which ends the process at this `extern "system"` boundary
    // rather than unwinding into the driver.
    for object in objects {
        if object.object_type == vk::ObjectType::QUEUE {
            relay.heard(vk::Queue::from_raw(object.object_handle), &text);
        }
    }
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::failure;

    /// MoltenVK 1.4.1's message of a failed Metal command buffer gives its code and Metal's
    /// description, the device kept or lost; other messages none (the forms our probe
    /// heard).
    #[test]
    fn a_failure_s_code_and_description_are_read_from_its_message() {
        let resumed = "mvk-warn: VK_ERROR_OUT_OF_DEVICE_MEMORY: Resumed VkDevice after \
                       MTLCommandBuffer \"vkQueueSubmit MTLCommandBuffer on Queue 0-0\" \
                       execution failed (code 1): Discarded (victim of GPU error/recovery)";
        assert_eq!(
            failure(resumed),
            Some((1, "Discarded (victim of GPU error/recovery)"))
        );
        let lost = "VK_ERROR_OUT_OF_DEVICE_MEMORY: Lost VkDevice after MTLCommandBuffer \"a \
                    (code 9) b\" execution failed (code 3): Caused GPU Address Fault Error";
        assert_eq!(failure(lost), Some((3, "Caused GPU Address Fault Error")));
        assert_eq!(failure("Loader Message: Removing layer"), None);
        assert_eq!(
            failure("Lost VkDevice after MTLCommandBuffer \"x\" execution failed (code 0)"),
            None
        );
        assert_eq!(
            failure("Resumed VkDevice after MTLCommandBuffer \"x\" execution failed (code 9)"),
            Some((9, ""))
        );
    }
}
