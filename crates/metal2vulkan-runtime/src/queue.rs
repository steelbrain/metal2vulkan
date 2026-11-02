//! `CommandQueue` (`MTLCommandQueue`).
//!
//! A queue is a logical stream of command buffers, run in commit order. The device's scheduler
//! submits them to the one `VkQueue` the queue took at creation: the device's `VkQueue`s are shared
//! round robin among its logical queues, but for the ones it reserved for queues made for
//! presenting (`Device::new_command_queue_for_presenting`).
//!
//! **What one queue costs another.** Two logical queues that took one `VkQueue` submit to one
//! Vulkan queue, which runs its submissions in order, so one's command buffers wait for the
//! other's whole command buffer, however long it runs. Metal's queues wait for each other only
//! inside one command, measured on an M3 Max under macOS 27.0.1 (a side queue's command buffers
//! run one at a time for a 1.5 s window, over a buffer the long command buffer never names, with
//! 9402 to 10344 completed and 0.8 to 1.6 ms at worst where nothing else ran):
//! - behind one compute dispatch of 2.50 s, 1 completed, after 2496 ms: the whole dispatch;
//! - behind one dispatch of 1.29 s over a grid that filled the GPU, 1 past 100 ms, after 1290 ms:
//!   the whole dispatch again, so it is the command and not the cores that holds the queue;
//! - behind the same 2.50 s as 256 dispatches of one command buffer, 11335 to 12992 completed,
//!   1.7 to 3.8 ms at worst and none past 100 ms: nothing at all.
//!
//! So a long command buffer of many commands holds a shared `VkQueue` where Metal holds no other
//! queue. The runtime answers that with queues rather than with scheduling, which one `VkQueue`
//! cannot give: a created device lends the round robin every queue its families have past the two
//! it reserves for presenting, up to 16 (`device/create.rs`), so a program that makes no more
//! command queues than that shares none. Past them, and on a device whose families have one queue
//! each (MoltenVK's four, RADV's and lavapipe's single graphics queue), queues share and the
//! waiting stands; each queue that takes a `VkQueue` another took is counted
//! (`WorkCounts::queues_sharing_a_vk_queue`).
//!
//! **The caller's own queues.** A caller that submits Vulkan work of its own beside Metal's, and
//! must not wait behind a command queue's command buffers nor hold one back, asks the device for
//! queues of its own families (`DeviceOptions::reserved_queues`, `DeviceHandles::reserved_queues`):
//! the round robin gives them up, one fewer `VkQueue` for each (on an RTX 5080, 13 rather than
//! 14), keeping its first four and presenting's two, and no command queue, presenting or not, and
//! no internal or staging copy ever submits to them. Their families are the device's, which its
//! buffers and images are shared among, so the caller's submissions there use the runtime's own
//! objects with no ownership transfer, ordered against its command buffers through semaphores
//! (a shared event's, or the caller's own). A queue a command queue lends
//! (`CommandQueue::with_vk_queue`) is no such queue: the caller's submissions there sit in that
//! command queue's order, and in every other command queue's that shares its `VkQueue`. A device
//! with no queue to spare (MoltenVK's, lavapipe's, a family of six) reserves none and says so
//! (`metal2vulkan-reserved-queues-absent`).
//!
//! A queue holds at most its `maxCommandBufferCount` uncompleted command buffers, as Metal's does
//! (measured on an M3 Max with macOS 26.7 and an M4 with macOS 27.0.1, with and without API
//! validation):
//! - `newCommandQueue` holds 64; `newCommandQueueWithMaxCommandBufferCount:` holds the count
//!   given (every count from 1 to `NSIntegerMax` is taken). A count of 0 fails Metal's command
//!   queue creation validation, which ends the process with or without API validation; the
//!   runtime refuses it (`Invalid`, `MaxCommandBufferCount`).
//! - A command buffer holds one of its queue's slots from its creation, whether it is made with
//!   retained references or not, until it completes (after its completed handlers, before its
//!   `wait_until_completed` returns), or until it is dropped uncommitted. Enqueued, committed and
//!   scheduled command buffers hold theirs. A completed one's comes back as its last completed
//!   handler returns, whether or not the program still holds it, and without waiting for the
//!   release of what its commands used: a blocked make returns while a no-copy buffer's
//!   deallocator run there still runs (probe `slotdealloc`, the M3 Max under macOS 27.0.1: 0.04
//!   to 0.3 ms after a handler of 0 to 200 ms, with a deallocator of 200 ms under way; counted,
//!   `WorkCounts::slot_return_nanos`).
//! - At the limit, `commandBuffer` blocks until a slot comes back; it never answers nil. The
//!   runtime blocks too, except where no slot can come back: in the queue's own context (its
//!   handlers and deallocators) while every slot is held by a committed command buffer, where
//!   Metal's never returns, it refuses at once (`WaitWouldDeadlock`, `command_buffer/own.rs`).
//! - The limit is the queue's alone: another queue's command buffers are made meanwhile.

use super::attachment_stand_in::BackingPool;
use super::command_buffer::hang::QueueHangs;
use super::command_buffer::own::{running, slot_wait, Answer, Context};
use super::command_buffer::revocation::QueueErrors;
use super::command_buffer::CommandBuffer;
use super::contexts;
use super::degradation::Degradation;
use super::device::{lock, Device};
use super::error::{Error, Invalid, Problem};
use super::texture::copy_images::CopyImages;
use super::texture::kept_image::KeptImages;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

/// The uncompleted command buffers a queue holds when made with `new_command_queue`: Metal's
/// default `maxCommandBufferCount`, measured.
const DEFAULT_MAX_COMMAND_BUFFER_COUNT: usize = 64;

/// A Metal command queue (`MTLCommandQueue`). Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct CommandQueue(Arc<QueueShared>);

impl std::fmt::Debug for CommandQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandQueue")
            .field("label", &self.label())
            .finish()
    }
}

struct QueueShared {
    device: Device,
    /// The queue's identity among the device's logical queues.
    id: u64,
    /// The index of the device's `VkQueue` the queue submits to.
    vk_queue: usize,
    label: Mutex<Option<String>>,
    /// The slots its uncompleted command buffers hold.
    slots: Arc<Slots>,
    /// The latest GPU end, in nanoseconds of the host's clock, of its command buffers completed
    /// so far, which the completion turns place in its order (`gpu_time.rs`). A leaf lock:
    /// taken under a command buffer's, never held while another is taken.
    latest_gpu_end: Mutex<u64>,
    /// The backings of memoryless attachments its passes render into images of their own
    /// (`attachment_stand_in::pooled_backing`). A leaf lock: never held while another is taken.
    memoryless_backings: Mutex<BackingPool>,
    /// The kept images its merges' keeps made, kept for its later keeps
    /// (`texture/kept_image.rs`).
    kept_images: KeptImages,
    /// The images its heaps' multisample textures' moves made, kept for its later moves
    /// (`texture/copy_images.rs`).
    copy_images: CopyImages,
    /// Its command buffers Metal's GPU hangs at that are in flight, and those committed with an
    /// event wait meanwhile (`command_buffer/hang.rs`).
    hangs: Arc<QueueHangs>,
    /// Its command buffers that completed with a GPU error they caused, which revoke it at the
    /// second (`command_buffer/revocation.rs`).
    errors: QueueErrors,
}

/// A queue's slots: one per uncompleted command buffer, at most `max`.
pub(crate) struct Slots {
    max: usize,
    held: Mutex<Held>,
    /// Signaled when a slot comes back.
    freed: Condvar,
    /// The device's slots open, over all its queues.
    uncommitted: Arc<Uncommitted>,
}

/// The command buffers of a device's queues made and not committed (nor dropped): its slots open,
/// over all its queues. A fetched render target takes its storage form only as the only one
/// commits (`texture/storage_form.rs`); a new one counts itself as it takes its slot, before it
/// reads that module's gate.
#[derive(Default)]
pub(crate) struct Uncommitted(AtomicUsize);

impl Uncommitted {
    /// How many there are now.
    pub(crate) fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// The slots held, and how many of them by command buffers not committed.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Held {
    pub(crate) all: usize,
    pub(crate) open: usize,
}

/// One slot of a queue, held by one command buffer; it comes back when dropped.
pub(crate) struct Slot {
    slots: Arc<Slots>,
    /// The command buffer holding it is not committed.
    open: bool,
}

impl Slots {
    /// `max` slots, none held, of a device whose slots open are counted in `uncommitted`.
    pub(crate) fn new(max: usize, uncommitted: Arc<Uncommitted>) -> Arc<Slots> {
        Arc::new(Slots {
            max,
            held: Mutex::new(Held::default()),
            freed: Condvar::new(),
            uncommitted,
        })
    }

    /// The slots held, locked.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Held> {
        lock(&self.held)
    }

    /// A slot for a new command buffer, if one is free in `held` ([`Slots::lock`]).
    pub(crate) fn take(self: &Arc<Self>, held: &mut Held) -> Option<Slot> {
        if held.all == self.max {
            return None;
        }
        held.all += 1;
        held.open += 1;
        self.uncommitted.0.fetch_add(1, Ordering::SeqCst);
        Some(Slot {
            slots: Arc::clone(self),
            open: true,
        })
    }
}

impl Slot {
    /// The command buffer holding the slot was committed: only its completion gives it back now.
    pub(crate) fn committed(&mut self) {
        if self.open {
            self.open = false;
            self.slots.lock().open -= 1;
            self.slots.uncommitted.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        {
            let mut held = self.slots.lock();
            held.all -= 1;
            if self.open {
                held.open -= 1;
                self.slots.uncommitted.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        self.slots.freed.notify_one();
    }
}

impl Device {
    /// `newCommandQueue`: a queue of at most 64 uncompleted command buffers, as Metal's.
    pub fn new_command_queue(&self) -> Result<CommandQueue, Error> {
        self.new_command_queue_with_max_command_buffer_count(DEFAULT_MAX_COMMAND_BUFFER_COUNT)
    }

    /// `newCommandQueueWithMaxCommandBufferCount:`: a queue of at most `count` uncompleted
    /// command buffers. A count of 0 is refused (`Invalid`, `MaxCommandBufferCount`), where
    /// Metal's validation ends the process.
    pub fn new_command_queue_with_max_command_buffer_count(
        &self,
        count: usize,
    ) -> Result<CommandQueue, Error> {
        if count == 0 {
            return Err(Error::Invalid(Invalid::new(
                "new_command_queue_with_max_command_buffer_count",
                Problem::MaxCommandBufferCount,
            )));
        }
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let (id, vk_queue) = self.next_queue();
        Ok(self.command_queue(id, vk_queue, count))
    }

    /// No Metal counterpart: a queue for presenting, of at most 64 uncompleted command buffers as
    /// [`new_command_queue`](Self::new_command_queue)'s, whose `VkQueue` is one the device
    /// reserved for presenting queues, which no queue of `new_command_queue` submits to. A
    /// present that blocks under that `VkQueue`'s lock in
    /// [`CommandQueue::with_vk_queue`](crate::CommandQueue::with_vk_queue) (`vkQueuePresentKHR`
    /// in FIFO mode on a window the compositor does not show, or `vkQueueWaitIdle` at a resize)
    /// then holds no other queue's submissions, as Metal's present holds no other queue.
    ///
    /// A device the runtime creates reserves up to two such `VkQueue`s, of its queue families'
    /// queues past the four the round robin takes (NVIDIA's graphics family has 16), before any
    /// it reserves for the caller
    /// ([`DeviceOptions::reserved_queues`](crate::DeviceOptions::reserved_queues)), and
    /// presenting queues take them in turn; one made past their number shares one with an
    /// earlier presenting queue. A device with no such queue (MoltenVK's four families of one
    /// queue, RADV's and lavapipe's single graphics queue, or a device the caller made,
    /// `Device::from_raw`) gives the queue its `VkQueue` round robin, as `new_command_queue`
    /// does, and announces `Degradation::SharedPresentQueue` (`shared-present-queue`) once.
    pub fn new_command_queue_for_presenting(&self) -> Result<CommandQueue, Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let (id, vk_queue) = match self.next_presenting_queue() {
            Some(reserved) => reserved,
            None => {
                self.degrade(Degradation::SharedPresentQueue, || {
                    (
                        "a command queue made for presenting".to_owned(),
                        "the device reserved no `VkQueue` for presenting: its queue families \
                         have no queue past the ones its other command queues share"
                            .to_owned(),
                    )
                });
                self.next_queue()
            }
        };
        Ok(self.command_queue(id, vk_queue, DEFAULT_MAX_COMMAND_BUFFER_COUNT))
    }

    /// A queue of identity `id` on the device's `VkQueue` at `vk_queue`, of at most `count`
    /// uncompleted command buffers.
    fn command_queue(&self, id: u64, vk_queue: usize, count: usize) -> CommandQueue {
        self.take_device_sampler();
        self.make_texture_pool();
        self.keep_room();
        CommandQueue(Arc::new(QueueShared {
            device: self.clone(),
            id,
            vk_queue,
            label: Mutex::new(None),
            slots: Slots::new(count, Arc::clone(&self.shared().uncommitted)),
            latest_gpu_end: Mutex::new(0),
            memoryless_backings: Mutex::new(BackingPool::default()),
            kept_images: KeptImages::default(),
            copy_images: CopyImages::default(),
            hangs: Arc::default(),
            errors: QueueErrors::default(),
        }))
    }
}

impl Drop for QueueShared {
    fn drop(&mut self) {
        // Every command buffer of the queue holds it, so none is left: no lease holds its pools'
        // blocks (`gpu_pools.rs`).
        self.device.forget_queue_pools(self.id);
    }
}

impl CommandQueue {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `commandBuffer`: a new command buffer of this queue. While the queue holds its
    /// `maxCommandBufferCount` uncompleted command buffers, it blocks until one completes or is
    /// dropped uncommitted, as Metal's does; in the queue's own context, where none can complete
    /// first, it is refused (`WaitWouldDeadlock` of `DeadlockCause::OwnQueue`) unless an
    /// uncommitted one may still be dropped. From another queue's handler it waits as
    /// [`CommandBuffer::wait_until_completed`](crate::CommandBuffer::wait_until_completed) does,
    /// refused where that one is.
    ///
    /// Once per fetched render target that takes its storage form (`fetched-target-storage`,
    /// docs/API.md) at the commit of a command buffer with a submission check, the next command
    /// buffer made waits until the device's scheduler ran that command buffer's checks and
    /// submitted it, at most 100 ms, and not at all on the scheduler's thread (a check that makes
    /// one); past that the target keeps its old image for a later commit.
    pub fn command_buffer(&self) -> Result<CommandBuffer, Error> {
        if self.0.device.is_lost() {
            return Err(Error::DeviceLost);
        }
        let slot = self.slot()?;
        if self.0.device.is_lost() {
            return Err(Error::DeviceLost);
        }
        // Counted with its slot: it waits out a fetched render target's switch to its storage
        // form, if one is under way (`texture/storage_form.rs`).
        self.0.device.storage_form_wait();
        Ok(CommandBuffer::new(self, slot))
    }

    /// A slot for a new command buffer, waiting for one to come back while all are held: counted
    /// (`WorkCounts::slot_waits`).
    fn slot(&self) -> Result<Slot, Error> {
        let slots = &self.0.slots;
        let mut held = slots.lock();
        if let Some(slot) = slots.take(&mut held) {
            return Ok(slot);
        }
        let waited = Instant::now();
        let context = self.context();
        let _blocked = match slot_wait(running(), context, held.open > 0) {
            Answer::Wait | Answer::Return => None,
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Block(from) => {
                // The handler threads' waits are counted outside the slots' lock.
                drop(held);
                let blocked = contexts::block(from, context)?;
                held = slots.lock();
                Some(blocked)
            }
        };
        loop {
            if let Some(slot) = slots.take(&mut held) {
                drop(held);
                let nanos = u64::try_from(waited.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.0.device.shared().counts.slot_waited(nanos);
                return Ok(slot);
            }
            held = slots
                .freed
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// The queue's identity among the device's logical queues.
    pub(super) fn id(&self) -> u64 {
        self.0.id
    }

    /// The index of the device's `VkQueue` the queue submits to.
    pub(super) fn vk_queue(&self) -> usize {
        self.0.vk_queue
    }

    /// The backings of memoryless attachments the queue keeps for its passes
    /// (`attachment_stand_in::pooled_backing`).
    pub(super) fn memoryless_backings(&self) -> &Mutex<BackingPool> {
        &self.0.memoryless_backings
    }

    /// The kept images the queue keeps for its merges' keeps (`texture/kept_image.rs`).
    pub(super) fn kept_images(&self) -> &KeptImages {
        &self.0.kept_images
    }

    /// The images the queue keeps for its heaps' multisample textures' moves
    /// (`texture/copy_images.rs`).
    pub(super) fn copy_images(&self) -> &CopyImages {
        &self.0.copy_images
    }

    /// Its command buffers Metal's GPU hangs at that are in flight, and those committed with an
    /// event wait meanwhile (`command_buffer/hang.rs`).
    pub(super) fn hangs(&self) -> &Arc<QueueHangs> {
        &self.0.hangs
    }

    /// Its command buffers that completed with a GPU error they caused
    /// (`command_buffer/revocation.rs`).
    pub(super) fn errors(&self) -> &QueueErrors {
        &self.0.errors
    }

    /// Place a command buffer of the queue that completed after every one placed before: `place`
    /// takes the latest GPU end of those and gives the command buffer's GPU end, which becomes
    /// the latest where later (`gpu_time.rs`).
    pub(super) fn place_gpu_times(&self, place: impl FnOnce(u64) -> u64) {
        let mut latest = lock(&self.0.latest_gpu_end);
        let end = place(*latest);
        *latest = (*latest).max(end);
    }

    /// The queue's serial context: its device and its identity there (`command_buffer/own.rs`).
    pub(super) fn context(&self) -> Context {
        Context {
            device: self.0.device.shared() as *const _ as usize,
            queue: self.0.id,
        }
    }
}
