//! `CommandQueue` (`MTLCommandQueue`).
//!
//! A queue is a logical stream of command buffers, run in commit order. The device's scheduler
//! submits them to the queue's `VkQueue`: the one it took at creation, the device's `VkQueue`s
//! shared round robin among its logical queues, but for the ones it reserved for queues made for
//! presenting (`Device::new_command_queue_for_presenting`). Under `QueueSharingPolicy::Adaptive`
//! (the default), on a device whose own check finds its GPU idle between consecutive work of two
//! `VkQueue`s, a queue moves where it holds no uncompleted command buffer but the one it makes
//! (`queue_share.rs`); under `PerQueue`, and on every other device, it keeps its first.
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
//! queue. Where queues keep their first `VkQueue`, the runtime answers that with queues rather
//! than with scheduling, which one `VkQueue` cannot give: a created device lends the round robin
//! every queue its families have past the two it reserves for presenting, up to 16
//! (`device/create.rs`), so a program that makes no more command queues than that shares none.
//! Past them, and on a device whose families have one queue each (MoltenVK's four, RADV's and
//! lavapipe's single graphics queue), queues share and the waiting stands; each queue that takes a
//! `VkQueue` another took is counted (`WorkCounts::queues_sharing_a_vk_queue`). Where queues move,
//! they share by choice, to spare the GPU's idling at each switch, and a queue whose command buffer
//! ran on the GPU for over 2 ms, by its own timestamps, is long: it holds the queues that shared
//! with it through that command buffer and while it holds work, then moves to a `VkQueue` of its
//! own at its next idle point, and no queue joins it (`queue-sharing`, `queue_share.rs`).
//!
//! **Isolation.** A queue made with `Device::new_command_queue_unshared` never moves, and no queue
//! joins its `VkQueue`: where the device shares, it takes a `VkQueue` no command queue holds where
//! one is free, so its command buffers (a compositor's) wait behind no other queue's from the
//! start, without the one stall a long queue's first long command buffer costs the queues that
//! shared with it.
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
//!
//! **Across the device's recoveries.** Metal's queue survives a GPU recovery. A queue here is its
//! device's root handle and, in a cell, its place on one generation of the device
//! (`device/line.rs`): its `VkQueue`, its slots, its pools and its hangs in flight, which its
//! command buffers hold. A recovery retires every queue's place; the queue takes a new one on the
//! new generation, as it took its first, at its next command buffer (or lent `VkQueue`), and its
//! label and its count of the GPU errors its command buffers caused (`revocation.rs`) go on.
//!
//! **The runtime's own queues.** The device's in-place check of render targets takes its passes on
//! a queue of its own (`in_place_check.rs`), made as a program's is but for Metal's device ID
//! rules: Metal's device makes its texture pool and takes its sampler ID at the program's first
//! command queue (`resource_ids.rs`), and the check runs on its thread while the program makes its
//! objects, so a queue of its that counted as the program's would keep the pool's first range
//! beside a texture reservation the program made after it, and ignore a sampler reservation.
//! Every place of such a queue, a recovery's new one included, touches neither rule.

use super::attachment_stand_in::BackingPool;
use super::command_buffer::hang::QueueHangs;
use super::command_buffer::own::{running, slot_wait, Answer, Context};
use super::command_buffer::revocation::QueueErrors;
use super::command_buffer::CommandBuffer;
use super::contexts;
use super::degradation::Degradation;
use super::device::{lock, Device};
use super::error::{Error, Invalid, Problem};
use super::queue_share::Choice;
use super::texture::copy_images::CopyImages;
use super::texture::kept_image::KeptImages;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

/// The uncompleted command buffers a queue holds when made with `new_command_queue`: Metal's
/// default `maxCommandBufferCount`, measured.
const DEFAULT_MAX_COMMAND_BUFFER_COUNT: usize = 64;

/// A Metal command queue (`MTLCommandQueue`). Cheap to clone (a retain); `Send + Sync`. It
/// survives the device's recoveries, as Metal's does (the module docs).
#[derive(Clone)]
pub struct CommandQueue(Arc<QueueLine>);

/// A command queue (the module docs): its device, what it was made as, what every place of it
/// shares, and its place on the generation that serves.
pub(crate) struct QueueLine {
    /// The device: a root handle (`device/line.rs`).
    device: Device,
    /// Which constructor made it.
    kind: Kind,
    /// Whether the program made it, for Metal's device ID rules.
    made_for: MadeFor,
    /// Its `maxCommandBufferCount`.
    count: usize,
    common: Arc<QueueCommon>,
    cell: Mutex<Place>,
}

/// What every place of a command queue shares (the module docs).
pub(crate) struct QueueCommon {
    label: Mutex<Option<String>>,
    /// Its command buffers that completed with a GPU error they caused, which revoke it at the
    /// second (`command_buffer/revocation.rs`).
    errors: QueueErrors,
}

/// A command queue's cell (the module docs).
enum Place {
    /// Its place on a generation of the device.
    Serving(QueueGen),
    /// A recovery made the generation of its place lost: it takes a new place at its next
    /// command buffer.
    Retired,
}

/// A command queue's place on one generation of its device: what the command buffers made there
/// hold. Cheap to clone (a retain).
#[derive(Clone)]
pub(crate) struct QueueGen(Arc<QueueShared>);

impl std::fmt::Debug for CommandQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandQueue")
            .field("label", &self.label())
            .finish()
    }
}

/// Which constructor made a queue, and so whether it may move between `VkQueue`s
/// (`queue_share.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `new_command_queue` (or with a count): one of the round robin's, which moves where the
    /// device shares `VkQueue`s.
    RoundRobin,
    /// `new_command_queue_unshared`: never moves, and where the device shares, none moves onto
    /// its `VkQueue`.
    Unshared,
    /// `new_command_queue_for_presenting`: never moves.
    Presenting,
}

/// Whom a queue was made for, and so whether its places count toward Metal's device ID rules
/// (the module docs, `resource_ids.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MadeFor {
    /// A constructor the program called: Metal's device makes its texture pool and takes its
    /// sampler ID at its first queue.
    Program,
    /// The runtime's own (`Device::own_command_queue_now`): touches neither rule.
    Runtime,
}

/// [`QueueShared::sharing`]: one of its command buffers ran long (`queue_share::LONG_HEAD`).
const LONG: u8 = 1;
/// [`QueueShared::sharing`]: a long queue that settled, for good.
const SETTLED: u8 = 2;
/// [`QueueShared::sharing`]: it lent its `VkQueue` (`CommandQueue::with_vk_queue`), so it keeps
/// it for good.
const PINNED: u8 = 4;

struct QueueShared {
    /// The generation of the device the place is on: a generation handle.
    device: Device,
    /// What every place of the queue shares.
    common: Arc<QueueCommon>,
    /// The queue's identity among the device's logical queues.
    id: u64,
    /// The index of the device's `VkQueue` the queue submits to. Changed only under its slots'
    /// lock, as it makes a command buffer while it holds no other uncompleted one, where the
    /// device shares `VkQueue`s (`queue_share.rs`): from one of its command buffers' making to
    /// that command buffer's completion, it reads the same.
    vk_queue: AtomicUsize,
    /// Which constructor made it.
    kind: Kind,
    /// Its part in the sharing: [`LONG`], [`SETTLED`], [`PINNED`].
    sharing: AtomicU8,
    /// The slots its uncompleted command buffers hold.
    slots: Arc<Slots>,
    /// The latest GPU end, in nanoseconds of the host's clock, of its command buffers completed
    /// so far, which the completion turns place in its order (`gpu_time.rs`). A leaf lock:
    /// taken under a command buffer's, never held while another is taken.
    latest_gpu_end: Mutex<u64>,
    /// The backings of memoryless attachments its passes render into images of their own, and of
    /// colors they store `DontCare` that render so (`attachment_stand_in::pooled_backing`). A leaf
    /// lock: never held while another is taken.
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
        self.at_serving(|device| device.new_command_queue_now())
    }

    /// [`Device::new_command_queue`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_command_queue_now(&self) -> Result<CommandQueue, Error> {
        self.new_command_queue_with_max_command_buffer_count_now(DEFAULT_MAX_COMMAND_BUFFER_COUNT)
    }

    /// `newCommandQueueWithMaxCommandBufferCount:`: a queue of at most `count` uncompleted
    /// command buffers. A count of 0 is refused (`Invalid`, `MaxCommandBufferCount`), where
    /// Metal's validation ends the process.
    pub fn new_command_queue_with_max_command_buffer_count(
        &self,
        count: usize,
    ) -> Result<CommandQueue, Error> {
        self.at_serving(|device| device.new_command_queue_with_max_command_buffer_count_now(count))
    }

    /// [`Device::new_command_queue_with_max_command_buffer_count`] on the generation that serves
    /// (`device/line.rs`).
    pub(crate) fn new_command_queue_with_max_command_buffer_count_now(
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
            return Err(self.lost_refusal());
        }
        Ok(self.command_queue(count, Kind::RoundRobin, MadeFor::Program))
    }

    /// No Metal counterpart: a queue the runtime makes for its own work (the module docs), as
    /// [`new_command_queue`](Self::new_command_queue)'s on the generation that serves, but that
    /// makes no texture pool and takes no sampler ID of Metal's device (`resource_ids.rs`).
    /// Refused as that call is on a lost device (`DeviceLost`).
    pub(crate) fn own_command_queue_now(&self) -> Result<CommandQueue, Error> {
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let count = DEFAULT_MAX_COMMAND_BUFFER_COUNT;
        Ok(self.command_queue(count, Kind::RoundRobin, MadeFor::Runtime))
    }

    /// No Metal counterpart: a command queue of at most `count` uncompleted command buffers, as
    /// [`new_command_queue_with_max_command_buffer_count`](Self::new_command_queue_with_max_command_buffer_count)'s,
    /// that shares its `VkQueue` with no other command queue, for a queue whose command buffers
    /// must never wait behind another queue's (a compositor's).
    ///
    /// Under [`QueueSharingPolicy::Adaptive`](metal2vulkan::device::QueueSharingPolicy::Adaptive)
    /// (the default), on a device whose own check found its GPU idle when consecutive work comes
    /// from another `VkQueue` (where the other command queues move among the `VkQueue`s of their
    /// family as they make a command buffer while they hold no other uncompleted one, and share
    /// them), it takes no part in that sharing, in either direction:
    /// - it keeps the `VkQueue` it takes here for its life;
    /// - no other command queue moves onto that `VkQueue`;
    /// - a command queue already on it moves to another `VkQueue` of its family at its next such
    ///   make, where one is not held by a long or unshared queue.
    ///
    /// Where the device shares, it takes a `VkQueue` of the round robin's that no command queue
    /// holds, the one used least recently; else one that no long or unshared queue holds, the one
    /// fewest command queues hold; else the round robin's next. One made before the device's check
    /// answered takes the round robin's next, and the guarantees above hold from the answer on.
    ///
    /// Under [`QueueSharingPolicy::PerQueue`](metal2vulkan::device::QueueSharingPolicy::PerQueue),
    /// and on a device whose check found no such idling (or that has no two `VkQueue`s of one
    /// family to share: MoltenVK's four families of one queue, RADV's and lavapipe's single
    /// graphics queue), no command queue moves and this is
    /// [`new_command_queue_with_max_command_buffer_count`](Self::new_command_queue_with_max_command_buffer_count)
    /// itself: the round robin's next `VkQueue`, shared past the round robin's `VkQueue`s as every
    /// command queue's is.
    ///
    /// Refused as that call is: a count of 0 (`Invalid`, `MaxCommandBufferCount`), and a lost
    /// device (`DeviceLost`).
    pub fn new_command_queue_unshared(&self, count: usize) -> Result<CommandQueue, Error> {
        self.at_serving(|device| device.new_command_queue_unshared_now(count))
    }

    /// [`Device::new_command_queue_unshared`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_command_queue_unshared_now(
        &self,
        count: usize,
    ) -> Result<CommandQueue, Error> {
        if count == 0 {
            return Err(Error::Invalid(Invalid::new(
                "new_command_queue_unshared",
                Problem::MaxCommandBufferCount,
            )));
        }
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        Ok(self.command_queue(count, Kind::Unshared, MadeFor::Program))
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
        self.at_serving(|device| device.new_command_queue_for_presenting_now())
    }

    /// [`Device::new_command_queue_for_presenting`] on the generation that serves
    /// (`device/line.rs`).
    pub(crate) fn new_command_queue_for_presenting_now(&self) -> Result<CommandQueue, Error> {
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let count = DEFAULT_MAX_COMMAND_BUFFER_COUNT;
        Ok(self.command_queue(count, Kind::Presenting, MadeFor::Program))
    }

    /// A queue of at most `count` uncompleted command buffers made by `kind`'s constructor for
    /// `made_for`, with its place on this generation, known to the device's line, whose
    /// recoveries retire it (`device/line.rs`).
    fn command_queue(&self, count: usize, kind: Kind, made_for: MadeFor) -> CommandQueue {
        let common = Arc::new(QueueCommon {
            label: Mutex::new(None),
            errors: QueueErrors::default(),
        });
        let place = self.queue_place((count, kind, made_for), &common);
        let queue = CommandQueue(Arc::new(QueueLine {
            device: self.root(),
            kind,
            made_for,
            count,
            common,
            cell: Mutex::new(Place::Serving(place)),
        }));
        self.line().queue_made(Arc::downgrade(&queue.0));
        queue
    }

    /// The `VkQueue` a queue `kind`'s constructor made takes on this generation, and its identity
    /// among the device's logical queues: the round robin's next; for an unshared queue one no
    /// command queue holds where the device shares; for a presenting one a reserved queue, else
    /// the round robin's next, announced (`shared-present-queue`).
    fn place_queue(&self, kind: Kind) -> (u64, usize) {
        match kind {
            Kind::RoundRobin => self.next_queue(),
            Kind::Unshared => match self.place_unshared() {
                Some(vk_queue) => (self.new_queue_id(), vk_queue),
                None => {
                    let (id, vk_queue) = self.next_queue();
                    self.shared().queues[vk_queue].share.unshared_took();
                    (id, vk_queue)
                }
            },
            Kind::Presenting => match self.next_presenting_queue() {
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
            },
        }
    }

    /// A queue's place on this generation (the module docs), of at most `count` uncompleted
    /// command buffers, made by `kind`'s constructor for `made_for`, `common` what its places
    /// share: on the `VkQueue` [`place_queue`](Self::place_queue) gives. One the program made
    /// takes Metal's device sampler ID and makes its texture pool, as Metal's first queue does
    /// (`resource_ids.rs`); the runtime's own touches neither. One on a round robin's `VkQueue`
    /// counts itself there (an unshared one counted itself already), and the first starts the
    /// device's check of its GPU's idling between `VkQueue`s (`queue_share.rs`).
    fn queue_place(
        &self,
        (count, kind, made_for): (usize, Kind, MadeFor),
        common: &Arc<QueueCommon>,
    ) -> QueueGen {
        let (id, vk_queue) = self.place_queue(kind);
        if made_for == MadeFor::Program {
            self.take_device_sampler();
            self.make_texture_pool();
        }
        self.keep_room();
        let shared = self.shared();
        if vk_queue < shared.round_robin() {
            if kind != Kind::Unshared {
                shared.queues[vk_queue].share.took();
            }
            self.start_queue_switch_check();
        }
        QueueGen(Arc::new(QueueShared {
            device: self.clone(),
            common: Arc::clone(common),
            id,
            vk_queue: AtomicUsize::new(vk_queue),
            kind,
            sharing: AtomicU8::new(0),
            slots: Slots::new(count, Arc::clone(&self.shared().uncommitted)),
            latest_gpu_end: Mutex::new(0),
            memoryless_backings: Mutex::new(BackingPool::default()),
            kept_images: KeptImages::default(),
            copy_images: CopyImages::default(),
            hangs: Arc::default(),
        }))
    }
}

impl QueueLine {
    /// The queue's place on the generation that serves: the one in its cell, else a new one made
    /// on that generation, after its recovery (the module docs). The cell's lock is not held
    /// while the device's line is asked for its generation, whose recovery retires the queue.
    fn serving(&self) -> QueueGen {
        if let Place::Serving(place) = &*lock(&self.cell) {
            if place.serves() {
                return place.clone();
            }
        }
        let device = self.device.current();
        let mut cell = lock(&self.cell);
        if let Place::Serving(place) = &*cell {
            // Another thread's call made it meanwhile, or the device cannot recover and the lost
            // generation answers.
            if place.serves() || place.0.device.same(&device) {
                return place.clone();
            }
        }
        let place = device.queue_place((self.count, self.kind, self.made_for), &self.common);
        let retired = std::mem::replace(&mut *cell, Place::Serving(place.clone()));
        drop(cell);
        drop(retired);
        place
    }

    /// A recovery made the generation of the queue's place lost: it takes a new place at its next
    /// command buffer (`device/line.rs`). The place goes once the command buffers made there went.
    pub(crate) fn retire(&self) {
        let retired = std::mem::replace(&mut *lock(&self.cell), Place::Retired);
        drop(retired);
    }
}

impl Drop for QueueShared {
    fn drop(&mut self) {
        // Every command buffer of the queue holds it, so none is left: no lease holds its pools'
        // blocks (`gpu_pools.rs`).
        self.device.forget_queue_pools(self.id);
        // Its count on its `VkQueue` goes, its long mark with it, so a queue that went long fences
        // its `VkQueue` no longer once it is gone (`queue_share.rs`).
        let shared = self.device.shared();
        let vk_queue = *self.vk_queue.get_mut();
        let share = &shared.queues[vk_queue].share;
        match self.kind {
            Kind::Unshared => share.unshared_left(),
            _ if vk_queue < shared.round_robin() => share.left(*self.sharing.get_mut() & LONG != 0),
            _ => {}
        }
    }
}

impl CommandQueue {
    /// `device`: a handle that follows the device across its recoveries (`device/line.rs`).
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `commandBuffer`: a new command buffer of this queue, on its place on the generation of the
    /// device that serves (the module docs). While the queue holds its `maxCommandBufferCount`
    /// uncompleted command buffers, it blocks until one completes or is dropped uncommitted, as
    /// Metal's does; in the queue's own context, where none can complete first, it is refused
    /// (`WaitWouldDeadlock` of `DeadlockCause::OwnQueue`) unless an uncommitted one may still be
    /// dropped. From another queue's handler it waits as
    /// [`CommandBuffer::wait_until_completed`](crate::CommandBuffer::wait_until_completed) does,
    /// refused where that one is. A device that recovers from the loss of its `VkDevice` makes the
    /// command buffer on its new one, waiting out the recovery; one that cannot is refused
    /// (`DeviceLost`).
    ///
    /// Once per fetched render target that takes its storage form (`fetched-target-storage`,
    /// docs/API.md) at the commit of a command buffer with a submission check, the next command
    /// buffer made waits until the device's scheduler ran that command buffer's checks and
    /// submitted it, at most 100 ms, and not at all on the scheduler's thread (a check that makes
    /// one); past that the target keeps its old image for a later commit.
    pub fn command_buffer(&self) -> Result<CommandBuffer, Error> {
        // A generation lost between the place's taking and the command buffer's making recovers,
        // and the queue takes its next place there; each turn is a recovery that made a new
        // `VkDevice`, and a line that cannot make one answers with its lost place
        // (`device/line.rs`).
        let mut place = self.0.serving();
        loop {
            match place.command_buffer(self) {
                Err(error) if place.device().is_lost() => {
                    let next = self.0.serving();
                    if Arc::ptr_eq(&next.0, &place.0) {
                        return Err(error);
                    }
                    place = next;
                }
                made => return made,
            }
        }
    }

    /// The queue's place on the generation that serves (the module docs).
    pub(crate) fn serving(&self) -> QueueGen {
        self.0.serving()
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.common.label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.common.label).clone()
    }

    /// No Metal counterpart, for the lanes that watch command queues share `VkQueue`s
    /// (`queue_share.rs`): the index of the device's `VkQueue` the queue submits to now, and
    /// whether it is long.
    #[doc(hidden)]
    pub fn vk_queue_sharing(&self) -> (usize, bool) {
        self.serving().vk_queue_sharing()
    }
}

impl QueueGen {
    /// The generation of the device the place is on: a generation handle.
    pub(crate) fn device(&self) -> &Device {
        &self.0.device
    }

    /// Whether the place is on the generation that serves: one relaxed load
    /// (`Device::generation_serves`).
    fn serves(&self) -> bool {
        self.0.device.generation_serves()
    }

    /// What every place of the queue shares: its identity across the device's recoveries
    /// (`device/line.rs`).
    pub(crate) fn common(&self) -> &Arc<QueueCommon> {
        &self.0.common
    }

    /// The queue's `label`.
    pub(crate) fn label(&self) -> Option<String> {
        lock(&self.0.common.label).clone()
    }

    /// A new command buffer of `root`, the queue this is a place of
    /// ([`CommandQueue::command_buffer`]).
    fn command_buffer(&self, root: &CommandQueue) -> Result<CommandBuffer, Error> {
        if self.0.device.is_lost() {
            return Err(self.0.device.lost_refusal());
        }
        let (slot, joined) = self.slot()?;
        if self.0.device.is_lost() {
            return Err(self.0.device.lost_refusal());
        }
        // Counted with its slot: it waits out a fetched render target's switch to its storage
        // form, if one is under way (`texture/storage_form.rs`).
        self.0.device.storage_form_wait();
        let command_buffer = CommandBuffer::new(self, root, slot);
        if joined {
            self.0.device.degrade_once(Degradation::QueueSharing, || {
                (
                    "a command queue's command buffer".to_owned(),
                    "its queue, holding no other uncompleted command buffer, moved onto a \
                     `VkQueue` with another command queue's work in flight: the device's GPU idles \
                     when consecutive work comes from another `VkQueue` (QueueSharingPolicy::Adaptive)"
                        .to_owned(),
                )
            });
        }
        Ok(command_buffer)
    }

    /// A slot for a new command buffer, waiting for one to come back while all are held: counted
    /// (`WorkCounts::slot_waits`). With it, whether the queue joined a `VkQueue` with another
    /// queue's work in flight as it took the slot ([`place`](Self::place)).
    fn slot(&self) -> Result<(Slot, bool), Error> {
        let slots = &self.0.slots;
        let mut held = slots.lock();
        if let Some(slot) = slots.take(&mut held) {
            let joined = self.place(&held);
            return Ok((slot, joined));
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
                let joined = self.place(&held);
                drop(held);
                let nanos = u64::try_from(waited.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.0.device.shared().counts.slot_waited(nanos);
                return Ok((slot, joined));
            }
            held = slots
                .freed
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Where the device shares `VkQueue`s, the queue's `VkQueue` chosen again (`queue_share.rs`)
    /// as it takes a slot, `held` its slots under their lock: only where the new command buffer is
    /// its only uncompleted one, and only for a round robin's queue that is not pinned or settled.
    /// A long queue settles; another takes [`Device::pick_vk_queue`]'s choice. Counted
    /// (`WorkCounts::queue_moves` and the rest); whether it joined a `VkQueue` with another
    /// queue's work in flight.
    fn place(&self, held: &Held) -> bool {
        let queue = &self.0;
        if held.all != 1 || queue.kind != Kind::RoundRobin {
            return false;
        }
        let device = &queue.device;
        if !device.shares_queues() {
            return false;
        }
        let sharing = queue.sharing.load(Ordering::Acquire);
        if sharing & (SETTLED | PINNED) != 0 {
            return false;
        }
        let current = queue.vk_queue.load(Ordering::Acquire);
        let counts = &device.shared().counts;
        if sharing & LONG != 0 {
            queue.sharing.fetch_or(SETTLED, Ordering::AcqRel);
            counts.long_queue_settled();
            if let Some(to) = device.settle_long_queue(current) {
                if device.move_queue(current, to, true) {
                    queue.vk_queue.store(to, Ordering::Release);
                    counts.queue_moved(false, false);
                }
            }
            return false;
        }
        let (to, joined) = match device.pick_vk_queue(current) {
            Choice::Stay => return false,
            Choice::Join(to) => (to, true),
            Choice::Leave(to) => (to, false),
        };
        let off_unshared = device.holds_unshared(current);
        if !device.move_queue(current, to, false) {
            return false;
        }
        queue.vk_queue.store(to, Ordering::Release);
        counts.queue_moved(joined, off_unshared);
        joined
    }

    /// One of the queue's command buffers ran on the GPU for over `queue_share::LONG_HEAD`, by
    /// its own timestamps: where the device shares, a round robin's queue is long from now on,
    /// counted once on its `VkQueue` (`WorkCounts::queues_gone_long`). Called by the completion
    /// turn that read them, before it finishes the command buffer, which holds its slot until
    /// then, so the queue moves meanwhile no more than it holds another `VkQueue`
    /// (`CommandBufferShared::place_gpu_times`).
    pub(super) fn went_long(&self) {
        let queue = &self.0;
        let shared = queue.device.shared();
        if queue.kind != Kind::RoundRobin
            || !shared.queue_switch.shares()
            || queue.sharing.fetch_or(LONG, Ordering::AcqRel) & LONG != 0
        {
            return;
        }
        let vk_queue = queue.vk_queue.load(Ordering::Acquire);
        shared.queues[vk_queue].share.went_long();
        shared.counts.queue_went_long();
    }

    /// The index of the device's `VkQueue` the queue submits to, pinned there for good: a lent
    /// `VkQueue` (`CommandQueue::with_vk_queue`) is the queue's for its life, so the caller's
    /// submissions there stay in its order.
    pub(super) fn pinned_vk_queue(&self) -> usize {
        let queue = &self.0;
        if queue.kind == Kind::RoundRobin && queue.sharing.load(Ordering::Acquire) & PINNED == 0 {
            // Under the slots' lock, as every move is: none is under way, and none follows.
            let _held = queue.slots.lock();
            queue.sharing.fetch_or(PINNED, Ordering::AcqRel);
        }
        queue.vk_queue.load(Ordering::Acquire)
    }

    /// The queue's identity among the device's logical queues.
    pub(super) fn id(&self) -> u64 {
        self.0.id
    }

    /// The index of the device's `VkQueue` the queue submits to: the same from one of its command
    /// buffers' making to that command buffer's completion ([`QueueShared::vk_queue`]).
    pub(super) fn vk_queue(&self) -> usize {
        self.0.vk_queue.load(Ordering::Acquire)
    }

    /// The index of the device's `VkQueue` the place submits to now, and whether it is long
    /// ([`CommandQueue::vk_queue_sharing`]).
    fn vk_queue_sharing(&self) -> (usize, bool) {
        let long = self.0.sharing.load(Ordering::Acquire) & LONG != 0;
        (self.vk_queue(), long)
    }

    /// The backings of memoryless attachments, and of colors stored `DontCare`, the queue keeps
    /// for its passes (`attachment_stand_in::pooled_backing`).
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
    /// (`command_buffer/revocation.rs`): the queue's, over every place of it.
    pub(super) fn errors(&self) -> &QueueErrors {
        &self.0.common.errors
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
