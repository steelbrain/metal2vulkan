//! What a device's recovery carries to its next generation (`line.rs`): the program's events and
//! the buffers of the runtime's memory whose bytes no pointer of the program's reaches or whose
//! pages the runtime owns, each the same Metal object on the new `VkDevice`, as Metal keeps its
//! objects across its GPU's recovery.
//!
//! - **Kept as made.** Each event (`EventShared::new`) and each buffer the program made of its
//!   own (`Device::visible`) is kept here, held weakly, by the generation it was made on.
//! - **Carried before the next generation serves.** The recovery takes the lost generation's live
//!   objects ([`Carry::take`]) and carries each to the new one before the line serves it
//!   ([`Device::carry_to`]), so no call meets the device half carried; a call that waits for the
//!   recovery meets every object carried.
//! - **None left behind.** A call that took its arguments before the loss completes on the lost
//!   generation (`Device::at_serving_taking`), and what it makes is kept here after the recovery
//!   took the generation's objects: its maker carries it then ([`Device::carry_late`]). The take
//!   and the keeping share one lock, so each object is carried once, by one or the other.
//! - **What each kind becomes.** An event: a new timeline semaphore at the value the host saw
//!   (`EventShared::carry`). A `Private` buffer: new memory, its contents lost with the
//!   `VkDevice` (it reads zero). A buffer in fresh pages (`fresh_pages.rs`): the same pages
//!   imported again, its contents and its `contents` pointer as they were (`Buffer::carry`). Each
//!   keeps its identity: its `ObjectId`, hazard key, Metal address (a share of its range) and
//!   range token, its label and purgeable state.
//! - **What stays the lost generation's**, refused where it meets the next
//!   (`RuntimeGap::MadeBeforeRecovery`): a buffer of pooled host-visible memory (`Shared`, and
//!   `Managed`), whose `contents` pointer is its block's mapping, which Vulkan gives no way to
//!   keep over new memory; a no-copy buffer, a heap's buffer, one over the caller's `VkBuffer` or
//!   exported memory, and an event whose semaphore was made exportable, whose importers hold the
//!   lost one. The recovery's notice counts those it kept by kind (`Degradation::DeviceRecovered`):
//!   every one but a heap's buffer, which its heap holds.
//! - **Each generation's work reaches its own parts** ([`RecordsOn`]). A command buffer of the
//!   lost generation may still record once the recovery carried its objects: an encoder a
//!   program's thread opened before the loss, a render encoder's end on the lowering thread.
//!   Inside a command buffer's work, a carried buffer answers its parts of that command buffer's
//!   generation while it has them there; elsewhere, those of the generation it was carried to
//!   last. Such a command buffer fails as its device's lost one does. A carried event answers
//!   the generation it was carried to last everywhere: a command buffer of the lost generation
//!   that waits for or signals it after the carry refuses it, as its device is the next
//!   generation (`Device::foreign`).
//! - **The lost generation's memory goes once nothing records on it** ([`Recording`]). Its
//!   command buffers that may still record are counted, from their making to their completion
//!   or their drop; once the recovery carried its objects and none is left, the carried buffers'
//!   `VkBuffer`s there are destroyed and their memory freed, and the generation parks and keeps
//!   nothing for reuse from then on (`DeviceShared::keep_nothing`): "When a device is lost, its
//!   child objects are not implicitly destroyed and their handles are still valid. Those objects
//!   must still be destroyed before their parents or the device can be destroyed" (vkspec 5.2.3,
//!   "Lost Device"). A carried event's semaphore there goes with the event: a host wait begun
//!   before the loss may wait on it. Each carried object holds the lost generation, as the
//!   objects left there do, so the lost `VkDevice` and its device-level objects stay until the
//!   last object made on it goes (`WorkCounts::vk_devices_live`).

use super::{lock, Device, DeviceShared};
use crate::buffer::{Buffer, WeakBuffer};
use crate::event::EventShared;
use crate::Error;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The least size at which the registry drops gone objects' entries.
const PRUNE_FLOOR: usize = 256;

/// What a lost generation lets go of once nothing records on it (the module docs): the Vulkan
/// objects of one carried object's parts there.
pub(crate) type LostParts = Box<dyn FnOnce() + Send>;

/// A generation's registry of what its recovery carries, and of the carried objects' parts it lets
/// go of once none of its command buffers may record (the module docs).
#[derive(Default)]
pub(crate) struct Carry {
    state: Mutex<State>,
    /// Its command buffers that may still record ([`Recording`]).
    recording: AtomicUsize,
    /// `State::taken`, read without the lock as a command buffer is made ([`Recording::begin`]).
    taken: AtomicBool,
}

#[derive(Default)]
struct State {
    buffers: Vec<WeakBuffer>,
    events: Vec<Weak<EventShared>>,
    /// The size at which the next keeping drops the gone objects' entries.
    prune_at: usize,
    /// The recovery took the generation's objects ([`Carry::take`]): what is kept from now on,
    /// its maker carries.
    taken: bool,
    /// The carried objects' parts on this generation, let go of once none of its command buffers
    /// may record.
    lost: Vec<LostParts>,
    /// None of its command buffers recorded once the recovery took its objects: what it keeps for
    /// reuse went (`DeviceShared::keep_nothing`).
    at_rest: bool,
}

impl State {
    /// Drop the gone objects' entries once the registry reached its size for that.
    fn prune(&mut self) {
        if self.buffers.len() + self.events.len() < self.prune_at {
            return;
        }
        self.buffers.retain(WeakBuffer::lives);
        self.events.retain(|event| event.strong_count() > 0);
        self.prune_at = (2 * (self.buffers.len() + self.events.len())).max(PRUNE_FLOOR);
    }
}

/// The live objects of a lost generation, which its recovery carries ([`Carry::take`]).
pub(crate) struct Taken {
    buffers: Vec<Buffer>,
    events: Vec<Arc<EventShared>>,
}

impl Carry {
    /// Keep `buffer`, which the program made on this generation: whether the recovery took the
    /// generation's objects already, so that its maker carries it ([`Device::carry_late`]).
    pub(crate) fn buffer(&self, buffer: &Buffer) -> bool {
        let mut state = lock(&self.state);
        state.prune();
        state.buffers.push(buffer.downgrade());
        state.taken
    }

    /// Keep `event`, made on this generation: as [`buffer`](Self::buffer).
    pub(crate) fn event(&self, event: &Arc<EventShared>) -> bool {
        let mut state = lock(&self.state);
        state.prune();
        state.events.push(Arc::downgrade(event));
        state.taken
    }

    /// The generation's live objects, for its recovery to carry; from now on, each one kept is its
    /// maker's to carry.
    fn take(&self) -> Taken {
        let mut state = lock(&self.state);
        state.taken = true;
        self.taken.store(true, Ordering::SeqCst);
        Taken {
            buffers: state
                .buffers
                .iter()
                .filter_map(WeakBuffer::upgrade)
                .collect(),
            events: state.events.iter().filter_map(Weak::upgrade).collect(),
        }
    }

    /// A carried object's `parts` on `shared`, this generation, let go of once none of its
    /// command buffers may record: here where none may.
    pub(crate) fn lost(&self, shared: &DeviceShared, parts: LostParts) {
        lock(&self.state).lost.push(parts);
        self.let_go_at_rest(shared);
    }

    /// On `shared`, this generation: let go of the carried objects' parts once the recovery took
    /// the generation's objects and none of its command buffers may record, and the first time,
    /// of what it keeps for reuse (`DeviceShared::keep_nothing`). Under the registry's lock, which
    /// a command buffer made after that takes ([`Recording::begin`]), so its work meets none of
    /// the parts let go of (`Buffer::s`). Called without a lock of a command buffer's.
    fn let_go_at_rest(&self, shared: &DeviceShared) {
        let mut state = lock(&self.state);
        if !state.taken || self.recording.load(Ordering::SeqCst) > 0 {
            return;
        }
        for parts in state.lost.drain(..) {
            parts();
        }
        if !state.at_rest {
            state.at_rest = true;
            shared.keep_nothing();
        }
    }
}

impl DeviceShared {
    /// A lost generation whose recovery carried its objects on, none of its command buffers left
    /// to record, on which nothing is made again: its parked buffers and images and the empty
    /// blocks it keeps go, and it parks and keeps none from now on, so the memory a carried
    /// object left there goes back to the device (`Recycler::close`, `Memory::keep_none`). Left
    /// to the process's end while it exits.
    fn keep_nothing(&self) {
        let Some(_busy) = crate::exit_guard::enter() else {
            return;
        };
        lock(&self.recycled_buffers).close(self.vk(), &mut lock(&self.memory));
        lock(&self.textures.recycled).close(self.vk(), &mut lock(&self.memory));
        lock(&self.memory).keep_none(self.vk());
    }
}

/// One command buffer's count among its generation's that may still record (`Carry::recording`):
/// from its making until its completion, or its drop where it never completed. Its end lets the
/// carried objects' parts on the generation go where it was the last ([`Carry::let_go_at_rest`]).
pub(crate) struct Recording {
    ended: AtomicBool,
}

impl Recording {
    /// A command buffer made on `shared`'s generation, which may record. On a generation whose
    /// recovery took its objects, after the carried objects' parts there were let go of, if they
    /// were: its work then meets them nowhere (`Buffer::s`).
    pub(crate) fn begin(shared: &DeviceShared) -> Recording {
        let carry = &shared.carry;
        carry.recording.fetch_add(1, Ordering::SeqCst);
        if carry.taken.load(Ordering::SeqCst) {
            drop(lock(&carry.state));
        }
        Recording {
            ended: AtomicBool::new(false),
        }
    }

    /// The command buffer records no more on `shared`'s generation, its own: completed, or
    /// dropped. Once per command buffer; called without a lock of the command buffer's.
    pub(crate) fn end(&self, shared: &DeviceShared) {
        if self.ended.swap(true, Ordering::AcqRel) {
            return;
        }
        if shared.carry.recording.fetch_sub(1, Ordering::SeqCst) == 1 {
            shared.carry.let_go_at_rest(shared);
        }
    }
}

thread_local! {
    /// The generation whose command buffer's work this thread does now ([`RecordsOn`]): the
    /// address of its `DeviceShared`.
    static RECORDS_ON: Cell<Option<usize>> = const { Cell::new(None) };
}

/// This thread doing the work of a command buffer of one generation, until it goes: a buffer its
/// device's recovery carried answers its parts of that generation meanwhile (`Buffer::s`), so a
/// command buffer of the lost generation records the lost generation's handles alone (the module
/// docs). Made with each scope of a command buffer's work (`CommandBufferShared::notices_for`);
/// one made inside another names its own generation until it goes. Neither `Send` nor `Sync`.
pub(crate) struct RecordsOn {
    outer: Option<usize>,
    _thread: std::marker::PhantomData<*const ()>,
}

impl RecordsOn {
    /// The work of a command buffer of `shared`'s generation, from now.
    pub(crate) fn enter(shared: &DeviceShared) -> RecordsOn {
        let address = std::ptr::from_ref(shared).addr();
        RecordsOn {
            outer: RECORDS_ON.with(|records| records.replace(Some(address))),
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for RecordsOn {
    fn drop(&mut self) {
        RECORDS_ON.with(|records| records.set(self.outer));
    }
}

/// The generation whose command buffer's work this thread does now, if any ([`RecordsOn`]): the
/// address of its `DeviceShared`.
pub(crate) fn records_on() -> Option<usize> {
    RECORDS_ON.with(Cell::get)
}

/// Why a buffer stays the lost generation's ([`Buffer::carry`]; the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Left {
    /// Its bytes are pooled host-visible memory (`Shared`, `Managed`), its `contents` pointer its
    /// block's mapping.
    Pooled,
    /// A no-copy buffer, over the caller's host pages.
    NoCopy,
    /// A heap's buffer.
    Heap,
    /// Over the caller's `VkBuffer`, or memory another API imports (`exported_memory.rs`).
    Caller,
    /// Its making on the new generation failed (out of memory, the driver's refusal).
    Failed,
}

/// What the carry of a buffer did with its contents ([`Buffer::carry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Contents {
    /// They are where they were: the same pages, imported again.
    Kept,
    /// Lost with the `VkDevice`: these bytes of new memory, which read zero.
    Lost(u64),
}

/// What one recovery carried, for its notice (`line.rs`).
#[derive(Default)]
pub(crate) struct Carried {
    events: u64,
    /// The buffers whose contents were kept.
    kept: u64,
    /// The buffers whose contents were lost, and their bytes.
    lost: (u64, u64),
    /// The events left the lost generation's: their semaphores were made exportable.
    events_left: u64,
    /// The buffers left the lost generation's, by why.
    left: [u64; 5],
}

impl Carried {
    /// The objects carried and left, as the recovery's notice says them: the pooled buffers
    /// left always, every other kind left where some was.
    pub(crate) fn say(&self) -> String {
        let (lost, lost_bytes) = self.lost;
        let kinds = [
            "pooled `Shared` or `Managed` buffers",
            "no-copy buffers",
            "heaps' buffers",
            "buffers over the caller's or exportable memory",
            "buffers whose making there failed",
        ];
        let left = (kinds.iter().zip(self.left).enumerate())
            .filter(|&(at, (_, count))| at == Left::Pooled as usize || count > 0)
            .map(|(_, (kind, count))| format!("{count} {kind}"))
            .chain(
                (self.events_left > 0)
                    .then(|| format!("{} events made exportable", self.events_left)),
            )
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "it carried {} events, {} buffers with their contents and {lost} `Private` buffers \
             ({lost_bytes} bytes) whose contents were lost (they read zero); left refused \
             (`MadeBeforeRecovery`): {left}",
            self.events, self.kept
        )
    }

    /// One buffer's carry.
    fn buffer(&mut self, carried: Result<Contents, Left>) {
        match carried {
            Ok(Contents::Kept) => self.kept += 1,
            Ok(Contents::Lost(bytes)) => {
                self.lost.0 += 1;
                self.lost.1 += bytes;
            }
            Err(left) => self.left[left as usize] += 1,
        }
    }

    /// One event's carry.
    fn event(&mut self, carried: Result<(), Error>) {
        match carried {
            Ok(()) => self.events += 1,
            Err(_) => self.events_left += 1,
        }
    }
}

impl Device {
    /// Carry this generation's live events and buffers, lost, to `next`, the generation its
    /// recovery made, before `next` serves (the module docs): what was carried and left, for the
    /// recovery's notice.
    pub(super) fn carry_to(&self, next: &Device) -> Carried {
        let Taken { buffers, events } = self.shared().carry.take();
        let mut carried = Carried::default();
        for event in &events {
            carried.event(event.carry(next));
        }
        for buffer in &buffers {
            carried.buffer(buffer.carry(next));
        }
        // Let go of where none of this generation's command buffers may record.
        self.shared().carry.let_go_at_rest(self.shared());
        carried
    }

    /// Carry an object made on this generation after its recovery took the generation's objects
    /// (`Carry::buffer`, `Carry::event`) with `carry`, to the generation that serves once that
    /// recovery made it (the module docs). Nothing where no later generation serves: a line that
    /// cannot recover, or a thread command buffers' completion depends on, whose calls reach the
    /// lost generation (`Line::current`).
    pub(crate) fn carry_late<T, E>(&self, carry: impl FnOnce(&Device) -> Result<T, E>) {
        let serving = self.line().current(self);
        if serving.same(self) {
            return;
        }
        // A refusal leaves the object the lost generation's, refused where it meets the next.
        let _carried = carry(&serving);
    }
}
