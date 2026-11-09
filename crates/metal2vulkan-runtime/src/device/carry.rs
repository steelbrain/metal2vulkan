//! What a device's recovery carries to its next generation (`line.rs`): the program's events, its
//! uses of guest RAM, its no-copy buffers and the buffers of the runtime's memory whose bytes no
//! pointer of the program's reaches or whose pages the runtime owns, each the same object on the
//! new `VkDevice`, as Metal keeps its objects across its GPU's recovery.
//!
//! - **Kept as made.** Each event (`EventShared::new`) is kept here, held weakly, by the
//!   generation it was made on; each buffer the program made of its own (`Device::visible`) is
//!   kept among the generation's program resources already (`Owners::buffer`), which the
//!   recovery reads ([`Carry::take`]), so making one costs a load of a flag more and no lock.
//! - **Carried before the next generation serves.** The recovery takes the lost generation's live
//!   objects ([`Carry::take`]) and carries each to the new one before the line serves it
//!   ([`Device::carry_to`]), so no call meets the device half carried; a call that waits for the
//!   recovery meets every object carried.
//! - **None left behind.** A call that took its arguments before the loss completes on the lost
//!   generation (`Device::at_serving_taking`), and what it makes after the recovery took the
//!   generation's objects its maker carries ([`Device::carry_late`]). The take names the buffers
//!   it took under the registry's lock, which a maker that finds the generation taken asks
//!   ([`Carry::late_buffer`]), and an event is kept under it, so each object is carried once, by
//!   one or the other.
//! - **What each kind becomes.** An event: a new timeline semaphore at the value the host saw
//!   (`EventShared::carry`). A `Private` buffer: new memory, its contents lost with the
//!   `VkDevice` (it reads zero). A buffer in fresh pages (`fresh_pages.rs`): the same pages
//!   imported again, its contents and its `contents` pointer as they were (`Buffer::carry`). A
//!   `Shared` or `Managed` buffer in a block of the runtime's own pages (`memory.rs`'s
//!   `Holds::PageBuffers`): the block's pages imported again, once for every buffer carried from
//!   it, the buffer at its offset there, its contents and `contents` pointer as they were. A
//!   no-copy buffer: made again over the caller's pages it lies in, of the memory they were given
//!   as (`HostPages::made_of`), as its making or last rebind made it there (the caller's gate and
//!   resolver of guest RAM runs asked again, on the recovering thread), its contents and
//!   `contents` pointer as they were; its deallocator and keep-alive wait for its release there
//!   and for its pages' import on each generation to be freed (`Release::carry_on`). A use of
//!   guest RAM (`HostRamUse`), before the buffers: made again on the new device, the lost one's
//!   let go of. Each buffer keeps its identity: its `ObjectId`, hazard key, Metal address (a
//!   share of its range) and range token, its label and purgeable state.
//! - **What stays the lost generation's**, refused where it meets the next
//!   (`RuntimeGap::MadeBeforeRecovery`): a buffer of pooled host-visible memory (`Shared`, and
//!   `Managed`) the device allocated, whose `contents` pointer is its block's mapping, which
//!   Vulkan gives no way to keep over new memory (where the device takes no blocks of the
//!   runtime's own pages for it: it imports no host pointer, its check found that a held import
//!   costs its submissions or has not answered, or no memory type of the buffer's class takes
//!   such pages; and a `Managed` buffer's host copy without unified memory); a detached no-copy
//!   buffer, or one whose last detach's deallocator waits
//!   for its next rebind; a heap's buffer, one over the caller's `VkBuffer` or exported memory,
//!   and an event whose semaphore was made exportable, whose importers hold the lost one. The recovery's notice counts those it kept by kind (`Degradation::DeviceRecovered`):
//!   every one but a heap's buffer, which its heap holds.
//! - **Each generation's work reaches its own parts** (`NoticesFor::generation`). A command buffer of the
//!   lost generation may still record once the recovery carried its objects: an encoder a
//!   program's thread opened before the loss, a render encoder's end on the lowering thread.
//!   Inside a command buffer's work, a carried buffer answers its parts of that command buffer's
//!   generation while it has them there; elsewhere, those of the generation it was carried to
//!   last. Such a command buffer fails as its device's lost one does. A carried event answers
//!   the generation it was carried to last everywhere: a command buffer of the lost generation
//!   that waits for or signals it after the carry refuses it, as its device is the next
//!   generation (`Device::foreign`). A call outside command buffers' work that reaches a buffer's
//!   Vulkan objects (a texture over its bytes, an export of its memory, an upload, a handle
//!   handed out) holds the parts it read (`Buffer::held_parts`): their release at the generation's
//!   rest waits for the call to end, and the call completes on the generation it began on. A
//!   handle handed out (`Buffer::vk_buffer`, `Event::vk_semaphore`), which the caller may use
//!   while the object lives, keeps its parts, and the lost `VkDevice`, until the object goes.
//! - **The lost generation goes once nothing records on it.** A command buffer holds its queue's
//!   slot on the generation from its making to its completion, or to its drop where it never
//!   completed (`queue.rs`'s `Slot`); the registry knows the generation's queues' slots
//!   ([`Carry::queue`]) and looks at them as the last of a queue's slots comes back once the
//!   recovery took the generation's objects ([`RestWatch`]). So a generation never recovered pays
//!   a load of a flag as a command buffer is made and as a queue's last slot comes back, and
//!   nothing more. Once the line serves past it and no slot is held, the carried buffers' `VkBuffer`s there
//!   are destroyed and their memory freed, a carried event's semaphore once no host call on it is
//!   under way, and each carried object lets go of its handle of the generation ([`MadeOn`]):
//!   "When a device is lost, its child objects are not implicitly destroyed and their handles are
//!   still valid. Those objects must still be destroyed before their parents or the device can be
//!   destroyed" (vkspec 5.2.3, "Lost Device"). Its device-level objects of work, what it keeps
//!   for reuse and its kept empty blocks go then too, and no command buffer is made on it again
//!   (`DeviceShared::keep_nothing`). So a carried object holds nothing of the lost generation:
//!   the lost `VkDevice` goes with the last object left on it (a pooled buffer, a pipeline state,
//!   a texture), through the teardown every generation's goes through, and holds its own objects
//!   alone meanwhile (`WorkCounts::vk_devices_live`).

use super::{lock, Device, DeviceShared};
use crate::buffer::Buffer;
use crate::event::EventShared;
use crate::host_ram::RamUseCell;
use crate::queue::Slots;
use crate::resource::ObjectId;
use crate::Error;
use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The least size at which the registry drops gone objects' entries.
const PRUNE_FLOOR: usize = 256;

/// One carried object's parts on a lost generation, which it lets go of once nothing records on it
/// (the module docs): their Vulkan objects, their memory, and the object's handle of the
/// generation ([`MadeOn`]).
pub(crate) trait LostParts: Send {
    /// Under the registry's lock: mark the parts let go of, so that the work of a command buffer
    /// made on the generation from now on meets them nowhere.
    fn mark(&self) {}

    /// Then, with no lock held: destroy their Vulkan objects, free their memory and let go of
    /// the object's handle of the generation.
    fn release(self: Box<Self>);
}

/// A generation's registry of what its recovery carries, and of the carried objects' parts it lets
/// go of once none of its command buffers may record (the module docs).
#[derive(Default)]
pub(crate) struct Carry {
    state: Mutex<State>,
    /// `State::taken`, read without the lock as a buffer or a command buffer is made
    /// ([`Carry::taken`]), and as a queue's last slot comes back ([`RestWatch`]).
    taken: Arc<AtomicBool>,
}

#[derive(Default)]
struct State {
    events: Vec<Weak<EventShared>>,
    /// The program's uses of guest RAM (`host_ram.rs`'s `HostRamUse`), held weakly: few, so their
    /// gone entries are dropped at each keeping.
    ram_uses: Vec<Weak<RamUseCell>>,
    /// The size at which the next keeping drops the gone events' entries.
    prune_at: usize,
    /// The recovery took the generation's objects ([`Carry::take`]): what is kept from now on,
    /// its maker carries.
    taken: bool,
    /// The buffers the recovery took ([`Carry::take`]), which their makers leave to it
    /// ([`Carry::late_buffer`]).
    took: HashSet<ObjectId>,
    /// The slots of the generation's queues, which its command buffers hold while they may
    /// record ([`Carry::queue`]).
    queues: Vec<Weak<Slots>>,
    /// The carried objects' parts on this generation, let go of once none of its command buffers
    /// may record.
    lost: Vec<Box<dyn LostParts>>,
    /// The line serves a later generation ([`Carry::served_past`]): no call's command buffer is
    /// made here any more but one that passed its loss check before the loss.
    past: bool,
    /// None of its command buffers recorded once the line served past it: what it keeps for reuse
    /// and its device-level objects went (`DeviceShared::keep_nothing`), and no command buffer is
    /// made here again ([`Carry::refuses`]).
    at_rest: bool,
}

impl State {
    /// Drop the gone events' entries once the registry reached its size for that.
    fn prune(&mut self) {
        if self.events.len() < self.prune_at {
            return;
        }
        self.events.retain(|event| event.strong_count() > 0);
        self.prune_at = (2 * self.events.len()).max(PRUNE_FLOOR);
    }

    /// Whether a command buffer of the generation holds a slot of one of its queues.
    fn slot_held(&self) -> bool {
        (self.queues.iter())
            .filter_map(Weak::upgrade)
            .any(|slots| slots.held_any())
    }
}

/// The live objects of a lost generation, which its recovery carries ([`Carry::take`]).
pub(crate) struct Taken {
    buffers: Vec<Buffer>,
    events: Vec<Arc<EventShared>>,
    ram_uses: Vec<Arc<RamUseCell>>,
}

impl Carry {
    /// Whether the recovery took the generation's objects ([`Carry::take`]): one load, which a
    /// maker of a buffer reads once the buffer is among the generation's program resources, and a
    /// maker of a command buffer once it holds its slot.
    pub(crate) fn taken(&self) -> bool {
        self.taken.load(Ordering::Acquire)
    }

    /// Whether the maker of `buffer`, which the program made on this generation and which is
    /// among its program resources (`Owners::buffer`), carries it ([`Device::carry_late`]): the
    /// recovery took the generation's objects ([`taken`](Self::taken)) and not this one. Its
    /// take reads the program resources under this lock: a buffer it missed was kept there after
    /// it read them, so after it set the flag, which its maker then finds set.
    pub(crate) fn late_buffer(&self, buffer: &Buffer) -> bool {
        let state = lock(&self.state);
        state.taken && !state.took.contains(&buffer.object_id())
    }

    /// Keep `event`, made on this generation: as [`buffer`](Self::buffer).
    pub(crate) fn event(&self, event: &Arc<EventShared>) -> bool {
        let mut state = lock(&self.state);
        state.prune();
        state.events.push(Arc::downgrade(event));
        state.taken
    }

    /// Keep `cell`, a use of guest RAM made on this generation: as [`event`](Self::event).
    pub(crate) fn ram_use(&self, cell: &Arc<RamUseCell>) -> bool {
        let mut state = lock(&self.state);
        state.ram_uses.retain(|cell| cell.strong_count() > 0);
        state.ram_uses.push(Arc::downgrade(cell));
        state.taken
    }

    /// The live objects of `shared`, this generation, for its recovery to carry: its events, its
    /// uses of guest RAM and its program resources' buffers (`Owners::buffers`). From now on, each
    /// one made is its maker's to carry.
    fn take(&self, shared: &DeviceShared) -> Taken {
        let mut state = lock(&self.state);
        state.taken = true;
        self.taken.store(true, Ordering::SeqCst);
        let buffers = shared.owners.buffers();
        state.took = buffers.iter().map(Buffer::object_id).collect();
        Taken {
            buffers,
            events: state.events.iter().filter_map(Weak::upgrade).collect(),
            ram_uses: state.ram_uses.iter().filter_map(Weak::upgrade).collect(),
        }
    }

    /// Know `slots`, the slots of a queue of `shared`, this generation, which its command
    /// buffers hold while they may record: what watches them for the generation's rest, which a
    /// queue's last slot to come back reads.
    pub(crate) fn queue(&self, shared: &Arc<DeviceShared>, slots: &Arc<Slots>) -> RestWatch {
        let mut state = lock(&self.state);
        state.queues.retain(|queue| queue.strong_count() > 0);
        state.queues.push(Arc::downgrade(slots));
        RestWatch {
            taken: Arc::clone(&self.taken),
            generation: Arc::downgrade(shared),
        }
    }

    /// Whether a command buffer made on this generation, holding its slot, is refused: the
    /// generation is at rest, its device-level objects of work gone ([`let_go_at_rest`]), for a
    /// call that passed its loss check before the loss to refuse as on the lost device. A load of
    /// a flag on a generation never recovered. The rest looks at the slots held under this lock,
    /// so a command buffer that holds its slot as the generation comes to rest is either seen
    /// there or sees the rest here.
    ///
    /// [`let_go_at_rest`]: Self::let_go_at_rest
    pub(crate) fn refuses(&self) -> bool {
        self.taken() && lock(&self.state).at_rest
    }

    /// Whether the generation came to rest ([`let_go_at_rest`](Self::let_go_at_rest)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn at_rest(&self) -> bool {
        lock(&self.state).at_rest
    }

    /// A carried object's `parts` on `shared`, this generation, let go of once none of its
    /// command buffers may record: here where none may.
    pub(crate) fn lost(&self, shared: &DeviceShared, parts: Box<dyn LostParts>) {
        lock(&self.state).lost.push(parts);
        self.let_go_at_rest(shared);
    }

    /// The line serves the generation `shared`'s recovery made, past `shared`, this one: let go
    /// of what it may ([`let_go_at_rest`](Self::let_go_at_rest)).
    pub(super) fn served_past(&self, shared: &DeviceShared) {
        lock(&self.state).past = true;
        self.let_go_at_rest(shared);
    }

    /// On `shared`, this generation, once the line serves past it and none of its command
    /// buffers may record: let go of the carried objects' parts, and the first time, of what it
    /// keeps for reuse and its device-level objects (`DeviceShared::keep_nothing`). The parts are
    /// marked under the registry's lock, which a command buffer made after that takes
    /// ([`Carry::refuses`]), so its work meets none of them (`Buffer::s`), and released after
    /// it. Called with none of a command buffer's locks held, by a holder of the generation.
    fn let_go_at_rest(&self, shared: &DeviceShared) {
        let (lost, first) = {
            let mut state = lock(&self.state);
            if !state.taken || !state.past || state.slot_held() {
                return;
            }
            let mut lost = Vec::with_capacity(state.lost.len());
            for parts in state.lost.drain(..) {
                parts.mark();
                lost.push(parts);
            }
            let first = !state.at_rest;
            state.at_rest = true;
            // A maker that finds the generation taken now finds it at rest: the recovery carried
            // what it took before the line served past it.
            state.took.clear();
            state.took.shrink_to_fit();
            (lost, first)
        };
        if first {
            shared.keep_nothing();
        }
        for parts in lost {
            parts.release();
        }
    }
}

impl DeviceShared {
    /// A lost generation whose recovery carried its objects on, the line serving past it and none
    /// of its command buffers left to record, on which no work is made again: its device-level
    /// objects of work (the internal submissions' and the staging queue's pools, the runtime's own
    /// kernels, the GPU pools' blocks, the mipmap scratch), its parked buffers and images and the
    /// empty blocks it keeps go, and it parks and keeps none from now on. So the objects left on
    /// it hold the bare `VkDevice` and their own memory alone, and a carried object's memory there
    /// goes back to the device (`Recycler::close`, `memory::keep_none`). Its teardown destroys the
    /// rest, and the `VkDevice`, as ever (`DeviceShared`'s drop). Left to the process's end while
    /// it exits.
    fn keep_nothing(&self) {
        let Some(_busy) = crate::exit_guard::enter() else {
            return;
        };
        // Its queues idle, each under its lock: its submission checks' thread may submit there
        // still (`DeviceShared::queues_idle`).
        self.queues_idle();
        if let Some(internal) = lock(&self.internal).take() {
            internal.destroy(self.vk());
        }
        if let Some(staging) = &self.staging {
            staging.destroy(self.vk());
        }
        lock(&self.owned_kernels).destroy(self.vk());
        lock(&self.mipmap_scratch).destroy(self.vk(), &mut lock(&self.memory));
        self.gpu_pools.destroy(self.vk(), &mut lock(&self.memory));
        lock(&self.recycled_buffers).close(self.vk(), &mut lock(&self.memory));
        lock(&self.textures.recycled).close(self.vk(), &mut lock(&self.memory));
        crate::memory::keep_none(&self.memory, self.vk());
    }
}

/// What a queue's slots on a generation read as the last of them comes back (`queue.rs`'s
/// `Slot`): whether the recovery took the generation's objects, and if so, the generation, to let
/// go of the carried objects' parts there where no slot of its queues is held now
/// ([`Carry::let_go_at_rest`]).
pub(crate) struct RestWatch {
    taken: Arc<AtomicBool>,
    generation: Weak<DeviceShared>,
}

impl RestWatch {
    /// The last of the queue's slots came back, with no lock of the queue's held: a load of a
    /// flag on a generation never recovered.
    pub(crate) fn last_slot_back(&self) {
        if !self.taken.load(Ordering::Acquire) {
            return;
        }
        if let Some(shared) = self.generation.upgrade() {
            shared.carry.let_go_at_rest(&shared);
        }
    }
}

/// Why a buffer stays the lost generation's ([`Buffer::carry`]; the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Left {
    /// Its bytes are pooled host-visible memory (`Shared`, `Managed`), its `contents` pointer its
    /// block's mapping.
    Pooled,
    /// A no-copy buffer detached from the caller's pages, or whose last detach's deallocator
    /// waits for its next rebind.
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
    /// The uses of guest RAM carried, and those the new generation refused.
    ram_uses: (u64, u64),
    /// The buffers left the lost generation's, by why.
    left: [u64; 5],
}

impl Carried {
    /// The objects carried and left, as the recovery's notice says them: the pooled buffers
    /// left always, every other kind left where some was, and the uses of guest RAM where any
    /// lived.
    pub(crate) fn say(&self) -> String {
        let (lost, lost_bytes) = self.lost;
        let (ram_uses, ram_uses_left) = self.ram_uses;
        let kinds = [
            "pooled `Shared` or `Managed` buffers",
            "detached no-copy buffers",
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
            .chain(
                (ram_uses_left > 0)
                    .then(|| format!("{ram_uses_left} guest RAM uses the new device refused")),
            )
            .collect::<Vec<_>>()
            .join(", ");
        let ram = match ram_uses {
            0 => String::new(),
            _ => format!("{ram_uses} guest RAM uses, "),
        };
        format!(
            "it carried {} events, {ram}{} buffers with their contents and {lost} `Private` \
             buffers ({lost_bytes} bytes) whose contents were lost (they read zero); left refused \
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

    /// One use of guest RAM's carry.
    fn ram_use(&mut self, carried: Result<(), Error>) {
        match carried {
            Ok(()) => self.ram_uses.0 += 1,
            Err(_) => self.ram_uses.1 += 1,
        }
    }
}

impl Device {
    /// Carry this generation's live events and buffers, lost, to `next`, the generation its
    /// recovery made, before `next` serves (the module docs): what was carried and left, for the
    /// recovery's notice.
    pub(super) fn carry_to(&self, next: &Device) -> Carried {
        let shared = self.shared();
        let Taken {
            buffers,
            events,
            ram_uses,
        } = shared.carry.take(shared);
        let mut carried = Carried::default();
        for event in &events {
            carried.event(event.carry(next));
        }
        // Before the buffers: a no-copy buffer over a RAM's pages lies in its chunks there.
        for cell in &ram_uses {
            carried.ram_use(cell.carry(next));
        }
        for buffer in &buffers {
            carried.buffer(buffer.carry(next));
        }
        carried
    }

    /// Carry an object made on this generation after its recovery took the generation's objects
    /// (`Carry::late_buffer`, `Carry::event`) with `carry`, to the generation that serves once that
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

/// The device of an object a recovery may carry (a buffer, its pages, an event): the generation it
/// was made on, as every object's device is, until the object's parts there were let go of
/// (`Carry::lost`); then let go of itself ([`MadeOn::let_go`]), so a carried object holds the lost
/// generation no longer and that generation goes with the objects left on it (the module docs).
/// Read as a [`Device`] at one indirection, as the field it stands for.
pub(crate) struct MadeOn {
    device: UnsafeCell<ManuallyDrop<Device>>,
    /// It was let go of.
    gone: AtomicBool,
}

// SAFETY: it is a `Device`, which is `Send` and `Sync`, read through shared references, and let
// go of once, when nothing reads it any more (`MadeOn::let_go`'s contract).
unsafe impl Sync for MadeOn {}

impl MadeOn {
    /// `device`'s handle.
    pub(crate) fn new(device: Device) -> MadeOn {
        MadeOn {
            device: UnsafeCell::new(ManuallyDrop::new(device)),
            gone: AtomicBool::new(false),
        }
    }

    /// Whether it was let go of: the object's parts on the generation went, the object carried.
    pub(crate) fn gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }

    /// Let go of the handle, once.
    ///
    /// # Safety
    ///
    /// Nothing reads it from now on but [`gone`](Self::gone): its object's parts on the
    /// generation were let go of and no work of the generation's reaches them
    /// (`Carry::let_go_at_rest`), and every other reach of the object goes to where it was
    /// carried.
    pub(crate) unsafe fn let_go(&self) {
        if self.gone.swap(true, Ordering::AcqRel) {
            return;
        }
        // SAFETY: the caller's: no reference to the handle is left, and none is taken again.
        unsafe { ManuallyDrop::drop(&mut *self.device.get()) };
    }
}

impl std::ops::Deref for MadeOn {
    type Target = Device;

    #[inline(always)]
    fn deref(&self) -> &Device {
        // SAFETY: it lives until `let_go`, after which nothing reads it (its contract).
        unsafe { &*self.device.get() }
    }
}

impl Drop for MadeOn {
    fn drop(&mut self) {
        if !*self.gone.get_mut() {
            // SAFETY: not let go of, and the last reference.
            unsafe { ManuallyDrop::drop(self.device.get_mut()) };
        }
    }
}
