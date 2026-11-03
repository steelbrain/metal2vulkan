//! Interop with the caller's own Vulkan work: the raw handles of the runtime's objects, and what
//! the device keeps for the caller's images and semaphores.
//!
//! - **Raw handles.** Unsafe accessors give the Vulkan objects behind a texture
//!   ([`Texture::vk_image`]), a buffer ([`Buffer::vk_buffer`]), an event ([`Event::vk_semaphore`],
//!   [`SharedEvent::vk_semaphore`]), the device ([`Device::raw`]) and a queue
//!   ([`CommandQueue::with_vk_queue`]).
//! - **The caller's submissions and an event** ([`Event::vk_wait`],
//!   [`Event::reserve_vk_signal`], and a shared event's same hooks): the event's semaphore counts
//!   signals, not Metal's values, so the caller asks the counter value a Metal value is, or
//!   reserves the next one for a value it signals (`event_timeline.rs`). Both kinds share one
//!   implementation; they differ in what a command buffer's wait for a reserved signal becomes: a
//!   plain event's (`MTLEvent`, whose value no host call sets) chains on its `VkQueue`, a shared
//!   event's is held on the host until the signal landed, as its every wait the host may satisfy
//!   is. The runtime keeps owning them: the caller never destroys one, and orders its own use of
//!   one against the runtime's through semaphores or completion.
//!   A queue's `VkQueue` is lent under the runtime's lock of it, which every submission takes,
//!   since Vulkan requires a queue's submissions to be externally synchronized.
//! - **The caller's images** (`texture/wrapped.rs`): textures over images the caller made, which
//!   the device's batches move between the caller's layout and `GENERAL`.
//! - **The caller's buffers** (`wrapped_buffer.rs`): buffers over `VkBuffer`s the caller made and
//!   bound to its own memory (an imported dma-buf, for example), `Shared` at a host address the
//!   caller gives, or `Private`.
//! - **The caller's binary semaphores** (`CommandBuffer::encode_wait_for_vk_semaphore` and
//!   `encode_signal_vk_semaphore`): Vulkan has no query for a semaphore's type, so the runtime
//!   refuses what it can tell is no binary semaphore of the device: a null handle, and its own
//!   timeline semaphores (events' and queues'). Every other contract is the caller's.

use super::buffer::ReadApart;
use super::device::relay::By;
use super::device::submit_stalls::Uses;
use super::error::Error;
use super::event::{CallerSignals, EventRef};
use super::event_timeline::{Signal, WaitFor};
use super::texture::wrapped::Wrapped;
use super::{Buffer, CommandQueue, Device, Event, SharedEvent, Texture};
use ash::vk;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// No Metal counterpart: how a submission of the caller's waits for an event's value
/// ([`Event::vk_wait`], [`SharedEvent::vk_wait`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkEventWait {
    /// The value is reached: the submission needs no wait.
    Reached,
    /// The submission waits for the event's timeline semaphore ([`Event::vk_semaphore`],
    /// [`SharedEvent::vk_semaphore`]) to reach this counter value: the index of the signal,
    /// submitted by the runtime or reserved by the caller, that reaches the value.
    Counter(u64),
    /// No counter value names the value yet: no signal submitted or reserved, and no host set,
    /// reaches it; or the one that does is seen only at its command buffer's completion, after
    /// the copied host buffers it may write took its writes (`event_timeline.rs`). A submission
    /// cannot wait for it on the GPU now: ask again once the signalling command buffer is
    /// scheduled ([`CommandBuffer::wait_until_scheduled`](super::CommandBuffer::wait_until_scheduled)),
    /// or, for the one seen at its completion, completed; or wait on the host, for a shared
    /// event ([`SharedEvent::wait_until_signaled_value`]).
    NotYet,
}

/// No Metal counterpart: what a submission of the caller's signals for an event's value
/// ([`Event::reserve_vk_signal`], [`SharedEvent::reserve_vk_signal`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkEventSignal {
    /// The value is not above what the event's signals, submitted or reserved, and a shared
    /// event's host sets already reach: Metal's signal would change nothing (an event's value
    /// only rises), so the submission signals nothing.
    Dropped,
    /// The submission signals the event's timeline semaphore ([`Event::vk_semaphore`],
    /// [`SharedEvent::vk_semaphore`]) to `counter`, once the semaphore reached `after` where
    /// given.
    Counter {
        /// The counter value the submission signals: the next index, reserved for the caller.
        counter: u64,
        /// The counter value the semaphore must reach before the signal: the signal reserved or
        /// submitted before it, on whatever queue, so the semaphore's signals run in order.
        after: Option<u64>,
    },
}

/// Why a semaphore cannot be one of the caller's binary semaphores.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemaphoreProblem {
    /// A null handle.
    Null,
    /// One of the runtime's own timeline semaphores: an event's ([`Event::vk_semaphore`],
    /// [`SharedEvent::vk_semaphore`]) or a queue's.
    Timeline,
}

/// What a device keeps for interop: the caller's images whose batches move them between layouts,
/// and the timeline semaphores its events hold.
#[derive(Default)]
pub(super) struct Interop {
    pub(super) wrapped: Wrapped,
    timelines: Mutex<HashSet<vk::Semaphore>>,
    /// The caller's reserved signals of the device's events, which its watcher waits for
    /// (`event.rs`).
    pub(super) caller_signals: Arc<CallerSignals>,
}

impl Interop {
    /// An event's timeline semaphore was made.
    pub(super) fn timeline_made(&self, semaphore: vk::Semaphore) {
        self.timelines().insert(semaphore);
    }

    /// An event's timeline semaphore is being destroyed.
    pub(super) fn timeline_gone(&self, semaphore: vk::Semaphore) {
        self.timelines().remove(&semaphore);
    }

    fn timelines(&self) -> std::sync::MutexGuard<'_, HashSet<vk::Semaphore>> {
        self.timelines.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// What the runtime can tell of `semaphore` as the caller's binary semaphore on a device whose
/// own timeline semaphores are `queues` (its queues') and `events` (its events').
pub(super) fn semaphore_problem(
    semaphore: vk::Semaphore,
    queues: impl IntoIterator<Item = vk::Semaphore>,
    events: &HashSet<vk::Semaphore>,
) -> Option<SemaphoreProblem> {
    if semaphore == vk::Semaphore::null() {
        Some(SemaphoreProblem::Null)
    } else if events.contains(&semaphore) || queues.into_iter().any(|queue| queue == semaphore) {
        Some(SemaphoreProblem::Timeline)
    } else {
        None
    }
}

impl Device {
    /// Why `semaphore` cannot be the caller's binary semaphore of this device, if the runtime can
    /// tell.
    pub(super) fn binary_semaphore_problem(
        &self,
        semaphore: vk::Semaphore,
    ) -> Option<SemaphoreProblem> {
        let shared = self.shared();
        semaphore_problem(
            semaphore,
            shared.queues.iter().map(|queue| queue.timeline),
            &shared.interop.timelines(),
        )
    }
}

impl Texture {
    /// No Metal counterpart (an interop hook): the texture's `VkImage`, the image of the texture
    /// and of every view of it, made by the runtime or the caller's
    /// ([`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image)). `None`
    /// where the texels are a buffer's bytes the runtime copies into and out of an image of its
    /// own (a texture buffer, a texture over a no-copy buffer's host pages, or a texture over
    /// another buffer that does not alias it), or another image's texels it so copies (a view of
    /// another component layout on a device whose views reinterpret none).
    ///
    /// A texture over a buffer's bytes whose image aliases them gives a linear image of
    /// `VK_IMAGE_TILING_LINEAR`, bound to the buffer's memory, the runtime's own: such a texture's
    /// rows lie one after another and its image's subresource holds no byte but its texels. A
    /// texture over a no-copy buffer's host pages (imported by their pointer or as dma-bufs) gives
    /// none: an image bound to them would start `VK_IMAGE_LAYOUT_UNDEFINED`, whose first move
    /// Vulkan lets read and write every byte bound to it (it is a shadow, `shadowed-texture`). A
    /// texture placed in a heap over host pages imported as dma-bufs (`HostMemory::with_dma_buf`)
    /// gives an image of `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` with `DRM_FORMAT_MOD_LINEAR`:
    /// ask its layout of its memory plane (`VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT`, not the color
    /// aspect), and its modifier with `vkGetImageDrmFormatModifierPropertiesEXT`.
    ///
    /// # Safety
    ///
    /// The image stays its owner's: the caller does not destroy one the runtime made or bind its
    /// memory, and uses it only while the texture lives, ordered against the runtime's work on
    /// it. An image the runtime made rests in `VK_IMAGE_LAYOUT_GENERAL`, owned by the device's
    /// queue family, whenever no batch of the runtime uses it, and the caller leaves it so; a
    /// wrapped image follows its layout contract
    /// ([`VkImageLayouts`](super::VkImageLayouts)).
    ///
    /// The texture keeps the image for good: a fetched render target whose image takes
    /// `STORAGE` later no longer does once its image was handed out.
    pub unsafe fn vk_image(&self) -> Option<vk::Image> {
        if self.shadow().is_some() {
            return None;
        }
        self.exported_image().ok()
    }
}

impl Buffer {
    /// The buffer's `VkBuffer`, and where the buffer's byte 0 lies in it: 0, but for a no-copy
    /// buffer bound below an unaligned pointer. No Metal counterpart: an interop hook. A buffer the
    /// runtime placed in its own memory has no `SHADER_DEVICE_ADDRESS` usage of its own (it takes
    /// its address from its memory block's buffer), so the caller asks it for no device address.
    ///
    /// # Safety
    ///
    /// The `VkBuffer` stays the runtime's: the caller does not destroy it, and uses it only while
    /// the buffer lives, ordered against the runtime's work on it.
    pub unsafe fn vk_buffer(&self) -> (vk::Buffer, u64) {
        self.at_for(0, ReadApart::VkBuffer)
    }

    /// No Metal counterpart (an interop hook): where the buffer's byte 0 lies in Vulkan memory,
    /// where its bytes are the caller's host pages in place, in one `VkDeviceMemory` the device
    /// imported by their pointer (`VK_EXT_external_memory_host`): a no-copy buffer's
    /// (`newBufferWithBytesNoCopy:length:options:deallocator:`), or a buffer's in a heap over the
    /// caller's host memory, whose pages the device imported by their pointer
    /// ([`HostBacking::ImportedByPointer`](crate::HostBacking::ImportedByPointer)) or bound in its
    /// import of a guest RAM chunk ([`HostBacking::InHostRam`](crate::HostBacking::InHostRam)). A
    /// `VkBuffer` of the caller's own bound there uses the bytes the runtime's commands use, with
    /// a usage the runtime's buffers lack (a video decode's bitstream source,
    /// `VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR`); Vulkan asks it to name the answer's
    /// [`handle_type`](VkHostMemory::handle_type) in its `VkExternalMemoryBufferCreateInfo`. The
    /// import is never a dedicated allocation, so any number of buffers bind to it.
    ///
    /// `None` for every other buffer: one in the device's own memory or the caller's `VkBuffer`;
    /// a copied host buffer ([`HostBacking::Copied`](crate::HostBacking::Copied)), whose memory is
    /// a copy the runtime brings in step with the pages only at its own commits and completions;
    /// pages imported as dma-bufs (several allocations, sparsely bound); pages of Metal's private
    /// option bit 24, whose import is of a read-only view that drops the GPU's writes; a detached
    /// buffer; and an import the device released while idle and could not make again (the device
    /// is then lost). As [`vk_buffer`](Self::vk_buffer) does, the call keeps an import the device
    /// would release while idle imported for good.
    ///
    /// # Safety
    ///
    /// The memory stays the runtime's: the caller never frees or maps it, and the buffers it
    /// binds to it are destroyed, their submissions completed, before the answer ends. The
    /// answer holds while the buffer lives and lies in the same pages: until the buffer's (for a
    /// heap's buffer, its heap's) [`rebind_bytes_no_copy`](Self::rebind_bytes_no_copy) or
    /// [`detach_backing`](Self::detach_backing). The caller orders its work on the bytes against
    /// the runtime's (an event's semaphore: [`Event::vk_wait`], [`Event::reserve_vk_signal`],
    /// [`SharedEvent::vk_wait`], [`SharedEvent::reserve_vk_signal`]) and against the host's
    /// writes to the pages, as for any use of the buffer.
    pub unsafe fn vk_host_memory(&self) -> Option<VkHostMemory> {
        self.host_memory_at(0)
    }
}

/// No Metal counterpart: where a buffer's bytes lie in the device's import of the caller's host
/// pages ([`Buffer::vk_host_memory`]).
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VkHostMemory {
    /// The import: memory made with `VkImportMemoryHostPointerInfoEXT` over the pages, or over a
    /// guest RAM chunk the pages lie in.
    pub memory: vk::DeviceMemory,
    /// Where the buffer's byte 0 lies in `memory`.
    pub offset: u64,
    /// The bytes of `memory` from `offset` that are the imported pages': the buffer's length, and
    /// the rest of the import past it.
    pub size: u64,
    /// The memory's type (`VkMemoryAllocateInfo::memoryTypeIndex`), which a buffer bound to it
    /// takes (`VkMemoryRequirements::memoryTypeBits`).
    pub type_index: u32,
    /// The handle type the memory was imported as, which a `VkBuffer` bound to it names in its
    /// `VkExternalMemoryBufferCreateInfo::handleTypes`:
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT`.
    pub handle_type: vk::ExternalMemoryHandleTypeFlags,
}

impl std::fmt::Debug for VkHostMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ash`'s flags print only with its `debug` feature, which the runtime leaves off.
        f.debug_struct("VkHostMemory")
            .field("memory", &self.memory)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("type_index", &self.type_index)
            .field(
                "handle_type",
                &format_args!("{:#x}", self.handle_type.as_raw()),
            )
            .finish()
    }
}

/// Either kind's timeline semaphore ([`Event::vk_semaphore`], [`SharedEvent::vk_semaphore`]).
fn semaphore(event: EventRef<'_>) -> vk::Semaphore {
    event.shared().semaphore()
}

/// Either kind's answer of how the caller's submission waits for `value` ([`Event::vk_wait`],
/// [`SharedEvent::vk_wait`]): the host's answer (`EventTimeline::landing`), the index of the
/// signal that reaches the value whether or not the runtime's own waits for it are held on the
/// host, which for a plain event is its command buffers' own wait.
fn vk_wait(event: EventRef<'_>, value: u64) -> VkEventWait {
    match event.shared().landing(value) {
        Some(WaitFor::Reached) => VkEventWait::Reached,
        Some(WaitFor::Index(index)) => VkEventWait::Counter(index),
        None => VkEventWait::NotYet,
    }
}

/// Either kind's reservation of a signal of `value` for the caller's submission
/// ([`Event::reserve_vk_signal`], [`SharedEvent::reserve_vk_signal`]): the timeline keeps it as
/// the event's next signal whatever its kind (`EventTimeline::signal_apart`); what a command
/// buffer's wait for it becomes is the timeline's answer for the kind (`EventTimeline::wait`).
fn reserve_vk_signal(event: EventRef<'_>, value: u64) -> Result<VkEventSignal, Error> {
    Ok(match event.shared().reserve_signal(value)? {
        Signal::Dropped => VkEventSignal::Dropped,
        Signal::Kept { index, after } => VkEventSignal::Counter {
            counter: index,
            after,
        },
    })
}

impl Event {
    /// No Metal counterpart (an interop hook; Metal's `MTLEvent` is backed by no Vulkan object):
    /// the event's timeline semaphore, which the caller's submissions wait for and signal to
    /// order their work against the runtime's command buffers' `encodeWaitForEvent:` and
    /// `encodeSignalEvent:` of the event. Refuses nothing.
    ///
    /// **Its counter is a signal index, not the event's value.** Metal's value only rises: a
    /// signal not above it changes nothing, and two signals land as their maximum. A Vulkan
    /// timeline semaphore takes only rising signals, so each signal the runtime keeps, its
    /// batches' and the caller's, raises the counter by one, and the runtime maps each index to
    /// its Metal value (`event_timeline.rs`). A submission waits for the counter value
    /// [`vk_wait`](Self::vk_wait) answers for a Metal value, and signals only the one
    /// [`reserve_vk_signal`](Self::reserve_vk_signal) reserved.
    ///
    /// # Safety
    ///
    /// The semaphore stays the runtime's: the caller never destroys it, never uses it as a
    /// binary semaphore, signals it only as a reservation names (never from the host but as
    /// that call says), and uses it only while the event lives: it keeps the event (a clone of
    /// it) until every submission of its own that waits for or signals the semaphore completed.
    pub unsafe fn vk_semaphore(&self) -> vk::Semaphore {
        semaphore(self.into())
    }

    /// No Metal counterpart (an interop hook): how a submission of the caller's waits for the
    /// event's value to reach `value`, as a command buffer's `encodeWaitForEvent:value:` waits:
    /// no wait where the value is reached, a wait for the event's timeline semaphore
    /// ([`vk_semaphore`](Self::vk_semaphore)) to reach the counter value the answer names, or
    /// none yet ([`VkEventWait::NotYet`]). Refuses nothing.
    ///
    /// No host call sets a plain event's value, so the GPU wait for the counter value is Metal's
    /// wait exactly: the signal it names is the one that reaches `value`, as for the runtime's
    /// own command buffers, whose waits for the event chain on their `VkQueue`s the same way. A
    /// counter value the semaphore already reached is a wait that ends at once.
    pub fn vk_wait(&self, value: u64) -> VkEventWait {
        vk_wait(self.into(), value)
    }

    /// No Metal counterpart (an interop hook): reserve a signal of `value` for a submission of
    /// the caller's, as a command buffer's `encodeSignalEvent:value:` signals. Where `value` is
    /// above everything the event's signals reach, the answer names the next counter value of the
    /// event's timeline semaphore ([`vk_semaphore`](Self::vk_semaphore)), which the caller's
    /// submission signals, after a wait for the signal before it (`after`, on whatever queue that
    /// one runs); else the signal is dropped, as Metal's maximum makes it a no-op.
    ///
    /// From the reservation, the runtime counts that signal among the event's: a command
    /// buffer's wait for `value` (or less) goes to its `VkQueue` as a wait for `counter`, whether
    /// or not the caller's submission was made yet (a wait before its signal, which Vulkan's
    /// timeline semaphores take), as a wait for a command buffer's signal of the event does; a
    /// later command buffer's signal of the event waits for it to land; and the device's watcher
    /// waits for it on the host and, once it landed, observes it, as for a command buffer's
    /// signal. No host call sets a plain event's value, so none of these waits is held on the
    /// host, unlike a shared event's ([`SharedEvent::reserve_vk_signal`]). The signal is ordered
    /// after the work the caller's submission orders it after, as Vulkan orders a semaphore
    /// signal; work the runtime submitted is no part of it.
    ///
    /// Refused where the device is lost (`DeviceLost`), and where the watcher's thread cannot
    /// start (`OutOfMemory(Host)`): nothing is reserved then.
    ///
    /// # Safety
    ///
    /// For a [`VkEventSignal::Counter`] answer, the caller submits exactly one signal operation of
    /// the semaphore to `counter`, in a submission that first waits for the semaphore to reach
    /// `after` where given, to a queue of the event's device, soon: every later signal of the
    /// event and every wait for a value it reaches waits for it, the runtime's command buffers'
    /// on their queues included, and a reservation never signalled stalls them for good. Those
    /// command buffers may be submitted before the signal is, so the caller's submission depends
    /// on none of them: it waits for no such command buffer's completion, and goes to a queue no
    /// command queue submits to (an extra or reserved queue,
    /// [`DeviceHandles::extra_queues`](crate::DeviceHandles::extra_queues),
    /// [`DeviceHandles::reserved_queues`](crate::DeviceHandles::reserved_queues)), or to a
    /// command queue's `VkQueue` ([`CommandQueue::with_vk_queue`]) behind none of them. Where the
    /// submission fails, the caller signals the semaphore to `counter` from the host
    /// (`vkSignalSemaphore`) once the semaphore reached `after`, so the rest goes on: a signal of
    /// the semaphore itself, which releases the waits on the `VkQueue`s as the submission's
    /// would, not a Metal host set, which a plain event has none of. The semaphore's other
    /// contract is [`vk_semaphore`](Self::vk_semaphore)'s; the runtime holds the event until the
    /// signal landed.
    pub unsafe fn reserve_vk_signal(&self, value: u64) -> Result<VkEventSignal, Error> {
        reserve_vk_signal(self.into(), value)
    }
}

impl SharedEvent {
    /// No Metal counterpart (an interop hook; Metal's `MTLSharedEvent` is backed by no Vulkan
    /// object): the event's timeline semaphore, which the caller's submissions wait for and
    /// signal to order their work against the runtime's command buffers' `encodeWaitForEvent:`
    /// and `encodeSignalEvent:` of the event.
    ///
    /// **Its counter is a signal index, not the event's value.** Metal's value only rises: a
    /// signal not above it changes nothing, two signals land as their maximum, and a host set
    /// (`setSignaledValue:`) takes effect at once. A Vulkan timeline semaphore takes only rising
    /// signals, and no host signal while a GPU signal is pending
    /// (`VUID-VkSemaphoreSignalInfo-value-03258`). So each signal the runtime keeps, its
    /// batches' and the caller's, raises the counter by one, and the runtime maps each index to
    /// its Metal value; a host set never touches the semaphore (`event_timeline.rs`). A
    /// submission waits for the counter value [`vk_wait`](Self::vk_wait) answers for a Metal
    /// value, and signals only the one [`reserve_vk_signal`](Self::reserve_vk_signal) reserved.
    ///
    /// # Safety
    ///
    /// The semaphore stays the runtime's: the caller never destroys it, never uses it as a
    /// binary semaphore, signals it only as a reservation names (never from the host but as
    /// that call says), and uses it only while the event lives: it keeps the event (a clone of
    /// it) until every submission of its own that waits for or signals the semaphore completed.
    pub unsafe fn vk_semaphore(&self) -> vk::Semaphore {
        semaphore(self.into())
    }

    /// No Metal counterpart (an interop hook): how a submission of the caller's waits for the
    /// event's value to reach `value`, as a command buffer's `encodeWaitForEvent:value:` waits:
    /// no wait where the value is reached, a wait for the event's timeline semaphore
    /// ([`vk_semaphore`](Self::vk_semaphore)) to reach the counter value the answer names, or
    /// none yet ([`VkEventWait::NotYet`]).
    ///
    /// Unlike Metal's wait, a GPU wait for a counter value is released only by the signal it
    /// names, not by a host set of the value meanwhile (`setSignaledValue:`), which Metal's wait
    /// would take at once: the signal comes, so the wait ends, later at most. A counter value the
    /// semaphore already reached is a wait that ends at once.
    pub fn vk_wait(&self, value: u64) -> VkEventWait {
        vk_wait(self.into(), value)
    }

    /// No Metal counterpart (an interop hook): reserve a signal of `value` for a submission of
    /// the caller's, as a command buffer's `encodeSignalEvent:value:` signals. Where `value` is
    /// above everything the event's signals and host sets reach, the answer names the next
    /// counter value of the event's timeline semaphore ([`vk_semaphore`](Self::vk_semaphore)),
    /// which the caller's submission signals, after a wait for the signal before it (`after`,
    /// on whatever queue that one runs); else the signal is dropped, as Metal's maximum makes it
    /// a no-op.
    ///
    /// From the reservation, the runtime counts that signal among the event's: a command
    /// buffer's wait for `value` (or less) waits for it, a later command buffer's signal of the
    /// event waits for it to land, and the device's watcher waits for it on the host and, once
    /// it landed, raises the event's value, wakes its host waits and posts its listeners'
    /// notifications, as for a command buffer's signal. The signal is ordered after the work the
    /// caller's submission orders it after, as Vulkan orders a semaphore signal; work the runtime
    /// submitted is no part of it.
    ///
    /// Refused where the device is lost (`DeviceLost`), and where the watcher's thread cannot
    /// start (`OutOfMemory(Host)`): nothing is reserved then.
    ///
    /// # Safety
    ///
    /// For a [`VkEventSignal::Counter`] answer, the caller submits exactly one signal operation of
    /// the semaphore to `counter`, in a submission that first waits for the semaphore to reach
    /// `after` where given, to a queue of the event's device, soon: every later signal of the
    /// event and every wait for a value it reaches waits for it, the runtime's command buffers'
    /// on their queues included, and a reservation never signalled stalls them for good. Where
    /// the submission fails, the caller signals the semaphore to `counter` from the host
    /// (`vkSignalSemaphore`) once the semaphore reached `after`, so the rest goes on. The
    /// semaphore's other contract is [`vk_semaphore`](Self::vk_semaphore)'s; the runtime holds
    /// the event until the signal landed.
    pub unsafe fn reserve_vk_signal(&self, value: u64) -> Result<VkEventSignal, Error> {
        reserve_vk_signal(self.into(), value)
    }
}

impl CommandQueue {
    /// `f` with the `VkQueue` this queue submits to and its family's index, under the runtime's
    /// lock of that `VkQueue`, which every submission to it takes: the caller's own submissions
    /// and presents (`vkQueuePresentKHR`) go there, externally synchronized as Vulkan requires.
    /// From the first call on the queue keeps that `VkQueue` for its life: where the device
    /// shares `VkQueue`s among its command queues (`QueueSharingPolicy::Adaptive`), it moves no
    /// more.
    /// On MoltenVK, where the runtime relays the driver's command-buffer failures, it submits no
    /// work with a fence of its own after `f` under the same lock, so a failure of the caller's
    /// submissions stays the caller's. There the runtime's frees of its memory wait for the work
    /// in flight (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), the caller's
    /// submissions in `f` among it: that submission of no work also signals a value the use took
    /// before `f` ran, and memory that goes while it is pending is freed once it completed.
    ///
    /// # Safety
    ///
    /// The caller does not keep the handle past `f`, and `f` makes no call into the runtime and
    /// waits for nothing that needs a later submission of the runtime's (the scheduler submits
    /// only under this lock).
    pub unsafe fn with_vk_queue<R>(&self, f: impl FnOnce(vk::Queue, u32) -> R) -> R {
        let shared = self.device().shared();
        // The caller's own submissions are GPU work of the program's: textures' host shadows are
        // read against the device's count of it (`texture/host_shadow.rs`).
        shared.program_gave_gpu_work();
        // The copies queued for the staging queue are made first, waited for: the caller's
        // submissions may read what they wrote (`texture/staging_copy.rs`). A failure there is a
        // lost device, which the caller's own submissions then meet.
        let _ = self.device().flush_staging_waited();
        // The queue keeps this `VkQueue` for good from here on, so the caller's submissions there
        // stay in its order (`queue_share.rs`).
        let vk_queue = self.pinned_vk_queue();
        let family = shared.queue_family_of(vk_queue);
        // Where the queue's failures are relayed, a mark after `f`'s submissions keeps a failure
        // of theirs the caller's (`device/relay.rs`); where freed memory waits for the work in
        // flight, it waits for that mark (`freed_memory.rs`).
        let used = shared.use_queue(vk_queue, By::Program, Uses::Unknown, |queue| {
            f(queue, family)
        });
        if let Some(parking) = shared.parking.get() {
            parking.watch(self.device());
        }
        used
    }
}
