//! Metal's automatic hazard tracking, as pure functions of what was recorded so far and what comes
//! next.
//!
//! Metal orders accesses on its own at two levels, and the runtime records a Vulkan barrier, or a
//! wait for another `VkQueue`, for each ordering it needs:
//! - **Inside an encoder** ([`EncoderUses`]). A blit encoder's commands and a serial compute
//!   encoder's dispatches run in encoded order: `MTLDispatchTypeSerial` "dispatches are executed
//!   in dispatched order", and a memory barrier on a serial encoder is "allowed, but ignored", so
//!   the order covers untracked resources too (Metal's headers; documented, not measured). Before
//!   each command, [`EncoderUses::order`] compares its uses with the encoder's earlier commands:
//!   tracked resources per resource and range. Untracked memory may alias across resources, so
//!   it orders by what a command's uses say of it ([`Reach`]). A command whose uses name every
//!   byte of untracked memory it touches (a blit's, a shadow's copy) orders per key and range
//!   against the others like it and against tracked uses, as each byte has one key and range
//!   wherever it is named: a heap's resources' bytes are the heap's (`Buffer::tracked`), an image
//!   bound in a heap's memory is a use of the heap's key, a view's uses are its texture's, a
//!   shadow's image and staging are its own, and the uses of a buffer or heap over host pages
//!   carry the keys of the others over the same bytes (`host_aliases.rs`). A command that may
//!   touch untracked memory its uses do not name (a shader's, through addresses, argument
//!   buffers and resident resources) orders against every untracked access, as one class, and
//!   every command orders against it.
//! - **Between encoders, command buffers and queues** ([`Tracker`]). The device's scheduler
//!   submits batches (a command buffer's encoders between its event waits and signals) in one
//!   order over all its logical queues, and [`Tracker::plan`] compares each batch's encoder
//!   [`Summary`]s (their tracked uses, the stages they ran in, and their fence waits and updates)
//!   with the device's state in that order. Metal orders tracked resources across queues in the
//!   order it schedules the command buffers, which is this order. Untracked resources are the
//!   program's to order, through fences, barriers and events, and the summary leaves them out.
//!   Metal evaluates a fence at submission, so a wait orders after the fence's last update in
//!   submission order. A render encoder's wait and update cover the draw stages its call names
//!   ([`FenceUse`]): a wait before the fragment stage alone lets the draws' earlier stages run
//!   before the update, and an update after the vertex stage alone lets a later wait run before
//!   the draws' fragments, as an M3 Max's did (`fence.rs`).
//!
//! **Barriers and waits.** Every image rests in `VK_IMAGE_LAYOUT_GENERAL` outside render passes,
//! so nothing here changes a layout; but for a held multisampled depth (`texture/held_depth.rs`),
//! whose moves between encoders the scheduler plans with the encoders, each a write of the whole
//! image outside them, which the plan orders as any other. An access after one of the same
//! `VkQueue` needs a [`Barrier`]: one global memory barrier, everything in `src` before everything
//! in `dst`, which orders buffer and image memory alike; Vulkan's submission order on one
//! `VkQueue` makes it cover the earlier batches of every logical queue there. A tracker that
//! orders batches by their ends ([`Tracker::by_ends`], `batch_order.rs`) plans none after an
//! access of an earlier batch of the same `VkQueue`: every batch there ends with
//! [`Barrier::to_all`], every write before every access of every later command, which submission
//! order makes cover the later batches of every logical queue (an execution dependency on every
//! stage, so a write after an earlier batch's read is ordered too). An access after one
//! of another `VkQueue` needs a [`Wait`] of its batch for that queue's timeline semaphore at the
//! earlier batch's [`Point`]: the signal makes the earlier writes available and the wait makes
//! them visible to the waiting stages of this batch and of every later one of its queue. That
//! batch was submitted before, so the wait never waits on an unsubmitted signal. Both may order
//! more than Metal requires, never less.
//!
//! **The state.** Per resource, per range: the last write on each queue that wrote it (one queue's
//! unless entries were merged), with the stages and access types it is already visible to on its
//! own queue and the stages of other queues that waited for it, and the reads since, one set of
//! stages per queue. A read needs a barrier or a wait from a write not yet visible to it. A write
//! needs one from the last writes (unless already visible to it) and from the reads since (an
//! execution dependency). A write drops the older entries its range holds: they are ordered
//! before it, and so before whatever follows it. A resource with more than [`ENTRIES_PER_KEY`]
//! entries has them merged into one per kind of range, which orders more and stays exact. A
//! write or read retires once its batch completed and the host saw it ([`Tracker::completed`]);
//! each `VkQueue`'s batches complete in its timeline's order. A queue submission orders
//! everything that happened before it on the host, and every batch with work waits for the
//! timelines at the values the host saw them reach (`command_buffer/submit.rs`), which makes what
//! completed work wrote visible to it, or, on a driver whose submissions stall on semaphore
//! operations (`DriverFact::SemaphoreOperationsStallSubmits`), starts with
//! [`Barrier::after_completed`]; so later work needs no barrier from it.
//!
//! **Kept cheap per batch.** The scheduler plans every batch on its thread (a commit's own, where
//! it submits inline), so the state's upkeep is held to what a batch touches. Each resource's state
//! is found through [`KeyMap`]'s hash of the runtime's own keys. A batch of one summary (a command
//! buffer of one encoder) is planned against the state itself, and admitted into it as its plan is
//! applied, with the barrier and waits planned (the scheduler does nothing else with the tracker in
//! between, `command_buffer/submit.rs`). A batch of several copies the entries of the resources it
//! uses into a list sorted by resource (one batch's few), not a map, and plans each summary against
//! what the earlier ones made of them. A plan's lists, and the copies', are buffers a plan applied
//! before gave back ([`Spares`]). An entry's writers, reads and waits are kept one per `VkQueue` in
//! a [`Few`], whose first is inline: on a device of one `VkQueue` a copy allocates nothing. And a
//! batch's completion retires only the resources and fences batches up to it touched, each
//! `VkQueue`'s listed in its submission order as their plans were applied, not every resource the
//! state holds.

use super::binding::ResidentPrefix;
use crate::device::functions::Vk;
use crate::key_hash::KeyMap;
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::ops::Range;

/// The most entries the state keeps for one resource before it merges them.
pub(super) const ENTRIES_PER_KEY: usize = 8;

/// The resources an encoder's part makes room for at its first tracked use.
const TRACKED_ROOM: usize = 8;

/// One side of a dependency: pipeline stages, and the access types in them.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(super) struct Access {
    pub(super) stages: vk::PipelineStageFlags2,
    pub(super) access: vk::AccessFlags2,
}

impl Access {
    /// No stage, and so no access.
    pub(super) const NONE: Access =
        Access::new(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);

    /// `access` in `stages`.
    pub(super) const fn new(stages: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Access {
        Access { stages, access }
    }

    /// Whether this side names no stage.
    pub(super) fn is_empty(self) -> bool {
        self.stages.is_empty()
    }

    /// Both sides' stages and access types.
    pub(super) fn union(self, other: Access) -> Access {
        Access::new(self.stages | other.stages, self.access | other.access)
    }

    fn intersection(self, other: Access) -> Access {
        Access::new(self.stages & other.stages, self.access & other.access)
    }

    /// Whether every stage and access type of `other` is in this side.
    fn covers(self, other: Access) -> bool {
        self.stages.contains(other.stages) && self.access.contains(other.access)
    }
}

// `ash` implements `Debug` for its flags only with its `debug` feature: the runtime prints raw
// values.
impl fmt::Debug for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Access {{ stages: {:#x}, access: {:#x} }}",
            self.stages.as_raw(),
            self.access.as_raw()
        )
    }
}

/// The part of a resource a use touches. One resource identity is one kind of storage: a buffer
/// (or heap) is bytes, and an image is aspects, mip levels and array layers. A texture view uses
/// its parent's identity and range; a texture aliasing a buffer uses the buffer's identity and
/// bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum UseRange {
    /// Bytes `start..end`.
    Bytes(Range<u64>),
    /// Aspects, mip levels and array layers of an image. 3D depth slices are not separate, since
    /// Vulkan's barriers cannot name them.
    Image {
        aspects: vk::ImageAspectFlags,
        levels: Range<u32>,
        layers: Range<u32>,
    },
}

impl UseRange {
    /// The range every untracked use of a command that may touch untracked memory its uses do
    /// not name takes inside an encoder, and every untracked use's in the class all of them
    /// order against ([`Reach::Unnamed`]): all untracked memory.
    const UNTRACKED: UseRange = UseRange::Bytes(0..u64::MAX);

    /// Whether the range holds nothing.
    pub(super) fn is_empty(&self) -> bool {
        match self {
            UseRange::Bytes(bytes) => bytes.is_empty(),
            UseRange::Image {
                aspects,
                levels,
                layers,
            } => aspects.is_empty() || levels.is_empty() || layers.is_empty(),
        }
    }

    /// Whether the two ranges share memory. Ranges of different kinds on one key are taken to
    /// share it, which is always correct.
    pub(super) fn overlaps(&self, other: &UseRange) -> bool {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => meet(a, b),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => a.intersects(*b) && meet(a_levels, b_levels) && meet(a_layers, b_layers),
            _ => true,
        }
    }

    /// Whether this range holds all of `other`.
    pub(super) fn contains(&self, other: &UseRange) -> bool {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => within(a, b),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => a.contains(*b) && within(a_levels, b_levels) && within(a_layers, b_layers),
            _ => false,
        }
    }

    /// The smallest range of the same kind holding both, or `None` across kinds.
    pub(super) fn hull(&self, other: &UseRange) -> Option<UseRange> {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => Some(UseRange::Bytes(hull(a, b))),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => Some(UseRange::Image {
                aspects: *a | *b,
                levels: hull(a_levels, b_levels),
                layers: hull(a_layers, b_layers),
            }),
            _ => None,
        }
    }
}

impl fmt::Debug for UseRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UseRange::Bytes(bytes) => write!(f, "Bytes({bytes:?})"),
            UseRange::Image {
                aspects,
                levels,
                layers,
            } => write!(
                f,
                "Image {{ aspects: {:#x}, levels: {levels:?}, layers: {layers:?} }}",
                aspects.as_raw()
            ),
        }
    }
}

fn meet<T: Ord>(a: &Range<T>, b: &Range<T>) -> bool {
    a.start < b.end && b.start < a.end
}

fn within<T: Ord>(outer: &Range<T>, inner: &Range<T>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

fn hull<T: Ord + Copy>(a: &Range<T>, b: &Range<T>) -> Range<T> {
    a.start.min(b.start)..a.end.max(b.end)
}

/// One command's, or one encoder's, access of one resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Use<K> {
    /// The resource's identity: never reused, shared by its views.
    pub(super) key: K,
    /// Whether Metal tracks the resource's hazards (its resolved `hazardTrackingMode`).
    pub(super) tracked: bool,
    pub(super) range: UseRange,
    /// The reads, empty when the use only writes.
    pub(super) read: Access,
    /// The writes, empty when the use only reads.
    pub(super) write: Access,
}

impl<K> Use<K> {
    /// A read of `range`.
    pub(super) fn reads(key: K, tracked: bool, range: UseRange, read: Access) -> Use<K> {
        Use {
            key,
            tracked,
            range,
            read,
            write: Access::NONE,
        }
    }

    /// A write of `range`.
    pub(super) fn writes(key: K, tracked: bool, range: UseRange, write: Access) -> Use<K> {
        Use {
            key,
            tracked,
            range,
            read: Access::NONE,
            write,
        }
    }

    /// Whether the use touches nothing: an empty range, or neither a read nor a write.
    pub(super) fn is_nothing(&self) -> bool {
        self.range.is_empty() || (self.read.is_empty() && self.write.is_empty())
    }

    /// Whether the use is a depth/stencil attachment's alone: its reads and writes are the
    /// attachment's, in the stages of the fragment tests, which load, test and store it.
    pub(super) fn is_depth_attachment(&self) -> bool {
        let stages = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let both = self.read.union(self.write);
        stages.contains(both.stages) && access.contains(both.access)
    }
}

/// One global memory barrier: every access of `src` before every access of `dst`. Without access
/// types on a side it is an execution dependency only (a write after a read).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Barrier {
    pub(super) src: Access,
    pub(super) dst: Access,
}

impl Barrier {
    /// The start of every batch with encoders on a driver whose submissions stall on semaphore
    /// operations (`command_buffer/submit.rs`): what completed batches wrote is visible to every
    /// later command. The tracker forgets a batch once it completed, so its writes are ordered by
    /// completion (its queue's timeline, observed on the host before this submission) and need
    /// only this visibility; the empty source waits for nothing.
    pub(super) fn after_completed() -> Barrier {
        Barrier {
            src: Access::NONE,
            dst: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        }
    }

    /// The end of every batch: every earlier write before the host's reads and writes, so the CPU
    /// sees the GPU's writes once the batch completes, and before the batch's signals.
    pub(super) fn to_host() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ | vk::AccessFlags2::HOST_WRITE,
            ),
        }
    }

    /// The end of every batch where batches are ordered by their ends (`batch_order.rs`):
    /// [`to_host`](Barrier::to_host)'s, and every earlier command before every access of every
    /// later command of its `VkQueue`, so later batches there need no barrier from its accesses
    /// ([`Tracker`]'s module docs).
    pub(super) fn to_all() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::HOST_READ
                    | vk::AccessFlags2::HOST_WRITE
                    | vk::AccessFlags2::MEMORY_READ
                    | vk::AccessFlags2::MEMORY_WRITE,
            ),
        }
    }

    /// Everything recorded before, before everything recorded after: a shadowed texture's copy
    /// in an encoder whose commands Metal does not order (`shadow.rs`).
    pub(super) fn everything() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        }
    }

    /// From a fill's writes (`vkCmdFillBuffer`) to the dispatches after it, which read or write
    /// what it wrote, as an indirect command too.
    pub(super) fn after_fill() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        }
    }

    /// A compute encoder's memory barrier (`memoryBarrierWithScope:` and
    /// `memoryBarrierWithResources:`): every shader write of the earlier dispatches before every
    /// shader, uniform and indirect-argument access of the later ones.
    pub(super) fn between_dispatches() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::UNIFORM_READ
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        }
    }

    /// Record the barrier on `command_buffer`: one global `VkMemoryBarrier` through
    /// `vkCmdPipelineBarrier`, with [`Barrier::legacy`]'s masks, the destination's `MEMORY_READ`
    /// named as the reads it stands for beside the destination stages
    /// (`image_barrier::MemoryReads`, the device's queue family for encoders, which records every
    /// such barrier). The runtime records every barrier
    /// this way, with or without synchronization2: each mask it makes maps to the same or a
    /// broader legacy mask, so one path is exact.
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `device` in the recording state, and no other
    /// thread records it.
    pub(super) unsafe fn record(self, device: &Vk, command_buffer: vk::CommandBuffer) {
        let masks = self.legacy();
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(masks.src_access)
            .dst_access_mask(device.reads().named(masks.dst_access, masks.dst_stages));
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                masks.src_stages,
                masks.dst_stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn add(&mut self, src: Access, dst: Access) {
        self.src = self.src.union(src);
        self.dst = self.dst.union(dst);
    }

    /// The barrier, when it orders anything.
    fn needed(self) -> Option<Barrier> {
        (!self.src.is_empty() && !self.dst.is_empty()).then_some(self)
    }

    /// The masks of the barrier without synchronization2 (`vkCmdPipelineBarrier` with one
    /// `VkMemoryBarrier`). The two flag families share the bits the older one defines; each newer
    /// bit maps to the older stage or access type that holds it, or to `ALL_COMMANDS` and the
    /// memory access types, which order more.
    pub(super) fn legacy(self) -> LegacyBarrier {
        LegacyBarrier {
            src_stages: legacy_stages(self.src.stages, vk::PipelineStageFlags::TOP_OF_PIPE),
            src_access: legacy_access(self.src.access),
            dst_stages: legacy_stages(self.dst.stages, vk::PipelineStageFlags::BOTTOM_OF_PIPE),
            dst_access: legacy_access(self.dst.access),
        }
    }
}

/// A [`Barrier`]'s masks for `vkCmdPipelineBarrier`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct LegacyBarrier {
    pub(super) src_stages: vk::PipelineStageFlags,
    pub(super) src_access: vk::AccessFlags,
    pub(super) dst_stages: vk::PipelineStageFlags,
    pub(super) dst_access: vk::AccessFlags,
}

impl fmt::Debug for LegacyBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LegacyBarrier {{ src: {:#x}/{:#x}, dst: {:#x}/{:#x} }}",
            self.src_stages.as_raw(),
            self.src_access.as_raw(),
            self.dst_stages.as_raw(),
            self.dst_access.as_raw()
        )
    }
}

/// The stage bits `VkPipelineStageFlags` defines, the same in both families (bits 0 to 25).
const LEGACY_STAGES: u64 = (1 << 26) - 1;
/// The access bits `VkAccessFlags` defines, the same in both families (bits 0 to 27).
const LEGACY_ACCESS: u64 = (1 << 28) - 1;

/// `stages` as `VkPipelineStageFlags` name them, each the same or a broader legacy stage, as a
/// barrier's ([`Barrier::legacy`]); `none` where `stages` is empty.
pub(super) fn legacy_stages(
    stages: vk::PipelineStageFlags2,
    none: vk::PipelineStageFlags,
) -> vk::PipelineStageFlags {
    use vk::PipelineStageFlags as Old;
    use vk::PipelineStageFlags2 as New;
    if stages.is_empty() {
        return none;
    }
    let mut old = Old::from_raw((stages.as_raw() & LEGACY_STAGES) as u32);
    let mut high = New::from_raw(stages.as_raw() & !LEGACY_STAGES);
    for (new, into) in [
        (
            New::COPY | New::RESOLVE | New::BLIT | New::CLEAR,
            Old::TRANSFER,
        ),
        (
            New::INDEX_INPUT | New::VERTEX_ATTRIBUTE_INPUT,
            Old::VERTEX_INPUT,
        ),
        // The pre-rasterization stages a device has: naming them one by one would name stages
        // whose features may be off.
        (New::PRE_RASTERIZATION_SHADERS, Old::ALL_GRAPHICS),
    ] {
        if high.intersects(new) {
            old |= into;
            high &= !new;
        }
    }
    if !high.is_empty() {
        old |= Old::ALL_COMMANDS;
    }
    old
}

fn legacy_access(access: vk::AccessFlags2) -> vk::AccessFlags {
    use vk::AccessFlags as Old;
    use vk::AccessFlags2 as New;
    let mut old = Old::from_raw((access.as_raw() & LEGACY_ACCESS) as u32);
    let mut high = New::from_raw(access.as_raw() & !LEGACY_ACCESS);
    for (new, into) in [
        (
            New::SHADER_SAMPLED_READ | New::SHADER_STORAGE_READ,
            Old::SHADER_READ,
        ),
        (New::SHADER_STORAGE_WRITE, Old::SHADER_WRITE),
    ] {
        if high.intersects(new) {
            old |= into;
            high &= !new;
        }
    }
    if !high.is_empty() {
        old |= Old::MEMORY_READ | Old::MEMORY_WRITE;
    }
    old
}

/// A batch's place in the device's submission order: the `VkQueue` it ran on (its index among
/// the device's queues), and the value that queue's timeline semaphore reaches once the batch and
/// every batch submitted before it on that queue completed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Point {
    pub(super) queue: usize,
    pub(super) value: u64,
}

/// A wait of a batch for another `VkQueue`: the batch's `stages` (and those of every later batch
/// of its queue) run once that queue's timeline semaphore reached `at.value`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Wait {
    pub(super) at: Point,
    pub(super) stages: vk::PipelineStageFlags2,
}

impl fmt::Debug for Wait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Wait {{ at: {:?}, stages: {:#x} }}",
            self.at,
            self.stages.as_raw()
        )
    }
}

impl Wait {
    /// The wait's stages for `vkQueueSubmit`'s `pWaitDstStageMask`: each maps to the same or a
    /// broader legacy stage, as a barrier's do ([`Barrier::legacy`]).
    pub(super) fn legacy_stages(self) -> vk::PipelineStageFlags {
        legacy_stages(self.stages, vk::PipelineStageFlags::ALL_COMMANDS)
    }
}

/// Add `wait` to `waits`, which hold at most one wait per queue: the latest value, for every
/// stage either names.
fn add_wait(waits: &mut Vec<Wait>, wait: Wait) {
    if wait.stages.is_empty() {
        return;
    }
    match waits
        .iter_mut()
        .find(|known| known.at.queue == wait.at.queue)
    {
        Some(known) => {
            known.at.value = known.at.value.max(wait.at.value);
            known.stages |= wait.stages;
        }
        None => waits.push(wait),
    }
}

/// What a use needs ordered before it: a barrier on its own queue, and waits for other queues.
#[derive(Default)]
struct Needs {
    barrier: Barrier,
    waits: Vec<Wait>,
}

/// A short list, its first value inline: of one value it allocates nothing, nor does a copy of it
/// (the module docs). Kept one per `VkQueue` (an entry's writers and reads, a writer's waits), so
/// on a device of one `VkQueue` those never allocate; and one per range (a resource's uses in an
/// encoder's part) and per part (an encoder's parts, one but where it splits). It keeps the order
/// values were pushed in, as a `Vec` would.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq))]
struct Few<T> {
    /// The first value; `None` only while `rest` is empty.
    first: Option<T>,
    rest: Vec<T>,
}

impl<T> Default for Few<T> {
    fn default() -> Few<T> {
        Few {
            first: None,
            rest: Vec::new(),
        }
    }
}

impl<T> Few<T> {
    /// A list of `value` alone.
    fn of(value: T) -> Few<T> {
        Few {
            first: Some(value),
            rest: Vec::new(),
        }
    }

    /// The last value, to change.
    fn last_mut(&mut self) -> Option<&mut T> {
        self.rest.last_mut().or(self.first.as_mut())
    }

    fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    /// Add `value` last.
    fn push(&mut self, value: T) {
        match self.first {
            None => self.first = Some(value),
            Some(_) => self.rest.push(value),
        }
    }

    /// [`push`](Self::push) `value`, the room for `room` values past the first made at once
    /// where it holds one value alone, so a list that grows to `room` allocates once.
    fn push_with_room(&mut self, value: T, room: usize) {
        if self.first.is_some() && self.rest.capacity() == 0 {
            self.rest.reserve_exact(room);
        }
        self.push(value);
    }

    fn iter(&self) -> std::iter::Chain<std::option::Iter<'_, T>, std::slice::Iter<'_, T>> {
        self.first.iter().chain(&self.rest)
    }

    fn iter_mut(
        &mut self,
    ) -> std::iter::Chain<std::option::IterMut<'_, T>, std::slice::IterMut<'_, T>> {
        self.first.iter_mut().chain(&mut self.rest)
    }

    /// Keep the values `keep` holds for, in order.
    fn retain_mut(&mut self, mut keep: impl FnMut(&mut T) -> bool) {
        if !self.first.as_mut().is_some_and(&mut keep) {
            self.first = None;
        }
        self.rest.retain_mut(keep);
        if self.first.is_none() && !self.rest.is_empty() {
            self.first = Some(self.rest.remove(0));
        }
    }

    fn len(&self) -> usize {
        usize::from(self.first.is_some()) + self.rest.len()
    }

    /// Change the values as `change` changes them as one list, in order.
    fn change_all(&mut self, change: impl FnOnce(&mut Vec<T>)) {
        if let Some(first) = self.first.take() {
            self.rest.insert(0, first);
        }
        change(&mut self.rest);
        if !self.rest.is_empty() {
            self.first = Some(self.rest.remove(0));
        }
    }
}

impl<'a, T> IntoIterator for &'a Few<T> {
    type Item = &'a T;
    type IntoIter = std::iter::Chain<std::option::Iter<'a, T>, std::slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> IntoIterator for Few<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<std::option::IntoIter<T>, std::vec::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.first.into_iter().chain(self.rest)
    }
}

/// One queue's last write over a range.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq))]
struct Writer {
    /// The batch that wrote.
    at: Point,
    /// The stages and access types of the writer's own queue the write is already visible to.
    visible: Access,
    /// The other queues that waited for the write, with the stages their waits cover: a timeline
    /// wait makes the write visible to every access of those stages.
    waited: Few<(usize, vk::PipelineStageFlags2)>,
}

impl Writer {
    /// The stages of `queue` a wait already ordered after the write.
    fn waited_by(&self, queue: usize) -> vk::PipelineStageFlags2 {
        self.waited
            .iter()
            .find(|(by, _)| *by == queue)
            .map_or(vk::PipelineStageFlags2::NONE, |&(_, stages)| stages)
    }

    /// Merge the write of the same queue in `other` into this one: what both are visible to.
    fn absorb(&mut self, other: &Writer) {
        self.at.value = self.at.value.max(other.at.value);
        self.visible = self.visible.intersection(other.visible);
        self.waited.retain_mut(|(queue, stages)| {
            *stages &= other.waited_by(*queue);
            !stages.is_empty()
        });
    }
}

/// One queue's reads of a range since the last write: their stages, and the latest batch.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
struct Reads {
    at: Point,
    stages: vk::PipelineStageFlags2,
}

/// What the state knows of one range of one resource.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq))]
struct Entry {
    range: UseRange,
    /// The access types and stages of the last writes, empty when none is known.
    write: Access,
    /// The last write on each queue that wrote the range, one per queue: one queue's unless
    /// entries were merged.
    writers: Few<Writer>,
    /// The reads since the last write, one per queue.
    reads: Few<Reads>,
}

impl Entry {
    /// The entry a use makes in a batch at `at`.
    fn of<K>(used: &Use<K>, at: Point) -> Entry {
        let mut entry = Entry {
            range: used.range.clone(),
            write: used.write,
            writers: Few::default(),
            reads: Few::default(),
        };
        if !used.write.is_empty() {
            entry.writers.push(Writer {
                at,
                visible: Access::NONE,
                waited: Few::default(),
            });
        }
        entry.add_reads(at, used.read.stages);
        entry
    }

    /// What a use at `at` of this entry's range needs ordered before it, where the accesses of
    /// its queue's batches below `since` are ordered by their ends (0 where none is).
    fn depend(&self, read: Access, write: Access, (at, since): (Point, u64), needs: &mut Needs) {
        if !read.is_empty() {
            self.after_writes(read, (at, since), needs);
        }
        if !write.is_empty() {
            self.after_writes(write, (at, since), needs);
            for reads in self.reads.iter() {
                if reads.at.queue == at.queue {
                    // Reads of a batch ordered by its end need nothing.
                    if reads.at.value >= since {
                        needs.barrier.add(
                            Access::new(reads.stages, vk::AccessFlags2::NONE),
                            Access::new(write.stages, vk::AccessFlags2::NONE),
                        );
                    }
                } else {
                    add_wait(
                        &mut needs.waits,
                        Wait {
                            at: reads.at,
                            stages: write.stages,
                        },
                    );
                }
            }
        }
    }

    /// What `access` at `at` needs after the last writes: a barrier from a write of its own
    /// queue not yet visible to it (a write of a batch below `since` its batch's end made
    /// visible), and a wait for a write of another queue not yet waited for.
    fn after_writes(&self, access: Access, (at, since): (Point, u64), needs: &mut Needs) {
        for writer in self.writers.iter() {
            if writer.at.queue == at.queue {
                if writer.at.value >= since && !writer.visible.covers(access) {
                    needs.barrier.add(self.write, access);
                }
            } else if !writer.waited_by(at.queue).contains(access.stages) {
                add_wait(
                    &mut needs.waits,
                    Wait {
                        at: writer.at,
                        stages: access.stages,
                    },
                );
            }
        }
    }

    /// Note what a batch at `at` ordered after the writes before its use: `barrier`, recorded on
    /// `at.queue`, and the batch's `waits` for other queues.
    fn ordered(&mut self, barrier: Option<Barrier>, waits: &[Wait], at: Point) {
        let write = self.write;
        for writer in self.writers.iter_mut() {
            if writer.at.queue == at.queue {
                if let Some(barrier) = barrier {
                    if barrier.src.covers(write) {
                        writer.visible = writer.visible.union(barrier.dst);
                    }
                }
            } else if let Some(wait) = waits
                .iter()
                .find(|wait| wait.at.queue == writer.at.queue && wait.at.value >= writer.at.value)
            {
                match writer.waited.iter_mut().find(|(by, _)| *by == at.queue) {
                    Some((_, stages)) => *stages |= wait.stages,
                    None => writer.waited.push((at.queue, wait.stages)),
                }
            }
        }
    }

    /// Add reads in `stages` at `at`.
    fn add_reads(&mut self, at: Point, stages: vk::PipelineStageFlags2) {
        if stages.is_empty() {
            return;
        }
        match self
            .reads
            .iter_mut()
            .find(|reads| reads.at.queue == at.queue)
        {
            Some(reads) => {
                reads.at.value = reads.at.value.max(at.value);
                reads.stages |= stages;
            }
            None => self.reads.push(Reads { at, stages }),
        }
    }

    /// Merge `other` into this entry over `range` (one that holds both): it orders after both.
    fn absorb(&mut self, other: Entry, range: UseRange) {
        self.range = range;
        self.write = self.write.union(other.write);
        for writer in other.writers {
            match self
                .writers
                .iter_mut()
                .find(|mine| mine.at.queue == writer.at.queue)
            {
                Some(mine) => mine.absorb(&writer),
                None => self.writers.push(writer),
            }
        }
        for reads in other.reads {
            self.add_reads(reads.at, reads.stages);
        }
    }

    /// Forget the writes and reads of the batches `completed` holds, by queue: the host saw them
    /// complete before any later submission. Whether nothing is left.
    fn retire(&mut self, completed: &[u64]) -> bool {
        let done = |at: Point| {
            completed
                .get(at.queue)
                .is_some_and(|&through| at.value <= through)
        };
        self.writers.retain_mut(|writer| !done(writer.at));
        if self.writers.is_empty() {
            self.write = Access::NONE;
        }
        self.reads.retain_mut(|reads| !done(reads.at));
        self.writers.is_empty() && self.reads.is_empty()
    }
}

/// Where entries of the hazard state are kept by resource: an encoder's map, the tracker's, or a
/// plan's copies ([`Overlay`]).
trait Entries<S> {
    /// The entries of `key`, where it has some.
    fn of(&self, key: &S) -> Option<&Vec<Entry>>;

    /// The entries of `key`, none at first.
    fn of_mut(&mut self, key: S) -> &mut Vec<Entry>;
}

impl<S: Eq + Hash, H: BuildHasher> Entries<S> for HashMap<S, Vec<Entry>, H> {
    fn of(&self, key: &S) -> Option<&Vec<Entry>> {
        self.get(key)
    }

    fn of_mut(&mut self, key: S) -> &mut Vec<Entry> {
        self.entry(key).or_default()
    }
}

/// What the uses (sorted by key, one per key and range) at `at` need after the accesses in
/// `state`, those of its queue's batches below `since` ordered by their ends.
fn depend<S>(state: &impl Entries<S>, uses: &[Use<S>], (at, since): (Point, u64)) -> Needs {
    let mut needs = Needs::default();
    for used in uses {
        for entry in state.of(&used.key).into_iter().flatten() {
            if entry.range.overlaps(&used.range) {
                entry.depend(used.read, used.write, (at, since), &mut needs);
            }
        }
    }
    needs
}

/// Add the uses (sorted by key, one per key and range) of a batch at `at` to `state`, after
/// `barrier` was recorded before them and with the batch's `waits`.
fn admit<S: Copy + Eq>(
    state: &mut impl Entries<S>,
    uses: &[Use<S>],
    barrier: Option<Barrier>,
    waits: &[Wait],
    at: Point,
) {
    for group in uses.chunk_by(|a, b| a.key == b.key) {
        let entries = state.of_mut(group[0].key);
        for entry in entries.iter_mut() {
            entry.ordered(barrier, waits, at);
        }
        // The older entries a write of the group covers are ordered before it.
        entries.retain(|entry| {
            !group
                .iter()
                .any(|used| !used.write.is_empty() && used.range.contains(&entry.range))
        });
        for used in group {
            // An older entry of the same range remains only when this use does not write.
            match entries.iter_mut().find(|entry| entry.range == used.range) {
                Some(entry) => entry.add_reads(at, used.read.stages),
                None => entries.push(Entry::of(used, at)),
            }
        }
        if entries.len() > ENTRIES_PER_KEY {
            coalesce(entries);
        }
    }
}

/// Merge a resource's entries into one per kind of range.
fn coalesce(entries: &mut Vec<Entry>) {
    let mut first = 0;
    while first < entries.len() {
        let mut next = first + 1;
        while next < entries.len() {
            match entries[first].range.hull(&entries[next].range) {
                Some(range) => {
                    let other = entries.swap_remove(next);
                    entries[first].absorb(other, range);
                }
                None => next += 1,
            }
        }
        first += 1;
    }
}

/// `uses` (none of which touches nothing) one per key and range, sorted by key, each key's ranges
/// in the order they first came, the reads and writes of a range's uses joined: in place, in their
/// own storage.
pub(super) fn merge<S: Copy + Ord>(uses: &mut Vec<Use<S>>) {
    // Stable: a key's uses keep their order.
    uses.sort_by_key(|used| used.key);
    // `uses[..kept]` are merged, `uses[group..kept]` those of the key at hand.
    let mut kept = 0;
    let mut group = 0;
    for at in 0..uses.len() {
        if group < kept && uses[group].key != uses[at].key {
            group = kept;
        }
        match (group..kept).find(|&k| uses[k].range == uses[at].range) {
            Some(k) => {
                let (read, write) = (uses[at].read, uses[at].write);
                uses[k].read = uses[k].read.union(read);
                uses[k].write = uses[k].write.union(write);
            }
            None => {
                uses.swap(kept, at);
                kept += 1;
            }
        }
    }
    uses.truncate(kept);
}

/// How far a command's uses say which untracked memory it touches, inside an encoder Metal orders
/// (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reach {
    /// Its uses name every byte of untracked memory it touches, each by the key and range that
    /// byte takes wherever it is named: a blit's command, a shadow's copy. It orders per key and
    /// range after the earlier commands like it, and after every untracked access of the others.
    Named,
    /// It may touch untracked memory its uses do not name (a shader's, through addresses,
    /// argument buffers and resident resources): it orders after every untracked access, as one
    /// class, and every later command orders after its untracked accesses.
    Unnamed,
}

/// The key a use takes inside an encoder: its key's ([`Class::Key`]) when tracked or of a
/// [`Reach::Named`] command; otherwise, and for an untracked use besides, a class of untracked
/// accesses over all untracked memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Class<K> {
    /// A tracked use, or an untracked use of a [`Reach::Named`] command, at its key and range:
    /// one key's bytes are the same memory whichever use names them (an untracked heap's use
    /// carries the key of a tracked buffer over the same host pages, `host_aliases.rs`).
    Key(K),
    /// Every untracked access of the encoder's commands, what a [`Reach::Unnamed`] command orders
    /// after.
    Untracked,
    /// The untracked accesses of [`Reach::Unnamed`] commands, what a [`Reach::Named`] command
    /// orders after.
    Unnamed,
}

/// The uses of one encoder: the order of its commands, and the summaries its commit reads, one
/// per part. An encoder is one part until it executes commands the GPU wrote: those run once the
/// work before them completed, so the encoder's commands before and after them go to batches of
/// their own, each with its own summary ([`EncoderUses::split`]).
pub(super) struct EncoderUses<K> {
    /// Whether Metal runs the commands in encoded order (blit encoders, serial compute encoders).
    ordered: bool,
    /// The accesses of the current part's commands so far, for that order.
    inside: KeyMap<Class<K>, Vec<Entry>>,
    /// The parts, in order: the last is the current one. Never empty; the first held in place, so
    /// an encoder of one part allocates none for them.
    parts: Few<Part<K>>,
    /// The fences waited for: before the encoder's first part.
    waits: Vec<FenceUse<K>>,
    /// The fences updated: after its last part.
    updates: Vec<FenceUse<K>>,
    /// The images it holds in their attachment layout (`texture/held_depth.rs`).
    held: Vec<K>,
    /// A use of a held image but as a depth/stencil attachment was recorded after its hold.
    broke: bool,
    /// The last command's uses as the encoder orders them ([`inside_uses`]): to compare with the
    /// earlier commands', and as it keeps them where those differ (a command with an untracked
    /// use). Each command's are written over the last one's storage.
    compared: Vec<Use<Class<K>>>,
    kept: Vec<Use<Class<K>>>,
}

/// A fence an encoder waits for or updates, and the stages of its draws the call names: a render
/// encoder's `waitForFence:beforeStages:` and `updateFence:afterStages:` name render stages
/// (`fence.rs`), which `draws` holds as the stages of `render::PASS_DRAW_STAGES` they run in;
/// `None` covers every stage of the encoder (a compute or blit encoder's call). Either way the
/// use covers the encoder's stages outside its draws (the runtime's own copies and dispatches
/// before and after its render pass instances), which only orders more.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct FenceUse<K> {
    fence: K,
    draws: Option<vk::PipelineStageFlags2>,
}

impl<K> FenceUse<K> {
    /// The stages, of an encoder whose commands ran in `stages`, the use covers: those of its
    /// draws it names and every one outside its draws.
    fn covered(&self, stages: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
        match self.draws {
            None => stages,
            Some(draws) => {
                let pass = super::render::PASS_DRAW_STAGES;
                stages & (draws | vk::PipelineStageFlags2::from_raw(!pass.as_raw()))
            }
        }
    }
}

/// Add `used` to `fences`, which hold each fence once: a fence named twice covers the stages of
/// both calls.
fn add_fence<K: PartialEq>(fences: &mut Vec<FenceUse<K>>, used: FenceUse<K>) {
    match fences.iter_mut().find(|known| known.fence == used.fence) {
        Some(known) => {
            known.draws = match (known.draws, used.draws) {
                (Some(a), Some(b)) => Some(a | b),
                _ => None,
            }
        }
        None => fences.push(used),
    }
}

/// The uses of one part of an encoder.
struct Part<K> {
    /// The tracked uses so far, one per range, at most [`ENTRIES_PER_KEY`] per resource: most
    /// resources have one, held inline ([`Few`]), so a command's first use of a resource
    /// allocates nothing of its own.
    tracked: KeyMap<K, Few<Use<K>>>,
    stages: vk::PipelineStageFlags2,
}

impl<K> Part<K> {
    fn new() -> Part<K> {
        Part {
            tracked: KeyMap::default(),
            stages: vk::PipelineStageFlags2::NONE,
        }
    }
}

impl<K: Copy + Ord + Hash> EncoderUses<K> {
    /// The uses of a new encoder, `ordered` when Metal runs its commands in encoded order.
    pub(super) fn new(ordered: bool) -> EncoderUses<K> {
        EncoderUses {
            ordered,
            inside: KeyMap::default(),
            parts: Few::of(Part::new()),
            waits: Vec::new(),
            updates: Vec::new(),
            held: Vec::new(),
            broke: false,
            compared: Vec::new(),
            kept: Vec::new(),
        }
    }

    /// Whether Metal runs the encoder's commands in encoded order.
    pub(super) fn ordered(&self) -> bool {
        self.ordered
    }

    /// The barrier a command with `uses`, which say what untracked memory it touches as `reach`
    /// does, needs after the encoder's earlier commands: none in an encoder whose commands are
    /// not ordered. Nothing the encoder orders by changes until [`EncoderUses::order`].
    pub(super) fn barrier(&mut self, uses: &[Use<K>], reach: Reach) -> Option<Barrier> {
        if !self.ordered {
            return None;
        }
        inside_uses(uses, reach, false, &mut self.compared);
        // One encoder runs on one queue: its commands need barriers, never waits.
        depend(&self.inside, &self.compared, (Point::default(), 0))
            .barrier
            .needed()
    }

    /// Order a command whose `uses` say what untracked memory it touches as `reach` does after the
    /// encoder's earlier commands, and note it recorded, in `stages`, after the barrier it needs
    /// ([`barrier`](Self::barrier)), which this answers. The command's uses as the encoder compares
    /// them are those it keeps where every use is tracked ([`inside_uses`]), so they are made once.
    pub(super) fn order(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<K>],
        reach: Reach,
    ) -> Option<Barrier> {
        let barrier = self.barrier(uses, reach);
        if self.ordered {
            let kept = match uses.iter().all(|used| used.tracked) {
                true => &self.compared,
                false => {
                    inside_uses(uses, reach, true, &mut self.kept);
                    &self.kept
                }
            };
            admit(&mut self.inside, kept, barrier, &[], Point::default());
        }
        self.note(stages, uses);
        barrier
    }

    /// Note a recorded command that may touch untracked memory its uses do not name
    /// ([`Reach::Unnamed`]): it ran in `stages`, with `uses`, after `barrier`.
    pub(super) fn record(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<K>],
        barrier: Option<Barrier>,
    ) {
        if self.ordered {
            inside_uses(uses, Reach::Unnamed, true, &mut self.kept);
            admit(&mut self.inside, &self.kept, barrier, &[], Point::default());
        }
        self.note(stages, uses);
    }

    /// Note a command the encoder orders recorded, in `stages`, with `uses`: its held images'
    /// holds, its part's stages and tracked uses.
    fn note(&mut self, stages: vk::PipelineStageFlags2, uses: &[Use<K>]) {
        if !self.held.is_empty() {
            let held = &self.held;
            self.broke |= uses.iter().any(|used| {
                held.contains(&used.key) && !used.is_nothing() && !used.is_depth_attachment()
            });
        }
        let part = self.current();
        // Every stage, untracked work's too: a fence wait orders all of the encoder's work.
        part.stages |= uses.iter().fold(stages, |all, used| {
            all | used.read.stages | used.write.stages
        });
        // Room for a few resources at once, made at the part's first: grown from nothing, the map
        // would allocate again at its fourth and eighth.
        if part.tracked.capacity() == 0 {
            part.tracked.reserve(TRACKED_ROOM);
        }
        for used in uses
            .iter()
            .filter(|used| used.tracked && !used.is_nothing())
        {
            let ranges = part.tracked.entry(used.key).or_default();
            match ranges.iter_mut().find(|known| known.range == used.range) {
                Some(known) => {
                    known.read = known.read.union(used.read);
                    known.write = known.write.union(used.write);
                }
                // A resource used at a second range (a buffer whose draws move along it) takes
                // room for every range it keeps before they merge, and the merge's own, at once.
                None => ranges.push_with_room(used.clone(), ENTRIES_PER_KEY + 1),
            }
            if ranges.len() > ENTRIES_PER_KEY {
                ranges.change_all(coalesce_uses);
            }
        }
    }

    /// Whether a command the encoder recorded so far writes memory one of `uses` reads: an
    /// overlapping range of a tracked resource. A command recorded before the encoder's earlier
    /// commands run (a render encoder's index bounds pass, recorded before its pass) would read
    /// it before that write. Untracked resources are not recorded, so they are not seen.
    pub(super) fn written(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|read| read.tracked && !read.read.is_empty())
            .any(|read| {
                self.parts.iter().any(|part| {
                    part.tracked.get(&read.key).is_some_and(|known| {
                        known
                            .iter()
                            .any(|k| !k.write.is_empty() && k.range.overlaps(&read.range))
                    })
                })
            })
    }

    /// Whether a command the encoder recorded so far reads or writes memory one of `uses`
    /// writes: an overlapping range of a tracked resource. A command recorded before the
    /// encoder's earlier commands run (a render encoder's emulated mesh dispatch, recorded before
    /// its draw's render pass instance at the latest) would write it before that access.
    pub(super) fn accessed(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|write| write.tracked && !write.write.is_empty())
            .any(|write| {
                self.parts.iter().any(|part| {
                    part.tracked
                        .get(&write.key)
                        .is_some_and(|known| known.iter().any(|k| k.range.overlaps(&write.range)))
                })
            })
    }

    /// The tracked uses of the commands recorded so far that ran in one of `stages`.
    pub(super) fn tracked_in(&self, stages: vk::PipelineStageFlags2) -> Vec<Use<K>> {
        self.parts
            .iter()
            .flat_map(|part| part.tracked.values().flatten())
            .filter(|known| (known.read.stages | known.write.stages).intersects(stages))
            .cloned()
            .collect()
    }

    /// Whether a command the encoder recorded so far touches, in one of `stages`, memory one of
    /// `uses` touches where either writes ([`touches`]).
    pub(super) fn touched_in(&self, uses: &[Use<K>], stages: vk::PipelineStageFlags2) -> bool {
        uses.iter().any(|used| {
            self.parts.iter().any(|part| {
                part.tracked
                    .get(&used.key)
                    .is_some_and(|known| known.iter().any(|k| touches(k, used, stages)))
            })
        })
    }

    /// `wait_for_fence`: the encoder runs after the fence's last update in commit order before
    /// it. Fences are waited for at the start of the encoder and updated at its end, so a wait
    /// after the encoder updated the fence waits for the update before the encoder, as plain
    /// Metal's compute, blit and render encoders do (an M4 read the earlier encoder's write in
    /// three runs of each); Metal's API validation asserts. Before the draw stages `draws` alone
    /// where `Some` ([`FenceUse`]): the encoder's draws in those stages, and its commands outside
    /// its draws, run after the update; its draws' other stages may run before it.
    pub(super) fn wait_for_fence(&mut self, fence: K, draws: Option<vk::PipelineStageFlags2>) {
        add_fence(&mut self.waits, FenceUse { fence, draws });
    }

    /// Whether the encoder waits for or updates a fence.
    pub(super) fn fenced(&self) -> bool {
        !self.waits.is_empty() || !self.updates.is_empty()
    }

    /// `update_fence`: the fence is updated after the encoder's work; after the draw stages
    /// `draws` alone where `Some` ([`FenceUse`]): a later wait orders after the encoder's draws in
    /// those stages and its commands outside its draws, not after its draws' other stages.
    pub(super) fn update_fence(&mut self, fence: K, draws: Option<vk::PipelineStageFlags2>) {
        add_fence(&mut self.updates, FenceUse { fence, draws });
    }

    /// Hold the image known by `key` in its attachment layout through the encoder
    /// (`texture/held_depth.rs`): a use of it recorded from now on but as a depth/stencil
    /// attachment breaks the hold ([`broke_hold`](Self::broke_hold)).
    pub(super) fn hold(&mut self, key: K) {
        if !self.held.contains(&key) {
            self.held.push(key);
        }
    }

    /// The images the encoder holds in their attachment layout.
    pub(super) fn held(&self) -> &[K] {
        &self.held
    }

    /// How many parts the encoder's commands were recorded in so far: one more at each
    /// [`split`](Self::split).
    pub(super) fn parts(&self) -> usize {
        self.parts.len()
    }

    /// Whether a use of a held image but as a depth/stencil attachment was recorded after its
    /// hold: the encoder's commands would find it in another layout than they were recorded for.
    pub(super) fn broke_hold(&self) -> bool {
        self.broke
    }

    /// The part commands are recorded into now.
    fn current(&mut self) -> &mut Part<K> {
        self.parts.last_mut().expect("an encoder has a part")
    }

    /// End the current part: the commands after this run in a part of their own, after
    /// everything before it completed, so they need no barrier after the commands before.
    pub(super) fn split(&mut self) {
        self.inside.clear();
        self.parts.push(Part::new());
    }

    /// Note a [`Barrier::everything`] recorded after the commands so far: every stage's every
    /// access before it happens before every later command's, so a later command needs no barrier
    /// after them. The part goes on: its summary keeps their uses.
    pub(super) fn ordered_everything(&mut self) {
        self.inside.clear();
    }

    /// The summary of the ended encoder, its parts joined (an encoder of one part: its part's).
    pub(super) fn finish(self) -> Summary<K> {
        self.finish_parts()
            .into_iter()
            .reduce(|mut summary, part| {
                summary.join(part);
                summary
            })
            .expect("an encoder has a part")
    }

    /// The summaries of the ended encoder's parts, in order: the fence waits go with the first,
    /// and the updates with the last.
    pub(super) fn finish_parts(self) -> Vec<Summary<K>> {
        let EncoderUses {
            parts,
            waits,
            updates,
            ..
        } = self;
        let mut summaries: Vec<Summary<K>> = parts
            .into_iter()
            .map(|part| {
                // By key, each resource's uses in the order they were recorded, as a stable sort
                // of the uses by key leaves them: a resource's uses are one list, so ordering
                // the resources orders them, and only their keys, each once, are sorted, each
                // beside its list where the part holds it (`by_key`).
                // Sized once: a flattened iterator tells no length, which would grow the vector
                // use by use.
                let count = part.tracked.values().map(Few::len).sum();
                let mut uses: Vec<Use<K>> = Vec::with_capacity(count);
                by_key(&part.tracked, |few| uses.extend(few.iter().cloned()));
                Summary {
                    uses,
                    stages: part.stages,
                    waits: Vec::new(),
                    updates: Vec::new(),
                }
            })
            .collect();
        if let Some(first) = summaries.first_mut() {
            first.waits = waits;
        }
        if let Some(last) = summaries.last_mut() {
            last.updates = updates;
        }
        summaries
    }
}

/// The most resources of an encoder's part whose keys [`by_key`] orders on the stack: an encoder
/// that names no more takes no allocation to order them.
const ORDERED_INLINE: usize = 16;

/// Each of `tracked`'s lists, in the order of their keys. Up to [`ORDERED_INLINE`] keys are
/// sorted in place on the stack, more in a vector of their own.
fn by_key<K: Copy + Ord, V, S>(tracked: &HashMap<K, V, S>, mut each: impl FnMut(&V)) {
    if tracked.len() <= ORDERED_INLINE {
        let mut inline: [Option<(K, &V)>; ORDERED_INLINE] = [None; ORDERED_INLINE];
        for (slot, (&key, value)) in inline.iter_mut().zip(tracked) {
            *slot = Some((key, value));
        }
        let order = &mut inline[..tracked.len()];
        order.sort_unstable_by_key(|entry| entry.map(|(key, _)| key));
        order.iter().flatten().for_each(|&(_, value)| each(value));
        return;
    }
    let mut order: Vec<(K, &V)> = tracked.iter().map(|(&key, value)| (key, value)).collect();
    order.sort_unstable_by_key(|&(key, _)| key);
    order.iter().for_each(|&(_, value)| each(value));
}

impl<K> fmt::Debug for EncoderUses<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncoderUses")
            .field("ordered", &self.ordered)
            .field("parts", &self.parts.len())
            .field("waits", &self.waits.len())
            .field("updates", &self.updates.len())
            .finish_non_exhaustive()
    }
}

/// The tracked writes of an unordered encoder's commands since its last memory barrier (a render
/// encoder's draws and tile dispatches, `render.rs`): the writes no barrier orders before the
/// commands after them. One entry per range, at most [`ENTRIES_PER_KEY`] per resource, merged
/// as [`EncoderUses`] merges its own, which claims more and stays exact.
pub(super) struct UnorderedWrites<K> {
    writes: KeyMap<K, Vec<Use<K>>>,
    /// The identity and count of the uses of what the encoder made resident the last
    /// [`add_draw`](Self::add_draw) since the last [`clear`](Self::clear) began with
    /// (`binding::ResidentPrefix`): each of their writes is held, merged or not, until it clears.
    added: Option<(u64, usize)>,
}

impl<K: Copy + Ord + Hash> UnorderedWrites<K> {
    /// No writes.
    pub(super) fn new() -> UnorderedWrites<K> {
        UnorderedWrites {
            writes: KeyMap::default(),
            added: None,
        }
    }

    /// Whether [`add`](Self::add) adds `used`: a tracked write of something.
    fn adds(used: &Use<K>) -> bool {
        used.tracked && !used.write.is_empty() && !used.range.is_empty()
    }

    /// Add the tracked writes of a command with `uses`.
    pub(super) fn add(&mut self, uses: &[Use<K>]) {
        for used in uses.iter().filter(|used| Self::adds(used)) {
            let ranges = self.writes.entry(used.key).or_default();
            ranges.push(Use {
                read: Access::NONE,
                ..used.clone()
            });
            if ranges.len() > ENTRIES_PER_KEY {
                coalesce_uses(ranges);
            }
        }
    }

    /// [`add`](Self::add) a draw's `uses`, which begin with the uses of what its encoder made
    /// resident of the identity `prefix`: those not again where the last call of this since the
    /// last [`clear`](Self::clear) began with the same, whose writes it holds as they were added
    /// (two draws' uses of one identity begin with the same uses, each of the same key, range,
    /// stages and access types: `binding::ResidentPrefix`). Whether it walked them.
    pub(super) fn add_draw(&mut self, (uses, prefix): (&[Use<K>], ResidentPrefix)) -> bool {
        let now = Some((prefix.id, prefix.len));
        let past = match self.added == now {
            true => prefix.len,
            false => 0,
        };
        #[cfg(debug_assertions)]
        for used in &uses[..past] {
            assert!(
                self.holds(used),
                "an encoder added a draw's writes by their identity less than its uses"
            );
        }
        self.add(&uses[past..]);
        self.added = now;
        past == 0
    }

    /// Whether it holds `used`'s write, where [`add`](Self::add) adds it: adding it again changes
    /// nothing a query reads (builds with debug assertions: [`add_draw`](Self::add_draw)'s
    /// check).
    #[cfg(debug_assertions)]
    fn holds(&self, used: &Use<K>) -> bool {
        let held =
            |known: &Use<K>| known.range.contains(&used.range) && known.write.covers(used.write);
        !Self::adds(used)
            || (self.writes.get(&used.key)).is_some_and(|known| known.iter().any(held))
    }

    /// A memory barrier orders every write so far before the commands after it.
    pub(super) fn clear(&mut self) {
        self.writes.clear();
        self.added = None;
    }

    /// Whether one of the writes overlaps memory one of `uses` reads, as
    /// [`EncoderUses::written`] asks of every write of the encoder.
    pub(super) fn written(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|read| read.tracked && !read.read.is_empty())
            .any(|read| {
                self.writes
                    .get(&read.key)
                    .is_some_and(|known| known.iter().any(|k| k.range.overlaps(&read.range)))
            })
    }
}

/// Uses of earlier commands kept by resource and range, tracked or not alike: one entry per range,
/// at most [`ENTRIES_PER_KEY`] per resource, merged as [`UnorderedWrites`] merges its own, which
/// claims more and stays exact.
#[derive(Clone)]
struct KeptUses<K> {
    uses: KeyMap<K, Vec<Use<K>>>,
}

impl<K: Copy + Ord + Hash> KeptUses<K> {
    fn new() -> KeptUses<K> {
        KeptUses {
            uses: KeyMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.uses.is_empty()
    }

    fn push(&mut self, used: Use<K>) {
        if used.range.is_empty() {
            return;
        }
        let ranges = self.uses.entry(used.key).or_default();
        ranges.push(used);
        if ranges.len() > ENTRIES_PER_KEY {
            coalesce_uses(ranges);
        }
    }

    /// Whether one of the kept uses of `used`'s resource is `held` (builds with debug assertions:
    /// the checks of what adds a draw's uses by their identity).
    #[cfg(debug_assertions)]
    fn holds(&self, used: &Use<K>, held: impl Fn(&Use<K>) -> bool) -> bool {
        (self.uses.get(&used.key)).is_some_and(|known| known.iter().any(held))
    }

    /// The kept uses of `used`'s resource whose ranges overlap its range.
    fn meeting<'a>(&'a self, used: &'a Use<K>) -> impl Iterator<Item = &'a Use<K>> + 'a {
        let known = self.uses.get(&used.key).map(Vec::as_slice).unwrap_or(&[]);
        known
            .iter()
            .filter(move |known| known.range.overlaps(&used.range))
    }
}

/// The memory a render pass instance's draws read as their indirect arguments (in
/// `DRAW_INDIRECT`) since the instance began (`render.rs`), tracked or not alike: Metal reads a
/// draw's arguments before a later draw of the encoder writes them, behind no barrier (measured),
/// whatever the resource's hazard tracking, and Vulkan orders the two only by a dependency.
#[derive(Clone)]
pub(super) struct IndirectReads<K> {
    reads: KeptUses<K>,
    /// The identity and count of the uses of what the encoder made resident the last
    /// [`add_draw`](Self::add_draw) began with (`binding::ResidentPrefix`): each of their reads is
    /// held, merged or not, while the instance lasts.
    added: Option<(u64, usize)>,
}

impl<K: Copy + Ord + Hash> IndirectReads<K> {
    /// No reads.
    pub(super) fn new() -> IndirectReads<K> {
        IndirectReads {
            reads: KeptUses::new(),
            added: None,
        }
    }

    /// Whether it holds no read.
    pub(super) fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }

    /// Whether [`add`](Self::add) adds `used`: a read of indirect arguments.
    fn adds(used: &Use<K>) -> bool {
        (used.read.stages).intersects(vk::PipelineStageFlags2::DRAW_INDIRECT)
    }

    /// Add the indirect argument reads of a draw with `uses`.
    pub(super) fn add(&mut self, uses: &[Use<K>]) {
        for used in uses.iter().filter(|used| Self::adds(used)) {
            self.reads.push(Use {
                write: Access::NONE,
                ..used.clone()
            });
        }
    }

    /// [`add`](Self::add) a draw's `uses`, which begin with the uses of what its encoder made
    /// resident of the identity `prefix`: those not again where the last call of this began with
    /// the same, whose reads it holds as they were added (two draws' uses of one identity begin
    /// with the same uses, each of the same key, range, stages and access types:
    /// `binding::ResidentPrefix`). Whether it walked them.
    pub(super) fn add_draw(&mut self, (uses, prefix): (&[Use<K>], ResidentPrefix)) -> bool {
        let now = Some((prefix.id, prefix.len));
        let past = match self.added == now {
            true => prefix.len,
            false => 0,
        };
        #[cfg(debug_assertions)]
        for used in &uses[..past] {
            let held =
                |known: &Use<K>| known.range.contains(&used.range) && known.read.covers(used.read);
            assert!(
                !Self::adds(used) || self.reads.holds(used, held),
                "an instance added a draw's indirect reads by their identity less than its uses"
            );
        }
        self.add(&uses[past..]);
        self.added = now;
        past == 0
    }

    /// The stages in which a later draw with `uses` writes memory one of the reads read: empty
    /// where it writes none of it.
    pub(super) fn written_in(&self, uses: &[Use<K>]) -> vk::PipelineStageFlags2 {
        uses.iter()
            .filter(|used| !used.write.is_empty() && self.reads.meeting(used).next().is_some())
            .fold(vk::PipelineStageFlags2::empty(), |stages, used| {
                stages | used.write.stages
            })
    }
}

/// The stages a draw runs before its fragments in which a shader writes memory: the vertex,
/// tessellation, geometry, object and mesh stages.
pub(super) const PRE_RASTERIZATION_WRITES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
            | vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER.as_raw()
            | vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER.as_raw()
            | vk::PipelineStageFlags2::GEOMETRY_SHADER.as_raw()
            | vk::PipelineStageFlags2::TASK_SHADER_EXT.as_raw()
            | vk::PipelineStageFlags2::MESH_SHADER_EXT.as_raw(),
    );

/// The memory a render pass instance's draws wrote in their stages before their fragments
/// ([`PRE_RASTERIZATION_WRITES`]) since the instance began (`render.rs`), tracked or not alike:
/// Metal's tile-based passes run every draw's vertex, object and mesh functions before any draw's
/// fragments, so a later draw's fragment shader reads and writes such memory after the write,
/// whatever the resource's hazard tracking, and Vulkan orders the two only by a dependency.
#[derive(Clone)]
pub(super) struct GeometryWrites<K> {
    writes: KeptUses<K>,
    /// The identity and count of the uses of what the encoder made resident the last
    /// [`add_draw`](Self::add_draw) began with (`binding::ResidentPrefix`): each of their writes
    /// before their fragments is held, merged or not, while the instance lasts.
    added: Option<(u64, usize)>,
}

impl<K: Copy + Ord + Hash> GeometryWrites<K> {
    /// No writes.
    pub(super) fn new() -> GeometryWrites<K> {
        GeometryWrites {
            writes: KeptUses::new(),
            added: None,
        }
    }

    /// Whether it holds no write.
    pub(super) fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }

    /// Add the writes before its fragments of a draw with `uses`: a use that writes in one of
    /// those stages and in the fragment stage too is taken as written before the fragments, which
    /// claims more and stays exact.
    pub(super) fn add(&mut self, uses: &[Use<K>]) {
        for used in uses {
            let Some(write) = Self::adds(used) else {
                continue;
            };
            self.writes.push(Use {
                read: Access::NONE,
                write,
                ..used.clone()
            });
        }
    }

    /// What [`add`](Self::add) adds of `used`: its write in the stages before its fragments,
    /// `None` where it writes in none.
    fn adds(used: &Use<K>) -> Option<Access> {
        let stages = used.write.stages & PRE_RASTERIZATION_WRITES;
        (!stages.is_empty()).then(|| Access::new(stages, used.write.access))
    }

    /// [`add`](Self::add) a draw's `uses`, which begin with the uses of what its encoder made
    /// resident of the identity `prefix`: those not again where the last call of this began with
    /// the same, whose writes it holds as they were added (two draws' uses of one identity begin
    /// with the same uses, each of the same key, range, stages and access types:
    /// `binding::ResidentPrefix`). Whether it walked them.
    pub(super) fn add_draw(&mut self, (uses, prefix): (&[Use<K>], ResidentPrefix)) -> bool {
        let now = Some((prefix.id, prefix.len));
        let past = match self.added == now {
            true => prefix.len,
            false => 0,
        };
        #[cfg(debug_assertions)]
        for used in &uses[..past] {
            let held = |write: Access| {
                move |known: &Use<K>| known.range.contains(&used.range) && known.write.covers(write)
            };
            assert!(
                Self::adds(used).is_none_or(|write| self.writes.holds(used, held(write))),
                "an instance added a draw's geometry writes by their identity less than its uses"
            );
        }
        self.add(&uses[past..]);
        self.added = now;
        past == 0
    }

    /// The memory dependency a later draw with `uses` needs before its fragment shader reads or
    /// writes memory one of the writes wrote: from the writes' stages and access types to the
    /// fragment shader and the later draw's access types there. `None` where its fragment
    /// shader touches none of it.
    pub(super) fn barrier_before(&self, uses: &[Use<K>]) -> Option<Barrier> {
        const FRAGMENT: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        // Where it holds no write, no use meets one: it walks none of them.
        if self.writes.is_empty() {
            return None;
        }
        let mut src = Access::NONE;
        let mut dst = Access::NONE;
        for used in uses {
            let mut fragment = Access::NONE;
            if used.read.stages.intersects(FRAGMENT) {
                fragment = fragment.union(Access::new(FRAGMENT, used.read.access));
            }
            if used.write.stages.intersects(FRAGMENT) {
                fragment = fragment.union(Access::new(FRAGMENT, used.write.access));
            }
            if fragment.is_empty() {
                continue;
            }
            for known in self.writes.meeting(used) {
                src = src.union(known.write);
                dst = dst.union(fragment);
            }
        }
        (!src.is_empty()).then_some(Barrier { src, dst })
    }
}

/// Whether `known`, a use of earlier commands, touches in one of `stages` memory `used` touches
/// where either writes: overlapping ranges of one tracked resource, `known` writing in `stages`
/// what `used` reads, or reading or writing in `stages` what `used` writes. `known`'s stages
/// are its commands' together, so it claims more and stays exact.
pub(super) fn touches<K: PartialEq>(
    known: &Use<K>,
    used: &Use<K>,
    stages: vk::PipelineStageFlags2,
) -> bool {
    let reads = !used.read.is_empty() && known.write.stages.intersects(stages);
    let writes =
        !used.write.is_empty() && (known.read.stages | known.write.stages).intersects(stages);
    used.tracked && known.key == used.key && (reads || writes) && known.range.overlaps(&used.range)
}

/// Merge one resource's uses into one per kind of range, which claims more and stays exact.
fn coalesce_uses<K>(uses: &mut Vec<Use<K>>) {
    let mut first = 0;
    while first < uses.len() {
        let mut next = first + 1;
        while next < uses.len() {
            match uses[first].range.hull(&uses[next].range) {
                Some(range) => {
                    let other = uses.swap_remove(next);
                    let merged = &mut uses[first];
                    merged.range = range;
                    merged.read = merged.read.union(other.read);
                    merged.write = merged.write.union(other.write);
                }
                None => next += 1,
            }
        }
        first += 1;
    }
}

/// The uses of a command of `reach` as the encoder orders them ([`Class`]), written over `into`:
/// to compare with the earlier commands' (`recorded` false), or as it keeps them (`recorded`
/// true), one per class and range, sorted by class, without those that touch nothing ([`merge`]).
/// A tracked use is its key's, either way. An untracked use of a [`Reach::Named`] command compares
/// with its key's and with the accesses of the commands that do not name theirs, and is kept as
/// its key's and as an access of all untracked memory; one of a [`Reach::Unnamed`] command
/// compares with every untracked access, and is kept as an access of all untracked memory, and of
/// the commands that do not name theirs.
fn inside_uses<K: Copy + Ord>(
    uses: &[Use<K>],
    reach: Reach,
    recorded: bool,
    into: &mut Vec<Use<Class<K>>>,
) {
    let class = |used: &Use<K>, key: Class<K>, range: UseRange| Use {
        key,
        tracked: used.tracked,
        range,
        read: used.read,
        write: used.write,
    };
    into.clear();
    for used in uses {
        let (first, second) = match (used.tracked, reach, recorded) {
            (true, _, _) => (Class::Key(used.key), None),
            (false, Reach::Named, false) => (Class::Key(used.key), Some(Class::Unnamed)),
            (false, Reach::Named, true) => (Class::Key(used.key), Some(Class::Untracked)),
            (false, Reach::Unnamed, false) => (Class::Untracked, None),
            (false, Reach::Unnamed, true) => (Class::Untracked, Some(Class::Unnamed)),
        };
        let range = match first {
            Class::Key(_) => used.range.clone(),
            Class::Untracked | Class::Unnamed => UseRange::UNTRACKED,
        };
        let classes = [Some(class(used, first, range))]
            .into_iter()
            .chain([second.map(|key| class(used, key, UseRange::UNTRACKED))])
            .flatten();
        into.extend(classes.filter(|used| !used.is_nothing()));
    }
    merge(into);
}

/// What one ended encoder hands to its queue's tracker at commit.
#[derive(Clone)]
pub(super) struct Summary<K> {
    /// The tracked uses, one per resource and range, sorted by resource.
    uses: Vec<Use<K>>,
    /// Every stage the encoder's commands ran in.
    stages: vk::PipelineStageFlags2,
    /// The fences waited for before the encoder.
    waits: Vec<FenceUse<K>>,
    /// The fences updated after the encoder.
    updates: Vec<FenceUse<K>>,
}

impl<K> fmt::Debug for Summary<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Summary")
            .field("uses", &self.uses.len())
            .field("stages", &format_args!("{:#x}", self.stages.as_raw()))
            .field("waits", &self.waits.len())
            .field("updates", &self.updates.len())
            .finish()
    }
}

impl<K: Copy + Ord> Summary<K> {
    /// The summary of work outside any encoder that makes `uses` in every stage, with no fence:
    /// a batch's layout transitions of the caller's images, planned among its encoders.
    pub(super) fn outside_encoders(mut uses: Vec<Use<K>>) -> Summary<K> {
        uses.sort_by_key(|used| used.key);
        Summary {
            uses,
            stages: vk::PipelineStageFlags2::ALL_COMMANDS,
            waits: Vec::new(),
            updates: Vec::new(),
        }
    }

    /// The summary of a resource state encoder, which records no command: it waits for the fences
    /// `waits` and updates `updates`, in every stage, so a wait on its queue orders after all of
    /// its batch's work before it, which its sparse binding precedes.
    pub(super) fn of_fences(waits: Vec<K>, updates: Vec<K>) -> Summary<K> {
        let every = |fence| FenceUse { fence, draws: None };
        Summary {
            uses: Vec::new(),
            stages: vk::PipelineStageFlags2::ALL_COMMANDS,
            waits: waits.into_iter().map(every).collect(),
            updates: updates.into_iter().map(every).collect(),
        }
    }

    /// The summary of an encoder whose commands do not run (`CommandBuffer::commit_failed`): its
    /// fences' waits and updates alone, in every stage, as [`of_fences`](Self::of_fences) makes
    /// them.
    pub(super) fn fences_only(&self) -> Summary<K> {
        let fences = |uses: &[FenceUse<K>]| uses.iter().map(|used| used.fence).collect();
        Summary::of_fences(fences(&self.waits), fences(&self.updates))
    }

    /// The resources of the tracked uses, in order, each once per range.
    pub(super) fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.uses.iter().map(|used| used.key)
    }

    /// Join `other`, the summary of work recorded apart that runs inside the same encoder (an
    /// indirect command buffer's commands, recorded at commit): its stages, fences and tracked
    /// uses, one per resource and range. A resource past [`ENTRIES_PER_KEY`] ranges takes a new
    /// range into one of the same kind, which claims more and stays exact. Both lists are sorted
    /// by resource, so each of `other`'s uses meets the known uses of its own resource alone,
    /// found by a binary search, and those of it joined before it ([`join_use`]): a join costs
    /// about the uses' count, not their product (`tests::tracker`).
    pub(super) fn join(&mut self, other: Summary<K>) {
        self.stages |= other.stages;
        debug_assert!(other.uses.is_sorted_by_key(|used| used.key));
        let known = self.uses.len();
        let mut added = Vec::new();
        let mut uses = other.uses.into_iter().peekable();
        while let Some(first) = uses.next() {
            let key = first.key;
            let start = self.uses[..known].partition_point(|used| used.key < key);
            let end = start + self.uses[start..known].partition_point(|used| used.key == key);
            let from = added.len();
            let mut next = Some(first);
            while let Some(used) = next {
                join_use((&mut self.uses[start..end], &mut added, from), used);
                next = uses.next_if(|used| used.key == key);
            }
        }
        // After the known uses, each resource's in the order they were added, as a stable sort
        // of the uses added after them leaves them.
        self.uses.append(&mut added);
        self.uses.sort_by_key(|used| used.key);
        for used in other.waits {
            add_fence(&mut self.waits, used);
        }
        for used in other.updates {
            add_fence(&mut self.updates, used);
        }
    }
}

/// Join `used` into the uses of its resource a summary knows (`known`'s, then `added`'s from
/// `from` on, in that order: [`Summary::join`]): the first of its range takes it; else, where the
/// resource has [`ENTRIES_PER_KEY`] ranges or more, the first whose range a hull of both is of the
/// same kind; else it is added. Only uses of its resource are joined into, as a scan of every
/// known use would find them.
fn join_use<K: Copy + PartialEq>(
    (known, added, from): (&mut [Use<K>], &mut Vec<Use<K>>, usize),
    used: Use<K>,
) {
    #[cfg(test)]
    JOIN_LOOKED_AT.with(|at| at.set(at.get() + (known.len() + added.len() - from) as u64));
    let key = used.key;
    let into = |known: &mut Use<K>| {
        known.read = known.read.union(used.read);
        known.write = known.write.union(used.write);
    };
    let mut theirs = (known.iter_mut()).chain(added[from..].iter_mut());
    if let Some(same) = theirs.find(|known| known.key == key && known.range == used.range) {
        into(same);
        return;
    }
    let ranges = (known.iter()).chain(&added[from..]);
    if ranges.filter(|known| known.key == key).count() >= ENTRIES_PER_KEY {
        let mut theirs = (known.iter_mut()).chain(added[from..].iter_mut());
        let hull = theirs.find_map(|known| {
            let range = (known.key == key).then(|| known.range.hull(&used.range))??;
            Some((known, range))
        });
        if let Some((known, range)) = hull {
            known.range = range;
            into(known);
            return;
        }
    }
    added.push(used);
}

#[cfg(test)]
thread_local! {
    /// The known uses [`join_use`] looked at on this thread, in tests.
    static JOIN_LOOKED_AT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many known uses joins looked at on this thread so far ([`join_use`]).
#[cfg(test)]
pub(super) fn join_looked_at() -> u64 {
    JOIN_LOOKED_AT.with(std::cell::Cell::get)
}

impl<K: Copy + Ord + Hash> EncoderUses<K> {
    /// The room its current part's list of `key`'s uses has past the first, where it tracks it.
    #[cfg(test)]
    pub(super) fn range_room(&self, key: K) -> Option<usize> {
        let part = self.parts.iter().last()?;
        part.tracked.get(&key).map(|few| few.rest.capacity())
    }

    /// Where the storage of the lists its last command was ordered by lies, and its room: the
    /// list compared with the earlier commands', then the one kept apart ([`inside_uses`]).
    #[cfg(test)]
    pub(super) fn ordered_storage(&self) -> [(usize, usize); 2] {
        [&self.compared, &self.kept].map(|list| (list.as_ptr().addr(), list.capacity()))
    }
}

impl<K> Summary<K> {
    /// The tracked uses.
    pub(super) fn uses(&self) -> &[Use<K>] {
        &self.uses
    }

    /// Every stage the encoder's commands ran in.
    pub(super) fn stages(&self) -> vk::PipelineStageFlags2 {
        self.stages
    }
}

/// What a batch's summaries make and wait for, as the scheduler compares a batch held for a
/// reason of the runtime's own with one committed after it on another queue (`scheduler/order.rs`):
/// its tracked uses, sorted by resource, and the fences it waits for and updates.
pub(super) struct Footprint<K> {
    uses: Vec<Use<K>>,
    waits: Vec<K>,
    updates: Vec<K>,
}

impl<K: Copy + Ord> Footprint<K> {
    /// The footprint of a batch whose work these summaries hold.
    pub(super) fn of<'a>(summaries: impl IntoIterator<Item = &'a Summary<K>>) -> Footprint<K>
    where
        K: 'a,
    {
        Footprint::gathered(summaries).sorted()
    }

    /// What these summaries' footprint holds, copied out of them, its uses not sorted yet
    /// ([`sorted`](Self::sorted)): a caller that must not hold the summaries' lock while it
    /// sorts gathers under it and sorts after.
    pub(super) fn gathered<'a>(summaries: impl IntoIterator<Item = &'a Summary<K>>) -> Footprint<K>
    where
        K: 'a,
    {
        let mut footprint = Footprint {
            uses: Vec::new(),
            waits: Vec::new(),
            updates: Vec::new(),
        };
        for summary in summaries {
            footprint.uses.extend(summary.uses.iter().cloned());
            footprint
                .waits
                .extend(summary.waits.iter().map(|used| used.fence));
            footprint
                .updates
                .extend(summary.updates.iter().map(|used| used.fence));
        }
        footprint
    }

    /// The footprint [`gathered`](Self::gathered) made, its uses sorted by resource.
    pub(super) fn sorted(mut self) -> Footprint<K> {
        self.uses.sort_by_key(|used| used.key);
        self
    }

    /// Whether a tracked use of the batch names a resource `named` answers for.
    pub(super) fn names_any(&self, mut named: impl FnMut(&K) -> bool) -> bool {
        self.uses.iter().any(|used| named(&used.key))
    }

    /// Whether the batch makes no tracked use and names no fence: no order across queues
    /// concerns it.
    fn is_empty(&self) -> bool {
        self.uses.is_empty() && self.waits.is_empty() && self.updates.is_empty()
    }

    /// Whether `self`, committed first, and `later` must run in that order: a tracked use of
    /// each shares a resource's range and one of them writes it, or one updates a fence the
    /// other waits for or updates.
    fn conflicts(&self, later: &Footprint<K>) -> bool {
        let fences = |updates: &[K], other: &Footprint<K>| {
            updates
                .iter()
                .any(|fence| other.waits.contains(fence) || other.updates.contains(fence))
        };
        if fences(&self.updates, later) || fences(&later.updates, self) {
            return true;
        }
        later.uses.iter().any(|used| {
            let first = self.uses.partition_point(|earlier| earlier.key < used.key);
            self.uses[first..]
                .iter()
                .take_while(|earlier| earlier.key == used.key)
                .any(|earlier| {
                    earlier.range.overlaps(&used.range)
                        && !(earlier.write.is_empty() && used.write.is_empty())
                })
        })
    }
}

/// Whether a batch whose footprint is `earlier`, committed first, and one whose footprint is
/// `later` must run in that order across queues; `None` is a batch whose uses are not known yet
/// (work a host step or its submission records), which may conflict with any batch that makes
/// a tracked use or names a fence.
pub(super) fn ordered<K: Copy + Ord>(
    earlier: Option<&Footprint<K>>,
    later: Option<&Footprint<K>>,
) -> bool {
    match (earlier, later) {
        (Some(earlier), Some(later)) => earlier.conflicts(later),
        (None, Some(known)) | (Some(known), None) => !known.is_empty(),
        (None, None) => true,
    }
}

/// A fence's last update in the device's submission order.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq))]
struct FenceUpdate {
    /// The stages it covers: those of the updating encoder's the call names
    /// ([`FenceUse::covered`]), and those of the fences it waited for.
    stages: vk::PipelineStageFlags2,
    /// The batch of the updating encoder.
    at: Point,
}

/// The device's hazard state, in the scheduler's submission order: every logical queue's batches,
/// on every `VkQueue`.
pub(super) struct Tracker<K> {
    resources: KeyMap<K, Vec<Entry>>,
    fences: KeyMap<K, FenceUpdate>,
    /// Per `VkQueue`: the timeline value through which every batch completed, as the host saw.
    completed: Vec<u64>,
    /// Per `VkQueue`: the resources and fences each applied plan touched, by its timeline value,
    /// in submission order: what its batch's completion may retire ([`Tracker::completed`]).
    touched: Vec<VecDeque<(u64, Touched<K>)>>,
    /// Lists a plan applied before gave back, cleared, for the next plans to fill
    /// ([`Spares`]).
    spares: Spares<K>,
    /// Every batch ends with [`Barrier::to_all`], so the accesses of earlier batches of a
    /// `VkQueue` need no barrier there ([`Tracker::by_ends`]).
    by_ends: bool,
}

/// A resource or a fence an applied plan touched ([`Tracker::touched`]).
#[derive(Clone, Copy)]
enum Touched<K> {
    Resource(K),
    Fence(K),
}

/// The most lists of entries a tracker keeps for plans to fill: more than a batch's resources
/// usually are, a bounded few hundred kilobytes at most.
const SPARE_ENTRIES: usize = 64;

/// The cleared lists a tracker keeps for its next plans: resources' entries (each list with the
/// capacity its last resource's grew to, at most what [`ENTRIES_PER_KEY`] lets one reach), and a
/// plan's own lists, the last applied plan's.
struct Spares<K> {
    entries: Vec<Vec<Entry>>,
    plans: Vec<PlanLists<K>>,
}

impl<K> Default for Spares<K> {
    fn default() -> Spares<K> {
        Spares {
            entries: Vec::new(),
            plans: Vec::new(),
        }
    }
}

/// A plan's own lists ([`Plan`]): its barriers, its waits, its resources' entries and the uses
/// of a batch of one summary.
struct PlanLists<K> {
    barriers: Vec<Option<Barrier>>,
    waits: Vec<Wait>,
    resources: Vec<(K, Vec<Entry>)>,
    uses: Vec<Use<K>>,
}

impl<K> Default for PlanLists<K> {
    fn default() -> PlanLists<K> {
        PlanLists {
            barriers: Vec::new(),
            waits: Vec::new(),
            resources: Vec::new(),
            uses: Vec::new(),
        }
    }
}

impl<K> Spares<K> {
    /// Keep `entries` for a later plan, cleared, while fewer than [`SPARE_ENTRIES`] are kept.
    fn keep(&mut self, mut entries: Vec<Entry>) {
        if self.entries.len() < SPARE_ENTRIES {
            entries.clear();
            self.entries.push(entries);
        }
    }
}

impl<K> Default for Tracker<K> {
    fn default() -> Tracker<K> {
        Tracker {
            resources: KeyMap::default(),
            fences: KeyMap::default(),
            completed: Vec::new(),
            touched: Vec::new(),
            spares: Spares::default(),
            by_ends: false,
        }
    }
}

impl<K> fmt::Debug for Tracker<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tracker")
            .field("resources", &self.resources.len())
            .field("fences", &self.fences.len())
            .field("completed", &self.completed)
            .finish()
    }
}

/// The entries of the resources one batch uses, as its plan changes them: copies of the state's,
/// sorted by resource. A batch uses a few resources, so a search of the sorted list finds one
/// where a map would hash.
struct Overlay<K>(Vec<(K, Vec<Entry>)>);

impl<K: Ord> Overlay<K> {
    /// Where `key` is, or would go.
    fn find(&self, key: &K) -> Result<usize, usize> {
        self.0.binary_search_by(|(known, _)| known.cmp(key))
    }

    /// Add `key` with the entries `fill` makes, unless it is listed.
    fn add(&mut self, key: K, fill: impl FnOnce() -> Vec<Entry>) {
        if let Err(at) = self.find(&key) {
            self.0.insert(at, (key, fill()));
        }
    }
}

/// The state's own entries, as a plan of one summary is applied to them ([`Plan::uses`]): a
/// resource they gain takes a list a plan gave back.
struct Kept<'a, K> {
    resources: &'a mut KeyMap<K, Vec<Entry>>,
    spares: &'a mut Spares<K>,
}

impl<K: Eq + Hash> Entries<K> for Kept<'_, K> {
    fn of(&self, key: &K) -> Option<&Vec<Entry>> {
        self.resources.get(key)
    }

    fn of_mut(&mut self, key: K) -> &mut Vec<Entry> {
        let spares = &mut self.spares.entries;
        (self.resources)
            .entry(key)
            .or_insert_with(|| spares.pop().unwrap_or_default())
    }
}

impl<K: Ord> Entries<K> for Overlay<K> {
    fn of(&self, key: &K) -> Option<&Vec<Entry>> {
        self.find(key).ok().map(|at| &self.0[at].1)
    }

    fn of_mut(&mut self, key: K) -> &mut Vec<Entry> {
        let at = match self.find(&key) {
            Ok(at) => at,
            Err(at) => {
                self.0.insert(at, (key, Vec::new()));
                at
            }
        };
        &mut self.0[at].1
    }
}

/// The barriers before each encoder of a batch submitted next, the batch's waits for other
/// queues, and the state after it. [`Tracker::apply`] keeps the state once the batch is
/// submitted; a batch that is never submitted drops its plan, and the state is as before.
pub(super) struct Plan<K> {
    /// The batch's place.
    at: Point,
    barriers: Vec<Option<Barrier>>,
    waits: Vec<Wait>,
    resources: Overlay<K>,
    /// A batch of one summary: its uses, which [`Tracker::apply`] admits into the state itself,
    /// after the plan's barrier and with its waits, as the plan would have admitted them into its
    /// copies (`resources`, then empty) and the copies would have replaced the state's: the
    /// scheduler does nothing else with the tracker between a plan and its application. A batch
    /// of several summaries plans each against the copies the earlier ones changed.
    uses: Vec<Use<K>>,
    fences: KeyMap<K, FenceUpdate>,
}

impl<K> fmt::Debug for Plan<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("barriers", &self.barriers)
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

impl<K> Plan<K> {
    /// The barrier before each encoder, in order.
    pub(super) fn barriers(&self) -> &[Option<Barrier>] {
        &self.barriers
    }

    /// The batch's waits for other queues' timelines, at most one per queue.
    pub(super) fn waits(&self) -> &[Wait] {
        &self.waits
    }

    /// How many resources' entries the plan copied from the state.
    #[cfg(test)]
    pub(super) fn copied(&self) -> usize {
        self.resources.0.len()
    }
}

impl<K> Tracker<K> {
    /// A tracker of a device whose batches each end with [`Barrier::to_all`] (`batch_order.rs`):
    /// it plans no barrier after an access of an earlier batch of the same `VkQueue`.
    pub(super) fn by_ends() -> Tracker<K> {
        Tracker {
            by_ends: true,
            ..Tracker::default()
        }
    }

    /// The value below which the batches of `at`'s queue are ordered by their ends: `at`'s own
    /// where every batch ends with [`Barrier::to_all`], else 0.
    fn since(&self, at: Point) -> u64 {
        match self.by_ends {
            true => at.value,
            false => 0,
        }
    }
}

impl<K: Copy + Ord + Hash> Tracker<K> {
    /// Every batch of `queue` through timeline value `value` completed, and the host saw it: the
    /// state forgets what they did. Only the resources and fences those batches touched are looked
    /// at: every write or read the state holds is at the place of a batch that touched its
    /// resource, so the completion of each such batch looks at the resource again.
    pub(super) fn completed(&mut self, queue: usize, value: u64) {
        if self.completed.len() <= queue {
            self.completed.resize(queue + 1, 0);
        }
        if value <= self.completed[queue] {
            return;
        }
        self.completed[queue] = value;
        let Some(touched) = self.touched.get_mut(queue) else {
            return;
        };
        let completed = &self.completed;
        while let Some(&(at, what)) = touched.front() {
            if at > value {
                break;
            }
            touched.pop_front();
            match what {
                Touched::Resource(key) => {
                    let Some(entries) = self.resources.get_mut(&key) else {
                        continue;
                    };
                    entries.retain_mut(|entry| !entry.retire(completed));
                    if entries.is_empty() {
                        if let Some(entries) = self.resources.remove(&key) {
                            self.spares.keep(entries);
                        }
                    }
                }
                Touched::Fence(fence) => {
                    let done = self.fences.get(&fence).is_some_and(|update| {
                        completed
                            .get(update.at.queue)
                            .is_some_and(|&through| update.at.value <= through)
                    });
                    if done {
                        self.fences.remove(&fence);
                    }
                }
            }
        }
    }

    /// The barriers before the `encoders` of the batch submitted next, at `at`, and its waits.
    /// The state is unchanged; the plan's lists come from those plans applied before gave back.
    pub(super) fn plan<'a>(
        &mut self,
        encoders: impl IntoIterator<Item = &'a Summary<K>>,
        at: Point,
    ) -> Plan<K>
    where
        K: 'a,
    {
        let since = self.since(at);
        let spares = &mut self.spares;
        let PlanLists {
            barriers,
            waits,
            resources,
            uses,
        } = spares.plans.pop().unwrap_or_default();
        let mut plan = Plan {
            at,
            barriers,
            waits,
            resources: Overlay(resources),
            uses,
            fences: KeyMap::default(),
        };
        let mut encoders = encoders.into_iter().peekable();
        while let Some(summary) = encoders.next() {
            // A batch of one summary: planned against the state, admitted into it as the plan
            // is applied ([`Plan::uses`]), its entries not copied.
            let alone = plan.barriers.is_empty() && encoders.peek().is_none();
            let mut barrier = Barrier::default();
            let mut waited = vk::PipelineStageFlags2::NONE;
            for wait in &summary.waits {
                let fence = &wait.fence;
                let update = plan.fences.get(fence).or_else(|| self.fences.get(fence));
                if let Some(update) = update {
                    waited |= update.stages;
                    // The stages the call names wait, and the encoder's work outside its draws.
                    let stages = wait.covered(summary.stages);
                    if update.at.queue == at.queue {
                        // An update of a batch ordered by its end needs nothing.
                        if update.at.value >= since {
                            barrier.add(
                                Access::new(update.stages, vk::AccessFlags2::MEMORY_WRITE),
                                Access::new(
                                    stages,
                                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                                ),
                            );
                        }
                    } else {
                        add_wait(
                            &mut plan.waits,
                            Wait {
                                at: update.at,
                                stages,
                            },
                        );
                    }
                }
            }
            if !alone {
                for used in &summary.uses {
                    plan.resources.add(used.key, || {
                        let mut entries = spares.entries.pop().unwrap_or_default();
                        if let Some(known) = self.resources.get(&used.key) {
                            entries.extend(known.iter().cloned());
                        }
                        entries
                    });
                }
            }
            let needs = match alone {
                true => depend(&self.resources, &summary.uses, (at, since)),
                false => depend(&plan.resources, &summary.uses, (at, since)),
            };
            barrier.add(needs.barrier.src, needs.barrier.dst);
            for wait in needs.waits {
                add_wait(&mut plan.waits, wait);
            }
            let barrier = barrier.needed();
            match alone {
                true => plan.uses.extend(summary.uses.iter().cloned()),
                false => admit(&mut plan.resources, &summary.uses, barrier, &plan.waits, at),
            }
            for update in &summary.updates {
                let stages = update.covered(summary.stages) | waited;
                plan.fences.insert(update.fence, FenceUpdate { stages, at });
            }
            plan.barriers.push(barrier);
        }
        plan
    }

    /// Keep the state after a planned batch that was submitted, and what it touched for its
    /// completion to retire ([`Tracker::completed`]); its lists, and those its resources held
    /// before, go back for later plans.
    pub(super) fn apply(&mut self, plan: Plan<K>) {
        let Plan {
            at,
            mut barriers,
            mut waits,
            resources: Overlay(mut resources),
            mut uses,
            fences,
        } = plan;
        if self.touched.len() <= at.queue {
            self.touched.resize_with(at.queue + 1, VecDeque::new);
        }
        let touched = &mut self.touched[at.queue];
        // A batch of one summary, after its one barrier ([`Plan::uses`]).
        if !uses.is_empty() {
            let mut kept = Kept {
                resources: &mut self.resources,
                spares: &mut self.spares,
            };
            let barrier = barriers.first().copied().flatten();
            admit(&mut kept, &uses, barrier, &waits, at);
            for group in uses.chunk_by(|a, b| a.key == b.key) {
                touched.push_back((at.value, Touched::Resource(group[0].key)));
            }
        }
        for (key, entries) in resources.drain(..) {
            touched.push_back((at.value, Touched::Resource(key)));
            let before = if entries.is_empty() {
                self.spares.keep(entries);
                self.resources.remove(&key)
            } else {
                self.resources.insert(key, entries)
            };
            if let Some(before) = before {
                self.spares.keep(before);
            }
        }
        for (fence, update) in fences {
            touched.push_back((update.at.value, Touched::Fence(fence)));
            self.fences.insert(fence, update);
        }
        barriers.clear();
        waits.clear();
        uses.clear();
        self.spares.plans.push(PlanLists {
            barriers,
            waits,
            resources,
            uses,
        });
    }

    /// Whether `other` holds the same state: every resource's entries and every fence's update,
    /// and the completed values.
    #[cfg(test)]
    pub(super) fn same_state(&self, other: &Tracker<K>) -> bool {
        type Sorted<'a, K> = (Vec<(&'a K, &'a Vec<Entry>)>, Vec<(&'a K, &'a FenceUpdate)>);
        fn sorted<K: Copy + Ord>(tracker: &Tracker<K>) -> Sorted<'_, K> {
            let mut resources: Vec<_> = tracker.resources.iter().collect();
            resources.sort_by_key(|&(key, _)| *key);
            let mut fences: Vec<_> = tracker.fences.iter().collect();
            fences.sort_by_key(|&(key, _)| *key);
            (resources, fences)
        }
        sorted(self) == sorted(other) && self.completed == other.completed
    }

    /// How many entries the state holds, over all resources and fences.
    #[cfg(test)]
    pub(super) fn entries(&self) -> usize {
        self.resources.values().map(Vec::len).sum::<usize>() + self.fences.len()
    }

    /// How many resources and fences the completions of the batches applied so far will look at
    /// ([`Tracker::completed`]).
    #[cfg(test)]
    pub(super) fn to_retire(&self) -> usize {
        self.touched.iter().map(VecDeque::len).sum()
    }

    /// Whether the state holds a write, a read or a fence update of a batch the host saw
    /// complete: what a look at every resource and fence would retire.
    #[cfg(test)]
    pub(super) fn holds_completed(&self) -> bool {
        let done =
            |at: Point| (self.completed.get(at.queue)).is_some_and(|&through| at.value <= through);
        let accesses = self.resources.values().flatten().any(|entry| {
            entry.writers.iter().any(|writer| done(writer.at))
                || entry.reads.iter().any(|reads| done(reads.at))
        });
        accesses || self.fences.values().any(|update| done(update.at))
    }
}
