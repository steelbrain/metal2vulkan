//! `synchronize_resource` and `synchronize_texture` of a buffer with a host copy on a device
//! without unified memory (`managed.rs`, `managed_copy.rs`): the copy out of the bytes the GPU
//! wrote since the buffer's last copy out, not of the whole buffer.
//!
//! - **What Metal asks.** Once the command buffer that holds the command completed, the CPU sees
//!   in the host copy every byte the GPU wrote into the buffer before the command, in the GPU's
//!   order, across command buffers and queues. Every other byte already holds there what the
//!   device copy holds: the copies in (the making's, a `did_modify_range`'s range, a rebind's)
//!   give the device copy the host copy's bytes, and an earlier copy out gave the host copy the
//!   device copy's. So the copy out moves the bytes the GPU's writes reached since the buffer's
//!   last copy out, a superset of them where they are not known exactly, and none where none
//!   did. A byte the CPU wrote in the host copy and the GPU did not write keeps the CPU's value,
//!   whatever the CPU said of it; a range the GPU wrote takes the GPU's, as on a Mac with a
//!   discrete GPU.
//! - **The writes.** The GPU's writes to the buffer are the tracked uses that write it of every
//!   command the runtime records (its encoders' summaries, their segments', what a command buffer
//!   made resident for writing: `tracker.rs`), each with its range: a kernel's or a draw's
//!   binding from its offset, a blit's exact bytes, a texture over the buffer its texels' span or
//!   the buffer whole. [`GpuWrites`] keeps them for each buffer synchronized once, each with the
//!   stamp of its batch's place in the scheduler's order over all queues (`command_buffer/
//!   submit.rs`), which is the order the tracker orders tracked work in.
//! - **The copy's place.** The command is tracked where it is encoded, a read of the buffer's
//!   range and a write of the host copy's, as any blit's; its copy goes into a `VkCommandBuffer`
//!   of its own among its encoder's (a segment, `encoder.rs`: the copies out of a run of
//!   synchronizes with nothing between them share one), recorded at its batch's submission
//!   ([`Deferred::SyncOut`](super::deferred::Deferred)), each after the barrier the encoder
//!   planned for it. There it copies the bytes of the writes noted for the buffer so far: those
//!   of every batch submitted before, of the encoders of its batch before its own, and of its own
//!   encoder. A write the tracker orders before the copy (every write submitted before it, the
//!   tracker's barrier or wait from it, Metal's order) is among them: one encoded after the
//!   synchronize in a command buffer committed before its own too. One submitted after it runs
//!   after the copy, the tracker ordering it after the copy's read.
//! - **A write let go of.** Once the command buffer that holds the copy completed, the copy is in
//!   the host copy, so the writes it took that ran before it are let go of: those noted before
//!   it (of earlier batches and of the earlier encoders of its own batch). Its own encoder's
//!   writes, and its command buffer's writes through what it made resident, which may run after
//!   it, stay for the next copy out, as does every write noted after it. A command buffer that
//!   failed lets go of nothing.
//! - **Copied whole** where the writes are not known: an untracked buffer, whose uses no summary
//!   holds; one whose `VkBuffer` was handed out (`Buffer::vk_buffer`) or whose memory is
//!   exported, which the caller's own work may write; the buffer's first synchronize, before
//!   which its writes were not kept; a command buffer that may write memory no use names
//!   (`State::writes_unnamed`: a root table, records a host step resolves) submitted since the
//!   buffer's last copy out that the host had not seen complete before that copy was decided; and
//!   a synchronize recorded in place (inside a predicated loop's laying, or where no new
//!   `VkCommandBuffer` could be made), which lets go of nothing
//!   (`WorkCounts::managed_synchronizes_whole`).
//! - **Pages no import maps** (`managed_copy.rs`'s `Side::Pages`): the GPU copies into a staging
//!   buffer of the range's length, which the command buffer's completion copies into the pages,
//!   the same bytes in both.
//!
//! Counted for buffers over the caller's pages, as the rest of their copies are
//! (`WorkCounts::managed_bytes_copied_out`, `managed_synchronizes_unwritten`,
//! `managed_bytes_unwritten_skipped`).

use super::buffer::{Buffer, WeakBuffer};
use super::device::lock;
use super::key_hash::KeyMap;
use super::managed_copy::CopyBack;
use super::resource::ResourceKey;
use super::tracker::{Barrier, Point, Summary, UseRange};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The most writes kept apart for one buffer, past which they are one write of the bytes from the
/// first's start to the last's end.
const WRITES_KEPT: usize = 16;

/// Where a copy out goes.
pub(super) enum Target {
    /// The runtime's own host copy's `VkBuffer`, whose byte 0 is the buffer's.
    Own(vk::Buffer),
    /// An import of the caller's pages: its `VkBuffer`, and where the range's first byte lies in
    /// it.
    Import(vk::Buffer, u64),
    /// Pages no import maps: a staging buffer of the range's bytes, which the command buffer's
    /// completion copies into the pages.
    Pages(CopyBack),
}

/// One `synchronize_resource`'s copy out of a range of a buffer with a host copy (the module
/// docs).
pub(super) struct SyncCopy {
    device: Device,
    buffer: Buffer,
    range: Range<u64>,
    /// The buffer's own `VkBuffer`, and where the range's first byte lies in it.
    own: (vk::Buffer, u64),
    to: Target,
    /// Its buffer lies over the caller's pages: its copies are counted.
    counted: bool,
    /// What it copies, once decided.
    decided: Mutex<Option<Decided>>,
}

/// What a copy out copies: the bytes `regions` of its range, in order and apart; and where it was
/// decided at its batch's submission, what its completion lets go of.
struct Decided {
    regions: Vec<Range<u64>>,
    retire: Option<Retire>,
}

/// What a completed copy out lets go of ([`GpuWrites::retire`]): the writes noted at or before
/// `stamp`, and the batches that may write memory no use names noted at or before `through`.
#[derive(Clone, Copy)]
struct Retire {
    stamp: u64,
    through: u64,
}

impl SyncCopy {
    /// The copy out of `range` of `buffer` on `device`, from its own bytes at `own` to `to`.
    pub(super) fn new(
        device: &Device,
        buffer: &Buffer,
        range: Range<u64>,
        own: (vk::Buffer, u64),
        to: Target,
    ) -> SyncCopy {
        SyncCopy {
            device: device.clone(),
            buffer: buffer.clone(),
            range,
            own,
            to,
            counted: buffer.caller_copy().is_some(),
            decided: Mutex::new(None),
        }
    }

    /// Whether the GPU's writes to its buffer are kept (the module docs, **Copied whole**): a
    /// tracked buffer whose memory is not exported. Its `VkBuffer`'s handing out is read as each
    /// copy is decided.
    pub(super) fn writes_known(&self) -> bool {
        self.buffer.identity().tracked && self.buffer.exported().is_none()
    }

    /// Its buffer's key.
    fn key(&self) -> ResourceKey {
        self.buffer.identity().key
    }

    /// Copy its range whole, decided now, where its encoder records it (the module docs,
    /// **Copied whole**).
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of its device in the recording state, outside a render
    /// pass, recorded by this caller alone, which holds the buffer and its target until it
    /// completes.
    pub(super) unsafe fn record_whole(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let regions = vec![self.range.clone()];
        // SAFETY: the caller's contract.
        unsafe { self.record(raw, command_buffer, &regions) };
        self.decided(regions, None, true);
    }

    /// Record the copies of `regions` of its range.
    ///
    /// # Safety
    ///
    /// As [`record_whole`](Self::record_whole)'s.
    unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer, regions: &[Range<u64>]) {
        let (to, at) = match &self.to {
            Target::Own(host) => (*host, self.range.start),
            Target::Import(host, at) => (*host, *at),
            Target::Pages(back) => back.staging_at(),
        };
        let (own, from) = self.own;
        let copies: Vec<vk::BufferCopy> = (regions.iter())
            .map(|region| {
                let past = region.start - self.range.start;
                vk::BufferCopy {
                    src_offset: from + past,
                    dst_offset: at + past,
                    size: region.end - region.start,
                }
            })
            .collect();
        if copies.is_empty() {
            return;
        }
        // SAFETY: the caller's contract; the regions lie in the range, which lies in both
        // buffers from `from` and `at`.
        unsafe { raw.cmd_copy_buffer(command_buffer, own, to, &copies) };
    }

    /// It copies `regions`, letting go of `retire` at its completion, the range `whole` where the
    /// writes were not known: counted.
    fn decided(&self, regions: Vec<Range<u64>>, retire: Option<Retire>, whole: bool) {
        if self.counted {
            let asked = self.range.end - self.range.start;
            let copied: u64 = regions.iter().map(|region| region.end - region.start).sum();
            let counts = &self.device.shared().counts;
            counts.managed_synchronize_decided(asked - copied, regions.is_empty(), whole);
            // An import's bytes are counted at the command; a staging buffer's at the completion
            // that copies them into the pages.
            if matches!(self.to, Target::Import(..)) {
                counts.managed_copied(copied, false);
            }
        }
        *lock(&self.decided) = Some(Decided { regions, retire });
    }

    /// Its command buffer's last batch finished, `completed` where no failure was relayed for its
    /// work: the bytes it copied into a staging buffer go into the pages, and the writes it took
    /// that ran before it are let go of (the module docs). Nothing where it failed, or where its
    /// copy was never recorded (its encoder was encoded again, `command_buffer/reencode.rs`).
    pub(super) fn completed(&self, completed: bool) {
        let Some(decided) = lock(&self.decided).take() else {
            return;
        };
        if !completed {
            return;
        }
        if let Target::Pages(back) = &self.to {
            let copied = back.run(self.range.start, &decided.regions);
            if self.counted {
                (self.device.shared().counts).managed_copied(copied, false);
            }
        }
        if let Some(retire) = decided.retire {
            (self.device.shared().gpu_writes).retire(self.key(), retire);
        }
    }
}

/// A segment's copies out, recorded at its batch's submission (the module docs, **The copy's
/// place**): each with the barrier its encoder planned before it, in the order of their commands.
#[derive(Default)]
pub(super) struct SyncOut(Mutex<Vec<(Option<Barrier>, Arc<SyncCopy>)>>);

impl SyncOut {
    /// `copy`, after `barrier`, to follow the copies out before it.
    pub(super) fn push(&self, barrier: Option<Barrier>, copy: Arc<SyncCopy>) {
        lock(&self.0).push((barrier, copy));
    }

    /// Decide each copy by `writes` and record it into `command_buffer`, after its barrier: the
    /// writes noted so far, those of its own encoder `own` gives by key, and `completed` says
    /// which batches the host saw complete (the module docs).
    ///
    /// # Safety
    ///
    /// As [`SyncCopy::record_whole`]'s, for each copy.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        writes: &GpuWrites,
        own: &dyn Fn(ResourceKey) -> Vec<Range<u64>>,
        completed: &dyn Fn(Point) -> bool,
    ) {
        let copies = lock(&self.0).clone();
        for (barrier, copy) in copies {
            if let Some(barrier) = barrier {
                // SAFETY: the caller's contract.
                unsafe { barrier.record(raw, command_buffer) };
            }
            let (regions, retire, whole) = writes.decide(&copy, &own(copy.key()), completed);
            // SAFETY: the caller's contract.
            unsafe { copy.record(raw, command_buffer, &regions) };
            copy.decided(regions, Some(retire), whole);
        }
    }
}

/// The GPU's writes to the buffers with a host copy synchronized at least once, since their last
/// copy out, in the scheduler's order (the module docs).
#[derive(Default)]
pub(super) struct GpuWrites {
    /// Some buffer's writes are kept: a batch notes none while none is.
    any: AtomicBool,
    known: Mutex<Known>,
}

#[derive(Default)]
struct Known {
    /// The buffers whose writes are kept, by key.
    buffers: KeyMap<ResourceKey, Written>,
    /// The order of the notes and decisions: each takes the next.
    stamp: u64,
    /// The batches that may write memory no use names whose completion no decision saw yet, by
    /// stamp, with their places.
    unnamed: Vec<(u64, Point)>,
    /// The stamp of the last such batch.
    unnamed_last: u64,
    /// `buffers`' length when its buffers gone were last let go of.
    swept: usize,
}

/// One buffer's writes not copied out yet ([`GpuWrites`]).
struct Written {
    of: WeakBuffer,
    /// Each write's bytes and stamp, in the order noted.
    writes: Vec<(Range<u64>, u64)>,
    /// The batches that may write memory no use names noted at or before this stamp ran before a
    /// copy out that completed.
    unnamed_through: u64,
}

impl GpuWrites {
    /// Whether some buffer's writes are kept: a batch's writes are noted only then.
    pub(super) fn keeps_any(&self) -> bool {
        self.any.load(Ordering::Acquire)
    }

    /// A batch's part was submitted: note its tracked writes `summaries` hold of the buffers kept,
    /// under one stamp.
    pub(super) fn note<'a>(&self, summaries: impl Iterator<Item = &'a Summary<ResourceKey>>) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        for (key, range) in summaries.flat_map(Summary::writes) {
            let Some(written) = known.buffers.get_mut(&key) else {
                continue;
            };
            let bytes = match range {
                UseRange::Bytes(bytes) => bytes.clone(),
                UseRange::Image { .. } => 0..u64::MAX,
            };
            written.add(bytes, stamp);
        }
    }

    /// A batch at `at` of a command buffer that may write memory no use names was submitted:
    /// every buffer is copied whole until a copy out decided once the host saw it complete
    /// completed. Those the host saw complete (`completed`) go from the list.
    pub(super) fn note_unnamed(&self, at: Point, completed: &dyn Fn(Point) -> bool) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        known.unnamed_last = stamp;
        known.unnamed.retain(|(_, at)| !completed(*at));
        known.unnamed.push((stamp, at));
    }

    /// The regions of `copy`'s range to copy out now, at its batch's submission, what its
    /// completion lets go of, and whether it copies its range whole for writes not known: the
    /// writes kept of its buffer, and `own`, its encoder's (the module docs). Its buffer's writes
    /// are kept from now on.
    fn decide(
        &self,
        copy: &SyncCopy,
        own: &[Range<u64>],
        completed: &dyn Fn(Point) -> bool,
    ) -> (Vec<Range<u64>>, Retire, bool) {
        let handed_out = copy.buffer.handed_out().load(Ordering::Acquire);
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        known.unnamed.retain(|(_, at)| !completed(*at));
        // Every batch that may write memory unnamed noted at or before `through` completed.
        let through = known.unnamed.first().map_or(stamp, |(noted, _)| noted - 1);
        let unnamed_last = known.unnamed_last;
        if !known.buffers.contains_key(&copy.key()) {
            known.sweep();
        }
        let Known { buffers, .. } = &mut *known;
        // A buffer's first: its writes before now are not known.
        let first = !buffers.contains_key(&copy.key());
        let written = buffers.entry(copy.key()).or_insert_with(|| Written {
            of: copy.buffer.downgrade(),
            writes: vec![(0..u64::MAX, stamp)],
            unnamed_through: 0,
        });
        self.any.store(true, Ordering::Release);
        let whole = first || handed_out || written.unnamed_through < unnamed_last;
        let range = &copy.range;
        let regions = match whole {
            true => vec![range.clone()],
            false => {
                let writes = (written.writes.iter()).map(|(bytes, _)| bytes);
                merged(writes.chain(own), range)
            }
        };
        (regions, Retire { stamp, through }, whole)
    }

    /// A copy out decided at `retire.stamp` completed: let go of the writes of the buffer of `key`
    /// noted at or before it, and of the batches that may write memory unnamed noted at or before
    /// `retire.through`.
    fn retire(&self, key: ResourceKey, retire: Retire) {
        let mut known = lock(&self.known);
        if let Some(written) = known.buffers.get_mut(&key) {
            written.writes.retain(|(_, stamp)| *stamp > retire.stamp);
            written.unnamed_through = written.unnamed_through.max(retire.through);
        }
    }
}

impl Known {
    /// Let go of the buffers gone, once the buffers kept doubled since they last were.
    fn sweep(&mut self) {
        if self.buffers.len() < 2 * self.swept.max(32) {
            return;
        }
        self.buffers.retain(|_, written| written.of.lives());
        self.swept = self.buffers.len();
    }
}

impl Written {
    /// Note a write of `bytes` at `stamp`, the latest: the writes before it inside its bytes go,
    /// as it is let go of no earlier than they are; past [`WRITES_KEPT`], all of them are one.
    fn add(&mut self, bytes: Range<u64>, stamp: u64) {
        if bytes.is_empty() {
            return;
        }
        (self.writes).retain(|(kept, _)| !(bytes.start <= kept.start && kept.end <= bytes.end));
        self.writes.push((bytes, stamp));
        if self.writes.len() > WRITES_KEPT {
            let start = self.writes.iter().map(|(kept, _)| kept.start).min();
            let end = self.writes.iter().map(|(kept, _)| kept.end).max();
            let (start, end) = (start.unwrap_or(0), end.unwrap_or(0));
            self.writes.clear();
            self.writes.push((start..end, stamp));
        }
    }
}

/// The bytes of `range` the ranges `writes` reach, merged where they overlap or touch, in order.
fn merged<'a>(writes: impl Iterator<Item = &'a Range<u64>>, range: &Range<u64>) -> Vec<Range<u64>> {
    let mut clipped: Vec<Range<u64>> = writes
        .map(|bytes| bytes.start.max(range.start)..bytes.end.min(range.end))
        .filter(|bytes| !bytes.is_empty())
        .collect();
    clipped.sort_by_key(|bytes| bytes.start);
    let mut regions: Vec<Range<u64>> = Vec::with_capacity(clipped.len());
    for bytes in clipped {
        match regions.last_mut() {
            Some(last) if bytes.start <= last.end => last.end = last.end.max(bytes.end),
            _ => regions.push(bytes),
        }
    }
    regions
}

/// The writes `summaries` hold of the buffer of `key`, each one's bytes.
pub(super) fn writes_of<'a>(
    summaries: impl Iterator<Item = &'a Summary<ResourceKey>>,
    key: ResourceKey,
) -> Vec<Range<u64>> {
    (summaries.flat_map(Summary::writes))
        .filter(|(written, _)| *written == key)
        .map(|(_, range)| match range {
            UseRange::Bytes(bytes) => bytes.clone(),
            UseRange::Image { .. } => 0..u64::MAX,
        })
        .collect()
}
