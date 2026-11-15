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
//!   command the runtime records (its encoders' summaries, their segments', what an encoder made
//!   resident for writing: `tracker.rs`), each with its range: a kernel's or a draw's binding from
//!   its offset, a blit's exact bytes, a texture over the buffer its texels' span or the buffer
//!   whole. [`GpuWrites`] keeps those of the batches submitted for each buffer synchronized once,
//!   each with the stamp of its batch's place in the scheduler's order over all queues
//!   (`command_buffer/submit.rs`), which is the order the tracker orders tracked work in, and the
//!   command buffer it is of.
//! - **The copy's place.** The command is tracked where it is encoded, a read of the buffer's
//!   range and a write of the host copy's, as any blit's, and its copy is decided and recorded
//!   there, after the barrier its tracking plans, in its encoder's own `VkCommandBuffer`: the
//!   bytes of the writes kept for the buffer (those of every batch submitted before), and of the
//!   commands of its own command buffer encoded before it (its ended encoders', their segments',
//!   what they made resident, its own encoder's commands before it: `State::written_before`).
//!   One that copies nothing records nothing but the barrier its tracking planned, where it
//!   planned one.
//! - **What the encoding cannot see.** A batch submitted between the encoding and its own batch's
//!   submission (one of a command buffer committed before the synchronize's and encoded after
//!   it) runs before it in the GPU's order. Its batch's submission settles the copy: the writes
//!   noted since its encoding, of other command buffers, are copied by a `VkCommandBuffer` of their
//!   own just before its part's, between barriers from everything before and to everything after
//!   (`WorkCounts::managed_synchronizes_topped_up`); a byte a command of its own command buffer
//!   before the synchronize also wrote is copied again in place, after it. One submitted after its
//!   batch runs after the copy, the tracker ordering it after the copy's read, and is not copied.
//! - **A write let go of.** Once the command buffer that holds the copy completed, the copy is in
//!   the host copy, so the writes it took are let go of: those noted before its batch's
//!   submission settled it. Its own batch's writes, noted after, stay for the next copy out (a
//!   command after the synchronize writes after the copy), as does every write noted later. A
//!   command buffer that failed lets go of nothing.
//! - **Copied whole** where the writes are not known: an untracked buffer, whose uses no summary
//!   holds; one whose `VkBuffer` was handed out (`Buffer::vk_buffer`) or whose memory is
//!   exported, which the caller's own work may write; the buffer's first synchronize, before
//!   which its writes were not kept; a command of its own command buffer before it that may write
//!   memory no use names, or whose recording finds its uses (`Deferred::reaches_unnamed`); a
//!   command buffer that may write memory no use names (`State::writes_unnamed`: a root table,
//!   records a host step resolves) submitted since the buffer's last copy out that the host had
//!   not seen complete before that copy was settled; and a synchronize inside a predicated loop's
//!   laying, which lets go of nothing (`WorkCounts::managed_synchronizes_whole`).
//! - **Pages no import maps** (`managed_copy.rs`'s `Side::Pages`): the GPU copies into a staging
//!   buffer of the range's length, which the command buffer's completion copies into the pages,
//!   the same bytes in both.
//!
//! Counted for buffers over the caller's pages, as the rest of their copies are
//! (`WorkCounts::managed_bytes_copied_out`, `managed_synchronizes_unwritten`,
//! `managed_bytes_unwritten_skipped`).

use super::buffer::{Buffer, WeakBuffer};
use super::deferred::Deferred;
use super::device::lock;
use super::encoder::Segment;
use super::key_hash::KeyMap;
use super::managed_copy::CopyBack;
use super::resource::ResourceKey;
use super::tracker::{Barrier, Point, Summary, UseRange};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

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
    /// What its encoding decided, until its batch's submission settles it.
    encoded: Mutex<Option<Encoded>>,
    /// What it copies, once settled.
    decided: Mutex<Option<Decided>>,
}

/// What a copy out's encoding decided (the module docs, **The copy's place**).
struct Encoded {
    /// The bytes of its range it copies in place.
    regions: Vec<Range<u64>>,
    /// It copies its range whole.
    whole: bool,
    /// Its decision's stamp among the writes noted ([`GpuWrites`]).
    stamp: u64,
    /// Its buffer's `VkBuffer` was handed out then.
    handed_out: bool,
    /// The `VkCommandBuffer` its copies went into; `None` until its encoder recorded them.
    raw: Option<vk::CommandBuffer>,
}

/// What a copy out copies: the bytes `regions` of its range, in order and apart; and where its
/// batch's submission settled it, what its completion lets go of.
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
            encoded: Mutex::new(None),
            decided: Mutex::new(None),
        }
    }

    /// Whether the GPU's writes to its buffer are kept (the module docs, **Copied whole**): a
    /// tracked buffer whose memory is not exported. Its `VkBuffer`'s handing out is read as each
    /// copy is decided.
    pub(super) fn writes_known(&self) -> bool {
        self.buffer.identity().tracked && !self.buffer.memory_exported()
    }

    /// Its buffer's key.
    pub(super) fn key(&self) -> ResourceKey {
        self.buffer.identity().key
    }

    /// Copy its range whole, decided now (the module docs, **Copied whole**).
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

    /// Decide its copy as it is encoded, by `writes` and `before`, the bytes of the buffer the
    /// commands of its command buffer before it may write, `None` where some may write it unknown
    /// (the module docs, **The copy's place**).
    pub(super) fn decide(&self, writes: &GpuWrites, before: Option<&[Range<u64>]>) {
        let handed_out = self.buffer.handed_out().load(Ordering::Acquire);
        let (regions, stamp, whole) = writes.decide(self, before, handed_out);
        *lock(&self.encoded) = Some(Encoded {
            regions,
            whole,
            stamp,
            handed_out,
            raw: None,
        });
    }

    /// Record the copies its encoding decided ([`decide`](Self::decide)) into `command_buffer`.
    ///
    /// # Safety
    ///
    /// As [`record_whole`](Self::record_whole)'s.
    pub(super) unsafe fn record_decided(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let mut encoded = lock(&self.encoded);
        let Some(encoded) = encoded.as_mut() else {
            return;
        };
        // SAFETY: the caller's contract.
        unsafe { self.record(raw, command_buffer, &encoded.regions) };
        encoded.raw = Some(command_buffer);
    }

    /// Whether its encoding decided its copy and recorded it into `raws`, and its batch's
    /// submission has not settled it yet.
    pub(super) fn unsettled_in(&self, raws: &[vk::CommandBuffer]) -> bool {
        lock(&self.encoded)
            .as_ref()
            .and_then(|encoded| encoded.raw)
            .is_some_and(|raw| raws.contains(&raw))
    }

    /// Settle its copy at its batch's submission (the module docs, **What the encoding cannot
    /// see**), by `writes`, of command buffer `by`, as `completed` says which batches the host saw
    /// complete: the bytes to copy before its part, none where every write noted since its
    /// encoding is copied in place.
    pub(super) fn settle(
        &self,
        writes: &GpuWrites,
        by: u64,
        completed: &dyn Fn(Point) -> bool,
    ) -> Vec<Range<u64>> {
        let Some(encoded) = lock(&self.encoded).take() else {
            return Vec::new();
        };
        let handed_out = self.buffer.handed_out().load(Ordering::Acquire);
        let (top_up, retire, whole) = writes.settle(self, &encoded, by, handed_out, completed);
        if self.counted && !top_up.is_empty() {
            (self.device.shared().counts).managed_synchronize_topped_up();
        }
        let both = top_up.iter().chain(&encoded.regions);
        let regions = merged(both, &self.range);
        self.decided(regions, Some(retire), encoded.whole || whole);
        top_up
    }

    /// Record the copies of `regions` of its range that its batch's submission settled before its
    /// part, between barriers from everything before and to everything after (the module docs,
    /// **What the encoding cannot see**).
    ///
    /// # Safety
    ///
    /// As [`record_whole`](Self::record_whole)'s; `command_buffer` goes to the queue just before
    /// the part that holds the copy's command.
    pub(super) unsafe fn record_top_up(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        regions: &[Range<u64>],
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.record(raw, command_buffer, regions);
            Barrier::everything().record(raw, command_buffer);
        }
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
    /// are let go of (the module docs). Nothing where it failed, or where its batch's submission
    /// never settled it (its encoder was encoded again, `command_buffer/reencode.rs`, or its part
    /// ran nothing).
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
    /// The batches that may write memory no use names whose completion no settling saw yet, by
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
    /// Its writes, in the order noted.
    writes: Vec<Write>,
    /// The batches that may write memory no use names noted at or before this stamp ran before a
    /// copy out that completed.
    unnamed_through: u64,
}

/// One write kept ([`Written`]).
struct Write {
    bytes: Range<u64>,
    /// The stamp it was noted at.
    stamp: u64,
    /// The command buffer whose batch wrote it (its object ID), 0 for writes of several or none.
    by: u64,
}

impl GpuWrites {
    /// Whether some buffer's writes are kept: a batch's writes are noted only then.
    pub(super) fn keeps_any(&self) -> bool {
        self.any.load(Ordering::Acquire)
    }

    /// A batch's parts of command buffer `by` were submitted: note the tracked writes `summaries`
    /// hold of the buffers kept, under one stamp.
    pub(super) fn note<'a>(
        &self,
        summaries: impl Iterator<Item = &'a Summary<ResourceKey>>,
        by: u64,
    ) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        for (key, range) in summaries.flat_map(Summary::writes) {
            let Some(written) = known.buffers.get_mut(&key) else {
                continue;
            };
            written.add(bytes_of(range), stamp, by);
        }
    }

    /// A batch at `at` of a command buffer that may write memory no use names was submitted:
    /// every buffer is copied whole until a copy out settled once the host saw it complete
    /// completed. Those the host saw complete (`completed`) go from the list.
    pub(super) fn note_unnamed(&self, at: Point, completed: &dyn Fn(Point) -> bool) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        known.unnamed_last = stamp;
        known.unnamed.retain(|(_, at)| !completed(*at));
        known.unnamed.push((stamp, at));
    }

    /// The regions of `copy`'s range its encoding copies, the decision's stamp, and whether it
    /// copies its range whole for writes not known: the writes kept of its buffer and `before`,
    /// its command buffer's before it, `None` where not known; `handed_out` where its buffer's
    /// `VkBuffer` was handed out. Its buffer's writes are kept from now on.
    fn decide(
        &self,
        copy: &SyncCopy,
        before: Option<&[Range<u64>]>,
        handed_out: bool,
    ) -> (Vec<Range<u64>>, u64, bool) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        let unnamed_last = known.unnamed_last;
        if !known.buffers.contains_key(&copy.key()) {
            known.sweep();
        }
        let Known { buffers, .. } = &mut *known;
        // A buffer's first: its writes before now are not known.
        let first = !buffers.contains_key(&copy.key());
        let written = buffers.entry(copy.key()).or_insert_with(|| Written {
            of: copy.buffer.downgrade(),
            writes: vec![Write {
                bytes: 0..u64::MAX,
                stamp,
                by: 0,
            }],
            unnamed_through: 0,
        });
        self.any.store(true, Ordering::Release);
        let range = &copy.range;
        let known = !first && !handed_out && written.unnamed_through >= unnamed_last;
        match (before, known) {
            (Some(before), true) => {
                let writes = (written.writes.iter()).map(|write| &write.bytes);
                (merged(writes.chain(before), range), stamp, false)
            }
            _ => (vec![range.clone()], stamp, true),
        }
    }

    /// The regions of `copy`'s range to copy before its part at its batch's submission, by
    /// command buffer `by`, what its completion lets go of, and whether that copy is its range
    /// whole for writes not known (the module docs, **What the encoding cannot see**): the writes
    /// of other command buffers noted since `encoded`, its range whole where its buffer's
    /// `VkBuffer` was handed out since (`handed_out` now) or a batch that may write memory no use
    /// names was noted since.
    fn settle(
        &self,
        copy: &SyncCopy,
        encoded: &Encoded,
        by: u64,
        handed_out: bool,
        completed: &dyn Fn(Point) -> bool,
    ) -> (Vec<Range<u64>>, Retire, bool) {
        let mut known = lock(&self.known);
        known.stamp += 1;
        let stamp = known.stamp;
        known.unnamed.retain(|(_, at)| !completed(*at));
        // Every batch that may write memory unnamed noted at or before `through` completed.
        let through = known.unnamed.first().map_or(stamp, |(noted, _)| noted - 1);
        let retire = Retire { stamp, through };
        if encoded.whole {
            return (Vec::new(), retire, false);
        }
        let range = &copy.range;
        let unknown = known.unnamed_last > encoded.stamp || (handed_out && !encoded.handed_out);
        match (unknown, known.buffers.get(&copy.key())) {
            (false, Some(written)) => {
                let since = (written.writes.iter())
                    .filter(|write| write.stamp > encoded.stamp && write.by != by)
                    .map(|write| &write.bytes);
                (merged(since, range), retire, false)
            }
            _ => (vec![range.clone()], retire, true),
        }
    }

    /// A copy out settled at `retire.stamp` completed: let go of the writes of the buffer of `key`
    /// noted at or before it, and of the batches that may write memory unnamed noted at or before
    /// `retire.through`.
    fn retire(&self, key: ResourceKey, retire: Retire) {
        let mut known = lock(&self.known);
        if let Some(written) = known.buffers.get_mut(&key) {
            written.writes.retain(|write| write.stamp > retire.stamp);
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
    /// Note a write of `bytes` at `stamp`, the latest, by command buffer `by`: the writes before
    /// it inside its bytes go, as it is let go of no earlier than they are; past [`WRITES_KEPT`],
    /// all of them are one, of the command buffer they all share, else of none.
    fn add(&mut self, bytes: Range<u64>, stamp: u64, by: u64) {
        if bytes.is_empty() {
            return;
        }
        (self.writes)
            .retain(|kept| !(bytes.start <= kept.bytes.start && kept.bytes.end <= bytes.end));
        self.writes.push(Write { bytes, stamp, by });
        if self.writes.len() > WRITES_KEPT {
            let start = self.writes.iter().map(|kept| kept.bytes.start).min();
            let end = self.writes.iter().map(|kept| kept.bytes.end).max();
            let (start, end) = (start.unwrap_or(0), end.unwrap_or(0));
            let by = match self.writes.iter().all(|kept| kept.by == by) {
                true => by,
                false => 0,
            };
            self.writes.clear();
            self.writes.push(Write {
                bytes: start..end,
                stamp,
                by,
            });
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

/// A tracked use's bytes: an image's, the whole of its key.
fn bytes_of(range: &UseRange) -> Range<u64> {
    match range {
        UseRange::Bytes(bytes) => bytes.clone(),
        UseRange::Image { .. } => 0..u64::MAX,
    }
}

/// Add to `into` the bytes of the buffer of `key` the writes `summary` holds write.
pub(super) fn writes_of(
    summary: &Summary<ResourceKey>,
    key: ResourceKey,
    into: &mut Vec<Range<u64>>,
) {
    let writes = summary.writes().filter(|(written, _)| *written == key);
    into.extend(writes.map(|(_, range)| bytes_of(range)));
}

/// Add to `into` the bytes of the buffer of `key` that `segment`, of an encoder before a copy out's
/// command or of its own encoder, may write beyond its encoder's tracked uses, as far as the
/// encoding knows on `device`: what its recording's or its known summary holds. `false` where its
/// recording finds its uses or may write memory no use names (`Deferred::reaches_unnamed`).
pub(super) fn segment_writes(
    segment: &Segment,
    device: &Device,
    key: ResourceKey,
    into: &mut Vec<Range<u64>>,
) -> bool {
    if let Some(done) = &segment.recorded {
        writes_of(&done.summary, key, into);
        return true;
    }
    if Deferred::reaches_unnamed(&segment.work) {
        return false;
    }
    if let Some(summary) = segment.work.known_summary(device) {
        writes_of(&summary, key, into);
    }
    true
}

/// Add to `into` the bytes of the buffer of `key` the uses `ranges` of an encoder write.
pub(super) fn ranges_of<'a>(
    ranges: impl Iterator<Item = &'a UseRange>,
    into: &mut Vec<Range<u64>>,
) {
    into.extend(ranges.map(bytes_of));
}
