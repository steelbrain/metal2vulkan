//! Copies between a guest RAM and a buffer, by the GPU, from the device's imports of the RAM's
//! chunks (a runtime extension: Metal has none): [`BlitCommandEncoder::copy_from_host_ram`]
//! gathers runs of the RAM into consecutive bytes of a buffer, and
//! [`BlitCommandEncoder::copy_to_host_ram`] scatters them back;
//! [`BlitCommandEncoder::copy_runs_from_host_ram`] and
//! [`BlitCommandEncoder::copy_runs_to_host_ram`] do the same with each run at a byte of the
//! buffer of its own (a buffer that keeps a surface's layout, its rows at a pitch). A VMM whose
//! guest's surface is a scatter of guest pages copies it with no view of the pages and, where the
//! RAM's chunks are imported already (`host_ram.rs`), no import of its own.
//!
//! - **The runs** are host addresses in the RAM's ranges, each starting on a 4-byte word ([`WORD`])
//!   and ending at any byte past it: a surface's whole pages, or each of its rows' texels alone,
//!   whose last bytes the guest's CPU may own past them. Each is at a byte of the buffer ([`Run`]):
//!   one after another from the call's offset ([`one_after_another`]), or each at its own. A run
//!   that starts where the one before it ended, in the RAM and in the buffer, is one run with it
//!   ([`merged`]); a run that starts before (a descending neighbour) stays apart. Hazard tracking
//!   takes the buffer's bytes from the lowest run's first to the highest's last as the copy's; the
//!   bytes between the runs (a row's padding) are neither read nor written. Each run is cut where a
//!   chunk ends ([`pieces`]), and each chunk the pieces touch is taken as an object over it takes
//!   it (`RamOnDevice::for_copy`): its import where it is made; else, where an object over it would
//!   be bound in its own run's import (`RamOnDevice::for_object`), an import of each of the copy's
//!   ranges of it alone (those less than 2 MiB apart one, at most [`MOST_ALONE`]:
//!   [`alone_ranges`]), made on the encoding thread, which faults in and pins those pages and no
//!   more, while the device's thread imports the chunk (the thread's window moved past it, or,
//!   where each listed import costs every submission, the chunk asked for): so no copy waits for a
//!   chunk's import. Those imports are held by the command buffer until it completed, then freed
//!   (`WorkCounts::run_imports_live`). Elsewhere (the thread stopped, failed the chunk or is not
//!   started, the allocation limit's margin, `RuntimeDriverDefect::AliasedImportBlitsUnordered`,
//!   more ranges than [`MOST_ALONE`], a range that is the whole chunk, a range's import that fails)
//!   the copy imports the chunk, or waits for its import under way, which is the chunk's one import
//!   (`Device::import_chunk`), made and counted as an object's is.
//! - **Runs of words: one dispatch per chunk.** Where every run also ends on a word, and lies at a
//!   word of the buffer's memory ([`in_words`]: a surface's whole pages, or the rows of one of
//!   4-byte texels), the pieces become a table of runs of at most a 4 KiB page each, each its first
//!   word in the chunk's import, its first word in the buffer and its words ([`plan`]: whole pages
//!   an entry a page), uploaded into the command buffer's arena, and the runs of each chunk one
//!   dispatch of `bind::pages_copy_module` (a gather or a scatter) over the chunk's whole
//!   `VkBuffer`, the buffer's bytes and the table: as many dispatches as chunks the runs touch,
//!   never one region per run, and no other word of the runs' pages read or written. A copy whose
//!   bytes pass the device's storage range, or whose runs of one chunk pass its workgroup count,
//!   takes more (each window of the buffer's bytes, each part of the runs). Counted:
//!   `WorkCounts::host_ram_copies`, `host_ram_copy_bytes` and `host_ram_copy_dispatches`.
//! - **Other runs: regions.** Where a run ends off a word (a row of 1- or 2-byte texels) or lies at
//!   a byte of the buffer off a word, or the runs write a destination byte twice (a scatter's that
//!   name a host byte twice, a gather's that name a byte of the buffer twice: [`share_a_byte`]),
//!   each piece is a region of `vkCmdCopyBuffer` between its import (the chunk's, or the pages it
//!   covers imported alone, as above: whole pages, imported as now) and the buffer, which takes
//!   byte offsets and sizes: the copy reads or writes the runs' bytes and no other byte of their
//!   pages, a row's padding and the pages around a surface left as they were. The regions of one
//!   import are one command, at most [`MOST_REGIONS`] each. Runs that name a destination byte twice
//!   take a round of commands for each ([`rounds`]), no byte written by two regions of one command,
//!   the later run's bytes kept; and a buffer over host pages the runs' bytes overlap is refused,
//!   no region reading and writing one byte. Counted: `WorkCounts::host_ram_copies`,
//!   `host_ram_copy_bytes` and `host_ram_copy_regions`.
//! - **Ordered as a copy between the buffer and a no-copy buffer over the pages.** Each chunk a
//!   copy uses has a key of its own, its uses' bytes those of the chunk, which hazard tracking
//!   places over the chunk's host bytes (`host_aliases.rs`'s placements), as it places every
//!   no-copy buffer and heap over the bytes it lies over: so the copy's uses (a read of the
//!   chunks' bytes it gathers, a write of those it scatters, and the buffer's write or read) are
//!   ordered, in the encoder and between encoders, command buffers and queues, against every
//!   other use of the same host bytes: the other copies', and those of a no-copy object over the
//!   RAM's own addresses. A chunk's uses name the bytes the runs touch, joined where they are more
//!   than [`MOST_RANGES`] (the gaps between the closest taken too), so a copy of thousands of runs
//!   takes a few uses, and orders more than its bytes need, never less.
//! - **Seen as Metal shows a no-copy buffer's bytes.** The chunks are host-coherent imports of the
//!   pages: the CPU's writes before the commit are seen by the copy (the submission makes them
//!   visible), and the GPU's are in the pages once the command buffer completed (every batch ends
//!   with a barrier to the host). Where the RAM came with a log of the writes to its pages
//!   (`HostRam::with_dirty_pages`), the chunk's key is the log's, so a scatter's writes are
//!   reported to it as the GPU's in place (`dirty_pages.rs`): every copy of those pages (a copied
//!   host buffer's, a shadow's, whose logs are the RAM's where their runs are) copies them in.
//! - **What is refused** ([`Problem::HostRam`]): a run that is empty, starts off a 4-byte word, or
//!   lies outside the RAM's ranges (or, one after another, past `u64`); a RAM the device does not
//!   use; of runs one after another, bytes of the buffer that start off a 4-byte word of its memory
//!   (runs each at a byte of their own are copied by regions there); a copy by regions with a
//!   buffer over host pages its bytes' span overlaps. A device without
//!   `VK_EXT_external_memory_host` uses no RAM (`Unsupported`, naming it); the buffer's bytes past
//!   its length are a range past it ([`Problem::RangeOutOfBounds`]: of runs each at a byte of their
//!   own, the first run's past it). A chunk the device cannot import fails the copy with the
//!   import's error.

use super::{HostRam, RamChunk, RamOnDevice};
use crate::blit::{bytes_in, reads_by, writes_by, BlitCommandEncoder, TRANSFER};
use crate::buffer::{Buffer, CallerMemory};
use crate::dirty_pages::PagesLog;
use crate::error::{invalid, Error, Problem};
use crate::owned_kernel::{OwnedBinds, OwnedModule, OwnedSet};
use crate::replay::BlitCall;
use crate::resource::ResourceKey;
use crate::tracker::{Access, Use, UseRange};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{PagesCopy, PagesCopyModule};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// The unit of a copy's runs: whole 4 KiB pages at 4 KiB-aligned host addresses (a guest's page;
/// every host page and every chunk is whole such pages).
pub(crate) const PAGE: u64 = PagesCopyModule::PAGE_BYTES as u64;

/// The most ranges of one chunk's bytes a copy's uses name; past them the closest are joined.
/// So a copy of thousands of runs keeps the command buffer's writes apart (`dirty_pages.rs`
/// keeps 4,096 writes of a command buffer and 64 ranges of a log apart, past which it takes every
/// copy, or every page, as written).
const MOST_RANGES: usize = 16;

/// The most ranges of a chunk not imported yet a copy imports alone while the device imports the
/// chunk (`RamOnDevice::for_copy`): each import is a fixed cost on the encoding thread (a 2 MiB
/// run's took 0.75 to 0.83 ms on the Radeon 780M and RX 7900 XTX), so a copy whose pages there
/// are more ranges than this imports the chunk, or waits for its import under way.
pub(crate) const MOST_ALONE: usize = 16;

/// The bytes between two of a chunk's ranges a copy imports alone below which they are one
/// import ([`alone_ranges`]): a huge page's.
const JOIN_GAP: u64 = 2 << 20;

/// The keys a RAM's copies use its chunks' bytes by on one device: one per chunk, made at the
/// first copy over it, placed over the chunk's host bytes for hazard tracking and, where the RAM
/// came with a log, kept as that log's (the module docs). Their places are let go of as the
/// RAM's use on the device ends ([`ChunkKeys::unplace`]).
pub(crate) struct ChunkKeys {
    keys: Box<[OnceLock<ChunkKey>]>,
}

/// One chunk's key, and the log of its pages' writes its uses report to.
struct ChunkKey {
    key: ResourceKey,
    /// Kept while the key lives: the device's table of logs holds it weakly.
    _log: Option<Arc<PagesLog>>,
}

impl ChunkKeys {
    /// The keys of `chunks` chunks, none made yet.
    pub(crate) fn new(chunks: usize) -> ChunkKeys {
        ChunkKeys {
            keys: (0..chunks).map(|_| OnceLock::new()).collect(),
        }
    }

    /// The RAM's use on `device` ended (`HostRamUse`'s drop): the keys lie over no host bytes
    /// from now on, so no later use of those bytes takes theirs. The commands recorded before
    /// keep the uses they took.
    pub(crate) fn unplace(&self, device: &Device) {
        let aliases = &device.shared().aliases;
        for key in self.keys.iter().filter_map(OnceLock::get) {
            aliases.unplace(key.key);
        }
    }
}

impl RamOnDevice {
    /// The key of chunk `slot`'s bytes on `device`: made at the first copy over the chunk, and
    /// placed over its host bytes, its uses reported to the RAM's log where it has one.
    fn chunk_key(&self, device: &Device, slot: usize) -> ResourceKey {
        let made = self.copies.keys[slot].get_or_init(|| {
            let shared = device.shared();
            let span = &self.slots[slot].span;
            let key = ResourceKey::new();
            shared
                .aliases
                .place(key, (span.start, span.end - span.start));
            let log = (self.dirty()).and_then(|dirty| {
                PagesLog::new((span.start, span.end - span.start), Some(dirty), None)
            });
            if let Some(log) = &log {
                shared.dirty.logged_in_place(key, log);
            }
            ChunkKey { key, _log: log }
        });
        made.key
    }
}

/// The unit a run starts on: a 4-byte word of host address. Its end may be any byte.
pub(crate) const WORD: u64 = 4;

/// The most regions of one `vkCmdCopyBuffer` of a copy by regions (`record_host_ram_copy`): a
/// copy of more takes more commands, so no one command's array grows with the runs (a laid-out
/// 4K surface's rows are 2160 runs a plane).
pub(crate) const MOST_REGIONS: usize = 1024;

/// A run of a copy: the buffer's byte its first byte is copied from or to, and the run's host
/// bytes.
pub(crate) type Run = (u64, Range<u64>);

/// `runs` laid one after another in the buffer from its byte `offset`, as `copy_from_host_ram`
/// and `copy_to_host_ram` take them; `None` where their bytes pass `u64`.
pub(crate) fn one_after_another(offset: u64, runs: &[Range<u64>]) -> Option<Vec<Run>> {
    let mut at = offset;
    (runs.iter())
        .map(|run| {
            let first = at;
            at = at.checked_add(run.end.saturating_sub(run.start))?;
            Some((first, run.clone()))
        })
        .collect()
}

/// `runs`, each starting on a [`WORD`] and holding at least a byte, with each that starts where
/// the one before it ended, in the RAM and in the buffer, joined to it; `None` where a run is
/// empty or starts off a word.
pub(crate) fn merged(runs: &[Run]) -> Option<Vec<Run>> {
    let mut out: Vec<Run> = Vec::with_capacity(runs.len());
    for (at, run) in runs {
        if run.start >= run.end || !run.start.is_multiple_of(WORD) {
            return None;
        }
        match out.last_mut() {
            Some((last_at, last))
                if last.end == run.start
                    && last_at.checked_add(last.end - last.start) == Some(*at) =>
            {
                last.end = run.end
            }
            _ => out.push((*at, run.clone())),
        }
    }
    Some(out)
}

/// Whether every one of `runs` (each starting on a word, [`merged`]) ends on a [`WORD`] too, and
/// lies from a word of the buffer: a copy of such runs is a dispatch of the pages copy per chunk;
/// any other, regions of `vkCmdCopyBuffer`.
pub(crate) fn in_words(runs: &[Run]) -> bool {
    (runs.iter()).all(|(at, run)| at.is_multiple_of(WORD) && run.end.is_multiple_of(WORD))
}

/// Whether two of `ranges` (a copy's destination bytes: a scatter's host bytes, a gather's in the
/// buffer) share a byte: such a copy writes a byte twice, which only a copy by regions orders
/// ([`rounds`]).
pub(crate) fn share_a_byte(ranges: &[Range<u64>]) -> bool {
    let mut sorted: Vec<(u64, u64)> = (ranges.iter())
        .map(|range| (range.start, range.end))
        .collect();
    sorted.sort_unstable();
    sorted.windows(2).any(|pair| pair[1].0 < pair[0].1)
}

/// The bytes of the buffer `pieces` (at their bytes from the copy's first) lie in.
pub(crate) fn in_buffer(pieces: &[Piece]) -> Vec<Range<u64>> {
    (pieces.iter())
        .map(|piece| piece.at..piece.at + piece.length)
        .collect()
}

/// The pieces of a copy by regions as rounds of `vkCmdCopyBuffer`s, from their destination bytes
/// (a scatter's host bytes, a gather's in the buffer): each round the pieces (by index, in order)
/// whose destination bytes no other piece of the round's touches, so that no byte is written by
/// two regions of one command; a later piece over bytes an earlier one of the round touched
/// starts the next round, ordered after it as a command that writes them again is. Only runs
/// that name a destination byte twice take more than one.
pub(crate) fn rounds(destination: &[Range<u64>]) -> Vec<Vec<usize>> {
    let mut rounds: Vec<Vec<usize>> = Vec::new();
    // The round's destination bytes so far, apart, by start.
    let mut taken: BTreeMap<u64, u64> = BTreeMap::new();
    for (index, range) in destination.iter().enumerate() {
        // Of the round's ranges, the one that starts last before this one ends is the only one
        // that may overlap it: those before it end before it starts.
        let before = taken.range(..range.end).next_back();
        let overlaps = before.is_some_and(|(_, &end)| end > range.start);
        match rounds.last_mut() {
            Some(round) if !overlaps => round.push(index),
            _ => {
                rounds.push(vec![index]);
                taken.clear();
            }
        }
        taken.insert(range.start, range.end);
    }
    rounds
}

/// A run's bytes inside one chunk: `length` bytes from byte `offset` of chunk `slot`, at byte
/// `at` of the buffer (from the copy's first, once recorded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub(crate) slot: usize,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) at: u64,
}

/// `runs` cut where a chunk of `spans` (the RAM's chunks, by start, apart) ends, in order, each
/// piece at its byte of the buffer, and the bytes they total; `None` where a byte of one lies in
/// no chunk (outside the RAM's ranges) or the total passes `u64`.
pub(crate) fn pieces(runs: &[Run], spans: &[Range<u64>]) -> Option<(Vec<Piece>, u64)> {
    let mut out = Vec::with_capacity(runs.len());
    let mut total: u64 = 0;
    for (run_at, run) in runs {
        let mut at = *run_at;
        let mut start = run.start;
        while start < run.end {
            let slot = super::span_of(spans, start)?;
            let span = &spans[slot];
            let end = run.end.min(span.end);
            let length = end - start;
            out.push(Piece {
                slot,
                offset: start - span.start,
                length,
                at,
            });
            at = at.checked_add(length)?;
            total = total.checked_add(length)?;
            start = end;
        }
    }
    Some((out, total))
}

/// One dispatch of a copy: the runs of import `slot` (of the copy's, in order: a chunk's, or a
/// range of one alone) whose bytes in the buffer lie in window `window` ([`plan`]), `entries` of
/// the table from `first_entry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    pub(crate) slot: usize,
    pub(crate) window: u64,
    pub(crate) first_entry: u32,
    pub(crate) entries: u32,
}

/// The table of `pieces` (each a whole number of words from a word, in their imports) and the
/// dispatches over it: each piece cut into runs of at most a [`PAGE`] from its start and where
/// the buffer's windows of `window_bytes` (a multiple of a page) end, each run an entry of its
/// first word in the import its piece's `slot` names, its first word in the buffer counted from
/// its window, and its words; the runs of one import and window together, in the order the
/// pieces give them, at most `most_groups` a dispatch. A piece of whole pages from a page is an
/// entry a page.
pub(crate) fn plan(
    pieces: &[Piece],
    window_bytes: u64,
    most_groups: u32,
) -> Option<(Vec<u8>, Vec<Dispatch>)> {
    let mut groups: BTreeMap<(usize, u64), Vec<[u32; 3]>> = BTreeMap::new();
    for piece in pieces {
        let mut done = 0;
        while done < piece.length {
            let at = piece.at + done;
            let window = at / window_bytes;
            let window_end = (window + 1) * window_bytes;
            let length = (piece.length - done).min(PAGE).min(window_end - at);
            let entry = [
                u32::try_from((piece.offset + done) / WORD).ok()?,
                u32::try_from((at - window * window_bytes) / WORD).ok()?,
                u32::try_from(length / WORD).ok()?,
            ];
            groups.entry((piece.slot, window)).or_default().push(entry);
            done += length;
        }
    }
    let mut table = Vec::new();
    let mut dispatches = Vec::new();
    let mut entry: u32 = 0;
    for ((slot, window), entries) in groups {
        for part in entries.chunks(most_groups.max(1) as usize) {
            let count = u32::try_from(part.len()).ok()?;
            dispatches.push(Dispatch {
                slot,
                window,
                first_entry: entry,
                entries: count,
            });
            for &[scattered, consecutive, words] in part {
                table.extend_from_slice(&PagesCopyModule::entry(scattered, consecutive, words));
            }
            entry = entry.checked_add(count)?;
        }
    }
    Some((table, dispatches))
}

/// The bytes of each chunk `pieces` touch, by chunk: sorted, those that meet or touch as one, and
/// at most `most` (the closest joined, the bytes between them taken too).
pub(crate) fn chunk_ranges(pieces: &[Piece], most: usize) -> BTreeMap<usize, Vec<Range<u64>>> {
    let mut by_chunk: BTreeMap<usize, Vec<Range<u64>>> = BTreeMap::new();
    for piece in pieces {
        let bytes = piece.offset..piece.offset + piece.length;
        by_chunk.entry(piece.slot).or_default().push(bytes);
    }
    (by_chunk.into_iter())
        .map(|(slot, ranges)| (slot, joined(ranges, most)))
        .collect()
}

/// The bytes of each chunk `pieces` touch, by chunk, as a copy imports them alone where the chunk
/// is not imported yet (`RamOnDevice::for_copy`): sorted, those less than [`JOIN_GAP`] apart as
/// one, never joined further.
pub(crate) fn alone_ranges(pieces: &[Piece]) -> BTreeMap<usize, Vec<Range<u64>>> {
    let mut by_chunk: BTreeMap<usize, Vec<Range<u64>>> = BTreeMap::new();
    for piece in pieces {
        let bytes = piece.offset..piece.offset + piece.length;
        by_chunk.entry(piece.slot).or_default().push(bytes);
    }
    for ranges in by_chunk.values_mut() {
        ranges.sort_by_key(|range| range.start);
        let mut out: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
        for range in ranges.drain(..) {
            match out.last_mut() {
                Some(last) if range.start <= last.end.saturating_add(JOIN_GAP - 1) => {
                    last.end = last.end.max(range.end);
                }
                _ => out.push(range),
            }
        }
        *ranges = out;
    }
    by_chunk
}

/// `ranges` sorted, those that meet or touch as one, then the closest joined until at most
/// `most` are left.
fn joined(mut ranges: Vec<Range<u64>>, most: usize) -> Vec<Range<u64>> {
    ranges.sort_by_key(|range| range.start);
    let mut out: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match out.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => out.push(range),
        }
    }
    let most = most.max(1);
    if out.len() <= most {
        return out;
    }
    // Keep the `most - 1` widest gaps; join across the others.
    let mut gaps: Vec<(u64, usize)> = (out.windows(2).enumerate())
        .map(|(at, pair)| (pair[1].start - pair[0].end, at))
        .collect();
    gaps.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut kept: Vec<usize> = gaps[..most - 1].iter().map(|&(_, at)| at).collect();
    kept.sort_unstable();
    let mut result = Vec::with_capacity(most);
    let mut start = out[0].start;
    for at in kept {
        result.push(start..out[at].end);
        start = out[at + 1].start;
    }
    result.push(start..out[out.len() - 1].end);
    result
}

/// A copy between a guest RAM and a buffer as its call gave it ([`BlitCall::HostRamCopy`]), which
/// an encoding again runs again (`replay.rs`).
pub(crate) struct HostRamCall {
    copy: PagesCopy,
    /// The device's use of the RAM; `None` where it uses none such.
    ram: Option<Arc<RamOnDevice>>,
    /// The runs, each at its byte of the buffer: merged where each starts on a word and holds a
    /// byte ([`merged`]), else as given (none where one laid after another passes `u64`).
    runs: Box<[Run]>,
    /// Whether the runs each start on a word and hold a byte (merged).
    checked: bool,
    /// Whether they each end on a word and lie from a word of the buffer besides ([`in_words`]):
    /// copied by the pages copy's dispatches (unless they write a destination byte twice), else
    /// by regions of `vkCmdCopyBuffer`.
    words: bool,
    buffer: Buffer,
    /// For a copy of consecutive bytes of the buffer (`copy_from_host_ram`, `copy_to_host_ram`),
    /// the first: its bytes are checked against the buffer as one range from it, and refused
    /// where it lies off a word of the buffer's memory. `None` for a copy of runs each at a byte
    /// of its own (`copy_runs_from_host_ram`, `copy_runs_to_host_ram`): each run's bytes are
    /// checked against the buffer, and runs off a word of its memory copied by regions.
    consecutive: Option<u64>,
}

impl HostRamCall {
    /// The call that copies as `copy` says between `ram` and `buffer` the runs `runs`, each at its
    /// byte of `buffer`, on `device`; `consecutive` as the field says.
    fn new(
        device: &Device,
        copy: PagesCopy,
        (ram, runs): (&HostRam, Option<Vec<Run>>),
        (buffer, consecutive): (&Buffer, Option<u64>),
    ) -> HostRamCall {
        let ram = (device.shared().host_ram.used().into_iter())
            .find(|used| std::ptr::eq(Arc::as_ptr(&used.ram), ram));
        let (runs, checked) = match runs.as_deref().and_then(merged) {
            Some(merged) => (merged.into_boxed_slice(), true),
            None => (runs.unwrap_or_default().into_boxed_slice(), false),
        };
        let words = checked && in_words(&runs);
        HostRamCall {
            copy,
            ram,
            runs,
            checked,
            words,
            buffer: buffer.clone(),
            consecutive,
        }
    }

    /// The call's name, as its refusals give it.
    fn name(&self) -> &'static str {
        match (self.copy, self.consecutive.is_some()) {
            (PagesCopy::Gather, true) => "copy_from_host_ram",
            (PagesCopy::Scatter, true) => "copy_to_host_ram",
            (PagesCopy::Gather, false) => "copy_runs_from_host_ram",
            (PagesCopy::Scatter, false) => "copy_runs_to_host_ram",
        }
    }

    /// The bytes it holds of its own, besides its size: its runs.
    pub(crate) fn extra(&self) -> usize {
        std::mem::size_of_val(&*self.runs)
    }
}

impl BlitCommandEncoder {
    /// Copy `runs` of the guest RAM `ram` into consecutive bytes of `destination` from
    /// `destination_offset`, in the order given, by the GPU, from the device's imports of the RAM's
    /// chunks: a runtime extension, Metal has none (a program copies from a no-copy buffer over the
    /// pages, `copy_from_buffer_to_buffer`). The runs are host addresses inside the RAM's ranges,
    /// each starting on a 4-byte word and ending at any byte past it. Adjacent ascending runs are
    /// one; where every run also ends on a 4-byte word (whole pages, or rows of 4-byte texels), the
    /// runs of each chunk they touch are one dispatch, and otherwise each run's bytes in a chunk
    /// are one region of a copy; neither reads another byte of their pages (the module docs of
    /// `host_ram/copy.rs`), with no view of the pages and no import of its own. It is ordered and
    /// seen as a copy from a no-copy buffer over the pages into `destination` would be: the CPU's
    /// writes to the pages before the commit are seen, hazard tracking orders it against every
    /// other use of the same host bytes (another copy's, a no-copy buffer's or heap's over the
    /// RAM's own addresses), and `destination` is written as by any blit. An object over another
    /// mapping of the same pages is ordered against it only as the program orders it, as two
    /// no-copy buffers over two mappings of one page are. Where a chunk is not imported yet, the
    /// runs' pages in it are imported alone for the copy while the device imports the chunk
    /// (`WorkCounts::run_imports_live`), as an object over it is bound in its own run's import;
    /// where that cannot be, the chunk is imported first, or its import waited for.
    ///
    /// No runs copy nothing.
    ///
    /// Refused: a device without `VK_EXT_external_memory_host` (`Unsupported`, naming it); a RAM
    /// the device does not use ([`Device::use_host_ram`](crate::Device::use_host_ram)), a run that
    /// is empty, starts off a 4-byte word or lies outside the RAM's ranges, bytes of
    /// `destination` that start off a 4-byte word of its memory, or, of a copy by regions (a run
    /// ending off a word), a `destination` over host pages the runs' bytes overlap (`Invalid`,
    /// `Problem::HostRam`);
    /// bytes past `destination`'s length or a buffer of another device (`Invalid`); and a chunk
    /// the device cannot import (the import's error). A caller falls back to its own copy.
    pub fn copy_from_host_ram(
        &mut self,
        ram: &HostRam,
        runs: &[Range<u64>],
        destination: &Buffer,
        destination_offset: u64,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let runs = one_after_another(destination_offset, runs);
        let call = HostRamCall::new(
            &device,
            PagesCopy::Gather,
            (ram, runs),
            (destination, Some(destination_offset)),
        );
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy the bytes of `source` from `source_offset` out into `runs` of the guest RAM `ram`, in
    /// the order given, by the GPU, into the device's imports of the RAM's chunks: the scatter
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) gathers, a runtime extension (Metal has
    /// none). Ordered and seen as a copy from `source` into a no-copy buffer over the pages
    /// would be: hazard tracking orders it against every other use of the same host bytes, the
    /// GPU's writes are in the pages once the command buffer completed, and where the RAM came
    /// with a log of its pages' writes (`HostRam::with_dirty_pages`) they are reported to it as
    /// the GPU's writes in place, so every copy the runtime keeps of those pages takes them. A
    /// run that starts or ends inside a page writes its bytes alone: the rest of the page keeps
    /// what it held. Runs that name a byte twice leave it as the later wrote it (they are copied
    /// by regions, a round for each).
    ///
    /// Refused as [`copy_from_host_ram`](Self::copy_from_host_ram), `source` in place of the
    /// destination.
    pub fn copy_to_host_ram(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        ram: &HostRam,
        runs: &[Range<u64>],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let runs = one_after_another(source_offset, runs);
        let call = HostRamCall::new(
            &device,
            PagesCopy::Scatter,
            (ram, runs),
            (source, Some(source_offset)),
        );
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy each of `runs` of the guest RAM `ram` into `destination` at its own byte: each run a
    /// byte of `destination` and a host run, whose bytes land from that byte on, in the order
    /// given. The form of [`copy_from_host_ram`](Self::copy_from_host_ram) for a buffer that keeps
    /// a surface's layout (row `r` at `base + r × pitch`, the bytes between the rows left as they
    /// were): one call for every row, a runtime extension (Metal has none). The host runs are as
    /// that call takes them (from a 4-byte word to any byte past it), and a run that starts where
    /// the one before it ended, in the RAM and in `destination`, is one with it. Where every run
    /// also ends on a 4-byte word and its byte of `destination` is a 4-byte word of the buffer's
    /// memory, the runs of each chunk they touch are one dispatch; otherwise, or where two runs
    /// name one byte of `destination` (the later run's bytes are kept), each run's bytes in a
    /// chunk are one region of a copy. Neither reads or writes another byte of the runs' pages
    /// or of `destination`. Ordered and seen as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) is, hazard tracking taking
    /// `destination`'s bytes from the lowest run's first to the highest's last.
    ///
    /// No runs copy nothing.
    ///
    /// Refused as [`copy_from_host_ram`](Self::copy_from_host_ram) is (`Unsupported` without
    /// `VK_EXT_external_memory_host`; a RAM the device does not use, a run that is empty, starts
    /// off a 4-byte word or lies outside the RAM's ranges: `Invalid`, `Problem::HostRam`; a
    /// buffer of another device; a chunk the device cannot import), and: a run whose bytes pass
    /// `destination`'s length (`Invalid`, `Problem::RangeOutOfBounds`, the first such run's); a
    /// copy by regions with a `destination` over host pages its bytes' span overlaps (`Invalid`,
    /// `Problem::HostRam`). A run at a byte of `destination` off a 4-byte word of its memory is
    /// not refused: the copy takes regions. A caller falls back to its own copy.
    pub fn copy_runs_from_host_ram(
        &mut self,
        ram: &HostRam,
        runs: &[(u64, Range<u64>)],
        destination: &Buffer,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let call = HostRamCall::new(
            &device,
            PagesCopy::Gather,
            (ram, Some(runs.to_vec())),
            (destination, None),
        );
        self.host_ram_copy(&Arc::new(call))
    }

    /// Copy the bytes of `source` out into `runs` of the guest RAM `ram`: each run a byte of
    /// `source` and a host run, which takes the bytes from that byte on, in the order given. The
    /// scatter [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram) gathers, and the form
    /// of [`copy_to_host_ram`](Self::copy_to_host_ram) for a buffer that keeps a surface's
    /// layout: one call for every row, a runtime extension (Metal has none). Copied as
    /// [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram) copies, but for runs that name
    /// one host byte twice, which are copied by regions and leave it as the later wrote it; seen
    /// as [`copy_to_host_ram`](Self::copy_to_host_ram) is. A run that starts or ends inside a page
    /// writes its bytes alone: the rest of the page keeps what it held.
    ///
    /// Refused as [`copy_runs_from_host_ram`](Self::copy_runs_from_host_ram), `source` in place
    /// of the destination.
    pub fn copy_runs_to_host_ram(
        &mut self,
        source: &Buffer,
        ram: &HostRam,
        runs: &[(u64, Range<u64>)],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let call = HostRamCall::new(
            &device,
            PagesCopy::Scatter,
            (ram, Some(runs.to_vec())),
            (source, None),
        );
        self.host_ram_copy(&Arc::new(call))
    }

    /// Run `call`, logged first where the encoder logs (`replay.rs`).
    pub(crate) fn host_ram_copy(&mut self, call: &Arc<HostRamCall>) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::HostRamCopy(Arc::clone(call)),
            |this| this.record_host_ram_copy(call),
        )
    }

    /// Check `call` and record it (the module docs).
    fn record_host_ram_copy(&mut self, call: &HostRamCall) -> Result<(), Error> {
        let call_name = call.name();
        let device = self.core().device().clone();
        self.core().check_device(call_name, call.buffer.device())?;
        let shared = device.shared();
        let importer = match &shared.host_import {
            Ok(importer) => importer,
            Err(unsupported) => return Err(Error::Unsupported(unsupported.clone())),
        };
        let refused = || invalid(call_name, Problem::HostRam);
        let ram = (call.ram.as_ref())
            .filter(|ram| shared.host_ram.uses(ram))
            .ok_or_else(refused)?;
        if !call.checked {
            return Err(refused());
        }
        if call.runs.is_empty() {
            return Ok(());
        }
        // The buffer's bytes the copy reads or writes lie in `span`: each run's bytes, checked
        // against the buffer, or, of runs one after another, one range from the first byte,
        // checked once the runs are in the RAM.
        let length = call.buffer.length();
        let mut span = 0..0;
        if call.consecutive.is_none() {
            let (mut low, mut high) = (u64::MAX, 0);
            for (at, run) in &call.runs {
                let bytes = bytes_in(call_name, *at, run.end - run.start, length)?;
                (low, high) = (low.min(bytes.start), high.max(bytes.end));
            }
            span = low..high;
        }
        let spans = ram.spans();
        let (mut pieces, total) = pieces(&call.runs, &spans).ok_or_else(refused)?;
        if let Some(offset) = call.consecutive {
            span = bytes_in(call_name, offset, total, length)?;
        }
        // Each piece at its byte of the copy's bytes, from the first.
        for piece in &mut pieces {
            piece.at -= span.start;
        }
        let (raw, at) = self.core().blit_at(&call.buffer, span.start)?;
        if call.consecutive.is_some() && !at.is_multiple_of(4) {
            return Err(refused());
        }
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let range = u64::from(limits.max_storage_buffer_range);
        let window_bytes = range.saturating_sub(align) / PAGE * PAGE;
        if window_bytes == 0 {
            return Err(crate::error::internal(
                "a storage range shorter than a page of a host RAM copy",
            ));
        }
        let most_groups = limits.max_compute_work_group_count[0];
        // The bytes of each chunk the pieces touch, and the imports they are read or written
        // through, held until the command buffer completes: the chunk's, or, where it is not
        // imported yet, one of each of those bytes alone while the device's thread imports it
        // (`RamOnDevice::for_copy`). Each piece is then counted in its import.
        let touched = chunk_ranges(&pieces, MOST_RANGES);
        let mut imports: Vec<Arc<RamChunk>> = Vec::new();
        let mut of_chunk: BTreeMap<usize, Range<usize>> = BTreeMap::new();
        let mut waited = Duration::ZERO;
        for (slot, ranges) in alone_ranges(&pieces) {
            let (made, took) = ram.for_copy(&device, importer, slot, &ranges)?;
            waited += took;
            if made
                .iter()
                .any(|import| import.span.end - import.span.start > range)
            {
                return Err(crate::error::internal(
                    "a chunk of host RAM past the device's storage range",
                ));
            }
            let first = imports.len();
            imports.extend(made);
            of_chunk.insert(slot, first..imports.len());
        }
        if !waited.is_zero() {
            let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
            shared.counts.host_ram_copy_waited(nanos);
        }
        let in_imports: Option<Vec<Piece>> = (pieces.iter())
            .map(|piece| {
                let host = spans[piece.slot].start + piece.offset;
                let holds = |&index: &usize| {
                    let span = &imports[index].span;
                    span.start <= host && host + piece.length <= span.end
                };
                let index = of_chunk.get(&piece.slot)?.clone().find(holds)?;
                Some(Piece {
                    slot: index,
                    offset: host - imports[index].span.start,
                    ..*piece
                })
            })
            .collect();
        let in_imports = in_imports.ok_or_else(|| {
            crate::error::internal("a host RAM copy's piece outside the imports made for it")
        })?;
        // A copy whose runs write a destination byte twice is ordered by rounds of regions
        // alone: a scatter's that name a host byte twice, a gather's that name a byte of the
        // buffer twice (never runs one after another).
        let twice = || match call.copy {
            PagesCopy::Scatter => {
                let host: Vec<Range<u64>> = (pieces.iter())
                    .map(|piece| {
                        let start = spans[piece.slot].start + piece.offset;
                        start..start + piece.length
                    })
                    .collect();
                share_a_byte(&host)
            }
            PagesCopy::Gather => call.consecutive.is_none() && share_a_byte(&in_buffer(&pieces)),
        };
        if !call.words || !at.is_multiple_of(4) || twice() {
            let pieces = (&pieces[..], &in_imports[..]);
            return self.record_regions(call, ram, pieces, imports, (raw, at, span, total));
        }
        let (table, dispatches) =
            plan(&in_imports, window_bytes, most_groups).ok_or_else(refused)?;
        let kernel = device.owned_kernel(OwnedModule::PagesCopy(call.copy))?;
        let upload = (self.core().command_buffer().state().uploads).upload(
            &device,
            &table,
            table.len() as u64,
        )?;
        let sets: Vec<OwnedSet> = (dispatches.iter())
            .map(|dispatch| {
                let chunk = &imports[dispatch.slot];
                let first = at + dispatch.window * window_bytes;
                let offset = first - first % align;
                let length =
                    (span.end - span.start - dispatch.window * window_bytes).min(window_bytes);
                let buffers = vec![
                    (chunk.raw, 0, chunk.span.end - chunk.span.start),
                    (raw, offset, first - offset + length),
                    (upload.raw(), upload.offset(), table.len() as u64),
                ];
                OwnedSet {
                    images: Vec::new(),
                    storage_images: Vec::new(),
                    buffers,
                }
            })
            .collect();
        let binds = OwnedBinds::new(&device, &kernel, &sets)?;
        let shapes: Vec<([u8; 16], u32)> = (dispatches.iter())
            .map(|dispatch| {
                let first = at + dispatch.window * window_bytes;
                // Below `align`, a multiple of 4: the word of the window's first byte.
                let first_word = ((first % align) / 4) as u32;
                let shape = PagesCopyModule::shape(dispatch.first_entry, first_word);
                (shape, dispatch.entries)
            })
            .collect();
        // The copy's uses: the chunks' bytes it reads or writes, by their keys, and the buffer's.
        let read = Access::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        let write = Access::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let mut uses = Vec::new();
        for (slot, ranges) in touched {
            let key = ram.chunk_key(&device, slot);
            for range in ranges {
                let range = UseRange::Bytes(range);
                uses.push(match call.copy {
                    PagesCopy::Gather => Use::reads(key, true, range, read),
                    PagesCopy::Scatter => Use::writes(key, true, range, write),
                });
            }
        }
        uses.push(match call.copy {
            PagesCopy::Gather => writes_by(&call.buffer, span, write),
            PagesCopy::Scatter => reads_by(&call.buffer, span, read),
        });
        let recorded = dispatches.len();
        self.command_in(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &uses,
            |functions, raw| {
                let point = vk::PipelineBindPoint::COMPUTE;
                // SAFETY: the encoder's command buffer is recording, outside a render pass; the
                // chunks' buffers, the buffer and the arena's region are live until it completes (held
                // below, by `blit_at` and by the command buffer); every binding lies inside its
                // buffer and within the device's storage range, and each entry's words inside its
                // bindings (`plan`, the windows), so the workgroups stay inside them.
                unsafe {
                    functions.cmd_bind_pipeline(raw, point, kernel.pipeline);
                    for (index, (shape, entries)) in shapes.iter().enumerate() {
                        binds.bind(functions, raw, &kernel, index);
                        functions.cmd_push_constants(
                            raw,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            shape,
                        );
                        let [x, y, z] = PagesCopyModule::groups(*entries);
                        functions.cmd_dispatch(raw, x, y, z);
                    }
                }
            },
        )?;
        shared.counts.host_ram_copied(total, recorded as u64);
        self.core().retain(imports);
        if let OwnedBinds::Sets(sets) = binds {
            self.core().retain(sets);
        }
        Ok(())
    }

    /// Record `call`, whose runs end off a word or are a scatter's that write a byte twice, as
    /// regions of `vkCmdCopyBuffer` between `imports` and the buffer's bytes `bytes`, from byte
    /// `at` of its `raw` (the module docs): `pieces` as cut by the RAM's chunks and as they lie in
    /// the imports, one region each, the regions of one import at most [`MOST_REGIONS`] a
    /// command. A scatter whose pieces write a byte twice takes a round of commands for each
    /// ([`rounds`]), each ordered after the one before as a later write of the same bytes is.
    /// Refused (`Problem::HostRam`) where the buffer lies over host pages the pieces' bytes
    /// overlap: no region may read and write one byte.
    fn record_regions(
        &mut self,
        call: &HostRamCall,
        ram: &RamOnDevice,
        (pieces, in_imports): (&[Piece], &[Piece]),
        imports: Vec<Arc<RamChunk>>,
        (raw, at, span, total): (vk::Buffer, u64, Range<u64>, u64),
    ) -> Result<(), Error> {
        let call_name = call.name();
        let device = self.core().device().clone();
        // Each piece's host bytes.
        let host: Vec<Range<u64>> = (in_imports.iter())
            .map(|piece| {
                let start = imports[piece.slot].span.start + piece.offset;
                start..start + piece.length
            })
            .collect();
        if call.buffer.caller_memory() == Some(CallerMemory::HostPages) {
            if let Some(contents) = call.buffer.contents() {
                let start = contents.as_ptr().addr() as u64 + span.start;
                let own = start..start + (span.end - span.start);
                if (host.iter()).any(|range| range.start < own.end && own.start < range.end) {
                    return Err(invalid(call_name, Problem::HostRam));
                }
            }
        }
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        // A scatter writes the pieces' host bytes; a gather the buffer's, each once where the
        // runs lie one after another.
        let by_round = match call.copy {
            PagesCopy::Gather if call.consecutive.is_some() => vec![(0..pieces.len()).collect()],
            PagesCopy::Gather => rounds(&in_buffer(pieces)),
            PagesCopy::Scatter => rounds(&host),
        };
        let mut regions_recorded = 0u64;
        for round in &by_round {
            // The round's regions, by import, in order.
            let mut by_import: BTreeMap<usize, Vec<vk::BufferCopy>> = BTreeMap::new();
            for &index in round {
                let piece = &in_imports[index];
                let (import_at, buffer_at) = (piece.offset, at + piece.at);
                let (src_offset, dst_offset) = match call.copy {
                    PagesCopy::Gather => (import_at, buffer_at),
                    PagesCopy::Scatter => (buffer_at, import_at),
                };
                let region = vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size: piece.length,
                };
                by_import.entry(piece.slot).or_default().push(region);
            }
            // The round's uses: the chunks' bytes it reads or writes, by their keys, and the
            // buffer's.
            let round_pieces: Vec<Piece> = round.iter().map(|&index| pieces[index]).collect();
            let mut uses = Vec::new();
            for (slot, ranges) in chunk_ranges(&round_pieces, MOST_RANGES) {
                let key = ram.chunk_key(&device, slot);
                for range in ranges {
                    let range = UseRange::Bytes(range);
                    uses.push(match call.copy {
                        PagesCopy::Gather => Use::reads(key, true, range, read),
                        PagesCopy::Scatter => Use::writes(key, true, range, write),
                    });
                }
            }
            uses.push(match call.copy {
                PagesCopy::Gather => writes_by(&call.buffer, span.clone(), write),
                PagesCopy::Scatter => reads_by(&call.buffer, span.clone(), read),
            });
            regions_recorded += round.len() as u64;
            let imports = &imports;
            self.command_in(TRANSFER, &uses, |functions, command_buffer| {
                for (slot, regions) in &by_import {
                    let chunk = imports[*slot].raw;
                    let (source, destination) = match call.copy {
                        PagesCopy::Gather => (chunk, raw),
                        PagesCopy::Scatter => (raw, chunk),
                    };
                    for part in regions.chunks(MOST_REGIONS) {
                        // SAFETY: the encoder's command buffer is recording, outside a render
                        // pass; the imports and the buffer are live until it completes (held
                        // below and by `blit_at`); each region lies inside its import (`pieces`,
                        // the imports made over them) and inside the buffer's bytes (`bytes_in`),
                        // and no byte is written by two regions of one command (`rounds`) nor
                        // read and written by one (the buffer apart from the pieces' bytes).
                        unsafe {
                            functions.cmd_copy_buffer(command_buffer, source, destination, part)
                        };
                    }
                }
            })?;
        }
        device.shared().counts.host_ram_copied(total, 0);
        device
            .shared()
            .counts
            .host_ram_copy_regions(regions_recorded);
        self.core().retain(imports);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// `runs` one after another from the buffer's byte 0.
    fn laid(runs: &[Range<u64>]) -> Vec<Run> {
        one_after_another(0, runs).unwrap()
    }

    /// `runs` one after another from byte 0, merged, as host runs alone.
    fn merged_runs(runs: &[Range<u64>]) -> Option<Vec<Range<u64>>> {
        let merged = merged(&laid(runs))?;
        Some(merged.into_iter().map(|(_, run)| run).collect())
    }

    #[test]
    fn adjacent_ascending_runs_merge_and_descending_ones_stay_apart() {
        let runs = [
            0..PAGE,
            PAGE..3 * PAGE,
            // A descending neighbour: it ends where the run before starts.
            9 * PAGE..10 * PAGE,
            8 * PAGE..9 * PAGE,
            7 * PAGE..8 * PAGE,
            // Ascending, but not adjacent.
            20 * PAGE..21 * PAGE,
        ];
        assert_eq!(
            merged_runs(&runs),
            Some(vec![
                0..3 * PAGE,
                9 * PAGE..10 * PAGE,
                8 * PAGE..9 * PAGE,
                7 * PAGE..8 * PAGE,
                20 * PAGE..21 * PAGE,
            ])
        );
        assert_eq!(merged(&[]), Some(vec![]));
        // Runs each at a byte of its own join only where they touch in the buffer too.
        let rows = [
            (0, 4 * PAGE..4 * PAGE + 100),
            (100, 4 * PAGE + 100..4 * PAGE + 200),
            (300, 4 * PAGE + 200..4 * PAGE + 300),
        ];
        assert_eq!(
            merged(&rows),
            Some(vec![
                (0, 4 * PAGE..4 * PAGE + 200),
                (300, 4 * PAGE + 200..4 * PAGE + 300)
            ])
        );
        assert_eq!(one_after_another(u64::MAX - 4, &one(0..8)), None);
    }

    /// One run, as a list.
    fn one(range: Range<u64>) -> Vec<Range<u64>> {
        std::iter::once(range).collect()
    }

    #[test]
    fn runs_off_a_word_or_empty_are_refused_and_ends_take_any_byte() {
        assert_eq!(merged_runs(&one(PAGE..PAGE)), None, "empty");
        assert_eq!(merged_runs(&one(1..PAGE + 1)), None, "a start off a word");
        assert_eq!(
            merged_runs(&one(PAGE + 2..PAGE + 8)),
            None,
            "a start off a word"
        );
        let backwards = Range {
            start: 2 * PAGE,
            end: PAGE,
        };
        assert_eq!(merged_runs(&one(backwards)), None, "backwards");
        // A start on a word, an end at any byte: an 854-texel row of R8 ends 2 bytes past one.
        assert_eq!(merged_runs(&one(16..16 + 854)), Some(one(16..16 + 854)));
        assert_eq!(merged_runs(&one(0..PAGE + 1)), Some(one(0..PAGE + 1)));
        // Rows that touch are one run; rows apart stay apart.
        let rows = [
            2048 * 3..2048 * 3 + 2046,
            2048 * 4..2048 * 4 + 2044,
            2048 * 4 + 2044..2048 * 5,
        ];
        assert_eq!(
            merged_runs(&rows),
            Some(vec![2048 * 3..2048 * 3 + 2046, 2048 * 4..2048 * 5])
        );
        assert!(in_words(&laid(&[0..PAGE, 3 * PAGE..5 * PAGE])));
        assert!(in_words(&laid(&[16..PAGE - 4, 3 * PAGE..5 * PAGE - 8])));
        assert!(!in_words(&laid(&[0..PAGE, 3 * PAGE..5 * PAGE - 2])));
        assert!(!in_words(&laid(&one(16..PAGE - 1))));
        assert!(in_words(&[]));
        // A run at a byte of the buffer off a word.
        assert!(in_words(&[(8, 16..PAGE), (16380, 3 * PAGE..4 * PAGE)]));
        assert!(!in_words(&[(8, 16..PAGE), (16382, 3 * PAGE..4 * PAGE)]));
        // Runs a scatter writes twice, and runs that only touch.
        assert!(share_a_byte(&[8..20, 0..12]));
        assert!(share_a_byte(&[0..100, 40..44]));
        assert!(!share_a_byte(&[8..20, 0..8, 20..24]));
        assert!(!share_a_byte(&[]));
    }

    /// A scatter's pieces take one round where no byte is named twice, and a new round at each
    /// piece over bytes the round took.
    #[test]
    fn a_round_of_regions_writes_no_byte_twice() {
        let rows: Vec<Range<u64>> = (0..4).map(|y| y * 2048..y * 2048 + 2046).collect();
        assert_eq!(rounds(&rows), vec![vec![0, 1, 2, 3]]);
        // The second names the first's last two bytes; the fourth the third's whole, and the
        // first's (an earlier round's) too.
        let twice = [16..100, 96..200, 300..400, 0..400];
        assert_eq!(rounds(&twice), vec![vec![0], vec![1, 2], vec![3]]);
        // Touching ranges share no byte.
        assert_eq!(rounds(&[0..4, 4..8, 8..12]), vec![vec![0, 1, 2]]);
        assert_eq!(rounds(&[]), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn runs_are_cut_where_a_chunk_ends_and_refused_outside_the_ram() {
        let spans = super::super::chunk_spans(&one(0..150 * MIB), super::super::CHUNK_BYTES);
        let runs = [64 * MIB - PAGE..64 * MIB + PAGE, 3 * PAGE..4 * PAGE];
        let (cut, total) = pieces(&laid(&runs), &spans).unwrap();
        assert_eq!(total, 3 * PAGE);
        assert_eq!(
            cut,
            vec![
                Piece {
                    slot: 0,
                    offset: 64 * MIB - PAGE,
                    length: PAGE,
                    at: 0,
                },
                Piece {
                    slot: 1,
                    offset: 0,
                    length: PAGE,
                    at: PAGE,
                },
                Piece {
                    slot: 0,
                    offset: 3 * PAGE,
                    length: PAGE,
                    at: 2 * PAGE,
                },
            ]
        );
        assert_eq!(
            pieces(&laid(&one(150 * MIB..150 * MIB + PAGE)), &spans),
            None
        );
        assert_eq!(
            pieces(&laid(&one(150 * MIB - PAGE..150 * MIB + PAGE)), &spans),
            None
        );
        // Runs each at a byte of their own: each piece at its run's byte, plus its bytes before.
        let rows = [(10 * PAGE, 64 * MIB - 8..64 * MIB + 8), (100, 4..12)];
        let (cut, total) = pieces(&rows, &spans).unwrap();
        assert_eq!(total, 24);
        let at: Vec<(usize, u64, u64, u64)> = (cut.iter())
            .map(|piece| (piece.slot, piece.offset, piece.length, piece.at))
            .collect();
        assert_eq!(
            at,
            vec![
                (0, 64 * MIB - 8, 8, 10 * PAGE),
                (1, 0, 8, 10 * PAGE + 8),
                (0, 4, 8, 100)
            ]
        );
    }

    /// The table's entries as (the import's word, the buffer's word in its window, the words).
    fn entries_of(table: &[u8]) -> Vec<[u32; 3]> {
        (table.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<u32>>()
            .as_chunks::<3>()
            .0
            .to_vec()
    }

    #[test]
    fn a_plan_takes_one_dispatch_per_chunk_with_every_page_in_order() {
        let pieces = [
            Piece {
                slot: 1,
                offset: 2 * PAGE,
                length: 2 * PAGE,
                at: 0,
            },
            Piece {
                slot: 0,
                offset: 0,
                length: PAGE,
                at: 2 * PAGE,
            },
            Piece {
                slot: 1,
                offset: 0,
                length: PAGE,
                at: 3 * PAGE,
            },
        ];
        let (table, dispatches) = plan(&pieces, 1 << 32, 65535).unwrap();
        // Whole pages: an entry a page, of a page's 1024 words.
        assert_eq!(
            entries_of(&table),
            vec![
                [0, 2048, 1024],
                [2048, 0, 1024],
                [3072, 1024, 1024],
                [0, 3072, 1024]
            ]
        );
        assert_eq!(
            dispatches,
            vec![
                Dispatch {
                    slot: 0,
                    window: 0,
                    first_entry: 0,
                    entries: 1,
                },
                Dispatch {
                    slot: 1,
                    window: 0,
                    first_entry: 1,
                    entries: 3,
                },
            ]
        );
        // Windows of two pages, and at most two workgroups a dispatch.
        let (_, dispatches) = plan(&pieces, 2 * PAGE, 2).unwrap();
        let shapes: Vec<(usize, u64, u32, u32)> = (dispatches.iter())
            .map(|d| (d.slot, d.window, d.first_entry, d.entries))
            .collect();
        assert_eq!(shapes, vec![(0, 1, 0, 1), (1, 0, 1, 2), (1, 1, 3, 1)]);
    }

    /// Runs of words: each cut into runs of at most a page from its start, and where a window of
    /// the buffer ends.
    #[test]
    fn a_plan_cuts_runs_of_words_at_a_page_and_where_a_window_ends() {
        let pieces = [
            // A row of 2044 bytes 16 bytes into a page.
            Piece {
                slot: 0,
                offset: 16,
                length: 2044,
                at: 0,
            },
            // Two pages and 100 bytes from a word into a page, across the window's end at 2
            // pages of the buffer.
            Piece {
                slot: 0,
                offset: 5 * PAGE + 4,
                length: 2 * PAGE + 100,
                at: 2044,
            },
            // A word, in the second window.
            Piece {
                slot: 0,
                offset: 9 * PAGE,
                length: 4,
                at: 2044 + 2 * PAGE + 100,
            },
        ];
        let (table, dispatches) = plan(&pieces, 2 * PAGE, 65535).unwrap();
        assert_eq!(
            entries_of(&table),
            vec![
                [4, 0, 511],
                // The second piece: a page from its start, what is left of the window, then the
                // rest from the next window's first word.
                [5121, 511, 1024],
                [6145, 1535, 513],
                [6658, 0, 536],
                [9216, 536, 1],
            ]
        );
        let shapes: Vec<(usize, u64, u32, u32)> = (dispatches.iter())
            .map(|d| (d.slot, d.window, d.first_entry, d.entries))
            .collect();
        assert_eq!(shapes, vec![(0, 0, 0, 3), (0, 1, 3, 2)]);
    }

    #[test]
    fn a_chunks_uses_join_touching_bytes_and_the_closest_past_the_most() {
        let piece = |offset: u64, length: u64| Piece {
            slot: 0,
            offset,
            length,
            at: 0,
        };
        // Descending neighbours touch: one range.
        let down = [
            piece(3 * PAGE, PAGE),
            piece(2 * PAGE, PAGE),
            piece(PAGE, PAGE),
        ];
        assert_eq!(chunk_ranges(&down, 4)[&0], vec![PAGE..4 * PAGE]);
        // Five apart, at most three: the two narrowest gaps join.
        let apart = [
            piece(0, PAGE),
            piece(2 * PAGE, PAGE),
            piece(10 * PAGE, PAGE),
            piece(12 * PAGE, PAGE),
            piece(30 * PAGE, PAGE),
        ];
        assert_eq!(
            chunk_ranges(&apart, 3)[&0],
            vec![0..3 * PAGE, 10 * PAGE..13 * PAGE, 30 * PAGE..31 * PAGE]
        );
    }

    #[test]
    fn a_copy_s_own_imports_join_bytes_less_than_a_huge_page_apart_and_no_more() {
        let piece = |slot: usize, offset: u64| Piece {
            slot,
            offset,
            length: PAGE,
            at: 0,
        };
        // Out of order, two within 2 MiB of each other, one 2 MiB past the second's end, and
        // one in another chunk.
        let pieces = [
            piece(0, 3 * MIB),
            piece(0, 0),
            piece(0, MIB),
            piece(1, 0),
            piece(0, 3 * MIB + PAGE + 2 * MIB),
        ];
        let ranges = alone_ranges(&pieces);
        assert_eq!(
            ranges[&0],
            vec![
                0..3 * MIB + PAGE,
                3 * MIB + PAGE + 2 * MIB..3 * MIB + 2 * PAGE + 2 * MIB
            ]
        );
        assert_eq!(ranges[&1], vec![0..PAGE]);
    }
}
