//! A Managed buffer's changed ranges, captured at the commit: where a Managed buffer has a device
//! copy (a device whose Vulkan device type is neither integrated nor CPU), `didModifyRange:` only
//! names a range ([`Changes::add`]), and the next commit on any queue of a command buffer with
//! commands copies the consumer's bytes of every range named since the last such commit, once
//! each, on the committing thread, before the call returns
//! ([`Work::capture`](crate::submit::Work::capture)). Commits take their turns one at a time, so
//! the command buffers keep the order of their captures. The copies into the device copies, and
//! then into the images of the textures over the ranges in images of their own
//! (`crate::over::ManagedOvers`), are recorded into command buffers of their own, [`CopyIn`]s,
//! which go ahead of the command buffer's own commands in queue order, each after a barrier that
//! puts every earlier command on the queue first. So a command buffer reads the bytes as they were
//! at its commit, and a write after the commit reaches none committed before it. Metal takes them
//! shortly after the commit, before the command buffer is scheduled, and a write or a
//! `didModifyRange:` made while the command buffer runs does not reach it.
//!
//! The bytes go into the capture ring: one host-visible allocation of [`CAPTURE_LEN`] bytes of the
//! Shared type, made at the first capture and kept for the device's life. A span is free again
//! once the submission that carries its copy has completed. A capture that finds no room waits for
//! the oldest span's submission. Where the oldest span is of a copy not yet submitted, the queues'
//! held groups go at once, and once everything committed before is submitted, the capture's own
//! ring-full goes as a copy-in submission of its own, which the commit waits for before it reuses
//! the ring: the ring is all the staging memory a capture holds, whatever it names. Where a command
//! buffer committed before is held on the host (an event wait the host has not met, or a command
//! buffer enqueued ahead of it and not committed), nothing more can go before the host acts, so the
//! rest of the capture takes a staging block of its own from the pool ([`overflow`]), which the
//! command buffer gives back as it retires.

use crate::buffer::DeviceCopy;
use crate::counters::{Counters, Gauge, Metric};
use crate::device::Core;
use crate::error::Error;
use crate::memory::{Block, Mapping, MemoryClass};
use crate::over::{self, Back, ManagedOvers};
use crate::record::{self, Reach, Recorder};
use crate::staging::{At, Region};
use crate::texture::{Image, OverCopy};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// The capture ring's bytes.
pub(crate) const CAPTURE_LEN: u64 = 64 << 20;

/// The longest staging block of its own a capture takes ([`overflow`]); a longer rest takes more.
const OVERFLOW_BLOCK: u64 = 256 << 20;

/// A Managed buffer's ranges named since the last capture.
#[derive(Clone)]
pub(crate) struct Modified {
    pub(crate) copy: Arc<DeviceCopy>,
    /// The block of the consumer's bytes, held, and the buffer's first byte in its mapping.
    pub(crate) cpu: Arc<Block>,
    pub(crate) contents: Mapping,
    /// The buffer's offset in `cpu`.
    pub(crate) offset: u64,
    /// Its textures over the device copy in images of their own.
    pub(crate) overs: Arc<ManagedOvers>,
    /// `start..end` in the buffer, by start, no two overlapping or touching.
    pub(crate) ranges: Vec<(u64, u64)>,
}

/// A Managed buffer whose range is named: its device copy, its consumer's bytes (the block, the
/// buffer's first byte in its mapping, and its offset in the block) and its textures over the
/// device copy in images of their own.
pub(crate) struct Named<'a> {
    pub(crate) copy: &'a Arc<DeviceCopy>,
    pub(crate) cpu: &'a Arc<Block>,
    pub(crate) contents: Mapping,
    pub(crate) offset: u64,
    pub(crate) overs: &'a Arc<ManagedOvers>,
}

/// The device's ranges named and not yet captured, in the order of each buffer's first, and the
/// capture ring.
#[derive(Default)]
pub(crate) struct Changes {
    modified: Vec<Modified>,
    /// Each buffer's place in `modified`, by its device copy's address.
    at: HashMap<usize, usize>,
    /// The ring, from the first capture on.
    pub(crate) region: Option<Region>,
    /// What a commit captured before it failed: the next commit's command buffer takes it first.
    pub(crate) carried: Vec<CopyIn>,
}

impl Changes {
    /// Names `start..end` of `buffer`.
    pub(crate) fn add(&mut self, buffer: Named<'_>, start: u64, end: u64) {
        let key = Arc::as_ptr(buffer.copy) as usize;
        match self.at.get(&key) {
            Some(&index) => merge(&mut self.modified[index].ranges, start, end),
            None => {
                self.at.insert(key, self.modified.len());
                self.modified.push(Modified {
                    copy: Arc::clone(buffer.copy),
                    cpu: Arc::clone(buffer.cpu),
                    contents: buffer.contents,
                    offset: buffer.offset,
                    overs: Arc::clone(buffer.overs),
                    ranges: vec![(start, end)],
                });
            }
        }
    }

    /// The ranges named since the last capture, taken.
    pub(crate) fn take(&mut self) -> Vec<Modified> {
        self.at.clear();
        std::mem::take(&mut self.modified)
    }

    /// Whether a range is named.
    pub(crate) fn named(&self) -> bool {
        !self.modified.is_empty()
    }

    /// A capture that read `spans` of the ring for `modified` will not run: its spans are given up
    /// and its ranges named again, for the next commit to capture.
    pub(crate) fn give_up(&mut self, spans: &[u64], modified: &[Modified]) {
        if let Some(region) = &mut self.region {
            for &number in spans {
                region.ring.set(number, At::Done);
            }
        }
        for modified in modified {
            for &(start, end) in &modified.ranges {
                self.add(modified.named(), start, end);
            }
        }
    }

    /// At the device's end: the ranges named are forgotten, and the copy-ins carried come back,
    /// for their recorders to be given back.
    pub(crate) fn clear(&mut self) -> (Vec<Modified>, Vec<CopyIn>) {
        (self.take(), std::mem::take(&mut self.carried))
    }
}

impl Modified {
    /// The buffer, to name a range of it again.
    fn named(&self) -> Named<'_> {
        Named {
            copy: &self.copy,
            cpu: &self.cpu,
            contents: self.contents,
            offset: self.offset,
            overs: &self.overs,
        }
    }
}

/// What one capture's command buffer copies: into the device copies and the images over them,
/// recorded into `command` of its own recorder, the spans of the ring they read, the staging
/// blocks of its own (an [`overflow`]) they read, and the images they copy. What takes it keeps it,
/// with each device copy and consumer's block, until it retires: the committing command buffer, or
/// the upload of a ring-full's submission of its own.
pub(crate) struct CopyIn {
    pub(crate) recorder: Recorder,
    pub(crate) command: vk::CommandBuffer,
    pub(crate) spans: Vec<u64>,
    pub(crate) modified: Vec<Modified>,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) staging: Vec<Overflowed>,
}

/// A staging block of an [`overflow`], counted in `Gauges::capture_bytes` while it lives; it goes
/// back to the pool as it drops.
pub(crate) struct Overflowed {
    block: Block,
    counters: Counters,
}

impl Drop for Overflowed {
    fn drop(&mut self) {
        self.counters
            .fall_by(Gauge::CaptureBytes, self.block.memory.len);
    }
}

impl CopyIn {
    /// Gives its recorder back and drops what it holds: once what took it retired, or where it
    /// never went.
    pub(crate) fn retire(self, core: &Core) {
        let CopyIn {
            recorder,
            modified,
            images,
            staging,
            ..
        } = self;
        core.recorders.give_back(core, recorder);
        drop((modified, images, staging));
    }
}

/// A copy-in being recorded: its copies are recorded together at its end, one command for each
/// run of copies from one source into one device copy.
pub(crate) struct Chunk {
    recorder: Recorder,
    command: vk::CommandBuffer,
    pub(crate) spans: Vec<u64>,
    /// Each copy's source buffer, the index of its buffer among the capture's, and its region.
    copies: Vec<(vk::Buffer, usize, vk::BufferCopy)>,
    images: Vec<Arc<Image>>,
    staging: Vec<Overflowed>,
}

impl Chunk {
    /// A copy-in's command buffer, recording, after the copies back `backs` (the first of a
    /// capture's), or a barrier that puts every earlier command first.
    ///
    /// Err: [`Error::Vulkan`] where a command pool or buffer cannot be made.
    pub(crate) fn begin(core: &Core, backs: Vec<Back>) -> Result<Chunk, Error> {
        let device = &core.device;
        let mut recorder = core.recorders.take(device)?;
        let command = match recorder.next(device) {
            Ok(command) => command,
            Err(error) => {
                core.recorders.give_back(core, recorder);
                return Err(error);
            }
        };
        let copies: Vec<OverCopy> = backs.iter().map(|(_, copy, _)| *copy).collect();
        // SAFETY: the command buffer is recording, and only this thread has it; each image is in
        // its home layout between commands, and the copy-in holds it.
        unsafe {
            match copies.is_empty() {
                true => {
                    record::barrier(device, command, Reach::Commands);
                    core.counters.count(Metric::Barriers);
                }
                false => over::record_side(device, command, &copies, false, &core.counters),
            }
        }
        Ok(Chunk {
            recorder,
            command,
            spans: Vec::new(),
            copies: Vec::new(),
            images: backs.into_iter().map(|(_, _, image)| image).collect(),
            staging: Vec::new(),
        })
    }

    /// A copy of `region` from `source` into the device copy of the capture's buffer `index`.
    pub(crate) fn copy(&mut self, source: vk::Buffer, index: usize, region: vk::BufferCopy) {
        self.copies.push((source, index, region));
    }

    /// Whether it copies from the ring.
    pub(crate) fn rings(&self) -> bool {
        !self.spans.is_empty()
    }

    /// Records its copies into the device copies of `named`, the capture's buffers, then the copies
    /// in `into` (the last of a capture's), or a barrier that makes the copies visible to every
    /// later command, and ends its command buffer, for [`Chunk::ended`].
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot be ended; the chunk is then given
    /// up.
    pub(crate) fn end(
        &self,
        core: &Core,
        named: &[Modified],
        into: &[(OverCopy, Arc<Image>)],
    ) -> Result<(), Error> {
        let device = &core.device;
        let (command, copies) = (self.command, &self.copies);
        let into_copies: Vec<OverCopy> = into.iter().map(|(copy, _)| *copy).collect();
        let mut at = 0;
        // SAFETY: the command buffer is recording, and only this thread has it; the ring's, the
        // staging blocks' and the device copies' buffers have transfer usage, each region lies in
        // them, what takes the copy-in keeps each block and device copy until it retires, and the
        // ring lives as long as the device; each image is in its home layout between commands.
        unsafe {
            while at < copies.len() {
                let (source, index, _) = copies[at];
                let run = copies[at..]
                    .iter()
                    .take_while(|&&(from, to, _)| from == source && to == index)
                    .count();
                let regions: Vec<vk::BufferCopy> = copies[at..at + run]
                    .iter()
                    .map(|&(_, _, region)| region)
                    .collect();
                device.cmd_copy_buffer(command, source, named[index].copy.buffer(), &regions);
                at += run;
            }
            match into_copies.is_empty() {
                true => {
                    record::barrier(device, command, Reach::Commands);
                    core.counters.count(Metric::Barriers);
                }
                false => over::record_side(device, command, &into_copies, true, &core.counters),
            }
        }
        // SAFETY: the command buffer is recording, and only this thread has it.
        unsafe { device.end_command_buffer(command) }.map_err(|result| Error::Vulkan {
            call: "vkEndCommandBuffer",
            result,
        })
    }

    /// The chunk, ended with `named` and `into` ([`Chunk::end`]): a copy-in holding `named` and
    /// the images of `into`.
    pub(crate) fn ended(self, named: Vec<Modified>, into: Vec<(OverCopy, Arc<Image>)>) -> CopyIn {
        let Chunk {
            recorder,
            command,
            spans,
            copies: _,
            mut images,
            staging,
        } = self;
        images.extend(into.into_iter().map(|(_, image)| image));
        CopyIn {
            recorder,
            command,
            spans,
            modified: named,
            images,
            staging,
        }
    }

    /// Gives up its recorder, for a capture that cannot finish: its spans come back.
    pub(crate) fn give_up(self, core: &Core) -> Vec<u64> {
        core.recorders.give_back(core, self.recorder);
        self.spans
    }
}

/// The copies back owed to the textures over `named`'s ranges in images of their own, taken: they
/// go ahead of a capture's copies, so the consumer's bytes land over them.
pub(crate) fn backs(named: &[Modified]) -> Vec<Back> {
    let mut backs = Vec::new();
    for modified in named.iter().filter(|modified| modified.overs.any()) {
        modified.overs.backs_before(&modified.ranges, &mut backs);
    }
    backs
}

/// The copies in the textures over `named`'s ranges in images of their own take, after a
/// capture's copies.
pub(crate) fn copies_in(named: &[Modified]) -> Vec<(OverCopy, Arc<Image>)> {
    let mut into = Vec::new();
    for modified in named.iter().filter(|modified| modified.overs.any()) {
        modified.overs.copies_in_after(&modified.ranges, &mut into);
    }
    into
}

/// The rest of a capture where work committed before it is held on the host: `pieces`, each a
/// buffer's index among `named` and a range `start..end` of it, copied on the host into staging
/// blocks of the Shared type of the capture's own, of at most [`OVERFLOW_BLOCK`] bytes each, taken
/// from the pool, and their copies into the device copies added to `chunk`, which holds the blocks
/// until what takes it retires, when they go back to the pool. Counted in
/// `Metrics::capture_overflows` and `capture_overflow_bytes`. Returns the bytes copied.
///
/// Err: what taking a block returns.
pub(crate) fn overflow(
    core: &Arc<Core>,
    named: &[Modified],
    pieces: &[(usize, u64, u64)],
    chunk: &mut Chunk,
) -> Result<u64, Error> {
    let total: u64 = pieces.iter().map(|&(_, start, end)| end - start).sum();
    let mut left = total;
    // The block being filled: its buffer, mapping and length, and how much of it is filled.
    let mut block: Option<(vk::Buffer, Mapping, u64, u64)> = None;
    for &(index, start, end) in pieces {
        let modified = &named[index];
        let mut at = start;
        while at < end {
            let (buffer, mapping, block_len, filled) = match block.take() {
                Some(open) if open.3 < open.2 => open,
                _ => {
                    let memory = core.take_memory(MemoryClass::Shared, left.min(OVERFLOW_BLOCK))?;
                    let Some(mapping) = memory.mapping else {
                        unreachable!("a Shared block is mapped");
                    };
                    let (buffer, block_len) = (memory.buffer, memory.len);
                    core.counters.rise_by(Gauge::CaptureBytes, block_len);
                    chunk.staging.push(Overflowed {
                        block: Block::new(memory, Arc::clone(core)),
                        counters: core.counters.clone(),
                    });
                    (buffer, mapping, block_len, 0)
                }
            };
            let len = (end - at).min(block_len - filled);
            let began = crate::host_time();
            // SAFETY: the block is the capture's alone, mapped, and holds `filled..filled + len`;
            // the consumer's buffer holds `at..at + len`; the two do not overlap.
            unsafe {
                modified
                    .contents
                    .at(at)
                    .as_ptr()
                    .copy_to_nonoverlapping(mapping.at(filled).as_ptr(), len as usize)
            };
            core.counters
                .capture_copied(crate::host_time().saturating_sub(began));
            chunk.copy(
                buffer,
                index,
                vk::BufferCopy {
                    src_offset: filled,
                    dst_offset: modified.copy.offset() + at,
                    size: len,
                },
            );
            at += len;
            left -= len;
            block = Some((buffer, mapping, block_len, filled + len));
        }
    }
    core.counters.count(Metric::CaptureOverflows);
    core.counters.overflowed(total);
    Ok(total)
}

/// Adds `start..end` to `ranges` (by start, no two overlapping or touching), joining the ranges it
/// overlaps or touches.
pub(crate) fn merge(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    // The first range that ends at or after `start`, and the first that starts after `end`.
    let first = ranges.partition_point(|&(_, stop)| stop < start);
    let last = ranges.partition_point(|&(begin, _)| begin <= end);
    if first == last {
        ranges.insert(first, (start, end));
        return;
    }
    let joined = (start.min(ranges[first].0), end.max(ranges[last - 1].1));
    ranges.drain(first + 1..last);
    ranges[first] = joined;
}
