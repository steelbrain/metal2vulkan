//! A Managed buffer's changed ranges, copied in at the commit: where a Managed buffer has a device
//! copy (a device whose Vulkan device type is neither integrated nor CPU), `didModifyRange:` only
//! names a range ([`Changes::add`]), and the next commit on any queue of a command buffer with
//! commands takes every range named since the last such commit, once each
//! ([`Work::capture`](crate::submit::Work::capture)). Its copy-in ([`copy_in`]) is a command
//! buffer of its own, [`CopyIn`], that goes ahead of the command buffer's own commands in queue
//! order: GPU copies of each range from the consumer's block into the device copy, after the
//! copies back owed to the textures over the ranges in images of their own and before the copies
//! in those textures take (`crate::over::ManagedOvers`). It holds each consumer's block and device
//! copy until what takes it retires. A copy-in records no barrier before its copies: the one that
//! ends every earlier segment, upload, fill and copy-in on the queue puts every earlier command
//! first, as it does for a segment (`record::end`), and the submission that carries it makes the
//! host's writes before it visible to the copies (Vulkan's host write ordering guarantee).
//!
//! So a command buffer committed after a range is named reads the bytes the consumer wrote there
//! before the commit, which is what Metal owes (`MTLBuffer.h`, `didModifyRange:`): a write makes
//! the range undefined for the GPU until a `didModifyRange:` and a commit of the command buffer
//! that reads it, and does not reach a command buffer committed before. The copy reads the
//! consumer's bytes as it runs, so a write after the commit may reach it, where Metal leaves the
//! range undefined.

use crate::buffer::DeviceCopy;
use crate::counters::Metric;
use crate::device::Core;
use crate::error::Error;
use crate::memory::Block;
use crate::over::{self, Back, ManagedOvers};
use crate::record::{self, Reach, Recorder};
use crate::texture::{Image, OverCopy};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// A Managed buffer's ranges named since the last capture.
#[derive(Clone)]
pub(crate) struct Modified {
    pub(crate) copy: Arc<DeviceCopy>,
    /// The block of the consumer's bytes, held.
    pub(crate) cpu: Arc<Block>,
    /// The buffer's offset in `cpu`.
    pub(crate) offset: u64,
    /// Its textures over the device copy in images of their own.
    pub(crate) overs: Arc<ManagedOvers>,
    /// `start..end` in the buffer, by start, no two overlapping or touching.
    pub(crate) ranges: Vec<(u64, u64)>,
}

/// A Managed buffer whose range is named: its device copy, its consumer's bytes (the block and
/// the buffer's offset in it) and its textures over the device copy in images of their own.
pub(crate) struct Named<'a> {
    pub(crate) copy: &'a Arc<DeviceCopy>,
    pub(crate) cpu: &'a Arc<Block>,
    pub(crate) offset: u64,
    pub(crate) overs: &'a Arc<ManagedOvers>,
}

/// The device's ranges named and not yet captured, in the order of each buffer's first.
#[derive(Default)]
pub(crate) struct Changes {
    modified: Vec<Modified>,
    /// Each buffer's place in `modified`, by its device copy's address.
    at: HashMap<usize, usize>,
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

    /// A capture of `modified` that will not run: its ranges are named again, for the next commit
    /// to capture.
    pub(crate) fn give_up(&mut self, modified: &[Modified]) {
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
            offset: self.offset,
            overs: &self.overs,
        }
    }

    /// Its bytes named, in all.
    pub(crate) fn bytes(&self) -> u64 {
        self.ranges.iter().map(|&(start, end)| end - start).sum()
    }
}

/// What one capture's command buffer copies: from the consumers' blocks into the device copies
/// and the images over them, recorded into `command` of its own recorder, and the images it
/// copies. What takes it keeps it, with each consumer's block and device copy, until it retires:
/// the committing command buffer.
pub(crate) struct CopyIn {
    pub(crate) recorder: Recorder,
    pub(crate) command: vk::CommandBuffer,
    pub(crate) modified: Vec<Modified>,
    pub(crate) images: Vec<Arc<Image>>,
}

impl CopyIn {
    /// Gives its recorder back and drops what it holds: once what took it retired, or where it
    /// never went.
    pub(crate) fn retire(self, core: &Core) {
        let CopyIn {
            recorder,
            modified,
            images,
            ..
        } = self;
        core.recorders.give_back(core, recorder);
        drop((modified, images));
    }
}

/// The copy-in of `named`, the ranges a capture takes: a command buffer of its own, recorded and
/// ended, with the copies back owed to the textures over the ranges in images of their own
/// first, then one `vkCmdCopyBuffer` for each buffer from the consumer's block into its device
/// copy, then the copies in those textures take, or a barrier that makes the copies visible to
/// every later command. It begins with no barrier of its own: the one ending each earlier command
/// buffer on the queue puts every earlier command first (`record::end`).
///
/// Err: [`Error::Vulkan`] where a command pool or buffer cannot be made or ended.
pub(crate) fn copy_in(core: &Core, named: &[Modified]) -> Result<CopyIn, Error> {
    let device = &core.device;
    let mut recorder = core.recorders.take(device)?;
    let command = match recorder.next(device) {
        Ok(command) => command,
        Err(error) => {
            core.recorders.give_back(core, recorder);
            return Err(error);
        }
    };
    let backs = backs(named);
    let into = copies_in(named);
    let back_copies: Vec<OverCopy> = backs.iter().map(|(_, copy, _)| *copy).collect();
    let into_copies: Vec<OverCopy> = into.iter().map(|(copy, _)| *copy).collect();
    // SAFETY: the command buffer is recording, and only this thread has it; the consumers'
    // blocks' and the device copies' buffers have transfer usage, each region lies in them, and
    // what takes the copy-in keeps each block and device copy until it retires; each image is in
    // its home layout between commands, and the copy-in holds it.
    unsafe {
        if !back_copies.is_empty() {
            over::record_side(device, command, &back_copies, false, &core.counters);
        }
        for modified in named {
            let (from, base) = modified.cpu.memory.range(modified.offset);
            let (to, _) = modified.copy.range(0);
            let regions: Vec<vk::BufferCopy> = (modified.ranges.iter())
                .map(|&(start, end)| vk::BufferCopy {
                    src_offset: base + start,
                    dst_offset: modified.copy.range(start).1,
                    size: end - start,
                })
                .collect();
            device.cmd_copy_buffer(command, from, to, &regions);
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
    if let Err(result) = unsafe { device.end_command_buffer(command) } {
        core.recorders.give_back(core, recorder);
        return Err(Error::Vulkan {
            call: "vkEndCommandBuffer",
            result,
        });
    }
    let mut images: Vec<Arc<Image>> = backs.into_iter().map(|(_, _, image)| image).collect();
    images.extend(into.into_iter().map(|(_, image)| image));
    Ok(CopyIn {
        recorder,
        command,
        modified: named.to_vec(),
        images,
    })
}

/// The copies back owed to the textures over `named`'s ranges in images of their own, taken: they
/// go ahead of a capture's copies, so the consumer's bytes land over them.
fn backs(named: &[Modified]) -> Vec<Back> {
    let mut backs = Vec::new();
    for modified in named.iter().filter(|modified| modified.overs.any()) {
        modified.overs.backs_before(&modified.ranges, &mut backs);
    }
    backs
}

/// The copies in the textures over `named`'s ranges in images of their own take, after a
/// capture's copies.
fn copies_in(named: &[Modified]) -> Vec<(OverCopy, Arc<Image>)> {
    let mut into = Vec::new();
    for modified in named.iter().filter(|modified| modified.overs.any()) {
        modified.overs.copies_in_after(&modified.ranges, &mut into);
    }
    into
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
