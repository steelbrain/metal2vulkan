//! A Managed buffer's changed ranges, copied in at the commit: where a Managed buffer has a device
//! copy (a device whose Vulkan device type is neither integrated nor CPU), `didModifyRange:` only
//! names a range ([`Changes::add`]), and the next commit on any queue of a command buffer with
//! commands takes every range named since the last such commit, once each
//! ([`Work::capture`](crate::submit::Work::capture)). Its copy-in ([`copy_in`]) is a command
//! buffer of its own, [`CopyIn`]: GPU copies of each range from the consumer's block into the
//! device copy, after the copies back owed to the textures over the ranges in images of their own
//! and before the copies in those textures take (`crate::over::ManagedOvers`). It waits in
//! [`Captured`] at the capturing commit's number, a point in the commit order, and the first
//! command buffer submitted whose commit number is at or after it takes it ([`taker`]): it goes in
//! that command buffer's submission, as a part of its own ahead of its commands and outside its
//! waits. A command buffer committed before the capture leaves it pending, so it never goes ahead
//! of older work of the queue that may write the range. The submission keeps each consumer's
//! block and device copy until it retires. A copy-in records no barrier before its copies: the one
//! that ends every earlier segment, upload, fill and copy-in on the queue puts every earlier
//! command first, as it does for a segment (`record::end`), and the submission that carries it
//! makes the host's writes before it visible to the copies (Vulkan's host write ordering
//! guarantee).
//!
//! So every command buffer committed after a range is named reads the bytes the consumer wrote
//! there before the commit, on any queue and wherever the capturing command buffer waits, which is
//! what Metal owes (`MTLBuffer.h`, `didModifyRange:`): a write makes the range "undefined for any
//! accesses by the GPU" until a `didModifyRange:` "followed by a commit of the MTLCommandBuffer
//! that will access the buffer", which "does not make the contents coherent for any previously
//! committed command buffers". One committed before that runs after a later one, held at an event
//! wait, may read them too, where the header leaves the contents undefined. The copy reads the
//! consumer's bytes as it runs, so a write after the commit may reach it, where Metal leaves the
//! range undefined.
//!
//! A range named and not yet captured holds its buffer weakly, and neither of its blocks: once the
//! buffer's last handle is gone, nothing can read its device copy (a command buffer, an encoder's
//! binding or a texture that could holds the buffer), so the capture forgets its ranges, and the
//! buffer's drop lets its blocks go at once, an import's release among them. Metal runs a no-copy
//! buffer's deallocator as it deallocates the buffer, and its `didModifyRange:` is owed only to a
//! command buffer that accesses the buffer, which then holds it.

use crate::buffer::{BufferInner, DeviceCopy};
use crate::counters::Metric;
use crate::device::Core;
use crate::error::Error;
use crate::memory::Block;
use crate::over::{self, Back, ManagedOvers};
use crate::record::{self, Reach, Recorder};
use crate::texture::{Image, OverCopy};
use crate::times;
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Weak};

/// A Managed buffer's ranges a capture took ([`Changes::take`]), and what its copy-in holds.
#[derive(Clone)]
pub(crate) struct Modified {
    /// The buffer, to name the ranges again where the copy-in never runs
    /// ([`Changes::give_up`]); held weakly, as a range named is.
    pub(crate) buffer: Weak<BufferInner>,
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

/// A Managed buffer's ranges named and not yet captured, `start..end` in the buffer, by start, no
/// two overlapping or touching. The buffer is held weakly (the module's doc).
struct Named {
    buffer: Weak<BufferInner>,
    ranges: Vec<(u64, u64)>,
}

/// The device's ranges named and not yet captured, in the order of each buffer's first.
#[derive(Default)]
pub(crate) struct Changes {
    named: Vec<Named>,
    /// Each buffer's place in `named`, by its address, which a weak hold keeps from reuse.
    at: HashMap<usize, usize>,
    /// The copy-ins captured and not yet taken.
    pub(crate) captured: Captured,
}

impl Changes {
    /// Names `start..end` of `buffer`, a Managed buffer with a device copy.
    pub(crate) fn add(&mut self, buffer: &Arc<BufferInner>, start: u64, end: u64) {
        let key = Arc::as_ptr(buffer) as usize;
        match self.at.get(&key) {
            Some(&index) => merge(&mut self.named[index].ranges, start, end),
            None => {
                self.at.insert(key, self.named.len());
                self.named.push(Named {
                    buffer: Arc::downgrade(buffer),
                    ranges: vec![(start, end)],
                });
            }
        }
    }

    /// The ranges named since the last capture, taken: those of the buffers that still live, each
    /// holding the buffer's blocks from here; a buffer gone has its ranges forgotten.
    pub(crate) fn take(&mut self) -> Vec<Modified> {
        self.at.clear();
        (self.named.drain(..))
            .filter_map(|named| {
                let buffer = named.buffer.upgrade()?;
                buffer.modified(named.ranges)
            })
            .collect()
    }

    /// A capture of `modified` that will not run: its ranges are named again, for the next commit
    /// to capture, where the buffer still lives.
    pub(crate) fn give_up(&mut self, modified: &[Modified]) {
        for modified in modified {
            let Some(buffer) = modified.buffer.upgrade() else {
                continue;
            };
            for &(start, end) in &modified.ranges {
                self.add(&buffer, start, end);
            }
        }
    }

    /// At the device's end: the ranges named are forgotten, and the copy-ins not taken come back,
    /// for their recorders to be given back.
    pub(crate) fn clear(&mut self) -> Vec<CopyIn> {
        self.at.clear();
        self.named.clear();
        let captured = self.captured.0.drain(..).map(|(_, copy_in)| copy_in);
        captured.collect()
    }
}

/// The copy-ins captured and not yet taken (a [`CopyIn`] each but in the tests), each with the
/// commit number of the command buffer whose commit captured it, in commit order.
pub(crate) struct Captured<C = CopyIn>(VecDeque<(u64, C)>);

impl<C> Default for Captured<C> {
    fn default() -> Captured<C> {
        Captured(VecDeque::new())
    }
}

impl<C> Captured<C> {
    /// `copy_in`, captured by the commit numbered `number`.
    pub(crate) fn push(&mut self, number: u64, copy_in: C) {
        self.0.push_back((number, copy_in));
    }

    /// The copy-ins a submission takes, in which command buffers numbered `leads` (in the order
    /// they go in it) go first: each with the index in `leads` of the one that takes it
    /// ([`taker`]). The rest stay.
    pub(crate) fn take(&mut self, leads: &[u64]) -> Vec<(usize, C)> {
        let mut taken = Vec::new();
        while let Some(&(number, _)) = self.0.front() {
            let Some(index) = taker(number, leads) else {
                break;
            };
            if let Some((_, copy_in)) = self.0.pop_front() {
                taken.push((index, copy_in));
            }
        }
        taken
    }
}

/// Which command buffer of a submission takes a copy-in captured by the commit numbered `captured`,
/// of those that go first in it numbered `leads`, in the order they go: the first numbered at or
/// after the capture. None where each was committed before it: the copy-in stays for a later one.
pub(crate) fn taker(captured: u64, leads: &[u64]) -> Option<usize> {
    leads.iter().position(|&lead| lead >= captured)
}

impl Modified {
    /// Its bytes named, in all.
    pub(crate) fn bytes(&self) -> u64 {
        self.ranges.iter().map(|&(start, end)| end - start).sum()
    }
}

/// What one capture's command buffer copies: from the consumers' blocks into the device copies
/// and the images over them, recorded into `command` of its own recorder, and the images it
/// copies. The submission that takes it keeps it, with each consumer's block and device copy, until
/// it retires.
pub(crate) struct CopyIn {
    pub(crate) recorder: Recorder,
    pub(crate) command: vk::CommandBuffer,
    pub(crate) modified: Vec<Modified>,
    pub(crate) images: Vec<Arc<Image>>,
    /// Its pair of timestamp queries, its start written first and its end last
    /// ([`times::begin_own`]); None where it runs untimed.
    pub(crate) stamps: Option<times::Pair>,
}

impl CopyIn {
    /// Gives its recorder back and drops what it holds, and its pair of timestamp queries, read
    /// where its submission completed (`end`, [`times::retire_own`]): once what took it retired,
    /// or ended at a loss, or where it never went.
    pub(crate) fn retire(self, core: &Core, end: times::OwnEnd) {
        let CopyIn {
            recorder,
            modified,
            images,
            stamps,
            ..
        } = self;
        times::retire_own(core, stamps, end);
        core.recorders.give_back(core, recorder);
        drop((modified, images));
    }
}

/// The copy-in of `named`, the ranges a capture takes: a command buffer of its own, recorded and
/// ended, with the copies back owed to the textures over the ranges in images of their own
/// first, then one `vkCmdCopyBuffer` for each buffer from the consumer's block into its device
/// copy, then the copies in those textures take, or a barrier that makes the copies visible to
/// every later command. It begins with no barrier of its own: the one ending each earlier command
/// buffer on the queue puts every earlier command first (`record::end`). Its timestamps are its
/// first command and its last ([`times::begin_own`]).
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
    let stamps = times::begin_own(core, command);
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
    times::end_own(device, command, stamps);
    // SAFETY: the command buffer is recording, and only this thread has it.
    if let Err(result) = unsafe { device.end_command_buffer(command) } {
        times::retire_own(core, stamps, times::OwnEnd::Unsent);
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
        stamps,
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
