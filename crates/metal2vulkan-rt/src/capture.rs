//! A Managed buffer's changed ranges, captured at the commit: where a Managed buffer has a device
//! copy (a device whose Vulkan device type is neither integrated nor CPU), `didModifyRange:` only
//! names a range ([`Changes::add`]), and the next commit on any queue of a command buffer with
//! commands copies the consumer's bytes of every range named since the last such commit into the
//! device's capture ring, on the committing thread, once each, before the call returns
//! ([`Work::capture`](crate::submit::Work::capture)). Commits take their turns one at a time, so
//! the command buffers keep the order of their captures. The copies from the ring into the device
//! copies, and then into the images of the textures over the ranges in images of their own
//! (`crate::over::ManagedOvers`), are recorded into a command buffer of their own, a [`CopyIn`],
//! which goes ahead of the command buffer's own commands in its submission, after a barrier that
//! puts every earlier command on the queue first. So a command buffer reads the bytes as they were
//! at its commit, and a write after the commit reaches none committed before it. Metal takes them
//! shortly after the commit, before the command buffer is scheduled, and a write or a
//! `didModifyRange:` made while the command buffer runs does not reach it.
//!
//! The ring is one host-visible allocation of [`CAPTURE_LEN`] bytes of the Shared type, made at the
//! first capture and kept for the device's life. A span is free again once the submission that
//! carries its command buffer has completed. A capture that finds no room waits for the oldest
//! span's submission; where the oldest span belongs to a command buffer not yet submitted (this
//! one, or one held), the queues' held groups go at once, and the rest of the capture is copied
//! from the consumer's bytes when the copy runs, as `Metrics::managed_bytes_live` counts: the ring
//! is never grown, the commit never waits on a command buffer that may wait on the host, and no
//! range is dropped.

use crate::buffer::DeviceCopy;
use crate::memory::Block;
use crate::over::ManagedOvers;
use crate::record::Recorder;
use crate::staging::{At, Region};
use crate::texture::Image;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// The capture ring's bytes.
pub(crate) const CAPTURE_LEN: u64 = 64 << 20;

/// A Managed buffer's ranges named since the last commit.
pub(crate) struct Modified {
    pub(crate) copy: Arc<DeviceCopy>,
    /// The block of the consumer's bytes, and the buffer's offset in it.
    pub(crate) cpu: Arc<Block>,
    pub(crate) offset: u64,
    /// Its textures over the device copy in images of their own.
    pub(crate) overs: Arc<ManagedOvers>,
    /// `start..end` in the buffer, by start, no two overlapping or touching.
    pub(crate) ranges: Vec<(u64, u64)>,
}

/// A Managed buffer whose range is named: its device copy, its consumer's bytes (the block, and
/// the buffer's offset in it) and its textures over the device copy in images of their own.
pub(crate) struct Named<'a> {
    pub(crate) copy: &'a Arc<DeviceCopy>,
    pub(crate) cpu: &'a Arc<Block>,
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
            offset: self.offset,
            overs: &self.overs,
        }
    }
}

/// What one commit captured: the copies into the device copies and the images over them, recorded
/// into `command` of its own recorder, and the spans of the ring they read. The command buffer
/// that takes it keeps it, with each device copy, consumer's block and image, until it retires.
pub(crate) struct CopyIn {
    pub(crate) recorder: Recorder,
    pub(crate) command: vk::CommandBuffer,
    pub(crate) spans: Vec<u64>,
    pub(crate) modified: Vec<Modified>,
    pub(crate) images: Vec<Arc<Image>>,
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
