//! The second copy of a Managed buffer of a stream recorded with two copies (Metal's
//! `hasUnifiedMemory` false there), played on a device whose Managed buffers are one copy
//! ([`rt::Device::has_unified_memory`]). Metal keeps two copies there: the CPU's bytes, which
//! `contents` answers and the stream's writes change, and the GPU's, which shaders, blits and
//! textures reach. A `didModifyRange:` makes the GPU's copy of the range equal the CPU's at the
//! next commit, over whatever the GPU wrote there; `synchronizeResource:` copies the GPU's copy
//! into the CPU's. Played as one copy, a kernel's write into the buffer would stay where the
//! recording's next `didModifyRange:` put the CPU's bytes back.
//!
//! So the backend keeps both: the buffer object is the GPU's copy, a Shared buffer in a place of
//! the backend's own, and this keeps the CPU's copy, a second such buffer, and the ranges named
//! since the last commit. The next commit, on any queue, copies them into the GPU's copy in a
//! command buffer of the backend's own on the committing command buffer's queue, committed just
//! before it.

use super::places::Place;
use metal2vulkan_runtime as rt;
use std::collections::HashMap;

/// One buffer's CPU copy, its place, its GPU copy, and the ranges of it named since the last
/// commit, in order, merged where they overlap or touch.
struct Second {
    cpu: rt::Buffer,
    place: Place,
    gpu: rt::Buffer,
    named: Vec<(u64, u64)>,
}

/// The CPU copies of the Managed buffers played with a second copy, and what they cost.
#[derive(Default)]
pub(super) struct Seconds {
    buffers: HashMap<u64, Second>,
    /// Released buffers whose named ranges the next commit still copies: a command buffer that
    /// named the buffer before its release may be committed after it.
    leaving: Vec<Second>,
    /// The buffers made with a second copy, and their bytes.
    made: u64,
    made_bytes: u64,
    /// The ranges copied into the GPU's copies at commits, and their bytes.
    copies: u64,
    copied_bytes: u64,
    /// The named ranges widened to whole 4-byte words for the copy (a blit copies whole words).
    widened: u64,
    /// The bytes of named ranges past a buffer's last whole word, which no blit reaches.
    uncopied_bytes: u64,
    /// The Managed buffers a two-copy stream places in a block of its own, played as one copy:
    /// the stream's addresses in the block are the buffer's, so it has no second copy.
    pub(super) in_block: u64,
}

/// What a commit copies: each buffer's CPU copy, its GPU copy, and its ranges, in whole words.
pub(super) struct Named {
    pub(super) cpu: rt::Buffer,
    pub(super) gpu: rt::Buffer,
    pub(super) ranges: Vec<(u64, u64)>,
}

impl Seconds {
    /// Object `object`'s CPU copy `cpu`, in `place`, of its GPU copy `gpu`.
    pub(super) fn insert(&mut self, object: u64, cpu: rt::Buffer, place: Place, gpu: rt::Buffer) {
        self.made += 1;
        self.made_bytes += cpu.length();
        let named = Vec::new();
        let second = Second {
            cpu,
            place,
            gpu,
            named,
        };
        self.buffers.insert(object, second);
    }

    /// Object `object`'s CPU copy, where it has one.
    pub(super) fn cpu(&self, object: u64) -> Option<&rt::Buffer> {
        self.buffers.get(&object).map(|second| &second.cpu)
    }

    /// `start..end` of object `object` named for the next commit: false where it has no second
    /// copy.
    pub(super) fn name(&mut self, object: u64, start: u64, end: u64) -> bool {
        let Some(second) = self.buffers.get_mut(&object) else {
            return false;
        };
        if start < end {
            merge(&mut second.named, start, end);
        }
        true
    }

    /// Whether a commit has ranges to copy.
    pub(super) fn named(&self) -> bool {
        let mut all = self.buffers.values().chain(&self.leaving);
        all.any(|second| !second.named.is_empty())
    }

    /// The ranges named since the last commit, taken, each widened to the whole 4-byte words
    /// inside its buffer, counted; and the places of the released buffers whose last ranges they
    /// were, for the commit to give back.
    pub(super) fn take(&mut self) -> (Vec<Named>, Vec<Place>) {
        let mut out = Vec::new();
        for second in self.buffers.values_mut().chain(&mut self.leaving) {
            if second.named.is_empty() {
                continue;
            }
            let words_end = second.cpu.length() / 4 * 4;
            let mut ranges = Vec::new();
            for (start, end) in second.named.drain(..) {
                let (from, to) = (start / 4 * 4, end.next_multiple_of(4).min(words_end));
                if (from, to) != (start, end) {
                    self.widened += 1;
                }
                self.uncopied_bytes += end.saturating_sub(words_end.max(start));
                if from < to {
                    self.copies += 1;
                    self.copied_bytes += to - from;
                    ranges.push((from, to));
                }
            }
            if !ranges.is_empty() {
                let (cpu, gpu) = (second.cpu.clone(), second.gpu.clone());
                out.push(Named { cpu, gpu, ranges });
            }
        }
        let places = self.leaving.drain(..).map(|second| second.place).collect();
        (out, places)
    }

    /// Object `object` released: its CPU copy's place, where it had one and no range of it waits
    /// for the next commit (which gives the place back once it has copied, `take`).
    pub(super) fn remove(&mut self, object: u64) -> Option<Place> {
        let second = self.buffers.remove(&object)?;
        if second.named.is_empty() {
            return Some(second.place);
        }
        self.leaving.push(second);
        None
    }

    /// The report's line: so that a play with second copies is never read as a one-copy guest's.
    pub(super) fn line(&self) -> String {
        format!(
            "managed second copies {}: Managed buffers of a two-copy stream played with a second \
             copy of the player's own on this one-copy device ({} bytes each copy); {} ranges \
             ({} bytes) copied into the GPU's copies at commits; {} ranges widened to whole \
             words; {} bytes past a buffer's last whole word not copied; {} Managed buffers in \
             the stream's blocks played as one copy",
            self.made,
            self.made_bytes,
            self.copies,
            self.copied_bytes,
            self.widened,
            self.uncopied_bytes,
            self.in_block
        )
    }
}

/// `start..end` into `ranges`, kept in order and merged where they overlap or touch.
pub(in crate::replay) fn merge(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    let at = ranges.partition_point(|&(_, e)| e < start);
    let mut merged = (start, end);
    while at < ranges.len() && ranges[at].0 <= merged.1 {
        let (s, e) = ranges.remove(at);
        merged = (merged.0.min(s), merged.1.max(e));
    }
    ranges.insert(at, merged);
}
