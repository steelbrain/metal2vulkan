//! A use of an image after a write through a buffer over its memory, in a new `VkCommandBuffer`,
//! on a driver whose image uses miss such writes inside one
//! (`DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer`).
//!
//! Vulkan makes a write through a buffer visible to a later use of a linear image in `GENERAL`
//! over the same memory once a memory dependency orders the two: both are host-accessible
//! aliases. The tracker orders them so (`tracker.rs`): one key names a heap's bytes and the
//! images bound in them, a buffer's bytes and the images over them, and a write of a range of
//! them orders every later use that meets it, inside an encoder and between the encoders of a
//! group that share a `VkCommandBuffer` (`encoder_fold.rs`). On NVIDIA's 620.70.06 on the RTX
//! 3090 a copy out of such an image, behind that global memory barrier in the same
//! `VkCommandBuffer` as the buffer's write, read every texel as it was before the write, where
//! the same two commands in two `VkCommandBuffer`s of one submission, the barrier between them,
//! read the write (the fact's measurement, `driver_facts.rs`).
//!
//! - **The rule.** On such a device an encoder notes the bytes each of its commands writes into
//!   the `VkCommandBuffer` it records into ([`AliasWrites`]); an encoder that joined a group of
//!   encoders starts from what the group's earlier encoders wrote there, its order's writes of
//!   bytes ([`FoldOrder::byte_writes`]). A command that uses an image of a key whose noted bytes
//!   meet the image's place ([`UseRange::shares_memory`]) goes in a new `VkCommandBuffer`: the
//!   encoder ends the one it records into and goes on in a new one, the command's barrier first
//!   (`EncoderCore::split_at_aliases`), which notes none. An encoder that joined a group goes on
//!   in a `VkCommandBuffer` of its own after the group's: its part's submission orders it after
//!   the group's first part, the tracker's barrier between them in a command buffer of barriers,
//!   the form measured to read the write. Counted: `WorkCounts::command_buffers_split_at_aliases`.
//! - **Render encoders.** A pass records no barrier, nor a new `VkCommandBuffer`, inside it, so
//!   the split goes before it. One whose pass records at its end joins a group as any encoder
//!   (`State::join_fold`). Its draws, recorded into its secondary, run after every command it
//!   records into the `VkCommandBuffer` before the pass, whichever came first: each draw's image
//!   uses are noted apart ([`AliasWrites::noted_apart`]), and where one meets a noted write, or a
//!   write noted later meets one ([`AliasWrites::note`]), the encoder owes a split. The first
//!   command it then records into the `VkCommandBuffer` (a shadow's copy before a later draw, or
//!   at its end the steps before its pass and the pass, whose uses hold its attachments') goes in
//!   a new one where it owes one, or where its own uses meet the writes: so every image the pass
//!   uses is checked against every write before it before the pass records. The images noted
//!   apart stay noted in the new one, whose later writes their draws still follow. One whose
//!   first instance records inline joins no group (`encoder_fold.rs`): its draws record from its
//!   first on, before its later calls are known.
//! - **Scope.** The measured direction alone: a use through an image after a write through a
//!   buffer. A use through a buffer after a write through an image is not measured and takes no
//!   split. A render pass's draws record inside its pass, where no `VkCommandBuffer` ends; what
//!   they write a later encoder's commands meet through the group's order, as any encoder's.
//! - **Elsewhere** nothing is noted and nothing splits: the device's encoders carry no
//!   [`AliasWrites`].

use crate::encoder_fold::FoldOrder;
use crate::resource::ResourceKey;
use crate::tracker::{Use, UseRange};
use metal2vulkan::device::DriverFact;
use std::collections::BTreeSet;
use std::ops::Range;

/// Whether a device whose driver has `facts` records a use of an image after a write through a
/// buffer over its memory in a new `VkCommandBuffer` (the module docs).
pub(crate) fn splits_at_aliases(facts: &BTreeSet<DriverFact>) -> bool {
    facts.contains(&DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer)
}

/// The most keys an [`AliasWrites`] notes the bytes of apart, in its writes and in its images
/// noted apart each: past them a new key makes every range meet, so the next image use, or the
/// next write, splits.
const KEPT: usize = 32;

/// Ranges of bytes of keys' memory: at most [`KEPT`] keys', a key's ranges joined into the range
/// that holds them all, and past that every key's.
#[derive(Default)]
struct Noted {
    ranges: Vec<(ResourceKey, Range<u64>)>,
    /// A range of more keys than [`KEPT`] was noted: every range meets one.
    every: bool,
}

impl Noted {
    /// Note `bytes` of `key`: joined into the key's noted range where it has one, else apart
    /// while fewer than [`KEPT`] keys are.
    fn add(&mut self, key: ResourceKey, bytes: Range<u64>) {
        if self.every {
            return;
        }
        if let Some((_, range)) = self.ranges.iter_mut().find(|(noted, _)| *noted == key) {
            range.start = range.start.min(bytes.start);
            range.end = range.end.max(bytes.end);
        } else if self.ranges.len() < KEPT {
            self.ranges.push((key, bytes));
        } else {
            self.every = true;
        }
    }

    /// Whether `bytes` of `key` meet a noted range.
    fn meets(&self, key: ResourceKey, bytes: &Range<u64>) -> bool {
        self.every
            || (self.ranges.iter()).any(|(noted, range)| {
                *noted == key && range.start < bytes.end && bytes.start < range.end
            })
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.every = false;
    }
}

/// What an encoder's commands on a device that splits at aliases noted (the module docs): the
/// bytes of each key the commands recorded into its current `VkCommandBuffer` wrote, and the
/// memory the images of its commands recorded apart use, which run after them (a pass's draws).
/// Each holds at most [`KEPT`] keys' ranges, and past that every key's.
#[derive(Default)]
pub(crate) struct AliasWrites {
    writes: Noted,
    /// The bytes the images of the commands recorded outside the `VkCommandBuffer`, which a
    /// later command there executes, use ([`noted_apart`](Self::noted_apart)): kept from one
    /// `VkCommandBuffer` to the next, as those commands run after what either records before them.
    apart: Noted,
    /// A command noted apart uses an image whose memory meets a noted write, or a write noted
    /// since meets an image one uses: the next command recorded into the `VkCommandBuffer` goes
    /// in a new one.
    owed: bool,
}

impl AliasWrites {
    /// What the earlier encoders of the group an encoder joined wrote into its `VkCommandBuffer`,
    /// the group's tail: their order's writes of bytes; nothing where it joined none.
    pub(crate) fn joining(order: Option<&FoldOrder<ResourceKey>>) -> AliasWrites {
        let mut writes = AliasWrites::default();
        for (key, bytes) in order.into_iter().flat_map(FoldOrder::byte_writes) {
            writes.writes.add(key, bytes);
        }
        writes
    }

    /// Note a command recorded into the `VkCommandBuffer` with `uses`: the bytes it writes, and
    /// whether one meets the memory of an image a command noted apart uses, which runs after it
    /// ([`noted_apart`](Self::noted_apart)): the next command recorded there goes in a new one.
    pub(crate) fn note(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses.iter().filter(|used| !used.write.is_empty()) {
            if let UseRange::Bytes(bytes) = &used.range {
                self.owed |= self.apart.meets(used.key, bytes);
                self.writes.add(used.key, bytes.clone());
            }
        }
    }

    /// Note a command recorded outside the `VkCommandBuffer` with `uses`, which a later command
    /// there executes (a draw of a pass recorded at its end, into its secondary): the memory its
    /// images use, and whether it meets a noted write. Where it does, now or once a later write
    /// meets it ([`note`](Self::note)), the next command recorded there goes in a new one
    /// ([`meets`](Self::meets)).
    pub(crate) fn noted_apart(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses {
            if let Some(bytes) = used.range.image_bytes() {
                self.owed |= self.writes.meets(used.key, &bytes);
                self.apart.add(used.key, bytes);
            }
        }
    }

    /// Whether a command with `uses` uses an image whose memory meets bytes a command before it
    /// in the `VkCommandBuffer` wrote, or follows a command noted apart whose images do
    /// ([`noted_apart`](Self::noted_apart)): it goes in a new one (the module docs).
    pub(crate) fn meets(&self, uses: &[Use<ResourceKey>]) -> bool {
        self.owed
            || (uses.iter()).any(|used| {
                (used.range.image_bytes()).is_some_and(|bytes| self.writes.meets(used.key, &bytes))
            })
    }

    /// The encoder went on in a new `VkCommandBuffer`: no command wrote anything there yet, and
    /// nothing is owed. The images noted apart stay: their commands run after what the new one
    /// records before them.
    pub(crate) fn begun(&mut self) {
        self.writes.clear();
        self.owed = false;
    }
}
