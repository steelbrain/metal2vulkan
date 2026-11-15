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
//!   the form measured to read the write. A render encoder records no barrier inside its pass,
//!   so on such a device it joins no group whose earlier encoders wrote bytes
//!   (`State::join_fold`). Counted: `WorkCounts::command_buffers_split_at_aliases`.
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

/// The most writes an [`AliasWrites`] notes apart: past them a new key's write makes every image
/// use meet a write, so the next one splits and the list starts again.
const KEPT: usize = 32;

/// The bytes of each key the commands recorded into an encoder's current `VkCommandBuffer` wrote
/// (the module docs), on a device that splits at aliases: at most [`KEPT`], a key's writes
/// joined into the range that holds them all where it has one, and past that every key's.
#[derive(Default)]
pub(crate) struct AliasWrites {
    writes: Vec<(ResourceKey, Range<u64>)>,
    /// A write of more keys than [`KEPT`] was noted: every image use meets one.
    every: bool,
}

impl AliasWrites {
    /// What the earlier encoders of the group an encoder joined wrote into its `VkCommandBuffer`,
    /// the group's tail: their order's writes of bytes; nothing where it joined none.
    pub(crate) fn joining(order: Option<&FoldOrder<ResourceKey>>) -> AliasWrites {
        let mut writes = AliasWrites::default();
        for (key, bytes) in order.into_iter().flat_map(FoldOrder::byte_writes) {
            writes.wrote(key, bytes);
        }
        writes
    }

    /// Note a command recorded with `uses`: the bytes it writes.
    pub(crate) fn note(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses.iter().filter(|used| !used.write.is_empty()) {
            if let UseRange::Bytes(bytes) = &used.range {
                self.wrote(used.key, bytes.clone());
            }
        }
    }

    /// Note a write of `bytes` of `key`: joined into the range of the key's noted writes where
    /// it has one, else apart while fewer than [`KEPT`] are.
    fn wrote(&mut self, key: ResourceKey, bytes: Range<u64>) {
        if self.every {
            return;
        }
        if let Some((_, range)) = self.writes.iter_mut().find(|(noted, _)| *noted == key) {
            range.start = range.start.min(bytes.start);
            range.end = range.end.max(bytes.end);
        } else if self.writes.len() < KEPT {
            self.writes.push((key, bytes));
        } else {
            self.every = true;
        }
    }

    /// Whether a command with `uses` uses an image whose memory meets bytes a command before it
    /// in the `VkCommandBuffer` wrote: it goes in a new one (the module docs).
    pub(crate) fn meets(&self, uses: &[Use<ResourceKey>]) -> bool {
        uses.iter()
            .filter(|used| matches!(used.range, UseRange::Image { .. }))
            .any(|used| {
                self.every
                    || (self.writes.iter()).any(|(key, bytes)| {
                        *key == used.key
                            && used.range.shares_memory(&UseRange::Bytes(bytes.clone()))
                    })
            })
    }

    /// The encoder went on in a new `VkCommandBuffer`: no command wrote anything there yet.
    pub(crate) fn begun(&mut self) {
        self.writes.clear();
        self.every = false;
    }
}
