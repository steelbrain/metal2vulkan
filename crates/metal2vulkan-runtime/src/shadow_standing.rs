//! A shadow's standing across command buffers: where only the runtime's commands write a
//! shadowed texture's truth (`ShadowCopies::only_commands_write`: a tracked `Private` buffer's
//! bytes the runtime made, an IOSurface plane's among them, or the texels of a tracked `Private`
//! image the runtime made, a usage shadow's or a view's truth), an image that a command buffer's
//! end left holding every texel of the truth still holds it at a later command buffer's first use
//! unless a command submitted since wrote the truth or the image. That first copy in then moves
//! nothing (`WorkCounts::shadow_copies_in_skipped_by_standing`), or, where the end left the image
//! stale inside a rectangle alone, the texels inside it alone.
//!
//! - **The order.** Every decision and every write it counts is taken as a batch is submitted,
//!   in the scheduler's order over all queues (`command_buffer/submit.rs`), which is the order
//!   the tracker orders tracked work in: a write submitted before a copy in that reads what it
//!   wrote runs before it, and one submitted after it runs after it. So the copy in reads the
//!   truth as every write before it in that order left it, as Metal's does.
//! - **What a command buffer leaves.** Its last batch, where it holds the command buffer's last
//!   encoder and nothing failed it, records each shadow its end left holding every texel of the
//!   truth (`ShadowCarry::standing_at_end`), as standing at what the writes counted so far leave:
//!   the batch's own writes counted first, its copies back among them. A shadow whose truth
//!   writes within a rectangle changed since its copy in, by the command buffer's own commands (a
//!   render pass's draws over a color attachment it loads, which the shadow's copy in preceded,
//!   or another shadow's copy back of what such a pass wrote: `shadow::Within`), stands stale
//!   inside the rectangle bounding them alone: those writes are among the counted ones, and every
//!   other texel of the image holds the truth.
//! - **What lets go.** A batch's tracked write of the truth's buffer or image or of the shadow's
//!   image (its key, whatever the range) lets go of the standing: another shadow's copy back over the same
//!   bytes, a blit's or a pass's write, a copy in that a command buffer's end did not leave
//!   standing. So does another API's write of an exported buffer's bytes, which the caller says
//!   it made (`Buffer::external_write_done`, `exported_memory.rs`), counted as that buffer's
//!   write as the call returns: the API's writes are the device's to read only from then on. So does any batch of a command buffer whose commands may reach memory no use names
//!   (a root table, an execution resolved from records: `State::reaches_unnamed`), for every
//!   shadow. Every other command's tracked uses are in its encoder's summary, whether or not the
//!   device's logs of host pages collected them (`CommandBuffer::note_dirty`). A shadow over a
//!   buffer whose uses are untracked, whose writes no summary holds, never stands; nor does one
//!   over a buffer whose `VkBuffer` was handed out (`Buffer::vk_buffer`), or of an image whose
//!   `VkImage` was (`Texture::vk_image`), which the caller's own work may write.
//! - **The copy in.** A command buffer's first copy in of such a shadow takes every texel, even
//!   for a command that reads inside a rectangle (a pass's scissored draws), so the image holds
//!   them all after it for the command buffer's later commands and for its end; but for a command
//!   that writes the truth apart from the shadow (a pass whose color attachment renders the bytes
//!   in place, read through the shadow by a draw of the pass, `texture/attached_reads.rs`), after
//!   which the image holds the truth whole only where a later copy takes again what it wrote: that
//!   copy moves what the command needs, as any other does. Made before any
//!   command of its encoder wrote what it reads, it goes to its batch's submission
//!   (`shadow::GatedCopyIn`), and copies nothing there where the shadow stands and no earlier
//!   encoder of the command buffer wrote the truth or the image, and the command buffer reaches
//!   nothing unnamed; where it stands stale inside a rectangle, the texels the copy would move
//!   inside it alone. Every other copy is as the module `shadow.rs` states.
//! - **Why a copy moved its texels.** Each command buffer's first copy in of a shadow whose truth
//!   only the runtime's commands write that moved its texels is counted by where it was decided
//!   ([`Unskipped`]) and, for a shadow that may stand, by what its standing held then
//!   ([`LetGo`], or none where it stood): `WorkCounts::shadow_texels_unskipped_*`. A copy
//!   decided as its encoder recorded it reads the standing by the batches submitted so far.
//! - **Why it never stood.** A shadow that may stand but never stood keeps what last kept a
//!   command buffer's end from leaving it standing ([`Missed`]): an encoder that ended with its
//!   image not holding the truth, a drop from what the command buffer carries to its later
//!   encoders, an end that held it in part, or an end that records no standing. Its
//!   unskipped copies count by it (`WorkCounts::shadow_texels_never_stood_*`).

use super::buffer::Buffer;
use super::key_hash::KeyMap;
use super::resource::ResourceKey;
use super::shadow::{ImageTruth, ShadowCopies};
use super::tracker::Summary;
use super::values::{HazardTrackingMode, StorageMode};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// Where a command buffer's first copy in of a shadow whose truth only the runtime's commands
/// write was decided, of one that moved its texels (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unskipped {
    /// The shadow never stands: its buffer is untracked, a heap's or over the caller's memory, its
    /// `VkBuffer` (or its image's `VkImage`) was handed out, or Metal drops its writes.
    Unstanding,
    /// Recorded between two render pass instances of a pass its encoder holds
    /// (`render/inline_splits.rs`), which no batch's submission decides.
    BetweenInstances,
    /// A command of its encoder wrote what it reads before it.
    AfterEncoderWrites,
    /// Recorded whole as its encoder recorded it: inside a predicated loop's laying, or where no
    /// new `VkCommandBuffer` could be made.
    RecordedWhole,
    /// At its batch's submission, where its command buffer may reach memory no use names, or an
    /// earlier encoder of it wrote the truth or the image.
    CommandBuffer,
    /// At its batch's submission, where its standing was let go of ([`LetGo`]).
    LetGo,
    /// At its batch's submission, where it stood stale inside a rectangle alone, whose texels
    /// the copy moved.
    StaleWithin,
}

/// What let go of a shadow's standing, or left it none (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LetGo {
    /// No command buffer's end left its image standing since it was made.
    NeverStood,
    /// A batch that may reach memory no use names was submitted since it stood.
    UnnamedSince,
    /// A batch submitted since it stood wrote its truth (or the host memory it joined): another
    /// shadow's copy back or merge over the same bytes, a blit's, a pass's or a kernel's write; or
    /// another API wrote an exported buffer's bytes, as its caller said
    /// (`Buffer::external_write_done`).
    TruthWritten,
    /// A batch submitted since it stood wrote its image: a copy in its standing did not leave.
    ImageWritten,
}

/// What last kept a command buffer's end from leaving a shadow standing, of one that never stood
/// (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Missed {
    /// No command buffer's end met it: none ended with it among what its encoders carried.
    Unmet,
    /// An encoder that used it ended with its image not holding the truth: a command wrote the
    /// truth another way after its copy in, another shadow's copy back or merge wrote the bytes,
    /// or nothing copied it in.
    NotCurrent,
    /// What the command buffer carried to its later encoders let it go: a later command wrote its
    /// truth another way, another shadow's copy back or merge wrote the bytes, work recorded apart
    /// stood between, or an encoder started after a wait, a signal or a host step.
    Dropped,
    /// The command buffer's end carried its image holding part of the truth.
    Partial,
    /// The command buffer's end records no standing: a failure, a rebind, memory no use names, or
    /// a last batch without its last encoder.
    End,
}

impl Missed {
    /// Every one, in the order `WorkCounts::shadow_texels_never_stood_*` counts them.
    pub(super) const ALL: [Missed; 5] = [
        Missed::Unmet,
        Missed::NotCurrent,
        Missed::Dropped,
        Missed::Partial,
        Missed::End,
    ];
}

/// The device's count of the writes that let go of its shadows' standings.
#[derive(Default)]
pub(super) struct Standings(Mutex<Watched>);

#[derive(Default)]
struct Watched {
    /// The keys a standing was recorded against: how many batches wrote each, and the shadows
    /// that stood against it, held weakly.
    keys: KeyMap<ResourceKey, Key>,
    /// How many batches of command buffers that may reach memory no use names were submitted.
    unnamed: u64,
    /// `keys`' length when its dead shadows were last let go of.
    pruned: usize,
}

#[derive(Default)]
struct Key {
    written: u64,
    shadows: Vec<Weak<ShadowCopies>>,
}

/// A shadow's standing across command buffers: what the writes counted stood at as a command
/// buffer's end last left its image holding every texel of the truth; and, for a shadow that may
/// stand, what tells whether its truth's handle was handed out.
#[derive(Default)]
pub(super) struct StandingAcross {
    stood: Mutex<Option<Stood>>,
    handle: Option<Handle>,
    /// What last kept a command buffer's end from leaving it standing ([`Missed`], as its `u8`).
    missed: AtomicU8,
}

/// The handle of a truth that may stand, which the caller's own work may write once it was
/// handed out.
enum Handle {
    /// A buffer of the runtime's own: whether its `VkBuffer` was handed out
    /// (`Buffer::vk_buffer`).
    Buffer(Arc<AtomicBool>),
    /// An image of the runtime's own, whose `VkImage` it tells (`Texture::vk_image`).
    Image(Arc<dyn ImageTruth>),
}

/// The writes counted against each of a shadow's keys, and the unnamed ones, as it stood; and
/// the rectangle bounding the texels of its image that writes within it changed since its copy
/// in (`shadow::Within`), stale there alone, `None` where none did.
struct Stood {
    written: Vec<(ResourceKey, u64)>,
    unnamed: u64,
    stale: Option<[u32; 4]>,
}

impl StandingAcross {
    /// The standing of a shadow over `buffer`'s bytes: one that may stand where the buffer is a
    /// tracked `Private` or `Memoryless` buffer the runtime made of its own (not a heap's, nor
    /// over the caller's memory), until its `VkBuffer` is handed out; none for any other.
    pub(super) fn over(buffer: &Buffer) -> StandingAcross {
        let private = matches!(
            buffer.resource_options().storage_mode(),
            Ok(StorageMode::Private | StorageMode::Memoryless)
        );
        let own = buffer.caller_memory().is_none() && buffer.heap().is_none();
        let tracked = buffer.hazard_tracking_mode() == HazardTrackingMode::Tracked;
        StandingAcross {
            stood: Mutex::new(None),
            handle: (private && own && tracked).then(|| Handle::Buffer(buffer.handed_out())),
            missed: AtomicU8::new(Missed::Unmet as u8),
        }
    }

    /// The standing of a shadow of `image`'s texels, where `stands` (a tracked `Private` or
    /// `Memoryless` image the runtime made in memory of its own, `shadow::Backing::PrivateImage`):
    /// one that may stand until its `VkImage` is handed out; none for any other.
    pub(super) fn over_image(image: Arc<dyn ImageTruth>, stands: bool) -> StandingAcross {
        StandingAcross {
            stood: Mutex::new(None),
            handle: stands.then_some(Handle::Image(image)),
            missed: AtomicU8::new(Missed::Unmet as u8),
        }
    }

    /// Whether the shadow may stand at all: its buffer's `VkBuffer`, or its image's `VkImage`,
    /// was never handed out.
    pub(super) fn may_stand(&self) -> bool {
        match &self.handle {
            Some(Handle::Buffer(out)) => !out.load(Ordering::Acquire),
            Some(Handle::Image(image)) => !image.handed_out(),
            None => false,
        }
    }

    fn stood(&self) -> MutexGuard<'_, Option<Stood>> {
        self.stood.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Note `why` a command buffer's end did not leave the shadow standing, where it may stand
    /// (the module docs).
    pub(super) fn missed(&self, why: Missed) {
        if self.handle.is_some() {
            self.missed.store(why as u8, Ordering::Relaxed);
        }
    }

    /// What last kept a command buffer's end from leaving the shadow standing.
    pub(super) fn last_missed(&self) -> Missed {
        let at = self.missed.load(Ordering::Relaxed);
        (Missed::ALL.into_iter())
            .find(|why| *why as u8 == at)
            .unwrap_or(Missed::Unmet)
    }
}

impl Standings {
    fn watched(&self) -> MutexGuard<'_, Watched> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// What let go of the standing of `copies`, a shadow that may stand, by the writes counted
    /// now, or left it none; `None` where it stands ([`standing`](Self::standing)).
    pub(super) fn let_go(&self, copies: &ShadowCopies) -> Option<LetGo> {
        self.standing(copies).err()
    }

    /// The standing of `copies`, a shadow that may stand, at a copy in submitted now: where it
    /// stands, its image holds every texel of its truth as it stood, and no write counted since
    /// let go of it, but for those inside the rectangle given, where writes within it had changed
    /// the truth since its copy in as it stood (`None` for none); else what let go of it, or left
    /// it none (the module docs).
    pub(super) fn standing(&self, copies: &ShadowCopies) -> Result<Option<[u32; 4]>, LetGo> {
        let watched = self.watched();
        let stood = copies.standing.stood();
        let Some(stood) = stood.as_ref() else {
            return Err(LetGo::NeverStood);
        };
        if stood.unnamed != watched.unnamed {
            return Err(LetGo::UnnamedSince);
        }
        let written = (stood.written.iter()).find(|(key, written)| {
            (watched.keys.get(key)).is_none_or(|known| known.written != *written)
        });
        let Some(&(key, _)) = written else {
            return Ok(stood.stale);
        };
        Err(match key == copies.shadow_key {
            true => LetGo::ImageWritten,
            false => LetGo::TruthWritten,
        })
    }

    /// Another API wrote the bytes of the buffer `key` names, and the caller said so once those
    /// writes completed (`Buffer::external_write_done`): every standing recorded against the key
    /// lets go, as a batch's write of it does ([`LetGo::TruthWritten`]), so the next command
    /// buffer's first copy in of each shadow over those bytes moves what the other API wrote. The
    /// caller makes the call while no command of the runtime's that uses the buffer is in flight
    /// (`exported_memory.rs`), so every batch submitted after it reads those writes.
    pub(super) fn written_apart(&self, key: ResourceKey) {
        if let Some(known) = self.watched().keys.get_mut(&key) {
            known.written += 1;
        }
    }

    /// A batch whose encoders' `summaries` these are was submitted: each key its tracked uses
    /// write, of those a standing was recorded against, counts one write more, and every one
    /// counts where the batch's command buffer may reach memory no use names (`unnamed`). Then
    /// each of `held` that may stand stands at what the counts reach: those a command buffer's end
    /// left holding every texel of their truth, where this batch ends it, each stale inside the
    /// rectangle given alone (the module docs).
    pub(super) fn submitted<'a>(
        &self,
        summaries: impl Iterator<Item = &'a Summary<ResourceKey>>,
        unnamed: bool,
        held: &[(Arc<ShadowCopies>, Option<[u32; 4]>)],
    ) {
        let held: Vec<&(Arc<ShadowCopies>, Option<[u32; 4]>)> = (held.iter())
            .filter(|(copies, _)| copies.stands_across())
            .collect();
        let mut watched = self.watched();
        if watched.keys.is_empty() && held.is_empty() {
            return;
        }
        if unnamed {
            watched.unnamed += 1;
        }
        if !watched.keys.is_empty() {
            for key in summaries.flat_map(Summary::written_keys) {
                if let Some(known) = watched.keys.get_mut(&key) {
                    known.written += 1;
                }
            }
        }
        for (copies, stale) in held {
            let keys = copies.standing_keys();
            let mut written = Vec::with_capacity(keys.len());
            for key in keys {
                let known = watched.keys.entry(key).or_default();
                let weak = Arc::downgrade(copies);
                if !(known.shadows.iter()).any(|shadow| Weak::ptr_eq(shadow, &weak)) {
                    known.shadows.retain(|shadow| shadow.strong_count() > 0);
                    known.shadows.push(weak);
                }
                written.push((key, known.written));
            }
            *copies.standing.stood() = Some(Stood {
                written,
                unnamed: watched.unnamed,
                stale: *stale,
            });
        }
        // The keys of shadows that are gone go once the map doubled since it was last swept, so
        // it holds about as many keys as live shadows stood against.
        if watched.keys.len() > 2 * watched.pruned.max(32) {
            watched.keys.retain(|_, known| {
                known.shadows.retain(|shadow| shadow.strong_count() > 0);
                !known.shadows.is_empty()
            });
            watched.pruned = watched.keys.len();
        }
    }
}
