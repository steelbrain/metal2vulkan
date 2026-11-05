//! A shadow's standing across command buffers: where only the runtime's commands write a
//! shadowed texture's truth (`ShadowCopies::only_commands_write`: a tracked `Private` buffer's
//! bytes the runtime made, an IOSurface plane's among them), an image that a command buffer's end
//! left holding every texel of the truth still holds it at a later command buffer's first use
//! unless a command written since wrote the truth or the image. That first copy in then moves
//! nothing (`WorkCounts::shadow_copies_in_skipped_by_standing`).
//!
//! - **The order.** Every decision and every write it counts is taken as a batch is submitted,
//!   in the scheduler's order over all queues (`command_buffer/submit.rs`), which is the order
//!   the tracker orders tracked work in: a write submitted before a copy in that reads what it
//!   wrote runs before it, and one submitted after it runs after it. So the copy in reads the
//!   truth as every write before it in that order left it, as Metal's does.
//! - **What a command buffer leaves.** Its last batch, where it holds the command buffer's last
//!   encoder and nothing failed it, records each shadow its end left holding every texel of the
//!   truth, none stale (`ShadowCarry::held_whole`), as standing at what the writes counted so far
//!   leave: the batch's own writes counted first, its copies back among them.
//! - **What lets go.** A batch's tracked write of the truth's buffer or of the shadow's image (its
//!   key, whatever the range) lets go of the standing: another shadow's copy back over the same
//!   bytes, a blit's or a pass's write, a copy in that a command buffer's end did not leave
//!   standing. So does any batch of a command buffer whose commands may reach memory no use names
//!   (a root table, an execution resolved from records: `State::reaches_unnamed`, or one that
//!   noted nothing as it encoded), for every shadow. A shadow over a buffer whose `VkBuffer` was
//!   handed out (`Buffer::vk_buffer`), which the caller's own work may write, never stands.
//! - **The copy in.** A command buffer's first copy in of such a shadow, before any command of
//!   its encoder wrote what it reads, goes to its batch's submission (`shadow::GatedCopyIn`), and
//!   copies nothing there where the shadow stands and no earlier encoder of the command buffer
//!   wrote the truth or the image, and the command buffer reaches nothing unnamed. Every other
//!   copy is as the module `shadow.rs` states.

use super::buffer::Buffer;
use super::key_hash::KeyMap;
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::tracker::Summary;
use super::values::StorageMode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

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
/// buffer's end last left its image holding every texel of the truth; and, for a shadow over a
/// buffer of the runtime's own, whether its `VkBuffer` was handed out.
#[derive(Default)]
pub(super) struct StandingAcross {
    stood: Mutex<Option<Stood>>,
    handed_out: Option<Arc<AtomicBool>>,
}

/// The writes counted against each of a shadow's keys, and the unnamed ones, as it stood.
struct Stood {
    written: Vec<(ResourceKey, u64)>,
    unnamed: u64,
}

impl StandingAcross {
    /// The standing of a shadow over `buffer`'s bytes: one that may stand where the buffer is a
    /// `Private` or `Memoryless` buffer the runtime made of its own (not a heap's, nor over the
    /// caller's memory), until its `VkBuffer` is handed out; none for any other.
    pub(super) fn over(buffer: &Buffer) -> StandingAcross {
        let private = matches!(
            buffer.resource_options().storage_mode(),
            Ok(StorageMode::Private | StorageMode::Memoryless)
        );
        let own = buffer.caller_memory().is_none() && buffer.heap().is_none();
        StandingAcross {
            stood: Mutex::new(None),
            handed_out: (private && own).then(|| buffer.handed_out()),
        }
    }

    /// Whether the shadow may stand at all: its buffer's `VkBuffer` was never handed out.
    pub(super) fn may_stand(&self) -> bool {
        (self.handed_out.as_ref()).is_some_and(|out| !out.load(Ordering::Acquire))
    }

    fn stood(&self) -> MutexGuard<'_, Option<Stood>> {
        self.stood.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Standings {
    fn watched(&self) -> MutexGuard<'_, Watched> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Whether `copies`' image holds every texel of its truth at a copy in submitted now: it
    /// stood, and no write counted since let go of it (the module docs).
    pub(super) fn stands(&self, copies: &ShadowCopies) -> bool {
        if !copies.stands_across() {
            return false;
        }
        let watched = self.watched();
        let stood = copies.standing.stood();
        let Some(stood) = stood.as_ref() else {
            return false;
        };
        stood.unnamed == watched.unnamed
            && (stood.written.iter()).all(|(key, written)| {
                (watched.keys.get(key)).is_some_and(|known| known.written == *written)
            })
    }

    /// A batch whose encoders' `summaries` these are was submitted: each key its tracked uses
    /// write, of those a standing was recorded against, counts one write more, and every one
    /// counts where the batch's command buffer may reach memory no use names (`unnamed`). Then
    /// each of `held` that may stand stands at what the counts reach: those a command buffer's end
    /// left holding every texel of their truth, where this batch ends it (the module docs).
    pub(super) fn submitted<'a>(
        &self,
        summaries: impl Iterator<Item = &'a Summary<ResourceKey>>,
        unnamed: bool,
        held: &[Arc<ShadowCopies>],
    ) {
        let held: Vec<&Arc<ShadowCopies>> = (held.iter())
            .filter(|copies| copies.stands_across())
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
        for copies in held {
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
