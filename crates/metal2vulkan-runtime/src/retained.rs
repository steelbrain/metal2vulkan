//! What an encoder holds until its command buffer completes: the objects its commands use, and the
//! ones it was handed that none used (`argument_table.rs`), in the order it retained them.
//!
//! - **Without a box each.** A shared object (a buffer, a texture, a sampler state, a pipeline
//!   state, a variant's objects, a made `VkPipeline`) is held by another reference to it, which
//!   costs no allocation; any other object is boxed. A draw retains several of the first kind.
//! - **One entry.** The holds go to the command buffer as one of its entries, boxed as they
//!   began: their storage, made at the first hold with room for [`FIRST_HOLDS`], is that box.
//! - **Let go of last first.** The command buffer lets go of what it holds from the last retained
//!   back, one entry at a time (`CommandBuffer`'s release, after its completed handlers): the
//!   holds go in that same order, so each object goes where its earliest hold goes, as when each
//!   hold was an entry of its own.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The holds an encoder's storage makes room for at its first: a pass's draws of a few pipeline
/// states, each of a few objects, without growing it.
const FIRST_HOLDS: usize = 32;

/// One object held.
enum Held {
    Shared(Arc<dyn Any + Send + Sync>),
    Boxed(Box<dyn Any + Send + Sync>),
}

/// The holds, in the order retained.
struct Holds(Vec<Held>);

/// Last retained first, as the command buffer lets go of its own entries.
impl Drop for Holds {
    fn drop(&mut self) {
        while let Some(held) = self.0.pop() {
            match held {
                Held::Shared(object) => drop(object),
                Held::Boxed(object) => drop(object),
            }
        }
    }
}

/// The encoder core whose command buffer holds an object an argument table binds, from a retain
/// of the table that kept its key among the core's (`argument_table::retain_into`): that core's
/// [`EncoderCore::id`](super::encoder::EncoderCore::id), 0 for none (ids start at 1). A core
/// holds what it retained until its command buffer completes, and no other core has its id, so
/// while the object lives the core named here holds it and keeps its key: that core retains it
/// again for nothing. Only a core that just retained the object names itself; a core that reads
/// another's id retains it as before.
#[derive(Default)]
pub(super) struct HeldBy(AtomicU64);

impl HeldBy {
    /// Whether the core `core` holds the object and keeps its key.
    pub(super) fn is(&self, core: u64) -> bool {
        self.0.load(Ordering::Relaxed) == core
    }

    /// Name `core`, which just retained the object and kept its key.
    pub(super) fn set(&self, core: u64) {
        self.0.store(core, Ordering::Relaxed);
    }
}

/// An encoder's holds (the module docs): none stored until the first.
#[derive(Default)]
pub(super) struct Retained(Option<Box<Holds>>);

impl Retained {
    /// Hold `object`, boxed.
    pub(super) fn boxed(&mut self, object: Box<dyn Any + Send + Sync>) {
        self.holds().push(Held::Boxed(object));
    }

    /// Hold `object`, a shared one, by another reference to it.
    pub(super) fn shared(&mut self, object: Arc<dyn Any + Send + Sync>) {
        self.holds().push(Held::Shared(object));
    }

    /// The holds' storage, made at the first.
    fn holds(&mut self) -> &mut Vec<Held> {
        let holds = self
            .0
            .get_or_insert_with(|| Box::new(Holds(Vec::with_capacity(FIRST_HOLDS))));
        &mut holds.0
    }

    /// How many holds it keeps (builds with debug assertions: `render/draw_again.rs`'s check).
    #[cfg(debug_assertions)]
    pub(super) fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |holds| holds.0.len())
    }

    /// The holds as one entry of a command buffer's (the module docs), or none where there are
    /// none.
    pub(super) fn into_entry(self) -> Option<Box<dyn Any + Send + Sync>> {
        self.0.map(|holds| holds as Box<dyn Any + Send + Sync>)
    }
}
