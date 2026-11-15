//! What an encoder holds until its command buffer completes: the objects its commands use, and the
//! ones it was handed that none used (`argument_table.rs`), in the order it retained them.
//!
//! - **Without a box each.** A shared object (a buffer, a texture, a sampler state, a pipeline
//!   state, a variant's objects, a made `VkPipeline`) is held by another reference to it, which
//!   costs no allocation; any other object is boxed. A draw retains several of the first kind.
//! - **Once each, lately.** A shared object among the last [`RECENT`] the encoder held by
//!   reference is not held again: the encoder holds it already, until its command buffer
//!   completes, so no other object has its address meanwhile. A dispatch or draw of the pipeline
//!   state, variant objects and pipelines of the one before it holds none of them again, and a
//!   hold of one by a reference of the caller's ([`Retained::shared_of`]) takes no other
//!   reference to it, so its count is not touched at all.
//! - **Once each, through the command buffer.** An encoder its command buffer opens starts from
//!   what the command buffer's last ended encoder held lately ([`Lately`]: the last [`RECENT`] it
//!   held by reference or found held, then those it started from, [`SEEDED`] at most), and holds
//!   none of those again. A command buffer lets go of no entry before its release (its parts'
//!   completions, failures and encodings again add entries; none takes one away), so each lies
//!   among its entries until it completes, and no other object has its address meanwhile. A
//!   program that opens an encoder for each dispatch so holds each object once a command buffer,
//!   as one that encodes them all in one encoder does, not once an encoder: one holds entry, not
//!   one an encoder, which its release takes and drops under one lock each. Only the encoders the
//!   program opens start so (`EncoderCore::open`): a core that records apart, a segment's or an
//!   encoding again's, holds everything it names, as a recording kept for later command buffers
//!   takes every object its core holds (`kept_commands.rs`).
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

/// The shared objects held last that a hold by reference looks among (the module docs): a
/// dispatch's or draw's few.
const RECENT: usize = 8;

/// The shared objects a command buffer's next encoder starts from as held lately (the module
/// docs): a dispatch's or draw's few, those of the dispatch or draw before them among them.
const SEEDED: usize = 16;

// What an encoder held by reference is handed on whole ([`Retained::lately`]).
const _: () = assert!(RECENT <= SEEDED);

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
/// another's id retains it as before. A core that found the object among those its command buffer
/// holds lately ([`Retained::holds_lately`]) names itself too: the command buffer holds it until
/// it completes.
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

/// The addresses of the shared objects a command buffer holds lately, latest first, 0 past the
/// last (no object's): those its last ended encoder held by reference or found held, which its
/// next encoder holds none of again (the module docs' **Once each, through the command buffer**).
#[derive(Clone, Copy, Default)]
pub(super) struct Lately([usize; SEEDED]);

/// An encoder's holds (the module docs): none stored until the first.
#[derive(Default)]
pub(super) struct Retained {
    holds: Option<Box<Holds>>,
    /// The addresses of the last [`RECENT`] shared objects held, 0 where none (no object's), and
    /// the place the next goes.
    recent: ([usize; RECENT], usize),
    /// What its command buffer held lately as its encoder opened ([`seeded`](Self::seeded)).
    seeded: Lately,
    /// The shared objects asked for that were among them, so not held again (builds with debug
    /// assertions: [`asked`](Self::asked)).
    #[cfg(debug_assertions)]
    found_lately: usize,
}

impl Retained {
    /// The holds of an encoder its command buffer opens holding `lately`, none of which it holds
    /// again (the module docs).
    pub(super) fn seeded(lately: Lately) -> Retained {
        Retained {
            seeded: lately,
            ..Retained::default()
        }
    }

    /// Whether the shared object at `address` is held already: among the last [`RECENT`] held by
    /// reference, or among what the command buffer held lately as the encoder opened.
    pub(super) fn holds_lately(&mut self, address: usize) -> bool {
        let found = self.recent.0.contains(&address) || self.seeded.0.contains(&address);
        #[cfg(debug_assertions)]
        if found {
            self.found_lately += 1;
        }
        found
    }

    /// What it holds lately, for its command buffer's next encoder: the last [`RECENT`] shared
    /// objects it held by reference, latest first, then those it started from, the first
    /// [`SEEDED`]. Each is there once: a hold by reference looks among both before its object goes
    /// among the first, so no object is among both, nor twice among either.
    pub(super) fn lately(&self) -> Lately {
        let (recent, next) = &self.recent;
        let mut lately = [0; SEEDED];
        let mut kept = 0;
        // The places fill from the first on, so a 0 going back is where none was held yet.
        for back in 1..=RECENT {
            let address = recent[(next + RECENT - back) % RECENT];
            if address == 0 {
                break;
            }
            lately[kept] = address;
            kept += 1;
        }
        lately[kept..].copy_from_slice(&self.seeded.0[..SEEDED - kept]);
        Lately(lately)
    }

    /// Hold `object`, boxed.
    pub(super) fn boxed(&mut self, object: Box<dyn Any + Send + Sync>) {
        self.holds().push(Held::Boxed(object));
    }

    /// Hold `object`, a shared one, by another reference to it: none where it is held lately
    /// already ([`holds_lately`](Self::holds_lately), the module docs).
    pub(super) fn shared(&mut self, object: Arc<dyn Any + Send + Sync>) {
        let address = Arc::as_ptr(&object).cast::<()>().addr();
        if !self.holds_lately(address) {
            self.hold_shared(address, object);
        }
    }

    /// [`shared`](Self::shared) of the object `object` refers to, its other reference taken only
    /// where it is not held lately: a hold of one held lately touches no reference count.
    pub(super) fn shared_of<T: Any + Send + Sync>(&mut self, object: &Arc<T>) {
        let address = Arc::as_ptr(object).cast::<()>().addr();
        if !self.holds_lately(address) {
            self.hold_shared(address, Arc::<T>::clone(object));
        }
    }

    /// Hold `object`, at `address`, not held lately, as the last held by reference.
    fn hold_shared(&mut self, address: usize, object: Arc<dyn Any + Send + Sync>) {
        let (recent, next) = &mut self.recent;
        recent[*next] = address;
        *next = (*next + 1) % RECENT;
        self.holds().push(Held::Shared(object));
    }

    /// Hold what `other` holds too, after its own, in its order.
    pub(super) fn join(&mut self, other: Retained) {
        let Some(mut holds) = other.holds else {
            return;
        };
        self.holds().append(&mut holds.0);
    }

    /// Another reference to each object it holds, in order, where every hold is of a shared
    /// object (`kept_commands.rs`); `None` where one is boxed.
    pub(super) fn all_shared(&self) -> Option<Vec<Arc<dyn Any + Send + Sync>>> {
        let Some(holds) = &self.holds else {
            return Some(Vec::new());
        };
        holds
            .0
            .iter()
            .map(|held| match held {
                Held::Shared(object) => Some(Arc::clone(object)),
                Held::Boxed(_) => None,
            })
            .collect()
    }

    /// The holds' storage, made at the first.
    fn holds(&mut self) -> &mut Vec<Held> {
        let holds = self
            .holds
            .get_or_insert_with(|| Box::new(Holds(Vec::with_capacity(FIRST_HOLDS))));
        &mut holds.0
    }

    /// How many holds it keeps (builds with debug assertions: `render/draw_again.rs`'s check;
    /// and the tests).
    #[cfg(any(test, debug_assertions))]
    pub(super) fn len(&self) -> usize {
        self.holds.as_ref().map_or(0, |holds| holds.0.len())
    }

    /// How many holds it was asked for: those it keeps, and the shared objects it found held
    /// lately (builds with debug assertions: `render/draw_again.rs`'s check, which a hold found
    /// so must not pass unseen).
    #[cfg(debug_assertions)]
    pub(super) fn asked(&self) -> usize {
        self.len() + self.found_lately
    }

    /// The holds as one entry of a command buffer's (the module docs), or none where there are
    /// none.
    pub(super) fn into_entry(self) -> Option<Box<dyn Any + Send + Sync>> {
        self.holds.map(|holds| holds as Box<dyn Any + Send + Sync>)
    }
}
