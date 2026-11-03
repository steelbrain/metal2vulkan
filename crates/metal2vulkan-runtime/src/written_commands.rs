//! What an indirect command buffer's records may hold ([`WrittenCommands`]), gathered from every
//! route that writes them as each is encoded, so that a commit can tell from the call alone what
//! the GPU may find there, whatever wrote it.
//!
//! - **The routes.** The CPU's setters add what the record holds after each edit: its draw or
//!   dispatch (a draw's primitive type among them), its barrier, and the state it names, with
//!   what that state's functions may write and whether they may encode. An encoder that made the
//!   buffer resident for writing adds what its states may write (`indirect_command_writes`, read
//!   from their functions' AIR): the state set when the buffer was made resident, each set after,
//!   and the states an execution in the encoder may run (the executed buffer's named states;
//!   inherited states are the encoder's own). A blit's copy adds the source's, and for good: the
//!   source's later gains reach the copy too, since what it copies is the source's records when
//!   the GPU runs it. A reset, the CPU's or a blit's, adds nothing. Nothing is taken away: a
//!   record written again keeps the earlier writes in the summary, as a GPU write may stay.
//! - **What it is for.** A clear bit is proven of every write encoded so far; a route encoded
//!   later (an encoder that makes the buffer resident for writing after a commit read the
//!   summary) adds its own then. A commit that left a buffer unmarked though its command buffer
//!   made it resident for writing (`write_routes.rs`) is followed by the buffers it executed:
//!   where one's summary first names a state whose functions may write records, it marks them.

use std::sync::{Arc, Mutex, Weak};

use metal2vulkan::pipeline::CommandWrites;

use super::device::lock;
use super::write_routes::{followed, Followed, Unmarked};

/// What the states an indirect command buffer's records may name may do: their functions' writes
/// into the buffers they reach, together, and whether one may encode, reset or copy indirect
/// commands at all (`can_encode_indirect_commands`, which a function that takes a command buffer
/// it never writes answers too).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Named {
    pub(crate) encodes: bool,
    pub(crate) writes: CommandWrites,
}

impl Named {
    /// Any state: a pipeline the GPU sets in a record, which no fact names.
    const ANY: Named = Named {
        encodes: true,
        writes: CommandWrites::ANY,
    };

    /// What either may do.
    pub(crate) fn union(self, other: Named) -> Named {
        Named {
            encodes: self.encodes || other.encodes,
            writes: self.writes.union(other.writes),
        }
    }
}

/// What one indirect command buffer's records may hold (the module docs).
#[derive(Debug, Default)]
pub(crate) struct WrittenCommands {
    held: Mutex<Held>,
}

#[derive(Debug, Default)]
struct Held {
    writes: CommandWrites,
    named: Named,
    /// The buffers a blit copied this one's records into.
    copies: Vec<Weak<WrittenCommands>>,
    /// The buffers commits left unmarked where this one's commands ran, until the states its
    /// records may name may write records (`follow`).
    followers: Vec<Weak<Unmarked>>,
}

impl Held {
    /// Add `writes` and `named`; where that changed what it holds, the buffers a blit copied its
    /// records into, which gain them too, and the followers it sets off: those it held, where the
    /// states it may name may now write records and could not before.
    fn grow(
        &mut self,
        writes: CommandWrites,
        named: Named,
    ) -> Option<(Vec<Arc<WrittenCommands>>, Followed)> {
        let (grown, grown_named) = (self.writes.union(writes), self.named.union(named));
        if grown == self.writes && grown_named == self.named {
            return None;
        }
        let sets_off = self.named.writes.is_empty() && !grown_named.writes.is_empty();
        self.writes = grown;
        self.named = grown_named;
        self.copies.retain(|copy| copy.strong_count() > 0);
        let followed = match sets_off {
            true => followed(&mut self.followers),
            false => Followed::default(),
        };
        Some((
            self.copies.iter().filter_map(Weak::upgrade).collect(),
            followed,
        ))
    }
}

impl WrittenCommands {
    /// The operations and primitive types the records may hold.
    pub(crate) fn writes(&self) -> CommandWrites {
        lock(&self.held).writes
    }

    /// What the states the records may name may do.
    pub(crate) fn named(&self) -> Named {
        lock(&self.held).named
    }

    /// Add `writes` and `named` to the buffer's, and to every buffer a blit copied its records
    /// into, at any remove. What adds nothing to the buffer's adds nothing to its copies', which
    /// hold the buffer's: it is looked at under the buffer's lock alone, as most setters' calls
    /// add nothing new. The followers it sets off, of any of them, the caller marks once it holds
    /// no lock of a buffer's.
    pub(crate) fn add(self: &Arc<Self>, writes: CommandWrites, named: Named) -> Followed {
        let (copies, mut followed) = match lock(&self.held).grow(writes, named) {
            Some(grown) => grown,
            None => return Followed::default(),
        };
        let mut seen: Vec<*const WrittenCommands> = vec![Arc::as_ptr(self)];
        let mut next = copies;
        while let Some(buffer) = next.pop() {
            if seen.contains(&Arc::as_ptr(&buffer)) {
                continue;
            }
            seen.push(Arc::as_ptr(&buffer));
            if let Some((copies, more)) = lock(&buffer.held).grow(writes, named) {
                next.extend(copies);
                followed.0.extend(more.0);
            }
        }
        followed
    }

    /// Hold `unmarked` weakly until the states the records may name may write records, which
    /// sets it off then (`add`); `false`, holding nothing, where they may already.
    pub(crate) fn follow(&self, unmarked: &Arc<Unmarked>) -> bool {
        let mut held = lock(&self.held);
        if !held.named.writes.is_empty() {
            return false;
        }
        held.followers
            .retain(|follower| follower.strong_count() > 0);
        held.followers.push(Arc::downgrade(unmarked));
        true
    }

    /// What the CPU's edit left in a record (`CommandRecord::writes`: its action and its
    /// barrier), and the state it names.
    pub(crate) fn add_record(
        self: &Arc<Self>,
        mut writes: CommandWrites,
        named: Option<Named>,
    ) -> Followed {
        if named.is_some() {
            writes.operations |= CommandWrites::PIPELINE;
        }
        self.add(writes, named.unwrap_or_default())
    }

    /// What a state that may write the records adds: its functions' writes, and where they may
    /// set a pipeline or copy a command (whose pipeline comes along), any state named. A copy
    /// brings whatever the source command held, which its writer's own operations do not state:
    /// a function that may copy a command may write anything.
    pub(crate) fn add_writer(self: &Arc<Self>, writes: CommandWrites) -> Followed {
        let (writes, named) = if writes.may(CommandWrites::COPY) {
            (CommandWrites::ANY, Named::ANY)
        } else if writes.may(CommandWrites::PIPELINE) {
            (writes, Named::ANY)
        } else {
            (writes, Named::default())
        };
        self.add(writes, named)
    }

    /// A blit copies `source`'s records into this buffer's: what the source holds now, and what
    /// it gains later.
    pub(crate) fn copied_from(self: &Arc<Self>, source: &Arc<WrittenCommands>) -> Followed {
        let (writes, named) = {
            let mut held = lock(&source.held);
            held.copies.retain(|copy| copy.strong_count() > 0);
            if !held
                .copies
                .iter()
                .any(|copy| copy.as_ptr() == Arc::as_ptr(self))
            {
                held.copies.push(Arc::downgrade(self));
            }
            (held.writes, held.named)
        };
        self.add(writes, named)
    }
}

/// The indirect command buffers an encoder made resident for writing, and what its state set last
/// may write into them: Metal's `useResource:usage:` reaches the encoder's later commands, so a
/// buffer made resident takes the state already set and every state set after it.
#[derive(Debug, Default)]
pub(crate) struct Writers {
    open: Vec<Arc<WrittenCommands>>,
    /// What the state set last may write.
    state: CommandWrites,
}

impl Writers {
    /// The encoder made the buffer of `written` resident for writing.
    pub(crate) fn made_writable(&mut self, written: &Arc<WrittenCommands>) {
        if self.open.iter().any(|open| Arc::ptr_eq(open, written)) {
            return;
        }
        written.add_writer(self.state).mark();
        self.open.push(Arc::clone(written));
    }

    /// The encoder set a state whose functions may write `writes`.
    pub(crate) fn state_set(&mut self, writes: CommandWrites) {
        self.state = writes;
        self.runs(writes);
    }

    /// The encoder runs functions that may write `writes` (an execution's commands under states
    /// their records name).
    pub(crate) fn runs(&self, writes: CommandWrites) {
        if writes.is_empty() {
            return;
        }
        for open in &self.open {
            open.add_writer(writes).mark();
        }
    }
}
