//! The pipeline state the CPU set on each command of an indirect command buffer
//! ([`SetStates`]), under the buffer's `pipelines` lock.
//!
//! A command names one of the distinct states set on the buffer's commands, by a slot of its own:
//! a set of a state another command names already stores the slot and counts the command, so a
//! set costs a store whatever the buffer's size, and the table keeps one entry a state, however
//! many commands name it. An entry goes with the last command that names it, so the table holds
//! no state none of the commands names. A recovery's carry carries each entry once
//! ([`entries_mut`](SetStates::entries_mut)). Each command's key is also kept where a setter
//! reads it without the lock ([`SetKeys`]).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// What tells two states the CPU set apart: the kind, the ID and the serial of the state's hold
/// of it (`IdStamp`), which no other state's hold of any ID shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct StateKey {
    pub(super) compute: bool,
    pub(super) id: u64,
    pub(super) serial: u64,
}

/// A state a command names, by the key that tells it apart (`CommandState`'s).
pub(super) trait Keyed {
    fn key(&self) -> StateKey;
}

/// The key of the state each command names, as [`SetStates`] holds it, readable without the
/// buffer's lock: two words a command, the state's ID (0 for none) and its serial above its kind,
/// stored under the lock as a set or a clear changes the command's state, none until a state is
/// set. A setter of the state a command names already finds it here and writes its ID again
/// without the lock, the table unchanged. A command's two words are stored one after the other:
/// a read of them while another thread sets the same command is the program's race, as two
/// threads encoding one command are on Metal.
pub(super) struct SetKeys {
    size: u32,
    words: OnceLock<Box<[AtomicU64]>>,
}

impl SetKeys {
    fn new(size: u32) -> SetKeys {
        SetKeys {
            size,
            words: OnceLock::new(),
        }
    }

    /// Whether command `index` (below the size) names the state of key `key`.
    pub(super) fn names(&self, index: u32, key: StateKey) -> bool {
        let Some(words) = self.words.get() else {
            return false;
        };
        let at = 2 * index as usize;
        words[at].load(Ordering::Acquire) == key.id
            && words[at + 1].load(Ordering::Acquire) == serial_and_kind(key)
    }

    /// Command `index` (below the size) names the state of key `key`, or none.
    fn store(&self, index: u32, key: Option<StateKey>) {
        let words = self.words.get_or_init(|| {
            (0..2 * self.size as usize)
                .map(|_| AtomicU64::new(0))
                .collect()
        });
        let at = 2 * index as usize;
        let (id, serial) = key.map_or((0, 0), |key| (key.id, serial_and_kind(key)));
        words[at].store(id, Ordering::Release);
        words[at + 1].store(serial, Ordering::Release);
    }

    /// No command names a state.
    fn clear_all(&self) {
        for word in self.words.get().into_iter().flatten() {
            word.store(0, Ordering::Release);
        }
    }
}

/// `key`'s serial above its kind, in one word.
fn serial_and_kind(key: StateKey) -> u64 {
    key.serial << 1 | u64::from(key.compute)
}

/// A distinct state, and how many commands name it.
struct Entry<S> {
    state: S,
    commands: u32,
}

/// The pipeline state the CPU set on each command, by index (the module docs): `S` a
/// `CommandState`.
pub(super) struct SetStates<S> {
    /// The buffer's commands.
    size: u32,
    /// A command's slot: 0 for none, else 1 plus its entry's place in `entries`. Empty until a
    /// state is set, so a buffer whose commands never name one holds nothing for them.
    slots: Vec<u32>,
    /// The distinct states, by place; `None` at a place no state holds, which `free` lists.
    entries: Vec<Option<Entry<S>>>,
    free: Vec<u32>,
    /// Each entry's place, by its state's key.
    places: HashMap<StateKey, u32>,
    /// The place of the entry set last, looked at first: a buffer's commands mostly name one
    /// state in turn.
    last: Option<(StateKey, u32)>,
    /// Each command's key, for the setters' look without the lock.
    keys: Arc<SetKeys>,
}

impl<S: Keyed> SetStates<S> {
    /// The table of a buffer of `size` commands: none set.
    pub(super) fn new(size: u32) -> SetStates<S> {
        SetStates::empty(size, Arc::new(SetKeys::new(size)))
    }

    /// The table of a buffer of `size` commands, none set, its keys in `keys`.
    fn empty(size: u32, keys: Arc<SetKeys>) -> SetStates<S> {
        SetStates {
            size,
            slots: Vec::new(),
            entries: Vec::new(),
            free: Vec::new(),
            places: HashMap::new(),
            last: None,
            keys,
        }
    }

    /// Each command's key, readable without the lock ([`SetKeys`]).
    pub(super) fn keys(&self) -> Arc<SetKeys> {
        Arc::clone(&self.keys)
    }

    /// The state the CPU set on command `index`.
    pub(super) fn get(&self, index: u32) -> Option<&S> {
        let slot = *self.slots.get(index as usize)?;
        self.entry(slot).map(|entry| &entry.state)
    }

    /// The states the CPU set on commands `range`, in order, skipping those it set none on.
    pub(super) fn in_range(&self, range: Range<u32>) -> impl Iterator<Item = &S> {
        let slots = match self.slots.is_empty() {
            true => &[][..],
            false => &self.slots[range.start as usize..range.end as usize],
        };
        slots
            .iter()
            .filter_map(|&slot| self.entry(slot).map(|entry| &entry.state))
    }

    /// Set the state of key `key` on command `index` (below the size): the entry of that key
    /// where a command names one, else `make`'s state, a new entry.
    pub(super) fn set(&mut self, index: u32, key: StateKey, make: impl FnOnce() -> S) {
        let place = match self.last {
            Some((last, place)) if last == key => place,
            _ => match self.places.get(&key) {
                Some(&place) => place,
                None => self.add(key, make()),
            },
        };
        self.last = Some((key, place));
        if self.slots.is_empty() {
            self.slots = vec![0; self.size as usize];
        }
        let slot = place + 1;
        let before = std::mem::replace(&mut self.slots[index as usize], slot);
        // Stored again where the command named the state already: the keys may have been
        // forgotten since (`forget_keys`).
        self.keys.store(index, Some(key));
        if before == slot {
            return;
        }
        if let Some(entry) = self.entries[place as usize].as_mut() {
            entry.commands += 1;
        }
        self.release(before);
    }

    /// Command `index` (below the size) names no state from now on.
    pub(super) fn clear(&mut self, index: u32) {
        if let Some(slot) = self.slots.get_mut(index as usize) {
            let before = std::mem::replace(slot, 0);
            if before != 0 {
                self.keys.store(index, None);
            }
            self.release(before);
        }
    }

    /// The keys read without the lock name no state until a set stores each command's again,
    /// the table unchanged: a recovery's carry forgets them, so that each command's next set of
    /// its state is checked again, under the lock, against the generation it was carried to.
    pub(super) fn forget_keys(&self) {
        self.keys.clear_all();
    }

    /// No command names a state from now on.
    pub(super) fn clear_all(&mut self) {
        self.keys.clear_all();
        *self = SetStates::empty(self.size, Arc::clone(&self.keys));
    }

    /// Each distinct state a command names, once.
    pub(super) fn entries_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.entries
            .iter_mut()
            .flatten()
            .map(|entry| &mut entry.state)
    }

    /// The entry of slot `slot`: none for 0.
    fn entry(&self, slot: u32) -> Option<&Entry<S>> {
        let place = slot.checked_sub(1)?;
        self.entries[place as usize].as_ref()
    }

    /// `state`, of key `key`, as a new entry no command names yet: its place.
    fn add(&mut self, key: StateKey, state: S) -> u32 {
        let entry = Some(Entry { state, commands: 0 });
        let place = match self.free.pop() {
            Some(place) => {
                self.entries[place as usize] = entry;
                place
            }
            None => {
                self.entries.push(entry);
                (self.entries.len() - 1) as u32
            }
        };
        self.places.insert(key, place);
        place
    }

    /// A command no longer names slot `slot`'s entry (none for 0): the entry goes with the last.
    fn release(&mut self, slot: u32) {
        let Some(place) = slot.checked_sub(1) else {
            return;
        };
        let entry = self.entries[place as usize]
            .as_mut()
            .expect("a command's slot names an entry");
        entry.commands -= 1;
        if entry.commands > 0 {
            return;
        }
        let gone = self.entries[place as usize]
            .take()
            .expect("a command's slot names an entry");
        self.places.remove(&gone.state.key());
        self.free.push(place);
        if self.last.is_some_and(|(_, last)| last == place) {
            self.last = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state of key `.0`, made `.1`th.
    #[derive(Debug, PartialEq)]
    struct Made(StateKey, u32);

    impl Keyed for Made {
        fn key(&self) -> StateKey {
            self.0
        }
    }

    fn key(id: u64) -> StateKey {
        StateKey {
            compute: false,
            id,
            serial: 1,
        }
    }

    /// One entry a distinct state, made once while a command names it, shared by the commands
    /// that name it, gone with the last of them and made again after; a reset command names
    /// none; the range skips commands that name none. The keys read without the lock follow
    /// every set and clear, kind included, and come back with a set after they were forgotten.
    #[test]
    fn a_state_is_kept_once_while_a_command_names_it() {
        let mut table = SetStates::new(6);
        let keys = table.keys();
        assert!(table.get(0).is_none());
        assert!(!keys.names(0, key(7)), "none before a set");
        assert_eq!(table.in_range(0..6).count(), 0);
        let mut made = 0;
        let mut set = |table: &mut SetStates<Made>, index, id| {
            table.set(index, key(id), || {
                made += 1;
                Made(key(id), made)
            })
        };
        for index in 0..4 {
            set(&mut table, index, 7);
        }
        set(&mut table, 4, 9);
        set(&mut table, 1, 9);
        assert_eq!(table.get(0), Some(&Made(key(7), 1)));
        assert_eq!(table.get(1), Some(&Made(key(9), 2)));
        assert_eq!(table.get(5), None);
        assert!(keys.names(0, key(7)) && keys.names(1, key(9)));
        assert!(!keys.names(1, key(7)) && !keys.names(5, key(7)));
        let compute = StateKey {
            compute: true,
            ..key(7)
        };
        assert!(!keys.names(0, compute), "the kind tells keys apart");
        let ids: Vec<u64> = table.in_range(0..6).map(|state| state.0.id).collect();
        assert_eq!(ids, [7, 9, 7, 7, 9]);
        assert_eq!(table.entries_mut().count(), 2);
        // The commands that name 7 go: its entry with the last.
        for index in [0, 2] {
            table.clear(index);
        }
        assert!(!keys.names(0, key(7)) && keys.names(3, key(7)));
        assert_eq!(table.entries_mut().count(), 2);
        set(&mut table, 3, 9);
        assert_eq!(table.entries_mut().count(), 1);
        assert_eq!(table.in_range(0..6).count(), 3);
        // Set again, it is made again, at the free place.
        set(&mut table, 0, 7);
        assert_eq!(table.get(0), Some(&Made(key(7), 3)));
        assert_eq!(table.entries_mut().count(), 2);
        // A command set its own state again changes nothing.
        set(&mut table, 0, 7);
        assert_eq!(table.get(0), Some(&Made(key(7), 3)));
        table.clear_all();
        assert_eq!(table.in_range(0..6).count(), 0);
        assert_eq!(table.entries_mut().count(), 0);
        assert!(!keys.names(0, key(7)), "the same keys, cleared");
        set(&mut table, 2, 9);
        assert!(keys.names(2, key(9)));
        // Forgotten, then stored again by a set of the state the command names already.
        table.forget_keys();
        assert!(!keys.names(2, key(9)));
        set(&mut table, 2, 9);
        assert!(keys.names(2, key(9)));
        assert_eq!(table.entries_mut().count(), 1);
    }
}
