//! The device's live buffers kept as they change (`sorted_chunks.rs`): a change costs one chunk,
//! and a copy made after it shares every other chunk with the copies before it, whatever the
//! length.

use crate::sorted_chunks::{SortedChunks, MOST};
use metal2vulkan::bind::{ResidentPart, ResidentRange, ResidentSet};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

thread_local! {
    /// The values cloned on this thread.
    static CLONES: Cell<usize> = const { Cell::new(0) };
}

/// A list of counted values, deriving nothing of its chunks.
type Map = SortedChunks<u64, Counted, ()>;

/// A value that counts its clones.
#[derive(Debug, PartialEq, Eq)]
struct Counted(u64);

impl Clone for Counted {
    fn clone(&self) -> Counted {
        CLONES.with(|clones| clones.set(clones.get() + 1));
        Counted(self.0)
    }
}

/// What `f` returned, and the values cloned while it ran.
fn clones_in<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = CLONES.with(Cell::get);
    let made = f();
    (made, CLONES.with(Cell::get) - before)
}

/// The chunks `a` and `b` share.
fn shared(a: &Map, b: &Map) -> usize {
    a.chunks()
        .iter()
        .filter(|chunk| b.chunks().iter().any(|other| Arc::ptr_eq(chunk, other)))
        .count()
}

/// The entries of `map` in order.
fn entries(map: &Map) -> Vec<u64> {
    map.iter().map(|(key, _)| *key).collect()
}

/// The regression check: in a list of 32768 buffers, a copy clones no entry, and each buffer made
/// or gone after it clones one chunk's entries at most, leaving every other chunk shared with the
/// copy, which keeps the list as it was. Before, each resident set made after a change held, sorted
/// and uploaded every listed buffer again.
#[test]
fn a_change_copies_one_chunk_and_the_copies_share_the_rest() {
    let mut map: Map = SortedChunks::default();
    for key in 0..32768u64 {
        assert!(map.insert(2 * key, Counted(key)));
    }
    let chunks = map.chunks().len();
    assert!(chunks >= 32768 / MOST, "{chunks} chunks");
    let (copy, cloned) = clones_in(|| map.clone());
    assert_eq!(cloned, 0, "a copy clones no entry");
    let before = entries(&copy);
    let (inserted, made) = clones_in(|| map.insert(20001, Counted(7)));
    assert!(inserted);
    assert!(made <= MOST, "a buffer made cloned {made} entries");
    assert!(
        shared(&map, &copy) >= chunks - 1,
        "chunks shared after a buffer made"
    );
    let (removed, gone) = clones_in(|| map.remove(&40000));
    assert_eq!(removed, Some(Counted(20000)));
    assert!(gone <= MOST, "a buffer gone cloned {gone} entries");
    assert!(
        shared(&map, &copy) >= chunks - 2,
        "chunks shared after a buffer gone"
    );
    assert_eq!(entries(&copy), before, "the copy keeps the list as it was");
    let mut after = before.clone();
    after.retain(|&key| key != 40000);
    let at = after.partition_point(|&key| key < 20001);
    after.insert(at, 20001);
    assert_eq!(entries(&map), after);
    // A change where no copy shares the chunk clones nothing.
    drop(copy);
    assert_eq!(clones_in(|| map.insert(20003, Counted(8))), (true, 0));
    assert_eq!(clones_in(|| map.remove(&20001)), (Some(Counted(7)), 0));
}

/// Through inserts and removes in a fixed pseudo-random order, against a `BTreeMap`: the entries
/// stay in key order, each chunk holds at most `MOST`, and the lookups answer as the map's.
#[test]
fn entries_stay_in_key_order_through_changes() {
    let mut map: Map = SortedChunks::default();
    let mut model = BTreeMap::new();
    let mut copies = Vec::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for step in 0..20000u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let key = seed % 3000;
        if seed % 5 < 3 {
            assert_eq!(map.insert(key, Counted(step)), !model.contains_key(&key));
            model.entry(key).or_insert(Counted(step));
        } else {
            assert_eq!(map.remove(&key), model.remove(&key));
        }
        if step % 997 == 0 {
            copies.push((map.clone(), model.clone()));
        }
    }
    let want: Vec<(u64, Counted)> = model.iter().map(|(k, v)| (*k, v.clone())).collect();
    let got: Vec<(u64, Counted)> = map.iter().cloned().collect();
    assert_eq!(got, want);
    assert!(map
        .chunks()
        .iter()
        .all(|chunk| !chunk.entries().is_empty() && chunk.entries().len() <= MOST));
    for probe in 0..3100u64 {
        assert_eq!(map.contains(&probe), model.contains_key(&probe));
        let below = map.last_where(|&key| key <= probe).map(|(key, _)| *key);
        assert_eq!(
            below,
            model.range(..=probe).next_back().map(|(key, _)| *key)
        );
    }
    // Each copy kept the list as it was when it was made.
    for (copy, model) in &copies {
        let got: Vec<(u64, Counted)> = copy.iter().cloned().collect();
        let want: Vec<(u64, Counted)> = model.iter().map(|(k, v)| (*k, v.clone())).collect();
        assert_eq!(got, want);
    }
    assert!(!map.is_empty());
    for key in model.keys() {
        assert!(map.remove(key).is_some());
    }
    assert!(map.is_empty() && map.chunks().is_empty());
}

/// What the live buffers derive of their chunks (`live_buffers.rs`): through inserts and removes in
/// a fixed pseudo-random order, the set joined of the chunks' parts (`ResidentSet::of_parts`) is
/// the set made of every listed range at once, entry for entry, for the list and for each copy
/// kept of it; a change makes the part of the chunk it changed again, or of the two it split into,
/// and no other; a copy makes none. The ranges are a list's: keyed by their buffer's address, some
/// starting below it (a no-copy buffer's pages before it), at hosts that disagree where they share
/// a start, some empty.
#[test]
fn the_set_joined_of_the_chunks_parts_is_the_set_of_the_list() {
    let made = Cell::new(0usize);
    let part = |chunk: &[(u64, ResidentRange)]| {
        made.set(made.get() + 1);
        Arc::new(ResidentPart::new(chunk.iter().map(|(_, range)| *range)))
    };
    let joined = |map: &SortedChunks<u64, ResidentRange, Arc<ResidentPart>>| {
        let before = made.get();
        let parts: Vec<Arc<ResidentPart>> = map.derived(part).cloned().collect();
        let set = ResidentSet::of_parts(parts.iter().map(|part| &**part));
        (set, made.get() - before)
    };
    let fresh = |map: &SortedChunks<u64, ResidentRange, Arc<ResidentPart>>| {
        ResidentSet::new(map.iter().map(|(_, range)| *range))
    };
    let mut map = SortedChunks::default();
    let mut copies = Vec::new();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for step in 0..6000u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let address = 0x10_0000 + 16 * (seed % 1500);
        let changed = match seed % 7 < 4 {
            true => {
                let below = match (seed >> 20) % 6 {
                    0 => 16 * ((seed >> 24) % 300),
                    _ => 0,
                };
                let length = match (seed >> 32) % 8 {
                    0 => 0,
                    _ => below + 1 + (seed >> 40) % 400,
                };
                let host = match (seed >> 48) % 4 {
                    0 => seed,
                    _ => address + 0x7000_0000,
                };
                let range = ResidentRange::new(host.wrapping_sub(below), address - below, length);
                map.insert(address, range)
            }
            false => map.remove(&address).is_some(),
        };
        let (set, parts) = joined(&map);
        assert_eq!(set, fresh(&map), "step {step}");
        let most = match changed {
            true => 2,
            false => 0,
        };
        let chunks = map.chunks().len();
        assert!(parts <= most, "step {step}: {parts} parts made of {chunks}");
        if step % 499 == 0 {
            copies.push(map.clone());
        }
    }
    for copy in &copies {
        let (set, parts) = joined(copy);
        assert_eq!(set, fresh(copy));
        assert_eq!(parts, 0, "a copy made parts again");
    }
}
