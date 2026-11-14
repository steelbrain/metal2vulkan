//! A sorted map whose copies share what they have in common: the device's live buffers
//! (`live_buffers.rs`), of which each resident set the device makes keeps a copy for the command
//! buffers that hold it.
//!
//! The entries lie in key order in chunks of at most [`MOST`] entries, each behind an `Arc`. A
//! copy ([`Clone`]) clones the chunks' `Arc`s, not the entries; an insertion or a removal changes
//! one chunk, which it copies first only where another copy still shares it (`Arc::make_mut`),
//! and splits a chunk past [`MOST`] in two and drops one left empty. So a change costs one
//! chunk's entries at most, whatever the length, and a copy made after it shares every other
//! chunk with the copies made before it.
//!
//! Each chunk keeps what its owner derives of its entries (`D`, [`SortedChunks::derived`]: the
//! live buffers' part of a resident set), made at most once for the chunk as it stands and shared
//! by every copy that shares the chunk; a change of the chunk drops it, and a chunk gone takes it
//! along. So what is derived of a list after a change is made again for the changed chunk alone.

use std::sync::{Arc, OnceLock};

/// The most entries a chunk holds: one that grows past it splits in two.
pub(crate) const MOST: usize = 64;

/// A map from `K` to `V`, sorted by `K`, in shared chunks, each with what is derived of it, `D`
/// (the module docs).
pub(crate) struct SortedChunks<K, V, D> {
    chunks: Vec<Arc<Chunk<K, V, D>>>,
    len: usize,
}

/// A chunk's entries in key order, and what was derived of them as they stand, once asked.
pub(crate) struct Chunk<K, V, D> {
    entries: Vec<(K, V)>,
    derived: OnceLock<D>,
}

impl<K: Clone, V: Clone, D> Clone for Chunk<K, V, D> {
    /// A copy to change (`Arc::make_mut`): its entries, with nothing derived of them yet.
    fn clone(&self) -> Chunk<K, V, D> {
        Chunk::new(self.entries.clone())
    }
}

impl<K, V, D> Chunk<K, V, D> {
    fn new(entries: Vec<(K, V)>) -> Chunk<K, V, D> {
        Chunk {
            entries,
            derived: OnceLock::new(),
        }
    }

    /// Its entries, to change: what was derived of them goes.
    fn entries_mut(&mut self) -> &mut Vec<(K, V)> {
        self.derived = OnceLock::new();
        &mut self.entries
    }

    /// Its entries in key order.
    #[cfg(test)]
    pub(crate) fn entries(&self) -> &[(K, V)] {
        &self.entries
    }
}

impl<K, V, D> Default for SortedChunks<K, V, D> {
    fn default() -> SortedChunks<K, V, D> {
        SortedChunks {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl<K, V, D> Clone for SortedChunks<K, V, D> {
    /// A copy sharing every chunk.
    fn clone(&self) -> SortedChunks<K, V, D> {
        SortedChunks {
            chunks: self.chunks.clone(),
            len: self.len,
        }
    }
}

impl<K: Ord + Copy, V: Clone, D> SortedChunks<K, V, D> {
    /// Whether it has no entry.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Its entries' count.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The chunk `key` belongs in: the last whose first key is at or below it, else the first.
    fn chunk_of(&self, key: &K) -> usize {
        self.chunk_where(|first| first <= key)
    }

    /// The last chunk whose first key `below` holds for, else the first; `below` holds for a
    /// prefix of the keys.
    fn chunk_where(&self, below: impl Fn(&K) -> bool) -> usize {
        self.chunks
            .partition_point(|chunk| (chunk.entries.first()).is_some_and(|(first, _)| below(first)))
            .saturating_sub(1)
    }

    /// Insert `value` at `key` where no entry has that key: whether it did.
    pub(crate) fn insert(&mut self, key: K, value: V) -> bool {
        if self.chunks.is_empty() {
            self.chunks.push(Arc::new(Chunk::new(vec![(key, value)])));
            self.len = 1;
            return true;
        }
        let at = self.chunk_of(&key);
        let found = self.chunks[at]
            .entries
            .binary_search_by(|(k, _)| k.cmp(&key));
        let Err(place) = found else {
            return false;
        };
        let chunk = Arc::make_mut(&mut self.chunks[at]).entries_mut();
        chunk.insert(place, (key, value));
        if chunk.len() > MOST {
            let back = chunk.split_off(chunk.len() / 2);
            self.chunks.insert(at + 1, Arc::new(Chunk::new(back)));
        }
        self.len += 1;
        true
    }

    /// Remove the entry at `key`: its value, where there was one.
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let at = self.chunk_of(key);
        let place = self
            .chunks
            .get(at)?
            .entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()?;
        let chunk = Arc::make_mut(&mut self.chunks[at]).entries_mut();
        let (_, value) = chunk.remove(place);
        if chunk.is_empty() {
            self.chunks.remove(at);
        }
        self.len -= 1;
        Some(value)
    }

    /// Whether it has an entry at `key`.
    pub(crate) fn contains(&self, key: &K) -> bool {
        let at = self.chunk_of(key);
        self.chunks.get(at).is_some_and(|chunk| {
            (chunk.entries)
                .binary_search_by(|(k, _)| k.cmp(key))
                .is_ok()
        })
    }

    /// The last entry whose key `below` holds for; `below` holds for a prefix of the keys.
    pub(crate) fn last_where(&self, below: impl Fn(&K) -> bool) -> Option<&(K, V)> {
        let chunk = &self.chunks.get(self.chunk_where(&below))?.entries;
        let after = chunk.partition_point(|(k, _)| below(k));
        chunk.get(after.checked_sub(1)?)
    }

    /// The entries in key order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(K, V)> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.entries.iter())
    }

    /// What `derive` makes of each chunk's entries, in key order: made at most once for a chunk as
    /// it stands, and kept with it for every copy that shares it (the module docs).
    pub(crate) fn derived<'a>(
        &'a self,
        derive: impl Fn(&[(K, V)]) -> D + 'a,
    ) -> impl Iterator<Item = &'a D> + 'a {
        (self.chunks.iter()).map(move |chunk| chunk.derived.get_or_init(|| derive(&chunk.entries)))
    }

    /// The chunks, for a test of what copies share.
    #[cfg(test)]
    pub(crate) fn chunks(&self) -> &[Arc<Chunk<K, V, D>>] {
        &self.chunks
    }
}
