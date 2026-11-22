//! The buffer a visible-function table is bound as.
//!
//! A `visible_function_table` argument compiles to a read-only buffer at its Metal buffer index
//! (`pipeline::table` has the word layout), so the binder binds a table like any other buffer:
//! `setVisibleFunctionTable:atBufferIndex:` is a [`BoundBuffer::Buffer`] at that index of the
//! stage's table ([`StageTable::buffers`]), over a buffer the consumer owns for the
//! `MTLVisibleFunctionTable` and fills with the table's words. The shader reads its words
//! through the buffer slot only, never through the guest address, so the bound buffer's
//! `gpu_address` is not read for it. `setVisibleFunctionTable:nil` binds nothing: the slot gets
//! the plan's zero-filled neutral, whose word 0 makes `is_null_visible_function_table` true.
//! `setFunction:atIndex:` writes one entry: the word of the function's
//! [`FunctionHandle`] (`LinkedFunctions::function_handle` of the stage the table belongs to), or
//! [`FunctionHandle::NULL_WORD`] for a null entry.
//!
//! [`BoundBuffer::Buffer`]: crate::bind::BoundBuffer::Buffer
//! [`StageTable::buffers`]: crate::bind::StageTable::buffers

use crate::air::canon::function_table::FIRST_ENTRY_WORD;
#[cfg(test)]
use crate::air::canon::function_table::{PRESENT_WORD, SIZE_WORD};
#[cfg(test)]
use crate::pipeline::table::FunctionHandle;

/// The function-table snapshot: the words of a command's listed tables
/// (`BindState::function_tables`) in one read-only storage buffer, which the consumer writes as the
/// command's command buffer commits
/// ([`WriteSource::FunctionTableSnapshot`](crate::bind::WriteSource::FunctionTableSnapshot)).
///
/// Word 0 is the null table's record: a word count of 0. Listed table `i`'s record follows the
/// records before it: its word count (its buffer's words: the table's header words and one
/// entry word for each of its size), then those words. The stage's function-table directory gives each listed
/// nonzero id the offset of its record's first word, and the null id and every id the directory
/// misses the null table's, 0: a read of word `w` of a table reads its record's word `1 + w`
/// below the record's count, and 0 at or past it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionTableSnapshot {
    /// The word offset of each listed table's record, in the list's order.
    offsets: Vec<u32>,
    /// The snapshot's length in words.
    words: u32,
}

impl FunctionTableSnapshot {
    /// The snapshot of tables of `sizes` entries, in the list's order; `None` when it would pass
    /// `u32::MAX` words.
    pub fn of(sizes: impl IntoIterator<Item = u32>) -> Option<FunctionTableSnapshot> {
        let mut offsets = Vec::new();
        let mut words: u32 = 1;
        for size in sizes {
            offsets.push(words);
            let record = FIRST_ENTRY_WORD.checked_add(size)?.checked_add(1)?;
            words = words.checked_add(record)?;
        }
        Some(FunctionTableSnapshot { offsets, words })
    }

    /// The word offset of each listed table's record, in the list's order: what the directory
    /// gives its id.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// The snapshot's length in bytes.
    #[cfg(test)]
    pub(crate) fn byte_len(&self) -> u64 {
        u64::from(self.words) * 4
    }

    /// The snapshot's bytes (little-endian words), of the listed tables' buffers `tables` in the
    /// list's order, each as [`FunctionTableWords::to_bytes`] gives it. `None` when `tables` is
    /// not one buffer of its size's length for each listed table.
    #[cfg(test)]
    pub(crate) fn bytes<'a>(&self, tables: impl IntoIterator<Item = &'a [u8]>) -> Option<Vec<u8>> {
        let len = usize::try_from(self.byte_len()).ok()?;
        let mut out = Vec::with_capacity(len);
        out.extend(0u32.to_le_bytes());
        let mut tables = tables.into_iter();
        for (i, &offset) in self.offsets.iter().enumerate() {
            let end = self.offsets.get(i + 1).copied().unwrap_or(self.words);
            let count = end - offset - 1;
            let table = tables.next()?;
            if table.len() as u64 != u64::from(count) * 4 {
                return None;
            }
            out.extend(count.to_le_bytes());
            out.extend_from_slice(table);
        }
        match tables.next() {
            Some(_) => None,
            None => Some(out),
        }
    }
}

/// The words of one visible-function table's buffer: bound, its size, and its entries.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FunctionTableWords {
    words: Vec<u32>,
}

#[cfg(test)]
impl FunctionTableWords {
    /// A bound table of `size` null entries (`newVisibleFunctionTableWithDescriptor:`).
    pub(crate) fn new(size: u32) -> FunctionTableWords {
        let len = FIRST_ENTRY_WORD as usize + size as usize;
        let mut words = vec![FunctionHandle::NULL_WORD; len];
        if let Some(w) = words.get_mut(PRESENT_WORD as usize) {
            *w = 1;
        }
        if let Some(w) = words.get_mut(SIZE_WORD as usize) {
            *w = size;
        }
        FunctionTableWords { words }
    }

    /// The table's size.
    pub(crate) fn size(&self) -> u32 {
        self.words.get(SIZE_WORD as usize).copied().unwrap_or(0)
    }

    /// Set entry `index` to `function`, or to null (`setFunction:atIndex:`). An index past the
    /// table's size is refused (`false`, nothing written), as Metal's validation refuses it.
    pub(crate) fn set(&mut self, index: u32, function: Option<FunctionHandle>) -> bool {
        if index >= self.size() {
            return false;
        }
        let at = FIRST_ENTRY_WORD as usize + index as usize;
        match self.words.get_mut(at) {
            Some(w) => {
                *w = Self::entry_word(function);
                true
            }
            None => false,
        }
    }

    /// Set entry `index` to Metal's opaque triangle intersection function
    /// (`setOpaqueTriangleIntersectionFunctionWithSignature:atIndex:` of an intersection function
    /// table): [`FunctionHandle::RESERVED_WORD`], which no function handle has. An index past the
    /// table's size is refused (`false`, nothing written).
    pub(crate) fn set_opaque_triangle(&mut self, index: u32) -> bool {
        if index >= self.size() {
            return false;
        }
        let at = FIRST_ENTRY_WORD as usize + index as usize;
        match self.words.get_mut(at) {
            Some(w) => {
                *w = FunctionHandle::RESERVED_WORD;
                true
            }
            None => false,
        }
    }

    /// The words, in buffer order.
    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }

    /// The buffer's bytes (little-endian words): what the consumer uploads.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The length in bytes of the buffer of a table of `size` entries.
    pub(crate) fn byte_len(size: u32) -> u64 {
        (u64::from(FIRST_ENTRY_WORD) + u64::from(size)) * 4
    }

    /// The words before the entries of a table of `size` entries: bound, then its size. A
    /// consumer that keeps the words in its own memory writes these, then each entry's
    /// [`entry_word`](Self::entry_word) at [`entry_offset`](Self::entry_offset).
    pub(crate) fn header(size: u32) -> [u32; FIRST_ENTRY_WORD as usize] {
        let mut words = [0; FIRST_ENTRY_WORD as usize];
        words[PRESENT_WORD as usize] = 1;
        words[SIZE_WORD as usize] = size;
        words
    }

    /// The offset in bytes of entry `index` in a table's buffer.
    pub(crate) fn entry_offset(index: u32) -> u64 {
        Self::byte_len(index)
    }

    /// The word of an entry holding `function`, or of a null entry.
    pub(crate) fn entry_word(function: Option<FunctionHandle>) -> u32 {
        function.map_or(FunctionHandle::NULL_WORD, FunctionHandle::word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table's buffer is a bound marker, its size, and one word per entry; a null entry is 0.
    #[test]
    fn a_table_is_its_marker_size_and_entries() {
        let mut t = FunctionTableWords::new(3);
        assert_eq!(t.words(), [1, 3, 0, 0, 0]);
        assert_eq!(FunctionTableWords::header(3), t.words()[..2]);
        assert_eq!(FunctionTableWords::entry_offset(1), 12);
        assert_eq!(FunctionTableWords::entry_word(None), 0);
        assert_eq!(FunctionTableWords::byte_len(3), 20);
        assert_eq!(t.to_bytes().len() as u64, FunctionTableWords::byte_len(3));
        assert!(!t.set(3, None), "past the size");
        assert_eq!(t.words(), [1, 3, 0, 0, 0]);
        assert!(t.set(2, None));
        assert_eq!(t.words(), [1, 3, 0, 0, 0]);
        let empty = FunctionTableWords::new(0);
        assert_eq!(empty.words(), [1, 0]);
        assert_eq!(empty.size(), 0);
    }

    /// A snapshot is the null table's record, then each table's word count and words; the
    /// offsets are where each record starts, and buffers of other lengths than the sizes' make
    /// none.
    #[test]
    fn a_snapshot_holds_each_table_after_its_word_count() {
        let snapshot = FunctionTableSnapshot::of([3, 0]).expect("snapshot");
        assert_eq!(snapshot.offsets(), [1, 7]);
        assert_eq!(snapshot.byte_len(), 4 * 10);
        let mut t = FunctionTableWords::new(3);
        assert!(t.set_opaque_triangle(1));
        let empty = FunctionTableWords::new(0);
        let (t, empty) = (t.to_bytes(), empty.to_bytes());
        let bytes = snapshot
            .bytes([t.as_slice(), empty.as_slice()])
            .expect("bytes");
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let entry = FunctionHandle::RESERVED_WORD;
        assert_eq!(words, [0, 5, 1, 3, 0, entry, 0, 2, 1, 0]);
        assert_eq!(snapshot.bytes([t.as_slice()]), None, "a table missing");
        assert_eq!(
            snapshot.bytes([empty.as_slice(), t.as_slice()]),
            None,
            "the sizes out of order"
        );
        assert_eq!(
            FunctionTableSnapshot::of([]).expect("empty").byte_len(),
            4,
            "the null table's record alone"
        );
        assert_eq!(FunctionTableSnapshot::of([u32::MAX]), None);
    }

    /// An intersection function table's opaque triangle entry is the reserved word 1.
    #[test]
    fn the_opaque_triangle_entry_is_the_reserved_word() {
        let mut t = FunctionTableWords::new(2);
        assert!(t.set_opaque_triangle(1));
        assert_eq!(t.words(), [1, 2, 0, FunctionHandle::RESERVED_WORD]);
        assert!(!t.set_opaque_triangle(2), "past the size");
        assert_eq!(t.words(), [1, 2, 0, 1]);
    }
}
