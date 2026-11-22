//! The words of a visible-function table's buffer, as the tests build them.
//!
//! A `visible_function_table` argument compiles to a read-only buffer at its Metal buffer index
//! (`pipeline::table` has the word layout): `setVisibleFunctionTable:atBufferIndex:` binds a
//! buffer the consumer owns for the `MTLVisibleFunctionTable` and fills with the table's words.
//! The shader reads its words through the buffer slot only, never through the guest address, so
//! the bound buffer's `gpu_address` is not read for it. A slot bound to no table reads zeros,
//! whose word 0 makes `is_null_visible_function_table` true. `setFunction:atIndex:` writes one
//! entry: the word of the function's `FunctionHandle` (`LinkedFunctions::function_handle` of the
//! stage the table belongs to), or `FunctionHandle::NULL_WORD` for a null entry.

#[cfg(test)]
use crate::air::canon::function_table::{FIRST_ENTRY_WORD, PRESENT_WORD, SIZE_WORD};
#[cfg(test)]
use crate::pipeline::table::FunctionHandle;

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
