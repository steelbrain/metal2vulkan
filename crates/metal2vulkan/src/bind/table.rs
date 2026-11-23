//! The words of a visible-function table's buffer.
//!
//! A `visible_function_table` argument compiles to a read-only buffer at its Metal buffer index
//! (`pipeline::table` has the word layout): `setVisibleFunctionTable:atBufferIndex:` binds a
//! buffer the consumer owns for the `MTLVisibleFunctionTable` and fills with the table's words,
//! [`function_table_header`] and then one word per entry at [`function_table_entry_offset`].
//! The shader reads its words through the buffer slot only, never through the guest address, so
//! the bound buffer's `gpu_address` is not read for it. A slot bound to no table reads zeros,
//! whose word 0 makes `is_null_visible_function_table` true. `setFunction:atIndex:` writes one
//! entry: the word of the function's `FunctionHandle` (`LinkedFunctions::function_handle` of the
//! stage the table belongs to), or `FunctionHandle::NULL_WORD` for a null entry.

use crate::air::canon::function_table::{FIRST_ENTRY_WORD, PRESENT_WORD, SIZE_WORD};
#[cfg(test)]
use crate::pipeline::table::FunctionHandle;

/// The words a visible-function table's buffer holds before its entries, for a table of `size`
/// entries (`MTLVisibleFunctionTableDescriptor.functionCount`): 1, which marks a bound table, then
/// `size`. Entry `i` is the word at [`function_table_entry_offset`]`(i)`: the
/// [`FunctionHandle::word`](crate::pipeline::FunctionHandle::word) of the function it calls, or
/// [`FunctionHandle::NULL_WORD`](crate::pipeline::FunctionHandle::NULL_WORD) for a null entry.
/// Every word is a little-endian `u32`.
pub fn function_table_header(size: u32) -> [u32; 2] {
    const _: () = assert!(
        FIRST_ENTRY_WORD == 2,
        "the header is the words before entry 0"
    );
    let mut words = [0; 2];
    words[PRESENT_WORD as usize] = 1;
    words[SIZE_WORD as usize] = size;
    words
}

/// The offset in bytes of entry `index` in a visible-function table's buffer, after its
/// [`function_table_header`]. The buffer of a table of `size` entries is
/// `function_table_entry_offset(size)` bytes long.
pub fn function_table_entry_offset(index: u32) -> u64 {
    (u64::from(FIRST_ENTRY_WORD) + u64::from(index)) * 4
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
        let mut words = function_table_header(size).to_vec();
        words.resize(
            FIRST_ENTRY_WORD as usize + size as usize,
            FunctionHandle::NULL_WORD,
        );
        FunctionTableWords { words }
    }

    /// The table's size.
    pub(crate) fn size(&self) -> u32 {
        self.words.get(SIZE_WORD as usize).copied().unwrap_or(0)
    }

    /// Set entry `index` to `function`, or to null (`setFunction:atIndex:`). An index past the
    /// table's size is refused (`false`, nothing written), as Metal's validation refuses it.
    pub(crate) fn set(&mut self, index: u32, function: Option<FunctionHandle>) -> bool {
        let word = function.map_or(FunctionHandle::NULL_WORD, FunctionHandle::word);
        self.set_word(index, word)
    }

    /// Set entry `index` to Metal's opaque triangle intersection function
    /// (`setOpaqueTriangleIntersectionFunctionWithSignature:atIndex:` of an intersection function
    /// table): [`FunctionHandle::RESERVED_WORD`], which no function handle has. An index past the
    /// table's size is refused (`false`, nothing written).
    pub(crate) fn set_opaque_triangle(&mut self, index: u32) -> bool {
        self.set_word(index, FunctionHandle::RESERVED_WORD)
    }

    /// Set entry `index` to `word`; `false`, nothing written, past the table's size.
    fn set_word(&mut self, index: u32, word: u32) -> bool {
        if index >= self.size() {
            return false;
        }
        let at = FIRST_ENTRY_WORD as usize + index as usize;
        match self.words.get_mut(at) {
            Some(w) => {
                *w = word;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table's buffer is a bound marker, its size, and one word per entry; a null entry is 0.
    #[test]
    fn a_table_is_its_marker_size_and_entries() {
        let mut t = FunctionTableWords::new(3);
        assert_eq!(t.words(), [1, 3, 0, 0, 0]);
        assert_eq!(function_table_header(3), t.words()[..2]);
        assert_eq!(function_table_entry_offset(1), 12);
        assert_eq!(function_table_entry_offset(3), 20);
        assert_eq!(t.to_bytes().len() as u64, function_table_entry_offset(3));
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
