//! Visible-function tables: the handles of the functions a pipeline stage can place in a table, and
//! the words a table's buffer holds, whose layout canonicalization reads
//! (`air::canon::function_table`, which defines the word indices).
//!
//! A `visible_function_table<T>` is bound at a Metal buffer index (`setVisibleFunctionTable:
//! atBufferIndex:`). The translation reads it as a read-only buffer of little-endian `u32`
//! words at that index:
//!
//! | word | meaning |
//! |---|---|
//! | 0 | 1 for a bound table; 0 in a zero-filled neutral buffer, so an unbound table is null (`is_null_visible_function_table`) |
//! | 1 | the table's size (`MTLVisibleFunctionTableDescriptor.functionCount`, `size()`) |
//! | 2 + i | entry `i`: [`FunctionHandle::NULL_WORD`] for a null entry, else a [`FunctionHandle`]'s word |
//!
//! A call through entry `i` compiles to a switch on that word over the functions the stage links
//! in `functions` and `binary_functions` (Metal's `functionHandleWithFunction:` accepts exactly
//! those): each case calls one of them, inlined. A null entry, a word no linked function has, or
//! a function of another type than the call's reaches the switch's default, which calls nothing
//! and yields zero. Calling a function through a pointer to another function type is undefined
//! in C++ (C++14 §5.2.10/6), which MSL builds on; that a call through a null entry is undefined
//! in Metal too is inferred (a null pointer designates no function to call), not stated by the
//! MSL specification or measured. An index `i` at or past the table's size is likewise taken as
//! undefined (inferred: Metal documents no bound check): the translation reads word `2 + i`
//! whatever it is, so what it reads past the table's words is the buffer's (robust buffer access
//! decides it). The consumer encodes a table's words with `bind::table::FunctionTableWords`.

use crate::pipeline::compile::key::LinkedKey;
use crate::pipeline::descriptor::{LinkedFunction, LinkedFunctions, LinkedList};

/// The handle of a function a pipeline stage can place in a visible-function table
/// (`MTLFunctionHandle`): the word a table entry holding it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionHandle(u32);

impl FunctionHandle {
    /// The word of a null entry.
    pub const NULL_WORD: u32 = 0;
    /// A word no function handle has: Metal reserves a function pointer's value 1 (the opaque
    /// triangle intersection function of an intersection function table), and the translation
    /// keeps a function pointer's integer value equal to its word.
    pub const RESERVED_WORD: u32 = 1;
    /// The word of the first function.
    const FIRST: u32 = crate::air::canon::function_table::FIRST_FUNCTION_WORD;

    /// The handle of the `n`-th callable function of a stage.
    fn nth(n: usize) -> Option<FunctionHandle> {
        let n = u32::try_from(n).ok()?;
        n.checked_add(Self::FIRST).map(FunctionHandle)
    }

    /// The word a table entry holding this function has.
    pub fn word(self) -> u32 {
        self.0
    }
}

/// Whether a function of `list` can be placed in a table (`functions`, `binary_functions`).
fn callable(list: LinkedList<'_>) -> bool {
    matches!(list, LinkedList::Functions | LinkedList::BinaryFunctions)
}

/// The handle of each of `keys` (a stage's linked functions in field order,
/// [`LinkedFunctions`]): the callable functions (`functions`, then `binary_functions`) are
/// numbered in order, one function (its AIR hash, name, specialized name, specialization, and
/// constants) listed twice having one handle. A function no callable list has has none.
pub(crate) fn handles(keys: &[LinkedKey<'_>]) -> Vec<Option<FunctionHandle>> {
    let mut callable_keys: Vec<&LinkedKey<'_>> = Vec::new();
    for k in keys.iter().filter(|k| callable(k.list)) {
        if !callable_keys.iter().any(|c| c.same_function(k)) {
            callable_keys.push(k);
        }
    }
    keys.iter()
        .map(|k| {
            let at = callable_keys.iter().position(|c| c.same_function(k))?;
            FunctionHandle::nth(at)
        })
        .collect()
}

impl LinkedFunctions {
    /// The handle of `f` in a table of a stage that links these functions
    /// (`functionHandleWithFunction:`), or `None` when neither `functions` nor
    /// `binary_functions` lists it (a function only in `private_functions` or a group cannot be
    /// placed in a table). `f` is the listed function made any way (`LinkedKey::same_definition`):
    /// by name, by descriptor, or compiled to a binary.
    pub fn function_handle(&self, f: &LinkedFunction) -> Option<FunctionHandle> {
        let keys = crate::pipeline::compile::linked_keys(self);
        let wanted = crate::pipeline::compile::linked_key(LinkedList::Functions, f);
        let at = keys
            .iter()
            .position(|k| callable(k.list) && k.same_definition(&wanted))?;
        handles(&keys).get(at).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind::FunctionTableWords;
    use crate::library::FunctionConstantValues;

    fn key<'a>(
        list: LinkedList<'a>,
        name: &'a str,
        hash: u8,
        constants: &'a FunctionConstantValues,
    ) -> LinkedKey<'a> {
        LinkedKey {
            list,
            name,
            specialized_name: None,
            specialized: false,
            air_hash: [hash; 32],
            constants,
        }
    }

    /// Handles number the functions a table can hold (`functions`, then `binary_functions`),
    /// one function listed twice has one handle, and a function only in `private_functions` or
    /// a group has none. The first handle's word is 2: 0 is a null entry and 1 is reserved.
    #[test]
    fn handles_number_the_callable_functions() {
        let none = FunctionConstantValues::defaults();
        let keys = [
            key(LinkedList::PrivateFunctions, "shared", 1, &none),
            key(LinkedList::Functions, "a", 2, &none),
            key(LinkedList::Group("g"), "grouped", 3, &none),
            key(LinkedList::BinaryFunctions, "b", 4, &none),
            key(LinkedList::Functions, "shared", 1, &none),
            key(LinkedList::BinaryFunctions, "a", 2, &none),
            // Another function of the name `a` (another module).
            key(LinkedList::Functions, "a", 5, &none),
        ];
        let words: Vec<Option<u32>> = handles(&keys)
            .into_iter()
            .map(|h| h.map(FunctionHandle::word))
            .collect();
        assert_eq!(
            words,
            [Some(4), Some(2), None, Some(3), Some(4), Some(2), Some(5)]
        );
        // `shared`'s handle in entry 1 of a two-entry table.
        let mut t = FunctionTableWords::new(2);
        let shared = handles(&keys).into_iter().flatten().next();
        assert!(t.set(1, shared));
        assert_eq!(t.words(), [1, 2, 0, 4]);
    }
}
