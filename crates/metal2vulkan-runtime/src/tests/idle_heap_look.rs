//! The idle sweep's looks at glibc's heap (`idle_sweep.rs`'s `HeapLook`): a trim at the second
//! look in a row that finds more than the floor (`compile_memory::FREE_FLOOR`) free above the
//! least free a look read since the last trim, and none for the free heap a trim left, which
//! glibc still counts free.

use crate::idle_sweep::HeapLook;
use crate::pipeline::compile_memory::FREE_FLOOR;

const MIB: usize = 1 << 20;

/// Whether each look of one sweep's thread at `free` bytes free, one after another, trims.
fn looks(free: &[usize]) -> Vec<bool> {
    let mut heap = HeapLook::default();
    free.iter().map(|&free| heap.look(free)).collect()
}

#[test]
fn free_heap_past_the_floor_is_trimmed_at_the_second_look() {
    let past = FREE_FLOOR + MIB;
    // Trimmed at the second look, and not again while it stays as the trim left it.
    assert_eq!(looks(&[past; 5]), [false, true, false, false, false]);
    // The floor itself is kept.
    assert_eq!(looks(&[FREE_FLOOR; 4]), [false; 4]);
    // Past the floor at no two looks in a row: allocated again in between.
    assert_eq!(looks(&[past, 0, past, 0]), [false; 4]);
}

#[test]
fn free_heap_freed_after_a_trim_is_trimmed_again() {
    let left = 1024 * MIB;
    // Allocated from the chunks the trim handed back, then freed again: trimmed again.
    assert_eq!(
        looks(&[left, left, MIB, left, left]),
        [false, true, false, false, true]
    );
    // The heap grew past what the trim left free, then let go of it: trimmed again.
    let more = left + FREE_FLOOR + MIB;
    assert_eq!(looks(&[left, left, more, more]), [false, true, false, true]);
    // Grew by the floor or less: kept.
    let floor_more = left + FREE_FLOOR;
    assert_eq!(
        looks(&[left, left, floor_more, floor_more]),
        [false, true, false, false]
    );
}
