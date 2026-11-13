//! Managed storage (`managed.rs`): its memory class.

use crate::memory::{candidates, MemoryClass};
use ash::vk;

fn ty(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
    vk::MemoryType {
        property_flags: flags,
        heap_index: 0,
    }
}

#[test]
fn managed_memory_is_host_coherent_and_cached_first() {
    use vk::MemoryPropertyFlags as P;
    let types = [
        ty(P::DEVICE_LOCAL),
        ty(P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_CACHED),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT),
        ty(P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED),
    ];
    // Cached coherent first; never non-coherent memory, which would need flushes; never
    // host-invisible memory.
    assert_eq!(
        candidates(&types, 0b11111, MemoryClass::Managed, false),
        [4, 1, 3]
    );
    // Without cached coherent memory, any host-visible coherent type.
    assert_eq!(
        candidates(&types, 0b01111, MemoryClass::Managed, false),
        [1, 3]
    );
}

/// The take rule of the ranges `did_modify_range` notes (`managed.rs`'s `Modified`): a commit
/// takes the ranges of the buffers its command buffer may use, in their order, and leaves the
/// rest noted in theirs, before any noted later, for the commit of a command buffer that uses
/// them; a range a noted one of the same buffer holds is noted once, and a buffer gone takes its
/// ranges with it. Buffers here are named by number.
#[test]
fn a_commit_takes_the_noted_ranges_of_the_buffers_its_command_buffer_uses() {
    use crate::managed::Noted;
    let mut noted: Noted<u32> = Noted::default();
    let note = |noted: &mut Noted<u32>, of: u32, range: std::ops::Range<u64>| {
        noted.note(of, range, |noted| *noted == of);
    };
    note(&mut noted, 1, 0..64);
    note(&mut noted, 2, 0..16);
    note(&mut noted, 1, 128..256);
    // Held by the first range of buffer 1: noted once. Buffer 2's own range is not buffer 1's.
    note(&mut noted, 1, 16..32);
    note(&mut noted, 3, 0..8);
    // A commit of a command buffer that uses buffer 1 alone takes buffer 1's ranges.
    let taken = noted.take(|of| Some(*of == 1));
    assert_eq!(taken, [(1, 0..64), (1, 128..256)]);
    // A range noted after it, of buffer 2, comes after the ones left.
    note(&mut noted, 2, 32..48);
    // Buffer 3 went: its range goes; one that uses nothing takes nothing.
    let taken = noted.take(|of| (*of != 3).then_some(false));
    assert!(taken.is_empty());
    // One that uses buffer 2 takes both of its ranges, in their order.
    let taken = noted.take(|of| Some(*of == 2));
    assert_eq!(taken, [(2, 0..16), (2, 32..48)]);
    assert!(noted.take(|_| Some(true)).is_empty());
}
