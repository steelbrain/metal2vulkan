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
/// takes the ranges of the buffers whose owners its command buffer may use, and of those no owner
/// holds, by buffer in the order each was first noted, a buffer's own in theirs, and leaves the
/// rest noted for the commit of a command buffer that uses them; a range a noted one of the same
/// buffer holds is noted once; and a buffer gone goes with its ranges, where its key is noted
/// again, or at the sweep once the buffers noted pass their bound, so ranges of buffers no commit
/// uses do not pile up. Buffers here are named by number, each its own owner but buffer 9, which
/// no owner holds.
#[test]
fn a_commit_takes_the_noted_ranges_of_the_buffers_its_command_buffer_uses() {
    use crate::managed::{MayUse, Noted};
    use crate::ObjectId;
    use std::collections::{BTreeMap, BTreeSet};
    let owners: BTreeMap<u32, ObjectId> = (1..=3).map(|of| (of, ObjectId::new())).collect();
    let gone: std::cell::RefCell<BTreeSet<u32>> = Default::default();
    let mut noted: Noted<u32, u32> = Noted::default();
    let note = |noted: &mut Noted<u32, u32>, of: u32, range: std::ops::Range<u64>| {
        let owner = owners.get(&of).copied();
        noted.note((of, owner), || of, range, |of| gone.borrow().contains(of));
    };
    let of =
        |buffers: &[u32]| -> MayUse { MayUse::Of(buffers.iter().map(|of| owners[of]).collect()) };
    note(&mut noted, 1, 0..64);
    note(&mut noted, 2, 0..16);
    note(&mut noted, 1, 128..256);
    // Held by the first range of buffer 1: noted once. Buffer 2's own range is not buffer 1's.
    note(&mut noted, 1, 16..32);
    note(&mut noted, 3, 0..8);
    note(&mut noted, 9, 0..4);
    assert_eq!(noted.owners(), 3);
    assert!(noted.owns(owners[&2]) && !noted.owns(ObjectId::new()));
    // A commit of a command buffer that uses buffer 1 alone takes buffer 1's ranges, and buffer
    // 9's, which no owner holds.
    let taken = noted.take(&of(&[1]));
    assert_eq!(taken, [(1, 0..64), (1, 128..256), (9, 0..4)]);
    assert!(!noted.owns(owners[&1]));
    // A range noted after it, of buffer 2, goes with buffer 2's first.
    note(&mut noted, 2, 32..48);
    // One that uses nothing takes nothing.
    assert!(noted.take(&of(&[])).is_empty());
    // One that uses buffers 2 and 3 takes both of 2's ranges, in their order, then 3's.
    let taken = noted.take(&of(&[3, 2]));
    assert_eq!(taken, [(2, 0..16), (2, 32..48), (3, 0..8)]);
    assert!(noted.take(&MayUse::All).is_empty());
    // A buffer gone, noted again under its key, starts anew.
    note(&mut noted, 1, 0..8);
    gone.borrow_mut().insert(1);
    note(&mut noted, 1, 8..16);
    assert_eq!(noted.take(&MayUse::All), [(1, 8..16)]);
    // Buffers gone that no commit took are dropped once the buffers noted pass the sweep's
    // bound: noted again past it, only the live are left.
    gone.borrow_mut().clear();
    let many: u32 = 1000;
    for of in 100..100 + many {
        note(&mut noted, of, 0..4);
    }
    gone.borrow_mut().extend(100..100 + many);
    for of in 2000..2000 + 2 * many {
        note(&mut noted, of, 0..4);
    }
    let left = noted.take(&MayUse::All);
    assert_eq!(
        left.len(),
        2 * many as usize,
        "the gone buffers' ranges were dropped"
    );
    assert!(left.iter().all(|(of, _)| *of >= 2000));
}
