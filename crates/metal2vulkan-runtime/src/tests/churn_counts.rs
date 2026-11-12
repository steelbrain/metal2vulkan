//! The per-thread counts of the runtime's Vulkan makings (`churn_counts.rs`), checked across a
//! thread the test starts: the source scan of `exit_guard`'s test skips this directory alone.

use crate::churn_counts::{count_on_this_thread, vk_made, ChurnCounters, ObjectChurn, VkMaking};
use std::panic::Location;

/// A thread's makings are counted for it alone: another thread's, such as a block the ready
/// thread makes ahead while a call runs, move the process's counts and not the call's
/// thread's; and a making counted on this thread for a call that waited for it moves this
/// thread's alone, the process having counted it where it was made.
#[test]
fn a_thread_s_makings_are_its_own_and_those_it_waited_for() {
    let snapshot = || ChurnCounters::default().snapshot().0;
    let process = |c: &ObjectChurn| c.vk_memory_allocations;
    let mine = |c: &ObjectChurn| {
        (
            c.vk_memory_allocations_this_thread,
            c.vk_buffers_made_this_thread,
            c.vk_buffer_bindings_this_thread,
        )
    };
    let start = snapshot();
    std::thread::spawn(|| {
        vk_made(VkMaking::Memory, Location::caller());
        vk_made(VkMaking::Buffer, Location::caller());
    })
    .join()
    .unwrap();
    let other = snapshot();
    assert_eq!(process(&other), process(&start) + 1, "the process's count");
    assert_eq!(mine(&other), mine(&start), "another thread's making");

    vk_made(VkMaking::Memory, Location::caller());
    let own = snapshot();
    assert_eq!(process(&own), process(&other) + 1);
    assert_eq!(mine(&own).0, mine(&other).0 + 1, "this thread's making");

    let mut waited = [0; 6];
    waited[VkMaking::Memory as usize] = 1;
    waited[VkMaking::Buffer as usize] = 1;
    waited[VkMaking::BufferBinding as usize] = 1;
    count_on_this_thread(waited);
    let after = snapshot();
    assert_eq!(process(&after), process(&own), "counted where it was made");
    assert_eq!(
        mine(&after),
        (mine(&own).0 + 1, mine(&own).1 + 1, mine(&own).2 + 1),
        "a making a call of this thread waited for"
    );
}
