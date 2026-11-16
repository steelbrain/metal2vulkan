//! The recycler's bounds (`recycle.rs`), on its book alone: no device, no memory.
//!
//! - **A command buffer's chunks park whole.** 256 buffers of one making, the dedicated 64 KiB
//!   upload chunks of a command buffer whose 256 dispatches each took a written slot's neutral
//!   region, park as it completes, and the next command buffer's 256 takes find every one of
//!   them: a count bound of 64 destroyed 192 of them, which the next command buffer made again.
//!   The bounds park up to their bytes of such chunks, 1024, and one more goes.
//! - **At most 64 hold an allocation of their own.** Past 64 parked buffers that each hold a
//!   `VkDeviceMemory` of their own (`Placement::holds_allocation`), the earliest parked of them
//!   goes, and none of the buffers in shared blocks parked beside them: a parked buffer of a
//!   dedicated block or an import counts against the device's `maxMemoryAllocationCount`
//!   (`allocations.rs`), and the count bound of shared ones does not add to that.

use crate::memory::Placement;
use crate::recycle::{Recycler, BUFFERS};
use ash::vk::{self, Handle};
use std::time::Instant;

/// A dedicated upload chunk's bytes (`upload.rs`).
const CHUNK: u64 = 64 << 10;

/// A placement of a chunk's bytes, in a block of its own where `alone` says.
fn placement(alone: bool) -> Placement {
    let placement = Placement::caller(vk::DeviceMemory::null(), 0, CHUNK, 0, None);
    match alone {
        true => placement.holding_allocation(),
        false => placement,
    }
}

/// Buffers `from..to` (their raw handles), made live of `making` and parked as their last holder
/// went, each holding an allocation of its own where `alone` says; the buffers that went to keep
/// the bounds.
fn park(
    recycler: &mut Recycler<vk::Buffer, u32>,
    (from, to): (u64, u64),
    making: u32,
    alone: bool,
    now: Instant,
) -> Vec<vk::Buffer> {
    let mut gone = Vec::new();
    for raw in (from..to).map(vk::Buffer::from_raw) {
        recycler.made(raw, making);
        let went = recycler.park(raw, placement(alone), now);
        gone.extend(
            went.expect("a live buffer parks")
                .into_iter()
                .map(|(raw, _)| raw),
        );
    }
    gone
}

/// How many takes of `making` find a parked buffer, out of `tries`.
fn taken(recycler: &mut Recycler<vk::Buffer, u32>, making: u32, tries: u64, now: Instant) -> u64 {
    let found = (0..tries).filter(|_| {
        let (taken, gone) = recycler.take(&making, now);
        assert!(
            gone.is_empty(),
            "a take within the bounds' time let nothing go"
        );
        taken.is_some()
    });
    found.count() as u64
}

#[test]
fn a_command_buffers_chunks_park_whole() {
    let now = Instant::now();
    let mut recycler = Recycler::new(BUFFERS);
    let gone = park(&mut recycler, (1, 257), 0, false, now);
    assert!(
        gone.is_empty(),
        "256 chunks of a command buffer parked, and {} went to keep the bounds",
        gone.len()
    );
    assert_eq!(
        taken(&mut recycler, 0, 256, now),
        256,
        "the next command buffer's 256 chunks each took a parked one"
    );
    let gone = park(&mut recycler, (1, 1025), 0, false, now);
    assert!(
        gone.is_empty(),
        "1024 chunks, the bounds' bytes of them, parked, and {} went",
        gone.len()
    );
    let gone = park(&mut recycler, (1025, 1026), 0, false, now);
    assert_eq!(
        gone,
        [vk::Buffer::from_raw(1)],
        "one chunk past the bounds' 1024: the earliest parked goes"
    );
}

#[test]
fn at_most_64_parked_buffers_hold_an_allocation_of_their_own() {
    let now = Instant::now();
    let mut recycler = Recycler::new(BUFFERS);
    let gone = park(&mut recycler, (1, 101), 0, false, now);
    assert!(gone.is_empty(), "100 buffers of shared blocks parked");
    let gone = park(&mut recycler, (101, 165), 1, true, now);
    assert!(
        gone.is_empty(),
        "64 buffers that hold an allocation of their own parked beside them"
    );
    let gone = park(&mut recycler, (165, 167), 1, true, now);
    assert_eq!(
        gone,
        [vk::Buffer::from_raw(101), vk::Buffer::from_raw(102)],
        "two more that hold an allocation of their own: the two earliest parked such go, and no \
         buffer of a shared block"
    );
    assert_eq!(
        taken(&mut recycler, 1, 66, now),
        64,
        "64 parked buffers hold an allocation of their own"
    );
    assert_eq!(
        taken(&mut recycler, 0, 100, now),
        100,
        "every buffer of a shared block stayed parked"
    );
}
