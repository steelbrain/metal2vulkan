//! A pool the runtime's encodings and commits place their own resources in gets its blocks made
//! ahead, off the program's thread, while it grows (`memory/ready.rs`): once the pool grew, each
//! creation call that needs a new block there finds one made ahead and makes none itself, so no
//! `vkAllocateMemory` of a block runs on the program's thread, where on RADV and on NVIDIA's
//! driver it held the device's submissions beside it.
//!
//! The run makes `Private` buffers of a guest's scroll, eight to a block, and waits for every
//! making to end between two creations: what it holds is where the blocks are made, not how soon
//! a making ends.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::freed_memory::device;
use crate::{Buffer, Device, ResourceOptions, WorkCounts};
use std::time::{Duration, Instant};

/// A guest's `Private` buffer, as a scroll's memory log placed them: eight to a pooled block.
const BUFFER_BYTES: u64 = 7_471_104;

/// The blocks the run grows the pool by once it grew the first time.
const GROWTH: u64 = 6;

/// The most buffers the run makes, well past what `GROWTH` blocks hold.
const MOST: usize = 400;

/// A scratch directory for a device's shader cache, removed when dropped.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let name = format!("m2v-blocks-ahead-{case}-{}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Wait for every block's making under way on `device` to end.
fn makings_ended(device: &Device) {
    let at = Instant::now();
    while !crate::device::lock(&device.shared().memory).readying_ended() {
        assert!(
            at.elapsed() < Duration::from_secs(30),
            "a making never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Make a buffer on `device` into `held`, then wait for the makings it started: the counts after.
fn make(device: &Device, held: &mut Vec<Buffer>) -> WorkCounts {
    assert!(held.len() < MOST, "the pool never grew as far");
    let buffer = device.new_buffer(BUFFER_BYTES, ResourceOptions::private());
    held.push(buffer.unwrap());
    makings_ended(device);
    device.work_counts()
}

#[test]
fn a_growing_pool_s_new_blocks_are_made_ahead_and_none_in_place() {
    let scratch = Scratch::new("growth");
    let device = device(&scratch.0);
    let mut held = Vec::new();
    // The first creation makes each pool's first block; the pool's first growth past it is made
    // in place, as nothing grew before it.
    let start = make(&device, &mut held);
    let mut counts = start;
    while counts.pooled_memory_blocks_made == start.pooled_memory_blocks_made {
        counts = make(&device, &mut held);
    }
    let before = counts;
    while counts.pooled_memory_blocks_made - before.pooled_memory_blocks_made < GROWTH {
        counts = make(&device, &mut held);
    }
    let in_place = counts.buffer_blocks_made_in_place - before.buffer_blocks_made_in_place;
    let ahead = counts.buffer_blocks_made_ahead - before.buffer_blocks_made_ahead;
    let waited_for = counts.blocks_ahead_waited_for - before.blocks_ahead_waited_for;
    println!(
        "BLOCKS-AHEAD {}: {} buffers of {BUFFER_BYTES} bytes; over the last {} blocks made, \
         {ahead} made ahead, {in_place} in place, {waited_for} waited for; the first growth made \
         {} in place",
        device.name(),
        held.len(),
        counts.pooled_memory_blocks_made - before.pooled_memory_blocks_made,
        before.buffer_blocks_made_in_place - start.buffer_blocks_made_in_place,
    );
    assert_eq!(
        in_place, 0,
        "a creation call made a block of buffers itself"
    );
    assert!(
        ahead >= GROWTH,
        "only {ahead} of the blocks were made ahead"
    );
    assert_eq!(waited_for, 0, "a call waited for a making that had ended");
}
