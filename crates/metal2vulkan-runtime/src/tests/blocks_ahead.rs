//! A pool the runtime's encodings and commits place their own resources in gets its blocks made
//! ahead, off the program's thread, while it grows (`memory/ready.rs`): once the pool grew, each
//! creation call that needs a new block there finds one made ahead and makes none itself, so no
//! `vkAllocateMemory` of a block runs on the program's thread, where on RADV and on NVIDIA's
//! driver it held the device's submissions beside it.
//!
//! The first run makes `Private` buffers of a guest's scroll, eight to a block, and waits for every
//! making to end between two creations: what it holds is where the blocks are made, not how soon
//! a making ends. The second makes them back to back beside two threads that keep the device's
//! queues in use with no pause of 1 ms, so a making waits out its whole wait for a burst's end
//! (`QUIET_MOST`) unless a call cuts it short: the creation call that needs the block while its
//! making still waits so returns well within that wait, the making allocating at once.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::freed_memory::device;
use crate::{Buffer, CommandBufferStatus, Device, ResourceOptions, WorkCounts};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How long a making waits for a burst of the device's submissions to end, at most
/// (`memory/ready.rs`'s `QUIET_MOST`).
const QUIET_MOST: Duration = Duration::from_millis(50);

/// The threads that keep the device's queues in use.
const STREAMS: usize = 3;

/// The makings a creation call may wait for before one of them is still in its wait for quiet.
const ATTEMPTS: u64 = 3;

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

/// Sets its flag as it goes, so the threads that keep the queues in use stop however the test
/// ends.
struct Stops<'a>(&'a AtomicBool);

impl Drop for Stops<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn a_creation_that_needs_a_block_whose_making_waits_for_quiet_waits_for_its_allocation_alone() {
    let scratch = Scratch::new("hurried");
    let device = device(&scratch.0);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let stops = Stops(&done);
        // Threads that each commit a one-word fill and wait for it, over and over: the device's
        // queues are in use with no pause of 1 ms between two uses.
        for _ in 0..STREAMS {
            scope.spawn(|| {
                let queue = device.new_command_queue().unwrap();
                let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
                while !done.load(Ordering::Acquire) {
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&word, 0..4, 1).unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer
                        .wait_until_completed_for(Duration::from_secs(60))
                        .unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                }
            });
        }
        let mut held = Vec::new();
        let new_buffer = |held: &mut Vec<Buffer>| {
            assert!(
                held.len() < MOST,
                "the pool never needed its block made ahead"
            );
            let made = Instant::now();
            let buffer = device.new_buffer(BUFFER_BYTES, ResourceOptions::private());
            held.push(buffer.unwrap());
            made.elapsed()
        };
        // The pool's first growth, made in place; the next creation starts its next block's
        // making, which waits for a pause the queues never make.
        new_buffer(&mut held);
        let start = device.work_counts();
        while device.work_counts().pooled_memory_blocks_made == start.pooled_memory_blocks_made {
            new_buffer(&mut held);
        }
        // Back to back until a creation call waited for a making its wait for quiet had not let
        // go: a pause the host's scheduling put between the queues' uses may end a making's wait
        // first, and the next making is started as the pool grows again.
        let before = device.work_counts();
        let mut longest = Duration::ZERO;
        loop {
            let counts = device.work_counts();
            let waited = counts.blocks_ahead_waited_for - before.blocks_ahead_waited_for;
            if counts.blocks_ahead_hurried > before.blocks_ahead_hurried || waited == ATTEMPTS {
                break;
            }
            longest = longest.max(new_buffer(&mut held));
        }
        let after = device.work_counts();
        drop(stops);
        let hurried = after.blocks_ahead_hurried - before.blocks_ahead_hurried;
        println!(
            "BLOCKS-AHEAD {}: beside queues kept in use, the creation calls after the pool's first \
             growth took {longest:?} at the longest; {hurried} making hurried, {} made ahead, {} \
             in place",
            device.name(),
            after.buffer_blocks_made_ahead - before.buffer_blocks_made_ahead,
            after.buffer_blocks_made_in_place - before.buffer_blocks_made_in_place,
        );
        assert_eq!(hurried, 1, "no making a call waited for was cut short");
        assert!(
            longest < QUIET_MOST,
            "a creation call waited {longest:?}, out to the making's wait for quiet"
        );
    });
}
