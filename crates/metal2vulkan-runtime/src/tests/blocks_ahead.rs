//! A pool the runtime's encodings and commits place their own resources in gets its blocks made
//! ahead, off the program's thread, while it grows (`memory/ready.rs`): once the pool grew, each
//! creation call that needs a new block there finds one made ahead and makes none itself, so no
//! `vkAllocateMemory` of a block runs on the program's thread, where on RADV and on NVIDIA's
//! driver it held the device's submissions beside it.
//!
//! The first run makes `Private` buffers of a guest's scroll, eight to a block, and waits for every
//! making to end between two creations: what it holds is where the blocks are made, not how soon
//! a making ends. A run of exportable buffers does the same: their pool, which the creation calls
//! themselves place in (`memory::allocate_exported`), gets its blocks made ahead too. The second
//! makes them back to back while a use of the device's queues is held open, so a making waits out
//! its whole wait for a burst's end (`QUIET_MOST`) unless a call cuts it short: the creation call
//! that needs the block while its making still waits so returns well within that wait, the making
//! allocating at once. The use is held rather than made by threads that submit without pause,
//! because that wait reads the holds of the device's queues (`device/queue_quiet.rs`), not the
//! GPU's work: threads that commit and wait leave a pause of 1 ms wherever the host delays them,
//! which ended each making's wait before the call came on a loaded host.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::freed_memory::device;
use crate::device::queue_quiet::QueueQuiet;
use crate::{Buffer, Device, Error, ResourceOptions, TextureDescriptor, WorkCounts};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::{Duration, Instant};

/// How long a making waits for a burst of the device's submissions to end, at most
/// (`memory/ready.rs`'s `QUIET_MOST`).
const QUIET_MOST: Duration = Duration::from_millis(50);

/// The makings whose wait for quiet may reach its bound before a creation call needs their block:
/// with a use held open nothing else ends that wait, so each is a host that took longer than
/// `QUIET_MOST` over the few creation calls between a making's start and its block's need.
const ATTEMPTS: u64 = 3;

/// A guest's `Private` buffer, as a scroll's memory log placed them: eight to a pooled block.
const BUFFER_BYTES: u64 = 7_471_104;

/// The blocks the run grows the pool by once it grew the first time.
const GROWTH: u64 = 6;

/// The most buffers the run makes, well past what `GROWTH` blocks hold.
const MOST: usize = 400;

/// The creation calls that place nothing in the host copies' pool once its making ended.
const CREATIONS: usize = 20;

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

/// The blocks the buffers' pool grew by from `before` to `after`, made ahead or in place.
fn grown(before: &WorkCounts, after: &WorkCounts) -> u64 {
    (after.buffer_blocks_made_ahead + after.buffer_blocks_made_in_place)
        - (before.buffer_blocks_made_ahead + before.buffer_blocks_made_in_place)
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
    // The pools the runtime keeps room in are known before the first creation: those of blocks
    // of its own pages join them once the device's submission check answered
    // (`memory.rs`'s `Holds::PageBuffers`), their first block made then.
    let _settled = device.settle_submit_checks(Duration::from_secs(60));
    let mut held = Vec::new();
    // The first creation makes each pool's first block; the buffers' pool's first growth past it
    // is made in place, as nothing grew before it. Other pools' blocks are not counted.
    let start = make(&device, &mut held);
    let mut counts = start;
    while grown(&start, &counts) == 0 {
        counts = make(&device, &mut held);
    }
    let before = counts;
    while grown(&before, &counts) < GROWTH {
        counts = make(&device, &mut held);
    }
    let in_place = counts.buffer_blocks_made_in_place - before.buffer_blocks_made_in_place;
    let ahead = counts.buffer_blocks_made_ahead - before.buffer_blocks_made_ahead;
    let waited_for = counts.blocks_ahead_waited_for - before.blocks_ahead_waited_for;
    println!(
        "BLOCKS-AHEAD {}: {} buffers of {BUFFER_BYTES} bytes; over the buffers' pool's last {} \
         blocks, {ahead} made ahead, {in_place} in place, {waited_for} waited for; its first \
         growth made {} in place",
        device.name(),
        held.len(),
        grown(&before, &counts),
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

/// The pool of exportable buffers grows as the runtime's own pools do: once it grew, each new
/// block there is made ahead, and no creation call makes one itself (on a Radeon 780M such a
/// making took the creation call 10.0 to 15.0 ms: `tests::reclaim_submit_bench`).
#[test]
fn a_growing_pool_of_exportable_buffers_gets_its_new_blocks_made_ahead() {
    let scratch = Scratch::new("exportable");
    let device = device(&scratch.0);
    let mut held = Vec::new();
    // The counts after an exportable buffer made into `held` and the makings it started; `None`
    // where the device makes no exportable buffer.
    let make = |held: &mut Vec<Buffer>| {
        assert!(held.len() < MOST, "the pool never grew as far");
        match device.new_exportable_buffer(BUFFER_BYTES, ResourceOptions::private()) {
            Err(Error::NotImplemented(_)) => return None,
            made => held.push(made.unwrap()),
        }
        makings_ended(&device);
        Some(device.work_counts())
    };
    // Every pooled block made from here on is the exportable pool's: the first creation made each
    // other pool's first block.
    let Some(start) = make(&mut held) else {
        println!(
            "BLOCKS-AHEAD {}: makes no exportable buffer, so no such pool grows",
            device.name()
        );
        return;
    };
    let blocks =
        |counts: &WorkCounts| counts.pooled_memory_blocks_made - start.pooled_memory_blocks_made;
    let mut counts = start;
    // Its first growth, made in place, as nothing grew before it.
    while blocks(&counts) < 1 {
        counts = make(&mut held).unwrap();
        let dedicated = counts.dedicated_memory_blocks_live - start.dedicated_memory_blocks_live;
        if blocks(&counts) == 0 && dedicated >= 2 {
            println!(
                "BLOCKS-AHEAD {}: places each exportable buffer in a dedicated allocation, so no \
                 such pool grows",
                device.name()
            );
            return;
        }
    }
    let before = counts;
    while blocks(&counts) - blocks(&before) < GROWTH {
        counts = make(&mut held).unwrap();
    }
    let in_place = counts.buffer_blocks_made_in_place - before.buffer_blocks_made_in_place;
    let ahead = counts.buffer_blocks_made_ahead - before.buffer_blocks_made_ahead;
    println!(
        "BLOCKS-AHEAD {}: {} exportable buffers of {BUFFER_BYTES} bytes; over their pool's last {} \
         blocks, {ahead} made ahead, {in_place} in place",
        device.name(),
        held.len(),
        blocks(&counts) - blocks(&before),
    );
    assert_eq!(
        in_place, 0,
        "a creation call made a block of exportable buffers itself"
    );
    assert_eq!(
        ahead,
        blocks(&counts) - blocks(&before),
        "only {ahead} of the blocks were made ahead"
    );
}

/// A use of the device's queues held open while it lives (`device/queue_quiet.rs`): a making's
/// wait for a burst's end sees a use under way throughout, so no pause ends it.
struct InUse<'a>(&'a QueueQuiet);

impl<'a> InUse<'a> {
    fn new(quiet: &'a QueueQuiet) -> InUse<'a> {
        quiet.began();
        InUse(quiet)
    }
}

impl Drop for InUse<'_> {
    fn drop(&mut self) {
        self.0.ended();
    }
}

#[test]
fn a_creation_that_needs_a_block_whose_making_waits_for_quiet_waits_for_its_allocation_alone() {
    let scratch = Scratch::new("hurried");
    let device = device(&scratch.0);
    // The pools the runtime keeps room in are known before the first creation, so no creation
    // call below makes the first block of a pool of its own pages (as in the first test).
    let _settled = device.settle_submit_checks(Duration::from_secs(60));
    let shared = device.shared();
    let in_use = InUse::new(&shared.queue_quiet);
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
    // The buffers' pool's first growth, made in place; the next creation starts its next block's
    // making, which waits for a pause the held use never leaves.
    new_buffer(&mut held);
    let start = device.work_counts();
    while grown(&start, &device.work_counts()) == 0 {
        new_buffer(&mut held);
    }
    // Back to back until a creation call cut a making's wait for quiet short. Nothing else ends
    // that wait but its bound, which only a host slower than `QUIET_MOST` over a few creation
    // calls reaches; the next making is started as the pool grows again.
    let before = device.work_counts();
    let past = |counts: &WorkCounts| {
        counts.blocks_ahead_past_quiet_wait - before.blocks_ahead_past_quiet_wait
    };
    let mut longest = Duration::ZERO;
    loop {
        let counts = device.work_counts();
        if counts.blocks_ahead_hurried > before.blocks_ahead_hurried || past(&counts) == ATTEMPTS {
            break;
        }
        longest = longest.max(new_buffer(&mut held));
    }
    let after = device.work_counts();
    drop(in_use);
    let hurried = after.blocks_ahead_hurried - before.blocks_ahead_hurried;
    println!(
        "BLOCKS-AHEAD {}: with a use of the queues held open, the creation calls after the pool's \
         first growth took {longest:?} at the longest; {hurried} making hurried, {} past its wait \
         for quiet, {} waited for, {} made ahead, {} in place",
        device.name(),
        past(&after),
        after.blocks_ahead_waited_for - before.blocks_ahead_waited_for,
        after.buffer_blocks_made_ahead - before.buffer_blocks_made_ahead,
        after.buffer_blocks_made_in_place - before.buffer_blocks_made_in_place,
    );
    assert_eq!(hurried, 1, "no making a call waited for was cut short");
    assert!(
        longest < QUIET_MOST,
        "a creation call waited {longest:?}, out to the making's wait for quiet"
    );
}

/// The block made ahead for the pool host copies place in joins it at the host copy's placement
/// that needs it, never at a creation call that places nothing there, however long ago its making
/// ended: that pool's blocks change only at its placements.
#[test]
fn the_host_copies_block_made_ahead_joins_its_pool_at_its_placement_alone() {
    let scratch = Scratch::new("host-copies");
    let device = device(&scratch.0);
    if !device.copies_color_on_the_host() {
        println!(
            "BLOCKS-AHEAD {}: copies no color texture on the host, so no block is made ahead for \
             host copies",
            device.name()
        );
        return;
    }
    // The device's first creation call starts the host copies' making; every creation call after
    // its end finds the block made.
    let queues: Vec<_> = (0..CREATIONS)
        .map(|at| {
            let queue = device.new_command_queue().unwrap();
            if at == 0 {
                makings_ended(&device);
            }
            queue
        })
        .collect();
    let created = device.work_counts();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let placed = device.work_counts();
    println!(
        "BLOCKS-AHEAD {}: host copies' blocks made ahead and added after {CREATIONS} creation \
         calls {}, after a texture's placement {}; made in place {}",
        device.name(),
        created.host_copy_blocks_readied,
        placed.host_copy_blocks_readied,
        placed.host_copy_blocks_made_in_place,
    );
    assert_eq!(
        created.host_copy_blocks_readied, 0,
        "a creation call that placed nothing in the host copies' pool added its block made ahead"
    );
    assert_eq!(
        (
            placed.host_copy_blocks_readied,
            placed.host_copy_blocks_made_in_place
        ),
        (1, 0),
        "the texture's placement did not take the host copies' block made ahead"
    );
    drop((texture, queues));
}
