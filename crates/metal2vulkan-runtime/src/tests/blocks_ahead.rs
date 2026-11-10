//! A pool the runtime's encodings and commits place their own resources in gets its blocks made
//! ahead, off the program's thread, while it grows or is near its next block (`memory/ready.rs`):
//! from its first growth on, each creation call that needs a new block there finds one made ahead
//! and makes none itself, so no `vkAllocateMemory` of a block runs on the program's thread, where
//! on RADV and on NVIDIA's driver it held the device's submissions beside it.
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
use crate::memory::{Holds, BLOCK_BYTES, MAKING_BOUND, MIN_PLAIN_BLOCK_BYTES};
use crate::{Buffer, Device, Error, ResourceOptions, Texture, TextureDescriptor, WorkCounts};
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

/// The side of the small-blocks runs' `RGBA8Unorm` textures: 4 MiB, four to a block of 16 MiB.
const TEXTURE_SIDE: u64 = 1024;

/// The large-blocks run's textures larger than a block of 16 MiB: 20 MiB, each in a block of
/// 32 MiB ([`LARGE_EVERY`] textures one of them), and how many it makes; then the textures of
/// [`TEXTURE_SIDE`] it makes back to back, 96 MiB, more than its own blocks made ahead hold.
const LARGE_SIDES: (u64, u64) = (2560, 2048);
const BURST: usize = 24;
const LARGE_EVERY: usize = 4;
const LARGE_TEXTURES: usize = 4;

/// The most textures the small-blocks run makes, well past what `GROWTH` blocks hold.
const MOST_TEXTURES: usize = 64;

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

/// A device told of a pooled block's making that took four times `memory::MAKING_BOUND`, so that
/// it makes its pooled blocks of plain memory of the least size (`Memory::plain_block_made`).
fn small_blocks_device(scratch: &Scratch) -> Device {
    let device = device(&scratch.0);
    {
        let mut memory = crate::device::lock(&device.shared().memory);
        memory.plain_block_made(BLOCK_BYTES, MAKING_BOUND * 4);
        assert_eq!(memory.plain_block_bytes(), MIN_PLAIN_BLOCK_BYTES);
    }
    device
}

/// A `Private` `RGBA8Unorm` texture of `width` by `height` on `device`, sampled and rendered to,
/// made into `held`; then wait for the makings it started.
fn texture(device: &Device, (width, height): (u64, u64), held: &mut Vec<Texture>) {
    assert!(
        held.len() < MOST_TEXTURES,
        "the pool of images never grew as far"
    );
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    held.push(device.new_texture(&descriptor).unwrap());
    makings_ended(device);
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
fn a_growing_pool_s_new_blocks_its_first_growth_s_among_them_are_made_ahead_and_none_in_place() {
    let scratch = Scratch::new("growth");
    let device = device(&scratch.0);
    // The pools the runtime keeps room in are known before the first creation: those of blocks
    // of its own pages join them once the device's submission check answered
    // (`memory.rs`'s `Holds::PageBuffers`), their first block made then.
    let _settled = device.settle_submit_checks(Duration::from_secs(60));
    let mut held = Vec::new();
    // The first creation makes each pool's first block; the buffers' pool's first growth past it
    // is made ahead, the pool near its next block before it. Other pools' blocks are not counted.
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
    let first_in_place = before.buffer_blocks_made_in_place - start.buffer_blocks_made_in_place;
    let first_ahead = before.buffer_blocks_made_ahead - start.buffer_blocks_made_ahead;
    println!(
        "BLOCKS-AHEAD {}: {} buffers of {BUFFER_BYTES} bytes; over the buffers' pool's last {} \
         blocks, {ahead} made ahead, {in_place} in place, {waited_for} waited for; its first \
         growth made {first_ahead} ahead, {first_in_place} in place",
        device.name(),
        held.len(),
        grown(&before, &counts),
    );
    assert_eq!(
        first_in_place, 0,
        "the pool's first growth was made by a creation call itself"
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
    // Its first growth, made for the creation call that needs it, as nothing grew before it.
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

/// A pool whose blocks of plain memory are made small after a slow making
/// (`Memory::plain_block_made`) keeps as many of them made ahead as one block of
/// `memory::BLOCK_BYTES` holds, the bytes it kept ahead before they were made small, and no
/// creation call makes a block of the pool itself while one can be made ahead: where none is
/// made or under way, the call starts the pool's making and waits for its first block, the pool's
/// first block among them. With one small block ahead at a time, a guest's creation calls on a
/// Radeon 780M whose blocks were made of 16 MiB made 82 to 114 blocks of images themselves.
#[test]
fn a_pool_of_small_blocks_keeps_a_block_s_bytes_ahead_and_no_creation_call_makes_one() {
    let scratch = Scratch::new("small-blocks");
    let device = small_blocks_device(&scratch);
    let start = device.work_counts();
    let blocks = |counts: &WorkCounts| {
        (counts.image_blocks_made_ahead + counts.image_blocks_made_in_place)
            - (start.image_blocks_made_ahead + start.image_blocks_made_in_place)
    };
    let mut held: Vec<Texture> = Vec::new();
    let mut counts = start;
    // The most each pool of images kept ahead, by its memory type: the textures' pool, and the
    // pool host copies place in where the device copies color textures on the host and its type
    // differs (an Iris Xe's host-cached type), which keeps its own.
    let mut most_ahead = std::collections::BTreeMap::<u32, u64>::new();
    // One texture at a time, every making it started ended before the next: what this holds is
    // where each block is made and what is kept ahead, not how soon a making ends.
    while blocks(&counts) < GROWTH {
        texture(&device, (TEXTURE_SIDE, TEXTURE_SIDE), &mut held);
        counts = device.work_counts();
        let ahead = crate::device::lock(&device.shared().memory).ahead_bytes(Holds::Images);
        for (type_index, bytes) in ahead {
            let most = most_ahead.entry(type_index).or_default();
            *most = (*most).max(bytes);
        }
    }
    let host_copies = crate::device::lock(&device.shared().memory).host_copies_pool();
    let in_place = counts.image_blocks_made_in_place - start.image_blocks_made_in_place;
    let ahead = counts.image_blocks_made_ahead - start.image_blocks_made_ahead;
    println!(
        "BLOCKS-AHEAD {}: {} textures of 4 MiB in blocks of {} MiB; of the images' pool's {} \
         blocks, {ahead} made ahead, {in_place} in place, {} waited for; most kept ahead by \
         memory type {:?} MiB; host copies' pool {host_copies:?}",
        device.name(),
        held.len(),
        MIN_PLAIN_BLOCK_BYTES >> 20,
        blocks(&counts),
        counts.blocks_ahead_waited_for - start.blocks_ahead_waited_for,
        (most_ahead.iter())
            .map(|(&type_index, &bytes)| (type_index, bytes >> 20))
            .collect::<Vec<_>>(),
    );
    assert_eq!(in_place, 0, "a creation call made a block of images itself");
    assert!(
        !most_ahead.is_empty() && most_ahead.values().all(|&most| most == BLOCK_BYTES),
        "a pool of images kept other than a block's worth ahead: {most_ahead:?} bytes by memory \
         type"
    );
}

/// A pool of small blocks whose placements larger than its blocks, each taking a block of its
/// own size (`memory::pooled_block_bytes`), are recent keeps a block of that size made ahead
/// too, which the calls that need one of the pool's own blocks leave to them: the first such
/// placement waits for the ready thread's making of its block, each later one finds its block
/// made, and a burst of the pool's own blocks' textures that outruns their makings leaves it made;
/// no creation call makes a block itself. What the pool holds made is read before each of those
/// placements, not the device's count of waits, which counts other pools' blocks too.
#[test]
fn placements_past_a_pool_s_small_blocks_find_a_block_of_their_size_made_ahead() {
    let scratch = Scratch::new("large-blocks");
    let device = small_blocks_device(&scratch);
    let shared = device.shared();
    let large_block = 2 * MIN_PLAIN_BLOCK_BYTES;
    let made = || crate::device::lock(&shared.memory).ahead_made(Holds::Images);
    let start = device.work_counts();
    let mut held: Vec<Texture> = Vec::new();
    let mut missed = Vec::new();
    // One texture at a time, every making it started ended before the next.
    for at in 0..LARGE_EVERY * LARGE_TEXTURES {
        let large = at % LARGE_EVERY == LARGE_EVERY - 1;
        if large && at >= LARGE_EVERY && !made().contains(&large_block) {
            missed.push(at);
        }
        let sides = match large {
            true => LARGE_SIDES,
            false => (TEXTURE_SIDE, TEXTURE_SIDE),
        };
        texture(&device, sides, &mut held);
    }
    let before_burst = made();
    // Back to back with a use of the queues held open, so that a making of the pool's own blocks
    // waits for a pause the use never leaves unless a call hurries it: the calls that find none of
    // them made leave the block of 32 MiB made.
    let after_burst = {
        let _in_use = InUse::new(&shared.queue_quiet);
        for _ in 0..BURST {
            let mut descriptor = TextureDescriptor::texture_2d(
                PixelFormat::RGBA8Unorm,
                TEXTURE_SIDE,
                TEXTURE_SIDE,
                false,
            );
            descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::private();
            held.push(device.new_texture(&descriptor).unwrap());
        }
        made()
    };
    makings_ended(&device);
    let counts = device.work_counts();
    let in_place = counts.image_blocks_made_in_place - start.image_blocks_made_in_place;
    let sizes = crate::device::lock(&shared.memory).pooled_block_sizes(Holds::Images);
    let mib = |sizes: &[u64]| sizes.iter().map(|size| size >> 20).collect::<Vec<_>>();
    println!(
        "BLOCKS-AHEAD {}: {} textures, one of 20 MiB every {LARGE_EVERY} then {BURST} back to \
         back, in blocks of {} MiB; blocks of images made ahead {}, in place {in_place}; the 20 \
         MiB textures after the first found no block of 32 MiB made at {missed:?}; made ahead \
         before the burst {:?} MiB, after it {:?} MiB; the pool's blocks {:?} MiB",
        device.name(),
        held.len(),
        MIN_PLAIN_BLOCK_BYTES >> 20,
        counts.image_blocks_made_ahead - start.image_blocks_made_ahead,
        mib(&before_burst),
        mib(&after_burst),
        mib(&sizes),
    );
    assert_eq!(in_place, 0, "a creation call made a block of images itself");
    assert_eq!(
        sizes.iter().filter(|&&size| size == large_block).count(),
        LARGE_TEXTURES,
        "the 20 MiB textures did not each take a block of 32 MiB"
    );
    assert!(
        missed.is_empty(),
        "20 MiB textures found no block of 32 MiB made ahead at {missed:?}"
    );
    assert!(
        after_burst.contains(&large_block),
        "the burst's calls took the block of 32 MiB made ahead: {:?} MiB",
        mib(&after_burst)
    );
}

/// The sizes of the mixed-sizes run's `RGBA8Unorm` textures, in turn: six of 4 MiB, one of
/// 14.1 MiB (2560 x 1440) and one of 31.6 MiB (3840 x 2160), as `tests::reclaim_submit_bench`'s
/// mixed arm makes them, and how many times it makes the eight.
const MIXED_SIDES: [(u64, u64); 8] = [
    (1024, 1024),
    (1024, 1024),
    (1024, 1024),
    (2560, 1440),
    (1024, 1024),
    (1024, 1024),
    (1024, 1024),
    (3840, 2160),
];
const MIXED_ROUNDS: usize = 4;

/// A pool of 64 MiB blocks whose placements take up to half a block each keeps its next block
/// made ahead of the call that needs it: after the pool's first block, no creation call waits for
/// a block of images, and none makes one. On a Radeon 780M, `tests::reclaim_submit_bench`'s mixed
/// arm waited for 8 of its 13 blocks of images in its quiet phase, each making started only as
/// the call needed it.
#[test]
fn mixed_sizes_find_their_blocks_made_ahead() {
    let scratch = Scratch::new("mixed");
    let device = device(&scratch.0);
    let shared = device.shared();
    let ahead = || crate::device::lock(&shared.memory).ahead_blocks();
    let mut held: Vec<Texture> = Vec::new();
    texture(&device, MIXED_SIDES[0], &mut held);
    let start = ahead();
    let mut waits = Vec::new();
    for at in 1..MIXED_SIDES.len() * MIXED_ROUNDS {
        let before = ahead().images_waited_for;
        texture(&device, MIXED_SIDES[at % MIXED_SIDES.len()], &mut held);
        let after = ahead();
        if after.images_waited_for > before {
            waits.push(at);
        }
        println!(
            "BLOCKS-AHEAD-MIXED {}: texture {at} of {:?}: images made ahead {}, waited for {}, \
             in place {}; made ahead now {:?} MiB",
            device.name(),
            MIXED_SIDES[at % MIXED_SIDES.len()],
            after.images_ahead - start.images_ahead,
            after.images_waited_for - start.images_waited_for,
            after.images_in_place - start.images_in_place,
            (crate::device::lock(&shared.memory)
                .ahead_made(Holds::Images)
                .iter())
            .map(|size| size >> 20)
            .collect::<Vec<_>>(),
        );
    }
    let end = ahead();
    let blocks =
        (end.images_ahead + end.images_in_place) - (start.images_ahead + start.images_in_place);
    println!(
        "BLOCKS-AHEAD {}: {} textures of mixed sizes in blocks of {} MiB; after the first, {blocks} \
         blocks of images, {} made ahead, {} waited for (at textures {waits:?}), {} in place",
        device.name(),
        held.len(),
        crate::device::lock(&shared.memory).plain_block_bytes() >> 20,
        end.images_ahead - start.images_ahead,
        end.images_waited_for - start.images_waited_for,
        end.images_in_place - start.images_in_place,
    );
    assert_eq!(
        end.images_in_place, start.images_in_place,
        "a creation call made a block of images itself"
    );
    assert!(
        waits.is_empty(),
        "creation calls waited for a block of images at textures {waits:?}"
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
    // The buffers' pool's first growth, made for the creation call that needs it; the next
    // creation starts its next block's making, which waits for a pause the held use never leaves.
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
