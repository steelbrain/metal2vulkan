//! A pool of a small heap gets no block before a placement needs one (`memory.rs`'s
//! `Memory::small`): a device's command queue, whose creation keeps room in the pools the
//! encodings and commits place in (`memory::keep_room`), makes no block of such a heap, and the
//! first buffer placed there makes the pool's first block, of its heap's block size
//! (`memory::pooled_block_bytes`). On an RTX 2080 without Resizable BAR the `WriteCombined` pool,
//! of the 246 MiB device-local host-visible heap, took a 64 MiB block at each device's first
//! command queue, 67 MiB of BAR1 a device.
//!
//! Every heap is made small by a stand-in for the device's blocks (`Memory::small_heaps_stand_in`),
//! so that the test runs the rule on every device, whatever its heaps.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::Scratch;
use super::freed_memory::device;
use crate::device::lock;
use crate::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};

/// The pooled block of every heap under the stand-in: a small heap's, as an RTX 2080's 246 MiB
/// heap takes (`memory::pooled_block_bytes`).
const SMALL_BLOCK: u64 = 8 << 20;

#[test]
fn a_small_heap_s_first_block_is_made_by_a_placement() {
    let scratch = Scratch::new("small-heap");
    let device = device(&scratch.0);
    let memory = &device.shared().memory;
    let opened = lock(memory).pooled_by_pool();
    assert!(
        opened.is_empty(),
        "pooled blocks as the device is made: {opened:?}"
    );
    lock(memory).small_heaps_stand_in(SMALL_BLOCK);
    // The first command queue keeps room in the encodings' pools: none of a small heap.
    let _queue = device.new_command_queue().unwrap();
    let heaps = lock(memory).heaps_live();
    assert!(
        heaps.iter().all(|&(_, live)| live == 0),
        "live bytes of each heap (size, live) after the first command queue: {heaps:?}"
    );
    assert!(
        lock(memory).ahead_by_pool().is_empty(),
        "a block made ahead"
    );
    // A write-combined buffer's placement makes its pool's first block, of the heap's size.
    let write_combined = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::WriteCombined,
        HazardTrackingMode::Default,
    );
    let _buffer = device.new_buffer(4096, write_combined).unwrap();
    let pooled = lock(memory).pooled_by_pool();
    let [((_, type_index), sizes)] = pooled.as_slice() else {
        panic!("the pools that hold a block after a buffer: {pooled:?}");
    };
    let bytes = lock(memory).pooled_block_bytes_of(*type_index);
    assert_eq!(bytes, SMALL_BLOCK);
    assert_eq!(sizes, &[bytes], "the buffer's pool's blocks");
    let heaps = lock(memory).heaps_live();
    let live: u64 = heaps.iter().map(|&(_, live)| live).sum();
    assert_eq!(
        live, bytes,
        "live bytes of each heap (size, live): {heaps:?}"
    );
    assert!(
        lock(memory).ahead_by_pool().is_empty(),
        "a block made ahead"
    );
}
