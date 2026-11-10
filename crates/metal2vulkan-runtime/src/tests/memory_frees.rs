//! Every `VkDeviceMemory` block the runtime's allocator frees, counted by what made it go
//! (`WorkCounts::memory_frees`, `memory.rs`'s `FreeCause`): a driver that unmaps a buffer
//! object's pages as it frees it charges that page-table update to the free, so what a workload
//! frees is what sets the cost and the cause says which of those frees a change could remove.
//!
//! A buffer with a block of its own keeps it for the next buffer of its size where its heap's
//! budget of empty blocks spares it (read once no block is being made ahead, below), and is
//! counted as a dedicated block where it does not; an
//! import of the caller's host memory, which is a dedicated block too, is counted apart as the
//! import it is; and the blocks counted are never more than the process's `vkFreeMemory` calls,
//! so no block is counted twice or counted that was never freed.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::freed_memory::{device, unused, Host, Runs};
use crate::work_counts::MemoryFrees;
use crate::{Device, ResourceOptions};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest the blocks made ahead may take to be made.
const WAIT: Duration = Duration::from_secs(60);

/// A scratch directory for a device's shader cache, removed when dropped.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(case: &str) -> Scratch {
        let name = format!("m2v-memory-frees-{case}-{}", std::process::id());
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

/// The bytes of a private buffer that takes a block of its own, past half a pooled block
/// (`memory.rs`'s `BLOCK_BYTES`), and that the recycler parks for no later buffer, past its bytes
/// (`recycle.rs`'s `BUFFERS`): its release frees its block there and then.
const OWN_BLOCK: u64 = 128 << 20;

/// Wait until no block is being made ahead for `device` (`memory/ready.rs`): the device's first
/// creation call starts makings on a thread of their own, each a `vkAllocateMemory` the device's
/// counts hold, and under load one landed between two readings of a test that counts its own
/// allocations (the Iris Xe, once in 22 runs of the runtime suite).
fn made_ahead(device: &Device) {
    let at = Instant::now();
    while !crate::device::lock(&device.shared().memory).readying_ended() {
        assert!(
            at.elapsed() < WAIT,
            "a block made ahead still making after {WAIT:?}"
        );
        std::thread::yield_now();
    }
}

/// What `device` has freed so far, by cause, and the process's `vkFreeMemory` calls.
fn frees(device: &Device) -> (MemoryFrees, u64) {
    let counts = device.work_counts();
    (counts.memory_frees, counts.object_churn.vk_memory_freed)
}

/// A buffer with a block of its own, released where its heap keeps no empty block (the budget's
/// test stand-in at 0), frees one block, counted as a dedicated block: one resource going away
/// with memory of its own.
#[test]
fn a_buffer_with_a_block_of_its_own_frees_a_dedicated_block() {
    let scratch = Scratch::new("dedicated");
    let device = device(&scratch.0);
    crate::device::lock(&device.shared().memory).kept_budget_stand_in = Some(0);
    let (before, _) = frees(&device);
    let buffer = device
        .new_buffer(OWN_BLOCK, ResourceOptions::private())
        .unwrap();
    assert_eq!(
        frees(&device).0.dedicated,
        before.dedicated,
        "a buffer made frees no block"
    );
    drop(buffer);
    let (after, total) = frees(&device);
    assert_eq!(
        after.dedicated,
        before.dedicated + 1,
        "the released buffer's block was freed as a dedicated block"
    );
    assert_eq!(
        after.blocks,
        before.blocks + 1,
        "exactly one block was freed, under one cause"
    );
    assert!(
        after.blocks <= total,
        "the allocator's blocks ({}) are no more than the process's vkFreeMemory calls ({total})",
        after.blocks,
    );
}

/// A buffer with a block of its own, released where its heap's budget spares the block, frees
/// none: the block is kept, and the next buffer of its size takes it with no allocation. Read once
/// the blocks the first creation call started making ahead are made ([`made_ahead`]).
#[test]
fn a_buffer_with_a_block_of_its_own_keeps_it_for_the_next_of_its_size() {
    let scratch = Scratch::new("kept");
    let device = device(&scratch.0);
    let buffer = device
        .new_buffer(OWN_BLOCK, ResourceOptions::private())
        .unwrap();
    made_ahead(&device);
    let (before, _) = frees(&device);
    let counts = device.work_counts();
    drop(buffer);
    let (after, _) = frees(&device);
    assert_eq!(
        after.blocks, before.blocks,
        "the released buffer's block was kept"
    );
    let kept = device.work_counts();
    assert_eq!(
        kept.empty_memory_blocks_live,
        counts.empty_memory_blocks_live + 1,
        "the block is kept empty"
    );
    let again = device
        .new_buffer(OWN_BLOCK, ResourceOptions::private())
        .unwrap();
    let taken = device.work_counts();
    assert_eq!(
        taken.object_churn.vk_memory_allocations, kept.object_churn.vk_memory_allocations,
        "the next buffer of its size took the kept block"
    );
    assert_eq!(
        taken.empty_memory_blocks_live,
        counts.empty_memory_blocks_live
    );
    drop(again);
}

/// An import of the caller's host memory is a dedicated block too, and is counted as the import
/// it is rather than as a dedicated block: the two have different cures, so the breakdown keeps
/// them apart.
#[test]
fn a_host_import_s_block_is_counted_as_an_import() {
    let scratch = Scratch::new("import");
    let device = device(&scratch.0);
    let host = Host::new();
    let runs = Arc::new(Runs::default());
    let (before, _) = frees(&device);
    let Some(buffer) = unused(&device, &host, &runs) else {
        return;
    };
    drop(buffer);
    let (after, total) = frees(&device);
    assert_eq!(
        after.host_import,
        before.host_import + 1,
        "the import's block was freed as a host import"
    );
    assert_eq!(
        after.dedicated, before.dedicated,
        "the import was not counted as a plain dedicated block"
    );
    assert_eq!(
        after.blocks,
        before.blocks + 1,
        "exactly one block was freed, under one cause"
    );
    assert!(
        after.blocks <= total,
        "the allocator's blocks ({}) are no more than the process's vkFreeMemory calls ({total})",
        after.blocks,
    );
    drop(device);
}
