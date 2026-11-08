//! Blocks of the pool a color texture's host-transfer image places in (`texture/host_copy.rs`),
//! made ready off the program's thread before any image is placed in them: allocated, then every
//! page of their mapping read once by the host.
//!
//! A host copy (`replace_region`, `get_bytes`) reads and writes through the block's mapping, which
//! the driver's copies share with the runtime's own `vkMapMemory` of the block. The first touch of
//! a page of that mapping faults into the kernel, and the fault waits for the memory's pending
//! kernel work. On RADV that work may sit behind the program's queued GPU work: once the GPU idled
//! into runtime suspend, the kernel clears a new block's memory before handing it out, the clear
//! is ordered after every submission of the process (the block shares the process's GPU address
//! space's reservation), and a fault on any of the process's memory waits for it. On an RX 7900
//! XTX (Mesa 26.2.6, kernel 7.3.1) the first `replace_region` into a texture placed in a block made
//! behind about 1.5 s of queued fills waited 516 to 684 ms in that clear (`amdgpu_gem_fault` in
//! `dma_resv_wait_timeout`, `tests/zeroed_start.rs`); awake, the kernel hands out memory it had
//! cleared, and the write took 160 us.
//!
//! - **Ahead.** `memory::keep_room`, which the device's creation calls run, starts a block's making
//!   here once the pool lacks room for half a block and holds no block made ready or being made;
//!   the first is started by the device's first creation call, before the program queued work.
//!   No image is placed in the block before every page was read, so its host copies fault on none
//!   of them, whatever kernel work is pending then (a later block's clear included).
//! - **Added at a placement.** The first placement in the pool that finds no room adds the block,
//!   waiting for its making where it is still under way, rather than making one whose first touch
//!   would wait instead (`memory::place`, `WorkCounts::host_copy_blocks_readied`); where none is
//!   held it makes one as before (`WorkCounts::host_copy_blocks_made_in_place`). The pool's blocks
//!   change only at its placements, never as a making here ends.
//! - **One at a time,** on a thread of its own that ends with the block; the device's teardown
//!   joins it and frees its block. Its Vulkan calls run in a section of the exit guard
//!   (`exit_guard.rs`): once the process exits, no block is made.
//! - **Not what a suspend evicts.** Where the kernel moves a block out of device memory as the GPU
//!   suspends and back at the next submission, the mapping's pages go with the move, and the first
//!   touch after it waits for the moves back (docs/FAST-PATHS.md, known gaps).

use super::{Block, BlockMaker, Holds};
use crate::device::functions::Functions;
use crate::exit_guard;
use std::sync::Arc;
use std::thread::JoinHandle;

/// The stride of the reads that touch a block's pages: the smallest page any host maps, so a
/// host with larger pages reads each of its pages more than once.
const PAGE: usize = 4096;

/// A block's making under way (the module docs).
pub(super) struct Readying {
    /// The pool it is for.
    pool: (Holds, u32),
    /// Its thread, which answers the block once every page was read: `None` where it could not be
    /// made, or the process exits.
    thread: JoinHandle<Option<Block>>,
}

impl Readying {
    /// Start making a block of `bytes` for `pool` with `maker` on a thread of its own, through
    /// `functions`: `None` where the host cannot start a thread.
    pub(super) fn start(
        maker: BlockMaker,
        functions: Arc<Functions>,
        pool: (Holds, u32),
        bytes: u64,
    ) -> Option<Readying> {
        let builder = std::thread::Builder::new().name("metal2vulkan-ready".into());
        let thread = exit_guard::spawn(builder, move || ready(&maker, &functions, pool, bytes));
        Some(Readying {
            pool,
            thread: thread.ok()?,
        })
    }

    /// The pool it is for.
    pub(super) fn pool(&self) -> (Holds, u32) {
        self.pool
    }

    /// Its block, waited for: `None` where it could not be made.
    pub(super) fn take(self) -> Option<Block> {
        self.thread.join().ok().flatten()
    }
}

/// Make a block of `bytes` for `pool` and read every page of its mapping (the module docs).
fn ready(
    maker: &BlockMaker,
    functions: &Functions,
    pool: (Holds, u32),
    bytes: u64,
) -> Option<Block> {
    let busy = exit_guard::enter()?;
    let block = maker.make(functions.get(), pool, bytes, None).ok()?;
    if let Some(mapped) = block.mapped {
        let length = usize::try_from(block.size).unwrap_or(usize::MAX);
        for offset in (0..length).step_by(PAGE) {
            // SAFETY: the offset lies within the block's mapping, made whole as the block was;
            // nothing is placed in the block yet, so nothing else reads or writes it.
            unsafe { std::ptr::read_volatile(mapped.as_ptr().add(offset)) };
        }
    }
    drop(busy);
    Some(block)
}
