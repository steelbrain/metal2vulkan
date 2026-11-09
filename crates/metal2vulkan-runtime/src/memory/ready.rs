//! Pooled blocks made ahead, off the program's thread, before any resource is placed in them:
//! for each pool the runtime's encodings and commits place their own resources in
//! (`memory::keep_room`) and each pool of exportable buffers (`memory::allocate_exported`) while
//! it grows, and for the pool a color texture's host-transfer image places in
//! (`texture/host_copy.rs`). Each is allocated right after a burst of the device's
//! submissions returned, then every page of its mapping is read once by the host.
//!
//! **Why right after a burst.** A block's `vkAllocateMemory` holds the device's submissions
//! while it runs, on whichever thread it runs. On RADV (a Radeon 780M, kernel 7.3.1) the kernel
//! allocates a block's pages under the reservation of the process's GPU address space, which each
//! submission's ioctl takes too: in a kernel trace of `tests::memory_block_bench`, a 64 MiB
//! block's page allocation took 10.2 ms at the median (31 ms at most), and a submission made
//! meanwhile waited for the reservation up to 29.4 ms. On NVIDIA's driver each `vkAllocateMemory` holds every
//! submission until it returns (`memory.rs`). So a making waits for a burst of the device's queue
//! uses to end (`device/queue_quiet.rs`), at most [`QUIET_MOST`], and runs in the gap before the
//! next: a program that presents at 60 Hz leaves 16.7 ms a frame less its frame's submissions.
//!
//! **Why the pages are read.** A host copy (`replace_region`, `get_bytes`) reads and writes
//! through the block's mapping, which the driver's copies share with the runtime's own
//! `vkMapMemory` of the block. The first touch of a page of that mapping faults into the kernel,
//! and the fault waits for the memory's pending kernel work. On RADV that work may sit behind the
//! program's queued GPU work: once the GPU idled into runtime suspend, the kernel clears a new
//! block's memory before handing it out, the clear is ordered after every submission of the
//! process (the block shares the process's GPU address space's reservation), and a fault on any
//! of the process's memory waits for it. On an RX 7900 XTX (Mesa 26.2.6, kernel 7.3.1) the first
//! `replace_region` into a texture placed in a block made behind about 1.5 s of queued fills
//! waited 516 to 684 ms in that clear (`amdgpu_gem_fault` in `dma_resv_wait_timeout`,
//! `tests/zeroed_start.rs`); awake, the kernel hands out memory it had cleared, and the write
//! took 160 us. A block not mapped has no page to read.
//!
//! - **Ahead.** `memory::keep_room`, which the device's creation calls run, starts a making for a
//!   pool the encodings and commits place in once the pool grew (a second or later pooled block
//!   was added to it within `KEPT_EMPTY_FOR`) and lacks room for a whole block, so that a growth
//!   like a guest's scroll on the 780M (from 20 to 63 live blocks of 64 MiB in about 20 s, about 4
//!   a second) finds a block made when it needs one; and, grown or not, once the pool is near its
//!   next block (`Pool::near_next`: it holds a pooled block and lacks room for three quarters of
//!   one, `memory::ahead_room`, a quarter of a block before the half block `keep_room` keeps), so
//!   that its first growth, and the first after a pause, finds one made too, where a guest's
//!   scroll on a Radeon 780M made its buffers' pool's first growth in place, on the guest's
//!   thread (`tests::reclaim_submit_bench`'s scroll arm: of each phase's 8 blocks, the first was
//!   made in place in three runs out of three, quiet and fragmented, and none is now). Such a
//!   pool keeps that block at rest for as long as it stays near its next one: one block at most a
//!   pool, none for a pool with room for three quarters of a block. And for the
//!   host copies' pool once that lacks room for half a block, the first by the device's first
//!   creation call, before the program queued work. A pool of exportable buffers, which the
//!   creation calls themselves place in, gets its block started the same way once it grew or is
//!   near its next block, so that the creation call that finds no room there takes a block made
//!   ahead rather than make one on the program's thread (on a Radeon 780M a creation call that
//!   made such a 64 MiB block itself took 10.0 to 15.0 ms in three of four phases of
//!   `tests::reclaim_submit_bench`, quiet and fragmented, and none took over 0.3 ms with the
//!   blocks made ahead); none of its blocks is made by a creation call that places nothing there.
//! - **Added where room is needed.** A creation call that finds such a pool without room for
//!   half a block (`memory::keep_room`), and a placement that finds no room (`memory::place`),
//!   add the pool's block made ahead once its making ended. Where the making is under way they
//!   wait for it, rather than make a second block: no two threads make one for the same pool, and
//!   none is made only to be freed. A making still waiting for a burst's end when a call comes
//!   allocates at once, so that the call waits for the allocation alone, never for the program's
//!   own submissions to pause; a making that reads its pages as a call comes reads no more of
//!   them, the pages left unread faulting in at their first touch, as a block made in place does.
//!   The host copies' pool's block is added by its placements alone, and never waited for, as its
//!   making reads pages whose touch may wait behind the program's GPU work: a placement there that
//!   finds the making under way makes its own block. Where no block is made ahead, the call makes
//!   its own (`WorkCounts::buffer_blocks_made_in_place`, `image_blocks_made_in_place`). A pool's
//!   blocks change only at those calls, never as a making here ends.
//! - **Within the allocation margin, none.** A making no call waits for yet makes no block while
//!   the device's allocations are within the margin of its `maxMemoryAllocationCount` the
//!   runtime's work ahead leaves to the objects (`allocations.rs`'s `Allocations::room_ahead`).
//! - **Bounded.** One making a pool at most, on a thread of its own that ends with the block.
//!   While it waits to be added, its bytes count against its heap's budget of kept empty blocks
//!   (`Memory::kept_budget`); one whose pool stopped growing for `KEPT_EMPTY_FOR`, the pool
//!   having room for three quarters of a block, goes (`Memory::sweep_stale`). The device's teardown stops a
//!   making that has not begun, joins every one and frees its block. Its Vulkan calls run in a
//!   section of the exit guard (`exit_guard.rs`): once the process exits, no block is made.
//! - **Counted** (`WorkCounts`): the blocks made ahead and added, by kind
//!   (`buffer_blocks_made_ahead`, `image_blocks_made_ahead`; of them the host copies' pool's,
//!   `host_copy_blocks_readied`), those a call waited for (`blocks_ahead_waited_for`; of them
//!   those whose wait for a burst's end the call cut short, `blocks_ahead_hurried`), and those
//!   whose wait for a burst's end reached [`QUIET_MOST`] (`blocks_ahead_past_quiet_wait`).
//! - **Not what a suspend evicts.** Where the kernel moves a block out of device memory as the GPU
//!   suspends and back at the next submission, the mapping's pages go with the move, and the first
//!   touch after it waits for the moves back (docs/FAST-PATHS.md, known gaps).

use super::{Block, BlockMaker, Holds};
use crate::device::functions::Functions;
use crate::device::lock;
use crate::device::queue_quiet::{QueueQuiet, Waited};
use crate::exit_guard;
use crate::work_counts::Counters;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The stride of the reads that touch a block's pages: the smallest page any host maps, so a
/// host with larger pages reads each of its pages more than once.
const PAGE: usize = 4096;

/// The bytes of a block's pages read between two looks at whether a call waits for the block
/// ([`ready`]).
const ASK_EVERY: usize = 64 * PAGE;

/// The longest a making waits for a burst of the device's submissions to end before it makes its
/// block all the same, and the time without a submission after which the device is taken to be
/// idle, its making begun at once (`QueueQuiet::wait_after_burst`): three frames at 60 Hz.
pub(super) const QUIET_MOST: Duration = Duration::from_millis(50);

/// A making's [`Shared::go`] word: wait for a burst's end, make at once, or make nothing.
const WAIT: u8 = 0;
const NOW: u8 = 1;
const STOP: u8 = 2;

/// A block's making under way, or ended and its block not yet added (the module docs).
pub(super) struct Readying {
    /// The pool it is for.
    pool: (Holds, u32),
    /// When it started.
    started: Instant,
    /// What its thread and the calls that wait for it share.
    shared: Arc<Shared>,
    /// Its thread, which ends once it left its answer in `shared`.
    thread: JoinHandle<()>,
}

/// What a making's thread and the calls that wait for it share.
struct Shared {
    /// What the thread does while it has not begun the making: [`WAIT`], [`NOW`] or [`STOP`].
    go: AtomicU8,
    /// A call waited for the making ([`Waiter::wait`]).
    waited_for: AtomicBool,
    /// Whether the making ended, and its block until a call takes it: `None` where it could not
    /// be made, it was stopped, or the process exits.
    made: Mutex<(bool, Option<Made>)>,
    /// The answer was left.
    ended: Condvar,
}

/// A block made ahead, and how its wait for a burst's end ended.
pub(super) struct Made {
    pub(super) block: Block,
    pub(super) waited: Waited,
}

/// A call's wait for a making under way ([`Readying::waiter`]), with no lock of the memory held.
pub(super) struct Waiter(Arc<Shared>);

impl Waiter {
    /// Ask the making to stop waiting for a burst's end, and wait for it to end.
    pub(super) fn wait(self) {
        let shared = self.0;
        shared.waited_for.store(true, Ordering::Release);
        let _ = shared
            .go
            .compare_exchange(WAIT, NOW, Ordering::AcqRel, Ordering::Acquire);
        let mut made = lock(&shared.made);
        while !made.0 {
            made = (shared.ended.wait(made)).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Readying {
    /// Start making a block of `bytes` for `pool` with `maker` on a thread of its own, through
    /// `functions`, once a burst of `quiet`'s uses ended, with a buffer of addresses counted in
    /// `addressed` where given: `None` where the host cannot start a thread.
    pub(super) fn start(
        maker: BlockMaker,
        functions: Arc<Functions>,
        quiet: Arc<QueueQuiet>,
        addressed: Option<Arc<Counters>>,
        (pool, bytes): ((Holds, u32), u64),
    ) -> Option<Readying> {
        let shared = Arc::new(Shared {
            go: AtomicU8::new(WAIT),
            waited_for: AtomicBool::new(false),
            made: Mutex::new((false, None)),
            ended: Condvar::new(),
        });
        let builder = std::thread::Builder::new().name("metal2vulkan-ready".into());
        let thread = {
            let shared = Arc::clone(&shared);
            exit_guard::spawn(builder, move || {
                let go = || shared.go.load(Ordering::Acquire);
                let waited = quiet.wait_after_burst(QUIET_MOST, || go() != WAIT);
                let made = match go() {
                    STOP => None,
                    _ => ready(
                        &maker,
                        &functions,
                        addressed.as_deref(),
                        (pool, bytes),
                        || go() != WAIT,
                    )
                    .map(|block| Made { block, waited }),
                };
                *lock(&shared.made) = (true, made);
                shared.ended.notify_all();
            })
        };
        Some(Readying {
            pool,
            started: Instant::now(),
            shared,
            thread: thread.ok()?,
        })
    }

    /// The pool it is for.
    pub(super) fn pool(&self) -> (Holds, u32) {
        self.pool
    }

    /// When it started.
    pub(super) fn started(&self) -> Instant {
        self.started
    }

    /// Whether its making ended, its block ready or its making failed: [`take`](Self::take) waits
    /// for nothing.
    pub(super) fn finished(&self) -> bool {
        lock(&self.shared.made).0
    }

    /// Whether a call waited for it ([`Waiter::wait`]).
    pub(super) fn waited_for(&self) -> bool {
        self.shared.waited_for.load(Ordering::Acquire)
    }

    /// A wait for its making, to run with no lock of the memory held.
    pub(super) fn waiter(&self) -> Waiter {
        Waiter(Arc::clone(&self.shared))
    }

    /// Its block, its thread joined: where its making has not ended, asked first to stop waiting
    /// for a burst's end, and waited for. `None` where it could not be made.
    pub(super) fn take(self) -> Option<Made> {
        let _ = self
            .shared
            .go
            .compare_exchange(WAIT, NOW, Ordering::AcqRel, Ordering::Acquire);
        drop(self.thread.join());
        lock(&self.shared.made).1.take()
    }

    /// Its block where its making began, its thread joined; a making that has not begun makes
    /// none.
    pub(super) fn stop(self) -> Option<Block> {
        let _ = self
            .shared
            .go
            .compare_exchange(WAIT, STOP, Ordering::AcqRel, Ordering::Acquire);
        drop(self.thread.join());
        let made = lock(&self.shared.made).1.take();
        made.map(|made| made.block)
    }
}

/// Make a block of `bytes` for `pool`, with a buffer of addresses counted in `addressed` where
/// given, and read every page of its mapping (the module docs) until `wanted` says a call waits
/// for the block (or the device's teardown stops the making): that call waits for the allocation
/// alone, and the pages left unread fault in at their first touch, as a block made in place does.
fn ready(
    maker: &BlockMaker,
    functions: &Functions,
    addressed: Option<&Counters>,
    (pool, bytes): ((Holds, u32), u64),
    wanted: impl Fn() -> bool,
) -> Option<Block> {
    let busy = exit_guard::enter()?;
    // Ahead of any placement, none within the margin of the device's allocations the runtime's
    // work ahead leaves (`allocations.rs`); a call waiting for it has its allocation made.
    if !wanted() && !crate::allocations::room_ahead(functions.get().handle()) {
        return None;
    }
    let block = maker.make(functions.get(), pool, bytes, addressed).ok()?;
    if let Some(mapped) = block.mapped {
        let length = usize::try_from(block.size).unwrap_or(usize::MAX);
        for offset in (0..length).step_by(PAGE) {
            if offset % ASK_EVERY == 0 && wanted() {
                break;
            }
            // SAFETY: the offset lies within the block's mapping, made whole as the block was;
            // nothing is placed in the block yet, so nothing else reads or writes it.
            unsafe { std::ptr::read_volatile(mapped.as_ptr().add(offset)) };
        }
    }
    drop(busy);
    Some(block)
}
