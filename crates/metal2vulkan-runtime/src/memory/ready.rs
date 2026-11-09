//! Pooled blocks made ahead, off the program's thread, before any resource is placed in them:
//! for each pool the runtime's encodings and commits place their own resources in
//! (`memory::keep_room`) and each pool of exportable buffers (`memory::allocate_exported`) while
//! it grows, and for the pool a color texture's host-transfer image places in
//! (`texture/host_copy.rs`). A pool's are allocated right after a burst of the device's
//! submissions returned, one after another, and every page of each one's mapping is read once by
//! the host.
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
//! The blocks of one wake are made one `vkAllocateMemory` each, so a submission waits for one
//! block's making at most, however many blocks the wake makes.
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
//!   made in place in three runs out of three, quiet and fragmented, and none is now). A pool
//!   keeps a block's worth of bytes ahead: as many of its blocks as one block of
//!   `memory::BLOCK_BYTES` holds (one of 64 MiB; four where its blocks of plain memory are made of
//!   16 MiB after a slow making, `Memory::plain_block_made`); and, while its placements larger
//!   than its blocks are recent (the last within `KEPT_EMPTY_FOR`), each of which takes a block of
//!   its own size (`memory::pooled_block_bytes`), one block of the largest such size, its own
//!   blocks filling the rest of the block's worth. They are made one after another once a burst
//!   ended; those it lacks are queued after the blocks its thread still makes, or made on a new
//!   thread where that ended. So each making stays as short as its block, and the pool's growth
//!   still finds its blocks made: with one block of 16 MiB ahead at a time, a guest's scroll on a
//!   Radeon 780M whose blocks were made so made 82 to 114 blocks of images on its own threads and
//!   waited for 110 to 159 of those made ahead. It keeps them for as long as it stays near its
//!   next block; none for a pool with room for three quarters of a block. And for the host
//!   copies' pool once that lacks room for half a block, the first by the device's first creation
//!   call, before the program queued work. A pool of exportable buffers, which the creation calls
//!   themselves place in, gets its blocks started the same way once it grew or is near its next
//!   block, so that the creation call that finds no room there takes a block made ahead rather
//!   than make one on the program's thread (on a Radeon 780M a creation call that made such a
//!   64 MiB block itself took 10.0 to 15.0 ms in three of four phases of
//!   `tests::reclaim_submit_bench`, quiet and fragmented, and none took over 0.3 ms with the
//!   blocks made ahead); none of its blocks is made by a creation call that places nothing there.
//! - **Added where room is needed.** A creation call that finds such a pool without room for
//!   half a block (`memory::keep_room`), and a placement that finds no room (`memory::place`),
//!   add the pool's smallest block made ahead of the size they need. Where none is made they wait
//!   for the one being made or still to make, or else for one the pool's thread makes first for
//!   them, the thread started where the pool has none, the rest of its block's worth made after
//!   it: no two threads make blocks for the same pool, none is made only to be freed, and a call
//!   makes no block of a pool whose blocks are made ahead where a making can serve it. A making
//!   still waiting for a burst's end when a call comes allocates at once, so that the call waits
//!   for the allocation alone, never for the program's own submissions to pause; while a call
//!   waits the thread reads no page, the pages left unread faulting in at their first touch, as a
//!   block made in place does. The host copies' pool's blocks are added by its placements alone,
//!   and a placement there never waits while its thread reads a block's pages, whose first touch
//!   may wait behind the program's GPU work: the call counts itself waiting before it looks, and
//!   the thread looks for a waiting call after it says it reads and before each run of reads. A
//!   call makes its own block (`WorkCounts::buffer_blocks_made_in_place`,
//!   `image_blocks_made_in_place`) only where no making serves it: a placement in the host copies'
//!   pool while its thread reads; a call whose making failed, or whose host starts no thread; and
//!   a placement before the device's first creation call, which gives the makings the device's
//!   functions and queues. A pool's blocks change only at those calls, never as a making here
//!   ends.
//! - **Within the allocation margin, none.** A making no call waits for yet makes no block while
//!   the device's allocations are within the margin of its `maxMemoryAllocationCount` the
//!   runtime's work ahead leaves to the objects (`allocations.rs`'s `Allocations::room_ahead`).
//! - **Bounded.** One thread a pool at most, which makes its blocks one after another and ends
//!   with the last; the next starts once it ended. The bytes of the blocks made and not yet added,
//!   and of those still to make, count against their heap's budget of kept empty blocks
//!   (`Memory::kept_budget`); a pool's that stopped growing for `KEPT_EMPTY_FOR`, the pool having
//!   room for three quarters of a block, go (`Memory::sweep_stale`). The device's teardown stops
//!   a making that has not begun, joins every thread and frees their blocks. Its Vulkan calls run
//!   in a section of the exit guard (`exit_guard.rs`): once the process exits, no block is made.
//! - **Counted** (`WorkCounts`): the blocks made ahead and added, by kind
//!   (`buffer_blocks_made_ahead`, `image_blocks_made_ahead`; of them the host copies' pool's,
//!   `host_copy_blocks_readied`), those made while a call waited (`blocks_ahead_waited_for`; of
//!   them those whose wait for a burst's end the call cut short, `blocks_ahead_hurried`), and
//!   those whose wait for a burst's end reached [`QUIET_MOST`] (`blocks_ahead_past_quiet_wait`).
//! - **Not what a suspend evicts.** Where the kernel moves a block out of device memory as the GPU
//!   suspends and back at the next submission, the mapping's pages go with the move, and the first
//!   touch after it waits for the moves back (docs/FAST-PATHS.md, known gaps).

use super::{Block, BlockMaker, Holds};
use crate::device::functions::Functions;
use crate::device::lock;
use crate::device::queue_quiet::{QueueQuiet, Waited};
use crate::exit_guard;
use crate::work_counts::Counters;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The stride of the reads that touch a block's pages: the smallest page any host maps, so a
/// host with larger pages reads each of its pages more than once.
const PAGE: usize = 4096;

/// The bytes of a block's pages read between two looks at whether a call waits for a block
/// ([`ready`]).
const ASK_EVERY: usize = 64 * PAGE;

/// The longest a making waits for a burst of the device's submissions to end before it makes its
/// block all the same, and the time without a submission after which the device is taken to be
/// idle, its making begun at once (`QueueQuiet::wait_after_burst`): three frames at 60 Hz.
pub(super) const QUIET_MOST: Duration = Duration::from_millis(50);

/// A thread's [`Shared::go`] word: wait for a burst's end, make at once, or make nothing more.
const WAIT: u8 = 0;
const NOW: u8 = 1;
const STOP: u8 = 2;

/// A pool's blocks made ahead: the thread that makes them, and the blocks made and not yet added
/// (the module docs).
pub(super) struct Readying {
    /// The pool it is for.
    pool: (Holds, u32),
    /// When it was last given blocks to make.
    started: Instant,
    /// What its thread and the calls that wait for its blocks share.
    shared: Arc<Shared>,
    /// Its thread, which ends once it has no block left to make; joined as the next one starts or
    /// the readying goes.
    thread: Option<JoinHandle<()>>,
}

/// What a readying's thread and the calls that wait for its blocks share.
struct Shared {
    /// What the thread does while it has not begun its first block: [`WAIT`], [`NOW`] or
    /// [`STOP`]; [`STOP`] also ends it before its next block.
    go: AtomicU8,
    /// The calls waiting for a block ([`Waiter`]), each counted from its making to its drop.
    waiting: AtomicUsize,
    /// The thread reads a block's pages now ([`ready`]).
    reading: AtomicBool,
    /// Its blocks made and not yet added, the one being made, those still to make, and whether
    /// its thread ended.
    state: Mutex<State>,
    /// A block was made, or the thread ended.
    changed: Condvar,
}

/// A readying's blocks ([`Shared::state`]).
struct State {
    /// Made and not yet added, in the order made.
    made: VecDeque<Made>,
    /// The bytes of the block being made.
    making: Option<u64>,
    /// The bytes of each block still to make, in turn.
    to_make: VecDeque<u64>,
    /// The thread ended: it had no block left to make, one failed to be made, it was stopped, or
    /// the process exits. Nothing is queued on it from then on.
    ended: bool,
}

impl State {
    /// Whether a block of `need` bytes or more is made.
    fn holds(&self, need: u64) -> bool {
        self.made.iter().any(|made| made.block.size >= need)
    }

    /// Whether a block of `need` bytes or more is made, being made or still to make.
    fn will_hold(&self, need: u64) -> bool {
        self.holds(need)
            || self.making.is_some_and(|bytes| bytes >= need)
            || self.to_make.iter().any(|&bytes| bytes >= need)
    }
}

/// A block made ahead, and how it was made.
pub(super) struct Made {
    pub(super) block: Block,
    /// How the wait for a burst's end before it ended: `None` for a block made after another of
    /// the same wake, or on a thread started for a call already waiting, which waits for none.
    pub(super) waited: Option<Waited>,
    /// A call waited for a block as it was made.
    pub(super) waited_for: bool,
}

/// A call's wait for a readying's block of `need` bytes or more ([`Readying::waiter`]), with no
/// lock of the memory held: counted among the waiting from its making to its drop, so that the
/// thread reads no page meanwhile.
pub(super) struct Waiter {
    shared: Arc<Shared>,
    need: u64,
}

impl Waiter {
    /// A wait on `shared` for a block of `need` bytes or more, counted from now.
    fn new(shared: &Arc<Shared>, need: u64) -> Waiter {
        shared.waiting.fetch_add(1, Ordering::SeqCst);
        Waiter {
            shared: Arc::clone(shared),
            need,
        }
    }

    /// Ask the thread to stop waiting for a burst's end, and wait for a block of the bytes it
    /// needs made, or for the thread to end.
    pub(super) fn wait(self) {
        let shared = &self.shared;
        let _ = shared
            .go
            .compare_exchange(WAIT, NOW, Ordering::AcqRel, Ordering::Acquire);
        let mut state = lock(&shared.state);
        while !state.holds(self.need) && !state.ended {
            state = (shared.changed.wait(state)).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        self.shared.waiting.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What a readying's thread makes its blocks with.
pub(super) struct Making {
    pub(super) maker: BlockMaker,
    pub(super) functions: Arc<Functions>,
    pub(super) quiet: Arc<QueueQuiet>,
    /// The counts of buffers of addresses, where the pool's blocks carry one.
    pub(super) addressed: Option<Arc<Counters>>,
}

impl Readying {
    /// Start making blocks of `sizes` bytes for `pool`, one after another, on a thread of its own:
    /// once a burst of the device's queue uses ended, or at once for a call that waits for a
    /// block of `need` bytes or more, whose wait it gives back. `None` where the host cannot
    /// start a thread.
    pub(super) fn start(
        making: Making,
        pool: (Holds, u32),
        sizes: VecDeque<u64>,
        need: Option<u64>,
    ) -> Option<(Readying, Option<Waiter>)> {
        let go = match need {
            Some(_) => NOW,
            None => WAIT,
        };
        let shared = Arc::new(Shared {
            go: AtomicU8::new(go),
            waiting: AtomicUsize::new(0),
            reading: AtomicBool::new(false),
            state: Mutex::new(State {
                made: VecDeque::new(),
                making: None,
                to_make: sizes,
                ended: false,
            }),
            changed: Condvar::new(),
        });
        // Counted before the thread runs, which so reads none of that block's pages.
        let waiter = need.map(|need| Waiter::new(&shared, need));
        let thread = spawn(making, &shared, pool).ok()?;
        let readying = Readying {
            pool,
            started: Instant::now(),
            shared,
            thread: Some(thread),
        };
        Some((readying, waiter))
    }

    /// Make blocks of `sizes` bytes too: after those its thread has still to make (before them
    /// where `first`), or, its thread ended, on a new one, which waits for a burst's end unless a
    /// call waits for a block. Whether they will be made.
    pub(super) fn queue(&mut self, making: Making, sizes: &[u64], first: bool) -> bool {
        self.started = Instant::now();
        {
            let mut state = lock(&self.shared.state);
            if !state.ended {
                match first {
                    true => sizes
                        .iter()
                        .rev()
                        .for_each(|&bytes| state.to_make.push_front(bytes)),
                    false => state.to_make.extend(sizes),
                }
                return true;
            }
        }
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
        let go = match self.shared.waiting.load(Ordering::SeqCst) {
            0 => WAIT,
            _ => NOW,
        };
        self.shared.go.store(go, Ordering::Release);
        {
            let mut state = lock(&self.shared.state);
            state.to_make = sizes.iter().copied().collect();
            state.ended = false;
        }
        match spawn(making, &self.shared, self.pool) {
            Ok(thread) => {
                self.thread = Some(thread);
                true
            }
            Err(_) => {
                let mut state = lock(&self.shared.state);
                state.to_make.clear();
                state.ended = true;
                false
            }
        }
    }

    /// The pool it is for.
    pub(super) fn pool(&self) -> (Holds, u32) {
        self.pool
    }

    /// When it was last given blocks to make.
    pub(super) fn started(&self) -> Instant {
        self.started
    }

    /// Whether its thread ended: it makes no block until it is given more.
    pub(super) fn finished(&self) -> bool {
        lock(&self.shared.state).ended
    }

    /// Whether it holds a block made and not yet added.
    pub(super) fn has_block(&self) -> bool {
        !lock(&self.shared.state).made.is_empty()
    }

    /// Whether a block of `need` bytes or more is made, being made or still to make.
    pub(super) fn will_hold(&self, need: u64) -> bool {
        lock(&self.shared.state).will_hold(need)
    }

    /// The bytes of each of its blocks made and not yet added, being made, and still to make.
    pub(super) fn held(&self) -> Vec<u64> {
        let state = lock(&self.shared.state);
        let made = state.made.iter().map(|made| made.block.size);
        (made
            .chain(state.making)
            .chain(state.to_make.iter().copied()))
        .collect()
    }

    /// What it holds against its heap's budget of kept empty blocks: the bytes of [`held`].
    ///
    /// [`held`]: Self::held
    pub(super) fn outstanding_bytes(&self) -> u64 {
        self.held().iter().sum()
    }

    /// Its smallest block made and not yet added of `need` bytes or more, taken.
    pub(super) fn take_block(&self, need: u64) -> Option<Made> {
        let mut state = lock(&self.shared.state);
        let at = (state.made.iter().enumerate())
            .filter(|(_, made)| made.block.size >= need)
            .min_by_key(|(_, made)| made.block.size)
            .map(|(at, _)| at)?;
        state.made.remove(at)
    }

    /// A wait for its block of `need` bytes or more, to run with no lock of the memory held.
    /// `None` where `unless_reading` and its thread reads a block's pages now: a page's first
    /// touch may wait behind the program's GPU work (the module docs).
    pub(super) fn waiter(&self, need: u64, unless_reading: bool) -> Option<Waiter> {
        let waiter = Waiter::new(&self.shared, need);
        // Counted first: a thread that begins its reads after this sees the wait and reads none,
        // and one that began them before is seen here.
        match unless_reading && self.shared.reading.load(Ordering::SeqCst) {
            true => None,
            false => Some(waiter),
        }
    }

    /// Its thread joined, where it ended (the readying holds no block): nothing is left of it.
    pub(super) fn end(mut self) {
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }

    /// Its blocks made, its thread asked to make no more and joined: a block whose making began
    /// is among them.
    pub(super) fn stop(mut self) -> Vec<Block> {
        self.shared.go.store(STOP, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
        let mut state = lock(&self.shared.state);
        state.made.drain(..).map(|made| made.block).collect()
    }
}

/// Start the thread that makes `pool`'s blocks into `shared` (the module docs): it waits for a
/// burst of the device's queue uses to end where its `go` says so, then makes the blocks still to
/// make one after another, those queued meanwhile among them, each made available as it is made,
/// until none is left, [`STOP`], or a block fails to be made.
fn spawn(
    making: Making,
    shared: &Arc<Shared>,
    pool: (Holds, u32),
) -> std::io::Result<JoinHandle<()>> {
    let builder = std::thread::Builder::new().name("metal2vulkan-ready".into());
    let shared = Arc::clone(shared);
    exit_guard::spawn(builder, move || {
        let Making {
            maker,
            functions,
            quiet,
            addressed,
        } = making;
        let go = || shared.go.load(Ordering::Acquire);
        let mut waited = match go() {
            WAIT => Some(quiet.wait_after_burst(QUIET_MOST, || go() != WAIT)),
            _ => None,
        };
        let wanted = || shared.waiting.load(Ordering::SeqCst) > 0 || go() == STOP;
        loop {
            if go() == STOP {
                break;
            }
            let bytes = {
                let mut state = lock(&shared.state);
                let Some(bytes) = state.to_make.pop_front() else {
                    // Ended with the queue seen empty, under the same lock as a queuing call.
                    state.ended = true;
                    drop(state);
                    shared.changed.notify_all();
                    return;
                };
                state.making = Some(bytes);
                bytes
            };
            let block = ready(
                &maker,
                &functions,
                addressed.as_deref(),
                (pool, bytes),
                (&shared.reading, &wanted),
            );
            let waited_for = shared.waiting.load(Ordering::SeqCst) > 0;
            let mut state = lock(&shared.state);
            state.making = None;
            let Some(block) = block else {
                break;
            };
            state.made.push_back(Made {
                block,
                waited: waited.take(),
                waited_for,
            });
            drop(state);
            shared.changed.notify_all();
        }
        let mut state = lock(&shared.state);
        state.making = None;
        state.to_make.clear();
        state.ended = true;
        drop(state);
        shared.changed.notify_all();
    })
}

/// Make a block of `bytes` for `pool`, with a buffer of addresses counted in `addressed` where
/// given, and read every page of its mapping (the module docs), `reading` set meanwhile, until
/// `wanted` says a call waits for a block (or the device's teardown stops the making): that call
/// waits for the allocation alone, and the pages left unread fault in at their first touch, as a
/// block made in place does.
fn ready(
    maker: &BlockMaker,
    functions: &Functions,
    addressed: Option<&Counters>,
    (pool, bytes): ((Holds, u32), u64),
    (reading, wanted): (&AtomicBool, &dyn Fn() -> bool),
) -> Option<Block> {
    let busy = exit_guard::enter()?;
    // Ahead of any placement, none within the margin of the device's allocations the runtime's
    // work ahead leaves (`allocations.rs`); a call waiting for it has its allocation made.
    if !wanted() && !crate::allocations::room_ahead(functions.get().handle()) {
        return None;
    }
    let block = maker.make(functions.get(), pool, bytes, addressed).ok()?;
    if let Some(mapped) = block.mapped {
        // Set before the first look at `wanted`, which a waiting call counted before it looked at
        // this ([`Readying::waiter`]).
        reading.store(true, Ordering::SeqCst);
        let length = usize::try_from(block.size).unwrap_or(usize::MAX);
        for offset in (0..length).step_by(PAGE) {
            if offset % ASK_EVERY == 0 && wanted() {
                break;
            }
            // SAFETY: the offset lies within the block's mapping, made whole as the block was;
            // nothing is placed in the block yet, so nothing else reads or writes it.
            unsafe { std::ptr::read_volatile(mapped.as_ptr().add(offset)) };
        }
        reading.store(false, Ordering::SeqCst);
    }
    drop(busy);
    Some(block)
}
