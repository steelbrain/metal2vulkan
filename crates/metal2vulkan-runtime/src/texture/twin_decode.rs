//! Decoded twins' CPU decodes, off the caller's thread.
//!
//! **Why.** Every host write into a decoded twin whose blocks take CPU copies decodes on the CPU,
//! and that once ran on the caller's thread before `replace_region` returned: 14 to 27 ms for a
//! 2048 x 2048 level on an RTX 3090's host, where Metal's call is a plain copy. Now the call
//! copies the blocks and queues their decode here, and returns.
//!
//! **The thread.** One per device, started with its first job ([`TwinDecodes::start`]), takes
//! the jobs in the order they were queued, so a later write's texels land in a twin after an
//! earlier one's. Each job decodes its region in bands of at most [`BAND_BYTES`] of texels, each
//! band's rows split over up to [`THREADS`] threads ([`decode_band`]; a PVRTC image's bands read
//! all of its blocks, `TwinDecode::whole_levels`), and copies each band
//! into the twin as the twin's copies go (on the host, or on the staging queue, waited for). A
//! new texture's start, the zero blocks' decode, is a constant fill there.
//!
//! **Who waits.** The scheduler holds each batch whose footprint names a texture with a job
//! queued or running until its jobs finished (`Scheduler::decode_queued`), so the first GPU work
//! that may read the twin waits, and nothing else; a job of an untracked texture, whose uses no
//! footprint names, holds every batch. A job's end opens a trail (`trails.rs`) that closes once
//! the thread dropped the job, which holds the image and so the device: a wait for a command
//! buffer that read the twin returns only then. The call waits for nothing but room (the owner's rule:
//! Metal's `replaceRegion` is an unsynchronized CPU write and never waits for GPU work): a batch
//! already submitted that reads the texture reads the twin as the thread's copies stand when it
//! runs. A CPU read of the texture reads its blocks and waits for nothing.
//!
//! **Bounded.** At most [`QUEUED_BYTES`] of written blocks wait; past that a call waits for room
//! (a job alone always goes in). A job holds its image until it finished.

use super::super::copy_plan::ImageRegion;
use super::super::device::lock;
use super::super::error::{Error, MemoryKind};
use super::super::exit_guard;
use super::decoded_twin::TwinDecode;
use super::Image;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

/// The most bytes of texels one band of a job decodes and copies at once.
pub(super) const BAND_BYTES: usize = 16 << 20;

/// The most threads one band's decode is split over.
pub(super) const THREADS: usize = 8;

/// The fewest block rows a thread of a band decodes: below it, fewer threads.
const ROWS_PER_THREAD: usize = 16;

/// The fewest texel rows one part of a band's host copy writes: below it, fewer parts.
pub(super) const ROWS_PER_PART: u32 = 64;

/// The most bytes of written blocks the jobs queued hold.
pub(crate) const QUEUED_BYTES: u64 = 256 << 20;

/// How long the thread sleeps at most without looking for a job or a stop.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One device's decode thread and its jobs.
#[derive(Default)]
pub(crate) struct TwinDecodes {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Default)]
struct Shared {
    queue: Mutex<Queue>,
    /// A job was queued, or the thread is asked to stop.
    work: Condvar,
    /// A job finished, or the thread ended.
    done: Condvar,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    /// The written blocks' bytes the queued and running jobs hold.
    queued: u64,
    stopping: bool,
    /// The thread ended: the device stopped it, or the process is exiting.
    ended: bool,
}

/// One region of an image to decode into its twin.
pub(super) struct Job {
    pub(super) image: Arc<Image>,
    pub(super) region: ImageRegion,
    /// The written blocks, laid out as `buffer_image_copy` reads them; `None` for a new
    /// texture's zero blocks.
    pub(super) blocks: Option<Vec<u8>>,
}

impl Job {
    fn bytes(&self) -> u64 {
        self.blocks.as_ref().map_or(0, |blocks| blocks.len() as u64)
    }

    /// Whether this job writes every texel `earlier` (of the same image) writes: a write of
    /// blocks over a region holding all of `earlier`'s. Jobs run in order, and a batch that may
    /// read the image waits for both, so `earlier`'s texels are never read.
    fn overwrites(&self, earlier: &Job) -> bool {
        let (a, b) = (&self.region, &earlier.region);
        let within = |start: u32, size: u32, inner: u32, inner_size: u32| {
            start <= inner && inner + inner_size <= start + size
        };
        Arc::ptr_eq(&self.image, &earlier.image)
            && self.blocks.is_some()
            && a.aspect == b.aspect
            && a.level == b.level
            && within(a.layer, a.layers, b.layer, b.layers)
            && (0..3).all(|axis| within(a.origin[axis], a.size[axis], b.origin[axis], b.size[axis]))
    }
}

impl TwinDecodes {
    /// Start the thread unless it runs: refused when the host cannot start one.
    pub(super) fn start(&self) -> Result<(), Error> {
        let mut thread = lock(&self.thread);
        if thread.is_none() {
            let shared = Arc::clone(&self.shared);
            let handle = exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-twin-decodes".into()),
                move || run(&shared),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
            *thread = Some(handle);
        }
        Ok(())
    }

    /// Queue `job` for the thread, which [`start`](Self::start) started. Waits while the jobs
    /// queued hold [`QUEUED_BYTES`] of blocks and this one would add to them.
    pub(super) fn queue(&self, job: Job) {
        let bytes = job.bytes();
        let mut queue = lock(&self.shared.queue);
        while queue.queued > 0 && queue.queued + bytes > QUEUED_BYTES && !queue.ended {
            queue = self
                .shared
                .done
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
        queue.queued += bytes;
        queue.jobs.push_back(job);
        drop(queue);
        self.shared.work.notify_all();
    }

    /// Ask the thread to end, and wait for it unless this is the thread. The device is being
    /// dropped, so no job is left: every job holds its image, and so the device.
    pub(crate) fn stop(&self) {
        lock(&self.shared.queue).stopping = true;
        self.shared.work.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id());
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The thread's loop: each job in turn, a section of the exit guard. It ends when the device
/// stops it, or once the process is exiting; a job left then stays undone.
fn run(shared: &Shared) {
    loop {
        let (job, covered) = {
            let mut queue = lock(&shared.queue);
            loop {
                if let Some(job) = queue.jobs.pop_front() {
                    // ---- A start a later queued write overwrites whole is skipped ----
                    let covered = job.blocks.is_none()
                        && queue.jobs.iter().any(|later| later.overwrites(&job));
                    break (job, covered);
                }
                if queue.stopping {
                    queue.ended = true;
                    drop(queue);
                    shared.done.notify_all();
                    return;
                }
                queue = shared
                    .work
                    .wait_timeout(queue, WAIT_SLICE)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        };
        let Some(busy) = exit_guard::enter() else {
            lock(&shared.queue).ended = true;
            shared.done.notify_all();
            return;
        };
        if !covered {
            job.image
                .finish_twin_decode(&job.region, job.blocks.as_deref());
        }
        // The batches that wait for it may go; the trail closes once the job went.
        let trail = job.image.twin_decoded();
        drop(busy);
        {
            let mut queue = lock(&shared.queue);
            queue.queued -= job.bytes();
        }
        shared.done.notify_all();
        // Last: it may be the image's last hold, and with it the device's, whose drop stops this
        // thread; then the trail.
        drop(job);
        drop(trail);
    }
}

/// Decode rows `first` to `first + rows` of one image `width` x `height` texels, whose blocks
/// are `blocks` (all of the image's), into `texels` (exactly the rows' texels), split over up to
/// [`THREADS`] threads of at least [`ROWS_PER_THREAD`] block rows each (`TwinDecode::decode_rows`:
/// a block decode's shares start at a block row, as `first` does); this thread takes its share. A
/// thread the host does not start leaves its share to the others.
pub(super) fn decode_band(
    decode: TwinDecode,
    blocks: &[u8],
    size: [u32; 2],
    [first, rows]: [u32; 2],
    texels: &mut [u8],
) {
    let bh = decode.block_extent()[1];
    let row = size[0] as usize * decode.texel_bytes() as usize;
    let host = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let down = rows.div_ceil(bh) as usize;
    let threads = host.min(THREADS).min(down / ROWS_PER_THREAD).max(1);
    if threads == 1 {
        decode.decode_rows(blocks, size, [first, rows], texels);
        return;
    }
    // Each share: its first row, its texel rows' bytes, and its rows, whole block rows.
    let per = (down.div_ceil(threads) as u32) * bh;
    let mut shares = Vec::with_capacity(threads);
    let mut rest = texels;
    for at in (0..rows).step_by(per as usize) {
        let h = per.min(rows - at);
        let (mine, after) = rest.split_at_mut(h as usize * row);
        rest = after;
        shares.push(Mutex::new(Some((first + at, mine, h))));
    }
    let next = AtomicUsize::new(0);
    let work = || {
        while let Some(share) = shares.get(next.fetch_add(1, Ordering::Relaxed)) {
            if let Some((y, texels, h)) = lock(share).take() {
                decode.decode_rows(blocks, size, [y, h], texels);
            }
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads {
            let builder = std::thread::Builder::new().name("metal2vulkan-twin-decode".into());
            if exit_guard::spawn_scoped(scope, builder, work).is_err() {
                break;
            }
        }
        work();
    });
}
