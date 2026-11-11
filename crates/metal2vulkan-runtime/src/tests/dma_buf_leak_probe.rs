//! A concurrent probe of the dma-bufs a kernel keeps after the runtime freed their imports: no-copy
//! buffers over dma-bufs of host pages made and freed from several threads at once, each round
//! counted against the kernel's own list of dma-bufs (Linux's `/sys/kernel/debug/dma_buf/bufinfo`).
//! The single-threaded cycles (`dma_buf_import_cycles.rs`) run each way of importing alone; this
//! runs them together, as a guest's boot does.
//!
//! The pages are the test's own: a sealed anonymous `memfd` (guest RAM, as QEMU's
//! `memory-backend-memfd,share=on` makes it) holding a pattern of its own offsets. Each buffer is
//! a view stitched of scattered runs of the file's pages (each run mapped with `MAP_FIXED`, in
//! order, as a consumer maps a guest's object contiguously), and its source makes one dma-buf of
//! the same runs in the same order through udmabuf's `UDMABUF_CREATE_LIST` (at most 1,024 runs,
//! udmabuf's list limit), named with the test's own tag (`DMA_BUF_SET_NAME`) so that the kernel's
//! list shows which dma-bufs are the test's. The buffer's length is drawn from a log-uniform
//! spread from 16 KiB to 64 MiB (udmabuf's size limit), whole host pages. The device is the one
//! the cfg selects, made without `VK_EXT_external_memory_host` (`tests/no_pointer_import.rs`), so
//! that every buffer is one dma-buf import on every device that lists the dma-buf route.
//!
//! Each worker, per cycle, makes a buffer, and (but where it is left unused) commits a command
//! buffer that copies a few KiB from each end of it, unwaited. It keeps a few buffers live and
//! frees one picked at random once it holds more: waited for and freed, or freed while its own
//! copy is in flight (its completion then frees the import), or freed with no GPU use. Every
//! second cycle it also commits a command buffer of other work (copies between two private
//! buffers of its own), unwaited, so that frees happen while submissions are in flight. One cycle
//! in eight copies into a buffer of its own, waits, and checks the copy against the view's bytes:
//! the dma-buf holds the view's pages, in order.
//!
//! - **4 threads**, **8 threads**: that many workers at once, on one command queue they share.
//! - **One thread imports, another frees**: the importer makes the buffers and commits their
//!   copies; it hands each to the freer through a channel, which frees them as above.
//!
//! Each round ends once every worker freed its buffers and the device's live imports fell back to
//! where they were (`WorkCounts::dma_buf_memories_live`); the test then reads the kernel's list
//! until the test's dma-bufs on it stop falling (at most 2 s), and prints one line
//! (`DMA-BUF-LEAK`): the round, its time, and the workers' time in their buffers' makings (each
//! one import), their frees and their waits for command buffers, the longest of each, the buffers
//! by how each was freed, the runtime's imports made and
//! freed and its live count, the imports released while idle and imported again
//! (`host_imports_released`, `host_imports_reimported`), and the kernel's count of the test's
//! dma-bufs still listed (and attached to a device) with their bytes: every one of them was freed
//! by the runtime and closed by the test, so each is one the kernel kept. A last line gives the
//! totals and the count once the device itself went. It checks the runtime's side alone: every
//! buffer one import, every import freed, and the copies' bytes; the kernel's count is the probe's
//! answer, printed.
//!
//! What the variables change, each read once (the first line prints them), so that a run can drop
//! one ingredient at a time:
//! - `DMA_BUF_LEAK_ROUNDS` (5), `DMA_BUF_LEAK_CYCLES` (50, a worker's cycles a round),
//!   `DMA_BUF_LEAK_DEPTH` (4, the buffers a worker keeps live).
//! - `DMA_BUF_LEAK_SOURCE`: `list` (the default, `UDMABUF_CREATE_LIST` over scattered runs) or
//!   `single` (one `UDMABUF_CREATE` of one run of the file, the view one mapping of it).
//! - `DMA_BUF_LEAK_MIN_KIB` (16), `DMA_BUF_LEAK_MAX_MIB` (64): the spread of lengths.
//! - `DMA_BUF_LEAK_FREE`: `mixed` (the default: a fifth left unused, the rest freed waited for or
//!   in flight, half each), `waited`, `inflight`, `unused`, or `idle` (each copy waited for, then
//!   the device waited idle, then the buffer freed).
//! - `DMA_BUF_LEAK_BUSY` (1): the other work; 0 leaves it out.
//! - `DMA_BUF_LEAK_QUEUES`: `shared` (the default, one command queue for all) or `own` (one each).
//! - `DMA_BUF_LEAK_WRITE` (0): 1 has each copy write the bytes it read back over the buffer's
//!   ends, so that the GPU writes the imported pages too; no copy is checked then, as buffers
//!   share pages.
//! - `DMA_BUF_LEAK_LINGER` (0): 1 keeps a quarter of the used buffers past the device's idle
//!   window (2.5 s), then copies from each again and frees it, so the device's residency thread
//!   releases their imports while other threads import, and the copy imports the kept dma-buf
//!   again (`host_residency.rs`).
//! - `DMA_BUF_LEAK_NAME` (1): 0 leaves the dma-bufs unnamed, and the kernel's count is then every
//!   udmabuf on its list less those listed before the first round.
//! - `DMA_BUF_LEAK_SUDO` (0): 1 reads the list through `sudo -n cat` where this user cannot read
//!   it. Where the list cannot be read the lines say so and give the runtime's side alone.
//! - `DMA_BUF_LEAK_FILE_MIB` (512): the guest RAM's size. `DMA_BUF_LEAK_SEED` (1).
//! - `DMA_BUF_LEAK_DEADLINE_S` (60): the probe's own bound on its rounds' time. Once it passed,
//!   each worker makes no more buffers, the round ends as any round does (its buffers freed, its
//!   command buffers waited for, its checks made) and no round follows; a line says so
//!   (`DMA-BUF-LEAK ... deadline`). Each wait for a command buffer is bounded too (60 s, a
//!   failure). A step that never returns (a driver's call) holds its round past both, so the
//!   rounds run on a thread of their own, and where they still run [`TAIL`] past the deadline the
//!   test fails at once, naming the workers' steps in progress and how long each has run, and
//!   leaves that thread to the process's end. Seen: on an RTX 5080 Laptop GPU (NVIDIA 615.71.09),
//!   beside the runtime suite, the eight threads' first round took 379 s; on an RTX 3090 (NVIDIA
//!   620.70.06) beside the suite, and on an RTX 2080 (620.70.06) alone, the four threads' first
//!   round never ended, a worker spinning inside the driver while the others waited for it.
//! - `DMA_BUF_LEAK_SERIAL`: `none` (the default), `imports` (one lock across the threads around each
//!   buffer's making, which imports its dma-buf, and each buffer's drop, which frees an import the
//!   GPU no longer uses) or `all` (that lock around the workers' commits too), to see whether the
//!   kernel keeps fewer imports where they never overlap. A free the runtime defers past a command
//!   buffer in flight runs on its own thread, outside the lock (`DMA_BUF_LEAK_FREE=waited` has none).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, or the device lists no dma-buf route, it says so and
//! passes without running.

use super::dma_buf_import_cycles::device_idle;
use super::host_residency::{host_page, udmabuf, udmabuf_list, udmabuf_openable, Scratch};
use super::merge_scratch::read;
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::device::lock;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DmaBufChunk, DmaBufRequest,
    HostMemory, ResourceOptions, WorkCounts,
};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const TEST: &str = "DMA-BUF-LEAK";
/// The kernel's list of dma-bufs.
const BUFINFO: &str = "/sys/kernel/debug/dma_buf/bufinfo";
/// udmabuf's default limits: the runs of one list, and the bytes of one dma-buf.
const MOST_RUNS: u64 = 1024;
const MOST_BYTES: u64 = 64 << 20;
/// The bytes each copy reads from each end of a buffer.
const PROBE: u64 = 4096;
/// The bytes of each buffer of a worker's other work, and the copies each of its command buffers
/// makes.
const WORK_BYTES: u64 = 16 << 20;
const WORK_COPIES: usize = 4;
/// The other work's command buffers a worker leaves in flight at most.
const MOST_PENDING: usize = 16;
/// How long a lingering buffer is left idle: past the device's idle window
/// (`host_residency::IDLE`).
const LINGER: Duration = Duration::from_millis(2500);
/// How long a round waits for the kernel's list to settle.
const KERNEL_SETTLE: Duration = Duration::from_secs(2);
const WAIT: Duration = Duration::from_secs(60);
/// How long past the deadline the rounds may run on, ending the round the deadline cut (its
/// frees, and its waits for command buffers, each bounded by [`WAIT`]), before the test fails.
const TAIL: Duration = Duration::from_secs(120);

/// `DMA_BUF_SET_NAME`, `_IOW('b', 1, const char *)` in the generic ioctl encoding.
const DMA_BUF_SET_NAME: u64 =
    (1 << 30) | ((size_of::<*const u8>() as u64) << 16) | ((b'b' as u64) << 8) | 1;

/// How a buffer is freed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Free {
    /// A fifth unused, the rest waited for or in flight, half each.
    Mixed,
    Waited,
    InFlight,
    Unused,
    /// Waited for, then the device waited idle.
    Idle,
}

/// How much of the probe's kernel work one lock serialises (`DMA_BUF_LEAK_SERIAL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Serial {
    /// No lock.
    None,
    /// Each buffer's making and drop.
    Imports,
    /// Those and the workers' commits.
    All,
}

/// The lock [`Serial`] takes.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The lock where `knobs` serialise `level`'s work.
fn serial(knobs: &Knobs, level: Serial) -> Option<std::sync::MutexGuard<'static, ()>> {
    (knobs.serial >= level).then(|| {
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    })
}

/// The variables' settings (the module docs).
#[derive(Clone, Debug)]
struct Knobs {
    rounds: usize,
    cycles: usize,
    depth: usize,
    list: bool,
    smallest: u64,
    largest: u64,
    free: Free,
    busy: bool,
    shared_queue: bool,
    linger: bool,
    write: bool,
    named: bool,
    sudo: bool,
    file: u64,
    seed: u64,
    serial: Serial,
    deadline: Duration,
}

impl Knobs {
    fn read() -> Knobs {
        let number = |name: &str, default: u64| -> u64 {
            match std::env::var(name) {
                Ok(value) => value
                    .parse()
                    .unwrap_or_else(|_| panic!("{TEST}: {name}={value} is no number")),
                Err(_) => default,
            }
        };
        let word = |name: &str, default: &str| std::env::var(name).unwrap_or(default.to_owned());
        let page = host_page() as u64;
        let free = match word("DMA_BUF_LEAK_FREE", "mixed").as_str() {
            "mixed" => Free::Mixed,
            "waited" => Free::Waited,
            "inflight" => Free::InFlight,
            "unused" => Free::Unused,
            "idle" => Free::Idle,
            other => {
                panic!("{TEST}: DMA_BUF_LEAK_FREE={other}: mixed, waited, inflight, unused or idle")
            }
        };
        let list = match word("DMA_BUF_LEAK_SOURCE", "list").as_str() {
            "list" => true,
            "single" => false,
            other => panic!("{TEST}: DMA_BUF_LEAK_SOURCE={other}: list or single"),
        };
        let shared_queue = match word("DMA_BUF_LEAK_QUEUES", "shared").as_str() {
            "shared" => true,
            "own" => false,
            other => panic!("{TEST}: DMA_BUF_LEAK_QUEUES={other}: shared or own"),
        };
        let largest = (number("DMA_BUF_LEAK_MAX_MIB", 64) << 20).min(MOST_BYTES);
        let knobs = Knobs {
            rounds: number("DMA_BUF_LEAK_ROUNDS", 5) as usize,
            cycles: number("DMA_BUF_LEAK_CYCLES", 50) as usize,
            depth: number("DMA_BUF_LEAK_DEPTH", 4).max(1) as usize,
            list,
            smallest: (number("DMA_BUF_LEAK_MIN_KIB", 16) << 10)
                .next_multiple_of(page)
                .min(largest),
            largest: largest / page * page,
            free,
            busy: number("DMA_BUF_LEAK_BUSY", 1) != 0,
            shared_queue,
            linger: number("DMA_BUF_LEAK_LINGER", 0) != 0,
            write: number("DMA_BUF_LEAK_WRITE", 0) != 0,
            named: number("DMA_BUF_LEAK_NAME", 1) != 0,
            sudo: number("DMA_BUF_LEAK_SUDO", 0) != 0,
            file: (number("DMA_BUF_LEAK_FILE_MIB", 512) << 20).max(MOST_BYTES),
            seed: number("DMA_BUF_LEAK_SEED", 1),
            serial: match word("DMA_BUF_LEAK_SERIAL", "none").as_str() {
                "none" => Serial::None,
                "imports" => Serial::Imports,
                "all" => Serial::All,
                other => panic!("{TEST}: DMA_BUF_LEAK_SERIAL={other}: none, imports or all"),
            },
            deadline: Duration::from_secs(number("DMA_BUF_LEAK_DEADLINE_S", 60)),
        };
        assert!(
            knobs.smallest >= page && knobs.smallest <= knobs.largest,
            "{TEST}: the lengths' spread"
        );
        knobs
    }
}

/// A xorshift64* generator: each worker's own, seeded from the run's seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Rng {
        Rng((seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15)).max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A number below `bound`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound.max(1)
    }

    /// A number in `0.0..1.0`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The guest RAM: a sealed anonymous `memfd`, each 4-byte word holding its own offset.
struct Ram {
    fd: Arc<OwnedFd>,
    pages: u64,
}

impl Ram {
    fn new(bytes: u64) -> Ram {
        // SAFETY: a new anonymous file, owned at once, then sized and sealed.
        let fd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-dma-buf-leak".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            );
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            let fd = OwnedFd::from_raw_fd(fd);
            assert_eq!(libc::ftruncate(fd.as_raw_fd(), bytes as libc::off_t), 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
            fd
        };
        const BLOCK: u64 = 1 << 20;
        for block in 0..bytes / BLOCK {
            let words: Vec<u8> = (0..BLOCK / 4)
                .flat_map(|word| ((block * BLOCK + word * 4) as u32).to_le_bytes())
                .collect();
            // SAFETY: `words` is readable for its length.
            let written = unsafe {
                libc::pwrite(
                    fd.as_raw_fd(),
                    words.as_ptr().cast(),
                    words.len(),
                    (block * BLOCK) as libc::off_t,
                )
            };
            assert_eq!(written, words.len() as isize, "pwrite");
        }
        Ram {
            fd: Arc::new(fd),
            pages: bytes / host_page() as u64,
        }
    }
}

/// What every worker shares.
struct Probe {
    device: Device,
    ram: Ram,
    knobs: Knobs,
    /// The name each dma-buf is given (`DMA_BUF_SET_NAME`), where they are named.
    tag: CString,
    /// A naming that failed, said once the round ends.
    unnamed: Arc<AtomicBool>,
    tally: Tally,
    /// When the first round began: the deadline counts from it ([`Knobs::deadline`]).
    began: Instant,
    /// The workers that stopped a round short for the deadline.
    cut: AtomicU64,
}

impl Probe {
    /// Whether the probe's deadline passed: the worker asking makes no more buffers this round,
    /// and no round follows.
    fn past_deadline(&self) -> bool {
        let past = self.began.elapsed() >= self.knobs.deadline;
        if past {
            self.cut.fetch_add(1, Ordering::Relaxed);
        }
        past
    }
}

/// The round's buffers by how each went.
#[derive(Default)]
struct Tally {
    made: AtomicU64,
    unused: AtomicU64,
    waited: AtomicU64,
    in_flight: AtomicU64,
    /// Freed in flight where the copy had completed meanwhile.
    in_flight_done: AtomicU64,
    idle: AtomicU64,
    lingered: AtomicU64,
    checked: AtomicU64,
    /// The workers' time in their buffers' makings, their frees and their waits for command
    /// buffers.
    making: Timed,
    freeing: Timed,
    waiting: Timed,
    /// Those steps in progress, which the test names where the rounds outrun their bound.
    steps: Arc<Steps>,
}

/// Time spent in one kind of step, in all and at the longest, in nanoseconds.
#[derive(Default)]
struct Timed {
    total: AtomicU64,
    longest: AtomicU64,
}

impl Timed {
    /// Run `step`, named `what` in `steps` while it runs, its time counted.
    fn time<T>(&self, steps: &Steps, what: &'static str, step: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let done = steps.run(what, step);
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.total.fetch_add(nanos, Ordering::Relaxed);
        self.longest.fetch_max(nanos, Ordering::Relaxed);
        done
    }

    /// The time so far, in all and at the longest, in milliseconds, said and set back to zero.
    fn take(&self) -> String {
        let total = self.total.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let longest = self.longest.swap(0, Ordering::Relaxed) as f64 / 1e6;
        format!("{total:.0} ms (longest {longest:.1} ms)")
    }
}

/// The workers' steps in progress, each named, with when it began.
#[derive(Default)]
struct Steps {
    next: AtomicU64,
    running: Mutex<BTreeMap<u64, (&'static str, Instant)>>,
}

impl Steps {
    /// Run `step`, listed as `what` while it runs.
    fn run<T>(&self, what: &'static str, step: impl FnOnce() -> T) -> T {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.running).insert(id, (what, Instant::now()));
        let done = step();
        lock(&self.running).remove(&id);
        done
    }

    /// The steps in progress, each with how long it has run, the longest first.
    fn said(&self) -> String {
        let running = lock(&self.running);
        let mut steps: Vec<_> = (running.values())
            .map(|(what, began)| (began.elapsed(), *what))
            .collect();
        steps.sort_by_key(|&(ran, _)| std::cmp::Reverse(ran));
        match steps.is_empty() {
            true => "no worker's making, free or wait in progress".to_owned(),
            false => (steps.iter())
                .map(|(ran, what)| format!("{what} for {:.1} s", ran.as_secs_f64()))
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

impl Tally {
    fn take(&self) -> [u64; 8] {
        [
            &self.made,
            &self.unused,
            &self.waited,
            &self.in_flight,
            &self.in_flight_done,
            &self.idle,
            &self.lingered,
            &self.checked,
        ]
        .map(|count| count.swap(0, Ordering::Relaxed))
    }
}

/// A buffer the probe made, with its copy's command buffer where it was used.
struct Live {
    buffer: Buffer,
    copy: Option<CommandBuffer>,
    /// Kept past the idle window, then copied from again ([`Knobs::linger`]).
    linger: bool,
    used: Instant,
}

/// A view stitched of `runs` of `fd`'s pages (each a file offset and a length), `length` bytes in
/// all: one reservation, each run mapped over it in order.
fn stitched(fd: &OwnedFd, runs: &[(u64, u64)], length: u64) -> NonNull<u8> {
    // SAFETY: a new reservation of addresses, then each run mapped inside it, over it.
    unsafe {
        let at = libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        );
        assert_ne!(at, libc::MAP_FAILED, "mmap of the view's reservation");
        let mut placed = 0;
        for &(offset, size) in runs {
            let run = libc::mmap(
                at.cast::<u8>().add(placed as usize).cast(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd.as_raw_fd(),
                offset as libc::off_t,
            );
            assert_eq!(
                run,
                at.cast::<u8>().add(placed as usize).cast(),
                "mmap of a run: {}",
                std::io::Error::last_os_error()
            );
            placed += size;
        }
        NonNull::new(at.cast()).unwrap()
    }
}

/// The runs of `runs` that hold bytes `from..from + length` of their concatenation.
fn runs_within(runs: &[(u64, u64)], from: u64, length: u64) -> Vec<(u64, u64)> {
    let end = from + length;
    let mut at = 0;
    let mut within = Vec::new();
    for &(offset, size) in runs {
        let (start, stop) = (from.max(at), end.min(at + size));
        if start < stop {
            within.push((offset + start - at, stop - start));
        }
        at += size;
    }
    within
}

impl Probe {
    /// A buffer over a new view of the RAM's pages, its length and runs drawn from `rng`.
    fn buffer(&self, rng: &mut Rng) -> Buffer {
        let page = host_page() as u64;
        let (low, high) = (
            (self.knobs.smallest as f64).ln(),
            (self.knobs.largest as f64).ln(),
        );
        let drawn = (low + rng.unit() * (high - low)).exp() as u64;
        let length = (drawn / page * page).clamp(self.knobs.smallest, self.knobs.largest);
        let pages = length / page;
        let count = match self.knobs.list {
            true => 1 + rng.below(pages.min(MOST_RUNS)),
            false => 1,
        };
        let runs: Vec<(u64, u64)> = (0..count)
            .map(|index| {
                let run = pages / count + u64::from(index < pages % count);
                (rng.below(self.ram.pages - run + 1) * page, run * page)
            })
            .collect();
        let view = stitched(&self.ram.fd, &runs, length);
        let base = view.as_ptr() as u64;
        let fd = Arc::clone(&self.ram.fd);
        let list = self.knobs.list;
        let tag = self.knobs.named.then(|| self.tag.clone());
        let unnamed = Arc::clone(&self.unnamed);
        let source = Arc::new(move |request: DmaBufRequest| {
            let from = (request.start.as_ptr() as u64)
                .checked_sub(base)
                .filter(|from| from + request.length <= length)
                .ok_or_else(|| std::io::Error::other("pages outside the view"))?;
            let within = runs_within(&runs, from, request.length);
            let chunk = match list {
                true => udmabuf_list(&fd, &within)?,
                false => udmabuf(&fd, within[0].0, within[0].1)?,
            };
            if let Some(tag) = &tag {
                // SAFETY: the ioctl reads the name, a string that lives for the call.
                let named =
                    unsafe { libc::ioctl(chunk.as_raw_fd(), DMA_BUF_SET_NAME as _, tag.as_ptr()) };
                if named != 0 {
                    eprintln!(
                        "{TEST}: DMA_BUF_SET_NAME: {}",
                        std::io::Error::last_os_error()
                    );
                    unnamed.store(true, Ordering::Relaxed);
                }
            }
            Ok(vec![DmaBufChunk {
                fd: chunk,
                length: request.length,
            }])
        });
        let deallocator: crate::Deallocator = Box::new(|at: NonNull<u8>, length: u64| {
            // SAFETY: the view this buffer's deallocator was given with, which nothing uses now
            // (with no keep-alive, the deallocator waits for the import's free).
            unsafe { libc::munmap(at.as_ptr().cast(), length as usize) };
        });
        // SAFETY: the view stays mapped until the deallocator runs, and the source's dma-buf holds
        // the view's pages, in order.
        let serialised = serial(&self.knobs, Serial::Imports);
        let tally = &self.tally;
        let made = tally.making.time(&tally.steps, "a buffer's making", || {
            // SAFETY: as above.
            unsafe {
                self.device.new_buffer_with_bytes_no_copy(
                    HostMemory::with_dma_buf(view, source),
                    length,
                    ResourceOptions::default(),
                    Some(deallocator),
                )
            }
        });
        let buffer = made.unwrap();
        drop(serialised);
        self.tally.made.fetch_add(1, Ordering::Relaxed);
        buffer
    }
}

/// Encode copies of `PROBE` bytes from each end of `buffer` into `out`, and where `write`, of
/// `out`'s bytes back over them.
fn copy_ends(command_buffer: &CommandBuffer, buffer: &Buffer, out: &Buffer, write: bool) {
    let length = buffer.length();
    let each = PROBE.min(length);
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, each)).unwrap();
    (encoder.copy_from_buffer_to_buffer(buffer, length - each, out, PROBE, each)).unwrap();
    if write {
        (encoder.copy_from_buffer_to_buffer(out, 0, buffer, 0, each)).unwrap();
        (encoder.copy_from_buffer_to_buffer(out, PROBE, buffer, length - each, each)).unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// Wait for `command_buffer`, which must complete without error, the wait timed in `tally`.
fn wait(tally: &Tally, command_buffer: &CommandBuffer) {
    let waited = tally
        .waiting
        .time(&tally.steps, "a wait for a command buffer", || {
            command_buffer.wait_until_completed_for(WAIT)
        });
    waited.unwrap();
    assert_eq!(command_buffer.error(), None, "{TEST}: a command buffer");
}

/// A worker's own: its copies' targets, its other work, and its command buffers left in flight.
struct Worker<'a> {
    probe: &'a Probe,
    queue: &'a CommandQueue,
    rng: Rng,
    /// The target of the copies nothing checks, and of the one a check waits for.
    out: Buffer,
    checked: Buffer,
    work: Option<(Buffer, Buffer)>,
    pending: Vec<CommandBuffer>,
    lingering: Vec<Live>,
}

impl<'a> Worker<'a> {
    /// Drop `buffer`, under the lock where the knobs serialise imports, the drop timed.
    fn release(&self, buffer: Buffer) {
        let _serialised = serial(&self.probe.knobs, Serial::Imports);
        let tally = &self.probe.tally;
        tally
            .freeing
            .time(&tally.steps, "a buffer's free", || drop(buffer));
    }

    fn new(probe: &'a Probe, queue: &'a CommandQueue, stream: u64) -> Worker<'a> {
        let device = &probe.device;
        let shared = || {
            device
                .new_buffer(2 * PROBE, ResourceOptions::shared())
                .unwrap()
        };
        let private = || {
            device
                .new_buffer(WORK_BYTES, ResourceOptions::private())
                .unwrap()
        };
        Worker {
            probe,
            queue,
            rng: Rng::new(probe.knobs.seed, stream),
            out: shared(),
            checked: shared(),
            work: probe.knobs.busy.then(|| (private(), private())),
            pending: Vec::new(),
            lingering: Vec::new(),
        }
    }

    /// Make a buffer and, but where it is left unused, commit its copy, unwaited; every second
    /// cycle commit other work first; one cycle in eight wait for the copy and check it.
    fn make(&mut self, cycle: usize) -> Live {
        let probe = self.probe;
        if cycle % 2 == 0 {
            self.other_work();
        }
        let buffer = probe.buffer(&mut self.rng);
        let unused = match probe.knobs.free {
            Free::Unused => true,
            Free::Mixed => self.rng.below(5) == 0,
            _ => false,
        };
        let copy = (!unused).then(|| {
            let check = cycle % 8 == 0 && !probe.knobs.write;
            let command_buffer = self.queue.command_buffer().unwrap();
            copy_ends(
                &command_buffer,
                &buffer,
                if check { &self.checked } else { &self.out },
                probe.knobs.write,
            );
            {
                let _serialised = serial(&self.probe.knobs, Serial::All);
                command_buffer.commit().unwrap();
            }
            if check {
                wait(&probe.tally, &command_buffer);
                self.check(&buffer);
            }
            command_buffer
        });
        let linger = probe.knobs.linger && copy.is_some() && self.rng.below(4) == 0;
        Live {
            buffer,
            copy,
            linger,
            used: Instant::now(),
        }
    }

    /// Check the waited copy of `buffer`'s ends against its view's bytes.
    fn check(&self, buffer: &Buffer) {
        let length = buffer.length();
        let each = PROBE.min(length) as usize;
        let view = buffer.contents().unwrap();
        // SAFETY: the view's bytes, mapped while the buffer lives, which the GPU only reads.
        let (first, last) = unsafe {
            (
                std::slice::from_raw_parts(view.as_ptr(), each).to_vec(),
                std::slice::from_raw_parts(view.as_ptr().add(length as usize - each), each)
                    .to_vec(),
            )
        };
        let copied = read(&self.checked, 2 * PROBE);
        assert!(
            copied[..each] == first[..] && copied[PROBE as usize..][..each] == last[..],
            "{TEST}: the copy of a buffer of {length} bytes reads its view's bytes"
        );
        self.probe.tally.checked.fetch_add(1, Ordering::Relaxed);
    }

    /// Commit a command buffer of other work, unwaited.
    fn other_work(&mut self) {
        let Some((a, b)) = &self.work else {
            return;
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        for copy in 0..WORK_COPIES {
            let (from, to) = if copy % 2 == 0 { (a, b) } else { (b, a) };
            (encoder.copy_from_buffer_to_buffer(from, 0, to, 0, WORK_BYTES)).unwrap();
        }
        encoder.end_encoding().unwrap();
        {
            let _serialised = serial(&self.probe.knobs, Serial::All);
            command_buffer.commit().unwrap();
        }
        self.pending.push(command_buffer);
        self.prune();
    }

    /// Let go of the command buffers that completed; wait for the oldest past [`MOST_PENDING`].
    fn prune(&mut self) {
        self.pending.retain(|command_buffer| {
            let done = command_buffer.status() == CommandBufferStatus::Completed;
            if done {
                assert_eq!(command_buffer.error(), None, "{TEST}: a command buffer");
            }
            !done
        });
        while self.pending.len() > MOST_PENDING {
            wait(&self.probe.tally, &self.pending.remove(0));
        }
    }

    /// Free `live` as the knobs say, or keep it lingering.
    fn free(&mut self, live: Live) {
        if live.linger {
            self.lingering.push(live);
            return;
        }
        let tally = &self.probe.tally;
        let Some(copy) = live.copy else {
            self.release(live.buffer);
            tally.unused.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let how = match self.probe.knobs.free {
            Free::Mixed => match self.rng.below(2) {
                0 => Free::Waited,
                _ => Free::InFlight,
            },
            how => how,
        };
        match how {
            Free::InFlight => {
                let done = copy.status() == CommandBufferStatus::Completed;
                self.release(live.buffer);
                match done {
                    true => tally.in_flight_done.fetch_add(1, Ordering::Relaxed),
                    false => tally.in_flight.fetch_add(1, Ordering::Relaxed),
                };
                self.pending.push(copy);
                self.prune();
            }
            Free::Idle => {
                wait(tally, &copy);
                drop(copy);
                device_idle(&self.probe.device);
                self.release(live.buffer);
                tally.idle.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                wait(tally, &copy);
                self.release(live.buffer);
                tally.waited.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Copy again from each lingering buffer left idle past [`LINGER`] (all of them where `all`,
    /// after the wait), wait, and free it.
    fn linger(&mut self, all: bool) {
        let mut index = 0;
        while index < self.lingering.len() {
            let idle = self.lingering[index].used.elapsed();
            if idle < LINGER && !all {
                index += 1;
                continue;
            }
            let live = self.lingering.swap_remove(index);
            std::thread::sleep(LINGER.saturating_sub(idle));
            if let Some(copy) = &live.copy {
                wait(&self.probe.tally, copy);
            }
            let command_buffer = self.queue.command_buffer().unwrap();
            copy_ends(
                &command_buffer,
                &live.buffer,
                &self.out,
                self.probe.knobs.write,
            );
            {
                let _serialised = serial(&self.probe.knobs, Serial::All);
                command_buffer.commit().unwrap();
            }
            wait(&self.probe.tally, &command_buffer);
            self.release(live.buffer);
            self.probe.tally.lingered.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The round's end: free what is left, the lingering ones last, and wait for what is in
    /// flight.
    fn finish(&mut self, live: Vec<Live>) {
        for live in live {
            self.free(live);
        }
        self.linger(true);
        for command_buffer in self.pending.drain(..) {
            wait(&self.probe.tally, &command_buffer);
        }
    }

    /// `cycles` cycles of making a buffer and freeing one picked at random past the depth, or as
    /// many as run before the probe's deadline ([`Probe::past_deadline`]).
    fn cycles(&mut self, cycles: usize) {
        let mut live: Vec<Live> = Vec::new();
        for cycle in 0..cycles {
            if self.probe.past_deadline() {
                break;
            }
            live.push(self.make(cycle));
            while live.len() > self.probe.knobs.depth {
                let index = self.rng.below(live.len() as u64) as usize;
                let picked = live.swap_remove(index);
                self.free(picked);
            }
            self.linger(false);
        }
        self.finish(live);
    }
}

/// The kernel's count of the probe's dma-bufs: listed, attached to a device, and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Listed {
    count: u64,
    attached: u64,
    bytes: u64,
}

/// The dma-bufs `text` (the kernel's `bufinfo`) lists whose exporter and name `ours` picks.
fn listed(text: &str, ours: impl Fn(&str, &str) -> bool) -> Listed {
    let mut listed = Listed::default();
    let mut current = false;
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let size = fields
            .first()
            .and_then(|size| size.trim().parse::<u64>().ok());
        if let (false, Some(size), true) = (line.starts_with('\t'), size, fields.len() >= 7) {
            current = ours(fields[4].trim(), fields[6..].join("\t").trim());
            if current {
                listed.count += 1;
                listed.bytes += size;
            }
        } else if let Some(attached) = (line.strip_prefix("Total "))
            .and_then(|rest| rest.strip_suffix(" devices attached"))
            .and_then(|count| count.trim().parse::<u64>().ok())
        {
            if current && attached > 0 {
                listed.attached += 1;
            }
            current = false;
        }
    }
    listed
}

/// The kernel's list, where this user (or `sudo -n`, where `knobs` say) can read it.
fn bufinfo(knobs: &Knobs) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(BUFINFO) {
        return Some(text);
    }
    if !knobs.sudo {
        return None;
    }
    let output = std::process::Command::new("sudo")
        .args(["-n", "cat", BUFINFO])
        .output()
        .ok()?;
    (output.status.success()).then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The probe's dma-bufs the kernel lists now: those named `tag`, else (unnamed) every udmabuf.
fn kernel_listed(knobs: &Knobs, tag: &CString) -> Option<Listed> {
    let text = bufinfo(knobs)?;
    let tag = tag.to_str().unwrap();
    Some(match knobs.named {
        true => listed(&text, |_, name| name == tag),
        false => listed(&text, |exporter, _| exporter == "udmabuf"),
    })
}

impl Probe {
    /// The probe's dma-bufs the kernel lists now ([`kernel_listed`]).
    fn listed(&self) -> Option<Listed> {
        kernel_listed(&self.knobs, &self.tag)
    }

    /// The kernel's list once the probe's dma-bufs on it stop falling toward `floor`, and how
    /// long that took.
    fn listed_settled(&self, floor: u64) -> Option<(Listed, Duration)> {
        let start = Instant::now();
        let mut now = self.listed()?;
        while now.count > floor && start.elapsed() < KERNEL_SETTLE {
            std::thread::sleep(Duration::from_millis(50));
            now = self.listed()?;
        }
        Some((now, start.elapsed()))
    }
}

/// The device's counts the lines give.
#[derive(Clone, Copy)]
struct Counts {
    made: u64,
    live: u64,
    released: u64,
    again: u64,
}

impl Counts {
    fn of(counts: &WorkCounts) -> Counts {
        Counts {
            made: counts.dma_buf_imports,
            live: counts.dma_buf_memories_live,
            released: counts.host_imports_released,
            again: counts.host_imports_reimported,
        }
    }
}

/// How the probe's workers run.
#[derive(Clone, Copy)]
enum Plan {
    /// That many workers at once.
    Threads(usize),
    /// One importer handing each buffer to one freer.
    Handoff,
}

/// Run `plan`'s rounds ([`rounds`]) on a thread of their own, bounded: where they still run
/// [`TAIL`] past the probe's deadline, the test fails, naming the workers' steps in progress, and
/// leaves that thread to the process's end.
fn probe(way: &str, plan: Plan) {
    let deadline = Knobs::read().deadline;
    let steps = Arc::new(Steps::default());
    let (ended, ends) = mpsc::channel::<()>();
    let runner = {
        let (way, steps) = (way.to_owned(), Arc::clone(&steps));
        std::thread::spawn(move || {
            // Dropped as the rounds end, whether they return or panic.
            let _ended = ended;
            rounds(&way, plan, steps);
        })
    };
    if let Err(mpsc::RecvTimeoutError::Timeout) = ends.recv_timeout(deadline + TAIL) {
        panic!(
            "{TEST}: {way}: the rounds still run {} s past the probe's deadline of {} s \
             (DMA_BUF_LEAK_DEADLINE_S); in progress: {}",
            TAIL.as_secs(),
            deadline.as_secs(),
            steps.said()
        );
    }
    if let Err(panic) = runner.join() {
        std::panic::resume_unwind(panic);
    }
}

/// Run `plan`'s rounds, the workers' steps in progress listed in `steps`; the lines.
fn rounds(way: &str, plan: Plan, steps: Arc<Steps>) {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        println!(
            "{TEST} on {}: {way}: no dma-buf route: not run",
            device.name()
        );
        return;
    }
    let knobs = Knobs::read();
    let probe = Probe {
        ram: Ram::new(knobs.file),
        tag: CString::new(format!("m2vleak{}", std::process::id())).unwrap(),
        unnamed: Arc::default(),
        tally: Tally {
            steps,
            ..Tally::default()
        },
        began: Instant::now(),
        cut: AtomicU64::new(0),
        device,
        knobs,
    };
    let device = &probe.device;
    let floor = probe.listed().map_or(0, |listed| match probe.knobs.named {
        true => 0,
        false => listed.count,
    });
    // The control: the kernel's list shows a buffer's dma-buf while it is imported, so that a
    // count of 0 after a round is the kernel's answer, not a name the list never shows.
    let control = probe.listed().map(|_| {
        let live = Counts::of(&device.work_counts()).live;
        let buffer = probe.buffer(&mut Rng::new(probe.knobs.seed, u64::MAX));
        let imported = probe.listed().unwrap();
        drop(buffer);
        while Counts::of(&device.work_counts()).live > live {
            std::thread::sleep(Duration::from_millis(10));
        }
        let (freed, _) = probe.listed_settled(floor).unwrap();
        probe.tally.take();
        assert!(
            imported.count > floor,
            "{TEST}: the kernel's list shows no dma-buf of the probe's while one is imported"
        );
        (imported, freed)
    });
    println!(
        "{TEST} on {}: {way}: {:?}; {}-byte pages; the kernel's list {}",
        device.name(),
        probe.knobs,
        host_page(),
        match control {
            Some((imported, freed)) => format!(
                "read, {floor} of the probe's kind listed before; the control: one buffer's \
                 dma-buf listed while imported ({} listed, {} attached to a device), {} after \
                 its free",
                imported.count, imported.attached, freed.count
            ),
            None => format!(
                "unreadable ({BUFINFO}: run as a user who reads it, or with DMA_BUF_LEAK_SUDO=1 \
                 where sudo asks no password)"
            ),
        }
    );
    let queues: Vec<CommandQueue> = match (plan, probe.knobs.shared_queue) {
        (Plan::Threads(threads), false) => (0..threads)
            .map(|_| device.new_command_queue().unwrap())
            .collect(),
        (Plan::Handoff, false) => (0..2)
            .map(|_| device.new_command_queue().unwrap())
            .collect(),
        (_, true) => vec![device.new_command_queue().unwrap()],
    };
    let queue = |index: usize| &queues[index % queues.len()];
    let start = Counts::of(&device.work_counts());
    let mut kept_total = Some(0);
    for round in 1..=probe.knobs.rounds {
        let round_began = Instant::now();
        let before = Counts::of(&device.work_counts());
        let listed_before = probe.listed();
        let cycles = probe.knobs.cycles;
        std::thread::scope(|scope| match plan {
            Plan::Threads(threads) => {
                for thread in 0..threads {
                    let probe = &probe;
                    let queue = queue(thread);
                    let stream = (round * 64 + thread) as u64;
                    scope.spawn(move || Worker::new(probe, queue, stream).cycles(cycles));
                }
            }
            Plan::Handoff => {
                let (send, receive) = mpsc::sync_channel::<Live>(probe.knobs.depth);
                let probe = &probe;
                let (importer, freer) = (queue(0), queue(1));
                scope.spawn(move || {
                    let mut worker = Worker::new(probe, importer, (round * 64) as u64);
                    for cycle in 0..cycles {
                        if probe.past_deadline() {
                            break;
                        }
                        let live = worker.make(cycle);
                        send.send(live).unwrap();
                    }
                    worker.finish(Vec::new());
                });
                scope.spawn(move || {
                    let mut worker = Worker::new(probe, freer, (round * 64 + 1) as u64);
                    let mut held: Vec<Live> = Vec::new();
                    for live in receive {
                        held.push(live);
                        if held.len() > worker.rng.below(probe.knobs.depth as u64) as usize {
                            let index = worker.rng.below(held.len() as u64) as usize;
                            let picked = held.swap_remove(index);
                            worker.free(picked);
                        }
                        worker.linger(false);
                    }
                    worker.finish(held);
                });
            }
        });
        let settle = Instant::now();
        let after = loop {
            let now = Counts::of(&device.work_counts());
            if now.live <= before.live || settle.elapsed() > WAIT {
                break now;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(
            !probe.unnamed.load(Ordering::Relaxed),
            "{TEST}: a dma-buf could not be named: run with DMA_BUF_LEAK_NAME=0"
        );
        let [made, unused, waited, in_flight, in_flight_done, idle, lingered, checked] =
            probe.tally.take();
        let times = format!(
            "{:.1} s; makings {}, frees {}, waits {}",
            round_began.elapsed().as_secs_f64(),
            probe.tally.making.take(),
            probe.tally.freeing.take(),
            probe.tally.waiting.take()
        );
        let imports = after.made - before.made;
        let freed = imports - (after.live - before.live);
        let floor_now = listed_before.map_or(floor, |listed| listed.count);
        let kernel = match probe.listed_settled(floor_now) {
            Some((listed, took)) => {
                let kept = listed.count.saturating_sub(floor_now);
                kept_total = kept_total.map(|total| total + kept);
                format!(
                    "kernel: the probe's dma-bufs listed {} ({} attached to a device, {} bytes), \
                     kept this round {kept} of {imports} ({:.1}%), settled in {} ms",
                    listed.count,
                    listed.attached,
                    listed.bytes,
                    kept as f64 * 100.0 / imports.max(1) as f64,
                    took.as_millis()
                )
            }
            None => {
                kept_total = None;
                "kernel: unread".to_owned()
            }
        };
        println!(
            "{TEST} on {}: {way}: round {round}/{}: {times}; buffers {made} (unused {unused}, \
             freed waited for {waited}, in flight {in_flight}, in flight but done \
             {in_flight_done}, after the device idled {idle}, lingered {lingered}; copies checked \
             {checked}); runtime: imports made {imports}, freed {freed}, live {}, released while \
             idle {}, imported again {}; {kernel}",
            device.name(),
            probe.knobs.rounds,
            after.live,
            after.released - before.released,
            after.again - before.again,
        );
        assert_eq!(
            (imports, freed),
            (made, made),
            "{TEST}: {way}: each buffer one import, each import freed"
        );
        assert!(
            checked > 0 || made < 8 || probe.knobs.write,
            "{TEST}: {way}: no copy checked"
        );
        if probe.cut.load(Ordering::Relaxed) > 0 {
            println!(
                "{TEST} on {}: {way}: the probe's deadline of {} s passed in round {round}: its \
                 workers stopped making buffers there, and no round follows \
                 (DMA_BUF_LEAK_DEADLINE_S)",
                device.name(),
                probe.knobs.deadline.as_secs()
            );
            break;
        }
    }
    let imports = Counts::of(&device.work_counts()).made - start.made;
    let name = device.name().to_string();
    drop(queues);
    let Probe {
        device,
        ram,
        knobs,
        tag,
        ..
    } = probe;
    drop(device);
    // The device's teardown, then a moment for the kernel's deferred frees.
    std::thread::sleep(Duration::from_millis(500));
    let gone = match kernel_listed(&knobs, &tag) {
        Some(listed) => format!(
            "once the device went, the probe's dma-bufs listed {} ({} attached to a device)",
            listed.count.saturating_sub(floor),
            listed.attached
        ),
        None => "kernel: unread".to_owned(),
    };
    drop(ram);
    let kept = match kept_total {
        Some(kept) => format!(
            "the kernel kept {kept} ({:.1}%)",
            kept as f64 * 100.0 / imports.max(1) as f64
        ),
        None => "the kernel's list unread".to_owned(),
    };
    println!("{TEST} on {name}: {way}: all rounds: imports {imports}, {kept}; {gone}");
}

/// Four workers on one command queue.
#[test]
fn dma_buf_leak_4_threads() {
    probe("4 threads", Plan::Threads(4));
}

/// Eight workers on one command queue.
#[test]
fn dma_buf_leak_8_threads() {
    probe("8 threads", Plan::Threads(8));
}

/// One thread imports the buffers and commits their copies, another frees them.
#[test]
fn dma_buf_leak_one_thread_imports_another_frees() {
    probe("one thread imports, another frees", Plan::Handoff);
}

/// The kernel's list read: the probe's dma-bufs by name, their bytes, and those attached.
#[test]
fn dma_buf_leak_reads_the_kernel_s_list() {
    let text = "\nDma-buf Objects:\n\
        size    \tflags   \tmode    \tcount   \texp_name\tino     \tname\n\
        00016384\t00000002\t00080007\t00000002\tudmabuf\t00000101\tm2vleak7\n\
        \t\tshared fence: drm_sched gfx signalled\n\
        \tAttached Devices:\n\
        \tgpu@0\n\
        Total 1 devices attached\n\n\
        00032768\t00000002\t00080007\t00000001\tudmabuf\t00000102\tm2vleak7\n\
        \tAttached Devices:\n\
        Total 0 devices attached\n\n\
        00004096\t00000002\t00080007\t00000001\tudmabuf\t00000103\t<none>\n\
        \tAttached Devices:\n\
        \tgpu@0\n\
        Total 1 devices attached\n\n\
        00008192\t00000002\t00080007\t00000001\tdrm\t00000104\tm2vleak8\n\
        \tAttached Devices:\n\
        Total 0 devices attached\n\n\
        \nTotal 4 objects, 61440 bytes\n";
    assert_eq!(
        listed(text, |_, name| name == "m2vleak7"),
        Listed {
            count: 2,
            attached: 1,
            bytes: 49152
        }
    );
    assert_eq!(
        listed(text, |exporter, _| exporter == "udmabuf"),
        Listed {
            count: 3,
            attached: 2,
            bytes: 53248
        }
    );
}
