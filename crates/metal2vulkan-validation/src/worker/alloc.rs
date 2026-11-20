//! The worker's counting global allocator.
//!
//! A worker binary installs [`CountingAllocator`] as its `#[global_allocator]`. It counts live
//! heap bytes, keeps a resettable peak, and can refuse allocations above a byte limit, so a row
//! that grows past the 500-MiB budget fails as an allocation failure inside the worker (the
//! process-group RSS watchdog in [`super::process`] stays the backstop).
//!
//! [`measure`] reports the peak live bytes attributable to one closure: the peak during the
//! closure minus the live bytes when it started. The counters are process-global, so measurements
//! are serialized and one is exact only while no other thread allocates; a worker runs one
//! attempt at a time.
//! When no binary installed the allocator the counters never move and [`measure`] reports `None`.
//!
//! [`peak_bytes`] is the high-water mark of live bytes above the budget's baseline (the live bytes
//! at [`enable_budget`], the fixed runtime state the budget does not charge), which no measurement
//! resets: a worker reports it beside the parent's sampled RSS in every result
//! ([`super::report`]), since a sampler polling every 10 ms can miss a short peak that the
//! allocator counts exactly. Under the budget it never exceeds the budget: an allocation past it is
//! refused, uncounted, and aborts the worker.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
/// The high-water mark of [`LIVE_BYTES`] since [`enable_budget`] (since the process started
/// before it); no [`measure`] resets it.
static HIGH_WATER_BYTES: AtomicUsize = AtomicUsize::new(0);
/// The live bytes at [`enable_budget`]: the fixed runtime state the budget does not charge, and
/// the baseline [`peak_bytes`] is reported above (0 before any budget).
static BUDGET_BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIMIT_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
/// One [`measure`] or [`bounded`] at a time: each resets the shared peak.
static MEASURING: Mutex<()> = Mutex::new(());
/// Whether a [`bounded`] body runs: an allocation its limit refuses ends the process.
static BOUNDED: AtomicBool = AtomicBool::new(false);

/// The exit status of a process an allocation took past a [`bounded`] budget: it ends at once
/// (`_exit`), every thread with it, after one line on stderr naming the breach.
pub const BOUNDED_EXIT_CODE: i32 = 87;

/// A [`System`]-backed allocator that counts live bytes, tracks their peak, and enforces the
/// limit set by [`enable_budget`].
pub struct CountingAllocator;

// SAFETY: every method forwards to `System` with the caller's layout; the counters only decide
// whether to forward an allocation and are adjusted to match what `System` did.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !reserve(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded unchanged; `layout` satisfies `GlobalAlloc::alloc`'s contract.
        let allocation = unsafe { System.alloc(layout) };
        if allocation.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !reserve(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: as for `alloc`.
        let allocation = unsafe { System.alloc_zeroed(layout) };
        if allocation.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator (hence by `System`) with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() && !reserve(new_size - layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded unchanged under `GlobalAlloc::realloc`'s contract.
        let allocation = unsafe { System.realloc(ptr, layout, new_size) };
        if allocation.is_null() {
            if new_size > layout.size() {
                LIVE_BYTES.fetch_sub(new_size - layout.size(), Ordering::Relaxed);
            }
        } else if new_size < layout.size() {
            LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        allocation
    }
}

/// Count `bytes` as live unless that would pass the limit; raise the peak.
fn reserve(bytes: usize) -> bool {
    match LIVE_BYTES.try_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
        live.checked_add(bytes)
            .filter(|next| *next <= LIMIT_BYTES.load(Ordering::Relaxed))
    }) {
        Ok(previous) => {
            PEAK_BYTES.fetch_max(previous + bytes, Ordering::Relaxed);
            HIGH_WATER_BYTES.fetch_max(previous + bytes, Ordering::Relaxed);
            true
        }
        Err(_) if BOUNDED.load(Ordering::Relaxed) => breach(),
        Err(_) => false,
    }
}

/// End the process for an allocation past a [`bounded`] budget: one line on stderr, then
/// `_exit(BOUNDED_EXIT_CODE)`, which allocates nothing and runs nothing more of any thread.
fn breach() -> ! {
    const LINE: &[u8] = b"an allocation passed a bounded call's memory budget: the process ends\n";
    #[cfg(unix)]
    // SAFETY: `write` of a static buffer to stderr and `_exit` are async-signal-safe and allocate
    // nothing, so they are sound inside the allocator.
    unsafe {
        libc::write(2, LINE.as_ptr().cast(), LINE.len());
        libc::_exit(BOUNDED_EXIT_CODE)
    }
    #[cfg(not(unix))]
    std::process::abort()
}

/// Live heap bytes counted so far (0 when the allocator is not installed).
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Whether this process counts its allocations (some binary installed [`CountingAllocator`]).
/// Any Rust process has allocated by the time it can ask, so the count is non-zero exactly then.
pub fn installed() -> bool {
    live_bytes() != 0
}

/// The most live heap bytes this process has held at once above the budget's baseline (the live
/// bytes at [`enable_budget`], which the budget does not charge; the whole process before any
/// budget), or `None` when the allocator is not installed. No [`measure`] resets it. An
/// allocation the budget refused never counted, so under [`enable_budget`] this is at most the
/// budget: it is recorded evidence the budget bounds, never an independent gate (an allocation
/// past the budget aborts the worker, a `WorkerExit` budget stop).
pub fn peak_bytes() -> Option<u64> {
    installed().then(|| {
        let peak = HIGH_WATER_BYTES.load(Ordering::Relaxed);
        peak.saturating_sub(BUDGET_BASELINE_BYTES.load(Ordering::Relaxed)) as u64
    })
}

/// Refuse allocations that would take live bytes more than `bytes` above what is live now. The
/// worker calls this once at startup, so its fixed runtime state is not charged to the budget; the
/// same state is the baseline [`peak_bytes`] is reported above, whose high-water mark restarts
/// here.
pub fn enable_budget(bytes: u64) {
    let bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
    let baseline = live_bytes();
    BUDGET_BASELINE_BYTES.store(baseline, Ordering::Relaxed);
    HIGH_WATER_BYTES.store(baseline, Ordering::Relaxed);
    LIMIT_BYTES.store(baseline.saturating_add(bytes), Ordering::Relaxed);
}

/// Run `body` and return its result with the peak live bytes attributable to it (the peak during
/// `body` minus the live bytes at its start), or `None` when the allocator is not installed.
pub fn measure<R>(body: impl FnOnce() -> R) -> (R, Option<u64>) {
    // A panic inside an earlier measurement poisons nothing this lock protects.
    let _measuring = MEASURING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let installed = installed();
    let baseline = live_bytes();
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    let result = body();
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    (
        result,
        installed.then(|| peak.saturating_sub(baseline) as u64),
    )
}

/// Run `body` with at most `bytes` live above what is live at its start, and return its result
/// with the peak live bytes attributable to it, as [`measure`] does (`None` when the allocator is
/// not installed, and then nothing is bounded). An allocation past the budget while `body` runs,
/// on any thread, ends the process ([`BOUNDED_EXIT_CODE`]): the device worker's pipeline makes run
/// under it, so a make past the 500-MiB ceiling ends the worker before the run submits its work.
/// The limit before the call, the process budget's, holds again once `body` returns, and holds
/// inside it where it is the lower.
pub fn bounded<R>(bytes: u64, body: impl FnOnce() -> R) -> (R, Option<u64>) {
    // A panic inside an earlier measurement poisons nothing this lock protects.
    let _measuring = MEASURING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    /// Puts the limit back however `body` ends.
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            BOUNDED.store(false, Ordering::Relaxed);
            LIMIT_BYTES.store(self.0, Ordering::Relaxed);
        }
    }
    let installed = installed();
    let baseline = live_bytes();
    let before = LIMIT_BYTES.load(Ordering::Relaxed);
    let bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
    let restore = Restore(before);
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    if installed {
        LIMIT_BYTES.store(
            before.min(baseline.saturating_add(bytes)),
            Ordering::Relaxed,
        );
        BOUNDED.store(true, Ordering::Relaxed);
    }
    let result = body();
    drop(restore);
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    (
        result,
        installed.then(|| peak.saturating_sub(baseline) as u64),
    )
}

/// Make the system allocator give every large freed buffer back to the system. [`super::run_json`]
/// calls it before a process supervises its first worker.
///
/// A supervisor frees large transient buffers on many threads: a row, its worker's input, its
/// worker's result. glibc's dynamic mmap threshold, raised by the first such free to that
/// buffer's size (up to 32 MiB), then serves the later ones from per-thread arenas, each of which
/// keeps up to twice the threshold free. So the census parent's RSS grew from profile to profile
/// while what it held did not. Setting the threshold turns the dynamic raise off. The value set is
/// glibc's own initial one, so no new size is chosen: a buffer above it is mapped when allocated
/// and unmapped when freed, as glibc does until its first large free. What a supervisor still
/// keeps is per thread arena, not per buffer: its heap's free top, trimmed in whole heap units
/// (a page; 2 MiB on AArch64, where glibc 2.43 sizes heaps in transparent-huge-page units) above
/// glibc's 128-KiB top pad. Arenas are reused, so that does not grow from sweep to sweep. Only
/// the calling process changes; a worker execs a fresh allocator. Other allocators keep their
/// own policy, and nothing happens there.
pub fn release_large_frees() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        /// glibc's `DEFAULT_MMAP_THRESHOLD_MIN`, the threshold a process starts with.
        const GLIBC_INITIAL_MMAP_THRESHOLD: libc::c_int = 128 * 1024;
        // SAFETY: `mallopt` sets one allocator parameter under the allocator's own lock; every
        // allocation stays valid, and later ones follow the new threshold.
        unsafe { libc::mallopt(libc::M_MMAP_THRESHOLD, GLIBC_INITIAL_MMAP_THRESHOLD) };
    }
}

/// Give the system allocator's free pages back to the system; a sweep's parent calls it when a
/// sweep starts and when it ends.
///
/// [`release_large_frees`] covers buffers above the threshold. Smaller chunks that a sweep or a
/// profile's report frees stay in the arenas' bins, fragmented between chunks still live: the
/// specialization audit's parent held 48 MiB live at each profile's end while its RSS went 123,
/// 185, 239 MiB. glibc's `malloc_trim` returns every whole free page inside a binned free chunk,
/// in every arena, and the main arena's top (never a thread arena's top: the threshold covers
/// those). Other allocators keep their own policy, and nothing happens there.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free memory to the system; every allocation stays valid.
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The lib test binary counts its allocations, so `measure` is exercised for real. Other test
    // threads allocate and free concurrently, so the assertions leave half the held bytes as slack.
    #[global_allocator]
    static TEST_ALLOCATOR: CountingAllocator = CountingAllocator;

    const HELD: usize = 64 << 20;

    #[test]
    fn measure_reports_the_bytes_a_body_holds_at_once_including_growth() {
        assert!(installed());
        let (len, peak) = measure(|| {
            let held = vec![0u8; HELD];
            std::hint::black_box(&held);
            held.len()
        });
        assert_eq!(len, HELD);
        assert!(peak.unwrap() >= (HELD / 2) as u64, "{peak:?}");

        let (_, peak) = measure(|| {
            let mut grown = Vec::<u8>::with_capacity(16);
            grown.resize(HELD, 0);
            std::hint::black_box(grown.len())
        });
        assert!(peak.unwrap() >= (HELD / 2) as u64, "{peak:?}");
    }

    #[test]
    fn the_process_peak_survives_every_measurement() {
        let held = vec![0u8; HELD];
        std::hint::black_box(&held);
        drop(held);
        let after_release = peak_bytes().unwrap();
        assert!(after_release >= HELD as u64, "{after_release}");
        // A later measurement resets its own peak, never the process's.
        let (len, measured) = measure(|| std::hint::black_box(vec![0u8; 16]).len());
        assert_eq!(len, 16);
        assert!(measured.is_some());
        assert!(peak_bytes().unwrap() >= after_release);
    }

    /// The peak is reported above the budget's baseline, as the budget charges it: the runtime
    /// state live at `enable_budget` and anything held and released before it are not counted.
    /// (An unlimited budget, so no allocation of this process is refused.)
    #[test]
    fn the_peak_is_attributed_above_the_budgets_baseline() {
        let before = vec![0u8; HELD];
        std::hint::black_box(&before);
        drop(before);
        let fixed = vec![0u8; HELD / 4];
        std::hint::black_box(&fixed);
        enable_budget(u64::MAX);
        // Neither the released allocation nor the state live at the call is charged.
        assert!(
            peak_bytes().unwrap() < (HELD / 4) as u64,
            "{:?}",
            peak_bytes()
        );
        let held = vec![0u8; HELD / 2];
        std::hint::black_box(&held);
        let peak = peak_bytes().unwrap();
        assert!(peak >= (HELD / 2) as u64, "{peak}");
        assert!(peak < (HELD / 2 + HELD / 4) as u64, "{peak}");
        drop((held, fixed));
    }

    /// A bounded body measures what it holds, as `measure` does, and the limit it ran under goes
    /// with it: an allocation larger than its budget succeeds once it has returned.
    #[test]
    fn a_bounded_body_measures_and_its_limit_goes_with_it() {
        let (len, peak) = bounded(2 * HELD as u64, || {
            let held = vec![0u8; HELD];
            std::hint::black_box(&held);
            held.len()
        });
        assert_eq!(len, HELD);
        assert!(peak.unwrap() >= (HELD / 2) as u64, "{peak:?}");
        let after = vec![0u8; 4 * HELD];
        assert_eq!(std::hint::black_box(&after).len(), 4 * HELD);
    }

    /// An allocation past a bounded body's budget ends the process with [`BOUNDED_EXIT_CODE`]
    /// and the breach's line, in a fresh process: the fixture allocates twice its budget.
    mod bounded_breach {
        use super::*;
        use std::process::Command;

        const FIXTURE: &str = "worker::alloc::tests::bounded_breach::breach_fixture";
        const MODE: &str = "M2V_TEST_BOUNDED_BREACH";

        /// The fixture: no-op unless [`MODE`] is set.
        #[test]
        fn breach_fixture() {
            if std::env::var_os(MODE).is_none() {
                return;
            }
            let _ = bounded(HELD as u64, || {
                std::hint::black_box(vec![1u8; 2 * HELD]).len()
            });
            println!("the bounded body returned");
        }

        #[test]
        fn an_allocation_past_the_budget_ends_the_process() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FIXTURE, "--nocapture"])
                .env(MODE, "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(
                output.status.code(),
                Some(BOUNDED_EXIT_CODE),
                "{stdout}{stderr}"
            );
            assert!(!stdout.contains("the bounded body returned"), "{stdout}");
            assert!(
                stderr.contains("an allocation passed a bounded call's memory budget"),
                "{stderr}"
            );
        }
    }

    /// A supervisor's freed transient buffers go back to the system once it has supervised a
    /// worker, sweep after sweep. Each sweep is modelled as a census profile's parent: threads each
    /// holding a large buffer at once (a row, its worker's input or result), then freeing it, after
    /// a larger buffer freed first raised glibc's dynamic mmap threshold. Each case runs in a fresh
    /// process, since the policy is process-wide.
    ///
    /// What a supervising process may still keep is per thread arena, not per buffer: the free
    /// top of its heap, which glibc trims in whole heap units and keeps its top pad (128 KiB)
    /// of. A unit is a page, except on AArch64, where glibc 2.43 sizes heaps in 2-MiB
    /// transparent-huge-page units by default, so each arena keeps up to 2 MiB there. A buffer
    /// that creates its thread's arena is placed in that arena's first heap, whatever the
    /// threshold, and that heap is trimmed back to one unit when the buffer is freed. So the
    /// bound is a quarter of a buffer per thread (4 MiB, twice the largest unit), and the arenas
    /// are reused, so a later sweep leaves no more than the second one did.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    mod supervisor_heap {
        use std::process::Command;
        use std::sync::{Arc, Barrier};

        const MIB: usize = 1 << 20;
        const THREADS: usize = 8;
        /// Larger than any heap unit, and with the first buffer's 2 MiB more still under
        /// glibc's 32-MiB cap on the dynamic threshold, so by default every buffer is kept.
        const BUFFER: usize = 16 * MIB;
        const SWEEPS: usize = 4;
        const FIXTURE: &str = "worker::alloc::tests::supervisor_heap::sweeps_fixture";
        const MODE: &str = "M2V_TEST_SUPERVISOR_HEAP";
        const LINE: &str = "M2V-RETAINED ";

        fn resident() -> usize {
            crate::worker::process::own_resident_bytes().unwrap() as usize
        }

        /// The fixture: no-op unless [`MODE`] is set. With `supervised`, it first supervises
        /// one worker through `run_json`, as a census parent does before its first row. It
        /// prints the resident bytes each sweep left above its starting point.
        #[test]
        fn sweeps_fixture() {
            let Ok(mode) = std::env::var(MODE) else {
                return;
            };
            if mode == "supervised" {
                let _ =
                    crate::worker::run_json::<u32>(Command::new("true"), "alloc-test", |_| Ok(()));
            }
            let baseline = resident();
            drop(std::hint::black_box(vec![1u8; BUFFER + 2 * MIB]));
            for _ in 0..SWEEPS {
                let barrier = Arc::new(Barrier::new(THREADS));
                let threads = (0..THREADS)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        std::thread::spawn(move || {
                            let buffer = std::hint::black_box(vec![1u8; BUFFER]);
                            barrier.wait();
                            drop(buffer);
                        })
                    })
                    .collect::<Vec<_>>();
                for thread in threads {
                    thread.join().unwrap();
                }
                println!("{LINE}{}", resident().saturating_sub(baseline));
            }
        }

        /// The resident bytes each sweep of the fixture left, in a fresh process.
        fn retained(mode: &str) -> Vec<usize> {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FIXTURE, "--nocapture"])
                .env(MODE, mode)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success(), "{stdout}");
            let retained = stdout
                .lines()
                .filter_map(|line| line.strip_prefix(LINE))
                .map(|bytes| bytes.parse().unwrap())
                .collect::<Vec<usize>>();
            assert_eq!(retained.len(), SWEEPS, "{stdout}");
            retained
        }

        #[test]
        fn a_supervisors_freed_buffers_go_back_to_the_system_sweep_after_sweep() {
            // The hazard, so this test can see it: by default glibc keeps the freed buffers.
            let kept = retained("default");
            let released = retained("supervised");
            eprintln!("resident bytes left per sweep: default {kept:?}, supervised {released:?}");
            assert!(kept[0] >= THREADS * BUFFER / 2, "{kept:?}");
            assert!(
                released.iter().all(|bytes| *bytes < THREADS * BUFFER / 4),
                "a supervisor kept its freed buffers: {released:?} (default {kept:?})"
            );
            assert!(
                released[2..].iter().all(|bytes| *bytes <= released[1]),
                "a supervisor's memory grew from sweep to sweep: {released:?}"
            );
        }

        /// One fragment of [`fragments_fixture`]'s heap: below the mmap threshold.
        const FRAGMENT: usize = 16 * 1024;
        /// 32 MiB of fragments, within one heap of one arena.
        const FRAGMENTS: usize = 2048;
        /// Every 16th fragment stays live, so 15 of 16 are freed between live ones.
        const KEEP_EVERY: usize = 16;
        const FRAGMENTS_FIXTURE: &str = "worker::alloc::tests::supervisor_heap::fragments_fixture";

        /// The fragments fixture: no-op unless [`MODE`] is set. It allocates [`FRAGMENTS`]
        /// fragments, frees all but every [`KEEP_EVERY`]th, as a sweep's rows and a profile's
        /// report leave small chunks freed between live ones, and with `released` gives the free
        /// pages back as a sweep's end does. It prints the resident bytes left above its start.
        #[test]
        fn fragments_fixture() {
            let Ok(mode) = std::env::var(MODE) else {
                return;
            };
            let baseline = resident();
            let fragments = (0..FRAGMENTS)
                .map(|_| std::hint::black_box(vec![1u8; FRAGMENT]))
                .collect::<Vec<_>>();
            let kept = fragments
                .into_iter()
                .enumerate()
                .filter_map(|(index, fragment)| (index % KEEP_EVERY == 0).then_some(fragment))
                .collect::<Vec<_>>();
            if mode == "released" {
                super::super::release_free_memory();
            }
            println!("{LINE}{}", resident().saturating_sub(baseline));
            drop(std::hint::black_box(kept));
        }

        /// The resident bytes the fragments fixture left, in a fresh process.
        fn fragmented(mode: &str) -> usize {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FRAGMENTS_FIXTURE, "--nocapture"])
                .env(MODE, mode)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success(), "{stdout}");
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(LINE))
                .unwrap_or_else(|| panic!("no measurement in {stdout}"))
                .parse()
                .unwrap()
        }

        /// The release at a sweep's end returns every whole free page inside the allocator's
        /// free chunks, which the fixed threshold does not cover. A freed run is 15 fragments
        /// (240 KiB), so only a partial page at each end of a run can stay: at least half of the
        /// freed bytes go back on any page size up to 64 KiB. Both cases share the process's
        /// other resident bytes, so their difference is what the release returned.
        #[test]
        fn a_sweeps_fragmented_frees_go_back_to_the_system_when_it_ends() {
            let freed = FRAGMENTS * FRAGMENT / KEEP_EVERY * (KEEP_EVERY - 1);
            let kept = fragmented("kept");
            let released = fragmented("released");
            eprintln!("resident bytes left: kept {kept}, released {released} (freed {freed})");
            // The hazard, so this test can see it: freed fragments between live ones stay.
            assert!(kept >= freed / 2, "{kept} of {freed}");
            let returned = kept.saturating_sub(released);
            assert!(
                returned >= freed / 2,
                "the release returned {returned} of {freed} freed bytes ({kept} kept)"
            );
        }
    }
}
