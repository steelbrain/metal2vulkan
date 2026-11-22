//! Budget gates over recorded engine rows.
//!
//! Every engine translation sweep records each row's measured cost (the worker's wall time
//! and process-group RSS, the meter's ledger peak and work units, the counting allocator's peak
//! attributable bytes, the final words, and each strategy's emitted words). [`BudgetReport`]
//! derives the L7 gates from those rows:
//!
//! - **Time and memory:** the slowest row, the p50/p99 attempt time, and the peak worker RSS,
//!   including rows the worker boundary stopped (their wall time and peak RSS up to the stop).
//!   Every row the boundary stopped (`failed timeout`, `failed oom`, `failed worker_exit`) and
//!   every in-process budget refusal (`limit_exceeded budget`, which trips before the ceilings)
//!   fails the gate whatever its measure, because a stop is a breach even when the sample that
//!   caught it is below the ceiling or missing.
//! - **Ledger coverage:** a row fails when its ledger peak is below 0.7 × the allocator peak, which
//!   catches input-scaled growth sites the ledger is not charged for.
//!
//! The output-size gate held each row's words to the old engine's words for the same row. It is
//! deleted with the old engine's census, its baseline; the C3F snapshot holds its final
//! measurement.
//!
//! Pipeline-creation time (over 5 s is a bug) is measured wherever validation creates an engine
//! pipeline on a driver: `corpus-pipeline`'s L4 sweep reports it per row, and the engine
//! candidate's compute executor fails a run whose creation breaches it ([`creation_within`]);
//! [`PIPELINE_CREATION_CEILING`] is the ceiling, and a kernel case may declare a bound of its own
//! for its pipeline (`AuthoredCase::creation_bound`). Each times its own
//! `vkCreate*Pipelines` calls by a [`CreationClock`]: the call's wall time less the time its thread
//! waited, runnable, for a CPU, so a host loaded by other processes does not move the measure by
//! that wait (its threads' turns on the cores the call shares, which slow the call's own work, stay
//! in it: the case's own bound is for those). Both tools turn the drivers' shader disk caches off
//! for their own process first ([`turn_driver_shader_caches_off`]), so a creation they time is a
//! compile, not a cache hit.

use crate::engine::{FailureKind, LimitKind};
use crate::translation_audit::CostRow;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// The ledger must account for at least this fraction (numerator, denominator) of the peak
/// attributable bytes the counting allocator saw.
pub const LEDGER_FLOOR: (u64, u64) = (7, 10);

/// A `vkCreate*Pipelines` call above this on a host class is a bug, handled like the 20-second
/// translation ceiling.
pub const PIPELINE_CREATION_CEILING: Duration = Duration::from_secs(5);

/// A `vkCreate*Pipelines` call of `what` that took `created` (a [`CreationClock`]'s reading, or a
/// time the runtime reports for its own makings, `Execution::pipeline_creation`, which is their
/// wall time), held to `bound`: a case's own (`AuthoredCase::creation_bound`, above the ceiling),
/// or [`PIPELINE_CREATION_CEILING`] where it declares none. Past it is an error, a failing result
/// like a translation over 20 s, never an observation.
pub fn creation_within(what: &str, created: Duration, bound: Duration) -> Result<(), String> {
    if created > bound {
        let of = match bound == PIPELINE_CREATION_CEILING {
            true => format!("the {} s pipeline-creation ceiling", bound.as_secs()),
            false => format!(
                "the case's own pipeline-creation bound of {} ms",
                bound.as_millis()
            ),
        };
        return Err(format!(
            "creating {what} took {} ms, above {of}",
            created.as_millis()
        ));
    }
    Ok(())
}

/// The clock of one `vkCreate*Pipelines` call: started on the calling thread just before the call,
/// read on the same thread just after it ([`CreationClock::read`]), it gives the call's wall time
/// less the time that thread waited, runnable, for a CPU.
///
/// The ceiling bounds the driver's work on our modules, and the drivers measured compile on the
/// calling thread: an RTX 3090's (NVIDIA 615.71.09) creation of a bicubic texture kernel took
/// 1414 ms of wall time and 1405 ms of the thread's CPU time. On a host whose other processes keep
/// the thread off a CPU, the wall time grows by that wait, which is the host's: the same kernel's
/// creation took 13.5 s under a batch's two GPU suites at nextest's default parallelism (a load of
/// 30 to 45 on 16 hardware threads), and passed alone. Every other wait stays in the reading: the
/// call blocked on a lock, a disk read, or work on another thread or in another process (MoltenVK
/// has Metal's compiler service compile each pipeline, which the calling thread waits for blocked).
///
/// Linux counts each thread's wait: the second field of `/proc/thread-self/schedstat`, "time spent
/// waiting on a runqueue" in nanoseconds (the kernel's `Documentation/scheduler/sched-stats.rst`).
/// Where the kernel gives no such count (another OS, or a kernel built without `CONFIG_SCHED_INFO`)
/// the clock reads the wall time.
///
/// The kernel adds a wait to the count as the thread gets a CPU back, and reading the count is a
/// system call, which a wait may interrupt: a wait between reading the instant and reading the
/// count would be left out of one and counted in the other. So each end reads the count on both
/// sides of its instant until the two agree, no wait between them, and where they never do it
/// keeps the side that leaves the wait in the reading: the count after the instant at the start,
/// before it at the end. The clock so never reads less than the call's time on a CPU and blocked.
pub struct CreationClock {
    started: Instant,
    /// The thread's run-queue wait so far, where the kernel counts it.
    queued: Option<Duration>,
    /// The wait counted is the thread's own, so the clock stays on it.
    _thread: PhantomData<*const ()>,
}

impl CreationClock {
    /// A clock started now, on the calling thread.
    pub fn start() -> CreationClock {
        let (_, started, queued) = paired();
        CreationClock {
            started,
            queued,
            _thread: PhantomData,
        }
    }

    /// The time since [`CreationClock::start`], less the time the thread waited for a CPU since
    /// then where the kernel counts it.
    pub fn read(&self) -> Duration {
        let (queued, now, _) = paired();
        let wall = now.saturating_duration_since(self.started);
        match (self.queued, queued) {
            (Some(before), Some(after)) => wall.saturating_sub(after.saturating_sub(before)),
            _ => wall,
        }
    }
}

/// How many times an end of a [`CreationClock`] reads the count on both sides of its instant
/// before it keeps a pair that disagrees.
const PAIRING_TRIES: usize = 4;

/// An instant with the calling thread's run-queue wait read just before and just after it
/// ([`CreationClock`]): read again until the two agree, at most [`PAIRING_TRIES`] times.
fn paired() -> (Option<Duration>, Instant, Option<Duration>) {
    let mut pair = (run_queue_wait(), Instant::now(), run_queue_wait());
    for _ in 1..PAIRING_TRIES {
        if pair.0 == pair.2 {
            break;
        }
        pair = (run_queue_wait(), Instant::now(), run_queue_wait());
    }
    pair
}

/// The calling thread's run-queue wait so far: the second field of `/proc/thread-self/schedstat`
/// ([`CreationClock`]); `None` where the kernel gives none.
fn run_queue_wait() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let waited = stat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(Duration::from_nanos(waited))
}

/// The settings that turn the Vulkan drivers' own shader disk caches off, so a pipeline creation
/// compiles its modules: Mesa's, which its Vulkan drivers read (ANV, RADV, lavapipe), NVIDIA's, and
/// Metal's, through which MoltenVK builds every pipeline (`MTL_SHADER_CACHE_SIZE=0`: on an M4,
/// every run with it compiled cold, a median 41 times a warm run's creation, where without it a
/// second run's hit Metal's cache). A creation that hits such a cache measures a
/// lookup, and nothing in the timing tells the two apart.
pub const DRIVER_SHADER_CACHES_OFF: [(&str, &str); 3] = [
    ("MESA_SHADER_CACHE_DISABLE", "true"),
    ("__GL_SHADER_DISK_CACHE", "0"),
    ("MTL_SHADER_CACHE_SIZE", "0"),
];

/// Sets [`DRIVER_SHADER_CACHES_OFF`] in this process's environment, over any value set before, so
/// the drivers this process loads and the processes it starts (a batch's runs) compile every
/// pipeline they create. A tool that times `vkCreate*Pipelines` against
/// [`PIPELINE_CREATION_CEILING`] calls it first in `main`, before it starts a thread or loads the
/// Vulkan loader.
pub fn turn_driver_shader_caches_off() {
    for (name, value) in DRIVER_SHADER_CACHES_OFF {
        std::env::set_var(name, value);
    }
}

/// Whether a ledger peak accounts for an allocator peak under [`LEDGER_FLOOR`].
pub fn ledger_covers(ledger_peak: u64, allocator_peak: u64) -> bool {
    u128::from(ledger_peak) * u128::from(LEDGER_FLOOR.1)
        >= u128::from(allocator_peak) * u128::from(LEDGER_FLOOR.0)
}

/// A row whose ledger peak is below the floor of its allocator peak.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerShortfall {
    pub air_sha256: String,
    pub ledger_peak: u64,
    pub allocator_peak: u64,
}

/// The L7 budget gates over a set of recorded rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BudgetReport {
    pub rows: usize,
    /// Rows the worker reported (with wall time).
    pub timed: usize,
    pub slowest: Option<(String, u64)>,
    pub p50_ms: Option<u64>,
    pub p99_ms: Option<u64>,
    pub peak_resident: Option<(String, u64)>,
    /// The largest counting-allocator peak a worker reported (above its budget's baseline), and
    /// its row: exact where the 10-ms RSS sampler behind [`BudgetReport::peak_resident`] can miss a
    /// short peak. It is recorded evidence the worker's in-child budget bounds by 500 MiB (an
    /// allocation past it aborts the worker, a budget stop), not an independent gate, and it covers
    /// only the [`BudgetReport::allocator_measured`] rows.
    pub peak_allocator: Option<(String, u64)>,
    /// Timed rows whose worker reported an allocator peak: rows recorded previously have none, so
    /// [`BudgetReport::peak_allocator`] is over these rows, not every timed one.
    pub allocator_measured: usize,
    /// Rows stopped at a time or memory ceiling, or refused by an in-process budget, with the
    /// recorded `(label, key)` of the stop ([`budget_stop`]).
    pub budget_stops: Vec<(String, String)>,
    /// Rows with both a ledger and an allocator peak.
    pub ledger_checked: usize,
    pub ledger_shortfalls: Vec<LedgerShortfall>,
    /// The lowest ledger / allocator ratio and its row.
    pub ledger_min_ratio: Option<(String, f64)>,
}

/// A [`BudgetReport`] built one row at a time, so a sweep's parent can stream its rows through it
/// instead of holding them: it keeps only what the report keeps (each timed row's wall time for
/// the percentiles, the rows that fail a gate, and the extremes).
#[derive(Default)]
pub struct BudgetAccumulator {
    report: BudgetReport,
    times: Vec<u64>,
}

impl BudgetAccumulator {
    /// An empty report.
    pub fn new() -> BudgetAccumulator {
        BudgetAccumulator::default()
    }

    /// Count `row` in the gates.
    pub fn add(&mut self, row: &CostRow) {
        self.report.rows += 1;
        if budget_stop(&row.outcome.0, &row.outcome.1) {
            self.report.budget_stops.push((
                row.air_sha256.clone(),
                format!("{} {}", row.outcome.0, row.outcome.1),
            ));
        }
        let cost = &row.cost;
        if let Some(worker) = cost.worker {
            self.times.push(worker.elapsed_ms);
            if self
                .report
                .slowest
                .as_ref()
                .is_none_or(|(_, ms)| worker.elapsed_ms > *ms)
            {
                self.report.slowest = Some((row.air_sha256.clone(), worker.elapsed_ms));
            }
            if worker.allocator_peak_bytes.is_some() {
                self.report.allocator_measured += 1;
            }
            for (measure, peak) in [
                (worker.peak_resident_bytes, &mut self.report.peak_resident),
                (worker.allocator_peak_bytes, &mut self.report.peak_allocator),
            ] {
                if let Some(bytes) = measure {
                    if peak.as_ref().is_none_or(|(_, most)| bytes > *most) {
                        *peak = Some((row.air_sha256.clone(), bytes));
                    }
                }
            }
        }
        if let Some(compile) = &cost.compile {
            if let Some(allocator_peak) = compile.allocator_peak.filter(|peak| *peak > 0) {
                self.report.ledger_checked += 1;
                let ratio = compile.ledger_peak as f64 / allocator_peak as f64;
                if self
                    .report
                    .ledger_min_ratio
                    .as_ref()
                    .is_none_or(|(_, min)| ratio < *min)
                {
                    self.report.ledger_min_ratio = Some((row.air_sha256.clone(), ratio));
                }
                if !ledger_covers(compile.ledger_peak, allocator_peak) {
                    self.report.ledger_shortfalls.push(LedgerShortfall {
                        air_sha256: row.air_sha256.clone(),
                        ledger_peak: compile.ledger_peak,
                        allocator_peak,
                    });
                }
            }
        }
    }

    /// The report over every added row.
    pub fn finish(mut self) -> BudgetReport {
        self.report.timed = self.times.len();
        self.times.sort_unstable();
        self.report.p50_ms = percentile(&self.times, 50);
        self.report.p99_ms = percentile(&self.times, 99);
        self.report
    }
}

impl BudgetReport {
    /// Derive the gates from `rows`.
    pub fn from_rows(rows: &[CostRow]) -> BudgetReport {
        let mut accumulator = BudgetAccumulator::new();
        for row in rows {
            accumulator.add(row);
        }
        accumulator.finish()
    }

    /// The per-attempt ceiling breaches (AGENTS.md: 20 s and 500 MiB per translation attempt),
    /// one line each: every budget stop, the slowest row above the deadline, and the peak worker
    /// RSS above the memory limit. Each is a failing result on its own, so the engine
    /// translation sweep exits non-zero on any of them in the batch it ran; the ledger gate is
    /// [`BudgetReport::failures`]' alone.
    pub fn ceiling_breaches(&self) -> Vec<String> {
        let mut breaches = Vec::new();
        for (row, outcome) in &self.budget_stops {
            breaches.push(format!("budget stop: {row} {outcome}"));
        }
        if let Some((row, ms)) = &self.slowest {
            if *ms > crate::worker::DEADLINE.as_millis() as u64 {
                breaches.push(format!("time: {row} took {ms} ms"));
            }
        }
        if let Some((row, rss)) = &self.peak_resident {
            if *rss > crate::worker::MEMORY_LIMIT_BYTES {
                breaches.push(format!("memory: {row} peaked at {rss} bytes"));
            }
        }
        // The larger-measure rule (`WorkerMeasure::peak_bytes`). A worker's in-child budget refuses
        // any allocation past 500 MiB above its baseline, so for a budgeted worker this only ever
        // confirms; an overshoot aborts it and is a budget stop above.
        if let Some((row, bytes)) = &self.peak_allocator {
            if *bytes > crate::worker::MEMORY_LIMIT_BYTES {
                breaches.push(format!(
                    "memory: {row} allocator peaked at {bytes} live heap bytes"
                ));
            }
        }
        breaches
    }

    /// Every failed gate, as one line each; empty when all pass: the
    /// [`ceiling breaches`](BudgetReport::ceiling_breaches), then the ledger gate.
    pub fn failures(&self) -> Vec<String> {
        let mut failures = self.ceiling_breaches();
        for s in &self.ledger_shortfalls {
            failures.push(format!(
                "ledger: {} ledger peak {} < {}/{} of allocator peak {}",
                s.air_sha256, s.ledger_peak, LEDGER_FLOOR.0, LEDGER_FLOOR.1, s.allocator_peak
            ));
        }
        failures
    }
}

/// Whether a recorded `(label, key)` is a budget breach whatever its measure: the worker boundary
/// stopped the attempt at the deadline (`timeout`), at the memory ceiling (`oom`), or by an abort
/// such as the counting allocator's budget refusal (`worker_exit`), the allocator failed a
/// reservation the engine's ledger admitted (`allocation_failed`, which the counting allocator's
/// refusal also produces), or the engine refused it at its work-unit or ledger budget
/// (`limit_exceeded budget`). The deadline and the allocation failure are transient: only a sweep's
/// own batch sees them.
pub fn budget_stop(label: &str, key: &str) -> bool {
    match label {
        "failed" => matches!(
            FailureKind::parse(key),
            Some(
                FailureKind::Timeout
                    | FailureKind::AllocationFailed
                    | FailureKind::Oom
                    | FailureKind::WorkerExit
            )
        ),
        "limit_exceeded" => {
            LimitKind::parse(key.split(' ').next().unwrap_or_default()) == Some(LimitKind::Budget)
        }
        _ => false,
    }
}

/// Nearest-rank percentile of sorted `values`.
fn percentile(sorted: &[u64], p: usize) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (p * sorted.len()).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AttemptCost;
    use crate::translation_audit::{RecordedCost, WorkerMeasure};
    use std::collections::BTreeMap;

    /// A creation at the ceiling passes; one past it is an error that names the pipeline and the
    /// ceiling.
    #[test]
    fn pipeline_creation_above_five_seconds_is_an_error() {
        let ceiling = PIPELINE_CREATION_CEILING;
        assert!(creation_within("the pipeline", ceiling, ceiling).is_ok());
        let over = PIPELINE_CREATION_CEILING + Duration::from_millis(1);
        let error = creation_within("the graphics pipeline", over, ceiling).unwrap_err();
        assert!(error.contains("the graphics pipeline"), "{error}");
        assert!(error.contains("5001 ms"), "{error}");
        assert!(error.contains("5 s"), "{error}");
        // The compute executor applies it with `?` to the runtime's count of its pipelines'
        // creation (`platform::runtime`): the run fails, never becoming a refusal or an
        // observation.
        let run = || -> Result<(), crate::check::admission::ExecutorError> {
            creation_within("the graphics pipeline", over, ceiling)?;
            Ok(())
        };
        match run() {
            Err(crate::check::admission::ExecutorError::Failed(why)) => {
                assert!(why.contains("pipeline-creation ceiling"), "{why}")
            }
            other => panic!("an over-ceiling creation must fail the run: {other:?}"),
        }
    }

    /// A case's own bound holds its pipeline's creation in place of the ceiling: a creation past
    /// the ceiling and within the bound passes, one past the bound is an error that names it.
    #[test]
    fn a_case_s_own_creation_bound_holds_in_place_of_the_ceiling() {
        let bound = Duration::from_secs(8);
        let past_ceiling = PIPELINE_CREATION_CEILING + Duration::from_millis(1);
        assert!(creation_within("the compute pipeline", past_ceiling, bound).is_ok());
        let over = bound + Duration::from_millis(1);
        let error = creation_within("the compute pipeline", over, bound).unwrap_err();
        assert!(error.contains("8001 ms"), "{error}");
        assert!(
            error.contains("own pipeline-creation bound of 8000 ms"),
            "{error}"
        );
    }

    /// A [`CreationClock`] counts the time its thread spends on a CPU and the time it blocks, and
    /// leaves out the time it waits for a CPU: a thread that sleeps reads its sleep; a thread that
    /// shares one CPU with a spinning thread (both pinned to it) while it spins for 100 ms of its
    /// own CPU time reads no less than that CPU time and well under its wall time, which holds the
    /// other thread's turns. Linux counts the wait (`/proc/thread-self/schedstat`).
    ///
    /// The spinner's turns also fall at every point of [`READINGS`] clocks of 50 us of the
    /// thread's CPU time each, some between an end's instant and its count of the wait; none reads
    /// less than its CPU time. With each end's count read on one side of its instant only, 1 to 32
    /// of them read less, most as 0 (ten runs on a 16-thread host).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_creation_clock_leaves_out_the_wait_for_a_cpu() {
        use std::sync::atomic::{AtomicBool, Ordering};
        assert!(
            run_queue_wait().is_some(),
            "the kernel counts no run-queue wait"
        );
        let clock = CreationClock::start();
        std::thread::sleep(Duration::from_millis(50));
        let slept = clock.read();
        assert!(slept >= Duration::from_millis(50), "{slept:?}");

        let thread_cpu = || {
            let mut now = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: `now` is a valid timespec for the call to fill.
            let read = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
            assert_eq!(read, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
            Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
        };
        // Pin the calling thread to the CPU it is on, and a spinning thread to the same CPU.
        // SAFETY: no arguments.
        let cpu = unsafe { libc::sched_getcpu() };
        assert!(cpu >= 0, "sched_getcpu");
        let pin = |cpu: usize| {
            // SAFETY: a zeroed set is an empty CPU set.
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            // SAFETY: `cpu` is a CPU number the kernel gave, within the set's size.
            unsafe { libc::CPU_SET(cpu, &mut set) };
            // SAFETY: `set` is a valid CPU set of its own size; 0 names the calling thread.
            let pinned = unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) };
            assert_eq!(pinned, 0, "sched_setaffinity");
        };
        // SAFETY: a zeroed set is an empty CPU set, which the call fills.
        let mut before: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        // SAFETY: `before` is a valid CPU set of its own size; 0 names the calling thread.
        let got =
            unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&before), &mut before) };
        assert_eq!(got, 0, "sched_getaffinity");
        let cpu = cpu as usize;
        pin(cpu);
        let stop = AtomicBool::new(false);
        let spin = |span: Duration| {
            let cpu_before = thread_cpu();
            while thread_cpu() - cpu_before < span {
                std::hint::spin_loop();
            }
            thread_cpu() - cpu_before
        };
        let (spent, wall, read, under) = std::thread::scope(|scope| {
            scope.spawn(|| {
                pin(cpu);
                while !stop.load(Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            });
            // Let the spinner take its first turn.
            std::thread::sleep(Duration::from_millis(10));
            let started = Instant::now();
            let clock = CreationClock::start();
            let spent = spin(Duration::from_millis(100));
            let read = clock.read();
            let wall = started.elapsed();
            let under: Vec<(Duration, Duration)> = (0..READINGS)
                .filter_map(|_| {
                    let clock = CreationClock::start();
                    let spent = spin(Duration::from_micros(50));
                    let read = clock.read();
                    (read + Duration::from_micros(20) < spent).then_some((read, spent))
                })
                .collect();
            stop.store(true, Ordering::Relaxed);
            (spent, wall, read, under)
        });
        // SAFETY: `before` is the thread's own earlier set, of its own size.
        let restored =
            unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&before), &before) };
        assert_eq!(restored, 0, "sched_setaffinity");
        assert!(
            read + Duration::from_millis(1) >= spent && read * 4 <= wall * 3,
            "the clock read {read:?} of {wall:?} for {spent:?} of the thread's CPU"
        );
        assert!(
            under.is_empty(),
            "{} of {READINGS} clocks read less than their thread's CPU time (reading, CPU): {under:?}",
            under.len()
        );
    }

    /// How many short clocks [`a_creation_clock_leaves_out_the_wait_for_a_cpu`] reads beside the
    /// spinner: about a second of its turns.
    #[cfg(target_os = "linux")]
    const READINGS: usize = 6000;

    /// The switch overrides values set before it, and a process started after it (a batch's run)
    /// inherits it. It changes this process's environment, so it runs in a child of the test
    /// binary, alone.
    #[cfg(unix)]
    #[test]
    fn driver_shader_caches_are_off_for_the_process_and_what_it_starts() {
        const CHILD: &str = "M2V_TEST_SHADER_CACHES_CHILD";
        let script = DRIVER_SHADER_CACHES_OFF
            .iter()
            .map(|(name, _)| format!("${{{name}-unset}}"))
            .collect::<Vec<_>>()
            .join(" ");
        let expected = DRIVER_SHADER_CACHES_OFF
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>()
            .join(" ");
        if std::env::var_os(CHILD).is_some() {
            turn_driver_shader_caches_off();
            let started = std::process::Command::new("sh")
                .args(["-c", &format!("printf '%s' \"{script}\"")])
                .output()
                .unwrap();
            assert_eq!(String::from_utf8_lossy(&started.stdout), expected);
            return;
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "budgets::tests::driver_shader_caches_are_off_for_the_process_and_what_it_starts",
            ])
            .env(CHILD, "1");
        for (name, _) in DRIVER_SHADER_CACHES_OFF {
            command.env(name, "1");
        }
        let child = command.output().unwrap();
        assert!(
            child.status.success(),
            "{}{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        // The child ran the test rather than filtering it out.
        assert!(
            String::from_utf8_lossy(&child.stdout).contains("1 passed"),
            "{}",
            String::from_utf8_lossy(&child.stdout)
        );
    }

    fn row(hash: &str, ms: u64, ledger: u64, alloc: u64, words: u64) -> CostRow {
        CostRow {
            air_sha256: hash.into(),
            outcome: ("translated".into(), String::new()),
            cost: RecordedCost {
                words: Some(words),
                compile: Some(AttemptCost {
                    ledger_peak: ledger,
                    work_units: 1,
                    allocator_peak: Some(alloc),
                    strategy_words: BTreeMap::from([
                        ("access.load".to_string(), words / 2),
                        ("arith.int.legalized".to_string(), words / 2),
                    ]),
                    spirv_sha256: None,
                }),
                worker: Some(WorkerMeasure {
                    elapsed_ms: ms,
                    peak_resident_bytes: Some(ms * 1000),
                    allocator_peak_bytes: Some(ms * 500),
                }),
            },
        }
    }

    fn stopped(hash: &str, label: &str, key: &str, worker: Option<WorkerMeasure>) -> CostRow {
        CostRow {
            air_sha256: hash.into(),
            outcome: (label.into(), key.into()),
            cost: RecordedCost {
                words: None,
                compile: None,
                worker,
            },
        }
    }

    #[test]
    fn rows_stopped_at_a_ceiling_fail_the_gate_with_or_without_a_measure() {
        // A timed-out row the parent measured up to the stop, and an OOM row with no measure:
        // both fail, and neither needs its sample to be above the ceiling.
        let rows = [
            row("a", 10, 80, 100, 100),
            stopped(
                "t",
                "failed",
                "timeout",
                Some(WorkerMeasure {
                    elapsed_ms: 20_000,
                    peak_resident_bytes: Some(1_000),
                    allocator_peak_bytes: None,
                }),
            ),
            stopped("o", "failed", "oom", None),
        ];
        let report = BudgetReport::from_rows(&rows);
        assert_eq!(report.timed, 2);
        assert_eq!(report.slowest, Some(("t".into(), 20_000)));
        assert_eq!(
            report.budget_stops,
            vec![
                ("t".to_string(), "failed timeout".to_string()),
                ("o".to_string(), "failed oom".to_string()),
            ]
        );
        let failures = report.failures();
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert_eq!(failures[0], "budget stop: t failed timeout");
        assert_eq!(failures[1], "budget stop: o failed oom");
        // Both are ceiling breaches, which fail the sweep that met them.
        assert_eq!(report.ceiling_breaches(), failures);

        for (label, key) in [
            ("failed", "timeout"),
            ("failed", "allocation_failed"),
            ("failed", "oom"),
            ("failed", "worker_exit"),
            ("limit_exceeded", "budget WorkUnits"),
        ] {
            assert!(budget_stop(label, key), "{label} {key}");
        }
        for (label, key) in [
            ("failed", "cancelled"),
            ("failed", "panic"),
            ("limit_exceeded", "structural Words"),
            ("translated", ""),
        ] {
            assert!(!budget_stop(label, key), "{label} {key}");
        }
    }

    #[test]
    fn ledger_floor_is_seven_tenths_of_the_allocator_peak() {
        assert!(ledger_covers(70, 100));
        assert!(!ledger_covers(69, 100));
        assert!(ledger_covers(0, 0));
        assert!(ledger_covers(u64::MAX, u64::MAX));
    }

    #[test]
    fn report_derives_time_memory_and_ledger_gates() {
        let rows = [
            row("a", 10, 80, 100, 100),
            row("b", 30, 50, 100, 400),
            row("c", 20, 90, 100, 120),
        ];
        let report = BudgetReport::from_rows(&rows);
        assert_eq!(report.rows, 3);
        assert_eq!(report.slowest, Some(("b".into(), 30)));
        assert_eq!((report.p50_ms, report.p99_ms), (Some(20), Some(30)));
        assert_eq!(report.peak_resident, Some(("b".into(), 30_000)));
        assert_eq!(report.peak_allocator, Some(("b".into(), 15_000)));
        assert_eq!(report.allocator_measured, 3);
        assert_eq!(report.ledger_checked, 3);
        assert_eq!(report.ledger_shortfalls.len(), 1);
        assert_eq!(report.ledger_shortfalls[0].air_sha256, "b");
        let failures = report.failures();
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].starts_with("ledger: b"));
        // A ledger failure is not a ceiling breach: the sweep does not fail on it (`--audit
        // budgets` does).
        assert!(report.ceiling_breaches().is_empty());

        // A row over the deadline or the memory limit is a breach even when it completed.
        let over = [
            row("slow", 20_001, 80, 100, 100),
            CostRow {
                cost: RecordedCost {
                    worker: Some(WorkerMeasure {
                        elapsed_ms: 5,
                        peak_resident_bytes: Some(crate::worker::MEMORY_LIMIT_BYTES + 1),
                        allocator_peak_bytes: None,
                    }),
                    ..row("big", 5, 80, 100, 100).cost
                },
                ..row("big", 5, 80, 100, 100)
            },
        ];
        let report = BudgetReport::from_rows(&over);
        let breaches = report.ceiling_breaches();
        assert_eq!(breaches.len(), 2, "{breaches:?}");
        assert!(breaches[0].starts_with("time: slow took 20001 ms"));
        assert!(breaches[1].starts_with("memory: big peaked at"));

        // A short peak the RSS sampler missed is recorded from the allocator, up to the most the
        // in-child budget admits (the whole budget), which is no breach; a row recorded before
        // the allocator was measured is not counted as measured.
        let missed = [
            CostRow {
                cost: RecordedCost {
                    worker: Some(WorkerMeasure {
                        elapsed_ms: 5,
                        peak_resident_bytes: Some(64 << 20),
                        allocator_peak_bytes: Some(crate::worker::MEMORY_LIMIT_BYTES),
                    }),
                    ..row("spike", 5, 80, 100, 100).cost
                },
                ..row("spike", 5, 80, 100, 100)
            },
            CostRow {
                cost: RecordedCost {
                    worker: Some(WorkerMeasure {
                        elapsed_ms: 5,
                        peak_resident_bytes: Some(1 << 20),
                        allocator_peak_bytes: None,
                    }),
                    ..row("older", 5, 80, 100, 100).cost
                },
                ..row("older", 5, 80, 100, 100)
            },
        ];
        let report = BudgetReport::from_rows(&missed);
        assert_eq!(report.peak_resident, Some(("spike".into(), 64 << 20)));
        assert_eq!(
            report.peak_allocator,
            Some(("spike".into(), crate::worker::MEMORY_LIMIT_BYTES))
        );
        assert_eq!((report.timed, report.allocator_measured), (2, 1));
        assert!(report.ceiling_breaches().is_empty());
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let values: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&values, 50), Some(50));
        assert_eq!(percentile(&values, 99), Some(99));
        assert_eq!(percentile(&[7], 99), Some(7));
        assert_eq!(percentile(&[], 50), None);
    }
}
