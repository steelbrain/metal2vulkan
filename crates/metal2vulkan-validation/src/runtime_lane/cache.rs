//! The runtime's shader cache on disk across processes, on the device the cfg selects. Each test
//! owns its empty cache directories ([`ScratchCache`]) and removes them.
//!
//! - A second process creating the same pipeline state reads the first one's translations: every
//!   lookup hits, nothing is compiled, each making's pipeline cache starts from the record the
//!   first process's making wrote, and the output is the same. The test prints both processes'
//!   creation times. A driver whose pipeline cache data ends inside the header it states cannot
//!   keep it: then, and only then, the second's makings start empty, and both processes must have
//!   counted that reason (`driver_data_truncated`).
//! - An entry under the right name that holds another translation (the function artifact of
//!   another case) is refused and removed, the function is translated again, and the output is
//!   the same.
//! - A state made again on the same device takes its translations from the device's memo: no lookup
//!   reaches the disk. Under a memory capacity no artifact fits in, the memo keeps nothing, and the
//!   same creation's lookups reach the disk and hit.
//! - A creation whose cancel token is cancelled, or whose deadline has passed, refuses typed and
//!   keeps nothing; once the translation is cached, the same creation compiles nothing and
//!   succeeds.
//! - Several threads making pipelines of new kernel cases at once each read their own making's
//!   cache (A save of the device's one cache read it while the threads made pipelines with it, and
//!   a pipeline added between that read's two calls made MoltenVK report "Error writing pipeline
//!   cache data"): every making's data is read, nothing fails, and nothing is reported.

use super::{
    at_plan, layer_reported_nothing_at_all, needs_later_steps, pipeline_state, public_case,
    public_case_names, public_cases, run_case, run_with, validated_device, validated_device_with,
    Messages, PublicCase, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::case::ExecutionSafety;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, Stage, StageSource};
use metal2vulkan::{Budget, CancelToken};
use metal2vulkan_runtime::{
    Degradation, Error, PipelineOptions, ShaderCacheStatistics, DEFAULT_MEMORY_CAPACITY,
};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The case the cross-process test runs.
const CASE: &str = "saturating-halving-and-wide-product-integers-at-their-edges";
/// The case whose function artifact the collision test puts under [`CASE`]'s name.
const OTHER_CASE: &str = "threadgroup-word";
/// The environment variable that names the cache directory to [`second_process`].
const SECOND_DIR: &str = "M2V_LANE_SHADER_CACHE_DIR";
/// The environment variable that names the case to [`second_process`] (by default [`CASE`])
/// and to [`a_named_case_cold_and_warm`].
const CASE_VAR: &str = "M2V_LANE_SHADER_CACHE_CASE";
/// The committed profile the survey translates for: the RADV device the timing run measures on.
const SURVEY_PROFILE: &str = "amd-780m-radv";
/// The line [`second_process`] reports on.
const REPORT: &str = "SHADER-CACHE-SECOND-PROCESS";
/// The threads [`makings_on_several_threads_each_read_their_own_cache`] makes pipelines on.
const MAKERS: usize = 4;
/// The loop-free kernel cases with public AIR it maps, the first in shard order: those that map
/// and need no later step are the cases it makes pipelines of.
const MADE_CASES: usize = 32;

/// What one device's run of a case saw.
struct CacheRun {
    output: Vec<u8>,
    statistics: ShaderCacheStatistics,
    /// `new_compute_pipeline_state_with_function`.
    creation: Duration,
    /// The run: its encoding (which makes the `VkPipeline`), commit and completion.
    run: Duration,
}

/// The makings' pipeline cache records (`M2VPPLC\0`, `shader_cache/driver.rs`) under `cache`'s
/// layout directory, by path, with their bytes (It counted them).
fn driver_records(cache: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    let Ok(builds) = std::fs::read_dir(cache.join("v1")) else {
        return found;
    };
    for build in builds {
        for group in std::fs::read_dir(build.unwrap().path()).unwrap() {
            for entry in std::fs::read_dir(group.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                let bytes = std::fs::read(&path).unwrap();
                if bytes.starts_with(b"M2VPPLC\0") {
                    found.insert(path, bytes);
                }
            }
        }
    }
    found
}

/// Run `case` once on a new device whose shader cache is `cache`; the device is gone when this
/// returns. Fails on any validation message.
fn cache_run(cache: &Path, case: &str) -> CacheRun {
    let messages = Messages::default();
    let run = {
        let device = validated_device(TEST_DEVICE, &messages, cache).unwrap();
        let case = public_case(case).unwrap();
        let start = Instant::now();
        let pipeline = pipeline_state(&device, &case).unwrap();
        let creation = start.elapsed();
        let start = Instant::now();
        let output = run_with(&device, &case, &pipeline).unwrap().output;
        let run = start.elapsed();
        CacheRun {
            output,
            statistics: device.shader_cache_statistics(),
            creation,
            run,
        }
    };
    layer_reported_nothing_at_all(&messages);
    run
}

/// What the second process reported.
struct Warm(String);

impl Warm {
    /// Field `name`'s value.
    fn field(&self, name: &str) -> String {
        self.0
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("no {name} in {}", self.0))
            .to_string()
    }

    /// Count `name`.
    fn count(&self, name: &str) -> u64 {
        self.field(name).parse().unwrap()
    }
}

/// Run `case` in this process with an empty cache, then in a second process with the cache the
/// first left; check that the second hit every translation, compiled nothing, started each
/// making's cache from the first's record of it (unless the driver's data is truncated, a reason
/// both processes count), and computed the same output; print both processes' times.
fn two_processes(case: &str) {
    let cache = ScratchCache::new().unwrap();
    let cold = cache_run(cache.path(), case);
    let first_records = driver_records(cache.path());
    // Each variant the run compiled is a function and a pipeline, each missed and written.
    let c = cold.statistics;
    let compiled = c.writes;
    assert!(
        c.hits == 0 && c.misses == compiled && compiled >= 2 && compiled.is_multiple_of(2),
        "the first process: {c:?}"
    );
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "runtime_lane::cache::second_process",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(SECOND_DIR, cache.path())
        .env(CASE_VAR, case)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child.status.success(),
        "the second process failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&child.stderr)
    );
    let warm = Warm(
        stdout
            .lines()
            // The harness may print the test's name on the same line first.
            .find_map(|line| Some(line[line.find(REPORT)? + REPORT.len()..].to_string()))
            .unwrap_or_else(|| panic!("the second process did not report:\n{stdout}")),
    );
    assert_eq!(
        (
            warm.count("hits"),
            warm.count("misses"),
            warm.count("rejected"),
            warm.count("writes")
        ),
        (compiled, 0, 0, 0),
        "the second process: {}",
        warm.0
    );
    // No process found the driver's data for another device or header version.
    assert_eq!(
        (
            c.driver_data_mismatched,
            warm.count("driver_data_mismatched")
        ),
        (0, 0),
        "the first process: {c:?}; the second: {}",
        warm.0
    );
    if c.driver_data_truncated == 0 {
        // Each of the first process's makings wrote its record; the second's makings each
        // started from it: one the driver reports found there read no data back, and every
        // other one wrote it again exactly where the driver's data was no longer the record's
        // (RADV's data after a hit is not the record's bytes).
        assert!(
            c.driver_saves > 0 && first_records.len() as u64 == c.driver_saves,
            "the first process: {c:?}"
        );
        let second_records = driver_records(cache.path());
        let rewritten = second_records
            .iter()
            .filter(|(path, bytes)| first_records.get(*path) != Some(*bytes))
            .count() as u64;
        assert_eq!(
            second_records.len(),
            first_records.len(),
            "the second process's records: the first's"
        );
        assert_eq!(
            (
                warm.count("driver_loads"),
                warm.count("driver_saves"),
                warm.count("driver_data_unchanged") + warm.count("driver_data_found"),
                warm.count("driver_data_truncated")
            ),
            (c.driver_saves, rewritten, c.driver_saves - rewritten, 0),
            "the second process: {}",
            warm.0
        );
    } else {
        // The driver's data ends inside its own header: nothing was kept, so the second
        // process's makings started empty, for the same counted reason.
        assert!(
            warm.count("driver_loads") == 0 && warm.count("driver_data_truncated") > 0,
            "the second process: {}",
            warm.0
        );
        assert!(driver_records(cache.path()).is_empty());
        println!("shader cache, {case}: the driver's pipeline cache data is truncated, not kept");
    }
    assert_eq!(
        warm.field("output"),
        hex(&cold.output),
        "the outputs differ"
    );
    println!(
        "shader cache, {case}: cold creation {} us, run {} us; warm creation {} us, run {} us",
        cold.creation.as_micros(),
        cold.run.as_micros(),
        warm.count("creation_us"),
        warm.count("run_us"),
    );
}

/// A second process creating the same pipeline state hits every translation and the driver's
/// saved cache, compiles nothing, and computes the same output.
#[test]
fn a_second_process_reads_the_first_ones_translations() {
    two_processes(CASE);
}

/// [`two_processes`] on the case [`CASE_VAR`] names: a timing run takes it in release mode on
/// the slowest translation [`rank_public_cases_by_translation_time`] found.
#[test]
#[ignore = "a release-mode measurement: a timing run names the case"]
fn a_named_case_cold_and_warm() {
    let case = std::env::var(CASE_VAR).expect("a timing run names the case");
    two_processes(&case);
}

/// The committed public kernel cases step 1 can run (buffers and threadgroup memory only, no
/// function constants), ranked by the time the compile layer takes to translate and compile
/// them for [`SURVEY_PROFILE`], slowest first: the ten slowest are printed as `RANKED` lines.
/// CPU only: it opens no device. A case that does not compile is left out.
#[test]
#[ignore = "a release-mode survey: a timing run takes it"]
fn rank_public_cases_by_translation_time() {
    let device = crate::engine::load_device(SURVEY_PROFILE).unwrap();
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    let mut timed = Vec::new();
    for name in public_case_names().unwrap() {
        let Ok(case) = public_case(&name) else {
            continue;
        };
        if needs_later_steps(&case.compute) || !case.compute.constants.is_empty() {
            continue;
        }
        let Ok(function) =
            Library::parse(&case.bitcode).and_then(|library| library.function(&case.compute.entry))
        else {
            continue;
        };
        let start = Instant::now();
        let compiled = device
            .compile_function(
                &FunctionDescriptor::new(function),
                &inputs,
                &Budget::standard(),
            )
            .and_then(|compiled| {
                device.compile_compute_pipeline(
                    &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
                    &Budget::standard(),
                )
            });
        let elapsed = start.elapsed();
        if compiled.is_ok() {
            timed.push((elapsed, name));
        }
    }
    timed.sort_by_key(|(elapsed, _)| std::cmp::Reverse(*elapsed));
    println!("{} cases compiled", timed.len());
    for (elapsed, name) in timed.iter().take(10) {
        println!("RANKED {} us {name}", elapsed.as_micros());
    }
    assert!(!timed.is_empty(), "no public case step 1 can run compiled");
}

/// The second process of [`a_second_process_reads_the_first_ones_translations`].
#[test]
#[ignore = "the second process of a_second_process_reads_the_first_ones_translations"]
fn second_process() {
    let dir = PathBuf::from(std::env::var_os(SECOND_DIR).expect("the first process names it"));
    let case = std::env::var(CASE_VAR).unwrap_or_else(|_| CASE.to_string());
    let warm = cache_run(&dir, &case);
    let s = warm.statistics;
    println!(
        "\n{REPORT} hits={} misses={} rejected={} writes={} driver_loads={} driver_saves={} \
         driver_data_truncated={} driver_data_mismatched={} driver_data_unchanged={} \
         driver_data_found={} \
         creation_us={} run_us={} output={}",
        s.hits,
        s.misses,
        s.rejected,
        s.writes,
        s.driver_loads,
        s.driver_saves,
        s.driver_data_truncated,
        s.driver_data_mismatched,
        s.driver_data_unchanged,
        s.driver_data_found,
        warm.creation.as_micros(),
        warm.run.as_micros(),
        hex(&warm.output)
    );
}

/// An entry under the right name holding another case's translated function is refused and
/// removed; the function is translated and written again, the pipeline still hits, and the
/// output is unchanged.
#[test]
fn an_entry_holding_another_translation_is_refused_and_replaced() {
    let other = ScratchCache::new().unwrap();
    cache_run(other.path(), OTHER_CASE);
    let cache = ScratchCache::new().unwrap();
    let cold = cache_run(cache.path(), CASE);
    let foreign = artifact_of_kind(other.path(), FUNCTION);
    let own = artifact_of_kind(cache.path(), FUNCTION);
    std::fs::copy(&foreign, &own).unwrap();
    let warm = cache_run(cache.path(), CASE);
    let s = warm.statistics;
    // The replaced function is refused, missed and written again; everything else hits.
    assert_eq!(
        (s.rejected, s.misses, s.writes, s.hits),
        (1, 1, 1, cold.statistics.writes - 1),
        "{s:?}"
    );
    assert_eq!(warm.output, cold.output);
    assert_ne!(
        std::fs::read(&own).unwrap(),
        std::fs::read(&foreign).unwrap()
    );
}

/// A state made again on the same device takes its function and pipeline from the device's
/// memo, so the disk cache counts no lookup, and runs to the same output; under a memory
/// capacity of one byte the memo keeps nothing, and the same two lookups reach the disk and hit.
#[test]
fn a_state_made_again_takes_the_memos_translations() {
    let case = public_case(CASE).unwrap();
    for (memory_capacity, disk_lookups) in [(DEFAULT_MEMORY_CAPACITY, 0), (1, 2)] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                options.shader_cache.memory_capacity = memory_capacity;
                at_plan(options)
            })
            .unwrap();
            let first = pipeline_state(&device, &case).unwrap();
            let output = run_with(&device, &case, &first).unwrap().output;
            let cold = device.shader_cache_statistics();
            let again = pipeline_state(&device, &case).unwrap();
            let warm = device.shader_cache_statistics();
            assert_eq!(
                (
                    warm.hits - cold.hits,
                    warm.misses - cold.misses,
                    warm.writes - cold.writes
                ),
                (disk_lookups, 0, 0),
                "memory capacity {memory_capacity}: {cold:?} then {warm:?}"
            );
            assert_eq!(run_with(&device, &case, &again).unwrap().output, output);
        }
        layer_reported_nothing_at_all(&messages);
    }
}

/// A creation whose token is cancelled, or whose deadline has passed, compiles nothing and
/// refuses typed (`Compile` holding `Cancelled`, or `Budget`); once a creation under the default
/// options has the translation in the device's memo, the same two creations compile nothing and
/// succeed.
#[test]
fn a_cancelled_or_late_creation_refuses_until_its_translation_is_cached() {
    let case = public_case(CASE).unwrap();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = device.new_library_with_air(&case.bitcode).unwrap();
        let function = library
            .new_function_with_name_constant_values(&case.compute.entry, &case.compute.constants)
            .unwrap();
        let token = CancelToken::new();
        token.cancel();
        let mut cancelled = PipelineOptions::default();
        cancelled.cancel = Some(token);
        let mut late = PipelineOptions::default();
        late.deadline = Some(Instant::now());
        let refused = |options: &PipelineOptions| {
            device
                .new_compute_pipeline_state_with_function(&function, options)
                .map(|_| ())
        };
        match refused(&cancelled) {
            Err(Error::Compile(metal2vulkan::error::Error::Cancelled)) => {}
            other => panic!("a cancelled creation: {other:?}"),
        }
        match refused(&late) {
            Err(Error::Compile(metal2vulkan::error::Error::Budget(_))) => {}
            other => panic!("a late creation: {other:?}"),
        }
        assert_eq!(
            device.shader_cache_statistics().writes,
            0,
            "a refused compile keeps nothing"
        );
        device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let compiled = device.shader_cache_statistics();
        for options in [&cancelled, &late] {
            device
                .new_compute_pipeline_state_with_function(&function, options)
                .unwrap();
        }
        let again = device.shader_cache_statistics();
        assert_eq!(
            (again.misses, again.writes),
            (compiled.misses, compiled.writes),
            "nothing compiled again"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The makings a device's statistics count by what became of the driver's pipeline cache data:
/// those its makings' saves kept, those the checks refused (where the data is truncated, every
/// read is refused), those that were the record the making loaded (A pipeline two cases share is
/// made again from the first making's record once its state went, and the driver's hit leaves
/// the data as the record was), and those the driver reported found in their record, whose data
/// was not read.
fn driver_makings(statistics: &ShaderCacheStatistics) -> u64 {
    statistics.driver_saves
        + statistics.driver_data_truncated
        + statistics.driver_data_mismatched
        + statistics.driver_data_unchanged
        + statistics.driver_data_found
}

/// [`MAKERS`] threads run [`MADE_CASES`]' cases once each, every run making the case's pipeline
/// state and its pipeline with a cache of its own: each making's data is read once it made its
/// pipeline (kept, refused, or its record's unchanged) unless the driver reports it found in its
/// record, one making counted for each the driver reports, no read or write failed, and nothing
/// is reported. A case the device refuses typed is counted, not failed.
#[test]
fn makings_on_several_threads_each_read_their_own_cache() {
    let seen = Cell::new(0usize);
    let cases: Vec<PublicCase> = public_cases(|case| {
        let made = case.stage == crate::case::Stage::Kernel
            && case.public_source.is_some()
            && case.execution_safety == ExecutionSafety::LoopFree;
        if !made {
            return false;
        }
        let index = seen.get();
        seen.set(index + 1);
        index < MADE_CASES
    })
    .unwrap()
    .into_iter()
    .filter_map(|(_, case)| case.ok())
    .filter(|case| !needs_later_steps(&case.compute))
    .collect();
    assert!(cases.len() >= MAKERS, "{} cases to make", cases.len());
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (statistics, failed) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let next = AtomicUsize::new(0);
        let failed = std::thread::scope(|scope| {
            let makers: Vec<_> = (0..MAKERS)
                .map(|_| {
                    scope.spawn(|| {
                        let mut failed = Vec::new();
                        while let Some(case) = cases.get(next.fetch_add(1, Ordering::Relaxed)) {
                            if let Err(error) = run_case(&device, case) {
                                failed.push(error);
                            }
                        }
                        failed
                    })
                })
                .collect();
            makers
                .into_iter()
                .flat_map(|maker| maker.join().unwrap())
                .collect::<Vec<_>>()
        });
        (device.shader_cache_statistics(), failed)
    };
    layer_reported_nothing_at_all(&messages);
    let made = cases.len() - failed.len();
    assert_eq!(
        (statistics.write_failures, statistics.read_failures),
        (0, 0),
        "{statistics:?}"
    );
    // The makings are the driver's reports, where it reports them, each counted once by what
    // became of its data: a pipeline two cases share is made again from its record once the
    // first case's state went.
    let reported = statistics.driver_pipeline_hits + statistics.driver_pipeline_misses;
    assert!(
        driver_makings(&statistics) > 0
            && (reported == 0 || driver_makings(&statistics) == reported),
        "{made} cases made, {} makings counted by their caches' data: {statistics:?}",
        driver_makings(&statistics)
    );
    let mut failed = failed;
    failed.sort();
    failed.dedup();
    println!(
        "driver cache: {} making(s), {} record(s), {} case(s) of {} made on {MAKERS} threads; {} \
         failed alike: {failed:?}",
        driver_makings(&statistics),
        driver_records(cache.path()).len(),
        made,
        cases.len(),
        failed.len()
    );
}

/// The kind [`an_entry_that_cannot_be_written_is_announced_and_ends_nothing`] takes: the
/// directory's build directory is made unwritable after the device opened it, so the entries a
/// creation hands over cannot be written.
pub(super) const UNWRITTEN_ENTRY: [Degradation; 1] = [Degradation::ShaderCacheEntryNotWritten];

/// The mode that lets the build directory be read and listed but not written.
#[cfg(unix)]
const READ_ONLY_DIR: u32 = 0o500;

/// The build directory under `cache`: `v1/<the one build there>`.
#[cfg(unix)]
fn build_dir(cache: &Path) -> PathBuf {
    let mut builds: Vec<PathBuf> = std::fs::read_dir(cache.join("v1"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    builds.sort();
    builds.pop().expect("the device made its build's directory")
}

/// Set `dir`'s permission bits to `mode`.
#[cfg(unix)]
fn set_mode(dir: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// An entry the directory will not take is announced once as
/// [`Degradation::ShaderCacheEntryNotWritten`] and ends nothing: the pipeline state is made, it
/// runs to the same output as one whose entries were written, and the device counts the failures.
/// The announcement names the directory and its cap. A device without unified memory announces
/// `SharedInSystemMemory` beside it, for the run's first `Shared` buffer.
///
/// The directory is made writable for the device to open (`Directory::open` checks that it takes
/// a write and fails the creation otherwise, which is the one shader-cache error there is), and
/// unwritable afterwards, which is the state a full disk or a directory someone else locked down
/// leaves. Unix only: it turns on the directory's mode bits. A process the mode bits do not bind
/// (one running as root, as in the Linux gate's container) writes into the directory all the
/// same, so nothing is refused there: the test checks that with a file of its own first, and where
/// the write goes through it says so (`CACHE-UNWRITABLE`) and holds nothing.
#[cfg(unix)]
#[test]
fn an_entry_that_cannot_be_written_is_announced_and_ends_nothing() {
    let case = public_case(CASE).unwrap();
    let writable = ScratchCache::new().unwrap();
    let expected = cache_run(writable.path(), CASE);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (statistics, output, notices, system_memory) = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let build = build_dir(cache.path());
        set_mode(&build, READ_ONLY_DIR);
        let probe = build.join("cache-unwritable-probe");
        if std::fs::write(&probe, b"probe").is_ok() {
            std::fs::remove_file(&probe).unwrap();
            set_mode(&build, 0o700);
            eprintln!(
                "CACHE-UNWRITABLE: this process wrote into a directory of mode {READ_ONLY_DIR:o} \
                 (the mode bits do not bind it, as they do not bind root), so no entry can be \
                 refused here; the test did not run"
            );
            return;
        }
        let made = pipeline_state(&device, &case);
        let state = match made {
            Ok(state) => state,
            Err(error) => {
                set_mode(&build, 0o700);
                panic!("a creation whose entries cannot be written must still make its state: {error:?}");
            }
        };
        let run = run_with(&device, &case, &state);
        set_mode(&build, 0o700);
        let output = run
            .expect("a run whose entries could not be written")
            .output;
        (
            device.shader_cache_statistics(),
            output,
            messages.take_notices(),
            super::degrade_notices::shared_in_system_memory(&device),
        )
    };
    assert_eq!(
        output, expected.output,
        "the output of a creation whose entries could not be written"
    );
    assert!(
        statistics.write_failures > 0,
        "the directory must have refused an entry at all: {statistics:?}"
    );
    let mut kinds: Vec<Degradation> = notices
        .iter()
        .filter_map(|notice| notice.degradation.as_ref().map(|d| d.degradation))
        .collect();
    kinds.sort();
    let mut expected = UNWRITTEN_ENTRY.to_vec();
    if system_memory {
        expected.push(Degradation::SharedInSystemMemory);
    }
    expected.sort();
    assert_eq!(
        kinds, expected,
        "the kinds a device whose cache entries could not be written announced"
    );
    let notice = notices
        .iter()
        .find(|notice| {
            notice
                .degradation
                .as_ref()
                .is_some_and(|d| UNWRITTEN_ENTRY.contains(&d.degradation))
        })
        .expect("the notice");
    let root = cache.path().display().to_string();
    let cap = metal2vulkan_runtime::DEFAULT_CAPACITY.to_string();
    assert!(
        notice.text.contains(&root) && notice.text.contains(&cap),
        "the notice names the directory and its cap: {}",
        notice.text
    );
}

/// The envelope kind of a translated function (`kind` is the `u32` at byte 16).
const FUNCTION: u32 = 1;

/// The first (by path) artifact entry of envelope kind `kind` under `cache`'s layout directory.
fn artifact_of_kind(cache: &Path, kind: u32) -> PathBuf {
    let mut found = Vec::new();
    for build in std::fs::read_dir(cache.join("v1")).unwrap() {
        for group in std::fs::read_dir(build.unwrap().path()).unwrap() {
            for entry in std::fs::read_dir(group.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                let bytes = std::fs::read(&path).unwrap();
                let artifact = bytes.starts_with(b"M2VARTF\0")
                    && bytes.get(16..20) == Some(&kind.to_le_bytes()[..]);
                if artifact {
                    found.push(path);
                }
            }
        }
    }
    found.sort();
    found.into_iter().next().expect("an artifact of that kind")
}

/// `bytes` in lowercase hex.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
