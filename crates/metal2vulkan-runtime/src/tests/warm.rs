//! Pipeline records (`warm/record.rs`) and the warm pass (`warm/pass.rs`): a device with a record
//! hook hands it one record per pipeline state creation that made its state; a device handed the
//! records an earlier process's device made over the same shader cache directory makes their
//! translations and `VkPipeline`s before the program creates them, takes nothing the program sees,
//! stops at its byte budget, and announces a record it cannot read and goes on.
//!
//! # What is counted, and why counts
//!
//! Every assertion is a count of the structure (records handed over and warmed, libraries
//! written, stages decoded, functions translated, identifiers taken, messages heard), never a
//! time: a count holds on any host under any load. The times the cross-process test prints (the
//! creations on a warmed device, on an unwarmed one over the same directory, and on a cold one)
//! are figures of this host, printed and not asserted.
//!
//! Each assertion that a count stayed still is paired with one that the same structure moved
//! elsewhere (the cold device translates, the program's creation takes an identifier, a creation
//! that made its state hands the hook a record), so a pass or a hook that did nothing at all
//! cannot pass it.
//!
//! Each test owns empty cache directories, removed when dropped. The cross-process test's
//! recording half runs as a process of its own.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll` and
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::Scratch;
use super::descriptor_key_once::descriptor;
use crate::{
    BinaryArchiveDescriptor, Device, DeviceKind, DeviceOptions, MessageSink, PipelineOptions,
    PipelineRecord, ShaderCacheOptions, WarmStatistics,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The states [`make_states`] creates, so the records each recording hands over.
const STATES: u64 = 3;

/// The environment variables that make [`record_process`] the recording half of
/// [`a_later_process_warms_what_an_earlier_one_recorded`].
const DIRECTORY: &str = "M2V_TEST_WARM_DIRECTORY";
const RECORDS: &str = "M2V_TEST_WARM_RECORDS";

/// What a device's message sink heard: its degrade notices, and the id names of its other
/// messages.
#[derive(Default)]
struct Heard {
    notices: AtomicU64,
    names: Mutex<Vec<String>>,
}

impl Heard {
    /// How many messages of id name `name` it heard.
    fn named(&self, name: &str) -> usize {
        self.names
            .lock()
            .unwrap()
            .iter()
            .filter(|heard| *heard == name)
            .count()
    }
}

/// A device over `cache`, handing its records to `records` where given, its messages to `heard`
/// where given.
fn device(
    cache: &Path,
    records: Option<&Arc<Mutex<Vec<PipelineRecord>>>>,
    heard: Option<&Arc<Heard>>,
) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    if let Some(records) = records {
        let records = Arc::clone(records);
        options.pipeline_records = Some(Arc::new(move |record| {
            records.lock().unwrap().push(record);
        }));
    }
    if let Some(heard) = heard {
        let heard = Arc::clone(heard);
        options.messages = Some(MessageSink::new(move |message| {
            match message.degradation.is_some() {
                true => heard.notices.fetch_add(1, Ordering::Relaxed),
                false => {
                    heard.names.lock().unwrap().push(message.id_name.clone());
                    0
                }
            };
        }));
    }
    Device::system_default_with(&options).unwrap()
}

/// The compute descriptor of `runtime_examples`' `scale_and_offset`, requiring its threadgroup
/// (so its creation makes its first dispatch's `VkPipeline`).
fn compute(device: &Device) -> ComputePipelineDescriptor {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let function = device
        .new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
        )
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let mut compute =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    compute.required_threads_per_threadgroup = Some([32, 1, 1]);
    compute
}

/// The descriptors of [`make_states`] on `device`: the `ubo_bind_offsets` pair's render
/// descriptor and [`compute`]'s. Their libraries are assembled here (`llvm-as`), apart from the
/// creations a test times.
fn descriptors(device: &Device) -> (RenderPipelineDescriptor, ComputePipelineDescriptor) {
    (descriptor(device), compute(device))
}

/// The states every test creates, in this order: a render state of the `ubo_bind_offsets` pair,
/// a compute state of [`compute`], and the same kernel's state with indirect command buffer
/// support (so its creation takes a resource ID). Each makes a predicted `VkPipeline`.
fn make_states(device: &Device) -> Vec<Box<dyn std::any::Any>> {
    make(device, &descriptors(device))
}

/// [`make_states`] of `descriptors`, [`descriptors`]' on `device`.
fn make(
    device: &Device,
    (render, compute): &(RenderPipelineDescriptor, ComputePipelineDescriptor),
) -> Vec<Box<dyn std::any::Any>> {
    let options = PipelineOptions::default();
    let render = device.new_render_pipeline_state(render, &options).unwrap();
    let mut compute = compute.clone();
    let plain = device
        .new_compute_pipeline_state(&compute, &options)
        .unwrap();
    compute.support_indirect_command_buffers = true;
    let indirect = device
        .new_compute_pipeline_state(&compute, &options)
        .unwrap();
    vec![Box::new(render), Box::new(plain), Box::new(indirect)]
}

/// The records a device over `cache` hands over for [`make_states`]; the device is gone when
/// this returns, so its cache thread wrote what it was handed.
fn recorded(cache: &Path) -> Vec<PipelineRecord> {
    let records = Arc::new(Mutex::new(Vec::new()));
    let device = device(cache, Some(&records), None);
    drop(make_states(&device));
    assert_eq!(device.warm_statistics().records_made, STATES);
    drop(device);
    let records = records.lock().unwrap().clone();
    assert_eq!(records.len() as u64, STATES, "one record per creation");
    records
}

/// `device`'s statistics once the warm passes over `records` records are done (each record
/// counted in one outcome), at most two minutes from now.
fn warmed(device: &Device, records: u64) -> WarmStatistics {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let s = device.warm_statistics();
        let done = s.records_warmed
            + s.records_translated
            + s.records_failed
            + s.records_unread
            + s.records_not_warmable
            + s.records_not_run;
        if done >= records {
            return s;
        }
        assert!(
            Instant::now() < deadline,
            "the warm pass did not finish: {s:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// What [`make_states`]' creations cost `device`, their libraries assembled before: their wall
/// time, and the stages they decoded, the functions they translated, the translations they
/// looked up in the shader cache directory, the driver pipeline caches they loaded from it, the
/// shader modules they made and the variant `VkPipeline`s they made.
fn creations(device: &Device) -> (Duration, [u64; 6]) {
    let descriptors = descriptors(device);
    let before = (device.work_counts(), device.shader_cache_statistics());
    let started = Instant::now();
    let states = make(device, &descriptors);
    let took = started.elapsed();
    let after = (device.work_counts(), device.shader_cache_statistics());
    drop(states);
    (
        took,
        [
            after.0.stage_entries_specialized - before.0.stage_entries_specialized,
            after.0.functions_translated - before.0.functions_translated,
            (after.1.hits + after.1.misses) - (before.1.hits + before.1.misses),
            after.1.driver_loads - before.1.driver_loads,
            after.0.shader_modules_made - before.0.shader_modules_made,
            after.0.variant_pipelines_made - before.0.variant_pipelines_made,
        ],
    )
}

/// Each creation that made its state hands the hook one record, in the order made, and counts
/// it; a creation that fails and a binary archive's add hand it nothing; the same descriptor
/// created again makes the same record; a record carried through its bytes is the same record;
/// and the two libraries the records name are written to the directory once each, however many
/// records name them.
#[test]
fn each_creation_hands_the_hook_one_record() {
    let scratch = Scratch::new("warm-hook");
    let records = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, Some(&records), None);
    let states = make_states(&device);
    let s = device.warm_statistics();
    assert_eq!(
        (s.records_made, s.records_unmade),
        (STATES, 0),
        "one record per creation: {s:?}"
    );
    assert_eq!(records.lock().unwrap().len() as u64, STATES);
    assert_eq!(
        s.libraries_kept, 2,
        "the render pair's library and the kernel's, each written once: {s:?}"
    );

    // A kernel as a vertex function: Metal's creation fails, and so no record.
    let kernel = compute(&device).kernel;
    let failing = RenderPipelineDescriptor::new(kernel);
    assert!(
        device
            .new_render_pipeline_state(&failing, &PipelineOptions::default())
            .is_err(),
        "a kernel as a vertex function is refused"
    );
    // An archive's add makes the state the creation would, and hands the hook nothing.
    let archive = device
        .new_binary_archive(&BinaryArchiveDescriptor::new(None))
        .unwrap();
    archive
        .add_compute_pipeline_functions(&compute(&device))
        .unwrap();
    assert_eq!(
        device.warm_statistics().records_made,
        STATES,
        "a failed creation or an archive's add made a record"
    );

    // The render state again: the same record, its library not written again.
    let again = device
        .new_render_pipeline_state(&descriptor(&device), &PipelineOptions::default())
        .unwrap();
    let s = device.warm_statistics();
    assert_eq!((s.records_made, s.libraries_kept), (STATES + 1, 2), "{s:?}");
    let records = records.lock().unwrap().clone();
    assert_eq!(
        records[STATES as usize], records[0],
        "equal creations make equal records"
    );
    assert_ne!(records[1], records[2], "two descriptors make two records");
    for record in &records {
        assert_eq!(
            PipelineRecord::from_bytes(&record.to_bytes()).as_ref(),
            Some(record),
            "a record read from its own bytes is the same record"
        );
    }
    drop((states, again));
}

/// One process records, and a later one warms its records, then makes the same states: the
/// warmed device's creations decode no stage, translate nothing, look nothing up in the
/// directory and load no driver pipeline cache, where an unwarmed device over the same directory
/// looks its translations up and loads its driver data in its creations, and a cold device
/// translates every function. The creations' times on the three devices are printed.
#[test]
fn a_later_process_warms_what_an_earlier_one_recorded() {
    let scratch = Scratch::new("warm-processes");
    let cache = scratch.0.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let file = scratch.0.join("records");
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::warm::record_process",
            "--ignored",
            "--test-threads=1",
        ])
        .env(DIRECTORY, &cache)
        .env(RECORDS, &file)
        .status()
        .unwrap();
    assert!(status.success(), "the recording process failed");
    let records = read_records(&file);
    assert_eq!(records.len() as u64, STATES);

    // The same directory without a warm pass: what the program's creations pay there.
    let unwarmed = device(&cache, None, None);
    let (unwarmed_took, unwarmed_made) = creations(&unwarmed);
    drop(unwarmed);

    let warm = device(&cache, None, None);
    let started = Instant::now();
    warm.warm(records, u64::MAX);
    let s = warmed(&warm, STATES);
    let pass_took = started.elapsed();
    assert_eq!(
        (s.records_warmed, s.records_failed, s.records_unread),
        (STATES, 0, 0),
        "every record warmed: {s:?}"
    );
    let (warm_took, warm_made) = creations(&warm);
    assert_eq!(
        warm_made[..4],
        [0, 0, 0, 0],
        "the warmed device's creations decode, translate, look up and load nothing: the pass did"
    );
    assert!(
        unwarmed_made[2] > 0 && unwarmed_made[3] > 0,
        "an unwarmed device's creations look their translations up and load their driver data, \
         or the warmed leg's zeros above are vacuous: {unwarmed_made:?}"
    );

    let cold_scratch = Scratch::new("warm-processes-cold");
    let cold = device(&cold_scratch.0, None, None);
    let (cold_took, cold_made) = creations(&cold);
    assert!(
        cold_made[1] > 0,
        "a cold device translates its functions, or the warm leg's zero above is vacuous"
    );
    let cache_stats = warm.shader_cache_statistics();
    println!(
        "warm pass: {} records warmed in {pass_took:?}, {} driver bytes added ({} driver pipeline \
         hits, {} misses over the warmed device). The {STATES} creations, as [stages decoded, \
         functions translated, directory lookups, driver caches loaded, shader modules made, \
         variant pipelines made]: {warm_took:?} \
         {warm_made:?} on the warmed device, {unwarmed_took:?} {unwarmed_made:?} on an unwarmed \
         device over the same directory, {cold_took:?} {cold_made:?} on a cold one",
        s.records_warmed,
        s.pipeline_bytes,
        cache_stats.driver_pipeline_hits,
        cache_stats.driver_pipeline_misses,
    );
}

/// The recording half of [`a_later_process_warms_what_an_earlier_one_recorded`], run as its own
/// process: records [`make_states`] over the directory it is named, and writes the records'
/// bytes to the file it is named, each after its length.
#[test]
#[ignore = "a child process of a_later_process_warms_what_an_earlier_one_recorded"]
fn record_process() {
    let cache = PathBuf::from(std::env::var_os(DIRECTORY).expect("the parent names the cache"));
    let file = PathBuf::from(std::env::var_os(RECORDS).expect("the parent names the file"));
    let mut bytes = Vec::new();
    for record in recorded(&cache) {
        let record = record.to_bytes();
        bytes.extend_from_slice(&(record.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&record);
    }
    std::fs::write(file, bytes).unwrap();
}

/// The records [`record_process`] wrote to `file`.
fn read_records(file: &Path) -> Vec<PipelineRecord> {
    let bytes = std::fs::read(file).unwrap();
    let mut records = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let length = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        at += 8;
        records.push(PipelineRecord::from_bytes(&bytes[at..at + length]).expect("a record"));
        at += length;
    }
    records
}

/// A warm pass takes no identifier, no sampler ID and no live variant of the program's, hands
/// the record hook nothing and announces nothing: the program's states made after it number their
/// identifiers on from those made before it, as with no pass between them, and the program's
/// creations announce what a cold device's announce.
#[test]
fn a_warm_pass_takes_nothing_the_program_sees() {
    let recording = Scratch::new("warm-unseen-recording");
    let records = recorded(&recording.0);
    let scratch = Scratch::new("warm-unseen");
    let heard = Arc::new(Heard::default());
    let hook = Arc::new(Mutex::new(Vec::new()));
    let warm = device(&scratch.0, Some(&hook), Some(&heard));
    // Two plain states' identifiers a step apart, before the pass: the step a state takes.
    let options = PipelineOptions::default();
    let state = || {
        warm.new_render_pipeline_state(&descriptor(&warm), &options)
            .unwrap()
    };
    let first = state();
    let second = state();
    let step = second.unique_identifier() - first.unique_identifier();
    assert!(
        step > 0,
        "a plain state's identifier moves at each creation"
    );
    let heard_before = heard.notices.load(Ordering::Relaxed);
    let ids_before = warm.sampler_ids_live();
    let made_before = warm.warm_statistics().records_made;
    assert_eq!(made_before, 2, "the program's creations made records");
    // The render record first, alone, while the program holds two states of its descriptor: the
    // pass makes its own variant's objects (its stages take the device's shader modules) rather
    // than take the program's live variant, which it could be the last to let go of.
    let stages = |device: &Device| {
        let counts = device.work_counts();
        counts.shader_modules_made + counts.shader_modules_shared
    };
    let stages_before = stages(&warm);
    warm.warm(records[..1].to_vec(), u64::MAX);
    let s = warmed(&warm, 1);
    assert_eq!(s.records_warmed, 1, "{s:?}");
    assert!(
        stages(&warm) > stages_before,
        "the pass took the program's live variant"
    );
    warm.warm(records[1..].to_vec(), u64::MAX);
    let s = warmed(&warm, STATES);
    assert_eq!(s.records_warmed, STATES, "{s:?}");
    assert_eq!(
        heard.notices.load(Ordering::Relaxed),
        heard_before,
        "the pass announced"
    );
    assert_eq!(
        warm.sampler_ids_live(),
        ids_before,
        "the pass holds sampler IDs"
    );
    assert_eq!(
        warm.warm_statistics().records_made,
        made_before,
        "the pass handed the record hook its states"
    );
    let third = state();
    assert_eq!(
        third.unique_identifier() - second.unique_identifier(),
        step,
        "the pass took identifiers between the program's states"
    );
    drop((first, second, third));
    // The program's creations announce on the warmed device what they announce on a cold one:
    // the pass counted nothing announced.
    let heard_warm = heard.notices.load(Ordering::Relaxed);
    drop(make_states(&warm));
    let warm_notices = heard.notices.load(Ordering::Relaxed) - heard_warm;
    let cold_scratch = Scratch::new("warm-unseen-cold");
    let cold_heard = Arc::new(Heard::default());
    let cold = device(&cold_scratch.0, None, Some(&cold_heard));
    let cold_state = cold
        .new_render_pipeline_state(&descriptor(&cold), &options)
        .unwrap();
    let _ = cold_state.unique_identifier();
    let cold_before = cold_heard.notices.load(Ordering::Relaxed);
    drop(make_states(&cold));
    let cold_notices = cold_heard.notices.load(Ordering::Relaxed) - cold_before;
    assert_eq!(
        warm_notices, cold_notices,
        "the warmed device's creations announced otherwise than a cold device's"
    );
    println!("the creations announced {warm_notices} notice(s) on both devices");
}

/// A pass stops before its next record once the driver data its makings added reaches its byte
/// budget: none is made under a budget of 0 bytes, and only the first under a budget of 1 byte
/// where that record's makings added any.
#[test]
fn a_byte_budget_stops_the_pass() {
    let recording = Scratch::new("warm-budget-recording");
    let records = recorded(&recording.0);
    let none = Scratch::new("warm-budget-none");
    let device_none = device(&none.0, None, None);
    device_none.warm(records.clone(), 0);
    let s = warmed(&device_none, STATES);
    assert_eq!(
        (s.records_not_run, s.records_warmed, s.pipeline_bytes),
        (STATES, 0, 0),
        "a budget of nothing warms nothing: {s:?}"
    );
    let one = Scratch::new("warm-budget-one");
    let device_one = device(&one.0, None, None);
    device_one.warm(records, 1);
    let s = warmed(&device_one, STATES);
    match s.pipeline_bytes {
        0 => println!("the driver added no pipeline cache data: the 1-byte leg is not judged"),
        _ => assert_eq!(
            (s.records_warmed, s.records_not_run),
            (1, STATES - 1),
            "the pass goes on past a spent budget: {s:?}"
        ),
    }
}

/// A record whose library the directory no longer holds is counted unread and announced once,
/// as a warning, and the pass goes on: with the kernel's library removed from the directory, the
/// two compute records handed around the render one are unread, the render one is warmed between
/// them, and the device's sink hears the warning once.
#[test]
fn an_unread_record_is_announced_and_the_pass_goes_on() {
    let scratch = Scratch::new("warm-unread");
    let records = recorded(&scratch.0);
    // The kernel's library record: the content directory's file holding the library whose
    // function table names `scale_and_offset` (`shader_cache/libraries.rs`).
    let mut removed = 0;
    for entry in walk(&scratch.0) {
        let bytes = std::fs::read(&entry).unwrap();
        let named = bytes
            .windows(b"scale_and_offset".len())
            .any(|window| window == b"scale_and_offset");
        if bytes.starts_with(b"M2VLIBR1") && named {
            std::fs::remove_file(&entry).unwrap();
            removed += 1;
        }
    }
    assert_eq!(removed, 1, "the directory holds the kernel's library once");
    let heard = Arc::new(Heard::default());
    let warm = device(&scratch.0, None, Some(&heard));
    // Carried as bytes, as a later process holds them: a record made in this process keeps its
    // libraries and reads nothing from the directory.
    let carried = |at: usize| PipelineRecord::from_bytes(&records[at].to_bytes()).unwrap();
    let order = vec![carried(1), carried(0), carried(2)];
    warm.warm(order, u64::MAX);
    let s = warmed(&warm, STATES);
    assert_eq!(
        (s.records_unread, s.records_warmed, s.records_not_run),
        (2, 1, 0),
        "the compute records unread, the render one between them warmed: {s:?}"
    );
    assert_eq!(
        heard.named(crate::warm::UNREAD),
        1,
        "the device's first unread record is announced, once"
    );
}

/// Every file under `directory`, recursively.
fn walk(directory: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut left = vec![directory.to_owned()];
    while let Some(at) = left.pop() {
        for entry in std::fs::read_dir(&at).unwrap() {
            let path = entry.unwrap().path();
            match path.is_dir() {
                true => left.push(path),
                false => files.push(path),
            }
        }
    }
    files
}
