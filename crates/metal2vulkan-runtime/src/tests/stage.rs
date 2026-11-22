//! Kernels past the stage bound: the `stages/` directory holds at most its bounds, the most
//! recently used kept; an entry cut short, of another key or another driver's header is a miss and
//! removed; a memory breach holds up to the memory ceiling it was kept under, and a helper's own
//! end down to the helper version it was kept under; the wire reads its own version alone; and on
//! a device, a kernel past the bound is refused until its helper fills its entry, then made from
//! that entry alone, leaving the device's own pipeline cache as it was and its content holding no
//! shader module, and a breach refuses it with no helper asked again, a memory breach on a device
//! given its ceiling or a lower one, and a helper that ended before its reply once kept.

use crate::stage::{keep_entry, read_entry, read_frame, write_frame, Entry, Framing, StageBounds};
use crate::StageBreach;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// A directory of its own under the OS temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("m2v-rt-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A pipeline cache header of a device.
const HEADER: [u8; 32] = [7; 32];

/// The key numbered `at`.
fn key(at: u8) -> [u8; 32] {
    [at; 32]
}

/// Cache data of `len` bytes, beginning with [`HEADER`].
fn data(len: usize) -> Vec<u8> {
    let mut data = HEADER.to_vec();
    data.resize(len, 0x5a);
    data
}

/// The keys of the entries `dir` holds, sorted.
fn held(dir: &Path) -> Vec<u8> {
    let mut held: Vec<u8> = (fs::read_dir(dir).expect("the directory").flatten())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            u8::from_str_radix(name.get(..2)?, 16).ok()
        })
        .collect();
    held.sort_unstable();
    held
}

/// Writes key `at`'s entry as the key, then `kept`, framed as the directory frames an entry.
fn write_raw(dir: &Path, at: u8, kept: &[u8]) {
    let body = [&key(at)[..], kept].concat();
    let framed = [&crate::cache::frame(&body)[..], &body[..]].concat();
    fs::write(dir.join(crate::cache::hex(&key(at))), framed).expect("an entry");
}

/// Marks key `at`'s entry as last used `seconds` after the epoch.
fn used_at(dir: &Path, at: u8, seconds: u64) {
    let path = dir.join(crate::cache::hex(&key(at)));
    let file = File::options().write(true).open(path).expect("the entry");
    let time = SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
    file.set_modified(time).expect("its time");
}

/// Past its count bound and past its byte bound, the directory keeps the most recently used
/// entries that fit, the one just written among them; a hit marks an entry used, so it outlives
/// older ones.
#[test]
fn the_stage_directory_keeps_its_bounds_oldest_use_out_first() {
    let scratch = Scratch::new("stage-bounds");
    let dir = &scratch.0;
    // By count: four entries of 100 bytes under a bound of three.
    let bounds = StageBounds {
        bytes: 1 << 20,
        entries: 3,
    };
    for at in 1..=3 {
        assert!(keep_entry(dir, &key(at), &Entry::Data(data(100)), bounds));
        used_at(dir, at, 1000 + u64::from(at));
    }
    // Key 1, the oldest, is used again: it is now the newest.
    assert_eq!(
        read_entry(dir, &key(1), &HEADER, 0),
        Some(Entry::Data(data(100)))
    );
    assert!(keep_entry(dir, &key(4), &Entry::Data(data(100)), bounds));
    assert_eq!(held(dir), [1, 3, 4], "key 2, the oldest use, went");
    // By bytes: each entry is its frame, its key and its kind beside the data. A bound of two
    // and a half entries keeps two.
    let len = fs::metadata(dir.join(crate::cache::hex(&key(4))))
        .expect("an entry")
        .len();
    let bounds = StageBounds {
        bytes: len * 5 / 2,
        entries: 32,
    };
    for (at, seconds) in [(1, 2000), (3, 2001), (4, 2002)] {
        used_at(dir, at, seconds);
    }
    assert!(keep_entry(dir, &key(5), &Entry::Data(data(100)), bounds));
    assert_eq!(held(dir), [4, 5], "the two most recently used fit");
    // A breach is an entry of the same directory, under the same bounds: of two entries, it and
    // key 5, written after key 4 was last used.
    let bounds = StageBounds {
        bytes: 1 << 20,
        entries: 2,
    };
    let breach = Entry::Breach(StageBreach::Memory, 0);
    assert!(keep_entry(dir, &key(6), &breach, bounds));
    assert_eq!(held(dir), [5, 6]);
    assert_eq!(read_entry(dir, &key(6), &HEADER, 0), Some(breach));
}

/// An entry cut short, one under another key's name, one whose data begins with another driver's
/// header, and one of an unknown breach are each a miss, and removed.
#[test]
fn a_cut_or_foreign_entry_is_a_miss_and_is_removed() {
    let scratch = Scratch::new("stage-foreign");
    let dir = &scratch.0;
    let bounds = crate::stage::STAGE_BOUNDS;
    let path = |at: u8| dir.join(crate::cache::hex(&key(at)));
    // Cut short.
    assert!(keep_entry(dir, &key(1), &Entry::Data(data(64)), bounds));
    let whole = fs::read(path(1)).expect("the entry");
    fs::write(path(1), &whole[..whole.len() - 1]).expect("cut");
    assert_eq!(read_entry(dir, &key(1), &HEADER, 0), None);
    assert!(!path(1).exists(), "the cut entry is removed");
    // Another key's entry under this key's name.
    assert!(keep_entry(dir, &key(2), &Entry::Data(data(64)), bounds));
    fs::rename(path(2), path(3)).expect("renamed");
    assert_eq!(read_entry(dir, &key(3), &HEADER, 0), None);
    assert!(!path(3).exists(), "the foreign entry is removed");
    // Data of another driver's header.
    assert!(keep_entry(dir, &key(4), &Entry::Data(data(64)), bounds));
    assert_eq!(read_entry(dir, &key(4), &[8; 32], 0), None);
    assert!(!path(4).exists(), "another driver's data is removed");
    // Each breach reads back as itself, and an unknown one is a miss.
    let breaches = [
        StageBreach::Memory,
        StageBreach::Time,
        StageBreach::Compile,
        StageBreach::CompileRequired,
    ];
    for (at, breach) in (10..).zip(breaches) {
        assert!(keep_entry(dir, &key(at), &Entry::Breach(breach, 0), bounds));
        let read = read_entry(dir, &key(at), &HEADER, 0);
        assert_eq!(read, Some(Entry::Breach(breach, 0)));
    }
    // An unknown breach, a time breach followed by a ceiling, and a memory breach followed by
    // half of one.
    for kept in [
        &[1, 0xff][..],
        &[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        &[1, 0, 0, 0, 0, 0],
    ] {
        write_raw(dir, 20, kept);
        assert_eq!(read_entry(dir, &key(20), &HEADER, 0), None, "{kept:?}");
        assert!(!path(20).exists(), "{kept:?} is removed");
    }
    // A key with no file is a miss.
    assert_eq!(read_entry(dir, &key(21), &HEADER, 0), None);
}

/// A memory breach kept under a ceiling holds at that ceiling and below; above it, the entry is a
/// miss and removed, so the make asks for a fill, and a fill's entry replaces the breach. A memory
/// breach of its byte alone holds at 0 alone. The other breaches keep their byte alone and hold,
/// and data reads back, under every ceiling.
#[test]
fn a_memory_breach_holds_to_its_ceiling_and_is_a_miss_above_it() {
    let scratch = Scratch::new("stage-ceiling");
    let dir = &scratch.0;
    let bounds = crate::stage::STAGE_BOUNDS;
    let path = |at: u8| dir.join(crate::cache::hex(&key(at)));
    let len = |at: u8| fs::metadata(path(at)).expect("the entry").len();
    let memory = |ceiling| Entry::Breach(StageBreach::Memory, ceiling);
    let a = 8 << 30;
    assert!(keep_entry(dir, &key(1), &memory(a), bounds));
    assert_eq!(len(1), 66, "the frame, the key, the breach and its ceiling");
    for ceiling in [0, a - 1, a] {
        let read = read_entry(dir, &key(1), &HEADER, ceiling);
        assert_eq!(read, Some(memory(a)), "at {ceiling}");
    }
    assert_eq!(read_entry(dir, &key(1), &HEADER, a + 1), None);
    assert!(
        !path(1).exists(),
        "the breach under a lower ceiling is removed"
    );
    assert!(keep_entry(dir, &key(2), &memory(a), bounds));
    assert!(keep_entry(dir, &key(2), &Entry::Data(data(64)), bounds));
    let read = read_entry(dir, &key(2), &HEADER, a);
    assert_eq!(
        read,
        Some(Entry::Data(data(64))),
        "the fill replaced the breach"
    );
    for ceiling in [0, 1, a] {
        write_raw(dir, 3, &[1, 0]);
        let read = read_entry(dir, &key(3), &HEADER, ceiling);
        assert_eq!(read, (ceiling == 0).then_some(memory(0)), "at {ceiling}");
    }
    let others = [
        StageBreach::Time,
        StageBreach::Compile,
        StageBreach::CompileRequired,
    ];
    for (at, breach) in (10..).zip(others) {
        assert!(keep_entry(dir, &key(at), &Entry::Breach(breach, 0), bounds));
        assert_eq!(len(at), 58, "{breach:?}: the frame, the key and the breach");
        for ceiling in [0, a, u64::MAX] {
            let read = read_entry(dir, &key(at), &HEADER, ceiling);
            assert_eq!(
                read,
                Some(Entry::Breach(breach, 0)),
                "{breach:?} at {ceiling}"
            );
        }
    }
    assert!(keep_entry(dir, &key(20), &Entry::Data(data(64)), bounds));
    let read = read_entry(dir, &key(20), &HEADER, u64::MAX);
    assert_eq!(read, Some(Entry::Data(data(64))));
}

/// The helper version this file last saw the helper's code at, with that code's digest
/// ([`helper_code`]). A change to the helper's code fails
/// [`an_ended_breach_holds_to_its_helper_version_and_is_a_miss_below_it`] until
/// `HELPER_VERSION` is bumped and the pair here is the new one.
const HELPER_CODE: (u32, &str) = (
    1,
    "e07cb0488c4727ba561f38bd47ba8e978ddffececb964a5762d5e6b7384ef0d6",
);

/// The helper's own code as stage.rs holds it: `serve_stage` through `compile`, and
/// `read_request`, each line trimmed, blank and comment lines left out.
fn helper_code() -> String {
    use sha2::{Digest, Sha256};
    let source = include_str!("../stage.rs");
    let span = |from: &str, to: &str| {
        let start = source
            .find(from)
            .unwrap_or_else(|| panic!("stage.rs: {from}"));
        let end = source[start..]
            .find(to)
            .unwrap_or_else(|| panic!("stage.rs: {to}"));
        &source[start..start + end]
    };
    let code = [
        span("pub fn serve_stage(", "\n/// What an entry holds."),
        span("fn read_request(", "\n}\n"),
    ];
    let lines: Vec<&str> = (code.iter().flat_map(|text| text.lines()))
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .collect();
    let digest = Sha256::digest(lines.join("\n").as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A helper's own end kept as a breach holds the helper version it was kept under, 4 bytes after
/// its byte, under every memory ceiling; one kept under a lower version, or with no version or
/// another length after its byte, is a miss and removed. The helper's code is the code the
/// version was last bumped for, so a fix of it that misses the bump fails here.
#[test]
fn an_ended_breach_holds_to_its_helper_version_and_is_a_miss_below_it() {
    let scratch = Scratch::new("stage-ended");
    let dir = &scratch.0;
    let bounds = crate::stage::STAGE_BOUNDS;
    let path = |at: u8| dir.join(crate::cache::hex(&key(at)));
    let version = crate::stage::HELPER_VERSION;
    let ended = Entry::Breach(StageBreach::Ended, u64::from(version));
    assert!(keep_entry(dir, &key(1), &ended, bounds));
    let len = fs::metadata(path(1)).expect("the entry").len();
    assert_eq!(
        len, 62,
        "the frame, the key, the breach and its helper version"
    );
    for ceiling in [0, 8 << 30, u64::MAX] {
        let read = read_entry(dir, &key(1), &HEADER, ceiling);
        assert_eq!(read.as_ref(), Some(&ended), "at {ceiling}");
    }
    let lower = (version - 1).to_le_bytes();
    for kept in [
        &[1, 4, lower[0], lower[1], lower[2], lower[3]][..],
        &[1, 4],
        &[1, 4, 1],
    ] {
        write_raw(dir, 2, kept);
        assert_eq!(read_entry(dir, &key(2), &HEADER, 0), None, "{kept:?}");
        assert!(!path(2).exists(), "{kept:?} is removed");
    }
    let code = helper_code();
    assert_eq!(
        (version, code.as_str()),
        HELPER_CODE,
        "the helper's code changed since HELPER_VERSION was bumped: bump it and record the new pair"
    );
}

/// A frame reads back whole; one of another wire version is refused as such; one cut short or
/// changed in one byte reads as cut.
#[test]
fn the_wire_reads_its_own_version_alone() {
    let payload: Vec<u8> = (0..=255).collect();
    let mut wire = Vec::new();
    write_frame(&mut wire, &payload).expect("written");
    assert_eq!(read_frame(&mut wire.as_slice()).ok(), Some(payload));
    // The version, after the 8-byte magic.
    let mut other = wire.clone();
    other[8] ^= 0x40;
    assert!(matches!(
        read_frame(&mut other.as_slice()),
        Err(Framing::Version)
    ));
    let cut = &wire[..wire.len() - 1];
    assert!(matches!(read_frame(&mut &cut[..]), Err(Framing::Cut)));
    let mut changed = wire.clone();
    *changed.last_mut().expect("a byte") ^= 1;
    assert!(matches!(
        read_frame(&mut changed.as_slice()),
        Err(Framing::Cut)
    ));
}

/// A consumer joins the makes and dispatches that ask for one pipeline's fill by its request:
/// cloned, sent between threads, and compared and hashed by its stage key.
#[test]
fn a_request_is_a_key_a_consumer_joins_fills_by() {
    fn joins<T: Clone + Send + Sync + Eq + std::hash::Hash>() {}
    joins::<crate::StageRequest>();
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use super::Scratch;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        serve_stage, ComputePipelineDescriptor, Device, DeviceOptions, Dispatch, Error,
        FunctionDescriptor, Gap, Library, StageBreach, StageGap, StageRequest, StageSource,
    };
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    use std::path::Path;
    use std::sync::atomic::Ordering::Relaxed;

    /// The tiling kernel of the engine fixture `a1_bool_from_float.ll`, whose one `VkPipeline` the
    /// state's make makes; None where `llvm-as` is missing.
    fn tiled() -> Option<ComputePipelineDescriptor> {
        let name = "a1_bool_from_float";
        let bitcode = match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return None;
            }
            Err(error) => panic!("{name}: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function(name).expect("the kernel");
        Some(ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        )))
    }

    /// A device on `dir`, or with no cache directory, given the stage memory ceiling `ceiling`.
    fn on(
        name: &str,
        physical: &crate::PhysicalDevice,
        dir: Option<&Path>,
        ceiling: u64,
    ) -> Device {
        let options = DeviceOptions {
            pipeline_cache_dir: dir.map(Path::to_owned),
            stage_memory_ceiling: ceiling,
            ..DeviceOptions::default()
        };
        Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// With the stage bound below every kernel's weight, the tiling kernel is past it. On a
    /// device with no cache directory, and on one whose driver writes no pipeline cache header
    /// (lavapipe), its make is refused ([`Gap::StageSize`]). Elsewhere it is refused unfilled; a
    /// helper served over two pipes on a thread of the test fills its entry, and the make made
    /// again creates the `VkPipeline` from that entry alone, with the fail-on-compile-required
    /// flag, counted, leaving the device's own pipeline cache's data as it was; the fill answers
    /// the helper's device creation and compile times. The content holds no shader module: a
    /// later variant, an indirect dispatch's at its own size, is refused unfilled with a request of
    /// its own, and once filled is made from its entry. A breach recorded for the request refuses a
    /// later device's make, with no helper asked. A memory breach kept under ceiling 0 is a miss
    /// on a device given ceiling 1, which asks for the same fill, and one kept there refuses there.
    #[test]
    fn a_kernel_past_the_stage_bound_is_made_only_from_its_helpers_fill() {
        let Some(descriptor) = tiled() else {
            return;
        };
        crate::stage::TEST_BOUND.store(1, Relaxed);
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let make = |device: &Device| {
            device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
        };
        for (name, physical) in &devices {
            let refused = make(&on(name, physical, None, 0)).err();
            assert!(
                matches!(
                    refused,
                    Some(Error::Unsupported(Gap::StageSize {
                        cause: StageGap::NoCacheDirectory,
                        ..
                    }))
                ),
                "{name}: {refused:?}"
            );
            let scratch = Scratch::new("stage-fill");
            let device = on(name, physical, Some(&scratch.0), 0);
            if !crate::tests::keeps_a_blob(name, &device) {
                let refused = make(&device).err();
                assert!(
                    matches!(
                        refused,
                        Some(Error::Unsupported(Gap::StageSize {
                            cause: StageGap::NoPipelineCache,
                            ..
                        }))
                    ),
                    "{name}: {refused:?}"
                );
                continue;
            }
            let unfilled = || match make(&device) {
                Err(Error::StageUnfilled(request)) => request,
                other => panic!("{name}: {:?}", other.err()),
            };
            let request = unfilled();
            assert!(request.weight() > 1, "{name}: {request:?}");
            // Each make of the pipeline asks for the same fill.
            assert_eq!(unfilled(), request, "{name}");
            let core = device.core();
            // SAFETY: the device is the caches'.
            let shared = unsafe { core.caches.data(&core.device) }.map(|data| data.len());
            let helpers: &crate::Instance = &instance;
            // A helper served over two pipes on a thread of the test fills `request`'s entry.
            let fill = |request: &StageRequest| {
                let (helper_in, to_helper) = std::io::pipe().expect("a pipe");
                let (from_helper, helper_out) = std::io::pipe().expect("a pipe");
                let served = std::thread::scope(|scope| {
                    let helper = scope.spawn(move || {
                        serve_stage(helpers, DeviceOptions::default(), helper_in, helper_out)
                    });
                    let filled = device.fill_stage(request, to_helper, from_helper);
                    (filled, helper.join().expect("the helper's thread"))
                });
                served.1.expect("the helper's pipes");
                (served.0).unwrap_or_else(|fill| panic!("{name}: the fill: {fill}"))
            };
            let spans = fill(&request);
            // The helper's own times: its device's creation and its compile, each a span.
            assert!(spans.device_nanos > 0, "{name}: {spans:?}");
            assert!(spans.compile_nanos > 0, "{name}: {spans:?}");
            let counters = device.counters();
            assert_eq!(counters.metrics().stage_fills, 1, "{name}");
            let state = make(&device).unwrap_or_else(|error| panic!("{name}: {error}"));
            let metrics = counters.metrics();
            assert_eq!(metrics.pipelines_from_stage_fills, 1, "{name}");
            // SAFETY: as above.
            let after = unsafe { core.caches.data(&core.device) }.map(|data| data.len());
            assert_eq!(after, shared, "{name}: the device's own pipeline cache");
            // The content holds no shader module: its make made one of its own and destroyed it
            // once the create returned.
            let whole = &state.inner.compiled.whole;
            assert!(!whole.holds_module(), "{name}: the content's shader module");
            // A later variant, an indirect dispatch's at its own threadgroup size, asks for a
            // fill of its own, and once filled is made from its entry, with a module of its own.
            let (dispatch, size) = (Dispatch::Indirect, [8, 1, 1]);
            let later = match state.prepare_dispatch(dispatch, size, &[]) {
                Err(Error::StageUnfilled(later)) => later,
                other => panic!("{name}: {other:?}"),
            };
            assert_ne!(later, request, "{name}: the later variant's pipeline");
            fill(&later);
            (state.prepare_dispatch(dispatch, size, &[]))
                .unwrap_or_else(|error| panic!("{name}: the later variant: {error}"));
            let made = state.dispatch_variant(dispatch, size, &[]);
            assert_eq!(made.map(|(_, made)| made), Some(true), "{name}");
            let metrics = counters.metrics();
            assert_eq!(metrics.stage_fills, 2, "{name}");
            assert_eq!(metrics.pipelines_from_stage_fills, 2, "{name}");
            assert!(!whole.holds_module(), "{name}: the content's shader module");
            drop(state);
            drop(device);
            let breached = |device: &Device| {
                device.record_stage_breach(&request, StageBreach::Memory);
                let refused = make(device).err();
                assert!(
                    matches!(
                        refused,
                        Some(Error::Unsupported(Gap::StageSize {
                            cause: StageGap::Breach(StageBreach::Memory),
                            ..
                        }))
                    ),
                    "{name}: {refused:?}"
                );
            };
            // A breach refuses a later device's make, with no helper asked.
            let device = on(name, physical, Some(&scratch.0), 0);
            breached(&device);
            drop(device);
            // That memory breach, kept under ceiling 0, is a miss on a device given ceiling 1,
            // which asks for the same fill; one kept there refuses there.
            let device = on(name, physical, Some(&scratch.0), 1);
            match make(&device) {
                Err(Error::StageUnfilled(again)) => assert_eq!(again, request, "{name}"),
                other => panic!("{name}: {:?}", other.err()),
            }
            breached(&device);
            println!(
                "{name}: refused unfilled, filled by its helper (device {} ms, compile {} ms), made \
                 from the entry alone, refused by its memory breach at ceiling 0, asked again at \
                 ceiling 1, then refused by a breach kept there",
                spans.device_nanos / 1_000_000,
                spans.compile_nanos / 1_000_000
            );
        }
    }

    /// A helper that ends before its reply, as one whose driver faults in the compile does, gives
    /// the fill [`crate::StageFill::Ended`]; the consumer keeps that end as
    /// [`StageBreach::Ended`], and the next make, and a later device's on the same directory, is
    /// refused ([`Gap::StageSize`]) with no helper asked again and no fill counted.
    #[test]
    fn a_helper_that_ends_before_its_reply_is_kept_as_an_ended_breach() {
        let Some(descriptor) = tiled() else {
            return;
        };
        crate::stage::TEST_BOUND.store(1, Relaxed);
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let make = |device: &Device| {
            device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers)
        };
        let refused_as_ended = |name: &str, device: &Device| {
            let refused = make(device).err();
            assert!(
                matches!(
                    refused,
                    Some(Error::Unsupported(Gap::StageSize {
                        cause: StageGap::Breach(StageBreach::Ended),
                        ..
                    }))
                ),
                "{name}: {refused:?}"
            );
        };
        for (name, physical) in &devices {
            let scratch = Scratch::new("stage-ended-device");
            let device = on(name, physical, Some(&scratch.0), 0);
            if !crate::tests::keeps_a_blob(name, &device) {
                continue;
            }
            let request = match make(&device) {
                Err(Error::StageUnfilled(request)) => request,
                other => panic!("{name}: {:?}", other.err()),
            };
            // The helper reads the request whole and ends with no reply.
            let (mut helper_in, to_helper) = std::io::pipe().expect("a pipe");
            let (from_helper, helper_out) = std::io::pipe().expect("a pipe");
            let filled = std::thread::scope(|scope| {
                scope.spawn(move || {
                    std::io::copy(&mut helper_in, &mut std::io::sink()).expect("the request");
                    drop(helper_out);
                });
                device.fill_stage(&request, to_helper, from_helper)
            });
            assert_eq!(filled.err(), Some(crate::StageFill::Ended), "{name}");
            device.record_stage_breach(&request, StageBreach::Ended);
            refused_as_ended(name, &device);
            drop(device);
            let device = on(name, physical, Some(&scratch.0), 0);
            refused_as_ended(name, &device);
            assert_eq!(device.counters().metrics().stage_fills, 0, "{name}");
            println!("{name}: a helper's own end before its reply kept, the make refused twice");
        }
    }

    /// With the kernels' stage bound below every stage's weight, a render state is still made in
    /// this process, as a render stage is refused only past the render stages' own bound, and it
    /// is counted: one translated render pipeline past the kernels' bound, at its heaviest stage's
    /// weight.
    #[test]
    fn a_render_state_past_the_kernel_bound_is_made_in_process_and_counted() {
        use crate::tests::draw::on_device::{descriptor, examples, state};
        let Some(library) = examples() else {
            return;
        };
        crate::stage::TEST_BOUND.store(1, Relaxed);
        let d = descriptor(&library, "colored_fragment", crate::PixelFormat::RGBA8Unorm);
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = on(name, physical, None, 0);
            state(name, &device, &d);
            let m = device.counters().metrics();
            assert_eq!(m.render_makes_past_kernel_bound, 1, "{name}");
            assert!(m.render_weight_past_kernel_bound_max > 1, "{name}: {m:?}");
            state(name, &device, &d);
            let again = device.counters().metrics();
            assert_eq!(
                again.render_makes_past_kernel_bound, 1,
                "{name}: the state is translated once"
            );
        }
    }
}
