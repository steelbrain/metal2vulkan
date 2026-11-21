//! Kernels past the stage bound: the `stages/` directory holds at most its bounds, the most
//! recently used kept; an entry cut short, of another key or another driver's header is a miss and
//! removed; the wire reads its own version alone; and on a device, a kernel past the bound is
//! refused until its helper fills its entry, then made from that entry alone, leaving the device's
//! own pipeline cache as it was, and a breach refuses it with no helper asked again.

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
        read_entry(dir, &key(1), &HEADER),
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
    let breach = Entry::Breach(StageBreach::Memory);
    assert!(keep_entry(dir, &key(6), &breach, bounds));
    assert_eq!(held(dir), [5, 6]);
    assert_eq!(read_entry(dir, &key(6), &HEADER), Some(breach));
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
    assert_eq!(read_entry(dir, &key(1), &HEADER), None);
    assert!(!path(1).exists(), "the cut entry is removed");
    // Another key's entry under this key's name.
    assert!(keep_entry(dir, &key(2), &Entry::Data(data(64)), bounds));
    fs::rename(path(2), path(3)).expect("renamed");
    assert_eq!(read_entry(dir, &key(3), &HEADER), None);
    assert!(!path(3).exists(), "the foreign entry is removed");
    // Data of another driver's header.
    assert!(keep_entry(dir, &key(4), &Entry::Data(data(64)), bounds));
    assert_eq!(read_entry(dir, &key(4), &[8; 32]), None);
    assert!(!path(4).exists(), "another driver's data is removed");
    // Each breach reads back as itself, and an unknown one is a miss.
    let breaches = [
        StageBreach::Memory,
        StageBreach::Time,
        StageBreach::Compile,
        StageBreach::CompileRequired,
    ];
    for (at, breach) in (10..).zip(breaches) {
        assert!(keep_entry(dir, &key(at), &Entry::Breach(breach), bounds));
        let read = read_entry(dir, &key(at), &HEADER);
        assert_eq!(read, Some(Entry::Breach(breach)));
    }
    let mut body = key(20).to_vec();
    body.extend([1, 0xff]);
    let framed = [&crate::cache::frame(&body)[..], &body[..]].concat();
    fs::write(path(20), framed).expect("an unknown breach");
    assert_eq!(read_entry(dir, &key(20), &HEADER), None);
    assert!(!path(20).exists(), "the unknown breach is removed");
    // A key with no file is a miss.
    assert_eq!(read_entry(dir, &key(21), &HEADER), None);
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
        serve_stage, ComputePipelineDescriptor, Device, DeviceOptions, Error, FunctionDescriptor,
        Gap, Library, StageBreach, StageGap, StageSource,
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

    /// A device on `dir`, or with no cache directory.
    fn on(name: &str, physical: &crate::PhysicalDevice, dir: Option<&Path>) -> Device {
        let options = DeviceOptions {
            pipeline_cache_dir: dir.map(Path::to_owned),
            ..DeviceOptions::default()
        };
        Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// With the stage bound below every kernel's weight, the tiling kernel is past it. On a
    /// device with no cache directory, and on one whose driver writes no pipeline cache header
    /// (lavapipe), its make is refused ([`Gap::StageSize`]). Elsewhere it is refused unfilled; a
    /// helper served over two pipes on a thread of the test fills its entry, and the make made
    /// again creates the `VkPipeline` from that entry alone, with the fail-on-compile-required
    /// flag, counted, leaving the device's own pipeline cache's data as it was. A breach recorded
    /// for the request refuses a later device's make, with no helper asked.
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
            let refused = make(&on(name, physical, None)).err();
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
            let device = on(name, physical, Some(&scratch.0));
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
            let (helper_in, to_helper) = std::io::pipe().expect("a pipe");
            let (from_helper, helper_out) = std::io::pipe().expect("a pipe");
            let helpers: &crate::Instance = &instance;
            let served = std::thread::scope(|scope| {
                let helper = scope.spawn(move || {
                    serve_stage(helpers, DeviceOptions::default(), helper_in, helper_out)
                });
                let filled = device.fill_stage(&request, to_helper, from_helper);
                (filled, helper.join().expect("the helper's thread"))
            });
            assert_eq!(served.0, Ok(()), "{name}: the fill");
            served.1.expect("the helper's pipes");
            let counters = device.counters();
            assert_eq!(counters.metrics().stage_fills, 1, "{name}");
            let state = make(&device).unwrap_or_else(|error| panic!("{name}: {error}"));
            let metrics = counters.metrics();
            assert_eq!(metrics.pipelines_from_stage_fills, 1, "{name}");
            // SAFETY: as above.
            let after = unsafe { core.caches.data(&core.device) }.map(|data| data.len());
            assert_eq!(after, shared, "{name}: the device's own pipeline cache");
            drop(state);
            drop(device);
            // A breach refuses a later device's make, with no helper asked.
            let device = on(name, physical, Some(&scratch.0));
            device.record_stage_breach(&request, StageBreach::Memory);
            let refused = make(&device).err();
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
            println!(
                "{name}: refused unfilled, filled by its helper, made from the entry alone, then \
                 refused by its breach"
            );
        }
    }
}
