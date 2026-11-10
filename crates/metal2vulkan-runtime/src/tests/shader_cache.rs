//! The shader cache on disk: its names and layout, what a lookup accepts, writers in several
//! processes, the cap, the sweep at creation, and the directory checks. Each test owns an empty
//! directory it removes ([`Scratch`]); the entries are the driver's pipeline cache frames
//! (`driver.rs`), whose decoder checks the identity and a checksum as the artifacts' does. The
//! driver's own data, in the content directory, crosses builds but not device identities, and is
//! not loaded past its restart size.

use crate::error::{Error, Problem, ShaderCacheError};
use crate::shader_cache::driver::{
    make, open, seal, Identity, LiveCaches, Making, Records, Refusal, LIVE,
};
use crate::shader_cache::store::{entry_name, Directory, Lookup, Store, LAYOUT, UNUSED_BUILD};
use crate::shader_cache::{record_name, CacheSetup, EntryKind, ShaderCacheOptions};
use ash::vk::{self, Handle};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// A directory a test owns: made empty, removed when dropped.
pub(super) struct Scratch(PathBuf);

impl Scratch {
    pub(super) fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "m2v-shader-cache-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        Scratch(dir)
    }

    pub(super) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) const BUILD: [u8; 32] = [7; 32];
const OTHER_BUILD: [u8; 32] = [8; 32];
const NO_PROFILE: [u8; 32] = [0; 32];

/// The store of `root` for `build`, with a cap of `capacity` bytes and no cache thread, so its
/// writes are the calling thread's.
pub(super) fn store(root: &Path, build: [u8; 32], capacity: u64) -> Store {
    Store::new(
        Directory::open(root.to_path_buf(), build, capacity).expect("a usable directory"),
        None,
        None,
    )
}

/// [`store`] with the queue `asked`, which nothing drains until a test does
/// (`tests::shader_cache_queue`).
pub(super) fn store_with_queue(
    root: &Path,
    build: [u8; 32],
    capacity: u64,
    asked: std::sync::Arc<crate::shader_cache::worker::Asked>,
) -> Store {
    Store::new(
        Directory::open(root.to_path_buf(), build, capacity).expect("a usable directory"),
        None,
        Some(asked),
    )
}

/// [`BUILD`] in the hex the directory names it by.
pub(super) fn hex_build() -> String {
    crate::shader_cache::store::hex(&BUILD)
}

/// The identity of a made-up device `n`.
fn identity(n: u32) -> Identity {
    Identity::new(0x1002, n, 7, [n as u8; 16])
}

/// Pipeline cache data for `identity` (its Vulkan header, then `payload`).
fn data(identity_n: u32, payload: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&32u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0x1002u32.to_le_bytes());
    data.extend_from_slice(&identity_n.to_le_bytes());
    data.extend_from_slice(&[identity_n as u8; 16]);
    data.extend_from_slice(payload);
    data
}

/// The entry name of device `n`'s data in `build`.
fn name(build: &[u8; 32], n: u32) -> String {
    entry_name(build, EntryKind::PipelineData, &[n as u8; 32], &NO_PROFILE)
}

/// Write device `n`'s framed `payload` under `name`.
fn put(store: &Store, name: &str, n: u32, payload: &[u8]) {
    let framed = seal(&identity(n), &data(n, payload));
    store.write_soon(name, framed).expect("a write");
}

/// Look up device `n`'s entry under `name`.
fn get(store: &Store, name: &str, n: u32) -> Lookup<Vec<u8>> {
    let identity = identity(n);
    store.load(name, |bytes| {
        open(&identity, bytes)
            .map(|data| bytes[data].to_vec())
            .ok_or(())
    })
}

/// An entry's name hashes every part of its key, and it lives at `<root>/v1/<build>/<first
/// byte>/<name>`.
#[test]
fn entry_names_cover_the_whole_key_and_the_layout_is_fixed() {
    let key = [1; 32];
    let profile = [2; 32];
    let base = entry_name(&BUILD, EntryKind::Function, &key, &profile);
    assert_eq!(base.len(), 64);
    assert!(base
        .bytes()
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    assert_eq!(
        base,
        entry_name(&BUILD, EntryKind::Function, &key, &profile)
    );
    let others = [
        entry_name(&OTHER_BUILD, EntryKind::Function, &key, &profile),
        entry_name(&BUILD, EntryKind::ComputePipeline, &key, &profile),
        entry_name(&BUILD, EntryKind::Function, &[3; 32], &profile),
        entry_name(&BUILD, EntryKind::Function, &key, &[4; 32]),
    ];
    for other in &others {
        assert_ne!(&base, other);
    }
    let scratch = Scratch::new();
    let directory = Directory::open(scratch.path().to_path_buf(), BUILD, 1 << 20).unwrap();
    let build_hex: String = BUILD.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        directory.entry_path(&base),
        scratch
            .path()
            .join(LAYOUT)
            .join(build_hex)
            .join(&base[..2])
            .join(&base)
    );
}

/// What one build wrote, another build never finds.
#[test]
fn another_build_reads_nothing_of_this_one() {
    let scratch = Scratch::new();
    let old = store(scratch.path(), OTHER_BUILD, 1 << 20);
    put(&old, &name(&OTHER_BUILD, 1), 1, b"old");
    assert_eq!(
        get(&old, &name(&OTHER_BUILD, 1), 1),
        Lookup::Hit(data(1, b"old"))
    );
    let new = store(scratch.path(), BUILD, 1 << 20);
    assert_eq!(get(&new, &name(&BUILD, 1), 1), Lookup::Absent);
}

/// A creation's record (linked-use answers, reflection, a variant's keys) written under one
/// `RUNTIME_VERSION` is not found under another in the same build's directory, while a
/// translated artifact's name does not move with it: the runtime's derivation from a descriptor
/// is named by the runtime's version, which the build directory's identity does not carry.
#[test]
fn a_creation_record_of_another_runtime_version_is_not_found() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let now = metal2vulkan::cache::RUNTIME_VERSION;
    let other = now.wrapping_add(1);
    for kind in [
        EntryKind::LinkedAnswers,
        EntryKind::RenderReflection,
        EntryKind::VariantPipeline,
    ] {
        let key = [5; 32];
        let written = record_name(&BUILD, kind, &key, &NO_PROFILE, now);
        put(&store, &written, 1, b"record");
        assert_eq!(
            get(&store, &written, 1),
            Lookup::Hit(data(1, b"record")),
            "{kind:?} under its own runtime version"
        );
        let asked = record_name(&BUILD, kind, &key, &NO_PROFILE, other);
        assert_ne!(asked, written, "{kind:?}");
        assert_eq!(
            get(&store, &asked, 1),
            Lookup::Absent,
            "{kind:?} under another runtime version"
        );
        assert_ne!(
            written,
            entry_name(&BUILD, kind, &key, &NO_PROFILE),
            "{kind:?}: a record's name is not an artifact's name of the same key"
        );
    }
}

/// An entry under the right name holding another key's contents is a miss, and is removed: a
/// name collision never returns a wrong entry.
#[test]
fn an_entry_holding_another_keys_contents_is_a_miss_and_removed() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let wanted = name(&BUILD, 1);
    put(&store, &wanted, 2, b"device two's data");
    let path = store.directory().entry_path(&wanted);
    assert!(path.is_file());
    assert_eq!(get(&store, &wanted, 1), Lookup::Rejected);
    assert!(!path.exists());
    assert_eq!(store.counters.rejected.load(Ordering::Relaxed), 1);
    assert_eq!(get(&store, &wanted, 1), Lookup::Absent);
}

/// A corrupt entry (one byte flipped) and a truncated one are misses, and are removed.
#[test]
fn corrupt_and_truncated_entries_are_misses_and_removed() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let wanted = name(&BUILD, 1);
    let path = store.directory().entry_path(&wanted);
    for damage in ["flip", "truncate", "empty"] {
        put(&store, &wanted, 1, b"some pipeline data");
        let mut bytes = fs::read(&path).unwrap();
        match damage {
            "flip" => {
                let middle = bytes.len() / 2;
                bytes[middle] ^= 0x40;
            }
            "truncate" => bytes.truncate(bytes.len() - 5),
            _ => bytes.clear(),
        }
        fs::write(&path, &bytes).unwrap();
        assert_eq!(get(&store, &wanted, 1), Lookup::Rejected, "{damage}");
        assert!(!path.exists(), "{damage}");
    }
    assert_eq!(store.counters.rejected.load(Ordering::Relaxed), 3);
}

/// A hit refreshes the entry's access time: it becomes the most recently used.
#[test]
fn a_hit_refreshes_the_access_time() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let wanted = name(&BUILD, 1);
    put(&store, &wanted, 1, b"data");
    let path = store.directory().entry_path(&wanted);
    let old = SystemTime::now() - Duration::from_secs(10 * 24 * 60 * 60);
    set_times(&path, old);
    assert!(matches!(get(&store, &wanted, 1), Lookup::Hit(_)));
    let accessed = fs::metadata(&path).unwrap().accessed().unwrap();
    assert!(accessed > old + Duration::from_secs(24 * 60 * 60));
}

/// Set `path`'s access and modification times to `when`.
fn set_times(path: &Path, when: SystemTime) {
    let file = fs::File::options().write(true).open(path).unwrap();
    file.set_times(fs::FileTimes::new().set_accessed(when).set_modified(when))
        .unwrap();
}

/// Past its cap the directory loses its least recently used entries, down to 90% of the cap,
/// whichever build wrote them.
#[test]
fn the_cap_removes_the_least_recently_used_entries() {
    let scratch = Scratch::new();
    // Ten entries of the other build and this one, each about 1 KiB; the cap holds about eight.
    let payload = vec![0x5a; 1000];
    let frame_len = seal(&identity(0), &data(0, &payload)).len() as u64;
    let capacity = frame_len * 8 + frame_len / 2;
    let old = store(scratch.path(), OTHER_BUILD, capacity);
    let new = store(scratch.path(), BUILD, capacity);
    let now = SystemTime::now();
    let mut paths = Vec::new();
    for n in 0..10u32 {
        let (store, build) = if n % 2 == 0 {
            (&old, &OTHER_BUILD)
        } else {
            (&new, &BUILD)
        };
        let wanted = name(build, n);
        put(store, &wanted, n, &payload);
        let path = store.directory().entry_path(&wanted);
        // Entry n was last used n minutes after entry 0.
        set_times(
            &path,
            now - Duration::from_secs(3600) + Duration::from_secs(60 * u64::from(n)),
        );
        paths.push(path);
    }
    new.evict(&AtomicBool::new(false));
    let kept: Vec<bool> = paths.iter().map(|path| path.exists()).collect();
    // 90% of the cap holds seven entries: the three least recently used go.
    assert_eq!(
        kept,
        [false, false, false, true, true, true, true, true, true, true]
    );
    assert_eq!(new.counters.evictions.load(Ordering::Relaxed), 3);
    assert!(!new.over_capacity());
}

/// Makings' records count against the cap with the builds' entries: past it, the least recently
/// used go first, records and entries alike.
#[test]
fn the_cap_removes_the_least_recently_used_records_with_the_entries() {
    let scratch = Scratch::new();
    let payload = vec![0x6b; 1000];
    let frame_len = seal(&identity(0), &data(0, &payload)).len() as u64;
    let capacity = frame_len * 8 + frame_len / 2;
    let store = store(scratch.path(), BUILD, capacity);
    let now = SystemTime::now();
    let mut paths = Vec::new();
    for n in 0..10u32 {
        let wanted = name(&BUILD, n);
        let framed = seal(&identity(n), &data(n, &payload));
        // Even ones are makings' records, odd ones the build's entries.
        let path = if n % 2 == 0 {
            store.write_record_soon(&wanted, framed).expect("a write");
            store.directory().record_path(&wanted)
        } else {
            put(&store, &wanted, n, &payload);
            store.directory().entry_path(&wanted)
        };
        // Each was last used n minutes after the first.
        set_times(
            &path,
            now - Duration::from_secs(3600) + Duration::from_secs(60 * u64::from(n)),
        );
        paths.push(path);
    }
    store.evict(&AtomicBool::new(false));
    let kept: Vec<bool> = paths.iter().map(|path| path.exists()).collect();
    assert_eq!(
        kept,
        [false, false, false, true, true, true, true, true, true, true]
    );
    assert_eq!(store.counters.evictions.load(Ordering::Relaxed), 3);
    assert!(!store.over_capacity());
}

/// A device's store says its first removal past the cap and its first entry not written once
/// each, as warnings naming the directory and the cap; later ones are counted alone.
#[test]
fn the_first_eviction_and_the_first_failed_write_are_each_said_once() {
    use crate::{DebugMessage, MessageSeverity, MessageSink};
    use std::sync::{Arc, Mutex};
    let scratch = Scratch::new();
    let payload = vec![0x6b; 1000];
    let frame_len = seal(&identity(0), &data(0, &payload)).len() as u64;
    let capacity = frame_len * 4 + frame_len / 2;
    let said: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let heard = Arc::clone(&said);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        assert_eq!(message.severity, MessageSeverity::Warning);
        heard
            .lock()
            .unwrap()
            .push((message.id_name.clone(), message.text.clone()));
    });
    let directory =
        Directory::open(scratch.path().to_path_buf(), BUILD, capacity).expect("a usable directory");
    let store = Store::new(directory, Some(sink), None);
    for round in 0..2u32 {
        for n in 0..6u32 {
            let wanted = name(&BUILD, 10 * round + n);
            store
                .write_record_soon(&wanted, seal(&identity(n), &data(n, &payload)))
                .expect("a write");
        }
        store.evict(&AtomicBool::new(false));
        store.write_failed();
    }
    assert!(store.counters.evictions.load(Ordering::Relaxed) > 2);
    assert_eq!(store.counters.write_failures.load(Ordering::Relaxed), 2);
    let said = said.lock().unwrap();
    let ids: Vec<&str> = said.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "metal2vulkan-shader-cache-evicted",
            crate::Degradation::ShaderCacheEntryNotWritten.code()
        ]
    );
    let root = scratch.path().display().to_string();
    for (_, text) in said.iter() {
        assert!(text.contains(&root), "names the directory: {text}");
        assert!(
            text.contains(&capacity.to_string()),
            "names the cap: {text}"
        );
    }
}

/// The sweep at device creation removes another build's directory nothing used for 30 days,
/// keeps a recently used one, never removes its own build's, and removes stale temporary files.
#[test]
fn the_sweep_removes_other_builds_unused_for_thirty_days() {
    let scratch = Scratch::new();
    let builds = [[1u8; 32], [2u8; 32]];
    let long_ago = SystemTime::now() - UNUSED_BUILD - Duration::from_secs(24 * 60 * 60);
    let recently = SystemTime::now() - Duration::from_secs(24 * 60 * 60);
    let mut build_dirs = Vec::new();
    for (build, when) in builds.iter().zip([long_ago, recently]) {
        let store = store(scratch.path(), *build, 1 << 20);
        let wanted = name(build, 1);
        put(&store, &wanted, 1, b"data");
        let path = store.directory().entry_path(&wanted);
        set_times(&path, when);
        set_dir_times(path.parent().unwrap(), when);
        set_dir_times(path.parent().unwrap().parent().unwrap(), when);
        build_dirs.push(path.parent().unwrap().parent().unwrap().to_path_buf());
    }
    // This build's own entry, unused as long: kept.
    let own = store(scratch.path(), BUILD, 1 << 20);
    let wanted = name(&BUILD, 1);
    put(&own, &wanted, 1, b"data");
    let own_path = own.directory().entry_path(&wanted);
    set_times(&own_path, long_ago);
    // A crashed writer's temporary file, an hour and more old: removed.
    let temporary = own_path.with_file_name(format!(".{wanted}.1.2.3.tmp"));
    fs::write(&temporary, b"partial").unwrap();
    set_times(&temporary, recently);
    own.sweep(&AtomicBool::new(false));
    assert!(!build_dirs[0].exists(), "the unused build stays");
    assert!(build_dirs[1].exists(), "the recently used build went");
    assert!(own_path.exists(), "this build's entry went");
    assert!(!temporary.exists(), "the stale temporary file stays");
}

/// Set the directory `path`'s times to `when`.
fn set_dir_times(path: &Path, when: SystemTime) {
    #[cfg(unix)]
    let dir = fs::File::open(path).unwrap();
    #[cfg(windows)]
    let dir = {
        use std::os::windows::fs::OpenOptionsExt;
        // Windows directory handles require backup semantics; changing their
        // times requires FILE_WRITE_ATTRIBUTES rather than ordinary file reads.
        fs::File::options()
            .access_mode(0x0100)
            .custom_flags(0x0200_0000)
            .open(path)
            .expect("directory handle with write-attributes access")
    };
    // These times drive the unused-build sweep; a failed update must fail the test.
    dir.set_times(fs::FileTimes::new().set_accessed(when).set_modified(when))
        .expect("set the test directory's timestamps");
}

/// A directory that cannot be made is a typed error naming it; so is a zero cap, on the
/// directory or on the memo.
#[test]
fn an_unusable_directory_is_a_typed_error() {
    let scratch = Scratch::new();
    fs::create_dir_all(scratch.path()).unwrap();
    let file = scratch.path().join("a-file");
    fs::write(&file, b"not a directory").unwrap();
    let below = file.join("cache");
    let error = CacheSetup::of(
        &ShaderCacheOptions::in_directory(&below),
        "system_default_with",
    )
    .unwrap_err();
    let Error::ShaderCache(ShaderCacheError { path, message, .. }) = &error else {
        panic!("not a shader cache error: {error:?}");
    };
    assert_eq!(path.as_deref(), Some(below.as_path()));
    assert!(!message.is_empty());
    assert!(error.to_string().contains(&below.display().to_string()));
    let zero_caps = |options: &mut ShaderCacheOptions| options.capacity = 0;
    let zero_memo = |options: &mut ShaderCacheOptions| options.memory_capacity = 0;
    for zero in [&zero_caps as &dyn Fn(&mut ShaderCacheOptions), &zero_memo] {
        let mut options = ShaderCacheOptions::in_directory(scratch.path().join("cache"));
        zero(&mut options);
        match CacheSetup::of(&options, "all_with") {
            Err(Error::Invalid(invalid)) => {
                assert_eq!(invalid.call, "all_with");
                assert_eq!(invalid.problem, Problem::ShaderCacheCapacity);
            }
            other => panic!("a zero cap was not refused: {other:?}"),
        }
    }
}

/// The directory and its build directory are made only their owner may enter.
#[cfg(unix)]
#[test]
fn the_directory_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = Scratch::new();
    let store = store(&scratch.path().join("made"), BUILD, 1 << 20);
    let wanted = name(&BUILD, 1);
    put(&store, &wanted, 1, b"data");
    let entry = store.directory().entry_path(&wanted);
    let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&scratch.path().join("made")), 0o700);
    assert_eq!(mode(entry.parent().unwrap()), 0o700);
    assert_eq!(mode(&entry), 0o600);
}

/// The environment variables that make [`writer_process`] a writer.
const WRITER_DIR: &str = "M2V_TEST_SHADER_CACHE_DIR";
const WRITER_ID: &str = "M2V_TEST_SHADER_CACHE_WRITER";
/// Entries every writer writes, and rounds of writing them.
const SHARED: u32 = 16;
const ROUNDS: u32 = 20;

/// Several processes write the same entries at once, each its own too, while this one reads:
/// a reader finds a whole entry or none, and afterwards every entry is whole and no temporary
/// file is left.
#[test]
fn writers_in_several_processes_leave_whole_entries() {
    let scratch = Scratch::new();
    let reader = store(scratch.path(), BUILD, 1 << 30);
    let children: Vec<std::process::Child> = (0..4)
        .map(|id| {
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tests::shader_cache::writer_process",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env(WRITER_DIR, scratch.path())
                .env(WRITER_ID, id.to_string())
                .spawn()
                .unwrap()
        })
        .collect();
    // Read while the writers write, until they all exited.
    let mut children: Vec<_> = children.into_iter().map(Some).collect();
    let started = std::time::Instant::now();
    let mut hits = 0u32;
    while children.iter().any(Option::is_some) {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "the writers did not finish"
        );
        for n in 0..SHARED {
            match get(&reader, &name(&BUILD, n), n) {
                Lookup::Hit(found) => {
                    assert_eq!(found, data(n, &shared_payload(n)));
                    hits += 1;
                }
                Lookup::Absent => {}
                other => panic!("a reader found a partial entry: {other:?}"),
            }
        }
        for slot in &mut children {
            if let Some(child) = slot {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "a writer failed");
                    *slot = None;
                }
            }
        }
    }
    assert!(hits > 0, "the reader never read while the writers wrote");
    for n in 0..SHARED {
        assert_eq!(
            get(&reader, &name(&BUILD, n), n),
            Lookup::Hit(data(n, &shared_payload(n)))
        );
    }
    for id in 0..4u32 {
        let n = 1000 + id;
        assert_eq!(
            get(&reader, &name(&BUILD, n), n),
            Lookup::Hit(data(n, b"own"))
        );
    }
    let build_dir = reader.directory().entry_path(&name(&BUILD, 0));
    let build_dir = build_dir.parent().unwrap().parent().unwrap();
    for group in fs::read_dir(build_dir).unwrap() {
        for file in fs::read_dir(group.unwrap().path()).unwrap() {
            let file = file.unwrap().file_name();
            assert!(
                !file.to_string_lossy().starts_with('.'),
                "left behind: {file:?}"
            );
        }
    }
}

/// The payload every writer writes for shared entry `n`: large enough that a write takes
/// several system calls.
fn shared_payload(n: u32) -> Vec<u8> {
    (0..64 * 1024u32).map(|i| (i ^ n) as u8).collect()
}

/// One writer of [`writers_in_several_processes_leave_whole_entries`], run as its own process.
#[test]
#[ignore = "a child process of writers_in_several_processes_leave_whole_entries"]
fn writer_process() {
    let dir = PathBuf::from(std::env::var_os(WRITER_DIR).expect("the parent names the directory"));
    let id: u32 = std::env::var(WRITER_ID).unwrap().parse().unwrap();
    let store = store(&dir, BUILD, 1 << 30);
    for _ in 0..ROUNDS {
        for n in 0..SHARED {
            put(&store, &name(&BUILD, n), n, &shared_payload(n));
        }
    }
    put(&store, &name(&BUILD, 1000 + id), 1000 + id, b"own");
}

/// The driver's frames: the identity, the checksum, and the data's own Vulkan header must all
/// hold.
#[test]
fn driver_frames_check_identity_checksum_and_vulkan_header() {
    let one = identity(1);
    let framed = seal(&one, &data(1, b"payload"));
    let found = open(&one, &framed).map(|at| framed[at].to_vec());
    assert_eq!(found, Some(data(1, b"payload")));
    assert_eq!(open(&identity(2), &framed), None, "another device's frame");
    let mut flipped = framed.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    assert_eq!(open(&one, &flipped), None, "a broken checksum");
    assert_eq!(
        open(&one, &framed[..framed.len() - 1]),
        None,
        "a truncated frame"
    );
    // Data whose own header names another device, framed for this one.
    let foreign = seal(&one, &data(2, b"payload"));
    assert_eq!(open(&one, &foreign), None, "a foreign Vulkan header");
    let short = seal(&one, &data(1, b"")[..31]);
    assert_eq!(open(&one, &short), None, "a header shorter than Vulkan's");
}

/// A making's record is named by its description, the driver's identity (a driver update is
/// another), the instance's layer settings and the device's profile, and by nothing of the
/// build's: it lives in the content directory, where another build's store finds it.
#[test]
fn a_making_s_record_is_named_by_what_it_reads_and_crosses_builds() {
    let records = |identity: Identity, settings: u8, profile: u8| Records {
        identity,
        settings: [settings; 32],
        profile: [profile; 32],
    };
    // The record's name of what a making reads, which is also what the device holds that
    // record's live `VkPipelineCache` under.
    let named = |records: Records, making: Making| records.name(&records.key(making));
    let making = || {
        Making::new("graphics")
            .words(&[1, 2, 3])
            .text(&(4u32, "five"))
    };
    let base = named(records(identity(1), 1, 1), making());
    assert_eq!(
        base,
        named(records(identity(1), 1, 1), making()),
        "a stable name"
    );
    for (other, why) in [
        (
            named(
                records(identity(1), 1, 1),
                Making::new("compute").words(&[1, 2, 3]),
            ),
            "another kind",
        ),
        (
            named(
                records(identity(1), 1, 1),
                Making::new("graphics").words(&[1, 2, 4]),
            ),
            "other words",
        ),
        (
            named(
                records(identity(1), 1, 1),
                Making::new("graphics")
                    .words(&[1, 2, 3])
                    .text(&(4u32, "six")),
            ),
            "another state",
        ),
        (
            named(
                records(Identity::new(0x1002, 1, 8, [1; 16]), 1, 1),
                making(),
            ),
            "another driver version",
        ),
        (
            named(records(identity(1), 2, 1), making()),
            "other layer settings",
        ),
        (
            named(records(identity(1), 1, 2), making()),
            "another profile",
        ),
    ] {
        assert_ne!(other, base, "{why}");
    }
    let scratch = Scratch::new();
    let one = identity(1);
    store(scratch.path(), BUILD, 1 << 20)
        .write_record_soon(&base, seal(&one, &data(1, b"payload")))
        .expect("a write");
    let other = store(scratch.path(), OTHER_BUILD, 1 << 20);
    let found = other.load_content(&base, |bytes| {
        open(&one, &bytes).map(|at| bytes[at].to_vec())
    });
    assert_eq!(found, Lookup::Hit(data(1, b"payload")));
}

/// A description's parts are each framed by their length: parts that join to the same bytes
/// otherwise split still name other records.
#[test]
fn a_making_s_parts_do_not_run_together() {
    let digest = |making: Making| making.finish();
    assert_ne!(
        digest(Making::new("a").text(&"bc")),
        digest(Making::new("ab").text(&"c"))
    );
    assert_ne!(
        digest(Making::new("k").words(&[1]).words(&[2, 3])),
        digest(Making::new("k").words(&[1, 2]).words(&[3]))
    );
}

/// A record framed for another device, or whose data's Vulkan header names another, is refused
/// and removed, counted: that making starts empty.
#[test]
fn a_record_of_another_device_is_refused_and_removed() {
    let scratch = Scratch::new();
    let one = identity(1);
    let store = store(scratch.path(), BUILD, 1 << 20);
    let name = name(&BUILD, 1);
    let path = store.directory().record_path(&name);
    for framed in [
        seal(&identity(2), &data(2, b"payload")),
        seal(&one, &data(2, b"payload")),
    ] {
        store.write_record_soon(&name, framed).expect("a write");
        let found = store.load_content(&name, |bytes| open(&one, &bytes).map(|_| ()));
        assert_eq!(found, Lookup::Rejected);
        assert!(!path.exists(), "the refused record was removed");
    }
    assert_eq!(store.counters.rejected.load(Ordering::Relaxed), 2);
}

/// The driver's own data is checked as a load checks it, and a refusal says why: data that ends
/// inside the header it states (a driver was seen returning 20 bytes of a 32-byte header at
/// `VK_SUCCESS`) is truncated; a header too small, of another version, or naming another device
/// is mismatched.
#[test]
fn driver_data_refusals_name_their_reason() {
    let one = identity(1);
    let whole = data(1, b"payload");
    assert_eq!(one.check_vulkan_header(&whole), Ok(()));
    assert_eq!(one.check_vulkan_header(&whole[..32]), Ok(()));
    assert_eq!(
        one.check_vulkan_header(&whole[..20]),
        Err(Refusal::Truncated)
    );
    assert_eq!(
        one.check_vulkan_header(&whole[..31]),
        Err(Refusal::Truncated)
    );
    assert_eq!(
        one.check_vulkan_header(&whole[..3]),
        Err(Refusal::Truncated)
    );
    assert_eq!(one.check_vulkan_header(&[]), Err(Refusal::Truncated));
    let mut longer = whole.clone();
    longer[..4].copy_from_slice(&64u32.to_le_bytes());
    assert_eq!(one.check_vulkan_header(&longer), Err(Refusal::Truncated));
    let mut small = whole.clone();
    small[..4].copy_from_slice(&16u32.to_le_bytes());
    assert_eq!(one.check_vulkan_header(&small), Err(Refusal::Mismatched));
    let mut version = whole.clone();
    version[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(one.check_vulkan_header(&version), Err(Refusal::Mismatched));
    assert_eq!(
        one.check_vulkan_header(&data(2, b"payload")),
        Err(Refusal::Mismatched)
    );
}

/// What a making's creation feedback says of its pipeline (`shader_cache::found_in_cache`): found
/// where the whole pipeline's feedback is valid and says `APPLICATION_PIPELINE_CACHE_HIT`, whatever
/// its stages' say (the form RADV, ANV and NVIDIA give); found where the whole pipeline's is valid
/// and every stage's is valid and says it (MoltenVK 1.4.1's form, its whole pipeline's only ever
/// valid); compiled where the whole pipeline's is valid otherwise (no stages, one stage not a
/// valid hit); unreported where the whole pipeline's is not valid.
#[test]
fn a_pipeline_is_found_in_its_cache_by_its_whole_feedback_or_every_stage_s() {
    use crate::shader_cache::found_in_cache;
    use ash::vk::{PipelineCreationFeedback, PipelineCreationFeedbackFlags as Flags};
    let feedback = |flags: Flags| PipelineCreationFeedback { flags, duration: 1 };
    let valid = feedback(Flags::VALID);
    let hit = feedback(Flags::VALID | Flags::APPLICATION_PIPELINE_CACHE_HIT);
    let unset = feedback(Flags::empty());
    // A hit that is not valid: the driver reported nothing.
    let unvalidated = feedback(Flags::APPLICATION_PIPELINE_CACHE_HIT);
    // The whole pipeline's hit.
    assert_eq!(found_in_cache(&hit, &[]), Some(true));
    assert_eq!(found_in_cache(&hit, &[valid, valid]), Some(true));
    assert_eq!(found_in_cache(&hit, &[unset, unset, unset]), Some(true));
    assert_eq!(found_in_cache(&hit, &[hit, valid]), Some(true));
    // The whole pipeline's valid, every stage's a valid hit.
    assert_eq!(found_in_cache(&valid, &[hit]), Some(true));
    assert_eq!(found_in_cache(&valid, &[hit, hit]), Some(true));
    assert_eq!(found_in_cache(&valid, &[hit; 5]), Some(true));
    // Mixed stages, or a stage's hit that is not valid: compiled.
    assert_eq!(found_in_cache(&valid, &[hit, valid]), Some(false));
    assert_eq!(found_in_cache(&valid, &[valid, hit]), Some(false));
    assert_eq!(found_in_cache(&valid, &[hit, unset]), Some(false));
    assert_eq!(found_in_cache(&valid, &[hit, unvalidated]), Some(false));
    assert_eq!(found_in_cache(&valid, &[valid]), Some(false));
    // No stage's feedback chained: the whole pipeline's alone.
    assert_eq!(found_in_cache(&valid, &[]), Some(false));
    // The whole pipeline's not valid: unreported, whatever the stages say.
    assert_eq!(found_in_cache(&unset, &[hit, hit]), None);
    assert_eq!(found_in_cache(&unvalidated, &[hit]), None);
    assert_eq!(found_in_cache(&unset, &[]), None);
}

thread_local! {
    /// The fake driver's entry points called on this thread ([`fake_driver`]), in order.
    static CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// The fake driver's `vkCreatePipelineCache`: a cache of a handle no earlier call gave, so a
/// test reads from the handle which cache a making was handed.
unsafe extern "system" fn create_cache(
    _: vk::Device,
    _: *const vk::PipelineCacheCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    cache: *mut vk::PipelineCache,
) -> vk::Result {
    CALLS.with(|calls| calls.borrow_mut().push("vkCreatePipelineCache"));
    let handle = CACHES.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the caller's out pointer, valid for one handle.
    unsafe { cache.write(vk::PipelineCache::from_raw(handle)) };
    vk::Result::SUCCESS
}

/// The handle the fake driver's next `vkCreatePipelineCache` gives: never zero, which is the
/// null handle.
static CACHES: AtomicU64 = AtomicU64::new(1);

/// The fake driver's `vkGetPipelineCacheData`: device 1's data.
unsafe extern "system" fn cache_data(
    _: vk::Device,
    _: vk::PipelineCache,
    size: *mut usize,
    out: *mut c_void,
) -> vk::Result {
    CALLS.with(|calls| calls.borrow_mut().push("vkGetPipelineCacheData"));
    let bytes = data(1, b"made");
    // SAFETY: the caller's size, and its buffer of that many bytes where it hands one.
    unsafe {
        if !out.is_null() {
            let length = bytes.len().min(*size);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), length);
        }
        *size = bytes.len();
    }
    vk::Result::SUCCESS
}

/// The fake driver's `vkDestroyPipelineCache`.
unsafe extern "system" fn destroy_cache(
    _: vk::Device,
    _: vk::PipelineCache,
    _: *const vk::AllocationCallbacks<'_>,
) {
    CALLS.with(|calls| calls.borrow_mut().push("vkDestroyPipelineCache"));
}

/// A device whose driver has only a pipeline cache's entry points, each noting its call on the
/// calling thread ([`CALLS`]).
fn fake_driver() -> ash::Device {
    // SAFETY: a table of the fakes above and ash's stubs; no other entry point is called.
    unsafe {
        ash::Device::load_with(
            |name| match name.to_bytes() {
                b"vkCreatePipelineCache" => {
                    create_cache as vk::PFN_vkCreatePipelineCache as *const c_void
                }
                b"vkGetPipelineCacheData" => {
                    cache_data as vk::PFN_vkGetPipelineCacheData as *const c_void
                }
                b"vkDestroyPipelineCache" => {
                    destroy_cache as vk::PFN_vkDestroyPipelineCache as *const c_void
                }
                _ => std::ptr::null(),
            },
            vk::Device::from_raw(1),
        )
    }
}

/// One making on the fake driver with a store of `store` and the device's caches `live`, the
/// process exiting `before` it began and `after` its `vkCreate*Pipelines` returned: the driver's
/// entry points it called (a run of calls of one, as `ash` reads data in two, counted once) and
/// the cache that call was handed. The driver reports nothing of the making.
fn made(
    store: &Store,
    live: &LiveCaches,
    before: bool,
    after: bool,
) -> (Vec<&'static str>, vk::PipelineCache) {
    reported(store, live, (before, after), None)
}

/// [`made`], the driver reporting `found` of the making (`shader_cache::found_in_cache`'s
/// answer).
fn reported(
    store: &Store,
    live: &LiveCaches,
    exiting: (bool, bool),
    found: Option<bool>,
) -> (Vec<&'static str>, vk::PipelineCache) {
    described(
        store,
        live,
        exiting,
        found,
        Making::new("compute").words(&[1, 2, 3]),
    )
}

/// [`reported`], of the making `making` describes.
fn described(
    store: &Store,
    live: &LiveCaches,
    (before, after): (bool, bool),
    found: Option<bool>,
    making: Making,
) -> (Vec<&'static str>, vk::PipelineCache) {
    CALLS.with(|calls| calls.borrow_mut().clear());
    let records = Records {
        identity: identity(1),
        settings: [0; 32],
        profile: NO_PROFILE,
    };
    let created = Cell::new(false);
    let mut handed = None;
    let result: Result<(), ()> = make(
        store,
        &records,
        live,
        &fake_driver(),
        making,
        || match created.get() {
            false => before,
            true => after,
        },
        |cache| {
            created.set(true);
            handed = Some(cache);
            (Ok(()), found)
        },
    );
    assert_eq!(result, Ok(()));
    let mut calls = CALLS.with(|calls| calls.take());
    calls.dedup();
    (calls, handed.expect("the making ran"))
}

/// The device's caches destroyed on the fake driver, as its teardown destroys them: the entry
/// points that took.
fn torn_down(live: &LiveCaches) -> Vec<&'static str> {
    CALLS.with(|calls| calls.borrow_mut().clear());
    // SAFETY: the fake device is live and no making is left.
    unsafe { live.destroy(&fake_driver()) };
    let mut calls = CALLS.with(|calls| calls.take());
    calls.dedup();
    calls
}

/// Once the process is exiting, a making calls the driver for its pipelines alone: one that ends
/// after the close (a pool piece's section runs on past it, `exit_guard`) does not read its
/// record's cache, and one that begins after makes no cache. Before, it makes the record's cache,
/// reads it and saves the record; the cache goes at the device's teardown, not with the making.
#[test]
fn a_making_once_the_process_is_exiting_calls_the_driver_for_its_pipelines_alone() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let saves = || store.counters.driver_saves.load(Ordering::Relaxed);
    // A device of its own for each making, as a later process is: each loads the record.
    let live = LiveCaches::default();
    let (calls, cache) = made(&store, &live, false, false);
    assert_eq!(calls, ["vkCreatePipelineCache", "vkGetPipelineCacheData"]);
    assert_ne!(cache, vk::PipelineCache::null());
    assert_eq!(saves(), 1);
    assert_eq!(torn_down(&live), ["vkDestroyPipelineCache"]);
    // The same making again, its record loaded, closed while the driver makes it.
    let live = LiveCaches::default();
    let (calls, cache) = made(&store, &live, false, true);
    assert_eq!(
        calls,
        ["vkCreatePipelineCache"],
        "a making that ended after the close read its cache"
    );
    assert_ne!(cache, vk::PipelineCache::null());
    assert_eq!(torn_down(&live), ["vkDestroyPipelineCache"]);
    let live = LiveCaches::default();
    let (calls, cache) = made(&store, &live, true, true);
    assert!(
        calls.is_empty(),
        "a making begun after the close called {calls:?}"
    );
    assert_eq!(cache, vk::PipelineCache::null());
    assert_eq!(saves(), 1, "a making saved once the process was exiting");
    assert!(
        torn_down(&live).is_empty(),
        "a making begun after the close left a cache behind"
    );
}

/// Every making of one record's name is handed the one `VkPipelineCache` the first of them made
/// (`driver_caches_shared`): no later making makes or loads one, and the record is written once,
/// since what each later making reads back is what the first wrote (`driver_data_unchanged`).
/// Another name makes its own.
#[test]
fn every_making_of_a_record_s_name_takes_the_first_s_cache() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let counter = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    let counts = || {
        let c = &store.counters;
        (
            counter(&c.driver_caches_shared),
            counter(&c.driver_saves),
            counter(&c.driver_data_unchanged),
        )
    };
    let live = LiveCaches::default();
    let (calls, first) = made(&store, &live, false, false);
    assert_eq!(calls, ["vkCreatePipelineCache", "vkGetPipelineCacheData"]);
    assert_eq!(counts(), (0, 1, 0));
    for _ in 0..3 {
        let (calls, again) = made(&store, &live, false, false);
        assert_eq!(
            calls,
            ["vkGetPipelineCacheData"],
            "a later making of the name made or loaded a cache"
        );
        assert_eq!(again, first, "a later making was handed another cache");
    }
    assert_eq!(counts(), (3, 1, 3));
    // Another name: its own cache, and the first's is still the first's.
    let (calls, other) = described(
        &store,
        &live,
        (false, false),
        None,
        Making::new("compute").words(&[4, 5, 6]),
    );
    assert_eq!(calls, ["vkCreatePipelineCache", "vkGetPipelineCacheData"]);
    assert_ne!(other, first);
    assert_eq!(counts(), (3, 2, 3));
    assert_eq!(torn_down(&live), ["vkDestroyPipelineCache"]);
}

/// A device keeps at most [`LIVE`] caches: the making of a further name is handed one of its own,
/// destroyed once it returned, so nothing of its name is kept and the teardown destroys the
/// bound's worth alone.
#[test]
fn a_device_keeps_at_most_the_bound_of_live_caches() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let live = LiveCaches::default();
    for n in 0..LIVE as u32 {
        let (calls, _) = described(
            &store,
            &live,
            (false, false),
            None,
            Making::new("compute").text(&n),
        );
        assert_eq!(calls, ["vkCreatePipelineCache", "vkGetPipelineCacheData"]);
    }
    let (calls, _) = described(
        &store,
        &live,
        (false, false),
        None,
        Making::new("compute").text(&"past the bound"),
    );
    assert_eq!(
        calls,
        [
            "vkCreatePipelineCache",
            "vkGetPipelineCacheData",
            "vkDestroyPipelineCache"
        ],
        "a making past the bound kept its cache"
    );
    // The same name again: the device kept nothing of it, so it makes and destroys one again.
    let (calls, _) = described(
        &store,
        &live,
        (false, false),
        None,
        Making::new("compute").text(&"past the bound"),
    );
    assert_eq!(
        calls,
        [
            "vkCreatePipelineCache",
            "vkGetPipelineCacheData",
            "vkDestroyPipelineCache"
        ]
    );
    assert_eq!(
        store.counters.driver_caches_shared.load(Ordering::Relaxed),
        0,
        "no making was handed another's cache"
    );
    assert_eq!(torn_down(&live), ["vkDestroyPipelineCache"]);
}

/// A making handed a cache loaded from its record that the driver reports found reads none of its
/// data back and writes nothing (`driver_data_found`); one whose cache was made empty, or one the
/// driver reports compiled, or one of a driver that reports nothing, reads its data and keeps it
/// as before: a save where the record held nothing of it, the bytes counted unchanged where it
/// did. Each making here is a device of its own, as a later process is, so each loads the record
/// rather than taking a cache another making left live.
#[test]
fn a_making_its_driver_found_in_its_record_reads_no_data_back() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 20);
    let counter = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    let counts = || {
        let c = &store.counters;
        (
            counter(&c.driver_loads),
            counter(&c.driver_saves),
            counter(&c.driver_data_unchanged),
            counter(&c.driver_data_found),
        )
    };
    let read_back = ["vkCreatePipelineCache", "vkGetPipelineCacheData"];
    // No record: the driver's report of a hit changes nothing (an empty cache holds nothing to
    // find), so the data is read and saved.
    let (calls, _) = reported(&store, &LiveCaches::default(), (false, false), Some(true));
    assert_eq!(calls, read_back);
    assert_eq!(counts(), (0, 1, 0, 0));
    // From the record, found: no data read back, nothing written.
    let (calls, _) = reported(&store, &LiveCaches::default(), (false, false), Some(true));
    assert_eq!(calls, ["vkCreatePipelineCache"]);
    assert_eq!(counts(), (1, 1, 0, 1));
    // From the record, compiled: read back, the record's bytes, unchanged.
    let (calls, _) = reported(&store, &LiveCaches::default(), (false, false), Some(false));
    assert_eq!(calls, read_back);
    assert_eq!(counts(), (2, 1, 1, 1));
    // From the record, unreported: read back and compared, as before the reports.
    let (calls, _) = made(&store, &LiveCaches::default(), false, false);
    assert_eq!(calls, read_back);
    assert_eq!(counts(), (3, 1, 2, 1));
}
