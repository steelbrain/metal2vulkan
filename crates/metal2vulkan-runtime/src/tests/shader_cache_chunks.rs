//! The content directory's large records kept as chunks (`shader_cache/chunks.rs`): written as
//! chunks and a manifest after them, read back whole, a run two records share kept once, a
//! missing or damaged chunk, the sweep's cleanup (unnamed chunks, dead writers' temporaries, the
//! records of a retired frame), a writer's lock putting it off, a cleanup's lock making a writer
//! write whole, and the cap. Each test owns an
//! empty directory ([`Scratch`]) and a store with no cache thread, so its writes are the calling
//! thread's; the records are opaque bytes, which a decoder of the bytes themselves reads back.

use super::shader_cache::{store, Scratch, BUILD};
use crate::shader_cache::store::{content, hex, Lookup, Store, LAYOUT};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

/// The first bytes of a manifest (`chunks::MAGIC`).
const MANIFEST: &[u8; 8] = b"M2VCHNK1";
/// The least record kept as chunks (`chunks::WHOLE`).
const WHOLE: usize = 256 << 10;
/// The longest chunk (`chunks::MAX`).
const MAX: usize = 128 << 10;

/// `length` bytes of a generator seeded with `seed`.
fn noise(seed: u64, length: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// The name of record `n`.
fn name(n: u32) -> String {
    hex(&Sha256::digest(n.to_le_bytes()).into())
}

/// The path of the record named `name` under `root`.
fn record_path(root: &Path, name: &str) -> PathBuf {
    root.join(LAYOUT)
        .join(hex(&content()))
        .join(&name[..2])
        .join(name)
}

/// The chunk directory under `root`.
fn chunk_dir(root: &Path) -> PathBuf {
    root.join("chunks")
}

/// The files of the chunk directory's groups under `root`, with their lengths.
fn chunk_files(root: &Path) -> Vec<(PathBuf, u64)> {
    let mut files = Vec::new();
    let Ok(groups) = fs::read_dir(chunk_dir(root)) else {
        return files;
    };
    for group in groups.flatten() {
        if !group.path().is_dir() {
            continue;
        }
        for file in fs::read_dir(group.path()).unwrap().flatten() {
            files.push((file.path(), file.metadata().unwrap().len()));
        }
    }
    files
}

/// The bytes of the chunks under `root`.
fn chunk_bytes(root: &Path) -> u64 {
    chunk_files(root).iter().map(|(_, length)| length).sum()
}

/// The record `name` of `store`, its bytes as read.
fn get(store: &Store, name: &str) -> Option<Vec<u8>> {
    match store.load_content(name, Some) {
        Lookup::Hit(bytes) => Some(bytes),
        Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
    }
}

/// A record of `WHOLE` bytes or more is a small manifest at its path over chunks that hold its
/// bytes, and reads back whole; a smaller one stays whole.
#[test]
fn a_large_record_is_kept_as_chunks_and_read_back_whole() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    let record = noise(1, 1 << 20);
    store.write_record_soon(&name(1), record.clone()).unwrap();
    let manifest = fs::read(record_path(scratch.path(), &name(1))).unwrap();
    assert_eq!(&manifest[..8], MANIFEST);
    assert!(
        manifest.len() < 4096,
        "a manifest of {} bytes",
        manifest.len()
    );
    let files = chunk_files(scratch.path());
    assert!(files.len() >= 8, "{} chunks of 1 MiB", files.len());
    assert!(files.iter().all(|(_, length)| *length as usize <= MAX));
    assert_eq!(chunk_bytes(scratch.path()), record.len() as u64);
    assert_eq!(get(&store, &name(1)).as_deref(), Some(&record[..]));
    let small = noise(2, WHOLE - 1);
    store.write_record_soon(&name(2), small.clone()).unwrap();
    assert_eq!(
        fs::read(record_path(scratch.path(), &name(2))).unwrap(),
        small
    );
    assert_eq!(chunk_bytes(scratch.path()), record.len() as u64);
}

/// Two records that share a run of 1.5 MiB at other offsets keep it once: the second writes its
/// own bytes and the few chunks across each end of the run alone.
#[test]
fn records_that_share_a_run_keep_it_once() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    let shared = noise(3, 3 << 19);
    let a = [noise(4, 3000), shared.clone(), noise(5, 5000)].concat();
    let b = [noise(6, 7000), shared.clone(), noise(7, 1000)].concat();
    store.write_record_soon(&name(1), a.clone()).unwrap();
    let after_a = chunk_bytes(scratch.path());
    let written = store.counters.written_bytes.load(Ordering::Relaxed);
    store.write_record_soon(&name(2), b.clone()).unwrap();
    let b_wrote = store.counters.written_bytes.load(Ordering::Relaxed) - written;
    let added = chunk_bytes(scratch.path()) - after_a;
    eprintln!(
        "CHUNKS a {} bytes, b wrote {b_wrote} of {} bytes",
        a.len(),
        b.len()
    );
    assert!(added <= 3 * MAX as u64 + 8000, "{added} bytes added for b");
    assert!(b_wrote < added + 4096, "{b_wrote} bytes written for b");
    assert_eq!(get(&store, &name(1)).as_deref(), Some(&a[..]));
    assert_eq!(get(&store, &name(2)).as_deref(), Some(&b[..]));
}

/// A record whose chunks cannot be written leaves no manifest: the manifest's rename comes after
/// every chunk's.
#[test]
fn a_manifest_is_written_only_after_its_chunks() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    // A file where the chunk directory goes: no chunk can be written.
    fs::write(chunk_dir(scratch.path()), b"in the way").unwrap();
    assert!(store
        .write_record_soon(&name(1), noise(1, 1 << 20))
        .is_err());
    assert!(!record_path(scratch.path(), &name(1)).exists());
    assert_eq!(get(&store, &name(1)), None);
}

/// A manifest whose chunk is gone reads as a miss, and the record's next write brings the chunk
/// back; a chunk whose bytes are not its digest's is written again too.
#[test]
fn a_record_naming_a_chunk_not_there_is_a_miss_until_written_again() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    let record = noise(1, 1 << 20);
    store.write_record_soon(&name(1), record.clone()).unwrap();
    let files = chunk_files(scratch.path());
    fs::remove_file(&files[0].0).unwrap();
    assert_eq!(get(&store, &name(1)), None);
    assert!(record_path(scratch.path(), &name(1)).exists());
    store.write_record_soon(&name(1), record.clone()).unwrap();
    assert_eq!(get(&store, &name(1)).as_deref(), Some(&record[..]));
    // Damaged in place, at its length: the next write puts it right.
    let (path, length) = &files[1];
    fs::write(path, vec![0u8; *length as usize]).unwrap();
    store.write_record_soon(&name(1), record.clone()).unwrap();
    assert_eq!(get(&store, &name(1)).as_deref(), Some(&record[..]));
}

/// The sweep at creation, where no cleanup ran yet: the content records of a retired frame (an
/// older version's driver data, `driver::RETIRED`) go, with the chunks no manifest names and a
/// dead writer's temporary file; every record a manifest stands for still reads whole.
#[test]
fn the_sweep_removes_retired_records_and_unnamed_chunks() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    // A tree of the version before: whole records, one of them past the chunks' bound.
    let mut retired = Vec::new();
    for n in 100..104u32 {
        let length = if n == 100 { 2 << 20 } else { 4096 };
        let bytes = [&b"M2VPPLC\0"[..], &noise(u64::from(n), length)].concat();
        let path = record_path(scratch.path(), &name(n));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        retired.push(path);
    }
    let a = noise(1, 1 << 20);
    let b = noise(2, 1 << 20);
    store.write_record_soon(&name(1), a).unwrap();
    store.write_record_soon(&name(2), b.clone()).unwrap();
    // The record written again over other bytes: its first chunks are named by no manifest.
    let again = noise(3, 1 << 20);
    store.write_record_soon(&name(1), again.clone()).unwrap();
    let dead = chunk_files(scratch.path())[0]
        .0
        .with_file_name(".dead.1.2.3.tmp");
    fs::write(&dead, b"a writer that died").unwrap();
    assert_eq!(chunk_bytes(scratch.path()), (3 << 20) + 18);
    store.sweep(&AtomicBool::new(false));
    assert!(
        retired.iter().all(|path| !path.exists()),
        "a retired record stayed"
    );
    assert!(!dead.exists());
    assert_eq!(chunk_bytes(scratch.path()), 2 << 20);
    assert!(chunk_dir(scratch.path()).join("state").exists());
    assert_eq!(get(&store, &name(1)).as_deref(), Some(&again[..]));
    assert_eq!(get(&store, &name(2)).as_deref(), Some(&b[..]));
}

/// A writer that holds the chunks' lock shared, here in another open file as another process's
/// would be, puts the sweep's cleanup off: nothing is removed, and the next sweep, the lock free,
/// cleans up.
#[test]
fn a_writer_holding_the_lock_puts_the_cleanup_off() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    store
        .write_record_soon(&name(1), noise(1, 1 << 20))
        .unwrap();
    store
        .write_record_soon(&name(1), noise(2, 1 << 20))
        .unwrap();
    assert_eq!(chunk_bytes(scratch.path()), 2 << 20);
    let lock = fs::File::open(chunk_dir(scratch.path()).join("lock")).unwrap();
    lock.lock_shared().unwrap();
    store.sweep(&AtomicBool::new(false));
    assert_eq!(chunk_bytes(scratch.path()), 2 << 20);
    lock.unlock().unwrap();
    store.sweep(&AtomicBool::new(false));
    assert_eq!(chunk_bytes(scratch.path()), 1 << 20);
}

/// A writer that meets a cleanup holding the lock, here another open file's exclusive lock,
/// writes its record whole rather than wait, and the record reads back.
#[test]
fn a_writer_meeting_a_cleanup_writes_its_record_whole() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 1 << 30);
    store
        .write_record_soon(&name(1), noise(1, 1 << 20))
        .unwrap();
    let lock = fs::File::open(chunk_dir(scratch.path()).join("lock")).unwrap();
    lock.lock().unwrap();
    let record = noise(2, 1 << 20);
    store.write_record_soon(&name(2), record.clone()).unwrap();
    lock.unlock().unwrap();
    assert_eq!(
        fs::read(record_path(scratch.path(), &name(2))).unwrap(),
        record
    );
    assert_eq!(chunk_bytes(scratch.path()), 1 << 20);
    assert_eq!(get(&store, &name(2)).as_deref(), Some(&record[..]));
}

/// Set the access and modification times of the file at `path`.
fn set_times(path: &Path, when: SystemTime) {
    let file = fs::File::options().write(true).open(path).unwrap();
    file.set_times(fs::FileTimes::new().set_accessed(when).set_modified(when))
        .unwrap();
}

/// The chunks count against the cap: past it, the least recently used record's manifest goes,
/// and the chunks only it named with it; the other record stays whole.
#[test]
fn the_cap_counts_chunks_and_an_evicted_record_takes_its_own() {
    let scratch = Scratch::new();
    let store = store(scratch.path(), BUILD, 5 << 19);
    let a = noise(1, 3 << 19);
    let b = noise(2, 3 << 19);
    store.write_record_soon(&name(1), a).unwrap();
    store.write_record_soon(&name(2), b.clone()).unwrap();
    let old = SystemTime::now() - Duration::from_secs(3600);
    set_times(&record_path(scratch.path(), &name(1)), old);
    store.evict(&AtomicBool::new(false));
    assert!(!record_path(scratch.path(), &name(1)).exists());
    assert_eq!(chunk_bytes(scratch.path()), 3 << 19);
    assert_eq!(get(&store, &name(2)).as_deref(), Some(&b[..]));
    assert_eq!(store.counters.evictions.load(Ordering::Relaxed), 1);
}
