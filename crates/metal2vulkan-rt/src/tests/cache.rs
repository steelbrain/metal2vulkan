//! The cache directory on the host: two generations under one cap, a hit in old moved into young,
//! a lookup that reads no state, the state file read at open with no listing and rebuilt where it
//! is torn or missing, a removal cut short (a process's exit in it included) finished by the next
//! open, an earlier version's pipeline data removed at open, torn files removed, a blob that does
//! not check refused, and stores in threads and processes sharing one directory without losing a
//! count or reading a file another was writing.

use crate::cache::{checksum, frame, unframe, State, Store};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A directory of its own under the OS temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("m2v-rt-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Scratch(dir)
    }

    /// The bytes of every file under the directory but the state file.
    fn bytes(&self) -> u64 {
        bytes_under(&self.0) - fs::metadata(self.0.join("state")).map_or(0, |meta| meta.len())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The bytes of every file under `dir`; none where it is missing. A file or directory the store's
/// removal thread takes while the walk lists it counts nothing: what the walk finds is never more
/// than the directory held when it started.
fn bytes_under(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .map(|entry| {
            let meta = entry.and_then(|entry| Ok((entry.metadata()?, entry.path())));
            match meta {
                Ok((meta, path)) if meta.is_dir() => bytes_under(&path),
                Ok((meta, _)) => meta.len(),
                Err(error) if error.kind() == ErrorKind::NotFound => 0,
                Err(error) => panic!("{}: {error}", dir.display()),
            }
        })
        .sum()
}

/// A file the store keeps: a lowering.
fn name(n: u32) -> String {
    format!("translations/{n:064x}")
}

/// `data` as its blob holds it, after its magic, length and checksum.
fn framed(data: &[u8]) -> Vec<u8> {
    [&frame(data)[..], data].concat()
}

/// Every byte as it is.
fn whole(bytes: &[u8]) -> Option<Vec<u8>> {
    Some(bytes.to_vec())
}

fn state(store: &Store) -> State {
    store.current().expect("the state")
}

#[test]
fn a_store_filled_past_its_cap_stays_within_it_and_moves_on() {
    let scratch = Scratch::new("store-cap");
    let store = Store::open(scratch.0.clone(), 1000);
    assert_eq!(state(&store), State { young: 1, bytes: 0 });
    for n in 0..40 {
        assert!(store.put(&name(n), &[n as u8; 100]), "file {n} written");
        // While a removal runs, the dead generation is on disk beside old and young.
        assert!(
            scratch.bytes() <= 1500,
            "{} bytes after {n}",
            scratch.bytes()
        );
        store.settle();
        let now = state(&store);
        assert!(now.bytes <= 500, "{now:?} after {n}");
        assert_eq!(
            bytes_under(&store.generation(now.young)),
            now.bytes,
            "young after {n}"
        );
        assert!(
            scratch.bytes() <= 1000,
            "{} bytes after {n}",
            scratch.bytes()
        );
    }
    // Five files of 100 bytes a generation: the fortieth file is in the eighth.
    assert_eq!(state(&store).young, 8);
    // The last ten files are kept, in old and young, and none before them. (A lookup would move
    // a file of old into young, and young is full.)
    let kept: Vec<bool> = (0..40)
        .map(|n| {
            [7, 8]
                .iter()
                .any(|&g| store.generation(g).join(name(n)).exists())
        })
        .collect();
    assert_eq!(kept, (0..40).map(|n| n >= 30).collect::<Vec<_>>());
    // A file of more than half the cap is not kept.
    assert!(!store.put(&name(40), &[0; 501]));
    assert_eq!(store.get(&name(40), whole), None);
}

#[test]
fn a_hit_in_old_moves_into_young_and_outlives_the_next_generation() {
    let scratch = Scratch::new("store-old");
    let store = Store::open(scratch.0.clone(), 1000);
    for n in 0..6 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    // The sixth write made generation 2 young: files 0 to 4 are in old.
    let young = state(&store).young;
    assert_eq!(young, 2);
    let (old, new) = (
        store.generation(1).join(name(0)),
        store.generation(2).join(name(0)),
    );
    assert!(old.exists() && !new.exists());
    assert_eq!(store.get(&name(0), whole), Some(vec![0; 100]));
    assert!(!old.exists() && new.exists(), "moved into young");
    assert_eq!(state(&store).bytes, 200, "the move counted");
    // Four more writes make generation 3 young; file 0 is in old, file 1 is gone.
    for n in 6..10 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    store.settle();
    assert_eq!(state(&store).young, 3);
    assert_eq!(store.get(&name(1), whole), None);
    assert_eq!(store.get(&name(0), whole), Some(vec![0; 100]));
}

#[test]
fn a_lookup_reads_no_state_and_sees_another_stores_move_at_its_next_write() {
    let scratch = Scratch::new("store-young");
    let first = Store::open(scratch.0.clone(), 1000);
    let second = Store::open(scratch.0.clone(), 1000);
    assert!(first.put(&name(0), &[0; 100]));
    let reads = first.state_reads.load(Relaxed);
    for _ in 0..10 {
        assert_eq!(first.get(&name(0), whole), Some(vec![0; 100]));
        assert_eq!(first.get(&name(1), whole), None);
    }
    assert_eq!(
        first.state_reads.load(Relaxed),
        reads,
        "no lookup read the state"
    );
    // The second store's fifth write makes generation 2 young. The first still knows generation
    // 1, so the file the second wrote there is a miss for it.
    for n in 1..6 {
        assert!(second.put(&name(2 * n), &[n as u8; 100]));
    }
    assert_eq!(state(&second).young, 2);
    assert_eq!(
        first.get(&name(10), whole),
        None,
        "a move the first has not seen"
    );
    assert_eq!(
        first.get(&name(0), whole),
        Some(vec![0; 100]),
        "what it knew"
    );
    // The first's next write reads the state under its lock, and its lookups follow.
    assert!(first.put(&name(12), &[6; 100]));
    assert_eq!(first.get(&name(10), whole), Some(vec![5; 100]));
}

#[test]
fn a_reopened_store_reads_its_state_and_lists_no_directory() {
    let scratch = Scratch::new("store-reopen");
    let store = Store::open(scratch.0.clone(), 1000);
    assert_eq!(
        store.listings.load(Relaxed),
        1,
        "a new store lists its directory once"
    );
    for n in 0..23 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    store.settle();
    let before = state(&store);
    drop(store);
    let reopened = Store::open(scratch.0.clone(), 1000);
    assert_eq!(state(&reopened), before);
    // Files 20 to 22 in young, file 15 in old.
    for n in [20, 21, 22, 15] {
        assert_eq!(
            reopened.get(&name(n), whole),
            Some(vec![n as u8; 100]),
            "file {n}"
        );
    }
    assert_eq!(reopened.get(&name(0), whole), None);
    assert_eq!(reopened.listings.load(Relaxed), 0, "no listing");
}

#[test]
fn a_removal_cut_short_is_finished_by_the_next_open() {
    let scratch = Scratch::new("store-cut");
    let store = Store::open(scratch.0.clone(), 1000);
    for n in 0..16 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    store.settle();
    let young = state(&store).young;
    assert_eq!(young, 4);
    drop(store);
    // A run ended in the middle of removing generation 2, and an earlier one left generation 1.
    for (number, n) in [(1, 90), (2, 91), (2, 92)] {
        let path = scratch.0.join(format!("g{number}")).join(name(n));
        fs::create_dir_all(path.parent().expect("a parent")).expect("the directory");
        fs::write(&path, [0; 100]).expect("the file");
    }
    let reopened = Store::open(scratch.0.clone(), 1000);
    reopened.settle();
    assert!(!reopened.generation(2).exists() && !reopened.generation(1).exists());
    assert!(reopened.generation(3).exists() && reopened.generation(4).exists());
    assert_eq!(
        reopened.listings.load(Relaxed),
        0,
        "the open stats, never lists"
    );
    assert_eq!(reopened.get(&name(15), whole), Some(vec![15; 100]));
}

/// The exit child's files a generation, of 10 bytes each.
const EXIT_FILES: u32 = 3000;
/// The exit child's cap: two generations of [`EXIT_FILES`] files.
const EXIT_CAP: u64 = 2 * EXIT_FILES as u64 * 10;
/// The directory the exit child works in, named by the parent.
const EXIT_DIR: &str = "M2V_RT_STORE_EXIT_DIR";

/// A child of the exit test: fills generations 1 and 2, makes 3 young with its last write, so 1
/// is dead and its removal thread starts, and exits at once. Does nothing when run alone.
#[test]
#[ignore = "a child of the exit-in-a-removal test"]
fn store_exit_child() {
    let Some(dir) = std::env::var_os(EXIT_DIR) else {
        return;
    };
    let store = Store::open(PathBuf::from(dir), EXIT_CAP);
    for n in 0..=2 * EXIT_FILES {
        assert!(store.put(&name(n), &[1; 10]));
    }
    std::process::exit(0);
}

#[test]
fn a_process_that_exits_in_a_removal_leaves_it_to_the_next_open() {
    let scratch = Scratch::new("store-exit");
    let exe = std::env::current_exe().expect("the test binary");
    let mut child = std::process::Command::new(&exe)
        .args(["tests::cache::store_exit_child", "--exact", "--ignored"])
        .env(EXIT_DIR, &scratch.0)
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("a child");
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the child") {
            break status;
        }
        if start.elapsed() > Duration::from_secs(120) {
            let _ = child.kill();
            panic!("the child ran past two minutes");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(status.success(), "the child wrote every file");
    let dead = scratch.0.join("g1");
    assert!(dead.exists(), "the child exited before its removal ended");
    let reopened = Store::open(scratch.0.clone(), EXIT_CAP);
    reopened.settle();
    assert!(!dead.exists(), "the next open finished the removal");
    assert_eq!(
        state(&reopened),
        State {
            young: 3,
            bytes: 10
        }
    );
    assert_eq!(reopened.listings.load(Relaxed), 0, "the state was whole");
    // The last file in young, the last of old moved into young.
    for n in [2 * EXIT_FILES, 2 * EXIT_FILES - 1] {
        assert_eq!(reopened.get(&name(n), whole), Some(vec![1; 10]), "file {n}");
    }
    assert!(scratch.bytes() <= EXIT_CAP);
}

#[test]
fn a_torn_or_missing_state_is_rebuilt_from_the_generations() {
    let scratch = Scratch::new("store-state");
    let store = Store::open(scratch.0.clone(), 1000);
    for n in 0..13 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    store.settle();
    assert_eq!(state(&store).young, 3);
    drop(store);
    for torn in [&b"cut short"[..], &[7; 24][..], &[][..]] {
        fs::write(scratch.0.join("state"), torn).expect("the state torn");
        let reopened = Store::open(scratch.0.clone(), 1000);
        assert_eq!(
            reopened.listings.load(Relaxed),
            1,
            "rebuilt from one listing"
        );
        // Young is the highest generation, counted full.
        assert_eq!(
            state(&reopened),
            State {
                young: 3,
                bytes: 500
            }
        );
        assert_eq!(reopened.get(&name(12), whole), Some(vec![12; 100]));
    }
    fs::remove_file(scratch.0.join("state")).expect("the state removed");
    let reopened = Store::open(scratch.0.clone(), 1000);
    assert_eq!(
        state(&reopened),
        State {
            young: 3,
            bytes: 500
        }
    );
    // The next write moves on, and the removal takes generation 2.
    assert!(reopened.put(&name(13), &[13; 100]));
    reopened.settle();
    assert_eq!(
        state(&reopened),
        State {
            young: 4,
            bytes: 100
        }
    );
    assert!(!reopened.generation(2).exists());
    assert!(scratch.bytes() <= 1000);
}

#[test]
fn a_torn_file_is_a_miss_and_is_removed() {
    let scratch = Scratch::new("store-torn");
    let store = Store::open(scratch.0.clone(), 1000);
    let data: Vec<u8> = (0..=255).collect();
    assert!(store.put(&name(1), &framed(&data)));
    let path = store.generation(1).join(name(1));
    let held = fs::read(&path).expect("the file");
    fs::write(&path, &held[..held.len() - 1]).expect("cut short");
    let check = |bytes: &[u8]| unframe(bytes).map(<[u8]>::to_vec);
    assert_eq!(store.get(&name(1), check), None);
    assert!(!path.exists(), "removed");
    // A file that holds other bytes of the same length than `holds` asks for stays.
    assert!(store.put(&name(2), b"lowered"));
    assert!(!store.holds(&name(2), b"refused"));
    assert!(store.holds(&name(2), b"lowered"));
    assert!(store.generation(1).join(name(2)).exists());
}

#[test]
fn stores_on_one_directory_share_its_files_and_its_count() {
    let scratch = Scratch::new("store-share");
    let first = Store::open(scratch.0.clone(), 1000);
    let second = Store::open(scratch.0.clone(), 1000);
    assert!(first.put(&name(1), b"compiled"));
    assert_eq!(second.get(&name(1), whole), Some(b"compiled".to_vec()));
    assert_eq!(state(&second).bytes, 8);
    assert!(first.put(&name(2), b"lowered"));
    assert!(second.holds(&name(2), b"lowered"));
    assert!(!second.holds(&name(4), b""));
    // A file another removed is a miss.
    fs::remove_file(first.generation(1).join(name(1))).expect("removed");
    assert_eq!(second.get(&name(1), whole), None);
    // A write renames over a file another holds open, which keeps reading what it opened.
    let mut open = fs::File::open(first.generation(1).join(name(2))).expect("open");
    assert!(second.put(&name(2), b"relowered"));
    let mut held = Vec::new();
    std::io::Read::read_to_end(&mut open, &mut held).expect("read");
    assert_eq!(held, b"lowered");
    assert_eq!(first.get(&name(2), whole), Some(b"relowered".to_vec()));
}

#[test]
fn a_blob_that_does_not_check_is_a_miss() {
    let data: Vec<u8> = (0..=255).collect();
    let file = framed(&data);
    assert_eq!(unframe(&file), Some(&data[..]));
    assert_eq!(unframe(&file[..file.len() - 1]), None, "cut short");
    let mut flipped = file.clone();
    flipped[100] ^= 1;
    assert_eq!(unframe(&flipped), None, "a flipped bit");
    let mut zeroed = file.clone();
    zeroed[40..72].fill(0);
    assert_eq!(unframe(&zeroed), None, "a zeroed run");
    assert_eq!(unframe(&[]), None, "empty");
    assert_eq!(unframe(&framed(&[])), Some(&[][..]), "no data");
}

/// Every one bit flipped in a buffer changes its checksum, at every offset whole words and the
/// tail past them cover, and so does its length.
#[test]
fn the_checksum_changes_with_any_one_bit() {
    let bytes: Vec<u8> = (0..77u32).map(|n| (n * 37 + 11) as u8).collect();
    let sum = checksum(&bytes);
    for at in 0..bytes.len() {
        for bit in 0..8 {
            let mut flipped = bytes.clone();
            flipped[at] ^= 1 << bit;
            assert_ne!(checksum(&flipped), sum, "byte {at} bit {bit}");
        }
    }
    assert_ne!(checksum(&bytes[..76]), sum, "one byte shorter");
    assert_ne!(
        checksum(&[0; 64]),
        checksum(&[0; 65]),
        "zeros of two lengths"
    );
}

/// The `pipelines/` directories an earlier version kept pipeline data in under young and old are
/// removed by the store's removal thread at open, and the lowerings stay.
#[test]
fn an_earlier_versions_pipeline_data_is_removed_at_open() {
    let scratch = Scratch::new("store-earlier");
    let store = Store::open(scratch.0.clone(), 1000);
    for n in 0..6 {
        assert!(store.put(&name(n), &[n as u8; 100]));
    }
    assert_eq!(state(&store).young, 2);
    drop(store);
    let earlier: Vec<PathBuf> = [1, 2]
        .iter()
        .map(|g| scratch.0.join(format!("g{g}/pipelines/{:032x}", 7)))
        .collect();
    for dir in &earlier {
        fs::create_dir_all(dir).expect("an earlier version's directory");
        fs::write(dir.join(format!("{:064x}", 1)), [1; 50]).expect("its file");
    }
    let reopened = Store::open(scratch.0.clone(), 1000);
    reopened.settle();
    for g in [1, 2] {
        let dir = scratch.0.join(format!("g{g}/pipelines"));
        assert!(!dir.exists(), "{} removed", dir.display());
    }
    for n in 0..6 {
        assert_eq!(
            reopened.get(&name(n), whole),
            Some(vec![n as u8; 100]),
            "file {n}"
        );
    }
}

/// The stress's cap: a generation of about twelve files, so the stores move on every few writes.
const STRESS_CAP: u64 = 4096;
/// The directory a stress child works in, named by the parent.
const STRESS_DIR: &str = "M2V_RT_STORE_STRESS_DIR";

/// The bytes of key `k`'s file, framed: a length and content of the key's own.
fn content(k: u32) -> Vec<u8> {
    framed(&vec![k as u8; 40 + 13 * k as usize])
}

/// `rounds` lookups of keys a seeded generator picks, each written where it missed; the count of
/// files read that were not the key's own bytes, whole.
fn stress(store: &Arc<Store>, seed: u64, rounds: usize) -> usize {
    let wrong = AtomicUsize::new(0);
    let mut x = seed | 1;
    for _ in 0..rounds {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let k = (x % 24) as u32;
        let want = content(k);
        let found = store.get(&name(k), |bytes| {
            if bytes != want {
                wrong.fetch_add(1, Relaxed);
            }
            Some(())
        });
        if found.is_none() {
            store.put(&name(k), &want);
        }
    }
    wrong.into_inner()
}

/// A child of the multi-process stress: runs it on the directory the parent names, and does
/// nothing when run alone.
#[test]
#[ignore = "a child of the multi-process stress"]
fn store_stress_child() {
    let Some(dir) = std::env::var_os(STRESS_DIR) else {
        return;
    };
    let store = Store::open(PathBuf::from(dir), STRESS_CAP);
    assert_eq!(stress(&store, u64::from(std::process::id()), 4000), 0);
    store.settle();
}

#[test]
fn stores_in_threads_and_processes_racing_writes_moves_and_flips_corrupt_nothing() {
    let scratch = Scratch::new("store-stress");
    let exe = std::env::current_exe().expect("the test binary");
    let children: Vec<_> = (0..3)
        .map(|_| {
            std::process::Command::new(&exe)
                .args(["tests::cache::store_stress_child", "--exact", "--ignored"])
                .env(STRESS_DIR, &scratch.0)
                .stdout(std::process::Stdio::piped())
                .spawn()
                .expect("a child")
        })
        .collect();
    let stores: Vec<Arc<Store>> = (0..3)
        .map(|_| Store::open(scratch.0.clone(), STRESS_CAP))
        .collect();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|t| {
                let store = &stores[t % 3];
                scope.spawn(move || stress(store, 0x9e37_79b9 + t as u64, 4000))
            })
            .collect();
        for worker in workers {
            assert_eq!(worker.join().expect("a worker"), 0, "files read whole");
        }
    });
    let start = Instant::now();
    for mut child in children {
        while child.try_wait().expect("the child").is_none() {
            if start.elapsed() > Duration::from_secs(120) {
                let _ = child.kill();
                panic!("a child ran past two minutes");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let out = child.wait_with_output().expect("the child's output");
        let said = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "a child read a file whole each time: {said}"
        );
        assert!(
            said.contains("1 passed"),
            "the child ran the stress: {said}"
        );
    }
    for store in &stores {
        store.settle();
    }
    // With every writer done, young counts at least what it holds: no count was lost.
    let now = state(&stores[0]);
    assert!(
        now.bytes >= bytes_under(&stores[0].generation(now.young)),
        "{now:?}"
    );
    assert!(now.young > 10, "the stores moved on: {now:?}");
    // Old and young, and at most a generation a removal could not finish while a writer still
    // wrote into it.
    assert!(
        scratch.bytes() <= STRESS_CAP * 3 / 2,
        "{} bytes",
        scratch.bytes()
    );
    for k in 0..24 {
        let read = stores[0].get(&name(k), |bytes| Some(bytes == content(k)));
        assert_ne!(read, Some(false), "key {k} whole");
    }
}
