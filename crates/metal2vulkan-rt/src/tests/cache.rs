//! The cache directory on the host: lowered kernels and pipelines' data share one cap, past which
//! the least recently used files go, by their time on disk across stores; stores on one directory
//! share its files, a file another removed is a miss, and pipeline data that does not check is a
//! miss.

use crate::cache::{frame, unframe, Store};
use std::fs;
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

    /// The bytes of every file under the directory.
    fn bytes(&self) -> u64 {
        fn under(dir: &Path) -> u64 {
            fs::read_dir(dir)
                .expect("the directory")
                .map(|entry| {
                    let entry = entry.expect("an entry");
                    let meta = entry.metadata().expect("its metadata");
                    match meta.is_dir() {
                        true => under(&entry.path()),
                        false => meta.len(),
                    }
                })
                .sum()
        }
        under(&self.0)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A file the store keeps: a lowering for even `n`, a pipeline's data for odd.
fn name(n: u8) -> String {
    match n % 2 {
        0 => format!("translations/{n:064x}"),
        _ => format!("pipelines/{:032x}/{n:064x}", 7),
    }
}

/// Writes `bytes` as `file` under `root`, as another device would, with its time on disk
/// `age` ago.
fn written(root: &Path, file: &str, bytes: &[u8], age: Duration) {
    let path = root.join(file);
    fs::create_dir_all(path.parent().expect("a parent")).expect("the directory");
    fs::write(&path, bytes).expect("the file");
    let opened = fs::File::options().write(true).open(&path).expect("open");
    opened
        .set_modified(SystemTime::now() - age)
        .expect("its time");
}

#[test]
fn a_store_filled_past_its_cap_keeps_the_most_recently_used_within_it() {
    let scratch = Scratch::new("store-cap");
    let store = Store::open(scratch.0.clone(), 1000);
    for n in 0..10 {
        assert!(store.put(&name(n), &[n; 300]), "file {n} written");
        // Read after every write, the first file stays the most recently used.
        assert_eq!(store.get(&name(0)), Some(vec![0; 300]), "file 0 after {n}");
        assert!(store.total() <= 1000, "{} bytes after {n}", store.total());
        assert_eq!(scratch.bytes(), store.total(), "the disk after {n}");
    }
    // Two lowerings (0, 8) and a pipeline's data (9) are the three most recent: both kinds share
    // the cap.
    let kept: Vec<bool> = (0..10).map(|n| store.get(&name(n)).is_some()).collect();
    let mut want = [false; 10];
    for n in [0, 8, 9] {
        want[n] = true;
    }
    assert_eq!(kept, want, "the files kept");
    // A file past the cap alone is not kept, and evicts nothing.
    assert!(!store.put(&name(10), &[10; 1001]));
    assert_eq!(store.get(&name(10)), None);
    assert_eq!(store.total(), 900);
    // Reopened with a smaller cap, the store evicts down to it.
    drop(store);
    let reopened = Store::open(scratch.0.clone(), 600);
    assert_eq!(reopened.total(), 600);
    assert_eq!(scratch.bytes(), 600);
}

#[test]
fn a_store_opened_on_files_others_wrote_evicts_the_least_recently_used_on_disk() {
    let scratch = Scratch::new("store-age");
    let hour = Duration::from_secs(3600);
    // Four files of 100 bytes, written 5, 4, 3 and 2 hours ago by other devices.
    for n in 0..4u8 {
        written(&scratch.0, &name(n), &[n; 100], hour * u32::from(5 - n));
    }
    // A use moves an hours-old file's time forward: file 0 becomes the most recent on disk.
    let store = Store::open(scratch.0.clone(), 1000);
    assert_eq!(store.total(), 400);
    assert_eq!(store.get(&name(0)), Some(vec![0; 100]));
    drop(store);
    // Reopened with room for two, the store keeps file 0 and file 3, the latest used.
    let reopened = Store::open(scratch.0.clone(), 200);
    let kept: Vec<bool> = (0..4).map(|n| reopened.get(&name(n)).is_some()).collect();
    assert_eq!(kept, [true, false, false, true], "the files kept");
    assert_eq!(scratch.bytes(), 200);
}

#[test]
fn stores_on_one_directory_share_its_files() {
    let scratch = Scratch::new("store-share");
    let first = Store::open(scratch.0.clone(), 1000);
    let second = Store::open(scratch.0.clone(), 1000);
    first.put(&name(1), b"compiled");
    assert_eq!(second.get(&name(1)), Some(b"compiled".to_vec()));
    assert_eq!(second.total(), 8);
    first.put(&name(2), b"lowered");
    assert!(second.holds(&name(2), b"lowered"));
    assert!(!second.holds(&name(2), b"refused"));
    assert!(!second.holds(&name(4), b""));
    // A file another removed is a miss, and leaves the store's count.
    fs::remove_file(scratch.0.join(name(1))).expect("removed");
    assert_eq!(second.get(&name(1)), None);
    assert_eq!(second.total(), 7);
}

#[test]
fn pipeline_data_that_does_not_check_is_a_miss() {
    let data: Vec<u8> = (0..=255).collect();
    let file = frame(&data);
    assert_eq!(unframe(&file), Some(&data[..]));
    assert_eq!(unframe(&file[..file.len() - 1]), None, "cut short");
    let mut flipped = file.clone();
    flipped[100] ^= 1;
    assert_eq!(unframe(&flipped), None, "a flipped bit");
    assert_eq!(unframe(&[]), None, "empty");
    assert_eq!(unframe(&frame(&[])), Some(&[][..]), "no data");
}
