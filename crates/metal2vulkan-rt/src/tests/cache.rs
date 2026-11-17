//! The cache directory on the host: a store filled past its cap keeps the most recently used
//! files within it, devices on one directory share its files, and pipeline data that does not
//! check is a miss.

use crate::cache::{frame, unframe, Store};
use std::fs;
use std::path::PathBuf;

/// A directory of its own under the OS temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("m2v-rt-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        Scratch(dir)
    }

    /// The bytes of every file in the directory.
    fn bytes(&self) -> u64 {
        fs::read_dir(&self.0)
            .expect("the directory")
            .map(|entry| {
                entry
                    .expect("an entry")
                    .metadata()
                    .expect("its metadata")
                    .len()
            })
            .sum()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A file name the store keeps: 64 hex digits.
fn name(n: u8) -> String {
    format!("{n:064x}")
}

#[test]
fn a_store_filled_past_its_cap_keeps_the_most_recently_used_within_it() {
    let scratch = Scratch::new("store-cap");
    let mut store = Store::open(scratch.0.clone(), 1000);
    for n in 0..10 {
        store.put(&name(n), &[n; 300]);
        // Read after every write, the first file stays the most recently used.
        assert_eq!(store.get(&name(0)), Some(vec![0; 300]), "file 0 after {n}");
        assert!(store.total() <= 1000, "{} bytes after {n}", store.total());
        assert_eq!(scratch.bytes(), store.total(), "the disk after {n}");
    }
    let kept: Vec<bool> = (0..10).map(|n| store.get(&name(n)).is_some()).collect();
    let mut want = [false; 10];
    for n in [0, 8, 9] {
        want[n] = true;
    }
    assert_eq!(kept, want, "the files kept");
    // A file past the cap alone is not kept, and evicts nothing.
    store.put(&name(10), &[10; 1001]);
    assert_eq!(store.get(&name(10)), None);
    assert_eq!(store.total(), 900);
    // Reopened with a smaller cap, the store evicts down to it.
    drop(store);
    let reopened = Store::open(scratch.0.clone(), 600);
    assert_eq!(reopened.total(), 600);
    assert_eq!(scratch.bytes(), 600);
}

#[test]
fn stores_on_one_directory_share_its_files() {
    let scratch = Scratch::new("store-share");
    let mut first = Store::open(scratch.0.clone(), 1000);
    let mut second = Store::open(scratch.0.clone(), 1000);
    first.put(&name(1), b"lowered");
    assert_eq!(second.get(&name(1)), Some(b"lowered".to_vec()));
    assert_eq!(second.total(), 7);
    assert!(second.holds(&name(1), b"lowered"));
    assert!(!second.holds(&name(1), b"refused"));
    assert!(!second.holds(&name(2), b""));
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
