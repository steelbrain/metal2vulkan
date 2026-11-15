//! The bounds on a process's FALLBACK repro bundles (`repro.rs`): past its bundles or its bytes
//! nothing more is written, and that is said once; a bundle already written is not written again
//! and is not said; a write that fails is said once and fails nothing. Each test holds a tally of
//! its own, so none depends on another's bundles or on the environment.

use crate::repro::{Bounded, Tally, BUNDLES, BYTES};
use metal2vulkan::repro::ReproBundle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// A private directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "m2v-repro-tally-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Scratch(path)
    }

    /// The bundles written under it.
    fn bundles(&self) -> usize {
        std::fs::read_dir(&self.0).map_or(0, |entries| entries.count())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write the bundle of `input` under `base` through `tally`, each line said into `said`.
fn write(tally: &Bounded, base: &Path, input: &[u8], said: &mut Vec<String>) {
    let bundle = || {
        Ok(Some(
            ReproBundle::new("input.bin", input).failure("why: refused"),
        ))
    };
    tally.write(base, "a call refused", bundle, &mut |line: String| {
        said.push(line)
    });
}

/// The process's bounds are 32 bundles and 256 MiB.
#[test]
fn a_process_writes_32_bundles_and_256_mib_at_most() {
    assert_eq!((BUNDLES, BYTES), (32, 256 << 20));
}

/// Past the bundles a tally takes, nothing more is written, and the bound is said once.
#[test]
fn past_its_bundles_nothing_more_is_written_and_the_bound_is_said_once() {
    let scratch = Scratch::new();
    let tally = Bounded(Mutex::new(Tally::new(BUNDLES, BYTES)));
    let mut said = Vec::new();
    for n in 0..BUNDLES + 3 {
        write(&tally, &scratch.0, &n.to_le_bytes(), &mut said);
    }
    assert_eq!(scratch.bundles(), BUNDLES as usize);
    let written = said.iter().filter(|l| l.contains("a call refused")).count();
    let bound = said
        .iter()
        .filter(|l| l.contains("no more are written"))
        .count();
    assert_eq!((written, bound), (BUNDLES as usize, 1), "{said:#?}");
}

/// A bundle that would take a tally past its bytes is not written, and nor is any later one,
/// however small: the bound is said once.
#[test]
fn past_its_bytes_nothing_more_is_written_and_the_bound_is_said_once() {
    let scratch = Scratch::new();
    let tally = Bounded(Mutex::new(Tally::new(BUNDLES, 4096)));
    let mut said = Vec::new();
    write(&tally, &scratch.0, &[1; 2048], &mut said);
    write(&tally, &scratch.0, &[2; 2048], &mut said);
    write(&tally, &scratch.0, &[3; 16], &mut said);
    assert_eq!(scratch.bundles(), 1, "{said:#?}");
    assert_eq!(said.len(), 2, "{said:#?}");
    assert!(said[1].contains("no more are written"), "{said:#?}");
}

/// A bundle whose directory is there is not written again and not said; the same bundle on
/// another tally (another process) finds it too.
#[test]
fn a_bundle_written_before_is_not_written_or_said_again() {
    let scratch = Scratch::new();
    let tally = Bounded(Mutex::new(Tally::new(BUNDLES, BYTES)));
    let mut said = Vec::new();
    write(&tally, &scratch.0, b"one", &mut said);
    write(&tally, &scratch.0, b"one", &mut said);
    let other = Bounded(Mutex::new(Tally::new(BUNDLES, BYTES)));
    write(&other, &scratch.0, b"one", &mut said);
    assert_eq!(scratch.bundles(), 1);
    assert_eq!(said.len(), 1, "{said:#?}");
}

/// A write that fails (the base is a file) is said once and fails nothing.
#[test]
fn a_failed_write_is_said_once() {
    let scratch = Scratch::new();
    let base = scratch.0.join("a-file");
    std::fs::write(&base, b"not a directory").unwrap();
    let tally = Bounded(Mutex::new(Tally::new(BUNDLES, BYTES)));
    let mut said = Vec::new();
    write(&tally, &base, b"one", &mut said);
    write(&tally, &base, b"two", &mut said);
    assert_eq!(said.len(), 1, "{said:#?}");
    assert!(said[0].contains("not written"), "{said:#?}");
}
