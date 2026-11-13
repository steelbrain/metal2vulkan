//! The free heap a program's threads leave is handed back while the device is idle
//! (`idle_sweep.rs`'s `HeapLook`). With the device's idle sweep running (from its first buffer),
//! the test's thread allocates [`CHUNKS`] chunks of [`CHUNK`] bytes, writes each, and frees all but
//! one in every [`KEPT`] and the last: glibc keeps the freed chunks as free heap between live
//! ones, resident, which no free hands back. With no large making and nothing else of the
//! program's, the device's thread trims (`WorkCounts::heap_trims`) within [`IDLE`] and the
//! process's resident set falls by [`HANDED_BACK_MIB`] or more. Without the idle sweep's look at
//! the heap, nothing trims it and the resident set stays.
//!
//! A device test of glibc's allocator: compiled only under the validation crate's GPU safety cfgs
//! (`m2v_gpu`, `m2v_cpu_icd`), on Linux with the GNU environment, on the kind of device the cfg
//! selects.

use super::blit_holds::{device, Scratch};
use super::spares_bench::process_memory;
use crate::idle_sweep::HEAP_EVERY;
use crate::ResourceOptions;
use std::time::{Duration, Instant};

/// Each chunk's bytes: 64 KiB, under the least size glibc maps a chunk apart at (128 KiB).
const CHUNK: usize = 64 << 10;

/// The chunks: 768 MiB.
const CHUNKS: usize = 12_288;

/// One chunk in every so many stays live, so the freed ones lie between live chunks.
const KEPT: usize = 16;

/// The least the resident set falls by, in MiB, of the 720 MiB freed.
const HANDED_BACK_MIB: u64 = 512;

/// The longest the device may take, idle, to trim: six looks, two of them due, with room for a
/// loaded host.
const IDLE: Duration = Duration::from_secs(6 * HEAP_EVERY.as_secs());

/// The process's resident set, in MiB.
fn resident_mib() -> u64 {
    process_memory()["vm-rss"] / 1024
}

#[test]
fn an_idle_device_hands_back_the_free_heap() {
    let scratch = Scratch::new("idle-heap-trim");
    let device = device(&scratch.0);
    // The device's first creation call that may wait starts its idle sweep (`Device::keep_room`).
    let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let trims = || device.work_counts().heap_trims;
    let before = trims();
    let mut chunks: Vec<Vec<u8>> = (0..CHUNKS).map(|i| vec![i as u8 | 1; CHUNK]).collect();
    let allocated = resident_mib();
    let mut at = 0;
    chunks.retain(|_| {
        let kept = at % KEPT == 0 || at == CHUNKS - 1;
        at += 1;
        kept
    });
    let freed = resident_mib();
    let idle = Instant::now();
    while trims() == before && idle.elapsed() < IDLE {
        std::thread::sleep(Duration::from_millis(200));
    }
    let handed_back = resident_mib();
    println!(
        "IDLE-HEAP-TRIM {}: resident {allocated} MiB with the chunks, {freed} after freeing, \
         {handed_back} after {:.1} s idle; {} trims",
        device.name(),
        idle.elapsed().as_secs_f64(),
        trims() - before
    );
    assert!(
        trims() > before,
        "{:.1} s idle, the device's thread did not hand the free heap back",
        idle.elapsed().as_secs_f64()
    );
    assert!(
        freed >= handed_back + HANDED_BACK_MIB,
        "the resident set fell from {freed} MiB to {handed_back} only"
    );
    drop((chunks, buffer));
}
