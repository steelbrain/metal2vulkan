//! The time a consumer's objects over fresh guest RAM take to make (`host_ram.rs`): 256 no-copy
//! heaps of 2 MiB over a RAM a device uses (`Device::use_host_ram`), each over a run of its own,
//! in three orders, each on fresh RAM of its own (a `memfd` mapped shared and advised for huge
//! pages on Linux, as QEMU's RAM is; anonymous shared memory elsewhere) and a use of its own:
//!
//! - **upward**: two bursts of 128 consecutive heaps, 1 s apart, over chunks 4 to 7 and 8 to 11
//!   of a RAM of 16, each heap the one above the last;
//! - **downward**: the same runs from the top, two bursts over chunks 11 to 8 and 7 to 4, each
//!   heap the one below the last;
//! - **scattered**: 256 heaps over distinct runs drawn over a RAM of 32 chunks, 10 ms apart.
//!
//! Then the same three in a consumer's shape (a VMM's take of a guest's 2 MiB of fresh RAM): a
//! RAM of 16 chunks, its takes back to back from the moment the use begins, each preceded by
//! the CPU's zeroing of its 2 MiB (untimed), and each the heap and a 64 KiB `Shared` buffer at
//! its offset 0: **upward** over the RAM's first 256 runs, **downward** over its last 256 from
//! the top, **scattered** over 256 distinct runs drawn over all of it.
//!
//! Each of the first three waits 1 s after the use begins. Each order prints the makings of 5 ms or more, the
//! slowest and the median, the chunks imported ahead of the objects and by their makings, and
//! the heaps bound in a chunk (`HOST-RAM-WINDOW`), and each making of 5 ms or more: its take,
//! its chunk, and what it made, the counts it moved (`HOST-RAM-WINDOW-SLOW`). The orders run
//! [`REPEATS`] times over RAM in 2 MiB pages, the main case (QEMU's file-backed RAM where shared
//! memory takes the advice for huge pages), then over RAM in 4 KiB pages (`RamPages`), each
//! order's line with the system's count of shared memory in huge pages at its end. Ignored: run it in release,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_window_bench -- --ignored
//! --nocapture` under `--cfg m2v_gpu`.

use super::host_ram_device::{ahead_device, fresh_ram_in, gate, itself, resolver, RamPages};
use crate::{HeapDescriptor, HeapType, HostMemory, ResourceOptions};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MIB: u64 = 1 << 20;
const CHUNK: u64 = 64 * MIB;
const HEAP: u64 = 2 * MIB;
const PER_CHUNK: u64 = CHUNK / HEAP;
const BURST: u64 = 128;
/// A making this slow is one the consumer's frame notices.
const SLOW: Duration = Duration::from_millis(5);
/// How many times each order runs in each kind of pages, each on fresh RAM.
const REPEATS: usize = 4;

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One order: its name, its RAM's chunks, each heap's offset into the RAM with the pause
/// before its making, and whether each take is a consumer's (the module docs).
struct Order {
    name: &'static str,
    chunks: u64,
    takes: Vec<(u64, Duration)>,
    consumer: bool,
}

/// The bytes of the buffer a consumer's take makes in its heap.
const TAKE_BUFFER: u64 = 64 << 10;

fn orders() -> Vec<Order> {
    let settle = Duration::from_secs(1);
    let burst = |first: u64, down: bool| {
        (0..BURST).map(move |index| {
            let heap = match down {
                false => first + index,
                true => first - index,
            };
            let pause = if index == 0 { settle } else { Duration::ZERO };
            (heap * HEAP, pause)
        })
    };
    let upward = burst(4 * PER_CHUNK, false).chain(burst(8 * PER_CHUNK, false));
    let downward = burst(12 * PER_CHUNK - 1, true).chain(burst(8 * PER_CHUNK - 1, true));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut drawn: Vec<u64> = Vec::new();
    while drawn.len() < 2 * BURST as usize {
        let heap = rng.next() % (32 * PER_CHUNK);
        if !drawn.contains(&heap) {
            drawn.push(heap);
        }
    }
    let scattered = (drawn.iter().enumerate()).map(|(index, &heap)| {
        let pause = match index {
            0 => settle,
            _ => Duration::from_millis(10),
        };
        (heap * HEAP, pause)
    });
    let consumer_chunks = 16;
    let slots = consumer_chunks * PER_CHUNK;
    let mut shuffled: Vec<u64> = (0..slots).collect();
    for index in 0..2 * BURST as usize {
        let other = index + (rng.next() % (slots - index as u64)) as usize;
        shuffled.swap(index, other);
    }
    shuffled.truncate(2 * BURST as usize);
    let back_to_back = |heaps: Vec<u64>| -> Vec<(u64, Duration)> {
        (heaps.into_iter())
            .map(|heap| (heap * HEAP, Duration::ZERO))
            .collect()
    };
    let order = |name, chunks, takes, consumer| Order {
        name,
        chunks,
        takes,
        consumer,
    };
    vec![
        order("upward", 16, upward.collect(), false),
        order("downward", 16, downward.collect(), false),
        order("scattered", 32, scattered.collect(), false),
        order(
            "consumer upward",
            consumer_chunks,
            back_to_back((0..2 * BURST).collect()),
            true,
        ),
        order(
            "consumer downward",
            consumer_chunks,
            back_to_back((slots - 2 * BURST..slots).rev().collect()),
            true,
        ),
        order(
            "consumer scattered",
            consumer_chunks,
            back_to_back(shuffled),
            true,
        ),
    ]
}

/// Make the order's heaps over fresh RAM in `pages`, timing each making, and print what it took.
fn run(device: &crate::Device, order: &Order, pages: RamPages) {
    let (ram, start, _) = fresh_ram_in(device, order.chunks * CHUNK, pages);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::default());
    let mut times = Vec::new();
    let mut heaps = Vec::new();
    let mut buffers = Vec::new();
    let mut slow_lines = Vec::new();
    let mut writes = Duration::ZERO;
    for (take, &(offset, pause)) in order.takes.iter().enumerate() {
        std::thread::sleep(pause);
        let at = (start + offset) as *mut u8;
        if order.consumer {
            let began = Instant::now();
            // SAFETY: the RAM's 2 MiB of this take, mapped and writable while the RAM lives, which
            // no object lies over yet.
            unsafe { std::ptr::write_bytes(at, 0, HEAP as usize) };
            writes += began.elapsed();
        }
        // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
        let memory = unsafe { HostMemory::new(NonNull::new(at).unwrap()) }
            .with_runs(resolver(&runs, itself))
            .with_import_gate(gate(&gates, true));
        let counts = device.work_counts();
        let began = Instant::now();
        // SAFETY: as above.
        let heap = unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) };
        let heap = heap.unwrap();
        let heap_took = began.elapsed();
        if order.consumer {
            let buffer = heap.new_buffer_at_offset(TAKE_BUFFER, ResourceOptions::shared(), 0);
            buffers.push(buffer.unwrap());
        }
        let took = began.elapsed();
        let now = device.work_counts();
        times.push(took);
        heaps.push(heap);
        if took >= SLOW {
            let delta = |field: fn(&crate::WorkCounts) -> u64| field(&now) - field(&counts);
            let ahead = delta(|counts| counts.host_ram_chunks_imported_ahead);
            slow_lines.push(format!(
                "take {take} (chunk {}, heap {} in it): {:.2} ms, the heap {:.2} ms; bound in its \
                 run's import {}, in the chunk's {}; chunks imported by the making {}, ahead by \
                 the thread meanwhile {ahead}; run imports live {}",
                offset / CHUNK,
                offset % CHUNK / HEAP,
                took.as_secs_f64() * 1e3,
                heap_took.as_secs_f64() * 1e3,
                delta(|counts| counts.run_bound_buffers_made),
                delta(|counts| counts.chunk_bound_buffers_made),
                delta(|counts| counts.host_ram_chunks_imported) - ahead,
                now.run_imports_live,
            ));
        }
    }
    let after = device.work_counts();
    let ms = |time: Duration| time.as_secs_f64() * 1e3;
    let slow = times.iter().filter(|&&time| time >= SLOW).count();
    let worst = times.iter().max().copied().unwrap_or_default();
    let total: Duration = times.iter().sum();
    times.sort();
    let median = times[times.len() / 2];
    let ahead = after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead;
    let imported = after.host_ram_chunks_imported - before.host_ram_chunks_imported;
    let bound = after.chunk_bound_buffers_made - before.chunk_bound_buffers_made;
    let run_bound = after.run_bound_buffers_made - before.run_bound_buffers_made;
    eprintln!(
        "HOST-RAM-WINDOW {} in {pages:?} pages on {}: {} makings, {slow} of 5 ms or more, \
         slowest {:.2} ms, median {:.3} ms, all {:.1} ms; chunks imported ahead {ahead}, by the \
         makings {}; heaps bound in a chunk {bound}, in their run's import {run_bound}; the gate \
         asked {} times; the CPU's writes before them {:.1} ms; shared memory in huge pages {:?} \
         kB",
        order.name,
        device.name(),
        times.len(),
        ms(worst),
        ms(median),
        ms(total),
        imported - ahead,
        gates.lock().unwrap().len(),
        ms(writes),
        shmem_huge_kb(),
    );
    for line in slow_lines {
        eprintln!(
            "HOST-RAM-WINDOW-SLOW {} in {pages:?} pages: {line}",
            order.name
        );
    }
    drop((buffers, heaps, used, ram));
}

/// The system's shared memory in huge pages, in kB, where it says (Linux).
fn shmem_huge_kb() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("ShmemHugePages:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// The orders of the module docs, on a device of the cfg's kind, as the product makes it.
#[test]
#[ignore = "a release-mode benchmark"]
fn host_ram_window_bench() {
    let Some(device) = ahead_device() else {
        return;
    };
    for pages in [RamPages::Huge, RamPages::Small] {
        for repeat in 0..REPEATS {
            for order in orders() {
                eprintln!(
                    "HOST-RAM-WINDOW {} in {pages:?} pages, repeat {repeat}",
                    order.name
                );
                run(&device, &order, pages);
            }
        }
    }
}
