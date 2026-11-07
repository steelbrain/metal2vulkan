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
//! Each order waits 1 s after the use begins. Each prints the makings of 5 ms or more, the
//! slowest and the median, the chunks imported ahead of the objects and by their makings, and
//! the heaps bound in a chunk (`HOST-RAM-WINDOW`). Ignored: run it in release,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_window_bench -- --ignored
//! --nocapture` under `--cfg m2v_gpu`.

use super::host_ram_device::{ahead_device, fresh_ram, gate, itself, resolver};
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

/// One order: its name, its RAM's chunks, and each heap's offset into the RAM with the pause
/// before its making.
struct Order {
    name: &'static str,
    chunks: u64,
    takes: Vec<(u64, Duration)>,
}

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
    vec![
        Order {
            name: "upward",
            chunks: 16,
            takes: upward.collect(),
        },
        Order {
            name: "downward",
            chunks: 16,
            takes: downward.collect(),
        },
        Order {
            name: "scattered",
            chunks: 32,
            takes: scattered.collect(),
        },
    ]
}

/// Make the order's heaps over fresh RAM, timing each making, and print what it took.
fn run(device: &crate::Device, order: &Order) {
    let (ram, start, _) = fresh_ram(device, order.chunks * CHUNK);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::default());
    let mut times = Vec::new();
    let mut heaps = Vec::new();
    for &(offset, pause) in &order.takes {
        std::thread::sleep(pause);
        // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
        let memory = unsafe { HostMemory::new(NonNull::new((start + offset) as *mut u8).unwrap()) }
            .with_runs(resolver(&runs, itself))
            .with_import_gate(gate(&gates, true));
        let began = Instant::now();
        // SAFETY: as above.
        let heap = unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) };
        times.push(began.elapsed());
        heaps.push(heap.unwrap());
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
    eprintln!(
        "HOST-RAM-WINDOW {} on {}: {} makings, {slow} of 5 ms or more, slowest {:.2} ms, median \
         {:.3} ms, all {:.1} ms; chunks imported ahead {ahead}, by the makings {}; heaps bound \
         in a chunk {bound}, the gate asked {} times",
        order.name,
        device.name(),
        times.len(),
        ms(worst),
        ms(median),
        ms(total),
        imported - ahead,
        gates.lock().unwrap().len(),
    );
    drop((heaps, used, ram));
}

/// The orders of the module docs, on a device of the cfg's kind, as the product makes it.
#[test]
#[ignore = "a release-mode benchmark"]
fn host_ram_window_bench() {
    let Some(device) = ahead_device() else {
        return;
    };
    for order in orders() {
        run(&device, &order);
    }
}
