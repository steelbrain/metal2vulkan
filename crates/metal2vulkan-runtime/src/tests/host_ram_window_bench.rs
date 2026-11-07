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
//! Each of the first three waits 1 s after the use begins. Each order prints the makings of 5 ms or
//! more, the slowest and the median, the chunks imported ahead of the objects and by their makings,
//! and the heaps bound in a chunk (`HOST-RAM-WINDOW`), and its first [`SLOW_LINES`] makings of 5 ms
//! or more: each one's take, its chunk, and what it made, the counts it moved
//! (`HOST-RAM-WINDOW-SLOW`). The orders run [`REPEATS`] times over RAM in 2 MiB pages, the main
//! case (QEMU's file-backed RAM where shared memory takes the advice for huge pages), then over RAM
//! in 4 KiB pages (`RamPages`), each order's line with the system's count of shared memory in huge
//! pages at its end.
//!
//! A second bench runs the consumer's three orders over RAM in 2 MiB pages on a device made with
//! the options a consumer (a VMM) makes its device with ([`consumer_device`]), in whichever build
//! it is compiled in, a debug build's checks included: each slow making also lists the driver's
//! messages that reached the sink while it ran, and each order the messages of all its makings.
//! Both are ignored: run them with `cargo test -p metal2vulkan-runtime --lib host_ram_window_bench
//! -- --ignored --nocapture` under `--cfg m2v_gpu`, in release (`--release`) and in debug.

use super::blit_holds::Scratch;
use super::host_ram_device::{ahead_device, fresh_ram_in, gate, itself, resolver, RamPages};
use crate::{
    Device, DeviceKind, DeviceOptions, HeapDescriptor, HeapType, HostMemory, MessageSeverity,
    MessageSink, ResourceOptions, ShaderCacheOptions,
};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
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
/// The most slow makings an order lists one by one.
const SLOW_LINES: usize = 40;

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

/// The messages a device's sink received: each one's arrival, severity and id name.
type Heard = Arc<Mutex<Vec<(Instant, MessageSeverity, String)>>>;

/// The ids of `heard`'s messages that arrived in `from..=to`, each with its count and the
/// severity of its last.
fn heard_in(heard: &Heard, from: Instant, to: Instant) -> String {
    let mut ids: BTreeMap<&str, (usize, MessageSeverity)> = BTreeMap::new();
    let heard = heard.lock().unwrap();
    for (at, severity, id) in heard.iter() {
        if (from..=to).contains(at) {
            let entry = ids.entry(id.as_str()).or_insert((0, *severity));
            *entry = (entry.0 + 1, *severity);
        }
    }
    let listed: Vec<String> = (ids.iter())
        .map(|(id, (count, severity))| format!("{id} ({severity:?}) x{count}"))
        .collect();
    match listed.is_empty() {
        true => "none".to_owned(),
        false => listed.join(", "),
    }
}

/// A device of the cfg's kind made with the options a consumer (a VMM) makes its device with,
/// its shader cache in `cache`; `None` where it imports no host memory. The options: a message
/// sink at `Info`, so the device turns on `VK_EXT_debug_utils` and the driver's messages of that
/// severity and above reach the sink, which keeps each in `heard`; GPU times from the host's
/// clock; a shader cache on disk; a video-decode queue beside the device's own, one queue held
/// back, and the video-decode extensions where the device has them. The consumer also asks for
/// its windows' instance extensions, which this device leaves out.
fn consumer_device(cache: &Path, heard: &Heard) -> Option<Device> {
    let heard = Arc::clone(heard);
    let sink = MessageSink::new(move |message| {
        let id = message.id_name.clone();
        heard
            .lock()
            .unwrap()
            .push((Instant::now(), message.severity, id));
    });
    let video = [
        "VK_KHR_video_queue",
        "VK_KHR_video_decode_queue",
        "VK_KHR_video_decode_h264",
        "VK_KHR_synchronization2",
        "VK_KHR_video_decode_h265",
        "VK_KHR_unified_image_layouts",
    ];
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(sink.with_least_severity(MessageSeverity::Info)),
        gpu_times_from_host: true,
        shader_cache: ShaderCacheOptions::in_directory(cache),
        extra_queues: vec![ash::vk::QueueFlags::VIDEO_DECODE_KHR],
        reserved_queues: 1,
        optional_device_extensions: (video.iter())
            .map(|&name| CString::new(name).unwrap())
            .collect(),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM: the device imports no host memory here; nothing to test");
        return None;
    }
    // As `ahead_device` asks it, so that no making meets the import that answers it.
    device.imports_shared_host_pointers().unwrap();
    Some(device)
}

/// Make the order's heaps over fresh RAM in `pages`, timing each making, and print what it took,
/// with the messages `heard` received meanwhile where given.
fn run(device: &Device, order: &Order, pages: RamPages, heard: Option<&Heard>) {
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
    let mut slow = 0;
    let first = Instant::now();
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
            slow += 1;
        }
        if took >= SLOW && slow_lines.len() < SLOW_LINES {
            let delta = |field: fn(&crate::WorkCounts) -> u64| field(&now) - field(&counts);
            let ahead = delta(|counts| counts.host_ram_chunks_imported_ahead);
            slow_lines.push(format!(
                "take {take} (chunk {}, heap {} in it): {:.2} ms, the heap {:.2} ms; bound in its \
                 run's import {}, in the chunk's {}; chunks imported by the making {}, ahead by \
                 the thread meanwhile {ahead}; run imports live {}{}",
                offset / CHUNK,
                offset % CHUNK / HEAP,
                took.as_secs_f64() * 1e3,
                heap_took.as_secs_f64() * 1e3,
                delta(|counts| counts.run_bound_buffers_made),
                delta(|counts| counts.chunk_bound_buffers_made),
                delta(|counts| counts.host_ram_chunks_imported) - ahead,
                now.run_imports_live,
                heard.map_or(String::new(), |heard| {
                    format!(
                        "; messages meanwhile: {}",
                        heard_in(heard, began, Instant::now())
                    )
                }),
            ));
        }
    }
    let after = device.work_counts();
    let ms = |time: Duration| time.as_secs_f64() * 1e3;
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
    if let Some(heard) = heard {
        eprintln!(
            "HOST-RAM-WINDOW {} in {pages:?} pages: the messages of its makings: {}",
            order.name,
            heard_in(heard, first, Instant::now())
        );
    }
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
                run(&device, &order, pages, None);
            }
        }
    }
}

/// The consumer's orders over RAM in 2 MiB pages, on a device made with a consumer's options
/// ([`consumer_device`]), in the build it is compiled in.
#[test]
#[ignore = "a benchmark"]
fn host_ram_window_bench_with_a_consumer_s_options() {
    let scratch = Scratch::new("window-bench");
    let heard = Heard::default();
    let made = Instant::now();
    let Some(device) = consumer_device(&scratch.0, &heard) else {
        return;
    };
    eprintln!(
        "HOST-RAM-WINDOW a consumer's device options on {}, debug assertions {}; the messages \
         of its making: {}",
        device.name(),
        cfg!(debug_assertions),
        heard_in(&heard, made, Instant::now())
    );
    for repeat in 0..REPEATS {
        for order in orders().into_iter().filter(|order| order.consumer) {
            eprintln!(
                "HOST-RAM-WINDOW {} in {:?} pages, repeat {repeat}",
                order.name,
                RamPages::Huge
            );
            run(&device, &order, RamPages::Huge, Some(&heard));
        }
    }
    drop(device);
}
