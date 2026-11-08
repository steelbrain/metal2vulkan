//! What one `vkAllocateMemory` of a block's size holds of the device's submissions, by size: a
//! thread commits a one-word fill on a queue of its own every millisecond and waits for it, while
//! the bench's own thread allocates device-local memory on the same `VkDevice`, one allocation
//! every [`GAP`], [`MAKES`] of each kind, and then frees them one by one. The kinds: plain
//! allocations of 16, 32 and 64 MiB (a pooled block's sizes to choose among), and of 29,736,576
//! bytes (a 3648 x 2038 `BGRA8Unorm` surface, a resource larger than half of a smaller block)
//! plain and as a buffer's dedicated allocation (`VkMemoryDedicatedAllocateInfo`). Printed
//! (`MAKE-HOLD`), per kind: the allocations' and the frees' spread, the spread of the pings whose
//! commit to completion overlapped an allocation and of those that overlapped no allocation or
//! free, and the allocation time a GiB of growth costs in such makes. Ignored: run it in release
//! under `--cfg m2v_gpu`, `cargo test --release -p metal2vulkan-runtime --lib
//! memory_make_hold_bench -- --ignored --nocapture`.

use crate::{Device, DeviceKind, DeviceOptions, ResourceOptions};
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The allocations of each kind.
const MAKES: usize = 24;
/// The time between two allocations, or two frees, so that each one's effect on the pings stands
/// apart.
const GAP: Duration = Duration::from_millis(40);
/// How often the pinger commits.
const PING: Duration = Duration::from_millis(1);
/// A 3648 x 2038 `BGRA8Unorm` surface's bytes.
const SURFACE_BYTES: u64 = 3648 * 2038 * 4;

/// An allocation the bench makes: its bytes, and whether it is a buffer's dedicated allocation.
#[derive(Clone, Copy)]
struct Kind {
    bytes: u64,
    dedicated: bool,
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.dedicated { "dedicated" } else { "plain" };
        write!(f, "{kind} {} bytes", self.bytes)
    }
}

/// What the bench measured of one kind: each allocation's and each free's span.
struct Spans {
    kind: Kind,
    makes: Vec<Range<Instant>>,
    frees: Vec<Range<Instant>>,
}

/// Sets its flag as it goes, so the pinger stops however the bench ends.
struct Stops<'a>(&'a AtomicBool);

impl Drop for Stops<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A milliseconds summary of `values`: median, 99th percentile and slowest, and how many took
/// 5 ms or more.
fn spread(mut values: Vec<Duration>) -> String {
    if values.is_empty() {
        return "none".to_owned();
    }
    values.sort();
    let ms = |at: usize| values[at.min(values.len() - 1)].as_secs_f64() * 1e3;
    let slow = values.iter().filter(|value| value.as_millis() >= 5).count();
    format!(
        "{} of them: median {:.3} ms, p99 {:.3} ms, slowest {:.3} ms, {slow} of 5 ms or more",
        values.len(),
        ms(values.len() / 2),
        ms(values.len() * 99 / 100),
        ms(values.len() - 1)
    )
}

fn overlaps(a: &Range<Instant>, b: &Range<Instant>) -> bool {
    a.start < b.end && b.start < a.end
}

/// The first memory type of `bits` that is device-local and not host-visible, else the first
/// device-local one: the type a `Private` resource's placement takes first.
fn device_local(properties: &vk::PhysicalDeviceMemoryProperties, bits: u32) -> u32 {
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    let allowed = |index: usize| bits & (1 << index) != 0;
    let local = |index: usize| {
        types[index]
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    };
    let visible = |index: usize| {
        types[index]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    };
    (0..types.len())
        .find(|&index| allowed(index) && local(index) && !visible(index))
        .or_else(|| (0..types.len()).find(|&index| allowed(index) && local(index)))
        .expect("a device-local memory type") as u32
}

/// Allocate and then free [`MAKES`] allocations of `kind` on `device`, [`GAP`] apart.
fn measure(device: &Device, kind: Kind) -> Spans {
    // SAFETY: the bench makes and destroys its own objects on the device while it lives, and the
    // runtime's own objects are not touched.
    let handles = unsafe { device.raw() };
    let raw = handles.device;
    // SAFETY: the physical device is the instance's.
    let properties = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    let mut made = Vec::new();
    let mut makes = Vec::new();
    for _ in 0..MAKES {
        let usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let info = vk::BufferCreateInfo::default()
            .size(kind.bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = match kind.dedicated {
            // SAFETY: `info` is valid; the buffer is destroyed below.
            true => unsafe { raw.create_buffer(&info, None) }.unwrap(),
            false => vk::Buffer::null(),
        };
        let bits = match kind.dedicated {
            // SAFETY: the buffer is live.
            true => unsafe { raw.get_buffer_memory_requirements(buffer) }.memory_type_bits,
            false => u32::MAX,
        };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(kind.bytes)
            .memory_type_index(device_local(&properties, bits));
        if kind.dedicated {
            allocate = allocate.push_next(&mut dedicated);
        }
        let started = Instant::now();
        // SAFETY: `allocate` and its chain live for the call; the type index is the device's.
        let memory = unsafe { raw.allocate_memory(&allocate, None) }.unwrap();
        makes.push(started..Instant::now());
        if kind.dedicated {
            // SAFETY: the memory is the buffer's dedicated allocation, of its size.
            unsafe { raw.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        }
        made.push((buffer, memory));
        std::thread::sleep(GAP);
    }
    let mut frees = Vec::new();
    for (buffer, memory) in made {
        let started = Instant::now();
        // SAFETY: nothing uses the buffer or the memory.
        unsafe {
            if buffer != vk::Buffer::null() {
                raw.destroy_buffer(buffer, None);
            }
            raw.free_memory(memory, None);
        }
        frees.push(started..Instant::now());
        std::thread::sleep(GAP);
    }
    Spans { kind, makes, frees }
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn memory_make_hold_bench() {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let done = AtomicBool::new(false);
    let kinds = [16 << 20, 32 << 20, 64 << 20, SURFACE_BYTES]
        .map(|bytes| Kind {
            bytes,
            dedicated: false,
        })
        .into_iter()
        .chain([Kind {
            bytes: SURFACE_BYTES,
            dedicated: true,
        }]);
    let (pings, measured) = std::thread::scope(|scope| {
        let stops = Stops(&done);
        let pinger = scope.spawn(|| {
            let queue = device.new_command_queue().unwrap();
            let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
            let mut pings = Vec::new();
            while !done.load(Ordering::Acquire) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&word, 0..4, 2).unwrap();
                blit.end_encoding().unwrap();
                let committed = Instant::now();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                pings.push(committed..Instant::now());
                std::thread::sleep(PING);
            }
            pings
        });
        // The pinger's queue and its first pings before the first allocation.
        std::thread::sleep(Duration::from_millis(500));
        let measured: Vec<Spans> = kinds.map(|kind| measure(&device, kind)).collect();
        drop(stops);
        (pinger.join().unwrap(), measured)
    });
    eprintln!("MAKE-HOLD device: {}", device.name());
    let quiet = pings.iter().filter(|ping| {
        !(measured.iter())
            .flat_map(|spans| spans.makes.iter().chain(&spans.frees))
            .any(|span| overlaps(ping, span))
    });
    let quiet = quiet.map(|ping| ping.end - ping.start).collect();
    eprintln!(
        "MAKE-HOLD pings beside no allocation or free: {}",
        spread(quiet)
    );
    for spans in &measured {
        let durations =
            |spans: &[Range<Instant>]| spans.iter().map(|span| span.end - span.start).collect();
        let makes: Vec<Duration> = durations(&spans.makes[..]);
        let mut sorted = makes.clone();
        sorted.sort();
        let per_gib = sorted[sorted.len() / 2].as_secs_f64() * 1e3 * (1u64 << 30) as f64
            / spans.kind.bytes as f64;
        let held = (pings.iter())
            .filter(|ping| spans.makes.iter().any(|make| overlaps(ping, make)))
            .map(|ping| ping.end - ping.start)
            .collect();
        eprintln!("MAKE-HOLD {}: allocations {}", spans.kind, spread(makes));
        eprintln!(
            "MAKE-HOLD {}: pings beside an allocation {}",
            spans.kind,
            spread(held)
        );
        eprintln!(
            "MAKE-HOLD {}: frees {}",
            spans.kind,
            spread(durations(&spans.frees[..]))
        );
        eprintln!(
            "MAKE-HOLD {}: a GiB of growth in such makes allocates for {per_gib:.1} ms at the \
             median",
            spans.kind
        );
    }
}
