//! What one kernel submission costs against the imports a device holds, on the device the cfg
//! selects, without the validation layer: report only, in release builds.
//!
//! RADV lists every imported `VkDeviceMemory` of the device in every command submission's BO list:
//! Mesa 26.2.4's `radv_alloc_memory` makes every memory but WSI's resident, unconditionally (26.0.8
//! wherever the kernel has always-valid BOs), and its amdgpu winsys leaves out of the global list
//! only local BOs, memory neither imported nor exported, which the kernel tracks per VM
//! (`radv_amdgpu_winsys_bo_make_resident`); each submission's list is the whole global list
//! (`radv_amdgpu_get_bo_list`). amdgpu's work per submission is per entry (Linux 7.0
//! `amdgpu_cs.c`: lookup and sort, lock, validate, VM update, fence sync and fence add), and a
//! userptr entry also faults its whole range in again. ANV on i915 lists every `VkDeviceMemory`
//! (`submit_costs`). The runtime's own memory is pooled in 64-MiB blocks, which RADV makes local.
//!
//! The lane holds `n` imports live, for each `n` of [`COUNTS`], of two kinds, each
//! [`IMPORT_BYTES`] over pages of its own: dma-bufs (udmabufs of a sealed `memfd`'s pages, as
//! QEMU's guest RAM, which RADV imports as dma-bufs where it refuses their pointer) and the
//! pointers of private anonymous pages (amdgpu userptrs on RADV). It then commits [`COMMITS`]
//! command buffers of one 4-KiB fill one at a time, each waited for, after [`WARM`] untimed;
//! for dma-bufs twice, at once with every import resident (as before the device released idle
//! imports), then once the device's residency thread released them after their first idle window
//! (`host_residency.rs` of the runtime), as many as the descriptors the process keeps allow. Printed as `SUBMIT-PER-IMPORT` lines, per kind, `n` and
//! residency: the imports the device counts (`WorkCounts::dma_buf_imports` made,
//! `WorkCounts::dma_buf_memories_live`, `WorkCounts::host_imports_live`,
//! `WorkCounts::host_imports_resident`) and its pooled blocks; a command buffer's time
//! from its kernelStartTime to its kernelEndTime (the scheduler's submission of it to Vulkan), its
//! 50th percentile and mean; the scheduler thread's CPU time per commit; and, where the
//! kernel-submission counter (`scripts/i915-execbuffers/counter.c`) is preloaded, the kernel
//! submissions per commit, the objects each lists (dma-bufs and userptrs among them) and the
//! microseconds each spent in the ioctl. Its slope over `n` is the kernel's cost per entry.
//!
//! A second measurement holds a guest's shape (`submit_per_import_guest_shape`): [`GUEST_IMPORTS`]
//! pointer imports, of which each frame's blits read [`GUEST_USED`], in frames of
//! [`GUEST_PER_FRAME`] command buffers. It measures frames at once, every import resident, then,
//! where the driver's pointer imports follow their address
//! (`DriverFact::PointerImportsFollowTheirAddress`), again once the device's residency thread
//! released the idle ones while paced frames kept the read ones in use. Printed as
//! `SUBMIT-GUEST-SHAPE` lines: the imports (`WorkCounts::host_imports_live`,
//! `host_imports_resident`, `host_imports_released`, `host_imports_reimported`), a frame's wall
//! time, the scheduler thread's CPU time a frame and, where the kernel-submission counter is
//! preloaded, the kernel submissions a frame, the objects and userptrs each lists and the
//! microseconds a frame spent in the ioctl.
//!
//! A third holds the guest's shape and releases [`RELEASES_PER_FRAME`] no-copy buffers a frame
//! midway through committing it, each over pages of its own (`submit_per_import_release_cost`):
//! the time each release takes on the committing thread, alternating two shapes. With a keep-alive
//! that unmaps the pages, as the consumer's guest releases its transfer buffers, the free goes to a
//! handler thread, the release returning first; with a deallocator that unmaps them and no
//! keep-alive, the free and the deallocator run in the release, as Metal runs the deallocator.
//! Printed as `IMPORT-RELEASE-COST` lines: the release's 50th and 90th percentiles and its time a
//! frame, a frame's wall time, and the releases posted.
//!
//! A fourth holds buffers of the runtime's own (`submit_per_fresh_page_buffer`): for each phase
//! of [`FRESH_PHASES`], that many `Device::new_buffer`s of that length and storage, longer than half
//! a pooled block, which lie in fresh pages imported by their pointer where the device takes them
//! (`fresh_pages.rs` of the runtime), then a `Private` control of the same lengths. It commits
//! [`COMMITS`] command buffers of one 4-KiB fill one at a time, each waited for, and
//! [`GUEST_FRAMES`] frames of [`GUEST_PER_FRAME`] fills. Printed as `SUBMIT-FRESH-PAGES` lines: the
//! fresh-page buffers the device made (`WorkCounts::fresh_page_buffers_made`, `fresh_page_bytes`),
//! its host imports and pooled blocks live, a command buffer's time from its kernelStartTime to its
//! kernelEndTime (50th percentile), a frame's wall time and the scheduler thread's CPU time a
//! commit and a frame.
//!
//! Ignored by default: measurements, not checks. Run them by name with `--run-ignored all`; on a
//! host without `/dev/udmabuf` access the dma-buf rows say so and are skipped.

use super::commit_costs::percentiles;
use super::commit_latency::named_cpu_us;
use super::dma_buf_imports::{no_copy as dma_buf_no_copy, udmabuf_openable, Calls, Memfd};
use super::host_memory::{no_copy, Mapping};
use super::submit_costs::Execbuffers;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions,
};
use metal2vulkan_runtime::{BufferContents, BufferDescriptor, Deallocator, HostMemory, KeepAlive};
use std::time::{Duration, Instant};

/// The live imports of each phase.
const COUNTS: [usize; 5] = [0, 100, 200, 400, 1000];
/// The bytes of each import: one 64-KiB chunk, RADV 26.2.4's sparse granule, so one dma-buf.
const IMPORT_BYTES: usize = 64 << 10;
/// Command buffers committed untimed first.
const WARM: usize = 30;
/// Command buffers measured.
const COMMITS: usize = 300;
/// The longest the probe waits for the residency thread to release the imports it may.
const RELEASED: Duration = Duration::from_secs(30);
/// The linux name (15 bytes) of the device's scheduler thread.
const SCHEDULER: &str = "metal2vulkan-sc";
/// The guest's shape (a macOS guest's desktop on the Iris Xe of RPL-P, its counts of 2026-10-17):
/// its live host imports, those its frames' blits read, and its command buffers a frame.
const GUEST_IMPORTS: usize = 520;
const GUEST_USED: usize = 35;
const GUEST_PER_FRAME: usize = 43;
/// The no-copy buffers released a frame by `submit_per_import_release_cost`.
const RELEASES_PER_FRAME: usize = 4;
/// The guest-shape frames measured, after [`GUEST_WARM`] untimed, each committed in a row and
/// waited for at its end.
const GUEST_FRAMES: usize = 120;
const GUEST_WARM: usize = 10;
/// The frame period while the device's residency thread releases the idle imports.
const GUEST_PERIOD: Duration = Duration::from_micros(16_667);

/// The phases of `submit_per_fresh_page_buffer`: a name, how many buffers, their length and their
/// storage.
const FRESH_PHASES: [(&str, usize, usize, bool); 7] = [
    ("none", 0, 0, true),
    ("1 Shared of 64 MiB", 1, 64 << 20, true),
    ("4 Shared of 64 MiB", 4, 64 << 20, true),
    ("1 Shared of 256 MiB", 1, 256 << 20, true),
    ("4 Shared of 256 MiB", 4, 256 << 20, true),
    ("4 Private of 256 MiB", 4, 256 << 20, false),
    ("none again", 0, 0, true),
];

/// The kind of import a phase holds.
#[derive(Clone, Copy, Debug)]
enum Kind {
    DmaBuf,
    Pointer,
}

/// A phase's live imports: the buffers, then the memory they lie over.
enum Held {
    DmaBuf {
        _buffers: Vec<Buffer>,
        _memfd: Memfd,
    },
    Pointer {
        _buffers: Vec<Buffer>,
        _pages: Mapping,
    },
}

/// `n` live imports of `kind` on `device`, each over pages of its own.
fn hold(device: &Device, kind: Kind, n: usize) -> Held {
    let len = (n * IMPORT_BYTES).max(IMPORT_BYTES);
    match kind {
        Kind::DmaBuf => {
            let memfd = Memfd::new(len);
            let calls = Calls::default();
            let source = memfd.source(&memfd.whole, &calls);
            let buffers = (0..n)
                .map(|i| {
                    let at = i * IMPORT_BYTES;
                    dma_buf_no_copy(device, &memfd.whole, at, IMPORT_BYTES, source.clone()).unwrap()
                })
                .collect();
            Held::DmaBuf {
                _buffers: buffers,
                _memfd: memfd,
            }
        }
        Kind::Pointer => {
            let pages = Mapping::anonymous(len);
            pages.fill(0x5a);
            let buffers = (0..n)
                .map(|i| {
                    // SAFETY: the pages are this process's, mapped read and write, and outlive
                    // the buffers (`Held`'s field order).
                    let at = unsafe { pages.pointer().add(i * IMPORT_BYTES) };
                    // SAFETY: as above.
                    unsafe { no_copy(device, at, IMPORT_BYTES, ResourceOptions::shared(), None) }
                        .unwrap()
                })
                .collect();
            Held::Pointer {
                _buffers: buffers,
                _pages: pages,
            }
        }
    }
}

/// A command buffer of `queue` filling `target`'s 4 KiB with `value`, not committed.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Commit `command_buffer` and wait for it: the seconds from its kernelStartTime to its
/// kernelEndTime.
fn submitted(command_buffer: CommandBuffer) -> f64 {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    command_buffer.kernel_end_time() - command_buffer.kernel_start_time()
}

/// The phase of `n` imports of `kind`: held, measured and printed. Dma-bufs twice: first at
/// once, every import resident; then again once the device's residency thread released those it
/// may (after their first idle window, as many as the descriptors the process keeps allow).
fn phase(device: &Device, queue: &CommandQueue, target: &Buffer, kind: Kind, n: usize) {
    let made = device.work_counts().dma_buf_imports;
    let held = hold(device, kind, n);
    measure(device, queue, target, (kind, n, "at once"), made);
    if matches!(kind, Kind::DmaBuf) {
        let start = Instant::now();
        while device.work_counts().host_imports_resident > 0 {
            assert!(start.elapsed() < RELEASED, "imports still resident");
            std::thread::sleep(Duration::from_millis(50));
        }
        let waited = format!("released after {:.1} s", start.elapsed().as_secs_f64());
        measure(device, queue, target, (kind, n, &waited), made);
    }
    drop(held);
}

/// Commit and time command buffers beside `kind`'s `n` imports in `residency`, and print what
/// they took; `made` the dma-buf chunks the device imported before the imports.
fn measure(
    device: &Device,
    queue: &CommandQueue,
    target: &Buffer,
    (kind, n, residency): (Kind, usize, &str),
    made: u64,
) {
    for value in 0..WARM {
        submitted(fill(queue, target, value as u8));
    }
    let counts = device.work_counts();
    let execbuffers = Execbuffers::read();
    let scheduler = named_cpu_us(SCHEDULER);
    let mut times: Vec<f64> = (0..COMMITS)
        .map(|value| submitted(fill(queue, target, value as u8)))
        .collect();
    let scheduler = (named_cpu_us(SCHEDULER) - scheduler) / COMMITS as f64;
    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2] * 1e6;
    let mean = times.iter().sum::<f64>() / times.len() as f64 * 1e6;
    let kernel = Execbuffers::read()
        .zip(execbuffers)
        .map(|(after, before)| {
            let submissions = after.execbuffers - before.execbuffers;
            let each = |after: u64, before: u64| (after - before) as f64 / submissions as f64;
            format!(
                "; {:.2} kernel submissions a commit, {:.0} objects each ({:.0} dma-bufs, {:.0} \
                 userptrs), {:.1} us each in the ioctl",
                submissions as f64 / COMMITS as f64,
                each(after.objects, before.objects),
                each(after.prime, before.prime),
                each(after.userptr, before.userptr),
                each(after.nanos, before.nanos) / 1e3,
            )
        })
        .unwrap_or_default();
    eprintln!(
        "SUBMIT-PER-IMPORT {}: {kind:?} {n} {residency}: {} dma-buf chunks imported ({} live), \
         {} host imports live ({} releasable resident), {} pooled blocks live; submit p50 \
         {p50:.1} us, mean {mean:.1} us; scheduler thread {scheduler:.1} us a commit{kernel}",
        device.name(),
        counts.dma_buf_imports - made,
        counts.dma_buf_memories_live,
        counts.host_imports_live,
        counts.host_imports_resident,
        counts.memory_blocks_live,
    );
}

/// The phases, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_import() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let dma_bufs = udmabuf_openable("submit per import");
    for kind in [Kind::DmaBuf, Kind::Pointer] {
        if matches!(kind, Kind::DmaBuf) && !dma_bufs {
            continue;
        }
        for n in COUNTS {
            phase(&device, &queue, &target, kind, n);
        }
    }
}

/// One guest-shape frame: [`GUEST_PER_FRAME`] command buffers committed in a row, the first of
/// which each copy 4 KiB out of one of `used` into `target`, the rest each fill `target`'s 4 KiB;
/// waited for at its end.
fn guest_frame(queue: &CommandQueue, target: &Buffer, used: &[Buffer]) {
    let mut last = None;
    for i in 0..GUEST_PER_FRAME {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        match used.get(i) {
            Some(import) => blit
                .copy_from_buffer_to_buffer(import, 0, target, 0, 4096)
                .unwrap(),
            None => blit.fill_buffer(target, 0..4096, i as u8).unwrap(),
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        last = Some(command_buffer);
    }
    let last = last.expect("a frame of command buffers");
    last.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(last.status(), CommandBufferStatus::Completed);
}

/// Guest-shape frames measured and printed as `phase`.
fn guest_measure(
    device: &Device,
    queue: &CommandQueue,
    target: &Buffer,
    used: &[Buffer],
    phase: &str,
) {
    for _ in 0..GUEST_WARM {
        guest_frame(queue, target, used);
    }
    let counts = device.work_counts();
    let execbuffers = Execbuffers::read();
    let scheduler = named_cpu_us(SCHEDULER);
    let times: Vec<f64> = (0..GUEST_FRAMES)
        .map(|_| {
            let start = Instant::now();
            guest_frame(queue, target, used);
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    let scheduler = (named_cpu_us(SCHEDULER) - scheduler) / GUEST_FRAMES as f64;
    let after = device.work_counts();
    let [p10, p50, p90] = percentiles(times);
    let kernel = Execbuffers::read()
        .zip(execbuffers)
        .map(|(after, before)| {
            let submissions = (after.execbuffers - before.execbuffers) as f64;
            let each = |after: u64, before: u64| (after - before) as f64 / submissions;
            format!(
                "; {:.1} kernel submissions a frame, {:.0} objects each ({:.0} userptrs), {:.0} us \
                 a frame in the ioctl ({:.1} us each)",
                submissions / GUEST_FRAMES as f64,
                each(after.objects, before.objects),
                each(after.userptr, before.userptr),
                (after.nanos - before.nanos) as f64 / 1e3 / GUEST_FRAMES as f64,
                each(after.nanos, before.nanos) / 1e3,
            )
        })
        .unwrap_or_default();
    eprintln!(
        "SUBMIT-GUEST-SHAPE {}: {phase}: {} host imports live, {} to {} releasable resident, {} \
         released and {} imported again in all; frame p50 {p50:.0} us (p10 {p10:.0}, p90 \
         {p90:.0}); scheduler thread {scheduler:.1} us a frame{kernel}",
        device.name(),
        after.host_imports_live,
        counts.host_imports_resident,
        after.host_imports_resident,
        after.host_imports_released,
        after.host_imports_reimported,
    );
}

/// The guest's shape, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_import_guest_shape() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let pages = Mapping::anonymous(GUEST_IMPORTS * IMPORT_BYTES);
    pages.fill(0x5a);
    let buffers: Vec<Buffer> = (0..GUEST_IMPORTS)
        .map(|i| {
            // SAFETY: the pages are this process's, mapped read and write, and outlive the
            // buffers (dropped first).
            let at = unsafe { pages.pointer().add(i * IMPORT_BYTES) };
            // SAFETY: as above.
            unsafe { no_copy(&device, at, IMPORT_BYTES, ResourceOptions::shared(), None) }.unwrap()
        })
        .collect();
    let used = &buffers[..GUEST_USED];
    guest_measure(&device, &queue, &target, used, "at once");
    let follows =
        (device.properties().driver_facts).contains(&DriverFact::PointerImportsFollowTheirAddress);
    if !follows {
        eprintln!(
            "SUBMIT-GUEST-SHAPE {}: pointer imports kept (the driver is not measured to have them \
             follow their address)",
            device.name()
        );
        return;
    }
    let start = Instant::now();
    while device.work_counts().host_imports_resident > GUEST_USED as u64 {
        assert!(start.elapsed() < RELEASED, "idle imports still resident");
        let frame = Instant::now();
        guest_frame(&queue, &target, used);
        std::thread::sleep(GUEST_PERIOD.saturating_sub(frame.elapsed()));
    }
    let waited = format!(
        "released after {:.1} s of paced frames",
        start.elapsed().as_secs_f64()
    );
    guest_measure(&device, &queue, &target, used, &waited);
    drop(buffers);
}

/// A no-copy buffer over [`IMPORT_BYTES`] of pages of its own, which its keep-alive unmaps where
/// `kept`, else its deallocator.
fn going_no_copy(device: &Device, kept: bool) -> Buffer {
    let mapping = Mapping::anonymous(IMPORT_BYTES);
    mapping.fill(0x3c);
    let pointer = mapping.pointer();
    // SAFETY: the keep-alive, or the deallocator, unmaps the pages once the device's import of
    // them is freed.
    let memory = unsafe { HostMemory::new(pointer) };
    let contents = match kept {
        true => BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: Some(Box::new(mapping) as KeepAlive),
        },
        false => BufferContents::NoCopy {
            memory,
            deallocator: Some(Box::new(move |_, _| drop(mapping)) as Deallocator),
            keep_alive: None,
        },
    };
    let descriptor =
        BufferDescriptor::new(IMPORT_BYTES as u64, ResourceOptions::shared(), contents);
    device.new_buffer_with_descriptor(descriptor).unwrap()
}

/// A guest-shape frame that releases `going` once half its command buffers are committed, each
/// release's time pushed to `releases`.
fn releasing_frame(
    queue: &CommandQueue,
    target: &Buffer,
    used: &[Buffer],
    going: Vec<Buffer>,
    releases: &mut Vec<f64>,
) {
    let mut going = Some(going);
    let mut last = None;
    for i in 0..GUEST_PER_FRAME {
        if i == GUEST_PER_FRAME / 2 {
            for buffer in going.take().into_iter().flatten() {
                let start = Instant::now();
                drop(buffer);
                releases.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        match used.get(i) {
            Some(import) => blit
                .copy_from_buffer_to_buffer(import, 0, target, 0, 4096)
                .unwrap(),
            None => blit.fill_buffer(target, 0..4096, i as u8).unwrap(),
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        last = Some(command_buffer);
    }
    let last = last.expect("a frame of command buffers");
    last.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(last.status(), CommandBufferStatus::Completed);
}

/// The release of no-copy buffers in the guest's shape, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_import_release_cost() {
    for kept in [true, false].repeat(2) {
        let shape = match kept {
            true => "a keep-alive",
            false => "a deallocator",
        };
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            // A consumer's device keeps no execution record.
            options.record_executions = false;
        })
        .unwrap();
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let pages = Mapping::anonymous(GUEST_IMPORTS * IMPORT_BYTES);
        pages.fill(0x5a);
        let buffers: Vec<Buffer> = (0..GUEST_IMPORTS)
            .map(|i| {
                // SAFETY: the pages are this process's, mapped read and write, and outlive the
                // buffers (dropped first).
                let at = unsafe { pages.pointer().add(i * IMPORT_BYTES) };
                // SAFETY: as above.
                unsafe { no_copy(&device, at, IMPORT_BYTES, ResourceOptions::shared(), None) }
                    .unwrap()
            })
            .collect();
        let used = &buffers[..GUEST_USED];
        let mut releases = Vec::new();
        let mut frames = Vec::new();
        let before = device.work_counts();
        for frame in 0..GUEST_WARM + GUEST_FRAMES {
            let going = (0..RELEASES_PER_FRAME)
                .map(|_| going_no_copy(&device, kept))
                .collect();
            let start = Instant::now();
            let mut taken = Vec::new();
            releasing_frame(&queue, &target, used, going, &mut taken);
            if frame >= GUEST_WARM {
                frames.push(start.elapsed().as_secs_f64() * 1e6);
                releases.extend(taken);
            }
        }
        let start = Instant::now();
        while device.work_counts().host_import_releases_pending > 0 {
            assert!(start.elapsed() < WAIT, "posted releases never ran");
            std::thread::yield_now();
        }
        let after = device.work_counts();
        let per_frame = releases.iter().sum::<f64>() / GUEST_FRAMES as f64;
        let [_, r50, r90] = percentiles(releases);
        let [_, f50, f90] = percentiles(frames);
        eprintln!(
            "IMPORT-RELEASE-COST {}: {shape}: {} host imports live; a release p50 \
             {r50:.1} us (p90 {r90:.1}), {per_frame:.0} us a frame on the committing thread; frame \
             p50 {f50:.0} us (p90 {f90:.0}); {} releases posted",
            device.name(),
            after.host_imports_live,
            after.host_import_releases_posted - before.host_import_releases_posted,
        );
        drop(buffers);
    }
}

/// Buffers of the runtime's own, held as each phase of [`FRESH_PHASES`] says, and what one commit
/// and one frame of fills cost beside them, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_per_fresh_page_buffer() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    for (name, count, length, shared) in FRESH_PHASES {
        let options = match shared {
            true => ResourceOptions::shared(),
            false => ResourceOptions::private(),
        };
        let before = device.work_counts();
        let held: Vec<Buffer> = (0..count)
            .map(|_| device.new_buffer(length as u64, options).unwrap())
            .collect();
        for value in 0..WARM {
            submitted(fill(&queue, &target, value as u8));
        }
        let scheduler = named_cpu_us(SCHEDULER);
        let times: Vec<f64> = (0..COMMITS)
            .map(|value| submitted(fill(&queue, &target, value as u8)) * 1e6)
            .collect();
        let per_commit = (named_cpu_us(SCHEDULER) - scheduler) / COMMITS as f64;
        let [_, c50, c90] = percentiles(times);
        for _ in 0..GUEST_WARM {
            guest_frame(&queue, &target, &[]);
        }
        let scheduler = named_cpu_us(SCHEDULER);
        let frames: Vec<f64> = (0..GUEST_FRAMES)
            .map(|_| {
                let start = Instant::now();
                guest_frame(&queue, &target, &[]);
                start.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        let per_frame = (named_cpu_us(SCHEDULER) - scheduler) / GUEST_FRAMES as f64;
        let [_, f50, f90] = percentiles(frames);
        let after = device.work_counts();
        eprintln!(
            "SUBMIT-FRESH-PAGES {}: {name}: {} fresh-page buffers made ({} MiB), {} host imports \
             live, {} pooled blocks live; commit p50 {c50:.1} us (p90 {c90:.1}), scheduler thread \
             {per_commit:.1} us a commit; frame of {GUEST_PER_FRAME} p50 {f50:.0} us (p90 \
             {f90:.0}), scheduler thread {per_frame:.0} us a frame",
            device.name(),
            after.fresh_page_buffers_made - before.fresh_page_buffers_made,
            (after.fresh_page_bytes - before.fresh_page_bytes) >> 20,
            after.host_imports_live,
            after.memory_blocks_live,
        );
        drop(held);
    }
}
