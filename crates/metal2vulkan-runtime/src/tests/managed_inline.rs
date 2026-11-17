//! A `Managed` buffer's range that `did_modify_range` names, written into its device copy inline on
//! a device without unified memory (`managed.rs`, `Buffer::upload_inline`): the submission that
//! takes it reads the range from the host copy as it is recorded and writes it by
//! `vkCmdUpdateBuffer`s, where it is at most the bound (`managed::INLINE_BYTES`) of whole words;
//! every other range is copied from the host copy as before. Over the runtime's own host copy and
//! the caller's pages, through their import and alone (the caller's gate refusing the import).
//!
//! - **The bytes.** One command buffer after ranges of 4 bytes, the bound, the bound and a word,
//!   past 65536 bytes, unaligned at either end, and one named twice with the CPU's write between
//!   (the later bytes win) reads every byte the CPU wrote, and the device copy keeps the GPU's
//!   bytes beside each range: no write is widened past what was named. Counted: the aligned ranges
//!   within the bound alone (`WorkCounts::managed_ranges_inlined`).
//! - **Around 65536.** Under a test's bound of 256 KiB, ranges of 65532, 65536, 65540 and 131072
//!   bytes (one, one, two and two `vkCmdUpdateBuffer`s) land whole.
//! - **The negative control.** Under a test's bound of 0 the same steps read the same bytes and
//!   count none.
//! - **Behind a batch still reading the device copy** (`MANAGED-INLINE-WAR`): a kernel spins on
//!   the buffer's first word, 0, loading it until its cap unless it reads another value; while it
//!   runs (its command buffer not complete once the next batch's submission was made), the CPU
//!   writes 1 there and names the word, and the next command buffer's kernel reads 1 at once.
//!   The spinning kernel makes every load of its cap: the upload landed after its reads. Under
//!   the default batch order on a device that orders batches by their ends, the upload takes no
//!   barrier of its own (`WorkCounts::managed_uploads_unbarriered`, the end of the batch before
//!   orders it); under the exact order it keeps its barrier (counted none). It proves the bytes
//!   and their order where a driver runs one queue's submissions side by side; on one that runs
//!   them one after another it cannot fail, as RADV on an RX 7900 XTX ran them (`batch_order.rs`).
//!   The negative control (ignored, run alone) forces the upload unbarriered under the exact order,
//!   whose batches end with no barrier to later commands: it fails only on a driver that runs the
//!   two side by side, and passed on that XTX (the reader made every load of its cap).
//! - **The bench** (ignored, `MANAGED-INLINE`; run in release): one-dispatch command buffers
//!   reading a `Managed` buffer after a `did_modify_range` of 64 B, 4 KiB, 16 KiB and 64 KiB,
//!   written inline or copied, and with no range named: the commit call's CPU time and the
//!   command buffer's GPU time, medians and 90th percentiles.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). Where the
//! device has unified memory (lavapipe's), a `Managed` buffer is one copy and nothing is copied
//! in: each leg says so and passes.

use super::blit_holds::{device, Scratch, WAIT};
use super::host_ram_device::gate;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    HostMemory, ImportRequest, PipelineOptions, ResourceOptions, Size, StorageMode,
};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The bytes of each leg's buffer.
const BYTES: u64 = 512 << 10;

/// The bound of the default device.
const BOUND: u64 = crate::managed::INLINE_BYTES;

/// The buffer's host copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    /// The runtime's own.
    Own,
    /// The caller's pages, through their import.
    Import,
    /// The caller's pages alone: the caller's gate refuses the import.
    Pages,
}

/// Anonymous private pages of the test's own, unmapped when dropped: declared before the device.
struct Pages(NonNull<u8>);

impl Pages {
    fn new() -> Pages {
        Pages(super::host_os::private(BYTES as usize))
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // The pages `new` made, which no buffer lies in any more.
        super::host_os::release(self.0, BYTES as usize);
    }
}

/// `Managed` storage.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A `Managed` buffer of [`BYTES`] whose host copy is `host`, over `pages` where it is the
/// caller's.
fn buffer(device: &Device, host: Host, pages: &Pages) -> Buffer {
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    // SAFETY: the pages outlive every buffer over them (declared before the device).
    let memory = unsafe { HostMemory::new(pages.0) };
    let memory = match host {
        Host::Own => return device.new_buffer(BYTES, managed()).unwrap(),
        Host::Import => memory,
        Host::Pages => memory.with_import_gate(gate(&asked, false)),
    };
    // SAFETY: as above.
    unsafe { device.new_buffer_with_bytes_no_copy(memory, BYTES, managed(), None) }.unwrap()
}

/// Set `range` of `buffer`'s host copy to `byte` and the range's offsets, from the CPU.
fn write(buffer: &Buffer, range: Range<u64>, byte: u8) {
    let contents = buffer.contents().unwrap();
    for at in range {
        // SAFETY: the host copy's bytes, which no command buffer reads now.
        unsafe {
            contents
                .as_ptr()
                .add(at as usize)
                .write(byte ^ (at % 251) as u8)
        };
    }
}

/// The host copy's bytes.
fn host_bytes(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: as in `write`.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) }.to_vec()
}

/// Run `encode`'s blit commands in one command buffer of `queue`, completed.
#[track_caller]
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
}

/// `buffer`'s device copy, as the GPU reads it: the command buffer that reads it takes the ranges
/// named before.
#[track_caller]
fn gpu_bytes(device: &Device, queue: &CommandQueue, buffer: &Buffer) -> Vec<u8> {
    let target = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    blit(queue, |blit| {
        blit.copy_from_buffer_to_buffer(buffer, 0, &target, 0, BYTES)
            .unwrap()
    });
    let contents = target.contents().unwrap();
    // SAFETY: the target's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) }.to_vec()
}

/// The first byte where `seen` and `expected` differ, for a message.
fn first_difference(seen: &[u8], expected: &[u8]) -> Option<(usize, u8, u8)> {
    (seen.iter().zip(expected).enumerate())
        .find(|(_, (seen, expected))| seen != expected)
        .map(|(at, (&seen, &expected))| (at, seen, expected))
}

/// The ranges of [`named_ranges`]' first command buffer, and whether each is written inline under
/// a bound of `bound`: each lies alone, with GPU-written bytes on each side.
fn first_ranges(bound: u64) -> Vec<(Range<u64>, bool)> {
    let at = |start: u64, length: u64| start..start + length;
    let inline = |range: &Range<u64>| {
        let length = range.end - range.start;
        range.start.is_multiple_of(4) && length.is_multiple_of(4) && length <= bound
    };
    [
        at(8, 4),
        at(1 << 10, BOUND),
        at(64 << 10, BOUND + 4),
        at(128 << 10, 65536 + 4),
        at((256 << 10) + 1, 8),
        at((260 << 10) + 4, 9),
        at((264 << 10) + 2, 6),
        at(300 << 10, 64),
    ]
    .into_iter()
    .map(|range| {
        let inline = inline(&range);
        (range, inline)
    })
    .collect()
}

/// The legs over `host` under `bound`, `None` for the default device's (the module docs): the
/// device copy after each command buffer holds every byte named, and the GPU's beside them.
fn named_ranges(host: Host, bound: Option<u64>) {
    let pages = Pages::new();
    let scratch = Scratch::new("managed-inline");
    let device = device(&scratch.0);
    if device.properties().has_unified_memory {
        eprintln!(
            "MANAGED-INLINE {} {host:?}: unified memory, one copy; nothing is copied in",
            device.name()
        );
        return;
    }
    if let Some(bound) = bound {
        (device.shared().managed_inline_bound).store(bound, Ordering::Relaxed);
    }
    let bound = bound.unwrap_or(BOUND);
    let queue = device.new_command_queue().unwrap();
    let buffer = buffer(&device, host, &pages);
    // The GPU's bytes in the device copy, which the host copy does not hold.
    blit(&queue, |blit| {
        blit.fill_buffer(&buffer, 0..BYTES, 0x5a).unwrap()
    });
    let mut expected = vec![0x5a; BYTES as usize];
    let ranges = first_ranges(bound);
    let before = device.work_counts();
    for (range, _) in &ranges {
        write(&buffer, range.clone(), 0x81);
        buffer.did_modify_range(range.clone()).unwrap();
    }
    // The last range named again after another write: the later bytes win.
    let (twice, _) = ranges.last().unwrap().clone();
    write(&buffer, twice.clone(), 0xc3);
    buffer.did_modify_range(twice).unwrap();
    let host_now = host_bytes(&buffer);
    for (range, _) in &ranges {
        let range = range.start as usize..range.end as usize;
        expected[range.clone()].copy_from_slice(&host_now[range]);
    }
    let seen = gpu_bytes(&device, &queue, &buffer);
    let after = device.work_counts();
    let inlined = after.managed_ranges_inlined - before.managed_ranges_inlined;
    let expected_inlined = ranges.iter().filter(|(_, inline)| *inline).count() as u64;
    eprintln!(
        "MANAGED-INLINE {} {host:?}, bound {bound}: {} ranges, {inlined} inlined",
        device.name(),
        ranges.len(),
    );
    assert_eq!(
        first_difference(&seen, &expected),
        None,
        "{host:?}, bound {bound}: the device copy after the first ranges (at, seen, expected)"
    );
    assert_eq!(
        inlined, expected_inlined,
        "{host:?}, bound {bound}: ranges inlined"
    );
    // Around 65536: each range whole, under a bound that holds them.
    if bound >= 256 << 10 {
        let around = [
            (8 << 10)..(8 << 10) + 65532,
            (80 << 10)..(80 << 10) + 65536,
            (160 << 10)..(160 << 10) + 65540,
            (240 << 10)..(240 << 10) + 131072,
        ];
        let before = device.work_counts();
        for range in &around {
            write(&buffer, range.clone(), 0x3c);
            buffer.did_modify_range(range.clone()).unwrap();
            let range = range.start as usize..range.end as usize;
            let host_now = host_bytes(&buffer);
            expected[range.clone()].copy_from_slice(&host_now[range]);
        }
        let seen = gpu_bytes(&device, &queue, &buffer);
        let after = device.work_counts();
        assert_eq!(
            first_difference(&seen, &expected),
            None,
            "{host:?}: the device copy after the ranges around 65536 (at, seen, expected)"
        );
        assert_eq!(
            after.managed_ranges_inlined - before.managed_ranges_inlined,
            around.len() as u64,
            "{host:?}: the ranges around 65536 inlined"
        );
    }
    drop((buffer, queue, device));
}

#[test]
fn ranges_within_the_bound_are_written_inline_over_the_runtime_s_host_copy() {
    named_ranges(Host::Own, None);
}

#[test]
fn ranges_within_the_bound_are_written_inline_over_the_caller_s_import() {
    named_ranges(Host::Import, None);
}

#[test]
fn ranges_within_the_bound_are_written_inline_over_the_caller_s_pages_alone() {
    named_ranges(Host::Pages, None);
}

#[test]
fn ranges_around_65536_are_written_inline_whole() {
    for host in [Host::Own, Host::Import, Host::Pages] {
        named_ranges(host, Some(256 << 10));
    }
}

#[test]
fn no_range_is_written_inline_under_a_bound_of_0() {
    for host in [Host::Own, Host::Import, Host::Pages] {
        named_ranges(host, Some(0));
    }
}

/// The spinning kernel's loads at most: on the RX 7900 XTX and the RTX 2080 well under a second,
/// and far longer than the next batch's commit and submission.
const SPIN_CAP: u32 = 1 << 21;

/// What [`upload_behind_a_reader`] saw.
#[derive(Debug)]
struct Behind {
    /// The loads the spinning kernel made: [`SPIN_CAP`] where it read the old word throughout.
    reader_loads: u32,
    /// The loads the next command buffer's kernel made: 0 where it read the new word.
    next_loads: u32,
    /// The spinning kernel's command buffer had not completed once the next batch's
    /// submission was made.
    overlapped: bool,
    /// The uploads the next batch recorded with no barrier of their own
    /// (`WorkCounts::managed_uploads_unbarriered`).
    unbarriered: u64,
    /// The device orders its batches by their ends (`batch_order.rs`).
    by_ends: bool,
}

/// The spinning kernel (`kernel_spin_on_an_unraised_word`): it loads the word at its slot 0 until
/// it reads a value other than 0 or made as many loads as the cap at its slot 2, and stores how
/// many it made into its slot 1.
fn spin(device: &Device) -> crate::ComputePipelineState {
    let ll = metal2vulkan_devtools::engine_dir().join("kernel_spin_on_an_unraised_word.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let function = library
        .new_function_with_name("spin_on_an_unraised_word")
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// A command buffer of `queue`'s whose one kernel spins on `buffer`'s first word up to `cap`
/// loads, storing how many into `loads`, committed.
fn spinning(
    queue: &CommandQueue,
    state: &crate::ComputePipelineState,
    (buffer, loads): (&Buffer, &Buffer),
    cap: u32,
) -> crate::CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(loads), 0, 1).unwrap();
    encoder.set_bytes(&cap.to_le_bytes(), 2).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// The first word of `loads`, a `Shared` buffer no command writes now.
fn loads_of(loads: &Buffer) -> u32 {
    // SAFETY: a word of the buffer's bytes, aligned, which no command writes now.
    unsafe { loads.contents().unwrap().cast::<u32>().as_ptr().read() }
}

/// The write-after-read steps (the module docs) on a device of the default policy but `exact`'s
/// batch order, its uploads forced unbarriered where `forced`; `None` on a device with unified
/// memory.
fn upload_behind_a_reader(exact: bool, forced: bool) -> Option<Behind> {
    let scratch = Scratch::new("managed-inline-war");
    let mut options = super::blit_holds::options(&scratch.0);
    if exact {
        options.policy.batch_order = metal2vulkan::device::BatchOrderPolicy::Exact;
    }
    let device = Device::system_default_with(&options).unwrap();
    if device.properties().has_unified_memory {
        let name = device.name();
        eprintln!("MANAGED-INLINE-WAR {name}: unified memory, one copy; nothing is copied in");
        return None;
    }
    (device.shared().uploads_unbarriered_forced).store(forced, Ordering::Relaxed);
    let pages = Pages::new();
    let queue = device.new_command_queue().unwrap();
    let state = spin(&device);
    let buffer = buffer(&device, Host::Own, &pages);
    let reader_loads = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let next_loads = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    // Every buffer's first work (a new buffer's fill) done first, so the next batch's first work
    // is the upload alone.
    for loads in [&reader_loads, &next_loads] {
        let warm = spinning(&queue, &state, (&buffer, loads), 1);
        warm.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(warm.status(), CommandBufferStatus::Completed);
    }
    let word = |value: u32| {
        // SAFETY: the host copy's first word, aligned, which no command reads (the GPU reads the
        // device copy).
        unsafe { (buffer.contents().unwrap().cast::<u32>().as_ptr()).write(value) };
        buffer.did_modify_range(0..4).unwrap();
    };
    word(0);
    let reader = spinning(&queue, &state, (&buffer, &reader_loads), SPIN_CAP);
    let deadline = Instant::now() + WAIT;
    while reader.kernel_start_time() == 0.0 {
        assert!(Instant::now() < deadline, "the reader was never submitted");
        std::thread::yield_now();
    }
    word(1);
    let before = device.work_counts();
    let next = spinning(&queue, &state, (&buffer, &next_loads), SPIN_CAP);
    while device.work_counts().queue_submissions == before.queue_submissions {
        assert!(
            Instant::now() < deadline,
            "the next batch was never submitted"
        );
        std::thread::yield_now();
    }
    let overlapped = reader.status() != CommandBufferStatus::Completed;
    for command_buffer in [&reader, &next] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            (command_buffer.status(), command_buffer.error()),
            (CommandBufferStatus::Completed, None)
        );
    }
    let after = device.work_counts();
    let behind = Behind {
        reader_loads: loads_of(&reader_loads),
        next_loads: loads_of(&next_loads),
        overlapped,
        unbarriered: after.managed_uploads_unbarriered - before.managed_uploads_unbarriered,
        by_ends: device.shared().by_ends,
    };
    eprintln!(
        "MANAGED-INLINE-WAR {}: exact order {exact}, forced {forced}: {behind:?}",
        device.name()
    );
    drop((buffer, state, queue, device, pages));
    Some(behind)
}

/// An upload behind a batch still reading the device copy lands after its reads, with no
/// barrier of its own where the device orders its batches by their ends (the module docs): a
/// proof where the driver runs one queue's submissions side by side, none where it does not.
#[test]
fn an_upload_behind_a_batch_reading_the_device_copy_lands_after_its_reads() {
    let Some(behind) = upload_behind_a_reader(false, false) else {
        return;
    };
    assert!(behind.overlapped, "the reader completed first: {behind:?}");
    assert_eq!(
        (behind.reader_loads, behind.next_loads),
        (SPIN_CAP, 0),
        "the reader read the old word throughout, the next batch the new one: {behind:?}"
    );
    assert_eq!(
        behind.unbarriered,
        u64::from(behind.by_ends),
        "the upload unbarriered where batches are ordered by their ends: {behind:?}"
    );
}

/// Under the exact batch order the upload keeps its barrier, with the same bytes.
#[test]
fn under_the_exact_batch_order_an_upload_keeps_its_barrier() {
    let Some(behind) = upload_behind_a_reader(true, false) else {
        return;
    };
    assert!(behind.overlapped, "the reader completed first: {behind:?}");
    assert_eq!((behind.reader_loads, behind.next_loads), (SPIN_CAP, 0));
    assert_eq!(behind.unbarriered, 0, "{behind:?}");
}

/// The negative control (the module docs): the upload forced unbarriered under the exact batch
/// order, whose batches end with no barrier to later commands. It fails where the driver runs the
/// next batch beside the reader (the reader then reads the new word and stops early); a driver
/// that runs one queue's submissions one after another passes it.
#[test]
#[ignore = "a negative control, run alone"]
fn negative_control_an_unbarriered_upload_under_the_exact_order() {
    let Some(behind) = upload_behind_a_reader(true, true) else {
        return;
    };
    assert_eq!(behind.unbarriered, 1, "{behind:?}");
    assert_eq!((behind.reader_loads, behind.next_loads), (SPIN_CAP, 0));
}

/// The 50th and 90th percentiles of `values`, nearest rank.
fn percentiles(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[((values.len() * p).div_ceil(100)).clamp(1, values.len()) - 1];
    [at(50), at(90)]
}

/// One-dispatch command buffers after a `did_modify_range` of each size, inline or copied, and
/// with none named (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn managed_inline_bench() {
    const FRAMES: usize = 200;
    const WARM: usize = 20;
    let pages = Pages::new();
    let scratch = Scratch::new("managed-inline-bench");
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    if device.properties().has_unified_memory {
        eprintln!("MANAGED-INLINE {name}: unified memory, one copy; nothing to measure");
        return;
    }
    let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let output = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    for host in [Host::Own, Host::Import] {
        let buffer = buffer(&device, host, &pages);
        for size in [64u64, 4 << 10, 16 << 10, 64 << 10] {
            for (mode, bound) in [("none", None), ("copy", Some(0)), ("inline", Some(1 << 20))] {
                if let Some(bound) = bound {
                    (device.shared().managed_inline_bound).store(bound, Ordering::Relaxed);
                }
                let before = device.work_counts();
                let (mut commits, mut gpus) = (Vec::new(), Vec::new());
                for frame in 0..FRAMES + WARM {
                    let value = (frame % 200) as f32;
                    let contents = buffer.contents().unwrap().cast::<f32>();
                    // SAFETY: the host copy's first floats, which no command buffer in flight
                    // reads (each frame's was waited for).
                    let floats = unsafe {
                        std::slice::from_raw_parts_mut(contents.as_ptr(), (size / 4) as usize)
                    };
                    floats.fill(value);
                    if bound.is_some() {
                        buffer.did_modify_range(0..size).unwrap();
                    }
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut encoder = command_buffer.compute_command_encoder().unwrap();
                    encoder.set_compute_pipeline_state(&state).unwrap();
                    encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
                    encoder.set_buffer(Some(&output), 0, 1).unwrap();
                    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
                    encoder
                        .dispatch_threads(Size::new(16, 1, 1), Size::new(16, 1, 1))
                        .unwrap();
                    encoder.end_encoding().unwrap();
                    let began = Instant::now();
                    command_buffer.commit().unwrap();
                    let commit = began.elapsed().as_secs_f64() * 1e6;
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                    let read = output.contents().unwrap().cast::<f32>();
                    // SAFETY: the first float of a `Shared` buffer the completed command buffer
                    // wrote: the kernel's first float times 1, plus its index, 0.
                    let read = unsafe { read.as_ptr().read() };
                    if bound.is_some() {
                        assert_eq!(read, value, "{host:?} {size} {mode}: frame {frame}");
                    }
                    let gpu = (command_buffer.gpu_end_time().unwrap()
                        - command_buffer.gpu_start_time().unwrap())
                        * 1e6;
                    if frame >= WARM {
                        commits.push(commit);
                        gpus.push(gpu);
                    }
                }
                let after = device.work_counts();
                let [commit50, commit90] = percentiles(commits);
                let [gpu50, gpu90] = percentiles(gpus);
                eprintln!(
                    "MANAGED-INLINE {name} {host:?} {size} B {mode}: n {FRAMES} commit_us p50 \
                     {commit50:.1} p90 {commit90:.1} gpu_us p50 {gpu50:.1} p90 {gpu90:.1}, \
                     inlined {}, unbarriered {}",
                    after.managed_ranges_inlined - before.managed_ranges_inlined,
                    after.managed_uploads_unbarriered - before.managed_uploads_unbarriered,
                );
            }
        }
    }
    drop((output, queue, state, device, pages));
}
