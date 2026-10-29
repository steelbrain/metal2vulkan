//! Copies of rows between buffers on the device the cfg selects (the runtime's `row_copies.rs`):
//! a blit between a buffer and a texture over a buffer, between two textures over buffers, and a
//! buffer-texture copy whose layout goes through a staging buffer, each of 64 rows whose rows lie
//! apart, is one compute dispatch that writes Metal's bytes where the device runs it faster
//! (`DriverFact::RowCopiesDispatchFaster`), and a copy's regions writing the same bytes elsewhere;
//! a copy of rows whose length is no whole number of 4-byte words, and one of a single row, stays
//! a copy's regions everywhere.
//!
//! `rows_between_buffers_copy_metals_bytes` checks every byte of each buffer and texture under the
//! validation layer: the rows a blit moves land at their places, and every byte between the rows,
//! before the first and past the last keeps its own. `WorkCounts::row_copies_dispatched` shows
//! which form each copy took. `a_shadow_s_staged_rows_go_back_as_one_dispatch` clears a texture
//! of 16-byte texels over a no-copy buffer 4 bytes past a page, whose shadow goes back through a
//! staging buffer, 32 rows as one dispatch where the device runs it faster, and checks every byte
//! of its pages.
//!
//! `row_copies_cost` (ignored; run it with `--ignored --nocapture`) prints `ROW-COPIES` lines: a
//! frame of one blit of `rows` rows of 1920 `BGRA8Unorm` texels from a tight buffer into a texture
//! over a buffer whose rows lie 8192 bytes apart, for 1080 rows down to 2, timed over
//! [`TIMED`] frames after [`WARM`]: the CPU time a frame of the scheduler's thread (from its
//! `schedstat`), of the encoding thread, which records it, the command buffer's GPU time
//! (`GPUEndTime - GPUStartTime`), and the commands recorded.
//!
//! `long_copies_of_any_size_copy_metals_bytes` checks every byte of blits between buffers of one
//! region each, of sizes a driver may record slowly as one region, which the runtime records as
//! two (its `device/functions.rs`).
//!
//! `span_copy_cost` (ignored, likewise) prints `SPAN-COPY` lines of those costs: a frame of one
//! blit of [`SPANS`]' sizes of bytes between two buffers, one region each (a frame's size as a
//! texture's 1920x1080 or 1920x1088 `BGRA8Unorm` texels, multiples of 4 KiB by a prime, 16 bytes
//! times a prime, primes over and under 16 KiB, and the most bytes under 4 KiB a split leaves to
//! one), with the regions the copies split, then a frame of a kernel's read of one texel of a
//! `YCBCR8_420_2P` texture whose planes span [`SURFACE_SPAN`] bytes, which converts through its
//! refresh: the first frame converts every texel, and each later one compares the unchanged
//! planes with its staging buffer's copy and copies none of their bytes.

use super::host_memory::{no_copy, on_device, Mapping, GUEST_PAGE};
use super::{
    buffer_with, layer_reported_nothing_at_all, loader, open_device, read_buffer, validated_device,
    ycbcr_textures, Messages, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, Device,
    DeviceOptions, IoSurfacePlane, IoSurfaceYCbCr, LoadAction, Origin, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, ShaderCacheOptions, Size, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The buffers' length.
const LEN: usize = 1 << 17;

/// Byte `i` of a source buffer.
fn source_byte(i: usize) -> u8 {
    (7 * i + 3) as u8
}

/// Byte `i` of a destination buffer, before a copy.
fn destination_byte(i: usize) -> u8 {
    ((5 * i + 1) as u8) ^ 0xa5
}

/// A texture of `pf`, `[w, h]` texels, over `buffer` at `offset`, its rows `bpr` apart.
fn over(buffer: &Buffer, pf: PixelFormat, (offset, bpr): (u64, u64), [w, h]: [u64; 2]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
    d.resource_options = buffer.resource_options();
    buffer.new_texture(&d, offset, bpr).unwrap()
}

/// An ordinary shared texture of `pf`, `[w, h]` texels.
fn plain(device: &Device, pf: PixelFormat, [w, h]: [u64; 2]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
    d.resource_options = ResourceOptions::shared();
    device.new_texture(&d).unwrap()
}

/// Encode one blit encoder by `encode` in a command buffer of its own, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// `bytes` with `rows` rows of `len` bytes from `from` (the `y`th at `from.0 + y · from.1`)
/// written at `to` (the `y`th at `to.0 + y · to.1`).
fn moved(
    mut bytes: Vec<u8>,
    source: &[u8],
    (from, from_apart): (usize, usize),
    (to, to_apart): (usize, usize),
    (rows, len): (usize, usize),
) -> Vec<u8> {
    for y in 0..rows {
        let (s, d) = (from + y * from_apart, to + y * to_apart);
        bytes[d..d + len].copy_from_slice(&source[s..s + len]);
    }
    bytes
}

/// The row copies `device` dispatches of each copy whose rows take a dispatch: 1 where it runs the
/// dispatch faster (`DriverFact::RowCopiesDispatchFaster`), else 0.
pub(super) fn dispatches(device: &Device) -> u64 {
    u64::from(
        device
            .properties()
            .driver_facts
            .contains(&DriverFact::RowCopiesDispatchFaster),
    )
}

/// The dispatched row copies and the row copies `device` counted since `before`.
pub(super) fn copies_since(device: &Device, before: &WorkCounts) -> (u64, u64) {
    let now = device.work_counts();
    (
        now.row_copies_dispatched - before.row_copies_dispatched,
        now.row_copies - before.row_copies,
    )
}

/// The probe's cases on `device` (the module docs), each checked byte for byte, with the form
/// each copy took.
fn row_copies(device: &Device) {
    let source_bytes: Vec<u8> = (0..LEN).map(source_byte).collect();
    let destination_bytes: Vec<u8> = (0..LEN).map(destination_byte).collect();
    let fresh = || {
        (
            buffer_with(device, &source_bytes).unwrap(),
            buffer_with(device, &destination_bytes).unwrap(),
        )
    };
    let read = |buffer: &Buffer| read_buffer(buffer, 0, LEN as u64).unwrap();
    let dispatched = dispatches(device);
    let rgba = PixelFormat::RGBA8Unorm;
    let [w, h] = [48u64, 64];
    let (row, rows) = (4 * w as usize, h as usize);

    // A tight buffer into a texture over a buffer whose rows lie 256 bytes apart.
    let (source, destination) = fresh();
    let texture = over(&destination, rgba, (256, 256), [w, h]);
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(w, h, 1);
        e.copy_from_buffer_to_texture(
            &source,
            128,
            4 * w,
            4 * w * h,
            size,
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &source_bytes,
        (128, row),
        (256, 256),
        (rows, row),
    );
    assert_eq!(
        read(&destination),
        want,
        "a buffer into a texture over a buffer"
    );
    assert_eq!(read(&source), source_bytes, "the source buffer");
    assert_eq!(
        copies_since(device, &before),
        (dispatched, 1),
        "one copy, dispatched where faster"
    );

    // A texture over a buffer (rows 320 bytes apart) into a tight buffer.
    let (source, destination) = fresh();
    let texture = over(&source, rgba, (512, 320), [w, h]);
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(w, h, 1);
        e.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            size,
            &destination,
            64,
            4 * w,
            4 * w * h,
            BlitOption::None,
        )
        .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &source_bytes,
        (512, 320),
        (64, row),
        (rows, row),
    );
    assert_eq!(
        read(&destination),
        want,
        "a texture over a buffer into a buffer"
    );
    assert_eq!(
        copies_since(device, &before),
        (dispatched, 1),
        "one copy, dispatched where faster"
    );

    // Between two textures over buffers, rows 320 and 256 bytes apart.
    let (source, destination) = fresh();
    let from = over(&source, rgba, (512, 320), [w, h]);
    let to = over(&destination, rgba, (256, 256), [w, h]);
    let before = device.work_counts();
    blit(device, |e| {
        let at = Origin::new(0, 0, 0);
        e.copy_from_texture_to_texture(&from, 0, 0, at, Size::new(w, h, 1), &to, 0, 0, at)
            .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &source_bytes,
        (512, 320),
        (256, 256),
        (rows, row),
    );
    assert_eq!(read(&destination), want, "two textures over buffers");
    assert_eq!(
        copies_since(device, &before),
        (dispatched, 1),
        "one copy, dispatched where faster"
    );

    // Rows of 37 bytes, no whole number of words: the copy's regions.
    let (source, destination) = fresh();
    let r8 = over(&destination, PixelFormat::R8Unorm, (256, 64), [37, h]);
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(37, h, 1);
        e.copy_from_buffer_to_texture(
            &source,
            3,
            37,
            37 * h,
            size,
            &r8,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &source_bytes,
        (3, 37),
        (256, 64),
        (rows, 37),
    );
    assert_eq!(read(&destination), want, "rows of 37 bytes");
    assert_eq!(
        copies_since(device, &before),
        (0, 1),
        "one copy, its regions"
    );
    // Its place in the runtime counted the call and its rows (`WorkCounts::copy_sites`).
    let site = |counts: &WorkCounts| {
        let site = counts.copy_sites.iter().find(|site| {
            site.command == "vkCmdCopyBuffer" && site.file.ends_with("src/row_copies.rs")
        });
        site.map_or((0, 0, 0), |site| (site.calls, site.regions, site.most))
    };
    let ((calls, regions, most), (calls_before, regions_before, _)) =
        (site(&device.work_counts()), site(&before));
    assert_eq!(
        (calls - calls_before, regions - regions_before),
        (1, h),
        "the copy's place"
    );
    assert!(most >= h, "the copy's place: at most {most} regions");

    // One row: the copy's region.
    let (source, destination) = fresh();
    let texture = over(&destination, rgba, (256, 256), [w, 1]);
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(w, 1, 1);
        e.copy_from_buffer_to_texture(
            &source,
            128,
            4 * w,
            4 * w,
            size,
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &source_bytes,
        (128, row),
        (256, 256),
        (1, row),
    );
    assert_eq!(read(&destination), want, "one row");
    assert_eq!(
        copies_since(device, &before),
        (0, 1),
        "one copy, its regions"
    );

    // A layout off Metal's rules into and out of an ordinary `RGBA16Float` texture: its rows at
    // byte 4 onward, 392 bytes apart (rounded down to 4 bytes, where plain Metal reads and writes
    // them), through a staging buffer.
    let half = PixelFormat::RGBA16Float;
    let (source, destination) = fresh();
    let texture = plain(device, half, [w, h]);
    let half_row = 8 * w as usize;
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(w, h, 1);
        e.copy_from_buffer_to_texture(
            &source,
            4,
            392,
            392 * h,
            size,
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    });
    let mut texels = vec![0; half_row * rows];
    texture
        .get_bytes(
            &mut texels,
            half_row as u64,
            0,
            Region::new_2d(0, 0, w, h),
            0,
            0,
        )
        .unwrap();
    let want = moved(
        vec![0; half_row * rows],
        &source_bytes,
        (4, 392),
        (0, half_row),
        (rows, half_row),
    );
    assert_eq!(texels, want, "a staged copy into a texture");
    assert_eq!(
        copies_since(device, &before),
        (dispatched, 1),
        "its rows, dispatched where faster"
    );
    let before = device.work_counts();
    blit(device, |e| {
        let size = Size::new(w, h, 1);
        e.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            size,
            &destination,
            4,
            392,
            392 * h,
            BlitOption::None,
        )
        .unwrap();
    });
    let want = moved(
        destination_bytes.clone(),
        &texels,
        (0, half_row),
        (4, 392),
        (rows, half_row),
    );
    assert_eq!(read(&destination), want, "a staged copy out of a texture");
    assert_eq!(
        copies_since(device, &before),
        (dispatched, 1),
        "its rows, dispatched where faster"
    );
}

#[test]
fn rows_between_buffers_copy_metals_bytes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        row_copies(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Blits between buffers of one region each whose sizes a driver may record slowly as one region
/// (ANV): 8,294,411 bytes (a prime) from the start of a buffer into the start of another, and
/// 65,521 bytes (a prime) from byte 7 into byte 13. Each writes Metal's bytes, every other byte
/// of the destinations keeps its own, and each region is recorded as two
/// (`WorkCounts::copy_regions_split`, the runtime's `device/functions.rs`).
#[test]
fn long_copies_of_any_size_copy_metals_bytes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        for (size, from, to) in [(8_294_411usize, 0usize, 0usize), (65_521, 7, 13)] {
            let len = size + 64;
            let source_bytes: Vec<u8> = (0..len).map(source_byte).collect();
            let destination_bytes: Vec<u8> = (0..len).map(destination_byte).collect();
            let source = buffer_with(&device, &source_bytes).unwrap();
            let destination = buffer_with(&device, &destination_bytes).unwrap();
            let before = device.work_counts().copy_regions_split;
            blit(&device, |e| {
                e.copy_from_buffer_to_buffer(
                    &source,
                    from as u64,
                    &destination,
                    to as u64,
                    size as u64,
                )
                .unwrap();
            });
            let split = device.work_counts().copy_regions_split - before;
            let want = moved(
                destination_bytes,
                &source_bytes,
                (from, 0),
                (to, 0),
                (1, size),
            );
            let got = read_buffer(&destination, 0, len as u64).unwrap();
            assert!(got == want, "a copy of {size} bytes: not Metal's bytes");
            assert_eq!(split, 1, "a copy of {size} bytes: its region split");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A texture of 16-byte texels over a no-copy buffer 4 bytes past a page, which Vulkan copies
/// only at multiples of 16 bytes: a pass that clears it writes its shadow, whose 32 rows go back
/// out of a staging buffer as one dispatch where the device runs it faster
/// (`DriverFact::RowCopiesDispatchFaster`), else as the copy's regions. The rows hold the clear,
/// and every byte between them and around the buffer keeps its own.
#[test]
fn a_shadow_s_staged_rows_go_back_as_one_dispatch() {
    on_device(|device| {
        let (at, [w, h], bpr) = (4usize, [8u64, 32], 256usize);
        let len = h as usize * bpr;
        let pages = Mapping::anonymous(2 * GUEST_PAGE);
        pages.fill(0x3c);
        let before: Vec<u8> = (0..len).map(destination_byte).collect();
        pages.write(at, &before);
        let pointer = std::ptr::NonNull::new(pages.pointer().as_ptr().wrapping_add(at)).unwrap();
        // SAFETY: the pages outlive the buffer and the texture, dropped first.
        let buffer =
            unsafe { no_copy(device, pointer, len, ResourceOptions::default(), None) }.unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, w, h, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, bpr as u64).unwrap();
        let counts = device.work_counts();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop((texture, buffer));
        let texel: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        let row: Vec<u8> = texel.repeat(w as usize);
        let mut want = vec![0x3c; 2 * GUEST_PAGE];
        want[at..at + len].copy_from_slice(&before);
        for y in 0..h as usize {
            let start = at + y * bpr;
            want[start..start + row.len()].copy_from_slice(&row);
        }
        assert_eq!(pages.read(0..2 * GUEST_PAGE), want, "the pages");
        let (dispatched, copies) = copies_since(device, &counts);
        assert_eq!(
            (dispatched, copies),
            (dispatches(device), 1),
            "the copy back's rows, dispatched where faster"
        );
    });
}

/// The untimed and the timed frames of each row count.
const WARM: usize = 20;
const TIMED: usize = 100;

/// The CPU time so far of the process's threads whose names start with `prefix` (Linux; zero
/// elsewhere).
fn threads_cpu(prefix: &str) -> Duration {
    let mut all = Duration::ZERO;
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return all;
    };
    for task in tasks.flatten() {
        let task = task.path();
        let (Ok(name), Ok(schedstat)) = (
            std::fs::read_to_string(task.join("comm")),
            std::fs::read_to_string(task.join("schedstat")),
        ) else {
            continue;
        };
        let nanos = schedstat
            .split_whitespace()
            .next()
            .and_then(|n| n.parse().ok());
        if let (true, Some(nanos)) = (name.starts_with(prefix), nanos) {
            all += Duration::from_nanos(nanos);
        }
    }
    all
}

/// The calling thread's CPU time so far.
fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec it is given and has no other effect.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// The costs a frame of `frame` on `device`, timed over [`TIMED`] frames after [`WARM`]: the CPU
/// time of the scheduler's thread and of the encoding thread, which records it, and the command
/// buffer's GPU time, as words; with `device`'s work counts before and
/// after the timed frames.
fn frame_costs(
    device: &Device,
    frame: impl Fn() -> CommandBuffer,
) -> (String, WorkCounts, WorkCounts) {
    for _ in 0..WARM {
        frame().wait_until_completed_for(WAIT).unwrap();
    }
    let counts = device.work_counts();
    let scheduler = threads_cpu("metal2vulkan-sc");
    let mut encoding = Duration::ZERO;
    let mut gpu = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        let start = thread_cpu();
        let command_buffer = frame();
        encoding += thread_cpu() - start;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let (begin, end) = (
            command_buffer.gpu_start_time().unwrap(),
            command_buffer.gpu_end_time().unwrap(),
        );
        gpu.push((end - begin) * 1e6);
    }
    let after = device.work_counts();
    let per = |d: Duration| d.as_secs_f64() * 1e3 / TIMED as f64;
    gpu.sort_by(f64::total_cmp);
    let costs = format!(
        "scheduler {:.3} ms, encoding {:.3} ms a frame; GPU {:.1} us (p90 {:.1})",
        per(threads_cpu("metal2vulkan-sc") - scheduler),
        per(encoding),
        gpu[TIMED / 2],
        gpu[TIMED * 9 / 10],
    );
    (costs, counts, after)
}

/// A count a frame of the timed frames, from its count before them and after.
fn each(after: u64, before: u64) -> f64 {
    (after - before) as f64 / TIMED as f64
}

#[test]
#[ignore = "a measurement: run it with --ignored --nocapture"]
fn row_copies_cost() {
    let cache = ScratchCache::new().unwrap();
    let mut options = DeviceOptions::default();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    let device = open_device(TEST_DEVICE, options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let (width, pitch) = (1920u64, 8192u64);
    for rows in [1080u64, 270, 64, 16, 8, 4, 2] {
        let source = device
            .new_buffer(4 * width * rows, ResourceOptions::shared())
            .unwrap();
        let destination = device
            .new_buffer(pitch * rows, ResourceOptions::shared())
            .unwrap();
        let texture = over(
            &destination,
            PixelFormat::BGRA8Unorm,
            (0, pitch),
            [width, rows],
        );
        let frame = || {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(width, rows, 1);
            encoder
                .copy_from_buffer_to_texture(
                    &source,
                    0,
                    4 * width,
                    4 * width * rows,
                    size,
                    &texture,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        };
        let (costs, counts, after) = frame_costs(&device, frame);
        eprintln!(
            "ROW-COPIES {}: {rows} rows: {costs}; a frame: {:.2} dispatched of {:.2} row copies, \
             {:.2} vkCmdCopyBuffer, {:.2} vkCmdDispatch replayed",
            device.name(),
            each(after.row_copies_dispatched, counts.row_copies_dispatched),
            each(after.row_copies, counts.row_copies),
            each(
                after.commands_replayed.cmd_copy_buffer,
                counts.commands_replayed.cmd_copy_buffer
            ),
            each(
                after.commands_replayed.cmd_dispatch,
                counts.commands_replayed.cmd_dispatch
            ),
        );
    }
}

/// The sizes `span_copy_cost` copies as one region (the module docs).
const SPANS: [u64; 11] = [
    8_294_400,
    8_355_840,
    2027 * 4096,
    4093 * 4096,
    8_294_576,
    SURFACE_SPAN,
    8_294_411,
    65_521,
    16_381,
    4_095,
    4_093,
];

/// The span of the bytes of `span_copy_cost`'s surface's planes: a 1920x1080 `420v` surface, its
/// luma rows 1920 bytes apart at byte 0, its chroma rows too from [`SURFACE_CHROMA_AT`]; 16 bytes
/// times 194,413, a prime.
const SURFACE_SPAN: u64 = SURFACE_CHROMA_AT + 1920 * 539 + 1920;
const SURFACE_CHROMA_AT: u64 = 2_073_808;

#[test]
#[ignore = "a measurement: run it with --ignored --nocapture"]
fn span_copy_cost() {
    let cache = ScratchCache::new().unwrap();
    let mut options = DeviceOptions::default();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    let device = open_device(TEST_DEVICE, options).unwrap();
    let queue = device.new_command_queue().unwrap();
    for size in SPANS {
        let source = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        let destination = device.new_buffer(size, ResourceOptions::shared()).unwrap();
        let frame = || {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encoder
                .copy_from_buffer_to_buffer(&source, 0, &destination, 0, size)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        };
        let (costs, counts, after) = frame_costs(&device, frame);
        eprintln!(
            "SPAN-COPY {}: a copy of {size} bytes as one region: {costs}; a frame: {:.2} \
             vkCmdCopyBuffer replayed, {:.2} regions split",
            device.name(),
            each(
                after.commands_replayed.cmd_copy_buffer,
                counts.commands_replayed.cmd_copy_buffer
            ),
            each(after.copy_regions_split, counts.copy_regions_split),
        );
    }
    let surface = device
        .new_buffer(SURFACE_SPAN, ResourceOptions::shared())
        .unwrap();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, 1920, 1080, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let facts = IoSurfaceYCbCr::new(
        u32::from_be_bytes(*b"420v"),
        0,
        [1920, 1080],
        SURFACE_CHROMA_AT,
        1920,
    );
    let plane = IoSurfacePlane::new(0, 1920, true).with_ycbcr(facts);
    let texture = surface.new_texture_with_iosurface_plane(&d, plane).unwrap();
    let library = device
        .new_library_with_air(&llvm_assemble_text(ycbcr_textures::KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name("read_texels").unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let out = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    // A kernel's read of one texel, after the texture's conversion through its refresh.
    let frame = || {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_texture(Some(&texture), 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder
            .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    let (costs, counts, after) = frame_costs(&device, frame);
    eprintln!(
        "SPAN-COPY {}: a read of a YCbCr texture whose planes span {SURFACE_SPAN} bytes: {costs}; \
         a frame: {:.2} vkCmdCopyBuffer, {:.2} vkCmdDispatch replayed",
        device.name(),
        each(
            after.commands_replayed.cmd_copy_buffer,
            counts.commands_replayed.cmd_copy_buffer
        ),
        each(
            after.commands_replayed.cmd_dispatch,
            counts.commands_replayed.cmd_dispatch
        ),
    );
}
