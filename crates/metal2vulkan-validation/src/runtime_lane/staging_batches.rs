//! Bursts of host calls on `Shared` and `Managed` color textures, as a guest process makes them at
//! its start (1776 `replaceRegion`s back to back, each into a small texture of its own), on
//! whichever path the device copies through: on the host (`texture/host_copy.rs` in the runtime),
//! or on its staging queue (`texture/staging_copy.rs`), where a burst's copies are queued and
//! carried by one submission there. Metal's contract holds either way, and each test checks it:
//!
//! - the call copies the caller's bytes before it returns: the caller may change or free them;
//! - every later command buffer's GPU use of the texture sees them, through the texture or a view
//!   of it;
//! - every CPU read sees them, through the texture or a view;
//! - overlapping writes into one texture keep their order.
//!
//! And bursts of new textures, which read zero on Metal: every read of a fresh texture, GPU or CPU,
//! reads zero, and a first write into one lands after its zeros.
//!
//! Where the device copies on its staging queue, the counts show the batching
//! (`WorkCounts::staging_moves_queued`, `staging_copies_queued`, `staging_flushes`,
//! `staging_gpu_waits`): no submission during a burst, one that carries it all, waited for on the
//! GPU by the first batch on each `VkQueue` after it that the host submits before it saw the copy
//! complete.
//! Each test prints the path it saw (`STAGING-BATCH` lines).

use super::host_memory::on_device;
use super::read_buffer;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CpuCacheMode, Device, HazardTrackingMode, HostBytes, Origin,
    Region, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// The staging queue's bound on the copies queued for one submission (`STAGED_COPIES`).
const STAGED_COPIES: usize = 4096;

fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A 2D texture of `format`, `side` texels square, sampled, of `options`.
fn texture(device: &Device, format: PixelFormat, side: u64, options: ResourceOptions) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    device.new_texture(&descriptor).unwrap()
}

/// `side`² 4-byte texels: byte `i` is `i * 7 + seed` modulo 251.
fn texels(side: u64, seed: u64) -> Vec<u8> {
    (0..4 * side * side)
        .map(|i| ((i * 7 + seed) % 251) as u8)
        .collect()
}

/// The whole first level of a texture `side` square, read back by `get_bytes`.
fn read_back(texture: &Texture, side: u64) -> Vec<u8> {
    let mut bytes = vec![0xee; (4 * side * side) as usize];
    let region = Region::new_2d(0, 0, side, side);
    texture
        .get_bytes(&mut bytes, 4 * side, 0, region, 0, 0)
        .unwrap();
    bytes
}

/// A buffer `command_buffer` fills with `texture`'s first level by a blit, `side` square.
fn copy_out(command_buffer: &CommandBuffer, texture: &Texture, side: u64) -> Buffer {
    let device = command_buffer.device();
    let out = device
        .new_buffer(4 * side * side, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &out,
        0,
        4 * side,
        0,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    out
}

/// The counts that tell the path apart: host copies, images started on the host, staging
/// submissions, first moves and copies queued for the staging queue, staging submissions that
/// carried them, GPU waits for them, internal submissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Path {
    host: u64,
    started: u64,
    staging: u64,
    moves: u64,
    queued: u64,
    flushes: u64,
    gpu_waits: u64,
    internal: u64,
}

impl Path {
    fn between(before: &WorkCounts, after: &WorkCounts) -> Path {
        Path {
            host: after.host_texel_copies - before.host_texel_copies,
            started: after.images_started_on_the_host - before.images_started_on_the_host,
            staging: after.staging_submissions - before.staging_submissions,
            moves: after.staging_moves_queued - before.staging_moves_queued,
            queued: after.staging_copies_queued - before.staging_copies_queued,
            flushes: after.staging_flushes - before.staging_flushes,
            gpu_waits: after.staging_gpu_waits - before.staging_gpu_waits,
            internal: after.internal_submissions - before.internal_submissions,
        }
    }

    /// The path a burst of `calls` writes took, said.
    fn said(&self, calls: usize) -> &'static str {
        match (self.host as usize == calls, self.queued as usize == calls) {
            (true, _) => "copied on the host",
            (false, true) => "queued for the staging queue",
            _ => "copied through submissions",
        }
    }
}

/// 1776 `replace_region`s back to back, each into a `Managed` texture of its own of
/// `RGBA8Unorm` or `BGRA8Unorm`, 16 to 256 texels square, from the caller's bytes at a pointer
/// (`replace_region_with_host_bytes`, the consumer's call): a later command buffer's blits read
/// every texture's bytes, and so does `get_bytes`. Where the device copies on its staging queue,
/// the burst made no submission, and the command buffer's batch made one there that carried all
/// 1776, which it waited for on the GPU.
#[test]
fn a_burst_of_replace_regions_shares_one_staging_submission() {
    const CALLS: usize = 1776;
    const SHAPES: [(PixelFormat, u64); 5] = [
        (PixelFormat::RGBA8Unorm, 16),
        (PixelFormat::BGRA8Unorm, 32),
        (PixelFormat::RGBA8Unorm, 64),
        (PixelFormat::BGRA8Unorm, 128),
        (PixelFormat::RGBA8Unorm, 256),
    ];
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let textures: Vec<(Texture, u64)> = (0..CALLS)
            .map(|i| {
                let (format, side) = SHAPES[i % SHAPES.len()];
                (texture(device, format, side, managed()), side)
            })
            .collect();
        let written: Vec<Vec<u8>> = (0..CALLS)
            .map(|i| texels(textures[i].1, i as u64))
            .collect();
        let before = device.work_counts();
        for ((texture, side), bytes) in textures.iter().zip(&written) {
            let region = Region::new_2d(0, 0, *side, *side);
            let host = HostBytes::from_slice(bytes);
            (texture.replace_region_with_host_bytes(region, 0, 0, host, 4 * side, 0)).unwrap();
        }
        let burst = Path::between(&before, &device.work_counts());
        // Every texture read by blits of one command buffer committed after the calls.
        let command_buffer = queue.command_buffer().unwrap();
        let outs: Vec<Buffer> = (textures.iter())
            .map(|(texture, side)| copy_out(&command_buffer, texture, *side))
            .collect();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}: the blits");
        let read = Path::between(&before, &device.work_counts());
        for (i, (out, bytes)) in outs.iter().zip(&written).enumerate() {
            assert!(
                read_buffer(out, 0, bytes.len() as u64).unwrap() == *bytes,
                "{name}: the blit of texture {i} reads its replace_region's bytes"
            );
        }
        // And `get_bytes` of each.
        for (i, ((texture, side), bytes)) in textures.iter().zip(&written).enumerate() {
            assert!(
                read_back(texture, *side) == *bytes,
                "{name}: get_bytes of texture {i} reads its replace_region's bytes"
            );
        }
        println!(
            "STAGING-BATCH {name}: {CALLS} replace_regions {}: the burst {burst:?}, with the \
             command buffer that read them {read:?}",
            burst.said(CALLS)
        );
        if burst.queued > 0 {
            assert_eq!(
                (burst.queued as usize, burst.staging, burst.internal),
                (CALLS, 0, 0),
                "{name}: a burst queued for the staging queue makes no submission"
            );
            assert_eq!(
                (read.staging, read.flushes),
                (1, 1),
                "{name}: the command buffer's batch carried the burst in one staging submission"
            );
            staging_waits_bounded(&name, &read);
        }
    });
}

/// One `replace_region` of a 64 MiB `Managed` texture, then `BATCHES` command buffers committed
/// back to back on one queue, each a blit of a 64-texel square of it of its own into a buffer:
/// every blit reads the call's bytes. Where the device copies on its staging queue, the first
/// batch carried the copy, and waited for it on the GPU unless the host saw it complete by then
/// (a loaded host can stall the batch's submission past the copy's end); the batches after it on
/// the queue, committed while the copy may still run, took no wait of their own: they follow that
/// wait in the `VkQueue`'s submission order, which orders them after the copy
/// (`command_buffer/submit.rs`). So the staging waits are at most one per staging submission, not
/// one per batch the host submitted before it saw the copy complete. The runtime's
/// `tests::staging_waits` holds the staging submission on the GPU until every batch was
/// submitted, so that the wait and the batches that follow it are seen whatever the host's
/// timing.
#[test]
fn batches_after_one_that_waited_for_a_staging_copy_follow_its_wait() {
    const SIDE: u64 = 4096;
    const SQUARE: u64 = 64;
    const BATCHES: u64 = 32;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let texture = texture(device, PixelFormat::RGBA8Unorm, SIDE, managed());
        let written = texels(SIDE, 3);
        let before = device.work_counts();
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        (texture.replace_region(region, 0, 0, &written, 4 * SIDE, 0)).unwrap();
        let burst = Path::between(&before, &device.work_counts());
        let squares: Vec<(u64, u64)> = (0..BATCHES)
            .map(|at| (SQUARE * (at % 8), SQUARE * (at / 8)))
            .collect();
        let mut last = None;
        let outs: Vec<Buffer> = (squares.iter())
            .map(|&(x, y)| {
                let command_buffer = queue.command_buffer().unwrap();
                let out = device
                    .new_buffer(4 * SQUARE * SQUARE, ResourceOptions::shared())
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_texture_to_buffer(
                    &texture,
                    0,
                    0,
                    Origin::new(x, y, 0),
                    Size::new(SQUARE, SQUARE, 1),
                    &out,
                    0,
                    4 * SQUARE,
                    0,
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                last = Some(command_buffer);
                out
            })
            .collect();
        let last = last.unwrap();
        last.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(last.error(), None, "{name}: the blits");
        let read = Path::between(&before, &device.work_counts());
        for (at, (out, &(x, y))) in outs.iter().zip(&squares).enumerate() {
            let want: Vec<u8> = (0..SQUARE)
                .flat_map(|row| {
                    let from = (4 * ((y + row) * SIDE + x)) as usize;
                    written[from..from + (4 * SQUARE) as usize].iter().copied()
                })
                .collect();
            assert!(
                read_buffer(out, 0, want.len() as u64).unwrap() == want,
                "{name}: batch {at}'s blit reads the replace_region's bytes"
            );
        }
        println!(
            "STAGING-BATCH {name}: one replace_region of {} MiB {}, then {BATCHES} batches on one \
             queue that read it: the call {burst:?}, with the batches {read:?}",
            (4 * SIDE * SIDE) >> 20,
            burst.said(1)
        );
        if burst.queued > 0 {
            assert_eq!(
                (read.staging, read.flushes),
                (1, 1),
                "{name}: the first batch carried the copy in one staging submission"
            );
            assert!(
                read.gpu_waits <= 1 + read.internal,
                "{name}: at most one batch on the queue waited for the staging submission on the \
                 GPU, and the batches after it none (internal submissions wait for it \
                 themselves): {} waits",
                read.gpu_waits
            );
            if read.gpu_waits == 0 {
                println!(
                    "STAGING-BATCH {name}: the host saw the staging submission complete before \
                     the first batch was submitted; no batch waited for it"
                );
            }
        }
    });
}

/// Overlapping writes into one texture keep their order, whether read by `get_bytes` or by a later
/// command buffer, through the texture or a view of it: the whole level written twice, then a
/// part of it, then a part through a view; the reads see the last whole write with the two parts
/// over it.
#[test]
fn overlapping_queued_writes_keep_their_order() {
    const SIDE: u64 = 64;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        for options in [ResourceOptions::shared(), managed()] {
            let mode = options.storage_mode();
            let texture = texture(device, PixelFormat::RGBA8Unorm, SIDE, options);
            let view = texture.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
            let whole = Region::new_2d(0, 0, SIDE, SIDE);
            let (first, second) = (texels(SIDE, 1), texels(SIDE, 2));
            let part = Region::new_2d(8, 8, 32, 32);
            let part_bytes = texels(32, 3);
            let through_view = Region::new_2d(24, 24, 32, 32);
            let view_bytes = texels(32, 4);
            let mut wanted = second.clone();
            for (region, bytes) in [(part, &part_bytes), (through_view, &view_bytes)] {
                for y in 0..32 {
                    let at = (4 * ((region.origin.y + y) * SIDE + region.origin.x)) as usize;
                    let from = (4 * y * 32) as usize;
                    wanted[at..at + 128].copy_from_slice(&bytes[from..from + 128]);
                }
            }
            let before = device.work_counts();
            texture
                .replace_region(whole, 0, 0, &first, 4 * SIDE, 0)
                .unwrap();
            texture
                .replace_region(whole, 0, 0, &second, 4 * SIDE, 0)
                .unwrap();
            texture
                .replace_region(part, 0, 0, &part_bytes, 128, 0)
                .unwrap();
            view.replace_region(through_view, 0, 0, &view_bytes, 128, 0)
                .unwrap();
            let writes = Path::between(&before, &device.work_counts());
            // A command buffer's blits of the texture and the view.
            let command_buffer = queue.command_buffer().unwrap();
            let (out, view_out) = (
                copy_out(&command_buffer, &texture, SIDE),
                copy_out(&command_buffer, &view, SIDE),
            );
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{mode:?} on {name}");
            let bytes = 4 * SIDE * SIDE;
            assert!(
                read_buffer(&out, 0, bytes).unwrap() == wanted,
                "{mode:?} on {name}: a blit of the texture reads the writes in their order"
            );
            assert!(
                read_buffer(&view_out, 0, bytes).unwrap() == wanted,
                "{mode:?} on {name}: a blit of the view reads the writes in their order"
            );
            // And the CPU, with the writes made again first, read before any batch.
            texture
                .replace_region(whole, 0, 0, &first, 4 * SIDE, 0)
                .unwrap();
            texture
                .replace_region(whole, 0, 0, &second, 4 * SIDE, 0)
                .unwrap();
            view.replace_region(part, 0, 0, &part_bytes, 128, 0)
                .unwrap();
            texture
                .replace_region(through_view, 0, 0, &view_bytes, 128, 0)
                .unwrap();
            let queued = device.work_counts();
            assert!(
                read_back(&view, SIDE) == wanted,
                "{mode:?} on {name}: get_bytes of the view reads the writes in their order"
            );
            let read = Path::between(&queued, &device.work_counts());
            assert!(
                read_back(&texture, SIDE) == wanted,
                "{mode:?} on {name}: get_bytes of the texture reads the writes in their order"
            );
            println!(
                "STAGING-BATCH {name}: {mode:?} overlapping writes {}: {writes:?}, the first \
                 read after more {read:?}",
                writes.said(4)
            );
            if writes.queued > 0 {
                assert_eq!(
                    (writes.queued, writes.staging),
                    (4, 0),
                    "{mode:?} on {name}"
                );
                assert_eq!(
                    (read.staging, read.flushes),
                    (1, 1),
                    "{mode:?} on {name}: the read carried the four writes queued before it"
                );
            }
        }
    });
}

/// The call copies the caller's bytes before it returns: bytes changed and freed right after the
/// call, before anything reads the texture, leave what the call was given.
#[test]
fn a_queued_write_holds_the_caller_s_bytes_as_they_were() {
    const SIDE: u64 = 128;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let (gpu_read, cpu_read) = (
            texture(device, PixelFormat::BGRA8Unorm, SIDE, managed()),
            texture(device, PixelFormat::BGRA8Unorm, SIDE, managed()),
        );
        let wanted = texels(SIDE, 9);
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        for texture in [&gpu_read, &cpu_read] {
            let mut bytes = wanted.clone();
            let host = HostBytes::from_slice(&bytes);
            (texture.replace_region_with_host_bytes(region, 0, 0, host, 4 * SIDE, 0)).unwrap();
            bytes.fill(0x5a);
            drop(bytes);
        }
        let command_buffer = queue.command_buffer().unwrap();
        let out = copy_out(&command_buffer, &gpu_read, SIDE);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
        assert!(
            read_buffer(&out, 0, wanted.len() as u64).unwrap() == wanted,
            "{name}: a later blit reads the bytes as the call was given them"
        );
        assert!(
            read_back(&cpu_read, SIDE) == wanted,
            "{name}: get_bytes reads the bytes as the call was given them"
        );
    });
}

/// Past the staging queue's bound on queued copies, the call that queues past it makes their
/// submission, waited for: a burst of `STAGED_COPIES + 4` writes into one texel makes one there
/// before anything reads it, and the read sees the last write.
#[test]
fn queued_writes_past_the_bound_are_made_by_the_call() {
    const CALLS: usize = STAGED_COPIES + 4;
    on_device(|device| {
        let name = device.name();
        let texture = texture(device, PixelFormat::RGBA8Unorm, 4, managed());
        let texel = Region::new_2d(1, 2, 1, 1);
        let before = device.work_counts();
        for i in 0..CALLS {
            let bytes = (i as u32).to_le_bytes();
            texture.replace_region(texel, 0, 0, &bytes, 4, 0).unwrap();
        }
        let burst = Path::between(&before, &device.work_counts());
        let mut read = [0; 4];
        texture.get_bytes(&mut read, 4, 0, texel, 0, 0).unwrap();
        assert_eq!(
            u32::from_le_bytes(read),
            CALLS as u32 - 1,
            "{name}: the last write"
        );
        println!(
            "STAGING-BATCH {name}: {CALLS} writes into one texel {}: {burst:?}",
            burst.said(CALLS)
        );
        if burst.queued > 0 {
            assert_eq!(
                (burst.staging, burst.flushes),
                (1, 1),
                "{name}: the call past the bound made one staging submission"
            );
        }
    });
}

/// A texture let go of with its write still queued takes the write with it: the next command
/// buffer's batch submits nothing for it on the staging queue, and runs.
#[test]
fn a_texture_gone_takes_its_queued_write() {
    const SIDE: u64 = 32;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let kept = texture(device, PixelFormat::RGBA8Unorm, SIDE, managed());
        let gone = texture(device, PixelFormat::RGBA8Unorm, SIDE, managed());
        let whole = Region::new_2d(0, 0, SIDE, SIDE);
        // Settle the kept texture's own writes first.
        kept.replace_region(whole, 0, 0, &texels(SIDE, 5), 4 * SIDE, 0)
            .unwrap();
        assert!(read_back(&kept, SIDE) == texels(SIDE, 5), "{name}");
        let before = device.work_counts();
        gone.replace_region(whole, 0, 0, &texels(SIDE, 6), 4 * SIDE, 0)
            .unwrap();
        drop(gone);
        let command_buffer = queue.command_buffer().unwrap();
        let out = copy_out(&command_buffer, &kept, SIDE);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
        assert!(
            read_buffer(&out, 0, 4 * SIDE * SIDE).unwrap() == texels(SIDE, 5),
            "{name}: the kept texture"
        );
        let path = Path::between(&before, &device.work_counts());
        println!("STAGING-BATCH {name}: a texture gone with its write queued: {path:?}");
        if path.queued > 0 {
            assert_eq!(
                (path.staging, path.flushes),
                (0, 0),
                "{name}: nothing submitted for a texture gone"
            );
        }
    });
}

/// A texture whose `VkImage` was handed out (`Texture::vk_image`), which the caller may use on
/// its own queues unseen: the hand-out makes the copies queued before it, waited for, and each
/// later write into that texture is made by its own call, waited for.
#[test]
fn a_handed_out_texture_s_writes_are_made_by_their_calls() {
    const SIDE: u64 = 32;
    on_device(|device| {
        let name = device.name();
        let texture = texture(device, PixelFormat::RGBA8Unorm, SIDE, managed());
        let whole = Region::new_2d(0, 0, SIDE, SIDE);
        let before = device.work_counts();
        texture
            .replace_region(whole, 0, 0, &texels(SIDE, 7), 4 * SIDE, 0)
            .unwrap();
        // SAFETY: the handle is only compared with null; the texture outlives it.
        let image = unsafe { texture.vk_image() };
        let handed = Path::between(&before, &device.work_counts());
        let later = device.work_counts();
        texture
            .replace_region(whole, 0, 0, &texels(SIDE, 8), 4 * SIDE, 0)
            .unwrap();
        let after = Path::between(&later, &device.work_counts());
        assert!(read_back(&texture, SIDE) == texels(SIDE, 8), "{name}");
        println!(
            "STAGING-BATCH {name}: a write before a hand-out ({}) {handed:?}, after it {after:?}",
            match image {
                Some(_) => "handed out",
                None => "no image",
            }
        );
        if handed.queued > 0 && image.is_some() {
            assert_eq!(
                (handed.staging, handed.flushes),
                (1, 1),
                "{name}: the hand-out made the queued write"
            );
            assert_eq!(
                (after.staging, after.flushes),
                (1, 1),
                "{name}: a handed-out texture's write is made by its call"
            );
        }
    });
}

/// 600 new `Managed` textures, as an icon cache makes them (16 to 512 texels square,
/// `BGRA8Unorm` and `RGBA8Unorm`), each made with no submission waited for: a later command
/// buffer's blits read zero from every one, and so does `get_bytes`. Where the device copies on
/// its staging queue, the moves and zeros were queued and the command buffer's batch carried all
/// 600 in one submission there, which it waited for on the GPU.
#[test]
fn a_burst_of_new_textures_makes_no_staging_submission() {
    const MADE: usize = 600;
    const SIDES: [u64; 6] = [16, 32, 64, 128, 256, 512];
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let before = device.work_counts();
        let textures: Vec<(Texture, u64)> = (0..MADE)
            .map(|i| {
                let side = SIDES[i % SIDES.len()];
                let format = match i % 2 {
                    0 => PixelFormat::BGRA8Unorm,
                    _ => PixelFormat::RGBA8Unorm,
                };
                (texture(device, format, side, managed()), side)
            })
            .collect();
        let made = Path::between(&before, &device.work_counts());
        let command_buffer = queue.command_buffer().unwrap();
        let outs: Vec<Buffer> = (textures.iter())
            .map(|(texture, side)| copy_out(&command_buffer, texture, *side))
            .collect();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}: the blits");
        let read = Path::between(&before, &device.work_counts());
        for (i, (out, (_, side))) in outs.iter().zip(&textures).enumerate() {
            let bytes = read_buffer(out, 0, 4 * side * side).unwrap();
            assert!(
                bytes.iter().all(|&b| b == 0),
                "{name}: the blit of new texture {i} reads zero"
            );
        }
        for (i, (texture, side)) in textures.iter().enumerate() {
            assert!(
                read_back(texture, *side).iter().all(|&b| b == 0),
                "{name}: get_bytes of new texture {i} reads zero"
            );
        }
        let route = match (made.started as usize == MADE, made.moves as usize == MADE) {
            (true, _) => "started on the host",
            (false, true) => "queued for the staging queue",
            _ => "started by submissions",
        };
        println!(
            "STAGING-BATCH {name}: {MADE} new textures {route}: made {made:?}, with the command \
             buffer that read them {read:?}"
        );
        if made.moves > 0 {
            assert_eq!(
                (made.moves as usize, made.staging, made.internal),
                (MADE, 0, 0),
                "{name}: new textures queued for the staging queue make no submission"
            );
            assert_eq!(
                (read.staging, read.flushes),
                (1, 1),
                "{name}: the command buffer's batch carried the 600 in one staging submission"
            );
            staging_waits_bounded(&name, &read);
        }
    });
}

/// The batch that carried a burst's one staging submission waited for it on the GPU at most
/// once, and only while the host had not seen it complete (`Device::staging_wait` in the
/// runtime): a staging submission the copy engine ends before the scheduler's thread checks its
/// wait needs none, which a fast device or a loaded host shows (the RTX 5080: 600 new textures'
/// zeros). The wait itself is proved apart, with the staging submission held on the GPU until
/// the batches were submitted (`tests::staging_waits` in the runtime).
fn staging_waits_bounded(name: &str, read: &Path) {
    assert!(
        read.gpu_waits <= 1 + read.internal,
        "{name}: the batch waited for the staging submission on the GPU at most once (internal \
         submissions wait for it themselves): {} waits",
        read.gpu_waits
    );
    if read.gpu_waits == 0 {
        println!(
            "STAGING-BATCH {name}: the host saw the staging submission complete before the batch \
             was submitted; it did not wait for it"
        );
    }
}

/// A new texture's first write lands after its zeros, whichever first reads it: a
/// `replace_region` of part of it read back by `get_bytes` and by a later command buffer's blit,
/// and a later command buffer's blit into part of it read back by `get_bytes`, each read zero
/// outside the part and the written bytes inside it; and a new texture read before any write
/// reads zero.
#[test]
fn a_new_texture_s_first_write_lands_after_its_zeros() {
    const SIDE: u64 = 64;
    const PART: u64 = 24;
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let part = Region::new_2d(8, 16, PART, PART);
        let written = texels(PART, 11);
        let mut wanted = vec![0; (4 * SIDE * SIDE) as usize];
        for y in 0..PART {
            let at = (4 * ((part.origin.y + y) * SIDE + part.origin.x)) as usize;
            let from = (4 * y * PART) as usize;
            let row = (4 * PART) as usize;
            wanted[at..at + row].copy_from_slice(&written[from..from + row]);
        }
        for options in [ResourceOptions::shared(), managed()] {
            let mode = options.storage_mode();
            // Written by the CPU, read by the CPU.
            let before = device.work_counts();
            let cpu = texture(device, PixelFormat::RGBA8Unorm, SIDE, options);
            cpu.replace_region(part, 0, 0, &written, 4 * PART, 0)
                .unwrap();
            let queued = Path::between(&before, &device.work_counts());
            assert!(
                read_back(&cpu, SIDE) == wanted,
                "{mode:?} on {name}: get_bytes reads zero around the first write"
            );
            let read = Path::between(&before, &device.work_counts());
            // Written by the CPU, read by a later command buffer.
            let gpu = texture(device, PixelFormat::RGBA8Unorm, SIDE, options);
            gpu.replace_region(part, 0, 0, &written, 4 * PART, 0)
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let out = copy_out(&command_buffer, &gpu, SIDE);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{mode:?} on {name}");
            assert!(
                read_buffer(&out, 0, 4 * SIDE * SIDE).unwrap() == wanted,
                "{mode:?} on {name}: a later blit reads zero around the first write"
            );
            // Written by a command buffer's blit, read by the CPU.
            let blitted = texture(device, PixelFormat::RGBA8Unorm, SIDE, options);
            let source = device
                .new_buffer_with_bytes(&written, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                4 * PART,
                0,
                Size::new(PART, PART, 1),
                &blitted,
                0,
                0,
                Origin::new(part.origin.x, part.origin.y, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{mode:?} on {name}");
            assert!(
                read_back(&blitted, SIDE) == wanted,
                "{mode:?} on {name}: get_bytes reads zero around a blit's first write"
            );
            // Read before any write.
            let fresh = texture(device, PixelFormat::RGBA8Unorm, SIDE, options);
            assert!(
                read_back(&fresh, SIDE).iter().all(|&b| b == 0),
                "{mode:?} on {name}: a new texture read at once reads zero"
            );
            println!(
                "STAGING-BATCH {name}: {mode:?} a new texture's first write: queued {queued:?}, \
                 read back {read:?}"
            );
            if queued.moves > 0 {
                assert_eq!(
                    (queued.moves, queued.queued, queued.staging),
                    (1, 1, 0),
                    "{mode:?} on {name}: the move and the write queued, nothing submitted"
                );
                assert_eq!(
                    (read.staging, read.flushes),
                    (1, 1),
                    "{mode:?} on {name}: the read carried the move and the write"
                );
            }
        }
    });
}
