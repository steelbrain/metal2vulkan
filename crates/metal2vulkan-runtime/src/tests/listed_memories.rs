//! The device memory a kernel submission lists while a compositor's working set is held, on the
//! device the cfg selects: report only.
//!
//! RADV lists each `VkDeviceMemory` it imported or allocated exportable in every kernel
//! submission and leaves the process's own buffer objects to the kernel; ANV on i915 lists every
//! `VkDeviceMemory` (`runtime_lane::submit_per_import`). The probe holds the shape of a guest
//! desktop: the runtime's own resources (two display-sized render targets, window-sized render
//! targets, icon-sized sampled textures, `Shared` and `Private` buffers) and the guest's surfaces
//! (no-copy buffers over a `memfd`'s pages whose source makes `udmabuf`s of them, a texture over
//! each), commits frames that copy surfaces into the windows, and prints (`LISTED-MEMORIES`):
//! each live block by its pool's kind, external or the device's own, dedicated or pooled, empty
//! or not (`memory.rs`'s `Memory::blocks_by_kind`); the dma-buf imports of buffers and of images;
//! and a commit's `vkQueueSubmit` time (`WorkCounts::batch_submit_nanos` over
//! `queue_submissions`) before the set, with it held, and after it went, at once and once the
//! kept empty blocks' quiet period passed.
//!
//! Ignored: run it by name in release under `--cfg m2v_gpu`, `--run-ignored all`. Where
//! `/dev/udmabuf` cannot be opened it says so and runs without the guest's surfaces.

use super::host_residency::{host_page, udmabuf_openable, Ram, Scratch};
use crate::{
    Buffer, Device, DeviceKind, DeviceOptions, HostMemory, Origin, ResourceOptions,
    ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "LISTED-MEMORIES";
const CHECK_WAIT: Duration = Duration::from_secs(120);
/// The display's size, a guest desktop's at the consumer's resolution.
const DISPLAY: (u64, u64) = (3648, 2038);
/// The windows' sizes, each a render target the surfaces are copied into.
const WINDOWS: [(u64, u64); 6] = [
    (1280, 800),
    (800, 600),
    (1600, 1000),
    (640, 480),
    (1024, 768),
    (1920, 1080),
];
/// Render targets of each window size.
const PER_WINDOW: usize = 4;
/// Sampled textures of an icon's size, and of a larger one.
const ICONS: usize = 96;
const LARGE_ICONS: usize = 16;
/// The guest's surfaces: each a no-copy buffer of [`SURFACE`] x [`SURFACE`] `BGRA8Unorm` texels,
/// its rows a multiple of the 256 bytes RADV lays a plane's rows at, and a texture over it.
const SURFACES: usize = 96;
const SURFACE: u64 = 512;
/// The frames committed with the set held, and the surfaces each copies.
const FRAMES: usize = 60;
const COPIES_PER_FRAME: usize = 8;
/// The single commits each measurement of a submission's time takes.
const COMMITS: usize = 300;
/// Past the kept empty blocks' quiet period (`memory.rs`'s `KEPT_EMPTY_FOR`, 10 s) and its sweep.
const PAST_QUIET: Duration = Duration::from_secs(12);

/// The device the cfg selects, its submission and in-place checks answered.
fn device(scratch: &Scratch) -> Device {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "{TEST}: the submission checks never answered"
    );
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the in-place check never answered"
    );
    device
}

/// A commit's `vkQueueSubmit` time over [`COMMITS`] commits of one 4-byte fill, each waited for:
/// microseconds a queue submission.
fn submit_us(device: &Device, word: &Buffer) -> f64 {
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    for _ in 0..COMMITS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(word, 0..4, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(crate::tests::SETTLE)
            .unwrap();
    }
    let after = device.work_counts();
    let submissions = (after.queue_submissions - before.queue_submissions).max(1);
    (after.batch_submit_nanos - before.batch_submit_nanos) as f64 / submissions as f64 / 1e3
}

/// Print `device`'s live memory: its blocks by kind, and its imports that lie in no block.
fn census(device: &Device, phase: &str) {
    let blocks = crate::device::lock(&device.shared().memory).blocks_by_kind();
    let mut by_kind: BTreeMap<(String, &str, &str, &str), usize> = BTreeMap::new();
    let mut external = 0;
    for (holds, type_index, dedicated, empty, outside) in &blocks {
        let key = (
            format!("{holds:?} type {type_index}"),
            if *outside { "external" } else { "own" },
            if *dedicated { "dedicated" } else { "pooled" },
            if *empty { "empty" } else { "placed" },
        );
        *by_kind.entry(key).or_default() += 1;
        external += usize::from(*outside);
    }
    let counts = device.work_counts();
    for ((pool, outside, dedicated, empty), count) in &by_kind {
        eprintln!("{TEST} {phase}: {count} blocks of {pool}, {outside}, {dedicated}, {empty}");
    }
    let imports = counts.dma_buf_memories_live + counts.dma_buf_image_memories_live;
    eprintln!(
        "{TEST} {phase}: blocks {} (external {external}); dma-buf imports of buffers {} and of \
         images {}; listed by RADV {} (external blocks and dma-buf imports), by ANV on i915 {} \
         (every memory); work counts: memory_blocks_live {} empty_memory_blocks_live {} \
         dedicated_memory_blocks_live {} host_imports_live {} host_imports_resident {} \
         dma_buf_images {} dma_buf_images_parked {} dma_buf_images_taken {} \
         dma_buf_images_parked_released {} dma_buf_textures_in_place {} \
         exported_dma_buf_images {}",
        blocks.len(),
        counts.dma_buf_memories_live,
        counts.dma_buf_image_memories_live,
        external as u64 + imports,
        blocks.len() as u64 + imports,
        counts.memory_blocks_live,
        counts.empty_memory_blocks_live,
        counts.dedicated_memory_blocks_live,
        counts.host_imports_live,
        counts.host_imports_resident,
        counts.dma_buf_images,
        counts.dma_buf_images_parked,
        counts.dma_buf_images_taken,
        counts.dma_buf_images_parked_released,
        counts.dma_buf_textures_in_place,
        counts.exported_dma_buf_images,
    );
}

/// A `Private` `BGRA8Unorm` texture of `size` with `usage`.
fn texture(device: &Device, (width, height): (u64, u64), usage: TextureUsage) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The runtime's own resources of a compositor's working set: render targets, sampled textures
/// and buffers.
fn compositors_own(device: &Device) -> (Vec<Texture>, Vec<Texture>, Vec<Buffer>) {
    let target = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let mut targets = vec![
        texture(device, DISPLAY, target),
        texture(device, DISPLAY, target),
    ];
    for size in WINDOWS {
        targets.extend((0..PER_WINDOW).map(|_| texture(device, size, target)));
    }
    let mut sampled: Vec<Texture> = (0..ICONS)
        .map(|_| texture(device, (256, 256), TextureUsage::SHADER_READ))
        .collect();
    sampled
        .extend((0..LARGE_ICONS).map(|_| texture(device, (512, 512), TextureUsage::SHADER_READ)));
    let mut buffers = Vec::new();
    for (count, bytes, options) in [
        (64, 64 << 10, ResourceOptions::shared()),
        (32, 1 << 20, ResourceOptions::private()),
        (8, 8 << 20, ResourceOptions::private()),
    ] {
        buffers.extend((0..count).map(|_| device.new_buffer(bytes, options).unwrap()));
    }
    (targets, sampled, buffers)
}

/// The guest's surfaces over `ram`: no-copy buffers over its pages, each with a texture over it.
fn surfaces(device: &Device, ram: &Ram, row: u64) -> Vec<(Buffer, Texture)> {
    let calls = Arc::new(AtomicUsize::new(0));
    let length = row * SURFACE;
    let stride = (length as usize).next_multiple_of(host_page());
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SURFACE, SURFACE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    (0..SURFACES)
        .map(|at| {
            // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
            let buffer = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::with_dma_buf(ram.pointer(at * stride), ram.source(&calls)),
                    length,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
            (buffer, texture)
        })
        .collect()
}

/// [`FRAMES`] frames: each copies [`COPIES_PER_FRAME`] surfaces into the window targets, in turn.
fn frames(device: &Device, targets: &[Texture], sources: &[&Texture]) {
    let queue = device.new_command_queue().unwrap();
    let mut next = 0;
    for _ in 0..FRAMES {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for copy in 0..COPIES_PER_FRAME {
            let source = sources[next % sources.len()];
            let target = &targets[2 + copy % (targets.len() - 2)];
            next += 1;
            let size = Size::new(
                source.width().min(target.width()),
                source.height().min(target.height()),
                1,
            );
            (blit.copy_from_texture_to_texture(
                source,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                target,
                0,
                0,
                Origin::new(0, 0, 0),
            ))
            .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(crate::tests::SETTLE)
            .unwrap();
    }
}

#[test]
#[ignore = "a measurement: run by name with --run-ignored all"]
fn listed_memories() {
    let scratch = Scratch::new();
    let device = device(&scratch);
    eprintln!("{TEST} device: {}", device.name());
    let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    census(&device, "fresh");
    eprintln!(
        "{TEST} fresh: a submission {:.1} us",
        submit_us(&device, &word)
    );
    let (targets, sampled, buffers) = compositors_own(&device);
    let row = SURFACE * 4;
    let length = (row * SURFACE) as usize;
    let file = length.next_multiple_of(host_page()) * SURFACES;
    let guest = udmabuf_openable(TEST).then(|| Ram::new(file, 0, file));
    let held = match &guest {
        Some(ram) => surfaces(&device, ram, row),
        None => Vec::new(),
    };
    let mut sources: Vec<&Texture> = held.iter().map(|(_, texture)| texture).collect();
    sources.extend(sampled.iter().take(SURFACES));
    frames(&device, &targets, &sources);
    census(&device, "held");
    eprintln!(
        "{TEST} held: a submission {:.1} us",
        submit_us(&device, &word)
    );
    drop(sources);
    drop((held, targets, sampled, buffers));
    census(&device, "released");
    eprintln!(
        "{TEST} released: a submission {:.1} us",
        submit_us(&device, &word)
    );
    std::thread::sleep(PAST_QUIET);
    census(&device, "quiet");
    eprintln!(
        "{TEST} quiet: a submission {:.1} us",
        submit_us(&device, &word)
    );
    drop(guest);
}
