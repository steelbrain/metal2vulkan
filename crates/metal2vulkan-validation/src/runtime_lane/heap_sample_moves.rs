//! A heap's multisample texture's moves between its image and Metal's tiles
//! (`texture/heap_tiled.rs`): the images they go through, what the process holds for them, and
//! what a second texture of the same shape adds. Two shapes: one whose images any queue's bound
//! holds, and the consumer's surface, whose it may not.

use super::host_memory::run;
use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, Error, HazardTrackingMode,
    HeapDescriptor, HeapType, Origin, ResourceOptions, Size, StorageMode, Texture,
    TextureDescriptor,
};

/// The process's physical footprint and its peak so far, in MiB, where the OS answers them
/// (`proc_pid_rusage`, macOS); `None` elsewhere.
fn footprint() -> Option<(u64, u64)> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `rusage_info_v4` is plain integers, for which zeros are a value.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let pid = i32::try_from(std::process::id()).ok()?;
        let at = (&mut info as *mut libc::rusage_info_v4).cast::<libc::rusage_info_t>();
        // SAFETY: the flavor names the struct `at` points at, which the call fills and nothing
        // else.
        let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, at) };
        (rc == 0).then_some((
            info.ri_phys_footprint >> 20,
            info.ri_lifetime_max_phys_footprint >> 20,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// A blit of a 16 x 16 square of each of `texture`'s aspects from a buffer, at `(x, 0)` of its
/// slice 0: the texture's shadow copied in before it and back after it.
fn blit_into(queue: &CommandQueue, texture: &Texture, buffer: &Buffer, x: u64) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (unit, option) in [
        (4, BlitOption::DepthFromDepthStencil),
        (1, BlitOption::StencilFromDepthStencil),
    ] {
        let row = 16 * 4 * unit;
        blit.copy_from_buffer_to_texture(
            buffer,
            0,
            row,
            row * 16,
            Size::new(16, 16, 1),
            texture,
            0,
            0,
            Origin::new(x, 0, 0),
            option,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// `Depth32Float_Stencil8` textures of 4 samples (`allowGPUOptimizedContents` off, which Metal
/// keeps uncompressed) in `Shared` placement heaps, a shadow of each one's samples in Metal's
/// tiles there where the device moves them (`WorkCounts::heap_textures_tiled`), blitted into
/// twice, then a second texture of the shape: the moves' images are made as the first blit's
/// copies are recorded, none as a texture is made, and every later copy, the second texture's
/// too, needs the images the first copy took (`WorkCounts::heap_sample_move_images_made` and
/// `heap_sample_move_images_lent`, `texture/copy_images.rs`).
///
/// Two shapes, each on a queue of its own, since the queue is what keeps the images:
///
/// - 256 x 256, whose moves' images are a few MiB, within the bound of any device's queue (a
///   sixty-fourth of its largest device-local heap): every copy after the first takes them all
///   from those its queue kept and makes none.
/// - 4096 x 2160, the consumer's surface, whose moves' images are about 295 MiB. A queue keeps
///   them where its bound holds them (an M3 Max's is over 1 GiB), and where it does not -- a
///   16 GiB GPU keeps 255 MiB -- a copy makes again those it could not keep, as `copy_images.rs`
///   states. So the claim held to here is the one that holds on every device: each copy takes as
///   many images as the first, and the `MOVES` lines print how many of them it made.
///
/// `MOVES` lines print the process's footprint at each step and its peak.
#[test]
fn a_large_heap_multisample_texture_s_moves() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device: &Device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let options = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    );
    let samples = device.new_buffer(16 * 16 * 4 * 4, options).unwrap();
    let print = |step: &str| match footprint() {
        Some((now, peak)) => eprintln!("MOVES {step}: footprint {now} MiB, peak {peak} MiB"),
        None => eprintln!("MOVES {step}"),
    };
    let moved = || {
        let counts = device.work_counts();
        (
            counts.heap_sample_move_images_made,
            counts.heap_sample_move_images_lent,
        )
    };
    print("start");
    for (side, within_the_bound) in [((256u64, 256u64), true), ((4096, 2160), false)] {
        let (width, height) = side;
        let shape = format!("{width}x{height}");
        // The queue keeps the images, so each shape's copies take only its own.
        let queue = device.new_command_queue().unwrap();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, width, height, false);
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = 4;
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = options;
        descriptor.allow_gpu_optimized_contents = false;
        let answer = match device.heap_texture_size_and_align(&descriptor) {
            Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
                eprintln!("MOVES: the device takes no such sample count");
                return;
            }
            answer => answer.unwrap(),
        };
        let mut kept = Vec::new();
        // The first copy's images made and lent, which every copy after it needs as many of.
        let mut first = None;
        for second in [false, true] {
            let what = if second { "second" } else { "first" };
            let what = format!("{shape} {what}");
            let heap = device
                .new_heap(&HeapDescriptor::new(
                    HeapType::Placement,
                    answer.size,
                    options,
                ))
                .unwrap();
            print(&format!("{what} heap made ({} MiB)", answer.size >> 20));
            let before = device.work_counts().heap_textures_tiled;
            let images = moved();
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let tiled = device.work_counts().heap_textures_tiled - before == 1;
            print(&format!("{what} texture made, tiled {tiled}"));
            if !tiled {
                return;
            }
            assert_eq!(moved(), images, "{what}: images made with the texture");
            for round in 0..2 {
                let (made, lent) = moved();
                blit_into(&queue, &texture, &samples, 16 * round);
                let (now_made, now_lent) = moved();
                let (made, lent) = (now_made - made, now_lent - lent);
                print(&format!(
                    "{what} texture blitted into, round {round}: {made} images made, {lent} lent"
                ));
                let Some((first_made, first_lent)) = first else {
                    assert!(
                        made > 0,
                        "{what}, round {round}: the first copy made no image"
                    );
                    first = Some((made, lent));
                    continue;
                };
                assert_eq!(
                    made + lent,
                    first_made + first_lent,
                    "{what}, round {round}: images taken"
                );
                if within_the_bound {
                    assert_eq!(made, 0, "{what}, round {round}: images made again");
                    assert_eq!(lent, first_made + first_lent, "{what}, round {round}: lent");
                }
            }
            kept.push((heap, texture));
        }
        drop(kept);
        print(&format!("{shape}: both gone"));
    }
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}
