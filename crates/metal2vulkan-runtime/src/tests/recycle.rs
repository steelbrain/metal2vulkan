//! Recycled images and buffers (`recycle.rs`), on the device the GPU cfg selects.
//!
//! - **A new texture reads what a new one reads.** A `Private` `BGRA8Unorm` render target, cleared
//!   to a colour and released, then a texture of the same descriptor: it takes the parked image
//!   (`WorkCounts::object_churn.images_recycled`), and a blit of it reads the same bytes as a
//!   texture of the same descriptor made while no image was parked, never the colour. Native
//!   Metal on the M3 Max under macOS 27.0.1 reads a fixed pattern from such a texture made right
//!   after one of the same shape was cleared and released, and zero from an uncompressed
//!   `Private` or a `Shared` one, never the old texels (the inplace-surface probe's `fresh` mode).
//!   On a device that presents a Mac keeping no texture compressed
//!   (`DeviceProperties::lossless_compression` false) a new `Private` texture holds what its
//!   memory held, written by no command (`WorkCounts::private_textures_unfilled`), as such a
//!   Mac's does (docs/API.md, the accepted differences): the recycled one reads the parked
//!   image's texels, the colour, or what a new one read.
//! - **Each frame's textures, recycled.** Eight such textures made, cleared and released each of
//!   sixty frames take parked images from the second frame on: no image made after the first
//!   frame's, and none parked past the bounds.
//! - **A burst of one-off makings keeps the proven image.** A texture of the descriptor made,
//!   cleared and released, made again from the parked image (its making came again), cleared and
//!   released; then more textures of one-off widths than the bounds keep, each made and released,
//!   which evicts some; then a texture of the descriptor: it takes the proven making's image (the
//!   same `VkImage`), which none of the one-offs evicted, and reads what a new texture reads
//!   (handed over from `UNDEFINED`, its first move written), never the colour.
//! - **A recycled buffer reads zero.** A `Private` and a `Shared` buffer, filled with a byte and
//!   released, then a buffer of the same length and options: it takes the parked buffer
//!   (`buffers_recycled`) and reads zero, as every new Metal buffer does (native Metal on the M3
//!   Max under macOS 27.0.1 read zero from `Shared` and `Private` buffers made right after one of
//!   the same length was filled and released: the inplace-surface probe's `fresh` mode).
//! - **Each frame's buffers, recycled.** Eight `Private` buffers made, filled and released each of
//!   sixty frames take parked buffers from the second frame on.

use super::blit_holds::{Scratch, WAIT};
use crate::recycle;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, LoadAction, Origin, RenderPassDescriptor, ResourceOptions, ShaderCacheOptions,
    Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::time::Instant;

const SIDE: u64 = 64;
/// A recycled buffer's length: not a multiple of 16, so its padding is recycled too.
const LENGTH: u64 = 4100;

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

fn descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// Clear `target` to a colour, waited for.
fn clear(queue: &CommandQueue, target: &Texture) {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of `texture`, blitted out.
fn read(device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let length = SIDE * SIDE * 4;
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &out,
        0,
        SIDE * 4,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    // SAFETY: the shared buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length as usize) }
        .to_vec()
}

/// Check `read`, what the recycled texture `second` of `device` reads, against `fresh` (what a new
/// texture of the descriptor read) and `cleared` (the parked image's colour): `fresh` where the
/// device presents compression, else either (the module docs), its making counted as unfilled.
fn reads_what_a_new_one_reads(
    device: &Device,
    unfilled: u64,
    read: &[u8],
    fresh: &[u8],
    cleared: &[u8],
) {
    if device.properties().lossless_compression {
        assert_eq!(unfilled, 0, "a new Private texture counted as unfilled");
        assert_eq!(
            read, fresh,
            "the recycled texture reads what a new one reads, not the old texels"
        );
        return;
    }
    assert_eq!(
        unfilled, 1,
        "the recycled texture's making counted as unfilled"
    );
    assert!(
        read == fresh || read == cleared,
        "the recycled texture reads what its memory held, the parked image's texels or a new one's"
    );
    let what = match read == cleared {
        true => "the parked image's texels",
        false => "what a new one reads",
    };
    println!("RECYCLE-UNFILLED on {}: {what}", device.name());
}

/// Wait until the device parked `count` images in all.
fn parked_at_least(device: &Device, count: u64) {
    let until = std::time::Instant::now() + WAIT;
    while device.work_counts().object_churn.images_parked < count {
        assert!(std::time::Instant::now() < until, "the images were parked");
        std::thread::yield_now();
    }
}

#[test]
fn a_recycled_texture_reads_what_a_new_one_reads() {
    let scratch = Scratch::new("recycle-fresh");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    // A texture made while nothing is parked: what a new texture of the descriptor reads.
    let first = device.new_texture(&descriptor()).unwrap();
    let fresh = read(&device, &queue, &first);
    clear(&queue, &first);
    let cleared = read(&device, &queue, &first);
    assert_ne!(fresh, cleared, "the clear changed the texels");
    let before = device.work_counts().object_churn;
    drop(first);
    parked_at_least(&device, before.images_parked + 1);
    let unfilled = device.work_counts().private_textures_unfilled;
    let second = device.new_texture(&descriptor()).unwrap();
    let unfilled = device.work_counts().private_textures_unfilled - unfilled;
    let after = device.work_counts().object_churn;
    assert_eq!(
        after.images_recycled - before.images_recycled,
        1,
        "the second texture took the first one's parked image"
    );
    let second = read(&device, &queue, &second);
    reads_what_a_new_one_reads(&device, unfilled, &second, &fresh, &cleared);
}

#[test]
fn each_frames_textures_take_parked_images() {
    let scratch = Scratch::new("recycle-frames");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    const FRAMES: u64 = 60;
    const TEXTURES: u64 = 8;
    let start = device.work_counts().object_churn;
    for frame in 1..=FRAMES {
        let textures: Vec<Texture> = (0..TEXTURES)
            .map(|_| device.new_texture(&descriptor()).unwrap())
            .collect();
        for texture in &textures {
            clear(&queue, texture);
        }
        drop(textures);
        // The images go as their command buffers' holds do, which a completion may let go of
        // after the wait returned.
        parked_at_least(&device, start.images_parked + frame * TEXTURES);
    }
    let end = device.work_counts().object_churn;
    let recycled = end.images_recycled - start.images_recycled;
    eprintln!(
        "RECYCLE: {FRAMES} frames of {TEXTURES} textures: {recycled} images recycled, {} parked, \
         {} evicted",
        end.images_parked - start.images_parked,
        end.images_evicted - start.images_evicted
    );
    assert_eq!(
        recycled,
        (FRAMES - 1) * TEXTURES,
        "every frame after the first recycles"
    );
}

#[test]
fn a_burst_of_one_off_makings_keeps_the_proven_image() {
    let scratch = Scratch::new("recycle-burst");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    const ONE_OFF: u64 = 64;
    // A burst slower than the images' park time lets the proven image go stale before the take,
    // as the bounds say it goes (an RTX 5080 run, 1 in 13): such a try is made again, its one-offs
    // of widths no earlier try made, at most this many times.
    const TRIES: u64 = 3;
    let first = device.new_texture(&descriptor()).unwrap();
    let fresh = read(&device, &queue, &first);
    clear(&queue, &first);
    let cleared = read(&device, &queue, &first);
    let parked = device.work_counts().object_churn.images_parked;
    drop(first);
    parked_at_least(&device, parked + 1);
    for tried in 0..TRIES {
        let start = device.work_counts().object_churn;
        // The making comes again: the image a take hands out is a proven one as it goes back.
        let second = device.new_texture(&descriptor()).unwrap();
        let proven = second.raw_image().unwrap();
        clear(&queue, &second);
        // Before the park: a take sooner after this than the park time finds the image unstale.
        let parking = Instant::now();
        drop(second);
        parked_at_least(&device, start.images_parked + 1);
        let before = device.work_counts().object_churn;
        assert_eq!(before.images_recycled - start.images_recycled, 1);
        // One-off widths, each made and released: more than the recycler keeps.
        for width in 1..=ONE_OFF {
            let mut one_off = descriptor();
            one_off.width = SIDE + tried * ONE_OFF + width;
            drop(device.new_texture(&one_off).unwrap());
        }
        parked_at_least(&device, before.images_parked + ONE_OFF);
        let burst = device.work_counts().object_churn;
        let unfilled = device.work_counts().private_textures_unfilled;
        assert_eq!(burst.images_recycled, before.images_recycled);
        assert!(
            burst.images_evicted > before.images_evicted,
            "the one-offs went past the recycler's bounds"
        );
        let third = device.new_texture(&descriptor()).unwrap();
        let taken = parking.elapsed();
        let unfilled = device.work_counts().private_textures_unfilled - unfilled;
        let after = device.work_counts().object_churn;
        if taken >= recycle::IMAGES.parked_for && tried + 1 < TRIES {
            eprintln!(
                "RECYCLE: try {tried}: the take came {taken:?} after the park, past the park \
                 time: made again"
            );
            // Parked before the next try's second texture takes it.
            drop(third);
            parked_at_least(&device, after.images_parked + 1);
            continue;
        }
        assert!(
            taken < recycle::IMAGES.parked_for,
            "every try's burst outlasted the park time ({taken:?} the last)"
        );
        assert_eq!(
            after.images_recycled - burst.images_recycled,
            1,
            "the texture took a parked image"
        );
        assert_eq!(
            third.raw_image().unwrap(),
            proven,
            "the proven making's parked image outlasted the burst"
        );
        let third = read(&device, &queue, &third);
        reads_what_a_new_one_reads(&device, unfilled, &third, &fresh, &cleared);
        eprintln!(
            "RECYCLE: a burst of {ONE_OFF} one-off makings: {} evicted, the proven image kept",
            burst.images_evicted - before.images_evicted
        );
        return;
    }
}

/// Fill `buffer` with `value`, waited for.
fn fill(queue: &CommandQueue, buffer: &Buffer, value: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of `buffer`, copied into `out` (`Shared`, at least as long), waited for.
fn read_buffer(queue: &CommandQueue, buffer: &Buffer, out: &Buffer) -> Vec<u8> {
    let length = buffer.length();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, out, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    // SAFETY: the shared buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), length as usize) }
        .to_vec()
}

/// Wait until the device parked `count` buffers in all.
fn buffers_parked_at_least(device: &Device, count: u64) {
    let until = std::time::Instant::now() + WAIT;
    while device.work_counts().object_churn.buffers_parked < count {
        assert!(std::time::Instant::now() < until, "the buffers were parked");
        std::thread::yield_now();
    }
}

#[test]
fn a_recycled_buffer_reads_zero() {
    let scratch = Scratch::new("recycle-buffer");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    // Live for the whole test, and longer than the recycled buffers: never parked here.
    let out = device
        .new_buffer(2 * LENGTH, ResourceOptions::shared())
        .unwrap();
    // Each recycled buffer, kept: one parked as the next options' first goes would answer that
    // wait.
    let mut kept = Vec::new();
    for options in [ResourceOptions::private(), ResourceOptions::shared()] {
        let first = device.new_buffer(LENGTH, options).unwrap();
        fill(&queue, &first, 0xab);
        assert!(
            read_buffer(&queue, &first, &out).iter().all(|&b| b == 0xab),
            "the fill wrote every byte"
        );
        let before = device.work_counts().object_churn;
        drop(first);
        buffers_parked_at_least(&device, before.buffers_parked + 1);
        let second = device.new_buffer(LENGTH, options).unwrap();
        let after = device.work_counts().object_churn;
        assert_eq!(
            after.buffers_recycled - before.buffers_recycled,
            1,
            "the second buffer took the first one's parked buffer ({options:?})"
        );
        if let Some(contents) = second.contents() {
            // SAFETY: the shared buffer's bytes, which no command uses yet.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LENGTH as usize) };
            assert!(bytes.iter().all(|&b| b == 0), "its contents read zero");
        }
        assert!(
            read_buffer(&queue, &second, &out).iter().all(|&b| b == 0),
            "the recycled buffer reads zero, not the old bytes ({options:?})"
        );
        kept.push(second);
    }
}

#[test]
fn each_frames_buffers_take_parked_buffers() {
    let scratch = Scratch::new("recycle-buffer-frames");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    const FRAMES: u64 = 60;
    const BUFFERS: u64 = 8;
    let start = device.work_counts().object_churn;
    for frame in 1..=FRAMES {
        let buffers: Vec<Buffer> = (0..BUFFERS)
            .map(|_| {
                device
                    .new_buffer(64 << 10, ResourceOptions::private())
                    .unwrap()
            })
            .collect();
        for buffer in &buffers {
            fill(&queue, buffer, 0x5a);
        }
        drop(buffers);
        // The buffers go as their command buffers' holds do, which a completion may let go of
        // after the wait returned.
        buffers_parked_at_least(&device, start.buffers_parked + frame * BUFFERS);
    }
    let end = device.work_counts().object_churn;
    let recycled = end.buffers_recycled - start.buffers_recycled;
    eprintln!(
        "RECYCLE: {FRAMES} frames of {BUFFERS} buffers: {recycled} buffers recycled, {} parked, \
         {} evicted, {} destroyed in the process",
        end.buffers_parked - start.buffers_parked,
        end.buffers_evicted - start.buffers_evicted,
        end.vk_buffers_destroyed - start.vk_buffers_destroyed
    );
    assert_eq!(
        recycled,
        (FRAMES - 1) * BUFFERS,
        "every frame after the first recycles"
    );
}
