//! What a new texture's first contents (the zeros, the pattern) cost, on the device the cfg
//! selects, without the validation layer.
//!
//! Each shape makes [`Shape::count`] textures of one descriptor, then commits a command buffer of
//! one 16-byte buffer fill and waits for it. The submission records the textures' first work
//! (`texture/first_move.rs`) in a `VkSubmitInfo` of its own before the fill's: Metal's pattern
//! where Metal compresses a `Private` texture, cleared or copied, a clear of its other levels and
//! of the other `Private` ones, and zeros copied into a `Shared` one, or written by the host as it
//! is made where the device copies texels from the host. Each sample is the host's time to make the
//! textures (a texture's share) and the host's time from the commit to the completion; the same
//! command buffer committed with no texture made just before (`none`) is sampled beside each.
//! [`WARM`] samples, then [`SAMPLES`] a shape; medians and fastest (`FRESH-COST`).
//!
//! The shapes: `RGBA8Unorm` 1024² render targets, eight `Private` and eight `Shared`; two
//! `Private` `RGBA8Unorm` 4096²; two `Private` mipmapped `RGBA16Float` 2048²; two `Private`
//! `Depth32Float` 2048²; two `Private` 4x `RGBA8Unorm` 1024².
//!
//! Measures; asserts only, once every shape is measured, that the last texture of each
//! single-sampled shape reads Metal's answer in its first level (the first work's path ran): the
//! `Private` ones Metal's pattern (`RGBA8Unorm` opaque magenta, `RGBA16Float` and `Depth32Float`
//! all-ones NaNs where the device writes them, zero announced where it does not), the `Shared`
//! one zero. On a device without Metal's compression (`DeviceProperties::lossless_compression`
//! false) a `Private` texture has no first work: it holds what its memory held, unchecked, and the
//! test asserts instead that each was counted so (`WorkCounts::private_textures_unfilled`).

use super::fresh_contents::{unfilled, unfilled_count, zeroed, FRESH_PATTERNS};
use super::{read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, Device, Origin, ResourceOptions, Size, StorageMode, Texture,
    TextureDescriptor,
};
use std::time::Instant;

/// The samples a shape takes before those it reports.
const WARM: usize = 3;

/// The samples a shape reports.
const SAMPLES: usize = 15;

/// A shape: its name, descriptor, the textures each sample makes, their bytes, and the texel its
/// first level reads where the test reads the last one back.
struct Shape {
    name: &'static str,
    descriptor: TextureDescriptor,
    count: usize,
    bytes: u64,
    texel: Option<&'static [u8]>,
}

/// A 2D `pf` texture of `side`², `options`: `count` a sample, each pixel `samples` of `texel`'s
/// size, its first level reading `texel` unless multisampled (then not read back).
fn shape(
    name: &'static str,
    pf: PixelFormat,
    (side, texel, samples): (u64, &'static [u8], u64),
    options: ResourceOptions,
    count: usize,
) -> Shape {
    let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
        descriptor.usage = TextureUsage::RENDER_TARGET;
    }
    Shape {
        name,
        descriptor,
        count,
        bytes: side * side * texel.len() as u64 * samples,
        texel: (samples == 1).then_some(texel),
    }
}

/// The shapes (the module docs).
fn shapes() -> Vec<Shape> {
    let private = ResourceOptions::private();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let magenta: &[u8] = &[0xff, 0, 0xff, 0xff];
    let mut mips = shape(
        "Private mipmapped RGBA16Float 2048²",
        PixelFormat::RGBA16Float,
        (2048, &[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], 1),
        private,
        2,
    );
    mips.descriptor.mipmap_level_count = 12;
    mips.bytes = mips.bytes * 4 / 3;
    vec![
        shape(
            "Private RGBA8Unorm 1024²",
            rgba8,
            (1024, magenta, 1),
            private,
            8,
        ),
        shape(
            "Shared RGBA8Unorm 1024²",
            rgba8,
            (1024, &[0; 4], 1),
            ResourceOptions::shared(),
            8,
        ),
        shape(
            "Private RGBA8Unorm 4096²",
            rgba8,
            (4096, magenta, 1),
            private,
            2,
        ),
        mips,
        shape(
            "Private Depth32Float 2048²",
            PixelFormat::Depth32Float,
            (2048, &[0xff; 4], 1),
            private,
            2,
        ),
        shape(
            "Private 4x RGBA8Unorm 1024²",
            rgba8,
            (1024, magenta, 4),
            private,
            2,
        ),
    ]
}

/// Commit one 16-byte fill of `scratch` on `queue` and wait: the host's time from the commit to
/// the completion, in milliseconds.
fn submit(queue: &CommandQueue, scratch: &Buffer) -> f64 {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(scratch, 0..16, 0).unwrap();
    blit.end_encoding().unwrap();
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wall = committed.elapsed().as_secs_f64() * 1e3;
    assert_eq!(command_buffer.error(), None);
    wall
}

/// The first level of `texture`, `texel` bytes a texel, through a blit.
fn first_level(device: &Device, queue: &CommandQueue, texture: &Texture, texel: u64) -> Vec<u8> {
    let (w, h) = (texture.width(), texture.height());
    let length = w * h * texel;
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
        Size::new(w, h, 1),
        &out,
        0,
        w * texel,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, length).unwrap()
}

/// The median and the fastest of `samples`.
fn stat(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (samples[samples.len() / 2], samples[0])
}

/// Each shape's making and first submission (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn new_texture_zero_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        options.announce_each = FRESH_PATTERNS.into_iter().collect();
    })
    .unwrap();
    let queue = device.new_command_queue().unwrap();
    let scratch = device.new_buffer(16, ResourceOptions::private()).unwrap();
    let on = device.name();
    let mut other = Vec::new();
    for shape in shapes() {
        let name = shape.name;
        let (mut made, mut first, mut none) = (Vec::new(), Vec::new(), Vec::new());
        let mut last = None;
        let private = matches!(
            shape.descriptor.resource_options.storage_mode(),
            Ok(StorageMode::Private)
        );
        let left = private && unfilled(&device);
        let before = unfilled_count(&device);
        for k in 0..WARM + SAMPLES {
            let alone = submit(&queue, &scratch);
            let began = Instant::now();
            let textures: Vec<Texture> = (0..shape.count)
                .map(|_| device.new_texture(&shape.descriptor).unwrap())
                .collect();
            let making = began.elapsed().as_secs_f64() * 1e3 / shape.count as f64;
            let submitted = submit(&queue, &scratch);
            if k >= WARM {
                made.push(making);
                first.push(submitted);
                none.push(alone);
            }
            last = textures.into_iter().last();
        }
        let each = u64::from(left) * shape.count as u64;
        assert_eq!(
            unfilled_count(&device) - before,
            each * (WARM + SAMPLES) as u64,
            "{name} on {on}: counted as made unfilled"
        );
        let (make, make_fast) = stat(made);
        let (wall, wall_fast) = stat(first);
        let (alone, alone_fast) = stat(none);
        let mib = shape.bytes as f64 / f64::from(1 << 20);
        println!(
            "FRESH-COST {name} on {on}, {} a sample, {mib:.1} MiB each: make median {make:.3} ms \
             a texture, fastest {make_fast:.3} ms; commit to completion median {wall:.3} ms, \
             fastest {wall_fast:.3} ms; none median {alone:.3} ms, fastest {alone_fast:.3} ms; \
             first work median {:.3} ms, {:.3} ms a texture",
            shape.count,
            wall - alone,
            (wall - alone) / shape.count as f64,
        );
        if let (Some(texel), Some(texture), false) = (shape.texel, &last, left) {
            let bytes = first_level(&device, &queue, texture, texel.len() as u64);
            // Zero where the device announced, at this read, that it could not write Metal's
            // pattern exactly.
            let zeroed = zeroed(&messages);
            let reads = |at: &[u8]| match zeroed {
                true => at.iter().all(|&b| b == 0),
                false => at == texel,
            };
            if !bytes.chunks(texel.len()).all(reads) {
                other.push(name);
            }
        }
    }
    assert!(
        other.is_empty(),
        "on {on}, the last texture's first level reads other than Metal's answer: {other:?}"
    );
    drop(cache);
}
