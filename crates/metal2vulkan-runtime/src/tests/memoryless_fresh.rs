//! A new memoryless texture reads zero, never what its image's memory held: a released texture's
//! samples, of another process of a consumer whose processes share the device
//! (`Device::new_texture_image`), on the device the GPU cfg selects.
//!
//! - **A recycled memoryless texture's resolve.** A memoryless 4x `BGRA8Unorm` color and a 4x
//!   `Depth32Float` depth, each cleared by a pass of no draw call and resolved, then released, and
//!   a memoryless texture of the same descriptor made: it takes the parked image
//!   (`WorkCounts::object_churn.images_recycled`), and a pass of no draw call that loads it
//!   `DontCare` and resolves it writes zero into its resolve texture, never the earlier clear,
//!   its making counted (`WorkCounts::memoryless_textures_zeroed`). Its resolve is the only
//!   command that reads it. Where memoryless memory is lazily allocated (MoltenVK) nothing is
//!   zeroed and the reading is printed alone.
//! - **Cost** (a measurement, ignored): the GPU span of one such pass over a 1920x1080 4x
//!   attachment, made once and passed again and again (each pass after its first),
//!   `MEMORYLESS-COST`.

use super::blit_holds::{Scratch, WAIT};
use crate::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, DeviceKind,
    DeviceOptions, HazardTrackingMode, LoadAction, Origin, RenderPassDescriptor, ResourceOptions,
    ShaderCacheOptions, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::path::Path;
use std::time::Instant;

/// The attachments' side.
const SIDE: u64 = 64;

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

/// A memoryless 4x render target of `pf`, `width` by `height`.
fn memoryless(pf: PixelFormat, width: u64, height: u64) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = 4;
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::new(
        StorageMode::Memoryless,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    descriptor
}

/// A `Private` single-sample render target of `pf`, `width` by `height`, a resolve's.
fn resolved(pf: PixelFormat, width: u64, height: u64) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// The descriptor of a pass of no draw call that loads `attachment` (the depth where `depth`)
/// with `load`, clearing a color to `value` in each channel but alpha or the depth to `value`,
/// and resolves it into `into`.
fn resolve_pass(
    attachment: &Texture,
    into: &Texture,
    depth: bool,
    load: LoadAction,
    value: f64,
) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    if depth {
        let depth = &mut pass.depth_attachment;
        depth.texture = Some(attachment.clone());
        depth.load_action = load;
        depth.store_action = StoreAction::MultisampleResolve;
        depth.clear_depth = value;
        depth.resolve_texture = Some(into.clone());
    } else {
        let color = &mut pass.color_attachments[0];
        color.texture = Some(attachment.clone());
        color.load_action = load;
        color.store_action = StoreAction::MultisampleResolve;
        color.clear_color = ClearColor::new(value, value, value, 1.0);
        color.resolve_texture = Some(into.clone());
    }
    pass
}

/// Encode and commit one pass of `pass`, waited for.
fn run(queue: &CommandQueue, pass: &RenderPassDescriptor) {
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.render_command_encoder(pass).unwrap())
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes of `texture`, four a texel, `SIDE`², blitted out.
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

#[test]
fn a_recycled_memoryless_texture_s_resolve_reads_zero_never_the_old_samples() {
    let scratch = Scratch::new("memoryless-fresh");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let cases = [
        (PixelFormat::BGRA8Unorm, false, [0u8; 4]),
        (PixelFormat::Depth32Float, true, [0u8; 4]),
    ];
    for (pf, depth, fresh) in cases {
        let first = device.new_texture(&memoryless(pf, SIDE, SIDE)).unwrap();
        let into = device.new_texture(&resolved(pf, SIDE, SIDE)).unwrap();
        run(
            &queue,
            &resolve_pass(&first, &into, depth, LoadAction::Clear, 0.25),
        );
        let cleared = read(&device, &queue, &into);
        assert!(
            cleared.chunks(4).all(|texel| texel != fresh),
            "{pf:?}: the first pass's clear resolved"
        );
        let before = device.work_counts().object_churn;
        drop(first);
        let until = Instant::now() + WAIT;
        while device.work_counts().object_churn.images_parked == before.images_parked {
            assert!(Instant::now() < until, "{pf:?}: the image was parked");
            std::thread::yield_now();
        }
        let zeroed = device.work_counts().memoryless_textures_zeroed;
        let second = device.new_texture(&memoryless(pf, SIDE, SIDE)).unwrap();
        let zeroed = device.work_counts().memoryless_textures_zeroed - zeroed;
        let after = device.work_counts().object_churn;
        assert_eq!(
            after.images_recycled - before.images_recycled,
            1,
            "{pf:?}: the second texture took the first one's parked image"
        );
        let into = device.new_texture(&resolved(pf, SIDE, SIDE)).unwrap();
        run(
            &queue,
            &resolve_pass(&second, &into, depth, LoadAction::DontCare, 1.0),
        );
        let read = read(&device, &queue, &into);
        let what = match (read == cleared, read.chunks(4).all(|texel| texel == fresh)) {
            (true, _) => "the released texture's samples",
            (false, true) => "zero",
            (false, false) => "other bytes",
        };
        println!("MEMORYLESS-FRESH {pf:?} on {}: {what}", device.name());
        // Lazily allocated memoryless memory (MoltenVK) holds nothing past a pass and is not
        // zeroed.
        if second.lazily_allocated() {
            assert_eq!(zeroed, 0, "{pf:?}: a lazily allocated texture zeroed");
            continue;
        }
        assert_eq!(zeroed, 1, "{pf:?}: the making counted as zeroed");
        assert_eq!(what, "zero", "{pf:?}");
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --ignored"]
fn memoryless_dont_care_pass_costs() {
    const WARM: usize = 5;
    const SAMPLES: usize = 41;
    let scratch = Scratch::new("memoryless-cost");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let (width, height) = (1920, 1080);
    for (pf, depth) in [
        (PixelFormat::BGRA8Unorm, false),
        (PixelFormat::Depth32Float, true),
    ] {
        let attachment = device.new_texture(&memoryless(pf, width, height)).unwrap();
        let into = device.new_texture(&resolved(pf, width, height)).unwrap();
        let pass = resolve_pass(&attachment, &into, depth, LoadAction::DontCare, 1.0);
        let mut spans: Vec<f64> = (0..WARM + SAMPLES)
            .map(|_| {
                let began = Instant::now();
                run(&queue, &pass);
                began.elapsed().as_secs_f64() * 1e3
            })
            .skip(WARM)
            .collect();
        spans.sort_by(f64::total_cmp);
        println!(
            "MEMORYLESS-COST {pf:?} 4x {width}x{height} DontCare, resolved, no draw, on {}: \
             commit to completion median {:.3} ms, fastest {:.3} ms",
            device.name(),
            spans[SAMPLES / 2],
            spans[0],
        );
    }
}
