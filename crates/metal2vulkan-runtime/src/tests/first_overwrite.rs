//! A new zero-filled `Private` texture whose first use writes every texel first takes no zeros
//! (`texture/first_move.rs`, `WorkCounts::private_textures_overwritten_unfilled`), on the device
//! the GPU cfg selects; one whose first use may read a texel first takes them. Only where the
//! device presents a Mac keeping no texture compressed (`DeviceProperties::lossless_compression`
//! false): elsewhere a new `Private` texture takes Metal's pattern and nothing is counted.
//!
//! - **Written whole first.** A render pass that loads it `Clear` and stores it, or a blit copy
//!   into its whole extent, as its command buffer's first encoder: counted, and it reads the clear
//!   or the copy. So too a texture whose image a released texture's parked one was
//!   (`recycle.rs`): it reads the clear, never the old colour.
//! - **Read or written in part first.** A pass that loads it `DontCare` or `Load`, a copy into a
//!   part of it, or a clearing pass after a render pass of another texture (whose commands may
//!   reach any texture through its resource ID): uncounted, and it reads zero wherever nothing
//!   wrote.
//! - **Cost** (a measurement, ignored): eight new 1024² `RGBA8Unorm` and two 2048²
//!   `Depth32Float` textures each cleared by a command buffer of its own, from the first making or
//!   commit to the last completion, `OVERWRITE-COST`. Made together before the first commit, the
//!   first submission takes every texture's first move and only its own texture goes unzeroed;
//!   each made just before its command buffer, every one does.

use super::blit_holds::{Scratch, WAIT};
use crate::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    LoadAction, Origin, RenderPassDescriptor, ResourceOptions, ShaderCacheOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::time::Instant;

/// The textures' side.
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

/// A `Private` render target of `pf`, `side`².
fn target(pf: PixelFormat, side: u64) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// A color pass over `texture` loading it `load`, clearing it to `value` in every channel, and
/// storing it.
fn color_pass(texture: &Texture, load: LoadAction, value: f64) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(value, value, value, value);
    pass
}

/// A depth pass over `texture` clearing it to `depth` and storing it.
fn depth_pass(texture: &Texture, depth: f64) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.depth_attachment;
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_depth = depth;
    pass
}

/// One command buffer of a pass of each of `passes`, in order, waited for.
fn run(queue: &CommandQueue, passes: &[RenderPassDescriptor]) {
    let command_buffer = queue.command_buffer().unwrap();
    for pass in passes {
        (command_buffer.render_command_encoder(pass).unwrap())
            .end_encoding()
            .unwrap();
    }
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// One command buffer of a copy of `bytes` into `texture` at `origin` over `size`, waited for.
fn copy_in(device: &Device, queue: &CommandQueue, texture: &Texture, origin: Origin, size: Size) {
    let bytes = vec![0x5a; (size.width * size.height * 4) as usize];
    let source = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        size.width * 4,
        bytes.len() as u64,
        size,
        texture,
        0,
        0,
        origin,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
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

/// The device's count of first moves that went without their zeros so far.
fn overwritten(device: &Device) -> u64 {
    device.work_counts().private_textures_overwritten_unfilled
}

/// Whether `device` zero-fills a new `Private` texture (the module docs).
fn zero_fills(device: &Device) -> bool {
    !device.properties().lossless_compression
}

#[test]
fn a_new_texture_written_whole_first_takes_no_zeros() {
    let scratch = Scratch::new("first-overwrite-whole");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let counted = u64::from(zero_fills(&device));
    // A clearing pass, its command buffer's first encoder.
    let texture = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    let before = overwritten(&device);
    run(&queue, &[color_pass(&texture, LoadAction::Clear, 1.0)]);
    assert_eq!(overwritten(&device) - before, counted, "a clearing pass");
    let bytes = read(&device, &queue, &texture);
    assert!(bytes.iter().all(|&b| b == 0xff), "the clear");
    // A clearing depth pass.
    let depth = device
        .new_texture(&target(PixelFormat::Depth32Float, SIDE))
        .unwrap();
    let before = overwritten(&device);
    run(&queue, &[depth_pass(&depth, 0.5)]);
    assert_eq!(
        overwritten(&device) - before,
        counted,
        "a clearing depth pass"
    );
    let bytes = read(&device, &queue, &depth);
    assert!(
        bytes.chunks(4).all(|texel| texel == 0.5f32.to_le_bytes()),
        "the depth clear"
    );
    // A copy into the whole extent.
    let copied = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    let before = overwritten(&device);
    let whole = Size::new(SIDE, SIDE, 1);
    copy_in(&device, &queue, &copied, Origin::new(0, 0, 0), whole);
    assert_eq!(overwritten(&device) - before, counted, "a whole copy");
    let bytes = read(&device, &queue, &copied);
    assert!(bytes.iter().all(|&b| b == 0x5a), "the copy");
    println!("FIRST-OVERWRITE whole on {}: {counted} each", device.name());
}

#[test]
fn a_recycled_texture_written_whole_first_reads_the_write_never_the_old_texels() {
    let scratch = Scratch::new("first-overwrite-recycled");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let first = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    run(&queue, &[color_pass(&first, LoadAction::Clear, 0.25)]);
    let old = read(&device, &queue, &first);
    let churn = device.work_counts().object_churn;
    drop(first);
    let until = Instant::now() + WAIT;
    while device.work_counts().object_churn.images_parked == churn.images_parked {
        assert!(Instant::now() < until, "the image was parked");
        std::thread::yield_now();
    }
    let second = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    let recycled = device.work_counts().object_churn.images_recycled - churn.images_recycled;
    assert_eq!(recycled, 1, "the second texture took the parked image");
    let before = overwritten(&device);
    run(&queue, &[color_pass(&second, LoadAction::Clear, 1.0)]);
    let counted = overwritten(&device) - before;
    assert_eq!(counted, u64::from(zero_fills(&device)));
    let bytes = read(&device, &queue, &second);
    assert_ne!(bytes, old, "the old texels");
    assert!(bytes.iter().all(|&b| b == 0xff), "the clear");
}

#[test]
fn a_new_texture_read_or_written_in_part_first_takes_its_zeros() {
    let scratch = Scratch::new("first-overwrite-part");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let fills = zero_fills(&device);
    for load in [LoadAction::DontCare, LoadAction::Load] {
        let texture = device
            .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
            .unwrap();
        let before = overwritten(&device);
        run(&queue, &[color_pass(&texture, load, 1.0)]);
        assert_eq!(overwritten(&device) - before, 0, "{load:?}");
        let bytes = read(&device, &queue, &texture);
        if fills {
            assert!(bytes.iter().all(|&b| b == 0), "{load:?}: zero");
        }
    }
    // A copy into a part: zero past it.
    let texture = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    let before = overwritten(&device);
    let half = Size::new(SIDE, SIDE / 2, 1);
    copy_in(&device, &queue, &texture, Origin::new(0, 0, 0), half);
    assert_eq!(overwritten(&device) - before, 0, "a copy into a part");
    let bytes = read(&device, &queue, &texture);
    let (top, bottom) = bytes.split_at(bytes.len() / 2);
    assert!(top.iter().all(|&b| b == 0x5a), "the copy");
    if fills {
        assert!(bottom.iter().all(|&b| b == 0), "zero past the copy");
    }
    // A clearing pass after a pass of another texture, whose commands may reach any texture.
    let other = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    run(&queue, &[color_pass(&other, LoadAction::Clear, 0.0)]);
    let texture = device
        .new_texture(&target(PixelFormat::RGBA8Unorm, SIDE))
        .unwrap();
    let before = overwritten(&device);
    let passes = [
        color_pass(&other, LoadAction::Clear, 0.0),
        color_pass(&texture, LoadAction::Clear, 1.0),
    ];
    run(&queue, &passes);
    assert_eq!(overwritten(&device) - before, 0, "after another pass");
    let bytes = read(&device, &queue, &texture);
    assert!(bytes.iter().all(|&b| b == 0xff), "the clear");
}

#[test]
#[ignore = "a measurement, not a check: run by name with --ignored"]
fn first_overwrite_costs() {
    const WARM: usize = 3;
    const SAMPLES: usize = 15;
    let scratch = Scratch::new("first-overwrite-cost");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let shapes = [
        ("RGBA8Unorm 1024²", PixelFormat::RGBA8Unorm, 1024, 8),
        ("Depth32Float 2048²", PixelFormat::Depth32Float, 2048, 2),
    ];
    let clear = |texture: &Texture, pf: PixelFormat| {
        let pass = match pf {
            PixelFormat::Depth32Float => depth_pass(texture, 0.5),
            _ => color_pass(texture, LoadAction::Clear, 1.0),
        };
        let command_buffer = queue.command_buffer().unwrap();
        (command_buffer.render_command_encoder(&pass).unwrap())
            .end_encoding()
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    for (name, pf, side, count) in shapes {
        for together in [true, false] {
            let mut spans = Vec::new();
            let before = overwritten(&device);
            for _ in 0..WARM + SAMPLES {
                let (began, textures, committed) = match together {
                    true => {
                        let textures: Vec<Texture> = (0..count)
                            .map(|_| device.new_texture(&target(pf, side)).unwrap())
                            .collect();
                        let began = Instant::now();
                        let committed: Vec<_> = textures.iter().map(|t| clear(t, pf)).collect();
                        (began, textures, committed)
                    }
                    false => {
                        let began = Instant::now();
                        let (textures, committed): (Vec<Texture>, Vec<_>) = (0..count)
                            .map(|_| {
                                let texture = device.new_texture(&target(pf, side)).unwrap();
                                let command_buffer = clear(&texture, pf);
                                (texture, command_buffer)
                            })
                            .unzip();
                        (began, textures, committed)
                    }
                };
                for command_buffer in &committed {
                    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
                }
                spans.push(began.elapsed().as_secs_f64() * 1e3);
                drop(textures);
            }
            let spans = &mut spans[WARM..];
            spans.sort_by(f64::total_cmp);
            let (made, from) = match together {
                true => ("made together", "first commit"),
                false => ("each made just before", "first making"),
            };
            println!(
                "OVERWRITE-COST {count} new {name} {made}, each cleared by a command buffer of its \
                 own, on {}: {from} to last completion median {:.3} ms, fastest {:.3} ms; {} of {} \
                 went unzeroed",
                device.name(),
                spans[SAMPLES / 2],
                spans[0],
                overwritten(&device) - before,
                count * (WARM + SAMPLES),
            );
        }
    }
}
