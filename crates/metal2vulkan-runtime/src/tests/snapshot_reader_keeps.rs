//! A reader through snapshots (`render/snapshot.rs`) keeps nothing, and forgets only the kept
//! draws that read what it wrote, its pass's colors and its snapshots (`render/draw_again.rs`,
//! past a reader through snapshots): a kept draw of another pipeline state that reads none of it
//! stays kept, and its next draw is rebound; one that reads the reader's color is forgotten.
//!
//! One pass over a [`SIDE`]-square private `BGRA8Unorm` target, loaded and stored, which a copy
//! before it fills with [`CLEARED`] but [`PRE`] over [`SAMPLED`], with no scissor, draws in turn:
//!
//! 1. the reader (`tob_read`: each fragment writes the texel of its slot-0 texture, the target's,
//!    under it) over [`FIRST_READ`], which, through the whole target's scissor, takes the
//!    snapshot by the area rule on a device that takes the snapshot path;
//! 2. the fill (`tob_solid`) of [`FILL`] with [`FILLED`], kept;
//! 3. the sampler (`tob_sample`: each fragment samples the target's texture at its own position)
//!    over [`SAMPLED`], which reads the texture's copy from before its instance and is kept: a
//!    kept draw that reads the reader's color;
//! 4. the reader over [`SECOND_READ`], in its instance as it stood: it forgets the sampler's kept
//!    draw and keeps the fill's;
//! 5. the fill: its kept draw past the reader, rebound;
//! 6. the sampler: forgotten, it takes every step and is kept again;
//! 7. the fill: rebound after the sampler's draw.
//!
//! The same frame runs again with a call that keeps no draw before each draw but the first
//! (`setTessellationFactorScale:`, which no draw of the pass reads), so each takes every step.
//!
//! Both frames answer alike: [`FILLED`] over [`FILL`], drawn last there; [`PRE`] over the rest of
//! [`SAMPLED`], the texels from before the pass the sampler reads; [`CLEARED`] elsewhere, the
//! readers writing back the texels they read, which no draw before them wrote. On a device that
//! takes the snapshot path, both readers read through snapshots, and of the first frame's draws
//! five find nothing kept (the first four and the sampler's second) where the second frame's seven
//! do; the fill's two draws after the reader are rebound, each taking every step instead in
//! builds with debug assertions and checking that the rebind worked out what every step works out
//! (`rebound.rs`). A device without the path reads through copies, which keeps the readers.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid`, `tob_read` and `tob_sample` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::through_storage;
use crate::{
    BlitOption, CommandQueue, Device, LoadAction, Origin, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::time::{Duration, Instant};

/// The target's side, in pixels: through the whole target's scissor, a reader of a 4-byte color
/// takes the snapshot by the area rule.
const SIDE: u64 = 1024;

/// The target's format.
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The rectangles the draws cover, `[x0, y0, x1, y1]`: the readers', apart from every other.
const FIRST_READ: [u64; 4] = [600, 600, 664, 664];
const SECOND_READ: [u64; 4] = [700, 700, 764, 764];
const FILL: [u64; 4] = [100, 100, 300, 300];
const SAMPLED: [u64; 4] = [200, 200, 400, 400];

/// The reds: the target's outside [`SAMPLED`] and inside it before the pass, and the fill's.
const CLEARED: u8 = 20;
const PRE: u8 = 160;
const FILLED: u8 = 90;

/// The state of `tob_rect` and `tob_renders.ll`'s `fragment`, into a color of [`FORMAT`].
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let function = |name: &str| {
        let function = library(device, "engine/tob_renders.ll")
            .new_function_with_name(name)
            .unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The pipeline states: the reader, the fill and the sampler.
struct States {
    read: RenderPipelineState,
    fill: RenderPipelineState,
    sample: RenderPipelineState,
}

/// A private [`FORMAT`] render target that shaders read.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    bytes_of(&[x(x0), y(y1), x(x1), y(y0)], f32::to_le_bytes)
}

/// Whether `rect` holds pixel `(x, y)`.
fn holds([x0, y0, x1, y1]: [u64; 4], (x, y): (u64, u64)) -> bool {
    (x0..x1).contains(&x) && (y0..y1).contains(&y)
}

/// The target's red at each pixel before the pass: [`PRE`] over [`SAMPLED`], else [`CLEARED`].
fn before_reds() -> Vec<u8> {
    (0..SIDE * SIDE)
        .map(|at| match holds(SAMPLED, (at % SIDE, at / SIDE)) {
            true => PRE,
            false => CLEARED,
        })
        .collect()
}

/// The target's red at each pixel after the pass (the module docs).
fn after_reds() -> Vec<u8> {
    let before = before_reds();
    (0..SIDE * SIDE)
        .map(|at| match holds(FILL, (at % SIDE, at / SIDE)) {
            true => FILLED,
            false => before[at as usize],
        })
        .collect()
}

/// Fill `target` with [`before_reds`], a command buffer of its own, completed.
fn fill_before(queue: &CommandQueue, target: &Texture) {
    let row = SIDE * 4;
    // `BGRA8Unorm`'s red is its third byte.
    let texels: Vec<u8> = (before_reds().into_iter())
        .flat_map(|red| [0, 0, red, 255])
        .collect();
    let source = (queue.device())
        .new_buffer_with_bytes(&texels, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_texture(
        &source,
        0,
        row,
        0,
        Size::new(SIDE, SIDE, 1),
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The red of each of `texture`'s pixels, by row.
fn reds(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let row = SIDE * 4;
    let back = (queue.device())
        .new_buffer(SIDE * row, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        row,
        SIDE * row,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let texels = super::merge_scratch::read(&back, SIDE * row);
    texels.chunks(4).map(|texel| texel[2]).collect()
}

/// The frame over a new target (the module docs), each draw but the first after a call that keeps
/// no draw where `apart`: the device's counts before and after the pass, and the target's reds
/// after it. Into a queue of its own, which no earlier frame's kept draw is carried along
/// (`render/carried.rs`).
fn frame(
    device: &Device,
    states: &States,
    sampler: &SamplerState,
    apart: bool,
) -> ((WorkCounts, WorkCounts), Vec<u8>) {
    let queue = &device.new_command_queue().unwrap();
    let target = target(device);
    fill_before(queue, &target);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    (encoder.set_fragment_sampler_state(Some(sampler), 0)).unwrap();
    let filled = f32::from(FILLED) / 255.0;
    let rgba = bytes_of(&[filled, 0.0, 0.0, 1.0], f32::to_le_bytes);
    encoder.set_fragment_bytes(&rgba, 0).unwrap();
    let mut drawn = 0;
    let mut draw =
        |encoder: &mut RenderCommandEncoder, state: &RenderPipelineState, rect: [u64; 4]| {
            encoder.set_render_pipeline_state(state).unwrap();
            encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
            if apart && drawn > 0 {
                encoder.set_tessellation_factor_scale(1.0);
            }
            (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
            drawn += 1;
        };
    draw(&mut encoder, &states.read, FIRST_READ);
    draw(&mut encoder, &states.fill, FILL);
    draw(&mut encoder, &states.sample, SAMPLED);
    draw(&mut encoder, &states.read, SECOND_READ);
    draw(&mut encoder, &states.fill, FILL);
    draw(&mut encoder, &states.sample, SAMPLED);
    draw(&mut encoder, &states.fill, FILL);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    ((before, after), reds(queue, &target))
}

/// What a frame's pass moved of the counts between `before` and `after`.
#[derive(Debug, PartialEq, Eq)]
struct Keeps {
    /// `WorkCounts::reading_draws_snapshotted`, `draws_none_kept`, `draws_rebound`,
    /// `draws_rebind_stopped`, `draws_unlike_kept` and `draws_unkept`.
    snapshotted: u64,
    none_kept: u64,
    rebound: u64,
    stopped: u64,
    unlike: u64,
    unkept: u64,
}

fn moved((before, after): &(WorkCounts, WorkCounts)) -> Keeps {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    Keeps {
        snapshotted: delta(|c| c.reading_draws_snapshotted),
        none_kept: delta(|c| c.draws_none_kept),
        rebound: delta(|c| c.draws_rebound),
        stopped: delta(|c| c.draws_rebind_stopped),
        unlike: delta(|c| c.draws_unlike_kept),
        unkept: delta(|c| c.draws_unkept),
    }
}

/// Wait until the device's pipeline pool makes nothing.
fn settled(device: &Device) {
    let until = Instant::now() + WAIT;
    while device.work_counts().pipeline_pool_busy != 0 {
        assert!(Instant::now() < until, "the pipeline pool still busy");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_reader_through_snapshots_forgets_only_the_kept_draws_that_read_its_color() {
    let scratch = Scratch::new("snapshot-reader-keeps");
    let device = device(&scratch.0);
    let states = States {
        read: state(&device, "tob_read"),
        fill: state(&device, "tob_solid"),
        sample: state(&device, "tob_sample"),
    };
    let sampler = (device.new_sampler_state(&SamplerDescriptor::new())).unwrap();
    // A frame first, so every pipeline its draws take is made before the frames counted.
    frame(&device, &states, &sampler, false);
    settled(&device);
    let (kept_counts, kept_reds) = frame(&device, &states, &sampler, false);
    let (apart_counts, apart_reds) = frame(&device, &states, &sampler, true);
    let want = after_reds();
    for (frame, reds) in [("kept", &kept_reds), ("apart", &apart_reds)] {
        if let Some(at) = (0..reds.len()).find(|&at| reds[at] != want[at]) {
            let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
            panic!(
                "{frame} frame: pixel ({x}, {y}) holds red {}, not {}",
                reds[at], want[at]
            );
        }
    }
    let (kept, apart) = (moved(&kept_counts), moved(&apart_counts));
    let snapshots = device.shared().color_write_masks.is_some()
        && through_storage(device.compile_device().profile());
    eprintln!(
        "SNAPSHOT-READER-KEEPS {}: snapshot path {snapshots}; kept {kept:?}; apart {apart:?}",
        device.name()
    );
    if !snapshots {
        assert_eq!(kept.snapshotted, 0, "the readers through snapshots");
        return;
    }
    // Five draws find nothing kept: the first four, and the sampler's second, which the second
    // reader forgot. The fill's two draws after the reader are rebound, or in builds with debug
    // assertions take every step to check the rebind (each the first after a draw kept).
    let debug = cfg!(debug_assertions);
    let want = Keeps {
        snapshotted: 2,
        none_kept: 5,
        rebound: u64::from(!debug) * 2,
        stopped: 0,
        unlike: 0,
        unkept: 2,
    };
    assert_eq!(kept, want, "the kept frame's keeps on {}", device.name());
    let apart_want = Keeps {
        snapshotted: 2,
        none_kept: 7,
        rebound: 0,
        stopped: 0,
        unlike: 0,
        unkept: 2,
    };
    assert_eq!(
        apart,
        apart_want,
        "the apart frame's keeps on {}",
        device.name()
    );
}
