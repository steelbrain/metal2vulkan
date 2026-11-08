//! Draws that read the texture of a color attachment of their own pass through its copy
//! (`texture/attached_reads.rs`), kept to draw again (`render/draw_again.rs`): the shadowed
//! textures a kept draw used are gathered by its instance, so a draw again, or a draw rebound that
//! uses the same shadows each as the kept draw used it, gathers nothing new, and a draw rebound
//! whose shadows differ takes every step.
//!
//! One pass over two private `BGRA8Unorm` targets, both loaded and stored, both cleared before by
//! a pass of their own to a red of their own, draws a rectangle with `tob_read` (each fragment
//! writes into the first target the texel of its slot-0 texture under it): the first target's
//! texels over [`FIRST_RECT`], three times (the pass's first draw, kept nothing by its instance;
//! the next, kept; the next, drawn again); over [`SECOND_RECT`], its vertex bytes bound again (a
//! draw rebound, its shadow the kept draw's); the second target's texels there (its shadow
//! another: it takes every step); through the scissor [`SCISSOR`] (its read's bounds another,
//! where the device bounds a read at its own pixels: it takes every step); and twice again. The
//! same frame runs again with a call that keeps no draw before each draw but the first
//! (`setTessellationFactorScale:`, which no draw of the pass reads), so each takes every step.
//!
//! Both frames answer as Metal does, the first target's texels from before the pass where its
//! own were read and the second's over [`SECOND_RECT`], with the same copies (the shadow copies,
//! the texels they move, the reads through copies and at the fragments' own pixels, the splits);
//! and the first frame's draws are kept, none for its shadows, so its draws again and rebound are
//! counted where the second frame has none.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect` and
//! `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::{
    BlitOption, ClearColor, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The targets' side, in pixels, and format.
const SIDE: u64 = 32;
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The reds the targets are cleared to.
const FIRST_RED: u8 = 0x40;
const SECOND_RED: u8 = 0xc0;

/// The rectangles the draws cover, `[x0, y0, x1, y1]`.
const FIRST_RECT: [u64; 4] = [0, 0, 16, 16];
const SECOND_RECT: [u64; 4] = [8, 4, 24, 20];

/// The scissor of the last draws, inside [`SECOND_RECT`].
const SCISSOR: ScissorRect = ScissorRect {
    x: 12,
    y: 8,
    width: 8,
    height: 8,
};

/// The state of `tob_rect` and `tob_read` into two colors of [`FORMAT`].
fn state(device: &Device) -> RenderPipelineState {
    let function = |name: &str| {
        let function = library(device, "engine/tob_renders.ll")
            .new_function_with_name(name)
            .unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_read"));
    for color in &mut descriptor.color_attachments[..2] {
        *color = Some(ColorAttachmentDescriptor::new(FORMAT));
    }
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A private [`FORMAT`] render target that shaders read.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The corners, in normalized device coordinates, of the targets' pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    bytes_of(&[x(x0), y(y1), x(x1), y(y0)], f32::to_le_bytes)
}

/// A pass descriptor over `targets`, each loaded `load` (cleared to its red) and stored.
fn over(targets: &[Texture; 2], load: LoadAction) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    for ((color, target), red) in (descriptor.color_attachments.iter_mut())
        .zip(targets)
        .zip([FIRST_RED, SECOND_RED])
    {
        color.texture = Some(target.clone());
        color.load_action = load;
        color.clear_color = ClearColor::new(f64::from(red) / 255.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
    }
    descriptor
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
    // `BGRA8Unorm`'s red is its third byte.
    texels.chunks(4).map(|texel| texel[2]).collect()
}

/// The frame over two new targets (the module docs), each draw but the first after a call that
/// keeps no draw where `apart`: the device's counts before and after the pass, and the first
/// target's reds and the second's after it.
fn frame(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    apart: bool,
) -> ((WorkCounts, WorkCounts), [Vec<u8>; 2]) {
    let device = queue.device();
    let targets = [target(device), target(device)];
    let command_buffer = queue.command_buffer().unwrap();
    let clears = command_buffer.render_command_encoder(&over(&targets, LoadAction::Clear));
    clears.unwrap().end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let pass = over(&targets, LoadAction::Load);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    let mut drawn = 0;
    let mut draw = |encoder: &mut crate::RenderCommandEncoder| {
        if apart && drawn > 0 {
            encoder.set_tessellation_factor_scale(1.0);
        }
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
        drawn += 1;
    };
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&corners(FIRST_RECT), 0).unwrap();
    (encoder.set_fragment_texture(Some(&targets[0]), 0)).unwrap();
    for _ in 0..3 {
        draw(&mut encoder);
    }
    encoder.set_vertex_bytes(&corners(SECOND_RECT), 0).unwrap();
    draw(&mut encoder);
    (encoder.set_fragment_texture(Some(&targets[1]), 0)).unwrap();
    draw(&mut encoder);
    encoder.set_scissor_rect(SCISSOR).unwrap();
    for _ in 0..3 {
        draw(&mut encoder);
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let reds = targets.each_ref().map(|target| reds(queue, target));
    ((before, after), reds)
}

/// What a frame's pass moved of the counts between `before` and `after`.
#[derive(Debug, PartialEq, Eq)]
struct Copies {
    /// `WorkCounts::shadow_copies`, `shadow_texels_copied_in`, `attachment_reads_through_copies`,
    /// `own_pixel_reads`, `attachment_read_splits` and `render_passes_begun_inline`.
    shadow_copies: u64,
    copied_in: u64,
    through_copies: u64,
    own_pixel: u64,
    splits: u64,
    inline: u64,
}

/// What a frame's pass kept, by the same counts: `draws_unkept`, `draws_unkept_shadowed`,
/// `draws_as_before`, `draws_rebound` and `draws_rebind_stopped`.
#[derive(Debug, PartialEq, Eq)]
struct Keeps {
    unkept: u64,
    unkept_shadowed: u64,
    as_before: u64,
    rebound: u64,
    stopped: u64,
}

fn moved((before, after): &(WorkCounts, WorkCounts)) -> (Copies, Keeps) {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    let copies = Copies {
        shadow_copies: delta(|c| c.shadow_copies),
        copied_in: delta(|c| c.shadow_texels_copied_in),
        through_copies: delta(|c| c.attachment_reads_through_copies),
        own_pixel: delta(|c| c.own_pixel_reads),
        splits: delta(|c| c.attachment_read_splits),
        inline: delta(|c| c.render_passes_begun_inline),
    };
    let keeps = Keeps {
        unkept: delta(|c| c.draws_unkept),
        unkept_shadowed: delta(|c| c.draws_unkept_shadowed),
        as_before: delta(|c| c.draws_as_before),
        rebound: delta(|c| c.draws_rebound),
        stopped: delta(|c| c.draws_rebind_stopped),
    };
    (copies, keeps)
}

/// The first target's reds after the frame: its own from before the pass, but the second
/// target's over [`SECOND_RECT`], which the draws of its texels covered last there.
fn first_reds() -> Vec<u8> {
    let [x0, y0, x1, y1] = SECOND_RECT;
    (0..SIDE * SIDE)
        .map(|at| {
            let (x, y) = (at % SIDE, at / SIDE);
            match (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                true => SECOND_RED,
                false => FIRST_RED,
            }
        })
        .collect()
}

#[test]
fn draws_that_read_their_pass_s_copies_are_kept_with_the_copies_of_every_step() {
    let scratch = Scratch::new("kept-shadowed-draws");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let (kept_counts, kept_reds) = frame(&queue, &state, false);
    let (apart_counts, apart_reds) = frame(&queue, &state, true);
    let want = [first_reds(), vec![SECOND_RED; (SIDE * SIDE) as usize]];
    for (frame, reds) in [("kept", &kept_reds), ("apart", &apart_reds)] {
        for (target, (reds, want)) in reds.iter().zip(&want).enumerate() {
            if let Some(at) = (0..reds.len()).find(|&at| reds[at] != want[at]) {
                let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
                panic!(
                    "{frame} frame: target {target}'s pixel ({x}, {y}) holds red {}, not {}",
                    reds[at], want[at]
                );
            }
        }
    }
    let (kept_copies, kept) = moved(&kept_counts);
    let (apart_copies, apart) = moved(&apart_counts);
    eprintln!(
        "KEPT-SHADOWED-DRAWS {}: kept {kept_copies:?} {kept:?}; apart {apart_copies:?} {apart:?}",
        device.name()
    );
    // The same copies, every draw a read through a copy.
    assert_eq!(kept_copies, apart_copies, "the kept frame's copies");
    assert_eq!(kept_copies.through_copies, 8, "the reads through copies");
    // The pass's first draw keeps nothing, by its instance alone. A draw again, or rebound, the
    // first after a draw kept and every second after it takes every step in builds with debug
    // assertions (`render/draw_again.rs`): of the draws again after the third draw kept, the
    // first; of those after the last, the first. The scissor's draw is stopped where the device
    // bounds the reads at their own pixels, as its read's bounds change, else rebound.
    let bounded = kept_copies.own_pixel > 0;
    let debug = cfg!(debug_assertions);
    let want = Keeps {
        unkept: 1,
        unkept_shadowed: 0,
        as_before: match debug {
            true => 1,
            false => 3,
        },
        rebound: 1 + u64::from(!bounded && !debug),
        stopped: 1 + u64::from(bounded),
    };
    assert_eq!(kept, want, "the kept frame's keeps on {}", device.name());
    let apart_want = Keeps {
        unkept: 1,
        unkept_shadowed: 0,
        as_before: 0,
        rebound: 0,
        stopped: 0,
    };
    assert_eq!(apart, apart_want, "the apart frame's keeps");
}
