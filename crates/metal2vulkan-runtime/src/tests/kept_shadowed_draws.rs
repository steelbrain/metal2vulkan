//! Draws that read the texture of a color attachment of their own pass through its copy
//! (`texture/attached_reads.rs`), kept to draw again (`render/draw_again.rs`): the shadowed
//! textures a kept draw used are gathered by its instance, so a draw again, or a draw rebound that
//! uses the same shadows each as the kept draw used it, gathers nothing new, and a draw rebound
//! whose shadows differ gathers its own into its instance, as every step does
//! (`render/draw_again/rebound.rs`).
//!
//! One pass over two private `BGRA8Unorm` targets, both loaded and stored, both cleared before by
//! a pass of their own to a red of their own, draws a rectangle with `tob_read` (each fragment
//! writes into the first target the texel of its slot-0 texture under it): the first target's
//! texels over [`FIRST_RECT`], three times (the pass's first draw, which begins its instance,
//! kept; the next two, drawn again); over [`SECOND_RECT`], its vertex bytes bound again (a
//! draw rebound, its shadow the kept draw's); the second target's texels there (a draw rebound,
//! its shadow another, which it gathers); through the scissor [`SCISSOR`] (a draw rebound, its
//! read's bounds another where the device bounds a read at its own pixels, which it gathers);
//! and twice again. The same frame runs again, into a queue of its own, with a call that keeps no
//! draw before each draw but the first (`setTessellationFactorScale:`, which no draw of the pass
//! reads), so each takes every step.
//!
//! Both frames answer as Metal does, the first target's texels from before the pass where its own
//! were read and the second's over [`SECOND_RECT`], with the same commands recorded, of every kind
//! (`WorkCounts::commands_recorded`: the copies, barriers, draws and queries among them, its begins
//! the command buffers the frame began, `WorkCounts::command_buffers_begun`), the same
//! copies (the shadow copies, the texels they move, the reads through copies and at the fragments'
//! own pixels, the splits) and the same coverage queries (`WorkCounts::coverage_queries`: on a
//! device that weighs a reader of its pass's color, each draw of the first target's texels is one,
//! whichever form it reads through, kept or not, `render/coverage.rs`); and the first frame's draws
//! are kept, drawn again or rebound, none stopped, where the second frame has none. A lane of draws
//! of other arguments runs both frames again, each draw's base instance another than the draw's
//! before it (no function reads it): each draw after the first is rebound with its arguments
//! (`WorkCounts::draws_rebound_arguments`), none stopped.
//!
//! A third lane draws with `tob_sample` (each fragment writes into the first target the texel of
//! its slot-0 texture sampled at its pixel's center: a read of anywhere, no snapshot's): over
//! [`FIRST_RECT`] from a third private texture of the targets' shape, cleared to [`SOURCE_RED`],
//! kept; then, after a barrier over the render targets from the fragment stage to the fragment
//! stage, which keeps that draw (it reads no color), over [`SECOND_RECT`] from the first target,
//! through its copy, at another base instance. Every step takes the split there, whose copy holds
//! what the draw before the barrier wrote, so that draw rebound stops at its check
//! (`WorkCounts::draws_rebind_stopped_arguments_checks`), before its shadows are looked at, with
//! the same commands and texels as its frame of every step.
//!
//! A fourth lane orders the draws so that each draw rebound that gathers its own shadow use is
//! the last to use it so, where a draw that takes every step, as the debug builds' check takes
//! every second draw rebound (`render/draw_again.rs`), gathers nothing of it after: draws of
//! other arguments under the scissor [`FIRST_SCISSOR`], the first three of the first target's
//! texels over [`FIRST_RECT`], the fourth over [`SECOND_RECT`]; the fifth of the second target's
//! texels (another shadow, its copy taken by no other draw); the sixth of the first target's
//! again under [`RIGHT_SCISSOR`]; the seventh over [`THIRD_RECT`] under [`CORNER_SCISSOR`], whose
//! read at its own pixels lies outside every bound the draws before it read within (the copy in
//! of the first target moves only the texels those bounds hold where the device cuts it). The
//! kept frame's copies, commands and texels are its frame of every step's. On a device that reads
//! a color's texture in place the fifth draw reads the second target in place, so the lane is left
//! out there, and says so.
//!
//! On a device that reads a color's texture in place (`render/snapshot.rs`'s in place: it enabled
//! `attachmentFeedbackLoopLayout`, and its passes fetch no color through input attachments,
//! `framebuffer_fetch::through_inputs`), the draws of the second target's texels, which write
//! nothing of it, read it in place instead, with no copy (`WorkCounts::reading_draws_in_place`),
//! and keep nothing (`WorkCounts::draws_unkept_shadowed`): the first stops its rebind (it would
//! read in place, `RenderLowering::reads_copy`), and the draw after them finds none kept.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_read` and `tob_sample` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{plan_device, Scratch, WAIT};
use super::draw_again::{checks_ended, commands_of};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::through_inputs;
use crate::{
    BlitOption, ClearColor, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeMap;

/// The targets' side, in pixels, and format.
const SIDE: u64 = 32;
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The reds the targets are cleared to.
const FIRST_RED: u8 = 0x40;
const SECOND_RED: u8 = 0xc0;

/// The red the third lane's sampled texture is cleared to.
const SOURCE_RED: u8 = 0x80;

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

/// The fourth lane's third rectangle, and its scissors: over [`FIRST_RECT`], right of it, and the
/// corner below and right of both (the module docs).
const THIRD_RECT: [u64; 4] = [16, 16, 32, 32];
const FIRST_SCISSOR: ScissorRect = ScissorRect {
    x: 0,
    y: 0,
    width: 16,
    height: 16,
};
const RIGHT_SCISSOR: ScissorRect = ScissorRect {
    x: 16,
    y: 4,
    width: 8,
    height: 16,
};
const CORNER_SCISSOR: ScissorRect = ScissorRect {
    x: 24,
    y: 24,
    width: 8,
    height: 8,
};

/// The state of `tob_rect` and the fragment function `fragment` into two colors of [`FORMAT`].
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let function = |name: &str| {
        let function = library(device, "engine/tob_renders.ll")
            .new_function_with_name(name)
            .unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function(fragment));
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

/// What a frame did: the device's counts before and after its pass, and the first target's reds
/// and the second's after it.
type Frame = ((WorkCounts, WorkCounts), [Vec<u8>; 2]);

/// A pass over two new targets, each cleared to its red by a pass before, loaded, its draws
/// encoded by `encode`: what it did.
fn frame_of(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut RenderCommandEncoder, &[Texture; 2]),
) -> Frame {
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
    encode(&mut encoder, &targets);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    let reds = targets.each_ref().map(|target| reds(queue, target));
    ((before, after), reds)
}

/// A draw of a frame (the module docs), its `drawn`th: after a call that keeps no draw where
/// `apart` and it is not the first, of base instance `drawn % 2` where `argued`.
fn draw(encoder: &mut RenderCommandEncoder, drawn: &mut u32, (apart, argued): (bool, bool)) {
    if apart && *drawn > 0 {
        encoder.set_tessellation_factor_scale(1.0);
    }
    let base = match argued {
        true => *drawn % 2,
        false => 0,
    };
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, base)).unwrap();
    *drawn += 1;
}

/// The frame of `tob_read`'s draws (the module docs), each but the first after a call that keeps
/// no draw where `apart`, each of a base instance other than the draw's before it where `argued`.
fn frame(queue: &CommandQueue, state: &RenderPipelineState, drawn_so: (bool, bool)) -> Frame {
    frame_of(queue, |encoder, targets| {
        let mut drawn = 0;
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_vertex_bytes(&corners(FIRST_RECT), 0).unwrap();
        (encoder.set_fragment_texture(Some(&targets[0]), 0)).unwrap();
        for _ in 0..3 {
            draw(encoder, &mut drawn, drawn_so);
        }
        encoder.set_vertex_bytes(&corners(SECOND_RECT), 0).unwrap();
        draw(encoder, &mut drawn, drawn_so);
        (encoder.set_fragment_texture(Some(&targets[1]), 0)).unwrap();
        draw(encoder, &mut drawn, drawn_so);
        encoder.set_scissor_rect(SCISSOR).unwrap();
        for _ in 0..3 {
            draw(encoder, &mut drawn, drawn_so);
        }
    })
}

/// The fourth lane's frame of `tob_read`'s draws of other arguments (the module docs), each but
/// the first after a call that keeps no draw where `apart`.
fn gathers_frame(queue: &CommandQueue, state: &RenderPipelineState, apart: bool) -> Frame {
    frame_of(queue, |encoder, targets| {
        let (mut drawn, drawn_so) = (0, (apart, true));
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_scissor_rect(FIRST_SCISSOR).unwrap();
        encoder.set_vertex_bytes(&corners(FIRST_RECT), 0).unwrap();
        (encoder.set_fragment_texture(Some(&targets[0]), 0)).unwrap();
        for _ in 0..3 {
            draw(encoder, &mut drawn, drawn_so);
        }
        encoder.set_vertex_bytes(&corners(SECOND_RECT), 0).unwrap();
        draw(encoder, &mut drawn, drawn_so);
        (encoder.set_fragment_texture(Some(&targets[1]), 0)).unwrap();
        draw(encoder, &mut drawn, drawn_so);
        (encoder.set_fragment_texture(Some(&targets[0]), 0)).unwrap();
        encoder.set_scissor_rect(RIGHT_SCISSOR).unwrap();
        draw(encoder, &mut drawn, drawn_so);
        encoder.set_vertex_bytes(&corners(THIRD_RECT), 0).unwrap();
        encoder.set_scissor_rect(CORNER_SCISSOR).unwrap();
        draw(encoder, &mut drawn, drawn_so);
    })
}

/// The frame of `tob_sample`'s draws across a barrier (the module docs), the first from `source`
/// through `sampler`, the second after a call that keeps no draw where `apart`.
fn barrier_frame(
    queue: &CommandQueue,
    state: &RenderPipelineState,
    (source, sampler): (&Texture, &SamplerState),
    apart: bool,
) -> Frame {
    frame_of(queue, |encoder, targets| {
        let mut drawn = 0;
        encoder.set_render_pipeline_state(state).unwrap();
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
        encoder.set_vertex_bytes(&corners(FIRST_RECT), 0).unwrap();
        (encoder.set_fragment_texture(Some(source), 0)).unwrap();
        draw(encoder, &mut drawn, (apart, true));
        let fragment = RenderStages::FRAGMENT;
        encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
        encoder.set_vertex_bytes(&corners(SECOND_RECT), 0).unwrap();
        (encoder.set_fragment_texture(Some(&targets[0]), 0)).unwrap();
        draw(encoder, &mut drawn, (apart, true));
    })
}

/// What a frame's pass moved of the counts between `before` and `after`.
#[derive(Debug, PartialEq, Eq)]
struct Copies {
    /// `WorkCounts::shadow_copies`, `shadow_texels_copied_in`, `shadow_texels_copied_in_images`,
    /// `attachment_reads_through_copies`, `own_pixel_reads`, `attachment_read_splits`,
    /// `render_passes_begun_inline`, `reading_draws_in_place` and `coverage_queries`.
    shadow_copies: u64,
    copied_in: u64,
    copied_in_images: u64,
    through_copies: u64,
    own_pixel: u64,
    splits: u64,
    inline: u64,
    in_place: u64,
    queries: u64,
}

/// What a frame's pass kept, by the same counts: `draws_unkept`, `draws_unkept_shadowed`,
/// `draws_as_before`, `draws_rebound` and `draws_rebind_stopped`; and of draws of other
/// arguments, `draws_rebound_arguments`, `draws_rebind_stopped_arguments` and of those
/// `draws_rebind_stopped_arguments_shadows` and `draws_rebind_stopped_arguments_checks`.
#[derive(Debug, PartialEq, Eq)]
struct Keeps {
    unkept: u64,
    unkept_shadowed: u64,
    as_before: u64,
    rebound: u64,
    stopped: u64,
    rebound_arguments: u64,
    stopped_arguments: u64,
    stopped_shadows: u64,
    stopped_checks: u64,
}

fn moved((before, after): &(WorkCounts, WorkCounts)) -> (Copies, Keeps) {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    let copies = Copies {
        shadow_copies: delta(|c| c.shadow_copies),
        copied_in: delta(|c| c.shadow_texels_copied_in),
        copied_in_images: delta(|c| c.shadow_texels_copied_in_images),
        through_copies: delta(|c| c.attachment_reads_through_copies),
        own_pixel: delta(|c| c.own_pixel_reads),
        splits: delta(|c| c.attachment_read_splits),
        inline: delta(|c| c.render_passes_begun_inline),
        in_place: delta(|c| c.reading_draws_in_place),
        queries: delta(|c| c.coverage_queries),
    };
    let keeps = Keeps {
        unkept: delta(|c| c.draws_unkept),
        unkept_shadowed: delta(|c| c.draws_unkept_shadowed),
        as_before: delta(|c| c.draws_as_before),
        rebound: delta(|c| c.draws_rebound),
        stopped: delta(|c| c.draws_rebind_stopped),
        rebound_arguments: delta(|c| c.draws_rebound_arguments),
        stopped_arguments: delta(|c| c.draws_rebind_stopped_arguments),
        stopped_shadows: delta(|c| c.draws_rebind_stopped_arguments_shadows),
        stopped_checks: delta(|c| c.draws_rebind_stopped_arguments_checks),
    };
    (copies, keeps)
}

/// The commands a frame recorded, by kind (`WorkCounts::commands_recorded`), its begins the
/// `VkCommandBuffer`s it began (`WorkCounts::command_buffers_begun`, which counts a primary begun
/// ahead in a spare command pool as the encoder that takes it records into it). Not the device's
/// count of `vkBeginCommandBuffer`, which takes such a primary as it is begun: the device's
/// begin-ahead thread begins it once a command buffer gave its pool back (`begin_ahead.rs`),
/// whenever that completion comes. So one begun ahead for the frame before, or
/// after the targets' clears, may land in either frame's window, and the frame's encoder may take
/// one begun before its window opened.
fn recorded(counts: &(WorkCounts, WorkCounts)) -> BTreeMap<String, u64> {
    let (before, after) = counts;
    let begun = after.command_buffers_begun - before.command_buffers_begun;
    let before = commands_of(&before.commands_recorded);
    let after = commands_of(&after.commands_recorded);
    let recorded: BTreeMap<String, u64> = (after.into_iter())
        .map(|(kind, count)| match kind == "begin_command_buffer" {
            true => (kind, begun),
            false => {
                let was = before.get(&kind).copied().unwrap_or(0);
                (kind, count - was)
            }
        })
        .filter(|&(_, count)| count > 0)
        .collect();
    let [calls, ahead, taken] = begins(counts);
    eprintln!(
        "KEPT-SHADOWED-BEGINS: {begun} begun, {calls} vkBeginCommandBuffer calls, {ahead} primaries \
         begun ahead, {taken} taken by encoders"
    );
    recorded
}

/// A frame's `vkBeginCommandBuffer` calls (`CommandsRecorded::begin_command_buffer`), and the
/// primaries begun ahead in spare command pools (`WorkCounts::primaries_begun_ahead`) and taken by
/// encoders (`WorkCounts::encoder_primaries_begun_ahead`) between its counts: what [`recorded`]
/// counts its begins apart from.
fn begins((before, after): &(WorkCounts, WorkCounts)) -> [u64; 3] {
    let calls = |counts: &WorkCounts| {
        let recorded = commands_of(&counts.commands_recorded);
        recorded.get("begin_command_buffer").copied().unwrap_or(0)
    };
    [
        calls(after) - calls(before),
        after.primaries_begun_ahead - before.primaries_begun_ahead,
        after.encoder_primaries_begun_ahead - before.encoder_primaries_begun_ahead,
    ]
}

/// The counts of draws rebound or stopped (`WorkCounts::draws_rebind*`, `draws_rebound*`) a
/// frame's pass moved, by name, for its printed line.
fn rebinds_moved((before, after): &(WorkCounts, WorkCounts)) -> BTreeMap<String, u64> {
    let fields = |counts: &WorkCounts| -> BTreeMap<String, u64> {
        let text = format!("{counts:?}");
        (text.split([',', '{']))
            .filter_map(|field| {
                let (name, count) = field.split_once(':')?;
                let name = name.trim();
                if !(name.starts_with("draws_rebind") || name.starts_with("draws_rebound")) {
                    return None;
                }
                Some((name.to_owned(), count.trim().parse().ok()?))
            })
            .collect()
    };
    let before = fields(before);
    (fields(after).into_iter())
        .map(|(name, count)| {
            let was = before.get(&name).copied().unwrap_or(0);
            (name, count - was)
        })
        .filter(|&(_, count)| count > 0)
        .collect()
}

/// Whether `device` reads a color's texture in place at a draw's own pixels where the draw writes
/// nothing of that color (`render/snapshot.rs`'s in place): it enabled
/// `attachmentFeedbackLoopLayout`, and its passes fetch no color through input attachments.
fn reads_in_place(device: &Device) -> bool {
    let enabled = (device.enabled().features).contains(&VkFeature::AttachmentFeedbackLoopLayout);
    enabled && !through_inputs(device.compile_device().profile())
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

/// Assert that each of `frames`, named, left the targets' reds `want`.
fn assert_reds(frames: &[(&str, &Frame)], want: &[Vec<u8>; 2]) {
    for (frame, (_, reds)) in frames {
        for (target, (reds, want)) in reds.iter().zip(want).enumerate() {
            if let Some(at) = (0..reds.len()).find(|&at| reds[at] != want[at]) {
                let (x, y) = (at as u64 % SIDE, at as u64 / SIDE);
                panic!(
                    "{frame} frame: target {target}'s pixel ({x}, {y}) holds red {}, not {}",
                    reds[at], want[at]
                );
            }
        }
    }
}

/// The kept and the apart frame of `tob_read`'s draws on a new device, of other arguments where
/// `argued` (the module docs): their texels checked, their copies and commands asserted the same;
/// what each kept, and whether the device read in place.
fn lane(scratch: &str, argued: bool) -> (Keeps, Keeps, bool) {
    let scratch = Scratch::new(scratch);
    let device = plan_device(&scratch.0);
    let state = state(&device, "tob_read");
    // Each frame into a queue of its own: the first frame's last draw, kept, is carried along its
    // queue into the next render encoder whose pass is its pass's like (`render/carried.rs`),
    // which the second frame's first draw would rebind. A frame of every step on a queue of its
    // own goes first, and the device's own checks that record commands end, before either frame
    // records (`draw_again::checks_ended`).
    let queues = [(); 3].map(|_| device.new_command_queue().unwrap());
    frame(&queues[2], &state, (true, argued));
    checks_ended(&device);
    let kept = frame(&queues[0], &state, (false, argued));
    let apart = frame(&queues[1], &state, (true, argued));
    let want = [first_reds(), vec![SECOND_RED; (SIDE * SIDE) as usize]];
    assert_reds(&[("kept", &kept), ("apart", &apart)], &want);
    let (kept_copies, kept_keeps) = moved(&kept.0);
    let (apart_copies, apart_keeps) = moved(&apart.0);
    eprintln!(
        "KEPT-SHADOWED-DRAWS {} (argued {argued}): kept {kept_copies:?} {kept_keeps:?} {:?}; \
         apart {apart_copies:?} {apart_keeps:?}",
        device.name(),
        rebinds_moved(&kept.0),
    );
    // The same copies and commands: every draw a read through a copy, but the four of the second
    // target's texels where the device reads them in place.
    assert_eq!(kept_copies, apart_copies, "the kept frame's copies");
    assert_eq!(
        recorded(&kept.0),
        recorded(&apart.0),
        "the kept frame's commands"
    );
    let in_place = reads_in_place(&device);
    let (through, read_in_place) = match in_place {
        true => (4, 4),
        false => (8, 0),
    };
    let reads = (kept_copies.through_copies, kept_copies.in_place);
    let what = "the reads through copies and in place";
    assert_eq!(
        reads,
        (through, read_in_place),
        "{what} on {}",
        device.name()
    );
    (kept_keeps, apart_keeps, in_place)
}

/// What a frame of every step keeps: nothing but the draws it does not keep, `unkept` of them for
/// its shadows.
fn apart_keeps(unkept: u64) -> Keeps {
    Keeps {
        unkept,
        unkept_shadowed: unkept,
        as_before: 0,
        rebound: 0,
        stopped: 0,
        rebound_arguments: 0,
        stopped_arguments: 0,
        stopped_shadows: 0,
        stopped_checks: 0,
    }
}

#[test]
fn draws_that_read_their_pass_s_copies_are_kept_with_the_copies_of_every_step() {
    let (kept, apart, in_place) = lane("kept-shadowed-draws", false);
    // Every draw through a copy keeps itself, the pass's first, which begins its instance, too;
    // a draw in place keeps nothing, so each after it finds none kept. A draw again, or rebound,
    // the first after a draw kept and every second after it takes every step in builds with debug
    // assertions (`render/draw_again.rs`): of the draws after the first, the first, third, fifth
    // and seventh. Through copies, the draw of the second target's texels is rebound, gathering
    // its shadow, and so is the scissor's, its read's bounds another where the device bounds the
    // reads at their own pixels; the last two are drawn again. In place, the draw of the second
    // target's texels stops its rebind (it would read in place).
    let debug = u64::from(cfg!(debug_assertions));
    let unkept = 4 * u64::from(in_place);
    let want = match in_place {
        true => Keeps {
            as_before: 2 - debug,
            rebound: 1 - debug,
            stopped: 1,
            ..apart_keeps(unkept)
        },
        false => Keeps {
            as_before: 4 - 2 * debug,
            rebound: 3 - 2 * debug,
            ..apart_keeps(unkept)
        },
    };
    assert_eq!(kept, want, "the kept frame's keeps");
    assert_eq!(apart, apart_keeps(unkept), "the apart frame's keeps");
}

/// The lane of draws of other arguments (the module docs): each draw after the first, of another
/// base instance than the draw before it, is rebound with its arguments, the second target's and
/// the scissor's among them, each gathering its shadow, with the copies, commands and texels of
/// every step.
#[test]
fn draws_of_other_arguments_that_read_other_copies_rebind_with_them() {
    let (kept, apart, in_place) = lane("kept-shadowed-argued", true);
    // As the first lane's, each draw after the first one of other arguments: none drawn again.
    // In place, the draw of the second target's texels stops (it would read in place), and the
    // draws after it find none kept.
    let debug = u64::from(cfg!(debug_assertions));
    let unkept = 4 * u64::from(in_place);
    let want = match in_place {
        true => Keeps {
            rebound: 3 - 2 * debug,
            stopped: 1,
            rebound_arguments: 3 - 2 * debug,
            stopped_arguments: 1,
            ..apart_keeps(unkept)
        },
        false => Keeps {
            rebound: 7 - 4 * debug,
            rebound_arguments: 7 - 4 * debug,
            ..apart_keeps(unkept)
        },
    };
    assert_eq!(kept, want, "the kept frame's keeps");
    assert_eq!(apart, apart_keeps(unkept), "the apart frame's keeps");
}

/// The lane across a barrier (the module docs): the draw that reads the first target's copy after
/// the barrier stops its rebind at its check, and takes the split as every step does, reading what
/// the draw before the barrier wrote.
#[test]
fn a_draw_rebound_that_reads_a_color_s_copy_after_a_barrier_takes_every_step() {
    let scratch = Scratch::new("kept-shadowed-barrier");
    let device = plan_device(&scratch.0);
    let queues = [(); 3].map(|_| device.new_command_queue().unwrap());
    // The sampled texture, cleared to its red by a pass of its own.
    let source = target(&device);
    let mut clear = RenderPassDescriptor::new();
    let color = &mut clear.color_attachments[0];
    color.texture = Some(source.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(SOURCE_RED) / 255.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let command_buffer = queues[2].command_buffer().unwrap();
    let encoder = command_buffer.render_command_encoder(&clear).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let state = state(&device, "tob_sample");
    barrier_frame(&queues[2], &state, (&source, &sampler), true);
    checks_ended(&device);
    let kept = barrier_frame(&queues[0], &state, (&source, &sampler), false);
    let apart = barrier_frame(&queues[1], &state, (&source, &sampler), true);
    // The first draw writes the source's red over the first rectangle, which the second reads
    // back where the rectangles meet.
    let [x0, y0, x1, y1] = FIRST_RECT;
    let first = (0..SIDE * SIDE)
        .map(|at| {
            let (x, y) = (at % SIDE, at / SIDE);
            match (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                true => SOURCE_RED,
                false => FIRST_RED,
            }
        })
        .collect();
    let want = [first, vec![SECOND_RED; (SIDE * SIDE) as usize]];
    assert_reds(&[("kept", &kept), ("apart", &apart)], &want);
    let (kept_copies, kept_keeps) = moved(&kept.0);
    let (apart_copies, apart_keeps) = moved(&apart.0);
    eprintln!(
        "KEPT-SHADOWED-BARRIER {}: kept {kept_copies:?} {kept_keeps:?} {:?}; apart \
         {apart_copies:?} {apart_keeps:?}",
        device.name(),
        rebinds_moved(&kept.0),
    );
    assert_eq!(kept_copies, apart_copies, "the kept frame's copies");
    assert_eq!(
        recorded(&kept.0),
        recorded(&apart.0),
        "the kept frame's commands"
    );
    assert_eq!(kept_copies.splits, 1, "the split after the barrier");
    let stops = |keeps: &Keeps| (keeps.stopped_arguments, keeps.stopped_checks);
    assert_eq!(
        stops(&kept_keeps),
        (1, 1),
        "the kept frame's stop at its check"
    );
    assert_eq!(stops(&apart_keeps), (0, 0), "the apart frame's stops");
}

/// The fourth lane (the module docs): each draw rebound that gathers its own shadow use, the
/// second target's or the first's read within new bounds, is the last to use it so, and the kept
/// frame's copies, commands and texels are still those of every step.
#[test]
fn draws_rebound_that_last_use_a_shadow_so_gather_it_themselves() {
    let scratch = Scratch::new("kept-shadowed-gathers");
    let device = plan_device(&scratch.0);
    if reads_in_place(&device) {
        eprintln!(
            "KEPT-SHADOWED-GATHERS {}: reads a color's texture in place; left out",
            device.name()
        );
        return;
    }
    let state = state(&device, "tob_read");
    let queues = [(); 3].map(|_| device.new_command_queue().unwrap());
    gathers_frame(&queues[2], &state, true);
    checks_ended(&device);
    let kept = gathers_frame(&queues[0], &state, false);
    let apart = gathers_frame(&queues[1], &state, true);
    // The fifth draw alone writes the second target's red, over the second rectangle inside the
    // first scissor; every other draw writes the first target's own.
    let first = (0..SIDE * SIDE)
        .map(|at| {
            let (x, y) = (at % SIDE, at / SIDE);
            match (8..16).contains(&x) && (4..16).contains(&y) {
                true => SECOND_RED,
                false => FIRST_RED,
            }
        })
        .collect();
    let want = [first, vec![SECOND_RED; (SIDE * SIDE) as usize]];
    assert_reds(&[("kept", &kept), ("apart", &apart)], &want);
    let (kept_copies, kept_keeps) = moved(&kept.0);
    let (apart_copies, apart_kept) = moved(&apart.0);
    eprintln!(
        "KEPT-SHADOWED-GATHERS {}: kept {kept_copies:?} {kept_keeps:?} {:?}; apart \
         {apart_copies:?} {apart_kept:?}",
        device.name(),
        rebinds_moved(&kept.0),
    );
    assert_eq!(kept_copies, apart_copies, "the kept frame's copies");
    assert_eq!(
        recorded(&kept.0),
        recorded(&apart.0),
        "the kept frame's commands"
    );
    // Each draw after the first one of other arguments, rebound: in builds with debug assertions,
    // the first, third and fifth take every step instead (`render/draw_again.rs`).
    let debug = u64::from(cfg!(debug_assertions));
    let want = Keeps {
        rebound: 6 - 3 * debug,
        rebound_arguments: 6 - 3 * debug,
        ..apart_keeps(0)
    };
    assert_eq!(kept_keeps, want, "the kept frame's keeps");
    assert_eq!(apart_kept, apart_keeps(0), "the apart frame's keeps");
}
