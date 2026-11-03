//! Runs of passes over a linear attachment rendered apart, merged once (`linear_run.rs`): one
//! command buffer mixing passes over a render target over a buffer's bytes (each loading and
//! storing it beside a cleared and stored `Depth32Float` depth attachment, blending a rectangle
//! over it) with the encoders and commands that end a run or pass it: a 1x1 attachmentless
//! render encoder (a `-P` stencilprobe's), a pass over another texture, a blit reading the
//! target's texels, an event signal, a pass naming the target's buffer, and a pass whose box
//! reaches past its run's.
//!
//! With the device's rule forced (`DeviceShared::linear_rule_forced`), each pass over the target
//! renders it apart, through the image it keeps, its changed words merged back
//! (`linear_attachment.rs`, `linear_merge.rs`). The sequence runs three times: over the target,
//! where its passes form the runs the steps make, merged once each; over the target again with a
//! blit between every two encoders, which ends every run, so every pass merges after itself; and
//! over a private texture of the same first texels, which merges nothing. The texels each blit
//! read mid-sequence, and those the sequence left, are the same in all three.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use super::merge_scratch::{blending, blit, bytes, drawn, first_bytes, staged, target, texels};
use super::merge_scratch::{Host, Over, H, TEXEL};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Device, EventRef, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use std::sync::atomic::Ordering;

/// One step of the sequence.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A pass over the target blending rectangle `i` (`drawn`).
    Draw(usize),
    /// A pass over the target blending a rectangle over all of it, of `i`'s color.
    Whole(usize),
    /// A 1x1 render encoder of no attachment, one draw (a `-P` stencilprobe's).
    Lone,
    /// A pass over another texture, binding the target's buffer as a fragment buffer its
    /// function does not read where `true`, else a buffer of its own.
    Other(bool),
    /// A blit copying the target's texels into a buffer, read after the command buffer.
    Read,
    /// An event signal.
    Signal,
}

use Step::{Draw, Lone, Other, Read, Signal, Whole};

/// The sequence, and the runs it makes over the target, each of the passes noted:
/// 1. `Whole(0)` to `Draw(4)`: the 1x1 encoder and the pass over another texture pass it; the
///    blit reading the texels ends it (5 passes);
/// 2. `Whole(5)`, `Draw(6)`: the signal ends it (2);
/// 3. `Whole(7)`, `Draw(8)`: the pass naming the target's buffer ends it (2);
/// 4. `Whole(9)`, `Draw(10)`: the blit ends it (2);
/// 5. `Draw(0)` over its own box, which `Draw(42)`'s reaches past (1);
/// 6. `Draw(42)` over the whole target, `Draw(43)`, and past the 1x1 encoder `Draw(44)`; the
///    blit ends it (3);
/// 7. `Whole(11)`, past the pass over another texture `Draw(12)`, to the command buffer's end
///    (2).
const STEPS: [Step; 26] = [
    Whole(0),
    Draw(1),
    Draw(2),
    Lone,
    Draw(3),
    Other(false),
    Draw(4),
    Read,
    Whole(5),
    Draw(6),
    Signal,
    Whole(7),
    Draw(8),
    Other(true),
    Whole(9),
    Draw(10),
    Read,
    Draw(0),
    Draw(42),
    Draw(43),
    Lone,
    Draw(44),
    Read,
    Whole(11),
    Other(false),
    Draw(12),
];

/// The passes over the target among [`STEPS`], and the runs they make.
const PASSES: u64 = 17;
const RUNS: u64 = 7;

/// The state that draws a rectangle (`tob_rect`) of the color in fragment buffer 0
/// (`tob_solid`) with no attachment.
fn attachmentless(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] = None;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What the sequence runs over: the target, its buffer, the other texture, and the states.
struct Scene<'a> {
    texture: &'a Texture,
    buffer: &'a Buffer,
    other: &'a Texture,
    blending: &'a RenderPipelineState,
    attachmentless: &'a RenderPipelineState,
}

/// A `Private` texture of `format`, `width` x [`H`] texels, with `usage`.
fn private_texture(
    device: &Device,
    format: PixelFormat,
    width: u64,
    usage: TextureUsage,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(format, width, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// Encode a pass over `texture` beside `depth`, loading and storing it, that blends the
/// rectangle `corners` of `rgba` with `state`; binding `named` as fragment buffer 1, which the
/// function does not read.
fn pass(
    command_buffer: &CommandBuffer,
    (texture, depth): (&Texture, &Texture),
    state: &RenderPipelineState,
    (corners, rgba): ([f32; 4], [f32; 4]),
    named: Option<&Buffer>,
) {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    descriptor.depth_attachment.texture = Some(depth.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
    (encoder.set_fragment_bytes(&bytes_of(&rgba, f32::to_le_bytes), 0)).unwrap();
    if let Some(named) = named {
        encoder.set_fragment_buffer(Some(named), 0, 1).unwrap();
    }
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
}

/// [`STEPS`] over `over` in one command buffer, committed and completed; a blit between every
/// two encoders where `apart`, copying 4 bytes between buffers of its own. The texels each
/// [`Read`] copied, packed, and the device's work counts as it completed.
fn sequence(queue: &CommandQueue, over: &Scene<'_>, apart: bool) -> (Vec<Vec<u8>>, WorkCounts) {
    let device = queue.device();
    let width = over.texture.width();
    let packed = width * TEXEL;
    let depth = private_texture(
        device,
        PixelFormat::Depth32Float,
        width,
        TextureUsage::RENDER_TARGET,
    );
    let unread = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let (from, into) = (staged(device, &[1; 4]), staged(device, &[0; 4]));
    let event = device.new_event().unwrap();
    let reads = (STEPS.iter())
        .filter(|step| matches!(step, Read))
        .map(|_| (device.new_buffer(packed * H, ResourceOptions::shared())).unwrap())
        .collect::<Vec<_>>();
    let mut read = reads.iter();
    let command_buffer = queue.command_buffer().unwrap();
    let whole = |i: usize| ([-1.0, -1.0, 1.0, 1.0], drawn(i).1);
    for (at, step) in STEPS.iter().enumerate() {
        if apart && at > 0 {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            (encoder.copy_from_buffer_to_buffer(&from, 0, &into, 0, 4)).unwrap();
            encoder.end_encoding().unwrap();
        }
        let target = (over.texture, &depth);
        match *step {
            Draw(i) => pass(&command_buffer, target, over.blending, drawn(i), None),
            Whole(i) => pass(&command_buffer, target, over.blending, whole(i), None),
            Lone => {
                let mut lone = RenderPassDescriptor::new();
                lone.render_target_width = 1;
                lone.render_target_height = 1;
                lone.default_raster_sample_count = 1;
                let mut encoder = command_buffer.render_command_encoder(&lone).unwrap();
                encoder
                    .set_render_pipeline_state(over.attachmentless)
                    .unwrap();
                let corners = [-1.0f32, -1.0, 1.0, 1.0];
                (encoder.set_vertex_bytes(&bytes_of(&corners, f32::to_le_bytes), 0)).unwrap();
                (encoder.set_fragment_bytes(&bytes_of(&[0.0f32; 4], f32::to_le_bytes), 0)).unwrap();
                (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
                encoder.end_encoding().unwrap();
            }
            Other(names) => {
                let named = match names {
                    true => over.buffer,
                    false => &unread,
                };
                let other = (over.other, &depth);
                pass(
                    &command_buffer,
                    other,
                    over.blending,
                    drawn(at),
                    Some(named),
                );
            }
            Read => {
                let mut encoder = command_buffer.blit_command_encoder().unwrap();
                let origin = Origin::new(0, 0, 0);
                let size = Size::new(width, H, 1);
                let out = read.next().unwrap();
                (encoder.copy_from_texture_to_buffer(
                    over.texture,
                    0,
                    0,
                    origin,
                    size,
                    out,
                    0,
                    packed,
                    packed * H,
                    BlitOption::None,
                ))
                .unwrap();
                encoder.end_encoding().unwrap();
            }
            Signal => {
                (command_buffer.encode_signal_event(EventRef::from(&event), at as u64)).unwrap()
            }
        }
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let reads = (reads.iter())
        .map(|out| super::merge_scratch::read(out, packed * H))
        .collect();
    (reads, device.work_counts())
}

/// The texels of `texture`, of `width` x [`H`], packed: read through a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let (width, packed) = (texture.width(), texture.width() * TEXEL);
    let out = (queue.device())
        .new_buffer(packed * H, ResourceOptions::shared())
        .unwrap();
    blit(queue, |b| {
        (b.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(width, H, 1),
            &out,
            0,
            packed,
            packed * H,
            BlitOption::None,
        ))
        .unwrap()
    });
    super::merge_scratch::read(&out, packed * H)
}

/// [`STEPS`] over a target over `over`'s buffer of `width` texels a row, `row` bytes apart, the
/// rule forced, as runs and with every pass apart, against a private texture's: the same texels
/// read mid-sequence and left at its end, and one merge a run (every pass's where apart), each
/// later pass of a run taking its run's merge over.
fn runs_hold_their_texels(over: Over, (width, row): (u64, u64)) {
    let cache = Scratch::new(&format!("linear-runs-{over:?}-{width}"));
    let host = Host::new(row * H);
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let blending = blending(&device);
    let attachmentless = attachmentless(&device);
    let packed = width * TEXEL;
    let target_usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let other = private_texture(&device, PixelFormat::RGBA8Unorm, width, target_usage);
    // The private texture of the target's first texels.
    let reference = private_texture(&device, PixelFormat::RGBA8Unorm, width, target_usage);
    let source = staged(&device, &first_bytes(width, packed));
    blit(&queue, |b| {
        (b.copy_from_buffer_to_texture(
            &source,
            0,
            packed,
            packed * H,
            Size::new(width, H, 1),
            &reference,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let unused = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let on_reference = Scene {
        texture: &reference,
        buffer: &unused,
        other: &other,
        blending: &blending,
        attachmentless: &attachmentless,
    };
    let (expected, _) = sequence(&queue, &on_reference, false);
    let left = texels_of(&queue, &reference);
    let forced = &device.shared().linear_rule_forced;
    for apart in [false, true] {
        let Some((buffer, texture)) = target(&queue, over, (width, row), &host) else {
            return;
        };
        assert!(
            texture.linear_image().unwrap(),
            "the target is a linear image"
        );
        let on_target = Scene {
            texture: &texture,
            buffer: &buffer,
            other: &other,
            blending: &blending,
            attachmentless: &attachmentless,
        };
        forced.store(true, Ordering::Relaxed);
        let before = device.work_counts();
        let (reads, after) = sequence(&queue, &on_target, apart);
        forced.store(false, Ordering::Relaxed);
        let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
        let through = delta(|c| c.linear_passes_through_copies);
        let recorded = delta(|c| c.linear_merges_recorded);
        let taken_over = delta(|c| c.linear_merges_taken_over);
        println!(
            "LINEAR-RUNS {over:?} {width}x{H}, rows of {row}, apart {apart}: {through} passes \
             through copies, linear merges {recorded} recorded and {taken_over} taken over"
        );
        assert_eq!(
            through, PASSES,
            "every pass over the target renders it apart"
        );
        let runs = match apart {
            true => PASSES,
            false => RUNS,
        };
        assert_eq!(recorded, runs, "one merge a run");
        assert_eq!(
            taken_over,
            PASSES - runs,
            "every later pass of a run takes its merge over"
        );
        for (at, (got, expected)) in reads.iter().zip(&expected).enumerate() {
            let differ = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
            assert_eq!(
                differ, 0,
                "read {at}: the texels differ from the private texture's"
            );
        }
        let got = texels(&bytes(&queue, over, &buffer, row * H), width, row);
        let differ = (0..got.len()).filter(|&i| got[i] != left[i]).count();
        assert_eq!(
            differ, 0,
            "the texels left differ from the private texture's"
        );
        drop((texture, buffer));
    }
}

/// Over a `Shared` buffer, rows packed (1024 bytes).
#[test]
fn runs_over_a_shared_buffer_hold_their_texels() {
    runs_hold_their_texels(Over::Shared, (256, 1024));
}

/// Over a `Private` buffer, packed rows of 1024 bytes.
#[test]
fn runs_over_a_private_buffer_hold_their_texels() {
    runs_hold_their_texels(Over::Private, (256, 1024));
}
