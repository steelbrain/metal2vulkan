//! A render encoder's memo of its draws' shapes (`render/draw_shape.rs`) beside the kinds a device
//! announces at every occurrence (`DeviceOptions::announce_each`): a draw whose shape's derivation
//! meets such a kind derives it anew and announces it at every draw, as a device without the memo
//! would; a draw whose shape meets none takes it from the memo, whatever other kinds the device
//! announces at each (`WorkCounts::draw_shapes_reused`, or `draws_as_before` for a draw of the
//! encoder's last draw again, which takes that draw's), as a device that announces every kind
//! once does. A draw after a new scissor rectangle reuses its shape too, through its own scissor
//! (`render_state::ScissorRects`). An encoder's first draw takes the shape an earlier encoder's
//! draw of its drawing state derived from the same values (`WorkCounts::draw_shapes_carried`), so
//! once a run derived them every draw of the next reuses its shape; a draw of other values derives
//! its own.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{until, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library, pass};
use crate::{
    CommandBufferStatus, Degradation, DegradeNotice, DepthStencilState, Device, DeviceKind,
    DeviceOptions, MessageSink, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ScissorRect, ShaderCacheOptions, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, CullMode, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The draws of each kind an encoder of a test draws.
const DRAWS: usize = 8;

/// The side of the target.
const SIDE: u64 = 16;

/// The kinds a consumer announces at every occurrence so that its calls hear them, and those its
/// process dies at by default: kinds of textures, heaps, buffers, command states and a variant's
/// bind, none of which a draw's shape meets.
const UNRELATED: [Degradation; 10] = [
    Degradation::HeapTwoPlaneTexture,
    Degradation::OptimizedDepthStencilTexture,
    Degradation::RasterizationRateMap,
    Degradation::HeapTextureUntiledLayout,
    Degradation::HeapTextureLosslessLayout,
    Degradation::UnmeasuredHeapTextureSize,
    Degradation::HalfOutputPrecision,
    Degradation::ReleasedCommandState,
    Degradation::PrivateResourceOption,
    Degradation::HostDeallocatorDelayed,
];

/// The kinds a `Depth16Unorm` draw's state announces (`render_state::announce_unorm_depth`),
/// which of them depending on the device's profile.
const UNORM_DEPTH: [Degradation; 2] =
    [Degradation::UnormDepthRounding, Degradation::UnormDepthBias];

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that announces the kinds of `each` at every occurrence,
/// its notices into `heard`.
fn device(cache: &Path, each: &[Degradation], heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        announce_each: each.iter().copied().collect(),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// How many notices of each kind `heard` holds, emptied.
fn counts(heard: &Heard) -> BTreeMap<Degradation, usize> {
    let mut counts = BTreeMap::new();
    for notice in heard.lock().unwrap().drain(..) {
        *counts.entry(notice.degradation).or_default() += 1;
    }
    counts
}

/// The state that draws a rectangle (`tob_rect`, `tob_solid`) into `RGBA8Unorm`, against a
/// `Depth16Unorm` depth where `depth`.
fn rectangle_state(device: &Device, depth: bool) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    if depth {
        descriptor.depth_format = Some(PixelFormat::Depth16Unorm);
    }
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `Private` texture of `format` that a pass renders.
fn target(device: &Device, format: PixelFormat) -> Texture {
    target_of(device, format, SIDE)
}

/// A `Private` texture of `format`, `side` pixels a side, that a pass renders.
fn target_of(device: &Device, format: PixelFormat, side: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// What an encoder of a test draws: a rectangle of `state` over the whole target, as triangles,
/// and before each, where `points`, as points (whose vertex function writes no point size, of no
/// point class: `UnsizedPoints`, `ShapeInputs::settings`), each [`DRAWS`] times; tested against
/// `depth` where given; each triangles' draw through a scissor rectangle of its own set before it
/// where `scissors`.
struct Draws<'a> {
    state: &'a RenderPipelineState,
    points: bool,
    depth: Option<(&'a Texture, &'a DepthStencilState)>,
    scissors: bool,
}

impl Draws<'_> {
    /// Encode the draws in one pass into a new `RGBA8Unorm` target, commit and wait: the draws
    /// that reused their shapes, each draw of the encoder's last draw again or rebound among them,
    /// which records as the draw it repeats, of that draw's shape (`render/draw_again.rs`).
    fn run(&self, device: &Device) -> u64 {
        let target = target(device, PixelFormat::RGBA8Unorm);
        let mut descriptor: RenderPassDescriptor = pass(&target);
        if let Some((depth, _)) = self.depth {
            let attachment = &mut descriptor.depth_attachment;
            attachment.texture = Some(depth.clone());
            attachment.load_action = crate::LoadAction::Clear;
            attachment.store_action = crate::StoreAction::Store;
            attachment.clear_depth = 1.0;
        }
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let reused = |counts: crate::WorkCounts| {
            counts.draw_shapes_reused + counts.draws_as_before + counts.draws_rebound
        };
        let before = reused(device.work_counts());
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        self.encode(&mut encoder);
        encoder.end_encoding().unwrap();
        let reused = reused(device.work_counts()) - before;
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        reused
    }

    fn encode(&self, encoder: &mut RenderCommandEncoder) {
        encoder.set_render_pipeline_state(self.state).unwrap();
        if let Some((_, depth_stencil)) = self.depth {
            encoder.set_depth_stencil_state(depth_stencil).unwrap();
        }
        let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        let color = bytes_of(&[0.2f32, 0.4, 0.6, 1.0], f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        for at in 0..DRAWS as u64 {
            if self.scissors {
                let rect = ScissorRect::new(at, 0, SIDE - at, SIDE);
                assert_eq!(encoder.set_scissor_rect(rect), Ok(()));
            }
            if self.points {
                let drawn = encoder.draw_primitives(PrimitiveType::Point, 0, 4, 1, 0);
                assert_eq!(drawn, Ok(()));
            }
            let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0);
            assert_eq!(drawn, Ok(()));
        }
    }
}

/// How many notices of each kind a run announced.
type Counts = BTreeMap<Degradation, usize>;

/// On a new device announcing the kinds of `each` at every occurrence, [`Draws`] of points where
/// `points` and against a `Depth16Unorm` depth tested `Less` and written where `depth`, run until
/// they find their keys' own pipelines made ([`warm`]), then once more: that run's reused draws and
/// notices. Each draw of it may reuse its shape: the first of each assembly the one an earlier
/// run's encoder derived.
fn second_run(each: &[Degradation], points: bool, depth: bool) -> (u64, Counts) {
    let scratch = Scratch::new("draw-shape-notices");
    let heard = Heard::default();
    let device = device(&scratch.0, each, &heard);
    let state = rectangle_state(&device, depth);
    let texture = target(&device, PixelFormat::Depth16Unorm);
    let mut descriptor = DepthStencilDescriptor::default();
    descriptor.depth_compare_function = CompareFunction::Less;
    descriptor.depth_write_enabled = true;
    let depth_stencil = device.new_depth_stencil_state(&descriptor).unwrap();
    let draws = Draws {
        state: &state,
        points,
        depth: depth.then_some((&texture, &depth_stencil)),
        scissors: false,
    };
    warm(&device, &draws);
    counts(&heard);
    let reused = draws.run(&device);
    (reused, counts(&heard))
}

/// Run `draws` on `device` until a run draws no key's residual form, the device's pool is idle,
/// and once more: each draw of that last run found its key's own pipeline made, so every shape it
/// kept holds its pipeline, for a later encoder's draws to take (a draw whose key's own is still
/// made on the pool draws its residual form, `pipeline/render/residual.rs`, or records against
/// the making, and keeps no pipeline).
fn warm(device: &Device, draws: &Draws<'_>) {
    until("every key's own pipeline made", || {
        let residual = device.work_counts().residual_form_draws;
        draws.run(device);
        device.work_counts().residual_form_draws == residual
    });
    until("the pipeline pool idle", || {
        device.work_counts().pipeline_pool_busy == 0
    });
    draws.run(device);
}

/// Draws whose shapes meet no notice reuse them on a device that announces kinds no shape meets
/// at every occurrence (a consumer's), as on one that announces every kind once: each draw.
#[test]
fn draws_reuse_shapes_beside_kinds_announced_at_each_occurrence() {
    let (once, _) = second_run(&[], false, false);
    assert_eq!(once, DRAWS as u64);
    let (each, heard) = second_run(&UNRELATED, false, false);
    assert_eq!(each, once, "{heard:?}");
}

/// A kind a draw's settings meet (`UnsizedPoints`), announced at every occurrence, is announced at
/// every draw that meets it, which derives its settings anew, while the encoder's other draws reuse
/// theirs; a device that announces it once announced it in the first run, and reuses both.
#[test]
fn a_kind_a_draws_settings_meet_is_announced_at_every_draw() {
    let (reused, heard) = second_run(&[Degradation::UnsizedPoints], true, false);
    assert_eq!(
        heard.get(&Degradation::UnsizedPoints),
        Some(&DRAWS),
        "{heard:?}"
    );
    assert_eq!(reused, DRAWS as u64, "the triangles' draws");
    let (reused, heard) = second_run(&[], true, false);
    assert_eq!(heard.get(&Degradation::UnsizedPoints), None, "{heard:?}");
    assert_eq!(reused, 2 * DRAWS as u64);
}

/// A kind a draw's state meets (those of a `Depth16Unorm` depth, [`UNORM_DEPTH`], which the
/// device's profile decides), announced at every occurrence, is announced at every draw, which
/// derives its state anew: each kind a device announcing them once announced (in its first run),
/// [`DRAWS`] times, and no other. Where the profile meets none, the draws reuse their shapes.
#[test]
fn a_kind_a_draws_state_meets_is_announced_at_every_draw() {
    let scratch = Scratch::new("draw-shape-notices-once");
    let heard = Heard::default();
    let met = {
        let device = device(&scratch.0, &[], &heard);
        let state = rectangle_state(&device, true);
        let texture = target(&device, PixelFormat::Depth16Unorm);
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = CompareFunction::Less;
        descriptor.depth_write_enabled = true;
        let depth_stencil = device.new_depth_stencil_state(&descriptor).unwrap();
        let draws = Draws {
            state: &state,
            points: false,
            depth: Some((&texture, &depth_stencil)),
            scissors: false,
        };
        draws.run(&device);
        counts(&heard)
    };
    let met: Vec<Degradation> = UNORM_DEPTH
        .into_iter()
        .filter(|kind| met.contains_key(kind))
        .collect();
    eprintln!("a Depth16Unorm draw's state meets {met:?}");
    let (reused, heard) = second_run(&UNORM_DEPTH, false, true);
    for kind in UNORM_DEPTH {
        let expected = met.contains(&kind).then_some(&DRAWS);
        assert_eq!(heard.get(&kind), expected, "{kind:?}: {heard:?}");
    }
    let expected = match met.is_empty() {
        true => DRAWS as u64,
        false => 0,
    };
    assert_eq!(reused, expected, "{met:?}");
}

/// Draws that each set a scissor rectangle of their own before they draw, as a window compositor's
/// do, reuse their shapes as draws through one rectangle do (the rectangles are no input of a
/// shape: `render_state::ScissorRects`), each of them, or are rebound from the draw before,
/// restated (`render/draw_again/restated.rs`), and each records its own scissor.
#[test]
fn draws_after_new_scissor_rectangles_reuse_shapes() {
    let scratch = Scratch::new("draw-shape-scissors");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = rectangle_state(&device, false);
    let draws = Draws {
        state: &state,
        points: false,
        depth: None,
        scissors: true,
    };
    warm(&device, &draws);
    let scissors = device.work_counts().commands_recorded.cmd_set_scissor;
    let reused = draws.run(&device);
    assert_eq!(reused, DRAWS as u64);
    let recorded = device.work_counts().commands_recorded.cmd_set_scissor - scissors;
    assert_eq!(recorded, DRAWS as u64, "a scissor each");
}

/// The draws of [`Draws`] of `state` (triangles alone) in one pass into a new `RGBA8Unorm` target
/// `side` pixels a side, behind a back-face cull where `culled`, committed and waited for: those
/// that took their shape from an earlier encoder's draw (`WorkCounts::draw_shapes_carried`).
fn carried_run(device: &Device, state: &RenderPipelineState, side: u64, culled: bool) -> u64 {
    let target = target_of(device, PixelFormat::RGBA8Unorm, side);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let carried = device.work_counts().draw_shapes_carried;
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    if culled {
        encoder.set_cull_mode(CullMode::Back);
    }
    let draws = Draws {
        state,
        points: false,
        depth: None,
        scissors: false,
    };
    draws.encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let carried = device.work_counts().draw_shapes_carried - carried;
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    carried
}

/// An encoder's first draw of a drawing state takes the shape an earlier encoder's draw of the
/// state derived from the same values, and its later draws their own encoder's; a draw of other
/// values derives its own: behind another cull mode, or in a pass of another size (its render
/// area and viewport). The state keeps the shapes of each, so the first values are found again.
#[test]
fn an_encoders_first_draw_takes_the_shape_of_the_same_values() {
    let scratch = Scratch::new("draw-shape-carried");
    let heard = Heard::default();
    let device = device(&scratch.0, &[], &heard);
    let state = rectangle_state(&device, false);
    let run = |side, culled| carried_run(&device, &state, side, culled);
    assert_eq!(run(SIDE, false), 0, "the state's first encoder");
    assert_eq!(run(SIDE, false), 1, "the same values");
    assert_eq!(run(SIDE, true), 0, "another cull mode");
    assert_eq!(run(2 * SIDE, false), 0, "a pass of another size");
    assert_eq!(run(SIDE, true), 1, "the culled values again");
    assert_eq!(run(SIDE, false), 1, "the first values again");
}
