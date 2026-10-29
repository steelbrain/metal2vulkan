//! Device tests of attachments larger than their pass's render area (the runtime's
//! `clear_reach.rs`) on the device the cfg selects, under the validation layer.
//!
//! Each pass is one that our probes measured (the M3 Max with macOS 26.7 and an
//! M4 with 27.0.1, the same answers, each case with and without Metal's API validation): a
//! 128x128 attachment whose texels the test sets first (a color's bytes 0x11, a depth 0.75, a
//! stencil 0x22), a render area of 40x40 (the descriptor's target size, or a smaller attachment),
//! cleared to (0.5, 0, 0, 1), depth 0.5 or stencil 0x40 and stored, and drawn by the owned
//! fixture `rt6_restart` (its vertex function places each vertex where buffer 0 holds it, at
//! depth 0.25; its fragment function writes (1/255, 0, 0, 1)): a triangle over the whole
//! target, one wholly outside it, or none. Metal's clear reaches past the render area to whole
//! 32x32 tiles in a pass that draws or of more layers than one, and over the whole attachment in
//! a pass of one layer that does not; a `DontCare` load keeps the texels in a pass that does not
//! draw and clears to zero (a color) or the clear value (depth, stencil) in one that does, which
//! is Metal's answer in each tile a draw touches and a near-equivalent elsewhere, announced once
//! ([`DONT_CARE_LOADS`]). A pass without a draw call that clears and resolves nothing records
//! nothing at all, its texels kept ([`a_pass_that_changes_no_texel_records_nothing`]).

use super::degrade_notices::shared_in_system_memory;
use super::degrade_notices::{assert_made_once, forced, made_render_kinds};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandQueue, Degradation, Device, Library, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ScissorRect, Size, StoreAction, Texture, TextureDescriptor, Viewport,
};
use std::collections::BTreeSet;

/// The kind a pass that draws over an attachment loaded with `DontCare` takes.
pub(super) const DONT_CARE_LOADS: [Degradation; 1] = [Degradation::DontCareLoadTiles];

/// The attachments' side, and the render area's.
pub(super) const SIDE: u64 = 128;
pub(super) const AREA: u64 = 40;

/// A color's texels: set before the pass, cleared, and drawn.
pub(super) const OLD: [u8; 4] = [0x11; 4];
pub(super) const RED: [u8; 4] = [0x80, 0x00, 0x00, 0xff];
pub(super) const DRAWN: [u8; 4] = [0x01, 0x00, 0x00, 0xff];
const ZERO: [u8; 4] = [0; 4];

/// What a pass draws: nothing, a triangle over the whole target, one wholly outside it, or a
/// draw call of no vertex.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Draw {
    None,
    Full,
    Outside,
    Empty,
}

/// The lane's device, its queue, and the fixture's library.
pub(super) struct Lane {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    /// The fixture's pipeline of `colors`, `depth` and `stencil` and `samples` (a fragment
    /// function where it has a color).
    pub(super) fn pipeline(
        &self,
        colors: &[PixelFormat],
        attachments: (Option<PixelFormat>, Option<PixelFormat>),
        samples: u64,
    ) -> RenderPipelineState {
        let descriptor = self.descriptor(colors, attachments, samples);
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The descriptor of [`Lane::pipeline`]'s pipeline.
    pub(super) fn descriptor(
        &self,
        colors: &[PixelFormat],
        (depth, stencil): (Option<PixelFormat>, Option<PixelFormat>),
        samples: u64,
    ) -> RenderPipelineDescriptor {
        let function = |name| {
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_restart_vertex")));
        if !colors.is_empty() {
            descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
        }
        for (index, &pf) in colors.iter().enumerate() {
            descriptor.color_attachments[index] = Some(ColorAttachmentDescriptor::new(pf));
        }
        descriptor.depth_format = depth;
        descriptor.stencil_format = stencil;
        descriptor.raster_sample_count = samples as u32;
        descriptor
    }

    /// The kinds the making of [`Lane::pipeline`]'s pipeline announces.
    pub(super) fn made(
        &self,
        colors: &[PixelFormat],
        attachments: (Option<PixelFormat>, Option<PixelFormat>),
        samples: u64,
    ) -> BTreeSet<Degradation> {
        made_render_kinds(&self.device, &self.descriptor(colors, attachments, samples))
    }

    /// A texture of `pf`, `side` wide and high, of `samples` (multisampled above 1), `levels`
    /// and `slices`, of `usage`, shared or private.
    pub(super) fn texture(&self, pf: PixelFormat, shape: Shape, usage: TextureUsage) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, shape.side, shape.side, false);
        descriptor.texture_type = match (shape.samples, shape.slices) {
            (1, 1) => TextureType::Type2D,
            (1, _) => TextureType::Type2DArray,
            _ => TextureType::Type2DMultisample,
        };
        descriptor.sample_count = shape.samples;
        descriptor.array_length = shape.slices;
        descriptor.mipmap_level_count = shape.levels;
        descriptor.usage = usage;
        descriptor.resource_options = if shape.shared {
            ResourceOptions::shared()
        } else {
            ResourceOptions::private()
        };
        self.device.new_texture(&descriptor).unwrap()
    }

    /// A shared color texture of `shape` whose every level and slice holds [`OLD`].
    pub(super) fn old_color(&self, pf: PixelFormat, shape: Shape, usage: TextureUsage) -> Texture {
        let texture = self.texture(
            pf,
            Shape {
                shared: true,
                ..shape
            },
            usage,
        );
        for level in 0..shape.levels {
            let side = (shape.side >> level).max(1);
            for slice in 0..shape.slices {
                let bytes = OLD.repeat((side * side) as usize);
                texture
                    .replace_region(
                        Region::new_2d(0, 0, side, side),
                        level,
                        slice,
                        &bytes,
                        4 * side,
                        4 * side * side,
                    )
                    .unwrap();
            }
        }
        texture
    }

    /// The texels of `texture`'s `level` and `slice`, `bytes` each (one aspect of a combined
    /// depth/stencil texture by `option`), through a blit into a shared buffer.
    pub(super) fn read(
        &self,
        texture: &Texture,
        (level, slice): (u64, u64),
        bytes: u64,
        option: BlitOption,
    ) -> Vec<u8> {
        let side = (texture.width() >> level).max(1);
        let length = bytes * side * side;
        let buffer = buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            slice,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &buffer,
            0,
            bytes * side,
            length,
            option,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        read_buffer(&buffer, 0, length).unwrap()
    }

    /// Encode `descriptor`'s pass, drawn as `draw` through `pipeline` (depth always passing and
    /// written, the stencil replaced by 0x41, where the pass has them), then commit and wait.
    pub(super) fn run(
        &self,
        descriptor: &RenderPassDescriptor,
        draw: Draw,
        pipeline: Option<&RenderPipelineState>,
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        let mut held = Vec::new();
        if let (Some(pipeline), true) = (pipeline, draw != Draw::None) {
            let mut always = StencilDescriptor::default();
            always.stencil_compare_function = CompareFunction::Always;
            always.depth_stencil_pass_operation = StencilOperation::Replace;
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = CompareFunction::Always;
            state.depth_write_enabled = descriptor.depth_attachment.texture.is_some();
            if descriptor.stencil_attachment.texture.is_some() {
                state.front_face_stencil = always;
                state.back_face_stencil = always;
            }
            let state = self.device.new_depth_stencil_state(&state).unwrap();
            encoder.set_render_pipeline_state(pipeline).unwrap();
            encoder.set_depth_stencil_state(&state).unwrap();
            encoder.set_stencil_reference_value(0x41);
            let corners: [f32; 6] = match draw {
                Draw::Full => [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
                _ => [2.0, 2.0, 3.0, 2.0, 2.0, 3.0],
            };
            let bytes: Vec<u8> = corners.iter().flat_map(|v| v.to_le_bytes()).collect();
            let corners = buffer_with(&self.device, &bytes).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            let count = if draw == Draw::Empty { 0 } else { 3 };
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, count, 1, 0)
                .unwrap();
            held.push(corners);
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        drop(held);
    }

    /// Encode `descriptor`'s pass with the triangle over the whole target drawn through
    /// `pipeline` under `viewport` and `scissor` where given, then commit and wait.
    fn run_within(
        &self,
        descriptor: &RenderPassDescriptor,
        pipeline: &RenderPipelineState,
        (viewport, scissor): (Option<Viewport>, Option<ScissorRect>),
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        encoder.set_render_pipeline_state(pipeline).unwrap();
        if let Some(viewport) = viewport {
            encoder.set_viewport(viewport).unwrap();
        }
        if let Some(scissor) = scissor {
            encoder.set_scissor_rect(scissor).unwrap();
        }
        let corners: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let bytes: Vec<u8> = corners.iter().flat_map(|v| v.to_le_bytes()).collect();
        let corners = buffer_with(&self.device, &bytes).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// A private depth or stencil texture of `pf` and `shape` cleared whole to `depth` and
    /// `stencil` by a pass of its own.
    pub(super) fn old_depth_stencil(
        &self,
        pf: PixelFormat,
        shape: Shape,
        (depth, stencil): (bool, bool),
    ) -> Texture {
        let texture = self.texture(pf, shape, TextureUsage::RENDER_TARGET);
        let mut descriptor = RenderPassDescriptor::new();
        if depth {
            descriptor.depth_attachment.texture = Some(texture.clone());
            descriptor.depth_attachment.load_action = LoadAction::Clear;
            descriptor.depth_attachment.clear_depth = 0.75;
            descriptor.depth_attachment.store_action = StoreAction::Store;
        }
        if stencil {
            descriptor.stencil_attachment.texture = Some(texture.clone());
            descriptor.stencil_attachment.load_action = LoadAction::Clear;
            descriptor.stencil_attachment.clear_stencil = 0x22;
            descriptor.stencil_attachment.store_action = StoreAction::Store;
        }
        self.run(&descriptor, Draw::None, None);
        texture
    }
}

/// A texture's shape: its side, samples, levels and slices, and whether it is shared.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) side: u64,
    pub(super) samples: u64,
    pub(super) levels: u64,
    pub(super) slices: u64,
    pub(super) shared: bool,
}

pub(super) const PLAIN: Shape = Shape {
    side: SIDE,
    samples: 1,
    levels: 1,
    slices: 1,
    shared: false,
};

/// The texels Metal leaves in a `side`² level: `drawn` over the render area `area` where the
/// pass draws there, `cleared` over the rest of the clear's reach `reach`, and `old` past it.
fn expected(
    side: u64,
    area: [u64; 2],
    reach: [u64; 2],
    texels: (Option<&[u8]>, &[u8], &[u8]),
) -> Vec<u8> {
    let (drawn, cleared, old) = texels;
    let mut out = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let texel = if x < area[0] && y < area[1] {
                drawn.unwrap_or(cleared)
            } else if x < reach[0] && y < reach[1] {
                cleared
            } else {
                old
            };
            out.extend_from_slice(texel);
        }
    }
    out
}

/// A pass over color attachment `texture` loaded by `load` (clearing to [`RED`]) and stored,
/// over a render area of `area` (the target size).
fn color_pass(texture: &Texture, load: LoadAction, area: [u64; 2]) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(texture.clone());
    target.load_action = load;
    target.clear_color = ClearColor::new(0.5, 0.0, 0.0, 1.0);
    target.store_action = StoreAction::Store;
    descriptor.render_target_width = area[0];
    descriptor.render_target_height = area[1];
    descriptor
}

pub(super) fn lane() -> (Messages, ScratchCache, Lane) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    (messages, cache, Lane::new(device))
}

pub(super) fn quiet(lane: Lane, messages: &Messages) {
    drop(lane);
    layer_reported_nothing_at_all(messages);
}

/// A case of `a_clear_reaches_past_the_render_area_as_metal_s`: its name, the render area, the
/// load, the draw, the reach Metal clears to, and whether the pass's first instance records inline.
type ColorCase = (&'static str, [u64; 2], LoadAction, Draw, [u64; 2], bool);

/// How many of the device's passes recorded their first instance inline so far.
fn inline_passes(lane: &Lane) -> u64 {
    lane.device.work_counts().inline_passes
}

/// A color attachment's `Clear` load reaches past the render area as Metal's (the probe's
/// `c_rgba8_draw`, `c_rgba8_nodraw`, `c_rgba8_a40x24_draw`, `c_rgba8_nothing`,
/// `c_rgba8_a100x100_draw`, `c_rgba8_small_depth_draw`, `c_rgba8_load_draw`; our probe's
/// `c_rgba8_level1_draw`, `c_rgba8_layered_nodraw`, `c_rgba8_emptydraw`): to whole 32x32 tiles
/// in a pass with a draw call (one wholly outside the target, or of no vertex, included), the
/// whole attachment in one without, the smaller attachment's extent setting the render area
/// alike, clipped to the attachment, at the attachment's level alone, and to the tiles of each
/// layer of a layered pass without a draw call; a `Load` keeps the texels past the render area. So does a texture without
/// `RenderTarget` usage, which renders through a stand-in. A pass whose draw records renders its
/// first instance inline over the reach where nothing else rules it out
/// (`render/inline_first.rs`), and one without such a draw, or with a depth attachment or a
/// stand-in, clears the reach before the pass: both Metal's answer.
#[test]
fn a_clear_reaches_past_the_render_area_as_metal_s() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let target = TextureUsage::RENDER_TARGET;
    let pipeline = lane.pipeline(&[rgba8], (None, None), 1);
    let small_depth = lane.pipeline(&[rgba8], (Some(PixelFormat::Depth32Float), None), 1);
    let cases: [ColorCase; 7] = [
        (
            "c_rgba8_draw",
            [AREA, AREA],
            LoadAction::Clear,
            Draw::Full,
            [64, 64],
            true,
        ),
        (
            "c_rgba8_nodraw",
            [AREA, AREA],
            LoadAction::Clear,
            Draw::None,
            [SIDE, SIDE],
            false,
        ),
        (
            "c_rgba8_a40x24_draw",
            [40, 24],
            LoadAction::Clear,
            Draw::Full,
            [64, 32],
            true,
        ),
        (
            "c_rgba8_nothing",
            [AREA, AREA],
            LoadAction::Clear,
            Draw::Outside,
            [64, 64],
            true,
        ),
        (
            "c_rgba8_emptydraw",
            [AREA, AREA],
            LoadAction::Clear,
            Draw::Empty,
            [64, 64],
            false,
        ),
        (
            "c_rgba8_a100x100_draw",
            [100, 100],
            LoadAction::Clear,
            Draw::Full,
            [SIDE, SIDE],
            true,
        ),
        (
            "c_rgba8_load_draw",
            [AREA, AREA],
            LoadAction::Load,
            Draw::Full,
            [AREA, AREA],
            true,
        ),
    ];
    for (name, area, load, draw, reach, inline) in cases {
        let texture = lane.old_color(rgba8, PLAIN, target);
        let before = inline_passes(&lane);
        lane.run(&color_pass(&texture, load, area), draw, Some(&pipeline));
        assert_eq!(
            inline_passes(&lane) - before,
            u64::from(inline),
            "{name}: inline"
        );
        let drawn = (draw == Draw::Full).then_some(&DRAWN[..]);
        let cleared = if load == LoadAction::Clear {
            &RED
        } else {
            &OLD
        };
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            expected(SIDE, area, reach, (drawn, cleared, &OLD)),
            "{name}"
        );
    }
    // The render area of a smaller depth attachment.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let depth = lane.texture(
        PixelFormat::Depth32Float,
        Shape {
            side: AREA,
            ..PLAIN
        },
        target,
    );
    let mut descriptor = color_pass(&texture, LoadAction::Clear, [0, 0]);
    descriptor.depth_attachment.texture = Some(depth);
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.store_action = StoreAction::DontCare;
    let before = inline_passes(&lane);
    lane.run(&descriptor, Draw::Full, Some(&small_depth));
    assert_eq!(
        inline_passes(&lane),
        before,
        "c_rgba8_small_depth_draw: not inline"
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        expected(SIDE, [AREA, AREA], [64, 64], (Some(&DRAWN), &RED, &OLD)),
        "c_rgba8_small_depth_draw"
    );
    // Level 1 of a mipmapped 256x256 texture: its level 0 keeps its texels.
    let mipmapped = Shape {
        side: 2 * SIDE,
        levels: 2,
        ..PLAIN
    };
    let texture = lane.old_color(rgba8, mipmapped, target);
    let mut descriptor = color_pass(&texture, LoadAction::Clear, [AREA, AREA]);
    descriptor.color_attachments[0].level = 1;
    let before = inline_passes(&lane);
    lane.run(&descriptor, Draw::Full, Some(&pipeline));
    assert_eq!(
        inline_passes(&lane) - before,
        1,
        "c_rgba8_level1_draw: inline"
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        OLD.repeat((4 * SIDE * SIDE) as usize),
        "c_rgba8_level1_draw, level 0"
    );
    assert_eq!(
        lane.read(&texture, (1, 0), 4, BlitOption::None),
        expected(SIDE, [AREA, AREA], [64, 64], (Some(&DRAWN), &RED, &OLD)),
        "c_rgba8_level1_draw, level 1"
    );
    // A layered pass of two slices that does not draw clears each layer's tiles, not all of it.
    let layered = Shape { slices: 2, ..PLAIN };
    let texture = lane.old_color(rgba8, layered, target);
    let mut descriptor = color_pass(&texture, LoadAction::Clear, [AREA, AREA]);
    descriptor.render_target_array_length = 2;
    lane.run(&descriptor, Draw::None, None);
    for slice in 0..2 {
        assert_eq!(
            lane.read(&texture, (0, slice), 4, BlitOption::None),
            expected(SIDE, [AREA, AREA], [64, 64], (None, &RED, &OLD)),
            "c_rgba8_layered_nodraw, slice {slice}"
        );
    }
    // A texture without `RenderTarget` usage, through its stand-in.
    let texture = lane.old_color(rgba8, PLAIN, TextureUsage::SHADER_READ);
    let before = inline_passes(&lane);
    lane.run(
        &color_pass(&texture, LoadAction::Clear, [AREA, AREA]),
        Draw::Full,
        Some(&pipeline),
    );
    assert_eq!(
        inline_passes(&lane),
        before,
        "a stand-in's texture: not inline"
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        expected(SIDE, [AREA, AREA], [64, 64], (Some(&DRAWN), &RED, &OLD)),
        "a stand-in's texture"
    );
    quiet(lane, &messages);
}

/// Two color attachments of a pass over a 40x24 render area, color 0 a 128x128 texture cleared,
/// color 1 one the fragment function leaves alone (its write mask none), white drawn over the
/// whole target: where both are 128x128 textures cleared, the clears reach alike, to 64x32, and
/// the pass records its first instance inline over that reach (`render/inline_first.rs`); where
/// color 1 is a 48x48 texture cleared, its clear reaches 48x32, and where it is loaded it keeps
/// its texels past the render area: the pass records as before, counted by the reach the
/// attachments do not share (`WorkCounts::passes_not_inline_actions_unequal_reach`). Each texture
/// holds Metal's answer (the probe's `c_rgba8_a40x24_draw`, each attachment's reach its own).
#[test]
fn colors_whose_clears_reach_alike_record_inline_over_their_reach() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let target = TextureUsage::RENDER_TARGET;
    let mut descriptor = lane.descriptor(&[rgba8, rgba8], (None, None), 1);
    let mut leaves = ColorAttachmentDescriptor::new(rgba8);
    leaves.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[1] = Some(leaves);
    let pipeline = lane
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let area = [40, 24];
    let unequal = |lane: &Lane| {
        lane.device
            .work_counts()
            .passes_not_inline_actions_unequal_reach
    };
    let small = Shape { side: 48, ..PLAIN };
    // Color 1's side, load, the reach Metal clears it to, and whether the pass records inline.
    let cases = [
        ("alike", PLAIN, LoadAction::Clear, [64, 32], true),
        (
            "a smaller texture",
            small,
            LoadAction::Clear,
            [48, 32],
            false,
        ),
        ("loaded", PLAIN, LoadAction::Load, area, false),
    ];
    for (name, shape, load, reach, inline) in cases {
        let first = lane.old_color(rgba8, PLAIN, target);
        let second = lane.old_color(rgba8, shape, target);
        let mut pass = color_pass(&first, LoadAction::Clear, area);
        let other = &mut pass.color_attachments[1];
        other.texture = Some(second.clone());
        other.load_action = load;
        other.clear_color = ClearColor::new(0.5, 0.0, 0.0, 1.0);
        other.store_action = StoreAction::Store;
        let (before, unequal_before) = (inline_passes(&lane), unequal(&lane));
        lane.run(&pass, Draw::Full, Some(&pipeline));
        assert_eq!(
            inline_passes(&lane) - before,
            u64::from(inline),
            "{name}: inline"
        );
        assert_eq!(
            unequal(&lane) - unequal_before,
            u64::from(!inline),
            "{name}: counted by its reach"
        );
        assert_eq!(
            lane.read(&first, (0, 0), 4, BlitOption::None),
            expected(SIDE, area, [64, 32], (Some(&DRAWN), &RED, &OLD)),
            "{name}: color 0"
        );
        let cleared = match load {
            LoadAction::Clear => &RED,
            _ => &OLD,
        };
        assert_eq!(
            lane.read(&second, (0, 0), 4, BlitOption::None),
            expected(shape.side, area, reach, (None, cleared, &OLD)),
            "{name}: color 1"
        );
    }
    quiet(lane, &messages);
}

/// A clear of a `half` NaN writes Metal's canonical quiet NaN (Metal probe case `nan_1_half_nan` on
/// the M3 Max and an M4: `0x7e00`) over the reach any clear has, through the scratch image of an
/// integer format the runtime clears it in (`nan_clear.rs`): to the tiles in a pass with a draw
/// call, over the whole attachment in one without, and through a stand-in. A 4-sample attachment's
/// clear runs valid (Metal's resolve of its NaN samples is unmeasured).
#[test]
fn a_nan_clear_reaches_as_metal_s() {
    let (messages, _cache, lane) = lane();
    let rg16 = PixelFormat::RG16Float;
    let pipeline = lane.pipeline(&[rg16], (None, None), 1);
    let nan = [0x00, 0x7e, 0x00, 0x3c];
    let pass = |texture: &Texture| {
        let mut descriptor = color_pass(texture, LoadAction::Clear, [AREA, AREA]);
        descriptor.color_attachments[0].clear_color = ClearColor::new(f64::NAN, 1.0, 0.5, f64::NAN);
        descriptor
    };
    let cases = [
        (
            "a draw call",
            TextureUsage::RENDER_TARGET,
            Draw::Outside,
            [64, 64],
        ),
        (
            "no draw call",
            TextureUsage::RENDER_TARGET,
            Draw::None,
            [SIDE, SIDE],
        ),
        (
            "a stand-in",
            TextureUsage::SHADER_READ,
            Draw::Outside,
            [64, 64],
        ),
    ];
    for (name, usage, draw, reach) in cases {
        let texture = lane.old_color(rg16, PLAIN, usage);
        lane.run(&pass(&texture), draw, Some(&pipeline));
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            expected(SIDE, [AREA, AREA], reach, (None, &nan, &OLD)),
            "{name}"
        );
    }
    let four = Shape {
        samples: 4,
        ..PLAIN
    };
    let multisampled = lane.texture(rg16, four, TextureUsage::RENDER_TARGET);
    lane.run(&pass(&multisampled), Draw::None, None);
    quiet(lane, &messages);
}

/// Depth and stencil attachments' `Clear` loads reach past the render area as a color's (the
/// probe's `d_d32_draw`, `d_d32_nodraw`, `d_d16_draw`, `s_s8_draw`, `ds_d32s8_draw`, and a
/// 4-sample color's `c_rgba8_ms4_draw`, read through a second pass that loads and resolves it
/// whole): 0.25 drawn and 0.5 cleared past the render area to the tiles, 0.75 past them; the
/// stencil 0x41 and 0x40, 0x22 past them.
#[test]
fn depth_stencil_and_multisampled_clears_reach_past_the_render_area_as_metal_s() {
    let (messages, _cache, lane) = lane();
    let area = [AREA, AREA];
    let word = |v: f32| v.to_le_bytes();
    for (name, pf, aspects, draw) in [
        (
            "d_d32_draw",
            PixelFormat::Depth32Float,
            (true, false),
            Draw::Full,
        ),
        (
            "d_d32_nodraw",
            PixelFormat::Depth32Float,
            (true, false),
            Draw::None,
        ),
        (
            "s_s8_draw",
            PixelFormat::Stencil8,
            (false, true),
            Draw::Full,
        ),
        (
            "ds_d32s8_draw",
            PixelFormat::Depth32FloatStencil8,
            (true, true),
            Draw::Full,
        ),
    ] {
        let texture = lane.old_depth_stencil(pf, PLAIN, aspects);
        let (depth, stencil) = aspects;
        let pipeline = lane.pipeline(&[], (depth.then_some(pf), stencil.then_some(pf)), 1);
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.render_target_width = AREA;
        descriptor.render_target_height = AREA;
        if depth {
            descriptor.depth_attachment.texture = Some(texture.clone());
            descriptor.depth_attachment.load_action = LoadAction::Clear;
            descriptor.depth_attachment.clear_depth = 0.5;
            descriptor.depth_attachment.store_action = StoreAction::Store;
        }
        if stencil {
            descriptor.stencil_attachment.texture = Some(texture.clone());
            descriptor.stencil_attachment.load_action = LoadAction::Clear;
            descriptor.stencil_attachment.clear_stencil = 0x40;
            descriptor.stencil_attachment.store_action = StoreAction::Store;
        }
        lane.run(&descriptor, draw, Some(&pipeline));
        let reach = match draw {
            Draw::None => [SIDE, SIDE],
            _ => [64, 64],
        };
        let shows = draw == Draw::Full;
        let combined = pf == PixelFormat::Depth32FloatStencil8;
        if depth {
            let option = match combined {
                true => BlitOption::DepthFromDepthStencil,
                false => BlitOption::None,
            };
            let drawn = word(0.25);
            assert_eq!(
                lane.read(&texture, (0, 0), 4, option),
                expected(
                    SIDE,
                    area,
                    reach,
                    (shows.then_some(&drawn[..]), &word(0.5), &word(0.75))
                ),
                "{name}, depth"
            );
        }
        if stencil {
            let option = match combined {
                true => BlitOption::StencilFromDepthStencil,
                false => BlitOption::None,
            };
            assert_eq!(
                lane.read(&texture, (0, 0), 1, option),
                expected(
                    SIDE,
                    area,
                    reach,
                    (shows.then_some(&[0x41][..]), &[0x40], &[0x22])
                ),
                "{name}, stencil"
            );
        }
    }
    // A 4-sample color attachment, its samples set to blue by a pass over all of it first.
    let rgba8 = PixelFormat::RGBA8Unorm;
    let msaa = lane.texture(
        rgba8,
        Shape {
            samples: 4,
            ..PLAIN
        },
        TextureUsage::RENDER_TARGET,
    );
    let mut fill = color_pass(&msaa, LoadAction::Clear, [SIDE, SIDE]);
    fill.color_attachments[0].clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    lane.run(&fill, Draw::None, None);
    let pipeline = lane.pipeline(&[rgba8], (None, None), 4);
    lane.run(
        &color_pass(&msaa, LoadAction::Clear, area),
        Draw::Full,
        Some(&pipeline),
    );
    let resolved = lane.texture(rgba8, PLAIN, TextureUsage::RENDER_TARGET);
    let mut read = color_pass(&msaa, LoadAction::Load, [SIDE, SIDE]);
    read.color_attachments[0].store_action = StoreAction::MultisampleResolve;
    read.color_attachments[0].resolve_texture = Some(resolved.clone());
    lane.run(&read, Draw::None, None);
    assert_eq!(
        lane.read(&resolved, (0, 0), 4, BlitOption::None),
        expected(
            SIDE,
            area,
            [64, 64],
            (Some(&DRAWN), &RED, &[0x00, 0x00, 0xff, 0xff])
        ),
        "c_rgba8_ms4_draw"
    );
    quiet(lane, &messages);
}

/// A pass without a draw call that clears nothing and resolves nothing keeps every texel, Metal's
/// answer, and the runtime records nothing for it (`WorkCounts::untouched_passes_skipped`): a color
/// loaded `DontCare` or `Load` and stored `Store` or `DontCare`, and beside a depth loaded
/// `DontCare` and stored. A `Clear` load clears the whole attachment there, a draw call of no
/// vertex keeps the texels, and a pass with a visibility result buffer keeps them: each records its
/// pass as before. The pass beside a depth, whose first instance never records inline, began its
/// draws' secondary as it opened: given up unused (`WorkCounts::unrun_instances_ended`), it is
/// never replayed (`CommandBuffer::give_up_raw`).
#[test]
fn a_pass_that_changes_no_texel_records_nothing() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let target = TextureUsage::RENDER_TARGET;
    let pipeline = lane.pipeline(&[rgba8], (None, None), 1);
    let whole = [SIDE, SIDE];
    let skipped = || lane.device.work_counts().untouched_passes_skipped;
    let old = OLD.repeat((SIDE * SIDE) as usize);
    for (load, store) in [
        (LoadAction::DontCare, StoreAction::Store),
        (LoadAction::Load, StoreAction::Store),
        (LoadAction::Load, StoreAction::DontCare),
    ] {
        let texture = lane.old_color(rgba8, PLAIN, target);
        let mut descriptor = color_pass(&texture, load, whole);
        descriptor.color_attachments[0].store_action = store;
        let before = skipped();
        lane.run(&descriptor, Draw::None, None);
        let what = format!("{load:?} and {store:?}, no draw call");
        assert_eq!(skipped() - before, 1, "{what}: passes skipped");
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            old,
            "{what}: the texels"
        );
    }
    // Beside a depth loaded `DontCare` and stored, which keeps its texels too.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let depth = PixelFormat::Depth32Float;
    let kept = lane.old_depth_stencil(depth, PLAIN, (true, false));
    let mut descriptor = color_pass(&texture, LoadAction::Load, whole);
    descriptor.depth_attachment.texture = Some(kept.clone());
    descriptor.depth_attachment.load_action = LoadAction::DontCare;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let before = skipped();
    let counted = lane.device.work_counts();
    lane.run(&descriptor, Draw::None, None);
    assert_eq!(skipped() - before, 1, "a depth beside: passes skipped");
    let after = lane.device.work_counts();
    let unrun = after.unrun_instances_ended - counted.unrun_instances_ended;
    assert_eq!(unrun, 1, "a depth beside: secondaries given up");
    eprintln!(
        "a pass beside a depth that changes nothing on {}: {unrun} secondaries given up",
        lane.device.name()
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        old,
        "a depth beside: the color's texels"
    );
    assert_eq!(
        lane.read(&kept, (0, 0), 4, BlitOption::None),
        0.75f32.to_le_bytes().repeat((SIDE * SIDE) as usize),
        "a depth beside: the depth's texels"
    );
    // A `Clear` load clears the whole attachment in a pass without a draw call.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let before = skipped();
    lane.run(
        &color_pass(&texture, LoadAction::Clear, whole),
        Draw::None,
        None,
    );
    assert_eq!(skipped() - before, 0, "a Clear load: passes skipped");
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        RED.repeat((SIDE * SIDE) as usize),
        "a Clear load: the texels"
    );
    // A draw call of no vertex, and a visibility result buffer: recorded, the texels kept.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let before = skipped();
    lane.run(
        &color_pass(&texture, LoadAction::DontCare, whole),
        Draw::Empty,
        Some(&pipeline),
    );
    assert_eq!(
        skipped() - before,
        0,
        "a draw call of no vertex: passes skipped"
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        old,
        "a draw call of no vertex: the texels"
    );
    let texture = lane.old_color(rgba8, PLAIN, target);
    let mut descriptor = color_pass(&texture, LoadAction::Load, whole);
    descriptor.visibility_result_buffer = Some(buffer_with(&lane.device, &[0; 64]).unwrap());
    let before = skipped();
    lane.run(&descriptor, Draw::None, None);
    assert_eq!(
        skipped() - before,
        0,
        "a visibility result buffer: passes skipped"
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        old,
        "a visibility result buffer: the texels"
    );
    drop((pipeline, kept));
    quiet(lane, &messages);
}

/// A `DontCare` load of an attachment the pass stores: kept in a pass
/// that does not draw (a draw call of no vertex, or through a pipeline that rasterizes nothing,
/// included), as Metal keeps it, with no notice; cleared to zero over every tile, Metal's answer
/// with no notice, where such a pass clears and stores its depth (the clear rule); in one that
/// draws, cleared to zero (a color) or the clear value (depth) over the render area and the tiles
/// past it, Metal's answer where the draws touch every tile (a triangle over the whole target) and
/// a near-equivalent where they touch none (a triangle wholly outside it, after which Metal's
/// texels are the old ones), announced once.
#[test]
fn a_dont_care_load_is_metal_s_in_the_tiles_draws_touch() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let target = TextureUsage::RENDER_TARGET;
    let pipeline = lane.pipeline(&[rgba8], (None, None), 1);
    let area = [AREA, AREA];
    let texture = lane.old_color(rgba8, PLAIN, target);
    lane.run(
        &color_pass(&texture, LoadAction::DontCare, area),
        Draw::Empty,
        Some(&pipeline),
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        OLD.repeat((SIDE * SIDE) as usize),
        "dl_rgba8_emptydraw"
    );
    // A draw call of no vertex in a pass that clears and stores its depth reaches every tile: zero
    // over the render area rounded up to whole tiles, as the depth's clear reaches.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let depth = PixelFormat::Depth32Float;
    let cleared = lane.old_depth_stencil(depth, PLAIN, (true, false));
    let mut descriptor = color_pass(&texture, LoadAction::DontCare, area);
    descriptor.depth_attachment.texture = Some(cleared.clone());
    descriptor.depth_attachment.load_action = LoadAction::Clear;
    descriptor.depth_attachment.clear_depth = 0.5;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let with_depth = lane.pipeline(&[rgba8], (Some(depth), None), 1);
    lane.run(&descriptor, Draw::Empty, Some(&with_depth));
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        expected(SIDE, area, [64, 64], (None, &ZERO, &OLD)),
        "dl_rgba8_emptydraw_depth_cleared"
    );
    let word = |v: f32| v.to_le_bytes();
    assert_eq!(
        lane.read(&cleared, (0, 0), 4, BlitOption::None),
        expected(SIDE, area, [64, 64], (None, &word(0.5), &word(0.75))),
        "dl_rgba8_emptydraw_depth_cleared, depth"
    );
    // A draw through a pipeline that rasterizes nothing reaches no tile: the owned fixture
    // `rvc_draw_bases`'s vertex function, which returns nothing, as Metal requires there, and
    // writes 16 bytes at element `(iid * 8 & 24) | (vid & 7)` of buffer 0.
    let bitcode = llvm_assemble(&engine_dir().join("rvc_draw_bases.ll")).unwrap();
    let library = lane.device.new_library_with_air(&bitcode).unwrap();
    let vertex = library
        .new_function_with_name("draw_bases_in_a_vertex_function")
        .unwrap()
        .function_descriptor();
    let mut unrasterized = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    unrasterized.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba8));
    unrasterized.rasterization_enabled = false;
    let unrasterized_made = made_render_kinds(&lane.device, &unrasterized);
    let unrasterized = lane
        .device
        .new_render_pipeline_state(&unrasterized, &PipelineOptions::default())
        .unwrap();
    let texture = lane.old_color(rgba8, PLAIN, target);
    let written = buffer_with(&lane.device, &[0; 512]).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&color_pass(&texture, LoadAction::DontCare, area))
        .unwrap();
    encoder.set_render_pipeline_state(&unrasterized).unwrap();
    encoder.set_vertex_buffer(Some(&written), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        OLD.repeat((SIDE * SIDE) as usize),
        "dl_rgba8_unrasterized"
    );
    // The pipelines' makings announce kinds of their own on some devices (their precision notes
    // and color conversions), and the reads' buffers are `Shared`.
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    first.extend(lane.made(&[rgba8], (None, None), 1));
    first.extend(lane.made(&[rgba8], (Some(depth), None), 1));
    first.extend(unrasterized_made);
    let announced = assert_made_once(&messages, &[], &first);
    assert!(
        !announced.contains_key(&Degradation::DontCareLoadTiles),
        "dl_rgba8_emptydraw, dl_rgba8_emptydraw_depth_cleared or dl_rgba8_unrasterized \
         announced {announced:?}"
    );
    for (name, draw, metal) in [
        (
            "dl_rgba8_full",
            Draw::Full,
            expected(SIDE, area, [64, 64], (Some(&DRAWN), &ZERO, &OLD)),
        ),
        (
            "dl_rgba8_nodraw",
            Draw::None,
            OLD.repeat((SIDE * SIDE) as usize),
        ),
    ] {
        let texture = lane.old_color(rgba8, PLAIN, target);
        lane.run(
            &color_pass(&texture, LoadAction::DontCare, area),
            draw,
            Some(&pipeline),
        );
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            metal,
            "{name}"
        );
    }
    // The near-equivalent: Metal keeps every texel, as no draw touches a tile.
    let texture = lane.old_color(rgba8, PLAIN, target);
    lane.run(
        &color_pass(&texture, LoadAction::DontCare, area),
        Draw::Outside,
        Some(&pipeline),
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        expected(SIDE, area, [64, 64], (None, &ZERO, &OLD)),
        "dl_rgba8_nothing, the near-equivalent"
    );
    let texture = lane.old_depth_stencil(depth, PLAIN, (true, false));
    let mut descriptor = RenderPassDescriptor::new();
    descriptor.render_target_width = AREA;
    descriptor.render_target_height = AREA;
    descriptor.depth_attachment.texture = Some(texture.clone());
    descriptor.depth_attachment.load_action = LoadAction::DontCare;
    descriptor.depth_attachment.clear_depth = 0.5;
    descriptor.depth_attachment.store_action = StoreAction::Store;
    let depth_only = lane.pipeline(&[], (Some(depth), None), 1);
    lane.run(&descriptor, Draw::Full, Some(&depth_only));
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        expected(
            SIDE,
            area,
            [64, 64],
            (Some(&word(0.25)), &word(0.5), &word(0.75))
        ),
        "dl_d32_full"
    );
    let made = lane.made(&[], (Some(depth), None), 1);
    assert_made_once(&messages, &DONT_CARE_LOADS, &made);
    quiet(lane, &messages);
}

/// A `DontCare` load of a 128x128 attachment the pass stores, over the whole of it, in a pass
/// whose triangle over the whole target draws through a 40x40 viewport at (48, 48), or under a
/// 40x40 scissor rectangle there (the probe's shapes on an Apple M4: the viewport and the
/// scissor each bound the tiles a draw zeroes exactly): the rectangle holds the draw's color, the
/// rest of its four 32x32 tiles, from (32, 32) to (96, 96), zero, and every other tile keeps its
/// texels, Metal's answer. A pass of a 16x16 tile (`tileWidth`, `tileHeight`; XRB's probes
/// pdc4-batch: the pass's tile size sets the grid Metal zeroes) zeroes only its tiles from
/// (48, 48) to (96, 96). The pass announces `dont-care-load-tiles` once (in the tiles inside, where a primitive covers
/// less of them, the runtime still clears every one). A pass whose one draw has an empty scissor
/// rectangle writes no pixel: it keeps every texel and announces nothing, as Metal's.
#[test]
fn a_dont_care_load_clears_only_the_tiles_a_draw_s_viewport_and_scissor_reach() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let target = TextureUsage::RENDER_TARGET;
    let pipeline = lane.pipeline(&[rgba8], (None, None), 1);
    let whole = [SIDE, SIDE];
    // No pixel: the scissor rectangle is empty.
    let texture = lane.old_color(rgba8, PLAIN, target);
    let empty = ScissorRect::new(48, 48, 0, 40);
    lane.run_within(
        &color_pass(&texture, LoadAction::DontCare, whole),
        &pipeline,
        (None, Some(empty)),
    );
    assert_eq!(
        lane.read(&texture, (0, 0), 4, BlitOption::None),
        OLD.repeat((SIDE * SIDE) as usize),
        "dl_rgba8_empty_scissor"
    );
    // The pipeline's making announces kinds of its own on some devices, and the reads'
    // buffers are `Shared`.
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&lane.device),
    )]);
    first.extend(lane.made(&[rgba8], (None, None), 1));
    let announced = assert_made_once(&messages, &[], &first);
    assert!(
        !announced.contains_key(&Degradation::DontCareLoadTiles),
        "dl_rgba8_empty_scissor announced {announced:?}"
    );
    // Metal's texels where the draw's 40x40 rectangle at (48, 48) reaches the tiles from `from`
    // to (96, 96).
    let metal = |from: u64| -> Vec<u8> {
        let inside = |x: u64, y: u64| (48..88).contains(&x) && (48..88).contains(&y);
        let tiled = |x: u64, y: u64| (from..96).contains(&x) && (from..96).contains(&y);
        (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .flat_map(|(x, y)| match (inside(x, y), tiled(x, y)) {
                (true, _) => DRAWN,
                (false, true) => ZERO,
                (false, false) => OLD,
            })
            .collect()
    };
    let viewport = Viewport::new(48.0, 48.0, 40.0, 40.0, 0.0, 1.0);
    let scissor = ScissorRect::new(48, 48, 40, 40);
    for (name, bound, tile) in [
        ("dl_rgba8_viewport40", (Some(viewport), None), None),
        ("dl_rgba8_scissor40", (None, Some(scissor)), None),
        ("dl_rgba8_scissor40_tile16", (None, Some(scissor)), Some(16)),
    ] {
        let texture = lane.old_color(rgba8, PLAIN, target);
        let mut descriptor = color_pass(&texture, LoadAction::DontCare, whole);
        if let Some(side) = tile {
            descriptor.tile_width = side;
            descriptor.tile_height = side;
        }
        lane.run_within(&descriptor, &pipeline, bound);
        assert_eq!(
            lane.read(&texture, (0, 0), 4, BlitOption::None),
            metal(tile.map_or(32, |_| 48)),
            "{name}"
        );
    }
    assert_made_once(&messages, &DONT_CARE_LOADS, &BTreeSet::new());
    quiet(lane, &messages);
}

/// [`a_dont_care_load_clears_only_the_tiles_a_draw_s_viewport_and_scissor_reach`]'s pass over a
/// texture without `RenderTarget` usage, which renders through a stand-in: a color's whole 128x128
/// render area under the 40x40 scissor rectangle at (48, 48), the same over a 40x40 render area
/// under a 16x16 one at (8, 8) (the draw reaches the tile at the origin alone, whose copy back
/// covers the clear's 64x64 reach), and a `Depth32Float` texture's whole render area under the
/// first. Each holds what a texture with that usage holds: the draw's color where it draws, zero (a
/// depth, the clear value) over the rest of the tiles it reaches, and every other tile keeps its
/// texels, Metal's answer, the stand-in's copy back returning them as they came.
#[test]
fn a_dont_care_load_through_a_stand_in_keeps_the_tiles_no_draw_reaches() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = lane.pipeline(&[rgba8], (None, None), 1);
    // Metal's texels where a draw's `scissor` rectangle reaches the 32x32 tiles from `tiles.0`
    // to `tiles.1` (each side alike).
    let metal = |scissor: [u64; 4], tiles: (u64, u64)| -> Vec<u8> {
        let [x0, y0, width, height] = scissor;
        let inside =
            |x: u64, y: u64| (x0..x0 + width).contains(&x) && (y0..y0 + height).contains(&y);
        let tiled =
            |x: u64, y: u64| (tiles.0..tiles.1).contains(&x) && (tiles.0..tiles.1).contains(&y);
        (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .flat_map(|(x, y)| match (inside(x, y), tiled(x, y)) {
                (true, _) => DRAWN,
                (false, true) => ZERO,
                (false, false) => OLD,
            })
            .collect()
    };
    for (name, area, scissor, tiles) in [
        (
            "dl_rgba8_scissor40_stand_in",
            [SIDE, SIDE],
            [48, 48, 40, 40],
            (32, 96),
        ),
        (
            "dl_rgba8_a40_scissor16_stand_in",
            [AREA, AREA],
            [8, 8, 16, 16],
            (0, 32),
        ),
    ] {
        let [x, y, width, height] = scissor;
        for usage in [TextureUsage::RENDER_TARGET, TextureUsage::SHADER_READ] {
            let texture = lane.old_color(rgba8, PLAIN, usage);
            lane.run_within(
                &color_pass(&texture, LoadAction::DontCare, area),
                &pipeline,
                (None, Some(ScissorRect::new(x, y, width, height))),
            );
            assert_eq!(
                lane.read(&texture, (0, 0), 4, BlitOption::None),
                metal(scissor, tiles),
                "{name}, {usage:?}"
            );
        }
    }
    // A depth texture filled with 0.75, then loaded with `DontCare` and stored.
    let d32 = PixelFormat::Depth32Float;
    let depth_only = lane.pipeline(&[], (Some(d32), None), 1);
    let filled = 0.75f32.to_le_bytes();
    let mut kept = Vec::new();
    for usage in [TextureUsage::RENDER_TARGET, TextureUsage::SHADER_READ] {
        let texture = lane.texture(d32, PLAIN, usage);
        let mut descriptor = RenderPassDescriptor::new();
        descriptor.depth_attachment.texture = Some(texture.clone());
        descriptor.depth_attachment.load_action = LoadAction::Clear;
        descriptor.depth_attachment.clear_depth = 0.75;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        lane.run(&descriptor, Draw::None, None);
        descriptor.depth_attachment.load_action = LoadAction::DontCare;
        lane.run_within(
            &descriptor,
            &depth_only,
            (None, Some(ScissorRect::new(48, 48, 40, 40))),
        );
        kept.push(lane.read(&texture, (0, 0), 4, BlitOption::None));
    }
    let outside = |at: usize| {
        let (x, y) = ((at as u64) % SIDE, (at as u64) / SIDE);
        !((32..96).contains(&x) && (32..96).contains(&y))
    };
    for (at, texel) in kept[0].chunks(4).enumerate() {
        if outside(at) {
            assert_eq!(texel, filled, "dl_d32_scissor40, texel {at}");
        }
    }
    if let [rendered, stand_in] = &kept[..] {
        assert_eq!(stand_in, rendered, "dl_d32_scissor40_stand_in");
    }
    // The textures without `RenderTarget` usage render through stand-ins
    // (`undeclared-usage-shadow`); the pipelines' makings announce kinds of their own on some
    // devices; and the reads' buffers are `Shared`.
    let mut first = forced(&[
        (Degradation::UndeclaredUsageShadow, true),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&lane.device),
        ),
    ]);
    first.extend(lane.made(&[rgba8], (None, None), 1));
    first.extend(lane.made(&[], (Some(d32), None), 1));
    assert_made_once(&messages, &DONT_CARE_LOADS, &first);
    quiet(lane, &messages);
}
