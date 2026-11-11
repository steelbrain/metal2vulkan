//! Device tests of depth and stencil attachments a render pass loads `DontCare`
//! (`clear_reach::dont_care_loads` in the runtime), on the device the cfg selects, under the
//! validation layer: separate depth and stencil textures the pass stores, and a `DontCare` load's
//! clear depth that is a NaN or lies outside `[0, 1]`, of one texture and of separate ones.
//!
//! **Metal's answers** (an M3 Max under macOS 27.0.1: these passes through Metal itself over the
//! same MSL, `dcs_dont_care_split`'s `dcs_rect` and `dcs_flag`, probe jobs m3-max-16665 and
//! m3-max-16673):
//!
//! - A depth and a stencil loaded `DontCare` and stored hold the same in separate `Depth32Float`
//!   and `Stencil8` textures as in one `Depth32Float_Stencil8`, each aspect apart, with and without
//!   API validation. Under a draw that writes both ([`MARK_DEPTH`], the stencil replaced by 1) over
//!   a small rectangle, the rectangle holds the draw's values, the rest of the 32x32 tiles the
//!   rectangle touches holds the pass's clear value in each aspect loaded `DontCare`, and every
//!   other texel keeps its earlier value ([`expected_tiles`]). A pass whose draw call draws no
//!   vertex leaves every texel as it was.
//! - A `DontCare` load's clear depth outside `[0, 1]` is clamped: 2.0 stores and tests as 1.0,
//!   -1.0 as 0. A NaN stores `0x7fc00000` in a 32-bit float depth and 0 in a `Depth16Unorm`, in
//!   the tiles a writing draw touches. A load the pass does not store tests as that value, a NaN
//!   above every value ([`Held`]). That holds for `Depth32Float`, `Depth16Unorm`,
//!   `Depth32Float_Stencil8` and separate `Depth32Float` and `Stencil8` textures, without API
//!   validation, which stops each such pass ("clear depth (nan) is outside of the range [0..1]").
//!   A `Clear` load's clear depth outside `[0, 1]` is clamped the same way.
//!
//! **The runtime's answers.** Metal's in every case here but one. Separate textures render
//! through a scratch image, into which the runtime copies a stored `DontCare` aspect, clearing
//! the tiles the draws reach after the copy. A NaN clear depth of a `DontCare` load clears to 0
//! in a `Depth16Unorm`, Metal's answer. In a 32-bit float depth it clears to 1.0, the value
//! Metal's test orders a NaN nearest, because a Vulkan clear value holds no NaN: the depth stores
//! 1.0 and a fragment at exactly 1.0 passes `Equal`. That pass announces `signed-depth-test`.
//! A `Depth16Unorm` that such a load clears to a depth a device rounding to nearest takes above
//! Metal's truncation announces `unorm-depth-rounding`, as a `Clear` load's does.
//!
//! The passes here carry only the probe's first color, and the lane's draws carry a scissor
//! rectangle equal to their rectangle, which bounds the tiles the runtime clears; the probe's
//! draws had none, and their primitives cover the same pixels, so the same tiles.

use super::split_fetch_order::{completed, texture};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandQueue, Degradation, DepthStencilState, Device, Library,
    LoadAction, Origin, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction, Texture, Viewport,
};
use std::collections::BTreeMap;

/// The depths: the poison pass's clear value, the probe pass's, and what a writing draw writes
/// (each draw's depth its viewport's `znear` and `zfar`).
const POISON_DEPTH: f64 = 0.625;
const CLEAR_DEPTH: f64 = 0.375;
const MARK_DEPTH: f64 = 0.125;

/// The stencils: the poison pass's clear value, the probe pass's, and what a writing draw's
/// `Replace` writes.
const POISON_STENCIL: u32 = 0xAB;
const CLEAR_STENCIL: u32 = 0x5C;
const MARK_STENCIL: u32 = 1;

/// The color the poison pass clears to, which the probe pass loads and keeps.
const POISON: [f64; 4] = [0.2, 0.133, 0.067, 0.267];

/// `dcs_flag`'s color as a `BGRA8Unorm` texel reads back (bytes in memory order).
const FLAG: [u8; 4] = [255, 0, 255, 255];

/// A depth/stencil arrangement of the probe's: the depth's format, the stencil's where the pass
/// has one, and whether the stencil is a texture of its own.
#[derive(Clone, Copy, Debug)]
struct Arrangement {
    name: &'static str,
    depth: PixelFormat,
    stencil: Option<PixelFormat>,
    separate: bool,
}

const SEPARATE: Arrangement = Arrangement {
    name: "separate",
    depth: PixelFormat::Depth32Float,
    stencil: Some(PixelFormat::Stencil8),
    separate: true,
};
const COMBINED: Arrangement = Arrangement {
    name: "d32s8",
    depth: PixelFormat::Depth32FloatStencil8,
    stencil: Some(PixelFormat::Depth32FloatStencil8),
    separate: false,
};
const D32: Arrangement = Arrangement {
    name: "d32",
    depth: PixelFormat::Depth32Float,
    stencil: None,
    separate: false,
};
const D16: Arrangement = Arrangement {
    name: "d16",
    depth: PixelFormat::Depth16Unorm,
    stencil: None,
    separate: false,
};

impl Arrangement {
    /// Whether the depth is a 16-bit unorm one.
    fn unorm16(self) -> bool {
        self.depth == PixelFormat::Depth16Unorm
    }

    /// The depth's bytes a texel.
    fn depth_bytes(self) -> u64 {
        if self.unorm16() {
            2
        } else {
            4
        }
    }

    /// The depth texture's stored word of `depth` as Metal's clear converts it: a 16-bit code
    /// truncated, a 32-bit float's bits.
    fn word(self, depth: f64) -> u32 {
        match self.unorm16() {
            true => (depth * 65535.0) as u32,
            false => (depth as f32).to_bits(),
        }
    }

    /// The word a draw at `depth` stores: a 16-bit code rounded to nearest.
    fn drawn_word(self, depth: f64) -> u32 {
        match self.unorm16() {
            true => (depth * 65535.0).round() as u32,
            false => (depth as f32).to_bits(),
        }
    }

    /// The blit options that read the depth and the stencil.
    fn options(self) -> (BlitOption, BlitOption) {
        match self.stencil.is_some() && !self.separate {
            true => (
                BlitOption::DepthFromDepthStencil,
                BlitOption::StencilFromDepthStencil,
            ),
            false => (BlitOption::None, BlitOption::None),
        }
    }
}

/// Which of the two aspects a tiles case loads `DontCare` and stores; it loads and stores the
/// other.
#[derive(Clone, Copy, Debug, PartialEq)]
enum DontCare {
    Both,
    Depth,
    Stencil,
}

/// The tiles cases' rectangles (left, top, right, bottom, in pixels) of a 128x128 target: one
/// inside a tile, one over four tiles' corners, and one pixel in the far tiles.
const RECTS: [(&str, [u32; 4]); 3] = [
    ("R40-8", [40, 8, 44, 12]),
    ("R28-28", [28, 28, 36, 36]),
    ("R100-90", [100, 90, 101, 91]),
];

/// The lane's device, its queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(exact: bool, messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
            at_plan(options);
            if exact {
                options.policy = DevicePolicy::default().exact_answers();
            }
        })
        .unwrap();
        let air = llvm_assemble(&engine_dir().join("dcs_dont_care_split.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&air).unwrap(),
            device,
        }
    }

    /// `dcs_rect` and `dcs_flag` into a `BGRA8Unorm` color and `arrangement`'s depth and stencil.
    fn pipeline(&self, arrangement: Arrangement) -> RenderPipelineState {
        let function = |name: &str| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("dcs_rect"));
        descriptor.fragment = Some(function("dcs_flag"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        descriptor.depth_format = Some(arrangement.depth);
        descriptor.stencil_format = arrangement.stencil;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A depth/stencil state testing the depth with `compare` and writing it where `write`, its
    /// stencil test `Always` and replacing where `write`.
    fn state(&self, compare: CompareFunction, write: bool) -> DepthStencilState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = compare;
        descriptor.depth_write_enabled = write;
        if write {
            for face in [
                &mut descriptor.front_face_stencil,
                &mut descriptor.back_face_stencil,
            ] {
                face.stencil_compare_function = CompareFunction::Always;
                face.depth_stencil_pass_operation = StencilOperation::Replace;
            }
        }
        self.device.new_depth_stencil_state(&descriptor).unwrap()
    }

    /// `arrangement`'s textures of `side`: the color, the depth, and the stencil (the depth's
    /// texture where it is not separate).
    fn textures(&self, arrangement: Arrangement, side: u64) -> Textures {
        let private = ResourceOptions::private();
        let target = (TextureUsage::RENDER_TARGET, private);
        let color = texture(
            &self.device,
            PixelFormat::BGRA8Unorm,
            (side, side),
            (
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                private,
            ),
        );
        let depth = texture(&self.device, arrangement.depth, (side, side), target);
        let stencil = match (arrangement.stencil, arrangement.separate) {
            (Some(pf), true) => Some(texture(&self.device, pf, (side, side), target)),
            (Some(_), false) => Some(depth.clone()),
            (None, _) => None,
        };
        Textures {
            arrangement,
            side,
            color,
            depth,
            stencil,
        }
    }

    /// `texture`'s texels of `bytes` each (one aspect of a combined texture by `option`), through
    /// a blit into a shared buffer.
    fn read(&self, texture: &Texture, bytes: u64, option: BlitOption) -> Vec<u8> {
        let side = texture.width();
        let length = bytes * side * side;
        let buffer = buffer_with(&self.device, &vec![0; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
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
        completed(&command_buffer);
        read_buffer(&buffer, 0, length).unwrap()
    }
}

/// One arrangement's textures.
struct Textures {
    arrangement: Arrangement,
    side: u64,
    color: Texture,
    depth: Texture,
    stencil: Option<Texture>,
}

impl Textures {
    /// A pass over the textures: the color loaded by `color` and stored, the depth and the
    /// stencil each loaded and stored as `actions` gives, cleared to `clears`.
    fn pass(
        &self,
        color: LoadAction,
        actions: [(LoadAction, StoreAction); 2],
        (clear_depth, clear_stencil): (f64, u32),
    ) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor::new();
        let first = &mut descriptor.color_attachments[0];
        first.texture = Some(self.color.clone());
        first.load_action = color;
        first.store_action = StoreAction::Store;
        first.clear_color = ClearColor::new(POISON[0], POISON[1], POISON[2], POISON[3]);
        let depth = &mut descriptor.depth_attachment;
        depth.texture = Some(self.depth.clone());
        (depth.load_action, depth.store_action) = actions[0];
        depth.clear_depth = clear_depth;
        if let Some(stencil) = &self.stencil {
            let attachment = &mut descriptor.stencil_attachment;
            attachment.texture = Some(stencil.clone());
            (attachment.load_action, attachment.store_action) = actions[1];
            attachment.clear_stencil = clear_stencil;
        }
        descriptor
    }

    /// Encode the poison pass into a new command buffer of `lane`'s: every target cleared to its
    /// poison and stored.
    fn poison(&self, lane: &Lane) -> metal2vulkan_runtime::CommandBuffer {
        let cleared = (LoadAction::Clear, StoreAction::Store);
        let command_buffer = lane.queue.command_buffer().unwrap();
        let descriptor = self.pass(
            LoadAction::Clear,
            [cleared; 2],
            (POISON_DEPTH, POISON_STENCIL),
        );
        let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
    }

    /// The depth's words after a pass.
    fn depth_words(&self, lane: &Lane) -> Vec<u32> {
        let bytes = self.arrangement.depth_bytes();
        let read = lane.read(&self.depth, bytes, self.arrangement.options().0);
        read.chunks(bytes as usize)
            .map(|word| match word {
                [a, b] => u32::from(u16::from_le_bytes([*a, *b])),
                [a, b, c, d] => u32::from_le_bytes([*a, *b, *c, *d]),
                _ => unreachable!("a depth of 2 or 4 bytes"),
            })
            .collect()
    }

    /// The stencil's values after a pass.
    fn stencils(&self, lane: &Lane) -> Vec<u32> {
        let stencil = self.stencil.as_ref().expect("a stencil");
        let read = lane.read(stencil, 1, self.arrangement.options().1);
        read.into_iter().map(u32::from).collect()
    }
}

/// Draw `dcs_flag` through `pipeline` under `state` over `rect` of a target of `side` at `depth`,
/// the scissor rectangle the rectangle's; a draw of no vertex where `rect` is `None`.
fn draw(
    encoder: &mut RenderCommandEncoder,
    (pipeline, state): (&RenderPipelineState, &DepthStencilState),
    side: u64,
    depth: f64,
    rect: Option<[u32; 4]>,
) {
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_depth_stencil_state(state).unwrap();
    encoder.set_stencil_reference_value(MARK_STENCIL);
    let whole = side as f64;
    let viewport = Viewport::new(0.0, 0.0, whole, whole, depth, depth);
    encoder.set_viewport(viewport).unwrap();
    let [left, top, right, bottom] = rect.unwrap_or([0, 0, side as u32, side as u32]);
    let scissor = ScissorRect::new(
        left.into(),
        top.into(),
        (right - left).into(),
        (bottom - top).into(),
    );
    encoder.set_scissor_rect(scissor).unwrap();
    let corners = [left, top, right, bottom]
        .map(|v| (v as f32).to_le_bytes())
        .concat();
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let size = [side as f32; 2].map(f32::to_le_bytes).concat();
    encoder.set_vertex_bytes(&size, 1).unwrap();
    let vertices = if rect.is_some() { 6 } else { 0 };
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, vertices, 1, 0)
        .unwrap();
}

/// Metal's value at (`x`, `y`) of an aspect after a tiles case's pass whose draw wrote `mark`
/// over `rect` (none where it drew no vertex): `mark` in the rectangle; the clear value in the
/// rest of the 32x32 tiles the rectangle touches, where the aspect is loaded `DontCare`; else
/// the poison.
fn expected_tiles(
    rect: Option<[u32; 4]>,
    dont_care: bool,
    (poison, clear, mark): (u32, u32, u32),
    (x, y): (u32, u32),
) -> u32 {
    let Some([left, top, right, bottom]) = rect else {
        return poison;
    };
    let inside = (left..right).contains(&x) && (top..bottom).contains(&y);
    let tiled = (left / 32 * 32..right.div_ceil(32) * 32).contains(&x)
        && (top / 32 * 32..bottom.div_ceil(32) * 32).contains(&y);
    match (inside, tiled && dont_care) {
        (true, _) => mark,
        (false, true) => clear,
        (false, false) => poison,
    }
}

/// The texels of `found` (a `side`² level) that are not `expected`'s at their (`x`, `y`),
/// counted by (expected, found).
fn mismatches(
    found: &[u32],
    side: u64,
    expected: impl Fn((u32, u32)) -> u32,
) -> BTreeMap<(u32, u32), usize> {
    let mut wrong = BTreeMap::new();
    for (at, &value) in found.iter().enumerate() {
        let at = (at as u64 % side, at as u64 / side);
        let want = expected((at.0 as u32, at.1 as u32));
        if want != value {
            *wrong.entry((want, value)).or_default() += 1;
        }
    }
    wrong
}

/// Whether `messages` took a notice of `kind`.
fn announced(messages: &Messages, kind: Degradation) -> bool {
    messages.take_notices().iter().any(|m| {
        m.degradation
            .as_ref()
            .is_some_and(|d| d.degradation == kind)
    })
}

/// A depth and a stencil loaded `DontCare` and stored, of separate textures and of one combined
/// texture, each aspect so loaded alone or both, under a writing draw over each of [`RECTS`] and
/// under a draw call of no vertex: Metal's texels in every case ([`expected_tiles`]), and
/// `dont-care-load-tiles` announced.
#[test]
fn separate_depth_and_stencil_loaded_dont_care_and_stored_hold_metal_s_tiles() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let lane = Lane::new(false, &messages, &cache);
    let mut failures = Vec::new();
    for arrangement in [SEPARATE, COMBINED] {
        let pipeline = lane.pipeline(arrangement);
        let writes = lane.state(CompareFunction::Always, true);
        let textures = lane.textures(arrangement, 128);
        let rects = RECTS
            .iter()
            .map(|&(name, rect)| (name, Some(rect)))
            .chain([("no vertex", None)]);
        for (name, rect) in rects {
            for dont_care in [DontCare::Both, DontCare::Depth, DontCare::Stencil] {
                let command_buffer = textures.poison(&lane);
                let (loaded, kept) = (
                    (LoadAction::DontCare, StoreAction::Store),
                    (LoadAction::Load, StoreAction::Store),
                );
                let actions = match dont_care {
                    DontCare::Both => [loaded, loaded],
                    DontCare::Depth => [loaded, kept],
                    DontCare::Stencil => [kept, loaded],
                };
                let descriptor =
                    textures.pass(LoadAction::Load, actions, (CLEAR_DEPTH, CLEAR_STENCIL));
                let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                draw(&mut encoder, (&pipeline, &writes), 128, MARK_DEPTH, rect);
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                completed(&command_buffer);
                let depth = (
                    arrangement.word(POISON_DEPTH),
                    arrangement.word(CLEAR_DEPTH),
                    arrangement.drawn_word(MARK_DEPTH),
                );
                let depth_dont_care = matches!(dont_care, DontCare::Both | DontCare::Depth);
                let stencil_dont_care = matches!(dont_care, DontCare::Both | DontCare::Stencil);
                let stencil = (POISON_STENCIL, CLEAR_STENCIL, MARK_STENCIL);
                let wrong_depth = mismatches(&textures.depth_words(&lane), 128, |at| {
                    expected_tiles(rect, depth_dont_care, depth, at)
                });
                let wrong_stencil = mismatches(&textures.stencils(&lane), 128, |at| {
                    expected_tiles(rect, stencil_dont_care, stencil, at)
                });
                let case = format!("{} {name} {dont_care:?} DontCare", arrangement.name);
                eprintln!(
                    "DONT-CARE-STORED on {}: {case}: texels not Metal's (expected, found): \
                     depth {wrong_depth:x?}, stencil {wrong_stencil:x?}",
                    lane.device.name()
                );
                if !wrong_depth.is_empty() || !wrong_stencil.is_empty() {
                    failures.push(format!(
                        "{case}: depth {wrong_depth:x?}, stencil {wrong_stencil:x?}"
                    ));
                }
            }
        }
    }
    assert!(
        announced(&messages, Degradation::DontCareLoadTiles),
        "a writing draw over a `DontCare` load announced no `dont-care-load-tiles`"
    );
    assert!(
        failures.is_empty(),
        "texels not Metal's:\n{}",
        failures.join("\n")
    );
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// The depth a `DontCare` load's clear leaves, as a depth test reads it: a value, or a NaN,
/// which Metal's test orders above every value.
#[derive(Clone, Copy, Debug)]
enum Held {
    Value(f64),
    Nan,
}

impl Held {
    /// Whether a fragment at `z` passes `compare` against it.
    fn passes(self, compare: CompareFunction, z: f64) -> bool {
        let Held::Value(held) = self else {
            return matches!(compare, CompareFunction::Less | CompareFunction::LessEqual);
        };
        match compare {
            CompareFunction::Less => z < held,
            CompareFunction::LessEqual => z <= held,
            CompareFunction::Greater => z > held,
            CompareFunction::GreaterEqual => z >= held,
            CompareFunction::Equal => z == held,
            _ => unreachable!("the tests' compare functions"),
        }
    }
}

/// The unstored cases' depth tests: a draw over the whole target at a depth, no depth writes.
const TESTS: [(&str, CompareFunction, f64); 6] = [
    ("less@0.5", CompareFunction::Less, 0.5),
    ("greater@0.5", CompareFunction::Greater, 0.5),
    ("equal@1", CompareFunction::Equal, 1.0),
    ("equal@0", CompareFunction::Equal, 0.0),
    ("lessequal@1", CompareFunction::LessEqual, 1.0),
    ("greaterequal@0", CompareFunction::GreaterEqual, 0.0),
];

/// A `DontCare` load's clear depth of a NaN of either sign, 2.0 and -1.0, under
/// `exact_answers()` (whose policy clears an unstored `DontCare` load), over each of
/// `Depth32Float`, `Depth16Unorm`, `Depth32Float_Stencil8` and separate `Depth32Float` and
/// `Stencil8` textures, of 64x64. Stored, under a writing draw over a 4x4 rectangle at (40, 8):
/// the rectangle holds the draw's depth, the rest of its tile Metal's clamped depth (a NaN's
/// `0x7fc00000` in a 32-bit float depth the runtime's 1.0), every other texel the poison. Not
/// stored, under each of [`TESTS`]: Metal's outcome at every pixel, but a 32-bit float depth's
/// NaN, which tests as the runtime's 1.0. A `Clear` load of 2.0 and -1.0, stored, holds the
/// clamped depth at every texel. The validation layer reports nothing (no clear value outside
/// `[0, 1]`, and no NaN), and the NaN in a 32-bit float depth announces `signed-depth-test`.
#[test]
fn a_dont_care_load_s_clear_depth_outside_0_1_or_nan_is_clamped_or_announced() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let lane = Lane::new(true, &messages, &cache);
    let side = 64;
    let values = [
        ("nan", f64::NAN),
        ("negative nan", -f64::from_bits(0x7ff8_0000_0000_1234)),
        ("2.0", 2.0),
        ("-1.0", -1.0),
    ];
    let mut failures = Vec::new();
    for arrangement in [D32, D16, COMBINED, SEPARATE] {
        let pipeline = lane.pipeline(arrangement);
        let writes = lane.state(CompareFunction::Always, true);
        let textures = lane.textures(arrangement, side);
        for (name, value) in values {
            // Metal's held depth, and the runtime's word for it in a stored texel.
            let held = match value.is_nan() {
                true if arrangement.unorm16() => Held::Value(0.0),
                true => Held::Nan,
                false => Held::Value(value.clamp(0.0, 1.0)),
            };
            let runtime = match held {
                Held::Value(depth) => depth,
                Held::Nan => 1.0,
            };
            let case = format!("{} {name}", arrangement.name);
            // Stored, under a writing draw over a 4x4 rectangle.
            let rect = [40, 8, 44, 12];
            let stencil = (LoadAction::Load, StoreAction::Store);
            let command_buffer = textures.poison(&lane);
            let descriptor = textures.pass(
                LoadAction::Load,
                [(LoadAction::DontCare, StoreAction::Store), stencil],
                (value, CLEAR_STENCIL),
            );
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            draw(
                &mut encoder,
                (&pipeline, &writes),
                side,
                MARK_DEPTH,
                Some(rect),
            );
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            completed(&command_buffer);
            let words = (
                arrangement.word(POISON_DEPTH),
                arrangement.word(runtime),
                arrangement.drawn_word(MARK_DEPTH),
            );
            let wrong = mismatches(&textures.depth_words(&lane), side, |at| {
                expected_tiles(Some(rect), true, words, at)
            });
            eprintln!(
                "DONT-CARE-CLEAR-DEPTH on {}: {case} stored: texels not the runtime's answer \
                 (expected, found) {wrong:x?}",
                lane.device.name()
            );
            if !wrong.is_empty() {
                failures.push(format!("{case} stored: {wrong:x?}"));
            }
            // A `Clear` load of a depth outside `[0, 1]`, stored: clamped at every texel.
            if !value.is_nan() {
                let command_buffer = textures.poison(&lane);
                let descriptor = textures.pass(
                    LoadAction::Load,
                    [(LoadAction::Clear, StoreAction::Store), stencil],
                    (value, CLEAR_STENCIL),
                );
                let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                completed(&command_buffer);
                let word = arrangement.word(runtime);
                let wrong = mismatches(&textures.depth_words(&lane), side, |_| word);
                if !wrong.is_empty() {
                    failures.push(format!("{case} `Clear` load: {wrong:x?}"));
                }
            }
            // Not stored, under each depth test.
            let mut outcomes = Vec::new();
            for (test, compare, z) in TESTS {
                let state = lane.state(compare, false);
                let command_buffer = textures.poison(&lane);
                let descriptor = textures.pass(
                    LoadAction::Load,
                    [(LoadAction::DontCare, StoreAction::DontCare), stencil],
                    (value, CLEAR_STENCIL),
                );
                let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                draw(
                    &mut encoder,
                    (&pipeline, &state),
                    side,
                    z,
                    Some([0, 0, 64, 64]),
                );
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                completed(&command_buffer);
                let texels = lane.read(&textures.color, 4, BlitOption::None);
                let flagged = texels.chunks(4).filter(|texel| *texel == FLAG).count();
                let metals = held.passes(compare, z);
                let runtimes = Held::Value(runtime).passes(compare, z);
                outcomes.push(format!("{test} {flagged}"));
                let want = if runtimes { (side * side) as usize } else { 0 };
                if flagged != want {
                    failures.push(format!(
                        "{case} unstored {test}: {flagged} of {} pixels passed, the runtime's \
                         answer {runtimes} (Metal's {metals})",
                        side * side
                    ));
                }
            }
            eprintln!(
                "DONT-CARE-CLEAR-DEPTH on {}: {case} unstored, pixels passing: {}",
                lane.device.name(),
                outcomes.join(", ")
            );
        }
    }
    assert!(
        announced(&messages, Degradation::SignedDepthTest),
        "a NaN clear depth of a `DontCare` load in a 32-bit float depth announced no \
         `signed-depth-test`"
    );
    assert!(
        failures.is_empty(),
        "not the runtime's answer:\n{}",
        failures.join("\n")
    );
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// A `Depth16Unorm` loaded `DontCare` and not stored, under `exact_answers()` (whose policy
/// clears it), in a pass whose draw over the whole target neither tests nor writes the depth: a
/// clear depth of 0, which every device converts to Metal's code, announces nothing; one of 0.5,
/// whose scaled value lies on a code's midpoint, which Metal truncates and a device rounding to
/// nearest takes to the code above, announces `unorm-depth-rounding`.
#[test]
fn a_dont_care_load_s_depth16_clear_announces_the_device_s_rounding() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let lane = Lane::new(true, &messages, &cache);
    let pipeline = lane.pipeline(D16);
    let untested = lane.state(CompareFunction::Always, false);
    let textures = lane.textures(D16, 64);
    for (clear, rounds) in [(0.0, false), (0.5, true)] {
        let command_buffer = textures.poison(&lane);
        let descriptor = textures.pass(
            LoadAction::Load,
            [(LoadAction::DontCare, StoreAction::DontCare); 2],
            (clear, CLEAR_STENCIL),
        );
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        draw(
            &mut encoder,
            (&pipeline, &untested),
            64,
            0.5,
            Some([0, 0, 64, 64]),
        );
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        completed(&command_buffer);
        assert_eq!(
            announced(&messages, Degradation::UnormDepthRounding),
            rounds,
            "a `DontCare` load's clear to {clear}: `unorm-depth-rounding` announced or not"
        );
    }
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}
