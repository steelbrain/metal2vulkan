//! Device tests of attachments a render pass loads `DontCare` and does not store, read by the
//! pass's own draws (`clear_reach::dont_care_loads` in the runtime), on the device the cfg
//! selects, under the validation layer.
//!
//! **Metal's answer** (an M3 Max, macOS 27.0.1, these passes run through Metal itself over the
//! same MSL, under the default policy's and `exact_answers()`'s descriptors alike, at both sizes):
//! such an attachment reads zero (a color) or the pass's clear value (a stencil) wherever a draw
//! of the pass reads it, never its earlier texels and never zero for the stencil. [`expected`]
//! is that answer, pixel by pixel.
//!
//! Each run poisons the targets first, with a pass of its own that clears and stores them all (a
//! first color of [`POISON`], a second `RG16Float` color of (0.75, 0.875), a `Stencil8` of 0xAB),
//! then runs an [`Arm`]: a pass that loads and stores the first color, and loads the second color
//! and the stencil `DontCare` and stores neither, its stencil clear value 0x5C. `dcs_mark` draws
//! over [`MARK`] (the stencil replaced by 1, the second color (0.25, 0.5)), which reaches only
//! the target's top left 32x32 tiles; `dcs_read` (a framebuffer fetch of the second color) and
//! `dcs_flag` draw over the whole target. Each pixel of the first color then says what the draw
//! over it read ([`class`]). A fetch after `dcs_mark`'s write splits the pass into render pass
//! instances on a device that fetches through the pixel interlock, the first of which leaves the
//! tiles `dcs_mark` does not reach alone; a fetch in the pass's only draw leaves the second color
//! out of the instance. Printed per arm, policy and size (`DONT-CARE-UNSTORED`): the instances
//! and splits, the evidence of which form ran.
//!
//! Every arm is run twice over the same textures, so a second run reads what an earlier pass
//! left in the device's memory, which a first run over new textures may not show. No arm names
//! `dont-care-load-tiles`: the attachments it covers are stored.

use super::split_fetch_order::{completed, encode_readback, texture};
use super::{at_plan, read_buffer, validated_device_with, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBuffer, Degradation, DepthStencilState, Device, LoadAction, PipelineOptions,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
    Texture,
};
use std::collections::BTreeMap;

/// `dcs_mark`'s rectangle (left, top, right, bottom, in pixels): of the target's 32x32 tiles,
/// those of columns 0 to 2 and rows 0 and 1.
const MARK: [f32; 4] = [0.0, 0.0, 80.0, 56.0];

/// The first color's poison, as the clear sets it (red, green, blue, alpha).
const POISON: [f64; 4] = [0.2, 0.133, 0.067, 0.267];

/// The stencil's poison, and the arms' pass's clear value, which no `DontCare` load names.
const POISON_STENCIL: u32 = 0xAB;
const CLEAR_STENCIL: u32 = 0x5C;

/// The runs of each arm, policy and size.
const RUNS: usize = 2;

/// What an arm's pass draws.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    /// `dcs_mark` over [`MARK`], then `dcs_read` over all, its stencil test `Always`: the read
    /// shows the second color.
    SplitRead,
    /// As [`Arm::SplitRead`], `dcs_read`'s stencil test `Equal 0`: the read shows the stencil too.
    SplitStencil,
    /// `dcs_read` alone over all, `Always`.
    FetchRead,
    /// `dcs_read` alone over all, `Equal 0`.
    FetchStencil,
    /// `dcs_mark` over [`MARK`], then `dcs_flag` over all, `Equal 0`: no fetch; the flag shows the
    /// stencil.
    PlainStencil,
    /// As [`Arm::PlainStencil`], `Equal` the poison's stencil.
    PlainKept,
    /// As [`Arm::PlainStencil`], `Equal` the pass's clear value.
    PlainCleared,
}

const ARMS: [Arm; 7] = [
    Arm::SplitRead,
    Arm::SplitStencil,
    Arm::FetchRead,
    Arm::FetchStencil,
    Arm::PlainStencil,
    Arm::PlainKept,
    Arm::PlainCleared,
];

/// Metal's class at pixel (`x`, `y`) after `arm` (the module docs): inside [`MARK`], `dcs_mark`'s
/// color, or what `dcs_read` reads of `dcs_mark`'s second color; outside it, a read of a zero
/// second color, or the stencil's clear value failing an `Equal 0` or `Equal` the poison test
/// and passing an `Equal` the clear value one.
fn expected(arm: Arm, (x, y): (u64, u64)) -> char {
    let marked = (x as f32) < MARK[2] && (y as f32) < MARK[3];
    match (arm, marked) {
        (Arm::SplitRead, true) => 'K',
        (Arm::SplitRead | Arm::FetchRead, _) => 'Z',
        (Arm::FetchStencil, _) => 'P',
        (Arm::PlainCleared, false) => 'F',
        (_, true) => 'M',
        (_, false) => 'P',
    }
}

/// The pipelines and stencil states.
struct Draws {
    mark: RenderPipelineState,
    read: RenderPipelineState,
    flag: RenderPipelineState,
    write_one: DepthStencilState,
    always: DepthStencilState,
    equal: DepthStencilState,
}

impl Draws {
    fn new(device: &Device, air: &[u8]) -> Draws {
        let library = device.new_library_with_air(air).unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |fragment: &str, second: ColorWriteMask| {
            let mut descriptor = RenderPipelineDescriptor::new(function("dcs_rect"));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RG16Float);
            color.write_mask = second;
            descriptor.color_attachments[1] = Some(color);
            descriptor.stencil_format = Some(PixelFormat::Stencil8);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let state = |compare: CompareFunction, pass: StencilOperation| {
            let mut descriptor = DepthStencilDescriptor::default();
            for face in [
                &mut descriptor.front_face_stencil,
                &mut descriptor.back_face_stencil,
            ] {
                face.stencil_compare_function = compare;
                face.depth_stencil_pass_operation = pass;
            }
            device.new_depth_stencil_state(&descriptor).unwrap()
        };
        Draws {
            mark: pipeline("dcs_mark", ColorWriteMask::ALL),
            read: pipeline("dcs_read", ColorWriteMask::NONE),
            flag: pipeline("dcs_flag", ColorWriteMask::NONE),
            write_one: state(CompareFunction::Always, StencilOperation::Replace),
            always: state(CompareFunction::Always, StencilOperation::Keep),
            equal: state(CompareFunction::Equal, StencilOperation::Keep),
        }
    }
}

/// Draw `pipeline` over `rect` of a target of `sides`, under `state` and stencil reference
/// `reference`.
fn draw(
    encoder: &mut RenderCommandEncoder,
    (pipeline, state, reference): (&RenderPipelineState, &DepthStencilState, u32),
    sides: (u64, u64),
    rect: [f32; 4],
) {
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_depth_stencil_state(state).unwrap();
    encoder.set_stencil_reference_value(reference);
    let corners = rect.map(f32::to_le_bytes).concat();
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let size = [sides.0 as f32, sides.1 as f32]
        .map(f32::to_le_bytes)
        .concat();
    encoder.set_vertex_bytes(&size, 1).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

/// A descriptor of the three targets, each loaded and stored as given.
fn targets(
    (color, second, stencil): (&Texture, &Texture, &Texture),
    loads: [LoadAction; 3],
    stores: [StoreAction; 3],
    clear_stencil: u32,
) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let first = &mut descriptor.color_attachments[0];
    first.texture = Some(color.clone());
    first.load_action = loads[0];
    first.store_action = stores[0];
    first.clear_color = ClearColor::new(POISON[0], POISON[1], POISON[2], POISON[3]);
    let next = &mut descriptor.color_attachments[1];
    next.texture = Some(second.clone());
    next.load_action = loads[1];
    next.store_action = stores[1];
    next.clear_color = ClearColor::new(0.75, 0.875, 0.0, 0.0);
    descriptor.stencil_attachment.texture = Some(stencil.clone());
    descriptor.stencil_attachment.load_action = loads[2];
    descriptor.stencil_attachment.store_action = stores[2];
    descriptor.stencil_attachment.clear_stencil = clear_stencil;
    descriptor
}

/// Encode a run of `arm` into `command_buffer`: the poison pass, the arm's pass, and the first
/// color's readback.
fn encode_run(
    command_buffer: &CommandBuffer,
    draws: &Draws,
    textures: (&Texture, &Texture, &Texture),
    (arm, sides): (Arm, (u64, u64)),
) -> metal2vulkan_runtime::Buffer {
    let poison = targets(
        textures,
        [LoadAction::Clear; 3],
        [StoreAction::Store; 3],
        POISON_STENCIL,
    );
    let encoder = command_buffer.render_command_encoder(&poison).unwrap();
    encoder.end_encoding().unwrap();
    let probe = targets(
        textures,
        [LoadAction::Load, LoadAction::DontCare, LoadAction::DontCare],
        [
            StoreAction::Store,
            StoreAction::DontCare,
            StoreAction::DontCare,
        ],
        CLEAR_STENCIL,
    );
    let mut encoder = command_buffer.render_command_encoder(&probe).unwrap();
    let all = [0.0, 0.0, sides.0 as f32, sides.1 as f32];
    let mark = (&draws.mark, &draws.write_one, 1);
    let (read, flag) = (&draws.read, &draws.flag);
    if !matches!(arm, Arm::FetchRead | Arm::FetchStencil) {
        draw(&mut encoder, mark, sides, MARK);
    }
    let last = match arm {
        Arm::SplitRead | Arm::FetchRead => (read, &draws.always, 0),
        Arm::SplitStencil | Arm::FetchStencil => (read, &draws.equal, 0),
        Arm::PlainStencil => (flag, &draws.equal, 0),
        Arm::PlainKept => (flag, &draws.equal, POISON_STENCIL),
        Arm::PlainCleared => (flag, &draws.equal, CLEAR_STENCIL),
    };
    draw(&mut encoder, last, sides, all);
    encoder.end_encoding().unwrap();
    encode_readback(command_buffer, textures.0, sides)
}

/// What a pixel's first color says (`BGRA8Unorm` bytes in memory order): `P` the poison (no draw
/// wrote it), `M` `dcs_mark`'s, `Z` a read of a zero second color, `S` a read of the poison's
/// second color, `K` a read of `dcs_mark`'s, `F` `dcs_flag`'s, `?` anything else.
fn class(bgra: &[u8]) -> char {
    match bgra {
        [17, 34, 51, 68] => 'P',
        [0, 128, 0, 255] => 'M',
        [255, 0, 0, 255] => 'Z',
        [255, 223, 191, 255] => 'S',
        [255, 128, 64, 255] => 'K',
        [255, 0, 255, 255] => 'F',
        _ => '?',
    }
}

/// Each pixel of `bytes` of a target of `sides` whose class is not [`expected`]'s, counted by
/// (expected, found), and a map of 16x16 cells, each the class of its first such pixel or `.`.
fn mismatches(
    arm: Arm,
    bytes: &[u8],
    (width, height): (u64, u64),
) -> (BTreeMap<(char, char), usize>, Vec<String>) {
    let mut counts = BTreeMap::new();
    let mut map = vec![vec!['.'; width.div_ceil(16) as usize]; height.div_ceil(16) as usize];
    for (at, bgra) in bytes.chunks(4).enumerate() {
        let (x, y) = (at as u64 % width, at as u64 / width);
        let (want, found) = (expected(arm, (x, y)), class(bgra));
        if want != found {
            *counts.entry((want, found)).or_default() += 1;
            let cell = &mut map[(y / 16) as usize][(x / 16) as usize];
            if *cell == '.' {
                *cell = found;
            }
        }
    }
    (counts, map.into_iter().map(String::from_iter).collect())
}

/// Each arm's first color after its pass, under the default policy and `exact_answers()`, at a
/// 224x128 and a 128x128 target, twice each: Metal's answer, pixel by pixel (the module docs),
/// and no `dont-care-load-tiles`.
#[test]
fn dont_care_loads_the_pass_does_not_store_read_metals_answer() {
    let air = llvm_assemble(&engine_dir().join("dcs_dont_care_split.ll")).unwrap();
    let mut failures = Vec::new();
    for (policy, exact) in [("default", false), ("exact", true)] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            at_plan(options);
            options.policy = match exact {
                true => DevicePolicy::default().exact_answers(),
                false => DevicePolicy::default(),
            };
        })
        .unwrap();
        let draws = Draws::new(&device, &air);
        let queue = device.new_command_queue().unwrap();
        for sides in [(224u64, 128u64), (128, 128)] {
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let private = ResourceOptions::private();
            let color = texture(&device, PixelFormat::BGRA8Unorm, sides, (usage, private));
            let target = (TextureUsage::RENDER_TARGET, private);
            let second = texture(&device, PixelFormat::RG16Float, sides, target);
            let stencil = texture(&device, PixelFormat::Stencil8, sides, target);
            for arm in ARMS {
                for run in 0..RUNS {
                    let counted = device.work_counts();
                    let command_buffer = queue.command_buffer().unwrap();
                    let textures = (&color, &second, &stencil);
                    let out = encode_run(&command_buffer, &draws, textures, (arm, sides));
                    let counts = device.work_counts();
                    command_buffer.commit().unwrap();
                    completed(&command_buffer);
                    let bytes = read_buffer(&out, 0, sides.0 * sides.1 * 4).unwrap();
                    let (wrong, map) = mismatches(arm, &bytes, sides);
                    let case = format!("{arm:?} {policy} {}x{} run {run}", sides.0, sides.1);
                    eprintln!(
                        "DONT-CARE-UNSTORED on {}: {case}: render pass instances {}, splits {}; \
                         pixels not Metal's (expected, found) {wrong:?}",
                        device.name(),
                        counts.render_passes_begun_instances
                            - counted.render_passes_begun_instances,
                        counts.instance_splits_shape - counted.instance_splits_shape,
                    );
                    if !wrong.is_empty() {
                        failures.push(format!("{case}: {wrong:?}, cells {map:?}"));
                    }
                }
            }
        }
        let tiles = messages.take_notices().iter().any(|m| {
            m.degradation
                .as_ref()
                .is_some_and(|d| d.degradation == Degradation::DontCareLoadTiles)
        });
        assert!(
            !tiles,
            "{policy}: `dont-care-load-tiles` named an attachment the pass does not store"
        );
    }
    assert!(
        failures.is_empty(),
        "pixels not Metal's answer:\n{}",
        failures.join("\n")
    );
}
