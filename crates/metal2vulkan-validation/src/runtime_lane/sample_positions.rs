//! Device tests of sample positions and the store action option that names them (the runtime's
//! `sample_positions.rs`) on the device the cfg selects, under the validation layer.
//!
//! Each pass is one that Metal probes measured (an M4 with macOS 27.0.1, each case with and without
//! API validation): a target of one pixel whose triangle covers what lies left of a vertical edge,
//! resolved. The probes drew white and read the resolved red; the lane draws through the owned
//! fixture `rt6_restart` (its vertex function places each vertex where buffer 0 holds it, its
//! fragment function writes alpha 1) over a target cleared to zero, so the resolved alpha is the
//! pixel's coverage. Each is compared as the number of covered samples, as Vulkan leaves a
//! resolve's rounding to the device. What Metal's validation stops, the runtime refuses typed, and
//! a pass Metal takes that the runtime has not built is refused `NotImplemented`.
//!
//! The passes with a depth or stencil attachment, and those of the store action
//! `CustomSampleDepthStore`, are the Metal probe's: drawn through the owned fixture
//! `cfb_sample_depth` over a 2x2 target and read back sample by sample, each sample's depth
//! compared with the depth Metal stored there.

use super::degrade_notices::{
    assert_once, counts, made_render_kinds, shared_in_system_memory, POSITIONED,
};
use super::presenting::{image_info, made_image, shared_among};
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_samples::encode;
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, VkExtension, VkProperty};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, StencilOperation, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::SampleCountFlags;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Attachment, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue,
    CpuCacheMode, Degradation, Device, Error, HazardTrackingMode, Invalid, Library, LoadAction,
    Origin, PipelineOptions, Problem, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPassProblem, RenderPipelineState, ResourceOptions, RuntimeGap, SamplePosition, Size,
    StorageMode, StoreAction, StoreActionOptions, Texture, TextureDescriptor, VkImageLayouts,
};
use std::collections::BTreeSet;

const CUSTOM: StoreActionOptions = StoreActionOptions::CUSTOM_SAMPLE_POSITIONS;

/// `count` positions at `(x, y)`.
fn at(x: f32, y: f32, count: usize) -> Vec<SamplePosition> {
    vec![SamplePosition::new(x, y); count]
}

/// The lane's device, its queue, and the fixture's pipelines by sample count.
struct Lane {
    device: Device,
    queue: CommandQueue,
    /// The pipelines by sample count, and whether they take a `Depth32Float` attachment.
    pipelines: Vec<(u64, bool, RenderPipelineState)>,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor()
        };
        // Four samples also with a depth attachment (`opt_depth_dontcare`).
        let pipelines = [(1, false), (2, false), (4, false), (4, true)]
            .into_iter()
            .filter(|&(samples, _)| {
                device
                    .properties()
                    .supports_texture_sample_count(samples)
                    .unwrap()
            })
            .map(|(samples, depth)| {
                let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                    function("rt6_restart_vertex"),
                ));
                descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
                descriptor.depth_format = depth.then_some(PixelFormat::Depth32Float);
                descriptor.raster_sample_count = samples as u32;
                let pipeline = device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .unwrap();
                (samples, depth, pipeline)
            })
            .collect();
        Lane {
            queue: device.new_command_queue().unwrap(),
            pipelines,
            device,
        }
    }

    /// Whether the device renders `samples` samples.
    fn renders(&self, samples: u64) -> bool {
        self.pipelines.iter().any(|(s, _, _)| *s == samples)
    }

    /// Whether the device takes custom positions at `count` samples.
    fn located(&self, count: u32) -> bool {
        let d = &self.device.properties().description;
        d.extensions.contains(&VkExtension::ExtSampleLocations)
            && d.limits
                .sample_location_sample_counts
                .contains(SampleCountFlags::from_raw(count))
    }

    /// The refusal a pass of `samples` samples with custom positions answers here, or `None`
    /// where the device programs them: the device's own lack where it takes no custom locations
    /// at the pass's sample count, which is the only count the grid is programmed at, whatever
    /// the number of positions the caller gave; then the runtime's gap where the device's
    /// locations may not differ between a pass's pipelines, so the pass would need them at its
    /// begin (the runtime's `sample_positions.rs`, `PositionRules`).
    fn unprogrammed(&self, samples: u32) -> Option<Error> {
        if !self.located(samples) {
            return Some(Error::Unsupported(Unsupported::needs(
                Feature::SamplePositions,
                Capability::Property(VkProperty::SampleLocationSampleCount(samples)),
            )));
        }
        let variable = self
            .device
            .properties()
            .description
            .limits
            .variable_sample_locations;
        (!variable).then_some(Error::NotImplemented(RuntimeGap::SamplePositionsAtBegin))
    }

    /// A `samples` render target of one pixel (multisampled above 1), private; or a shared
    /// single-sample one, a resolve texture.
    fn texture(&self, pf: PixelFormat, samples: u64, shared: bool) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(pf, 1, 1, false);
        if samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = samples;
        }
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = if shared {
            ResourceOptions::shared()
        } else {
            ResourceOptions::private()
        };
        self.device.new_texture(&descriptor).unwrap()
    }

    /// The probes' pass: a `samples` target of one pixel cleared to zero, stored by `store`
    /// with `options`, with `positions`; resolved into the returned texture where `resolves`
    /// (the store action, or the encoder's, resolves).
    fn pass(
        &self,
        samples: u64,
        positions: Vec<SamplePosition>,
        (store, options): (StoreAction, StoreActionOptions),
        resolves: bool,
    ) -> (RenderPassDescriptor, Option<Texture>) {
        let mut descriptor = RenderPassDescriptor::new();
        let resolve = resolves.then(|| self.texture(PixelFormat::RGBA8Unorm, 1, true));
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(self.texture(PixelFormat::RGBA8Unorm, samples, samples == 1));
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = store;
        target.store_action_options = options;
        target.resolve_texture = resolve.clone();
        descriptor.sample_positions = positions;
        (descriptor, resolve)
    }

    /// Draw the probes' triangle, covering what lies left of `edge` (normalized device
    /// coordinates), in `descriptor`'s pass of `samples` (through the pipeline that takes its
    /// depth attachment, where it has one), then `late` on the encoder; commit and wait. The
    /// encoder's refusal, if any.
    fn draw(
        &self,
        descriptor: &RenderPassDescriptor,
        samples: u64,
        edge: f32,
        late: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let depth = descriptor.depth_attachment.texture.is_some();
        let (_, _, pipeline) = self
            .pipelines
            .iter()
            .find(|(s, d, _)| (*s, *d) == (samples, depth))
            .unwrap();
        let triangle: Vec<u8> = [edge, -3.0, edge, 3.0, -3.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let triangle = buffer_with(&self.device, &triangle).unwrap();
        let command_buffer = self.queue.command_buffer()?;
        let mut encoder = command_buffer.render_command_encoder(descriptor)?;
        encoder.set_render_pipeline_state(pipeline)?;
        encoder.set_vertex_buffer(Some(&triangle), 0, 0)?;
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        late(&mut encoder)?;
        encoder.end_encoding()?;
        command_buffer.commit()?;
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(())
    }

    /// The samples of `resolve`'s pixel the pass of `samples` covered: its alpha, out of 255.
    fn covered(&self, resolve: &Texture, samples: u64) -> u64 {
        let mut texel = [0u8; 4];
        resolve
            .get_bytes(&mut texel, 4, 4, Region::new_2d(0, 0, 1, 1), 0, 0)
            .unwrap();
        (u64::from(texel[3]) * samples + 127) / 255
    }

    /// The samples the probes' pass of `samples`, with `positions` and the edge at `edge`,
    /// covers, resolved by `MultisampleResolve`.
    fn coverage(&self, samples: u64, positions: Vec<SamplePosition>, edge: f32) -> u64 {
        let store = (StoreAction::MultisampleResolve, StoreActionOptions::NONE);
        let (descriptor, resolve) = self.pass(samples, positions, store, true);
        self.draw(&descriptor, samples, edge, |_| Ok(())).unwrap();
        self.covered(&resolve.unwrap(), samples)
    }
}

fn invalid(call: &'static str, problem: RenderPassProblem) -> Error {
    Error::Invalid(Invalid::new(call, Problem::RenderPass(problem)))
}

fn lane() -> (Messages, ScratchCache, Lane) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    (messages, cache, Lane::new(device))
}

fn quiet(lane: Lane, messages: &Messages) {
    drop(lane);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Each sample lands where Metal put it: the device's own positions (4x: two of the four left
/// of the pixel's middle, 2x: one of two), custom ones left of, right of and across the edge,
/// and each coordinate rounded to the nearest sixteenth of a pixel, a tie up, at most 15/16
/// (Metal covered at 0.53 and 0.51 but not at 0.55, 0.53125, 0.5625 or 0.5624 against an edge
/// at 0.52, and at 0.97 and 0.99 against one at 0.95).
#[test]
fn the_samples_land_where_metal_puts_them() {
    let (messages, _cache, lane) = lane();
    // (name, samples, positions, edge, the samples Metal covered)
    let cases: [(&str, u64, Vec<SamplePosition>, f32, u64); 14] = [
        ("sp_default", 4, Vec::new(), 0.0, 2),
        ("sp_custom_left", 4, at(0.25, 0.5, 4), 0.0, 4),
        ("sp_custom_right", 4, at(0.75, 0.5, 4), 0.0, 0),
        (
            "sp_custom_mixed",
            4,
            [at(0.25, 0.5, 1), at(0.75, 0.5, 3)].concat(),
            0.0,
            1,
        ),
        ("pos_off_grid", 4, at(0.53, 0.5, 4), 0.04, 4),
        ("pos_off_grid_low", 4, at(0.51, 0.5, 4), 0.04, 4),
        ("pos_high", 4, at(0.97, 0.5, 4), 0.9, 4),
        ("round_0_55", 4, at(0.55, 0.5, 4), 0.04, 0),
        ("round_0_53125", 4, at(0.53125, 0.5, 4), 0.04, 0),
        ("round_0_5625", 4, at(0.5625, 0.5, 4), 0.04, 0),
        ("round_0_5624", 4, at(0.5624, 0.5, 4), 0.04, 0),
        ("round_0_99", 4, at(0.99, 0.5, 4), 0.9, 4),
        ("two_samples", 2, at(0.25, 0.5, 2), 0.0, 2),
        ("two_samples_default", 2, Vec::new(), 0.0, 1),
    ];
    let device = lane.device.name().to_owned();
    for (name, samples, positions, edge, metal) in cases {
        if !lane.renders(samples) {
            eprintln!("{name} on {device}: no {samples}x textures");
            continue;
        }
        let unprogrammed = (!positions.is_empty())
            .then(|| lane.unprogrammed(samples as u32))
            .flatten();
        if let Some(refusal) = unprogrammed {
            let store = (StoreAction::MultisampleResolve, StoreActionOptions::NONE);
            let (descriptor, _) = lane.pass(samples, positions, store, true);
            assert_eq!(
                lane.draw(&descriptor, samples, edge, |_| Ok(())),
                Err(refusal),
                "{name} on {device}"
            );
            continue;
        }
        let covered = lane.coverage(samples, positions, edge);
        assert_eq!(covered, metal, "{name} on {device}: the covered samples");
    }
    quiet(lane, &messages);
}

/// Fewer positions than the pass's samples, which Metal's validation stops, as plain Metal
/// runs them (an M4: 2 positions in a pass of 4 samples put samples 2 and 3
/// at the pixel's corner, (0, 0)): two positions at (0.75, 0.5) leave two samples covered left
/// of an edge at 0.1 and at 0.5 (Metal's own positions, none and two), and two at (0.25, 0.5)
/// all four left of 0.5. Refused where the device programs no such pass (`Lane::unprogrammed`):
/// the position count is never asked of the device, only the pass's sample count.
#[test]
fn fewer_positions_than_samples_put_the_rest_at_the_corner() {
    let (messages, _cache, lane) = lane();
    let device = lane.device.name().to_owned();
    if !lane.renders(4) {
        eprintln!("no 4x textures on {device}");
        return quiet(lane, &messages);
    }
    // (positions, the edge in normalized device coordinates, the samples Metal covers)
    let cases = [
        (at(0.75, 0.5, 2), -0.8, 2),
        (at(0.75, 0.5, 2), 0.0, 2),
        (at(0.25, 0.5, 2), 0.0, 4),
    ];
    for (positions, edge, metal) in cases {
        let refusal = lane.unprogrammed(4);
        if let Some(refusal) = refusal {
            let store = (StoreAction::MultisampleResolve, StoreActionOptions::NONE);
            let (descriptor, _) = lane.pass(4, positions, store, true);
            assert_eq!(
                lane.draw(&descriptor, 4, edge, |_| Ok(())),
                Err(refusal),
                "on {device}"
            );
            continue;
        }
        let covered = lane.coverage(4, positions, edge);
        assert_eq!(
            covered, metal,
            "edge {edge} on {device}: the covered samples"
        );
    }
    quiet(lane, &messages);
}

/// What Metal's validation stops, refused typed: positions outside `[0, 1)`, of one sample, of a
/// count the device takes none at or more than the pass's samples. The option
/// `CustomSamplePositions` of a color or depth attachment, taken in every form, as plain Metal
/// takes it: under `Store` and `StoreAndMultisampleResolve`, and where Metal's validation stops it,
/// under `DontCare` or `MultisampleResolve` (the descriptor's, and a late one), on the descriptor
/// without positions, and set late over a store action the descriptor knew; late, and late without
/// positions.
#[test]
fn sample_positions_and_their_option_are_refused_as_metal_refuses_them() {
    let (messages, _cache, lane) = lane();
    let encoder_call = "render_command_encoder";
    let none = StoreActionOptions::NONE;
    let positions = || at(0.25, 0.5, 4);
    let msr = StoreAction::MultisampleResolve;

    // The descriptor's positions.
    for (name, samples, positions, problem) in [
        (
            "pos_one",
            4,
            at(1.0, 0.5, 4),
            RenderPassProblem::SamplePosition { index: 0 },
        ),
        (
            "pos_negative",
            4,
            at(-0.1, 0.5, 4),
            RenderPassProblem::SamplePosition { index: 0 },
        ),
        (
            "pos_count_one_on_4x",
            4,
            at(0.25, 0.5, 1),
            RenderPassProblem::SamplePositionCount { count: 1 },
        ),
        (
            "pos_1x_custom",
            1,
            at(0.25, 0.5, 1),
            RenderPassProblem::SamplePositionCount { count: 1 },
        ),
        (
            "pos_count_8_on_4x",
            4,
            at(0.25, 0.5, 8),
            if lane.located(8) {
                RenderPassProblem::SamplePositionsSamples {
                    positions: 8,
                    samples: 4,
                }
            } else {
                RenderPassProblem::SamplePositionCount { count: 8 }
            },
        ),
    ] {
        let store = if samples == 1 {
            StoreAction::Store
        } else {
            msr
        };
        let (descriptor, _) = lane.pass(samples, positions, (store, none), store.resolves());
        assert_eq!(
            lane.draw(&descriptor, samples, 0.0, |_| Ok(())),
            Err(invalid(encoder_call, problem)),
            "{name}"
        );
    }

    // The descriptor's option (`opt_dontcare`, `sp_custom_storeopt` and `opt_store_no_positions`
    // stop Metal's validation, and plain Metal runs them: changed).
    for (name, store, with_positions) in [
        ("opt_dontcare", StoreAction::DontCare, true),
        ("sp_custom_storeopt", msr, true),
        ("opt_store_no_positions", StoreAction::Store, false),
        ("opt_store", StoreAction::Store, true),
        (
            "opt_store_resolve",
            StoreAction::StoreAndMultisampleResolve,
            true,
        ),
    ] {
        let positions = if with_positions {
            positions()
        } else {
            Vec::new()
        };
        let (descriptor, resolve) = lane.pass(4, positions, (store, CUSTOM), store.resolves());
        lane.draw(&descriptor, 4, 0.0, |_| Ok(()))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if store.resolves() {
            assert_eq!(lane.covered(&resolve.unwrap(), 4), 4, "{name}");
        }
    }

    // The encoder's option.
    let late = |options: StoreActionOptions, store: StoreAction| {
        move |encoder: &mut RenderCommandEncoder| {
            encoder.set_color_store_action_options(options, 0)?;
            encoder.set_color_store_action(store, 0)
        }
    };
    for (name, store, with_positions) in [
        ("opt_late_dontcare", StoreAction::DontCare, true),
        ("sp_late_options", msr, true),
        ("opt_late_store", StoreAction::Store, true),
        (
            "opt_late_store_resolve",
            StoreAction::StoreAndMultisampleResolve,
            true,
        ),
        ("opt_late_no_positions", StoreAction::Store, false),
    ] {
        let positions = if with_positions {
            positions()
        } else {
            Vec::new()
        };
        let unknown = (StoreAction::Unknown, none);
        let (descriptor, resolve) = lane.pass(4, positions, unknown, store.resolves());
        lane.draw(&descriptor, 4, 0.0, late(CUSTOM, store))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if store.resolves() {
            assert_eq!(lane.covered(&resolve.unwrap(), 4), 4, "{name}");
        }
    }
    // Options over a store action the descriptor knew: Metal's validation stops them, and plain
    // Metal keeps the descriptor's and completes (probe 15d), as the encoder does.
    for (name, options, late_options) in [
        ("opt_late_over_known", none, CUSTOM),
        ("opt_late_none_over_known", CUSTOM, none),
    ] {
        let (descriptor, _) = lane.pass(4, positions(), (StoreAction::Store, options), false);
        let drawn = lane.draw(&descriptor, 4, 0.0, |encoder| {
            encoder.set_color_store_action_options(late_options, 0)
        });
        assert_eq!(drawn, Ok(()), "{name}");
    }
    let (descriptor, _) = lane.pass(4, positions(), (StoreAction::Store, none), false);
    assert_eq!(
        lane.draw(&descriptor, 4, 0.0, |encoder| {
            encoder.set_color_store_action_options(CUSTOM, 1)
        }),
        Err(invalid(
            "set_color_store_action_options",
            RenderPassProblem::NoAttachment(Attachment::Color(1))
        ))
    );

    // A depth attachment: its option under DontCare is taken as the color's is, as plain Metal
    // takes it (`opt_depth_dontcare`, coverage 255; the passes that take positions with depth
    // are `depth_takes_the_custom_positions`'s).
    let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
    let (mut descriptor, resolve) = lane.pass(4, positions(), (msr, none), true);
    let attachment = &mut descriptor.depth_attachment;
    attachment.texture = Some(depth.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::DontCare;
    attachment.store_action_options = CUSTOM;
    lane.draw(&descriptor, 4, 0.0, |_| Ok(()))
        .unwrap_or_else(|e| panic!("opt_depth_dontcare: {e}"));
    assert_eq!(lane.covered(&resolve.unwrap(), 4), 4, "opt_depth_dontcare");
    quiet(lane, &messages);
}

// Custom sample positions with a depth or stencil attachment, and `CustomSampleDepthStore`: the
// Metal probe's passes (the M3 Max with macOS 26.7 and an M4 with macOS 27.0.1, each case with and
// without API validation, the same), drawn with the owned fixture `cfb_sample_depth` (each vertex
// where buffer 0 holds it, depth included) over a 2x2 target and read back sample by sample
// (`candidate::runtime_samples::encode`).

/// The extent of the depth passes' targets.
const EXTENT: [u32; 2] = [2, 2];

/// The probe's four positions, another four, and two.
const P4: [(f32, f32); 4] = [(0.1, 0.2), (0.3, 0.8), (0.6, 0.45), (0.9, 0.05)];
const Q4: [(f32, f32); 4] = [(0.7, 0.7), (0.2, 0.3), (0.45, 0.9), (0.05, 0.55)];
const P2: [(f32, f32); 2] = [(0.2, 0.7), (0.8, 0.3)];

/// The device's own positions, which Metal's are (the probe's `dz_default*`: Vulkan's standard
/// locations).
const STANDARD4: [(f32, f32); 4] = [
    (0.375, 0.125),
    (0.875, 0.375),
    (0.125, 0.625),
    (0.625, 0.875),
];
const STANDARD2: [(f32, f32); 2] = [(0.75, 0.75), (0.25, 0.25)];

fn positions_of(points: &[(f32, f32)]) -> Vec<SamplePosition> {
    points
        .iter()
        .map(|&(x, y)| SamplePosition::new(x, y))
        .collect()
}

/// A coordinate as Metal samples at it (`sample_positions.rs`): the nearest sixteenth, a tie up,
/// at most 15/16.
fn rounded(v: f32) -> f32 {
    (v * 16.0 + 0.5).floor().min(15.0) / 16.0
}

/// The probe's planes of depth, each a triangle over the whole 2x2 target: `X` holds
/// 0.375 + px/8 and `Y` 0.375 + py/8, `px` and `py` in pixels from the top left, so a sample's
/// depth tells where it lies; `Edge` covers what lies left of the first column's middle, at 0.5,
/// and `Flat` all of it at 0.5.
#[derive(Clone, Copy, Debug)]
enum Plane {
    X,
    Y,
    Edge,
    Flat,
}

impl Plane {
    fn vertices(self) -> [[f32; 4]; 3] {
        match self {
            Plane::X => [
                [-1.0, -1.0, 0.375, 1.0],
                [3.0, -1.0, 0.875, 1.0],
                [-1.0, 3.0, 0.375, 1.0],
            ],
            Plane::Y => [
                [-1.0, -1.0, 0.625, 1.0],
                [3.0, -1.0, 0.625, 1.0],
                [-1.0, 3.0, 0.125, 1.0],
            ],
            Plane::Edge => [
                [-0.5, -3.0, 0.5, 1.0],
                [-0.5, 3.0, 0.5, 1.0],
                [-3.0, 0.0, 0.5, 1.0],
            ],
            Plane::Flat => [
                [-1.0, -1.0, 0.5, 1.0],
                [3.0, -1.0, 0.5, 1.0],
                [-1.0, 3.0, 0.5, 1.0],
            ],
        }
    }

    /// The depth of `points`' samples in each pixel of the target, pixel by pixel, row by row,
    /// as Metal stored them (`X` and `Y` only).
    fn depths(self, points: &[(f32, f32)]) -> Vec<f32> {
        let mut out = Vec::new();
        for py in 0..EXTENT[1] {
            for px in 0..EXTENT[0] {
                for &(x, y) in points {
                    let along = match self {
                        Plane::X => px as f32 + rounded(x),
                        Plane::Y => py as f32 + rounded(y),
                        Plane::Edge | Plane::Flat => unreachable!("its depth is flat"),
                    };
                    out.push(0.375 + along / 8.0);
                }
            }
        }
        out
    }
}

/// What each sample of the edge's pass covers at `points`: of the first column, those left of
/// its middle.
fn edge_covers(points: &[(f32, f32)]) -> Vec<bool> {
    let mut out = Vec::new();
    for _py in 0..EXTENT[1] {
        for px in 0..EXTENT[0] {
            for &(x, _) in points {
                out.push(px == 0 && rounded(x) < 0.5);
            }
        }
    }
    out
}

/// One pass of the depth lane: its attachments and actions, and what it draws.
struct DepthPass {
    samples: u64,
    positions: Vec<SamplePosition>,
    color: Option<Texture>,
    color_store: StoreAction,
    depth: Option<Texture>,
    depth_load: LoadAction,
    clear_depth: f64,
    depth_store: StoreAction,
    depth_options: StoreActionOptions,
    depth_resolve: Option<Texture>,
    stencil: Option<Texture>,
    stencil_load: LoadAction,
    stencil_store: StoreAction,
    stencil_compare: CompareFunction,
    reference: u32,
    plane: Plane,
    compare: CompareFunction,
    draw: bool,
}

impl DepthPass {
    /// The probe's defaults: four samples at the device's positions, the depth cleared to 1 and
    /// stored, the stencil cleared to 0x11 and stored, plane `X` drawn under `Always`, the
    /// stencil replaced by 0x5a under `Always`.
    fn new() -> DepthPass {
        DepthPass {
            samples: 4,
            positions: Vec::new(),
            color: None,
            color_store: StoreAction::Store,
            depth: None,
            depth_load: LoadAction::Clear,
            clear_depth: 1.0,
            depth_store: StoreAction::Store,
            depth_options: StoreActionOptions::NONE,
            depth_resolve: None,
            stencil: None,
            stencil_load: LoadAction::Clear,
            stencil_store: StoreAction::Store,
            stencil_compare: CompareFunction::Always,
            reference: 0x5a,
            plane: Plane::X,
            compare: CompareFunction::Always,
            draw: true,
        }
    }
}

/// The depth lane: the device, its queue and the fixture's library.
struct Depths {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Depths {
    /// The descriptor of the pipeline a draw of `pass` takes.
    fn pipeline(&self, pass: &DepthPass) -> RenderPipelineDescriptor {
        let mut pipeline = RenderPipelineDescriptor::new(self.function("cfb_depth_vertex"));
        let color = pass.color.as_ref().map(Texture::pixel_format);
        pipeline.fragment = Some(self.function(match color {
            Some(_) => "cfb_depth_white",
            None => "cfb_depth_none",
        }));
        if let Some(pf) = color {
            pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
        }
        pipeline.depth_format = pass.depth.as_ref().map(Texture::pixel_format);
        pipeline.stencil_format = pass.stencil.as_ref().map(Texture::pixel_format);
        pipeline.raster_sample_count = pass.samples as u32;
        pipeline
    }

    /// The kinds the making of the pipeline a draw of `pass` takes announces
    /// (`made_render_kinds`), and its `Shared` vertices' system memory.
    fn made(&self, pass: &DepthPass) -> BTreeSet<Degradation> {
        let mut kinds = made_render_kinds(&self.device, &self.pipeline(pass));
        if shared_in_system_memory(&self.device) {
            kinds.insert(Degradation::SharedInSystemMemory);
        }
        kinds
    }

    fn new(device: Device) -> Depths {
        let bitcode = llvm_assemble(&engine_dir().join("cfb_sample_depth.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        Depths {
            queue: device.new_command_queue().unwrap(),
            library,
            device,
        }
    }

    /// A 2x2 render target of `pf` and `samples` that shaders read (a combined format through
    /// its `X32Stencil8` view), private; or memoryless.
    fn texture(&self, pf: PixelFormat, samples: u64, memoryless: bool) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, EXTENT[0].into(), EXTENT[1].into(), false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
        }
        d.usage = match (memoryless, pf) {
            (true, _) => TextureUsage::RENDER_TARGET,
            // The stencil read's view, and the color read's (`runtime_samples::encode`).
            (false, PixelFormat::Depth32FloatStencil8 | PixelFormat::RGBA8Unorm) => {
                TextureUsage::RENDER_TARGET
                    | TextureUsage::SHADER_READ
                    | TextureUsage::PIXEL_FORMAT_VIEW
            }
            (false, _) => TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        };
        d.resource_options = match memoryless {
            true => ResourceOptions::new(
                StorageMode::Memoryless,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            ),
            false => ResourceOptions::private(),
        };
        self.device.new_texture(&d).unwrap()
    }

    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// Encode `pass`, then `late` on the encoder; commit and wait. The first refusal.
    fn run(
        &self,
        pass: &DepthPass,
        late: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let command_buffer = self.queue.command_buffer()?;
        let held = self.run_in(&command_buffer, pass, late)?;
        finish(&command_buffer, held);
        Ok(())
    }

    /// Encode `pass`, then `late` on the encoder, into `command_buffer`: the first refusal, else
    /// the buffers the draw reads, which the caller holds until the command buffer completes.
    fn run_in(
        &self,
        command_buffer: &CommandBuffer,
        pass: &DepthPass,
        late: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<Vec<Buffer>, Error> {
        let mut descriptor = RenderPassDescriptor::new();
        if let Some(color) = &pass.color {
            let target = &mut descriptor.color_attachments[0];
            target.texture = Some(color.clone());
            target.load_action = LoadAction::Clear;
            target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            target.store_action = pass.color_store;
        }
        let target = &mut descriptor.depth_attachment;
        target.texture = pass.depth.clone();
        target.load_action = pass.depth_load;
        target.clear_depth = pass.clear_depth;
        target.store_action = pass.depth_store;
        target.store_action_options = pass.depth_options;
        target.resolve_texture = pass.depth_resolve.clone();
        let target = &mut descriptor.stencil_attachment;
        target.texture = pass.stencil.clone();
        target.load_action = pass.stencil_load;
        target.clear_stencil = 0x11;
        target.store_action = pass.stencil_store;
        descriptor.sample_positions = pass.positions.clone();
        let mut encoder = command_buffer.render_command_encoder(&descriptor)?;
        let mut held = Vec::new();
        if pass.draw {
            let pipeline = self
                .device
                .new_render_pipeline_state(&self.pipeline(pass), &PipelineOptions::default())
                .unwrap();
            let mut state = DepthStencilDescriptor::default();
            if pass.depth.is_some() {
                state.depth_compare_function = pass.compare;
                state.depth_write_enabled = true;
            }
            if pass.stencil.is_some() {
                let mut face = StencilDescriptor::default();
                face.stencil_compare_function = pass.stencil_compare;
                face.depth_stencil_pass_operation = StencilOperation::Replace;
                state.front_face_stencil = face;
                state.back_face_stencil = face;
            }
            let state = self.device.new_depth_stencil_state(&state).unwrap();
            let vertices: Vec<u8> = pass
                .plane
                .vertices()
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let vertices = buffer_with(&self.device, &vertices).unwrap();
            encoder.set_render_pipeline_state(&pipeline)?;
            encoder.set_depth_stencil_state(&state)?;
            encoder.set_stencil_reference_value(pass.reference);
            encoder.set_vertex_buffer(Some(&vertices), 0, 0)?;
            encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
            held.push(vertices);
        }
        late(&mut encoder)?;
        encoder.end_encoding()?;
        Ok(held)
    }

    /// Every sample of `texture` (2x2, `samples`), read as `read` through `source`, bytes.
    fn read(&self, source: &Texture, read: SampleRead, samples: u64) -> Vec<u8> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let back = encode(
            &self.device,
            &command_buffer,
            source,
            read,
            EXTENT,
            samples as u32,
        )
        .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        back.bytes().unwrap()
    }

    /// Each sample's depth.
    fn depths(&self, texture: &Texture, samples: u64) -> Vec<f32> {
        floats(&self.read(texture, SampleRead::Depth, samples))
    }

    /// Each pixel's depth of a single-sampled `Depth32Float` texture, copied to a buffer.
    fn single_depths(&self, texture: &Texture) -> Vec<f32> {
        let length = u64::from(EXTENT[0] * EXTENT[1]) * 4;
        let buffer = self
            .device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(EXTENT[0].into(), EXTENT[1].into(), 1),
            &buffer,
            0,
            u64::from(EXTENT[0]) * 4,
            length,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        read_buffer(&buffer, 0, length)
            .unwrap()
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Each sample's stencil (a combined texture's through its `X32Stencil8` view).
    fn stencils(&self, texture: &Texture, samples: u64) -> Vec<u8> {
        let source = match texture.pixel_format() {
            PixelFormat::Depth32FloatStencil8 => {
                texture.new_texture_view(PixelFormat::X32Stencil8).unwrap()
            }
            _ => texture.clone(),
        };
        self.read(&source, SampleRead::Stencil, samples)
    }

    /// Whether each sample of an `RGBA8Unorm` target was drawn (its red channel).
    fn drawn(&self, texture: &Texture, samples: u64) -> Vec<bool> {
        self.read(texture, SampleRead::Color(ReadbackView::R32Uint), samples)
            .chunks(4)
            .map(|texel| texel[0] == 0xff)
            .collect()
    }

    /// A caller's 2x2 image of `format` (Metal's `pf`) and four samples, made without
    /// `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, wrapped as a private render
    /// target shaders read (`new_texture_with_vk_image`); and the image and its memory, which the
    /// test destroys after the texture and every command buffer that used it.
    fn callers(
        &self,
        pf: PixelFormat,
        format: ash::vk::Format,
    ) -> (Texture, ash::vk::Image, ash::vk::DeviceMemory) {
        use ash::vk;
        // SAFETY: used only while the device lives.
        let raw = unsafe { self.device.raw() };
        // SAFETY: a physical device of the instance.
        let memory = unsafe {
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device)
        };
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let info = image_info(format, usage)
            .extent(vk::Extent3D {
                width: EXTENT[0],
                height: EXTENT[1],
                depth: 1,
            })
            .samples(vk::SampleCountFlags::TYPE_4);
        let info = shared_among(info, raw.queue_families);
        // SAFETY: the device is live.
        let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
        let mut d = TextureDescriptor::texture_2d(pf, EXTENT[0].into(), EXTENT[1].into(), false);
        d.texture_type = TextureType::Type2DMultisample;
        d.sample_count = 4;
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let layouts = VkImageLayouts::new(
            vk::ImageLayout::UNDEFINED.into(),
            vk::ImageLayout::GENERAL.into(),
        );
        // SAFETY: the image is the device's, made with `info`, and nothing else uses it.
        let texture = unsafe {
            self.device
                .new_texture_with_vk_image(&d, image, &info, layouts)
        }
        .unwrap();
        (texture, image, image_memory)
    }

    /// Whether the device takes custom positions at `samples` with a depth attachment: it takes
    /// them at that count (`sample_positions.rs`).
    fn located(&self, samples: u64) -> bool {
        let d = &self.device.properties().description;
        d.extensions.contains(&VkExtension::ExtSampleLocations)
            && d.limits
                .sample_location_sample_counts
                .contains(SampleCountFlags::from_raw(samples as u32))
            && d.limits.variable_sample_locations
    }
}

/// Commit `command_buffer` and wait for it, then drop `held`, what its commands read.
fn finish(command_buffer: &CommandBuffer, held: Vec<Buffer>) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drop(held);
}

/// Little-endian `f32`s.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Each sample's depth, against the depths Metal stored: the same within a millionth (Metal's
/// are exact; a device's interpolation may round the last bits), and within a 16-bit unit for
/// `Depth16Unorm`.
fn assert_depths(got: &[f32], metal: &[f32], tolerance: f32, name: &str) {
    assert_eq!(got.len(), metal.len(), "{name}: {got:?}");
    let off = got
        .iter()
        .zip(metal)
        .any(|(g, m)| (g - m).abs() > tolerance);
    assert!(!off, "{name}: stored {got:?}, Metal {metal:?}");
}

fn depth_lane() -> (Messages, ScratchCache, Depths) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    (messages, cache, Depths::new(device))
}

fn depths_quiet(lane: Depths, messages: &Messages) {
    drop(lane);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// A depth attachment takes the pass's custom positions (`dz_*`): each sample stores the plane's
/// depth at its position rounded to the sixteenth, four positions and two, `Depth32Float` and
/// `Depth16Unorm`, beside a color attachment too; the depth test compares there
/// (`dz_custom4_test`: cleared to 0.4375, `Less`, two samples of each first-column pixel
/// drawn), and so does a memoryless depth texture's (`dz_custom4_memoryless`). The device's own
/// positions are Vulkan's standard ones, as Metal's are. A device that takes no positions at a
/// count is skipped there (the color lane checks its refusal). On a driver that keeps the depth
/// per sample only through a copy (`DriverFact::PositionedDepthReadAtStandardPositions`) the
/// texture's own image renders through the stand-in, announced once; elsewhere nothing is
/// announced.
#[test]
fn depth_takes_the_custom_positions() {
    let (messages, _cache, lane) = depth_lane();
    let device = lane.device.name().to_owned();
    let float = PixelFormat::Depth32Float;
    for (name, samples, points, custom) in [
        ("default4", 4, &STANDARD4[..], false),
        ("custom4", 4, &P4[..], true),
        ("default2", 2, &STANDARD2[..], false),
        ("custom2", 2, &P2[..], true),
    ] {
        if !lane
            .device
            .properties()
            .supports_texture_sample_count(samples)
            .unwrap()
            || (custom && !lane.located(samples))
        {
            eprintln!("dz_{name} on {device}: not taken");
            continue;
        }
        for plane in [Plane::X, Plane::Y] {
            let depth = lane.texture(float, samples, false);
            let mut pass = DepthPass::new();
            pass.samples = samples;
            pass.depth = Some(depth.clone());
            pass.plane = plane;
            if custom {
                pass.positions = positions_of(points);
            }
            lane.run(&pass, |_| Ok(())).unwrap();
            let name = format!("dz_{name}_{plane:?} on {device}");
            assert_depths(
                &lane.depths(&depth, samples),
                &plane.depths(points),
                1e-6,
                &name,
            );
        }
    }
    if !lane.located(4) {
        eprintln!("the 4x depth passes on {device}: not taken");
        depths_quiet(lane, &messages);
        return;
    }
    let metal = Plane::X.depths(&P4);

    // Beside a color attachment, every sample drawn.
    let depth = lane.texture(float, 4, false);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    pass.color = Some(color.clone());
    lane.run(&pass, |_| Ok(())).unwrap();
    assert_depths(&lane.depths(&depth, 4), &metal, 1e-6, "dz_custom4_color");
    assert_eq!(lane.drawn(&color, 4), vec![true; 16], "dz_custom4_color");

    // `Depth16Unorm`, each depth in 16-bit units.
    let depth = lane.texture(PixelFormat::Depth16Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    lane.run(&pass, |_| Ok(())).unwrap();
    assert_depths(
        &lane.depths(&depth, 4),
        &metal,
        1.0 / 65535.0,
        "dz16_custom4_x",
    );

    // The depth test at the positions: cleared to 0.4375 (plane X at the first column's middle),
    // `Less`; Metal drew samples 0 and 1 of each first-column pixel, and kept the clear elsewhere.
    let passed: Vec<bool> = metal.iter().map(|&d| d < 0.4375).collect();
    let kept: Vec<f32> = metal.iter().map(|&d| d.min(0.4375)).collect();
    for memoryless in [false, true] {
        let name = match memoryless {
            false => "dz_custom4_test",
            true => "dz_custom4_memoryless",
        };
        let depth = lane.texture(float, 4, memoryless);
        let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
        let mut pass = DepthPass::new();
        pass.positions = positions_of(&P4);
        pass.depth = Some(depth.clone());
        pass.color = Some(color.clone());
        pass.clear_depth = 0.4375;
        pass.compare = CompareFunction::Less;
        if memoryless {
            pass.depth_store = StoreAction::DontCare;
        }
        lane.run(&pass, |_| Ok(())).unwrap();
        assert_eq!(lane.drawn(&color, 4), passed, "{name}");
        if !memoryless {
            assert_depths(&lane.depths(&depth, 4), &kept, 1e-6, name);
        }
    }
    // A driver that keeps the depth per sample only through a copy renders the texture's own
    // image through the stand-in, announced once naming the fact; elsewhere nothing is.
    let notices = messages.take_notices();
    let planes = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::PositionedDepthReadAtStandardPositions);
    let announced = counts(&notices)
        .get(&Degradation::PositionsThroughScratch)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        announced,
        usize::from(planes),
        "positions-through-scratch on {device}"
    );
    if planes {
        let reason = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::PositionsThroughScratch)
            .map(|notice| notice.reason.clone())
            .unwrap();
        assert!(
            reason.contains("PositionedDepthReadAtStandardPositions"),
            "the stand-in's notice on {device}: {reason}"
        );
    }
    eprintln!("the 4x depth passes on {device}: through the stand-in {planes}");
    depths_quiet(lane, &messages);
}

/// A caller's image made without `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`
/// takes the pass's custom positions as Metal's texture takes them (a probe), through a stand-in
/// made with the flag, its aspects copied in and back: a `Depth32Float` image cleared and drawn
/// stores plane X's depth at each position (`dz_custom4_x`); one cleared to 0.4375 by a pass
/// without a draw, then loaded and drawn under `Less`, draws and keeps what Metal drew and kept
/// (`dz_custom4_test`). A `Depth32Float_Stencil8` image as both attachments stores the edge's
/// depths at the positions (`dsz_custom4_x`); a pass of it as the depth alone, cleared to 0.75
/// without a draw, keeps its stencil; and a pass that loads both and draws a flat plane under
/// `Equal` 0x5a colors exactly the samples the edge covered, keeping the depths. The stand-in is
/// announced once, naming the caller.
#[test]
fn a_callers_image_takes_the_positions_through_a_stand_in() {
    let (messages, _cache, lane) = depth_lane();
    let device = lane.device.name().to_owned();
    if !lane.located(4) {
        eprintln!("the 4x passes of a caller's image on {device}: not taken");
        depths_quiet(lane, &messages);
        return;
    }
    let mut made = Vec::new();

    // `Depth32Float`, cleared and drawn.
    let (depth, image, memory) =
        lane.callers(PixelFormat::Depth32Float, ash::vk::Format::D32_SFLOAT);
    made.push((image, memory));
    let metal = Plane::X.depths(&P4);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    lane.run(&pass, |_| Ok(())).unwrap();
    // The drawing passes' pipelines' makings (the sites).
    let mut first = lane.made(&pass);
    let name = format!("a caller's dz_custom4_x on {device}");
    assert_depths(&lane.depths(&depth, 4), &metal, 1e-6, &name);

    // Loaded, under `Less`: Metal drew samples 0 and 1 of each first-column pixel.
    let mut clear = DepthPass::new();
    clear.positions = positions_of(&P4);
    clear.depth = Some(depth.clone());
    clear.clear_depth = 0.4375;
    clear.draw = false;
    lane.run(&clear, |_| Ok(())).unwrap();
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    pass.depth_load = LoadAction::Load;
    pass.color = Some(color.clone());
    pass.compare = CompareFunction::Less;
    lane.run(&pass, |_| Ok(())).unwrap();
    first.extend(lane.made(&pass));
    let passed: Vec<bool> = metal.iter().map(|&d| d < 0.4375).collect();
    let kept: Vec<f32> = metal.iter().map(|&d| d.min(0.4375)).collect();
    let name = format!("a caller's dz_custom4_test on {device}");
    assert_eq!(lane.drawn(&color, 4), passed, "{name}");
    assert_depths(&lane.depths(&depth, 4), &kept, 1e-6, &name);

    // `Depth32Float_Stencil8`, as both attachments, then as the depth alone, then both loaded.
    let (combined, image, memory) = lane.callers(
        PixelFormat::Depth32FloatStencil8,
        ash::vk::Format::D32_SFLOAT_S8_UINT,
    );
    made.push((image, memory));
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(combined.clone());
    pass.stencil = Some(combined.clone());
    pass.plane = Plane::Edge;
    lane.run(&pass, |_| Ok(())).unwrap();
    first.extend(lane.made(&pass));
    let covers = edge_covers(&P4);
    let depths: Vec<f32> = covers.iter().map(|&c| if c { 0.5 } else { 1.0 }).collect();
    let name = format!("a caller's dsz_custom4_x on {device}");
    assert_depths(&lane.depths(&combined, 4), &depths, 0.0, &name);
    let mut alone = DepthPass::new();
    alone.positions = positions_of(&P4);
    alone.depth = Some(combined.clone());
    alone.clear_depth = 0.75;
    alone.draw = false;
    lane.run(&alone, |_| Ok(())).unwrap();
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(combined.clone());
    pass.depth_load = LoadAction::Load;
    pass.stencil = Some(combined.clone());
    pass.stencil_load = LoadAction::Load;
    pass.stencil_compare = CompareFunction::Equal;
    pass.color = Some(color.clone());
    pass.plane = Plane::Flat;
    lane.run(&pass, |_| Ok(())).unwrap();
    first.extend(lane.made(&pass));
    let name = format!("a caller's stencil kept and tested on {device}");
    assert_eq!(lane.drawn(&color, 4), covers, "{name}");
    let depths: Vec<f32> = covers.iter().map(|&c| if c { 0.5 } else { 0.75 }).collect();
    assert_depths(&lane.depths(&combined, 4), &depths, 0.0, &name);

    let notices = messages.take_notices();
    let mut kinds: Vec<Degradation> = first.into_iter().collect();
    kinds.extend(POSITIONED);
    assert_once(&notices, &kinds);
    let reason = notices
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::PositionsThroughScratch)
        .map(|notice| notice.reason.clone())
        .unwrap();
    assert!(
        reason.contains("the caller made it without it"),
        "the stand-in's notice on {device}: {reason}"
    );
    drop((depth, combined));
    // SAFETY: used only while the device lives; nothing uses the images: their textures are gone
    // and every command buffer completed.
    let raw = unsafe { lane.device.raw() };
    for (image, memory) in made {
        // SAFETY: as above.
        unsafe {
            raw.device.destroy_image(image, None);
            raw.device.free_memory(memory, None);
        }
    }
    depths_quiet(lane, &messages);
}

/// The stencil test covers the samples at the custom positions (`dsz_custom4_x`, a
/// `Depth32Float_Stencil8` texture: the edge's samples left of the first column's middle take
/// 0x5a and its depth 0.5, the rest keep 0x11 and 1); a `Stencil8` texture alone covers them at
/// the device's positions (`s8_default4_edge`) and at custom ones (`s8_custom4_edge`), which a
/// device whose `Stencil8` image has no depth aspect renders through a combined scratch
/// (`depth_stencil_scratch.rs`).
#[test]
fn stencil_covers_the_samples_at_the_positions() {
    let (messages, _cache, lane) = depth_lane();
    let device = lane.device.name().to_owned();
    let stencils = |covers: Vec<bool>| -> Vec<u8> {
        covers
            .into_iter()
            .map(|c| if c { 0x5a } else { 0x11 })
            .collect()
    };

    let texture = lane.texture(PixelFormat::Stencil8, 4, false);
    let mut pass = DepthPass::new();
    pass.stencil = Some(texture.clone());
    pass.plane = Plane::Edge;
    lane.run(&pass, |_| Ok(())).unwrap();
    assert_eq!(
        lane.stencils(&texture, 4),
        stencils(edge_covers(&STANDARD4)),
        "s8_default4_edge on {device}"
    );
    if !lane.located(4) {
        eprintln!("the 4x stencil passes on {device}: not taken");
        depths_quiet(lane, &messages);
        return;
    }

    let combined = lane.texture(PixelFormat::Depth32FloatStencil8, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(combined.clone());
    pass.stencil = Some(combined.clone());
    pass.plane = Plane::Edge;
    lane.run(&pass, |_| Ok(())).unwrap();
    let covers = edge_covers(&P4);
    let depths: Vec<f32> = covers.iter().map(|&c| if c { 0.5 } else { 1.0 }).collect();
    assert_depths(&lane.depths(&combined, 4), &depths, 0.0, "dsz_custom4_x");
    assert_eq!(
        lane.stencils(&combined, 4),
        stencils(covers.clone()),
        "dsz_custom4_x on {device}"
    );

    let texture = lane.texture(PixelFormat::Stencil8, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.stencil = Some(texture.clone());
    pass.plane = Plane::Edge;
    lane.run(&pass, |_| Ok(())).unwrap();
    assert_eq!(
        lane.stencils(&texture, 4),
        stencils(covers),
        "s8_custom4_edge on {device}"
    );
    depths_quiet(lane, &messages);
}

/// A `Stencil8` texture alone keeps its samples across passes at other positions (Metal probe
/// 11): loaded and not drawn, at Q4 (`s8_custom4_then_other_load`) and at the device's positions
/// (`s8_custom4_then_default_load`), every sample is the one stored at P4; the edge drawn again at
/// Q4 with reference 0x33 replaces exactly the samples it covers at Q4 and keeps the rest
/// (`s8_custom4_then_other_edge`); a flat draw at Q4 under `Equal` 0x5a colors exactly the samples
/// that stored 0x5a at P4 (`s8_custom4_then_other_test`); and a pass that does not store its
/// stencil still tests it (`s8_custom4_dontcare`: the edge's color at P4).
#[test]
fn a_stencil_alone_keeps_its_samples_in_later_passes() {
    let (messages, _cache, lane) = depth_lane();
    let device = lane.device.name().to_owned();
    if !lane.located(4) {
        eprintln!("the 4x stencil passes on {device}: not taken");
        depths_quiet(lane, &messages);
        return;
    }
    let first = |lane: &Depths| -> Texture {
        let texture = lane.texture(PixelFormat::Stencil8, 4, false);
        let mut pass = DepthPass::new();
        pass.positions = positions_of(&P4);
        pass.stencil = Some(texture.clone());
        pass.plane = Plane::Edge;
        lane.run(&pass, |_| Ok(())).unwrap();
        texture
    };
    let at_p4 = edge_covers(&P4);
    let stored: Vec<u8> = at_p4.iter().map(|&c| if c { 0x5a } else { 0x11 }).collect();

    for (name, positions) in [
        ("s8_custom4_then_other_load", positions_of(&Q4)),
        ("s8_custom4_then_default_load", Vec::new()),
    ] {
        let texture = first(&lane);
        let mut again = DepthPass::new();
        again.positions = positions;
        again.stencil = Some(texture.clone());
        again.stencil_load = LoadAction::Load;
        again.draw = false;
        lane.run(&again, |_| Ok(())).unwrap();
        assert_eq!(lane.stencils(&texture, 4), stored, "{name} on {device}");
    }

    let texture = first(&lane);
    let mut again = DepthPass::new();
    again.positions = positions_of(&Q4);
    again.stencil = Some(texture.clone());
    again.stencil_load = LoadAction::Load;
    again.plane = Plane::Edge;
    again.reference = 0x33;
    lane.run(&again, |_| Ok(())).unwrap();
    let redrawn: Vec<u8> = edge_covers(&Q4)
        .iter()
        .zip(&stored)
        .map(|(&c, &s)| if c { 0x33 } else { s })
        .collect();
    assert_eq!(
        lane.stencils(&texture, 4),
        redrawn,
        "s8_custom4_then_other_edge on {device}"
    );

    let texture = first(&lane);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut again = DepthPass::new();
    again.positions = positions_of(&Q4);
    again.stencil = Some(texture.clone());
    again.stencil_load = LoadAction::Load;
    again.stencil_compare = CompareFunction::Equal;
    again.color = Some(color.clone());
    again.plane = Plane::Flat;
    lane.run(&again, |_| Ok(())).unwrap();
    let name = "s8_custom4_then_other_test";
    assert_eq!(lane.stencils(&texture, 4), stored, "{name} on {device}");
    assert_eq!(lane.drawn(&color, 4), at_p4, "{name} on {device}");

    let texture = lane.texture(PixelFormat::Stencil8, 4, false);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.stencil = Some(texture);
    pass.stencil_store = StoreAction::DontCare;
    pass.color = Some(color.clone());
    pass.plane = Plane::Edge;
    lane.run(&pass, |_| Ok(())).unwrap();
    assert_eq!(
        lane.drawn(&color, 4),
        at_p4,
        "s8_custom4_dontcare on {device}"
    );
    depths_quiet(lane, &messages);
}

/// Positions move rasterization alone: a later pass that loads the depth keeps every sample
/// stored at P4, at the device's positions (`dz_custom4_then_default_load`) and at others
/// (`dz_custom4_then_other_load`), and one that draws plane X again at Q4 under `Less` draws
/// exactly the samples whose depth at the new position lies below the stored one, and stores the
/// lesser (`dz_custom4_then_other_test`: samples 1 to 3 of each pixel). The depths stored at P4
/// read in the command buffer that stored them, and the other order, a depth stored at the device's
/// positions and loaded or drawn again at P4, alone and after a pass at Q4 in one command buffer.
#[test]
fn later_passes_keep_the_stored_samples() {
    let (messages, _cache, lane) = depth_lane();
    if !lane.located(4) {
        eprintln!("the 4x depth passes on {}: not taken", lane.device.name());
        depths_quiet(lane, &messages);
        return;
    }
    let stored = Plane::X.depths(&P4);
    for (name, positions) in [
        ("dz_custom4_then_default_load", Vec::new()),
        ("dz_custom4_then_other_load", positions_of(&Q4)),
    ] {
        let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
        let mut pass = DepthPass::new();
        pass.positions = positions_of(&P4);
        pass.depth = Some(depth.clone());
        lane.run(&pass, |_| Ok(())).unwrap();
        let mut again = DepthPass::new();
        again.positions = positions;
        again.depth = Some(depth.clone());
        again.depth_load = LoadAction::Load;
        again.draw = false;
        lane.run(&again, |_| Ok(())).unwrap();
        assert_depths(&lane.depths(&depth, 4), &stored, 1e-6, name);
    }

    // Read in the command buffer that stored it (`dz_custom4_read_in_its_command_buffer`).
    let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    let command_buffer = lane.queue.command_buffer().unwrap();
    let held = lane.run_in(&command_buffer, &pass, |_| Ok(())).unwrap();
    let back = encode(
        &lane.device,
        &command_buffer,
        &depth,
        SampleRead::Depth,
        EXTENT,
        4,
    )
    .unwrap();
    finish(&command_buffer, held);
    let name = "dz_custom4_read_in_its_command_buffer";
    assert_depths(&floats(&back.bytes().unwrap()), &stored, 1e-6, name);

    // The other order, a depth stored at the device's positions and loaded by a pass at P4,
    // keeps each stored sample by the same rule (Metal's probe measured the order above): in a
    // command buffer of its own (`dz_default4_then_custom_load`), and after a pass at Q4 in the
    // same one (`dz_default4_then_custom_load_after_other`); and one that draws plane X at P4
    // under `Less` draws where the depth at P4 lies below the stored one, and keeps the lesser
    // (`dz_default4_then_custom_test`: samples 0 and 1 of each pixel).
    let standard = Plane::X.depths(&STANDARD4);
    for (name, after_other) in [
        ("dz_default4_then_custom_load", false),
        ("dz_default4_then_custom_load_after_other", true),
    ] {
        let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
        let mut pass = DepthPass::new();
        pass.depth = Some(depth.clone());
        lane.run(&pass, |_| Ok(())).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut held = Vec::new();
        if after_other {
            let mut other = DepthPass::new();
            other.positions = positions_of(&Q4);
            other.depth = Some(lane.texture(PixelFormat::Depth32Float, 4, false));
            held.extend(lane.run_in(&command_buffer, &other, |_| Ok(())).unwrap());
        }
        let mut again = DepthPass::new();
        again.positions = positions_of(&P4);
        again.depth = Some(depth.clone());
        again.depth_load = LoadAction::Load;
        again.draw = false;
        held.extend(lane.run_in(&command_buffer, &again, |_| Ok(())).unwrap());
        finish(&command_buffer, held);
        assert_depths(&lane.depths(&depth, 4), &standard, 1e-6, name);
    }
    let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.depth = Some(depth.clone());
    lane.run(&pass, |_| Ok(())).unwrap();
    let mut again = DepthPass::new();
    again.positions = positions_of(&P4);
    again.depth = Some(depth.clone());
    again.depth_load = LoadAction::Load;
    again.color = Some(color.clone());
    again.compare = CompareFunction::Less;
    lane.run(&again, |_| Ok(())).unwrap();
    let passed: Vec<bool> = stored.iter().zip(&standard).map(|(p, s)| p < s).collect();
    let kept: Vec<f32> = stored
        .iter()
        .zip(&standard)
        .map(|(p, s)| p.min(*s))
        .collect();
    assert_eq!(
        passed.iter().filter(|&&p| p).count(),
        8,
        "two samples a pixel"
    );
    let name = "dz_default4_then_custom_test";
    assert_eq!(lane.drawn(&color, 4), passed, "{name}");
    assert_depths(&lane.depths(&depth, 4), &kept, 1e-6, name);

    let depth = lane.texture(PixelFormat::Depth32Float, 4, false);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let mut pass = DepthPass::new();
    pass.positions = positions_of(&P4);
    pass.depth = Some(depth.clone());
    lane.run(&pass, |_| Ok(())).unwrap();
    let mut again = DepthPass::new();
    again.positions = positions_of(&Q4);
    again.depth = Some(depth.clone());
    again.depth_load = LoadAction::Load;
    again.color = Some(color.clone());
    again.compare = CompareFunction::Less;
    lane.run(&again, |_| Ok(())).unwrap();
    let moved = Plane::X.depths(&Q4);
    let passed: Vec<bool> = moved.iter().zip(&stored).map(|(m, s)| m < s).collect();
    let kept: Vec<f32> = moved.iter().zip(&stored).map(|(m, s)| m.min(*s)).collect();
    assert_eq!(
        passed.iter().filter(|&&p| p).count(),
        12,
        "Metal drew three samples a pixel"
    );
    assert_eq!(lane.drawn(&color, 4), passed, "dz_custom4_then_other_test");
    assert_depths(
        &lane.depths(&depth, 4),
        &kept,
        1e-6,
        "dz_custom4_then_other_test",
    );
    depths_quiet(lane, &messages);
}

/// `CustomSampleDepthStore` stores a depth attachment as `Store` does (`csds_*`): at custom
/// positions and the device's, at one sample, with the option `CustomSamplePositions` (which
/// Metal's validation stops and Metal without it takes: rule 7), late, and loaded by a later
/// pass. What Metal stops, refused typed: on a color or stencil attachment, the descriptor's
/// (Metal's own check, both modes); with a resolve texture and on a memoryless texture, as
/// `Store` is; late over a store action the descriptor knew. A late one on a color or stencil
/// attachment, which plain Metal leaves unstored and its validation stops, stores as `DontCare`.
#[test]
fn custom_sample_depth_store_stores_as_store() {
    let (messages, _cache, lane) = depth_lane();
    let device = lane.device.name().to_owned();
    let custom = StoreAction::CustomSampleDepthStore;
    let float = PixelFormat::Depth32Float;
    let located = lane.located(4);
    let encoder_call = "render_command_encoder";

    // Stored as `Store`: (name, samples, at P4, the option, set late).
    for (name, samples, at_p4, option, late) in [
        ("csds_depth_custom4", 4, true, false, false),
        ("csds_depth_default4", 4, false, false, false),
        ("csds_depth_1x", 1, false, false, false),
        ("csds_depth_custom4_options", 4, true, true, false),
        ("csds_late_depth", 4, true, false, true),
        ("csds_late_depth_default4", 4, false, false, true),
    ] {
        if at_p4 && !located {
            eprintln!("{name} on {device}: no custom positions at 4");
            continue;
        }
        let depth = lane.texture(float, samples, false);
        let mut pass = DepthPass::new();
        pass.samples = samples;
        pass.depth = Some(depth.clone());
        pass.depth_store = if late { StoreAction::Unknown } else { custom };
        let points: Vec<(f32, f32)> = match (samples, at_p4) {
            (1, _) => vec![(0.5, 0.5)],
            (_, true) => P4.to_vec(),
            _ => STANDARD4.to_vec(),
        };
        if at_p4 {
            pass.positions = positions_of(&P4);
        }
        if option {
            pass.depth_options = CUSTOM;
        }
        lane.run(&pass, |encoder| match late {
            true => encoder.set_depth_store_action(custom),
            false => Ok(()),
        })
        .unwrap_or_else(|e| panic!("{name} on {device}: {e:?}"));
        let read = match samples {
            1 => lane.single_depths(&depth),
            _ => lane.depths(&depth, samples),
        };
        assert_depths(&read, &Plane::X.depths(&points), 1e-6, name);
    }

    // Loaded by a later pass at the device's positions.
    if located {
        let depth = lane.texture(float, 4, false);
        let mut pass = DepthPass::new();
        pass.positions = positions_of(&P4);
        pass.depth = Some(depth.clone());
        pass.depth_store = custom;
        lane.run(&pass, |_| Ok(())).unwrap();
        let mut again = DepthPass::new();
        again.depth = Some(depth.clone());
        again.depth_load = LoadAction::Load;
        again.draw = false;
        lane.run(&again, |_| Ok(())).unwrap();
        let name = "csds_depth_then_load";
        assert_depths(&lane.depths(&depth, 4), &Plane::X.depths(&P4), 1e-6, name);
    }

    // Refused.
    let combined = lane.texture(PixelFormat::Depth32FloatStencil8, 4, false);
    let stencil8 = lane.texture(PixelFormat::Stencil8, 4, false);
    let color = lane.texture(PixelFormat::RGBA8Unorm, 4, false);
    let csds = |which| {
        invalid(
            encoder_call,
            RenderPassProblem::CustomSampleDepthStore(which),
        )
    };
    let mut pass = DepthPass::new();
    pass.depth = Some(combined.clone());
    pass.stencil = Some(combined.clone());
    pass.stencil_store = custom;
    assert_eq!(
        lane.run(&pass, |_| Ok(())),
        Err(csds(Attachment::Stencil)),
        "csds_stencil"
    );
    let mut pass = DepthPass::new();
    pass.stencil = Some(stencil8.clone());
    pass.stencil_store = custom;
    assert_eq!(
        lane.run(&pass, |_| Ok(())),
        Err(csds(Attachment::Stencil)),
        "csds_stencil8_only"
    );
    let mut pass = DepthPass::new();
    pass.color = Some(color.clone());
    pass.color_store = custom;
    assert_eq!(
        lane.run(&pass, |_| Ok(())),
        Err(csds(Attachment::Color(0))),
        "csds_color_default4"
    );
    let resolve = lane.texture(float, 1, false);
    let mut pass = DepthPass::new();
    pass.depth = Some(lane.texture(float, 4, false));
    pass.depth_store = custom;
    pass.depth_resolve = Some(resolve);
    // As `Store` with a resolve texture: the texture is left untouched and the depth stored, as
    // plain Metal does.
    assert_eq!(lane.run(&pass, |_| Ok(())), Ok(()), "csds_depth_resolve");
    let mut pass = DepthPass::new();
    pass.depth = Some(lane.texture(float, 4, true));
    pass.depth_store = custom;
    assert_eq!(
        lane.run(&pass, |_| Ok(())),
        Err(invalid(
            encoder_call,
            RenderPassProblem::MemorylessStore(Attachment::Depth)
        )),
        "csds_memoryless"
    );

    // Late.
    let mut pass = DepthPass::new();
    pass.depth = Some(lane.texture(float, 4, false));
    assert_eq!(
        lane.run(&pass, |encoder| encoder.set_depth_store_action(custom)),
        Err(invalid(
            "set_depth_store_action",
            RenderPassProblem::StoreActionKnown(Attachment::Depth)
        )),
        "csds_late_over_known"
    );
    let mut pass = DepthPass::new();
    pass.depth = Some(combined.clone());
    pass.stencil = Some(combined.clone());
    pass.depth_store = StoreAction::Unknown;
    pass.stencil_store = StoreAction::Unknown;
    // A late one of a color or stencil attachment, which plain Metal leaves unstored, stores as
    // `DontCare`.
    assert_eq!(
        lane.run(&pass, |encoder| {
            encoder.set_depth_store_action(StoreAction::Store)?;
            encoder.set_stencil_store_action(custom)
        }),
        Ok(()),
        "csds_late_stencil"
    );
    let mut pass = DepthPass::new();
    pass.color = Some(color);
    pass.color_store = StoreAction::Unknown;
    assert_eq!(
        lane.run(&pass, |encoder| encoder.set_color_store_action(custom, 0)),
        Ok(()),
        "csds_late_color"
    );
    depths_quiet(lane, &messages);
}
