//! Device tests of `[[barycentric_coord]]`'s three weights on the device the cfg selects, under
//! the validation layer. The owned fixture `bary_weights` places each vertex at the clip-space
//! position buffer 0 holds for it, and its fragment functions return the three weights as a
//! `float3`, perspective-correct (`bw_perspective`) and not (`bw_linear`), into an `RGBA32Float`
//! attachment cleared to `(-1, -1, -1, 0)`, so a texel of alpha 1 is a pixel the draw covered.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, the fixture's functions and draws):
//!
//! - The third weight is `1 - (x + y)` of the first two in `float`, at every pixel of triangles and
//!   lines, perspective-correct or not ([`THIRD_IS_ONE_MINUS_THE_FIRST_TWO`]'s draws; also 2,341
//!   of 2,341 samples at four samples). The device's own third lane is not: the RTX 5080 Laptop
//!   GPU's `BaryCoordKHR.z` was `1 - (x + y)` at 42 of 574 pixels with perspective and none
//!   without, and exactly 0 on a line, where Metal's is not.
//! - At points where every weight is a short binary fraction, the weights are those fractions
//!   exactly: the fullscreen triangle over 4x4 pixels, and a horizontal line through the centers of
//!   row 3 of 8x8 ([`exact_triangle`], [`exact_line`]).
//!
//! A device without `fragmentShaderBarycentric` reads the weights as varyings a generated geometry
//! stage writes, one per topology class, of which a draw runs its class's: each check runs on the
//! device as made and again with the feature withheld ([`Path`]), and every draw asserts how many
//! geometry stages the pipeline it executed carries (three through varyings, none through the
//! builtins). [`CARRIED`]'s draws compare what `bv_weights`, which reads varyings (a smooth `float4`,
//! a flat `uint` and a `float2`), the weights and the primitive ID, reads of the rest with what
//! `bv_plain`, which reads no weights and so has no geometry stage, reads at every pixel. A device
//! with neither way, such as the `vk12-floor` profile's, refuses the functions as needing either
//! `fragmentShaderBarycentric` or `geometryShader`: every device is checked so with both features
//! withheld.

use super::{
    buffer_with, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Degradation, Device, DeviceOptions, Error, ExecutedPipeline, Library,
    LoadAction, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, TextureDescriptor,
};

/// The fixture's fragment functions: perspective-correct weights, then linear ones.
const FRAGMENTS: [&str; 2] = ["bw_perspective", "bw_linear"];

/// A draw whose third weight is checked at every pixel it covers: its primitive type, its
/// vertices' clip-space positions, its square target's side, and how many pixels it covers at
/// least (the M3 Max covered 574 and 86).
#[derive(Clone, Copy)]
struct Draw {
    what: &'static str,
    primitive: PrimitiveType,
    positions: &'static [[f32; 4]],
    size: u64,
    covers_at_least: usize,
}

/// Draws whose weights are no short binary fractions: two triangles whose vertices have `w` from
/// 1 to 5, and three slanted lines of `w` 1 to 5, each over 32x32 pixels (the probes' draws).
const THIRD_IS_ONE_MINUS_THE_FIRST_TWO: [Draw; 2] = [
    Draw {
        what: "triangles",
        primitive: PrimitiveType::Triangle,
        positions: &[
            [-0.95, -0.93, 0.0, 1.0],
            [0.91 * 3.0, -0.97 * 3.0, 0.1, 3.0],
            [-0.83 * 2.0, 0.89 * 2.0, 0.0, 2.0],
            [0.97 * 1.5, -0.2 * 1.5, 0.0, 1.5],
            [0.93 * 5.0, 0.95 * 5.0, 0.0, 5.0],
            [-0.1 * 1.25, 0.97 * 1.25, 0.0, 1.25],
        ],
        size: 32,
        covers_at_least: 500,
    },
    Draw {
        what: "lines",
        primitive: PrimitiveType::Line,
        positions: &[
            [-0.93, -0.71, 0.0, 1.0],
            [0.89 * 3.0, 0.37 * 3.0, 0.0, 3.0],
            [-0.37 * 2.0, 0.91 * 2.0, 0.0, 2.0],
            [0.61 * 1.5, -0.93 * 1.5, 0.0, 1.5],
            [-0.97 * 1.25, 0.13 * 1.25, 0.0, 1.25],
            [0.95 * 5.0, 0.29 * 5.0, 0.0, 5.0],
        ],
        size: 32,
        covers_at_least: 50,
    },
];

/// [`CARRIED`]'s draws, one of each topology that draws weights: lists and strips of triangles and
/// lines, with `w` from 1 to 5, over 32x32 pixels.
const CARRIED: [Draw; 4] = [
    THIRD_IS_ONE_MINUS_THE_FIRST_TWO[0],
    Draw {
        what: "a triangle strip",
        primitive: PrimitiveType::TriangleStrip,
        positions: &[
            [-0.9, -0.9, 0.0, 1.0],
            [0.9 * 2.0, -0.8 * 2.0, 0.0, 2.0],
            [-0.8 * 1.5, 0.9 * 1.5, 0.0, 1.5],
            [0.85 * 3.0, 0.7 * 3.0, 0.0, 3.0],
            [0.1 * 1.25, 0.95 * 1.25, 0.0, 1.25],
        ],
        size: 32,
        covers_at_least: 500,
    },
    THIRD_IS_ONE_MINUS_THE_FIRST_TWO[1],
    Draw {
        what: "a line strip",
        primitive: PrimitiveType::LineStrip,
        positions: &[
            [-0.93, -0.71, 0.0, 1.0],
            [0.89 * 3.0, 0.37 * 3.0, 0.0, 3.0],
            [-0.37 * 2.0, 0.91 * 2.0, 0.0, 2.0],
            [0.61 * 1.5, -0.93 * 1.5, 0.0, 1.5],
        ],
        size: 32,
        covers_at_least: 50,
    },
];

/// How many geometry stages a pipeline that reads the weights as varyings carries: the point,
/// line and triangle classes'.
const CLASSES: usize = 3;

/// How the lane's device reads the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// The device as the lane makes it: `BaryCoordKHR` where it has `fragmentShaderBarycentric`,
    /// varyings a geometry stage writes where not.
    Device,
    /// With `fragmentShaderBarycentric` withheld: varyings a geometry stage writes.
    Withheld,
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn with(
        messages: &Messages,
        cache: &ScratchCache,
        adjust: impl FnOnce(&mut DeviceOptions),
    ) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), adjust).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("bary_weights.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    /// Whether the device's compile profile enables `feature`.
    fn enables(&self, feature: VkFeature) -> bool {
        self.device
            .compile_device()
            .profile()
            .enabled()
            .features
            .contains(&feature)
    }

    /// How many geometry stages a pipeline whose fragment function reads the weights carries on
    /// the device: none where it reads `BaryCoordKHR`, else [`CLASSES`].
    fn weight_stages(&self) -> usize {
        match self.enables(VkFeature::FragmentShaderBarycentric) {
            true => 0,
            false => CLASSES,
        }
    }

    /// The pipeline of `vertex` and `fragment` into attachments of `formats`.
    fn pipeline(
        &self,
        vertex: &str,
        fragment: &str,
        formats: &[PixelFormat],
    ) -> Result<RenderPipelineState, Error> {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        for (i, format) in formats.iter().enumerate() {
            descriptor.color_attachments[i] = Some(ColorAttachmentDescriptor::new(*format));
        }
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The texels each attachment of `formats` holds over a `size` x `size` target after one draw
    /// of `primitive` through `positions` with `pipeline`, each as its four words: a float
    /// attachment cleared to `(-1, -1, -1, 0)`, an integer one to 0. The pipeline the draw
    /// executed must carry `geometry` geometry stages.
    fn draw(
        &self,
        pipeline: &RenderPipelineState,
        formats: &[PixelFormat],
        geometry: usize,
        primitive: PrimitiveType,
        positions: &[[f32; 4]],
        size: u64,
    ) -> Vec<Vec<[u32; 4]>> {
        let targets: Vec<_> = formats
            .iter()
            .map(|format| {
                let mut descriptor = TextureDescriptor::texture_2d(*format, size, size, false);
                descriptor.usage = TextureUsage::RENDER_TARGET;
                descriptor.resource_options = ResourceOptions::shared();
                self.device.new_texture(&descriptor).unwrap()
            })
            .collect();
        let bytes: Vec<u8> = positions
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let vertices = buffer_with(&self.device, &bytes).unwrap();
        let mut pass = RenderPassDescriptor::new();
        for (i, (target, format)) in targets.iter().zip(formats).enumerate() {
            let color = &mut pass.color_attachments[i];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = match format {
                PixelFormat::RGBA32Uint => ClearColor::new(0.0, 0.0, 0.0, 0.0),
                _ => ClearColor::new(-1.0, -1.0, -1.0, 0.0),
            };
            color.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(pipeline).unwrap();
        e.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
        e.draw_primitives(primitive, 0, positions.len() as u32, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let executions = command_buffer.executions();
        let ExecutedPipeline::Render(executed) = executions[0].pipeline() else {
            panic!("the draw executed no render pipeline");
        };
        let stages = executed
            .stages
            .iter()
            .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
            .count();
        assert_eq!(
            stages,
            geometry,
            "the geometry stages of the pipeline the draw executed on {}",
            self.device.name()
        );
        targets
            .iter()
            .map(|target| {
                let mut out = vec![0u8; (size * size * 16) as usize];
                target
                    .get_bytes(
                        &mut out,
                        size * 16,
                        size * size * 16,
                        Region::new_2d(0, 0, size, size),
                        0,
                        0,
                    )
                    .unwrap();
                out.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|t| {
                        let (words, _) = t.as_chunks::<4>();
                        std::array::from_fn(|i| u32::from_le_bytes(words[i]))
                    })
                    .collect()
            })
            .collect()
    }

    /// [`draw`](Self::draw) of `pipeline`, of a weights function, into one `RGBA32Float`
    /// attachment.
    fn draw_weights(
        &self,
        pipeline: &RenderPipelineState,
        primitive: PrimitiveType,
        positions: &[[f32; 4]],
        size: u64,
    ) -> Vec<[u32; 4]> {
        let formats = [PixelFormat::RGBA32Float];
        let geometry = self.weight_stages();
        let mut out = self.draw(pipeline, &formats, geometry, primitive, positions, size);
        out.remove(0)
    }
}

/// Whether `texel` is a pixel the draw covered: its alpha is the functions' 1, not the clear's 0.
fn covered(texel: &[u32; 4]) -> bool {
    texel[3] == 1.0f32.to_bits()
}

/// Metal's third weight of first weights `x` and `y`: `1 - (x + y)` in `float`.
fn third(x: u32, y: u32) -> u32 {
    (1.0f32 - (f32::from_bits(x) + f32::from_bits(y))).to_bits()
}

/// The fullscreen triangle's weights at pixel `(c, r)` of 4x4, as the M3 Max answered them:
/// `((8 - 2c + 2r) / 16, (2c + 1) / 16, (7 - 2r) / 16)`, every texel covered.
fn exact_triangle(c: u32, r: u32) -> [f32; 3] {
    let sixteenths = |n: u32| n as f32 / 16.0;
    [
        sixteenths(8 - 2 * c + 2 * r),
        sixteenths(2 * c + 1),
        sixteenths(7 - 2 * r),
    ]
}

/// The line's weights at pixel `c` of row 3 of 8x8, as the M3 Max answered them (the line covered
/// row 3 alone): `((15 - 2c) / 16, (2c + 1) / 16, 0)`.
fn exact_line(c: u32) -> [f32; 3] {
    [(15 - 2 * c) as f32 / 16.0, (2 * c + 1) as f32 / 16.0, 0.0]
}

/// The refusal of a device with neither way to read the weights: `fragmentShaderBarycentric`'s
/// builtins, or varyings a geometry stage writes.
fn needs_either_way() -> Needs {
    Needs::AnyOf(vec![
        vec![Capability::Feature(VkFeature::FragmentShaderBarycentric)],
        vec![Capability::Feature(VkFeature::GeometryShader)],
    ])
}

/// Each fragment function's pipeline on the lane's device, or `None` where the device refuses
/// it, having neither `fragmentShaderBarycentric` nor `geometryShader` (printed).
fn pipelines(lane: &Lane) -> Option<Vec<(&'static str, RenderPipelineState)>> {
    let name = lane.device.name().to_owned();
    let mut out = Vec::new();
    for fragment in FRAGMENTS {
        match lane.pipeline("bw_vertex", fragment, &[PixelFormat::RGBA32Float]) {
            Ok(pipeline) => out.push((fragment, pipeline)),
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    (&u.feature, &u.needs),
                    (&Feature::Barycentrics, &needs_either_way()),
                    "{fragment} on {name}"
                );
                assert!(
                    !lane.enables(VkFeature::FragmentShaderBarycentric)
                        && !lane.enables(VkFeature::GeometryShader),
                    "{fragment} on {name}: refused with a way to read the weights: {u:?}"
                );
                eprintln!("refused: {fragment} on {name}: {u:?}");
                return None;
            }
            Err(other) => panic!("{fragment} on {name}: {other}"),
        }
    }
    Some(out)
}

/// `check` on a lane device made by `adjust`; the validation layer must report nothing.
fn on_lane(adjust: impl FnOnce(&mut DeviceOptions), check: impl FnOnce(&Lane)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::with(&messages, &cache, adjust);
    check(&lane);
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// `check` on the lane device of `path`.
fn on_path(path: Path, check: impl FnOnce(&Lane)) {
    on_lane(
        |options| {
            if path == Path::Withheld {
                options
                    .withheld_features
                    .insert(VkFeature::FragmentShaderBarycentric);
            }
        },
        check,
    )
}

/// The third weight is `1 - (x + y)` of the first two, bit for bit, at every pixel of triangles
/// and lines, perspective-correct or not.
fn third_weight_is_metal_s(lane: &Lane) {
    let name = lane.device.name().to_owned();
    let Some(pipelines) = pipelines(lane) else {
        return;
    };
    for (fragment, pipeline) in &pipelines {
        for d in &THIRD_IS_ONE_MINUS_THE_FIRST_TWO {
            let texels = lane.draw_weights(pipeline, d.primitive, d.positions, d.size);
            let pixels: Vec<&[u32; 4]> = texels.iter().filter(|t| covered(t)).collect();
            assert!(
                pixels.len() >= d.covers_at_least,
                "{fragment}, {}, on {name}: {} pixels covered",
                d.what,
                pixels.len()
            );
            for t in &pixels {
                assert_eq!(
                    t[2],
                    third(t[0], t[1]),
                    "{fragment}, {}, on {name}: the weights {:?}",
                    d.what,
                    t[..3]
                        .iter()
                        .map(|w| f32::from_bits(*w))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    eprintln!(
        "the third barycentric weight on {name}, {} geometry stages: Metal's",
        lane.weight_stages()
    );
}

/// At points where every weight is a short binary fraction, the weights are the M3 Max's exactly:
/// the fullscreen triangle over 4x4 pixels, and the line through row 3's centers of 8x8.
fn weights_at_exact_points_are_metal_s(lane: &Lane) {
    let name = lane.device.name().to_owned();
    let Some(pipelines) = pipelines(lane) else {
        return;
    };
    let bits = |w: [f32; 3]| w.map(f32::to_bits);
    for (fragment, pipeline) in &pipelines {
        let triangle = [
            [-1.0, -1.0, 0.0, 1.0],
            [3.0, -1.0, 0.0, 1.0],
            [-1.0, 3.0, 0.0, 1.0],
        ];
        let texels = lane.draw_weights(pipeline, PrimitiveType::Triangle, &triangle, 4);
        for (i, t) in texels.iter().enumerate() {
            let (c, r) = (i as u32 % 4, i as u32 / 4);
            assert!(
                covered(t),
                "{fragment}, triangle, on {name}: ({c}, {r}) uncovered"
            );
            assert_eq!(
                t[..3],
                bits(exact_triangle(c, r)),
                "{fragment}, triangle, on {name}: ({c}, {r})"
            );
        }
        let line = [[-1.0, 0.125, 0.0, 1.0], [1.0, 0.125, 0.0, 1.0]];
        let texels = lane.draw_weights(pipeline, PrimitiveType::Line, &line, 8);
        let mut drawn = 0;
        for (i, t) in texels.iter().enumerate() {
            let (c, r) = (i as u32 % 8, i as u32 / 8);
            if !covered(t) {
                continue;
            }
            assert_eq!(r, 3, "{fragment}, line, on {name}: ({c}, {r}) covered");
            assert_eq!(
                t[..3],
                bits(exact_line(c)),
                "{fragment}, line, on {name}: ({c}, {r})"
            );
            drawn += 1;
        }
        assert!(
            drawn >= 6,
            "{fragment}, line, on {name}: {drawn} pixels covered"
        );
    }
    eprintln!(
        "barycentric weights at exact points on {name}, {} geometry stages: Metal's",
        lane.weight_stages()
    );
}

/// The third weight is `1 - (x + y)` of the first two, bit for bit, at every pixel of triangles
/// and lines, perspective-correct or not: Metal's rule, not the device's own third lane.
#[test]
fn the_third_weight_is_one_minus_the_first_two() {
    on_path(Path::Device, third_weight_is_metal_s);
}

/// At points where every weight is a short binary fraction, the weights are the M3 Max's exactly.
#[test]
fn the_weights_at_exact_points_are_metal_s() {
    on_path(Path::Device, weights_at_exact_points_are_metal_s);
}

/// [`the_third_weight_is_one_minus_the_first_two`] with `fragmentShaderBarycentric` withheld: the
/// weights are varyings a geometry stage writes.
#[test]
fn through_a_geometry_stage_the_third_weight_is_one_minus_the_first_two() {
    on_path(Path::Withheld, third_weight_is_metal_s);
}

/// [`the_weights_at_exact_points_are_metal_s`] with `fragmentShaderBarycentric` withheld.
#[test]
fn through_a_geometry_stage_the_weights_at_exact_points_are_metal_s() {
    on_path(Path::Withheld, weights_at_exact_points_are_metal_s);
}

/// With `fragmentShaderBarycentric` withheld, the geometry stage carries the fragment function's
/// varyings and primitive ID unchanged: at every pixel of [`CARRIED`]'s draws, `bv_weights` covers
/// what `bv_plain` (no geometry stage) covers and reads the same words of the smooth `float4`, the
/// flat `uint`, the `float2` and the primitive ID, and its third weight is `1 - (x + y)`. Skipped,
/// visibly, on a device without `geometryShader`, which refuses `bv_weights`
/// ([`pipelines`]' check).
#[test]
fn a_geometry_stage_carries_the_varyings_and_the_primitive_id() {
    on_path(Path::Withheld, |lane| {
        let name = lane.device.name().to_owned();
        if !lane.enables(VkFeature::GeometryShader) {
            println!("SKIPPED {name}: no geometryShader");
            return;
        }
        let formats = [
            PixelFormat::RGBA32Float,
            PixelFormat::RGBA32Float,
            PixelFormat::RGBA32Uint,
        ];
        let weights = lane.pipeline("bv_vertex", "bv_weights", &formats).unwrap();
        let plain = lane.pipeline("bv_vertex", "bv_plain", &formats).unwrap();
        for d in &CARRIED {
            let draw = |pipeline, geometry| {
                lane.draw(
                    pipeline,
                    &formats,
                    geometry,
                    d.primitive,
                    d.positions,
                    d.size,
                )
            };
            let read = draw(&weights, CLASSES);
            let expected = draw(&plain, 0);
            let tags: Vec<u32> = (0..d.positions.len() as u32).map(|v| v * 7 + 3).collect();
            let mut pixels = 0;
            for (i, (w, p)) in read[0].iter().zip(&expected[0]).enumerate() {
                let at = (i as u64 % d.size, i as u64 / d.size);
                assert_eq!(covered(w), covered(p), "{}, on {name}: {at:?}", d.what);
                if !covered(w) {
                    continue;
                }
                pixels += 1;
                assert_eq!(w[2], third(w[0], w[1]), "{}, on {name}: {at:?}", d.what);
                assert_eq!(
                    (read[1][i], read[2][i]),
                    (expected[1][i], expected[2][i]),
                    "{}, on {name}: {at:?}: the varyings and primitive ID",
                    d.what
                );
                assert!(
                    tags.contains(&read[2][i][0]),
                    "{}, on {name}: {at:?}: flat {}",
                    d.what,
                    read[2][i][0]
                );
            }
            assert!(
                pixels >= d.covers_at_least,
                "{}, on {name}: {pixels} pixels covered",
                d.what
            );
        }
        eprintln!("varyings through the barycentric geometry stage on {name}: unchanged");
    });
}

/// The kind a render pipeline state whose draws run a synthesized geometry stage announces
/// ([`a_pipeline_with_a_geometry_stage_announces_it_and_one_without_does_not`]).
pub(super) const SYNTHESIZED_GEOMETRY: [Degradation; 1] = [Degradation::SynthesizedGeometryStage];

/// A render pipeline state whose draws run a synthesized geometry stage announces it, once per
/// device, naming the barycentric weights its fragment function reads: `bv_weights`'s where the
/// device reads the weights as varyings (with `fragmentShaderBarycentric` withheld, or a device
/// without it), the second state of the same functions announcing nothing again. `bv_plain`'s,
/// which reads no weights and carries no geometry stage, announces nothing on either path, and
/// `bv_weights`'s announces nothing where the device reads the builtins. Skipped, visibly, where
/// the device has neither way (`pipelines`' check).
#[test]
fn a_pipeline_with_a_geometry_stage_announces_it_and_one_without_does_not() {
    for path in [Path::Device, Path::Withheld] {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let lane = Lane::with(&messages, &cache, |options| {
            if path == Path::Withheld {
                options
                    .withheld_features
                    .insert(VkFeature::FragmentShaderBarycentric);
            }
        });
        let name = lane.device.name().to_owned();
        let staged = |messages: &Messages| {
            let notices = messages.take_notices();
            notices
                .into_iter()
                .filter_map(|m| m.degradation)
                .filter(|n| SYNTHESIZED_GEOMETRY.contains(&n.degradation))
                .collect::<Vec<_>>()
        };
        let formats = [
            PixelFormat::RGBA32Float,
            PixelFormat::RGBA32Float,
            PixelFormat::RGBA32Uint,
        ];
        let geometry = lane.weight_stages() == CLASSES;
        if geometry && !lane.enables(VkFeature::GeometryShader) {
            println!("SKIPPED {name}, {path:?}: no geometryShader");
            continue;
        }
        lane.pipeline("bv_vertex", "bv_plain", &formats).unwrap();
        assert_eq!(
            staged(&messages),
            Vec::new(),
            "bv_plain on {name}, {path:?}"
        );
        lane.pipeline("bv_vertex", "bv_weights", &formats).unwrap();
        let announced = staged(&messages);
        if geometry {
            assert_eq!(
                announced.len(),
                1,
                "bv_weights on {name}, {path:?}: {announced:?}"
            );
            let notice = &announced[0];
            assert!(
                notice.reason.contains("[[barycentric_coord]]")
                    && !notice.reason.contains("slice")
                    && !notice.reason.contains("viewport"),
                "bv_weights on {name}, {path:?}: {notice:?}"
            );
            assert!(
                notice.object.contains("bv_vertex") && notice.object.contains("bv_weights"),
                "bv_weights on {name}, {path:?}: {notice:?}"
            );
        } else {
            assert_eq!(announced, Vec::new(), "bv_weights on {name}, {path:?}");
        }
        lane.pipeline("bv_vertex", "bv_weights", &formats).unwrap();
        assert_eq!(
            staged(&messages),
            Vec::new(),
            "a second bv_weights on {name}, {path:?}"
        );
        eprintln!(
            "SYNTHESIZED-GEOMETRY {name}, {path:?}: {}",
            if geometry {
                "announced once"
            } else {
                "no stage, nothing announced"
            }
        );
        drop(lane);
        layer_reported_nothing_at_all(&messages);
    }
}

/// With both ways to read the weights withheld (`fragmentShaderBarycentric` and `geometryShader`,
/// which the Vulkan 1.2 floor lacks too), the weight functions are refused as needing either
/// ([`pipelines`]' check), on every device.
#[test]
fn with_neither_way_the_weight_functions_are_refused_needing_either() {
    on_lane(
        |options| {
            options.withheld_features.extend([
                VkFeature::FragmentShaderBarycentric,
                VkFeature::GeometryShader,
            ]);
        },
        |lane| {
            assert!(
                pipelines(lane).is_none(),
                "the weight functions were made on {} with neither way to read the weights",
                lane.device.name()
            );
        },
    );
}
