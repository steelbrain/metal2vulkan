//! A rate-mapped pass's draws through their region-copy variant, row by row against region by
//! region (`render/fan_out.rs`, **Region-copy variant**): a direct draw records each row of the
//! table's regions as many at a time as a multi-draw takes, under the row's scissor, the borders
//! between its regions resting on each copy's clip distances alone; an indirect draw records each
//! region alone, under that region's scissor, through the same variant and the same vertex
//! function. Each pixel of the pass belongs to one region, so the two draw the same texels where
//! no border between a recording's regions holds a sample: a sample covered by two copies would
//! be blended twice, and one covered by none left out.
//!
//! The pass clears an `RGBA16Float` target to zero and the depth to 1 and blends every draw
//! additively, each fragment adding one to a channel: the ground plane of
//! `rate_mapped_depth_equal` (64 by 64 quads in perspective) adds to red, `Less`, writing depth;
//! then thin triangles across the screen at many angles and wide ones whose vertices lie far
//! outside it, at several `w`, one of them crossing `w = 0`, add to green, `Always`, writing no
//! depth; then a second pass loads both and draws the ground again, `Equal`, adding to blue. Every
//! texel's channels count the fragments each class shaded there: the frame drawn directly and
//! the frame drawn indirectly are compared bit for bit, and in each the `Equal` pass shades each
//! texel the ground covered once.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `cfb_depth_vertex` of
//! the owned fixture `validation/fixtures/public/engine/cfb_sample_depth.ll`, which draws each
//! vertex where its buffer holds it, and `tob_solid` of `engine/tob_renders.ll`, which returns its
//! buffer's colour.

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use super::rate_mapped_depth_equal::{ground_corners, QUADS};
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, DepthStencilState, Device, Library,
    LoadAction, PipelineOptions, RasterizationRateMap, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, IndexType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The ground's indices.
const GROUND: u32 = QUADS * QUADS * 6;
/// The map's screen, wider than high, and its quality samples on both axes.
const SCREEN: [u32; 2] = [768, 640];
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];
/// The thin triangles across the screen.
const SLIVERS: u32 = 24;

/// How the pass's draws are recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// `drawIndexedPrimitives`: a box of regions a recording.
    Direct,
    /// `drawIndexedPrimitives` from an indirect buffer: a region a recording.
    Indirect,
}

/// A clip-space vertex at normalized device coordinates `x`, `y` and depth `z`, of `w`.
fn at([x, y, z]: [f32; 3], w: f32) -> [f32; 4] {
    [x * w, y * w, z * w, w]
}

/// The triangles that add to green, three clip-space vertices each: thin ones through the
/// screen's middle at `SLIVERS` angles, a pair whose long edges run a hair off horizontal and off
/// vertical, and wide ones whose vertices lie far outside the screen at several `w`, and one
/// crossing `w = 0`.
fn overlay() -> Vec<[f32; 4]> {
    let mut vertices = Vec::new();
    for i in 0..SLIVERS {
        let angle = std::f32::consts::PI * i as f32 / SLIVERS as f32;
        let (sin, cos) = angle.sin_cos();
        let (length, width) = (1.3, 0.015);
        let [a, b] = [1.0f32, -1.0].map(|s| [0.1 + s * length * cos, -0.05 + s * length * sin]);
        let c = [b[0] - width * sin, b[1] + width * cos];
        for [x, y] in [a, b, c] {
            vertices.push(at([x, y, 0.5], 1.0));
        }
    }
    vertices.extend([
        at([-1.4, 0.33, 0.4], 1.0),
        at([1.4, 0.3302, 0.4], 2.0),
        at([1.4, 0.36, 0.4], 0.5),
        at([-0.21, -1.4, 0.4], 1.0),
        at([-0.2104, 1.4, 0.4], 3.0),
        at([-0.25, 1.4, 0.4], 1.0),
        at([-40.0, -30.0, 0.25], 1.0),
        at([35.0, -1.0, 0.25], 3.0),
        at([0.3, 45.0, 0.25], 0.5),
        at([-0.9, 0.95, 0.75], 1.0),
        at([25.0, 0.1, 0.75], 0.2),
        at([-0.2, -60.0, 0.75], 7.0),
        [0.5, 0.2, 0.1, 1.0],
        [-2.0, 0.3, -0.5, -1.0],
        [0.1, -0.9, 0.5, 1.0],
    ]);
    vertices
}

/// What a frame draws: the positions (the ground's corners, then the overlay's vertices), the
/// indices (the ground's, then the overlay's) and each draw's colour, depth state and indices.
struct Frame {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    less: DepthStencilState,
    always: DepthStencilState,
    equal: DepthStencilState,
    positions: Buffer,
    indices: Buffer,
    colors: [Buffer; 3],
    /// `MTLDrawIndexedPrimitivesIndirectArguments` of the three draws, 20 bytes each.
    arguments: Buffer,
    /// Each draw's first index and its indices.
    draws: [(u32, u32); 3],
    map: RasterizationRateMap,
}

impl Frame {
    fn new(device: Device) -> Frame {
        let (depth, solid) = (
            library(&device, "engine/cfb_sample_depth.ll"),
            library(&device, "engine/tob_renders.ll"),
        );
        let function = |library: &Library, name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(&depth, "cfb_depth_vertex"));
        descriptor.fragment = Some(function(&solid, "tob_solid"));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::One;
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let state = |compare: CompareFunction, write: bool| {
            let mut descriptor = DepthStencilDescriptor::default();
            descriptor.depth_compare_function = compare;
            descriptor.depth_write_enabled = write;
            device.new_depth_stencil_state(&descriptor).unwrap()
        };
        let corners = ground_corners();
        let overlay = overlay();
        let positions: Vec<u8> = (corners.iter().chain(&overlay))
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let corner = |x: u32, y: u32| y * (QUADS + 1) + x;
        let ground = (0..QUADS * QUADS).flat_map(|quad| {
            let (x, y) = (quad % QUADS, quad / QUADS);
            let [a, b, c, d] = [
                corner(x, y),
                corner(x + 1, y),
                corner(x, y + 1),
                corner(x + 1, y + 1),
            ];
            [a, b, c, b, d, c]
        });
        let base = corners.len() as u32;
        let indices: Vec<u8> = (ground.chain(base..base + overlay.len() as u32))
            .flat_map(u32::to_le_bytes)
            .collect();
        let draws = [(0, GROUND), (GROUND, overlay.len() as u32), (0, GROUND)];
        let arguments: Vec<u8> = (draws.iter())
            .flat_map(|&(first, count)| [count, 1, first, 0, 0])
            .flat_map(u32::to_le_bytes)
            .collect();
        let buffer = |bytes: &[u8]| {
            (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
        };
        let colors = [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        .map(|c| {
            let bytes: Vec<u8> = c.into_iter().flat_map(f32::to_le_bytes).collect();
            buffer(&bytes)
        });
        let mut map = RateMapDescriptor::new(SCREEN);
        map.layers = vec![RateMapLayerDescriptor::new(
            QUALITY.to_vec(),
            QUALITY.to_vec(),
        )];
        Frame {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            less: state(CompareFunction::Less, true),
            always: state(CompareFunction::Always, false),
            equal: state(CompareFunction::Equal, false),
            positions: buffer(&positions),
            indices: buffer(&indices),
            colors,
            arguments: buffer(&arguments),
            draws,
            map: device.new_rasterization_rate_map(&map).unwrap(),
            device,
        }
    }

    /// The two passes, each draw recorded by `route`: the target's texels, eight bytes each, and
    /// the device's counts before and after.
    fn frame(&self, route: Route) -> (Vec<u8>, WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let texture = |format, options| {
            let mut d = TextureDescriptor::texture_2d(format, size.width, size.height, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = options;
            self.device.new_texture(&d).unwrap()
        };
        let target = texture(PixelFormat::RGBA16Float, ResourceOptions::shared());
        let depth = texture(PixelFormat::Depth32Float, ResourceOptions::private());
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        // The first pass draws the ground and the overlay, the second the ground again.
        for (pass, draws) in [(0, 0..2), (1, 2..3)] {
            let mut descriptor = RenderPassDescriptor::new();
            let color = &mut descriptor.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = match pass {
                0 => LoadAction::Clear,
                _ => LoadAction::Load,
            };
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::Store;
            descriptor.depth_attachment.texture = Some(depth.clone());
            descriptor.depth_attachment.load_action = color.load_action;
            descriptor.depth_attachment.clear_depth = 1.0;
            descriptor.depth_attachment.store_action = StoreAction::Store;
            descriptor.rasterization_rate_map = Some(self.map.clone());
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            encoder.set_render_pipeline_state(&self.pipeline).unwrap();
            encoder
                .set_vertex_buffer(Some(&self.positions), 0, 0)
                .unwrap();
            for draw in draws {
                let state = [&self.less, &self.always, &self.equal][draw];
                encoder.set_depth_stencil_state(state).unwrap();
                encoder
                    .set_fragment_buffer(Some(&self.colors[draw]), 0, 0)
                    .unwrap();
                let (first, count) = self.draws[draw];
                match route {
                    Route::Direct => encoder
                        .draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            count,
                            IndexType::UInt32,
                            &self.indices,
                            u64::from(first) * 4,
                            1,
                            0,
                            0,
                        )
                        .unwrap(),
                    Route::Indirect => encoder
                        .draw_indexed_primitives_indirect(
                            PrimitiveType::Triangle,
                            IndexType::UInt32,
                            &self.indices,
                            0,
                            &self.arguments,
                            draw as u64 * 20,
                        )
                        .unwrap(),
                }
            }
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(command_buffer);
        let after = self.device.work_counts();
        let (width, height) = (size.width, size.height);
        let mut bytes = vec![0u8; (width * height * 8) as usize];
        target
            .get_bytes(
                &mut bytes,
                width * 8,
                0,
                Region::new_2d(0, 0, width, height),
                0,
                0,
            )
            .unwrap();
        (bytes, before, after)
    }
}

/// The value of half-precision bits `h`, a whole number of fragments.
fn half(h: u16) -> f32 {
    let (exponent, mantissa) = (i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
    match exponent {
        0 => mantissa * 2f32.powi(-24),
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Each texel's red, green and blue: the fragments of each class it took.
fn channels(bytes: &[u8]) -> Vec<[f32; 3]> {
    (bytes.chunks(8))
        .map(|t| [0, 2, 4].map(|c| half(u16::from_le_bytes([t[c], t[c + 1]]))))
        .collect()
}

/// The draws through the region-copy variant and their recordings between two counts.
fn copies(before: &WorkCounts, after: &WorkCounts) -> [u64; 2] {
    [
        after.rate_map_region_copy_draws - before.rate_map_region_copy_draws,
        after.rate_map_region_copy_recordings - before.rate_map_region_copy_recordings,
    ]
}

/// Drawn directly, each draw records the map's regions row by row, as many at a time as a
/// multi-draw takes, and indirectly one region a recording; the frames' texels are equal bit for
/// bit, every texel the ground covered took its `Equal` pass once, and the overlay's triangles
/// overlap.
#[test]
fn a_box_of_regions_draws_the_texels_its_regions_draw_one_at_a_time() {
    let scratch = Scratch::new("rate-mapped-region-boxes");
    let mut options = options(&scratch.0);
    options.policy.rate_map_regions = RateMapRegionPolicy::FanOut;
    let frame = Frame::new(Device::system_default_with(&options).unwrap());
    let compile = frame.device.compile_device();
    let features = &compile.profile().enabled().features;
    let copied = [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
        VkFeature::ShaderDrawParameters,
    ]
    .iter()
    .all(|feature| features.contains(feature));
    // Cold, then warm: the counts of the warm frames.
    frame.frame(Route::Direct);
    let (boxed, before, after) = frame.frame(Route::Direct);
    let direct = copies(&before, &after);
    let (alone, before, after) = frame.frame(Route::Indirect);
    let indirect = copies(&before, &after);
    let shared = frame.device.shared();
    let multi = shared.multi_draws.as_ref().map(|m| m.copies());
    let standard = shared.standard_sample_locations;
    // Texels whose `Equal` pass shaded otherwise than the ground covered them.
    let unequal = |bytes: &[u8]| (channels(bytes).iter()).filter(|[r, _, b]| r != b).count();
    let texels = channels(&boxed);
    let overlapped = (texels.iter()).filter(|[_, g, _]| *g >= 2.0).count();
    let ground = (texels.iter()).filter(|[r, _, _]| *r > 0.0).count();
    let unequal = [unequal(&boxed), unequal(&alone)];
    let differ = (boxed.chunks(8).zip(alone.chunks(8)))
        .filter(|(a, b)| a != b)
        .count();
    let first = (boxed.chunks(8).zip(alone.chunks(8)).enumerate())
        .find(|(_, (a, b))| a != b)
        .map(|(at, (a, b))| (at, a.to_vec(), b.to_vec()));
    eprintln!(
        "RATE-MAP-REGION-BOXES region-copy draws and recordings: direct {direct:?}, indirect \
         {indirect:?}; multi-draw {multi:?}, standard locations {standard}; {ground} texels of \
         the ground, {unequal:?} whose equal pass differs, {overlapped} overlapped; {differ} \
         texels differ, the first {first:?}"
    );
    assert!(ground > 0, "the ground drew nothing");
    assert!(
        overlapped > 0,
        "no texel took two of the overlay's triangles"
    );
    assert_eq!(
        unequal, [0; 2],
        "texels the equal pass shaded otherwise than the ground, direct and indirect"
    );
    assert_eq!(
        differ, 0,
        "texels a box draws otherwise than its regions alone"
    );
    match copied {
        true => {
            assert_eq!(
                [direct[0], indirect[0]],
                [3, 3],
                "draws through the variant"
            );
            let regions = indirect[1] / 3;
            assert_eq!(indirect[1], 3 * regions, "one recording a region");
            assert!(regions > 16, "the map's regions: {regions}");
            match multi {
                // Each draw a recording a row of the table: as many recordings as it has rows,
                // which split its regions into rows of several.
                Some(most) if standard && u64::from(most) >= regions => {
                    let rows = direct[1] / 3;
                    assert_eq!(direct[1], 3 * rows, "each draw's recordings alike");
                    assert!(
                        rows > 1 && regions.is_multiple_of(rows) && regions / rows > 1,
                        "{rows} recordings a draw of {regions} regions"
                    );
                }
                Some(_) if standard => {
                    assert!(direct[1] < indirect[1], "rows in fewer recordings")
                }
                _ => assert_eq!(direct[1], indirect[1], "one region a recording"),
            }
        }
        false => assert_eq!([direct, indirect], [[0; 2]; 2]),
    }
}
