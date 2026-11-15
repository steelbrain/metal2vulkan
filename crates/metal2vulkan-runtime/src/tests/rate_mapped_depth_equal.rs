//! A rate-mapped pass's depth, drawn by one route, tested `Equal` by a later pass drawing the same
//! triangles by another (`rate_mapped_pass.rs`, **Fan-out**): the encoder's draws are drawn region
//! by region, and an execution's commands a host step records are fanned out, each copy's window
//! coordinates computed in the geometry stage, the triangles the probe flags region by region.
//! Under `RateMapRegionPolicy::FanOut` on a device that fans out, a draw drawn region by region
//! takes its region-copy variant, whose vertex function puts each vertex in its region as the
//! fan-out's stage puts a copy's (`render/fan_out.rs`, **Region-copy variant**); under `Exact` each
//! region through its own viewport. A
//! depth prepass and an `Equal` pass over the same vertices are Metal's way to shade each pixel
//! once; Metal rasterizes both passes alike, so the second shades every pixel the first covered.
//!
//! The first pass clears the colour to black and the depth to 1 and draws a ground plane in
//! perspective (64 by 64 quads, every `w` positive) red, `Less`, writing depth; the second loads
//! both and draws the same vertices white, `Equal`, writing none. Each pass draws the ground in two
//! halves and, between them, one triangle with a `w` negative beside two positive ones, in front of
//! the ground, which the probe flags. On a device that routes a fanning pass's draws on the GPU
//! (`render/gpu_routes.rs`), the GPU routes each of the encoder's draws as the host step's probe
//! decides an execution of the same draws: fanned out where the probe flags none of its triangles,
//! else through the region-copy variant (the triangle, and a half one of whose triangles touches
//! more regions than the fan-out's copies). A pixel left red is one the first
//! pass covered and the second's depth test failed at: a hole of black in Metal's terms, where the
//! second pass is the lit surface. Each leg draws the first pass by one route and the second by
//! another, under `RateMapRegionPolicy::FanOut` and under `Exact`.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `cfb_depth_vertex` and
//! `cfb_depth_white` of the owned fixture `validation/fixtures/public/engine/cfb_sample_depth.ll`,
//! which draws each vertex where its buffer holds it, and `tob_solid` of `engine/tob_renders.ll`.

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::write_routes::GpuWriter;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, DepthStencilState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions,
    RasterizationRateMap, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The ground's quads on a side.
pub(super) const QUADS: u32 = 64;
const INDICES: u32 = QUADS * QUADS * 6;
/// The map's screen side, and its quality samples on both axes.
const SCREEN: u32 = 512;
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];
/// The passes: the first, then the second.
const FIRST: u64 = 0;
const SECOND: u64 = 1;
/// The ground's corners ([`ground_corners`]), after which the flagged triangle's vertices lie.
const CORNERS: u32 = (QUADS + 1) * (QUADS + 1);
/// Each pass's draws, in order, the first index and the count of each: the ground's first half,
/// the flagged triangle ([`flagged_triangle`]) and the ground's second half. The indirect command
/// buffer holds them for each pass, the first pass's first.
const DRAWS: [(u32, u32); 3] = [(0, INDICES / 2), (INDICES, 3), (INDICES / 2, INDICES / 2)];

/// How a pass draws the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// On the encoder: region by region.
    Encoder,
    /// Executed from the indirect command buffer at the host step: fanned out under `FanOut`.
    Executed,
}

struct Ground {
    device: Device,
    queue: CommandQueue,
    red: RenderPipelineState,
    white: RenderPipelineState,
    less: DepthStencilState,
    equal: DepthStencilState,
    positions: Buffer,
    indices: Buffer,
    color: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
}

/// The ground plane's corners in clip space, four floats each: a camera 1.7 above it looking 12
/// degrees down, a 60-degree field of view, depth from 0.1 to 200 in Metal's `[0, 1]`.
pub(super) fn ground_corners() -> Vec<[f32; 4]> {
    let pitch = 12f64.to_radians();
    let (sin, cos) = pitch.sin_cos();
    let f = 1.0 / 30f64.to_radians().tan();
    let (near, far) = (0.1f64, 200.0f64);
    let mut corners = Vec::new();
    for j in 0..=QUADS {
        for i in 0..=QUADS {
            let x = -30.0 + 60.0 * f64::from(i) / f64::from(QUADS);
            let z = 2.0 + 78.0 * f64::from(j) / f64::from(QUADS);
            let (dx, dy, dz) = (x, -1.7, z);
            let vx = dx;
            let vy = dy * cos + dz * sin;
            let vz = -dy * sin + dz * cos;
            let clip_z = vz * far / (far - near) - near * far / (far - near);
            corners.push([f * vx, f * vy, clip_z, vz].map(|v| v as f32));
        }
    }
    corners
}

/// A triangle in front of the ground in clip space, four floats a vertex: two vertices near the
/// bottom of the view with `w` 1, and a third with `w` −1, so the probe flags it; its part where
/// `w` is at least its `z` widens up a strip in the middle of the view.
fn flagged_triangle() -> [[f32; 4]; 3] {
    [
        [-0.3, -0.9, 0.2, 1.0],
        [0.3, -0.9, 0.2, 1.0],
        [0.0, 3.0, 0.2, -1.0],
    ]
}

impl Ground {
    fn new(device: Device) -> Ground {
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
        let pipeline = |fragment: StageSource| {
            let mut descriptor =
                RenderPipelineDescriptor::new(function(&depth, "cfb_depth_vertex"));
            descriptor.fragment = Some(fragment);
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.depth_format = Some(PixelFormat::Depth32Float);
            descriptor.support_indirect_command_buffers = true;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let red = pipeline(function(&solid, "tob_solid"));
        let white = pipeline(function(&depth, "cfb_depth_white"));
        let state = |compare: CompareFunction, write: bool| {
            let mut descriptor = DepthStencilDescriptor::default();
            descriptor.depth_compare_function = compare;
            descriptor.depth_write_enabled = write;
            device.new_depth_stencil_state(&descriptor).unwrap()
        };
        let (less, equal) = (
            state(CompareFunction::Less, true),
            state(CompareFunction::Equal, false),
        );
        let positions: Vec<u8> = ground_corners()
            .into_iter()
            .chain(flagged_triangle())
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        let corner = |x: u32, y: u32| y * (QUADS + 1) + x;
        let indices: Vec<u8> = (0..QUADS * QUADS)
            .flat_map(|quad| {
                let (x, y) = (quad % QUADS, quad / QUADS);
                let [a, b, c, d] = [
                    corner(x, y),
                    corner(x + 1, y),
                    corner(x, y + 1),
                    corner(x + 1, y + 1),
                ];
                [a, b, c, b, d, c]
            })
            .chain(CORNERS..CORNERS + 3)
            .flat_map(u32::to_le_bytes)
            .collect();
        let color: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let buffer = |bytes: &[u8]| {
            (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
        };
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW_INDEXED;
        icb.inherit_buffers = false;
        icb.max_vertex_buffer_bind_count = 1;
        icb.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&icb, 2 * DRAWS.len() as u64, ResourceOptions::shared())
            .unwrap();
        let mut map = RateMapDescriptor::new([SCREEN, SCREEN]);
        map.layers = vec![RateMapLayerDescriptor::new(
            QUALITY.to_vec(),
            QUALITY.to_vec(),
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        let ground = Ground {
            queue: device.new_command_queue().unwrap(),
            red,
            white,
            less,
            equal,
            positions: buffer(&positions),
            indices: buffer(&indices),
            color: buffer(&color),
            icb,
            map,
            device,
        };
        for (pass, pipeline) in [(FIRST, &ground.red), (SECOND, &ground.white)] {
            for (k, &(first, count)) in DRAWS.iter().enumerate() {
                let at = pass * DRAWS.len() as u64 + k as u64;
                let command = ground.icb.indirect_render_command_at(at).unwrap();
                command.set_render_pipeline_state(pipeline).unwrap();
                command
                    .set_vertex_buffer(Some(&ground.positions), 0, 0)
                    .unwrap();
                command
                    .set_fragment_buffer(Some(&ground.color), 0, 0)
                    .unwrap();
                command
                    .draw_indexed_primitives(
                        PrimitiveType::Triangle,
                        u64::from(count),
                        IndexType::UInt32,
                        &ground.indices,
                        u64::from(first) * 4,
                        1,
                        0,
                        0,
                    )
                    .unwrap();
            }
        }
        // The GPU may write the records: the host step records the passes.
        ground.icb.written_by_gpu(GpuWriter::Reset);
        drop(ground.icb.gpu_write());
        ground
    }

    /// The two passes, the first drawn by `first` and the second by `second`, in one command
    /// buffer: the target's red, white and black pixels, the command buffer's GPU time in
    /// nanoseconds (0 where the device times none), and the device's counts before and after.
    fn frame(&self, first: Route, second: Route) -> ([usize; 3], u64, WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let texture = |format, options| {
            let mut d = TextureDescriptor::texture_2d(format, size.width, size.height, false);
            d.usage = TextureUsage::RENDER_TARGET;
            d.resource_options = options;
            self.device.new_texture(&d).unwrap()
        };
        let target = texture(PixelFormat::RGBA8Unorm, ResourceOptions::shared());
        let depth = texture(PixelFormat::Depth32Float, ResourceOptions::private());
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        for (pass, route) in [(FIRST, first), (SECOND, second)] {
            let mut descriptor = RenderPassDescriptor::new();
            let color = &mut descriptor.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = match pass {
                FIRST => LoadAction::Clear,
                _ => LoadAction::Load,
            };
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
            color.store_action = StoreAction::Store;
            descriptor.depth_attachment.texture = Some(depth.clone());
            descriptor.depth_attachment.load_action = color.load_action;
            descriptor.depth_attachment.clear_depth = 1.0;
            descriptor.depth_attachment.store_action = StoreAction::Store;
            descriptor.rasterization_rate_map = Some(self.map.clone());
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            let (pipeline, state) = match pass {
                FIRST => (&self.red, &self.less),
                _ => (&self.white, &self.equal),
            };
            encoder.set_depth_stencil_state(state).unwrap();
            encoder.set_render_pipeline_state(pipeline).unwrap();
            match route {
                Route::Executed => {
                    let first = pass * DRAWS.len() as u64;
                    let commands = first..first + DRAWS.len() as u64;
                    encoder
                        .execute_commands_in_buffer(&self.icb, commands)
                        .unwrap()
                }
                Route::Encoder => {
                    encoder
                        .set_vertex_buffer(Some(&self.positions), 0, 0)
                        .unwrap();
                    encoder
                        .set_fragment_buffer(Some(&self.color), 0, 0)
                        .unwrap();
                    for (first, count) in DRAWS {
                        encoder
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
                            .unwrap();
                    }
                }
            }
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let gpu = match (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        ) {
            (Ok(start), Ok(end)) => ((end - start) * 1e9) as u64,
            _ => 0,
        };
        drop(command_buffer);
        let after = self.device.work_counts();
        let (width, height) = (size.width, size.height);
        let mut bytes = vec![0u8; (width * height * 4) as usize];
        target
            .get_bytes(
                &mut bytes,
                width * 4,
                0,
                Region::new_2d(0, 0, width, height),
                0,
                0,
            )
            .unwrap();
        let mut counts = [0usize; 3];
        for texel in bytes.chunks(4) {
            match texel[..3] {
                [255, 0, 0] => counts[0] += 1,
                [255, 255, 255] => counts[1] += 1,
                _ => counts[2] += 1,
            }
        }
        (counts, gpu, before, after)
    }
}

/// The device the cfg selects under `policy`, its shader cache in `cache`.
fn device_under(cache: &Path, policy: RateMapRegionPolicy) -> Device {
    let mut options = options(cache);
    options.policy.rate_map_regions = policy;
    Device::system_default_with(&options).unwrap()
}

/// Whether `device` enables the features the fan-out and its probe take.
fn fans_out(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let features = &profile.enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// Whether `device` routes a fanning pass's draws on the GPU (`render/gpu_routes.rs`): it fans out,
/// it draws region copies, and it enables the indirect-count draws and many draws an indirect
/// draw.
fn routes_on_gpu(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let features = &profile.enabled().features;
    fans_out(device)
        && [
            VkFeature::ShaderOutputViewportIndex,
            VkFeature::ShaderDrawParameters,
            VkFeature::DrawIndirectCount,
            VkFeature::MultiDrawIndirect,
        ]
        .iter()
        .all(|feature| features.contains(feature))
}

/// The draws the GPU routed, and those of them it fanned out and drew through the region-copy
/// variant, between two counts.
fn gpu_routed(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.rate_map_gpu_routed_draws - before.rate_map_gpu_routed_draws,
        after.rate_map_gpu_fanned_draws - before.rate_map_gpu_fanned_draws,
        after.rate_map_gpu_copied_draws - before.rate_map_gpu_copied_draws,
    ]
}

/// The fanned-out draws, the flagged ones, the ones in parts, and the draws through the regions
/// variant and their recordings between two counts.
fn fanned(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.rate_map_fanned_draws - before.rate_map_fanned_draws,
        after.rate_map_flagged_draws - before.rate_map_flagged_draws,
        after.rate_map_parted_draws - before.rate_map_parted_draws,
        after.rate_map_region_copy_draws - before.rate_map_region_copy_draws,
        after.rate_map_region_copy_recordings - before.rate_map_region_copy_recordings,
    ]
}

/// Each pairing of routes, under each policy, and the encoder's again once its pipelines are
/// made: the second pass shades every pixel the first covered, so no pixel is left red. Under
/// `FanOut` on a device that fans out, the encoder's draws are drawn through their regions
/// variant; under `Exact` none is. On a device that routes them on the GPU, every draw it routed
/// took one route, and once the pipelines are made it routes each of the encoder's draws, fanning
/// out as many as the host step fans out whole of the executed commands and drawing the ones it
/// flags through the region-copy variant, some of each; under `Exact` it routes none.
#[test]
fn an_equal_depth_pass_shades_every_pixel_its_prepass_covered_whatever_the_routes() {
    // The last pairing again, its pipelines made: the warm frame of the encoder's routes.
    let routes = [
        (Route::Encoder, Route::Encoder),
        (Route::Encoder, Route::Executed),
        (Route::Executed, Route::Encoder),
        (Route::Executed, Route::Executed),
        (Route::Encoder, Route::Encoder),
    ];
    let mut holes = Vec::new();
    for policy in [RateMapRegionPolicy::Exact, RateMapRegionPolicy::FanOut] {
        let scratch = Scratch::new("rate-mapped-depth-equal");
        let ground = Ground::new(device_under(&scratch.0, policy));
        let routing = policy == RateMapRegionPolicy::FanOut && routes_on_gpu(&ground.device);
        // The host step's draws fanned out whole and flagged, of both passes executed.
        let mut host = None;
        for (frame, (first, second)) in routes.into_iter().enumerate() {
            let ([red, white, black], gpu, before, after) = ground.frame(first, second);
            let counts = fanned(&before, &after);
            let [routed, gpu_fanned, gpu_copied] = gpu_routed(&before, &after);
            let made = after.variant_pipelines_made - before.variant_pipelines_made;
            let create = after.render_pipeline_create_nanos - before.render_pipeline_create_nanos;
            eprintln!(
                "RATE-MAP-DEPTH-EQUAL {policy:?} {first:?} then {second:?}: {red} red, {white} \
                 white, {black} black; fanned, flagged, parted, region copy draws, recordings \
                 {counts:?}; {routed} draws routed on the GPU, {gpu_fanned} fanned out, \
                 {gpu_copied} through the region-copy variant; {made} pipelines made in {create} \
                 ns; GPU {gpu} ns"
            );
            assert!(white > 0, "the second pass shaded nothing");
            let encoded = [first, second]
                .iter()
                .filter(|&&r| r == Route::Encoder)
                .count();
            match policy == RateMapRegionPolicy::FanOut && fans_out(&ground.device) {
                true => assert!(
                    counts[3] >= encoded as u64,
                    "the encoder's draws through the region-copy variant"
                ),
                false => assert_eq!(counts[3], 0, "draws through the region-copy variant"),
            }
            assert_eq!(
                gpu_fanned + gpu_copied,
                routed,
                "every draw the GPU routed took one route"
            );
            if (first, second) == (Route::Executed, Route::Executed) {
                host = Some([counts[0], counts[1]]);
            }
            let encoded = encoded as u64 * DRAWS.len() as u64;
            match (routing, frame + 1 == routes.len()) {
                (true, true) => {
                    let [fanned, flagged] = host.expect("the executed passes ran first");
                    assert_eq!(
                        [routed, gpu_fanned, gpu_copied],
                        [encoded, fanned, flagged],
                        "the encoder's draws routed as the host step's probe decides them"
                    );
                    assert!(
                        gpu_fanned > 0 && gpu_copied > 0,
                        "both routes taken: {gpu_fanned} fanned out, {gpu_copied} copied"
                    );
                }
                (true, false) => assert!(routed <= encoded, "routed draws past the encoder's"),
                (false, _) => assert_eq!(routed, 0, "draws routed on the GPU"),
            }
            if red != 0 {
                holes.push((policy, first, second, red));
            }
        }
    }
    assert!(
        holes.is_empty(),
        "pixels the first pass covered and the equal pass did not: {holes:?}"
    );
}
