//! The routes' kernel of a rate-mapped pass's draws the GPU routes, made ahead on the device's
//! pool (`render/gpu_routes.rs`, **The kernel, made ahead**): an instance whose encoder ends
//! before it is made makes none itself and runs no steps, and each of its routed draws draws its
//! region-copy route; once it is made, the same draws are fanned out, over the same pixels.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_triangle` and
//! `tob_solid` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`.

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use crate::pipeline::pool::held;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, LoadAction, PipelineOptions,
    RasterizationRateMap, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::time::{Duration, Instant};

/// The draws of a frame past the first: small triangles over a grid of this many a side.
const SIDE: u32 = 4;

struct Lane {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    positions: Buffer,
    indices: Buffer,
    color: Buffer,
    map: RasterizationRateMap,
    target: Texture,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let renders = library(&device, "engine/tob_renders.ll");
        let function = |name| {
            StageSource::Function(
                renders
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("tob_triangle"));
        descriptor.fragment = Some(function("tob_solid"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        // Each triangle a twentieth of the view wide: it touches few regions, so the probe
        // flags none.
        let positions: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|k| {
                let at = |i: u32| -0.8 + 1.6 * i as f32 / SIDE as f32;
                let (x, y) = (at(k % SIDE), at(k / SIDE));
                [[x, y], [x + 0.1, y], [x, y + 0.1]]
            })
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        let indices: Vec<u8> = (0..3 * SIDE * SIDE).flat_map(u32::to_le_bytes).collect();
        let color: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let buffer = |bytes: &[u8]| {
            (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
        };
        let quality = [0.3f32, 0.6, 1.0, 0.6, 0.3];
        let mut map = RateMapDescriptor::new([512, 512]);
        map.layers = vec![RateMapLayerDescriptor::new(
            quality.to_vec(),
            quality.to_vec(),
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        let size = map.physical_size_for_layer(0);
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Lane {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            positions: buffer(&positions),
            indices: buffer(&indices),
            color: buffer(&color),
            target: device.new_texture(&target).unwrap(),
            map,
            device,
        }
    }

    /// A frame of the first `draws` triangles, a draw each: the target's red pixels, and the
    /// device's counts before and after.
    fn frame(&self, draws: u32) -> (usize, WorkCounts, WorkCounts) {
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        pass.rasterization_rate_map = Some(self.map.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.positions), 0, 0)
            .unwrap();
        encoder
            .set_fragment_buffer(Some(&self.color), 0, 0)
            .unwrap();
        for k in 0..u64::from(draws) {
            encoder
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    3,
                    IndexType::UInt32,
                    &self.indices,
                    k * 12,
                    1,
                    0,
                    0,
                )
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(command_buffer);
        let after = self.device.work_counts();
        let (width, height) = (self.target.width(), self.target.height());
        let mut bytes = vec![0u8; (width * height * 4) as usize];
        (self.target)
            .get_bytes(
                &mut bytes,
                width * 4,
                0,
                Region::new_2d(0, 0, width, height),
                0,
                0,
            )
            .unwrap();
        let red = bytes.chunks(4).filter(|t| t[..3] == [255, 0, 0]).count();
        (red, before, after)
    }
}

/// Whether `device` routes a fanning pass's draws on the GPU (`render/gpu_routes.rs`).
fn routes_on_gpu(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let features = &profile.enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
        VkFeature::ShaderOutputViewportIndex,
        VkFeature::ShaderDrawParameters,
        VkFeature::DrawIndirectCount,
        VkFeature::MultiDrawIndirect,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// Wait until `done` holds, for ten seconds at most.
fn until(what: &str, done: impl Fn() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(10), "{what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The routes' counts between two counts: routed, fanned out, through the region-copy variant.
fn routes(before: &WorkCounts, after: &WorkCounts) -> [u64; 3] {
    [
        after.rate_map_gpu_routed_draws - before.rate_map_gpu_routed_draws,
        after.rate_map_gpu_fanned_draws - before.rate_map_gpu_fanned_draws,
        after.rate_map_gpu_copied_draws - before.rate_map_gpu_copied_draws,
    ]
}

/// The first frame draws once, its region-copy variant being made, so it begins the fan-out's
/// variants and no routes' kernel; with the pool held, the next frame's routed draws begin the
/// kernel, which is not made as their encoder ends: the encoder makes none, and the draws are
/// drawn by their region-copy route; with the pool let go and the kernel made, the same draws
/// are fanned out, over the same pixels.
#[test]
fn an_instance_ending_before_its_routes_kernel_is_made_draws_the_region_copy_route() {
    let scratch = Scratch::new("rate-mapped-route-kernel");
    let lane = Lane::new(Device::system_default_with(&options(&scratch.0)).unwrap());
    if !routes_on_gpu(&lane.device) {
        eprintln!("RATE-MAP-ROUTE-KERNEL: the device routes no draw on the GPU");
        return;
    }
    let draws = SIDE * SIDE;
    let (red, ..) = lane.frame(1);
    assert!(red > 0, "the first frame drew nothing");
    assert!(
        lane.device.route_counts_kernel().is_none(),
        "the first frame's draw, its region-copy variant being made, began the routes' kernel"
    );
    let pool = &lane.device.shared().pipeline_pool;
    until("the fan-out's variants still being made", || {
        pool.busy() == 0
    });
    let hold = held::hold();
    let (held_red, before, after) = lane.frame(draws);
    let made = after.owned_kernels_made - before.owned_kernels_made;
    let held_routes = routes(&before, &after);
    let copies = after.rate_map_region_copy_draws - before.rate_map_region_copy_draws;
    drop(hold);
    until("the routes' kernel never made", || {
        lane.device.route_counts_kernel().is_some()
    });
    let (made_red, before, after) = lane.frame(draws);
    let made_routes = routes(&before, &after);
    eprintln!(
        "RATE-MAP-ROUTE-KERNEL: held, {held_red} red, {made} kernels made, routed, fanned, \
         copied {held_routes:?}, {copies} region-copy draws; made, {made_red} red, {made_routes:?}"
    );
    assert_eq!(made, 0, "the encoder made the routes' kernel itself");
    assert_eq!(held_routes, [0, 0, 0], "draws routed without the kernel");
    assert_eq!(u64::from(draws), copies, "the draws' region-copy route");
    assert_eq!(
        made_routes,
        [u64::from(draws), u64::from(draws), 0],
        "the draws fanned out once the kernel is made"
    );
    assert_eq!(held_red, made_red, "the routes' pixels");
}
