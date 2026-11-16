//! A rate-mapped pass's executed commands at a host step under `RateMapRegionPolicy::FanOut`, whose
//! triangles lie in a few regions (`render/fan_out.rs`, **Boxed draws**): each such draw is drawn
//! through its region-copy variant into the boxes of the regions its triangles reach by the probe's
//! bounds, not fanned out, where those regions' draws hold no more work than its triangles'
//! geometry stage (`rate_mapped_pass::boxes_cheaper`). Four commands draw a grid of small
//! triangles, twice, over a small patch of the screen: one inside a square of 10 screen pixels, one
//! over a square of 25 by 30 across the corner where four regions meet, one of two instances, and
//! one without indices; three others are fanned out, whose region draws would cost more than their
//! geometry stage: a grid over the whole target, one quad, twice, over a square of 60 pixels, and a
//! strip of 120 by 40 across ten regions, its box 50 vertex-regions a triangle: more than the 24 a
//! triangle's geometry stage costs (`GEOMETRY_TRIANGLE_WORK`), as the boxes of the pieces lane's
//! patches frame held about 84, and took about 2.5 and 2.8 times their fan-out on an Iris Xe and a
//! Radeon 780M. Each command blends its own colour at half alpha over what the commands before it
//! drew, so a copy drawn twice, left out of its region, or drawn out of the commands' order changes
//! the pixels: the target's bytes are those of the same commands drawn on the encoder, every one
//! through its region-copy variant into every region.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_triangle` and
//! `tob_solid` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use super::rate_mapped_fan_out::{device_under, differing};
use crate::write_routes::GpuWriter;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RasterizationRateMap, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{RateMapRegionPolicy, VkFeature};
use metal2vulkan::metal::{
    BlendFactor, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// The map's screen side, and its quality samples on both axes.
const SCREEN: u32 = 512;
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];

/// A patch's quads on a side, and the whole target's grid's.
const PATCH: u32 = 8;
const WHOLE: u32 = 16;

/// What a command draws: its patch's screen rectangle `[x, y, width, height]` in screen pixels,
/// its quads on a side, its instances, and whether it draws through indices.
struct Patch {
    at: [f32; 4],
    quads: u32,
    instances: u32,
    indexed: bool,
}

/// The commands, in order: boxed, fanned out across ten regions, boxed of two instances, fanned
/// out over the whole target, fanned out of four triangles, boxed without indices, boxed across
/// four regions. The map's 12 segments of each axis begin at screen pixels 0, 68, 122, 166, 204,
/// 237, 269, 302, 339, 382, 434 and 499 (y from the top, a patch's `y` from the bottom).
const PATCHES: [Patch; 7] = [
    Patch {
        at: [100.0, 140.0, 10.0, 10.0],
        quads: PATCH,
        instances: 1,
        indexed: true,
    },
    Patch {
        at: [196.0, 300.0, 120.0, 40.0],
        quads: PATCH,
        instances: 1,
        indexed: true,
    },
    Patch {
        at: [400.0, 60.0, 12.0, 12.0],
        quads: PATCH,
        instances: 2,
        indexed: true,
    },
    Patch {
        at: [0.0, 0.0, SCREEN as f32, SCREEN as f32],
        quads: WHOLE,
        instances: 1,
        indexed: true,
    },
    Patch {
        at: [300.0, 200.0, 60.0, 60.0],
        quads: 1,
        instances: 1,
        indexed: true,
    },
    Patch {
        at: [40.0, 420.0, 10.0, 10.0],
        quads: PATCH,
        instances: 1,
        indexed: false,
    },
    Patch {
        at: [225.0, 227.0, 25.0, 30.0],
        quads: 2 * PATCH,
        instances: 1,
        indexed: true,
    },
];

/// The commands the host step boxes.
const BOXED: u64 = 4;

/// A command's drawing: its positions, its indices (`None`: none, its vertices in order), and
/// their count.
struct Drawing {
    positions: Buffer,
    indices: Option<Buffer>,
    count: u32,
}

/// The device, its queue, the pipeline, the commands' drawings and colours, the indirect command
/// buffer that holds them, and the map.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    drawings: Vec<Drawing>,
    colors: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
}

/// The corners of `patch`'s grid in normalized device coordinates, row by row, and its triangles'
/// corners, twice.
fn grid(patch: &Patch) -> (Vec<[f32; 2]>, Vec<u32>) {
    let [x, y, width, height] = patch.at;
    let n = patch.quads;
    let ndc = |pixels: f32| pixels * 2.0 / SCREEN as f32 - 1.0;
    let corners = (0..=n)
        .flat_map(|j| {
            (0..=n).map(move |i| {
                let at = |start: f32, extent: f32, k: u32| start + extent * k as f32 / n as f32;
                [ndc(at(x, width, i)), ndc(at(y, height, j))]
            })
        })
        .collect();
    let corner = |i: u32, j: u32| j * (n + 1) + i;
    let triangles: Vec<u32> = (0..n * n)
        .flat_map(|quad| {
            let (i, j) = (quad % n, quad / n);
            let [a, b, c, d] = [
                corner(i, j),
                corner(i + 1, j),
                corner(i, j + 1),
                corner(i + 1, j + 1),
            ];
            [a, b, c, b, d, c]
        })
        .collect();
    (corners, [&triangles[..], &triangles[..]].concat())
}

impl Workload {
    /// The workload on `device`.
    fn new(device: Device) -> Workload {
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
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
        color.source_alpha_blend_factor = BlendFactor::SourceAlpha;
        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        descriptor.color_attachments[0] = Some(color);
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let buffer = |bytes: &[u8]| {
            (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
        };
        let floats = |points: &[[f32; 2]]| -> Vec<u8> {
            points
                .iter()
                .flatten()
                .flat_map(|f| f.to_le_bytes())
                .collect()
        };
        let drawings: Vec<Drawing> = (PATCHES.iter())
            .map(|patch| {
                let (corners, indices) = grid(patch);
                let count = indices.len() as u32;
                match patch.indexed {
                    true => Drawing {
                        positions: buffer(&floats(&corners)),
                        indices: Some(buffer(
                            &(indices.iter())
                                .flat_map(|i| i.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )),
                        count,
                    },
                    false => {
                        let listed: Vec<[f32; 2]> =
                            indices.iter().map(|&i| corners[i as usize]).collect();
                        Drawing {
                            positions: buffer(&floats(&listed)),
                            indices: None,
                            count,
                        }
                    }
                }
            })
            .collect();
        // Each command's colour at half alpha, 256 bytes apart.
        let colors: Vec<u8> = (0..PATCHES.len())
            .flat_map(|k| {
                let k = k as f32;
                let mut slot = [k / 6.0, 1.0 - k / 6.0, 0.25 + k / 12.0, 0.5]
                    .map(f32::to_le_bytes)
                    .concat();
                slot.resize(256, 0);
                slot
            })
            .collect();
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW | IndirectCommandType::DRAW_INDEXED;
        icb.inherit_buffers = false;
        icb.max_vertex_buffer_bind_count = 1;
        icb.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&icb, PATCHES.len() as u64, ResourceOptions::shared())
            .unwrap();
        let mut map = RateMapDescriptor::new([SCREEN, SCREEN]);
        map.layers = vec![RateMapLayerDescriptor::new(
            QUALITY.to_vec(),
            QUALITY.to_vec(),
        )];
        let map = device.new_rasterization_rate_map(&map).unwrap();
        let workload = Workload {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            drawings,
            colors: buffer(&colors),
            icb,
            map,
            device,
        };
        for (k, (patch, drawing)) in PATCHES.iter().zip(&workload.drawings).enumerate() {
            let command = workload.icb.indirect_render_command_at(k as u64).unwrap();
            command
                .set_render_pipeline_state(&workload.pipeline)
                .unwrap();
            command
                .set_vertex_buffer(Some(&drawing.positions), 0, 0)
                .unwrap();
            command
                .set_fragment_buffer(Some(&workload.colors), k as u64 * 256, 0)
                .unwrap();
            let instances = u64::from(patch.instances);
            match &drawing.indices {
                Some(indices) => command.draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    u64::from(drawing.count),
                    IndexType::UInt32,
                    indices,
                    0,
                    instances,
                    0,
                    0,
                ),
                None => command.draw_primitives(
                    PrimitiveType::Triangle,
                    0,
                    u64::from(drawing.count),
                    instances,
                    0,
                ),
            }
            .unwrap();
        }
        // The GPU may write the records: the host step records the pass.
        workload.icb.written_by_gpu(GpuWriter::Reset);
        drop(workload.icb.gpu_write());
        workload
    }

    /// One frame into a target of the map's physical size cleared to opaque black: the commands
    /// executed from the indirect command buffer where `executed`, else drawn on the encoder.
    /// The target's bytes, and the device's counts over the frame.
    fn frame(&self, executed: bool) -> (Vec<u8>, WorkCounts, WorkCounts) {
        let size = self.map.physical_size_for_layer(0);
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&d).unwrap();
        let before = self.device.work_counts();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        pass.rasterization_rate_map = Some(self.map.clone());
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        match executed {
            true => encoder
                .execute_commands_in_buffer(&self.icb, 0..PATCHES.len() as u64)
                .unwrap(),
            false => {
                for (k, (patch, drawing)) in PATCHES.iter().zip(&self.drawings).enumerate() {
                    encoder
                        .set_vertex_buffer(Some(&drawing.positions), 0, 0)
                        .unwrap();
                    encoder
                        .set_fragment_buffer(Some(&self.colors), k as u64 * 256, 0)
                        .unwrap();
                    match &drawing.indices {
                        Some(indices) => encoder.draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            drawing.count,
                            IndexType::UInt32,
                            indices,
                            0,
                            patch.instances,
                            0,
                            0,
                        ),
                        None => encoder.draw_primitives(
                            PrimitiveType::Triangle,
                            0,
                            drawing.count,
                            patch.instances,
                            0,
                        ),
                    }
                    .unwrap();
                }
            }
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        drop(command_buffer);
        let after = self.device.work_counts();
        let (width, height) = (size.width, size.height);
        let mut bytes = vec![0u8; (width * height * 4) as usize];
        let region = Region::new_2d(0, 0, width, height);
        target
            .get_bytes(&mut bytes, width * 4, 0, region, 0, 0)
            .unwrap();
        (bytes, before, after)
    }
}

/// Whether `device` fans out and draws region copies: it enables the features the fan-out, its
/// probe and the region-copy variant take.
fn boxes(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    let features = &profile.enabled().features;
    [
        VkFeature::GeometryShader,
        VkFeature::ShaderClipDistance,
        VkFeature::VertexPipelineStoresAndAtomics,
        VkFeature::ShaderOutputViewportIndex,
        VkFeature::ShaderDrawParameters,
    ]
    .iter()
    .all(|feature| features.contains(feature))
}

/// The probes, the draws they flagged, the draws fanned out, the draws boxed, the draws through
/// the region-copy variant, and the boxed draws' vertex work between two counts.
fn counts(before: &WorkCounts, after: &WorkCounts) -> [u64; 6] {
    [
        after.rate_map_probes - before.rate_map_probes,
        after.rate_map_flagged_draws - before.rate_map_flagged_draws,
        after.rate_map_fanned_draws - before.rate_map_fanned_draws,
        after.rate_map_boxed_draws - before.rate_map_boxed_draws,
        after.rate_map_region_copy_draws - before.rate_map_region_copy_draws,
        after.rate_map_boxed_vertex_regions - before.rate_map_boxed_vertex_regions,
    ]
}

/// Under `FanOut` the host step probes once, flags nothing, boxes the four patches and fans out
/// the strip, the whole target's grid and the wide quad; the boxed draws reach fewer regions than
/// the table holds and at least one each. The pixels are the encoder's, which draws every command
/// through its region-copy variant into every region. Under `Exact` nothing is boxed.
#[test]
fn a_host_step_boxes_the_draws_whose_triangles_lie_in_few_regions() {
    let scratch = Scratch::new("rate-mapped-boxed-draws");
    let workload = Workload::new(device_under(&scratch.0, RateMapRegionPolicy::FanOut));
    let (drawn, before, after) = workload.frame(false);
    let whole =
        after.rate_map_whole_copy_vertex_regions - before.rate_map_whole_copy_vertex_regions;
    let (boxed, before, after) = workload.frame(true);
    let seen = counts(&before, &after);
    eprintln!(
        "RATE-MAP-BOXED probes, flagged, fanned, boxed, region-copy draws, boxed vertex work \
         {seen:?}; the encoder's vertex work {whole}"
    );
    // Each command's vertices or indices times its instances.
    let vertices = |patch: &Patch| u64::from(6 * patch.quads * patch.quads * 2 * patch.instances);
    let all: u64 = PATCHES.iter().map(vertices).sum();
    let boxed_vertices: u64 = [0, 2, 5, 6].map(|k| vertices(&PATCHES[k])).iter().sum();
    match boxes(&workload.device) {
        true => {
            let table = whole / all;
            assert!(table > 1 && whole == all * table, "the encoder's {whole}");
            assert_eq!(seen[..5], [1, 0, 3, BOXED, BOXED], "{seen:?}");
            assert!(
                (boxed_vertices..boxed_vertices * table).contains(&seen[5]),
                "the boxed draws' vertex work {} of at most {} into every region",
                seen[5],
                boxed_vertices * table
            );
        }
        false => assert_eq!(seen[3..], [0; 3], "{seen:?}"),
    }
    let (differ, first) = differing(&drawn, &boxed);
    assert_eq!(
        differ, 0,
        "texels the boxed execution draws otherwise, the first {first:?}"
    );
    // Under `Exact` no probe runs and nothing is boxed.
    let exact = Workload::new(device_under(&scratch.0, RateMapRegionPolicy::Exact));
    let (_, before, after) = exact.frame(true);
    assert_eq!(counts(&before, &after), [0; 6], "under Exact");
}

/// The first frame of a device that drew none of the commands' states before: its host step boxes
/// the four patches through their region-copy variant and fans the others out, with the pipelines
/// `commit`'s settle began, keyed as the step keys them, and makes none itself
/// (`WorkCounts::host_step_pipelines_made`).
#[test]
fn a_host_step_s_first_boxed_frame_makes_no_pipeline() {
    let scratch = Scratch::new("rate-mapped-boxed-first-frame");
    let workload = Workload::new(device_under(&scratch.0, RateMapRegionPolicy::FanOut));
    let (_, before, after) = workload.frame(true);
    let seen = counts(&before, &after);
    let made = after.host_step_pipelines_made - before.host_step_pipelines_made;
    eprintln!(
        "RATE-MAP-BOXED-FIRST-FRAME probes, flagged, fanned, boxed, region-copy draws, boxed \
         vertex work {seen:?}; {made} made at the step"
    );
    if boxes(&workload.device) {
        assert_eq!(seen[3], BOXED, "draws boxed {seen:?}");
    }
    assert_eq!(made, 0, "pipelines the host step made");
}
