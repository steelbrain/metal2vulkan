//! A rate-mapped pass's executed commands recorded in pieces (`rate_mapped_pass.rs`, **Pieces**),
//! under a piece work small enough that each command's draw is drawn as ranges of its indices and
//! the pass is cut between them: the target's bytes are those of the same commands drawn on the
//! encoder, whose draws are never cut, and of the execution under the default piece work, which
//! cuts nothing here. Each command draws a grid of triangles twice over the target, blended at
//! half alpha in its own colour, through 32-bit indices or 16-bit ones from an offset: a range
//! lost, drawn twice or out of order, or a piece out of order, changes the pixels. A device asked
//! to answer GPU times from the host's clock (`DeviceOptions::gpu_times_from_host`) times every
//! piece of the cut frame all the same, where its driver's timestamps do not wait for every
//! queue's.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_triangle` and
//! `tob_solid` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`
//! (`llvm-as`).

use super::blit_holds::{device, options, Scratch, WAIT};
use crate::gpu_time::FromHost;
use crate::rate_mapped_pass::set_piece_work;
use crate::{
    Buffer, ClearColor, CommandBufferStatus, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RasterizationRateMap, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::{
    BlendFactor, IndexType, IndirectCommandType, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The grid's quads on a side, and the commands.
const GRID: u32 = 16;
const COMMANDS: u32 = 8;
/// Each command's indices: the grid's triangles, twice.
const INDICES: u32 = 2 * GRID * GRID * 6;
/// The 16-bit indices before a command's own, so its offset is not 0.
const LEAD: u32 = 6;
/// The map's screen side, and its quality samples on both axes.
const SCREEN: u32 = 512;
const QUALITY: [f32; 5] = [0.3, 0.6, 1.0, 0.6, 0.3];
/// The piece work the cut frame is recorded under.
const SMALL_WORK: u64 = 1 << 18;

/// The device, its queue, and what the frames draw with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    positions: Buffer,
    colors: Buffer,
    wide: Buffer,
    narrow: Buffer,
    icb: IndirectCommandBuffer,
    map: RasterizationRateMap,
}

/// A command's index buffer, its type and its offset: 32-bit from 0 for an even command, 16-bit
/// past [`LEAD`] for an odd one.
fn indices_of(workload: &Workload, command: u32) -> (&Buffer, IndexType, u64) {
    match command % 2 {
        0 => (&workload.wide, IndexType::UInt32, 0),
        _ => (&workload.narrow, IndexType::UInt16, u64::from(LEAD) * 2),
    }
}

impl Workload {
    fn new(device: Device) -> Workload {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/fixtures/public/engine/tob_renders.ll");
        let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
        let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
        let library = device.new_library_with_compile_library(library);
        let function = |name| {
            StageSource::Function(
                library
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
        // The grid's corners over the target, and its triangles twice.
        let at = |i: u32| i as f32 * 2.0 / GRID as f32 - 1.0;
        let positions: Vec<u8> = (0..=GRID)
            .flat_map(|y| (0..=GRID).map(move |x| [at(x), at(y)]))
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        let corner = |x: u32, y: u32| y * (GRID + 1) + x;
        let grid: Vec<u32> = (0..GRID * GRID)
            .flat_map(|quad| {
                let (x, y) = (quad % GRID, quad / GRID);
                let [a, b, c, d] = [
                    corner(x, y),
                    corner(x + 1, y),
                    corner(x, y + 1),
                    corner(x + 1, y + 1),
                ];
                [a, b, c, b, d, c]
            })
            .collect();
        let twice = [&grid[..], &grid[..]].concat();
        let wide: Vec<u8> = twice.iter().flat_map(|i| i.to_le_bytes()).collect();
        let narrow: Vec<u8> = (std::iter::repeat_n(0u16, LEAD as usize))
            .chain(twice.iter().map(|&i| i as u16))
            .flat_map(u16::to_le_bytes)
            .collect();
        // Each command's colour at half alpha, 256 bytes apart.
        let colors: Vec<u8> = (0..COMMANDS)
            .flat_map(|k| {
                let k = k as f32;
                let mut slot = [k / 8.0, 1.0 - k / 8.0, 0.25 + k / 16.0, 0.5]
                    .map(f32::to_le_bytes)
                    .concat();
                slot.resize(256, 0);
                slot
            })
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
            .new_indirect_command_buffer(&icb, u64::from(COMMANDS), ResourceOptions::shared())
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
            positions: buffer(&positions),
            colors: buffer(&colors),
            wide: buffer(&wide),
            narrow: buffer(&narrow),
            icb,
            map,
            device,
        };
        for k in 0..COMMANDS {
            let command = workload
                .icb
                .indirect_render_command_at(u64::from(k))
                .unwrap();
            command
                .set_render_pipeline_state(&workload.pipeline)
                .unwrap();
            command
                .set_vertex_buffer(Some(&workload.positions), 0, 0)
                .unwrap();
            command
                .set_fragment_buffer(Some(&workload.colors), u64::from(k) * 256, 0)
                .unwrap();
            let (indices, ty, offset) = indices_of(&workload, k);
            command
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    u64::from(INDICES),
                    ty,
                    indices,
                    offset,
                    1,
                    0,
                    0,
                )
                .unwrap();
        }
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
                .execute_commands_in_buffer(&self.icb, 0..u64::from(COMMANDS))
                .unwrap(),
            false => {
                for k in 0..COMMANDS {
                    let (indices, ty, offset) = indices_of(self, k);
                    encoder
                        .set_vertex_buffer(Some(&self.positions), 0, 0)
                        .unwrap();
                    encoder
                        .set_fragment_buffer(Some(&self.colors), u64::from(k) * 256, 0)
                        .unwrap();
                    encoder
                        .draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            INDICES,
                            ty,
                            indices,
                            offset,
                            1,
                            0,
                            0,
                        )
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

/// Under a small piece work every command's draw is drawn as ranges of its indices and the pass
/// is cut into pieces; the pixels are those of the encoder's own draws and of the uncut
/// execution, which neither cut nor range.
#[test]
fn executed_commands_cut_into_pieces_and_ranges_draw_the_uncut_pixels() {
    let scratch = Scratch::new("rate-mapped-pieces");
    let workload = Workload::new(device(&scratch.0));
    let (drawn, before, after) = workload.frame(false);
    let cuts = |before: &WorkCounts, after: &WorkCounts| {
        (
            after.rate_map_pieces - before.rate_map_pieces,
            after.rate_map_index_ranges - before.rate_map_index_ranges,
        )
    };
    assert_eq!(cuts(&before, &after), (0, 0), "the encoder's draws");
    let (uncut, before, after) = workload.frame(true);
    assert_eq!(cuts(&before, &after), (0, 0), "the default piece work");
    set_piece_work(Some(SMALL_WORK));
    let (cut, before, after) = workload.frame(true);
    set_piece_work(None);
    let (pieces, ranges) = cuts(&before, &after);
    let timed = after.rate_map_pieces_timed - before.rate_map_pieces_timed;
    let nanos = after.rate_map_piece_gpu_nanos - before.rate_map_piece_gpu_nanos;
    eprintln!(
        "RATE-MAP-PIECES pieces {pieces} ranges {ranges} timed {timed} gpu_nanos {nanos} \
         most {}",
        after.rate_map_piece_gpu_nanos_most
    );
    assert!(pieces > 0 && ranges > 0, "{pieces} pieces, {ranges} ranges");
    // Every pixel took every command's blends: none is the clear's black.
    assert!(
        drawn.chunks(4).all(|texel| texel[..3] != [0, 0, 0]),
        "a pixel the commands did not draw"
    );
    assert!(
        drawn == uncut,
        "the uncut execution's pixels differ from the encoder's"
    );
    assert!(
        cut == uncut,
        "the cut execution's pixels differ from the uncut one's"
    );
}

/// A device asked to answer command buffers' GPU times from the host's clock still times every
/// piece of a pass recorded in pieces, where its queue family writes timestamps and its driver's
/// do not wait for every queue's (the piece timer's: `rate_mapped_pass.rs`, **Pieces**).
#[test]
fn a_device_answering_gpu_times_from_the_host_times_its_pieces() {
    let scratch = Scratch::new("rate-mapped-pieces-host-times");
    let mut asked = options(&scratch.0);
    asked.gpu_times_from_host = true;
    let workload = Workload::new(Device::system_default_with(&asked).unwrap());
    let timing = &workload.device.shared().timing;
    assert!(
        timing.answers_from_host(),
        "the device answers from the host"
    );
    set_piece_work(Some(SMALL_WORK));
    let (_, before, after) = workload.frame(true);
    set_piece_work(None);
    let pieces = after.rate_map_pieces - before.rate_map_pieces;
    let timed = after.rate_map_pieces_timed - before.rate_map_pieces_timed;
    eprintln!("RATE-MAP-PIECES from the host: pieces {pieces} timed {timed}");
    assert!(pieces > 0, "the frame was not cut");
    let times = timing.clock().is_some() && timing.host_clock_reason() == Some(FromHost::Asked);
    assert_eq!(
        timed,
        if times { pieces + 1 } else { 0 },
        "{pieces} pieces past the first, {timed} timed"
    );
}
