//! A fragment function's threadgroup memory, its tile's (`lower::access::tile_memory`,
//! `runtime/render/tile_memory.rs`), on the device the cfg selects, under the validation layer. The
//! owned fixture `cfit_fragment_tile_memory`'s functions run in passes of one `R32Uint` attachment
//! of 40x24 pixels in 16x16 tiles, 3x2 of them, whose `threadgroupMemoryLength` is 64 bytes, each
//! draw a rectangle of pixels (`cfit_frag_tile_rect`):
//!
//! - `count` adds 1 to word 0 of its argument 0 in each fragment, `store` writes `v` to word 1,
//!   and `report` writes words 0 and 1 to `out[2 * tile]` and `out[2 * tile + 1]`.
//! - `or` sets its pixel's bit of the tile's bitset (argument 0, eight words), the splat
//!   rasterizer's shape; the tile functions `clear` (each bitset word to 0) and `resolve` (each
//!   word ORed into `out[tile * 8 + word]`) run between the draws.
//!
//! As Metal does on an M4:
//!
//! - Every fragment of a tile shares its memory, which starts at zero: a draw over the pass counts
//!   each tile's covered pixels, and at 2 and 4 samples (each the device takes) each covered
//!   sample.
//! - Draws see each other's writes in encode order: the later of two draws' stores stays, and a
//!   draw over part of a tile writes the whole tile's word.
//! - Tile dispatches and draws share the memory both ways: a clear between two draws of `or`
//!   leaves only the later draw's bits for the resolve, and without it both draws' bits stay.
//! - A draw of a function whose argument the encoder never set is the binder's
//!   `MissingThreadgroupMemory`, where Metal's validation stops the draw.
//!
//! And an indirect command buffer's commands executed in the pass reach the memory as the
//! encoder's draws do, with the arguments the encoder set at the call.

use super::tile_states::on_device;
use super::tile_states::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{BindError, Feature};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Device, Error, IndirectCommandBufferDescriptor, Library,
    LoadAction, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The pass's extent.
const AREA: [u32; 2] = [40, 24];

/// The pass's tile side.
const TILE: u32 = 16;

/// The tiles across and down.
const GRID: [u32; 2] = [3, 2];

/// The pass's tiles.
const TILES: u32 = GRID[0] * GRID[1];

/// The bitset's words a tile.
const WORDS: u32 = TILE * TILE / 32;

/// A rectangle of pixels: `x0, y0, x1, y1`, the ends excluded.
type Rect = [u32; 4];

/// The whole pass.
const ALL: Rect = [0, 0, AREA[0], AREA[1]];

/// The fixture's functions on one device.
struct Fixture {
    device: Device,
    library: Library,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let bitcode = llvm_assemble(&engine_dir().join("cfit_fragment_tile_memory.ll")).unwrap();
        Fixture {
            library: device.new_library_with_air(&bitcode).unwrap(),
            device: device.clone(),
        }
    }

    fn source(&self, name: &str) -> StageSource {
        let function = self.library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    }

    /// The state of the rectangle and fragment function `cfit_frag_tile_<fragment>` over one
    /// `R32Uint` attachment of `samples`.
    fn draw_state(&self, fragment: &str, samples: u32) -> RenderPipelineState {
        self.state(fragment, samples, false).unwrap()
    }

    /// The same of one sample, made for indirect command buffers' commands.
    fn command_state(&self, fragment: &str) -> RenderPipelineState {
        self.state(fragment, 1, true).unwrap()
    }

    fn state(
        &self,
        fragment: &str,
        samples: u32,
        indirect: bool,
    ) -> Result<RenderPipelineState, Error> {
        let mut descriptor = RenderPipelineDescriptor::new(self.source("cfit_frag_tile_rect"));
        descriptor.fragment = Some(self.source(&format!("cfit_frag_tile_{fragment}")));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        descriptor.raster_sample_count = samples;
        descriptor.support_indirect_command_buffers = indirect;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The tile state of `cfit_tile_bits_<kernel>` over one `R32Uint` attachment.
    fn tile_state(&self, kernel: &str) -> RenderPipelineState {
        let mut descriptor =
            TileRenderPipelineDescriptor::new(self.source(&format!("cfit_tile_bits_{kernel}")));
        descriptor.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
        self.device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The pass's attachment, of `samples`.
    fn target(&self, samples: u32) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(
            PixelFormat::R32Uint,
            AREA[0].into(),
            AREA[1].into(),
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        if samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = samples.into();
        }
        self.device.new_texture(&descriptor).unwrap()
    }

    /// A buffer of `words`.
    fn words(&self, words: &[u32]) -> Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        buffer_with(&self.device, &bytes).unwrap()
    }

    /// A buffer of `n` words, each `0xABABABAB` until written.
    fn out(&self, n: u32) -> Buffer {
        buffer_with(&self.device, &vec![0xAB; (n * 4) as usize]).unwrap()
    }

    /// Encode one pass over a new attachment of `samples` by `encode`, and run it.
    fn run(&self, samples: u32, encode: impl FnOnce(&mut RenderCommandEncoder)) {
        let target = self.target(samples);
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target);
        pass.color_attachments[0].load_action = LoadAction::Clear;
        pass.color_attachments[0].store_action = StoreAction::DontCare;
        pass.tile_width = TILE.into();
        pass.tile_height = TILE.into();
        pass.threadgroup_memory_length = 64;
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
}

/// The words of `buffer`.
fn words(buffer: &Buffer) -> Vec<u32> {
    read_buffer(buffer, 0, buffer.length())
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// The pass's render area, as the rectangle's vertex buffer 1 takes it.
fn render_area(f: &Fixture) -> Buffer {
    let size: Vec<u8> = [AREA[0] as f32, AREA[1] as f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    buffer_with(&f.device, &size).unwrap()
}

/// The tiles across and the tile side, as fragment buffer 1 takes them.
fn tiles(f: &Fixture) -> Buffer {
    f.words(&[GRID[0], TILE])
}

/// The corners of `rect`, as the rectangle's vertex buffer 0 takes them.
fn corners(f: &Fixture, rect: Rect) -> Buffer {
    let corners: Vec<u8> = rect
        .iter()
        .flat_map(|&v| (v as f32).to_le_bytes())
        .collect();
    buffer_with(&f.device, &corners).unwrap()
}

/// The encoder state every draw of the fixture reads: the rectangle's render area at vertex
/// buffer 1, and the tiles across and the tile side at fragment buffer 1.
fn bind_pass(f: &Fixture, encoder: &mut RenderCommandEncoder) -> [Buffer; 2] {
    let size = render_area(f);
    let pass = tiles(f);
    encoder.set_vertex_buffer(Some(&size), 0, 1).unwrap();
    encoder.set_fragment_buffer(Some(&pass), 0, 1).unwrap();
    [size, pass]
}

/// Draw `state`'s rectangle `rect`.
fn draw(
    f: &Fixture,
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
    rect: Rect,
) -> Buffer {
    let corners = corners(f, rect);
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
        .unwrap();
    corners
}

/// A tile dispatch of `state`, 4x4 threads a tile, with the bitset's words at tile buffer 1.
fn dispatch(
    f: &Fixture,
    encoder: &mut RenderCommandEncoder,
    state: &RenderPipelineState,
) -> Buffer {
    let words = f.words(&[WORDS]);
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_tile_buffer(Some(&words), 0, 1).unwrap();
    encoder
        .dispatch_threads_per_tile(Size::new(4, 4, 1))
        .unwrap();
    words
}

/// The tile of pixel `(x, y)`.
fn tile_of(x: u32, y: u32) -> u32 {
    (y / TILE) * GRID[0] + x / TILE
}

/// The pixels of `rect` inside the pass.
fn pixels(rect: Rect) -> impl Iterator<Item = (u32, u32)> {
    let [x0, y0, x1, y1] = rect;
    (y0..y1.min(AREA[1])).flat_map(move |y| (x0..x1.min(AREA[0])).map(move |x| (x, y)))
}

/// Each tile's pixels of `rect`.
fn covered(rect: Rect) -> Vec<u32> {
    let mut counts = vec![0; TILES as usize];
    for (x, y) in pixels(rect) {
        counts[tile_of(x, y) as usize] += 1;
    }
    counts
}

/// Each tile's bitset of the pixels of `rects`, tile by tile.
fn bitsets(rects: &[Rect]) -> Vec<u32> {
    let mut bits = vec![0u32; (TILES * WORDS) as usize];
    for &rect in rects {
        for (x, y) in pixels(rect) {
            let i = (y % TILE) * TILE + x % TILE;
            bits[(tile_of(x, y) * WORDS + i / 32) as usize] |= 1 << (i % 32);
        }
    }
    bits
}

#[test]
fn fragments_share_their_tiles_memory_and_count_each_sample() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let mut multisampled = Vec::new();
        for samples in [1, 2, 4] {
            // A device takes the multisampled `R32Uint` attachment at some of these counts
            // (lavapipe on x86 not at 2): one it does not take is refused as Metal's device
            // refuses a count it does not support.
            let count = match f.state("count", samples, false) {
                Err(Error::Unsupported(u))
                    if samples > 1 && u.feature == Feature::RasterSampleCount =>
                {
                    continue
                }
                state => state.unwrap(),
            };
            if samples > 1 {
                multisampled.push(samples);
            }
            let report = f.draw_state("report", samples);
            let out = f.out(2 * TILES);
            f.run(samples, |encoder| {
                let _pass = bind_pass(&f, encoder);
                encoder.set_threadgroup_memory_length(64, 0, 0).unwrap();
                let _a = draw(&f, encoder, &count, ALL);
                encoder.set_fragment_buffer(Some(&out), 0, 0).unwrap();
                let _b = draw(&f, encoder, &report, ALL);
            });
            let expected: Vec<u32> = covered(ALL)
                .into_iter()
                .flat_map(|pixels| [pixels * samples, 0])
                .collect();
            assert_eq!(words(&out), expected, "{samples} samples");
        }
        assert!(!multisampled.is_empty(), "no multisampled count ran");
    });
}

#[test]
fn draws_see_each_others_tile_memory_writes_in_order() {
    on_device(|device| {
        let f = Fixture::new(device);
        let store = f.draw_state("store", 1);
        let report = f.draw_state("report", 1);
        let outs = [f.out(2 * TILES), f.out(2 * TILES)];
        let values = [5u32, 9, 7].map(|v| f.words(&[v]));
        let left: Rect = [0, 0, 20, AREA[1]];
        f.run(1, |encoder| {
            let _pass = bind_pass(&f, encoder);
            encoder.set_threadgroup_memory_length(64, 0, 0).unwrap();
            let mut held = Vec::new();
            for (value, rect) in values.iter().zip([ALL, ALL]) {
                encoder.set_fragment_buffer(Some(value), 0, 0).unwrap();
                held.push(draw(&f, encoder, &store, rect));
            }
            encoder.set_fragment_buffer(Some(&outs[0]), 0, 0).unwrap();
            held.push(draw(&f, encoder, &report, ALL));
            encoder.set_fragment_buffer(Some(&values[2]), 0, 0).unwrap();
            held.push(draw(&f, encoder, &store, left));
            encoder.set_fragment_buffer(Some(&outs[1]), 0, 0).unwrap();
            held.push(draw(&f, encoder, &report, ALL));
        });
        let report_of = |word: &dyn Fn(u32) -> u32| -> Vec<u32> {
            (0..TILES).flat_map(|tile| [0, word(tile)]).collect()
        };
        assert_eq!(words(&outs[0]), report_of(&|_| 9), "the later store");
        // A draw over part of a tile writes the tile's word: the left 20 pixels reach the first
        // two columns of tiles.
        assert_eq!(
            words(&outs[1]),
            report_of(&|tile| if tile % GRID[0] < 2 { 7 } else { 9 }),
            "a store over the left 20 pixels"
        );
    });
}

#[test]
fn the_splat_shape_sets_its_tiles_bitsets_between_tile_dispatches() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let or = f.draw_state("or", 1);
        let clear = f.tile_state("clear");
        let resolve = f.tile_state("resolve");
        let a: Rect = [2, 3, 13, 9];
        let b: Rect = [10, 6, 35, 20];
        let outs = [
            f.words(&[0; (TILES * WORDS) as usize]),
            f.words(&[0; (TILES * WORDS) as usize]),
        ];
        // Clear, two splats, resolve: both splats' bits.
        f.run(1, |encoder| {
            let _pass = bind_pass(&f, encoder);
            encoder.set_threadgroup_memory_length(32, 0, 0).unwrap();
            let _c = dispatch(&f, encoder, &clear);
            let _a = draw(&f, encoder, &or, a);
            let _b = draw(&f, encoder, &or, b);
            encoder.set_tile_buffer(Some(&outs[0]), 0, 0).unwrap();
            let _r = dispatch(&f, encoder, &resolve);
        });
        assert_eq!(words(&outs[0]), bitsets(&[a, b]), "both splats");
        // A splat, the clear, a splat, resolve: the later splat's bits alone.
        f.run(1, |encoder| {
            let _pass = bind_pass(&f, encoder);
            encoder.set_threadgroup_memory_length(32, 0, 0).unwrap();
            let _a = draw(&f, encoder, &or, a);
            let _c = dispatch(&f, encoder, &clear);
            let _b = draw(&f, encoder, &or, b);
            encoder.set_tile_buffer(Some(&outs[1]), 0, 0).unwrap();
            let _r = dispatch(&f, encoder, &resolve);
        });
        assert_eq!(words(&outs[1]), bitsets(&[b]), "the splat after the clear");
    });
}

/// An execution's commands reach the tile memory as the encoder's draws do, with the arguments
/// the encoder set at the call: a draw counts, then an execution's two commands count and a third
/// reports, and a draw after the execution reports too. Each tile counts three times its pixels in
/// both reports: the commands' report sees the draw's count and theirs, ordered after them inside
/// the execution, and the draw after it sees theirs.
#[test]
fn an_executions_draws_reach_the_tile_memory_in_order() {
    on_device(|device| {
        let f = Fixture::new(device);
        let (count, report) = (f.command_state("count"), f.command_state("report"));
        let outs = [f.out(2 * TILES), f.out(2 * TILES)];
        let (all, size, pass) = (corners(&f, ALL), render_area(&f), tiles(&f));
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 2;
        descriptor.max_fragment_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 3, ResourceOptions::shared())
            .unwrap();
        for (index, state) in [(0, &count), (1, &count), (2, &report)] {
            let command = icb.indirect_render_command_at(index).unwrap();
            command.set_render_pipeline_state(state).unwrap();
            command.set_vertex_buffer(Some(&all), 0, 0).unwrap();
            command.set_vertex_buffer(Some(&size), 0, 1).unwrap();
            command.set_fragment_buffer(Some(&outs[0]), 0, 0).unwrap();
            command.set_fragment_buffer(Some(&pass), 0, 1).unwrap();
            command
                .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
                .unwrap();
        }
        let (direct_count, direct_report) = (f.draw_state("count", 1), f.draw_state("report", 1));
        f.run(1, |encoder| {
            let _pass = bind_pass(&f, encoder);
            let resident = [
                (&all, ResourceUsage::READ, RenderStages::VERTEX),
                (&size, ResourceUsage::READ, RenderStages::VERTEX),
                (&pass, ResourceUsage::READ, RenderStages::FRAGMENT),
                (&outs[0], ResourceUsage::WRITE, RenderStages::FRAGMENT),
            ];
            for (buffer, usage, stages) in resident {
                encoder
                    .use_resource(buffer.as_resource(), usage, stages)
                    .unwrap();
            }
            encoder.set_threadgroup_memory_length(64, 0, 0).unwrap();
            let _a = draw(&f, encoder, &direct_count, ALL);
            encoder.execute_commands_in_buffer(&icb, 0..3).unwrap();
            encoder.set_fragment_buffer(Some(&outs[1]), 0, 0).unwrap();
            let _b = draw(&f, encoder, &direct_report, ALL);
        });
        let expected: Vec<u32> = covered(ALL)
            .into_iter()
            .flat_map(|pixels| [3 * pixels, 0])
            .collect();
        assert_eq!(words(&outs[0]), expected, "the commands' report");
        assert_eq!(words(&outs[1]), expected, "the draw's after the execution");
    });
}

#[test]
fn a_draw_of_an_unset_tile_argument_is_refused_as_metal_refuses_it() {
    on_device(|device| {
        let f = Fixture::new(device);
        let count = f.draw_state("count", 1);
        f.run(1, |encoder| {
            let _pass = bind_pass(&f, encoder);
            let corners = f.words(&[0; 4]);
            encoder.set_render_pipeline_state(&count).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            assert_eq!(
                encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0),
                Err(Error::Bind(BindError::MissingThreadgroupMemory {
                    index: 0
                }))
            );
        });
    });
}
