//! A tile function's explicit imageblock that stands alone, the pass's imageblock storage, which
//! the pass's fragment functions' imageblocks are in (`lower::access::tile`,
//! `runtime/fragment_imageblock.rs`), on the device the cfg selects, under the validation layer.
//! The owned fixture `cfit_tile_imageblock`'s functions run in passes of one `R32Uint` attachment
//! of 40x24 pixels in 16x16 tiles, 3x2 of them, whose imageblock sample length is the pipelines' 16
//! bytes, each draw a rectangle of pixels (`cfit_ib_rect`) and each tile dispatch one thread per
//! pixel of the tile:
//!
//! - `splat` sets bit `s` of its pixel's eight-byte cell (`lo`, then `hi`) under a raster order
//!   group, the splat rasterizer's shape; the tile functions `clear` (every cell to a seed), `add`
//!   (to every cell's low word) and `resolve` (every cell of the render area to `out[2 * pixel]`
//!   and `out[2 * pixel + 1]`) run between the draws, and `resolve_halves` resolves through a
//!   layout of two halves and a word of the same eight bytes.
//!
//! As Metal does on an M4:
//!
//! - Tile dispatches and draws share each pixel's cell both ways and in encode order: a clear's
//!   seed reaches the splats, which the resolve reads with their bits; a clear between splats
//!   leaves the later splats' bits alone.
//! - A dispatch reads what the dispatch before it wrote, and a pass's cells start at zero.
//! - A layout of other members but the same size reads the same bytes.
//! - A tile state answers its imageblock's sample length as a fragment function's (16 bytes for
//!   the eight-byte cell beside an `R32Uint` attachment). Metal's validation stops it in a pass
//!   that gives less ("the pipelineState's per sample imageBlock usage(16) is greater than the
//!   encoder's perSample imageBlock usage(4)"); plain Metal runs it there, its cell's words at or
//!   past the pass's bytes rounded up to 8 reading zero (probes, `tile_kept_bytes.rs`), so a pass
//!   of the attachment's 4 bytes keeps none of the cell.
//!
//! And as Metal does on the M3 Max and an M4 without its API validation, which stops both (a
//! probe):
//!
//! - A draw through another master of the cell's size and alignment (the owned fixture
//!   `cfb_imageblock_masters`, the splat's two words swapped) reads and writes the same bytes,
//!   each at its own offsets.
//! - A tile function whose layout is aliased onto the attachment and no larger than its texel
//!   (`aliased`, one `uint`), which the runtime keeps in its dispatch's threadgroup memory, runs
//!   beside the pass's storage in either order: it writes the attachment's texel, and the
//!   storage's cells keep what the draws and dispatches around it wrote.

use super::tile_states::on_device;
use super::tile_states::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, Device, Error, Library, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};

/// The pass's extent.
const AREA: [u32; 2] = [40, 24];

/// The pass's tile side.
const TILE: u32 = 16;

/// The pass's pixels.
const PIXELS: u32 = AREA[0] * AREA[1];

/// The pipelines' imageblock bytes per sample: the eight-byte cell, then the attachment's four
/// bytes rounded up to 8.
const SAMPLE_LENGTH: u64 = 16;

/// A rectangle of pixels: `x0, y0, x1, y1`, the ends excluded.
type Rect = [u32; 4];

/// The fixtures' functions on one device: `cfit_tile_imageblock`'s, and `cfb_imageblock_masters`'s
/// splat through another master.
struct Fixture {
    device: Device,
    library: Library,
    masters: Library,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let library = |name: &str| {
            let bitcode = llvm_assemble(&engine_dir().join(format!("{name}.ll"))).unwrap();
            device.new_library_with_air(&bitcode).unwrap()
        };
        Fixture {
            library: library("cfit_tile_imageblock"),
            masters: library("cfb_imageblock_masters"),
            device: device.clone(),
        }
    }

    /// The function `name` of `library`.
    fn source(library: &Library, name: &str) -> StageSource {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    }

    /// The state of the rectangle and `cfit_ib_splat` over one `R32Uint` attachment, or why the
    /// device makes none (`Unsupported`: a fragment imageblock needs the ordered pixel interlock).
    fn splat_state(&self) -> Result<RenderPipelineState, Error> {
        self.draw_state(&self.library, "cfit_ib_rect", "cfit_ib_splat")
    }

    /// The state of the rectangle and `cfb_ib_splat_swapped`, whose cell is another master of
    /// the splat's eight bytes, over one `R32Uint` attachment, or why the device makes none.
    fn swapped_state(&self) -> Result<RenderPipelineState, Error> {
        self.draw_state(&self.masters, "cfb_ib_rect", "cfb_ib_splat_swapped")
    }

    /// The state of `vertex` and `fragment` of `library` over one `R32Uint` attachment.
    fn draw_state(
        &self,
        library: &Library,
        vertex: &str,
        fragment: &str,
    ) -> Result<RenderPipelineState, Error> {
        let mut descriptor = RenderPipelineDescriptor::new(Fixture::source(library, vertex));
        descriptor.fragment = Some(Fixture::source(library, fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// The tile state of `cfit_ib_tile_<kernel>` over one `R32Uint` attachment.
    fn tile_state(&self, kernel: &str) -> RenderPipelineState {
        let mut descriptor = TileRenderPipelineDescriptor::new(Fixture::source(
            &self.library,
            &format!("cfit_ib_tile_{kernel}"),
        ));
        descriptor.color_attachment_formats[0] = Some(PixelFormat::R32Uint);
        self.device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A buffer of `words`.
    fn words(&self, words: &[u32]) -> Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        buffer_with(&self.device, &bytes).unwrap()
    }

    /// The resolve's buffer: two words a pixel, each `0xABABABAB` until written.
    fn out(&self) -> Buffer {
        buffer_with(&self.device, &vec![0xAB; (2 * PIXELS * 4) as usize]).unwrap()
    }

    /// Encode one pass whose imageblock sample length is `sample_length` over a new attachment,
    /// cleared to zero and stored, by `encode`, run it, and answer the attachment.
    fn run(&self, sample_length: u64, encode: impl FnOnce(&mut RenderCommandEncoder)) -> Texture {
        let mut descriptor = TextureDescriptor::texture_2d(
            PixelFormat::R32Uint,
            AREA[0].into(),
            AREA[1].into(),
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target.clone());
        pass.color_attachments[0].load_action = LoadAction::Clear;
        pass.color_attachments[0].store_action = StoreAction::Store;
        pass.tile_width = TILE.into();
        pass.tile_height = TILE.into();
        pass.imageblock_sample_length = sample_length;
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        target
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

/// The texels of `target`, an attachment of the pass's extent, in row order.
fn texels(target: &Texture) -> Vec<u32> {
    let [width, height] = AREA.map(u64::from);
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
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// The buffers a draw or dispatch reads, held until the pass runs.
#[derive(Default)]
struct Held(Vec<Buffer>);

impl Held {
    /// The pass's render area at vertex buffer 1 (as floats) and tile buffer 1 is set by each
    /// dispatch; the vertex one once.
    fn bind_area(&mut self, f: &Fixture, encoder: &mut RenderCommandEncoder) {
        let size: Vec<u8> = [AREA[0] as f32, AREA[1] as f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let size = buffer_with(&f.device, &size).unwrap();
        encoder.set_vertex_buffer(Some(&size), 0, 1).unwrap();
        self.0.push(size);
    }

    /// Draw `splat`'s rectangle `rect`, setting bit `s`.
    fn splat(
        &mut self,
        f: &Fixture,
        encoder: &mut RenderCommandEncoder,
        splat: &RenderPipelineState,
        s: u32,
        rect: Rect,
    ) -> Result<(), Error> {
        let corners: Vec<u8> = rect
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        let corners = buffer_with(&f.device, &corners).unwrap();
        let index = f.words(&[s]);
        encoder.set_render_pipeline_state(splat)?;
        encoder.set_vertex_buffer(Some(&corners), 0, 0)?;
        encoder.set_fragment_buffer(Some(&index), 0, 0)?;
        let drawn = encoder.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0);
        self.0.extend([corners, index]);
        drawn
    }

    /// A tile dispatch of `state`, one thread per pixel of the tile, with `argument` at tile
    /// buffer 1 and `out` at tile buffer 0.
    fn dispatch(
        &mut self,
        f: &Fixture,
        encoder: &mut RenderCommandEncoder,
        state: &RenderPipelineState,
        argument: &[u32],
        out: Option<&Buffer>,
    ) -> Result<(), Error> {
        let argument = f.words(argument);
        encoder.set_render_pipeline_state(state)?;
        encoder.set_tile_buffer(Some(&argument), 0, 1)?;
        encoder.set_tile_buffer(out, 0, 0)?;
        let dispatched = encoder.dispatch_threads_per_tile(Size::new(TILE.into(), TILE.into(), 1));
        self.0.push(argument);
        dispatched
    }
}

/// The splats' rectangles: splat `s` at `(7s mod 32, 5s mod 20)`, 9x5 pixels, cut to the pass.
fn splat_rects(count: u32) -> Vec<Rect> {
    (0..count)
        .map(|s| {
            let (x, y) = ((7 * s) % 32, (5 * s) % 20);
            [x, y, (x + 9).min(AREA[0]), (y + 5).min(AREA[1])]
        })
        .collect()
}

/// The resolve of every pixel's cell after a clear to `seed` and the splats `first..` of
/// `rects`: two words a pixel.
fn resolved(seed: [u32; 2], rects: &[Rect], first: u32) -> Vec<u32> {
    let mut cells = vec![seed; PIXELS as usize];
    for (s, &[x0, y0, x1, y1]) in (first..).zip(&rects[first as usize..]) {
        for y in y0..y1 {
            for x in x0..x1 {
                let cell = &mut cells[(y * AREA[0] + x) as usize];
                cell[(s / 32) as usize] |= 1 << (s % 32);
            }
        }
    }
    cells.concat()
}

/// The fragment imageblock's state, or `None` (said) where the device makes none.
fn splat_or_skip(f: &Fixture) -> Option<RenderPipelineState> {
    match f.splat_state() {
        Ok(state) => Some(state),
        Err(Error::Unsupported(why)) => {
            eprintln!("cfit_ib_splat is not made on this device: {why}");
            None
        }
        Err(error) => panic!("cfit_ib_splat: {error}"),
    }
}

#[test]
fn the_splat_shape_sets_each_pixels_bits_between_tile_dispatches() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let Some(splat) = splat_or_skip(&f) else {
            return;
        };
        let clear = f.tile_state("clear");
        let resolve = f.tile_state("resolve");
        // Splats 0 to 39, both words' bits.
        let rects = splat_rects(40);
        let seed = [0, 1 << 31];
        let outs = [f.out(), f.out()];
        // Clear, every splat, resolve: the seed and every splat's bits.
        f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            held.dispatch(&f, encoder, &clear, &seed, None).unwrap();
            for (s, &rect) in (0..).zip(&rects) {
                held.splat(&f, encoder, &splat, s, rect).unwrap();
            }
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[0]))
                .unwrap();
        });
        assert_eq!(words(&outs[0]), resolved(seed, &rects, 0), "every splat");
        // Splats 0 to 19, the clear, splats 20 to 39, resolve: the later splats' bits alone.
        f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            for (s, &rect) in (0..).zip(&rects) {
                if s == 20 {
                    held.dispatch(&f, encoder, &clear, &seed, None).unwrap();
                }
                held.splat(&f, encoder, &splat, s, rect).unwrap();
            }
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[1]))
                .unwrap();
        });
        assert_eq!(
            words(&outs[1]),
            resolved(seed, &rects, 20),
            "the splats after the clear"
        );
    });
}

#[test]
fn a_tile_functions_imageblock_persists_through_the_pass_and_starts_at_zero() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let clear = f.tile_state("clear");
        let add = f.tile_state("add");
        let resolve = f.tile_state("resolve");
        let halves = f.tile_state("resolve_halves");
        let outs = [f.out(), f.out(), f.out()];
        let every = |cell: [u32; 2]| -> Vec<u32> { (0..PIXELS).flat_map(|_| cell).collect() };
        // Clear, add, resolve: each dispatch reads what the one before it wrote.
        f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &clear, &[0x100, 7], None)
                .unwrap();
            held.dispatch(&f, encoder, &add, &[0x1000], None).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[0]))
                .unwrap();
        });
        assert_eq!(words(&outs[0]), every([0x1100, 7]), "clear, add, resolve");
        // A new pass's cells, after the pass above wrote every cell: zero.
        f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[1]))
                .unwrap();
        });
        assert_eq!(words(&outs[1]), every([0, 0]), "a pass's start");
        // The same bytes through two halves and a word.
        f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &clear, &[0x0002_0001, 9], None)
                .unwrap();
            held.dispatch(&f, encoder, &halves, &AREA, Some(&outs[2]))
                .unwrap();
        });
        assert_eq!(
            words(&outs[2]),
            every([0x0002_0001, 9]),
            "another layout of the size"
        );
    });
}

#[test]
fn a_tile_state_answers_its_imageblocks_sample_length_and_a_pass_of_less_keeps_none_of_it() {
    on_device(|device| {
        let f = Fixture::new(device);
        let clear = f.tile_state("clear");
        let resolve = f.tile_state("resolve");
        assert_eq!(clear.imageblock_sample_length(), Ok(SAMPLE_LENGTH));
        assert_eq!(f.tile_state("aliased").imageblock_sample_length(), Ok(4));
        // A pass of no imageblock sample length gives the attachment's four bytes, rounded up
        // to 8: the cell, at byte 8, keeps no word.
        let out = f.out();
        f.run(0, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &clear, &[0x100, 7], None)
                .unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&out))
                .unwrap();
        });
        assert_eq!(
            words(&out),
            vec![0; 2 * PIXELS as usize],
            "a pass of 4 bytes"
        );
    });
}

/// A tile function whose layout is aliased onto the attachment, no larger than its texel, runs
/// beside the pass's imageblock storage, made by a draw or a dispatch, in either order, as Metal
/// does without its API validation (a probe): the attachment takes its write, and the storage's
/// cells keep what reached them, the resolve reading them after it.
#[test]
fn an_aliased_layout_of_the_attachments_bytes_runs_beside_the_pass_imageblock() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let clear = f.tile_state("clear");
        let aliased = f.tile_state("aliased");
        let resolve = f.tile_state("resolve");
        let outs = [f.out(), f.out(), f.out(), f.out()];
        let every = |word: u32| -> Vec<u32> { vec![word; PIXELS as usize] };
        let cells = |cell: [u32; 2]| -> Vec<u32> { (0..PIXELS).flat_map(|_| cell).collect() };
        // A dispatch reaches the storage, the aliased one writes the texels, the resolve reads.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &clear, &[0x100, 7], None)
                .unwrap();
            held.dispatch(&f, encoder, &aliased, &[5], None).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[0]))
                .unwrap();
        });
        assert_eq!(texels(&target), every(5), "the aliased write after a clear");
        assert_eq!(words(&outs[0]), cells([0x100, 7]), "the clear's cells");
        // The aliased dispatch first: the storage it never reached starts at zero.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.dispatch(&f, encoder, &aliased, &[5], None).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[1]))
                .unwrap();
        });
        assert_eq!(texels(&target), every(5), "the aliased write first");
        assert_eq!(words(&outs[1]), cells([0, 0]), "the cells after it");
        let Some(splat) = splat_or_skip(&f) else {
            return;
        };
        let rect = [0, 0, AREA[0], AREA[1]];
        // A draw reaches the storage (and writes its cell's low word to the texel), then the
        // aliased dispatch writes the texels.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            held.splat(&f, encoder, &splat, 3, rect).unwrap();
            held.dispatch(&f, encoder, &aliased, &[5], None).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[2]))
                .unwrap();
        });
        assert_eq!(texels(&target), every(5), "the aliased write after a draw");
        assert_eq!(words(&outs[2]), cells([1 << 3, 0]), "the draw's cells");
        // The aliased dispatch, then a draw that reaches the storage and writes the texels.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            held.dispatch(&f, encoder, &aliased, &[5], None).unwrap();
            held.splat(&f, encoder, &splat, 3, rect).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[3]))
                .unwrap();
        });
        assert_eq!(
            texels(&target),
            every(1 << 3),
            "a draw after the aliased write"
        );
        assert_eq!(words(&outs[3]), cells([1 << 3, 0]), "its cells");
    });
}

/// A draw whose fragment function views another master of the splat cell's eight bytes and
/// alignment (`cfb_ib_splat_swapped`, its two words swapped) reads and writes the bytes the
/// splat's draws did, each at its own offsets, in either order, as Metal's does without its API
/// validation (a probe on the M3 Max and an M4; the validation stops it, "input image-block layout
/// must match the previous state image-block layout"): the swapped view's `lo` is the splat's `hi`,
/// and each draw's pixel takes its own `lo`.
#[test]
fn another_master_of_the_cells_size_and_alignment_views_the_same_bytes() {
    on_device_with(at_plan, |device| {
        let f = Fixture::new(device);
        let Some(splat) = splat_or_skip(&f) else {
            return;
        };
        let swapped = f.swapped_state().unwrap();
        let resolve = f.tile_state("resolve");
        let outs = [f.out(), f.out()];
        let rect = [0, 0, AREA[0], AREA[1]];
        let every = |word: u32| -> Vec<u32> { vec![word; PIXELS as usize] };
        let cells = |cell: [u32; 2]| -> Vec<u32> { (0..PIXELS).flat_map(|_| cell).collect() };
        // The splat's bit 3 (its `lo`), the swapped view's bit 5 (the splat's `hi`), the splat's
        // bit 36 (its `hi`, bit 4): the pixel takes the last draw's `lo`.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            held.splat(&f, encoder, &splat, 3, rect).unwrap();
            held.splat(&f, encoder, &swapped, 5, rect).unwrap();
            held.splat(&f, encoder, &splat, 36, rect).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[0]))
                .unwrap();
        });
        assert_eq!(texels(&target), every(1 << 3), "the splat's last draw");
        assert_eq!(
            words(&outs[0]),
            cells([1 << 3, (1 << 5) | (1 << 4)]),
            "the cells after the splat first"
        );
        // The swapped view first: its bit 5, the splat's bit 3, its bit 7.
        let target = f.run(SAMPLE_LENGTH, |encoder| {
            let mut held = Held::default();
            held.bind_area(&f, encoder);
            held.splat(&f, encoder, &swapped, 5, rect).unwrap();
            held.splat(&f, encoder, &splat, 3, rect).unwrap();
            held.splat(&f, encoder, &swapped, 7, rect).unwrap();
            held.dispatch(&f, encoder, &resolve, &AREA, Some(&outs[1]))
                .unwrap();
        });
        assert_eq!(
            texels(&target),
            every((1 << 5) | (1 << 7)),
            "the swapped view's last draw"
        );
        assert_eq!(
            words(&outs[1]),
            cells([1 << 3, (1 << 5) | (1 << 7)]),
            "the cells after the swapped view first"
        );
    });
}
