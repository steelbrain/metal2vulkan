//! A tile dispatch's cells past its pass's attachments (`render/tile_dispatch.rs`) on the device
//! the cfg selects, under the validation layer. The owned fixture `cfb_tile_norm`'s functions
//! `cfb_tn_read4` and `cfb_tn_write4` read each cell of an `RGBA8Unorm` attachment into a buffer,
//! or write the buffer into the cells, one thread a cell, in a pass of 16x16 tiles over an
//! attachment of 10x20 pixels: one tile across and two down, so the cells past the attachment's
//! right edge, and the second tile's past its bottom edge, are cells of no pixel. As the command
//! replays measured them on an M4 (plain Metal; the `ptile3` variants, this pass's shape, and the
//! `ptile` and `ptile2` ones, `RGBA16Float` attachments of 282x88 in 32x32 tiles):
//!
//! - when the pass begins, a loaded attachment's cells past it hold its texel nearest each,
//!   clamped to its edge, and a cleared one's the clear color;
//! - a tile function's writes to them are what the pass's next dispatch reads.
//!
//! [`a_shadowed_attachment_s_tiles_past_its_render_area_hold_its_texels`] runs a draw and a
//! dispatch over a render area smaller than a shadowed attachment, whose tiles reach past the area
//! inside it.

use super::no_copy_textures::{floats, rects, staged};
use super::tile_states::on_device;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{StageSource, TileRenderPipelineDescriptor};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, Device, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor,
};

/// The tile's side.
const TILE: u64 = 16;

/// The attachment's width and height.
const WIDTH: u64 = 10;
const HEIGHT: u64 = 20;

/// The cells of the pass's tiles, one across and two down: cell `(x, y)` is `cfb_tn_read4`'s
/// element `16 y + x`.
const CELLS: u64 = TILE * 2 * TILE;

/// The codes the clear color `(0.25, 0.5, 0.75, 1)` takes in `RGBA8Unorm` (Metal's, measured).
const CLEARED: [u8; 4] = [64, 128, 191, 255];

/// The codes of the loaded attachment's texel `(x, y)`: red, green, blue and alpha.
fn texel(x: u64, y: u64) -> [u8; 4] {
    [(16 * x + 1) as u8, (8 * y + 2) as u8, 200, 255]
}

/// The tile state of `cfb_tile_norm`'s function `entry` over one `RGBA8Unorm` color.
fn state(device: &Device, entry: &str) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("cfb_tile_norm.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    let mut descriptor =
        TileRenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
    descriptor.threadgroup_size_matches_tile_size = true;
    device
        .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A new attachment holding [`texel`] of each pixel.
fn attachment(device: &Device) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).flat_map(move |x| texel(x, y)))
        .collect();
    let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
    texture
        .replace_region(region, 0, 0, &bytes, WIDTH * 4, WIDTH * HEIGHT * 4)
        .unwrap();
    texture
}

/// Every texel's codes of `target`, in row order.
fn texels(target: &Texture) -> Vec<[u8; 4]> {
    let mut bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let region = Region::new_2d(0, 0, WIDTH, HEIGHT);
    target
        .get_bytes(&mut bytes, WIDTH * 4, WIDTH * HEIGHT * 4, region, 0, 0)
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|t| [t[0], t[1], t[2], t[3]])
        .collect()
}

/// Each cell's codes in a buffer `cfb_tn_read4` filled: each value is a code's, `c / 255`, as the
/// device converts it to `float`, within `f32::EPSILON` of the quotient, relative (the RTX 5080 Laptop GPU
/// reads 200's as 0.7843138, the quotient's next `f32` up, as a product by `1 / 255` gives it;
/// Vulkan states the conversion, not its precision).
fn codes(buffer: &Buffer) -> Vec<[u8; 4]> {
    let bytes = read_buffer(buffer, 0, CELLS * 16).unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let mut codes = [0u8; 4];
            for (k, lane) in cell.as_chunks::<4>().0.iter().enumerate() {
                let value = f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]);
                let code = (value * 255.0).round();
                let quotient = code / 255.0;
                assert!(
                    (0.0..=255.0).contains(&code)
                        && (value - quotient).abs() <= f32::EPSILON * quotient,
                    "cell {i} channel {k}: {value} is no code's"
                );
                codes[k] = code as u8;
            }
            codes
        })
        .collect()
}

/// Run a pass over `target`, loaded with `load` (cleared to `(0.25, 0.5, 0.75, 1)`) and stored,
/// in 16x16 tiles, that dispatches each state in turn, a thread a cell, its buffer at tile buffer
/// 0.
fn run(
    device: &Device,
    target: &Texture,
    load: LoadAction,
    dispatches: &[(&RenderPipelineState, &Buffer)],
) {
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachments[0].texture = Some(target.clone());
    pass.color_attachments[0].load_action = load;
    pass.color_attachments[0].store_action = StoreAction::Store;
    pass.color_attachments[0].clear_color = ClearColor::new(0.25, 0.5, 0.75, 1.0);
    pass.tile_width = TILE;
    pass.tile_height = TILE;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    for (state, buffer) in dispatches {
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_tile_buffer(Some(buffer), 0, 0).unwrap();
        encoder
            .dispatch_threads_per_tile(Size::new(TILE, TILE, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A pass's first dispatch reads, in the cells past a loaded attachment, its texel nearest each
/// (clamped to its edge), and past a cleared one the clear color (Metal's `ptile3-load`); the
/// attachment keeps what it held. A cleared channel holds either of the two codes nearest the
/// clear's scaled value where the device's clear is not Metal's conversion (no
/// `DriverFact::ColorClearIsMetals`), as Vulkan lets a clear store (the RTX 5080 Laptop GPU stores
/// 0.5, 127.5 codes, as 127, Metal 128).
#[test]
fn cells_past_the_attachment_hold_metals_values_when_the_pass_begins() {
    on_device(|device| {
        let read = state(device, "cfb_tn_read4");
        let either = !(device.properties().driver_facts)
            .contains(&metal2vulkan::device::DriverFact::ColorClearIsMetals);
        for load in [LoadAction::Load, LoadAction::Clear] {
            let want = |x: u64, y: u64| match load {
                LoadAction::Clear => CLEARED,
                _ => texel(x.min(WIDTH - 1), y.min(HEIGHT - 1)),
            };
            // Metal's codes, or the device's conversion of the clear.
            let holds = |have: [u8; 4], want: [u8; 4]| {
                have == want
                    || (load == LoadAction::Clear
                        && either
                        && (have.iter().zip([0.25f64, 0.5, 0.75, 1.0]))
                            .all(|(&h, c)| (f64::from(h) - c * 255.0).abs() < 1.0))
            };
            let target = attachment(device);
            let out = buffer_with(device, &vec![0xAB; (CELLS * 16) as usize]).unwrap();
            run(device, &target, load, &[(&read, &out)]);
            for (i, have) in codes(&out).into_iter().enumerate() {
                let (x, y) = (i as u64 % TILE, i as u64 / TILE);
                let want = want(x, y);
                assert!(
                    holds(have, want),
                    "{load:?}: cell ({x}, {y}): {have:?}, not {want:?}"
                );
            }
            for (i, have) in texels(&target).into_iter().enumerate() {
                let (x, y) = (i as u64 % WIDTH, i as u64 / WIDTH);
                let want = want(x, y);
                assert!(
                    holds(have, want),
                    "{load:?}: texel ({x}, {y}) kept: {have:?}, not {want:?}"
                );
            }
        }
    });
}

/// A dispatch's writes to the cells past the attachment are what the pass's next dispatch reads
/// there (Metal's `ptile3-persist` and `ptile`), and its writes inside it what the attachment
/// stores.
#[test]
fn writes_past_the_attachment_last_for_the_pass() {
    on_device(|device| {
        let write = state(device, "cfb_tn_write4");
        let read = state(device, "cfb_tn_read4");
        let written = |i: u64| {
            [
                (3 * i) as u8,
                (5 * i + 7) as u8,
                (i / 2) as u8,
                (255 - i % 256) as u8,
            ]
        };
        let values: Vec<u8> = (0..CELLS)
            .flat_map(|i| written(i).map(|code| f32::from(code) / 255.0))
            .flat_map(f32::to_le_bytes)
            .collect();
        let input = buffer_with(device, &values).unwrap();
        let out = buffer_with(device, &vec![0xAB; (CELLS * 16) as usize]).unwrap();
        let target = attachment(device);
        run(
            device,
            &target,
            LoadAction::Load,
            &[(&write, &input), (&read, &out)],
        );
        for (i, have) in codes(&out).into_iter().enumerate() {
            let (x, y) = (i as u64 % TILE, i as u64 / TILE);
            assert_eq!(have, written(i as u64), "cell ({x}, {y})");
        }
        for (i, have) in texels(&target).into_iter().enumerate() {
            let (x, y) = (i as u64 % WIDTH, i as u64 / WIDTH);
            assert_eq!(have, written(y * TILE + x), "texel ({x}, {y})");
        }
    });
}

/// The attachment of [`a_shadowed_attachment_s_tiles_past_its_render_area_hold_its_texels`]: its
/// width and height, and its rows' pitch, padded past their 160 bytes of texels.
const SHADOWED: [u64; 2] = [40, 40];
const SHADOWED_ROW: u64 = 176;

/// That pass's render area, one tile across and two down, whose tiles reach past it inside the
/// attachment, and the scissor rectangle of its draw inside it.
const AREA: [u64; 2] = [12, 20];
const SCISSOR: [u64; 4] = [3, 5, 6, 9];

/// The codes of that draw's color `(1, 0.2, 0.6, 1)`.
const DRAWN: [u8; 4] = [255, 51, 153, 255];

/// A pass that loads a shadowed attachment (a texture over a buffer's padded rows, the runtime's
/// `shadow.rs`) over a render area smaller than it, of a draw inside the area and then a dispatch
/// of `cfb_tn_read4`: the dispatch reads, in each cell of the tiles over the area, the drawn color
/// inside the draw's scissor rectangle and the attachment's texel elsewhere, past the area too,
/// where the dispatch's stores reach (`render/tile_dispatch.rs`): the shadow's image holds what
/// the dispatch reads, though the pass's draw reaches a part of it alone. The buffer holds the
/// drawn color inside the rectangle, and every other byte as it was.
#[test]
fn a_shadowed_attachment_s_tiles_past_its_render_area_hold_its_texels() {
    on_device(|device| {
        let [w, h] = SHADOWED;
        let len = SHADOWED_ROW * h;
        let mut bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        for y in 0..h {
            for x in 0..w {
                let at = (y * SHADOWED_ROW + x * 4) as usize;
                bytes[at..at + 4].copy_from_slice(&texel(x, y));
            }
        }
        let buffer = staged(device, &bytes);
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, w, h, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = buffer.resource_options();
        let target = buffer.new_texture(&d, 0, SHADOWED_ROW).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { target.vk_image() }.is_none(), "a shadow");
        let draw = rects(device, PixelFormat::RGBA8Unorm, false);
        let read = state(device, "cfb_tn_read4");
        let corners = staged(device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(device, &floats([1.0, 0.2, 0.6, 1.0]));
        let out = buffer_with(device, &vec![0xAB; (CELLS * 16) as usize]).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(target.clone());
        pass.color_attachments[0].load_action = LoadAction::Load;
        pass.color_attachments[0].store_action = StoreAction::Store;
        [pass.render_target_width, pass.render_target_height] = AREA;
        pass.tile_width = TILE;
        pass.tile_height = TILE;
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&draw).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        let [x, y, width, height] = SCISSOR;
        encoder
            .set_scissor_rect(ScissorRect::new(x, y, width, height))
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.set_render_pipeline_state(&read).unwrap();
        encoder.set_tile_buffer(Some(&out), 0, 0).unwrap();
        encoder
            .dispatch_threads_per_tile(Size::new(TILE, TILE, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let drawn = |x: u64, y: u64| {
            (SCISSOR[0]..SCISSOR[0] + SCISSOR[2]).contains(&x)
                && (SCISSOR[1]..SCISSOR[1] + SCISSOR[3]).contains(&y)
        };
        let want = |x: u64, y: u64| if drawn(x, y) { DRAWN } else { texel(x, y) };
        for (i, have) in codes(&out).into_iter().enumerate() {
            let (x, y) = (i as u64 % TILE, i as u64 / TILE);
            assert_eq!(have, want(x, y), "cell ({x}, {y})");
        }
        let mut expected = bytes.clone();
        for y in 0..h {
            for x in 0..w {
                let at = (y * SHADOWED_ROW + x * 4) as usize;
                expected[at..at + 4].copy_from_slice(&want(x, y));
            }
        }
        let contents = buffer.contents().unwrap();
        let mut now = vec![0u8; len as usize];
        // SAFETY: the buffer's `len` bytes, which no command uses once the command buffer
        // completed.
        unsafe { std::ptr::copy_nonoverlapping(contents.as_ptr(), now.as_mut_ptr(), now.len()) };
        let differ = (0..now.len()).find(|&i| now[i] != expected[i]);
        assert_eq!(differ, None, "the first byte of the buffer that differs");
    });
}
