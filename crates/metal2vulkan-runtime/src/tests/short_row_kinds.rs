//! A blit between a buffer and a depth, stencil, buffer-backed or two-plane extended-range
//! texture at a `bytesPerRow` below the row runs as plain Metal's blit runs it: each row at
//! `y x bytesPerRow`, read whole from there into the texture, and written there out of it in
//! turn, the later row's bytes standing, one of the orders plain Metal writes such rows in (an
//! Apple M3 Max under macOS 27.0.1, `metal2vulkan-conformance/probes/short_row_kinds.m`). Each was
//! refused: `Copy(BytesPerRow)`, or `NotImplemented(Format)` for the two-plane texture.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, Device, Origin, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's width and height.
const SIDE: [u64; 2] = [32, 4];

/// A kind of texture: its format, the bytes a texel moves in a blit, the blit's option, and
/// whether it lies over a buffer.
struct Kind {
    pf: PixelFormat,
    unit: u64,
    option: BlitOption,
    over_buffer: bool,
}

const KINDS: [Kind; 6] = [
    Kind {
        pf: PixelFormat::Depth16Unorm,
        unit: 2,
        option: BlitOption::None,
        over_buffer: false,
    },
    Kind {
        pf: PixelFormat::Stencil8,
        unit: 1,
        option: BlitOption::None,
        over_buffer: false,
    },
    Kind {
        pf: PixelFormat::Depth32FloatStencil8,
        unit: 4,
        option: BlitOption::DepthFromDepthStencil,
        over_buffer: false,
    },
    Kind {
        pf: PixelFormat::Depth32FloatStencil8,
        unit: 1,
        option: BlitOption::StencilFromDepthStencil,
        over_buffer: false,
    },
    Kind {
        pf: PixelFormat::RGBA8Unorm,
        unit: 4,
        option: BlitOption::None,
        over_buffer: true,
    },
    Kind {
        pf: PixelFormat::RGB10A82PXR10,
        unit: 4,
        option: BlitOption::None,
        over_buffer: false,
    },
];

/// A byte unique to `index` below 2^24.
fn pattern(index: u64) -> u8 {
    let mut x = (index as u32).wrapping_mul(2654435761) ^ 40503;
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x ^ (x >> 24)) as u8
}

/// `length` bytes unique to their offset, as `kind` holds them: a float depth's words, each its
/// own value in [0, 1) (no copy changes one), other kinds' bytes as they come.
fn bytes(kind: &Kind, length: u64) -> Vec<u8> {
    match kind.pf == PixelFormat::Depth32FloatStencil8 && kind.unit == 4 {
        true => (0..length / 4)
            .flat_map(|word| ((word + 1) as f32 / 65536.0).to_le_bytes())
            .collect(),
        false => (0..length).map(pattern).collect(),
    }
}

/// The kind's texture; for one over a buffer, the buffer too, held as long as the texture.
fn texture(device: &Device, kind: &Kind) -> (Texture, Option<Buffer>) {
    let [width, height] = SIDE;
    let mut descriptor = TextureDescriptor::texture_2d(kind.pf, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    if !kind.over_buffer {
        descriptor.resource_options = ResourceOptions::private();
        return (device.new_texture(&descriptor).unwrap(), None);
    }
    let row = width * kind.unit;
    let backing = device
        .new_buffer(row * height, ResourceOptions::shared())
        .unwrap();
    descriptor.resource_options = backing.resource_options();
    let texture = backing.new_texture(&descriptor, 0, row).unwrap();
    (texture, Some(backing))
}

/// `length` bytes of the `Shared` buffer `buffer`, whose commands all completed.
fn read(buffer: &Buffer, length: u64) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of at least `length` bytes whose commands all completed.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length as usize) }.to_vec()
}

/// The pitches below the row: half of it, and half plus a texel.
fn pitches(row: u64, unit: u64) -> [u64; 2] {
    [row / 2, row / 2 + unit]
}

#[test]
fn a_copy_in_at_a_pitch_below_the_row_reads_each_row_from_its_offset() {
    let scratch = Scratch::new("short-row-kinds-in");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let [width, height] = SIDE;
    let (size, at) = (Size::new(width, height, 1), Origin::new(0, 0, 0));
    for kind in &KINDS {
        let row = width * kind.unit;
        for bpr in pitches(row, kind.unit) {
            let what = format!("{:?} {:?} at {bpr}", kind.pf, kind.option);
            let (texture, _backing) = texture(&device, kind);
            let source = bytes(kind, bpr * (height - 1) + row);
            let source = device
                .new_buffer_with_bytes(&source, ResourceOptions::shared())
                .unwrap();
            let out = device
                .new_buffer(row * height, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            (blit.copy_from_buffer_to_texture(
                &source,
                0,
                bpr,
                bpr * height,
                size,
                &texture,
                0,
                0,
                at,
                kind.option,
            ))
            .unwrap_or_else(|error| panic!("{what}: the copy in answered {error:?}"));
            (blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                at,
                size,
                &out,
                0,
                row,
                row * height,
                kind.option,
            ))
            .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{what}");
            let source = read(&source, bpr * (height - 1) + row);
            let texels = read(&out, row * height);
            for y in 0..height {
                let (got, from) = ((y * row) as usize, (y * bpr) as usize);
                assert_eq!(
                    &texels[got..got + row as usize],
                    &source[from..from + row as usize],
                    "{what}: row {y}"
                );
            }
        }
    }
}

#[test]
fn a_copy_out_at_a_pitch_below_the_row_leaves_the_later_rows_of_every_kind() {
    let scratch = Scratch::new("short-row-kinds-out");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let [width, height] = SIDE;
    let (size, at) = (Size::new(width, height, 1), Origin::new(0, 0, 0));
    for kind in &KINDS {
        let row = width * kind.unit;
        for bpr in pitches(row, kind.unit) {
            let what = format!("{:?} {:?} at {bpr}", kind.pf, kind.option);
            let (texture, _backing) = texture(&device, kind);
            let texels = bytes(kind, row * height);
            let source = device
                .new_buffer_with_bytes(&texels, ResourceOptions::shared())
                .unwrap();
            let span = bpr * (height - 1) + row;
            let blank = vec![0xccu8; (span + 64) as usize];
            let out = device
                .new_buffer_with_bytes(&blank, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            (blit.copy_from_buffer_to_texture(
                &source,
                0,
                row,
                row * height,
                size,
                &texture,
                0,
                0,
                at,
                kind.option,
            ))
            .unwrap();
            (blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                at,
                size,
                &out,
                0,
                bpr,
                bpr * height,
                kind.option,
            ))
            .unwrap_or_else(|error| panic!("{what}: the copy out answered {error:?}"));
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "{what}");
            // The rows in turn: each later row over the earlier's.
            let mut expected = blank.clone();
            for y in 0..height {
                let (at, from) = ((y * bpr) as usize, (y * row) as usize);
                expected[at..at + row as usize].copy_from_slice(&texels[from..from + row as usize]);
            }
            assert_eq!(read(&out, span + 64), expected, "{what}");
        }
    }
}
