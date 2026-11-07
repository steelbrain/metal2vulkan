//! A blit's copy into a one-plane YCbCr texture at the row `MTLPixelFormatGetInfoForDevice`
//! describes ([`Device::pixel_format_info`](crate::Device::pixel_format_info): a byte a texel for
//! `YCBCR8_422_1P` and two for `YCBCR10_444_1P`, half the bytes the texel's copies move) is made
//! as plain Metal makes it: Metal's aligned path at a pitch below the row, each row read whole from
//! its own row's offset, the rows overlapping in the source (an Apple M3 Max under macOS 27.0.1,
//! `metal2vulkan-conformance/probes/short_row_blits.m`). It was refused `Copy(BytesPerRow)`. A
//! copy out at that row stays refused so: Metal writes the overlapping rows in no one order (900
//! copies out at half the row, 300 shapes of 12 formats in 3 processes: the later row's bytes
//! standing at some widths, every even row's at others, neither order at most, and 15 shapes
//! answering otherwise in another process).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, CopyProblem, Error, Origin, Problem, ResourceOptions, Size, TextureDescriptor,
};
use metal2vulkan::metal::PixelFormat;

/// The texture's width and height.
const SIDE: [u64; 2] = [32, 4];

/// A byte unique to `index` below 2^24.
fn pattern(index: u64) -> u8 {
    let mut x = (index as u32).wrapping_mul(2654435761) ^ 40503;
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x ^ (x >> 24)) as u8
}

#[test]
fn a_copy_into_a_one_plane_texture_at_its_described_row_reads_each_row_from_its_offset() {
    let scratch = Scratch::new("described-row-copies");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let [width, height] = SIDE;
    for (pf, unit) in [
        (PixelFormat::YCBCR84221P, 2),
        (PixelFormat::YCBCR104441P, 4),
    ] {
        let described = device.pixel_format_info(u64::from(pf.as_raw())).words[0];
        assert_eq!(described * 2, unit, "{pf:?}: the described texel");
        let (row, bpr) = (width * unit, width * described);
        let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
        descriptor.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&descriptor).unwrap();
        let source: Vec<u8> = (0..bpr * (height - 1) + row).map(pattern).collect();
        let source = (device.new_buffer_with_bytes(&source, ResourceOptions::shared())).unwrap();
        let out = (device.new_buffer(row * height, ResourceOptions::shared())).unwrap();
        let size = Size::new(width, height, 1);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let at = Origin::new(0, 0, 0);
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
            BlitOption::None,
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
            row,
            row * height,
            BlitOption::None,
        ))
        .unwrap();
        // Out at the described row: Metal's overlapping writes stand in no one order.
        let refused = blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            at,
            size,
            &out,
            0,
            bpr,
            bpr * height,
            BlitOption::None,
        );
        match refused {
            Err(Error::Invalid(invalid))
                if invalid.problem == Problem::Copy(CopyProblem::BytesPerRow) => {}
            other => panic!("{pf:?}: the copy out at the described row answered {other:?}"),
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let read = out.contents().unwrap();
        // SAFETY: a `Shared` buffer of `row * height` bytes whose commands all completed.
        let read = unsafe { std::slice::from_raw_parts(read.as_ptr(), (row * height) as usize) };
        for y in 0..height {
            let texels = &read[(y * row) as usize..((y + 1) * row) as usize];
            let expected: Vec<u8> = (y * bpr..y * bpr + row).map(pattern).collect();
            assert_eq!(texels, &expected[..], "{pf:?}: row {y}");
        }
    }
}
