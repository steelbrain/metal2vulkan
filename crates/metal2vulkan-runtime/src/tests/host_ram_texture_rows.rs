//! The regions of a copy between a guest RAM and a texture on the imports
//! (`host_ram/texture.rs`'s `joined_rows`): a row's piece that continues a region of the row
//! before (the same import, first texel and texels, its bytes the pitch on) is one more row of
//! it; any other piece starts a region, and with no pitch every piece is one. And a staged copy's
//! regions between the arena and the texture (`staged_regions`): one region of its rows at a
//! pitch of at most 2^31 - 1 bytes, else one a row.

use crate::blit_plan::BufferImageRegion;
use crate::copy_plan::ImageRegion;
use crate::host_ram::texture::{joined_rows, staged_regions, Rows};
use metal2vulkan::vk::ImageAspectFlags;

const PITCH: u64 = 1280;

/// A piece of row `y`: `texels` texels from texel `x`, from byte `offset` of import `slot`.
fn piece(slot: usize, offset: u64, (x, y): (u64, u64), texels: u64) -> Rows {
    Rows {
        slot,
        offset,
        x,
        y,
        texels,
        rows: 1,
    }
}

#[test]
fn a_row_s_piece_that_continues_the_row_before_s_joins_its_region() {
    // Rows 0 to 2 whole in import 0; row 3 across the import's end; rows 4 and 5 whole in
    // import 1.
    let pieces = [
        piece(0, 64, (0, 0), 300),
        piece(0, 64 + PITCH, (0, 1), 300),
        piece(0, 64 + 2 * PITCH, (0, 2), 300),
        piece(0, 64 + 3 * PITCH, (0, 3), 100),
        piece(1, 0, (100, 3), 200),
        piece(1, 800 + PITCH - 1200, (0, 4), 300),
        piece(1, 800 + 2 * PITCH - 1200, (0, 5), 300),
    ];
    let joined = joined_rows(&pieces, Some(PITCH));
    let three = Rows {
        rows: 3,
        ..pieces[0]
    };
    let two = Rows {
        rows: 2,
        ..pieces[5]
    };
    assert_eq!(joined, [three, pieces[3], pieces[4], two]);
    // No pitch: a region a piece.
    assert_eq!(joined_rows(&pieces, None), pieces);
}

#[test]
fn a_piece_off_the_region_s_import_texels_pitch_or_next_row_starts_its_own() {
    let first = piece(0, 0, (8, 0), 16);
    for other in [
        // Another import.
        piece(1, PITCH, (8, 1), 16),
        // Another first texel, or texels.
        piece(0, PITCH + 4, (9, 1), 16),
        piece(0, PITCH, (8, 1), 15),
        // Bytes off the pitch.
        piece(0, PITCH + 4, (8, 1), 16),
        // A row past the next.
        piece(0, 2 * PITCH, (8, 2), 16),
    ] {
        assert_eq!(
            joined_rows(&[first, other], Some(PITCH)),
            [first, other],
            "{other:?}"
        );
    }
    // Two regions side by side in a row, each joined by its own piece of the next.
    let left = piece(0, 0, (0, 0), 8);
    let right = piece(0, 32, (8, 0), 8);
    let pieces = [
        left,
        right,
        piece(0, PITCH, (0, 1), 8),
        piece(0, PITCH + 32, (8, 1), 8),
    ];
    let joined = joined_rows(&pieces, Some(PITCH));
    assert_eq!(
        joined,
        [Rows { rows: 2, ..left }, Rows { rows: 2, ..right }]
    );
}

#[test]
fn a_staged_copy_s_rows_past_a_pitch_of_2_31_minus_1_bytes_are_a_region_each() {
    // `VUID-vkCmdCopyBufferToImage-bufferRowLength-09108`,
    // `VUID-vkCmdCopyImageToBuffer-bufferRowLength-09108`: `bufferRowLength` times the texel's
    // bytes at most 2^31 - 1.
    const LIMIT: u64 = (1 << 31) - 1;
    const TEXEL: u64 = 4;
    let image = ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level: 1,
        layer: 2,
        layers: 1,
        origin: [4, 8, 0],
        size: [16, 3, 1],
    };
    // The widest pitch of whole texels within the limit: one region of the rows.
    let within = LIMIT / TEXEL * TEXEL;
    let joined = BufferImageRegion {
        buffer_offset: 0,
        row_length: (within / TEXEL) as u32,
        image_height: 0,
        image,
    };
    assert_eq!(staged_regions(image, within, TEXEL), [joined]);
    // A pitch of 2^31 bytes: a region a row, each at its row's byte of the arena's region.
    let past = 1 << 31;
    let regions = staged_regions(image, past, TEXEL);
    assert_eq!(regions.len(), 3);
    for (y, region) in (0u32..).zip(&regions) {
        assert!(u64::from(region.row_length) * TEXEL <= LIMIT, "{region:?}");
        let row = BufferImageRegion {
            buffer_offset: u64::from(y) * past,
            row_length: 0,
            image_height: 0,
            image: ImageRegion {
                origin: [4, 8 + y, 0],
                size: [16, 1, 1],
                ..image
            },
        };
        assert_eq!(*region, row);
    }
    // One row: its region, whatever the pitch.
    let one = ImageRegion {
        size: [16, 1, 1],
        ..image
    };
    for pitch in [64, past, 1 << 40] {
        let region = BufferImageRegion {
            buffer_offset: 0,
            row_length: 0,
            image_height: 0,
            image: one,
        };
        assert_eq!(staged_regions(one, pitch, TEXEL), [region], "{pitch}");
    }
}
