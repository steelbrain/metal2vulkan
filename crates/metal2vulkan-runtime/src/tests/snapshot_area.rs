//! `render/snapshot.rs`'s area rule: a draw that reads its pass's texture through a snapshot or
//! takes the split by its form's cost at the share of its scissors least favourable to it.

use crate::render::snapshot_within;

/// The pixels of the consumer's desktop framebuffer, 3662x2039.
const DESKTOP: u64 = 3662 * 2039;

/// A reader through no scissor over a private 4-byte target of the desktop's size takes the
/// snapshot, whatever it covers: covering the whole target the snapshot weighs 4.9 times the split
/// (20 bytes a pixel against the copy's 4, each beside a reader's fixed cost of a mebibyte), but
/// covering nothing the split weighs 29.5 times the snapshot. So do readers over 1080p and 720p
/// targets.
#[test]
fn a_reader_through_no_scissor_over_a_large_color_takes_the_snapshot() {
    assert!(snapshot_within(DESKTOP, DESKTOP * 4));
    assert!(snapshot_within(1920 * 1080, 1920 * 1080 * 4));
    assert!(snapshot_within(1280 * 720, 1280 * 720 * 4));
}

/// A 4-byte color's reader through its whole level takes the snapshot from 786,432 pixels
/// (1024x768) and an 8-byte one's from 65,536 (256x256), not one pixel fewer; a 16-byte one's at
/// any size.
#[test]
fn a_reader_through_its_whole_level_takes_the_snapshot_from_a_size_by_its_texel_bytes() {
    assert!(snapshot_within(786_432, 786_432 * 4));
    assert!(!snapshot_within(786_431, 786_431 * 4));
    assert!(snapshot_within(65_536, 65_536 * 8));
    assert!(!snapshot_within(65_535, 65_535 * 8));
    for pixels in [1, 64, 1024, 65_536, DESKTOP] {
        assert!(snapshot_within(pixels, pixels * 16), "{pixels}");
    }
}

/// A 4-byte color's reader takes the snapshot through scissors of up to two fifths of its level
/// at any size: over a 32x32 `RGBA8Unorm` target, through a quarter of it (256 pixels) but not
/// through the whole (1024).
#[test]
fn a_reader_through_two_fifths_of_a_small_level_takes_the_snapshot() {
    assert!(snapshot_within(256, 1024 * 4));
    assert!(!snapshot_within(1024, 1024 * 4));
    for level in [1024, 65_536, 1 << 20] {
        assert!(snapshot_within(level * 2 / 5, level * 4), "{level}");
    }
}

/// Every reader whose snapshot over its whole scissors weighs no more than the split's copy takes
/// the snapshot, as before the fixed costs were weighed.
#[test]
fn a_snapshot_no_dearer_than_the_copy_over_its_whole_scissors_is_taken() {
    for bytes in [0, 4, 400, 4096, 1 << 20, 30_000_000, 1 << 34] {
        for pixels in [0, bytes / 40, bytes / 20] {
            assert!(
                snapshot_within(pixels, bytes),
                "{pixels} pixels, {bytes} bytes"
            );
        }
    }
}
