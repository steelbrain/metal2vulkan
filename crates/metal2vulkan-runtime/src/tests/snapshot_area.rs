//! `render/snapshot.rs`'s area rule: a draw that reads its pass's texture through a snapshot or
//! takes the split by its form's cost at the share of its scissors least favourable to it, the
//! split's by the bytes its copy moves: a one-level, one-slice color's texels around the draw's
//! scissors, a pixel wider each way, or every texel of the color where it reads through none.

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

/// A 4-byte color's reader through its whole level, the split copying every texel, takes the
/// snapshot from 786,432 pixels (1024x768) and an 8-byte one's from 65,536 (256x256), not one
/// pixel fewer; a 16-byte one's at any size.
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

/// A reader through small scissors splits where its color's texels are narrow, the split
/// copying only the texels around them: through an 8x8 scissor (the copy's 10x10 texels) a 4-byte
/// color's reader splits and an 8-byte one's takes the snapshot; over a 1920x1080 4-byte color,
/// a reader through a 960x540 scissor splits and one through 1280x720 takes the snapshot.
#[test]
fn a_reader_through_small_scissors_weighs_the_copy_around_them() {
    assert!(!snapshot_within(64, 10 * 10 * 4));
    assert!(snapshot_within(64, 10 * 10 * 8));
    assert!(!snapshot_within(960 * 540, 962 * 542 * 4));
    assert!(snapshot_within(1280 * 720, 1282 * 722 * 4));
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
