//! The formats the PVRTC decode answers for.

use super::PvrtcDecode;
use crate::metal::PixelFormat;

/// The eight PVRTC formats decode, and no other format does.
#[test]
fn the_pvrtc_formats_alone_decode() {
    use PixelFormat as P;
    let decoded: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| PvrtcDecode::of(pf).is_some())
        .collect();
    assert_eq!(
        decoded,
        [
            P::PVRTCRGB2BPP,
            P::PVRTCRGB2BPPSRGB,
            P::PVRTCRGB4BPP,
            P::PVRTCRGB4BPPSRGB,
            P::PVRTCRGBA2BPP,
            P::PVRTCRGBA2BPPSRGB,
            P::PVRTCRGBA4BPP,
            P::PVRTCRGBA4BPPSRGB,
        ]
    );
}
