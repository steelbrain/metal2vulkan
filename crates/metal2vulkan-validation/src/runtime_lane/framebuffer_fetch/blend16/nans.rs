//! A blended NaN of either sign or payload: the parent's fixtures (`fbl_fetch_blend16` under fast
//! math, `fbl_fetch_blend16_safe` without), their loaded `half4` and `float4` outputs, drawn over
//! texels whose color channels are, one in four, a NaN of either sign and of several payloads, an
//! infinity of either sign, a signed zero or one, from sources whose color channels are, one in
//! four, an infinity of either sign, a NaN, a signed zero or one, under blends whose factors are
//! none of them zero: the parent's one over one minus the source alpha under fast math, and
//! without it one over one, one over one minus the source alpha, the source color over the
//! destination color, the destination color over one minus the destination alpha and the blend
//! color over one, each added, subtracted and subtracted in reverse.
//!
//! Metal's answer (plain Metal on the M3 Max under macOS 27.0.1, our own MSL, an
//! `RGBA16Float` target blended by five such blends, `half4` and `float4` outputs, fast and safe
//! math, over NaN texels `0x7c01`, `0x7d55`, `0x7e00`, `0x7e01`, `0x7fff`, `0xfc01`, `0xfd55`,
//! `0xfe00` and `0xffff`, NaN sources of those bits, and infinities: every one of the 235,560 NaN
//! results stored as `0x7e00`, and each of the 4,080 sums by one and one of an infinity and a
//! finite operand stored as that infinity): the parent's model, every NaN `0x7e00`, the rounded
//! forms without fast math ([`super::zero`]'s). Before the blend's exact multiply-add took the
//! device's result for an infinite sum (`fma::float_fma_exact`), a `float4` output's infinite sum
//! stored a NaN (1,080 of 4,096 texels on the RTX 3090 and the 780M). A NaN of the function's blend
//! is the device's arithmetic's, which the storage image's store converts (RADV and NVIDIA keep its
//! sign and payload, ANV stores `0xfe00`): every draw announces `device-nan-encoding`, and a
//! channel Metal stores as a NaN holds any NaN. Each draw prints a `FETCH-BLEND16-NAN` line.

use super::{
    half_bits, stores_metals_blend, Blending, Case, Data, Sweep, Words, DIFFERENCE_COLOR, FAST,
    OVER, TEXELS,
};
use metal2vulkan::metal::{BlendFactor, BlendOperation};

/// The texels' specials, as `half` bits: NaNs of either sign and several payloads, infinities,
/// signed zeros and one.
const TEXEL_SPECIALS: [u16; 14] = [
    0x7c01, 0x7d55, 0x7e00, 0x7e01, 0x7fff, 0xfc01, 0xfd55, 0xfe00, 0xffff, 0x7c00, 0xfc00, 0x0000,
    0x8000, 0x3c00,
];

/// The sources' specials: infinities, a NaN, signed zeros and one.
const SOURCE_SPECIALS: [f64; 6] = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -0.0, 1.0];

/// The parent's data, but one color channel of a texel and of a source in four a special.
fn data(case: Case) -> Data {
    let mut words = Words(0x6e61_6e73_0b1e_7d16);
    let texels: Vec<u16> = (0..TEXELS * 4)
        .map(|k| {
            let w = words.next();
            if k % 4 != 3 && w & 3 == 0 {
                return TEXEL_SPECIALS[((w >> 2) % 14) as usize];
            }
            let top = if k % 4 == 3 { 1 } else { 2 };
            half_bits(words.half(top), false)
        })
        .collect();
    let sources = (0..TEXELS * 2)
        .map(|_| {
            [2, 2, 2, 1].map(|top| {
                let w = words.next();
                if top == 2 && w & 3 == 0 {
                    return SOURCE_SPECIALS[((w >> 2) % 6) as usize];
                }
                match case.float {
                    true => f64::from(words.float(top)),
                    false => words.half(top),
                }
            })
        })
        .collect();
    (texels, sources)
}

/// The fast-math fixture's sweep over the specials.
const FAST_NANS: Sweep = Sweep { data, ..FAST };

/// The safe-math fixture's sweep over the specials.
const SAFE_NANS: Sweep = Sweep {
    data,
    ..super::zero::SAFE
};

/// The safe sweep's blends: each pair of factors, none zero, under each operation, under the
/// blend color [`DIFFERENCE_COLOR`].
fn blends() -> Vec<Blending> {
    use BlendFactor as F;
    let pairs = [
        (F::One, F::One),
        (F::One, F::OneMinusSourceAlpha),
        (F::SourceColor, F::DestinationColor),
        (F::DestinationColor, F::OneMinusDestinationAlpha),
        (F::BlendColor, F::One),
    ];
    let operations = [
        BlendOperation::Add,
        BlendOperation::Subtract,
        BlendOperation::ReverseSubtract,
    ];
    pairs
        .iter()
        .flat_map(|&(source, destination)| {
            operations.map(|operation| Blending {
                source,
                destination,
                operation,
                color: DIFFERENCE_COLOR,
            })
        })
        .collect()
}

/// Without fast math: each draw's channels are Metal's, every NaN `0x7e00` (any NaN where the draw
/// announces `device-nan-encoding`), where it announces none of [`super::NOTES`], and on a device
/// that fetches through the pixel interlock it announces none of them (the module docs).
#[test]
fn a_safe_blended_nan_of_either_sign_or_payload_stores_metals_nan() {
    stores_metals_blend(&SAFE_NANS, &blends(), "FETCH-BLEND16-NAN");
}

/// [`a_safe_blended_nan_of_either_sign_or_payload_stores_metals_nan`] under fast math, one over
/// one minus the source alpha.
#[test]
fn a_fast_blended_nan_of_either_sign_or_payload_stores_metals_nan() {
    stores_metals_blend(&FAST_NANS, &[OVER], "FETCH-BLEND16-NAN");
}
