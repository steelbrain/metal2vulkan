//! A fetch of a blended 16-bit float attachment without fast math, on the device the cfg selects,
//! under the validation layer: the owned fixture `fbl_fetch_blend16_safe`, as the parent's
//! `fbl_fetch_blend16` but compiled without fast math. Color 0 is a `SIDE`² private `RGBA16Float`
//! target blended ([`blends`]) with a zero factor beside each other factor the guest's blends read,
//! either way round, and two zero factors, under each operation, for color and alpha alike; color
//! 1 an `RGBA16Float` target of mask none. Each function reads color 0 and returns, at pixel
//! `i = SIDE · y + x`, a loaded source chosen by the fetched alpha (`src[i + SIDE²]` where it
//! exceeds 2, else `src[i]`): `fbl_safe_loaded` a `half4`, `fbl_safe_loaded_float` a `float4`. One
//! draw of the fullscreen triangle under each blend.
//!
//! Metal's answer (plain Metal on the M3 Max under macOS 27.0.1, our own MSL of this
//! shape, each operation with a zero factor beside each factor, color and alpha, into
//! `RGBA16Float` and `RG16Float` from `half4` and `float4` outputs, three seeds of 16,384 pixels
//! whose sources and texels hold signed zeros, infinities and NaNs: every lane): each product
//! rounded in the output's arithmetic, a zero factor's kept (`x · 0`: a NaN where `x` is an
//! infinity or a NaN, else a zero of the signed operand's sign), their IEEE 754 sum rounded (`-0`
//! only where both terms are `-0`); a `float4` output's result converted to `half` toward zero;
//! `SourceAlphaSaturated` the minimum of the source alpha and one minus the destination's, a NaN
//! operand giving the other and two zeros `+0`; every NaN stored as `0x7e00`. The texels and
//! sources are those of the parent's sweep, but one in four a special: a signed zero, an infinity,
//! a NaN (the texels' of either sign) or one, so a zero factor meets each and the saturated factor
//! meets a destination alpha of one.
//!
//! Where the draw announces none of [`super::NOTES`], every channel is Metal's, a channel Metal
//! stores as a NaN any NaN where the draw announces `device-nan-encoding`: a device not measured
//! to store the blend's NaN as Metal's in a storage image (ANV, which stores `0xfe00`; before the
//! note, its draws stored that unannounced). On a device that fetches through the pixel interlock
//! (RADV, ANV, NVIDIA, MoltenVK) the function computes Metal's blend itself, exactly, announcing
//! of [`super::NOTES`] those [`super::interlocked_notes`] names alone (the store's conversion of a
//! `float` output), which is asserted. Through input attachments
//! (lavapipe) the device blends, one and zero, added, too (announcing
//! `half-output-precision`, which Metal's kept product by zero there needs); what it announces and
//! how many channels are Metal's is printed.
//! Each draw prints a `FETCH-BLEND16-ZERO` line naming its function and blend. Before the shader
//! kept a zero factor's product, it dropped the term, which kept neither the NaN of an infinite or
//! NaN operand nor the zero sum's sign.

use super::{
    h, half_bits, stores_metals_blend, Blending, Case, Data, Sweep, Words, DIFFERENCE_COLOR, TEXELS,
};
use metal2vulkan::metal::{BlendFactor, BlendOperation};

/// The safe-math fixture's sweep.
pub(super) const SAFE: Sweep = Sweep {
    fixture: "fbl_fetch_blend16_safe",
    vertex: "fbl_safe_vertex",
    cases: [
        Case {
            entry: "fbl_safe_loaded",
            float: false,
        },
        Case {
            entry: "fbl_safe_loaded_float",
            float: true,
        },
    ],
    data,
    metal,
};

/// A zero factor beside each other factor, either way round, and two zero factors, under each
/// operation, for color and alpha, under the blend color [`DIFFERENCE_COLOR`].
fn blends() -> Vec<Blending> {
    use BlendFactor as F;
    let others = [
        F::One,
        F::SourceColor,
        F::OneMinusSourceAlpha,
        F::DestinationColor,
        F::OneMinusDestinationAlpha,
        F::SourceAlphaSaturated,
        F::BlendColor,
    ];
    let pairs = (others.iter())
        .flat_map(|&o| [(F::Zero, o), (o, F::Zero)])
        .chain([(F::Zero, F::Zero)]);
    let operations = [
        BlendOperation::Add,
        BlendOperation::Subtract,
        BlendOperation::ReverseSubtract,
    ];
    pairs
        .flat_map(|(source, destination)| {
            operations.map(|operation| Blending {
                source,
                destination,
                operation,
                color: DIFFERENCE_COLOR,
            })
        })
        .collect()
}

/// The texels' specials, as `half` bits: signed zeros, infinities, NaNs and one.
const TEXEL_SPECIALS: [u16; 7] = [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x3c00];

/// The sources' specials: signed zeros, infinities, a NaN and one.
const SOURCE_SPECIALS: [f64; 6] = [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0];

/// The parent's data, but one texel channel and one source channel in four a special.
fn data(case: Case) -> Data {
    let mut words = Words(0x5a4e_0fac_7031_b1e5);
    let texels: Vec<u16> = (0..TEXELS * 4)
        .map(|k| {
            let w = words.next();
            if w & 3 == 0 {
                return TEXEL_SPECIALS[((w >> 2) % 7) as usize];
            }
            let top = if k % 4 == 3 { 1 } else { 2 };
            half_bits(words.half(top), false)
        })
        .collect();
    let sources = (0..TEXELS * 2)
        .map(|_| {
            [2, 2, 2, 1].map(|top| {
                let w = words.next();
                if w & 3 == 0 {
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

/// Metal's `min` in `SourceAlphaSaturated`: a NaN operand gives the other, two zeros `+0`.
fn metal_min(x: f64, y: f64) -> f64 {
    match (x.is_nan(), y.is_nan()) {
        (true, _) => y,
        (_, true) => x,
        _ if x == 0.0 && y == 0.0 => 0.0,
        _ => x.min(y),
    }
}

/// Metal's blend `b` without fast math of `s` over the texel `d`, as `half` bits:
/// each product rounded in the output's arithmetic, a zero factor's kept, the sum rounded, a
/// `float` result converted toward zero. One term is a zero or a NaN, so the `f64` sum is exact.
fn metal(case: Case, b: Blending, s: [f64; 4], d: [f64; 4]) -> [u16; 4] {
    use BlendFactor as F;
    let r = |x: f64| match case.float {
        true => f64::from(x as f32),
        false => h(x),
    };
    std::array::from_fn(|c| {
        let factor = |f: BlendFactor| match f {
            F::Zero => 0.0,
            F::One => 1.0,
            F::SourceColor => s[c],
            F::OneMinusSourceAlpha => r(1.0 - s[3]),
            F::DestinationColor => d[c],
            F::OneMinusDestinationAlpha => r(1.0 - d[3]),
            F::SourceAlphaSaturated if c == 3 => 1.0,
            F::SourceAlphaSaturated => metal_min(s[3], r(1.0 - d[3])),
            F::BlendColor => r(f64::from(b.color[c])),
            _ => panic!("no factor {f:?} here"),
        };
        // Every product of two `half`s, or of a `float` and a `half`, is exact in `f64`.
        let ps = r(s[c] * factor(b.source));
        let pd = r(d[c] * factor(b.destination));
        let v = r(match b.operation {
            BlendOperation::Add => ps + pd,
            BlendOperation::Subtract => ps - pd,
            _ => pd - ps,
        });
        half_bits(v, case.float)
    })
}

/// Each draw's channels are Metal's where it announces none of [`super::NOTES`], and on a device
/// that fetches through the pixel interlock it announces those [`super::interlocked_notes`] names
/// alone (the module docs).
#[test]
fn a_safe_blended_fetch_keeps_a_zero_factors_product() {
    stores_metals_blend(&SAFE, &blends(), "FETCH-BLEND16-ZERO");
}
