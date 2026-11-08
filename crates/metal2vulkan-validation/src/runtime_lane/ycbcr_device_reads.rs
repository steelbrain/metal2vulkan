//! The device's own Y′CbCr conversion of a two-plane YCbCr texture, as the YCbCr lanes check it
//! (the runtime's `ycbcr_device.rs`). Where the device converts a texture's planes itself
//! ([`device_conversions`] counts it as the texture is made), each texel of its image is Vulkan's
//! conversion of them ([`vulkan_read`]) within [`DEVICE_BOUND`], and not Metal's fixed-point read:
//! a lane that holds such a texture's image to Metal's steps holds it to these instead. It is
//! announced (`ycbcr-filter-not-exact`) at the texture's first conversion, the first sampling of
//! its planes through the device's conversion ([`device_samples`] grows): a texture whose planes
//! every function reads in place, whose samples are the function's own and not its image's,
//! announces nothing. The runtime takes the device's conversion of a 4:2:0 or 4:2:2 format only at
//! Metal's midpoint through a linear chroma filter, whose weights Vulkan states as Metal's (3 and
//! 1 along a halved axis), so a lane's reconstruction is Metal's sum of weighted codes over 16,
//! unrounded.

use metal2vulkan::bind::{YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrRange};
use metal2vulkan_runtime::Device;

/// The largest distance of a channel, in units of `1/2048`, from Vulkan's conversion
/// ([`vulkan_read`]; before the sRGB curve, for an `_sRGB` twin) the device's may take, past the
/// `half` a converted texture's image stores it in: at most 0.038 measured on lavapipe (Mesa
/// 26.0.8, LLVM 21.1.8), whose 8-bit 4:4:4 layout is the one it converts, every texel of the
/// lanes' surfaces.
pub(super) const DEVICE_BOUND: f64 = 0.04;

/// The textures the device has converted itself so far
/// (`WorkCounts::ycbcr_device_conversions`): a texture made while it grows is the device's.
pub(super) fn device_conversions(device: &Device) -> u64 {
    device.work_counts().ycbcr_device_conversions
}

/// The samplings of planes' images through the device's own conversion so far
/// (`WorkCounts::ycbcr_device_samples`): one grows while the device converts a texture, which
/// announces `ycbcr-filter-not-exact` at the texture's first conversion.
pub(super) fn device_samples(device: &Device) -> u64 {
    device.work_counts().ycbcr_device_samples
}

/// Vulkan's conversion of a texel of luma code `y` and chroma `cb` and `cr` (codes, reconstructed
/// unrounded) of `codes` under `conversion`: the range's expansion and the model's matrix,
/// unclamped, red, green and blue (before the sRGB curve, for an `_sRGB` twin).
pub(super) fn vulkan_read(
    codes: YCbCrCodes,
    (range, matrix): (YCbCrRange, YCbCrMatrix),
    [y, cb, cr]: [f64; 3],
) -> [f64; 3] {
    let n = codes.bits as i32;
    let max = f64::from((1u32 << n) - 1);
    let scale = f64::from(1u32 << (n - 8));
    let (luma, cb, cr) = match range {
        YCbCrRange::Video => (
            (y - 16.0 * scale) / (219.0 * scale),
            (cb - 128.0 * scale) / (224.0 * scale),
            (cr - 128.0 * scale) / (224.0 * scale),
        ),
        _ => {
            let mid = f64::from(1u32 << (n - 1));
            (y / max, (cb - mid) / max, (cr - mid) / max)
        }
    };
    let (kr, kb) = match matrix {
        YCbCrMatrix::Bt709 => (0.2126, 0.0722),
        YCbCrMatrix::Bt2020 => (0.2627, 0.0593),
        _ => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    [
        luma + 2.0 * (1.0 - kr) * cr,
        luma - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr,
        luma + 2.0 * (1.0 - kb) * cb,
    ]
}

/// `v` through the sRGB encoding: the inverse of the decoding curve the sample module applies to
/// an `_sRGB` twin's texels.
fn encoded(v: f64) -> f64 {
    match v <= 0.04045 / 12.92 {
        true => v * 12.92,
        false => 1.055 * v.powf(1.0 / 2.4) - 0.055,
    }
}

/// Asserts that the `float4`s `texels` of a texture of `output` the device converted hold
/// `want` ([`vulkan_read`] of each texel) within [`DEVICE_BOUND`], an `_sRGB` twin's through the
/// inverse of the curve, alpha 1. A converted format's texels, which its image stores as
/// `half`s, and `half` reads also take the `half`'s rounding of each value, within one unit in
/// the last place (`PrecisionNote::NativeHalfConversion`). The largest
/// distance, in units of `1/2048`, is printed.
pub(super) fn assert_device_reads(
    texels: &[u8],
    want: &[[f64; 3]],
    (output, half): (YCbCrOutput, bool),
    case: &str,
) {
    assert_ne!(
        output,
        YCbCrOutput::Codes,
        "{case}: codes are read through ours"
    );
    let mut most = 0.0f64;
    for (i, (t, w)) in texels.as_chunks::<16>().0.iter().zip(want).enumerate() {
        let c = |i: usize| f64::from(f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()));
        assert_eq!(c(3), 1.0, "{case}: texel {i}'s alpha");
        for (k, &want) in w.iter().enumerate() {
            let got = c(k);
            // A `half`'s unit in the last place at the value: the value it rounded lies within it.
            let ulp = match half || output == YCbCrOutput::Converted {
                true => 2f64.powi(got.abs().max(2f64.powi(-14)).log2().floor() as i32 - 10),
                false => 0.0,
            };
            let decode = |v: f64| match output {
                YCbCrOutput::Srgb => encoded(v),
                _ => v,
            };
            let (low, high) = (decode(got - ulp), decode(got + ulp));
            let far = (low - want).max(want - high).max(0.0);
            most = most.max(far * 2048.0);
            assert!(
                far <= DEVICE_BOUND / 2048.0,
                "{case}: texel {i} channel {k}: {got}, Vulkan's conversion {want} ({} / 2048 off)",
                far * 2048.0
            );
        }
    }
    println!("YCBCR-DEVICE {case}: within {most:.4} / 2048 of Vulkan's conversion");
}
