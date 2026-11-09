//! The texels Metal's color fills write (`fill_value.rs`): every texel of Metal's fills our probes
//! read back, and the rules' edges.

use crate::fill_value::{device_apart, device_converts, device_nan, fill_texel};
use metal2vulkan::metal::PixelFormat;
use PixelFormat as P;

/// The texel `fill_texel` packs, which every case here has.
fn bytes(pf: PixelFormat, c: [f64; 4]) -> Vec<u8> {
    fill_texel(pf, c).unwrap_or_else(|| panic!("no packing for {pf:?}"))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The code of a one-channel texel of `pf` for the red value `x`, as a little-endian integer.
fn code(pf: PixelFormat, x: f64) -> u64 {
    let b = bytes(pf, [x, 0.0, 0.0, 0.0]);
    b.iter()
        .rev()
        .fold(0, |v, &byte| (v << 8) | u64::from(byte))
}

/// The probe's (`fil.m`) special values, the first colors of its sweep, every channel alike.
const SPECIALS: [f64; 68] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    0.25,
    0.75,
    1.0 / 3.0,
    1e-3,
    -1e-3,
    f64::INFINITY,
    f64::NEG_INFINITY,
    5e-324,
    -5e-324,
    1e-310,
    1.401298464324817e-45,
    1.1754943508222875e-38,
    5.960464477539063e-08,
    2.9802322387695312e-08,
    4.470348358154297e-08,
    6.103515625e-05,
    65504.0,
    65519.99999999999,
    65520.0,
    65536.0,
    1e10,
    -1e10,
    3.4028234663852886e38,
    3.4028235677973366e38,
    1e39,
    -1e39,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    4294967295.0,
    4294967295.5,
    4294967296.0,
    255.5,
    127.5,
    -128.5,
    32767.5,
    65535.5,
    0.5 / 255.0,
    1.5 / 255.0,
    1.0 / 510.0,
    384.0 / 510.0,
    894.0 / 510.0,
    -384.0 / 510.0,
    1.2529411764705882,
    -0.7529411764705882,
    0.5 / 1023.0,
    1.5 / 1023.0,
    0.5 / 65535.0,
    64512.0,
    0.04045,
    0.0031308,
    1e-300,
    1e300,
    65408.0,
    65409.0,
    65472.0,
    4.0,
    15.5,
    31.5,
];

/// The probe's sweep: 4,096 colors, the first 68 [`SPECIALS`] in every channel, the rest each
/// channel a NaN of five patterns (1 in 7), a special (3 in 7) or a random value, from its
/// xorshift generator (its build fused the random values' multiply-adds).
fn sweep_colors() -> Vec<[f64; 4]> {
    let mut colors = generated(4096);
    for (color, &special) in colors.iter_mut().zip(&SPECIALS) {
        *color = [special; 4];
    }
    colors
}

/// The probe's generator's first `n` colors, before its specials overwrite any.
fn generated(n: usize) -> Vec<[f64; 4]> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const NANS: [u64; 5] = [
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0123,
        0xfff0_0000_0000_0001,
    ];
    (0..n)
        .map(|_| {
            [0; 4].map(|_| {
                let r = next();
                if r % 7 == 0 {
                    return f64::from_bits(NANS[((r >> 8) % 5) as usize]);
                }
                if r % 7 < 4 {
                    return SPECIALS[((r >> 8) % SPECIALS.len() as u64) as usize];
                }
                let u = (next() >> 11) as f64 / 9007199254740992.0;
                match (r >> 20) % 4 {
                    0 => 5.0f64.mul_add(u, -2.0),
                    1 => u,
                    2 => 140000.0f64.mul_add(u, -70000.0),
                    _ => u * 2f64.powi(((r >> 32) % 160) as i32 - 80),
                }
            })
        })
        .collect()
}

/// The probe's NaN and edge colors: each of 24 NaNs and 26 doubles at rounding
/// edges in every channel, then in each channel apart over (0.25, 0.5, 0.75, 1).
fn edge_colors() -> Vec<[f64; 4]> {
    const NAN_BITS: [u64; 24] = [
        0x7ff0_0000_0000_0001,
        0x7ff0_0000_2000_0000,
        0x7ff0_0000_4000_0000,
        0x7ff0_0400_0000_0000,
        0x7ff2_0000_0000_0000,
        0x7ff4_0000_0000_0000,
        0x7ff7_ffff_ffff_ffff,
        0x7ff0_2000_0000_0000,
        0x7ff8_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_2000_0000,
        0x7ff8_0400_0000_0000,
        0x7ff8_2000_0000_0000,
        0x7ffa_0000_0000_0000,
        0x7ffc_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0xfff0_0000_0000_0001,
        0xfff4_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0xfffc_0000_0000_0000,
        0xfff8_0400_0000_0000,
        0xffff_ffff_ffff_ffff,
        0x7ff0_0000_0010_0000,
        0x7ff8_0000_0010_0000,
    ];
    let p = |e: i32| 2f64.powi(e);
    let extra = [
        1.0 + p(-9) - p(-40),
        1.0 + p(-9) + p(-40),
        2.0 - p(-40),
        2.0 - p(-30),
        1.0 + p(-9),
        65408.0 + p(-30),
        65407.99999,
        65471.99999,
        65472.0,
        65472.0 + p(-30),
        p(-16) - p(-60),
        p(-17) + p(-26) - p(-60),
        p(-24) * 1.5 - p(-70),
        2147483647.5,
        2147483647.25,
        2147483648.5,
        4294967294.5,
        4294967295.25,
        4294967295.75,
        2147483647.0,
        -2147483648.5,
        -2147483648.25,
        -2147483647.5,
        0.5 - p(-40),
        0.5 + p(-40),
        1.0 / 3.0 + p(-35),
    ];
    let base = [0.25, 0.5, 0.75, 1.0];
    let values = NAN_BITS.iter().map(|&b| f64::from_bits(b)).chain(extra);
    let mut out = Vec::new();
    for v in values {
        out.push([v; 4]);
        for channel in 0..4 {
            let mut color = base;
            color[channel] = v;
            out.push(color);
        }
    }
    out
}

/// FNV-1a (64-bit) of `bytes`.
fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Metal's texels, FNV-1a of each format's texels in color order: its raw value, the sweep's
/// and the NaN and edge colors', on an Apple M4 under macOS 27.0.1.
const METAL: [(u32, u64, u64); 70] = [
    (1, 0xac652e68d8828b8b, 0x4ddf512a557ec47b),
    (10, 0x4bfbe07fba17a0e7, 0x35a785406c3be905),
    (11, 0x8de5966167926603, 0x70d2e5a864890355),
    (12, 0x359fa4f9bc18abef, 0xcd58634f4cb98c59),
    (13, 0xf15ff8284be2f1f7, 0x45af9776a4ee8641),
    (14, 0x19a60266e1111d31, 0xd32dcb82dde0a241),
    (20, 0x40ebfaa934e9f7ee, 0x21c38ea5c5aefd8d),
    (22, 0x0de023b1e9b3eea6, 0x939028f6625680e9),
    (23, 0x859b676900e62890, 0xda4ee275f2659bc5),
    (24, 0x2ab7a71fdb174a42, 0xfd4f0ab5d5cef731),
    (25, 0xb5c36613a52bedf6, 0x1cd2cf9cceaaf1bd),
    (26, 0xd75fcc6f3b02e872, 0x3d85df4cbe15e74d),
    (30, 0xec8be1a8622ead6d, 0xeac9a232cf46b9fd),
    (31, 0xec1c0d0afe4ece01, 0xe00f64472d61c9f9),
    (32, 0xef57828311ad7b8a, 0xa59a2b70067f495d),
    (33, 0x210f6df6772680fa, 0x5df7f2d7719f1b9d),
    (34, 0xd5e76ae4ef586479, 0x6a407230d86bca9d),
    (40, 0x8feb6de472b995b6, 0x4bfc32556ccb7295),
    (41, 0xa6eb89db22a67f60, 0x060ef4d3654288ed),
    (42, 0x230bcc5fc2f6f094, 0x7e2c4aad77c4334d),
    (43, 0x486f4512bd2bad6a, 0x883f9e1d00751d6d),
    (53, 0xb05fc5a97cc32960, 0xfa7537c714de0415),
    (54, 0x87b6c4bfadb9054c, 0x9342ae4c6475907d),
    (55, 0xe70de2ca5c7ee499, 0xe98a468c36e76fa5),
    (60, 0x43b8055b344419b9, 0x0293d80143afb905),
    (62, 0x4be33cca143be412, 0x4749d441f3b6f99d),
    (63, 0x552eded5a8d23872, 0x3e385ea248968325),
    (64, 0xb7ba58f0cdd2ce58, 0xc5b22f00d0ab7a8d),
    (65, 0x8a0e0e51230d9db2, 0x0706281acaca526d),
    (66, 0x9eb8f2927a9ee2ee, 0xf242e0c7ce7e93f5),
    (70, 0xc747538909dca69d, 0xc7b16b8fae3ed2c1),
    (71, 0xb9f829753d2b9f16, 0xfcd316d7380bfe75),
    (72, 0x3f75150ee26790f0, 0xda8df0edbe8ad1f9),
    (73, 0xcfc2e265793cb922, 0x2342da747115d435),
    (74, 0x68f611146f92ff83, 0x81cf9d0e7b361135),
    (80, 0x73d80356b425db69, 0x2ef6e17cb0c9a075),
    (81, 0xad98321985ae6576, 0xfee41c37c4d929dd),
    (84, 0xfd46401f46717fed, 0xf18455f9fb6ac575),
    (90, 0xff8f3b6ddc39bb8f, 0x873cd9fd7688870d),
    (91, 0x07cc6e213fa384a6, 0xab0f9376948ce8ed),
    (92, 0x7590b2e71f3e364b, 0xbb0e96bbde45bb51),
    (93, 0xd2965dbfad784c9e, 0xbea8b5dff91f39aa),
    (94, 0x6d0733399c0eea0f, 0xa6a3991745e3a99d),
    (103, 0x560186d6aee1e975, 0x5b133a9ecb31a9e5),
    (104, 0x9138ffd86d2e9a72, 0xb3b2ed3e0907d335),
    (105, 0x9545e1e1bb71fcb6, 0x17c92e6e79b9badd),
    (110, 0xc1bb25941bea7354, 0x2babf1fa9b2d888d),
    (112, 0x8a338facd6d2f2d3, 0x1054b4be9c72d1dd),
    (113, 0x73b8c80503613bbe, 0xc2af3fe24a976865),
    (114, 0x400aa861f2535b72, 0x5d3ac3ba9ecada55),
    (115, 0xfc517ad33dbfab57, 0xd74cbcbad1b16bdd),
    (116, 0x1872f93ffe8c9596, 0x3090174849baa165),
    (123, 0xe7735556ac97438e, 0x6f1b190dbc2d11a5),
    (124, 0x53a7ba7164cb4305, 0xb2bc064158c09e45),
    (125, 0xa12937bf6478549b, 0x22f2f04fd290d0b5),
    (300, 0x9b6e74cd90343c61, 0x7335203d08237579),
    (301, 0x60c8237c4cb6326a, 0x96d15b39ff2001d5),
    (302, 0x4cb0a9bc0ea58cff, 0x5e684e3b6f57626d),
    (303, 0x0b705dfc20945ea7, 0x4de9d52cfd5cba05),
    (304, 0xb6a522c54fdebd1e, 0xb92cb053bda75805),
    (313, 0x99c6a25d798bb007, 0x45af17661c96c7b5),
    (314, 0xcdd7b58554b4f4f1, 0xc10455cbf77eab81),
    (315, 0xfc517ad33dbfab57, 0xd74cbcbad1b16bdd),
    (323, 0x3f74d41bad2f3dc2, 0x5bbfda880fac6645),
    (324, 0x9a2a7a03ca02eb5b, 0x4f0ee3ad91a1c0bd),
    (325, 0xa12937bf6478549b, 0x22f2f04fd290d0b5),
    (552, 0x20052ba4f8ee93dd, 0x173f0630d9d0e345),
    (553, 0xed4fff60bff8d530, 0x1a3a41ef34857ca5),
    (554, 0x2514a852a9b65387, 0x7c1cd6d137e68489),
    (555, 0x56b898d4036711f1, 0xe5cfd31ec7cf1d25),
];

/// Every texel of Metal's fills of the probes' colors, in each of 70 formats.
#[test]
fn every_measured_fill_texel_is_metal_s() {
    let sweep = sweep_colors();
    let edges = edge_colors();
    assert_eq!(edges.len(), 250);
    let mut wrong = Vec::new();
    for &(raw, swept, edged) in &METAL {
        let pf = PixelFormat::try_from(raw).unwrap();
        let digest = |colors: &[[f64; 4]]| fnv(colors.iter().flat_map(|&c| bytes(pf, c)));
        if digest(&sweep) != swept || digest(&edges) != edged {
            wrong.push(pf);
        }
    }
    assert!(wrong.is_empty(), "fills unlike Metal's: {wrong:?}");
}

/// On every device a plain fill is announced wherever a normalized channel's scaled color is off a
/// whole code, where Vulkan lets the device's clear store either of the two codes closest to it,
/// as the RTX 5080 Laptop GPU's clears stored. A whole code, and a float channel, is not.
#[test]
fn every_fill_off_a_whole_code_is_announced() {
    for (pf, color) in [
        (P::R16Unorm, [0.001, 0.0, 0.0, 0.0]),
        (P::RG8Unorm, [0.0, 0.904, 0.0, 0.0]),
        (P::R8Snorm, [0.649_799_282_935_052, 0.0, 0.0, 0.0]),
        (P::B5G6R5Unorm, [0.0, 0.0, 0.752_941_176_470_588_2, 0.0]),
    ] {
        assert!(device_apart(pf, color), "{pf:?} {color:?}");
    }
    for (pf, color) in [
        (P::RGBA8Unorm, [0.0, 1.0, 0.0, 1.0]),
        (P::R8Snorm, [-1.0, 0.0, 0.0, 0.0]),
        (P::R32Float, [0.1, 0.0, 0.0, 0.0]),
    ] {
        assert!(!device_apart(pf, color), "{pf:?} {color:?}");
    }
}

/// Unsigned normalized: a 4- or 8-bit channel rounds the exact product half up, any other width
/// the product's `float` (the probe's thresholds).
#[test]
fn unorm_rounds_its_product_by_width() {
    // 0.5/65535's `float` times 65535 rounds to 0.5 in `float`, which rounds up; a clear stores 0.
    assert_eq!(code(P::R16Unorm, 0.5 / 65535.0), 1);
    assert_eq!(code(P::RGB10A2Unorm, 0.5 / 1023.0), 1);
    // 2-bit alpha: 0.8333333134651184 · 3 is 2.49999994, whose `float` is 2.5.
    let alpha = |a: f64| bytes(P::RGB10A2Unorm, [0.0, 0.0, 0.0, a])[3] >> 6;
    assert_eq!(alpha(0.8333333134651184), 3);
    assert_eq!(alpha(0.1666666567325592), 0);
    // A1BGR5's alpha of 0.5 stores 1 (a clear's `half` rounding stores 0).
    assert_eq!(bytes(P::A1BGR5Unorm, [0.0, 0.0, 0.0, 0.5])[0] & 1, 1);
    // 8-bit: the exact product, 0.25 · 255 = 63.75.
    assert_eq!(code(P::R8Unorm, 0.25), 64);
    assert_eq!(code(P::R8Unorm, f64::NAN), 0);
}

/// sRGB: the code whose decoded value is nearest; a clear's encoding curve rounds 0.9429933939
/// to 0xf9, the fill to 0xf8.
#[test]
fn srgb_takes_the_nearest_decoded_code() {
    assert_eq!(code(P::R8UnormSRGB, 0.9429933939111084), 0xf8);
    assert_eq!(code(P::R8UnormSRGB, 0.9004610341771649), 0xf3);
    assert_eq!(code(P::R8UnormSRGB, 0.5), 0xbc);
    assert_eq!(code(P::R8UnormSRGB, -1.0), 0);
    assert_eq!(code(P::R8UnormSRGB, 2.0), 0xff);
    assert_eq!(code(P::R8UnormSRGB, f64::NAN), 0);
}

/// Integers: truncated and saturated from the double; a quiet NaN 0 (the smallest signed), a
/// signalling one the largest; 32-bit doubles between 2^31 - 1 and 2^31 store 2^31.
#[test]
fn integers_truncate_and_sort_nans_by_kind() {
    let quiet = f64::from_bits(0x7ff8_0000_0000_0000);
    let signalling = f64::from_bits(0xfff0_0000_0000_0001);
    assert_eq!(code(P::R8Uint, quiet), 0);
    assert_eq!(code(P::R8Uint, signalling), 0xff);
    assert_eq!(code(P::R8Sint, quiet), 0x80);
    assert_eq!(code(P::R8Sint, signalling), 0x7f);
    assert_eq!(code(P::R32Uint, 2147483647.25), 0x8000_0000);
    assert_eq!(code(P::R32Uint, 2147483647.0), 0x7fff_ffff);
    assert_eq!(code(P::R32Uint, 4294967295.75), 0xffff_ffff);
    assert_eq!(code(P::R32Sint, -2147483648.5), 0x8000_0000);
    assert_eq!(code(P::R16Sint, -2.5), 0xfffe);
}

/// Floats: a NaN keeps its sign and the top of its payload, quieted.
#[test]
fn float_nans_keep_their_payload() {
    let nan = |bits: u64| f64::from_bits(bits);
    assert_eq!(code(P::R16Float, nan(0x7ff0_0400_0000_0000)), 0x7e01);
    assert_eq!(code(P::R16Float, nan(0xfff4_0000_0000_0000)), 0xff00);
    assert_eq!(code(P::R16Float, 65519.99999999999), 0x7c00);
    assert_eq!(code(P::R32Float, nan(0x7ff0_0000_2000_0000)), 0x7fc0_0001);
    assert_eq!(code(P::R32Float, nan(0xfff8_0000_0000_0000)), 0xffc0_0000);
}

/// RG11B10Float: nearest even, saturating; below the smallest normal the `float` is first cut
/// to a multiple of 2^-37, so 2^-21 (1 + 2^-17), past the half step, still stores 0, and the
/// least `float` storing 1 is 2^-21 (1 + 2^-16); NaN all ones.
#[test]
fn small_floats_round_to_even_after_a_cut() {
    let red = |x: f64| code(P::RG11B10Float, x) & 0x7ff;
    let p = |e: i32| 2f64.powi(e);
    assert_eq!(red(p(-21) * (1.0 + p(-17))), 0);
    assert_eq!(red(p(-21) * (1.0 + p(-16))), 1);
    assert_eq!(red(1.5 * p(-20)), 2);
    assert_eq!(red(0.001), 0x142);
    assert_eq!(red(65504.0), 0x7bf);
    assert_eq!(red(f64::INFINITY), 0x7c0);
    assert_eq!(red(1e39), 0x7c0);
    assert_eq!(red(-1.0), 0);
    assert_eq!(red(f64::NAN), 0x7ff);
}

/// Extended range: `⌊(v + 384/510) · 510 + ½⌋` in `float` steps; 0.25 stores 511, not 512.
#[test]
fn extended_range_fills_are_packed() {
    assert_eq!(hex(&bytes(P::BGR10XR, [0.25; 4])), "fffdf71f");
    assert_eq!(hex(&bytes(P::BGR10XR, [0.0; 4])), "80010618");
    assert_eq!(hex(&bytes(P::BGRA10XR, [1.0; 4])), "80df80df80df80df");
    assert_eq!(hex(&bytes(P::BGRA10XR, [-1.0; 4])), "0000000000000060");
    assert_eq!(hex(&bytes(P::BGRA10XRSRGB, [0.5; 4])), "c0bdc0bdc0bdc09f");
    assert_eq!(hex(&bytes(P::BGR10XRSRGB, [-0.5; 4])), "09249000");
    assert_eq!(hex(&bytes(P::BGR10XRSRGB, [f64::NAN; 4])), "80010618");
}

/// RGB9E5Float: the `float`s, halves up, the exponent raised where the largest rounds to 512.
#[test]
fn shared_exponents_round_half_up() {
    let texel =
        |v: f64| u32::from_le_bytes(bytes(P::RGB9E5Float, [v, v, v, 1.0]).try_into().unwrap());
    // 1 + 2^-9 - 2^-40 is the `float` 1 + 2^-9, a half step above 256 · 2^-8.
    let edge = 1.0 + 2f64.powi(-9) - 2f64.powi(-40);
    assert_eq!(texel(edge) & 0x1ff, 257);
    assert_eq!(texel(32767.5), 0xfc02_0100);
    assert_eq!(texel(-1.0), 0);
}

/// The texel a color fill without a pixel format wrote into a texture of each format on an M4
/// under macOS 27.0.1 (our probe `tfl3`): the colors (0.25, 0.5, 0.75, 1),
/// (2, -1, 0.5, 0.3) and (0, 0, 0, 0). The fill with the texture's own format wrote the same.
#[test]
fn metal_fill_texels_are_reproduced() {
    let colors = [
        [0.25, 0.5, 0.75, 1.0],
        [2.0, -1.0, 0.5, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let cases: &[(PixelFormat, &str, &str, &str)] = &[
        (P::R8Unorm, "40", "ff", "00"),
        (P::R8Snorm, "20", "7f", "00"),
        (P::R8Uint, "00", "02", "00"),
        (P::R16Unorm, "0040", "ffff", "0000"),
        (P::R16Uint, "0000", "0200", "0000"),
        (P::R16Float, "0034", "0040", "0000"),
        (P::RG8Unorm, "4080", "ff00", "0000"),
        (P::B5G6R5Unorm, "1744", "10f8", "0000"),
        (P::A1BGR5Unorm, "2f44", "20f8", "0000"),
        (P::ABGR4Unorm, "bf48", "85f0", "0000"),
        (P::BGR5A1Unorm, "17a2", "107c", "0000"),
        (P::R32Uint, "00000000", "02000000", "00000000"),
        (P::R32Float, "0000803e", "00000040", "00000000"),
        (P::RG16Unorm, "00400080", "ffff0000", "00000000"),
        (P::RG16Float, "00340038", "004000bc", "00000000"),
        (P::RGBA8Unorm, "4080bfff", "ff00804d", "00000000"),
        (P::RGBA8UnormSRGB, "89bce1ff", "ff00bc4d", "00000000"),
        (P::BGRA8Unorm, "bf8040ff", "8000ff4d", "00000000"),
        (P::BGRA8UnormSRGB, "e1bc89ff", "bc00ff4d", "00000000"),
        (P::RGB10A2Unorm, "0001f8ef", "ff030060", "00000000"),
        (P::RG11B10Float, "40031c74", "00040070", "00000000"),
        (P::RGB9E5Float, "8000027e", "00010089", "00000000"),
        (P::BGR10A2Unorm, "ff0208d0", "0002f07f", "00000000"),
        (
            P::RG32Uint,
            "0000000000000000",
            "0200000000000000",
            "0000000000000000",
        ),
        (
            P::RG32Float,
            "0000803e0000003f",
            "00000040000080bf",
            "0000000000000000",
        ),
        (
            P::RGBA16Unorm,
            "00400080ffbfffff",
            "ffff00000080cd4c",
            "0000000000000000",
        ),
        (
            P::RGBA16Float,
            "00340038003a003c",
            "004000bc0038cd34",
            "0000000000000000",
        ),
        (
            P::RGBA32Uint,
            "00000000000000000000000001000000",
            "02000000000000000000000000000000",
            "00000000000000000000000000000000",
        ),
        (
            P::RGBA32Float,
            "0000803e0000003f0000403f0000803f",
            "00000040000080bf0000003f9a99993e",
            "00000000000000000000000000000000",
        ),
    ];
    for &(pf, a, b, c) in cases {
        for (color, want) in colors.iter().zip([a, b, c]) {
            assert_eq!(hex(&bytes(pf, *color)), want, "{pf:?} {color:?}");
        }
    }
    for pf in [
        P::Depth16Unorm,
        P::Depth32Float,
        P::BC1RGBA,
        P::RGB10A82PXR10,
    ] {
        assert_eq!(fill_texel(pf, colors[0]), None, "{pf:?}");
    }
}

/// `ARGB8Unorm`'s color fill: Metal's bytes alpha, red, green and blue (the format scout's
/// `fill color` cell on an M4 under macOS 27.0.1, a 16 x 16 private texture filled with
/// (0.25, 0.5, 0.75, 1): every texel `ff 40 80 bf`), each channel converted as
/// `RGBA8Unorm`'s is.
#[test]
fn argb8_fill_texels_put_alpha_first() {
    assert_eq!(
        hex(&bytes(P::ARGB8Unorm, [0.25, 0.5, 0.75, 1.0])),
        "ff4080bf"
    );
    let rgba = bytes(P::RGBA8Unorm, [2.0, -1.0, 0.5, 0.3]);
    let argb = bytes(P::ARGB8Unorm, [2.0, -1.0, 0.5, 0.3]);
    assert_eq!(argb, [rgba[3], rgba[0], rgba[1], rgba[2]]);
}

/// The RGBX formats' color fills on an M4 under macOS 27.0.1 (our probe
/// `fpv1`: every texel of a 64 x 16 shared texture filled with each color): the twin's red,
/// green and blue, and an X field of zeros, or of the alpha for the float formats.
#[test]
fn rgbx_fill_texels_are_reproduced() {
    let colors = [
        [0.25, 0.5, 0.75, 1.0],
        [0.1, 0.9, 0.3, 0.5],
        [1.5, -0.5, 7.0, 0.0],
    ];
    let zero8 = "0000000000000000";
    let zero16 = "00000000000000000000000000000000";
    let cases: &[(PixelFormat, &str, &str, &str)] = &[
        (P::RGBX8Unorm, "4080bf00", "1ae54d00", "ff00ff00"),
        (P::RGBX8UnormSRGB, "89bce100", "59f39500", "ff00ff00"),
        (P::RGBX8Snorm, "20405f00", "0d722600", "7fc07f00"),
        (P::RGBX8Uint, "00000000", "00000000", "01000700"),
        (P::RGBX8Sint, "00000000", "00000000", "01000700"),
        (P::RGBX16Uint, zero8, zero8, "0100000007000000"),
        (P::RGBX16Sint, zero8, zero8, "0100000007000000"),
        (
            P::RGBX16Float,
            "00340038003a003c",
            "662e333bcd340038",
            "003e00b800470000",
        ),
        (
            P::RGBX32Uint,
            zero16,
            zero16,
            "01000000000000000700000000000000",
        ),
        (
            P::RGBX32Sint,
            zero16,
            zero16,
            "01000000000000000700000000000000",
        ),
        (
            P::RGBX32Float,
            "0000803e0000003f0000403f0000803f",
            "cdcccc3d6666663f9a99993e0000003f",
            "0000c03f000000bf0000e04000000000",
        ),
    ];
    for &(pf, a, b, c) in cases {
        for (color, want) in colors.iter().zip([a, b, c]) {
            assert_eq!(hex(&bytes(pf, *color)), want, "{pf:?} {color:?}");
        }
    }
}

/// The words of `bytes`, little-endian.
fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks(2)
        .map(|w| u16::from_le_bytes([w[0], w[1]]))
        .collect()
}

/// Each channel's `float`, then the bfloat code nearest it, ties to even, an overflow the
/// infinity; a NaN keeps its sign and the top of its payload, quieted (Metal's
/// fills of 22 colors and of every `half` value, an Apple M4 under macOS 27.0.1). One word for
/// each channel the format has.
#[test]
fn a_bfloat_color_fill_packs_each_channel_s_float() {
    let fill = |v: f64| words(&bytes(P::R16Bfloat, [v, 0.5, 0.25, 1.0]))[0];
    // Ties to even: `1 + 2^-8` lies between codes 0x3f80 and 0x3f81, `1 + 3 · 2^-8` between
    // 0x3f81 and 0x3f82.
    assert_eq!(fill(1.0 + 1.0 / 256.0), 0x3f80);
    assert_eq!(fill(1.0 + 3.0 / 256.0), 0x3f82);
    // Not through `half`: 65520 is the bfloat 65536, not the `half` infinity.
    assert_eq!(fill(65520.0), 0x4780);
    assert_eq!(fill(f64::INFINITY), 0x7f80);
    assert_eq!(fill(f64::NEG_INFINITY), 0xff80);
    // The `float` overflow, and a value whose code rounds past the largest finite one.
    assert_eq!(fill(1e300), 0x7f80);
    assert_eq!(fill(3.4e38), 0x7f80);
    assert_eq!(fill(-1e-300), 0x8000);
    assert_eq!(fill(-0.0), 0x8000);
    // A NaN's sign and payload: Swift's `-Double.nan`, and the `half` 0x7c08 widened (its quiet
    // bit set, its payload 0x208 in the `double`'s top bits).
    assert_eq!(fill(f64::from_bits(0x7ff8_0000_0000_0000)), 0x7fc0);
    assert_eq!(fill(f64::from_bits(0xfff8_0000_0000_0000)), 0xffc0);
    assert_eq!(
        fill(f64::from_bits(0x7ff0_0000_0000_0000 | (0x208 << 42))),
        0x7fc1
    );
    assert_eq!(
        fill(f64::from_bits(0xfff0_0000_0000_0000 | (0x3ff << 42))),
        0xffff
    );
    let rgba = [0.5, -2.0, 0.1, 1.0];
    assert_eq!(words(&bytes(P::R16Bfloat, rgba)), [0x3f00]);
    assert_eq!(words(&bytes(P::RG16Bfloat, rgba)), [0x3f00, 0xc000]);
    assert_eq!(
        words(&bytes(P::RGBA16Bfloat, rgba)),
        [0x3f00, 0xc000, 0x3dcd, 0x3f80]
    );
}

/// The target formats of the probe's fills of a color packed in another format (`fil.m`'s
/// `crossall`), in its order, with their texel bytes.
const CROSS_TARGETS: [(PixelFormat, usize); 18] = [
    (P::RGBA32Uint, 16),
    (P::RGBA32Sint, 16),
    (P::RGBA32Float, 16),
    (P::RG32Uint, 8),
    (P::RG32Float, 8),
    (P::RGBA16Uint, 8),
    (P::RGBA16Float, 8),
    (P::RGBA16Bfloat, 8),
    (P::R32Uint, 4),
    (P::R32Float, 4),
    (P::RG16Uint, 4),
    (P::RGBA8Unorm, 4),
    (P::R16Uint, 2),
    (P::R8Uint, 1),
    (P::R16Bfloat, 2),
    (P::BGRA10XR, 8),
    (P::BGR10XR, 4),
    (P::RGB9E5Float, 4),
];

/// Metal's fills of a color packed in another format (an Apple M4 under macOS
/// 27.0.1): for each packing format's raw value, FNV-1a of the texels its 128 colors left in
/// textures of each of [`CROSS_TARGETS`], color after color, target after target.
const METAL_CROSS: [(u32, u64); 70] = [
    (1, 0xc8b15a6662e56be8),
    (10, 0x2d64bb7974f1d5d5),
    (11, 0x428f5a01caf8330a),
    (12, 0x6a298f81a7f82d5b),
    (13, 0x27d7a0053264a327),
    (14, 0xfb3cd9d182553d44),
    (20, 0x87eb404be36f0b51),
    (22, 0xdc04f37ee28a0aa2),
    (23, 0x62b7110caed090dc),
    (24, 0x5a20006e74c71c39),
    (25, 0x55ab8b2e09e9d37f),
    (26, 0x4dfcf900e7e34d98),
    (30, 0x573ee7529263bc99),
    (31, 0x87991a129e1ccb64),
    (32, 0x350a7f08713d2071),
    (33, 0x9f4eaa11ae5a091d),
    (34, 0x6b06820e15d4d728),
    (40, 0x81739838ac71e542),
    (41, 0x686226718f662232),
    (42, 0xf01a37512bdcde90),
    (43, 0x0d5290251e79424a),
    (53, 0x13b0e153009ebc33),
    (54, 0xa7841464616ac344),
    (55, 0x2a2091278519eb8c),
    (60, 0x6287910f9b4fbda5),
    (62, 0x4cfa419b9781193e),
    (63, 0x0fa05cef5801d37c),
    (64, 0xac4b1802aa292b35),
    (65, 0xb3b3f1a0d7d1f8cf),
    (66, 0x97a7e245b472936c),
    (70, 0xf75bf15bb33d8859),
    (71, 0x52c54d6a6efff4d8),
    (72, 0x69cb1e40af7269b5),
    (73, 0x3c8088f205aaa635),
    (74, 0x2dd86d9bd12d22b8),
    (80, 0x904ef55132fca062),
    (81, 0x68a5a8b924cd7d41),
    (84, 0xdf552981845a1802),
    (90, 0x39165f6ac4f5f3cd),
    (91, 0x71abbb12cd0c44f1),
    (92, 0xdc1c8fbe3ffcaa82),
    (93, 0x6ec2f8c378157f92),
    (94, 0x6215f6133a4f6912),
    (103, 0xd49b8169015dfe07),
    (104, 0x0416899095daf0be),
    (105, 0x4c744cf0dc8976b4),
    (110, 0x3964227f144a6653),
    (112, 0x58eac6b773f5136a),
    (113, 0x293337ea681d59f4),
    (114, 0x4e7f84adb4d02b07),
    (115, 0xed6f9595081a9ab3),
    (116, 0x6b439371b1812e60),
    (123, 0xa0ed16bd24c8e875),
    (124, 0x95805536266ab854),
    (125, 0x453c111d1476bd0c),
    (300, 0x49cff44f4f4a3525),
    (301, 0x7b15471ca026dbfc),
    (302, 0x7e64a5372fefa5f9),
    (303, 0x5b21cb122ef843c5),
    (304, 0x592d9675cc248cdc),
    (313, 0x920b199a234d20a6),
    (314, 0xc74c6db3e73fcfd5),
    (315, 0xed6f9595081a9ab3),
    (323, 0xd725d6e6ee0b0166),
    (324, 0x43c7fe8d3c96ee71),
    (325, 0x453c111d1476bd0c),
    (552, 0xf8dc4dc82a9c033f),
    (553, 0x7e65568e91576787),
    (554, 0x8e2ab99739c5cef4),
    (555, 0x1c801cdd3c1955bf),
];

/// A fill of a color packed in another format (`fillTexture:...:color:pixelFormat:`) writes that
/// format's texel repeated from the texel's first byte and cut at the texture's texel, whatever
/// the two formats' sizes (128 colors packed in each of 70 formats, the
/// extended-range, `_sRGB` and bfloat ones among them, into textures of 18 formats of 1 to 16
/// bytes, every texel Metal's).
#[test]
fn a_color_packed_in_another_format_repeats_its_texel() {
    let mut colors = generated(128);
    for (color, &special) in colors.iter_mut().zip(&SPECIALS[..32]) {
        *color = [special; 4];
    }
    colors[32] = [0.25, 0.5, 0.75, 1.0];
    colors[33] = [1.0, 0.0, 0.0, 0.0];
    let bits = colors.iter().flatten().flat_map(|v| v.to_le_bytes());
    assert_eq!(fnv(bits), 0x6910_f481_0603_9dba, "the probe's colors");
    let mut wrong = Vec::new();
    for (raw, metal) in METAL_CROSS {
        let pf = PixelFormat::try_from(raw).unwrap();
        let texels = colors.iter().flat_map(|&c| {
            let texel = bytes(pf, c);
            CROSS_TARGETS.iter().flat_map(move |&(_, size)| {
                texel.iter().copied().cycle().take(size).collect::<Vec<_>>()
            })
        });
        if fnv(texels) != metal {
            wrong.push(pf);
        }
    }
    assert!(wrong.is_empty(), "packings unlike Metal's: {wrong:?}");
}

/// The special values of our probe `pack.m`: color `i` of its first colors is
/// `(s[i], s[i + 1], s[i + 2], s[i + 3])`, indices modulo the count.
const PACK_SPECIALS: [f64; 88] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    0.25,
    0.75,
    1.0 / 3.0,
    1e-3,
    -1e-3,
    f64::INFINITY,
    f64::NEG_INFINITY,
    5e-324,
    -5e-324,
    1e-310,
    1.401298464324817e-45,
    1.1754943508222875e-38,
    5.960464477539063e-08,
    2.9802322387695312e-08,
    4.470348358154297e-08,
    6.103515625e-05,
    65504.0,
    65519.99999999999,
    65520.0,
    65536.0,
    1e10,
    -1e10,
    3.4028234663852886e38,
    3.4028235677973366e38,
    1e39,
    -1e39,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    4294967295.0,
    4294967295.5,
    4294967296.0,
    255.5,
    127.5,
    -128.5,
    32767.5,
    65535.5,
    0.5 / 255.0,
    1.5 / 255.0,
    1.0 / 510.0,
    384.0 / 510.0,
    894.0 / 510.0,
    -384.0 / 510.0,
    1.2529411764705882,
    -0.7529411764705882,
    0.5 / 1023.0,
    1.5 / 1023.0,
    0.5 / 65535.0,
    64512.0,
    0.04045,
    0.0031308,
    1e-300,
    1e300,
    65408.0,
    65409.0,
    65472.0,
    4.0,
    15.5,
    31.5,
    0.5 / 15.0,
    1.5 / 15.0,
    0.5 / 31.0,
    0.5 / 63.0,
    7.5,
    127.0,
    128.0,
    255.0,
    256.0,
    -127.0,
    -129.0,
    32767.0,
    32768.0,
    -32768.0,
    -32769.0,
    65535.0,
    65536.5,
    1023.0,
    1024.0,
    3.0,
];

/// Metal's packed texels on the M3 Max: per raw format, FNV-1a 64 of each color's 16 bytes.
const M3_PACKED: [(u32, u64); 88] = [
    (1, 0x09edfec5ceb46c15),
    (10, 0xc9c13c3903ee05c5),
    (11, 0x91840698df2fb605),
    (12, 0x8c34d3a3b420f295),
    (13, 0xa30563d1b5860fc5),
    (14, 0x7a06935f97b26a95),
    (20, 0xc68135c31da61975),
    (22, 0xd52cb0e596e5e745),
    (23, 0x49edaec300395f25),
    (24, 0xf4f5acc15bc1c8a5),
    (25, 0x55724d4ab40b2ff5),
    (30, 0xa8c53744806ef6a5),
    (31, 0x11b930bc12d532f5),
    (32, 0x13106c5643ad0915),
    (33, 0xb7a7b47ede539845),
    (34, 0x7adcca3460058905),
    (40, 0x3fc0aabdf07ed805),
    (41, 0x7fac691931059b95),
    (42, 0x786ebc82b38e2f25),
    (43, 0x93dffcdb93bdb775),
    (45, 0xff23f3bfc14a8de9),
    (46, 0x819c9cc67ae4c0f1),
    (47, 0xdb9aae03dd1048af),
    (48, 0x0096ddf59054447f),
    (53, 0xd9b569bf0bbab065),
    (54, 0xa60a7baf26ce086d),
    (55, 0x94f4d16646f6d9a5),
    (60, 0xc77a5424bbb783e5),
    (62, 0xdde97108d590f225),
    (63, 0xbcf717c6a1a4ff6d),
    (64, 0x9f6f4a4e74ed7705),
    (65, 0x2199e75fae189785),
    (70, 0xe53edf01bb6f303d),
    (71, 0xbc2b181e0d9a0d75),
    (72, 0xf251a7f2f355f4cd),
    (73, 0xdade99c5318f42cd),
    (74, 0xebd7b4edfa7fc595),
    (80, 0xd251a65f568ed3dd),
    (81, 0x062e31c5a7c12b85),
    (90, 0xbcd8486fe7fb2fcd),
    (91, 0xe88d71af8a76776d),
    (92, 0x197e32a73afd2edd),
    (93, 0xbf905eba1ea91305),
    (94, 0xa8e269919fa4af4d),
    (554, 0x1a355cbcdf254f85),
    (555, 0xa3e7e7627a037ea5),
    (95, 0x6e5a585f02d9d6f6),
    (96, 0x162c1d4e842c288d),
    (97, 0x0053089b91be3522),
    (98, 0xc919f9f873a5096f),
    (99, 0xd93a5277330f8870),
    (103, 0xd1eb13b2e48809dd),
    (104, 0xe383cd9bd4cd3715),
    (105, 0x66524edfb6e70455),
    (110, 0xa7bb1705af705785),
    (112, 0x5cbe5bff4a89d42d),
    (113, 0xbcacf412ae240b25),
    (114, 0x331d1a1c86612dc1),
    (115, 0xafcf19bb02807b41),
    (552, 0x2c686bb72df7ed55),
    (553, 0x00aa8ffa43139175),
    (120, 0xab081fb39e2d9619),
    (121, 0x30e12dd8a6191db4),
    (122, 0x6f446a2ed5ecf115),
    (123, 0xd188239b19020738),
    (124, 0x82f23e151c22d281),
    (125, 0x9665315fcfe99c0d),
    (84, 0x153cbd6614195e0d),
    (556, 0xe0eae52c31abe409),
    (26, 0x67e399d5c9c84ea5),
    (44, 0x1a0f09262b647425),
    (66, 0x87e99425fb7ee38d),
    (85, 0xd39dcc42959a4285),
    (100, 0xe6d2826c04edd16d),
    (116, 0xb19bd6f18907e405),
    (300, 0x3227de0cfe6987d5),
    (301, 0x6fd02644396c579d),
    (302, 0x73da0649d9d0ec65),
    (303, 0xe9be188576515e65),
    (304, 0xde7a0a42b5222c25),
    (313, 0x5ce2bd9ecff9ac71),
    (314, 0xff1039e411496579),
    (315, 0xafcf19bb02807b41),
    (323, 0x034d7a4213b5384d),
    (324, 0x7475d06a8616a702),
    (325, 0x9665315fcfe99c0d),
    (637, 0xd973f20b29bc22f5),
    (638, 0x522fb7d7164e51e5),
];

/// The raw formats a color fill packs no color in on the M3 Max: its bytes are the calling
/// thread's stack.
const M3_UNPACKED: [u32; 213] = [
    0, 130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 150, 151, 152, 153, 160, 161, 162, 163,
    164, 165, 166, 167, 170, 172, 174, 176, 178, 179, 180, 181, 182, 183, 186, 187, 188, 189, 190,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214,
    215, 216, 217, 218, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 240,
    241, 250, 252, 253, 255, 260, 261, 262, 263, 550, 551, 500, 502, 503, 505, 506, 507, 508, 509,
    510, 520, 522, 523, 525, 526, 527, 528, 529, 530, 540, 541, 542, 543, 544, 545, 546, 547, 548,
    564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 580, 581, 582, 583, 584, 585, 591,
    592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 608, 609, 610, 611, 612, 613, 615, 616,
    617, 618, 619, 620, 651, 652, 185, 191, 203, 209, 221, 227, 501, 504, 521, 524, 560, 562, 563,
    576, 577, 578, 579, 586, 587, 588, 589, 590, 603, 604, 605, 606, 607, 621, 622, 623, 624, 625,
    626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 639, 640, 641, 642, 643, 644, 645, 646,
    647, 648, 649, 650,
];

/// The NaNs of our probe `pack.m`, by their bits: each in turn in every channel of a color whose
/// other channels are 0.5.
const PACK_NANS: [u64; 8] = [
    0x7ff8_0000_0000_0000,
    0xfff8_0000_0000_0000,
    0x7ff0_0000_0000_0001,
    0xfff0_0000_0000_0001,
    0x7ffc_0000_0000_0000,
    0x7ff4_0000_0000_0000,
    0x7fff_ffff_ffff_ffff,
    0x7ff8_0000_0012_3456,
];

/// The colors of our probe `pack.m`, in its order: the specials' colors, 1024 colors of four
/// doubles from its xorshift (the top 53 bits of a step as a fraction of `2^53`, times 6, less
/// 3), then the NaNs'.
fn pack_colors() -> Vec<[f64; 4]> {
    let n = PACK_SPECIALS.len();
    let mut colors: Vec<[f64; 4]> = (0..n)
        .map(|i| [0, 1, 2, 3].map(|c| PACK_SPECIALS[(i + c) % n]))
        .collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..1024 {
        colors.push([0; 4].map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / 9_007_199_254_740_992.0 * 6.0 - 3.0
        }));
    }
    for bits in PACK_NANS {
        for channel in 0..4 {
            let mut color = [0.5; 4];
            color[channel] = f64::from_bits(bits);
            colors.push(color);
        }
    }
    colors
}

/// Every pixel format on an Apple M3 Max under macOS 27.0.1 (our probes `classify.m` and
/// `pack.m`): a color fill packed in one of 88 formats writes its texel, which the runtime packs
/// byte for byte for each of the probe's 1,144 colors (each texel repeated or cut to the 16 bytes
/// of an `RGBA32Uint` texel); in each of the other 213 it writes bytes of the calling thread's
/// stack (a probe that set the stack below its call to one byte read that byte back), and the
/// runtime packs nothing.
#[test]
fn the_m3_packs_these_formats_and_no_other() {
    let colors = pack_colors();
    assert_eq!(colors.len(), 1144);
    let mut wrong = Vec::new();
    for (raw, metal) in M3_PACKED {
        let pf = PixelFormat::try_from(raw).unwrap();
        let ours = colors.iter().flat_map(|&c| {
            let texel = bytes(pf, c);
            texel.iter().copied().cycle().take(16).collect::<Vec<_>>()
        });
        if fnv(ours) != metal {
            wrong.push(pf);
        }
    }
    assert!(wrong.is_empty(), "packings unlike Metal's: {wrong:?}");
    for raw in M3_UNPACKED {
        let pf = PixelFormat::try_from(raw).unwrap();
        assert_eq!(fill_texel(pf, [0.5; 4]), None, "{pf:?}");
    }
}

/// A plain fill is the device's clear of the color only where the format's fill converts a
/// `float` (`device_converts`), and the clear is announced where the device's conversion may store
/// another texel than Metal's fill packs (`device_apart`: a normalized value off a whole code, a
/// tie among them, which the fill rounds half up, an sRGB value inside `(0, 1)`, a value a half,
/// small float or shared exponent does not hold), or a NaN it converts as it likes
/// (`device_nan`).
#[test]
fn the_device_s_clear_is_announced_where_its_conversion_is_in_question() {
    for pf in [
        P::RGBA8Unorm,
        P::RGBA8UnormSRGB,
        P::RG8Snorm,
        P::RGBA16Float,
        P::R32Float,
        P::RG11B10Float,
        P::RGB9E5Float,
        P::B5G6R5Unorm,
        P::RGBX8Unorm,
    ] {
        assert!(device_converts(pf), "{pf:?}");
    }
    for pf in [
        P::RGBA8Uint,
        P::RG16Sint,
        P::R16Bfloat,
        P::BGRA10XR,
        P::BGR10XRSRGB,
        P::Depth32Float,
        P::BC1RGBA,
    ] {
        assert!(!device_converts(pf), "{pf:?}");
    }
    let one = |x: f64| [x, 0.0, 0.0, 1.0];
    let apart = [
        (P::RGBA8Unorm, one(0.5), true),
        (P::RGBA8Unorm, one(0.25), true),
        (P::RGBA8Unorm, one(1.0), false),
        (P::RGBA8Unorm, one(f64::NAN), false),
        // Off a tie by a hundredth of a code, off a whole code all the same.
        (P::RGBA8Unorm, one(127.51 / 255.0), true),
        (P::B5G6R5Unorm, one(15.51 / 31.0), true),
        (P::R16Unorm, one(0.5 / 65535.0), true),
        (P::RG8Snorm, one(-0.5 / 127.0), true),
        (P::RG8Snorm, one(-1.0), false),
        (P::RGBA8UnormSRGB, one(0.5), true),
        (P::RGBA8UnormSRGB, [0.0, 1.0, 0.0, 0.5], true),
        (P::RGBA8UnormSRGB, [0.0, 1.0, 0.0, 0.25], true),
        (P::RGBA8UnormSRGB, [0.0, 1.0, 0.0, 1.0], false),
        (P::RGBA16Float, one(0.1), true),
        (P::RGBA16Float, one(0.5), false),
        (P::RGBA16Float, one(1e6), true),
        (P::R32Float, one(0.1), false),
        (P::RG11B10Float, one(0.1), true),
        (P::RG11B10Float, one(0.5), false),
        (P::RG11B10Float, one(-1.0), false),
        (P::RGB9E5Float, [1.0, 0.5, 0.25, 0.0], false),
        (P::RGB9E5Float, [0.1, 0.5, 0.25, 0.0], true),
        (P::RGB9E5Float, [1e6, 0.0, 0.0, 0.0], true),
    ];
    for (pf, color, want) in apart {
        assert_eq!(device_apart(pf, color), want, "{pf:?} {color:?}");
    }
    for (pf, want) in [
        (P::RGBA16Float, true),
        (P::RG11B10Float, true),
        (P::R32Float, false),
        (P::RGBA8Unorm, false),
        (P::RGB9E5Float, false),
    ] {
        assert_eq!(device_nan(pf, one(f64::NAN)), want, "{pf:?}");
    }
}
