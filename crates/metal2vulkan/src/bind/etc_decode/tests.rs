//! The ETC2 and EAC decode against the specification's rules and Metal's reads.

use super::{EtcDecode, EtcTexels};
use crate::metal::PixelFormat;
use crate::vk::Format;

/// The ETC2 and EAC formats alone decode, each sRGB form to its unorm form's codes.
#[test]
fn the_etc2_and_eac_formats_alone_decode() {
    use PixelFormat as P;
    let decoded: Vec<(PixelFormat, EtcDecode)> = PixelFormat::ALL
        .iter()
        .filter_map(|&pf| EtcDecode::of(pf).map(|d| (pf, d)))
        .collect();
    assert_eq!(
        decoded,
        [
            (P::EACR11Unorm, EtcDecode::R11Unorm),
            (P::EACR11Snorm, EtcDecode::R11Snorm),
            (P::EACRG11Unorm, EtcDecode::Rg11Unorm),
            (P::EACRG11Snorm, EtcDecode::Rg11Snorm),
            (P::EACRGBA8, EtcDecode::Rgba8),
            (P::EACRGBA8SRGB, EtcDecode::Rgba8),
            (P::ETC2RGB8, EtcDecode::Rgb8),
            (P::ETC2RGB8SRGB, EtcDecode::Rgb8),
            (P::ETC2RGB8A1, EtcDecode::Rgb8A1),
            (P::ETC2RGB8A1SRGB, EtcDecode::Rgb8A1),
        ]
    );
    for decode in EtcDecode::ALL {
        let (compact, float) = match decode {
            EtcDecode::R11Unorm => (Format::R16_UNORM, Format::R32_SFLOAT),
            EtcDecode::R11Snorm => (Format::R16_SNORM, Format::R32_SFLOAT),
            EtcDecode::Rg11Unorm => (Format::R16G16_UNORM, Format::R32G32_SFLOAT),
            EtcDecode::Rg11Snorm => (Format::R16G16_SNORM, Format::R32G32_SFLOAT),
            _ => (Format::R8G8B8A8_UNORM, Format::R8G8B8A8_UNORM),
        };
        assert_eq!(
            decode.texel_format(EtcTexels::Compact),
            compact,
            "{decode:?}"
        );
        assert_eq!(decode.texel_format(EtcTexels::Float), float, "{decode:?}");
        assert_eq!(decode.eleven_bit(), compact != float, "{decode:?}");
        for texels in [EtcTexels::Compact, EtcTexels::Float] {
            let format = decode.texel_format(texels);
            let bytes = format.block().map(|b| b.size).expect("a block");
            assert_eq!(decode.texel_bytes(texels), bytes, "{decode:?} {texels:?}");
        }
    }
}

/// A block's 64 bits, most significant first, as its bytes.
fn block(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

/// The `float` Metal reads of an 11-bit block's texel's red.
fn value(decode: EtcDecode, values: [i32; 2]) -> f32 {
    decode.metal_value(values[0])
}

/// The zero blocks (a new texture's) decode by the rules: an individual-mode block of black with
/// table 0's `+2` (alpha 0 in `EAC_RGBA8`, whose zero alpha block reads 0), RGB8A1's transparent
/// differential block whose index 0 is the modifier 0, and an 11-bit block whose multiplier of 0
/// adds the modifier `-3` itself.
#[test]
fn the_zero_blocks_decode_by_the_rules() {
    let zero = [0u8; 16];
    let codes = |v: [i32; 2]| v[0].to_le_bytes();
    assert_eq!(
        codes(EtcDecode::Rgb8.block_values(&zero)[5]),
        [2, 2, 2, 255]
    );
    assert_eq!(
        codes(EtcDecode::Rgb8A1.block_values(&zero)[5]),
        [0, 0, 0, 255]
    );
    assert_eq!(codes(EtcDecode::Rgba8.block_values(&zero)[5]), [2, 2, 2, 0]);
    let r11 = EtcDecode::R11Unorm;
    assert_eq!(value(r11, r11.block_values(&zero)[3]), 1.0 / 2047.0);
    let s11 = EtcDecode::R11Snorm;
    assert_eq!(value(s11, s11.block_values(&zero)[3]), -3.0 / 1023.0);
    assert_eq!(EtcDecode::Rg11Unorm.block_values(&zero)[0], [1, 1]);
}

/// Each mode of an RGB block, by its bits: the subblocks and their tables, the T, H and planar
/// modes, and punch-through's transparent index.
#[test]
fn each_mode_decodes_by_the_specification() {
    let texel = |decode: EtcDecode, v: u64, x: usize, y: usize| {
        decode.block_values(&block(v))[4 * y + x][0].to_le_bytes()
    };
    // Individual mode, flipped: the top subblock red 0x8 (136) with table 0, the bottom one blue
    // 0xf (255) with table 7. Texel (0, 0) index 0 (+2); texel (0, 3) index 3 (-183): its msb is
    // bit 16 + 3 and its lsb bit 3.
    let v: u64 = 0x80 << 56 | 0x0f << 40 | 0b000_111 << 34 | 1 << 32 | 1 << 19 | 1 << 3;
    assert_eq!(texel(EtcDecode::Rgb8, v, 0, 0), [138, 2, 2, 255]);
    assert_eq!(texel(EtcDecode::Rgb8, v, 0, 3), [0, 0, 72, 255]);
    // Differential mode: red 16 with delta -1, so the right subblock's red is 15; unflipped.
    let v: u64 = 16 << 59 | 0b111 << 56 | 1 << 33;
    assert_eq!(texel(EtcDecode::Rgb8, v, 0, 0), [134, 2, 2, 255]);
    assert_eq!(texel(EtcDecode::Rgb8, v, 3, 0), [125, 2, 2, 255]);
    // T mode: red 31 + delta 1 overflows. c1 = (0xf, 0, 0), c2 = (0, 0xf, 0), distance index 7
    // (64): index 0 paints c1, index 1 c2 + 64, index 3 c2 - 64.
    let t: u64 =
        31 << 59 | 1 << 56 | 0b11 << 59 | 0b11 << 56 | 0xf << 40 | 0b11 << 34 | 1 << 33 | 1 << 32;
    // Texel 0: index 0; texel 1 (0, 1): index 1 (lsb bit 1); texel 2 (0, 2): index 3.
    let t = t | 1 << 1 | 1 << 18 | 1 << 2;
    assert_eq!(texel(EtcDecode::Rgb8, t, 0, 0), [255, 0, 0, 255]);
    assert_eq!(texel(EtcDecode::Rgb8, t, 0, 1), [64, 255, 64, 255]);
    assert_eq!(texel(EtcDecode::Rgb8, t, 0, 2), [0, 191, 0, 255]);
    // Without the opaque flag, RGB8A1 reads T mode's index 2 as transparent black: texel (1, 0)
    // is texel number 4, msb bit 20.
    let t_clear = (t & !(1 << 33)) | 1 << 20;
    assert_eq!(texel(EtcDecode::Rgb8A1, t_clear, 1, 0), [0, 0, 0, 0]);
    assert_eq!(texel(EtcDecode::Rgb8, t | 1 << 20, 1, 0), [0, 255, 0, 255]);
    // H mode: red 15 + delta 0 fits, green 0 + delta -4 overflows (bit 50, which H mode does
    // not read). c1 = (0xf, 0, 0) and c2 = 0, so c1 >= c2 sets the distance index's low bit:
    // index 1, distance 6, and index 0 paints c1 + 6.
    let h: u64 = 0xf << 59 | 0b100 << 48 | 1 << 33;
    assert_eq!(texel(EtcDecode::Rgb8, h, 0, 0), [255, 6, 6, 255]);
    // Planar mode: blue 31 + delta 1 overflows. O = (0, 0, 0) where its bits are clear; H's red
    // 63 (255) and V's 0: red climbs by (x (255 - O) + 2) >> 2 across. The overflow bits (blue
    // 31 at bits 47..43 and delta 1 at 40) carry no color in this mode.
    let p: u64 = 31 << 43 | 1 << 40 | 1 << 33 | 0b11111 << 34 | 1 << 32;
    let reds: Vec<u8> = (0..4).map(|x| texel(EtcDecode::Rgb8, p, x, 0)[0]).collect();
    assert_eq!(reds, [0, 64, 128, 191]);
}

/// EAC's modifiers and multipliers: alpha clamps to a byte, an 11-bit value of a nonzero
/// multiplier steps by `8 m`, a signed base of -128 reads as -127.
#[test]
fn eac_blocks_decode_by_the_specification() {
    // Base 200, multiplier 15, table 0: index 7 (+14) clamps alpha to 255, index 3 (-15)
    // reads 200 - 225, clamped to 0.
    let v: u64 = 200 << 56 | 15 << 52 | 7 << 45 | 3 << 42;
    let alpha =
        |p: usize| EtcDecode::Rgba8.block_values(&[block(v), [0; 8]].concat())[p][0] as u32 >> 24;
    assert_eq!(alpha(0), 255);
    assert_eq!(alpha(4), 0);
    // Unsigned: base 100, multiplier 2, table 13, index 6 (+2): 804 + 32 = 836.
    let v: u64 = 100 << 56 | 2 << 52 | 13 << 48 | 6 << 45;
    assert_eq!(EtcDecode::R11Unorm.block_values(&block(v))[0], [836, 0]);
    // Signed: base -128 reads -127, multiplier 1, table 0, index 3 (-15): -1016 - 120 clamps
    // to -1023.
    let v: u64 = 0x80 << 56 | 1 << 52 | 3 << 45;
    assert_eq!(EtcDecode::R11Snorm.block_values(&block(v))[0], [-1023, 0]);
    // Index 4 (+2): -1016 + 16 = -1000.
    let v: u64 = 0x80 << 56 | 1 << 52 | 4 << 45;
    assert_eq!(EtcDecode::R11Snorm.block_values(&block(v))[0], [-1000, 0]);
}

/// `n` bytes of the xorshift64* stream from `seed`, each word little-endian.
fn stream(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    out.truncate(n);
    out
}

/// FNV-1a, 64 bits.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A 256 x 256 texture of 4096 random blocks of each format, every texel decoded: the digest of
/// the exact twin's texels is that of the texels an Apple M3 Max under macOS 27.0.1 reads from the
/// same blocks (each 8-bit channel `c / 255` as its code `c`, each 11-bit channel's `float` bits).
#[test]
fn random_blocks_decode_as_metal_reads_them() {
    let eight = stream(0x9E37_79B9_7F4A_7C15, 32768);
    let sixteen = stream(0xD1B5_4A32_D192_ED03, 65536);
    let cases = [
        (EtcDecode::Rgb8, &eight, 0x8782_0f87_1ba3_b4ad_u64),
        (EtcDecode::Rgb8A1, &eight, 0x03a5_c438_f97c_d615),
        (EtcDecode::Rgba8, &sixteen, 0xca13_5c88_829c_cc4a),
        (EtcDecode::R11Unorm, &eight, 0xcfab_277c_ac91_845d),
        (EtcDecode::R11Snorm, &eight, 0x035f_1fa4_f6d2_5985),
        (EtcDecode::Rg11Unorm, &sixteen, 0x85f3_0ebd_d7f4_627d),
        (EtcDecode::Rg11Snorm, &sixteen, 0x0926_22a9_720a_4150),
    ];
    for (decode, blocks, metal) in cases {
        let texels = decode.decode(EtcTexels::Float, blocks, [256, 256, 1]);
        assert_eq!(fnv(&texels), metal, "{decode:?}");
    }
}

/// A region's texels lie as a copy into the twin reads them: an edge block's texels past the
/// region are not written, and each layer's follow the last.
#[test]
fn a_region_decodes_its_own_texels() {
    let blocks = stream(7, 8 * 4);
    // 6 x 3 texels of 2 layers: 2 x 1 blocks a layer.
    let texels = EtcDecode::Rgb8.decode(EtcTexels::Compact, &blocks, [6, 3, 2]);
    assert_eq!(texels.len(), 6 * 3 * 2 * 4);
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..6 {
                let b = z * 2 + x / 4;
                let want = EtcDecode::Rgb8.block_values(&blocks[b * 8..b * 8 + 8])[4 * y + x % 4]
                    [0]
                .to_le_bytes();
                let at = ((z * 3 + y) * 6 + x) * 4;
                assert_eq!(texels[at..at + 4], want, "({x}, {y}, {z})");
            }
        }
    }
}

/// The compact twin's 16-bit code of every 11-bit value is the one nearest Metal's value: within
/// half a step of it, `1/131070` unsigned and `1/65534` signed.
#[test]
fn the_compact_codes_are_the_nearest() {
    for (decode, values, top) in [
        (EtcDecode::R11Unorm, 0..=2047, 65535.0),
        (EtcDecode::R11Snorm, -1023..=1023, 32767.0),
    ] {
        for c in values {
            let mut out = [0u8; 2];
            decode.twin_texel(EtcTexels::Compact, [c, 0], &mut out);
            let code = i16::from_le_bytes(out);
            let read = match decode {
                EtcDecode::R11Snorm => f64::from(code) / top,
                _ => f64::from(code as u16) / top,
            };
            let metal = f64::from(c)
                / match decode {
                    EtcDecode::R11Snorm => 1023.0,
                    _ => 2047.0,
                };
            assert!(
                (read - metal).abs() <= 0.5 / top + 1e-12,
                "{decode:?} {c}: {read} against {metal}"
            );
        }
    }
    // The ends are exact.
    let mut out = [0u8; 4];
    EtcDecode::Rg11Snorm.twin_texel(EtcTexels::Compact, [-1023, 1023], &mut out);
    assert_eq!(out, [0x01, 0x80, 0xff, 0x7f]);
}
