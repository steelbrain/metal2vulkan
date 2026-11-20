//! The ASTC decode against blocks the specification's rules decide outright, and against Metal's
//! reads of seeded random blocks on an Apple M3 Max (macOS 27.0.1).

mod hdr;

use super::{AstcDecode, AstcTexels, Illegal};
use crate::metal::PixelFormat;

/// Every ASTC format decodes, linear, sRGB and HDR, of the fourteen 2D footprints Vulkan's ASTC
/// formats have and of Apple's 4x2 and 8x4, and no other format does.
#[test]
fn every_astc_format_of_every_profile_alone_decodes() {
    let decoded: Vec<(PixelFormat, AstcDecode)> = PixelFormat::ALL
        .iter()
        .filter_map(|&pf| AstcDecode::of(pf).map(|d| (pf, d)))
        .collect();
    assert_eq!(decoded.len(), 16 * 3);
    for (pf, d) in decoded {
        let name = pf.name();
        let footprint = format!("_{}x{}_", d.width, d.height);
        assert!(name.contains(&footprint), "{name}: {d:?}");
        assert_eq!(name.ends_with("_sRGB"), d.srgb, "{name}");
        assert_eq!(name.ends_with("_HDR"), d.hdr, "{name}");
        assert!(name.ends_with("_LDR") || d.srgb || d.hdr, "{name}");
    }
    for pf in [
        PixelFormat::BC7RGBAUnorm,
        PixelFormat::PVRTCRGBA4BPP,
        PixelFormat::RGBA8Unorm,
    ] {
        assert_eq!(AstcDecode::of(pf), None, "{pf:?}");
    }
}

fn eight() -> AstcDecode {
    AstcDecode::of(PixelFormat::ASTC8x8LDR).expect("a decode")
}

/// A void-extent block reads its four 16-bit values everywhere; an HDR one, one whose reserved
/// bits 10 and 11 are not both set (Metal reads it magenta), one of an empty extent, and the zero
/// block (a reserved block mode, a new texture's) are error blocks.
#[test]
fn void_extent_and_error_blocks_read_as_the_rules_say() {
    let d = eight();
    let color = [0x1234u16, 0xfedc, 0x0000, 0xffff];
    let block = |low: u64| {
        let high = color
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &c)| acc | u64::from(c) << (16 * i));
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&low.to_le_bytes());
        bytes[8..].copy_from_slice(&high.to_le_bytes());
        bytes
    };
    // Mode bits 0-8 `0x1FC`, reserved bits 10-11 set, every extent coordinate all ones.
    let constant = 0xDFC | ((1u64 << 52) - 1) << 12;
    assert_eq!(d.block_values(&block(constant)), Some(vec![color; 64]));
    let why = |low: u64| d.block_values_why(&block(low)).err();
    assert_eq!(why(constant | 1 << 9), Some(Illegal::HdrVoidExtent));
    assert_eq!(
        why(constant & !(1 << 11)),
        Some(Illegal::VoidExtentReserved)
    );
    // Extents of minimum 5 and maximum 5 across.
    let empty = 0xDFC | 5 << 12 | 5 << 25 | 1 << 38 | 9 << 51;
    assert_eq!(why(empty), Some(Illegal::VoidExtentEmpty));
    assert_eq!(
        d.block_values_why(&[0; 16]).err(),
        Some(Illegal::ReservedMode)
    );
    // Error blocks read magenta: every channel 0 or 1.
    let mut out = vec![0; 64 * 16];
    d.decode_into(AstcTexels::Float, &[0; 16], [8, 8, 1], &mut out);
    let first: Vec<f32> = out[..16]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&w| f32::from_le_bytes(w))
        .collect();
    assert_eq!(first, [1.0, 0.0, 1.0, 1.0]);
}

/// A block built by the specification's layout: block mode `0x213` (a 4 x 2 grid of 32-level
/// weights, 40 bits), one partition, luminance direct (mode 0) of the 8-bit endpoints `0x40` and
/// `0xC0` (256 levels fit in the 71 colour bits). Weights of 0 read the first endpoint widened,
/// `e · 257`; weights of 64 (all ones) the second; each read is `(C >> 4) / 4095`.
#[test]
fn a_block_of_the_specification_s_layout_reads_its_endpoints() {
    let d = eight();
    let base: u128 = 0x213 | 0x40 << 17 | 0xC0 << 25;
    let zero = d.block_values(&base.to_le_bytes()).expect("valid");
    assert!(zero.iter().all(|&t| t == [0x4040, 0x4040, 0x4040, 0xffff]));
    let ones = base | ((1u128 << 40) - 1) << 88;
    let one = d.block_values(&ones.to_le_bytes()).expect("valid");
    assert!(one.iter().all(|&t| t == [0xc0c0, 0xc0c0, 0xc0c0, 0xffff]));
    let value = d.metal_value([0x4040, 0, 0xffff, 0]);
    assert_eq!(value, [1028.0 / 4095.0, 0.0, 1.0, 0.0]);
    // The sRGB form widens to `e · 256 + 128`, reads the step of `C >> 8` and alpha over 255.
    let srgb = AstcDecode::of(PixelFormat::ASTC8x8SRGB).expect("a decode");
    let texels = srgb.block_values(&base.to_le_bytes()).expect("valid");
    assert!(texels
        .iter()
        .all(|&t| t == [0x4080, 0x4080, 0x4080, 0xff80]));
    let step = f32::from(crate::lower::arith::srgb::SRGB_DECODE_STEPS[0x40]);
    assert_eq!(srgb.metal_value(texels[0])[0], step / 4095.0);
    assert_eq!(srgb.metal_value(texels[0])[3], 1.0);
}

/// The blocks Metal read for the committed digests: `count` blocks of xorshift64* from
/// `seed | 1`, two words each (low first) and a third draw, which makes 1 in 8 an LDR void-extent
/// block (reserved bits set), half of those with all-ones extents.
fn seeded_blocks(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut out = Vec::with_capacity(count * 16);
    for _ in 0..count {
        let (mut low, high, k) = (next(), next(), next());
        if k % 8 == 0 {
            low = (low & !0xFFF) | 0xDFC;
            if (k >> 8) & 1 == 1 {
                low |= ((1 << 52) - 1) << 12;
            }
        }
        out.extend_from_slice(&low.to_le_bytes());
        out.extend_from_slice(&high.to_le_bytes());
    }
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Every texel of 1024 seeded blocks (32 x 32 of them, seeded by the LDR form's raw value) of each
/// footprint, linear and sRGB, Apple's 4x2 and 8x4 among them, is Metal's read: the `float`
/// texels' FNV-1a digest equals the digest of what an Apple M3 Max (macOS 27.0.1) read of the same
/// blocks (`texture.read` into `float4`, our probes). Of each footprint 86 (Apple's 4x2) to 283
/// blocks are valid, the rest error blocks of every kind; the probe's wider run (4096 valid-biased
/// blocks a footprint) matched as well.
#[test]
fn seeded_blocks_read_as_metal_read_them() {
    const DIGESTS: [(u32, u64, u64); 16] = [
        (204, 0x1ec2_3532_f7f1_3267, 0xeac0_f14e_3953_0337),
        (205, 0x8e4e_246a_e848_6e2a, 0xb1cf_2627_d93f_c546),
        (206, 0xc623_781c_7ea9_9899, 0xdfe3_b392_ea47_125d),
        (207, 0xdd08_c14c_7a5f_3780, 0x0420_3c4a_f814_a7a0),
        (208, 0xe44f_f52e_9286_d9c4, 0xf220_12e7_30fa_4242),
        (210, 0x564f_e279_bb73_0b0f, 0x01db_e80d_55f7_ef64),
        (211, 0x85b6_584e_6ad0_f498, 0xb7d5_305f_87e2_ed09),
        (212, 0x0f54_d281_abd1_482e, 0x1726_b4ef_e2b7_f156),
        (213, 0x2cf6_29e2_d4d2_8730, 0xf660_ea51_94d0_1e84),
        (214, 0xf6e3_2d62_dc3f_0a81, 0x74ed_bfbb_ce3d_0597),
        (215, 0x340c_e23a_543b_2199, 0x69fe_1b2e_0cb5_cb34),
        (216, 0xe5eb_eef0_6b57_ea2c, 0x979d_0419_fa81_a371),
        (217, 0x3473_96ec_0297_8568, 0x4533_32fd_ec0e_d21b),
        (218, 0xa285_9133_6da5_5eff, 0x87a5_68d7_f82c_594a),
        // Apple's 4x2 and 8x4 footprints (the probe `astc_passes_writes_footprints`, `digest`).
        (203, 0x0ce8_1b3e_88fe_e772, 0x0e48_5529_95cd_c8c2),
        (209, 0x5acd_5090_d21d_770b, 0x3663_c642_a5fa_92aa),
    ];
    for (raw, linear, srgb) in DIGESTS {
        let blocks = seeded_blocks(u64::from(raw), 1024);
        for (raw, digest) in [(raw, linear), (raw - 18, srgb)] {
            let pf = PixelFormat::try_from(raw).expect("an ASTC format");
            let d = AstcDecode::of(pf).expect("a decode");
            let size = [32 * d.width, 32 * d.height, 1];
            let texels = d.decode(AstcTexels::Float, &blocks, size);
            assert_eq!(fnv1a(&texels), digest, "{pf:?}");
            let valid = blocks
                .as_chunks::<16>()
                .0
                .iter()
                .filter(|b| d.block_values(b).is_some())
                .count();
            assert!(valid > 80, "{pf:?}: {valid} valid blocks");
        }
    }
}

/// The 8-bit twin holds the code nearest each linear value (at most 1/510 off it) and an sRGB
/// format's own codes `C >> 8`, whose alpha reads exactly.
#[test]
fn the_eight_bit_twin_holds_the_nearest_codes() {
    let blocks = seeded_blocks(212, 256);
    for pf in [PixelFormat::ASTC8x8LDR, PixelFormat::ASTC8x8SRGB] {
        let d = AstcDecode::of(pf).expect("a decode");
        let size = [16 * 8, 16 * 8, 1];
        let exact = d.decode(AstcTexels::Float, &blocks, size);
        let codes = d.decode(AstcTexels::Unorm8, &blocks, size);
        assert_eq!(codes.len() * 4, exact.len());
        for (i, (&code, &value)) in codes.iter().zip(exact.as_chunks::<4>().0).enumerate() {
            let ch = i % 4;
            let value = f32::from_le_bytes(value);
            let read = f32::from(code) / 255.0;
            match (d.srgb, ch) {
                (true, 3) => assert_eq!(read, value),
                (true, _) => {
                    let step = crate::lower::arith::srgb::SRGB_DECODE_STEPS[usize::from(code)];
                    assert_eq!(f32::from(step) / 4095.0, value);
                }
                (false, _) => assert!((read - value).abs() <= 1.0 / 510.0, "{read} {value}"),
            }
        }
    }
}
