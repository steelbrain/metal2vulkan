//! The HDR profile of the ASTC decode against Metal's reads of the same blocks on an Apple M3 Max
//! (macOS 27.0.1, our probe `astc_hdr_textures`, its `reads` case: `texture.read` as `float4` and
//! as `half4`, and a nearest `sample` at each texel's centre, all three the same bits), and the
//! GPU's decode against the CPU's.

use super::{assert_kernel_decodes, fnv1a, modes_of, two_layers};
use crate::bind::astc_decode::kernel::typed_module;
use crate::bind::astc_decode::{hdr, AstcDecode, AstcTexels};
use crate::metal::PixelFormat;
use crate::spirv::interp::Order;
use crate::spirv::testing::{assert_valid, committed_profiles};
use crate::vk::Format;

/// The sixteen `_HDR` formats.
fn hdr_formats() -> Vec<PixelFormat> {
    (PixelFormat::ALL.iter().copied())
        .filter(|&pf| AstcDecode::of(pf).is_some_and(|d| d.hdr))
        .collect()
}

/// The blocks Metal read for the committed digests: `count` blocks of xorshift64* from
/// `seed | 1`, two words each (low first) and a third draw, which makes 1 in 16 an LDR
/// void-extent block and 1 in 16 an HDR one (all-ones extents), a quarter of those with a red of
/// an infinity or a NaN.
fn hdr_seeded_blocks(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut out = Vec::with_capacity(count * 16);
    for _ in 0..count {
        let (mut low, mut high, k) = (next(), next(), next());
        let extents = ((1 << 52) - 1) << 12;
        match k % 16 {
            0 => low = (low & !0xFFF) | 0xDFC | extents,
            1 => {
                low = (low & !0xFFF) | 0xFFC | extents;
                if (k >> 8) & 3 == 0 {
                    let red = [0x7C00, 0xFC00, 0x7E00, 0xFFFF][((k >> 10) & 3) as usize];
                    high = (high & !0xFFFF) | red;
                }
            }
            _ => {}
        }
        out.extend_from_slice(&low.to_le_bytes());
        out.extend_from_slice(&high.to_le_bytes());
    }
    out
}

/// The blocks across and down of each digest's region.
const ACROSS: u32 = 64;
const DOWN: u32 = 32;

/// The FNV-1a digest of the `float` texels Metal read of `hdr_seeded_blocks(raw · 7919, 2048)`,
/// 64 x 32 blocks of each `_HDR` format, row-major.
const DIGESTS: [(u32, u64); 16] = [
    (222, 0x6a09_0dd1_d708_385e),
    (223, 0x5133_22f1_31e3_5360),
    (224, 0xe57f_f92a_3ad4_6a73),
    (225, 0x97e6_b9fe_9af5_e71f),
    (226, 0x58b4_5167_a9b1_5511),
    (228, 0x22f1_9ce0_a470_9156),
    (229, 0xb86d_7e1b_f6d2_b26a),
    (230, 0xad74_7148_f5d3_864a),
    (231, 0xb01a_d9b4_7552_08fc),
    (232, 0xceda_c123_1d34_03f2),
    (233, 0xdfce_a1ea_d3ab_ae0e),
    (234, 0x7bed_6c2f_1759_6320),
    (235, 0x5952_aae3_8906_bbe0),
    (236, 0x57c9_02f4_a42a_e0d1),
    // Apple's 4x2 and 8x4 footprints (the probe `astc_passes_writes_footprints`, `digest`).
    (221, 0x4273_ba75_4057_14f4),
    (227, 0xd465_4f44_6d5e_52ba),
];

/// Every texel of 2048 seeded blocks of each footprint (270, Apple's 4x2, to 629 of them valid,
/// decoding through every colour endpoint mode, two to four partitions and two planes; 1 in 16 an
/// HDR void-extent block and 1 in 16 an LDR one) reads as Metal read it, bit for bit.
#[test]
fn hdr_seeded_blocks_read_as_metal_read_them() {
    for (raw, digest) in DIGESTS {
        let pf = PixelFormat::try_from(raw).expect("an ASTC HDR format");
        let d = AstcDecode::of(pf).expect("a decode");
        let blocks = hdr_seeded_blocks(u64::from(raw) * 7919, (ACROSS * DOWN) as usize);
        let size = [ACROSS * d.width, DOWN * d.height, 1];
        let texels = d.decode(AstcTexels::Float, &blocks, size);
        assert_eq!(fnv1a(&texels), digest, "{pf:?}");
        let valid = (blocks.as_chunks::<16>().0.iter())
            .filter(|b| d.block_values(b).is_some())
            .count();
        assert!(valid > 250, "{pf:?}: {valid} valid blocks");
    }
}

/// Blocks of the seeded sets above and the `float` bits Metal read at three of each one's
/// texels (row-major in the block): an error block; each colour endpoint mode alone, one plane
/// and two; two to four partitions of HDR modes, LDR modes and both; HDR void-extent blocks of
/// finite, negative, infinite and NaN reds and LDR ones; and a valid block of each footprint.
#[rustfmt::skip]
const METAL_READS: [(u32, u128, [(u32, [u32; 4]); 3]); 66] = [
    // err
    (222, 0xe86d49d569a646c5_157a20b628dfb331, [(0, [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]), (15, [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]), (10, [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000])]),
    // void-ldr
    (222, 0xb9ae249842aac39d_fffffffffffffdfc, [(0, [0x3f438000, 0x3e854000, 0x3e126000, 0x3f39a000]), (15, [0x3f438000, 0x3e854000, 0x3e126000, 0x3f39a000]), (10, [0x3f438000, 0x3e854000, 0x3e126000, 0x3f39a000])]),
    // void-hdr-inf
    (222, 0x2a617e9f121bfc00_fffffffffffffffc, [(0, [0xff800000, 0x3a436000, 0x7fc00000, 0x3d4c2000]), (15, [0xff800000, 0x3a436000, 0x7fc00000, 0x3d4c2000]), (10, [0xff800000, 0x3a436000, 0x7fc00000, 0x3d4c2000])]),
    // void-hdr-pos
    (222, 0xd2321ef1100b56e5_fffffffffffffffc, [(0, [0x42dca000, 0x3a016000, 0x3bde2000, 0xc2464000]), (15, [0x42dca000, 0x3a016000, 0x3bde2000, 0xc2464000]), (10, [0x42dca000, 0x3a016000, 0x3bde2000, 0xc2464000])]),
    // void-hdr-neg
    (222, 0xd3065db68753929a_fffffffffffffffc, [(0, [0xba534000, 0xb8ea6000, 0x43b6c000, 0xc260c000]), (15, [0xba534000, 0xb8ea6000, 0x43b6c000, 0xc260c000]), (10, [0xba534000, 0xb8ea6000, 0x43b6c000, 0xc260c000])]),
    // cem 7 dual
    (222, 0x7f3ccd72d9677ace_89182f01c2fce74d, [(0, [0x00000000, 0x3b0cc000, 0x3b370000, 0x3f800000]), (15, [0x00000000, 0x3b8ba000, 0x3b610000, 0x3f800000]), (10, [0x00000000, 0x3b2b8000, 0x3b314000, 0x3f800000])]),
    // cem 13 dual
    (222, 0x16f0ffb731cdb936_d1b6a41a16eba75e, [(0, [0x3f0e4000, 0x3e87c000, 0x3ec68000, 0x3f6a8000]), (15, [0x3f1ec000, 0x3e982000, 0x3ee8a000, 0x3f5cc000]), (10, [0x3f212000, 0x3e90c000, 0x3eed6000, 0x3f5ae000])]),
    // void-hdr-nan
    (222, 0xb88fda898562ffff_fffffffffffffffc, [(0, [0x7fc00000, 0xb8ac4000, 0xc3512000, 0xbf11e000]), (15, [0x7fc00000, 0xb8ac4000, 0xc3512000, 0xbf11e000]), (10, [0x7fc00000, 0xb8ac4000, 0xc3512000, 0xbf11e000])]),
    // cem 13
    (222, 0x9c9da9f6151198c5_3068a75cc72da242, [(0, [0x3ef8c000, 0x3f370000, 0x3f612000, 0x3e87e000]), (15, [0x3efa6000, 0x3f388000, 0x3f624000, 0x3e8b8000]), (10, [0x3ef72000, 0x3f35a000, 0x3f602000, 0x3e842000])]),
    // cem 15 dual
    (222, 0x40fab47bc47aabd5_d083ed4316e7e53f, [(0, [0x44124000, 0x44048000, 0x44352000, 0x432f4000]), (15, [0x44350000, 0x440c8000, 0x446fc000, 0x42f60000]), (10, [0x4422a000, 0x44100000, 0x44500000, 0x4311a000])]),
    // cem 0 dual
    (222, 0xead83c4fb9532207_79427839945405bd, [(0, [0x3ef4e000, 0x3ef4e000, 0x3ef4e000, 0x3f800000]), (15, [0x3e28a000, 0x3e28a000, 0x3e28a000, 0x3f800000]), (10, [0x3e908000, 0x3e908000, 0x3e908000, 0x3f800000])]),
    // 3 partitions mixed
    (222, 0x5e3ba09ce600a381_0b8bde1a7ce53212, [(0, [0x3f7c2000, 0x3f7c2000, 0x3f7c2000, 0x3f800000]), (15, [0x3e4e2000, 0x3e4e2000, 0x3e4e2000, 0x3e352000]), (10, [0x38e52000, 0x38eb8000, 0x38dee000, 0x3f800000])]),
    // cem 11 dual
    (222, 0xa293be3420c21a54_df95e4f03b916433, [(0, [0x428f6000, 0x3d812000, 0x41580000, 0x3f800000]), (15, [0x443a4000, 0x3e868000, 0x3db80000, 0x3f800000]), (10, [0x3b16e000, 0x38eb6000, 0x3e3a8000, 0x3f800000])]),
    // 2 partitions hdr dual
    (222, 0x3e9fdb8b7ecb39a0_0615778998a8ef3d, [(0, [0x43936000, 0x43936000, 0x43936000, 0x3f800000]), (15, [0x439f6000, 0x439f6000, 0x439f6000, 0x3f800000]), (10, [0x3f8b4000, 0x3f8b4000, 0x3f8b4000, 0x3f800000])]),
    // 4 partitions mixed
    (222, 0x8c0257a00f3ab443_95f69a7b80c1ba23, [(0, [0x3e5d6000, 0x3e5d6000, 0x3e5d6000, 0x3f800000]), (15, [0x3e58a000, 0x3e58a000, 0x3e58a000, 0x3f800000]), (10, [0x3e4c8000, 0x3e4c8000, 0x3e4c8000, 0x3f800000])]),
    // 3 partitions ldr
    (222, 0x9931371b6b2d30c3_ac9f206504c7334f, [(0, [0x3f188000, 0x3f188000, 0x3f188000, 0x3f800000]), (15, [0x3f11a000, 0x3f11a000, 0x3f11a000, 0x3f800000]), (10, [0x3f12e000, 0x3f12e000, 0x3f12e000, 0x3f800000])]),
    // 2 partitions mixed dual
    (222, 0xaf614c8769713eb9_54124097c8a5cf3e, [(0, [0x42400000, 0x42400000, 0x42400000, 0x3f800000]), (15, [0x3e3e2000, 0x3e3e2000, 0x3e400000, 0x3f800000]), (10, [0x42400000, 0x42400000, 0x42400000, 0x3f800000])]),
    // cem 5 dual
    (222, 0x6e43ae48152763c2_ccd154d0e7fca78e, [(0, [0x3ef98000, 0x3ef8a000, 0x3ef98000, 0x3f3ce000]), (15, [0x3efb4000, 0x3ef1c000, 0x3efb4000, 0x3f3a4000]), (10, [0x3ef98000, 0x3ef4e000, 0x3ef98000, 0x3f3ce000])]),
    // cem 4
    (222, 0xf6bfaa834f974096_e37b3777447483be, [(0, [0x3ef06000, 0x3ef06000, 0x3ef06000, 0x3f28a000]), (15, [0x3ec94000, 0x3ec94000, 0x3ec94000, 0x3f2ea000]), (10, [0x3f172000, 0x3f172000, 0x3f172000, 0x3f1f0000])]),
    // cem 3 dual
    (222, 0xc02b3a053236fbce_8332bb0a46dc674d, [(0, [0x3ad9a000, 0x3ad9a000, 0x3ad9a000, 0x3f800000]), (15, [0x3ad6a000, 0x3ad6a000, 0x3ad6a000, 0x3f800000]), (10, [0x3ad82000, 0x3ad82000, 0x3ad82000, 0x3f800000])]),
    // 3 partitions ldr dual
    (222, 0x7a25fbfd653a04ea_1716d05b11d7f59f, [(0, [0x3f732000, 0x3f000000, 0x3e108000, 0x3f800000]), (15, [0x00000000, 0x3e774000, 0x3f082000, 0x3f800000]), (10, [0x00000000, 0x3e878000, 0x3f110000, 0x3f800000])]),
    // cem 6 dual
    (222, 0x145eaf1c46d5d07c_b095f941c458c411, [(0, [0x3e2ca000, 0x3f62e000, 0x3f1d8000, 0x3f800000]), (15, [0x3e30a000, 0x3f5ec000, 0x3f20a000, 0x3f800000]), (10, [0x3e2e0000, 0x3f618000, 0x3f1ea000, 0x3f800000])]),
    // 2 partitions ldr dual
    (222, 0xa3fa1686f58ef6d8_f16860e0347c2d8f, [(0, [0x3f038000, 0x3f28e000, 0x3eb72000, 0x3f72e000]), (15, [0x3f5ec000, 0x3e928000, 0x00000000, 0x3f680000]), (10, [0x3f740000, 0x3ea0a000, 0x00000000, 0x3f68e000])]),
    // cem 2
    (222, 0x94661b82420a85e6_1c79fc9c9326434d, [(0, [0x3f210000, 0x3f210000, 0x3f210000, 0x3f800000]), (15, [0x3f210000, 0x3f210000, 0x3f210000, 0x3f800000]), (10, [0x3eb38000, 0x3eb38000, 0x3eb38000, 0x3f800000])]),
    // cem 1
    (222, 0xf51d03be2199961b_727f1e26642e235d, [(0, [0x3e050000, 0x3e050000, 0x3e050000, 0x3f800000]), (15, [0x3e14a000, 0x3e14a000, 0x3e14a000, 0x3f800000]), (10, [0x3da5a000, 0x3da5a000, 0x3da5a000, 0x3f800000])]),
    // 2 partitions hdr
    (222, 0xfa36981ad786e1f3_38c19663b7ce2853, [(0, [0x3ed00000, 0x3e8c0000, 0x3e400000, 0x00000000]), (15, [0x3e9c8000, 0x3e2ca000, 0x3eae0000, 0x3ec80000]), (10, [0x3e9d6000, 0x3e6f6000, 0x3de80000, 0x3df62000])]),
    // 3 partitions hdr dual
    (222, 0x6756f515114601b7_647090f3ee43978d, [(0, [0x42582000, 0x427b2000, 0x42832000, 0x3f800000]), (15, [0x3eeb2000, 0x3f8de000, 0x3f612000, 0x3f800000]), (10, [0x3fb38000, 0x3f8de000, 0x402b8000, 0x3f800000])]),
    // 3 partitions mixed dual
    (222, 0x45779ed466bb7392_20bda394209fd602, [(0, [0x3f536000, 0x3f77a000, 0x3f77a000, 0x3f800000]), (15, [0x3f3b6000, 0x3f46e000, 0x3f46e000, 0x3f800000]), (10, [0x40782000, 0x40ffa000, 0x40ffa000, 0x3f800000])]),
    // 3 partitions hdr
    (222, 0x5f3ec8540ba9e096_479b8542a41e3221, [(0, [0x39b64000, 0x39b64000, 0x39b64000, 0x3f800000]), (15, [0x3bcc4000, 0x3bcc4000, 0x3bcc4000, 0x3f800000]), (10, [0x405d0000, 0x405d0000, 0x405d0000, 0x3f800000])]),
    // 2 partitions ldr
    (222, 0x8fa7b6c29c3747a7_51b438f577110823, [(0, [0x3f4bc000, 0x3ef6e000, 0x3d988000, 0x3f72e000]), (15, [0x3d8b8000, 0x3ddde000, 0x3e24a000, 0x3ec08000]), (10, [0x3e2a2000, 0x3e85c000, 0x3ec2c000, 0x3eba6000])]),
    // cem 1 dual
    (222, 0x46fb287e3a0fa01e_6bbd5b19100e2621, [(0, [0x3f026000, 0x3f042000, 0x3f026000, 0x3f800000]), (15, [0x3f06e000, 0x3f07c000, 0x3f06e000, 0x3f800000]), (10, [0x3f01c000, 0x3f06e000, 0x3f01c000, 0x3f800000])]),
    // cem 4 dual
    (222, 0x54f53c8e7ed7eb31_17b832999122871d, [(0, [0x3f29a000, 0x3f29a000, 0x3f29a000, 0x3e7ac000]), (15, [0x3f15e000, 0x3f15e000, 0x3f15e000, 0x3e86e000]), (10, [0x3f27e000, 0x3f27e000, 0x3f27e000, 0x3e5de000])]),
    // 2 partitions mixed
    (222, 0x7c9131ffa527862f_5e6ea1648299abdf, [(0, [0x436ec000, 0x436ec000, 0x436ec000, 0x3f800000]), (15, [0x43420000, 0x43420000, 0x43420000, 0x3f800000]), (10, [0x434f2000, 0x434f2000, 0x434f2000, 0x3f800000])]),
    // cem 0
    (222, 0xb916065c401501df_7cbbfad637fa03df, [(0, [0x3e252000, 0x3e252000, 0x3e252000, 0x3f800000]), (15, [0x3ea7a000, 0x3ea7a000, 0x3ea7a000, 0x3f800000]), (10, [0x3f6c4000, 0x3f6c4000, 0x3f6c4000, 0x3f800000])]),
    // cem 14
    (222, 0xadf223ea7a0a0079_24f44d867077c1ce, [(0, [0x3f89c000, 0x4051a000, 0x404d8000, 0x3e21e000]), (15, [0x404c0000, 0x40d80000, 0x40e70000, 0x00000000]), (10, [0x3f67e000, 0x403bc000, 0x40350000, 0x3e3c2000])]),
    // cem 3
    (222, 0x3519a1d947cc55ab_88135e645fd861de, [(0, [0x3be70000, 0x3be70000, 0x3be70000, 0x3f800000]), (15, [0x3bc60000, 0x3bc60000, 0x3bc60000, 0x3f800000]), (10, [0x3bdb4000, 0x3bdb4000, 0x3bdb4000, 0x3f800000])]),
    // cem 10
    (222, 0x2e3c80c7e18ea016_f005831942f3439f, [(0, [0x3ede2000, 0x3f13c000, 0x3f006000, 0x3e9f6000]), (15, [0x3ed88000, 0x3f100000, 0x3efa4000, 0x3e894000]), (10, [0x3ed20000, 0x3f0ba000, 0x3ef2a000, 0x3e5ea000])]),
    // cem 10 dual
    (222, 0xd89763279c0e26c9_e6201562a13b473e, [(0, [0x3eb1c000, 0x3d242000, 0x3ec7c000, 0x3f0ca000]), (15, [0x3ee12000, 0x3cd46000, 0x3efd6000, 0x3f304000]), (10, [0x3eef6000, 0x3e304000, 0x3f06c000, 0x3f3ae000])]),
    // cem 8
    (222, 0x2fcdad61105d593a_5f003d0a44c503cf, [(0, [0x3e82a000, 0x3e8bc000, 0x3e964000, 0x3f800000]), (15, [0x3ec50000, 0x3edf6000, 0x3edfe000, 0x3f800000]), (10, [0x3e5c2000, 0x3e63a000, 0x3e7ec000, 0x3f800000])]),
    // cem 9
    (222, 0x9924818df8fcf0a4_d64c4754be6d23ae, [(0, [0x3d8cc000, 0x3e558000, 0x3d6c2000, 0x3f800000]), (15, [0x3db8a000, 0x3e50c000, 0x3d988000, 0x3f800000]), (10, [0x3d686000, 0x3e582000, 0x3d458000, 0x3f800000])]),
    // cem 11
    (222, 0x4a7d5d49e3418d98_128e31896d0763bf, [(0, [0x41ac4000, 0x4092a000, 0x413da000, 0x3f800000]), (15, [0x41398000, 0x40116000, 0x40b7c000, 0x3f800000]), (10, [0x4183c000, 0x405c6000, 0x410ae000, 0x3f800000])]),
    // cem 12 dual
    (222, 0x2e475dcc936ade0c_1b5086d84d21878f, [(0, [0x3f2ea000, 0x3e970000, 0x3e4b0000, 0x3f18e000]), (15, [0x3f104000, 0x3ea86000, 0x3e8c6000, 0x3ebf2000]), (10, [0x3f23c000, 0x3e9c4000, 0x3e62c000, 0x3f076000])]),
    // cem 9 dual
    (222, 0x0e3068382e8183bc_c58f700f2baf2411, [(0, [0x3f6be000, 0x3f038000, 0x3ec6c000, 0x3f800000]), (15, [0x3f6be000, 0x3f038000, 0x3ec6c000, 0x3f800000]), (10, [0x3f728000, 0x3f15e000, 0x3eb30000, 0x3f800000])]),
    // 4 partitions hdr
    (222, 0x102c2d6c68242279_043a9229a65d7b9e, [(0, [0x4193a000, 0x4193a000, 0x4193a000, 0x3f800000]), (15, [0x3c3ac000, 0x3c3ac000, 0x3c3ac000, 0x3f800000]), (10, [0x41944000, 0x41944000, 0x41944000, 0x3f800000])]),
    // void-ldr
    (223, 0x99163ee11717288b_fffffffffffffdfc, [(0, [0x3e222000, 0x3db8a000, 0x3e7b8000, 0x3f190000]), (19, [0x3e222000, 0x3db8a000, 0x3e7b8000, 0x3f190000]), (12, [0x3e222000, 0x3db8a000, 0x3e7b8000, 0x3f190000])]),
    // 4 partitions ldr
    (223, 0xa05d7854a6d2cd40_53e0ae2a6324da11, [(0, [0x3f13e000, 0x3f13e000, 0x3f13e000, 0x3ee2e000]), (19, [0x3e60e000, 0x3e60e000, 0x3e60e000, 0x3e202000]), (12, [0x3f2d8000, 0x3ee2e000, 0x3f0b0000, 0x3f800000])]),
    // cem 7
    (223, 0xad5838c5b99fe182_3a79aa683470e251, [(0, [0x40db6000, 0x40a6c000, 0x40b2c000, 0x3f800000]), (19, [0x40824000, 0x40472000, 0x40532000, 0x3f800000]), (12, [0x40aae000, 0x407ea000, 0x40882000, 0x3f800000])]),
    // cem 2 dual
    (223, 0x98b64edeb75e657c_382624cbf8704411, [(0, [0x41980000, 0x477fe000, 0x41980000, 0x3f800000]), (19, [0x477fe000, 0x3b800000, 0x477fe000, 0x3f800000]), (12, [0x42be0000, 0x3c3c0000, 0x42be0000, 0x3f800000])]),
    // cem 6
    (223, 0xf4c31f108b8b384b_b51b64a4bc70c231, [(0, [0x3e4da000, 0x3ead4000, 0x3e974000, 0x3f800000]), (19, [0x3e60e000, 0x3ebca000, 0x3ea4a000, 0x3f800000]), (12, [0x3e3dc000, 0x3ea08000, 0x3e8c4000, 0x3f800000])]),
    // cem 8 dual
    (223, 0x8c2fdea2fad7b493_79101b36c41b04b3, [(0, [0x3f09c000, 0x3ef9e000, 0x3f5bc000, 0x3f800000]), (19, [0x3f342000, 0x3ee50000, 0x3f5bc000, 0x3f800000]), (12, [0x3f0d2000, 0x3ef82000, 0x3f5bc000, 0x3f800000])]),
    // cem 15
    (223, 0xc57564e8bd5c3d24_1e7cc77d7789e3ae, [(0, [0x3a484000, 0x3956c000, 0x388ea000, 0x3889c000]), (19, [0x3aef2000, 0x39fa6000, 0x39570000, 0x38818000]), (12, [0x3a5fe000, 0x39712000, 0x38a74000, 0x3888a000])]),
    // cem 14 dual
    (223, 0x5100ce41748ceab3_ddec4ef22bdfc7af, [(0, [0x3a726000, 0x38ee8000, 0x392b4000, 0x3f48e000]), (19, [0x3b848000, 0x390ec000, 0x3a510000, 0x3f090000]), (12, [0x3bb90000, 0x3927e000, 0x3a926000, 0x3ef66000])]),
    // cem 12
    (223, 0x114e4deeef29fd64_b58c62bb2edb83ae, [(0, [0x3f0a8000, 0x3edde000, 0x3f15a000, 0x3f554000]), (19, [0x3f076000, 0x3ecfa000, 0x3f0b6000, 0x3f5c6000]), (12, [0x3f002000, 0x3eae4000, 0x3ee78000, 0x3f6d0000])]),
    // void-ldr
    (224, 0x55675aef26be4740_fffffffffffffdfc, [(0, [0x3e8e8000, 0x3e1ae000, 0x3eb5c000, 0x3eaac000]), (24, [0x3e8e8000, 0x3e1ae000, 0x3eb5c000, 0x3eaac000]), (12, [0x3e8e8000, 0x3e1ae000, 0x3eb5c000, 0x3eaac000])]),
    // cem 5
    (224, 0xe7c0d9f83dba9004_304791e932d8a2f1, [(0, [0x3e13e000, 0x3e13e000, 0x3e13e000, 0x3d0a8000]), (24, [0x3e13e000, 0x3e13e000, 0x3e13e000, 0x3d0a8000]), (12, [0x3df0e000, 0x3df0e000, 0x3df0e000, 0x00000000])]),
    // cem 9
    (225, 0xa9a4d847fc8251a1_ebccd423d79520d1, [(0, [0x3f686000, 0x3ef2e000, 0x3f70e000, 0x3f800000]), (29, [0x3f6ce000, 0x3efae000, 0x3f73e000, 0x3f800000]), (15, [0x3f678000, 0x3ef16000, 0x3f704000, 0x3f800000])]),
    // void-hdr-neg
    (226, 0xb3fd6c0928078ee5_fffffffffffffffc, [(0, [0xb9dca000, 0x3d00e000, 0x45812000, 0xbe7fa000]), (35, [0xb9dca000, 0x3d00e000, 0x45812000, 0xbe7fa000]), (21, [0xb9dca000, 0x3d00e000, 0x45812000, 0xbe7fa000])]),
    // 2 partitions mixed dual
    (228, 0x6f6a96701f681ecc_f4e42d79f9bd6f0d, [(0, [0x3f0be000, 0x3de70000, 0x3ece8000, 0x3f440000]), (39, [0x3c57e000, 0x00000000, 0x370d0000, 0x3f800000]), (20, [0x3c5d0000, 0x00000000, 0x37720000, 0x3f800000])]),
    // cem 8
    (229, 0xbc8fd56adf71d5d6_a6acc21687f70035, [(0, [0x3ef3e000, 0x3e8a8000, 0x3ea9a000, 0x3f800000]), (47, [0x3e968000, 0x3eb4a000, 0x3ea6a000, 0x3f800000]), (28, [0x3f28a000, 0x3e40c000, 0x3eaca000, 0x3f800000])]),
    // 2 partitions ldr
    (230, 0xb79efd64dce8e4af_59cb4f0bf73869a2, [(0, [0x3e9a6000, 0x3f0b0000, 0x3eaa4000, 0x3ecda000]), (63, [0x3f068000, 0x3f72e000, 0x3f148000, 0x3f2fa000]), (36, [0x3f23c000, 0x3e7ae000, 0x3e4a2000, 0x3f4fe000])]),
    // void-hdr-neg
    (231, 0x24b430408393a98a_fffffffffffffffc, [(0, [0xbd314000, 0xb864c000, 0x3e080000, 0x3c968000]), (49, [0xbd314000, 0xb864c000, 0x3e080000, 0x3c968000]), (25, [0xbd314000, 0xb864c000, 0x3e080000, 0x3c968000])]),
    // cem 7 dual
    (232, 0xed0efeab2feb0356_9a58aa70c472e4b1, [(0, [0x4615c000, 0x4612c000, 0x45e84000, 0x3f800000]), (59, [0x4615c000, 0x46698000, 0x463a0000, 0x3f800000]), (35, [0x461a4000, 0x4650c000, 0x4624c000, 0x3f800000])]),
    // void-ldr
    (233, 0xed6b088216cd6192_fffffffffffffdfc, [(0, [0x3ec32000, 0x3db66000, 0x3d082000, 0x3f6d6000]), (79, [0x3ec32000, 0x3db66000, 0x3d082000, 0x3f6d6000]), (45, [0x3ec32000, 0x3db66000, 0x3d082000, 0x3f6d6000])]),
    // void-hdr-neg
    (234, 0xa9446498a76db9f6_fffffffffffffffc, [(0, [0xbf3ec000, 0xbceda000, 0x44930000, 0xbd288000]), (99, [0xbf3ec000, 0xbceda000, 0x44930000, 0xbd288000]), (55, [0xbf3ec000, 0xbceda000, 0x44930000, 0xbd288000])]),
    // cem 0
    (235, 0xcfca04716b3f698e_106689c5a6c40213, [(0, [0x3f294000, 0x3f294000, 0x3f294000, 0x3f800000]), (119, [0x3f420000, 0x3f420000, 0x3f420000, 0x3f800000]), (66, [0x3f322000, 0x3f322000, 0x3f322000, 0x3f800000])]),
    // 3 partitions mixed
    (236, 0xa601fa811a3ba287_84c670cb3942937f, [(0, [0x3e200000, 0x3e200000, 0x3e200000, 0x3f014000]), (143, [0x3f49c000, 0x3f062000, 0x00000000, 0x3e762000]), (78, [0x45aa2000, 0x45920000, 0x45a22000, 0x3f800000])]),
];

#[test]
fn hdr_blocks_read_metal_s_bits() {
    for (raw, block, texels) in METAL_READS {
        let pf = PixelFormat::try_from(raw).expect("an ASTC HDR format");
        let d = AstcDecode::of(pf).expect("a decode");
        let values = d.block_values(&block.to_le_bytes());
        for (at, metal) in texels {
            let ours = match &values {
                Some(values) => d.metal_value(values[at as usize]),
                None => d.metal_value([super::super::HDR_ERROR; 4]),
            };
            assert_eq!(
                ours.map(f32::to_bits),
                metal,
                "{pf:?} {block:032x} texel {at}"
            );
        }
    }
}

/// The two conversions of a 16-bit interpolation result into a half, at their edges: a
/// logarithmic one's exponent and three mantissa pieces, clamped to the largest finite half; a
/// linear one's `C / 65536` truncated, subnormal below 4, and 1 for 65535.
#[test]
fn the_hdr_profile_s_halves() {
    // 1.0 is `0x780` widened; the mantissa's pieces meet at 512 and 1536.
    assert_eq!(hdr::lns_to_half(0x7800), 0x3C00);
    assert_eq!(hdr::lns_to_half(0x7800 | 511), 0x3C00 | (3 * 511) >> 3);
    assert_eq!(
        hdr::lns_to_half(0x7800 | 512),
        0x3C00 | (4 * 512 - 512) >> 3
    );
    assert_eq!(
        hdr::lns_to_half(0x7800 | 1536),
        0x3C00 | (5 * 1536 - 2048) >> 3
    );
    assert_eq!(hdr::lns_to_half(0xFFF0), 0x7BFF);
    assert_eq!(hdr::lns_to_half(0), 0);
    assert_eq!(hdr::unorm16_to_half(0xFFFF), 0x3C00);
    assert_eq!(hdr::unorm16_to_half(0xFFFE), 0x3BFF);
    assert_eq!(hdr::unorm16_to_half(0x8000), 0x3800);
    assert_eq!(hdr::unorm16_to_half(4), 0x0400);
    assert_eq!(hdr::unorm16_to_half(3), 0x0300);
    assert_eq!(hdr::unorm16_to_half(0), 0);
    for h in [
        0x0001u16, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x8001, 0xBC00,
    ] {
        let f = hdr::half_value(h);
        assert_eq!(f64::from(f), half_reference(h), "{h:04x}");
    }
    assert_eq!(hdr::half_value(0x7E00).to_bits(), 0x7FC0_0000);
}

/// A half's value by its definition.
fn half_reference(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((h >> 10) & 0x1F), f64::from(h & 0x3FF));
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 => sign * f64::INFINITY,
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// An HDR format's texel forms: its compact one is halves, which it holds for `Unorm8`; `Float`
/// is `float`s.
#[test]
fn an_hdr_format_s_twin_holds_halves() {
    let d = AstcDecode::of(PixelFormat::ASTC6x5HDR).expect("a decode");
    assert!(d.hdr && !d.srgb && d.block_extent() == [6, 5]);
    for texels in [AstcTexels::Unorm8, AstcTexels::Half] {
        assert_eq!(d.form(texels), AstcTexels::Half);
        assert_eq!(d.texel_format(texels), Format::R16G16B16A16_SFLOAT);
        assert_eq!(d.texel_pixel_format(texels), PixelFormat::RGBA16Float);
        assert_eq!(d.texel_bytes(texels), 8);
    }
    assert_eq!(
        d.texel_format(AstcTexels::Float),
        Format::R32G32B32A32_SFLOAT
    );
    let ldr = AstcDecode::of(PixelFormat::ASTC6x5LDR).expect("a decode");
    assert_eq!(ldr.form(AstcTexels::Half), AstcTexels::Unorm8);
    // The halves are Metal's values: the `float`s widen them exactly.
    let blocks = hdr_seeded_blocks(225 * 7919, 64);
    let size = [8 * 6, 8 * 5, 1];
    let halves = d.decode(AstcTexels::Half, &blocks, size);
    let floats = d.decode(AstcTexels::Float, &blocks, size);
    for (half, float) in halves
        .as_chunks::<2>()
        .0
        .iter()
        .zip(floats.as_chunks::<4>().0)
    {
        let widened = hdr::half_value(u16::from_le_bytes(*half));
        assert_eq!(widened.to_bits(), u32::from_le_bytes(*float));
    }
}

#[test]
fn the_astc_hdr_decode_kernels_are_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    for pf in [PixelFormat::ASTC4x4HDR, PixelFormat::ASTC12x12HDR] {
        let decode = AstcDecode::of(pf).expect("a decode");
        for texels in [AstcTexels::Half, AstcTexels::Float] {
            for (_, p) in &profiles {
                assert_valid(typed_module(decode, texels).expect("module"), p);
            }
        }
    }
}

/// Seeded blocks of `decode` that cover the HDR profile's kinds of valid block, then `invalid`
/// error blocks: each valid one that adds a colour endpoint mode, a partition count, two planes,
/// or an HDR or LDR void-extent block, seen fewer than `each` times so far.
fn covering_hdr_blocks(decode: AstcDecode, seed: u64, each: usize, invalid: usize) -> Vec<u8> {
    let (mut modes, mut partitions, mut planes, mut void) = ([0; 16], [0; 5], 0, [0; 2]);
    let (mut good, mut bad) = (Vec::new(), Vec::new());
    for block in hdr_seeded_blocks(seed, 32768).as_chunks::<16>().0 {
        if decode.block_values(block).is_none() {
            if bad.len() < invalid * 16 {
                bad.extend_from_slice(block);
            }
            continue;
        }
        let bits = u128::from_le_bytes(*block);
        if bits & 0x1FF == 0x1FC {
            let hdr = ((bits >> 9) & 1) as usize;
            if void[hdr] < each {
                void[hdr] += 1;
                good.extend_from_slice(block);
            }
            continue;
        }
        let (kinds, dual) = modes_of(block);
        let adds = kinds.iter().any(|&m| modes[m as usize] < each)
            || partitions[kinds.len()] < each
            || (dual && planes < each);
        if adds {
            for &m in &kinds {
                modes[m as usize] += 1;
            }
            partitions[kinds.len()] += 1;
            planes += usize::from(dual);
            good.extend_from_slice(block);
        }
    }
    assert!(modes.iter().all(|&n| n >= each), "{decode:?}: {modes:?}");
    assert!(void.iter().all(|&n| n >= each), "{decode:?}: {void:?}");
    good.extend(bad);
    good
}

/// The GPU's decode is the CPU's, bit for bit, as halves and as `float`s, in regions of two layers
/// with partial blocks at their right and bottom edges: every texel of seeded 4x4 blocks that
/// cover every colour endpoint mode, partition count, two planes and HDR and LDR void-extent
/// blocks, and of error blocks (in both orders); and of a few such blocks of each other footprint.
#[test]
fn the_astc_hdr_kernel_is_the_decode_on_every_texel() {
    let both = [Order::Forward, Order::Reverse];
    let four = AstcDecode::of(PixelFormat::ASTC4x4HDR).expect("a decode");
    let (blocks, size) = two_layers(four, covering_hdr_blocks(four, 222, 2, 4), 8);
    assert_kernel_decodes(four, AstcTexels::Half, &blocks, size, &both);
    assert_kernel_decodes(four, AstcTexels::Float, &blocks, size, &both[..1]);
    for pf in hdr_formats()
        .into_iter()
        .filter(|&pf| pf != PixelFormat::ASTC4x4HDR)
    {
        let decode = AstcDecode::of(pf).expect("a decode");
        let mut blocks = covering_hdr_blocks(decode, u64::from(pf.as_raw()), 1, 1);
        blocks.drain(16 * 4..blocks.len() - 16);
        let (blocks, size) = two_layers(decode, blocks, 2);
        assert_kernel_decodes(decode, AstcTexels::Half, &blocks, size, &both[..1]);
    }
}
