//! The ASTC decode against blocks the specification's rules decide outright, and against Metal's
//! reads of seeded random blocks on an Apple M3 Max (macOS 27.0.1).

mod hdr;

use super::kernel::typed_module;
use super::{AstcDecode, AstcTexels, Illegal};
use crate::bind::block_decode::{BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

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

#[test]
fn the_astc_decode_kernels_are_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    for pf in [
        PixelFormat::ASTC4x4LDR,
        PixelFormat::ASTC12x12SRGB,
        PixelFormat::ASTC4x2LDR,
        PixelFormat::ASTC8x4SRGB,
    ] {
        let decode = AstcDecode::of(pf).expect("a decode");
        for texels in [AstcTexels::Unorm8, AstcTexels::Float] {
            for (_, p) in &profiles {
                assert_valid(typed_module(decode, texels).expect("module"), p);
            }
        }
    }
}

/// The kernel's texels of `blocks` of a region of `size` of `decode`, in `order`; the texel
/// buffer's words past the region's stay unwritten.
fn kernel_texels(
    decode: AstcDecode,
    texels: AstcTexels,
    blocks: &[u8],
    size: [u32; 3],
    order: Order,
) -> Vec<u8> {
    let m = typed_module(decode, texels).expect("module");
    let [w, h, d] = size;
    let length = (w * h * d * decode.texel_bytes(texels)) as usize;
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(SHAPE),
                address: None,
                bytes: BlockDecodeModule::shape(size).to_vec(),
            },
            Buffer {
                slot: Some(BLOCKS),
                address: None,
                bytes: blocks.to_vec(),
            },
            Buffer {
                slot: Some(TEXELS),
                address: None,
                bytes: vec![0xcd; length + 64],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let groups = |n: u32| n.div_ceil(BlockDecodeModule::LOCAL_SIZE);
    let cfg = Config {
        order,
        workgroups: [groups(w) + 1, groups(h), d + 1],
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).expect("run").buffers;
    let written = out.swap_remove(2);
    assert!(written[length..].iter().all(|&b| b == 0xcd), "{decode:?}");
    written[..length].to_vec()
}

/// The colour endpoint modes of the partitions of `block`, a valid block that is no void-extent
/// block, and whether it has two planes.
fn modes_of(block: &[u8; 16]) -> (Vec<u32>, bool) {
    let bits = u128::from_le_bytes(*block);
    let mode = super::block_mode((bits & 0x7FF) as u32).expect("a valid mode");
    let count = ((bits >> 11) & 3) as u32 + 1;
    if count == 1 {
        return (vec![((bits >> 13) & 15) as u32], mode.dual);
    }
    let low = ((bits >> 23) & 63) as u32;
    if low & 3 == 0 {
        return (vec![low >> 2; count as usize], mode.dual);
    }
    // The configuration's high bits lie just below the weights.
    let [gw, gh] = mode.grid;
    let weight_bits = mode.range.ise_bits(gw * gh * (1 + u32::from(mode.dual)));
    let extra = 3 * count - 4;
    let config = low | (super::field(bits, 128 - weight_bits - extra, extra) as u32) << 6;
    let modes = (0..count)
        .map(|i| {
            let class = (config & 3) - 1 + ((config >> (2 + i)) & 1);
            class << 2 | (config >> (2 + count + 2 * i)) & 3
        })
        .collect();
    (modes, mode.dual)
}

/// Seeded blocks of `decode` that cover its valid blocks' kinds, then `invalid` error blocks: of
/// the seeded blocks of `seed`, each valid one that adds a colour endpoint mode, a partition
/// count, two planes or a void-extent block seen fewer than `each` times so far.
fn covering_blocks(decode: AstcDecode, seed: u64, each: usize, invalid: usize) -> Vec<u8> {
    let (mut modes, mut partitions, mut planes, mut void) = ([0; 16], [0; 5], 0, 0);
    let (mut good, mut bad) = (Vec::new(), Vec::new());
    for block in seeded_blocks(seed, 8192).as_chunks::<16>().0 {
        if decode.block_values(block).is_none() {
            if bad.len() < invalid * 16 {
                bad.extend_from_slice(block);
            }
            continue;
        }
        let bits = u128::from_le_bytes(*block);
        if bits & 0x1FF == 0x1FC {
            if void < each {
                void += 1;
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
    good.extend(bad);
    good
}

/// The GPU's decode equals the CPU's, in `orders`: the first texel of `size` where they differ,
/// named.
fn assert_kernel_decodes(
    decode: AstcDecode,
    texels: AstcTexels,
    blocks: &[u8],
    size: [u32; 3],
    orders: &[Order],
) {
    let want = decode.decode(texels, blocks, size);
    for &order in orders {
        let got = kernel_texels(decode, texels, blocks, size, order);
        let bytes = decode.texel_bytes(texels) as usize;
        let differs = (got.chunks(bytes).zip(want.chunks(bytes))).position(|(g, w)| g != w);
        if let Some(t) = differs {
            let [w, h, _] = size.map(|axis| axis as usize);
            let (x, y, z) = (t % w, t / w % h, t / (w * h));
            let [bw, bh] = decode.block_extent().map(|axis| axis as usize);
            let b = (z * h.div_ceil(bh) + y / bh) * w.div_ceil(bw) + x / bw;
            panic!(
                "{decode:?} {texels:?} {order:?}: texel ({x}, {y}, {z}) of block {b} \
                 {:032x}: {:?} against the CPU's {:?}",
                u128::from_le_bytes(blocks[16 * b..16 * b + 16].try_into().expect("a block")),
                &got[t * bytes..(t + 1) * bytes],
                &want[t * bytes..(t + 1) * bytes],
            );
        }
    }
}

/// `blocks` padded with zero blocks (error blocks) to two layers of rows of `across`, and the
/// region of `decode` they fill, its last column and row of blocks partial.
fn two_layers(decode: AstcDecode, mut blocks: Vec<u8>, across: usize) -> (Vec<u8>, [u32; 3]) {
    let rows = (blocks.len() / 16).div_ceil(2 * across);
    blocks.resize(2 * rows * across * 16, 0);
    let size = [
        across as u32 * decode.width - 1,
        rows as u32 * decode.height - 2,
        2,
    ];
    (blocks, size)
}

/// The GPU's decode is the CPU's, bit for bit, linear and sRGB, in regions of two layers with
/// partial blocks at their right and bottom edges: every texel of seeded 4x4 blocks that cover
/// every colour endpoint mode (an HDR one's partition reading magenta), partition count, two
/// planes and void-extent blocks ([`covering_blocks`]) and of error blocks, as `float`s and as
/// 8-bit codes (in both orders); of a dozen such blocks of the 5x5 and 6x6 footprints (fewer and
/// more than 31 texels: the partition function's two forms), and of three of each other
/// footprint, as `float`s.
#[test]
fn the_astc_kernel_is_the_decode_on_every_texel() {
    let both = [Order::Forward, Order::Reverse];
    let four = AstcDecode::of(PixelFormat::ASTC4x4LDR).expect("a decode");
    let (blocks, size) = two_layers(four, covering_blocks(four, 204, 2, 4), 8);
    for decode in [four, AstcDecode { srgb: true, ..four }] {
        assert_kernel_decodes(decode, AstcTexels::Float, &blocks, size, &both[..1]);
        assert_kernel_decodes(decode, AstcTexels::Unorm8, &blocks, size, &both);
    }
    for raw in 205..=218u32 {
        let Some(linear) = PixelFormat::try_from(raw).ok().and_then(AstcDecode::of) else {
            continue;
        };
        let valid = match linear.block_extent() {
            [5, 5] | [6, 6] => 12,
            _ => 3,
        };
        let mut blocks = covering_blocks(linear, u64::from(raw), 1, 0);
        blocks.truncate(16 * valid);
        blocks.extend_from_slice(&covering_blocks(linear, u64::from(raw), 0, 1));
        let (blocks, size) = two_layers(linear, blocks, 2);
        let srgb = AstcDecode {
            srgb: true,
            ..linear
        };
        for decode in [linear, srgb] {
            assert_kernel_decodes(decode, AstcTexels::Float, &blocks, size, &both[..1]);
        }
    }
}

/// The blocks the comparison above decodes cover every colour endpoint mode, partition count, two
/// planes and void-extent blocks, on the 4x4 footprint (whose small grids leave the fewest valid
/// blocks) and the 12x12.
#[test]
fn the_compared_blocks_cover_every_mode() {
    for pf in [PixelFormat::ASTC4x4LDR, PixelFormat::ASTC12x12LDR] {
        let decode = AstcDecode::of(pf).expect("a decode");
        let blocks = covering_blocks(decode, u64::from(pf.as_raw()), 2, 0);
        let (mut modes, mut partitions, mut planes, mut void) = ([0; 16], [0; 5], 0, 0);
        for block in blocks.as_chunks::<16>().0 {
            if u128::from_le_bytes(*block) & 0x1FF == 0x1FC {
                void += 1;
                continue;
            }
            let (kinds, dual) = modes_of(block);
            for m in &kinds {
                modes[*m as usize] += 1;
            }
            partitions[kinds.len()] += 1;
            planes += usize::from(dual);
        }
        assert!(void > 0 && planes > 0, "{pf:?}");
        assert!(
            partitions[1..].iter().all(|&n| n > 0),
            "{pf:?}: {partitions:?}"
        );
        assert!(modes.iter().all(|&n| n > 0), "{pf:?}: {modes:?}");
    }
}
