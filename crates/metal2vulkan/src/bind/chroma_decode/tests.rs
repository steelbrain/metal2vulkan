use super::{
    convert_typed_module, sample_typed_module, typed_module, ChromaConvertModule, ChromaDecode,
    ChromaSampleModule, IMAGE,
};
use crate::bind::block_decode::{BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::spirv::interp::{run, run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// `n` bytes of the xorshift64* stream from `seed`.
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

/// Each texel reads its own green and its red and blue interpolated with the pair beside it,
/// `(3 c + n + 2) >> 2`, alpha 255, in either order; a row's end reads its own pair.
#[test]
fn a_texel_interpolates_red_and_blue_with_the_pair_beside_it() {
    // `GBGR422` (G0 B G1 R): pair 0 red 0x40 blue 0x20, pair 1 red 0x80 blue 0x61.
    let pairs = [0x10, 0x20, 0x30, 0x40, 0x50, 0x61, 0x70, 0x80];
    let red = |c: u32, n: u32| ((3 * c + n + 2) >> 2) as u8;
    assert_eq!(
        ChromaDecode::Gbgr.decode(&pairs, [4, 1, 1]),
        [
            [0x40, 0x10, 0x20, 0xff],
            [red(0x40, 0x80), 0x30, red(0x20, 0x61), 0xff],
            [red(0x80, 0x40), 0x50, red(0x61, 0x20), 0xff],
            [0x80, 0x70, 0x61, 0xff],
        ]
        .concat()
    );
    // `BGRG422` (B G0 R G1): pair 0 red 0x30 blue 0x10, pair 1 red 0x70 blue 0x50.
    assert_eq!(
        ChromaDecode::Bgrg.decode(&pairs, [4, 1, 1]),
        [
            [0x30, 0x20, 0x10, 0xff],
            [red(0x30, 0x70), 0x40, red(0x10, 0x50), 0xff],
            [red(0x70, 0x30), 0x61, red(0x50, 0x10), 0xff],
            [0x70, 0x80, 0x50, 0xff],
        ]
        .concat()
    );
    // One pair a row reads itself on both sides; rows read their own pairs alone.
    assert_eq!(
        ChromaDecode::Gbgr.decode(&pairs, [2, 2, 1]),
        [
            [0x40, 0x10, 0x20, 0xff],
            [0x40, 0x30, 0x20, 0xff],
            [0x80, 0x50, 0x61, 0xff],
            [0x80, 0x70, 0x61, 0xff],
        ]
        .concat()
    );
}

/// Metal's reads of our M3 probe's textures (16 x 4 texels of both formats over the same bytes,
/// each the top byte of one xorshift64* step from `0x422422422`): the FNV-1a digest of the codes
/// of every texel's `float` read (each `c / 255` exactly, row by row) is this decode's.
#[test]
fn the_probe_s_textures_decode_as_metal_reads_them() {
    let mut x: u64 = 0x4_2242_2422;
    let bytes: Vec<u8> = (0..16 * 4 * 2)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect();
    let fnv = |codes: &[u8]| {
        codes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &c| {
            (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
        })
    };
    for (decode, digest) in [
        (ChromaDecode::Gbgr, 0x9f08_e255_b1af_8809),
        (ChromaDecode::Bgrg, 0xb1c8_3abd_161a_5a90),
    ] {
        let codes = decode.decode(&bytes, [16, 4, 1]);
        assert_eq!(fnv(&codes), digest, "{decode:?}: {:02x?}", &codes[..16]);
    }
}

#[test]
fn the_chroma_decode_kernels_are_valid_on_every_committed_profile() {
    let profiles = committed_profiles().expect("profiles");
    for decode in ChromaDecode::ALL {
        for (_, p) in &profiles {
            assert_valid(typed_module(decode).expect("module"), p);
        }
    }
    // The device's own conversion's sample kernel, one for both formats.
    for (_, p) in &profiles {
        assert_valid(sample_typed_module().expect("sample module"), p);
    }
    // The in-place kernel of either order.
    for decode in ChromaDecode::ALL {
        for (_, p) in &profiles {
            assert_valid(convert_typed_module(decode).expect("in-place module"), p);
        }
    }
}

/// The sample module's shape: the band's width and rows, its first row, and the image's extent,
/// each a little-endian word; Vulkan's formats of the pairs, in either order.
#[test]
fn the_chroma_sample_shape_and_formats() {
    let shape = ChromaSampleModule::shape([16, 3], 5, [16, 12]);
    let words: Vec<u32> = (shape.as_chunks::<4>().0.iter())
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert_eq!(words, [16, 3, 5, 16, 12]);
    use crate::vk::Format;
    assert_eq!(ChromaDecode::Gbgr.format(), Format::G8B8G8R8_422_UNORM);
    assert_eq!(ChromaDecode::Bgrg.format(), Format::B8G8R8G8_422_UNORM);
}

/// The kernel's texels of a region `size` of `decode`'s `pairs`; the texel buffer's bytes past the
/// region's stay unwritten.
fn kernel_texels(decode: ChromaDecode, pairs: &[u8], size: [u32; 3]) -> Vec<u8> {
    let m = typed_module(decode).expect("module");
    let [w, h, d] = size;
    let length = (w * h * d * ChromaDecode::TEXEL_BYTES) as usize;
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
                bytes: pairs.to_vec(),
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
        order: Order::Forward,
        workgroups: [groups(w) + 1, groups(h), d + 1],
        ..Config::default()
    };
    let mut out = run(&m, &cfg, res).expect("run").buffers;
    let written = out.swap_remove(2);
    assert!(written[length..].iter().all(|&b| b == 0xcd), "{decode:?}");
    written[..length].to_vec()
}

/// The kernel writes the CPU's decode on every texel of both orders: random pairs, two images, odd
/// widths and heights, rows of one pair and of one texel.
#[test]
fn the_chroma_kernel_is_the_decode_on_every_texel() {
    for (k, decode) in ChromaDecode::ALL.into_iter().enumerate() {
        for size in [[16u32, 12, 2], [13, 9, 1], [7, 5, 2], [2, 1, 1], [1, 1, 1]] {
            let [w, h, d] = size;
            let n = (w.div_ceil(2) * h * d * ChromaDecode::BLOCK_BYTES) as usize;
            let pairs = stream(0x422_0000 + k as u64, n);
            let cpu = decode.decode(&pairs, size);
            let gpu = kernel_texels(decode, &pairs, size);
            assert_eq!(gpu, cpu, "{decode:?} {size:?}");
        }
    }
}

/// The in-place kernel's texels of a texture of `size` whose rows of pairs start at byte `first`
/// of `bytes`, `row` bytes apart, both invocation orders alike; the bytes are only read.
fn in_place_texels(
    decode: ChromaDecode,
    bytes: &[u8],
    size: [u32; 2],
    (first, row): (u32, u32),
) -> Vec<u8> {
    let m = convert_typed_module(decode).expect("module");
    let [w, h] = size;
    let groups = [w, h].map(|n| n.div_ceil(ChromaConvertModule::LOCAL_SIZE));
    let mut runs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let part = |slot, bytes: Vec<u8>| Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        };
        let shape = ChromaConvertModule::shape(size, first, row);
        let res = Resources {
            buffers: vec![part(SHAPE, shape.to_vec()), part(BLOCKS, bytes.to_vec())],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [groups[0] + 1, groups[1], 1],
            ..Config::default()
        };
        let image = Image::new(IMAGE, [w, h, 1], 1, 1, 1);
        let (outcome, images) = run_with_images(&m, &cfg, res, vec![image]).expect("run");
        assert_eq!(
            outcome.buffers[1], bytes,
            "{decode:?}: the bytes are only read"
        );
        let texels: Vec<u8> = (images[0].texels.iter())
            .flat_map(|t| t[0].to_le_bytes())
            .collect();
        runs.push(texels);
    }
    assert_eq!(runs[0], runs[1], "{decode:?}: order independent");
    runs.swap_remove(0)
}

/// The in-place kernel writes the CPU's decode of the rows as they lie: pairs at any byte (every
/// lane of a word), rows padded past their pairs, odd widths, a row of one pair and of one texel,
/// in both orders; the bytes past the last row's pairs are not the image's.
#[test]
fn the_in_place_kernel_is_the_decode_of_the_rows_where_they_lie() {
    for (k, decode) in ChromaDecode::ALL.into_iter().enumerate() {
        for (size, first, pad) in [
            ([16u32, 12], 0u32, 0u32),
            ([16, 12], 3, 16),
            ([13, 9], 1, 6),
            ([7, 5], 2, 1),
            ([2, 1], 17, 0),
            ([1, 3], 6, 9),
        ] {
            let [w, h] = size;
            let pairs_row = w.div_ceil(2) * ChromaDecode::BLOCK_BYTES;
            let row = pairs_row + pad;
            let length = (first + row * h + 8) as usize;
            let bytes = stream(0x4_2200 + k as u64 + u64::from(first), length);
            let tight: Vec<u8> = (0..h)
                .flat_map(|y| {
                    let at = (first + y * row) as usize;
                    bytes[at..at + pairs_row as usize].to_vec()
                })
                .collect();
            let cpu = decode.decode(&tight, [w, h, 1]);
            let gpu = in_place_texels(decode, &bytes, size, (first, row));
            assert_eq!(gpu, cpu, "{decode:?} {size:?} first {first} row {row}");
        }
    }
}

/// The in-place shape: the size, the first pair's byte and the row's bytes, each a little-endian
/// word.
#[test]
fn the_in_place_shape() {
    let shape = ChromaConvertModule::shape([32, 4], 7, 64);
    let words: Vec<u32> = (shape.as_chunks::<4>().0.iter())
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert_eq!(words, [32, 4, 7, 64]);
}
