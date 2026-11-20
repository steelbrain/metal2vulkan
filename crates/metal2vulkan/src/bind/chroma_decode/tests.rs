use super::ChromaDecode;

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
