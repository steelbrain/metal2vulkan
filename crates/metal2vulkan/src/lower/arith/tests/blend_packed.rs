//! A packed 16-bit word attachment's blend on the interpreter: the word read as Metal's blend
//! reads it (`stage_io::word_color::decode`), blended (`extended_range::blend::blend_packed16`),
//! encoded as Metal's word (`stage_io::word_color::encode`), and the fields of the channels the
//! write mask keeps taken from the word read, against words Metal stored. The words are the M3
//! Max's under macOS 27.0.1 (plain Metal, our own MSL, a target loaded with the destination's
//! words): nine states of the packed 16-bit blend sweep (`Blend::packed16`) of every form the
//! planner names (a fused product, a contracted pair, a common factor, a lerp of the blend color,
//! minimum and maximum, write masks of one color channel, of red and alpha, and of all but red),
//! each into every packed 16-bit format from `half4` and `float4` outputs under fast and safe math,
//! over six pixels of the sweep: one with an infinite lane, one past both ends of `[0, 1]`, four
//! inside it.

use super::*;
use crate::lower::arith::extended_range::blend::{blend_packed16, Blend, BlendMath};
use crate::metal::format::WordAttachment;
use crate::metal::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat as P};
use crate::pipeline::lower_inputs::BlendEquation;

/// The pixels: the output's four lanes (as `f32` bits) and the destination's word.
const PIXELS: [([u32; 4], u16); 6] = [
    ([0x3f2dcb14, 0x3d46318c, 0x3df3cf3d, 0xff800000], 0x4955),
    ([0x3f2cee30, 0xbe679b5a, 0x3eefbefc, 0x3ff7099d], 0xcae6),
    ([0x3f1b0000, 0x3df42000, 0x3f54c000, 0x3f362000], 0x68b4),
    ([0x3eb7870d, 0x3ebb68d0, 0x3ebcc000, 0x3f2b2000], 0x06ad),
    ([0x3e082000, 0x3f665d72, 0x3f5fa000, 0x3f7aa000], 0x844b),
    ([0x3d088889, 0x3f154000, 0x3e972000, 0x3f02c000], 0x3165),
];

/// The states: source and destination factors and operation of color, then alpha (Metal's
/// enumerations), the write mask, and the blend color (as `f32` bits).
const STATES: [([u32; 7], [u32; 4]); 9] = [
    (
        [1, 5, 0, 1, 5, 0, 15],
        [0x3e99999a, 0x3f19999a, 0x3f666666, 0x3ee66666],
    ),
    (
        [4, 5, 0, 1, 5, 0, 15],
        [0x3e99999a, 0x3f19999a, 0x3f666666, 0x3ee66666],
    ),
    (
        [14, 14, 0, 14, 1, 3, 4],
        [0x3f76ceb7, 0x3e9e8f0f, 0x3ec3fe9a, 0x3da46ba8],
    ),
    (
        [11, 3, 0, 7, 14, 4, 15],
        [0x3ee6c29e, 0x3f92f0c0, 0x3f3abf47, 0x3e9e5b86],
    ),
    (
        [5, 6, 2, 14, 7, 1, 15],
        [0x3ec91e09, 0x3f2e55d9, 0xbe263f9d, 0x3f9d77bb],
    ),
    (
        [11, 2, 2, 11, 13, 1, 15],
        [0x3e9b4ea9, 0x3f017c31, 0x3e62035b, 0x3df4d713],
    ),
    (
        [1, 1, 4, 1, 1, 4, 15],
        [0x3e99999a, 0x3f19999a, 0x3f666666, 0x3ee66666],
    ),
    (
        [13, 12, 4, 14, 13, 0, 9],
        [0x3e9b462a, 0xbdd04230, 0x3f403c27, 0x3eb75716],
    ),
    (
        [11, 11, 2, 9, 2, 1, 7],
        [0x3f10cf42, 0x3e591def, 0xbe1e5284, 0x3f7b28e6],
    ),
];

/// Metal's words, one row each: the state, the format, the output (`half4` or `float4`), the math
/// (fast or safe), and the word each pixel stored, in hex.
const WORDS: &str = "\
0 B5G6R5 half fast f1b9 a80f b93f 5d30 273b 254c
0 B5G6R5 half safe f1b9 a80f b93f 5d30 273b 254c
0 B5G6R5 float fast f1b9 a80f b93f 5d30 273b 254c
0 B5G6R5 float safe f1b9 a80f b93f 5d30 273b 254c
0 A1BGR5 half fast f1dd a81f b93f 5d27 2737 2565
0 A1BGR5 half safe f1dd a81f b93f 5d27 2737 2565
0 A1BGR5 float fast f1dd a81f b93f 5d27 2737 2565
0 A1BGR5 float safe f1dd a81f b93f 5d27 2737 2565
0 ABGR4 half fast ea75 a07f b4fc 579e 2edf 297a
0 ABGR4 half safe ea75 a07f b4fc 579e 2edf 297a
0 ABGR4 float fast ea75 a07f b4fc 579e 2edf 297a
0 ABGR4 float safe ea75 a07f b4fc 579e 2edf 297a
0 BGR5A1 half fast 7d79 d40f e8bf ae50 939b 9eec
0 BGR5A1 half safe 7d79 d40f e8bf ae50 939b 9eec
0 BGR5A1 float fast 7d79 d40f e8bf ae50 939b 9eec
0 BGR5A1 float safe 7d79 d40f e8bf ae50 939b 9eec
1 B5G6R5 half fast 4955 a80f 88f8 3c2c 271b 1b07
1 B5G6R5 half safe 4955 a80f 88f8 3c2c 271b 1b07
1 B5G6R5 float fast 4955 a80f 88f8 3c2c 271b 1b07
1 B5G6R5 float safe 4955 a80f 88f8 3c2c 271b 1b07
1 A1BGR5 half fast 4955 a81f 88f5 3c1f 2737 1b1b
1 A1BGR5 half safe 4955 a81f 88f5 3c1f 2737 1b1b
1 A1BGR5 float fast 4955 a81f 88f5 3c1f 2737 1b1b
1 A1BGR5 float safe 4955 a81f 88f5 3c1f 2737 1b1b
1 ABGR4 half fast 4955 a07f 84cc 467e 2ddf 255a
1 ABGR4 half safe 4955 a07f 84cc 467e 2ddf 255a
1 ABGR4 float fast 4955 a07f 84cc 467e 2ddf 255a
1 ABGR4 float safe 4955 a07f 84cc 467e 2ddf 255a
1 BGR5A1 half fast 4955 d40f d498 a1ec 937b 99e7
1 BGR5A1 half safe 4955 d40f d498 a1ec 937b 99e7
1 BGR5A1 float fast 4955 d40f d498 a1ec 937b 99e7
1 BGR5A1 float safe 4955 d40f d498 a1ec 937b 99e7
2 B5G6R5 half fast 4995 caa6 6994 07ed 87eb 3585
2 B5G6R5 half safe 4995 caa6 6994 07ed 87eb 3585
2 B5G6R5 float fast 4995 caa6 6994 07ed 87eb 3585
2 B5G6R5 float safe 4995 caa6 6994 07ed 87eb 3585
2 A1BGR5 half fast 4995 caa6 6974 07ed 87cb 3565
2 A1BGR5 half safe 4995 caa6 6974 07ed 87cb 3565
2 A1BGR5 float fast 4995 caa6 6974 07ed 87cb 3565
2 A1BGR5 float safe 4995 caa6 6974 07ed 87cb 3565
2 ABGR4 half fast 4955 c9e6 69b4 0bad 8f4b 3965
2 ABGR4 half safe 4955 c9e6 69b4 0bad 8f4b 3965
2 ABGR4 float fast 4955 c9e6 69b4 0bad 8f4b 3965
2 ABGR4 float safe 4955 c9e6 69b4 0bad 8f4b 3965
2 BGR5A1 half fast 4975 caa6 6914 07cd 878b 3365
2 BGR5A1 half safe 4975 caa6 6914 07cd 878b 3365
2 BGR5A1 float fast 4975 caa6 6914 07cd 878b 3365
2 BGR5A1 float safe 4975 caa6 6914 07cd 878b 3365
3 B5G6R5 half fast 61b5 92ee 7196 2f31 8795 352a
3 B5G6R5 half safe 61b5 92ee 7196 2f31 8795 352a
3 B5G6R5 float fast 61b5 92ee 7196 2f31 8795 352a
3 B5G6R5 float safe 61b5 92ee 7196 2f31 8795 352a
3 A1BGR5 half fast 6197 92eb 716f 2f2d 87a9 3527
3 A1BGR5 half safe 6199 92eb 716f 2f2d 87a9 3527
3 A1BGR5 float fast 6197 92eb 716f 2f2d 87a9 3527
3 A1BGR5 float safe 6199 92eb 716f 2f2d 87a9 3527
3 ABGR4 half fast 6965 8adf 69bb 29ad 8eaf 3978
3 ABGR4 half safe 6965 8adf 69bb 29ad 8eaf 3978
3 ABGR4 float fast 6965 8adf 69bb 29ad 8eaf 3978
3 ABGR4 float safe 6965 8adf 69bb 29ad 8eaf 3978
3 BGR5A1 half fast 3d75 beee cd16 9b31 8f95 b2ea
3 BGR5A1 half safe 3d75 beee cd16 9b31 8f95 b2ea
3 BGR5A1 float fast 3d75 beee cd16 9b31 8f95 b2ea
3 BGR5A1 float safe 3d75 beee cd16 9b31 8f95 b2ea
4 B5G6R5 half fast 000b a101 0005 04a2 4223 0800
4 B5G6R5 half safe 000b a101 0005 04a2 4223 0800
4 B5G6R5 float fast 000b a101 0005 04a2 4223 0800
4 B5G6R5 float safe 000b a101 0005 04a2 4223 0800
4 A1BGR5 half fast 0000 a118 001c 0498 4240 080c
4 A1BGR5 half safe 0000 a118 001c 0498 4240 080c
4 A1BGR5 float fast 0000 a118 001c 0498 4240 080c
4 A1BGR5 float safe 0000 a118 001c 0498 4240 080c
4 ABGR4 half fast 0500 a7d0 0440 0150 4110 0000
4 ABGR4 half safe 0500 a7d0 0440 0150 4110 0000
4 ABGR4 float fast 0500 a7d0 0440 0150 4110 0000
4 ABGR4 float safe 0500 a7d0 0440 0150 4110 0000
4 BGR5A1 half fast 004b 2a21 4005 0142 0003 1000
4 BGR5A1 half safe 004b 2a21 4005 0142 0003 1000
4 BGR5A1 float fast 004b 2a21 4005 0142 0003 1000
4 BGR5A1 float safe 004b 2a21 4005 0142 0003 1000
5 B5G6R5 half fast 0002 5800 100b 0102 0844 0000
5 B5G6R5 half safe 0002 5800 100b 0102 0844 0000
5 B5G6R5 float fast 0002 5800 100b 0102 0844 0000
5 B5G6R5 float safe 0002 5800 100b 0102 0844 0000
5 A1BGR5 half fast 0000 580c 1020 010c 0840 0006
5 A1BGR5 half safe 0000 580c 1020 010c 0840 0006
5 A1BGR5 float fast 0000 580c 1020 010c 0840 0006
5 A1BGR5 float safe 0000 580c 1020 010c 0840 0006
5 ABGR4 half fast 0000 5051 1061 0020 0010 0010
5 ABGR4 half safe 0000 5051 1061 0020 0010 0010
5 ABGR4 float fast 0000 5051 1061 0020 0010 0010
5 ABGR4 float safe 0000 5051 1061 0020 0010 0010
5 BGR5A1 half fast 1802 1800 280b 0042 0004 0000
5 BGR5A1 half safe 1802 1800 280b 0042 0004 0000
5 BGR5A1 float fast 1802 1800 280b 0042 0004 0000
5 BGR5A1 float safe 1802 1800 280b 0042 0004 0000
6 B5G6R5 half fast a955 caef 991a 5ead 873b 34a9
6 B5G6R5 half safe a955 caef 991a 5ead 873b 34a9
6 B5G6R5 float fast a955 caef 991a 5ead 873b 34a9
6 B5G6R5 float safe a955 caef 991a 5ead 873b 34a9
6 A1BGR5 half fast a955 cae7 9935 5ead 8737 34a5
6 A1BGR5 half safe a955 cae7 9935 5ead 8737 34a5
6 A1BGR5 float fast a955 cae7 9935 5ead 8737 34a5
6 A1BGR5 float safe a955 cae7 9935 5ead 8737 34a5
6 ABGR4 half fast a955 caef 98cb 56ad 8ddf 3968
6 ABGR4 half safe a955 caef 98cb 56ad 8ddf 3968
6 ABGR4 float fast a955 caef 98cb 56ad 8ddf 3968
6 ABGR4 float safe a955 caef 98cb 56ad 8ddf 3968
6 BGR5A1 half fast 5555 d6ef e8ba aead 939b b249
6 BGR5A1 half safe 5555 d6ef e8ba aead 939b b249
6 BGR5A1 float fast 5555 d6ef e8ba aead 939b b249
6 BGR5A1 float safe 5555 d6ef e8ba aead 939b b249
7 B5G6R5 half fast a955 cae6 98b4 5ead 844b 3165
7 B5G6R5 half safe a955 cae6 98b4 5ead 844b 3165
7 B5G6R5 float fast a955 cae6 98b4 5ead 844b 3165
7 B5G6R5 float safe a955 cae6 98b4 5ead 844b 3165
7 A1BGR5 half fast a954 cae7 98b4 5ead 844b 3165
7 A1BGR5 half safe a954 cae7 98b4 5ead 844b 3165
7 A1BGR5 float fast a954 cae7 98b4 5ead 844b 3165
7 A1BGR5 float safe a954 cae7 98b4 5ead 844b 3165
7 ABGR4 half fast a952 caec 98b8 56ab 844d 3167
7 ABGR4 half safe a952 caec 98b8 56ab 844d 3167
7 ABGR4 float fast a952 caec 98b8 56ab 844d 3167
7 ABGR4 float safe a952 caec 98b8 56ab 844d 3167
7 BGR5A1 half fast 5555 d6e6 68b4 2ead 904b 3165
7 BGR5A1 half safe 5555 d6e6 68b4 2ead 904b 3165
7 BGR5A1 float fast 5555 d6e6 68b4 2ead 904b 3165
7 BGR5A1 float safe 5555 d6e6 68b4 2ead 904b 3165
8 B5G6R5 half fast 4820 c8a0 6800 00c0 8000 3000
8 B5G6R5 half safe 4820 c8a0 6800 00c0 8000 3000
8 B5G6R5 float fast 4820 c8a0 6800 00c0 8000 3000
8 B5G6R5 float safe 4820 c8a0 6800 00c0 8000 3000
8 A1BGR5 half fast 4840 c881 6801 00c0 8000 3000
8 A1BGR5 half safe 4840 c881 6801 00c0 8000 3000
8 A1BGR5 float fast 4840 c881 6801 00c0 8000 3000
8 A1BGR5 float safe 4840 c881 6801 00c0 8000 3000
8 ABGR4 half fast 4200 c203 6105 0000 8000 3003
8 ABGR4 half safe 4200 c203 6105 0000 8000 3003
8 ABGR4 float fast 4200 c203 6105 0000 8000 3003
8 ABGR4 float safe 4200 c203 6105 0000 8000 3003
8 BGR5A1 half fast 4840 48a0 e800 8440 0400 b000
8 BGR5A1 half safe 4840 48a0 e800 8440 0400 b000
8 BGR5A1 float fast 4840 48a0 e800 8440 0400 b000
8 BGR5A1 float safe 4840 48a0 e800 8440 0400 b000
";

#[test]
fn a_packed_word_attachment_blend_stores_metals_words() {
    let f4 = vec(float(32), 4);
    let u4 = vec(int(32), 4);
    let factor = |x: u32| BlendFactor::try_from(x).expect("a factor");
    let operation = |x: u32| BlendOperation::try_from(x).expect("an operation");
    for row in WORDS.lines() {
        let cols: Vec<&str> = row.split_whitespace().collect();
        let state: usize = cols[0].parse().expect("a state");
        let pf = match cols[1] {
            "B5G6R5" => P::B5G6R5Unorm,
            "A1BGR5" => P::A1BGR5Unorm,
            "ABGR4" => P::ABGR4Unorm,
            _ => P::BGR5A1Unorm,
        };
        let (half, fast) = (cols[2] == "half", cols[3] == "fast");
        let words: Vec<u16> = (cols[4..].iter())
            .map(|w| u16::from_str_radix(w, 16).expect("a word"))
            .collect();
        assert_eq!(words.len(), PIXELS.len(), "{row}");
        let (st, color) = STATES[state];
        let eq = BlendEquation {
            source_rgb: factor(st[0]),
            destination_rgb: factor(st[1]),
            rgb_operation: operation(st[2]),
            source_alpha: factor(st[3]),
            destination_alpha: factor(st[4]),
            alpha_operation: operation(st[5]),
        };
        let mask = ColorWriteMask::try_from(st[6]).expect("a mask");
        let word = WordAttachment::layout(pf).expect("a word layout");
        // A channel the format lacks counts as written, as the storage form plans it.
        let written = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .into_iter()
        .zip(word.fields())
        .filter(|&(c, f)| mask.contains(c) || f.is_none())
        .fold(ColorWriteMask::NONE, |m, (c, _)| m.union(c));
        let kept: u32 = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .into_iter()
        .zip(word.fields())
        .filter_map(|(c, f)| f.filter(|_| !mask.contains(c)))
        .fold(0, |m, f| m | (((1u32 << f.bits) - 1) << f.shift));
        let math = match fast {
            true => BlendMath::Fast,
            false => BlendMath::Rounded,
        };
        let what = format!("state {state} {pf:?} half {half} fast {fast}");
        let b = Blend::packed16(&eq, written, math).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let cases: Vec<Case> = PIXELS
            .iter()
            .map(|(s, d)| {
                let s =
                    s.iter()
                        .map(|&x| match half {
                            true => (f16_to_f64(f64_to_f16(f64::from(f32::from_bits(x)))) as f32)
                                .to_bits(),
                            false => x,
                        })
                        .map(u64::from)
                        .collect();
                let c = color.iter().map(|&x| u64::from(x)).collect();
                vec![s, vec![u64::from(*d)], c]
            })
            .collect();
        let expected = words;
        for q in HALF_FORMS {
            check_accept(
                &format!("the packed word blend, {what}, half by {q:?}"),
                &[f4, int(32), u4],
                int(32),
                &|fb, x| {
                    fb.module().set_quantize_form(q);
                    let d = crate::lower::stage_io::word_color::decode(fb, word, x[1])?;
                    let color = b.reads_constant().then_some(x[2]);
                    let v = blend_packed16(fb, &b, (x[0], d), color)?;
                    let w = crate::lower::stage_io::word_color::encode(fb, word, (f4, v))?;
                    let u = Form {
                        held: Held::Int {
                            bits: 32,
                            container: 32,
                        },
                        lanes: 1,
                    };
                    let keep = crate::lower::arith::ops::splat(fb, u, u64::from(kept))?;
                    let write = crate::lower::arith::ops::splat(fb, u, u64::from(!kept))?;
                    let old = fb.binary(crate::spirv::ir::BinOp::BitwiseAnd, x[1], keep)?;
                    let new = fb.binary(crate::spirv::ir::BinOp::BitwiseAnd, w, write)?;
                    fb.binary(crate::spirv::ir::BinOp::BitwiseOr, old, new)
                },
                &cases,
                &|c| {
                    let i = PIXELS
                        .iter()
                        .position(|(_, d)| u64::from(*d) == c[1][0])
                        .expect("a pixel");
                    Some(vec![u64::from(expected[i])])
                },
                &|g, w| g == w,
            );
        }
    }
}
