//! The bytes Metal's private texture fills write (`blit_fill.rs`): each multisampled sample's,
//! from our probe `ms.m`'s read-backs, and a texture buffer's, from our probe `tb.m`'s (the M3
//! Max under macOS 27.0.1), and the formats a fill has a unit for.

use crate::blit_fill::{fill_unit, read_pattern, sample_units, words, StreamFill};
use crate::blit_fill::{tile_copies, tile_extent, unit_of, TEXTURE_BUFFER_PATTERN_BYTES};
use metal2vulkan::metal::{PixelFormat, TextureType};
use std::ops::Range;

/// Bytes from hex.
fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// Each sample's bytes as hex.
fn samples(pattern: &str, bytes: u32, count: u32) -> Vec<String> {
    sample_units(&hex(pattern), bytes, count)
        .iter()
        .map(|unit| unit.iter().map(|b| format!("{b:02x}")).collect())
        .collect()
}

/// The probe's 16-byte pattern.
const P16: &str = "00112233445566778899aabbccddeeff";

/// The probe's 32-byte pattern.
const P32: &str = "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f";

/// Where a texel's samples hold 16 bytes or fewer, sample `s` takes the pattern's bytes from
/// `s · b`, the pattern repeated and cut at the samples' bytes; past 16, each sample the pattern
/// cut at its texel.
#[test]
fn multisampled_texels_take_metal_s_bytes_of_the_pattern() {
    // `R8Unorm`, `Stencil8` at 4 and 2 samples.
    assert_eq!(samples(P16, 1, 4), ["00", "11", "22", "33"]);
    assert_eq!(samples(P32, 1, 2), ["00", "11"]);
    // `RG8Unorm`, `Depth16Unorm` and the 16-bit packed formats.
    assert_eq!(samples(P16, 2, 4), ["0011", "2233", "4455", "6677"]);
    assert_eq!(samples(P32, 2, 2), ["0011", "2233"]);
    // `RGBA8Unorm`, `Depth32Float`, `BGR10_XR`, `RGB10A2Unorm_sRGB`; a 32-byte pattern cut at 16.
    let four = ["00112233", "44556677", "8899aabb", "ccddeeff"];
    assert_eq!(samples(P16, 4, 4), four);
    assert_eq!(samples(P32, 4, 4), four);
    assert_eq!(samples(P32, 4, 2), ["00112233", "44556677"]);
    // `RGBA16Float`, `BGRA10_XR`: 16 bytes at 2 samples, the texel's 8 each at 4.
    assert_eq!(samples(P32, 8, 2), ["0011223344556677", "8899aabbccddeeff"]);
    assert_eq!(samples(P16, 8, 4), ["0011223344556677"; 4]);
    // `RGBA32Float` at 2 and 4.
    assert_eq!(samples(P32, 16, 2), [P16; 2]);
    assert_eq!(samples(P16, 16, 4), [P16; 4]);
    // Patterns that do not divide the samples' bytes.
    assert_eq!(samples("a1b2c3", 1, 4), ["a1", "b2", "c3", "a1"]);
    assert_eq!(samples("a1b2c3", 2, 4), ["a1b2", "c3a1", "b2c3", "a1b2"]);
    assert_eq!(
        samples("a1b2c3d4e5", 4, 4),
        ["a1b2c3d4", "e5a1b2c3", "d4e5a1b2", "c3d4e5a1"]
    );
    assert_eq!(samples("a1b2c3d4e5", 4, 2), ["a1b2c3d4", "e5a1b2c3"]);
    assert_eq!(
        samples("a1b2c3d4e5f6", 8, 2),
        ["a1b2c3d4e5f6a1b2", "c3d4e5f6a1b2c3d4"]
    );
    assert_eq!(
        samples("a1b2c3", 16, 4),
        ["a1b2c3a1b2c3a1b2c3a1b2c3a1b2c3a1"; 4]
    );
    // A 20-byte pattern into 16.
    assert_eq!(
        samples("0102030405060708090a0b0c0d0e0f1011121314", 4, 4),
        ["01020304", "05060708", "090a0b0c", "0d0e0f10"]
    );
}

/// A sample's bytes as the words of a clear value of the unsigned format of its size.
#[test]
fn sample_bytes_become_clear_words() {
    assert_eq!(words(&hex("a1")), [0xa1, 0, 0, 0]);
    assert_eq!(words(&hex("a1b2")), [0xb2a1, 0, 0, 0]);
    assert_eq!(words(&hex("00112233")), [0x3322_1100, 0, 0, 0]);
    assert_eq!(
        words(&hex("0011223344556677")),
        [0x3322_1100, 0x7766_5544, 0, 0]
    );
    assert_eq!(
        words(&hex(P16)),
        [0x3322_1100, 0x7766_5544, 0xbbaa_9988, 0xffee_ddcc]
    );
}

/// Every pixel format the crate holds a texture of (one with aspects) has a fill unit, but the
/// 24-bit depth formats, which Apple GPUs make no texture of (refused at the texture's making),
/// and the formats held for copies alone, whose fills are refused (Metal's are not measured).
#[test]
fn every_texture_format_has_a_fill_unit() {
    use metal2vulkan::metal::format::{aspects, copies_only};
    use PixelFormat as P;
    let none: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| aspects(pf).is_some() && !copies_only(pf) && fill_unit(pf).is_none())
        .collect();
    assert_eq!(
        none,
        [P::Depth24UnormStencil8, P::X24Stencil8],
        "formats without a fill unit"
    );
}

/// The pattern of `n` bytes `a1 a2 ..`, the probe's.
fn counting(n: usize, from: u8) -> Vec<u8> {
    (0..n).map(|i| from.wrapping_add(i as u8)).collect()
}

/// The bytes `bytes` of a texture buffer's buffer a fill of `pattern` writes, as hex.
fn streamed(pattern: &[u8], bytes: Range<u64>) -> String {
    let stream = StreamFill::new(pattern);
    bytes
        .clone()
        .map(|at| format!("{:02x}", stream.byte(&bytes, at)))
        .collect()
}

/// A texture buffer's fill writes the bytes Metal's writes: the pattern by the masked index of the
/// region's byte, but in each 16-byte chunk of the buffer the region covers whole, 16 bytes of the
/// block of the pattern's whole copies, zeros past them (`tb.m`'s cases, the region's bytes from
/// the texture's offset plus `x` texels).
#[test]
fn texture_buffer_fills_write_metal_s_bytes() {
    // 8 bytes into 5 `R8Unorm` texels from byte 7, and 8 `RGBA8Unorm` texels from byte 80.
    assert_eq!(streamed(&counting(8, 0xa1), 7..12), "a1a2a3a4a5");
    assert_eq!(
        streamed(&counting(8, 0xa1), 80..112),
        "a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8a1a2a3a4a5a6a7a8"
    );
    // 3 bytes, 12 `RGBA8Unorm` texels from 0: 5 copies a chunk, a zero past them.
    assert_eq!(
        streamed(&counting(3, 0xa1), 0..48),
        "a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300"
    );
    // 3 bytes, 30 `R8Unorm` texels from 5: `p0 p0 p2 p2` outside the chunk.
    assert_eq!(
        streamed(&counting(3, 0xa1), 5..35),
        "a1a1a3a3a1a1a3a3a1a1a3a1a2a3a1a2a3a1a2a3a1a2a3a1a2a300a3a1a1"
    );
    // 12 bytes from byte 8: the chunks read from the block's byte 8, past the one copy.
    assert_eq!(
        streamed(&counting(12, 0xa1), 8..48),
        "a1a2a3a4a1a2a3a4a9aaabac000000000000000000000000a9aaabac000000000000000000000000"
    );
    // 5 bytes into 40 `RG8Unorm` texels from byte 12.
    assert_eq!(
        streamed(&counting(5, 0xa1), 12..92),
        "a1a1a1a1a5a1a2a3a4a5a1a2a3a4a50000000000a5a1a2a3a4a5a1a2a3a4a50000000000\
         a5a1a2a3a4a5a1a2a3a4a50000000000a5a1a2a3a4a5a1a2a3a4a50000000000a5a5a5a5a1a1a1a1a5a5a5a5"
    );
    // 17 bytes into 12 `RGBA32Float` texels: a 32-byte block of one copy.
    let seventeen = "a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1000000000000000000000000000000";
    assert_eq!(streamed(&counting(17, 0xa1), 0..192), seventeen.repeat(6));
    // 24 bytes into 30 `RGBA16Float` texels from byte 88.
    let copy = "a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b80000000000000000";
    assert_eq!(
        streamed(&counting(24, 0xa1), 88..328),
        format!("a1a2a3a4a5a6a7a8{}a1a2a3a4a5a6a7a8", copy.repeat(7))
    );
    // 100 bytes into 40 `RGBA8Unorm` texels from byte 4: a 128-byte block.
    assert_eq!(
        streamed(&counting(100, 0x10), 4..164),
        "101112131011121310111213101112131415161718191a1b1c1d1e1f101112131415161718191a1b1c1d1e1f\
         303132333435363738393a3b3c3d3e3f303132333435363738393a3b3c3d3e3f505152535455565758595a5b\
         5c5d5e5f505152535455565758595a5b5c5d5e5f707172730000000000000000000000007071727300000000\
         0000000000000000101112131415161718191a1b1c1d1e1f10111213"
    );
}

/// A texture buffer's fill as its spans (each its unit repeated from the span's first byte)
/// writes every byte the fill does, and announces the bytes past the pattern's copies where a
/// chunk reads one: patterns of 1 to 70 bytes over regions of 0 to 200 bytes from 0 to 47.
#[test]
fn texture_buffer_fill_spans_are_its_bytes() {
    for n in 1..=70 {
        let pattern = counting(n, 0x01);
        let stream = StreamFill::new(&pattern);
        let copies = (16.max(n.next_power_of_two()) / n * n) as u64;
        for start in 0..48u64 {
            for length in 0..200u64 {
                let bytes = start..start + length;
                let mut written = vec![None; (start + length) as usize];
                for (span, unit) in stream.spans(bytes.clone()) {
                    for (k, at) in span.enumerate() {
                        assert!(written[at as usize].is_none(), "{n} {bytes:?}: twice");
                        written[at as usize] = Some(unit[k % unit.len()]);
                    }
                }
                let want: Vec<Option<u8>> = (0..start + length)
                    .map(|at| bytes.contains(&at).then(|| stream.byte(&bytes, at)))
                    .collect();
                assert_eq!(written, want, "{n} {bytes:?}");
                // Brute force: a whole chunk's byte read from the block's copies or past them.
                let leftover = (start.next_multiple_of(16)..(start + length) / 16 * 16).any(|at| {
                    let chunk = at / 16 * 16;
                    ((chunk - start) & (n as u64 - 1) & !3) + at % 16 >= copies
                });
                assert_eq!(
                    stream.reads_leftover(bytes.clone()),
                    leftover,
                    "{n} {bytes:?}"
                );
            }
        }
    }
}

/// The `n` bytes of our probe `tblf.m`'s pattern, a xorshift stream: no run of it repeats.
fn probe_stream(n: usize) -> Vec<u8> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8 | 1
        })
        .collect()
}

/// A long pattern's fill of a texture buffer writes every byte the rule gives, its block made
/// here as the brute force makes it (the pattern's whole copies, then zeros), while the fill's
/// spans hold units of at most the block's bytes, read from the pattern: patterns of 4097, 16385
/// and 32752 bytes (Metal's longest) into 100,000 bytes from byte 4, the probe `tblf.m`'s
/// fills of up to 100,000 `R8Uint` texels, which Metal writes so byte for byte.
#[test]
fn a_long_texture_buffer_pattern_streams_in_units_of_its_block() {
    for n in [4097, 16385, TEXTURE_BUFFER_PATTERN_BYTES] {
        let pattern = probe_stream(n);
        let block = n.next_power_of_two().max(16);
        let copies = block / n * n;
        let mut memory: Vec<u8> = pattern.iter().copied().cycle().take(copies).collect();
        memory.resize(block + 16, 0);
        let bytes = 4..100_004u64;
        let stream = StreamFill::new(&pattern);
        let mut written = vec![None; bytes.end as usize];
        for (span, unit) in stream.spans(bytes.clone()) {
            assert!(unit.len() <= block, "{n}: a unit of {} bytes", unit.len());
            for (k, at) in span.enumerate() {
                assert!(written[at as usize].is_none(), "{n}: {at} twice");
                written[at as usize] = Some(unit[k % unit.len()]);
            }
        }
        for at in bytes.clone() {
            let want = match (16..100_000).contains(&at) {
                true => memory[(((at / 16 * 16 - 4) as usize & (n - 1)) & !3) + at as usize % 16],
                false => pattern[(at - 4) as usize & (n - 1)],
            };
            assert_eq!(written[at as usize], Some(want), "{n}: byte {at}");
            assert_eq!(stream.byte(&bytes, at), want, "{n}: byte {at}");
        }
        assert_eq!(written[..4], [None; 4]);
    }
}

/// The encoder's log keeps the bytes of a pattern its fill reads: a texture buffer's whole
/// pattern, cut one byte past Metal's longest (refused again where the call is encoded again);
/// another texture's first 16, from which every unit and every multisampled texel's samples
/// come out as from the whole pattern.
#[test]
fn a_fill_s_log_keeps_the_bytes_its_fill_reads() {
    let long = probe_stream(1 << 20);
    let buffer = read_pattern(TextureType::TextureBuffer, &long);
    assert_eq!(buffer, &long[..TEXTURE_BUFFER_PATTERN_BYTES + 1]);
    assert_eq!(
        read_pattern(TextureType::TextureBuffer, &long[..300]),
        &long[..300]
    );
    for n in [1, 3, 15, 16, 17, 64, 1 << 20] {
        let kept = read_pattern(TextureType::Type2D, &long[..n]);
        assert_eq!(kept, &long[..n.min(16)]);
        for bytes in 1..=16 {
            assert_eq!(
                unit_of(kept, bytes),
                unit_of(&long[..n], bytes),
                "{n} {bytes}"
            );
            for count in [2, 4, 8] {
                assert_eq!(
                    sample_units(kept, bytes, count),
                    sample_units(&long[..n], bytes, count),
                    "{n} {bytes} {count}"
                );
            }
        }
    }
}

/// A fill's pattern length falls on either side of what misuse is ([`check_pattern`]). No bytes at
/// all is input plain Metal takes, whatever the texture's type, so it is the runtime's own gap: its
/// API validation layer stops the call, and plain Metal, the layer off, writes the region from
/// bytes the call never gave (an Apple M3 Max under macOS 27.0.1, our probe `emptyfill.m`: a 2D
/// texture over a buffer, one of its own memory, a texture buffer and a 4-sample one all took the
/// fill and completed, the first two writing the unit under the pattern's pointer and the texture
/// buffer the fill's leftovers). A texture buffer's pattern of more than
/// [`TEXTURE_BUFFER_PATTERN_BYTES`] is misuse, in which Metal's own process ends, and no other
/// texture's fill has an upper bound (our probes `tblf.m` and `tbz.m`, the same M3 Max).
#[test]
fn a_fill_s_pattern_length_separates_the_runtime_s_gap_from_misuse() {
    use crate::blit_fill::check_pattern;
    use crate::error::{invalid, Error, Problem, RuntimeGap};
    const CALL: &str = "fill_texture";
    const TYPES: [TextureType; 4] = [
        TextureType::Type2D,
        TextureType::TextureBuffer,
        TextureType::Type2DMultisample,
        TextureType::Cube,
    ];
    let longest = vec![0u8; TEXTURE_BUFFER_PATTERN_BYTES];
    for ty in TYPES {
        assert_eq!(
            check_pattern(CALL, ty, &[]),
            Err(Error::NotImplemented(RuntimeGap::EmptyFillPattern)),
            "{ty:?} of no bytes"
        );
        assert_eq!(check_pattern(CALL, ty, &[0]), Ok(()), "{ty:?} of one byte");
        assert_eq!(
            check_pattern(CALL, ty, &longest),
            Ok(()),
            "{ty:?} of Metal's longest"
        );
    }
    // One byte past it, a texture buffer's fill is misuse; no other texture's fill is bounded.
    let long = vec![0u8; TEXTURE_BUFFER_PATTERN_BYTES + 1];
    assert_eq!(
        check_pattern(CALL, TextureType::TextureBuffer, &long),
        Err(invalid(
            CALL,
            Problem::LongTextureBufferFillPattern {
                length: TEXTURE_BUFFER_PATTERN_BYTES as u64 + 1,
            }
        ))
    );
    for ty in TYPES
        .into_iter()
        .filter(|&ty| ty != TextureType::TextureBuffer)
    {
        assert_eq!(
            check_pattern(CALL, ty, &long),
            Ok(()),
            "{ty:?} of a long one"
        );
    }
}

/// A plain color fill's tile is the region's extent rounded up to a power of two of at least 16
/// texels, no larger than the texture's first level, its longer side then halved until it holds a
/// MiB at most, whatever the texel's bytes and samples (`blit_fill.rs`).
#[test]
fn a_fill_tile_is_the_region_rounded_up_to_a_mib() {
    // `RGBA8Unorm`: a 256 x 256 region and a 16 x 16 one whole, a 2048 x 2048 one 512 x 512.
    assert_eq!(tile_extent([256, 256], 4, [256, 256]), [256, 256]);
    assert_eq!(tile_extent([16, 16], 4, [256, 256]), [16, 16]);
    assert_eq!(tile_extent([2048, 2048], 4, [2048, 2048]), [512, 512]);
    // Near sizes take one extent, which the texture's first level bounds.
    assert_eq!(tile_extent([1, 1], 4, [256, 256]), [16, 16]);
    assert_eq!(tile_extent([17, 9], 4, [256, 256]), [32, 16]);
    assert_eq!(tile_extent([200, 33], 4, [300, 300]), [256, 64]);
    assert_eq!(tile_extent([260, 10], 4, [300, 300]), [300, 16]);
    assert_eq!(tile_extent([5, 1], 4, [8, 1]), [8, 1]);
    // `RGBA32Float` at 4 samples: 64 bytes a texel.
    assert_eq!(tile_extent([2048, 2048], 64, [2048, 2048]), [128, 128]);
    // A 1D row, and a narrow column, halved along their length (rounded up).
    assert_eq!(tile_extent([16384, 1], 16, [16384, 1]), [16384, 1]);
    assert_eq!(tile_extent([3, 100_000], 16, [3, 100_000]), [3, 12500]);
    assert_eq!(tile_extent([1, 1], 128, [1, 1]), [1, 1]);
    for (size, bytes) in [([2048, 2048], 16), ([1000, 777], 8), ([16384, 16384], 128)] {
        let [w, h] = tile_extent(size, bytes, size);
        assert!(
            u64::from(w) * u64::from(h) * bytes <= 1 << 20,
            "{size:?} {bytes}"
        );
    }
}

/// The copies out of a tile cover the fill's region once, from its origin, each part from the
/// tile's origin and no larger than the tile, in the image's level and layer.
#[test]
fn fill_tile_copies_cover_the_region_once() {
    let copies = tile_copies([512, 512], [3, 5, 2], [1100, 600], (4, 7));
    assert_eq!(copies.len(), 3 * 2);
    let mut texels = 0u64;
    for copy in &copies {
        assert_eq!(
            (copy.src.level, copy.src.layer, copy.src.origin),
            (0, 0, [0, 0, 0])
        );
        assert_eq!((copy.dst.level, copy.dst.layer, copy.dst.layers), (4, 7, 1));
        assert_eq!(copy.src.size, copy.dst.size);
        let [w, h, d] = copy.dst.size;
        assert!(w <= 512 && h <= 512 && d == 1);
        let [x, y, z] = copy.dst.origin;
        assert!(x >= 3 && y >= 5 && z == 2 && x + w <= 1103 && y + h <= 605);
        texels += u64::from(w) * u64::from(h);
    }
    assert_eq!(texels, 1100 * 600);
    // A region inside the tile is one copy of its size.
    let one = tile_copies([16, 16], [32, 48, 0], [16, 16], (0, 0));
    assert_eq!(one.len(), 1);
    assert_eq!(
        (one[0].dst.origin, one[0].dst.size),
        ([32, 48, 0], [16, 16, 1])
    );
}
