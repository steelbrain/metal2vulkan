//! Metal's own bytes and answers, measured on an Apple M4 by our own probes and committed as
//! `apple_vectors.txt`: every measured map's 24,624 bytes (their SHA-256), and a sample of its CPU
//! and GPU answers, bit for bit, the sign of a zero and a `NaN`'s bits included: inside the screen,
//! in the half pixel before a tile boundary at a multiple of 32, past both ends, at denormals,
//! `NaN`s, infinities and far values, at offsets that lie near a half step, and a shader's layers 1
//! to 5 over the bytes followed by a pattern. A shader's `uint2` answers, measured on an Apple M3
//! Max by our probe `rate_map_integer_overloads`, are `apple_integer_vectors.txt`.

use super::*;
use crate::air::canon::rate_map_data as format;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const VECTORS: &str = include_str!("apple_vectors.txt");
/// A shader's `uint2` answers on an Apple M3 Max (the file's header has its format).
pub(crate) const INTEGER_VECTORS: &str = include_str!("apple_integer_vectors.txt");

/// The words past the parameter data in the probe's padded buffer: word `i` of the buffer.
fn pattern(i: u32) -> u32 {
    if i.is_multiple_of(2) {
        ((1 + i % 7) as f32).to_bits() - (3 << 23)
    } else {
        let offset = ((i * 37) % 2001) as i32 - 1000;
        let boundary = if i.is_multiple_of(3) {
            0xff
        } else {
            (i * 11) % 64
        };
        ((offset as u32) << 8) | boundary
    }
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The descriptor of a vector line's layers: `h:...|v:...` per layer, `;` between, `-` for none.
pub(crate) fn descriptor(width: &str, height: &str, layers: &str) -> RateMapDescriptor {
    let size = |v: &str| v.parse::<u32>().expect("extent");
    let mut d = RateMapDescriptor::new([size(width), size(height)]);
    if layers != "-" {
        for layer in layers.split(';') {
            let (h, v) = layer.split_once('|').expect("axes");
            let samples = |axis: &str, tag: &str| -> Vec<f32> {
                let list = axis.strip_prefix(tag).expect("tag");
                list.split(',')
                    .map(|q| q.parse().expect("sample"))
                    .collect()
            };
            d.layers.push(RateMapLayerDescriptor::new(
                samples(h, "h:"),
                samples(v, "v:"),
            ));
        }
    }
    d
}

/// Every measured map's bytes are Metal's, and every sampled answer is Metal's, bit for bit.
#[test]
fn metals_bytes_and_answers() {
    let mut maps = BTreeMap::new();
    let mut padded = BTreeMap::new();
    let (mut points, mut mismatches) = (0usize, Vec::new());
    for line in VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["map", id, width, height, layers, want] => {
                let map = RasterizationRateMap::new(&descriptor(width, height, layers))
                    .unwrap_or_else(|e| panic!("{id}: {e}"));
                assert_eq!(map.parameter_data_size(), 24_624, "{id}");
                assert_eq!(
                    sha(&map.to_bytes()),
                    *want,
                    "{id}: the bytes are not Metal's"
                );
                maps.insert(id.to_string(), map);
            }
            ["padded", id, bytes, want] => {
                let map = &maps[*id];
                let total = bytes.parse::<u32>().expect("bytes") / 4;
                let words: Vec<u32> = (0..total)
                    .map(|i| {
                        map.words()
                            .get(i as usize)
                            .copied()
                            .unwrap_or_else(|| pattern(i))
                    })
                    .collect();
                let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                assert_eq!(sha(&raw), *want, "{id}: the padded buffer");
                padded.insert(id.to_string(), words);
            }
            [id, side, kind, layer, axis, input, output] => {
                let map = &maps[*id];
                let layer: u32 = layer.parse().expect("layer");
                let axis: usize = axis.parse().expect("axis");
                let bits = |v: &str| u32::from_str_radix(v, 16).expect("bits");
                let x = f32::from_bits(bits(input));
                let mut c = [0.0f32; 2];
                c[axis] = x;
                let got = match (*side, *kind) {
                    ("cpu", "s2p") => map.screen_to_physical(c, layer),
                    ("cpu", "p2s") => map.physical_to_screen(c, layer),
                    ("gpu", "s2p") => map.shader_screen_to_physical(c, layer),
                    ("gpu", "p2s") => map.shader_physical_to_screen(c, layer),
                    ("pad", "s2p") => format::shader_screen_to_physical(&padded[*id], layer, c),
                    ("pad", "p2s") => format::shader_physical_to_screen(&padded[*id], layer, c),
                    other => panic!("{other:?}"),
                };
                points += 1;
                if got[axis].to_bits() != bits(output) {
                    mismatches.push(format!("{line}: got {:08x}", got[axis].to_bits()));
                }
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(points > 4000, "{points}");
    assert!(
        mismatches.is_empty(),
        "{} of {points} answers are not Metal's:\n{}",
        mismatches.len(),
        mismatches[..mismatches.len().min(20)].join("\n")
    );
}

/// A shader's `uint2` maps answer as Metal's, bit for bit, over each measured map's bytes, which
/// are Metal's (`apple_integer_vectors.txt`, an Apple M3 Max under macOS 27.0.1): screen to
/// physical the `float2` decoder's answer at the pixel's centre converted to `uint`, physical to
/// screen the bits of a `float` the granule's entry gives, at the maps' layers and two past them.
#[test]
fn metals_integer_answers() {
    let mut maps = BTreeMap::new();
    let (mut points, mut mismatches) = (0usize, Vec::new());
    for line in INTEGER_VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["map", id, width, height, layers, want] => {
                let map = RasterizationRateMap::new(&descriptor(width, height, layers))
                    .unwrap_or_else(|e| panic!("{id}: {e}"));
                assert_eq!(
                    sha(&map.to_bytes()),
                    *want,
                    "{id}: the bytes are not Metal's"
                );
                maps.insert(id.to_string(), map);
            }
            [id, layer, axis, input, to_physical, to_screen] => {
                let words = maps[*id].words();
                let number = |v: &str| v.parse::<u32>().expect("number");
                let (layer, axis) = (number(layer), number(axis) as usize);
                let mut c = [0u32; 2];
                c[axis] = number(input);
                let got = [
                    format::shader_screen_to_physical_integer(words, layer, c)[axis],
                    format::shader_physical_to_screen_integer(words, layer, c)[axis],
                ];
                let want = [
                    number(to_physical),
                    u32::from_str_radix(to_screen, 16).expect("bits"),
                ];
                points += 1;
                if got != want {
                    mismatches.push(format!("{line}: got {} {:08x}", got[0], got[1]));
                }
            }
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(points, 1561, "the file's answers, every one read");
    assert!(
        mismatches.is_empty(),
        "{} of {points} answers are not Metal's:\n{}",
        mismatches.len(),
        mismatches[..mismatches.len().min(20)].join("\n")
    );
}

/// A reset to and from a map of a zero screen extent writes Metal's words (an Apple M3 Max under
/// macOS 27.0.1, our probe `vec`, one map per process): a 64 by 64 map of quality 0.5 reset to 0
/// by 64 keeps the entries the narrower map does not write, and the same map reset the other way
/// holds exactly what a fresh 64 by 64 map holds.
#[test]
fn metals_words_across_a_zero_extent_reset() {
    let half = |screen| {
        let mut d = RateMapDescriptor::new(screen);
        d.layers
            .push(RateMapLayerDescriptor::new(vec![0.5], vec![0.5]));
        d
    };
    let wide = RasterizationRateMap::new(&half([64, 64])).expect("64 by 64");
    let narrowed =
        RasterizationRateMap::new_over(&half([0, 64]), wide.words()).expect("reset to 0 by 64");
    assert_eq!(narrowed.physical_size(0), Some([0, 33]));
    assert_eq!(
        sha(&narrowed.to_bytes()),
        "e8367ffa698309495b1907fddd1a752dee56f95d953f513aa11a2a900f7c221d",
        "the reset to a zero screen width is not Metal's"
    );
    let narrow = RasterizationRateMap::new(&half([0, 64])).expect("0 by 64");
    let widened =
        RasterizationRateMap::new_over(&half([64, 64]), narrow.words()).expect("reset to 64 by 64");
    assert_eq!(widened.to_bytes(), wide.to_bytes());
    assert_eq!(
        sha(&widened.to_bytes()),
        "4c21eca9c26d2f2d4f50ce61db86bda15021461bf41b9eccc94798443a93dd5e",
        "the reset from a zero screen width is not Metal's"
    );
}
