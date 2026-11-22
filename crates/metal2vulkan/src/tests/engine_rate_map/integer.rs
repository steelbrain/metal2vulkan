//! The `uint2` overloads end to end, in the fixture `rate_map_integer` (compiled from our MSL by
//! `scripts/regen-air-fixtures`): its kernel and its fragment function compile on every tested
//! profile and pass L3. On the CPU interpreter, which keeps
//! `float` denormals as lavapipe does, the kernel answers what `rate_map_data`'s shader decoders
//! compute with denormals kept ([`kept_integer`]), for every layer and two past a map's, inside
//! the screen, at and around each multiple of 32, past both extents and at far coordinates; where
//! no denormal arises, that is Metal's answer (`shader_screen_to_physical_integer` and
//! `shader_physical_to_screen_integer`, measured on an Apple M3 Max).

use super::*;

const FIXTURE: &str = "../metal2vulkan-validation/fixtures/public/engine/rate_map_integer.ll";

/// The fixture's module, assembled; `None` (the test skips visibly) without `llvm-as`.
fn fixture() -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    assemble(&std::fs::read_to_string(path).expect("the fixture"))
}

/// A shader's `uint2` map of `c` over parameter data `words` in layer `layer`, to physical
/// coordinates if `to_physical`, as the translated decoder computes it on a device that keeps
/// `float` denormals: screen to physical the `float2` decoder's arithmetic ([`kept`]) at the
/// pixel's centre, converted saturating; physical to screen the bits of `(0 - offset) ·
/// reciprocal` of granule `c >> 5`'s entry, IEEE `f32`, a read past the words 0.
fn kept_integer(words: &[u32], layer: u32, c: [u32; 2], to_physical: bool) -> [u32; 2] {
    if to_physical {
        let centre = c.map(|v| v as f32 + 0.5);
        return kept(words, layer, centre, true).map(format::shader_index);
    }
    let word = |i: u32| words.get(i as usize).copied().unwrap_or(0);
    let axis = |axis: u32, p: u32| {
        let at = format::table(layer, axis, false)
            .wrapping_add((p >> format::GRANULE_SHIFT).wrapping_mul(format::ENTRY_WORDS));
        let (reciprocal, offset) = (word(at), word(at.wrapping_add(1)));
        ((0.0 - f32::from_bits(offset)) * f32::from_bits(reciprocal)).to_bits()
    };
    [axis(0, c[0]), axis(1, c[1])]
}

/// Bit for bit; a physical-to-screen answer that is a `NaN`'s bits matches any `NaN`'s (the
/// module leaves a `NaN`'s bits to the device, which the interpreter does not model).
fn same_answer(got: [u32; 4], want: [u32; 4]) -> bool {
    let nan = |b: u32| f32::from_bits(b).is_nan();
    got[..2] == want[..2]
        && (got[2..].iter())
            .zip(&want[2..])
            .all(|(&g, &w)| g == w || (nan(g) && nan(w)))
}

/// The coordinates every map is decoded at: the first granule's, each multiple of 32 and its
/// neighbours past both extents, and coordinates whose index reads far past the words.
fn integer_points(screen: [u32; 2]) -> Vec<[u32; 2]> {
    let reach = screen[0].max(screen[1]) + 96;
    let mut out: Vec<[u32; 2]> = (0..34).map(|i| [i, 33 - i]).collect();
    for edge in (32..reach).step_by(32) {
        for (i, c) in [edge - 1, edge, edge + 1].into_iter().enumerate() {
            out.push([c, reach - c + i as u32]);
        }
    }
    out.extend([[16_383, 1 << 20], [1 << 24, 7], [u32::MAX, 1 << 31]]);
    out
}

fn uints(c: &[[u32; 2]]) -> Vec<u8> {
    c.iter()
        .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
        .collect()
}

/// Both `uint2` maps of every layer, as the shader decoders' arithmetic computes them with the
/// device's denormals ([`kept_integer`]), on every profile; Metal's answers at every point of a map's own layers.
#[test]
fn the_integer_maps_answer_as_metal() {
    let Some(bc) = fixture() else {
        return;
    };
    let (mut metals, mut apart) = (0, 0);
    for stem in PROFILES {
        let what = stem;
        let p = profile_with(stem);
        for entry in ["rate_map_integer", "rate_map_integer_pixel"] {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what} {entry}: L3: {e}"));
        }
        let (plan, m) = typed_module(&bc, "rate_map_integer", &p).expect("module");
        for (name, map) in maps() {
            let words = map.words();
            let pts = integer_points(map.screen_size());
            for layer in 0..map.layer_count() + 2 {
                let res = Resources {
                    buffers: vec![
                        bound(0, &map.to_bytes()),
                        bound(1, &uints(&pts)),
                        bound(2, &vec![0u8; pts.len() * 16]),
                        bound(3, &layer.to_le_bytes()),
                    ],
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let cfg = Config {
                    workgroups: [pts.len() as u32, 1, 1],
                    ..Config::default()
                };
                let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                for (i, &c) in pts.iter().enumerate() {
                    let mut got = [0u32; 4];
                    for (k, b) in out.buffers[2][i * 16..i * 16 + 16].chunks(4).enumerate() {
                        got[k] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    }
                    let [s, t] = [true, false].map(|to| kept_integer(words, layer, c, to));
                    let want = [s[0], s[1], t[0], t[1]];
                    assert!(
                        same_answer(got, want),
                        "{what}: {name} layer {layer} at {c:?}: {got:x?}, kept {want:x?}"
                    );
                    let [s, t] = [
                        format::shader_screen_to_physical_integer(words, layer, c),
                        format::shader_physical_to_screen_integer(words, layer, c),
                    ];
                    if same_answer(want, [s[0], s[1], t[0], t[1]]) {
                        metals += 1;
                    } else {
                        assert!(
                            layer >= map.layer_count(),
                            "{what}: {name} layer {layer} at {c:?}: {want:x?}, Metal's \
                             {s:?} {t:x?}"
                        );
                        apart += 1;
                    }
                }
            }
        }
    }
    // Both kinds of point were decoded, so neither comparison is vacuous.
    assert!(
        metals > 0 && apart > 0,
        "{metals} points Metal's, {apart} apart"
    );
}

/// The translated kernel answers Metal's recorded `uint2` answers (`apple_integer_vectors.txt`,
/// an Apple M3 Max under macOS 27.0.1) at every recorded point of each map's own layers, on the
/// CPU interpreter, every tested profile: screen to physical and physical to screen, bit for bit.
/// The points of layers past a map's read other tables' words, whose `float`s may be denormal,
/// which the interpreter keeps where Metal flushes them (the device's own denormals,
/// `PrecisionNote::DeviceDenormals`); [`the_integer_maps_answer_as_metal`] holds them to the
/// decoder's arithmetic instead.
#[test]
fn the_translated_integer_maps_answer_metals_recorded_answers() {
    use crate::bind::rate_map::apple_tests::{descriptor, INTEGER_VECTORS};
    use std::collections::BTreeMap;
    let Some(bc) = fixture() else {
        return;
    };
    // Each map, and its recorded points by layer: the axis, the coordinate and both answers.
    let mut maps: BTreeMap<&str, RasterizationRateMap> = BTreeMap::new();
    let mut points: BTreeMap<(&str, u32), Vec<(usize, u32, [u32; 2])>> = BTreeMap::new();
    for line in INTEGER_VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let number = |v: &str| v.parse::<u32>().expect("number");
        match f.as_slice() {
            ["map", id, width, height, layers, _] => {
                let map = RasterizationRateMap::new(&descriptor(width, height, layers))
                    .unwrap_or_else(|e| panic!("{id}: {e}"));
                maps.insert(id, map);
            }
            [id, layer, axis, input, to_physical, to_screen] => {
                let to_screen = u32::from_str_radix(to_screen, 16).expect("bits");
                let point = (
                    number(axis) as usize,
                    number(input),
                    [number(to_physical), to_screen],
                );
                points.entry((id, number(layer))).or_default().push(point);
            }
            other => panic!("{other:?}"),
        }
    }
    let mut compared = 0usize;
    for stem in PROFILES {
        let p = profile_with(stem);
        let (plan, m) = typed_module(&bc, "rate_map_integer", &p).expect("module");
        for (&(id, layer), recorded) in &points {
            let map = &maps[id];
            if layer >= map.layer_count() {
                continue;
            }
            let coordinates: Vec<[u32; 2]> = (recorded.iter())
                .map(|&(axis, c, _)| {
                    let mut at = [0u32; 2];
                    at[axis] = c;
                    at
                })
                .collect();
            let res = Resources {
                buffers: vec![
                    bound(0, &map.to_bytes()),
                    bound(1, &uints(&coordinates)),
                    bound(2, &vec![0u8; coordinates.len() * 16]),
                    bound(3, &layer.to_le_bytes()),
                ],
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let cfg = Config {
                workgroups: [coordinates.len() as u32, 1, 1],
                ..Config::default()
            };
            let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            for (i, &(axis, c, want)) in recorded.iter().enumerate() {
                let word = |k: usize| {
                    let at = i * 16 + k * 4;
                    let b = &out.buffers[2][at..at + 4];
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                };
                let got = [word(axis), word(2 + axis)];
                assert_eq!(
                    got, want,
                    "{stem}: {id} layer {layer} axis {axis} at {c}: got {got:x?}, Metal's {want:x?}"
                );
                compared += 1;
            }
        }
    }
    // Every recorded point of a map's own layers, on every profile.
    assert_eq!(
        compared,
        587 * PROFILES.len(),
        "the recorded points compared"
    );
}
