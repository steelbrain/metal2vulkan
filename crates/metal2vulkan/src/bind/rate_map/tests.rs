//! The consumer's rasterization-rate map: the compile rule, the parameter data, and the CPU
//! answers, checked against exact arithmetic and against Apple's own map (Metal's measured bytes
//! and answers bit for bit are `apple_tests`).

use super::*;
use crate::air::canon::rate_map_data as format;

fn layer(h: &[f32], v: &[f32]) -> RateMapLayerDescriptor {
    RateMapLayerDescriptor::new(h.to_vec(), v.to_vec())
}

fn map(screen: [u32; 2], layers: &[RateMapLayerDescriptor]) -> RasterizationRateMap {
    let mut d = RateMapDescriptor::new(screen);
    d.layers = layers.to_vec();
    RasterizationRateMap::new(&d).expect("map")
}

/// Quality 1 everywhere is the identity, inside the screen and on the lines past it.
#[test]
fn full_quality_is_the_identity() {
    let m = map([64, 48], &[layer(&[1.0], &[1.0, 1.0, 1.0])]);
    assert_eq!(m.physical_size(0), Some([64, 48]));
    assert_eq!(m.physical_granularity(), [32, 32]);
    assert_eq!(m.layer(0).expect("layer")[1].tiles(), [(32, 32), (16, 16)]);
    for c in [
        [0.0, 0.0],
        [1.5, 2.5],
        [31.25, 47.75],
        [63.5, 47.5],
        [64.0, 48.0],
        [80.0, 7.25],
        [-3.5, 2.0],
        [8.125, 55.875],
    ] {
        assert_eq!(m.screen_to_physical(c, 0), c, "{c:?}");
        assert_eq!(m.physical_to_screen(c, 0), c, "{c:?}");
    }
}

/// Tiles of `spans` screen pixels for 32 physical ones each, then the last tile.
fn tiles(spans: &[u32], last: (u32, u32)) -> Vec<(u32, u32)> {
    spans.iter().map(|&s| (s, 32)).chain([last]).collect()
}

/// `count` tiles of `(s, p)`, then the last tile.
fn run(count: usize, tile: (u32, u32), last: (u32, u32)) -> Vec<(u32, u32)> {
    std::iter::repeat_n(tile, count).chain([last]).collect()
}

/// The exact line's value within Apple's fixed point (a scale of `K / 2^24`, an offset of 1/256
/// pixels).
fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1.0e-4 * want.abs().max(1.0)
}

/// Apple M3's half quality: 63 screen pixels per 32-pixel tile, and a last tile by Apple's own
/// rule.
#[test]
fn half_quality_takes_apples_63_pixel_tiles() {
    let m = map([1024, 100], &[layer(&[0.5], &[0.5])]);
    // 16 tiles of 63 end at 1008; the 16 left map one to one. 100 is 63, then 37 to 32.
    assert_eq!(m.physical_size(0), Some([528, 64]));
    let [x, y] = m.layer(0).expect("layer");
    assert_eq!(x.tiles(), run(16, (63, 32), (16, 16)));
    assert_eq!(y.tiles(), [(63, 32), (37, 32)]);
    for (screen, physical) in [
        (0.0f32, 0.0f32),
        (63.0, 32.0),
        (1008.0, 512.0),
        (1024.0, 528.0),
    ] {
        let p = m.screen_to_physical([screen, 0.0], 0)[0];
        assert!(close(p, physical), "{screen}: {p}");
        let s = m.physical_to_screen([physical, 0.0], 0)[0];
        assert!(close(s, screen), "{physical}: {s}");
    }
    // The and the maps: 64 at 0.5 is 33 physical pixels, 32 at 0.5 the identity, and the 37x23 map
    // of `[0.5, 1]` across and `[1, 0.5]` down 36x23.
    assert_eq!(
        map([64, 64], &[layer(&[0.5], &[0.5])]).physical_size(0),
        Some([33, 33])
    );
    assert!(map([32, 32], &[layer(&[0.5], &[0.5])]).is_identity());
    let skewed = map([37, 23], &[layer(&[0.5, 1.0], &[1.0, 0.5])]);
    let [x, y] = skewed.layer(0).expect("layer");
    assert_eq!(
        (x.tiles(), y.tiles()),
        (&[(33, 32), (4, 4)][..], &[(23, 23)][..])
    );
}

/// Every sample is biased by 1/256 and clamped to `[0.01, 1]`, `NaN` reading 1, as Apple's
/// compile does: the physical width Apple M3 gave each sample alone over 8192 pixels.
#[test]
fn samples_are_biased_and_clamped_as_apples() {
    let measured: [(f32, u32); 14] = [
        (1.0 / 128.0, 97),
        (1.0 / 256.0, 92),
        (1e-6, 92),
        (0.0, 92),
        (-0.5, 92),
        (-0.0, 92),
        (1.5, 8192),
        (2.0, 8192),
        (f32::INFINITY, 8192),
        (f32::NEG_INFINITY, 92),
        (f32::NAN, 8192),
        (f32::from_bits(1), 92),
        (0.015625, 161),
        (0.015_624_999, 161),
    ];
    for (q, physical) in measured {
        let m = map([8192, 32], &[layer(&[q], &[1.0])]);
        assert_eq!(m.physical_size(0), Some([physical, 32]), "{q}");
    }
    // Quality 0 is 0.01: tiles of 3200 screen pixels, and the last 1792 to 28.
    let m = map([8192, 32], &[layer(&[0.0], &[1.0])]);
    assert_eq!(
        m.layer(0).expect("layer")[0].tiles(),
        run(2, (3200, 32), (1792, 28))
    );
    // 1/3 as an f32 is 0.333333343, and 0.25: tiles of 94 and 126.
    let m = map([960, 960], &[layer(&[1.0 / 3.0], &[0.25])]);
    let [x, y] = m.layer(0).expect("layer");
    assert_eq!(x.tiles(), run(10, (94, 32), (20, 16)));
    assert_eq!(y.tiles()[0], (126, 32));
}

/// Descriptors Apple M3 compiled (the probe, macOS 26.7): the compile reproduces every tile, on
/// maps of one and two samples, piecewise steps, many random samples, two layers (which share their
/// vertical axis), the screen limit, and two descriptors whose tile ends on an exact `f32` tie.
#[test]
fn the_compile_reproduces_apples_measured_maps() {
    type Expected = Vec<[Vec<(u32, u32)>; 2]>;
    let one = |h: Vec<(u32, u32)>, v: Vec<(u32, u32)>| -> Expected { vec![[h, v]] };
    let flat = |w: u32| -> Vec<(u32, u32)> {
        let mut t = vec![(32, 32); (w / 32) as usize];
        if !w.is_multiple_of(32) {
            t.push((w % 32, w % 32));
        }
        t
    };
    let cases: Vec<(&str, [u32; 2], Vec<RateMapLayerDescriptor>, Expected)> = vec![
        (
            "0.5 over 8192",
            [8192, 32],
            vec![layer(&[0.5], &[1.0])],
            one(run(130, (63, 32), (2, 2)), flat(32)),
        ),
        (
            "1/32 over 256",
            [256, 32],
            vec![layer(&[0.03125], &[1.0])],
            one(vec![(256, 16)], flat(32)),
        ),
        (
            "1/64 over 33",
            [33, 32],
            vec![layer(&[0.015625], &[1.0])],
            one(vec![(33, 24)], flat(32)),
        ),
        (
            "0.5 to 1 over 1024",
            [1024, 32],
            vec![layer(&[0.5, 1.0], &[1.0])],
            one(
                tiles(
                    &[
                        60, 57, 54, 52, 50, 48, 46, 45, 44, 42, 41, 40, 39, 38, 38, 37, 36, 35, 35,
                        34, 33, 33, 32, 32,
                    ],
                    (23, 23),
                ),
                flat(32),
            ),
        ),
        (
            "1 to 0.5 over 1024",
            [1024, 32],
            vec![layer(&[1.0, 0.5], &[1.0])],
            one(
                tiles(
                    &[
                        32, 32, 33, 33, 34, 34, 35, 36, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                        49, 51, 53, 55, 58,
                    ],
                    (36, 24),
                ),
                flat(32),
            ),
        ),
        (
            "0.5 to 0.3 over 256, a tie at 80",
            [256, 32],
            vec![layer(&[0.5, 0.3], &[1.0])],
            one(tiles(&[63, 70, 80], (43, 16)), flat(32)),
        ),
        (
            "0.6 to 0.125 over 1024, a tie at 64",
            [1024, 32],
            vec![layer(&[0.6, 0.125], &[1.0])],
            one(
                tiles(
                    &[52, 55, 57, 60, 64, 68, 72, 78, 86, 97, 112, 137],
                    (86, 16),
                ),
                flat(32),
            ),
        ),
        (
            "steps 1, 0.25 six times, 1 over 1000",
            [1000, 32],
            vec![layer(
                &[1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0],
                &[1.0],
            )],
            one(
                tiles(
                    &[32, 38, 50, 85, 126, 126, 126, 126, 126, 66, 44, 35],
                    (20, 20),
                ),
                flat(32),
            ),
        ),
        (
            "0.3 to 0.9 by 0.1 to 0.7 over 1000x333",
            [1000, 333],
            vec![layer(&[0.3, 0.9], &[0.1, 0.7])],
            one(
                tiles(
                    &[
                        89, 79, 71, 65, 61, 57, 54, 51, 49, 47, 45, 43, 42, 41, 39, 38, 37, 36, 35,
                    ],
                    (21, 21),
                ),
                tiles(&[107, 74, 59, 50], (43, 32)),
            ),
        ),
        (
            "random samples over 257x1441",
            [257, 1441],
            vec![layer(
                &[0.5, 0.240_902_42, 0.447_834_43, 1.0, 0.0],
                &[
                    0.75,
                    0.186_202_47,
                    0.824_300_95,
                    0.25,
                    0.658_140_06,
                    0.351_100_33,
                    0.115_004_36,
                ],
            )],
            one(
                tiles(&[63, 67, 39, 32, 36], (20, 16)),
                tiles(
                    &[
                        42, 48, 58, 78, 84, 59, 48, 41, 38, 40, 46, 54, 70, 85, 65, 55, 48, 48, 53,
                        59, 69, 86, 113,
                    ],
                    (54, 27),
                ),
            ),
        ),
        (
            "two layers over 1000x1000",
            [1000, 1000],
            vec![layer(&[0.9, 0.2], &[0.6]), layer(&[0.2, 0.9], &[0.3])],
            {
                let v = run(19, (52, 32), (12, 8));
                vec![
                    [
                        tiles(
                            &[
                                35, 36, 37, 38, 39, 41, 42, 44, 46, 48, 51, 54, 58, 63, 69, 77, 89,
                                107,
                            ],
                            (26, 16),
                        ),
                        v.clone(),
                    ],
                    [
                        tiles(
                            &[
                                113, 92, 79, 70, 64, 59, 55, 52, 49, 47, 44, 43, 41, 40, 38, 37,
                                36, 35,
                            ],
                            (6, 6),
                        ),
                        v,
                    ],
                ]
            },
        ),
        (
            "0.5 over 16384",
            [16384, 32],
            vec![layer(&[0.5], &[1.0])],
            one(run(260, (63, 32), (4, 4)), flat(32)),
        ),
        (
            "1 over 32x16384",
            [32, 16384],
            vec![layer(&[1.0], &[1.0])],
            one(flat(32), flat(16384)),
        ),
    ];
    for (what, screen, layers, expected) in cases {
        let m = map(screen, &layers);
        for (l, axes) in expected.iter().enumerate() {
            let got = m.layer(l as u32).expect("layer");
            for a in 0..2 {
                assert_eq!(
                    got[a].tiles(),
                    axes[a].as_slice(),
                    "{what}: layer {l} axis {a}"
                );
            }
        }
    }
}

/// Pseudo-random descriptors inside the measured bounds: tiles cover the screen, every tile but
/// the last is 32 physical pixels and the last at most 32 and at most its screen pixels, and
/// every tile's quality covers the function at each integer coordinate it spans.
#[test]
fn compiled_tiles_cover_the_function_at_every_pixel() {
    let mut seed = 0x2545_f491_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed >> 8
    };
    for _ in 0..60 {
        let w = 8 + next() % 3000;
        let n = 1 + (next() % 9) as usize;
        let samples: Vec<f32> = (0..n).map(|_| (next() % 1001) as f32 / 1000.0).collect();
        let m = map([w, 7], &[layer(&samples, &[1.0])]);
        let axis = &m.layer(0).expect("layer")[0];
        assert_eq!(axis.screen_extent(), w);
        let q = Quality::new(w, &samples);
        let tiles = axis.tiles();
        let mut s = 0u32;
        for (k, &(se, pe)) in tiles.iter().enumerate() {
            let last = k + 1 == tiles.len();
            assert!(pe <= se && pe <= 32, "{samples:?} tile {k}");
            if !last {
                assert_eq!(pe, 32);
                for x in s..=s + se {
                    assert!(
                        se as f32 <= (32.0 / q.at(x)).floor(),
                        "{samples:?} tile {k} at {x}"
                    );
                }
            }
            s += se;
        }
    }
}

/// The words of a small map: its horizontal tables at layer 0's, its vertical ones, and zeros
/// elsewhere (Apple's layout).
#[test]
fn parameter_data_layout() {
    let m = map([40, 32], &[layer(&[0.5], &[1.0])]);
    // Horizontal: 40 pixels at half quality are one last tile, of 32 physical pixels by Apple's
    // rule (no ratio of sixteenths fits); vertical: the identity over 32 pixels.
    assert_eq!(m.layer(0).expect("layer")[0].tiles(), [(40, 32)]);
    let x = format::tile_line(0, 0, 40, 32);
    let y = format::tile_line(0, 0, 32, 32);
    let to_screen =
        |l: format::TileLine| [l.reciprocal.to_bits(), (l.offset as f32 / 256.0).to_bits()];
    let to_physical = |l: format::TileLine| [l.scale.to_bits(), ((l.offset as u32) << 8) | 0xff];
    let mut want = vec![0u32; format::WORDS as usize];
    // `⌈32/32⌉ + 1` physical-to-screen entries per axis; `⌈40/32⌉ + 2` and `⌈32/32⌉ + 2`
    // screen-to-physical ones.
    for (at, words) in [
        (0, to_screen(x)),
        (2, to_screen(x)),
        (4104, to_screen(y)),
        (4106, to_screen(y)),
    ] {
        want[at..at + 2].copy_from_slice(&words);
    }
    for j in 0..4 {
        want[2052 + 2 * j..2054 + 2 * j].copy_from_slice(&to_physical(x));
    }
    for j in 0..3 {
        want[5130 + 2 * j..5132 + 2 * j].copy_from_slice(&to_physical(y));
    }
    assert_eq!(m.words(), want);
    assert_eq!(m.parameter_data_size(), 24_624);
    assert_eq!(m.to_bytes().len() as u64, m.parameter_data_size());
    assert_eq!(m.physical_size(0), Some([32, 32]));
}

/// A descriptor of no layers compiles to a map of none: no horizontal axis, and a vertical one of
/// the least quality everywhere (256 screen rows to 16 physical ones).
#[test]
fn a_map_of_no_layers_has_a_vertical_axis() {
    let m = map([256, 256], &[]);
    assert_eq!(m.layer_count(), 0);
    assert_eq!(m.physical_size(0), None);
    assert_eq!(m.physical_granularity(), [32, 32]);
    assert_eq!(m.vertical().tiles(), [(256, 16)]);
    assert!(m.words()[..4104].iter().all(|&w| w == 0));
    assert_eq!(m.to_bytes().len(), 24_624);
}

/// One axis of a map Apple M3 compiled: its name, screen extent, tiles as `(screen pixels,
/// physical pixels)`, and `(x, screen to physical, physical to screen)` of Apple's CPU map.
type MeasuredAxis = (
    &'static str,
    u32,
    Vec<(u32, u32)>,
    &'static [(f32, f32, f32)],
);

/// Apple's maps from Apple M3 (`MTLRasterizationRateMap`, macOS 26.7): their tiles, read from the
/// CPU map at each 32-pixel physical boundary, and `(x, screen to physical, physical to screen)`
/// of the CPU map on the horizontal axis, to four decimals. Rebuilt from those tiles, this
/// crate's map answers Apple's to those decimals: a line per tile in Apple's fixed point, and the
/// first and last tiles' lines past the screen.
#[test]
fn apples_tiles_map_as_apple_maps_them() {
    let cases: [MeasuredAxis; 3] = [
        (
            "quality 1, 0.5, 1 over 1024",
            1024,
            [
                32, 33, 34, 35, 36, 38, 40, 42, 44, 47, 50, 54, 59, 54, 50, 46, 44, 41, 39, 38, 36,
                35, 34, 32,
            ]
            .iter()
            .map(|&s| (s, 32))
            .chain([(31, 31)])
            .collect(),
            &[
                (-2.0, -2.0, -2.0),
                (0.0, 0.0, 0.0),
                (31.5, 31.5, 31.5),
                (46.5, 46.0597, 46.9541),
                (95.0, 92.236, 97.9368),
                (192.0, 178.528, 207.9981),
                (386.0, 323.2002, 488.6874),
                (500.25, 392.2712, 677.1074),
                (777.5, 574.8292, 1002.5),
                (990.5, 765.5, 1215.5),
                (999.75, 774.75, 1224.75),
                (1001.25, 776.25, 1226.25),
                (1024.0, 799.0, 1249.0),
                (1025.5, 800.5, 1250.5),
            ],
        ),
        (
            "quality 0.3, 1, 0.7 over 1000",
            1000,
            [
                77, 63, 55, 49, 45, 41, 39, 36, 34, 33, 32, 32, 32, 33, 34, 34, 35, 36, 37, 38, 39,
                40, 41, 43,
            ]
            .iter()
            .map(|&s| (s, 32))
            .chain([(22, 16)])
            .collect(),
            &[
                (-2.0, -0.8312, -4.8125),
                (0.0, 0.0, 0.0),
                (31.5, 13.0909, 75.7969),
                (46.5, 19.3247, 105.5435),
                (95.0, 41.1446, 193.2788),
                (192.0, 94.256, 330.0004),
                (386.0, 239.1111, 538.0),
                (500.25, 348.25, 656.5171),
                (777.5, 608.4205, 991.0635),
                (990.5, 777.0902, 1283.9385),
                (999.75, 783.8175, 1296.6572),
                (1001.25, 784.9084, 1298.7197),
            ],
        ),
        (
            "quality 0.5 over 1024",
            1024,
            std::iter::repeat_n((63, 32), 16)
                .chain([(16, 16)])
                .collect(),
            &[
                (-2.0, -1.0159, -3.9375),
                (0.0, 0.0, 0.0),
                (31.5, 16.0, 62.0156),
                (46.5, 23.6191, 91.5469),
                (95.0, 48.254, 187.0312),
                (192.0, 97.5238, 378.0),
                (386.0, 196.0635, 759.9374),
                (500.25, 254.0952, 984.8671),
                (777.5, 394.9207, 1273.5),
                (990.5, 503.1111, 1486.5),
                (999.75, 507.8095, 1495.75),
                (1001.25, 508.5715, 1497.25),
                (1024.0, 528.0, 1520.0),
                (1025.5, 529.5, 1521.5),
            ],
        ),
    ];
    // Apple's compile of the three descriptors is these tiles.
    for ((what, w, tiles, _), samples) in
        cases
            .iter()
            .zip([&[1.0, 0.5, 1.0][..], &[0.3, 1.0, 0.7][..], &[0.5][..]])
    {
        let m = map([*w, 32], &[layer(samples, &[1.0])]);
        assert_eq!(
            m.layer(0).expect("layer")[0].tiles(),
            tiles.as_slice(),
            "{what}"
        );
    }
    for (what, w, tiles, points) in cases {
        let x = RateMapAxis::from_tiles(32, &tiles).expect("tiles");
        let y = RateMapAxis::from_tiles(32, &[(32, 32)]).expect("tiles");
        assert_eq!(x.screen_extent(), w, "{what}");
        let m = RasterizationRateMap::from_layers([w, 32], vec![[x, y]]).expect("map");
        for &(c, to_physical, to_screen) in points {
            let p = m.screen_to_physical([c, 0.0], 0)[0];
            let s = m.physical_to_screen([c, 0.0], 0)[0];
            assert!(
                (p - to_physical).abs() < 1.5e-4,
                "{what}: screen {c} -> {p}, Apple {to_physical}"
            );
            assert!(
                (s - to_screen).abs() < 1.5e-4,
                "{what}: physical {c} -> {s}, Apple {to_screen}"
            );
        }
    }
}

/// Each layer maps by its own horizontal tiles; the layers share one vertical axis, which
/// follows the larger of their vertical functions, as Apple's compile does; and a layer past the
/// map's reads the last.
#[test]
fn layers_share_the_vertical_axis() {
    let m = map(
        [800, 600],
        &[layer(&[1.0, 0.5], &[0.5, 1.0]), layer(&[0.25], &[1.0])],
    );
    assert_eq!(m.layer_count(), 2);
    let [x0, y0] = m.layer(0).expect("layer");
    let [x1, y1] = m.layer(1).expect("layer");
    assert_ne!(x0, x1);
    assert_eq!(y0, y1);
    // Layer 1's vertical quality is 1 everywhere, so the shared axis is the identity.
    assert_eq!(y0.physical_extent(), 600);
    let c = [700.0, 100.0];
    let l0 = m.screen_to_physical(c, 0);
    let l1 = m.screen_to_physical(c, 1);
    assert_ne!(l0, l1);
    // Layer 1 at 0.25: tiles of 126, so 700 is 70 into the sixth, from physical 160, on that
    // tile's line in Apple's fixed point.
    let line = format::tile_line(630, 160, 126, 32);
    assert_eq!(
        l1,
        [
            700.0f32.mul_add(line.scale, line.offset as f32 / 256.0),
            100.0
        ]
    );
    assert_eq!(m.screen_to_physical(c, 7), l1);
    assert_eq!(m.physical_to_screen(l1, 9), m.physical_to_screen(l1, 1));
}

/// A shader's screen-to-physical answer reads the next tile's line in the half pixel before a
/// tile that starts at a multiple of 32, as Apple's GPU decoder does, where the map's does not;
/// each evaluates a tile's line as Apple's does, the map with one rounding and a shader with two.
#[test]
fn a_shader_reads_the_next_tiles_line_before_a_boundary_at_a_multiple_of_32() {
    // Tiles of 64, 40, and 50 screen pixels: boundaries at 64 (a multiple of 32) and at 104 (not).
    let x = RateMapAxis::from_tiles(32, &[(64, 32), (40, 32), (50, 32), (6, 6)]).expect("tiles");
    let y = RateMapAxis::from_tiles(32, &[(64, 32), (36, 32)]).expect("tiles");
    let m = RasterizationRateMap::from_layers([160, 100], vec![[x, y]]).expect("map");
    // Tile 0's line, tile 1's (from screen 64 at physical 32), and tile 2's; and the vertical
    // axis's second.
    let lines = [
        format::tile_line(0, 0, 64, 32),
        format::tile_line(64, 32, 40, 32),
        format::tile_line(104, 64, 50, 32),
        format::tile_line(64, 32, 36, 32),
    ];
    let unit = |l: format::TileLine| l.offset as f32 / 256.0;
    let map_at = |k: usize, s: f32| s.mul_add(lines[k].scale, unit(lines[k]));
    let shader_at = |k: usize, s: f32| s * lines[k].scale + unit(lines[k]);
    let below_64 = 64.0f32.next_down();
    let below_half = 63.5f32.next_down();
    for s in [63.5, 63.75, below_64] {
        assert_eq!(
            m.shader_screen_to_physical([s, 0.0], 0)[0],
            shader_at(1, s),
            "{s}"
        );
        assert_eq!(m.screen_to_physical([s, 0.0], 0)[0], map_at(0, s), "{s}");
        // The vertical axis's boundary at 64 too.
        assert_eq!(
            m.shader_screen_to_physical([0.0, s], 0)[1],
            shader_at(3, s),
            "{s}"
        );
        assert_eq!(m.screen_to_physical([0.0, s], 0)[1], map_at(0, s), "{s}");
    }
    for s in [0.0, 31.5, 32.0, 63.0, below_half, 63.25] {
        assert_eq!(
            m.shader_screen_to_physical([s, 0.0], 0)[0],
            shader_at(0, s),
            "{s}"
        );
        assert_eq!(m.screen_to_physical([s, 0.0], 0)[0], map_at(0, s), "{s}");
    }
    // At 104 (not a multiple of 32) both read the line of the tile that holds the coordinate.
    for s in [103.0, 103.5, 103.75] {
        assert_eq!(
            m.shader_screen_to_physical([s, 0.0], 0)[0],
            shader_at(1, s),
            "{s}"
        );
        assert_eq!(m.screen_to_physical([s, 0.0], 0)[0], map_at(1, s), "{s}");
    }
    assert_eq!(
        m.shader_screen_to_physical([104.0, 0.0], 0)[0],
        shader_at(2, 104.0)
    );
}

/// Physical to screen undoes screen to physical, up to the lines' 1/256-pixel offsets, which
/// differ between neighbouring tiles at their boundary by up to 1/256 physical pixels.
#[test]
fn the_two_maps_are_inverse() {
    let m = map(
        [1920, 1080],
        &[layer(
            &[0.1, 0.4, 1.0, 0.7, 0.2],
            &[1.0, 0.9, 0.3, 0.3, 0.8, 1.0],
        )],
    );
    let [pw, ph] = m.physical_size(0).expect("size");
    assert!(pw < 1920 && ph < 1080);
    for i in 0..400 {
        let c = [i as f32 * 4.8 + 0.37, i as f32 * 2.7 + 0.11];
        let back = m.physical_to_screen(m.screen_to_physical(c, 0), 0);
        assert!((back[0] - c[0]).abs() < 0.05, "{c:?} -> {back:?}");
        assert!((back[1] - c[1]).abs() < 0.05, "{c:?} -> {back:?}");
    }
    let edge = m.screen_to_physical([1920.0, 1080.0], 0);
    assert!((edge[0] - pw as f32).abs() < 0.01 && (edge[1] - ph as f32).abs() < 0.01);
}

/// Malformed descriptors and tiles, and tiles Metal's parameter data cannot hold, are refused.
#[test]
fn malformed_maps_are_refused() {
    assert_eq!(
        RasterizationRateMap::from_layers([16, 16], Vec::new()),
        Err(RateMapError::NoLayers)
    );
    let mut d = RateMapDescriptor::new([70000, 16]);
    d.layers.push(layer(&[1.0], &[1.0]));
    assert_eq!(
        RasterizationRateMap::new(&d),
        Err(RateMapError::ScreenExtent { extent: 70000 })
    );
    let mut d = RateMapDescriptor::new([16, 16]);
    d.layers.push(layer(&[1.0], &[]));
    assert_eq!(
        RasterizationRateMap::new(&d),
        Err(RateMapError::NoSamples { layer: 0, axis: 1 })
    );
    for granularity in [24, 16, 64] {
        assert_eq!(
            RateMapAxis::from_tiles(granularity, &[(16, 16)]),
            Err(RateMapError::Granularity { granularity })
        );
    }
    assert_eq!(
        RateMapAxis::from_tiles(32, &[]),
        Err(RateMapError::Tile { tile: 0 })
    );
    assert_eq!(
        RateMapAxis::from_tiles(32, &[(40, 16), (8, 8)]),
        Err(RateMapError::Tile { tile: 0 }),
        "a tile before the last is the granularity wide"
    );
    assert_eq!(
        RateMapAxis::from_tiles(32, &[(32, 32), (8, 9)]),
        Err(RateMapError::Tile { tile: 1 }),
        "quality above 1:1"
    );
    // 16384 screen pixels fit the tables; one more does not.
    let wide = vec![(32, 32); 512];
    assert!(RateMapAxis::from_tiles(32, &wide).is_ok());
    let wider = [wide.as_slice(), &[(1, 1)]].concat();
    assert_eq!(
        RateMapAxis::from_tiles(32, &wider),
        Err(RateMapError::ScreenExtent { extent: 16385 })
    );
    let x = RateMapAxis::from_tiles(32, &[(32, 32)]).expect("tiles");
    let y = RateMapAxis::from_tiles(32, &[(16, 16)]).expect("tiles");
    let y2 = RateMapAxis::from_tiles(32, &[(16, 8)]).expect("tiles");
    assert_eq!(
        RasterizationRateMap::from_layers([33, 16], vec![[x.clone(), y.clone()]]),
        Err(RateMapError::ExtentMismatch { layer: 0, axis: 0 })
    );
    assert_eq!(
        RasterizationRateMap::from_layers([32, 16], vec![[x.clone(), y.clone()], [x.clone(), y2]]),
        Err(RateMapError::VerticalAxis { layer: 1 })
    );
    let three = vec![[x.clone(), y.clone()], [x.clone(), y.clone()], [x, y]];
    assert_eq!(
        RasterizationRateMap::from_layers([32, 16], three),
        Err(RateMapError::LayerCount { count: 3 })
    );
}

/// A tile whose line's offset lies halfway between two of Metal's 1/256-pixel steps is refused:
/// how Apple rounds it is not measured, and its compile makes none.
#[test]
fn an_offset_tie_is_refused() {
    // Tiles of 64 and 32 screen pixels, then 512 to 8 physical ones from screen 96 and physical
    // 64: 256 · (64 · 512 - 96 · 8) / 512 = 16000, no tie.
    assert!(RateMapAxis::from_tiles(32, &[(64, 32), (32, 32), (512, 8)]).is_ok());
    let mut tiles = vec![(32, 32); 3];
    // From screen 96 and physical 96, 513 screen pixels to 1 physical one: no tie either.
    tiles.push((513, 1));
    assert!(RateMapAxis::from_tiles(32, &tiles).is_ok());
    // Screen 33 to physical 32 at the start of a 1024-pixel tile of 2: 256 · (32 · 1024 - 33 ·
    // 2) / 1024 = 8175.5, a tie.
    assert!(format::offset_tie(33, 32, 1024, 2));
    assert_eq!(
        RateMapAxis::from_tiles(32, &[(33, 32), (1024, 2)]),
        Err(RateMapError::OffsetTie { tile: 1 })
    );
}

/// Outside the bounds Apple's compile is measured in, the compile refuses rather than guess, and
/// just inside them it compiles.
#[test]
fn unmeasured_descriptors_are_refused() {
    let new = |screen: [u32; 2], layers: &[RateMapLayerDescriptor]| {
        let mut d = RateMapDescriptor::new(screen);
        d.layers = layers.to_vec();
        RasterizationRateMap::new(&d)
    };
    let unpinned = |reason| Err(RateMapError::Unpinned { reason });
    let one = [layer(&[1.0], &[1.0])];
    assert_eq!(
        new([16385, 32], &one),
        unpinned(UnpinnedCompile::ScreenExtent { extent: 16385 })
    );
    assert_eq!(
        new([32, 32768], &one),
        unpinned(UnpinnedCompile::ScreenExtent { extent: 32768 })
    );
    assert!(new([16384, 16384], &one).is_ok());
    let l = layer(&[0.5], &[0.5]);
    assert_eq!(
        new([64, 64], &[l.clone(), l.clone(), l.clone()]),
        unpinned(UnpinnedCompile::LayerCount { count: 3 })
    );
    assert!(new([64, 64], &[l.clone(), l]).is_ok());
    // Nine samples over three pixels; four over three compile.
    let nine = [0.5, 0.1, 1.0, 0.25, 0.9, 0.25, 0.4, 0.2, 0.3];
    assert_eq!(
        new([32, 3], &[layer(&[1.0], &nine)]),
        unpinned(UnpinnedCompile::DenseSamples {
            layer: 0,
            axis: 1,
            samples: 9,
            extent: 3
        })
    );
    assert!(new([32, 3], &[layer(&[1.0], &nine[..4])]).is_ok());
    // Samples that are not all equal: at most 9 on a map of one layer and 4 on a map of two were
    // measured. Equal samples are one quality at any count (64 were measured).
    let ten = [0.5, 0.1, 1.0, 0.25, 0.9, 0.25, 0.4, 0.2, 0.3, 0.7];
    assert_eq!(
        new([64, 64], &[layer(&ten, &[1.0])]),
        unpinned(UnpinnedCompile::SampleCount {
            layer: 0,
            axis: 0,
            samples: 10,
            max: 9
        })
    );
    assert!(new([64, 64], &[layer(&ten[..9], &[1.0])]).is_ok());
    assert!(new([1000, 64], &[layer(&[0.5; 64], &[1.0])]).is_ok());
    let flat = layer(&[0.5], &[0.5]);
    assert_eq!(
        new([64, 64], &[flat.clone(), layer(&[1.0], &ten[..5])]),
        unpinned(UnpinnedCompile::SampleCount {
            layer: 1,
            axis: 1,
            samples: 5,
            max: 4
        })
    );
    assert!(new([64, 64], &[flat.clone(), layer(&[1.0], &ten[..4])]).is_ok());
    assert!(new([64, 64], &[flat, layer(&[1.0], &[0.25; 9])]).is_ok());
    // An out-of-range sample beside another; alone, it compiles as Apple's does.
    assert_eq!(
        new([64, 64], &[layer(&[0.5, f32::NAN], &[1.0])]),
        unpinned(UnpinnedCompile::SampleOutOfRange {
            layer: 0,
            axis: 0,
            index: 1
        })
    );
    assert_eq!(
        new([64, 64], &[layer(&[1.0], &[-0.5, 0.5])]),
        unpinned(UnpinnedCompile::SampleOutOfRange {
            layer: 0,
            axis: 1,
            index: 0
        })
    );
    assert!(new([64, 64], &[layer(&[f32::NAN], &[-0.5])]).is_ok());
    assert!(RateMapError::Unpinned {
        reason: UnpinnedCompile::LayerCount { count: 3 }
    }
    .to_string()
    .contains("3 layers"));
}

/// FNV-1a, 64 bits, of `bytes`: a short name for Metal's measured parameter data.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// An axis of no screen pixels has one tile of no screen and no physical pixels, whatever its
/// quality samples, so the layer's physical extent on it is 0 and the map compiles (Metal's API
/// validation stops the descriptor; plain Metal compiles it, and `apple_tests` holds its bytes).
/// Its physical-to-screen line is the least step the fixed point holds.
#[test]
fn an_axis_of_no_screen_pixels_has_one_empty_tile() {
    for samples in [
        &[0.5][..],
        &[1.0],
        &[0.01],
        &[0.0],
        &[0.5, 1.0],
        &[0.25, 0.75, 0.25],
        &[f32::NAN],
        &[-3.0, 7.0],
    ] {
        let m = map([0, 64], &[layer(samples, &[0.5])]);
        assert_eq!(m.physical_size(0), Some([0, 33]), "{samples:?}");
        let axes = m.layer(0).expect("layer");
        assert_eq!(axes[0].tiles(), [(0, 0)], "{samples:?}");
        assert_eq!(axes[0].screen_extent(), 0, "{samples:?}");
        assert_eq!(axes[0].physical_extent(), 0, "{samples:?}");
        // Every sample set gives the one degenerate axis, so the bytes are the same.
        assert_eq!(
            m.to_bytes(),
            map([0, 64], &[layer(&[0.5], &[0.5])]).to_bytes(),
            "{samples:?}"
        );
    }
    // The least step: scale 2^-24, its reciprocal 2^24, offset 0, in two entries of each table.
    let m = map([0, 64], &[layer(&[0.5], &[0.5])]);
    let words = m.words();
    let to_screen = format::HORIZONTAL_TO_SCREEN as usize;
    let to_physical = format::HORIZONTAL_TO_PHYSICAL as usize;
    for entry in 0..2 {
        assert_eq!(
            words[to_screen + 2 * entry..to_screen + 2 * entry + 2],
            [(16_777_216.0f32).to_bits(), 0],
            "physical to screen {entry}"
        );
        assert_eq!(
            words[to_physical + 2 * entry..to_physical + 2 * entry + 2],
            [(1.0f32 / 16_777_216.0).to_bits(), format::NO_BOUNDARY],
            "screen to physical {entry}"
        );
    }
    assert!(
        words[to_screen + 4..to_screen + format::TABLE_WORDS as usize]
            .iter()
            .all(|&w| w == 0),
        "no entry past the axis's two"
    );
    // Both axes alike, and a map of no screen pixels at all.
    assert_eq!(
        map([64, 0], &[layer(&[0.5], &[0.5])]).physical_size(0),
        Some([33, 0])
    );
    assert_eq!(
        map([0, 0], &[layer(&[0.5], &[0.5])]).physical_size(0),
        Some([0, 0])
    );
}
