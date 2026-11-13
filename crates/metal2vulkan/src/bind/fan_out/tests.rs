use super::*;
use crate::bind::rate_map::{
    RasterizationRateMap, RateMapDescriptor, RateMapLayerDescriptor, RegionLimits,
};

/// A map over `screen` whose quality falls from 1 at the centre to 0.3 at the edges on both
/// axes, planned for a viewport over the whole screen: each region's Vulkan viewport, y flipped
/// as the runtime draws it (`y + height`, `-height`), and its scissor; and the target's viewport.
fn falling(screen: [u32; 2]) -> (Vec<([f32; 4], [u32; 4])>, [f32; 4]) {
    let quality = vec![0.3, 0.6, 1.0, 0.6, 0.3];
    let mut desc = RateMapDescriptor::new(screen);
    desc.layers
        .push(RateMapLayerDescriptor::new(quality.clone(), quality));
    let map = RasterizationRateMap::new(&desc).expect("map");
    let limits = RegionLimits::new([16384, 16384], [-32768.0, 32767.0]);
    let whole = [0.0, 0.0, f64::from(screen[0]), f64::from(screen[1])];
    let plan = map.regions(0, whole, None, &limits).expect("plan");
    let regions = (plan.regions().iter())
        .map(|r| {
            let [x, y, w, h] = r.viewport();
            let [sx, sy] = r.scissor_offset();
            let [sw, sh] = r.scissor_extent();
            ([x, y + h, w, -h], [sx, sy, sw, sh])
        })
        .filter(|(_, s)| s[2] > 0 && s[3] > 0)
        .collect();
    let [pw, ph] = plan.physical_size().map(|v| v as f32);
    (regions, [0.0, ph, pw, -ph])
}

/// The table's entry `e`, as floats.
fn entry(table: &[u8], e: u32) -> [f32; 4] {
    let at = e as usize * 16;
    [0, 1, 2, 3].map(|i| {
        let word = &table[at + 4 * i..at + 4 * i + 4];
        f32::from_le_bytes(word.try_into().unwrap())
    })
}

/// The segment counts of the table.
fn counts(table: &[u8]) -> [u32; 2] {
    let word = |i: usize| u32::from_le_bytes(table[4 * i..4 * i + 4].try_into().unwrap());
    [word(0), word(1)]
}

/// The last segment of axis `a` whose least coordinate is at most `v`, or the first: the stage's
/// bisection.
fn find(table: &[u8], a: u32, v: f32) -> u32 {
    let (mut lo, mut hi) = (0, counts(table)[a as usize] - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        match entry(table, line_entry(a, mid))[0] <= v {
            true => lo = mid,
            false => hi = mid - 1,
        }
    }
    lo
}

/// A small deterministic generator of floats in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Through each region's line in the table, a clip-space point lands at the window coordinate
/// the region's own viewport maps it to, and its clip distances hold it within the region's
/// scissor exactly where that coordinate lies inside.
#[test]
fn each_copy_lands_where_its_region_draws_and_clips_to_its_pixels() {
    let (regions, target) = falling([3648, 1734]);
    let table = table(target, &regions).expect("table");
    assert_eq!(table.len() as u64, TABLE_BYTES);
    let [nx, ny] = counts(&table);
    assert_eq!(nx as usize * ny as usize, regions.len());
    let mut random = Lcg(7);
    for _ in 0..2000 {
        let w = 0.1 + 4.0 * random.next();
        let point = [0, 1].map(|_| (2.4 * random.next() - 1.2) * w);
        for a in 0..2u32 {
            let au = a as usize;
            for k in 0..counts(&table)[au] {
                let [_, _, sa, sb] = entry(&table, line_entry(a, k));
                let [half, lo, hi, _] = entry(&table, clip_entry(a, k));
                // The segment's region: its viewport on the axis from its line.
                let (origin, extent, pixels) = (regions.iter())
                    .map(|&(v, s)| (v[au], v[au + 2], [s[au], s[au] + s[au + 2]]))
                    .find(|&(x, ex, _)| {
                        let a_of = f64::from(ex) / f64::from(target[au + 2]);
                        (a_of - f64::from(sa)).abs() < 1e-6
                            && ((2.0 * f64::from(x) + f64::from(ex) - 2.0 * f64::from(target[au]))
                                / f64::from(target[au + 2])
                                - 1.0
                                - f64::from(sb))
                            .abs()
                                < 1e-5
                    })
                    .expect("the segment's region");
                let x = point[au];
                let window = f64::from(origin) + (x / w + 1.0) * f64::from(extent) / 2.0;
                let moved = f64::from(sa) * x + f64::from(sb) * w;
                let through =
                    f64::from(target[au]) + (moved / w + 1.0) * f64::from(target[au + 2]) / 2.0;
                assert!(
                    (window - through).abs() < 2e-3,
                    "axis {a} segment {k}: {window} through its viewport, {through} through the table"
                );
                let d_lo = f64::from(half) * x + f64::from(lo) * w;
                let d_hi = -f64::from(half) * x + f64::from(hi) * w;
                let inside = window >= f64::from(pixels[0]) && window <= f64::from(pixels[1]);
                let margin = (window - f64::from(pixels[0]))
                    .abs()
                    .min((window - f64::from(pixels[1])).abs());
                if margin > 1e-2 {
                    assert_eq!(
                        d_lo >= 0.0 && d_hi >= 0.0,
                        inside,
                        "axis {a} segment {k}: window {window} in {pixels:?}"
                    );
                }
            }
        }
    }
}

/// The scissors of the table's regions, in its order, are those its lines and clips draw: region
/// `k`'s the pixels of x segment `k mod nx` and y segment `k / nx`; and they are the plan's
/// regions' scissors, each once.
#[test]
fn each_region_s_scissor_is_its_segments_pixels_in_the_table_s_order() {
    let (regions, target) = falling([3648, 1734]);
    let table = table(target, &regions).expect("table");
    let scissors = scissors(&regions).expect("scissors");
    let [nx, ny] = counts(&table);
    assert_eq!(scissors.len(), (nx * ny) as usize);
    // The pixels of segment `k` of axis `a`: those of the region whose viewport the line maps.
    let pixels = |a: u32, k: u32| {
        let au = a as usize;
        let [_, _, sa, sb] = entry(&table, line_entry(a, k));
        (regions.iter())
            .find(|&&(v, _)| {
                let (x, ex) = (f64::from(v[au]), f64::from(v[au + 2]));
                let (t0, tw) = (f64::from(target[au]), f64::from(target[au + 2]));
                (ex / tw - f64::from(sa)).abs() < 1e-6
                    && ((2.0 * x + ex - 2.0 * t0) / tw - 1.0 - f64::from(sb)).abs() < 1e-5
            })
            .map(|&(_, s)| [s[au], s[au] + s[au + 2]])
            .expect("the segment's region")
    };
    for (k, scissor) in (0u32..).zip(&scissors) {
        let ([x0, x1], [y0, y1]) = (pixels(0, k % nx), pixels(1, k / nx));
        assert_eq!(*scissor, [x0, y0, x1 - x0, y1 - y0], "region {k}");
    }
    let mut theirs: Vec<[u32; 4]> = regions.iter().map(|&(_, s)| s).collect();
    let mut ours = scissors;
    theirs.sort_unstable();
    ours.sort_unstable();
    assert_eq!(ours, theirs);
}

/// The segments the bisection finds for an interval of normalized device coordinates hold every
/// segment with a pixel centre inside it.
#[test]
fn the_bisection_reaches_every_segment_with_a_pixel_centre_inside_the_bounds() {
    for screen in [[3648, 1734], [1024, 1024], [100, 70]] {
        let (regions, target) = falling(screen);
        let table = table(target, &regions).expect("table");
        let mut random = Lcg(11);
        for _ in 0..500 {
            for a in 0..2u32 {
                let au = a as usize;
                let (p, q) = (2.2 * random.next() - 1.1, 2.2 * random.next() - 1.1);
                let span = 0.2 * random.next();
                let (lo, hi) = (p.min(q), p.min(q) + span.min((p - q).abs()));
                let (i0, i1) = (find(&table, a, lo as f32), find(&table, a, hi as f32));
                for &(viewport, scissor) in &regions {
                    let (origin, extent) = (f64::from(viewport[au]), f64::from(viewport[au + 2]));
                    let centres = (scissor[au]..scissor[au] + scissor[au + 2])
                        .map(|p| 2.0 * (f64::from(p) + 0.5 - origin) / extent - 1.0);
                    if !centres.clone().any(|n| n >= lo && n <= hi) {
                        continue;
                    }
                    let ndc_lo = centres.fold(f64::INFINITY, f64::min);
                    let k = (0..counts(&table)[au])
                        .find(|&k| {
                            let [l, h, _, _] = entry(&table, line_entry(a, k));
                            f64::from(l) <= ndc_lo && ndc_lo <= f64::from(h)
                        })
                        .expect("the region's segment");
                    assert!(
                        (i0..=i1).contains(&k),
                        "{screen:?} axis {a}: [{lo}, {hi}] reaches segment {k}, the bisection \
                         [{i0}, {i1}]"
                    );
                }
            }
        }
    }
}

/// Regions that are not one segment of each axis per pair, or none, take no table.
#[test]
fn regions_off_the_grid_take_no_table() {
    let (regions, target) = falling([1024, 1024]);
    assert!(table(target, &regions[1..]).is_none());
    assert!(table(target, &[]).is_none());
    assert!(scissors(&regions[1..]).is_none());
}
