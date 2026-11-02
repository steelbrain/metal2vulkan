//! A triangle's band ([`LEFT`], [`RIGHT`], [`EDGE_LEVELS`]) against the quad domain the Vulkan
//! specification tessellates at those levels: every triangle it may make either covers nothing
//! (two of its vertices at one point) or is one of the three triangles that fan each edge's
//! parallelogram from the middle of its first end, each made once; and every triangle that covers
//! anything winds as a line's band does where the stage keeps `n`, the other way where it turns
//! it, for triangles of every shape and direction.

use super::{Point, A, B, EDGE_LEVELS, LEFT, RIGHT};

/// A vertex of the quad domain: the `k`th of the `u = 0` edge (at `v = k / 11`), of the `u = 1`
/// edge (at `v = k / 6`), or inner vertex `I1` or `I2`.
#[derive(Clone, Copy, Debug)]
enum Vertex {
    Left(usize),
    Right(usize),
    Inner(usize),
}

impl Vertex {
    /// Its `(u, v)`.
    fn domain(self) -> [f64; 2] {
        match self {
            Vertex::Left(k) => [0.0, k as f64 / (LEFT.len() - 1) as f64],
            Vertex::Right(k) => [1.0, k as f64 / (RIGHT.len() - 1) as f64],
            Vertex::Inner(i) => [(i + 1) as f64 / 3.0, 0.5],
        }
    }

    /// The band's point the evaluation stage places it at.
    fn point(self) -> Point {
        match self {
            Vertex::Left(k) => LEFT[k],
            Vertex::Right(k) => RIGHT[k],
            Vertex::Inner(0) => A,
            Vertex::Inner(_) => B,
        }
    }
}

/// Every triangle the quad domain may be cut into at [`EDGE_LEVELS`] under equal spacing: its
/// inner levels `(3, 2)` leave the inner rectangle the segment `I1 I2`, so the `u = 0` edge's
/// vertices fan to `I1` and the `u = 1` edge's to `I2`; the `v = 0` and `v = 1` edges, one segment
/// each, meet the segment in two triangles along either diagonal, both listed.
fn domain_triangles() -> Vec<[Vertex; 3]> {
    let ([left, bottom, right, top], inner) = EDGE_LEVELS;
    assert_eq!((inner, bottom, top), ([3, 2], 1, 1));
    let (left, right) = (left as usize, right as usize);
    assert_eq!((left + 1, right + 1), (LEFT.len(), RIGHT.len()));
    let mut out = Vec::new();
    for k in 0..left {
        out.push([Vertex::Inner(0), Vertex::Left(k), Vertex::Left(k + 1)]);
    }
    for k in 0..right {
        out.push([Vertex::Inner(1), Vertex::Right(k), Vertex::Right(k + 1)]);
    }
    let (i1, i2) = (Vertex::Inner(0), Vertex::Inner(1));
    for (l, r) in [
        (Vertex::Left(0), Vertex::Right(0)),
        (Vertex::Left(left), Vertex::Right(right)),
    ] {
        out.extend([[l, r, i2], [l, i2, i1], [l, r, i1], [r, i2, i1]]);
    }
    out
}

/// Twice the signed area of the triangle `[p, q, r]`.
fn area([p, q, r]: [[f64; 2]; 3]) -> f64 {
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
}

/// The minor axis of the segment from `s` to `e` (the module docs).
fn minor_axis(s: [f64; 2], e: [f64; 2]) -> [f64; 2] {
    let (dx, dy) = (e[0] - s[0], e[1] - s[1]);
    match dx.abs() >= dy.abs() {
        true => [0.0, dx.signum()],
        false => [-dy.signum(), 0.0],
    }
}

/// `point` of the band of triangle `[a, b, c]` (framebuffer pixels) at half width `h`, `n` taken
/// as `facing n`.
fn place(point: Point, [a, b, c]: [[f64; 2]; 3], h: f64, facing: f64) -> [f64; 2] {
    let [s, e] = [[a, b], [a, c], [b, c]][point.edge as usize];
    let n = minor_axis(s, e);
    let at = if point.end { e } else { s };
    let across = f64::from(point.side) * h * facing;
    [at[0] + n[0] * across, at[1] + n[1] * across]
}

/// Each point's code decodes to the point, as the evaluation stage decodes it.
#[test]
fn each_point_decodes_from_its_code() {
    for &point in LEFT.iter().chain(&RIGHT) {
        let code = point.code();
        let rest = code % 6;
        let decoded = Point {
            edge: code / 6,
            end: rest >= 3,
            side: (rest % 3) as i32 - 1,
        };
        assert_eq!(decoded, point, "{code}");
    }
}

/// Every triangle the domain may be cut into is degenerate or one of the nine that fan the three
/// parallelograms, each made once.
#[test]
fn the_fans_cover_each_parallelogram_once() {
    let mut made: Vec<[Point; 3]> = Vec::new();
    for t in domain_triangles() {
        let [p, q, r] = t.map(Vertex::point);
        if p == q || q == r || p == r {
            continue;
        }
        made.push([p, q, r]);
    }
    let key = |t: &[Point; 3]| {
        let mut codes = t.map(Point::code);
        codes.sort();
        codes
    };
    let mut made: Vec<[u32; 3]> = made.iter().map(key).collect();
    made.sort();
    let at = super::at;
    let mut expected: Vec<[u32; 3]> = (0..3)
        .flat_map(|edge| {
            // `a` is `ab`'s and `ac`'s first end, `b` `bc`'s.
            let middle = if edge == 2 { B } else { A };
            let corners = [
                at(edge, false, 1),
                at(edge, true, 1),
                at(edge, true, -1),
                at(edge, false, -1),
            ];
            (0..3).map(move |k| key(&[middle, corners[k], corners[k + 1]]))
        })
        .collect();
    expected.sort();
    assert_eq!(made, expected);
}

/// Every triangle that covers anything winds, from the domain's order to the framebuffer's, as
/// a line's band does where `n` is kept, and the other way where it is turned: for triangles of
/// both windings, x- and y-major edges and every direction.
#[test]
fn every_band_triangle_winds_as_a_lines_band_turned_by_the_facing() {
    // A line's band: corner `(u, v)` at endpoint `u`, `(2v - 1) h` across.
    let line = |s: [f64; 2], e: [f64; 2], h: f64| {
        let n = minor_axis(s, e);
        move |[u, v]: [f64; 2]| {
            let at = if u > 0.5 { e } else { s };
            let across = (2.0 * v - 1.0) * h;
            [at[0] + n[0] * across, at[1] + n[1] * across]
        }
    };
    let mut line_winding = Vec::new();
    for (s, e) in [([0.0, 0.0], [9.0, 2.0]), ([3.0, 8.0], [1.0, -6.0])] {
        let place = line(s, e, 1.5);
        for t in [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ] {
            line_winding.push(area(t).signum() * area(t.map(&place)).signum());
        }
    }
    line_winding.dedup();
    let [line_winding] = line_winding[..] else {
        panic!("a line's band winds two ways: {line_winding:?}");
    };
    let triangles = [
        [[0.0, 0.0], [40.0, 10.0], [10.0, 30.0]],
        [[0.0, 0.0], [10.0, 30.0], [40.0, 10.0]],
        [[5.0, -3.0], [-20.0, 4.0], [7.0, -25.0]],
        [[-8.0, 12.0], [3.0, 12.0], [3.0, -1.0]],
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
    ];
    let mut checked = 0;
    for triangle in triangles {
        for facing in [1.0, -1.0] {
            for t in domain_triangles() {
                let points = t.map(Vertex::point);
                let [p, q, r] = points;
                if p == q || q == r || p == r {
                    continue;
                }
                let placed = points.map(|p| place(p, triangle, 2.0, facing));
                let winding = area(t.map(Vertex::domain)).signum() * area(placed).signum();
                assert_eq!(
                    winding,
                    facing * line_winding,
                    "{triangle:?} facing {facing}: {t:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, triangles.len() * 2 * 9);
}
