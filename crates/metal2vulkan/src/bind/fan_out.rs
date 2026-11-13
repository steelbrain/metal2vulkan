//! The region table of a rate-mapped pass drawn by fan-out: each primitive once, its vertices
//! shaded once, and a geometry stage that emits a copy of each triangle per region it touches
//! (`pipeline::link::geometry`, the fan-out stage), where the regions route
//! ([`RateMapRegions`](crate::bind::RateMapRegions)) records each draw once per region.
//!
//! A region of the plan is a viewport and a scissor: its line maps the guest's normalized device
//! coordinates to physical pixels, and its scissor keeps its own pixels. The fan-out draw has one
//! viewport over the whole target, so a copy of a triangle for a region carries both in clip
//! space, per axis:
//!
//! - its position, `x' = a·x + b·w`, where the target's viewport `(X₀, W₀)` maps `x'` to the
//!   same window coordinate as the region's viewport `(X, W)` maps `x`: `a = W / W₀` and
//!   `b = (2X + W − 2X₀) / W₀ − 1` (Vulkan's viewport transform, `window = X + (ndc + 1)·W / 2`,
//!   whatever the sign of `W`);
//! - two clip distances that hold the window coordinate within the scissor's pixels `[p₀, p₁)`:
//!   `(W/2)·x + (X + W/2 − p₀)·w` and `−(W/2)·x + (p₁ − X − W/2)·w`, the window coordinate times
//!   `w` less `p₀`, and `p₁` less it. Clipping keeps a fragment whose pixel centre lies inside, and
//!   a pixel centre lies half a pixel from every scissor edge, so each pixel the triangle covers is
//!   shaded by the one copy of its region.
//!
//! A copy's fragments are the region route's up to rounding: its window coordinates are the
//! region viewport's computed another way, so a pixel centre that lies within a rounding error of
//! a triangle's edge may be covered by one route and not the other (a near-equivalent, rule 1's
//! fast default; the regions route is the exact one).
//!
//! The table holds, for each axis, the segments of the plan's regions that hold a pixel, in the
//! order of their normalized device coordinates: a region is a segment of each axis. The stage
//! reads a triangle's bounds in normalized device coordinates (`x / w`, every `w` positive),
//! finds the segments of each axis they reach by bisection, and emits a copy for each pair, at
//! most its copies; a triangle with more, or with a `w` not positive and another positive, is
//! the probe's to find, whose bounds are widened by [`WIDEN_PIXELS`] of a pixel so that a draw
//! the probe passed reaches no more.
//!
//! The layout (`std430`, 16-byte entries): a header of the counts `uvec4(nx, ny, 0, 0)` and the
//! widenings `vec4(ex, ey, 0, 0)`, then [`MAX_SEGMENTS`] lines and as many clips per axis, x's
//! then y's: a line `vec4(ndc_lo, ndc_hi, a, b)`, a clip `vec4(W/2, X + W/2 − p₀, p₁ − X − W/2,
//! 0)`.

/// The words of a probe's flags each probed draw takes: its flag, then [`PROBE_SETS`] sets of
/// [`PROBE_COUNT_WORDS`] (`pipeline::link::fan_out`, the probe), from word `PROBE_DRAW_WORDS·i`
/// for the draw of index `i`, each zero before the probe. The consumer sums a draw's counts,
/// draws a draw none of whose wide words is marked through its variant of few copies
/// (`RegionFanOut::DrawFew`), takes the first and the last flagged primitive of a flagged draw
/// from the least of the third set and the greatest of the fourth, and sums the fifth for its
/// flagged triangles.
pub const PROBE_DRAW_WORDS: u32 = 1 + PROBE_SETS * PROBE_COUNT_WORDS;

/// The sets of [`PROBE_COUNT_WORDS`] after a probed draw's flag ([`PROBE_DRAW_WORDS`]), in
/// order: the counts its triangles add their copies into; the marks of its triangles that it
/// does not flag and that reach more than [`FEW_COPIES`] pairs of segments; the greatest
/// `u32::MAX − PrimitiveId` of its flagged triangles; the greatest `PrimitiveId + 1` of them;
/// and the counts its flagged triangles add one each into.
pub const PROBE_SETS: u32 = 5;

/// The words a probed draw's triangles add their copies into, and in each of the other sets of
/// its words ([`PROBE_DRAW_WORDS`]) as many, a triangle the one of each set its `PrimitiveId`
/// picks (`PrimitiveId mod PROBE_COUNT_WORDS`): one
/// word every triangle of a draw added into made the probe 15.4 to 20.7 ms of an RTX 3090's GPU
/// time for 64 draws of 306,036 triangles; spread over these, 1.45 ms (NVIDIA 620.70.06, the
/// pieces lane).
pub const PROBE_COUNT_WORDS: u32 = 32;

/// The most copies of a triangle a draw's variant of few copies emits (`RegionFanOut::DrawFew`),
/// which a draw whose triangles each reach at most this many pairs of segments draws through: its
/// geometry stage declares the output of this many copies' vertices, not the most copies'. The
/// pieces lane's draws are all of few copies: in one run a Radeon 780M (Mesa 26.2.6) drew two warm
/// frames in 29.7 and 30.8 ms of command buffer GPU time, where the same draws through the most
/// copies took 36.8 and 34.5 ms; an Iris Xe (Mesa 26.2.6) drew it in 40.0 ms and an RTX 3090
/// (NVIDIA 620.70.06) in 6.1 ms, as through the most copies in earlier runs (40.5 and 6.2 ms).
pub const FEW_COPIES: u32 = 4;

/// The most segments an axis holds: [`RateMapRegions::MAX_REGIONS`](crate::bind::RateMapRegions) is
/// this many per axis squared.
pub const MAX_SEGMENTS: u32 = 128;

/// The entries before the lines: the counts and the widenings.
pub const HEADER_ENTRIES: u32 = 2;

/// The table's bytes.
pub const TABLE_BYTES: u64 = (HEADER_ENTRIES as u64 + 4 * MAX_SEGMENTS as u64) * 16;

/// How far, in physical pixels, the probe widens a triangle's bounds before it counts the
/// segments they reach: past the difference rounding makes between the probe's and the draw's
/// vertex positions, each its own pipeline's.
pub const WIDEN_PIXELS: f64 = 0.125;

/// The entry of axis `axis`'s line `k`.
pub const fn line_entry(axis: u32, k: u32) -> u32 {
    HEADER_ENTRIES + axis * MAX_SEGMENTS + k
}

/// The entry of axis `axis`'s clip `k`.
pub const fn clip_entry(axis: u32, k: u32) -> u32 {
    HEADER_ENTRIES + 2 * MAX_SEGMENTS + axis * MAX_SEGMENTS + k
}

/// One segment of an axis: the region viewport's `(X, W)` on it, Vulkan's (`W` negative where
/// the consumer flips the axis), and its scissor's pixels `[p₀, p₁)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Segment {
    origin: f64,
    extent: f64,
    pixels: [u32; 2],
}

impl Segment {
    /// Its normalized device coordinates, least first: those of its pixels' edges.
    fn ndc(&self) -> [f64; 2] {
        let at = |p: u32| 2.0 * (f64::from(p) - self.origin) / self.extent - 1.0;
        let [a, b] = [at(self.pixels[0]), at(self.pixels[1])];
        [a.min(b), a.max(b)]
    }

    /// Its line `vec4(ndc_lo, ndc_hi, a, b)` for the target's viewport `(X₀, W₀)` on its axis.
    fn line(&self, (x0, w0): (f64, f64)) -> [f32; 4] {
        let [lo, hi] = self.ndc();
        let a = self.extent / w0;
        let b = (2.0 * self.origin + self.extent - 2.0 * x0) / w0 - 1.0;
        [lo, hi, a, b].map(|v| v as f32)
    }

    /// Its clip `vec4(W/2, X + W/2 − p₀, p₁ − X − W/2, 0)`.
    fn clip(&self) -> [f32; 4] {
        let half = self.extent / 2.0;
        let centre = self.origin + half;
        [
            half,
            centre - f64::from(self.pixels[0]),
            f64::from(self.pixels[1]) - centre,
            0.0,
        ]
        .map(|v| v as f32)
    }
}

/// The region table of a fan-out draw through the Vulkan viewport `target` (`[x, y, width,
/// height]`, its height negative where the consumer flips y) over the regions `regions`, each its
/// Vulkan viewport and its scissor `[x, y, width, height]`, those that hold a pixel (the module
/// docs). `None` where the regions are not one segment of each axis per pair (a plan's are), or an
/// axis has more than [`MAX_SEGMENTS`] or none.
pub fn table(target: [f32; 4], regions: &[([f32; 4], [u32; 4])]) -> Option<Vec<u8>> {
    let axis = |a: usize| {
        let mut segments: Vec<Segment> = Vec::new();
        for &(viewport, scissor) in regions {
            let segment = Segment {
                origin: f64::from(viewport[a]),
                extent: f64::from(viewport[a + 2]),
                pixels: [scissor[a], scissor[a] + scissor[a + 2]],
            };
            if !segments.contains(&segment) {
                segments.push(segment);
            }
        }
        segments.sort_by(|s, t| s.ndc()[0].total_cmp(&t.ndc()[0]));
        segments
    };
    let [xs, ys] = [axis(0), axis(1)];
    let fits = |s: &[Segment]| !s.is_empty() && s.len() <= MAX_SEGMENTS as usize;
    if !fits(&xs) || !fits(&ys) || xs.len() * ys.len() != regions.len() {
        return None;
    }
    let mut entries = vec![[0u32; 4]; (TABLE_BYTES / 16) as usize];
    entries[0] = [xs.len() as u32, ys.len() as u32, 0, 0];
    // A physical pixel is `2 / |W|` of normalized device coordinates; the widening is the least
    // segment's.
    let widen = |s: &[Segment]| {
        (s.iter())
            .map(|s| WIDEN_PIXELS * 2.0 / s.extent.abs())
            .fold(0.0f64, f64::max) as f32
    };
    entries[1] = [widen(&xs), widen(&ys), 0.0, 0.0].map(f32::to_bits);
    for (a, segments) in [&xs, &ys].into_iter().enumerate() {
        let a32 = a as u32;
        let target = (f64::from(target[a]), f64::from(target[a + 2]));
        for (k, segment) in segments.iter().enumerate() {
            let k = k as u32;
            entries[line_entry(a32, k) as usize] = segment.line(target).map(f32::to_bits);
            entries[clip_entry(a32, k) as usize] = segment.clip().map(f32::to_bits);
        }
    }
    Some(
        entries
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    )
}

#[cfg(test)]
mod tests;
