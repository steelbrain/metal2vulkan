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
//! A copy's fragments are its region viewport's up to rounding: its window coordinates are the
//! region viewport's computed another way, so a pixel centre that lies within a rounding error of
//! a triangle's edge may be covered by the copy and not through the region's own viewport (a
//! near-equivalent, rule 1's fast default; `RateMapRegionPolicy::Exact`, which draws every region
//! through its own viewport, is the exact one).
//!
//! **One route per pass.** Where a pass fans its triangles out, the draws it records region by
//! region take their region-copy variant (`RegionFanOut::RegionCopies`): region `k` of a
//! recording the first it pushes (`PushField::RegionFirst`) plus its draw's `DrawIndex` among a
//! multi-draw's (0 for a draw that is no multi-draw's), the segments `(k mod nx, k / nx)`, and
//! the vertex function putting each vertex in region `k` by the emitter of the fan-out stage's
//! copies (`pipeline::link::fan_out::region_copy`). It stores `DrawIndex` as `ViewportIndex`, and
//! every viewport the recording sets is the target's, each scissor its region's ([`scissors`]):
//! the rasterizer rejects a triangle outside the region by its scissor, as it does through a
//! region's own viewport, and the viewport transform is the fan-out's whatever the index,
//! Vulkan's being "determined by the selected viewport's width and height in pixels, px and py,
//! respectively, and its center (ox, oy) (also in pixels), as well as its depth range"
//! (*Controlling the Viewport*). Every route then hands the rasterizer the same position and clip
//! distances for a vertex in a region, through the same viewport transform: a later pass that
//! tests `Equal` the depth one draw stored meets it where the other route drew it. The copy's
//! arithmetic is decorated `NoContraction`, each operation correctly rounded on its own (Vulkan's
//! *Precision and Operation of SPIR-V Instructions*), and its `Position` and `ClipDistance`
//! outputs are `Invariant`. The routes run the copy in different pipelines and different stages
//! (the region-copy variant's vertex function, the fan-out variant's geometry stage), each on the
//! position its own vertex function computes: the identity rests on those correctly rounded
//! operations and on the guest's position computation compiling alike in both pipelines, which
//! Vulkan's invariance appendix does not promise across pipelines (its Rule 4), and which is
//! checked by running it (`docs/FAST-PATHS.md`, known gaps).
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
/// [`PROBE_COUNT_WORDS`] (`pipeline::link::fan_out`, the probe), then its blocks' bounds
/// ([`PROBE_BOUNDS_AT`]), from word `PROBE_DRAW_WORDS·i` for the draw of index `i`, each zero
/// before the probe. The consumer sums a draw's counts, draws a draw none of whose wide words is
/// marked through its variant of few copies (`RegionFanOut::DrawFew`), takes the first and the
/// last flagged primitive of a flagged draw from the least of the third set and the greatest of
/// the fourth, sums the fifth for its flagged triangles, reads the sixth as the bits of its
/// blocks that hold a flagged one, which it draws region by region, sums the seventh for its
/// flagged triangles with some `w` not positive beside a positive one, and draws a run of blocks
/// only into the regions its blocks' bounds reach.
pub const PROBE_DRAW_WORDS: u32 = PROBE_BOUNDS_AT + PROBE_BLOCKS;

/// The sets of [`PROBE_COUNT_WORDS`] after a probed draw's flag ([`PROBE_DRAW_WORDS`]), in
/// order: the counts its triangles add their copies into; the marks of its triangles that it
/// does not flag and that reach more than [`FEW_COPIES`] pairs of segments; the greatest
/// `u32::MAX − PrimitiveId` of its flagged triangles; the greatest `PrimitiveId + 1` of them;
/// the counts its flagged triangles add one each into; the bits of its blocks of
/// [`BLOCK_TRIANGLES`] that hold a flagged triangle, block `b` bit `b mod 32` of word `b / 32`
/// ([`PROBE_BLOCKS`] of them, the last also every triangle past them; [`BLOCK_SET`]); and the
/// counts its flagged triangles with some `w` not positive beside a positive one add one each
/// into ([`STRADDLE_SET`]).
pub const PROBE_SETS: u32 = 7;

/// The set of a probed draw's words that holds the bits of its blocks ([`PROBE_SETS`]).
pub const BLOCK_SET: u32 = 5;

/// The set of a probed draw's words its flagged triangles with some `w` not positive beside a
/// positive one count into ([`PROBE_SETS`]).
pub const STRADDLE_SET: u32 = 6;

/// The word of a probed draw's words ([`PROBE_DRAW_WORDS`]) its blocks' bounds begin at: one
/// word a block of [`BLOCK_TRIANGLES`] ([`PROBE_BLOCKS`] of them, the last also every triangle
/// past them), the bits of the [`BOUND_BUCKETS`] buckets of each axis its triangles' bounds
/// reach ([`bucket`]), x's in bits 0 to 15 and y's in bits 16 to 31, widened as the probe's
/// bounds are; for a triangle with some `w` not positive beside a positive one, the bounds of
/// its part where `w` is positive, unbounded on each side an edge crosses `w = 0`.
pub const PROBE_BOUNDS_AT: u32 = 1 + PROBE_SETS * PROBE_COUNT_WORDS;

/// The buckets of segments of each axis a block's bounds hold ([`PROBE_BOUNDS_AT`], [`bucket`]).
pub const BOUND_BUCKETS: u32 = 16;

/// The bucket of segment `segment` of an axis of `segments` segments ([`BOUND_BUCKETS`]):
/// `segment · BOUND_BUCKETS / segments`, so each bucket holds consecutive segments and every
/// segment one bucket.
pub const fn bucket(segment: u32, segments: u32) -> u32 {
    segment * BOUND_BUCKETS / segments
}

/// The triangles of a probed draw's block (`PrimitiveId / BLOCK_TRIANGLES`; [`PROBE_SETS`]):
/// [`PROBE_BLOCKS`] of them cover 65,536 triangles, more than a command of the pieces lane draws
/// (161,820 indices, 53,940 triangles), and a block drawn region by region for one flagged
/// triangle draws at most 63 others so.
pub const BLOCK_TRIANGLES: u32 = 64;

/// The blocks of a probed draw whose bits its last set holds ([`PROBE_SETS`]): one a bit.
pub const PROBE_BLOCKS: u32 = 32 * PROBE_COUNT_WORDS;

// A block is a shift of the primitive's index.
const _: () = assert!(BLOCK_TRIANGLES.is_power_of_two());

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

/// The segments of each axis of `regions` ([`table`]'s), in the order of their normalized device
/// coordinates, or `None` where they make no table.
fn segments(regions: &[([f32; 4], [u32; 4])]) -> Option<[Vec<Segment>; 2]> {
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
    Some([xs, ys])
}

/// The region table of a fan-out draw through the Vulkan viewport `target` (`[x, y, width,
/// height]`, its height negative where the consumer flips y) over the regions `regions`, each its
/// Vulkan viewport and its scissor `[x, y, width, height]`, those that hold a pixel (the module
/// docs). `None` where the regions are not one segment of each axis per pair (a plan's are), or an
/// axis has more than [`MAX_SEGMENTS`] or none.
pub fn table(target: [f32; 4], regions: &[([f32; 4], [u32; 4])]) -> Option<Vec<u8>> {
    let [xs, ys] = segments(regions)?;
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

/// The scissors `[x, y, width, height]` of the regions of [`table`]'s table over `regions`, in
/// the table's order: region `k` of segments `(k mod nx, k / nx)` (the module docs, **One route
/// per pass**), the scissor a region-copy variant's draw of region `k` takes. `None` where
/// [`table`] makes none.
pub fn scissors(regions: &[([f32; 4], [u32; 4])]) -> Option<Vec<[u32; 4]>> {
    let [xs, ys] = segments(regions)?;
    let scissor = |x: &Segment, y: &Segment| {
        let ([x0, x1], [y0, y1]) = (x.pixels, y.pixels);
        [x0, y0, x1 - x0, y1 - y0]
    };
    Some(
        (ys.iter())
            .flat_map(|y| xs.iter().map(|x| scissor(x, y)))
            .collect(),
    )
}

#[cfg(test)]
mod tests;
