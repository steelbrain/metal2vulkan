//! A rate-mapped render pass drawn region by region: the option (a), on every device (the user's
//! decision of 2026-10-03: "Regions everywhere"). It is Metal's pass in structure; its lines are
//! this crate's tiles', within the bound of Apple's, so a pass over a map that is not the identity
//! is a near-equivalent ([`RateMapRegions::precision`]).
//!
//! Metal rasterizes a pass whose `MTLRenderPassDescriptor.rasterizationRateMap` is set into a
//! target of the map's physical size: inside one tile the map from screen to physical
//! coordinates is the tile's line, `P_k + (s - S_k) * (p_k / s_k)`, and a fragment is the physical
//! pixel whose center that line's inverse sends into the primitive, its attributes evaluated at
//! that screen point. The same transform is a Vulkan viewport: the guest's screen-space viewport
//! composed with a tile's line is another viewport. So the pass is drawn once per *region*, a
//! rectangle of tiles sharing one line on each axis, with that region's viewport and a scissor of
//! the region's physical pixels: every physical pixel is shaded by exactly one region, at the
//! position, with the attributes and the coverage of its tile's line.
//!
//! - Consecutive tiles of an axis whose ratios `p / s` are equal lie on one line (each starts
//!   where the previous ends), so they form one *segment*; a layer's regions are its horizontal
//!   segments times its vertical ones. A map of one quality per axis has at most two segments
//!   per axis (the last tile's ratio may differ); the worst case is one per tile.
//! - Tile boundaries are multiples of the granularity (32 physical pixels), so no 2x2 quad of
//!   physical pixels spans two regions: a derivative reads one region's line, as Metal's does.
//! - Regions are disjoint in physical pixels, so each pixel's fragments arrive in the draw's
//!   primitive order, as one draw's do: blending, depth and stencil tests, and framebuffer reads
//!   see what they see in Metal's pass.
//! - **The cost** is draws: each draw of the pass is recorded once per region
//!   ([`RateMapRegions::regions`]) whose pixels its viewport reaches (a region it does not reach
//!   shades nothing), so the vertex stage runs once per such region. A vertex function
//!   with side effects (device writes or atomics) would repeat them:
//!   [`RateMapRegions::admits`] refuses such a pipeline in a pass of more than one region. The
//!   vertex stage's resident-set misses repeat too: the `ExecutionStatus` miss count
//!   ([`StatusReport::resident_misses`](crate::bind::StatusReport::resident_misses)) counts each
//!   once per region. Fragment invocations are Metal's: one per physical pixel a primitive
//!   covers. A plan holds at most [`RateMapRegions::MAX_REGIONS`] regions.
//! - The consumer allocates the targets at least [`RateMapRegions::physical_size`], draws with
//!   each region's viewport (Metal's convention, y down; a Vulkan consumer that flips y with a
//!   negative height uses `y + height` and `-height`) and scissor, which stands for the guest's
//!   ([`RasterizationRateMap::regions`] takes the guest's), and answers the guest's map and
//!   decoder from the same [`RasterizationRateMap`], so the image and every decoder that reads it
//!   agree with Metal's up to the lines below. The viewport's depth range is the guest's.
//! - **The lines are this crate's**. A region draws with its tiles' exact lines, where Apple's are
//!   its fixed-point ones (a scale of `K / 2^24` and an offset of 1/256 pixels), whose tile starts
//!   fall between pixels: Metal evaluates a fragment at a screen point within about 0.025 pixels of
//!   the region path's (0.0194 between the exact lines and Apple's CPU map on the measured maps,
//!   plus the 0.0052 between Metal's evaluated point and Apple's CPU map), so coverage may differ
//!   within that distance of an edge; and Metal leaves an axis's last physical line unshaded where
//!   Apple's tile end falls short of the screen's edge and the line's center lies in the last
//!   screen pixel, which happens on some axes whose last tile maps fewer than 2 screen pixels to
//!   each physical one, by Apple's private arithmetic. An axis of the identity is Apple's exactly.
//!   So [`RateMapRegions::precision`] is `PrecisionNote::RasterizationRateMap` unless both of the
//!   layer's axes are the identity, and the consumer records it for the pass.
//! - **The viewport's fractions.** Over a map that is not the identity a region's viewport bounds
//!   are fractional. A device may interpret viewport bounds at only `viewportSubPixelBits` bits of
//!   sub-pixel precision; one that rounds them to `2^-b` physical pixels moves a fragment's
//!   evaluated point by up to `2^-b` times the tile's screen pixels per physical one, on top of the
//!   bound above. MoltenVK on Apple M3 and lavapipe both report 0 bits, yet keep the fractions:
//!   the position cases `fe7630f4` and `6e087167` match Metal on both, and each would differ in at
//!   least 46 texels had the bounds been rounded to whole pixels, in any direction. Elsewhere it is
//!   not measured; the note is recorded all the same.
//!
//! [`RasterizationRateMap::regions`] refuses ([`RegionError`]) a layer the map lacks; a guest
//! viewport that is not finite, has no area, or is flipped (a negative extent); a guest scissor
//! that does not cover the screen over a map that is not the identity (where Metal applies a
//! scissor in a rate-mapped pass is not measured); a plan of more than
//! [`RateMapRegions::MAX_REGIONS`] regions; and a region whose viewport exceeds the device's
//! `maxViewportDimensions`, leaves its `viewportBoundsRange`, or has an extent that is not a
//! normal `f32`. [`RateMapRegions::admits`] refuses a layered pipeline (several map layers in one
//! pass: each layer's regions differ, which one draw cannot select), one that selects viewports
//! per primitive (the plan sets one viewport per region), and one whose vertex function writes
//! memory in a plan of several regions; [`RateMapRegions::admits_planned_layers`] admits a
//! layered pipeline whose primitives the consumer has bounded to slices this plan's layer is the
//! map's layer of (one slice, or the map's layers alike), as Metal rasterizes each primitive by
//! the map's layer of its slice.

use std::fmt;

use super::{RasterizationRateMap, RateMapAxis};
use crate::pipeline::{GraphicsState, PrecisionNote};

/// The device limits a region plan must respect (`VkPhysicalDeviceLimits`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct RegionLimits {
    /// `maxViewportDimensions`.
    pub max_viewport_dimensions: [u32; 2],
    /// `viewportBoundsRange`: the least and greatest viewport coordinate.
    pub viewport_bounds_range: [f32; 2],
}

impl RegionLimits {
    /// Limits of these values.
    pub fn new(max_viewport_dimensions: [u32; 2], viewport_bounds_range: [f32; 2]) -> RegionLimits {
        RegionLimits {
            max_viewport_dimensions,
            viewport_bounds_range,
        }
    }
}

/// One region of a pass: its viewport and its scissor, in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateMapRegion {
    viewport: [f32; 4],
    scissor_offset: [u32; 2],
    scissor_extent: [u32; 2],
}

impl RateMapRegion {
    /// `[x, y, width, height]` in physical pixels, in Metal's convention (`MTLViewport`: origin at
    /// the top left, y down): the guest's screen-space viewport composed with the region's line.
    pub fn viewport(&self) -> [f32; 4] {
        self.viewport
    }

    /// The scissor's offset: the region's first physical pixel, within the guest's scissor.
    pub fn scissor_offset(&self) -> [u32; 2] {
        self.scissor_offset
    }

    /// The scissor's extent: the region's physical pixels within the guest's scissor (0 where
    /// the guest's scissor leaves none of them).
    pub fn scissor_extent(&self) -> [u32; 2] {
        self.scissor_extent
    }
}

/// One layer of a rate-mapped pass, planned region by region.
#[derive(Clone, Debug, PartialEq)]
pub struct RateMapRegions {
    physical_size: [u32; 2],
    regions: Vec<RateMapRegion>,
    identity: bool,
}

impl RateMapRegions {
    /// The most regions a plan holds: 128 segments per axis. A layer compiled by
    /// [`RasterizationRateMap::new`] has at most `⌈W / 32⌉` segments on an axis of `W` screen
    /// pixels (every tile before the last spans at least its 32 physical pixels), so every such
    /// layer whose screen fits 4096 x 4096 is planned; a larger one is refused
    /// ([`RegionError::TooManyRegions`]) where it has more, which bounds both the plan and the
    /// draws each guest draw becomes.
    pub const MAX_REGIONS: usize = 128 * 128;

    /// The layer's physical size: the least target extent the pass draws into.
    pub fn physical_size(&self) -> [u32; 2] {
        self.physical_size
    }

    /// The regions, row by row; each draw of the pass is recorded once per region.
    pub fn regions(&self) -> &[RateMapRegion] {
        &self.regions
    }

    /// The pass's near-equivalent: `PrecisionNote::RasterizationRateMap` unless both of the
    /// layer's axes are the identity (every tile as many screen pixels as physical ones), where
    /// the regions draw with this crate's tile lines, within the bound of Apple's, and may shade an
    /// axis's last physical line where Metal leaves it. `None`: the pass is Metal's.
    pub fn precision(&self) -> Option<PrecisionNote> {
        (!self.identity).then_some(PrecisionNote::RasterizationRateMap)
    }

    /// Whether the plan draws a pipeline of `graphics` as Metal does. Refused:
    ///
    /// - [`RegionError::LayeredPipeline`]: a stage declares `Layer`
    ///   ([`GraphicsState::declares_layer`]) or the pipeline amplifies vertices
    ///   ([`GraphicsState::max_vertex_amplification_count`] above 1), so its primitives may reach
    ///   another layer than the one planned;
    /// - [`RegionError::ViewportArray`]: the vertex function selects each primitive's viewport
    ///   ([`GraphicsState::writes_viewport_index`]), where the plan sets one per region;
    /// - [`RegionError::VertexSideEffects`]: the vertex function writes memory
    ///   ([`GraphicsState::vertex_writes_memory`]) and the plan has more than one region, so the
    ///   writes would repeat once per region.
    pub fn admits(&self, graphics: &GraphicsState) -> Result<(), RegionError> {
        if graphics.declares_layer || graphics.max_vertex_amplification_count > 1 {
            return Err(RegionError::LayeredPipeline);
        }
        self.admits_planned_layers(graphics)
    }

    /// [`RateMapRegions::admits`] for a draw whose every primitive renders a layer this plan is
    /// the plan of: one slice of the pass alone (a draw of one amplification, or one whose view
    /// mapping moves it to its slice), or any slice where the map's layers are alike. Metal
    /// rasterizes each primitive by the map's layer of the slice it renders (Metal probes on the M3
    /// Max and an M4: over a map whose two layers differ, layered draws, amplified ones and one
    /// moved to slice 1 each shaded a slice by that slice's layer, whatever its amplification), so
    /// the consumer, which knows the slices, bounds them and plans their layer. Refused as `admits`
    /// refuses but for [`RegionError::LayeredPipeline`].
    pub fn admits_planned_layers(&self, graphics: &GraphicsState) -> Result<(), RegionError> {
        if graphics.writes_viewport_index {
            return Err(RegionError::ViewportArray);
        }
        if graphics.vertex_writes_memory && self.regions.len() > 1 {
            return Err(RegionError::VertexSideEffects {
                regions: self.regions.len(),
            });
        }
        Ok(())
    }
}

/// Why a pass cannot be planned region by region.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum RegionError {
    /// The map has no such layer.
    Layer { layer: u32 },
    /// The guest's viewport `[x, y, width, height]` has a coordinate that is not finite.
    ViewportNotFinite { viewport: [f64; 4] },
    /// The guest's viewport has no extent on `axis`.
    EmptyViewport { axis: usize },
    /// The guest's viewport has a negative extent on `axis` (a flipped viewport), which is not
    /// planned.
    NegativeViewport { axis: usize, extent: f64 },
    /// The guest's scissor `[x, y, width, height]` does not cover the screen, over a map that is
    /// not the identity: where Metal applies a scissor in such a pass is not measured.
    Scissor { scissor: [u32; 4] },
    /// The plan would hold `regions` regions, above [`RateMapRegions::MAX_REGIONS`].
    TooManyRegions { regions: u64, max: usize },
    /// A region's viewport extent exceeds `maxViewportDimensions` on `axis`.
    ViewportDimensions { axis: usize, extent: f64, max: u32 },
    /// A region's viewport leaves `viewportBoundsRange` on `axis`.
    ViewportBounds {
        axis: usize,
        from: f64,
        to: f64,
        range: [f32; 2],
    },
    /// A region's viewport extent on `axis` is not a normal `f32`: the guest's extent is too
    /// small for the region's line to keep it.
    RegionExtent { axis: usize, extent: f64 },
    /// The pipeline is layered: it declares `Layer` or amplifies vertices
    /// ([`RateMapRegions::admits`]).
    LayeredPipeline,
    /// The pipeline's vertex function selects each primitive's viewport
    /// ([`RateMapRegions::admits`]).
    ViewportArray,
    /// The pipeline's vertex function writes memory, which a plan of `regions` regions would run
    /// once per region ([`RateMapRegions::admits`]).
    VertexSideEffects { regions: usize },
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionError::Layer { layer } => write!(f, "the rate map has no layer {layer}"),
            RegionError::ViewportNotFinite { viewport } => {
                write!(f, "the viewport {viewport:?} is not finite")
            }
            RegionError::EmptyViewport { axis } => {
                write!(f, "the viewport has no extent on axis {axis}")
            }
            RegionError::NegativeViewport { axis, extent } => write!(
                f,
                "the viewport's extent {extent} on axis {axis} is negative, a flipped viewport, \
                 which a rate-mapped pass does not plan"
            ),
            RegionError::Scissor { scissor } => write!(
                f,
                "the scissor {scissor:?} does not cover the screen, and where Metal applies a \
                 scissor in a pass over a rate map that is not the identity is not measured"
            ),
            RegionError::TooManyRegions { regions, max } => write!(
                f,
                "the pass would be drawn in {regions} regions, above the {max} a plan holds"
            ),
            RegionError::ViewportDimensions { axis, extent, max } => write!(
                f,
                "a region's viewport extent {extent} on axis {axis} exceeds maxViewportDimensions \
                 {max}"
            ),
            RegionError::ViewportBounds {
                axis,
                from,
                to,
                range,
            } => write!(
                f,
                "a region's viewport [{from}, {to}] on axis {axis} leaves viewportBoundsRange \
                 {range:?}"
            ),
            RegionError::RegionExtent { axis, extent } => write!(
                f,
                "a region's viewport extent {extent} on axis {axis} is not a normal f32"
            ),
            RegionError::LayeredPipeline => f.write_str(
                "the pipeline is layered (it declares a render target slice or amplifies \
                 vertices), which a rate-mapped pass does not plan",
            ),
            RegionError::ViewportArray => f.write_str(
                "the vertex function selects its viewport, where a rate-mapped pass sets one per \
                 region",
            ),
            RegionError::VertexSideEffects { regions } => write!(
                f,
                "the vertex function writes memory, which a rate-mapped pass drawn in {regions} \
                 regions would repeat once per region"
            ),
        }
    }
}

impl std::error::Error for RegionError {}

/// A run of consecutive tiles on one line: its first screen and physical pixel, its physical
/// end, and its ratio `p / s` as a fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    screen: u32,
    physical: u32,
    physical_end: u32,
    p: u32,
    s: u32,
}

/// An axis's segments: consecutive tiles of equal ratio merged.
fn segments(axis: &RateMapAxis) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let (mut screen, mut physical) = (0u32, 0u32);
    for &(s, p) in axis.tiles() {
        match out.last_mut() {
            Some(last) if u64::from(last.p) * u64::from(s) == u64::from(p) * u64::from(last.s) => {
                last.physical_end += p;
            }
            _ => out.push(Segment {
                screen,
                physical,
                physical_end: physical + p,
                p,
                s,
            }),
        }
        screen += s;
        physical += p;
    }
    out
}

/// One segment's viewport on one axis, `(origin, extent)` in `f32`, and its scissor,
/// `(offset, extent)`, within the guest's scissor `clip` (`[from, to)` in physical pixels).
type Line = (f32, f32, u32, u32);

/// The viewport of `seg` on axis `a` for the guest's viewport `origin` and `size`, checked
/// against `limits`, and its scissor within `clip`.
fn line(
    a: usize,
    seg: &Segment,
    origin: f64,
    size: f64,
    clip: [u64; 2],
    limits: &RegionLimits,
) -> Result<Line, RegionError> {
    let ratio = f64::from(seg.p) / f64::from(seg.s);
    let from = f64::from(seg.physical) + (origin - f64::from(seg.screen)) * ratio;
    let extent = size * ratio;
    // The viewport the device receives is these `f32`s, so the device's limits are checked on
    // them.
    let (from32, extent32) = (from as f32, extent as f32);
    if !extent32.is_normal() {
        return Err(RegionError::RegionExtent { axis: a, extent });
    }
    let max = limits.max_viewport_dimensions[a];
    if f64::from(extent32) > f64::from(max) {
        return Err(RegionError::ViewportDimensions {
            axis: a,
            extent,
            max,
        });
    }
    let [lo, hi] = limits.viewport_bounds_range;
    let to = f64::from(from32) + f64::from(extent32);
    if f64::from(from32) < f64::from(lo) || to > f64::from(hi) {
        return Err(RegionError::ViewportBounds {
            axis: a,
            from,
            to: from + extent,
            range: limits.viewport_bounds_range,
        });
    }
    // The region's physical pixels within the guest's scissor; an empty intersection keeps the
    // region's start, so the offset stays inside the physical image.
    let (start, end) = (u64::from(seg.physical), u64::from(seg.physical_end));
    let lo = clip[0].clamp(start, end);
    let hi = clip[1].clamp(lo, end);
    // Both lie in `[start, end]`, inside the axis's physical extent, a `u32`.
    Ok((from32, extent32, lo as u32, (hi - lo) as u32))
}

impl RasterizationRateMap {
    /// Layer `layer` of a pass drawn region by region (the option (a)): for the guest's viewport
    /// `[x, y, width, height]` in screen pixels (`MTLViewport`, y down) and its scissor `[x, y,
    /// width, height]` (`MTLScissorRect`, `None` where the guest set none), each region's physical
    /// viewport and scissor, within `limits`. Over a layer of the identity, whose screen and
    /// physical pixels are one, the guest's scissor clips the regions'; over any other, it must
    /// cover the screen ([`RegionError::Scissor`]).
    pub fn regions(
        &self,
        layer: u32,
        viewport: [f64; 4],
        scissor: Option<[u32; 4]>,
        limits: &RegionLimits,
    ) -> Result<RateMapRegions, RegionError> {
        let axes = self.layer(layer).ok_or(RegionError::Layer { layer })?;
        if viewport.iter().any(|v| !v.is_finite()) {
            return Err(RegionError::ViewportNotFinite { viewport });
        }
        let [vx, vy, vw, vh] = viewport;
        for (axis, extent) in [vw, vh].into_iter().enumerate() {
            if extent < 0.0 {
                return Err(RegionError::NegativeViewport { axis, extent });
            }
            if extent == 0.0 {
                return Err(RegionError::EmptyViewport { axis });
            }
        }
        let identity = axes
            .iter()
            .all(|axis| axis.tiles().iter().all(|&(s, p)| s == p));
        let physical_size = [axes[0].physical_extent(), axes[1].physical_extent()];
        let whole = |a: usize| [0, u64::from(physical_size[a])];
        let clip = match scissor {
            None => [whole(0), whole(1)],
            Some([x, y, w, h]) if identity => [
                [u64::from(x), u64::from(x) + u64::from(w)],
                [u64::from(y), u64::from(y) + u64::from(h)],
            ],
            // Whether Metal reads it in screen or in physical pixels, it covers the image.
            Some([0, 0, w, h]) if w >= self.screen_size()[0] && h >= self.screen_size()[1] => {
                [whole(0), whole(1)]
            }
            Some(rect) => return Err(RegionError::Scissor { scissor: rect }),
        };
        let [xs, ys] = [segments(&axes[0]), segments(&axes[1])];
        let count = xs.len() as u64 * ys.len() as u64;
        if count > RateMapRegions::MAX_REGIONS as u64 {
            return Err(RegionError::TooManyRegions {
                regions: count,
                max: RateMapRegions::MAX_REGIONS,
            });
        }
        let per_axis = |a: usize, segs: &[Segment], origin: f64, size: f64| {
            segs.iter()
                .map(|seg| line(a, seg, origin, size, clip[a], limits))
                .collect::<Result<Vec<Line>, RegionError>>()
        };
        let xs = per_axis(0, &xs, vx, vw)?;
        let ys = per_axis(1, &ys, vy, vh)?;
        let regions = ys
            .iter()
            .flat_map(|&(y, h, sy, ey)| {
                xs.iter().map(move |&(x, w, sx, ex)| RateMapRegion {
                    viewport: [x, y, w, h],
                    scissor_offset: [sx, sy],
                    scissor_extent: [ex, ey],
                })
            })
            .collect();
        Ok(RateMapRegions {
            physical_size,
            regions,
            identity,
        })
    }
}

#[cfg(test)]
mod tests;
