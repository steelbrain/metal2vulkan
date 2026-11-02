//! The line width: Apple's `setLineWidth:`.
//!
//! Measured on the Apple M3 Max with macOS 27.0.1 (the consumer's probes and
//! `crates/metal2vulkan-conformance/probes/line_widths.m`, single-sampled, every pixel of a 16x16
//! and a 32x32 target):
//!
//! - The width is the encoder's: each encoder starts at 1, and a width holds across draws and
//!   pipeline states until the next call or the encoder's end.
//! - Metal draws `w' = floor(16 w) / 16`, clamped to `[0, 16]`: a negative width, `-0` and NaN
//!   draw as 0, and 16 and above, infinity among them, as 16 ([`LineWidth::of`]).
//! - At `w' = 1` a line is Metal's ordinary line, the half-open diamond-exit rule (the runtime's
//!   Bresenham lines). At every other `w'`, below 1 too, it is the parallelogram Vulkan's
//!   non-strict wide lines may be: its major axis the larger of the segment's framebuffer extents,
//!   x on a tie; its two minor-axis edges through the endpoints, each `w'` long and centred on
//!   its endpoint; its other two edges parallel to the segment; the ends not extended; a pixel
//!   drawn where its centre lies inside, by the polygon rules' ties (top and left edges in). The
//!   width is not rounded to whole pixels. The endpoints are first snapped to 1/256 pixel, a half
//!   rounded up. At `w' = 0` a line draws exactly as at 1/16, pixel for pixel: the centres within
//!   1/32 pixel of the segment along its minor axis, not only those on it.
//! - A strip or loop draws one parallelogram per segment, with no joins: where two overlap a pixel
//!   takes two fragments (at `w' = 1` a shared vertex is drawn once, as the diamond rule's).
//! - Triangles whose fill mode is `Lines` draw their edges by the same rule; in fill mode `Fill`
//!   the width changes nothing, nor for points.
//!
//! Not measured: other Apple GPUs; what a band's fragments read (their varyings and depth across
//! the width); a width finer than 1/16; multisampled targets beyond 4 samples at width 1; an
//! execution of indirect render commands' lines; a rasterization-rate-mapped pass's.
//!
//! The runtime draws each draw of lines by a route ([`LineRoute`]):
//!
//! - **Plain** at `w' = 1`, and for draws that draw no lines: as every line the runtime draws. A
//!   draw that rasterizes nothing is one, and so is a direct draw whose vertices or indices
//!   complete no primitive of its type (a list of lines of one vertex, triangles of two), which
//!   Metal draws as nothing ([`draws_primitives`]); an indirect draw's count is the GPU's, so it
//!   takes its route as any other.
//! - **Device**: the device's own lines at `w'` (`lineWidth`), where it enables `wideLines` and
//!   `w'` lies in its `lineWidthRange` and above 0. Exact where they are Metal's parallelogram:
//!   `strictLines` unset, `nonStrictWideLinesUseParallelogram` set (Vulkan states the shape only
//!   for widths above 1), and `w'` on `lineWidthGranularity` from the range's bottom. Elsewhere
//!   the device's line differs (a strict device's rectangle covers the pixels within half the
//!   width of the segment across it, square ends; a Bresenham one's columns; a width rounded to
//!   the granularity): announced (`Degradation::DeviceWideLines`), or under
//!   `WideLinePolicy::Exact` the band below, where it can draw.
//! - **Band**: the pipeline's band variant (`VariantKey::line_band`, the translator's
//!   `pipeline::link::band`), two triangles per line over Metal's parallelogram, through a
//!   tessellation control and evaluation stage (`tessellationShader`). It is every other width's
//!   route: no `wideLines` (MoltenVK), `w'` outside the range, and `w' = 0`, which the band draws
//!   at 1/16, as Metal does. A strip or a loop is drawn as the list of its segments
//!   (`render/restart.rs`); a flat input under the last provoking vertex reads the
//!   list led by each line's last vertex (`render/provoking.rs`), which covers the same pixels, as
//!   the parallelogram is the same either way.
//! - **Nearest**: the device's own lines at the width nearest Metal's they draw (`w'`, 1/16 at 0,
//!   clamped into `lineWidthRange` where it enables `wideLines`, else 1), where neither route above
//!   draws `w'`, announced (`Degradation::LinesAtNearestWidth`, with why): a triangle's edges in
//!   fill mode `Lines` and a mesh draw's lines, which the band does not draw; a line without
//!   `tessellationShader`; a band under vertex amplification above 1 (its viewports); and every
//!   line drawn in a rasterization-rate-mapped pass or as an execution's command, whose width
//!   Metal's answer is not measured for. Without `wideLines` they are 1 pixel wide whatever `w'`.
//!
//! Refused (`NotImplemented(RuntimeGap::WideLines)`): a line strip's band whose list of segments
//! the runtime cannot draw (`render/restart.rs`, as for a list of the last provoking vertex); and
//! as the translator refuses the band variant (`pipeline::link::band`: a vertex function that
//! selects a slice or viewport, a fragment function that reads barycentric weights, synthesized
//! geometry, multiview, stages past `maxTessellationEvaluationOutputComponents`).
//!
//! A multisampled target is drawn by the same routes: the band's triangles cover Metal's
//! parallelogram at each sample, as a non-strict device's lines do (the Vulkan specification's
//! Basic Line Segment Rasterization). Metal's width-1 lines there (the parallelogram at the
//! samples) are drawn as the runtime's plain lines.

use super::draw_shape::ShapeInputs;
use super::{Assembly, RenderLowering};
use crate::degradation::Degradation;
use crate::pipeline::{FixedState, GraphicsState as DrawingState};
use metal2vulkan::bind::{DrawCall, RewriteProgram};
use metal2vulkan::device::{DeviceProfile, VkFeature, WideLinePolicy};
use metal2vulkan::metal::{PrimitiveType, TriangleFillMode};
use metal2vulkan::pipeline::LineBandState;
use metal2vulkan::vk;

/// Metal's width of a line, `w'`: in sixteenths of a pixel, 0 to 256 (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LineWidth(u16);

impl LineWidth {
    /// Metal's default, every encoder's at its start.
    pub(crate) const ONE: LineWidth = LineWidth(16);

    /// The width Metal draws for `setLineWidth:`'s `width`: `floor(16 width) / 16` in `[0, 16]`,
    /// NaN and every width not above 0 as 0.
    pub(crate) fn of(width: f32) -> LineWidth {
        if width.is_nan() || width <= 0.0 {
            return LineWidth(0);
        }
        if width >= 16.0 {
            return LineWidth(256);
        }
        // In `(0, 256)`: the cast truncates the floor exactly.
        LineWidth((width * 16.0).floor() as u16)
    }

    /// The width in framebuffer pixels.
    pub(crate) fn pixels(self) -> f32 {
        f32::from(self.0) / 16.0
    }
}

/// How a draw draws its lines at the encoder's width (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum LineRoute {
    /// As every line the runtime draws: Metal's width 1, or a draw of no lines.
    Plain,
    /// The device's own lines `width` pixels wide (`lineWidth`), Metal's width, which differ from
    /// Metal's parallelogram where `differs` (announced).
    Device { width: f32, differs: bool },
    /// The pipeline's band variant, `half` pixels to each side of the segment.
    Band { half: f32 },
    /// The device's own lines `width` pixels wide, the width nearest Metal's they draw, where
    /// neither route above draws Metal's, for `why` (announced).
    Nearest { width: f32, why: Unrouted },
}

/// Why a draw's lines take the nearest width the device's lines draw (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unrouted {
    /// A triangle's edges in fill mode `Lines` or a mesh draw's lines, which the band does not
    /// draw.
    Unbanded,
    /// No `tessellationShader`, which the band needs.
    Untessellated,
    /// Vertex amplification above 1, whose views the band does not draw.
    Amplified,
    /// A rasterization-rate-mapped pass, whose lines Metal's answer is not measured for.
    RateMapped,
    /// An execution's command, whose lines Metal's answer is not measured for.
    Executed,
}

impl LineRoute {
    /// Whether the draw runs the band variant.
    pub(super) fn band(self) -> bool {
        matches!(self, LineRoute::Band { .. })
    }

    /// `fixed` as the route draws: the device's lines at their width; a band's triangles filled,
    /// uncullable and front-facing, as a line is (`LineBandState`).
    pub(super) fn fix(self, fixed: &mut FixedState) {
        match self {
            LineRoute::Plain => {}
            LineRoute::Device { width, .. } | LineRoute::Nearest { width, .. } => {
                fixed.line_width = width.to_bits()
            }
            LineRoute::Band { .. } => {
                fixed.cull_mode = vk::CullModeFlags::NONE;
                fixed.front_face = LineBandState::FRONT_FACE;
                fixed.polygon_mode = vk::PolygonMode::FILL;
            }
        }
    }
}

/// What of a draw the width changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drawn {
    /// Nothing: points, filled triangles, a draw that rasterizes nothing or draws no primitive.
    Nothing,
    /// Line primitives, which the band draws.
    Lines,
    /// Lines the band does not draw: a triangle's edges in fill mode `Lines`, an emulated mesh's
    /// lines, and a mesh draw's (whose lines the runtime does not tell from its triangles).
    Other,
}

/// Whether `draw` of `assembly` may draw a primitive: every draw but a direct one whose vertices
/// or indices complete no primitive of its type (`RewriteProgram::capacity`: a list of lines of
/// one vertex, a strip of triangles of two, quads of three), which Metal draws as nothing. An
/// indirect draw's count is the GPU's, a patch draw's patches are drawn or culled by the GPU's
/// factors, and a mesh draw's primitives are its mesh function's: each may draw one.
pub(super) fn draws_primitives(assembly: Assembly, draw: &DrawCall) -> bool {
    let count = match *draw {
        DrawCall::Primitives { vertex_count, .. } => vertex_count,
        DrawCall::Indexed { index_count, .. } => index_count,
        _ => return true,
    };
    (assembly.program()).is_none_or(|program| program.capacity(u64::from(count)) > 0)
}

/// What `assembly` draws that the width changes, in fill mode `fill`, under a pipeline that
/// rasterizes nothing where `discards`, whose emulated meshes' primitives have `mesh_corners`
/// corners (`None` for a mesh draw the device draws), of a draw that may draw a primitive where
/// `primitives` ([`draws_primitives`]).
fn drawn(
    assembly: Assembly,
    fill: TriangleFillMode,
    discards: bool,
    mesh_corners: Option<u32>,
    primitives: bool,
) -> Drawn {
    if discards || !primitives {
        return Drawn::Nothing;
    }
    let edges = match fill {
        TriangleFillMode::Lines => Drawn::Other,
        _ => Drawn::Nothing,
    };
    match assembly {
        Assembly::Primitives(PrimitiveType::Point) => Drawn::Nothing,
        Assembly::Primitives(PrimitiveType::Line | PrimitiveType::LineStrip)
        | Assembly::Private(RewriteProgram::LineLoop) => Drawn::Lines,
        Assembly::Primitives(PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
        | Assembly::Private(_)
        | Assembly::Patches => edges,
        Assembly::Meshes => match mesh_corners {
            Some(1) => Drawn::Nothing,
            Some(2) | None => Drawn::Other,
            Some(_) => edges,
        },
    }
}

/// The route a draw of `drawn` at Metal's width `width` takes on a device of `profile` under
/// `policy` (the module docs), drawn with vertex amplification above 1 where `amplifies`, in a
/// rasterization-rate-mapped pass where `rate_mapped`, and as an execution's command where
/// `executes`.
fn route(
    profile: &DeviceProfile,
    policy: WideLinePolicy,
    width: LineWidth,
    drawn: Drawn,
    (amplifies, rate_mapped, executes): (bool, bool, bool),
) -> LineRoute {
    if width == LineWidth::ONE || drawn == Drawn::Nothing {
        return LineRoute::Plain;
    }
    let enabled = profile.enabled();
    let limits = profile.limits();
    let w = width.pixels();
    let [low, high] = limits.line_width_range_bits.map(f32::from_bits);
    let granularity = f32::from_bits(limits.line_width_granularity_bits);
    let wide_lines = enabled.features.contains(&VkFeature::WideLines);
    // The width nearest the one Metal draws (`w'`, 1/16 at 0) the device's lines draw: in their
    // range, else 1.
    let nearest = |why| LineRoute::Nearest {
        width: match wide_lines {
            true => w.max(1.0 / 16.0).max(low).min(high),
            false => 1.0,
        },
        why,
    };
    if rate_mapped {
        return nearest(Unrouted::RateMapped);
    }
    if executes {
        return nearest(Unrouted::Executed);
    }
    let drawable = wide_lines && w > 0.0 && (low..=high).contains(&w);
    // On the granularity's grid from the range's bottom: the widths the device draws without
    // rounding.
    let on_grid = granularity > 0.0 && {
        let steps = (w - low) / granularity;
        steps == steps.round()
    };
    let exact = drawable
        && !limits.strict_lines
        && limits.non_strict_wide_lines_use_parallelogram
        && w > 1.0
        && on_grid;
    let tessellates = enabled.features.contains(&VkFeature::TessellationShader);
    // Where the band can draw the draw's lines.
    let band = drawn == Drawn::Lines && tessellates && !amplifies;
    if exact || (drawable && (policy == WideLinePolicy::Device || !band)) {
        return LineRoute::Device {
            width: w,
            differs: !exact,
        };
    }
    if !band {
        return nearest(match drawn {
            Drawn::Lines if !tessellates => Unrouted::Untessellated,
            Drawn::Lines => Unrouted::Amplified,
            _ => Unrouted::Unbanded,
        });
    }
    // Metal draws `w' = 0` as 1/16 (the module docs).
    let half = w.max(1.0 / 16.0) / 2.0;
    LineRoute::Band { half }
}

impl ShapeInputs<'_> {
    /// The route `call`'s draw of `pipeline` as `assembly` draws its lines by at the encoder's
    /// width (the module docs), of a draw that may draw a primitive where `primitives`
    /// ([`draws_primitives`]); announcing a device line that differs from Metal's, and one at
    /// the nearest width, where `announce`.
    pub(super) fn line_route(
        &self,
        call: &'static str,
        (pipeline, assembly, primitives): (&DrawingState, Assembly, bool),
        announce: bool,
    ) -> LineRoute {
        let graphics = &pipeline.base().graphics;
        let drawn = drawn(
            assembly,
            self.state.fill_mode(),
            graphics.rasterizer_discard,
            graphics.mesh_records.as_ref().map(|r| r.corners),
            primitives,
        );
        let compile = self.device.compile_device();
        let width = self.state.line_width();
        let route = route(
            compile.profile(),
            compile.policy().wide_lines,
            width,
            drawn,
            (
                self.amplification.amplifies(),
                self.rate_map.is_some(),
                self.executes,
            ),
        );
        match route {
            LineRoute::Device { differs: true, .. } if announce => {
                announce_device_lines(self.device, call, width.pixels());
            }
            LineRoute::Nearest { width: drawn, why } if announce => {
                announce_nearest_lines(self.device, call, (width.pixels(), drawn), why);
            }
            _ => {}
        }
        route
    }
}

/// Announce `call`'s draw of lines `w` pixels wide by the device's own lines, which differ from
/// Metal's parallelogram there (the module docs).
fn announce_device_lines(device: &crate::Device, call: &'static str, w: f32) {
    let limits = device.compile_device().profile().limits();
    let [low, high] = limits.line_width_range_bits.map(f32::from_bits);
    let granularity = f32::from_bits(limits.line_width_granularity_bits);
    let parallelogram = limits.non_strict_wide_lines_use_parallelogram;
    let shape = match (limits.strict_lines, parallelogram) {
        (true, _) => "strict lines, the rectangle the width spans across the segment",
        (false, true) if w < 1.0 => "lines, whose shape below a width of 1 Vulkan leaves to it",
        (false, true) => "lines, which round the width to their granularity",
        (false, false) => "lines, which Vulkan does not state as parallelograms",
    };
    device.degrade(Degradation::DeviceWideLines, || {
        (
            format!("{call}: a draw of lines at setLineWidth: {w}"),
            format!(
                "the device draws them as its own {shape} (lineWidthRange [{low}, {high}], \
                 lineWidthGranularity {granularity}), where Metal draws the parallelogram whose \
                 minor-axis edges pass through the endpoints"
            ),
        )
    });
}

/// Announce `call`'s draw of lines at Metal's width `w` by the device's own lines `drawn` pixels
/// wide, the nearest width they draw, for `why` (the module docs).
fn announce_nearest_lines(
    device: &crate::Device,
    call: &'static str,
    (w, drawn): (f32, f32),
    why: Unrouted,
) {
    let profile = device.compile_device().profile();
    let [low, high] = profile.limits().line_width_range_bits.map(f32::from_bits);
    let range = match profile.enabled().features.contains(&VkFeature::WideLines) {
        true => format!("lineWidthRange [{low}, {high}]"),
        false => "no wideLines: 1 pixel alone".to_owned(),
    };
    let (what, cause) = match why {
        Unrouted::Unbanded => (
            "a triangle's edges in fill mode Lines or a mesh's lines",
            "the runtime's band of two triangles draws lines alone",
        ),
        Unrouted::Untessellated => (
            "lines",
            "the band of two triangles needs tessellationShader",
        ),
        Unrouted::Amplified => (
            "lines",
            "the band does not draw vertex amplification's views",
        ),
        Unrouted::RateMapped => (
            "lines",
            "Metal's width is not measured in a rasterization-rate-mapped pass",
        ),
        Unrouted::Executed => (
            "lines",
            "Metal's width is not measured for an execution's commands",
        ),
    };
    device.degrade(Degradation::LinesAtNearestWidth, || {
        (
            format!("{call}: a draw of {what} at setLineWidth: {w}"),
            format!(
                "drawn by the device's own lines {drawn} pixels wide, the width nearest Metal's \
                 they draw ({range}), as {cause}"
            ),
        )
    });
}

impl RenderLowering {
    /// The lowering of `setLineWidth:`'s `width`: the encoder's draws from here draw lines
    /// `LineWidth::of(width)` wide (the module docs).
    pub(crate) fn set_encoder_line_width(&mut self, width: f32) {
        let width = LineWidth::of(width);
        // Only a change takes the state to a new version (`versioned.rs`), so the draws' memo
        // holds across a call that repeats the width.
        if self.state.line_width() != width {
            self.state.set_line_width(width);
        }
    }
}

#[cfg(test)]
mod tests;
