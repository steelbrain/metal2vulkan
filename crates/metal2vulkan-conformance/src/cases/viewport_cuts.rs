//! Which pixels a **viewport** leaves of a draw that reaches past it.
//!
//! A draw's geometry is clipped to the view volume, and the viewport maps that volume onto the
//! target, so the viewport's edges are where a draw that reaches past them is cut. Two questions
//! of where exactly, each a case of one pass:
//!
//! | case | what it draws, and what the target then says |
//! |---|---|
//! | `fractional-viewport-coverage` | a filled draw covering the whole of each of four viewports whose edges lie inside a pixel: which pixels an edge that cuts through a pixel keeps |
//! | `lines-fill-past-the-viewport` | three triangles under `MTLTriangleFillModeLines` whose edges cross, or wholly surround, an 8 by 8 viewport: which pixels the cut edges light, and whether the cut itself draws a line |
//!
//! **Each records the whole target**, the pixels no draw may reach among them, because a viewport
//! is a claim about where a draw did not land as much as where it did.
//!
//! What Metal answers here is not stated in advance. Vulkan clips a primitive to its clip volume,
//! whose `x` and `y` bounds the viewport maps onto its own edges, and a fill then covers the
//! pixels whose centres the clipped primitive holds. `point-viewport-coverage` found Metal doing
//! something else for points (its fitted rule, held against its recording): Metal keeps a point
//! whose centre lies within `1.5 w` of the viewport's centre on each axis, past the clip volume,
//! and covers only pixels the viewport overlaps, any part of a pixel being enough. Each case here
//! therefore draws its geometry both inside that `1.5 w` band and past it, and the module's tests
//! hold Metal's recording only to what every one of those rules gives: a pixel the viewport holds
//! whole is drawn, and a pixel it does not reach is not. The pixels between are the cases'
//! question, and the recording is their answer.
//!
//! **The expectation is arithmetic where it is stated at all.** The target is `RGBA32Float`, so a
//! texel is the `float4` the fragment function returned: `tob_solid` returns the colour its
//! binding holds, every draw's colour is its own, and none is the clear.
//!
//! Window coordinates here are Metal's: `x = originX + (x_c / w + 1) * width / 2` and
//! `y = originY + (1 - y_c / w) * height / 2`, `y` growing down the target.
//!
//! Nothing here can fault a GPU: one pass a case, at most eight draws of three or six vertices
//! over a 48 by 24 target, no loop in the fixture, and every viewport inside its target.

use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec, ViewportRect,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TriangleFillMode};

/// The name of the fractional viewport's case.
pub const FRACTIONAL: &str = "fractional-viewport-coverage";

/// The name of the line-fill case.
pub const LINES: &str = "lines-fill-past-the-viewport";

/// The format every target here carries: a `float` stored as it is.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The clear every pass loads its target with, which a pixel no draw wrote holds.
const CLEAR: f32 = -7.5;

/// The owned fixture the rectangle, the triangle and the solid colour come from.
const FIXTURE: &str = "tob_renders";

/// `tob_rect`: a rectangle from a buffer's two clip-space corners, six vertices.
const RECT: &str = "tob_rect";

/// `tob_triangle`: the triangle of the three points a buffer holds.
const TRIANGLE: &str = "tob_triangle";

/// `tob_solid`: the `float4` a fragment buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// The entry points a case's shader reference names.
const ENTRIES: &str = "tob_rect+tob_triangle+tob_solid";

/// The draw a rectangle takes: two triangles of six vertices, one instance.
const RECT_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 6,
    instances: 1,
};

/// The draw one triangle takes: three vertices, one instance.
const TRIANGLE_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How far apart the colours lie in a colours buffer.
const COLOR_STRIDE: u64 = 16;

/// How far apart the triangles lie in a points buffer: three `float2`s take 24 bytes and the
/// stride is the next multiple of 16.
const POINT_STRIDE: u64 = 32;

/// How many runs record mode takes each case over. One pass of one queue, nothing a device may
/// reorder, so the runs are expected to fold into one outcome; eight say so on evidence.
const RUNS: u32 = 8;

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(FractionalViewports),
        Box::new(LinesPastTheViewport),
    ]
}

/// The colours a buffer of `colors` holds, one after the other.
fn colors_buffer(colors: &[[f32; 4]]) -> Vec<u8> {
    colors
        .iter()
        .flatten()
        .flat_map(|channel| channel.to_le_bytes())
        .collect()
}

/// `points`, one triangle every [`POINT_STRIDE`] bytes.
fn points_buffer(triangles: &[[[f32; 2]; 3]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for triangle in triangles {
        let start = bytes.len();
        for point in triangle {
            bytes.extend(point.iter().flat_map(|axis| axis.to_le_bytes()));
        }
        bytes.resize(start + POINT_STRIDE as usize, 0);
    }
    bytes
}

/// A private render target of [`FORMAT`] and `size`, of `MTLTextureUsageRenderTarget` alone.
fn target_spec(size: [u32; 2]) -> TargetSpec {
    TargetSpec::plain(FORMAT, size)
}

/// Bytes the whole of a level 0 of `size` takes.
fn plane([width, height]: [u32; 2]) -> u32 {
    TEXEL * width * height
}

/// The layout a target's whole level 0 is read under: one row of the layout a row of texels, one
/// element a channel.
fn layout([width, height]: [u32; 2]) -> ByteLayout {
    let row_pitch = TEXEL * width;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [width, height, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * width,
        row_pitch,
        rows: height,
        image_pitch: row_pitch * height,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The shader reference both texts of the one fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs the fixture's own texts are.
fn fixture_inputs() -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of the module's one fixture.
fn library(side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// The pass that clears `target` and runs `steps`, and the piece that copies the target's level 0
/// into `into`.
fn work(target: Handle, into: Handle, size: [u32; 2], steps: Vec<RenderStep>) -> Vec<RenderWork> {
    vec![
        RenderWork::Pass(
            RenderPass::clearing(vec![PassColor::cleared(target, [f64::from(CLEAR); 4])])
                .drawing(steps),
        ),
        RenderWork::ReadTexture {
            texture: target,
            into,
            offset: 0,
            row_pitch: TEXEL * size[0],
            size,
            plane: Plane::Whole,
            at: ReadAt::BASE,
        },
    ]
}

/// Run `work` as one command buffer, emit what each of its two pieces answered, and read the
/// target's texels back out of `read_back`.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    work: &[RenderWork],
    read_back: Handle,
    size: [u32; 2],
) -> Result<(), Abort> {
    let outcome = call(sites, "render", side.render(work), |outcome| {
        outcome.status.clone()
    })?;
    let names = ["pass", "read"];
    let mut pieces = Node::branch("pieces", Vec::new());
    for (at, what) in names.iter().enumerate() {
        pieces = pieces.with(Node::leaf(
            *what,
            Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    sites.push(pieces);
    if let Some((at, failure)) = names
        .iter()
        .enumerate()
        .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (*what, f)))
    {
        return Err(Abort {
            at: format!("pieces/{at}"),
            failure: failure.clone(),
        });
    }
    let bytes = call(
        sites,
        "read-back",
        side.read_buffer(read_back, 0, plane(size) as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout(size)))),
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// fractional-viewport-coverage

/// The side of one cell of the fractional case's target, in pixels: each draw's viewport lies in
/// a cell of its own, with pixels it does not reach on every side.
const CELL: u32 = 12;

/// Where each viewport's near edge lies past its cell's pixel 2, and its extent, the same on both
/// axes. One column of cells each.
///
/// - an edge on a pixel boundary, the far edge `.7` into a pixel: the far row and column hold
///   their pixel centres although the viewport holds only most of each pixel;
/// - both edges `.3` and `.7` into a pixel: the near pixel's centre too is held by a viewport that
///   holds only most of the pixel;
/// - both edges `.75` and `.25` into a pixel: the near and the far pixel each overlap the viewport
///   by a quarter, and neither centre is held;
/// - both edges on a pixel centre.
const ARMS: [(f64, f64); 4] = [(0.0, 7.7), (0.3, 7.4), (0.75, 6.5), (0.5, 7.0)];

/// The covering rectangle's corners, as `tob_rect` takes them: past the clip square on every side,
/// and inside the `1.5 w` band Metal keeps a point in.
const COVER: [f32; 4] = [-1.25, -1.25, 1.25, 1.25];

/// The covering triangle: the clip square inside it, a unit from each leg and more from the third
/// edge, and two corners well past the band Metal keeps a point in.
const BEYOND: [[f32; 2]; 3] = [[-2.0, -2.0], [6.0, -2.0], [-2.0, 6.0]];

/// The colour each draw stores, in draw order: the rectangle through each viewport, then the
/// triangle through each. Every channel an exact binary fraction and every colour its own.
const FRACTIONAL_COLORS: [[f32; 4]; 8] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.125, 0.375, 0.625, 0.875],
    [2.5, -1.25, 0.0, 1.0],
    [-0.5, 3.75, 0.5, 0.25],
    [1.5, -2.25, 4.5, -0.75],
    [3.25, 0.125, -3.5, 0.5],
    [-1.75, 2.125, 1.25, 0.75],
    [0.625, -4.5, 2.75, -1.5],
];

/// The fractional case's target: a row of cells for the rectangle and one for the triangle.
const FRACTIONAL_SIZE: [u32; 2] = [CELL * ARMS.len() as u32, CELL * 2];

/// The viewport of arm `arm` in the cell row `row`.
fn arm_viewport(arm: usize, row: u32) -> ViewportRect {
    let (near, extent) = ARMS[arm];
    let cell = [CELL * arm as u32, CELL * row].map(f64::from);
    ViewportRect::new([cell[0] + 2.0 + near, cell[1] + 2.0 + near], [extent; 2])
}

/// Every viewport the fractional case sets, in draw order.
fn fractional_viewports() -> Vec<ViewportRect> {
    (0..2)
        .flat_map(|row| (0..ARMS.len()).map(move |arm| arm_viewport(arm, row)))
        .collect()
}

/// A covering fill through four fractional viewports, twice: a rectangle that reaches a little
/// past each, and a triangle that reaches far past.
pub struct FractionalViewports;

impl FractionalViewports {
    /// The pass's two pipelines: the rectangle and the triangle, each the solid colour.
    fn pipelines(colors: &[Option<ColorState>]) -> [RenderPipelineSpec<'_>; 2] {
        [
            RenderPipelineSpec::new(RECT, SOLID, colors),
            RenderPipelineSpec::new(TRIANGLE, SOLID, colors),
        ]
    }

    /// The pass's steps: each pipeline with its geometry, then one draw through each viewport.
    fn steps(states: [Handle; 2], geometry: [Handle; 2], colors: Handle) -> Vec<RenderStep> {
        let mut steps = Vec::new();
        let viewports = fractional_viewports();
        for (row, (state, buffer)) in states.into_iter().zip(geometry).enumerate() {
            steps.push(RenderStep::Pipeline(state));
            steps.push(RenderStep::VertexBuffer {
                buffer,
                offset: 0,
                index: 0,
            });
            for arm in 0..ARMS.len() {
                let at = row * ARMS.len() + arm;
                steps.push(RenderStep::Viewport(viewports[at]));
                steps.push(RenderStep::FragmentBuffer {
                    buffer: colors,
                    offset: at as u64 * COLOR_STRIDE,
                    index: 0,
                });
                steps.push(match row {
                    0 => RECT_DRAW,
                    _ => TRIANGLE_DRAW,
                });
            }
        }
        steps
    }

    /// The rectangle's buffer: its two corners.
    fn cover_buffer() -> Vec<u8> {
        COVER.iter().flat_map(|edge| edge.to_le_bytes()).collect()
    }
}

impl Case for FractionalViewports {
    fn name(&self) -> &'static str {
        FRACTIONAL
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors = [Some(ColorState::new(FORMAT))];
        for (at, pipeline) in Self::pipelines(&colors).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes(
            "target",
            &target_spec(FRACTIONAL_SIZE).words(),
        ));
        inputs.push(Input::bytes("buffer:cover", &Self::cover_buffer()));
        inputs.push(Input::bytes("buffer:beyond", &points_buffer(&[BEYOND])));
        inputs.push(Input::bytes(
            "buffer:colors",
            &colors_buffer(&FRACTIONAL_COLORS),
        ));
        // The pass's own words carry every viewport by its bits; a handle's number is a side's
        // own, so a placeholder stands in for each.
        let placeholder = Handle(0);
        let pass =
            RenderPass::clearing(vec![PassColor::cleared(placeholder, [f64::from(CLEAR); 4])])
                .drawing(Self::steps([placeholder; 2], [placeholder; 2], placeholder));
        inputs.push(Input::bytes("pass", &pass.words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&colors_state).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let target = made(sites, "target", side.target(&target_spec(FRACTIONAL_SIZE)))?;
        let cover = made(sites, "buffer:cover", side.buffer(&Self::cover_buffer()))?;
        let beyond = made(
            sites,
            "buffer:beyond",
            side.buffer(&points_buffer(&[BEYOND])),
        )?;
        let colors = made(
            sites,
            "buffer:colors",
            side.buffer(&colors_buffer(&FRACTIONAL_COLORS)),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(FRACTIONAL_SIZE) as usize),
        )?;
        let steps = Self::steps([states[0], states[1]], [cover, beyond], colors);
        let work = work(target, read_back, FRACTIONAL_SIZE, steps);
        drawn(side, sites, &work, read_back, FRACTIONAL_SIZE)
    }
}

// ------------------------------------------------------------------------------------------------
// lines-fill-past-the-viewport

/// The side of one cell of the line case's target: each draw's viewport lies in a cell of its own,
/// four pixels in from each side.
const LINE_CELL: u32 = 16;

/// The three triangles, in clip space, each through the viewport of its own cell:
///
/// 1. one corner inside the viewport, the other two past its right and its bottom edge and inside
///    the `1.5 w` band Metal keeps a point in, so every edge crosses the viewport's edge;
/// 2. the same shape with the two outer corners far past that band;
/// 3. a triangle the whole viewport lies inside, every edge more than a pixel outside it and two
///    corners past the band, so whatever it lights is the cut alone.
///
/// Every coordinate is a multiple of 1/64 and every edge passes at least a sixteenth of a pixel
/// from every pixel centre and every corner of a pixel's diamond in the target, so which pixels an
/// edge lights is never a tie of the line rule (`the_line_edges_pass_off_every_tie`).
const LINE_TRIANGLES: [[[f32; 2]; 3]; 3] = [
    [
        [-0.203125, 0.046875],
        [1.25, 0.09375],
        [-0.203125, -1.359375],
    ],
    [
        [-0.390625, 0.328125],
        [2.953125, 0.296875],
        [-0.484375, -3.203125],
    ],
    [
        [-1.328125, -1.453125],
        [4.5, -1.453125],
        [-1.328125, 4.40625],
    ],
];

/// The colour each line draw stores.
const LINE_COLORS: [[f32; 4]; 3] = [
    [0.25, 0.5, 0.75, 1.0],
    [2.5, -1.25, 0.0, 1.0],
    [-1.75, 2.125, 1.25, 0.75],
];

/// The line case's target: one row of three cells.
const LINE_SIZE: [u32; 2] = [LINE_CELL * LINE_TRIANGLES.len() as u32, LINE_CELL];

/// The viewport of the line draw `at`: 8 by 8 pixels, four in from each side of its cell. Its
/// edges lie on pixel boundaries, so the question of a fractional edge stays the other case's.
fn line_viewport(at: usize) -> ViewportRect {
    ViewportRect::new([f64::from(LINE_CELL * at as u32) + 4.0, 4.0], [8.0, 8.0])
}

/// Three triangles under `MTLTriangleFillModeLines`, each past the viewport it is drawn through.
pub struct LinesPastTheViewport;

impl LinesPastTheViewport {
    /// The one pipeline: the triangle and the solid colour.
    fn pipeline(colors: &[Option<ColorState>]) -> RenderPipelineSpec<'_> {
        RenderPipelineSpec::new(TRIANGLE, SOLID, colors)
    }

    /// The pass's steps: the fill mode of lines, then each triangle through its viewport.
    fn steps(state: Handle, points: Handle, colors: Handle) -> Vec<RenderStep> {
        let mut steps = vec![
            RenderStep::Pipeline(state),
            RenderStep::FillMode(TriangleFillMode::Lines),
        ];
        for at in 0..LINE_TRIANGLES.len() {
            steps.push(RenderStep::Viewport(line_viewport(at)));
            steps.push(RenderStep::VertexBuffer {
                buffer: points,
                offset: at as u64 * POINT_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::FragmentBuffer {
                buffer: colors,
                offset: at as u64 * COLOR_STRIDE,
                index: 0,
            });
            steps.push(TRIANGLE_DRAW);
        }
        steps
    }
}

impl Case for LinesPastTheViewport {
    fn name(&self) -> &'static str {
        LINES
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs()?;
        let colors = [Some(ColorState::new(FORMAT))];
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&colors).words(),
        ));
        inputs.push(Input::bytes("target", &target_spec(LINE_SIZE).words()));
        inputs.push(Input::bytes(
            "buffer:points",
            &points_buffer(&LINE_TRIANGLES),
        ));
        inputs.push(Input::bytes("buffer:colors", &colors_buffer(&LINE_COLORS)));
        // The pass's own words carry the fill mode and every viewport; a placeholder stands in for
        // each handle.
        let placeholder = Handle(0);
        let pass =
            RenderPass::clearing(vec![PassColor::cleared(placeholder, [f64::from(CLEAR); 4])])
                .drawing(Self::steps(placeholder, placeholder, placeholder));
        inputs.push(Input::bytes("pass", &pass.words()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites)?;
        let colors_state = [Some(ColorState::new(FORMAT))];
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&colors_state)),
        )?;
        let target = made(sites, "target", side.target(&target_spec(LINE_SIZE)))?;
        let points = made(
            sites,
            "buffer:points",
            side.buffer(&points_buffer(&LINE_TRIANGLES)),
        )?;
        let colors = made(
            sites,
            "buffer:colors",
            side.buffer(&colors_buffer(&LINE_COLORS)),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(plane(LINE_SIZE) as usize),
        )?;
        let work = work(
            target,
            read_back,
            LINE_SIZE,
            Self::steps(state, points, colors),
        );
        drawn(side, sites, &work, read_back, LINE_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// How far `point-viewport-coverage` found Metal keeping a point past the viewport's centre,
    /// in `w` on each axis. The geometry here is drawn on both sides of it.
    const KEPT_BAND: f64 = 1.5;

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing has recorded it.
    /// The runs are expected to fold into one outcome, and this says so where they did not.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The target's recorded texels of `case`, as `float4`s in row order.
    fn recorded(case: &str, size: [u32; 2]) -> Option<Vec<[f32; 4]>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), plane(size) as usize);
        Some(
            bytes
                .chunks_exact(TEXEL as usize)
                .map(|texel| {
                    let mut channels = [0f32; 4];
                    for (channel, word) in channels.iter_mut().zip(texel.chunks_exact(4)) {
                        *channel = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                    }
                    channels
                })
                .collect(),
        )
    }

    /// How much of pixel `pixel` along one axis the span `[low, high)` holds, from 0 to 1.
    fn overlap(pixel: u32, low: f64, high: f64) -> f64 {
        let pixel = f64::from(pixel);
        ((pixel + 1.0).min(high) - pixel.max(low)).max(0.0)
    }

    /// How much of pixel `(x, y)` `viewport` holds, from 0 to 1.
    fn held(viewport: &ViewportRect, x: u32, y: u32) -> f64 {
        let along = |pixel, axis: usize| {
            let low = viewport.origin[axis];
            overlap(pixel, low, low + viewport.size[axis])
        };
        along(x, 0) * along(y, 1)
    }

    /// Where a pixel's centre lies against the span `[low, high)` along one axis.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Centre {
        /// The span holds the whole pixel.
        Whole,
        /// The span holds part of the pixel and its centre.
        In,
        /// The span holds part of the pixel and not its centre.
        Out,
        /// The span's edge runs through the pixel's centre.
        OnEdge,
    }

    /// Where pixel `pixel`'s centre lies against `[low, high)`, for a pixel the span reaches.
    fn centre(pixel: u32, low: f64, high: f64) -> Centre {
        let at = f64::from(pixel) + 0.5;
        if overlap(pixel, low, high) == 1.0 {
            Centre::Whole
        } else if at == low || at == high {
            Centre::OnEdge
        } else if low < at && at < high {
            Centre::In
        } else {
            Centre::Out
        }
    }

    /// The window coordinates of the clip-space point `point` through `viewport`, `y` down the
    /// target, as the module's docs state Metal's transform.
    fn window(point: [f32; 2], viewport: &ViewportRect) -> [f64; 2] {
        let [x, y] = point.map(f64::from);
        [
            viewport.origin[0] + (x + 1.0) * viewport.size[0] / 2.0,
            viewport.origin[1] + (1.0 - y) * viewport.size[1] / 2.0,
        ]
    }

    /// The distance from `point` to the segment from `a` to `b`.
    fn distance(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let along = [b[0] - a[0], b[1] - a[1]];
        let to = [point[0] - a[0], point[1] - a[1]];
        let length = along[0] * along[0] + along[1] * along[1];
        let t = ((to[0] * along[0] + to[1] * along[1]) / length).clamp(0.0, 1.0);
        let nearest = [a[0] + t * along[0], a[1] + t * along[1]];
        ((point[0] - nearest[0]).powi(2) + (point[1] - nearest[1]).powi(2)).sqrt()
    }

    /// Whether the clip-space point `point` lies strictly inside the clip square.
    fn inside_clip(point: [f32; 2]) -> bool {
        point.iter().all(|axis| axis.abs() < 1.0)
    }

    /// **Every viewport lies inside its own cell, with pixels it does not reach on every side**,
    /// so a pixel's cell names the one draw that may write it and the pixels around a viewport are
    /// the case's "just outside".
    #[test]
    fn every_viewport_lies_inside_its_own_cell() {
        let viewports = fractional_viewports();
        assert_eq!(viewports.len(), FRACTIONAL_COLORS.len());
        for (at, viewport) in viewports.iter().enumerate() {
            let cell = [at % ARMS.len(), at / ARMS.len()].map(|c| f64::from(CELL * c as u32));
            for axis in 0..2 {
                let low = viewport.origin[axis];
                let high = low + viewport.size[axis];
                assert!(low.floor() >= cell[axis] + 1.0, "{viewport:?}");
                assert!(
                    high.ceil() <= cell[axis] + f64::from(CELL) - 1.0,
                    "{viewport:?}"
                );
                assert!(high <= f64::from(FRACTIONAL_SIZE[axis]), "{viewport:?}");
            }
        }
        for at in 0..LINE_TRIANGLES.len() {
            let viewport = line_viewport(at);
            let cell = f64::from(LINE_CELL * at as u32);
            assert!(viewport.origin[0] >= cell + 1.0);
            assert!(viewport.origin[0] + viewport.size[0] <= cell + f64::from(LINE_CELL) - 1.0);
            assert!(viewport.origin[1] >= 1.0);
            assert!(viewport.origin[1] + viewport.size[1] <= f64::from(LINE_SIZE[1]) - 1.0);
        }
    }

    /// **Each arm puts the edge it is for to Metal**: where the near and the far pixel's centres lie
    /// against the viewport, on both axes, in both rows.
    #[test]
    fn each_arm_cuts_its_edge_pixels_as_its_docs_say() {
        let expected = [
            (Centre::Whole, Centre::In),
            (Centre::In, Centre::In),
            (Centre::Out, Centre::Out),
            (Centre::OnEdge, Centre::OnEdge),
        ];
        for (at, viewport) in fractional_viewports().iter().enumerate() {
            for axis in 0..2 {
                let low = viewport.origin[axis];
                let high = low + viewport.size[axis];
                let near = low.floor() as u32;
                let far = high.ceil() as u32 - 1;
                assert_eq!(
                    (centre(near, low, high), centre(far, low, high)),
                    expected[at % ARMS.len()],
                    "{viewport:?} along axis {axis}"
                );
                // The pixels on either side of the viewport it does not reach at all.
                assert_eq!(overlap(near - 1, low, high), 0.0);
                assert_eq!(overlap(far + 1, low, high), 0.0);
            }
        }
    }

    /// **The rectangle reaches a little past each viewport, inside Metal's kept band, and the
    /// triangle far past it**, each covering the whole clip square.
    #[test]
    fn the_covering_draws_reach_past_the_viewport_on_both_sides_of_the_band() {
        for corner in COVER {
            let reach = f64::from(corner.abs());
            assert!(reach > 1.0 && reach < KEPT_BAND, "{corner}");
        }
        assert!(COVER[0] < COVER[2] && COVER[1] < COVER[3]);
        // The triangle's legs lie along x = -2 and y = -2 and its third edge along x + y = 4, so
        // the clip square lies inside it: a unit from each leg, and its corner (1, 1), where
        // x + y = 2, is the square's nearest point to the third edge.
        let [a, b, c] = BEYOND;
        assert_eq!((a[0], c[0]), (-2.0, -2.0));
        assert_eq!((a[1], b[1]), (-2.0, -2.0));
        assert_eq!((b[0] + b[1], c[0] + c[1]), (4.0, 4.0));
        assert!(BEYOND
            .iter()
            .flatten()
            .any(|axis| f64::from(axis.abs()) > KEPT_BAND));
    }

    /// **Every draw stores a colour of its own**, none of them the clear, so a pixel's colour names
    /// the draw that wrote it.
    #[test]
    fn every_draw_stores_a_colour_of_its_own() {
        for colors in [&FRACTIONAL_COLORS[..], &LINE_COLORS[..]] {
            let distinct: BTreeSet<[u32; 4]> =
                colors.iter().map(|color| color.map(f32::to_bits)).collect();
            assert_eq!(distinct.len(), colors.len());
            assert!(!distinct.contains(&[CLEAR; 4].map(f32::to_bits)));
        }
    }

    /// **The line triangles cross the viewport's edges as their docs say**: the first two with one
    /// corner inside the clip square and two outside, the first inside Metal's kept band and the
    /// second past it; the third around the whole viewport, every edge more than a pixel outside.
    #[test]
    fn the_line_triangles_cross_or_surround_the_viewport() {
        for (at, triangle) in LINE_TRIANGLES[..2].iter().enumerate() {
            assert!(inside_clip(triangle[0]), "{at}");
            assert!(
                !inside_clip(triangle[1]) && !inside_clip(triangle[2]),
                "{at}"
            );
            let past_band = triangle
                .iter()
                .flatten()
                .any(|axis| f64::from(axis.abs()) > KEPT_BAND);
            assert_eq!(past_band, at == 1, "{at}");
        }
        let viewport = line_viewport(2);
        let corners = LINE_TRIANGLES[2].map(|point| window(point, &viewport));
        let low = viewport.origin;
        let high = [low[0] + viewport.size[0], low[1] + viewport.size[1]];
        for k in 0..3 {
            let (a, b) = (corners[k], corners[(k + 1) % 3]);
            for x in [low[0], high[0]] {
                for y in [low[1], high[1]] {
                    let d = distance([x, y], a, b);
                    assert!(
                        d > 1.0,
                        "edge {k} passes {d} from the viewport's corner ({x}, {y})"
                    );
                    // The viewport's corner lies on the triangle's inner side of the edge.
                    let cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
                    let inner = {
                        let c = corners[(k + 2) % 3];
                        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    };
                    assert!(cross * inner > 0.0, "edge {k} cuts the viewport");
                }
            }
        }
        assert!(LINE_TRIANGLES[2]
            .iter()
            .flatten()
            .any(|axis| f64::from(axis.abs()) > KEPT_BAND));
    }

    /// **No edge of a line triangle meets a tie of the line rule**: every edge, whole and uncut,
    /// passes at least a sixteenth of a pixel from every pixel centre of the target and every
    /// corner of a pixel's diamond, so which pixel an edge lights is decided by where it runs and
    /// never by how a device breaks a tie.
    ///
    /// A cut the clip makes along a viewport edge lies on pixel boundaries, so it runs through
    /// diamond corners by construction; that is part of what the case asks, not a tie it avoids.
    #[test]
    fn the_line_edges_pass_off_every_tie() {
        let mut ties = Vec::new();
        for y in 0..=LINE_SIZE[1] {
            for x in 0..=LINE_SIZE[0] {
                let [x, y] = [f64::from(x), f64::from(y)];
                ties.extend([[x + 0.5, y + 0.5], [x + 0.5, y], [x, y + 0.5]]);
            }
        }
        for (at, triangle) in LINE_TRIANGLES.iter().enumerate() {
            for point in triangle.iter().flatten() {
                assert_eq!((point * 64.0).fract(), 0.0, "{point} is not a 64th");
            }
            let corners = triangle.map(|point| window(point, &line_viewport(at)));
            for k in 0..3 {
                let (a, b) = (corners[k], corners[(k + 1) % 3]);
                let (nearest, tie) = ties
                    .iter()
                    .map(|tie| (distance(*tie, a, b), *tie))
                    .min_by(|l, r| l.0.total_cmp(&r.0))
                    .unwrap();
                assert!(
                    nearest >= 1.0 / 16.0,
                    "triangle {at}'s edge {k} passes {nearest} from {tie:?}"
                );
            }
        }
    }

    /// **Metal's recorded target holds what any rule of a viewport gives**: a pixel the viewport
    /// holds whole is its draw's colour, a pixel it does not reach is the clear, and a pixel it cuts
    /// is one or the other. Which one is the case's question, and the recording its answer.
    #[test]
    fn metal_drew_each_fractional_viewport_within_its_pixels() {
        let Some(texels) = recorded(FRACTIONAL, FRACTIONAL_SIZE) else {
            println!("{FRACTIONAL}: no recording in the tree yet");
            return;
        };
        let viewports = fractional_viewports();
        let mut wrong = Vec::new();
        for y in 0..FRACTIONAL_SIZE[1] {
            for x in 0..FRACTIONAL_SIZE[0] {
                let at = ((y / CELL) * ARMS.len() as u32 + x / CELL) as usize;
                let got = texels[(y * FRACTIONAL_SIZE[0] + x) as usize];
                let share = held(&viewports[at], x, y);
                let allowed: &[[f32; 4]] = if share == 1.0 {
                    &[FRACTIONAL_COLORS[at]]
                } else if share == 0.0 {
                    &[[CLEAR; 4]]
                } else {
                    &[FRACTIONAL_COLORS[at], [CLEAR; 4]]
                };
                if !allowed.contains(&got) {
                    wrong.push(format!(
                        "({x}, {y}) holds {got:?}, and viewport {at} holds {share} of it"
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// **Metal's recorded line target holds what any rule of a cut edge gives**: a pixel a cell's
    /// viewport does not reach, nor touch, is the clear; the inside of the surrounded viewport,
    /// a pixel in from its edges, is the clear; and the two crossing triangles each lit a pixel of
    /// their viewport.
    #[test]
    fn metal_lit_the_line_edges_only_where_a_viewport_reaches() {
        let Some(texels) = recorded(LINES, LINE_SIZE) else {
            println!("{LINES}: no recording in the tree yet");
            return;
        };
        let mut wrong = Vec::new();
        let mut lit = [0usize; 3];
        for y in 0..LINE_SIZE[1] {
            for x in 0..LINE_SIZE[0] {
                let at = (x / LINE_CELL) as usize;
                let got = texels[(y * LINE_SIZE[0] + x) as usize];
                let viewport = line_viewport(at);
                let reach = |pixel: u32, axis: usize| {
                    let low = viewport.origin[axis];
                    let high = low + viewport.size[axis];
                    (low - 1.0..high + 1.0).contains(&(f64::from(pixel) + 0.5))
                };
                let inside = held(&viewport, x, y) == 1.0;
                let interior = (1.0..7.0).contains(&(f64::from(x) - viewport.origin[0]))
                    && (1.0..7.0).contains(&(f64::from(y) - viewport.origin[1]));
                let reached = reach(x, 0) && reach(y, 1);
                if got == LINE_COLORS[at] {
                    lit[at] += usize::from(inside);
                    if !reached || (at == 2 && interior) {
                        wrong.push(format!("({x}, {y}) holds draw {at}'s colour"));
                    }
                } else if got != [CLEAR; 4] {
                    wrong.push(format!("({x}, {y}) holds {got:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert!(
            lit[0] > 0 && lit[1] > 0,
            "a crossing triangle lit nothing: {lit:?}"
        );
    }

    /// **Every case of the family names itself once**, and is held to the family's terms.
    #[test]
    fn the_family_names_two_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(names, [FRACTIONAL, LINES]);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // No value of either case is loosened, and neither declares a notice: the line rule's
            // note is among the suite's native notes.
            assert!(case.allowances().is_empty(), "{}", case.name());
        }
        for size in [FRACTIONAL_SIZE, LINE_SIZE] {
            assert_eq!(layout(size).extent(), plane(size) as usize);
        }
    }
}
