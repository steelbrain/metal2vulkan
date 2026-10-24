//! **`renderTargetWidth` and `renderTargetHeight` smaller than the attachment**, crossed with the
//! load action, with whether the pass has a draw call at all, and with the viewport and scissor
//! that bound the draw: which texels of an attachment larger than its pass's render area a pass
//! changes, and to what.
//!
//! The two numbers were swept on the **descriptor** ([`crate::pass_descriptors`]'s number sweep
//! reads each of the pass's eight numbers at 0, 1, its documented limit and either side of it) and
//! no case ever rendered through a render area smaller than its attachment. Our runtime, meanwhile,
//! carries a whole measured model of what Metal does there (`metal2vulkan_runtime`'s
//! `clear_reach.rs`: 44 Metal cases on an M3 Max and an M4, every answer the same on both, each
//! with and without API validation), and every rule in it was reached by a probe and by nothing
//! else:
//!
//! - a **`Clear`** load of an attachment the pass stores clears it from the origin over the render
//!   area **rounded up to whole tiles of [`TILE`] pixels a side**, clipped to the attachment, in a
//!   pass **with a draw call** — one whose primitives cover nothing included;
//! - a pass of one layer **without a draw call** clears the whole level and slice, whatever its
//!   render area;
//! - a **`Load`** load keeps the texels past the render area.
//!
//! So the answer is a function of the render area, the load action and whether a draw call was
//! encoded **together**, and a case that fixes two of the three says nothing about the other cells.
//! `pass-bound-clear-reach` is the grid, ten cells, each named for what it alone shows.
//!
//! # The two cells that carry the most
//!
//! **`area-33x33` beside `area-32x32`.** One more pixel of render area in each axis pushes the
//! clear **eight** more pixels in each axis: 32 rounds up to 32 and 33 rounds up to 64, clipped to
//! the attachment's 40. So the first keeps the attachment's prior value past 32 and the second
//! clears the whole thing. A side that clipped the clear to the render area exactly, or that
//! rounded to some other tile, answers one of the two wrongly. Neither cell can show it alone —
//! `area-32x32` alone reads as "the clear stops at the render area" and `area-33x33` alone as "the
//! clear covers the attachment".
//!
//! **`draw-covers-nothing` beside `no-draw-call`.** Both passes leave every texel of the render
//! area at the clear and draw nothing anywhere, and they differ in the one thing Metal reads: a
//! pass with a draw call clears its render area's **tiles**, and a pass without one clears the
//! whole level. The draw in `draw-covers-nothing` is a real `drawPrimitives:` whose viewport and
//! scissor do not overlap, so its primitives cover no pixel. A side that decided "has a draw call"
//! by whether a draw *wrote* anything answers the two cells alike, and both of them wrongly.
//!
//! # What each axis is for, and the two pairs that turn out independent
//!
//! | cell | render area | load | draw |
//! |---|---|---|---|
//! | `whole-area/clear/drawing` | 40x40 | `Clear` | the scissored draw |
//! | `area-32x32/clear/drawing` | 32x32 | `Clear` | the scissored draw |
//! | `area-33x33/clear/drawing` | 33x33 | `Clear` | the scissored draw |
//! | `area-8x40/clear/drawing` | 8x40 | `Clear` | the scissored draw |
//! | `area-40x8/clear/drawing` | 40x8 | `Clear` | the scissored draw |
//! | `area-32x32/clear/no-draw-call` | 32x32 | `Clear` | none |
//! | `area-32x32/clear/draw-covers-nothing` | 32x32 | `Clear` | a draw whose viewport misses its scissor |
//! | `area-32x32/load/drawing` | 32x32 | `Load` | the scissored draw |
//! | `whole-area/load/drawing` | 40x40 | `Load` | the scissored draw |
//! | `area-32x32/load/no-draw-call` | 32x32 | `Load` | none |
//!
//! Every axis moves a plane: the render area between the five `Clear` drawing cells, the load
//! action between `area-32x32/clear/drawing` and `area-32x32/load/drawing`, and the draw between
//! `drawing`, `draw-covers-nothing` and `no-draw-call`. The `8x40` and `40x8` cells are there
//! because the rounding is **per axis**: a side that rounded the area's larger side, or took one
//! axis's reach for both, answers them wrongly and answers the square ones correctly.
//!
//! **Two pairs of cells are equal, and each equality is a claim rather than a gap.** Said here
//! rather than left to be noticed:
//!
//! - `whole-area/clear/drawing` and `area-33x33/clear/drawing` hold **one** plane, because 33
//!   rounds up past the attachment and is clipped back to it. That is the whole content of the
//!   `33` cell: the reach is the rounded area *clipped*, not the rounded area.
//! - `whole-area/load/drawing` and `area-32x32/load/drawing` hold **one** plane, because a `Load`
//!   load clears nothing, so the render area has nothing to bound. That is the independence claim:
//!   **the pass's bound does not move what a `Load` keeps, and it does not move the draw's own
//!   bounds either** — the drawn 8 by 8 is in the same place under a 32x32 area as under a 40x40
//!   one, so a side that read the viewport or the scissor in render-area coordinates rather than
//!   the attachment's fails it. A per-value sweep of `renderTargetWidth` cannot make either claim.
//!
//! # What this case can and cannot prove **on a device that renders through Metal**
//!
//! Said here because it was measured rather than reasoned, and because a reader who skips it will
//! over-read a green run on this Mac.
//!
//! Our runtime reaches Metal's reach by clearing it before the pass itself
//! (`clear_reach.rs`'s `ReachClears`) — but **on MoltenVK it would get the same answer without
//! doing so**, because the driver underneath is Metal and Metal's own tiles do it. Measured:
//! replacing [`reach`]'s rounding with the render area exactly, in the runtime, moves **not one
//! texel** of this grid on an M3 Max. So a green run here says the two legs agree and says
//! **nothing** about whether our reach model is right; the [`TILE`] of 32, the rounding up and the
//! clipping are still held by a probe alone on every non-Metal driver.
//!
//! What a green run here *does* say is measured too, by the falsification that does land: making
//! the render area ignore `renderTargetWidth` and `renderTargetHeight` diverges at exactly the
//! four cells whose area is smaller than the attachment **and** whose load is `Clear` **and** which
//! have a draw call, and leaves `whole-area/*`, `area-33x33` (whose reach is already the whole
//! attachment), `area-32x32/clear/no-draw-call` and every `Load` cell exact. So the two numbers
//! reach the device and bound the clear, per axis, and that is a claim only the pair can make: at
//! the whole-attachment value, or under a `Load` load, or without a draw call, the bound is
//! invisible.
//!
//! Closing the other half wants this case **replayed on a driver that is not Metal** (lavapipe,
//! RADV, ANV), where the runtime must synthesize the reach and the model is on the observable path.
//!
//! # Why every value here is defined on both sides
//!
//! A texel past the render area is only a comparable answer because the attachment's prior value
//! is **stated**: a priming pass of the same command buffer clears the whole attachment and
//! stores, so every one of the 1600 texels holds a known value before the test pass opens. Without
//! it a texel the test pass did not touch would be whatever the texture was made with, which is
//! not Metal's rule to compare.
//!
//! `DontCare` is **not** swept, in either position. A `DontCare` **load** of an attachment the pass
//! stores keeps its texels only in the tiles no primitive touched, and which tiles those are
//! depends on where the GPU places the vertices — our runtime reaches Metal's answer in the tiles
//! a draw reaches and announces a near-equivalent for the rest
//! (`Degradation::DontCareLoadTiles`), so a cell over it would compare a device's rasterizer. A
//! `DontCare` **store** leaves the attachment undefined on both sides.
//!
//! # Why this family is safe
//!
//! One command buffer: a priming pass and a test pass per cell over a 40 by 40 single-sampled
//! target, and one blit a cell. At most one draw of three vertices a pass, no loop in either stage,
//! **every viewport and every scissor inside its own cell's render area** (which is what keeps the
//! case off `setScissorRect:` past the render target, the one shape Metal's validation calls a
//! programming error), every vertex inside the clip volume, and no multisampling.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, ScissorBox,
    TargetSpec, ViewportRect,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{LoadAction, StoreAction};

/// The owned fixture: the fullscreen triangle at buffer 0's depth, and the fragment that returns
/// white.
const FIXTURE: &str = "rt6_resolves";

/// `rt6_resolve_full`: `float4(4 * a - 1, z, 1)` at each of three vertex ids.
const VERTEX: &str = "rt6_resolve_full";

/// `rt6_resolve_white`: `float4(1)`, whose red reaches the attachment as [`DRAWN`].
const FRAGMENT: &str = "rt6_resolve_white";

/// The side of the attachment, in texels.
///
/// **Not a round number on purpose**: a tile of [`TILE`] pixels does not divide it, so the reach's
/// rounding and its clipping to the attachment are two different things a cell can tell apart. A
/// 32- or 64-wide attachment could not.
const SIDE: u32 = 40;

/// The side, in pixels, of the tiles Metal's `Clear` load clears in past the render area, on both
/// axes: what `metal2vulkan_runtime`'s `clear_reach.rs` measured on an M3 Max and an M4 at every
/// format, sample count and tile size it tried.
///
/// It is stated here as the **number this family expects Metal to answer**, and the recording is
/// what says whether it does: the grid is built so that a different tile size moves at least one
/// cell's plane.
const TILE: u32 = 32;

/// The attachment's format: one `float` a texel, which a blit reads back as itself and which holds
/// every value this family states exactly.
const FORMAT: PixelFormat = PixelFormat::R32Float;

/// Bytes one texel takes.
const TEXEL: u32 = 4;

/// Bytes the attachment's whole level 0 takes.
const PLANE: u32 = TEXEL * SIDE * SIDE;

/// The value the priming pass leaves in every texel of the attachment, so a texel the test pass
/// did not touch is a **stated** answer rather than whatever the texture was made with.
const PRIOR: f64 = 0.25;

/// The value a test pass's `Clear` load writes.
const CLEAR: f64 = 0.5;

/// The red `rt6_resolve_white` returns, which is what a drawn texel holds.
const DRAWN: f64 = 1.0;

/// The depth the triangle carries, as the `float` its vertex buffer holds. The pass attaches no
/// depth, so any finite depth inside the clip volume draws the same.
const Z: f32 = 0.5;

/// The side of the rectangle the drawing cells' draw is confined to.
///
/// Eight, which is at or below **every** cell's render area on both axes, so one scissor and one
/// viewport serve every cell and no cell's draw can reach past its own bound.
const DRAW_SIDE: u32 = 8;

/// How many runs record mode takes a case of this family over.
const RUNS: u32 = 8;

/// What a cell's pass does about drawing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Draws {
    /// A `drawPrimitives:` whose viewport and scissor are the same 8 by 8 rectangle at the origin,
    /// so it writes exactly those pixels.
    Scissored,
    /// A `drawPrimitives:` whose viewport and scissor **do not overlap**, so its primitives cover
    /// no pixel: a draw call that draws nothing, which Metal counts as a draw call all the same.
    CoveringNothing,
    /// No draw call at all, which is the pass Metal clears whole.
    None,
}

/// One cell: the pass's stated render area, its load action, and what it does about drawing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    name: &'static str,
    area: [u32; 2],
    load: LoadAction,
    draws: Draws,
}

/// The ten cells, in the order the module's table states them.
const CELLS: [Cell; 10] = [
    Cell {
        name: "whole-area/clear/drawing",
        area: [SIDE, SIDE],
        load: LoadAction::Clear,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-32x32/clear/drawing",
        area: [32, 32],
        load: LoadAction::Clear,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-33x33/clear/drawing",
        area: [33, 33],
        load: LoadAction::Clear,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-8x40/clear/drawing",
        area: [8, SIDE],
        load: LoadAction::Clear,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-40x8/clear/drawing",
        area: [SIDE, 8],
        load: LoadAction::Clear,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-32x32/clear/no-draw-call",
        area: [32, 32],
        load: LoadAction::Clear,
        draws: Draws::None,
    },
    Cell {
        name: "area-32x32/clear/draw-covers-nothing",
        area: [32, 32],
        load: LoadAction::Clear,
        draws: Draws::CoveringNothing,
    },
    Cell {
        name: "area-32x32/load/drawing",
        area: [32, 32],
        load: LoadAction::Load,
        draws: Draws::Scissored,
    },
    Cell {
        name: "whole-area/load/drawing",
        area: [SIDE, SIDE],
        load: LoadAction::Load,
        draws: Draws::Scissored,
    },
    Cell {
        name: "area-32x32/load/no-draw-call",
        area: [32, 32],
        load: LoadAction::Load,
        draws: Draws::None,
    },
];

/// The rectangle the drawing cells write: the 8 by 8 at the origin.
const DRAWN_RECT: ScissorBox = ScissorBox::new(0, 0, DRAW_SIDE as u64, DRAW_SIDE as u64);

/// The scissor rectangle a `CoveringNothing` draw states: the same 8 by 8 at the origin, which its
/// viewport does not reach.
const MISSED_RECT: ScissorBox = DRAWN_RECT;

/// The viewport a `Scissored` draw states: the drawn rectangle itself, so the fullscreen triangle
/// lands exactly on it.
fn drawing_viewport() -> ViewportRect {
    ViewportRect::new([0.0, 0.0], [f64::from(DRAW_SIDE), f64::from(DRAW_SIDE)])
}

/// The viewport a `CoveringNothing` draw states: the 16 by 16 at (16, 16), which is **inside the
/// 32 by 32 render area** of every cell that states it and does not overlap [`MISSED_RECT`].
///
/// Inside the render area deliberately: a viewport or a scissor past it is not a shape this family
/// hands Metal, and the module's own test holds every cell to that.
fn missing_viewport() -> ViewportRect {
    ViewportRect::new([16.0, 16.0], [16.0, 16.0])
}

/// The extent from the origin a `Clear` load of this family's attachment clears in a pass over
/// `area` that has a draw call where `called`: Metal's rule as `clear_reach.rs` measured it — the
/// render area rounded up to whole [`TILE`]s and clipped to the attachment, or the whole attachment
/// in a pass of one layer without a draw call.
///
/// Stated here as a rule and applied to every cell, so the recording is held to a reason rather
/// than to itself.
fn reach(area: [u32; 2], called: bool) -> [u32; 2] {
    match called {
        false => [SIDE; 2],
        true => [0, 1].map(|axis| area[axis].div_ceil(TILE).saturating_mul(TILE).min(SIDE)),
    }
}

/// The attachment: a plain single-sampled render target, which is what a pass that only clears and
/// draws into it needs.
fn target() -> TargetSpec {
    TargetSpec::plain(FORMAT, [SIDE; 2])
}

/// The one colour state the pipeline holds.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(FORMAT))]
}

/// The pipeline: the fullscreen triangle and the white fragment over the one attachment.
fn pipeline_spec<'a>(states: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, states)
}

/// The layout the attachment's whole level 0 is read under: one `float` an element, so a diff names
/// the texel whose value moved.
fn layout() -> ByteLayout {
    let row_pitch = TEXEL * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The draw one triangle takes.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// The steps a cell's test pass encodes: nothing at all for [`Draws::None`], else the pipeline, the
/// depth the triangle carries, the viewport and scissor the cell states, and one triangle.
fn steps(draws: Draws, pipeline: Handle) -> Vec<RenderStep> {
    let (viewport, scissor) = match draws {
        Draws::None => return Vec::new(),
        Draws::Scissored => (drawing_viewport(), DRAWN_RECT),
        Draws::CoveringNothing => (missing_viewport(), MISSED_RECT),
    };
    vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::VertexBytes {
            bytes: Z.to_le_bytes().to_vec(),
            index: 0,
        },
        RenderStep::Viewport(viewport),
        RenderStep::Scissor(scissor),
        TRIANGLE,
    ]
}

/// The priming pass: the attachment cleared to [`PRIOR`] and stored, over **no stated render
/// area** and with no draw call, which Metal clears whole, so every texel holds a stated value
/// before a test pass opens.
fn priming(texture: Handle) -> RenderPass {
    RenderPass::clearing(vec![
        PassColor::cleared(texture, [PRIOR; 4]).stored(StoreAction::Store)
    ])
}

/// A cell's test pass: the attachment under the cell's load action, cleared to [`CLEAR`] and
/// stored, over the cell's stated render area, with the cell's own draw.
fn pass(cell: &Cell, texture: Handle, pipeline: Handle) -> RenderPass {
    let mut color = PassColor::cleared(texture, [CLEAR; 4]);
    color.load = cell.load;
    color.store = StoreAction::Store;
    let mut built = RenderPass::clearing(vec![color]);
    built.size = Some(cell.area);
    built.drawing(steps(cell.draws, pipeline))
}

/// The load action's enumerator name, which is also what a recording's site names hold.
fn load_name(load: LoadAction) -> &'static str {
    match load {
        LoadAction::DontCare => "MTLLoadActionDontCare",
        LoadAction::Load => "MTLLoadActionLoad",
        LoadAction::Clear => "MTLLoadActionClear",
        _ => "MTLLoadActionUnknown",
    }
}

/// The shader reference both texts of the fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: format!("{VERTEX}+{FRAGMENT}"),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(BoundReach)]
}

/// The grid: ten cells of render area by load action by draw.
pub struct BoundReach;

impl Case for BoundReach {
    fn name(&self) -> &'static str {
        "pass-bound-clear-reach"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // **Nothing is loosened and nothing is declared.** Every texel Metal leaves is one we must
        // leave, every attachment of every pass here is stored with `Store`, and no cell states a
        // `DontCare` load, which is the one shape of this subject our runtime reaches by a
        // near-equivalent. A case that declared a notice it cannot provoke would be a case nobody
        // can read.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let states = colors();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &pipeline_spec(&states).words()),
            Input::bytes("target", &target().words()),
            Input::bytes("depth", &Z.to_le_bytes()),
        ];
        // The three values a texel can hold, by their own bits, so one of them moving by an `f64`
        // step is a change to the case's inputs rather than a quiet pass against another grid.
        let mut values = Vec::new();
        for value in [PRIOR, CLEAR, DRAWN] {
            values.extend(value.to_bits().to_le_bytes());
        }
        inputs.push(Input::bytes("values", &values));
        // **The reach each cell expects**, by the rule above, so a change to the rule this family
        // holds Metal to is a change to the case's key rather than a quiet pass against another
        // grid's answer. It is an input and not an observation: the recording's bytes are what
        // Metal answered, and this is what the case claimed before asking.
        let mut reaches = Vec::new();
        for cell in CELLS {
            for axis in reach(cell.area, cell.draws != Draws::None) {
                reaches.extend(axis.to_le_bytes());
            }
        }
        inputs.push(Input::bytes("reach", &reaches));
        inputs.push(Input::bytes("prime", &priming(Handle(0)).words()));
        for cell in CELLS {
            inputs.push(Input::bytes(
                format!("pass:{}", cell.name),
                &pass(&cell, Handle(0), Handle(0)).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let states = colors();
        let pipeline = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &pipeline_spec(&states)),
        )?;

        // One attachment and one read-back buffer a cell, so no cell's answer can be another
        // cell's leftovers and each cell's priming pass is its own.
        let mut targets = Vec::new();
        let mut reads = Vec::new();
        for cell in CELLS {
            targets.push(made(
                sites,
                &format!("target:{}", cell.name),
                side.target(&target()),
            )?);
            reads.push(made(
                sites,
                &format!("buffer:read:{}", cell.name),
                side.zeroed_buffer(PLANE as usize),
            )?);
        }

        // One command buffer: a priming pass, a test pass and a blit a cell, in that order, so the
        // whole grid is one submission on both sides.
        let mut work = Vec::new();
        let mut names = Vec::new();
        for (at, cell) in CELLS.iter().enumerate() {
            work.push(RenderWork::Pass(priming(targets[at])));
            names.push(format!("prime:{}", cell.name));
            work.push(RenderWork::Pass(pass(cell, targets[at], pipeline)));
            names.push(format!("pass:{}", cell.name));
            work.push(RenderWork::ReadTexture {
                texture: targets[at],
                into: reads[at],
                offset: 0,
                row_pitch: TEXEL * SIDE,
                size: [SIDE; 2],
            });
            names.push(format!("read:{}", cell.name));
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let mut pieces = Node::branch("pieces", Vec::new());
        for (at, what) in names.iter().enumerate() {
            pieces = pieces.with(Node::leaf(
                what.clone(),
                Observed::value(match outcome.refused.get(at).and_then(Option::as_ref) {
                    Some(failure) => failure.recorded(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        sites.push(pieces);
        if let Some((what, failure)) = names
            .iter()
            .enumerate()
            .find_map(|(at, what)| outcome.refused.get(at)?.as_ref().map(|f| (what, f)))
        {
            return Err(Abort {
                at: format!("pieces/{what}"),
                failure: failure.clone(),
            });
        }

        let mut planes = Node::branch("planes", Vec::new());
        for (at, cell) in CELLS.iter().enumerate() {
            let bytes = call(
                sites,
                &format!("read-back/{}", cell.name),
                side.read_buffer(reads[at], 0, PLANE as usize),
                |_| Value::Absent,
            )?;
            planes = planes.with(Node::leaf(
                format!("{}/{}", cell.name, load_name(cell.load)),
                Observed::bytes(bytes, layout()),
            ));
        }
        sites.push(planes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::collections::{BTreeMap, BTreeSet};

    /// What the texel at `(x, y)` must hold after `cell`'s pass: the drawn value where the cell's
    /// draw reached it, else the clear where a `Clear` load's reach covers it, else the attachment's
    /// prior value.
    ///
    /// Metal's rule applied here rather than read back from a recording, which is what lets the
    /// recording be held to a reason.
    fn expected(cell: &Cell, x: u32, y: u32) -> f32 {
        let drawn = cell.draws == Draws::Scissored
            && x < DRAWN_RECT.width as u32
            && y < DRAWN_RECT.height as u32;
        if drawn {
            return DRAWN as f32;
        }
        let called = cell.draws != Draws::None;
        let reached = reach(cell.area, called);
        match cell.load == LoadAction::Clear && x < reached[0] && y < reached[1] {
            true => CLEAR as f32,
            false => PRIOR as f32,
        }
    }

    /// The whole plane `cell` must leave, as the bits of its 1600 texels.
    fn plane(cell: &Cell) -> Vec<u32> {
        (0..SIDE * SIDE)
            .map(|at| expected(cell, at % SIDE, at / SIDE).to_bits())
            .collect()
    }

    /// **The three values a texel can hold are exact and distinct**, so every region of every plane
    /// names which of the three put it there.
    #[test]
    fn the_three_values_are_exact_and_distinct() {
        let bits: BTreeSet<u32> = [PRIOR, CLEAR, DRAWN]
            .iter()
            .map(|value| {
                assert!((0.0..=1.0).contains(value), "{value} is outside [0, 1]");
                // A multiple of 2^-2 is exact in a `float` and in an `R32Float` texel alike, so
                // the round trip through the attachment changes nothing.
                assert_eq!(value * 4.0, (value * 4.0).round(), "{value} is not exact");
                (*value as f32).to_bits()
            })
            .collect();
        assert_eq!(bits.len(), 3, "two of the three values are the same");
        assert!((0.0..1.0).contains(&Z));
    }

    /// **The reach rule is the one `clear_reach.rs` measured**, read at the values this grid uses
    /// and at the edges either side of a tile.
    ///
    /// Written as its own test because every cell's expectation rests on it: if this is not Metal's
    /// rule the recording will say so, and this is what names the rule the recording is held to.
    #[test]
    fn the_reach_rounds_up_to_a_whole_tile_and_clips_to_the_attachment() {
        assert_eq!(TILE, 32);
        assert!(
            SIDE % TILE != 0,
            "a tile divides the attachment, so no cell can tell the rounding from the clipping"
        );
        // Rounded up to a whole tile, per axis, and clipped to the attachment.
        assert_eq!(reach([32, 32], true), [32, 32]);
        assert_eq!(reach([33, 33], true), [SIDE, SIDE]);
        assert_eq!(reach([8, SIDE], true), [32, SIDE]);
        assert_eq!(reach([SIDE, 8], true), [SIDE, 32]);
        assert_eq!(reach([SIDE, SIDE], true), [SIDE, SIDE]);
        // One pixel either side of the tile edge, which is what makes the 32/33 pair sharp.
        assert_eq!(reach([31, 31], true), [32, 32]);
        assert_eq!(reach([1, 1], true), [32, 32]);
        // And a pass of one layer without a draw call clears the whole level, whatever its area.
        for area in [[1, 1], [32, 32], [33, 33], [SIDE, SIDE]] {
            assert_eq!(reach(area, false), [SIDE, SIDE], "{area:?}");
        }
    }

    /// **Every axis of the grid moves a plane, and every cell the module names as equal to another
    /// really is.**
    ///
    /// Ten cells, eight distinct planes, and the two equalities are the module's two stated
    /// claims. A sweep that could not fire is a sweep that compares nothing many times, so this is
    /// held with no device.
    #[test]
    fn each_axis_moves_a_plane_and_the_two_equalities_are_the_stated_ones() {
        let by_plane: BTreeMap<Vec<u32>, Vec<&str>> =
            CELLS.iter().fold(BTreeMap::new(), |mut held, cell| {
                held.entry(plane(cell)).or_default().push(cell.name);
                held
            });
        assert_eq!(
            by_plane.len(),
            8,
            "the ten cells answer {} planes, not the eight the module states: {:?}",
            by_plane.len(),
            by_plane.values().collect::<Vec<_>>()
        );
        let equal: BTreeSet<Vec<&str>> = by_plane
            .values()
            .filter(|names| names.len() > 1)
            .cloned()
            .collect();
        assert_eq!(
            equal,
            [
                vec!["whole-area/clear/drawing", "area-33x33/clear/drawing"],
                vec!["area-32x32/load/drawing", "whole-area/load/drawing"],
            ]
            .into_iter()
            .collect::<BTreeSet<Vec<&str>>>(),
            "the cells that hold one plane are not the two pairs the module names"
        );
        // The render-area axis, under `Clear` with a draw: the five areas answer four planes, and
        // the fifth is the 33 cell the module names as equal to the whole-area one.
        let areas: BTreeSet<Vec<u32>> = CELLS
            .iter()
            .filter(|cell| cell.load == LoadAction::Clear && cell.draws == Draws::Scissored)
            .map(plane)
            .collect();
        assert_eq!(areas.len(), 4);
        // The per-axis rounding: the 8x40 and 40x8 cells answer two different planes, so a side
        // that took one axis's reach for both fails one of them.
        let wide = CELLS.iter().find(|c| c.name == "area-8x40/clear/drawing");
        let tall = CELLS.iter().find(|c| c.name == "area-40x8/clear/drawing");
        assert_ne!(
            plane(wide.expect("the wide cell")),
            plane(tall.expect("the tall cell"))
        );
        // The draw axis under one area and one load: `drawing`, `draw-covers-nothing` and
        // `no-draw-call` are three planes, which is the pair the module leans on hardest.
        let draws: BTreeSet<Vec<u32>> = CELLS
            .iter()
            .filter(|cell| cell.area == [32, 32] && cell.load == LoadAction::Clear)
            .map(plane)
            .collect();
        assert_eq!(
            draws.len(),
            3,
            "the three draw shapes do not answer three planes"
        );
        // And the load axis under one area and one draw: two planes.
        let loads: BTreeSet<Vec<u32>> = CELLS
            .iter()
            .filter(|cell| cell.area == [32, 32] && cell.draws == Draws::Scissored)
            .map(plane)
            .collect();
        assert_eq!(loads.len(), 2);
    }

    /// **Every viewport and every scissor of every cell is inside that cell's own render area.**
    ///
    /// This is the guard that keeps the family off `setScissorRect:` past the render target, the
    /// one shape of that call Metal's own validation calls a programming error. Decided from the
    /// cells' own numbers, so it is reached with no device, and it is held against the render area
    /// rather than the attachment, which is the stricter of the two.
    #[test]
    fn every_viewport_and_scissor_is_inside_its_cells_render_area() {
        // The check sees a rectangle that reaches past a bound, so a green run means the cells are
        // inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, 33, 1).past([32, 32]));
        for cell in CELLS {
            assert!(
                cell.area[0] <= SIDE && cell.area[1] <= SIDE,
                "{}: the render area reaches past the attachment",
                cell.name
            );
            for step in steps(cell.draws, Handle(0)) {
                match step {
                    RenderStep::Scissor(rect) => {
                        assert!(
                            !rect.past(cell.area),
                            "{}: the scissor {rect:?} reaches past the render area {:?}",
                            cell.name,
                            cell.area
                        );
                        assert!(!rect.past([SIDE; 2]), "{}", cell.name);
                    }
                    RenderStep::Viewport(view) => {
                        for axis in [0, 1] {
                            let far = view.origin[axis] + view.size[axis];
                            assert!(
                                view.origin[axis] >= 0.0 && far <= f64::from(cell.area[axis]),
                                "{}: the viewport reaches {far} on axis {axis}, past the render \
                                 area {:?}",
                                cell.name,
                                cell.area
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        // A `CoveringNothing` draw's viewport really does miss its scissor, or the cell would draw
        // after all and would say nothing about a draw call that covers no pixel.
        let view = missing_viewport();
        assert!(
            view.origin[0] >= MISSED_RECT.width as f64
                || view.origin[1] >= MISSED_RECT.height as f64,
            "the missing viewport overlaps the scissor it is supposed to miss"
        );
        // And a `Scissored` draw's viewport is exactly its scissor, so the drawn region is the
        // rectangle the expectation names.
        let drawing = drawing_viewport();
        assert_eq!(drawing.origin, [0.0, 0.0]);
        assert_eq!(
            drawing.size,
            [DRAWN_RECT.width as f64, DRAWN_RECT.height as f64]
        );
    }

    /// **The grid names its ten cells apart**, states every pass's render area, stores every
    /// attachment, and encodes a draw call exactly where the cell says.
    #[test]
    fn the_grid_states_ten_cells_apart() {
        let names: BTreeSet<&str> = CELLS.iter().map(|cell| cell.name).collect();
        assert_eq!(names.len(), CELLS.len(), "two cells share a name");
        let sites: BTreeSet<String> = CELLS
            .iter()
            .map(|cell| format!("{}/{}", cell.name, load_name(cell.load)))
            .collect();
        assert_eq!(sites.len(), CELLS.len(), "two cells share a site name");
        assert!(!sites.iter().any(|site| site.ends_with("Unknown")));
        for cell in CELLS {
            let built = pass(&cell, Handle(0), Handle(0));
            assert_eq!(
                built.size,
                Some(cell.area),
                "{}: the pass does not state its render area",
                cell.name
            );
            assert_eq!(built.colors.len(), 1);
            // Every attachment is stored, so no texel of any cell is undefined, and no cell loads
            // `DontCare`, which is the one shape of this subject reached by a near-equivalent.
            assert_eq!(built.colors[0].store, StoreAction::Store);
            assert_ne!(built.colors[0].load, LoadAction::DontCare);
            assert_eq!(built.colors[0].resolve, None);
            let drew = built
                .steps
                .iter()
                .any(|step| matches!(step, RenderStep::Draw { .. }));
            assert_eq!(
                drew,
                cell.draws != Draws::None,
                "{}: the pass encodes a draw call where the cell says it does not",
                cell.name
            );
        }
        // The priming pass clears the whole attachment and stores it, with no stated area and no
        // draw, which is the pass Metal clears whole.
        let primed = priming(Handle(0));
        assert_eq!(primed.size, None);
        assert!(primed.steps.is_empty());
        assert_eq!(primed.colors[0].load, LoadAction::Clear);
        assert_eq!(primed.colors[0].store, StoreAction::Store);
        // The case loosens nothing at all.
        for case in all() {
            assert!(case.allowances().is_empty());
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
        }
        assert_eq!(layout().extent(), PLANE as usize);
        assert_eq!(target().samples, 1);
    }

    /// The one outcome of the case's recording, where the tree holds one.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so the case's answer moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// **Metal's recorded answer is the reach rule's, at every one of the 16,000 texels of the
    /// grid.**
    ///
    /// This is where the rule the runtime holds as a literal — the tile of 32 pixels, the rounding
    /// up, the clipping, and a draw call that draws nothing still counting as one — is put beside
    /// Metal by a case for the first time.
    #[test]
    fn metal_cleared_each_cells_reach_and_kept_the_rest() {
        let case = "pass-bound-clear-reach";
        let Some(transcript) = only_outcome(case) else {
            println!("{case}: no recording in the tree yet");
            return;
        };
        for cell in CELLS {
            let site = format!("{}/{}", cell.name, load_name(cell.load));
            let node = transcript
                .root
                .child("planes")
                .and_then(|node| node.child(&site))
                .unwrap_or_else(|| panic!("{case}: no planes/{site} site"));
            let Value::Bytes(bytes) = &node.observed.value else {
                panic!("{case}: planes/{site} holds no bytes");
            };
            assert_eq!(bytes.len(), PLANE as usize);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = ((y * SIDE + x) * TEXEL) as usize;
                    let mut word = [0u8; 4];
                    word.copy_from_slice(&bytes[at..at + 4]);
                    let held = f32::from_le_bytes(word);
                    let want = expected(&cell, x, y);
                    assert_eq!(
                        held.to_bits(),
                        want.to_bits(),
                        "{site}: texel ({x}, {y}) holds {held}, not {want}"
                    );
                }
            }
        }
    }
}
