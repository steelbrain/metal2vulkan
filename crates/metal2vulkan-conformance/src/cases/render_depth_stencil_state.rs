//! What a render encoder's **depth and stencil state** does to the draw that follows it: the depth
//! bias and its slope scale and clamp, the depth clip mode, and the two faces' stencil reference
//! values, together and apart.
//!
//! These are the encoder's state setters whose effect is a **test** rather than a place: nothing
//! about the geometry moves, and whether a fragment survives is the whole answer. So each case
//! here fills a depth or a stencil attachment first and then draws the same triangle into one
//! cell after another, changing one setting between each pair, and the target records which draw's
//! fragments got through.
//!
//! | case | what it sets, and what the target then says |
//! |---|---|
//! | `render-state-depth-bias` | `setDepthBias:slopeScale:clamp:` over a flat triangle at the depth already stored, and over a sloped one with and without a slope scale and a clamp, under `MTLCompareFunctionLess` |
//! | `render-state-depth-clip` | `setDepthClipMode:` over a triangle in front of the near plane, which `Clip` discards and `Clamp` draws |
//! | `render-state-stencil-references` | `setStencilReferenceValue:` and `setStencilFrontReferenceValue:backReferenceValue:` under a state whose two faces both compare `Equal` |
//!
//! **The depth prefill is a draw of the same pass**, under a pipeline whose colour write mask is
//! none and a state that writes depth and tests nothing. That matters more than it looks: the
//! depth the later draws are tested against is then a depth this pass wrote rather than a clear
//! value, so a side that programmed the bias into the write instead of the test, or lost the
//! prefill's depth, is told apart from Metal. It writes no colour at all, so the colour target is
//! still the clear everywhere the region draws did not reach.
//!
//! **Every expectation is arithmetic this module states.** The colour target is `RGBA32Float` and
//! the fragment function returns `float4(1.0)`, so a texel is either four ones or the pass's
//! clear, and which one it is follows from the depths the module writes down:
//!
//! - the flat triangle sits at exactly the stored depth, so `Less` fails and a constant bias of
//!   one minimum resolvable difference is what makes it pass;
//! - the sloped triangle's depth crosses the stored depth in the middle of its cell, and its slope
//!   is one eightieth of the depth range a pixel, so a slope scale of minus two moves the row the
//!   crossing lands on by exactly two, and a clamp of one row's worth of depth moves it by one;
//! - the near triangle is in front of the near plane, so `Clip` has nothing to draw and `Clamp`
//!   has the near plane's own depth, which is less than the stored one.
//!
//! The stencil map is stated **without assuming Metal's facing convention**: it reads one cell to
//! learn which of the two triangles is the front-facing one and holds the other nine to the
//! consequences, each of which is falsifiable whichever way the convention goes.
//!
//! Nothing here can fault a GPU: one pass a case, at most eleven draws of three vertices over a 16
//! by 16 target, no loop in either fixture, every scissor rectangle inside the target, and no bias
//! larger in magnitude than two.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, DepthBiasValues, PassColor, PassDepth, PassStencil, Plane, ReadAt, RenderPass,
    RenderPipelineSpec, RenderStep, RenderWork, ScissorBox, StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, DepthClipMode, PixelFormat, PrimitiveType, StencilOperation,
};
use metal2vulkan_runtime::{Degradation, StoreAction};

/// The colour target's format: four ones or the clear, nothing in between.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`FORMAT`] takes.
const TEXEL: u32 = 16;

/// The side of every target here, in texels.
const SIDE: u32 = 16;

/// Bytes the colour target's whole level 0 takes.
const PLANE: u32 = TEXEL * SIDE * SIDE;

/// The clear the colour target is loaded with: a value the fragment function cannot return, so a
/// cell nothing drew into is read rather than inferred.
const CLEAR: f32 = -7.5;

/// The depth format the depth cases attach.
const DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// The stencil format the reference case attaches.
const STENCIL_FORMAT: PixelFormat = PixelFormat::Stencil8;

/// The depth the pass's own first draw writes over the whole target, and the depth every later
/// draw of a depth case is tested against. An exact binary fraction, well inside `[0, 1]`.
const STORED_DEPTH: f64 = 0.5;

/// The depth the depth attachment is cleared to before the prefill draw: the far plane, so a
/// prefill that wrote nothing leaves a depth every later draw would pass, which is what makes the
/// first cell's clear a reading of the prefill as well as of the bias.
const CLEARED_DEPTH: f64 = 1.0;

/// The stencil the stencil attachment is cleared to, and the reference a passing draw states.
const STORED_STENCIL: u32 = 7;

/// A reference no draw's stencil equals, so a draw that states it is discarded.
const OTHER_STENCIL: u32 = 3;

/// The owned fixture the depth cases' two stages come from: each vertex where buffer 0 holds it,
/// its depth included, and a fragment function that writes white.
const DEPTH_FIXTURE: &str = "cfb_sample_depth";

/// `cfb_depth_vertex`: the `float4` buffer 0 holds for the vertex, unchanged.
const DEPTH_VERTEX: &str = "cfb_depth_vertex";

/// `cfb_depth_white`: `float4(1.0)`.
const WHITE: &str = "cfb_depth_white";

/// The entry points the depth cases' shader reference names.
const DEPTH_ENTRIES: &str = "cfb_depth_vertex+cfb_depth_white";

/// The owned fixture the stencil case's two stages come from.
const STENCIL_FIXTURE: &str = "tob_renders";

/// `tob_triangle`: the triangle of the three points a buffer holds.
const STENCIL_VERTEX: &str = "tob_triangle";

/// `tob_solid`: the `float4` a fragment buffer holds, unchanged.
const SOLID: &str = "tob_solid";

/// The entry points the stencil case's shader reference names.
const STENCIL_ENTRIES: &str = "tob_triangle+tob_solid";

/// The draw one triangle takes: three vertices, one instance.
const TRIANGLE_DRAW: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How many runs record mode takes a case of this module over, and why eight rather than three:
/// the same reason `render_state`'s cases ask for eight. One pass, one queue, nothing ordered by
/// anything a device may reorder, so the runs are expected to fold into one outcome and the
/// recording says so on evidence. A depth test whose answer moved between runs would be the
/// finding.
const RUNS: u32 = 8;

/// The 4 by 4 cell at column `c`, row `r` of a 16 by 16 target.
const fn cell(c: u32, r: u32) -> ScissorBox {
    ScissorBox::new((4 * c) as u64, (4 * r) as u64, 4, 4)
}

/// The colour target: private, of `MTLTextureUsageRenderTarget` alone.
fn color_target() -> TargetSpec {
    TargetSpec::plain(FORMAT, [SIDE; 2])
}

/// The layout the colour target's whole level 0 is read under.
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
        row_elements: 4 * SIDE,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The piece that copies the colour target's level 0 into a buffer the host reads.
fn read_target(texture: Handle, into: Handle) -> RenderWork {
    RenderWork::ReadTexture {
        texture,
        into,
        offset: 0,
        row_pitch: TEXEL * SIDE,
        size: [SIDE; 2],
        plane: Plane::Whole,
        at: ReadAt::BASE,
    }
}

/// The shader reference both texts of `fixture` give, under `entries`.
/// The notices every row of the family declares, each measured firing for all three rows.
///
/// None of them loosens a value: every one sits at the notices site, so a texel that differs
/// still fails the case.
fn pass_notices() -> Vec<Allowance> {
    vec![
        // The depth attachment nothing reads is stored `DontCare`, and a device whose `DontCare`
        // store is not Metal's announces that it took the near-equivalent. Measured on llvmpipe
        // and on the RTX 5080.
        crate::replay::notice_allowance(Degradation::DontCareStore.code()),
        // A device that clamps a depth bias outside the unit range where Metal does not says so.
        // Measured on llvmpipe.
        crate::replay::notice_allowance(Degradation::BiasedDepthPastUnitRange.code()),
        // The buffer each pass reads its colour target back through is `Shared`
        // ([`crate::replay::shared_memory_notice`], measured on the 5080).
        crate::replay::shared_memory_notice(),
    ]
}

fn shader_refs(fixture: &'static str, entries: &str) -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(fixture);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{fixture}{extension}"),
                entry: entries.to_owned(),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// The two inputs `fixture`'s own texts are.
fn fixture_inputs(fixture: &'static str) -> Result<Vec<Input>, String> {
    let held = Fixture::new(fixture);
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
        },
    ])
}

/// The library of `fixture`.
fn library(side: &mut dyn Side, sites: &mut Sites, fixture: &'static str) -> Result<Handle, Abort> {
    let loaded = Fixture::new(fixture).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    made(sites, "library", side.library(&loaded.source()))
}

/// The words a `MTLDepthStencilDescriptor` fingerprints as: the call's own fields, each by its raw
/// value.
///
/// Each case states its own, as the suite's other depth and stencil cases do:
/// [`DepthStencilDescriptor`] is a `#[non_exhaustive]` public type of another crate, so a case
/// cannot destructure it and a helper here would be the only spelling to keep in step with Metal's
/// own.
fn state_words(state: &DepthStencilDescriptor) -> Vec<u8> {
    let mut bytes = state.depth_compare_function.as_raw().to_le_bytes().to_vec();
    bytes.push(u8::from(state.depth_write_enabled));
    for face in [&state.front_face_stencil, &state.back_face_stencil] {
        bytes.extend(face.stencil_compare_function.as_raw().to_le_bytes());
        bytes.extend(face.stencil_failure_operation.as_raw().to_le_bytes());
        bytes.extend(face.depth_failure_operation.as_raw().to_le_bytes());
        bytes.extend(face.depth_stencil_pass_operation.as_raw().to_le_bytes());
        bytes.extend(face.read_mask.to_le_bytes());
        bytes.extend(face.write_mask.to_le_bytes());
    }
    bytes
}

/// The state the prefill draw sets: write the depth, test nothing.
fn writing_state() -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state
}

/// The state every region draw of a depth case sets: keep a fragment whose depth is less than the
/// stored one, and write nothing, so one draw's bias cannot reach the next draw's test.
fn testing_state() -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Less;
    state.depth_write_enabled = false;
    state
}

/// The state the stencil case sets: both faces keep a fragment whose stencil equals the face's
/// reference, and no operation writes the stencil, so each cell's draw is independent of every
/// other's.
fn equal_state() -> DepthStencilDescriptor {
    let mut face = StencilDescriptor::default();
    face.stencil_compare_function = CompareFunction::Equal;
    face.stencil_failure_operation = StencilOperation::Keep;
    face.depth_failure_operation = StencilOperation::Keep;
    face.depth_stencil_pass_operation = StencilOperation::Keep;
    let mut state = DepthStencilDescriptor::default();
    state.front_face_stencil = face;
    state.back_face_stencil = face;
    state
}

/// Run `work` as one command buffer, emit what each piece answered, and read the colour target's
/// texels back.
fn drawn(
    side: &mut dyn Side,
    sites: &mut Sites,
    work: &[RenderWork],
    read_back: Handle,
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
        side.read_buffer(read_back, 0, PLANE as usize),
        |_| Value::Absent,
    )?;
    sites.push(
        Node::branch("drawn", Vec::new())
            .with(Node::leaf("texels", Observed::bytes(bytes, layout()))),
    );
    Ok(())
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(DepthBias),
        Box::new(DepthClip),
        Box::new(StencilReferences),
    ]
}

// ------------------------------------------------------------------------------------------------
// The depth cases' geometry

/// How far apart the triangles lie in the positions buffer: three `float4`s each.
const POSITION_STRIDE: u64 = 48;

/// The clip-space corners every triangle of the depth cases takes, which cover the whole clip
/// square and more, so what confines a draw is the scissor rectangle the case set.
const CORNERS: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

/// The depth the sloped triangle carries at each corner: `0.5 + 0.1 * y`, so its depth crosses
/// [`STORED_DEPTH`] at the middle of the target and its slope along the window's rows is one
/// eightieth of the depth range, which is the `m` Metal's slope scale multiplies.
const SLOPED_DEPTHS: [f32; 3] = [0.4, 0.4, 0.8];

/// The four triangles the depth cases' positions buffer holds, by index: the flat one at the
/// stored depth, the sloped one, the one in front of the near plane, and one plainly nearer than
/// the stored depth.
const FLAT: usize = 0;
const SLOPED: usize = 1;
const NEAR: usize = 2;
const MID: usize = 3;

/// The depth each of those triangles carries at each corner.
fn triangle_depths(which: usize) -> [f32; 3] {
    match which {
        SLOPED => SLOPED_DEPTHS,
        NEAR => [-0.5; 3],
        MID => [0.25; 3],
        _ => [STORED_DEPTH as f32; 3],
    }
}

/// The positions buffer the depth cases bind at vertex buffer 0.
fn positions_buffer() -> Vec<u8> {
    let mut bytes = Vec::new();
    for which in [FLAT, SLOPED, NEAR, MID] {
        let depths = triangle_depths(which);
        for (corner, depth) in CORNERS.iter().zip(depths) {
            for word in [corner[0], corner[1], depth, 1.0] {
                bytes.extend(word.to_le_bytes());
            }
        }
    }
    bytes
}

/// The steps that write [`STORED_DEPTH`] over the whole target without touching a colour: the
/// pipeline whose write mask is none, the state that writes depth and tests nothing, the flat
/// triangle, and one draw with no scissor rectangle of its own.
fn prefill_steps(prefill: Handle, writing: Handle, positions: Handle) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(prefill),
        RenderStep::DepthStencilState(writing),
        RenderStep::VertexBuffer {
            buffer: positions,
            offset: FLAT as u64 * POSITION_STRIDE,
            index: 0,
        },
        TRIANGLE_DRAW,
    ]
}

// ------------------------------------------------------------------------------------------------
// render-state-depth-bias

/// The cell each flat-triangle bias draw is confined to, and the bias it sets.
fn bias_cells() -> [(ScissorBox, DepthBiasValues); 3] {
    [
        // No bias: the flat triangle sits at exactly the stored depth, so `Less` fails.
        (cell(0, 0), DepthBiasValues::new(0.0, 0.0, 0.0)),
        // One minimum resolvable difference nearer: `Less` passes.
        (cell(1, 0), DepthBiasValues::new(-1.0, 0.0, 0.0)),
        // The bias set back to none, which the first cell's answer is the control for.
        (cell(2, 0), DepthBiasValues::new(0.0, 0.0, 0.0)),
    ]
}

/// The cell the sloped triangle is drawn into with no slope scale, eight rows tall so the row its
/// depth crosses [`STORED_DEPTH`] on is inside it.
const SLOPE_PLAIN: ScissorBox = ScissorBox::new(0, 4, 4, 8);

/// The cell the sloped triangle is drawn into under a slope scale of [`SLOPE_SCALE`].
const SLOPE_SCALED: ScissorBox = ScissorBox::new(4, 4, 4, 8);

/// The cell the sloped triangle is drawn into under the same slope scale and a clamp that bounds
/// its offset to half of it.
const SLOPE_HELD: ScissorBox = ScissorBox::new(8, 4, 4, 8);

/// The slope scale the second and third sloped draws set: minus two, which is two pixel rows'
/// worth of this triangle's depth, so it moves the row the crossing lands on by exactly two.
const SLOPE_SCALE: f32 = -2.0;

/// The clamp the third sloped draw states: negative, so it bounds a negative offset, and one pixel
/// row's worth of depth, so the bounded offset moves the crossing by one row instead of two.
///
/// **It is the clamp the case states because it is the one that can be read.** A clamp that bounds
/// a *constant* bias cannot be: measured on an Apple M3 Max (macOS 27.0.1, build 26A434), a bias
/// of -1 under a clamp of -1e-9 drew exactly as the same bias under no clamp did, over eight runs.
/// One minimum resolvable difference of a `Depth32Float` depth near 0.5 is `2^-24`, and -1e-9 is
/// far below it, so a depth bounded to -1e-9 is the stored depth once a `float` holds it -- and it
/// passed `Less` anyway, which reads as the comparison seeing more precision than the attachment
/// stores. A clamp whose own magnitude is a fortieth of the depth range leaves no room for that
/// reading, and it is the form a program actually uses: a bound on a slope-scaled offset.
const HELD_CLAMP: f32 = -0.0125;

/// `setDepthBias:slopeScale:clamp:` over a flat triangle at the stored depth and a sloped one that
/// crosses it.
pub struct DepthBias;

impl DepthBias {
    /// The pipelines: the depth-only prefill and the colour-writing region draws.
    fn pipelines(
        masked: &'static [Option<ColorState>],
        plain: &'static [Option<ColorState>],
    ) -> [RenderPipelineSpec<'static>; 2] {
        [
            RenderPipelineSpec::new(DEPTH_VERTEX, WHITE, masked)
                .depth_stencil(Some(DEPTH_FORMAT), None),
            RenderPipelineSpec::new(DEPTH_VERTEX, WHITE, plain)
                .depth_stencil(Some(DEPTH_FORMAT), None),
        ]
    }

    /// The pass's steps.
    fn steps(
        prefill: Handle,
        coloring: Handle,
        writing: Handle,
        testing: Handle,
        positions: Handle,
    ) -> Vec<RenderStep> {
        let mut steps = prefill_steps(prefill, writing, positions);
        steps.push(RenderStep::Pipeline(coloring));
        steps.push(RenderStep::DepthStencilState(testing));
        for (box_of, bias) in bias_cells() {
            steps.push(RenderStep::DepthBias(bias));
            steps.push(RenderStep::Scissor(box_of));
            steps.push(TRIANGLE_DRAW);
        }
        steps.push(RenderStep::VertexBuffer {
            buffer: positions,
            offset: SLOPED as u64 * POSITION_STRIDE,
            index: 0,
        });
        steps.push(RenderStep::DepthBias(DepthBiasValues::new(0.0, 0.0, 0.0)));
        steps.push(RenderStep::Scissor(SLOPE_PLAIN));
        steps.push(TRIANGLE_DRAW);
        steps.push(RenderStep::DepthBias(DepthBiasValues::new(
            0.0,
            SLOPE_SCALE,
            0.0,
        )));
        steps.push(RenderStep::Scissor(SLOPE_SCALED));
        steps.push(TRIANGLE_DRAW);
        steps.push(RenderStep::DepthBias(DepthBiasValues::new(
            0.0,
            SLOPE_SCALE,
            HELD_CLAMP,
        )));
        steps.push(RenderStep::Scissor(SLOPE_HELD));
        steps.push(TRIANGLE_DRAW);
        steps
    }
}

/// The colour states the depth cases' two pipelines take: no channel written, and every channel.
const MASKED: [Option<ColorState>; 1] =
    [Some(ColorState::new(FORMAT).masked(ColorWriteMask::NONE))];
const PLAIN: [Option<ColorState>; 1] = [Some(ColorState::new(FORMAT))];

impl Case for DepthBias {
    fn name(&self) -> &'static str {
        "render-state-depth-bias"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // **Nothing is loosened.** A texel is four ones or the pass's clear, both of which a
        // `RGBA32Float` attachment stores exactly, and the fragment function does no arithmetic a
        // device's float bounds could reach.
        //
        // Three kinds are declared over the notices site, each measured firing for all three rows
        // of the family. The depth attachment nothing reads is stored `DontCare`, and a device
        // whose `DontCare` store is not Metal's announces that it took the near-equivalent
        // (`dont-care-store`, llvmpipe and the RTX 5080). A device that clamps a depth bias
        // outside the unit range where Metal does not says so
        // (`biased-depth-past-unit-range`, llvmpipe). And the buffer each pass reads its colour
        // target back through is `Shared` ([`crate::replay::shared_memory_notice`], the 5080).
        pass_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(DEPTH_FIXTURE, DEPTH_ENTRIES)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(DEPTH_FIXTURE)?;
        for (at, pipeline) in Self::pipelines(&MASKED, &PLAIN).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes("target:color", &color_target().words()));
        inputs.push(Input::bytes(
            "target:depth",
            &TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2]).words(),
        ));
        inputs.push(Input::bytes("buffer:positions", &positions_buffer()));
        inputs.push(Input::bytes(
            "state:writing",
            &state_words(&writing_state()),
        ));
        inputs.push(Input::bytes(
            "state:testing",
            &state_words(&testing_state()),
        ));
        inputs.push(Input::bytes("clear:color", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("clear:depth", &CLEARED_DEPTH.to_le_bytes()));
        let mut biases = Vec::new();
        for (box_of, bias) in bias_cells() {
            biases.extend(box_of.words());
            biases.extend(bias.words());
        }
        for box_of in [SLOPE_PLAIN, SLOPE_SCALED, SLOPE_HELD] {
            biases.extend(box_of.words());
        }
        biases.extend(SLOPE_SCALE.to_le_bytes());
        biases.extend(HELD_CLAMP.to_le_bytes());
        inputs.push(Input::bytes("biases", &biases));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites, DEPTH_FIXTURE)?;
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&MASKED, &PLAIN).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let color = made(sites, "target:color", side.target(&color_target()))?;
        let depth = made(
            sites,
            "target:depth",
            side.target(&TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2])),
        )?;
        let positions = made(sites, "buffer:positions", side.buffer(&positions_buffer()))?;
        let writing = made(
            sites,
            "state:writing",
            side.depth_stencil_state(&writing_state()),
        )?;
        let testing = made(
            sites,
            "state:testing",
            side.depth_stencil_state(&testing_state()),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(color, [f64::from(CLEAR); 4])])
                    .with_depth(
                        PassDepth::cleared(depth, CLEARED_DEPTH).stored(StoreAction::DontCare),
                    )
                    .drawing(Self::steps(
                        states[0], states[1], writing, testing, positions,
                    )),
            ),
            read_target(color, read_back),
        ];
        drawn(side, sites, &work, read_back)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-depth-clip

/// The four cells of the depth-clip case, in draw order: a control the near plane has nothing to
/// do with, the near triangle under each clip mode, and the mode set back.
const CONTROL_CELL: ScissorBox = cell(0, 0);
const CLIPPED_CELL: ScissorBox = cell(1, 0);
const CLAMPED_CELL: ScissorBox = cell(2, 0);
const CLIPPED_AGAIN_CELL: ScissorBox = cell(3, 0);

/// `setDepthClipMode:` over a triangle in front of the near plane.
pub struct DepthClip;

impl DepthClip {
    /// The pipelines, as the bias case's are.
    fn pipelines(
        masked: &'static [Option<ColorState>],
        plain: &'static [Option<ColorState>],
    ) -> [RenderPipelineSpec<'static>; 2] {
        DepthBias::pipelines(masked, plain)
    }

    /// The pass's steps.
    fn steps(
        prefill: Handle,
        coloring: Handle,
        writing: Handle,
        testing: Handle,
        positions: Handle,
    ) -> Vec<RenderStep> {
        let at = |which: usize| RenderStep::VertexBuffer {
            buffer: positions,
            offset: which as u64 * POSITION_STRIDE,
            index: 0,
        };
        let mut steps = prefill_steps(prefill, writing, positions);
        steps.extend([
            RenderStep::Pipeline(coloring),
            RenderStep::DepthStencilState(testing),
            // The control: a triangle plainly nearer than the stored depth, under Metal's own clip
            // mode. Without it a case whose every cell came out clear would read as the clip mode
            // working, and the setup being broken would read the same.
            at(MID),
            RenderStep::Scissor(CONTROL_CELL),
            TRIANGLE_DRAW,
            // The near triangle, clipped: `Clip` is Metal's default and the case states it, so the
            // cell's answer is the mode's rather than the default's.
            RenderStep::DepthClip(DepthClipMode::Clip),
            at(NEAR),
            RenderStep::Scissor(CLIPPED_CELL),
            TRIANGLE_DRAW,
            // The same triangle, clamped: its depth becomes the near plane's, which is less than
            // the stored depth, so it draws.
            RenderStep::DepthClip(DepthClipMode::Clamp),
            RenderStep::Scissor(CLAMPED_CELL),
            TRIANGLE_DRAW,
            // The mode set back, which the second cell's answer is the control for.
            RenderStep::DepthClip(DepthClipMode::Clip),
            RenderStep::Scissor(CLIPPED_AGAIN_CELL),
            TRIANGLE_DRAW,
        ]);
        steps
    }
}

impl Case for DepthClip {
    fn name(&self) -> &'static str {
        "render-state-depth-clip"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened, and the same three declarations, for the reason
        // `render-state-depth-bias` states.
        pass_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(DEPTH_FIXTURE, DEPTH_ENTRIES)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(DEPTH_FIXTURE)?;
        for (at, pipeline) in Self::pipelines(&MASKED, &PLAIN).iter().enumerate() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{at}"),
                &pipeline.words(),
            ));
        }
        inputs.push(Input::bytes("target:color", &color_target().words()));
        inputs.push(Input::bytes(
            "target:depth",
            &TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2]).words(),
        ));
        inputs.push(Input::bytes("buffer:positions", &positions_buffer()));
        inputs.push(Input::bytes(
            "state:writing",
            &state_words(&writing_state()),
        ));
        inputs.push(Input::bytes(
            "state:testing",
            &state_words(&testing_state()),
        ));
        inputs.push(Input::bytes("clear:color", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("clear:depth", &CLEARED_DEPTH.to_le_bytes()));
        let cells: Vec<u8> = [CONTROL_CELL, CLIPPED_CELL, CLAMPED_CELL, CLIPPED_AGAIN_CELL]
            .iter()
            .flat_map(ScissorBox::words)
            .collect();
        inputs.push(Input::bytes("cells", &cells));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites, DEPTH_FIXTURE)?;
        let mut states = Vec::new();
        for (at, pipeline) in Self::pipelines(&MASKED, &PLAIN).iter().enumerate() {
            states.push(made(
                sites,
                &format!("render-pipeline:{at}"),
                side.render_pipeline(library, pipeline),
            )?);
        }
        let color = made(sites, "target:color", side.target(&color_target()))?;
        let depth = made(
            sites,
            "target:depth",
            side.target(&TargetSpec::plain(DEPTH_FORMAT, [SIDE; 2])),
        )?;
        let positions = made(sites, "buffer:positions", side.buffer(&positions_buffer()))?;
        let writing = made(
            sites,
            "state:writing",
            side.depth_stencil_state(&writing_state()),
        )?;
        let testing = made(
            sites,
            "state:testing",
            side.depth_stencil_state(&testing_state()),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(color, [f64::from(CLEAR); 4])])
                    .with_depth(
                        PassDepth::cleared(depth, CLEARED_DEPTH).stored(StoreAction::DontCare),
                    )
                    .drawing(Self::steps(
                        states[0], states[1], writing, testing, positions,
                    )),
            ),
            read_target(color, read_back),
        ];
        drawn(side, sites, &work, read_back)
    }
}

// ------------------------------------------------------------------------------------------------
// render-state-stencil-references

/// The clip-space points of the first triangle the stencil case draws: the whole clip square and
/// more, so its scissor rectangle is what confines it.
const FIRST_TRIANGLE: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

/// The same three points in the opposite order, which is the same triangle of the opposite
/// winding.
const SECOND_TRIANGLE: [[f32; 2]; 3] = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]];

/// How far apart the two triangles lie in the points buffer: three `float2`s take 24 bytes and the
/// stride is the next multiple of 16, the offset granularity the suite's render cases bind at.
const POINT_STRIDE: u64 = 32;

/// The two triangles' points as one buffer.
fn points_buffer() -> Vec<u8> {
    let mut bytes = Vec::new();
    for triangle in [FIRST_TRIANGLE, SECOND_TRIANGLE] {
        let start = bytes.len();
        for point in triangle {
            bytes.extend(point.iter().flat_map(|axis| axis.to_le_bytes()));
        }
        bytes.resize(start + POINT_STRIDE as usize, 0);
    }
    bytes
}

/// The colour the stencil case's draws store, which its clear is not.
const STENCIL_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// The colour as the fragment function's `constant float4 &` binding takes it.
fn stencil_color_bytes() -> Vec<u8> {
    STENCIL_COLOR
        .iter()
        .flat_map(|channel| channel.to_le_bytes())
        .collect()
}

/// What one draw of the stencil case sets: the references it states, which triangle it draws, and
/// the cell it is confined to.
struct ReferenceDraw {
    /// The front and back references, or `None` where the draw leaves the previous draw's.
    references: Option<(u32, u32)>,
    /// Whether the two references arrive through `setStencilReferenceValue:`, which sets both
    /// faces at once, rather than through the call that sets them apart. `None` carries no call.
    together: bool,
    /// `false` for [`FIRST_TRIANGLE`], `true` for [`SECOND_TRIANGLE`].
    reversed: bool,
    cell: ScissorBox,
}

/// The ten draws, in order.
///
/// The first four state one reference for both faces, through the call that sets them together:
/// the stored stencil, which both faces then equal, and a value neither equals. The next four
/// state the two faces apart, each way round. The last two set them together again, which is what
/// says the together form reaches both faces after the apart form has set them differently.
fn reference_draws() -> Vec<ReferenceDraw> {
    let draw = |references, together, reversed, cell| ReferenceDraw {
        references,
        together,
        reversed,
        cell,
    };
    vec![
        draw(
            Some((STORED_STENCIL, STORED_STENCIL)),
            true,
            false,
            cell(0, 0),
        ),
        draw(None, true, true, cell(1, 0)),
        draw(
            Some((OTHER_STENCIL, OTHER_STENCIL)),
            true,
            false,
            cell(2, 0),
        ),
        draw(None, true, true, cell(3, 0)),
        draw(
            Some((STORED_STENCIL, OTHER_STENCIL)),
            false,
            false,
            cell(0, 1),
        ),
        draw(None, false, true, cell(1, 1)),
        draw(
            Some((OTHER_STENCIL, STORED_STENCIL)),
            false,
            false,
            cell(2, 1),
        ),
        draw(None, false, true, cell(3, 1)),
        draw(
            Some((STORED_STENCIL, STORED_STENCIL)),
            true,
            false,
            cell(0, 2),
        ),
        draw(None, true, true, cell(1, 2)),
    ]
}

/// `setStencilReferenceValue:` and `setStencilFrontReferenceValue:backReferenceValue:` under a
/// state whose two faces both compare `Equal`.
pub struct StencilReferences;

impl StencilReferences {
    /// The one pipeline: the triangle, the solid colour, and the stencil attachment's format,
    /// which Metal's descriptor must declare before a pipeline may test the stencil.
    fn pipeline(colors: &'static [Option<ColorState>]) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::of(
            StageFunction::named(STENCIL_VERTEX),
            Some(StageFunction::named(SOLID)),
            colors,
        )
        .depth_stencil(None, Some(STENCIL_FORMAT))
    }

    /// The pass's steps.
    fn steps(state: Handle, equal: Handle, points: Handle, color: Handle) -> Vec<RenderStep> {
        let mut steps = vec![
            RenderStep::Pipeline(state),
            RenderStep::DepthStencilState(equal),
            RenderStep::FragmentBuffer {
                buffer: color,
                offset: 0,
                index: 0,
            },
        ];
        for draw in reference_draws() {
            if let Some((front, back)) = draw.references {
                steps.push(match draw.together {
                    true => RenderStep::StencilReference(front),
                    false => RenderStep::StencilReferences { front, back },
                });
            }
            steps.push(RenderStep::VertexBuffer {
                buffer: points,
                offset: u64::from(draw.reversed) * POINT_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::Scissor(draw.cell));
            steps.push(TRIANGLE_DRAW);
        }
        steps
    }
}

impl Case for StencilReferences {
    fn name(&self) -> &'static str {
        "render-state-stencil-references"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened, and the same three declarations, for the reason
        // `render-state-depth-bias` states.
        pass_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(STENCIL_FIXTURE, STENCIL_ENTRIES)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs(STENCIL_FIXTURE)?;
        inputs.push(Input::bytes(
            "render-pipeline",
            &Self::pipeline(&PLAIN).words(),
        ));
        inputs.push(Input::bytes("target:color", &color_target().words()));
        inputs.push(Input::bytes(
            "target:stencil",
            &TargetSpec::plain(STENCIL_FORMAT, [SIDE; 2]).words(),
        ));
        inputs.push(Input::bytes("buffer:points", &points_buffer()));
        inputs.push(Input::bytes("buffer:color", &stencil_color_bytes()));
        inputs.push(Input::bytes("state:equal", &state_words(&equal_state())));
        inputs.push(Input::bytes("clear:color", &CLEAR.to_le_bytes()));
        inputs.push(Input::bytes("clear:stencil", &STORED_STENCIL.to_le_bytes()));
        let mut references = Vec::new();
        for draw in reference_draws() {
            references.extend(draw.cell.words());
            references.push(u8::from(draw.together));
            references.push(u8::from(draw.reversed));
            if let Some((front, back)) = draw.references {
                references.extend(front.to_le_bytes());
                references.extend(back.to_le_bytes());
            }
        }
        inputs.push(Input::bytes("references", &references));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let library = library(side, sites, STENCIL_FIXTURE)?;
        let state = made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &Self::pipeline(&PLAIN)),
        )?;
        let color_texture = made(sites, "target:color", side.target(&color_target()))?;
        let stencil = made(
            sites,
            "target:stencil",
            side.target(&TargetSpec::plain(STENCIL_FORMAT, [SIDE; 2])),
        )?;
        let points = made(sites, "buffer:points", side.buffer(&points_buffer()))?;
        let color = made(sites, "buffer:color", side.buffer(&stencil_color_bytes()))?;
        let equal = made(
            sites,
            "state:equal",
            side.depth_stencil_state(&equal_state()),
        )?;
        let read_back = made(
            sites,
            "buffer:read-back",
            side.zeroed_buffer(PLANE as usize),
        )?;
        let work = vec![
            RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::cleared(
                    color_texture,
                    [f64::from(CLEAR); 4],
                )])
                .with_stencil(
                    PassStencil::cleared(stencil, STORED_STENCIL).stored(StoreAction::DontCare),
                )
                .drawing(Self::steps(state, equal, points, color)),
            ),
            read_target(color_texture, read_back),
        ];
        drawn(side, sites, &work, read_back)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allowance::Loosening;
    use crate::replay::NOTICES_SITE;
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing recorded it.
    ///
    /// The assertion is the module's own check on determinism: every case here is one pass of one
    /// queue with nothing a device may reorder, so the eight runs are expected to fold into one
    /// outcome, and a case whose depth or stencil test answered differently between runs says so
    /// here rather than having one run read as the answer.
    fn only_outcome(case: &str) -> Option<Transcript> {
        let recording = crate::recording::read(case).ok()?;
        let answer = recording.answers.first()?;
        assert_eq!(
            answer.outcomes.len(),
            1,
            "{case}: {} outcomes over {} runs, so the test's answer moved between runs",
            answer.outcomes.len(),
            answer.runs
        );
        Some(answer.outcomes[0].transcript.clone())
    }

    /// The colour target's recorded texels of `case`.
    fn recorded(case: &str) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child("texels"))
            .expect("no drawn/texels site");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("drawn/texels holds no bytes");
        };
        assert_eq!(bytes.len(), PLANE as usize);
        Some(bytes.clone())
    }

    /// The `float4` at texel `(x, y)`.
    fn value(bytes: &[u8], x: u32, y: u32) -> [f32; 4] {
        let at = ((y * SIDE + x) * TEXEL) as usize;
        let mut channels = [0f32; 4];
        for (channel, held) in channels.iter_mut().enumerate() {
            let word = at + 4 * channel;
            *held = f32::from_le_bytes([
                bytes[word],
                bytes[word + 1],
                bytes[word + 2],
                bytes[word + 3],
            ]);
        }
        channels
    }

    /// Whether every texel of `rect` holds `color`, or every one holds the clear, and which: the
    /// answer a cell gives when the only two values a draw of these cases can leave are those two.
    fn cell_drawn(bytes: &[u8], rect: ScissorBox, color: [f32; 4]) -> bool {
        let mut colored = 0usize;
        let mut cleared = 0usize;
        let mut other = Vec::new();
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let got = value(bytes, x as u32, y as u32);
                if got == color {
                    colored += 1;
                } else if got == [CLEAR; 4] {
                    cleared += 1;
                } else {
                    other.push(format!("({x}, {y}): {got:?}"));
                }
            }
        }
        assert!(
            other.is_empty(),
            "{rect:?} holds {} texel(s) that are neither the drawn colour nor the clear: {}",
            other.len(),
            other.join(", ")
        );
        let whole = (rect.width * rect.height) as usize;
        assert!(
            colored == whole || cleared == whole,
            "{rect:?} is {colored} of {whole} drawn, which is neither drawn nor clear"
        );
        colored == whole
    }

    /// Which rows of `rect` are drawn, as a set, for the sloped cells whose answer is a boundary
    /// rather than a whole cell.
    fn drawn_rows(bytes: &[u8], rect: ScissorBox, color: [f32; 4]) -> BTreeSet<u32> {
        let mut rows = BTreeSet::new();
        for y in rect.y..rect.y + rect.height {
            let mut colored = 0usize;
            for x in rect.x..rect.x + rect.width {
                let got = value(bytes, x as u32, y as u32);
                assert!(
                    got == color || got == [CLEAR; 4],
                    "({x}, {y}) is neither the drawn colour nor the clear: {got:?}"
                );
                if got == color {
                    colored += 1;
                }
            }
            assert!(
                colored == 0 || colored == rect.width as usize,
                "row {y} of {rect:?} is {colored} of {} drawn, and the triangle's depth does not \
                 vary along x",
                rect.width
            );
            if colored > 0 {
                rows.insert(y as u32);
            }
        }
        rows
    }

    /// How much of the depth range the sloped triangle covers in one pixel row, which is the
    /// `m` Metal's slope scale multiplies: `0.1` of the range over the `8` rows a unit of clip `y`
    /// takes on a 16-row target.
    const SLOPE_PER_ROW: f64 = 0.0125;

    /// The depth the sloped triangle has at the centre of pixel row `row`, which is what the depth
    /// test reads: `0.6 - 0.0125 * (row + 0.5)`, from `0.5 + 0.1 * y` and `y = 1 - row / 8`.
    fn sloped_depth(row: u32) -> f64 {
        0.6 - SLOPE_PER_ROW * (f64::from(row) + 0.5)
    }

    /// Whether the sloped triangle's fragment in `row` survives `MTLCompareFunctionLess` against
    /// [`STORED_DEPTH`] under a depth offset of `offset`.
    fn sloped_passes(row: u32, offset: f64) -> bool {
        sloped_depth(row) + offset < STORED_DEPTH
    }

    /// The depth offset each of the three sloped cells draws under, in the order the case draws
    /// them: none, the slope scale's own, and the slope scale's bounded by the clamp.
    ///
    /// Metal's offset is `m * slopeScale + r * depthBias`, bounded by `clamp` where that is not
    /// zero -- toward zero, since a negative clamp bounds a negative offset. The constant term is
    /// nothing here, so each cell's offset is arithmetic over `SLOPE_PER_ROW` alone.
    fn sloped_offsets() -> [f64; 3] {
        let scaled = f64::from(SLOPE_SCALE) * SLOPE_PER_ROW;
        [0.0, scaled, scaled.max(f64::from(HELD_CLAMP))]
    }

    /// What the depth cases' fragment function returns in every channel.
    const WHITE_TEXEL: [f32; 4] = [1.0; 4];

    /// **Every scissor rectangle the module states lies inside its target.**
    ///
    /// A rectangle reaching past the attachments is the one shape of `setScissorRect:` Metal's own
    /// validation calls a programming error; what plain Metal's driver programs for one is
    /// unmeasured, so no case here states one, and the guard holds before a device is asked.
    #[test]
    fn every_cell_lies_inside_the_target() {
        // The check sees a rectangle that reaches past one, so a green run means the cells are
        // inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, SIDE as u64 + 1, 1).past([SIDE; 2]));
        let mut cells: Vec<ScissorBox> = bias_cells().iter().map(|(rect, _)| *rect).collect();
        cells.extend([SLOPE_PLAIN, SLOPE_SCALED, SLOPE_HELD]);
        cells.extend([CONTROL_CELL, CLIPPED_CELL, CLAMPED_CELL, CLIPPED_AGAIN_CELL]);
        cells.extend(reference_draws().iter().map(|draw| draw.cell));
        for rect in cells {
            assert!(!rect.past([SIDE; 2]), "{rect:?} reaches past the target");
        }
    }

    /// **The depths the module writes down are the depths its positions carry**, and they leave
    /// the depth test an answer with room to spare.
    ///
    /// Stated over the arithmetic rather than over a recording: the whole expectation of the two
    /// depth cases is that a fragment's depth is or is not less than [`STORED_DEPTH`], and a value
    /// a hair from it would make the answer a rounding rule instead.
    #[test]
    fn the_stated_depths_match_the_positions_and_clear_the_test_by_a_margin() {
        // The flat triangle sits at exactly the stored depth, which is what makes a constant bias
        // the only thing that can carry it past `Less`.
        assert_eq!(triangle_depths(FLAT), [STORED_DEPTH as f32; 3]);
        // The sloped triangle's depth is `0.5 + 0.1 * y` at every corner, so its slope along the
        // window's rows is `SLOPE_PER_ROW` and along the columns is nothing.
        for (corner, depth) in CORNERS.iter().zip(triangle_depths(SLOPED)) {
            assert_eq!(depth, 0.5 + 0.1 * corner[1]);
        }
        let step = sloped_depth(0) - sloped_depth(1);
        assert!((step - SLOPE_PER_ROW).abs() < 1e-12, "{step} per row");
        // The near triangle is in front of the near plane and the control is behind it.
        assert!(triangle_depths(NEAR).iter().all(|z| *z < 0.0));
        assert!(triangle_depths(MID)
            .iter()
            .all(|z| (0.0..STORED_DEPTH as f32).contains(z)));
        // Every row of the two sloped cells answers the test by at least a thousandth of the
        // depth range, which a `Depth32Float` holds to a part in sixteen million, so no row's
        // answer turns on rounding.
        for row in SLOPE_PLAIN.y..SLOPE_PLAIN.y + SLOPE_PLAIN.height {
            for offset in sloped_offsets() {
                let margin = (sloped_depth(row as u32) + offset - STORED_DEPTH).abs();
                assert!(
                    margin > 1e-3,
                    "row {row} at an offset of {offset}: {margin}"
                );
            }
        }
        // The clamp the third sloped cell states bounds the slope-scaled offset to half of it, so
        // its own reading is one pixel row rather than two. Stated in the `f32` Metal is handed,
        // because the nearest `f32` to half a row's depth is not the `f64` that names it.
        assert_eq!(
            HELD_CLAMP,
            (f64::from(SLOPE_SCALE) * SLOPE_PER_ROW / 2.0) as f32
        );
    }

    /// **The two sloped cells disagree, by exactly the rows the slope scale moves.**
    ///
    /// Stated over the arithmetic, so it holds with no device: a slope scale of minus two subtracts
    /// two rows' worth of depth, so the crossing moves two rows and the scaled cell draws two more.
    #[test]
    fn the_slope_scale_moves_the_crossing_by_two_rows_and_the_clamp_by_one() {
        let rows = |offset: f64| -> BTreeSet<u32> {
            (SLOPE_PLAIN.y..SLOPE_PLAIN.y + SLOPE_PLAIN.height)
                .filter(|row| sloped_passes(*row as u32, offset))
                .map(|row| row as u32)
                .collect()
        };
        let [none, scaled, held] = sloped_offsets().map(rows);
        // Each offset keeps every row the smaller one had and adds to it, so the three cells are
        // nested and the differences between them are the readings.
        assert!(none.is_subset(&held) && held.is_subset(&scaled));
        assert_eq!(
            scaled.len() - none.len(),
            2,
            "the slope scale moved the crossing by {} rows, not two: {none:?} against {scaled:?}",
            scaled.len() - none.len()
        );
        assert_eq!(
            held.len() - none.len(),
            1,
            "the clamp left the crossing {} rows from the unbiased one, not one: {none:?} against \
             {held:?}",
            held.len() - none.len()
        );
        // No cell is all-drawn or all-clear, so each is a boundary the case can read rather than
        // one more whole cell.
        let whole = SLOPE_PLAIN.height as usize;
        for (what, rows) in [("plain", &none), ("scaled", &scaled), ("held", &held)] {
            assert!(
                !rows.is_empty() && rows.len() < whole,
                "the {what} sloped cell is {} of {whole} rows",
                rows.len()
            );
        }
        // The three cells are the same eight rows of the target, in three columns, so their
        // answers are comparable.
        for rect in [SLOPE_SCALED, SLOPE_HELD] {
            assert_eq!(rect.y, SLOPE_PLAIN.y);
            assert_eq!(rect.height, SLOPE_PLAIN.height);
            assert_ne!(rect.x, SLOPE_PLAIN.x);
        }
        assert_ne!(SLOPE_SCALED.x, SLOPE_HELD.x);
    }

    /// **Metal's recorded target holds what the bias each draw was given says it should.**
    #[test]
    fn metal_biased_each_depth_draw_as_its_bias_says() {
        let Some(bytes) = recorded(DepthBias.name()) else {
            println!("render-state-depth-bias: no recording in the tree yet");
            return;
        };
        let cells = bias_cells();
        let held: Vec<bool> = cells
            .iter()
            .map(|(rect, _)| cell_drawn(&bytes, *rect, WHITE_TEXEL))
            .collect();
        // The flat triangle sits at exactly the stored depth, so `Less` fails with no bias.
        assert!(
            !held[0],
            "the unbiased flat triangle drew, so `Less` passed at an equal depth"
        );
        // One minimum resolvable difference nearer is what carries it past the test.
        assert!(
            held[1],
            "a constant bias of -1 did not carry the flat triangle past `Less`"
        );
        // And the bias set back to none puts it behind the test again.
        assert!(!held[2], "the bias was not set back to none");
        // The three sloped cells: the rows each drew, against the rows its depth and its offset
        // say. The second is the slope scale's reading and the third is the clamp's.
        for (rect, offset) in [SLOPE_PLAIN, SLOPE_SCALED, SLOPE_HELD]
            .into_iter()
            .zip(sloped_offsets())
        {
            let want: BTreeSet<u32> = (rect.y..rect.y + rect.height)
                .filter(|row| sloped_passes(*row as u32, offset))
                .map(|row| row as u32)
                .collect();
            assert_eq!(
                drawn_rows(&bytes, rect, WHITE_TEXEL),
                want,
                "the rows {rect:?} drew at a depth offset of {offset} are not the rows its depth \
                 says"
            );
        }
        // Every texel outside every cell holds the clear, which says the prefill wrote no colour
        // and no region draw escaped its scissor rectangle.
        let mut used: Vec<ScissorBox> = cells.iter().map(|(rect, _)| *rect).collect();
        used.extend([SLOPE_PLAIN, SLOPE_SCALED, SLOPE_HELD]);
        assert_cleared_outside(&bytes, &used);
    }

    /// **Metal's recorded target holds what each clip mode says it should.**
    #[test]
    fn metal_clipped_or_clamped_the_near_triangle_as_the_mode_says() {
        let Some(bytes) = recorded(DepthClip.name()) else {
            println!("render-state-depth-clip: no recording in the tree yet");
            return;
        };
        // The control draws, so a case whose other cells came out clear is reading the clip mode
        // rather than a broken setup.
        assert!(
            cell_drawn(&bytes, CONTROL_CELL, WHITE_TEXEL),
            "the control triangle, plainly nearer than the stored depth, did not draw"
        );
        assert!(
            !cell_drawn(&bytes, CLIPPED_CELL, WHITE_TEXEL),
            "`MTLDepthClipModeClip` drew a triangle in front of the near plane"
        );
        assert!(
            cell_drawn(&bytes, CLAMPED_CELL, WHITE_TEXEL),
            "`MTLDepthClipModeClamp` did not draw a triangle in front of the near plane at the \
             near plane's own depth"
        );
        assert!(
            !cell_drawn(&bytes, CLIPPED_AGAIN_CELL, WHITE_TEXEL),
            "the clip mode was not set back to `Clip`"
        );
        assert_cleared_outside(
            &bytes,
            &[CONTROL_CELL, CLIPPED_CELL, CLAMPED_CELL, CLIPPED_AGAIN_CELL],
        );
    }

    /// Which cells each reference draw's colour is expected in, given `first_is_front`: whether
    /// [`FIRST_TRIANGLE`] is the front-facing one under the winding the pass leaves at Metal's
    /// default.
    ///
    /// The convention is read from a cell rather than assumed, because what the case asks is what
    /// the two reference values **do**, not which way round Metal calls a triangle.
    fn reference_drawn(first_is_front: bool) -> [bool; 10] {
        let front = first_is_front;
        [
            // Both faces at the stored stencil: both triangles pass.
            true, true, // Both faces at a value neither equals: neither passes.
            false, false,
            // The front face at the stored stencil and the back face elsewhere: the front-facing
            // one of the pair passes.
            front, !front, // The two faces swapped: the other one passes.
            !front, front,
            // Both faces together again, which the apart form had set differently.
            true, true,
        ]
    }

    /// **The reference rule holds for either facing convention**, which is what lets the case
    /// state its map without assuming one.
    #[test]
    fn the_reference_rule_holds_for_either_convention() {
        for front in [false, true] {
            let drawn = reference_drawn(front);
            // One reference for both faces: both triangles answer alike, whichever way round.
            assert_eq!(drawn[0], drawn[1]);
            assert_eq!(drawn[2], drawn[3]);
            assert_ne!(drawn[0], drawn[2]);
            // The two faces apart: exactly one of each pair draws, and swapping the two values
            // swaps which.
            assert_ne!(drawn[4], drawn[5]);
            assert_ne!(drawn[6], drawn[7]);
            assert_ne!(drawn[4], drawn[6]);
            // The together form, after the apart form, reaches both faces again.
            assert!(drawn[8] && drawn[9]);
        }
        // The ten draws take ten distinct cells, and the two references are two values.
        let draws = reference_draws();
        assert_eq!(draws.len(), 10);
        let cells: BTreeSet<[u64; 4]> = draws
            .iter()
            .map(|draw| [draw.cell.x, draw.cell.y, draw.cell.width, draw.cell.height])
            .collect();
        assert_eq!(cells.len(), draws.len(), "two draws share a cell");
        assert_ne!(STORED_STENCIL, OTHER_STENCIL);
        assert_eq!(
            STORED_STENCIL & 0xff,
            STORED_STENCIL,
            "a Stencil8 keeps 8 bits"
        );
        assert_eq!(OTHER_STENCIL & 0xff, OTHER_STENCIL);
        // The stencil case's colour is not its clear.
        assert_ne!(STENCIL_COLOR, [CLEAR; 4]);
    }

    /// **Metal's recorded target holds what each pair of stencil references says it should.**
    #[test]
    fn metal_tested_each_face_against_the_reference_it_was_given() {
        let Some(bytes) = recorded(StencilReferences.name()) else {
            println!("render-state-stencil-references: no recording in the tree yet");
            return;
        };
        let draws = reference_draws();
        let held: Vec<bool> = draws
            .iter()
            .map(|draw| cell_drawn(&bytes, draw.cell, STENCIL_COLOR))
            .collect();
        // The convention is read from the fifth cell, the first whose answer depends on it.
        let expected = reference_drawn(held[4]);
        assert_eq!(
            held.as_slice(),
            expected.as_slice(),
            "the cells Metal drew are not what the two faces' references imply (the first \
             triangle is {} under the pass's own winding)",
            match held[4] {
                true => "front-facing",
                false => "back-facing",
            }
        );
        assert_cleared_outside(
            &bytes,
            &draws.iter().map(|draw| draw.cell).collect::<Vec<_>>(),
        );
    }

    /// Every texel outside every rectangle of `used` holds the clear, which is what says no draw
    /// escaped its scissor rectangle and the prefill wrote no colour.
    fn assert_cleared_outside(bytes: &[u8], used: &[ScissorBox]) {
        let inside: BTreeSet<(u64, u64)> = used
            .iter()
            .flat_map(|rect| {
                let rect = *rect;
                (rect.y..rect.y + rect.height)
                    .flat_map(move |y| (rect.x..rect.x + rect.width).map(move |x| (x, y)))
            })
            .collect();
        for y in 0..u64::from(SIDE) {
            for x in 0..u64::from(SIDE) {
                if inside.contains(&(x, y)) {
                    continue;
                }
                assert_eq!(
                    value(bytes, x as u32, y as u32),
                    [CLEAR; 4],
                    "({x}, {y}) is outside every cell and is not the clear"
                );
            }
        }
    }

    /// **Every case of the family names itself once**, under the settings the module states.
    #[test]
    fn the_family_names_three_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 3);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            // Every case declares the family's three notices and loosens no value, so each
            // allowance sits at the notices site: a value tolerance added later fails here.
            let allowances = case.allowances();
            assert_eq!(allowances.len(), pass_notices().len(), "{}", case.name());
            for allowance in &allowances {
                assert_eq!(allowance.path, format!("case/{NOTICES_SITE}/*"));
                assert_eq!(allowance.loosening, Loosening::Unchecked);
            }
        }
        // The prefill pipeline writes no channel and the region pipeline writes every one, which
        // is what keeps the prefill's depth out of the colour target.
        assert_eq!(MASKED[0].unwrap().write_mask, ColorWriteMask::NONE);
        assert_eq!(PLAIN[0].unwrap().write_mask, ColorWriteMask::ALL);
        let [prefill, coloring] = DepthBias::pipelines(&MASKED, &PLAIN);
        assert_ne!(prefill.words(), coloring.words());
        // The two depth states differ in both fields that matter here.
        assert_ne!(state_words(&writing_state()), state_words(&testing_state()));
        assert_ne!(state_words(&equal_state()), state_words(&testing_state()));
        assert_eq!(layout().extent(), PLANE as usize);
    }
}
