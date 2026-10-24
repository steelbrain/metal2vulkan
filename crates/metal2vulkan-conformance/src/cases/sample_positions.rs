//! Where Metal puts a multisampled pass's samples: the device's own positions, custom ones, and
//! what Metal does to a coordinate that is not already a sixteenth of a pixel.
//!
//! Ported from the runtime lane `runtime_lane/sample_positions.rs`, whose first test is the Metal
//! probe's pass: a one-pixel multisampled `RGBA8Unorm` target cleared to zero, one triangle
//! covering everything left of a vertical edge, resolved into a single-sampled texture. The owned
//! fixture `rt6_restart` draws it — its vertex function places each vertex where buffer 0 holds it,
//! its fragment function writes alpha 1 — so the resolved alpha **is** the pixel's coverage, and
//! the number of covered samples follows from it.
//!
//! Each case records the resolved texel's four bytes and, beside them, the samples that coverage
//! comes to. The bytes are the stronger claim and they are compared exactly: a resolve of this
//! family's target averages `k` white samples out of `n`, so a venue whose resolve rounds
//! differently is a difference worth seeing rather than one worth tolerating. The count is the
//! claim the lane makes, and `tests::the_recorded_coverage_is_the_lanes_table` holds the recordings
//! to the lane's own numbers, which is the evidence this port owes the lane.
//!
//! # Which rows Metal may be asked, and which it may not
//!
//! This is the family where the input a case states reaches the rasterizer's own sample grid, so
//! the separation is the first thing about it.
//!
//! **Asked (`MetalSource::Run`)** — every row of the `COVERAGE` table. Each states as many
//! positions as the pass has samples, every coordinate inside `[0, 1)`, at a count Metal takes
//! custom positions at, or states none at all. These are the descriptors the lane's probe ran on
//! Metal **with and without** its API validation layer, and they are what Metal documents as
//! valid. One 1x1 target and three vertices a pass: nothing loops and nothing is large.
//!
//! The last two of them also name `MTLStoreActionOptionCustomSamplePositions` on the colour
//! attachment. Metal's API validation stops that option under `MultisampleResolve`, and plain
//! Metal keeps the resolve as it would without it — so those rows are where the recording's
//! **second oracle** earns its place: the plain leg says what Metal did, and the validated leg's
//! own complaint is recorded beside it, never compared with us.
//!
//! **Carried, and never re-recorded** — the `FEWER` table. Fewer positions than the pass's samples
//! is a count Metal's API validation stops, and the probe measured what plain Metal does with it: the
//! samples past the positions rasterize at the pixel's corner. The answer is written down, so the
//! row compares; the row is not re-recorded because the neighbouring shape, **one position past**
//! the pass's samples, is measured to end Metal's process
//! (`metal2vulkan_runtime::RenderPassProblem::SamplePositionsSamples`), and a record run that may
//! walk this family must not be one keystroke away from that.
//!
//! **Not ported at all** — the lane's refusal rows: a coordinate at 1.0 or below 0, one position in
//! a pass of four samples, a 1x pass with a position, and eight positions in a pass of four. Every
//! one of them is a row whose only Metal fact is that *Metal's API validation* stops it, and this
//! suite's ruling is that a validated stop is not Metal refusing a call: plain Metal is the
//! behavioural oracle. Nobody has measured what plain Metal does with those four shapes, and this
//! suite does not find out by trying — the eight-position one is measured fatal, and a coordinate
//! outside the pixel is a value Metal asserts on and that reaches the sample grid. So they have no
//! oracle value here and they stay runtime-lane rows, where the expectation is tied to Metal's
//! validation layer rather than to Metal. [`crate::render::forbidden_positions`] refuses the two
//! fatal shapes from the input's own shape, so a case cannot build one by accident.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{answered, Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::{ShaderRef, APPLE_M4};
use crate::render::{
    ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::replay::NOTICES_SITE;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites, Transcript, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{Degradation, SamplePosition, StoreAction, StoreActionOptions};

/// The owned fixture the probe drew with: each vertex where buffer 0 holds it, and alpha 1.
const FIXTURE: &str = "rt6_restart";
const VERTEX: &str = "rt6_restart_vertex";
const FRAGMENT: &str = "rt6_restart_count";

/// The target: one pixel of `RGBA8Unorm`, multisampled, resolved into a single-sampled one.
const COLOR: PixelFormat = PixelFormat::RGBA8Unorm;

/// The resolved texel's bytes, which is the whole read-back.
const TEXEL: u32 = 4;

/// A position at `(x, y)`, as a `const` table states one.
const fn at(x: f32, y: f32) -> SamplePosition {
    SamplePosition { x, y }
}

/// One row of the family: the pass's samples, its positions, the edge the triangle covers what
/// lies left of, and the samples the lane's table says Metal covered.
struct Shape {
    name: &'static str,
    samples: u32,
    /// The pass's positions, or empty for the device's own, which are Metal's.
    positions: &'static [SamplePosition],
    /// The vertical edge in normalized device coordinates.
    edge: f32,
    /// The samples Metal covered, as the lane's table holds it.
    covered: u64,
    /// The attachment's `storeAction`. The lane's own is `MultisampleResolve`, which keeps the
    /// resolve and nothing of the multisampled texture; two rows ask the other resolving action.
    store: StoreAction,
    /// The attachment's `storeActionOptions`.
    options: StoreActionOptions,
}

impl Shape {
    /// A row at the lane's own store action and Metal's default options.
    const fn new(
        name: &'static str,
        samples: u32,
        positions: &'static [SamplePosition],
        edge: f32,
        covered: u64,
    ) -> Shape {
        Shape {
            name,
            samples,
            positions,
            edge,
            covered,
            store: StoreAction::MultisampleResolve,
            options: StoreActionOptions::NONE,
        }
    }

    /// The row whose attachment keeps the multisampled texture as well as the resolve.
    const fn stored_too(mut self) -> Shape {
        self.store = StoreAction::StoreAndMultisampleResolve;
        self
    }

    /// The row whose attachment names `MTLStoreActionOptionCustomSamplePositions`.
    const fn with_custom_option(mut self) -> Shape {
        self.options = StoreActionOptions::CUSTOM_SAMPLE_POSITIONS;
        self
    }
}

/// Four positions at `(x, 0.5)`: the shape every row of the lane's table but the mixed one states.
macro_rules! four {
    ($x:expr) => {
        &[at($x, 0.5), at($x, 0.5), at($x, 0.5), at($x, 0.5)]
    };
}

/// The rows Metal is asked, in the order the report lists them: the lane's own table, name for
/// name.
///
/// The last four are the rounding question. Metal samples at the nearest sixteenth of a pixel, a
/// tie up, at most 15/16, which is why 0.53 and 0.51 cover against an edge at 0.52 while 0.55,
/// 0.53125, 0.5625 and 0.5624 do not, and why 0.97 and 0.99 both cover against an edge at 0.95.
const COVERAGE: &[Shape] = &[
    Shape::new("sample-positions-default-4x", 4, &[], 0.0, 2),
    Shape::new("sample-positions-custom-left", 4, four!(0.25), 0.0, 4),
    Shape::new("sample-positions-custom-right", 4, four!(0.75), 0.0, 0),
    Shape::new(
        "sample-positions-custom-mixed",
        4,
        &[at(0.25, 0.5), at(0.75, 0.5), at(0.75, 0.5), at(0.75, 0.5)],
        0.0,
        1,
    ),
    Shape::new("sample-positions-off-grid", 4, four!(0.53), 0.04, 4),
    Shape::new("sample-positions-off-grid-low", 4, four!(0.51), 0.04, 4),
    Shape::new("sample-positions-high", 4, four!(0.97), 0.9, 4),
    Shape::new("sample-positions-rounded-0-55", 4, four!(0.55), 0.04, 0),
    Shape::new(
        "sample-positions-rounded-0-53125",
        4,
        four!(0.53125),
        0.04,
        0,
    ),
    Shape::new("sample-positions-rounded-0-5625", 4, four!(0.5625), 0.04, 0),
    Shape::new("sample-positions-rounded-0-5624", 4, four!(0.5624), 0.04, 0),
    Shape::new("sample-positions-rounded-0-99", 4, four!(0.99), 0.9, 4),
    Shape::new(
        "sample-positions-custom-2x",
        2,
        &[at(0.25, 0.5), at(0.25, 0.5)],
        0.0,
        2,
    ),
    Shape::new("sample-positions-default-2x", 2, &[], 0.0, 1),
    // `sp_custom_storeopt`: the option under `MultisampleResolve`, which Metal's API
    // validation stops and plain Metal runs, keeping the resolve as it would without it.
    Shape::new(
        "sample-positions-option-under-resolve",
        4,
        four!(0.25),
        0.0,
        4,
    )
    .with_custom_option(),
    // `opt_store_resolve`: the option under `StoreAndMultisampleResolve`, which Metal takes in
    // both modes, since that action does keep the samples the option names.
    Shape::new(
        "sample-positions-option-under-store-and-resolve",
        4,
        four!(0.25),
        0.0,
        4,
    )
    .stored_too()
    .with_custom_option(),
];

/// The rows whose Metal answer is carried: two positions in a pass of four samples, which Metal's
/// API validation stops and plain Metal runs, putting samples 2 and 3 at the pixel's corner.
const FEWER: &[Shape] = &[
    Shape::new(
        "sample-positions-fewer-right-of-a-far-edge",
        4,
        &[at(0.75, 0.5), at(0.75, 0.5)],
        -0.8,
        2,
    ),
    Shape::new(
        "sample-positions-fewer-right-of-the-middle",
        4,
        &[at(0.75, 0.5), at(0.75, 0.5)],
        0.0,
        2,
    ),
    Shape::new(
        "sample-positions-fewer-left-of-the-middle",
        4,
        &[at(0.25, 0.5), at(0.25, 0.5)],
        0.0,
        4,
    ),
];

/// Where the carried rows' Metal answer was measured, and why record mode must not replace it.
const CARRIED_FROM: &str = "the Metal probe behind runtime_lane/sample_positions.rs";
const NEVER_RE_RECORD: &str = "fewer custom sample positions than the pass's samples is a count \
                               Metal's API validation stops, and one position past the pass's \
                               samples is measured to end Metal's process: a record run must not \
                               walk this family";

/// How many runs the carried measurement stands on: the probe ran each pass once, with and without
/// Metal's API validation layer.
const MEASURED_RUNS: u32 = 1;

/// The degrade notices one of these passes may announce. A list rather than a device query, because
/// a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
const PASS_NOTICES: &[Degradation] = &[
    // A `Shared` buffer the device keeps in system memory, which the triangle's vertices are.
    Degradation::SharedInSystemMemory,
    // The device's own compiler rewrote the stage's fast-math arithmetic as it made the pipeline.
    Degradation::FastMathRewrites,
    // The fixture's vertex function writes `[[point_size]]`, which a device without
    // `shaderPointSize`'s guarantee takes as its own point size. Announced on lavapipe.
    Degradation::DevicePointSize,
    // A device that resolves the multisampled attachment through a pass of its own rather than in
    // the store, and converts the colour in a shader to do it. Announced on lavapipe, where the
    // whole resolve is emulated; MoltenVK resolves in the store and announces neither.
    Degradation::ShaderResolve,
    Degradation::ShaderColorConversion,
    // The multisampled attachment of a `MultisampleResolve` store is Vulkan's `DONT_CARE`, which
    // Apple's GPU is measured to leave at the earlier texels unless its tiler overflows. Announced
    // on MoltenVK, where that driver fact holds. Nothing of this case reads the multisampled
    // attachment -- the resolved texel is the whole answer, and it compares exactly -- so naming
    // the kind declares what our side announced and loosens no byte.
    Degradation::DontCareStore,
];

/// The three vertices of the probe's triangle, which covers what lies left of `edge`.
fn triangle(edge: f32) -> Vec<u8> {
    [edge, -3.0, edge, 3.0, -3.0, 0.0]
        .iter()
        .flat_map(|coordinate| coordinate.to_le_bytes())
        .collect()
}

/// The samples a resolved texel's coverage comes to: its alpha out of 255, times the pass's
/// samples, to the nearest whole sample.
///
/// A resolve of this target averages the white samples, so an alpha of `k * 255 / n` is `k` samples
/// of `n`. The rounding is the lane's own rule, so the number this records is the number the lane's
/// table holds.
fn covered(texel: &[u8], samples: u32) -> u64 {
    let alpha = texel.get(3).copied().unwrap_or(0);
    (u64::from(alpha) * u64::from(samples) + 127) / 255
}

/// The samples `covered` comes to, as a site records it.
fn coverage(texel: &[u8], samples: u32) -> Value {
    Value::Lines(vec![format!(
        "{} of {samples} samples",
        covered(texel, samples)
    )])
}

/// The layout the resolved texel is read under, so a diff names the channel rather than an offset.
fn layout() -> ByteLayout {
    ByteLayout::buffer(0, u64::from(TEXEL), Element::U8)
}

/// The one color state the pipeline holds, at `samples` samples.
fn colors() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(COLOR))]
}

/// The pipeline the shape makes.
fn pipeline_spec<'a>(shape: &Shape, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::new(VERTEX, FRAGMENT, colors).multisampled(shape.samples)
}

/// The multisampled target the shape renders into: one pixel, private, which is what Metal stores
/// no multisampled texture a host reads into.
fn target_spec(shape: &Shape) -> TargetSpec {
    TargetSpec::plain(COLOR, [1, 1]).multisampled(shape.samples)
}

/// The single-sampled texture the pass resolves into, and the blit reads back.
fn resolve_spec() -> TargetSpec {
    TargetSpec::plain(COLOR, [1, 1])
}

/// The sites the shape's set-up emits, in the order it emits them.
const SETUP_SITES: [&str; 6] = [
    "library",
    "render-pipeline",
    "target",
    "resolve",
    "vertices",
    "read-back",
];

/// The shaders a shape runs, as committed fixtures.
fn shaders() -> Result<Vec<ShaderRef>, String> {
    let fixture = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: format!("{VERTEX}+{FRAGMENT}"),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

/// Everything a shape feeds its run.
fn inputs(shape: &Shape) -> Result<Vec<Input>, String> {
    let fixture = Fixture::new(FIXTURE);
    let states = colors();
    Ok(vec![
        Input {
            name: "air".to_owned(),
            digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        },
        Input {
            name: "air-typed".to_owned(),
            digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
        },
        Input::bytes("render-pipeline", &pipeline_spec(shape, &states).words()),
        Input::bytes("target", &target_spec(shape).words()),
        Input::bytes("resolve", &resolve_spec().words()),
        // The pass's own words, which carry the positions by their Metal call's name and each
        // coordinate by its bits, so a position that moved reads as `Unrecorded`.
        Input::bytes("pass", &pass(shape, Handles::fingerprint()).words()),
        Input::bytes("vertices", &triangle(shape.edge)),
    ])
}

/// The handles a pass names. [`Handles::fingerprint`] is the set an input fingerprint is taken
/// over: a side's handle numbers are its own and none of them reaches the words, so any set states
/// the same pass.
#[derive(Clone, Copy)]
struct Handles {
    target: crate::case::Handle,
    resolve: crate::case::Handle,
    pipeline: crate::case::Handle,
    vertices: crate::case::Handle,
}

impl Handles {
    const fn fingerprint() -> Self {
        let zero = crate::case::Handle(0);
        Self {
            target: zero,
            resolve: zero,
            pipeline: zero,
            vertices: zero,
        }
    }
}

/// The probe's pass: the multisampled target cleared to zero and resolved, the triangle drawn, and
/// the shape's positions where it states any.
fn pass(shape: &Shape, handles: Handles) -> RenderPass {
    RenderPass::clearing(vec![PassColor::cleared(handles.target, [0.0; 4])
        .resolving(handles.resolve)
        .stored(shape.store)
        .with_options(shape.options)])
    .drawing(vec![
        RenderStep::Pipeline(handles.pipeline),
        RenderStep::vertex_buffer(handles.vertices, 0),
        RenderStep::triangle(),
    ])
    .at(shape.positions.to_vec())
}

/// Set the shape up on `side`, run its one command buffer, and emit every site but the answer:
/// the resolved texel's bytes, for the caller to record as it records them.
fn drive(shape: &Shape, side: &mut dyn Side, sites: &mut Sites) -> Result<Vec<u8>, Abort> {
    let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })?;
    let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
    let states = colors();
    let pipeline = crate::case::made(
        sites,
        "render-pipeline",
        side.render_pipeline(library, &pipeline_spec(shape, &states)),
    )?;
    let target = crate::case::made(sites, "target", side.target(&target_spec(shape)))?;
    let resolve = crate::case::made(sites, "resolve", side.target(&resolve_spec()))?;
    let vertices = crate::case::made(sites, "vertices", side.buffer(&triangle(shape.edge)))?;
    let read_back = crate::case::made(sites, "read-back", side.buffer(&vec![0u8; TEXEL as usize]))?;
    let handles = Handles {
        target,
        resolve,
        pipeline,
        vertices,
    };
    let work = vec![
        RenderWork::Pass(pass(shape, handles)),
        RenderWork::ReadTexture {
            texture: resolve,
            into: read_back,
            offset: 0,
            row_pitch: TEXEL,
            size: [1, 1],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        },
    ];
    let outcome = side.render(&work).map_err(|failure| Abort {
        at: "render".to_owned(),
        failure,
    })?;
    // What each piece answered, which is `None` throughout on Metal's side: a piece of ours that
    // refuses compares against Metal answering and reads as a refusal rather than a divergence.
    let mut calls = Node::branch("calls", Vec::new());
    for (piece, refused) in outcome.refused.iter().enumerate() {
        calls = calls.with(Node::leaf(
            format!("{piece}-{}", work[piece].call()),
            Observed::value(match refused {
                Some(failure) => failure.recorded(),
                None => Value::ok(Value::Absent),
            }),
        ));
    }
    sites.push(calls);
    if let Some((piece, failure)) = outcome
        .refused
        .iter()
        .enumerate()
        .find_map(|(piece, refused)| refused.as_ref().map(|failure| (piece, failure)))
    {
        return Err(Abort {
            at: format!("calls/{piece}-{}", work[piece].call()),
            failure: failure.clone(),
        });
    }
    sites.leaf("status", Observed::value(Value::ok(outcome.status.clone())));
    side.read_buffer(read_back, 0, TEXEL as usize)
        .map_err(|failure| Abort {
            at: "resolved".to_owned(),
            failure,
        })
}

/// The notices site, loosened to the kinds a pass of this family may announce.
fn allowances() -> Vec<Allowance> {
    PASS_NOTICES
        .iter()
        .map(|kind| {
            Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(*kind),
            )
        })
        .collect()
}

/// One row Metal is asked: it records the resolved texel and the coverage it comes to.
pub struct Coverage {
    shape: &'static Shape,
}

impl Case for Coverage {
    fn name(&self) -> &'static str {
        self.shape.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        allowances()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        inputs(self.shape)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let texel = drive(self.shape, side, sites)?;
        sites.leaf(
            "coverage",
            Observed::value(coverage(&texel, self.shape.samples)),
        );
        sites.leaf("resolved", Observed::bytes(texel, layout()));
        Ok(())
    }
}

/// One row whose Metal answer is carried: the probe measured the samples its coverage came to and
/// not the texel, so the coverage is the whole answer.
pub struct Fewer {
    shape: &'static Shape,
}

impl Case for Fewer {
    fn name(&self) -> &'static str {
        self.shape.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::carried_only(CARRIED_FROM, NEVER_RE_RECORD).measured_on(APPLE_M4)
    }

    fn allowances(&self) -> Vec<Allowance> {
        allowances()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        inputs(self.shape)
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        for call in SETUP_SITES {
            sites.leaf(call, answered());
        }
        let mut calls = Node::branch("calls", Vec::new());
        for (piece, call) in ["renderCommandEncoder", "copyFromTextureToBuffer"]
            .into_iter()
            .enumerate()
        {
            calls = calls.with(Node::leaf(
                format!("{piece}-{call}"),
                Observed::value(Value::ok(Value::Absent)),
            ));
        }
        sites.push(calls);
        sites.leaf(
            "status",
            Observed::value(Value::ok(Value::unit_variant("Completed"))),
        );
        sites.leaf(
            "coverage",
            Observed::value(Value::Lines(vec![format!(
                "{} of {} samples",
                self.shape.covered, self.shape.samples
            )])),
        );
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(Some((MEASURED_RUNS, sites.finish())))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let texel = drive(self.shape, side, sites)?;
        sites.leaf(
            "coverage",
            Observed::value(coverage(&texel, self.shape.samples)),
        );
        Ok(())
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for shape in COVERAGE {
        cases.push(Box::new(Coverage { shape }));
    }
    for shape in FEWER {
        cases.push(Box::new(Fewer { shape }));
    }
    cases
}

#[cfg(test)]
mod tests {
    //! The cross-check the port owes the lane: the coverage Metal produced, as the recordings hold
    //! it, against the lane's own table.
    //!
    //! The table is the lane's (`runtime_lane/sample_positions.rs`), restated here because a
    //! conformance case may not depend on the validation crate. A disagreement is a finding about
    //! one of the two chips or about the port, never something to edit away.

    use super::*;
    use crate::recording::Recording;

    /// The one outcome of `name`'s recorded Metal answer.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The site named `path` of `transcript`.
    fn site<'a>(case: &str, transcript: &'a Transcript, path: &str) -> &'a Node {
        let mut node = &transcript.root;
        for name in path.split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("{case}: no site {path} (stopped at {name})"));
        }
        node
    }

    /// Metal's own coverage on this M3 Max is the lane's number, and the resolved texel it was read
    /// from is `k * 255 / n` in alpha with nothing in the other channels.
    ///
    /// Every row Metal is asked must have a recording, so a row that was never recorded fails here
    /// rather than being skipped: the table is the claim, and a claim with no run behind it is what
    /// this suite exists to stop.
    #[test]
    fn the_recorded_coverage_is_the_lanes_table() {
        for shape in COVERAGE {
            let recorded = only_outcome(shape.name);
            let want = format!("{} of {} samples", shape.covered, shape.samples);
            assert_eq!(
                site(shape.name, &recorded, "coverage").observed.value,
                Value::Lines(vec![want.clone()]),
                "{}: the lane's table says {want}",
                shape.name
            );
            let Value::Bytes(texel) = &site(shape.name, &recorded, "resolved").observed.value
            else {
                panic!("{}: the resolved site holds no bytes", shape.name);
            };
            assert_eq!(texel.len(), TEXEL as usize, "{}", shape.name);
            assert_eq!(
                covered(texel, shape.samples),
                shape.covered,
                "{}: the texel {texel:?} and the lane's number",
                shape.name
            );
            // The fragment function writes red 1/255 and alpha 1, and nothing else; the resolve
            // averages both, so green and blue stay zero whatever the coverage.
            assert_eq!(
                (texel[1], texel[2]),
                (0, 0),
                "{}: the resolve wrote green or blue",
                shape.name
            );
        }
    }

    /// Every carried row states the lane's number as its Metal answer, over the runs the probe
    /// took, and forbids record mode from replacing it.
    #[test]
    fn every_carried_row_states_the_lanes_number_and_forbids_re_recording() {
        for shape in FEWER {
            let case = Fewer { shape };
            let source = case.metal_source();
            assert!(
                source.forbidden().is_some(),
                "{}: record mode may run it on Metal",
                shape.name
            );
            let (runs, transcript) = case
                .carried_transcript()
                .unwrap()
                .unwrap_or_else(|| panic!("{}: no carried transcript", shape.name));
            assert_eq!(runs, MEASURED_RUNS);
            assert_eq!(
                site(shape.name, &transcript, "coverage").observed.value,
                Value::Lines(vec![format!(
                    "{} of {} samples",
                    shape.covered, shape.samples
                )])
            );
            // The sites the carried transcript states are the sites a run emits, which is what
            // lets the two compare through one path.
            for call in SETUP_SITES {
                site(shape.name, &transcript, call);
            }
            site(shape.name, &transcript, "calls/0-renderCommandEncoder");
            site(shape.name, &transcript, "status");
        }
    }

    /// Every row of the family states positions the Metal leg's guard takes, or none at all.
    ///
    /// This is the separation made mechanical: a row whose count passes the pass's samples, or
    /// whose coordinate leaves the pixel, cannot be added to either table without failing here.
    #[test]
    fn no_row_states_a_shape_the_metal_leg_may_not_encode() {
        for shape in COVERAGE.iter().chain(FEWER) {
            assert!(
                crate::render::forbidden_positions(shape.positions, shape.samples).is_none(),
                "{}: the guard refuses its positions",
                shape.name
            );
        }
    }

    /// Every case names a row of one of the two tables, states its inputs and loosens **only** its
    /// notices site.
    ///
    /// The coverage a pass produced is a number Metal wrote; a number that differs is a bug, never
    /// a near-equivalent. A case that loosened its coverage or its texel would fail here.
    #[test]
    fn every_case_is_tabled_and_loosens_its_notices_alone() {
        let named: Vec<&str> = COVERAGE.iter().chain(FEWER).map(|row| row.name).collect();
        assert_eq!(named.len(), COVERAGE.len() + FEWER.len());
        for case in all() {
            assert!(
                named.contains(&case.name()),
                "{}: no row in either table",
                case.name()
            );
            let inputs = case.inputs().unwrap();
            for wanted in ["air", "air-typed", "pass", "vertices", "target"] {
                assert!(
                    inputs.iter().any(|input| input.name == wanted),
                    "{}: no {wanted} in its inputs",
                    case.name()
                );
            }
            case.shaders().unwrap();
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site",
                    case.name()
                );
                match &allowance.justified_by {
                    Justification::Degradation(code) => assert!(
                        PASS_NOTICES.iter().any(|kind| kind.code() == code),
                        "{}: {code} is not a pass notice of this family",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
        }
    }

    /// Two rows that differ only in a position fingerprint apart, so one row's recording can never
    /// stand for another's.
    #[test]
    fn two_rows_that_differ_in_a_position_have_different_inputs() {
        let words = |name: &str| {
            let shape = COVERAGE
                .iter()
                .find(|row| row.name == name)
                .expect("a row of the table");
            inputs(shape)
                .unwrap()
                .into_iter()
                .find(|input| input.name == "pass")
                .expect("the pass's words")
                .digest
        };
        // The same edge, the same samples, and a coordinate Metal rounds to the same sixteenth.
        assert_ne!(
            words("sample-positions-rounded-0-5625"),
            words("sample-positions-rounded-0-5624")
        );
        assert_ne!(
            words("sample-positions-custom-left"),
            words("sample-positions-custom-right")
        );
        // And the device's own positions are not four positions.
        assert_ne!(
            words("sample-positions-default-4x"),
            words("sample-positions-custom-left")
        );
    }
}
