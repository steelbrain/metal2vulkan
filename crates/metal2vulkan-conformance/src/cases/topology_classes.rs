//! `MTLRenderPipelineDescriptor.inputPrimitiveTopology`: which of Metal's four topology classes a
//! pipeline declares, and what declaring one changes.
//!
//! The field had exactly one value in any recording before this family —
//! `super::render_descriptors` states `MTLPrimitiveTopologyClassTriangle` on its set-every-field
//! descriptor and reads it back — so `docs/CONFORMANCE-COVERAGE.md` named `Point` and `Line` as
//! two of the suite's uncompared values, and the round's cheapest remaining item was the round
//! trip. The round trip is **not** what this family is, because a round trip of a field nothing
//! reads is a comparison of our own getter against itself.
//!
//! # What the class changes, measured
//!
//! Measured in a throwaway probe on this Apple M3 Max, macOS 27.0.1, over every class crossed with
//! every drawn primitive type and with both shapes of vertex function. One cell of that cross is
//! the whole observable:
//!
//! **A vertex function that writes `[[point_size]]` is refused under `Line` and `Triangle` and
//! created under `Point` and `Unspecified`.** Metal's words are its own:
//! `Vertex shader writes point size but inputPrimitiveTopology is MTLPrimitiveTopologyClassLine`.
//! That is a **creation** answer, not a drawn one, and it is the only answer in the cross that
//! separates `Point` from `Line`, which is exactly the pair the coverage document asks for.
//!
//! **Everything else about the class is independent**, and saying so is the other half of the
//! result:
//!
//! - A vertex function that writes no point size creates under all four classes, and a triangle
//!   draw of it stores the **identical** image under each — 64 of 64 pixels lit in the probe, under
//!   `Unspecified`, `Point`, `Line` and `Triangle` alike. So the class does not constrain the
//!   drawn primitive type: a `Point` pipeline draws triangles.
//! - A line draw under `Point` draws the same line as one under `Unspecified` or `Line`, and a
//!   point draw under `Point` the same point as under `Unspecified`.
//! - A triangle draw of a **point-size-writing** function stores nothing — 0 of 64 pixels — and it
//!   stores nothing under `Unspecified` too, so that is the function's doing and not the class's.
//!   The family states it rather than crossing it, because a cell whose answer is a property of one
//!   of its axes alone is the cross this suite's own rule says not to author.
//!
//! So the two cases below are the creation cross, which is where the answers differ, and one drawn
//! control, which is where they must not.
//!
//! # The cases
//!
//! - **`topology-class-point-size-function`** makes the pipeline of a vertex function that writes
//!   `[[point_size]]` under each of the four classes, recording each creation answer at its own
//!   site, and draws a point with each pipeline Metal made. Two sites are refusals and two are
//!   pipelines, and which two is the comparison.
//! - **`topology-class-plain-function`** makes the pipeline of a function that writes no point size
//!   under each of the four classes and draws a triangle with each. Four sites, one image. This is
//!   the control that says the first case's refusals are about the point size and not about the
//!   class alone, and without it a side that refused every stated class would pass the first case
//!   for the wrong reason.
//!
//! # Nothing here can fault a GPU
//!
//! One 8 by 8 `RGBA8Unorm` target of the case's own memory per case, one primitive per draw, no
//! loop, and every pass's attachment written and read back by the same command buffer.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Answer, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, TargetSpec,
};
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveTopologyClass, PrimitiveType};
use metal2vulkan_runtime::Degradation;

/// The owned fixture of the render-state lane: its `rt6_point` writes `[[point_size]]`, its
/// `rt6_full` is the fullscreen triangle and writes none, and both take their depth from buffer 0.
const FIXTURE: &str = "rt6_render_state";

/// The vertex function that writes `[[point_size]]`: one point of 4 pixels at the centre.
const POINT_VERTEX: &str = "rt6_point";

/// The vertex function that writes no point size: the fullscreen triangle.
const PLAIN_VERTEX: &str = "rt6_full";

/// The fragment function both draws run: white.
const FRAGMENT: &str = "rt6_white";

/// The target's side.
const SIDE: u32 = 8;

/// The target's format, whose codes Metal's tile store leaves alone, so every texel is exact.
const FORMAT: PixelFormat = PixelFormat::RGBA8Unorm;

/// `Rt6Depth { z, slope }`, the two floats the vertex functions read from buffer 0: the centre of
/// the depth range and no slope, so nothing here depends on a depth test the pass does not have.
const DEPTH: [f32; 2] = [0.5, 0.0];

/// The four classes, in the order `MTLPrimitiveTopologyClass` declares them, each with the name its
/// site takes. Every value of the type, so no case of the family sweeps an ordinal.
const CLASSES: &[(&str, PrimitiveTopologyClass)] = &[
    ("unspecified", PrimitiveTopologyClass::Unspecified),
    ("point", PrimitiveTopologyClass::Point),
    ("line", PrimitiveTopologyClass::Line),
    ("triangle", PrimitiveTopologyClass::Triangle),
];

/// What one case asks about: which vertex function the pipelines are made of, and what it draws.
#[derive(Clone, Copy)]
struct Subject {
    name: &'static str,
    /// The vertex function, which either writes `[[point_size]]` or does not.
    vertex: &'static str,
    /// The primitive the draw draws, and how many vertices it takes.
    primitive: PrimitiveType,
    vertices: u32,
}

/// The two cases: the point-size function's creation cross, and the plain function's control.
const SUBJECTS: &[Subject] = &[
    Subject {
        name: "topology-class-point-size-function",
        vertex: POINT_VERTEX,
        primitive: PrimitiveType::Point,
        vertices: 1,
    },
    Subject {
        name: "topology-class-plain-function",
        vertex: PLAIN_VERTEX,
        primitive: PrimitiveType::Triangle,
        vertices: 3,
    },
];

/// `answer` recorded at `at` as **whether the pipeline was made**, answering the handle where it
/// was, and never aborting.
///
/// A `Bool` and not the typed error, which is the one judgement this family makes about its own
/// comparison. Both sides refuse a point-size-writing vertex function under `Line` and `Triangle` —
/// that agreement is the conformance fact — but they spell the refusal differently and always will:
/// the Metal side names every creation failure `NewRenderPipelineState` and carries Metal's English
/// sentence (`Vertex shader writes point size but inputPrimitiveTopology is
/// MTLPrimitiveTopologyClassLine`), while ours is a typed `Error` of our own vocabulary. Comparing
/// those words would fail every implementation including a correct one, and loosening the site to
/// `Unchecked` would stop it comparing the thing the case is for.
///
/// A venue's honest gap is still recorded, as [`crate::case::probe`] records it, so a device that
/// cannot make such a pipeline at all is not read as a refusal of ours.
fn created(sites: &mut Sites, at: &str, answer: Answer<Handle>) -> Option<Handle> {
    match answer {
        Ok(handle) => {
            sites.value(at, Value::Bool(true));
            Some(handle)
        }
        Err(failure) => {
            sites.value(at, Value::Bool(false));
            if let Some(needs) = &failure.lacks {
                sites.device_gap(at, needs.clone());
            }
            None
        }
    }
}

/// One case of the family.
pub struct TopologyClassCase {
    subject: Subject,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<TopologyClassCase> {
    SUBJECTS
        .iter()
        .map(|subject| TopologyClassCase { subject: *subject })
        .collect()
}

/// The one color state every pipeline of the family holds.
const fn color_state() -> [Option<ColorState>; 1] {
    [Some(ColorState::new(FORMAT))]
}

/// The layout the target's whole level 0 is read under: one element a channel, so a diff names the
/// row and the channel rather than a byte offset.
fn target_layout() -> ByteLayout {
    let row_pitch = 4 * SIDE;
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row_pitch,
        row_pitch,
        rows: SIDE,
        image_pitch: row_pitch * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

impl TopologyClassCase {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The pipeline the case makes under `class`.
    fn pipeline_spec<'a>(
        &self,
        colors: &'a [Option<ColorState>],
        class: PrimitiveTopologyClass,
    ) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(self.subject.vertex, FRAGMENT, colors).topology_class(class)
    }
}

impl Case for TopologyClassCase {
    fn name(&self) -> &'static str {
        self.subject.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No texel and no creation answer is loosened: a creation answer is an answer or a refusal,
        // and `RGBA8Unorm` codes survive Metal's store, so which pixels a point or a triangle lit
        // is exact on every device or it is a bug.
        //
        // The notices site is loosened for the plain function's case alone, by the one notice its
        // draw may announce. `rt6_full` computes its clip position from the vertex id under the
        // fixture's default fast math, so a device whose own float rewrites are not Metal's has the
        // translator announce it; which device that is is the device's answer and not the case's,
        // and it moves no pixel of a fullscreen triangle. `rt6_point` computes nothing and
        // announces nothing, which is why its case declares no allowance at all -- and that
        // asymmetry is itself a small confirmation that the notice is the arithmetic's and not the
        // topology class's.
        //
        // Two more are declared for both cases, and both belong to the venue rather than to a
        // topology class: a device whose own conversion of a float written to a normalized colour
        // attachment is not Metal's converts it in the shader instead (`shader-color-conversion`,
        // measured on llvmpipe and on the RTX 5080), and the positions and the read-back are
        // `Shared` ([`crate::replay::shared_memory_notice`], measured on the 5080). Neither
        // loosens a texel.
        let mut allowances: Vec<Allowance> = vec![
            crate::replay::notice_allowance(Degradation::ShaderColorConversion.code()),
            crate::replay::shared_memory_notice(),
        ];
        // And the point-size function's case alone declares what a device without
        // `shaderPointSize`'s guarantee announces for the size it writes: measured on llvmpipe for
        // that row and for neither the plain row nor any venue whose driver is Metal. It is the
        // same asymmetry the fast-math notice has, read the other way round.
        if self.subject.vertex == POINT_VERTEX {
            allowances.push(crate::replay::notice_allowance(
                Degradation::DevicePointSize.code(),
            ));
        }
        if self.subject.vertex == PLAIN_VERTEX {
            allowances.push(Allowance::new(
                format!("case/{NOTICES_SITE}/*"),
                Loosening::Unchecked,
                Justification::notice(Degradation::FastMathRewrites),
            ));
        }
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: format!("{}+{FRAGMENT}", self.subject.vertex),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let colors = color_state();
        let target = TargetSpec::plain(FORMAT, [SIDE, SIDE]).shared();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("target", &target.words()),
            Input::bytes(
                "buffer:0",
                &DEPTH
                    .iter()
                    .flat_map(|z| z.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ];
        // Every class the case states reaches the key, so a recording taken for one class cannot
        // be read for another.
        for (name, class) in CLASSES {
            inputs.push(Input::bytes(
                format!("render-pipeline:{name}"),
                &self.pipeline_spec(&colors, *class).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let colors = color_state();
        let depth: Vec<u8> = DEPTH.iter().flat_map(|z| z.to_le_bytes()).collect();
        let buffer = crate::case::made(sites, "buffer:0", side.buffer(&depth))?;
        for (name, class) in CLASSES {
            // Whether the creation answered, at its own site, and the sweep carries on past a
            // refusal: two of the four are refusals in the point-size case and that is the
            // comparison, so an abort would lose the other three.
            let made = created(
                sites,
                &format!("render-pipeline:{name}"),
                side.render_pipeline(library, &self.pipeline_spec(&colors, *class)),
            );
            let Some(state) = made else { continue };
            // Metal made it, so what it draws is comparable too: a class that created a pipeline
            // and then drew something else would pass a creation-only case.
            let target = crate::case::made(
                sites,
                &format!("target:{name}"),
                side.target(&TargetSpec::plain(FORMAT, [SIDE, SIDE]).shared()),
            )?;
            let steps = vec![
                RenderStep::Pipeline(state),
                RenderStep::vertex_buffer(buffer, 0),
                RenderStep::Draw {
                    primitive: self.subject.primitive,
                    first: 0,
                    count: self.subject.vertices,
                    instances: 1,
                },
            ];
            let outcome = side
                .render(&[RenderWork::Pass(
                    RenderPass::clearing(vec![PassColor::cleared(target, [0.0, 0.0, 0.0, 0.0])])
                        .drawing(steps),
                )])
                .map_err(|failure| stop(format!("drawn:{name}/pass"), failure))?;
            let mut node = Node::branch(format!("drawn:{name}"), Vec::new());
            let mut calls = Node::branch("calls", Vec::new());
            for (index, refused) in outcome.refused.iter().enumerate() {
                calls = calls.with(Node::leaf(
                    format!("{index}-render"),
                    Observed::value(match refused {
                        Some(failure) => failure.recorded(),
                        None => Value::ok(Value::Absent),
                    }),
                ));
            }
            node = node.with(calls);
            if let Some(failure) = outcome.refused.iter().flatten().next() {
                sites.push(node);
                return Err(Abort {
                    at: format!("drawn:{name}/calls/0-render"),
                    failure: failure.clone(),
                });
            }
            node = node.with(Node::leaf(
                "status",
                Observed::value(Value::ok(outcome.status.clone())),
            ));
            node = node.with(Node::leaf(
                "texels",
                Observed::bytes(
                    side.read_texture(target, [SIDE, SIDE], 4 * SIDE)
                        .map_err(|failure| stop(format!("drawn:{name}/texels"), failure))?,
                    target_layout(),
                ),
            ));
            sites.push(node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! What the recordings must hold, derived from the measured rule rather than asserted.
    //!
    //! The rule the probe measured: a pipeline whose vertex function writes `[[point_size]]` is
    //! created under `Unspecified` and `Point` and refused under `Line` and `Triangle`; one whose
    //! function writes none is created under all four and draws the same image under each.
    //!
    //! Both directions are checked, and the second is what keeps the family from being vacuous: if
    //! the four classes of the plain case ever stopped agreeing, or the two refusals of the
    //! point-size case ever stopped being exactly those two, a test here fails and says which.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// The classes under which Metal creates a pipeline whose vertex function writes
    /// `[[point_size]]`, measured.
    const ACCEPTS_POINT_SIZE: &[PrimitiveTopologyClass] = &[
        PrimitiveTopologyClass::Unspecified,
        PrimitiveTopologyClass::Point,
    ];

    /// Whether `subject`'s vertex function writes `[[point_size]]`, derived from which function it
    /// is rather than stated twice.
    fn writes_point_size(subject: &Subject) -> bool {
        subject.vertex == POINT_VERTEX
    }

    /// The one outcome of `name`'s recorded Metal answer; a case of this family is settled, so its
    /// runs fold into one.
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

    /// The node at `path` of `transcript`, or `None` where the transcript holds no such site.
    fn node_at<'a>(transcript: &'a Transcript, path: &str) -> Option<&'a Node> {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node.child(name)?;
        }
        Some(node)
    }

    /// The bytes the site at `path` holds.
    fn bytes(transcript: &Transcript, path: &str) -> Vec<u8> {
        let node = node_at(transcript, path).unwrap_or_else(|| panic!("no site {path}"));
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{path} holds no bytes");
        };
        bytes.clone()
    }

    /// Exactly the two classes the probe measured create a pipeline of a point-size-writing vertex
    /// function, and the other two are refusals, in the committed Metal recording.
    #[test]
    fn metal_refuses_a_point_size_function_under_the_line_and_triangle_classes() {
        let recorded = only_outcome("topology-class-point-size-function");
        for (name, class) in CLASSES {
            let path = format!("case/render-pipeline:{name}");
            let node = node_at(&recorded, &path).unwrap_or_else(|| panic!("no site {path}"));
            let accepted = ACCEPTS_POINT_SIZE.contains(class);
            assert_eq!(
                node.observed.value,
                Value::Bool(accepted),
                "{name}: Metal did not {} the point-size function",
                if accepted { "create" } else { "refuse" }
            );
            // And the drawn site is there exactly where the creation answered.
            assert_eq!(
                node_at(&recorded, &format!("case/drawn:{name}/texels")).is_some(),
                accepted,
                "{name}: a drawn site that does not follow its creation answer"
            );
        }
    }

    /// Every class draws the same image for a vertex function that writes no point size, so the
    /// class constrains neither the creation nor the drawn primitive type there.
    ///
    /// This is the family's non-vacuity guard in the other direction: the point-size case's two
    /// refusals mean something only because the same four classes refuse nothing here.
    #[test]
    fn the_class_changes_nothing_for_a_function_that_writes_no_point_size() {
        let recorded = only_outcome("topology-class-plain-function");
        let mut images = Vec::new();
        for (name, _) in CLASSES {
            let path = format!("case/render-pipeline:{name}");
            let node = node_at(&recorded, &path).unwrap_or_else(|| panic!("no site {path}"));
            assert_eq!(
                node.observed.value,
                Value::Bool(true),
                "{name}: Metal refused a function that writes no point size"
            );
            images.push(bytes(&recorded, &format!("case/drawn:{name}/texels")));
        }
        assert_eq!(images.len(), 4);
        for image in &images[1..] {
            assert_eq!(
                image, &images[0],
                "two classes drew different images for a function with no point size"
            );
        }
        // And the image is the fullscreen triangle: every texel white, so the control is a draw
        // that happened rather than four empty targets agreeing.
        assert_eq!(images[0], [0xffu8; 4].repeat((SIDE * SIDE) as usize));
    }

    /// The family states every value of `MTLPrimitiveTopologyClass` by name, which is what closes
    /// the coverage document's two missing values, and each class fingerprints its own pipeline.
    #[test]
    fn every_class_is_named_and_fingerprints_apart() {
        assert_eq!(
            CLASSES.len(),
            4,
            "MTLPrimitiveTopologyClass has four values"
        );
        for case in all() {
            let colors = color_state();
            let mut words: Vec<Vec<u8>> = Vec::new();
            for (_, class) in CLASSES {
                words.push(case.pipeline_spec(&colors, *class).words());
            }
            for (i, a) in words.iter().enumerate() {
                for b in &words[i + 1..] {
                    assert_ne!(a, b, "{}: two classes fingerprint alike", case.name());
                }
            }
            // Only the notices site may be loosened. Both cases declare the three kinds that
            // belong to the device -- its colour conversion, its point-size guarantee and its
            // memory model -- and only the case whose function has arithmetic to rewrite declares
            // the fast-math notice: the point-size function computes nothing, which is what says
            // that notice is the arithmetic's and not the topology class's.
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site",
                    case.name()
                );
            }
            let fast_math =
                Justification::Degradation(Degradation::FastMathRewrites.code().to_owned());
            let names_fast_math = case
                .allowances()
                .iter()
                .any(|allowance| allowance.justified_by == fast_math);
            assert_eq!(
                names_fast_math,
                !writes_point_size(&case.subject),
                "{}: the point-size case declares the fast-math notice, or the plain one does not",
                case.name()
            );
            assert_eq!(
                case.allowances().len(),
                2 + usize::from(writes_point_size(&case.subject)) + usize::from(names_fast_math),
                "{}: an unexpected number of declarations",
                case.name()
            );
            case.shaders().unwrap();
        }
        // The two subjects are the two shapes of vertex function, which is the axis the class
        // crosses: one writes point size and the other does not.
        let writes: Vec<bool> = SUBJECTS.iter().map(writes_point_size).collect();
        assert_eq!(writes, vec![true, false]);
    }
}
