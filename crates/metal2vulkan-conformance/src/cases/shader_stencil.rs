//! What a fragment function's `[[stencil]]` output leaves in the stencil attachment.
//!
//! Metal takes the output's low 8 bits as **both faces' stencil reference** for the fragment,
//! replacing whatever `setStencilReferenceValue:` set, and then runs the encoder's depth/stencil
//! state against it. The question the family asks is whether a device hands the shader's value to
//! its fixed-function stencil test the way Metal does, under a state whose two faces disagree and
//! whose masks are not all ones.
//!
//! The pass, as the probe this ports drew it: a 4 by 1 `Stencil8` target whose stencil starts at
//! `0x40`, the encoder's stencil reference `0x99`, and three draws of the fullscreen triangle:
//!
//! 1. `stx_stencil`'s `stx_stencil_by_pixel`, which returns word `x` of a buffer, under both faces
//!    `Always` and `Replace`: the stencil becomes each pixel's output.
//! 2. A pipeline **without a fragment function**, under both faces `Always` and `IncrementClamp`:
//!    every pixel's stencil goes up by one. The draw carries no output, so what it increments is
//!    what draw 1 wrote, which is how the case shows the output reached the real stencil buffer
//!    rather than a value the test alone saw.
//! 3. `stx_stencil_by_pixel` again with other words, under a **front** face of `Never` beside a
//!    **back** face of `Equal` with read mask `0x0f`, write mask `0xf0`, failure `Invert` and pass
//!    `Zero`. Metal's fullscreen triangle is back-facing under the default winding, so this draw
//!    takes the back face; a device that took the front face would leave draw 2's stencil
//!    untouched, which the recorded bytes tell apart from Metal's answer in every pixel.
//!
//! Two rows, differing only in how the stencil starts: `stencil-output` clears the attachment to
//! `0x40` as the pass opens, and `stencil-output-loaded` has an earlier pass of the same command
//! buffer clear it and store, then loads it. The second row is the one that shows the output
//! tested against a stencil the pass did not itself clear.
//!
//! The expectation is **computed, not copied**: the module's `metal_rule` test applies
//! Metal's own stencil rules — compare under the read mask, pick the failure or pass operation,
//! write the result under the write mask — to the three draws, and checks the answer against the
//! table the runtime lane carries. The lane's table was measured on an M4, a machine this project
//! no longer has, so record mode measures the same pass on the oracle for the first time; the rule
//! is what says whether the two agree for a reason.
//!
//! Nothing here can fault a GPU: two render passes of three triangles each over a 4 by 1 target,
//! no loop in either fixture, and one blit of four bytes.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    PassStencil, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{CompareFunction, PixelFormat, StencilOperation};
use metal2vulkan_runtime::{Degradation, LoadAction};

/// The fixture the vertex function comes from: Metal's fullscreen triangle at a depth its vertex
/// buffer holds.
const VERTEX_FIXTURE: &str = "rt6_resolves";
const VERTEX: &str = "rt6_resolve_full";

/// The depth the triangle takes, which its vertex buffer holds as one `float`. The pass attaches
/// no depth, so any finite depth inside the clip volume draws the same; 0.5 is the probe's.
const Z: f32 = 0.5;

/// The fixture the `[[stencil]]` output comes from.
const STENCIL_FIXTURE: &str = "stx_stencil";
const FRAGMENT: &str = "stx_stencil_by_pixel";

/// The target's width; it is one pixel high.
const WIDTH: u32 = 4;

/// The target's format: one 8-bit stencil value a texel, which a blit reads back as itself.
const FORMAT: PixelFormat = PixelFormat::Stencil8;

/// The stencil the attachment starts at.
const CLEARED: u32 = 0x40;

/// The first draw's `[[stencil]]` outputs, by pixel.
const FIRST: [u32; 4] = [0x10, 0x50, 0x3f, 0x40];

/// The third draw's `[[stencil]]` outputs, by pixel.
const THIRD: [u32; 4] = [0x11, 0x00, 0x50, 0x41];

/// The encoder's stencil reference, which each fragment's output replaces. It is deliberately a
/// value no draw's rule would produce, so a device that used the encoder's reference instead of
/// the output is told apart from Metal in every pixel.
const REFERENCE: u32 = 0x99;

/// The buffer the fragment function reads its width from, at fragment buffer index 1.
fn width_bytes() -> Vec<u8> {
    WIDTH.to_le_bytes().to_vec()
}

/// A draw's outputs as the bytes of its fragment buffer 0.
fn output_bytes(outputs: [u32; 4]) -> Vec<u8> {
    outputs.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The target: a private `Stencil8` render target a shader may also read, as the lane makes its.
fn target_spec() -> TargetSpec {
    TargetSpec::read(FORMAT, [WIDTH, 1])
}

/// One face of a depth/stencil state: its compare function, failure and pass operations, the depth
/// failure operation `Keep` (the pass attaches no depth, so nothing can fail it) and all mask bits.
fn face(
    compare: CompareFunction,
    failure: StencilOperation,
    pass: StencilOperation,
) -> StencilDescriptor {
    let mut descriptor = StencilDescriptor::default();
    descriptor.stencil_compare_function = compare;
    descriptor.stencil_failure_operation = failure;
    descriptor.depth_stencil_pass_operation = pass;
    descriptor
}

/// The three draws' states, in order: both faces `Always` and `Replace`; both `Always` and
/// `IncrementClamp`; and a front face of `Never` beside a back face of `Equal` with read mask
/// `0x0f`, write mask `0xf0`, failure `Invert` and pass `Zero`.
fn states() -> [DepthStencilDescriptor; 3] {
    use CompareFunction as C;
    use StencilOperation as O;
    let state = |front: StencilDescriptor, back: StencilDescriptor| {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.front_face_stencil = front;
        descriptor.back_face_stencil = back;
        descriptor
    };
    let replace = face(C::Always, O::Keep, O::Replace);
    let increment = face(C::Always, O::Keep, O::IncrementClamp);
    let mut equal = face(C::Equal, O::Invert, O::Zero);
    equal.read_mask = 0x0f;
    equal.write_mask = 0xf0;
    [
        state(replace, replace),
        state(increment, increment),
        state(face(C::Never, O::Keep, O::Keep), equal),
    ]
}

/// The pipeline of the fullscreen triangle and the `[[stencil]]` output, over the stencil
/// attachment's format and no colour attachment.
fn output_pipeline(stencil: Handle) -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(
        StageFunction::named(VERTEX),
        Some(StageFunction::of(stencil, FRAGMENT)),
        &[],
    )
    .depth_stencil(None, Some(FORMAT))
}

/// The pipeline of the fullscreen triangle **without a fragment function**, which is how a draw
/// that only steps the stencil is stated.
fn plain_pipeline() -> RenderPipelineSpec<'static> {
    RenderPipelineSpec::of(StageFunction::named(VERTEX), None, &[])
        .depth_stencil(None, Some(FORMAT))
}

/// The layout the target is read under: one row of one-byte elements, so a diff names the pixel.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Stencil,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, 1, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: WIDTH,
        row_pitch: WIDTH,
        rows: 1,
        image_pitch: WIDTH,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// How a row's stencil attachment starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Start {
    /// The pass clears the attachment itself as it opens.
    Cleared,
    /// An earlier pass of the same command buffer clears it and stores; this one loads it.
    Loaded,
}

/// One row: the pass under one way of starting the stencil.
pub struct ShaderStencil {
    name: &'static str,
    start: Start,
}

/// The suite's two rows.
pub fn all() -> Vec<ShaderStencil> {
    vec![
        ShaderStencil {
            name: "stencil-output",
            start: Start::Cleared,
        },
        ShaderStencil {
            name: "stencil-output-loaded",
            start: Start::Loaded,
        },
    ]
}

/// The fixture `name` under the entry point `entry`.
fn fixture(name: &'static str, entry: &'static str) -> Fixture {
    Fixture { name, entry }
}

impl ShaderStencil {
    /// The stencil attachment of the row's drawing pass.
    fn attachment(&self, target: Handle) -> PassStencil {
        match self.start {
            Start::Cleared => PassStencil::cleared(target, CLEARED),
            // Built from the cleared attachment with the load action changed, rather than stated
            // field by field: the two rows then differ in their load action alone by construction,
            // and a field added to `PassStencil` by another family reaches both rows instead of
            // failing to compile here.
            Start::Loaded => {
                let mut stencil = PassStencil::cleared(target, CLEARED);
                stencil.load = LoadAction::Load;
                stencil
            }
        }
    }
}

impl Case for ShaderStencil {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Exact: every stencil byte Metal leaves is a byte we must leave. A device without
        // `VK_EXT_shader_stencil_export` runs the shader-side stencil instead, which is a route
        // and not a difference -- it must still answer Metal's bytes, so it gets no loosening
        // here. The route is **declared** rather than loosened, which is what lets the case be
        // read on a device that takes it.
        //
        // All three were measured firing for both rows on the RTX 5080: the shader-side stencil
        // itself (`shader-side-stencil`), the render pass that route splits
        // (`render-pass-split`), and the `Shared` buffer the pass reads its stencil back through
        // ([`crate::replay::shared_memory_notice`]).
        vec![
            crate::replay::notice_allowance(Degradation::ShaderSideStencil.code()),
            crate::replay::notice_allowance(Degradation::RenderPassSplit.code()),
            crate::replay::shared_memory_notice(),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (name, entry) in [(VERTEX_FIXTURE, VERTEX), (STENCIL_FIXTURE, FRAGMENT)] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{name}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        fixture(name, entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for (label, name, entry) in [
            ("vertex", VERTEX_FIXTURE, VERTEX),
            ("fragment", STENCIL_FIXTURE, FRAGMENT),
        ] {
            let held = fixture(name, entry);
            inputs.push(Input {
                name: format!("{label}:air"),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            });
            inputs.push(Input {
                name: format!("{label}:air-typed"),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            });
        }
        inputs.extend([
            Input::bytes("target", &target_spec().words()),
            // A handle the pipeline's fragment stage names never reaches the fingerprint, so a
            // stage of another library fingerprints by its entry alone; stating handle 1 here is
            // only to name the spec whose words these are.
            Input::bytes("pipeline:output", &output_pipeline(Handle(1)).words()),
            Input::bytes("pipeline:plain", &plain_pipeline().words()),
            Input::bytes("buffer:z", &Z.to_le_bytes()),
            Input::bytes("buffer:width", &width_bytes()),
            Input::bytes("buffer:first", &output_bytes(FIRST)),
            Input::bytes("buffer:third", &output_bytes(THIRD)),
            Input::bytes("reference", &REFERENCE.to_le_bytes()),
            Input::bytes("clear", &CLEARED.to_le_bytes()),
        ]);
        // The three states are the question the case asks, so each field of each face is an input.
        for (at, state) in states().iter().enumerate() {
            inputs.push(Input::bytes(format!("state:{at}"), &state_words(state)));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let not_read = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let vertex_source = fixture(VERTEX_FIXTURE, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/vertex", why))?;
        let vertex_library = made(
            sites,
            "library/vertex",
            side.library(&vertex_source.source()),
        )?;
        let stencil_source = fixture(STENCIL_FIXTURE, FRAGMENT)
            .loaded()
            .map_err(|why| not_read("library/stencil", why))?;
        let stencil_library = made(
            sites,
            "library/stencil",
            side.library(&stencil_source.source()),
        )?;

        let target = made(sites, "target", side.target(&target_spec()))?;
        let output = made(
            sites,
            "pipeline/output",
            side.render_pipeline(vertex_library, &output_pipeline(stencil_library)),
        )?;
        let plain = made(
            sites,
            "pipeline/plain",
            side.render_pipeline(vertex_library, &plain_pipeline()),
        )?;
        let mut held_states = Vec::with_capacity(3);
        for (at, state) in states().iter().enumerate() {
            held_states.push(made(
                sites,
                &format!("state/{at}"),
                side.depth_stencil_state(state),
            )?);
        }

        let z = made(sites, "buffer/z", side.buffer(&Z.to_le_bytes()))?;
        let width = made(sites, "buffer/width", side.buffer(&width_bytes()))?;
        let first = made(sites, "buffer/first", side.buffer(&output_bytes(FIRST)))?;
        let third = made(sites, "buffer/third", side.buffer(&output_bytes(THIRD)))?;
        let read_back = made(
            sites,
            "buffer/read-back",
            side.zeroed_buffer(WIDTH as usize),
        )?;

        // The three draws, each under its own state: the output binding `first`, the plain pipeline
        // binding nothing of its own, the output binding `third`.
        let mut steps = vec![
            RenderStep::vertex_buffer(z, 0),
            RenderStep::StencilReference(REFERENCE),
            RenderStep::fragment_buffer(width, 1),
        ];
        for (pipeline, state, bound) in [
            (output, held_states[0], Some(first)),
            (plain, held_states[1], None),
            (output, held_states[2], Some(third)),
        ] {
            steps.push(RenderStep::Pipeline(pipeline));
            steps.push(RenderStep::DepthStencilState(state));
            if let Some(buffer) = bound {
                steps.push(RenderStep::fragment_buffer(buffer, 0));
            }
            steps.push(RenderStep::triangle());
        }

        let mut work = Vec::new();
        // The loaded row's earlier pass: it clears the attachment and stores, encoding no draw, so
        // the drawing pass tests against a stencil it did not clear itself.
        if self.start == Start::Loaded {
            work.push(RenderWork::Pass(
                RenderPass::clearing(Vec::new())
                    .with_stencil(PassStencil::cleared(target, CLEARED)),
            ));
        }
        work.push(RenderWork::Pass(
            RenderPass::clearing(Vec::new())
                .with_stencil(self.attachment(target))
                .drawing(steps),
        ));
        work.push(RenderWork::ReadTexture {
            texture: target,
            into: read_back,
            offset: 0,
            row_pitch: WIDTH,
            size: [WIDTH, 1],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        });

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let mut program = Node::branch("program", Vec::new()).with(Node::leaf(
            "status",
            Observed::value(outcome.status.clone()),
        ));
        for (piece, refused) in work.iter().zip(&outcome.refused) {
            if let Some(failure) = refused {
                program = program.with(Node::leaf(
                    piece.call(),
                    Observed::value(failure.recorded()),
                ));
            }
        }
        if let Some(error) = &outcome.error {
            program = program.with(Node::leaf("error", Observed::value(error.clone())));
        }
        sites.push(program);

        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(read_back, 0, WIDTH as usize),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("stencil", Observed::bytes(bytes, layout())));
        Ok(())
    }
}

/// A depth/stencil descriptor's fields as the bytes a case's input fingerprint covers: the depth
/// compare and write, then each face's compare, three operations and two masks.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Metal's stencil after the three draws, by pixel, as
    /// `runtime_lane::shader_stencil::METAL` carries it. The lane measured it on an **M4** with
    /// macOS 27.0.1, a machine this project no longer has, so it is a cross-check of the rule
    /// below and never a stand-in for the recording.
    const LANE: [u8; 4] = [0x01, 0xa1, 0x00, 0x01];

    /// Metal's stencil rule for one pixel under one face: the reference and the stored value are
    /// compared under the read mask, the failure or pass operation gives the new value, and only
    /// the write mask's bits of it are stored.
    fn stencil_step(stored: u8, reference: u8, face: &StencilDescriptor) -> u8 {
        let read = face.read_mask as u8;
        let passed = match face.stencil_compare_function {
            CompareFunction::Never => false,
            CompareFunction::Always => true,
            CompareFunction::Equal => (reference & read) == (stored & read),
            CompareFunction::NotEqual => (reference & read) != (stored & read),
            CompareFunction::Less => (reference & read) < (stored & read),
            CompareFunction::LessEqual => (reference & read) <= (stored & read),
            CompareFunction::Greater => (reference & read) > (stored & read),
            CompareFunction::GreaterEqual => (reference & read) >= (stored & read),
            // The rule is stated for every compare function Metal has; a variant added to the
            // enum is a question this helper has not been told the answer to, and saying so is
            // better than guessing one.
            other => panic!("no stencil rule stated for {other:?}"),
        };
        let operation = match passed {
            true => face.depth_stencil_pass_operation,
            false => face.stencil_failure_operation,
        };
        let computed = match operation {
            StencilOperation::Keep => return stored,
            StencilOperation::Zero => 0,
            StencilOperation::Replace => reference,
            StencilOperation::IncrementClamp => stored.saturating_add(1),
            StencilOperation::DecrementClamp => stored.saturating_sub(1),
            StencilOperation::Invert => !stored,
            StencilOperation::IncrementWrap => stored.wrapping_add(1),
            StencilOperation::DecrementWrap => stored.wrapping_sub(1),
            other => panic!("no stencil rule stated for {other:?}"),
        };
        let write = face.write_mask as u8;
        (stored & !write) | (computed & write)
    }

    /// What Metal's rules give for the pass, by pixel, where the fullscreen triangle takes
    /// `back`'s face: the clear, then each draw's reference under its state's face. Draw 2 carries
    /// no `[[stencil]]` output, so its reference is the encoder's.
    fn by_the_rule(back: bool) -> [u8; 4] {
        let states = states();
        let mut stencil = [CLEARED as u8; 4];
        for (at, outputs) in [(0usize, Some(FIRST)), (1, None), (2, Some(THIRD))] {
            let state = &states[at];
            let face = match back {
                true => &state.back_face_stencil,
                false => &state.front_face_stencil,
            };
            for (pixel, value) in stencil.iter_mut().enumerate() {
                let reference = match outputs {
                    Some(outputs) => outputs[pixel] as u8,
                    None => REFERENCE as u8,
                };
                *value = stencil_step(*value, reference, face);
            }
        }
        stencil
    }

    /// Metal's rules, applied to the three draws with the triangle back-facing, give exactly the
    /// bytes the runtime lane measured on Metal. That is the case's expectation earning its place:
    /// the table is not copied into the case, it is derived, and the derivation agrees with the
    /// measurement.
    #[test]
    fn metal_rule() {
        assert_eq!(by_the_rule(true), LANE);
    }

    /// The two faces are genuinely different questions: had the triangle been front-facing, draw 3
    /// would have kept draw 2's stencil, which differs from Metal's answer in every pixel. So the
    /// recorded bytes tell the winding apart rather than passing either way.
    #[test]
    fn the_front_face_would_answer_otherwise() {
        let front = by_the_rule(false);
        assert_eq!(front, [0x11, 0x51, 0x40, 0x41]);
        for pixel in 0..4 {
            assert_ne!(
                front[pixel], LANE[pixel],
                "pixel {pixel} answers the same either way"
            );
        }
    }

    /// The encoder's reference is not what any draw's answer uses: replacing every output with it
    /// gives other bytes, so a device that ignored the `[[stencil]]` output and used
    /// `setStencilReferenceValue:` is a divergence the recording catches.
    #[test]
    fn the_encoders_reference_would_answer_otherwise() {
        let states = states();
        let mut stencil = [CLEARED as u8; 4];
        for state in &states {
            for value in stencil.iter_mut() {
                *value = stencil_step(*value, REFERENCE as u8, &state.back_face_stencil);
            }
        }
        assert_ne!(stencil, LANE);
    }

    /// The case's bookkeeping holds: two rows of distinct names, the layout covers the four
    /// stencil bytes exactly, and the three states are three different states.
    #[test]
    fn the_cases_bookkeeping_holds() {
        let rows = all();
        assert_eq!(rows.len(), 2);
        let names: BTreeSet<&str> = rows.iter().map(|row| row.name).collect();
        assert_eq!(names.len(), rows.len());
        let layout = layout();
        assert_eq!(
            layout.rows * layout.row_elements * layout.element.size(),
            WIDTH
        );
        let states = states();
        let words: BTreeSet<Vec<u8>> = states.iter().map(state_words).collect();
        assert_eq!(words.len(), states.len());
    }

    /// The two rows ask different questions of the same pass: the cleared row's attachment loads
    /// by `Clear` and the loaded row's by `Load`, so their input fingerprints differ.
    #[test]
    fn the_rows_attachments_differ() {
        let rows = all();
        let target = Handle(0);
        let cleared = rows[0].attachment(target);
        let loaded = rows[1].attachment(target);
        assert_eq!(cleared.load, LoadAction::Clear);
        assert_eq!(loaded.load, LoadAction::Load);
        assert_ne!(cleared.words(), loaded.words());
    }
}
