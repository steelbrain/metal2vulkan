//! The **whole option space** of `MTLDepthStencilDescriptor`: every `MTLCompareFunction` as a depth
//! comparison and again as a stencil comparison, every `MTLStencilOperation` in each slot that can
//! fire it, `depthWriteEnabled` both ways, the two masks at their ends and at a one-bit drop, and
//! the two faces set apart under a draw that produces both windings.
//!
//! A small set of enums whose product decides every fragment's fate, and the suite compared a
//! corner of it: four of `MTLCompareFunction`'s eight values and five of `MTLStencilOperation`'s
//! eight appeared in no recording at all. The families beside this one ask what a *setter* does
//! ([`crate::cases::render_depth_stencil_state`]: the depth bias, the clip mode, the two stencil
//! references) and what a `[[stencil]]` output does ([`crate::cases::shader_stencil`]). This one
//! asks what each **value** of the state itself does, and it is built so that no value is compared
//! vacuously.
//!
//! | case | what it sweeps, and what the target then says |
//! |---|---|
//! | `depthstencil-depth-compares` | all eight compare functions as the **depth** comparison against a primed `Depth32Float`, each against a fragment below, at and above the stored depth; and `depthWriteEnabled` both ways, read by a second draw |
//! | `depthstencil-depth-codes` | the same eight against a primed `Depth16Unorm`, including a fragment whose depth is equal to the stored one **only after the format's quantisation**, and the two ends of the depth range |
//! | `depthstencil-stencil-compares` | all eight as the **stencil** comparison against a primed `Stencil8`, each against a reference below, at and above the stored stencil; and `readMask` at 0, 0xff and a mask that drops exactly one bit |
//! | `depthstencil-stencil-operations` | all eight stencil operations in `depthStencilPassOperation` and in `stencilFailureOperation`, each driven at a stored stencil of 0, 128 and 255; and `writeMask` at 0, 0xff, a nibble each way and a one-bit drop |
//! | `depthstencil-faces` | the two faces set **differently** under both windings of the same triangle, the per-face read masks, and the one-sided and two-sided spellings of the same reference |
//! | `depthstencil-depth-slot` | `depthFailureOperation`, the slot only a failed depth test reaches: all eight operations at the same three stored stencils, the four depth-test-by-stencil-test outcomes, and whether a stencil failure suppresses the depth write |
//! | `depthstencil-mask-crossings` | the two masks **crossed with each other and with the operation**: twelve rows that state `readMask` and `writeMask` apart at once, including the four operations whose answer tells a mask applied to the operation's result from one applied to its operand, and the read mask deciding which slot fires |
//! | `depthstencil-facing-faces` | the two faces set differently **crossed with `setFrontFacingWinding:` and `setCullMode:`**: three cull modes by both windings by both windings of one triangle, so a side whose culling is right and whose face choice is wrong is visible |
//!
//! # Why each sweep is not vacuous
//!
//! A sweep that cannot change what the case reads back compares nothing many times. Each of these
//! is a **complete discriminator**, stated here and asserted by the module's own tests with no
//! device:
//!
//! - **The eight compare functions against three inputs.** A fragment below, equal to and above the
//!   stored value gives each function a three-bit answer, and the eight answers are the eight
//!   distinct patterns of three bits: `Never` is `000`, `Less` `100`, `Equal` `010`, `LessEqual`
//!   `110`, `Greater` `001`, `NotEqual` `101`, `GreaterEqual` `011`, `Always` `111`. So every value
//!   differs from every other in at least one of its three cells, both as the depth comparison and
//!   as the stencil comparison. Nothing is indistinguishable by construction.
//! - **The eight stencil operations at three stored values.** Each operation's answers at a stored
//!   stencil of 0, 128 and 255 are a triple, and the eight triples are distinct: `Keep`
//!   `(0, 128, 255)`, `Zero` `(0, 0, 0)`, `Replace` `(90, 90, 90)`, `IncrementClamp`
//!   `(1, 129, 255)`, `DecrementClamp` `(0, 127, 254)`, `Invert` `(255, 127, 0)`, `IncrementWrap`
//!   `(1, 129, 0)`, `DecrementWrap` `(255, 127, 254)`. Three stored values are what it takes: at
//!   128 alone `IncrementClamp` and `IncrementWrap` agree, `DecrementClamp` and `DecrementWrap`
//!   agree, and `DecrementClamp` and `Invert` agree as well, so a sweep driven only in the middle
//!   of the range cannot tell a device that wraps where Metal clamps from one that does not. The
//!   ends are where they separate, and each is driven at both.
//! - **The masks.** `readMask` is read by a comparison whose answer the mask changes
//!   (`Equal` against a reference one bit away from the stored value, under `0xff` and under a mask
//!   that drops that one bit), and `writeMask` by an operation whose result the mask changes.
//!   `readMask` 0 makes every comparison one of 0 against 0, which is the answer `Equal` and
//!   `Less` disagree on, so the zero is read rather than assumed inert.
//! - **The two faces.** The two windings of one triangle are drawn under a state whose faces carry
//!   *different* operations, so a device that applied the front face's rules to a back-facing
//!   primitive leaves the other face's value in the stencil. The case does not assume which
//!   winding Metal calls front: it reads that from one cell and holds the rest to it, and both
//!   readings are falsifiable.
//! - **The two masks crossed.** A row that states `readMask` and `writeMask` apart at once has
//!   three wrong answers beside its right one, each a different crossing: the read mask used where
//!   the write mask belongs, the write mask used where the read mask belongs, and the two swapped.
//!   The rows are chosen so all three differ from Metal's, which neither mask's own sweep can do:
//!   the read-mask sweep writes nothing and the write-mask sweep compares under a whole read mask.
//! - **The operation crossed with the write mask.** `IncrementClamp`, `DecrementClamp`,
//!   `IncrementWrap` and `DecrementWrap` carry between the mask's bits, so a mask applied to the
//!   operation's **operand** gives a different byte from one applied to its **result** — but only
//!   at stored values where the carry or the borrow crosses the mask's edge. `Replace`, `Zero` and
//!   `Invert` commute with masking and can never tell the two apart, which is why the sweep that
//!   claimed to is named below.
//! - **The facing cross.** Culling and the face choice are read from the same two settings, so a
//!   side may cull correctly and still take the wrong face's rules. Twelve cells of
//!   `depthstencil-facing-faces` separate them: the three cull modes by both windings by both
//!   windings of one triangle, answered in the stencil rather than in whether the draw survived.
//!
//! Values indistinguishable by construction, named rather than left to be noticed:
//!
//! - At a stored stencil of **0**, `Keep`, `Zero` and `DecrementClamp` all leave 0, and
//!   `IncrementClamp` and `IncrementWrap` both leave 1, and `Invert` and `DecrementWrap` both
//!   leave 255. At **255**, `Keep` and `IncrementClamp` both leave 255, `Zero`, `Invert` and
//!   `IncrementWrap` all leave 0, and `DecrementClamp` and `DecrementWrap` both leave 254. No
//!   single stored value separates all eight; the three together do, and that is the whole reason
//!   there are three.
//! - `MTLCompareFunctionNever` and `MTLCompareFunctionAlways` ignore both operands, so their cells
//!   say nothing about the operands and only about the two constants. They are swept because the
//!   constants are what the other six are told apart from.
//! - `MTLStencilOperationReplace`, `Zero` and `Invert` give the same byte whether the write mask
//!   narrows the operation's operand or its result, because each is a bitwise function applied
//!   bit by bit. `WRITE_MASKS`'s `(IncrementWrap, 0xf0)` row is documented as the row that tells
//!   the two apart and does not: from `0xa5` both rules leave `0xa5`, the stored value itself, so
//!   that cell cannot even tell a write from a draw that wrote nothing. `CROSSINGS` carries the
//!   five rows that do separate them.
//!
//! # The depth format's quantisation
//!
//! `depthstencil-depth-codes` is the one case here whose answer the module does not state outright.
//! A `Depth16Unorm` attachment holds 65,536 codes, and `depth_texel_reads` has Metal's own reading
//! of every one of them: code `c` reads as `c / 65535`. So a fragment whose depth is a fraction of
//! a code above the stored code's own depth is **above** the stored value as a float and **equal**
//! to it once either side's conversion to the attachment's format has happened, and `Equal` and
//! `NotEqual` answer differently depending on which domain the comparison is in. The case draws
//! that fragment under all eight functions and the module's test reads the domain from the
//! `Equal` cell and holds the other thirty-one cells of the block to it, so either answer is a
//! consistent rule and a mixture of the two is a failure. That is the honest shape: what the
//! comparison's domain is on this chip is a measurement, not something to assume from a spec.
//!
//! # Bounds
//!
//! Every value list is the generated table's (`CompareFunction::ALL`, `StencilOperation::ALL`),
//! held by the module's own `the_swept_lists_are_the_generated_tables`, so a value Apple adds to
//! either enum fails this module rather than quietly lowering its ratio. The masks and the stored
//! stencils are bounded by the attachment: a `Stencil8` holds 8 bits, so 0, 0xff and the one-bit
//! drops are the whole of the mask's end behaviour, and 0, 128 and 255 are the ends and the middle
//! of the stored value's range. No device query bounds any of it — Metal states no limit on a
//! depth/stencil descriptor's fields, and `MTLDevice` answers none.
//!
//! Nothing here can fault a GPU: one pass a case, at most sixty-one draws of three vertices over a
//! 32 by 32 target, no loop in either stage, every scissor rectangle inside the attachments, every
//! vertex inside the clip volume, and no state Metal's own validation refuses.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, PassStencil, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, ScissorBox, StageFunction, TargetSpec,
};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    ColorWriteMask, CompareFunction, CullMode, PixelFormat, PrimitiveType, StencilOperation,
    Winding,
};
use metal2vulkan_runtime::{Degradation, StoreAction};

// ------------------------------------------------------------------------------------------------
// The geometry every case here shares

/// The side of every attachment, in texels.
const SIDE: u32 = 32;

/// The side of one cell: one row of the sweep's answer.
const CELL: u32 = 4;

/// Cells along one axis.
const GRID: u32 = SIDE / CELL;

/// The colour target's format: the fragment function returns four ones, so a texel is four ones or
/// the pass's clear and nothing in between.
const COLOR_FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// Bytes one texel of [`COLOR_FORMAT`] takes.
const TEXEL: u32 = 16;

/// Bytes the colour target's whole level 0 takes.
const COLOR_PLANE: u32 = TEXEL * SIDE * SIDE;

/// Bytes the stencil target's whole level 0 takes: one byte a texel.
const STENCIL_PLANE: u32 = SIDE * SIDE;

/// The clear the colour target is loaded with: a value the fragment function cannot return, so a
/// cell nothing drew into is read rather than inferred.
const CLEAR: f32 = -7.5;

/// The depth the depth attachment is cleared to before any prefill: the far plane, so a prefill
/// that wrote nothing leaves a depth most comparisons would pass, and a broken prime reads as a
/// whole block of wrong answers rather than as the sweep's.
const CLEARED_DEPTH: f64 = 1.0;

/// The owned fixture both stages come from.
const FIXTURE: &str = "cfb_sample_depth";

/// `cfb_depth_vertex`: the `float4` buffer 0 holds for the vertex, unchanged, so the case states
/// each vertex's clip position and its depth.
const VERTEX: &str = "cfb_depth_vertex";

/// `cfb_depth_white`: `float4(1.0)`.
const FRAGMENT: &str = "cfb_depth_white";

/// The entry points the shader reference names.
const ENTRIES: &str = "cfb_depth_vertex+cfb_depth_white";

/// The draw one triangle takes: three vertices, one instance.
const TRIANGLE: RenderStep = RenderStep::Draw {
    primitive: PrimitiveType::Triangle,
    first: 0,
    count: 3,
    instances: 1,
};

/// How many runs record mode takes a case of this module over, for the reason
/// `render_depth_stencil_state` states: one pass, one queue, nothing a device may reorder, so the
/// runs are expected to fold into one outcome and a test whose answer moved between runs is the
/// finding rather than one run being read as the answer.
const RUNS: u32 = 8;

/// The cell at index `at`, laid out row major: column `at % GRID`, row `at / GRID`.
fn cell(at: usize) -> ScissorBox {
    let column = (at as u32 % GRID) * CELL;
    let row = (at as u32 / GRID) * CELL;
    ScissorBox::new(
        u64::from(column),
        u64::from(row),
        u64::from(CELL),
        u64::from(CELL),
    )
}

/// The full-width band of `rows` cell rows from cell row `row`: the rectangle a prime draw covers.
fn band(row: u32, rows: u32) -> ScissorBox {
    ScissorBox::new(
        0,
        u64::from(row * CELL),
        u64::from(SIDE),
        u64::from(rows * CELL),
    )
}

/// The colour target: private, of `MTLTextureUsageRenderTarget` alone.
fn color_target() -> TargetSpec {
    TargetSpec::plain(COLOR_FORMAT, [SIDE; 2])
}

/// A depth or stencil target of `format`, as the colour target is made.
fn aspect_target(format: PixelFormat) -> TargetSpec {
    TargetSpec::plain(format, [SIDE; 2])
}

/// The layout the colour target's whole level 0 is read under.
fn color_layout() -> ByteLayout {
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

/// The layout the stencil target's whole level 0 is read under: one byte an element, so a diff
/// names the texel whose stencil moved.
fn stencil_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Stencil,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: SIDE,
        row_pitch: SIDE,
        rows: SIDE,
        image_pitch: SIDE * SIDE,
        images: 1,
        gaps: Gaps::Compared,
    }
}

// ------------------------------------------------------------------------------------------------
// The triangles

/// The clip-space corners of a triangle that covers the whole clip square and more, so what
/// confines a draw is the scissor rectangle its row states.
const CORNERS: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

/// The same three corners in the opposite order, which is the same triangle of the opposite
/// winding: the second of the two a facing sweep needs.
const REVERSED_CORNERS: [[f32; 2]; 3] = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]];

/// How far apart two triangles lie in the positions buffer: three `float4`s each, which is a
/// multiple of the `float4` the vertex function indexes.
const POSITION_STRIDE: u64 = 48;

/// One triangle of a case's positions buffer: the depth it carries at every corner, and whether it
/// is wound the other way.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Triangle {
    depth: f32,
    reversed: bool,
}

impl Triangle {
    /// The forward-wound triangle at `depth`.
    const fn at(depth: f32) -> Self {
        Self {
            depth,
            reversed: false,
        }
    }

    /// The same triangle wound the other way.
    const fn reversed(depth: f32) -> Self {
        Self {
            depth,
            reversed: true,
        }
    }
}

/// `triangles` as the bytes of the buffer bound at vertex buffer 0: twelve `float`s a triangle,
/// each vertex `(x, y, depth, 1)`.
fn positions_bytes(triangles: &[Triangle]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for triangle in triangles {
        let corners = match triangle.reversed {
            true => REVERSED_CORNERS,
            false => CORNERS,
        };
        for corner in corners {
            for word in [corner[0], corner[1], triangle.depth, 1.0] {
                bytes.extend(word.to_le_bytes());
            }
        }
    }
    bytes
}

// ------------------------------------------------------------------------------------------------
// The state vocabulary

/// The words a `MTLDepthStencilDescriptor` fingerprints as: the call's own fields, each by its raw
/// value.
///
/// Stated here as the module's own, as the suite's other depth and stencil cases state theirs:
/// [`DepthStencilDescriptor`] is a `#[non_exhaustive]` public type of another crate, so a case
/// cannot destructure it and one shared helper would be another spelling of Metal's own to keep in
/// step.
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

/// One face of a state: its comparison, its three operations and its two masks, every field stated
/// so no row of a sweep rests on a default.
fn face(
    compare: CompareFunction,
    failure: StencilOperation,
    depth_failure: StencilOperation,
    pass: StencilOperation,
    read_mask: u32,
    write_mask: u32,
) -> StencilDescriptor {
    let mut held = StencilDescriptor::default();
    held.stencil_compare_function = compare;
    held.stencil_failure_operation = failure;
    held.depth_failure_operation = depth_failure;
    held.depth_stencil_pass_operation = pass;
    held.read_mask = read_mask;
    held.write_mask = write_mask;
    held
}

/// The state whose depth comparison is `compare`, which writes depth where `write`, and whose two
/// faces are Metal's own defaults.
fn depth_state(compare: CompareFunction, write: bool) -> DepthStencilDescriptor {
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = compare;
    state.depth_write_enabled = write;
    state
}

/// The state whose two faces are `front` and `back`, under the depth comparison `depth` and no
/// depth write.
fn faced(
    depth: CompareFunction,
    front: StencilDescriptor,
    back: StencilDescriptor,
) -> DepthStencilDescriptor {
    let mut state = depth_state(depth, false);
    state.front_face_stencil = front;
    state.back_face_stencil = back;
    state
}

/// The state whose two faces are the same `face`: the one-sided spelling of a two-sided state.
fn both_faces(depth: CompareFunction, held: StencilDescriptor) -> DepthStencilDescriptor {
    faced(depth, held, held)
}

/// Every mask bit a `Stencil8` can hold.
const ALL_BITS: u32 = 0xff;

/// The stencil reference a draw of this module states where the reference is not itself the
/// question: a value none of the stored stencils and none of the operations' results equals, so a
/// cell that holds it is a cell `Replace` reached.
const TEST_REFERENCE: u32 = 0x5a;

// ------------------------------------------------------------------------------------------------
// The program a case is

/// Which pipeline a row draws under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pipe {
    /// The colour-writing pipeline: what a cell the case reads as colour is drawn by.
    Coloring,
    /// The same stages under a colour write mask of none, so a draw whose only job is to leave a
    /// depth or a stencil behind writes no colour and the colour target still reads as the clear
    /// everywhere no sweep cell drew.
    Masked,
    /// Vertex only, no fragment stage and no colour attachment: the pipeline of a case whose
    /// attachments are a stencil and, where it has one, a depth.
    NoFragment,
}

/// How a row states the stencil reference: through the call that sets both faces at once, or
/// through the one that sets them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reference {
    /// `setStencilReferenceValue:`.
    Together(u32),
    /// `setStencilFrontReferenceValue:backReferenceValue:`.
    Apart(u32, u32),
}

impl Reference {
    /// The step the reference is set by.
    fn step(self) -> RenderStep {
        match self {
            Reference::Together(value) => RenderStep::StencilReference(value),
            Reference::Apart(front, back) => RenderStep::StencilReferences { front, back },
        }
    }

    /// The words a case's input fingerprint covers.
    fn words(self) -> Vec<u8> {
        match self {
            Reference::Together(value) => {
                let mut bytes = vec![0];
                bytes.extend(value.to_le_bytes());
                bytes
            }
            Reference::Apart(front, back) => {
                let mut bytes = vec![1];
                bytes.extend(front.to_le_bytes());
                bytes.extend(back.to_le_bytes());
                bytes
            }
        }
    }
}

/// The two encoder settings that decide which **face** a primitive is, stated together because
/// neither decides it alone: `setFrontFacingWinding:` says which winding Metal calls front, and
/// `setCullMode:` says which of the two faces is thrown away before it reaches the stencil at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FacingState {
    cull: CullMode,
    winding: Winding,
}

/// One draw: the pipeline, the state, the triangle, the reference and the rectangle it is confined
/// to. A prime draw is one whose rectangle is a band rather than a cell.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Row {
    pipe: Pipe,
    /// The index of the state in the program's state list.
    state: usize,
    /// The index of the triangle in the program's triangle list.
    triangle: usize,
    reference: Reference,
    rect: ScissorBox,
    /// The cull mode and the front-facing winding the row states, where it states them. A row of
    /// `None` leaves both at Metal's own defaults, which every case but
    /// [`FacesFacing`] relies on.
    facing: Option<FacingState>,
}

/// What one case of this module runs: its attachments, the states and triangles it states, and the
/// draws in order.
struct Program {
    /// The depth attachment's format, where the case has one.
    depth: Option<PixelFormat>,
    /// Whether the case attaches a stencil target and reads it back.
    stencil: bool,
    /// Whether the case attaches the colour target and reads it back.
    color: bool,
    /// `clearStencil`, where the case has a stencil attachment.
    clear_stencil: u32,
    triangles: Vec<Triangle>,
    states: Vec<DepthStencilDescriptor>,
    rows: Vec<Row>,
}

impl Program {
    /// The pipelines the program makes, in the order a row's [`Pipe`] names them.
    fn pipes(&self) -> Vec<Pipe> {
        match self.color {
            true => vec![Pipe::Coloring, Pipe::Masked],
            false => vec![Pipe::NoFragment],
        }
    }

    /// The pipeline spec of `pipe` under the program's attachment formats.
    fn pipeline(&self, pipe: Pipe) -> RenderPipelineSpec<'static> {
        let stencil = self.stencil.then_some(STENCIL_FORMAT);
        let spec = match pipe {
            Pipe::Coloring => RenderPipelineSpec::new(VERTEX, FRAGMENT, &PLAIN),
            Pipe::Masked => RenderPipelineSpec::new(VERTEX, FRAGMENT, &MASKED),
            Pipe::NoFragment => RenderPipelineSpec::of(StageFunction::named(VERTEX), None, &[]),
        };
        spec.depth_stencil(self.depth, stencil)
    }

    /// The steps the pass encodes: every row's five settings and its draw, each stated in full so
    /// no row's answer rests on what the row before it left behind.
    fn steps(&self, pipelines: &[Handle], states: &[Handle], positions: Handle) -> Vec<RenderStep> {
        let pipes = self.pipes();
        let mut steps = Vec::new();
        for row in &self.rows {
            let at = pipes
                .iter()
                .position(|held| *held == row.pipe)
                .expect("a row names a pipeline its program does not make");
            steps.push(RenderStep::Pipeline(pipelines[at]));
            steps.push(RenderStep::DepthStencilState(states[row.state]));
            steps.push(row.reference.step());
            steps.push(RenderStep::VertexBuffer {
                buffer: positions,
                offset: row.triangle as u64 * POSITION_STRIDE,
                index: 0,
            });
            steps.push(RenderStep::Scissor(row.rect));
            if let Some(held) = row.facing {
                steps.push(RenderStep::Cull(held.cull));
                steps.push(RenderStep::FrontFacing(held.winding));
            }
            steps.push(TRIANGLE);
        }
        steps
    }

    /// The inputs the program is, as a case's fingerprint covers them.
    fn inputs(&self) -> Result<Vec<Input>, String> {
        let held = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(held.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(held.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for pipe in self.pipes() {
            inputs.push(Input::bytes(
                format!("render-pipeline:{pipe:?}"),
                &self.pipeline(pipe).words(),
            ));
        }
        if self.color {
            inputs.push(Input::bytes("target:color", &color_target().words()));
            inputs.push(Input::bytes("clear:color", &CLEAR.to_le_bytes()));
        }
        if let Some(format) = self.depth {
            inputs.push(Input::bytes("target:depth", &aspect_target(format).words()));
            inputs.push(Input::bytes("clear:depth", &CLEARED_DEPTH.to_le_bytes()));
        }
        if self.stencil {
            inputs.push(Input::bytes(
                "target:stencil",
                &aspect_target(STENCIL_FORMAT).words(),
            ));
            inputs.push(Input::bytes(
                "clear:stencil",
                &self.clear_stencil.to_le_bytes(),
            ));
        }
        inputs.push(Input::bytes(
            "buffer:positions",
            &positions_bytes(&self.triangles),
        ));
        for (at, state) in self.states.iter().enumerate() {
            inputs.push(Input::bytes(format!("state:{at}"), &state_words(state)));
        }
        let mut rows = Vec::new();
        for row in &self.rows {
            rows.push(
                self.pipes()
                    .iter()
                    .position(|p| *p == row.pipe)
                    .unwrap_or(0xff) as u8,
            );
            rows.extend((row.state as u32).to_le_bytes());
            rows.extend((row.triangle as u32).to_le_bytes());
            rows.extend(row.reference.words());
            rows.extend(row.rect.words());
            // Written **only where the row states them**, as the target's optional backing is, so
            // the field left every case recorded before it valid rather than moving their inputs.
            if let Some(held) = row.facing {
                rows.extend(held.cull.as_raw().to_le_bytes());
                rows.extend(held.winding.as_raw().to_le_bytes());
            }
        }
        inputs.push(Input::bytes("rows", &rows));
        Ok(inputs)
    }

    /// Run the program and emit what it drew: the colour target's texels where it has one, and the
    /// stencil target's bytes where it has one.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let mut pipelines = Vec::new();
        for pipe in self.pipes() {
            pipelines.push(made(
                sites,
                &format!("render-pipeline:{pipe:?}"),
                side.render_pipeline(library, &self.pipeline(pipe)),
            )?);
        }
        let color = match self.color {
            true => Some(made(sites, "target:color", side.target(&color_target()))?),
            false => None,
        };
        let depth = match self.depth {
            Some(format) => Some(made(
                sites,
                "target:depth",
                side.target(&aspect_target(format)),
            )?),
            None => None,
        };
        let stencil = match self.stencil {
            true => Some(made(
                sites,
                "target:stencil",
                side.target(&aspect_target(STENCIL_FORMAT)),
            )?),
            false => None,
        };
        let positions = made(
            sites,
            "buffer:positions",
            side.buffer(&positions_bytes(&self.triangles)),
        )?;
        let mut states = Vec::new();
        for (at, state) in self.states.iter().enumerate() {
            states.push(made(
                sites,
                &format!("state:{at}"),
                side.depth_stencil_state(state),
            )?);
        }

        let colors = color
            .map(|texture| vec![PassColor::cleared(texture, [f64::from(CLEAR); 4])])
            .unwrap_or_default();
        let mut pass = RenderPass::clearing(colors);
        if let Some(texture) = depth {
            // The depth attachment is never read back, so it is stored `DontCare`: what a case here
            // asks of the depth is what the comparison did with it, which the colour or the stencil
            // answers.
            pass = pass.with_depth(
                PassDepth::cleared(texture, CLEARED_DEPTH).stored(StoreAction::DontCare),
            );
        }
        if let Some(texture) = stencil {
            pass = pass.with_stencil(PassStencil::cleared(texture, self.clear_stencil));
        }
        pass = pass.drawing(self.steps(&pipelines, &states, positions));

        let color_back = match color {
            Some(_) => Some(made(
                sites,
                "buffer:read-color",
                side.zeroed_buffer(COLOR_PLANE as usize),
            )?),
            None => None,
        };
        let stencil_back = match stencil {
            Some(_) => Some(made(
                sites,
                "buffer:read-stencil",
                side.zeroed_buffer(STENCIL_PLANE as usize),
            )?),
            None => None,
        };
        let mut work = vec![RenderWork::Pass(pass)];
        let mut names = vec!["pass"];
        if let (Some(texture), Some(into)) = (color, color_back) {
            work.push(RenderWork::ReadTexture {
                texture,
                into,
                offset: 0,
                row_pitch: TEXEL * SIDE,
                size: [SIDE; 2],
            });
            names.push("read-color");
        }
        if let (Some(texture), Some(into)) = (stencil, stencil_back) {
            work.push(RenderWork::ReadTexture {
                texture,
                into,
                offset: 0,
                row_pitch: SIDE,
                size: [SIDE; 2],
            });
            names.push("read-stencil");
        }

        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
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

        let mut drawn = Node::branch("drawn", Vec::new());
        if let Some(buffer) = color_back {
            let bytes = call(
                sites,
                "read-back/color",
                side.read_buffer(buffer, 0, COLOR_PLANE as usize),
                |_| Value::Absent,
            )?;
            drawn = drawn.with(Node::leaf("texels", Observed::bytes(bytes, color_layout())));
        }
        if let Some(buffer) = stencil_back {
            let bytes = call(
                sites,
                "read-back/stencil",
                side.read_buffer(buffer, 0, STENCIL_PLANE as usize),
                |_| Value::Absent,
            )?;
            drawn = drawn.with(Node::leaf(
                "stencil",
                Observed::bytes(bytes, stencil_layout()),
            ));
        }
        sites.push(drawn);
        Ok(())
    }
}

/// The stencil attachment's format: one 8-bit value a texel, which a blit reads back as itself.
const STENCIL_FORMAT: PixelFormat = PixelFormat::Stencil8;

/// The colour states the two colour pipelines take: every channel, and none.
const PLAIN: [Option<ColorState>; 1] = [Some(ColorState::new(COLOR_FORMAT))];
const MASKED: [Option<ColorState>; 1] = [Some(
    ColorState::new(COLOR_FORMAT).masked(ColorWriteMask::NONE),
)];

/// The shader reference both texts of the fixture give.
fn shader_refs() -> Result<Vec<ShaderRef>, String> {
    let held = Fixture::new(FIXTURE);
    [RUNTIME_TEXT, METAL_TEXT]
        .into_iter()
        .map(|extension| {
            Ok(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRIES.to_owned(),
                source_sha256: crate::sha256_hex(held.text(extension)?.as_bytes()),
            })
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// The swept value lists
//
// Each is the generated table's own order, named value by value so the suite's own option-space
// axis reads them, and held against `CompareFunction::ALL` and `StencilOperation::ALL` by
// `tests::the_swept_lists_are_the_generated_tables`.

/// Every `MTLCompareFunction`, in the SDK's declaration order.
const COMPARES: [CompareFunction; 8] = [
    CompareFunction::Never,
    CompareFunction::Less,
    CompareFunction::Equal,
    CompareFunction::LessEqual,
    CompareFunction::Greater,
    CompareFunction::NotEqual,
    CompareFunction::GreaterEqual,
    CompareFunction::Always,
];

/// Every `MTLStencilOperation`, in the SDK's declaration order.
const OPERATIONS: [StencilOperation; 8] = [
    StencilOperation::Keep,
    StencilOperation::Zero,
    StencilOperation::Replace,
    StencilOperation::IncrementClamp,
    StencilOperation::DecrementClamp,
    StencilOperation::Invert,
    StencilOperation::IncrementWrap,
    StencilOperation::DecrementWrap,
];

/// The three stored stencils every operation is driven at: the two ends of a `Stencil8`'s range,
/// where the clamping and the wrapping operations separate, and the middle, where they do not.
const STARTS: [u32; 3] = [0, 128, 255];

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(DepthCompares),
        Box::new(DepthCodes),
        Box::new(StencilCompares),
        Box::new(StencilOperations),
        Box::new(Faces),
        Box::new(DepthSlot),
        Box::new(MaskCrossings),
        Box::new(FacesFacing),
    ]
}

// ------------------------------------------------------------------------------------------------
// depthstencil-depth-compares

/// The depth the prefill writes over the whole target, and the depth every sweep draw is compared
/// against: an exact binary fraction well inside `[0, 1]`.
const STORED_DEPTH: f32 = 0.5;

/// The three depths one compare function's row is driven at: below the stored depth, exactly at it,
/// and above it. Exact binary fractions, each a quarter of the range from the stored depth, so no
/// cell's answer turns on rounding.
const DEPTH_INPUTS: [f32; 3] = [0.25, STORED_DEPTH, 0.75];

/// The depth the `depthWriteEnabled` probe draws at: between the depth its first draw writes and
/// the stored depth, so it passes `Less` against one and fails it against the other.
const PROBE_DEPTH: f32 = 0.375;

/// All eight compare functions as the depth comparison, and `depthWriteEnabled` both ways.
pub struct DepthCompares;

/// The cells the `depthWriteEnabled` rows take, after the sweep's twenty-four.
const WRITE_ON: usize = 24;
const WRITE_OFF: usize = 25;
const WRITE_FAILED: usize = 26;

impl DepthCompares {
    /// The program: a colour target and a `Depth32Float` depth attachment.
    fn program() -> Program {
        // Triangle 0 primes the depth; 1..4 are the sweep's three inputs; 4 is the probe's depth.
        let triangles = vec![
            Triangle::at(STORED_DEPTH),
            Triangle::at(DEPTH_INPUTS[0]),
            Triangle::at(DEPTH_INPUTS[1]),
            Triangle::at(DEPTH_INPUTS[2]),
            Triangle::at(PROBE_DEPTH),
        ];
        // State 0 primes the depth. States 1..9 are the sweep's, one a compare function, each
        // without a depth write so one cell's draw cannot reach the next cell's comparison.
        let mut states = vec![depth_state(CompareFunction::Always, true)];
        states.extend(COMPARES.map(|compare| depth_state(compare, false)));
        // State 9 is `Less` with the depth write **on**, which is the one field the three
        // `depthWriteEnabled` cells differ in.
        states.push(depth_state(CompareFunction::Less, true));
        let writing = states.len() - 1;
        let testing = |at: usize| 1 + at;

        let reference = Reference::Together(0);
        let row = |pipe, state, triangle, rect| Row {
            pipe,
            state,
            triangle,
            reference,
            rect,
            facing: None,
        };
        // The prime: the whole target to the stored depth, writing no colour.
        let mut rows = vec![row(Pipe::Masked, 0, 0, band(0, GRID))];
        // The sweep: a column a compare function, a row an input.
        for (input, _) in DEPTH_INPUTS.iter().enumerate() {
            for (at, _) in COMPARES.iter().enumerate() {
                rows.push(row(
                    Pipe::Coloring,
                    testing(at),
                    1 + input,
                    cell(input * GRID as usize + at),
                ));
            }
        }
        // `depthWriteEnabled` on: a masked draw at 0.25 that passes `Less` and writes its depth,
        // then a probe at 0.375 that `Less` now fails. The cell comes out clear.
        rows.push(row(Pipe::Masked, writing, 1, cell(WRITE_ON)));
        rows.push(row(Pipe::Coloring, testing(1), 4, cell(WRITE_ON)));
        // The same pair with the write off: the stored depth is still 0.5, so the probe passes and
        // the cell is drawn. The two cells differ in that one field alone.
        rows.push(row(Pipe::Masked, testing(1), 1, cell(WRITE_OFF)));
        rows.push(row(Pipe::Coloring, testing(1), 4, cell(WRITE_OFF)));
        // The write on, under a draw whose comparison **fails**: a depth of 0.75 against `Less`,
        // which must leave the stored depth alone, so the probe passes and the cell is drawn.
        rows.push(row(Pipe::Masked, writing, 3, cell(WRITE_FAILED)));
        rows.push(row(Pipe::Coloring, testing(1), 4, cell(WRITE_FAILED)));

        Program {
            depth: Some(PixelFormat::Depth32Float),
            stencil: false,
            color: true,
            clear_stencil: 0,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for DepthCompares {
    fn name(&self) -> &'static str {
        "depthstencil-depth-compares"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing loosened and no notice declared: a texel is four ones or the pass's clear, both
        // of which a `RGBA32Float` attachment stores exactly, and the fragment function does no
        // arithmetic a device's float bounds could reach.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-depth-codes

/// The codes a `Depth16Unorm` attachment has.
const CODES: f64 = 65_535.0;

/// The code the first block's depth is primed to: the middle of the range, so a code either side of
/// it is inside it.
const CODE: u32 = 32_768;

/// The depth code `code` holds, as Metal's own reading of a `Depth16Unorm` texel says
/// (`depth_texel_reads`: code `c` reads as `c / 65535`).
fn depth_of_code(code: u32) -> f32 {
    (f64::from(code) / CODES) as f32
}

/// A depth inside code `code`: `fraction` of a code above the code's own value.
///
/// **Both of this module's fractions lie in `[0, 0.5)`**, which is the interval on which Metal's
/// two candidate conversions agree: rounding to nearest takes `(code - 0.5, code + 0.5)` to `code`
/// and truncating takes `[code, code + 1)` to it, so a depth stated here lands on `code` whichever
/// one a device does. A code's own depth will not do for that — `32768 / 65535` is not exactly
/// representable as an `f32`, and the nearest `f32` to it multiplies back to a hair *below* 32768,
/// so a device that truncates would put it on 32767. The whole block would then be asking about the
/// conversion's rounding instead of about the comparison's domain, which is a different question.
fn inside_code(code: u32, fraction: f64) -> f32 {
    ((f64::from(code) + fraction) / CODES) as f32
}

/// The fraction a depth that only has to land on its code takes.
const LOW_FRACTION: f64 = 0.1;

/// The fraction the quantising depth takes: the same code, a float above [`LOW_FRACTION`]'s.
const HIGH_FRACTION: f64 = 0.4;

/// The four depths the middle block's rows are driven at: two codes below, the stored code's own
/// depth, the same code higher up inside it, and two codes above.
fn code_inputs() -> [f32; 4] {
    let mut held = [
        inside_code(CODE - 2, LOW_FRACTION),
        inside_code(CODE, LOW_FRACTION),
        inside_code(CODE, LOW_FRACTION),
        inside_code(CODE + 2, LOW_FRACTION),
    ];
    held[QUANTISED_INPUT] = inside_code(CODE, HIGH_FRACTION);
    held
}

/// Which of [`code_inputs`] is the quantisation question: above the stored depth as a float, on the
/// stored code once the attachment holds it.
const QUANTISED_INPUT: usize = 2;

/// The depth the middle block is primed to, which is also its second input, so that input is equal
/// to the stored value in either domain and the block's reading rests on the third input alone.
fn code_prime() -> f32 {
    inside_code(CODE, LOW_FRACTION)
}

/// The two depths a range-end block is driven at: the end's own depth, which is exact in an `f32`
/// and on the end's code under either conversion, and a depth one code inside the range.
///
/// `inward` is the code one step into the range, so the second input is below the high end and
/// above the low one in **both** domains, which is what keeps the end blocks free of the
/// quantisation question the middle block asks.
fn end_inputs(code: u32, inward: u32) -> [f32; 2] {
    [depth_of_code(code), inside_code(inward, LOW_FRACTION)]
}

/// The two ends of the depth range, as codes.
const LOW_CODE: u32 = 0;
const HIGH_CODE: u32 = 65_535;

/// All eight compare functions against a `Depth16Unorm` attachment, including the fragment that is
/// equal to the stored depth only after the format's quantisation, and the two ends of the range.
pub struct DepthCodes;

impl DepthCodes {
    /// The program: a colour target and a `Depth16Unorm` depth attachment in three bands, each
    /// primed to its own code.
    fn program() -> Program {
        // Triangles: the three primes, then the middle block's four inputs, then each end's two.
        let mut triangles = vec![
            Triangle::at(code_prime()),
            Triangle::at(depth_of_code(LOW_CODE)),
            Triangle::at(depth_of_code(HIGH_CODE)),
        ];
        triangles.extend(code_inputs().map(Triangle::at));
        triangles.extend(end_inputs(LOW_CODE, LOW_CODE + 1).map(Triangle::at));
        triangles.extend(end_inputs(HIGH_CODE, HIGH_CODE - 1).map(Triangle::at));
        let middle = 3;
        let low = middle + code_inputs().len();
        let high = low + 2;

        let mut states = vec![depth_state(CompareFunction::Always, true)];
        states.extend(COMPARES.map(|compare| depth_state(compare, false)));

        let reference = Reference::Together(0);
        let row = |state, triangle, rect| Row {
            pipe: Pipe::Coloring,
            state,
            triangle,
            reference,
            rect,
            facing: None,
        };
        let prime = |triangle, rect| Row {
            pipe: Pipe::Masked,
            state: 0,
            triangle,
            reference,
            rect,
            facing: None,
        };
        // Three primes, one a band: the middle code over the first four cell rows, the low code
        // over the next two and the high code over the last two.
        let mut rows = vec![
            prime(0, band(0, 4)),
            prime(1, band(4, 2)),
            prime(2, band(6, 2)),
        ];
        // The middle block: a column a compare function, a row an input.
        for (input, _) in code_inputs().iter().enumerate() {
            for (at, _) in COMPARES.iter().enumerate() {
                rows.push(row(
                    1 + at,
                    middle + input,
                    cell(input * GRID as usize + at),
                ));
            }
        }
        // The two end blocks, each two rows of the same eight columns.
        for (block, start) in [(low, 4usize), (high, 6usize)] {
            for input in 0..2usize {
                for (at, _) in COMPARES.iter().enumerate() {
                    rows.push(row(
                        1 + at,
                        block + input,
                        cell((start + input) * GRID as usize + at),
                    ));
                }
            }
        }

        Program {
            depth: Some(PixelFormat::Depth16Unorm),
            stencil: false,
            color: true,
            clear_stencil: 0,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for DepthCodes {
    fn name(&self) -> &'static str {
        "depthstencil-depth-codes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-stencil-compares

/// The stencil the attachment is cleared to, which every comparison of the stencil sweep is held
/// against.
const STORED_STENCIL: u32 = 0x40;

/// The three references one compare function's row is driven at: below the stored stencil, equal to
/// it, and above it.
const STENCIL_INPUTS: [u32; 3] = [0x20, STORED_STENCIL, 0x60];

/// The depth every draw of the two stencil-only cases carries: any depth inside the clip volume
/// draws the same, as neither case attaches a depth.
const FLAT: f32 = 0.5;

/// All eight compare functions as the stencil comparison, and `readMask` at its ends and at a
/// one-bit drop.
pub struct StencilCompares;

/// The cells the `readMask` rows take, after the sweep's twenty-four.
const MASK_CELLS: usize = 24;

/// The `readMask` rows: the mask, the reference, and why the cell answers as it does.
///
/// Every row compares `Equal` or `Less` of the reference against [`STORED_STENCIL`] under the mask,
/// so each is `(ref & mask) OP (0x40 & mask)`:
///
/// | mask | reference | comparison | answer |
/// |---|---|---|---|
/// | `0xff` | `0x41` | `Equal` | `0x41 != 0x40`, clear — the control the one-bit drop is read against |
/// | `0xfe` | `0x41` | `Equal` | the mask drops bit 0 of the reference, `0x40 == 0x40`, drawn |
/// | `0xff` | `0x40` | `Equal` | `0x40 == 0x40`, drawn |
/// | `0x00` | `0x41` | `Equal` | `0 == 0`, drawn — the zero mask is not inert |
/// | `0x00` | `0x41` | `Less` | `0 < 0` is false, clear — which is what says the zero mask was read |
/// | `0x3f` | `0x40` | `Equal` | the mask drops bit 6 of the **stored** value, `0x00 == 0x00`, drawn |
const READ_MASKS: [(u32, u32, CompareFunction, bool); 6] = [
    (0xff, 0x41, CompareFunction::Equal, false),
    (0xfe, 0x41, CompareFunction::Equal, true),
    (0xff, 0x40, CompareFunction::Equal, true),
    (0x00, 0x41, CompareFunction::Equal, true),
    (0x00, 0x41, CompareFunction::Less, false),
    (0x3f, 0x40, CompareFunction::Equal, true),
];

impl StencilCompares {
    /// The program: a colour target and a `Stencil8` attachment, no depth.
    fn program() -> Program {
        let triangles = vec![Triangle::at(FLAT)];
        // The sweep's eight states, then the six the mask rows take. Every operation is `Keep` and
        // every write mask is none, so no draw changes the stencil and each cell is independent of
        // every other.
        let kept = |compare, read_mask| {
            both_faces(
                CompareFunction::Always,
                face(
                    compare,
                    StencilOperation::Keep,
                    StencilOperation::Keep,
                    StencilOperation::Keep,
                    read_mask,
                    0,
                ),
            )
        };
        let mut states: Vec<DepthStencilDescriptor> =
            COMPARES.iter().map(|c| kept(*c, ALL_BITS)).collect();
        for (mask, _, compare, _) in READ_MASKS {
            states.push(kept(compare, mask));
        }

        let mut rows = Vec::new();
        for (input, held) in STENCIL_INPUTS.iter().enumerate() {
            for (at, _) in COMPARES.iter().enumerate() {
                rows.push(Row {
                    pipe: Pipe::Coloring,
                    state: at,
                    triangle: 0,
                    reference: Reference::Together(*held),
                    rect: cell(input * GRID as usize + at),
                    facing: None,
                });
            }
        }
        for (at, (_, reference, _, _)) in READ_MASKS.iter().enumerate() {
            rows.push(Row {
                pipe: Pipe::Coloring,
                state: COMPARES.len() + at,
                triangle: 0,
                reference: Reference::Together(*reference),
                rect: cell(MASK_CELLS + at),
                facing: None,
            });
        }

        Program {
            depth: None,
            stencil: true,
            color: true,
            clear_stencil: STORED_STENCIL,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for StencilCompares {
    fn name(&self) -> &'static str {
        "depthstencil-stencil-compares"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-stencil-operations

/// The stored stencil the `writeMask` rows are primed to: a value whose two nibbles differ and
/// which no operation's result on it equals, so a masked write is read rather than inferred.
const MASKED_START: u32 = 0xa5;

/// The `writeMask` rows: the operation, the mask, and the stencil the rule leaves.
///
/// Every row starts at [`MASKED_START`] with the reference [`TEST_REFERENCE`] under a stencil
/// comparison of `Always`, so the pass operation fires and the write is
/// `(start & !mask) | (result & mask)`:
///
/// | operation | mask | result | written |
/// |---|---|---|---|
/// | `Replace` | `0xff` | `0x5a` | `0x5a` |
/// | `Replace` | `0x00` | `0x5a` | `0xa5` — nothing written |
/// | `Replace` | `0x0f` | `0x5a` | `0xaa` |
/// | `Replace` | `0xfe` | `0x5a` | `0x5b` — the mask drops exactly one bit |
/// | `Invert` | `0xff` | `0x5a` | `0x5a` |
/// | `Invert` | `0x0f` | `0x5a` | `0xaa` |
/// | `IncrementWrap` | `0xf0` | `0xa6` | `0xa5` — the mask keeps the nibble the step did not touch. It does **not** tell a mask applied after the operation from one applied to its operand: from `0xa5` both rules leave `0xa5`, which is the stored value itself, so this cell cannot tell a masked write from a draw that wrote nothing either. `CROSSINGS` carries the rows that can ([`MaskCrossings`]) |
/// | `Zero` | `0x0f` | `0x00` | `0xa0` |
const WRITE_MASKS: [(StencilOperation, u32); 8] = [
    (StencilOperation::Replace, 0xff),
    (StencilOperation::Replace, 0x00),
    (StencilOperation::Replace, 0x0f),
    (StencilOperation::Replace, 0xfe),
    (StencilOperation::Invert, 0xff),
    (StencilOperation::Invert, 0x0f),
    (StencilOperation::IncrementWrap, 0xf0),
    (StencilOperation::Zero, 0x0f),
];

/// The cell row the `writeMask` rows take, after the two slots' six.
const WRITE_MASK_ROW: u32 = 6;

/// Every stencil operation in `depthStencilPassOperation` and in `stencilFailureOperation`, each at
/// a stored stencil of 0, 128 and 255, and `writeMask` at its ends and inside.
pub struct StencilOperations;

impl StencilOperations {
    /// The program: a `Stencil8` attachment alone, read back as the answer.
    fn program() -> Program {
        let triangles = vec![Triangle::at(FLAT)];
        // State 0 primes a band to the reference the row carries.
        let mut states = vec![both_faces(
            CompareFunction::Always,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Replace,
                ALL_BITS,
                ALL_BITS,
            ),
        )];
        // The pass slot: the comparison is `Always`, so the fragment passes and the pass operation
        // fires. One state an operation.
        let pass_slot = states.len();
        states.extend(OPERATIONS.map(|op| {
            both_faces(
                CompareFunction::Always,
                face(
                    CompareFunction::Always,
                    StencilOperation::Keep,
                    StencilOperation::Keep,
                    op,
                    ALL_BITS,
                    ALL_BITS,
                ),
            )
        }));
        // The failure slot: the comparison is `Never`, so the fragment fails the stencil test and
        // only the failure operation fires. The depth-failure slot cannot be reached without a
        // depth attachment, which is `depthstencil-depth-slot`'s question.
        let fail_slot = states.len();
        states.extend(OPERATIONS.map(|op| {
            both_faces(
                CompareFunction::Always,
                face(
                    CompareFunction::Never,
                    op,
                    StencilOperation::Keep,
                    StencilOperation::Keep,
                    ALL_BITS,
                    ALL_BITS,
                ),
            )
        }));
        let masks = states.len();
        states.extend(WRITE_MASKS.map(|(op, mask)| {
            both_faces(
                CompareFunction::Always,
                face(
                    CompareFunction::Always,
                    StencilOperation::Keep,
                    StencilOperation::Keep,
                    op,
                    ALL_BITS,
                    mask,
                ),
            )
        }));

        let draw = |state, reference, rect| Row {
            pipe: Pipe::NoFragment,
            state,
            triangle: 0,
            reference: Reference::Together(reference),
            rect,
            facing: None,
        };
        // The primes: the pass slot's second and third rows and the failure slot's, plus the
        // `writeMask` row. The first row of each slot is the pass's own clear of 0.
        let mut rows = Vec::new();
        for (at, start) in STARTS.iter().enumerate() {
            if *start == 0 {
                continue;
            }
            rows.push(draw(0, *start, band(at as u32, 1)));
            rows.push(draw(0, *start, band(STARTS.len() as u32 + at as u32, 1)));
        }
        rows.push(draw(0, MASKED_START, band(WRITE_MASK_ROW, 1)));
        // The two slots: a column an operation, a row a stored stencil.
        for (slot, base) in [(pass_slot, 0usize), (fail_slot, STARTS.len())] {
            for (start, _) in STARTS.iter().enumerate() {
                for (at, _) in OPERATIONS.iter().enumerate() {
                    rows.push(draw(
                        slot + at,
                        TEST_REFERENCE,
                        cell((base + start) * GRID as usize + at),
                    ));
                }
            }
        }
        // The `writeMask` row.
        for (at, _) in WRITE_MASKS.iter().enumerate() {
            rows.push(draw(
                masks + at,
                TEST_REFERENCE,
                cell(WRITE_MASK_ROW as usize * GRID as usize + at),
            ));
        }

        Program {
            depth: None,
            stencil: true,
            color: false,
            clear_stencil: 0,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for StencilOperations {
    fn name(&self) -> &'static str {
        "depthstencil-stencil-operations"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Exact: every stencil byte Metal leaves is a byte we must leave.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-faces

/// The stencil the facing case's attachment is cleared to.
const FACE_START: u32 = 0x40;

/// A reference one bit away from [`FACE_START`], which the per-face read masks are read with.
const FACE_NEAR: u32 = 0x41;

/// A reference nothing equals under any of the case's masks, so a face given it fails its
/// comparison.
const FACE_FAR: u32 = 0x20;

/// The two faces set differently under both windings of the same triangle, the per-face read masks,
/// and the one-sided and two-sided spellings of the same reference.
pub struct Faces;

impl Faces {
    /// The program: a `Stencil8` attachment alone, read back as the answer.
    ///
    /// The two faces' pass operations are `IncrementClamp` and `DecrementClamp`, which leave `0x41`
    /// and `0x3f` from `0x40`: two values that say which face's rules a winding took. Nothing here
    /// assumes which winding Metal calls front — the module's test reads that from one cell and
    /// holds the rest to it.
    fn program() -> Program {
        let triangles = vec![Triangle::at(FLAT), Triangle::reversed(FLAT)];
        let inc = |read_mask| {
            face(
                CompareFunction::Equal,
                StencilOperation::Zero,
                StencilOperation::Keep,
                StencilOperation::IncrementClamp,
                read_mask,
                ALL_BITS,
            )
        };
        let dec = |read_mask| {
            face(
                CompareFunction::Equal,
                StencilOperation::Zero,
                StencilOperation::Keep,
                StencilOperation::DecrementClamp,
                read_mask,
                ALL_BITS,
            )
        };
        let states = vec![
            // 0: the front face steps up, the back face steps down.
            faced(CompareFunction::Always, inc(ALL_BITS), dec(ALL_BITS)),
            // 1: the two swapped, so each winding's answer swaps with it.
            faced(CompareFunction::Always, dec(ALL_BITS), inc(ALL_BITS)),
            // 2: the one-sided spelling of state 0's front face, on both faces, so a winding's
            // answer no longer depends on which face it took.
            both_faces(CompareFunction::Always, inc(ALL_BITS)),
            // 3: the two faces' **read masks** apart. Under the reference `0x41` the front face's
            // whole-byte comparison fails and its failure operation (`Zero`) fires, while the back
            // face's masked comparison drops bit 0 and passes, so its pass operation does.
            faced(CompareFunction::Always, inc(ALL_BITS), dec(0xfe)),
        ];

        let row = |state, triangle, reference, rect| Row {
            pipe: Pipe::NoFragment,
            state,
            triangle,
            reference,
            rect,
            facing: None,
        };
        let together = Reference::Together(FACE_START);
        let rows = vec![
            // Cells 0 and 1: state 0 under each winding. Their answers differ, and which is which
            // is Metal's facing convention.
            row(0, 0, together, cell(0)),
            row(0, 1, together, cell(1)),
            // Cells 2 and 3: the faces swapped, so each winding's answer swaps.
            row(1, 0, together, cell(2)),
            row(1, 1, together, cell(3)),
            // Cells 4 and 5: the one-sided state, where the two windings answer alike. Read beside
            // cells 0 and 1, this is the one-sided and two-sided spellings of the same rule.
            row(2, 0, together, cell(4)),
            row(2, 1, together, cell(5)),
            // Cell 6: the two-sided reference call with both values equal, which must answer
            // exactly as cell 0's one-sided call did.
            row(0, 0, Reference::Apart(FACE_START, FACE_START), cell(6)),
            // Cell 7: the two references apart, the back one equalling nothing. Under the winding
            // of cell 1 the face that draws fails its comparison, so its failure operation fires
            // and the cell holds neither of cell 1's two values.
            row(0, 1, Reference::Apart(FACE_START, FACE_FAR), cell(7)),
            // Cells 8 and 9: the per-face read masks, under a reference one bit from the stored
            // value. One winding's face compares the whole byte and fails; the other's drops that
            // bit and passes.
            row(3, 0, Reference::Together(FACE_NEAR), cell(8)),
            row(3, 1, Reference::Together(FACE_NEAR), cell(9)),
        ];

        Program {
            depth: None,
            stencil: true,
            color: false,
            clear_stencil: FACE_START,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for Faces {
    fn name(&self) -> &'static str {
        "depthstencil-faces"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-depth-slot

/// The stencil the matrix and depth-write cells start at: a value none of the three slots' results
/// equals, so a cell that still holds it is a cell no slot reached.
const SLOT_START: u32 = 0x5a;

/// The cell row the four depth-test-by-stencil-test cells and the two depth-write cells take, after
/// the operation sweep's three.
const MATRIX_ROW: u32 = 3;

/// `depthFailureOperation`, the one slot a failed depth test reaches: every operation at the same
/// three stored stencils, the four outcomes of the two tests, and whether a stencil failure
/// suppresses the depth write.
///
/// **This is the one case of the module whose pass attaches a depth and a stencil as two separate
/// textures**, which is the shape Metal takes (a `Depth32Float` beside a `Stencil8`, of identical
/// dimensions, which its validation requires) and which a Vulkan subpass has no counterpart for:
/// the runtime renders such a pass into a scratch image of the combined format and copies each
/// aspect in and out. Nothing else in the suite compares that path against Metal.
///
/// Measured on an Apple M3 Max (macOS 27.0.1, MoltenVK): **every value the case reads is Metal's
/// own**, and the one difference is that our side announces
/// `metal2vulkan-degrade-separate-depth-stencil`, which the case therefore names. That notice is a
/// `Performance` degradation -- Metal's answer at a cost Metal does not pay, the two copies a
/// single-sampled aspect takes through a staging buffer -- so the values stay exact and nothing but
/// the notice site is loosened. The allowance fails the case if the notice stops firing, so it
/// cannot outlive its reason.
pub struct DepthSlot;

impl DepthSlot {
    /// The program: a colour target, a `Depth32Float` depth attachment and a `Stencil8` stencil
    /// attachment.
    fn program() -> Program {
        // Triangle 0 primes the depth and is the matrix's failing depth; 1 is its passing depth;
        // 2 is the depth-write cells' first draw and 3 their probe.
        let triangles = vec![
            Triangle::at(STORED_DEPTH),
            Triangle::at(DEPTH_INPUTS[0]),
            Triangle::at(DEPTH_INPUTS[2]),
            Triangle::at(PROBE_DEPTH),
        ];

        // 0: prime the depth over the whole target.
        let mut states = vec![depth_state(CompareFunction::Always, true)];
        // 1: prime a band's stencil to the row's reference, with the depth comparison `Always` so
        // the pass operation is the one that fires.
        states.push(both_faces(
            CompareFunction::Always,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Replace,
                ALL_BITS,
                ALL_BITS,
            ),
        ));
        // 2..10: the depth-failure slot. The depth comparison is `Never`, so every fragment fails
        // the depth test however the attachment's depths lie, and the stencil comparison is
        // `Always`, so the stencil test passes first: exactly the one path that reaches this slot.
        let slot = states.len();
        states.extend(OPERATIONS.map(|op| {
            both_faces(
                CompareFunction::Never,
                face(
                    CompareFunction::Always,
                    StencilOperation::Keep,
                    op,
                    StencilOperation::Keep,
                    ALL_BITS,
                    ALL_BITS,
                ),
            )
        }));
        // The matrix state: one state whose three slots carry three different operations, so the
        // stencil the cell holds names which slot fired. `Less` against the primed depth of 0.5
        // decides the depth test from the triangle's own depth rather than from a constant
        // comparison, and `Equal` against the row's reference decides the stencil test.
        let matrix = states.len();
        states.push(faced(
            CompareFunction::Less,
            face(
                CompareFunction::Equal,
                StencilOperation::Zero,
                StencilOperation::Invert,
                StencilOperation::IncrementClamp,
                ALL_BITS,
                ALL_BITS,
            ),
            face(
                CompareFunction::Equal,
                StencilOperation::Zero,
                StencilOperation::Invert,
                StencilOperation::IncrementClamp,
                ALL_BITS,
                ALL_BITS,
            ),
        ));
        // The depth-write pair: a state that writes depth under a passing depth comparison and a
        // **failing** stencil comparison, and the same with the stencil comparison passing.
        let write_failing = states.len();
        states.push(both_faces(
            CompareFunction::Always,
            face(
                CompareFunction::Never,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Keep,
                ALL_BITS,
                0,
            ),
        ));
        {
            let last = states.len() - 1;
            states[last].depth_write_enabled = true;
        }
        let write_passing = states.len();
        states.push(both_faces(
            CompareFunction::Always,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Keep,
                ALL_BITS,
                0,
            ),
        ));
        {
            let last = states.len() - 1;
            states[last].depth_write_enabled = true;
        }
        // The probe: `Less` without a write, and no stencil effect at all.
        let probe = states.len();
        states.push(both_faces(
            CompareFunction::Less,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Keep,
                ALL_BITS,
                0,
            ),
        ));

        let row = |pipe, state, triangle, reference, rect| Row {
            pipe,
            state,
            triangle,
            reference,
            rect,
            facing: None,
        };
        let plain = Reference::Together(TEST_REFERENCE);
        // The depth prime over the whole target, writing no colour.
        let mut rows = vec![row(Pipe::Masked, 0, 0, plain, band(0, GRID))];
        // The stencil primes: the operation sweep's second and third rows, and the matrix row.
        for (at, start) in STARTS.iter().enumerate() {
            if *start == 0 {
                continue;
            }
            rows.push(row(
                Pipe::Masked,
                1,
                0,
                Reference::Together(*start),
                band(at as u32, 1),
            ));
        }
        rows.push(row(
            Pipe::Masked,
            1,
            0,
            Reference::Together(SLOT_START),
            band(MATRIX_ROW, 1),
        ));
        // The sweep: a column an operation, a row a stored stencil. Masked, because what these
        // cells answer is the stencil.
        for (start, _) in STARTS.iter().enumerate() {
            for (at, _) in OPERATIONS.iter().enumerate() {
                rows.push(row(
                    Pipe::Masked,
                    slot + at,
                    0,
                    plain,
                    cell(start * GRID as usize + at),
                ));
            }
        }
        // The four outcomes, in the order the module's table states them: both tests passing, the
        // depth test failing, the stencil test failing, and both failing. The colour target says
        // whether the fragment survived and the stencil says which slot fired.
        let base = MATRIX_ROW as usize * GRID as usize;
        for (at, (triangle, reference)) in [
            (1, SLOT_START),
            (2, SLOT_START),
            (1, FACE_FAR),
            (2, FACE_FAR),
        ]
        .into_iter()
        .enumerate()
        {
            rows.push(row(
                Pipe::Coloring,
                matrix,
                triangle,
                Reference::Together(reference),
                cell(base + at),
            ));
        }
        // The two depth-write cells: a draw that writes depth under a failing stencil test, and the
        // same under a passing one, each read by a probe at a depth between the two. The first
        // cell's probe draws where Metal suppressed the write and is clear where it did not.
        for (at, state) in [write_failing, write_passing].into_iter().enumerate() {
            let held = cell(base + 4 + at);
            rows.push(row(Pipe::Masked, state, 1, plain, held));
            rows.push(row(Pipe::Coloring, probe, 3, plain, held));
        }

        Program {
            depth: Some(PixelFormat::Depth32Float),
            stencil: true,
            color: true,
            clear_stencil: 0,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for DepthSlot {
    fn name(&self) -> &'static str {
        "depthstencil-depth-slot"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Only the notice set is loosened. A pass whose depth and stencil attachments are two
        // textures renders through a scratch image of the combined format, which the runtime
        // announces; it is a `Performance` degradation, so it buys the copies and not the answer,
        // and every texel and every stencil byte of this case still compares exactly.
        vec![crate::replay::notice_allowance(
            Degradation::SeparateDepthStencil.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-mask-crossings

/// One row of `depthstencil-mask-crossings`: the stored stencil, the two masks, the comparison and
/// the reference that together decide which slot fires, the two slots a pass with no depth
/// attachment can reach, and the stencil Metal's rule leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Crossing {
    /// The stencil the row's cell is primed to before its draw.
    start: u32,
    read_mask: u32,
    write_mask: u32,
    compare: CompareFunction,
    reference: u32,
    pass: StencilOperation,
    failure: StencilOperation,
    /// The stencil the row leaves, which the module's own test derives again from Metal's rule
    /// rather than reading it back from a recording.
    left: u32,
    /// Whether applying the **write mask to the operation's operand** instead of to its result
    /// moves this row's answer: the one crossing of the two masks with the operation that no
    /// single-value sweep of either can see.
    operand_visible: bool,
}

/// The rows, each a crossing of the two masks with the operation or with the slot choice.
///
/// Every row primes its own cell to `start`, states both masks, and draws once under a state whose
/// `depthStencilPassOperation` and `stencilFailureOperation` differ. The attachment is a `Stencil8`
/// alone, so the depth-failure slot is unreachable and which of the two remaining slots fires is
/// the comparison's answer **under the read mask**.
///
/// | start | read | write | compare | reference | pass | failure | left | why this row is here |
/// |---|---|---|---|---|---|---|---|---|
/// | `0x5a` | `0x0f` | `0xf0` | `Equal` | `0x1a` | `Replace` | `Zero` | `0x1a` | the two masks apart, each in the other's reach: applying either where the other belongs moves the cell three different ways |
/// | `0x5a` | `0xf0` | `0x0f` | `Equal` | `0x51` | `Replace` | `Zero` | `0x51` | the mirror of it, so neither answer can be the other mask's by coincidence |
/// | `0x10` | `0xff` | `0x0f` | `Always` | `0x5a` | `DecrementClamp` | `Keep` | `0x1f` | `DecrementClamp` borrows across the mask's edge: masking its operand clamps at 0 and leaves `0x10` |
/// | `0x0f` | `0xff` | `0xf0` | `Always` | `0x5a` | `IncrementClamp` | `Keep` | `0x1f` | the carry out of the low nibble is what the mask then stores |
/// | `0xff` | `0xff` | `0xf0` | `Always` | `0x5a` | `IncrementWrap` | `Keep` | `0x0f` | the wrap's modulus: 256, not the mask's width |
/// | `0x0f` | `0xff` | `0xf0` | `Always` | `0x5a` | `IncrementWrap` | `Keep` | `0x1f` | the same wrap below its modulus, so the row above is read against a step that does not wrap |
/// | `0x02` | `0xff` | `0x55` | `Always` | `0x5a` | `DecrementWrap` | `Keep` | `0x03` | a mask whose bits are not one run, which is where `DecrementWrap`'s borrow separates |
/// | `0x5a` | `0xff` | `0x0f` | `Always` | `0x5a` | `Invert` | `Keep` | `0x55` | the control: `Invert` commutes with masking, so this row's answer is the same under either rule and it is swept for the mask and not for the operand |
/// | `0x5a` | `0x00` | `0xff` | `Equal` | `0x1a` | `Replace` | `Zero` | `0x1a` | a read mask of 0 compares 0 against 0 and passes, while the write mask is whole |
/// | `0x5a` | `0xff` | `0x00` | `Equal` | `0x1a` | `Replace` | `Invert` | `0x5a` | the failure slot fires and a write mask of 0 stores none of it |
/// | `0x5a` | `0xf0` | `0xff` | `Equal` | `0x5b` | `Replace` | `Invert` | `0x5b` | the read mask drops the bit the reference differs in, so the **pass** slot fires |
/// | `0x5a` | `0x0f` | `0xff` | `Equal` | `0x5b` | `Replace` | `Invert` | `0xa5` | the same reference under the other mask, so the **failure** slot fires: the read mask crossed with the slot choice |
const CROSSINGS: [Crossing; 12] = [
    Crossing {
        start: 0x5a,
        read_mask: 0x0f,
        write_mask: 0xf0,
        compare: CompareFunction::Equal,
        reference: 0x1a,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Zero,
        left: 0x1a,
        operand_visible: false,
    },
    Crossing {
        start: 0x5a,
        read_mask: 0xf0,
        write_mask: 0x0f,
        compare: CompareFunction::Equal,
        reference: 0x51,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Zero,
        left: 0x51,
        operand_visible: false,
    },
    Crossing {
        start: 0x10,
        read_mask: ALL_BITS,
        write_mask: 0x0f,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::DecrementClamp,
        failure: StencilOperation::Keep,
        left: 0x1f,
        operand_visible: true,
    },
    Crossing {
        start: 0x0f,
        read_mask: ALL_BITS,
        write_mask: 0xf0,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::IncrementClamp,
        failure: StencilOperation::Keep,
        left: 0x1f,
        operand_visible: true,
    },
    Crossing {
        start: 0xff,
        read_mask: ALL_BITS,
        write_mask: 0xf0,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::IncrementWrap,
        failure: StencilOperation::Keep,
        left: 0x0f,
        operand_visible: true,
    },
    Crossing {
        start: 0x0f,
        read_mask: ALL_BITS,
        write_mask: 0xf0,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::IncrementWrap,
        failure: StencilOperation::Keep,
        left: 0x1f,
        operand_visible: true,
    },
    Crossing {
        start: 0x02,
        read_mask: ALL_BITS,
        write_mask: 0x55,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::DecrementWrap,
        failure: StencilOperation::Keep,
        left: 0x03,
        operand_visible: true,
    },
    Crossing {
        start: 0x5a,
        read_mask: ALL_BITS,
        write_mask: 0x0f,
        compare: CompareFunction::Always,
        reference: TEST_REFERENCE,
        pass: StencilOperation::Invert,
        failure: StencilOperation::Keep,
        left: 0x55,
        operand_visible: false,
    },
    Crossing {
        start: 0x5a,
        read_mask: 0x00,
        write_mask: ALL_BITS,
        compare: CompareFunction::Equal,
        reference: 0x1a,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Zero,
        left: 0x1a,
        operand_visible: false,
    },
    Crossing {
        start: 0x5a,
        read_mask: ALL_BITS,
        write_mask: 0x00,
        compare: CompareFunction::Equal,
        reference: 0x1a,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Invert,
        left: 0x5a,
        operand_visible: false,
    },
    Crossing {
        start: 0x5a,
        read_mask: 0xf0,
        write_mask: ALL_BITS,
        compare: CompareFunction::Equal,
        reference: 0x5b,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Invert,
        left: 0x5b,
        operand_visible: false,
    },
    Crossing {
        start: 0x5a,
        read_mask: 0x0f,
        write_mask: ALL_BITS,
        compare: CompareFunction::Equal,
        reference: 0x5b,
        pass: StencilOperation::Replace,
        failure: StencilOperation::Invert,
        left: 0xa5,
        operand_visible: false,
    },
];

/// `readMask` crossed with `writeMask` and with the operation: the one mask narrows the
/// comparison's operands and the other narrows the store, and a side that applies either where the
/// other belongs passes every single-value sweep of both.
///
/// The family beside this one sweeps each mask alone
/// ([`StencilCompares`]'s read-mask rows, [`StencilOperations`]'s write-mask rows), and that is
/// what leaves the crossing open: a read-mask row whose every operation is `Keep` cannot see a
/// write, and a write-mask row whose read mask is whole cannot see a comparison narrowed. Twelve
/// rows here state both masks at once.
///
/// It also closes a sweep that could not fire. `WRITE_MASKS`'s `(IncrementWrap, 0xf0)` row is
/// documented as what tells a write mask applied **after** the operation from one applied to its
/// operand, and it cannot: from `0xa5` both rules leave `0xa5`, which is the stored value itself,
/// so the row's cell is also indistinguishable from a draw that wrote nothing at all. The five
/// rows here marked `operand_visible` are the ones whose two answers differ, held apart by
/// `tests::the_crossings_see_an_operand_masked_operation_and_the_old_table_cannot` with no
/// device.
pub struct MaskCrossings;

impl MaskCrossings {
    /// The program: a `Stencil8` attachment alone, read back as the answer.
    fn program() -> Program {
        let triangles = vec![Triangle::at(FLAT)];
        // State 0 primes a cell to the row's own stored stencil.
        let mut states = vec![both_faces(
            CompareFunction::Always,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::Replace,
                ALL_BITS,
                ALL_BITS,
            ),
        )];
        states.extend(CROSSINGS.map(|row| {
            both_faces(
                CompareFunction::Always,
                face(
                    row.compare,
                    row.failure,
                    StencilOperation::Keep,
                    row.pass,
                    row.read_mask,
                    row.write_mask,
                ),
            )
        }));

        let draw = |state, reference, rect| Row {
            pipe: Pipe::NoFragment,
            state,
            triangle: 0,
            reference: Reference::Together(reference),
            rect,
            facing: None,
        };
        let mut rows = Vec::new();
        for (at, row) in CROSSINGS.iter().enumerate() {
            rows.push(draw(0, row.start, cell(at)));
            rows.push(draw(1 + at, row.reference, cell(at)));
        }

        Program {
            depth: None,
            stencil: true,
            color: false,
            clear_stencil: 0,
            triangles,
            states,
            rows,
        }
    }
}

impl Case for MaskCrossings {
    fn name(&self) -> &'static str {
        "depthstencil-mask-crossings"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Exact: every stencil byte Metal leaves is a byte we must leave.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

// ------------------------------------------------------------------------------------------------
// depthstencil-facing-faces

/// The three cull modes, in the generated table's order.
const CULLS: [CullMode; 3] = [CullMode::None, CullMode::Front, CullMode::Back];

/// Both windings, in the generated table's order.
const WINDINGS: [Winding; 2] = [Winding::Clockwise, Winding::CounterClockwise];

/// The two faces set differently, crossed with `setFrontFacingWinding:` and `setCullMode:`.
///
/// Which face's rules a primitive takes is not a property of the primitive: it is the primitive's
/// winding read against the encoder's `frontFacingWinding`, and `cullMode` then decides whether
/// that face reaches the stencil at all. Three settings decide one answer, and each of them is
/// already swept alone — [`Faces`] sets the two faces apart under Metal's **default** winding and
/// no culling, and `render-state-facing` sweeps the cull mode and the winding over a **colour**
/// target, where what is read is whether the primitive drew. Neither can see a side whose culling
/// is right and whose face choice is wrong, which is the one thing twelve cells of this shape read.
///
/// Twelve cells: three cull modes, by both windings, by both windings of one triangle. The state's
/// front face steps the stencil up and its back face steps it down, both under a stencil
/// comparison of `Always`, so a cell holds `0x41` where the front face's rules applied, `0x3f`
/// where the back face's did, and the cleared `0x40` where the primitive was culled. The three
/// values are distinct, so no cell's answer is two readings at once.
///
/// Nothing here assumes which winding Metal calls front for the module's own triangle: the module's test reads that from one cell and holds the other eleven to it, and
/// both readings are falsifiable.
pub struct FacesFacing;

impl FacesFacing {
    /// The program: a `Stencil8` attachment alone, read back as the answer.
    fn program() -> Program {
        let triangles = vec![Triangle::at(FLAT), Triangle::reversed(FLAT)];
        // One state: the front face steps up, the back face steps down. The comparison is `Always`
        // on both faces, so the pass operation fires whichever face the primitive is and the cell
        // answers the face choice alone.
        let states = vec![faced(
            CompareFunction::Always,
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::IncrementClamp,
                ALL_BITS,
                ALL_BITS,
            ),
            face(
                CompareFunction::Always,
                StencilOperation::Keep,
                StencilOperation::Keep,
                StencilOperation::DecrementClamp,
                ALL_BITS,
                ALL_BITS,
            ),
        )];

        let mut rows = Vec::new();
        for (at, (cull, winding, triangle)) in facing_cells().into_iter().enumerate() {
            rows.push(Row {
                pipe: Pipe::NoFragment,
                state: 0,
                triangle,
                reference: Reference::Together(FACE_START),
                rect: cell(at),
                facing: Some(FacingState { cull, winding }),
            });
        }

        Program {
            depth: None,
            stencil: true,
            color: false,
            clear_stencil: FACE_START,
            triangles,
            states,
            rows,
        }
    }
}

/// The twelve cells in the order they are laid out: a cull mode, a front-facing winding and which
/// of the two windings of one triangle is drawn.
fn facing_cells() -> Vec<(CullMode, Winding, usize)> {
    let mut cells = Vec::new();
    for cull in CULLS {
        for winding in WINDINGS {
            for triangle in [0, 1] {
                cells.push((cull, winding, triangle));
            }
        }
    }
    cells
}

impl Case for FacesFacing {
    fn name(&self) -> &'static str {
        "depthstencil-facing-faces"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn record_runs(&self) -> u32 {
        RUNS
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Exact: every stencil byte Metal leaves is a byte we must leave. Metal's cull mode and
        // front-facing winding are both encoder state the runtime carries into the pipeline it is
        // about to draw with, so no pass of this case is split and nothing is announced.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::program().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::program().run(side, sites)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Transcript;
    use std::cmp::Ordering;
    use std::collections::{BTreeMap, BTreeSet};

    /// Cells a target holds, which only a check over the whole target counts.
    const CELLS: usize = (GRID * GRID) as usize;

    /// What the fragment function returns in every channel.
    const WHITE: [f32; 4] = [1.0; 4];

    // --------------------------------------------------------------------------------------------
    // Metal's own rules, applied here rather than copied from a recording

    /// Whether `function` holds of an `ordering` of the incoming value against the stored one.
    ///
    /// Metal compares the **incoming** value on the left: a depth comparison of `Less` keeps a
    /// fragment whose depth is less than the stored depth, and a stencil comparison of `Less` keeps
    /// one whose masked reference is less than the masked stored stencil.
    fn holds(function: CompareFunction, ordering: Ordering) -> bool {
        match function {
            CompareFunction::Never => false,
            CompareFunction::Less => ordering == Ordering::Less,
            CompareFunction::Equal => ordering == Ordering::Equal,
            CompareFunction::LessEqual => ordering != Ordering::Greater,
            CompareFunction::Greater => ordering == Ordering::Greater,
            CompareFunction::NotEqual => ordering != Ordering::Equal,
            CompareFunction::GreaterEqual => ordering != Ordering::Less,
            CompareFunction::Always => true,
            other => panic!(
                "{} is a compare function this module states no rule for",
                other.name()
            ),
        }
    }

    /// What `operation` leaves in an 8-bit stencil that holds `start`, under the reference
    /// `reference`.
    fn applied(operation: StencilOperation, start: u32, reference: u32) -> u32 {
        match operation {
            StencilOperation::Keep => start,
            StencilOperation::Zero => 0,
            StencilOperation::Replace => reference & 0xff,
            StencilOperation::IncrementClamp => (start + 1).min(0xff),
            StencilOperation::DecrementClamp => start.saturating_sub(1),
            StencilOperation::Invert => !start & 0xff,
            StencilOperation::IncrementWrap => (start + 1) & 0xff,
            StencilOperation::DecrementWrap => (start + 0xff) & 0xff,
            other => panic!(
                "{} is a stencil operation this module states no rule for",
                other.name()
            ),
        }
    }

    /// The steps one row of a program encodes: its six settings, and the two facing calls where
    /// the row states them.
    fn row_steps(row: &Row) -> usize {
        match row.facing {
            Some(_) => 8,
            None => 6,
        }
    }

    /// What the stencil holds once `result` is written into a stencil of `start` under `mask`.
    fn written(start: u32, result: u32, mask: u32) -> u32 {
        (start & !mask & 0xff) | (result & mask & 0xff)
    }

    // --------------------------------------------------------------------------------------------
    // The non-vacuity proofs: no device, no recording

    /// **Every value this module sweeps is the generated table's**, in its order and with nothing
    /// left out.
    ///
    /// The denominator of a depth ratio has to come from the SDK rather than from a list written
    /// here, or a value Apple adds lowers the ratio with nothing failing. The lists are spelled out
    /// value by value in the module's own text — which is what the suite's option-space axis reads
    /// — and this is what holds them to `CompareFunction::ALL` and `StencilOperation::ALL`.
    #[test]
    fn the_swept_lists_are_the_generated_tables() {
        assert_eq!(COMPARES.as_slice(), CompareFunction::ALL);
        assert_eq!(OPERATIONS.as_slice(), StencilOperation::ALL);
        // And the rules above answer for every one of them, so a new value panics loudly in this
        // test rather than taking a wildcard arm in a sweep's expectation.
        for function in CompareFunction::ALL {
            for ordering in [Ordering::Less, Ordering::Equal, Ordering::Greater] {
                let _ = holds(*function, ordering);
            }
        }
        for operation in StencilOperation::ALL {
            let _ = applied(*operation, 1, 2);
        }
    }

    /// **Three inputs tell all eight compare functions apart**, which is what makes the two
    /// comparison sweeps non-vacuous: each value's three cells are a distinct pattern, so changing
    /// the value changes at least one cell of its row.
    #[test]
    fn three_inputs_tell_the_eight_compare_functions_apart() {
        let pattern = |function: CompareFunction| {
            [Ordering::Less, Ordering::Equal, Ordering::Greater].map(|o| holds(function, o))
        };
        let seen: BTreeMap<[bool; 3], Vec<&str>> =
            COMPARES.iter().fold(BTreeMap::new(), |mut held, function| {
                held.entry(pattern(*function))
                    .or_default()
                    .push(function.name());
                held
            });
        assert_eq!(
            seen.len(),
            COMPARES.len(),
            "two compare functions answer the three inputs alike, so the sweep compares them \
             vacuously: {seen:?}"
        );
        // And no two inputs are the same input: one below, one equal, one above, in each sweep.
        // The depths are constants, so their order is a compile-time fact and is stated as one.
        const {
            assert!(DEPTH_INPUTS[0] < STORED_DEPTH && DEPTH_INPUTS[2] > STORED_DEPTH);
        }
        assert_eq!(DEPTH_INPUTS[1], STORED_DEPTH);
        assert!(STENCIL_INPUTS[0] < STORED_STENCIL && STENCIL_INPUTS[2] > STORED_STENCIL);
        assert_eq!(STENCIL_INPUTS[1], STORED_STENCIL);
        // Every stencil value the sweep states fits the eight bits a `Stencil8` holds.
        for value in STENCIL_INPUTS.iter().chain(STARTS.iter()) {
            assert_eq!(*value & 0xff, *value);
        }
    }

    /// **The three stored stencils tell all eight operations apart, and no one of them does.**
    ///
    /// This is the row that catches an implementation that wraps where Metal clamps: the clamping
    /// and the wrapping pair agree everywhere except at the end of the range they clamp at. The
    /// test states both halves — that the three together separate all eight, and which pairs each
    /// single value leaves indistinguishable — so a later edit that drops a stored value fails here
    /// with the pairs it would stop telling apart.
    #[test]
    fn the_three_stored_stencils_tell_the_eight_operations_apart() {
        let triple =
            |operation: StencilOperation| STARTS.map(|s| applied(operation, s, TEST_REFERENCE));
        let seen: BTreeMap<[u32; 3], Vec<&str>> =
            OPERATIONS
                .iter()
                .fold(BTreeMap::new(), |mut held, operation| {
                    held.entry(triple(*operation))
                        .or_default()
                        .push(operation.name());
                    held
                });
        assert_eq!(
            seen.len(),
            OPERATIONS.len(),
            "two stencil operations answer all three stored values alike: {seen:?}"
        );
        // The triples the module's prose states, so the prose is checked rather than trusted.
        assert_eq!(triple(StencilOperation::Keep), [0, 128, 255]);
        assert_eq!(triple(StencilOperation::Zero), [0, 0, 0]);
        assert_eq!(triple(StencilOperation::Replace), [0x5a, 0x5a, 0x5a]);
        assert_eq!(triple(StencilOperation::IncrementClamp), [1, 129, 255]);
        assert_eq!(triple(StencilOperation::DecrementClamp), [0, 127, 254]);
        assert_eq!(triple(StencilOperation::Invert), [255, 127, 0]);
        assert_eq!(triple(StencilOperation::IncrementWrap), [1, 129, 0]);
        assert_eq!(triple(StencilOperation::DecrementWrap), [255, 127, 254]);
        // No single stored value separates all eight, which is why there are three. The pairs each
        // one leaves together are named, not left to be noticed.
        let together = |start: u32| -> BTreeSet<(&str, &str)> {
            let mut pairs = BTreeSet::new();
            for (at, one) in OPERATIONS.iter().enumerate() {
                for other in &OPERATIONS[at + 1..] {
                    if applied(*one, start, TEST_REFERENCE)
                        == applied(*other, start, TEST_REFERENCE)
                    {
                        pairs.insert((one.name(), other.name()));
                    }
                }
            }
            pairs
        };
        for start in STARTS {
            assert!(
                !together(start).is_empty(),
                "a stored stencil of {start} separates all eight operations on its own, which the \
                 module's prose says no single value does"
            );
        }
        // The middle of the range is where the clamping and the wrapping pair agree, and the ends
        // are where each separates.
        assert!(together(128).contains(&(
            StencilOperation::IncrementClamp.name(),
            StencilOperation::IncrementWrap.name()
        )));
        assert!(together(128).contains(&(
            StencilOperation::DecrementClamp.name(),
            StencilOperation::DecrementWrap.name()
        )));
        assert!(!together(255).contains(&(
            StencilOperation::IncrementClamp.name(),
            StencilOperation::IncrementWrap.name()
        )));
        assert!(!together(0).contains(&(
            StencilOperation::DecrementClamp.name(),
            StencilOperation::DecrementWrap.name()
        )));
    }

    /// **Each `readMask` row's stated answer is the rule's answer**, so the table in the module's
    /// prose is derived rather than asserted, and a zero mask is read rather than assumed inert.
    #[test]
    fn the_read_mask_rows_are_what_the_masked_comparison_says() {
        for (mask, reference, compare, stated) in READ_MASKS {
            let ours = holds(compare, (reference & mask).cmp(&(STORED_STENCIL & mask)));
            assert_eq!(
                ours,
                stated,
                "readMask {mask:#04x} with reference {reference:#04x} under {}: the rule says \
                 {ours} and the row says {stated}",
                compare.name()
            );
        }
        // The masks reach the three places the sweep owes: both ends, and a mask that drops exactly
        // one bit of the reference.
        let masks: BTreeSet<u32> = READ_MASKS.iter().map(|(mask, _, _, _)| *mask).collect();
        assert!(masks.contains(&0x00) && masks.contains(&0xff));
        assert!(masks.contains(&0xfe), "no mask drops exactly one bit");
        assert_eq!((0xffu32 ^ 0xfe).count_ones(), 1);
        // And the two rows that share the zero mask disagree, which is what says the zero was read.
        let zeroed: Vec<bool> = READ_MASKS
            .iter()
            .filter(|(mask, _, _, _)| *mask == 0)
            .map(|(_, _, _, stated)| *stated)
            .collect();
        assert_eq!(zeroed, vec![true, false]);
    }

    /// **Each `writeMask` row's stated answer is the rule's answer**, and the masks reach both ends,
    /// a nibble each way and a one-bit drop.
    #[test]
    fn the_write_mask_rows_are_what_the_masked_write_says() {
        let got: Vec<u32> = WRITE_MASKS
            .iter()
            .map(|(operation, mask)| {
                written(
                    MASKED_START,
                    applied(*operation, MASKED_START, TEST_REFERENCE),
                    *mask,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![0x5a, 0xa5, 0xaa, 0x5b, 0x5a, 0xaa, 0xa5, 0xa0],
            "the masked writes are not the ones the module's table states"
        );
        let masks: BTreeSet<u32> = WRITE_MASKS.iter().map(|(_, mask)| *mask).collect();
        assert!(masks.contains(&0x00) && masks.contains(&0xff));
        assert!(masks.contains(&0x0f) && masks.contains(&0xf0));
        assert!(masks.contains(&0xfe), "no mask drops exactly one bit");
        // The `IncrementWrap` row is the one that tells a mask applied after the operation from one
        // applied to its operand: masked afterwards it leaves the stencil alone, and applied to the
        // masked operand it would not.
        assert_eq!(
            written(
                MASKED_START,
                applied(
                    StencilOperation::IncrementWrap,
                    MASKED_START,
                    TEST_REFERENCE
                ),
                0xf0
            ),
            MASKED_START
        );
        assert_ne!(
            applied(
                StencilOperation::IncrementWrap,
                MASKED_START & 0xf0,
                TEST_REFERENCE
            ),
            MASKED_START
        );
        // A mask of none writes nothing, whatever the operation says.
        for operation in OPERATIONS {
            assert_eq!(
                written(
                    MASKED_START,
                    applied(operation, MASKED_START, TEST_REFERENCE),
                    0
                ),
                MASKED_START
            );
        }
    }

    /// **The quantising depth is above the stored code's own depth as a float and on that code once
    /// the attachment holds it** — which is the whole of what the `Depth16Unorm` block asks.
    ///
    /// Stated over the arithmetic, so it holds with no device: the offset is below half a code
    /// either way round, so the depth lands on the code whether the conversion rounds to nearest or
    /// truncates, and it is about a hundred steps of an `f32`'s last place away from the code's own
    /// depth, which is not a difference a device could lose.
    #[test]
    fn the_quantising_depth_is_a_float_above_the_code_it_lands_on() {
        let stored = code_prime();
        let nudged = code_inputs()[QUANTISED_INPUT];
        assert!(
            nudged > stored,
            "{nudged} is not above {stored}, so the quantisation cell asks nothing"
        );
        let steps = nudged.to_bits() - stored.to_bits();
        assert!(
            steps > 32,
            "{steps} steps of the last place is too close to read"
        );
        // Every depth the case states lands on the code it is meant to under **both** candidate
        // conversions, so the block asks about the comparison's domain and not about the
        // conversion's rounding.
        let both = |depth: f32, want: u32, what: &str| {
            let scaled = f64::from(depth) * CODES;
            assert_eq!(
                scaled.round() as u32,
                want,
                "{what} ({depth}) rounds to the wrong code"
            );
            assert_eq!(
                scaled as u32, want,
                "{what} ({depth}) truncates to the wrong code"
            );
        };
        both(stored, CODE, "the middle block's prime");
        for (input, want) in code_inputs()
            .into_iter()
            .zip([CODE - 2, CODE, CODE, CODE + 2])
        {
            both(input, want, "a middle input");
        }
        // The two ends are the ends: each end's own depth is exact in an `f32`, so it needs no
        // fraction, and the inward input is one code in under either conversion.
        assert_eq!(depth_of_code(LOW_CODE), 0.0);
        assert_eq!(depth_of_code(HIGH_CODE), 1.0);
        for (end, inward, want) in [
            (LOW_CODE, LOW_CODE + 1, LOW_CODE + 1),
            (HIGH_CODE, HIGH_CODE - 1, HIGH_CODE - 1),
        ] {
            let held = end_inputs(end, inward);
            both(held[0], end, "a range end");
            both(held[1], want, "a range end's inward input");
            assert_ne!(held[0], held[1]);
        }
        // The two fractions are inside the interval the two conversions agree on, which is the one
        // fact every line above rests on.
        for fraction in [LOW_FRACTION, HIGH_FRACTION] {
            assert!((0.0..0.5).contains(&fraction));
        }
    }

    // --------------------------------------------------------------------------------------------
    // The programs' own shape

    /// Every program of the family, with the case it belongs to.
    fn programs() -> Vec<(&'static str, Program)> {
        vec![
            (DepthCompares.name(), DepthCompares::program()),
            (DepthCodes.name(), DepthCodes::program()),
            (StencilCompares.name(), StencilCompares::program()),
            (StencilOperations.name(), StencilOperations::program()),
            (Faces.name(), Faces::program()),
            (DepthSlot.name(), DepthSlot::program()),
            (MaskCrossings.name(), MaskCrossings::program()),
            (FacesFacing.name(), FacesFacing::program()),
        ]
    }

    /// **Every row of every program names a state, a triangle and a pipeline it has**, and every
    /// rectangle lies inside the attachments.
    ///
    /// A rectangle reaching past an attachment is the one shape of `setScissorRect:` Metal's own
    /// validation calls a programming error, and what plain Metal's driver programs for one is
    /// unmeasured, so no case here states one and the guard holds before a device is asked.
    #[test]
    fn every_row_names_what_its_program_holds_and_stays_inside_the_target() {
        // The check sees a rectangle that reaches past the target, so a green run means the rows
        // are inside rather than that the check cannot fire.
        assert!(ScissorBox::new(0, 0, u64::from(SIDE) + 1, 1).past([SIDE; 2]));
        for (name, program) in programs() {
            let pipes = program.pipes();
            assert!(!program.rows.is_empty(), "{name} encodes no draw");
            for (at, row) in program.rows.iter().enumerate() {
                assert!(
                    pipes.contains(&row.pipe),
                    "{name} row {at} names {:?}, which it does not make",
                    row.pipe
                );
                assert!(
                    row.state < program.states.len(),
                    "{name} row {at}: no such state"
                );
                assert!(
                    row.triangle < program.triangles.len(),
                    "{name} row {at}: no such triangle"
                );
                assert!(
                    !row.rect.past([SIDE; 2]),
                    "{name} row {at}: {:?} reaches past the target",
                    row.rect
                );
            }
            // Every triangle lies inside the clip volume, so nothing a case reads was clipped and
            // the sweep's answer is the comparison's.
            for (at, triangle) in program.triangles.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&triangle.depth),
                    "{name} triangle {at} is at depth {}, outside the clip volume",
                    triangle.depth
                );
            }
            // The pass encodes six steps a row: the pipeline, the state, the reference, the vertex
            // buffer, the scissor rectangle and the draw. Every row states all five settings, so no
            // row's answer rests on what the row before it left behind.
            let steps = program.steps(
                &pipes
                    .iter()
                    .enumerate()
                    .map(|(at, _)| Handle(at as u32))
                    .collect::<Vec<_>>(),
                &program
                    .states
                    .iter()
                    .enumerate()
                    .map(|(at, _)| Handle(at as u32))
                    .collect::<Vec<_>>(),
                Handle(0),
            );
            let encoded: usize = program.rows.iter().map(row_steps).sum();
            assert_eq!(steps.len(), encoded, "{name}");
            // The positions buffer holds exactly the triangles the rows index.
            assert_eq!(
                positions_bytes(&program.triangles).len() as u64,
                program.triangles.len() as u64 * POSITION_STRIDE,
                "{name}"
            );
            // A program with no colour attachment makes the one pipeline that has no fragment
            // stage and no colour state, which is the only shape Metal takes for such a pass.
            assert_eq!(program.color, pipes.contains(&Pipe::Coloring), "{name}");
            assert_eq!(!program.color, pipes.contains(&Pipe::NoFragment), "{name}");
        }
    }

    /// **The two colour pipelines differ in their write mask alone**, which is what keeps a prime
    /// draw's depth or stencil out of the colour target.
    #[test]
    fn the_masked_pipeline_writes_no_channel_and_the_other_writes_every_one() {
        assert_eq!(MASKED[0].unwrap().write_mask, ColorWriteMask::NONE);
        assert_eq!(PLAIN[0].unwrap().write_mask, ColorWriteMask::ALL);
        let program = DepthCompares::program();
        assert_ne!(
            program.pipeline(Pipe::Coloring).words(),
            program.pipeline(Pipe::Masked).words()
        );
        assert_eq!(color_layout().extent(), COLOR_PLANE as usize);
        assert_eq!(stencil_layout().extent(), STENCIL_PLANE as usize);
        assert_eq!(CELLS, 64);
    }

    /// **Every case of the family names itself once**, under the settings the module states.
    #[test]
    fn the_family_names_eight_cases_apart() {
        let names: Vec<&str> = all().iter().map(|case| case.name()).collect();
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(distinct.len(), names.len(), "two cases share a name");
        assert_eq!(names.len(), 8);
        for case in all() {
            assert_eq!(case.record_runs(), RUNS);
            assert!(matches!(case.metal_source(), MetalSource::Run));
            assert!(case.inputs().is_ok());
            // Five of the six loosen nothing at all: a texel is white or the clear and a stencil
            // byte is a stencil byte, both of which the attachments hold exactly. The sixth names
            // one notice and no value tolerance, because the path it is the only case to take is
            // announced.
            let allowed = case.allowances();
            match case.name() == DepthSlot.name() {
                true => assert_eq!(allowed.len(), 1, "{}", case.name()),
                false => assert!(allowed.is_empty(), "{}", case.name()),
            }
        }
        // The one notice the family names is a cost rather than a near-equivalent, which is what
        // makes loosening the notice site alone honest: a `Precision` degradation would owe a value
        // tolerance beside it.
        assert_eq!(
            Degradation::SeparateDepthStencil.category(),
            metal2vulkan_runtime::DegradationCategory::Performance
        );
        // The six programs between them attach every combination the family needs: depth alone,
        // stencil alone, and the two together as separate textures.
        let shapes: BTreeSet<(Option<u32>, bool)> = programs()
            .iter()
            .map(|(_, program)| (program.depth.map(PixelFormat::as_raw), program.stencil))
            .collect();
        assert!(shapes.contains(&(Some(PixelFormat::Depth32Float.as_raw()), false)));
        assert!(shapes.contains(&(Some(PixelFormat::Depth16Unorm.as_raw()), false)));
        assert!(shapes.contains(&(None, true)));
        assert!(shapes.contains(&(Some(PixelFormat::Depth32Float.as_raw()), true)));
    }

    // --------------------------------------------------------------------------------------------
    // What Metal recorded

    /// The one outcome of `case`'s recorded Metal answer, or `None` where nothing recorded it.
    ///
    /// The assertion is the module's own check on determinism: every case here is one pass of one
    /// queue with nothing a device may reorder, so the eight runs are expected to fold into one
    /// outcome, and a case whose comparison answered differently between runs says so here rather
    /// than having one run read as the answer.
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

    /// The bytes of `case`'s `drawn/<what>` site.
    fn recorded(case: &str, what: &str, length: usize) -> Option<Vec<u8>> {
        let transcript = only_outcome(case)?;
        let node = transcript
            .root
            .child("drawn")
            .and_then(|node| node.child(what))
            .unwrap_or_else(|| panic!("{case}: no drawn/{what} site"));
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: drawn/{what} holds no bytes");
        };
        assert_eq!(bytes.len(), length);
        Some(bytes.clone())
    }

    /// The `float4` at texel `(x, y)` of a colour read-back.
    fn texel(bytes: &[u8], x: u32, y: u32) -> [f32; 4] {
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

    /// Whether every texel of cell `at` holds the fragment function's white, or every one holds the
    /// clear, and which.
    fn cell_drawn(bytes: &[u8], at: usize) -> bool {
        let rect = cell(at);
        let mut colored = 0usize;
        let mut cleared = 0usize;
        let mut other = Vec::new();
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let got = texel(bytes, x as u32, y as u32);
                if got == WHITE {
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
            "cell {at} holds {} texel(s) that are neither white nor the clear: {}",
            other.len(),
            other.join(", ")
        );
        let whole = (rect.width * rect.height) as usize;
        assert!(
            colored == whole || cleared == whole,
            "cell {at} is {colored} of {whole} drawn, which is neither drawn nor clear"
        );
        colored == whole
    }

    /// The one stencil byte every texel of cell `at` holds.
    fn cell_stencil(bytes: &[u8], at: usize) -> u32 {
        let rect = cell(at);
        let mut seen = BTreeSet::new();
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                seen.insert(bytes[(y * u64::from(SIDE) + x) as usize]);
            }
        }
        assert_eq!(
            seen.len(),
            1,
            "cell {at} holds {} different stencil bytes: {seen:?}",
            seen.len()
        );
        u32::from(*seen.iter().next().expect("a cell of no texels"))
    }

    /// Every colour texel outside the cells of `used` holds the clear.
    fn color_cleared_outside(bytes: &[u8], used: &[usize]) {
        let inside: BTreeSet<usize> = used.iter().copied().collect();
        for at in 0..CELLS {
            if inside.contains(&at) {
                continue;
            }
            assert!(
                !cell_drawn(bytes, at),
                "cell {at} is outside every row of the case and is not the clear"
            );
        }
    }

    /// **Metal's recorded target holds what each depth comparison and each `depthWriteEnabled`
    /// says it should.**
    #[test]
    fn metal_compared_each_depth_as_its_function_says() {
        let Some(bytes) = recorded(DepthCompares.name(), "texels", COLOR_PLANE as usize) else {
            println!("depthstencil-depth-compares: no recording in the tree yet");
            return;
        };
        for (input, depth) in DEPTH_INPUTS.iter().enumerate() {
            for (at, function) in COMPARES.iter().enumerate() {
                let want = holds(
                    *function,
                    depth
                        .partial_cmp(&STORED_DEPTH)
                        .expect("a depth that is NaN"),
                );
                assert_eq!(
                    cell_drawn(&bytes, input * GRID as usize + at),
                    want,
                    "a fragment at {depth} against a stored {STORED_DEPTH} under {} did not \
                     answer as the function says",
                    function.name()
                );
            }
        }
        // The write enabled: the first draw's depth reached the attachment, so the probe failed.
        assert!(
            !cell_drawn(&bytes, WRITE_ON),
            "a passing draw under `depthWriteEnabled` did not leave its depth behind"
        );
        // The write disabled: the stored depth is untouched, so the probe passed. This cell and the
        // one before it differ in that field alone.
        assert!(
            cell_drawn(&bytes, WRITE_OFF),
            "a draw without `depthWriteEnabled` left its depth behind"
        );
        // The write enabled under a comparison that fails: no fragment survives, so nothing is
        // written and the probe passes.
        assert!(
            cell_drawn(&bytes, WRITE_FAILED),
            "a draw that failed the depth comparison still wrote its depth"
        );
        let used: Vec<usize> = (0..DEPTH_INPUTS.len() * COMPARES.len())
            .chain([WRITE_ON, WRITE_OFF, WRITE_FAILED])
            .collect();
        color_cleared_outside(&bytes, &used);
    }

    /// **Metal's recorded target says which domain a `Depth16Unorm` comparison happens in, and the
    /// whole block is consistent with that one reading.**
    ///
    /// The domain is read from the `Equal` cell of the quantising input rather than assumed: a
    /// fragment a fraction of a code above the stored code is equal to it as a code and above it as
    /// a float, so that one cell names the rule and the other thirty-one are held to it. Either
    /// reading is a consistent rule; a mixture of the two is the failure.
    #[test]
    fn metal_compared_each_depth_code_in_one_domain_throughout() {
        let Some(bytes) = recorded(DepthCodes.name(), "texels", COLOR_PLANE as usize) else {
            println!("depthstencil-depth-codes: no recording in the tree yet");
            return;
        };
        let equal = COMPARES
            .iter()
            .position(|f| *f == CompareFunction::Equal)
            .expect("no Equal in the swept list");
        let quantised = cell_drawn(&bytes, QUANTISED_INPUT * GRID as usize + equal);
        // What each of the four inputs is, as an ordering against the stored code, under the domain
        // the recording named.
        let orderings = [
            Ordering::Less,
            Ordering::Equal,
            match quantised {
                true => Ordering::Equal,
                false => Ordering::Greater,
            },
            Ordering::Greater,
        ];
        for (input, ordering) in orderings.iter().enumerate() {
            for (at, function) in COMPARES.iter().enumerate() {
                assert_eq!(
                    cell_drawn(&bytes, input * GRID as usize + at),
                    holds(*function, *ordering),
                    "input {input} under {} is not what a comparison {} the format's \
                     quantisation says",
                    function.name(),
                    match quantised {
                        true => "after",
                        false => "before",
                    }
                );
            }
        }
        // The two ends of the range, which no quantisation question touches: at each end the
        // fragment is exactly the stored value, and one code inward it is not.
        for (block, inward) in [(4usize, Ordering::Greater), (6usize, Ordering::Less)] {
            for (input, ordering) in [Ordering::Equal, inward].iter().enumerate() {
                for (at, function) in COMPARES.iter().enumerate() {
                    assert_eq!(
                        cell_drawn(&bytes, (block + input) * GRID as usize + at),
                        holds(*function, *ordering),
                        "the range end's input {input} under {} is not what the comparison says",
                        function.name()
                    );
                }
            }
        }
        // A fragment at exactly 0 and exactly 1 survived clipping, which is what makes the two end
        // blocks a reading of the comparison rather than of the clip volume.
        let always = COMPARES
            .iter()
            .position(|f| *f == CompareFunction::Always)
            .expect("no Always in the swept list");
        for block in [4usize, 6] {
            assert!(
                cell_drawn(&bytes, block * GRID as usize + always),
                "the fragment at the depth range's end was clipped, so its block compares nothing"
            );
        }
        color_cleared_outside(&bytes, &(0..CELLS).collect::<Vec<_>>());
    }

    /// **Metal's recorded target holds what each stencil comparison and each `readMask` says it
    /// should.**
    #[test]
    fn metal_compared_each_stencil_as_its_function_and_mask_say() {
        let Some(bytes) = recorded(StencilCompares.name(), "texels", COLOR_PLANE as usize) else {
            println!("depthstencil-stencil-compares: no recording in the tree yet");
            return;
        };
        for (input, reference) in STENCIL_INPUTS.iter().enumerate() {
            for (at, function) in COMPARES.iter().enumerate() {
                assert_eq!(
                    cell_drawn(&bytes, input * GRID as usize + at),
                    holds(*function, reference.cmp(&STORED_STENCIL)),
                    "a reference of {reference:#04x} against a stored {STORED_STENCIL:#04x} \
                     under {} did not answer as the function says",
                    function.name()
                );
            }
        }
        for (at, (mask, reference, function, stated)) in READ_MASKS.iter().enumerate() {
            assert_eq!(
                cell_drawn(&bytes, MASK_CELLS + at),
                *stated,
                "readMask {mask:#04x} with reference {reference:#04x} under {} did not answer as \
                 the masked comparison says",
                function.name()
            );
        }
        let used: Vec<usize> = (0..STENCIL_INPUTS.len() * COMPARES.len())
            .chain(MASK_CELLS..MASK_CELLS + READ_MASKS.len())
            .collect();
        color_cleared_outside(&bytes, &used);
    }

    /// **Metal's recorded stencil holds what each operation leaves in each slot, at each stored
    /// value, and what each `writeMask` lets through.**
    #[test]
    fn metal_applied_each_stencil_operation_as_its_rule_says() {
        let Some(bytes) = recorded(StencilOperations.name(), "stencil", STENCIL_PLANE as usize)
        else {
            println!("depthstencil-stencil-operations: no recording in the tree yet");
            return;
        };
        // The pass slot and the failure slot answer by the same rule: the slot decides which
        // operation fires, not what the operation does.
        for (slot, base) in [
            ("depthStencilPassOperation", 0usize),
            ("stencilFailureOperation", STARTS.len()),
        ] {
            for (start, held) in STARTS.iter().enumerate() {
                for (at, operation) in OPERATIONS.iter().enumerate() {
                    assert_eq!(
                        cell_stencil(&bytes, (base + start) * GRID as usize + at),
                        applied(*operation, *held, TEST_REFERENCE),
                        "{} in {slot} at a stored {held} did not leave what its rule says",
                        operation.name()
                    );
                }
            }
        }
        for (at, (operation, mask)) in WRITE_MASKS.iter().enumerate() {
            assert_eq!(
                cell_stencil(&bytes, WRITE_MASK_ROW as usize * GRID as usize + at),
                written(
                    MASKED_START,
                    applied(*operation, MASKED_START, TEST_REFERENCE),
                    *mask
                ),
                "{} under writeMask {mask:#04x} did not leave what the masked write says",
                operation.name()
            );
        }
        // The last cell row holds no row of the case, so it is the pass's own clear: what says no
        // draw escaped its rectangle and no prime band reached past its own row.
        for at in (GRID as usize - 1) * GRID as usize..CELLS {
            assert_eq!(
                cell_stencil(&bytes, at),
                0,
                "cell {at} holds no row of the case and is not the clear"
            );
        }
    }

    /// Which stencil each of the facing case's ten cells holds, given `first_is_front`: whether the
    /// forward-wound triangle is the front-facing one under the winding the pass leaves at Metal's
    /// default.
    ///
    /// The convention is read from a cell rather than assumed, because what the case asks is what
    /// the two **faces** do, not which way round Metal calls a triangle.
    fn faces_expected(first_is_front: bool) -> [u32; 10] {
        let up = FACE_START + 1;
        let down = FACE_START - 1;
        // The face a winding takes: the forward triangle takes the front face exactly where the
        // convention says it is front-facing.
        let forward_up = first_is_front;
        [
            // State 0: the front face steps up and the back face steps down.
            if forward_up { up } else { down },
            if forward_up { down } else { up },
            // State 1, the two swapped: each winding's answer swaps with them.
            if forward_up { down } else { up },
            if forward_up { up } else { down },
            // State 2, the one-sided spelling: both windings step up.
            up,
            up,
            // The two-sided reference call with equal values, which must answer as cell 0 did.
            if forward_up { up } else { down },
            // The two references apart, under the reversed winding. Where that winding is the back
            // face it is given a reference nothing equals, so its comparison fails and its failure
            // operation (`Zero`) fires; where it is the front face it keeps the front reference and
            // steps up.
            if forward_up { 0 } else { up },
            // The per-face read masks under a reference one bit from the stored value: the
            // whole-byte face fails and zeroes, the masked face passes and steps down.
            if forward_up { 0 } else { down },
            if forward_up { down } else { 0 },
        ]
    }

    /// **The facing rule holds for either convention**, which is what lets the case state its map
    /// without assuming one, and the two conventions give different maps, which is what makes the
    /// reading a reading.
    #[test]
    fn the_facing_rule_holds_for_either_convention() {
        let one = faces_expected(true);
        let other = faces_expected(false);
        assert_ne!(
            one, other,
            "the facing convention changes nothing the case reads"
        );
        for held in [one, other] {
            // The two windings disagree under a two-sided state and agree under a one-sided one.
            assert_ne!(held[0], held[1]);
            assert_ne!(held[2], held[3]);
            assert_eq!(held[4], held[5]);
            // Swapping the faces swaps each winding's answer.
            assert_eq!(held[0], held[3]);
            assert_eq!(held[1], held[2]);
            // The one-sided and the two-sided spellings of the same reference answer alike.
            assert_eq!(held[0], held[6]);
            // The per-face masks give the two windings different answers again.
            assert_ne!(held[8], held[9]);
        }
        // The two values the pass operations leave are inside a `Stencil8` and are not the clear.
        for value in [FACE_START + 1, FACE_START - 1] {
            assert_eq!(value & 0xff, value);
            assert_ne!(value, FACE_START);
        }
        assert_ne!(FACE_NEAR, FACE_START);
        assert_eq!((FACE_NEAR ^ FACE_START).count_ones(), 1);
    }

    /// **Metal's recorded stencil holds what the two faces, read apart, say it should.**
    #[test]
    fn metal_took_each_winding_through_its_own_face() {
        let Some(bytes) = recorded(Faces.name(), "stencil", STENCIL_PLANE as usize) else {
            println!("depthstencil-faces: no recording in the tree yet");
            return;
        };
        let first = cell_stencil(&bytes, 0);
        assert!(
            first == FACE_START + 1 || first == FACE_START - 1,
            "the first cell holds {first:#04x}, which is neither face's pass operation's answer"
        );
        let first_is_front = first == FACE_START + 1;
        let want = faces_expected(first_is_front);
        let got: Vec<u32> = (0..want.len()).map(|at| cell_stencil(&bytes, at)).collect();
        assert_eq!(
            got.as_slice(),
            want.as_slice(),
            "the stencils Metal left are not what the two faces imply (the forward-wound triangle \
             is {} under the pass's own winding)",
            match first_is_front {
                true => "front-facing",
                false => "back-facing",
            }
        );
        for at in want.len()..CELLS {
            assert_eq!(
                cell_stencil(&bytes, at),
                FACE_START,
                "cell {at} holds no row of the case and is not the clear"
            );
        }
    }

    /// **Metal's recorded stencil and colour hold what the depth-failure slot, the four outcomes of
    /// the two tests, and the depth write under a failed stencil test say they should.**
    #[test]
    fn metal_reached_the_depth_failure_slot_and_only_it() {
        let Some(stencil) = recorded(DepthSlot.name(), "stencil", STENCIL_PLANE as usize) else {
            println!("depthstencil-depth-slot: no recording in the tree yet");
            return;
        };
        let color = recorded(DepthSlot.name(), "texels", COLOR_PLANE as usize)
            .expect("a recording with a stencil but no colour");
        // The slot: every operation at every stored value, under a depth comparison that fails and
        // a stencil comparison that passes.
        for (start, held) in STARTS.iter().enumerate() {
            for (at, operation) in OPERATIONS.iter().enumerate() {
                assert_eq!(
                    cell_stencil(&stencil, start * GRID as usize + at),
                    applied(*operation, *held, TEST_REFERENCE),
                    "{} in depthFailureOperation at a stored {held} did not leave what its rule \
                     says",
                    operation.name()
                );
            }
        }
        // The four outcomes, by the slot each one fires: the pass operation, the depth-failure
        // operation, and the stencil-failure operation twice, since a failed stencil test is never
        // followed by a depth test.
        let base = MATRIX_ROW as usize * GRID as usize;
        let pass = applied(StencilOperation::IncrementClamp, SLOT_START, TEST_REFERENCE);
        let depth_failed = applied(StencilOperation::Invert, SLOT_START, TEST_REFERENCE);
        let stencil_failed = applied(StencilOperation::Zero, SLOT_START, TEST_REFERENCE);
        for (at, (want, drawn, why)) in [
            (pass, true, "both tests passing fires the pass operation"),
            (
                depth_failed,
                false,
                "the depth test failing fires the depth-failure operation",
            ),
            (
                stencil_failed,
                false,
                "the stencil test failing fires the stencil-failure operation",
            ),
            (
                stencil_failed,
                false,
                "both failing fires the stencil-failure operation alone, so a failed stencil test \
                 is never followed by a depth test",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(cell_stencil(&stencil, base + at), want, "{why}");
            assert_eq!(cell_drawn(&color, base + at), drawn, "{why}");
        }
        // The three slots' answers are three different values, so the cell above names which slot
        // fired rather than agreeing by accident.
        assert_eq!(
            BTreeSet::from([pass, depth_failed, stencil_failed, SLOT_START]).len(),
            4
        );
        // The depth write under a failed stencil test: Metal discards the fragment, so the depth is
        // untouched and the probe passes. The cell beside it, whose stencil test passes, has the
        // write and the probe fails.
        assert!(
            cell_drawn(&color, base + 4),
            "a fragment that failed the stencil test still wrote its depth"
        );
        assert!(
            !cell_drawn(&color, base + 5),
            "a fragment that passed both tests under `depthWriteEnabled` did not write its depth"
        );
        // Those two cells' stencil is untouched: both states and the probe write no bit.
        for at in [base + 4, base + 5] {
            assert_eq!(cell_stencil(&stencil, at), SLOT_START);
        }
        // Everything else: the colour is the clear outside the one matrix cell and the one probe
        // cell that drew, and the stencil past the primed rows is the pass's own clear.
        color_cleared_outside(&color, &[base, base + 4]);
        for at in (MATRIX_ROW as usize + 1) * GRID as usize..CELLS {
            assert_eq!(
                cell_stencil(&stencil, at),
                0,
                "cell {at} holds no row of the case and is not the clear"
            );
        }
    }

    /// What `operation` leaves under `mask` when the mask narrows the operation's **operand**
    /// rather than its result: the crossing of the two that no single-value sweep of either can
    /// see.
    ///
    /// This is not Metal's rule and is not asserted to be. It is the wrong rule the crossing rows
    /// exist to separate from the right one, written here so the separation is a computation rather
    /// than a claim.
    fn operand_masked(operation: StencilOperation, start: u32, reference: u32, mask: u32) -> u32 {
        let narrowed = start & mask & 0xff;
        written(start, applied(operation, narrowed, reference), mask)
    }

    /// **Every crossing row's answer is what Metal's rule leaves**, derived here from the same
    /// three steps Metal takes — the comparison under the read mask, the operation of the slot that
    /// fires, and the write under the write mask — rather than copied from a recording.
    #[test]
    fn the_crossing_rows_are_what_metals_three_steps_say() {
        for (at, row) in CROSSINGS.iter().enumerate() {
            let ordering =
                (row.reference & row.read_mask & 0xff).cmp(&(row.start & row.read_mask & 0xff));
            let operation = match holds(row.compare, ordering) {
                true => row.pass,
                false => row.failure,
            };
            assert_eq!(
                written(
                    row.start,
                    applied(operation, row.start, row.reference),
                    row.write_mask
                ),
                row.left,
                "crossing {at} states {:#04x}, which is not what the rule leaves",
                row.left
            );
            // Every mask is inside a `Stencil8`'s eight bits, and so is every value, so no row's
            // answer turns on a bit the attachment does not hold.
            for value in [row.start, row.read_mask, row.write_mask, row.left] {
                assert_eq!(
                    value & 0xff,
                    value,
                    "crossing {at} names a value past 8 bits"
                );
            }
        }
        // Exactly one row leaves the stencil it started at: the `writeMask` of 0 control, which is
        // read against the row above it. Every other row's cell is distinguishable from a draw that
        // did nothing, so a cell that holds its start is a failure rather than a pass.
        let unchanged: Vec<usize> = CROSSINGS
            .iter()
            .enumerate()
            .filter(|(_, row)| row.left == row.start)
            .map(|(at, _)| at)
            .collect();
        assert_eq!(unchanged, vec![9], "{unchanged:?} leave the stencil alone");
    }

    /// **The crossing rows see a write mask applied to the operation's operand, and
    /// `WRITE_MASKS` cannot.**
    ///
    /// The existing write-mask sweep carries `Replace`, `Invert`, `Zero` and one `IncrementWrap`
    /// row. The first three are bitwise functions of the stored value, so narrowing the operand and
    /// narrowing the result give the same byte for every mask; and the `IncrementWrap` row is at
    /// `0xa5` under `0xf0`, where the carry does not cross the mask's edge, so it gives the same
    /// byte too. The whole table is therefore **invariant** under the wrong rule, which this
    /// asserts rather than describes — and five rows of `CROSSINGS` are not.
    #[test]
    fn the_crossings_see_an_operand_masked_operation_and_the_old_table_cannot() {
        for (at, (operation, mask)) in WRITE_MASKS.iter().enumerate() {
            assert_eq!(
                written(
                    MASKED_START,
                    applied(*operation, MASKED_START, TEST_REFERENCE),
                    *mask
                ),
                operand_masked(*operation, MASKED_START, TEST_REFERENCE, *mask),
                "WRITE_MASKS row {at} does separate the two rules, so the module's claim that it \
                 cannot is now wrong"
            );
        }
        let mut seen = 0;
        for (at, row) in CROSSINGS.iter().enumerate() {
            let ordering =
                (row.reference & row.read_mask & 0xff).cmp(&(row.start & row.read_mask & 0xff));
            let operation = match holds(row.compare, ordering) {
                true => row.pass,
                false => row.failure,
            };
            let wrong = operand_masked(operation, row.start, row.reference, row.write_mask);
            assert_eq!(
                wrong != row.left,
                row.operand_visible,
                "crossing {at} ({}) says operand_visible = {}, and the two rules leave \
                 {:#04x} and {wrong:#04x}",
                operation.name(),
                row.operand_visible,
                row.left
            );
            seen += u32::from(row.operand_visible);
        }
        assert_eq!(seen, 5, "the rows that separate the two rules are not five");
    }

    /// **Each of the three ways the two masks can be crossed moves a row of `CROSSINGS`**, which is
    /// what makes the pair's rows the pair's rather than two single-value rows side by side.
    ///
    /// The three wrong rules are: the read mask used where the write mask belongs, the write mask
    /// used where the read mask belongs, and the two swapped. Each is applied to the first two rows
    /// — the only two whose masks are both away from `0xff` — and each gives a different byte from
    /// Metal's.
    #[test]
    fn each_mask_crossing_moves_the_rows_that_state_both_masks() {
        let answer = |row: &Crossing, read: u32, write: u32| {
            let ordering = (row.reference & read & 0xff).cmp(&(row.start & read & 0xff));
            let operation = match holds(row.compare, ordering) {
                true => row.pass,
                false => row.failure,
            };
            written(
                row.start,
                applied(operation, row.start, row.reference),
                write,
            )
        };
        let both_narrow: Vec<&Crossing> = CROSSINGS
            .iter()
            .filter(|row| row.read_mask != ALL_BITS && row.write_mask != ALL_BITS)
            .collect();
        assert_eq!(both_narrow.len(), 2, "the crossed rows are not two");
        for row in both_narrow {
            for (what, read, write) in [
                (
                    "the read mask where the write mask belongs",
                    row.read_mask,
                    row.read_mask,
                ),
                (
                    "the write mask where the read mask belongs",
                    row.write_mask,
                    row.write_mask,
                ),
                ("the two swapped", row.write_mask, row.read_mask),
            ] {
                assert_ne!(
                    answer(row, read, write),
                    row.left,
                    "a side that used {what} answers this row exactly as Metal does"
                );
            }
        }
    }

    /// Which stencil each of the facing case's twelve cells holds, given `forward_is_clockwise`:
    /// whether the forward-wound triangle of [`CORNERS`] is the clockwise one as Metal's rasterizer
    /// reads it.
    ///
    /// The convention is read from a cell rather than assumed, as [`faces_expected`] reads its own:
    /// what the case asks is whether the two settings together pick the face, not which way round
    /// Metal calls a triangle.
    fn facing_expected(forward_is_clockwise: bool) -> Vec<u32> {
        let up = FACE_START + 1;
        let down = FACE_START - 1;
        facing_cells()
            .into_iter()
            .map(|(cull, winding, triangle)| {
                // The drawn triangle's own winding, and then whether that is the face the encoder
                // calls front.
                let clockwise = match triangle {
                    0 => forward_is_clockwise,
                    _ => !forward_is_clockwise,
                };
                let front = clockwise == (winding == Winding::Clockwise);
                let culled = match cull {
                    CullMode::Front => front,
                    CullMode::Back => !front,
                    _ => false,
                };
                match (culled, front) {
                    (true, _) => FACE_START,
                    (false, true) => up,
                    (false, false) => down,
                }
            })
            .collect()
    }

    /// **The facing rule holds for either convention, and the two conventions give different
    /// maps**, which is what lets the case state its expectation without assuming one and what
    /// makes the reading a reading rather than a definition.
    #[test]
    fn the_facing_cross_separates_the_cull_mode_from_the_face_choice() {
        let one = facing_expected(true);
        let other = facing_expected(false);
        assert_eq!(one.len(), 12);
        assert_ne!(one, other, "the convention changes nothing the case reads");
        let cells = facing_cells();
        // Twelve distinct settings, so no cell is another's answer under a different name.
        let distinct: BTreeSet<(u32, u32, usize)> = cells
            .iter()
            .map(|(cull, winding, triangle)| (cull.as_raw(), winding.as_raw(), *triangle))
            .collect();
        assert_eq!(distinct.len(), 12);
        for held in [&one, &other] {
            // Under `CullMode::None` nothing is culled, so each of the four cells holds a face's
            // own answer and the two windings of one setting disagree.
            for (at, value) in held.iter().enumerate().take(4) {
                assert_ne!(*value, FACE_START, "cell {at} is culled under no culling");
            }
            assert_ne!(held[0], held[1]);
            assert_ne!(held[2], held[3]);
            // Changing `setFrontFacingWinding:` swaps which winding is the front face, so the
            // first pair is the second pair reversed.
            assert_eq!(held[0], held[3]);
            assert_eq!(held[1], held[2]);
            // Under each one-sided cull mode exactly one winding of each setting survives, and the
            // one that survives is the other face from the one the other cull mode keeps.
            for base in [4, 8] {
                let survived: Vec<usize> = (base..base + 4)
                    .filter(|at| held[*at] != FACE_START)
                    .collect();
                assert_eq!(survived.len(), 2, "cells {base}..{} ", base + 4);
            }
            // `CullMode::Front` keeps the back face's answer and `CullMode::Back` the front's,
            // which is what says the cull mode and the face choice are read from the same setting
            // and are not the same thing.
            for value in held.iter().take(8).skip(4) {
                assert!(*value == FACE_START || *value == FACE_START - 1);
            }
            for value in held.iter().take(12).skip(8) {
                assert!(*value == FACE_START || *value == FACE_START + 1);
            }
        }
        // The three values a cell can hold are distinct, so a cell's answer is one reading.
        let values = BTreeSet::from([FACE_START, FACE_START + 1, FACE_START - 1]);
        assert_eq!(values.len(), 3);
    }

    /// **Metal's recorded stencil holds what the two faces, the winding and the cull mode together
    /// say it should.**
    #[test]
    fn metal_picked_the_face_from_the_winding_under_every_cull_mode() {
        let Some(bytes) = recorded(FacesFacing.name(), "stencil", STENCIL_PLANE as usize) else {
            println!("depthstencil-facing-faces: no recording in the tree yet");
            return;
        };
        let first = cell_stencil(&bytes, 0);
        assert!(
            first == FACE_START + 1 || first == FACE_START - 1,
            "the first cell holds {first:#04x}, which is neither face's pass operation's answer \
             and is not what a cell under no culling can hold"
        );
        // Cell 0 draws the forward triangle under `frontFacingWinding = Clockwise` and no culling,
        // so it steps up exactly where the forward triangle is the clockwise one.
        let forward_is_clockwise = first == FACE_START + 1;
        let want = facing_expected(forward_is_clockwise);
        let got: Vec<u32> = (0..want.len()).map(|at| cell_stencil(&bytes, at)).collect();
        assert_eq!(
            got,
            want,
            "the stencils Metal left are not what the facing cross implies (the forward-wound \
             triangle is {} as the rasterizer reads it)",
            match forward_is_clockwise {
                true => "clockwise",
                false => "counter-clockwise",
            }
        );
        for at in want.len()..CELLS {
            assert_eq!(
                cell_stencil(&bytes, at),
                FACE_START,
                "cell {at} holds no row of the case and is not the clear"
            );
        }
    }

    /// **Metal's recorded stencil holds what the two masks, crossed with each other and with the
    /// operation, say it should.**
    #[test]
    fn metal_crossed_the_two_masks_as_its_rule_says() {
        let Some(bytes) = recorded(MaskCrossings.name(), "stencil", STENCIL_PLANE as usize) else {
            println!("depthstencil-mask-crossings: no recording in the tree yet");
            return;
        };
        for (at, row) in CROSSINGS.iter().enumerate() {
            assert_eq!(
                cell_stencil(&bytes, at),
                row.left,
                "crossing {at} (read {:#04x}, write {:#04x}, {} on a pass and {} on a failure) \
                 did not leave what the two masks imply",
                row.read_mask,
                row.write_mask,
                row.pass.name(),
                row.failure.name()
            );
        }
        for at in CROSSINGS.len()..CELLS {
            assert_eq!(
                cell_stencil(&bytes, at),
                0,
                "cell {at} holds no row of the case and is not the clear"
            );
        }
    }
}
