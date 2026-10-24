//! Tessellation's option space: what the tessellator generates from a patch's factors, swept over
//! every value of the five enumerations a patch draw can set, over the factor values Metal's own
//! rules single out, and over `setTessellationFactorScale:` crossed with the factor it scales.
//!
//! No case of the suite drew a patch before this family, so `MTLTessellationPartitionMode`,
//! `MTLTessellationFactorStepFunction`, `MTLTessellationControlPointIndexType`, `MTLPatchType` and
//! two values of `MTLVertexStepFunction` were set by a descriptor-fact case and read back by its
//! getter, with nothing downstream of them ever running. This family runs them.
//!
//! `setTessellationFactorScale:` and `tessellationFactorScaleEnabled` were the same: the call
//! reached no case at all, and the field was set and read back with nothing reading it. Metal's
//! documentation describes the scale's special values — 0, a negative one, a NaN, an infinity — as
//! culling, and nothing here had measured any of them. [`FactorScales`] measures them, and the
//! answer is not the documentation's (below).
//!
//! # The observable
//!
//! Four of the six cases are built on the two committed domain fixtures,
//! `k7_tess_quad_domain` and `k7_tess_triangle_domain`, which were written for the validation
//! lane and are widened here rather than copied. Each is a `vertex void` post-tessellation vertex
//! function with rasterization disabled, and for every domain point the tessellator calls it with
//! it stores a mark at that point's cell of a 65-by-65 grid, `marks[j · 65 + i]` for
//! `i = rint(u · grid.x)` and `j = rint(v · grid.y)`, the two scales coming from the patch's own
//! per-patch attribute. **A mark is idempotent and the marks are a set**, so the answer is the
//! tessellator's point set whatever the invocation count and whatever order the GPU ran them in.
//! That is what makes a tessellation case comparable at all: the invocation count is not Metal's
//! contract and the order is nobody's.
//!
//! The fixtures also store each point's coordinates, bit for bit, past the marks. **This family
//! does not compare them.** Two domain points that round to one cell store their coordinates at
//! the same word, and which of them wrote it last is the GPU's business, so a case that compared
//! those words would be comparing an invocation order. The cost is a resolution bound and the
//! family states it: a 65-cell axis resolves a coordinate to `1/64`, so a position difference
//! below `1/128` would not show in a mark. A sharper comparison needs a fixture that stores a
//! point's coordinates at a word its own coordinates name, and no committed one does. It would buy
//! the factor, scale and non-finite sweeps nothing: each of their answers is a **point count** at a
//! whole level — a cull, a range end, or a level of 1, 2, 4, 8, 9 or 16 — and two such levels never
//! place two points inside one cell of the grid the sweep states. The bound is open for the
//! partition modes, whose fractional spacings differ in where they place a point and not in how
//! many they place.
//!
//! The other two cases are built on `rt6_patch_index`, whose patch stores one word per patch
//! holding its per-patch attribute, its patch index and its three control points' tags. Four words
//! are the whole answer, which is what makes the control-point indexing and the two tessellation
//! vertex steps cheap to ask about.
//!
//! # The degrade notices every case here names
//!
//! A tessellation pipeline of ours announces `tessellation-winding` and
//! `tessellation-factor-edge-cases` unconditionally, `tessellation-factor-instance` where the
//! factor step is a per-instance one, and `tessellation-factor-scale` where the descriptor enables
//! the scale: the control stage's own module docs say those are readings
//! taken from Metal's documentation **without a Metal observation**. Every case here therefore
//! names them. It does not loosen one field with them: a notice declaration unchecks the notices
//! site alone, and every observed value still compares exactly. What the family adds is that those
//! three readings now have a measurement behind them — the answers these cases recorded are
//! Metal's own, on the venue they were recorded on.
//!
//! `tessellation-triangle-order` and `emulated-tessellation-draws` are named too, for the venues
//! that announce them: a device whose tessellator is not Apple's orders a patch's triangles its own
//! way, and a device without `tessellationShader` draws the emulation's. Neither can move a mark,
//! because a mark is a set member and not a position in a triangle stream; they are named because
//! a notice our run announces that no case names is a divergence. Neither fired on MoltenVK over an
//! Apple M3 Max, which is what says that venue draws a patch through Metal's own tessellator.
//!
//! # What the recorded answers settle
//!
//! The numbers below are Metal's, from the committed recordings, on an Apple M3 Max under macOS
//! 27.0.1. They are here because each is what makes a swept value non-vacuous, and three of them
//! answer a question the product had to guess at.
//!
//! - **The four partition modes answer four different point sets, on both domains.** Marked cells
//!   over the swept factors: 33, 24, 28 and 29 for `pow2`, `integer`, `fractionalOdd` and
//!   `fractionalEven` on the quad, and 23, 15, 16 and 21 on the triangle. No two collapse
//!   (`tests::no_two_partition_modes_answered_the_same_marks`), which is only true because the
//!   swept factors are non-integral: at an integral factor the four agree by construction and the
//!   sweep would compare one answer four times.
//! - **The factor clamp is two collapses and three answers.** 0 marks nothing (the patch is
//!   discarded), 0.5 and 1 both mark the four corners, and 64 and 65 both mark all 4225 cells. The
//!   two collapses are Metal's own clamp — up to 1 below the floor, down to the maximum above it —
//!   and `tests::the_factor_clamp_is_metals_own_two_collapses` holds them, so a side that stopped
//!   clamping would part a pair this family states must stay equal.
//! - **Metal's factor address counts the base instance.** The `base-instance-1` leg of
//!   `MTLTessellationFactorStepFunctionPerInstance` marks 25 cells, which is factor-4's record and
//!   not factor-1's, so a draw of one instance from base instance 1 reads the record instance 1
//!   reads. That is exactly the reading the control stage's module docs list as taken from Metal's
//!   documentation **without** a Metal observation — "whether the per-instance step's instance
//!   index includes the base instance (taken to, as `[[instance_id]]` does)" — and the product's
//!   assumption is what Metal does.
//! - **A typed control-point index is implemented and exact.** Both `UInt16` and `UInt32` read the
//!   case's reversing permutation and answer the same four words, which is the answer the untyped
//!   draw does **not** give. `pipeline::descriptor`'s `TessellationDescriptor` doc comment says a
//!   control-point index type other than `None` "is not implemented"; it is, and the comment is
//!   stale.
//! - **Metal applies the factor scale before the clamp, not after.** A factor of 0.5 scaled by 2
//!   marks the four corners, which is a level of 1 — the product clamped up to the floor. Clamped
//!   first it would be `max(0.5, 1) = 1` scaled to 2, which marks nine cells. And 17 halved under a
//!   ceiling of 16 marks 100 cells, which is a level of 9 (`8.5` rounded up), not the 81 cells a
//!   level of 8 marks. Halving 2, which is 1 either way round, marks four cells in both: that is
//!   the control beside the two, and it is what says they moved for the order.
//! - **Metal rounds the scale to `half` before it multiplies.** 2.0005 marks exactly what a scale
//!   of 2 marks, 81 cells. A scale of `2^-15`, which is subnormal as a `half`, does **not** cull:
//!   its product with 4 clamps up to the floor and marks the four corners. A scale of `2^-30`,
//!   which is 0 as a `half`, culls. So the rounding is where the scale stops being a number.
//! - **Of the scales Metal's documentation calls culling, one does not cull.** 0, a negative one, a
//!   NaN and `-∞` each mark nothing, and so does a draw of a pipeline that enables the scale and
//!   never sets one. **`+∞` marks all 289 cells**, which is what the maximum factor draws — the
//!   documentation's "infinity" covers only the negative one. Nothing before this measured any of
//!   the four, and the runtime's reading of `+∞` (the scale pushed as `2^32`) is Metal's.
//! - **`tessellationFactorScaleEnabled` gates the whole scale, setter and cull together.** A scale
//!   of 2 on a pipeline that does not enable it marks what factor 4 marks, 25 cells, and so does a
//!   scale of 0 — the same 0 that marks nothing on a pipeline that enables it.
//! - **The edge factors Metal culls are one class, and two near-misses are not in it.** A NaN, a
//!   negative number, a negative zero and `-∞` on the four edges each mark nothing, whatever the
//!   inside factors are. `+∞` on the edges marks 73 cells (a level of 16 with the inside at 4) and
//!   the least positive subnormal marks 13 (a level of 1 with the inside at 4), so neither culls.
//! - **An inside factor is clamped and never culls.** Each of the five values that is not a finite
//!   number above the floor — NaN, negative, negative zero, `-∞` and the least subnormal — marks
//!   the same 17 cells, which is the inner level 1: the boundary ring plus the one centre point.
//!   That collapse is Metal's own clamp, reaching the range's lower end from both directions, and
//!   saying so is the result. `+∞` inside marks 241, the range's upper end, and the control marks
//!   25.
//!
//! # Why no case here can hang the device
//!
//! Every case draws **one** patch per pass, or four patches whose every factor is 1. The largest
//! point set any case asks for is the 65-by-65 grid of a quad at factor 64, which is 4225 domain
//! points and 8192 triangles in one patch: less work than a single full-screen triangle's
//! fragments. Rasterization is disabled in every pass, there is no fragment stage, and no case
//! sets a factor above `maxTessellationFactor` without the pipeline stating that maximum, so
//! nothing here can ask the tessellator for an unbounded stream. The two sweeps that state a
//! factor or a scale Metal's own validation refuses — a NaN, an infinity, a negative number — ask
//! for **less** work and not more: each either culls the patch or reaches the ceiling the pipeline
//! states, and both are bounded by that ceiling of 16.
//!
//! # The device's own bounds
//!
//! `maxTessellationFactor` is a descriptor field whose range Metal documents as `1..=64`, and 64 is
//! the largest value this family states. The patch control-point count is the **function's**
//! (`[[patch(quad, 1)]]`, `[[patch(triangle, 3)]]`) and a draw states the same number, so no case
//! can ask for a count its function does not declare. Nothing here reaches a shape Metal's API
//! validation refuses, so this family adds no refusal-surface row.

use crate::allowance::Allowance;
use crate::case::{call, made, probe, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, TessellationState,
    VertexAttribute, VertexBufferLayout, VertexLayout,
};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    TessellationControlPointIndexType as IndexType, TessellationFactorStepFunction as Step,
    TessellationPartitionMode as Partition, VertexFormat, VertexStepFunction,
};
use metal2vulkan_runtime::Degradation;

// ---- The domain fixtures ----------------------------------------------------------------------

/// The quad domain fixture: `[[patch(quad, 1)]]`, one per-patch `float4` attribute at index 0, the
/// marks buffer at vertex buffer index 1.
const QUAD: Fixture = Fixture::new("k7_tess_quad_domain");

/// The triangle domain fixture, the same shape over `[[patch(triangle, 1)]]`.
const TRIANGLE: Fixture = Fixture::new("k7_tess_triangle_domain");

/// The patch-index fixture: `[[patch(triangle, 3)]]`, a control-point `uint` at attribute 0 and a
/// per-patch `uint` at attribute 1, the marks buffer at vertex buffer index 2.
const PATCHES: Fixture = Fixture::new("rt6_patch_index");

/// Cells on one axis of a domain fixture's grid: `i` and `j` are clamped to `0..=64`.
const AXIS: u32 = 65;

/// Words of a domain fixture's mark region: one per cell of the grid.
const MARK_WORDS: u32 = AXIS * AXIS;

/// Words one domain fixture's whole output takes, marks and stored coordinates together: the quad
/// stores two coordinates per cell and the triangle three, so the triangle's region is the longer
/// one and both are sized by it. The coordinates are written by the shader and never compared (the
/// module docs); the room has to be there or a store would reach the next region.
const REGION_WORDS: u32 = MARK_WORDS + 3 * MARK_WORDS;

/// The byte every mark region starts from, which is a multiple of 4 as a vertex buffer binding's
/// offset must be.
const REGION_BYTES: u32 = REGION_WORDS * 4;

// ---- Factors ----------------------------------------------------------------------------------

/// The `half` bits of `value`, which must be a non-negative finite number `half` holds exactly.
///
/// Every factor this family states is exactly a `half` — each is a small dyadic number — so a value
/// that is not is a mistake in a table rather than a rounding to decide here, and the assertions
/// say so rather than quietly rounding it. `tests::the_half_bits_are_metals_own` holds the table.
fn half_bits(value: f32) -> u16 {
    assert!(
        value.is_finite() && value >= 0.0,
        "a factor is finite and not negative: {value}"
    );
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x007f_ffff;
    assert!(
        (-14..=15).contains(&exponent),
        "a factor is inside half's normal range: {value}"
    );
    assert_eq!(
        mantissa & 0x1fff,
        0,
        "a factor is exactly a half: {value} is not"
    );
    (((exponent + 15) as u16) << 10) | (mantissa >> 13) as u16
}

/// One patch's `MTLQuadTessellationFactorsHalf`: four edge factors and two inside ones, 12 bytes.
fn quad_factors(edges: [f32; 4], inside: [f32; 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12);
    for factor in edges.iter().chain(inside.iter()) {
        bytes.extend(half_bits(*factor).to_le_bytes());
    }
    bytes
}

/// One patch's `MTLTriangleTessellationFactorsHalf`: three edge factors and one inside, 8 bytes.
fn triangle_factors(edges: [f32; 3], inside: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8);
    for factor in edges.iter().chain(std::iter::once(&inside)) {
        bytes.extend(half_bits(*factor).to_le_bytes());
    }
    bytes
}

/// The per-patch `float4` a domain fixture reads as its two grid scales: `(scale, scale, 0, 0)`.
fn grid_record(scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for value in [scale, scale, 0.0, 0.0] {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

// ---- The vocabulary each domain case shares ---------------------------------------------------

/// Which domain fixture a case draws, which is the `MTLPatchType` its function declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Domain {
    Quad,
    Triangle,
}

impl Domain {
    fn fixture(self) -> Fixture {
        match self {
            Domain::Quad => QUAD,
            Domain::Triangle => TRIANGLE,
        }
    }

    /// The `MTLPatchType` enumerator the fixture's `[[patch(...)]]` declares, which is the name the
    /// case's own branch site carries so the depth census credits the value.
    fn patch_type(self) -> &'static str {
        match self {
            Domain::Quad => "MTLPatchTypeQuad",
            Domain::Triangle => "MTLPatchTypeTriangle",
        }
    }

    /// One patch's factor record: the same numbers on every edge and inside, so a sweep over the
    /// partition modes states one factor list and the two domains read as much of it as they have.
    fn factors(self, edges: [f32; 4], inside: [f32; 2]) -> Vec<u8> {
        match self {
            Domain::Quad => quad_factors(edges, inside),
            Domain::Triangle => triangle_factors([edges[0], edges[1], edges[2]], inside[0]),
        }
    }
}

/// The vertex descriptor a domain fixture's one per-patch `float4` attribute is fetched under:
/// attribute 0 at byte 0 of buffer 0, whose layout steps **per patch**.
const DOMAIN_ATTRIBUTES: [VertexAttribute; 1] =
    [VertexAttribute::new(0, VertexFormat::Float4, 0, 0)];

/// Buffer 0's layout: one 16-byte `float4` a patch, under `MTLVertexStepFunctionPerPatch`.
const DOMAIN_LAYOUTS: [VertexBufferLayout; 1] = [VertexBufferLayout {
    index: 0,
    stride: 16,
    step: VertexStepFunction::PerPatch,
    rate: 1,
}];

/// The pipeline one domain site makes: the fixture's `vertex void` function, rasterization
/// disabled, the per-patch attribute fetched under the per-patch step, and `state`'s tessellation
/// fields.
fn domain_pipeline<'a>(entry: &'a str, state: TessellationState) -> RenderPipelineSpec<'a> {
    RenderPipelineSpec::tessellating(
        StageFunction::named(entry),
        VertexLayout::new(&DOMAIN_ATTRIBUTES, &DOMAIN_LAYOUTS),
        state,
    )
}

/// The layout a mark region is compared under: one row of `AXIS` words a grid row, every word
/// compared, so a diff names the cell rather than a byte offset.
fn marks_layout(offset: u64) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset,
            length: u64::from(MARK_WORDS * 4),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: AXIS,
        row_pitch: AXIS * 4,
        rows: AXIS,
        image_pitch: MARK_WORDS * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The notices every tessellation pipeline of ours can announce, which every case here names.
///
/// The first two are unconditional on a tessellation pipeline. The last two are a venue's: a device
/// whose tessellator is not Apple's announces `tessellation-triangle-order`, and one without
/// `tessellationShader` announces `emulated-tessellation-draws`. Neither fires on MoltenVK over an
/// Apple GPU (verified: the replay of `tessellation-patch-indices` on an Apple M3 Max fired two of
/// the declarations and no other), and both are named so a run on a venue that does announce one
/// reads it as declared rather than as an undeclared divergence.
///
/// `tessellation-factor-instance` is **not** here: only a per-instance factor step provokes it, and
/// only [`FactorSteps`] sets one, so naming it everywhere else would be a declaration no case could
/// ever reach.
fn tessellation_notices() -> Vec<Allowance> {
    [
        Degradation::TessellationWinding,
        Degradation::TessellationFactorEdgeCases,
        Degradation::TessellationTriangleOrder,
        Degradation::EmulatedTessellationDraws,
    ]
    .into_iter()
    .map(|notice| crate::replay::notice_allowance(notice.code()))
    .collect()
}

// ---- The partition-mode and factor sweeps over a domain ---------------------------------------

/// Every `MTLTessellationPartitionMode`, in the SDK table's own declaration order.
fn partitions() -> Vec<Partition> {
    Partition::ALL.to_vec()
}

/// The site one partition mode's answer is emitted at: the SDK enumerator's **own name**, because
/// the depth census credits a value only where its enumerator appears by name in a compared
/// transcript.
fn partition_site(mode: Partition) -> &'static str {
    mode.name()
}

/// The edge factors the partition sweep states, and the inside ones beside them.
///
/// **Non-integral, and four different numbers.** The four modes agree on an integral factor by
/// construction: `integer` and both fractional spacings leave it alone, and `pow2` leaves a power
/// of two alone, so a sweep at factors 2 and 4 would compare one answer four times. At 2.5 the four
/// part — `pow2` rounds to 4, `integer` to 3, and the two fractional spacings place their
/// transition segments differently — and the four edges carry four different factors so each mode's
/// rounding is asked four questions in one patch rather than one.
const SWEPT_EDGES: [f32; 4] = [2.5, 3.5, 4.5, 5.5];

/// The inside factors beside [`SWEPT_EDGES`]: two more non-integral numbers, which is where the
/// interior rings' spacing comes from.
const SWEPT_INSIDE: [f32; 2] = [2.5, 3.5];

/// The ceiling the partition sweep states, above every factor it uses so no factor is clamped: the
/// sweep is about the rounding and not about the clamp, which the factor sweep asks about.
const SWEPT_MAX: u32 = 16;

/// The grid scale the partition sweep states: the full 65-cell axis, which resolves a domain
/// coordinate to `1/64`.
const FINE_GRID: f32 = 64.0;

/// A sweep over the four partition modes on one domain.
pub struct PartitionModes {
    name: &'static str,
    domain: Domain,
}

/// A sweep over one bounded factor list on one domain.
pub struct EdgeFactors {
    name: &'static str,
    domain: Domain,
}

/// A sweep over the four factor step functions.
pub struct FactorSteps {
    name: &'static str,
}

/// A sweep over the three control-point index types, with the two tessellation vertex steps read
/// under the untyped one.
pub struct PatchIndices {
    name: &'static str,
}

/// Every case of the family, in the order the report lists them.
pub fn boxed() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for domain in [Domain::Quad, Domain::Triangle] {
        cases.push(Box::new(PartitionModes {
            name: match domain {
                Domain::Quad => "tessellation-partition-modes-quad",
                Domain::Triangle => "tessellation-partition-modes-triangle",
            },
            domain,
        }));
    }
    cases.push(Box::new(EdgeFactors {
        name: "tessellation-edge-factors",
        domain: Domain::Quad,
    }));
    cases.push(Box::new(NonFiniteFactors {
        name: "tessellation-non-finite-factors",
    }));
    cases.push(Box::new(FactorSteps {
        name: "tessellation-factor-steps",
    }));
    cases.push(Box::new(FactorScales {
        name: "tessellation-factor-scales",
    }));
    cases.push(Box::new(PatchIndices {
        name: "tessellation-patch-indices",
    }));
    cases
}

impl PartitionModes {
    /// The fixture under the case's entry point.
    fn fixture(&self) -> Fixture {
        self.domain.fixture()
    }

    /// The tessellation state one mode's pipeline states.
    fn state(&self, mode: Partition) -> TessellationState {
        TessellationState::new()
            .partitioned(mode)
            .clamped_to(SWEPT_MAX)
    }
}

impl Case for PartitionModes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing observed is loosened: a mark is a word that is 0 or 1 and a cell is a cell.
        // These are the notice declarations the module docs explain.
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(self.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut inputs = air_inputs(fixture)?;
        inputs.push(Input::bytes("grid", &grid_record(FINE_GRID)));
        inputs.push(Input::bytes(
            "factors",
            &self.domain.factors(SWEPT_EDGES, SWEPT_INSIDE),
        ));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; partitions().len() * REGION_BYTES as usize],
        ));
        let mut pipelines = Vec::new();
        for mode in partitions() {
            pipelines.extend(partition_site(mode).as_bytes());
            pipelines.extend(domain_pipeline(fixture.entry, self.state(mode)).words());
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = self.fixture();
        let loaded = load(fixture)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let grid = made(sites, "buffer/grid", side.buffer(&grid_record(FINE_GRID)))?;
        let factors = made(
            sites,
            "buffer/factors",
            side.buffer(&self.domain.factors(SWEPT_EDGES, SWEPT_INSIDE)),
        )?;
        let modes = partitions();
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![0u8; modes.len() * REGION_BYTES as usize]),
        )?;

        let mut pipelines = Vec::with_capacity(modes.len());
        for &mode in &modes {
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", partition_site(mode)),
                side.render_pipeline(library, &domain_pipeline(fixture.entry, self.state(mode))),
                |_| Value::Absent,
            ));
        }

        let mut work = Vec::new();
        let mut drawn = Vec::new();
        for (at, pipeline) in pipelines.iter().enumerate() {
            let Some(pipeline) = pipeline else { continue };
            drawn.push(at);
            work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(
                one_patch_steps(*pipeline, grid, factors, marks, at as u32 * REGION_BYTES, 1),
            )));
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        // The four modes hang under one branch named after the `MTLPatchType` the case's function
        // declares, so a transcript holds that enumerator by name beside the modes that ran under
        // it and the tree says which domain each answer is of.
        let mut nodes = Vec::with_capacity(modes.len());
        for (at, &mode) in modes.iter().enumerate() {
            let base = at as u64 * u64::from(REGION_BYTES);
            let region = read_marks(side, sites, partition_site(mode), marks, base)?;
            let mut node = Node::branch(partition_site(mode), Vec::new()).with(Node::leaf(
                "marks",
                Observed::bytes(region, marks_layout(base)),
            ));
            if let Some(pass) = drawn.iter().position(|held| *held == at) {
                if let Some(Some(failure)) = outcome.refused.get(pass) {
                    node = node.with(Node::leaf("pass", Observed::value(failure.recorded())));
                }
            }
            nodes.push(node);
        }
        sites.push(Node::branch(self.domain.patch_type(), nodes));
        Ok(())
    }
}

/// The factor lists [`EdgeFactors`] sweeps, each with the site it is emitted at.
///
/// The five values Metal's own rules make special, on every edge and inside at once:
///
/// - **0** discards the patch, so the answer is a region with no mark in it at all. That is the
///   family's one control that cannot be produced by any arithmetic error: a tessellator that
///   drew anything here would show up as marks where nothing may be marked.
/// - **0.5**, below 1, which Metal clamps **up** to 1. Its answer is 1's by construction, and the
///   case says so rather than pretending the two are separate questions.
/// - **1**, the floor: the four corners of the domain and nothing between them.
/// - **64**, the pipeline's stated `maxTessellationFactor` and Metal's documented ceiling for the
///   field: every one of the 4225 cells of the grid.
/// - **65**, one above it, which the clamp takes back to 64. Its answer is 64's by construction.
///
/// So the list is five values, three distinct answers, and the two collapses are Metal's clamp
/// rather than a hole in the sweep. The sweep is run under `integer` partitioning, whose spacing is
/// uniform, so a factor is the point count minus one and the clamp is read directly off the marks.
const FACTOR_SITES: [(&str, f32); 5] = [
    ("factor-0", 0.0),
    ("factor-below-1", 0.5),
    ("factor-1", 1.0),
    ("factor-max", 64.0),
    ("factor-above-max", 65.0),
];

/// The ceiling the factor sweep states: Metal documents `maxTessellationFactor` as `1..=64`, and
/// this is the top of it, so `factor-above-max` is above a maximum the device really has.
const FACTOR_MAX: u32 = 64;

impl EdgeFactors {
    fn fixture(&self) -> Fixture {
        self.domain.fixture()
    }

    /// The one pipeline every site of the sweep draws with: the factors are the buffer's and not
    /// the pipeline's, so one pipeline answers all five.
    fn state(&self) -> TessellationState {
        TessellationState::new()
            .partitioned(Partition::Integer)
            .clamped_to(FACTOR_MAX)
    }

    /// The factor buffer: one 12-byte record a site, in [`FACTOR_SITES`]' order.
    fn records(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (_, factor) in FACTOR_SITES {
            bytes.extend(self.domain.factors([factor; 4], [factor; 2]));
        }
        bytes
    }

    /// Bytes one factor record takes, which is the stride a per-patch step would read it at and the
    /// offset this case binds each site's record from.
    fn record_bytes(&self) -> u32 {
        match self.domain {
            Domain::Quad => 12,
            Domain::Triangle => 8,
        }
    }
}

impl Case for EdgeFactors {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(self.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut inputs = air_inputs(fixture)?;
        inputs.push(Input::bytes("grid", &grid_record(FINE_GRID)));
        inputs.push(Input::bytes("factors", &self.records()));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; FACTOR_SITES.len() * REGION_BYTES as usize],
        ));
        inputs.push(Input::bytes(
            "pipeline",
            &domain_pipeline(fixture.entry, self.state()).words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = self.fixture();
        let loaded = load(fixture)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let grid = made(sites, "buffer/grid", side.buffer(&grid_record(FINE_GRID)))?;
        let factors = made(sites, "buffer/factors", side.buffer(&self.records()))?;
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![0u8; FACTOR_SITES.len() * REGION_BYTES as usize]),
        )?;
        let pipeline = probe(
            sites,
            "pipeline",
            side.render_pipeline(library, &domain_pipeline(fixture.entry, self.state())),
            |_| Value::Absent,
        );

        let mut work = Vec::new();
        if let Some(pipeline) = pipeline {
            for (at, _) in FACTOR_SITES.iter().enumerate() {
                let mut steps =
                    one_patch_steps(pipeline, grid, factors, marks, at as u32 * REGION_BYTES, 1);
                // The site's own factor record, bound where the constant step reads it.
                steps[1] = RenderStep::TessellationFactorBuffer {
                    buffer: Some(factors),
                    offset: u64::from(at as u32 * self.record_bytes()),
                    instance_stride: 0,
                };
                work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps)));
            }
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        for (at, (site, _)) in FACTOR_SITES.iter().enumerate() {
            let base = at as u64 * u64::from(REGION_BYTES);
            let region = read_marks(side, sites, site, marks, base)?;
            let mut node = Node::branch(*site, Vec::new()).with(Node::leaf(
                "marks",
                Observed::bytes(region, marks_layout(base)),
            ));
            if let Some(Some(failure)) = outcome.refused.get(at) {
                node = node.with(Node::leaf("pass", Observed::value(failure.recorded())));
            }
            sites.push(node);
        }
        Ok(())
    }
}

// ---- The factors Metal's rules treat as neither a number nor a positive one --------------------

/// The grid scale the two sweeps below state: 16 cells on an axis, which tells the levels 1, 2, 4,
/// 8, 9 and 16 apart and keeps each leg's compared region to seventeen rows of the mark grid.
const MEDIUM_GRID: f32 = 16.0;

/// Grid rows a medium leg can reach: `j = rint(v · 16)` for `v` in `[0, 1]`, so `j` is in `0..=16`.
const MEDIUM_ROWS: u32 = 17;

/// Words of a medium region a leg compares: `MEDIUM_ROWS` whole rows of the mark grid, so the
/// prefix holds every mark such a draw can make and a diff names the cell.
const MEDIUM_WORDS: u32 = MEDIUM_ROWS * AXIS;

/// The ceiling both medium sweeps state, so a level of 16 is the maximum and a factor above it is
/// clamped.
const MEDIUM_MAX: u32 = 16;

/// `half`'s bit pattern for 4.0, the finite factor the non-finite sweep puts where the value it is
/// asking about does not go, and the one the scale sweep scales.
const FINITE: u16 = 0x4400;

/// The `half` bit patterns Metal's own factor rules single out, each neither a finite positive
/// number nor the one the sweep's control states.
///
/// Metal's rules (the control stage's module docs, which take them from Metal's documentation):
/// an **edge** factor that is negative, zero or NaN culls the patch, and every factor is then
/// clamped to its partition's range, where a NaN takes the range's lower end and an infinity its
/// upper one. That is two rules reading one value, and which of the two a value meets depends on
/// where it sits, so the sweep crosses each value with its placement rather than setting all six
/// factors at once.
const NON_FINITE: [(&str, u16); 6] = [
    ("nan", 0x7e00),
    ("negative", 0xbc00),
    ("negative-zero", 0x8000),
    ("positive-infinity", 0x7c00),
    ("negative-infinity", 0xfc00),
    // The least positive subnormal `half`, which is a finite positive number and so meets neither
    // rule: it is here because the control stage's own docs list a subnormal edge factor as a
    // reading taken without a Metal observation.
    ("subnormal", 0x0001),
];

/// Where a non-finite sweep cell puts the value it asks about; the other factors are [`FINITE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// The four edge factors, where Metal's cull reads them.
    Edges,
    /// The two inside factors, which no cull reads and only the clamp does.
    Inside,
    /// Both, which is a question of its own only where the value does not cull.
    Both,
}

impl Placement {
    fn site(self) -> &'static str {
        match self {
            Placement::Edges => "edges",
            Placement::Inside => "inside",
            Placement::Both => "both",
        }
    }

    /// One patch's quad factor record with `value` in this placement.
    fn record(self, value: u16) -> Vec<u8> {
        let (edges, inside) = match self {
            Placement::Edges => (value, FINITE),
            Placement::Inside => (FINITE, value),
            Placement::Both => (value, value),
        };
        raw_quad_factors([edges; 4], [inside; 2])
    }
}

/// One patch's `MTLQuadTessellationFactorsHalf` from the `half` bit patterns themselves, which is
/// how a case states a factor [`half_bits`] refuses: a NaN, a negative number, an infinity or a
/// subnormal.
fn raw_quad_factors(edges: [u16; 4], inside: [u16; 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12);
    for bits in edges.iter().chain(inside.iter()) {
        bytes.extend(bits.to_le_bytes());
    }
    bytes
}

/// The cells the non-finite sweep states: the control, then each value of [`NON_FINITE`] in the
/// two placements Metal's two rules read, and `Both` for the two values that cull under neither.
///
/// `Both` is **not** stated for a culling value, and that is a reason rather than an omission: an
/// edge cull discards the patch whatever the inside factors are, which the value's `Edges` cell
/// already asks (its insides are [`FINITE`]), so a `Both` cell beside it would compare one empty
/// region twice.
fn non_finite_cells() -> Vec<NonFiniteCell> {
    let mut cells = vec![NonFiniteCell {
        value: "all-finite",
        placement: None,
        record: raw_quad_factors([FINITE; 4], [FINITE; 2]),
    }];
    for (value, bits) in NON_FINITE {
        for placement in [Placement::Edges, Placement::Inside, Placement::Both] {
            if placement == Placement::Both && culls_an_edge(bits) {
                continue;
            }
            cells.push(NonFiniteCell {
                value,
                placement: Some(placement),
                record: placement.record(bits),
            });
        }
    }
    cells
}

/// One cell of the non-finite sweep: the value it asks about, where that value goes, and the
/// patch's factor record it comes to.
///
/// A cell's answer is emitted at the placement's site under the value's, so the transcript is a
/// tree of values each holding the placements asked of it, and the control — which has no
/// placement, every factor being [`FINITE`] — holds its marks directly.
struct NonFiniteCell {
    value: &'static str,
    placement: Option<Placement>,
    record: Vec<u8>,
}

/// Whether Metal's first rule culls a patch whose edge factor is `bits`: a `half` that is
/// negative, zero or NaN, which is every bit pattern outside `[0x0001, 0x7c00]`.
fn culls_an_edge(bits: u16) -> bool {
    !(0x0001..=0x7c00).contains(&bits)
}

/// The non-finite factor sweep: the edge and inside factors Metal's cull and clamp read as
/// something other than a number, on the committed quad fixture.
pub struct NonFiniteFactors {
    name: &'static str,
}

impl NonFiniteFactors {
    /// The one pipeline every cell draws with: `integer` partitioning, whose spacing is uniform,
    /// so a level is the segment count and the marks say what each rule produced.
    fn state(&self) -> TessellationState {
        TessellationState::new()
            .partitioned(Partition::Integer)
            .clamped_to(MEDIUM_MAX)
    }

    /// The factor buffer: one 12-byte record a cell, in [`non_finite_cells`]' order.
    fn records(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for cell in non_finite_cells() {
            bytes.extend(cell.record);
        }
        bytes
    }
}

impl Case for NonFiniteFactors {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(QUAD)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = air_inputs(QUAD)?;
        inputs.push(Input::bytes("grid", &grid_record(MEDIUM_GRID)));
        inputs.push(Input::bytes("factors", &self.records()));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; non_finite_cells().len() * REGION_BYTES as usize],
        ));
        inputs.push(Input::bytes(
            "pipeline",
            &domain_pipeline(QUAD.entry, self.state()).words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = load(QUAD)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let grid = made(sites, "buffer/grid", side.buffer(&grid_record(MEDIUM_GRID)))?;
        let factors = made(sites, "buffer/factors", side.buffer(&self.records()))?;
        let cells = non_finite_cells();
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![0u8; cells.len() * REGION_BYTES as usize]),
        )?;
        let pipeline = probe(
            sites,
            "pipeline",
            side.render_pipeline(library, &domain_pipeline(QUAD.entry, self.state())),
            |_| Value::Absent,
        );

        let mut work = Vec::new();
        if let Some(pipeline) = pipeline {
            for at in 0..cells.len() {
                let mut steps =
                    one_patch_steps(pipeline, grid, factors, marks, at as u32 * REGION_BYTES, 1);
                // The cell's own record, bound where the constant step reads it.
                steps[1] = RenderStep::TessellationFactorBuffer {
                    buffer: Some(factors),
                    offset: at as u64 * u64::from(QUAD_FACTOR_BYTES),
                    instance_stride: 0,
                };
                work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps)));
            }
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        // Each value is one branch holding the placements asked of it, and the control — which has
        // no placement — holds its marks directly, so the transcript says which of Metal's two
        // rules each answer is of.
        let mut groups: Vec<(&'static str, Vec<Node>)> = Vec::new();
        for (at, cell) in cells.iter().enumerate() {
            let base = at as u64 * u64::from(REGION_BYTES);
            let site = match cell.placement {
                None => cell.value.to_owned(),
                Some(placement) => format!("{}/{}", cell.value, placement.site()),
            };
            let region = read_medium_marks(side, sites, &site, marks, base)?;
            let mut leaves = vec![Node::leaf(
                "marks",
                Observed::bytes(region, medium_layout(base)),
            )];
            if let Some(Some(failure)) = outcome.refused.get(at) {
                leaves.push(Node::leaf("pass", Observed::value(failure.recorded())));
            }
            let held = match cell.placement {
                None => leaves,
                Some(placement) => vec![Node::branch(placement.site(), leaves)],
            };
            match groups.last_mut() {
                Some((value, nodes)) if *value == cell.value => nodes.extend(held),
                _ => groups.push((cell.value, held)),
            }
        }
        for (value, nodes) in groups {
            sites.push(Node::branch(value, nodes));
        }
        Ok(())
    }
}

// ---- The factor step functions -----------------------------------------------------------------

/// Every `MTLTessellationFactorStepFunction`, in the SDK table's own declaration order.
fn steps() -> Vec<Step> {
    Step::ALL.to_vec()
}

/// The grid scale the step sweep states: nine cells on an axis, which is enough to tell factors 1,
/// 2, 4 and 8 apart (2, 3, 5 and 9 points on an edge) and keeps each leg's compared region to the
/// 529 words those cells span.
const COARSE_GRID: f32 = 8.0;

/// Words of a coarse region a leg compares: every cell a grid of scale 8 can reach, which is
/// `j · 65 + i` for `i, j` in `0..=8`. A domain coordinate is in `[0, 1]`, so `i` and `j` are never
/// above 8 and the prefix holds **every** mark the draw could make.
const COARSE_WORDS: u32 = 8 * AXIS + 9;

/// The four factor records the step sweep lays out, one a record of 12 bytes: factors 1, 2, 4 and
/// 8, whose point sets are 2, 3, 5 and 9 points on an edge and so are plainly different sets.
const STEP_FACTORS: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

/// Bytes one quad factor record takes, which is the stride Metal reads a per-patch step at.
const QUAD_FACTOR_BYTES: u32 = 12;

/// The instance stride the step sweep binds: two records, so instance 1 reaches record 2 and the
/// per-patch step inside it reaches record 3.
const STEP_INSTANCE_STRIDE: u32 = 2 * QUAD_FACTOR_BYTES;

/// One leg of the step sweep: a draw narrow enough that its marks say which factor record it read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Leg {
    /// The site the leg's answer is emitted at.
    site: &'static str,
    /// `patchStart`, with a patch count of 1.
    patch: u32,
    /// `instanceCount`.
    instances: u32,
    /// `baseInstance`.
    base_instance: u32,
}

/// The four legs each step function is asked about, chosen so that **no two of the four step
/// functions answer the same four legs**.
///
/// The naive sweep — one patch and one instance, the instance picked with `baseInstance` — cannot
/// do that, and the reason is the open question this family measures: whether the factor address's
/// instance index counts the base instance is a reading the control stage took from Metal's
/// documentation with no Metal observation behind it (`PrecisionNote::TessellationFactorInstance`).
/// If it does not count, every `baseInstance` leg reads record 0 and `PerInstance` answers exactly
/// what `Constant` answers on every leg — a sweep comparing one answer twice. So the sweep reaches
/// a second instance the way that cannot depend on that reading, by **drawing** two instances, and
/// keeps the `baseInstance` leg beside it as the question rather than as the mechanism.
///
/// Two instances write one marks region, and that is sound rather than a compromise: a mark is a
/// set member, so the region holds the union of the two instances' point sets. The records are
/// nested by construction (factor 1's points are factor 4's, which is what [`STEP_FACTORS`] is
/// chosen for), so the union of record 0 and record 2 **is** record 2's set and the leg reads as
/// cleanly as a single draw would.
///
/// What each step function then answers, by record:
///
/// | step | `patch-0` | `patch-1` | `two-instances` | `base-instance-1` |
/// |---|---|---|---|---|
/// | `Constant` | 0 | 0 | 0 | 0 |
/// | `PerPatch` | 0 | 1 | 0 | 0 |
/// | `PerInstance` | 0 | 0 | 2 | 0 or 2 |
/// | `PerPatchAndPerInstance` | 0 | 1 | 2 | 0 or 2 |
///
/// `PerPatch` parts from `Constant` at `patch-1`, `PerInstance` from `Constant` at
/// `two-instances`, and `PerPatchAndPerInstance` from each of the other three at one leg or the
/// other, whichever way the base-instance reading falls. `tests::no_two_step_functions_can_answer
/// _the_same_legs` holds that from the addressing rather than from the recording.
const STEP_LEGS: [Leg; 4] = [
    Leg {
        site: "patch-0",
        patch: 0,
        instances: 1,
        base_instance: 0,
    },
    Leg {
        site: "patch-1",
        patch: 1,
        instances: 1,
        base_instance: 0,
    },
    Leg {
        site: "two-instances",
        patch: 0,
        instances: 2,
        base_instance: 0,
    },
    Leg {
        site: "base-instance-1",
        patch: 0,
        instances: 1,
        base_instance: 1,
    },
];

impl FactorSteps {
    fn state(&self, step: Step) -> TessellationState {
        TessellationState::new()
            .partitioned(Partition::Integer)
            .clamped_to(SWEPT_MAX)
            .stepping(step)
    }

    /// The factor buffer: the four records of [`STEP_FACTORS`], 12 bytes each.
    fn records(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for factor in STEP_FACTORS {
            bytes.extend(quad_factors([factor; 4], [factor; 2]));
        }
        bytes
    }

    /// The per-patch grid records: the same scale for both patches, so the grid is not an axis of
    /// this case and a difference in a mark is the factor's.
    fn grids(&self) -> Vec<u8> {
        let mut bytes = grid_record(COARSE_GRID);
        bytes.extend(grid_record(COARSE_GRID));
        bytes
    }

    /// The leg at `leg` of the step function at `step`, as a region index of the marks buffer.
    fn region(step: usize, leg: usize) -> u32 {
        (step * STEP_LEGS.len() + leg) as u32
    }

    /// Regions the marks buffer holds: one a leg of a step function.
    fn regions() -> usize {
        steps().len() * STEP_LEGS.len()
    }
}

impl Case for FactorSteps {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // This is the one case that sets a per-instance factor step, so it is the one case whose
        // run can announce the notice the instance addressing is read under.
        let mut all = tessellation_notices();
        all.push(crate::replay::notice_allowance(
            Degradation::TessellationFactorInstance.code(),
        ));
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(QUAD)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = air_inputs(QUAD)?;
        inputs.push(Input::bytes("grids", &self.grids()));
        inputs.push(Input::bytes("factors", &self.records()));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; Self::regions() * REGION_BYTES as usize],
        ));
        let mut pipelines = Vec::new();
        for step in steps() {
            pipelines.extend(step.name().as_bytes());
            pipelines.extend(domain_pipeline(QUAD.entry, self.state(step)).words());
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = load(QUAD)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let grids = made(sites, "buffer/grids", side.buffer(&self.grids()))?;
        let factors = made(sites, "buffer/factors", side.buffer(&self.records()))?;
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![0u8; Self::regions() * REGION_BYTES as usize]),
        )?;
        let all = steps();
        let mut pipelines = Vec::with_capacity(all.len());
        for &step in &all {
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", step.name()),
                side.render_pipeline(library, &domain_pipeline(QUAD.entry, self.state(step))),
                |_| Value::Absent,
            ));
        }

        let mut work = Vec::new();
        for (at, pipeline) in pipelines.iter().enumerate() {
            let Some(pipeline) = pipeline else { continue };
            for (leg, held) in STEP_LEGS.iter().enumerate() {
                let region = Self::region(at, leg) * REGION_BYTES;
                work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(vec![
                    RenderStep::Pipeline(*pipeline),
                    RenderStep::TessellationFactorBuffer {
                        buffer: Some(factors),
                        offset: 0,
                        instance_stride: u64::from(STEP_INSTANCE_STRIDE),
                    },
                    RenderStep::VertexBuffer {
                        buffer: grids,
                        offset: 0,
                        index: 0,
                    },
                    RenderStep::VertexBuffer {
                        buffer: marks,
                        offset: u64::from(region),
                        index: 1,
                    },
                    RenderStep::DrawPatches {
                        control_points: 1,
                        first: held.patch,
                        count: 1,
                        patch_indices: None,
                        patch_index_offset: 0,
                        instances: held.instances,
                        base_instance: held.base_instance,
                    },
                ])));
            }
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        let mut pass = 0;
        for (at, &step) in all.iter().enumerate() {
            let mut node = Node::branch(step.name(), Vec::new());
            for (leg, named) in STEP_LEGS.iter().enumerate() {
                let base = u64::from(Self::region(at, leg) * REGION_BYTES);
                let site = format!("{}/{}", step.name(), named.site);
                let region = read_coarse_marks(side, sites, &site, marks, base)?;
                let mut held = Node::branch(named.site, Vec::new()).with(Node::leaf(
                    "marks",
                    Observed::bytes(region, coarse_layout(base)),
                ));
                if pipelines[at].is_some() {
                    if let Some(Some(failure)) = outcome.refused.get(pass) {
                        held = held.with(Node::leaf("pass", Observed::value(failure.recorded())));
                    }
                    pass += 1;
                }
                node = node.with(held);
            }
            sites.push(node);
        }
        Ok(())
    }
}

// ---- The factor scale, crossed with the factor it scales ---------------------------------------

/// A quiet NaN's `f32` bits, stated here rather than taken from `f32::NAN` so the payload a case
/// sets is the case's own and a recording keys by it.
const NAN_BITS: u32 = 0x7fc0_0000;

/// `+∞` as `f32` bits.
const POSITIVE_INFINITY_BITS: u32 = 0x7f80_0000;

/// `-∞` as `f32` bits.
const NEGATIVE_INFINITY_BITS: u32 = 0xff80_0000;

/// `value`'s `f32` bits, as a scale a cell sets.
const fn scale_bits(value: f32) -> Option<u32> {
    Some(value.to_bits())
}

/// The factor records the scale sweep lays out, one a record of [`QUAD_FACTOR_BYTES`], every edge
/// and inside factor the same number.
///
/// Four numbers, each there for a cell that crosses it with a scale: 4 is the one most cells
/// scale, and 0.5, 2 and 17 are the three the clamp's two ends read — 0.5 below the floor, 17
/// above the ceiling [`MEDIUM_MAX`] states, and 2 the number whose halving reaches the floor from
/// above.
const SCALE_FACTORS: [f32; 4] = [4.0, 0.5, 2.0, 17.0];

/// One cell of the scale sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScaleCell {
    /// The site the cell's answer is emitted at.
    site: &'static str,
    /// Whether the cell's pipeline states `tessellationFactorScaleEnabled`.
    enabled: bool,
    /// The factor record, as an index of [`SCALE_FACTORS`].
    record: usize,
    /// The `f32` bits of the scale the pass sets, or `None` for a pass that sets **no** scale,
    /// which is its own shape rather than a scale of 1.
    scale: Option<u32>,
}

/// The cells the scale sweep states.
///
/// # What each cell is for
///
/// The first group holds the factor at 4 and moves the scale alone, which is the axis
/// `setTessellationFactorScale:` is: `scale-unset` is the pass that never makes the call,
/// `scale-1` and `scale-2` are two scales that draw two different point sets, and
/// `scale-rounds-to-2-as-half` is a third number whose answer is `scale-2`'s **because Metal
/// rounds the scale to `half` first** — a collapse that is the measurement rather than a hole.
///
/// `scale-subnormal-as-half` and `scale-zero-as-half` are the pair that says where that rounding
/// takes the scale to nothing: `2^-15` is subnormal as a `half` and its product still draws,
/// `2^-30` is 0 as a `half` and culls. Neither is a number Metal's own validation takes, and both
/// are what plain Metal does with it. A scale that is subnormal as a **`f32`** is not a cell: it is
/// 0 as a `half`, so it is `scale-zero-as-half`'s answer by the same rounding.
///
/// The second group is the three cells that read the **order** of the scale and the clamp, which is
/// the one thing a scale crossed with a factor can ask and a scale alone cannot:
///
/// - `scale-2-of-below-1` scales 0.5 by 2. Scaled before the clamp that is `1`, clamped to 1;
///   clamped first it is `max(0.5, 1) = 1` scaled to 2. One segment an edge against two.
/// - `scale-half-of-above-max` halves 17 under a ceiling of 16. Scaled first that is 8.5, which
///   `integer` rounds up to 9; clamped first it is 16 halved to 8.
/// - `scale-half-of-2` halves 2, which is 1 **either way round**. It is the control beside the
///   two above: a side that moved the clamp would leave this cell alone, so the cell says the
///   other two moved for the order and not for the arithmetic.
///
/// The third group is `tessellationFactorScaleEnabled` itself, which is a field and not a call:
/// `scale-disabled` sets a scale of 2 on a pipeline that does not enable it and draws what factor
/// 4 draws, and `scale-0-disabled` sets a scale that culls every patch of an enabling pipeline and
/// draws the same. Those two are what say the cull is the **field's** and not the setter's.
const SCALE_CELLS: [ScaleCell; 16] = [
    ScaleCell {
        site: "scale-unset",
        enabled: true,
        record: 0,
        scale: None,
    },
    ScaleCell {
        site: "scale-1",
        enabled: true,
        record: 0,
        scale: scale_bits(1.0),
    },
    ScaleCell {
        site: "scale-2",
        enabled: true,
        record: 0,
        scale: scale_bits(2.0),
    },
    ScaleCell {
        site: "scale-rounds-to-2-as-half",
        enabled: true,
        record: 0,
        scale: scale_bits(2.0005),
    },
    ScaleCell {
        site: "scale-0",
        enabled: true,
        record: 0,
        scale: scale_bits(0.0),
    },
    ScaleCell {
        site: "scale-negative",
        enabled: true,
        record: 0,
        scale: scale_bits(-1.0),
    },
    ScaleCell {
        site: "scale-nan",
        enabled: true,
        record: 0,
        scale: Some(NAN_BITS),
    },
    ScaleCell {
        site: "scale-positive-infinity",
        enabled: true,
        record: 0,
        scale: Some(POSITIVE_INFINITY_BITS),
    },
    ScaleCell {
        site: "scale-negative-infinity",
        enabled: true,
        record: 0,
        scale: Some(NEGATIVE_INFINITY_BITS),
    },
    ScaleCell {
        site: "scale-subnormal-as-half",
        enabled: true,
        record: 0,
        // 2^-15, below `half`'s least normal and well above its least subnormal.
        scale: scale_bits(3.051_757_8e-5),
    },
    ScaleCell {
        site: "scale-zero-as-half",
        enabled: true,
        record: 0,
        // 2^-30, which rounds to 0 as a `half`.
        scale: scale_bits(9.313_225_7e-10),
    },
    ScaleCell {
        site: "scale-2-of-below-1",
        enabled: true,
        record: 1,
        scale: scale_bits(2.0),
    },
    ScaleCell {
        site: "scale-half-of-2",
        enabled: true,
        record: 2,
        scale: scale_bits(0.5),
    },
    ScaleCell {
        site: "scale-half-of-above-max",
        enabled: true,
        record: 3,
        scale: scale_bits(0.5),
    },
    ScaleCell {
        site: "scale-disabled",
        enabled: false,
        record: 0,
        scale: scale_bits(2.0),
    },
    ScaleCell {
        site: "scale-0-disabled",
        enabled: false,
        record: 0,
        scale: scale_bits(0.0),
    },
];

/// A sweep over `setTessellationFactorScale:` crossed with the factor it scales, and over
/// `tessellationFactorScaleEnabled` beside it.
pub struct FactorScales {
    name: &'static str,
}

impl FactorScales {
    /// The tessellation state of a cell's pipeline: `integer` partitioning, whose spacing is
    /// uniform, so a level is the segment count and the marks read the product directly.
    fn state(enabled: bool) -> TessellationState {
        let state = TessellationState::new()
            .partitioned(Partition::Integer)
            .clamped_to(MEDIUM_MAX);
        match enabled {
            true => state.scaled(),
            false => state,
        }
    }

    /// The site a pipeline is made at, which is the field value it states: a case's transcript
    /// holds `tessellationFactorScaleEnabled` by name on both legs.
    fn pipeline_site(enabled: bool) -> &'static str {
        match enabled {
            true => "tessellationFactorScaleEnabled-true",
            false => "tessellationFactorScaleEnabled-false",
        }
    }

    /// The factor buffer: the four records of [`SCALE_FACTORS`], 12 bytes each.
    fn records(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for factor in SCALE_FACTORS {
            bytes.extend(quad_factors([factor; 4], [factor; 2]));
        }
        bytes
    }

    /// The steps one cell's pass holds: the cell's pipeline, its factor record, the grid, the
    /// marks region, the scale where the cell sets one, and a draw of one patch.
    fn steps(cell: &ScaleCell, held: &Handles, at: usize) -> Vec<RenderStep> {
        let mut steps = vec![
            RenderStep::Pipeline(held.pipeline),
            RenderStep::TessellationFactorBuffer {
                buffer: Some(held.factors),
                offset: cell.record as u64 * u64::from(QUAD_FACTOR_BYTES),
                instance_stride: 0,
            },
            RenderStep::VertexBuffer {
                buffer: held.grid,
                offset: 0,
                index: 0,
            },
            RenderStep::VertexBuffer {
                buffer: held.marks,
                offset: at as u64 * u64::from(REGION_BYTES),
                index: 1,
            },
        ];
        if let Some(bits) = cell.scale {
            steps.push(RenderStep::TessellationFactorScale { bits });
        }
        steps.push(RenderStep::DrawPatches {
            control_points: 1,
            first: 0,
            count: 1,
            patch_indices: None,
            patch_index_offset: 0,
            instances: 1,
            base_instance: 0,
        });
        steps
    }
}

/// What one cell's pass of [`FactorScales`] binds, which is one pipeline and three buffers.
struct Handles {
    pipeline: Handle,
    grid: Handle,
    factors: Handle,
    marks: Handle,
}

impl Case for FactorScales {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // This is the one case that enables the factor scale, so it is the one case whose run can
        // announce the notice the scale is read under.
        let mut all = tessellation_notices();
        all.push(crate::replay::notice_allowance(
            Degradation::TessellationFactorScale.code(),
        ));
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(QUAD)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = air_inputs(QUAD)?;
        inputs.push(Input::bytes("grid", &grid_record(MEDIUM_GRID)));
        inputs.push(Input::bytes("factors", &self.records()));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; SCALE_CELLS.len() * REGION_BYTES as usize],
        ));
        let mut pipelines = Vec::new();
        for enabled in [true, false] {
            pipelines.extend(Self::pipeline_site(enabled).as_bytes());
            pipelines.extend(domain_pipeline(QUAD.entry, Self::state(enabled)).words());
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        let mut cells = Vec::new();
        for cell in SCALE_CELLS {
            cells.extend(cell.site.as_bytes());
            cells.push(u8::from(cell.enabled));
            cells.extend((cell.record as u32).to_le_bytes());
            cells.extend(cell.scale.unwrap_or(0).to_le_bytes());
            cells.push(u8::from(cell.scale.is_some()));
        }
        inputs.push(Input::bytes("cells", &cells));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = load(QUAD)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let grid = made(sites, "buffer/grid", side.buffer(&grid_record(MEDIUM_GRID)))?;
        let factors = made(sites, "buffer/factors", side.buffer(&self.records()))?;
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![0u8; SCALE_CELLS.len() * REGION_BYTES as usize]),
        )?;
        let mut pipelines = Vec::with_capacity(2);
        for enabled in [true, false] {
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", Self::pipeline_site(enabled)),
                side.render_pipeline(library, &domain_pipeline(QUAD.entry, Self::state(enabled))),
                |_| Value::Absent,
            ));
        }

        let mut work = Vec::new();
        let mut drawn = Vec::new();
        for (at, cell) in SCALE_CELLS.iter().enumerate() {
            let Some(pipeline) = pipelines[usize::from(!cell.enabled)] else {
                continue;
            };
            drawn.push(at);
            let held = Handles {
                pipeline,
                grid,
                factors,
                marks,
            };
            work.push(RenderWork::Pass(
                RenderPass::sized([1, 1]).drawing(Self::steps(cell, &held, at)),
            ));
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        // Each cell hangs under the `tessellationFactorScaleEnabled` value its pipeline states, so
        // the field is in the transcript by name beside the cells that ran under it.
        for enabled in [true, false] {
            let mut nodes = Vec::new();
            for (at, cell) in SCALE_CELLS.iter().enumerate() {
                if cell.enabled != enabled {
                    continue;
                }
                let base = at as u64 * u64::from(REGION_BYTES);
                let region = read_medium_marks(side, sites, cell.site, marks, base)?;
                let mut node = Node::branch(cell.site, Vec::new()).with(Node::leaf(
                    "marks",
                    Observed::bytes(region, medium_layout(base)),
                ));
                if let Some(pass) = drawn.iter().position(|held| *held == at) {
                    if let Some(Some(failure)) = outcome.refused.get(pass) {
                        node = node.with(Node::leaf("pass", Observed::value(failure.recorded())));
                    }
                }
                nodes.push(node);
            }
            sites.push(Node::branch(Self::pipeline_site(enabled), nodes));
        }
        Ok(())
    }
}

/// The layout a medium mark region is compared under: [`MEDIUM_ROWS`] whole rows of the mark grid,
/// one row of `AXIS` words a grid row, so a diff names the cell.
fn medium_layout(offset: u64) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset,
            length: u64::from(MEDIUM_WORDS * 4),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: AXIS,
        row_pitch: AXIS * 4,
        rows: MEDIUM_ROWS,
        image_pitch: MEDIUM_WORDS * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The rows a medium grid can reach of the region at `base`.
fn read_medium_marks(
    side: &mut dyn Side,
    sites: &mut Sites,
    site: &str,
    marks: Handle,
    base: u64,
) -> Result<Vec<u8>, Abort> {
    call(
        sites,
        &format!("read-back/{site}"),
        side.read_buffer(marks, base, MEDIUM_WORDS as usize * 4),
        |_| Value::Absent,
    )
}

/// The layout a coarse mark region is compared under: the [`COARSE_WORDS`] a grid of scale 8 can
/// reach, one row of `AXIS` words a grid row.
fn coarse_layout(offset: u64) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset,
            length: u64::from(COARSE_WORDS * 4),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: COARSE_WORDS,
        row_pitch: COARSE_WORDS * 4,
        rows: 1,
        image_pitch: COARSE_WORDS * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
}

// ---- The control-point index types and the two tessellation vertex steps -----------------------

/// Patches [`PatchIndices`] draws, each of three control points.
const PATCH_COUNT: u32 = 4;

/// Control points the patch fixture's function declares, and so the number its draws state.
const CONTROL_POINTS: u32 = 3;

/// Words one site of [`PatchIndices`] reads: one a patch drawn.
const PATCH_WORDS: u32 = PATCH_COUNT;

/// Bytes one site's region takes, 16-byte aligned so every binding offset is a multiple of 4.
const PATCH_REGION_BYTES: u32 = 16;

/// The control-point tags: tag `k` at element `k`, so patch `p`'s three control points are
/// `(3p, 3p + 1, 3p + 2)` where no index buffer moves them, and each patch's first tag names its
/// own mark (`tag / 3`).
fn control_point_records() -> Vec<u8> {
    let mut bytes = Vec::new();
    for tag in 0..PATCH_COUNT * CONTROL_POINTS {
        bytes.extend(tag.to_le_bytes());
    }
    bytes
}

/// The per-patch tags: `7 - p` for patch `p`, which is neither the patch index nor the element
/// index, so a side that fetched either instead reads wrong.
fn per_patch_records() -> Vec<u8> {
    let mut bytes = Vec::new();
    for patch in 0..PATCH_COUNT {
        bytes.extend((7 - patch).to_le_bytes());
    }
    bytes
}

/// The control-point index permutation both typed sites state: index `j` names element `11 - j`, so
/// patch `p`'s control points are `(11 - 3p, 10 - 3p, 9 - 3p)` and every patch's mark moves. A
/// permutation rather than the identity is the whole point: the identity's answer is the untyped
/// draw's, and the two could then not be told apart.
fn control_point_indices(index_type: IndexType) -> Vec<u8> {
    let mut bytes = Vec::new();
    for j in 0..PATCH_COUNT * CONTROL_POINTS {
        let index = PATCH_COUNT * CONTROL_POINTS - 1 - j;
        match index_type {
            IndexType::UInt16 => bytes.extend((index as u16).to_le_bytes()),
            _ => bytes.extend(index.to_le_bytes()),
        }
    }
    bytes
}

/// The three control-point index types, in the SDK table's own declaration order.
fn index_types() -> Vec<IndexType> {
    IndexType::ALL.to_vec()
}

/// The patch fixture's vertex descriptor: the control-point `uint` at attribute 0 of buffer 0 under
/// `MTLVertexStepFunctionPerPatchControlPoint`, and the per-patch `uint` at attribute 1 of buffer 1
/// under `MTLVertexStepFunctionPerPatch`. Both are tessellation's own steps and no other case of
/// the suite sets either.
const PATCH_ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute::new(0, VertexFormat::UInt, 0, 0),
    VertexAttribute::new(1, VertexFormat::UInt, 0, 1),
];

/// The two layouts [`PATCH_ATTRIBUTES`] are fetched under.
const PATCH_LAYOUTS: [VertexBufferLayout; 2] = [
    VertexBufferLayout {
        index: 0,
        stride: 4,
        step: VertexStepFunction::PerPatchControlPoint,
        rate: 1,
    },
    VertexBufferLayout {
        index: 1,
        stride: 4,
        step: VertexStepFunction::PerPatch,
        rate: 1,
    },
];

impl PatchIndices {
    fn state(&self, index_type: IndexType) -> TessellationState {
        TessellationState::new()
            .partitioned(Partition::Integer)
            .clamped_to(SWEPT_MAX)
            .indexed_by(index_type)
    }

    fn pipeline(&self, index_type: IndexType) -> RenderPipelineSpec<'static> {
        RenderPipelineSpec::tessellating(
            StageFunction::named(PATCHES.entry),
            VertexLayout::new(&PATCH_ATTRIBUTES, &PATCH_LAYOUTS),
            self.state(index_type),
        )
    }

    /// The factor buffer: one record a patch, every factor 1, so each patch draws the one triangle
    /// its three corners make and the case costs twelve domain points in all.
    fn factors(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..PATCH_COUNT {
            bytes.extend(triangle_factors([1.0; 3], 1.0));
        }
        bytes
    }
}

impl Case for PatchIndices {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(PATCHES)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = air_inputs(PATCHES)?;
        inputs.push(Input::bytes("control-points", &control_point_records()));
        inputs.push(Input::bytes("per-patch", &per_patch_records()));
        inputs.push(Input::bytes("factors", &self.factors()));
        inputs.push(Input::bytes(
            "marks",
            &vec![0u8; index_types().len() * PATCH_REGION_BYTES as usize],
        ));
        let mut pipelines = Vec::new();
        for index_type in index_types() {
            pipelines.extend(index_type.name().as_bytes());
            pipelines.extend(self.pipeline(index_type).words());
            if index_type != IndexType::None {
                pipelines.extend(control_point_indices(index_type));
            }
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = load(PATCHES)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let points = made(
            sites,
            "buffer/control-points",
            side.buffer(&control_point_records()),
        )?;
        let per_patch = made(sites, "buffer/per-patch", side.buffer(&per_patch_records()))?;
        let factors = made(sites, "buffer/factors", side.buffer(&self.factors()))?;
        let marks = made(
            sites,
            "buffer/marks",
            side.buffer(&vec![
                0u8;
                index_types().len() * PATCH_REGION_BYTES as usize
            ]),
        )?;
        let types = index_types();
        let mut indices = Vec::with_capacity(types.len());
        for &index_type in &types {
            indices.push(match index_type {
                IndexType::None => None,
                held => Some(made(
                    sites,
                    &format!("buffer/{}", held.name()),
                    side.buffer(&control_point_indices(held)),
                )?),
            });
        }
        let mut pipelines = Vec::with_capacity(types.len());
        for &index_type in &types {
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", index_type.name()),
                side.render_pipeline(library, &self.pipeline(index_type)),
                |_| Value::Absent,
            ));
        }

        let mut work = Vec::new();
        let mut drawn = Vec::new();
        for (at, pipeline) in pipelines.iter().enumerate() {
            let Some(pipeline) = pipeline else { continue };
            drawn.push(at);
            let region = at as u32 * PATCH_REGION_BYTES;
            let mut steps = vec![
                RenderStep::Pipeline(*pipeline),
                RenderStep::TessellationFactorBuffer {
                    buffer: Some(factors),
                    offset: 0,
                    instance_stride: 0,
                },
                RenderStep::VertexBuffer {
                    buffer: points,
                    offset: 0,
                    index: 0,
                },
                RenderStep::VertexBuffer {
                    buffer: per_patch,
                    offset: 0,
                    index: 1,
                },
                RenderStep::VertexBuffer {
                    buffer: marks,
                    offset: u64::from(region),
                    index: 2,
                },
            ];
            steps.push(match indices[at] {
                None => RenderStep::DrawPatches {
                    control_points: CONTROL_POINTS,
                    first: 0,
                    count: PATCH_COUNT,
                    patch_indices: None,
                    patch_index_offset: 0,
                    instances: 1,
                    base_instance: 0,
                },
                Some(buffer) => RenderStep::DrawIndexedPatches {
                    control_points: CONTROL_POINTS,
                    first: 0,
                    count: PATCH_COUNT,
                    patch_indices: None,
                    patch_index_offset: 0,
                    control_point_indices: buffer,
                    control_point_index_offset: 0,
                    instances: 1,
                    base_instance: 0,
                },
            });
            work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps)));
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        for (at, &index_type) in types.iter().enumerate() {
            let base = at as u64 * u64::from(PATCH_REGION_BYTES);
            let bytes = call(
                sites,
                &format!("read-back/{}", index_type.name()),
                side.read_buffer(marks, base, PATCH_WORDS as usize * 4),
                |_| Value::Absent,
            )?;
            let words = to_words(&bytes);
            let mut node = Node::branch(index_type.name(), Vec::new());
            node = match index_type {
                // Under the untyped draw each patch reads its own control points and its own
                // per-patch element, so the one word a patch stored is split into the three fields
                // the three fetches produced and each field is compared at the site of the step
                // that fetched it. Every bit of Metal's answer is in exactly one of them: the word
                // is `ptag << 24 | patch << 16 | t0 << 8 | t1 << 4 | t2`, and
                // `tests::the_three_patch_fields_cover_every_bit_of_the_word` holds that.
                IndexType::None => node
                    .with(Node::leaf(
                        "MTLVertexStepFunctionPerPatchControlPoint",
                        Observed::bytes(
                            words
                                .iter()
                                .flat_map(|word| {
                                    [
                                        ((word >> 8) & 0xff) as u8,
                                        ((word >> 4) & 0xf) as u8,
                                        (word & 0xf) as u8,
                                    ]
                                })
                                .collect(),
                            patch_field_layout(base, 3),
                        ),
                    ))
                    .with(Node::leaf(
                        "MTLVertexStepFunctionPerPatch",
                        Observed::bytes(
                            words.iter().map(|word| (word >> 24) as u8).collect(),
                            patch_field_layout(base, 1),
                        ),
                    ))
                    .with(Node::leaf(
                        "patch-index",
                        Observed::bytes(
                            words.iter().map(|word| (word >> 16) as u8).collect(),
                            patch_field_layout(base, 1),
                        ),
                    )),
                // A typed draw's answer is the whole word: what the index type decides is which
                // control point each tag came from, and the word holds all three tags at once.
                _ => node.with(Node::leaf(
                    "words",
                    Observed::bytes(bytes.clone(), patch_words_layout(base)),
                )),
            };
            if let Some(pass) = drawn.iter().position(|held| *held == at) {
                if let Some(Some(failure)) = outcome.refused.get(pass) {
                    node = node.with(Node::leaf("pass", Observed::value(failure.recorded())));
                }
            }
            sites.push(node);
        }
        Ok(())
    }
}

/// The layout one field of the patch words is compared under: `width` bytes a patch.
fn patch_field_layout(offset: u64, width: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset,
            length: u64::from(PATCH_WORDS * width),
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: width,
        row_pitch: width,
        rows: PATCH_WORDS,
        image_pitch: PATCH_WORDS * width,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The layout a typed site's whole words are compared under: one word a patch.
fn patch_words_layout(offset: u64) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset,
            length: u64::from(PATCH_WORDS * 4),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 1,
        row_pitch: 4,
        rows: PATCH_WORDS,
        image_pitch: PATCH_WORDS * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
}

// ---- What the cases share ----------------------------------------------------------------------

/// The steps one patch of a domain fixture is drawn with: the pipeline, the factor buffer from its
/// start, the grid record, the marks region, and a draw of one patch of one instance.
fn one_patch_steps(
    pipeline: Handle,
    grid: Handle,
    factors: Handle,
    marks: Handle,
    region: u32,
    control_points: u32,
) -> Vec<RenderStep> {
    vec![
        RenderStep::Pipeline(pipeline),
        RenderStep::TessellationFactorBuffer {
            buffer: Some(factors),
            offset: 0,
            instance_stride: 0,
        },
        RenderStep::VertexBuffer {
            buffer: grid,
            offset: 0,
            index: 0,
        },
        RenderStep::VertexBuffer {
            buffer: marks,
            offset: u64::from(region),
            index: 1,
        },
        RenderStep::DrawPatches {
            control_points,
            first: 0,
            count: 1,
            patch_indices: None,
            patch_index_offset: 0,
            instances: 1,
            base_instance: 0,
        },
    ]
}

/// The whole mark grid of the region at `base`.
fn read_marks(
    side: &mut dyn Side,
    sites: &mut Sites,
    site: &str,
    marks: Handle,
    base: u64,
) -> Result<Vec<u8>, Abort> {
    call(
        sites,
        &format!("read-back/{site}"),
        side.read_buffer(marks, base, MARK_WORDS as usize * 4),
        |_| Value::Absent,
    )
}

/// The cells a coarse grid can reach of the region at `base`.
fn read_coarse_marks(
    side: &mut dyn Side,
    sites: &mut Sites,
    site: &str,
    marks: Handle,
    base: u64,
) -> Result<Vec<u8>, Abort> {
    call(
        sites,
        &format!("read-back/{site}"),
        side.read_buffer(marks, base, COARSE_WORDS as usize * 4),
        |_| Value::Absent,
    )
}

/// `bytes` as little-endian words, the trailing bytes of a length that is not a multiple of four
/// left out.
fn to_words(bytes: &[u8]) -> Vec<u32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().copied().map(u32::from_le_bytes).collect()
}

/// The two texts of `fixture` a case fingerprints.
fn air_inputs(fixture: Fixture) -> Result<Vec<Input>, String> {
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

/// The shader references of `fixture`, both dialects.
fn shaders_of(fixture: Fixture) -> Result<Vec<ShaderRef>, String> {
    let mut shaders = Vec::new();
    for extension in [RUNTIME_TEXT, METAL_TEXT] {
        shaders.push(ShaderRef {
            fixture: format!("{}{extension}", fixture.name),
            entry: fixture.entry.to_owned(),
            source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
        });
    }
    Ok(shaders)
}

/// `fixture` in both dialects, or the abort a case stops at where it cannot be read.
fn load(fixture: Fixture) -> Result<crate::fixture::Loaded, Abort> {
    fixture.loaded().map_err(|why| Abort {
        at: "library".to_owned(),
        failure: Failure::named("FixtureNotRead", why),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes the recording of `case` holds at the leaf `path` (node names from the root,
    /// `/`-separated, the reserved `case` root left out), or `None` where it holds no such leaf.
    fn recorded_bytes(case: &str, path: &str) -> Option<Vec<u8>> {
        let recording = crate::recording::read(case).ok()?;
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let mut node = &outcome.transcript.root;
                for name in path.split('/') {
                    node = node.child(name)?;
                }
                if let Value::Bytes(bytes) = &node.observed.value {
                    return Some(bytes.clone());
                }
            }
        }
        None
    }

    /// The cells the mark region `bytes` holds a nonzero word at.
    fn marked(bytes: &[u8]) -> usize {
        to_words(bytes)
            .into_iter()
            .filter(|word| *word != 0)
            .count()
    }

    #[test]
    fn no_two_partition_modes_answered_the_same_marks() {
        // The partition sweep's non-vacuity, off Metal's own recorded answers: four modes, four
        // different point sets, on each domain. A factor list that let two of them agree would be
        // a sweep comparing one answer twice, and this is what fails if the list is edited.
        for (case, patch_type) in [
            ("tessellation-partition-modes-quad", "MTLPatchTypeQuad"),
            (
                "tessellation-partition-modes-triangle",
                "MTLPatchTypeTriangle",
            ),
        ] {
            let answers: Vec<(Partition, Vec<u8>)> = partitions()
                .into_iter()
                .filter_map(|mode| {
                    let path = format!("{patch_type}/{}/marks", partition_site(mode));
                    recorded_bytes(case, &path).map(|bytes| (mode, bytes))
                })
                .collect();
            assert_eq!(
                answers.len(),
                partitions().len(),
                "{case}: every mode's marks are recorded under {patch_type}"
            );
            for (a, (first, left)) in answers.iter().enumerate() {
                assert!(
                    marked(left) > 0,
                    "{case}: {first:?} marked no cell at all, so it compares nothing"
                );
                for (second, right) in answers.iter().skip(a + 1) {
                    assert_ne!(
                        left, right,
                        "{case}: {first:?} and {second:?} answered the same marks, so one of the \
                         two compares nothing new"
                    );
                }
            }
        }
    }

    #[test]
    fn the_factor_clamp_is_metals_own_two_collapses() {
        // What the factor sweep measured, held as the claim the module docs make: a factor of 0
        // marks nothing, the pair around the floor agree, the pair around the maximum agree, and
        // the three answers are different from each other. Both collapses are Metal's clamp, so a
        // side that stopped clamping parts a pair this test says must stay equal, and a side that
        // stopped culling marks a region this test says must be empty.
        let at = |site: &str| {
            recorded_bytes("tessellation-edge-factors", &format!("{site}/marks"))
                .unwrap_or_else(|| panic!("the recording holds {site}"))
        };
        let (zero, below, one, max, above) = (
            at("factor-0"),
            at("factor-below-1"),
            at("factor-1"),
            at("factor-max"),
            at("factor-above-max"),
        );
        assert_eq!(marked(&zero), 0, "a factor of 0 discards the patch");
        assert_eq!(below, one, "Metal clamps a factor below 1 up to 1");
        assert_eq!(above, max, "Metal clamps a factor above the maximum to it");
        assert_eq!(marked(&one), 4, "a factor of 1 marks the domain's corners");
        assert_eq!(
            marked(&max),
            MARK_WORDS as usize,
            "a factor of 64 marks every cell of the 65-by-65 grid"
        );
    }

    /// The mark bytes the scale sweep's recording holds for `site`.
    fn scaled_marks(site: &str) -> Vec<u8> {
        let enabled = SCALE_CELLS
            .iter()
            .find(|cell| cell.site == site)
            .unwrap_or_else(|| panic!("the table holds {site}"))
            .enabled;
        let path = format!("{}/{site}/marks", FactorScales::pipeline_site(enabled));
        recorded_bytes("tessellation-factor-scales", &path)
            .unwrap_or_else(|| panic!("the recording holds {path}"))
    }

    /// The mark bytes the non-finite sweep's recording holds for `site`.
    fn non_finite_marks(site: &str) -> Vec<u8> {
        recorded_bytes("tessellation-non-finite-factors", &format!("{site}/marks"))
            .unwrap_or_else(|| panic!("the recording holds {site}"))
    }

    #[test]
    fn metal_scales_a_factor_before_it_clamps_it() {
        // What the scale sweep measured, held as the claim the module docs make. Clamping first
        // would take `scale-2-of-below-1` to a level of 2 and `scale-half-of-above-max` to one of
        // 8, so a side that moved the clamp parts both of these from Metal's answer — and leaves
        // the control alone, which is what says the pair reads the order.
        assert_eq!(
            marked(&scaled_marks("scale-2-of-below-1")),
            4,
            "0.5 scaled by 2 is the floor, so it marks the corners and not a 3-by-3 grid"
        );
        assert_eq!(
            marked(&scaled_marks("scale-half-of-above-max")),
            100,
            "17 halved under a ceiling of 16 is a level of 9, whose 10 points an axis mark 100 \
             cells; clamping first would be a level of 8 and 81 cells"
        );
        assert_eq!(
            marked(&scaled_marks("scale-half-of-2")),
            4,
            "the control: 2 halved is the floor whichever end the clamp is"
        );
    }

    #[test]
    fn metal_rounds_the_scale_to_half_before_it_multiplies() {
        // The collapse the sweep states, and the pair that says where the rounding takes a scale to
        // nothing: a scale subnormal as a `half` still draws, one that is 0 as a `half` culls.
        assert_eq!(
            scaled_marks("scale-rounds-to-2-as-half"),
            scaled_marks("scale-2"),
            "2.0005 is 2.0 as a `half`, so it draws what a scale of 2 draws"
        );
        assert_eq!(marked(&scaled_marks("scale-subnormal-as-half")), 4);
        assert_eq!(marked(&scaled_marks("scale-zero-as-half")), 0);
    }

    #[test]
    fn one_scale_metals_documentation_calls_culling_does_not_cull() {
        // Metal's documentation describes a scale of 0, a negative one, a NaN and an infinity as
        // culling. Measured on the oracle, three of the four do and `+∞` does not: it draws what
        // the maximum factor draws. A pipeline that enables the scale and never sets one culls too,
        // which is the fifth shape and is not a value at all.
        for site in [
            "scale-unset",
            "scale-0",
            "scale-negative",
            "scale-nan",
            "scale-negative-infinity",
        ] {
            assert_eq!(marked(&scaled_marks(site)), 0, "{site} culls the patch");
        }
        assert_eq!(
            marked(&scaled_marks("scale-positive-infinity")),
            MEDIUM_ROWS as usize * MEDIUM_ROWS as usize,
            "`+∞` reaches the ceiling, which marks every cell a grid of scale 16 has"
        );
    }

    #[test]
    fn the_enable_field_gates_the_setter_and_the_cull_together() {
        // `tessellationFactorScaleEnabled` false ignores the scale entirely: a scale of 2 draws
        // what the unscaled factor draws, and so does the scale of 0 that culls every patch of an
        // enabling pipeline. The last assertion is what makes the field's two values a real pair.
        let unscaled = scaled_marks("scale-1");
        assert_eq!(marked(&unscaled), 25, "factor 4 is a level of 4");
        assert_eq!(scaled_marks("scale-disabled"), unscaled);
        assert_eq!(scaled_marks("scale-0-disabled"), unscaled);
        assert_ne!(
            scaled_marks("scale-0"),
            scaled_marks("scale-0-disabled"),
            "the same scale culls under the field and is ignored without it"
        );
    }

    #[test]
    fn the_edge_factors_metal_culls_are_one_class_and_two_near_misses_are_not_in_it() {
        // Metal's first rule reads "negative, zero, or NaN", and the four values the sweep states
        // from that class answer one empty region each — a collapse that is Metal's own rule, with
        // the inside factors at 4 beside them so the cull is seen to override a healthy inside.
        // `+∞` and the least positive subnormal are the two that are **not** in the class: each
        // draws, and the three answers are all different.
        for named in ["nan", "negative", "negative-zero", "negative-infinity"] {
            let site = format!("{named}/edges");
            assert_eq!(marked(&non_finite_marks(&site)), 0, "{site}");
        }
        let control = non_finite_marks("all-finite");
        assert_eq!(marked(&control), 25, "the control is a level of 4");
        let infinite = non_finite_marks("positive-infinity/edges");
        let subnormal = non_finite_marks("subnormal/edges");
        assert_eq!(marked(&infinite), 73, "a level of 16 with the inside at 4");
        assert_eq!(marked(&subnormal), 13, "a level of 1 with the inside at 4");
        assert_ne!(infinite, control);
        assert_ne!(subnormal, control);
        assert_ne!(infinite, subnormal);
    }

    #[test]
    fn an_inside_factor_reaches_a_range_end_and_never_culls() {
        // No cull reads an inside factor, so each of the five values that is not a finite number
        // above the floor reaches the range's **lower** end and marks the inner level 1: the
        // boundary ring of 16 points and the one centre point. The five answers are equal, which
        // is Metal's clamp arriving at one place from both directions and not a hole in the sweep;
        // `+∞` reaches the other end, and the control is neither.
        let lower = non_finite_marks("nan/inside");
        assert_eq!(
            marked(&lower),
            17,
            "the inner level 1: the ring and a centre"
        );
        for named in [
            "negative",
            "negative-zero",
            "negative-infinity",
            "subnormal",
        ] {
            let site = format!("{named}/inside");
            assert_eq!(non_finite_marks(&site), lower, "{site}");
        }
        let upper = non_finite_marks("positive-infinity/inside");
        assert_eq!(
            marked(&upper),
            241,
            "the inner level 16 inside a level-4 ring"
        );
        assert_ne!(upper, lower);
        assert_ne!(non_finite_marks("all-finite"), lower);
        assert_ne!(non_finite_marks("all-finite"), upper);
        // The `Both` cells the sweep does state are a third answer each, which is what earns them
        // their place beside the one-sided pair.
        assert_eq!(marked(&non_finite_marks("positive-infinity/both")), 289);
        assert_eq!(marked(&non_finite_marks("subnormal/both")), 4);
    }

    #[test]
    fn metals_factor_address_counts_the_base_instance() {
        // The reading the control stage took from Metal's documentation with no observation behind
        // it, now measured: a draw of one instance from base instance 1 reads the record instance 1
        // reads, not instance 0's. The two per-instance steps are the ones the address's instance
        // term reaches, so both are held, and the per-patch leg beside each is the control that
        // says the move is the instance's and not the patch's.
        for step in [Step::PerInstance, Step::PerPatchAndPerInstance] {
            let at = |leg: &str| {
                recorded_bytes(
                    "tessellation-factor-steps",
                    &format!("{}/{leg}/marks", step.name()),
                )
                .unwrap_or_else(|| panic!("the recording holds {}/{leg}", step.name()))
            };
            let (first, base_one, two) =
                (at("patch-0"), at("base-instance-1"), at("two-instances"));
            assert_ne!(
                first, base_one,
                "{step:?}: base instance 1 read instance 0's record, so Metal's factor address \
                 does not count the base instance and the control stage's reading is wrong"
            );
            assert_eq!(
                base_one, two,
                "{step:?}: base instance 1 read the record a second drawn instance reads"
            );
        }
    }

    #[test]
    fn a_typed_control_point_index_answers_what_the_untyped_draw_does_not() {
        // Both widths read the case's reversing permutation, so they answer the same words; the
        // untyped draw reads each patch's own control points, so it does not. The first assertion
        // is what says the two widths are a real pair rather than two spellings of one, and the
        // second is what says the permutation reached the tessellator at all.
        let at = |index_type: IndexType| {
            recorded_bytes(
                "tessellation-patch-indices",
                &format!("{}/words", index_type.name()),
            )
        };
        let short = at(IndexType::UInt16).expect("the UInt16 words");
        let long = at(IndexType::UInt32).expect("the UInt32 words");
        assert_eq!(
            short, long,
            "the two index widths name the same permutation, so they answer the same words"
        );
        let tags = recorded_bytes(
            "tessellation-patch-indices",
            "MTLTessellationControlPointIndexTypeNone/MTLVertexStepFunctionPerPatchControlPoint",
        )
        .expect("the untyped draw's control-point tags");
        // The untyped draw's patch `p` reads tags `(3p, 3p + 1, 3p + 2)`, which the reversing
        // permutation does not give any patch.
        let expected: Vec<u8> = (0..PATCH_COUNT * CONTROL_POINTS)
            .map(|tag| tag as u8)
            .collect();
        assert_eq!(
            tags, expected,
            "the untyped draw reads each patch's own control points in order"
        );
    }

    #[test]
    fn the_half_bits_are_metals_own() {
        // Each is the bit pattern `half` holds for the value, worked out from the format's own
        // fields (one sign bit, five exponent bits of bias 15, ten significand bits).
        assert_eq!(half_bits(0.0), 0x0000);
        assert_eq!(half_bits(0.5), 0x3800);
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(2.0), 0x4000);
        assert_eq!(half_bits(2.5), 0x4100);
        assert_eq!(half_bits(3.5), 0x4300);
        assert_eq!(half_bits(4.0), 0x4400);
        assert_eq!(half_bits(4.5), 0x4480);
        assert_eq!(half_bits(5.5), 0x4580);
        assert_eq!(half_bits(8.0), 0x4800);
        assert_eq!(half_bits(64.0), 0x5400);
        assert_eq!(half_bits(65.0), 0x5410);
    }

    #[test]
    fn every_factor_the_family_states_is_exactly_a_half() {
        // The assertion inside `half_bits` is the guard; this is what runs it over the whole table
        // the cases state, so a factor added to one of them cannot be a rounding nobody noticed.
        for factor in SWEPT_EDGES.iter().chain(SWEPT_INSIDE.iter()) {
            let _ = half_bits(*factor);
        }
        for (_, factor) in FACTOR_SITES {
            let _ = half_bits(factor);
        }
        for factor in STEP_FACTORS {
            let _ = half_bits(factor);
        }
    }

    #[test]
    fn the_three_patch_fields_cover_every_bit_of_the_word() {
        // The fixture stores `ptag << 24 | patch << 16 | t0 << 8 | t1 << 4 | t2`, and the untyped
        // site splits it into the per-patch byte, the patch-index byte and the three tags. No bit
        // of a word the case's own tables can produce lies outside those fields, so the split
        // drops nothing and compares nothing twice.
        for patch in 0..PATCH_COUNT {
            let ptag = 7 - patch;
            let (t0, t1, t2) = (3 * patch, 3 * patch + 1, 3 * patch + 2);
            let word = (ptag << 24) | (patch << 16) | (t0 << 8) | (t1 << 4) | t2;
            let fields =
                (((word >> 24) & 0xff) << 24) | (((word >> 16) & 0xff) << 16) | (word & 0xfff);
            assert_eq!(word, fields, "patch {patch}");
        }
    }

    /// Which factor record `step` reads for `leg`, by the addressing the control stage's module
    /// docs state: `offset + position · size` for a per-patch step, `+ instance · instanceStride`
    /// for a per-instance one. `counts_base` says whether the instance index includes the base
    /// instance, which is the open reading the case measures, so the separation is checked **both**
    /// ways round and does not rest on it.
    ///
    /// The answer is a set of records, because a draw of two instances marks the union of theirs.
    fn records_read(step: Step, leg: Leg, counts_base: bool) -> std::collections::BTreeSet<u32> {
        let patch = match step {
            Step::PerPatch | Step::PerPatchAndPerInstance => leg.patch,
            _ => 0,
        };
        let mut read = std::collections::BTreeSet::new();
        for drawn in 0..leg.instances {
            let instance = match (step, counts_base) {
                (Step::PerInstance | Step::PerPatchAndPerInstance, true) => {
                    drawn + leg.base_instance
                }
                (Step::PerInstance | Step::PerPatchAndPerInstance, false) => drawn,
                _ => 0,
            };
            let address = patch * QUAD_FACTOR_BYTES + instance * STEP_INSTANCE_STRIDE;
            read.insert(address / QUAD_FACTOR_BYTES);
        }
        read
    }

    #[test]
    fn no_two_step_functions_can_answer_the_same_legs() {
        // The sweep's non-vacuity, argued from Metal's addressing rather than read off a
        // recording: no two of the four step functions read the same records over the four legs,
        // whichever way the base-instance reading falls. A leg list that lost that property would
        // be a sweep comparing one answer twice, and this is what fails if one is edited.
        for counts_base in [false, true] {
            let answers: Vec<Vec<_>> = steps()
                .into_iter()
                .map(|step| {
                    STEP_LEGS
                        .iter()
                        .map(|leg| records_read(step, *leg, counts_base))
                        .collect()
                })
                .collect();
            for (a, first) in answers.iter().enumerate() {
                for (b, second) in answers.iter().enumerate().skip(a + 1) {
                    assert_ne!(
                        first,
                        second,
                        "{:?} and {:?} answer the same legs with counts_base = {counts_base}",
                        steps()[a],
                        steps()[b]
                    );
                }
            }
        }
    }

    #[test]
    fn the_step_sweeps_factor_records_are_nested_point_sets() {
        // A leg that draws two instances marks the union of the two records' point sets, and the
        // leg reads as a single record only because the records are nested. Under `integer`
        // partitioning a factor is the segment count, so record `k`'s points are record `k + 1`'s
        // exactly when each factor divides the next.
        for pair in STEP_FACTORS.windows(2) {
            let (small, large) = (pair[0], pair[1]);
            assert_eq!(
                large % small,
                0.0,
                "factor {small} must divide {large} or the two instances' marks would not nest"
            );
        }
    }

    #[test]
    fn a_coarse_region_holds_every_cell_its_grid_can_reach() {
        // A domain coordinate is in `[0, 1]`, so `rint(u * 8)` is in `0..=8` and the largest cell
        // a coarse draw can mark is `8 * 65 + 8`. The compared prefix is one word longer than that,
        // so no mark the draw could make lies outside it.
        assert_eq!(COARSE_WORDS, 8 * AXIS + 9);
        assert_eq!(COARSE_WORDS - 1, 8 * AXIS + 8);
        // The prefix is inside the mark grid, so a coarse leg reads marks and never a stored
        // coordinate. Both are constants, so the check belongs in a const block: clippy refuses a
        // runtime assertion whose value the compiler already knows.
        const { assert!(COARSE_WORDS <= MARK_WORDS) };
    }

    #[test]
    fn every_region_offset_is_a_vertex_binding_offset_metal_takes() {
        // Metal requires a vertex buffer binding's offset to be a multiple of 4.
        assert_eq!(REGION_BYTES % 4, 0);
        assert_eq!(PATCH_REGION_BYTES % 4, 0);
        assert_eq!(QUAD_FACTOR_BYTES % 4, 0);
        assert_eq!(STEP_INSTANCE_STRIDE % 4, 0);
    }

    #[test]
    fn the_case_names_are_the_familys_own() {
        let names: Vec<&str> = boxed().iter().map(|case| case.name()).collect();
        assert_eq!(
            names,
            vec![
                "tessellation-partition-modes-quad",
                "tessellation-partition-modes-triangle",
                "tessellation-edge-factors",
                "tessellation-non-finite-factors",
                "tessellation-factor-steps",
                "tessellation-factor-scales",
                "tessellation-patch-indices",
            ]
        );
    }

    #[test]
    fn a_medium_region_holds_every_cell_its_grid_can_reach() {
        // A domain coordinate is in `[0, 1]`, so `rint(v * 16)` is in `0..=16` and the largest grid
        // row a medium draw can mark is row 16. The compared prefix is whole rows up to it, so no
        // mark the draw could make lies outside it, and it lies inside the mark grid, so a medium
        // leg reads marks and never a stored coordinate.
        assert_eq!(MEDIUM_ROWS, MEDIUM_GRID as u32 + 1);
        assert_eq!(MEDIUM_WORDS, MEDIUM_ROWS * AXIS);
        const { assert!(MEDIUM_WORDS <= MARK_WORDS) };
        // The compared layout covers the region exactly: whole rows, no byte read twice and none
        // left out.
        assert_eq!(MEDIUM_ROWS * AXIS * 4, MEDIUM_WORDS * 4);
    }

    #[test]
    fn the_non_finite_bit_patterns_are_the_halves_they_are_named_for() {
        // Each is read back through `half`'s own fields — one sign bit, five exponent bits of bias
        // 15, ten significand bits — so a table entry that is not the value its name says fails
        // here rather than in a recording nobody can read.
        let class = |bits: u16| {
            let sign = bits >> 15;
            let exponent = (bits >> 10) & 0x1f;
            let significand = bits & 0x3ff;
            (sign, exponent, significand)
        };
        assert_eq!(class(FINITE), (0, 17, 0), "4.0 is 2^2 · 1.0");
        let at = |name: &str| {
            NON_FINITE
                .iter()
                .find(|(held, _)| *held == name)
                .map(|(_, bits)| *bits)
                .unwrap_or_else(|| panic!("the table holds {name}"))
        };
        // A NaN: the maximum exponent with a non-zero significand.
        let (sign, exponent, significand) = class(at("nan"));
        assert_eq!((sign, exponent), (0, 0x1f));
        assert_ne!(significand, 0);
        assert_eq!(class(at("negative")), (1, 15, 0), "-1.0");
        assert_eq!(class(at("negative-zero")), (1, 0, 0), "-0.0");
        assert_eq!(class(at("positive-infinity")), (0, 0x1f, 0));
        assert_eq!(class(at("negative-infinity")), (1, 0x1f, 0));
        assert_eq!(
            class(at("subnormal")),
            (0, 0, 1),
            "the least positive subnormal"
        );
    }

    #[test]
    fn metals_cull_rule_classifies_the_non_finite_table_as_its_words_do() {
        // Metal's first rule is "negative, zero, or NaN", and `culls_an_edge` is the bit test the
        // sweep's `Both` placements are left out by, so the two are held against each other over
        // the table rather than only in prose.
        for (name, bits) in NON_FINITE {
            let culls = culls_an_edge(bits);
            let expected = matches!(
                name,
                "nan" | "negative" | "negative-zero" | "negative-infinity"
            );
            assert_eq!(culls, expected, "{name}");
        }
        // And over every `half` there is, so the test is the rule and not six examples: a pattern
        // culls exactly where it is a NaN, a negative number or a zero.
        for bits in 0u16..=0xffff {
            let sign = bits >> 15 == 1;
            let exponent = (bits >> 10) & 0x1f;
            let significand = bits & 0x3ff;
            let nan = exponent == 0x1f && significand != 0;
            let zero = bits & 0x7fff == 0;
            assert_eq!(culls_an_edge(bits), sign || zero || nan, "{bits:#06x}");
        }
    }

    #[test]
    fn every_non_finite_cell_asks_a_record_no_other_cell_asks() {
        // The sweep's non-vacuity at the input end: fifteen cells, fifteen different factor
        // records. Two cells stating one record would compare one answer twice, and the `Both`
        // placements the table leaves out are exactly the ones that would.
        let cells = non_finite_cells();
        assert_eq!(cells.len(), 15);
        let named = |cell: &NonFiniteCell| match cell.placement {
            None => cell.value.to_owned(),
            Some(placement) => format!("{}/{}", cell.value, placement.site()),
        };
        for (a, first) in cells.iter().enumerate() {
            for second in cells.iter().skip(a + 1) {
                assert_ne!(
                    first.record,
                    second.record,
                    "{} and {} state one record",
                    named(first),
                    named(second)
                );
            }
        }
    }

    #[test]
    fn every_scale_cell_asks_a_question_no_other_cell_asks() {
        // The same at the scale sweep's input end: no two cells state the same pipeline, record
        // and scale. The scale is compared by bits, so the two NaN-free zeroes — no scale set and
        // a scale of 0 — are two cells and not one.
        for (a, first) in SCALE_CELLS.iter().enumerate() {
            for second in SCALE_CELLS.iter().skip(a + 1) {
                assert_ne!(
                    (first.enabled, first.record, first.scale),
                    (second.enabled, second.record, second.scale),
                    "{} and {} state one question",
                    first.site,
                    second.site
                );
                assert_ne!(first.site, second.site);
            }
        }
    }

    #[test]
    fn the_scale_cells_order_cells_part_under_one_order_and_not_the_other() {
        // The three cells that read the order of the scale and the clamp, argued from the
        // arithmetic rather than off a recording: two of them answer different levels depending on
        // which comes first, and the third answers the same level either way. A table edited so
        // that no cell parted would be a sweep that could not see the order at all, and a table
        // whose control parted would not be a control.
        let max = MEDIUM_MAX as f32;
        // Scaled before the clamp, as the control stage does it, and an `integer` level rounded up.
        let before = |factor: f32, scale: f32| (factor * scale).clamp(1.0, max).ceil();
        // Clamped first, which is the falsification's order.
        let after = |factor: f32, scale: f32| (factor.clamp(1.0, max) * scale).ceil();
        let parts = |site: &str| {
            let cell = SCALE_CELLS
                .iter()
                .find(|cell| cell.site == site)
                .unwrap_or_else(|| panic!("the table holds {site}"));
            let factor = SCALE_FACTORS[cell.record];
            let scale = f32::from_bits(cell.scale.expect("the cell sets a scale"));
            before(factor, scale) != after(factor, scale)
        };
        assert!(parts("scale-2-of-below-1"), "0.5 scaled by 2");
        assert!(parts("scale-half-of-above-max"), "17 halved under 16");
        assert!(
            !parts("scale-half-of-2"),
            "2 halved is 1 whichever end the clamp is"
        );
    }

    #[test]
    fn the_scale_sweep_reaches_both_values_of_the_enable_field_and_every_scale_class() {
        // Every value of `tessellationFactorScaleEnabled` is drawn, and the scale axis reaches
        // each class Metal's `half` rounding puts a scale in. Each membership is argued from
        // `half`'s own boundaries rather than through a conversion this test would have to own:
        // the least normal is `2^-14`, the least subnormal `2^-24`, and the gap above 2.0 is
        // `2^-9`.
        assert!(SCALE_CELLS.iter().any(|cell| cell.enabled));
        assert!(SCALE_CELLS.iter().any(|cell| !cell.enabled));
        let scales: Vec<Option<u32>> = SCALE_CELLS.iter().map(|cell| cell.scale).collect();
        assert!(scales.contains(&None), "a pass that sets no scale");
        assert!(scales.contains(&Some(NAN_BITS)));
        assert!(scales.contains(&Some(POSITIVE_INFINITY_BITS)));
        assert!(scales.contains(&Some(NEGATIVE_INFINITY_BITS)));
        assert!(scales.contains(&scale_bits(0.0)));
        assert!(scales.contains(&scale_bits(-1.0)));
        let at = |site: &str| {
            let cell = SCALE_CELLS
                .iter()
                .find(|cell| cell.site == site)
                .unwrap_or_else(|| panic!("the table holds {site}"));
            f32::from_bits(cell.scale.expect("the cell sets a scale"))
        };
        // The collapse `scale-rounds-to-2-as-half` states: the number is nearer 2.0 than the next
        // `half` above it, so Metal's rounding takes it to 2.0 and its answer is `scale-2`'s.
        let rounding = at("scale-rounds-to-2-as-half");
        assert!(rounding > 2.0, "a third number and not 2.0 itself");
        assert!(
            rounding - 2.0 < 2f32.powi(-9) / 2.0,
            "nearer 2.0 than the `half` above it"
        );
        // Subnormal as a `half`: below the least normal and above the least subnormal.
        let subnormal = at("scale-subnormal-as-half");
        assert!(subnormal < 2f32.powi(-14) && subnormal > 2f32.powi(-24));
        // 0 as a `half`: below half the least subnormal, so nearest-even rounds it to zero.
        assert!(at("scale-zero-as-half") < 2f32.powi(-25));
    }
}
