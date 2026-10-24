//! Tessellation's option space **crossed**: what one value of one tessellation enumeration answers
//! is a function of what the others hold, and these cases are the cells where that is readable.
//!
//! `tessellation_sweeps` put every value of `MTLTessellationPartitionMode`,
//! `MTLTessellationFactorStepFunction`, `MTLTessellationControlPointIndexType` and `MTLPatchType`
//! to Metal, each at one setting of the others: the four partition modes at one factor list, the
//! factor list under one mode, the scale under one mode, the degenerate factors under one mode, and
//! each domain in its own case. Every one of those answers is exact, and none of them can say
//! whether a mode's rule and a factor's value decide an answer **together**. They do, and these
//! five cases are where:
//!
//! | case | the cross |
//! |---|---|
//! | `tessellation-partition-factors` | the four modes crossed with eight factor values |
//! | `tessellation-partition-degenerates` | the four modes crossed with the factors Metal's cull and clamp read as something other than a positive number |
//! | `tessellation-partition-insides-quad`, `-triangle` | `MTLPatchType` crossed with the four modes and with each inside factor slot moved alone |
//! | `tessellation-partition-scales` | the four modes crossed with `setTessellationFactorScale:` |
//! | `tessellation-partition-steps` | the four modes crossed with the two factor step functions that move the factor address |
//!
//! # The observable, and why these cases share it
//!
//! Every case here draws one patch a pass of a committed domain fixture (`k7_tess_quad_domain`,
//! `k7_tess_triangle_domain`) and compares the **mark grid** the fixture's post-tessellation vertex
//! function writes: a mark at `marks[j · 65 + i]` for each domain point the tessellator called it
//! with, `i = rint(u · 16)` and `j = rint(v · 16)`. `tessellation_sweeps`' module docs state why a
//! mark set is the comparable answer and the coordinates it stores beside it are not, and this
//! family's comparison is that one, over those fixtures, through that module's own helpers.
//!
//! The grid scale is 16 and `maxTessellationFactor` is 16 in every case here, so a seventeen-cell
//! axis resolves the uniform levels 1 to 16 to distinct cell sets and no case can ask the
//! tessellator for a point set larger than a 17-by-17 grid. **A marked-cell count is not a level**
//! under the two fractional spacings: their transition segments place points off the uniform grid,
//! and two of them can round into one cell. The counts below are therefore reported as counts, and
//! every claim about them is a claim about two recorded cells being equal or unequal, which is what
//! the comparison itself reads.
//!
//! # What the recorded answers settle
//!
//! Metal's own, from the committed recordings on an Apple M3 Max under macOS 27.0.1.
//!
//! - **Which modes agree is the factor's to decide.** Marked cells over the eight factors, in the
//!   SDK's mode order: `pow2` 4, 4, 9, 25, 25, 25, 289, 289; `integer` 4, 4, 9, 16, 16, 25, 256,
//!   289; `fractionalOdd` 4, 4, 16, 16, 16, 36, 256, 256; `fractionalEven` 9, 9, 9, 25, 25, 25,
//!   225, 289. Five of the six pairs of modes **agree at one of those factors and part at
//!   another**, which no sweep over one factor list and no sweep under one mode can see
//!   (`tests::the_modes_agree_at_one_factor_and_part_at_another`). The sixth pair, the two
//!   fractional spacings, parts at every factor stated here.
//! - **Two of those partings are the clamp range's ends, and they are the mode's own.** At factor
//!   17, above the ceiling of 16, `fractionalOdd` answers 256 cells where the other three answer
//!   289; at factor 1, `fractionalEven` answers 9 where the other three answer 4. At factor 15 and
//!   at factor 2 those same two pairs agree exactly.
//! - **Metal's cull runs before any mode's range.** An edge factor of 0 marks **nothing** under all
//!   four modes, `fractionalEven` included, so the cull is not a clamp that happens to reach zero
//!   (`tests::a_zero_edge_factor_culls_under_every_partition_mode`).
//! - **A degenerate factor the cull does not read is clamped under the mode's own rule.** A zero
//!   and a NaN inside factor answer the **same** cell set under every mode, which is the clamp's
//!   lower end reached from both directions, and an infinite one the upper end; on each of those
//!   four cells `fractionalOdd` answers a different set from the other three, which agree. So what
//!   a degenerate factor becomes is the pair's: nothing where an edge cull reads it, and the
//!   mode's range end where it does not.
//! - **The second inside factor is the quad's and not the triangle's.** Moving the second slot from
//!   2 to 5 moves every quad cell under every mode and moves **no** triangle cell under any mode;
//!   moving the first slot moves both (`tests::the_second_inside_factor_is_the_quads_alone`). That
//!   is `MTLPatchType` deciding how many of the inside factors the mode's rule is applied to, which
//!   neither a patch type on its own nor an inside factor on its own states.
//! - **What a mode does with the half a scale produced is the pair's.** Under `integer` and under
//!   `fractionalEven`, a scaled factor that is a whole number answers the same cell set — 81 cells
//!   for a product of 8 and 121 for a product of 10 — and a product that is not answers different
//!   ones: 100 against 81 at 8.5, and 16 against 25 at 2.5. `setTessellationFactorScale:` is the
//!   only way to hand a partition rule a factor a `half` factor buffer could not have held, and
//!   `tessellation_sweeps` measured it under `integer` alone.
//! - **The factor step function and the partition mode are independent.** A per-patch step reading
//!   record 1 and a per-instance step whose two instances read records 0 and 2 both answer exactly
//!   what the factor cross answers for the factor their address reaches, under all four modes
//!   (`tests::the_step_function_and_the_partition_mode_are_independent`). A cross whose cells are
//!   the product of its axes is a result and not a gap, and it is the one pair here that is not
//!   declared in the crossings ledger.
//!
//! # Why no case here can hang the device
//!
//! One patch a pass, every ceiling at 16, every grid scale at 16. The largest point set any pass
//! asks for is a quad at level 16: 289 domain points and 512 triangles, with rasterization disabled
//! and no fragment stage. The degenerate factors ask for **less** work than the control, each
//! either culling the patch or reaching the ceiling the pipeline states, and the scales are four
//! finite numbers whose products the same ceiling bounds.

use crate::allowance::Allowance;
use crate::case::{call, made, probe, Abort, Case, Handle, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{RenderPass, RenderStep, RenderWork, TessellationState};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::{
    PatchType, TessellationFactorStepFunction as Step, TessellationPartitionMode as Partition,
};
use metal2vulkan_runtime::Degradation;

use super::tessellation_sweeps::{
    air_inputs, domain_pipeline, grid_record, load, medium_layout, quad_factors, raw_quad_factors,
    read_medium_marks, shaders_of, tessellation_notices, Domain, FINITE, MEDIUM_GRID, MEDIUM_MAX,
    QUAD_FACTOR_BYTES, REGION_BYTES,
};

// ---- The vocabulary every case of the family states its cells in ------------------------------

/// Every `MTLTessellationPartitionMode`, named one by one in the SDK table's own declaration
/// order: a cross's text has to state the values it varies, and `tests::the_four_modes_are_every
/// _partition_mode` holds the list against the generated one.
const PARTITIONS: [Partition; 4] = [
    Partition::Pow2,
    Partition::Integer,
    Partition::FractionalOdd,
    Partition::FractionalEven,
];

/// [`PARTITIONS`] as a list.
fn partitions() -> Vec<Partition> {
    PARTITIONS.to_vec()
}

/// The `MTLPatchType` a domain fixture's `[[patch(...)]]` declares, which is the site an inside
/// cross hangs a domain's answers at.
fn patch_type(domain: Domain) -> PatchType {
    match domain {
        Domain::Quad => PatchType::Quad,
        Domain::Triangle => PatchType::Triangle,
    }
}

/// One pass of a cross: where its answer hangs in the transcript, which pipeline it draws with,
/// and the draw's own parameters.
struct Leg {
    /// The nodes the leg's answer hangs under, from the case's root outwards. The last of them
    /// holds the leg's own leaves, so two legs that share a prefix share those branches and the
    /// tree says which axis each level is.
    path: Vec<&'static str>,
    /// Which of the cross's pipelines the leg draws with, so a cell names the pipeline whose
    /// tessellation state is one of its axes.
    pipeline: usize,
    /// `setTessellationFactorBuffer:`'s offset: the record a constant or per-instance step reads,
    /// and the base a per-patch step adds its patch's stride to.
    offset: u32,
    /// `setTessellationFactorBuffer:`'s instance stride.
    instance_stride: u32,
    /// The `f32` bits of the scale the leg sets, or `None` for a leg that makes no such call.
    scale: Option<u32>,
    /// `patchStart`.
    first: u32,
    /// `instanceCount`.
    instances: u32,
}

impl Leg {
    /// A leg of one patch of one instance that reads the factor record at `offset`.
    fn plain(path: Vec<&'static str>, pipeline: usize, offset: u32) -> Self {
        Leg {
            path,
            pipeline,
            offset,
            instance_stride: 0,
            scale: None,
            first: 0,
            instances: 1,
        }
    }

    /// The leg with the scale `bits`.
    fn scaled(mut self, bits: u32) -> Self {
        self.scale = Some(bits);
        self
    }

    /// The leg that draws patch `first` of one instance.
    fn at_patch(mut self, first: u32) -> Self {
        self.first = first;
        self
    }

    /// The leg that draws `instances` instances at the factor buffer's `instance_stride`.
    fn over_instances(mut self, instances: u32, instance_stride: u32) -> Self {
        self.instances = instances;
        self.instance_stride = instance_stride;
        self
    }
}

/// A cross one case draws: the domain fixture, the pipelines whose states are one axis, the factor
/// records and per-patch grid records the draws read, and one leg a cell.
struct Cross {
    domain: Domain,
    /// The pipelines, each with the site it is made at: the enumerator names of the values the
    /// pipeline's state holds, so the depth census credits each value where it ran and no two
    /// pipelines of one cross share a site.
    pipelines: Vec<(String, TessellationState)>,
    /// The factor buffer, laid out as the legs' offsets address it.
    factors: Vec<u8>,
    /// The per-patch grid records, one a patch a draw can reach.
    grids: Vec<u8>,
    legs: Vec<Leg>,
}

impl Cross {
    /// The marks buffer: one [`REGION_BYTES`] region a leg.
    fn marks(&self) -> Vec<u8> {
        vec![0u8; self.legs.len() * REGION_BYTES as usize]
    }

    /// The byte a leg's mark region starts at.
    fn region(&self, at: usize) -> u64 {
        at as u64 * u64::from(REGION_BYTES)
    }

    /// The pipeline words every site of the cross states, each behind the site's own name, which
    /// is what puts a pipeline's whole tessellation state in the case's input fingerprint.
    fn pipeline_words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (site, state) in &self.pipelines {
            bytes.extend(site.as_bytes());
            bytes.extend(domain_pipeline(self.domain.fixture().entry, *state).words());
        }
        bytes
    }

    /// The legs' own table, so a leg whose draw or site moved moves the case's key: the inputs
    /// hold the pass words too, and this is the half that names the cells.
    fn leg_words(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for leg in &self.legs {
            for name in &leg.path {
                bytes.extend(name.as_bytes());
                bytes.push(b'/');
            }
            bytes.extend((leg.pipeline as u32).to_le_bytes());
            bytes.extend(leg.offset.to_le_bytes());
            bytes.extend(leg.instance_stride.to_le_bytes());
            bytes.extend(leg.scale.unwrap_or(0).to_le_bytes());
            bytes.extend(leg.first.to_le_bytes());
            bytes.extend(leg.instances.to_le_bytes());
        }
        bytes
    }

    /// The inputs a case of this family states.
    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = air_inputs(self.domain.fixture())?;
        inputs.push(Input::bytes("grids", &self.grids));
        inputs.push(Input::bytes("factors", &self.factors));
        inputs.push(Input::bytes("marks", &self.marks()));
        inputs.push(Input::bytes("pipelines", &self.pipeline_words()));
        inputs.push(Input::bytes("legs", &self.leg_words()));
        Ok(inputs)
    }

    /// The steps one leg's pass holds.
    fn steps(&self, leg: &Leg, pipeline: Handle, held: &Bound, at: usize) -> Vec<RenderStep> {
        let mut steps = vec![
            RenderStep::Pipeline(pipeline),
            RenderStep::TessellationFactorBuffer {
                buffer: Some(held.factors),
                offset: u64::from(leg.offset),
                instance_stride: u64::from(leg.instance_stride),
            },
            RenderStep::VertexBuffer {
                buffer: held.grids,
                offset: 0,
                index: 0,
            },
            RenderStep::VertexBuffer {
                buffer: held.marks,
                offset: self.region(at),
                index: 1,
            },
        ];
        if let Some(bits) = leg.scale {
            steps.push(RenderStep::TessellationFactorScale { bits });
        }
        steps.push(RenderStep::DrawPatches {
            control_points: 1,
            first: leg.first,
            count: 1,
            patch_indices: None,
            patch_index_offset: 0,
            instances: leg.instances,
            base_instance: 0,
        });
        steps
    }

    /// Make everything the cross needs, encode one pass a leg, read each leg's mark region back
    /// and emit the tree the legs' paths name.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = self.domain.fixture();
        let loaded = load(fixture)?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let held = Bound {
            grids: made(sites, "buffer/grids", side.buffer(&self.grids))?,
            factors: made(sites, "buffer/factors", side.buffer(&self.factors))?,
            marks: made(sites, "buffer/marks", side.buffer(&self.marks()))?,
        };
        let mut pipelines = Vec::with_capacity(self.pipelines.len());
        for (site, state) in &self.pipelines {
            pipelines.push(probe(
                sites,
                &format!("{site}/pipeline"),
                side.render_pipeline(library, &domain_pipeline(fixture.entry, *state)),
                |_| Value::Absent,
            ));
        }

        let mut work = Vec::new();
        let mut encoded = Vec::new();
        for (at, leg) in self.legs.iter().enumerate() {
            let Some(pipeline) = pipelines[leg.pipeline] else {
                continue;
            };
            encoded.push(at);
            work.push(RenderWork::Pass(
                RenderPass::sized([1, 1]).drawing(self.steps(leg, pipeline, &held, at)),
            ));
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;

        let mut answered = Vec::with_capacity(self.legs.len());
        for (at, leg) in self.legs.iter().enumerate() {
            let site = leg.path.join("/");
            let region = read_medium_marks(side, sites, &site, held.marks, self.region(at))?;
            let mut leaves = vec![Node::leaf(
                "marks",
                Observed::bytes(region, medium_layout(self.region(at))),
            )];
            if let Some(pass) = encoded.iter().position(|held| *held == at) {
                if let Some(Some(failure)) = outcome.refused.get(pass) {
                    leaves.push(Node::leaf("pass", Observed::value(failure.recorded())));
                }
            }
            answered.push((leg.path.clone(), leaves));
        }
        for node in grouped(0, &answered) {
            sites.push(node);
        }
        Ok(())
    }
}

/// What one leg's pass binds.
struct Bound {
    grids: Handle,
    factors: Handle,
    marks: Handle,
}

/// The branches of `answered`, which all share their paths' first `level` names, grouped by the
/// name at `level`: a leg whose path ends there holds its own leaves, and one that goes deeper
/// hangs under a branch the legs beside it share.
fn grouped(level: usize, answered: &[(Vec<&'static str>, Vec<Node>)]) -> Vec<Node> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < answered.len() {
        let name = answered[at].0[level];
        let mut end = at;
        while end < answered.len() && answered[end].0[level] == name {
            end += 1;
        }
        let group = &answered[at..end];
        let children = match level + 1 == group[0].0.len() {
            true => group[0].1.clone(),
            false => grouped(level + 1, group),
        };
        out.push(Node::branch(name, children));
        at = end;
    }
    out
}

/// The tessellation state one partition mode's pipeline states, under the family's one ceiling.
fn state(mode: Partition) -> TessellationState {
    TessellationState::new()
        .partitioned(mode)
        .clamped_to(MEDIUM_MAX)
}

/// The pipelines a case whose one axis is the partition mode makes: one a mode, at the mode's own
/// enumerator name.
fn partition_pipelines() -> Vec<(String, TessellationState)> {
    partitions()
        .into_iter()
        .map(|mode| (mode.name().to_owned(), state(mode)))
        .collect()
}

// ---- tessellation-partition-factors -----------------------------------------------------------

/// The factor values the mode-by-factor cross states, each on every edge and inside factor of the
/// patch, with the site its answers hang at.
///
/// Eight numbers, each there for a cell where two of the modes agree or part:
///
/// - **0.5**, below every mode's floor, which each mode clamps **up** to its own: 1 for three of
///   them and 2 for `fractionalEven`.
/// - **1**, `integer`'s and `fractionalOdd`'s floor and below `fractionalEven`'s.
/// - **2**, an even whole number: `integer` and `fractionalEven` leave it, `fractionalOdd` rounds
///   it up to 3, and `pow2` leaves it because it is a power of two.
/// - **3**, an odd whole number, where the two fractional modes part the other way: `fractionalOdd`
///   leaves it and `fractionalEven` rounds it up to 4, which is also `pow2`'s answer.
/// - **2.5** and **3.5**, two fractions, where each mode's rounding **and** its placement of the
///   transition segments answer. These are the only two factors the mode sweep of
///   `tessellation_sweeps` could ask about, and it asked them of one list of four.
/// - **15**, below the ceiling and `fractionalOdd`'s own range end.
/// - **17**, above the ceiling of 16, which `fractionalOdd` clamps **down** to 15 and the other
///   three to 16.
const FACTOR_CELLS: [(&str, f32); 8] = [
    ("factor-below-floor", 0.5),
    ("factor-1", 1.0),
    ("factor-2", 2.0),
    ("factor-3", 3.0),
    ("factor-2-5", 2.5),
    ("factor-3-5", 3.5),
    ("factor-15", 15.0),
    ("factor-above-max", 17.0),
];

/// The partition modes crossed with the factor value.
pub struct PartitionFactors;

impl PartitionFactors {
    const NAME: &'static str = "tessellation-partition-factors";

    fn cross() -> Cross {
        let mut factors = Vec::new();
        for (_, factor) in FACTOR_CELLS {
            factors.extend(quad_factors([factor; 4], [factor; 2]));
        }
        let mut legs = Vec::new();
        for (mode, held) in partitions().into_iter().enumerate() {
            for (cell, (site, _)) in FACTOR_CELLS.iter().enumerate() {
                legs.push(Leg::plain(
                    vec![held.name(), site],
                    mode,
                    cell as u32 * QUAD_FACTOR_BYTES,
                ));
            }
        }
        Cross {
            domain: Domain::Quad,
            pipelines: partition_pipelines(),
            factors,
            grids: grid_record(MEDIUM_GRID),
            legs,
        }
    }
}

impl Case for PartitionFactors {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(Domain::Quad.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::cross().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::cross().run(side, sites)
    }
}

// ---- tessellation-partition-degenerates -------------------------------------------------------

/// The records the mode-by-degenerate cross states: the site, the `half` bits of the four edge
/// factors, and the `half` bits of the two inside ones.
///
/// Metal reads a factor that is not a positive number by two rules, and the partition mode meets
/// them differently: an **edge** factor that is negative, zero or NaN culls the patch, and every
/// other such factor is clamped to a range. So the cross is five records, each in the one placement
/// that makes its rule readable, with the factor the rule does not read pinned at [`FINITE`]:
///
/// - `edges-zero` is the cull, asked of each mode. It marks nothing under all four, which is what
///   says the cull is not a clamp that reaches a range's lower end.
/// - `inside-zero` and `inside-nan` are the clamp's lower end, which no cull reads. They answer the
///   **same** cell set as each other under every mode, and `fractionalOdd`'s set is not the other
///   three's.
/// - `inside-infinite` is the clamp's upper end on an inside factor, and `edges-infinite` the same
///   on the edges, which Metal's cull does **not** read. The same three modes agree and the same
///   one parts.
const DEGENERATE_CELLS: [(&str, u16, u16); 5] = [
    ("edges-zero", 0x0000, FINITE),
    ("inside-zero", FINITE, 0x0000),
    ("inside-nan", FINITE, 0x7e00),
    ("inside-infinite", FINITE, 0x7c00),
    ("edges-infinite", 0x7c00, FINITE),
];

/// The partition modes crossed with the factors Metal's cull and clamp read as something other
/// than a positive number.
pub struct PartitionDegenerates;

impl PartitionDegenerates {
    const NAME: &'static str = "tessellation-partition-degenerates";

    fn cross() -> Cross {
        let mut factors = Vec::new();
        for (_, edges, inside) in DEGENERATE_CELLS {
            factors.extend(raw_quad_factors([edges; 4], [inside; 2]));
        }
        let mut legs = Vec::new();
        for (mode, held) in partitions().into_iter().enumerate() {
            for (cell, (site, _, _)) in DEGENERATE_CELLS.iter().enumerate() {
                legs.push(Leg::plain(
                    vec![held.name(), site],
                    mode,
                    cell as u32 * QUAD_FACTOR_BYTES,
                ));
            }
        }
        Cross {
            domain: Domain::Quad,
            pipelines: partition_pipelines(),
            factors,
            grids: grid_record(MEDIUM_GRID),
            legs,
        }
    }
}

impl Case for PartitionDegenerates {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        tessellation_notices()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(Domain::Quad.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::cross().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::cross().run(side, sites)
    }
}

// ---- tessellation-partition-insides -----------------------------------------------------------

/// The edge factor every cell of the inside cross pins, so the boundary is the same under every
/// mode and a difference in a mark is an inside factor's.
const PINNED_EDGE: f32 = 4.0;

/// The inside factor pairs the cross states, with the site each hangs at.
///
/// Four records over two numbers, laid out so that **each inside slot is moved alone**:
/// `inside-2-and-2` and `inside-2-and-5` differ in the second slot only, and `inside-2-and-2` and
/// `inside-5-and-2` in the first. A quad's function reads both slots and a triangle's reads the
/// first, so the pair that moves the second slot is two answers on the quad and **one** answer on
/// the triangle. The fourth record is a pair of fractions, where each mode's rounding of the two
/// slots is asked at once.
const INSIDE_CELLS: [(&str, [f32; 2]); 4] = [
    ("inside-2-and-2", [2.0, 2.0]),
    ("inside-2-and-5", [2.0, 5.0]),
    ("inside-5-and-2", [5.0, 2.0]),
    ("inside-2-5-and-3-5", [2.5, 3.5]),
];

/// `MTLPatchType` crossed with the partition mode and the inside factors.
pub struct PartitionInsides {
    name: &'static str,
    domain: Domain,
}

impl PartitionInsides {
    fn cross(&self) -> Cross {
        let mut factors = Vec::new();
        for (_, inside) in INSIDE_CELLS {
            factors.extend(self.domain.factors([PINNED_EDGE; 4], inside));
        }
        // A triangle's record is eight bytes and a quad's twelve, so the stride the legs address
        // the buffer at is the domain's own.
        let stride = (factors.len() / INSIDE_CELLS.len()) as u32;
        let mut legs = Vec::new();
        for (mode, held) in partitions().into_iter().enumerate() {
            for (cell, (site, _)) in INSIDE_CELLS.iter().enumerate() {
                legs.push(Leg::plain(
                    vec![patch_type(self.domain).name(), held.name(), site],
                    mode,
                    cell as u32 * stride,
                ));
            }
        }
        Cross {
            domain: self.domain,
            pipelines: partition_pipelines(),
            factors,
            grids: grid_record(MEDIUM_GRID),
            legs,
        }
    }
}

impl Case for PartitionInsides {
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
        shaders_of(self.domain.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        self.cross().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        self.cross().run(side, sites)
    }
}

// ---- tessellation-partition-scales ------------------------------------------------------------

/// The factor records the mode-by-scale cross lays out, in the buffer's order.
const SCALE_FACTORS: [f32; 3] = [4.0, 17.0, 5.0];

/// The cells the mode-by-scale cross states: the site, the record of [`SCALE_FACTORS`] the leg
/// reads, and the `f32` scale it multiplies by.
///
/// `tessellation_sweeps`' scale sweep measured `setTessellationFactorScale:` under `integer`
/// partitioning alone, where a scaled factor is rounded **up** to a whole number. Three of these
/// four cells state a product that is not a whole number, which is the only thing a scale can hand
/// a partition rule that a factor buffer cannot: a `half` factor is a number each mode's rule has
/// its own answer for, and a product of two of them is a number the buffer could not have held.
///
/// - `scale-2-of-4` is the control: a product of 8, a whole even number, where the scale is a
///   factor by another name.
/// - `scale-half-of-17` halves a factor above the ceiling. Metal scales before it clamps
///   (`tessellation_sweeps`), so the product is 8.5 under every mode and what becomes of the half
///   is the mode's.
/// - `scale-half-of-5` is 2.5, the same question at a level the grid resolves most sharply.
/// - `scale-2-of-5` is 10, a whole number no `half` factor of this family states, which is the
///   control beside the two fractions: a mode that moved for the fraction leaves this cell alone.
const SCALE_CELLS: [(&str, usize, f32); 4] = [
    ("scale-2-of-4", 0, 2.0),
    ("scale-half-of-17", 1, 0.5),
    ("scale-half-of-5", 2, 0.5),
    ("scale-2-of-5", 2, 2.0),
];

/// The partition modes crossed with `setTessellationFactorScale:`.
pub struct PartitionScales;

impl PartitionScales {
    const NAME: &'static str = "tessellation-partition-scales";

    fn cross() -> Cross {
        let mut factors = Vec::new();
        for factor in SCALE_FACTORS {
            factors.extend(quad_factors([factor; 4], [factor; 2]));
        }
        let pipelines: Vec<(String, TessellationState)> = partitions()
            .into_iter()
            .map(|mode| (mode.name().to_owned(), state(mode).scaled()))
            .collect();
        let mut legs = Vec::new();
        for (mode, held) in partitions().into_iter().enumerate() {
            for (site, record, scale) in SCALE_CELLS {
                legs.push(
                    Leg::plain(
                        vec![held.name(), site],
                        mode,
                        record as u32 * QUAD_FACTOR_BYTES,
                    )
                    .scaled(scale.to_bits()),
                );
            }
        }
        Cross {
            domain: Domain::Quad,
            pipelines,
            factors,
            grids: grid_record(MEDIUM_GRID),
            legs,
        }
    }
}

impl Case for PartitionScales {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every pipeline here enables the factor scale, so every run announces the notice the
        // scale is read under. Nothing observed is loosened.
        let mut all = tessellation_notices();
        all.push(crate::replay::notice_allowance(
            Degradation::TessellationFactorScale.code(),
        ));
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(Domain::Quad.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::cross().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::cross().run(side, sites)
    }
}

// ---- tessellation-partition-steps -------------------------------------------------------------

/// The factor records the mode-by-step cross lays out: record 1 is what the per-patch leg's patch 1
/// addresses, and records 0 and 2 are what the per-instance leg's two instances address at
/// [`STEP_INSTANCE_STRIDE`]. Records 1 and 2 are the same factor, so both legs are asked the same
/// factor by two different addressings.
const STEP_FACTORS: [f32; 3] = [1.0, 17.0, 17.0];

/// The instance stride the per-instance leg binds: one record, so instance `n` reads record `n`.
const STEP_INSTANCE_STRIDE: u32 = 2 * QUAD_FACTOR_BYTES;

/// The partition modes crossed with the two factor step functions that address a record.
///
/// `Constant` and `PerPatch` are the two the sweep of `tessellation_sweeps` separated at a patch
/// start, and `PerInstance` the one it separated by drawing two instances; this cross takes the two
/// that **move** the address — a per-patch step at patch 1, and a per-instance step over two
/// instances — and asks each under all four modes. Both legs read a record of factor 17, which is
/// above the ceiling and so the one factor whose level is the mode's own to decide.
///
/// The per-instance leg's two instances write one mark region, which is sound because a mark is a
/// set member and record 0's level-1 or level-2 point set is a subset of record 2's, whatever the
/// mode (`tests::the_step_function_and_the_partition_mode_are_independent` reads the union against
/// the factor case's own answer for 17).
pub struct PartitionSteps;

impl PartitionSteps {
    const NAME: &'static str = "tessellation-partition-steps";

    fn cross() -> Cross {
        let mut factors = Vec::new();
        for factor in STEP_FACTORS {
            factors.extend(quad_factors([factor; 4], [factor; 2]));
        }
        let mut pipelines = Vec::new();
        for mode in partitions() {
            for step in [Step::PerPatch, Step::PerInstance] {
                pipelines.push((
                    format!("{}/{}", mode.name(), step.name()),
                    state(mode).stepping(step),
                ));
            }
        }
        let mut legs = Vec::new();
        for (mode, held) in partitions().into_iter().enumerate() {
            // The per-patch leg draws patch 1, whose own stride the step adds to the bound
            // offset; the per-instance leg draws two instances at a stride of one record.
            legs.push(
                Leg::plain(vec![held.name(), Step::PerPatch.name()], 2 * mode, 0).at_patch(1),
            );
            legs.push(
                Leg::plain(vec![held.name(), Step::PerInstance.name()], 2 * mode + 1, 0)
                    .over_instances(2, STEP_INSTANCE_STRIDE),
            );
        }
        let mut grids = grid_record(MEDIUM_GRID);
        // Two per-patch grid records, the same scale in both: the per-patch leg draws patch 1, and
        // the fixture's own per-patch attribute steps with the patch.
        grids.extend(grid_record(MEDIUM_GRID));
        Cross {
            domain: Domain::Quad,
            pipelines,
            factors,
            grids,
            legs,
        }
    }
}

impl Case for PartitionSteps {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Half the pipelines here state a per-instance factor step, which is what provokes the
        // notice the instance addressing is read under.
        let mut all = tessellation_notices();
        all.push(crate::replay::notice_allowance(
            Degradation::TessellationFactorInstance.code(),
        ));
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shaders_of(Domain::Quad.fixture())
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Self::cross().inputs()
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        Self::cross().run(side, sites)
    }
}

// ---- The family -------------------------------------------------------------------------------

/// Every case of the family, in the order the report lists them.
pub fn boxed() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> =
        vec![Box::new(PartitionFactors), Box::new(PartitionDegenerates)];
    for domain in [Domain::Quad, Domain::Triangle] {
        cases.push(Box::new(PartitionInsides {
            name: match domain {
                Domain::Quad => "tessellation-partition-insides-quad",
                Domain::Triangle => "tessellation-partition-insides-triangle",
            },
            domain,
        }));
    }
    cases.push(Box::new(PartitionScales));
    cases.push(Box::new(PartitionSteps));
    cases
}

#[cfg(test)]
mod tests {
    use super::super::tessellation_sweeps::to_words;
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
        to_words(bytes).into_iter().filter(|w| *w != 0).count()
    }

    /// One cell of the factor cross: the marks of `mode` at `site`.
    fn factor_marks(mode: Partition, site: &str) -> Vec<u8> {
        marks_of(PartitionFactors::NAME, mode, site)
    }

    /// One cell of the degenerate cross.
    fn degenerate_marks(mode: Partition, site: &str) -> Vec<u8> {
        marks_of(PartitionDegenerates::NAME, mode, site)
    }

    /// One cell of the scale cross.
    fn scale_marks(mode: Partition, site: &str) -> Vec<u8> {
        marks_of(PartitionScales::NAME, mode, site)
    }

    /// One cell of a cross whose outer site is the mode's own enumerator.
    fn marks_of(case: &str, mode: Partition, site: &str) -> Vec<u8> {
        recorded_bytes(case, &format!("{}/{site}/marks", mode.name()))
            .unwrap_or_else(|| panic!("the recorded marks of {case} at {} {site}", mode.name()))
    }

    /// One cell of the inside cross, whose outer site is the patch type's enumerator.
    fn inside_marks(domain: Domain, mode: Partition, site: &str) -> Vec<u8> {
        let case = match domain {
            Domain::Quad => "tessellation-partition-insides-quad",
            Domain::Triangle => "tessellation-partition-insides-triangle",
        };
        recorded_bytes(
            case,
            &format!("{}/{}/{site}/marks", patch_type(domain).name(), mode.name()),
        )
        .unwrap_or_else(|| panic!("the recorded marks of {case} at {} {site}", mode.name()))
    }

    /// The four modes the crosses name are every `MTLTessellationPartitionMode` there is, so a
    /// value the SDK grows is a failure here rather than a cell quietly missing from every cross.
    #[test]
    fn the_four_modes_are_every_partition_mode() {
        assert_eq!(PARTITIONS.to_vec(), Partition::ALL.to_vec());
    }

    /// Every factor cell is recorded and every one of them marks the grid's four corners at least,
    /// which is what says a cell drew at all.
    #[test]
    fn every_factor_cell_is_recorded() {
        for mode in partitions() {
            for (site, _) in FACTOR_CELLS {
                let bytes = factor_marks(mode, site);
                assert_eq!(
                    bytes.len() % 4,
                    0,
                    "{} at {site} is whole words",
                    mode.name()
                );
                assert!(
                    marked(&bytes) >= 4,
                    "{} at {site} marked {} cells",
                    mode.name(),
                    marked(&bytes)
                );
            }
        }
    }

    /// **The cross is not the product of two sweeps.** Five of the six pairs of modes answer the
    /// same point set at one factor and different ones at another, so neither a mode on its own nor
    /// a factor on its own decides a cell. Two of the witnesses are the clamp range's own ends:
    /// `fractionalOdd`'s range stops one below the ceiling and `fractionalEven`'s starts at 2.
    #[test]
    fn the_modes_agree_at_one_factor_and_part_at_another() {
        let witnesses = [
            (
                Partition::Pow2,
                Partition::Integer,
                "factor-1",
                "factor-2-5",
            ),
            (
                Partition::Pow2,
                Partition::FractionalOdd,
                "factor-1",
                "factor-2",
            ),
            (
                Partition::Pow2,
                Partition::FractionalEven,
                "factor-2",
                "factor-3",
            ),
            // `fractionalOdd` clamps a factor above the ceiling to one below it, and agrees with
            // `integer` at a factor inside both ranges.
            (
                Partition::Integer,
                Partition::FractionalOdd,
                "factor-15",
                "factor-above-max",
            ),
            // `fractionalEven`'s range begins at 2, so it parts from `integer` at `integer`'s own
            // floor and agrees at an even whole factor.
            (
                Partition::Integer,
                Partition::FractionalEven,
                "factor-2",
                "factor-1",
            ),
        ];
        for (one, other, agree, part) in witnesses {
            assert_eq!(
                factor_marks(one, agree),
                factor_marks(other, agree),
                "{} and {} at {agree}",
                one.name(),
                other.name()
            );
            assert_ne!(
                factor_marks(one, part),
                factor_marks(other, part),
                "{} and {} at {part}",
                one.name(),
                other.name()
            );
        }
    }

    /// The sixth pair, the two fractional spacings, parts at **every** factor this case states, so
    /// the claim above is five pairs and not six. Saying which pair has no agreeing cell here is
    /// the honest half of it.
    #[test]
    fn the_two_fractional_spacings_part_at_every_factor() {
        for (site, _) in FACTOR_CELLS {
            assert_ne!(
                factor_marks(Partition::FractionalOdd, site),
                factor_marks(Partition::FractionalEven, site),
                "the two fractional spacings at {site}"
            );
        }
    }

    /// An edge factor of zero culls the patch under **every** mode, so Metal's cull is not a clamp
    /// that reaches a range's lower end: `fractionalEven`, whose range begins at 2, marks nothing
    /// here as well.
    #[test]
    fn a_zero_edge_factor_culls_under_every_partition_mode() {
        for mode in partitions() {
            let bytes = degenerate_marks(mode, "edges-zero");
            assert_eq!(
                marked(&bytes),
                0,
                "{} marked {} cells at a zero edge factor",
                mode.name(),
                marked(&bytes)
            );
        }
    }

    /// A degenerate factor the cull does not read is clamped under the **mode's** own rule: the
    /// zero and the NaN inside factor answer one set under each mode, which is the clamp's lower
    /// end reached from both directions, and on each of the four clamped cells `fractionalOdd`
    /// answers a set the other three do not.
    #[test]
    fn a_clamped_degenerate_factor_is_the_partition_modes_own() {
        for mode in partitions() {
            assert_eq!(
                degenerate_marks(mode, "inside-zero"),
                degenerate_marks(mode, "inside-nan"),
                "the clamp's lower end from both directions under {}",
                mode.name()
            );
        }
        for site in [
            "inside-zero",
            "inside-nan",
            "inside-infinite",
            "edges-infinite",
        ] {
            let held = degenerate_marks(Partition::Integer, site);
            assert_eq!(
                held,
                degenerate_marks(Partition::Pow2, site),
                "pow2 and integer at {site}"
            );
            assert_eq!(
                held,
                degenerate_marks(Partition::FractionalEven, site),
                "fractionalEven and integer at {site}"
            );
            assert_ne!(
                held,
                degenerate_marks(Partition::FractionalOdd, site),
                "fractionalOdd and integer at {site}"
            );
        }
    }

    /// **The second inside factor is the quad's and not the triangle's.** Moving the second slot
    /// moves every quad cell under every mode and no triangle cell under any mode; moving the first
    /// slot moves both. So the patch type decides how many of the inside factors the mode's rule is
    /// applied to, which is a claim about the pair and about neither axis alone.
    #[test]
    fn the_second_inside_factor_is_the_quads_alone() {
        for mode in partitions() {
            for (domain, moves_with_the_second) in [(Domain::Quad, true), (Domain::Triangle, false)]
            {
                let base = inside_marks(domain, mode, "inside-2-and-2");
                let second = inside_marks(domain, mode, "inside-2-and-5");
                let first = inside_marks(domain, mode, "inside-5-and-2");
                match moves_with_the_second {
                    true => assert_ne!(
                        base,
                        second,
                        "{} moved with the second inside factor under {}",
                        patch_type(domain).name(),
                        mode.name()
                    ),
                    false => assert_eq!(
                        base,
                        second,
                        "{} held still under the second inside factor under {}",
                        patch_type(domain).name(),
                        mode.name()
                    ),
                }
                assert_ne!(
                    base,
                    first,
                    "{} moved with the first inside factor under {}",
                    patch_type(domain).name(),
                    mode.name()
                );
            }
        }
    }

    /// **What a mode does with the half a scale produced is the pair's.** `integer` and
    /// `fractionalEven` answer the same cell set at both of the whole products the case states and
    /// different ones at both of the fractional products, so the scale's own value does not decide
    /// the cell and neither does the mode.
    #[test]
    fn the_scaled_fraction_is_the_partition_modes_to_round() {
        for site in ["scale-2-of-4", "scale-2-of-5"] {
            assert_eq!(
                scale_marks(Partition::Integer, site),
                scale_marks(Partition::FractionalEven, site),
                "integer and fractionalEven at {site}, a whole product"
            );
        }
        for site in ["scale-half-of-17", "scale-half-of-5"] {
            assert_ne!(
                scale_marks(Partition::Integer, site),
                scale_marks(Partition::FractionalEven, site),
                "integer and fractionalEven at {site}, a product that is not whole"
            );
        }
    }

    /// The factor step function and the partition mode are **independent**: a per-patch step at
    /// patch 1 and a per-instance step over two instances each answer exactly what the factor
    /// cross answers for the factor their address reaches, under all four modes.
    #[test]
    fn the_step_function_and_the_partition_mode_are_independent() {
        for mode in partitions() {
            let expected = factor_marks(mode, "factor-above-max");
            for step in [Step::PerPatch, Step::PerInstance] {
                let bytes = recorded_bytes(
                    PartitionSteps::NAME,
                    &format!("{}/{}/marks", mode.name(), step.name()),
                )
                .unwrap_or_else(|| {
                    panic!("the recorded marks of {} at {}", mode.name(), step.name())
                });
                assert_eq!(
                    bytes,
                    expected,
                    "{} under {} against the factor cross's own answer",
                    mode.name(),
                    step.name()
                );
            }
        }
    }
}
