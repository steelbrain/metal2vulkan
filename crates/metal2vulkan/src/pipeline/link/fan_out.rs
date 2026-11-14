//! The fan-out geometry stage of a render pipeline variant that draws a rate-mapped pass's
//! triangles once (`VariantKey::region_fan_out`): the vertex function runs once per vertex, and
//! the triangle class's geometry stage emits a copy of each triangle for each region it touches,
//! through the region table's lines and clip distances (`bind::fan_out`, its layout and math),
//! where the regions route records each draw once per region.
//!
//! The stage passes through what [`GeometryStages`] passes through (the vertex outputs the
//! fragment stage reads, the vertex function's clip distances, `PrimitiveId`), emits each copy's
//! vertices in the order it receives them, so a copy keeps the triangle's winding and provoking
//! vertex, and writes four clip distances past the vertex function's, which keep the copy's
//! region's pixels. Its point and line classes' stages are the plain ones: the variant draws
//! triangles alone.
//!
//! - **Draw** ([`RegionFanOut::Draw`]): a triangle every `w` of which is positive emits one copy
//!   per pair of segments its bounds reach, at most [`FanOutStage::copies`]; one with no positive
//!   `w`, or one past the segments, emits none, and so does one with some `w` not positive, which
//!   the probe found. Each copy's vertices are [`region_copy`]'s, the emitter a vertex function
//!   that draws its region's copy shares (`LoweringInputs::region_copies`), and the stage's
//!   `Position` and `ClipDistance` outputs are `Invariant`. The copies are a loop the driver is
//!   asked not to unroll (`DontUnroll`): its
//!   body compiled once, the pipeline is made in about a third of the time 16 unrolled copies
//!   took (on an RTX 3090 45.2 against 81.3 ms for the probe's and the draw's, Iris Xe 20.0
//!   against 66.9 ms, Radeon 780M 14.4 against 44.5 ms), with the same GPU time. A draw of few
//!   copies ([`RegionFanOut::DrawFew`]) is the same stage of at most `bind::fan_out::FEW_COPIES`
//!   copies, which declares the output of that many.
//! - **Probe** ([`RegionFanOut::Probe`]): emits nothing. Of the flags buffer's words of the
//!   index `i` the draw pushes (`PushField::RegionProbe`; [`PROBE_DRAW_WORDS`] of them from
//!   `PROBE_DRAW_WORDS·i`), it writes 1 into the first for a triangle the draw cannot fan out: one
//!   with some `w` not positive and another positive, or whose bounds, widened by the table's
//!   widening, reach more pairs than the copies; and adds the copies the draw emits of each
//!   triangle it fans out, the pairs its bounds reach unwidened, into the count its `PrimitiveId`
//!   picks of the [`PROBE_COUNT_WORDS`] after it, so the consumer sums the draw's work and no
//!   word takes every triangle's add; for a triangle it does not flag whose bounds, widened,
//!   reach more pairs than [`FEW_COPIES`], writes 1 into the word its `PrimitiveId` picks of the
//!   as many after those, which mark the draw one its variant of few copies cannot draw; and for
//!   a triangle it flags, raises the word its `PrimitiveId` picks of the as many after those to
//!   `u32::MAX − PrimitiveId`, and of the as many after them to `PrimitiveId + 1`, so the
//!   consumer finds the draw's first and last flagged triangle, adds one into the word it
//!   picks of the fifth set, so the consumer counts them, and sets the bit of its block of
//!   [`BLOCK_TRIANGLES`] in the sixth ([`BLOCK_SET`]), so the consumer draws only the blocks that
//!   hold a flagged triangle region by region; for a flagged triangle with some `w` not positive
//!   beside a positive one, adds one into the word it picks of the seventh ([`STRADDLE_SET`]).
//!   Of every triangle it reaches, it sets the bits of the buckets of segments its widened bounds
//!   reach in its block's bounds ([`PROBE_BOUNDS_AT`], [`BOUND_BUCKETS`]), every bit for one with
//!   some `w` not positive beside a positive one, so the consumer draws a run of blocks only into
//!   the regions its blocks reach. Its writes need `vertexPipelineStoresAndAtomics`.

use super::geometry::{array, kf, ku, scalar, GeometryStages};
use crate::bind::fan_out::{
    clip_entry, line_entry, BLOCK_SET, BLOCK_TRIANGLES, BOUND_BUCKETS, FEW_COPIES, MAX_SEGMENTS,
    PROBE_BLOCKS, PROBE_BOUNDS_AT, PROBE_COUNT_WORDS, PROBE_DRAW_WORDS, STRADDLE_SET,
};
use crate::pipeline::graphics::RegionFanOut;
use crate::pipeline::stage_io::IoNumeric;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::ops::group::GroupMode;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConvOp, ExecMode, ExecModel, GlslOp, GroupArith, IoDecor,
    IoDir, LayoutRule, MemAccess, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};
use crate::spv::BuiltIn as B;

/// The most copies of a triangle a draw's stage emits: past it, a device's geometry stage
/// reserves output for vertices most triangles never emit.
pub const MOST_COPIES: u32 = 16;

/// The bytes of the probe's push range: the index of the flag it writes.
pub const PUSH_BYTES: u32 = 4;

/// The binding symbol of the region table (`SyntheticBinding::RateMapRegions`), declared with
/// the vertex stage's symbols.
pub const REGIONS_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::RateMapRegions),
    index: 0,
};

/// The binding symbol of the probe's flags (`SyntheticBinding::RateMapFlags`), declared with the
/// vertex stage's symbols.
pub const FLAGS_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::RateMapFlags),
    index: 0,
};

/// A fan-out variant's triangle stage (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FanOutStage {
    /// What it does.
    pub mode: RegionFanOut,
    /// The most copies of a triangle it emits, what the device's geometry limits leave for its
    /// vertices' outputs, at most [`MOST_COPIES`]: set once link placed the varyings.
    pub copies: u32,
    /// The byte offset of the probe's push range ([`PUSH_BYTES`]); 0 for a draw.
    pub push_offset: u32,
    /// Whether a probe's triangles of a subgroup raise each block's bounds word once, their
    /// bounds ORed together (`bind::fan_out::PROBE_BOUNDS_AT`), rather than each its own: the
    /// device runs subgroup operations of the basic, arithmetic and ballot classes in the
    /// geometry stage (`VkProperty::SubgroupGeometryStage`). Set once link placed the varyings,
    /// as the copies are; false for a draw.
    pub subgroup_bounds: bool,
}

impl FanOutStage {
    /// The stage of `mode`, its copies and push offset not yet placed.
    pub fn new(mode: RegionFanOut) -> FanOutStage {
        FanOutStage {
            mode,
            copies: 0,
            push_offset: 0,
            subgroup_bounds: false,
        }
    }
}

/// The values the copies are made from.
struct Copies {
    table: GlobalId,
    positions: [Value; 3],
    own_clips: Option<GlobalId>,
    own: u32,
    clips: GlobalId,
    position: GlobalId,
    carried: Vec<(GlobalId, GlobalId)>,
    primitive: Option<(GlobalId, GlobalId)>,
    vec4: TypeId,
    f32_: TypeId,
}

/// The table's word `word` (a `uint` value).
fn word(fb: &mut FunctionBuilder<'_>, table: GlobalId, word: Value) -> BResult<Value> {
    let base = fb.module().global_value(table)?;
    let member = ku(fb, 0)?;
    let ptr = fb.access_chain(base, &[member, word])?;
    fb.load(ptr, MemAccess::default())
}

/// Component `c` of entry `entry` plus `k` of the table, as a float: `k` a `uint` value, or none.
fn component(
    fb: &mut FunctionBuilder<'_>,
    (table, f32_): (GlobalId, TypeId),
    entry: u32,
    k: Option<Value>,
    c: u32,
) -> BResult<Value> {
    let at = ku(fb, entry * 4 + c)?;
    let at = match k {
        Some(k) => {
            let four = ku(fb, 4)?;
            let k4 = fb.binary(BinOp::IMul, k, four)?;
            fb.binary(BinOp::IAdd, at, k4)?
        }
        None => at,
    };
    let bits = word(fb, table, at)?;
    fb.convert(ConvOp::Bitcast, bits, f32_)
}

/// The last segment of axis `axis` (of `n`) whose least coordinate is at most `v`, or the first:
/// a bisection of `log2(MAX_SEGMENTS)` steps over the table's lines.
fn find(
    fb: &mut FunctionBuilder<'_>,
    table: (GlobalId, TypeId),
    axis: u32,
    n: Value,
    v: Value,
) -> BResult<Value> {
    let zero = ku(fb, 0)?;
    let one = ku(fb, 1)?;
    let empty = fb.binary(BinOp::IEqual, n, zero)?;
    let last = fb.binary(BinOp::ISub, n, one)?;
    let mut lo = zero;
    let mut hi = fb.select(empty, zero, last)?;
    for _ in 0..MAX_SEGMENTS.ilog2() {
        let open = fb.binary(BinOp::ULessThan, lo, hi)?;
        let sum = fb.binary(BinOp::IAdd, lo, hi)?;
        let sum = fb.binary(BinOp::IAdd, sum, one)?;
        let mid = fb.binary(BinOp::ShiftRightLogical, sum, one)?;
        let at = component(fb, table, line_entry(axis, 0), Some(mid), 0)?;
        let below = fb.binary(BinOp::FOrdLessThanEqual, at, v)?;
        let above = fb.unary(UnOp::LogicalNot, below)?;
        let up = fb.binary(BinOp::LogicalAnd, open, below)?;
        let down = fb.binary(BinOp::LogicalAnd, open, above)?;
        let before = fb.binary(BinOp::ISub, mid, one)?;
        lo = fb.select(up, mid, lo)?;
        hi = fb.select(down, before, hi)?;
    }
    Ok(lo)
}

/// The least and the greatest normalized coordinate of axis `axis` (0 x, 1 y) that a triangle of
/// clip-space `positions` reaches where its `w` is positive (`positive`, a vertex's; `ndc`, its
/// coordinate over its `w`), as the probe bounds a straddling triangle's block: those of its
/// vertices of positive `w`, and infinity on each side an edge from such a vertex `a` to one `b`
/// of `w` not positive crosses `w = 0`, the sign of `w_a·v_b − v_a·w_b`, a sign within a
/// rounding margin of zero (or not a number) taking both sides. A triangle of every `w` positive
/// reaches from its least vertex's to its greatest's; one of none, nothing.
fn visible_reach(
    fb: &mut FunctionBuilder<'_>,
    positions: [Value; 3],
    positive: [Value; 3],
    ndc: [Value; 3],
    axis: u32,
) -> BResult<(Value, Value)> {
    let below = kf(fb, f32::NEG_INFINITY)?;
    let above = kf(fb, f32::INFINITY)?;
    let mut least = Vec::with_capacity(3);
    let mut greatest = Vec::with_capacity(3);
    for (&positive, &v) in positive.iter().zip(&ndc) {
        least.push(fb.select(positive, v, above)?);
        greatest.push(fb.select(positive, v, below)?);
    }
    let mut lo = extreme(fb, GlslOp::FMin, [least[0], least[1], least[2]])?;
    let mut hi = extreme(fb, GlslOp::FMax, [greatest[0], greatest[1], greatest[2]])?;
    let zero = kf(fb, 0.0)?;
    // The margin: 2^-16 of the products' magnitudes, past a float's rounding of their difference.
    let margin = kf(fb, 1.0 / 65536.0)?;
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        let crosses = fb.binary(BinOp::LogicalNotEqual, positive[a], positive[b])?;
        let v_a = fb.composite_extract(positions[a], &[axis])?;
        let w_a = fb.composite_extract(positions[a], &[3])?;
        let v_b = fb.composite_extract(positions[b], &[axis])?;
        let w_b = fb.composite_extract(positions[b], &[3])?;
        let m = fb.binary(BinOp::FMul, w_a, v_b)?;
        let n = fb.binary(BinOp::FMul, v_a, w_b)?;
        let side = fb.binary(BinOp::FSub, m, n)?;
        let flipped = fb.unary(UnOp::FNegate, side)?;
        let side = fb.select(positive[a], side, flipped)?;
        let m = fb.glsl(GlslOp::FAbs, &[m])?;
        let n = fb.glsl(GlslOp::FAbs, &[n])?;
        let size = fb.binary(BinOp::FAdd, m, n)?;
        let size = fb.binary(BinOp::FMul, size, margin)?;
        let up = fb.binary(BinOp::FAdd, side, size)?;
        let up = fb.binary(BinOp::FUnordGreaterThanEqual, up, zero)?;
        let up = fb.binary(BinOp::LogicalAnd, crosses, up)?;
        let down = fb.binary(BinOp::FSub, side, size)?;
        let down = fb.binary(BinOp::FUnordLessThanEqual, down, zero)?;
        let down = fb.binary(BinOp::LogicalAnd, crosses, down)?;
        hi = fb.select(up, above, hi)?;
        lo = fb.select(down, below, lo)?;
    }
    Ok((lo, hi))
}

/// The least or the greatest of three floats.
fn extreme(fb: &mut FunctionBuilder<'_>, op: GlslOp, v: [Value; 3]) -> BResult<Value> {
    let two = fb.glsl(op, &[v[0], v[1]])?;
    fb.glsl(op, &[two, v[2]])
}

/// Declare the region table (`SyntheticBinding::RateMapRegions`, [`REGIONS_SLOT`]) in `mb`: a
/// `Block { uint[] }` storage buffer, which a fan-out stage reads and a vertex function that
/// draws its region's copy (`LoweringInputs::region_copies`) reads alike.
pub(crate) fn declare_table(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let words = mb.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
    let block = mb.ty(Type::Struct {
        members: [words].into(),
        layout: Some(layout),
    })?;
    mb.resource(REGIONS_SLOT, StorageClass::StorageBuffer, block)
}

/// The segments of region `k` (a `uint`) of the table `table`, in its order (`bind::fan_out`,
/// **One route per pass**): `(k mod nx, k / nx)`, `nx` its segments of axis x.
pub(crate) fn region_segments(
    fb: &mut FunctionBuilder<'_>,
    table: GlobalId,
    k: Value,
) -> BResult<(Value, Value)> {
    let header = ku(fb, 0)?;
    let nx = word(fb, table, header)?;
    let sx = fb.binary(BinOp::UMod, k, nx)?;
    let sy = fb.binary(BinOp::UDiv, k, nx)?;
    Ok((sx, sy))
}

/// The terms of the copy for segments `sx` and `sy` (`uint`s) of the table `table` (its global
/// and `float` type), of axis x then y: `[a, b, W/2, X + W/2 − p₀, p₁ − X − W/2]`, its line's
/// `a` and `b` and its clip (the layout of `bind::fan_out`).
pub(crate) fn region_terms(
    fb: &mut FunctionBuilder<'_>,
    table: (GlobalId, TypeId),
    (sx, sy): (Value, Value),
) -> BResult<[[Value; 5]; 2]> {
    let mut axes = [[sx; 5]; 2];
    for (axis, s) in [(0, sx), (1, sy)] {
        let terms = &mut axes[axis as usize];
        terms[0] = component(fb, table, line_entry(axis, 0), Some(s), 2)?;
        terms[1] = component(fb, table, line_entry(axis, 0), Some(s), 3)?;
        for (t, c) in [(2, 0), (3, 1), (4, 2)] {
            terms[t] = component(fb, table, clip_entry(axis, 0), Some(s), c)?;
        }
    }
    Ok(axes)
}

/// The copy of the clip-space position `p` (a `vec4` of type `vec4`) through `terms`
/// ([`region_terms`]): `(a·x + b·w, a·y + b·w, z, w)`, each axis's own `a` and `b`, and the clip
/// distances `W/2·v + lo·w` and `hi·w − W/2·v` of x then y. The one emitter of a region's copy:
/// a fan-out stage's of each vertex it emits, and a vertex function's that draws its region's
/// copy (`LoweringInputs::region_copies`). Each operation is decorated `NoContraction`, so each
/// is correctly rounded on its own (Vulkan's *Precision and Operation of SPIR-V Instructions*)
/// and both put a vertex in a region at the same bits (`bind::fan_out`, **One route per pass**).
pub(crate) fn region_copy(
    fb: &mut FunctionBuilder<'_>,
    terms: &[[Value; 5]; 2],
    p: Value,
    vec4: TypeId,
) -> BResult<(Value, [Value; 4])> {
    let w = fb.composite_extract(p, &[3])?;
    let z = fb.composite_extract(p, &[2])?;
    let mut moved = [w; 2];
    let mut distances = [w; 4];
    for (axis, &[a, b, half, lo, hi]) in terms.iter().enumerate() {
        let v = fb.composite_extract(p, &[axis as u32])?;
        let av = fb.binary_no_contraction(BinOp::FMul, a, v)?;
        let bw = fb.binary_no_contraction(BinOp::FMul, b, w)?;
        moved[axis] = fb.binary_no_contraction(BinOp::FAdd, av, bw)?;
        let hv = fb.binary_no_contraction(BinOp::FMul, half, v)?;
        let low = fb.binary_no_contraction(BinOp::FMul, lo, w)?;
        distances[2 * axis] = fb.binary_no_contraction(BinOp::FAdd, hv, low)?;
        let high = fb.binary_no_contraction(BinOp::FMul, hi, w)?;
        distances[2 * axis + 1] = fb.binary_no_contraction(BinOp::FSub, high, hv)?;
    }
    let position = fb.composite_construct(vec4, &[moved[0], moved[1], z, w])?;
    Ok((position, distances))
}

/// Emit the copy of the triangle for segments `sx` and `sy`: each vertex through their lines,
/// with their clip distances past the vertex function's ([`region_copy`]).
fn emit_copy(fb: &mut FunctionBuilder<'_>, c: &Copies, (sx, sy): (Value, Value)) -> BResult<()> {
    let terms = region_terms(fb, (c.table, c.f32_), (sx, sy))?;
    for (k, &p) in c.positions.iter().enumerate() {
        let at = ku(fb, k as u32)?;
        let (position, distances) = region_copy(fb, &terms, p, c.vec4)?;
        let to = fb.module().global_value(c.position)?;
        fb.store(to, position, MemAccess::default())?;
        let clips = fb.module().global_value(c.clips)?;
        if let Some(own) = c.own_clips {
            let from = fb.module().global_value(own)?;
            for e in 0..c.own {
                let element = ku(fb, e)?;
                let ptr = fb.access_chain(from, &[at, element])?;
                let value = fb.load(ptr, MemAccess::default())?;
                let to = fb.access_chain(clips, &[element])?;
                fb.store(to, value, MemAccess::default())?;
            }
        }
        for (e, &d) in distances.iter().enumerate() {
            let element = ku(fb, c.own + e as u32)?;
            let to = fb.access_chain(clips, &[element])?;
            fb.store(to, d, MemAccess::default())?;
        }
        for &(input, output) in &c.carried {
            let base = fb.module().global_value(input)?;
            let from = fb.access_chain(base, &[at])?;
            let value = fb.load(from, MemAccess::default())?;
            let to = fb.module().global_value(output)?;
            fb.store(to, value, MemAccess::default())?;
        }
        if let Some((input, output)) = c.primitive {
            let from = fb.module().global_value(input)?;
            let value = fb.load(from, MemAccess::default())?;
            let to = fb.module().global_value(output)?;
            fb.store(to, value, MemAccess::default())?;
        }
        fb.emit_vertex()?;
    }
    fb.end_primitive()
}

/// The fan-out variant's triangle stage of `stages` (the module docs).
pub(super) fn module(stages: &GeometryStages, fan: FanOutStage) -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let u32_ = scalar(&mut mb, IoNumeric::Uint)?;
    let f32_ = scalar(&mut mb, IoNumeric::Float)?;
    let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
    let void = mb.ty(Type::Void)?;
    let ty = array(&mut mb, vec4, 3)?;
    let positions = mb.builtin(B::Position, IoDir::Input, ty)?;
    let position = mb.builtin(B::Position, IoDir::Output, vec4)?;
    mb.invariant(position)?;
    let own = stages.clip_distances;
    let own_clips = match own {
        0 => None,
        n => {
            let clip = array(&mut mb, f32_, n)?;
            let ty = array(&mut mb, clip, 3)?;
            Some(mb.builtin(B::ClipDistance, IoDir::Input, ty)?)
        }
    };
    let ty = array(&mut mb, f32_, own + 4)?;
    let clips = mb.builtin(B::ClipDistance, IoDir::Output, ty)?;
    mb.invariant(clips)?;
    let packed = IoDecor {
        packed: true,
        ..IoDecor::default()
    };
    let mut carried = Vec::with_capacity(stages.varyings.len());
    for (key, io) in &stages.varyings {
        let elem = scalar(&mut mb, io.numeric)?;
        let ty = match io.components {
            1 => elem,
            c => mb.ty(Type::Vector { elem, len: c })?,
        };
        let input_ty = array(&mut mb, ty, 3)?;
        carried.push((
            mb.io(key.clone(), IoDir::Input, input_ty, packed)?,
            mb.io(key.clone(), IoDir::Output, ty, packed)?,
        ));
    }
    let primitive = match stages.primitive_id {
        true => Some((
            mb.builtin(B::PrimitiveId, IoDir::Input, u32_)?,
            mb.builtin(B::PrimitiveId, IoDir::Output, u32_)?,
        )),
        false => None,
    };
    let table = declare_table(&mut mb)?;
    let probe = match fan.mode {
        RegionFanOut::Probe => {
            let words = mb.ty(Type::RuntimeArray {
                elem: u32_,
                stride: Some(4),
            })?;
            let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
            let block = mb.ty(Type::Struct {
                members: [words].into(),
                layout: Some(layout),
            })?;
            let flags = mb.resource(FLAGS_SLOT, StorageClass::StorageBuffer, block)?;
            let member = (mb.push_member(PushField::RegionProbe, u32_, fan.push_offset)?).index;
            // The probe picks a triangle's count by its `PrimitiveId`.
            let id = match primitive {
                Some((input, _)) => input,
                None => mb.builtin(B::PrimitiveId, IoDir::Input, u32_)?,
            };
            Some((flags, member, mb.push_global()?, id))
        }
        RegionFanOut::Draw | RegionFanOut::DrawFew | RegionFanOut::RegionCopies => None,
    };
    let mut fb = mb.function(void)?;
    let mut loaded = Vec::with_capacity(3);
    for k in 0..3 {
        let at = ku(&mut fb, k)?;
        let base = fb.module().global_value(positions)?;
        let from = fb.access_chain(base, &[at])?;
        loaded.push(fb.load(from, MemAccess::default())?);
    }
    let positions = [loaded[0], loaded[1], loaded[2]];
    let zero = kf(&mut fb, 0.0)?;
    let mut positive = Vec::with_capacity(3);
    for &p in &positions {
        let w = fb.composite_extract(p, &[3])?;
        positive.push(fb.binary(BinOp::FOrdGreaterThan, w, zero)?);
    }
    let two = fb.binary(BinOp::LogicalAnd, positive[0], positive[1])?;
    let every = fb.binary(BinOp::LogicalAnd, two, positive[2])?;
    let two = fb.binary(BinOp::LogicalOr, positive[0], positive[1])?;
    let some = fb.binary(BinOp::LogicalOr, two, positive[2])?;
    let table_f = (table, f32_);
    let header = ku(&mut fb, 0)?;
    let nx = word(&mut fb, table, header)?;
    let header = ku(&mut fb, 1)?;
    let ny = word(&mut fb, table, header)?;
    let one = ku(&mut fb, 1)?;
    // Per axis: the first segment its bounds reach, how many, and whether they reach any.
    let mut reach = Vec::with_capacity(2);
    for (axis, n) in [(0u32, nx), (1, ny)] {
        let mut ndc = Vec::with_capacity(3);
        for &p in &positions {
            let v = fb.composite_extract(p, &[axis])?;
            let w = fb.composite_extract(p, &[3])?;
            ndc.push(fb.binary(BinOp::FDiv, v, w)?);
        }
        let ndc = [ndc[0], ndc[1], ndc[2]];
        let (mut lo, mut hi) = match probe {
            Some(_) => {
                let positive = [positive[0], positive[1], positive[2]];
                visible_reach(&mut fb, positions, positive, ndc, axis)?
            }
            None => (
                extreme(&mut fb, GlslOp::FMin, ndc)?,
                extreme(&mut fb, GlslOp::FMax, ndc)?,
            ),
        };
        if probe.is_some() {
            let widen = component(&mut fb, table_f, 1, None, axis)?;
            lo = fb.binary(BinOp::FSub, lo, widen)?;
            hi = fb.binary(BinOp::FAdd, hi, widen)?;
        }
        let first = component(&mut fb, table_f, line_entry(axis, 0), None, 0)?;
        let last = fb.binary(BinOp::ISub, n, one)?;
        let end = component(&mut fb, table_f, line_entry(axis, 0), Some(last), 1)?;
        let after = fb.binary(BinOp::FOrdGreaterThanEqual, hi, first)?;
        let before = fb.binary(BinOp::FOrdLessThanEqual, lo, end)?;
        let inside = fb.binary(BinOp::LogicalAnd, after, before)?;
        let i0 = find(&mut fb, table_f, axis, n, lo)?;
        let i1 = find(&mut fb, table_f, axis, n, hi)?;
        let span = fb.binary(BinOp::ISub, i1, i0)?;
        let span = fb.binary(BinOp::IAdd, span, one)?;
        reach.push((i0, span, inside));
    }
    let (x0, xs, x_in) = reach[0];
    let (y0, ys, y_in) = reach[1];
    let both = fb.binary(BinOp::LogicalAnd, x_in, y_in)?;
    let fans = fb.binary(BinOp::LogicalAnd, every, both)?;
    let pairs = fb.binary(BinOp::IMul, xs, ys)?;
    let none = ku(&mut fb, 0)?;
    let count = fb.select(fans, pairs, none)?;
    let copies = fan.copies.max(1);
    let mut ctx = ();
    match probe {
        Some((flags, member, push, id)) => {
            // Some `w` not positive beside a positive one, or more pairs than the copies.
            let straddles = fb.unary(UnOp::LogicalNot, every)?;
            let straddles = fb.binary(BinOp::LogicalAnd, some, straddles)?;
            let most = ku(&mut fb, copies)?;
            let over = fb.binary(BinOp::UGreaterThan, count, most)?;
            let flagged = fb.binary(BinOp::LogicalOr, straddles, over)?;
            // The draw's words: its flag, then the counts of the copies its triangles emit, this
            // triangle's the one its `PrimitiveId` picks.
            let base = fb.module().global_value(push)?;
            let index = ku(&mut fb, member)?;
            let ptr = fb.access_chain(base, &[index])?;
            let pushed = fb.load(ptr, MemAccess::default())?;
            let draw_words = ku(&mut fb, PROBE_DRAW_WORDS)?;
            let flag_at = fb.binary(BinOp::IMul, pushed, draw_words)?;
            let from = fb.module().global_value(id)?;
            let id = fb.load(from, MemAccess::default())?;
            let mask = ku(&mut fb, PROBE_COUNT_WORDS - 1)?;
            let lane = fb.binary(BinOp::BitwiseAnd, id, mask)?;
            let lane = fb.binary(BinOp::IAdd, lane, one)?;
            let count_at = fb.binary(BinOp::IAdd, flag_at, lane)?;
            let word_ptr = move |fb: &mut FunctionBuilder<'_>, at: Value| {
                let base = fb.module().global_value(flags)?;
                let zero = ku(fb, 0)?;
                fb.access_chain(base, &[zero, at])
            };
            // A flagged triangle: the draw's flag, its `PrimitiveId` raised into the first's and
            // the last's words it picks, and one added into the flagged count it picks: the
            // sets after the counts, its word the same lane of each.
            let lanes = ku(&mut fb, PROBE_COUNT_WORDS)?;
            let wide_at = fb.binary(BinOp::IAdd, count_at, lanes)?;
            let first_at = fb.binary(BinOp::IAdd, wide_at, lanes)?;
            let last_at = fb.binary(BinOp::IAdd, first_at, lanes)?;
            let flagged_at = fb.binary(BinOp::IAdd, last_at, lanes)?;
            let most_id = ku(&mut fb, u32::MAX)?;
            let first = fb.binary(BinOp::ISub, most_id, id)?;
            let last = fb.binary(BinOp::IAdd, id, one)?;
            // Its block's bit, in the last set: the block of its `PrimitiveId`, the last for one
            // past the blocks.
            let shift = ku(&mut fb, BLOCK_TRIANGLES.ilog2())?;
            let block = fb.binary(BinOp::ShiftRightLogical, id, shift)?;
            let most_block = ku(&mut fb, PROBE_BLOCKS - 1)?;
            let block = fb.glsl(GlslOp::UMin, &[block, most_block])?;
            let five = ku(&mut fb, 5)?;
            let block_word = fb.binary(BinOp::ShiftRightLogical, block, five)?;
            let blocks = ku(&mut fb, 1 + BLOCK_SET * PROBE_COUNT_WORDS)?;
            let block_at = fb.binary(BinOp::IAdd, flag_at, blocks)?;
            let block_at = fb.binary(BinOp::IAdd, block_at, block_word)?;
            let low = ku(&mut fb, 31)?;
            let bit = fb.binary(BinOp::BitwiseAnd, block, low)?;
            let bit = fb.binary(BinOp::ShiftLeftLogical, one, bit)?;
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let ptr = word_ptr(fb, flag_at)?;
                let one = ku(fb, 1)?;
                fb.atomic_store(ptr, Scope::Device, Semantics::RELAXED, one)?;
                for (at, value) in [(first_at, first), (last_at, last)] {
                    let ptr = word_ptr(fb, at)?;
                    fb.atomic(
                        AtomicOp::UMax,
                        ptr,
                        Scope::Device,
                        Semantics::RELAXED,
                        value,
                    )?;
                }
                let ptr = word_ptr(fb, flagged_at)?;
                fb.atomic(AtomicOp::IAdd, ptr, Scope::Device, Semantics::RELAXED, one)?;
                let ptr = word_ptr(fb, block_at)?;
                fb.atomic(AtomicOp::Or, ptr, Scope::Device, Semantics::RELAXED, bit)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut ctx, flagged, arm, None)?;
            // A straddling triangle: one added into the straddle count its `PrimitiveId` picks.
            let straddle_at = ku(&mut fb, 1 + STRADDLE_SET * PROBE_COUNT_WORDS)?;
            let straddle_at = fb.binary(BinOp::IAdd, flag_at, straddle_at)?;
            let lane_at = fb.binary(BinOp::ISub, lane, one)?;
            let straddle_at = fb.binary(BinOp::IAdd, straddle_at, lane_at)?;
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let ptr = word_ptr(fb, straddle_at)?;
                let one = ku(fb, 1)?;
                fb.atomic(AtomicOp::IAdd, ptr, Scope::Device, Semantics::RELAXED, one)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut ctx, straddles, arm, None)?;
            // Its block's bounds: the buckets of segments its widened bounds reach of each axis
            // where its `w` is positive (`visible_reach`), x's in the low half and y's in the
            // high, for a triangle of some `w` positive that reaches the regions
            // (`bind::fan_out::PROBE_BOUNDS_AT`).
            let buckets = ku(&mut fb, BOUND_BUCKETS)?;
            let mut halves = Vec::with_capacity(2);
            for (from, span, n) in [(x0, xs, nx), (y0, ys, ny)] {
                let lo = fb.binary(BinOp::IMul, from, buckets)?;
                let lo = fb.binary(BinOp::UDiv, lo, n)?;
                let to = fb.binary(BinOp::IAdd, from, span)?;
                let to = fb.binary(BinOp::ISub, to, one)?;
                let hi = fb.binary(BinOp::IMul, to, buckets)?;
                let hi = fb.binary(BinOp::UDiv, hi, n)?;
                let two = ku(&mut fb, 2)?;
                let past = fb.binary(BinOp::ShiftLeftLogical, two, hi)?;
                let below = fb.binary(BinOp::ShiftLeftLogical, one, lo)?;
                halves.push(fb.binary(BinOp::ISub, past, below)?);
            }
            let sixteen = ku(&mut fb, 16)?;
            let high = fb.binary(BinOp::ShiftLeftLogical, halves[1], sixteen)?;
            let bounds = fb.binary(BinOp::BitwiseOr, halves[0], high)?;
            let bounded = fb.binary(BinOp::LogicalAnd, some, both)?;
            let bounds_at = ku(&mut fb, PROBE_BOUNDS_AT)?;
            let bounds_at = fb.binary(BinOp::IAdd, flag_at, bounds_at)?;
            let bounds_at = fb.binary(BinOp::IAdd, bounds_at, block)?;
            let arm: ArmFn<'_, ()> = match fan.subgroup_bounds {
                // A subgroup's triangles raise a block's word once (`FanOutStage::subgroup_bounds`):
                // the lanes of the first lane's word OR their bounds together and one of them
                // raises it, then the lanes of another word, until none is left.
                true => Box::new(move |fb, ctx| {
                    let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
                        let first = fb.group_broadcast_first(bounds_at)?;
                        let mine = fb.binary(BinOp::IEqual, bounds_at, first)?;
                        let raise: ArmFn<'_, ()> = Box::new(move |fb, ctx| {
                            let all =
                                fb.group_arith(GroupArith::BitwiseOr, GroupMode::Reduce, bounds)?;
                            let elected = fb.group_elect()?;
                            let write: ArmFn<'_, ()> = Box::new(move |fb, _| {
                                let ptr = word_ptr(fb, bounds_at)?;
                                fb.atomic(
                                    AtomicOp::Or,
                                    ptr,
                                    Scope::Device,
                                    Semantics::RELAXED,
                                    all,
                                )?;
                                Ok(ArmEnd::FallThrough(Vec::new()))
                            });
                            fb.if_(ctx, elected, write, None)?;
                            lc.break_(fb)?;
                            Ok(ArmEnd::Exits)
                        });
                        fb.if_(ctx, mine, raise, None)?;
                        Ok(())
                    });
                    let spec = LoopSpec {
                        unrolling: crate::spirv::ir::Unrolling::DontUnroll,
                        ..LoopSpec::default()
                    };
                    fb.loop_(ctx, spec, body)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                false => Box::new(move |fb, _| {
                    let ptr = word_ptr(fb, bounds_at)?;
                    fb.atomic(AtomicOp::Or, ptr, Scope::Device, Semantics::RELAXED, bounds)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
            };
            fb.if_(&mut ctx, bounded, arm, None)?;
            let counted = fb.binary(BinOp::UGreaterThan, count, none)?;
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let ptr = word_ptr(fb, count_at)?;
                fb.atomic(
                    AtomicOp::IAdd,
                    ptr,
                    Scope::Device,
                    Semantics::RELAXED,
                    count,
                )?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut ctx, counted, arm, None)?;
            // A triangle it does not flag of more pairs than a draw of few copies emits: the wide
            // word its `PrimitiveId` picks.
            let few = ku(&mut fb, FEW_COPIES)?;
            let wide = fb.binary(BinOp::UGreaterThan, count, few)?;
            let fanned = fb.unary(UnOp::LogicalNot, flagged)?;
            let wide = fb.binary(BinOp::LogicalAnd, fanned, wide)?;
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let ptr = word_ptr(fb, wide_at)?;
                let one = ku(fb, 1)?;
                fb.atomic_store(ptr, Scope::Device, Semantics::RELAXED, one)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(&mut ctx, wide, arm, None)?;
        }
        None => {
            let c = Copies {
                table,
                positions,
                own_clips,
                own,
                clips,
                position,
                carried,
                primitive,
                vec4,
                f32_,
            };
            // Copy `i` of the `count`, no more than the stage's copies, its index carried in a
            // `Function` variable.
            let u32_ty = fb.uint_ty(32)?;
            let at = fb.local_var(u32_ty, None)?;
            let at = fb.var_ptr(at)?;
            let zero = ku(&mut fb, 0)?;
            fb.store(at, zero, MemAccess::default())?;
            let most = ku(&mut fb, copies)?;
            let count = fb.glsl(GlslOp::UMin, &[count, most])?;
            let c = &c;
            let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
                let i = fb.load(at, MemAccess::default())?;
                let done = fb.binary(BinOp::UGreaterThanEqual, i, count)?;
                let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
                    lc.break_(fb)?;
                    Ok(ArmEnd::Exits)
                });
                fb.if_(ctx, done, exit, None)?;
                let across = fb.binary(BinOp::UMod, i, xs)?;
                let down = fb.binary(BinOp::UDiv, i, xs)?;
                let sx = fb.binary(BinOp::IAdd, x0, across)?;
                let sy = fb.binary(BinOp::IAdd, y0, down)?;
                emit_copy(fb, c, (sx, sy))?;
                let one = ku(fb, 1)?;
                let next = fb.binary(BinOp::IAdd, i, one)?;
                fb.store(at, next, MemAccess::default())
            });
            let spec = LoopSpec {
                unrolling: crate::spirv::ir::Unrolling::DontUnroll,
                ..LoopSpec::default()
            };
            fb.loop_(&mut ctx, spec, body)?;
        }
    }
    fb.ret(None)?;
    let f = fb.finish()?;
    let vertices = match fan.mode {
        RegionFanOut::Draw | RegionFanOut::DrawFew | RegionFanOut::RegionCopies => 3 * copies,
        RegionFanOut::Probe => 3,
    };
    let modes = vec![
        ExecMode::Triangles,
        ExecMode::OutputTriangleStrip,
        ExecMode::OutputVertices { n: vertices },
        ExecMode::Invocations { n: 1 },
    ];
    mb.entry(ExecModel::Geometry, f, modes)?;
    mb.finalize()
}
