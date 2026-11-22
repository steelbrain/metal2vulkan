//! Emission of the ray tracing strategies.
//!
//! An intersect call becomes:
//!
//! ```text
//! flags, cull_all = flags(settings)
//! if ray_valid(origin, direction, tmin, tmax) && !cull_all {
//!     OpRayQueryInitializeKHR q, as, flags, mask, origin, tmin, direction, tmax
//!     loop { if !OpRayQueryProceedKHR q { break }
//!            if candidate type == triangle { OpRayQueryConfirmIntersectionKHR q } }
//!     record = committed fields; result = committed type != none ? record : 0
//! } else { result = 0 }
//! if hit && multi-level { instance_ids[0] = instance; user_instance_ids[0] = custom index }
//! ```
//!
//! **Flags.** Metal's settings map onto ray flags as follows (Metal enumerations from the
//! `__METAL_RAYTRACING_*` constants):
//!
//! - `triangle_cull_mode` front (1) / back (2) name faces by the intersector's winding, clockwise
//!   (0) by default, which is Vulkan's front face; counterclockwise (1) swaps them. They become
//!   `CullFrontFacingTrianglesKHR` / `CullBackFacingTrianglesKHR`, and the front-facing result is
//!   Vulkan's front face XOR counterclockwise.
//! - `geometry_cull_mode` bits: triangle (1) `SkipTrianglesKHR`, bounding box (2) `SkipAABBsKHR`,
//!   curve (4) nothing (Vulkan structures hold no curves). Both skips cull everything.
//! - `forced_opacity` opaque (1) / non-opaque (2) are `OpaqueKHR` / `NoOpaqueKHR`; with no forced
//!   opacity, `opacity_cull_mode` opaque (1) / non-opaque (2) are `CullOpaqueKHR` /
//!   `CullNoOpaqueKHR`. Vulkan allows one of the four, so a forced opacity decides first and the
//!   cull then either removes everything (it culls the forced opacity) or nothing.
//! - `accept_any_intersection` is `TerminateOnFirstHitKHR`.
//!
//! Everything culled skips the traversal. The assumed geometry type, curve shapes, and identity
//! transforms are promises about the structure, which change no result, so they are not read.
//!
//! **The ray** is valid when its origin and direction are finite, `min_distance >= 0`, and
//! `min_distance <= max_distance` (ordered: a NaN distance fails). Vulkan leaves any other ray
//! undefined (VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06348 to -06351), so an invalid ray skips
//! the traversal and misses.
//!
//! **Candidates.** With a null intersection function table every triangle candidate (a
//! non-opaque triangle) is committed, and a bounding-box candidate, whose hit only an
//! intersection function computes, is left uncommitted. Through a table argument, Apple's
//! `MTLIntersectionFunctionTable.h` decides: "Ray/triangle intersections are always accepted if an
//! intersection function table is not provided", and with a table, an entry never set ignores the
//! intersection "if a call to the function at the given index would be required". A triangle
//! candidate is therefore committed when the table is unbound (its present word is 0: a nil
//! table is no table) or its entry is the opaque triangle function
//! (`FunctionHandle::RESERVED_WORD`, the table layout `pipeline::table`'s); an entry never set
//! (`FunctionHandle::NULL_WORD`), an entry at or past the table's size (undefined in Metal,
//! which does not check the index), and any other word (no intersection function is linked,
//! `Entry::intersection_candidates`) leave it uncommitted, as they leave every bounding box. The entry's index is the candidate instance's
//! shader-binding-table record offset, which holds Metal's table offset.
//!
//! **Result.** On a hit the fields are the committed intersection's: type (Vulkan's committed
//! type, none 0 and triangle 1 as in Metal), `T`, primitive, geometry, instance index, instance
//! custom index (Metal's user instance id), barycentrics, front face, and the world-to-object and
//! object-to-world columns. A multi-level intersector's instance count is 1 on a hit (a Vulkan
//! top-level structure holds one level of instances), and the hit's instance index and custom index
//! are stored into the first element of its id arrays. On a miss every field is zero and the arrays
//! are left as they are.
//!
//! **Recorded.** The hit distance and barycentrics are the device's arithmetic, and the forced
//! opacity order, the zero miss (and a multi-level miss's unwritten id arrays), the invalid-ray
//! skip, the uncommitted bounding boxes, and object-space facing are conventions no Metal
//! reading has settled yet. Selection records `PrecisionNote::DeviceRayIntersection` for every
//! lowered call, so validation compares its observations byte for byte and explains a difference
//! rather than accepting one.

use crate::air::canon::function_table::{FIRST_ENTRY_WORD, PRESENT_WORD, SIZE_WORD};
use crate::air::intrinsics::{AirIntrinsic, Instancing, Intersect, IntersectRole, ResultField};
use crate::air::ir::{InstId, InstKind, Value};
use crate::lower::access::local;
use crate::lower::arith;
use crate::lower::emit::EmitCx;
use crate::lower::raytracing::{structure, table, IntersectPlan, RayTracingStrategy};
use crate::memory::views::{self, View};
use crate::pipeline::sym::PushField;
use crate::pipeline::table::FunctionHandle;
use crate::spirv::builder::constructs::{ArmEnd, LoopCtx, LoopSpec};
use crate::spirv::builder::ops::ray::RayInit;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, GlslOp, RayQueryValue, Type as SType, UnOp};

/// `CullBackFacingTrianglesKHR`.
const CULL_BACK: u32 = 0x10;
/// `CullFrontFacingTrianglesKHR`.
const CULL_FRONT: u32 = 0x20;
/// `SkipTrianglesKHR`.
const SKIP_TRIANGLES: u32 = 0x100;
/// `SkipAABBsKHR`.
const SKIP_AABBS: u32 = 0x200;
/// `OpaqueKHR`.
const OPAQUE: u32 = 0x1;
/// `NoOpaqueKHR`.
const NO_OPAQUE: u32 = 0x2;
/// `CullOpaqueKHR`.
const CULL_OPAQUE: u32 = 0x40;
/// `CullNoOpaqueKHR`.
const CULL_NO_OPAQUE: u32 = 0x80;
/// `TerminateOnFirstHitKHR`.
const TERMINATE_ON_FIRST_HIT: u32 = 0x4;

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &RayTracingStrategy,
) -> BResult<()> {
    match s {
        RayTracingStrategy::NullTable
        | RayTracingStrategy::PrimitiveData
        | RayTracingStrategy::UnreadStore => Ok(()),
        RayTracingStrategy::Intersect(p) => intersect(cx, fb, at, p),
        // The binder's count of the bound structure.
        RayTracingStrategy::InstanceCount(slot) => {
            let x = cx.push_field(fb, PushField::InstanceCount(*slot))?;
            define(cx, at, x)
        }
        RayTracingStrategy::Field(member) => {
            let agg = match cx.body.insts.get(at).map(|i| &i.kind) {
                Some(InstKind::ExtractValue { agg, .. }) => *agg,
                _ => return Err(cx.fail("a ray tracing field that is no extractvalue")),
            };
            let record = operand(cx, fb, agg)?;
            let x = fb.composite_extract(record, &[*member])?;
            define(cx, at, x)
        }
    }
}

fn define(cx: &mut EmitCx<'_>, at: InstId, x: SValue) -> BResult<()> {
    match cx.plan.values.inst(at) {
        Some(v) => {
            cx.define(v, x);
            Ok(())
        }
        None => Err(cx.fail(format!("{at:?} is out of the value range"))),
    }
}

fn operand(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<SValue> {
    match cx.plan.values.id(v) {
        Some(id) => cx.operand(fb, id),
        None => Err(cx.fail("a ray tracing operand out of the value range")),
    }
}

fn u32c(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn bool_c(fb: &mut FunctionBuilder<'_>, b: bool) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(b)),
    })?;
    fb.module().const_value(c)
}

fn f32c(fb: &mut FunctionBuilder<'_>, bits: u32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(bits)),
    })?;
    fb.module().const_value(c)
}

/// `a == k` for a `u32` value `a`.
fn is(fb: &mut FunctionBuilder<'_>, a: SValue, k: u32) -> BResult<SValue> {
    let k = u32c(fb, k)?;
    fb.binary(BinOp::IEqual, a, k)
}

/// `c ? bits : 0`.
fn bit_if(fb: &mut FunctionBuilder<'_>, c: SValue, bits: u32) -> BResult<SValue> {
    let (on, off) = (u32c(fb, bits)?, u32c(fb, 0)?);
    fb.select(c, on, off)
}

/// The intersector's settings, as lowered values: `u32` enumerations and a `bool`.
struct Settings {
    winding: SValue,
    triangle_cull: SValue,
    geometry_cull: SValue,
    opacity_cull: SValue,
    forced_opacity: SValue,
    accept_any: SValue,
}

/// The ray flags the settings select, and whether they cull every primitive (module docs).
fn flags(fb: &mut FunctionBuilder<'_>, s: &Settings) -> BResult<(SValue, SValue, SValue)> {
    let ccw = is(fb, s.winding, 1)?;
    let metal_front = is(fb, s.triangle_cull, 1)?;
    let metal_back = is(fb, s.triangle_cull, 2)?;
    let vk_front = fb.select(ccw, metal_back, metal_front)?;
    let vk_back = fb.select(ccw, metal_front, metal_back)?;
    let one = u32c(fb, 1)?;
    let two = u32c(fb, 2)?;
    let zero = u32c(fb, 0)?;
    let tri_bit = fb.binary(BinOp::BitwiseAnd, s.geometry_cull, one)?;
    let skip_tri = fb.binary(BinOp::INotEqual, tri_bit, zero)?;
    let box_bit = fb.binary(BinOp::BitwiseAnd, s.geometry_cull, two)?;
    let skip_box = fb.binary(BinOp::INotEqual, box_bit, zero)?;
    // Facing culls are meaningless (and invalid, VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06890)
    // with triangles skipped.
    let front = bit_if(fb, vk_front, CULL_FRONT)?;
    let back = bit_if(fb, vk_back, CULL_BACK)?;
    let facing = fb.binary(BinOp::BitwiseOr, front, back)?;
    let facing = fb.select(skip_tri, zero, facing)?;
    let st = bit_if(fb, skip_tri, SKIP_TRIANGLES)?;
    let sb = bit_if(fb, skip_box, SKIP_AABBS)?;
    let skips = fb.binary(BinOp::BitwiseOr, st, sb)?;
    let forced_opaque = is(fb, s.forced_opacity, 1)?;
    let forced_non_opaque = is(fb, s.forced_opacity, 2)?;
    let cull_opaque = is(fb, s.opacity_cull, 1)?;
    let cull_non_opaque = is(fb, s.opacity_cull, 2)?;
    let co = bit_if(fb, cull_opaque, CULL_OPAQUE)?;
    let cn = bit_if(fb, cull_non_opaque, CULL_NO_OPAQUE)?;
    let culled = fb.binary(BinOp::BitwiseOr, co, cn)?;
    let no = u32c(fb, NO_OPAQUE)?;
    let opacity = fb.select(forced_non_opaque, no, culled)?;
    let op = u32c(fb, OPAQUE)?;
    let opacity = fb.select(forced_opaque, op, opacity)?;
    let term = bit_if(fb, s.accept_any, TERMINATE_ON_FIRST_HIT)?;
    let mut word = facing;
    for part in [skips, opacity, term] {
        word = fb.binary(BinOp::BitwiseOr, word, part)?;
    }
    let all_skipped = fb.binary(BinOp::LogicalAnd, skip_tri, skip_box)?;
    let opaque_culled = fb.binary(BinOp::LogicalAnd, forced_opaque, cull_opaque)?;
    let non_opaque_culled = fb.binary(BinOp::LogicalAnd, forced_non_opaque, cull_non_opaque)?;
    let cull_all = fb.binary(BinOp::LogicalOr, all_skipped, opaque_culled)?;
    let cull_all = fb.binary(BinOp::LogicalOr, cull_all, non_opaque_culled)?;
    Ok((word, cull_all, ccw))
}

/// Whether the ray is one Vulkan defines (module docs).
fn ray_valid(
    fb: &mut FunctionBuilder<'_>,
    origin: SValue,
    direction: SValue,
    tmin: SValue,
    tmax: SValue,
) -> BResult<SValue> {
    let f32_ = fb.module().ty(SType::Float { width: 32 })?;
    let vec3 = fb.module().ty(SType::Vector { elem: f32_, len: 3 })?;
    let inf = fb.module().constant(Constant::Scalar {
        ty: f32_,
        bits: ConstBits(u64::from(f32::INFINITY.to_bits())),
    })?;
    let inf3 = fb.module().constant(Constant::Composite {
        ty: vec3,
        parts: vec![inf; 3].into(),
    })?;
    let inf3 = fb.module().const_value(inf3)?;
    let mut valid = None;
    for v in [origin, direction] {
        let a = fb.glsl(GlslOp::FAbs, &[v])?;
        let lanes = fb.binary(BinOp::FOrdLessThan, a, inf3)?;
        let finite = fb.unary(UnOp::All, lanes)?;
        valid = Some(match valid {
            None => finite,
            Some(p) => fb.binary(BinOp::LogicalAnd, p, finite)?,
        });
    }
    let zero = f32c(fb, 0)?;
    let nonnegative = fb.binary(BinOp::FOrdGreaterThanEqual, tmin, zero)?;
    let ordered = fb.binary(BinOp::FOrdLessThanEqual, tmin, tmax)?;
    let interval = fb.binary(BinOp::LogicalAnd, nonnegative, ordered)?;
    match valid {
        Some(p) => fb.binary(BinOp::LogicalAnd, p, interval),
        None => Ok(interval),
    }
}

/// The operands of one intersect call.
struct Call<'a> {
    i: Intersect,
    args: &'a [Value],
}

impl Call<'_> {
    fn arg(&self, role: IntersectRole) -> Option<Value> {
        self.i.index(role).and_then(|k| self.args.get(k)).copied()
    }
}

fn intersect(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    p: &IntersectPlan,
) -> BResult<()> {
    let body = cx.body;
    let call = match body.insts.get(at).map(|i| &i.kind) {
        Some(InstKind::Call(c)) => match &c.callee {
            crate::air::ir::Callee::Air {
                op: AirIntrinsic::Intersect(i),
                ..
            } => Call {
                i: *i,
                args: c.args.as_ref(),
            },
            _ => return Err(cx.fail("an intersect strategy on another call")),
        },
        _ => return Err(cx.fail("an intersect strategy on no call")),
    };
    let get = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, role| match call.arg(role) {
        Some(v) => operand(cx, fb, v),
        None => Err(cx.fail(format!("an intersect call without its {role:?} operand"))),
    };
    let origin = get(cx, fb, IntersectRole::Origin)?;
    let direction = get(cx, fb, IntersectRole::Direction)?;
    let tmin = get(cx, fb, IntersectRole::MinDistance)?;
    let tmax = get(cx, fb, IntersectRole::MaxDistance)?;
    // A primitive structure is bound in one instance of mask 0xFF.
    let cull_mask = match p.tags.instancing {
        Instancing::None => u32c(fb, 0xFF)?,
        _ => get(cx, fb, IntersectRole::Mask)?,
    };
    let settings = Settings {
        winding: get(cx, fb, IntersectRole::Winding)?,
        triangle_cull: get(cx, fb, IntersectRole::TriangleCull)?,
        geometry_cull: get(cx, fb, IntersectRole::GeometryCull)?,
        opacity_cull: get(cx, fb, IntersectRole::OpacityCull)?,
        forced_opacity: get(cx, fb, IntersectRole::ForcedOpacity)?,
        accept_any: get(cx, fb, IntersectRole::AcceptAny)?,
    };
    let accel = structure(cx, fb, p.slot)?;
    let table = match p.table {
        Some(slot) => Some(table(cx, fb, slot)?),
        None => None,
    };
    let (flags, cull_all, ccw) = flags(fb, &settings)?;
    let valid = ray_valid(fb, origin, direction, tmin, tmax)?;
    let traced = fb.unary(UnOp::LogicalNot, cull_all)?;
    let go = fb.binary(BinOp::LogicalAnd, valid, traced)?;
    let Some(record_ty) = arith::spirv_type(fb.module(), &p.record)? else {
        return Err(cx.fail("an intersect record without a lowered type"));
    };
    let zero = fb.module().constant(Constant::Null { ty: record_ty })?;
    let zero = fb.module().const_value(zero)?;
    let query = fb.ray_query()?;
    let ray = RayInit {
        accel,
        flags,
        cull_mask,
        origin,
        tmin,
        direction,
        tmax,
    };
    let fields: Vec<(ResultField, TypeId)> = {
        let parts: &[DemoteTy] = match &p.record {
            DemoteTy::Composite(parts) => parts,
            _ => return Err(cx.fail("an intersect record that is no composite")),
        };
        let mut out = Vec::with_capacity(parts.len());
        let shape = p.tags.result();
        for (f, part) in shape
            .fields()
            .filter(|f| *f != ResultField::PrimitiveData)
            .zip(parts.iter())
        {
            let Some(ty) = arith::spirv_type(fb.module(), part)? else {
                return Err(cx.fail("an intersect record member without a type"));
            };
            out.push((f, ty));
        }
        out
    };
    // A multi-level intersector also carries out the hit and its instance's two ids.
    let levels = p.levels.is_some();
    let (no, nought) = (bool_c(fb, false)?, u32c(fb, 0)?);
    let mut fault = None;
    let merged = fb.if_(
        &mut fault,
        go,
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, fault: &mut Option<ResultField>| {
                fb.ray_query_initialize(query, ray)?;
                traverse(fb, query, table)?;
                let (record, hit) = committed(fb, query, &fields, ccw, record_ty, fault)?;
                let mut out = vec![record];
                if levels {
                    let id = fb.ray_query_get(query, RayQueryValue::InstanceId, true)?;
                    let user = fb.ray_query_get(query, RayQueryValue::InstanceCustomIndex, true)?;
                    out.extend([hit, id, user]);
                }
                Ok(ArmEnd::FallThrough(out))
            },
        ),
        Some(Box::new(
            move |_: &mut FunctionBuilder<'_>, _: &mut Option<ResultField>| {
                let mut out = vec![zero];
                if levels {
                    out.extend([no, nought, nought]);
                }
                Ok(ArmEnd::FallThrough(out))
            },
        )),
    )?;
    if let Some(f) = fault {
        return Err(cx.fail(format!(
            "an intersect record field {f:?} this family refuses"
        )));
    }
    match merged.0.first() {
        Some(x) => define(cx, at, *x)?,
        None => return Err(cx.fail("an intersect traversal without a result")),
    }
    let Some(ids) = p.levels.as_deref() else {
        return Ok(());
    };
    let [_, hit, id, user] = merged.0.as_slice() else {
        return Err(cx.fail("a multi-level traversal without its ids"));
    };
    let (hit, id, user) = (*hit, *id, *user);
    // A hit traversed one instance (the binder builds one level): Metal writes its ids at the first
    // element of each array; a miss writes none.
    fb.if_(
        cx,
        hit,
        Box::new(move |fb: &mut FunctionBuilder<'_>, cx: &mut EmitCx<'_>| {
            local::emit_intrinsic_store(cx, fb, &ids.instance, id)?;
            local::emit_intrinsic_store(cx, fb, &ids.user, user)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        }),
        None,
    )?;
    Ok(())
}

/// Word `word` of `view`, or 0 at or past its bound range `len` (in words).
fn contained_word(
    fb: &mut FunctionBuilder<'_>,
    view: View,
    len: SValue,
    word: SValue,
) -> BResult<SValue> {
    let inb = fb.binary(BinOp::ULessThan, word, len)?;
    let zero = u32c(fb, 0)?;
    let at = fb.select(inb, word, zero)?;
    let w = views::read_word(fb, view, at)?;
    fb.select(inb, w, zero)
}

/// Whether the intersection function table (`view`, its words starting at word `base`) accepts
/// the query's candidate triangle: [`entry_accepts`] at the candidate instance's
/// shader-binding-table record offset.
fn table_accepts(
    fb: &mut FunctionBuilder<'_>,
    query: SValue,
    table: (View, SValue),
) -> BResult<SValue> {
    let index = fb.ray_query_get(
        query,
        RayQueryValue::InstanceShaderBindingTableRecordOffset,
        false,
    )?;
    entry_accepts(fb, table, index)
}

/// Whether the table (`view`, its words starting at word `base`, laid out as `pipeline::table`
/// says) accepts a candidate triangle at entry `index`: the table is unbound (its present word
/// is [`FunctionHandle::NULL_WORD`], as in a zero-filled neutral buffer), or `index` is below
/// its size and the entry's word is [`FunctionHandle::RESERVED_WORD`], the opaque triangle
/// function (module docs). Every word is read within the bound range; an entry past the size or
/// the range reads as null.
fn entry_accepts(
    fb: &mut FunctionBuilder<'_>,
    (view, base): (View, SValue),
    index: SValue,
) -> BResult<SValue> {
    let global = fb.module().global_value(view.global)?;
    let len = fb.array_length(global, 0)?;
    let at_present = word_after(fb, base, PRESENT_WORD)?;
    let present = contained_word(fb, view, len, at_present)?;
    let at_size = word_after(fb, base, SIZE_WORD)?;
    let size = contained_word(fb, view, len, at_size)?;
    let in_table = fb.binary(BinOp::ULessThan, index, size)?;
    // Below the range too, so `first + index` cannot wrap back into it.
    let in_range = fb.binary(BinOp::ULessThan, index, len)?;
    let in_table = fb.binary(BinOp::LogicalAnd, in_table, in_range)?;
    let first = word_after(fb, base, FIRST_ENTRY_WORD)?;
    let at_entry = fb.binary(BinOp::IAdd, first, index)?;
    let entry = contained_word(fb, view, len, at_entry)?;
    let null = u32c(fb, FunctionHandle::NULL_WORD)?;
    let entry = fb.select(in_table, entry, null)?;
    let unbound = is(fb, present, FunctionHandle::NULL_WORD)?;
    let opaque = is(fb, entry, FunctionHandle::RESERVED_WORD)?;
    fb.binary(BinOp::LogicalOr, unbound, opaque)
}

/// Word `base + word`.
fn word_after(fb: &mut FunctionBuilder<'_>, base: SValue, word: u32) -> BResult<SValue> {
    let word = u32c(fb, word)?;
    fb.binary(BinOp::IAdd, base, word)
}

/// Proceed until traversal completes, committing every triangle candidate the table accepts
/// (every one through the null table, `table` `None`).
fn traverse(
    fb: &mut FunctionBuilder<'_>,
    query: SValue,
    table: Option<(View, SValue)>,
) -> BResult<()> {
    fb.loop_(
        &mut (),
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                let more = fb.ray_query_proceed(query)?;
                let done = fb.unary(UnOp::LogicalNot, more)?;
                fb.if_(
                    ctx,
                    done,
                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                // Candidate type 0 is a (non-opaque) triangle; 1, a bounding box, stays
                // uncommitted without an intersection function.
                let kind = fb.ray_query_get(query, RayQueryValue::Type, false)?;
                let triangle = is(fb, kind, 0)?;
                let commit = match table {
                    Some(t) => {
                        let accepts = table_accepts(fb, query, t)?;
                        fb.binary(BinOp::LogicalAnd, triangle, accepts)?
                    }
                    None => triangle,
                };
                fb.if_(
                    ctx,
                    commit,
                    Box::new(move |fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        fb.ray_query_confirm(query)?;
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }),
                    None,
                )?;
                Ok(())
            },
        ),
    )
}

/// The record of the committed intersection's `fields` (each with its member's lowered type), or
/// zero when nothing was committed, and whether something was.
fn committed(
    fb: &mut FunctionBuilder<'_>,
    query: SValue,
    fields: &[(ResultField, TypeId)],
    ccw: SValue,
    record_ty: TypeId,
    fault: &mut Option<ResultField>,
) -> BResult<(SValue, SValue)> {
    let get = |fb: &mut FunctionBuilder<'_>, v| fb.ray_query_get(query, v, true);
    let mut parts = Vec::with_capacity(fields.len());
    let mut kind = None;
    let mut transforms = [None, None];
    for (f, member) in fields {
        let x = match f {
            ResultField::Type => {
                let k = get(fb, RayQueryValue::Type)?;
                kind = Some(k);
                k
            }
            ResultField::Distance => get(fb, RayQueryValue::T)?,
            ResultField::PrimitiveId => get(fb, RayQueryValue::PrimitiveIndex)?,
            ResultField::GeometryId => get(fb, RayQueryValue::GeometryIndex)?,
            ResultField::InstanceId => get(fb, RayQueryValue::InstanceId)?,
            ResultField::UserInstanceId => get(fb, RayQueryValue::InstanceCustomIndex)?,
            ResultField::Barycentrics => get(fb, RayQueryValue::Barycentrics)?,
            ResultField::FrontFacing => {
                let front = get(fb, RayQueryValue::FrontFace)?;
                fb.binary(BinOp::LogicalNotEqual, front, ccw)?
            }
            ResultField::WorldToObject(n) | ResultField::ObjectToWorld(n) => {
                let (slot, value) = match f {
                    ResultField::WorldToObject(_) => (0, RayQueryValue::WorldToObject),
                    _ => (1, RayQueryValue::ObjectToWorld),
                };
                let m = match transforms.get(slot).copied().flatten() {
                    Some(m) => m,
                    None => {
                        let m = get(fb, value)?;
                        if let Some(t) = transforms.get_mut(slot) {
                            *t = Some(m);
                        }
                        m
                    }
                };
                fb.composite_extract(m, &[u32::from(*n)])?
            }
            // A hit traversed the one instance level a Vulkan top-level structure holds; a miss is
            // zero with every other field.
            ResultField::InstanceCount => {
                let one = fb.module().constant(Constant::Scalar {
                    ty: *member,
                    bits: ConstBits(1),
                })?;
                fb.module().const_value(one)?
            }
            ResultField::PrimitiveData | ResultField::CurveParameter => {
                *fault = Some(*f);
                let zero = fb.module().constant(Constant::Null { ty: record_ty })?;
                let zero = fb.module().const_value(zero)?;
                let no = bool_c(fb, false)?;
                return Ok((zero, no));
            }
        };
        parts.push(x);
    }
    let record = fb.composite_construct(record_ty, &parts)?;
    let Some(kind) = kind else {
        *fault = Some(ResultField::Type);
        let no = bool_c(fb, false)?;
        return Ok((record, no));
    };
    let none = u32c(fb, 0)?;
    let hit = fb.binary(BinOp::INotEqual, kind, none)?;
    let zero = fb.module().constant(Constant::Null { ty: record_ty })?;
    let zero = fb.module().const_value(zero)?;
    Ok((fb.select(hit, record, zero)?, hit))
}

#[cfg(test)]
mod tests;
