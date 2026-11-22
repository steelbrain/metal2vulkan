//! Ray intersection through `VK_KHR_ray_query`.
//!
//! An acceleration-structure argument (`primitive_acceleration_structure`,
//! `instance_acceleration_structure`) at `[[buffer(n)]]` is an `OpTypeAccelerationStructureKHR`
//! descriptor bound at that slot (`ResourceKind::AccelerationStructure`); the binder binds a
//! top-level structure there, wrapping a primitive-level one in one identity instance.
//!
//! An `air.intersect` call ([`Intersect`]) is one complete traversal: a ray query initialized
//! with the ray and the flags the intersector's settings select, `OpRayQueryProceedKHR` until
//! traversal completes, committing the triangle candidates its intersection function table
//! accepts, and the committed intersection read into the result's fields (`emit`). The call's
//! value is a record of the result's fields without its primitive-data pointer
//! ([`result_repr`]); each `extractvalue` of it is a [`RayTracingStrategy::Field`].
//! `air.get_null_intersection_function_table` has no value: its only readers are intersect
//! calls, which read it as "no intersection functions".
//!
//! The primitive-data pointer points into the acceleration structure, at data Metal copied there
//! from the geometry's primitive-data buffer; no Vulkan structure holds such data. Memory gives it
//! the call's opaque root ([`primitive_data_root`]), and this family owns it: its `extractvalue`,
//! from the result or from an aggregate it was inserted into, has no value
//! ([`RayTracingStrategy::PrimitiveData`]). It may be moved (inserted into an aggregate,
//! extracted again, carried as a member of a member-wise phi, which holds zeros for it) and stored
//! into thread memory whose stored bytes nothing reads ([`plan_unread_stores`]): such a store
//! writes nothing ([`RayTracingStrategy::UnreadStore`]). Every other reader (a load or store
//! through it, a compare, a conversion, a merge, any other store) reads the data or its address,
//! and is refused ([`primitive_data_reader`]).
//!
//! [`primitive_data_root`]: crate::memory::provenance::primitive_data_root
//!
//! A multi-level intersector (`intersector<instancing, max_levels<n>>`) traverses a Vulkan
//! top-level structure, which holds one level of instances: the binder refuses a Metal instance
//! of an instance structure (`bind::acceleration`). So a hit traversed exactly one instance: its
//! `instance_count` is 1, and the call writes that instance's id and user id into the first
//! element of the instance-id and user-instance-id arrays it is given ([`LevelIds`]); a miss's
//! count is 0, with every other field, and it writes neither.
//!
//! **The instance count**. `air.get_instance_count_instance_acceleration_structure`
//! (`get_instance_count()`) of an entry argument's instance structure reads the push field
//! `PushField::InstanceCount` of its slot (`RuntimeFact::AccelerationStructureInstanceCount`):
//! no Vulkan instruction reads a structure's instance count, and the consumer that holds the
//! Metal structure knows it, so the binder writes it (`StageTable::acceleration_structures`, 0
//! for an unbound slot). A structure argument whose every reader is such a query needs no
//! descriptor, so no ray tracing feature: its count is exact on every device wherever the
//! consumer knows the count when it records the command (`bind::BoundAccelerationStructure`).
//! The other queries of a structure (its instances' primitive structures, its instance
//! descriptors) stay refused.
//!
//! An `intersection_function_table` argument at `[[buffer(n)]]` is a read-only buffer of `u32`
//! words at that slot, the layout of a visible-function table (`pipeline::table`): word 0 marks a
//! bound table, word 1 is its size, and word `2 + i` entry `i`, which is `0` for an entry never set
//! and `1` (`FunctionHandle::RESERVED_WORD`) for Metal's opaque triangle intersection function
//! (`setOpaqueTriangleIntersectionFunctionWithSignature:`). Its only readers are intersect calls.
//! The candidate's entry is the one at its instance's shader-binding-table record offset, where the
//! binder carries Metal's table offset (`bind::acceleration`). An intersection function the
//! pipeline links would be a case of the entry's switch; linking one is not implemented, so an
//! intersect call through a table of a stage that may link one is refused
//! ([`Entry::intersection_candidates`]).
//!
//! [`Entry::intersection_candidates`]: crate::air::canon::Entry::intersection_candidates
//!
//! Every lowered intersect call records `PrecisionNote::DeviceRayIntersection`: the device
//! computes the hit, and the conventions no Metal reading has settled yet are this lowering's.
//!
//! Refused, each a typed `Unsupported`: an intersect call through an intersection function table
//! of a stage that may link an intersection function, through a table that is not one entry
//! argument, or through an intersection function buffer, and a table argument read otherwise
//! (a `set_buffer`, refused at that reader) (`IntersectionFunctionTable`); motion
//! (`NotRepresentable`: a ray query has no time, which decides before the rest of the call);
//! curves; a structure that is not an entry argument (an argument-buffer member) or an array of
//! them; a result used other than by `extractvalue`, or its primitive-data pointer read; and a
//! profile without `rayQuery`, `accelerationStructure`, and `bufferDeviceAddress`. A multi-level
//! intersector whose `max_levels` is computed, or a constant outside the 2 to 32 Metal's compiler
//! accepts, is not Metal's (`InvalidInput`). An instance count of a structure that is not an entry
//! argument is refused (`NotImplemented`); the count itself needs no device feature.

mod emit;

pub use emit::emit;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, AccelerationStructureKind, ObjectArg, ParamBinding, ParamRole};
use crate::air::error::IntrinsicError;
use crate::air::intrinsics::LlvmIntrinsic;
use crate::air::intrinsics::{
    AccelerationStructure, AirIntrinsic, FunctionTable, Instancing, Intersect, IntersectRole,
    IntersectTags, ResultField,
};
use crate::air::ir::{
    AddrSpace, ArgIdx, Body, Callee, CastOp, Const, FuncId, Inst, InstId, InstKind, Type, TypeId,
    Value,
};
use crate::budget::{tree_bytes, vec_bytes, Growth, Meter};
use crate::device::requirements::VkFeature;
use crate::error::{
    Capability, Error, Feature, InputError, InputErrorKind, InternalError, InternalKind, Location,
    LowerFamily, Unsupported,
};
use crate::lower::access::local::{self, IntrinsicStore};
use crate::lower::arith;
use crate::lower::emit::EmitCx;
use crate::lower::select::{decl_name, param_binding, FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::memory::provenance::primitive_data_root;
use crate::memory::provenance::{OffsetRange, Root};
use crate::memory::views::{self, View};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{
    AccessSet, BufferForm, BufferKind, BufferSlot, SlotInterface, Window,
};
use crate::pipeline::sym::{
    AccelerationStructureLevel, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, GlobalId, Value as SValue};
use crate::spirv::ir::{ExecMode, MemAccess, StorageClass, Type as SType};

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::RayTracing)).into()
}

/// The family's part of the per-function plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RayTracingPlan {
    /// The acceleration structures the function binds, by Metal buffer slot.
    pub structures: BTreeMap<u32, AccelerationStructureLevel>,
    /// The intersection function tables the function reads, by Metal buffer slot.
    pub tables: BTreeSet<u32>,
    /// The readers of every intersect call, null table, and intersection function table
    /// argument in the body, found once.
    readers: Option<BTreeMap<Value, Vec<InstId>>>,
    /// Whether the body calls `air.intersect` (found by [`plan_unread_stores`]): without a call
    /// no pointer is primitive data.
    intersects: bool,
    /// The stores of a primitive-data pointer whose bytes nothing reads ([`plan_unread_stores`]).
    pub unread: BTreeSet<InstId>,
}

/// The strategies of the family's instructions.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RayTracingStrategy {
    /// `air.get_null_intersection_function_table`: no value.
    NullTable,
    /// An `air.intersect` call: one ray query traversal.
    Intersect(Box<IntersectPlan>),
    /// `extractvalue` of an intersect result: this member of the lowered record.
    Field(u32),
    /// `extractvalue` of an intersection's primitive-data pointer, from the result or from an
    /// aggregate it was inserted into: no value. Its readers only move it, or store it where
    /// nothing reads it back.
    PrimitiveData,
    /// A store of a primitive-data pointer into thread memory whose stored bytes nothing reads
    /// ([`plan_unread_stores`]): nothing.
    UnreadStore,
    /// `air.get_instance_count_instance_acceleration_structure` of the structure at this buffer
    /// slot: the binder's push field.
    InstanceCount(SlotRef),
}

/// How one intersect call lowers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntersectPlan {
    /// The Metal buffer slot of its acceleration structure.
    pub slot: u32,
    pub tags: IntersectTags,
    /// The lowered record: the result's fields, in order, without the primitive-data pointer.
    pub record: DemoteTy,
    /// The Metal buffer slot of the intersection function table it reads, or `None` for the
    /// null table.
    pub table: Option<u32>,
    /// A multi-level intersector's instance-id stores, `None` for any other.
    pub levels: Option<Box<LevelIds>>,
}

/// Where a multi-level intersector writes the ids of the instances a hit traversed: the first
/// element of its instance-id array and of its user-instance-id array, each a `u32` store through
/// the call's thread pointer. A Vulkan top-level structure holds one level of instances (the binder
/// refuses a Metal instance of an instance structure, `bind::acceleration`), so a hit traverses
/// exactly one instance: Metal's `instance_count` is 1 and its one id is the first element's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelIds {
    pub instance: IntrinsicStore,
    pub user: IntrinsicStore,
}

/// An upper estimate of an intersect call's words without its record's fields: the settings'
/// flags, the ray check, the query variable, the traversal loop, and the two arms and their
/// merge. Measured at 353 words.
const INTERSECT_WORDS: u32 = 380;
/// An upper estimate of one record field's words: its committed get, its operand of the record,
/// and its share of the front-face comparison or of a transform's one get (at most 11, the front
/// face; measured at 125 words for the 16 fields of an instanced world-space triangle record).
const INTERSECT_FIELD_WORDS: u32 = 12;
/// An upper estimate of reading a candidate's intersection function table entry: the bind
/// residual, the bound range, three contained word reads, and the acceptance. Measured:
/// `kernel_instance_as_intersect.ll` emits 188 words more than its null-table variant on
/// `lavapipe`, the table's descriptor included.
const INTERSECT_TABLE_WORDS: u32 = 170;
/// An upper estimate of a multi-level intersector's id writes besides their two stores: the two
/// committed gets, their merge with the miss, and the hit's branch.
const LEVEL_IDS_WORDS: u32 = 48;
/// An upper estimate of an instance count read: the member index constant, the access chain, and
/// the load of the push field (or its argument-table read).
const INSTANCE_COUNT_WORDS: u32 = 16;

impl IntersectPlan {
    /// An upper estimate of the words the call emits.
    fn words(&self) -> u32 {
        let fields = match &self.record {
            DemoteTy::Composite(parts) => u32::try_from(parts.len()).unwrap_or(u32::MAX),
            _ => 1,
        };
        let table = if self.table.is_some() {
            INTERSECT_TABLE_WORDS
        } else {
            0
        };
        // The two ids' gets and stores, under the hit's branch.
        let levels = self.levels.as_ref().map_or(0, |l| {
            LEVEL_IDS_WORDS
                .saturating_add(l.instance.words())
                .saturating_add(l.user.words())
        });
        INTERSECT_WORDS
            .saturating_add(INTERSECT_FIELD_WORDS.saturating_mul(fields))
            .saturating_add(table)
            .saturating_add(levels)
    }
}

impl FamilyStrategy for RayTracingStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            RayTracingStrategy::NullTable => "raytracing.null_table",
            RayTracingStrategy::Intersect(_) => "raytracing.intersect",
            RayTracingStrategy::Field(_) => "raytracing.field",
            RayTracingStrategy::PrimitiveData => "raytracing.primitive_data",
            RayTracingStrategy::UnreadStore => "raytracing.unread_store",
            RayTracingStrategy::InstanceCount(_) => "raytracing.instance_count",
        })
    }

    fn words(&self) -> u32 {
        match self {
            RayTracingStrategy::NullTable
            | RayTracingStrategy::PrimitiveData
            | RayTracingStrategy::UnreadStore => 0,
            RayTracingStrategy::Intersect(p) => p.words(),
            RayTracingStrategy::Field(_) => 5,
            // The push block's member pointer and its load.
            RayTracingStrategy::InstanceCount(_) => INSTANCE_COUNT_WORDS,
        }
    }

    /// The null table and a primitive-data pointer have no value: their readers ignore them.
    /// A store has none either.
    fn yields_value(&self) -> bool {
        !matches!(
            self,
            RayTracingStrategy::NullTable
                | RayTracingStrategy::PrimitiveData
                | RayTracingStrategy::UnreadStore
        )
    }
}

/// The family's module-scope objects: each acceleration structure's descriptor variable and each
/// intersection function table's word view.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    pub structures: BTreeMap<u32, GlobalId>,
    pub tables: BTreeMap<u32, View>,
}

/// The buffer slot of the intersection function table at Metal buffer `slot` of `stage`.
fn table_slot(stage: Stage, slot: u32) -> SlotRef {
    SlotRef {
        stage,
        class: SlotClass::Buffer,
        index: slot,
    }
}

/// The artifact's statement of every intersection function table the function reads: a
/// read-only `Whole` buffer slot of its Metal buffer index, a storage view of its words, which
/// the binder binds like any other buffer slot of the stage.
pub fn slot_interfaces(plan: &Plan) -> impl Iterator<Item = (SlotRef, SlotInterface)> + '_ {
    plan.families.raytracing.tables.iter().map(|&slot| {
        (
            table_slot(plan.stage, slot),
            SlotInterface::Buffer(BufferSlot {
                access: AccessSet::READ,
                window: Window::Whole,
                form: BufferForm::Descriptor,
                kind: BufferKind::IntersectionFunctionTable,
            }),
        )
    })
}

/// Declare the plan's acceleration-structure descriptors and table views.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let mut d = Declared::default();
    for &slot in &plan.families.raytracing.tables {
        let view = views::declare(mb, table_slot(plan.stage, slot))?;
        d.tables.insert(slot, view);
    }
    for (slot, level) in &plan.families.raytracing.structures {
        let ty = mb.ty(SType::AccelerationStructure)?;
        let g = mb.resource(
            ResourceSlot {
                stage: plan.stage,
                kind: ResourceKind::AccelerationStructure(*level),
                index: *slot,
            },
            StorageClass::UniformConstant,
            ty,
        )?;
        d.structures.insert(*slot, g);
    }
    Ok(d)
}

/// The Metal buffer slot of acceleration-structure argument `b`: a single argument, not an
/// array of them.
fn slot_of(b: &ParamBinding) -> Result<(u32, AccelerationStructureLevel), Error> {
    let ParamRole::AccelerationStructure { kind, arg } = &b.role else {
        return Err(invariant("an acceleration structure without its role"));
    };
    let (AbiInt::Const(slot), AbiInt::Const(1)) = (arg.location.index, arg.location.count) else {
        return Err(Unsupported::not_implemented(Feature::RayTracing).into());
    };
    let level = match kind {
        AccelerationStructureKind::Primitive => AccelerationStructureLevel::Primitive,
        AccelerationStructureKind::Instance => AccelerationStructureLevel::Instance,
    };
    Ok((slot, level))
}

/// Plan a used acceleration-structure argument (its descriptor, on a profile that can build and
/// query acceleration structures) or intersection function table argument (its words). A structure
/// argument whose every reader is an instance count query has no descriptor: each query reads the
/// binder's count, which needs no device feature.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    if let ParamRole::IntersectionTable(arg) = &b.role {
        return select_table(cx, b, arg);
    }
    let at = Value::Arg(ArgIdx::new(b.param));
    let read = readers(cx, at);
    let counts_only = !read.is_empty()
        && read.into_iter().all(|r| {
            cx.body
                .insts
                .get(r)
                .and_then(instance_count_of)
                .is_some_and(|(args, _)| args == [at])
        });
    if counts_only {
        return Ok(());
    }
    let enabled = &cx.profile.fields().enabled.features;
    let missing: Vec<Capability> = [
        VkFeature::RayQuery,
        VkFeature::AccelerationStructure,
        VkFeature::BufferDeviceAddress,
    ]
    .into_iter()
    .filter(|f| !enabled.contains(f))
    .map(Capability::Feature)
    .collect();
    if !missing.is_empty() {
        return Err(Unsupported::needs_any_of(Feature::RayTracing, vec![missing]).into());
    }
    let (slot, level) = slot_of(b)?;
    if cx
        .families
        .raytracing
        .structures
        .insert(slot, level)
        .is_none()
    {
        cx.descriptors.acceleration_structures =
            cx.descriptors.acceleration_structures.saturating_add(1);
    }
    Ok(())
}

/// The Metal buffer slot of intersection function table argument `arg`: a single argument, not
/// an array of them.
fn table_of(arg: &ObjectArg) -> Result<u32, Error> {
    match (arg.location.index, arg.location.count) {
        (AbiInt::Const(slot), AbiInt::Const(1)) => Ok(slot),
        _ => Err(Unsupported::not_implemented(Feature::IntersectionFunctionTable).into()),
    }
}

/// Plan a used intersection function table argument: every reader is an intersect call that reads
/// it as its table only, and the table is a read-only word view at its Metal buffer slot, read at
/// the slot's bind residual. Any other reader (`set_buffer` and the other table writes a kernel may
/// make, or a table passed to another operation) is refused at that reader, so the census can site
/// it.
fn select_table(cx: &mut SelectCx<'_>, b: &ParamBinding, arg: &ObjectArg) -> Result<(), Error> {
    let slot = table_of(arg)?;
    let at = Value::Arg(ArgIdx::new(b.param));
    if let Some(r) = readers(cx, at)
        .into_iter()
        .find(|r| !only_table(cx, *r, at))
    {
        let refusal = Unsupported::not_implemented(Feature::IntersectionFunctionTable);
        let block = cx
            .body
            .blocks
            .iter()
            .find(|(_, blk)| blk.start <= r && r < blk.end)
            .map(|(id, _)| id.raw());
        return Err(match block {
            Some(block) => refusal
                .at(Location::function(
                    cx.entry.ep.name.clone(),
                    Some(block),
                    Some(r.raw()),
                ))
                .into(),
            None => refusal.into(),
        });
    }
    if cx.families.raytracing.tables.insert(slot) {
        cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
    }
    cx.push
        .insert(PushField::SlotBindOffset(table_slot(cx.stage, slot)));
    Ok(())
}

/// Whether instruction `r` is an intersect call that reads `v` as its table and nowhere else.
fn only_table(cx: &SelectCx<'_>, r: InstId, v: Value) -> bool {
    cx.body
        .insts
        .get(r)
        .and_then(intersect_of)
        .is_some_and(|(i, args, _)| {
            args.iter()
                .enumerate()
                .all(|(k, a)| *a != v || i.index(IntersectRole::Table) == Some(k))
        })
}

/// Whether `v` is an intersection function table argument of the entry.
fn is_table_param(cx: &SelectCx<'_>, v: Value) -> bool {
    param_binding(cx.entry, v).is_some_and(|b| matches!(b.role, ParamRole::IntersectionTable(_)))
}

/// Whether `v` is an acceleration-structure argument of the entry.
fn is_structure_param(cx: &SelectCx<'_>, v: Value) -> bool {
    param_binding(cx.entry, v)
        .is_some_and(|b| matches!(b.role, ParamRole::AccelerationStructure { .. }))
}

/// The operands and declaration of `inst` when it calls
/// `air.get_instance_count_instance_acceleration_structure`.
fn instance_count_of(inst: &Inst) -> Option<(&[Value], FuncId)> {
    let InstKind::Call(call) = &inst.kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::AccelerationStructure(AccelerationStructure::InstanceCount),
            decl,
        } => Some((call.args.as_ref(), decl)),
        _ => None,
    }
}

/// The value of an acceleration-structure argument: its descriptor, loaded. An intersection
/// function table argument has none: its only readers are intersect calls, which read its words.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    if matches!(b.role, ParamRole::IntersectionTable(_)) {
        return Err(cx.fail("an intersection function table argument has no value"));
    }
    match slot_of(b) {
        Ok((slot, _)) => structure(cx, fb, slot),
        Err(_) => Err(cx.fail("an acceleration structure argument without a slot")),
    }
}

/// The acceleration structure bound at buffer slot `slot`, loaded in the open block.
fn structure(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, slot: u32) -> BResult<SValue> {
    let Some(g) = cx.decl.families.raytracing.structures.get(&slot).copied() else {
        return Err(cx.fail(format!(
            "acceleration structure slot {slot} is not declared"
        )));
    };
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default())
}

/// The word view of the intersection function table at Metal buffer `slot`, and the word its
/// bind residual starts at (the table's buffer is bound at a 4-byte aligned offset: a table is
/// its own buffer, bound whole), in the open block.
fn table(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, slot: u32) -> BResult<(View, SValue)> {
    let Some(view) = cx.decl.families.raytracing.tables.get(&slot).copied() else {
        return Err(cx.fail(format!(
            "intersection function table slot {slot} is not declared"
        )));
    };
    let residual = cx.push_field(
        fb,
        PushField::SlotBindOffset(table_slot(cx.plan.stage, slot)),
    )?;
    let two = fb.u32_const(2)?;
    let two = fb.module().const_value(two)?;
    let base = fb.binary(crate::spirv::ir::BinOp::ShiftRightLogical, residual, two)?;
    Ok((view, base))
}

/// The intersect intrinsic `inst` calls, if it is an intersect call.
fn intersect_of(inst: &Inst) -> Option<(Intersect, &[Value], FuncId)> {
    let InstKind::Call(call) = &inst.kind else {
        return None;
    };
    match &call.callee {
        Callee::Air {
            op: AirIntrinsic::Intersect(i),
            decl,
        } => Some((*i, call.args.as_ref(), *decl)),
        _ => None,
    }
}

/// Whether `inst` is `air.get_null_intersection_function_table`.
fn is_null_table(inst: &Inst) -> bool {
    matches!(
        &inst.kind,
        InstKind::Call(call) if matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::FunctionTable(FunctionTable::NullIntersection),
                ..
            }
        )
    )
}

/// Whether `v` is the result of an intersect call of `body` (its `extractvalue`s are this
/// family's).
pub fn is_intersect_result(body: &Body, v: Value) -> bool {
    let Value::Inst(d) = v else { return false };
    body.insts
        .get(d)
        .is_some_and(|inst| intersect_of(inst).is_some())
}

/// Whether pointer value `v` may be an intersection's primitive-data pointer: its provenance holds
/// an intersect call's opaque root.
pub fn is_primitive_data(cx: &SelectCx<'_>, v: Value) -> bool {
    cx.families.raytracing.intersects
        && cx
            .memory
            .roots_of(v)
            .is_some_and(|roots| roots.iter().any(|r| primitive_data_root(cx.entry, *r)))
}

/// The refusal of instruction `at` when it reads an intersection's primitive-data pointer other
/// than by moving it: its only readers are an `insertvalue` that puts it into an aggregate and a
/// store of it whose bytes nothing reads ([`plan_unread_stores`]). A load or store through it, a
/// compare, a conversion, a merge (`phi`, `select`, `freeze`), an address computation, a call
/// operand, and any other store read the data or its address, which no Vulkan structure holds. (An
/// `extractvalue` reads the aggregate, not the pointer.)
pub fn primitive_data_reader(cx: &SelectCx<'_>, at: InstId, inst: &Inst) -> Option<Error> {
    let mut reads = false;
    inst.kind.for_each_value(|v| {
        if reads || !is_primitive_data(cx, v) {
            return;
        }
        reads = match &inst.kind {
            InstKind::InsertValue { val, .. } => *val != v,
            InstKind::Store { val, ptr, .. } => {
                *ptr == v || *val != v || !cx.families.raytracing.unread.contains(&at)
            }
            _ => true,
        };
    });
    reads.then(primitive_data_read)
}

/// Find the stores of an intersection's primitive-data pointer whose bytes nothing reads, once,
/// before the instruction walk ([`RayTracingPlan::unread`]), and whether the body calls
/// `air.intersect` at all (no pointer of a body without a call is primitive data). Such a store
/// writes nothing, exactly, since no read observes what it would have written. A store qualifies
/// when it is a plain store through a thread pointer at one constant offset into allocas, and
/// every read of those allocas (a load, an atomic, a memory intrinsic's source, any other call's
/// pointer operand) lies wholly outside its eight bytes:
///
/// - a read through a thread pointer that may be `Dynamic` may read any alloca, and a read of
///   unknown length, or a call's operand, any byte of its allocas; either keeps every store it
///   may reach;
/// - a read's bytes start in its pointer's offset range, or, when that is unknown, in the
///   element range its `getelementptr` chain's indices select: a dynamic index selects one
///   element of its array (an index outside the array is undefined in Metal, as `memory::alloca`
///   reads it).
///
/// A read visits only the candidates into the allocas it may reach, and only a read that reaches
/// one computes its range. Every walk ticks its length, and the range walk settles each pointer
/// definition once per body, so the work is linear in the body: a body of many such stores and
/// reads is refused at the work cap, never slowed past it.
pub fn plan_unread_stores(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    let mut intersects = false;
    for inst in body.insts.values() {
        m.tick(1)?;
        if intersect_of(inst).is_some() {
            intersects = true;
            break;
        }
    }
    cx.families.raytracing.intersects = intersects;
    if !intersects {
        return Ok(());
    }
    // Each candidate store and the offset of its eight bytes in its allocas, and the candidates
    // into each alloca (indices into `stores`), so a read visits only the stores it may reach.
    let mut stores: Vec<(InstId, i64)> = Vec::new();
    let mut by_alloca: BTreeMap<InstId, Vec<usize>> = BTreeMap::new();
    let mut charged = 0usize;
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Store {
            val,
            ptr,
            volatile: false,
            atomic: None,
            ..
        } = &inst.kind
        else {
            continue;
        };
        if !is_primitive_data(cx, *val) || is_primitive_data(cx, *ptr) {
            continue;
        }
        let Some(at) = cx.memory.offset(*ptr).and_then(OffsetRange::exact) else {
            continue;
        };
        let Some(roots) = cx.memory.roots_of(*ptr) else {
            continue;
        };
        m.tick(units(roots.len()))?;
        let allocas: Option<Vec<InstId>> = roots
            .iter()
            .map(|r| match r {
                Root::Alloca(a) => Some(*a),
                _ => None,
            })
            .collect();
        let Some(allocas) = allocas.filter(|a| !a.is_empty()) else {
            continue;
        };
        // The store, and per alloca its index and (at most) a new entry of the map, twice.
        let bytes = std::mem::size_of::<(InstId, i64)>().saturating_add(
            allocas.len().saturating_mul(
                std::mem::size_of::<usize>()
                    .saturating_add(2 * std::mem::size_of::<(InstId, Vec<usize>)>()),
            ),
        );
        m.charge(bytes)?;
        charged = charged.saturating_add(bytes);
        let k = stores.len();
        stores.push((id, at));
        for a in allocas {
            by_alloca.entry(a).or_default().push(k);
        }
    }
    let mut ranges = RangeWalk::default();
    let planned = (|| -> Result<BTreeSet<InstId>, Error> {
        if stores.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut read = vec![false; stores.len()];
        m.charge(read.len())?;
        charged = charged.saturating_add(read.len());
        // The candidates no read has reached yet: none left ends the walk.
        let mut open = stores.len();
        for (_, inst) in body.insts.iter() {
            if open == 0 {
                break;
            }
            m.tick(1)?;
            for (p, len) in reads(cx, inst) {
                if !matches!(
                    cx.value_ty(p).and_then(|t| cx.entry.module.types().get(t)),
                    Some(Type::Ptr {
                        space: AddrSpace::Thread
                    })
                ) {
                    continue;
                }
                // A read that may reach any alloca keeps every store.
                let roots = match cx.memory.roots_of(p) {
                    Some(roots) if !roots.contains(&Root::Dynamic) => roots,
                    _ => {
                        m.tick(units(read.len()))?;
                        read.iter_mut().for_each(|r| *r = true);
                        open = 0;
                        break;
                    }
                };
                // The candidates into one root's alloca; a read reaching none needs no range (a
                // saving of work only: its range would decide nothing).
                let near = |r: &Root| match r {
                    Root::Alloca(a) => by_alloca.get(a),
                    _ => None,
                };
                m.tick(units(roots.len()))?;
                if !roots.iter().any(|r| near(r).is_some()) {
                    continue;
                }
                let bytes = match len {
                    Some(n) => ranges
                        .range(cx, p, m)?
                        .and_then(|(lo, hi)| Some((lo, hi.checked_add(i64::try_from(n).ok()?)?))),
                    None => None,
                };
                m.tick(units(roots.len()))?;
                for r in roots {
                    let Some(near) = near(r) else {
                        continue;
                    };
                    m.tick(units(near.len()))?;
                    for &k in near {
                        let (Some((_, at)), Some(seen)) = (stores.get(k), read.get_mut(k)) else {
                            continue;
                        };
                        let clear = bytes.is_some_and(|(lo, hi)| {
                            hi <= *at || at.checked_add(8).is_some_and(|end| lo >= end)
                        });
                        if !clear && !*seen {
                            *seen = true;
                            open = open.saturating_sub(1);
                        }
                    }
                }
            }
        }
        Ok(stores
            .iter()
            .zip(&read)
            .filter(|(_, r)| !**r)
            .map(|((id, _), _)| *id)
            .collect())
    })();
    m.release(charged.saturating_add(ranges.growth.charged()));
    cx.families.raytracing.unread = planned?;
    Ok(())
}

/// The work units of a walk over `n` items: one each, saturating.
fn units(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The memory instruction `inst` reads, each pointer with the bytes it reads there (`None` when
/// unknown): a load's, an atomic's, a `memcpy` or `memmove` source's, and every pointer operand
/// of any other call but a memory intrinsic's destination.
fn reads(cx: &SelectCx<'_>, inst: &Inst) -> Vec<(Value, Option<u64>)> {
    let size = |t: Option<TypeId>| t.and_then(|t| cx.memory.layouts.store_size(t));
    match &inst.kind {
        InstKind::Load { ptr, .. } => vec![(*ptr, size(Some(inst.ty)))],
        InstKind::AtomicRmw { ptr, val, .. } => vec![(*ptr, size(cx.value_ty(*val)))],
        InstKind::CmpXchg { ptr, cmp, .. } => vec![(*ptr, size(cx.value_ty(*cmp)))],
        InstKind::Call(call) => {
            let (written, source) = match &call.callee {
                Callee::Llvm {
                    op: LlvmIntrinsic::Memset { .. },
                    ..
                } => (Some(0), None),
                Callee::Llvm {
                    op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                    ..
                } => (Some(0), Some(1)),
                _ => (None, None),
            };
            let length = call.args.get(2).and_then(|&l| match l {
                Value::Const(c) => match cx.entry.module.consts().get(c).map(|k| &k.value) {
                    Some(Const::Int(n)) => Some(*n),
                    _ => None,
                },
                _ => None,
            });
            call.args
                .iter()
                .enumerate()
                .filter(|(k, _)| Some(*k) != written)
                .map(|(k, &a)| (a, if Some(k) == source { length } else { None }))
                .collect()
        }
        _ => Vec::new(),
    }
}

/// A byte range from an alloca's start, `[lo, hi]`.
type Range = (i64, i64);

/// What [`RangeWalk`] knows of a pointer definition.
#[derive(Clone, Copy, Debug)]
enum Walked {
    /// On the walk's path: reaching it again closes a cycle.
    Open,
    /// Its range, or `None` when nothing bounds it.
    Done(Option<Range>),
}

/// A definition on [`RangeWalk`]'s path: its id, how many of its operands are folded, and their
/// fold (`None` before the first).
type Frame = (InstId, usize, Option<Option<Range>>);

/// The range of byte offsets each thread pointer may hold from its allocas' start, for
/// [`plan_unread_stores`]: its offset range or, when that is unknown, the range its
/// `getelementptr` chain from the allocas selects, a dynamic array or vector index ranging over
/// its elements (an index outside them is undefined in Metal); `None` when neither bounds it (a
/// dynamic first index, a pointer cycle, any other definition).
///
/// A definition's range depends only on the definitions it reaches, and `None` absorbs every
/// fold (a `getelementptr` of an unbounded base, a join with an unbounded arm), so a pointer
/// whose walk meets `None` anywhere, a cycle included, is `None` itself. The walk is iterative
/// (no depth bound) and settles each definition once per body: a definition shared by many
/// reads, or by both arms of many `select`s, is walked once, and the work is linear in the body.
/// Reaching a definition on the path again closes a cycle, which every definition on the path
/// reaches, so all are `None`.
#[derive(Default)]
struct RangeWalk {
    known: BTreeMap<InstId, Walked>,
    path: Vec<Frame>,
    /// The ledger's share of `known` and `path`.
    growth: Growth,
}

impl RangeWalk {
    /// The range of pointer `p`.
    fn range(
        &mut self,
        cx: &SelectCx<'_>,
        p: Value,
        m: &mut Meter,
    ) -> Result<Option<Range>, Error> {
        m.tick(1)?;
        let root = match self.settled(cx, p) {
            Settled::Range(r) => return Ok(r),
            Settled::Cycle => return Ok(None),
            Settled::Walk(i) => i,
        };
        self.path.clear();
        self.open(root, m)?;
        // The range of the operand just settled, folded into the definition on top of the path.
        let mut settled: Option<Option<Range>> = None;
        loop {
            m.tick(1)?;
            let Some(frame) = self.path.last_mut() else {
                return Err(invariant("an empty pointer range walk"));
            };
            let (id, next, acc) = (frame.0, &mut frame.1, &mut frame.2);
            let inst = cx
                .body
                .insts
                .get(id)
                .ok_or_else(|| invariant("a pointer range walk off the body"))?;
            if let Some(r) = settled.take() {
                let folded = match &inst.kind {
                    InstKind::Gep {
                        src_ty, indices, ..
                    } => r.and_then(|b| gep_range(cx, b, *src_ty, indices)),
                    InstKind::Select { .. } | InstKind::Phi { .. } => match *acc {
                        None => r,
                        Some(a) => join(a, r),
                    },
                    _ => r,
                };
                if folded.is_none() {
                    return self.unbounded(m);
                }
                *acc = Some(folded);
                *next += 1;
            }
            let Some(v) = operand(inst, *next) else {
                // Every operand folded: the definition is settled.
                let r = acc.flatten();
                self.path.pop();
                self.settle(id, Walked::Done(r), m)?;
                if self.path.is_empty() {
                    return Ok(r);
                }
                settled = Some(r);
                continue;
            };
            match self.settled(cx, v) {
                Settled::Range(r) => settled = Some(r),
                Settled::Cycle => return self.unbounded(m),
                Settled::Walk(i) => self.open(i, m)?,
            }
        }
    }

    /// What is known of `v` without walking it: its offset range, a settled or open definition's,
    /// an alloca's start, or `None` for a value that is not a definition the walk folds.
    fn settled(&self, cx: &SelectCx<'_>, v: Value) -> Settled {
        if let Some(OffsetRange::Span { lo, hi }) = cx.memory.offset(v) {
            return Settled::Range(Some((lo, hi)));
        }
        let Value::Inst(i) = v else {
            return Settled::Range(None);
        };
        match self.known.get(&i) {
            Some(Walked::Done(r)) => return Settled::Range(*r),
            Some(Walked::Open) => return Settled::Cycle,
            None => {}
        }
        match cx.body.insts.get(i).map(|inst| &inst.kind) {
            Some(InstKind::Alloca { .. }) => Settled::Range(Some((0, 0))),
            Some(
                InstKind::Gep { .. }
                | InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ..
                }
                | InstKind::Freeze(_)
                | InstKind::Select { .. }
                | InstKind::Phi { .. },
            ) => Settled::Walk(i),
            _ => Settled::Range(None),
        }
    }

    /// Put definition `i` on the path.
    fn open(&mut self, i: InstId, m: &mut Meter) -> Result<(), Error> {
        self.path.push((i, 0, None));
        self.settle(i, Walked::Open, m)
    }

    /// Record what is known of definition `i`, charging the growth.
    fn settle(&mut self, i: InstId, w: Walked, m: &mut Meter) -> Result<(), Error> {
        self.known.insert(i, w);
        let footprint = tree_bytes(self.known.len(), std::mem::size_of::<(InstId, Walked)>())
            .saturating_add(vec_bytes(&self.path));
        Ok(self.growth.charge_to(footprint, m)?)
    }

    /// The walk met `None`: every definition on the path reaches it, so is `None`.
    fn unbounded(&mut self, m: &mut Meter) -> Result<Option<Range>, Error> {
        m.tick(units(self.path.len()))?;
        while let Some((id, _, _)) = self.path.pop() {
            self.known.insert(id, Walked::Done(None));
        }
        Ok(None)
    }
}

/// What [`RangeWalk::settled`] knows of a value.
enum Settled {
    /// Its range, without a walk.
    Range(Option<Range>),
    /// A definition on the walk's path.
    Cycle,
    /// A definition to walk.
    Walk(InstId),
}

/// Operand `k` of `inst` that [`RangeWalk`] folds: a `getelementptr`'s base, a cast's or
/// `freeze`'s operand, a `select`'s arms, a `phi`'s incoming values.
fn operand(inst: &Inst, k: usize) -> Option<Value> {
    match &inst.kind {
        InstKind::Gep { base, .. } => (k == 0).then_some(*base),
        InstKind::Cast { v, .. } | InstKind::Freeze(v) => (k == 0).then_some(*v),
        InstKind::Select { t, f, .. } => [*t, *f].get(k).copied(),
        InstKind::Phi { incoming, .. } => incoming.get(k).map(|(x, _)| *x),
        _ => None,
    }
}

/// The range covering two ranges, `None` when either is.
fn join(a: Option<Range>, b: Option<Range>) -> Option<Range> {
    match (a, b) {
        (Some((a, b)), Some((c, d))) => Some((a.min(c), b.max(d))),
        _ => None,
    }
}

/// Index an array or vector of `len` elements of `size` bytes from offsets `[lo, hi]`: the
/// constant index `k`'s element, or any element for a dynamic one.
fn element_index(lo: &mut i64, hi: &mut i64, size: u64, len: u64, k: Option<i64>) -> Option<()> {
    let size = i64::try_from(size).ok()?;
    match k {
        Some(k) => {
            let d = k.checked_mul(size)?;
            *lo = lo.checked_add(d)?;
            *hi = hi.checked_add(d)?;
        }
        None => {
            let last = i64::try_from(len.checked_sub(1)?).ok()?;
            *hi = hi.checked_add(last.checked_mul(size)?)?;
        }
    }
    Some(())
}

/// The offsets of `getelementptr src_ty, base, indices` from a base in `[lo, hi]`: a constant
/// index adds its displacement, a dynamic array or vector index adds `[0, (len - 1) * size]`,
/// and a dynamic first index (pointer arithmetic past the object) bounds nothing.
fn gep_range(
    cx: &SelectCx<'_>,
    (mut lo, mut hi): (i64, i64),
    src_ty: TypeId,
    indices: &[Value],
) -> Option<(i64, i64)> {
    let layouts = &cx.memory.layouts;
    let types = cx.entry.module.types();
    let konst = |v: Value| match v {
        Value::Const(c) => match cx.entry.module.consts().get(c).map(|k| &k.value) {
            Some(Const::Int(n)) => {
                let bits = cx.value_ty(v).and_then(|t| match types.get(t) {
                    Some(Type::Int(b)) => Some(*b),
                    _ => None,
                })?;
                // Sign-extend the index from its width.
                let shift = 64u32.checked_sub(u32::from(bits))?;
                Some(((*n as i64) << shift) >> shift)
            }
            _ => None,
        },
        _ => None,
    };
    let add = |lo: &mut i64, hi: &mut i64, d: i64| -> Option<()> {
        *lo = lo.checked_add(d)?;
        *hi = hi.checked_add(d)?;
        Some(())
    };
    let mut cur = src_ty;
    for (n, &i) in indices.iter().enumerate() {
        if n == 0 {
            let size = i64::try_from(layouts.alloc_size(src_ty)?).ok()?;
            add(&mut lo, &mut hi, konst(i)?.checked_mul(size)?)?;
            continue;
        }
        match types.get(cur)? {
            Type::Struct { fields, .. } => {
                let k = usize::try_from(konst(i)?).ok()?;
                let off = i64::try_from(*layouts.field_offsets(cur)?.get(k)?).ok()?;
                add(&mut lo, &mut hi, off)?;
                cur = *fields.get(k)?;
            }
            Type::Array { elem, len } => {
                element_index(&mut lo, &mut hi, layouts.alloc_size(*elem)?, *len, konst(i))?;
                cur = *elem;
            }
            Type::Vector { elem, len } => {
                element_index(
                    &mut lo,
                    &mut hi,
                    layouts.alloc_size(*elem)?,
                    u64::from(*len),
                    konst(i),
                )?;
                cur = *elem;
            }
            _ => return None,
        }
    }
    Some((lo, hi))
}

/// The refusal of reading an intersection's primitive data.
pub fn primitive_data_read() -> Error {
    Unsupported::not_implemented(Feature::RayTracing).into()
}

/// The lowered representation of `v` when it is an intersect call's result: the record of its
/// fields without the primitive-data pointer.
pub fn result_repr(cx: &SelectCx<'_>, v: Value) -> Option<Result<DemoteTy, Error>> {
    let Value::Inst(d) = v else { return None };
    let inst = cx.body.insts.get(d)?;
    let (i, _, _) = intersect_of(inst)?;
    Some(record_repr(cx, i.tags, inst.ty))
}

/// The record of result type `ty` of an intersect call with `tags`: each field's arith
/// representation, in order, skipping the primitive-data pointer.
fn record_repr(cx: &SelectCx<'_>, tags: IntersectTags, ty: TypeId) -> Result<DemoteTy, Error> {
    let Some(Type::Struct { fields, .. }) = cx.entry.module.types().get(ty) else {
        return Err(invariant("an intersect result that is no struct"));
    };
    let mut parts = Vec::with_capacity(fields.len());
    for (field, ty) in tags.result().fields().zip(fields.iter()) {
        if field != ResultField::PrimitiveData {
            parts.push(arith::lowered_type(cx.entry, cx.profile, *ty)?);
        }
    }
    Ok(DemoteTy::Composite(parts.into()))
}

/// The readers of `at` (an intersect call, a null table, an intersection function table
/// argument, or an acceleration-structure argument) in the body, from a map computed once per
/// function over the operands of every instruction of every block. A block's terminator is its
/// last instruction (the verifier holds every body to that), so terminators are among them.
fn readers(cx: &mut SelectCx<'_>, at: Value) -> Vec<InstId> {
    if cx.families.raytracing.readers.is_none() {
        let body = cx.body;
        let mut map: BTreeMap<Value, Vec<InstId>> = BTreeMap::new();
        for (_, block) in body.blocks.iter() {
            for id in block.inst_ids() {
                let Some(inst) = body.insts.get(id) else {
                    continue;
                };
                inst.kind.for_each_value(|v| {
                    let ours = match v {
                        Value::Inst(d) => body
                            .insts
                            .get(d)
                            .is_some_and(|x| is_null_table(x) || intersect_of(x).is_some()),
                        Value::Arg(_) => is_table_param(cx, v) || is_structure_param(cx, v),
                        Value::Const(_) => false,
                    };
                    if ours {
                        let r = map.entry(v).or_default();
                        if r.last() != Some(&id) {
                            r.push(id);
                        }
                    }
                });
            }
        }
        cx.families.raytracing.readers = Some(map);
    }
    cx.families
        .raytracing
        .readers
        .as_ref()
        .and_then(|map| map.get(&at).cloned())
        .unwrap_or_default()
}

/// Select the strategy of ray tracing instruction `at`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<RayTracingStrategy, Error> {
    if is_null_table(inst) {
        return null_table(cx, at);
    }
    if let Some((i, args, decl)) = intersect_of(inst) {
        return intersect(cx, at, inst, i, args, decl, m);
    }
    if let Some((args, decl)) = instance_count_of(inst) {
        return instance_count(cx, args, decl);
    }
    match &inst.kind {
        InstKind::ExtractValue { agg, indices } if is_intersect_result(cx.body, *agg) => {
            field(cx, *agg, indices)
        }
        // A store of a primitive-data pointer whose bytes nothing reads.
        InstKind::Store { .. } if cx.families.raytracing.unread.contains(&at) => {
            Ok(RayTracingStrategy::UnreadStore)
        }
        // A primitive-data pointer carried through an aggregate.
        InstKind::ExtractValue { .. } if is_primitive_data(cx, Value::Inst(at)) => {
            Ok(RayTracingStrategy::PrimitiveData)
        }
        _ => Err(invariant("a ray tracing instruction of another kind")),
    }
}

/// `get_instance_count()` of the instance structure argument `args[0]`: the binder's count of
/// the structure bound at its slot. A structure that is not an entry argument (an argument-buffer
/// member, an instance's own structure) has no slot the binder knows.
fn instance_count(
    cx: &mut SelectCx<'_>,
    args: &[Value],
    decl: FuncId,
) -> Result<RayTracingStrategy, Error> {
    let level = match args {
        [s] => cx.param(*s).map(slot_of),
        _ => return Err(invariant("an instance count query without its structure")),
    };
    let index = match level {
        Some(Ok((index, AccelerationStructureLevel::Instance))) => index,
        _ => {
            return Err(Unsupported::not_implemented(Feature::RayTracing)
                .naming(decl_name(cx, decl))
                .into())
        }
    };
    let slot = SlotRef {
        stage: cx.stage,
        class: SlotClass::Buffer,
        index,
    };
    cx.facts
        .insert(RuntimeFact::AccelerationStructureInstanceCount { slot });
    cx.push.insert(PushField::InstanceCount(slot));
    Ok(RayTracingStrategy::InstanceCount(slot))
}

/// The null table: every reader is an intersect call that reads it as its table only.
fn null_table(cx: &mut SelectCx<'_>, at: InstId) -> Result<RayTracingStrategy, Error> {
    let v = Value::Inst(at);
    if !readers(cx, v).into_iter().all(|r| only_table(cx, r, v)) {
        return Err(Unsupported::not_implemented(Feature::IntersectionFunctionTable).into());
    }
    Ok(RayTracingStrategy::NullTable)
}

fn intersect(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    i: Intersect,
    args: &[Value],
    decl: FuncId,
    m: &mut Meter,
) -> Result<RayTracingStrategy, Error> {
    let refuse =
        |cx: &SelectCx<'_>, u: Unsupported| -> Error { u.naming(decl_name(cx, decl)).into() };
    let t = i.tags;
    // A ray query has no time: Vulkan traces motion only in ray tracing pipelines
    // (`VK_NV_ray_tracing_motion_blur`). This decides first: no lowering of the rest of the
    // call (its intersection function buffer among them) could trace it.
    if t.motion() {
        return Err(refuse(
            cx,
            Unsupported::not_representable(Feature::RayTracing),
        ));
    }
    if t.function_buffer {
        return Err(refuse(
            cx,
            Unsupported::not_implemented(Feature::IntersectionFunctionTable),
        ));
    }
    if t.curve_data {
        return Err(refuse(
            cx,
            Unsupported::not_implemented(Feature::RayTracing),
        ));
    }
    let arg = |role| i.index(role).and_then(|k| args.get(k)).copied();
    // The null table, or an entry argument's table read as words. A table of a stage that may link
    // an intersection function would need that function's case.
    let table = match arg(IntersectRole::Table) {
        Some(Value::Inst(d)) if cx.body.insts.get(d).is_some_and(is_null_table) => None,
        Some(v) => match cx.param(v).map(|b| &b.role) {
            Some(ParamRole::IntersectionTable(a)) if !cx.entry.intersection_candidates => {
                Some(table_of(a)?)
            }
            _ => {
                return Err(refuse(
                    cx,
                    Unsupported::not_implemented(Feature::IntersectionFunctionTable),
                ))
            }
        },
        None => return Err(invariant("an intersect call without its table")),
    };
    // The structure is an entry argument whose level the tags name.
    let Some(b) = arg(IntersectRole::Structure).and_then(|s| cx.param(s)) else {
        return Err(refuse(
            cx,
            Unsupported::not_implemented(Feature::RayTracing),
        ));
    };
    let (slot, level) = slot_of(b)?;
    let levels_agree = matches!(
        (t.instancing, level),
        (Instancing::None, AccelerationStructureLevel::Primitive)
            | (
                Instancing::Single | Instancing::Multi,
                AccelerationStructureLevel::Instance
            )
    );
    if !levels_agree {
        return Err(refuse(
            cx,
            Unsupported::not_implemented(Feature::RayTracing),
        ));
    }
    // Every reader extracts one field (the primitive-data pointer's readers are refused where
    // they read it).
    for r in readers(cx, Value::Inst(at)) {
        let ok = match cx.body.insts.get(r).map(|x| &x.kind) {
            Some(InstKind::ExtractValue { agg, indices }) => {
                *agg == Value::Inst(at) && indices.len() == 1
            }
            _ => false,
        };
        if !ok {
            return Err(refuse(
                cx,
                Unsupported::not_implemented(Feature::RayTracing),
            ));
        }
    }
    let record = record_repr(cx, t, inst.ty)?;
    let levels = match t.instancing {
        Instancing::Multi => Some(Box::new(level_ids(cx, at, i, args, inst.ty, m).map_err(
            |e| match e {
                Error::Unsupported(u) if u.symbol.is_none() => u.naming(decl_name(cx, decl)).into(),
                other => other,
            },
        )?)),
        Instancing::None | Instancing::Single => None,
    };
    // The device computes the hit, and the conventions Metal has not been read on are this
    // lowering's: the artifact says so.
    cx.precision.insert(PrecisionNote::DeviceRayIntersection);
    Ok(RayTracingStrategy::Intersect(Box::new(IntersectPlan {
        slot,
        tags: t,
        record,
        table,
        levels,
    })))
}

/// The levels `max_levels<n>` may name: the range Metal's compiler accepts (measured with metal
/// 32023.921: `max_levels<1>` and `max_levels<33>` do not compile, 2 to 32 do).
const MAX_LEVELS: std::ops::RangeInclusive<u64> = 2..=32;

/// The id stores of multi-level intersector call `at` (`i`, operands `args`, result type
/// `result`): the first element of each id array, a `u32` of the AIR type of the result's `type`
/// field. `max_levels` is the template parameter `max_levels<n>`, so a constant in [`MAX_LEVELS`],
/// and the arrays (of `n - 1` elements) hold that element; a run-time value or another constant is
/// not Metal's (`InvalidInput`). A pointer outside the private arena and typed allocas is refused
/// by the local family.
fn level_ids(
    cx: &mut SelectCx<'_>,
    at: InstId,
    i: Intersect,
    args: &[Value],
    result: TypeId,
    m: &mut Meter,
) -> Result<LevelIds, Error> {
    let arg = |role| i.index(role).and_then(|k| args.get(k)).copied();
    let malformed = |kind| {
        Error::InvalidInput(InputError::new(
            InputErrorKind::Intrinsic(kind),
            Location::function(cx.entry.ep.name.clone(), None, Some(at.raw())),
        ))
    };
    let levels = match arg(IntersectRole::MaxLevels) {
        // A null constant is 0, as `air::intrinsics::sig::Immediates` reads an immediate: the
        // decoder spells a scalar integer zero `Int(0)`, but the canonical passes fold
        // `freeze i8 undef`, or an `extractvalue` of `zeroinitializer`, to a null `i8`.
        Some(Value::Const(c)) => match cx.entry.module.consts().get(c).map(|k| &k.value) {
            Some(Const::Int(n)) => Some(*n),
            Some(Const::Null) => Some(0),
            _ => None,
        },
        _ => None,
    };
    match levels {
        None => return Err(malformed(IntrinsicError::ImmediateNotConstant)),
        Some(n) if !MAX_LEVELS.contains(&n) => {
            return Err(malformed(IntrinsicError::SignatureMismatch))
        }
        Some(_) => {}
    }
    let word = match cx.entry.module.types().get(result) {
        Some(Type::Struct { fields, .. }) => fields.first().copied(),
        _ => None,
    }
    .ok_or_else(|| invariant("an intersect result without its type field"))?;
    let (Some(ids), Some(users)) = (
        arg(IntersectRole::InstanceIds),
        arg(IntersectRole::UserInstanceIds),
    ) else {
        return Err(invariant(
            "a multi-level intersect call without its id arrays",
        ));
    };
    Ok(LevelIds {
        instance: local::intrinsic_store(cx, ids, word, m)?,
        user: local::intrinsic_store(cx, users, word, m)?,
    })
}

/// `extractvalue` of an intersect result: its member in the lowered record.
fn field(cx: &SelectCx<'_>, agg: Value, indices: &[u32]) -> Result<RayTracingStrategy, Error> {
    let Value::Inst(d) = agg else {
        return Err(invariant("a field of a value that is no intersect call"));
    };
    let Some((i, _, _)) = cx.body.insts.get(d).and_then(intersect_of) else {
        return Err(invariant("a field of a value that is no intersect call"));
    };
    let [k] = indices else {
        return Err(not_lowered());
    };
    let data = i.tags.result().index(ResultField::PrimitiveData);
    match data {
        // The primitive-data pointer has no value.
        Some(p) if p == *k => Ok(RayTracingStrategy::PrimitiveData),
        Some(p) if p < *k => Ok(RayTracingStrategy::Field(k - 1)),
        _ => Ok(RayTracingStrategy::Field(*k)),
    }
}
