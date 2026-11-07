//! `lower::sync::barrier`: barriers, `air.atomic.fence`, and derivatives.
//!
//! - **Barriers** (`air.wg.barrier`, `air.simdgroup.barrier`). The execution scope is the one
//!   `air::intrinsics::sync` decodes (an AIR 2.8 barrier's scope operand, never its spelling; an
//!   AIR 2.9 barrier's name): a threadgroup barrier is `OpControlBarrier(Workgroup, …)`, compute
//!   only (a threadgroup barrier in a render stage is tile shading); a simdgroup barrier is
//!   `OpControlBarrier(Subgroup, …)` in any stage, and makes the function's simd class cross-lane,
//!   so the subgroup tier pins 32 lanes. Metal's `mem_flags` order the named memories among the
//!   threads of the barrier's memory scope: the execution scope for an AIR 2.8 barrier, and for an
//!   AIR 2.9 one its `thread_scope` operand (`Subgroup`, `Workgroup`, compute only, or `Device`,
//!   reported to the memory-model family as a fence's is; `thread_scope_thread` orders nothing
//!   another thread observes, a pure execution barrier). The semantics are the barrier's order with
//!   `UniformMemory` (`mem_device`), `WorkgroupMemory` (`mem_threadgroup`, and
//!   `mem_threadgroup_imageblock`: a kernel's explicit imageblock is staged in the threadgroup
//!   arena, and a render stage's is refused), and `ImageMemory` (`mem_texture`); `WorkgroupMemory`
//!   also for `mem_object_data` in an object or mesh function, whose payload it names
//!   (`SPV_EXT_mesh_shader` makes `WorkgroupMemory` cover `TaskPayloadWorkgroupEXT`), and nothing
//!   in any other stage; no flag is a pure execution barrier (`None` semantics). The order is
//!   `AcquireRelease` for `seq_cst` (as in Vulkan, and every AIR 2.8 barrier is `seq_cst`) and
//!   `acq_rel`, `Acquire` or `Release` for MSL 4.1's `acquire` and `release`, and none for
//!   `relaxed`, a pure execution barrier. A barrier that orders no memory takes its execution scope
//!   as its memory scope. Barriers are tangled.
//! - **Simdgroup scope in a kernel** ([`simdgroup_scope`]). A kernel's simdgroup is the subgroup,
//!   or a 32-lane cluster of it, only in the `Direct` and `Cluster32` tiers. In the other tiers
//!   (`Emulated`, and `NoSubgroup`/`Linear`, which pin no subgroup layout) its 32 linear threads
//!   may span subgroups, so a simdgroup barrier or fence is emitted at `Workgroup` scope, which
//!   orders a superset of the simdgroup's threads. In the `Emulated` tier every simdgroup barrier
//!   is in workgroup-uniform control flow (`simd::workgroup`).
//! - **Fences** (`air.atomic.fence`): `OpMemoryBarrier` at the Metal scope (simdgroup →
//!   `Subgroup`, threadgroup → `Workgroup`, device and the legacy all-devices scope → `Device`,
//!   the widest scope of one logical device) with the order (`seq_cst` is `AcquireRelease`, as in
//!   Vulkan). A fence that orders nothing another invocation can observe (a relaxed order, no
//!   memory flag, or thread scope) emits nothing. A `seq_cst` fence, and a program's `seq_cst`
//!   barrier whose memory scope is the device, is also ordered with the command buffer's other
//!   such fences ([`seq_cst_order`]): an acquire-release increment of one status word, so two
//!   threads that each store, fence and load the other's word never both read the old values, as on
//!   Metal.
//! - **Memory model**. Under the Vulkan memory model a barrier or fence that names memory also
//!   makes it available and visible (`MakeAvailable | MakeVisible`); under `GLSL450` it does not
//!   (those bits need the Vulkan model). The semantics are computed at emission from the plan's
//!   model ([`memory_model`]), which the memory-model family chooses; a `Device`-scope fence is
//!   reported to it, since under the Vulkan model it needs `vulkanMemoryModelDeviceScope`.
//! - **Derivatives** (`dfdx`, `dfdy`, `fwidth`, and the `fast_` spellings, whose precision Metal
//!   does not specify either): `OpDPdxFine`, `OpDPdyFine`, `OpFwidthFine` in fragment, since
//!   Metal's are fine: each pixel's x derivative is its own quad row's difference and its y
//!   derivative its own column's (measured on an M3), where an unqualified `OpDPdx` may be a coarse
//!   one a driver reuses for the whole quad. `DerivativeControl` is core in Vulkan. A kernel's are
//!   `OpDPdx`, `OpDPdy`, `OpFwidth` (Metal's kind there is not measured), and need
//!   `computeDerivativeGroupLinear` and use `DerivativeGroupLinearKHR`, because Metal compute quads
//!   are four consecutive linear threads; the kernel then depends on `LocalSizeConstraint{multiple:
//!   4}`, and the flattened simd tiers refuse it (`ComputeDerivativesWithSimd`,
//!   `dispatch::plan_kernel`). A vertex function has no derivatives. Derivatives are tangled.

use crate::air::intrinsics::{
    AirIntrinsic, Derivative, DerivativeOp, MemFlags, MemoryOrder as AirOrder, Sync, ThreadScope,
};
use crate::air::ir::{Callee, Inst, InstId, InstKind, Value, ValueId};
use crate::device::requirements::VkFeature;
use crate::device::subgroup::{OpBits, SimdClass, Tier};
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::{FamilyStrategy, Strategy};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ir::{DerivOp, ExecMode, MemoryOrder, Scope, Semantics};
use std::borrow::Cow;

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Barrier)).into()
}

/// The memory model a module's barriers are emitted under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierModel {
    Glsl450,
    Vulkan,
}

/// The memory model of `plan`'s module, which the memory-model family chooses in its plan
/// (`sync::memory_model`).
pub fn memory_model(plan: &Plan) -> BarrierModel {
    plan.families.memory_model.model
}

/// The semantics of Metal memory flags `flags` at ordering `order` under `model`: `None` (the
/// relaxed, storage-free operand) when no flag names memory.
pub fn semantics(flags: MemFlags, order: MemoryOrder, model: BarrierModel) -> Semantics {
    let s = Semantics {
        order,
        uniform_memory: flags.device,
        workgroup_memory: flags.threadgroup || flags.threadgroup_imageblock || flags.object_data,
        image_memory: flags.texture,
        ..Semantics::RELAXED
    };
    if !s.has_storage() || order == MemoryOrder::Relaxed {
        return Semantics::RELAXED;
    }
    let vulkan = model == BarrierModel::Vulkan;
    let release = matches!(order, MemoryOrder::Release | MemoryOrder::AcquireRelease);
    let acquire = matches!(order, MemoryOrder::Acquire | MemoryOrder::AcquireRelease);
    Semantics {
        make_available: vulkan && release,
        make_visible: vulkan && acquire,
        ..s
    }
}

/// An `OpControlBarrier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Barrier {
    /// `Workgroup` or `Subgroup`: the execution scope.
    pub scope: Scope,
    /// The memory scope: `scope`, unless an AIR 2.9 barrier's `thread_scope` names another and
    /// the barrier orders memory.
    pub memory: Scope,
    pub flags: MemFlags,
    /// The barrier's order at the Vulkan order.
    pub order: MemoryOrder,
    /// Whether the program placed the barrier or `lower::simd::lockstep` inserted it: emitted
    /// alike, named apart in the census.
    pub origin: Origin,
    /// The program's `seq_cst` barrier orders memory at `Device` scope, past its threadgroup:
    /// it is also a sequentially consistent fence there ([`seq_cst_order`]).
    pub seq_cst: bool,
}

/// Who placed a barrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The program.
    Program,
    /// `lower::simd::lockstep`, to order threadgroup or device accesses that only Apple's
    /// simdgroup lockstep ordered.
    Lockstep,
    /// The same, in a kernel whose lockstep order is exact only on emulated simdgroups, so it
    /// takes the `Emulated` tier on every profile (`air::Lockstep::emulated`).
    LockstepEmulated,
}

/// An `OpMemoryBarrier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fence {
    pub scope: Scope,
    pub flags: MemFlags,
    pub order: MemoryOrder,
    /// The fence is `seq_cst`: it is also ordered with the command buffer's other such fences
    /// ([`seq_cst_order`]).
    pub seq_cst: bool,
}

/// A derivative of `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deriv {
    pub op: DerivOp,
    pub value: ValueId,
}

/// The sync::barrier strategies.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BarrierStrategy {
    /// `OpControlBarrier` (threadgroup or simdgroup barrier).
    Barrier(Barrier),
    /// `OpMemoryBarrier` (`air.atomic.fence`).
    Fence(Fence),
    /// A fence that orders nothing another invocation observes: nothing is emitted.
    NoFence,
    /// `OpDPdx`, `OpDPdy`, or `OpFwidth`, or the fine kind of each in fragment.
    Derivative(Deriv),
}

impl FamilyStrategy for BarrierStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            BarrierStrategy::Barrier(b) => match (b.scope, b.origin) {
                (Scope::Subgroup, Origin::Program) => "barrier.simdgroup",
                (_, Origin::Program) => "barrier.threadgroup",
                (Scope::Subgroup, Origin::Lockstep) => "barrier.lockstep.simdgroup",
                (_, Origin::Lockstep) => "barrier.lockstep.threadgroup",
                (Scope::Subgroup, Origin::LockstepEmulated) => {
                    "barrier.lockstep.emulated.simdgroup"
                }
                (_, Origin::LockstepEmulated) => "barrier.lockstep.emulated.threadgroup",
            },
            BarrierStrategy::Fence(f) if f.seq_cst => "barrier.fence.seq_cst",
            BarrierStrategy::Fence(_) => "barrier.fence",
            BarrierStrategy::NoFence => "barrier.fence.none",
            BarrierStrategy::Derivative(_) => "barrier.derivative",
        })
    }

    fn words(&self) -> u32 {
        match self {
            // The instruction and its scope and semantics constants; a sequentially consistent
            // one's status word address and atomic.
            BarrierStrategy::Barrier(b) => 16 + SEQ_CST_WORDS * u32::from(b.seq_cst),
            BarrierStrategy::Fence(f) => 12 + SEQ_CST_WORDS * u32::from(f.seq_cst),
            BarrierStrategy::NoFence => 0,
            BarrierStrategy::Derivative(_) => 4,
        }
    }
}

/// The barrier family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BarrierPlan {
    /// A kernel uses derivatives (`DerivativeGroupLinearKHR`).
    pub compute_derivatives: bool,
    /// The threadgroup barriers `lower::simd::lockstep` inserted: each stands at a place its
    /// placement proved the whole threadgroup reaches together
    /// (`uniform::Uniformity::under_barriers`), so, though tangled, none depends on how subgroups
    /// reconverge, as the emulated tier's exchanges between workgroup barriers in workgroup-uniform
    /// control flow do not. A program's barrier is not among them: it keeps the profile's
    /// reconvergence.
    pub uniform_placed: std::collections::BTreeSet<ValueId>,
    /// `lower::simd::lockstep` inserted a simdgroup barrier: the simd class takes it at
    /// [`lockstep_class`], once the program's own operations have set theirs.
    pub lockstep_simdgroup: bool,
    /// The inserted simdgroup barriers are the kernel's only simdgroup semantics, and they order
    /// only pairs that meet where a simdgroup spans two rows of its threadgroup
    /// (`air::Lockstep::rows`): on a flattened tier the kernel records `RuntimeFact::RowLockstep`.
    pub rows_only: bool,
}

/// Whether tangled instruction `v` depends on how subgroups reconverge: every one but a threadgroup
/// barrier the lockstep pass placed where the threadgroup runs together
/// ([`BarrierPlan::uniform_placed`]).
pub fn reconverges(plan: &BarrierPlan, v: ValueId) -> bool {
    !plan.uniform_placed.contains(&v)
}

/// Raise the simd class of a function the lockstep pass gave simdgroup barriers, once selection
/// has set the class of the program's own operations, and record whether those barriers are its
/// only simdgroup semantics and order only pairs that meet across rows
/// ([`BarrierPlan::rows_only`]): the program's class is then none and it has no simdgroup
/// operation, whose class its family raises later.
pub fn lockstep_class(cx: &mut SelectCx<'_>) {
    if !cx.families.barrier.lockstep_simdgroup {
        return;
    }
    cx.families.barrier.rows_only =
        cx.simd == SimdClass::None && !cx.families.simd.groups && cx.entry.lockstep.rows;
    cx.simd = with_simdgroup_barrier(cx.simd);
}

/// Complete the barrier family's plan: a kernel with derivatives depends on a local size that is
/// a multiple of 4 (one derivative group of four linear invocations).
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.families.barrier.compute_derivatives {
        cx.facts
            .insert(RuntimeFact::LocalSizeConstraint { multiple: 4 });
    }
    Ok(())
}

/// The barrier family's module-scope objects (none).
#[derive(Clone, Debug, Default)]
pub struct Declared {}

/// Add the derivative group mode of a kernel with derivatives to `modes`.
pub fn declare(
    _mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    if plan.families.barrier.compute_derivatives {
        modes.push(ExecMode::DerivativeGroupLinear);
    }
    Ok(Declared::default())
}

/// The class of a function that also uses a simdgroup barrier: cross-lane, which the matrix class
/// already implies.
fn with_simdgroup_barrier(c: SimdClass) -> SimdClass {
    match c {
        SimdClass::None | SimdClass::LaneArithmetic => SimdClass::CrossLane(OpBits::BASIC),
        SimdClass::Quad => SimdClass::CrossLane(OpBits::BASIC.union(OpBits::QUAD)),
        SimdClass::CrossLane(ops) => SimdClass::CrossLane(ops.union(OpBits::BASIC)),
        other => other,
    }
}

fn fence_order(o: AirOrder) -> MemoryOrder {
    match o {
        AirOrder::Relaxed => MemoryOrder::Relaxed,
        AirOrder::Acquire => MemoryOrder::Acquire,
        AirOrder::Release => MemoryOrder::Release,
        AirOrder::AcqRel | AirOrder::SeqCst => MemoryOrder::AcquireRelease,
    }
}

fn tile_shading() -> Error {
    Unsupported::not_implemented(Feature::TileShading).into()
}

/// `flags` as they name a `stage` function's memory: `mem_object_data` names an object or mesh
/// function's payload, which `WorkgroupMemory` covers (`SPV_EXT_mesh_shader`), and no memory of any
/// other stage.
fn stage_flags(stage: Stage, flags: MemFlags) -> MemFlags {
    MemFlags {
        object_data: flags.object_data && matches!(stage, Stage::Object | Stage::Mesh),
        ..flags
    }
}

/// Select the strategy of instruction `inst` of this family.
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<BarrierStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(not_lowered());
    };
    let Callee::Air { op, .. } = &call.callee else {
        return Err(not_lowered());
    };
    let id = cx
        .values
        .inst(at)
        .ok_or_else(|| cx.invariant("an instruction out of the value range"))?;
    match op {
        AirIntrinsic::Sync(
            s @ Sync::Barrier {
                flags,
                scope,
                memory,
                ..
            },
        ) => {
            let origin = match (cx.entry.lockstep.inserted(at), cx.entry.lockstep.emulated) {
                (false, _) => Origin::Program,
                (true, false) => Origin::Lockstep,
                (true, true) => Origin::LockstepEmulated,
            };
            let order = s
                .barrier_order()
                .ok_or_else(|| cx.invariant("a barrier without its order"))?;
            let flags = stage_flags(cx.stage, *flags);
            select_barrier(cx, (id, origin), (flags, order), (*scope, *memory))
        }
        AirIntrinsic::Sync(Sync::Fence {
            flags,
            order,
            scope,
        }) => select_fence(cx, stage_flags(cx.stage, *flags), *order, *scope),
        AirIntrinsic::Derivative(d) => select_derivative(cx, id, d, &call.args),
        _ => Err(not_lowered()),
    }
}

/// Select a barrier of execution scope `scope` and memory scope `memory` that orders `flags`' memories
/// at `order`.
fn select_barrier(
    cx: &mut SelectCx<'_>,
    (id, origin): (ValueId, Origin),
    (flags, order): (MemFlags, AirOrder),
    (scope, memory): (ThreadScope, ThreadScope),
) -> Result<BarrierStrategy, Error> {
    let scope = match scope {
        ThreadScope::Threadgroup if cx.stage.has_workgroup() => Scope::Workgroup,
        ThreadScope::Threadgroup => return Err(tile_shading()),
        // An inserted barrier's class is taken after selection ([`lockstep_class`]).
        ThreadScope::Simdgroup if origin != Origin::Program => {
            cx.families.barrier.lockstep_simdgroup = true;
            Scope::Subgroup
        }
        ThreadScope::Simdgroup => {
            cx.simd = with_simdgroup_barrier(cx.simd);
            Scope::Subgroup
        }
        _ => return Err(cx.invariant("a barrier scope decode does not produce")),
    };
    // A threadgroup barrier in a render stage is refused above; subgroup-scope memory is valid
    // in every stage. Threadgroup memory outside a workgroup stage is tile memory.
    if !cx.stage.has_workgroup() && (flags.threadgroup || flags.threadgroup_imageblock) {
        return Err(tile_shading());
    }
    // Memory visible to the thread alone orders nothing another thread observes.
    let air_order = order;
    let order = match memory {
        ThreadScope::Thread => MemoryOrder::Relaxed,
        _ => fence_order(order),
    };
    let memory = if semantics(flags, order, BarrierModel::Glsl450) == Semantics::RELAXED {
        scope
    } else {
        match memory {
            ThreadScope::Thread => scope,
            ThreadScope::Simdgroup => Scope::Subgroup,
            ThreadScope::Threadgroup if cx.stage.has_workgroup() => Scope::Workgroup,
            ThreadScope::Threadgroup => return Err(tile_shading()),
            ThreadScope::Device | ThreadScope::AllDevices => {
                // Under the Vulkan memory model it needs `vulkanMemoryModelDeviceScope`.
                crate::lower::sync::memory_model::note_device_scope(cx);
                Scope::Device
            }
        }
    };
    // A program's `seq_cst` barrier ordering memory past its threadgroup is also a sequentially
    // consistent fence there; within its execution scope the barrier itself orders every
    // participant.
    let seq_cst = origin == Origin::Program
        && air_order == AirOrder::SeqCst
        && memory == Scope::Device
        && order != MemoryOrder::Relaxed;
    if seq_cst {
        require_seq_cst(cx)?;
    }
    cx.tangled.insert(id);
    if origin != Origin::Program && scope == Scope::Workgroup {
        cx.families.barrier.uniform_placed.insert(id);
    }
    Ok(BarrierStrategy::Barrier(Barrier {
        scope,
        memory,
        flags,
        order,
        origin,
        seq_cst,
    }))
}

fn select_fence(
    cx: &mut SelectCx<'_>,
    flags: MemFlags,
    order: AirOrder,
    scope: ThreadScope,
) -> Result<BarrierStrategy, Error> {
    let seq_cst = order == AirOrder::SeqCst;
    let order = fence_order(order);
    let scope = match scope {
        ThreadScope::Thread => return Ok(BarrierStrategy::NoFence),
        ThreadScope::Simdgroup => Scope::Subgroup,
        ThreadScope::Threadgroup if cx.stage.has_workgroup() => Scope::Workgroup,
        ThreadScope::Threadgroup => return Err(tile_shading()),
        ThreadScope::Device | ThreadScope::AllDevices => Scope::Device,
    };
    if !cx.stage.has_workgroup() && (flags.threadgroup || flags.threadgroup_imageblock) {
        return Err(tile_shading());
    }
    if semantics(flags, order, BarrierModel::Glsl450) == Semantics::RELAXED {
        return Ok(BarrierStrategy::NoFence);
    }
    if scope == Scope::Device {
        // Under the Vulkan memory model it needs `vulkanMemoryModelDeviceScope`.
        crate::lower::sync::memory_model::note_device_scope(cx);
    }
    if seq_cst {
        require_seq_cst(cx)?;
    }
    Ok(BarrierStrategy::Fence(Fence {
        scope,
        flags,
        order,
        seq_cst,
    }))
}

/// The words [`seq_cst_order`] adds: the status word's address and the atomic, with its
/// constants.
pub const SEQ_CST_WORDS: u32 = 48;

/// What a sequentially consistent fence needs ([`seq_cst_order`]): the command buffer's
/// `ExecutionStatus` word (`absent::state_status`), and the stage's atomics; a device without them
/// refuses the fence as `SeqCstFence`, naming what it needs.
pub fn require_seq_cst(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    use crate::lower::access::absent::{state_status, status_short, StatusBinding, StatusShort};
    if let Some(StatusShort::Stores(f)) = status_short(cx) {
        return Err(Unsupported::needs(Feature::SeqCstFence, Capability::Feature(f)).into());
    }
    state_status(cx, StatusBinding::Required);
    Ok(())
}

/// Order a sequentially consistent fence of `flags`' memories at `scope` with every other such
/// fence of the command buffer, after the fence's own barrier: an acquire-release increment of
/// the command buffer's `StatusWord::SeqCstFences` at `scope`, whose semantics are the fence's.
/// Vulkan has no sequentially consistent order (its memory model reads `SequentiallyConsistent`
/// as `AcquireRelease`), and an acquire-release fence lets two threads that each store and then
/// load the other's word both read the old values (store buffering), which Metal's `seq_cst`
/// fence forbids (none in 13.1 M pairs on an M3, where a relaxed pair showed 2,684).
/// The increments of one word are totally ordered, and each reads the one before it, so of two
/// such fences the later synchronizes with the earlier: what preceded the earlier fence happens
/// before what follows the later one.
pub fn seq_cst_order(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    scope: Scope,
    flags: MemFlags,
) -> BResult<()> {
    let word = crate::lower::access::absent::status_word(
        cx,
        fb,
        crate::bind::status::StatusWord::SeqCstFences,
    )?;
    let sem = semantics(flags, MemoryOrder::AcquireRelease, memory_model(cx.plan));
    let one = crate::memory::leaf::k32(fb, 1)?;
    fb.atomic(crate::spirv::ir::AtomicOp::IAdd, word, scope, sem, one)?;
    Ok(())
}

fn select_derivative(
    cx: &mut SelectCx<'_>,
    id: ValueId,
    d: &Derivative,
    args: &[Value],
) -> Result<BarrierStrategy, Error> {
    match cx.stage {
        Stage::Fragment => {}
        Stage::Kernel => {
            if !cx.profile.fields().images.compute_derivative_linear {
                return Err(Unsupported::needs(
                    Feature::ComputeDerivatives,
                    Capability::Feature(VkFeature::ComputeDerivativeGroupLinear),
                )
                .into());
            }
            cx.families.barrier.compute_derivatives = true;
        }
        Stage::Vertex => {
            return Err(Unsupported::not_representable(Feature::ComputeDerivatives).into())
        }
        // Task and mesh derivatives (`meshAndTaskShaderDerivatives`) are not lowered.
        Stage::Object | Stage::Mesh => {
            return Err(Unsupported::not_implemented(Feature::ComputeDerivatives).into())
        }
    }
    let [a] = args else {
        return Err(cx.invariant("a derivative without one operand"));
    };
    // The operand and the result share the operand's lowered representation.
    let ty = cx
        .value_ty(*a)
        .ok_or_else(|| cx.invariant("a derivative operand without a type"))?;
    cx.lowered_repr(*a, ty)?;
    let value = cx
        .values
        .id(*a)
        .ok_or_else(|| cx.invariant("a derivative operand out of range"))?;
    // A fragment function's derivatives are Metal's fine ones (its own row's and column's
    // differences, measured on an M3); a kernel's keep the unqualified kind, unmeasured.
    let fine = cx.stage == Stage::Fragment;
    let op = match (d.op, fine) {
        (DerivativeOp::Dfdx, true) => DerivOp::DPdxFine,
        (DerivativeOp::Dfdy, true) => DerivOp::DPdyFine,
        (DerivativeOp::Fwidth, true) => DerivOp::FwidthFine,
        (DerivativeOp::Dfdx, false) => DerivOp::DPdx,
        (DerivativeOp::Dfdy, false) => DerivOp::DPdy,
        (DerivativeOp::Fwidth, false) => DerivOp::Fwidth,
    };
    cx.tangled.insert(id);
    Ok(BarrierStrategy::Derivative(Deriv { op, value }))
}

/// The scope a barrier or fence of Metal scope `scope` is emitted at: a simdgroup scope is
/// `Workgroup` in a kernel whose simdgroups may span subgroups (every tier but `Direct` and
/// `Cluster32`), and the scope itself otherwise.
pub fn simdgroup_scope(plan: &Plan, scope: Scope) -> Scope {
    scope_on(plan.stage, plan.simd.1, scope)
}

/// [`simdgroup_scope`] for a function of `stage` in simd tier `tier`, before its plan exists.
pub fn scope_on(stage: Stage, tier: Tier, scope: Scope) -> Scope {
    let spans = stage == Stage::Kernel && !matches!(tier, Tier::Direct | Tier::Cluster32);
    if scope == Scope::Subgroup && spans {
        Scope::Workgroup
    } else {
        scope
    }
}

/// Whether strategy `s`, in a function of `stage` in simd tier `tier`, is a control barrier
/// emitted at `Workgroup` scope: one every invocation of the workgroup must execute together.
pub fn is_workgroup_barrier(s: &Strategy, stage: Stage, tier: Tier) -> bool {
    matches!(
        s,
        Strategy::Barrier(BarrierStrategy::Barrier(b))
            if scope_on(stage, tier, b.scope) == Scope::Workgroup
    )
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &BarrierStrategy,
) -> BResult<()> {
    let model = memory_model(cx.plan);
    match s {
        BarrierStrategy::Barrier(b) => {
            let sem = semantics(b.flags, b.order, model);
            let scope = simdgroup_scope(cx.plan, b.scope);
            let memory = simdgroup_scope(cx.plan, b.memory);
            fb.control_barrier(scope, memory, sem)?;
            match b.seq_cst {
                true => seq_cst_order(cx, fb, memory, b.flags),
                false => Ok(()),
            }
        }
        BarrierStrategy::Fence(f) => {
            let scope = simdgroup_scope(cx.plan, f.scope);
            fb.memory_barrier(scope, semantics(f.flags, f.order, model))?;
            match f.seq_cst {
                true => seq_cst_order(cx, fb, scope, f.flags),
                false => Ok(()),
            }
        }
        BarrierStrategy::NoFence => Ok(()),
        BarrierStrategy::Derivative(d) => {
            let a = cx.operand(fb, d.value)?;
            let x = fb.derivative(d.op, a)?;
            match cx.plan.values.inst(at) {
                Some(v) => {
                    cx.define(v, x);
                    Ok(())
                }
                None => Err(cx.fail(format!("{at:?} is out of range"))),
            }
        }
    }
}

#[cfg(test)]
mod tests;
