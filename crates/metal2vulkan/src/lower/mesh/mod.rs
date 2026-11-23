//! `lower::mesh`: an object function's mesh grid and a mesh function's outputs.
//!
//! An object function is a `TaskEXT` entry and a mesh function a `MeshEXT` entry. Both run in
//! threadgroups like a kernel: their builtins and local size are `lower::dispatch`'s (the local
//! size is a draw-time value, the kernel's specialization constants).
//!
//! **The object function's grid.** `set_threadgroups_per_grid` stores into a workgroup `uvec3`
//! ([`Declared::grid`]). The entry is a wrapper around the walked body ([`wrap_entry`]): thread 0
//! zeroes the grid, the threadgroup synchronizes, the body runs, the threadgroup synchronizes
//! again, and every invocation ends with `OpEmitMeshTasksEXT` of the stored grid and the payload.
//! Metal's answers hold where one thread's store is the last: a grid never set, or set to zero
//! along an axis, launches no mesh threadgroup; the last call of one thread holds. Two threads
//! storing different grids race in Vulkan, where Metal's later call in program order holds
//! (recorded, not emulated). A grid past Metal's limits on a mesh grid launches none, as Metal's
//! does, and so does one past the device's mesh limits, which Vulkan requires of every emitted
//! grid ([`bounded_grid`]; every committed profile with the extension holds Metal's within its
//! own).
//!
//! **The payload** (`object_data`, `air.payload`). An object function's payload is an arena
//! (`memory::arena`, `Arena::Payload`): a `TaskPayloadWorkgroupEXT` array of whole words of its
//! parameter's `air.arg_type_size` (at least the smallest arena), which the object function's
//! threads read and write as threadgroup memory and its wrapper passes to
//! `OpEmitMeshTasksEXT`; a mesh function declares such an array of its own parameter's size, or
//! of its pipeline's object function's arena where that is larger ([`payload_extent`]: Metal sizes
//! the payload by the type the object function declares, and a mesh function reads past its own
//! type within it), and reads it (a store into it is refused: Vulkan's mesh payload is read-only,
//! and Metal's mesh function takes it `const`). A payload pointer is an offset into it, the
//! parameter's 0. Its accesses are word-composed as the
//! threadgroup arena's, under the Vulkan memory model as relaxed `Workgroup` word atomics (no
//! payload access may carry `NonPrivatePointer`); `mem_object_data` orders it with
//! `WorkgroupMemory`. Its arena lies within `maxTaskPayloadSize`, and with the entry's workgroup
//! memory within `maxTaskPayloadAndSharedMemorySize` or `maxMeshPayloadAndSharedMemorySize`
//! ([`check_payload`], [`workgroup_limit`]). A payload pointer stored to memory, or a volatile
//! payload access, is not implemented.
//!
//! **The mesh function's outputs.** A mesh entry declares its outputs at the mesh type's `NV`
//! vertices and `NP` primitives (`OutputVertices`, `OutputPrimitivesEXT`, and the topology), each
//! output an array of one element per vertex or per primitive, and its entry wrapper:
//!
//! - calls `OpSetMeshOutputsEXT(NV, NP)` first, in uniform control flow;
//! - zeroes every primitive's indices (Metal's unwritten index reads 0, so a slot whose indices
//!   were not all written draws only if the indices it holds make a primitive), clears every
//!   primitive's `CullPrimitiveEXT`, and gives every primitive its slot number as its
//!   `PrimitiveId` (Metal's fragment `[[primitive_id]]` where the mesh function writes none: the
//!   primitive's slot in its threadgroup, culled slots keeping their numbers), each invocation
//!   taking every `N`-th slot from its own index, `N` the threadgroup's threads;
//! - zeroes the primitive count, a workgroup word ([`Declared::count`]), which
//!   `set_primitive_count` stores (never called: nothing draws; the last call of one thread
//!   holds);
//! - synchronizes, runs the body, synchronizes again, and culls every slot at or past the count,
//!   so a count past the slots written draws the written slots and nothing else.
//!
//! A point mesh whose vertex type has no `point_size` culls every primitive (Metal draws none);
//! it still writes `PointSize = 1.0` for each vertex, which a point topology needs.
//!
//! The wrapper's words (the grid, the count) take the function's coherence of shared memory
//! (`NonPrivatePointer` under the Vulkan model), so its barriers order them.
//!
//! Each `set_*` writes straight to its output array: `set_position`, `set_point_size`,
//! `set_clip_distance` and `set_vertex_data` at their vertex, `set_primitive_culled`,
//! `set_primitive_id` and `set_primitive_data` at their primitive, `set_index` at its index (the
//! primitive `index / k`, component `index % k`, `k` the topology's indices per primitive), and
//! `set_indices` its `n` indices from the index rounded down to a multiple of `n`. A vertex,
//! primitive, index or clip distance outside the declared arrays is not stored (Metal does not
//! define it; Vulkan's out-of-bounds output store is undefined behavior). `set_point_size` stores
//! a size past Metal's largest point, 511 pixels, as 511 and one not greater than 0 as 1.0, as a
//! vertex function's `[[point_size]]` is stored (`stage_io::metal_point_size`). Vertex data and
//! primitive data are user outputs keyed by their field's user key (a fragment input with the same
//! key reads them, the Metal linking rule); primitive data is decorated `PerPrimitiveEXT` and
//! listed as `IoRole::PrimitiveVarying`.
//!
//! `set_render_target_array_index` and `set_viewport_array_index` write the primitive's `Layer` and
//! `ViewportIndex` by the vertex function's rules for an index past the pass's layers or the
//! draw's viewports: 0 at or past the pass's length or the draw's viewport count, which the
//! function reads from push fields, and the index as written to a raw per-primitive varying, which
//! Metal's fragment function reads ([`target_index`]).
//!
//! Its hooks into `select` and `lower::emit`: the entry ([`plan_entry`]), the handle parameters
//! ([`select_param`]), the operations ([`select`], [`emit`]), the module-scope objects and
//! execution modes ([`declare`]), and the entry wrapper ([`wrap_entry`]).

use crate::air::abi::{
    MeshField, MeshFieldRole, MeshOutputs, MeshTopology, ParamBinding, ParamRole,
};
use crate::air::intrinsics::{AirIntrinsic, Elem, MemFlags, Mesh as MeshOp, NumTy};
use crate::air::ir::{Callee, Const, Inst, InstId, InstKind, Value};
use crate::air::Entry;
use crate::bind::mesh_arguments::{METAL_MESH_GRID_AXIS, METAL_MESH_GRID_TOTAL};
use crate::device::requirements::{VkExtension, VkFeature, VkLimit};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::dispatch::KernelBuiltin;
use crate::lower::emit::EmitCx;
use crate::lower::select::{Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::lower::sync::barrier::{memory_model, semantics};
use crate::lower::sync::memory_model::coherence;
use crate::memory::arena::{self, WorkgroupLimit};
use crate::memory::leaf::Coherence;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::stage_io::{IoNumeric, IoRole, IoType, IoVariable, StageBuiltin};
use crate::pipeline::sym::{AirUserKey, PushField, Stage, VaryingKey};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ExecMode, GlslOp, IoDecor, IoDir, MemAccess, MemoryOrder, Scope,
    StorageClass, Type,
};
use crate::spv::BuiltIn as B;
use std::borrow::Cow;

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Mesh)).into()
}

/// What a mesh function's outputs hold: its mesh type's counts and topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshShape {
    /// `NV`.
    pub vertices: u32,
    /// `NP`.
    pub primitives: u32,
    pub topology: MeshTopology,
}

impl MeshShape {
    /// The indices of one primitive: 1, 2 or 3.
    pub fn per_primitive(self) -> u32 {
        match self.topology {
            MeshTopology::Point => 1,
            MeshTopology::Line => 2,
            MeshTopology::Triangle => 3,
        }
    }
}

/// The element of an output array: a 32-bit scalar or vector, or a `float` array (a vertex's
/// clip distances).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Element {
    /// `lanes` lanes of `numeric`.
    Num {
        numeric: IoNumeric,
        lanes: u8,
    },
    Bool,
    /// A `float` array of this length.
    Floats(u32),
}

impl Element {
    fn spirv_type(self, mb: &mut ModuleBuilder) -> BResult<TypeId> {
        let scalar = |mb: &mut ModuleBuilder, n: IoNumeric| -> BResult<TypeId> {
            match n {
                IoNumeric::Float => mb.ty(Type::Float { width: 32 }),
                IoNumeric::Sint => mb.ty(Type::Int {
                    width: 32,
                    signed: true,
                }),
                _ => mb.ty(Type::Int {
                    width: 32,
                    signed: false,
                }),
            }
        };
        match self {
            Element::Num { numeric, lanes: 1 } => scalar(mb, numeric),
            Element::Num { numeric, lanes } => {
                let elem = scalar(mb, numeric)?;
                mb.ty(Type::Vector { elem, len: lanes })
            }
            Element::Bool => mb.ty(Type::Bool),
            Element::Floats(n) => {
                let elem = mb.ty(Type::Float { width: 32 })?;
                array(mb, elem, n)
            }
        }
    }
}

/// What one output array of a mesh entry is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    Builtin(crate::spv::BuiltIn),
    /// A user output whose `Location` link relocates by `key`.
    User(VaryingKey),
}

/// One output array of a mesh entry: one element per vertex, or per primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Output {
    pub kind: OutputKind,
    pub element: Element,
    pub per_primitive: bool,
}

/// The mesh family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshPlan {
    /// An object function that sets its mesh grid: it declares the grid and its wrapper emits
    /// the mesh tasks. An object function that never sets it emits none.
    pub object: bool,
    /// The limits an object function's wrapper bounds the grid it emits by, along each axis and
    /// in all: Metal's, within the device's mesh limits where the stages are native
    /// ([`bounded_grid`]).
    pub grid_limits: Option<([u32; 3], u32)>,
    /// A mesh function's output shape.
    pub mesh: Option<MeshShape>,
    /// A mesh function's output arrays, in declaration order: [`CULLED`] and [`PRIMITIVE_ID`]
    /// first, then what its operations write.
    pub outputs: Vec<Output>,
    /// A point mesh whose vertex type has no `point_size`: it draws nothing (the module docs).
    pub unsized_points: bool,
    /// An object or mesh function's payload arena's bytes (`arena::payload_bytes` of
    /// [`payload_extent`]), `None` without a payload (`StageInterface::payload`).
    pub payload: Option<u32>,
    /// An object function, or a kernel under `LoweringInputs::partial_threadgroups`: it runs a
    /// partial threadgroup's threads alone where its region says so ([`plan_partial_groups`]).
    pub partial_groups: bool,
}

/// The index in [`MeshPlan::outputs`] of every mesh entry's `CullPrimitiveEXT`.
pub const CULLED: usize = 0;
/// The index in [`MeshPlan::outputs`] of every mesh entry's `PrimitiveId`.
pub const PRIMITIVE_ID: usize = 1;

impl MeshPlan {
    /// The index of an output equal to `o`, adding it when there is none.
    fn output(&mut self, o: Output) -> usize {
        match self.outputs.iter().position(|x| *x == o) {
            Some(i) => i,
            None => {
                self.outputs.push(o);
                self.outputs.len() - 1
            }
        }
    }

    /// The index of builtin output `b`, if the plan has it.
    fn builtin(&self, b: crate::spv::BuiltIn) -> Option<usize> {
        self.outputs
            .iter()
            .position(|o| o.kind == OutputKind::Builtin(b))
    }
}

/// The lowering of one object or mesh operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshStrategy {
    /// `set_threadgroups_per_grid`: the grid is stored into the entry's grid variable.
    Grid,
    /// A per-vertex or per-primitive output's element written whole (position, point size,
    /// vertex or primitive data, culled, primitive id): the output's index in
    /// [`MeshPlan::outputs`].
    Element { output: usize },
    /// `set_clip_distance(vertex, index, value)`: element `index` of the vertex's clip distances.
    ClipDistance { output: usize, count: u32 },
    /// `set_render_target_array_index` or `set_viewport_array_index`: the primitive's `Layer` or
    /// `ViewportIndex` builtin (`output`) holds the index where it is below the `u32` push field
    /// `count` and 0 where it is not, and its raw varying (`raw`) the index as written.
    TargetIndex {
        output: usize,
        raw: usize,
        count: PushField,
    },
    /// `set_index` (`lanes` 1) or `set_indices` (`lanes` 2 or 4).
    Indices { lanes: u32 },
    /// `set_primitive_count`.
    Count,
    /// A write to a field its function-constant predicate gates off: Metal's mesh type has no
    /// such field, and the write stores nothing.
    GatedOff,
}

impl FamilyStrategy for MeshStrategy {
    fn name(&self, _: &crate::lower::select::FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            MeshStrategy::Grid => "mesh.grid",
            MeshStrategy::Element { .. } => "mesh.element",
            MeshStrategy::ClipDistance { .. } => "mesh.clip_distance",
            MeshStrategy::TargetIndex { .. } => "mesh.target_index",
            MeshStrategy::Indices { lanes: 1 } => "mesh.index",
            MeshStrategy::Indices { .. } => "mesh.indices",
            MeshStrategy::Count => "mesh.count",
            MeshStrategy::GatedOff => "mesh.gated_off",
        })
    }

    fn words(&self) -> u32 {
        match self {
            MeshStrategy::GatedOff => 0,
            MeshStrategy::Grid | MeshStrategy::Count => 8,
            MeshStrategy::Element { .. } => 32,
            MeshStrategy::ClipDistance { .. } => 40,
            // Two element stores, the push word's load, and the compare and select.
            MeshStrategy::TargetIndex { .. } => 88,
            MeshStrategy::Indices { lanes } => 48u32.saturating_mul(*lanes),
        }
    }

    fn yields_value(&self) -> bool {
        false
    }
}

/// The mesh output type of the entry's `air.mesh` parameter.
fn mesh_outputs<'e>(cx: &SelectCx<'e>) -> Option<&'e MeshOutputs> {
    cx.entry.ep.params.iter().find_map(|b| match &b.role {
        ParamRole::Mesh(m) => Some(&**m),
        _ => None,
    })
}

/// What an object (`TaskEXT`) entry needs enabled, `VK_EXT_mesh_shader` and `taskShader`, and a
/// mesh (`MeshEXT`) entry, `VK_EXT_mesh_shader` and `meshShader`: the missing ones, as the
/// refusal names them.
fn missing_caps(cx: &SelectCx<'_>) -> Vec<Capability> {
    let enabled = cx.profile.enabled();
    let mut missing = Vec::new();
    if !enabled.extensions.contains(&VkExtension::ExtMeshShader) {
        missing.push(Capability::Extension(VkExtension::ExtMeshShader));
    }
    let feature = match cx.stage {
        Stage::Object => VkFeature::TaskShader,
        _ => VkFeature::MeshShader,
    };
    if !enabled.features.contains(&feature) {
        missing.push(Capability::Feature(feature));
    }
    missing
}

/// Whether profile `p` runs mesh functions: it enables `VK_EXT_mesh_shader` and `meshShader`.
pub fn mesh_stages(p: &DeviceProfile) -> bool {
    let enabled = p.enabled();
    enabled.extensions.contains(&VkExtension::ExtMeshShader)
        && enabled.features.contains(&VkFeature::MeshShader)
}

/// Whether profile `p` compiles an object function folded (`LoweringInputs::folded_groups`), so
/// that its draws run any object grid Metal runs in pieces within the task stage's limits
/// (`bind::mesh_arguments`): it runs mesh functions natively, and has device addresses (the
/// region record), 64-bit integers (the threadgroup's number), `multiDrawIndirect` (a direct draw's
/// pieces, and `maxDrawIndirectCount` at least 65,535) and `drawIndirectCount` (an indirect
/// draw's).
pub fn folds_object_grids(p: &DeviceProfile) -> bool {
    let features = &p.enabled().features;
    mesh_stages(p)
        && p.fields().buffer_access == crate::device::profile::BufferAccess::Addressed
        && features.contains(&VkFeature::ShaderInt64)
        && features.contains(&VkFeature::MultiDrawIndirect)
        && features.contains(&VkFeature::DrawIndirectCount)
}

/// The bytes of the entry's payload (`air.payload`'s `air.arg_type_size`), if it has one.
pub fn payload_size(e: &Entry) -> Option<u64> {
    e.ep.params.iter().find_map(|b| match b.role {
        ParamRole::Payload { .. } => b.ty.size.map(u64::from),
        _ => None,
    })
}

/// The bytes the entry's payload arena holds, if it has a payload: its own ([`payload_size`]),
/// and an object or mesh function's at least its pipeline's (`LoweringInputs::object_payload`).
/// Metal sizes a pipeline's payload by the type its object function declares
/// (`payloadMemoryLength` 0), and a mesh function's read of its payload past the type it declares
/// itself reads the object function's within it (measured on the M3 Max under macOS 27.0.1); a
/// descriptor's `payloadMemoryLength` sizes the payload of both stages (Metal's header).
pub fn payload_extent(cx: &SelectCx<'_>) -> Option<u64> {
    let own = payload_size(cx.entry)?;
    let object = match cx.stage {
        Stage::Object | Stage::Mesh => cx.inputs.object_payload.map_or(0, u64::from),
        _ => 0,
    };
    Some(own.max(object))
}

/// The workgroup memory limit of the entry ([`stage_workgroup_limit`] of its stage and payload).
pub fn workgroup_limit(cx: &SelectCx<'_>) -> WorkgroupLimit {
    stage_workgroup_limit(cx.stage, cx.profile, payload_extent(cx))
}

/// The workgroup memory limit of a `stage` entry on `profile` ([`arena::workgroup_limit`]), with
/// its payload arena's bytes where it has a payload of `payload` bytes (`arena::payload_bytes`):
/// an object or mesh entry's workgroup memory and payload share a limit. The binder
/// checks an object function's arena with the encoder's lengths against it
/// (`RuntimeFact::ObjectThreadgroupArena`).
pub fn stage_workgroup_limit(
    stage: Stage,
    profile: &DeviceProfile,
    payload: Option<u64>,
) -> WorkgroupLimit {
    let w = arena::workgroup_limit(stage, &profile.fields().limits);
    match payload {
        Some(size) => w.with_payload(arena::payload_bytes(size)),
        None => w,
    }
}

/// The check of a payload of `size` bytes: its arena within `maxTaskPayloadSize`, and, with the
/// wrapper's workgroup words, within the stage's limit on its workgroup memory and payload together
/// (the function's own workgroup memory joins that check where it is planned).
pub fn check_payload(cx: &SelectCx<'_>, size: u64) -> Result<(), Error> {
    let bytes = arena::payload_bytes(size);
    let available = cx.profile.fields().limits.mesh.max_task_payload_size;
    if bytes > u64::from(available) {
        return Err(Unsupported::needs(
            Feature::MeshShading,
            Capability::Limit {
                limit: VkLimit::MaxTaskPayloadSize,
                required: bytes,
                available: u64::from(available),
            },
        )
        .into());
    }
    workgroup_limit(cx).check(0)
}

/// Plan an object function's partial threadgroups, and a kernel's under
/// `LoweringInputs::partial_threadgroups`: it reads its region (`PushField::DispatchRegion`) and
/// answers a partial threadgroup of a `drawMeshThreads`, or of a `dispatchThreads`, as Metal runs
/// it, with the threads left (`lower::dispatch`'s partial threadgroups;
/// `RuntimeFact::PartialObjectThreadgroups`, `RuntimeFact::PartialThreadgroups`). Its invocations
/// past them return before the body where the body holds no workgroup barrier; where it holds
/// one, the body is walked predicated from its threadgroup's threads
/// (`simd::divergent::barriers`, `Predication::partial_groups`), so they walk to each barrier
/// without running anything else of it. A kernel's `[[stage_in]]` attributes are fetched at entry
/// after the returns, and a walked invocation's fetch reads a grid thread's element
/// (`lower::stage_io::stage_in`). A tile function, one with compute derivatives (whose quads a
/// returned lane would leave partial) and one with a cooperative imageblock slice write
/// (`texture::cooperative`, whose cells every invocation of the workgroup shares) are refused the
/// kernel's input, `Unsupported{PartialThreadgroups, NotImplemented}`. Called before the push
/// fields are placed.
pub fn plan_partial_groups(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    match cx.stage {
        Stage::Object => {}
        Stage::Kernel if cx.inputs.partial_threadgroups => {
            if crate::lower::access::tile::is_tile(cx)
                || cx.families.barrier.compute_derivatives
                || !cx.families.texture.cooperative.is_empty()
            {
                return Err(Unsupported::not_implemented(Feature::PartialThreadgroups).into());
            }
        }
        _ => return Ok(()),
    }
    cx.families.mesh.partial_groups = true;
    cx.push.insert(PushField::DispatchRegion);
    Ok(())
}

/// Plan the entry: an object function needs `VK_EXT_mesh_shader` and `taskShader` enabled, and a
/// mesh function `VK_EXT_mesh_shader` and `meshShader` (`Unsupported{MeshShading}` naming the
/// missing ones); a mesh function's shape, its always-declared outputs ([`CULLED`],
/// [`PRIMITIVE_ID`]). Nothing for any other stage.
pub fn plan_entry(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if !matches!(cx.stage, Stage::Object | Stage::Mesh) {
        return Ok(());
    }
    let missing = missing_caps(cx);
    if !missing.is_empty() {
        return Err(Unsupported::needs_any_of(Feature::MeshShading, vec![missing]).into());
    }
    let payload = payload_extent(cx)
        .map(|size| u32::try_from(arena::payload_bytes(size)))
        .transpose()
        .map_err(|_| cx.invariant("a payload past a u32's bytes"))?;
    cx.families.mesh.payload = payload;
    if cx.stage != Stage::Mesh {
        return Ok(());
    }
    let Some(out) = mesh_outputs(cx) else {
        return Err(cx.invariant("a mesh function without its mesh parameter"));
    };
    let shape = MeshShape {
        vertices: out.max_vertices,
        primitives: out.max_primitives,
        topology: out.topology,
    };
    if shape.vertices == 0 || shape.primitives == 0 {
        return Err(cx.invariant("a mesh type without vertices or primitives"));
    }
    // `OutputVertices` and `OutputPrimitivesEXT` are the type's `NV` and `NP`, each within the
    // device's limit.
    let limits = &cx.profile.fields().limits.mesh;
    for (limit, required, available) in [
        (
            VkLimit::MaxMeshOutputVertices,
            shape.vertices,
            limits.max_mesh_output_vertices,
        ),
        (
            VkLimit::MaxMeshOutputPrimitives,
            shape.primitives,
            limits.max_mesh_output_primitives,
        ),
    ] {
        if required > available {
            return Err(Unsupported::needs(
                Feature::MeshShading,
                Capability::Limit {
                    limit,
                    required: u64::from(required),
                    available: u64::from(available),
                },
            )
            .into());
        }
    }
    let point_size = out
        .vertex
        .iter()
        .any(|f| matches!(f.role, MeshFieldRole::PointSize) && !f.gated_off);
    // The rasterizer reads a vertex's size for points alone: a line or triangle mesh's size draws
    // nothing, whatever the device's largest point.
    if point_size && shape.topology == MeshTopology::Point {
        crate::lower::stage_io::note_device_point_size(cx);
    }
    let plan = &mut cx.families.mesh;
    plan.mesh = Some(shape);
    plan.unsized_points = shape.topology == MeshTopology::Point && !point_size;
    let culled = plan.output(Output {
        kind: OutputKind::Builtin(B::CullPrimitiveEXT),
        element: Element::Bool,
        per_primitive: true,
    });
    let id = plan.output(Output {
        kind: OutputKind::Builtin(B::PrimitiveId),
        element: uint(1),
        per_primitive: true,
    });
    if (culled, id) != (CULLED, PRIMITIVE_ID) {
        return Err(cx.invariant("the mesh outputs were planned twice"));
    }
    // A point mesh writes `PointSize` for each vertex: its own, or 1.0 where it draws nothing.
    if plan.unsized_points {
        size_output(cx);
    }
    Ok(())
}

fn uint(lanes: u8) -> Element {
    Element::Num {
        numeric: IoNumeric::Uint,
        lanes,
    }
}

/// The per-vertex `PointSize` output.
fn size_output(cx: &mut SelectCx<'_>) -> usize {
    contract(cx, StageBuiltin::PointSize { default: false });
    cx.families.mesh.output(Output {
        kind: OutputKind::Builtin(B::PointSize),
        element: Element::Num {
            numeric: IoNumeric::Float,
            lanes: 1,
        },
        per_primitive: false,
    })
}

/// List `b` among the stage interface's builtins.
fn contract(cx: &mut SelectCx<'_>, b: StageBuiltin) {
    let list = &mut cx.families.stage_io.interface.builtins;
    if !list.contains(&b) {
        list.push(b);
        list.sort();
    }
}

/// Plan a used handle parameter: an object function's mesh grid, or a mesh function's mesh.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    match (&b.role, cx.stage) {
        (ParamRole::MeshGridProperties, Stage::Object) | (ParamRole::Mesh(_), Stage::Mesh) => {
            Ok(())
        }
        _ => Err(cx.invariant("a mesh handle outside its stage")),
    }
}

/// The integer constant `v`.
fn const_u32(cx: &SelectCx<'_>, v: Value) -> Option<u32> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c)?.value {
        Const::Int(b) => u32::try_from(b).ok(),
        _ => None,
    }
}

/// The element of a user output of `ty`, a 32-bit interface value: floats held as `float`
/// (narrow ones widened), integers as `uint` (narrow ones zero-extended), as a vertex function's
/// varyings are.
fn data_element(ty: NumTy) -> Result<(Element, IoType), Error> {
    let (numeric, bits) = match ty.elem {
        Elem::F16 => (IoNumeric::Float, 16),
        Elem::F32 => (IoNumeric::Float, 32),
        Elem::I8 => (IoNumeric::Uint, 8),
        Elem::I16 => (IoNumeric::Uint, 16),
        Elem::I32 => (IoNumeric::Uint, 32),
        _ => return Err(not_lowered()),
    };
    let lanes = u8::try_from(ty.lanes)
        .ok()
        .filter(|l| (1..=4).contains(l))
        .ok_or_else(not_lowered)?;
    Ok((
        Element::Num { numeric, lanes },
        IoType {
            numeric,
            components: lanes,
            source_bits: bits,
        },
    ))
}

/// The user output of the data field in slot `slot` of `fields` (a vertex's, or with
/// `per_primitive` a primitive's), written as `ty`; `None` for a field gated off.
fn data_output(
    cx: &mut SelectCx<'_>,
    fields: &[MeshField],
    slot: u32,
    ty: NumTy,
    per_primitive: bool,
) -> Result<Option<usize>, Error> {
    let field = fields.iter().find_map(|f| match &f.role {
        MeshFieldRole::VertexData { slot: s, key } if !per_primitive && *s == slot => {
            Some((key, f.gated_off))
        }
        MeshFieldRole::PrimitiveData { slot: s, key } if per_primitive && *s == slot => {
            Some((key, f.gated_off))
        }
        _ => None,
    });
    let Some((key, gated_off)) = field else {
        return Err(cx.invariant("a mesh data slot its type does not have"));
    };
    if gated_off {
        return Ok(None);
    }
    let (element, ty) = data_element(ty)?;
    let user = AirUserKey(key.0.clone());
    let vkey = VaryingKey::user(&key.0);
    let role = if per_primitive {
        IoRole::PrimitiveVarying { user }
    } else {
        IoRole::Varying {
            user,
            interpolation: None,
        }
    };
    let var = IoVariable {
        key: vkey.clone(),
        role,
        ty,
    };
    let list = &mut cx.families.stage_io.interface.outputs;
    // One key per output: a second, different write of the same key is not a Metal interface.
    if list.iter().any(|v| v.key == var.key && *v != var) {
        return Err(not_lowered());
    }
    if !list.contains(&var) {
        list.push(var);
    }
    Ok(Some(cx.families.mesh.output(Output {
        kind: OutputKind::User(vkey),
        element,
        per_primitive,
    })))
}

/// Whether the field of `fields` that `is` picks exists and its predicate gates it off.
fn gated_off(fields: &[MeshField], is: impl Fn(&MeshFieldRole) -> bool) -> bool {
    fields.iter().any(|f| is(&f.role) && f.gated_off)
}

/// A primitive's render target slice index or viewport index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetIndex {
    /// `[[render_target_array_index]]`.
    Layer,
    /// `[[viewport_array_index]]`.
    Viewport,
}

/// The strategy of a write of a primitive's render target slice index or viewport index: the
/// vertex function's rules, per primitive. `Layer` holds the index below the pass's length
/// (`PushField::RenderTargetArrayLength`) and 0 at or past it, where Metal renders the pass's first
/// layer; `ViewportIndex` holds it below the draw's viewport count (`PushField::ViewportCount`) and
/// 0 at or past it, where Metal draws through viewport 0. A mesh pipeline draws without multiview,
/// so `Layer` is written in every variant, and in a pass of one layer it holds 0. The raw
/// per-primitive varying holds the index as written, which Metal's fragment function reads.
fn target_index(cx: &mut SelectCx<'_>, which: TargetIndex) -> Result<MeshStrategy, Error> {
    let frag = cx.profile.fields().fragment;
    let (b, contracted, key, role, count) = match which {
        TargetIndex::Layer => {
            if !frag.output_layer {
                return Err(Unsupported::needs(
                    Feature::LayeredRendering,
                    Capability::Feature(VkFeature::ShaderOutputLayer),
                )
                .into());
            }
            cx.facts.insert(RuntimeFact::RenderTargetArrayLength {
                layer_dropped: false,
            });
            (
                B::Layer,
                StageBuiltin::Layer,
                VaryingKey::render_target_array_index(),
                IoRole::RenderTargetArrayIndex,
                PushField::RenderTargetArrayLength,
            )
        }
        TargetIndex::Viewport => {
            if !frag.output_viewport_index {
                return Err(Unsupported::needs(
                    Feature::LayeredRendering,
                    Capability::Feature(VkFeature::ShaderOutputViewportIndex),
                )
                .into());
            }
            cx.facts.insert(RuntimeFact::ViewportCount);
            (
                B::ViewportIndex,
                StageBuiltin::ViewportIndex,
                VaryingKey::viewport_index(),
                IoRole::ViewportIndex,
                PushField::ViewportCount,
            )
        }
    };
    cx.push.insert(count);
    contract(cx, contracted);
    let output = cx.families.mesh.output(Output {
        kind: OutputKind::Builtin(b),
        element: uint(1),
        per_primitive: true,
    });
    let var = IoVariable {
        key: key.clone(),
        role,
        ty: IoType {
            numeric: IoNumeric::Uint,
            components: 1,
            source_bits: 32,
        },
    };
    let list = &mut cx.families.stage_io.interface.outputs;
    if !list.contains(&var) {
        list.push(var);
    }
    let raw = cx.families.mesh.output(Output {
        kind: OutputKind::User(key),
        element: uint(1),
        per_primitive: true,
    });
    Ok(MeshStrategy::TargetIndex { output, raw, count })
}

/// Select the strategy of operation `inst`.
pub fn select(cx: &mut SelectCx<'_>, _at: InstId, inst: &Inst) -> Result<MeshStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(cx.invariant("a non-call routed to the mesh family"));
    };
    let Callee::Air {
        op: AirIntrinsic::Mesh(op),
        ..
    } = &call.callee
    else {
        return Err(cx.invariant("a call routed to the mesh family is no mesh operation"));
    };
    if *op == MeshOp::ThreadgroupsPerGrid {
        if cx.stage != Stage::Object {
            return Err(cx.invariant("a mesh grid outside an object function"));
        }
        let limits = &cx.profile.fields().limits.mesh;
        // Metal's own limits, within the device's mesh limits.
        cx.families.mesh.grid_limits = Some((
            limits
                .max_mesh_work_group_count
                .map(|axis| axis.min(METAL_MESH_GRID_AXIS)),
            limits
                .max_mesh_work_group_total_count
                .min(METAL_MESH_GRID_TOTAL),
        ));
        cx.families.mesh.object = true;
        return Ok(MeshStrategy::Grid);
    }
    let (Some(_), Some(out)) = (cx.families.mesh.mesh, mesh_outputs(cx)) else {
        return Err(cx.invariant("a mesh operation outside a mesh function"));
    };
    let builtin = |cx: &mut SelectCx<'_>, b, element, per_primitive| -> usize {
        cx.families.mesh.output(Output {
            kind: OutputKind::Builtin(b),
            element,
            per_primitive,
        })
    };
    let float = |lanes| Element::Num {
        numeric: IoNumeric::Float,
        lanes,
    };
    // A write to a field its predicate gates off stores nothing.
    let off = match op {
        MeshOp::Position => gated_off(&out.vertex, |r| matches!(r, MeshFieldRole::Position { .. })),
        MeshOp::PointSize => gated_off(&out.vertex, |r| matches!(r, MeshFieldRole::PointSize)),
        MeshOp::ClipDistance => gated_off(&out.vertex, |r| {
            matches!(r, MeshFieldRole::ClipDistance { .. })
        }),
        MeshOp::PrimitiveCulled => gated_off(&out.primitive, |r| {
            matches!(r, MeshFieldRole::PrimitiveCulled)
        }),
        MeshOp::PrimitiveId => {
            gated_off(&out.primitive, |r| matches!(r, MeshFieldRole::PrimitiveId))
        }
        MeshOp::RenderTargetArrayIndex(_) => gated_off(&out.primitive, |r| {
            matches!(r, MeshFieldRole::RenderTargetArrayIndex)
        }),
        MeshOp::ViewportArrayIndex(_) => gated_off(&out.primitive, |r| {
            matches!(r, MeshFieldRole::ViewportArrayIndex)
        }),
        _ => false,
    };
    if off {
        return Ok(MeshStrategy::GatedOff);
    }
    Ok(match op {
        MeshOp::ThreadgroupsPerGrid => return Err(cx.invariant("a grid past its arm")),
        MeshOp::Position => {
            let invariant = out
                .vertex
                .iter()
                .any(|f| matches!(f.role, MeshFieldRole::Position { invariant: true }));
            contract(cx, StageBuiltin::Position { invariant });
            MeshStrategy::Element {
                output: builtin(cx, B::Position, float(4), false),
            }
        }
        MeshOp::PointSize => MeshStrategy::Element {
            output: size_output(cx),
        },
        MeshOp::ClipDistance => {
            let count = out.vertex.iter().find_map(|f| match f.role {
                MeshFieldRole::ClipDistance { array_size } => Some(array_size.unwrap_or(1)),
                _ => None,
            });
            let Some(count) = count.filter(|n| *n > 0) else {
                return Err(cx.invariant("a clip distance its vertex type does not have"));
            };
            if !cx.profile.fields().fragment.clip_distance {
                return Err(Unsupported::needs(
                    Feature::ClipDistance,
                    Capability::Feature(VkFeature::ShaderClipDistance),
                )
                .into());
            }
            let max = cx.profile.fields().limits.max_clip_distances;
            if count > max {
                return Err(Unsupported::needs(
                    Feature::ClipDistance,
                    Capability::Limit {
                        limit: VkLimit::MaxClipDistances,
                        required: u64::from(count),
                        available: u64::from(max),
                    },
                )
                .into());
            }
            contract(cx, StageBuiltin::ClipDistance { count });
            MeshStrategy::ClipDistance {
                output: builtin(cx, B::ClipDistance, Element::Floats(count), false),
                count,
            }
        }
        MeshOp::VertexData(ty) | MeshOp::PrimitiveData(ty) => {
            let per_primitive = matches!(op, MeshOp::PrimitiveData(_));
            let slot = call.args.get(1).and_then(|v| const_u32(cx, *v));
            let Some(slot) = slot else {
                return Err(not_lowered());
            };
            let fields = if per_primitive {
                &out.primitive
            } else {
                &out.vertex
            };
            match data_output(cx, fields, slot, *ty, per_primitive)? {
                Some(output) => MeshStrategy::Element { output },
                None => MeshStrategy::GatedOff,
            }
        }
        MeshOp::Index => MeshStrategy::Indices { lanes: 1 },
        MeshOp::Indices(n @ (2 | 4)) => MeshStrategy::Indices { lanes: *n },
        MeshOp::Indices(_) => return Err(cx.invariant("an index vector of another length")),
        MeshOp::PrimitiveCulled => MeshStrategy::Element { output: CULLED },
        MeshOp::PrimitiveId => MeshStrategy::Element {
            output: PRIMITIVE_ID,
        },
        MeshOp::RenderTargetArrayIndex(_) => target_index(cx, TargetIndex::Layer)?,
        MeshOp::ViewportArrayIndex(_) => target_index(cx, TargetIndex::Viewport)?,
        MeshOp::PrimitiveCount => MeshStrategy::Count,
    })
}

/// The family's module-scope objects.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// One global per [`MeshPlan::outputs`] entry.
    pub outputs: Vec<GlobalId>,
    /// The primitive indices (`PrimitivePoint/Line/TriangleIndicesEXT`).
    pub indices: Option<GlobalId>,
    /// The primitive count, a workgroup `uint`.
    pub count: Option<GlobalId>,
    /// The object function's mesh grid, a workgroup `uvec3`.
    pub grid: Option<GlobalId>,
}

fn u32_ty(mb: &mut ModuleBuilder) -> BResult<TypeId> {
    mb.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// `[elem; n]`, unstrided.
fn array(mb: &mut ModuleBuilder, elem: TypeId, n: u32) -> BResult<TypeId> {
    let u32_ = u32_ty(mb)?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(n)),
    })?;
    mb.ty(Type::Array {
        elem,
        len,
        stride: None,
    })
}

/// Declare the plan's grid, outputs, indices and count, and a mesh entry's execution modes.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let mp = &plan.families.mesh;
    let mut d = Declared::default();
    if mp.object {
        let u32_ = u32_ty(mb)?;
        let uvec3 = mb.ty(Type::Vector { elem: u32_, len: 3 })?;
        d.grid = Some(mb.plain_global(StorageClass::Workgroup, uvec3, None)?);
    }
    let Some(shape) = mp.mesh else {
        return Ok(d);
    };
    modes.push(ExecMode::OutputVertices { n: shape.vertices });
    modes.push(ExecMode::OutputPrimitives {
        n: shape.primitives,
    });
    modes.push(match shape.topology {
        MeshTopology::Point => ExecMode::OutputPoints,
        MeshTopology::Line => ExecMode::OutputLines,
        MeshTopology::Triangle => ExecMode::OutputTriangles,
    });
    for o in &mp.outputs {
        let n = if o.per_primitive {
            shape.primitives
        } else {
            shape.vertices
        };
        let elem = o.element.spirv_type(mb)?;
        let ty = array(mb, elem, n)?;
        let g = match &o.kind {
            OutputKind::Builtin(b) => mb.builtin(*b, IoDir::Output, ty)?,
            OutputKind::User(key) => {
                let deco = IoDecor {
                    packed: true,
                    ..IoDecor::default()
                };
                let g = mb.io(key.clone(), IoDir::Output, ty, deco)?;
                if o.per_primitive {
                    mb.per_primitive(g)?;
                }
                g
            }
        };
        d.outputs.push(g);
    }
    let invariant = plan
        .families
        .stage_io
        .interface
        .builtins
        .contains(&StageBuiltin::Position { invariant: true });
    if let (true, Some(i)) = (invariant, mp.builtin(B::Position)) {
        if let Some(g) = d.outputs.get(i) {
            mb.invariant(*g)?;
        }
    }
    let u32_ = u32_ty(mb)?;
    let (index, b) = match shape.topology {
        MeshTopology::Point => (u32_, B::PrimitivePointIndicesEXT),
        MeshTopology::Line => (
            mb.ty(Type::Vector { elem: u32_, len: 2 })?,
            B::PrimitiveLineIndicesEXT,
        ),
        MeshTopology::Triangle => (
            mb.ty(Type::Vector { elem: u32_, len: 3 })?,
            B::PrimitiveTriangleIndicesEXT,
        ),
    };
    let ty = array(mb, index, shape.primitives)?;
    d.indices = Some(mb.builtin(b, IoDir::Output, ty)?);
    d.count = Some(mb.plain_global(StorageClass::Workgroup, u32_, None)?);
    Ok(d)
}

/// The operands of call `id`.
fn call_args<'a>(cx: &EmitCx<'a>, id: InstId) -> Option<&'a [Value]> {
    let body: &'a crate::air::ir::Body = cx.body;
    match body.insts.get(id).map(|i| &i.kind) {
        Some(InstKind::Call(call)) => Some(&call.args),
        _ => None,
    }
}

/// The value of operand `i` of an operation's operands `args`.
fn arg(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    args: &[Value],
    i: usize,
) -> BResult<SValue> {
    let Some(v) = args.get(i).copied() else {
        return Err(cx.fail("a mesh operation without its operand"));
    };
    let Some(id) = cx.plan.values.id(v) else {
        return Err(cx.fail("a mesh operand out of range"));
    };
    cx.operand(fb, id)
}

fn k32(fb: &mut FunctionBuilder<'_>, n: u32) -> BResult<SValue> {
    let c = fb.u32_const(n)?;
    fb.module().const_value(c)
}

/// The memory operands of a load or (`store`) a store of one of the wrapper's workgroup words
/// (the grid, the count), which other invocations read after a barrier: the
/// function's coherence of shared memory (`NonPrivatePointer` under the Vulkan model).
fn word_access(
    coherence: Coherence,
    fb: &mut FunctionBuilder<'_>,
    store: bool,
) -> BResult<MemAccess> {
    coherence.access(fb, MemAccess::default(), store)
}

/// Run `then` where `cond` holds.
fn when<'f, C>(
    fb: &mut FunctionBuilder<'_>,
    cx: &mut C,
    cond: SValue,
    then: impl FnOnce(&mut FunctionBuilder<'_>, &mut C) -> BResult<()> + 'f,
) -> BResult<()> {
    let arm: ArmFn<'f, C> = Box::new(move |fb, cx| {
        then(fb, cx)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(cx, cond, arm, None)?;
    Ok(())
}

/// Store `value` at element `at` (and `inner`, a vector component or array element within it)
/// of output array `g` of `len` elements, where `at` is inside it.
fn store_element(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    g: GlobalId,
    len: u32,
    at: SValue,
    inner: Option<SValue>,
    value: SValue,
) -> BResult<()> {
    let n = k32(fb, len)?;
    let inside = fb.binary(BinOp::ULessThan, at, n)?;
    when(fb, cx, inside, move |fb, _| {
        let base = fb.module().global_value(g)?;
        let ptr = match inner {
            Some(i) => fb.access_chain(base, &[at, i])?,
            None => fb.access_chain(base, &[at])?,
        };
        fb.store(ptr, value, MemAccess::default())
    })
}

/// Store `value` at element `at` (and `inner`, a vector component or array element within it)
/// of output `output`: its output array's element.
fn store_output(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    output: usize,
    at: SValue,
    inner: Option<SValue>,
    value: SValue,
) -> BResult<()> {
    let (g, len) = output_of(cx, output)?;
    store_element(cx, fb, g, len, at, inner, value)
}

/// The output array of `MeshPlan::outputs[output]` and its length.
fn output_of(cx: &mut EmitCx<'_>, output: usize) -> BResult<(GlobalId, u32)> {
    let mp = &cx.plan.families.mesh;
    let (Some(o), Some(g), Some(shape)) = (
        mp.outputs.get(output),
        cx.decl.families.mesh.outputs.get(output).copied(),
        mp.mesh,
    ) else {
        return Err(cx.fail("a mesh output that was not declared"));
    };
    let len = if o.per_primitive {
        shape.primitives
    } else {
        shape.vertices
    };
    Ok((g, len))
}

/// Store index `value` (a `u32`) at index `at` of the primitives' indices.
fn store_index(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: SValue,
    value: SValue,
) -> BResult<()> {
    let (Some(g), Some(shape)) = (cx.decl.families.mesh.indices, cx.plan.families.mesh.mesh) else {
        return Err(cx.fail("mesh indices that were not declared"));
    };
    let k = shape.per_primitive();
    let len = shape.primitives.saturating_mul(k);
    if k == 1 {
        return store_element(cx, fb, g, len, at, None, value);
    }
    let n = k32(fb, len)?;
    let inside = fb.binary(BinOp::ULessThan, at, n)?;
    let kk = k32(fb, k)?;
    let prim = fb.binary(BinOp::UDiv, at, kk)?;
    let comp = fb.binary(BinOp::UMod, at, kk)?;
    when(fb, cx, inside, move |fb, _| {
        let base = fb.module().global_value(g)?;
        let ptr = fb.access_chain(base, &[prim, comp])?;
        fb.store(ptr, value, MemAccess::default())
    })
}

/// Emit operation `id` under strategy `s`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    id: InstId,
    s: &MeshStrategy,
) -> BResult<()> {
    let Some(args) = call_args(cx, id) else {
        return Err(cx.fail(format!("{id:?} is not a mesh operation")));
    };
    match *s {
        MeshStrategy::GatedOff => Ok(()),
        MeshStrategy::Grid => {
            let Some(g) = cx.decl.families.mesh.grid else {
                return Err(cx.fail("a mesh grid that was not declared"));
            };
            let grid = arg(cx, fb, args, 1)?;
            let p = fb.module().global_value(g)?;
            let access = word_access(coherence(cx.plan), fb, true)?;
            fb.store(p, grid, access)
        }
        MeshStrategy::Count => {
            let Some(g) = cx.decl.families.mesh.count else {
                return Err(cx.fail("a primitive count that was not declared"));
            };
            let n = arg(cx, fb, args, 1)?;
            let p = fb.module().global_value(g)?;
            let access = word_access(coherence(cx.plan), fb, true)?;
            fb.store(p, n, access)
        }
        MeshStrategy::Element { output } => {
            // A data operation's slot precedes its vertex or primitive.
            let user = matches!(
                cx.plan.families.mesh.outputs.get(output).map(|o| &o.kind),
                Some(OutputKind::User(_))
            );
            let (at, value) = if user { (2, 3) } else { (1, 2) };
            let at = arg(cx, fb, args, at)?;
            let value = arg(cx, fb, args, value)?;
            let value = match cx.plan.families.mesh.outputs.get(output).map(|o| &o.kind) {
                Some(OutputKind::Builtin(B::PointSize)) => {
                    crate::lower::stage_io::metal_point_size(fb, value)?
                }
                _ => value,
            };
            store_output(cx, fb, output, at, None, value)
        }
        MeshStrategy::ClipDistance { output, count } => {
            let at = arg(cx, fb, args, 1)?;
            let index = arg(cx, fb, args, 2)?;
            let value = arg(cx, fb, args, 3)?;
            let n = k32(fb, count)?;
            let inside = fb.binary(BinOp::ULessThan, index, n)?;
            when(fb, cx, inside, move |fb, cx| {
                store_output(cx, fb, output, at, Some(index), value)
            })
        }
        MeshStrategy::TargetIndex { output, raw, count } => {
            let at = arg(cx, fb, args, 1)?;
            // The index, held zero-extended in its `u32` container.
            let value = arg(cx, fb, args, 2)?;
            let n = cx.push_field(fb, count)?;
            let below = fb.binary(BinOp::ULessThan, value, n)?;
            let zero = k32(fb, 0)?;
            let selected = fb.select(below, value, zero)?;
            store_output(cx, fb, output, at, None, selected)?;
            store_output(cx, fb, raw, at, None, value)
        }
        MeshStrategy::Indices { lanes } => {
            let at = arg(cx, fb, args, 1)?;
            let value = arg(cx, fb, args, 2)?;
            if lanes == 1 {
                return store_index(cx, fb, at, value);
            }
            // `set_indices` writes at the index rounded down to a multiple of its length.
            let mask = k32(fb, !(lanes - 1))?;
            let first = fb.binary(BinOp::BitwiseAnd, at, mask)?;
            for j in 0..lanes {
                let lane = fb.composite_extract(value, &[j])?;
                let off = k32(fb, j)?;
                let at = fb.binary(BinOp::IAdd, first, off)?;
                store_index(cx, fb, at, lane)?;
            }
            Ok(())
        }
    }
}

/// The open block's workgroup barrier over threadgroup memory, which also orders the mesh
/// outputs' writes, as GLSL's `barrier()` in a mesh shader does.
fn barrier(cx: &EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let flags = MemFlags {
        threadgroup: true,
        ..Default::default()
    };
    let sem = semantics(flags, MemoryOrder::AcquireRelease, memory_model(cx.plan));
    fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// The mesh grid an object function launches for the grid `g` it set: `g` where it lies within
/// [`MeshPlan::grid_limits`], and none past them. Those are Metal's limits on a mesh grid
/// (`bind::METAL_MESH_GRID_AXIS` along an axis, `bind::METAL_MESH_GRID_TOTAL` in all: past them
/// Metal launches no mesh threadgroup from the object threadgroup, measured on the M3 Max), held
/// within the device's mesh limits where the stages are native (`maxMeshWorkGroupCount` on each
/// axis, `maxMeshWorkGroupTotalCount` in all), which Vulkan requires of every grid a task shader
/// emits.
fn bounded_grid(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    g: [SValue; 3],
) -> BResult<[SValue; 3]> {
    let Some((count, total)) = cx.plan.families.mesh.grid_limits else {
        return Err(cx.fail("an object wrapper without its grid limits"));
    };
    let one = k32(fb, 1)?;
    let [x, y, z] = g;
    // `x·y ≤ total` as `x ≤ total / max(y, 1)`, then `x·y·z ≤ total` as `x·y ≤ total / max(z,
    // 1)`: the product compared wraps only where the first test fails.
    let total = k32(fb, total)?;
    let ny = fb.glsl(GlslOp::UMax, &[y, one])?;
    let room = fb.binary(BinOp::UDiv, total, ny)?;
    let mut all = fb.binary(BinOp::ULessThanEqual, x, room)?;
    let xy = fb.binary(BinOp::IMul, x, y)?;
    let nz = fb.glsl(GlslOp::UMax, &[z, one])?;
    let room = fb.binary(BinOp::UDiv, total, nz)?;
    let c = fb.binary(BinOp::ULessThanEqual, xy, room)?;
    all = fb.binary(BinOp::LogicalAnd, all, c)?;
    for (v, max) in g.into_iter().zip(count) {
        let max = k32(fb, max)?;
        let c = fb.binary(BinOp::ULessThanEqual, v, max)?;
        all = fb.binary(BinOp::LogicalAnd, all, c)?;
    }
    let zero = k32(fb, 0)?;
    Ok([
        fb.select(all, x, zero)?,
        fb.select(all, y, zero)?,
        fb.select(all, z, zero)?,
    ])
}

/// A dispatch builtin the wrapper reads.
fn builtin(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, b: KernelBuiltin) -> BResult<SValue> {
    let decl: &crate::lower::emit::Declared = cx.decl;
    let Some(k) = decl.families.dispatch.kernel.as_ref() else {
        return Err(cx.fail("an object or mesh wrapper without its dispatch"));
    };
    k.value(fb, b, &mut |fb| {
        cx.push_field(fb, PushField::DispatchRegion)
    })
}

/// The builtins an entry wrapper reads (`dispatch::plan_kernel` derives them): its thread and
/// its threadgroup's threads.
pub const BUILTINS: [KernelBuiltin; 2] = [
    KernelBuiltin::ThreadIndexInThreadgroup,
    KernelBuiltin::ThreadsPerThreadgroup,
];

/// Whether the entry has a wrapper that reads [`BUILTINS`].
pub fn wraps(plan: &MeshPlan) -> bool {
    plan.object || plan.mesh.is_some()
}

/// Run `each(k)` for `k = lin, lin + n, ...` below `limit`.
fn strided<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    (lin, n): (SValue, SValue),
    limit: u32,
    each: impl FnOnce(&mut FunctionBuilder<'_>, &mut EmitCx<'c>, SValue) -> BResult<()> + 'c,
) -> BResult<()> {
    let u = u32_ty(fb.module())?;
    let limit = k32(fb, limit)?;
    let var = fb.local_var(u, None)?;
    let counter = fb.var_ptr(var)?;
    fb.store(counter, lin, MemAccess::default())?;
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, limit)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let next = fb.binary(BinOp::IAdd, k, n)?;
        fb.store(counter, next, MemAccess::default())?;
        each(fb, cx, k)
    });
    fb.loop_(cx, LoopSpec::default(), body)
}

/// A constant of `ty` with `bits`, as a value.
fn konst(fb: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<SValue> {
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    fb.module().const_value(c)
}

/// The entry-wrapper hook: an object function's body between zeroing its grid and launching the
/// grid it stored, a mesh function's between initializing its outputs and culling the slots past
/// its count (the module docs); any other function's body is its entry.
pub fn wrap_entry(cx: &mut EmitCx<'_>, mb: &mut ModuleBuilder, body: FuncId) -> BResult<FuncId> {
    let mp = &cx.plan.families.mesh;
    let (object, mesh) = (mp.object, mp.mesh);
    if !object && mesh.is_none() {
        return Ok(body);
    }
    let void = mb.ty(Type::Void)?;
    let mut fb = mb.function(void)?;
    let lin = builtin(cx, &mut fb, KernelBuiltin::ThreadIndexInThreadgroup)?;
    let zero = k32(&mut fb, 0)?;
    let first = fb.binary(BinOp::IEqual, lin, zero)?;
    let coherence = coherence(cx.plan);
    if object {
        let Some(g) = cx.decl.families.mesh.grid else {
            return Err(cx.fail("an object wrapper without its grid"));
        };
        let u32_ = u32_ty(fb.module())?;
        let uvec3 = fb.module().ty(Type::Vector { elem: u32_, len: 3 })?;
        let none = fb.composite_construct(uvec3, &[zero, zero, zero])?;
        when(&mut fb, cx, first, move |fb, _| {
            let p = fb.module().global_value(g)?;
            let access = word_access(coherence, fb, true)?;
            fb.store(p, none, access)
        })?;
        barrier(cx, &mut fb)?;
        fb.call(body, &[])?;
        barrier(cx, &mut fb)?;
        let p = fb.module().global_value(g)?;
        let access = word_access(coherence, &mut fb, false)?;
        let grid = fb.load(p, access)?;
        let [x, y, z] = [0, 1, 2].map(|i| fb.composite_extract(grid, &[i]));
        let [x, y, z] = bounded_grid(cx, &mut fb, [x?, y?, z?])?;
        let payload = cx.decl.families.access.local.payload.and_then(|d| d.global);
        fb.emit_mesh_tasks(x, y, z, payload)?;
        return fb.finish();
    }
    let Some(shape) = mesh else {
        return Err(cx.fail("a mesh wrapper without its shape"));
    };
    let d = &cx.decl.families.mesh;
    let (Some(indices), Some(count)) = (d.indices, d.count) else {
        return Err(cx.fail("a mesh wrapper without its indices or count"));
    };
    let culled = d.outputs.get(CULLED).copied();
    let ids = d.outputs.get(PRIMITIVE_ID).copied();
    let sizes = if cx.plan.families.mesh.unsized_points {
        let i = cx.plan.families.mesh.builtin(B::PointSize);
        i.and_then(|i| d.outputs.get(i).copied())
    } else {
        None
    };
    let (Some(culled), Some(ids)) = (culled, ids) else {
        return Err(cx.fail("a mesh wrapper without its culled flags or primitive ids"));
    };
    let threads = builtin(cx, &mut fb, KernelBuiltin::ThreadsPerThreadgroup)?;
    let [tx, ty, tz] = [0, 1, 2].map(|i| fb.composite_extract(threads, &[i]));
    let txy = fb.binary(BinOp::IMul, tx?, ty?)?;
    let n = fb.binary(BinOp::IMul, txy, tz?)?;
    let nv = k32(&mut fb, shape.vertices)?;
    let np = k32(&mut fb, shape.primitives)?;
    fb.set_mesh_outputs(nv, np)?;
    let bool_ = fb.module().ty(Type::Bool)?;
    let (no, yes) = (konst(&mut fb, bool_, 0)?, konst(&mut fb, bool_, 1)?);
    let index_ty = match shape.per_primitive() {
        1 => u32_ty(fb.module())?,
        k => {
            let u32_ = u32_ty(fb.module())?;
            fb.module().ty(Type::Vector {
                elem: u32_,
                len: k as u8,
            })?
        }
    };
    let none = fb.module().constant(Constant::Null { ty: index_ty })?;
    let none = fb.module().const_value(none)?;
    // The indices are zeroed in place.
    strided(cx, &mut fb, (lin, n), shape.primitives, move |fb, _, p| {
        for (g, v) in [(indices, none), (culled, no), (ids, p)] {
            let base = fb.module().global_value(g)?;
            let ptr = fb.access_chain(base, &[p])?;
            fb.store(ptr, v, MemAccess::default())?;
        }
        Ok(())
    })?;
    if let Some(g) = sizes {
        let f32_ = fb.module().ty(Type::Float { width: 32 })?;
        let one = konst(&mut fb, f32_, u64::from(1f32.to_bits()))?;
        strided(cx, &mut fb, (lin, n), shape.vertices, move |fb, _, v| {
            let base = fb.module().global_value(g)?;
            let ptr = fb.access_chain(base, &[v])?;
            fb.store(ptr, one, MemAccess::default())
        })?;
    }
    when(&mut fb, cx, first, move |fb, _| {
        let p = fb.module().global_value(count)?;
        let access = word_access(coherence, fb, true)?;
        fb.store(p, zero, access)
    })?;
    barrier(cx, &mut fb)?;
    fb.call(body, &[])?;
    barrier(cx, &mut fb)?;
    // A point mesh without point sizes draws nothing: every slot is past its count.
    let drawn = if cx.plan.families.mesh.unsized_points {
        zero
    } else {
        let p = fb.module().global_value(count)?;
        let access = word_access(coherence, &mut fb, false)?;
        fb.load(p, access)?
    };
    strided(cx, &mut fb, (lin, n), shape.primitives, move |fb, cx, p| {
        let past = fb.binary(BinOp::UGreaterThanEqual, p, drawn)?;
        when(fb, cx, past, move |fb, _| {
            let base = fb.module().global_value(culled)?;
            let ptr = fb.access_chain(base, &[p])?;
            fb.store(ptr, yes, MemAccess::default())
        })
    })?;
    fb.ret(None)?;
    fb.finish()
}
