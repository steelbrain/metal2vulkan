//! Module-level rules: the call graph, stage rules, derived facts, and the profile rules.

use crate::budget::Meter;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::{Requirements, VkFeature};
use crate::device::subgroup::SubgroupRequirement;
use crate::graph::{Graph, Sccs};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::{geometry_per_vertex, mesh_arrayed};
use crate::spirv::error::Dangling;
use crate::spirv::ids::{BlockId, FuncId, GlobalId};
use crate::spirv::ir::{
    ExecModel, GlobalKind, Inst, IoDir, StorageClass, Terminator, Type, TypedModule, ValueOrigin,
};
use crate::spirv::requirements::{derive, interface, metal_stage, ModuleView};

use super::cfg::{block_of, Defs};
use super::{Ledger, VerifyError, Violation};

/// The static call graph in compressed form: the callees of function `f` are
/// `callees[start[f]..start[f + 1]]`, with duplicates.
pub(super) struct CallGraph {
    start: Vec<u32>,
    callees: Vec<u32>,
    /// Function -> whether it holds a tangled instruction itself.
    direct: Vec<bool>,
    entry: u32,
}

impl CallGraph {
    pub(super) fn build(
        m: &TypedModule,
        meter: &mut Meter,
        ledger: &mut Ledger,
    ) -> Result<CallGraph, VerifyError> {
        let n = m.functions.len();
        let mut start = ledger.filled(meter, n + 1, 0u32)?;
        let mut direct = ledger.filled(meter, n, false)?;
        let mut calls = 0usize;
        for f in &m.functions {
            for b in &f.blocks {
                calls += block_of(m, *b)?
                    .insts
                    .iter()
                    .filter(|x| matches!(x.inst, Inst::Call { .. }))
                    .count();
            }
        }
        let mut callees = ledger.filled(meter, calls, 0u32)?;
        let mut next = 0usize;
        for (fi, f) in m.functions.iter().enumerate() {
            let mut tangled = false;
            for b in &f.blocks {
                for x in &block_of(m, *b)?.insts {
                    tangled |= x.inst.is_tangled();
                    if let Inst::Call { func, .. } = &x.inst {
                        if m.functions.get(func.index()).is_none() {
                            return Err(Violation::Dangling(Dangling::Func(*func)).into());
                        }
                        if let Some(slot) = callees.get_mut(next) {
                            *slot = func.0;
                        }
                        next += 1;
                    }
                }
            }
            if let Some(d) = direct.get_mut(fi) {
                *d = tangled;
            }
            if let Some(s) = start.get_mut(fi + 1) {
                *s = u32::try_from(next).unwrap_or(u32::MAX);
            }
        }
        if m.functions.get(m.entry.func.index()).is_none() {
            return Err(Violation::Dangling(Dangling::Func(m.entry.func)).into());
        }
        Ok(CallGraph {
            start,
            callees,
            direct,
            entry: m.entry.func.0,
        })
    }

    fn callees_of(&self, f: u32) -> &[u32] {
        let s = self.start.get(f as usize).copied().unwrap_or(0) as usize;
        let e = self.start.get(f as usize + 1).copied().unwrap_or(0) as usize;
        self.callees.get(s..e).unwrap_or(&[])
    }

    /// Refuse a cycle in the call graph (Vulkan forbids recursion), then return, per function,
    /// whether it executes a tangled instruction directly or through a call.
    pub(super) fn check_and_propagate(&self, meter: &mut Meter) -> Result<Vec<bool>, VerifyError> {
        let sccs = Sccs::compute(self, meter)?;
        let r = self.propagate(&sccs);
        meter.release(sccs.ledger_bytes());
        r
    }

    fn propagate(&self, sccs: &Sccs<u32>) -> Result<Vec<bool>, VerifyError> {
        let mut tangled = self.direct.clone();
        // Components are numbered callees first (reverse topological order).
        for c in 0..u32::try_from(sccs.len()).unwrap_or(u32::MAX) {
            if sccs.is_cyclic(c) {
                let f = sccs.members(c).next().unwrap_or(0);
                return Err(Violation::Recursion(FuncId::new(f)).into());
            }
            for f in sccs.members(c) {
                let t = self
                    .callees_of(f)
                    .iter()
                    .any(|g| tangled.get(*g as usize).copied().unwrap_or(false));
                if let Some(x) = tangled.get_mut(f as usize) {
                    *x |= t;
                }
            }
        }
        Ok(tangled)
    }
}

impl Graph for CallGraph {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.direct.len()
    }

    fn entry(&self) -> u32 {
        self.entry
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.callees_of(node).iter().copied()
    }
}

/// Every atomic's pointer operand is a place: a variable, or the result of an access chain or a
/// texel pointer. One pass over the instructions, metered per block like [`Defs::build`].
pub(super) fn check_places(
    m: &TypedModule,
    defs: &Defs,
    meter: &mut Meter,
) -> Result<(), VerifyError> {
    for (bi, b) in m.blocks.iter().enumerate() {
        meter.tick(1)?;
        for ptr in b.insts.iter().filter_map(|n| n.inst.atomic_pointer()) {
            let origin = m
                .values
                .get(ptr.index())
                .map(|d| d.origin)
                .ok_or(Violation::Dangling(Dangling::Value(ptr)))?;
            let place = match origin {
                ValueOrigin::Global(_) | ValueOrigin::Var(_) => true,
                ValueOrigin::Inst(_) => defs
                    .block(ptr)
                    .zip(defs.index(ptr))
                    .and_then(|(d, i)| m.blocks.get(d.index())?.insts.get(i))
                    .is_some_and(|n| n.inst.is_place()),
                _ => false,
            };
            if !place {
                return Err(Violation::AtomicNotPlace {
                    ptr,
                    block: BlockId::new(bi as u32),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Kills only in fragment modules, and only in the entry under an interlock mode; every builtin
/// exists in the entry's execution model and direction.
pub(super) fn check_stage(m: &TypedModule) -> Result<(), VerifyError> {
    let interlock = m.entry.modes.iter().any(|x| x.is_interlock());
    for (i, b) in m.blocks.iter().enumerate() {
        if !matches!(b.term, Terminator::Kill | Terminator::TerminateInvocation) {
            continue;
        }
        if m.entry.model != ExecModel::Fragment {
            return Err(Violation::KillForModel(BlockId::new(i as u32)).into());
        }
        if interlock && b.func != m.entry.func {
            return Err(Violation::KillOutsideEntry(b.func).into());
        }
    }
    let mesh_entry = matches!(m.entry.model, ExecModel::TaskEXT | ExecModel::MeshEXT);
    for (gi, g) in m.globals.iter().enumerate() {
        let id = GlobalId::new(gi as u32);
        if g.class == StorageClass::TaskPayloadWorkgroupEXT && !mesh_entry {
            return Err(Violation::PayloadForModel(id).into());
        }
        if let GlobalKind::BuiltIn { builtin, dir } = g.kind {
            // A mesh entry declares its outputs as arrays over its vertices or primitives; a
            // geometry entry its per-vertex inputs, and a tessellation control entry its
            // per-vertex inputs and outputs, as arrays over the primitive's or patch's vertices,
            // and an evaluation entry its per-vertex inputs so; no other model declares that
            // form.
            let per_element = m.entry.model == ExecModel::MeshEXT && dir == IoDir::Output;
            let arrays_vertices = matches!(
                (m.entry.model, dir),
                (
                    ExecModel::Geometry | ExecModel::TessellationEvaluation,
                    IoDir::Input
                ) | (ExecModel::TessellationControl, _)
            );
            let per_vertex = arrays_vertices && geometry_per_vertex(builtin);
            let vertex_form = arrays_vertices && element_array(m, builtin, g.pointee)?;
            // A control entry's arrayed outputs are its per-vertex form, checked above.
            let mesh_form = m.entry.model != ExecModel::TessellationControl
                && mesh_array_form(m, builtin, dir, g.pointee)?;
            if !builtin_in_model(builtin, dir, m.entry.model)
                || per_element != mesh_form
                || per_vertex != vertex_form
            {
                return Err(Violation::BuiltInForModel(id).into());
            }
        }
    }
    Ok(())
}

/// Whether builtin `b`'s variable of type `ty` has a mesh entry's per-element form: an array of
/// the builtin's type (`ClipDistance` and `CullDistance`, already arrays, as arrays of arrays).
fn mesh_array_form(
    m: &TypedModule,
    b: crate::spv::BuiltIn,
    dir: IoDir,
    ty: crate::spirv::ids::TypeId,
) -> Result<bool, VerifyError> {
    if dir != IoDir::Output || !mesh_arrayed(b) {
        return Ok(false);
    }
    element_array(m, b, ty)
}

/// Whether builtin `b`'s variable type `ty` is an array of the builtin's own type (`ClipDistance`,
/// `CullDistance` and the tessellation levels, already arrays, as arrays of arrays).
fn element_array(
    m: &TypedModule,
    b: crate::spv::BuiltIn,
    ty: crate::spirv::ids::TypeId,
) -> Result<bool, VerifyError> {
    use crate::spv::BuiltIn as B;
    let Type::Array { elem, .. } = m.types.get(ty)? else {
        return Ok(false);
    };
    Ok(match b {
        B::ClipDistance | B::CullDistance | B::TessLevelOuter | B::TessLevelInner => {
            matches!(m.types.get(*elem)?, Type::Array { .. })
        }
        _ => true,
    })
}

/// Whether the execution model has `b` in direction `dir` (Vulkan builtin tables, restricted to
/// the builtins the builder declares).
fn builtin_in_model(b: crate::spv::BuiltIn, dir: IoDir, model: ExecModel) -> bool {
    use crate::spv::BuiltIn as B;
    let every_stage_input = matches!(b, B::SubgroupSize | B::SubgroupLocalInvocationId);
    match (model, dir) {
        (_, IoDir::Input) if every_stage_input => true,
        (ExecModel::GLCompute, IoDir::Input) => matches!(
            b,
            B::GlobalInvocationId
                | B::LocalInvocationId
                | B::WorkgroupId
                | B::NumWorkgroups
                | B::LocalInvocationIndex
                | B::NumSubgroups
                | B::SubgroupId
        ),
        (ExecModel::GLCompute, IoDir::Output) => false,
        (ExecModel::Vertex, IoDir::Input) => matches!(
            b,
            B::VertexIndex
                | B::InstanceIndex
                | B::BaseVertex
                | B::BaseInstance
                | B::DrawIndex
                | B::ViewIndex
        ),
        (ExecModel::Vertex, IoDir::Output) => matches!(
            b,
            B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
                | B::Layer
                | B::ViewportIndex
        ),
        (ExecModel::Fragment, IoDir::Input) => matches!(
            b,
            B::FragCoord
                | B::PointCoord
                | B::FrontFacing
                | B::SampleId
                | B::SamplePosition
                | B::SampleMask
                | B::HelperInvocation
                | B::PrimitiveId
                | B::Layer
                | B::ViewportIndex
                | B::ClipDistance
                | B::CullDistance
                | B::ViewIndex
                | B::BaryCoordKHR
                | B::BaryCoordNoPerspKHR
        ),
        (ExecModel::Fragment, IoDir::Output) => {
            matches!(b, B::FragDepth | B::SampleMask | B::FragStencilRefEXT)
        }
        // VUID-ViewIndex-ViewIndex-04401: every graphics model has `ViewIndex`.
        (ExecModel::TessellationEvaluation, IoDir::Input) => {
            matches!(
                b,
                B::TessCoord
                    | B::PrimitiveId
                    | B::PatchVertices
                    | B::ViewIndex
                    | B::Position
                    | B::PointSize
                    | B::ClipDistance
                    | B::CullDistance
            )
        }
        (ExecModel::TessellationEvaluation, IoDir::Output) => matches!(
            b,
            B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
                | B::Layer
                | B::ViewportIndex
        ),
        (ExecModel::TessellationControl, IoDir::Input) => {
            matches!(
                b,
                B::InvocationId
                    | B::PrimitiveId
                    | B::PatchVertices
                    | B::ViewIndex
                    | B::Position
                    | B::PointSize
                    | B::ClipDistance
                    | B::CullDistance
            )
        }
        (ExecModel::TessellationControl, IoDir::Output) => matches!(
            b,
            B::TessLevelOuter
                | B::TessLevelInner
                | B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
        ),
        // An object or mesh entry's workgroup builtins and draw parameters
        // (VUID-DrawIndex-DrawIndex-04207, VUID-ViewIndex-ViewIndex-04401).
        (ExecModel::TaskEXT | ExecModel::MeshEXT, IoDir::Input) => matches!(
            b,
            B::GlobalInvocationId
                | B::LocalInvocationId
                | B::WorkgroupId
                | B::NumWorkgroups
                | B::LocalInvocationIndex
                | B::NumSubgroups
                | B::SubgroupId
                | B::DrawIndex
                | B::ViewIndex
        ),
        (ExecModel::TaskEXT, IoDir::Output) => false,
        (ExecModel::Geometry, IoDir::Input) => matches!(
            b,
            B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
                | B::PrimitiveId
                | B::InvocationId
                | B::ViewIndex
        ),
        (ExecModel::Geometry, IoDir::Output) => matches!(
            b,
            B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
                | B::PrimitiveId
                | B::Layer
                | B::ViewportIndex
        ),
        (ExecModel::MeshEXT, IoDir::Output) => matches!(
            b,
            B::Position
                | B::PointSize
                | B::ClipDistance
                | B::CullDistance
                | B::PrimitiveId
                | B::Layer
                | B::ViewportIndex
                | B::CullPrimitiveEXT
                | B::PrimitivePointIndicesEXT
                | B::PrimitiveLineIndicesEXT
                | B::PrimitiveTriangleIndicesEXT
        ),
    }
}

/// The recorded entry interface is what the module contains, and its requirements are that plus
/// the facts of its own subgroup requirement: a transform that changed the module recorded them
/// again.
pub(super) fn check_derived(m: &TypedModule) -> Result<(), VerifyError> {
    let view = ModuleView {
        types: &m.types,
        consts: &m.consts,
        values: &m.values,
        globals: &m.globals,
        functions: &m.functions,
        blocks: &m.blocks,
        entry: &m.entry,
        enabling: &m.enabling,
        per_primitive: &m.per_primitive,
    };
    if interface(&view)? != m.interface {
        return Err(Violation::InterfaceStale.into());
    }
    let stage = metal_stage(m.entry.model);
    if !requirements_cover(&derive(&view)?, &m.subgroup, stage, &m.requirements) {
        return Err(Violation::RequirementsStale.into());
    }
    Ok(())
}

/// Whether `recorded` states exactly the derived requirements plus the plan facts the module's
/// own subgroup requirement needs: a pinned size's `subgroupSizeControl`,
/// `requiredSubgroupSizeStages` entry, and size range, and full subgroups' stage flag. A plan
/// fact the requirement does not justify, or one it needs and `recorded` lacks, is stale; no
/// other fact is added beside what derivation finds.
fn requirements_cover(
    derived: &Requirements,
    subgroup: &SubgroupRequirement,
    stage: Stage,
    recorded: &Requirements,
) -> bool {
    let mut expected = derived.clone();
    subgroup.add_device_facts(stage, &mut expected);
    expected == *recorded
}

/// The profile rules: no `Workgroup` initializer without `shaderZeroInitializeWorkgroupMemory`, and
/// requirements ⊆ profile.
pub(super) fn profile_rules(m: &TypedModule, p: &DeviceProfile) -> Result<(), VerifyError> {
    let zero_init = p
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderZeroInitializeWorkgroupMemory);
    if !zero_init {
        if let Some(gi) = m
            .globals
            .iter()
            .position(|g| g.class == StorageClass::Workgroup && g.init.is_some())
        {
            return Err(Violation::WorkgroupInitializer(GlobalId::new(gi as u32)).into());
        }
    }
    p.check(&m.requirements)
        .map_err(VerifyError::ExceedsProfile)
}
