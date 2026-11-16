//! `NonWritable` on the read-only storage resources of graphics stages, and on the unwritten
//! buffers of compute stages.
//!
//! Vulkan requires every storage buffer, storage image, and storage texel buffer variable of a
//! vertex or tessellation (fragment) stage to be decorated `NonWritable` unless the pipeline enables
//! `vertexPipelineStoresAndAtomics` (`fragmentStoresAndAtomics`)
//! (`VUID-RuntimeSpirv-NonWritable-06340`, `-06341`). The requirement derivation already asks for
//! the feature when a module writes such memory (`spirv::requirements`), so a module that only
//! reads it must say so, or it is invalid on a device (or a consumer) without the feature.
//!
//! A compute module's candidates are only the buffer bindings the lowering states unwritten
//! (`ModuleBuilder::unwritten_buffer`: an unaliased Metal buffer argument no write of the function
//! reaches, or a uniform view's storage twin), and the walk below must agree. The decoration says the module does not write the
//! memory, which a device may read as read-only memory: RADV reads a uniform address of it with a
//! scalar load, which an aliasing write would not be ordered with.
//!
//! The decision is structural and conservative: a resource variable is read-only when no store,
//! memory copy, atomic, or image write reaches it or any other variable of its binding (a Metal
//! buffer slot's word view and its narrow views, `ModuleBuilder::resource_view`, are one memory, so
//! a view nothing writes still holds memory another view writes). A pointer or image is traced to
//! its variable through `OpAccessChain`, `OpLoad` of an image, and `OpImageTexelPointer`; a written
//! pointer or image that does not trace to one variable (a function parameter, a phi, a selected
//! value) counts as a write to every variable of its kind, so a variable is never marked
//! `NonWritable` while something may write it.

use std::collections::{BTreeMap, BTreeSet};

use crate::pipeline::sym::ResourceSlot;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    ExecModel, GlobalKind, ImageUsage, Inst, RayQueryInst, StorageClass, Type, TypedModule,
    ValueOrigin,
};

/// Which kind of storage resource a variable is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Buffer,
    Image,
}

/// The resource variables of a graphics module (vertex, tessellation, or fragment) that nothing
/// writes; of a compute module, the variables of its unwritten buffer bindings that nothing writes
/// (see the module docs).
pub(super) fn read_only_resources(m: &TypedModule) -> BResult<BTreeSet<GlobalId>> {
    let mut out = BTreeSet::new();
    let compute = matches!(m.entry.model, ExecModel::GLCompute);
    if compute && m.unwritten_buffers.is_empty() {
        return Ok(out);
    }
    let mut candidates: Vec<(GlobalId, Kind, ResourceSlot)> = Vec::new();
    for (i, g) in m.globals.iter().enumerate() {
        let GlobalKind::Resource(slot) = g.kind else {
            continue;
        };
        if compute && !m.unwritten_buffers.contains(&slot) {
            continue;
        }
        let kind = match g.class {
            StorageClass::StorageBuffer => Some(Kind::Buffer),
            StorageClass::UniformConstant if storage_image(m, g.pointee)? => Some(Kind::Image),
            _ => None,
        };
        if let (Some(kind), Ok(i)) = (kind, u32::try_from(i)) {
            candidates.push((GlobalId::new(i), kind, slot));
        }
    }
    if candidates.is_empty() {
        return Ok(out);
    }
    // What each traced value derives from.
    let mut from: BTreeMap<Value, Value> = BTreeMap::new();
    for b in &m.blocks {
        for node in &b.insts {
            let Some(r) = node.result else { continue };
            match &node.inst {
                Inst::AccessChain { base, .. } => {
                    from.insert(r, *base);
                }
                Inst::ImageTexelPointer { image, .. } => {
                    from.insert(r, *image);
                }
                Inst::Load { ptr, .. } if is_image(m, m.value(r)?.ty)? => {
                    from.insert(r, *ptr);
                }
                _ => {}
            }
        }
    }
    let root = |mut v: Value| -> BResult<Option<GlobalId>> {
        // Each step moves to an operand defined earlier in the derivation; the map is finite, so
        // the walk ends within its size.
        for _ in 0..=from.len() {
            match m.value(v)?.origin {
                ValueOrigin::Global(g) => return Ok(Some(g)),
                ValueOrigin::Inst(_) => match from.get(&v) {
                    Some(b) => v = *b,
                    None => return Ok(None),
                },
                _ => return Ok(None),
            }
        }
        Ok(None)
    };
    let mut written: BTreeSet<GlobalId> = BTreeSet::new();
    let mut every: BTreeSet<Kind> = BTreeSet::new();
    let mut write = |v: Value, kind: Kind| -> BResult<()> {
        match root(v)? {
            Some(g) => {
                written.insert(g);
            }
            None => {
                every.insert(kind);
            }
        }
        Ok(())
    };
    for b in &m.blocks {
        for node in &b.insts {
            match memory_write(&node.inst) {
                Some(Write::Pointer(ptr)) => match pointer_class(m, ptr)? {
                    Some(StorageClass::StorageBuffer) => write(ptr, Kind::Buffer)?,
                    Some(StorageClass::Image) => write(ptr, Kind::Image)?,
                    _ => {}
                },
                Some(Write::Image(image)) => write(image, Kind::Image)?,
                None => {}
            }
        }
    }
    // A binding is written when any of its variables is.
    let written_slots: BTreeSet<ResourceSlot> = candidates
        .iter()
        .filter(|(g, _, _)| written.contains(g))
        .map(|(_, _, slot)| *slot)
        .collect();
    for (g, kind, slot) in candidates {
        if !every.contains(&kind) && !written_slots.contains(&slot) {
            out.insert(g);
        }
    }
    Ok(out)
}

/// The memory an instruction writes.
enum Write {
    /// Through a pointer (a store, memory copy destination, or atomic).
    Pointer(Value),
    /// Through an image value (`OpImageWrite`).
    Image(Value),
}

/// What `inst` writes, if anything. The match is exhaustive on purpose: a new instruction must
/// be classified here, so a write can never be mistaken for a read and its variable decorated
/// `NonWritable`. A call writes nothing itself; the callee's own writes are seen through its
/// parameters, which never trace to a variable.
fn memory_write(inst: &Inst) -> Option<Write> {
    match inst {
        Inst::Store { ptr, .. }
        | Inst::AtomicStore { ptr, .. }
        | Inst::Atomic { ptr, .. }
        | Inst::AtomicCompareExchange { ptr, .. } => Some(Write::Pointer(*ptr)),
        Inst::ImageWrite { image, .. } => Some(Write::Image(*image)),
        // Initializing, proceeding, and confirming update the query's own state.
        Inst::RayQuery(
            RayQueryInst::Initialize { query, .. }
            | RayQueryInst::Proceed { query }
            | RayQueryInst::Confirm { query },
        ) => Some(Write::Pointer(*query)),
        Inst::RayQuery(RayQueryInst::Get { .. }) => None,
        Inst::Load { .. }
        | Inst::Unary { .. }
        | Inst::Binary { .. }
        | Inst::Select { .. }
        | Inst::BitFieldInsert { .. }
        | Inst::BitFieldExtract { .. }
        | Inst::Fma { .. }
        | Inst::Convert { .. }
        | Inst::CompositeConstruct { .. }
        | Inst::CompositeExtract { .. }
        | Inst::CompositeInsert { .. }
        | Inst::VectorExtractDynamic { .. }
        | Inst::VectorInsertDynamic { .. }
        | Inst::VectorShuffle { .. }
        | Inst::Undef
        | Inst::AccessChain { .. }
        | Inst::ArrayLength { .. }
        | Inst::Call { .. }
        | Inst::AtomicLoad { .. }
        | Inst::ImageTexelPointer { .. }
        | Inst::ControlBarrier { .. }
        | Inst::MemoryBarrier { .. }
        | Inst::SampledImage { .. }
        | Inst::Image(_)
        | Inst::ImageSample { .. }
        | Inst::ImageFetch { .. }
        | Inst::ImageGather { .. }
        | Inst::ImageDrefGather { .. }
        | Inst::ImageRead { .. }
        | Inst::ImageSparseTexelsResident(_)
        | Inst::ImageQuery { .. }
        | Inst::ImageQueryLod { .. }
        | Inst::Derivative { .. }
        | Inst::Group { .. }
        | Inst::Glsl { .. }
        | Inst::BeginInterlock
        | Inst::EndInterlock
        | Inst::DemoteToHelperInvocation
        | Inst::IsHelperInvocation
        | Inst::SetMeshOutputs { .. }
        | Inst::EmitVertex
        | Inst::EndPrimitive => None,
    }
}

/// The storage class of pointer value `v` (`None` for a non-pointer).
fn pointer_class(m: &TypedModule, v: Value) -> BResult<Option<StorageClass>> {
    Ok(match m.types.get(m.value(v)?.ty)? {
        Type::Pointer { class, .. } => Some(*class),
        _ => None,
    })
}

fn is_image(m: &TypedModule, t: TypeId) -> BResult<bool> {
    Ok(matches!(m.types.get(t)?, Type::Image(_)))
}

/// Whether `t` is a storage image (or storage texel buffer), or an array of them.
fn storage_image(m: &TypedModule, t: TypeId) -> BResult<bool> {
    let mut t = t;
    loop {
        match m.types.get(t)? {
            Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => t = *elem,
            Type::Image(i) => return Ok(i.usage == ImageUsage::Storage),
            _ => return Ok(false),
        }
    }
}
