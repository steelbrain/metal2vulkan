//! The link contract of one function: what link needs from a module without its words, read
//! structurally from the finished [`TypedModule`] before serialization.
//!
//! Every descriptor binding symbol gets the Vulkan descriptor type its variable's storage class
//! and data type declare, and every specialization constant its size and default. The module's
//! user interface variables must be exactly the ones its stage-IO interface ([`StageIo`]) lists, in
//! the listed direction and with the listed type (`StageIo::declared_type`: an extended-range color
//! output's is the raw texel's), so link numbers every `Location` key from that one contract; an
//! emulated mesh function's outputs are records instead. Nothing here reads a name.

use std::collections::BTreeMap;

use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::interface::{StageDescriptor, StageSpec};
use crate::pipeline::stage_io::{IoNumeric, IoType, StageIo};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, VaryingKey};
use crate::spirv::ids::{ConstId, TypeId};
use crate::spirv::ir::{
    ConstKind, Constant, Decl, Dim, ExecModel, GlobalKind, ImageUsage, IoDir, StorageClass, Type,
    TypedModule,
};
use crate::vk;

/// The link-relevant symbols a module declares, each list sorted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageSymbols {
    pub descriptors: Vec<StageDescriptor>,
    pub spec: Vec<StageSpec>,
}

/// Read the link contract of `m` and check that its user interface variables are exactly those `io`
/// lists. A module emitted from a plan always satisfies both; a shape outside them is an invariant
/// failure.
pub fn stage_symbols(m: &TypedModule, io: &StageIo) -> Result<StageSymbols, Error> {
    let mut out = StageSymbols::default();
    let mut inputs: BTreeMap<&VaryingKey, IoType> = BTreeMap::new();
    let mut outputs: BTreeMap<&VaryingKey, IoType> = BTreeMap::new();
    for (vars, listed) in [(&io.inputs, &mut inputs), (&io.outputs, &mut outputs)] {
        for v in vars {
            if listed.insert(&v.key, io.declared_type(v)).is_some() {
                return Err(invariant("a stage-IO key listed twice"));
            }
        }
    }
    let mut declared = 0usize;
    for g in m.globals() {
        match &g.kind {
            GlobalKind::Resource(slot) => {
                out.descriptors
                    .push(descriptor(m, *slot, g.class, g.pointee)?);
            }
            GlobalKind::Io { key, dir, .. } => {
                let listed = match dir {
                    IoDir::Input => &inputs,
                    IoDir::Output => &outputs,
                };
                let ty = listed
                    .get(key)
                    .ok_or_else(|| invariant("a user interface variable the interface omits"))?;
                // A mesh entry's output holds one element per vertex or primitive.
                let pointee = match (m.entry().model, dir, m.types().get(g.pointee)?) {
                    (ExecModel::MeshEXT, IoDir::Output, Type::Array { elem, .. }) => *elem,
                    _ => g.pointee,
                };
                if !io_type_matches(m, pointee, *ty)? {
                    return Err(invariant("a user interface variable of another type"));
                }
                declared += 1;
            }
            GlobalKind::PushBlock
            | GlobalKind::BuiltIn { .. }
            | GlobalKind::Plain
            | GlobalKind::FixedIo { .. } => {}
        }
    }
    // A compute entry's listed outputs are an emulated mesh function's: it writes them to its
    // records binding and declares no interface variable for them.
    let records = m.entry().model == ExecModel::GLCompute && !outputs.is_empty();
    let listed = inputs.len() + if records { 0 } else { outputs.len() };
    if declared != listed {
        return Err(invariant(
            "an interface variable the module does not declare",
        ));
    }
    if records
        && !out
            .descriptors
            .iter()
            .any(|d| d.slot == crate::lower::mesh::RECORDS_SLOT)
    {
        return Err(invariant("a compute entry's outputs without its records"));
    }
    for d in m.decls() {
        if let Decl::Const(c) = d {
            let def = m.consts().get(*c)?;
            if let ConstKind::Spec { key, default } = def.kind {
                out.spec
                    .push(StageSpec::new(key, spec_size(m, def.ty)?, default.0));
            }
        }
    }
    out.descriptors.sort();
    // Several views of one binding (a buffer slot's word and narrow views) are one descriptor;
    // views that disagree on its type are an invariant failure.
    out.descriptors.dedup();
    if out
        .descriptors
        .windows(2)
        .any(|w| matches!(w, [a, b] if a.slot == b.slot))
    {
        return Err(invariant("views of one binding with different descriptors"));
    }
    out.spec.sort();
    Ok(out)
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        format!("link contract: {detail}"),
    ))
}

/// Whether data type `ty` is the scalar or vector `want` declares (32-bit `float` or
/// unsigned-integer lanes; signed-integer lanes for a color output to a signed-integer
/// attachment).
fn io_type_matches(m: &TypedModule, ty: TypeId, want: IoType) -> Result<bool, Error> {
    let types = m.types();
    let (elem, lanes) = match types.get(ty)? {
        Type::Vector { elem, len } => (*elem, *len),
        _ => (ty, 1),
    };
    let numeric = match types.get(elem)? {
        Type::Float { width: 32 } => IoNumeric::Float,
        Type::Int {
            width: 32,
            signed: false,
        } => IoNumeric::Uint,
        Type::Int {
            width: 32,
            signed: true,
        } => IoNumeric::Sint,
        _ => return Ok(false),
    };
    Ok(numeric == want.numeric && lanes == want.components)
}

/// The `VkSpecializationMapEntry` size of a scalar specialization constant: `VkBool32` for a
/// `bool`, the scalar's own size otherwise.
fn spec_size(m: &TypedModule, ty: TypeId) -> Result<u32, Error> {
    match m.types().get(ty)? {
        Type::Bool => Ok(4),
        Type::Int { width, .. } | Type::Float { width } => Ok(u32::from(*width).div_ceil(8)),
        _ => Err(invariant("a non-scalar specialization constant")),
    }
}

/// The value of a plain integer constant (an array length).
fn const_u32(m: &TypedModule, c: ConstId) -> Result<u32, Error> {
    match &m.consts().get(c)?.kind {
        ConstKind::Plain(Constant::Scalar { bits, .. }) => {
            u32::try_from(bits.0).map_err(|_| invariant("an array length beyond u32"))
        }
        _ => Err(invariant("an array length that is not a plain constant")),
    }
}

/// The descriptor a resource variable of storage class `class` and data type `ty` needs.
fn descriptor(
    m: &TypedModule,
    slot: ResourceSlot,
    class: StorageClass,
    ty: TypeId,
) -> Result<StageDescriptor, Error> {
    let types = m.types();
    // A descriptor array's elements share one binding (count = its length). A runtime descriptor
    // array is exactly a set-1 heap array, whose count link sets from the profile: its contract
    // count is 0.
    let heap = matches!(slot.kind, ResourceKind::Heap(_));
    let (elem, count) = match types.get(ty)? {
        Type::Array {
            elem,
            len,
            stride: None,
        } if !heap => (*elem, const_u32(m, *len)?),
        Type::RuntimeArray { stride: None, elem } if heap => (*elem, 0),
        _ if heap => return Err(invariant("a heap binding that is no runtime array")),
        Type::RuntimeArray { stride: None, .. } => {
            return Err(invariant("a runtime descriptor array outside the heap"))
        }
        _ => (ty, 1),
    };
    let dt = match class {
        StorageClass::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        StorageClass::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        StorageClass::UniformConstant => match types.get(elem)? {
            Type::Sampler => vk::DescriptorType::SAMPLER,
            Type::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Type::SampledImage { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Type::Image(img) => match (img.dim, img.usage) {
                (Dim::SubpassData, _) => vk::DescriptorType::INPUT_ATTACHMENT,
                (Dim::Buffer, ImageUsage::Sampled) => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                (Dim::Buffer, ImageUsage::Storage) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                (_, ImageUsage::Sampled) => vk::DescriptorType::SAMPLED_IMAGE,
                (_, ImageUsage::Storage) => vk::DescriptorType::STORAGE_IMAGE,
            },
            _ => return Err(invariant("a non-opaque UniformConstant resource")),
        },
        _ => {
            return Err(invariant(
                "a resource outside the descriptor storage classes",
            ))
        }
    };
    Ok(StageDescriptor::new(slot, dt, count))
}
