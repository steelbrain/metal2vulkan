//! The device self-check module: a storage-buffer binding one entry point uses through **several
//! variables**.
//!
//! A plan's slot declares one variable per view at its binding -- a word view, and the narrow,
//! `uvec2` and `uvec4` unit views an access of it needs (`memory::views`, all of them at the one
//! binding, so the stage counts one descriptor). A consumer that binds such a descriptor
//! *dynamically* (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC`, the offset given as the set is
//! bound) lets commands that differ only in that offset share one descriptor set; a consumer that
//! cannot must give each command a set of its own.
//!
//! Whether a device takes one is not a Vulkan question -- the spec puts no limit on the variables
//! at a binding, dynamic or not -- but a driver that translates SPIR-V to another shader language
//! may handle it wrongly. So the consumer asks the device, by creating one pipeline of exactly
//! that shape and seeing whether it is made: this module is that pipeline's shader.
//!
//! **The module.** One `GLCompute` entry `main` with local size (1, 1, 1), no push constants, and
//! at [`DynamicViewsModule::BINDING`] of set [`DynamicViewsModule::SET`] one storage buffer
//! declared through a word view and **one variable per unit view the profile's storage views can
//! declare** ([`Units::storage_on`]), which is every kind a plan can ask for: so a device that
//! makes this pipeline makes one of any plan's binding, whose variables are a subset of these. The
//! entry references each variable (the length of each view's runtime array, summed and stored
//! through the word view), since a variable a module declares but its entry point does not
//! reference is not one of the binding's (`pipeline::link`'s `count_variables`).

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::memory::leaf::Units;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The crate-owned self-check module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicViewsModule {
    /// SPIR-V 1.5 words, entry point `main`, one binding as below.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device, plus the 8- and
    /// 16-bit storage access its profile's unit views already need).
    pub requirements: Requirements,
}

impl DynamicViewsModule {
    /// The one descriptor set.
    pub const SET: u32 = 0;
    /// The one binding, which the module uses through several variables.
    pub const BINDING: u32 = 0;
    /// Invocations of its one workgroup.
    pub const LOCAL_SIZE: u32 = 1;
    /// Bytes the consumer binds at the binding: enough for the one word the entry stores, and a
    /// multiple of every unit view's stride.
    pub const BYTES: u64 = 64;
}

/// The binding symbol of the views in the typed module.
const VIEWS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// A `Block { elem values[]; }` of `stride`-byte elements.
fn block_of(b: &mut ModuleBuilder, elem: TypeId, stride: u32) -> BResult<TypeId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })
}

/// The self-check module as a typed module, with one variable at [`VIEWS`] per view of `units`
/// beside its word view (the module docs).
fn typed_module(units: Units) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let words = block_of(&mut b, u32_, 4)?;
    let word = b.resource(VIEWS, StorageClass::StorageBuffer, words)?;
    let mut views: Vec<GlobalId> = Vec::new();
    for (declared, width, stride) in [(units.u16, 16, 2), (units.u8, 8, 1)] {
        if !declared {
            continue;
        }
        let narrow = b.ty(Type::Int {
            width,
            signed: false,
        })?;
        let block = block_of(&mut b, narrow, stride)?;
        views.push(b.resource_view(VIEWS, StorageClass::StorageBuffer, block)?);
    }
    for (declared, len) in [(units.vec2, 2u8), (units.vec4, 4)] {
        if !declared {
            continue;
        }
        let uvec = b.ty(Type::Vector { elem: u32_, len })?;
        let block = block_of(&mut b, uvec, 4 * u32::from(len))?;
        views.push(b.resource_view(VIEWS, StorageClass::StorageBuffer, block)?);
    }
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let base = f.module().global_value(word)?;
    // Each view's own runtime-array length, which references its variable and reads no memory.
    let mut total = f.array_length(base, 0)?;
    for view in views {
        let p = f.module().global_value(view)?;
        let len = f.array_length(p, 0)?;
        total = f.binary(BinOp::IAdd, total, len)?;
    }
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let at = f.access_chain(base, &[zero, zero])?;
    f.store(at, total, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DynamicViewsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The set and binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(DynamicViewsModule::SET),
        Symbol::Binding(s) if *s == VIEWS => Some(DynamicViewsModule::BINDING),
        _ => None,
    }
}

/// The self-check module of `device`: built with the typed builder, checked by the owned verifier
/// against the device's profile (requirements included), serialized, and numbered with
/// [`DynamicViewsModule`]'s set and binding.
pub fn dynamic_views_module(device: &Device) -> Result<DynamicViewsModule, Error> {
    let profile = device.profile();
    let m = typed_module(Units::storage_on(profile))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the dynamic-views self-check needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(DynamicViewsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::typed_module;
    use crate::memory::leaf::Units;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// The module is valid on every committed profile, with the units that profile's storage views
    /// can declare.
    #[test]
    fn the_dynamic_views_self_check_is_valid_on_every_committed_profile() {
        for (stem, p) in committed_profiles().expect("profiles") {
            let units = Units::storage_on(&p);
            let m = typed_module(units).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            assert_valid(m, &p);
        }
    }
}
