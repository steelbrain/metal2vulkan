//! The draw of the runtime's in-place check of render targets over host pages
//! ([`check_draw_modules`]): one triangle over the viewport, writing zero to its one color. The
//! check draws it, through a scissor of the image's last column, into an image in place over the
//! pages, while the CPU writes the bytes beside the image's texels, so that a device whose draws
//! write back the bytes of a row past its last texels fails the check.
//!
//! - **The vertex stage** is the sample planes merge's (`bind::sample_planes`): `VertexIndex` 0,
//!   1 and 2 at clip positions (-1, -1), (3, -1) and (-1, 3).
//! - **The fragment stage** writes a `vec4` of zeros at `Location` 0: a color of any float or
//!   normalized format takes it as zero in every channel, as the check's pass clears it.

use std::sync::Arc;

use super::sample_planes::merge_vertex_typed_module;
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::Symbol;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    ConstBits, Constant, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule,
};

/// The in-place check's draw modules of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckDrawModules {
    /// The vertex stage: SPIR-V words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V words, entry point `main`, no resources, the output at
    /// `Location` 0 a `vec4` of zeros.
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device: nothing past the shader capability.
    pub requirements: Requirements,
}

/// The draw's fragment stage as a typed module: zero at `Location` 0.
pub(crate) fn fragment_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let output = b.fixed_io(0, IoDir::Output, vec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = |f: &mut FunctionBuilder<'_>| -> BResult<Value> {
        let c = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(0),
        })?;
        f.module().const_value(c)
    };
    let zeros = [zero(&mut f)?, zero(&mut f)?, zero(&mut f)?, zero(&mut f)?];
    let color = f.composite_construct(vec4, &zeros)?;
    let output = f.module().global_value(output)?;
    f.store(output, color, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// `m` verified against `device`'s profile (requirements included) and serialized; `what` names
/// it in an internal error, which a profile without what so plain a stage needs is.
fn finish(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    let exceeds = |c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    };
    profile.check(m.requirements()).map_err(exceeds)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(exceeds)?;
    // The stages name no resource: nothing to number.
    let words = crate::spirv::serialize::relocate(&r, &|_: &Symbol| None)?;
    Ok((words.into(), r.requirements))
}

/// The in-place check's draw modules on `device`: built with the typed builder, checked by the
/// owned verifier against the device's profile (requirements included) and serialized. An
/// internal error where the profile lacks what they need.
pub fn check_draw_modules(device: &Device) -> Result<CheckDrawModules, Error> {
    let (vertex, vertex_needs) = finish(
        device,
        merge_vertex_typed_module()?,
        "in-place check draw's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        fragment_typed_module()?,
        "in-place check draw's fragment stage",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(CheckDrawModules {
        vertex,
        fragment,
        requirements,
    })
}

#[cfg(test)]
mod tests;
