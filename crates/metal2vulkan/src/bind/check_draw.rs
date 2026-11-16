//! The draws of the runtime's device checks ([`check_draw_modules`]): one triangle over the
//! viewport. The in-place check of render targets over host pages draws it writing zero to its
//! one color, through a scissor of the image's last column, into an image in place over the
//! pages, while the CPU writes the bytes beside the image's texels, so that a device whose draws
//! write back the bytes of a row past its last texels fails the check. The tiling-cost check
//! draws it over the whole of a target, fetching each texel from a source of the target's size,
//! and writing zero, into a linear image and an optimal one, and times both.
//!
//! - **The vertex stage** is the sample planes merge's (`bind::sample_planes`): `VertexIndex` 0,
//!   1 and 2 at clip positions (-1, -1), (3, -1) and (-1, 3).
//! - **The fragment stage** writes a `vec4` of zeros at `Location` 0: a color of any float or
//!   normalized format takes it as zero in every channel, as the check's pass clears it.
//! - **The fetch stage** fetches the texel at `FragCoord.xy`, level 0, of a sampled 2D image of
//!   `float` texels at set 0, binding [`CheckDrawModules::FETCH_BINDING`], and writes it at
//!   `Location` 0.

use std::sync::Arc;

use super::sample_planes::merge_vertex_typed_module;
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage,
    IoDir, MemAccess, StorageClass, Type, TypedModule,
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
    /// The fetch stage: SPIR-V words, entry point `main`, one sampled 2D image of `float` texels
    /// at set 0, binding [`FETCH_BINDING`](Self::FETCH_BINDING), the output at `Location` 0 a
    /// `vec4`, the texel at `FragCoord.xy`, level 0.
    pub fetch: Arc<[u32]>,
    /// What the two stages need from the device: nothing past the shader capability.
    pub requirements: Requirements,
}

impl CheckDrawModules {
    /// The binding, in set 0, of the fetch stage's sampled image.
    pub const FETCH_BINDING: u32 = 0;
}

/// The fetch stage's sampled image.
const FETCH_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 0,
};

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

/// The fetch stage as a typed module: the texel at `FragCoord.xy`, level 0, of the sampled image
/// [`FETCH_SOURCE`], at `Location` 0.
pub(crate) fn fetch_fragment_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let image = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let source = b.resource(FETCH_SOURCE, StorageClass::UniformConstant, image)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let output = b.fixed_io(0, IoDir::Output, vec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let at = f.composite_construct(uvec2, &[x, y])?;
    let source = f.module().global_value(source)?;
    let source = f.load(source, MemAccess::default())?;
    let level = f.u32_const(0)?;
    let level = f.module().const_value(level)?;
    let ops = ImageOperands {
        lod: Some(level),
        ..ImageOperands::default()
    };
    let texel = f.fetch(source, at, &ops)?;
    let output = f.module().global_value(output)?;
    f.store(output, texel, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The fixed binding numbers of the fetch stage's symbols; the other stages name none.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s) if *s == FETCH_SOURCE => Some(CheckDrawModules::FETCH_BINDING),
        _ => None,
    }
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
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The device checks' draw modules on `device`: built with the typed builder, checked by the
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
    let (fetch, fetch_needs) = finish(
        device,
        fetch_fragment_typed_module()?,
        "tiling-cost check draw's fetch stage",
    )?;
    let profile = device.profile();
    let requirements = profile.union_requirements(
        &profile.union_requirements(&vertex_needs, &fragment_needs),
        &fetch_needs,
    );
    Ok(CheckDrawModules {
        vertex,
        fragment,
        fetch,
        requirements,
    })
}

#[cfg(test)]
mod tests;
