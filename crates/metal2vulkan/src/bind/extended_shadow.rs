//! The `half` shadow of an extended-range color attachment, set from the attachment's codes.
//!
//! Metal's tile memory holds a `BGR10_XR`, `BGRA10_XR` or `RGB10A8_2P_XR10` attachment (and the
//! `_sRGB` twins) as `half` through a render pass, and a `Load` fills it with the `half` of each
//! stored code's value. A consumer that keeps the attachment's shadow for a pass
//! (`BindState::extended_shadows`) sets it before the render pass instance whose draws first read
//! and write it: a `Clear` load by clearing the shadow to the clear color's `half`s
//! (`ExtendedRange::clear_tile_value`), a `Load` by one dispatch of
//! [`extended_shadow_module`], which writes each texel's tile value
//! (`lower::arith::extended_range::tile_value`, the fetch's own decode) into the shadow.
//!
//! **The module** (one per device and form). One `GLCompute` entry `main` with local size
//! ([`ExtendedShadowModule::LOCAL_SIZE`], [`ExtendedShadowModule::LOCAL_SIZE`], 1), no push
//! constants, and two storage images in set 0:
//!
//! | Binding | Image |
//! |---|---|
//! | [`ExtendedShadowModule::CODES_BINDING`] | the attachment's codes: a 2D view of its level and layer in the form's raw format (`ExtendedRange::raw_format`), read |
//! | [`ExtendedShadowModule::SHADOW_BINDING`] | the shadow: a single-sample 2D `R16G16B16A16_SFLOAT` image, written |
//!
//! Invocation `(x, y)` inside the shadow's extent writes texel `(x, y)` of the shadow from texel
//! `(x, y)` of the codes; one past it writes nothing. The consumer dispatches
//! `⌈width / 8⌉ × ⌈height / 8⌉` workgroups over the shadow's extent, which lies inside the
//! attachment's.
//!
//! **Synchronization** (consumer): the codes' last writes → `SHADER_STORAGE_READ` in
//! `COMPUTE_SHADER`, and the shadow's last reads and writes → `SHADER_STORAGE_WRITE` there,
//! before the dispatch; its writes → the fragment shader's reads and writes of the shadow after.

use std::sync::Arc;

use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::arith::extended_range;
use crate::metal::format::ExtendedRange;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{
    BinOp, Dim, ExecMode, ExecModel, ImageOperands, ImageQuery, ImageType, ImageUsage, IoDir,
    MemAccess, StorageClass, TexelMemory, Type, TypedModule,
};

/// The crate-owned shadow load of one device and form, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtendedShadowModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: the raw format's typed storage access
    /// (`shaderStorageImageExtendedFormats` for `R32G32_UINT`).
    pub requirements: Requirements,
}

impl ExtendedShadowModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The attachment's codes, a storage image of the form's raw format.
    pub const CODES_BINDING: u32 = 0;
    /// The shadow, an `R16G16B16A16_SFLOAT` storage image.
    pub const SHADOW_BINDING: u32 = 1;
    /// Invocations of one workgroup along x, and along y.
    pub const LOCAL_SIZE: u32 = 8;
}

/// The binding symbol of the codes ([`ExtendedShadowModule::CODES_BINDING`]).
const CODES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the shadow ([`ExtendedShadowModule::SHADOW_BINDING`]).
const SHADOW: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};

/// The shadow load of `form` as a typed module, before serialization: the module
/// [`extended_shadow_module`] serializes. `None` for a form whose tile memory is unmeasured (a
/// padded or bfloat form) or whose raw format has no typed storage declaration.
pub(crate) fn extended_shadow_typed_module(form: ExtendedRange) -> Option<BResult<TypedModule>> {
    if form.padded() || form.bfloat() {
        return None;
    }
    let declared = crate::metal::format::spirv_image_format(form.raw_format())?;
    Some(build(form, declared))
}

fn build(form: ExtendedRange, declared: crate::spv::ImageFormat) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image = |b: &mut ModuleBuilder, sampled_type, format| {
        b.ty(Type::Image(ImageType {
            sampled_type,
            dim: Dim::D2,
            depth: false,
            arrayed: false,
            multisampled: false,
            usage: ImageUsage::Storage,
            format,
        }))
    };
    let codes_ty = image(&mut b, u32_, declared)?;
    let shadow_ty = image(&mut b, f32_, crate::spv::ImageFormat::Rgba16f)?;
    let codes = b.resource(CODES, StorageClass::UniformConstant, codes_ty)?;
    let shadow = b.resource(SHADOW, StorageClass::UniformConstant, shadow_ty)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let codes = f.module().global_value(codes)?;
    let codes = f.load(codes, MemAccess::default())?;
    let shadow = f.module().global_value(shadow)?;
    let shadow = f.load(shadow, MemAccess::default())?;
    let size = f.query(ImageQuery::Size, shadow)?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let width = f.composite_extract(size, &[0])?;
    let height = f.composite_extract(size, &[1])?;
    let inside_x = f.binary(BinOp::ULessThan, x, width)?;
    let inside_y = f.binary(BinOp::ULessThan, y, height)?;
    let inside = f.binary(BinOp::LogicalAnd, inside_x, inside_y)?;
    // Invocation `(x, y)` inside the shadow sets texel `(x, y)` to the codes' tile value there.
    let load: ArmFn<'_, ()> = Box::new(move |f, _| {
        let pixel = f.composite_construct(uvec2, &[x, y])?;
        let plain = ImageOperands::default();
        let raw = f.read(codes, pixel, &plain, TexelMemory::default())?;
        let value = extended_range::tile_value(f, form, raw)?;
        f.write(shadow, pixel, value, &plain, TexelMemory::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, load, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ExtendedShadowModule::LOCAL_SIZE,
            y: ExtendedShadowModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &crate::pipeline::sym::Symbol) -> Option<u32> {
    use crate::pipeline::sym::Symbol;
    match sym {
        Symbol::DescriptorSet(_) => Some(ExtendedShadowModule::SET),
        Symbol::Binding(s) if *s == CODES => Some(ExtendedShadowModule::CODES_BINDING),
        Symbol::Binding(s) if *s == SHADOW => Some(ExtendedShadowModule::SHADOW_BINDING),
        _ => None,
    }
}

/// The shadow load of `form` on `device` (the module docs): built with the typed builder, checked
/// by the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed bindings of [`ExtendedShadowModule`]. A padded or bfloat form is
/// `NotImplemented(Format)` (its tile memory is unmeasured, so no draw shadows it); a device
/// without what the module needs is `Unsupported`, naming it, under `FramebufferFetch` (the
/// shadowed draws' storage form).
pub fn extended_shadow_module(
    device: &Device,
    form: ExtendedRange,
) -> Result<ExtendedShadowModule, Error> {
    let Some(m) = extended_shadow_typed_module(form) else {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    };
    let profile = device.profile();
    let m = m?;
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(Feature::FramebufferFetch, c)))?;
    let mut meter = crate::budget::Meter::new(&crate::budget::Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::RequirementsExceedProfile,
            format!("the extended shadow load needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ExtendedShadowModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
