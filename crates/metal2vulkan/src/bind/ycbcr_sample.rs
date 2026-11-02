//! A two-plane YCbCr texture's texels as the device's own sampler Y′CbCr conversion reads them,
//! written into the texture's image.
//!
//! Where the device converts a texture's planes itself (a `VkSamplerYcbcrConversion` of the
//! planes' multi-planar format, its model, range, chroma offsets and chroma filter), the
//! consumer copies the planes into an image of that format and one dispatch of
//! [`ycbcr_sample_module`] samples it through a sampler of that conversion at the centre of each
//! luma texel, writing each texel into the texture's image in the layout
//! [`ycbcr_convert`](super::ycbcr_convert) writes Metal's: the device's model conversion and its
//! chroma reconstruction take the place of Metal's fixed-point ones, and every pipeline that
//! reads the texture reads its image as before. What a texel holds is the device's answer: Vulkan
//! states the reconstruction's weights (`VkChromaLocation`, `VkFilter`) and the model's matrices,
//! and leaves the precision of each step at least that of the format's codes, a clamp of the
//! result into `[0, 1]` optional.
//!
//! **The module** (one per device and output; the sampler, its conversion and the planes' format
//! are the consumer's layout's, so one module samples any of them). One `GLCompute` entry `main`
//! with local size ([`YCbCrSampleModule::LOCAL_SIZE`], [`YCbCrSampleModule::LOCAL_SIZE`], 1),
//! [`YCbCrSampleModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0
//! ([`YCbCrSampleModule::push`]), and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`YCbCrSampleModule::PLANES_BINDING`] | the planes' image, a 2D combined image sampler, through a sampler of the device's Y′CbCr conversion (immutable, as Vulkan requires), sampled at level 0 |
//! | [`YCbCrSampleModule::IMAGE_BINDING`] | the texture's image, a 2D storage image, written as [`YCbCrConvertModule::IMAGE_BINDING`](super::YCbCrConvertModule::IMAGE_BINDING) states for the output |
//!
//! The push constants' words, in order: the texture's width and height in luma texels, then the
//! planes' image's (each the texture's, rounded up to the format's chroma block). Invocation
//! `(x, y)` inside the texture samples `((x + 0.5) / W, (y + 0.5) / H)` of the image's `W x H` and
//! writes texel `(x, y)` of the sample `(r, g, b)`:
//!
//! - **Converted** ([`YCbCrOutput::Converted`]): `(r, g, b, 1)`, the device's conversion of the
//!   surface's matrix and range.
//! - **sRGB** ([`YCbCrOutput::Srgb`]): each channel `c` of that conversion through the sRGB
//!   curve, `c / 12.92` up to `0.04045` and `((c + 0.055) / 1.055)^2.4` above (`float`
//!   arithmetic), alpha 1.
//! - **Codes** ([`YCbCrOutput::Codes`], a conversion of the identity model): the sample's red
//!   (Cr), green (Y) and blue (Cb) as codes of `b` bits, each `roundEven(clamp(c, 0, 1) · (2^b
//!   - 1))`, packed as the conversion module packs them.
//!
//! An invocation outside the texture writes nothing. The consumer dispatches
//! `⌈w / 8⌉ x ⌈h / 8⌉` workgroups.
//!
//! **Synchronization** (consumer): the planes' copy into the image (`TRANSFER_WRITE`) →
//! `SHADER_SAMPLED_READ` in `COMPUTE_SHADER` before the dispatch; its image writes
//! (`SHADER_STORAGE_WRITE`) → whatever reads the texture after.

use std::sync::Arc;

use super::ycbcr_convert::{scalar, YCbCrCodes, YCbCrOutput};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{
    CombinedSampler, PushField, ResourceKind, ResourceSlot, SamplerRef, Stage, Symbol,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType, ImageUsage, IoDir,
    MemAccess, StorageClass, TexelMemory, Type, TypedModule,
};

/// The crate-owned YCbCr sample module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YCbCrSampleModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl YCbCrSampleModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The planes' image through the device's Y′CbCr conversion: a combined image sampler.
    pub const PLANES_BINDING: u32 = 0;
    /// The texture's image: a 2D storage image, written.
    pub const IMAGE_BINDING: u32 = 1;
    /// Bytes of the `COMPUTE` push-constant range at offset 0 ([`push`](Self::push)).
    pub const PUSH_BYTES: u32 = 16;
    /// Invocations of one workgroup along x, and along y: one texel each.
    pub const LOCAL_SIZE: u32 = 8;

    /// The push bytes of a texture of `size` luma texels over a planes' image of `extent` (the
    /// module docs).
    pub fn push(size: [u32; 2], extent: [u32; 2]) -> [u8; Self::PUSH_BYTES as usize] {
        let words = [size[0], size[1], extent[0], extent[1]];
        let mut bytes = [0; Self::PUSH_BYTES as usize];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the planes' combined image sampler in the typed module.
const PLANES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::CombinedTextureSampler(CombinedSampler {
        sampler: SamplerRef::Slot(0),
        compare: false,
        integer: false,
        unnormalized: false,
    }),
    index: 0,
};
/// The binding symbol of the texture's image in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};

/// The push field the sizes are declared under: a crate-owned module's own words, which no
/// selection places.
const SIZES: PushField = PushField::DispatchRegion;

/// The sample module as a typed module, before serialization, writing `codes`' output.
fn typed_module(codes: YCbCrCodes) -> BResult<TypedModule> {
    let output = codes.output;
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let (sampled_type, format) = match output {
        YCbCrOutput::Converted => (f32_, crate::spv::ImageFormat::Rgba16f),
        YCbCrOutput::Srgb => (f32_, crate::spv::ImageFormat::Rgba32f),
        YCbCrOutput::Codes => (u32_, crate::spv::ImageFormat::R32ui),
    };
    let texel_ty = b.ty(Type::Vector {
        elem: sampled_type,
        len: 4,
    })?;
    let planes_ty = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let combined_ty = b.ty(Type::SampledImage { image: planes_ty })?;
    let planes = b.resource(PLANES, StorageClass::UniformConstant, combined_ty)?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_ty)?;
    let m_sizes = b.push_member(SIZES, uvec4, 0)?;
    let push = b.push_global()?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let float = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v: f32| {
        scalar(f, f32_, u64::from(v.to_bits()))
    };
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let push = f.module().global_value(push)?;
    let member = scalar(&mut f, u32_, u64::from(m_sizes.index))?;
    let sizes = f.access_chain(push, &[member])?;
    let sizes = f.load(sizes, MemAccess::default())?;
    let width = f.composite_extract(sizes, &[0])?;
    let height = f.composite_extract(sizes, &[1])?;
    let across = f.composite_extract(sizes, &[2])?;
    let down = f.composite_extract(sizes, &[3])?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let planes = f.module().global_value(planes)?;
    let image = f.module().global_value(image)?;
    let sampled: ArmFn<'_, ()> = Box::new(move |f, _| {
        // The centre of luma texel `(x, y)` of the planes' image, normalized.
        let half = float(f, 0.5)?;
        let mut coord = Vec::with_capacity(2);
        for (at, extent) in [(x, across), (y, down)] {
            let at = f.convert(ConvOp::ConvertUToF, at, f32_)?;
            let at = f.binary(BinOp::FAdd, at, half)?;
            let extent = f.convert(ConvOp::ConvertUToF, extent, f32_)?;
            coord.push(f.binary(BinOp::FDiv, at, extent)?);
        }
        let coord = f.composite_construct(vec2, &coord)?;
        let combined = f.load(planes, MemAccess::default())?;
        let lod = float(f, 0.0)?;
        let ops = ImageOperands {
            lod: Some(lod),
            ..ImageOperands::default()
        };
        let sample = f.sample(combined, coord, None, false, &ops)?;
        let one = float(f, 1.0)?;
        let rgb: Vec<Value> = (0..3)
            .map(|c| f.composite_extract(sample, &[c]))
            .collect::<BResult<_>>()?;
        let texel = match output {
            YCbCrOutput::Converted => {
                f.composite_construct(vec4, &[rgb[0], rgb[1], rgb[2], one])?
            }
            YCbCrOutput::Srgb => {
                // The sRGB curve, in `float` arithmetic.
                let knee = float(f, 0.04045)?;
                let slope = float(f, 12.92)?;
                let offset = float(f, 0.055)?;
                let scale = float(f, 1.055)?;
                let power = float(f, 2.4)?;
                let mut channels = Vec::with_capacity(4);
                for c in rgb {
                    let low = f.binary(BinOp::FDiv, c, slope)?;
                    let high = f.binary(BinOp::FAdd, c, offset)?;
                    let high = f.binary(BinOp::FDiv, high, scale)?;
                    let high = f.glsl(GlslOp::Pow, &[high, power])?;
                    let below = f.binary(BinOp::FOrdLessThanEqual, c, knee)?;
                    channels.push(f.select(below, low, high)?);
                }
                channels.push(one);
                f.composite_construct(vec4, &channels)?
            }
            YCbCrOutput::Codes => {
                // Red Cr, green Y and blue Cb, each rounded to a code.
                let zero = float(f, 0.0)?;
                let max = (1u32 << codes.bits) - 1;
                let largest = float(f, max as f32)?;
                let mut code = Vec::with_capacity(3);
                for c in rgb {
                    let c = f.glsl(GlslOp::FClamp, &[c, zero, one])?;
                    let c = f.binary(BinOp::FMul, c, largest)?;
                    code.push(f.glsl(GlslOp::RoundEven, &[c])?);
                }
                // `A2B10G10R10_UNORM_PACK32`'s or `R8G8B8A8_UNORM`'s bits, `b` each: red Cr,
                // green Y, blue Cb, and alpha every bit above.
                let mut packed = scalar(f, u32_, u64::from(u32::MAX << (3 * codes.bits)))?;
                for (i, c) in code.into_iter().enumerate() {
                    let c = f.convert(ConvOp::ConvertFToU, c, u32_)?;
                    let shift = scalar(f, u32_, u64::from(i as u32 * codes.bits))?;
                    let c = f.binary(BinOp::ShiftLeftLogical, c, shift)?;
                    packed = f.binary(BinOp::BitwiseOr, packed, c)?;
                }
                let zero = scalar(f, u32_, 0)?;
                f.composite_construct(texel_ty, &[packed, zero, zero, zero])?
            }
        };
        let at = f.composite_construct(uvec2, &[x, y])?;
        let image = f.load(image, MemAccess::default())?;
        f.write(
            image,
            at,
            texel,
            &ImageOperands::default(),
            TexelMemory::PRIVATE,
        )?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, sampled, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: YCbCrSampleModule::LOCAL_SIZE,
            y: YCbCrSampleModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(YCbCrSampleModule::SET),
        Symbol::Binding(s) if *s == PLANES => Some(YCbCrSampleModule::PLANES_BINDING),
        Symbol::Binding(s) if *s == IMAGE => Some(YCbCrSampleModule::IMAGE_BINDING),
        _ => None,
    }
}

/// The YCbCr sample module of `device` that writes the texels of `codes`' output (its
/// [`YCbCrCodes::output`], and its [`YCbCrCodes::bits`] where it reads codes): built with the
/// typed builder, checked by the owned verifier against the device's profile (requirements
/// included), serialized, and numbered with the fixed bindings of [`YCbCrSampleModule`]. The
/// sampler and the planes' image are the consumer's (the module docs), so one module samples
/// every conversion of an output.
pub fn ycbcr_sample_module(device: &Device, codes: YCbCrCodes) -> Result<YCbCrSampleModule, Error> {
    let profile = device.profile();
    let m = typed_module(codes)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the YCbCr sample module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(YCbCrSampleModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::typed_module;
    use crate::bind::YCbCrCodes;
    use crate::metal::format::ycbcr_format;
    use crate::metal::PixelFormat;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Every layout's module on every committed profile.
    #[test]
    fn the_ycbcr_sample_kernel_is_valid_on_every_committed_profile() {
        let mut layouts: Vec<YCbCrCodes> = Vec::new();
        for pf in PixelFormat::ALL.iter().copied() {
            if let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) {
                if !layouts.contains(&codes) {
                    layouts.push(codes);
                }
            }
        }
        assert!(layouts.len() >= 3, "{layouts:?}");
        for (_, p) in committed_profiles().expect("profiles") {
            for &codes in &layouts {
                assert_valid(typed_module(codes).expect("module"), &p);
            }
        }
    }
}
