//! The copies between a buffer and a multisampled color texture, through a single-sample "wide"
//! image in which each pixel's samples sit side by side.
//!
//! Vulkan copies no multisampled image to or from a buffer. Metal's
//! `copyFromBuffer:...toTexture:` and `copyFromTexture:...toBuffer:` read and write such a
//! texture per sample, the pixel's samples consecutive in the buffer: texel `(x, y)` sample `s`
//! of an `n`-sample texture lies at `offset + y * bytesPerRow + (x * n + s) * bpp` (measured on
//! an Apple M4 under macOS 27.0.1). So the buffer, read row by row, is a
//! single-sample image `width * n` texels wide, column `x * n + s` holding sample `s` of pixel
//! `x`; the runtime copies the buffer to and from that wide image with ordinary buffer-image
//! copies (one per row, so any `bytesPerRow`, overlapping rows included, is exact) and moves the
//! samples between it and the multisampled texture with these two crate-owned modules, each
//! through an unsigned-integer view of the texel size ([`PlanesCopy`]), so every texel's
//! bits cross unchanged whatever the format. `n` is baked into each module, so the split is one
//! per [`PlanesCopy`] and sample count, and the merge (whose fragment stage writes an unsigned
//! `uvec4` whatever the attachment's format) one per sample count.
//!
//! - **The split** ([`interleave_split_module`], texture to buffer): one `GLCompute` entry `main`
//!   with local size ([`InterleaveSplitModule::LOCAL_SIZE`], [`InterleaveSplitModule::LOCAL_SIZE`],
//!   1) and two images in set 0:
//!
//!   | Binding | Image | View |
//!   |---|---|---|
//!   | [`InterleaveSplitModule::SOURCE_BINDING`] | the multisampled texture | a sampled 2D view of its level and layer, in [`PlanesCopy::format`] |
//!   | [`InterleaveSplitModule::WIDE_BINDING`] | the wide image | a storage 2D view, in [`PlanesCopy::format`] |
//!
//!   Invocation `(x, y, s)` inside the wide image's pixel grid (`width / n` by `height`, `s` in
//!   `0..n`) fetches sample `s` of texel `(x, y)` and writes it to texel `(x * n + s, y)` of the
//!   wide image. Dispatch `ceil((width / n) / LOCAL_SIZE)`, `ceil(height / LOCAL_SIZE)`, `n`
//!   workgroups.
//! - **The merge** ([`interleave_merge_modules`], buffer to texture): the fullscreen-triangle
//!   vertex entry ([`merge_vertex_typed_module`], which the depth/stencil copies share) and a `Fragment`
//!   entry `main` with one sampled 2D image, the wide image in [`PlanesCopy::format`], at
//!   [`InterleaveMergeModules::WIDE_BINDING`] of set 0. Each invocation at texel `(x, y)` sample
//!   `s` fetches wide texel `(x * n + s, y)` and writes it, an unsigned `uvec4`, to the color
//!   output at `Location` 0. Run at sample rate (`sampleShadingEnable`, `minSampleShading` 1)
//!   into a render pass whose one color attachment is the multisampled texture's view in
//!   [`PlanesCopy::format`], over the region, without blending: each invocation writes its own
//!   sample. The fragment module reads `SampleId`, so it needs `sampleRateShading`.
//!
//! **Synchronization** (the runtime): for a copy into the texture, the buffer's host writes
//! before the merge's reads of the wide image, and the texture's earlier accesses before the
//! merge's color writes; for a copy out, the texture's earlier accesses before the split's
//! sampled reads, and the split's writes to the wide image before the image-to-buffer copy.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, ImageOperands, ImageQuery, ImageType,
    ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule,
};
use crate::vk::Format;

/// The unsigned-integer format of a color format's texel size, through which the split and the
/// merge move a texel's bits unchanged (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlanesCopy {
    /// One byte: `R8_UINT`.
    R8,
    /// Two bytes: `R16_UINT`.
    R16,
    /// Four bytes: `R32_UINT`.
    R32,
    /// Eight bytes: `R32G32_UINT`.
    Rg32,
    /// Sixteen bytes: `R32G32B32A32_UINT`.
    Rgba32,
}

impl PlanesCopy {
    /// Every copy format, by texel size.
    pub const ALL: [PlanesCopy; 5] = [
        PlanesCopy::R8,
        PlanesCopy::R16,
        PlanesCopy::R32,
        PlanesCopy::Rg32,
        PlanesCopy::Rgba32,
    ];

    /// The Vulkan format of the copy's views, an unsigned-integer one.
    pub fn format(self) -> Format {
        match self {
            PlanesCopy::R8 => Format::R8_UINT,
            PlanesCopy::R16 => Format::R16_UINT,
            PlanesCopy::R32 => Format::R32_UINT,
            PlanesCopy::Rg32 => Format::R32G32_UINT,
            PlanesCopy::Rgba32 => Format::R32G32B32A32_UINT,
        }
    }

    /// The SPIR-V image format of a storage view in [`PlanesCopy::format`].
    pub(crate) fn spirv_format(self) -> crate::spv::ImageFormat {
        match self {
            PlanesCopy::R8 => crate::spv::ImageFormat::R8ui,
            PlanesCopy::R16 => crate::spv::ImageFormat::R16ui,
            PlanesCopy::R32 => crate::spv::ImageFormat::R32ui,
            PlanesCopy::Rg32 => crate::spv::ImageFormat::Rg32ui,
            PlanesCopy::Rgba32 => crate::spv::ImageFormat::Rgba32ui,
        }
    }
}

/// The crate-owned split module of one copy format and sample count on one device.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterleaveSplitModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: a storage image of an extended format
    /// (`shaderStorageImageExtendedFormats`) for [`PlanesCopy::R8`], [`PlanesCopy::R16`] and
    /// [`PlanesCopy::Rg32`].
    pub requirements: Requirements,
}

impl InterleaveSplitModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The multisampled texture, a sampled image.
    pub const SOURCE_BINDING: u32 = 0;
    /// The wide image, a storage image.
    pub const WIDE_BINDING: u32 = 1;
    /// Invocations of one workgroup along x, and along y.
    pub const LOCAL_SIZE: u32 = 8;
}

/// The crate-owned merge modules of one sample count on one device (one for every copy format:
/// the fragment stage writes an unsigned `uvec4`, which the attachment's format takes).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterleaveMergeModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the wide image at
    /// [`InterleaveMergeModules::WIDE_BINDING`] of set 0, the output at `Location` 0.
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device: `sampleRateShading`.
    pub requirements: Requirements,
}

impl InterleaveMergeModules {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The wide image, a sampled image.
    pub const WIDE_BINDING: u32 = 0;
}

/// The binding symbol of the split's multisampled source.
const SPLIT_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the split's wide image.
const SPLIT_WIDE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};
/// The binding symbol of the merge's wide image in its typed fragment module.
const MERGE_WIDE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 0,
};

/// The unsigned 32-bit scalar type.
fn u32_ty(b: &mut ModuleBuilder) -> BResult<TypeId> {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// An unsigned 2D image type of `usage` and `format`, multisampled as asked (never arrayed: the
/// wide image is one layer, the source a single level and layer).
fn uint_image(
    b: &mut ModuleBuilder,
    usage: ImageUsage,
    format: crate::spv::ImageFormat,
    multisampled: bool,
) -> BResult<TypeId> {
    let sampled_type = u32_ty(b)?;
    b.ty(Type::Image(ImageType {
        sampled_type,
        dim: crate::spirv::ir::Dim::D2,
        depth: false,
        arrayed: false,
        multisampled,
        usage,
        format,
    }))
}

/// The unsigned 32-bit constant `v` as a value.
fn uint(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The split module of `copy` at `samples` as a typed module: its bindings are [`SPLIT_SOURCE`]
/// and [`SPLIT_WIDE`].
pub fn split_typed_module(copy: PlanesCopy, samples: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let unknown = crate::spv::ImageFormat::Unknown;
    let source_ty = uint_image(&mut b, ImageUsage::Sampled, unknown, true)?;
    let wide_ty = uint_image(&mut b, ImageUsage::Storage, copy.spirv_format(), false)?;
    let source = b.resource(SPLIT_SOURCE, StorageClass::UniformConstant, source_ty)?;
    let wide = b.resource(SPLIT_WIDE, StorageClass::UniformConstant, wide_ty)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let wide = f.module().global_value(wide)?;
    let wide = f.load(wide, MemAccess::default())?;
    // The wide image is `width_texels * n` wide and `height` tall; the pixel grid is `width_texels`
    // by `height` by `n`.
    let size = f.query(ImageQuery::Size, wide)?;
    let wide_width = f.composite_extract(size, &[0])?;
    let wide_height = f.composite_extract(size, &[1])?;
    let n = uint(&mut f, samples)?;
    let width_texels = f.binary(BinOp::UDiv, wide_width, n)?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let s = f.composite_extract(invocation, &[2])?;
    let x_in = f.binary(BinOp::ULessThan, x, width_texels)?;
    let y_in = f.binary(BinOp::ULessThan, y, wide_height)?;
    let s_in = f.binary(BinOp::ULessThan, s, n)?;
    let xy = f.binary(BinOp::LogicalAnd, x_in, y_in)?;
    let inside = f.binary(BinOp::LogicalAnd, xy, s_in)?;
    let source = f.module().global_value(source)?;
    let copy_sample: ArmFn<'_, ()> = Box::new(move |f, _| {
        let source = f.load(source, MemAccess::default())?;
        let pixel = f.composite_construct(uvec2, &[x, y])?;
        let ops = ImageOperands {
            sample: Some(s),
            ..ImageOperands::default()
        };
        let texel = f.fetch(source, pixel, &ops)?;
        // The wide column is `x * n + s`.
        let col = f.binary(BinOp::IMul, x, n)?;
        let col = f.binary(BinOp::IAdd, col, s)?;
        let at = f.composite_construct(uvec2, &[col, y])?;
        f.write(
            wide,
            at,
            texel,
            &ImageOperands::default(),
            TexelMemory::default(),
        )?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy_sample, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: InterleaveSplitModule::LOCAL_SIZE,
            y: InterleaveSplitModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The merge's fragment stage at `samples` as a typed module. Its binding is the symbol
/// [`MERGE_WIDE`].
pub fn merge_fragment_typed_module(samples: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let unknown = crate::spv::ImageFormat::Unknown;
    let wide_ty = uint_image(&mut b, ImageUsage::Sampled, unknown, false)?;
    let wide = b.resource(MERGE_WIDE, StorageClass::UniformConstant, wide_ty)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, i32_)?;
    let output = b.fixed_io(0, IoDir::Output, uvec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    let sample = f.module().global_value(sample)?;
    let sample = f.load(sample, MemAccess::default())?;
    let s = f.convert(crate::spirv::ir::ConvOp::Bitcast, sample, u32_)?;
    // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(crate::spirv::ir::ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(crate::spirv::ir::ConvOp::ConvertFToU, y, u32_)?;
    let n = uint(&mut f, samples)?;
    let col = f.binary(BinOp::IMul, x, n)?;
    let col = f.binary(BinOp::IAdd, col, s)?;
    let at = f.composite_construct(uvec2, &[col, y])?;
    let wide = f.module().global_value(wide)?;
    let wide = f.load(wide, MemAccess::default())?;
    let level = uint(&mut f, 0)?;
    let ops = ImageOperands {
        lod: Some(level),
        ..ImageOperands::default()
    };
    let texel = f.fetch(wide, at, &ops)?;
    let output = f.module().global_value(output)?;
    f.store(output, texel, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The fixed binding numbers of the modules' symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s) if *s == SPLIT_SOURCE => Some(InterleaveSplitModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SPLIT_WIDE => Some(InterleaveSplitModule::WIDE_BINDING),
        Symbol::Binding(s) if *s == MERGE_WIDE => Some(InterleaveMergeModules::WIDE_BINDING),
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered with
/// the fixed bindings; `what` names it in an internal error. Refused, `Unsupported` naming the
/// first capability, where the device lacks what `m` needs.
fn finish(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(Feature::Format, c)))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The split module of `copy` at `samples` on `device`, built, verified, serialized and
/// numbered. Refused, `Unsupported` (feature `Format`) naming what the device lacks: an extended
/// copy format's storage without `shaderStorageImageExtendedFormats`.
pub fn interleave_split_module(
    device: &Device,
    copy: PlanesCopy,
    samples: u32,
) -> Result<InterleaveSplitModule, Error> {
    let (spirv, requirements) = finish(
        device,
        split_typed_module(copy, samples)?,
        "interleave split",
    )?;
    Ok(InterleaveSplitModule {
        spirv,
        requirements,
    })
}

/// The merge modules at `samples` on `device`, built and checked as [`interleave_split_module`]
/// is; refused likewise where the device lacks `sampleRateShading`.
pub fn interleave_merge_modules(
    device: &Device,
    samples: u32,
) -> Result<InterleaveMergeModules, Error> {
    let (vertex, vertex_needs) = finish(
        device,
        merge_vertex_typed_module()?,
        "interleave merge's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        merge_fragment_typed_module(samples)?,
        "interleave merge",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(InterleaveMergeModules {
        vertex,
        fragment,
        requirements,
    })
}

/// The merge's vertex stage as a typed module: one triangle over the viewport, `VertexIndex`
/// 0, 1, 2 at clip positions (-1, -1), (3, -1), (-1, 3).
pub(super) fn merge_vertex_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let index = b.builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, i32_)?;
    let position = b.builtin(crate::spv::BuiltIn::Position, IoDir::Output, vec4)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let int = |f: &mut FunctionBuilder<'_>, v: i32| -> BResult<Value> {
        let c = f.module().constant(Constant::Scalar {
            ty: i32_,
            bits: ConstBits(u64::from(v as u32)),
        })?;
        f.module().const_value(c)
    };
    let float = |f: &mut FunctionBuilder<'_>, v: f32| -> BResult<Value> {
        let c = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(v.to_bits())),
        })?;
        f.module().const_value(c)
    };
    let index = f.module().global_value(index)?;
    let index = f.load(index, MemAccess::default())?;
    let one = int(&mut f, 1)?;
    let two = int(&mut f, 2)?;
    // x = (i << 1) & 2, y = i & 2: (0, 0), (2, 0), (0, 2); the position is twice that, less 1.
    let shifted = f.binary(BinOp::ShiftLeftLogical, index, one)?;
    let x = f.binary(BinOp::BitwiseAnd, shifted, two)?;
    let y = f.binary(BinOp::BitwiseAnd, index, two)?;
    let x = f.convert(ConvOp::ConvertSToF, x, f32_)?;
    let y = f.convert(ConvOp::ConvertSToF, y, f32_)?;
    let twice = float(&mut f, 2.0)?;
    let less = float(&mut f, 1.0)?;
    let x = f.binary(BinOp::FMul, x, twice)?;
    let x = f.binary(BinOp::FSub, x, less)?;
    let y = f.binary(BinOp::FMul, y, twice)?;
    let y = f.binary(BinOp::FSub, y, less)?;
    let zero = float(&mut f, 0.0)?;
    let at = f.composite_construct(vec4, &[x, y, zero, less])?;
    let position = f.module().global_value(position)?;
    f.store(position, at, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Vertex, id, Vec::new())?;
    b.finalize()
}

#[cfg(test)]
mod tests;
