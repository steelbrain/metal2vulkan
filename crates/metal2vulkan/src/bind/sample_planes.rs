//! The copies between a multisampled color attachment and its sample planes.
//!
//! On a device without multisampled storage images, the storage form of framebuffer fetch reads
//! and writes a multisampled pass's fetched attachment as sample planes
//! (`StorageAttachmentView::SamplePlanes`): a single-sample 2D-array image of one layer per sample,
//! addressed `(x, y, SampleId)`. The consumer keeps the planes and moves the samples between
//! them and the attachment around the fetching draws. These crate-owned modules are the two
//! moves, each through an unsigned-integer view of the attachment format's texel size
//! ([`PlanesCopy`]), so every texel's bits cross unchanged whatever the format; `A8_UNORM`, whose
//! class no unsigned format shares, moves through its own format's `float`s
//! ([`PlanesCopy::Alpha8`]), which keep each 8-bit code:
//!
//! - **The split** ([`planes_split_module`], one per [`PlanesCopy`]): one `GLCompute` entry `main`
//!   with local size ([`PlanesSplitModule::LOCAL_SIZE`], [`PlanesSplitModule::LOCAL_SIZE`], 1)
//!   and two images in set 0:
//!
//!   | Binding | Image | View |
//!   |---|---|---|
//!   | [`PlanesSplitModule::SOURCE_BINDING`] | the multisampled attachment | a sampled 2D view of its level and layer, in [`PlanesCopy::format`] |
//!   | [`PlanesSplitModule::PLANES_BINDING`] | the planes | a storage 2D-array view of every layer, in [`PlanesCopy::format`] (declared without a format for [`PlanesCopy::Alpha8`]) |
//!
//!   Invocation `(x, y, s)` inside the planes' size fetches sample `s` of pixel `(x, y)` and
//!   writes it to texel `(x, y)` of layer `s`. Dispatch `ceil(width / LOCAL_SIZE)`,
//!   `ceil(height / LOCAL_SIZE)`, `samples` workgroups over planes of the render area's size.
//! - **The merge** ([`planes_merge_modules`]): a `Vertex` entry `main` that draws one triangle over
//!   the whole viewport from `VertexIndex` 0 to 2 (no vertex input), and a `Fragment` entry `main`
//!   with one sampled 2D-array image, the planes in [`PlanesCopy::format`], at
//!   [`PlanesMergeModules::PLANES_BINDING`] of set 0. Each invocation fetches texel
//!   `(FragCoord.xy, SampleId)` and writes it, an unsigned `uvec4` (a `vec4` for
//!   [`PlanesCopy::Alpha8`]), to the color output at `Location` 0. Run at sample rate (`sampleShadingEnable`, `minSampleShading` 1) into a
//!   render pass whose one color attachment is the multisampled attachment's view in
//!   [`PlanesCopy::format`], over the render area, without blending: each invocation writes its
//!   own sample. The fragment module reads `SampleId`, so it needs `sampleRateShading`.
//!
//! **Synchronization** (consumer): the attachment's writes before the split's sampled reads, and
//! the planes' earlier accesses before its storage writes; the planes' storage writes before
//! the fetching draws; their writes before the merge's sampled reads, and the attachment's
//! earlier accesses before the merge's color writes.

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
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageOperands, ImageQuery,
    ImageType, ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule,
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
    /// One byte of `A8_UNORM`'s own compatibility class ("8-bit alpha"), which no view of another
    /// format reads: moved through `A8_UNORM` views as the `float` each reads, `code / 255` in
    /// alpha, which its store keeps as the code. Not a bits copy ([`PlanesCopy::for_format`]):
    /// the sample planes' copies alone take it ([`PlanesCopy::for_planes`]).
    Alpha8,
}

impl PlanesCopy {
    /// Every copy format, by texel size, then [`PlanesCopy::Alpha8`].
    pub const ALL: [PlanesCopy; 6] = [
        PlanesCopy::R8,
        PlanesCopy::R16,
        PlanesCopy::R32,
        PlanesCopy::Rg32,
        PlanesCopy::Rgba32,
        PlanesCopy::Alpha8,
    ];

    /// The copy format of `format` through which a copy moves its texels' bits: a color format
    /// of one texel per block, of 1, 2, 4, 8 or 16 bytes, which shares the compatibility class of
    /// that unsigned format, so a view of it moves the texel's bits. `None` for a depth, stencil,
    /// compressed, multi-planar or 3-component 96-bit format, and for one of a class of its own
    /// (`A8_UNORM`'s "8-bit alpha", the 4-pack 10- and 12-bit formats', the 4:2:2 formats'),
    /// which no view of another format reads (VUID-VkImageCreateInfo-pNext-06722).
    pub fn for_format(format: Format) -> Option<PlanesCopy> {
        let block = format.block()?;
        if block.color.is_none() || block.depth.is_some() || block.stencil.is_some() {
            return None;
        }
        if block.texels != 1 {
            return None;
        }
        let copy = match block.size {
            1 => PlanesCopy::R8,
            2 => PlanesCopy::R16,
            4 => PlanesCopy::R32,
            8 => PlanesCopy::Rg32,
            16 => PlanesCopy::Rgba32,
            _ => return None,
        };
        copy.format()
            .block()
            .is_some_and(|own| own.class == block.class)
            .then_some(copy)
    }

    /// The Vulkan format of the copy's views: an unsigned-integer one, or `A8_UNORM` for
    /// [`PlanesCopy::Alpha8`].
    pub fn format(self) -> Format {
        match self {
            PlanesCopy::R8 => Format::R8_UINT,
            PlanesCopy::R16 => Format::R16_UINT,
            PlanesCopy::R32 => Format::R32_UINT,
            PlanesCopy::Rg32 => Format::R32G32_UINT,
            PlanesCopy::Rgba32 => Format::R32G32B32A32_UINT,
            PlanesCopy::Alpha8 => Format::A8_UNORM,
        }
    }

    /// Whether the copy moves texels as their bits, through an unsigned-integer format; else as
    /// the `float`s of their own format ([`PlanesCopy::Alpha8`]).
    pub fn moves_bits(self) -> bool {
        self != PlanesCopy::Alpha8
    }

    /// The SPIR-V image format of a storage view in [`PlanesCopy::format`]: `Unknown` for
    /// [`PlanesCopy::Alpha8`], whose format no SPIR-V format names.
    pub(crate) fn spirv_format(self) -> crate::spv::ImageFormat {
        match self {
            PlanesCopy::R8 => crate::spv::ImageFormat::R8ui,
            PlanesCopy::R16 => crate::spv::ImageFormat::R16ui,
            PlanesCopy::R32 => crate::spv::ImageFormat::R32ui,
            PlanesCopy::Rg32 => crate::spv::ImageFormat::Rg32ui,
            PlanesCopy::Rgba32 => crate::spv::ImageFormat::Rgba32ui,
            PlanesCopy::Alpha8 => crate::spv::ImageFormat::Unknown,
        }
    }
}

/// The crate-owned split module of one copy format on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanesSplitModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: a storage image of an extended format
    /// (`shaderStorageImageExtendedFormats`) for [`PlanesCopy::R8`], [`PlanesCopy::R16`] and
    /// [`PlanesCopy::Rg32`], and a storage write without a format (the SPIR-V capability
    /// `StorageImageWriteWithoutFormat`, which the device resolves, `Requirements`) for
    /// [`PlanesCopy::Alpha8`].
    pub requirements: Requirements,
}

impl PlanesSplitModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The multisampled attachment, a sampled image.
    pub const SOURCE_BINDING: u32 = 0;
    /// The planes, a storage image.
    pub const PLANES_BINDING: u32 = 1;
    /// Invocations of one workgroup along x, and along y.
    pub const LOCAL_SIZE: u32 = 8;
}

/// The crate-owned merge modules of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanesMergeModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the planes at
    /// [`PlanesMergeModules::PLANES_BINDING`] of set 0, the output at `Location` 0: a `uvec4`, or
    /// a `vec4` for [`PlanesCopy::Alpha8`].
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device: `sampleRateShading`.
    pub requirements: Requirements,
}

impl PlanesMergeModules {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The planes, a sampled image.
    pub const PLANES_BINDING: u32 = 0;
}

/// The binding symbol of the split's multisampled attachment in its typed module
/// ([`PlanesSplitModule::SOURCE_BINDING`]).
const SPLIT_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the split's planes in its typed module
/// ([`PlanesSplitModule::PLANES_BINDING`]).
const SPLIT_PLANES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};
/// The binding symbol of the merge's planes in its typed fragment module
/// ([`PlanesMergeModules::PLANES_BINDING`]).
const MERGE_PLANES: ResourceSlot = ResourceSlot {
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

/// The scalar type of `copy`'s texels: unsigned 32-bit, or 32-bit float for a copy that does not
/// move bits ([`PlanesCopy::moves_bits`]).
fn texel_scalar(b: &mut ModuleBuilder, copy: PlanesCopy) -> BResult<TypeId> {
    match copy.moves_bits() {
        true => u32_ty(b),
        false => b.ty(Type::Float { width: 32 }),
    }
}

/// A 2D image type of `copy`'s texels ([`texel_scalar`]), of `usage` and `format`, arrayed or
/// multisampled as asked.
fn copy_image(
    b: &mut ModuleBuilder,
    copy: PlanesCopy,
    usage: ImageUsage,
    format: crate::spv::ImageFormat,
    arrayed: bool,
    multisampled: bool,
) -> BResult<TypeId> {
    let sampled_type = texel_scalar(b, copy)?;
    b.ty(Type::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed,
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

/// The split module of `copy` as a typed module, before serialization: the module
/// [`planes_split_module`] serializes. Its bindings are the symbols [`SPLIT_SOURCE`] and
/// [`SPLIT_PLANES`].
pub fn split_typed_module(copy: PlanesCopy) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let unknown = crate::spv::ImageFormat::Unknown;
    let source_ty = copy_image(&mut b, copy, ImageUsage::Sampled, unknown, false, true)?;
    let planes_ty = copy_image(
        &mut b,
        copy,
        ImageUsage::Storage,
        copy.spirv_format(),
        true,
        false,
    )?;
    let source = b.resource(SPLIT_SOURCE, StorageClass::UniformConstant, source_ty)?;
    let planes = b.resource(SPLIT_PLANES, StorageClass::UniformConstant, planes_ty)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let planes = f.module().global_value(planes)?;
    let planes = f.load(planes, MemAccess::default())?;
    let size = f.query(ImageQuery::Size, planes)?;
    let mut inside = None;
    for axis in 0..3 {
        let at = f.composite_extract(invocation, &[axis])?;
        let end = f.composite_extract(size, &[axis])?;
        let below = f.binary(BinOp::ULessThan, at, end)?;
        inside = Some(match inside {
            None => below,
            Some(all) => f.binary(BinOp::LogicalAnd, all, below)?,
        });
    }
    let inside = inside.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let source = f.module().global_value(source)?;
    // Invocation `(x, y, s)` inside the planes copies sample `s` of pixel `(x, y)` into layer `s`.
    let copy_sample: ArmFn<'_, ()> = Box::new(move |f, _| {
        let source = f.load(source, MemAccess::default())?;
        let x = f.composite_extract(invocation, &[0])?;
        let y = f.composite_extract(invocation, &[1])?;
        let s = f.composite_extract(invocation, &[2])?;
        let pixel = f.composite_construct(uvec2, &[x, y])?;
        let ops = ImageOperands {
            sample: Some(s),
            ..ImageOperands::default()
        };
        let texel = f.fetch(source, pixel, &ops)?;
        f.write(
            planes,
            invocation,
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
            x: PlanesSplitModule::LOCAL_SIZE,
            y: PlanesSplitModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The merge's vertex stage as a typed module: one triangle over the viewport, `VertexIndex`
/// 0, 1, 2 at clip positions (-1, -1), (3, -1), (-1, 3).
pub fn merge_vertex_typed_module() -> BResult<TypedModule> {
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

/// The merge's fragment stage of `copy` as a typed module. Its binding is the symbol
/// [`MERGE_PLANES`].
pub fn merge_fragment_typed_module(copy: PlanesCopy) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let scalar = texel_scalar(&mut b, copy)?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let texel4 = b.ty(Type::Vector {
        elem: scalar,
        len: 4,
    })?;
    let unknown = crate::spv::ImageFormat::Unknown;
    let planes_ty = copy_image(&mut b, copy, ImageUsage::Sampled, unknown, true, false)?;
    let planes = b.resource(MERGE_PLANES, StorageClass::UniformConstant, planes_ty)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, i32_)?;
    let output = b.fixed_io(0, IoDir::Output, texel4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    let sample = f.module().global_value(sample)?;
    let sample = f.load(sample, MemAccess::default())?;
    let layer = f.convert(ConvOp::Bitcast, sample, u32_)?;
    // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let at = f.composite_construct(uvec3, &[x, y, layer])?;
    let planes = f.module().global_value(planes)?;
    let planes = f.load(planes, MemAccess::default())?;
    let level = uint(&mut f, 0)?;
    let ops = ImageOperands {
        lod: Some(level),
        ..ImageOperands::default()
    };
    let texel = f.fetch(planes, at, &ops)?;
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
        Symbol::Binding(s) if *s == SPLIT_SOURCE => Some(PlanesSplitModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SPLIT_PLANES => Some(PlanesSplitModule::PLANES_BINDING),
        Symbol::Binding(s) if *s == MERGE_PLANES => Some(PlanesMergeModules::PLANES_BINDING),
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed bindings; `what` names it in an internal error. Refused, `Unsupported` naming
/// the first capability, where the device lacks what `m` needs.
fn finish(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(Feature::FramebufferFetch, c)))?;
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

/// The split module of `copy` on `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`PlanesSplitModule`]. Refused, `Unsupported` (feature
/// `FramebufferFetch`) naming what the device lacks, where it lacks what the split needs: an
/// extended copy format's storage without `shaderStorageImageExtendedFormats`.
pub fn planes_split_module(device: &Device, copy: PlanesCopy) -> Result<PlanesSplitModule, Error> {
    let (spirv, requirements) = finish(device, split_typed_module(copy)?, "planes split")?;
    Ok(PlanesSplitModule {
        spirv,
        requirements,
    })
}

/// The merge modules of `copy` on `device`, built and checked as [`planes_split_module`] is;
/// refused likewise where the device lacks `sampleRateShading`. Every bits copy
/// ([`PlanesCopy::moves_bits`]) takes the same modules.
pub fn planes_merge_modules(
    device: &Device,
    copy: PlanesCopy,
) -> Result<PlanesMergeModules, Error> {
    let (vertex, vertex_needs) = finish(
        device,
        merge_vertex_typed_module()?,
        "planes merge's vertex stage",
    )?;
    let (fragment, fragment_needs) =
        finish(device, merge_fragment_typed_module(copy)?, "planes merge")?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(PlanesMergeModules {
        vertex,
        fragment,
        requirements,
    })
}

#[cfg(test)]
mod tests;
