//! The copies of a multisampled depth or stencil aspect without colour copies.
//!
//! Vulkan copies a multisampled aspect to or from another image only through maintenance8's
//! depth-stencil/colour `vkCmdCopyImage`. These crate-owned modules move
//! each sample another way. The source is read one sample per invocation, through a sampled view
//! of its copied aspect ([`CopySource::Aspect`]: a multisampled 2D image, unsigned for stencil,
//! float for depth) or as the shader-side stencil's shadow ([`CopySource::Shadow`]: a
//! multisampled `R8_UINT` storage image). The destination is written by a draw into it as the
//! depth/stencil attachment, at sample rate ([`depth_stencil_copy_modules`]), or by a kernel into
//! a storage image ([`stencil_split_module`]):
//!
//! - **A draw** ([`DepthStencilCopyModules`]): a `Vertex` entry `main` that draws one triangle
//!   over the whole viewport from `VertexIndex` 0 to 2 (no vertex input), and a `Fragment` entry
//!   `main` that reads the source's texel `(FragCoord.xy, SampleId)`, at
//!   [`DepthStencilCopyModules::SOURCE_BINDING`] of set [`DepthStencilCopyModules::SET`], and does
//!   what its [`DepthStencilCopy`] says with the texel's first component:
//!   - [`DepthStencilCopy::StencilPlane`]: kills the invocation where the plane's bit is clear.
//!     Eight draws, one per [`StencilBit`], over a stencil cleared to 0, each under write mask
//!     [`StencilBit::mask`], reference 0xFF, compare `ALWAYS` and pass op `REPLACE`, write every
//!     sample's stencil bit by bit: exact without stencil export.
//!   - [`DepthStencilCopy::StencilExport`]: writes it as `FragStencilRefEXT`; one draw under
//!     compare `ALWAYS`, pass op `REPLACE` and write mask 0xFF, where the device exports stencil.
//!   - [`DepthStencilCopy::Depth`]: writes it as `FragDepth`; one draw under depth compare
//!     `ALWAYS` with depth writes on. Exact only where the device stores the float `FragDepth`
//!     holds unchanged, which a consumer establishes per device before claiming the copy exact.
//!   - [`DepthStencilCopy::DepthBits`] and [`DepthStencilCopy::StencilBits`]: write its bits (a
//!     depth's `f32` bit-cast, a stencil's value) as the first component of a `uvec4` colour
//!     output at `Location` 0, into a multisampled unsigned colour attachment (`R32_UINT`,
//!     `R8_UINT`) with no depth/stencil attachment: the other way, a depth or stencil aspect's
//!     samples into colour.
//!   - [`DepthStencilCopy::Depth16Code`] and [`DepthStencilCopy::Depth16Bits`]: a 16-bit unsigned
//!     normalized depth, carried as its **code** rather than as a float, in an unsigned colour
//!     image of the code's bits (`R16_UINT`). `Depth16Code` reads a code `c` there and writes
//!     `c / 65535` as `FragDepth`; `Depth16Bits` reads a sample of a 16-bit depth aspect and
//!     writes `round(depth * 65535)` as [`DepthStencilCopy::DepthBits`] writes its bits.
//!
//!     The code is the carrier because it makes the pair exact by construction rather than by a
//!     measured tolerance. `c / 65535` is one correctly rounded division, so it is bit for bit
//!     the `float` nearest `c / 65535` — the very value a driver's fragment depth is measured to
//!     store back as `c` ([`DepthStencilCopy::Depth`]'s rule, measured by a probe that writes
//!     exactly these depths).
//!     The other way, `round(d * 65535)` recovers `c` from any `d` within 0.498 of a step of
//!     `c / 65535`: over all 65536 codes the largest `|d * 65535 - c|` is 1.96e-3, whether or not
//!     the multiply and add are contracted into one fused operation, against a half-step budget
//!     of 0.5. Carrying the depth as a float instead would add a float-to-unorm conversion that
//!     Vulkan leaves to the device and nobody has measured.
//!
//!   The fragment stage reads `SampleId`, so it runs once per sample (`sampleRateShading`): the
//!   consumer enables sample shading with `minSampleShading` 1, and each invocation writes its own
//!   sample. A pixel's `FragCoord` is its center, `(x + 0.5, y + 0.5)`, truncated to `(x, y)`.
//! - **The depth stand-in's copy back** ([`DepthStencilCopy::DepthFromStandIn`]): the draw's
//!   fragment stage reads sample `SampleId` of pixel `FragCoord.xy` of a depth stand-in, an
//!   `R32_SFLOAT` storage image ([`DepthStandInForm`]: multisampled, or the sample planes of a 2D
//!   array whose layer `s` holds sample `s`), and writes it as `FragDepth`, as
//!   [`DepthStencilCopy::Depth`] does from an aspect.
//! - **The depth split** ([`DepthSplitModule`]): as the stencil split below, from the float depth
//!   aspect into a depth stand-in of either form: the stand-in's copy in. Its invocation `(x, y,
//!   s)` inside the stand-in's extent and the source's samples writes sample `s` of texel `(x, y)`
//!   of a multisampled stand-in, or texel `(x, y, s)` of the planes.
//! - **The stencil split** ([`StencilSplitModule`]): a `GLCompute` entry `main` with local size
//!   ([`StencilSplitModule::LOCAL_SIZE`], [`StencilSplitModule::LOCAL_SIZE`], 1) whose invocation
//!   `(x, y, s)` inside the destination fetches sample `s` of pixel `(x, y)` of the unsigned source
//!   aspect at [`StencilSplitModule::SOURCE_BINDING`] and writes it to sample `s` of texel `(x, y)`
//!   of the multisampled `R8_UINT` storage image at [`StencilSplitModule::DESTINATION_BINDING`]:
//!   the shadow's copy in. Dispatch `ceil(width / LOCAL_SIZE)`, `ceil(height / LOCAL_SIZE)`,
//!   `samples` workgroups over a destination of the copied area.
//!
//! **Synchronization** (consumer): the source's writes before the copy's reads, and the
//! destination's earlier accesses before its attachment or storage writes; the copy's writes
//! before whatever reads the destination next.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, KillForm};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageOperands, ImageQuery,
    ImageType, ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule,
};

/// Where a copy reads its source (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CopySource {
    /// A sampled view of the source's copied aspect: a multisampled 2D image, unsigned for a
    /// stencil aspect and float for a depth aspect.
    Aspect,
    /// The shader-side stencil's shadow: a multisampled `R8_UINT` storage image.
    Shadow,
}

/// One bit of an 8-bit stencil: the plane a [`DepthStencilCopy::StencilPlane`] draw writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StencilBit(u8);

impl StencilBit {
    /// The eight bits, from the lowest.
    pub const ALL: [StencilBit; 8] = [
        StencilBit(0),
        StencilBit(1),
        StencilBit(2),
        StencilBit(3),
        StencilBit(4),
        StencilBit(5),
        StencilBit(6),
        StencilBit(7),
    ];

    /// The bit's mask, `1 << bit`: the plane draw's stencil write mask.
    pub fn mask(self) -> u32 {
        1 << self.0
    }
}

/// What a draw's fragment stage writes from the source's texel (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DepthStencilCopy {
    /// One bit plane of the stencil: the invocation is killed where its bit is clear.
    StencilPlane { source: CopySource, bit: StencilBit },
    /// The stencil, as `FragStencilRefEXT` (`VK_EXT_shader_stencil_export`).
    StencilExport { source: CopySource },
    /// The depth, as `FragDepth`, from a sampled view of a depth aspect.
    Depth,
    /// The depth's bits, its `f32` bit-cast, as the first component of the unsigned colour output
    /// at `Location` 0, from a sampled view of a depth aspect.
    DepthBits,
    /// The stencil's value as the first component of the unsigned colour output at `Location` 0,
    /// from a sampled view of a stencil aspect.
    StencilBits,
    /// The depth a 16-bit unsigned normalized code stands for, `code / 65535`, as `FragDepth`,
    /// from a sampled view of an unsigned colour image of such codes (`R16_UINT`): a 16-bit
    /// depth's copy in, exact by construction (the module docs).
    Depth16Code,
    /// A 16-bit unsigned normalized depth's code, `round(depth * 65535)`, as the first component
    /// of the unsigned colour output at `Location` 0, from a sampled view of a 16-bit depth
    /// aspect: a 16-bit depth's copy out, exact by construction (the module docs).
    Depth16Bits,
    /// The depth, as `FragDepth`, from a depth stand-in of `form`.
    DepthFromStandIn { form: DepthStandInForm },
}

/// The form of a depth stand-in: an `R32_SFLOAT` storage image of the pass's extent that holds its
/// depth attachment's samples while a fragment function tests depth itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DepthStandInForm {
    /// A multisampled 2D image: sample `s` of texel `(x, y)` holds sample `s` of pixel `(x, y)`
    /// (`shaderStorageImageMultisample`).
    Multisampled,
    /// Sample planes: a single-sample 2D-array image whose texel `(x, y)` of layer `s` holds sample
    /// `s` of pixel `(x, y)`.
    SamplePlanes,
}

/// The crate-owned depth split of one device and stand-in form, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthSplitModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: multisampled storage images for a multisampled
    /// stand-in.
    pub requirements: Requirements,
}

impl DepthSplitModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source, a sampled image of the depth aspect.
    pub const SOURCE_BINDING: u32 = 0;
    /// The destination, the stand-in's `R32_SFLOAT` storage image.
    pub const DESTINATION_BINDING: u32 = 1;
    /// Invocations of one workgroup along x, and along y.
    pub const LOCAL_SIZE: u32 = 8;
}

/// The `R32_SFLOAT` storage image type of a depth stand-in of `form` (`f32_` the float type).
fn stand_in_image(b: &mut ModuleBuilder, f32_: TypeId, form: DepthStandInForm) -> BResult<TypeId> {
    let format = crate::spv::ImageFormat::R32f;
    match form {
        DepthStandInForm::Multisampled => multisampled_image(b, f32_, ImageUsage::Storage, format),
        DepthStandInForm::SamplePlanes => b.ty(Type::Image(ImageType {
            sampled_type: f32_,
            dim: Dim::D2,
            depth: false,
            arrayed: true,
            multisampled: false,
            usage: ImageUsage::Storage,
            format,
        })),
    }
}

/// The crate-owned modules of one copy draw on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthStencilCopyModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the source at
    /// [`DepthStencilCopyModules::SOURCE_BINDING`] of set [`DepthStencilCopyModules::SET`].
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device: `sampleRateShading`; for the shadow,
    /// multisampled storage images of an extended format; for the export, the extension.
    pub requirements: Requirements,
}

impl DepthStencilCopyModules {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The source: a sampled image ([`CopySource::Aspect`]) or a storage image
    /// ([`CopySource::Shadow`]).
    pub const SOURCE_BINDING: u32 = 0;
}

/// The crate-owned stencil split of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StencilSplitModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: multisampled storage images
    /// (`shaderStorageImageMultisample`) of an extended format
    /// (`shaderStorageImageExtendedFormats`).
    pub requirements: Requirements,
}

impl StencilSplitModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source, a sampled image of the stencil aspect.
    pub const SOURCE_BINDING: u32 = 0;
    /// The destination, a multisampled `R8_UINT` storage image.
    pub const DESTINATION_BINDING: u32 = 1;
    /// Invocations of one workgroup along x, and along y.
    pub const LOCAL_SIZE: u32 = 8;
}

/// The binding symbol of a draw's source ([`DepthStencilCopyModules::SOURCE_BINDING`]).
const DRAW_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the split's source ([`StencilSplitModule::SOURCE_BINDING`]).
const SPLIT_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the split's destination ([`StencilSplitModule::DESTINATION_BINDING`]).
const SPLIT_DESTINATION: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};

/// The unsigned 32-bit scalar type.
pub(super) fn u32_ty(b: &mut ModuleBuilder) -> BResult<TypeId> {
    b.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// A multisampled 2D image of `sampled_type`, `usage` and `format`.
pub(super) fn multisampled_image(
    b: &mut ModuleBuilder,
    sampled_type: TypeId,
    usage: ImageUsage,
    format: crate::spv::ImageFormat,
) -> BResult<TypeId> {
    b.ty(Type::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: true,
        usage,
        format,
    }))
}

/// The unsigned 32-bit constant `v` as a value.
pub(super) fn uint(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The 32-bit float constant `v` as a value, of the float type `f32_`.
fn float(f: &mut FunctionBuilder<'_>, f32_: TypeId, v: f32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: f32_,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    f.module().const_value(c)
}

/// The largest 16-bit unsigned normalized code, the scale between a code and its depth.
const CODE_SCALE: f32 = 65535.0;

/// The fragment stage of `copy` as a typed module (the module docs). Its binding is the symbol
/// [`DRAW_SOURCE`].
pub fn fragment_typed_module(copy: DepthStencilCopy) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let source = match copy {
        DepthStencilCopy::StencilPlane { source, .. }
        | DepthStencilCopy::StencilExport { source } => source,
        DepthStencilCopy::Depth
        | DepthStencilCopy::DepthBits
        | DepthStencilCopy::StencilBits
        | DepthStencilCopy::Depth16Code
        | DepthStencilCopy::Depth16Bits
        | DepthStencilCopy::DepthFromStandIn { .. } => CopySource::Aspect,
    };
    let stand_in = match copy {
        DepthStencilCopy::DepthFromStandIn { form } => Some(form),
        _ => None,
    };
    let source_ty = match (copy, source) {
        (DepthStencilCopy::DepthFromStandIn { form }, _) => stand_in_image(&mut b, f32_, form)?,
        (
            DepthStencilCopy::Depth | DepthStencilCopy::DepthBits | DepthStencilCopy::Depth16Bits,
            _,
        ) => multisampled_image(
            &mut b,
            f32_,
            ImageUsage::Sampled,
            crate::spv::ImageFormat::Unknown,
        )?,
        (_, CopySource::Aspect) => multisampled_image(
            &mut b,
            u32_,
            ImageUsage::Sampled,
            crate::spv::ImageFormat::Unknown,
        )?,
        (_, CopySource::Shadow) => multisampled_image(
            &mut b,
            u32_,
            ImageUsage::Storage,
            crate::spv::ImageFormat::R8ui,
        )?,
    };
    let image = b.resource(DRAW_SOURCE, StorageClass::UniformConstant, source_ty)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, i32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    let sample = f.module().global_value(sample)?;
    let sample = f.load(sample, MemAccess::default())?;
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let pixel = f.composite_construct(uvec2, &[x, y])?;
    let image = f.module().global_value(image)?;
    let image = f.load(image, MemAccess::default())?;
    let ops = ImageOperands {
        sample: Some(sample),
        ..ImageOperands::default()
    };
    let texel = match (stand_in, source) {
        (Some(DepthStandInForm::Multisampled), _) => {
            f.read(image, pixel, &ops, TexelMemory::default())?
        }
        (Some(DepthStandInForm::SamplePlanes), _) => {
            let uvec3 = f.module().ty(Type::Vector { elem: u32_, len: 3 })?;
            let layer = f.convert(ConvOp::Bitcast, sample, u32_)?;
            let at = f.composite_construct(uvec3, &[x, y, layer])?;
            let plain = ImageOperands::default();
            f.read(image, at, &plain, TexelMemory::default())?
        }
        (None, CopySource::Aspect) => f.fetch(image, pixel, &ops)?,
        (None, CopySource::Shadow) => f.read(image, pixel, &ops, TexelMemory::default())?,
    };
    let value = f.composite_extract(texel, &[0])?;
    match copy {
        DepthStencilCopy::StencilPlane { bit, .. } => {
            let mask = uint(&mut f, bit.mask())?;
            let set = f.binary(BinOp::BitwiseAnd, value, mask)?;
            let zero = uint(&mut f, 0)?;
            let clear = f.binary(BinOp::IEqual, set, zero)?;
            // An invocation whose sample's bit is clear writes nothing.
            let kill: ArmFn<'_, ()> = Box::new(|f, _| {
                f.kill(KillForm::Kill)?;
                Ok(ArmEnd::Exits)
            });
            f.if_(&mut (), clear, kill, None)?;
        }
        DepthStencilCopy::StencilExport { .. } => {
            let reference = f.convert(ConvOp::Bitcast, value, i32_)?;
            let output =
                f.module()
                    .builtin(crate::spv::BuiltIn::FragStencilRefEXT, IoDir::Output, i32_)?;
            let output = f.module().global_value(output)?;
            f.store(output, reference, MemAccess::default())?;
        }
        DepthStencilCopy::Depth | DepthStencilCopy::DepthFromStandIn { .. } => {
            let output = f
                .module()
                .builtin(crate::spv::BuiltIn::FragDepth, IoDir::Output, f32_)?;
            let output = f.module().global_value(output)?;
            f.store(output, value, MemAccess::default())?;
        }
        DepthStencilCopy::Depth16Code => {
            // The code's depth: one correctly rounded division, so the `float` nearest
            // `code / 65535` (the module docs).
            let code = f.convert(ConvOp::ConvertUToF, value, f32_)?;
            let scale = float(&mut f, f32_, CODE_SCALE)?;
            let depth = f.binary(BinOp::FDiv, code, scale)?;
            let output = f
                .module()
                .builtin(crate::spv::BuiltIn::FragDepth, IoDir::Output, f32_)?;
            let output = f.module().global_value(output)?;
            f.store(output, depth, MemAccess::default())?;
        }
        DepthStencilCopy::DepthBits
        | DepthStencilCopy::StencilBits
        | DepthStencilCopy::Depth16Bits => {
            // The colour output, at `Location` 0.
            let uvec4 = f.module().ty(Type::Vector { elem: u32_, len: 4 })?;
            let output = f.module().fixed_io(0, IoDir::Output, uvec4, false)?;
            let bits = match copy {
                DepthStencilCopy::DepthBits => f.convert(ConvOp::Bitcast, value, u32_)?,
                // `round(depth * 65535)`: the conversion to unsigned truncates, so the half is
                // added first. Exact for every code (the module docs).
                DepthStencilCopy::Depth16Bits => {
                    let scale = float(&mut f, f32_, CODE_SCALE)?;
                    let half = float(&mut f, f32_, 0.5)?;
                    let scaled = f.binary(BinOp::FMul, value, scale)?;
                    let rounded = f.binary(BinOp::FAdd, scaled, half)?;
                    f.convert(ConvOp::ConvertFToU, rounded, u32_)?
                }
                _ => value,
            };
            let zero = uint(&mut f, 0)?;
            let texel = f.composite_construct(uvec4, &[bits, zero, zero, zero])?;
            let output = f.module().global_value(output)?;
            f.store(output, texel, MemAccess::default())?;
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    let mut modes = vec![ExecMode::OriginUpperLeft];
    match copy {
        DepthStencilCopy::StencilPlane { .. } => {}
        DepthStencilCopy::StencilExport { .. } => modes.push(ExecMode::StencilRefReplacing),
        DepthStencilCopy::Depth
        | DepthStencilCopy::Depth16Code
        | DepthStencilCopy::DepthFromStandIn { .. } => modes.push(ExecMode::DepthReplacing),
        DepthStencilCopy::DepthBits
        | DepthStencilCopy::StencilBits
        | DepthStencilCopy::Depth16Bits => {}
    }
    b.entry(ExecModel::Fragment, id, modes)?;
    b.finalize()
}

/// The stencil split as a typed module, before serialization: the module
/// [`stencil_split_module`] serializes. Its bindings are the symbols [`SPLIT_SOURCE`] and
/// [`SPLIT_DESTINATION`].
pub fn split_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let source_ty = multisampled_image(
        &mut b,
        u32_,
        ImageUsage::Sampled,
        crate::spv::ImageFormat::Unknown,
    )?;
    let destination_ty = multisampled_image(
        &mut b,
        u32_,
        ImageUsage::Storage,
        crate::spv::ImageFormat::R8ui,
    )?;
    let source = b.resource(SPLIT_SOURCE, StorageClass::UniformConstant, source_ty)?;
    let destination = b.resource(
        SPLIT_DESTINATION,
        StorageClass::UniformConstant,
        destination_ty,
    )?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let destination = f.module().global_value(destination)?;
    let destination = f.load(destination, MemAccess::default())?;
    let size = f.query(ImageQuery::Size, destination)?;
    let samples = f.query(ImageQuery::Samples, destination)?;
    let mut inside = None;
    for axis in 0..3 {
        let at = f.composite_extract(invocation, &[axis])?;
        let end = if axis < 2 {
            f.composite_extract(size, &[axis])?
        } else {
            samples
        };
        let below = f.binary(BinOp::ULessThan, at, end)?;
        inside = Some(match inside {
            None => below,
            Some(all) => f.binary(BinOp::LogicalAnd, all, below)?,
        });
    }
    let inside = inside.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let source = f.module().global_value(source)?;
    // Invocation `(x, y, s)` inside the destination copies sample `s` of pixel `(x, y)`.
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
        f.write(destination, pixel, texel, &ops, TexelMemory::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy_sample, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: StencilSplitModule::LOCAL_SIZE,
            y: StencilSplitModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The depth split into a stand-in of `form` as a typed module, before serialization: the module
/// [`depth_split_module`] serializes. Its bindings are the symbols [`SPLIT_SOURCE`] and
/// [`SPLIT_DESTINATION`], numbered as the stencil split's.
pub fn depth_split_typed_module(form: DepthStandInForm) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let source_ty = multisampled_image(
        &mut b,
        f32_,
        ImageUsage::Sampled,
        crate::spv::ImageFormat::Unknown,
    )?;
    let destination_ty = stand_in_image(&mut b, f32_, form)?;
    let source = b.resource(SPLIT_SOURCE, StorageClass::UniformConstant, source_ty)?;
    let destination = b.resource(
        SPLIT_DESTINATION,
        StorageClass::UniformConstant,
        destination_ty,
    )?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let destination = f.module().global_value(destination)?;
    let destination = f.load(destination, MemAccess::default())?;
    let source = f.module().global_value(source)?;
    let source = f.load(source, MemAccess::default())?;
    let size = f.query(ImageQuery::Size, destination)?;
    let samples = f.query(ImageQuery::Samples, source)?;
    let mut inside = None;
    for axis in 0..3 {
        let at = f.composite_extract(invocation, &[axis])?;
        let end = if axis < 2 {
            f.composite_extract(size, &[axis])?
        } else {
            samples
        };
        let below = f.binary(BinOp::ULessThan, at, end)?;
        inside = Some(match inside {
            None => below,
            Some(all) => f.binary(BinOp::LogicalAnd, all, below)?,
        });
    }
    let inside = inside.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    // Invocation `(x, y, s)` inside the stand-in copies sample `s` of pixel `(x, y)`.
    let copy_sample: ArmFn<'_, ()> = Box::new(move |f, _| {
        let x = f.composite_extract(invocation, &[0])?;
        let y = f.composite_extract(invocation, &[1])?;
        let s = f.composite_extract(invocation, &[2])?;
        let pixel = f.composite_construct(uvec2, &[x, y])?;
        let ops = ImageOperands {
            sample: Some(s),
            ..ImageOperands::default()
        };
        let texel = f.fetch(source, pixel, &ops)?;
        match form {
            DepthStandInForm::Multisampled => {
                f.write(destination, pixel, texel, &ops, TexelMemory::default())?;
            }
            DepthStandInForm::SamplePlanes => {
                let plain = ImageOperands::default();
                f.write(
                    destination,
                    invocation,
                    texel,
                    &plain,
                    TexelMemory::default(),
                )?;
            }
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy_sample, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DepthSplitModule::LOCAL_SIZE,
            y: DepthSplitModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the modules' symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s) if *s == DRAW_SOURCE => Some(DepthStencilCopyModules::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SPLIT_SOURCE => Some(StencilSplitModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SPLIT_DESTINATION => {
            Some(StencilSplitModule::DESTINATION_BINDING)
        }
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed bindings; `what` names it in an internal error.
/// Refused, `Unsupported` naming `feature` and the first capability, where the device lacks what
/// `m` needs.
pub(super) fn finish(
    device: &Device,
    m: TypedModule,
    feature: &Feature,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(feature.clone(), c)))?;
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

/// The modules of the copy draw `copy` on `device`: built with the typed builder (the vertex
/// stage is the sample planes merge's, `bind::sample_planes`), checked by the owned verifier
/// against the device's profile (requirements included), serialized, and numbered with the fixed
/// binding of [`DepthStencilCopyModules`]. Refused, `Unsupported` naming `feature` (what the
/// consumer copies for) and what the device lacks, where it lacks what the draw needs.
pub fn depth_stencil_copy_modules(
    device: &Device,
    copy: DepthStencilCopy,
    feature: Feature,
) -> Result<DepthStencilCopyModules, Error> {
    let (vertex, vertex_needs) = finish(
        device,
        super::sample_planes::merge_vertex_typed_module()?,
        &feature,
        "depth/stencil copy's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        fragment_typed_module(copy)?,
        &feature,
        "depth/stencil copy",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(DepthStencilCopyModules {
        vertex,
        fragment,
        requirements,
    })
}

/// The stencil split on `device`, built and checked as [`depth_stencil_copy_modules`] is;
/// refused likewise where the device lacks multisampled storage images of `R8_UINT`.
pub fn stencil_split_module(
    device: &Device,
    feature: Feature,
) -> Result<StencilSplitModule, Error> {
    let (spirv, requirements) = finish(device, split_typed_module()?, &feature, "stencil split")?;
    Ok(StencilSplitModule {
        spirv,
        requirements,
    })
}

/// The depth split into a stand-in of `form` on `device`, built and checked as
/// [`depth_stencil_copy_modules`] is; refused likewise where the device lacks what it needs
/// (multisampled storage images, for a multisampled stand-in).
pub fn depth_split_module(
    device: &Device,
    form: DepthStandInForm,
    feature: Feature,
) -> Result<DepthSplitModule, Error> {
    let (spirv, requirements) = finish(
        device,
        depth_split_typed_module(form)?,
        &feature,
        "depth split",
    )?;
    Ok(DepthSplitModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
