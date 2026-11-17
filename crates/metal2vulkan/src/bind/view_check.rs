//! The kernels of the runtime's view check ([`view_check_module`]): what reading a `constant`
//! window costs through each view a uniform view can take, measured on the device itself. Its
//! verdicts are the description's measured workarounds
//! ([`DeviceDescription::measured`](crate::device::DeviceDescription::measured)):
//! `DriverWorkaround::UniformBufferPerLaneReadsSerialize` and
//! `DriverWorkaround::ExternalUniformReadsSkipACache`.
//!
//! **The module** (one per [`CheckView`] and [`CheckReads`]). One `GLCompute` entry `main` with
//! local size ([`ViewCheckModule::LOCAL_SIZE`], 1, 1), [`ViewCheckModule::PUSH_BYTES`] of `COMPUTE`
//! push constants at offset 0 ([`ViewCheckModule::push`]), and two bindings in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`ViewCheckModule::WINDOW_BINDING`] | the window, [`ViewCheckModule::WINDOW_VECS`] `uvec4`s: a uniform buffer (`Uniform`, std140), a storage buffer (`StorageBuffer`, std430) or a uniform texel buffer (`Dim` `Buffer`, `Sampled` 1, `u32` texels, viewed as `R32G32B32A32_UINT`) |
//! | [`ViewCheckModule::OUT_BINDING`] | a storage buffer of 4-byte words, one an invocation |
//!
//! The push words, in order: `mask`, the window's vectors less one (a power of two less one, at
//! most [`ViewCheckModule::WINDOW_VECS`] less one), and `reads`, the steps of each invocation's
//! loop; two words of 0. The mask is a push word, not a literal: a driver that sees an index
//! bounded at compile time may compile the read otherwise (ANV, Mesa 26.2.6 on the Iris Xe of
//! RPL-P, read a uniform buffer's one word 4.8 times as long with a literal mask). Invocation `g`
//! (`GlobalInvocationId.x`) starts with `acc = g` and, for `i` from 0 below `reads`, reads:
//!
//! - [`CheckReads::Uniform`]: vector `(7 i) & mask`, one address for every invocation;
//! - [`CheckReads::PerLane`]: vector `(g + 7 i) & mask`, each invocation its own;
//! - [`CheckReads::Fields`]: vectors 0 and 1, the same two at constant offsets every step (a
//!   struct's fields);
//!
//! folding each vector `v` read as `acc = (acc ^ v.x) + ((v.y + v.z) ^ v.w)`, then `acc + i`, and
//! stores `acc` as word `g` of the out binding. Nothing else is read or written.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, Dim, ExecMode, ExecModel, ImageOperands, ImageType,
    ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

use super::index_rewrite::for_range;
use super::shadow_merge::{u32_value, words_buffer};

/// The view a [`view_check_module`] reads its window through.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckView {
    /// A uniform buffer (`UNIFORM_BUFFER`).
    Uniform,
    /// A storage buffer (`STORAGE_BUFFER`).
    Storage,
    /// A uniform texel buffer (`UNIFORM_TEXEL_BUFFER`) of `R32G32B32A32_UINT` texels.
    Texel,
}

impl CheckView {
    /// Every view.
    pub const ALL: [CheckView; 3] = [CheckView::Uniform, CheckView::Storage, CheckView::Texel];
}

/// Which vectors of the window a [`view_check_module`]'s invocations read (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckReads {
    /// One address for every invocation, a new one each step.
    Uniform,
    /// Each invocation its own address, a new one each step.
    PerLane,
    /// The same two vectors at constant offsets, every step.
    Fields,
}

impl CheckReads {
    /// Every pattern.
    pub const ALL: [CheckReads; 3] = [CheckReads::Uniform, CheckReads::PerLane, CheckReads::Fields];
}

/// A view check kernel of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewCheckModule {
    /// SPIR-V words, entry point `main`, bindings numbered as the module docs say in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ViewCheckModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The window: the view's binding.
    pub const WINDOW_BINDING: u32 = 0;
    /// The words the invocations store, a storage buffer.
    pub const OUT_BINDING: u32 = 1;
    /// The `uvec4`s the window's declaration holds: 4 KiB. A dispatch reads the first `mask + 1`.
    pub const WINDOW_VECS: u32 = 256;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a dispatch reading the window's first `vecs` vectors (a power of two, at
    /// most [`WINDOW_VECS`](Self::WINDOW_VECS)) `reads` times in each invocation.
    pub fn push(vecs: u32, reads: u32) -> [u8; 16] {
        let words = [vecs.saturating_sub(1), reads, 0, 0];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the window in the typed module.
const WINDOW: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the out words in the typed module.
const OUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the words are declared under: a crate-owned module's own, which no selection
/// places.
const SHAPE: PushField = PushField::DispatchRegion;

/// The window's declaration through `view`.
fn window(b: &mut ModuleBuilder, view: CheckView, u32_: TypeId) -> BResult<GlobalId> {
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let (class, rule, member) = match view {
        CheckView::Texel => {
            let image = b.ty(Type::Image(ImageType {
                sampled_type: u32_,
                dim: Dim::Buffer,
                depth: false,
                arrayed: false,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            return b.resource(WINDOW, StorageClass::UniformConstant, image);
        }
        CheckView::Uniform => {
            let len = b.constant(Constant::Scalar {
                ty: u32_,
                bits: ConstBits(u64::from(ViewCheckModule::WINDOW_VECS)),
            })?;
            let array = b.ty(Type::Array {
                elem: uvec4,
                len,
                stride: Some(16),
            })?;
            (StorageClass::Uniform, LayoutRule::Std140, array)
        }
        CheckView::Storage => {
            let array = b.ty(Type::RuntimeArray {
                elem: uvec4,
                stride: Some(16),
            })?;
            (StorageClass::StorageBuffer, LayoutRule::Std430, array)
        }
    };
    let layout = natural_layout(b.types(), rule, true, &[member])?;
    let block = b.ty(Type::Struct {
        members: [member].into(),
        layout: Some(layout),
    })?;
    b.resource(WINDOW, class, block)
}

/// Vector `index` of the window `global` through `view`: a `uvec4`.
fn read(
    f: &mut FunctionBuilder<'_>,
    view: CheckView,
    global: GlobalId,
    index: Value,
) -> BResult<Value> {
    let base = f.module().global_value(global)?;
    match view {
        CheckView::Texel => {
            let image = f.load(base, MemAccess::default())?;
            f.fetch(image, index, &ImageOperands::default())
        }
        CheckView::Uniform | CheckView::Storage => {
            let zero = u32_value(f, 0)?;
            let p = f.access_chain(base, &[zero, index])?;
            f.load(p, MemAccess::default())
        }
    }
}

/// `acc` with the vector `v` folded in (the module docs).
fn fold(f: &mut FunctionBuilder<'_>, acc: Value, v: Value) -> BResult<Value> {
    let x = f.composite_extract(v, &[0])?;
    let y = f.composite_extract(v, &[1])?;
    let z = f.composite_extract(v, &[2])?;
    let w = f.composite_extract(v, &[3])?;
    let a = f.binary(BinOp::BitwiseXor, acc, x)?;
    let b = f.binary(BinOp::IAdd, y, z)?;
    let b = f.binary(BinOp::BitwiseXor, b, w)?;
    f.binary(BinOp::IAdd, a, b)
}

/// The view check kernel of `view` and `reads` as a typed module, before serialization (the
/// module docs).
pub(crate) fn typed_module(view: CheckView, reads: CheckReads) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let window = window(&mut b, view, u32_)?;
    let out = words_buffer(&mut b, OUT, u32_)?;
    let global = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let seven = u32_value(&mut f, 7)?;
    let global = f.module().global_value(global)?;
    let global = f.load(global, MemAccess::default())?;
    let g = f.composite_extract(global, &[0])?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let mask = f.composite_extract(shape, &[0])?;
    let steps = f.composite_extract(shape, &[1])?;
    let acc = f.local_var(u32_, None)?;
    let acc = f.var_ptr(acc)?;
    f.store(acc, g, MemAccess::default())?;
    for_range(&mut f, u32_, (zero, steps, one), move |f, i| {
        let mut now = f.load(acc, MemAccess::default())?;
        let step = f.binary(BinOp::IMul, i, seven)?;
        let indices = match reads {
            CheckReads::Uniform => vec![f.binary(BinOp::BitwiseAnd, step, mask)?],
            CheckReads::PerLane => {
                let at = f.binary(BinOp::IAdd, g, step)?;
                vec![f.binary(BinOp::BitwiseAnd, at, mask)?]
            }
            CheckReads::Fields => vec![zero, one],
        };
        for index in indices {
            let v = read(f, view, window, index)?;
            now = fold(f, now, v)?;
        }
        let now = f.binary(BinOp::IAdd, now, i)?;
        f.store(acc, now, MemAccess::default())
    })?;
    let out = f.module().global_value(out)?;
    let done = f.load(acc, MemAccess::default())?;
    let p = f.access_chain(out, &[zero, g])?;
    f.store(p, done, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ViewCheckModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ViewCheckModule::SET),
        Symbol::Binding(s) if *s == WINDOW => Some(ViewCheckModule::WINDOW_BINDING),
        Symbol::Binding(s) if *s == OUT => Some(ViewCheckModule::OUT_BINDING),
        _ => None,
    }
}

/// The view check kernel of `device` reading its window through `view` as `reads` says: built
/// with the typed builder, checked by the owned verifier against the device's profile
/// (requirements included), serialized, and numbered with the fixed bindings of
/// [`ViewCheckModule`]. An internal error where the profile lacks what it needs.
pub fn view_check_module(
    device: &Device,
    view: CheckView,
    reads: CheckReads,
) -> Result<ViewCheckModule, Error> {
    let profile = device.profile();
    let m = typed_module(view, reads)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the view check's kernel needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ViewCheckModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, CheckReads, CheckView};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Every view and pattern is valid on every committed profile.
    #[test]
    fn the_view_check_kernels_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for view in CheckView::ALL {
                for reads in CheckReads::ALL {
                    assert_valid(typed_module(view, reads).expect("module"), &p);
                }
            }
        }
    }
}
