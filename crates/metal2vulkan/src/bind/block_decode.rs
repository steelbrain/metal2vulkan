//! The shape every crate-owned decode module shares: a carried texture's blocks decoded into the
//! texels of its decoded twin, for a consumer that keeps one, an ASTC
//! ([`astc_decode_module`](super::astc_decode_module)), PVRTC
//! ([`pvrtc_decode_module`](super::pvrtc_decode_module)), ETC2 or EAC
//! ([`etc_decode_module`](super::etc_decode_module)) texture's on a device without its format, or a
//! 4:2:2 texture's ([`chroma_decode_module`](super::chroma_decode_module)).
//!
//! **The module** (one per device and decode). One `GLCompute` entry `main` with local size
//! ([`BlockDecodeModule::LOCAL_SIZE`], [`BlockDecodeModule::LOCAL_SIZE`], 1), no push constants,
//! and in set 0 three storage buffers of 4-byte words:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`BlockDecodeModule::SHAPE_BINDING`] | the region's shape ([`BlockDecodeModule::shape`], or a decode's own) |
//! | [`BlockDecodeModule::BLOCKS_BINDING`] | the region's blocks, read: tight, as a copy of the region out of the image lays them (each layer's rows of blocks, its first row first) |
//! | [`BlockDecodeModule::TEXELS_BINDING`] | the region's texels, written: tight, as a copy into an image of the decode's texel format reads them |
//!
//! Invocation `(x, y, z)` inside the region's width, height and depth (layers, or a 3D level's
//! slices) writes texel `(x, y)` of layer `z` from its block; one outside it writes nothing. The
//! consumer dispatches `⌈w / 8⌉ x ⌈h / 8⌉ x d` workgroups and keeps each binding within
//! `maxStorageBufferRange`, splitting a larger region into bands of whole block rows.
//!
//! **Synchronization** (consumer): the writes of the shape and the blocks (`TRANSFER_WRITE`) →
//! `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its texel writes
//! (`SHADER_STORAGE_WRITE`) → `TRANSFER_READ` before the texels are copied into the image.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId};
use crate::spirv::ir::{ConstBits, Constant, StorageClass, Type, TypedModule};

/// A crate-owned decode module of one device and decode, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDecodeModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl BlockDecodeModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The region's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 0;
    /// The region's blocks: a storage buffer, read.
    pub const BLOCKS_BINDING: u32 = 1;
    /// The region's texels: a storage buffer, written.
    pub const TEXELS_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = 5;
    /// Invocations of one workgroup along x, and along y: one texel each.
    pub const LOCAL_SIZE: u32 = 8;

    /// The shape's bytes of a region `width` x `height` x `depth` (layers, or a 3D level's
    /// slices): its size, and its blocks across and down.
    pub fn shape([width, height, depth]: [u32; 3]) -> [u8; Self::SHAPE_WORDS as usize * 4] {
        let words = [width, height, depth, width.div_ceil(4), height.div_ceil(4)];
        let mut bytes = [0; Self::SHAPE_WORDS as usize * 4];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the shape in the typed module.
pub(super) const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the blocks in the typed module.
pub(super) const BLOCKS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the texels in the typed module.
pub(super) const TEXELS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// A `u32[n]` `Private` table initialized to `values`.
pub(super) fn table(b: &mut ModuleBuilder, u32_: TypeId, values: &[u32]) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(values.len() as u64),
    })?;
    let arr = b.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(values.len());
    for &v in values {
        parts.push(b.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(v)),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, arr, Some(init))
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(BlockDecodeModule::SET),
        Symbol::Binding(s) if *s == SHAPE => Some(BlockDecodeModule::SHAPE_BINDING),
        Symbol::Binding(s) if *s == BLOCKS => Some(BlockDecodeModule::BLOCKS_BINDING),
        Symbol::Binding(s) if *s == TEXELS => Some(BlockDecodeModule::TEXELS_BINDING),
        _ => None,
    }
}

/// The decode module of `device` of the typed module `m` (whose bindings are the symbols
/// `SHAPE`, `BLOCKS` and `TEXELS`, `what` its kind): checked by the owned verifier against the
/// device's profile (requirements included), serialized, and numbered with the fixed bindings of
/// [`BlockDecodeModule`].
pub(super) fn decode_module(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<BlockDecodeModule, Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(BlockDecodeModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}
