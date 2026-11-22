//! The pipeline constant blob.
//!
//! A program-scope constant that is not a closed, type-consistent `Private` variable (its
//! address merges, escapes, is compared, or it is punned or copied with `memcpy`) lives in the
//! **constant blob**: the initializers' little-endian bytes under the AIR datalayout, each at its
//! alignment (`memory::plan`).
//!
//! Where the blob lives is its [`BlobHome`]. Off the identity domains it is **bound**: the artifact
//! carries the bytes (`FunctionArtifact.blob`); the binder uploads them once per pipeline and binds
//! them as `WriteSource::ConstantBlob` at the stage's `ResourceSlot{stage, Synthetic(ConstantBlob),
//! 0}`, a read-only storage buffer (`StorageBuffer { u32[] }`, the word view of `memory::views`)
//! covering at least the blob rounded up to 16 bytes (the padding contract). Under the identity
//! domains, where a function binds no descriptor, the blob is the **module's**: a `Private` array
//! of `u32` words whose initializer is the bytes, zero-padded to 16 bytes as a bound blob is
//! ([`module_bytes`]), and the artifact carries none. A blob past [`MODULE_BYTES`] is refused
//! there, naming the constant data.
//!
//! A pointer into the blob is its `u32` byte offset from the blob's start: a constant global's
//! address is its blob offset, and GEPs, casts, and `select` are integer operations (the arena
//! pointer form). Loads are word-composed leaves (`memory::leaf`). A bound blob's loads take the
//! normative check against the bound size (`OpArrayLength · 4`), and the module's words are
//! checked against their length on every load: no robustness covers `Private` memory.

use std::sync::Arc;

use crate::memory::leaf::WordMem;
use crate::memory::views::{self, View};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{ConstBits, Constant, MemAccess, StorageClass, Type};

/// Where a function's blob lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobHome {
    /// A read-only storage buffer the binder uploads and binds at the stage's blob symbol.
    Bound,
    /// The module's own `Private` words (the identity domains).
    Module,
}

/// The blob a function reads: its bytes, and where they live. The artifact carries a bound blob's
/// bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobUse {
    pub bytes: Arc<[u8]>,
    pub home: BlobHome,
}

/// The most bytes a blob the module holds ([`BlobHome::Module`]) may have:
/// `alloca::MAX_UNSHARED`, the most the typed objects of one function may declare. A typed
/// constant up to that size is already a `Private` variable with its initializer, so the module's
/// blob gives the make no larger initializer than one of those does.
pub const MODULE_BYTES: u64 = crate::memory::alloca::MAX_UNSHARED;

/// The bytes the module's words of a blob of `len` bytes cover: `len` rounded up to 16, the
/// padding a bound blob's range covers.
pub fn module_bytes(len: u64) -> u64 {
    len.div_ceil(16).saturating_mul(16)
}

/// The binding symbol of `stage`'s constant blob.
pub fn symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
        index: 0,
    }
}

/// A declared blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobDecl {
    /// The storage word view at the stage's blob symbol.
    Bound(View),
    /// The module's `Private` words.
    Module(ModuleWords),
}

/// The module's words of a blob: a `Private` array of `words` `u32`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleWords {
    pub global: GlobalId,
    pub words: u32,
}

/// Declare `stage`'s blob `blob` where it lives.
pub fn declare(mb: &mut ModuleBuilder, stage: Stage, blob: &BlobUse) -> BResult<BlobDecl> {
    match blob.home {
        BlobHome::Bound => views::declare_at(mb, symbol(stage)).map(BlobDecl::Bound),
        BlobHome::Module => declare_module(mb, &blob.bytes).map(BlobDecl::Module),
    }
}

/// The module's words of `bytes`: a `Private` `u32` array initialized with the bytes'
/// little-endian words, zero-padded to [`module_bytes`].
fn declare_module(mb: &mut ModuleBuilder, bytes: &[u8]) -> BResult<ModuleWords> {
    let len = module_bytes(bytes.len() as u64) / 4;
    let words = u32::try_from(len).map_err(|_| BuildError::IdSpaceExhausted)?;
    let u32_ty = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let count = mb.constant(Constant::Scalar {
        ty: u32_ty,
        bits: ConstBits(u64::from(words)),
    })?;
    let ty = mb.ty(Type::Array {
        elem: u32_ty,
        len: count,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(len as usize);
    for at in 0..len as usize {
        let mut word = [0u8; 4];
        for (i, b) in word.iter_mut().enumerate() {
            *b = bytes.get(at * 4 + i).copied().unwrap_or(0);
        }
        parts.push(mb.constant(Constant::Scalar {
            ty: u32_ty,
            bits: ConstBits(u64::from(u32::from_le_bytes(word))),
        })?);
    }
    let init = mb.constant(Constant::Composite {
        ty,
        parts: parts.into(),
    })?;
    let global = mb.plain_global(StorageClass::Private, ty, Some(init))?;
    Ok(ModuleWords { global, words })
}

/// The module's words are constant: they are read, never written.
impl WordMem for ModuleWords {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: Value) -> BResult<Value> {
        let base = fb.module().global_value(self.global)?;
        let p = fb.access_chain(base, &[word])?;
        fb.load(p, MemAccess::default())
    }

    fn write(&mut self, _: &mut FunctionBuilder<'_>, _: Value, _: Value) -> BResult<()> {
        Err(BuildError::Dangling(Dangling::Global(self.global)))
    }

    fn write_bits(
        &mut self,
        _: &mut FunctionBuilder<'_>,
        _: Value,
        _: Value,
        _: Value,
    ) -> BResult<()> {
        Err(BuildError::Dangling(Dangling::Global(self.global)))
    }
}
