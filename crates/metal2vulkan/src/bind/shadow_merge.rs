//! The merge of a texture's writes into its buffer's bytes, and of a linear attachment's backing
//! into its surface's bytes.
//!
//! A texture over a buffer that the device cannot lay as Metal does lives in a shadow: an image
//! or a staging buffer, copied in from the buffer's bytes before a command and back after it.
//! Where one command writes through such a texture and also writes the same buffer's bytes
//! another way, a copy back would replace the other write's bytes with the shadow's, where Metal
//! keeps both writes when they touch different bytes. The consumer then merges instead: before
//! the command it copies the shadow's texels into a *kept* buffer, laid out as in the buffer from
//! the texture's first byte; after it, into a *changed* buffer, and the buffer's bytes from the
//! texture's first to its last into a *current* buffer; one dispatch of
//! [`shadow_merge_module`] takes, in each byte of the current buffer, the changed buffer's byte
//! where it differs from the kept one; and it copies the current buffer back into the buffer's
//! bytes. A byte written both ways is the texture's where its value changed, as Metal leaves
//! such a race unordered.
//!
//! A consumer may also bind the buffer's own bytes as the current buffer, over a region of rows:
//! a linear attachment rendered through an image of the consumer's, whose pass must not write
//! the pixels its draws left as they were. The dispatch writes no word whose kept and changed words
//! are equal, so another writer's bytes there are never rewritten.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`ShadowMergeModule::LOCAL_SIZE`], 1, 1), [`ShadowMergeModule::PUSH_BYTES`] of `COMPUTE`
//! push constants at offset 0 (the region's shape, [`ShadowMergeModule::shape`]), and three
//! storage buffers in set 0, each a whole number of 4-byte words:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`ShadowMergeModule::KEPT_BINDING`] | the region's words before the command, row after row |
//! | [`ShadowMergeModule::CHANGED_BINDING`] | the region's words after it, laid out alike |
//! | [`ShadowMergeModule::CURRENT_BINDING`] | the bytes the region lies in, merged in place |
//!
//! The shape's words, in order: `width`, the words of one row of the region (at least 1);
//! `count`, the region's words (`width` times its rows); `row_words`, the current buffer's rows
//! apart in words; and `first_word`, the word of the region's first in the current binding.
//! Invocation `l` of workgroup `g` takes the words `i = 4Ng + Nk + l` for `k` from 0 to 3
//! ([`ShadowMergeModule::WORDS`], `N` [`ShadowMergeModule::LOCAL_SIZE`]), one after the other; a
//! word `i` below `count` takes word `i` of the kept and changed buffers and,
//! where they differ, word `first_word + (i / width) · row_words + i % width` of the current
//! buffer: each of its bytes where the changed word's differs from the kept one's becomes the
//! changed word's, and the others keep the current word's. Where the kept and changed words are
//! equal it reads and writes nothing of the current buffer; a word past `count` does nothing.
//! A dispatch takes [`ShadowMergeModule::groups`] workgroups. Contiguous bytes are one row:
//! `width` and `count` the words, `row_words` any, `first_word` 0. A consumer whose bytes
//! exceed `maxStorageBufferRange` or `maxComputeWorkGroupCount[0]` workgroups binds them in
//! windows, one dispatch each: the merge of one byte reads no other byte.
//!
//! Four words an invocation: at one word an invocation the kernel's time went to its
//! invocations, not to memory. On an M3 Max under MoltenVK 1.4.1, one dispatch over a 1920x1080
//! surface's words, in its own Metal encoder (about 17 us of it), took 99 us with no word changed
//! and 107 us with every word changed at one word an invocation, and 46 us and 74 us at four.
//!
//! Each of the four a workgroup apart, not four in a row: workgroup `g` takes the same `4N`
//! words either way, and every word's merge reads and writes that word alone, so which
//! invocation takes which word is free; but `4Ng + Nk + l` puts the `k`th store of consecutive
//! `l` at consecutive words, where `4(Ng + l) + k` leaves three words between them. On an
//! RTX 5080 Laptop GPU (NVIDIA 615.71.09), a store of 8 MiB of imported host memory that leaves
//! words between the lanes of a warp takes hundreds of times the time of the same bytes stored
//! without gaps: 0.32 ms at one word an invocation in order and at four a workgroup apart, 590 ms
//! at four in a row, and 383, 140 and 51 ms at one word per 8, 16 and 32 bytes; the merge's own
//! shape (compare, store where changed) 0.85 ms a workgroup apart against 609 ms in a row, with
//! loads of every shape at 0.32 ms. On an RX 7900 XTX (RADV) and a Radeon 780M the two shapes are
//! within measurement of each other.
//!
//! **Whole blocks** ([`ShadowMergeStores::Blocks`], the consumer's choice by the merge's
//! structure). The `LOCAL_SIZE` words a workgroup takes in one step, `4Ng + Nk` to
//! `4Ng + Nk + N - 1`, are one block of [`ShadowMergeModule::BLOCK_BYTES`]: each invocation first
//! compares its words of the kept and changed buffers and marks the step's block in the
//! workgroup's memory where its word differs (an atomic or, after the flags are cleared and a
//! barrier); after a second barrier, every word below `count` of a marked block is merged as
//! above, a word with no changed byte stored as the current buffer held it when read, and no word
//! of an unmarked block is read or written. So every block holding a change is stored whole, its
//! words consecutive across the invocations (a run per row it crosses), and the stores leave no
//! gaps inside it: on an RTX 5080 Laptop GPU, stores into imported host memory that leave words
//! between a warp's lanes take hundreds of times the time of the same bytes stored without gaps
//! (above). The bytes are the same as word by word where no other writer writes the current
//! buffer during the dispatch; one that writes a byte of a marked block outside its changed bytes
//! between the read and the store is undone. A consumer takes this form only where every word of
//! the region is bytes its commands wrote, or bytes Metal's own store of them writes back (a
//! render pass's tiles), so that such a write is a race with them (`ShadowMergeStores::Blocks`).
//!
//! **Synchronization** (consumer): the copies' `TRANSFER_WRITE` → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its writes →
//! `TRANSFER_READ` before the current buffer is copied back, or the bytes' next use.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, IoDir, LayoutRule,
    MemAccess, MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule,
};

/// The crate-owned shadow merge module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowMergeModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ShadowMergeModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The shadow's texels before the command, a storage buffer.
    pub const KEPT_BINDING: u32 = 0;
    /// The shadow's texels after the command, a storage buffer.
    pub const CHANGED_BINDING: u32 = 1;
    /// The buffer's bytes after the command, a storage buffer the dispatch merges into.
    pub const CURRENT_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: [`WORDS`](Self::WORDS) 4-byte words each.
    pub const LOCAL_SIZE: u32 = 64;
    /// The words one invocation merges, one after the other.
    pub const WORDS: u32 = 4;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the region's shape.
    pub const PUSH_BYTES: u32 = 16;
    /// Bytes of one block under whole-block stores ([`ShadowMergeStores::Blocks`]): the
    /// `LOCAL_SIZE` consecutive words a workgroup takes in one step (the module docs).
    pub const BLOCK_BYTES: u32 = 4 * Self::LOCAL_SIZE;

    /// The workgroups of a dispatch over `count` words.
    pub fn groups(count: u32) -> u32 {
        count.div_ceil(Self::LOCAL_SIZE * Self::WORDS)
    }

    /// The push bytes of a region of `count` words, `width` a row (at least 1), its rows
    /// `row_words` apart in the current buffer from its word `first_word` (the module docs).
    pub fn shape(width: u32, count: u32, row_words: u32, first_word: u32) -> [u8; 16] {
        let words = [width, count, row_words, first_word];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the kept buffer in the typed module.
const KEPT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the changed buffer in the typed module.
const CHANGED: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the current buffer in the typed module.
const CURRENT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
pub(super) fn words_buffer(
    b: &mut ModuleBuilder,
    sym: ResourceSlot,
    u32_: TypeId,
) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

pub(super) fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places (the region of the merge's dispatch).
const SHAPE: PushField = PushField::DispatchRegion;

/// The merge module as a typed module, before serialization: the module
/// [`shadow_merge_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `KEPT`, `CHANGED` and `CURRENT`, and its push field the shape, a `uvec4` (the module
/// docs).
fn typed_module(blocks: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let kept = words_buffer(&mut b, KEPT, u32_)?;
    let changed = words_buffer(&mut b, CHANGED, u32_)?;
    let current = words_buffer(&mut b, CURRENT, u32_)?;
    let block_flags = match blocks {
        true => Some(block_flags(&mut b, u32_, ShadowMergeModule::WORDS)?),
        false => None,
    };
    let group = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let local = b.builtin(crate::spv::BuiltIn::LocalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let group = f.module().global_value(group)?;
    let group = f.load(group, MemAccess::default())?;
    let g = f.composite_extract(group, &[0])?;
    let local = f.module().global_value(local)?;
    let local = f.load(local, MemAccess::default())?;
    let l = f.composite_extract(local, &[0])?;
    let kept = f.module().global_value(kept)?;
    let changed = f.module().global_value(changed)?;
    let current = f.module().global_value(current)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let width = f.composite_extract(shape, &[0])?;
    let count = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let first_word = f.composite_extract(shape, &[3])?;
    // The workgroup's first word, plus the invocation's place in it: its words lie `LOCAL_SIZE`
    // apart from there, so each of the four stores below falls on consecutive words across
    // consecutive invocations (the module docs).
    let span = u32_value(
        &mut f,
        ShadowMergeModule::WORDS * ShadowMergeModule::LOCAL_SIZE,
    )?;
    let base = f.binary(BinOp::IMul, g, span)?;
    let base = f.binary(BinOp::IAdd, base, l)?;
    if let Some(block_flags) = block_flags {
        // Whole blocks: each step's words, the workgroup's `LOCAL_SIZE` consecutive words, are
        // one block, stored whole where any of them changed (the module docs).
        let flags = Blocks::new(&mut f, block_flags, l)?;
        let mut words = Vec::new();
        for k in 0..ShadowMergeModule::WORDS {
            let i = match k {
                0 => base,
                k => {
                    let step = u32_value(&mut f, k * ShadowMergeModule::LOCAL_SIZE)?;
                    f.binary(BinOp::IAdd, base, step)?
                }
            };
            let inside = f.binary(BinOp::ULessThan, i, count)?;
            // A word past the region reads the region's first, and counts as unchanged.
            let read = f.select(inside, i, zero)?;
            let p = f.access_chain(kept, &[zero, read])?;
            let before = f.load(p, MemAccess::default())?;
            let p = f.access_chain(changed, &[zero, read])?;
            let after = f.load(p, MemAccess::default())?;
            let differ = f.binary(BinOp::BitwiseXor, before, after)?;
            let differ = f.select(inside, differ, zero)?;
            flags.mark(&mut f, k, differ)?;
            words.push((i, inside, after, differ));
        }
        barrier(&mut f)?;
        for (k, (i, inside, after, differ)) in (0..).zip(words) {
            let store = flags.stores(&mut f, k, inside)?;
            let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                let y = f.binary(BinOp::UDiv, i, width)?;
                let x = f.binary(BinOp::UMod, i, width)?;
                let row = f.binary(BinOp::IMul, y, row_words)?;
                let word = f.binary(BinOp::IAdd, first_word, row)?;
                let word = f.binary(BinOp::IAdd, word, x)?;
                let at = f.access_chain(current, &[zero, word])?;
                merge_word(f, at, after, differ)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), store, write, None)?;
        }
    } else {
        // The invocation's words, one after the other, each as one invocation took it alone.
        for k in 0..ShadowMergeModule::WORDS {
            let i = match k {
                0 => base,
                k => {
                    let step = u32_value(&mut f, k * ShadowMergeModule::LOCAL_SIZE)?;
                    f.binary(BinOp::IAdd, base, step)?
                }
            };
            let inside = f.binary(BinOp::ULessThan, i, count)?;
            let merge: ArmFn<'_, ()> = Box::new(move |f, _| {
                let p = f.access_chain(kept, &[zero, i])?;
                let before = f.load(p, MemAccess::default())?;
                let p = f.access_chain(changed, &[zero, i])?;
                let after = f.load(p, MemAccess::default())?;
                let differ = f.binary(BinOp::BitwiseXor, before, after)?;
                let moved = f.binary(BinOp::INotEqual, differ, zero)?;
                // A word the command left as it was is neither read nor written.
                let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                    let y = f.binary(BinOp::UDiv, i, width)?;
                    let x = f.binary(BinOp::UMod, i, width)?;
                    let row = f.binary(BinOp::IMul, y, row_words)?;
                    let word = f.binary(BinOp::IAdd, first_word, row)?;
                    let word = f.binary(BinOp::IAdd, word, x)?;
                    let at = f.access_chain(current, &[zero, word])?;
                    merge_word(f, at, after, differ)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                f.if_(&mut (), moved, write, None)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), inside, merge, None)?;
        }
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ShadowMergeModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The workgroup's flags of its blocks under whole-block stores: one word per block, `steps` of
/// them (the module docs).
pub(super) fn block_flags(b: &mut ModuleBuilder, u32_: TypeId, steps: u32) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(steps)),
    })?;
    let flags = b.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    b.plain_global(StorageClass::Workgroup, flags, None)
}

/// The workgroup barrier between the blocks' marks and their stores.
pub(super) fn barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// The flags of a workgroup's blocks under whole-block stores, each zero until a word of its
/// block changed (the module docs).
pub(super) struct Blocks {
    flags: Value,
    zero: Value,
}

impl Blocks {
    /// The flags `flags` (a [`block_flags`] global) cleared, invocation `l` clearing flag `l`
    /// where it has one, then a barrier, so each mark lands after every clear.
    pub(super) fn new(f: &mut FunctionBuilder<'_>, flags: GlobalId, l: Value) -> BResult<Blocks> {
        let zero = u32_value(f, 0)?;
        let flags = f.module().global_value(flags)?;
        let steps = u32_value(f, ShadowMergeModule::WORDS)?;
        let own = f.binary(BinOp::ULessThan, l, steps)?;
        let clear: ArmFn<'_, ()> = Box::new(move |f, _| {
            let at = f.access_chain(flags, &[l])?;
            f.store(at, zero, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), own, clear, None)?;
        barrier(f)?;
        Ok(Blocks { flags, zero })
    }

    /// Mark block `k` changed where `differ` (a word's xor of its kept and changed words) is not
    /// zero.
    pub(super) fn mark(&self, f: &mut FunctionBuilder<'_>, k: u32, differ: Value) -> BResult<()> {
        let (flags, zero) = (self.flags, self.zero);
        let moved = f.binary(BinOp::INotEqual, differ, zero)?;
        let at = u32_value(f, k)?;
        let mark: ArmFn<'_, ()> = Box::new(move |f, _| {
            let at = f.access_chain(flags, &[at])?;
            let one = u32_value(f, 1)?;
            f.atomic(AtomicOp::Or, at, Scope::Workgroup, Semantics::RELAXED, one)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), moved, mark, None)?;
        Ok(())
    }

    /// Whether a word of block `k`, `inside` the region, is stored: its block changed. Read after
    /// the barrier that follows every mark.
    pub(super) fn stores(
        &self,
        f: &mut FunctionBuilder<'_>,
        k: u32,
        inside: Value,
    ) -> BResult<Value> {
        let at = u32_value(f, k)?;
        let at = f.access_chain(self.flags, &[at])?;
        let flag = f.load(at, MemAccess::default())?;
        let changed = f.binary(BinOp::INotEqual, flag, self.zero)?;
        f.binary(BinOp::LogicalAnd, inside, changed)
    }
}

/// Merge into the current word at `at` the changed word `after`, `differ` its xor with the kept
/// word: each byte where they differ becomes `after`'s, and the others keep the current word's
/// (the module docs); where `differ` is zero (a word of a changed block under whole-block
/// stores), the current word is stored as read.
pub(super) fn merge_word(
    f: &mut FunctionBuilder<'_>,
    at: Value,
    after: Value,
    differ: Value,
) -> BResult<()> {
    let zero = u32_value(f, 0)?;
    let all = u32_value(f, u32::MAX)?;
    let now = f.load(at, MemAccess::default())?;
    // The bytes that differ, each 0xff in the mask.
    let mut mask = zero;
    for byte in 0..4 {
        let lane = u32_value(f, 0xff << (8 * byte))?;
        let bits = f.binary(BinOp::BitwiseAnd, differ, lane)?;
        let moved = f.binary(BinOp::INotEqual, bits, zero)?;
        let taken = f.select(moved, lane, zero)?;
        mask = f.binary(BinOp::BitwiseOr, mask, taken)?;
    }
    let others = f.binary(BinOp::BitwiseXor, mask, all)?;
    let left = f.binary(BinOp::BitwiseAnd, now, others)?;
    let taken = f.binary(BinOp::BitwiseAnd, after, mask)?;
    let merged = f.binary(BinOp::BitwiseOr, left, taken)?;
    f.store(at, merged, MemAccess::default())?;
    Ok(())
}

/// How a merge module stores the region's changed bytes into the current buffer (the module
/// docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShadowMergeStores {
    /// The changed words alone: a word with no changed byte is neither read nor written, so
    /// another writer's bytes there are never rewritten.
    Words,
    /// Whole blocks of [`ShadowMergeModule::BLOCK_BYTES`]: a block holding a changed word is
    /// stored whole, its words the commands left as read, and no other block is read or
    /// written. For a region every byte of which the consumer's commands wrote (a command that
    /// replaced every texel of a texture, in a region of its texels' rows): another writer's
    /// write to a byte of it is a race with those commands, which Metal leaves unordered.
    Blocks,
}

impl ShadowMergeStores {
    fn blocks(self) -> bool {
        self == ShadowMergeStores::Blocks
    }
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ShadowMergeModule::SET),
        Symbol::Binding(s) if *s == KEPT => Some(ShadowMergeModule::KEPT_BINDING),
        Symbol::Binding(s) if *s == CHANGED => Some(ShadowMergeModule::CHANGED_BINDING),
        Symbol::Binding(s) if *s == CURRENT => Some(ShadowMergeModule::CURRENT_BINDING),
        _ => None,
    }
}

/// The shadow merge module of `device`, storing as `stores` says: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`ShadowMergeModule`].
pub fn shadow_merge_module(
    device: &Device,
    stores: ShadowMergeStores,
) -> Result<ShadowMergeModule, Error> {
    let profile = device.profile();
    let m = typed_module(stores.blocks())?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the shadow merge needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ShadowMergeModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, ShadowMergeModule, CHANGED, CURRENT, KEPT};
    use crate::spirv::interp::Outcome;
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_merge_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for blocks in [false, true] {
                assert_valid(typed_module(blocks).expect("module"), &p);
            }
        }
    }

    /// Run the kernel over `kept`, `changed` and `current` as one row of `kept`'s words, in
    /// `groups` workgroups, both invocation orders: the current buffer after it.
    fn merged(kept: &[u8], changed: &[u8], current: &[u8], groups: u32) -> Vec<u8> {
        let words = (kept.len() / 4) as u32;
        let shape = ShadowMergeModule::shape(words, words, 0, 0);
        merged_in(kept, changed, current, &shape, groups)
    }

    /// Run the kernel over `kept`, `changed` and `current` with the push bytes `shape`, in
    /// `groups` workgroups, both invocation orders and both forms of its stores (whole blocks and
    /// changed words, which give the same bytes where no other writer runs beside the kernel):
    /// the current buffer after it, the others unchanged.
    fn merged_in(
        kept: &[u8],
        changed: &[u8],
        current: &[u8],
        shape: &[u8; 16],
        groups: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        for blocks in [false, true] {
            let (bytes, outcome) = merged_form(kept, changed, current, shape, groups, blocks);
            // Whole blocks take two workgroup barriers (the clear, the marks); words none.
            let barriers = if blocks { 2 } else { 0 };
            assert!(outcome.iter().all(|i| i.barriers == barriers), "{blocks}");
            out.push(bytes);
        }
        assert_eq!(out[0], out[1], "whole blocks store the bytes words do");
        out.swap_remove(0)
    }

    /// [`merged_in`] in one form of the stores, whole `blocks` or words: the current buffer
    /// after it, and the invocations' stats.
    fn merged_form(
        kept: &[u8],
        changed: &[u8],
        current: &[u8],
        shape: &[u8; 16],
        groups: u32,
        blocks: bool,
    ) -> (Vec<u8>, Vec<crate::spirv::interp::InvocationStats>) {
        let m = typed_module(blocks).expect("module");
        let mut out = Vec::new();
        let mut stats = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let part = |slot, bytes: &[u8]| Buffer {
                slot: Some(slot),
                address: None,
                bytes: bytes.to_vec(),
            };
            let res = Resources {
                buffers: vec![
                    part(KEPT, kept),
                    part(CHANGED, changed),
                    part(CURRENT, current),
                ],
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups, 1, 1],
                ..Config::default()
            };
            let Outcome {
                buffers,
                invocations,
                ..
            } = run(&m, &cfg, res).expect("run");
            assert_eq!(buffers[0], kept, "the kept buffer is only read");
            assert_eq!(buffers[1], changed, "the changed buffer is only read");
            out.push(buffers[2].clone());
            stats = invocations;
        }
        assert_eq!(out[0], out[1], "order independent");
        (out.swap_remove(0), stats)
    }

    /// Each byte the texture changed is the changed buffer's; every other byte keeps the current
    /// buffer's, the other way's writes among them, inside one word too: Metal probe cases
    /// `k.two_ways_disjoint` (a texel through the texture, the next one through the buffer) and
    /// `k.two_ways_gap` (the buffer's bytes between rows), at byte granularity, on the M3 Max and
    /// an M4.
    #[test]
    fn the_bytes_the_texture_changed_merge_into_the_current_bytes() {
        let kept: Vec<u8> = (0..32u8).collect();
        let mut changed = kept.clone();
        // The texture writes texel 0 and one byte of word 5; a write of an equal value is no
        // change.
        changed[0..4].copy_from_slice(&[0xff, 0x00, 0x00, 0xff]);
        changed[21] = 0xa5;
        changed[9] = kept[9];
        let mut current = kept.clone();
        // The other way writes texel 1, and the other bytes of word 5.
        current[4..8].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
        current[20] = 0x11;
        current[22] = 0x22;
        current[23] = 0x33;
        let mut want = current.clone();
        want[0..4].copy_from_slice(&[0xff, 0x00, 0x00, 0xff]);
        want[21] = 0xa5;
        assert_eq!(merged(&kept, &changed, &current, 1), want);
    }

    /// A region of rows inside the current buffer: each changed word lands at its row's place,
    /// from `first_word`, `row_words` apart; a word whose kept and changed words are equal keeps
    /// the current buffer's, another writer's included; and every word outside the region keeps
    /// its own (a surface's bytes beside a pass's box).
    #[test]
    fn a_region_of_rows_merges_its_changed_words_alone() {
        // Two rows of three words, from word 2 of rows of five.
        let kept: Vec<u8> = (0..24u8).collect();
        let mut changed = kept.clone();
        // The pass changes word 0 (row 0) and word 4 (row 1, its second); it writes word 5 with
        // its own value, which is no change.
        changed[0..4].copy_from_slice(&[0xa0, 0xa1, 0xa2, 0xa3]);
        changed[17] = 0xb1;
        let current: Vec<u8> = (100..160u8).collect();
        let mut want = current.clone();
        // Row 0's word 0 is current word 2; row 1's word 1 is current word 2 + 5 + 1 = 8, of
        // which only byte 1 changed.
        want[8..12].copy_from_slice(&[0xa0, 0xa1, 0xa2, 0xa3]);
        want[33] = 0xb1;
        let shape = ShadowMergeModule::shape(3, 6, 5, 2);
        assert_eq!(merged_in(&kept, &changed, &current, &shape, 1), want);
    }

    /// The merge's rule word by word, over a region whose words are not a multiple of the
    /// `WORDS * LOCAL_SIZE` a workgroup takes (so the last workgroup's invocations run past
    /// them, each word of the tail guarded on its own) and over a mix of words the command left
    /// as they were, words it changed whole and words of which it changed one byte: the current
    /// buffer is what the rule gives, computed here without the kernel, and no word between the
    /// region's rows, before its first or past its last changes. Which invocation takes which
    /// word is the layout's own business: each word's merge reads and writes that word alone.
    #[test]
    fn a_tailed_region_of_rows_merges_word_for_word_by_the_rule() {
        // 37 words a row, 19 rows: 703 words, 2 whole workgroups and 191 of a third.
        let (width, rows, row_words, first_word) = (37u32, 19u32, 53u32, 11u32);
        let count = width * rows;
        assert_ne!(
            count % (ShadowMergeModule::WORDS * ShadowMergeModule::LOCAL_SIZE),
            0,
            "the region's words leave a tail",
        );
        let byte = |i: u32, b: u32| (i.wrapping_mul(7).wrapping_add(b * 31) % 251) as u8;
        let kept: Vec<u8> = (0..4 * count).map(|at| byte(at / 4, at % 4)).collect();
        // Word i: unchanged where i % 3 == 0, changed whole where i % 3 == 1, and its byte
        // i % 4 alone where i % 3 == 2.
        let mut changed = kept.clone();
        for i in 0..count {
            match i % 3 {
                0 => {}
                1 => {
                    for b in 0..4 {
                        changed[(4 * i + b) as usize] ^= 0xff;
                    }
                }
                _ => changed[(4 * i + i % 4) as usize] ^= 0x5a,
            }
        }
        // The current buffer: the region's rows `row_words` apart from `first_word`, and two
        // words past the last row, every byte another writer's.
        let last = first_word + (rows - 1) * row_words + width;
        let current: Vec<u8> = (0..4 * (last + 2))
            .map(|at| byte(at / 4 + 1000, at % 4) ^ 0xa5)
            .collect();
        let mut want = current.clone();
        for i in 0..count {
            let word = first_word + (i / width) * row_words + i % width;
            for b in 0..4 {
                let (k, c) = (kept[(4 * i + b) as usize], changed[(4 * i + b) as usize]);
                let equal = kept[(4 * i) as usize..][..4] == changed[(4 * i) as usize..][..4];
                if !equal && k != c {
                    want[(4 * word + b) as usize] = c;
                }
            }
        }
        let shape = ShadowMergeModule::shape(width, count, row_words, first_word);
        let groups = ShadowMergeModule::groups(count);
        assert_eq!(groups, 3, "three workgroups, the last one over the tail");
        assert_eq!(merged_in(&kept, &changed, &current, &shape, groups), want);
    }

    /// Whole blocks: a block with no changed word reads and writes no word of the current
    /// buffer, and a word of a changed block the command left keeps the current buffer's bytes.
    /// The current buffer holds the first block's words alone, so an access to any other block's
    /// traps (out of bounds) and fails the run; the words form touches the changed word alone.
    #[test]
    fn whole_blocks_touch_no_word_of_an_unchanged_block() {
        let block = ShadowMergeModule::BLOCK_BYTES as usize;
        // Two workgroups' words: eight blocks.
        let words = 2 * (ShadowMergeModule::WORDS * ShadowMergeModule::LOCAL_SIZE) as usize;
        let kept = vec![0u8; 4 * words];
        let mut changed = kept.clone();
        // One byte of word 3, in the first block.
        changed[13] = 0x5a;
        let current: Vec<u8> = (0..block).map(|i| (i % 251) as u8 ^ 0xc3).collect();
        let mut want = current.clone();
        want[13] = 0x5a;
        let count = words as u32;
        let shape = ShadowMergeModule::shape(count, count, 0, 0);
        for blocks in [true, false] {
            let (bytes, _) = merged_form(&kept, &changed, &current, &shape, 2, blocks);
            assert_eq!(bytes, want, "{blocks}");
        }
    }

    /// Invocations past the words write nothing, and every word of several workgroups merges.
    #[test]
    fn every_word_merges_across_workgroups_and_none_past_them() {
        let words = 64 * 3 + 5;
        let kept = vec![0u8; 4 * words];
        let changed: Vec<u8> = (0..4 * words).map(|i| (i % 7) as u8).collect();
        let current = vec![0xeeu8; 4 * words];
        let want: Vec<u8> = changed
            .iter()
            .map(|&c| if c == 0 { 0xee } else { c })
            .collect();
        assert_eq!(merged(&kept, &changed, &current, 4), want);
    }
}
