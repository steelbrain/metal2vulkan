//! The combine of a split pass's continuation queries (step 6's fourth part).
//!
//! A query begins and ends in one subpass, so where a Metal render pass runs as several render
//! pass instances (a split) a visibility result whose mode spans them is the answer of several
//! queries: the first, copied to the result's offset as every query is, and its continuations,
//! one per later instance, each copied to a word of a scratch buffer of the consumer's
//! (`vkCmdCopyQueryPoolResults`). One dispatch of [`visibility_combine_module`] then combines each
//! result's continuation words into its word. A `Boolean` result ORs them: an imprecise query's
//! count is any nonzero value when a sample passed, and a sum of such counts could wrap to zero, so
//! the OR is zero exactly where no instance passed a sample. A `Counting` result, of precise
//! queries, sums them: the samples that passed in every instance.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`VisibilityCombineModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers in
//! set 0:
//!
//! | Binding | Buffer | Range |
//! |---|---|---|
//! | [`VisibilityCombineModule::TABLE_BINDING`] | the consumer's upload of [`VisibilityCombine::table`] | exactly its bytes |
//! | [`VisibilityCombineModule::DEST_BINDING`] | the visibility buffer | [`VisibilityCombine::dest_range`] |
//! | [`VisibilityCombineModule::SOURCES_BINDING`] | the continuation words | all of them |
//!
//! Invocation `i` below the table's entries takes entry `i`, four words: the index of a result's
//! 64-bit word inside the bound visibility range, the index of its first continuation word among
//! the sources, its continuation count, and 1 to sum them (0 to OR them). It combines that many
//! consecutive source words into the result's word: ORed half by half, or added as 64-bit
//! integers, wrapping. A word past its bound range is skipped, and the loop stops at the sources'
//! end, so a wrongly bound range can never make the combine fault or run past its words. Each
//! result is one entry, so no two invocations write one word.
//!
//! **Synchronization** (consumer): the result and continuation copies' `TRANSFER_WRITE` →
//! `SHADER_STORAGE_READ` and `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the combine, and
//! the upload's host write before the submission; the combine's writes are then ordered before the
//! buffer's later uses as any compute write is.

use std::sync::Arc;

use crate::bind::instantiate::BufferRange;
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// Bytes of one result in the visibility buffer, and of one continuation word.
const RESULT_BYTES: u64 = 8;

/// One dispatch of [`visibility_combine_module`]: it combines, in place, the continuation words
/// of the results whose words lie in `dest_range` of the visibility buffer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VisibilityCombine {
    /// The range of the visibility buffer to bind at [`VisibilityCombineModule::DEST_BINDING`]:
    /// from the first word's offset aligned down to `minStorageBufferOffsetAlignment` through
    /// the last word's end.
    pub dest_range: BufferRange,
    /// The table to upload (little-endian `u32`s) and bind, exactly, at
    /// [`VisibilityCombineModule::TABLE_BINDING`]: per result, ascending by offset, the index of
    /// its 64-bit word inside `dest_range`, the index of its first 64-bit continuation word among
    /// the sources, its continuation count, and 1 where they are summed (0 where ORed).
    pub table: Vec<u32>,
    /// Workgroups to dispatch along x, each of [`VisibilityCombineModule::LOCAL_SIZE`]
    /// invocations.
    pub group_count: u32,
}

/// One result with continuations, as [`VisibilityCombine::plan`] takes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContinuedResult {
    /// The result's offset in the visibility buffer as bound (a multiple of 8).
    pub offset: u64,
    /// The index of its first 64-bit continuation word among the sources.
    pub first: u32,
    /// Its continuation words, consecutive from `first`.
    pub count: u32,
    /// Whether they are summed (a `Counting` result, of precise queries) rather than ORed (a
    /// `Boolean` one).
    pub sum: bool,
}

impl ContinuedResult {
    /// The result at `offset` with `count` continuation words from word `first`, summed where
    /// `sum` and ORed otherwise.
    pub fn new(offset: u64, first: u32, count: u32, sum: bool) -> ContinuedResult {
        ContinuedResult {
            offset,
            first,
            count,
            sum,
        }
    }
}

impl VisibilityCombine {
    /// Words of one table entry.
    pub const ENTRY_WORDS: usize = 4;

    /// The dispatches that combine `results` (distinct offsets, any order), each binding the
    /// visibility buffer at an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`)
    /// over at most `max_range` bytes (`maxStorageBufferRange`) and dispatching at most
    /// `max_groups` workgroups (`maxComputeWorkGroupCount[0]`). Results too far apart for one
    /// binding, or too many for one dispatch, take more dispatches. No dispatch when `results` is
    /// empty. `None` when an offset is not a multiple of 8 (Metal requires it of a visibility
    /// offset), its word overflows, or one word does not fit `max_range` past its aligned offset,
    /// or `max_groups` is 0; when two results share an offset; or when a result's continuation
    /// words run past `max_range` bytes of sources, which the consumer binds from their first
    /// word.
    pub fn plan(
        results: &[ContinuedResult],
        alignment: u64,
        max_range: u32,
        max_groups: u32,
    ) -> Option<Vec<VisibilityCombine>> {
        let mut sorted = results.to_vec();
        sorted.sort_unstable_by_key(|r| r.offset);
        if sorted.windows(2).any(|w| w[0].offset == w[1].offset) {
            return None;
        }
        let source_words = u64::from(max_range) / RESULT_BYTES;
        if sorted
            .iter()
            .any(|r| u64::from(r.first) + u64::from(r.count) > source_words)
        {
            return None;
        }
        let max_entries =
            u64::from(max_groups).checked_mul(u64::from(VisibilityCombineModule::LOCAL_SIZE))?;
        let items = sorted
            .into_iter()
            .map(|r| (r.offset, [r.first, r.count, u32::from(r.sum)]));
        windows(items, alignment, max_range, max_entries)?
            .into_iter()
            .map(|(dest_range, entries)| {
                let table: Vec<u32> = entries
                    .iter()
                    .flat_map(|&(index, [first, count, sum])| [index, first, count, sum])
                    .collect();
                let groups =
                    (entries.len() as u64).div_ceil(u64::from(VisibilityCombineModule::LOCAL_SIZE));
                Some(VisibilityCombine {
                    dest_range,
                    table,
                    group_count: u32::try_from(groups).ok()?,
                })
            })
            .collect()
    }

    /// The table as the bytes to upload.
    pub fn table_bytes(&self) -> Vec<u8> {
        self.table.iter().flat_map(|t| t.to_le_bytes()).collect()
    }
}

/// The windows of the visibility buffer that bind `items`, each at Metal's offset of its result
/// (ascending, each once): each window binds from an offset aligned down to `alignment` (at
/// least 8) over at most `max_range` bytes, and takes at most `max_items` items, each with the
/// index of its 64-bit word inside the window. A dispatch per window. `None` when an offset is not
/// a multiple of 8, its word overflows, or one word does not fit `max_range` past its aligned
/// offset, or `max_items` is 0 (with an item).
fn windows<T>(
    items: impl IntoIterator<Item = (u64, T)>,
    alignment: u64,
    max_range: u32,
    max_items: u64,
) -> Option<Vec<(BufferRange, Vec<(u32, T)>)>> {
    let a = alignment.max(RESULT_BYTES);
    let mut out: Vec<(BufferRange, Vec<(u32, T)>)> = Vec::new();
    for (offset, item) in items {
        if !offset.is_multiple_of(RESULT_BYTES) {
            return None;
        }
        let end = offset.checked_add(RESULT_BYTES)?;
        let joins = out.last().is_some_and(|(range, taken)| {
            end - range.offset <= u64::from(max_range) && (taken.len() as u64) < max_items
        });
        if !joins {
            let start = offset - offset % a;
            if end - start > u64::from(max_range) || max_items == 0 {
                return None;
            }
            let range = BufferRange {
                offset: start,
                range: 0,
            };
            out.push((range, Vec::new()));
        }
        let (range, taken) = out.last_mut()?;
        // Inside `max_range` (a `u32`) of the range's start: the index fits.
        let index = u32::try_from((offset - range.offset) / RESULT_BYTES).ok()?;
        taken.push((index, item));
        range.range = end - range.offset;
    }
    Some(out)
}

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

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The crate-owned continuation combine module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibilityCombineModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl VisibilityCombineModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The table of results, a storage buffer.
    pub const TABLE_BINDING: u32 = 0;
    /// The visibility buffer, a storage buffer.
    pub const DEST_BINDING: u32 = 1;
    /// The continuation words, a storage buffer.
    pub const SOURCES_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The binding symbol of the table in the typed module
/// ([`VisibilityCombineModule::TABLE_BINDING`]).
pub const TABLE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the visibility buffer in the typed module
/// ([`VisibilityCombineModule::DEST_BINDING`]).
pub const DEST: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the continuation words in the typed module
/// ([`VisibilityCombineModule::SOURCES_BINDING`]).
pub const SOURCES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// The combine module as a typed module, before serialization: the module
/// [`visibility_combine_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`TABLE`], [`DEST`] and [`SOURCES`].
pub fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let table = words_buffer(&mut b, TABLE, u32_)?;
    let dest = words_buffer(&mut b, DEST, u32_)?;
    let sources = words_buffer(&mut b, SOURCES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero_const = f.u32_const(0)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let two = u32_value(&mut f, 2)?;
    let three = u32_value(&mut f, 3)?;
    let entry_words = u32_value(&mut f, VisibilityCombine::ENTRY_WORDS as u32)?;
    // The loop's counter and the result's two halves, as it combines them.
    let step = f.local_var(u32_, Some(zero_const))?;
    let step = f.var_ptr(step)?;
    let low_acc = f.local_var(u32_, Some(zero_const))?;
    let low_acc = f.var_ptr(low_acc)?;
    let high_acc = f.local_var(u32_, Some(zero_const))?;
    let high_acc = f.var_ptr(high_acc)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let table = f.module().global_value(table)?;
    let table_len = f.array_length(table, 0)?;
    let entries = f.binary(BinOp::UDiv, table_len, entry_words)?;
    let dest = f.module().global_value(dest)?;
    let dest_len = f.array_length(dest, 0)?;
    let sources = f.module().global_value(sources)?;
    let sources_len = f.array_length(sources, 0)?;
    let source_words = f.binary(BinOp::ShiftRightLogical, sources_len, one)?;
    let listed = f.binary(BinOp::ULessThan, i, entries)?;
    // Invocation `i` takes entry `i`: its result's word, when both halves lie in the range.
    let combine: ArmFn<'_, ()> = Box::new(move |f, _| {
        let base = f.binary(BinOp::IMul, i, entry_words)?;
        let entry = |f: &mut FunctionBuilder<'_>, k: Value| {
            let at = f.binary(BinOp::IAdd, base, k)?;
            let p = f.access_chain(table, &[zero, at])?;
            f.load(p, MemAccess::default())
        };
        let target = entry(f, zero)?;
        let first = entry(f, one)?;
        let count = entry(f, two)?;
        let sum = entry(f, three)?;
        let sum = f.binary(BinOp::INotEqual, sum, zero)?;
        let low_index = f.binary(BinOp::ShiftLeftLogical, target, one)?;
        let high_index = f.binary(BinOp::IAdd, low_index, one)?;
        let inside = f.binary(BinOp::ULessThan, high_index, dest_len)?;
        let combine_in: ArmFn<'_, ()> = Box::new(move |f, _| {
            // The words it may read: `count`, cut at the sources' end.
            let past = f.binary(BinOp::UGreaterThanEqual, first, source_words)?;
            let room = f.binary(BinOp::ISub, source_words, first)?;
            let room = f.select(past, zero, room)?;
            let fits = f.binary(BinOp::ULessThanEqual, count, room)?;
            let n = f.select(fits, count, room)?;
            let low_ptr = f.access_chain(dest, &[zero, low_index])?;
            let high_ptr = f.access_chain(dest, &[zero, high_index])?;
            let low = f.load(low_ptr, MemAccess::default())?;
            let high = f.load(high_ptr, MemAccess::default())?;
            f.store(low_acc, low, MemAccess::default())?;
            f.store(high_acc, high, MemAccess::default())?;
            f.store(step, zero, MemAccess::default())?;
            let body: LoopFn<'_, ()> = Box::new(move |f, cx, lc| {
                let k = f.load(step, MemAccess::default())?;
                let done = f.binary(BinOp::UGreaterThanEqual, k, n)?;
                let exit: ArmFn<'_, ()> = Box::new(move |f, _| {
                    lc.break_(f)?;
                    Ok(ArmEnd::Exits)
                });
                f.if_(cx, done, exit, None)?;
                let next = f.binary(BinOp::IAdd, k, one)?;
                f.store(step, next, MemAccess::default())?;
                let word = f.binary(BinOp::IAdd, first, k)?;
                let low_source = f.binary(BinOp::ShiftLeftLogical, word, one)?;
                let high_source = f.binary(BinOp::IAdd, low_source, one)?;
                let low_ptr = f.access_chain(sources, &[zero, low_source])?;
                let high_ptr = f.access_chain(sources, &[zero, high_source])?;
                let low = f.load(low_ptr, MemAccess::default())?;
                let high = f.load(high_ptr, MemAccess::default())?;
                let low_so_far = f.load(low_acc, MemAccess::default())?;
                let high_so_far = f.load(high_acc, MemAccess::default())?;
                // ORed half by half, or added with the low half's carry into the high half.
                let low_or = f.binary(BinOp::BitwiseOr, low_so_far, low)?;
                let high_or = f.binary(BinOp::BitwiseOr, high_so_far, high)?;
                let low_sum = f.binary(BinOp::IAdd, low_so_far, low)?;
                let carried = f.binary(BinOp::ULessThan, low_sum, low_so_far)?;
                let carry = f.select(carried, one, zero)?;
                let high_sum = f.binary(BinOp::IAdd, high_so_far, high)?;
                let high_sum = f.binary(BinOp::IAdd, high_sum, carry)?;
                let low_next = f.select(sum, low_sum, low_or)?;
                let high_next = f.select(sum, high_sum, high_or)?;
                f.store(low_acc, low_next, MemAccess::default())?;
                f.store(high_acc, high_next, MemAccess::default())?;
                Ok(())
            });
            f.loop_(&mut (), LoopSpec::default(), body)?;
            let low = f.load(low_acc, MemAccess::default())?;
            let high = f.load(high_acc, MemAccess::default())?;
            f.store(low_ptr, low, MemAccess::default())?;
            f.store(high_ptr, high, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), inside, combine_in, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), listed, combine, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: VisibilityCombineModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(VisibilityCombineModule::SET),
        Symbol::Binding(s) if *s == TABLE => Some(VisibilityCombineModule::TABLE_BINDING),
        Symbol::Binding(s) if *s == DEST => Some(VisibilityCombineModule::DEST_BINDING),
        Symbol::Binding(s) if *s == SOURCES => Some(VisibilityCombineModule::SOURCES_BINDING),
        _ => None,
    }
}

/// The continuation combine module of `device`: built with the typed builder, checked by the
/// owned verifier against the device's profile (requirements included), serialized, and numbered
/// with the fixed bindings of [`VisibilityCombineModule`].
pub fn visibility_combine_module(device: &Device) -> Result<VisibilityCombineModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the visibility combine needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(VisibilityCombineModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
