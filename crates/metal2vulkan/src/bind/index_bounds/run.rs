//! A run of the index bounds pass: indexed indirect draws over one arguments buffer and one index
//! buffer, planned by one dispatch of one invocation and filled by one dispatch for all of them,
//! where a draw's own pass takes two dispatches between three barriers. Each draw's plan and fill
//! are its own pass's, over its slot of the run's words, so its two draws and its reservation are
//! the ones its own pass makes, the plans reserving one after another as passes recorded in turn
//! do.
//!
//! - **The words** ([`IndexBoundsRun::words`], at both modules' `WORDS_BINDING`):
//!   [`IndexBoundsRun::HEADER_WORDS`] words of header, then one slot of
//!   [`IndexBoundsRun::SLOT_WORDS`] words per draw.
//!
//!   | Header word | Written by | Contents |
//!   |---|---|---|
//!   | 0 | the consumer | the run's draws |
//!   | 1 to 3 | the plan | the fill's `VkDispatchIndirectCommand` ([`IndexBoundsRun::DISPATCH_OFFSET`]): the run's fill workgroups, at most word 4's along x and as few along y as hold them |
//!   | 4 | the consumer | the most workgroups of a dispatch along x (`maxComputeWorkGroupCount[0]`) |
//!   | 5 to 7 | nobody | zero |
//!
//!   A slot's words 0 to 23 are a pass's bounds words (the parent module's table), its two draws
//!   at its words 8 and 13; its word
//!   24, which the plan writes, is the draw's first fill workgroup (the fill workgroups of the
//!   draws before it).
//! - **The plan** ([`index_bounds_plan_run_module`]) plans each draw in turn, then writes the
//!   fill's dispatch; **the fill** ([`index_bounds_fill_run_module`]) finds its workgroup's draw
//!   (the last whose first fill workgroup is at or before it; workgroup `(x, y)` is the run's
//!   `y · width + x`, `width` the dispatch's along x) and fills its words. The draws' fill
//!   workgroups summed may pass what one dispatch takes along x where each draw's alone does not,
//!   so the dispatch spreads them over y; a workgroup past the last draw's fills nothing.
//! - **Bindings**: a pass's, the arguments and the indices the run's bytes
//!   ([`IndexBoundsRun::arguments_range`], [`IndexBoundsRun::indices_range`]), so one binding
//!   serves every draw.
//! - **Synchronization** (consumer): a pass's, the run's plan and fill in place of a pass's.

use super::{
    fill_binding_of, fill_draw, fill_module_of, finished, k, load_word, plan_binding_of, plan_draw,
    plan_module_of, store_word, word_ptr, word_range, IndexBoundsFillModule, IndexBoundsPass,
    IndexBoundsPlanModule, Output, Parameter, Words,
};
use crate::bind::index_rewrite::{for_range, when};
use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::device::Device;
use crate::error::Error;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, MemAccess};

/// A run of indexed indirect draws of the index bounds pass (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexBoundsRun {
    /// The bound arguments bytes.
    arguments: BufferRange,
    /// The bound index bytes, and their end.
    indices: Option<(BufferRange, u64)>,
    alignment: u64,
    max_range: u32,
    /// The most workgroups of a dispatch along x.
    max_groups: u32,
    pool_bytes: u32,
    /// Each draw's slot.
    slots: Vec<[u32; IndexBoundsRun::SLOT_WORDS as usize]>,
}

/// One draw of a run: where its Metal arguments lie (in the bound arguments buffer's offsets),
/// its index size (2 or 4), its index offset, and the end of the bytes its fetch reads (in the
/// bound index buffer's offsets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundsRunDraw {
    /// The offset of the draw's `MTLDrawIndexedPrimitivesIndirectArguments`.
    pub arguments_offset: u64,
    /// The draw's index size in bytes.
    pub index_size: u64,
    /// The draw's index offset: a multiple of its index size.
    pub index_offset: u64,
    /// Where the draw's fetch stops reading bytes: a position from there on reads index 0.
    pub index_end: u64,
}

/// A slot's word past a pass's bounds words: the draw's first fill workgroup.
const FIRST_GROUP: u32 = 24;

impl IndexBoundsRun {
    /// Words of the header.
    pub const HEADER_WORDS: u32 = 8;
    /// Words of a draw's slot: a pass's bounds words and the draw's first fill workgroup,
    /// rounded up to 16 bytes.
    pub const SLOT_WORDS: u32 = 28;
    /// Bytes of a draw's slot.
    pub const SLOT_BYTES: u64 = 4 * Self::SLOT_WORDS as u64;
    /// The fill's `VkDispatchIndirectCommand` in the words.
    pub const DISPATCH_OFFSET: u64 = 4;

    /// An empty run of draws whose arguments lie in the bytes from `arguments_start` to
    /// `arguments_end` of a buffer, and with `indices` the bytes of an index buffer it binds
    /// (`None` where none lies inside it), each bound at an offset aligned to `alignment`
    /// (`minStorageBufferOffsetAlignment`) over at most `max_range` bytes
    /// (`maxStorageBufferRange`), into a pool of `pool_bytes` index bytes, its fill dispatched
    /// over at most `max_groups` workgroups along x (`maxComputeWorkGroupCount[0]`). `None` where
    /// the arguments' bytes are none, where bytes are past `max_range`, and where `max_groups` is
    /// 0.
    pub fn new(
        (arguments_start, arguments_end): (u64, u64),
        indices: Option<(u64, u64)>,
        (alignment, max_range, max_groups): (u64, u32, u32),
        pool_bytes: u32,
    ) -> Option<IndexBoundsRun> {
        if arguments_end <= arguments_start || max_groups == 0 {
            return None;
        }
        let (arguments, _) = word_range(arguments_start, arguments_end, alignment, max_range)?;
        let indices = match indices {
            Some((start, end)) if end > start => {
                let (range, _) = word_range(start, end, alignment, max_range)?;
                Some((range, range.offset + range.range))
            }
            _ => None,
        };
        Some(IndexBoundsRun {
            arguments,
            indices,
            alignment,
            max_range,
            max_groups,
            pool_bytes,
            slots: Vec::new(),
        })
    }

    /// Add `draw` to the run: its slot. `None` where the run cannot take it, and is as before:
    /// where its own pass would not be made ([`IndexBoundsPass::new`]), and where its arguments or
    /// the indices it reads lie outside the run's bound bytes.
    pub fn push(&mut self, draw: BoundsRunDraw) -> Option<u32> {
        let slot = u32::try_from(self.slots.len()).ok()?;
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, draw.arguments_offset),
            draw.index_size,
            draw.index_offset,
            draw.index_end,
            self.alignment,
            self.max_range,
            self.pool_bytes,
        )?;
        let mut parameters = pass.parameters;
        // Its arguments and indices within the run's bound bytes: their words there.
        let arguments_end = draw.arguments_offset.checked_add(20)?;
        if draw.arguments_offset < self.arguments.offset
            || arguments_end > self.arguments.offset + self.arguments.range
        {
            return None;
        }
        parameters[Parameter::ArgumentsWord as usize] =
            u32::try_from((draw.arguments_offset - self.arguments.offset) / 4).ok()?;
        if pass.indices_range.is_some() {
            let inside = u64::from(parameters[Parameter::Inside as usize]);
            let end = draw
                .index_offset
                .checked_add(inside.checked_mul(draw.index_size)?)?;
            let (range, bound_end) = self.indices?;
            if draw.index_offset < range.offset || end > bound_end {
                return None;
            }
            parameters[Parameter::IndicesByte as usize] =
                u32::try_from(draw.index_offset - range.offset).ok()?;
        }
        let mut words = [0; Self::SLOT_WORDS as usize];
        words[..IndexBoundsPass::PARAMETER_WORDS].copy_from_slice(&parameters);
        self.slots.push(words);
        Some(slot)
    }

    /// The run's draws.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the run has no draw.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The range of the indirect buffer to bind at [`IndexBoundsPlanModule::ARGUMENTS_BINDING`].
    pub fn arguments_range(&self) -> BufferRange {
        self.arguments
    }

    /// The range of the index buffer to bind at [`IndexBoundsFillModule::INDICES_BINDING`]: the
    /// bytes [`new`](Self::new) names, aligned down, through a whole word; `None` where they are
    /// none (the fill then reads nothing).
    pub fn indices_range(&self) -> Option<BufferRange> {
        self.indices.map(|(range, _)| range)
    }

    /// The words to upload: the header, then each draw's slot in turn.
    pub fn words(&self) -> Vec<u8> {
        let mut header = [0; Self::HEADER_WORDS as usize];
        header[0] = self.slots.len() as u32;
        header[4] = self.max_groups;
        header
            .iter()
            .chain(self.slots.iter().flatten())
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }
}

/// The first word of the slot of draw `d`.
fn slot_word(f: &mut FunctionBuilder<'_>, d: Value) -> BResult<Value> {
    let per = k(f, IndexBoundsRun::SLOT_WORDS)?;
    let header = k(f, IndexBoundsRun::HEADER_WORDS)?;
    let at = f.binary(BinOp::IMul, d, per)?;
    f.binary(BinOp::IAdd, header, at)
}

/// The plan of a run: each draw's plan in turn over its slot, each draw's first fill workgroup
/// the fill workgroups before it, then the fill's dispatch of all of them.
pub(super) fn plan_run(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    words: Value,
    sources: (Value, Value, Value),
) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let draws = load_word(f, words, 0)?;
    let total_var = f.local_var(u32_, None)?;
    let total = f.var_ptr(total_var)?;
    f.store(total, zero, MemAccess::default())?;
    for_range(f, u32_, (zero, draws, one), move |f, d| {
        let first = slot_word(f, d)?;
        let slot = Words {
            buffer: words,
            slot: Some(first),
        };
        plan_draw(f, slot, sources)?;
        let groups = slot.load(f, Output::FillDispatch as u32)?;
        let before = f.load(total, MemAccess::default())?;
        slot.store(f, FIRST_GROUP, before)?;
        let after = f.binary(BinOp::IAdd, before, groups)?;
        f.store(total, after, MemAccess::default())
    })?;
    // As few workgroups along y as hold them at most the widest along x, then as few along x
    // as they need: none of either where there are none.
    let total = f.load(total, MemAccess::default())?;
    let widest = load_word(f, words, 4)?;
    let rows = divided_up(f, total, widest)?;
    let some = f.binary(BinOp::ULessThan, zero, rows)?;
    let divisor = f.select(some, rows, one)?;
    let width = divided_up(f, total, divisor)?;
    for (at, v) in [(1, width), (2, rows), (3, one)] {
        let at = k(f, at)?;
        store_word(f, words, at, v)?;
    }
    Ok(())
}

/// `n / d` rounded up, for `d` above 0: without `n + d - 1`, which a `u32` may not hold.
fn divided_up(f: &mut FunctionBuilder<'_>, n: Value, d: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let whole = f.binary(BinOp::UDiv, n, d)?;
    let rest = f.binary(BinOp::UMod, n, d)?;
    let partial = f.binary(BinOp::INotEqual, rest, zero)?;
    let extra = f.select(partial, one, zero)?;
    f.binary(BinOp::IAdd, whole, extra)
}

/// The fill of a run at global invocation `(global, row)`: its workgroup's draw, the last whose
/// first fill workgroup is at or before it (the first's being 0), fills its word. Workgroup
/// `(x, row)` is the run's `row · width + x`, `width` the dispatch's workgroups along x.
pub(super) fn fill_run(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    words: Value,
    sources: (Value, Value),
    (global, row): (Value, Value),
) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let local = k(f, IndexBoundsFillModule::LOCAL_SIZE)?;
    let column = f.binary(BinOp::UDiv, global, local)?;
    let width = load_word(f, words, 1)?;
    let before = f.binary(BinOp::IMul, row, width)?;
    let group = f.binary(BinOp::IAdd, before, column)?;
    let lin = f.binary(BinOp::UMod, global, local)?;
    let draws = load_word(f, words, 0)?;
    // Each step halves `[lo, hi)`; 32 steps end every search a `u32` counts.
    let lo_var = f.local_var(u32_, None)?;
    let lo = f.var_ptr(lo_var)?;
    f.store(lo, zero, MemAccess::default())?;
    let hi_var = f.local_var(u32_, None)?;
    let hi = f.var_ptr(hi_var)?;
    f.store(hi, draws, MemAccess::default())?;
    let steps = k(f, 32)?;
    for_range(f, u32_, (zero, steps, one), move |f, _| {
        let l = f.load(lo, MemAccess::default())?;
        let h = f.load(hi, MemAccess::default())?;
        let span = f.binary(BinOp::ISub, h, l)?;
        let one = k(f, 1)?;
        let wide = f.binary(BinOp::UGreaterThan, span, one)?;
        when(f, wide, move |f| {
            let sum = f.binary(BinOp::IAdd, l, h)?;
            let one = k(f, 1)?;
            let mid = f.binary(BinOp::ShiftRightLogical, sum, one)?;
            let first = slot_word(f, mid)?;
            let at = k(f, FIRST_GROUP)?;
            let at = f.binary(BinOp::IAdd, first, at)?;
            let p = word_ptr(f, words, at)?;
            let first_group = f.load(p, MemAccess::default())?;
            let before = f.binary(BinOp::ULessThanEqual, first_group, group)?;
            let l2 = f.select(before, mid, l)?;
            let h2 = f.select(before, h, mid)?;
            f.store(lo, l2, MemAccess::default())?;
            f.store(hi, h2, MemAccess::default())
        })
    })?;
    let any = f.binary(BinOp::ULessThan, zero, draws)?;
    when(f, any, move |f| {
        let found = f.load(lo, MemAccess::default())?;
        let first = slot_word(f, found)?;
        let slot = Words {
            buffer: words,
            slot: Some(first),
        };
        let first_group = slot.load(f, FIRST_GROUP)?;
        let local = k(f, IndexBoundsFillModule::LOCAL_SIZE)?;
        let groups_in = f.binary(BinOp::ISub, group, first_group)?;
        let w = f.binary(BinOp::IMul, groups_in, local)?;
        let w = f.binary(BinOp::IAdd, w, lin)?;
        fill_draw(f, slot, sources, w)
    })
}

/// The index bounds plan module of a run ([`IndexBoundsRun`]) for `device`, as
/// [`index_bounds_plan_module`](super::index_bounds_plan_module)'s is made, with its bindings.
pub fn index_bounds_plan_run_module(device: &Device) -> Result<IndexBoundsPlanModule, Error> {
    let (spirv, requirements) = finished(
        device,
        plan_module_of(true)?,
        plan_binding_of,
        "the index bounds plan run",
    )?;
    Ok(IndexBoundsPlanModule {
        spirv,
        requirements,
    })
}

/// The index bounds fill module of a run ([`IndexBoundsRun`]) for `device`, as
/// [`index_bounds_fill_module`](super::index_bounds_fill_module)'s is made, with its bindings,
/// dispatched indirectly from the words' [`IndexBoundsRun::DISPATCH_OFFSET`].
pub fn index_bounds_fill_run_module(device: &Device) -> Result<IndexBoundsFillModule, Error> {
    let (spirv, requirements) = finished(
        device,
        fill_module_of(true)?,
        fill_binding_of,
        "the index bounds fill run",
    )?;
    Ok(IndexBoundsFillModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
