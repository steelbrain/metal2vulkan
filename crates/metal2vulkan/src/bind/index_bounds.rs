//! The index bounds pass of an indexed indirect draw.
//!
//! Without `robustBufferAccess2` every index a Vulkan draw fetches must lie inside the bound
//! index range (`VUID-VkDrawIndexedIndirectCommand-robustBufferAccess2-08798`), and an index
//! offset at or past the buffer's length may lie past the `VkBuffer`, where it cannot be bound.
//! Metal draws each position inside the buffer at its index and each position past it at index 0
//! plus the base vertex (measured from offset 0). An indexed indirect draw's counts are GPU memory,
//! so where nothing else asks for its arguments on the CPU (`DrawCommands::index_bounds_pass`) the
//! consumer may bound its fetch on the GPU instead of reading them: two dispatches before the
//! render pass instance, then two indirect draws in it where Metal has one.
//!
//! - **The plan** ([`index_bounds_plan_module`], one invocation) reads Metal's
//!   `MTLDrawIndexedPrimitivesIndirectArguments` and writes both draws' arguments into the draw's
//!   bounds words ([`IndexBoundsPass`]). A draw whose positions all lie inside the buffer's whole
//!   indices from its offset is the first draw's, over the Metal index buffer as it is bound; the
//!   second draws nothing. Any other draw of indices is the second's, over its reservation in the
//!   command buffer's index pool, and the first draws nothing. Exactly one of the two carries the
//!   primitives, so primitive assembly sees Metal's index stream. The plan reserves the second
//!   draw's bytes from the pool's counter (the pool's first word, bytes from its index region's
//!   start); a draw whose indices do not fit draws nothing, and the plan sets
//!   [`StatusBits::INDEX_POOL_OVERFLOW`], which the consumer reports as the command buffer's error.
//!   Nothing is drawn short silently.
//! - **The fill** ([`index_bounds_fill_module`], [`IndexBoundsFillModule::LOCAL_SIZE`] invocations
//!   per workgroup, dispatched indirectly with the plan's group count) writes the reservation,
//!   word by word: each position's index where it lies inside the buffer's whole indices, else 0.
//!   A word of 16-bit indices holds two positions of one draw (a reservation is a whole number of
//!   words); a half past the draw's count is 0.
//!
//! **The bounds words** (one `u32` array of [`IndexBoundsPass::BYTES`], a storage buffer to both
//! modules and the indirect source of the dispatch and both draws):
//!
//! | Words | Written by | Contents |
//! |---|---|---|
//! | 0 to 7 | the consumer | [`IndexBoundsPass::parameters`]: the arguments' first word in the bound arguments range, the index offset's byte in the bound indices range, the whole indices from the offset (saturated to `u32`), 1 for 32-bit indices, the pool's index bytes; then zeros |
//! | 8 to 12 | the plan | the first draw's `VkDrawIndexedIndirectCommand` ([`IndexBoundsPass::FIRST_DRAW_OFFSET`]) |
//! | 13 to 17 | the plan | the second draw's ([`IndexBoundsPass::SECOND_DRAW_OFFSET`]) |
//! | 18 to 20 | the plan | the fill's `VkDispatchIndirectCommand` ([`IndexBoundsPass::FILL_DISPATCH_OFFSET`]) |
//! | 21 to 23 | the plan | the fill's first pool word, first Metal position, and count |
//!
//! **The pool**: [`IndexBoundsPass::POOL_HEADER_BYTES`] of header (the counter in its first word,
//! 0 before the command buffer's first plan) and then its index bytes, which the second draw binds
//! as its index buffer from the header's end. Reservations are whole words. The plan reads and
//! writes only the header's counter, and the fill writes only index bytes, so a consumer may bind
//! the plan's pool and the fill's pool as two buffers: a header whose counter all of a command
//! buffer's plans share, and one whose bytes from its own header's end hold the indices (the
//! runtime binds a counter in host-written memory and a block of device memory).
//!
//! **Bindings** (set 0, storage buffers): the plan's [`IndexBoundsPlanModule::WORDS_BINDING`],
//! `ARGUMENTS_BINDING` ([`IndexBoundsPass::arguments_range`]), `POOL_BINDING` (the whole pool) and
//! `STATUS_BINDING` (the `ExecutionStatus` buffer); the fill's
//! [`IndexBoundsFillModule::WORDS_BINDING`], `INDICES_BINDING`
//! ([`IndexBoundsPass::indices_range`], or any 4 bytes where it is `None`: nothing reads them) and
//! `POOL_BINDING`. Every read of the arguments and indices is contained (an index past the bound
//! range reads 0), so a wrongly bound range can never make a pass fault.
//!
//! **Synchronization** (consumer): the arguments' and indices' writes → `SHADER_READ` in
//! `COMPUTE_SHADER`, and the upload of the parameters, before the plan; plans one after another
//! (each reads and writes the pool's counter; one invocation each, so a barrier between them is
//! all the counter needs); the plan's writes → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` and
//! `SHADER_READ` in `COMPUTE_SHADER` before the fill; the fill's and plan's writes →
//! `INDEX_READ` in `INDEX_INPUT` and `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` before the draws.

use std::sync::Arc;

use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::status::{StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, Scope,
    Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// One indexed indirect draw's index bounds pass: what the consumer binds and uploads for the
/// plan and the fill (the module docs say how).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexBoundsPass {
    /// The range of the Metal indirect buffer to bind at
    /// [`IndexBoundsPlanModule::ARGUMENTS_BINDING`]: from the arguments' offset aligned down to
    /// `minStorageBufferOffsetAlignment` (and to 4) through their 20 bytes.
    pub arguments_range: BufferRange,
    /// The range of the Metal index buffer to bind at [`IndexBoundsFillModule::INDICES_BINDING`]:
    /// from the index offset aligned down likewise through the whole indices inside the buffer,
    /// to a whole word (`word_range`); `None` when none lies inside it from the offset (the
    /// fill then reads nothing).
    pub indices_range: Option<BufferRange>,
    /// The bounds words' first [`IndexBoundsPass::PARAMETER_WORDS`] words, which the consumer
    /// uploads before the plan.
    pub parameters: [u32; IndexBoundsPass::PARAMETER_WORDS],
}

impl IndexBoundsPass {
    /// Bytes of the bounds words.
    pub const BYTES: u64 = 96;
    /// The consumer's parameter words at the bounds words' start.
    pub const PARAMETER_WORDS: usize = 8;
    /// The first draw's `VkDrawIndexedIndirectCommand` in the bounds words.
    pub const FIRST_DRAW_OFFSET: u64 = 32;
    /// The second draw's `VkDrawIndexedIndirectCommand` in the bounds words.
    pub const SECOND_DRAW_OFFSET: u64 = 52;
    /// The pool's header: its counter, then reserved bytes; its index bytes follow, where the
    /// second draw binds the pool as its index buffer.
    pub const POOL_HEADER_BYTES: u64 = 16;
    /// Bytes of Metal's `MTLDrawIndexedPrimitivesIndirectArguments`.
    const ARGUMENT_BYTES: u64 = 20;

    /// The pass of a draw whose arguments are at `arguments`, drawing indices of `index_size`
    /// bytes (2 or 4) from `index_offset` of an index buffer whose bytes end at `index_end` (its
    /// length, in the same terms as the offset; its `VkBuffer` holds the bytes to `index_end`
    /// rounded up to 4 there, as the padding does), with ranges bound at offsets aligned to
    /// `alignment` (`minStorageBufferOffsetAlignment`) over at most `max_range` bytes
    /// (`maxStorageBufferRange`), into a pool of `pool_bytes` index bytes (a multiple of 4).
    /// `None` when the arguments' offset is not a multiple of 4, the index size is neither 2 nor
    /// 4, the index offset is not a multiple of it, the pool's bytes are not a multiple of 4, or a
    /// range overflows or exceeds `max_range`.
    pub fn new(
        arguments: IndirectRef,
        index_size: u64,
        index_offset: u64,
        index_end: u64,
        alignment: u64,
        max_range: u32,
        pool_bytes: u32,
    ) -> Option<IndexBoundsPass> {
        if !arguments.offset.is_multiple_of(4)
            || !matches!(index_size, 2 | 4)
            || !index_offset.is_multiple_of(index_size)
            || !pool_bytes.is_multiple_of(4)
        {
            return None;
        }
        let aligned = |offset: u64, end: u64| word_range(offset, end, alignment, max_range);
        let arguments_end = arguments.offset.checked_add(Self::ARGUMENT_BYTES)?;
        let (arguments_range, arguments_residual) = aligned(arguments.offset, arguments_end)?;
        let inside = index_end.saturating_sub(index_offset) / index_size;
        let (indices_range, indices_residual) = if inside == 0 {
            (None, 0)
        } else {
            let end = index_offset.checked_add(inside.checked_mul(index_size)?)?;
            let (range, residual) = aligned(index_offset, end)?;
            (Some(range), residual)
        };
        let mut parameters = [0; Self::PARAMETER_WORDS];
        parameters[Parameter::ArgumentsWord as usize] =
            u32::try_from(arguments_residual / 4).ok()?;
        parameters[Parameter::IndicesByte as usize] = u32::try_from(indices_residual).ok()?;
        parameters[Parameter::Inside as usize] = u32::try_from(inside).unwrap_or(u32::MAX);
        parameters[Parameter::Wide as usize] = u32::from(index_size == 4);
        parameters[Parameter::PoolBytes as usize] = pool_bytes;
        Some(IndexBoundsPass {
            arguments_range,
            indices_range,
            parameters,
        })
    }

}

/// The consumer's parameter words, by index into the bounds words.
#[derive(Clone, Copy)]
enum Parameter {
    ArgumentsWord = 0,
    IndicesByte = 1,
    Inside = 2,
    Wide = 3,
    PoolBytes = 4,
}

/// The plan's words in the bounds words, by index.
#[derive(Clone, Copy)]
enum Output {
    FirstDraw = 8,
    SecondDraw = 13,
    FillDispatch = 18,
    FillPoolWord = 21,
    FillStart = 22,
    FillCount = 23,
}

/// The crate-owned plan module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexBoundsPlanModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexBoundsPlanModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's bounds words, a storage buffer.
    pub const WORDS_BINDING: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const ARGUMENTS_BINDING: u32 = 1;
    /// The command buffer's index pool, a storage buffer.
    pub const POOL_BINDING: u32 = 2;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 3;
}

/// The crate-owned fill module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexBoundsFillModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexBoundsFillModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's bounds words, a storage buffer.
    pub const WORDS_BINDING: u32 = 0;
    /// The Metal index buffer, a storage buffer.
    pub const INDICES_BINDING: u32 = 1;
    /// The command buffer's index pool, a storage buffer.
    pub const POOL_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: each writes one word of the reservation.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The bounds words' binding symbol in both typed modules.
pub const WORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The plan's arguments binding symbol ([`IndexBoundsPlanModule::ARGUMENTS_BINDING`]); the
/// fill's indices binding symbol ([`IndexBoundsFillModule::INDICES_BINDING`]).
pub const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The pool's binding symbol in both typed modules.
pub const POOL: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The status buffer's binding symbol in the plan ([`IndexBoundsPlanModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
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

fn k(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// Word `index` of the runtime array of `buffer`.
fn word_ptr(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    f.access_chain(buffer, &[zero, index])
}

fn load_word(f: &mut FunctionBuilder<'_>, buffer: Value, index: u32) -> BResult<Value> {
    let i = k(f, index)?;
    let p = word_ptr(f, buffer, i)?;
    f.load(p, MemAccess::default())
}

fn store_word(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value, v: Value) -> BResult<()> {
    let p = word_ptr(f, buffer, index)?;
    f.store(p, v, MemAccess::default())
}

/// A draw's bounds words: the words buffer, and where a run's slot starts in it (`None` for a
/// pass's own words, from word 0).
#[derive(Clone, Copy)]
struct Words {
    buffer: Value,
    slot: Option<Value>,
}

impl Words {
    /// The index into the buffer of the draw's word `index`.
    fn at(self, f: &mut FunctionBuilder<'_>, index: u32) -> BResult<Value> {
        let i = k(f, index)?;
        match self.slot {
            None => Ok(i),
            Some(first) => f.binary(BinOp::IAdd, first, i),
        }
    }

    /// The draw's word `index`.
    fn load(self, f: &mut FunctionBuilder<'_>, index: u32) -> BResult<Value> {
        let i = self.at(f, index)?;
        let p = word_ptr(f, self.buffer, i)?;
        f.load(p, MemAccess::default())
    }

    /// Store `v` as the draw's word `index`.
    fn store(self, f: &mut FunctionBuilder<'_>, index: u32, v: Value) -> BResult<()> {
        let i = self.at(f, index)?;
        store_word(f, self.buffer, i, v)
    }
}

/// The range of a buffer that a pass binds as a storage buffer of words to read the bytes from
/// `offset` to `end` (in the `VkBuffer`'s offsets), and `offset`'s byte in it: from `offset`
/// aligned down to `alignment` (`minStorageBufferOffsetAlignment`, and to 4) through `end`
/// rounded up to a whole word, which the `VkBuffer` holds (the padding). A range of no whole words
/// would leave its last bytes out of every word the pass reads (`OpArrayLength` rounds down), so
/// the last index of an odd count of 16-bit indices from a multiple of 4, or of an even count from
/// 2 past one, would read 0; the bytes past `end` in the last word are never taken as an index.
/// `None` where the range overflows or exceeds `max_range` (`maxStorageBufferRange`).
pub(crate) fn word_range(
    offset: u64,
    end: u64,
    alignment: u64,
    max_range: u32,
) -> Option<(BufferRange, u64)> {
    let a = alignment.max(4);
    let start = offset - offset % a;
    let range = end.checked_next_multiple_of(4)?.checked_sub(start)?;
    if range > u64::from(max_range) {
        return None;
    }
    Some((
        BufferRange {
            offset: start,
            range,
        },
        offset - start,
    ))
}

/// Word `index` of `buffer` where it lies inside the buffer's bound range, else 0 (contained).
fn contained_word(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    let len = f.array_length(buffer, 0)?;
    let inb = f.binary(BinOp::ULessThan, index, len)?;
    let safe = f.select(inb, index, zero)?;
    let p = word_ptr(f, buffer, safe)?;
    let loaded = f.load(p, MemAccess::default())?;
    f.select(inb, loaded, zero)
}

/// The plan as a typed module, before serialization: the module [`index_bounds_plan_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols [`WORDS`], [`SOURCE`]
/// (the arguments), [`POOL`], and [`STATUS`].
pub fn plan_typed_module() -> BResult<TypedModule> {
    plan_module_of(false)
}

/// The plan of a run of draws (`run`) or of one draw, as a typed module: one invocation, which
/// plans a run's draws one after another, each from its slot (`run::IndexBoundsRun`).
pub(crate) fn plan_module_of(run: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let arguments = words_buffer(&mut b, SOURCE, u32_)?;
    let pool = words_buffer(&mut b, POOL, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let words = f.module().global_value(words)?;
    let arguments = f.module().global_value(arguments)?;
    let pool = f.module().global_value(pool)?;
    let status = f.module().global_value(status)?;
    let sources = (arguments, pool, status);
    match run {
        false => plan_draw(
            &mut f,
            Words {
                buffer: words,
                slot: None,
            },
            sources,
        )?,
        true => run::plan_run(&mut f, u32_, words, sources)?,
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The plan of the draw whose bounds words are `words`, over the arguments, pool and status.
fn plan_draw(
    f: &mut FunctionBuilder<'_>,
    words: Words,
    (arguments, pool, status): (Value, Value, Value),
) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let two = k(f, 2)?;
    let three = k(f, 3)?;
    let four = k(f, 4)?;

    let first = words.load(f, Parameter::ArgumentsWord as u32)?;
    let inside = words.load(f, Parameter::Inside as u32)?;
    let wide = words.load(f, Parameter::Wide as u32)?;
    let pool_bytes = words.load(f, Parameter::PoolBytes as u32)?;
    // Metal's arguments: indexCount, instanceCount, indexStart, baseVertex, baseInstance.
    let mut argument = Vec::with_capacity(5);
    for i in 0..5 {
        let i = k(f, i)?;
        let index = f.binary(BinOp::IAdd, first, i)?;
        argument.push(contained_word(f, arguments, index)?);
    }
    let (count, instances, start, base, base_instance) = (
        argument[0],
        argument[1],
        argument[2],
        argument[3],
        argument[4],
    );

    // Inside: every position from `start` lies among the whole indices from the offset.
    let fits_count = f.binary(BinOp::ULessThanEqual, count, inside)?;
    let room_inside = f.binary(BinOp::ISub, inside, count)?;
    let starts_inside = f.binary(BinOp::ULessThanEqual, start, room_inside)?;
    let in_range = f.binary(BinOp::LogicalAnd, fits_count, starts_inside)?;
    let not_in_range = f.unary(UnOp::LogicalNot, in_range)?;
    let any = f.binary(BinOp::UGreaterThan, count, zero)?;
    let over = f.binary(BinOp::LogicalAnd, not_in_range, any)?;

    // The reservation, in bytes from the pool's index region (the counter never passes its size).
    let is_wide = f.binary(BinOp::INotEqual, wide, zero)?;
    let size = f.select(is_wide, four, two)?;
    let counter_index = zero;
    let counter_ptr = word_ptr(f, pool, counter_index)?;
    let reserved = f.load(counter_ptr, MemAccess::default())?;
    let room = f.binary(BinOp::ISub, pool_bytes, reserved)?;
    let room_indices = f.binary(BinOp::UDiv, room, size)?;
    let fits_pool = f.binary(BinOp::ULessThanEqual, count, room_indices)?;
    let place = f.binary(BinOp::LogicalAnd, over, fits_pool)?;
    let not_fits = f.unary(UnOp::LogicalNot, fits_pool)?;
    let overflow = f.binary(BinOp::LogicalAnd, over, not_fits)?;
    // Whole words: `count * size` rounded up to 4 (no overflow where it fits the pool).
    let index_bytes = f.binary(BinOp::IMul, count, size)?;
    let padded = f.binary(BinOp::IAdd, index_bytes, three)?;
    let word_count = f.binary(BinOp::ShiftRightLogical, padded, two)?;
    let taken = f.binary(BinOp::ShiftLeftLogical, word_count, two)?;
    let advanced = f.binary(BinOp::IAdd, reserved, taken)?;
    let counter = f.select(place, advanced, reserved)?;
    f.store(counter_ptr, counter, MemAccess::default())?;

    // The first draw: Metal's where every position lies inside, else none.
    let first_count = f.select(in_range, count, zero)?;
    let first_start = f.select(in_range, start, zero)?;
    // The second: the reservation where it was placed, else none.
    let second_count = f.select(place, count, zero)?;
    let reserved_index = f.binary(BinOp::UDiv, reserved, size)?;
    let second_start = f.select(place, reserved_index, zero)?;
    for (at, draw_count, draw_start) in [
        (Output::FirstDraw as u32, first_count, first_start),
        (Output::SecondDraw as u32, second_count, second_start),
    ] {
        for (i, v) in [draw_count, instances, draw_start, base, base_instance]
            .into_iter()
            .enumerate()
        {
            words.store(f, at + i as u32, v)?;
        }
    }
    // The fill: a workgroup per LOCAL_SIZE words of the reservation.
    let fill_words = f.select(place, word_count, zero)?;
    let local = k(f, IndexBoundsFillModule::LOCAL_SIZE - 1)?;
    let rounded = f.binary(BinOp::IAdd, fill_words, local)?;
    let local_size = k(f, IndexBoundsFillModule::LOCAL_SIZE)?;
    let groups = f.binary(BinOp::UDiv, rounded, local_size)?;
    let reserved_word = f.binary(BinOp::ShiftRightLogical, reserved, two)?;
    let header_words = k(f, (IndexBoundsPass::POOL_HEADER_BYTES / 4) as u32)?;
    let pool_word = f.binary(BinOp::IAdd, header_words, reserved_word)?;
    for (at, v) in [
        (Output::FillDispatch as u32, groups),
        (Output::FillDispatch as u32 + 1, one),
        (Output::FillDispatch as u32 + 2, one),
        (Output::FillPoolWord as u32, pool_word),
        (Output::FillStart as u32, start),
        (Output::FillCount as u32, second_count),
    ] {
        words.store(f, at, v)?;
    }
    // A draw of indices the pool cannot hold draws nothing and says so.
    let word = k(f, StatusWord::Status.index())?;
    let bit = k(f, StatusBits::INDEX_POOL_OVERFLOW.0)?;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let p = word_ptr(fb, status, word)?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), overflow, then, None)?;
    Ok(())
}

/// The fill as a typed module, before serialization: the module [`index_bounds_fill_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols [`WORDS`], [`SOURCE`]
/// (the indices), and [`POOL`].
pub fn fill_typed_module() -> BResult<TypedModule> {
    fill_module_of(false)
}

/// The fill of a run of draws (`run`) or of one draw, as a typed module: a run's workgroup fills
/// its part of the draw whose fill groups hold it (`run::IndexBoundsRun`).
pub(crate) fn fill_module_of(run: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let indices = words_buffer(&mut b, SOURCE, u32_)?;
    let pool = words_buffer(&mut b, POOL, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let words = f.module().global_value(words)?;
    let indices = f.module().global_value(indices)?;
    let pool = f.module().global_value(pool)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let w = f.composite_extract(invocation, &[0])?;
    let row = f.composite_extract(invocation, &[1])?;
    match run {
        false => fill_draw(
            &mut f,
            Words {
                buffer: words,
                slot: None,
            },
            (indices, pool),
            w,
        )?,
        true => run::fill_run(&mut f, u32_, words, (indices, pool), (w, row))?,
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexBoundsFillModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fill of word `w` of the reservation of the draw whose bounds words are `words`, from the
/// indices into the pool.
fn fill_draw(
    f: &mut FunctionBuilder<'_>,
    words: Words,
    (indices, pool): (Value, Value),
    w: Value,
) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let sixteen = k(f, 16)?;

    let first_byte = words.load(f, Parameter::IndicesByte as u32)?;
    let inside = words.load(f, Parameter::Inside as u32)?;
    let wide = words.load(f, Parameter::Wide as u32)?;
    let pool_word = words.load(f, Output::FillPoolWord as u32)?;
    let start = words.load(f, Output::FillStart as u32)?;
    let count = words.load(f, Output::FillCount as u32)?;
    let is_wide = f.binary(BinOp::INotEqual, wide, zero)?;
    // Words of the reservation: one per 32-bit index, one per two 16-bit ones.
    let plus_one = f.binary(BinOp::IAdd, count, one)?;
    let halves = f.binary(BinOp::ShiftRightLogical, plus_one, one)?;
    let word_count = f.select(is_wide, count, halves)?;
    let listed = f.binary(BinOp::ULessThan, w, word_count)?;

    // The index at Metal position `p`, where it lies inside the whole indices (and `p` did not
    // wrap past `start`), else 0; 4 bytes each where `wide`, else 2.
    let fetch = move |f: &mut FunctionBuilder<'_>, p: Value, wide: bool| -> BResult<Value> {
        let zero = k(f, 0)?;
        let unwrapped = f.binary(BinOp::UGreaterThanEqual, p, start)?;
        let below = f.binary(BinOp::ULessThan, p, inside)?;
        let valid = f.binary(BinOp::LogicalAnd, unwrapped, below)?;
        let size_shift = k(f, if wide { 2 } else { 1 })?;
        let offset = f.binary(BinOp::ShiftLeftLogical, p, size_shift)?;
        let byte = f.binary(BinOp::IAdd, first_byte, offset)?;
        let two = k(f, 2)?;
        let word = f.binary(BinOp::ShiftRightLogical, byte, two)?;
        let safe = f.select(valid, word, zero)?;
        let loaded = contained_word(f, indices, safe)?;
        let value = if wide {
            loaded
        } else {
            let three = k(f, 3)?;
            let half = f.binary(BinOp::BitwiseAnd, byte, two)?;
            let shift = f.binary(BinOp::ShiftLeftLogical, half, three)?;
            let shifted = f.binary(BinOp::ShiftRightLogical, loaded, shift)?;
            let mask = k(f, 0xFFFF)?;
            f.binary(BinOp::BitwiseAnd, shifted, mask)?
        };
        f.select(valid, value, zero)
    };
    let fill: ArmFn<'_, ()> = Box::new(move |f, _| {
        // 32-bit: position start + w.
        let p32 = f.binary(BinOp::IAdd, start, w)?;
        let v32 = fetch(f, p32, true)?;
        // 16-bit: positions start + 2w and the next, the second past the count 0.
        let twice = f.binary(BinOp::ShiftLeftLogical, w, one)?;
        let lo_p = f.binary(BinOp::IAdd, start, twice)?;
        let hi_p = f.binary(BinOp::IAdd, lo_p, one)?;
        let lo = fetch(f, lo_p, false)?;
        let hi = fetch(f, hi_p, false)?;
        let hi_position = f.binary(BinOp::IAdd, twice, one)?;
        let counted = f.binary(BinOp::ULessThan, hi_position, count)?;
        let hi = f.select(counted, hi, zero)?;
        let high = f.binary(BinOp::ShiftLeftLogical, hi, sixteen)?;
        let v16 = f.binary(BinOp::BitwiseOr, lo, high)?;
        let v = f.select(is_wide, v32, v16)?;
        let at = f.binary(BinOp::IAdd, pool_word, w)?;
        let len = f.array_length(pool, 0)?;
        let inb = f.binary(BinOp::ULessThan, at, len)?;
        let write: ArmFn<'_, ()> = Box::new(move |f, _| {
            store_word(f, pool, at, v)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), inb, write, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), listed, fill, None)?;
    Ok(())
}

/// The fixed binding numbers of the plan's symbols.
fn plan_binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexBoundsPlanModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(IndexBoundsPlanModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == SOURCE => Some(IndexBoundsPlanModule::ARGUMENTS_BINDING),
        Symbol::Binding(s) if *s == POOL => Some(IndexBoundsPlanModule::POOL_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(IndexBoundsPlanModule::STATUS_BINDING),
        _ => None,
    }
}

/// The fixed binding numbers of the fill's symbols.
fn fill_binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexBoundsFillModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(IndexBoundsFillModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == SOURCE => Some(IndexBoundsFillModule::INDICES_BINDING),
        Symbol::Binding(s) if *s == POOL => Some(IndexBoundsFillModule::POOL_BINDING),
        _ => None,
    }
}

/// `m` checked by the owned verifier against `device`'s profile (requirements included),
/// serialized, and numbered with `binding_of`; `what` names it in an internal error.
pub(crate) fn finished(
    device: &Device,
    m: TypedModule,
    binding_of: fn(&Symbol) -> Option<u32>,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("{what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The index bounds plan module of `device`: built with the typed builder, verified, serialized,
/// and numbered with the fixed bindings of [`IndexBoundsPlanModule`].
pub fn index_bounds_plan_module(device: &Device) -> Result<IndexBoundsPlanModule, Error> {
    let (spirv, requirements) = finished(
        device,
        plan_typed_module()?,
        plan_binding_of,
        "the index bounds plan",
    )?;
    Ok(IndexBoundsPlanModule {
        spirv,
        requirements,
    })
}

/// The index bounds fill module of `device`: built with the typed builder, verified, serialized,
/// and numbered with the fixed bindings of [`IndexBoundsFillModule`].
pub fn index_bounds_fill_module(device: &Device) -> Result<IndexBoundsFillModule, Error> {
    let (spirv, requirements) = finished(
        device,
        fill_typed_module()?,
        fill_binding_of,
        "the index bounds fill",
    )?;
    Ok(IndexBoundsFillModule {
        spirv,
        requirements,
    })
}

mod run;
pub use run::{
    index_bounds_fill_run_module, index_bounds_plan_run_module, BoundsRunDraw, IndexBoundsRun,
};

#[cfg(test)]
mod tests;
