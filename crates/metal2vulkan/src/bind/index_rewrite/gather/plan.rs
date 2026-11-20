//! The index gather pass's plan of an indirect run (the parent module): one workgroup reads each
//! draw's Metal arguments (`MTLDrawIndexedPrimitivesIndirectArguments`: count, instances, first
//! index, base vertex, base instance; or, without indices, `MTLDrawPrimitivesIndirectArguments`:
//! count, instances, vertex start, base instance), and
//!
//! - takes each draw's list's indices from its count ([`RewriteProgram::capacity_terms`] in the
//!   table's header), none for a draw of no instance;
//! - reserves the lists, in the run's order, in the command buffer's index pool (the index bounds
//!   pass's, `index_bounds.rs`: its counter in the bound pool's first word, bytes from its index
//!   region's start, which lies [`IndexBoundsPass::POOL_HEADER_BYTES`] into the block the gather
//!   writes). A list the pool's room cannot hold is none: its draw draws nothing, and the plan sets
//!   [`StatusBits::INDEX_POOL_OVERFLOW`], which the consumer reports as the command buffer's
//!   error. Nothing is drawn short silently;
//! - writes each draw's record (its first workgroup, its list's indices and first word in the
//!   pool, its first index), the run's workgroups and the gather's `VkDispatchIndirectCommand`
//!   (at most [`IndexGatherModule::MAX_GROUPS`] workgroups, which take the run's in turn), and
//!   each draw's `VkDrawIndexedIndirectCommand` at its place's [`IndexGatherPlanModule::DRAW_BYTES`]
//!   of the draws' binding: its list's indices, Metal's instances, its first index in the pool's
//!   index region, its base vertex (Metal's base vertex, or its vertex start), Metal's base
//!   instance.
//!
//! **Bindings** (set 0, storage buffers): [`IndexGatherPlanModule::TABLE_BINDING`] (the run's
//! table), `ARGUMENTS_BINDING` ([`IndexGather::arguments_range`]), `POOL_BINDING` (the pool's
//! header: its counter), `STATUS_BINDING` (the `ExecutionStatus` buffer) and `DRAWS_BINDING`
//! (the draws' commands). Every read of the arguments is contained (a word past the bound range
//! reads 0).
//!
//! **Synchronization** (consumer): the arguments' writes and the table's upload → `SHADER_READ`
//! in `COMPUTE_SHADER` before the plan; plans and the pool's other users one after another (each
//! reads and writes its counter); the plan's writes → `SHADER_READ` in `COMPUTE_SHADER` and
//! `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` before the gather, and → `INDIRECT_COMMAND_READ` in
//! `DRAW_INDIRECT` before the draws.

use super::super::{
    barrier, contained, contained_store, div_ceil, finished, for_range, k, load_at, saturating_sub,
    store, when, word_ptr, words_buffer, Symbol, STATUS, WORDS,
};
use super::{IndexGather, IndexGatherModule};
use crate::bind::index_bounds::IndexBoundsPass;
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, IoDir, MemAccess, Scope,
    Semantics, StorageClass, Type, TypedModule,
};
use std::sync::Arc;

#[cfg(doc)]
use super::super::RewriteProgram;

/// The crate-owned plan module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexGatherPlanModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexGatherPlanModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The run's table ([`IndexGather::table`]), a storage buffer.
    pub const TABLE_BINDING: u32 = 0;
    /// The Metal arguments buffer, a storage buffer.
    pub const ARGUMENTS_BINDING: u32 = 1;
    /// The index pool's header (its counter), a storage buffer.
    pub const POOL_BINDING: u32 = 2;
    /// The `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 3;
    /// The draws' `VkDrawIndexedIndirectCommand`s, a storage buffer.
    pub const DRAWS_BINDING: u32 = 4;
    /// Invocations of the one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 128;
    /// The most draws of an indirect run.
    pub const MAX_DRAWS: u32 = 512;
}

/// The arguments' binding symbol ([`IndexGatherPlanModule::ARGUMENTS_BINDING`]).
pub(super) const ARGUMENTS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The pool's binding symbol ([`IndexGatherPlanModule::POOL_BINDING`]).
pub(super) const POOL: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The draws' binding symbol ([`IndexGatherPlanModule::DRAWS_BINDING`]).
pub(super) const DRAWS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 4,
};

/// The module's buffers and workgroup lists.
#[derive(Clone, Copy)]
struct Globals {
    table: Value,
    arguments: Value,
    draws: Value,
    /// Per draw: its list's indices.
    lists: Value,
    /// Per draw: its list's first word from the run's.
    firsts: Value,
    /// Per draw: its first workgroup.
    groups: Value,
    /// The run's first word in the pool's index region, at 0.
    base: Value,
}

/// Word `i` of record `draw`'s.
fn record_word(f: &mut FunctionBuilder<'_>, draw: Value, i: u32) -> BResult<Value> {
    let header = k(f, IndexGatherModule::HEADER_WORDS)?;
    let per = k(f, IndexGatherModule::RECORD_WORDS)?;
    let at = f.binary(BinOp::IMul, draw, per)?;
    let at = f.binary(BinOp::IAdd, header, at)?;
    let i = k(f, i)?;
    f.binary(BinOp::IAdd, at, i)
}

/// Draw `i`'s Metal arguments: its first word's place in the arguments, then five words
/// (contained; a draw without indices has four).
fn draw_arguments(f: &mut FunctionBuilder<'_>, g: Globals, i: Value) -> BResult<[Value; 5]> {
    let at = record_word(f, i, 7)?;
    let first = contained(f, g.table, at)?;
    let mut words = [first; 5];
    for (n, word) in words.iter_mut().enumerate() {
        let n = k(f, n as u32)?;
        let w = f.binary(BinOp::IAdd, first, n)?;
        *word = contained(f, g.arguments, w)?;
    }
    Ok(words)
}

/// Entry `i` of workgroup list `list`, by its pointer.
fn entry(f: &mut FunctionBuilder<'_>, list: Value, i: Value) -> BResult<Value> {
    f.access_chain(list, &[i])
}

/// The plan as a typed module, before serialization: the module [`index_gather_plan_module`]
/// serializes, which the CPU interpreter runs.
pub(crate) fn plan_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let most = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(IndexGatherPlanModule::MAX_DRAWS)),
    })?;
    let list_ty = b.ty(Type::Array {
        elem: u32_,
        len: most,
        stride: None,
    })?;
    let one_c = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(1),
    })?;
    let one_ty = b.ty(Type::Array {
        elem: u32_,
        len: one_c,
        stride: None,
    })?;
    let lists = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let firsts = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let groups = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let base = b.plain_global(StorageClass::Workgroup, one_ty, None)?;
    let table = words_buffer(&mut b, WORDS, u32_)?;
    let arguments = words_buffer(&mut b, ARGUMENTS, u32_)?;
    let pool = words_buffer(&mut b, POOL, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let draws = words_buffer(&mut b, DRAWS, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let g = Globals {
        table: f.module().global_value(table)?,
        arguments: f.module().global_value(arguments)?,
        draws: f.module().global_value(draws)?,
        lists: f.module().global_value(lists)?,
        firsts: f.module().global_value(firsts)?,
        groups: f.module().global_value(groups)?,
        base: f.module().global_value(base)?,
    };
    let pool = f.module().global_value(pool)?;
    let status = f.module().global_value(status)?;
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let zero = k(&mut f, 0)?;
    let one = k(&mut f, 1)?;
    let lanes = k(&mut f, IndexGatherPlanModule::LOCAL_SIZE)?;
    let draws_here = load_at(&mut f, g.table, 0)?;
    let most = k(&mut f, IndexGatherPlanModule::MAX_DRAWS)?;
    let n = f.glsl(GlslOp::UMin, &[draws_here, most])?;
    let sequential = load_at(&mut f, g.table, 1)?;
    let sequential = f.binary(BinOp::INotEqual, sequential, zero)?;
    let pool_bytes = load_at(&mut f, g.table, 3)?;
    let lead = load_at(&mut f, g.table, 7)?;
    let step = load_at(&mut f, g.table, 8)?;
    let step = f.glsl(GlslOp::UMax, &[step, one])?;
    let per = load_at(&mut f, g.table, 9)?;

    // Each draw's list's indices, `u32::MAX` past a `u32` (which no room holds); none for a draw
    // of no instance.
    for_range(&mut f, u32_, (lin, n, lanes), move |f, i| {
        let [count, instances, ..] = draw_arguments(f, g, i)?;
        let zero = k(f, 0)?;
        let past_lead = saturating_sub(f, count, lead)?;
        let units = f.binary(BinOp::UDiv, past_lead, step)?;
        let (indices, past) = checked_mul(f, units, per, zero)?;
        let max = k(f, u32::MAX)?;
        let indices = f.select(past, max, indices)?;
        let none = f.binary(BinOp::IEqual, instances, zero)?;
        let indices = f.select(none, zero, indices)?;
        let p = entry(f, g.lists, i)?;
        f.store(p, indices, MemAccess::default())
    })?;
    barrier(&mut f)?;

    // The first invocation places the lists in the pool's room, in order.
    let first = f.binary(BinOp::IEqual, lin, zero)?;
    when(&mut f, first, move |f| {
        place(f, g, (u32_, n, pool, status, pool_bytes))
    })?;
    barrier(&mut f)?;

    // Each draw's record and command.
    for_range(&mut f, u32_, (lin, n, lanes), move |f, i| {
        let [_, instances, third, fourth, fifth] = draw_arguments(f, g, i)?;
        let zero = k(f, 0)?;
        let start = f.select(sequential, zero, third)?;
        let base_vertex = f.select(sequential, third, fourth)?;
        let base_instance = f.select(sequential, fourth, fifth)?;
        let p = entry(f, g.lists, i)?;
        let indices = f.load(p, MemAccess::default())?;
        let p = entry(f, g.firsts, i)?;
        let from_run = f.load(p, MemAccess::default())?;
        let p = entry(f, g.groups, i)?;
        let first_group = f.load(p, MemAccess::default())?;
        let p = entry(f, g.base, zero)?;
        let base = f.load(p, MemAccess::default())?;
        let first_index = f.binary(BinOp::IAdd, base, from_run)?;
        let header = k(f, (IndexBoundsPass::POOL_HEADER_BYTES / 4) as u32)?;
        let first_word = f.binary(BinOp::IAdd, header, first_index)?;
        for (word, value) in [(0, first_group), (1, indices), (2, first_word), (6, start)] {
            let at = record_word(f, i, word)?;
            contained_store(f, g.table, at, value)?;
        }
        let five = k(f, 5)?;
        let at = f.binary(BinOp::IMul, i, five)?;
        for (n, value) in [indices, instances, first_index, base_vertex, base_instance]
            .into_iter()
            .enumerate()
        {
            let n = k(f, n as u32)?;
            let w = f.binary(BinOp::IAdd, at, n)?;
            contained_store(f, g.draws, w, value)?;
        }
        Ok(())
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexGatherPlanModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The first invocation's placing of the run's `n` lists in `pool`'s room of `pool_bytes`, in
/// order: each list's first word and first workgroup, a list the room cannot hold none (and the
/// overflow bit in `status`); the counter, the run's workgroups and the gather's dispatch.
fn place(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    (u32_, n, pool, status, pool_bytes): (crate::spirv::ids::TypeId, Value, Value, Value, Value),
) -> BResult<()> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let four = k(f, 4)?;
    let counter_ptr = word_ptr(f, pool, zero)?;
    let counter = f.load(counter_ptr, MemAccess::default())?;
    let used = f.glsl(GlslOp::UMin, &[counter, pool_bytes])?;
    let room = f.binary(BinOp::ISub, pool_bytes, used)?;
    let room = f.binary(BinOp::UDiv, room, four)?;
    let local = |f: &mut FunctionBuilder<'_>, init: Value| -> BResult<Value> {
        let var = f.local_var(u32_, None)?;
        let ptr = f.var_ptr(var)?;
        f.store(ptr, init, MemAccess::default())?;
        Ok(ptr)
    };
    let placed = local(f, zero)?;
    let workgroups = local(f, zero)?;
    let over = local(f, zero)?;
    for_range(f, u32_, (zero, n, one), move |f, i| {
        let p = entry(f, g.lists, i)?;
        let indices = f.load(p, MemAccess::default())?;
        let at = f.load(placed, MemAccess::default())?;
        let left = f.binary(BinOp::ISub, room, at)?;
        let fits = f.binary(BinOp::ULessThanEqual, indices, left)?;
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let indices = f.select(fits, indices, zero)?;
        f.store(p, indices, MemAccess::default())?;
        let missed = f.select(fits, zero, one)?;
        let was = f.load(over, MemAccess::default())?;
        let now = f.binary(BinOp::BitwiseOr, was, missed)?;
        f.store(over, now, MemAccess::default())?;
        let p = entry(f, g.firsts, i)?;
        f.store(p, at, MemAccess::default())?;
        let groups_at = f.load(workgroups, MemAccess::default())?;
        let p = entry(f, g.groups, i)?;
        f.store(p, groups_at, MemAccess::default())?;
        let next = f.binary(BinOp::IAdd, at, indices)?;
        f.store(placed, next, MemAccess::default())?;
        let lanes = k(f, IndexGatherModule::LOCAL_SIZE)?;
        let own = div_ceil(f, indices, lanes)?;
        let next = f.binary(BinOp::IAdd, groups_at, own)?;
        f.store(workgroups, next, MemAccess::default())
    })?;
    let placed = f.load(placed, MemAccess::default())?;
    let any = f.binary(BinOp::UGreaterThan, placed, zero)?;
    when(f, any, move |f| {
        let four = k(f, 4)?;
        let bytes = f.binary(BinOp::IMul, placed, four)?;
        let next = f.binary(BinOp::IAdd, used, bytes)?;
        f.store(counter_ptr, next, MemAccess::default())
    })?;
    let over = f.load(over, MemAccess::default())?;
    let over = f.binary(BinOp::INotEqual, over, zero)?;
    when(f, over, move |f| {
        let word = k(f, StatusWord::Status.index())?;
        let bit = k(f, StatusBits::INDEX_POOL_OVERFLOW.0)?;
        let p = word_ptr(f, status, word)?;
        f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
        Ok(())
    })?;
    let workgroups = f.load(workgroups, MemAccess::default())?;
    let most = k(f, IndexGatherModule::MAX_GROUPS)?;
    let dispatched = f.glsl(GlslOp::UMin, &[workgroups, most])?;
    let first = k(f, IndexGather::DISPATCH_OFFSET as u32 / 4)?;
    let two = k(f, 2)?;
    store(f, g.table, two, workgroups)?;
    for (n, value) in [dispatched, one, one].into_iter().enumerate() {
        let n = k(f, n as u32)?;
        let w = f.binary(BinOp::IAdd, first, n)?;
        store(f, g.table, w, value)?;
    }
    let base_words = f.binary(BinOp::UDiv, used, four)?;
    let p = entry(f, g.base, zero)?;
    f.store(p, base_words, MemAccess::default())?;
    Ok(())
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexGatherPlanModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(IndexGatherPlanModule::TABLE_BINDING),
        Symbol::Binding(s) if *s == ARGUMENTS => Some(IndexGatherPlanModule::ARGUMENTS_BINDING),
        Symbol::Binding(s) if *s == POOL => Some(IndexGatherPlanModule::POOL_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(IndexGatherPlanModule::STATUS_BINDING),
        Symbol::Binding(s) if *s == DRAWS => Some(IndexGatherPlanModule::DRAWS_BINDING),
        _ => None,
    }
}

/// The index gather plan module for `device`: built with the typed builder, verified, serialized,
/// and numbered with the fixed bindings of [`IndexGatherPlanModule`].
pub fn index_gather_plan_module(device: &Device) -> Result<IndexGatherPlanModule, Error> {
    let (spirv, requirements) = finished(
        device,
        plan_typed_module()?,
        binding_of,
        "the index gather plan",
    )?;
    Ok(IndexGatherPlanModule {
        spirv,
        requirements,
    })
}
