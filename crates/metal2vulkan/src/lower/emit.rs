//! `lower::Emitter`: the `CodeEmitter` of a `Plan`, and the module driver that runs the
//! structurizer and emission.
//!
//! [`emit_module`] declares the module-scope objects the plan selected (the slots' views, the push
//! block in placement order, the kernel's workgroup size), structurizes the entry's CFG through
//! [`EntryView`], and lets P8 (`structurize::emit`) walk the tree with an [`Emitter`], which emits
//! each block by routing every instruction's [`Strategy`] to its family's emit function. It then
//! lets `fragment::wrap_entry` choose the entry function (the walked body itself, or a raster order
//! group's interlocked wrapper that calls it), declares the entry point, finalizes, runs P9
//! (`promote_single_store`), removes what no instruction reads, taking the lowering's offers
//! that make the module smaller (`spirv::dead::remove_dead_keeping`), and, under the 16-bit float
//! controls, holds the `half` values of a module that computes `half` in 16 bits there
//! (`spirv::half_hold`).
//!
//! The emitter owns its value environment: the SPIR-V value of each AIR value in the block being
//! emitted, reset at the top of every block, and the values the entry hook (`stage_io::emit_entry`;
//! `dispatch::emit_entry`) emits in the function's first block before the walk and holds for every
//! block, which that block dominates; the push fields the plan places are loaded there too. A value
//! used outside its block is demoted (the plan lists it); its block stores it into its variable
//! after the definition, and each other block that uses it loads it once at its top. A phi's
//! incoming value counts as a use at the end of its incoming block, where it is read. Each phi's
//! block loads the phi's incoming variable at its top, and each predecessor writes its incoming
//! value there as a parallel copy before its terminator.
//!
//! The lowered types and constants are the families' (`arith::spirv_type`,
//! `arith::materialize`, `access::spirv_type`), and so are parameter values (each family's
//! `param_value`, routed by `select::param_family`), the entry's return (`stage_io::emit_return`
//! stores outputs, `fragment::emit_return` ends the function), and module-scope objects and
//! execution modes (each family's `declare`, kept in [`FamilyDecls`]), so this driver never grows
//! with them.
//!
//! Emission cannot fail except with a `BuildError`. A plan/emission mismatch (an instruction
//! without a strategy, an operand without a value) is recorded as an internal invariant error in
//! the emitter and surfaced by [`emit_module`] in place of the builder error that aborted the walk.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::air::ir::{Body, InstId, InstKind, Value, ValueId, ValueIndex};
use crate::air::Entry;
use crate::budget::{tree_bytes, vec_bytes, Growth, Meter};
use crate::error::{Error, InternalError, InternalKind, LowerFamily};
use crate::lower::select::{entry_param_family, param_binding, FamilyPlans, Plan, SelectorForm};
use crate::lower::strategy::Strategy;
use crate::lower::view::EntryView;
use crate::lower::{
    access, arith, dispatch, fragment, mesh, raytracing, sampler, simd, stage_io, sync,
    tessellation, texture,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, SlotRef, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::dead::{remove_dead_keeping, Offer};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{BlockId as SBlockId, DemoteTy, GlobalId, TypeId, Value as SValue, VarId};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, MemAccess, PushMember, Type,
    TypedModule, ValueOrigin,
};
use crate::structurize::emit::{CodeEmitter, Poll, StructVars};
use crate::structurize::promote::promote_single_store;

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Each family's module-scope objects, declared by its `declare` hook.
#[derive(Clone, Debug, Default)]
pub struct FamilyDecls {
    pub arith: arith::Declared,
    pub access: access::Declared,
    pub texture: texture::Declared,
    pub sampler: sampler::Declared,
    pub simd: simd::Declared,
    pub barrier: sync::barrier::Declared,
    pub memory_model: sync::memory_model::Declared,
    pub fragment: fragment::Declared,
    pub stage_io: stage_io::Declared,
    pub dispatch: dispatch::Declared,
    pub tessellation: tessellation::Declared,
    pub raytracing: raytracing::Declared,
    pub mesh: mesh::Declared,
}

/// The module-scope objects the plan's strategies use.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The push members, in placement order.
    pub push: Vec<PushMember>,
    pub push_global: Option<GlobalId>,
    pub families: FamilyDecls,
    /// The entry's execution modes the families added.
    pub modes: Vec<ExecMode>,
}

/// Declare the plan's module-scope objects: the access family's views, the push block, then
/// every other family's objects.
fn declare(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<Declared> {
    // Before any instruction: every `half` rounding's quantize the device's form.
    mb.set_quantize_form(plan.families.arith.quantize);
    mb.set_native_half(plan.families.arith.native_half);
    mb.set_native_half_nans(plan.families.arith.native_half_nans);
    mb.set_half_controls(plan.families.arith.controls);
    let mut d = Declared::default();
    let modes = &mut d.modes;
    d.families.access = access::declare(mb, plan, modes)?;
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            continue;
        };
        let ty = match crate::lower::push::shape(f.field) {
            crate::lower::push::FieldShape::U32 => u32_,
            // Interned only when placed: a 64-bit type requires `shaderInt64`.
            crate::lower::push::FieldShape::U64 => mb.ty(Type::Int {
                width: 64,
                signed: false,
            })?,
            crate::lower::push::FieldShape::UVec2 => mb.ty(Type::Vector { elem: u32_, len: 2 })?,
            crate::lower::push::FieldShape::UVec4Array(n) => {
                let v4 = mb.ty(Type::Vector { elem: u32_, len: 4 })?;
                let len = mb.constant(Constant::Scalar {
                    ty: u32_,
                    bits: ConstBits(u64::from(n)),
                })?;
                mb.ty(Type::Array {
                    elem: v4,
                    len,
                    stride: Some(16),
                })?
            }
        };
        d.push.push(mb.push_member(f.field, ty, offset)?);
    }
    // The floor's tail word, above every placed field (`PushPlacement::tail_word`).
    if let Some(at) = plan.push.tail_word {
        d.push.push(mb.push_member(PushField::TailWord, u32_, at)?);
    }
    if !d.push.is_empty() {
        d.push_global = Some(mb.push_global()?);
    }
    let modes = &mut d.modes;
    let f = &mut d.families;
    f.dispatch = dispatch::declare(mb, plan, modes)?;
    f.stage_io = stage_io::declare(mb, plan, modes)?;
    f.tessellation = tessellation::declare(mb, plan, modes)?;
    f.fragment = fragment::declare(mb, plan, modes)?;
    f.texture = texture::declare(mb, plan, modes)?;
    f.sampler = sampler::declare(mb, plan, modes)?;
    f.raytracing = raytracing::declare(mb, plan, modes)?;
    f.mesh = mesh::declare(mb, plan, modes)?;
    f.simd = simd::declare(mb, plan, modes)?;
    f.barrier = sync::barrier::declare(mb, plan, modes)?;
    f.memory_model = sync::memory_model::declare(mb, plan, modes)?;
    f.arith = arith::declare(mb, plan, modes)?;
    Ok(d)
}

/// What family emit functions read and write.
pub struct EmitCx<'a> {
    pub entry: &'a Entry,
    pub body: &'a Body,
    pub plan: &'a Plan,
    pub decl: &'a Declared,
    /// The SPIR-V value of each AIR value in the block being emitted, by [`ValueId`], and the
    /// values [`EmitCx::hold`] binds for the whole function.
    env: Vec<Option<SValue>>,
    /// The values bound in `env` since the block opened, so opening the next block clears them.
    bound: Vec<ValueId>,
    /// Invocation-invariant values emitted since the block opened, in the function's entry block
    /// or in the block the entry hooks end in, each with the SPIR-V block it was emitted in
    /// ([`Invariant`]).
    memo: Vec<(Invariant, SValue, SBlockId)>,
    /// The SPIR-V blocks the open block's top-level code ran through, in order: a value emitted
    /// in one of them dominates everything the block emits after it.
    spine: Vec<SBlockId>,
    /// The function's entry block: a value emitted there dominates every block, so its memo
    /// entries outlive the block that emitted them.
    entry_block: Option<SBlockId>,
    /// The block the entry hooks end in, where the walk of the body starts: it dominates every
    /// block of the body, so its memo entries outlive the block that emitted them too. It is the
    /// entry block but where a hook closed a construct there: a kernel's tail guard
    /// (`EmitCx::emit_tail_guard`) or a dispatch past its grid (`dispatch::emit_entry`) returns
    /// from an arm, and the hooks after it, the sink's address
    /// among them (`access::dynamic::emit_entry`), run in its merge.
    hooks_block: Option<SBlockId>,
    /// While a workgroup synchronization of a predicated walk is emitted, the calling invocation's
    /// live flag: a lane that is not live takes part with null operands. `None`: every invocation
    /// executing the synchronization is live.
    pub live: Option<SValue>,
    /// With `live`, in a divergent block: the lane takes part in the exchange as active only
    /// where `live` holds. In a workgroup-uniform block every invocation is live or none is, so
    /// every lane takes part as active (a result nobody is live for is never stored).
    pub active: Option<SValue>,
    /// The flag a predicated walk starts `live` from, computed in the function's entry: whether
    /// the invocation is one of its partial object threadgroup's threads (`dispatch::emit_entry`).
    /// `None`: every invocation runs the body.
    pub(crate) entry_live: Option<SValue>,
    /// The memory-plan slots a buffer store emitted so far merges a partial word into, through
    /// the sub-word store's two atomics, which read the word they modify. Whether a store merges
    /// depends on the leaf the lowering chose and on the units the slot's views hold on the
    /// profile, so emission records it as it emits the store (`memory::leaf::store`), and the
    /// artifact reports a read of each such slot ([`Emitted::merged`]).
    pub(crate) merged: BTreeSet<usize>,
    fault: Option<Error>,
    /// The sampling routines the function's samples call, each defined at its first call
    /// (`sampler::routine`).
    pub(crate) sampling: crate::lower::sampler::routine::Routines,
    /// A merged draw's vertex function's command, vertex, instance and slot table, which
    /// `stage_io::emit_draw_entry` finds before the body and before any other hook reads its slot
    /// table (`stage_io::MergedRead`): defined at entry, where they dominate every later block of
    /// the function, which may read them.
    pub(crate) merged_draw: Option<crate::bind::MergedVertex>,
    /// A merged draw's fragment function's command's fragment slot table, which
    /// `stage_io::emit_draw_entry` finds before the body and before any other hook reads it
    /// (`stage_io::StageIoPlan::merged_fragment`), defined in the function's straight-line entry.
    pub(crate) merged_fragment_table: Option<SValue>,
}

/// A value that is the same wherever an invocation computes it, so a block computes it once, and
/// the function once when its entry block computes it: a push field (the push block and the
/// `ArgumentTable` are read-only), which the entry hook loads in the entry block
/// (`EmitCx::load_push_fields`). A dispatch parameter is derived once at entry
/// (`dispatch::emit_entry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Invariant {
    Push(PushField),
    /// A fragment function's tile's first word in the pass's tile memory buffer
    /// (`access::tile_memory::base`), computed at entry (`access::tile_memory::emit_entry`).
    TileBase,
    /// A kernel's slot of the runtime's private arena, placed as a pointer to its words
    /// ([`EmitCx::arena_slot`]), in the function's first block.
    ArenaSlot,
    /// The table word of the argument buffer's sampler of a handle, tested for a cubic filter
    /// (`sampler::emit`'s reach test): an operation through the same handle where an earlier test
    /// dominates it ran that test on the same sampler, so it needs none of its own.
    CubicTested(ValueId),
    /// An indexed capture form's raw index (`stage_io::read_captured_index`), read at entry
    /// (`stage_io::emit_draw_entry`): the index buffer is read-only to the draw.
    CapturedIndex,
    /// The private list form's base vertex (`stage_io::read_listed_base`), read at entry
    /// (`stage_io::emit_draw_entry`): the list is read-only to the draw.
    ListedBase,
    /// The root table's sink address (`access::dynamic::read_sink_host`), read at entry
    /// (`access::dynamic::emit_entry`): the root table is read-only to the dispatch.
    SinkHost,
    /// A checked slot's `PushField::SlotBounds` split ([`EmitCx::slot_bounds`]): its buffer's
    /// bytes before the bind offset (`false`) or its length (`true`).
    Bounds(SlotRef, bool),
    /// A checked slot's limit for accesses of a size in bytes ([`EmitCx::slot_limit`]).
    Limit(SlotRef, u32),
    /// A 16-byte line of the function's parameter block ([`EmitCx::block_line`]).
    BlockLine(u32),
}

impl<'a> EmitCx<'a> {
    /// A context for a new function of the module (an entry wrapper): the same entry, plan, and
    /// declarations, with no value bound and nothing remembered, so no value of another function is
    /// recalled.
    pub(crate) fn for_new_function(&self) -> EmitCx<'a> {
        EmitCx {
            entry: self.entry,
            body: self.body,
            plan: self.plan,
            decl: self.decl,
            env: Vec::new(),
            bound: Vec::new(),
            memo: Vec::new(),
            spine: Vec::new(),
            entry_block: None,
            hooks_block: None,
            live: None,
            active: None,
            entry_live: None,
            merged: BTreeSet::new(),
            fault: None,
            sampling: Default::default(),
            merged_draw: None,
            merged_fragment_table: None,
        }
    }

    /// Take over what the emission of a routine in `routine`'s context recorded: its fault, which
    /// aborts this function's walk as its own would, and the buffer slots its stores merged.
    pub(crate) fn adopt(&mut self, routine: EmitCx<'_>) {
        if self.fault.is_none() {
            self.fault = routine.fault;
        }
        self.merged.extend(routine.merged);
    }

    /// Record a plan/emission mismatch and abort the walk with a builder error.
    pub fn fail(&mut self, detail: impl Into<String>) -> BuildError {
        if self.fault.is_none() {
            self.fault = Some(invariant(detail));
        }
        BuildError::Dangling(Dangling::Value(SValue::new(u32::MAX)))
    }

    /// The value of `key` emitted earlier where it dominates the open SPIR-V block: in that
    /// block, in a block of the open block's spine, in the function's entry block, or in the
    /// block the entry hooks end in.
    fn recall(&self, fb: &FunctionBuilder<'_>, key: Invariant) -> Option<SValue> {
        let here = fb.current_block()?;
        self.memo
            .iter()
            .find(|(k, _, _)| *k == key)
            .filter(|(_, _, at)| *at == here || self.dominates_body(*at) || self.spine.contains(at))
            .map(|(_, v, _)| *v)
    }

    /// Whether block `at` dominates every block of the body: the function's entry block, or the
    /// block the entry hooks end in.
    fn dominates_body(&self, at: SBlockId) -> bool {
        Some(at) == self.entry_block || Some(at) == self.hooks_block
    }

    /// Record the function's entry block before the entry hooks run: a hook that runs after one
    /// closed a construct (the floor's tail guard) recalls what the first block holds, a push
    /// field [`EmitCx::load_push_fields`] loaded there, rather than loading it again where the
    /// hooks stand.
    fn begin_hooks(&mut self, fb: &FunctionBuilder<'_>) -> BResult<()> {
        self.entry_block = fb.entry_block();
        Ok(())
    }

    /// Record the open block as the one the entry hooks end in, where the walk starts.
    fn end_hooks(&mut self, fb: &FunctionBuilder<'_>) -> BResult<()> {
        self.hooks_block = fb.current_block();
        Ok(())
    }

    /// Record `v` as the value of `key` in the open SPIR-V block (replacing an entry that does
    /// not dominate it).
    fn remember(&mut self, fb: &FunctionBuilder<'_>, key: Invariant, v: SValue) {
        let Some(here) = fb.current_block() else {
            return;
        };
        match self.memo.iter_mut().find(|(k, _, _)| *k == key) {
            Some(e) => *e = (key, v, here),
            None => self.memo.push((key, v, here)),
        }
    }

    /// Record `v`, made in the function's entry block, as the value of `key` there, so every
    /// block recalls it.
    fn remember_at_entry(&mut self, fb: &FunctionBuilder<'_>, key: Invariant, v: SValue) {
        let Some(entry) = fb.entry_block() else {
            return;
        };
        match self.memo.iter_mut().find(|(k, _, _)| *k == key) {
            Some(e) => *e = (key, v, entry),
            None => self.memo.push((key, v, entry)),
        }
    }

    /// Record the open SPIR-V block as part of the spine: the block's top-level code runs there.
    fn on_spine(&mut self, fb: &FunctionBuilder<'_>) {
        if let Some(here) = fb.current_block() {
            if self.spine.last() != Some(&here) {
                self.spine.push(here);
            }
        }
    }

    /// Push field `field` in the open block, from the push block or the `ArgumentTable`, loaded
    /// once per block where it dominates; [`EmitCx::load_push_fields`] loads every placed field in
    /// the function's first block, which dominates them all.
    pub fn push_field(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        field: PushField,
    ) -> BResult<SValue> {
        if let Some(v) = self.recall(fb, Invariant::Push(field)) {
            return Ok(v);
        }
        let v = self.load_push_field(fb, field)?;
        self.remember(fb, Invariant::Push(field), v);
        Ok(v)
    }

    /// Whether the cubic reach test of the argument buffer's sampler of `handle` ran where it
    /// dominates the open block ([`EmitCx::cubic_tested`] recorded it there).
    pub(crate) fn cubic_test_dominates(&self, fb: &FunctionBuilder<'_>, handle: ValueId) -> bool {
        self.recall(fb, Invariant::CubicTested(handle)).is_some()
    }

    /// Record that the cubic reach test of the argument buffer's sampler of `handle`, which read
    /// `word`, ran in the open block.
    pub(crate) fn cubic_tested(&mut self, fb: &FunctionBuilder<'_>, handle: ValueId, word: SValue) {
        self.remember(fb, Invariant::CubicTested(handle), word);
    }

    /// The first word of a fragment function's tile memory in the pass's buffer, the tile that
    /// `tile` places (`access::tile_memory::base`), computed once where it dominates:
    /// `access::tile_memory::emit_entry` computes it in the function's first block.
    pub(crate) fn tile_base(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        tile: crate::memory::arena::FragmentTile,
    ) -> BResult<SValue> {
        if let Some(v) = self.recall(fb, Invariant::TileBase) {
            return Ok(v);
        }
        let v = crate::lower::access::tile_memory::base(self, fb, tile)?;
        self.remember(fb, Invariant::TileBase, v);
        Ok(v)
    }

    /// The invocation's slot of the runtime's memory that holds the kernel's private arena, placed:
    /// a `PhysicalStorageBuffer` pointer to its words, a runtime array of `u32` at the slot's
    /// device address (`dispatch::KernelDecl::arena_slot`, of the pushed
    /// `PushField::PrivateArenaAddress` and the slot's bytes, `RuntimeFact::PrivateArena`), so
    /// an access of a word is an access chain from it (`memory::arena::ArenaMem::ptr`). Made once
    /// in the function's first block, wherever it is first asked for: its operands are that push
    /// field, which [`EmitCx::load_push_fields`] loads there, the invocation's built-ins and
    /// constants, so it dominates every access of the arena. Made where it is asked for, it would
    /// be made again in every block of the body: `dispatch::emit_entry` asks after the floor's
    /// tail guard closed a construct, and later hooks may close more before the walk.
    pub(crate) fn arena_slot(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        if let Some(v) = self.recall(fb, Invariant::ArenaSlot) {
            return Ok(v);
        }
        let slot_bytes = self.plan.facts.iter().find_map(|f| match *f {
            RuntimeFact::PrivateArena { slot_bytes } => Some(slot_bytes),
            _ => None,
        });
        let Some(slot_bytes) = slot_bytes else {
            return Err(self.fail("a private arena in device memory without its slot's bytes"));
        };
        let decl: &Declared = self.decl;
        let Some(kernel) = decl.families.dispatch.kernel.as_ref() else {
            return Err(self.fail("a private arena in device memory without its kernel"));
        };
        let v = fb.in_entry_block(|fb| {
            let arena = self.push_field(fb, PushField::PrivateArenaAddress)?;
            let slot = kernel.arena_slot(fb, arena, slot_bytes)?;
            let u32_ = fb.module().ty(Type::Int {
                width: 32,
                signed: false,
            })?;
            let words = fb.module().ty(Type::RuntimeArray {
                elem: u32_,
                stride: Some(4),
            })?;
            fb.device_place(slot, words)
        })?;
        self.remember_at_entry(fb, Invariant::ArenaSlot, v);
        Ok(v)
    }

    /// An indexed capture form's raw index, read once where it dominates:
    /// `stage_io::emit_draw_entry` reads it in the function's first block, which dominates every block of the function, the
    /// merge of the restart test it then closes included.
    pub(crate) fn captured_index(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let first = fb.entry_block();
        let read = self
            .memo
            .iter()
            .find(|(k, _, at)| *k == Invariant::CapturedIndex && Some(*at) == first);
        if let Some(&(_, v, _)) = read {
            return Ok(v);
        }
        if let Some(v) = self.recall(fb, Invariant::CapturedIndex) {
            return Ok(v);
        }
        let v = crate::lower::stage_io::read_captured_index(self, fb)?;
        self.remember(fb, Invariant::CapturedIndex, v);
        Ok(v)
    }

    /// The private list form's base vertex, read once where it dominates, as
    /// [`EmitCx::captured_index`] reads the index.
    pub(crate) fn listed_base(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let first = fb.entry_block();
        let read = self
            .memo
            .iter()
            .find(|(k, _, at)| *k == Invariant::ListedBase && Some(*at) == first);
        if let Some(&(_, v, _)) = read {
            return Ok(v);
        }
        if let Some(v) = self.recall(fb, Invariant::ListedBase) {
            return Ok(v);
        }
        let v = crate::lower::stage_io::read_listed_base(self, fb)?;
        self.remember(fb, Invariant::ListedBase, v);
        Ok(v)
    }

    /// The root table's sink address, read once where it dominates: `access::dynamic::emit_entry`
    /// reads it in the block the entry hooks end in, so no wide access reads the table for it.
    pub(crate) fn sink_host(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        if let Some(v) = self.recall(fb, Invariant::SinkHost) {
            return Ok(v);
        }
        let v = access::dynamic::read_sink_host(self, fb)?;
        self.remember(fb, Invariant::SinkHost, v);
        Ok(v)
    }

    /// The work before the body: load each push field the plan places once, in the function's first
    /// block, where it dominates every block. Where the body first reads a field never moves its
    /// load, so a read that canonicalization deletes cannot make the others load it again; a placed
    /// field no instruction reads leaves with the other unused values (`spirv::dead`).
    ///
    /// A kernel loads its fields in placement order, then returns where its tail holds what the
    /// consumer never pushes ([`EmitCx::emit_tail_guard`]). A render stage loads the fields read
    /// from their push members first, then folds the tail into them ([`EmitCx::fold_tail`]), then
    /// the spilled fields and those the function derives, which may read through the folded values
    /// (the parameter block's address, a slot table's).
    fn load_push_fields(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        if self.plan.stage == Stage::Kernel {
            self.load_fields(fb, |_, _| true)?;
            return self.emit_tail_guard(fb);
        }
        let member = |s: &Self, field: PushField| {
            !s.derives(field) && s.decl.push.iter().any(|m| m.field == field)
        };
        self.load_fields(fb, member)?;
        self.fold_tail(fb)?;
        self.load_fields(fb, |s, field| !member(s, field))
    }

    /// Load each field the plan places that `which` takes, in placement order.
    fn load_fields(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        which: impl Fn(&Self, PushField) -> bool,
    ) -> BResult<()> {
        let plan: &Plan = self.plan;
        for placed in &plan.push.fields {
            if !which(self, placed.field) {
                continue;
            }
            match placed.field {
                PushField::SlotBounds(slot) => {
                    self.slot_bounds(fb, slot)?;
                }
                field => {
                    self.push_field(fb, field)?;
                }
            }
        }
        Ok(())
    }

    /// Where the plan's placement ends with a tail (`lower::push::tail`): read it from the push
    /// block, and test whether it holds the value the consumer never pushes there
    /// ([`crate::lower::push::tail_never_pushed`]); `None` where there is no tail or no such
    /// value. A tail field the function reads from its member is the load
    /// [`EmitCx::load_push_fields`] made in this block, so the member is loaded once; the tail
    /// word, and a field the function derives elsewhere, are loaded here.
    fn tail_test(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<Option<(PushField, SValue)>> {
        let Some(field) = crate::lower::push::tail(&self.plan.push) else {
            return Ok(None);
        };
        let member = (self.decl.push.iter()).find(|m| m.field == field);
        let (Some(member), Some(push)) = (member, self.decl.push_global) else {
            return Err(self.fail("a placed tail the push block does not hold"));
        };
        let value = match field != PushField::TailWord && !self.derives(field) {
            true => self.push_field(fb, field)?,
            false => Self::load_member(fb, push, member.index)?,
        };
        let never = crate::lower::push::tail_never_pushed(fb, value, field)?;
        Ok(never.map(|never| (field, never)))
    }

    /// A kernel's tail ([`EmitCx::tail_test`]): return from the function where it holds the value
    /// the consumer never pushes there ([`crate::lower::push::return_never_pushed`]). The branch
    /// is never taken; everything the function does after it depends on the tail's read. A return
    /// at entry is a kernel's own shape: its entry also returns invocations before its body where
    /// its dispatch passes its grid (`dispatch::emit_entry`).
    fn emit_tail_guard(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        match self.tail_test(fb)? {
            Some((_, never)) => crate::lower::push::return_never_pushed(fb, never),
            None => Ok(()),
        }
    }

    /// A render stage's tail ([`EmitCx::tail_test`]): make each other field read from its push
    /// member, as the function's every use recalls it, the member's value where the test fails and
    /// the null of its type where it holds ([`crate::lower::push::fold_never_pushed`]): the tail's
    /// read lives while any other push read does, and no branch makes the body depend on it
    /// (`lower::push`: a fragment stage has no other return at its entry). A tail field of the
    /// function's own folds into the others alone: its own uses need no select to read it.
    fn fold_tail(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        let Some((field, never)) = self.tail_test(fb)? else {
            return Ok(());
        };
        let plan: &Plan = self.plan;
        for placed in &plan.push.fields {
            let other = placed.field;
            if other == field
                || self.derives(other)
                || !self.decl.push.iter().any(|m| m.field == other)
            {
                continue;
            }
            let keys = match other {
                PushField::SlotBounds(slot) => vec![
                    Invariant::Push(other),
                    Invariant::Bounds(slot, false),
                    Invariant::Bounds(slot, true),
                ],
                _ => vec![Invariant::Push(other)],
            };
            for key in keys {
                let Some(v) = self.recall(fb, key) else {
                    return Err(self.fail("a pushed field not loaded before the tail folds"));
                };
                let folded = crate::lower::push::fold_never_pushed(fb, never, v)?;
                self.remember(fb, key, folded);
            }
        }
        Ok(())
    }

    /// Checked slot `slot`'s bounds (`PushField::SlotBounds`), split: its buffer's bytes before
    /// the bind offset and its length, each a `u32`, where they dominate the open block.
    /// [`EmitCx::load_push_fields`] splits them in the function's first block, so a checked
    /// access adds no extract of its own.
    pub(crate) fn slot_bounds(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        slot: SlotRef,
    ) -> BResult<(SValue, SValue)> {
        let (before, length) = (
            Invariant::Bounds(slot, false),
            Invariant::Bounds(slot, true),
        );
        if let (Some(b), Some(l)) = (self.recall(fb, before), self.recall(fb, length)) {
            return Ok((b, l));
        }
        let bounds = self.push_field(fb, PushField::SlotBounds(slot))?;
        let b = fb.composite_extract(bounds, &[0])?;
        let l = fb.composite_extract(bounds, &[1])?;
        self.remember(fb, before, b);
        self.remember(fb, length, l);
        Ok((b, l))
    }

    /// Checked slot `slot`'s buffer bytes before the bind offset and its limit for `n`-byte
    /// accesses (`psb::limit`), where they dominate the open block: the pair an access checks
    /// with (`psb::in_buffer`). The limit is made once per slot and size, in the function's first
    /// block where the bounds are split there ([`EmitCx::load_push_fields`]), so every checked
    /// access of that size anywhere in the function adds one add and one compare.
    pub(crate) fn slot_limit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        slot: SlotRef,
        n: u32,
    ) -> BResult<(SValue, SValue)> {
        let (before, length) = self.slot_bounds(fb, slot)?;
        let key = Invariant::Limit(slot, n);
        if let Some(limit) = self.recall(fb, key) {
            return Ok((before, limit));
        }
        let entry = fb.entry_block();
        let at_entry = matches!(fb.origin(length)?, ValueOrigin::Inst(b) if Some(b) == entry);
        let limit = match at_entry && fb.current_block() != entry {
            true => {
                let limit = fb.in_entry_block(|fb| access::psb::limit(fb, length, n))?;
                self.remember_at_entry(fb, key, limit);
                limit
            }
            false => {
                let limit = access::psb::limit(fb, length, n)?;
                self.remember(fb, key, limit);
                limit
            }
        };
        Ok((before, limit))
    }

    /// Line `line` of the function's parameter block at device address `block`
    /// (`access::table::read_line`), read once where it dominates the open block:
    /// [`EmitCx::load_push_fields`] reads every spilled field in the function's first block, so
    /// the fields that share a line share one 16-byte load.
    pub(crate) fn block_line(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        block: SValue,
        line: u32,
    ) -> BResult<SValue> {
        let key = Invariant::BlockLine(line);
        if let Some(v) = self.recall(fb, key) {
            return Ok(v);
        }
        let v = access::table::read_line(fb, block, line)?;
        self.remember(fb, key, v);
        Ok(v)
    }

    /// Whether the function derives push field `field` from something other than its own
    /// member or `ArgumentTable` entry ([`EmitCx::load_push_field`]). A device-replay variant
    /// reads a slot's Metal address from its root entry, and its dispatch region from the region
    /// record of its command and region; a flat variant reads its grid from its dispatch's region
    /// record, and a folded variant its region and group counts; a merged draw's functions read
    /// their command's slot tables.
    fn derives(&self, field: PushField) -> bool {
        let flat = self
            .plan
            .dispatch
            .as_ref()
            .is_some_and(|d| d.flat_grid || d.folded_groups);
        let families = &self.plan.families;
        match field {
            PushField::RootTableAddress => {
                families.stage_io.merged.is_some() || families.stage_io.merged_fragment.is_some()
            }
            PushField::SlotGuestBase(_) => families.access.replay,
            PushField::DispatchRegion => families.access.replay || flat,
            _ => false,
        }
    }

    /// Load push field `field` in the open block.
    fn load_push_field(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        field: PushField,
    ) -> BResult<SValue> {
        if self.derives(field) {
            return match field {
                // A merged draw's vertex function reads its command's slot table.
                PushField::RootTableAddress if self.plan.families.stage_io.merged.is_some() => {
                    match self.merged_draw {
                        Some(m) => Ok(m.table),
                        None => Err(self.fail("a merged draw's slot table read before its entry")),
                    }
                }
                // A merged draw's fragment function reads its command's fragment slot table.
                PushField::RootTableAddress => self.merged_fragment_table.ok_or_else(|| {
                    self.fail("a merged draw's fragment slot table read before its entry")
                }),
                PushField::SlotGuestBase(slot) => access::replay_guest_base(self, fb, slot),
                PushField::DispatchRegion => access::replay_region(self, fb),
                _ => Err(self.fail("a derived push field without its source")),
            };
        }
        let member = self.decl.push.iter().find(|m| m.field == field).copied();
        let (Some(member), Some(g)) = (member, self.decl.push_global) else {
            // A field the push block does not hold is in the stage's `ArgumentTable`, which the
            // access family declares and reads.
            return access::table_field(self, fb, field);
        };
        Self::load_member(fb, g, member.index)
    }

    /// Load member `index` of push block `push` in the open block.
    fn load_member(fb: &mut FunctionBuilder<'_>, push: GlobalId, index: u32) -> BResult<SValue> {
        let base = fb.module().global_value(push)?;
        let index = fb.u32_const(index)?;
        let index = fb.module().const_value(index)?;
        let ptr = fb.access_chain(base, &[index])?;
        fb.load(ptr, MemAccess::default())
    }

    /// The SPIR-V value of AIR value `v` in the open block: a value bound in the block, a
    /// parameter's value from its family, or a constant.
    pub fn operand(&mut self, fb: &mut FunctionBuilder<'_>, v: ValueId) -> BResult<SValue> {
        if let Some(Some(x)) = self.env.get(v.index()) {
            return Ok(*x);
        }
        let val = self.plan.values.value(v);
        let entry = self.entry;
        if let Some(b) = val.and_then(|val| param_binding(entry, val)) {
            return match entry_param_family(entry, self.plan.stage, &b.role) {
                Ok(LowerFamily::Access) => access::param_value(self, fb, b),
                Ok(LowerFamily::Texture) => texture::param_value(self, fb, b),
                Ok(LowerFamily::Sampler) => sampler::param_value(self, fb, b),
                // `dispatch::emit_entry` holds every dispatch parameter the body reads.
                Ok(LowerFamily::Dispatch) => Err(self.fail(format!(
                    "dispatch parameter {} was not derived at entry",
                    b.param
                ))),
                Ok(LowerFamily::StageIo) => stage_io::param_value(self, fb, b),
                Ok(LowerFamily::Fragment) => fragment::param_value(self, fb, b),
                Ok(LowerFamily::Tessellation) => tessellation::param_value(self, fb, b),
                Ok(LowerFamily::RayTracing) => raytracing::param_value(self, fb, b),
                Ok(LowerFamily::Simd) => simd::param_value(self, fb, b),
                _ => Err(self.fail(format!("parameter {} has no family value", b.param))),
            };
        }
        // A removed texture or sampler the body still reads.
        if let Some(b) = val.and_then(|val| access::handle::absent_param(entry, val)) {
            return access::handle::absent_param_value(self, fb, b);
        }
        let wide = &self.plan.families.arith;
        let Some(k) = val.and_then(|val| arith::constant_in(self.entry, wide, val)) else {
            // A null buffer pointer constant is the null fat pointer.
            if let Some(x) = access::pointer::null_operand(self, fb, v)? {
                return Ok(x);
            }
            return Err(self.fail(format!("{v:?} has no value in this block")));
        };
        match arith::materialize(fb, &k)? {
            Some(x) => Ok(x),
            None => Err(self.fail(format!("{v:?} is a constant arith does not materialize"))),
        }
    }

    /// Bind AIR value `v` to `x` for the whole function: `x` was emitted by the entry hook
    /// (`stage_io::emit_entry`, `access::dynamic::emit_entry`) before the walk, in a block that
    /// dominates every block of the body (the function's first, or a merged vertex's search's
    /// merge, `stage_io::emit_draw_entry`), so opening a block does not forget it.
    pub fn hold(&mut self, v: ValueId, x: SValue) {
        if let Some(slot) = self.env.get_mut(v.index()) {
            *slot = Some(x);
        }
    }

    /// Bind instruction result `v` to `x` in the open block.
    pub fn define(&mut self, v: ValueId, x: SValue) {
        if let Some(slot) = self.env.get_mut(v.index()) {
            if slot.replace(x).is_none() {
                self.bound.push(v);
            }
        }
    }

    /// Forget every value of the previous block: a value another block defined is read only
    /// through its demotion variable, so a missed load is an invariant error, never a stale id.
    fn open_block(&mut self) {
        self.unwind((0, 0));
    }

    /// The bound values and spine blocks so far, for [`EmitCx::unwind`].
    fn mark(&self) -> (usize, usize) {
        (self.bound.len(), self.spine.len())
    }

    /// Forget the values bound and the spine blocks entered since `mark`, and the invariant
    /// values emitted off the remaining spine: a construct was closed, and what its arms defined
    /// does not dominate what follows it.
    fn unwind(&mut self, (bound, spine): (usize, usize)) {
        let later: Vec<ValueId> = self.bound.drain(bound.min(self.bound.len())..).collect();
        for v in later {
            if let Some(slot) = self.env.get_mut(v.index()) {
                *slot = None;
            }
        }
        self.spine.truncate(spine);
        let (entry, hooks) = (self.entry_block, self.hooks_block);
        let kept = &self.spine;
        self.memo
            .retain(|(_, _, at)| Some(*at) == entry || Some(*at) == hooks || kept.contains(at));
    }
}

/// The phis of each block, and the demoted values each block reads from other blocks, by block.
type BlockValues = BTreeMap<u32, Vec<ValueId>>;

/// The phis of each block and the block-crossing values each block imports (loads at its top),
/// and the peak heap bytes of the sets that computed them:
/// every demoted value an instruction of the block uses, and every demoted incoming value of a
/// successor's phi on the edge from the block (the parallel copy reads it there), except values
/// the block defines itself. Each import is listed once per block. What an instruction or a copy
/// reads is `FamilyPlans::reads` and `FamilyPlans::incoming_reads` (an index extension's narrow
/// operand; the value a forwarding `extractvalue` forwards; a member-wise phi's members): the hooks
/// `select` computes the block-crossing values with.
fn block_values(
    body: &Body,
    values: ValueIndex,
    cross: &[(ValueId, DemoteTy)],
    families: &FamilyPlans,
) -> (BlockValues, BlockValues, usize) {
    // ---- number, not sorted maps; the scratch charged is still the maps' ----
    // Whether each value crosses blocks, and the block defining each that does (`u32::MAX`: none).
    let mut crossing = vec![false; values.len()];
    let mut crossing_count = 0usize;
    for (v, _) in cross {
        if let Some(c) = crossing.get_mut(v.index()) {
            crossing_count += usize::from(!*c);
            *c = true;
        }
    }
    let crosses = |v: &ValueId| crossing.get(v.index()).copied().unwrap_or(false);
    let mut home = vec![u32::MAX; values.len()];
    let mut homed = 0usize;
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            if let Some(v) = values.inst(id).filter(crosses) {
                if let Some(h) = home.get_mut(v.index()) {
                    homed += usize::from(*h == u32::MAX);
                    *h = b.raw();
                }
            }
        }
    }
    let mut phis = BlockValues::new();
    let mut imports = BlockValues::new();
    let mut seen: BTreeSet<(u32, ValueId)> = BTreeSet::new();
    let mut import = |at: u32, v: Value| {
        let Some(id) = values.id(v).filter(crosses) else {
            return;
        };
        if home.get(id.index()).copied() != Some(at) && seen.insert((at, id)) {
            imports.entry(at).or_default().push(id);
        }
    };
    let index = families.read_index(body.insts.len());
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            if let InstKind::Phi { incoming, .. } = &inst.kind {
                if let Some(v) = values.inst(id) {
                    phis.entry(b.raw()).or_default().push(v);
                }
                for (v, from) in incoming.iter() {
                    families
                        .incoming_reads(body, values, (id, *v, *from), |r| import(from.raw(), r));
                }
                continue;
            }
            families.reads_indexed(&index, body, id, inst, |r| import(b.raw(), r));
        }
    }
    // The sets that built the lists are gone; their peak is charged with the emitter.
    let scratch = tree_bytes(crossing_count, std::mem::size_of::<ValueId>())
        + tree_bytes(homed, std::mem::size_of::<(ValueId, u32)>())
        + tree_bytes(seen.len(), std::mem::size_of::<(u32, ValueId)>());
    (phis, imports, scratch)
}

/// The heap bytes of per-block lists.
fn lists_bytes(lists: &BlockValues) -> usize {
    tree_bytes(lists.len(), std::mem::size_of::<(u32, Vec<ValueId>)>())
        + lists.values().map(vec_bytes).sum::<usize>()
}

/// The `CodeEmitter` of a plan.
pub struct Emitter<'a> {
    cx: EmitCx<'a>,
    /// Shared with each block's emission rather than copied into it: the variables grow with the
    /// function, so a copy per block would make the walk's work quadratic in its size.
    vars: Option<Rc<StructVars>>,
    /// The phis of each block, and the AIR values each block uses from other blocks.
    phis: BlockValues,
    imports: BlockValues,
    /// Demoted values whose emitted value is a plain constant: a block importing one after its
    /// definition was emitted uses the constant instead of loading it.
    constants: BTreeMap<ValueId, SValue>,
    /// The heap bytes of `phis` and `imports`, which do not change after [`Emitter::new`].
    lists: usize,
    /// The peak heap bytes of the transient sets [`Emitter::new`] computed the lists with.
    scratch: usize,
}

impl<'a> Emitter<'a> {
    fn new(entry: &'a Entry, body: &'a Body, plan: &'a Plan, decl: &'a Declared) -> Emitter<'a> {
        let (phis, imports, scratch) = block_values(body, plan.values, &plan.cross, &plan.families);
        let lists = lists_bytes(&phis) + lists_bytes(&imports);
        Emitter {
            cx: EmitCx {
                entry,
                body,
                plan,
                decl,
                env: vec![None; plan.values.len()],
                bound: Vec::new(),
                memo: Vec::new(),
                spine: Vec::new(),
                entry_block: None,
                hooks_block: None,
                live: None,
                active: None,
                entry_live: None,
                merged: BTreeSet::new(),
                fault: None,
                sampling: Default::default(),
                merged_draw: None,
                merged_fragment_table: None,
            },
            vars: None,
            phis,
            imports,
            constants: BTreeMap::new(),
            lists,
            scratch,
        }
    }

    /// The heap bytes the emitter holds, for the byte ledger: the value environment, the per-block
    /// phi and import lists, the demoted constants, and the open block's lists.
    fn footprint(&self) -> usize {
        vec_bytes(&self.cx.env)
            + vec_bytes(&self.cx.bound)
            + vec_bytes(&self.cx.memo)
            + vec_bytes(&self.cx.spine)
            + self.lists
            + tree_bytes(
                self.constants.len(),
                std::mem::size_of::<(ValueId, SValue)>(),
            )
    }

    fn vars(&mut self) -> BResult<Rc<StructVars>> {
        match &self.vars {
            Some(v) => Ok(Rc::clone(v)),
            None => Err(self
                .cx
                .fail("a block emitted before the variables were bound")),
        }
    }

    fn load_var(fb: &mut FunctionBuilder<'_>, var: VarId) -> BResult<SValue> {
        let p = fb.var_ptr(var)?;
        fb.load(p, MemAccess::default())
    }

    fn store_var(fb: &mut FunctionBuilder<'_>, var: VarId, x: SValue) -> BResult<()> {
        let p = fb.var_ptr(var)?;
        fb.store(p, x, MemAccess::default())
    }

    /// Emit instruction `id` (value `v`) and store its demoted variable after it.
    fn inst(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        id: InstId,
        v: ValueId,
        vars: &StructVars,
    ) -> BResult<()> {
        self.emit_inst(fb, id, v)?;
        self.cx.on_spine(fb);
        if let Some(var) = vars.demoted_var(v) {
            let x = self.cx.operand(fb, v)?;
            Self::store_var(fb, var, x)?;
            if fb.is_plain_const(x)? {
                self.constants.insert(v, x);
            }
        }
        Ok(())
    }

    /// Load block `b`'s phis from their incoming variables and copy each demoted one to its
    /// demoted variable. `copies_only`: bind nothing (a predicated segment reloads them where
    /// used).
    fn phi_loads(
        &mut self,
        b: crate::air::ir::BlockId,
        fb: &mut FunctionBuilder<'_>,
        vars: &StructVars,
        copies_only: bool,
    ) -> BResult<()> {
        for v in self.phis.get(&b.raw()).cloned().unwrap_or_default() {
            let Some(var) = vars.phi_var(v) else {
                return Err(self.cx.fail(format!("phi {v:?} has no variable")));
            };
            let dv = vars.demoted_var(v);
            if copies_only && dv.is_none() {
                continue;
            }
            let x = Self::load_var(fb, var)?;
            if !copies_only {
                self.cx.define(v, x);
            }
            if let Some(dv) = dv {
                Self::store_var(fb, dv, x)?;
            }
        }
        Ok(())
    }

    /// The parallel copy into the successors' phi variables at the end of block `b`: read every
    /// value first, once per variable (a successor reached by several edges takes one copy).
    /// `reload`: a value not bound here is reloaded from its variable (a predicated segment).
    fn copies(
        &mut self,
        b: crate::air::ir::BlockId,
        fb: &mut FunctionBuilder<'_>,
        vars: &StructVars,
        reload: bool,
    ) -> BResult<()> {
        let body = self.cx.body;
        let values = self.cx.plan.values;
        let mut succs: Vec<crate::air::ir::BlockId> = body.successors(b).collect();
        succs.sort_unstable();
        succs.dedup();
        let mut copies: BTreeMap<VarId, SValue> = BTreeMap::new();
        for succ in succs {
            for (sid, inst) in body.block_insts(succ) {
                let InstKind::Phi { incoming, .. } = &inst.kind else {
                    continue;
                };
                let Some(phi) = values.inst(sid) else {
                    return Err(self.cx.fail(format!("{sid:?} is out of range")));
                };
                let Some(var) = vars.phi_var(phi) else {
                    return Err(self.cx.fail(format!("phi {phi:?} has no variable")));
                };
                let Some((val, _)) = incoming.iter().find(|(_, from)| *from == b) else {
                    continue;
                };
                if reload {
                    let mut reads: Vec<ValueId> = Vec::new();
                    self.cx
                        .plan
                        .families
                        .incoming_reads(body, values, (sid, *val, b), |r| {
                            if let (Value::Inst(_), Some(id)) = (r, values.id(r)) {
                                reads.push(id);
                            }
                        });
                    for r in reads {
                        self.ensure(fb, r)?;
                    }
                }
                // A member-wise phi's value is assembled from its incoming aggregate's members.
                if arith::aggregate::incoming(&self.cx.plan.families.arith, sid, b).is_some() {
                    let Some(x) = arith::aggregate::emit_incoming(&mut self.cx, fb, sid, b)? else {
                        return Err(self.cx.fail("a member-wise phi without its assembly"));
                    };
                    copies.insert(var, x);
                    continue;
                }
                let Some(src) = values.id(*val) else {
                    return Err(self.cx.fail("a phi operand out of range"));
                };
                // A null pointer constant takes the phi's representation.
                let x = access::pointer::phi_incoming(&mut self.cx, fb, phi, src)?;
                copies.insert(var, x);
            }
        }
        for (var, x) in copies {
            Self::store_var(fb, var, x)?;
        }
        Ok(())
    }

    /// Bind value `v` in the open block of a predicated walk if it is not bound: from the
    /// demoted constants, its demoted variable, or its phi's incoming variable. Parameters and
    /// constants are produced where used.
    fn ensure(&mut self, fb: &mut FunctionBuilder<'_>, v: ValueId) -> BResult<()> {
        if self.cx.env.get(v.index()).copied().flatten().is_some() {
            return Ok(());
        }
        if let Some(k) = self.constants.get(&v).copied() {
            self.cx.define(v, k);
            return Ok(());
        }
        let vars = self.vars()?;
        let var = vars.demoted_var(v).or_else(|| vars.phi_var(v));
        if let Some(var) = var {
            let x = Self::load_var(fb, var)?;
            self.cx.define(v, x);
        }
        Ok(())
    }

    /// Bind the values instruction `id` reads, as the block imports them (`block_values`,
    /// through `FamilyPlans::reads`: a forwarding `extractvalue` reads the value it forwards).
    fn ensure_operands(&mut self, fb: &mut FunctionBuilder<'_>, id: InstId) -> BResult<()> {
        let body = self.cx.body;
        let Some(inst) = body.insts.get(id) else {
            return Err(self.cx.fail(format!("{id:?} is missing")));
        };
        let mut reads: Vec<ValueId> = Vec::new();
        let values = self.cx.plan.values;
        self.cx.plan.families.reads(body, id, inst, |v| {
            if let (Value::Inst(_), Some(x)) = (v, values.id(v)) {
                reads.push(x);
            }
        });
        for v in reads {
            self.ensure(fb, v)?;
        }
        Ok(())
    }

    /// One segment of a predicated block: `seg` under `if live`, after the phis' demoted copies
    /// when `first` and before the parallel copy when `last`. What the segment bound is forgotten
    /// after it.
    fn segment(
        &mut self,
        b: crate::air::ir::BlockId,
        fb: &mut FunctionBuilder<'_>,
        live: SValue,
        seg: &[(InstId, ValueId)],
        first: bool,
        last: bool,
        vars: &StructVars,
        poll: &mut Poll<'_>,
    ) -> BResult<()> {
        let phis = first
            && self
                .phis
                .get(&b.raw())
                .is_some_and(|ps| ps.iter().any(|v| vars.demoted_var(*v).is_some()));
        if seg.is_empty() && !phis && !last {
            return Ok(());
        }
        let mark = self.cx.mark();
        let seg = seg.to_vec();
        fb.if_(
            self,
            live,
            Box::new(move |fb, em: &mut Self| {
                em.cx.on_spine(fb);
                if phis {
                    em.phi_loads(b, fb, vars, true)?;
                }
                for (id, v) in seg {
                    em.ensure_operands(fb, id)?;
                    em.inst(fb, id, v, vars)?;
                    poll.count(fb)?;
                }
                if last {
                    em.copies(b, fb, vars, true)?;
                }
                Ok(if fb.current_block().is_some() {
                    ArmEnd::FallThrough(Vec::new())
                } else {
                    ArmEnd::Exits
                })
            }),
            None,
        )?;
        self.cx.unwind(mark);
        self.cx.on_spine(fb);
        Ok(())
    }

    /// A synchronization of a predicated block: its operands reloaded, emitted with `live` as the
    /// lane's active flag (every lane is active in a workgroup-uniform block, where every
    /// invocation is live or none is), its demoted result stored where live.
    fn sync_inst(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        id: InstId,
        v: ValueId,
        live: SValue,
        uniform: bool,
        vars: &StructVars,
    ) -> BResult<()> {
        self.ensure_operands(fb, id)?;
        self.cx.live = Some(live);
        self.cx.active = (!uniform).then_some(live);
        let r = self.emit_inst(fb, id, v);
        self.cx.live = None;
        self.cx.active = None;
        r?;
        self.cx.on_spine(fb);
        if let Some(var) = vars.demoted_var(v) {
            let x = self.cx.operand(fb, v)?;
            let mark = self.cx.mark();
            fb.if_(
                self,
                live,
                Box::new(move |fb, _em: &mut Self| {
                    Self::store_var(fb, var, x)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                }),
                None,
            )?;
            self.cx.unwind(mark);
            self.cx.on_spine(fb);
        }
        Ok(())
    }

    fn emit_inst(&mut self, fb: &mut FunctionBuilder<'_>, id: InstId, v: ValueId) -> BResult<()> {
        if self.cx.plan.families.fragment.guarded.contains(&id) {
            return self.emit_guarded(fb, id, v);
        }
        self.emit_strategy(fb, id, v)
    }

    /// A side effect a predicated discard site reaches (`FragmentPlan::guarded`), emitted under `if
    /// !discarded` (`fragment::not_discarded`): a discarded invocation's stores, atomics and image
    /// writes do not land, as Metal's discarded thread's do not. Its result, an atomic's old value,
    /// is merged with zero for a discarded invocation, whose atomic Metal does not perform. What
    /// the arm bound is forgotten after it, as after a predicated segment.
    fn emit_guarded(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        id: InstId,
        v: ValueId,
    ) -> BResult<()> {
        let live = fragment::not_discarded(&mut self.cx, fb)?;
        let mark = self.cx.mark();
        let mut ctx: (&mut Self, Option<TypeId>) = (&mut *self, None);
        let run: ArmFn<'_, (&mut Self, Option<TypeId>)> = Box::new(move |fb, (em, ty)| {
            em.cx.on_spine(fb);
            em.emit_strategy(fb, id, v)?;
            if fb.current_block().is_none() {
                return Ok(ArmEnd::Exits);
            }
            Ok(ArmEnd::FallThrough(
                match em.cx.env.get(v.index()).copied().flatten() {
                    Some(x) => {
                        *ty = Some(fb.value_ty(x)?);
                        vec![x]
                    }
                    None => Vec::new(),
                },
            ))
        });
        let skip: ArmFn<'_, (&mut Self, Option<TypeId>)> = Box::new(move |fb, (_, ty)| {
            Ok(ArmEnd::FallThrough(match *ty {
                Some(t) => {
                    let zero = fb.module().constant(Constant::Null { ty: t })?;
                    vec![fb.module().const_value(zero)?]
                }
                None => Vec::new(),
            }))
        });
        let merged = fb.if_(&mut ctx, live, run, Some(skip))?;
        self.cx.unwind(mark);
        self.cx.on_spine(fb);
        if let Some(x) = merged.0.first() {
            self.cx.define(v, *x);
        }
        Ok(())
    }

    fn emit_strategy(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        id: InstId,
        v: ValueId,
    ) -> BResult<()> {
        let plan = self.cx.plan;
        let Some(s) = plan.strategy(v) else {
            return Err(self.cx.fail(format!("{id:?} has no strategy")));
        };
        let cx = &mut self.cx;
        fragment::emit_quad_exit(cx, fb, id)?;
        match s {
            Strategy::Arith(s) => arith::emit(cx, fb, id, s),
            Strategy::Access(s) => access::emit(cx, fb, id, s),
            Strategy::Texture(s) => texture::emit(cx, fb, id, s),
            Strategy::Sampler(s) => sampler::emit(cx, fb, id, s),
            Strategy::Simd(s) => simd::emit(cx, fb, id, s),
            Strategy::Barrier(s) => sync::barrier::emit(cx, fb, id, s),
            Strategy::MemoryModel(s) => sync::memory_model::emit(cx, fb, id, s),
            Strategy::Fragment(s) => fragment::emit(cx, fb, id, s),
            Strategy::Tessellation(s) => tessellation::emit(cx, fb, id, s),
            Strategy::RayTracing(s) => raytracing::emit(cx, fb, id, s),
            Strategy::Mesh(s) => mesh::emit(cx, fb, id, s),
        }
    }
}

impl CodeEmitter for Emitter<'_> {
    fn var_type(&mut self, ty: &DemoteTy, m: &mut ModuleBuilder) -> BResult<TypeId> {
        let found = match ty {
            DemoteTy::FatPtr { .. }
            | DemoteTy::Address
            | DemoteTy::ArenaOffset
            | DemoteTy::HandleIndex => access::spirv_type(m, ty)?,
            _ => arith::spirv_type(m, ty)?,
        };
        match found {
            Some(t) => Ok(t),
            None => Err(self
                .cx
                .fail(format!("no lowered type for demotion type {ty:?}"))),
        }
    }

    fn bind(&mut self, vars: &StructVars) -> BResult<()> {
        self.vars = Some(Rc::new(vars.clone()));
        Ok(())
    }

    fn block(
        &mut self,
        b: crate::air::ir::BlockId,
        fb: &mut FunctionBuilder<'_>,
        poll: &mut Poll<'_>,
    ) -> BResult<()> {
        let vars = self.vars()?;
        if self.cx.entry_block.is_none() {
            self.cx.entry_block = fb.entry_block();
        }
        self.cx.open_block();
        self.cx.on_spine(fb);
        let body = self.cx.body;
        let values = self.cx.plan.values;
        // Loads at the top: demoted values used here, and this block's phis.
        for v in self.imports.get(&b.raw()).cloned().unwrap_or_default() {
            let Some(var) = vars.demoted_var(v) else {
                return Err(self.cx.fail(format!("{v:?} is not demoted")));
            };
            let x = match self.constants.get(&v) {
                Some(k) => *k,
                None => Self::load_var(fb, var)?,
            };
            self.cx.define(v, x);
        }
        self.phi_loads(b, fb, &vars, false)?;
        let Some(block) = body.blocks.get(b) else {
            return Err(self.cx.fail(format!("{b:?} is not a block")));
        };
        for id in block.inst_ids() {
            let Some(inst) = body.insts.get(id) else {
                return Err(self.cx.fail(format!("{id:?} is missing")));
            };
            if inst.kind.is_terminator() || matches!(inst.kind, InstKind::Phi { .. }) {
                continue;
            }
            let Some(v) = values.inst(id) else {
                return Err(self.cx.fail(format!("{id:?} is out of range")));
            };
            self.inst(fb, id, v, &vars)?;
            poll.count(fb)?;
        }
        self.copies(b, fb, &vars, false)
    }

    fn predicated(&self) -> bool {
        self.cx.plan.families.simd.predication.is_some()
    }

    fn entry_live(&self) -> Option<SValue> {
        self.cx.entry_live
    }

    fn syncs(&self, b: crate::air::ir::BlockId) -> bool {
        self.cx
            .plan
            .families
            .simd
            .predication
            .as_ref()
            .is_some_and(|p| p.syncs(b))
    }

    fn uniform_block(&self, b: crate::air::ir::BlockId) -> bool {
        self.cx
            .plan
            .families
            .simd
            .predication
            .as_ref()
            .is_some_and(|p| p.is_uniform(b))
    }

    fn uniform_cond(&self, v: ValueId) -> bool {
        self.cx
            .plan
            .families
            .simd
            .predication
            .as_ref()
            .is_some_and(|p| p.uniform_cond(v))
    }

    /// Block `b` in a predicated walk: its instructions in segments between its synchronizations,
    /// each segment guarded by `live` (the phis' demoted copies in the first, the parallel copy in
    /// the last); each synchronization unguarded with `live` as the lane's active flag (in a
    /// divergent block), its operands reloaded from their variables (a value a guard defined does
    /// not dominate it) and its result stored where live.
    fn block_live(
        &mut self,
        b: crate::air::ir::BlockId,
        live: Option<SValue>,
        fb: &mut FunctionBuilder<'_>,
        poll: &mut Poll<'_>,
    ) -> BResult<()> {
        let Some(live) = live else {
            return self.block(b, fb, poll);
        };
        let vars = self.vars()?;
        if self.cx.entry_block.is_none() {
            self.cx.entry_block = fb.entry_block();
        }
        self.cx.open_block();
        self.cx.on_spine(fb);
        let body = self.cx.body;
        let plan = self.cx.plan;
        let uniform = self.uniform_block(b);
        let Some(block) = body.blocks.get(b) else {
            return Err(self.cx.fail(format!("{b:?} is not a block")));
        };
        let mut seg: Vec<(InstId, ValueId)> = Vec::new();
        let mut first = true;
        for id in block.inst_ids() {
            let Some(inst) = body.insts.get(id) else {
                return Err(self.cx.fail(format!("{id:?} is missing")));
            };
            if inst.kind.is_terminator() || matches!(inst.kind, InstKind::Phi { .. }) {
                continue;
            }
            let Some(v) = plan.values.inst(id) else {
                return Err(self.cx.fail(format!("{id:?} is out of range")));
            };
            let sync = plan.strategy(v).is_some_and(|s| {
                plan.families
                    .simd
                    .predication
                    .as_ref()
                    .is_some_and(|p| p.is_sync(s))
            });
            if !sync {
                seg.push((id, v));
                continue;
            }
            self.segment(b, fb, live, &seg, first, false, &vars, poll)?;
            seg.clear();
            first = false;
            self.sync_inst(fb, id, v, live, uniform, &vars)?;
            poll.count(fb)?;
        }
        self.segment(b, fb, live, &seg, first, true, &vars, poll)
    }

    fn vote(&mut self, live: SValue, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        simd::workgroup::vote(&mut self.cx, fb, live)
    }

    fn forget(&mut self) {
        self.cx.open_block();
    }

    fn cond(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        // The guard of a slice write every invocation shares is taken by every invocation
        // (`texture::cooperative`); the condition has no other use.
        if self.cx.plan.families.texture.cooperative.contains(&v) {
            let b = fb.module().ty(crate::spirv::ir::Type::Bool)?;
            let t = fb.module().constant(crate::spirv::ir::Constant::Scalar {
                ty: b,
                bits: crate::spirv::ir::ConstBits(1),
            })?;
            return fb.module().const_value(t);
        }
        if self.predicated() {
            self.ensure(fb, v)?;
        }
        self.cx.operand(fb, v)
    }

    fn selector(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let Some(form) = self.cx.plan.selectors.get(&v).copied() else {
            return Err(self.cx.fail(format!("switch selector {v:?} has no form")));
        };
        if self.predicated() {
            self.ensure(fb, v)?;
        }
        let x = self.cx.operand(fb, v)?;
        let u32_ = fb.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let k = |fb: &mut FunctionBuilder<'_>, n: u32| -> BResult<SValue> {
            let c = fb.u32_const(n)?;
            fb.module().const_value(c)
        };
        match form {
            SelectorForm::Direct => Ok(x),
            SelectorForm::Bool => {
                let (one, zero) = (k(fb, 1)?, k(fb, 0)?);
                fb.select(x, one, zero)
            }
            SelectorForm::ZeroExtend => fb.convert(ConvOp::UConvert, x, u32_),
            SelectorForm::Narrow64 { miss } => {
                let u64_ = fb.module().ty(Type::Int {
                    width: 64,
                    signed: false,
                })?;
                let wide = |fb: &mut FunctionBuilder<'_>, n: u64| -> BResult<SValue> {
                    let c = fb.module().constant(Constant::Scalar {
                        ty: u64_,
                        bits: ConstBits(n),
                    })?;
                    fb.module().const_value(c)
                };
                let shift = wide(fb, 32)?;
                let high = fb.binary(BinOp::ShiftRightLogical, x, shift)?;
                let zero = wide(fb, 0)?;
                let fits = fb.binary(BinOp::IEqual, high, zero)?;
                let low = fb.convert(ConvOp::UConvert, x, u32_)?;
                let miss = k(fb, miss)?;
                fb.select(fits, low, miss)
            }
            SelectorForm::Words { miss } => {
                let low = fb.composite_extract(x, &[0])?;
                let high = fb.composite_extract(x, &[1])?;
                let zero = k(fb, 0)?;
                let fits = fb.binary(BinOp::IEqual, high, zero)?;
                let miss = k(fb, miss)?;
                fb.select(fits, low, miss)
            }
        }
    }

    fn on_return(&mut self, v: Option<ValueId>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        stage_io::emit_return(&mut self.cx, fb, v)?;
        fragment::emit_return(&mut self.cx, fb)
    }
}

/// A finalized module and what its emission recorded that the module's artifact reports.
#[non_exhaustive]
pub struct Emitted {
    pub module: TypedModule,
    /// The buffer slots a store merged a partial word into through the sub-word store's atomics:
    /// the module reads each of them, whatever the function's own accesses are.
    pub merged: BTreeSet<SlotRef>,
    /// The ledger bytes charged for the module as its builder grew: they go when serialization
    /// consumes it.
    pub held: usize,
}

/// Structurize and emit `e` under `plan` into a finalized module with P9 applied.
pub fn emit_module(e: &Entry, plan: &Plan, m: &mut Meter) -> Result<Emitted, Error> {
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry function has no body"))?;
    let view = EntryView::new(e, body, plan);
    let structured = m.weighted(crate::pipeline::compile::weights::STRUCTURIZE, |m| {
        crate::structurize::structurize(&view, m)
    })?;
    let mut mb = ModuleBuilder::new();
    let emitted = (|| -> Result<(TypedModule, Vec<Offer>, BTreeSet<SlotRef>, usize), Error> {
        let decl = declare(&mut mb, plan)?;
        // The builder is charged as it grows: the declarations here, the entry hooks' code
        // before the walk, the walk's growth by `structurize::emit` as it emits, and what the
        // walk left once it returns.
        let mut held = Growth::default();
        held.charge_to(mb.footprint(), m)?;
        let mut em = Emitter::new(e, body, plan, &decl);
        // The emitter's tables once built, with the peak of the sets that built them, and what
        // they grew to once the walk returns.
        let mut tables = Growth::default();
        tables.charge_to(em.footprint(), m)?;
        m.charge(em.scratch)?;
        m.release(em.scratch);
        let void = mb.ty(Type::Void)?;
        let walked = {
            let mut fb = mb.function(void)?;
            // The entry hook: what must happen before any instruction of the body (the pulled
            // vertex fetch) or must not depend on where the body first uses it (a push field; a
            // dispatch builtin parameter) is emitted in the function's first block, ahead of
            // the walk.
            let r = em
                .cx
                .begin_hooks(&fb)
                // The push fields and the floor's tail: a kernel's returns at entry, a render
                // stage's folds into its other fields (`EmitCx::load_push_fields`).
                .and_then(|()| em.cx.load_push_fields(&mut fb))
                .and_then(|()| dispatch::emit_entry(&mut em.cx, &mut fb))
                // What the invocation draws (a merged draw's command and its slot tables) before
                // any hook reads a slot table.
                .and_then(|()| stage_io::emit_draw_entry(&mut em.cx, &mut fb))
                .and_then(|()| access::dynamic::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| stage_io::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| tessellation::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| fragment::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| em.cx.end_hooks(&fb))
                .map_err(Error::from)
                .and_then(|()| {
                    held.charge_to(fb.module().footprint(), m)
                        .map_err(Error::from)
                })
                .and_then(|()| crate::structurize::emit::emit(&structured, &mut fb, &mut em, m));
            match r {
                Ok(_) => fb.finish().map_err(Error::from),
                Err(e) => Err(e),
            }
        };
        let grown = tables.charge_to(em.footprint(), m);
        m.release(tables.charged());
        // A recorded fault replaces the builder error.
        let func = match (walked, &em.cx.fault) {
            (_, Some(fault)) => return Err(fault.clone()),
            (r, None) => r?,
        };
        grown?;
        // The tile staging's entry-wrapper hook: a tile function that stages an attachment runs
        // `body` between the loads and the stores of its tile's cells.
        let mut wrapper = em.cx.for_new_function();
        let func = match (
            access::tile::wrap_entry(&mut wrapper, &mut mb, func),
            &wrapper.fault,
        ) {
            (_, Some(fault)) => return Err(fault.clone()),
            (r, None) => r?,
        };
        // An object or mesh function's wrapper: its grid's launch, or its outputs'
        // initialization and the culling of the slots past its count.
        let func = match (
            mesh::wrap_entry(&mut wrapper, &mut mb, func),
            &wrapper.fault,
        ) {
            (_, Some(fault)) => return Err(fault.clone()),
            (r, None) => r?,
        };
        let merged = em
            .cx
            .merged
            .iter()
            .chain(&wrapper.merged)
            .filter_map(|&i| plan.slots.get(i).copied().flatten().map(|c| c.slot))
            .collect();
        held.charge_to(mb.footprint(), m)?;
        // The entry-wrapper hook: a raster order group's body runs inside the interlock
        // critical section of a new entry function, and the body of a function looping over its
        // covered samples once per sample from a new entry function; otherwise the body is the
        // entry.
        let func = fragment::wrap_entry(&mut mb, plan, &decl.families, func)?;
        let model = match plan.stage {
            Stage::Kernel => ExecModel::GLCompute,
            // A post-tessellation vertex function.
            Stage::Vertex if plan.families.tessellation.eval.is_some() => {
                ExecModel::TessellationEvaluation
            }
            Stage::Vertex => ExecModel::Vertex,
            Stage::Fragment => ExecModel::Fragment,
            Stage::Object => ExecModel::TaskEXT,
            Stage::Mesh => ExecModel::MeshEXT,
        };
        mb.entry(model, func, decl.modes.clone())?;
        // The pinned size and full subgroups the plan chose need device facts the SPIR-V does
        // not state; the module's requirements record them.
        mb.subgroup(plan.subgroup);
        // The lowering's offers outlive the builder; charged until weighed.
        let offers = mb.offers().to_vec();
        let module = mb.finalize()?;
        m.charge(std::mem::size_of_val(offers.as_slice()))?;
        Ok((module, offers, merged, held.charged()))
    })();
    structured.release(m);
    let (mut module, offers, merged, held) = emitted?;
    promote_single_store(&mut module, m)?;
    // What no instruction reads leaves the module, after P9 turned its loads into copies, and the
    // offers that make it smaller are taken.
    let r = remove_dead_keeping(&mut module, &offers, m);
    m.release(std::mem::size_of_val(offers.as_slice()));
    r?;
    // Under the 16-bit float controls, a module that computes `half` in 16 bits holds its `half`
    // values there (`spirv::half_hold`).
    if plan.families.arith.controls {
        crate::spirv::half_hold::hold(&mut module, m)?;
    }
    Ok(Emitted {
        module,
        merged,
        held,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::ir::build::ModuleBuilder as AirBuilder;
    use crate::air::ir::inst::FastMath;
    use crate::air::ir::BlockId;
    use crate::spirv::ids::ScalarTy;

    /// A phi's incoming value is read by the parallel copy at the end of its incoming block, so
    /// that block loads it from its demotion variable even when it defines and otherwise uses
    /// nothing; the defining block never loads its own value.
    #[test]
    fn a_phi_copy_imports_its_incoming_value_into_the_incoming_block() {
        let mut mb = AirBuilder::new();
        let void = mb.void();
        let i1 = mb.int(1);
        let fty = mb.fn_ty(void, &[]);
        let t = mb.const_int(i1, 1);
        let f = mb.const_int(i1, 0);
        let mut fb = mb.define("k", fty);
        let a = BlockId::new(0);
        let (b, c, d) = (fb.block(), fb.block(), fb.block());
        fb.br(b);
        fb.switch_to(b);
        // `p` is defined in `b` and flows into `d`'s phi on the edge from `c`.
        let p = fb.push(
            i1,
            InstKind::Phi {
                incoming: Box::new([(t, a)]),
                fmf: FastMath::NONE,
            },
        );
        fb.cond_br(p, c, d);
        fb.switch_to(c);
        fb.br(d);
        fb.switch_to(d);
        fb.push(
            i1,
            InstKind::Phi {
                incoming: Box::new([(p, c), (f, b)]),
                fmf: FastMath::NONE,
            },
        );
        fb.ret(None);
        let func = fb.finish();
        let module = mb.finish().expect("module");
        let body = module
            .func(func)
            .and_then(|f| f.body.as_ref())
            .expect("body");
        let values = ValueIndex::new(0, body.insts.len(), module.consts().len()).expect("index");
        let first = |blk: BlockId| {
            let (id, _) = body.block_insts(blk).next().expect("an instruction");
            values.inst(id).expect("in range")
        };
        let (pv, rv) = (first(b), first(d));
        let cross = [(pv, DemoteTy::Scalar(ScalarTy::Bool))];
        let (phis, imports, _) = block_values(body, values, &cross, &FamilyPlans::default());
        assert_eq!(
            imports.get(&c.raw()),
            Some(&vec![pv]),
            "the copy's block loads it"
        );
        assert_eq!(imports.get(&b.raw()), None, "its own block does not");
        assert_eq!(
            imports.get(&d.raw()),
            None,
            "the phi's block reads its variable"
        );
        assert_eq!(phis.get(&b.raw()), Some(&vec![pv]));
        assert_eq!(phis.get(&d.raw()), Some(&vec![rv]));
    }
}
