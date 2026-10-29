//! `lower::Emitter`: the `CodeEmitter` of a `Plan`, and the module driver that runs the
//! structurizer and emission.
//!
//! [`emit_module`] declares the module-scope objects the plan selected (the slots' views, the push
//! block in placement order, the kernel's workgroup size), structurizes the entry's CFG through
//! [`EntryView`], and lets P8 (`structurize::emit`) walk the tree with an [`Emitter`], which emits
//! each block by routing every instruction's [`Strategy`] to its family's emit function. It then
//! lets `fragment::wrap_entry` choose the entry function (the walked body itself, or a raster order
//! group's interlocked wrapper that calls it), declares the entry point, finalizes, runs P9
//! (`promote_single_store`), and removes what no instruction reads, taking the lowering's offers
//! that make the module smaller (`spirv::dead::remove_dead_keeping`).
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
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, SlotRef, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::dead::{remove_dead_keeping, Offer};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{
    BlockId as SBlockId, ConstId, DemoteTy, FuncId, GlobalId, TypeId, Value as SValue, VarId,
};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, MemAccess, PushMember, Type,
    TypedModule,
};
use crate::structurize::emit::{CodeEmitter, StructVars};
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
    // Before any instruction: every multiply the device must round on its own takes the
    // profile's form.
    mb.product_form(plan.families.arith.product);
    // And every narrow integer's mask the driver's form.
    mb.set_mask_form(plan.families.arith.mask);
    // And every `half` rounding's quantize the device's form.
    mb.set_quantize_form(plan.families.arith.quantize);
    mb.set_native_half(plan.families.arith.native_half);
    mb.set_native_half_nans(plan.families.arith.native_half_nans);
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
    /// Invocation-invariant values emitted since the block opened or in the function's entry
    /// block, each with the SPIR-V block it was emitted in ([`Invariant`]).
    memo: Vec<(Invariant, SValue, SBlockId)>,
    /// The SPIR-V blocks the open block's top-level code ran through, in order: a value emitted
    /// in one of them dominates everything the block emits after it.
    spine: Vec<SBlockId>,
    /// The function's entry block: a value emitted there dominates every block, so its memo
    /// entries outlive the block that emitted them.
    entry_block: Option<SBlockId>,
    /// While a workgroup synchronization of a predicated walk is emitted, the calling invocation's
    /// live flag: a lane that is not live takes part with null operands. `None`: every invocation
    /// executing the synchronization is live.
    pub live: Option<SValue>,
    /// With `live`, in a divergent block: the lane takes part in the exchange as active only
    /// where `live` holds. In a workgroup-uniform block every invocation is live or none is, so
    /// every lane takes part as active (a result nobody is live for is never stored).
    pub active: Option<SValue>,
    /// The memory-plan slots a buffer store emitted so far merges a partial word into, through
    /// the sub-word store's two atomics, which read the word they modify. Whether a store merges
    /// depends on the leaf the lowering chose and on the units the slot's views hold on the
    /// profile, so emission records it as it emits the store (`memory::leaf::store`), and the
    /// artifact reports a read of each such slot ([`Emitted::merged`]).
    pub(crate) merged: BTreeSet<usize>,
    /// The function whose first block computed the constants it reads as data, with each
    /// constant's local value there (`specialized::emit_entry`), for
    /// `TypedModule::localize_specs` once the module is finalized.
    pub(crate) localized: Option<(FuncId, Vec<(ConstId, SValue)>)>,
    fault: Option<Error>,
    /// The sampling routines the function's samples call, each defined at its first call
    /// (`sampler::routine`).
    pub(crate) sampling: crate::lower::sampler::routine::Routines,
    /// A merged draw's vertex function's command, vertex, instance and slot table, which
    /// `stage_io::emit_entry` finds before the body (`stage_io::MergedRead`): defined in the
    /// function's straight-line entry, so every later block of the function may read them.
    pub(crate) merged_draw: Option<crate::bind::MergedVertex>,
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
    /// The table word of the argument buffer's sampler of a handle, tested for a cubic filter
    /// (`sampler::emit`'s reach test): an operation through the same handle where an earlier test
    /// dominates it ran that test on the same sampler, so it needs none of its own.
    CubicTested(ValueId),
    /// An indexed capture form's raw index (`stage_io::read_captured_index`), read at entry
    /// (`stage_io::emit_entry`): the index buffer is read-only to the draw.
    CapturedIndex,
    /// The private list form's base vertex (`stage_io::read_listed_base`), read at entry
    /// (`stage_io::emit_entry`): the list is read-only to the draw.
    ListedBase,
    /// The root table's sink address (`access::dynamic::read_sink_host`), read at entry
    /// (`access::dynamic::emit_entry`): the root table is read-only to the dispatch.
    SinkHost,
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
            live: None,
            active: None,
            merged: BTreeSet::new(),
            localized: None,
            fault: None,
            sampling: Default::default(),
            merged_draw: None,
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
    /// block, in a block of the open block's spine, or in the function's entry block.
    fn recall(&self, fb: &FunctionBuilder<'_>, key: Invariant) -> Option<SValue> {
        let here = fb.current_block()?;
        self.memo
            .iter()
            .find(|(k, _, _)| *k == key)
            .filter(|(_, _, at)| {
                *at == here || Some(*at) == self.entry_block || self.spine.contains(at)
            })
            .map(|(_, v, _)| *v)
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

    /// An indexed capture form's raw index, read once where it dominates: `stage_io::emit_entry`
    /// reads it in the function's first block, which dominates every block of the function, the
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
    /// reads it in the function's first block, so no wide access reads the table for it.
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
    fn load_push_fields(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        let plan: &Plan = self.plan;
        for placed in &plan.push.fields {
            self.push_field(fb, placed.field)?;
        }
        Ok(())
    }

    /// Load push field `field` in the open block.
    fn load_push_field(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        field: PushField,
    ) -> BResult<SValue> {
        // A device-replay variant reads a slot's Metal address from its root entry, and its
        // dispatch region from the region record of its command and region; a flat variant reads
        // its grid from its dispatch's region record, and a folded variant its region and group
        // counts.
        let flat = self
            .plan
            .dispatch
            .as_ref()
            .is_some_and(|d| d.flat_grid || d.folded_groups);
        match field {
            // A merged draw's vertex function reads its command's slot table.
            PushField::RootTableAddress if self.plan.families.stage_io.merged.is_some() => {
                return match self.merged_draw {
                    Some(m) => Ok(m.table),
                    None => Err(self.fail("a merged draw's slot table read before its entry")),
                };
            }
            PushField::SlotGuestBase(slot) if self.plan.families.access.replay => {
                return access::replay_guest_base(self, fb, slot)
            }
            PushField::DispatchRegion if self.plan.families.access.replay || flat => {
                return access::replay_region(self, fb)
            }
            _ => {}
        }
        let member = self.decl.push.iter().find(|m| m.field == field).copied();
        let (Some(member), Some(g)) = (member, self.decl.push_global) else {
            // A field the push block does not hold is in the stage's `ArgumentTable`, which the
            // access family declares and reads.
            return access::table_field(self, fb, field);
        };
        let base = fb.module().global_value(g)?;
        let index = fb.u32_const(member.index)?;
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
    /// (`stage_io::emit_entry`) in the function's first block, before the walk, so its definition
    /// dominates every block and opening a block does not forget it.
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
        let entry = self.entry_block;
        let kept = &self.spine;
        self.memo
            .retain(|(_, _, at)| Some(*at) == entry || kept.contains(at));
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
                live: None,
                active: None,
                merged: BTreeSet::new(),
                localized: None,
                fault: None,
                sampling: Default::default(),
                merged_draw: None,
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
            DemoteTy::FatPtr { .. } | DemoteTy::ArenaOffset | DemoteTy::HandleIndex => {
                access::spirv_type(m, ty)?
            }
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

    fn block(&mut self, b: crate::air::ir::BlockId, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
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
        }
        self.copies(b, fb, &vars, false)
    }

    fn predicated(&self) -> bool {
        self.cx.plan.families.simd.predication.is_some()
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
    ) -> BResult<()> {
        let Some(live) = live else {
            return self.block(b, fb);
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
            self.segment(b, fb, live, &seg, first, false, &vars)?;
            seg.clear();
            first = false;
            self.sync_inst(fb, id, v, live, uniform, &vars)?;
        }
        self.segment(b, fb, live, &seg, first, true, &vars)
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
        fragment::emit_return(&mut self.cx, fb, v)
    }
}

/// A finalized module and what its emission recorded that the module's artifact reports.
#[non_exhaustive]
pub struct Emitted {
    pub module: TypedModule,
    /// The buffer slots a store merged a partial word into through the sub-word store's atomics:
    /// the module reads each of them, whatever the function's own accesses are.
    pub merged: BTreeSet<SlotRef>,
}

/// Structurize and emit `e` under `plan` into a finalized module with P9 applied.
pub fn emit_module(e: &Entry, plan: &Plan, m: &mut Meter) -> Result<Emitted, Error> {
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry function has no body"))?;
    let view = EntryView::new(e, body, plan);
    let structured = crate::structurize::structurize(&view, m)?;
    let mut mb = ModuleBuilder::new();
    let emitted = (|| -> Result<(TypedModule, Vec<Offer>, BTreeSet<SlotRef>), Error> {
        let decl = declare(&mut mb, plan)?;
        // The builder is charged as it grows: the declarations here, the walk's growth by
        // `structurize::emit` after each block, and what the walk left once it returns.
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
            // the walk. The simdgroup lane counted at entry comes first, ahead of any branch
            // the other hooks emit, where every invocation of the simdgroup is active.
            let r = crate::lower::simd::emit_entry(&mut em.cx, &mut fb)
                .and_then(|()| em.cx.load_push_fields(&mut fb))
                // The bound values the function reads as data, right after their fields.
                .and_then(|()| crate::lower::specialized::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| dispatch::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| access::dynamic::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| stage_io::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| tessellation::emit_entry(&mut em.cx, &mut fb))
                .and_then(|()| fragment::emit_entry(&mut em.cx, &mut fb))
                .map_err(Error::from)
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
        // An emulated post-tessellation vertex function's wrapper: the tessellator's point for
        // the vertex it draws, and the function's patch inputs.
        let func = match (
            tessellation::wrap_entry(&mut wrapper, &mut mb, func),
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
            // A post-tessellation vertex function, but one emulated on a device without
            // `tessellationShader`, a plain vertex entry.
            Stage::Vertex
                if plan
                    .families
                    .tessellation
                    .eval
                    .as_ref()
                    .is_some_and(|e| e.emulated.is_none()) =>
            {
                ExecModel::TessellationEvaluation
            }
            Stage::Vertex => ExecModel::Vertex,
            Stage::Fragment => ExecModel::Fragment,
            // An emulated object or mesh function is a compute entry.
            Stage::Object | Stage::Mesh if plan.families.mesh.emulated => ExecModel::GLCompute,
            Stage::Object => ExecModel::TaskEXT,
            Stage::Mesh => ExecModel::MeshEXT,
        };
        mb.entry(model, func, decl.modes.clone())?;
        // The pinned size and full subgroups the plan chose need device facts the SPIR-V does
        // not state; the module's requirements record them.
        mb.subgroup(plan.subgroup);
        // The lowering's offers outlive the builder; charged until weighed.
        let offers = mb.offers().to_vec();
        let mut module = mb.finalize()?;
        // Every read of a constant the function reads as data takes its local value, and the
        // constants nothing else reads leave.
        if let Some((body, local)) = &em.cx.localized {
            module.localize_specs(*body, local)?;
        }
        m.charge(std::mem::size_of_val(offers.as_slice()))?;
        Ok((module, offers, merged))
    })();
    structured.release(m);
    let (mut module, offers, merged) = emitted?;
    promote_single_store(&mut module, m)?;
    // What no instruction reads leaves the module, after P9 turned its loads into copies, and the
    // offers that make it smaller are taken.
    let r = remove_dead_keeping(&mut module, &offers, m);
    m.release(std::mem::size_of_val(offers.as_slice()));
    r?;
    Ok(Emitted { module, merged })
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
