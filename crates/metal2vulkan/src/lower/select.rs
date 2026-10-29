//! `select`: the total routing of every AIR instruction, intrinsic family, and entry parameter to
//! its lowering family, and the per-function `Plan`.
//!
//! Selection runs on the canonical SSA [`Entry`] before structurization and decides everything
//! that changes which instructions the function contains: one [`Strategy`] per instruction that
//! needs lowering, the slots' buffer forms, the push placement, the switch-selector legalization,
//! the demotion types of block-crossing values, the dispatch plan, and the subgroup tier. Emission
//! then cannot fail except with a `BuildError` (an internal error).
//!
//! Routing is by instruction structure and by the typed intrinsic families decode produced
//! (`AirIntrinsic`, `LlvmIntrinsic`), never by name:
//!
//! - `arith`: integer and float arithmetic, casts, compares, selects, vector and aggregate ops
//!   on non-pointers; `air` math, conversions, packing; `llvm` math.
//! - `access`: loads, stores, GEPs, allocas, atomics, pointer casts, compares, and selects; `air`
//!   atomics, coherent accesses, async copies, a kernel's explicit imageblock cells and extent;
//!   `llvm.mem*`.
//! - `texture`, `sampler`: the `air` texture and sampler families.
//! - `simd`: `air` simdgroup, quad, uniformity, and matrix ops; AGX lane intrinsics.
//! - `sync::barrier`: `air` barriers and derivatives; `sync::memory_model`: fences.
//! - `fragment`: `air` fragment intrinsics.
//! - `stage_io`: stage IO parameters and outputs; `dispatch`: kernel dispatch builtins.
//! - Phis and terminators: the structurizer and `lower::emit`.
//!
//! Out-of-scope families (fragment imageblocks and tile shading, ray tracing, tensors, GPU-encoded
//! commands, function tables) and constructs canonicalization removes (direct calls,
//! function-constant intrinsics) are refused here.
//!
//! Every decision that belongs to a family is the family's hook, and this file only combines what
//! the families contribute: parameters ([`param_family`], each family's `select_param`), returns
//! (`stage_io::select_return`), each family's own plan ([`FamilyPlans`], completed by its
//! `finish`), the slots' forms, extra transport, and push-table support
//! (`access::finish_slots`, `access::accept_placement`), and the accumulators of [`SelectCx`]:
//! descriptor counts per kind ([`DescriptorCounts`]), the simd class, and tangled instructions.
//! The generic checks over those accumulators (per-stage descriptor limits, the subgroup tier,
//! the size estimate) stay here.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{Builtin, ImageblockKind, ParamBinding, ParamRole};
use crate::air::error::Location;
use crate::air::intrinsics::{AirIntrinsic, FunctionConstant, FunctionTable, LlvmIntrinsic};
use crate::air::ir::{
    Body, CastOp, FuncId, Idx, Inst, InstId, InstKind, Type, TypeId, Value, ValueId, ValueIndex,
};
use crate::air::Entry;
use crate::budget::{tree_bytes, Growth, Meter};
use crate::device::function_storage;
use crate::device::profile::Limits;
use crate::device::requirements::{VkFeature, VkLimit};
use crate::device::subgroup::{self, SimdClass, SubgroupRequirement, Tier};
use crate::device::DeviceProfile;
use crate::error::{
    Capability, DiagName, Error, Feature, InternalError, InternalKind, LowerFamily, Unsupported,
};
use crate::lower::dispatch::DispatchPlan;
use crate::lower::estimate::{self, SizeEstimate};
use crate::lower::push;
use crate::lower::strategy::{FamilyStrategy, Strategy};
use crate::lower::{
    access, arith, dispatch, fragment, mesh, raytracing, sampler, simd, stage_io, sync,
    tessellation, texture,
};
use crate::memory::static_root::{self, MemoryPlan};
use crate::pipeline::artifact::{PrecisionNote, SpecClause, SpecTest, SpecializedNote};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{AccessSet, BufferForm, PushPlacement, Window};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{PushField, SlotRef, SpecKey, Stage};
use crate::spirv::ids::{DemoteTy, ScalarTy};

/// The per-function lowering inputs: exactly the public projection that is part of the compile key.
pub type LowerInputs = LoweringInputs;

/// How a switch selector becomes the 32-bit `OpSwitch` selector: equal to a case literal exactly
/// when the source selector is, and to no literal otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectorForm {
    /// A 32-bit integer, used as is.
    Direct,
    /// An `i1`: `select(v, 1, 0)`.
    Bool,
    /// An `i8`/`i16`: zero-extended (case literals are zero-extended bit patterns).
    ZeroExtend,
    /// An `i64`: its low word when the high word is 0, else `miss`, a `u32` that is no case
    /// literal of any switch on the value (literals fit 32 bits, WideSwitchLiteral).
    Narrow64 { miss: u32 },
    /// An `i33`..`i64` held as two words without `shaderInt64`: the low word when the high word is
    /// 0, else `miss`, as [`SelectorForm::Narrow64`] reads the 64-bit container.
    Words { miss: u32 },
}

/// A slot's selected form with its proven use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotChoice {
    pub slot: SlotRef,
    pub form: BufferForm,
    pub access: AccessSet,
    pub window: Window,
}

/// The descriptors one function's stage uses, per kind. Families add theirs while selecting
/// (`SelectCx::descriptors`); `select` checks the total against the stage's limits. A combined
/// image-sampler counts as one sampler and one sampled image, a uniform texel buffer as a sampled
/// image, and a storage texel buffer as a storage image (Vulkan's per-stage limit rules).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DescriptorCounts {
    pub samplers: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_images: u32,
    pub storage_images: u32,
    pub input_attachments: u32,
    /// Fragment color attachments: no descriptor, but they count in `maxPerStageResources`.
    pub color_attachments: u32,
    /// Acceleration structures: their own limit, outside `maxPerStageResources`.
    pub acceleration_structures: u32,
}

impl DescriptorCounts {
    /// The kind-wise sum, saturating.
    pub fn plus(&self, o: &DescriptorCounts) -> DescriptorCounts {
        DescriptorCounts {
            samplers: self.samplers.saturating_add(o.samplers),
            uniform_buffers: self.uniform_buffers.saturating_add(o.uniform_buffers),
            storage_buffers: self.storage_buffers.saturating_add(o.storage_buffers),
            sampled_images: self.sampled_images.saturating_add(o.sampled_images),
            storage_images: self.storage_images.saturating_add(o.storage_images),
            input_attachments: self.input_attachments.saturating_add(o.input_attachments),
            color_attachments: self.color_attachments.saturating_add(o.color_attachments),
            acceleration_structures: self
                .acceleration_structures
                .saturating_add(o.acceleration_structures),
        }
    }

    /// The number of descriptors, over every kind that has one.
    pub fn descriptors(&self) -> u64 {
        [
            self.samplers,
            self.uniform_buffers,
            self.storage_buffers,
            self.sampled_images,
            self.storage_images,
            self.input_attachments,
            self.acceleration_structures,
        ]
        .iter()
        .map(|n| u64::from(*n))
        .sum()
    }

    /// `maxPerStageResources`' count: every kind except samplers.
    fn resources(&self) -> u64 {
        [
            self.uniform_buffers,
            self.storage_buffers,
            self.sampled_images,
            self.storage_images,
            self.input_attachments,
            self.color_attachments,
        ]
        .iter()
        .map(|n| u64::from(*n))
        .sum()
    }

    /// The first per-stage limit these counts exceed on `limits`: `(limit, required, available)`.
    pub fn exceeded(&self, limits: &Limits) -> Option<(VkLimit, u64, u64)> {
        let l = limits;
        [
            (
                VkLimit::MaxPerStageDescriptorSamplers,
                u64::from(self.samplers),
                l.max_per_stage_descriptor_samplers,
            ),
            (
                VkLimit::MaxPerStageDescriptorUniformBuffers,
                u64::from(self.uniform_buffers),
                l.max_per_stage_descriptor_uniform_buffers,
            ),
            (
                VkLimit::MaxPerStageDescriptorStorageBuffers,
                u64::from(self.storage_buffers),
                l.max_per_stage_descriptor_storage_buffers,
            ),
            (
                VkLimit::MaxPerStageDescriptorSampledImages,
                u64::from(self.sampled_images),
                l.max_per_stage_descriptor_sampled_images,
            ),
            (
                VkLimit::MaxPerStageDescriptorStorageImages,
                u64::from(self.storage_images),
                l.max_per_stage_descriptor_storage_images,
            ),
            (
                VkLimit::MaxPerStageDescriptorInputAttachments,
                u64::from(self.input_attachments),
                l.max_per_stage_descriptor_input_attachments,
            ),
            (
                VkLimit::MaxPerStageResources,
                self.resources(),
                l.max_per_stage_resources,
            ),
            (
                VkLimit::MaxPerStageDescriptorAccelerationStructures,
                u64::from(self.acceleration_structures),
                l.max_per_stage_descriptor_acceleration_structures,
            ),
        ]
        .into_iter()
        .map(|(limit, n, max)| (limit, n, u64::from(max)))
        .find(|(_, n, max)| n > max)
    }
}

/// Each family's own part of the per-function plan: the family defines the type, fills it while
/// selecting (through `SelectCx::families`) and in its `finish`, and reads it back while
/// declaring and emitting (`Plan::families`). `select` and `lower::emit` only carry it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FamilyPlans {
    pub arith: arith::ArithPlan,
    pub access: access::AccessPlan,
    pub texture: texture::TexturePlan,
    pub sampler: sampler::SamplerPlan,
    pub simd: simd::SimdPlan,
    pub barrier: sync::barrier::BarrierPlan,
    pub memory_model: sync::memory_model::MemoryModelPlan,
    pub fragment: fragment::FragmentPlan,
    pub stage_io: stage_io::StageIoPlan,
    pub tessellation: tessellation::TessellationPlan,
    pub raytracing: raytracing::RayTracingPlan,
    pub mesh: mesh::MeshPlan,
}

impl FamilyPlans {
    /// Call `f` with each value instruction `id` (`inst`) of `body` reads where it stands, as
    /// the families lower it: none for an instruction that emits nothing (a planned index
    /// extension, or a `ptrtoint` whose bits go only into forwarding cells); the value a forwarding
    /// `extractvalue` forwards, instead of its aggregate; and otherwise each operand, a use of an index extension reading the extension's narrow
    /// operand. A typed access whose chain takes its `getelementptr`'s index operands reads those
    /// too. A phi reads nothing where it stands: its copies read at the end of each incoming block
    /// ([`FamilyPlans::incoming_reads`]).
    ///
    /// This is the one hook for what an instruction reads: the block-crossing values (`select`),
    /// each block's imports and the predicated walk's reloads (`lower::emit`), and the predicated
    /// walk's demotions (`simd::divergent`) all read through it, so a value one of them carries to
    /// a use, the others load there.
    pub fn reads(&self, body: &Body, id: InstId, inst: &Inst, f: impl FnMut(Value)) {
        self.reads_in(None, body, id, inst, f);
    }

    /// [`FamilyPlans::reads`], looking a plan up only for an instruction `index` says it names.
    pub fn reads_indexed(
        &self,
        index: &ReadIndex,
        body: &Body,
        id: InstId,
        inst: &Inst,
        f: impl FnMut(Value),
    ) {
        self.reads_in(Some(index), body, id, inst, f);
    }

    /// The instructions the plans [`FamilyPlans::reads`] reads name, for a body of `insts`
    /// instructions.
    pub fn read_index(&self, insts: usize) -> ReadIndex {
        let mut flags = vec![0u8; insts];
        let mut mark = |id: InstId, bit: u8| {
            if let Some(f) = flags.get_mut(id.index()) {
                *f |= bit;
            }
        };
        for &id in self
            .access
            .index_extensions
            .iter()
            .chain(&self.access.cell_bits)
        {
            mark(id, ReadIndex::READS_NOTHING);
        }
        for &id in self.arith.forwards.keys() {
            mark(id, ReadIndex::FORWARDED);
        }
        for &id in self.access.typed_indices.keys() {
            mark(id, ReadIndex::TYPED_INDICES);
        }
        ReadIndex { flags }
    }

    /// [`FamilyPlans::reads`], with `index` where given.
    fn reads_in(
        &self,
        index: Option<&ReadIndex>,
        body: &Body,
        id: InstId,
        inst: &Inst,
        mut f: impl FnMut(Value),
    ) {
        let named = |id: InstId, bit: u8| index.is_none_or(|ix| ix.names(id, bit));
        if named(id, ReadIndex::READS_NOTHING) && access::index::reads_nothing(&self.access, id) {
            return;
        }
        let read = |v: Value| match v {
            Value::Inst(d) if !named(d, ReadIndex::READS_NOTHING) => v,
            _ => access::index::read_value(body, &self.access, v),
        };
        let forwarded = if named(id, ReadIndex::FORWARDED) {
            arith::aggregate::forwarded(&self.arith, id)
        } else {
            None
        };
        match (&inst.kind, forwarded) {
            (InstKind::Phi { .. }, _) => {}
            (_, Some(v)) => f(read(v)),
            (kind, None) => kind.for_each_value(|v| f(read(v))),
        }
        // A typed access whose chain takes its GEP's index operands reads them too.
        if named(id, ReadIndex::TYPED_INDICES) {
            for &v in self.access.typed_indices.get(&id).into_iter().flatten() {
                f(read(v));
            }
        }
    }

    /// Call `f` with each value the parallel copy on the edge from block `from` into phi `phi`
    /// reads at the end of `from`, `v` being the phi's incoming value on that edge: the values a
    /// member-wise phi's assembly reads, or else `v`, a use of an index extension reading the
    /// extension's narrow operand. The phi counterpart of [`FamilyPlans::reads`].
    pub fn incoming_reads(
        &self,
        body: &Body,
        values: ValueIndex,
        (phi, v, from): (InstId, Value, crate::air::ir::BlockId),
        mut f: impl FnMut(Value),
    ) {
        let read = |v: Value| access::index::read_value(body, &self.access, v);
        match arith::aggregate::incoming(&self.arith, phi, from) {
            Some(build) => {
                let mut reads = Vec::new();
                build.reads(&mut reads);
                for r in reads.into_iter().filter_map(|r| values.value(r)) {
                    f(read(r));
                }
            }
            None => f(read(v)),
        }
    }
}

/// Which instructions the plans [`FamilyPlans::reads`] reads name, by instruction number: a walk
/// of every instruction's reads ([`FamilyPlans::reads_indexed`]) searches a plan's sorted set
/// only for an instruction it names. An instruction outside the index is looked up.
pub struct ReadIndex {
    flags: Vec<u8>,
}

impl ReadIndex {
    /// A planned index extension or forwarding-cell `ptrtoint` (`access::index::reads_nothing`).
    const READS_NOTHING: u8 = 1;
    /// A forwarding `extractvalue` (`arith::aggregate::forwarded`).
    const FORWARDED: u8 = 2;
    /// A typed access that reads its GEP's index operands (`access::AccessPlan::typed_indices`).
    const TYPED_INDICES: u8 = 4;

    /// Whether a plan of kind `bit` may name instruction `id`.
    fn names(&self, id: InstId, bit: u8) -> bool {
        self.flags.get(id.index()).is_none_or(|f| f & bit != 0)
    }
}

/// The per-function plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub stage: Stage,
    pub values: ValueIndex,
    /// One per instruction needing lowering, by [`ValueId`].
    pub strategy: Vec<Option<Strategy>>,
    /// By memory-plan slot index; `None` for slots the function does not access.
    pub slots: Vec<Option<SlotChoice>>,
    pub push: PushPlacement,
    /// The legalization of every switch selector.
    pub selectors: BTreeMap<ValueId, SelectorForm>,
    /// Block-crossing values with their demotion type, sorted.
    pub cross: Vec<(ValueId, DemoteTy)>,
    /// Phis with the demotion type of their incoming variable, sorted.
    pub phis: Vec<(ValueId, DemoteTy)>,
    /// Tangled instructions (subgroup/quad ops, derivatives, implicit-LOD samples, barriers).
    pub tangled: BTreeSet<ValueId>,
    pub dispatch: Option<DispatchPlan>,
    pub simd: (SimdClass, Tier),
    pub subgroup: SubgroupRequirement,
    /// The stage's descriptors per kind.
    pub descriptors: DescriptorCounts,
    pub families: FamilyPlans,
    pub facts: Vec<RuntimeFact>,
    pub precision: Vec<PrecisionNote>,
    /// The notes of `precision` an access keeps only on the arms its specialization selects,
    /// with their conditions ([`resolve_notes`]).
    pub specialized: Vec<SpecializedNote>,
    pub estimate: SizeEstimate,
    /// The private memory a thread keeps, in bytes, as selection holds it to the device's
    /// Function storage (`MemoryPlan::private_bytes`).
    pub private_bytes: u64,
}

impl Plan {
    /// The strategy of `v`, if it has one.
    pub fn strategy(&self, v: ValueId) -> Option<&Strategy> {
        self.strategy.get(v.index()).and_then(Option::as_ref)
    }

    /// The census of the plan's strategies: uses and estimated words per strategy name, each use
    /// named as the finished plan lowers it (`FamilyStrategy::name`).
    ///
    /// An instruction of a nil texture's arm, which the default presence specialization never runs
    /// (`TexturePlan::absent_arm`), counts as a use of its strategy and of [`texture::ABSENT_ARM`],
    /// and its words count under [`texture::ABSENT_ARM`] alone, so the census still names every
    /// strategy the module uses and the words of that arm are one entry.
    pub fn census(&self) -> BTreeMap<String, (u32, u64)> {
        let mut out: BTreeMap<String, (u32, u64)> = BTreeMap::new();
        let arm = &self.families.texture.absent_arm;
        for (i, s) in self.strategy.iter().enumerate() {
            let Some(s) = s else { continue };
            let words = u64::from(s.words());
            // The separate form's sampled images: their uses and words under their own name, the
            // nil arm's words under the nil arm's.
            let separate = separate_samplings(&self.families, s);
            let separate_words = u64::from(separate) * u64::from(sampler::SEPARATE_WORDS);
            if separate != 0 {
                tally(&mut out, sampler::SEPARATE_BINDING.into(), separate, 0);
            }
            if ValueId::from_index(i).is_some_and(|v| arm.binary_search(&v).is_ok()) {
                tally(&mut out, s.name(&self.families), 1, 0);
                tally(
                    &mut out,
                    texture::ABSENT_ARM.into(),
                    1,
                    words + separate_words,
                );
            } else {
                tally(&mut out, s.name(&self.families), 1, words);
                if separate != 0 {
                    tally(
                        &mut out,
                        sampler::SEPARATE_BINDING.into(),
                        0,
                        separate_words,
                    );
                }
            }
        }
        // A shared routine's words are its callers', once, as are a heap handle's
        // out-of-bounds-alpha masks.
        let access = access::routine_words(&self.families.access);
        let heap = texture::routine_words(&self.families.texture, &self.families.access);
        let sampler = sampler::routine_words(&self.families.sampler);
        for (name, words) in arith::routine_words(&self.families.arith)
            .chain(access)
            .chain(heap)
            .chain(sampler)
        {
            tally(&mut out, name.into(), 0, words);
        }
        // The barriers a device's lockstep stands for: uses, no words.
        let device = self.families.simd.device_lockstep;
        if device != 0 {
            tally(&mut out, simd::DEVICE_LOCKSTEP.into(), device, 0);
        }
        out
    }
}

/// The sampling operations of strategy `s` that read separate bindings: none unless the stage takes
/// its combined bindings apart.
fn separate_samplings(families: &FamilyPlans, s: &Strategy) -> u32 {
    match s {
        Strategy::Texture(t) if families.sampler.separate => t.combined_samplings(),
        _ => 0,
    }
}

/// Adds `uses` and `words` to the census entry `name`, allocating the name only for a new entry.
fn tally(out: &mut BTreeMap<String, (u32, u64)>, name: Cow<'static, str>, uses: u32, words: u64) {
    if let Some(e) = out.get_mut(name.as_ref()) {
        e.0 = e.0.saturating_add(uses);
        e.1 = e.1.saturating_add(words);
    } else {
        out.insert(name.into_owned(), (uses, words));
    }
}

/// What families read and contribute while selecting.
pub struct SelectCx<'a> {
    pub entry: &'a Entry,
    pub body: &'a Body,
    pub memory: &'a MemoryPlan,
    pub profile: &'a DeviceProfile,
    pub inputs: &'a LowerInputs,
    pub stage: Stage,
    pub values: ValueIndex,
    /// Push fields the function's strategies read.
    pub push: BTreeSet<PushField>,
    /// Runtime facts the function depends on, on the ledger as [`charge_facts`] charges them.
    pub facts: BTreeSet<RuntimeFact>,
    pub precision: BTreeSet<PrecisionNote>,
    /// The notes accesses keep only on the arms the module selects by a slot's specialization,
    /// each with its clauses ([`SelectCx::note_when`]); a note `precision` also holds is
    /// unconditional.
    pub(crate) specialized: BTreeMap<PrecisionNote, BTreeSet<NoteClause>>,
    /// The ledger bytes `specialized` holds, charged as [`SelectCx::note_when`] keeps each clause
    /// and released once `finish` has resolved them.
    noted: usize,
    /// The descriptors the families' selections use, per kind (checked against the per-stage
    /// limits once the access family has chosen its slot forms).
    pub descriptors: DescriptorCounts,
    /// The function's simdgroup class; the simd family raises it, `select` tiers it.
    pub simd: SimdClass,
    /// Tangled instructions (subgroup/quad ops, derivatives, implicit-LOD samples, barriers),
    /// which the families that select them add.
    pub tangled: BTreeSet<ValueId>,
    /// Each family's own plan.
    pub families: FamilyPlans,
    /// The offset shapes the access family has read while selecting this function's typed
    /// accesses, each definition's once; released when the instruction walk ends.
    pub(crate) shapes: access::local::Shapes,
    /// What each root set a buffer pointer of the function names holds, read once per set before
    /// any instruction is selected.
    pub(crate) root_sets: access::root_sets::RootSetFacts,
    /// What each pointer phi's incoming values are, read once per phi before any instruction is
    /// selected.
    pub(crate) phis: access::phis::PhiFacts,
    func: FuncId,
    /// Whether each instruction's result is observed, by `InstId` ([`observed_results`]).
    observed: Vec<bool>,
}

impl SelectCx<'_> {
    /// Record `note` for an access that keeps its near-equivalent only where one of `when`'s
    /// clauses holds: each clause the arms of one specialization select. An empty `when` records
    /// nothing (the access is exact under every specialization); a clause with no test holds
    /// always, so the note is unconditional, and a note already unconditional keeps no clause. Each
    /// clause kept for the first time is charged to `m` while selection holds it
    /// ([`clause_bytes`]): a switch over many candidate slots keeps clauses for each.
    pub(crate) fn note_when(
        &mut self,
        note: PrecisionNote,
        when: Vec<NoteClause>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        if when.iter().any(BTreeSet::is_empty) {
            self.precision.insert(note);
            return Ok(());
        }
        if when.is_empty() || self.precision.contains(&note) {
            return Ok(());
        }
        let clauses = match self.specialized.entry(note) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                let bytes = tree_bytes(
                    1,
                    std::mem::size_of::<(PrecisionNote, BTreeSet<NoteClause>)>(),
                );
                m.charge(bytes)?;
                self.noted = self.noted.saturating_add(bytes);
                e.insert(BTreeSet::new())
            }
        };
        for clause in when {
            if clauses.contains(&clause) {
                continue;
            }
            let bytes = clause_bytes(&clause);
            m.charge(bytes)?;
            self.noted = self.noted.saturating_add(bytes);
            clauses.insert(clause);
        }
        Ok(())
    }

    /// The AIR type of `v` in the entry function.
    pub fn value_ty(&self, v: Value) -> Option<TypeId> {
        self.entry.module.value_ty(self.func, v)
    }

    /// `v` as a constant lowering materializes (`lower::arith::constant`).
    pub fn constant(&self, v: Value) -> Option<arith::LoweredConst> {
        arith::constant(self.entry, v)
    }

    /// The lowered representation of value `v` of AIR type `ty`: a pointer's is the access
    /// family's, every other value's the arith family's. Both families own their legality, so
    /// this routing never changes when they grow.
    pub fn lowered_repr(&self, v: Value, ty: TypeId) -> Result<DemoteTy, Error> {
        // A member-wise phi is the composite of its members' representations.
        if let Some(d) = arith::aggregate::phi_repr(&self.families.arith, v) {
            return Ok(d.clone());
        }
        // An intersect result is the ray tracing family's record.
        if let Some(record) = raytracing::result_repr(self, v) {
            return record;
        }
        if matches!(self.entry.module.types().get(ty), Some(Type::Ptr { .. })) {
            access::lowered_pointer(self, v)
        } else if let Some(word) = access::index::word_repr(&self.families.access, v) {
            // An index word is held as its low word.
            Ok(word)
        } else if access::handle::is_word(&self.families.access, v) {
            // A handle word is held as its handle index.
            Ok(DemoteTy::HandleIndex)
        } else if let Some(p) = access::cell::word_pointer(self, v) {
            // A pointer word from a forwarding cell is held as its pointer.
            access::lowered_pointer(self, p)
        } else {
            arith::lowered_type(self.entry, self.profile, ty)
        }
    }

    /// An internal invariant failure.
    pub fn invariant(&self, detail: &str) -> Error {
        invariant(detail)
    }

    /// The entry's binding of parameter `v`, if `v` is a parameter the entry binds: found through
    /// the memory plan's index, as [`param_binding`] finds it.
    pub fn param(&self, v: Value) -> Option<&ParamBinding> {
        let Value::Arg(a) = v else { return None };
        self.memory.abi.binding(self.entry, a.raw())
    }

    /// Whether the result of instruction `id` is observed: read by an instruction or terminator
    /// other than a phi, or by a phi whose result is observed ([`observed_results`]).
    pub fn is_observed(&self, id: InstId) -> bool {
        self.observed
            .get(id.raw() as usize)
            .copied()
            .unwrap_or(true)
    }
}

/// Whether each instruction's result is observed, by `InstId`: read by an instruction or
/// terminator of the body other than a phi, or by a phi whose own result is observed. Decode
/// drops `llvm.assume`, so a result that fed only assumptions, directly or through phis, is
/// unobserved. One pass over the body's instructions, then one over the phis it reaches; the bitmap
/// is charged to `m` and released by the caller (`insts.len()` bytes).
fn observed_results(body: &Body, m: &mut Meter) -> Result<Vec<bool>, Error> {
    let n = body.insts.len();
    let work_bytes = n.saturating_mul(std::mem::size_of::<InstId>());
    m.charge(n.saturating_add(work_bytes))?;
    let mut observed = vec![false; n];
    let mut phis = Vec::new();
    for (_, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            match body.insts.get(id).map(|inst| &inst.kind) {
                Some(InstKind::Phi { .. }) => phis.push(id),
                Some(kind) => kind.for_each_value(|v| {
                    observe(&mut observed, v);
                }),
                None => {}
            }
        }
    }
    // An observed phi observes its incoming values; a phi is queued at most once, when it first
    // becomes observed (or at the start, when an instruction observes it).
    let mut work: Vec<InstId> = phis
        .into_iter()
        .filter(|p| observed.get(p.raw() as usize).copied().unwrap_or(false))
        .collect();
    while let Some(p) = work.pop() {
        m.tick(1)?;
        let Some(InstKind::Phi { incoming, .. }) = body.insts.get(p).map(|inst| &inst.kind) else {
            continue;
        };
        for &(v, _) in incoming.iter() {
            if let (true, Value::Inst(i)) = (observe(&mut observed, v), v) {
                if matches!(
                    body.insts.get(i).map(|inst| &inst.kind),
                    Some(InstKind::Phi { .. })
                ) {
                    work.push(i);
                }
            }
        }
    }
    m.release(work_bytes);
    Ok(observed)
}

/// Mark `v` observed if it is an instruction's result; whether it was not already.
fn observe(observed: &mut [bool], v: Value) -> bool {
    let Value::Inst(i) = v else { return false };
    match observed.get_mut(i.raw() as usize) {
        Some(o) if !*o => {
            *o = true;
            true
        }
        _ => false,
    }
}

/// The entry's binding of parameter `v`.
pub fn param_binding(e: &Entry, v: Value) -> Option<&ParamBinding> {
    let Value::Arg(a) = v else { return None };
    e.ep.params.iter().find(|b| b.param == a.raw())
}

/// The family that owns a parameter of role `role` in a `stage` entry (its planning, its value,
/// and its module-scope objects), or the refusal of a role whose family is not implemented yet.
pub fn param_family(stage: Stage, role: &ParamRole) -> Result<LowerFamily, Error> {
    let refuse =
        |f: Feature| -> Result<LowerFamily, Error> { Err(Unsupported::not_implemented(f).into()) };
    Ok(match role {
        ParamRole::Buffer(_)
        | ParamRole::ArgumentBuffer(_)
        | ParamRole::Threadgroup(_)
        | ParamRole::IndirectConstant { .. } => LowerFamily::Access,
        ParamRole::Texture(_) => LowerFamily::Texture,
        ParamRole::Sampler { .. } => LowerFamily::Sampler,
        // The stage-in region's builtins go with the stage inputs.
        ParamRole::Builtin(Builtin::StageInGridOrigin | Builtin::StageInGridSize)
            if stage == Stage::Kernel =>
        {
            LowerFamily::StageIo
        }
        // A mesh function's amplification is a render stage's: its id from the push word the
        // binder writes for each amplification it draws, its count the specialization constant.
        ParamRole::Builtin(Builtin::AmplificationId | Builtin::AmplificationCount)
            if stage == Stage::Mesh =>
        {
            LowerFamily::StageIo
        }
        // An object or mesh function's builtins are a kernel's.
        ParamRole::Builtin(_) if matches!(stage, Stage::Kernel | Stage::Object | Stage::Mesh) => {
            LowerFamily::Dispatch
        }
        // A fragment's lane in its 2x2 quad is its subgroup lane's, its lane in its simdgroup the
        // subgroup's where that is Metal's, and its simdgroup's width 32.
        ParamRole::Builtin(
            Builtin::ThreadIndexInQuadgroup
            | Builtin::ThreadIndexInSimdgroup
            | Builtin::ThreadsPerSimdgroup,
        ) if stage == Stage::Fragment => LowerFamily::Simd,
        ParamRole::Builtin(_)
        | ParamRole::StageIn { .. }
        | ParamRole::VertexInput { .. }
        | ParamRole::FragmentInput(_)
        | ParamRole::PatchInput(_) => LowerFamily::StageIo,
        ParamRole::ColorInput { .. } => LowerFamily::Fragment,
        // A kernel's explicit imageblock is threadgroup staging, reached only through its
        // intrinsics, which refuse what it cannot model; the implicit one is tile shading.
        ParamRole::Imageblock(ib) if ib.kind == ImageblockKind::Implicit || ib.alias_implicit => {
            return refuse(Feature::TileShading)
        }
        ParamRole::Imageblock(ib)
            if stage == Stage::Kernel && ib.kind == ImageblockKind::Explicit =>
        {
            LowerFamily::Access
        }
        // A fragment function's imageblock view is the pass's imageblock storage.
        ParamRole::Imageblock(ib)
            if stage == Stage::Fragment && ib.kind == ImageblockKind::Data =>
        {
            LowerFamily::Fragment
        }
        ParamRole::Imageblock(_) => return refuse(Feature::Imageblocks),
        ParamRole::AccelerationStructure { .. } => LowerFamily::RayTracing,
        ParamRole::FunctionTable(_) => return refuse(Feature::DynamicCall),
        // Its only readers are intersect calls, which read its words.
        ParamRole::IntersectionTable(_) => LowerFamily::RayTracing,
        ParamRole::CommandBuffer(_)
        | ParamRole::ComputePipelineState(_)
        | ParamRole::RenderPipelineState(_) => return refuse(Feature::ArgumentBufferObjects),
        // An argument buffer's member alone.
        ParamRole::DepthStencilState(_) => return refuse(Feature::IndirectRenderState),
        ParamRole::MeshGridProperties | ParamRole::Mesh(_) => LowerFamily::Mesh,
        // An object or mesh function's payload is an arena (`access::local`).
        ParamRole::Payload { .. } if matches!(stage, Stage::Object | Stage::Mesh) => {
            LowerFamily::Access
        }
        ParamRole::Payload { .. } => return refuse(Feature::MeshShading),
    })
}

/// [`param_family`] for a parameter of entry `e`: a post-tessellation vertex function's builtins
/// and patch inputs are `lower::tessellation`'s.
pub fn entry_param_family(e: &Entry, stage: Stage, role: &ParamRole) -> Result<LowerFamily, Error> {
    if tessellation::owns_param(e, role) {
        return Ok(LowerFamily::Tessellation);
    }
    param_family(stage, role)
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

fn not_lowered(f: LowerFamily) -> Error {
    Unsupported::not_implemented(Feature::Lowering(f)).into()
}

/// The family an instruction routes to, or its refusal.
enum Route {
    Family(LowerFamily),
    /// Handled by the structurizer and `lower::emit` (phis, terminators).
    Control,
    Refuse(Error),
}

fn is_ptr(cx: &SelectCx<'_>, v: Value) -> bool {
    cx.value_ty(v)
        .and_then(|t| cx.entry.module.types().get(t))
        .is_some_and(|t| matches!(t, Type::Ptr { .. }))
}

/// The symbol of declaration `decl`, for a refusal that names it.
pub(crate) fn decl_name(cx: &SelectCx<'_>, decl: FuncId) -> DiagName {
    let module = &cx.entry.module;
    let bytes = module
        .func(decl)
        .and_then(|f| module.symbols().get(f.name))
        .unwrap_or(b"?");
    DiagName::new(String::from_utf8_lossy(bytes))
}

fn route_air(cx: &SelectCx<'_>, op: &AirIntrinsic, decl: FuncId) -> Route {
    use AirIntrinsic as A;
    match op {
        A::Math(_) | A::Convert(_) | A::Pack(_) => Route::Family(LowerFamily::Arith),
        A::Atomic(_) | A::Coherent(_) | A::AsyncCopy(_) => Route::Family(LowerFamily::Access),
        A::Texture(_) => Route::Family(LowerFamily::Texture),
        A::Sampler(_) => Route::Family(LowerFamily::Sampler),
        A::BoolReduce(_)
        | A::Group(_)
        | A::IsUniform(_)
        | A::SimdgroupSize(_)
        | A::SimdMatrix(_) => Route::Family(LowerFamily::Simd),
        A::Sync(_) | A::Derivative(_) => Route::Family(LowerFamily::Barrier),
        A::Fragment(_) => Route::Family(LowerFamily::Fragment),
        // A read of an argument the function constants removed, which specialization made.
        A::FunctionConstant(
            FunctionConstant::AbsentRead(_) | FunctionConstant::AbsentMemory(_),
        ) => Route::Family(LowerFamily::Access),
        A::FunctionConstant(_) => Route::Refuse(invariant(
            "a function-constant intrinsic survived canonicalization",
        )),
        // A kernel's explicit imageblock, and a tile function's implicit one.
        A::Imageblock(_) => Route::Family(LowerFamily::Access),
        // The null intersection table only feeds intersect calls.
        A::FunctionTable(FunctionTable::NullIntersection) | A::Intersect(_) => {
            Route::Family(LowerFamily::RayTracing)
        }
        // Filling an intersection function table's buffer slot.
        A::FunctionTable(FunctionTable::SetIntersectionBuffer) => Route::Refuse(
            Unsupported::not_implemented(Feature::IntersectionFunctionTable)
                .naming(decl_name(cx, decl))
                .into(),
        ),
        // A table's word read by its resource id, which devirtualization made.
        A::FunctionTable(FunctionTable::Word) => Route::Family(LowerFamily::Access),
        A::FunctionTable(_) => {
            Route::Refuse(Unsupported::not_implemented(Feature::DynamicCall).into())
        }
        // A structure's instance count is the binder's.
        A::AccelerationStructure(crate::air::intrinsics::AccelerationStructure::InstanceCount) => {
            Route::Family(LowerFamily::RayTracing)
        }
        A::AccelerationStructure(_) => Route::Refuse(
            Unsupported::not_implemented(Feature::RayTracing)
                .naming(decl_name(cx, decl))
                .into(),
        ),
        A::Tensor(_) => Route::Refuse(Unsupported::not_implemented(Feature::Tensors).into()),
        // An encoder operation writes its command's record through device addresses.
        A::IndirectCommand(_) => Route::Family(LowerFamily::Access),
        A::Mesh(_) => Route::Family(LowerFamily::Mesh),
        // Canonicalization made every read of a global built-in at a constant lane a read of a
        // built-in parameter, but a fragment's read of one Metal refuses there, which is
        // refused as its own feature; a read at another lane is not lowered.
        A::GlobalBuiltin(g) if cx.stage == Stage::Fragment && g.var.metal_refuses_in_fragment() => {
            Route::Refuse(
                Unsupported::not_implemented(Feature::GlobalBuiltinInFragment)
                    .naming(decl_name(cx, decl))
                    .into(),
            )
        }
        A::GlobalBuiltin(_) => Route::Refuse(
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::Dispatch)).into(),
        ),
    }
}

/// Exhaustive, so a newly decoded variant is a compile error here rather than a silent route.
/// The memory markers (`llvm.lifetime.*`, `llvm.assume`, `llvm.experimental.noalias.scope.decl`)
/// never reach it: decode drops them (`air::intrinsics::Dropped`; dropping a lifetime marker only
/// extends a lifetime), so they have no variant.
fn route_llvm(op: &LlvmIntrinsic) -> Route {
    use LlvmIntrinsic as L;
    match op {
        L::Memset { .. } | L::Memcpy { .. } | L::Memmove { .. } => {
            Route::Family(LowerFamily::Access)
        }
        L::Agx(_) => Route::Family(LowerFamily::Simd),
        L::Float { .. } | L::Int { .. } | L::Abs { .. } | L::Cttz { .. } | L::Ctlz { .. } => {
            Route::Family(LowerFamily::Arith)
        }
    }
}

/// The total routing of instruction `id`.
fn route(cx: &SelectCx<'_>, id: InstId, inst: &Inst) -> Route {
    use crate::air::ir::Callee;
    // An intersection's primitive-data pointer is only moved; any other reader is refused.
    if let Some(e) = raytracing::primitive_data_reader(cx, id, inst) {
        return Route::Refuse(e);
    }
    match &inst.kind {
        InstKind::Phi { .. }
        | InstKind::Ret(_)
        | InstKind::Br { .. }
        | InstKind::CondBr { .. }
        | InstKind::Switch { .. }
        | InstKind::Unreachable => Route::Control,
        // A store of an intersection's primitive-data pointer is the ray tracing family's.
        InstKind::Store { val, .. } if raytracing::is_primitive_data(cx, *val) => {
            Route::Family(LowerFamily::RayTracing)
        }
        InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::Gep { .. }
        | InstKind::AtomicRmw { .. }
        | InstKind::CmpXchg { .. } => Route::Family(LowerFamily::Access),
        InstKind::Fence(_) => Route::Family(LowerFamily::MemoryModel),
        InstKind::Cast {
            op: CastOp::PtrToInt | CastOp::IntToPtr | CastOp::AddrSpaceCast,
            ..
        } => Route::Family(LowerFamily::Access),
        InstKind::Cast { v, .. } | InstKind::Freeze(v) if is_ptr(cx, *v) => {
            Route::Family(LowerFamily::Access)
        }
        InstKind::ICmp { a, .. } if is_ptr(cx, *a) => Route::Family(LowerFamily::Access),
        InstKind::Select { t, .. } if is_ptr(cx, *t) => Route::Family(LowerFamily::Access),
        // A field of an intersect result is the ray tracing family's, and so is its primitive-data
        // pointer carried through an aggregate.
        InstKind::ExtractValue { agg, .. }
            if raytracing::is_intersect_result(cx.body, *agg)
                || raytracing::is_primitive_data(cx, Value::Inst(id)) =>
        {
            Route::Family(LowerFamily::RayTracing)
        }
        InstKind::BinOp { .. }
        | InstKind::FNeg { .. }
        | InstKind::Cast { .. }
        | InstKind::ICmp { .. }
        | InstKind::FCmp { .. }
        | InstKind::Select { .. }
        | InstKind::Freeze(_)
        | InstKind::ExtractElement { .. }
        | InstKind::InsertElement { .. }
        | InstKind::ShuffleVector { .. }
        | InstKind::ExtractValue { .. }
        | InstKind::InsertValue { .. } => Route::Family(LowerFamily::Arith),
        InstKind::Call(call) => match &call.callee {
            Callee::Air { op, decl } => route_air(cx, op, *decl),
            Callee::Llvm { op, .. } => route_llvm(op),
            // Canon's post-check refuses every indirect call by what it calls.
            Callee::Indirect { .. } => Route::Refuse(invariant("an indirect call survived canon")),
            // The one direct call canonicalization keeps: a post-tessellation entry's
            // control-point fetch.
            Callee::Direct(f) if tessellation::is_control_point_fetch(cx.entry, *f) => {
                Route::Family(LowerFamily::Tessellation)
            }
            Callee::Direct(_) => Route::Refuse(invariant("a direct call survived inlining")),
            Callee::UnknownIntrinsic(_) => {
                Route::Refuse(Unsupported::not_implemented(Feature::UnknownIntrinsic).into())
            }
        },
    }
}

/// Select the family strategy of `inst`.
fn select_inst(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    family: LowerFamily,
    m: &mut Meter,
) -> Result<Strategy, Error> {
    Ok(match family {
        LowerFamily::Arith => Strategy::Arith(arith::select(cx, at, inst, m)?),
        LowerFamily::Access => Strategy::Access(access::select(cx, at, inst, m)?),
        LowerFamily::Texture => Strategy::Texture(texture::select(cx, at, inst, m)?),
        LowerFamily::Sampler => Strategy::Sampler(sampler::select(cx, at, inst)?),
        LowerFamily::Simd => Strategy::Simd(simd::select(cx, at, inst, m)?),
        LowerFamily::Barrier => Strategy::Barrier(sync::barrier::select(cx, at, inst)?),
        LowerFamily::MemoryModel => {
            Strategy::MemoryModel(sync::memory_model::select(cx, at, inst)?)
        }
        LowerFamily::Fragment => Strategy::Fragment(fragment::select(cx, at, inst)?),
        LowerFamily::Tessellation => Strategy::Tessellation(tessellation::select(cx, at, inst)?),
        LowerFamily::RayTracing => Strategy::RayTracing(raytracing::select(cx, at, inst, m)?),
        LowerFamily::Mesh => Strategy::Mesh(mesh::select(cx, at, inst)?),
        LowerFamily::StageIo | LowerFamily::Dispatch => {
            return Err(invariant("an instruction routed to an interface family"))
        }
    })
}

/// Plan every used parameter through its family ([`param_family`]). A family's `select_param`
/// succeeds only for a parameter whose value its `param_value` then produces, in the value's
/// lowered representation (`SelectCx::lowered_repr`), wherever an instruction or terminator reads
/// it.
fn select_params(cx: &mut SelectCx<'_>, used: &[bool]) -> Result<(), Error> {
    let entry = cx.entry;
    // Every argument without a binding is one a function-constant predicate removed, and
    // canonicalization gave each read of it its absent value or refused it, but for the removed
    // textures and samplers it kept, which are the absent handle and need no plan of their own.
    let unbound = used.iter().enumerate().any(|(i, u)| {
        *u && !entry.ep.params.iter().any(|b| b.param as usize == i)
            && !entry.absent_handles.iter().any(|b| b.param as usize == i)
    });
    if unbound {
        return Err(invariant("a read of an argument without a binding"));
    }
    for b in &entry.ep.params {
        if !used.get(b.param as usize).copied().unwrap_or(false) {
            continue;
        }
        select_param(cx, b).map_err(|e| located_param(e, cx, Some(b.param)))?;
    }
    Ok(())
}

/// Plan the used parameter `b` through its family ([`entry_param_family`]).
fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    match entry_param_family(cx.entry, cx.stage, &b.role)? {
        LowerFamily::Access => access::select_param(cx, b),
        LowerFamily::Texture => texture::select_param(cx, b),
        LowerFamily::Sampler => sampler::select_param(cx, b),
        LowerFamily::Dispatch => dispatch::select_param(cx, b),
        LowerFamily::StageIo => stage_io::select_param(cx, b),
        LowerFamily::Fragment => fragment::select_param(cx, b),
        LowerFamily::Tessellation => tessellation::select_param(cx, b),
        LowerFamily::RayTracing => raytracing::select_param(cx, b),
        LowerFamily::Simd => simd::select_param(cx, b),
        LowerFamily::Mesh => mesh::select_param(cx, b),
        LowerFamily::Arith | LowerFamily::Barrier | LowerFamily::MemoryModel => Err(invariant(
            "a parameter routed to an instruction-only family",
        )),
    }
}

/// The smallest `u32` that is no literal of `literals`. One exists whenever fewer than 2^32
/// literals fit 32 bits (the structurizer refuses wider ones, `WideSwitchLiteral`).
pub fn miss_literal(literals: &BTreeSet<u64>) -> Option<u32> {
    let mut miss = 0u64;
    for &l in literals {
        if l == miss {
            miss += 1;
        } else if l > miss {
            break;
        }
    }
    u32::try_from(miss).ok()
}

/// The 32-bit selector form of a switch selector of AIR type `air` whose lowered representation
/// (`lower::arith::lowered_type`) is `lowered` and whose switches have `literals`.
///
/// Each form assumes the lowered value holds the AIR integer zero-extended, so its bits are the
/// case literals': an `i8`/`i16` held at its own width, an `Int(2..=32)` in the 32-bit container,
/// an `Int(33..=64)` in the 64-bit one, or, without `shaderInt64`, as two words (a `uvec2`, low
/// word first). Any other pairing of AIR type and representation is not a form this function knows
/// and is refused until one is added for it.
pub fn selector_form(
    air: &Type,
    lowered: &DemoteTy,
    int64: bool,
    literals: &BTreeSet<u64>,
) -> Result<SelectorForm, Error> {
    let words = DemoteTy::Vector {
        elem: ScalarTy::Int { bits: 32 },
        len: 2,
    };
    if matches!(air, Type::Int(33..=64)) && !int64 && *lowered == words {
        let miss = miss_literal(literals).ok_or_else(|| not_lowered(LowerFamily::Arith))?;
        return Ok(SelectorForm::Words { miss });
    }
    let DemoteTy::Scalar(repr) = lowered else {
        return Err(not_lowered(LowerFamily::Arith));
    };
    match (air, *repr) {
        (Type::Int(1), ScalarTy::Bool) => Ok(SelectorForm::Bool),
        (Type::Int(w @ (8 | 16)), ScalarTy::Int { bits }) if bits == *w => {
            Ok(SelectorForm::ZeroExtend)
        }
        // Every integer up to 32 bits is held zero-extended in a `u32`, and case literals are
        // zero-extended bit patterns: the container is the selector.
        (Type::Int(2..=32), ScalarTy::Int { bits: 32 }) => Ok(SelectorForm::Direct),
        (Type::Int(33..=64), ScalarTy::Int { bits: 64 }) => {
            if !int64 {
                return Err(Unsupported::needs(
                    Feature::Int64Arithmetic,
                    Capability::Feature(VkFeature::ShaderInt64),
                )
                .into());
            }
            let miss = miss_literal(literals).ok_or_else(|| not_lowered(LowerFamily::Arith))?;
            Ok(SelectorForm::Narrow64 { miss })
        }
        _ => Err(not_lowered(LowerFamily::Arith)),
    }
}

/// Whether `p`'s device holds `e`'s private memory: what a driver backs with scratch, the plan's
/// `private_bytes` (the private arena, and each typed `alloca`, call stacks among them, and each
/// constant held as a `Private` variable that an access indexes at run time), may take a thread
/// up to `device::function_storage::ceiling`: the profile's `function_storage`, or Metal's 192 KiB
/// for a vertex or fragment function on a device not measured. Past it the entry is
/// `Unsupported{PrivateMemory}`, or `Unsupported{Recursion}` with a call stack, needing more.
/// Past the profile's own value (a device not measured holds the floor, 16 KiB) an entry within
/// that ceiling records `PrecisionNote::FunctionStorageAssumed`, which it returns. A private arena
/// past its `u32` offsets is first the translator's own bound, a property of its encoding that
/// selection refuses at the arena's first pointer (`LimitExceeded(PrivateArenaBytes)`), so the
/// device's storage is not asked.
fn function_storage(
    e: &Entry,
    mp: &MemoryPlan,
    p: &DeviceProfile,
) -> Result<Option<PrecisionNote>, Error> {
    let required = mp.private_bytes;
    let storage = p.fields().function_storage;
    if mp.priv_arena.static_size > crate::budget::PRIVATE_ARENA_BYTES {
        return Ok(None);
    }
    let available = function_storage::ceiling(storage, static_root::stage_of(e));
    if required > available {
        let feature = match e.call_stack {
            Some(_) => Feature::Recursion,
            None => Feature::PrivateMemory,
        };
        let limit = Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required,
            available,
        };
        return Err(Unsupported::needs(feature, limit).into());
    }
    Ok((required > storage.bytes()).then_some(PrecisionNote::FunctionStorageAssumed))
}

/// Select the plan of `e`.
pub fn select(
    e: &Entry,
    mp: &MemoryPlan,
    p: &DeviceProfile,
    inputs: &LowerInputs,
    m: &mut Meter,
) -> Result<Plan, Error> {
    let func = e
        .module
        .func(e.func)
        .ok_or_else(|| invariant("the entry function is missing"))?;
    let body = func
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry function has no body"))?;
    let storage_note = function_storage(e, mp, p)?;
    let stage = static_root::stage_of(e);
    let observed = observed_results(body, m)?;
    let observed_bytes = observed.len();
    let root_sets = match access::root_sets::plan(e, mp, m) {
        Ok(r) => r,
        Err(err) => {
            m.release(observed_bytes);
            return Err(err);
        }
    };
    let root_bytes = root_sets.charged();
    let cx = SelectCx {
        entry: e,
        body,
        memory: mp,
        profile: p,
        inputs,
        stage,
        values: mp.values,
        push: BTreeSet::new(),
        // A call stack's depth, which a compiled kernel's pipeline must match.
        facts: (e.call_stack.into_iter())
            .map(|stack| RuntimeFact::CallStackDepth { depth: stack.depth })
            .collect(),
        // Private memory past the floor of a device not measured rests on no measurement, as
        // does a group operation in a branch no measurement says Metal speculates. A rate-map
        // decoder canon compiled is Apple's over Metal's bytes: no note.
        precision: (storage_note.into_iter())
            .chain(
                e.unmeasured_speculation
                    .then_some(PrecisionNote::UnmeasuredGroupSpeculation),
            )
            .collect(),
        specialized: BTreeMap::new(),
        noted: 0,
        descriptors: DescriptorCounts::default(),
        simd: SimdClass::None,
        tangled: BTreeSet::new(),
        families: FamilyPlans::default(),
        shapes: access::local::Shapes::default(),
        root_sets,
        phis: access::phis::PhiFacts::default(),
        func: e.func,
        observed,
    };
    let result = select_entry(cx, body, func.params.len(), m);
    m.release(observed_bytes);
    m.release(root_bytes);
    result
}

fn select_entry(
    mut cx: SelectCx<'_>,
    body: &Body,
    params: usize,
    m: &mut Meter,
) -> Result<Plan, Error> {
    // The varyings a vertex function writes the same at every vertex of a primitive, which a
    // fragment function of its pipeline reads flat where it interpolates them, and which decide
    // where it stores the slice and viewport indices it writes.
    if cx.stage == Stage::Vertex {
        cx.families.stage_io.interface.uniform_outputs =
            stage_io::uniform_varyings(cx.entry, cx.inputs, m)?
                .into_iter()
                .collect();
    }
    tessellation::plan_entry(&mut cx).map_err(|e| located_param(e, &cx, None))?;
    // The framebuffer-fetch forms first: an output to a storage-form attachment is no interface
    // variable.
    fragment::plan_entry(&mut cx).map_err(|e| located_param(e, &cx, None))?;
    stage_io::plan_entry(&mut cx).map_err(|e| located_param(e, &cx, None))?;
    mesh::plan_entry(&mut cx).map_err(|e| located_param(e, &cx, None))?;

    // Uses: which parameters are read, and the block of every use (for demotion).
    let n = cx.values.len();
    let bytes = n.saturating_mul(std::mem::size_of::<Option<Strategy>>() + 8);
    m.charge(bytes)?;
    // The pointer phis' incoming values, once per phi, before any use reads them.
    let phis = match access::phis::plan(&cx, m) {
        Ok(p) => p,
        Err(err) => {
            m.release(bytes);
            return Err(err);
        }
    };
    let phi_bytes = phis.charged();
    cx.phis = phis;
    // The facts the function holds, on the ledger from the walk to the plan ([`charge_facts`]):
    // a function sampling many slots holds a fact for each.
    let mut fact_bytes = Growth::default();
    let selected = select_body(&mut cx, body, params, &mut fact_bytes, m);
    // Only the instruction walk reads typed-access shapes.
    cx.shapes.release(m);
    // The specialized notes' clauses are held until `finish` resolves them, and the facts until it
    // moves them into the plan.
    let noted = cx.noted;
    let result = selected.and_then(|sel| finish(cx, sel, &mut fact_bytes, m));
    m.release(noted);
    m.release(fact_bytes.charged());
    m.release(bytes);
    m.release(phi_bytes);
    result
}

/// Charge to `m` the growth of the facts `cx` holds since `held` last charged them (`tree_bytes`
/// of each entry).
fn charge_facts(cx: &SelectCx<'_>, held: &mut Growth, m: &mut Meter) -> Result<(), Error> {
    let footprint = tree_bytes(cx.facts.len(), std::mem::size_of::<RuntimeFact>());
    held.charge_to(footprint, m)?;
    Ok(())
}

/// What the instruction walk produced.
struct BodySelection {
    strategy: Vec<Option<Strategy>>,
    selectors: BTreeMap<ValueId, SelectorForm>,
    cross: Vec<(ValueId, DemoteTy)>,
    phis: Vec<(ValueId, DemoteTy)>,
    blocks: u64,
}

fn located(e: Error, cx: &SelectCx<'_>, block: u32, inst: InstId) -> Error {
    match e {
        Error::Unsupported(u) if !u.is_placed() => Error::Unsupported(u.placed_at(
            Location::function(cx.entry.ep.name.clone(), Some(block), Some(inst.raw())),
        )),
        other => other,
    }
}

/// An interface refusal without a location, located at the entry's ABI metadata: at argument
/// `arg` for a parameter's, else at the entry (its outputs and execution modes). One tied to a
/// render pipeline attachment keeps it there.
fn located_param(e: Error, cx: &SelectCx<'_>, arg: Option<u32>) -> Error {
    match e {
        Error::Unsupported(u) if !u.is_placed() => Error::Unsupported(u.placed_at(
            Location::metadata(Some(cx.entry.ep.name.clone()), arg, None),
        )),
        other => other,
    }
}

fn select_body(
    cx: &mut SelectCx<'_>,
    body: &Body,
    params: usize,
    fact_bytes: &mut Growth,
    m: &mut Meter,
) -> Result<BodySelection, Error> {
    let values = cx.values;
    let mut strategy: Vec<Option<Strategy>> = vec![None; values.len()];
    let mut used = vec![false; params];
    // The defining block of every instruction, and whether it yields a lowered SSA value.
    let mut def_block: Vec<u32> = vec![u32::MAX; body.insts.len()];
    let mut switch_literals: BTreeMap<Value, BTreeSet<u64>> = BTreeMap::new();
    let mut phis = Vec::new();
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            if let Some(d) = def_block.get_mut(id.raw() as usize) {
                *d = b.raw();
            }
        }
    }
    for inst in body.insts.values() {
        inst.kind.for_each_value(|v| {
            if let Value::Arg(a) = v {
                if let Some(u) = used.get_mut(a.raw() as usize) {
                    *u = true;
                }
            }
        });
    }
    // The index extensions and words, before any family asks for them.
    access::index::plan_extensions(cx, m)?;
    // The handle words, before any use of one is selected.
    access::handle::plan_words(cx, m)?;
    // The spin reads, before any atomic is selected.
    access::atomic::spin::plan(cx, m)?;
    // The indirect-command object words, before any use of one is selected.
    access::encode::plan_objects(cx, m)?;
    // The encodings each pointer-cell slot group's loads read, before any store.
    access::cell::plan_reads(cx, m)?;
    // The primitive-data stores nothing reads back, before any store.
    raytracing::plan_unread_stores(cx, m)?;
    select_params(cx, &used)?;
    // The member-wise phis' representations, before any use of one is selected.
    arith::aggregate::plan_phis(cx, m)?;
    // A blended extended-range output, whose blend the function computes by the device's
    // arithmetic.
    arith::output_precision::note_extended_range_sources(cx)?;
    // The `half` chains' operations held at `float` precision, before any of them is selected.
    cx.families.arith.held_at_float = arith::half_chain::plan(cx, m)?;

    let mut blocks = 0u64;
    for (b, block) in body.blocks.iter() {
        blocks += 1;
        for id in block.inst_ids() {
            m.tick(1)?;
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| invariant("a block names a missing instruction"))?;
            let vid = values
                .inst(id)
                .ok_or_else(|| invariant("an instruction out of the value range"))?;
            let outcome = match route(cx, id, inst) {
                Route::Refuse(e) => Err(e),
                Route::Family(f) => select_inst(cx, id, inst, f, m).map(|s| {
                    if let Some(slot) = strategy.get_mut(vid.index()) {
                        *slot = Some(s);
                    }
                }),
                Route::Control => match &inst.kind {
                    InstKind::Phi { .. } => cx
                        .lowered_repr(Value::Inst(id), inst.ty)
                        .map(|t| phis.push((vid, t))),
                    InstKind::Ret(v) => stage_io::select_return(cx, id, *v),
                    InstKind::Switch { v, cases, .. } => {
                        switch_literals
                            .entry(*v)
                            .or_default()
                            .extend(cases.iter().map(|(l, _)| *l));
                        Ok(())
                    }
                    _ => Ok(()),
                },
            };
            outcome.map_err(|e| located(e, cx, b.raw(), id))?;
            charge_facts(cx, fact_bytes, m)?;
        }
    }

    // Each member-wise phi's value on each edge, assembled from values emission produces.
    {
        let produced = |v: Value| match v {
            Value::Const(_) => cx.constant(v).is_some(),
            Value::Arg(_) => {
                cx.param(v).is_some() || access::handle::absent_param(cx.entry, v).is_some()
            }
            Value::Inst(i) => {
                let valued = values
                    .inst(i)
                    .and_then(|id| strategy.get(id.index()))
                    .and_then(Option::as_ref)
                    .is_some_and(|s| s.yields_value());
                let phi = body
                    .insts
                    .get(i)
                    .is_some_and(|inst| matches!(inst.kind, InstKind::Phi { .. }));
                valued || phi
            }
        };
        let located_phi = |(e, phi): (Error, InstId)| match body
            .blocks
            .iter()
            .find(|(_, blk)| blk.inst_ids().any(|i| i == phi))
        {
            Some((b, _)) => located(e, cx, b.raw(), phi),
            None => e,
        };
        let incoming = arith::aggregate::plan_incoming(cx, &produced, m).map_err(located_phi)?;
        cx.families.arith.incoming = incoming;
    }
    // Operands of terminators must be values emission can produce, in their lowered
    // representation. A parameter reaching here was planned by its family (`select_params`
    // refuses every used parameter its family cannot produce).
    let available = |v: Value, strategy: &[Option<Strategy>]| -> Result<DemoteTy, Error> {
        let ty = cx
            .value_ty(v)
            .ok_or_else(|| invariant("a terminator operand without a type"))?;
        let produced = match v {
            Value::Const(_) => cx.constant(v).is_some(),
            Value::Arg(_) => {
                cx.param(v).is_some() || access::handle::absent_param(cx.entry, v).is_some()
            }
            Value::Inst(i) => {
                let lowered = values
                    .inst(i)
                    .and_then(|id| strategy.get(id.index()))
                    .is_some_and(Option::is_some);
                let phi = body
                    .insts
                    .get(i)
                    .is_some_and(|inst| matches!(inst.kind, InstKind::Phi { .. }));
                lowered || phi
            }
        };
        if !produced {
            return Err(not_lowered(LowerFamily::Arith));
        }
        cx.lowered_repr(v, ty)
    };
    let mut selectors = BTreeMap::new();
    for (b, block) in body.blocks.iter() {
        let Some(t) = block.last().and_then(|t| body.insts.get(t).map(|i| (t, i))) else {
            continue;
        };
        let (tid, term) = t;
        let check = |v: Value| -> Result<DemoteTy, Error> {
            available(v, &strategy).map_err(|e| located(e, cx, b.raw(), tid))
        };
        match &term.kind {
            InstKind::CondBr { c, .. } => {
                check(*c)?;
            }
            InstKind::Switch { v, .. } => {
                let lowered = check(*v)?;
                let air = cx
                    .value_ty(*v)
                    .and_then(|t| cx.entry.module.types().get(t))
                    .ok_or_else(|| invariant("a switch selector without a type"))?;
                let lits = switch_literals.get(v).cloned().unwrap_or_default();
                let int64 = cx.profile.fields().leaf.int64;
                let form = selector_form(air, &lowered, int64, &lits)
                    .map_err(|e| located(e, cx, b.raw(), tid))?;
                let id = values
                    .id(*v)
                    .ok_or_else(|| invariant("a switch selector out of range"))?;
                selectors.insert(id, form);
            }
            _ => {}
        }
    }

    // Block-crossing values: an instruction result used outside its block; a phi's incoming
    // value is used at the end of its incoming block (the parallel copy). What an instruction or a
    // copy reads is `FamilyPlans::reads` and `FamilyPlans::incoming_reads` (an index extension's
    // narrow operand; the value a forwarding `extractvalue` forwards; a member-wise phi's members),
    // the hooks `lower::emit` imports and the predicated walk demotes and reloads through. A result
    // whose strategy yields no value (the read sampler; an aggregate that holds a pointer) has
    // nothing to carry: each of its users was selected above, and either ignores it (a texture
    // read, an `extractvalue` that the memory analysis resolves) or was refused.
    let mut cross: BTreeMap<ValueId, DemoteTy> = BTreeMap::new();
    let index = cx.families.read_index(body.insts.len());
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            let mut uses: Vec<(Value, u32)> = Vec::new();
            match &inst.kind {
                InstKind::Phi { incoming, .. } => {
                    for (v, from) in incoming.iter() {
                        cx.families
                            .incoming_reads(body, values, (id, *v, *from), |r| {
                                uses.push((r, from.raw()))
                            });
                    }
                }
                _ => cx
                    .families
                    .reads_indexed(&index, body, id, inst, |r| uses.push((r, b.raw()))),
            }
            for (v, at) in uses {
                let Value::Inst(d) = v else { continue };
                let home = def_block.get(d.raw() as usize).copied().unwrap_or(u32::MAX);
                if home == at {
                    continue;
                }
                let Some(def) = body.insts.get(d) else {
                    continue;
                };
                let id = values
                    .inst(d)
                    .ok_or_else(|| invariant("a use out of the value range"))?;
                let valueless = strategy
                    .get(id.index())
                    .and_then(Option::as_ref)
                    .is_some_and(|s| !s.yields_value());
                if valueless {
                    continue;
                }
                let ty = cx
                    .lowered_repr(v, def.ty)
                    .map_err(|e| located(e, cx, b.raw(), d))?;
                cross.insert(id, ty);
            }
        }
    }
    phis.sort();
    Ok(BodySelection {
        strategy,
        selectors,
        cross: cross.into_iter().collect(),
        phis,
        blocks,
    })
}

fn finish(
    mut cx: SelectCx<'_>,
    mut sel: BodySelection,
    fact_bytes: &mut Growth,
    m: &mut Meter,
) -> Result<Plan, Error> {
    let p = cx.profile;
    let stage = cx.stage;
    // Each family completes its own plan and contributes its descriptors, facts, simd class,
    // and tangled instructions before the combined checks below.
    stage_io::finish(&mut cx)?;
    if stage == Stage::Vertex {
        cx.families.stage_io.interface.instanced_attributes =
            cx.inputs.instanced_attributes.iter().copied().collect();
    }
    fragment::finish(&mut cx, m)?;
    access::tile::finish(&mut cx)?;
    texture::finish(&mut cx)?;
    sampler::finish(&mut cx)?;
    // The bound texture and sampler values the function reads as data.
    crate::lower::specialized::request(&mut cx);
    // The lane reads where some lane may not take part (a kernel's outside workgroup-uniform
    // control flow, every one of another stage), whose subgroup-tier form needs `Ballot`, before
    // the simd class selects the tier; the emulated tier reads a kernel's same analysis below.
    let uniformity = simd::partly_live(&mut cx, &sel.strategy, m)?;
    simd::finish(&mut cx)?;
    tessellation::finish(&mut cx)?;
    sync::barrier::finish(&mut cx)?;
    sync::memory_model::finish(&mut cx)?;
    arith::finish(&mut cx)?;

    let slots = access::finish_slots(&mut cx, m)?;
    access::absent::finish(&mut cx);
    sync::memory_model::finish_interlock(&mut cx, &slots)?;
    charge_facts(&cx, fact_bytes, m)?;
    let descriptors = cx.descriptors;
    if let Some((limit, required, available)) = descriptors.exceeded(&p.fields().limits) {
        return Err(Unsupported::needs(
            Feature::DescriptorCount,
            Capability::Limit {
                limit,
                required,
                available,
            },
        )
        .into());
    }

    stage_io::finish_merged(&mut cx)?;
    let placement = push::place(stage, &cx.push);
    access::accept_placement(&cx, &placement)?;

    // A kernel compiled for dispatches with a partial simdgroup emulates its simdgroups whatever
    // tier the profile's subgroups give them, and depends on that variant. A kernel whose lockstep
    // order is only exact on emulated simdgroups emulates them on every dispatch, which no variant
    // selects.
    let model = &p.fields().subgroup;
    let variant = cx.inputs.emulated_simdgroups && stage == Stage::Kernel;
    let tier = if variant || (cx.entry.lockstep.emulated && stage == Stage::Kernel) {
        if variant {
            cx.facts.insert(RuntimeFact::EmulatedSimdgroups);
        }
        subgroup::emulated_tier(&cx.simd, stage, model)?
    } else {
        subgroup::tier(&cx.simd, stage, model)?
    };
    if tier == Tier::Emulated {
        // The emulated tier's exchange carries its own active flags.
        cx.families.simd.partly_live.clear();
        simd::workgroup::check(
            &mut cx,
            &sel.strategy,
            &mut sel.cross,
            uniformity.as_ref(),
            m,
        )?;
    } else {
        // A workgroup barrier part of the workgroup skips is walked predicated in every tier.
        simd::divergent::barriers(
            &mut cx,
            &sel.strategy,
            &mut sel.cross,
            tier,
            uniformity.as_ref(),
            m,
        )?;
    }
    if let Some(u) = &uniformity {
        m.release(u.ledger_bytes());
    }
    // A fragment function's group operations run full quads, as Metal runs its helpers.
    simd::full_quads(&mut cx)?;
    let requirement = SubgroupRequirement {
        // A vertex function's subgroups its driver runs as Metal does without the stage listed: its
        // module needs that fact in place of the stage.
        metal_vertex_simdgroups: cx.families.simd.prefix_lane,
        ..subgroup::requirement(tier, stage, &p.fields().subgroup)
    };
    let dispatch = match stage {
        Stage::Kernel | Stage::Object | Stage::Mesh => Some(dispatch::plan_kernel(&mut cx, tier)?),
        Stage::Vertex | Stage::Fragment => None,
    };

    // The instructions the default presence specialization never runs, for the census alone.
    cx.families.texture.absent_arm =
        texture::absent_arm(cx.entry, cx.body, &sel.strategy, cx.values, m)?;

    let mut estimate = SizeEstimate::default();
    estimate.add(estimate::MODULE_WORDS);
    estimate.add(estimate::SLOT_WORDS.saturating_mul(descriptors.descriptors()));
    estimate.add(estimate::PUSH_FIELD_WORDS.saturating_mul(placement.fields.len() as u64));
    estimate.add(estimate::BLOCK_WORDS.saturating_mul(sel.blocks));
    estimate
        .add(estimate::VARIABLE_WORDS.saturating_mul((sel.cross.len() + sel.phis.len()) as u64));
    for s in sel.strategy.iter().flatten() {
        estimate.add(u64::from(s.words()));
        // The separate form's sampled images.
        let separate = separate_samplings(&cx.families, s);
        estimate.add(u64::from(separate) * u64::from(sampler::SEPARATE_WORDS));
        if tier == Tier::Emulated {
            estimate.add(u64::from(simd::workgroup::extra_words(s)));
        }
    }
    for v in &cx.families.simd.partly_live {
        if let Some(s) = sel.strategy.get(v.index()).and_then(Option::as_ref) {
            estimate.add(u64::from(simd::partly_live_words(s, tier)));
        }
    }
    if let Some(p) = &cx.families.simd.predication {
        estimate.add(simd::divergent::extra_words(
            cx.body,
            cx.values,
            &sel.strategy,
            p.set,
        ));
    }
    let access = access::routine_words(&cx.families.access);
    let heap = texture::routine_words(&cx.families.texture, &cx.families.access);
    let sampler = sampler::routine_words(&cx.families.sampler);
    for (_, words) in arith::routine_words(&cx.families.arith)
        .chain(access)
        .chain(heap)
        .chain(sampler)
    {
        estimate.add(words);
    }
    estimate.add(stage_io::estimate_words(&cx.families.stage_io));
    estimate.add(mesh::estimate_words(&cx.families.mesh));
    if dispatch.as_ref().is_some_and(|d| d.flat_grid) {
        estimate.add(dispatch::FLAT_GRID_WORDS);
    }
    if dispatch.as_ref().is_some_and(|d| d.folded_groups) {
        estimate.add(dispatch::FOLDED_GROUPS_WORDS);
    }
    estimate.add(tessellation::estimate_words(&cx.families.tessellation));
    estimate.add(
        cx.families
            .access
            .tile
            .as_ref()
            .map_or(0, access::tile::TilePlan::words_estimate),
    );

    let (precision, specialized) = resolve_notes(&cx.families, cx.precision, cx.specialized);
    Ok(Plan {
        stage,
        values: cx.values,
        strategy: sel.strategy,
        slots,
        push: placement,
        selectors: sel.selectors,
        cross: sel.cross,
        phis: sel.phis,
        tangled: cx.tangled,
        dispatch,
        simd: (cx.simd, tier),
        subgroup: requirement,
        descriptors,
        families: cx.families,
        facts: cx.facts.into_iter().collect(),
        precision,
        specialized,
        estimate,
        private_bytes: cx.memory.private_bytes,
    })
}

/// The constant a [`NoteTest`] reads: a sampler slot's `SamplerState` word, by the function's slot
/// index, or a texture slot's `UnmeasuredTexels` boolean, or the class of the format bound at a
/// texture slot, which no constant carries (`SpecKey::BoundTexelFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum NoteOn {
    Sampler(u32),
    Unmeasured(u32),
    Bound(u32),
}

/// A test of one slot's specialization constant `v`: it holds where `(v & mask == bits) == equal`,
/// as the declared constant's [`SpecTest`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NoteTest {
    pub on: NoteOn,
    pub mask: u32,
    pub bits: u32,
    pub equal: bool,
}

impl NoteTest {
    /// The test that `on`'s value masked by `mask` is `bits`.
    pub(crate) fn equal(on: NoteOn, mask: u32, bits: u32) -> NoteTest {
        NoteTest {
            on,
            mask,
            bits,
            equal: true,
        }
    }

    /// The test that `on`'s value masked by `mask` is not `bits`.
    pub(crate) fn differ(on: NoteOn, mask: u32, bits: u32) -> NoteTest {
        NoteTest {
            on,
            mask,
            bits,
            equal: false,
        }
    }
}

/// A conjunction of [`NoteTest`]s.
pub(crate) type NoteClause = BTreeSet<NoteTest>;

/// The tests one B-tree node holds: a set of tests allocates a whole node however few it holds.
const NODE_TESTS: usize = 11;

/// The ledger bytes of clause `c` held in a note's set: its entry in the set, and its own set of
/// tests, at least one whole node of [`NODE_TESTS`] tests.
fn clause_bytes(c: &NoteClause) -> usize {
    let test = std::mem::size_of::<NoteTest>();
    tree_bytes(1, std::mem::size_of::<NoteClause>())
        .saturating_add(tree_bytes(c.len(), test).max(NODE_TESTS * test))
}

/// Each test of `tests` alone, as clauses: the disjunction of the tests.
pub(crate) fn any_of(tests: impl IntoIterator<Item = NoteTest>) -> Vec<NoteClause> {
    tests.into_iter().map(|t| [t].into()).collect()
}

/// The declared constant `t` reads, as its [`SpecTest`], or `None` where the function declares no
/// such constant, or a sampler word whose fields leave out a bit the test reads. A texture slot's
/// bound format class (`NoteOn::Bound`) is no constant: its test reads `SpecKey::BoundTexelFormat`,
/// which the runtime takes from the bound texture.
fn declared(families: &FamilyPlans, t: NoteTest) -> Option<SpecTest> {
    let key = match t.on {
        NoteOn::Unmeasured(index) => {
            let s = families.texture.slots.get(&index)?;
            s.unmeasured.then_some(SpecKey::UnmeasuredTexels(s.slot))?
        }
        // Read from the bound texture by the runtime, never a constant: every slot has one.
        NoteOn::Bound(index) => SpecKey::BoundTexelFormat(families.texture.slots.get(&index)?.slot),
        NoteOn::Sampler(index) => {
            let r = families.sampler.slots.get(&index)?;
            let fields = r.fields();
            (fields != 0 && t.mask & !fields == 0).then_some(SpecKey::SamplerState {
                slot: r.slot,
                fields,
            })?
        }
    };
    Some(SpecTest {
        key,
        mask: t.mask,
        bits: t.bits,
        equal: t.equal,
    })
}

/// `clause` over the function's declared constants, or `None` where a test reads none.
fn declared_clause(families: &FamilyPlans, clause: &NoteClause) -> Option<SpecClause> {
    let mut all = clause
        .iter()
        .map(|&t| declared(families, t))
        .collect::<Option<Vec<_>>>()?;
    all.sort_unstable();
    all.dedup();
    Some(SpecClause { all })
}

/// The module's notes and its specialized notes: every note `precision` holds, and each of
/// `specialized` whose clauses all read constants the function declares, with those clauses
/// (sorted, each once); a specialized note `precision` holds, or one whose clause reads a constant
/// the function does not declare, is unconditional. The notes come back sorted, the specialized
/// ones among them.
fn resolve_notes(
    families: &FamilyPlans,
    mut precision: BTreeSet<PrecisionNote>,
    specialized: BTreeMap<PrecisionNote, BTreeSet<NoteClause>>,
) -> (Vec<PrecisionNote>, Vec<SpecializedNote>) {
    let mut kept = Vec::new();
    for (note, clauses) in specialized {
        if precision.contains(&note) {
            continue;
        }
        let when = clauses
            .iter()
            .map(|c| declared_clause(families, c))
            .collect::<Option<BTreeSet<_>>>();
        precision.insert(note);
        if let Some(when) = when {
            kept.push(SpecializedNote {
                note,
                when: when.into_iter().collect(),
            });
        }
    }
    (precision.into_iter().collect(), kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selector form is reached from the AIR type and the lowered representation the arith
    /// family gives it, and a representation of another width is refused rather than reinterpreted.
    #[test]
    fn selector_forms_follow_the_lowered_representation() {
        let int = |bits| DemoteTy::Scalar(ScalarTy::Int { bits });
        let lits: BTreeSet<u64> = [0, 1, 5].into_iter().collect();
        let form =
            |air: Type, lowered: &DemoteTy, int64| selector_form(&air, lowered, int64, &lits);
        assert_eq!(
            form(Type::Int(1), &DemoteTy::Scalar(ScalarTy::Bool), false),
            Ok(SelectorForm::Bool)
        );
        assert_eq!(
            form(Type::Int(8), &int(8), false),
            Ok(SelectorForm::ZeroExtend)
        );
        assert_eq!(
            form(Type::Int(16), &int(16), false),
            Ok(SelectorForm::ZeroExtend)
        );
        assert_eq!(
            form(Type::Int(32), &int(32), false),
            Ok(SelectorForm::Direct)
        );
        assert_eq!(
            form(Type::Int(64), &int(64), true),
            Ok(SelectorForm::Narrow64 { miss: 2 })
        );
        assert_eq!(
            form(Type::Int(64), &int(64), false),
            Err(Unsupported::needs(
                Feature::Int64Arithmetic,
                Capability::Feature(VkFeature::ShaderInt64),
            )
            .into())
        );
        // The held forms: narrow widths are zero-extended in their container.
        for w in [2, 8, 16, 24, 31] {
            assert_eq!(
                form(Type::Int(w), &int(32), false),
                Ok(SelectorForm::Direct),
                "i{w}"
            );
        }
        // Without `shaderInt64` the selector is two words.
        let words = DemoteTy::Vector {
            elem: ScalarTy::Int { bits: 32 },
            len: 2,
        };
        for w in [33, 40, 63, 64] {
            assert_eq!(
                form(Type::Int(w), &int(64), true),
                Ok(SelectorForm::Narrow64 { miss: 2 }),
                "i{w}"
            );
            assert_eq!(
                form(Type::Int(w), &int(64), false),
                Err(Unsupported::needs(
                    Feature::Int64Arithmetic,
                    Capability::Feature(VkFeature::ShaderInt64),
                )
                .into()),
                "i{w}"
            );
            assert_eq!(
                form(Type::Int(w), &words, false),
                Ok(SelectorForm::Words { miss: 2 }),
                "i{w}"
            );
            // Two words are the representation only without `shaderInt64`.
            assert_eq!(
                form(Type::Int(w), &words, true),
                Err(not_lowered(LowerFamily::Arith)),
                "i{w}"
            );
        }
        // A two-word representation of a narrower integer is not a selector form.
        assert_eq!(
            form(Type::Int(32), &words, false),
            Err(not_lowered(LowerFamily::Arith))
        );
        for (air, lowered) in [
            (Type::Int(16), int(8)),
            (Type::Int(64), int(32)),
            (Type::Int(40), int(32)),
            (Type::Int(24), int(64)),
            (Type::Int(32), DemoteTy::Scalar(ScalarTy::Bool)),
            (Type::Int(24), int(24)),
            (
                Type::Int(8),
                DemoteTy::Vector {
                    elem: ScalarTy::Int { bits: 8 },
                    len: 2,
                },
            ),
        ] {
            assert_eq!(
                form(air, &lowered, true),
                Err(not_lowered(LowerFamily::Arith)),
                "{lowered:?}"
            );
        }
    }

    /// Every per-stage descriptor limit is checked for its own kind, samplers stay out of
    /// `maxPerStageResources`, and color attachments count only there.
    #[test]
    fn descriptor_counts_check_every_per_stage_limit() {
        let limits = Limits {
            max_per_stage_descriptor_samplers: 2,
            max_per_stage_descriptor_uniform_buffers: 2,
            max_per_stage_descriptor_storage_buffers: 2,
            max_per_stage_descriptor_sampled_images: 2,
            max_per_stage_descriptor_storage_images: 2,
            max_per_stage_descriptor_input_attachments: 2,
            max_per_stage_resources: 4,
            max_per_stage_descriptor_acceleration_structures: 2,
            ..Limits::default()
        };
        let none = DescriptorCounts::default();
        assert_eq!(none.exceeded(&limits), None);
        let cases: [(DescriptorCounts, VkLimit); 8] = [
            (
                DescriptorCounts {
                    samplers: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorSamplers,
            ),
            (
                DescriptorCounts {
                    uniform_buffers: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorUniformBuffers,
            ),
            (
                DescriptorCounts {
                    storage_buffers: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorStorageBuffers,
            ),
            (
                DescriptorCounts {
                    sampled_images: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorSampledImages,
            ),
            (
                DescriptorCounts {
                    storage_images: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorStorageImages,
            ),
            (
                DescriptorCounts {
                    input_attachments: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorInputAttachments,
            ),
            (
                DescriptorCounts {
                    storage_buffers: 2,
                    sampled_images: 2,
                    color_attachments: 1,
                    ..none
                },
                VkLimit::MaxPerStageResources,
            ),
            (
                DescriptorCounts {
                    acceleration_structures: 3,
                    ..none
                },
                VkLimit::MaxPerStageDescriptorAccelerationStructures,
            ),
        ];
        for (c, limit) in cases {
            assert_eq!(c.exceeded(&limits).map(|e| e.0), Some(limit), "{c:?}");
        }
        let samplers_are_no_resource = DescriptorCounts {
            samplers: 2,
            storage_buffers: 2,
            sampled_images: 2,
            ..none
        };
        assert_eq!(samplers_are_no_resource.exceeded(&limits), None);
        assert_eq!(samplers_are_no_resource.descriptors(), 6);
        let sum = samplers_are_no_resource.plus(&DescriptorCounts {
            storage_buffers: 1,
            ..none
        });
        assert_eq!(sum.storage_buffers, 3);
    }

    /// The 64-bit selector's miss value is no case literal, so a selector whose high word is set
    /// matches no case.
    #[test]
    fn the_miss_literal_is_the_smallest_unused_u32() {
        let set = |v: &[u64]| v.iter().copied().collect::<BTreeSet<u64>>();
        assert_eq!(miss_literal(&set(&[])), Some(0));
        assert_eq!(miss_literal(&set(&[1, 2, 7])), Some(0));
        assert_eq!(miss_literal(&set(&[0, 1, 2, 7])), Some(3));
        assert_eq!(miss_literal(&set(&[0, 2])), Some(1));
        assert_eq!(miss_literal(&set(&[0, u64::from(u32::MAX)])), Some(1));
        for lits in [set(&[0, 1, 5]), set(&[3, 4]), set(&[0, 1, 2, 3])] {
            let m = miss_literal(&lits).expect("free");
            assert!(!lits.contains(&u64::from(m)));
        }
    }

    /// Every parameter role of a family not implemented yet is refused, in every stage, as its
    /// family's typed feature.
    #[test]
    fn object_parameter_roles_are_refused_as_their_family() {
        use crate::air::abi::{
            AbiInt, AccelerationStructureKind, ImageblockArg, ImageblockKind, ObjectArg, Slot,
        };
        let arg = ObjectArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access: None,
        };
        let roles = [
            (ParamRole::FunctionTable(arg), Feature::DynamicCall),
            (
                ParamRole::CommandBuffer(arg),
                Feature::ArgumentBufferObjects,
            ),
            (
                ParamRole::ComputePipelineState(arg),
                Feature::ArgumentBufferObjects,
            ),
            (
                ParamRole::RenderPipelineState(arg),
                Feature::ArgumentBufferObjects,
            ),
            (
                ParamRole::DepthStencilState(arg),
                Feature::IndirectRenderState,
            ),
        ];
        let imageblock = |kind, alias_implicit| {
            ParamRole::Imageblock(ImageblockArg {
                kind,
                data_size: Some(16),
                layout: None,
                master: None,
                alias_implicit,
            })
        };
        for stage in [Stage::Kernel, Stage::Vertex, Stage::Fragment] {
            for (role, feature) in &roles {
                assert_eq!(
                    param_family(stage, role),
                    Err(Unsupported::not_implemented(feature.clone()).into()),
                    "{stage:?} {role:?}"
                );
            }
            // A kernel's explicit imageblock is the access family's threadgroup staging; the
            // implicit imageblock, and an explicit one aliased onto it, are tile shading.
            let explicit = if stage == Stage::Kernel {
                Ok(LowerFamily::Access)
            } else {
                Err(Unsupported::not_implemented(Feature::Imageblocks).into())
            };
            assert_eq!(
                param_family(stage, &imageblock(ImageblockKind::Explicit, false)),
                explicit,
                "{stage:?}"
            );
            for role in [
                imageblock(ImageblockKind::Implicit, false),
                imageblock(ImageblockKind::Explicit, true),
            ] {
                assert_eq!(
                    param_family(stage, &role),
                    Err(Unsupported::not_implemented(Feature::TileShading).into()),
                    "{stage:?} {role:?}"
                );
            }
            // A fragment function's imageblock view is the fragment family's.
            let data = if stage == Stage::Fragment {
                Ok(LowerFamily::Fragment)
            } else {
                Err(Unsupported::not_implemented(Feature::Imageblocks).into())
            };
            assert_eq!(
                param_family(stage, &imageblock(ImageblockKind::Data, false)),
                data,
                "{stage:?}"
            );
            // An acceleration structure is the ray tracing family's, which decides on the
            // device, and so is an intersection function table, which intersect calls read.
            assert_eq!(
                param_family(stage, &ParamRole::IntersectionTable(arg)),
                Ok(LowerFamily::RayTracing),
                "{stage:?}"
            );
            for kind in [
                AccelerationStructureKind::Primitive,
                AccelerationStructureKind::Instance,
            ] {
                assert_eq!(
                    param_family(stage, &ParamRole::AccelerationStructure { kind, arg }),
                    Ok(LowerFamily::RayTracing),
                    "{stage:?} {kind:?}"
                );
            }
        }
    }
}
