//! Helper invocations kept out of the branches whose code SPIRV-Cross writes as `optnone` calls,
//! on a driver whose compiler loses a simdgroup's storage writes there.
//!
//! Apple's compiler loses every texture write a simdgroup (8 by 4 pixels) makes after a
//! `[[clang::optnone]]` function is called in a branch that only its non-live threads enter, in a
//! fragment function that samples a texture through an argument buffer: the quad helpers past a
//! primitive's right and bottom edges (the probe: 1,040 to 1,063 of 2,584 texels lost a
//! run, in 98 or 99 blocks, with no discard at all), and the threads a discard demotes. With the
//! call plain, in straight-line code, with no sample, or with the resources bound directly, every
//! write landed, and color outputs landed in every form. SPIRV-Cross (MoltenVK 1.4.1's, as
//! Homebrew's) writes these SPIR-V instructions as `optnone` calls ([`optnone`]): an `OpFAdd`,
//! `OpFSub` or `OpFMul` decorated `NoContraction` (`spvFAdd`, `spvFSub`, `spvFMul`, which the
//! lowering uses wherever an operation must be rounded on its own), `OpQuantizeToF16`,
//! `OpUMulExtended` and `OpSMulExtended` (`spvMulExtended`, which `mulhi` and `madhi` take), and
//! `GLSL.std.450` `Reflect`. MoltenVK binds every resource through argument buffers, and on an
//! Apple GPU carries `DriverWorkaround::DiscardDropsSimdgroupStores`.
//!
//! There, in a fragment entry whose code samples (`OpImageSample*` or a gather) and writes storage
//! (an image write, or a store or atomic through a `StorageBuffer`, `PhysicalStorageBuffer`,
//! `Uniform` or `Image` pointer), [`guard`] sends the helper invocations of each selection that
//! has such a call in an arm, with a storage write after it, to an arm that has none: the
//! selection's condition is and'ed with "not a helper" (`OpIsHelperInvocationEXT`, which
//! SPIRV-Cross reads from the `gl_HelperInvocation` it sets at entry from
//! `simd_is_helper_thread()` and at each discard) where the false arm is the safe one, or'ed with
//! "a helper" where the true arm is, and a switch's selector is the safe case's literal for a
//! helper (measured both conditions' forms in plain Metal on an M4). Selections
//! are guarded innermost first; an arm whose calls an inner guard keeps helpers out of has none
//! left.
//!
//! A selection whose condition or selector every invocation of the draw computes alike (built only
//! from constants, specialization constants among them, and loads through push constant, uniform
//! block and resource handle pointers, by arithmetic, composites, selects and access chains) sends
//! a quad's helper invocations with its live ones, never alone: no call in its arms is where only
//! helpers may run it, so it is neither guarded nor counted as exposed.
//!
//! A guard is exact, so it is placed only where nothing a helper invocation can show changes:
//!
//! - every arm stays inside the selection and leaves it only for its merge (no return, kill,
//!   break or continue), so a helper reaches the merge on either route and no loop it is in runs
//!   longer or shorter;
//! - no arm holds a tangled instruction (implicit-LOD sampling, `OpImageQueryLod`, derivatives,
//!   subgroup operations), directly or in a function it calls, since the helpers in it change;
//! - nothing a helper computes differently on the new route reaches one: the merge's phis, and
//!   the `Function` and `Private` variables the arms store to, are followed through every
//!   instruction, phi, variable, parameter and return value of the entry's functions, and a
//!   tangled instruction reading one, or a branch or switch on one in a module with any tangled
//!   instruction, leaves the selection as it is.
//!
//! A helper invocation's own writes and outputs have no effect, so its new route changes nothing
//! else. A call can stay where only helper invocations may run it before a storage write: an arm
//! of a selection that cannot be guarded exactly and that no enclosing guard covers, or a loop,
//! whose helpers may iterate alone.
//!
//! There the call is harmless where a tangled instruction follows it: the defect loses nothing where the helpers make
//! an implicit-LOD sample, a derivative, an LOD query or a simdgroup operation after the call. In
//! plain Metal on an M4, 0 of 2,584 texels were lost with any of them after the call in its arm (a
//! sample at a coordinate the call computes or of its own, `dfdx`, `fwidth`, either LOD query,
//! `simd_sum`, `simd_broadcast`, a ballot, a ballot and shuffles, `quad_shuffle`, `quad_sum`), with
//! a sample reading the call's value after the branch's merge, and with one in a branch inside the
//! arm; 1,046 to 1,063 were lost with the same instruction before the call, with a helper test
//! after it, or with nothing after it, the written texture read-write or write-only alike. So a
//! call counts only where a path from it ends (a return, kill or termination) without passing a
//! tangled instruction that reads what the call computes, through the instructions after it.
//! Reading it is what keeps the instruction after the call in the Metal code: SPIRV-Cross moves a
//! single-use expression to its use, so a tangled instruction merely later in the module may come
//! before the call there (the probe's `hoist` forms). It is followed through `Function` and
//! `Private` variables too, but not back from a later block's phi or store. A call outside every
//! selection and loop of a callee counts at its call sites. An arm a guard sends helpers past holds
//! no tangled instruction, so the guards are as before.
//!
//! **Fences.** Each call that counts is fenced ([`Fences`]): an identity shuffle follows it, its
//! result read in every invocation from the invocation's own lane (`OpGroupNonUniformShuffle` at
//! `SubgroupLocalInvocationId`; a `MulExtended`'s structure member by member, put together again),
//! and every use of the result reads the shuffled value. Every invocation, a helper too, reads its
//! own value, so no answer changes and no lane reads another's (a guard's exactness holds); the
//! shuffle is the simdgroup operation after the call that reads what it computes, which keeps the
//! simdgroup's writes. The caller places the fences once every analysis of the module has run
//! ([`Fences::apply`]): they would read the shuffle as a move between lanes. A call stays bare where
//! the profile has no fragment shuffle (`subgroupSupportedStages` without the fragment stage, or
//! `subgroupSupportedOperations` without `SHUFFLE`) or its result is a lane no shuffle moves (a
//! 64-bit or `bool` one, or an 8- or 16-bit one without `shaderSubgroupExtendedTypes`); then
//! [`Guarded::exposed`] says so, and the caller records
//! `PrecisionNote::HelperBranchDropsSimdgroupStores`: the writes may be lost as the defect loses
//! them.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::Meter;
use crate::device::{DeviceProfile, DriverWorkaround};
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ids::{BlockId, ConstId, FuncId, GlobalId, TypeId, Value, VarId};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, Contraction, Decl, ExecModel, GlobalKind, GlobalVar, GlslOp,
    GroupInst, Inst, InstNode, IoDir, MemAccess, Merge, Scope, ScopeId, ShuffleOp, StorageClass,
    Terminator, Type, TypedModule, UnOp, ValueDef, ValueOrigin,
};

/// The census name of the selections [`guard`] guards ([`Guarded::selections`]), which the
/// runtime's pipeline creation log prints as `helper-guard <stage>`.
pub const STRATEGY: &str = "fragment.helper-guard";

/// "None" in the dense position tables.
const NONE: u32 = u32::MAX;

/// The most whole-module passes a taint analysis takes before it calls its guard inexact.
const MAX_PASSES: u32 = 64;

/// The census name of the `optnone` calls [`guard`] fences ([`Guarded::fences`]), which the
/// runtime's pipeline creation log prints as `helper-fence <stage>`.
pub const FENCE_STRATEGY: &str = "fragment.helper-fence";

/// What [`guard`] did to a module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Guarded {
    /// The selections whose helper invocations now take an arm without an `optnone` call.
    pub selections: u32,
    /// The `optnone` calls an identity shuffle of what they compute follows ([`Fences`]).
    pub fences: u32,
    /// An `optnone` call stays where only helper invocations may run it before a storage write.
    pub exposed: bool,
}

/// The `optnone` calls a guard left where only helper invocations may make them before a storage
/// write, with no tangled instruction reading what they compute after them, each to be followed
/// by an identity shuffle of its result ([`Fences::apply`]). [`guard`] decides them; the caller
/// applies them once every analysis of the module has run, since a shuffle is an identity that
/// those analyses would read as a move between lanes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fences {
    /// Each call's block and result.
    sites: Vec<(BlockId, Value)>,
}

/// The fences a profile can place: a fragment entry's `OpGroupNonUniformShuffle`
/// (`subgroupSupportedStages` with the fragment stage, `subgroupSupportedOperations` with
/// `SHUFFLE`), and of 8- and 16-bit values (`shaderSubgroupExtendedTypes`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FenceForms {
    shuffle: bool,
    extended: bool,
}

/// [`guard`] where `p`'s driver has the defect (`DriverWorkaround::DiscardDropsSimdgroupStores`);
/// nothing elsewhere. A helper query needs `shaderDemoteToHelperInvocation`: without it nothing
/// is guarded. A call that stays where only helpers may run it is fenced where the profile can
/// place the fence ([`Fences`]), else reported.
pub fn guard_for(
    m: &mut TypedModule,
    p: &DeviceProfile,
    meter: &mut Meter,
) -> Result<(Guarded, Fences), Error> {
    let fields = p.fields();
    if !fields
        .workarounds
        .contains(&DriverWorkaround::DiscardDropsSimdgroupStores)
    {
        return Ok((Guarded::default(), Fences::default()));
    }
    let fragment = &fields.subgroup.fragment;
    let forms = FenceForms {
        shuffle: fragment.supported
            && fragment
                .ops
                .contains(crate::device::subgroup::OpBits::SHUFFLE),
        extended: fields
            .enabled
            .features
            .contains(&crate::device::requirements::VkFeature::ShaderSubgroupExtendedTypes),
    };
    guard(m, fields.fragment.demote_to_helper, forms, meter)
}

/// Keep `m`'s helper invocations out of the arms that hold an `optnone` call before a storage
/// write (the module docs), where `m` is a fragment entry that samples and writes storage, and
/// fence each call that stays where only they may make it (`forms`, [`Fences`]).
/// `query`: the module may ask whether its invocation is a helper (`OpIsHelperInvocationEXT`).
fn guard(
    m: &mut TypedModule,
    query: bool,
    forms: FenceForms,
    meter: &mut Meter,
) -> Result<(Guarded, Fences), Error> {
    if m.entry.model != ExecModel::Fragment {
        return Ok((Guarded::default(), Fences::default()));
    }
    let (guards, fences, exposed) = {
        let facts = Facts::new(m, meter)?;
        if !facts.samples
            || !facts
                .writes
                .get(m.entry.func.index())
                .copied()
                .unwrap_or(false)
        {
            return Ok((Guarded::default(), Fences::default()));
        }
        let Plan {
            guards,
            bare,
            problem,
        } = facts.plan(query, meter)?;
        // Every bare call is fenced where its result can be shuffled; the note stays where one
        // cannot.
        let mut fences = Fences::default();
        let mut exposed = false;
        if problem {
            for (b, v) in bare {
                match forms.shuffle && facts.fenceable(v, forms.extended)? {
                    true => fences.sites.push((b, v)),
                    false => exposed = true,
                }
            }
        }
        (guards, fences, exposed)
    };
    let selections = u32::try_from(guards.len()).map_err(|_| invariant("guard count"))?;
    if !guards.is_empty() {
        for g in &guards {
            apply(m, g)?;
        }
        m.rederive()?;
    }
    let count = u32::try_from(fences.sites.len()).map_err(|_| invariant("fence count"))?;
    Ok((
        Guarded {
            selections,
            fences: count,
            exposed,
        },
        fences,
    ))
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Whether SPIRV-Cross writes `inst` as a call of an `[[clang::optnone]]` function.
fn optnone(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Binary {
            op: BinOp::FAdd | BinOp::FSub | BinOp::FMul,
            contract: Contraction::Forbidden,
            ..
        } | Inst::Binary {
            op: BinOp::UMulExtended | BinOp::SMulExtended,
            ..
        } | Inst::Unary {
            op: UnOp::QuantizeToF16,
            ..
        } | Inst::Glsl {
            op: GlslOp::Reflect,
            ..
        }
    )
}

/// Whether `inst` samples a texture through a sampler.
fn samples(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::ImageSample { .. } | Inst::ImageGather { .. } | Inst::ImageDrefGather { .. }
    )
}

/// Whether a terminator leaves its function or ends its invocation.
fn exits(t: &Terminator) -> bool {
    !matches!(
        t,
        Terminator::Branch(_) | Terminator::CondBranch { .. } | Terminator::Switch { .. }
    )
}

/// The branch targets of a terminator.
fn targets(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Branch(x) => vec![*x],
        Terminator::CondBranch { t, f, .. } => vec![*t, *f],
        Terminator::Switch { default, cases, .. } => std::iter::once(*default)
            .chain(cases.iter().map(|c| c.1))
            .collect(),
        _ => Vec::new(),
    }
}

/// Where a pointer points: into a `Function` variable, a `Private` variable, or anything else
/// (storage, an output, an image), whose writes a helper invocation cannot read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Root {
    Local(VarId),
    Private(GlobalId),
    Other,
}

/// How a guarded selection's header sends a helper invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// To the false target: the condition and'ed with "not a helper".
    False,
    /// To the true target: the condition or'ed with "a helper".
    True,
    /// To the target of this selector value.
    Case(u64),
}

/// One selection to guard: its header and the route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Guard {
    header: BlockId,
    route: Route,
}

/// What [`Facts::plan`] decides.
struct Plan {
    /// The selections to guard.
    guards: Vec<Guard>,
    /// Every bare `optnone` call (its block and result: one a path from which ends without
    /// passing a tangled instruction that reads what it computes) where only helper invocations
    /// may make it, or in a callee, outside its selections and loops, that its callers may call
    /// so.
    bare: Vec<(BlockId, Value)>,
    /// A bare call where only helper invocations may make it.
    problem: bool,
}

/// A structured construct of one function: the layout interval `[start, end)` from its header to
/// its merge.
#[derive(Clone, Copy, Debug)]
struct Construct {
    header: BlockId,
    merge: BlockId,
    start: u32,
    end: u32,
    selection: bool,
}

/// What [`Facts::uniform`] has decided: the instruction of each result of the entry's functions
/// (built at the first question), and each value decided.
#[derive(Default)]
struct Uniformity<'a> {
    defs: Option<BTreeMap<Value, &'a Inst>>,
    decided: BTreeMap<Value, bool>,
}

/// Whether the storage class holds what every invocation of a draw reads alike: push constants,
/// uniform blocks (which no shader writes), and resource handles.
fn uniform_class(class: StorageClass) -> bool {
    matches!(
        class,
        StorageClass::PushConstant | StorageClass::Uniform | StorageClass::UniformConstant
    )
}

/// Whether `inst` computes the same result in every invocation that gives it the same operands,
/// reading no invocation's own state: arithmetic, logic, conversions, composites, and an access
/// chain or load (of a class [`uniform_class`] admits).
fn same_from_same(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Unary { .. }
            | Inst::Binary { .. }
            | Inst::Select { .. }
            | Inst::BitFieldInsert { .. }
            | Inst::Fma { .. }
            | Inst::BitFieldExtract { .. }
            | Inst::Convert { .. }
            | Inst::CompositeConstruct { .. }
            | Inst::CompositeExtract { .. }
            | Inst::CompositeInsert { .. }
            | Inst::VectorExtractDynamic { .. }
            | Inst::VectorInsertDynamic { .. }
            | Inst::VectorShuffle { .. }
            | Inst::Glsl { .. }
            | Inst::AccessChain { .. }
            | Inst::Load { .. }
    )
}

/// What the analysis knows of the module's functions.
struct Facts<'a> {
    m: &'a TypedModule,
    /// The functions the entry reaches, callers before callees.
    order: Vec<FuncId>,
    /// By block: its position in its function's layout.
    pos: Vec<u32>,
    /// By function: it, or a function it calls, writes storage.
    writes: Vec<bool>,
    /// By function: it, or a function it calls, holds a tangled instruction.
    tangled: Vec<bool>,
    /// By function: it, or a function it calls, holds an `optnone` call.
    calls_optnone: Vec<bool>,
    /// By function: a storage write may follow one of its returns, in a caller.
    after_return: Vec<bool>,
    /// By block: a storage write may follow the block's start.
    reach_write: Vec<bool>,
    /// The entry's code samples a texture.
    samples: bool,
    /// The pointer each access chain or copy result points into.
    roots: BTreeMap<Value, Root>,
}

impl<'a> Facts<'a> {
    fn new(m: &'a TypedModule, meter: &mut Meter) -> Result<Facts<'a>, Error> {
        let nf = m.functions.len();
        let callees = |f: FuncId| -> Vec<FuncId> {
            let mut out = Vec::new();
            for b in m
                .functions
                .get(f.index())
                .map_or(&[][..], |f| &f.blocks[..])
            {
                for node in m.blocks.get(b.index()).map_or(&[][..], |b| &b.insts[..]) {
                    if let Inst::Call { func, .. } = node.inst {
                        out.push(func);
                    }
                }
            }
            out
        };
        // Postorder of the call graph from the entry (SPIR-V has no recursion).
        let mut post = Vec::new();
        let mut seen = vec![false; nf];
        let mut stack = vec![(m.entry.func, callees(m.entry.func), 0usize)];
        if let Some(s) = seen.get_mut(m.entry.func.index()) {
            *s = true;
        }
        while let Some(top) = stack.last_mut() {
            let (f, next) = (top.0, top.1.get(top.2).copied());
            top.2 += 1;
            match next {
                Some(c) => {
                    if let Some(s) = seen.get_mut(c.index()).filter(|s| !**s) {
                        *s = true;
                        stack.push((c, callees(c), 0));
                    }
                }
                None => {
                    post.push(f);
                    stack.pop();
                }
            }
        }
        let order: Vec<FuncId> = post.iter().rev().copied().collect();
        let mut pos = vec![NONE; m.blocks.len()];
        for f in &order {
            let func = m
                .functions
                .get(f.index())
                .ok_or_else(|| invariant("a called function"))?;
            for (p, b) in func.blocks.iter().enumerate() {
                if let Some(x) = pos.get_mut(b.index()) {
                    *x = u32::try_from(p).map_err(|_| invariant("block positions"))?;
                }
            }
        }
        let mut facts = Facts {
            m,
            order,
            pos,
            writes: vec![false; nf],
            tangled: vec![false; nf],
            calls_optnone: vec![false; nf],
            after_return: vec![false; nf],
            reach_write: vec![false; m.blocks.len()],
            samples: false,
            roots: BTreeMap::new(),
        };
        // Callees first: what each function or its callees hold.
        for f in &post {
            let (mut w, mut t, mut o) = (false, false, false);
            for b in facts.blocks_of(*f) {
                let blk = facts.block(b)?;
                meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
                for node in &blk.insts {
                    let inst = &node.inst;
                    w |= facts.writes_inst(inst);
                    t |= inst.is_tangled() || facts.calls(inst, &facts.tangled);
                    o |= optnone(inst) || facts.calls(inst, &facts.calls_optnone);
                    facts.samples |= samples(inst);
                    let root = match inst {
                        Inst::AccessChain { base, .. } => Some(facts.root(*base)),
                        Inst::Unary {
                            op: UnOp::CopyObject,
                            a,
                        } if facts.is_pointer(*a) => Some(facts.root(*a)),
                        _ => None,
                    };
                    if let (Some(r), Some(v)) = (root, node.result) {
                        facts.roots.insert(v, r);
                    }
                }
            }
            for (table, v) in [
                (&mut facts.writes, w),
                (&mut facts.tangled, t),
                (&mut facts.calls_optnone, o),
            ] {
                if let Some(x) = table.get_mut(f.index()) {
                    *x = v;
                }
            }
        }
        // Callers first: whether a write may follow a return, then each block's reach.
        let mut sites: Vec<Vec<(FuncId, BlockId, usize)>> = vec![Vec::new(); nf];
        for g in &facts.order {
            for b in facts.blocks_of(*g) {
                for (i, node) in facts.block(b)?.insts.iter().enumerate() {
                    if let Inst::Call { func, .. } = node.inst {
                        if let Some(s) = sites.get_mut(func.index()) {
                            s.push((*g, b, i));
                        }
                    }
                }
            }
        }
        for f in facts.order.clone() {
            let mut after = false;
            for &(g, b, i) in sites.get(f.index()).map_or(&[][..], |s| &s[..]) {
                after |= facts.write_after(g, b, i + 1)?;
            }
            if let Some(x) = facts.after_return.get_mut(f.index()) {
                *x = after;
            }
            facts.reach_writes(f, meter)?;
        }
        Ok(facts)
    }

    /// Whether every invocation of a draw takes the same arm of the selection headed by
    /// `header`: its condition or selector is [`Facts::uniform`].
    fn uniform_branch(
        &self,
        header: BlockId,
        u: &mut Uniformity<'a>,
        meter: &mut Meter,
    ) -> Result<bool, Error> {
        match self.block(header)?.term {
            Terminator::CondBranch { cond: v, .. } | Terminator::Switch { sel: v, .. } => {
                self.uniform(v, u, meter)
            }
            _ => Ok(false),
        }
    }

    /// Whether every invocation of a draw computes `v` alike: a constant (a specialization
    /// constant among them), a push constant's, uniform block's or resource's pointer, or what
    /// [`same_from_same`] computes from such values alone (a load reading one of those
    /// classes). A phi, a parameter, a `Function` variable, and anything else are not.
    fn uniform(&self, v: Value, u: &mut Uniformity<'a>, meter: &mut Meter) -> Result<bool, Error> {
        let m = self.m;
        if u.defs.is_none() {
            let mut defs = BTreeMap::new();
            for f in &self.order {
                for b in self.blocks_of(*f) {
                    let blk = self.block(b)?;
                    meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
                    for node in &blk.insts {
                        if let Some(r) = node.result {
                            defs.insert(r, &node.inst);
                        }
                    }
                }
            }
            u.defs = Some(defs);
        }
        let Uniformity { defs, decided } = u;
        let defs = defs.as_ref().ok_or_else(|| invariant("the definitions"))?;
        // Operands first: a value is decided once all of its operands are (SSA without phis has
        // no cycle).
        let mut stack = vec![(v, false)];
        while let Some((x, expanded)) = stack.pop() {
            if decided.contains_key(&x) {
                continue;
            }
            meter.tick(1)?;
            let inst = match m.values.get(x.index()).map(|d| d.origin) {
                Some(ValueOrigin::Const(_)) => {
                    decided.insert(x, true);
                    continue;
                }
                Some(ValueOrigin::Global(g)) => {
                    let class = m.globals.get(g.index()).map(|gv| gv.class);
                    decided.insert(x, class.is_some_and(uniform_class));
                    continue;
                }
                Some(ValueOrigin::Inst(_)) => defs.get(&x).copied(),
                _ => None,
            };
            let Some(inst) = inst.filter(|i| same_from_same(i)) else {
                decided.insert(x, false);
                continue;
            };
            if let Inst::Load { ptr, .. } = inst {
                if !self.class(*ptr).is_some_and(uniform_class) {
                    decided.insert(x, false);
                    continue;
                }
            }
            let mut operands = Vec::new();
            inst.for_each_value(|o| operands.push(o));
            if expanded {
                let all = operands
                    .iter()
                    .all(|o| decided.get(o).copied().unwrap_or(false));
                decided.insert(x, all);
            } else {
                stack.push((x, true));
                stack.extend(
                    operands
                        .into_iter()
                        .filter(|o| !decided.contains_key(o))
                        .map(|o| (o, false)),
                );
            }
        }
        Ok(decided.get(&v).copied().unwrap_or(false))
    }

    /// Whether the result `v` of an `optnone` call can be shuffled between lanes
    /// ([`Fences::apply`]): a scalar or vector of 32-bit integers or floats, or of 8- or 16-bit
    /// ones where `extended` (`shaderSubgroupExtendedTypes`), or a structure of such members (a
    /// `MulExtended`'s). A 64-bit or `bool` lane is not: Metal's simdgroup shuffles move no
    /// 64-bit value.
    fn fenceable(&self, v: Value, extended: bool) -> Result<bool, Error> {
        let ty = self
            .m
            .values
            .get(v.index())
            .map(|d| d.ty)
            .ok_or_else(|| invariant("an optnone call's result"))?;
        match self.m.types.get(ty)? {
            Type::Struct { members, .. } => {
                for member in members.iter() {
                    if !lane_type(self.m, *member, extended)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => lane_type(self.m, ty, extended),
        }
    }

    fn blocks_of(&self, f: FuncId) -> Vec<BlockId> {
        self.m
            .functions
            .get(f.index())
            .map_or_else(Vec::new, |f| f.blocks.clone())
    }

    fn block(&self, b: BlockId) -> Result<&'a crate::spirv::ir::Block, Error> {
        self.m
            .blocks
            .get(b.index())
            .ok_or_else(|| invariant("a block of a reached function"))
    }

    fn pos(&self, b: BlockId) -> u32 {
        self.pos.get(b.index()).copied().unwrap_or(NONE)
    }

    /// Whether `inst` calls a function `table` marks.
    fn calls(&self, inst: &Inst, table: &[bool]) -> bool {
        matches!(inst, Inst::Call { func, .. } if table.get(func.index()).copied().unwrap_or(false))
    }

    fn is_pointer(&self, v: Value) -> bool {
        self.m
            .values
            .get(v.index())
            .and_then(|d| self.m.types.get(d.ty).ok())
            .is_some_and(|t| matches!(t, Type::Pointer { .. }))
    }

    /// The storage class of pointer `v`.
    fn class(&self, v: Value) -> Option<StorageClass> {
        let d = self.m.values.get(v.index())?;
        match self.m.types.get(d.ty).ok()? {
            Type::Pointer { class, .. } => Some(*class),
            _ => None,
        }
    }

    /// Where pointer `v` points; `None` for a parameter (the caller's).
    fn root(&self, v: Value) -> Root {
        match self.m.values.get(v.index()).map(|d| d.origin) {
            Some(ValueOrigin::Var(var)) => Root::Local(var),
            Some(ValueOrigin::Global(g)) => match self.m.globals.get(g.index()) {
                Some(gv) if gv.class == StorageClass::Private => Root::Private(g),
                _ => Root::Other,
            },
            _ => self.roots.get(&v).copied().unwrap_or(Root::Other),
        }
    }

    /// Whether `inst` writes storage, itself or in a function it calls.
    fn writes_inst(&self, inst: &Inst) -> bool {
        let ptr = match inst {
            Inst::ImageWrite { .. } => return true,
            Inst::Call { .. } => return self.calls(inst, &self.writes),
            Inst::Store { ptr, .. } => *ptr,
            _ => match inst.atomic_pointer() {
                Some(p) => p,
                None => return false,
            },
        };
        matches!(
            self.class(ptr),
            Some(
                StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::Uniform
                    | StorageClass::Image
            )
        )
    }

    /// Whether a storage write may follow instruction `from` of block `b` of `f`: one later in
    /// the block, at a successor's reach, or after `f` returns.
    fn write_after(&self, f: FuncId, b: BlockId, from: usize) -> Result<bool, Error> {
        let blk = self.block(b)?;
        if blk
            .insts
            .get(from..)
            .unwrap_or(&[])
            .iter()
            .any(|n| self.writes_inst(&n.inst))
        {
            return Ok(true);
        }
        if exits(&blk.term) {
            return Ok(
                matches!(blk.term, Terminator::Return | Terminator::ReturnValue(_))
                    && self.after_return.get(f.index()).copied().unwrap_or(false),
            );
        }
        Ok(targets(&blk.term)
            .iter()
            .any(|t| self.reach_write.get(t.index()).copied().unwrap_or(false)))
    }

    /// Fill `reach_write` for `f`'s blocks, to a fixpoint over its loops.
    fn reach_writes(&mut self, f: FuncId, meter: &mut Meter) -> Result<(), Error> {
        let blocks = self.blocks_of(f);
        loop {
            let mut changed = false;
            meter.tick(u32::try_from(blocks.len()).unwrap_or(u32::MAX))?;
            for b in blocks.iter().rev() {
                if self.reach_write.get(b.index()).copied().unwrap_or(true) {
                    continue;
                }
                if self.write_after(f, *b, 0)? {
                    if let Some(x) = self.reach_write.get_mut(b.index()) {
                        *x = true;
                    }
                    changed = true;
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }

    /// The structured constructs of `f`.
    fn constructs(&self, f: FuncId) -> Result<Vec<Construct>, Error> {
        let mut out = Vec::new();
        for b in self.blocks_of(f) {
            let blk = self.block(b)?;
            let (merge, selection) = match blk.merge {
                Some(Merge::Selection { merge }) => (merge, true),
                Some(Merge::Loop { merge, .. }) => (merge, false),
                None => continue,
            };
            out.push(Construct {
                header: b,
                merge,
                start: self.pos(b),
                end: self.pos(merge),
                selection,
            });
        }
        Ok(out)
    }

    /// The blocks of `f` an arm entered at `t` runs before selection `c`'s merge, by position, or
    /// `None` where the arm leaves the selection otherwise (a return, kill, break or continue).
    fn arm(&self, f: &[BlockId], c: &Construct, t: BlockId) -> Result<Option<Vec<u32>>, Error> {
        let mut seen = BTreeSet::new();
        let mut work = vec![t];
        while let Some(b) = work.pop() {
            if b == c.merge {
                continue;
            }
            let p = self.pos(b);
            if p <= c.start || p >= c.end || f.get(p as usize) != Some(&b) {
                return Ok(None);
            }
            if !seen.insert(p) {
                continue;
            }
            let blk = self.block(b)?;
            if exits(&blk.term) {
                return Ok(None);
            }
            work.extend(targets(&blk.term));
        }
        Ok(Some(seen.into_iter().collect()))
    }

    /// Decide the guards, callees first, and whether a call stays exposed.
    fn plan(&self, query: bool, meter: &mut Meter) -> Result<Plan, Error> {
        let m = self.m;
        let nf = m.functions.len();
        // By function: an `optnone` call no guard covers, before a write.
        let mut exposed_fn = vec![false; nf];
        // By function: a call it leaves open that a path returns past with no tangled
        // instruction reading what the call computes.
        let mut bare_fn = vec![false; nf];
        let mut problem = false;
        let mut guards = Vec::new();
        let mut bare = Vec::new();
        // Every guard exact at once, where the union of all candidates' taint reaches nothing.
        let mut all_exact: Option<bool> = None;
        let mut uniformity = Uniformity::default();
        for f in self.order.iter().rev() {
            let blocks = self.blocks_of(*f);
            let n = blocks.len();
            let mut exposed = vec![false; n];
            for (p, b) in blocks.iter().enumerate() {
                let blk = self.block(*b)?;
                meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
                for (i, node) in blk.insts.iter().enumerate() {
                    // A callee's own calls are exposed where a write follows them, inside it or
                    // after it returns.
                    let hit = if optnone(&node.inst) {
                        self.write_after(*f, *b, i + 1)?
                    } else {
                        self.calls(&node.inst, &exposed_fn)
                    };
                    if hit {
                        if let Some(x) = exposed.get_mut(p) {
                            *x = true;
                        }
                        break;
                    }
                }
            }
            if !exposed.contains(&true) {
                continue;
            }
            // A selection whose condition every invocation of the draw computes alike sends a
            // quad's helpers along with its live invocations, never alone: no call in its arms is
            // where only helpers may run it, so it is no construct here.
            let mut cons = Vec::new();
            for c in self.constructs(*f)? {
                if c.selection && self.uniform_branch(c.header, &mut uniformity, meter)? {
                    continue;
                }
                cons.push(c);
            }
            cons.sort_by_key(|c| std::cmp::Reverse(c.start));
            let mut shielded = vec![false; n];
            for c in cons.iter().filter(|c| c.selection) {
                if m.switches_at_use.contains(&c.header) {
                    continue;
                }
                let live = |p: &u32| {
                    exposed.get(*p as usize).copied().unwrap_or(false)
                        && !shielded.get(*p as usize).copied().unwrap_or(false)
                };
                let inside = (c.start + 1)..c.end;
                if !inside.clone().any(|p| live(&p)) {
                    continue;
                }
                let Some(route) = self.route(&blocks, c, &exposed, &shielded)? else {
                    continue;
                };
                if !query {
                    continue;
                }
                if self.arms_tangled(&blocks, c)? {
                    continue;
                }
                if all_exact.is_none() {
                    let every = self.candidates()?;
                    all_exact = Some(self.exact(&every, meter)?);
                }
                let exact = all_exact == Some(true) || self.exact(&[(*f, *c)], meter)?;
                if !exact {
                    continue;
                }
                guards.push(Guard {
                    header: c.header,
                    route: route.0,
                });
                for p in route.1 {
                    if let Some(x) = shielded.get_mut(p as usize) {
                        *x = true;
                    }
                }
            }
            // What stays exposed: in a function, any call; where only helpers may run it, a
            // call inside a selection's arm or a loop.
            let open: Vec<usize> = (0..n)
                .filter(|p| {
                    exposed.get(*p).copied().unwrap_or(false)
                        && !shielded.get(*p).copied().unwrap_or(false)
                })
                .collect();
            if let Some(x) = exposed_fn.get_mut(f.index()) {
                *x = !open.is_empty();
            }
            // What the note counts: an open call a path from which ends (a return, kill or
            // termination) without passing a tangled instruction that reads what the call
            // computes. A call inside a selection or loop whose condition is not draw-uniform
            // records the note; one outside every such construct of a callee is its callers' to
            // count, through the call. Guards are as before: an arm they guard holds no tangled
            // instruction at all. Every function's bare calls are listed, whatever the note.
            let mut inner = Vec::new();
            let mut outer = Vec::new();
            for &p in &open {
                let at = u32::try_from(p).unwrap_or(NONE);
                let inside = cons.iter().any(|c| {
                    if c.selection {
                        c.start < at && at < c.end
                    } else {
                        c.start <= at && at < c.end
                    }
                });
                if !inside && *f == m.entry.func {
                    continue;
                }
                let Some(b) = blocks.get(p) else {
                    continue;
                };
                for (i, node) in self.block(*b)?.insts.iter().enumerate() {
                    let hit = if optnone(&node.inst) {
                        self.write_after(*f, *b, i + 1)?
                    } else {
                        self.calls(&node.inst, &bare_fn)
                    };
                    if hit && inside {
                        inner.push((p, i));
                    } else if hit {
                        outer.push((p, i));
                    }
                }
            }
            let inner = self.bare(&blocks, &inner, meter)?;
            let outer = self.bare(&blocks, &outer, meter)?;
            if !inner.is_empty() {
                problem = true;
            } else if !outer.is_empty() {
                if let Some(x) = bare_fn.get_mut(f.index()) {
                    *x = true;
                }
            }
            for (p, i) in inner.into_iter().chain(outer) {
                let Some(b) = blocks.get(p) else {
                    continue;
                };
                // A call of a function is cured inside the function.
                if let Some(node) = self.block(*b)?.insts.get(i).filter(|n| optnone(&n.inst)) {
                    let v = node
                        .result
                        .ok_or_else(|| invariant("an optnone call's result"))?;
                    bare.push((*b, v));
                }
            }
        }
        Ok(Plan {
            guards,
            bare,
            problem,
        })
    }

    /// The calls of `calls` (block position and instruction index in `blocks`, one function's
    /// layout) a path from which ends without passing a tangled instruction that reads what the
    /// call computes. The calls are taken 64 at a time: one pass over the function's layout from
    /// the first of them marks, by block, the calls each tangled instruction there reads (a bit a
    /// call, through every operand, phi, and `Function` or `Private` variable stored to, from the
    /// call on, in layout order, where a block's dominators come first; what a block reaches only
    /// through a later block's phi or store is not followed), then each call's paths are walked to
    /// their end or to such a block.
    fn bare(
        &self,
        blocks: &[BlockId],
        calls: &[(usize, usize)],
        meter: &mut Meter,
    ) -> Result<Vec<(usize, usize)>, Error> {
        let mut out = Vec::new();
        for batch in calls.chunks(64) {
            let bit: BTreeMap<(usize, usize), u64> = batch
                .iter()
                .zip(0..64)
                .map(|(c, k)| (*c, 1u64 << k))
                .collect();
            let first = batch.iter().map(|c| c.0).min().unwrap_or(blocks.len());
            let mut reads: BTreeMap<Value, u64> = BTreeMap::new();
            let mut held: BTreeMap<Root, u64> = BTreeMap::new();
            let mut meets = vec![0u64; blocks.len()];
            for (q, b) in blocks.iter().enumerate().skip(first) {
                let blk = self.block(*b)?;
                meter.tick(u32::try_from(blk.insts.len() + blk.phis.len()).unwrap_or(u32::MAX))?;
                for phi in &blk.phis {
                    let mask = phi
                        .incoming
                        .iter()
                        .fold(0, |a, (v, _)| a | reads.get(v).copied().unwrap_or(0));
                    if mask != 0 {
                        reads.insert(phi.result, mask);
                    }
                }
                for (i, node) in blk.insts.iter().enumerate() {
                    let mut mask = bit.get(&(q, i)).copied().unwrap_or(0);
                    let mut read = 0;
                    node.inst
                        .for_each_value(|v| read |= reads.get(&v).copied().unwrap_or(0));
                    match node.inst {
                        Inst::Load { ptr, .. } => {
                            read |= held.get(&self.root(ptr)).copied().unwrap_or(0);
                        }
                        Inst::Store { ptr, value, .. } => {
                            let root = self.root(ptr);
                            let stored = reads.get(&value).copied().unwrap_or(0);
                            if root != Root::Other && stored != 0 {
                                *held.entry(root).or_insert(0) |= stored;
                            }
                        }
                        _ => {}
                    }
                    if node.inst.is_tangled() {
                        if let Some(x) = meets.get_mut(q) {
                            *x |= read;
                        }
                    }
                    mask |= read;
                    if let (Some(r), true) = (node.result, mask != 0) {
                        reads.insert(r, mask);
                    }
                }
            }
            for (k, &(p, i)) in batch.iter().enumerate() {
                if self.bare_path(blocks, &meets, p, 1u64 << k)? {
                    out.push((p, i));
                }
            }
        }
        Ok(out)
    }

    /// Whether a path from the block at position `p` of `blocks` ends (a return, kill or
    /// termination) without entering a block whose `meets` holds `bit`, or leaves the layout.
    fn bare_path(
        &self,
        blocks: &[BlockId],
        meets: &[u64],
        p: usize,
        bit: u64,
    ) -> Result<bool, Error> {
        let met = |q: usize| meets.get(q).is_some_and(|x| x & bit != 0);
        if met(p) {
            return Ok(false);
        }
        let Some(&b) = blocks.get(p) else {
            return Err(invariant("a block of the function"));
        };
        let mut seen = BTreeSet::from([p]);
        let mut work = vec![b];
        while let Some(x) = work.pop() {
            let blk = self.block(x)?;
            match blk.term {
                Terminator::Unreachable => continue,
                ref t if exits(t) => return Ok(true),
                _ => {}
            }
            for t in targets(&blk.term) {
                let q = self.pos(t) as usize;
                if blocks.get(q) != Some(&t) {
                    return Ok(true);
                }
                if !met(q) && seen.insert(q) {
                    work.push(t);
                }
            }
        }
        Ok(false)
    }

    /// Where selection `c` can send its helper invocations: its route, with the positions of the
    /// arms they no longer enter. `None` where an arm leaves the selection otherwise than at its
    /// merge, or every arm runs an exposed call.
    fn route(
        &self,
        blocks: &[BlockId],
        c: &Construct,
        exposed: &[bool],
        shielded: &[bool],
    ) -> Result<Option<(Route, Vec<u32>)>, Error> {
        let blk = self.block(c.header)?;
        let tgs = targets(&blk.term);
        let mut arms: Vec<(BlockId, Vec<u32>, bool)> = Vec::new();
        for t in tgs.iter().copied().collect::<BTreeSet<_>>() {
            let Some(region) = self.arm(blocks, c, t)? else {
                return Ok(None);
            };
            let hot = region.iter().any(|p| {
                exposed.get(*p as usize).copied().unwrap_or(false)
                    && !shielded.get(*p as usize).copied().unwrap_or(false)
            });
            arms.push((t, region, hot));
        }
        // The merge itself when an arm is empty, else the first arm without a call.
        let safe = if tgs.contains(&c.merge) {
            c.merge
        } else {
            match arms.iter().find(|a| !a.2) {
                Some(a) => a.0,
                None => return Ok(None),
            }
        };
        let route = match &blk.term {
            Terminator::CondBranch { t, .. } if *t == safe => Route::True,
            Terminator::CondBranch { f, .. } if *f == safe => Route::False,
            Terminator::Switch { default, cases, .. } => match cases.iter().find(|x| x.1 == safe) {
                Some(x) => Route::Case(x.0),
                None if *default == safe => {
                    let used: BTreeSet<u64> = cases.iter().map(|x| x.0).collect();
                    let free = (0u64..).find(|v| !used.contains(v)).unwrap_or(0);
                    Route::Case(free)
                }
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let away = arms
            .into_iter()
            .filter(|a| a.0 != safe)
            .flat_map(|a| a.1)
            .collect();
        Ok(Some((route, away)))
    }

    /// Whether a block inside selection `c` holds a tangled instruction or calls a function that
    /// does.
    fn arms_tangled(&self, blocks: &[BlockId], c: &Construct) -> Result<bool, Error> {
        for p in (c.start + 1)..c.end {
            let Some(b) = blocks.get(p as usize) else {
                continue;
            };
            if self
                .block(*b)?
                .insts
                .iter()
                .any(|n| n.inst.is_tangled() || self.calls(&n.inst, &self.tangled))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Every selection with an `optnone` call (or a call of a function holding one) in its arms.
    fn candidates(&self) -> Result<Vec<(FuncId, Construct)>, Error> {
        let mut out = Vec::new();
        for f in &self.order {
            let blocks = self.blocks_of(*f);
            for c in self.constructs(*f)?.into_iter().filter(|c| c.selection) {
                let mut hot = false;
                for p in (c.start + 1)..c.end {
                    if let Some(b) = blocks.get(p as usize) {
                        hot |=
                            self.block(*b)?.insts.iter().any(|n| {
                                optnone(&n.inst) || self.calls(&n.inst, &self.calls_optnone)
                            });
                    }
                }
                if hot {
                    out.push((*f, c));
                }
            }
        }
        Ok(out)
    }

    /// Whether rerouting the helper invocations of `selections` changes nothing a tangled
    /// instruction reads (the module docs): what a helper computes differently from there (each
    /// merge's phis, and the variables the arms store to) is followed to a fixpoint through the
    /// entry's functions.
    fn exact(&self, selections: &[(FuncId, Construct)], meter: &mut Meter) -> Result<bool, Error> {
        let m = self.m;
        let tangled_anywhere = self
            .tangled
            .get(m.entry.func.index())
            .copied()
            .unwrap_or(false);
        let mut taint = vec![false; m.values.len()];
        meter.charge(taint.len())?;
        let r = self.taint(selections, &mut taint, tangled_anywhere, meter);
        meter.release(taint.len());
        r
    }

    fn taint(
        &self,
        selections: &[(FuncId, Construct)],
        taint: &mut [bool],
        tangled_anywhere: bool,
        meter: &mut Meter,
    ) -> Result<bool, Error> {
        let m = self.m;
        let mut vars: BTreeSet<Root> = BTreeSet::new();
        let mut returns = vec![false; m.functions.len()];
        let set = |taint: &mut [bool], v: Value| -> bool {
            match taint.get_mut(v.index()) {
                Some(x) if !*x => {
                    *x = true;
                    true
                }
                _ => false,
            }
        };
        let is = |taint: &[bool], v: Value| taint.get(v.index()).copied().unwrap_or(false);
        for (f, c) in selections {
            let blocks = self.blocks_of(*f);
            for phi in &self.block(c.merge)?.phis {
                set(taint, phi.result);
            }
            for p in (c.start + 1)..c.end {
                let Some(b) = blocks.get(p as usize) else {
                    continue;
                };
                for node in &self.block(*b)?.insts {
                    let ptrs: Vec<Value> = match &node.inst {
                        Inst::Store { ptr, .. } => vec![*ptr],
                        Inst::Call { args, .. } => args
                            .iter()
                            .copied()
                            .filter(|a| self.is_pointer(*a))
                            .collect(),
                        other => other.atomic_pointer().into_iter().collect(),
                    };
                    for ptr in ptrs {
                        match self.root(ptr) {
                            Root::Other if self.class(ptr) == Some(StorageClass::Function) => {
                                // A parameter's pointer: the caller's variable.
                                return Ok(false);
                            }
                            Root::Other => {}
                            r => {
                                vars.insert(r);
                            }
                        }
                    }
                }
            }
        }
        let mut passes = 0;
        loop {
            passes += 1;
            if passes > MAX_PASSES {
                return Ok(false);
            }
            let mut changed = false;
            for f in &self.order {
                let func = m
                    .functions
                    .get(f.index())
                    .ok_or_else(|| invariant("a reached function"))?;
                for b in &func.blocks {
                    let blk = self.block(*b)?;
                    meter.tick(
                        u32::try_from(blk.insts.len() + blk.phis.len()).unwrap_or(u32::MAX),
                    )?;
                    for phi in &blk.phis {
                        if phi.incoming.iter().any(|(v, _)| is(taint, *v)) {
                            changed |= set(taint, phi.result);
                        }
                    }
                    for node in &blk.insts {
                        let inst = &node.inst;
                        let mut hit = false;
                        inst.for_each_value(|v| hit |= is(taint, v));
                        if let Inst::Load { ptr, .. } = inst {
                            hit |= vars.contains(&self.root(*ptr));
                        }
                        if let Inst::Call { func: callee, args } = inst {
                            // A tainted argument taints its parameter, for every call (the
                            // callee's own instructions are followed in the same passes), and a
                            // tainted return value every call's result.
                            let callee_fn = m
                                .functions
                                .get(callee.index())
                                .ok_or_else(|| invariant("a called function"))?;
                            for (param, arg) in callee_fn.params.iter().zip(args.iter()) {
                                if is(taint, *arg) {
                                    changed |= set(taint, *param);
                                }
                            }
                            hit |= returns.get(callee.index()).copied().unwrap_or(false);
                        }
                        if !hit {
                            continue;
                        }
                        if inst.is_tangled() {
                            return Ok(false);
                        }
                        let ptr = match inst {
                            Inst::Store { ptr, .. } => Some(*ptr),
                            other => other.atomic_pointer(),
                        };
                        if let Some(ptr) = ptr {
                            match self.root(ptr) {
                                Root::Other if self.class(ptr) == Some(StorageClass::Function) => {
                                    return Ok(false);
                                }
                                Root::Other => {}
                                r => changed |= vars.insert(r),
                            }
                        }
                        if let Some(r) = node.result {
                            changed |= set(taint, r);
                        }
                    }
                    match blk.term {
                        Terminator::CondBranch { cond: v, .. }
                        | Terminator::Switch { sel: v, .. }
                            if tangled_anywhere && is(taint, v) =>
                        {
                            return Ok(false);
                        }
                        Terminator::ReturnValue(v) if is(taint, v) => {
                            if let Some(x) = returns.get_mut(f.index()) {
                                if !*x {
                                    *x = true;
                                    changed = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !changed {
                return Ok(true);
            }
        }
    }
}

/// Whether a value of type `ty` is a lane a simdgroup shuffle moves ([`Facts::fenceable`]).
fn lane_type(m: &TypedModule, ty: TypeId, extended: bool) -> Result<bool, Error> {
    let scalar = match m.types.get(ty)? {
        Type::Vector { elem, .. } => *elem,
        _ => ty,
    };
    Ok(match m.types.get(scalar)? {
        Type::Int { width, .. } => *width == 32 || (extended && matches!(*width, 8 | 16)),
        Type::Float { width } => *width == 32 || (extended && *width == 16),
        _ => false,
    })
}

impl Fences {
    /// Follow each fenced call with an identity shuffle of what it computes: its result read in
    /// every invocation from the invocation's own lane (`OpGroupNonUniformShuffle` at
    /// `SubgroupLocalInvocationId`, a structure's members each), which every later use reads in
    /// its place. Every invocation, a helper too, reads its own value, so no answer changes; the
    /// shuffle is the tangled instruction after the call that keeps the simdgroup's writes on
    /// Apple's compiler (the module docs). The entry reads the lane through the module's
    /// `SubgroupLocalInvocationId` input, declared where the module has none.
    pub fn apply(&self, m: &mut TypedModule) -> Result<(), Error> {
        if self.sites.is_empty() {
            return Ok(());
        }
        let u32_ty = ty(
            m,
            Type::Int {
                width: 32,
                signed: false,
            },
        )?;
        let lane = lane_input(m, u32_ty)?;
        let scope = ScopeId {
            scope: Scope::Subgroup,
            id: constant_id(
                m,
                Constant::Scalar {
                    ty: u32_ty,
                    bits: ConstBits(Scope::Subgroup.to_spirv() as u64),
                },
            )?,
        };
        for &(b, v) in &self.sites {
            fence(m, (b, v), lane, u32_ty, scope)?;
        }
        m.rederive()?;
        Ok(())
    }
}

/// The pointer of the module's `SubgroupLocalInvocationId` input, declared where it has none.
fn lane_input(m: &mut TypedModule, u32_ty: TypeId) -> Result<Value, Error> {
    let builtin = crate::spv::BuiltIn::SubgroupLocalInvocationId;
    let found = m.globals.iter().find(|g| {
        g.kind
            == GlobalKind::BuiltIn {
                builtin,
                dir: IoDir::Input,
            }
            && g.pointee == u32_ty
    });
    if let Some(g) = found {
        return Ok(g.value);
    }
    let class = StorageClass::Input;
    let ptr_ty = ty(
        m,
        Type::Pointer {
            class,
            pointee: u32_ty,
        },
    )?;
    let id = GlobalId::new(u32::try_from(m.globals.len()).map_err(|_| invariant("global ids"))?);
    let value = new_value(m, ptr_ty, ValueOrigin::Global(id))?;
    m.globals.push(GlobalVar {
        class,
        pointee: u32_ty,
        ptr_ty,
        value,
        init: None,
        kind: GlobalKind::BuiltIn {
            builtin,
            dir: IoDir::Input,
        },
    });
    m.decls.push(Decl::Global(id));
    Ok(value)
}

/// Follow the call of block `b` whose result is `v` with an identity shuffle of `v` (a
/// structure's members each, put together again), renaming every use of `v` in its function to
/// the shuffled value.
fn fence(
    m: &mut TypedModule,
    (b, v): (BlockId, Value),
    lane: Value,
    u32_ty: TypeId,
    scope: ScopeId,
) -> Result<(), Error> {
    let blk = m
        .blocks
        .get(b.index())
        .ok_or_else(|| invariant("a fenced call's block"))?;
    let func = blk.func;
    let at = blk
        .insts
        .iter()
        .position(|n| n.result == Some(v))
        .ok_or_else(|| invariant("a fenced call"))?;
    let ty_of = |m: &TypedModule, x: Value| {
        m.values
            .get(x.index())
            .map(|d| d.ty)
            .ok_or_else(|| invariant("a fenced value"))
    };
    let result_ty = ty_of(m, v)?;
    let fenced = new_value(m, result_ty, ValueOrigin::Inst(b))?;
    // Every use reads the shuffled value; the shuffle, placed after this, reads the call's.
    let rename = |x: &mut Value| {
        if *x == v {
            *x = fenced;
        }
    };
    for blk in m.blocks.iter_mut().filter(|blk| blk.func == func) {
        for phi in &mut blk.phis {
            for (x, _) in &mut phi.incoming {
                rename(x);
            }
        }
        for node in &mut blk.insts {
            node.inst.for_each_value_mut(rename);
        }
        blk.term.for_each_value_mut(rename);
    }
    if m.stored_texels.contains(&v) {
        m.stored_texels.insert(fenced);
    }
    let mut nodes = Vec::new();
    let id = new_value(m, u32_ty, ValueOrigin::Inst(b))?;
    nodes.push(InstNode {
        result: Some(id),
        inst: Inst::Load {
            ptr: lane,
            access: MemAccess::default(),
        },
    });
    let shuffle = |value| Inst::Group {
        scope,
        op: GroupInst::Shuffle {
            op: ShuffleOp::Shuffle,
            value,
            x: id,
        },
    };
    let members = match m.types.get(result_ty)? {
        Type::Struct { members, .. } => Some(members.to_vec()),
        _ => None,
    };
    match members {
        Some(members) => {
            let mut parts = Vec::with_capacity(members.len());
            for (k, member) in members.into_iter().enumerate() {
                let index = u32::try_from(k).map_err(|_| invariant("a member index"))?;
                let part = new_value(m, member, ValueOrigin::Inst(b))?;
                nodes.push(InstNode {
                    result: Some(part),
                    inst: Inst::CompositeExtract {
                        composite: v,
                        indices: Box::new([index]),
                    },
                });
                let moved = new_value(m, member, ValueOrigin::Inst(b))?;
                nodes.push(InstNode {
                    result: Some(moved),
                    inst: shuffle(part),
                });
                parts.push(moved);
            }
            nodes.push(InstNode {
                result: Some(fenced),
                inst: Inst::CompositeConstruct {
                    parts: parts.into_boxed_slice(),
                },
            });
        }
        None => nodes.push(InstNode {
            result: Some(fenced),
            inst: shuffle(v),
        }),
    }
    let blk = m
        .blocks
        .get_mut(b.index())
        .ok_or_else(|| invariant("a fenced call's block"))?;
    let tail = blk.insts.split_off(at + 1);
    blk.insts.extend(nodes);
    blk.insts.extend(tail);
    Ok(())
}

/// Make the header of `g` send its helper invocations by its route.
fn apply(m: &mut TypedModule, g: &Guard) -> Result<(), Error> {
    let bool_ty = ty(m, Type::Bool)?;
    let h = value(m, g.header, bool_ty, Inst::IsHelperInvocation)?;
    let blk = m
        .blocks
        .get(g.header.index())
        .ok_or_else(|| invariant("a guarded header"))?;
    let new = match (&blk.term, g.route) {
        (Terminator::CondBranch { cond, .. }, Route::False) => {
            let cond = *cond;
            let not = value(
                m,
                g.header,
                bool_ty,
                Inst::Unary {
                    op: UnOp::LogicalNot,
                    a: h,
                },
            )?;
            value(
                m,
                g.header,
                bool_ty,
                Inst::Binary {
                    op: BinOp::LogicalAnd,
                    a: cond,
                    b: not,
                    contract: Contraction::Allowed,
                },
            )?
        }
        (Terminator::CondBranch { cond, .. }, Route::True) => {
            let cond = *cond;
            value(
                m,
                g.header,
                bool_ty,
                Inst::Binary {
                    op: BinOp::LogicalOr,
                    a: cond,
                    b: h,
                    contract: Contraction::Allowed,
                },
            )?
        }
        (Terminator::Switch { sel, .. }, Route::Case(lit)) => {
            let sel = *sel;
            let sel_ty = m
                .values
                .get(sel.index())
                .map(|d| d.ty)
                .ok_or_else(|| invariant("a switch selector"))?;
            let lit = constant(
                m,
                Constant::Scalar {
                    ty: sel_ty,
                    bits: ConstBits(lit),
                },
            )?;
            value(
                m,
                g.header,
                sel_ty,
                Inst::Select {
                    cond: h,
                    t: lit,
                    f: sel,
                },
            )?
        }
        _ => return Err(invariant("a guard's route does not fit its header")),
    };
    let blk = m
        .blocks
        .get_mut(g.header.index())
        .ok_or_else(|| invariant("a guarded header"))?;
    match &mut blk.term {
        Terminator::CondBranch { cond, .. } => *cond = new,
        Terminator::Switch { sel, .. } => *sel = new,
        _ => return Err(invariant("a guarded header without a branch")),
    }
    Ok(())
}

/// Intern type `t`, declaring it when new.
fn ty(m: &mut TypedModule, t: Type) -> Result<crate::spirv::ids::TypeId, Error> {
    let (id, fresh) = m.types.intern(t, &m.consts)?;
    if fresh {
        m.decls.push(Decl::Type(id));
    }
    Ok(id)
}

fn new_value(
    m: &mut TypedModule,
    ty: crate::spirv::ids::TypeId,
    origin: ValueOrigin,
) -> Result<Value, Error> {
    let v = Value::new(u32::try_from(m.values.len()).map_err(|_| invariant("value ids"))?);
    m.values.push(ValueDef { ty, origin });
    Ok(v)
}

/// Append `inst` of result type `ty` to block `b`'s instructions; its result.
fn value(
    m: &mut TypedModule,
    b: BlockId,
    ty: crate::spirv::ids::TypeId,
    inst: Inst,
) -> Result<Value, Error> {
    let v = new_value(m, ty, ValueOrigin::Inst(b))?;
    m.blocks
        .get_mut(b.index())
        .ok_or_else(|| invariant("a guarded header"))?
        .insts
        .push(InstNode {
            result: Some(v),
            inst,
        });
    Ok(v)
}

/// The value of plain constant `c`, declaring it when new.
fn constant(m: &mut TypedModule, c: Constant) -> Result<Value, Error> {
    let id = constant_id(m, c)?;
    Ok(m.const_value(id)?)
}

/// Plain constant `c`, declared when new.
fn constant_id(m: &mut TypedModule, c: Constant) -> Result<ConstId, Error> {
    let ty = c.ty();
    let (id, fresh) = m.consts.intern(c, &m.types)?;
    if fresh {
        if id.index() != m.const_values.len() {
            return Err(invariant("constant ids"));
        }
        let v = new_value(m, ty, ValueOrigin::Const(id))?;
        m.const_values.push(v);
        m.decls.push(Decl::Const(id));
    }
    Ok(id)
}
