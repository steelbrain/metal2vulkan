//! The full finalize verifier.
//!
//! [`verify`] runs on every finished [`TypedModule`] before it is serialized, in every build,
//! release included. The builder already makes these rules hold by construction; the verifier is
//! the always-on check that catches a builder, structurizer, or lowering bug before any bytes
//! leave the crate. A violation is an internal error
//! (`Error::Internal(Verify)`), never an input error.
//!
//! Rules, by family:
//!
//! - **Ownership.** Every block belongs to exactly one function and names it; every branch and
//!   merge target is a block of the same function; a function's first block is never a branch
//!   target.
//! - **SSA.** Every result is defined once, where its value definition says; parameters and
//!   `Function` variables belong to their function; every use is dominated by its definition
//!   (an `OpPhi` operand by the end of its incoming block); every `OpPhi` lists each predecessor
//!   exactly once and nothing else. Dominance comes from `graph`.
//! - **Structured control flow.** Constructs are the layout intervals `[header, merge)` the
//!   builder's preorder layout produces. They nest; every block follows its immediate dominator; a
//!   header dominates its construct, its merge, and its continue target; a construct is entered
//!   only through its header and left only to its merge or to the innermost loop's merge or
//!   continue target; the only backward edge is a loop's continue block to its header.
//!   `OpBranchConditional`/`OpSwitch` carry a selection merge, `OpLoopMerge` headers end in
//!   `OpBranch`; switch literals are distinct and fit the selector; case arms have the switch as
//!   their only predecessor (no fall-through); no block is the merge or continue target of two
//!   headers.
//! - **Canonical loops.** Every `OpLoopMerge` has exactly one
//!   [`LoopMeta`](crate::spirv::ir::LoopMeta) naming it, and every `LoopMeta` names one; the header
//!   is empty, entered from its preheader and its single continue block, which branches back to it.
//! - **Tangled instructions.** No tangled instruction sits inside a switch-at-use, directly or
//!   through a call.
//! - **Places.** Every atomic's pointer operand is a place: a variable, or the result of an access
//!   chain or a texel pointer. SPIRV-Cross writes an atomic's pointer as `&expr`, which MSL accepts
//!   only for a place, so an atomic on a converted device address is refused (the builder's
//!   `device_place` gives the address as a place).
//! - **Functions.** The static call graph has no cycle.
//! - **Stage rules.** `OpKill`/`OpTerminateInvocation` only in fragment modules, and only in the
//!   entry function when an interlock mode is declared; every builtin variable exists in the
//!   entry's execution model and direction.
//! - **Derived facts.** The recorded entry interface and requirements equal what the module
//!   contains (a transform that changes the module re-derives them).
//! - **Profile rules** ([`verify`] only): no `Workgroup` initializer unless the profile enables
//!   `shaderZeroInitializeWorkgroupMemory`, and the requirements are a subset of the profile
//!   (`Internal(RequirementsExceedProfile)`).
//!
//! A pass counts what it covered as it checks it, and that count is the module's public
//! [`ModuleStats`]: the verifier runs on every translation, so the statistics cost nothing beyond
//! it. A last pass over every instruction decides whether the module is inert
//! ([`ModuleStats::inert`], `effects`).
//!
//! Work is linear in the module plus, per branch edge, the number of constructs it leaves; it is
//! metered on the per-attempt [`Meter`] and every table is charged to its ledger.

mod cfg;
mod effects;
mod rules;

#[cfg(test)]
mod tests;

use std::fmt;

use crate::budget::Meter;
use crate::device::profile::DeviceProfile;
use crate::error::{Capability, Error, InternalError, InternalKind};
use crate::pipeline::artifact::ModuleStats;
use crate::spirv::error::{BuildError, Dangling};
use crate::spirv::ids::{BlockId, FuncId, GlobalId, Value};
use crate::spirv::ir::TypedModule;

/// A broken verifier rule, naming where.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    /// An id outside the module's arenas.
    Dangling(Dangling),
    /// Re-deriving the interface or requirements failed.
    Rederive(BuildError),
    // ---- Ownership ----
    /// A function with no blocks.
    EmptyFunction(FuncId),
    /// A block listed by no function, by two, or by a function it does not name.
    BlockOwnership(BlockId),
    /// A branch, merge, or continue target in another function.
    ForeignTarget { from: BlockId, to: BlockId },
    /// A function's first block is a branch target.
    EntryBlockTarget(BlockId),
    // ---- SSA ----
    /// A value whose recorded origin does not name the instruction, phi, parameter, or variable
    /// that defines it.
    ResultOrigin(Value),
    /// A value defined twice.
    DefinedTwice(Value),
    /// A use of an instruction or phi result that nothing defines.
    Undefined { value: Value, block: BlockId },
    /// A use of a parameter, variable, or result of another function.
    ForeignValue { value: Value, block: BlockId },
    /// A use not dominated by its definition.
    NotDominated { value: Value, block: BlockId },
    /// An `OpPhi` whose incoming blocks are not exactly the block's predecessors, each once.
    PhiIncoming { phi: Value, block: BlockId },
    // ---- Structured control flow ----
    /// A merge declaration that does not fit the terminator (`OpSelectionMerge` needs a
    /// conditional branch or switch, `OpLoopMerge` an unconditional branch).
    MergeTerminator(BlockId),
    /// A conditional branch or switch without a selection merge.
    MissingMerge(BlockId),
    /// A block that is the merge or continue target of two headers.
    SharedMerge(BlockId),
    /// A merge laid out before its header, or a continue target outside its loop.
    ConstructLayout(BlockId),
    /// Two constructs that overlap without nesting (named by the inner header).
    ConstructNesting(BlockId),
    /// A reachable block laid out before its immediate dominator.
    LayoutDominance(BlockId),
    /// A construct header that does not dominate this reachable block of its construct, its
    /// merge, or its continue target.
    HeaderDominance { header: BlockId, block: BlockId },
    /// A backward branch that is not a loop's continue block returning to its header.
    BackEdge { from: BlockId, to: BlockId },
    /// A branch into a construct other than to its header.
    ConstructEntry { from: BlockId, to: BlockId },
    /// A branch out of a construct other than to its merge or the innermost loop's merge or
    /// continue target.
    ConstructExit { from: BlockId, to: BlockId },
    /// A switch literal listed twice or outside the selector's width.
    SwitchLiterals(BlockId),
    /// A case arm reached from somewhere other than its switch (fall-through).
    CaseFallthrough { header: BlockId, case: BlockId },
    // ---- Loops ----
    /// An `OpLoopMerge` with no `LoopMeta`.
    LoopMetaMissing(BlockId),
    /// A `LoopMeta` that names no loop header, or disagrees with it (index into the loop table).
    LoopMetaMismatch(u32),
    /// A loop that is not of canonical shape (named by its header).
    LoopShape(BlockId),
    // ---- Switches at use ----
    /// A switch-at-use record that names no switch header.
    SwitchAtUseHeader(BlockId),
    /// A tangled instruction inside a switch-at-use.
    TangledInSwitchAtUse { header: BlockId, block: BlockId },
    /// A call inside a switch-at-use to a function that executes a tangled instruction.
    TangledCallInSwitchAtUse { header: BlockId, callee: FuncId },
    // ---- Functions and stages ----
    /// A function on a cycle of the static call graph.
    Recursion(FuncId),
    /// `OpKill`/`OpTerminateInvocation` in a non-fragment module.
    KillForModel(BlockId),
    /// `OpKill`/`OpTerminateInvocation` outside the entry function under an interlock mode.
    KillOutsideEntry(FuncId),
    /// A builtin variable the entry's execution model does not have in that direction, or in a
    /// form (a mesh entry's per-vertex or per-primitive array) the model does not declare it in.
    BuiltInForModel(GlobalId),
    /// A `TaskPayloadWorkgroupEXT` variable outside a task or mesh entry.
    PayloadForModel(GlobalId),
    // ---- Derived facts ----
    /// The recorded entry interface differs from the re-derived one.
    InterfaceStale,
    /// The recorded requirements differ from what the module contains plus the facts of its own
    /// subgroup requirement.
    RequirementsStale,
    // ---- Profile ----
    /// A `Workgroup` variable with an initializer on a profile without
    /// `shaderZeroInitializeWorkgroupMemory`.
    WorkgroupInitializer(GlobalId),
    /// An atomic in `block` whose pointer operand `ptr` is not a place (a variable, an access
    /// chain, or a texel pointer).
    AtomicNotPlace { ptr: Value, block: BlockId },
}

/// Why verification failed.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// A broken rule (an internal bug upstream of the verifier).
    Rule(Violation),
    /// The module needs a capability the profile lacks.
    ExceedsProfile(Capability),
    /// The meter refused (work cap, ledger, deadline, cancellation).
    Budget(Error),
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::Rule(v) => write!(f, "SPIR-V verifier rule broken: {v:?}"),
            VerifyError::ExceedsProfile(c) => {
                write!(f, "module requires {c:?}, which the profile lacks")
            }
            VerifyError::Budget(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VerifyError {}

impl From<Violation> for VerifyError {
    fn from(v: Violation) -> VerifyError {
        VerifyError::Rule(v)
    }
}

impl From<Dangling> for VerifyError {
    fn from(d: Dangling) -> VerifyError {
        VerifyError::Rule(Violation::Dangling(d))
    }
}

impl From<BuildError> for VerifyError {
    fn from(e: BuildError) -> VerifyError {
        match e {
            BuildError::Dangling(d) => VerifyError::Rule(Violation::Dangling(d)),
            other => VerifyError::Rule(Violation::Rederive(other)),
        }
    }
}

impl From<Error> for VerifyError {
    fn from(e: Error) -> VerifyError {
        VerifyError::Budget(e)
    }
}

impl From<crate::air::error::Error> for VerifyError {
    fn from(e: crate::air::error::Error) -> VerifyError {
        VerifyError::Budget(e.into())
    }
}

impl From<VerifyError> for Error {
    fn from(e: VerifyError) -> Error {
        match e {
            VerifyError::Rule(v) => {
                Error::Internal(InternalError::new(InternalKind::Verify, format!("{v:?}")))
            }
            VerifyError::ExceedsProfile(c) => Error::Internal(InternalError::new(
                InternalKind::RequirementsExceedProfile,
                format!("{c:?}"),
            )),
            VerifyError::Budget(e) => e,
        }
    }
}

/// The full verifier: every structural rule, then the profile rules. A broken rule is
/// `Internal(Verify)`; requirements beyond the profile are `Internal(RequirementsExceedProfile)`.
pub fn verify(
    m: &TypedModule,
    profile: &DeviceProfile,
    meter: &mut Meter,
) -> Result<ModuleStats, Error> {
    check(m, profile, meter).map_err(Error::from)
}

/// [`verify`] with the typed failure.
pub fn check(
    m: &TypedModule,
    profile: &DeviceProfile,
    meter: &mut Meter,
) -> Result<ModuleStats, VerifyError> {
    let report = check_structure(m, meter)?;
    rules::profile_rules(m, profile)?;
    Ok(report)
}

/// Every rule that does not depend on a device profile. Tests of builder families that no
/// committed profile enables use this; products always run [`verify`].
pub fn check_structure(m: &TypedModule, meter: &mut Meter) -> Result<ModuleStats, VerifyError> {
    let mut ledger = Ledger { bytes: 0 };
    let r = check_in(m, meter, &mut ledger);
    meter.release(ledger.bytes);
    r
}

fn check_in(
    m: &TypedModule,
    meter: &mut Meter,
    ledger: &mut Ledger,
) -> Result<ModuleStats, VerifyError> {
    let layout = cfg::Layout::build(m, meter, ledger)?;
    let defs = cfg::Defs::build(m, meter, ledger)?;
    let calls = rules::CallGraph::build(m, meter, ledger)?;
    let tangled_fn = calls.check_and_propagate(meter)?;
    let mut report = ModuleStats {
        functions: u32::try_from(m.functions.len()).unwrap_or(u32::MAX),
        blocks: u32::try_from(m.blocks.len()).unwrap_or(u32::MAX),
        loops: u32::try_from(m.loops.len()).unwrap_or(u32::MAX),
        ..ModuleStats::default()
    };
    let mut constructs = cfg::ModuleConstructs::new(m, meter, ledger)?;
    let facts = cfg::ModuleFacts {
        m,
        layout: &layout,
        defs: &defs,
        tangled_fn: &tangled_fn,
    };
    for fi in 0..m.functions.len() {
        let func = FuncId::new(u32::try_from(fi).unwrap_or(u32::MAX));
        cfg::check_function(&facts, func, &mut constructs, &mut report, meter)?;
    }
    rules::check_places(m, &defs, meter)?;
    rules::check_stage(m)?;
    rules::check_derived(m)?;
    report.inert = effects::inert(m, meter)?;
    Ok(report)
}

/// Bytes charged to the meter for the verifier's own tables, released when it returns.
pub(crate) struct Ledger {
    bytes: usize,
}

impl Ledger {
    /// An empty `Vec` with room for `n` elements, charged to `meter`.
    pub(crate) fn with_capacity<T>(
        &mut self,
        meter: &mut Meter,
        n: usize,
    ) -> Result<Vec<T>, VerifyError> {
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LimitExceeded(crate::error::Limit::LedgerBytes))?;
        meter.charge(bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        let mut v = Vec::new();
        v.try_reserve_exact(n).map_err(|e| {
            Error::Internal(InternalError::new(
                InternalKind::AllocationFailed,
                format!("reserving {bytes} ledger-admitted bytes: {e}"),
            ))
        })?;
        Ok(v)
    }

    /// A `Vec` of `n` copies of `fill`, charged to `meter`.
    pub(crate) fn filled<T: Clone>(
        &mut self,
        meter: &mut Meter,
        n: usize,
        fill: T,
    ) -> Result<Vec<T>, VerifyError> {
        let mut v = self.with_capacity(meter, n)?;
        v.resize(n, fill);
        Ok(v)
    }
}
