//! `lower::simd`: simdgroup and quad operations, the AGX lane intrinsics ([`agx`]), and the 8x8
//! simdgroup matrices ([`matrix`]).
//!
//! **Simdgroup matrices** (`air.simdgroup_matrix_8x8_*`) are per-lane fragments of Apple's
//! distributed 8x8 layout, multiplied with subgroup shuffles ([`matrix`]); matrices outside
//! kernels are refused as `Unsupported{SimdgroupMatrix, NotImplemented}`, and the 16x16x16
//! products, which Metal makes only on an Apple10 GPU and later, as
//! `Unsupported{SimdgroupMatrix16x16x16, NotImplemented}`.
//!
//! **Width.** A Metal simdgroup is 32 lanes on every profile: `get_simdgroup_size` is the constant
//! 32, and every operation below acts on the 32-lane simdgroup of the calling lane.
//!
//! **Tiers.** The function's simd class (`SelectCx::simd`) is raised to `CrossLane(ops)` with the
//! operation classes the emitted SPIR-V needs, and `device::subgroup::tier` maps it to the tier
//! emission follows (`Plan::simd`):
//!
//! - `Direct` (a 32-lane subgroup; for a quad-only kernel, the subgroup at the size the device
//!   gives, below) and `PartialFragment` (a narrower fragment subgroup whose upper Metal lanes are
//!   permanently inactive): the simdgroup is the subgroup, so reductions, scans, votes,
//!   `is_first`, and `broadcast_first` are the whole-subgroup `OpGroupNonUniform*` instructions,
//!   and the Metal lane is `SubgroupLocalInvocationId`.
//! - `Cluster32` (a 64- or 128-lane subgroup): the simdgroup is the aligned 32-lane cluster of
//!   the calling lane, `base = L & ~31`, Metal lane `L & 31`. Reductions are `ClusteredReduce`
//!   with cluster size 32, votes and `is_first`/`broadcast_first` read the cluster's ballot word
//!   (`L >> 5`), absolute shuffles are `base | (idx & 31)`, and prefix sums are an exact
//!   in-order sum over the active lanes of the cluster.
//! - `Emulated` (a kernel no subgroup tier runs, [`workgroup`], or any kernel compiled under
//!   `LoweringInputs::emulated_simdgroups`, the variant a dispatch with a partial simdgroup runs
//!   on): simdgroup `k` is the linear threads `[32k, 32k + 32)` of `LocalInvocationIndex`, and
//!   every operation that reads other lanes exchanges them through a workgroup-memory scratch
//!   between workgroup barriers. An operation outside workgroup-uniform control flow ([`uniform`])
//!   runs predicated ([`divergent`]). A multiply-add's cell, shuffle, broadcast, or fill from a
//!   lane past the thread count, or not live in a divergent block, reads the calling lane's value
//!   instead: a thread reads only from an active thread (MSL 4.1 section 6.10.2), so the value is
//!   not promised.
//!
//! **The widest mode.** Where the device's compiler chooses a kernel's SIMD width itself
//! (`subgroupSizeControl` with `minSubgroupSize` below 32 and `maxSubgroupSize` 32, as ANV's
//! 8 to 32), the `Direct` tier's pin to 32 lanes with full subgroups compiles the kernel in the
//! widest mode, where each lane's value takes four times the registers of the narrowest one: a
//! kernel whose values the compiler would hold in registers at a narrower width spills there, and
//! its compile time grows to seconds and past them. A kernel the `Direct` tier would run on such a
//! device takes the `Emulated` tier instead when its entry, inlined, holds more than
//! [`WIDEST_PIN_FLOOR`] instructions and its values are estimated to hold more than
//! [`WIDEST_PIN_WORDS`] 32-bit words at once ([`pressure`], [`widest_pin`]): both tiers are exact
//! (the `Emulated` tier keeps every simdgroup operation and lockstep barrier, through workgroup
//! memory and barriers), and the emulated kernel takes no required size, so the driver chooses
//! its width. The census counts such a kernel once as [`WIDEST_PIN`]. A kernel the `Emulated`
//! tier refuses (a 64-bit exchange, a scratch beside its arena past the device's shared memory)
//! keeps the `Direct` tier.
//!
//! The bounds come from an authored kernel measured on an Iris Xe (RPL-P, ANV): `S` steps, each
//! a branch on one of `K` live `float` accumulators whose arm stores a device word and loads the
//! next (a lockstep pair in divergent flow: a subgroup barrier there), with `P` the arm's accesses
//! go through a pointer it loads from memory (a resident-set search a step), then every
//! accumulator updated. Pressure, not size, separates the cases: the driver's compile
//! (`vkCreateComputePipelines`, ms) and the GPU time of a dispatch of 12,288 threads (ms), pinned
//! (`Direct`) and emulated, one run each (compile times varied up to twice between runs):
//!
//! | `P K S`    | instructions | live words | compile, `Direct` | `Emulated` | GPU, `Direct` | `Emulated` |
//! |------------|-------------:|-----------:|------------------:|-----------:|--------------:|-----------:|
//! | `0 4 32`   |          818 |         12 |                20 |         35 |        0.0099 |     0.0184 |
//! | `0 16 32`  |        1,994 |         24 |                63 |         73 |        0.029  |     0.028  |
//! | `0 16 128` |        7,850 |         24 |               589 |        559 |        0.129  |     0.159  |
//! | `1 4 32`   |          874 |         34 |               783 |        766 |        0.010  |     0.018  |
//! | `0 32 64`  |        7,050 |         40 |             6,652 |        282 |        1.061  |     0.107  |
//! | `1 16 2`   |          166 |         46 |                51 |         31 |        0.0076 |     0.0050 |
//! | `1 16 4`   |          290 |         46 |               117 |         61 |        0.0154 |     0.0057 |
//! | `1 16 16`  |        1,042 |         46 |               971 |        300 |        0.070  |     0.015  |
//! | `1 16 64`  |        4,066 |         46 |            11,987 |      2,585 |        0.319  |     0.074  |
//! | `1 16 512` |       32,290 |         46 |   past 8 GiB, end |      3,292 |             - |     1.709  |
//! | `0 48 8`   |        1,362 |         56 |               458 |         38 |        0.168  |     0.021  |
//! | `0 48 16`  |        2,618 |         56 |             1,332 |         85 |        0.400  |     0.037  |
//! | `1 32 32`  |        3,618 |         62 |            18,428 |        996 |        1.221  |     0.051  |
//! | `1 48 32`  |        5,186 |         78 |            25,799 |      1,277 |        1.985  |     0.081  |
//!
//! At 34 estimated words or fewer the two compile alike (within 1.8 times either way) and the
//! pinned kernel runs as fast or faster; from 40 the pinned compile takes 1.6 to 24 times the
//! emulated one and its GPU time 1.5 to 25 times, so [`WIDEST_PIN_WORDS`] is 36. At or below
//! [`WIDEST_PIN_FLOOR`] instructions every pinned compile measured took at most 102 ms. Answers
//! were the same bits on both tiers in every run. The estimate's own cost is linear in the body:
//! 5 ms of a 3.4 s release translation of a 162,914-instruction kernel.
//!
//! **Quad-only kernels.** The 32 lanes are needed only by operations that read outside a quad; a
//! kernel with none takes its quads from the subgroup at the size the device gives; the pin to 32
//! stays where 32 is the device's narrowest size. A kernel whose every cross-lane operation is a
//! quad operation ([`quad_only`]: no simdgroup operation, multiply-add, or simdgroup barrier or
//! fence) takes the `Direct` tier with full subgroups and no required size where its kernel stage
//! runs full subgroups and its narrowest subgroup is below 32 lanes and at least 4
//! (`device::subgroup::quad_free`, as ANV's 8 to 32 and lavapipe's 8): the size varies where the
//! stage can pin one (`ALLOW_VARYING_SUBGROUP_SIZE`), and the local size is a multiple of the
//! stage's largest size (`device::subgroup::quad_requirement`). A quad is four subgroup lanes
//! from a multiple of 4, all in one subgroup (Vulkan 9.27.9), and a dispatch runs one size
//! (Vulkan 9.5.2), so at every size of 4 lanes or more each quad is four lanes of one subgroup;
//! the kernel's simdgroup builtins derive from the linear index (`lower::dispatch`), whatever the
//! size. Such a kernel pins nothing, so the widest mode is not weighed for it. A threadgroup that
//! is not a multiple of the largest size runs the emulated variant, as a pinned kernel's does. A
//! kernel with compute derivatives keeps the tier its class takes: their quads need its
//! two-dimensional local size, which the flattened tiers do not keep.
//!
//! **Partly live groups on subgroups**. On `Direct`, `PartialFragment` and `Cluster32`, an
//! operation that reads another lane's value by its index (a matrix or AGX multiply-add's cells; a
//! simdgroup or quad shuffle, shuffle-xor, relative shuffle, rotation, broadcast by lane, or fill,
//! [`group::reads_lanes`]) where some lane of its group may not take part ([`partly_live`]) takes
//! the ballot of the lanes executing it and reads a lane whose bit is clear through the calling
//! lane, so no shuffle names an invocation outside the shuffle's tangle: SPIR-V makes such a
//! shuffle's result poison, and a branch or an index that consumes poison is undefined behavior.
//! What a lane reads in place of one that takes no part is unspecified: a thread reads only from
//! an active thread (MSL 4.1 section 6.10.2), so the value is not promised. A kernel's simdgroups
//! are whole there (`subgroup::requirement`), so in a uniform block every lane takes part and the
//! plain shuffles stand. Outside kernels every such operation takes that form but a fragment's
//! read within its quad (**Helpers**, below), and on
//! `PartialFragment` a source at or above the subgroup's size, a permanently inactive Metal lane,
//! is read through the calling lane without the ballot's bit extract or a shuffle naming it.
//!
//! **Helpers**. A fragment function whose group operations read, combine, or count other lanes
//! declares `RequireFullQuadsKHR`, so each quad starts with its four invocations, as Metal runs the
//! helpers of a quad the primitive or the target cuts, and they take part in its group operations
//! (a fill reads a helper's value, a quad sum folds it in). Vulkan runs its quad instructions over
//! a quad's helpers in every entry, and its other group operations over them only where the entry
//! declares `MaximallyReconvergesKHR` (section 9.31, "Helper Invocations"), as every such entry
//! does (**Reconvergence**, below): a profile with `shaderQuadControl` has maximal reconvergence,
//! which every device with the one supports (`Device::from_queries` refuses a description listing
//! the one without the other). Every quad operation there but a fill by a constant modulo whose
//! segment is wider than the quad (`group::constant_fill_width`) reads its quad through quad
//! instructions alone: AMD's open Linux driver leaves a partly covered quad's helper out of a
//! ballot and a shuffle even under `MaximallyReconvergesKHR`. A profile without
//! `shaderQuadControl` refuses such a function.
//!
//! In compute the flattened tiers derive every Metal thread builtin from the subgroup partition
//! (`lower::dispatch`), so simdgroup `k` is exactly the linear threads `[32k, 32k + 32)` and a
//! quad is four consecutive linear threads whatever order the driver gives subgroups. Relative
//! shuffles are absolute shuffles whose source is the calling lane when the Metal source is outside
//! the simdgroup or quad (Metal keeps the lane's own value there), so no lowering reads outside the
//! calling lane's own simdgroup.
//!
//! **Quads** are the aligned groups of four subgroup lanes in every stage (Vulkan's quads, which
//! are Metal's 2×2 pixel quads in fragment and four consecutive linear threads in the flattened
//! compute tiers): shuffles with `(L & ~3) | (idx & 3)`, `OpGroupNonUniformQuadSwap` and
//! `OpGroupNonUniformQuadBroadcast` for constant masks and lanes where the stage takes Vulkan's
//! quad instructions (else the shuffles from the same lanes), reductions, prefix sums and products
//! over the quad's four lanes in Metal's order, each lane read by a shuffle under its bit of the
//! ballot of the active lanes, and the ballot word for votes, `is_first`, the first active lane's
//! broadcast, the active mask, and `quad_ballot` (the quad's four bits, quad lane `j` at bit `j`).
//! A fragment's (**Helpers** above) read lane `j` by `OpGroupNonUniformQuadBroadcast` under
//! `OpGroupNonUniformQuadAnyKHR` of "the quad lane is `j`", the shuffles, broadcasts and fills too
//! (the calling lane's own value where the lane they name takes no part), take the active lanes'
//! bits from those votes (of the predicate too, for `quad_ballot`), and vote by
//! `OpGroupNonUniformQuadAllKHR` and `…AnyKHR`; the operations of one block, which run on the
//! same active lanes, share the four votes and the lanes read of each value. In a block every
//! invocation executes together, every lane of the quad takes part ([`whole_quads`]), so there it
//! takes no vote: the active mask is the constant `0xF`, `is_first` is quad lane 0, the first
//! lane's broadcast is `OpGroupNonUniformQuadBroadcast` of lane 0, `quad_ballot` gathers each
//! lane's bit by a horizontal and a vertical `OpGroupNonUniformQuadSwap`, and a broadcast or
//! shuffle-xor by a constant lane or mask is its quad instruction. A simdgroup's
//! `simd_active_threads_mask` is the ballot of `true`, held as the 64-bit ballot is. A fragment's
//! `[[thread_index_in_quadgroup]]` is the subgroup lane's low two bits
//! ([`select_param`]), or, on a profile without the fragment quad tier, the pixel's place in its
//! aligned 2x2 block, read from `FragCoord`.
//!
//! **Values** are held as `lower::arith` holds them: narrow integers zero-extended in a `u32`,
//! `half` and `bfloat` widened in an `f32`. Moving lanes is exact on the container. A signed
//! `min`/`max` of a narrow integer sign-extends first; every integer result is masked back to its
//! width (a product's low bits are its operands' whatever their sign), and a narrow float sum or
//! product is rounded to its format. Float sums and products record
//! `PrecisionNote::ReductionOrder`. A `double` needs `shaderSubgroupExtendedTypes`. Metal has no
//! simdgroup or quad function that moves or combines 64-bit integers, so a call of one is
//! `InvalidInput` on every profile; a 64-bit mask is not such a value (the ballot's is two words
//! without `shaderInt64`).
//!
//! **Not subgroup operations.** `air.all`/`air.any` of a `bool` vector are `OpAll`/`OpAny`, and
//! `air.is_uniform` (Metal's `uniform<T>` assertion, whose result the frontend feeds to
//! `llvm.assume`) is the asserted `true`; neither raises the simd class. The asserted `true` is
//! sound only while nothing observes the result (`SelectCx::is_observed`: it fed only the
//! assumptions decode drops, directly or through phis); any other use is refused as
//! `Lowering(Simd)`.
//!
//! **Reconvergence**. An entry with a tangled instruction declares `MaximallyReconvergesKHR` where
//! the profile has `shaderMaximalReconvergence`; otherwise it declares
//! `SubgroupUniformControlFlowKHR` where available, and the artifact records
//! `Reconvergence::ImplementationDefined` for every profile without maximal reconvergence. A
//! threadgroup barrier the lockstep pass inserted stands where the whole threadgroup runs together
//! and depends on no reconvergence (`sync::barrier::BarrierPlan::uniform_placed`): an entry whose
//! only tangled instructions are such barriers declares no mode and records `Guaranteed`, as does
//! an emulated kernel whose others are emulated sites (`workgroup::check`).

pub mod agx;
pub mod divergent;
mod group;
pub mod lockstep;
pub mod matrix;
pub mod pressure;
#[cfg(test)]
mod tests;
pub mod uniform;
pub mod workgroup;

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::air::error::IntrinsicError;
use crate::air::intrinsics::{AirIntrinsic, Elem, Group, GroupOp, GroupScope, LlvmIntrinsic, Sign};
use crate::air::ir::{Call, Callee, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::device::requirements::VkFeature;
use crate::device::subgroup::{self, OpBits, SimdClass, Tier};
use crate::error::{
    Capability, Error, Feature, InputError, InputErrorKind, Location, LowerFamily, Unsupported,
};
use crate::lower::arith;
use crate::lower::arith::legal::{form_on, wide_form_on, Form, Held, LoweredConst};
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::{FamilyStrategy, Strategy};
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{ExecMode, GroupArith, IoDir, Type};

pub(crate) use group::QuadSeen;
pub use group::{cluster_ops, direct_ops};

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Simd)).into()
}

/// One simdgroup or quad operation as selected: the AIR operation, the form of its value (for a
/// vote or `is_first` the `bool` result, for a ballot or the active mask the integer result), the
/// reduction it computes, and its constant lane or mask operand when it has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupPlan {
    pub scope: GroupScope,
    pub op: GroupOp,
    pub form: Form,
    /// The combining operation of a reduction or prefix sum.
    pub arith: Option<GroupArith>,
    /// The constant lane (quad broadcast), mask (quad shuffle-xor) or modulo (shuffle-and-fill)
    /// operand.
    pub konst: Option<u32>,
    /// The function is a kernel whose stage takes Vulkan's quad instructions
    /// ([`SubgroupModel::quad_instructions`](crate::device::subgroup::SubgroupModel::quad_instructions)),
    /// which a quad broadcast or shuffle-xor by a constant lane or mask then emits outside its
    /// partly live blocks; elsewhere it is a shuffle from the lane it reads, through the live lanes
    /// outside kernels, where a quad may lack a lane.
    pub quad_instructions: bool,
    /// The function is a fragment function, whose quads hold helpers ([`full_quads`]): a quad
    /// operation that reads only its quad reads it through quad instructions, which Vulkan runs
    /// over the quad's helpers whatever the entry's reconvergence (section 9.31, "Helper
    /// Invocations").
    pub helpers: bool,
}

/// The simd strategies.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimdStrategy {
    /// `get_simdgroup_size`: the constant 32 in the result's form.
    Width(Form),
    /// `air.all`/`air.any` of a `bool` vector: `OpAll`/`OpAny`.
    VectorVote { all: bool },
    /// `air.is_uniform` whose result nothing observes: the asserted `true`.
    Assumed,
    /// A simdgroup or quad operation, lowered by the function's tier.
    Group(GroupPlan),
    /// An AGX lane intrinsic.
    Agx(agx::AgxStrategy),
    /// An 8x8 simdgroup-matrix operation.
    Matrix(matrix::MatrixStrategy),
}

impl FamilyStrategy for SimdStrategy {
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str> {
        match self {
            SimdStrategy::Width(_) => "simd.width".into(),
            SimdStrategy::VectorVote { .. } => "simd.vector_vote".into(),
            SimdStrategy::Assumed => "simd.assumed_uniform".into(),
            SimdStrategy::Group(g) => group::name(g).into(),
            SimdStrategy::Agx(a) => a.name(&plans.access),
            SimdStrategy::Matrix(m) => m.name(&plans.access),
        }
    }

    fn words(&self) -> u32 {
        match self {
            SimdStrategy::Width(_) | SimdStrategy::Assumed => 12,
            SimdStrategy::VectorVote { .. } => 8,
            SimdStrategy::Group(g) => group::words(g),
            SimdStrategy::Agx(a) => a.words(),
            SimdStrategy::Matrix(m) => m.words(),
        }
    }
}

/// How the entry states its reconvergence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReconvergenceMode {
    /// No tangled instruction, or a profile with neither mode.
    #[default]
    None,
    /// `MaximallyReconvergesKHR`.
    Maximal,
    /// `SubgroupUniformControlFlowKHR`.
    UniformControlFlow,
}

/// The simd family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SimdPlan {
    /// The operation classes the family's operations need in the `Direct` and
    /// `PartialFragment` tiers.
    pub direct_ops: OpBits,
    /// The operation classes they need in the `Cluster32` tier.
    pub cluster_ops: OpBits,
    /// Some operation of the family is a simdgroup or quad operation (it reads the subgroup
    /// lane).
    pub groups: bool,
    /// The entry's reconvergence execution mode.
    pub mode: ReconvergenceMode,
    /// A tangled instruction runs without maximal reconvergence on this profile.
    pub implementation_defined: bool,
    /// In the `Emulated` tier, the words each lane publishes to the scratch in the widest
    /// exchange of the function; 0 when no operation exchanges values.
    pub exchange_words: u32,
    /// In the `Emulated` tier under the identity domains, the tier's exchange and vote barriers
    /// run under a workgroup-uniform branch on `1 < N`, the threadgroup's threads: a threadgroup
    /// of one thread exchanges with itself alone, in program order, and executes none of them
    /// (`workgroup::Declared::several`).
    pub guarded_barriers: bool,
    /// The predicated walk of a kernel with a synchronization outside workgroup-uniform control
    /// flow: an emulated site or threadgroup barrier in the `Emulated` tier, a workgroup barrier in
    /// every other tier.
    pub predication: Option<divergent::Predication>,
    /// The kernel took the `Emulated` tier in place of the `Direct` tier's pin to the device's
    /// widest mode ([`widest_pin`]), which the census counts as [`WIDEST_PIN`].
    pub widest_pin: bool,
    /// The operations that read another lane where some lane of the group may not take part:
    /// the matrix and AGX multiply-adds and the group operations that read a lane's value by its
    /// index ([`group::reads_lanes`]), in a kernel's blocks `uniform::Uniformity` cannot prove
    /// workgroup uniform and in every block of another stage. On a subgroup tier each reads through
    /// the ballot of the lanes executing it, so a lane that does not take part is read through the
    /// calling lane and no shuffle names it; [`partly_live`].
    pub partly_live: BTreeSet<ValueId>,
    /// A fragment's quad operations whose quads are whole ([`whole_quads`]): each reads its quad's
    /// four lanes with no vote of which take part.
    pub whole_quads: BTreeSet<ValueId>,
    /// Some operation reads, combines, or counts other lanes of its simdgroup or quad (every
    /// group operation, an AGX multiply-add): in a fragment function its answer counts the
    /// helpers Metal runs in a partly covered quad ([`full_quads`]).
    pub observes_lanes: bool,
    /// The fragment entry declares `RequireFullQuadsKHR`, so every quad starts with its four
    /// invocations and the helpers take part in the group operations, as Metal's do.
    pub full_quads: bool,
    /// A fragment's `[[thread_index_in_quadgroup]]` on a profile without fragment subgroups that
    /// hold whole quads: the stage interface's `FragCoord` object the lane is derived from
    /// ([`select_param`]). `None` where the lane is the subgroup lane's low two bits.
    pub quad_lane_coord: Option<usize>,
    /// A fragment's `[[thread_index_in_simdgroup]]` is the lane of the device's grouping
    /// ([`select_param`]): the function records `PrecisionNote::FragmentSimdgroupGrouping`, and
    /// [`finish`] refuses it unless the stage's tier gives 32-lane simdgroups, the ones its
    /// operations and `[[threads_per_simdgroup]]` name: a 32-lane subgroup (`Tier::Direct`, the
    /// subgroup lane) or a 32-lane cluster of a 64- or 128-lane one (`Tier::Cluster32`, the
    /// subgroup lane's low five bits).
    pub device_lane: bool,
    /// Some operation reads or combines other lanes of the simdgroup (a simdgroup-scope group
    /// operation, an AGX multiply-add), or the function reads the lane itself, so its answer
    /// depends on which invocations share the simdgroup ([`finish`]). A quad operation does not set
    /// it.
    pub simdgroup_members: bool,
    /// A kernel whose every cross-lane operation stays inside its quad ([`quad_only`]) on a
    /// device whose narrowest kernel subgroup is below 32 lanes (`subgroup::quad_free`): its
    /// `Direct` tier pins no size and takes its quads from the subgroup at the size the device
    /// gives (`subgroup::quad_requirement`).
    pub quad_free: bool,
}

/// Whether every cross-lane operation of a function stays inside its quad: some group operation,
/// each of quad scope, and no simdgroup-scope operation, multiply-add, barrier or fence. Such a
/// function needs no 32-lane group: its quads are four aligned subgroup lanes at any subgroup
/// size that is a multiple of 4 (Vulkan 9.27.9), and its simdgroup builtins derive from the
/// linear index (`lower::dispatch`), whatever the subgroup size.
pub fn quad_only(strategy: &[Option<Strategy>]) -> bool {
    use crate::lower::sync::barrier::BarrierStrategy;
    use crate::spirv::ir::Scope;
    let mut quad = false;
    for s in strategy.iter().flatten() {
        match s {
            Strategy::Simd(SimdStrategy::Group(g)) => {
                if g.scope != GroupScope::Quad {
                    return false;
                }
                quad = true;
            }
            Strategy::Simd(SimdStrategy::Agx(_) | SimdStrategy::Matrix(_)) => return false,
            Strategy::Barrier(BarrierStrategy::Barrier(b)) if b.scope == Scope::Subgroup => {
                return false
            }
            Strategy::Barrier(BarrierStrategy::Fence(f)) if f.scope == Scope::Subgroup => {
                return false
            }
            _ => {}
        }
    }
    quad
}

/// Whether [`widest_pin`] weighs a kernel for the `Emulated` tier, or the kernel keeps the
/// `Direct` tier's pin to the device's widest mode whatever its pressure: a kernel whose emulated
/// module passed the compiled stage bound is planned again keeping it (`pipeline::compile`'s stage
/// bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidestPin {
    /// [`widest_pin`] weighs the kernel.
    Weighed,
    /// The kernel keeps the `Direct` tier's pin.
    Kept,
}

/// The census name of a kernel that took the `Emulated` tier in place of the `Direct` tier's pin
/// to the device's widest mode ([`widest_pin`]): one use, no words.
pub const WIDEST_PIN: &str = "simd.emulated.widest-pin";

/// The instructions of a kernel's inlined entry at or below which [`widest_pin`] keeps the
/// `Direct` tier whatever its pressure (the module docs, "The widest mode").
pub const WIDEST_PIN_FLOOR: usize = 256;

/// The estimated peak of live 32-bit words ([`pressure::peak_words`]) above which [`widest_pin`]
/// takes the `Emulated` tier in place of the `Direct` tier's pin to the widest mode (the module
/// docs, "The widest mode").
pub const WIDEST_PIN_WORDS: u64 = 36;

/// Whether the kernel of `cx`, whose class the `Direct` tier runs on its device, takes the
/// `Emulated` tier instead (the module docs, "The widest mode"): the device's kernel stage pins a
/// size between a `minSubgroupSize` below 32 and a `maxSubgroupSize` of 32, and the entry holds
/// more than [`WIDEST_PIN_FLOOR`] instructions whose values are estimated to hold more than
/// [`WIDEST_PIN_WORDS`] 32-bit words at once. Read from the device's facts and the entry's
/// structure alone; the estimate runs only where the device's facts and the size call for it.
pub fn widest_pin(cx: &SelectCx<'_>, m: &mut Meter) -> Result<bool, Error> {
    let model = &cx.profile.fields().subgroup;
    let st = &model.kernel;
    let widest = cx.stage == Stage::Kernel
        && st.pinnable
        && st.sizes.min < group::SIMD_WIDTH
        && st.sizes.max == group::SIMD_WIDTH
        && cx.body.insts.len() > WIDEST_PIN_FLOOR
        && subgroup::tier(&cx.simd, Stage::Kernel, model) == Ok(Tier::Direct);
    if !widest {
        return Ok(false);
    }
    Ok(pressure::peak_words(cx.entry, cx.body, m)? > WIDEST_PIN_WORDS)
}

/// The artifact's reconvergence: `ImplementationDefined` when the entry holds a tangled instruction
/// and the profile has no maximal reconvergence.
pub fn reconvergence(plan: &Plan) -> Reconvergence {
    if plan.families.simd.implementation_defined {
        Reconvergence::ImplementationDefined
    } else {
        Reconvergence::Guaranteed
    }
}

/// `c` raised by operation classes `ops`: a cross-lane class with them added. The matrix class
/// already implies cross-lane use.
pub fn raise(c: SimdClass, ops: OpBits) -> SimdClass {
    if ops == OpBits::NONE {
        return c;
    }
    match c {
        SimdClass::None | SimdClass::LaneArithmetic => SimdClass::CrossLane(ops),
        SimdClass::Quad => SimdClass::CrossLane(ops.union(OpBits::QUAD)),
        SimdClass::CrossLane(o) => SimdClass::CrossLane(o.union(ops)),
        other => other,
    }
}

/// Complete the simd family's plan: raise the function's simd class to the operation classes of
/// the tier the class selects, and choose the entry's reconvergence mode.
///
/// The classes differ per tier (a `Direct` vote is `OpGroupNonUniformAll`, a `Cluster32` one a
/// ballot), so the class is first raised with the `Direct` classes; when that selects
/// `Cluster32`, or is refused where the `Cluster32` classes would select `Cluster32`, the class
/// is raised with the `Cluster32` classes instead. `tier` chooses between `Direct` and
/// `Cluster32` by subgroup size alone, so the final class selects the tier whose classes it
/// holds, or the refusal `select` then reports.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    // A fragment function's simdgroups are the pixels the device's rasterizer packs into a
    // subgroup, so an operation over the simdgroup's members combines a simdgroup Metal allows, not
    // the reference's. So does a quad operation beside a helper query in a function that discards,
    // whose quads Apple's GPU may split ([`discarding_quads_observed`]).
    if cx.stage == Stage::Fragment
        && (cx.families.simd.simdgroup_members || discarding_quads_observed(cx))
    {
        cx.precision
            .insert(PrecisionNote::FragmentSimdgroupGrouping);
    }
    // A vertex function's simdgroups and quads are the vertices the device packs into a subgroup,
    // so every operation over other lanes combines a group Metal allows, not the reference's.
    if cx.stage == Stage::Vertex && cx.families.simd.observes_lanes {
        cx.precision.insert(PrecisionNote::VertexSimdgroupGrouping);
    }
    let plan = &cx.families.simd;
    if plan.groups {
        let model = &cx.profile.fields().subgroup;
        let base = cx.simd;
        let direct = raise(base, plan.direct_ops);
        let cluster = raise(base, plan.cluster_ops);
        // The emulated tier is the kernels' fallback: a `Cluster32` tier the cluster classes
        // reach is preferred to it, except in a kernel compiled for emulated simdgroups or one
        // whose lockstep order needs them, whose tier is `Emulated` whatever its class.
        let emulated = (cx.inputs.emulated_simdgroups || cx.entry.lockstep.emulated)
            && cx.stage == Stage::Kernel;
        cx.simd = if emulated {
            direct
        } else {
            match subgroup::tier(&direct, cx.stage, model) {
                Ok(Tier::Cluster32) => cluster,
                Ok(Tier::Emulated) | Err(_)
                    if subgroup::tier(&cluster, cx.stage, model) == Ok(Tier::Cluster32) =>
                {
                    cluster
                }
                _ => direct,
            }
        };
    }
    // The device's lane is the lane of the simdgroups the function's operations run over, and
    // `[[threads_per_simdgroup]]` names 32 of them: a 32-lane subgroup (`Direct`), or a 32-lane
    // cluster of a 64- or 128-lane subgroup (`Cluster32`), whose lane is the subgroup lane's low
    // five bits, as its operations' lanes are. On a subgroup narrower than 32 (`PartialFragment`)
    // the lane is refused, needing a 32-lane fragment stage; a refused tier is `select`'s
    // refusal.
    if cx.families.simd.device_lane {
        let model = &cx.profile.fields().subgroup;
        if subgroup::tier(&cx.simd, cx.stage, model) == Ok(Tier::PartialFragment) {
            return Err(Unsupported::needs_any_of(
                Feature::SimdWidth,
                vec![subgroup::width_needs(cx.stage, model.stage(cx.stage))],
            )
            .into());
        }
    }
    let barriers = &cx.families.barrier;
    if cx
        .tangled
        .iter()
        .any(|&v| crate::lower::sync::barrier::reconverges(barriers, v))
    {
        let model = &cx.profile.fields().subgroup;
        let (mode, defined) = if model.maximal_reconvergence {
            (ReconvergenceMode::Maximal, false)
        } else if model.uniform_control_flow {
            (ReconvergenceMode::UniformControlFlow, true)
        } else {
            (ReconvergenceMode::None, true)
        };
        cx.families.simd.mode = mode;
        cx.families.simd.implementation_defined = defined;
    }
    Ok(())
}

/// Complete a fragment function's plan once its tier is chosen: where its group operations read,
/// combine, or count other lanes ([`SimdPlan::observes_lanes`]), the entry runs full quads
/// ([`SimdPlan::full_quads`]), and a profile without `shaderQuadControl` refuses it. Metal runs the
/// helpers of a quad the primitive or the target cuts, and they take part: a fill or shuffle reads
/// a helper's value, a reduction folds it in (measured on the M3 Max and the M4). Vulkan launches
/// them only under `RequireFullQuadsKHR`; without it a helper may be absent, and no other form
/// gives its value. Its quad instructions run over the helpers it launches, and its other group
/// operations where the entry maximally reconverges (the module's **Helpers**). After the tier,
/// so a stage without the classes is refused as before.
pub fn full_quads(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment || !cx.families.simd.observes_lanes {
        return Ok(());
    }
    if cx.profile.fields().subgroup.quad_control {
        cx.families.simd.full_quads = true;
        return Ok(());
    }
    Err(Unsupported::needs(
        Feature::SubgroupOperations,
        Capability::Feature(VkFeature::ShaderQuadControl),
    )
    .into())
}

/// Whether a fragment function that discards asks whether its thread is a helper and runs a group
/// operation over other lanes ([`SimdPlan::observes_lanes`]), so an operation may read which of
/// its quad's lanes are helpers. Apple's GPU shades some quads of a function that discards as
/// several partial quads from the function's start, each run holding the quad's other covered
/// pixels as helpers: on the M3 Max (macOS 26.7 and 27.0.1 alike, and MoltenVK there), under the
/// full-screen triangle with every third pixel discarded, a live pixel's `quad_ballot` of
/// `quad_is_helper_thread` names covered pixels of its quad as helpers in 3 of the 16 quads of an
/// 8x8 target and 2 to 4 of the 12 of a 7x5 one, asked before or after the discard, at MSL 2.2 and
/// 4.0, where the function without the discard runs every quad whole (the public
/// `helper-threads-*` cases). A Vulkan device shades a primitive's covered pixels of a quad as one
/// quad, and nothing in Vulkan states or pins which quads Apple's rasterizer splits. A helper
/// computes what the covered pixel would, so only a helper query tells the runs apart.
fn discarding_quads_observed(cx: &SelectCx<'_>) -> bool {
    !cx.families.fragment.discards.is_empty()
        && cx.families.fragment.helper_queries
        && cx.families.simd.observes_lanes
}

/// The simd family's module-scope objects: the lane builtin a simdgroup or quad operation reads
/// (`SubgroupLocalInvocationId`, or `LocalInvocationIndex` in the `Emulated` tier), the
/// `SubgroupSize` builtin a divergent multiply-add reads on the `PartialFragment` tier, the
/// emulated tier's thread count and scratch, and outside that tier the workgroup word a predicated
/// kernel's loops vote in.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    pub lane: Option<GlobalId>,
    pub size: Option<GlobalId>,
    pub emu: Option<workgroup::Declared>,
    pub vote: Option<GlobalId>,
}

/// Declare the lane builtin the family's operations read (and in the `Emulated` tier the thread
/// count and the scratch; elsewhere a predicated kernel's vote word) and add the entry's
/// reconvergence mode to `modes`.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let simd = &plan.families.simd;
    match simd.mode {
        ReconvergenceMode::None => {}
        ReconvergenceMode::Maximal => modes.push(ExecMode::MaximallyReconverges),
        ReconvergenceMode::UniformControlFlow => modes.push(ExecMode::SubgroupUniformControlFlow),
    }
    if simd.full_quads {
        modes.push(ExecMode::RequireFullQuads);
    }
    // A predicated kernel's loop votes use the emulated tier's scratch even where no operation
    // reads another lane (a kernel whose only simdgroup operation is a barrier).
    if !simd.groups && simd.predication.is_none() {
        return Ok(Declared::default());
    }
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    if plan.simd.1 == Tier::Emulated {
        let lane = mb.builtin(
            crate::spv::BuiltIn::LocalInvocationIndex,
            IoDir::Input,
            u32_,
        )?;
        return Ok(Declared {
            lane: Some(lane),
            size: None,
            emu: Some(workgroup::declare(
                mb,
                plan.stage,
                simd.exchange_words,
                simd.guarded_barriers,
            )?),
            vote: None,
        });
    }
    // Outside the emulated tier a predicated kernel votes in a word of its own, and only a
    // simdgroup or quad operation reads the lane.
    let vote = match &simd.predication {
        Some(p) if p.vote_word => Some(workgroup::declare_vote(mb)?),
        _ => None,
    };
    if !simd.groups {
        return Ok(Declared {
            vote,
            ..Declared::default()
        });
    }
    let lane = mb.builtin(
        crate::spv::BuiltIn::SubgroupLocalInvocationId,
        IoDir::Input,
        u32_,
    )?;
    let size = if plan.simd.1 == Tier::PartialFragment && !simd.partly_live.is_empty() {
        Some(mb.builtin(crate::spv::BuiltIn::SubgroupSize, IoDir::Input, u32_)?)
    } else {
        None
    };
    Ok(Declared {
        lane: Some(lane),
        size,
        emu: None,
        vote,
    })
}

/// Whether strategy `s` is a matrix or AGX multiply-add (`matrix::tile_mad`).
pub fn is_multiply_add(s: &Strategy) -> bool {
    matches!(
        s,
        Strategy::Simd(
            SimdStrategy::Matrix(matrix::MatrixStrategy::Mac(_))
                | SimdStrategy::Agx(agx::AgxStrategy::MatMad(_))
        )
    )
}

/// Whether strategy `s` reads another lane's value through the ballot of the lanes executing it
/// where some lane may not take part ([`partly_live`]): a matrix or AGX multiply-add, or a group
/// operation that reads a lane by its index.
fn reads_live_lanes(s: &Strategy) -> bool {
    match s {
        Strategy::Simd(SimdStrategy::Group(g)) => group::reads_lanes(g),
        _ => is_multiply_add(s),
    }
}

/// Record the operations of the body that read another lane where some lane of the group may
/// not take part (`SimdPlan::partly_live`), before the tier is chosen: on a subgroup tier such an
/// operation ballots its live lanes and reads a lane that does not take part through the calling
/// lane, so no shuffle names an inactive invocation (SPIR-V makes that shuffle's result poison);
/// the value read in its place is not promised (MSL 4.1 section 6.10.2). The family's operation
/// classes gain `Ballot` (`OpGroupNonUniformBallot`, `OpGroupNonUniformBallotBitExtract`), and
/// `Shuffle` when a group operation is among them (a quad instruction's lane is then read by a
/// shuffle).
///
/// A kernel's subgroup is every lane of its simdgroup on the `Direct` and `Cluster32` tiers,
/// which require full subgroups and a local size that is a multiple of the subgroup size
/// (`subgroup::requirement`; a dispatch with a partial simdgroup runs the emulated variant), so in
/// a workgroup-uniform block every lane takes part and no lane is past the thread count: such an
/// operation keeps its plain shuffles. Uniformity is conservative (a block every lane of a
/// simdgroup executes, under a branch on the simdgroup index, is divergent to it); the ballot then
/// has every bit, and the select keeps every source. Another stage's groups are not required whole
/// (a vertex simdgroup past the draw's vertex count, a fragment's uncovered pixels), and neither
/// is a kernel's partial threadgroup's last simdgroup (`LoweringInputs::partial_threadgroups`),
/// so every such operation there ballots its live lanes, and its uniformity is not computed.
///
/// Returns a kernel's analysis when the body has such an operation, for the emulated tier's own
/// use (`workgroup::check`), so it is computed once; the caller releases its ledger bytes.
pub fn partly_live(
    cx: &mut SelectCx<'_>,
    strategy: &[Option<Strategy>],
    m: &mut Meter,
) -> Result<Option<uniform::Uniformity>, Error> {
    if !strategy.iter().flatten().any(reads_live_lanes) {
        return Ok(None);
    }
    // A kernel's partial threadgroup leaves its lanes past the grid out of every block, as
    // another stage's groups are not whole.
    let u = if cx.stage == Stage::Kernel && !cx.inputs.partial_threadgroups {
        Some(divergent::workgroup_uniformity(cx, m)?)
    } else {
        None
    };
    let mut reads = BTreeSet::new();
    let mut groups = false;
    for (b, block) in cx.body.blocks.iter() {
        if u.as_ref().is_some_and(|u| u.is_uniform(b)) {
            continue;
        }
        for id in block.inst_ids() {
            let Some(v) = cx.values.inst(id) else {
                continue;
            };
            let Some(s) = strategy.get(v.index()).and_then(Option::as_ref) else {
                continue;
            };
            if reads_live_lanes(s) {
                reads.insert(v);
                groups |= matches!(s, Strategy::Simd(SimdStrategy::Group(_)));
            }
        }
    }
    let simd = &mut cx.families.simd;
    if !reads.is_empty() {
        let ops = if groups {
            OpBits::BALLOT.union(OpBits::SHUFFLE)
        } else {
            OpBits::BALLOT
        };
        simd.direct_ops = simd.direct_ops.union(ops);
        simd.cluster_ops = simd.cluster_ops.union(ops);
    }
    simd.partly_live = reads;
    Ok(u)
}

/// Record a fragment's quad operations whose quads are whole (`SimdPlan::whole_quads`): those
/// that read only their quad (`group::fragment_quad`) in a block `uniform::Uniformity` proves
/// every invocation executes together, every stage input divergent. Each quad's four invocations
/// then run the block: the entry declares full quads ([`full_quads`]), a quad's helpers run it as
/// its covered invocations do, and a discarded invocation runs on as a helper or a predicated
/// lane (`fragment::discard`), as an M4 keeps a discarded thread a lane of its quad's operations.
/// So such an operation reads its quad with no vote, the quad's active lanes being all four, as
/// Metal counts them. An entry Metal runs once per sample (`fragment::per_sample`) keeps the votes.
pub fn whole_quads(
    cx: &mut SelectCx<'_>,
    strategy: &[Option<Strategy>],
    m: &mut Meter,
) -> Result<(), Error> {
    let quad = |s: &Strategy| matches!(s, Strategy::Simd(SimdStrategy::Group(g)) if group::fragment_quad(g));
    if cx.stage != Stage::Fragment
        || !strategy.iter().flatten().any(quad)
        || crate::lower::fragment::per_sample(cx)
    {
        return Ok(());
    }
    let u = uniform::Uniformity::of(cx.entry, cx.body, m)?;
    let mut whole = BTreeSet::new();
    for (b, block) in cx.body.blocks.iter() {
        if !u.is_uniform(b) {
            continue;
        }
        for id in block.inst_ids() {
            let Some(v) = cx.values.inst(id) else {
                continue;
            };
            if strategy
                .get(v.index())
                .and_then(Option::as_ref)
                .is_some_and(quad)
            {
                whole.insert(v);
            }
        }
    }
    m.release(u.ledger_bytes());
    cx.families.simd.whole_quads = whole;
    Ok(())
}

/// Whether the operation `at` is a fragment's quad operation whose quad is whole
/// ([`whole_quads`]).
pub(super) fn whole_quad_at(cx: &EmitCx<'_>, at: InstId) -> bool {
    cx.plan
        .values
        .inst(at)
        .is_some_and(|v| cx.plan.families.simd.whole_quads.contains(&v))
}

/// Whether the operation `at` reads other lanes through the ballot of its live lanes
/// ([`partly_live`]).
pub(super) fn partly_live_at(cx: &EmitCx<'_>, at: InstId) -> bool {
    cx.plan
        .values
        .inst(at)
        .is_some_and(|v| cx.plan.families.simd.partly_live.contains(&v))
}

/// Plan a used parameter this family owns (`select::param_family`), each a fragment function's
/// (every integer form of it is held in a `u32`):
///
/// - `[[thread_index_in_quadgroup]]`. Vulkan's quad scope is four adjacent subgroup lanes from a
///   multiple of 4, which in a fragment shader are the 2x2 pixels in Metal's order (top left, top
///   right, bottom left, bottom right; the quad operations' model), so where the profile has
///   fragment subgroups that hold whole quads the value is `SubgroupLocalInvocationId & 3`, and the
///   class is raised with the basic class alone. Where it has none (no fragment subgroups, or ones
///   narrower than a quad), the lane is the pixel's place in its aligned 2x2 block, `(x & 1) + 2 *
///   (y & 1)` of the pixel `FragCoord` lies in, and raises no class. That is what Metal gives: our
///   probe read the lane of every written pixel equal to it on the M3 Max, single-sampled,
///   at 4x pixel rate, and at 4x sample rate (where `FragCoord` is a sample's position inside its
///   pixel). A quad operation in the same function still needs the tier and is refused by it.
/// - `[[thread_index_in_simdgroup]]`. Which pixels share a simdgroup, and in what order, is Apple's
///   rasterizer's packing: on the M3 Max and the M4 a primitive that covers an aligned 8x4 block
///   whole runs it as one simdgroup, its quads in row-major order, and elsewhere the covered quads
///   of several blocks and primitives share one. Nothing in Vulkan states or pins a device's
///   packing. The lane is the tier's simdgroup lane and raises the basic class. It is the lane of
///   the device's grouping, which Metal does not rule out: the function records
///   `PrecisionNote::FragmentSimdgroupGrouping`, as its simdgroup operations do, and [`finish`]
///   refuses it `Unsupported{SimdWidth}` unless the stage's tier gives 32-lane simdgroups (a
///   32-lane subgroup, or a 32-lane cluster of a wider one, whose lane is the subgroup lane's low
///   five bits). Under the loop
///   over the covered samples one invocation runs a pixel's samples, each of which Metal runs in a
///   lane of its own, so it is refused there `NotImplemented(PixelCoverageAtSampleRate)`.
/// - `[[threads_per_simdgroup]]`: 32, as `get_simdgroup_size` is, raising no class.
pub fn select_param(cx: &mut SelectCx<'_>, b: &crate::air::abi::ParamBinding) -> Result<(), Error> {
    use crate::air::abi::{Builtin, ParamRole};
    let builtin = match &b.role {
        ParamRole::Builtin(
            builtin @ (Builtin::ThreadIndexInQuadgroup
            | Builtin::ThreadIndexInSimdgroup
            | Builtin::ThreadsPerSimdgroup),
        ) if cx.stage == Stage::Fragment => *builtin,
        _ => return Err(cx.invariant("a parameter routed to the simd family it does not own")),
    };
    let ty = cx
        .value_ty(Value::Arg(crate::air::ir::ArgIdx::new(b.param)))
        .ok_or_else(|| cx.invariant("a fragment lane parameter without a type"))?;
    let form = form_on(cx.entry, cx.profile, ty)?;
    if !form.is_int() || form.lanes != 1 || form.container() != 32 {
        return Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::StageIo)).into());
    }
    let ops = match builtin {
        Builtin::ThreadsPerSimdgroup => return Ok(()),
        Builtin::ThreadIndexInSimdgroup => {
            if crate::lower::fragment::loops_samples(cx) {
                return Err(
                    Unsupported::not_implemented(Feature::PixelCoverageAtSampleRate).into(),
                );
            }
            // The device's own grouping, announced as its operations' is ([`finish`] checks
            // the tier gives 32-lane simdgroups).
            cx.families.simd.device_lane = true;
            cx.families.simd.simdgroup_members = true;
            OpBits::BASIC
        }
        _ => {
            // A quad is four adjacent subgroup lanes from a multiple of 4 wherever the stage has
            // subgroups (Vulkan 9.27.9), so the lane needs subgroups holding whole quads, of the
            // basic class alone.
            let model = &cx.profile.fields().subgroup;
            let lane = subgroup::tier(&SimdClass::LaneArithmetic, Stage::Fragment, model);
            let whole =
                lane != Ok(Tier::PartialFragment) || model.fragment.api_size.is_multiple_of(4);
            if lane.is_err() || !whole {
                let coord = crate::lower::stage_io::frag_coord(cx);
                cx.families.simd.quad_lane_coord = Some(coord);
                return Ok(());
            }
            OpBits::BASIC
        }
    };
    cx.simd = raise(cx.simd, ops);
    let simd = &mut cx.families.simd;
    simd.direct_ops = simd.direct_ops.union(ops);
    simd.cluster_ops = simd.cluster_ops.union(ops);
    simd.groups = true;
    Ok(())
}

/// The value of a fragment lane parameter ([`select_param`]), a `u32`: the simdgroup width 32;
/// the simdgroup lane, the subgroup's Metal lane; or the quad lane, the subgroup lane's low two
/// bits, or, without fragment subgroups that hold whole quads, `(x & 1) | ((y & 1) << 1)` of the
/// pixel `FragCoord` lies in.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &crate::air::abi::ParamBinding,
) -> BResult<SValue> {
    use crate::air::abi::{Builtin, ParamRole};
    use crate::spirv::ir::{BinOp, ConvOp};
    match b.role {
        ParamRole::Builtin(Builtin::ThreadsPerSimdgroup) => return group::k(fb, group::SIMD_WIDTH),
        ParamRole::Builtin(Builtin::ThreadIndexInSimdgroup) => {
            return group::lane(cx, fb)?.metal(fb);
        }
        _ => {}
    }
    let Some(coord) = cx.plan.families.simd.quad_lane_coord else {
        let ln = group::lane(cx, fb)?;
        return group::bin_k(fb, BinOp::BitwiseAnd, ln.l, 3);
    };
    let frag = crate::lower::stage_io::object_global(cx, coord)?;
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, crate::spirv::ir::MemAccess::default())?;
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    // `FragCoord` is the pixel's center, or a sample's position inside the pixel: its integer
    // part is the pixel.
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    let x = group::bin_k(fb, BinOp::BitwiseAnd, x, 1)?;
    let y = group::bin_k(fb, BinOp::BitwiseAnd, y, 1)?;
    let y = group::bin_k(fb, BinOp::ShiftLeftLogical, y, 1)?;
    fb.binary(BinOp::BitwiseOr, x, y)
}

/// Every operand must be a value emission produces in its lowered representation: a constant
/// `lower::arith` materializes, or a value of a type lowered on the profile.
fn check_operands(cx: &SelectCx<'_>, operands: &[Value]) -> Result<(), Error> {
    for v in operands {
        if matches!(v, Value::Const(_)) && cx.constant(*v).is_none() {
            return Err(not_lowered());
        }
        let ty = cx
            .value_ty(*v)
            .ok_or_else(|| cx.invariant("a simd operand without a type"))?;
        cx.lowered_repr(*v, ty)?;
    }
    Ok(())
}

/// Select the strategy of instruction `inst` of this family; a memory operand's candidate slots
/// are metered in `meter`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    meter: &mut Meter,
) -> Result<SimdStrategy, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Err(cx.invariant("a non-call routed to the simd family"));
    };
    let op = match &call.callee {
        Callee::Air { op, .. } => op,
        Callee::Llvm {
            op: LlvmIntrinsic::Agx(a),
            ..
        } => return agx::select(cx, at, inst, call, a, meter).map(SimdStrategy::Agx),
        _ => return Err(cx.invariant("a non-simd call routed to the simd family")),
    };
    if let AirIntrinsic::SimdMatrix(m) = op {
        return matrix::select(cx, at, call, m, meter).map(SimdStrategy::Matrix);
    }
    check_operands(cx, &call.args)?;
    match op {
        AirIntrinsic::SimdgroupSize(_) => {
            let form = form_on(cx.entry, cx.profile, inst.ty)?;
            if !form.is_int() {
                return Err(not_lowered());
            }
            Ok(SimdStrategy::Width(form))
        }
        AirIntrinsic::BoolReduce(b) => Ok(SimdStrategy::VectorVote { all: b.all }),
        // The asserted `true` stands for the result only where nothing observes it (it fed
        // `llvm.assume`, which decode drops, directly or through phis). A result read anywhere else
        // is a runtime uniformity test, which is not lowered.
        AirIntrinsic::IsUniform(_) if !cx.is_observed(at) => Ok(SimdStrategy::Assumed),
        AirIntrinsic::IsUniform(_) => Err(not_lowered()),
        AirIntrinsic::Group(g) => select_group(cx, at, inst, call, g).map(SimdStrategy::Group),
        _ => Err(cx.invariant("a non-simd intrinsic routed to the simd family")),
    }
}

/// The combining operation of a reduction or prefix sum `op` on values of form `f` with integer
/// sign token `sign`, or `None` for an operation that combines nothing.
fn combine(op: GroupOp, f: Form, sign: Option<Sign>) -> Result<Option<GroupArith>, Error> {
    use GroupArith as A;
    let signed = sign == Some(Sign::Signed);
    let r = match (op, f.held) {
        (
            GroupOp::Sum | GroupOp::PrefixInclusiveSum | GroupOp::PrefixExclusiveSum,
            Held::Int { .. },
        ) => A::IAdd,
        (
            GroupOp::Sum | GroupOp::PrefixInclusiveSum | GroupOp::PrefixExclusiveSum,
            Held::Float(_),
        ) => A::FAdd,
        (
            GroupOp::Product | GroupOp::PrefixInclusiveProduct | GroupOp::PrefixExclusiveProduct,
            Held::Int { .. },
        ) => A::IMul,
        (
            GroupOp::Product | GroupOp::PrefixInclusiveProduct | GroupOp::PrefixExclusiveProduct,
            Held::Float(_),
        ) => A::FMul,
        (GroupOp::Min, Held::Int { .. }) if signed => A::SMin,
        (GroupOp::Min, Held::Int { .. }) => A::UMin,
        (GroupOp::Min, Held::Float(_)) => A::FMin,
        (GroupOp::Max, Held::Int { .. }) if signed => A::SMax,
        (GroupOp::Max, Held::Int { .. }) => A::UMax,
        (GroupOp::Max, Held::Float(_)) => A::FMax,
        (GroupOp::And, Held::Int { .. }) => A::BitwiseAnd,
        (GroupOp::Or, Held::Int { .. }) => A::BitwiseOr,
        (GroupOp::Xor, Held::Int { .. }) => A::BitwiseXor,
        (GroupOp::And, Held::Bool) => A::LogicalAnd,
        (GroupOp::Or, Held::Bool) => A::LogicalOr,
        (GroupOp::Xor, Held::Bool) => A::LogicalXor,
        (
            GroupOp::Sum
            | GroupOp::PrefixInclusiveSum
            | GroupOp::PrefixExclusiveSum
            | GroupOp::Product
            | GroupOp::PrefixInclusiveProduct
            | GroupOp::PrefixExclusiveProduct
            | GroupOp::Min
            | GroupOp::Max
            | GroupOp::And
            | GroupOp::Or
            | GroupOp::Xor,
            _,
        ) => return Err(not_lowered()),
        _ => return Ok(None),
    };
    Ok(Some(r))
}

/// Whether operation `op` moves or combines values of its form across lanes (a ballot and the
/// active mask only read `bool`s).
fn moves_values(op: GroupOp) -> bool {
    !matches!(
        op,
        GroupOp::All
            | GroupOp::Any
            | GroupOp::Ballot
            | GroupOp::IsFirst
            | GroupOp::ActiveThreadsMask
    )
}

fn select_group(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    call: &Call,
    g: &Group,
) -> Result<GroupPlan, Error> {
    let args = &call.args;
    let id = cx
        .values
        .inst(at)
        .ok_or_else(|| cx.invariant("an instruction out of the value range"))?;
    // Metal moves or combines no 64-bit integer across lanes (`metal_simdgroup` and
    // `metal_quadgroup` enable their functions for 8- to 32-bit integers, `half`, `float`, and
    // `double` only), so such a call is not Metal's, on any profile. A mask is not such a value:
    // the ballot and the active-threads mask read only `bool`s.
    if moves_values(g.op) && g.ty.is_some_and(|t| t.elem == Elem::I64) {
        return Err(Error::InvalidInput(InputError::new(
            InputErrorKind::Intrinsic(IntrinsicError::SignatureMismatch),
            Location::function(cx.entry.ep.name.clone(), None, Some(at.raw())),
        )));
    }
    // A simdgroup's 64-bit ballot or active mask is two words without `shaderInt64`.
    let mask = matches!(g.op, GroupOp::Ballot | GroupOp::ActiveThreadsMask);
    let form = if mask && g.scope == GroupScope::Simdgroup {
        wide_form_on(cx.entry, cx.profile, inst.ty)?
    } else {
        form_on(cx.entry, cx.profile, inst.ty)?
    };
    if form.container() == 64
        && moves_values(g.op)
        && !cx
            .profile
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderSubgroupExtendedTypes)
    {
        return Err(Unsupported::needs(
            Feature::SubgroupOperations,
            Capability::Feature(VkFeature::ShaderSubgroupExtendedTypes),
        )
        .into());
    }
    let arith = combine(g.op, form, g.sign)?;
    // The constant lane of a quad broadcast or mask of a quad shuffle-xor selects the quad
    // instructions where the stage takes them; a lane-varying one is a shuffle. A
    // shuffle-and-fill's constant modulo (a `ushort`) takes Metal's constant rule, a runtime one
    // its runtime rule.
    let operand = |i: usize, mask: u64| match args.get(i) {
        Some(v @ Value::Const(_)) => match cx.constant(*v) {
            Some(LoweredConst::Scalar { bits, .. }) => Some((bits & mask) as u32),
            _ => None,
        },
        _ => None,
    };
    let konst = match (g.scope, g.op) {
        (GroupScope::Quad, GroupOp::Broadcast | GroupOp::ShuffleXor) => operand(1, 3),
        (_, GroupOp::ShuffleAndFillUp | GroupOp::ShuffleAndFillDown) => operand(3, 0xFFFF),
        _ => None,
    };
    let plan = GroupPlan {
        scope: g.scope,
        op: g.op,
        form,
        arith,
        konst,
        quad_instructions: cx.stage == Stage::Kernel
            && cx.profile.fields().subgroup.quad_instructions(cx.stage),
        helpers: cx.stage == Stage::Fragment,
    };
    group::check(&plan).map_err(|()| not_lowered())?;
    if matches!(
        arith,
        Some(GroupArith::FAdd | GroupArith::FMul | GroupArith::FMin | GroupArith::FMax)
    ) {
        // The combination is float arithmetic on every tier: without all of `nnan`, `ninf`, and
        // `nsz` on the call, the entry keeps signed zeros, infinities, and NaNs.
        arith::fp_env(cx, call.fmf, form);
    }
    // A float sum's or product's association order is the device's; Metal's measured order is a
    // pairwise tree over the lanes for the reductions. A quad's reads its lanes on every tier in
    // Metal's order (measured): a prefix sum or product one by one in lane order, a reduction as
    // `(x0 op x1) op (x2 op x3)`.
    if matches!(arith, Some(GroupArith::FAdd | GroupArith::FMul)) && g.scope != GroupScope::Quad {
        cx.precision.insert(PrecisionNote::ReductionOrder);
    }
    let simd = &mut cx.families.simd;
    simd.direct_ops = simd.direct_ops.union(direct_ops(&plan));
    simd.cluster_ops = simd.cluster_ops.union(cluster_ops(&plan));
    simd.groups = true;
    simd.observes_lanes = true;
    simd.simdgroup_members |= plan.scope == GroupScope::Simdgroup;
    cx.tangled.insert(id);
    Ok(plan)
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &SimdStrategy,
) -> BResult<()> {
    match s {
        SimdStrategy::Agx(a) => agx::emit(cx, fb, at, a),
        SimdStrategy::Matrix(m) => matrix::emit(cx, fb, at, m),
        _ => group::emit(cx, fb, at, s),
    }
}
