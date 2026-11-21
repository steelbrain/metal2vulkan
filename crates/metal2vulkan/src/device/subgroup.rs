//! Simd usage classes, tiers, and subgroup requirements.
//!
//! The Metal simdgroup width is always 32. [`tier`] maps a function's [`SimdClass`], its stage, and
//! the profile's [`SubgroupModel`] to one [`Tier`]; the artifact records the resulting
//! [`SubgroupRequirement`]. Local size is never an input.

use crate::device::derive::SUBGROUP_OPS;
use crate::device::requirements::{
    Requirements, StageCreateFlag, SubgroupOp, VkFeature, VkLimit, VkProperty,
};
use crate::error::{Capability, Feature, Unsupported};
use crate::pipeline::sym::Stage;

/// A set of subgroup operation classes (`VkSubgroupFeatureFlags` bit layout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpBits(pub u32);

impl OpBits {
    pub const NONE: OpBits = OpBits(0);
    pub const BASIC: OpBits = OpBits(1 << 0);
    pub const VOTE: OpBits = OpBits(1 << 1);
    pub const ARITHMETIC: OpBits = OpBits(1 << 2);
    pub const BALLOT: OpBits = OpBits(1 << 3);
    pub const SHUFFLE: OpBits = OpBits(1 << 4);
    pub const SHUFFLE_RELATIVE: OpBits = OpBits(1 << 5);
    pub const CLUSTERED: OpBits = OpBits(1 << 6);
    pub const QUAD: OpBits = OpBits(1 << 7);

    /// The bit of one operation class.
    pub const fn of(op: SubgroupOp) -> OpBits {
        match op {
            SubgroupOp::Basic => OpBits::BASIC,
            SubgroupOp::Vote => OpBits::VOTE,
            SubgroupOp::Arithmetic => OpBits::ARITHMETIC,
            SubgroupOp::Ballot => OpBits::BALLOT,
            SubgroupOp::Shuffle => OpBits::SHUFFLE,
            SubgroupOp::ShuffleRelative => OpBits::SHUFFLE_RELATIVE,
            SubgroupOp::Clustered => OpBits::CLUSTERED,
            SubgroupOp::Quad => OpBits::QUAD,
        }
    }

    /// Every class in `other` is in `self`.
    pub const fn contains(self, other: OpBits) -> bool {
        self.0 & other.0 == other.0
    }

    /// The classes in either set.
    pub const fn union(self, other: OpBits) -> OpBits {
        OpBits(self.0 | other.0)
    }
}

/// How a function uses simdgroup / quad semantics.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimdClass {
    /// No simdgroup semantics.
    None,
    /// Only per-lane arithmetic on simd builtins (e.g. lane index).
    LaneArithmetic,
    /// Quad operations. In compute and vertex this is tiered as `CrossLane(QUAD)`.
    Quad,
    /// Cross-lane operations of these classes.
    CrossLane(OpBits),
    /// Simdgroup matrix operations.
    Matrix,
}

/// How simdgroups are realized for a (class, stage, profile).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// No subgroup operations are emitted.
    NoSubgroup,
    /// Lane builtins are derived from the linear invocation index.
    Linear,
    /// A pinned 32-wide subgroup maps one-to-one onto a simdgroup.
    Direct,
    /// A wider pinned subgroup is split into 32-lane clusters.
    Cluster32,
    /// Fragment with possibly partially occupied simdgroups.
    PartialFragment,
    /// A kernel whose stage cannot run a 32-lane simdgroup on subgroups (the width cannot be
    /// pinned, or an operation class is missing): simdgroups of the linear threads
    /// `[32k, 32k + 32)` emulated through workgroup memory between workgroup barriers.
    Emulated,
}

/// Minimum and maximum subgroup size a stage can run with (powers of two in between).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SubgroupSizes {
    pub min: u32,
    pub max: u32,
}

/// Subgroup facts of one stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StageSubgroup {
    /// `subgroupSupportedStages` contains the stage.
    pub supported: bool,
    /// Operation classes supported in the stage.
    pub ops: OpBits,
    /// The API-reported `subgroupSize`.
    pub api_size: u32,
    /// The sizes the stage may actually run with.
    pub sizes: SubgroupSizes,
    /// A required subgroup size can be pinned for the stage.
    pub pinnable: bool,
    /// Full subgroups can be required for the stage.
    pub full_subgroups: bool,
}

/// The subgroup facts of a stage the profile does not record (an object or a mesh function): no
/// subgroup support.
static UNMODELED_STAGE: StageSubgroup = StageSubgroup {
    supported: false,
    ops: OpBits::NONE,
    api_size: 0,
    sizes: SubgroupSizes { min: 0, max: 0 },
    pinnable: false,
    full_subgroups: false,
};

/// The profile's subgroup model ([`DeviceProfile::subgroups`]).
///
/// [`DeviceProfile::subgroups`]: crate::device::DeviceProfile::subgroups
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SubgroupModel {
    pub vertex: StageSubgroup,
    pub fragment: StageSubgroup,
    pub kernel: StageSubgroup,
    /// `subgroupQuadOperationsInAllStages`.
    pub quad_all_stages: bool,
    /// `shaderMaximalReconvergence`.
    pub maximal_reconvergence: bool,
    /// `shaderSubgroupUniformControlFlow`.
    pub uniform_control_flow: bool,
    /// `shaderQuadControl`: a fragment entry may declare `RequireFullQuadsKHR`, so helpers take
    /// part in its group operations as Metal's do.
    pub quad_control: bool,
    /// `maxComputeWorkgroupSubgroups`.
    pub max_compute_workgroup_subgroups: u32,
}

/// What a function needs from its dispatch's subgroup configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupRequirement {
    /// A required subgroup size, if the tier pins one.
    pub size: Option<u32>,
    /// Full subgroups are required.
    pub full_subgroups: bool,
    /// Every dispatch region's local size must be a multiple of this (1: no constraint).
    pub local_size_multiple: u32,
}

impl SubgroupRequirement {
    /// No subgroup requirement.
    pub const NONE: SubgroupRequirement = SubgroupRequirement {
        size: None,
        full_subgroups: false,
        local_size_multiple: 1,
    };

    /// Add to `r` what this requirement of a `stage` function needs of the device beyond its
    /// module. The pipeline states the policy outside the SPIR-V
    /// (`StageModule::required_subgroup_size` and `stage_flags`), so derivation cannot find it:
    ///
    /// - a pinned size (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`) needs
    ///   `subgroupSizeControl`, the stage in `requiredSubgroupSizeStages`, and the size inside
    ///   `[minSubgroupSize, maxSubgroupSize]`;
    /// - full subgroups (`REQUIRE_FULL_SUBGROUPS`) need the stage flag, whose feature is
    ///   `computeFullSubgroups`.
    ///
    /// A requirement that pins nothing and needs no full subgroups adds nothing. A limit `r`
    /// already names keeps the stricter value.
    pub fn add_device_facts(&self, stage: Stage, r: &mut Requirements) {
        if let Some(size) = self.size {
            r.features.insert(VkFeature::SubgroupSizeControl);
            r.properties
                .insert(VkProperty::RequiredSubgroupSizeStage(stage));
            let size = u64::from(size);
            // The device's minimum at most the size, its maximum at least the size.
            let min = r.limits.entry(VkLimit::MinSubgroupSize).or_insert(size);
            *min = (*min).min(size);
            let max = r.limits.entry(VkLimit::MaxSubgroupSize).or_insert(size);
            *max = (*max).max(size);
        }
        if self.full_subgroups {
            r.stage_flags.insert(StageCreateFlag::RequireFullSubgroups);
        }
    }
}

impl SubgroupModel {
    /// The subgroup facts of one Metal stage.
    pub fn stage(&self, stage: Stage) -> &StageSubgroup {
        match stage {
            Stage::Vertex => &self.vertex,
            Stage::Fragment => &self.fragment,
            Stage::Kernel => &self.kernel,
            // The profile records no task or mesh stage's subgroup facts: an object or mesh
            // function lowers as on a device without subgroups there.
            Stage::Object | Stage::Mesh => &UNMODELED_STAGE,
        }
    }

    /// Whether a `stage` function can take Vulkan's quad instructions
    /// (`OpGroupNonUniformQuadBroadcast`, `OpGroupNonUniformQuadSwap`): the stage has subgroups
    /// of the quad class, and a vertex stage also `subgroupQuadOperationsInAllStages`, as the
    /// simd tiering asks of the quad class. Without them a constant lane's quad broadcast or
    /// shuffle-xor is a shuffle (`lower::simd`).
    pub fn quad_instructions(&self, stage: Stage) -> bool {
        let st = self.stage(stage);
        st.supported
            && st.ops.contains(OpBits::QUAD)
            && (stage != Stage::Vertex || self.quad_all_stages)
    }

    /// The device's `[minSubgroupSize, maxSubgroupSize]`: the range of a stage that can pin a
    /// size (every such stage has that range), or the kernel's API size where none can.
    pub fn pinnable_sizes(&self) -> SubgroupSizes {
        [&self.kernel, &self.fragment, &self.vertex]
            .into_iter()
            .find(|s| s.pinnable)
            .unwrap_or(&self.kernel)
            .sizes
    }
}

impl StageSubgroup {
    /// The stage can run with subgroup size `s`: pinned inside the size-control range, or the
    /// API-constant size of a SPIR-V 1.5 module when the stage cannot be pinned.
    pub fn fits(&self, s: u32) -> bool {
        self.sizes.min <= s && s <= self.sizes.max
    }

    /// The size a Cluster32 tier runs with: the smallest of 64 and 128 the stage can run with.
    fn cluster_size(&self) -> Option<u32> {
        [64, 128].into_iter().find(|&s| self.fits(s))
    }
}

/// The operation classes a usage class needs in a stage, and the refused feature if they are
/// missing. Compute and vertex quads are cross-lane quad operations over consecutive lanes.
///
/// Simdgroup matrices are lowered with shuffles over a 32-lane subgroup (`lower::simd`); the
/// operation set they need is `BASIC | SHUFFLE` until that lowering names a larger one.
fn needed_ops(class: &SimdClass) -> (OpBits, Feature) {
    match *class {
        SimdClass::None => (OpBits::NONE, Feature::SubgroupOperations),
        SimdClass::LaneArithmetic => (OpBits::BASIC, Feature::SubgroupOperations),
        SimdClass::Quad => (OpBits::BASIC.union(OpBits::QUAD), Feature::QuadOperations),
        SimdClass::CrossLane(ops) => (OpBits::BASIC.union(ops), Feature::SubgroupOperations),
        SimdClass::Matrix => (
            OpBits::BASIC.union(OpBits::SHUFFLE),
            Feature::SimdgroupMatrix,
        ),
    }
}

/// The simd tier of a (usage class, stage, profile). It depends on the class, the stage, and the
/// profile, never on the local size.
///
/// - `None` needs nothing; compute `LaneArithmetic` derives lanes from the linear index (`Linear`).
/// - Every other use needs the stage in `subgroupSupportedStages` and the class's operations
///   (`Unsupported{SubgroupOperations | QuadOperations | SimdgroupMatrix}` otherwise); quads
///   outside fragment and compute also need `subgroupQuadOperationsInAllStages`.
/// - Compute (quads included) needs full subgroups and a 32-wide subgroup (`Direct`) or a
///   64/128-wide one split into 32-lane clusters (`Cluster32`); else `Unsupported{SimdWidth}`.
/// - Vertex and fragment: 32 → `Direct`; 64/128 → `Cluster32`; narrower → `PartialFragment` (the
///   simdgroup is the whole subgroup, upper lanes permanently inactive, which Metal permits for a
///   partially occupied simdgroup). Quads need `4 | S`.
/// - Simdgroup matrices are `Direct` only.
/// - A kernel that none of these admits is `Emulated`: its simdgroups need no subgroup feature,
///   only workgroup memory and barriers, which every compute stage has. Whether the function's
///   operations can be emulated exactly (every one in workgroup-uniform control flow) and whether
///   the workgroup memory fits are the function's facts, which `select` checks.
pub fn tier(class: &SimdClass, stage: Stage, model: &SubgroupModel) -> Result<Tier, Unsupported> {
    match subgroup_tier(class, stage, model) {
        Err(_) if stage == Stage::Kernel && emulable(class) => Ok(Tier::Emulated),
        t => t,
    }
}

/// The tier of a function compiled under `VariantKey::emulated_simdgroups`: `Emulated` for a kernel
/// whose simdgroups it realizes, whatever tier the profile's subgroups would give them, since the
/// emulated tier is exact for every threadgroup size (a partial simdgroup's active lanes are those
/// below the thread count); [`tier`] for every other class and stage.
pub fn emulated_tier(
    class: &SimdClass,
    stage: Stage,
    model: &SubgroupModel,
) -> Result<Tier, Unsupported> {
    if stage == Stage::Kernel && emulable(class) {
        Ok(Tier::Emulated)
    } else {
        tier(class, stage, model)
    }
}

/// Whether a kernel of class `c` has simdgroups the `Emulated` tier realizes.
fn emulable(c: &SimdClass) -> bool {
    matches!(
        c,
        SimdClass::Quad | SimdClass::CrossLane(_) | SimdClass::Matrix
    )
}

/// The subgroup tier of a (usage class, stage, profile): `tier` without the emulation.
fn subgroup_tier(
    class: &SimdClass,
    stage: Stage,
    model: &SubgroupModel,
) -> Result<Tier, Unsupported> {
    let class = match (class, stage) {
        (SimdClass::None, _) => return Ok(Tier::NoSubgroup),
        // An object or mesh threadgroup's simdgroups are a kernel's, 32 consecutive linear threads
        // (an execution width of 32).
        (SimdClass::LaneArithmetic, Stage::Kernel | Stage::Object | Stage::Mesh) => {
            return Ok(Tier::Linear)
        }
        // Metal compute and vertex quads are four consecutive linear threads, which Vulkan does
        // not relate to subgroup lanes: tier them as cross-lane quad operations.
        (SimdClass::Quad, Stage::Kernel | Stage::Vertex) => SimdClass::CrossLane(OpBits::QUAD),
        (c, _) => *c,
    };
    let st = model.stage(stage);
    let (ops, feature) = needed_ops(&class);
    if !st.supported {
        return Err(Unsupported::needs(
            feature,
            Capability::Property(VkProperty::SubgroupStage(stage)),
        ));
    }
    if !st.ops.contains(ops) {
        // The device's classes are the compute stage's, which Vulkan always lists: a class the
        // device has but the stage lacks is one the stage would have where the device listed it.
        let mut missing = Vec::new();
        for op in SUBGROUP_OPS {
            let bit = OpBits::of(op);
            if !ops.contains(bit) || st.ops.contains(bit) {
                continue;
            }
            let need = if model.kernel.ops.contains(bit) {
                Capability::Property(VkProperty::SubgroupStage(stage))
            } else {
                Capability::Property(VkProperty::SubgroupOperation(op))
            };
            if !missing.contains(&need) {
                missing.push(need);
            }
        }
        return Err(Unsupported::needs_any_of(feature, vec![missing]));
    }
    if ops.contains(OpBits::QUAD) && stage == Stage::Vertex && !model.quad_all_stages {
        return Err(Unsupported::needs(
            Feature::QuadOperations,
            Capability::Property(VkProperty::SubgroupQuadOperationsInAllStages),
        ));
    }
    let matrix = class == SimdClass::Matrix;
    let width = || Unsupported::needs_any_of(Feature::SimdWidth, vec![width_needs(stage, st)]);
    if stage == Stage::Kernel {
        if !st.full_subgroups {
            return Err(width());
        }
        if st.fits(32) {
            return Ok(Tier::Direct);
        }
        if !matrix && st.cluster_size().is_some() {
            return Ok(Tier::Cluster32);
        }
        return Err(width());
    }
    if st.fits(32) {
        return Ok(Tier::Direct);
    }
    if matrix {
        return Err(width());
    }
    if st.cluster_size().is_some() {
        return Ok(Tier::Cluster32);
    }
    // Not pinnable to 32 or more: the API-constant size is below 32.
    let s = st.api_size;
    if s == 0 || s >= 32 || !s.is_power_of_two() {
        return Err(width());
    }
    if ops.contains(OpBits::QUAD) && !s.is_multiple_of(4) {
        return Err(Unsupported::needs(
            Feature::QuadOperations,
            Capability::Limit {
                limit: VkLimit::MinSubgroupSize,
                required: 4,
                available: u64::from(s),
            },
        ));
    }
    Ok(Tier::PartialFragment)
}

/// What stage `stage` (facts `st`) lacks to run 32-lane subgroups (`Tier::Direct`): a required
/// subgroup size where the stage takes none, full subgroups for a kernel, and 32 within its
/// sizes.
pub fn width_needs(stage: Stage, st: &StageSubgroup) -> Vec<Capability> {
    let mut needs = Vec::new();
    if !st.pinnable {
        needs.push(Capability::Feature(VkFeature::SubgroupSizeControl));
        needs.push(Capability::Property(VkProperty::RequiredSubgroupSizeStage(
            stage,
        )));
    }
    if stage == Stage::Kernel && !st.full_subgroups {
        needs.push(Capability::Feature(VkFeature::ComputeFullSubgroups));
    }
    if st.sizes.max < 32 {
        needs.push(Capability::Limit {
            limit: VkLimit::MaxSubgroupSize,
            required: 32,
            available: u64::from(st.sizes.max),
        });
    }
    if st.sizes.min > 32 {
        needs.push(Capability::Limit {
            limit: VkLimit::MinSubgroupSize,
            required: 32,
            available: u64::from(st.sizes.min),
        });
    }
    needs
}

/// What a tier asks of the pipeline's subgroup configuration. Compute `Direct`/`Cluster32` need
/// full subgroups and a local size that is a multiple of the subgroup size; a size is pinned only
/// where the stage is pinnable (otherwise the API-constant size is the one `tier` accepted).
pub fn requirement(t: Tier, stage: Stage, model: &SubgroupModel) -> SubgroupRequirement {
    let st = model.stage(stage);
    let size = match t {
        Tier::NoSubgroup | Tier::Linear | Tier::PartialFragment | Tier::Emulated => {
            return SubgroupRequirement::NONE
        }
        Tier::Direct => 32,
        Tier::Cluster32 => st.cluster_size().unwrap_or(st.api_size),
    };
    let compute = stage == Stage::Kernel;
    SubgroupRequirement {
        size: st.pinnable.then_some(size),
        full_subgroups: compute,
        local_size_multiple: if compute { size } else { 1 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_bits_are_the_vulkan_flag_layout() {
        let ops = [
            SubgroupOp::Basic,
            SubgroupOp::Vote,
            SubgroupOp::Arithmetic,
            SubgroupOp::Ballot,
            SubgroupOp::Shuffle,
            SubgroupOp::ShuffleRelative,
            SubgroupOp::Clustered,
            SubgroupOp::Quad,
        ];
        let mut all = OpBits::NONE;
        for (i, op) in ops.into_iter().enumerate() {
            assert_eq!(OpBits::of(op), OpBits(1 << i), "{op:?}");
            assert!(!all.contains(OpBits::of(op)));
            all = all.union(OpBits::of(op));
        }
        assert_eq!(all, OpBits(0xff));
    }

    #[test]
    fn subgroup_types_are_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<SimdClass>();
        send_sync::<Tier>();
        send_sync::<SubgroupModel>();
        send_sync::<SubgroupRequirement>();
    }
}
