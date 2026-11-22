//! Pipeline artifacts: what `compile_*_pipeline` returns.
//!
//! Artifacts are immutable plain data; SPIR-V words sit behind an `Arc`, so clones are cheap. Every
//! module is valid for the profile it was compiled against, under exactly `requirements`. Encoding
//! for disk caches is `cache::PipelineArtifact`, not serde.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::cache::{Hash256, PipelineKey, SpecializationKey};
use crate::device::requirements::Requirements;
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::graphics::GraphicsState;
use crate::pipeline::interface::{LevelReads, PipelineInterface, SampledLevels};
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::sym::SpecKey;
use crate::pipeline::vertex::VertexInputState;
use crate::vk;

/// A compiled render pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderPipeline {
    pub key: PipelineKey,
    /// The fingerprint of the profile it was compiled for: a cache load checks it.
    pub profile: Hash256,
    /// Vertex, then fragment when rasterization is enabled. A post-tessellation vertex function's
    /// pipeline has the synthesized vertex and tessellation control stages, then its evaluation
    /// stage, then the fragment stage.
    pub stages: Vec<StageModule>,
    pub layout: PipelineLayout,
    pub vertex_input: VertexInputState,
    pub graphics: GraphicsState,
    pub dependencies: Vec<RuntimeDependency>,
    pub interface: PipelineInterface,
    /// Everything the modules need from the device (a superset of the enable set).
    pub requirements: Requirements,
    /// Specialization and variant bits the pipeline can take.
    pub variant_bits: u32,
}

impl RenderPipeline {
    /// The bytes this pipeline keeps live: its own storage, its stages' and its dependencies',
    /// and each stage's SPIR-V words, constant blob, entry point, specialization constants,
    /// notes and attachment lists.
    ///
    /// This is what a bounded memo of compiled pipelines weighs an entry by where the entry is
    /// held rather than encoded. A memo that also stores the pipeline on disk has the encoding's
    /// length already and weighs by that; one that does not has this, which is the same quantity
    /// measured on the live object, so an entry is weighed the same whichever way it was kept and
    /// the memo's byte bound holds either way. Words a stage shares with another holder are
    /// counted here, as its encoding counts them.
    ///
    /// A floor rather than an exact total. The SPIR-V spans three orders of magnitude between
    /// pipelines and every word of it is counted; the layout, the vertex input, the graphics
    /// state, the interface, the requirements and each stage's strategy census are counted at
    /// their own sizes and their nested maps are not walked, so the figure is low by an amount
    /// bounded by Metal's slot and attachment counts, which no module can grow.
    pub fn footprint(&self) -> u64 {
        let stages: u64 = self.stages.iter().map(stage_footprint).sum();
        size_of::<RenderPipeline>() as u64
            + vector_bytes::<StageModule>(self.stages.capacity())
            + vector_bytes::<RuntimeDependency>(self.dependencies.capacity())
            + stages
    }
}

/// The heap bytes of a vector of `T` at `capacity`.
fn vector_bytes<T>(capacity: usize) -> u64 {
    capacity as u64 * size_of::<T>() as u64
}

/// The bytes `stage` keeps live beside its own storage, which its pipeline's `stages` vector
/// holds ([`RenderPipeline::footprint`]).
fn stage_footprint(stage: &StageModule) -> u64 {
    let clauses: u64 = stage
        .specialized_notes
        .iter()
        .map(|note| {
            vector_bytes::<SpecClause>(note.when.capacity())
                + note
                    .when
                    .iter()
                    .map(|clause| vector_bytes::<SpecTest>(clause.all.capacity()))
                    .sum::<u64>()
        })
        .sum();
    stage.entry_point.capacity() as u64
        + vector_bytes::<u32>(stage.spirv.len())
        + stage
            .constant_blob
            .as_ref()
            .map_or(0, |blob| blob.len() as u64)
        + vector_bytes::<SpecConstant>(stage.specialization.constants.capacity())
        + vector_bytes::<PrecisionNote>(stage.precision_notes.capacity())
        + vector_bytes::<SpecializedNote>(stage.specialized_notes.capacity())
        + clauses
        + vector_bytes::<u32>(stage.rounded_colors.capacity() + stage.own_pixel_textures.capacity())
        + vector_bytes::<LevelReads>(stage.level_reads.capacity())
        + (stage.level_reads.iter())
            .map(|reads| vector_bytes::<SampledLevels>(reads.sampled.capacity()))
            .sum::<u64>()
}

/// A compiled compute pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputePipeline {
    pub key: PipelineKey,
    /// The fingerprint of the profile it was compiled for: a cache load checks it.
    pub profile: Hash256,
    pub stage: StageModule,
    pub layout: PipelineLayout,
    pub dependencies: Vec<RuntimeDependency>,
    pub interface: PipelineInterface,
    /// Answers for Metal's per-pipeline compute queries.
    pub metal_info: ComputePipelineMetalInfo,
    pub requirements: Requirements,
    pub variant_bits: u32,
}

impl ComputePipeline {
    /// The bytes this pipeline keeps live: its own storage, its dependencies', and its stage's
    /// SPIR-V words, constant blob, entry point, specialization constants, notes and attachment
    /// lists, counted as [`RenderPipeline::footprint`] counts a render pipeline's: what a bounded
    /// memo weighs an entry by where the entry is held rather than encoded. A floor, as that is:
    /// the layout, the interface and the requirements are counted at their own sizes.
    pub fn footprint(&self) -> u64 {
        size_of::<ComputePipeline>() as u64
            + vector_bytes::<RuntimeDependency>(self.dependencies.capacity())
            + stage_footprint(&self.stage)
    }
}

/// One stage's final SPIR-V and the facts the consumer needs to create its shader stage.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageModule {
    pub stage: vk::ShaderStage,
    /// The entry point name (NUL-free by construction).
    pub entry_point: String,
    /// SPIR-V 1.5 words, relocated for this pipeline.
    pub spirv: Arc<[u32]>,
    /// The specialization constants with their defaults. A kernel's defaults are placeholders the
    /// binder replaces (a local size of 1): create each compute `VkPipeline` from a planned
    /// dispatch's [`DispatchRegion`](crate::bind::DispatchRegion), its `specialization`,
    /// `stage_flags`, and `required_subgroup_size`, never from this stage's defaults.
    pub specialization: SpecializationLayout,
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, when the tier pins a size. For a
    /// kernel, each [`DispatchRegion`](crate::bind::DispatchRegion) carries the value to create its
    /// pipeline with.
    pub required_subgroup_size: Option<u32>,
    /// Stage create flags (`REQUIRE_FULL_SUBGROUPS`). For a kernel, each
    /// [`DispatchRegion`](crate::bind::DispatchRegion) carries the flags to create its pipeline
    /// with: a region that is not whole subgroups runs on a variant without them. These flags at
    /// the default specialization's local size of 1 are invalid usage
    /// (`VUID-VkPipelineShaderStageCreateInfo-flags-02759`, or `-pNext-02757` with a required
    /// subgroup size).
    pub stage_flags: vk::PipelineShaderStageCreateFlags,
    pub reconvergence: Reconvergence,
    /// Where the output deliberately differs from exact Metal semantics.
    pub precision_notes: Vec<PrecisionNote>,
    /// Which lowering strategies produced the module (coverage gate).
    pub strategies: StrategyCensus,
    /// The constant data this stage reads at its `Synthetic(ConstantBlob)` binding, which the
    /// binder fills from `WriteSource::ConstantBlob { stage }`. Each stage has its own.
    pub constant_blob: Option<Arc<[u8]>>,
    /// The color attachments whose float writes this fragment stage rounds in the shader
    /// (`StageIo::rounded_colors`), sorted: a half-float attachment of a pipeline under
    /// `TextureWriteRoundingMode::NearestEven`, whose rounding no device's conversion gives. Each
    /// such write costs the stage's instructions per channel; empty for every other stage.
    pub rounded_colors: Vec<u32>,
    /// The notes of `precision_notes` that hold only under some specializations: a note whose
    /// every access keeps its near-equivalent only on the arms the module selects by its
    /// specialization constants (the bound sampler's word, `SpecKey::SamplerState`, and whether
    /// it is a nearest pixel sampler,
    /// `SpecKey::NearestPixelSampler`), with the condition under which a specialized
    /// module runs such an arm. Sorted by note, each note at most once. Empty for a module whose
    /// every note holds under every specialization. [`StageModule::precision_notes_bound`]
    /// gives the notes of the module as a pipeline specializes it.
    pub specialized_notes: Vec<SpecializedNote>,
    /// The texture slots this fragment stage reads only at the fragment's own pixel, sorted
    /// (`StageInterface::own_pixel_textures`): a draw's reads of such a slot lie inside the
    /// rectangle bounding its scissors, a pixel wider each way for its helper invocations, on a
    /// framebuffer below `2^24` pixels a side. Empty for every other stage.
    pub own_pixel_textures: Vec<u32>,
    /// The levels of its bound view each texture slot's reads reach, one entry a slot, sorted by
    /// slot (`StageInterface::level_reads`). Empty for a kernel and a stage the link makes.
    pub level_reads: Vec<LevelReads>,
    /// The private memory a thread of the stage keeps, in bytes
    /// (`StageInterface::private_bytes`): what `PrecisionNote::FunctionStorageAssumed`'s
    /// announcement names. 0 for a stage the link makes itself.
    pub private_bytes: u64,
    /// What [`spirv`](Self::spirv) contains: its functions, blocks, instructions, structured
    /// constructs and loops.
    pub stats: ModuleStats,
}

impl StageModule {
    /// The SPIR-V as little-endian bytes, for consumers that take a byte slice.
    pub fn spirv_bytes(&self) -> Vec<u8> {
        self.spirv.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The stage's precision notes where a pipeline creates it under `specialization` and a draw
    /// or dispatch binds the values `bound` gives, where a test of a key no pipeline's
    /// key carries (the bound format's class, `SpecKey::BoundTexelFormat`; a bound sampler's word,
    /// `SpecKey::BoundSamplerWord`; what the command's handles can name) reads the value `bound`
    /// gives for it (`BindingPlan::specialized_value`); a test of a key neither gives holds.
    pub fn precision_notes_bound(
        &self,
        specialization: &SpecializationKey,
        bound: &dyn Fn(SpecKey) -> Option<u64>,
    ) -> Vec<PrecisionNote> {
        self.precision_notes
            .iter()
            .copied()
            .filter(
                |&note| match self.specialized_notes.iter().find(|s| s.note == note) {
                    None => true,
                    Some(s) => s
                        .when
                        .iter()
                        .any(|clause| self.clause_holds(clause, specialization, bound)),
                },
            )
            .collect()
    }

    /// Whether every test of `clause` holds under `specialization`, a constant the key does not
    /// give reading its default, and a key no pipeline's key carries (the bound format's class, a
    /// bound sampler's word) reading the value `bound` gives.
    fn clause_holds(
        &self,
        clause: &SpecClause,
        specialization: &SpecializationKey,
        bound: &dyn Fn(SpecKey) -> Option<u64>,
    ) -> bool {
        let value = |key: SpecKey| -> Option<u64> {
            let Some(c) = self.specialization.constants.iter().find(|c| c.key == key) else {
                return bound(key);
            };
            let given = specialization
                .entries
                .iter()
                .find(|e| e.constant_id == c.id)
                .and_then(|e| {
                    let start = e.offset as usize;
                    let bytes = specialization.data.get(start..start + e.size as usize)?;
                    let mut le = [0u8; 8];
                    le.get_mut(..bytes.len())?.copy_from_slice(bytes);
                    Some(u64::from_le_bytes(le))
                });
            Some(given.unwrap_or(c.default))
        };
        clause.all.iter().all(|t| match value(t.key) {
            // A test of a constant the stage does not declare and `bound` does not give holds:
            // the conservative answer (a runtime gives the bound format's class,
            // `SpecKey::BoundTexelFormat`, and the bound samplers' words through `bound`).
            None => true,
            Some(v) => (v & u64::from(t.mask) == u64::from(t.bits)) == t.equal,
        })
    }
}

/// What one translated module contains: the counts the finalize verifier takes of it on every
/// translation, in every build.
///
/// They are the module [`StageModule::spirv`] holds: the verifier counts the module it checked,
/// and serialization and relocation assign literals without adding or removing an instruction.
/// [`CompiledFunction::module_stats`](crate::library::CompiledFunction::module_stats) states the
/// same counts for a function compiled on its own, which its linked stage repeats.
///
/// A caller reads them to watch the size of what it hands a driver: a larger module costs more to
/// compile on the drivers measured. Canonicalization inlines every direct call before the module
/// is emitted, so the counts are the whole program's, not one function's.
///
/// How closely [`blocks`](Self::blocks) stands for the program a driver compiles depends on
/// [`functions`](Self::functions). A driver inlines before it charges, so a module of few
/// functions is one whose blocks are the program's; a module that calls a many-block function from
/// many sites holds a program far larger than its own blocks, and the two cases read alike in
/// `blocks` alone. Lowering emits such functions itself: the handle-table and root-table searches
/// are unrolled branch-free steps, emitted once and called at every access.
///
/// The counts compare one module against the next of the same shader, not one shader against
/// another. Measured cost per block differed several-fold between a synthetic module and a real
/// workload, so a per-block figure taken from one shader predicts nothing for the next;
/// [`instructions`](Self::instructions) is the second number for that reason.
///
/// [`blocks`](Self::blocks) alone does not say whose blocks they are. Translation adds blocks of
/// its own (inlining, and the structurizer's layout of control flow as SPIR-V requires), and the
/// input has blocks of its own, and a count that has risen can be either.
/// [`loops`](Self::loops) and [`constructs`](Self::constructs) separate them: the structurizer
/// makes no loop the input does not have, so a `blocks` that rises while `loops` holds still is
/// translation's own work or a changed shape of the same control flow, not new control flow in the
/// shader. The AIR side of that comparison is the caller's own, through `metal2vulkan-air`: for
/// the counts the input itself holds, before inlining, decode it and count the blocks of each
/// function's body (`metal2vulkan_air::decode`, then each `ir::Function`'s `body`); for the ones
/// canonicalization produced, walk the entry
/// [`DeviceProfile::specialized_entry`](crate::device::DeviceProfile::specialized_entry) gives,
/// which is that AIR specialized and canonicalized.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleStats {
    /// Functions: the entry point, and the ones canonicalization and lowering left beside it.
    /// A module of few functions is one whose [`blocks`](Self::blocks) stand for the whole
    /// program a driver compiles.
    pub functions: u32,
    /// Basic blocks, over every function the module holds.
    pub blocks: u32,
    /// Blocks reachable from their function's entry, never above [`blocks`](Self::blocks). It is
    /// below it by the merge blocks SPIR-V requires a construct to declare that no path reaches —
    /// both arms of a conditional inside a loop leaving the construct, say — which the module
    /// terminates with `OpUnreachable`. A bound, not a lever: the difference is blocks the driver
    /// still reads and no invocation runs.
    pub reachable_blocks: u32,
    /// Instructions, phis and terminators. It moved with [`blocks`](Self::blocks) in every
    /// module measured, and is cheaper to reason about.
    pub instructions: u64,
    /// Structured constructs: selection, switch and loop headers.
    pub constructs: u32,
    /// Loops. The structurizer makes none of its own, so this is the input's own loop count.
    pub loops: u32,
    /// Whether the module is inert: it holds no loop, and no instruction of any of its functions
    /// writes memory other than an invocation's own and its workgroup's (`Function`, `Private`
    /// and `Workgroup` storage), writes an image, or emits or drops an output of its stage. So
    /// every invocation ends, and a dispatch of a kernel of it leaves nothing behind but the time
    /// it took. An instruction the classification does not
    /// name as reading or computing alone (an interlock, a ray query) makes it not inert.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inert: bool,
    /// The entry's weight: its instructions with each call counted as its callee's body, as a
    /// driver inlines every call, after the repeated regions are rolled. A stage that weighs more
    /// than [`COMPILED_STAGE_WEIGHT`](crate::pipeline::COMPILED_STAGE_WEIGHT) is one whose
    /// creation may take the host's memory: `metal2vulkan-runtime` compiles it only in a helper
    /// process the consumer caps. 0 for a stage the link makes itself.
    #[cfg_attr(feature = "serde", serde(default))]
    pub entry_weight: u64,
}

/// A precision note a stage records only where a specialized module runs an arm that keeps its
/// near-equivalent: the note holds under a specialization exactly when one of the clauses of `when`
/// holds.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecializedNote {
    pub note: PrecisionNote,
    /// The clauses, sorted and each once; never empty.
    pub when: Vec<SpecClause>,
}

/// A conjunction of [`SpecTest`]s: it holds under a specialization when every test does.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecClause {
    /// The tests, sorted and each once; never empty.
    pub all: Vec<SpecTest>,
}

/// A test of one specialization constant's value `v`: it holds when `(v & mask == bits) == equal`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecTest {
    /// The key: a sampler slot's `SpecKey::SamplerState` or `SpecKey::NearestPixelSampler`, as the
    /// stage's [`SpecializationLayout`] declares it, or a bound value no pipeline's key carries
    /// (`SpecKey::BoundTexelFormat`, `BoundSamplerWord`, `HeapRawTextures`, `HeapNanFamilies`).
    pub key: SpecKey,
    pub mask: u32,
    pub bits: u32,
    /// Whether the test holds where the masked value equals `bits` (else where it differs).
    pub equal: bool,
}

impl SpecTest {
    /// The test that `key`'s value masked by `mask` is `bits`.
    pub fn equal(key: SpecKey, mask: u32, bits: u32) -> SpecTest {
        SpecTest {
            key,
            mask,
            bits,
            equal: true,
        }
    }

    /// The test that `key`'s value masked by `mask` is not `bits`.
    pub fn differ(key: SpecKey, mask: u32, bits: u32) -> SpecTest {
        SpecTest {
            key,
            mask,
            bits,
            equal: false,
        }
    }
}

/// The specialization constants a module declares.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecializationLayout {
    /// In `SpecId` order.
    pub constants: Vec<SpecConstant>,
}

/// One specialization constant: its link-assigned id, meaning, size, and default.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecConstant {
    pub id: u32,
    pub key: SpecKey,
    /// Size in bytes (4 for every key today).
    pub size: u32,
    /// The value the module uses when the consumer does not specialize it, as its low bytes.
    pub default: u64,
    /// The bits of the value the stage never reads and none of its specialized notes tests: the
    /// binder clears them before the value reaches a pipeline's key and the driver, so two values
    /// equal on the other bits make one `VkPipeline`. 0 where every bit may be read.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unread_bits: u64,
}

/// A runtime fact the pipeline depends on and how this profile honors it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeDependency {
    pub fact: RuntimeFact,
    pub mechanism: Mechanism,
}

impl RuntimeDependency {
    /// The dependency on `fact`, met by `mechanism`.
    pub fn new(fact: RuntimeFact, mechanism: Mechanism) -> RuntimeDependency {
        RuntimeDependency { fact, mechanism }
    }
}

/// Answers to Metal's per-pipeline compute queries.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputePipelineMetalInfo {
    /// `maxComputeWorkGroupInvocations`, further limited in the flattened tiers, by emulated
    /// simdgroups' scratch, by the function's own maximum and by the descriptor's (a compute or
    /// tile descriptor's `max_total_threads_per_threadgroup`). A tile pipeline's
    /// (`Device::compile_tile_pipeline`) is at most 1024, the 32x32 tile threadgroup, and is the
    /// largest threadgroup of even sides within those limits, as Metal rounds it. A descriptor's
    /// `required_threads_per_threadgroup` does not change it. Metal's state answers that size's
    /// threads instead, and so does `metal2vulkan-runtime`'s.
    pub max_total_threads_per_threadgroup: u32,
    /// `maxTotalThreadsPerThreadgroup` on Apple's GPUs, before this device's limits: the least
    /// of 1024, the function's own maximum and the descriptor's (a compute pipeline answers a
    /// descriptor's limit as given, on an M3 Max and an M4), and a tile pipeline's
    /// rounded down to the largest threadgroup of even sides within it. A dispatch whose
    /// threadgroup is past it runs nothing, as plain Metal's does; one within it but past
    /// [`max_total_threads_per_threadgroup`](Self::max_total_threads_per_threadgroup) is this
    /// device's gap, refused typed. Apple's answer below 1024 for a kernel whose registers do not
    /// fit is not modeled.
    pub metal_max_total_threads_per_threadgroup: u32,
    /// Always 32.
    pub thread_execution_width: u32,
    /// Bytes of statically sized threadgroup memory: the static part of Metal's own layout
    /// (`KernelFacts::static_threadgroup_bytes`), as Metal reports it: rounded up to a multiple of
    /// 16 bytes. The threadgroup memory of
    /// ours beside it, an emulated simdgroup's scratch and the arena's minimum or a predicated
    /// walk's vote word, is not counted: the planner counts it in each dispatch's total and
    /// refuses a threadgroup whose `[[threadgroup(n)]]` lengths fit beside this length and not
    /// beside it (`ThreadgroupMemorySize`, the device's gap).
    pub static_threadgroup_memory_length: u32,
    /// Echoed from the descriptor.
    pub support_indirect_command_buffers: bool,
}

/// Whether the module's convergence matches Metal's on this profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Reconvergence {
    /// Maximal reconvergence (or no tangled operation depends on it).
    Guaranteed,
    /// Reconvergence after divergence is implementation-defined on this profile.
    ImplementationDefined,
}

/// A documented difference from exact Metal semantics that the output accepts. Validation
/// derives each observation's comparison class from these, never by hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrecisionNote {
    /// Widened half fma rounds twice.
    F16FmaDoubleRounding,
    /// Widened half-integer conversions round twice.
    F16IntConvDoubleRounding,
    /// Bias sampling emulated with gradients may pick another level and another anisotropic
    /// footprint than Metal's: derivatives hoisted out of a per-fragment texture choice, which
    /// scale by `2^B` through every sampler, and a bias whose sum with the sampler's own may lie
    /// beyond the device's `maxSamplerLodBias`, which Vulkan clamps, or that an anisotropic sampler
    /// applies to its footprint's minor axis alone; and, on a device without `maintenance8`, a
    /// fragment's implicit-level sample with a dynamic texel offset, whose derivatives are taken
    /// before the switch over the offset's values (`lower::sampler::OffsetArms`). The gradients
    /// carry Metal's own reading of the bias, `B`; `Exp2`'s precision and the device's level
    /// arithmetic are the device's. Hoisted derivatives of a 2D color sample without a bias, a
    /// `min_lod_clamp` or an offset, through samplers without a bias of their own, go to the
    /// device's sampler whole (`PrecisionNote::DeviceSampling`) and do not record it.
    GradFromBias,
    /// Transcendentals answer within Apple's stated bound, not Metal's bits: the device's own
    /// functions where they are Metal's (MoltenVK), and the device's precision for a call no
    /// other note bounds. Apple states no answer at a NaN operand for a function compiled with
    /// fast math, and a device whose compiler takes its math mode for the whole entry (MoltenVK)
    /// answers a precise function of an entry Metal compiles with fast math by its fast form: a
    /// NaN operand's answer may be a number there (`precise::sin`, `precise::cos` and
    /// `precise::sincos` of a NaN answer `0` on the Apple M3 Max through MoltenVK 1.4.2, where
    /// Metal answers the NaN). Not recorded by precise `fmod`, whose answer is the one right value on
    /// every device.
    DeviceTranscendentals,
    /// Float reductions use the device's association order.
    ReductionOrder,
    /// `float` denormals follow the device: Metal reads an `f32` denormal operand of its float
    /// arithmetic, comparisons, and functions as the zero of its sign and flushes a denormal result
    /// to one, where a Vulkan device keeps or flushes them by its own float controls (this
    /// translation declares no `DenormFlushToZero`). Recorded where the module leaves an
    /// operation's denormals to the device: `+ - *`, the comparisons, `fma`, `dot`, `sqrt`,
    /// `rsqrt`, `trunc`, `rint` and `ldexp` on `float` (by the device's products), `fneg`, `fabs`,
    /// `floor` and `ceil` on `float`, the divisions and the remainder, the device's maths functions
    /// (`fast::fmod`'s fused multiply-add, `sign` and `fract` among them), `mix` by `FMix`, `fast::sqrt`
    /// by `Sqrt`, `dot` of vectors of other than two to four lanes, whose chunks' products the
    /// device sums where Metal flushes each one below `2^-126`, beside their own notes, and a
    /// conversion to or from `double`, whose reading Metal's is not measured. The operations that
    /// read a denormal on its bits answer Metal on every device: `copysign`; and so do the
    /// functions the device answers as Metal's own. Where the minimum and maximum functions
    /// (`llvm.minnum` and `llvm.maxnum` among them, with any flags) and `saturate` are the device's
    /// `NMin`, `NMax` and `NClamp`, their pick of a denormal is
    /// [`PrecisionNote::NativeMinMax`]'s. A nearest pixel-coordinate sample at its texel's centre
    /// `floor(x) + 0.5`, and a pixel fetch's texel `floor(x) + k` under an offset, records it for
    /// that `floor`: Metal's sampler reads a subnormal `x` itself, a negative one as texel -1,
    /// where the device's `floor` reads it by its float controls (MoltenVK's as the zero of its
    /// sign, texel 0, measured on an Apple M3 Max). An address mode that reads texel -1 as texel 0
    /// (clamp to edge, mirror clamp to edge, mirrored repeat) answers alike, so the note holds
    /// only where the centre can be the sample's coordinate and the `s` or `t` axis takes another
    /// mode (repeat, clamp to zero, clamp to the border): through a constexpr sampler where its
    /// modes are such, through a runtime sampler slot held where the bound state is a nearest
    /// pixel sampler (`SpecKey::NearestPixelSampler`; for an integer texture, where it states
    /// pixel coordinates) and its word states such a mode (`SpecKey::BoundSamplerWord`, which the
    /// runtime reads at each draw or dispatch), never under a normalized state; through an
    /// argument buffer's sampler [`PrecisionNote::ArgumentPixelSamplerScaling`] records it
    /// instead. A pixel fetch under an offset records it outright. Validation derives no tolerance and no explanation from it: every
    /// `float` computation is exposed to it.
    DeviceDenormals,
    /// A pixel-coordinate sampler's coordinate reaches the device's normalized sampler divided by
    /// the texture's level-0 size in the shader, and the device's sampler reads its texels from
    /// that coordinate in its own arithmetic: the position it reads, `(x / N) N` in the device's
    /// rounding and its sub-texel precision, lies within a few units in the last place of the
    /// coordinate `x` as given, where Metal's sampler reads `x` itself on its own sub-texel grid. So
    /// a linear sample or a gather reads Metal's texels and weights but where `x` lies that close
    /// to the edge of a texel pair, a gather's 2x2 block or one of the sub-texel steps of the
    /// filter's weight, where it may read the neighbouring pair, block or step. Recorded on every
    /// device where no other form takes the coordinate: a linear sample or a gather through a
    /// constexpr pixel sampler of an array or a 3D texture (Vulkan's unnormalized sampler takes
    /// neither) or of a texture whose residency the function observes (whose footprint is read
    /// from the normalized coordinate), a level query, and every linear sample, gather and
    /// comparison through a runtime sampler slot (whose state the function does not know), held
    /// where the bound sampler states pixel coordinates and, for a sample or comparison, that it
    /// is not a nearest one of one isotropic level (`SpecKey::NearestPixelSampler`). Not
    /// recorded where Metal's texel or footprint
    /// follows from `x` itself: a constexpr pixel sampler's sample or gather of a 1D or 2D texture
    /// that is no array hands `x` (moved by its offset) to the device's unnormalized sampler; a
    /// nearest sample elsewhere hands the device its texel's centre `floor(x) + 0.5` over the size,
    /// where the device reads exactly that texel (through a runtime slot behind the slot's
    /// `NearestPixelSampler` specialization); and a nearest constexpr sampler clamped to the edge
    /// fetches its texel (`sampler.pixel_fetch`). Measured: the device answers as its normalized
    /// sampler at a coordinate within one `f32` ulp of the quotient `x / N` on MoltenVK, and at the
    /// quotient itself on lavapipe (800 coordinates at and around each texel edge of a 5x3
    /// texture, every filter and address mode); on MoltenVK a linear sample through a runtime or
    /// argument buffer's sampler or of an array reads another weight step than Metal's at under 1%
    /// of the coordinates at and around each texel edge of textures 258, 21 and 37 texels wide
    /// (where `x - 0.5` lies at a half step of Metal's `1/256` grid); a linear downscale by 8 of
    /// textures 1000 to 1920 texels wide is within 1/255 of Metal's on MoltenVK and 2/255 on
    /// lavapipe (a weight one 1/256 step off on each axis); other devices unmeasured. A cube's
    /// pixel sampler is a normalized one. An argument buffer's sampler records
    /// [`PrecisionNote::ArgumentPixelSamplerScaling`] instead.
    ///
    /// Validation derives no tolerance from it: at an edge the value is the neighbouring pair's,
    /// block's or step's, whatever it holds.
    PixelSamplerScaling,
    /// Float division, remainder, square root, fused multiply-add, and dot products use Vulkan's
    /// precision bounds (e.g. 2.5 ULP division, a possibly unfused `Fma`) rather than correct
    /// rounding. A fast (`arcp`) division is the device's `OpFDiv`, where Metal's is `x` times a
    /// reciprocal it rounds within about 0.64 ulp but not correctly. Not recorded by precise
    /// `fmod`, whose answer is the one right value on every device.
    DeviceArithmetic,
    /// Float operations LLVM requires to keep signed zeros, infinities, and NaNs run without
    /// `SignedZeroInfNanPreserve`, which the device does not support at their width. Without the
    /// mode the device may answer such a zero with either sign.
    DeviceSignedZeroInfNan,
    /// A ray intersection traversed by the device's ray query. The hit distance and the
    /// barycentrics are the device's arithmetic. The conventions no Metal reading has settled yet
    /// are the translation's: forced opacity decides before the opacity cull, a miss or a skipped
    /// traversal reads zero in every field, an invalid ray (including a negative `min_distance`)
    /// skips the traversal, a bounding-box candidate stays uncommitted without an intersection
    /// function, and facing is decided in object space by the winding. Through an intersection
    /// function table: a nil (unbound) table accepts every triangle candidate, an entry at or past
    /// the table's size is ignored like an entry never set (the candidate stays uncommitted), and
    /// the user instance id of an instance built from a default descriptor, which has no `userID`,
    /// is 0 because the consumer passes 0, a choice: what Metal
    /// reads there is unmeasured.
    DeviceRayIntersection,
    /// Float arithmetic whose AIR lets a compiler rewrite it with a neighbouring operation: an
    /// addition or subtraction and the multiply feeding it both carry LLVM's `contract`, two
    /// chained additions, subtractions, or multiplies both carry `reassoc`, or an `llvm.fmuladd`
    /// (fused or not, as the implementation chooses). The module states each operation in the
    /// AIR's order with its own flags: one carrying both `contract` and `reassoc` is left to the
    /// device's compiler, which groups and fuses it as it chooses, and any other is kept apart
    /// (`NoContraction`). The AIR's value is any of the rewrites those flags allow, and Metal's
    /// compiler takes its own grouping and fusion, so the roundings may differ: by any amount
    /// relative to the result where a sum of mixed signs is grouped otherwise, or a multiply-add
    /// fused on one side alone. On a device without `shaderFloat16` a chain of such `half` additions, subtractions and
    /// products is held at `float` precision between the roundings where its values leave it
    /// (`lower::arith::half_chain`): the result may differ from the rounded operations' by the
    /// roundings skipped, and a value inside the chain past `half`'s range stays finite. On
    /// MoltenVK, a `half` addition, subtraction or product carrying both flags is the device's
    /// 16-bit instruction left undecorated, whose conversions Metal's fast compile folds away: it
    /// answers the `float` result of the widened operands, unrounded, with the same bound.
    /// Recorded on every device.
    FastMathRewrites,
    /// A tessellation pipeline's vertex order: Metal's clockwise output winding is declared
    /// `VertexOrderCcw` and its counter-clockwise one `VertexOrderCw`, as MoltenVK maps them under
    /// Vulkan's default upper-left domain origin. Measured for both windings against Metal's bytes
    /// (recorded on an Apple M3 Max) by a consumer's device test,
    /// `tessellation_states_tessellate_as_metals`, on RADV (Radeon 780M, RX 7900 XTX), NVIDIA
    /// Turing and MoltenVK; other drivers are not measured, where the other mapping would reverse
    /// the facing of every generated triangle.
    TessellationWinding,
    /// A tessellation pipeline's factors at the edges of Metal's stated rules: a subnormal edge
    /// factor is positive and does not cull (its level is the range's lower end), where a flush to
    /// zero would cull the patch; and a NaN inside factor's level is the range's lower end. Metal
    /// answers so under the `integer` partition (2^-24 edges, a NaN inside); under the others it is
    /// not read.
    TessellationFactorEdgeCases,
    /// A tessellation pipeline with `tessellationFactorScaleEnabled` multiplies each factor by the
    /// encoder's scale rounded to `half`, exactly in `float`, after the cull. Metal rounds the
    /// scale so and does not round the product to nearest (factor `0x47F3` by `0x3007`, 1.000445,
    /// drew two `integer` segments, and a product below `half`'s least subnormal does not cull);
    /// whether the fractional partitions' levels take the product in `float` or rounded up to
    /// `half` is not read. A scaled `integer` level is rounded up to an integer by the control
    /// stage, exact on every device; a scaled fractional level is the product, which a driver that
    /// hands the levels to Metal's `half` factors (MoltenVK) rounds to the nearest `half`.
    TessellationFactorScale,
    /// A tessellation pipeline whose factor step function is per instance reads instance `i`'s
    /// factors at `i · instanceStride`, `i` counting the draw's base instance as `[[instance_id]]`
    /// does. Whether Metal's step counts from the base instance is not read; a base instance of 0
    /// is exact.
    TessellationFactorInstance,
    /// A tessellation pipeline of a fractional partition (`fractionalOdd`, `fractionalEven`) drawn
    /// through the device's tessellation stages: Vulkan leaves where a fractional spacing places
    /// its vertices to the implementation (its two shorter segments' length need only fall
    /// monotonically with the rounded level less the level, and their place on the edge is the
    /// implementation's, "Tessellator Spacing"), where Metal's tessellator places them by its own
    /// rule. Measured equal to Metal's, at factors within the device's range, by a consumer's
    /// device tests on RADV (Radeon 780M, RX 7900 XTX), NVIDIA Turing, ANV (Intel Iris Xe) and
    /// MoltenVK; other drivers, lavapipe among them, and factors past the range are not measured.
    /// Recorded on the evaluation stage.
    ///
    /// Validation derives no tolerance from it.
    TessellationFractionalSpacing,
    /// An 8-bit float converts to an integer from the format's documented value (MSL 4.1's fp8
    /// format table for `f8e4m3fn` and `f8e5m2`; LLVM's `Float8E4M3` for `f8e4m3`) with Metal's
    /// saturating float-to-integer rule, where Metal's device here answers otherwise: Metal on
    /// Apple M3 compiles these reads but extends the carrier byte as an integer, the same for every
    /// format, so the result is unverified on Metal. The note marks only these reads. The widening
    /// to a float carries none: Metal on Apple M3 does not compile it, so no Metal answer
    /// contradicts it (and none verifies it either).
    Fp8Value,
    /// A fragment function of a render pipeline that rasterizes more than one sample reads a value
    /// Metal reads at the pixel center in every invocation and Vulkan evaluates at an
    /// implementation-defined position within the fragment: `[[position]]`'s z or w (the
    /// interpolated depth and `1/w`, whose position Vulkan does not state), a center-sampled
    /// varying on a device without its pixel form (interpolation functions whose
    /// `InterpolateAtOffset` reads from the pixel center: "an implementation-defined position
    /// within the fragment area being shaded"), a centroid varying or an `interpolant` read through
    /// `interpolate_at_centroid` (anywhere in the fragment's covered area, where Metal, measured on
    /// Apple M3 Max, moves it to the location of the lowest covered sample only under partial
    /// coverage), an `interpolant` read through `interpolate_at_center` as a center varying,
    /// `[[point_coord]]` ("the fragment center, or the location of a
    /// sample"), or `[[barycentric_coord]]` (its builtin; the varying form without
    /// `fragmentShaderBarycentric` is read at the pixel center, as a center varying). A function
    /// that runs per sample under sample shading reads all but the center varying (which link
    /// refuses there without its pixel form) where Vulkan evaluates them for its sample, where
    /// Metal reads every one as the pixel's in each sample's run (Apple M3 Max at 4x, measured).
    /// Bound: each is evaluated at a point of the pixel (the center or a sample's location), so it
    /// differs from Metal's by at most its change over half the pixel's diagonal, and a centroid
    /// varying by at most its change across the pixel's covered area. Exact where the device
    /// evaluates a read where Metal does: measured per sample at 4x, lavapipe evaluates
    /// `[[position]]`'s z at the sample and the others at the pixel, and ANV (Intel UHD 770)
    /// `[[point_coord]]` and a centroid varying at the sample and `[[position]]`'s z and w at the
    /// pixel, where they equal Metal's to the last bit or two.
    MultisamplePosition,
    /// A filter the module emulates by fetches sums its taps in the device's arithmetic, where
    /// Metal's sampler has its own datapath: Metal's bicubic filter (every format), and an
    /// extended-range texture's linear, 3D and linear-mip-filter samples (`BGR10_XR`,
    /// `BGRA10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins, the padded forms; read through the
    /// slot's `extended_range` variant or an argument buffer's raw element). The footprint, the
    /// fraction (`1/256` of a texel), the addressing of each tap, the filter each format selects
    /// (a 32-bit float format's bicubic sample and a level under four texels bilinear), the level
    /// a mip filter reads, and each format's clamp are Metal's; the arithmetic is the device's:
    /// the Catmull-Rom weights computed in `f32` from the fraction and the taps summed by its
    /// multiply and add, a bilinear, trilinear or level blend by its `FMix` at the `f32`
    /// fractions, and a `half` result the device's narrowing of the `f32` value. Metal's own
    /// weights are a fixed-point table (each within `4.83e-4` of the polynomial's per axis,
    /// together), it sums an 8- or 16-bit normalized or extended-range format's codes exactly onto
    /// the format's grid with half-up rounding (a linear sum's terms truncated first), it sums the
    /// float formats through its own accumulators and rounds a half format's sum to `half` (away
    /// from zero at the six ties the conformance case `bicubic-texel-classes` holds), and it
    /// truncates a linear mip filter's level fraction onto sixty-fourths: a lane may differ by a
    /// few grid steps (`1/4080` at 8 bits), by the weights' distance times the taps (`2.1e-3` at
    /// most on an extended-range `_sRGB` texture's taps), and by up to `1/64` of the two levels'
    /// difference under a linear mip filter. A tap of an 8-bit sRGB texture held as its UNORM twin
    /// is the device's decode of its code ([`PrecisionNote::SrgbTexelDecode`]). Through a runtime
    /// sampler slot the note is specialized on the word of the sampler bound there
    /// (`SpecKey::BoundSamplerWord`, which the runtime reads at each draw or dispatch): it holds
    /// where that word filters (a linear minification or magnification filter, anisotropy, a cubic
    /// filter or a linear mip filter), and a nearest word's sample, the decoded texel at Metal's
    /// position, holds none. Through an argument buffer it holds where a texture the stage's
    /// handles can name is held by its raw texels in a form whose sample the function filters
    /// (`SpecKey::HeapRawTextures`, which a plan gives): a command whose heap holds no such
    /// texture samples through the device's own sampler, or reads a bfloat form's nearest texel.
    ///
    /// Inferences, taken by family: the signed clamp's lower bound, the treatment of the formats
    /// no case reads (`RG8UnormSRGB`, `RG8Unorm`, `RG8Snorm`, `RG16Unorm`, `BGRA8Unorm`,
    /// `A8Unorm`, `Depth32Float`). The footprint is exact for every coordinate within
    /// `[-2^23 + 1/2, 2^23]` texels, where its 1/256 step fits an `i32`. Past that range a
    /// coordinate below `2^120` texels saturates it, the translation's choice. A non-finite or
    /// larger coordinate gives an unspecified footprint whose taps stay in bounds. Metal's value
    /// is not measured for either.
    ///
    /// The note's class is unbounded: a lane that differs for any of these reasons is reported
    /// as explained, never accepted as a match.
    BicubicFilterArithmetic,
    /// A stage samples or gathers a texture whose residency it observes
    /// (`sparse_color::resident()`) through the slot's sparse form, which the module has where the
    /// profile enables `shaderResourceResidency`. Metal's status word is measured: every access of
    /// a texture that is not sparse is resident, inside it or outside; a read outside any texture
    /// is resident; and an access of a sparse texture is not resident where a texel it counts is
    /// unmapped, which reads zero memory (the out-of-bounds texel: zero, alpha 1 where the format
    /// has none). The module answers all of that exactly for reads, on every device. A pixel
    /// fetch reads Metal's value too: where the bound texture is a sparse image, a texel the word
    /// says is unmapped reads zero, and the lanes Metal reads as constants (a channel the format
    /// lacks, through the texture's swizzle) are set after every access (`SpecKey::ConstantLanes`);
    /// none of them is noted. A sample's or gather's residency is the device's, noted: the word is
    /// the device's `OpImageSparse*` residency, whose footprint may count a texel of zero weight
    /// Metal does not (Vulkan's linear 2x2 footprint does, measured on lavapipe), and may call a
    /// footprint of the border alone (a `clamp_to_border` access wholly outside the texture)
    /// not resident where Metal answers resident (lavapipe does, by nearest, linear and gather
    /// sampling, beside a mapped edge too; Vulkan states only whether a texel accessed is a sparse
    /// unbound texel), and its lanes
    /// that are no constants over a texel in a block of a sparse image that holds no mapped tile
    /// are the device's non-resident value (zero on a device with `residencyNonResidentStrict`;
    /// any value elsewhere, which the runtime announces as `non-strict-residency`). Where the
    /// profile does not enable `shaderResourceResidency` the device has no residency of its own and
    /// nothing is noted: the bound image holds Metal's unmapped value, and the slot's tile map
    /// answers the word, by Metal's measured footprint rule for a sample or gather through a
    /// constexpr sampler (`lower::sampler::footprint`), exactly.
    SparseResidency,
    /// A `float` NaN result's bit pattern follows the device: Metal answers the canonical
    /// `0x7fc00000` for the NaN results of its float arithmetic and functions, and Vulkan states no
    /// encoding, so a device may answer another sign or payload (llvmpipe: `0xffc00000`). Recorded
    /// wherever the module leaves a `float` operation that can produce a NaN to the device; not by
    /// Metal's precise and fast `float` divisions. Also recorded by every read, sample, gather, and
    /// pixel fetch of a `half` texel: Metal hands back the texel's own NaN, sign and payload, and
    /// Vulkan states no encoding for a 16-bit float texel's decode (MoltenVK answers the canonical
    /// NaN for every one); on a device that rounds a `half` through its conversion under the float
    /// controls (ANV, RADV), the NaN that conversion makes of the device's decode. Also recorded
    /// by a `half` texel's write to a storage texture that may hold a float format, a slot whose
    /// storage choice declares none or a heap element: Metal stores a 16-bit float texel's NaN as
    /// the half's own bits and a 32-bit or packed small float one's as its canonical NaN, and the
    /// device converts the half widened. Also recorded by a `float` write to a storage texture that
    /// may hold a half-float or packed small float format, and by a `float` color output written as
    /// computed to such an attachment, where the device is not measured to store Metal's NaN there:
    /// Metal stores every NaN as its canonical one, and RADV and NVIDIA keep a NaN's payload bits
    /// (lavapipe stores `0xffc00000` as `0xfe00`, ANV every NaN). Where the write's texture slot is
    /// declared without a format, the note is recorded under the format bound there (a
    /// [`SpecializedNote`] of `SpecKey::BoundTexelFormat`), which the runtime announces at a draw
    /// or dispatch that binds such a texture; a `half` write to a heap element, under the families
    /// of the textures the stage's handles can name (`SpecKey::HeapNanFamilies`), which the runtime
    /// announces at a draw or dispatch whose command can name one of a family whose stored NaN is
    /// the device's on that profile.
    /// Also recorded on a render pipeline's vertex stage where its vertex descriptor gives a float
    /// format narrower than 32 bits (a `half` format, `FloatRG11B10`) to an attribute that
    /// fixed-function vertex input fetches and the function reads as `float`
    /// (`pipeline::link::vertex`): Metal widens its NaN to `0x7fc00000`, and the device's
    /// widening need not. A pulled attribute's decode records it in the module (a vertex
    /// function's, a kernel's `[[stage_in]]` attribute's, a post-tessellation function's). Also
    /// recorded by a `float` read of a heap's texture buffer on a device without
    /// Metal's sampler: Metal hands back a half-float texel's NaN as `0x7fc00000`, and the handle
    /// carries no class to convert it by. The runtime announces it too for a GPU blit through an
    /// `RG11B10Float` texture Metal keeps losslessly compressed: Metal's compression makes a NaN
    /// field the quiet NaN (`0x7e0`, `0x3f0`), and the device's copy keeps its bits. And for a
    /// plain color fill to a NaN of a half or small-float texture the device's clear writes: the
    /// device converts the NaN as it likes, where Metal's fill keeps a half NaN's sign and payload
    /// and makes a small float's all ones. Not recorded by a texture write of a constant texel none
    /// of whose lanes is a NaN. Validation derives no tolerance and no explanation from
    /// it, as for `DeviceDenormals`.
    DeviceNanEncoding,
    /// The device compares a depth texture's texels (a sample or a gather with comparison), on
    /// every device (narrowed to comparisons). Metal's `Depth32Float` texels hold any `float`,
    /// and the device's comparison reads them from the texture's depth image, which Vulkan gives
    /// no value outside `[0, 1]` without `VK_EXT_depth_range_unrestricted`, which no runtime
    /// enables: a copy may take none (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), so a consumer
    /// clamps its uploads, and a render write is clamped to the viewport's depth range. A
    /// comparison whose texel Metal holds outside `[0, 1]` can then differ, whatever the driver is
    /// measured to keep (MoltenVK). Values within `[0, 1]` are exact. A function that reads, samples or gathers a depth
    /// texture without comparing records nothing: its interface's `TextureSlot::access` states
    /// the use, and a consumer that uploads a value into a depth image its device clamps announces
    /// that upload (docs/API.md section 5).
    RestrictedDepthRange,
    /// An explicit-gradient sample of a cube texture takes Metal's footprint in a form a Vulkan
    /// device reads alike, but for what is not reproduced. Metal (measured on the Apple M3 Max
    /// under macOS 27.0.1) drops each gradient's component on the direction's major axis `a` and
    /// reads the face's two tangent components over `2 |m_a|`, as a 2D sample of the face reads
    /// its gradients. Through an isotropic sampler the module passes each gradient's two tangent
    /// components combined by Metal's length rule (`max(M + 3m/16, 49M/64 + 21m/32)`) on one
    /// tangent, from which Vulkan's cube derivative rule reads that length exactly; through an
    /// anisotropic sampler, the face gradients without their normal component, mirrored on the
    /// `-x`, `-y` and `+z` faces as Metal mirrors them. Not reproduced: the level is the device's
    /// logarithm of the footprint, where Metal's piecewise-linear one minifies only past a
    /// footprint of `1 + 3/2048` and blends levels in 256ths, and the device filters an
    /// anisotropic footprint by its own rule (its taps and weights). The major axis is Metal's,
    /// `x` over `y` over `z` at equal magnitudes (measured on Apple M4), its ties broken in the
    /// direction handed to the device. A level near a filter or level boundary can therefore
    /// select another level or filter than Metal's. A runtime or argument buffer's sampler takes
    /// the form its word's anisotropy states.
    ///
    /// Not recorded where the module takes Metal's footprint itself through an anisotropic
    /// constexpr sampler on a device without `samplerAnisotropy` (`lower::sampler::anisotropic`):
    /// Metal's level and taps (`PrecisionNote::AnisotropicFilter`).
    ///
    /// Also recorded for every cube access whose level comes from the fragment's derivatives (an
    /// implicit-level sample, one whose gradients a switch-at-use hoisted, or an LOD query): its
    /// level is the device's, of the direction handed to it, and Metal's rule there is not
    /// measured. A tie so broken scales a component by at most `1 - 2^-23`, which moves a face
    /// coordinate by at most `2^-24` of the face, a derivative the device takes between two lanes
    /// by at most `2^-23` (`2^-9` texels on Metal's largest face, 16,384, so the level by at most
    /// `log2(1 + 2^-9)`, about 0.003, where the footprint is at least a texel), and a hoisted
    /// gradient's projection by at most `2^-24` of the major component's relative change: within
    /// Vulkan's own level-of-detail approximation, but able to cross a level or filter boundary.
    CubeGradientFootprint,
    /// The module reads, gathers, or samples a texture with a `half` result, which is the device's
    /// `f32` texel rounded to nearest even (`narrow::round_texel`) on every device, where Metal
    /// converts the texel to `half` by the bound format. A read, nearest sample, or gather of a
    /// slot bound in an 8-bit sRGB format the device holds as its UNORM twin decodes the code
    /// exactly instead, and a bicubic sample sums the codes' steps exactly, under the slot's
    /// `SrgbTexture` variant, and records nothing. What a filtered sample converts is the filter's
    /// value ([`PrecisionNote::HalfSampleFilter`]). Validation derives no tolerance from it.
    ///
    /// Native device sampling (a texel conversion): every access with a `half` result records this
    /// note, a heap element's and a depth texture's included. The recorded bound: the device's
    /// `float` texel (within [`PrecisionNote::DeviceTexelConversion`]'s bound of Metal's) rounded
    /// to nearest even is within one `half` ulp of Metal's `half` result (Metal rounds a 32-bit
    /// float sample toward zero, a half-float linear sum with ties away from zero, and a normalized
    /// value onto its grid first), but for a 32-bit float texel's sample past `±65504`, which Metal
    /// saturates to `±65504` and the rounding takes to the infinity of its sign; a NaN is the
    /// device's decode ([`PrecisionNote::DeviceNanEncoding`]).
    HalfTexelConversion,
    /// The module samples a texture with a `half` result through a filter (linear minification
    /// or magnification, a linear mip filter, anisotropy, or a runtime sampler) by Vulkan's Y′CbCr
    /// model of the codes of a two-plane YCbCr texture whose planes the function reads in place
    /// (`lower::sampler::in_place`), which blends the codes, not Metal's exact filter of the
    /// format, on every device: recorded under the slot's bound converted 4:2:0 format and,
    /// through a runtime sampler slot, the bound word's linear filter that addresses no border.
    /// The `float` result's counterpart is [`PrecisionNote::FloatSampleFilter`]. Validation derives
    /// no tolerance from it: the class is unbounded.
    HalfSampleFilter,
    /// The module reads, samples, or gathers a texture with a `half` or `float` result through
    /// the device's sRGB decode, or decodes the device's filter of its codes, where the texture may
    /// be bound in an 8-bit sRGB format (`metal::texel_format::srgb_coded`: the 8-bit sRGB formats
    /// and the sRGB forms of BC1 to BC3, BC7, PVRTC, ETC2, EAC and ASTC). Metal decodes such a
    /// texel's color channels through its twelve-bit table (`k / 4095` for the code's step `k`)
    /// and its alpha as `c / 255`, and its filters sum the decoded steps. Wherever the device
    /// samples the format, its decode is the only form: the slot's view is an sRGB view of the
    /// image (an `R8G8B8A8_SRGB` view of a decoded twin's codes where the device lacks a
    /// compressed format), and a read, sample, gather, bicubic tap or input-attachment read takes
    /// the device's value, Vulkan's sRGB EOTF applied before filtering (the specification sets no
    /// precision for it). Recorded under the bound format's class (`BicubicTexels::SRGB`, a test
    /// of `SpecKey::BoundTexelFormat` the runtime reads at a draw or dispatch binding such a
    /// texture), outright for an argument-buffer (heap) element, whose format is never known,
    /// and at the pipeline for a `float` input attachment whose color attachment is an sRGB format
    /// (`StageIo::decoded_fetches`). Measured over every code of `RGBA8Unorm_sRGB`:
    /// MoltenVK on an Apple M3 Max decodes each as Metal does, bit
    /// for bit; lavapipe (LLVM 21.1.8, arm64) each color code within 5.25 table steps (0.00128, a
    /// third of a code's `1/255`) of Metal's value, alpha within one `f32` ulp; other devices are
    /// not measured.
    ///
    /// Where the device cannot sample the bound format, whose texture it holds as the format's
    /// UNORM twin (`metal::texel_format::srgb_twin_on`: `R8Unorm_sRGB` and `RG8Unorm_sRGB` on
    /// lavapipe, `RG8Unorm_sRGB` on ANV and NVIDIA among the committed profiles), the slot's
    /// `SrgbTexture` variant decodes what the twin's view reads by the MSL transfer through the
    /// device's division and `Pow` (`pack::srgb_decode_texel`): a read, a sample, a fetch, a gather
    /// and a bicubic tap one texel's codes, a filtered sample (a linear filter, a linear mip
    /// filter, anisotropy, or a runtime sampler) the device's filter of the codes, the decode of
    /// the codes' blend where Metal blends the decoded texels. So does a read of an 8-bit sRGB
    /// storage texture held through its UNORM alias (`StorageImageClass::SrgbEncode`, Vulkan having
    /// no sRGB storage view), at a slot or held by an argument buffer
    /// (`lower::texture::heap_storage_conversions`). Both are recorded outright, on every device:
    /// Metal's table rounds the transfer to a twelve-bit step, up to half a step (`1 / 8190`) from
    /// it, where MoltenVK's `Pow` read within `1.83e-7` of the transfer through the alias.
    /// Validation derives no tolerance from it: the class is unbounded.
    SrgbTexelDecode,
    /// The module samples a texture with a `float` result through a filter by Vulkan's Y′CbCr
    /// model of the codes of a two-plane YCbCr texture whose planes the function reads in place
    /// (`lower::sampler::in_place`), as [`PrecisionNote::HalfSampleFilter`] states for a `half`
    /// result: a sample is filtered unless its sampler has nearest minification and
    /// magnification, no mip filter or a nearest one, and no anisotropy. Validation derives no
    /// tolerance from it: the class is unbounded.
    FloatSampleFilter,
    /// A vertex or fragment function keeps more private memory (its `alloca`s, a call stack among
    /// them, or constants held as `Private` variables) than the floor, 16 KiB a thread, on a device
    /// whose Function storage was not measured, up to the 192 KiB a thread Metal was measured to
    /// hold (`device::function_storage`). A driver backs that storage with scratch memory, and
    /// Vulkan states no amount it holds: past what it holds a driver can answer wrongly or hang the
    /// GPU with no error. Probed, the RTX 5080 Laptop GPU hung at 48 KiB and the RTX 3090 at 192
    /// KiB (Xid 109), and the Iris Xe answered wrongly at 48 KiB and the Radeon 780M at 128 KiB.
    /// The module runs on this device all the same, resting on no measurement; a kernel, an object
    /// or a mesh function past the floor there is refused instead. Validation keeps a difference as
    /// a mismatch: the class is derived as if the note were absent.
    FunctionStorageAssumed,
    /// A fragment function runs a non-quad simdgroup operation (a reduction, scan, ballot,
    /// broadcast, shuffle, vote, or simdgroup matrix) before a discard on a path to it, in a
    /// shape where Metal's placement of it is not known. Apple M3 Max's and M4's compiler moves a
    /// simdgroup sum past an `if (c) discard_fragment();` when nothing between them reads it or
    /// has a side effect, so a quad whose every thread the discard drops is not among its lanes,
    /// and the module moves it there too; where the discard's condition reads the sum, or a store
    /// or an atomic lies between, Metal counts the quad as the module does. Every other shape keeps
    /// the operation ahead of the discard, counting that quad, which Metal's may leave out: another
    /// operation than a sum, a use of its result between, a nested discard (measured placed both
    /// ways). After the discard the module counts Metal's lanes: a discarded thread is a lane while
    /// its quad keeps a live thread, and a quad of discarded threads and helpers alone leaves.
    /// Validation keeps the row open (C3): the moved lanes move whatever later words read the
    /// operation's result.
    QuadsCountedBeforeDiscard,
    /// The module places a texel offset on an axis that may repeat or mirror as Metal does in an
    /// access of an argument buffer's texture: in an emulated raw sample (`heap_raw`) or a bicubic
    /// filter's footprint, on every device. Metal drops the offset on a repeating
    /// or mirrored axis whose level size lies in `[2, 7]` and whose texture's base-level size is
    /// not a power of two, reading the base-level size of the texture a view was made from; a heap
    /// element's access can read only the bound view's. So on a view whose first level lies above
    /// its parent's base level, an axis of view size 2 or 4 under a parent whose base-level size is
    /// not a power of two keeps an offset Metal drops, and the access reads the texel or footprint
    /// one offset away. Any other texture or view places it as Metal does, and a texture slot's
    /// access reads the parent's through the slot's `SpecKey::OffsetBaseAxes`, recording nothing.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a mismatch
    /// for review (no authored case binds such a view).
    ViewBaseTexelOffset,
    /// A fragment function writes a float to a color channel whose conversion the device makes
    /// where it is not measured to make Metal's: a normalized, sRGB or small float channel the
    /// attachment stores as written, an attachment that blends with factors other than one and zero (`ChannelWrite::Device`), the alpha of
    /// attachment 0 under alpha-to-coverage, a format whose conversion Metal was not measured with
    /// (the packed 16-bit unorm formats, `A8Unorm`, the packed floats). (A `bfloat` output is
    /// refused, as Metal's compiler rejects one.) Near a tie between two codes the device may store
    /// the other one. Metal rounds a normalized channel's exact product with its maximum code to
    /// nearest, its one exact tie (0.5 in an 8-bit channel, 127.5) up, encodes sRGB through its
    /// twelve-bit table and rounds a value written to a half-float channel toward zero. ANV, RADV
    /// and lavapipe round the `f32` product with ties to even and encode sRGB by their own
    /// approximations. NVIDIA's 615.71.09 on an RTX 5080 Laptop GPU stores an 8-bit unorm channel's
    /// lower code for an input on a tie or up to two `f32` steps above one (0.5 as `0x7f`, the
    /// `f32` nearest 1.5/255 as 1 and the one nearest 2.5/255 as 2, where Metal stores `0x80`, 2
    /// and 3), and Metal's code elsewhere: measured against an Apple M3 Max under macOS 27.0.1 over
    /// every code's exact value and every tie with two `f32` steps either side, in `BGRA8Unorm`
    /// and `RGBA8Unorm`, a draw of an indirect command buffer's as a plain draw. A device may
    /// round a small float channel otherwise. A NaN stores as its conversion makes it (Metal stores a normalized
    /// channel's as 0, a blended snorm channel's as -1.0's code). A 32-bit float attachment that so blends
    /// stores the device's blend arithmetic (NVIDIA's 615.71.09 one `f32` ulp from Metal's in some
    /// source-over sums, a NaN as `0x7fffffff` where Metal stores `0x7fc00000`).
    ///
    /// The recorded bound past one code, measured on lavapipe (LLVM 21.1.8) against the conformance
    /// suite's recordings of plain Metal on an Apple M3 Max under macOS 27.0.1: a blend into an
    /// 8-bit unorm attachment (`RGBA8Unorm`, `ARGB8Unorm`) stores within 2 codes of Metal's; a
    /// fragment function that fetches an 8-bit sRGB attachment and writes it again, within 2 codes
    /// once a later draw reads an earlier one's code (the device's decode,
    /// [`PrecisionNote::SrgbTexelDecode`], beside its encode); a blend into an 8-bit sRGB
    /// attachment, within 4 codes. On lavapipe (Mesa 26.2.4, LLVM 23.1.1), against the probe
    /// `msaa_word_formats.m` on that M3 Max: a blend into a 4-sample `A1BGR5Unorm` attachment the
    /// device renders itself, within 2 codes of Metal's, over a sample whose fill it stored a code
    /// from Metal's. On RADV (Mesa 26.2.6 on the Radeon 780M), against the same probe: a blend
    /// into an `RGB9E5Float` attachment the device renders itself takes the source's channels and
    /// alpha as halves rounded toward zero, a value past 65504 as 65504, where Metal blends the
    /// `float`s (Vulkan asks a blend for no more precision and range than the attachment's): at 2
    /// and 4 samples, 30 of 96 blended samples not Metal's, up to 33 units of the larger value's
    /// last mantissa bit apart where the source passes 65504 and 7 where the half's alpha weighs a
    /// destination far larger than the source. Another device is not bounded past one code: an
    /// NVIDIA RTX 5080 Laptop GPU's blend into an 8-bit sRGB attachment stored up to 11 codes from
    /// Metal's in the same suite.
    ///
    /// Also: a function writes a `float` (or `half`) texel to a storage texture that may hold a
    /// normalized or half-float format, or `ARGB8Unorm`, which the device converts: Metal stores
    /// each normalized channel's exact product rounded with a tie away from zero (a 1-bit channel 1
    /// only above one half) and a half rounded toward zero. An integer texel a function writes to a
    /// storage texture of the float class that may hold a half-float or small-float format
    /// (`RG11B10Float`), its lanes' bits as the `float` written, records it on every device:
    /// Metal's integer store rounds such a channel to nearest even, the device's store as it rounds
    /// a `float` write (toward zero on every measured driver: within one unit in the last place of
    /// the channel's value, a finite value past 65504 stored as 65504 where Metal's `half` is the
    /// infinity).
    ///
    /// Also: a blend into a packed 16-bit unorm attachment (`B5G6R5Unorm`, `A1BGR5Unorm`,
    /// `ABGR4Unorm`, `BGR5A1Unorm`) reads what the device's attachment holds, its channels' codes,
    /// where Metal's tile memory holds the pass's clear and each earlier blend's result finer than
    /// those codes until the store (measured on an Apple M3 Max under macOS 27.0.1: in a pass that
    /// clears and blends, and over two blends in one pass, every channel where the two readings
    /// store different codes stored the finer one's, `half` precision fitting all of them,
    /// single-sampled and 4x alike; a pass that loads what an earlier pass stored blends against
    /// the codes). Such a blend may store a code one apart from Metal's. So does the blend the
    /// function computes into such an attachment the device renders through its word view
    /// (`metal::format::WordAttachment`), which under fast math also sums a few states otherwise
    /// than Metal (3,527 of 94,732,288 channels measured a code apart, none further). `RGBA8Unorm`
    /// and `RGB10A2Unorm` hold their codes there, as the device does.
    ///
    /// Also (the runtime announces it): a render pass clears a normalized, sRGB, packed 16-bit,
    /// half or small-float color attachment to a color whose conversion is in question. The device
    /// converts the clear color, which Vulkan lets store either of the two codes closest to a
    /// normalized channel's scaled value (asking for the nearest), leaving an sRGB encode and a
    /// float's rounding to it, where Metal's clear rounds each channel to nearest, a tie to even (a
    /// packed 16-bit channel through `half`) and truncates a small float's and a shared exponent's
    /// mantissas. A normalized channel lies off a whole code there, and stores one of the two
    /// closest.
    ///
    /// Also (the runtime announces it): a plain color fill (`fillTexture…color:` without
    /// `pixelFormat:`) of a texture of such a format the device renders, which the device's clear
    /// of the color writes, to a color whose conversion is in question against Metal's fill
    /// packing, which rounds a normalized tie half up and takes the sRGB code whose decoded value
    /// is nearest.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceColorConversion,
    /// A fragment function reads a varying the device interpolates (`center_perspective`,
    /// `centroid`, `sample` or `no_perspective`, not `flat`), on every device. Vulkan states no
    /// precision for interpolation, and every device measured but MoltenVK (where Metal
    /// interpolates) rounds otherwise than Metal: on a perspective triangle list, texcoords over a
    /// full-screen quad and a colour whose `w` varies, 31% to 45% of the lanes RADV, ANV, lavapipe
    /// and NVIDIA's driver interpolated were Metal's bit for bit, the rest up to about `3e-4` apart
    /// (lavapipe, which snaps vertices otherwise) and `1.5e-4` elsewhere. The exact path,
    /// interpolating each varying in the shader by Metal's formula, is not built: Metal's formula
    /// is not known.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceInterpolation,
    /// A fragment function runs a quad or simdgroup operation, recorded on every device: a device
    /// may shade a triangle it clips as several triangles, and then a quad a seam between the
    /// pieces crosses runs once per piece, each run with the other piece's pixels as helper
    /// invocations, so the operation counts or combines fewer active lanes than Metal's, whose
    /// quad holds all four. Metal permits a partly occupied quad, so the device's answer is one
    /// Metal's API allows, not the reference's; no Vulkan query tells such a device apart, and no
    /// Vulkan mechanism keeps the clipped triangle one primitive (a guard band emulated by scaling
    /// the viewport is exact only within the viewport limits).
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, which C3 names as the answer gap of a case that meets it.
    QuadsSplitByClipping,
    /// A fragment function runs a simdgroup operation (a reduction, scan, vote, ballot, shuffle,
    /// broadcast or AGX multiply-add) or reads its `[[thread_index_in_simdgroup]]`, or discards,
    /// asks whether its thread is a helper (`quad_is_helper_thread`, `simd_is_helper_thread`) and
    /// runs a quad operation, on every device: a device's fragment subgroups are not Metal's
    /// simdgroups.
    /// A simdgroup operation runs over the pixels the device's rasterizer packs into a subgroup,
    /// so a sum, a ballot or a shuffle combines other pixels than on an Apple GPU, whose packing is
    /// measured, and the lane is the pixel's place in the device's subgroup, or in its aligned 32
    /// lanes where the subgroup holds 64 or 128 (`Tier::Cluster32`: each such run of 32 lanes is
    /// one simdgroup, as `[[threads_per_simdgroup]]`'s 32 says). Metal does not fix which pixels
    /// share a simdgroup, so each device subgroup, or each run of 32 lanes of one, is a simdgroup
    /// Metal's API allows, not the reference's. An Apple GPU also shades some quads of a function that discards as
    /// several partial quads, from the function's start, each run holding the quad's other covered
    /// pixels as helpers, where the device shades a primitive's covered pixels of a quad as one
    /// quad: a ballot of the helper query names covered pixels of the quad as helpers on the Apple
    /// GPU and none on the device (measured on the M3 Max under the full-screen triangle with every
    /// third pixel discarded: 3 of the 16 quads of an 8x8 target, 2 to 4 of the 12 of a 7x5 one;
    /// the same function without the discard runs every quad whole). Metal permits a partly
    /// occupied quad and does not fix which quads it splits, so the device's quad is one Metal's
    /// API allows. Elsewhere a quad is the same 2x2 pixels on every device, and a helper computes
    /// what its covered pixel would, so no other quad operation tells the runs apart.
    ///
    /// A device whose fragment subgroup holds fewer than two quads runs it as a partly occupied
    /// simdgroup (`Tier::PartialFragment`), which Metal permits: there the operation combines the
    /// one quad's pixels alone, so a reduction ahead of a `discard_fragment` counts no quad the
    /// discard drops, where an Apple GPU's 32-lane simdgroup counts the dropped quads of its 8x4
    /// pixel block. Measured on arm64 lavapipe, whose fragment stage states one subgroup size,
    /// 4 (`[minSubgroupSize, maxSubgroupSize]` is `[4, 4]`): thirteen discard shapes all answer the
    /// survivors, where Metal's answer counts the
    /// dropped quads in four of them. No required subgroup size reaches 32 lanes there, and
    /// nothing in Vulkan relates the invocations of two fragment subgroups, so no exact form
    /// exists on such a device.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    FragmentSimdgroupGrouping,
    /// A fragment function writes a float to a half-float color channel of a render pipeline
    /// under `TextureWriteRoundingMode::NearestEven` that blends with factors other than one and
    /// zero (`ChannelWrite::BlendedNearestEven`): Metal rounds the blend's result to nearest
    /// even, and the device's blend converts it by its own rounding (MoltenVK's toward zero), so
    /// a blended value off the half grid may store the neighbouring half; under a constant factor
    /// Metal's result is neither rule's on 2,376 of 580,508 `RGBA16Float` channels near the half
    /// denormals. The exact path, the blend in the function over an ordered input attachment, is
    /// not built.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    BlendedWriteRounding,
    /// A fragment function writes an attachment held with its channels rotated (`ARGB8Unorm`)
    /// whose blend is not exact by construction (`LoweringInputs::rotated_blends`): every blend but
    /// a minimum, a maximum, and one whose Vulkan factors are all `ZERO` or `ONE`. A blend may
    /// compute a factor pair apart in its color and alpha arithmetic, as Apple's does, and the
    /// rotation moves Metal's blue into the image's alpha and Metal's alpha into its red, so a
    /// channel may store one code apart from Metal's (up to 1,767 of 65,536 tie-heavy texels over
    /// 440 swept blend states on an M4). No query states how a device's blend rounds.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    RotatedBlendRounding,
    /// The sampler resource IDs a pipeline state takes for the constexpr samplers of the visible
    /// functions its stage links may be other than Metal's: two or more linked functions bring
    /// constexpr samplers, which Metal numbers in an order the pipeline's inputs do not decide
    /// (it differed between two M4s for the same program); or a function a table can hold that
    /// linking did not import could not be planned, so its samplers are not known; or a
    /// dynamic library's symbol, a binary function or a function-pointer group's function brings
    /// them, whose numbering is not measured. The state numbers them in linking's order. Only the
    /// IDs of sampler states made while such a state lives can differ; no shader value does.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    LinkedSamplerNumbering,
    /// An access through an argument buffer's sampler whose word the specialized module cannot read
    /// may hand the device a pixel coordinate scaled by the texture's size, where the sampler
    /// states pixel coordinates: the device's sampler reads its texels from that coordinate in its
    /// own arithmetic, with [`PrecisionNote::PixelSamplerScaling`]'s measured bound (the
    /// neighbouring texel pair, block or sub-texel step where the coordinate lies within a few
    /// units in the last place of an edge of one). Recorded on every device for every such access;
    /// a sample through a sampler whose word states a nearest pixel sampler of one isotropic level
    /// (`SamplerWord::nearest_pixel`) hands the device its texel's centre instead, where the device
    /// reads exactly Metal's texel but at a negative subnormal coordinate, which the centre's
    /// `floor` reads by the device's float controls where Metal's sampler reads texel -1
    /// ([`PrecisionNote::DeviceDenormals`]'s reading, which this note records for such a sampler,
    /// the runtime announcing it only while a pixel state lives). It is its own note, apart from
    /// [`PrecisionNote::PixelSamplerScaling`], because only the bound states decide it: the runtime
    /// announces it at the first draw or dispatch of a pipeline that records it while the device
    /// lists a sampler state of pixel coordinates, never at the pipeline's making.
    ///
    /// Validation derives no tolerance from it: at an edge the value is the neighbouring pair's,
    /// block's or step's, whatever it holds.
    ArgumentPixelSamplerScaling,
    /// A fragment function returns a `half` color output (a scalar or vector of AIR's `half`) to
    /// an attachment the pipeline has, and either its module is compiled without fast math
    /// (`air.compile.fast_math_disable`) and that attachment blends, factors one and zero
    /// included, with some channel written (`StageIo::half_blend_colors`, which the pipeline's
    /// stage reads), or its module names no math mode (under fast math the pipeline's stage records
    /// [`PrecisionNote::HalfOutputFastMathBlend`] instead). Without fast math Metal stores the
    /// `half` the function returns (a `half4` output and a `float4` of its `half` store alike under
    /// every blend and format), so an unblended attachment stores Metal's value and records
    /// nothing. For a module naming no math mode Metal may hand the output's last operation to the
    /// attachment at `float` precision, as it does under fast math: on an Apple M4 (macOS 27.0.1) a
    /// `half` product reaches the blend unrounded, the blend's result alone rounded to the format
    /// (`h(s·t + d)`, where the function here rounds the output to `half` first, `h(h(s·t) + d)`;
    /// an exact fit of every channel of a 64x48 and a 61x37 probe). And without fast math Metal
    /// blends a `half` output into a 16-bit float attachment in `half` arithmetic: the output's
    /// `half`, each factor, each product and the sum rounded to `half` to nearest even (every
    /// channel of 78 blended draws, 13 blends into `R16Float`, `RG16Float` and `RGBA16Float` over
    /// two sweeps, on an M3 Max under macOS 26.7), and into a 32-bit float one as a multiply-add of
    /// the `half` in `float`, each product rounded. A device whose blend is its own adds in `float`
    /// and rounds the sum once. So a blended channel (and, naming no math mode, one whose format's
    /// conversion of the `float` and of its `half` differ) may store a value apart from Metal's: by
    /// at most 2.5 units in the last place of a `half` at the largest operand's magnitude (the
    /// source, the destination, or the two products' magnitudes summed; a factor's counted too, the
    /// same) on NVIDIA, for the blend color and one minus it, and two on RADV, ANV and lavapipe,
    /// absolute, for a 16-bit float attachment; two units in the last place of a `float` there (one
    /// on NVIDIA) for a 32-bit float one (at 0bcc0163e5: an RTX 3090 and an
    /// RTX 5080 Laptop GPU on NVIDIA's 615.71.09, a Radeon 780M on RADV, an Iris Xe of RPL-P on
    /// ANV and lavapipe, Mesa 26.2.4, the 14 blends Metal measured over the two sweeps, each handed
    /// the output widened to `float`, as every device receives it); and 1 or 2 codes of an 8-bit or
    /// extended-range format.
    ///
    /// Also: a `float` color output (a scalar or vector of AIR's `float`) of a module compiled
    /// without fast math to a 16-bit float attachment that blends by one and zero, added, for
    /// color and alpha (`StageIo::float_blend_colors`). Into a 16-bit float attachment Metal's blend
    /// of a module without fast math keeps a zero factor's product, the destination's by zero
    /// there: a NaN of an infinite or NaN texel, and `+0` of a `-0` output over a texel of `+0` or
    /// above (plain Metal on the M3 Max under macOS 27.0.1, `half4` and `float4`
    /// outputs). The pipeline blends the output by the device's blend, whose answer for those
    /// texels is the device's own,
    /// where a `float` output would otherwise store as written.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    HalfOutputPrecision,
    /// The module samples a texture through an anisotropic sampler with Metal's anisotropic
    /// filter emulated by the device's samples at explicit levels (`lower::sampler::anisotropic`),
    /// on a device without `samplerAnisotropy`, every Vulkan sampler being isotropic there, each
    /// tap the device's isotropic filter: a sample without comparison of a 2D texture or 2D array
    /// at the fragment's implicit level or with explicit gradients, or of a cube or cube array with
    /// explicit gradients, through a constexpr sampler of `maxAnisotropy` above 1, or through a
    /// runtime sampler slot whose word states anisotropy (under that word alone). A device with
    /// the feature samples with its own anisotropic filter. Metal's footprint (its gradient lengths,
    /// level rule, tap count thresholds, tap weights and direction) is reproduced as measured on
    /// an Apple M4 (the M3 Max answering identically), in the device's arithmetic: its `Log2` and
    /// `Exp2` where Metal takes a piecewise-linear `log2` and its chords, the level and the
    /// spacing unrounded where Metal rounds them to 256ths and 4096ths, and under a linear mip
    /// filter the device's own blend of the two levels at the fractional level where Metal blends
    /// them by 256ths. The residue:
    /// - the level may differ from Metal's by its rounding (Metal's `log2` is within about
    ///   `0.006` of a level of the true one), and a tap count at a threshold may differ by one
    ///   count;
    /// - the tap spacing is a fitted approximation for three or more taps below the cap (up to
    ///   about 18/256 of a texel at the outer tap of 10 or more taps), and otherwise the device's
    ///   unrounded value;
    /// - the blend of two levels is the device's linear mip filter at the fractional level, within
    ///   its `mipmapPrecisionBits`;
    /// - each tap is the device's own filter at the level the shader hands it, so its footprint
    ///   and sum are the device's (Metal's own on MoltenVK; elsewhere
    ///   [`PrecisionNote::DeviceSampling`]);
    /// - a cube's taps step across the direction's face by Metal's face gradients (Metal under
    ///   macOS 27 samples a cube as the face's 2D plane, measured on the Apple M3 Max), the
    ///   direction moved and projected onto the face again by the device, and a tap past the
    ///   face's edge reads the next face as the device projects it, where Metal's is not measured.
    ///
    /// Measured against the M4's answers on `rgba8Unorm`
    /// at `maxAnisotropy` 2 to 16, with Metal's logarithm and blend (the device's are not yet
    /// measured there). With a linear filter, every channel is within a code: at most
    /// 0.53 (mean up to 0.071) on lavapipe, 0.41 (0.027) on ANV (Iris Xe), 0.91 (0.043) on RADV
    /// (Radeon 780M; its bilinear filter's own residue) and 0.68 (0.075) on an RTX 3090. An RTX
    /// 3090's own anisotropic filter misses by up to 62 codes (mean 6.8). With a nearest filter,
    /// the same on every device: up to 24.5 codes where a tap lies within the spacing's residue of
    /// a texel edge and reads the neighbouring texel. That is 4 of 5,072 channels (mean up to
    /// 0.024) at `maxAnisotropy` 8 or less, and 37 to 47 (mean up to 0.066) at 16. The 3090's own
    /// nearest anisotropic filter misses by up to 194 (mean 16). Validation derives no tolerance
    /// from it.
    AnisotropicFilter,
    /// The module samples a texture through a sampler whose anisotropy is known where Metal's
    /// anisotropic filter is not emulated ([`PrecisionNote::AnisotropicFilter`]'s reach), on a
    /// device without `samplerAnisotropy`, which filters it isotropically: a cube, a 3D texture, a
    /// comparison, a sample whose residency the function observes, or a texture of another class
    /// than its declared one, at an implicit level or with explicit gradients, through a constexpr
    /// sampler of `maxAnisotropy` above 1 or a runtime sampler slot whose word states anisotropy
    /// (under that word alone). An argument buffer's anisotropic sampler, whose anisotropy the
    /// function does not know, is refused there (`SamplerAnisotropy`). Validation derives no
    /// tolerance from it.
    DeviceAnisotropicFilter,
    /// A vertex function writes `[[point_size]]`, or a mesh function that outputs points
    /// `set_point_size` (a line or triangle mesh's size draws nothing, and records no note), on a
    /// device whose `pointSizeRange` ends below Metal's largest point, 511 pixels
    /// ([`Limits::max_point_size`](crate::device::Limits::max_point_size): 255 on ANV, 256 on
    /// lavapipe). The module stores a size past 511 as 511, as Metal draws it, and the device
    /// clamps a size past its own bound to that bound, so a point between the device's bound and
    /// 511 pixels draws smaller than on Metal (a square of the bound's side). Drawing such a
    /// point as two triangles is the exact path, not built.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DevicePointSize,
    /// A fragment function compiled for a multisampled pipeline reads `[[sample_mask]]`, or reads
    /// a centroid varying and runs once per pixel, on every device (no Vulkan query tells Apple's
    /// rasterizer apart):
    /// Apple's GPU draws a triangle that extends past the viewport as pieces whose shared edges
    /// cross pixels, and such a pixel's fragments can run once per piece, each reading that
    /// piece's coverage (on an M3 Max and an M4), where the device hands the pixel's fragment its
    /// coverage by the whole primitive (lavapipe by its own pieces). Where Metal splits a pixel,
    /// the function reads the whole coverage once instead of each piece's, and a centroid varying
    /// at the centroid of the whole coverage instead of a piece's; no Vulkan mechanism follows
    /// Apple's clipper. A function that runs per sample reads a centroid varying at its sample,
    /// which no split moves.
    ///
    /// Validation derives no tolerance and no explanation from it: C3 names it as the answer gap
    /// of a case that meets it (`ClippedPrimitiveSplit`).
    ClippedCoverageSplit,
    /// A fragment function that reads `[[sample_mask]]`, runs once per covered sample on Metal
    /// and writes an output (a color, depth, stencil or sample mask) is compiled for a
    /// multisampled pipeline on a device that does not run it per sample itself: it runs once
    /// per pixel and its body once per covered sample, lowest first, from a loop over the
    /// coverage (`CoverageLoop::Looped`, `lower::fragment::coverage`). Metal writes each covered
    /// sample's outputs from that sample's own run, which keeps the sample where its sample mask
    /// output has the sample's bit (an M3 Max under macOS 27.0.1, our probe `co`). The invocation
    /// writes the samples Metal's runs keep, each pass's sample mask ANDed with its sample's bit,
    /// but every one of them takes the last pass's outputs, so a sample's output differs from
    /// Metal's where the passes compute different outputs (from `[[sample_id]]`, a
    /// sample-qualified varying, or memory an earlier pass wrote). Vulkan hands an invocation
    /// under sample shading only its own sample's bit of the coverage, so no form runs each
    /// sample's pass in its own invocation and reads the pixel's coverage.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    LoopedSampleOutputs,
    /// A fragment function of a module compiled with fast math (`air.compile.fast_math_enable`)
    /// returns a `half` color output to an attachment the pipeline has that stores some channel
    /// (its write mask names one of its format's channels; `StageIo::half_fast_colors`, which the
    /// pipeline's stage reads), where [`PrecisionNote::HalfOutputPrecision`]'s conditions hold.
    /// Metal's compiler
    /// folds the attachment's blend and conversion into the function and, under fast math, fuses
    /// and regroups them per pipeline: it reads the output's `float` or its `half`, rounds each
    /// factor and product to `half` or keeps it, and takes a lerp or a common factor, by the
    /// attachment's channel count, the blend and the function's own arithmetic (an
    /// M3 Max under macOS 26.7, the consumer's sweep of a `half4` computed in `float` into
    /// `R16Float`, `RG16Float` and `RGBA16Float`: one and one reads the `float` into the one- and
    /// two-channel formats and the `half` into the four-channel one; source alpha and one minus it,
    /// destination color and one minus it, and the blend color and one minus it fit no single form;
    /// into `RGBA32Float` every channel reads the `half`, fused as `fma(d, Fd, h·Fs)` for most
    /// factor pairs, and Metal's compiler takes another fused form for the alpha lane of the
    /// function this module becomes on MoltenVK). No form the AIR and the blend state decide is
    /// Metal's, so the device's blend of the output's `half` is a near-equivalent, within the
    /// owner's acceptance of fast-math regrouping (the user's answer of 2026-10-03, extended to
    /// blends 2026-10-11): it stores a value at most three units in the last place of a `half` at
    /// the largest operand's magnitude (the source, the destination, a factor, or the two products'
    /// magnitudes summed) apart from Metal's, absolute, into a 16-bit float attachment, and at most
    /// two units in the last place of a `float` there into a 32-bit float one (MoltenVK on the M3
    /// Max, handed the output widened to `float`, against plain Metal over every blended draw of
    /// the sweep; the device's `float` blend modelled on the same draws, three units). A consumer
    /// may relay it as a known difference. The output reaches every device widened to `float`.
    ///
    /// The pipeline's stage also records it for an attachment of `StageIo::half_fast_colors` that
    /// stores some channel **without** blending (`color_writes::stores_output`, which is the test
    /// it makes), and that is deliberate rather than the blend's name read loosely: under fast math
    /// Metal's compiler may keep the output's last operation at `float` and let the attachment's own
    /// conversion round once, where this rounds the output to `half` first, so a channel whose format
    /// converts the `float` and its `half` to different codes may store a value apart from Metal's.
    /// It is the cause [`PrecisionNote::HalfOutputPrecision`] records for a module naming no math
    /// mode, under a name that reads as a blend's alone. Measured through a write that stores the
    /// output's `half`: into `BGRA8Unorm`, one code apart from Metal's in
    /// 2.2 to 2.5% of pixels for an output computed through one division or one `fwidth`, and in
    /// 2.1 to 4.5% for a coverage computed through both, where the same arithmetic with a `float`
    /// output stores Metal's codes (MoltenVK 1.4.2 on an Apple M3 Max, macOS 27.0.1). The 12
    /// `xr-draw-*` conformance cases, through an output variable, store Metal's codes.
    ///
    /// Validation derives no tolerance from it; C3 takes a row it leaves unbounded as showing
    /// the fast-math regrouping cause.
    HalfOutputFastMathBlend,
    /// Native device sampling: the module hands a sample, gather, comparison or level query to
    /// the device's own sampler on a device whose sampler is not Metal's (the module requires
    /// `VkProperty::DeviceSampling`). Its filter, footprint, level selection, anisotropy, decode
    /// and comparison are the device's. So are its level clamp, the sampler descriptor's
    /// `lodMinClamp` and `lodMaxClamp` as given, which Metal holds on its `1/64` grid
    /// (`metal::sampler::vk_sampler`), a nearest mip filter's level at a clamp halfway between two
    /// levels (Metal reads the lower, `ceil(λ + 0.5) - 1`; Vulkan lets the device round either
    /// way, and lavapipe reads the upper at `lodMinClamp` 1.5), and its reading of a subnormal normalized coordinate, which
    /// Metal's texture unit reads as the zero of its sign before the address mode decides (a
    /// clamp-to-zero axis then reads texel 0 just outside the level, measured on an Apple M3 Max
    /// on `R32Float`).
    ///
    /// The recorded bound, measured against an Apple M4 on 2D textures of one level (the deviation
    /// probe: 4,096 coordinates per case, a third near the 1/256 grid, under clamp to edge, repeat,
    /// mirrored repeat and clamp to zero, in `R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`,
    /// `RGBA8Unorm_sRGB`, `RGBA16Unorm` and `RGBA16Float`): a linear filter's largest difference of
    /// any channel, in codes of 1/255, is 0.919 on NVIDIA's 615.71.09 (RTX 3090, RTX 5080 Laptop
    /// GPU), 0.548 on RADV (Mesa 26.2.4, Radeon 780M), 0.918 on ANV (Mesa 26.2.4, Iris Xe of RPL-P,
    /// UHD 770 of RPL-S) and 1.0 on lavapipe (Mesa 26.2.4); an sRGB texture's, 1.832, 0.764, 0.912
    /// and 0.781. A nearest sample at a coordinate within the device's sub-texel precision of a
    /// texel boundary may read the neighbouring texel, whose difference it then answers: 0.1% of
    /// the probe's coordinates on NVIDIA and ANV, 0.02% on RADV, none on lavapipe. Level selection,
    /// anisotropy, cube maps, 3D textures, gathers, comparisons and offsets are not measured.
    ///
    /// A linear sample of a two-plane YCbCr texture is the device's filter of its converted
    /// texels: the texels hold Metal's reads exactly, but Metal sums their steps on the format's
    /// grid at its own weights, each term truncated. Against that sum, at 65,536 random coordinates on each of five
    /// `YCBCR8_420_2P` surfaces of both ranges and three matrices, `float` and `half` results, the
    /// largest difference of any channel, in codes of 1/255, is 1.183 on NVIDIA's 615.71.09 (RTX
    /// 3090), 0.877 on RADV (Mesa 26.0.8, Radeon 780M and RX 7900 XTX), 1.058 on ANV (Mesa 26.2.4,
    /// Iris Xe of RPL-P) and 0.872 on lavapipe (Mesa 26.2.4); at coordinates on the 1/256 grid,
    /// 0.498, 0.125, 0.125 and 0.125, and 0.56 on NVIDIA's RTX 5080 Laptop GPU over 10-bit and
    /// packed textures (615.71.09). Where the device converts the texels itself, its filter's sum
    /// is of its own conversion's texels.
    ///
    /// Validation derives no tolerance from it.
    DeviceSampling,
    /// Native device maths: the module computes a `float` or `half` math-library function by the
    /// device's own `GLSL.std.450` instruction (`Exp`, `Log`, `Pow`, `Sinh`, `Cosh` and `Tanh`
    /// among them; a composition of the device's `Exp2` and `Log2` only for `exp10` and `log10`,
    /// which no instruction is) where Metal's function is not the device's own; `fast::sqrt` the
    /// device's `Sqrt`; the fast `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and `atanh` the
    /// device's instructions, where MSL Table 8.2 defines Metal's by formulas (`fast::sin` and
    /// `fast::cos` are `+0` past `float(π/2) · 2^22` and at an infinity or NaN, `fast::tanh` a NaN
    /// once `exp(2x)` overflows); `fabs`, `sign` and `fract` the device's `FAbs`, `FSign` and
    /// `Fract`, their special operands the instruction's (Metal's `sign` of a NaN, a zero and an
    /// `f32` denormal is `+0`, its `fract` of an infinity `+0`, where `FSign` keeps a zero's sign
    /// and leaves a NaN's answer undefined and `Fract` of an infinity is a NaN); a `half`
    /// function's zeros' and NaNs' signs its `float` instruction's (Metal's `half` `sin`, `tan`
    /// and `asin` of `-0` are `+0`, its `half` `asin` NaN takes the operand's sign), and on every
    /// device a `half` `fmod`'s zero remainder its `float` remainder's, the dividend's sign
    /// (`half` `fmod(-2, 0.5)` is `-0`, where Metal answers `+0`); `pow`, `powr` and `atan2`
    /// C99's special values around the device's instruction, decided by the device's
    /// comparisons, so `pow(x, 1)` is the device's `Pow` (Metal answers `x`) and an `f32`
    /// denormal operand reads as the device reads it; and `round` C99's, from the device's
    /// `Trunc`, on every device (`round(-0.3)` is `-0`, where Metal answers `+0`).
    /// **Recorded bound:** Vulkan's precision table for the instruction, where Metal promises 4 to
    /// 6 ulp precise and its fast-math table fast (MSL Tables 8.1, 8.2): `Exp2` and `Exp` 3 + 2·|x|
    /// ulp; `Log2` and `Log` 3 ulp outside `[0.5, 2]` and `2^-21` absolute inside; `Pow` inherited
    /// from `Exp2(y · Log2(x))`; `Sqrt` inherited from `1 / InverseSqrt` (`fast::sqrt`'s too, where
    /// Metal's is `x · rsqrt(x)`; a `half` `sqrt` the device's 16-bit `Sqrt`, its bound in `half`
    /// ulp, where the module computes `half` in 16 bits under the 16-bit float controls, and the
    /// `float` one rounded to `half` elsewhere); `FAbs` and `FSign` the correct result, `Fract` and `Trunc`
    /// correctly rounded, `round`'s and a `half` `fmod`'s zero of either sign; `InverseSqrt` 2 ulp; `Sin` and `Cos` `2^-11` absolute in `[-π,
    /// π]` (both answers in `[-1, 1]` beyond); `Tan` inherited from `Sin / Cos`; `Asin`, `Acos`,
    /// `Atan` and `Atan2` 4096 ulp; the hyperbolic functions inherited from their `Exp` and `Log`
    /// formulas. A denormal operand and a NaN's encoding are the device's (`DeviceDenormals`,
    /// `DeviceNanEncoding` where recorded). Validation derives no tolerance from it.
    NativeTranscendentals,
    /// Native device maths: a precise `float` or `half` division, or Metal's fast division or
    /// reciprocal, is the device's `OpFDiv` where the device's division is not Metal's own.
    /// **Recorded bound:** Vulkan's 2.5 ulp of the
    /// quotient for a divisor in `[2^-126, 2^126]`, where Metal's precise division is correctly
    /// rounded and its fast one `x` times a reciprocal within about 0.64 ulp; beyond that range
    /// a precise division's operands are scaled by `2^-2` into it, exactly, and a fast one is the
    /// device's as Metal's flushes there. A `half` quotient is the `float` one rounded to `half`
    /// once more: within one unit in the last place of the `half` result; or, where the module
    /// computes `half` in 16 bits under the 16-bit float controls, the device's 16-bit `OpFDiv`,
    /// within Vulkan's 2.5 `half` ulp for a divisor in `[2^-14, 2^14]`, the operands of a divisor
    /// past `2^14` scaled by `2^-2` and those of a denormal divisor by `2^10` into it, exactly but
    /// where the quotient rounds to a zero or overflows anyway. Validation derives no tolerance
    /// from it.
    NativeDivision,
    /// Native device maths: a `float` fused multiply-add, or a `half` one where the device's
    /// 16-bit `Fma` is not measured to be Metal's, is the device's `GLSL.std.450 Fma`
    /// (on the `half` operands held in `float`, its result rounded to `half`), where Metal's rounds
    /// once. On a device `Device::from_queries` made that computes `half` in 16 bits under the
    /// 16-bit float controls, a `half` one is the device's 16-bit `Fma` on the `half` operands.
    /// **Recorded bound:** a `half` result held in `float` within one unit in its last place of
    /// Metal's (the product of two `half` values is exact in `float`, so only the two roundings
    /// differ); a `float` one, and a 16-bit one, within half a unit in the last place of the
    /// product plus half a unit in the last place of the result, in its own format (Vulkan lets
    /// `Fma` round the product and then the sum). Validation derives no tolerance from it.
    NativeFma,
    /// Native device sampling (a texel conversion): the module hands a read, sample or gather
    /// with a `float` result the device's own conversion of the texel's codes. Recorded for every
    /// such access of a
    /// texture slot or a heap texture (not a heap's texture buffer,
    /// [`PrecisionNote::DeviceNanEncoding`]). The recorded bound, of a texel whose access reads
    /// one texel of one level:
    /// - an 8-, 10- or 16-bit normalized code `c`: Metal's `c / (2^b - 1)` correctly rounded;
    ///   the device's within one `f32` ulp of it where measured (lavapipe multiplies by the
    ///   reciprocal: some 8-bit codes one ulp high, some 10-bit codes of `BGR10A2Unorm`, a 16-bit
    ///   code and 512 `Depth16Unorm` codes one ulp low), or a 10-bit code widened by replicating
    ///   its bits to 16, `(c << 6 | c >> 4) / 65535` within one `f32` ulp, which lies within 0.015
    ///   of a code of Metal's (NVIDIA's RTX 5080 Laptop GPU, a two-plane YCbCr texture's
    ///   `A2B10G10R10_UNORM_PACK32` codes);
    /// - a 4-, 5- or 6-bit channel of a packed format: the device may widen it by replicating its
    ///   bits (ANV, NVIDIA), within 0.18 of a code (`1 / (2^b - 1)`) where it replicates a 5- or
    ///   6-bit channel to 8 bits and exactly for a 4-bit one; or convert it by the reciprocal,
    ///   within one `f32` ulp of Metal's `c / (2^b - 1)` (lavapipe: some codes of `ABGR4Unorm`
    ///   one ulp high, and of `A1BGR5Unorm`, `B5G6R5Unorm` and `BGR5A1Unorm` one ulp apart);
    /// - a half-float or small-float texel's NaN: the device's encoding, where Metal answers
    ///   `0x7fc00000`;
    /// - a 32-bit float texel's NaN or denormal in a gather: the device's encoding of a NaN, and a
    ///   denormal flushed to a zero of its sign, where Metal hands back the texel's own bits
    ///   (NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU gathers every NaN as `0x7fffffff` and
    ///   every denormal as a zero, and reads and samples both as stored);
    /// - a 32-bit float texel's NaN or denormal in a nearest sample: the device's encoding of a
    ///   NaN, and a denormal flushed to a zero of its sign on a device that flushes it there,
    ///   where Metal hands back the texel's own bits (ANV's Mesa 26.2.6 on the Iris Xe samples
    ///   every NaN as `0xffc00000` and every denormal as a zero of its sign, and reads and gathers
    ///   both as stored; RADV's Mesa 26.2.6 on the RX 7900 XTX and the Radeon 780M samples a
    ///   denormal as a zero, and gathers it as stored);
    /// - every other class: the device's value, which Metal's matches.
    ///
    /// A filtered sample's value is the filter's. Validation derives no tolerance from it.
    DeviceTexelConversion,
    /// Native vertex input (a conversion): a vertex attribute of a normalized format
    /// (`UCharNormalized` through `Short4Normalized` of either sign, `UChar4Normalized_BGRA`,
    /// `Int1010102Normalized`, `UInt1010102Normalized`) that fixed-function vertex input fetches
    /// is the device's conversion of its codes, through the format's own Vulkan `UNORM` or `SNORM`
    /// vertex format. Metal's fetch answers `c / (2^b - 1)` unsigned and
    /// `max(c / (2^(b-1) - 1), -1)` signed, correctly rounded, but an `Int1010102Normalized`
    /// 10-bit field's, the code times `float(1 / 511)` rounded to `float`, and to a `half`
    /// attribute the quotient rounded once to `half` (measured at every code on an Apple M3 Max,
    /// macOS 27.0.1); Vulkan states the correctly rounded quotient and a driver may compute it
    /// otherwise (by the reciprocal). Recorded on a render pipeline's vertex stage where its vertex
    /// descriptor gives such a format to an attribute fixed-function input fetches
    /// (`pipeline::link::vertex`), whatever width the function declares the attribute with (a
    /// `half` attribute reads the device's `float` value rounded to `half`), on every profile. A
    /// function compiled alone records nothing for them: its module is the same
    /// for every format the device fetches. A **pulled** attribute (one the shader fetches from
    /// its buffer, a kernel's `[[stage_in]]` attribute and a post-tessellation function's
    /// included) is the module's own product of the code and its maximum's reciprocal,
    /// `float(c) · float(1 / (2^b - 1))` (signed `max(float(c) · float(1 / (2^(b-1) - 1)), -1)`),
    /// rounded once; the function records the note on every profile where that misses Metal's
    /// fetch at some code: read as `float`, every normalized format but `Int1010102Normalized`
    /// (whose 10-bit fields Metal computes as this product); read as `half`, a 16-bit format.
    /// **Recorded bound:** within one `f32` ulp of Metal's `float` where measured, but a signed
    /// code past -1 on lavapipe per instance. llvmpipe multiplies by the reciprocal: read at every
    /// code (Mesa 26.0.8 on LLVM 21.1.8, arm64), every normalized format is within one ulp of
    /// Metal's per vertex (some 8-bit unsigned codes one ulp high, 127 as `0x3EFEFF00` where Metal
    /// reads `0x3EFEFEFF`; some 16-bit, signed and `UInt1010102Normalized` codes one ulp low;
    /// `Int1010102Normalized` Metal's), and per instance the most negative code of an 8- or 16-bit
    /// signed format reads unclamped, `-2^(b-1) / (2^(b-1) - 1)` (-128 as -1.0079, -32768 as
    /// -1.00003), where Metal reads -1. RADV's unsigned formats are Metal's at the sites measured,
    /// its signed ones unmeasured. A 16-bit format read as `half` within one `half` ulp of Metal's,
    /// on a device whose `float` is Metal's: its `float` quotient rounded to `half` again differs
    /// at codes 65455 and 65519 unsigned, 32759 and -32759 signed. A pulled attribute's product,
    /// computed exhaustively: one `f32` ulp from Metal's quotient at 126 of the 256 8-bit
    /// unsigned codes (high, 127 as `0x3EFEFF00` where Metal reads `0x3EFEFEFF`), and toward zero
    /// at 16 signed 8-bit codes, 24 unsigned 10-bit codes, and 512 unsigned and 1,536 signed
    /// 16-bit codes; exact at 0, at the maximum and at -1; read as `half`, one `half` ulp at the
    /// same four 16-bit codes. Validation derives no tolerance from it.
    DeviceVertexConversion,
    /// Native half rounding (a conversion): the module rounds a `half` result it computes in
    /// `float` (a `half` operation's, a function's, a conversion from `float`'s, a texel's) by
    /// the device's conversion of the `float` to 16 bits and its widening (`OpFConvert` of the
    /// value itself), on a device with `shaderFloat16` whose float controls do not make that
    /// conversion Metal's (`RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for
    /// 16 bits, under which ANV and RADV round through the conversion exactly and record no
    /// note): NVIDIA, whose driver folds the conversion's round trip even under those controls,
    /// MoltenVK, lavapipe, and any device without them. NVIDIA's modules compute under those controls all the same: a `half` `+`,
    /// `-`, `*`, `/` and `sqrt` there are the device's 16-bit instructions, whose results take
    /// no such rounding. Without them Vulkan leaves the conversion's rounding mode and its 16-bit
    /// denormals to the device. Also recorded under those controls on RADV where the `half` is the
    /// result of an `f32` addition, subtraction, multiply or fused multiply-add: its compiler fuses
    /// the operation into the conversion and rounds the exact result once, one unit in the last
    /// place from
    /// Metal's `half` of the `float` result where that result lies on a `half` midpoint. On a
    /// device `Device::from_queries` made, which reads no driver identity, every device with those
    /// controls takes the pair (NVIDIA's included), and every module that rounds through it records
    /// this note, a static note of the rounding: a driver's compiler may fold the pair or fuse the
    /// operation so. The recorded bound, against Metal's `half(float)` (to nearest even on the half grid, half denormals kept,
    /// infinity from 65520 up, `0x7E00` for a NaN):
    /// - a finite result within one unit in the last place of the `half`: NVIDIA's 615.71.09
    ///   rounds to nearest even on the RTX 5080 Laptop GPU (every `float` pattern, loaded and
    ///   computed) and toward zero on the RTX 3090; lavapipe (Mesa 26.0.8) a loaded value toward
    ///   zero; Metal's compiler on MoltenVK folds the conversion into the instruction computing
    ///   the value, which rounds toward zero (`a1_half_tanh_regimes`, case 27353187: 1,204 of its
    ///   16,384 halves one ulp toward zero on the M3 Max);
    /// - or no rounding at all: NVIDIA's 615.71.09 on the RTX 3090 drops the conversion pair, with
    ///   or without a `NoContraction` barrier, and 620.70.06 drops it under the 16-bit
    ///   `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` too (of all 2^32
    ///   inputs, every `float` that is no `half` comes back unchanged); lavapipe drops it in arithmetic on Mesa 26.2.4 with
    ///   LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, barrier or not, whose
    ///   arithmetic measured rounded. The result is then the `float` computation itself: its
    ///   differences from Metal's compound across operations, and a value past the `half`'s range
    ///   stays finite until a `half` store narrows it by the device's conversion (to infinity, or
    ///   to 65504 rounding toward zero). The RTX 5080 Laptop GPU on the
    ///   same driver rounds a computed sum (8192 of 8192 sums Metal's) but drops the pair of a
    ///   `half` color output converted from `float` that it stores unblended into a 32-bit float
    ///   attachment: the `float` itself, whose `half` is what it stores into a 16-bit float one
    ///   (`R32Float`, `RG32Float` and `RGBA32Float`);
    /// - on a device that rounds toward zero, a finite value from 65520 up is 65504 where
    ///   Metal's is infinity;
    /// - a result below the smallest normal `half` (`2^-14`) may be the zero of its sign on a
    ///   device that flushes 16-bit denormals (the RTX 5080 keeps them);
    /// - a NaN's encoding and the sign of a zero result are the device's: on the RTX 5080 a
    ///   computed `half` NaN is `0x7FFF` (where 18 C3 cases differed in NaNs, signs of zero,
    ///   `atan2` and `pow` at special operands, and `half` texel and pack conversions).
    ///
    /// Recorded for every such rounding, a conversion of a `float` the function holds (an AIR
    /// `fptrunc` or `air.convert` from `float`), of a texel's value, or of a `half` operation or
    /// function the module computes in `float` (Metal's `half` `tanh` answers 7 of the 65,536
    /// `half`s otherwise than the conversion of its `float` `tanh`). A texel's value is an image
    /// instruction's texel, a texel the module decodes itself (an extended-range, padded or bfloat
    /// code) or a merge of texels (a heap read's fetched texel and its decode routine's): where
    /// Metal's `half` texel is not that rounding of the value, the access records its own note
    /// too ([`PrecisionNote::HalfTexelConversion`], outright for a heap element;
    /// [`PrecisionNote::DevicePackConversion`] for a decoded texel).
    ///
    /// Every `half` a module reads from its bits (a load, a reinterpretation, a texel's bits) is
    /// the device's widening of them, on every device: an `OpFConvert` to 32 bits of the bits read
    /// as a 16-bit float with `shaderFloat16`, `GLSL.std.450 UnpackHalf2x16` of them without it.
    /// So too every `half` a module encodes to its bits (a store, a reinterpretation, a texel's
    /// bits) where they are not the bits it was read from: the device's narrowing of it, an
    /// `OpFConvert` to 16 bits read as a word with `shaderFloat16`, `GLSL.std.450 PackHalf2x16`
    /// without it. A module that reads or encodes one records this note too where that widening
    /// or narrowing is not under the float controls' `DenormPreserve` for 16 bits (ANV, RADV,
    /// NVIDIA), so on every device without `shaderFloat16`: its bound is the last two items', a half denormal read or
    /// encoded as the zero of its sign on a device that flushes 16-bit denormals (the Vulkan spec
    /// lets a device flush a denormal "obtained via unpacking an integer" without `DenormPreserve`
    /// for 16 bits, and gives `UnpackHalf2x16` and `PackHalf2x16` implementation-defined
    /// precision) and a NaN's bits the device's ([`PrecisionNote::NativeNanEncoding`], which the
    /// module records); every other `half` widens to its exact value and encodes to its own bits.
    ///
    /// Validation derives no tolerance from it.
    NativeHalfConversion,
    /// Native half rounding (a quantize): the module rounds a `half` result it computes in
    /// `float` (a `half` operation's, a function's, a conversion from `float`'s, a texel's) by the
    /// device's `OpQuantizeToF16` alone, on every device without `shaderFloat16`. An integer's
    /// `half` is not this rounding: it rounds by integer arithmetic, exactly, around NVIDIA's
    /// 615.71.09 keeping an integer conversion unrounded under the quantize. Vulkan leaves the
    /// quantize's rounding mode to the device, and the SPIR-V specification lets it answer either
    /// zero for a magnitude too small for a normal `half`, a zero's own included. The recorded bound, against Metal's `half(float)`: a finite result within one unit in the last
    /// place of the `half`; a value from 65504 up to 65520 may be infinity where Metal's is 65504
    /// (NVIDIA's 615.71.09 rounds `0x477FEFFF` to infinity); a value below the smallest normal
    /// `half` (`2^-14`) may round to a zero, of either sign, where Metal answers the half denormal,
    /// or `2^-14` from just below it (the quantize tests the value it rounds; NIR's
    /// `fquantize2f16` flushes every one, SwiftShader's too); where the rounding's input is a
    /// zero or too small for a normal `half`, the result may be either zero, so a negative zero
    /// may come out positive (RADV's on Polaris does), which a program sees in a reciprocal's or
    /// `atan2`'s sign and in the bits written to a `half` target or buffer; a NaN's encoding is
    /// the device's.
    /// Validation derives no tolerance from it.
    NativeHalfQuantize,
    /// Native device maths: a `float` or `half` operation's NaN result is the device's, its sign
    /// and payload unselected, where Metal answers a NaN the device is not measured to answer:
    /// every NaN Metal answers on a device whose arithmetic is not measured to answer Metal's
    /// canonical NaN, and every NaN but the canonical one (an operand's bits, or the canonical NaN
    /// with an operand's sign) on a device measured to answer it (MoltenVK on an Apple GPU).
    /// **Recorded bound:** where
    /// Metal answers a NaN the device answers a NaN (the module keeps
    /// `SignedZeroInfNanPreserve` where Metal's entry keeps NaNs), its sign and payload the
    /// device's own (llvmpipe: `0xffc00000`), where Metal answers its canonical `0x7fc00000`
    /// (`0x7e00` as `half`), an operand's bits (`round`, `fast::tan`, and precise `pow`
    /// of a NaN base to the exponent one), or the canonical
    /// NaN with an operand's sign (`asin`, `atan`, `asinh`, `atanh`); a `float` `fneg` and `fabs`
    /// keep a NaN operand's payload on Metal, and a `half` `fract` returns it. Not recorded for a
    /// `float` `ldexp`, which the module computes on the bits and so answers a NaN operand itself,
    /// as Metal does, on every device. Every non-NaN result
    /// is as without it. Also recorded by every module that reads a `half` from its bits (a load, a
    /// reinterpretation, a texel's bits), on every device: the read is the device's widening (an
    /// `OpFConvert` of the bits read as a 16-bit float with `shaderFloat16`, `UnpackHalf2x16`
    /// without it), and a NaN widens to a NaN whose bits are the device's, where Metal widens
    /// every `half` NaN to `0x7fc00000` and moves one it does not widen with its bits, so a `half`
    /// NaN the module moves as a value (through the program's select, phi or sign operation) may
    /// leave with the device's bits; a `half` stored back unchanged keeps its bits, through a
    /// containment check or the choice among a pointer's candidate buffers too. And by every module
    /// that encodes a `half` to its bits where they are not the bits it was read from (a store, a
    /// reinterpretation, a texel's bits), on every device: the encode is the device's narrowing
    /// (an `OpFConvert` to 16 bits with `shaderFloat16`, `PackHalf2x16` without it), and a NaN
    /// narrows to a NaN whose bits are the device's, where Metal keeps a `half` NaN's sign and
    /// payload. Every other `half` widens to its value and narrows to its bits, but a denormal on a
    /// device that may flush 16-bit ones ([`PrecisionNote::NativeHalfConversion`]). And by every
    /// module whose entry reaches a `float` constant whose bits are a signalling NaN, the
    /// program's own or one Metal's compiler leaves where a NaN operand decides an operation:
    /// Metal keeps its bits, and the device's move of it as a `float` may quiet it. Validation
    /// derives no tolerance from it.
    NativeNanEncoding,
    /// Native device maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its
    /// `clamp(x, +0, 1)`) on `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or
    /// `NClamp`, on a device not measured to answer Metal's operand; a `saturate` of a format
    /// whose operand Metal's other functions do not pick (`bfloat`, `double`) on every device; and
    /// `median3` on every device, `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and
    /// `NMax`, whose pick of two NaNs and of an `f32` denormal is the device's.
    /// **Recorded bound:** the
    /// result is one of the operands, the one Metal answers but at three ties and one undefined
    /// answer: of two NaNs a NaN (its bits the device's), where Metal answers the second operand's;
    /// of `+0` and `-0` either zero, where Metal answers the second; an `f32` denormal beside a
    /// zero or another denormal compared by its value where the device's `NMin` and `NMax` keep
    /// denormals, whether or not its arithmetic flushes them, where Metal compares it as a zero
    /// and answers the second; and a `fast::median3` whose first two operands are NaNs and whose
    /// third is not answers the third, where Metal's answers the second, a NaN. The third is
    /// precise `median3`'s answer there on Metal and on the device, and MSL leaves a NaN operand
    /// under fast math undefined (section 8.1), so the fast form's NaN is one reading of an
    /// undefined answer (the M3 Max's under macOS 27.0.1: the eight lanes of `nan_minmax_fast`
    /// whose first two operands are NaNs beside a `+0`). Validation derives no tolerance from it.
    NativeMinMax,
    /// Native device maths: Metal's `dot` on `float` and `half` is the device's `OpDot` (a `half`
    /// one summed in `float` and rounded once to `half`, or the device's 16-bit `OpDot` where the
    /// module computes `half` in 16 bits under the 16-bit float controls and holds its operands
    /// there, `spirv::half_hold`), not Metal's fused chain of its products, on every device.
    /// **Recorded bound:** a `float` `dot` of `n` lanes within
    /// `(3n - 1) * 2^-24 * sum |a_i * b_i|` of Metal's chain, plus `n * 2^-126` where a product is
    /// an `f32` denormal the device keeps (Vulkan's `OpDot` inherits `OpFMul`'s and `OpFAdd`'s
    /// rounding, `2n - 1` roundings, where Metal's chain rounds `n` times); a `half` one rounded
    /// from `float` within `(n + 3) * 2^-11 * sum |a_i * b_i| + (n + 2) * 2^-25` of Metal's chain of
    /// `half` fused multiply-adds, and a 16-bit one within
    /// `(3n - 1) * 2^-11 * sum |a_i * b_i| + (3n - 1) * 2^-25` (`2n - 1` correctly rounded 16-bit
    /// roundings under the controls). Validation derives no tolerance from it.
    NativeDot,
    /// Native device maths: Metal's `mix(x, y, t)` on `float` and `half` is the device's `FMix`
    /// (a `half` one computed in `float` and rounded once to `half`, or the device's 16-bit `FMix`
    /// where the module computes `half` in 16 bits under the 16-bit float controls and holds its
    /// operands there, `spirv::half_hold`), not Metal's `fma(t, y, fma(-t, x, x))`, on every
    /// device. **Recorded bound:** a `float` result within
    /// `2^-21 * (|x * (1 - t)| + |y * t|)` of Metal's (Vulkan's `FMix` inherits
    /// `x * (1 - t) + y * t`'s rounding, four roundings, where Metal's rounds twice); a `half` one
    /// rounded from `float` within `3 * 2^-10 * (|x * (1 - t)| + |y * t|) + 2^-23`, and a 16-bit
    /// one within `2^-8 * (|x * (1 - t)| + |y * t|) + 2^-22`. Validation derives no tolerance
    /// from it.
    NativeMix,
    /// Native device maths (a conversion): `air.pack` or `air.unpack` of a normalized format is
    /// the device's: `GLSL.std.450`'s `PackUnorm4x8`, `PackSnorm4x8`, `PackUnorm2x16` and
    /// `PackSnorm2x16` (and their `Unpack`s) for the four formats it names, and for `rgb10a2` and
    /// `rgb565` those instructions' definition on each field, the device's `NClamp`, multiply,
    /// `RoundEven` and conversion (a pack) or its multiply by the scale's reciprocal (an unpack);
    /// `unorm4x8_srgb`'s colour channels take the MSL transfer functions by the device's multiply,
    /// division and `Pow` around the 8-bit instruction. Metal rounds the exact product to its code
    /// and divides correctly rounded (`unorm565` in `float`: it rounds the product to `float`
    /// first, and multiplies by the `float` nearest the reciprocal, as the definition does),
    /// answers code 0 for a NaN channel, and takes an sRGB colour
    /// through its twelve-bit table: a decode answers its step `k / 4095` nearest the transfer, an
    /// encode the byte of that table's step. Also: a write of an 8-bit sRGB storage texture held
    /// through its UNORM alias (`StorageImageClass::SrgbEncode`, Vulkan having no sRGB storage
    /// view), at a slot or held by an argument buffer, encodes by `unorm4x8_srgb`'s pack.
    /// Recorded on every device. **Recorded bound:**
    /// Vulkan bounds the instructions by their definition (`round(clamp(c) * scale)` and
    /// `c / scale`): a packed code within one of Metal's, apart only where `clamp(c) * scale` lies
    /// within its `float` rounding of a half-integer (`Round` may tie either way); an unpacked
    /// value within `OpFDiv`'s 2.5 ulp of the correctly rounded quotient; an sRGB code within one of
    /// Metal's (the transfer's `Pow` inherited from `Exp2(y * Log2(x))`), and a decoded sRGB
    /// colour within half a twelve-bit step (`1 / 8190`) of Metal's step, beyond its division's
    /// and `Pow`'s bounds; a NaN channel's code, and a `half` result rounded once more to `half`,
    /// the device's. The same conversions serve, noted alike: an implicit imageblock access of a
    /// normalized attachment (a `half` one's load rounding the unpacked `float` once more, where
    /// Metal rounds `c / D` once), a fragment function's read and write of an 8-bit sRGB attachment
    /// through its word view. So does every conversion of an extended-range, padded 10-bit or
    /// bfloat texel
    /// (`lower::arith::extended_range`; `ArithPlan::converts` records it), on every device, no
    /// Vulkan format holding them: a read `float(c - 384)` (a padded code's `float(c)`) times the
    /// `f32` nearest `1 / 510` (`1 / 1023`), a write `RoundEven` of the clamped value's `f32`
    /// product plus 384, an sRGB twin's colour the MSL transfer by the device's `Pow` on the
    /// magnitude, a `half` result the device's narrowing, a bfloat NaN as it is (a write's kept a
    /// NaN by its quiet bit), where Metal divides correctly rounded, rounds the exact product,
    /// rounds a `half` with ties away from zero, converts the sRGB twins through its own tables,
    /// makes a bfloat NaN `0x7fc00000` or `0x7fc0` and rounds a bfloat sample's `half` toward zero:
    /// a linear value within one `f32` ulp of Metal's, a code within one (`R10Unorm_X6_sRGB`'s and
    /// `RGB10A2Unorm_sRGB`'s colour within two below `0.0031308`, where Metal rounds the value to
    /// `2^-13` before its table, a step there spanning 1.6 codes), an sRGB twin's decoded value
    /// within half a `2^-13` step of Metal's table (`R10Unorm_X6_sRGB`'s and
    /// `RGB10A2Unorm_sRGB`'s colour within 1.64 steps) beyond the `Pow`'s bound, a `half` result
    /// within one `half` ulp. A `float` or `half` read, sample or gather through an argument
    /// buffer that decodes a texture the heap holds by its raw texels alone
    /// (`lower::texture::heap_raw`) records it as a specialized note: it holds where a texture
    /// the stage's handles can name is one (`SpecKey::HeapRawTextures`, which a plan gives),
    /// unless the stage records it outright for another conversion. Validation derives no
    /// tolerance from it.
    DevicePackConversion,
    /// A tessellation pipeline drawn through the device's tessellation stages (a device with
    /// `tessellationShader`), on every device (no Vulkan query tells Apple's tessellator apart):
    /// the device's tessellator orders each patch's triangles and picks each one's first corner,
    /// and where Vulkan leaves it to the implementation chooses the triangles that join a patch's
    /// rings, where Metal's tessellator draws the D3D11 reference's triangles in its own order,
    /// each triangle's first corner by its stitch step. The domain points are Metal's; a flat varying (the first corner's), a value interpolated across
    /// another triangulation, and the order in which a patch's triangles reach the blend or the
    /// depth test may differ: every patch measured differs on the RTX 3090, 55-95% of them on RADV
    /// and lavapipe. MoltenVK on an Apple GPU hands the patches to Metal's own tessellator and
    /// records none. Recorded on the evaluation stage.
    /// Validation derives no tolerance from it.
    TessellationTriangleOrder,
    /// The stage's function holds a branch whose arms hold a quad or simdgroup operation that
    /// answers over the lanes of the branch's block when Metal's backend speculates the arms (a
    /// quad shuffle, broadcast or vote, a simdgroup shuffle, `simd_is_first`), where no
    /// measurement says whether Metal speculates them: an arm of an instruction, size or shape no
    /// probe on Apple's GPU covers. Each branch Metal was measured to speculate is speculated
    /// (`air::canon::speculate`); this one keeps the AIR's lanes. **Residual:** that operation may
    /// answer over the lanes that take the arm where Metal's answers over every lane of the
    /// branch's block (a quad's first lane, a shuffle's source lane, a vote). Nothing else in the
    /// stage changes. Validation derives no tolerance from it.
    UnmeasuredGroupSpeculation,
    /// An extended-range texture's fragment sample at an implicit level and its sample with
    /// explicit gradients (`BGR10_XR`, `BGRA10_XR` and their sRGB twins), which read texels in the
    /// shader on every device: they take Vulkan's level of detail computed in the shader and then
    /// Metal's level rule (`lower::sampler::level`). Vulkan's level of detail, whose scale factor
    /// may be approximated, is not Metal's, which is not measured from derivatives. Recorded through
    /// a constexpr sampler with a mip filter, and through a runtime sampler where its word states
    /// one. A bias past the limit and hoisted derivatives record [`PrecisionNote::GradFromBias`]
    /// instead. **Residual:** the sample may read another level, or another filter, than Metal's
    /// near a boundary. Validation derives no tolerance from it.
    DeviceSampleLevel,
    /// A render pipeline covers by alpha (`alphaToCoverageEnabled`) under
    /// `TextureWriteRoundingMode::NearestEven`, and its fragment function writes a `float` alpha
    /// to a half-float attachment 0 that stores it as written: Metal stores the alpha rounded to
    /// the nearest-even half and reads the coverage from the alpha before rounding (the coverage
    /// of 16,384 alphas the same under either mode on an Apple M3 Max), which no device
    /// separates, so the function rounds the alpha and the coverage reads the rounded one. The
    /// stored alpha is Metal's; a fragment whose alpha lies within half a `half` step of a
    /// coverage threshold may cover one sample more or fewer (2 of the 16,384 alphas on the
    /// M3 Max, each just past the midpoint between two halves). That bound is of the rounding
    /// alone: which samples an alpha covers is the device's own mapping, recorded as
    /// [`PrecisionNote::DeviceAlphaCoverage`] where it is not measured to be Metal's. The exact
    /// path, Metal's coverage computed in the function, is open only where the device covers as
    /// Metal does (MoltenVK), and needs Metal's coverage pattern, which is not modelled.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    CoverageReadsRoundedAlpha,
    /// Native device coverage: a render pipeline covers by alpha (`alphaToCoverageEnabled`, kept
    /// where its fragment function writes a four-component float to attachment 0,
    /// `GraphicsState::alpha_to_coverage`), on every device. The device converts the alpha to its
    /// coverage mask by its own algorithm. Metal's, measured over 639 alphas a sample count on an
    /// Apple M3 Max under macOS 27.0.1, covers a pixel at one sample where its alpha is above 1/2,
    /// and at two and four samples `ceil(alpha * n - t)` of its `n` samples, `t` a 2x2 ordered
    /// dither of the pixel's window coordinates (1/8 and 5/8 at even and odd `x` of an even row,
    /// 7/8 and 3/8 of an odd one), either count within an `f32` ulp of a threshold. Vulkan fixes no
    /// algorithm ("No specific algorithm is specified for converting the alpha value to a temporary
    /// coverage mask. It is intended that the number of 1's in this value be proportional to the
    /// alpha value (clamped to \[0,1\]), with all 1's corresponding to a value of 1.0 and all 0's
    /// corresponding to 0.0. The algorithm may be different at different framebuffer
    /// coordinates."), so a pixel may cover other samples than Metal's: at one sample, a whole
    /// fragment stored or dropped. Recorded on the fragment stage.
    ///
    /// **Recorded bound,** over that sweep against Metal's reading, pixel by pixel: on lavapipe
    /// (Mesa 26.0.8 on LLVM 21.1.8, arm64), which takes the same dither at one sample as at four,
    /// every pixel covers `floor(alpha * n)` or `ceil(alpha * n)` samples, as Metal's does, so it
    /// lies within one sample of Metal's, and is Metal's where `alpha * n` is an integer. At one
    /// sample a pixel whose alpha lies between its dither threshold and 1/2 is stored where Metal
    /// drops it, or dropped where Metal stores it (2,560 of the 10,224 pixels); at four samples a
    /// pixel is one sample apart only where `alpha * 4` lies within an `f32` ulp of a threshold
    /// (96 of 10,224). NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU stores the fragment at one
    /// sample at an alpha of 1/2, where Metal drops it (the conformance case `a2c-on-1-alpha-50`);
    /// beyond that row, and on every other device, the count is unmeasured, and Vulkan bounds it
    /// by nothing but its intent above.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceAlphaCoverage,
    /// A fragment function reads an `interpolant` through `interpolate_at_offset(offset)`, which
    /// Metal evaluates at the offset as given from the pixel's upper-left corner, at any distance
    /// and on no grid (the Apple M3 Max, macOS 27.0.1: an offset of 0.999 read at 0.999, one of
    /// `1e9` extrapolated to `1e9`, every step of a 1/512-pixel sweep a step of its own, a NaN or
    /// infinite offset NaN). The module reads `GLSL.std.450 InterpolateAtOffset` at the offset
    /// moved to the device's origin (the pixel center, less a half; on MoltenVK, whose
    /// SPIRV-Cross hands the offset to Metal plus 7/16, less 7/16), clamped to the device's
    /// `[minInterpolationOffset, maxInterpolationOffset]`, which Vulkan requires
    /// (`VUID-RuntimeSpirv-Offset-06344` to `-06347`), a NaN to the minimum. The device then
    /// rounds the offset to its `subPixelInterpolationOffsetBits` (at least 4: sixteenths), so the
    /// read lies within half a step of Metal's where the offset is in range, and at the range's
    /// nearer end where it is not (Metal: past the pixel, or NaN). On MoltenVK, where Metal reads
    /// the offset itself, the read is Metal's within the range (`[-1/16, 15/16]` from the corner
    /// on the M3 Max, whose limits are `[-0.5, 0.5]`).
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceInterpolationOffset,
    /// A fragment function reads a perspective input at a sample or an offset other than the
    /// pixel center (`interpolate_at_sample`, `interpolate_at_offset`, or a sample-qualified
    /// varying a function looping over its covered samples reads at each pass's sample: `GLSL.std.450
    /// InterpolateAtSample` or `InterpolateAtOffset`), recorded on every device: a device may
    /// answer such a read as the pixel center's value extrapolated along the quad's coarse
    /// derivatives, as RADV does. Metal interpolates in perspective at the location itself. The extrapolation is first-order, so the read is off
    /// by the input's curvature across the distance, which grows with the triangle's spread of
    /// clip `w` and the distance from the quad's top-left pixel: on a triangle of `w` 1, 2 and 4
    /// over a 6x6 target, up to about 1.0 in a lane whose vertices hand 3, -2 and 0.5 (the
    /// Radeon 780M, Mesa 26.2.6). A no-perspective input is linear in the framebuffer, so its
    /// read is exact and records nothing. The answer is the device's own interpolation, which
    /// Vulkan states no precision for; no form computes it otherwise.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    ExtrapolatedInterpolation,
    /// A fragment function reads a color attachment (framebuffer fetch), on every device. It
    /// reads the attachment as an input attachment, and the consumer records a barrier of the subpass's
    /// dependency on itself, by region, from the color writes of the draws before to the input
    /// attachment reads of a draw that fetches (and from those reads to the color writes of the
    /// draws after): a fetch reads what every earlier draw stored at its pixel, Metal's answer.
    /// Under the identity domains it reads the attachment's texture at its own pixel instead
    /// ([`PushField::FetchedColor`](crate::pipeline::PushField::FetchedColor)), and the
    /// consumer orders each fetching draw after the earlier draws of its pass by a barrier between
    /// them. Vulkan orders nothing between the overlapping fragments of one draw, where Metal
    /// orders them in primitive order: a fragment that covers a pixel an earlier fragment of its
    /// own draw covered may read the texel from before that fragment's write, and its own write may land
    /// before that one. Exact for a draw whose primitives do not overlap at a pixel (one
    /// fullscreen triangle, a rectangle of two triangles, disjoint quads); otherwise each such
    /// pixel holds the result of some order of the draw's fragments there, not necessarily
    /// Metal's. A raster order group on the color input, or on a color output of an attachment
    /// the function reads, takes the same form there, without an interlock.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch for review.
    UnorderedFetchWithinDraw,
    /// A render pipeline draws its private logic operation through Vulkan's `logicOp`
    /// ([`GraphicsState::logic_op`](crate::pipeline::GraphicsState::logic_op)), the device's
    /// native form, into an attachment whose codes Metal's operation takes otherwise: an
    /// `RGB10A2Unorm`, `BGR10A2Unorm` or signed normalized attachment a first output of the
    /// fragment function writes, with a channel its write mask keeps. Metal widens each stored
    /// 10-bit code (and the 2-bit alpha) to 16 bits, converts the fragment's value to a 16-bit
    /// code, operates, and narrows the result back to the channel's code, rounded to nearest; and
    /// it reads a signed channel's lowest code (`0x80`, or `0x8000`) as the code above it and
    /// stores a result of the lowest code as the code above (probe `logic_operation` on an Apple
    /// M3 Max, macOS 27.0.1). Vulkan's operation takes the fragment's value converted to the
    /// channel's own code and the stored code as it is, and stores its result as it is, so a
    /// channel may store another code than Metal's: lavapipe's xor into `RGB10A2Unorm` stored
    /// green 938 where Metal's rule gives 937, and its xor of `0xe0` over an `RGBA8Snorm`
    /// channel's `0x80` stored `0x60` where Metal's gives `0x61`. The distance is not bounded,
    /// since the operation's result follows the codes' bits, not their values. A device without
    /// `logicOp` refuses the operation. The fragment stage records it whatever the
    /// specialization.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    NativeLogicOperation,
    /// A sampler state's `lodBias` that the device's samplers cannot add as it is, where the GPU
    /// the device presents adds it to every sample's level
    /// ([`SamplerUse::lod_bias`](crate::metal::sampler::SamplerUse::lod_bias), as a Mac with an
    /// Intel or AMD GPU does): past the device's `maxSamplerLodBias` (an infinity included) its
    /// samplers add that limit, of the bias's sign; a NaN bias, or one other than 0 where the
    /// device's portability subset withholds `samplerMipLodBias` (MoltenVK), they add none
    /// ([`DeviceLodBias`](crate::metal::sampler::DeviceLodBias)). The sampler state carries it,
    /// fixed by its bias and the device's limit
    /// ([`DeviceLodBias::note`](crate::metal::sampler::DeviceLodBias::note)); no pipeline stage
    /// records it. **Residual:** a sample whose level with what the device adds has not reached
    /// the last level the sampler reads (a positive bias) or level 0 (a negative one) reads
    /// another level than Metal's, which adds the whole bias and then clamps the level into the
    /// texture's levels; what Metal samples for a NaN bias is not measured.
    ///
    /// Validation derives no tolerance from it.
    CappedLodBias,
    /// The stage holds a tangled instruction (a simdgroup or quad operation, or a barrier whose
    /// lanes depend on reconvergence) on a device without maximal reconvergence
    /// (`Reconvergence::ImplementationDefined`): after divergent control flow the device decides
    /// which lanes run it together, where Metal's lanes reconverge maximally, so such an operation
    /// may see other lanes active than Metal's. Under subgroup-uniform control flow
    /// (`shaderSubgroupUniformControlFlow`) the lanes reconverge where control flow is uniform.
    /// Static per stage: recorded for any tangled instruction in the entry, as the stage's
    /// `reconvergence` is.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceReconvergence,
    /// The kernel holds two accesses of threadgroup or device memory, at least one a write, that
    /// different threads of one SIMD-group may make to the same bytes between two barriers, and
    /// no barrier or atomic orders them: Metal's answer there comes from Apple's GPUs running a
    /// SIMD-group's 32 threads in lockstep, each instruction completing for every active thread
    /// before the next starts, so such a pair runs in program order there (a lane reads the slot
    /// another lane of its SIMD-group wrote one instruction earlier). The Metal Shading Language
    /// orders threads' memory accesses only through barriers and atomics and calls such a pair a
    /// data race. Under the identity domains (`metal2vulkan-runtime`) the module keeps the pair as
    /// the AIR states it, so the device runs the threads in its own order between barriers, and the
    /// values the pair reads and leaves may be other than Metal's. Static per kernel, from the
    /// AIR's structure alone: recorded on every device for a kernel in which the translator finds
    /// such a pair. A profile that orders the pairs with barriers of its own (the resident-table
    /// domains) records none.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    SimdgroupAccessOrder,
    /// The render pipeline draws into a single-sampled color attachment of a signed normalized
    /// format (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`).
    /// Metal's pass, once it draws, writes each 32 by 32 tile a draw reaches back through its
    /// channels' values, so a channel's minimum code (-128, or -32768), which reads as -1.0 as the
    /// code above it does, is stored as that code (-127, or -32767), in the texels no fragment
    /// writes too (measured on an Apple M3 Max under macOS 27.0.1: every such code of each tile a
    /// draw reaches, none in other tiles, none in a pass that draws nothing). The device's pass
    /// keeps the bytes its draws do not write. Static per pipeline: recorded on the fragment stage
    /// (the vertex stage where the pipeline has none) of every such pipeline, whatever its passes
    /// load.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    SnormMinimumKept,
    /// A draw of triangles under a depth bias (`setDepthBias:slopeScale:clamp:` with a bias or a
    /// slope scale other than 0): the slope measure, and a fixed-point attachment's unit, are the
    /// device's; Apple states neither. Vulkan biases by `m × slopeScale + r × depthBias` ("Depth
    /// Bias"), where the maximum depth slope `m` is the device's (exact, or the larger of the two
    /// derivatives' magnitudes), and the unit `r` is `2^(e - n)` of the primitive's largest depth
    /// exponent for a float depth attachment and the device's for a fixed-point one (at most
    /// `2 × 2^-n`); a biased depth past the viewport's depth range is undefined there ("Depth
    /// Clamping and Range Adjustment"). Static per operation: no pipeline stage records it;
    /// `metal2vulkan-runtime`'s render encoder carries it (`RenderCommandEncoder::DEPTH_BIAS_NOTE`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceDepthBias,
    /// A depth test or depth write of a 32-bit float depth (a depth-stencil state that compares by
    /// other than `Always` or writes): a denormal depth is the device's. Every driver measured but
    /// MoltenVK on an Apple M4 compares a denormal depth otherwise than Metal, as zero (a positive
    /// one) or below zero (a negative one), where Metal compares a positive one above zero and a
    /// negative one not below it; and RADV and ANV store a denormal a GPU write gives a depth image
    /// as a zero, where Metal keeps it. Static per operation: no pipeline stage records it;
    /// `metal2vulkan-runtime`'s depth-stencil state carries it
    /// (`DepthStencilState::precision_note`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DepthDenormals,
    /// `generateMipmapsForTexture:` of a filterable colour format: each level below the first is
    /// the device's linear filter of the one above, its blit (`vkCmdBlitImage` with
    /// `VK_FILTER_LINEAR`, the level above sampled at each texel's centre scaled into it). Apple's
    /// documentation states no filter. Metal filters bilinearly at the same scaled centres,
    /// measured on an M3 Max and an M4, but by Apple's datapath, whose answers differ between those
    /// GPUs by up to a unit in the last place. Where a level halves an even size, each sample falls
    /// between two texels on the axis, so the filter is a box of them at the device's filter
    /// precision; an sRGB format is filtered in linear space. Static per operation: no pipeline
    /// stage records it; `metal2vulkan-runtime`'s blit encoder carries it
    /// (`BlitCommandEncoder::GENERATE_MIPMAPS_NOTE`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceFilteredMipmaps,
    /// A multisample color resolve by Vulkan's `AVERAGE` mode, which leaves the rounding of the
    /// average, and the sRGB conversion, to the device: Metal rounds a normalized channel's average
    /// to nearest with ties up and averages an sRGB channel in linear space by its own 12-bit
    /// tables, where a device may round a tie down and Vulkan only recommends the linear average
    /// for an sRGB format; a float channel's average may differ in its last bits, and a NaN or
    /// infinite sample's average is the device's. Measured against Metal's rules at 2 and 4
    /// samples:
    /// on RADV, ANV, NVIDIA and lavapipe every unorm, snorm and packed 16-bit unorm channel within
    /// one code of Metal's, an sRGB channel within two. Static per operation: no pipeline stage
    /// records it; `metal2vulkan-runtime`'s render pass carries it for every color attachment it
    /// resolves (`RenderPassDescriptor::precision_note`), on every device.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceRoundedResolve,
    /// A texture, or a view, whose own pixel format is block-compressed (BC1 to BC7, and the ASTC,
    /// ETC2 and EAC formats where the device makes them): a read, sample or gather of its texels
    /// answers the device's decode of its blocks (an sRGB form's code through the device's sRGB
    /// decode), where Metal interpolates a block's endpoints with 6-bit weights. Its blocks'
    /// bytes, copies and sizes are Metal's. Static per texture, decided by its pixel format alone
    /// on every device: no pipeline stage records it; `metal2vulkan-runtime`'s texture carries it
    /// (`Texture::precision_note`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    CompressedTexelDecode,
    /// The render pipeline draws into a single-sampled `RGB9E5Float` color attachment. Metal's
    /// pass, once it draws, writes each 32 by 32 tile a draw reaches back through the texels'
    /// values, so a word whose largest mantissa is below 256 under an exponent above 0 is stored
    /// normalized, the same value under a smaller exponent, in the texels no fragment writes too
    /// (measured on an Apple M3 Max under macOS 27.0.1). The device's pass keeps the words its
    /// draws do not write. Static per pipeline, as [`PrecisionNote::SnormMinimumKept`] is:
    /// recorded on the fragment stage (the vertex stage where the pipeline has none) of every such
    /// pipeline, whatever its passes load.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    SharedExponentKept,
    /// A sampler, a sampler state or a constexpr sampler, that reads its border (`ClampToZero` or
    /// `ClampToBorderColor` on an axis, both of which read the sampler's colour) with an
    /// opaque-black border colour: its border texel sampled through a view whose Vulkan component
    /// mapping is not the identity (the texture's swizzle, or the mapping its pixel format takes)
    /// is the device's answer on a device without `borderColorSwizzleFromImage`, not Metal's,
    /// which is the border colour in the format's channels and then the swizzle. Vulkan leaves
    /// that texel undefined there unless the sampler names the view's mapping
    /// (`VkSamplerBorderColorComponentMappingCreateInfoEXT`), which a sampler that meets every view
    /// cannot; the NVIDIA driver reads `(0, 0, 0, 1)`. Where the device has the feature, the texel
    /// is read through the view's mapping, as Metal's is. Static, decided on every device by the
    /// sampler alone: no translated stage records it; on `metal2vulkan-runtime` a sampler state carries
    /// it by its descriptor (`SamplerState::precision_note`), and a pipeline whose function samples
    /// through such a constexpr sampler carries it among its notes by the sampler's constants
    /// (`precision_notes`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    SwizzledOpaqueBlackBorder,
    /// A draw of lines (a line or a line strip, or a triangle's edges under the `Lines` fill mode)
    /// covers the pixels the device's line rule gives, on every device. Metal covers the pixels
    /// whose diamond a line exits. Vulkan's Bresenham mode, which draws lines elsewhere, lets an
    /// implementation's fragment lie up to one pixel in x or y from the diamond-exit rule's, and
    /// its fragment count differ by one (the Vulkan specification, "Bresenham Line Segment
    /// Rasterization"), and its diamond is open, so no Vulkan control decides the pixel of a line
    /// that only touches a diamond's vertex: where a line of the conformance case
    /// `primitive-type-restart-indices` runs through the boundary point (3.5, 1.0), Metal on an
    /// Apple M3 Max covers the lower pixel and NVIDIA's driver the upper. A state that enables alpha to coverage or alpha to one draws its lines by the device's
    /// default rule instead, as Vulkan's Bresenham mode takes neither
    /// (`VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766`): a device whose
    /// `strictLines` is set (lavapipe, NVIDIA's driver) covers the rectangle of the line's width,
    /// so the pixels at a line's two ends, and on either side of a pixel boundary the line runs
    /// along, may differ from Metal's. Static, of the draw: no translated stage records it; on
    /// `metal2vulkan-runtime` a draw of lines carries it (`RenderCommandEncoder::LINE_NOTE`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceLineRasterization,
    /// A fragment function that writes memory (`StageInterface::writes_memory`) in a render
    /// pipeline of more than one sample runs as many times per pixel as the device runs it, each
    /// run reading the pixel's whole coverage as `[[sample_mask]]` (a function that loops over its
    /// covered samples, `CoverageLoop::Looped`, runs its loop in each). Documented Metal names no
    /// run count for it: MSL's section 5.5 names only sample-qualified inputs, `[[sample_id]]` and
    /// `[[color(m)]]` as making a function run per sample. Apple's GPUs were measured to run such a
    /// function once per covered sample (an M3 Max and an M4). Vulkan runs a fragment shader
    /// without sample shading at least once per pixel, and more where the device chooses ("Multiple
    /// fragment shader invocations may be executed for the same fragment for any number of
    /// implementation-dependent reasons", vkspec 35.5). So its writes repeat as often as the
    /// device's runs, and whatever the function derives from its own writes across runs follows
    /// the device's count too: its stored colours and its `[[sample_mask]]` output included, not
    /// only memory. Static, decided by the pipeline's sample count and the function's writes: the
    /// pipeline's compile records it on its fragment stage.
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceFragmentRuns,
    /// A `Boolean` or `Counting` visibility result (`setVisibilityResultMode:offset:`): the count
    /// of samples that passed is the device's precise occlusion query's
    /// (`VK_QUERY_CONTROL_PRECISE_BIT`), where Metal documents "the number of samples that passed
    /// depth and stencil tests". Vulkan lets devices count otherwise ("Occlusion Queries"):
    /// "Setting VK_QUERY_CONTROL_PRECISE_BIT does not guarantee that different implementations
    /// return the same number of samples in an occlusion query. Some implementations may kill
    /// fragments in the pre-rasterization shader stage, and these killed fragments do not
    /// contribute to the final result of the query", and "Some implementations may track the
    /// number of samples that pass the per-fragment tests multiplied for each color output"; and
    /// on a device whose `earlyFragmentMultisampleCoverageAfterSampleCounting` is false, a fragment
    /// function with early fragment tests may have its shading and its coverage "performed before
    /// sample counting" ("Fragment Operations"). A `Boolean` result is the same count, nonzero
    /// exactly where the count is. Static per operation: no pipeline stage records it;
    /// `metal2vulkan-runtime`'s render encoder carries it
    /// (`RenderCommandEncoder::VISIBILITY_NOTE`).
    ///
    /// Validation derives no tolerance from it: a difference stays a mismatch, which the note
    /// explains.
    DeviceOcclusionCount,
    /// A vertex function runs a simdgroup or quad operation (a reduction, scan, vote, ballot,
    /// shuffle, broadcast or AGX multiply-add), on every device. Metal's shading language defines
    /// its SIMD-group and quad-group functions for kernel and fragment functions (MSL 4.1 sections
    /// 6.10.2 and 6.10.3), and no vertex input names a lane (Table 5.2): in a vertex function the
    /// groups are the device's own. The operation runs over the vertices the device packs into a
    /// subgroup, in the device's lane order, so a sum, a vote or a shuffle can combine other
    /// vertices than on the M3 Max, whose answers group 32 consecutive vertices. Vulkan promises
    /// full subgroups only to stages with workgroups, and nothing pins which vertices share one:
    /// measured on an Iris Xe (RPL-P, Mesa 26.2.6), whose vertex `SubgroupSize` is 32, each
    /// subgroup of 64 points holds 8 of them, the even or the odd vertices of a run of 16, and the
    /// other 24 lanes take no part.
    ///
    /// Validation derives no tolerance from it: a difference stays a noted difference for review.
    VertexSimdgroupGrouping,
}

/// Which lowering strategies a module used, with their word costs.
///
/// Keys name `Strategy` variants (`Family::Variant`) for reporting only; translation never reads
/// them. One key names no variant: `texture.is_null.absent_arm` holds the words of a nil texture's
/// arm (the blocks the default presence specialization, every static texture present, never
/// reaches), and its uses count those instructions, which also count as uses of their own
/// strategies (whose words leave them).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrategyCensus {
    pub entries: BTreeMap<String, StrategyUse>,
}

/// How often a strategy was selected and the words it emitted.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrategyUse {
    pub uses: u32,
    pub words: u64,
}
