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
use crate::pipeline::interface::PipelineInterface;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::sym::{SlotRef, SpecKey};
use crate::pipeline::vertex::VertexInputState;
use crate::vk;

/// A compiled render pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderPipeline {
    pub key: PipelineKey,
    /// The fingerprint of the profile it was compiled for: a cache load checks it.
    pub profile: Hash256,
    /// Vertex, then fragment when rasterization is enabled. A post-tessellation vertex
    /// function's pipeline has the synthesized vertex and tessellation control stages, then its
    /// evaluation stage, then the fragment stage. A pipeline whose fragment function reads
    /// `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric`, or whose vertex
    /// function selects its slice or viewport on a device without the output, has three
    /// synthesized geometry stages (one set for every such need) before its fragment stage, or
    /// last without one, for the point, line and triangle topology classes in that order: a
    /// `VkPipeline` runs only the one of its draw's class ([`runs_stage`](Self::runs_stage)).
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
    /// Whether a `VkPipeline` drawing `topology` runs stage `at` of
    /// [`stages`](Self::stages): every stage but a geometry stage of another topology class (a
    /// pipeline's geometry stages are the point, line and triangle classes' in that order). A
    /// stage past the list runs nothing.
    pub fn runs_stage(&self, at: usize, topology: vk::PrimitiveTopology) -> bool {
        use vk::PrimitiveTopology as T;
        let Some(stage) = self.stages.get(at) else {
            return false;
        };
        if stage.stage != vk::ShaderStage::GEOMETRY {
            return true;
        }
        let class = match topology {
            T::POINT_LIST => 0,
            T::LINE_LIST | T::LINE_STRIP => 1,
            T::TRIANGLE_LIST | T::TRIANGLE_STRIP | T::TRIANGLE_FAN => 2,
            _ => return false,
        };
        let nth = self.stages[..at]
            .iter()
            .filter(|s| s.stage == vk::ShaderStage::GEOMETRY)
            .count();
        nth == class
    }

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
        + vector_bytes::<u32>(
            stage.rounded_colors.capacity()
                + stage.own_pixel_textures.capacity()
                + stage.snapshot_textures.capacity(),
        )
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
    /// specialization constants (the bound format's class, `SpecKey::TexelFormat`; the bound
    /// sampler's word, `SpecKey::SamplerState`), with the condition under which a specialized
    /// module runs such an arm. Sorted by note, each note at most once. Empty for a module whose
    /// every note holds under every specialization. [`StageModule::precision_notes_under`]
    /// gives the notes of the module as a pipeline specializes it.
    pub specialized_notes: Vec<SpecializedNote>,
    /// The texture slots this fragment stage reads only at the fragment's own pixel, sorted
    /// (`StageInterface::own_pixel_textures`): a draw's reads of such a slot lie inside the
    /// rectangle bounding its scissors, a pixel wider each way for its helper invocations, on a
    /// framebuffer below `2^24` pixels a side. Empty for every other stage.
    pub own_pixel_textures: Vec<u32>,
    /// The texture slots of `own_pixel_textures` a draw can read through a snapshot an earlier
    /// draw of the same primitives took (`StageInterface::snapshot_textures`), sorted. Empty for
    /// every other stage.
    pub snapshot_textures: Vec<u32>,
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

    /// The stage's precision notes where a pipeline creates it under `specialization`: every note
    /// of [`precision_notes`](Self::precision_notes) that no
    /// [`specialized_notes`](Self::specialized_notes) entry names, and each specialized note
    /// one of whose clauses holds under the key's values. A constant the key does not give
    /// reads its default ([`SpecializationLayout`]), so the empty key gives the notes of the
    /// default specialization. Sorted, each note once. `precision_notes` itself is the answer
    /// where no specialization is known.
    pub fn precision_notes_under(&self, specialization: &SpecializationKey) -> Vec<PrecisionNote> {
        self.precision_notes_bound(specialization, &|_| None)
    }

    /// The stage's precision notes where a pipeline creates it under `specialization` and a draw
    /// or dispatch binds the values `bound` gives: as
    /// [`precision_notes_under`](Self::precision_notes_under), where a test of a key the stage
    /// reads as data (`PushField::Specialized`), which no pipeline's key carries, reads the value
    /// `bound` gives for it, the one the binder pushes; a test of a key neither gives holds.
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

    /// The clauses of `note`'s [`specialized_notes`](Self::specialized_notes) entry that hold
    /// where a pipeline creates the stage under `specialization`, as
    /// [`precision_notes_under`](Self::precision_notes_under) reads them: the bound formats and
    /// samplers, by slot, that leave the note on an arm that is not Metal's. Empty where no
    /// entry names `note` or none of its clauses holds.
    pub fn note_clauses_under(
        &self,
        note: PrecisionNote,
        specialization: &SpecializationKey,
    ) -> Vec<&SpecClause> {
        self.note_clauses_bound(note, specialization, &|_| None)
    }

    /// The clauses of `note`'s [`specialized_notes`](Self::specialized_notes) entry that hold
    /// where a pipeline creates the stage under `specialization` and a draw or dispatch binds
    /// the values `bound` gives, as [`precision_notes_bound`](Self::precision_notes_bound) reads
    /// them.
    pub fn note_clauses_bound(
        &self,
        note: PrecisionNote,
        specialization: &SpecializationKey,
        bound: &dyn Fn(SpecKey) -> Option<u64>,
    ) -> Vec<&SpecClause> {
        self.specialized_notes
            .iter()
            .filter(|s| s.note == note)
            .flat_map(|s| &s.when)
            .filter(|clause| self.clause_holds(clause, specialization, bound))
            .collect()
    }

    /// The clauses of `note`'s [`specialized_notes`](Self::specialized_notes) entry that test
    /// only the formats bound at texture slots ([`SpecClause::tests_bound_formats`]) and hold
    /// where `bound` gives the class of the format bound at each slot they name
    /// (`metal::texel_format::texel_format`), or `None` where it is not known, of which such a
    /// test holds. A runtime reads them at a draw or dispatch, from the textures it binds: the
    /// key of its pipeline carries no such class. Empty where no entry names `note` or none of
    /// those clauses holds.
    pub fn bound_format_clauses(
        &self,
        note: PrecisionNote,
        bound: &dyn Fn(SlotRef) -> Option<u32>,
    ) -> Vec<&SpecClause> {
        self.specialized_notes
            .iter()
            .filter(|s| s.note == note)
            .flat_map(|s| &s.when)
            .filter(|clause| clause.tests_bound_formats() && clause.holds_of_bound(bound))
            .collect()
    }

    /// Whether every test of `clause` holds under `specialization`, a constant the key does not
    /// give reading its default, and a key the stage reads as data reading the value `bound`
    /// gives.
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
            // `SpecKey::BoundTexelFormat`, through `bound_format_clauses`, and the keys the stage
            // reads as data through `bound`).
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

impl SpecClause {
    /// Whether every test reads the class of the format bound at a texture slot
    /// (`SpecKey::BoundTexelFormat`), which no pipeline's key carries: a clause a runtime reads
    /// at a draw or dispatch, from the textures it binds
    /// ([`StageModule::bound_format_clauses`]).
    pub fn tests_bound_formats(&self) -> bool {
        self.all
            .iter()
            .all(|t| matches!(t.key, SpecKey::BoundTexelFormat(_)))
    }

    /// Whether a test reads a value the function reads as data (`SpecKey::read_as_data`), which
    /// no pipeline's key carries: a clause a runtime reads at each draw or dispatch, from the
    /// values its plan pushes ([`StageModule::note_clauses_bound`]).
    pub fn tests_bound_values(&self) -> bool {
        self.all.iter().any(|t| t.key.read_as_data())
    }

    /// Whether every test of the bound formats holds where `bound` gives the class of the format
    /// bound at each slot (`None` where it is not known, of which the test holds); a test of any
    /// other key holds here.
    pub(crate) fn holds_of_bound(&self, bound: &dyn Fn(SlotRef) -> Option<u32>) -> bool {
        self.all.iter().all(|t| match t.key {
            SpecKey::BoundTexelFormat(slot) => {
                bound(slot).is_none_or(|v| (v & t.mask == t.bits) == t.equal)
            }
            _ => true,
        })
    }
}

/// A test of one specialization constant's value `v`: it holds when `(v & mask == bits) == equal`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecTest {
    /// The constant: a texture slot's `SpecKey::TexelFormat` or a sampler slot's
    /// `SpecKey::SamplerState`, as the stage's [`SpecializationLayout`] declares it.
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
    /// largest threadgroup of even sides within those limits, as Metal rounds it.
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
    /// Bytes of statically sized threadgroup memory. A kernel whose simdgroups are emulated
    /// also counts the scratch of `max_total_threads_per_threadgroup` threads and the declared
    /// arena's minimum, so any `[[threadgroup(n)]]` lengths that fit beside it fit.
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
    /// scale by `2^B` through every sampler, and, on a device that does not sample through Metal's
    /// own sampler, a bias whose sum with the sampler's own may lie beyond the device's
    /// `maxSamplerLodBias`, which Vulkan clamps, or that an anisotropic sampler applies to its
    /// footprint's minor axis alone; and, on a device without `maintenance8`, a fragment's
    /// implicit-level sample with a dynamic texel offset, whose derivatives are taken before the
    /// switch over the offset's values (`lower::sampler::OffsetArms`). The gradients carry Metal's
    /// own reading of the bias, `B`; `Exp2`'s precision and the device's level arithmetic are the device's. A device that samples
    /// through Metal's own sampler (MoltenVK) records it only for a constexpr sampler with a bias
    /// of its own and for hoisted derivatives, except those of a 2D color sample without a bias, a
    /// `min_lod_clamp` or an offset through samplers without a bias of their own, which there are
    /// Metal's own sample.
    GradFromBias,
    /// Transcendentals answer within Apple's stated bound, not Metal's bits: the device's own
    /// functions where they are Metal's (MoltenVK), and the device's precision for a call no
    /// other note bounds. Not recorded by precise `fmod`, whose answer is the one right value on
    /// every device, nor, on a device whose `float` functions are Metal's own, in an entry Metal
    /// compiles with fast math ([`EntryMath::Fast`]) by a call the device answers as Metal's own
    /// function; there its fast `sinpi`, `cospi`, and `tanpi`, near-equivalents, still record it.
    DeviceTranscendentals,
    /// Float reductions use the device's association order.
    ReductionOrder,
    /// `float` denormals follow the device: Metal reads an `f32` denormal operand of its float
    /// arithmetic, comparisons, and functions as the zero of its sign and flushes a denormal result
    /// to one, where a Vulkan device keeps or flushes them by its own float controls (this
    /// translation declares no `DenormFlushToZero`). Recorded where the module leaves an
    /// operation's denormals to a device not measured to read them as Metal does: `+ - *`, the
    /// comparisons, `fma`, `dot`, `sqrt`, `rsqrt`, `trunc`, `rint` and `ldexp` on `float` (by the
    /// device's products) but on a device that takes
    /// the measured `DriverDeviation::MetalFloatDenormals`; and on every device the divisions and
    /// the remainder, `floor`, `ceil` and `fneg`, the device's maths functions (`fast::fmod`'s
    /// fused multiply-add, `fabs`, `sign` and `fract` among them), `mix` by `FMix`, `fast::sqrt`
    /// by `Sqrt`, `dot` of vectors of other than two to four lanes, whose chunks' products the
    /// device sums where Metal flushes each one below `2^-126`, beside their own notes, and a
    /// conversion to or from `double`, whose reading Metal's is not measured. The operations that
    /// read a denormal on its bits answer Metal on every device: `copysign`; and so do the
    /// functions the device answers as Metal's own. Where the minimum and maximum functions
    /// (`llvm.minnum` and `llvm.maxnum` among them, with any flags) and `saturate` are the device's
    /// `NMin`, `NMax` and `NClamp`, their pick of a denormal is
    /// [`PrecisionNote::NativeMinMax`]'s. Validation derives no tolerance and no explanation from
    /// it: every `float` computation is exposed to it.
    DeviceDenormals,
    /// A pixel-coordinate sampler's coordinate reaches the device's sampler divided by the
    /// texture's level-0 size in the shader (Vulkan's unnormalized samplers take no gather,
    /// offset, comparison, array, 3D image or level of detail), and the device's sampler reads its
    /// texels from that coordinate in its own arithmetic: the position it reads, `(x / N) N` in
    /// the device's rounding and its sub-texel precision, lies within a few units in the last place
    /// of the coordinate `x` as given, where Metal reads `x` itself. So a sample or gather reads
    /// Metal's texels and weights but where `x` lies that close to a texel edge (a nearest
    /// sample's texel, a gather's 2x2 block, a linear filter's pair of texels) or to the edge of
    /// one of the device's sub-texel steps (a linear filter's weight), where it may read the
    /// neighbouring texel, block or step. Measured: the device answers as its normalized sampler at
    /// a coordinate within one `f32` ulp of the quotient `x / N` on MoltenVK, and at the quotient
    /// itself on lavapipe (800 coordinates at and around each texel edge of a 5x3 texture, every
    /// filter and address mode); a linear downscale by 8 of textures 1000 to 1920 texels wide is
    /// within 1/255 of Metal's on MoltenVK and 2/255 on lavapipe (a weight one 1/256 step off on
    /// each axis); other devices unmeasured. Recorded on every device for every access through a
    /// constexpr pixel sampler but its exact fetch (nearest, clamped to the edge,
    /// `sampler.pixel_fetch`), and for every access through a runtime sampler slot, held where the
    /// bound sampler states pixel coordinates. A cube's pixel sampler is a normalized one. An
    /// argument buffer's sampler records [`PrecisionNote::ArgumentPixelSamplerScaling`] instead.
    ///
    /// Validation derives no tolerance from it: at a texel edge the value is the neighbouring
    /// texel's, whatever it holds.
    PixelSamplerScaling,
    /// Float division, remainder, square root, fused multiply-add, and dot products use Vulkan's
    /// precision bounds (e.g. 2.5 ULP division, a possibly unfused `Fma`) rather than correct
    /// rounding. A fast (`arcp`) division is the device's `OpFDiv`, where Metal's is `x` times a
    /// reciprocal it rounds within about 0.64 ulp but not correctly. Not recorded by precise
    /// `fmod`, whose answer is the one right value on every device, nor, on a device whose `float`
    /// functions are Metal's own (MoltenVK), in an entry Metal compiles with fast math
    /// ([`EntryMath::Fast`]), by a fast division, which is Metal's own there, or by a call the
    /// device answers as Metal's own function.
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
    /// is 0 because the consumer passes 0 (`bind::MetalInstance::user_id`), a choice: what Metal
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
    /// fused on one side alone. On a device without `shaderFloat16`, under
    /// `HalfChainPolicy::AtExits` (the default), a chain of such `half` additions, subtractions and
    /// products is held at `float` precision between the roundings where its values leave it
    /// (`lower::arith::half_chain`): the result may differ from the rounded operations' by the
    /// roundings skipped, and a value inside the chain past `half`'s range stays finite.
    /// Recorded on every device.
    FastMathRewrites,
    /// A tessellation pipeline's vertex order: Metal's clockwise output winding is declared
    /// `VertexOrderCcw` and its counter-clockwise one `VertexOrderCw`, as MoltenVK maps them under
    /// Vulkan's default upper-left domain origin. Not read on Metal: the other mapping reverses the
    /// facing of every generated triangle.
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
    /// within the fragment area being shaded"), a centroid varying (anywhere in the fragment's
    /// covered area, where Metal, measured on Apple M3 Max, moves it to a covered sample's location
    /// only under partial coverage), `[[point_coord]]` ("the fragment center, or the location of a
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
    /// A render pass drawn by regions over a rasterization-rate map that is not the identity
    /// (`bind::RateMapRegions::precision`): its regions' lines are this crate's tiles' exact lines,
    /// where Apple's are its fixed-point ones, so a fragment's evaluated point lies within the
    /// bound of Metal's. A module that decodes a map records none: its decoder is Apple's GPU
    /// arithmetic over Metal's parameter data, bit for bit (it once recorded this note, when the
    /// data was this crate's own layout).
    RasterizationRateMap,
    /// Metal's bicubic filter, emulated by fetches with Metal's weights, the 8- and 16-bit
    /// normalized and the extended-range formats' exact sum rounded half up onto the format's grid
    /// (an extended-range one clamped into the format's decoded endpoints), and the 32-bit float
    /// and half-float formats' sums through Metal's accumulators (a 1D sum and a 32-bit float
    /// format's 2D bilinear one through its linear accumulator, a half format's 2D Catmull-Rom sum
    /// through its two-pass one, an axis whose four taps clamp onto one texel weighed as that texel
    /// alone; every lane of the P2d probe and of the float fixtures, of the held-out P2e and P2f
    /// probes, and of the 16 half and small-float cases), differs from an Apple M3 Max in what is
    /// not yet pinned:
    /// - A 2D normalized sum within about `2^-21` of the format's range (`1` unsigned, `2`
    ///   signed) of a rounding tie: Metal may round it to the other side, one grid step away
    ///   (`1/65535` or `1/32767` for 16-bit, `1/4080` or `1/2032` for 8-bit). No model of Metal's
    ///   2D datapath tried fits; its 1D sums are exact.
    /// - An extended-range texture's 2D sum (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their
    ///   `_sRGB` twins, read through the slot's `extended_range` variant) within 0.014 of a grid
    ///   step of a rounding tie: Metal may round it to the other side, one step away (`1/8160` a
    ///   color lane, `1/8192` an `_sRGB` color lane, `1/4080` alpha); 4,050 of the 3,317,760 lanes
    ///   measured (every one at most 0.013968 of a step from a tie), every other lane exact. No
    ///   model of Metal's datapath tried fits.
    /// - An sRGB sum: Metal decodes each tap's code through its twelve-bit table and sums the
    ///   steps exactly; the emulation sums the device's decoded taps in `f32`, the device's decode
    ///   being the only form where it samples the format ([`PrecisionNote::SrgbTexelDecode`]). A
    ///   texture slot bound in an 8-bit sRGB format the device holds as its UNORM twin decodes
    ///   each tap's codes under its `SrgbTexture` variant by the device's transfer
    ///   ([`PrecisionNote::SrgbTexelDecode`]) and sums them exactly on the grids `1/4095` (color,
    ///   the step nearest each tap) and `1/(16 · 255)` (alpha), so only the first bullet's 2D
    ///   residue applies to that sum. The packed normalized formats and `Depth16Unorm` sum in `f32` (their exact
    ///   rule is not measured).
    ///
    /// The footprint, the fraction, the addressing of each tap, the filter each format selects,
    /// the weights, each format's clamp and rounding, and the non-finite results are Metal's, but
    /// for inferences: the signed clamp's lower bound, the treatment of the formats no case reads
    /// (`RG8UnormSRGB`, and the exact rule's `RG8Unorm`, `RG8Snorm`, `RG16Unorm`, `BGRA8Unorm`,
    /// and `A8Unorm`), the sums of `Depth32Float` (the 32-bit float formats') and the small
    /// floats' non-finite results (`RG11B10Float`, `RGB9E5Float`: the half formats', whose
    /// finite 2D sums are measured), taken by family, a 1D half sum's non-finite results, and two
    /// 32-bit float sub-rules no lane tells apart: whether a denormal tap is flushed first (the
    /// emulation keeps it) and then the sign of a zero result (the emulation gives `+0`). The
    /// footprint is exact for every coordinate within `[-2^23 + 1/2, 2^23]` texels, where its 1/256
    /// step fits an `i32`. Past that range a coordinate below `2^120` texels saturates it, the
    /// translation's choice. A non-finite or larger coordinate gives an unspecified footprint whose
    /// taps stay in bounds. Metal's value is not measured for either.
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
    /// Metal does not (Vulkan's linear 2x2 footprint does, measured on lavapipe), and its lanes
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
    /// or dispatch that binds such a texture.
    /// Also recorded on a render pipeline's vertex stage where its vertex descriptor gives a float
    /// format narrower than 32 bits (a `half` format, `FloatRG11B10`) to an attribute that
    /// fixed-function vertex input fetches and the function reads as `float`
    /// (`pipeline::link::vertex`): Metal widens its NaN to `0x7fc00000`, and the device's
    /// widening need not. A pulled attribute's decode records it in the module (a vertex
    /// function's, a kernel's `[[stage_in]]` attribute's, a post-tessellation function's). Also
    /// recorded by a
    /// 16-bit float attachment's blend the function computes in the storage form: a NaN of the
    /// blend is the device's arithmetic's, which the storage image's store converts, where Metal
    /// stores every NaN of the blend as `0x7e00`. Also recorded by a `float` read of a heap's texture buffer on a device without
    /// Metal's sampler: Metal hands back a half-float texel's NaN as `0x7fc00000`, and the handle
    /// carries no class to convert it by. The runtime announces it too for a GPU blit through an
    /// `RG11B10Float` texture Metal keeps losslessly compressed: Metal's compression makes a NaN
    /// field the quiet NaN (`0x7e0`, `0x3f0`), and the device's copy keeps its bits. And for a
    /// plain color fill to a NaN of a half or small-float texture the device's clear writes: the
    /// device converts the NaN as it likes, where Metal's fill keeps a half NaN's sign and payload
    /// and makes a small float's all ones. Validation derives no tolerance and no explanation from
    /// it, as for `DeviceDenormals`.
    DeviceNanEncoding,
    /// The device compares a depth texture's texels (a sample or a gather with comparison) on a
    /// device without `VK_EXT_depth_range_unrestricted` enabled whose driver is not measured to
    /// keep depth values outside `[0, 1]` without it (narrowed to comparisons). Metal's
    /// `Depth32Float` texels hold any `float`, and the device's comparison reads them from the
    /// texture's depth image. Vulkan leaves a value outside `[0, 1]` uploaded there undefined
    /// without the extension (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and a device may store
    /// it clamped (RADV on a Radeon 780M and ANV read 1000.0 back as 1.0, measured) or keep it
    /// (MoltenVK and lavapipe keep it, measured), so a comparison whose texel and reference both
    /// lie outside `[0, 1]` on the same side can differ. Values within `[0, 1]`, and every texel on
    /// a device with the extension enabled or a driver measured to keep such values without it
    /// (`VkProperty::DepthImageKeepsValuesOutsideUnitRange`, which such a module requires instead),
    /// are exact. A function that reads, samples or gathers a depth texture without comparing
    /// records nothing: its interface's `TextureSlot::access` states the use, and a consumer that
    /// uploads a value into a depth image its device may clamp announces that upload (docs/API.md
    /// section 5).
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
    /// Metal's level and taps (`PrecisionNote::AnisotropicFilter`). A driver whose route is not
    /// measured takes Vulkan's rule's form (`CubeGradientRoute::Vulkan`), its reading of it the
    /// device's own, recorded here.
    ///
    /// Through MoltenVK's graphics route under macOS 27 (`CubeGradientRoute::AppleProjected`),
    /// SPIRV-Cross's `spvGradientCube` hands Metal an `x` or `z` face's gradients as Metal's own,
    /// exactly, but writes a `y` face's `x` tangent on both of its tangents, so its `z` tangent is
    /// lost: the module hands a `y` face Metal's length of its two tangents, which Metal reads on
    /// both, an isotropic sample's footprint kept up to one rounding of `64/91` (measured: every
    /// lane Metal's) and an anisotropic one's ellipse laid on the face's diagonal. Sampling a `y`
    /// face through a 2D-array view of the cube with the face gradients would give Metal's own
    /// anisotropic sample on every lane whose filter stays inside the face (measured), and differ
    /// less than the fixup where it crosses the face's edge; the binder does not offer that view
    /// yet.
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
    /// Native device sampling (a texel conversion): off Apple's GPUs (without
    /// `VkProperty::MetalSampler`) every access with a `half` result records this note. The
    /// recorded bound: the device's `float` texel (within
    /// [`PrecisionNote::DeviceTexelConversion`]'s bound of Metal's) rounded to nearest even is
    /// within one `half` ulp of Metal's `half` result (Metal rounds a 32-bit float sample toward
    /// zero, a half-float linear sum with ties away from zero, and a normalized value onto its grid
    /// first), but for a 32-bit float texel's sample past `±65504`, which Metal saturates to
    /// `±65504` and the rounding takes to the infinity of its sign; a NaN is the device's decode
    /// ([`PrecisionNote::DeviceNanEncoding`]).
    ///
    /// On a device that samples through Metal's own sampler the `f32` texel is Metal's `float`
    /// result, and its nearest even `half` is Metal's `half` but for the classes the note is
    /// recorded under, by the slot's bound format (`SpecKey::TexelFormat`): a 32-bit float
    /// texel's sample, which Metal rounds toward zero (within one `half` ulp, but past `±65504` the
    /// infinity of its sign where Metal saturates to `±65504`); a 16-bit normalized texel, whose
    /// `float` steps `65455 / 65535`, `65519 / 65535` and `32759 / 32767` round twice (one `half`
    /// ulp); and a format whose conversion is not measured (unbounded). A heap element and a depth
    /// texture record it wherever they are accessed. A two-plane YCbCr texture on the grid `8192`
    /// (the `_sRGB` twins, the unpacked 12-bit and the extended-range formats) keeps Metal's
    /// rounding of one texel's step with ties away from zero, but in a sample through a runtime
    /// sampler slot: where its word reads one texel, the step rounds to nearest even, one `half`
    /// ulp apart at a tie, announced by the runtime at a draw or dispatch that binds such a texture
    /// there.
    HalfTexelConversion,
    /// The module samples a texture with a `half` result through a filter (linear minification
    /// or magnification, a linear mip filter, anisotropy, or a runtime sampler) that Metal's own
    /// sampler (`VkProperty::MetalSampler`, MoltenVK) answers otherwise than Metal filters the
    /// format bound there: a two-plane YCbCr texture's (`YCBCR8_420_2P` and its kin), whose view
    /// holds Metal's reads exactly but which Metal's sampler filters as an `RGBA16Float` or
    /// `RGBA32Float` texture's texels, within one step of the format's grid, where Metal sums the
    /// steps on the grid, each term truncated (`lower::sampler::linear`). Recorded under the slot's
    /// bound format, which the runtime reads at each draw or dispatch and announces as
    /// `ycbcr-filter-not-exact`, through a runtime sampler whatever its word states. A sample
    /// through a constexpr sampler that reads one texel of one level (nearest minification and
    /// magnification, no mip filter or a nearest one, no anisotropy) and a gather are not filtered
    /// and do not carry it. A device that samples natively records
    /// [`PrecisionNote::DeviceSampling`] instead. The `float` result's counterpart is
    /// [`PrecisionNote::FloatSampleFilter`]. Validation derives no tolerance from it: the class is
    /// unbounded.
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
    /// (`StageIo::decoded_fetches`). On a device that samples through Metal's own texture unit
    /// (`VkProperty::MetalSampler`: MoltenVK on Apple's GPU) the device's decode is Metal's, and
    /// the module requires the property instead. Measured over every code of `RGBA8Unorm_sRGB`
    /// (`runtime_lane::srgb_decode`): MoltenVK on an Apple M3 Max decodes each as Metal does, bit
    /// for bit; lavapipe (LLVM 21.1.8, arm64) each color code within 5.25 table steps (0.00128, a
    /// third of a code's `1/255`) of Metal's value, alpha within one `f32` ulp; other devices are
    /// not measured.
    ///
    /// Where the device cannot sample the bound format, whose texture it holds as the format's
    /// UNORM twin (`metal::texel_format::srgb_twin_on`: `R8Unorm_sRGB` and `RG8Unorm_sRGB` on
    /// lavapipe, `RG8Unorm_sRGB` on ANV and NVIDIA among the committed profiles), the slot's
    /// `SrgbTexture` variant decodes what the twin's view reads by the MSL transfer through the
    /// device's division and `Pow` (`pack::srgb_decode_texel`): a read, a sample, a fetch, a
    /// gather and a bicubic tap one texel's codes, a filtered sample (a linear filter, a linear
    /// mip filter, anisotropy, or a runtime sampler) the device's filter of the codes, the decode
    /// of the codes' blend where Metal blends the decoded texels. So does a read of an 8-bit sRGB
    /// storage texture held through its UNORM alias (`StorageImageClass::SrgbEncode`, Vulkan
    /// having no sRGB storage view). Both are recorded outright, on every device: Metal's table
    /// rounds the transfer to a twelve-bit step, up to half a step (`1 / 8190`) from it, where
    /// MoltenVK's `Pow` read within `1.83e-7` of the transfer through the alias
    /// (`runtime_lane::srgb_storage`). Validation derives no tolerance from it: the class is
    /// unbounded.
    SrgbTexelDecode,
    /// The module samples a texture with a `float` result through a filter that Metal's own
    /// sampler (`VkProperty::MetalSampler`, MoltenVK) answers otherwise than Metal filters the
    /// format bound there: a two-plane YCbCr texture's, as for [`PrecisionNote::HalfSampleFilter`]
    /// (a depth texture's sample and a comparison's included). A sample is filtered unless it goes
    /// through a constexpr sampler with nearest minification and magnification, no mip filter or a
    /// nearest one, and no anisotropy: linear minification or magnification, a linear mip filter,
    /// anisotropy, and a runtime sampler (whose filter is not known when the function compiles) all
    /// filter. Recorded under the slot's bound format, announced as `ycbcr-filter-not-exact`. A
    /// device that samples natively records [`PrecisionNote::DeviceSampling`] instead. Validation
    /// derives no tolerance from it: the class is unbounded.
    FloatSampleFilter,
    /// A vertex or fragment function keeps more private memory (its `alloca`s, a call stack among
    /// them, or constants held as `Private` variables) than the floor, 16 KiB a thread, on a device
    /// whose Function storage was not measured, up to the 192 KiB a thread Metal was measured to
    /// hold (`device::function_storage`). A driver backs that storage with scratch memory, and
    /// Vulkan states no amount it holds: past what it holds a driver can answer wrongly or hang the
    /// GPU with no error. Probed, the RTX 5080 Laptop GPU hung at 48 KiB and the RTX 3090 at
    /// 192 KiB (Xid 109), and the Iris Xe answered wrongly at 48 KiB and the Radeon 780M at
    /// 128 KiB. The module runs on this device all the same, resting on no measurement; a kernel,
    /// an object or a mesh function past the floor there is refused instead. Measuring the device
    /// (`runtime_lane::private_memory`) and adding its row removes the note up to its value.
    /// Validation keeps a difference as a mismatch: the class is derived as if the note were
    /// absent.
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
    /// attachment stores as written, on a device other than MoltenVK on an Apple GPU
    /// (`VkProperty::MetalColorWrites`) not measured to convert that kind of channel as Metal does
    /// (`DriverDeviation::MetalSmallFloatColorWrites`, `MetalNorm16ColorWrites`), an attachment that
    /// blends with factors other than one and zero (`ChannelWrite::Device`), the alpha of
    /// attachment 0 under alpha-to-coverage, a format whose conversion Metal was not measured with
    /// (the packed 16-bit unorm formats, `A8Unorm`, the packed floats). (A `bfloat` output is
    /// refused, as Metal's compiler rejects one.) Near a tie between two codes the device may store
    /// the other one (every Vulkan driver measured but MoltenVK rounds the `f32` product with ties
    /// to even, encodes sRGB by its own approximation and rounds a half to nearest, where Metal
    /// rounds the exact product, encodes through its twelve-bit table and rounds a half toward
    /// zero), and a NaN stores as its conversion makes it (Metal stores a normalized channel's as
    /// 0, a blended snorm channel's as -1.0's code). A 32-bit float attachment that so blends
    /// stores the device's blend arithmetic (NVIDIA's 615.71.09 one `f32` ulp from Metal's in some
    /// source-over sums, a NaN as `0x7fffffff` where Metal stores `0x7fc00000`).
    ///
    /// The recorded bound past one code, measured on lavapipe (LLVM 21.1.8) against the conformance
    /// suite's recordings of plain Metal on an Apple M3 Max under macOS 27.0.1: a blend into an
    /// 8-bit unorm attachment (`RGBA8Unorm`, `ARGB8Unorm`) stores within 2 codes of Metal's; a
    /// fragment function that fetches an 8-bit sRGB attachment and writes it again, within 2 codes
    /// once a later draw reads an earlier one's code (the device's decode,
    /// [`PrecisionNote::SrgbTexelDecode`], beside its encode); a blend into an 8-bit sRGB
    /// attachment, within 4 codes. Another device is not bounded past one code: an NVIDIA RTX 5080
    /// Laptop GPU's blend into an 8-bit sRGB attachment stored up to 11 codes from Metal's in the
    /// same suite.
    ///
    /// Also: a function writes a `float` (or `half`) texel to a storage texture that may hold a
    /// normalized or half-float format, or `ARGB8Unorm`, which the device converts where it is not
    /// measured to store Metal's codes (`DriverDeviation::MetalNormStorageWrites`,
    /// `MetalHalfStorageWrites`; a 1-bit alpha on a device measured to store every other
    /// normalized channel's): Metal stores each normalized channel's exact product rounded with a
    /// tie away from zero (a 1-bit channel 1 only above one half) and a half rounded toward zero.
    /// An integer texel a function writes to a storage texture of the float class that may hold a
    /// half-float or small-float format (`RG11B10Float`), its lanes' bits as the `float` written,
    /// records it on every device: Metal's integer store rounds such a channel to nearest even,
    /// the device's store as it rounds a `float` write (toward zero on every measured driver:
    /// within one unit in the last place of the channel's value, a finite value past 65504 stored
    /// as 65504 where Metal's `half` is the infinity). So does a fragment function's `float` output
    /// blended in the shader into a 16-bit float attachment it renders through a storage image (the
    /// pixel interlock's fetch), whose `half` the store rounds, on a profile without
    /// `DriverDeviation::MetalHalfStorageWrites`.
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
    /// Also: a blend the function computes in the storage form into an 8-bit unorm attachment
    /// (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`), whose result the storage image's store
    /// converts to its code, where Metal rounds the exact product with 255 (a NaN 0); the store may
    /// take the other code near a tie.
    ///
    /// Also (the runtime announces it): a render pass clears a normalized, sRGB, packed 16-bit,
    /// half or small-float color attachment to a color whose conversion is in question. The
    /// device converts the clear color, which Vulkan asks it to round to nearest, leaving a tie,
    /// an sRGB encode and a float's rounding to it, where Metal's clear rounds each channel to
    /// nearest, a tie to even (a packed 16-bit channel through `half`) and truncates a small
    /// float's and a shared exponent's mantissas. A normalized channel lies within 1/1024 of a
    /// midpoint between two codes there, and stores one of the two. Not for a clear by the load
    /// operation of an attachment the pass's render area covers on a driver whose load operation
    /// clears as Metal's `Clear` load action does (`DriverFact::ColorClearIsMetals`: MoltenVK on
    /// the Apple9 GPUs under macOS 27, measured on the M3 Max).
    ///
    /// Also (the runtime announces it): a plain color fill (`fillTexture…color:` without
    /// `pixelFormat:`) of a texture of such a format the device renders, which the device's clear
    /// of the color writes, to a color whose conversion is in question against Metal's fill
    /// packing, which rounds a normalized tie half up and takes the sRGB code whose decoded value
    /// is nearest (a packed 16-bit channel's margin widened by a step through `half`).
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceColorConversion,
    /// A fragment function reads a varying the device interpolates (`center_perspective`,
    /// `centroid`, `sample` or `no_perspective`, not `flat`) on a device not measured to
    /// interpolate as Metal does (without
    /// [`DriverDeviation::MetalInterpolation`](crate::error::DriverDeviation::MetalInterpolation),
    /// which only MoltenVK takes: there Metal interpolates). Vulkan states no precision for
    /// interpolation, and every other device measured rounds otherwise than Metal: on a perspective
    /// triangle list, texcoords over a full-screen quad and a colour whose `w` varies, 31% to 45%
    /// of the lanes RADV, ANV, lavapipe and NVIDIA's driver interpolated were Metal's bit for bit,
    /// the rest up to about `3e-4` apart (lavapipe, which snaps vertices otherwise) and `1.5e-4`
    /// elsewhere. The exact path, interpolating each varying in the shader by Metal's formula,
    /// is not built: Metal's formula is not known.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DeviceInterpolation,
    /// A fragment function runs a quad or simdgroup operation on a device that shades a
    /// triangle it clips as several triangles
    /// ([`DriverWorkaround::ClippedTrianglesSplitPixelFragments`]): a quad a seam between the
    /// pieces crosses runs once per piece, each run with the other piece's pixels as helper
    /// invocations, so the operation counts or combines fewer active lanes than Metal's, whose
    /// quad holds all four. Metal permits a partly occupied quad, so the device's answer is one
    /// Metal's API allows, not the reference's; no Vulkan mechanism keeps the clipped triangle
    /// one primitive (a guard band emulated by scaling the viewport is exact only within the
    /// viewport limits).
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, which C3 names as the answer gap of a case that meets it.
    ///
    /// [`DriverWorkaround::ClippedTrianglesSplitPixelFragments`]: crate::device::DriverWorkaround::ClippedTrianglesSplitPixelFragments
    QuadsSplitByClipping,
    /// A vertex function runs a simdgroup or quad operation on a device whose vertex stage takes
    /// no required subgroup size and whose driver is measured to group vertices otherwise than
    /// Apple's GPUs ([`DriverWorkaround::VertexGroupingNotApples`]: ANV, 8 vertices a subgroup,
    /// interleaved): the operation runs over the device's group of vertices, so a sum, a ballot
    /// or a shuffle combines other vertices than on an Apple GPU, whose simdgroup holds 32
    /// consecutive vertices. Metal does not fix which vertices share a simdgroup, so each
    /// device group is a simdgroup Metal's API allows, not the reference's; with no required
    /// size for the stage, nothing pins the grouping.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    ///
    /// [`DriverWorkaround::VertexGroupingNotApples`]: crate::device::DriverWorkaround::VertexGroupingNotApples
    VertexSimdgroupGrouping,
    /// A fragment function runs a simdgroup operation (a reduction, scan, vote, ballot, shuffle,
    /// broadcast or AGX multiply-add) or reads its `[[thread_index_in_simdgroup]]` on a device
    /// whose fragment subgroups are not Metal's simdgroups (the profile does not take
    /// [`DriverDeviation::MetalFragmentSimdgroups`](crate::error::DriverDeviation::MetalFragmentSimdgroups)):
    /// the operation runs over the pixels the device's rasterizer packs into a subgroup, so a sum,
    /// a ballot or a shuffle combines other pixels than on an Apple GPU, whose packing is
    /// measured, and the lane is the pixel's place in the device's subgroup. Metal does
    /// not fix which pixels share a simdgroup, so each device subgroup is a simdgroup Metal's API
    /// allows, not the reference's. Quad operations are not concerned: a quad is the same 2x2
    /// pixels on every device.
    ///
    /// A device whose fragment subgroup holds fewer than two quads runs it as a partly occupied
    /// simdgroup (`Tier::PartialFragment`), which Metal permits: there the operation combines the
    /// one quad's pixels alone, so a reduction ahead of a `discard_fragment` counts no quad the
    /// discard drops, where an Apple GPU's 32-lane simdgroup counts the dropped quads of its 8x4
    /// pixel block. Measured on arm64 lavapipe, whose fragment stage states one subgroup size,
    /// 4 (`[minSubgroupSize, maxSubgroupSize]` is `[4, 4]`): the thirteen shapes of
    /// `runtime_lane::discard_hoist` all answer the survivors, where Metal's answer counts the
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
    /// denormals. The exact path, the blend in the function under the pixel
    /// interlock, is not built.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    BlendedWriteRounding,
    /// A fragment function writes an attachment held with its channels rotated (`ARGB8Unorm`)
    /// whose blend is none of the measured forms (`RotatedBlendForm::Unmeasured`), or the
    /// premultiplied form through a first output that is not 32-bit float: Apple's blend
    /// computes a factor pair apart in its color and alpha arithmetic, and the rotation moves
    /// Metal's blue into the image's alpha and Metal's alpha into its red, so a channel may store
    /// one code apart from Metal's (up to 1,767 of 65,536 tie-heavy texels over 440 swept blend
    /// states on an M4). The exact path for a blend state is its measured form
    /// (`pipeline::rotated_blend`'s table).
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
    /// A fragment function runs its draw's depth test itself at more than one sample on a device
    /// whose `FragCoord.z` is the pixel centre's there
    /// ([`VkProperty::CenterFragCoordDepth`](crate::device::VkProperty::CenterFragCoordDepth)), and
    /// computes each sample's depth from its vertex function's window depth read at the sample,
    /// plus `FragCoord.z` less that depth read at the centre, which carries the depth bias. On the
    /// M4 that is the depth MoltenVK stores at every sample of flat, sloped and steep triangles
    /// at the viewport's default depth range, under a constant bias, and where the clamp decides;
    /// under perspective, a slope-scaled bias, or another depth range a sample's depth may be up
    /// to 2 ulp from it, so a depth test between surfaces that close may decide
    /// otherwise than Metal's. No draw's state is known when the module is made, so the function
    /// records the note wherever it computes the depth.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    ComputedSampleDepth,
    /// An access through an argument buffer's sampler whose word the specialized module cannot read
    /// may hand the device a pixel coordinate scaled by the texture's size, where the sampler
    /// states pixel coordinates: the device's sampler reads its texels from that coordinate in its
    /// own arithmetic, with [`PrecisionNote::PixelSamplerScaling`]'s measured bound (the
    /// neighbouring texel, block or sub-texel step where the coordinate lies within a few units in
    /// the last place of an edge of one). Recorded on every device for every such access; a sample
    /// without a comparison through a sampler whose word states the exact fetch (nearest, clamped
    /// to the edge, one level, isotropic) reads that fetch instead. It is its own note, apart from
    /// [`PrecisionNote::PixelSamplerScaling`], because only the bound states decide it: the runtime
    /// announces it at the first draw or dispatch of a pipeline that records it while the device
    /// lists a sampler state of pixel coordinates, never at the pipeline's making.
    ///
    /// Validation derives no tolerance from it: at a texel edge the value is the neighbouring
    /// texel's, whatever it holds.
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
    /// the output widened to `float`; RADV and ANV, which enable `storageInputOutput16`, receive
    /// it as a 16-bit `half`, `StageIo::half_outputs`, which is not measured there); and 1 or 2
    /// codes of an 8-bit or extended-range format. MoltenVK 1.4.2 on the M3 Max (macOS 27.0.1),
    /// which receives the 16-bit `half`, stores the value of Metal's measured rule in every channel
    /// of every measured blend draw of the sweep into `R16Float`, `RG16Float` and `RGBA16Float`
    /// (28 of 28 each, `runtime_lane::half_output_blends`), and into `R32Float`, `RG32Float` and
    /// `RGBA32Float` 28, 28 and 26 of 28, the two off at one channel by a zero's sign.
    ///
    /// Also: a `float` color output (a scalar or vector of AIR's `float`) of a module compiled
    /// without fast math to a 16-bit float attachment that blends by one and zero, added, for
    /// color and alpha (`StageIo::float_blend_colors`). Into a 16-bit float attachment Metal's blend
    /// of a module without fast math keeps a zero factor's product, the destination's by zero
    /// there: a NaN of an infinite or NaN texel, and `+0` of a `-0` output over a texel of `+0` or
    /// above (plain Metal on the M3 Max under macOS 27.0.1, `half4` and `float4`
    /// outputs). Where the function does not compute that blend in the storage form, the pipeline
    /// blends the output by the device's blend, whose answer for those texels is the device's own,
    /// where a `float` output would otherwise store as written.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    HalfOutputPrecision,
    /// A fragment function's `half` color output (a scalar or vector of AIR's `half`) is blended
    /// into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, or an
    /// `_sRGB` twin), whose raw codes no device's blend unit reads, so the function blends in the
    /// shader (`lower::arith::extended_range::blend::native`), in the device's arithmetic: a blend
    /// unit's `S·Fs op D·Fd` in `float`, in the blend equation's order, the source, the second
    /// source and the blend color clamped to the format's range first. Metal's compiler folds the
    /// blend into the function and fuses, regroups and rounds each step to `half` instead, so a
    /// blended channel may store another code than Metal's; the bound is not measured. Recorded
    /// for every blended output that stores a channel, under either math. A `float` output
    /// records [`PrecisionNote::ExtendedRangeFloatBlendSource`] instead, so a function of both
    /// records both.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    ExtendedRangeBlendSource,
    /// A fragment function meets the store-dropping defect of
    /// [`DriverWorkaround::DiscardDropsSimdgroupStores`] (MoltenVK on an Apple GPU: Apple's
    /// compiler drops the texture writes of every live thread of a simdgroup holding a discarded
    /// thread, where the function discards, then writes storage, samples, and calls
    /// SPIRV-Cross's `optnone` arithmetic helpers), and a discard site that would discard at
    /// once cannot wait until after the writes: a store, atomic, image write or non-quad
    /// subgroup operation follows it, which a predicated discard does not yet guard
    /// (`lower::fragment::discard`). The module keeps its forms, so the live fragments of a
    /// simdgroup (8 by 4 pixels) that holds a discarded one may lose their texture writes (and,
    /// not measured, their buffer stores and atomics); color outputs are not affected.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    ///
    /// [`DriverWorkaround::DiscardDropsSimdgroupStores`]: crate::device::DriverWorkaround::DiscardDropsSimdgroupStores
    DiscardDropsSimdgroupStores,
    /// A fragment function meets the store-dropping defect of
    /// [`DriverWorkaround::DiscardDropsSimdgroupStores`] without a discard: SPIRV-Cross writes an
    /// operation of the module as an `optnone` call (a sum, difference or product decorated
    /// `NoContraction`, among others) where only helper invocations (the quad helpers past a
    /// primitive's edges, or demoted invocations) may run it before a storage write (not in a
    /// selection on a value every invocation of a draw computes alike), and keeping them out would
    /// not be exact (`spirv::helper_branches`): a derivative or subgroup operation reads what they
    /// compute there or after, an arm leaves its selection otherwise than at its merge, or the call
    /// is in a loop whose helper invocations may iterate alone. Only where a path from the call
    /// ends without a tangled instruction (an implicit-LOD sample, an LOD query, a derivative, a
    /// subgroup operation) that reads what it computes: one after it keeps the writes. The live
    /// fragments of a simdgroup (8 by 4 pixels) whose helpers make the call alone may lose their
    /// texture writes (and, not measured, their buffer stores and atomics); color outputs are not
    /// affected.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    ///
    /// [`DriverWorkaround::DiscardDropsSimdgroupStores`]: crate::device::DriverWorkaround::DiscardDropsSimdgroupStores
    HelperBranchDropsSimdgroupStores,
    /// [`PrecisionNote::ExtendedRangeBlendSource`] for a `float` color output (a scalar or vector
    /// of AIR's `float`): blended into an extended-range attachment in the shader in the device's
    /// arithmetic, where Metal's compiler rounds the output and each step of the blend to `half`.
    /// A blended channel may store another code than Metal's; the bound is not measured. Its own
    /// note, apart from the `half` output's, so a consumer can tell the two apart.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    ExtendedRangeFloatBlendSource,
    /// The module samples a texture through an anisotropic sampler with Metal's anisotropic
    /// filter emulated by the device's samples at explicit levels (`lower::sampler::anisotropic`),
    /// on a device without `samplerAnisotropy`, every Vulkan sampler being isotropic there, each
    /// tap the device's isotropic filter: a sample without comparison of a 2D texture or 2D array
    /// at the fragment's implicit level or with explicit gradients, or of a cube or cube array with
    /// explicit gradients, through a constexpr sampler of `maxAnisotropy` above 1, or through a
    /// runtime sampler slot whose word states anisotropy (under that word alone). A device with
    /// the feature samples with its own anisotropic filter. Metal's footprint (its gradient lengths,
    /// piecewise-linear `log2`, level, tap count, tap weights and direction) is reproduced as
    /// measured on an Apple M4 (the M3 Max answering identically), with this residue:
    /// - the level is off by `1/256` of a level on about 1% of anisotropic samples;
    /// - the tap spacing is Metal's at the ratio cap and for two taps (but where the ratio lies
    ///   within `0.07/256` of a rounding tie, on about 5% of two-tap footprints, by up to 1.8/256
    ///   of a texel), and a fitted approximation for three or more taps below the cap (exact on
    ///   1,021 of 1,674 measured windows; up to about 18/256 of a texel at the outer tap of 10 or
    ///   more taps);
    /// - each tap is the device's own filter at the level the shader hands it, so its footprint
    ///   and sum are the device's (Metal's own on MoltenVK; elsewhere
    ///   [`PrecisionNote::DeviceSampling`]);
    /// - a cube's taps step across the direction's face by Metal's face gradients (Metal under
    ///   macOS 27 samples a cube as the face's 2D plane, measured on the Apple M3 Max), the
    ///   direction moved and projected onto the face again by the device, and a tap past the
    ///   face's edge reads the next face as the device projects it, where Metal's is not measured.
    ///
    /// Measured by `runtime_lane::anisotropic_filter` against the M4's answers on `rgba8Unorm`
    /// at `maxAnisotropy` 2 to 16. With a linear filter, every channel is within a code: at most
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
    /// A vertex function writes `[[point_size]]`, or a mesh function `set_point_size`, on a device
    /// whose `pointSizeRange` ends below Metal's largest point, 511 pixels
    /// ([`Limits::max_point_size`](crate::device::Limits::max_point_size): 255 on ANV, 256 on
    /// lavapipe). The module stores a size past 511 as 511, as Metal draws it, and the device
    /// clamps a size past its own bound to that bound, so a point between the device's bound and
    /// 511 pixels draws smaller than on Metal (a square of the bound's side). Drawing such a
    /// point as two triangles is the exact path, not built.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review.
    DevicePointSize,
    /// A fragment function compiled for a multisampled pipeline reads `[[sample_mask]]` on a
    /// device whose rasterizer is not Apple's
    /// ([`DriverWorkaround::ClippedCoverageNotMetals`](crate::device::DriverWorkaround::ClippedCoverageNotMetals)):
    /// Apple's GPU draws a triangle that extends past the viewport as pieces whose shared edges
    /// cross pixels, and such a pixel's fragments can run once per piece, each reading that
    /// piece's coverage (on an M3 Max and an M4), where the device hands the pixel's fragment its
    /// coverage by the whole primitive (lavapipe by its own pieces). Where Metal splits a pixel,
    /// the function reads the whole coverage once instead of each piece's; no Vulkan mechanism
    /// follows Apple's clipper.
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
    /// pipeline's stage reads, or the interlocked storage form's write), where
    /// [`PrecisionNote::HalfOutputPrecision`]'s conditions hold; or computes the blend of a 16-bit
    /// float attachment it fetches where Metal's compiler may regroup it (below). Metal's compiler
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
    /// may relay it as a known difference. Where the device enables `storageInputOutput16` the
    /// output reaches it as a 16-bit `half` (`StageIo::half_outputs`): MoltenVK 1.4.2 on the M3
    /// Max (macOS 27.0.1) then stores Metal's codes at every word of every blend of the
    /// conformance family `half-output-blends-*` (a loaded `half4`, 21 blends into each 16- and
    /// 32-bit float format); a computed output is not measured there.
    ///
    /// A fragment function that fetches a blended `R16Float`, `RG16Float` or `RGBA16Float`
    /// attachment in the interlocked storage form computes the blend itself, no Vulkan blend
    /// applying to a storage write (`lower::fragment::fetch`): in the arithmetic of its `half` or
    /// `float` output of one to four lanes, as Metal's compiler appends it (plain Metal on the M3
    /// Max under macOS 27.0.1: every channel of 677 and 1,350 blend states into each format, half
    /// and float outputs, safe and fast math). Under fast math it records this note where the blend
    /// is not Metal's exactly: where Metal's compiler may fold the output's own arithmetic into the
    /// blend (a lane computed by a fast-math operation, a constant, or the fetched color: a
    /// returned product fused into the blend's one minus the source alpha), or the blend's fast
    /// form is a sum, a lerp, a common factor or a product rounded beside a fused one (but a blend
    /// constant's difference), which the sweeps fit in most states but not all. It then stores
    /// within 2 units in the last place of a `half` of Metal's at the operands' magnitude, 16 at
    /// the stored value's under cancellation (below one at the function's own largest term). Where
    /// every lane of the output is opaque to the blend and its form fits every state, the blend is
    /// Metal's and the function records nothing.
    ///
    /// The pipeline's stage also records it for an attachment of `StageIo::half_fast_colors` that
    /// stores some channel **without** blending (`color_writes::stores_output`, which is the test
    /// it makes), and that is deliberate rather than the blend's name read loosely: under fast math
    /// Metal's compiler may keep the output's last operation at `float` and let the attachment's own
    /// conversion round once, where this rounds the output to `half` first, so a channel whose format
    /// converts the `float` and its `half` to different codes may store a value apart from Metal's.
    /// It is the cause [`PrecisionNote::HalfOutputPrecision`] records for a module naming no math
    /// mode, under a name that reads as a blend's alone. **Unmeasured**: no sweep has made an
    /// unblended fast-math store differ. The 12 `xr-draw-*` conformance cases are the closest
    /// reading of it and all 12 store Metal's codes, so for those shapes the note is a may-differ
    /// and not a measured difference. Narrowing it to a blend would stop announcing a class the
    /// `HalfOutputPrecision` measurement says exists, so it stays where it fires until a sweep over
    /// the formats whose two conversions disagree says otherwise.
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
    /// on `R32Float`). A device that samples through Metal's own sampler (MoltenVK on Apple's GPU)
    /// never records it.
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
    /// 0.498, 0.125, 0.125 and 0.125.
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
    /// Metal's is `x · rsqrt(x)`); `FAbs` and `FSign` the correct result, `Fract` and `Trunc`
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
    /// once more: within one unit in the last place of the `half` result. Validation derives no
    /// tolerance from it.
    NativeDivision,
    /// Native device maths: a `float` fused multiply-add on a device without `OpFmaKHR`, or a
    /// `half` one where the device's 16-bit `Fma` is not measured to be Metal's and the device has
    /// no 16-bit `OpFmaKHR` under the 16-bit float controls, is the device's `GLSL.std.450 Fma`
    /// (on the `half` operands held in `float`, its result rounded to `half`), where Metal's rounds
    /// once. **Recorded bound:** a `half` result within one unit in its last place of
    /// Metal's (the product of two `half` values is exact in `float`, so only the two roundings
    /// differ); a `float` one within half a unit in the last place of the product plus half a
    /// unit in the last place of the result (Vulkan lets `Fma` round the product and then the
    /// sum). Validation derives no tolerance from it.
    NativeFma,
    /// Native device sampling (a texel conversion): the module hands a read, sample or gather
    /// with a `float` result the device's own conversion of the texel's codes, off Apple's GPUs
    /// (`VkProperty::MetalSampler` keeps Metal's value). Recorded for every such access of a
    /// texture slot or a heap texture (not a heap's texture buffer,
    /// [`PrecisionNote::DeviceNanEncoding`]). The recorded bound, of a texel whose access reads
    /// one texel of one level:
    /// - an 8-, 10- or 16-bit normalized code `c`: Metal's `c / (2^b - 1)` correctly rounded;
    ///   the device's within one `f32` ulp of it where measured (lavapipe multiplies by the
    ///   reciprocal: some 8-bit codes one ulp high, some 10-bit codes of `BGR10A2Unorm`, a 16-bit
    ///   code and 512 `Depth16Unorm` codes one ulp low);
    /// - a 4-, 5- or 6-bit channel of a packed format: the device may widen it by replicating its
    ///   bits (ANV, NVIDIA), within 0.18 of a code (`1 / (2^b - 1)`) where it replicates a 5- or
    ///   6-bit channel to 8 bits and exactly for a 4-bit one; or convert it by the reciprocal,
    ///   within one `f32` ulp of Metal's `c / (2^b - 1)` (lavapipe: some codes of `ABGR4Unorm`
    ///   one ulp high, and of `A1BGR5Unorm`, `B5G6R5Unorm` and `BGR5A1Unorm` one ulp apart);
    /// - a half-float or small-float texel's NaN: the device's encoding, where Metal answers
    ///   `0x7fc00000`;
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
    /// `half` attribute reads the device's `float` value rounded to `half`), on every profile but
    /// one that takes `DriverDeviation::MetalNormalizedVertexFetch` (MoltenVK on Apple's GPUs of
    /// the Apple M3 Max's macOS 27 identity), whose `float` is measured to be Metal's at every
    /// code: there such an attribute records nothing and the pipeline requires
    /// `VkProperty::MetalNormalizedVertexFetch`, but for a 16-bit format read as `half`, which
    /// rounds twice. A function compiled alone records nothing for them: its module is the same
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
    /// note): NVIDIA, whose driver folds the conversion's round trip even under those controls
    /// (`DriverWorkaround::HalfConversionPairFolded`), MoltenVK, lavapipe, and any device
    /// without them. Without them Vulkan leaves the conversion's rounding mode and its 16-bit
    /// denormals to the device. The recorded
    /// bound, against Metal's `half(float)` (to nearest even on the half grid, half denormals kept,
    /// infinity from 65520 up, `0x7E00` for a NaN):
    /// - a finite result within one unit in the last place of the `half`: NVIDIA's 615.71.09
    ///   rounds to nearest even on the RTX 5080 Laptop GPU (every `float` pattern, loaded and
    ///   computed) and toward zero on the RTX 3090; lavapipe (Mesa 26.0.8) a loaded value toward
    ///   zero; Metal's compiler on MoltenVK folds the conversion into the instruction computing
    ///   the value, which rounds toward zero (`a1_half_tanh_regimes`, case 27353187: 1,204 of its
    ///   16,384 halves one ulp toward zero on the M3 Max);
    /// - or no rounding at all: NVIDIA's 615.71.09 on the RTX 3090 drops the conversion pair, with
    ///   or without a `NoContraction` barrier; lavapipe drops it in arithmetic on Mesa 26.2.4 with
    ///   LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, barrier or not, whose
    ///   arithmetic measured rounded. The result is then the `float` computation itself: its
    ///   differences from Metal's compound across operations, and a value past the `half`'s range
    ///   stays finite until a `half` store narrows it by the device's conversion (to infinity, or
    ///   to 65504 rounding toward zero). The RTX 5080 Laptop GPU on the
    ///   same driver rounds here (8192 of 8192 sums Metal's);
    /// - on a device that rounds toward zero, a finite value from 65520 up is 65504 where
    ///   Metal's is infinity;
    /// - a result below the smallest normal `half` (`2^-14`) may be the zero of its sign on a
    ///   device that flushes 16-bit denormals (the RTX 5080 keeps them);
    /// - a NaN's encoding and the sign of a zero result are the device's: on the RTX 5080 a
    ///   computed `half` NaN is `0x7FFF` (where 18 C3 cases differed in NaNs, signs of zero,
    ///   `atan2` and `pow` at special operands, and `half` texel and pack conversions).
    ///
    /// Every `half` a module reads from its bits (a load, a reinterpretation, a texel's bits) is
    /// the device's widening of them, on every device: an `OpFConvert` to 32 bits of the bits read
    /// as a 16-bit float with `shaderFloat16`, `GLSL.std.450 UnpackHalf2x16` of them without it.
    /// So too every `half` a module encodes to its bits (a store, a reinterpretation, a texel's
    /// bits) where they are not the bits it was read from: the device's narrowing of it, an
    /// `OpFConvert` to 16 bits read as a word with `shaderFloat16`, `GLSL.std.450 PackHalf2x16`
    /// without it. A module that reads or encodes one records this note too where that widening
    /// or narrowing is neither under the float controls' `DenormPreserve` for 16 bits (ANV, RADV)
    /// nor measured to read and narrow every `half` as Metal's (MoltenVK's conversion is,
    /// `DriverDeviation::MetalHalfArithmetic`, and the module requires it there), so on every
    /// device without `shaderFloat16`: its bound is the last two items', a half denormal read or
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
    /// quantize's rounding mode to the device, and the SPIR-V specification lets it answer a zero
    /// for a magnitude too small for a normal `half`. The recorded bound, against Metal's `half(float)`: a finite result within one unit in the last
    /// place of the `half`; a value from 65504 up to 65520 may be infinity where Metal's is 65504
    /// (NVIDIA's 615.71.09 rounds `0x477FEFFF` to infinity); a value below the smallest normal
    /// `half` (`2^-14`) may round to a zero, of either sign, where Metal answers the half denormal,
    /// or `2^-14` from just below it (the quantize tests the value it rounds; NIR's
    /// `fquantize2f16` flushes every one, SwiftShader's too); a NaN's encoding is the device's.
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
    /// (`0x7e00` as `half`), an operand's bits (`round`, `ldexp`, `fast::tan`, and precise `pow`
    /// of a NaN base to the exponent one), or the canonical
    /// NaN with an operand's sign (`asin`, `atan`, `asinh`, `atanh`); a `float` `fneg` and `fabs`
    /// keep a NaN operand's payload on Metal, and a `half` `fract` returns it. Every non-NaN result
    /// is as without it. Also recorded by every module that reads a `half` from its bits (a load, a
    /// reinterpretation, a texel's bits), on every device: the read is the device's widening (an
    /// `OpFConvert` of the bits read as a 16-bit float with `shaderFloat16`, `UnpackHalf2x16`
    /// without it), and a NaN widens to a NaN whose bits are the device's, where Metal widens
    /// every `half` NaN to `0x7fc00000` and moves one it does not widen with its bits, so a `half`
    /// NaN the module moves as a value (through a select, a phi or a sign operation) may leave
    /// with the device's bits; a `half` stored back unchanged keeps its bits. And by every module
    /// that encodes a `half` to its bits where they are not the bits it was read from (a store, a
    /// reinterpretation, a texel's bits), on every device: the encode is the device's narrowing
    /// (an `OpFConvert` to 16 bits with `shaderFloat16`, `PackHalf2x16` without it), and a NaN
    /// narrows to a NaN whose bits are the device's, where Metal keeps a `half` NaN's sign and
    /// payload. Every other `half` widens to its value and narrows to its bits, but a denormal on a
    /// device that may flush 16-bit ones ([`PrecisionNote::NativeHalfConversion`]). Validation
    /// derives no tolerance from it.
    NativeNanEncoding,
    /// Native device maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its
    /// `clamp(x, +0, 1)`) on `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or
    /// `NClamp`, on a device not measured to answer Metal's operand; a `saturate` of a format
    /// whose operand Metal's other functions do not pick (`bfloat`, `double`) on every device; and
    /// `median3` on every device, `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and
    /// `NMax`, whose pick of two NaNs and of an `f32` denormal is the device's (Metal's
    /// `fast::median3` answers `b` for two NaN first operands).
    /// **Recorded bound:** the
    /// result is one of the operands, the one Metal answers but at three ties: of two NaNs a NaN
    /// (its bits the device's), where Metal answers the second operand's; of `+0` and `-0` either
    /// zero, where Metal answers the second; and an `f32` denormal beside a zero or another
    /// denormal compared by its value where the device keeps denormals, where Metal compares it as
    /// a zero and answers the second. Validation derives no tolerance from it.
    NativeMinMax,
    /// Native device maths: Metal's `dot` on `float` and `half` is the device's `OpDot` (a `half`
    /// one summed in `float` and rounded once to `half`), not Metal's fused chain of its products,
    /// on every device but for a `float` one where the device's functions are Metal's own
    /// (MoltenVK, Metal's own `dot`). **Recorded bound:** a `float` `dot` of `n`
    /// lanes within `(3n - 1) * 2^-24 * sum |a_i * b_i|` of Metal's chain, plus `n * 2^-126` where
    /// a product is an `f32` denormal the device keeps (Vulkan's `OpDot` inherits `OpFMul`'s and
    /// `OpFAdd`'s rounding, `2n - 1` roundings, where Metal's chain rounds `n` times); a `half`
    /// one within `(n + 3) * 2^-11 * sum |a_i * b_i| + (n + 2) * 2^-25` of Metal's chain of `half`
    /// fused multiply-adds. Validation derives no tolerance from it.
    NativeDot,
    /// Native device maths: Metal's `mix(x, y, t)` on `float` and `half` is the device's `FMix`
    /// (a `half` one computed in `float` and rounded once to `half`), not Metal's
    /// `fma(t, y, fma(-t, x, x))`, on every device but for a `float` one where the device's
    /// functions are Metal's own (MoltenVK, Metal's own `mix`). **Recorded bound:** a `float`
    /// result within `2^-21 * (|x * (1 - t)| + |y * t|)` of Metal's (Vulkan's `FMix` inherits
    /// `x * (1 - t) + y * t`'s rounding, four roundings, where Metal's rounds twice); a `half`
    /// one within `3 * 2^-10 * (|x * (1 - t)| + |y * t|) + 2^-23`. Validation derives no
    /// tolerance from it.
    NativeMix,
    /// Native device maths (a conversion): `air.pack` or `air.unpack` of a normalized format is
    /// the device's: `GLSL.std.450`'s `PackUnorm4x8`, `PackSnorm4x8`, `PackUnorm2x16` and
    /// `PackSnorm2x16` (and their `Unpack`s) for the four formats it names, and for `rgb10a2` and
    /// `rgb565` those instructions' definition on each field, the device's `NClamp`, multiply,
    /// `RoundEven` and conversion (a pack) or its multiply by the scale's reciprocal (an unpack);
    /// `unorm4x8_srgb`'s colour channels take the MSL transfer functions by the device's multiply,
    /// division and `Pow` around the 8-bit instruction. Metal rounds the exact product to its code
    /// and divides correctly rounded, answers code 0 for a NaN channel, and takes an sRGB colour
    /// through its twelve-bit table: a decode answers its step `k / 4095` nearest the transfer, an
    /// encode the byte of that table's step. Also: a write of an 8-bit sRGB storage texture held
    /// through its UNORM alias (`StorageImageClass::SrgbEncode`, Vulkan having no sRGB storage
    /// view) encodes by `unorm4x8_srgb`'s pack. Recorded on every device but for
    /// a `float` conversion of the four named formats on a device whose functions are Metal's own
    /// (MoltenVK: Metal's own `pack_float_to_*` and `unpack_*_to_float`). **Recorded bound:**
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
    /// in the interlocked storage form or through its word view, and a snapshot read's decode of a
    /// normalized code (`c` times the `f32` nearest `1 / D`, where Metal reads `c / D` correctly
    /// rounded). Validation derives no tolerance from it.
    DevicePackConversion,
    /// A render pipeline whose color attachments the device draws in more than one group
    /// (`GraphicsState::color_group`, the module docs of `pipeline::compile::color_group`): a
    /// device without Metal's eight color attachments draws each of its draws once per group,
    /// each group's variant writing that group's attachments over its own copy of the depth and
    /// stencil the pass loads. Every variant's vertex stage is the same module, its `Position`
    /// decorated `Invariant`, and every fragment variant computes `FragDepth`, `SampleMask` and
    /// `FragStencilRefEXT` by the same expressions, decorated `Invariant` (SPIR-V: "computed
    /// invariantly with respect to other shaders computing the same expressions"). Vulkan's
    /// invariance appendix promises no more across pipelines of other fragment modules and
    /// attachments (its Rule 4 holds for "identical pipelines", its Corollary 1 across the state
    /// of its Rule 2). **Residual:** groups past the first may differ at discard or depth edges if
    /// the driver compiles the variants differently. Nothing broader. Recorded only by a fragment
    /// stage of such a pipeline. Validation derives no tolerance from it.
    ColorAttachmentGroups,
    /// A tessellation pipeline drawn through the device's tessellation stages (a device with
    /// `tessellationShader`) whose tessellator is not Apple's
    /// ([`DriverWorkaround::TessellatorNotMetals`](crate::device::DriverWorkaround::TessellatorNotMetals)):
    /// the device's tessellator orders each patch's triangles and picks each one's first corner,
    /// and where Vulkan leaves it to the implementation chooses the triangles that join a patch's
    /// rings, where Metal's tessellator draws the D3D11 reference's triangles in its own order,
    /// each triangle's first corner by its stitch step (`lower::tessellation::tessellator`). The
    /// domain points are Metal's; a flat varying (the first corner's), a value interpolated across
    /// another triangulation, and the order in which a patch's triangles reach the blend or the
    /// depth test may differ: every patch measured differs on the RTX 3090, 55-95% of them on RADV
    /// and lavapipe. MoltenVK on an Apple GPU hands the patches to Metal's own tessellator and
    /// records none; so does a pipeline on a device without `tessellationShader`, which draws
    /// Metal's triangles in Metal's order through the emulation. Recorded on the evaluation stage.
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
    /// A sample at an implicit level or with gradients through a constexpr sampler with
    /// anisotropy, no mip filter (`mip_filter::none`) and different minification and magnification
    /// filters, on a device that samples through Metal's own sampler (MoltenVK on Apple's GPU,
    /// `DriverDeviation::MetalSampler`). Vulkan has no sampler without mipmapping: the device's
    /// sampler takes a nearest mip filter with the level clamp `[0, 0.25]`, which keeps Metal's
    /// choice of filter, but at `maxAnisotropy` 2 to 16 no clamp reads Metal's `NotMipmapped`
    /// footprint (`metal::sampler::vk_sampler`). **Residual:** the sample filters another
    /// anisotropic footprint than Metal's. A sampler state the runtime makes of that shape is
    /// announced as it is made instead. Validation derives no tolerance from it.
    NotMipmappedAnisotropicFilter,
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
    /// `GraphicsState::alpha_to_coverage`) on a device not measured to cover as Metal does
    /// (without
    /// [`DriverDeviation::MetalAlphaCoverage`](crate::error::DriverDeviation::MetalAlphaCoverage),
    /// which only MoltenVK on Apple's GPUs of the Apple M3 Max's macOS 27 identity takes: there
    /// Metal covers). The device converts the
    /// alpha to its coverage mask by its own algorithm. Metal's, measured over 639 alphas a sample
    /// count on an Apple M3 Max under macOS 27.0.1 (`runtime_lane::alpha_coverage`), covers a pixel
    /// at one sample where its alpha is above 1/2, and at two and four samples
    /// `ceil(alpha * n - t)` of its `n` samples, `t` a 2x2 ordered dither of the pixel's window
    /// coordinates (1/8 and 5/8 at even and odd `x` of an even row, 7/8 and 3/8 of an odd one),
    /// either count within an `f32` ulp of a threshold. Vulkan fixes no algorithm ("No specific
    /// algorithm is specified for converting the alpha value to a temporary coverage mask. It is
    /// intended that the number of 1's in this value be proportional to the alpha value (clamped
    /// to \[0,1\]), with all 1's corresponding to a value of 1.0 and all 0's corresponding to 0.0.
    /// The algorithm may be different at different framebuffer coordinates."), so a pixel may
    /// cover other samples than Metal's: at one sample, a whole fragment stored or dropped.
    /// Recorded on the fragment stage.
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
    /// A fragment function reads a color attachment the pipeline blends, in the interlocked
    /// storage form (`lower::fragment::fetch`, on a device that orders framebuffer fetch only
    /// through the pixel interlock), of a format whose blend arithmetic of Metal's the translation
    /// does not build: a format held directly whose channels read as floats, but a 16-bit float
    /// one, an sRGB one and an extended-range one (a 32-bit float, 16-bit normalized, 8-bit signed
    /// normalized, 10-bit, packed float or `A8Unorm` format, among others), an 8-bit unorm one
    /// written by an output of fewer than four lanes, and a 16-bit float or 8-bit unorm one under a
    /// blend of the second source or beside a second (index 1) output. No Vulkan blend applies to
    /// a storage write, so the shader computes the pipeline's blend itself: in `float`, each
    /// factor, product and sum
    /// rounded to nearest even, a factor of zero keeping its product, the source and the blend
    /// color clamped first to a normalized format's range (`[0, 1]`, `[-1, 1]`; a NaN to its low
    /// end) as Vulkan's blend clamps them, an output's missing lane zero, the result stored through
    /// the device's conversion. Metal's blend arithmetic for these formats is not measured, so the
    /// recorded bound is not known; a blend in `float` of a 32-bit float format agrees with one
    /// that fuses a product within that product's rounding, and one of a format of fewer bits than
    /// a `float` within the format's own rounding of the result where Metal's arithmetic is at
    /// least as fine. A factor of the second source reads the function's second output, clamped as
    /// the source. An 8-bit sRGB attachment (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`,
    /// `BGRA8Unorm_sRGB`) under a blend of the second source, a one- or two-channel one under a
    /// lerp, each in a module compiled with fast math, and one written by an output of fewer than
    /// four lanes, record it too: there the shader computes Metal's 8-bit unorm blend's rounded
    /// forms over the decoded texel, in the output's type (a missing lane zero), and stores the
    /// result by Metal's sRGB encode (`extended_range::Blend::srgb8`), which is Metal's without
    /// fast math at four lanes; under fast math Metal regroups the blend, unmeasured.
    /// Validation derives no tolerance from it.
    UnmeasuredFetchBlend,
}

/// The math mode of an entry's module, which decides, on a device whose `float` functions are
/// Metal's own (MoltenVK), how Metal compiles the entry's functions and its fast divisions, and
/// so which of its operations record a precision note: a module declares `SignedZeroInfNanPreserve`
/// where a `float` operation of the entry lacks one of the `nnan`, `ninf`, and `nsz` flags (an
/// entry Metal compiled without fast math) and the device supports the mode at the operation's
/// width. [`PrecisionNote::of_float_operator`] and [`PrecisionNote::of_math_call`] read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EntryMath {
    /// The module declares no `SignedZeroInfNanPreserve`: Metal compiles it with fast math, its
    /// fast division and its fast functions Metal's own.
    Fast,
    /// The module declares `SignedZeroInfNanPreserve`: Metal compiles it with its precise
    /// functions and a correctly rounded division.
    Precise,
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
