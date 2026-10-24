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
    /// The color attachments whose float writes this fragment stage rounds to the codes Metal
    /// stores in the shader (`StageIo::rounded_colors`), sorted: a normalized, sRGB or
    /// half-float attachment on a device whose own conversion is not Metal's. Each such write
    /// costs the stage's instructions per channel; empty for every other stage.
    pub rounded_colors: Vec<u32>,
    /// The notes of `precision_notes` that hold only under some specializations: a note whose
    /// every access keeps its near-equivalent only on the arms the module selects by its
    /// specialization constants (the bound format's class, `SpecKey::TexelFormat`; the bound
    /// sampler's word, `SpecKey::SamplerState`), with the condition under which a specialized
    /// module runs such an arm. Sorted by note, each note at most once. Empty for a module whose
    /// every note holds under every specialization. [`StageModule::precision_notes_under`]
    /// gives the notes of the module as a pipeline specializes it.
    pub specialized_notes: Vec<SpecializedNote>,
    /// The color attachments this fragment stage writes as the bits of the half Metal stores,
    /// through a 16-bit output, rounding none of their channels (`StageIo::half_bit_colors`),
    /// sorted: a `half` output to a half-float attachment on a device that stores a 16-bit
    /// output's bits but not a 32-bit output's NaN as Metal does. Each such write costs the stage
    /// a few instructions per output; empty for every other stage.
    pub half_bit_colors: Vec<u32>,
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
    /// functions, or the translation's routines where a device's instruction may leave the bound
    /// (and, for the functions no routine covers yet, the device's precision). Not recorded by
    /// `sqrt`, `rsqrt`, and `fmod`, precise, whose answer is the one right value on every device,
    /// nor, on a device whose `float` functions are Metal's own (MoltenVK), in an entry Metal
    /// compiles with fast math ([`EntryMath::Fast`]) by a call the device answers as Metal's own
    /// function; there its fast `sinpi`, `cospi`, and `tanpi`, near-equivalents, and the
    /// translation's own routines still record it.
    DeviceTranscendentals,
    /// Float reductions use the device's association order.
    ReductionOrder,
    /// `float` denormals follow the device: Metal reads an `f32` denormal operand of its float
    /// arithmetic, comparisons, and functions as the zero of its sign and flushes a denormal result
    /// to one, where a Vulkan device keeps or flushes them by its own float controls (this
    /// translation declares no `DenormFlushToZero`). Recorded only where the module leaves an
    /// operation's denormals to the device: the instructions of native device maths
    /// (`FloatMathsPolicy::Native`, `fast::fmod`'s fused multiply-add among them) and of the
    /// device's order (`RoundingOrderPolicy::Native`: `mix` by `FMix`, `fast::sqrt` by `Sqrt`,
    /// and `dot` of vectors of other than two to four lanes, whose chunks' products the device
    /// sums where Metal flushes each one below `2^-126`), beside their own notes; and, under every
    /// policy, a conversion to or from `double`, whose reading Metal's is not measured. Every other
    /// `float` operation reads its denormals as Metal does on every device (the device's where the
    /// profile takes the measured `DriverDeviation::MetalFloatDenormals`, an exponent-field select
    /// elsewhere): `+ - *`, the comparisons, the divisions, `fma`, `mix`, `dot`, `sqrt`, `rsqrt`,
    /// `floor` and `ceil`, `ldexp` (on its operand's bits), the minimum and maximum functions
    /// (`llvm.minnum` and `llvm.maxnum` among them, with any flags), and the functions the
    /// translation computes itself or the device answers as Metal's own; `mix`, `atan2`, `sinpi`,
    /// `cospi`, `tanpi` and `fast::fmod` in both modes, and two chained `fma`s, answer Metal's
    /// words on an Apple M3 Max under macOS 27.0.1 at and around denormal operands,
    /// but where the device's fused multiply-add under native device maths rounds a result whose
    /// exact value is below `2^-126` up to `2^-126`, which Metal flushes (docs/FAST-PATHS.md,
    /// known gaps). Validation derives no tolerance and no explanation from it: every `float`
    /// computation is exposed to it.
    DeviceDenormals,
    /// Pixel-coordinate samplers scale coordinates in the shader, rounding differently.
    /// Recorded for an access through a runtime sampler slot whose word may state pixel
    /// coordinates, and for an access through a constexpr pixel sampler that reads Metal's texels
    /// from none of: an exact fetch, the centre of the texel Metal floors its coordinate to, a
    /// gather snapped from the coordinate as given, a linear sample's emulation or footprint step;
    /// which leaves a linear sample the emulation does not reach on a device without
    /// `VkProperty::MetalSampler`. A runtime slot's sample, a comparison's included, is exact too
    /// where the bound sampler filters nearest (it reads that centre), and its gather of a 2D
    /// texture or array snaps from the coordinate as given behind the sampler's word; its linear
    /// filters are the device's (on a device with the property snapped onto the footprint cell of a
    /// word stating a linear pixel sampler). A cube's pixel sampler is a normalized one. An
    /// argument buffer's sampler records [`PrecisionNote::ArgumentPixelSamplerScaling`] instead.
    PixelSamplerScaling,
    /// Float division, remainder, square root, fused multiply-add, and dot products use Vulkan's
    /// precision bounds (e.g. 2.5 ULP division, a possibly unfused `Fma`) rather than correct
    /// rounding. A fast (`arcp`) division's reciprocal, which Metal rounds within about 0.64
    /// ulp but not correctly, is the device's or, where the profile has `OpFmaKHR`, `RN(1 / y)`,
    /// which differs from Metal's by an ulp near rounding midpoints. Not recorded by `sqrt` and
    /// `fmod`, precise, whose answer is the one right value on every device, nor, on a device whose
    /// `float` functions are Metal's own (MoltenVK), in an entry Metal compiles with fast math
    /// ([`EntryMath::Fast`]), by a fast division, which is Metal's own there, or by a call the
    /// device answers as Metal's own function.
    DeviceArithmetic,
    /// Float operations LLVM requires to keep signed zeros, infinities, and NaNs run without
    /// `SignedZeroInfNanPreserve`, which the device does not support at their width. Under
    /// `FloatMathsPolicy::Exact` so does an operation that answers Metal's zero of its sign: a
    /// fused multiply-add the translation computes itself (in `double` or in integers without
    /// `OpFmaKHR`, and a `half` one through `float`), one by `OpFmaKHR` whose result is at most
    /// `2^-126` in magnitude, and a `float` product flushed below `2^-126`. Without the mode the
    /// device may answer such a zero with either sign.
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
    /// (fused or not, as the implementation chooses). The AIR's value is any of the rewrites those
    /// flags allow, and Metal's compiler and the device's may each take a different one, so their
    /// roundings may differ.
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
    /// - An sRGB sum of an argument-buffer (heap) element, whose format is never known: Metal
    ///   decodes each tap's code through its twelve-bit table and sums the steps exactly; the
    ///   emulation sums the device's decoded values in `f32`, which Vulkan does not bound
    ///   ([`PrecisionNote::SrgbTexelDecode`]). A texture slot bound in an 8-bit sRGB format reads
    ///   the raw codes under its `SrgbTexture` variant and sums the steps exactly as Metal does,
    ///   on the grids `1/4095` (color) and `1/(16 · 255)` (alpha), so only the first bullet's
    ///   2D residue applies to it. The packed normalized formats and `Depth16Unorm` sum in `f32`
    ///   (their exact rule is not measured).
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
    /// has none). The module answers all of that exactly for reads, on every device, and for a
    /// sample or gather inside Metal's measured footprint rule (a constexpr sampler) the word too.
    /// A pixel fetch, and a nearest sample or a gather inside the rule, read Metal's value too:
    /// where the bound texture is a sparse image, a texel the word says is unmapped reads zero, and
    /// the lanes Metal reads as constants (a channel the format lacks, through the texture's
    /// swizzle) are set after the access (`SpecKey::ConstantLanes`); none of them is noted. What
    /// stays the device's is noted: a filtered sample's lanes that are no constants over a texel in
    /// a block of a sparse image that holds no mapped tile (zero on a device with
    /// `residencyNonResidentStrict`; any value elsewhere, which the runtime announces as
    /// `non-strict-residency`), and outside the footprint rule (a runtime sampler, mip levels,
    /// repeat or mirrored addressing, an offset) the word, the device's `OpImageSparse*` residency,
    /// whose footprint may count a texel of zero weight Metal does not (Vulkan's linear 2x2
    /// footprint does, measured on lavapipe), and those lanes of a fetch-like access. Where the
    /// profile does not enable `shaderResourceResidency` nothing is noted: the bound image holds
    /// Metal's unmapped value, and the accesses its tile map answers are exact.
    SparseResidency,
    /// A `float` NaN result's bit pattern follows the device: Metal answers the canonical
    /// `0x7fc00000` for the NaN results of its float arithmetic and functions, and Vulkan states no
    /// encoding, so a device may answer another sign or payload (llvmpipe: `0xffc00000`). Recorded
    /// wherever the module leaves a `float` operation that can produce a NaN to the device; not by
    /// the functions that select Metal's NaN themselves, nor by Metal's precise and fast `float`
    /// divisions. Also recorded by every read, sample, gather, and pixel fetch of a `half` texel:
    /// Metal hands back the texel's own NaN, sign and payload, and Vulkan states no encoding for a
    /// 16-bit float texel's decode (MoltenVK answers the canonical NaN for every one); on a device
    /// that rounds a `half` through its conversion under the float controls (ANV, RADV), the NaN
    /// that conversion makes of the device's decode, by default (`NanMinMaxPolicy::Native`). Also
    /// recorded by a `half` texel's write to a storage texture that may hold a float format: Metal
    /// stores a 16-bit float texel's NaN as the half's own bits and a 32-bit one's as `0x7fc00000`,
    /// and the device converts the half widened. Also recorded by a 16-bit float attachment's blend
    /// the function computes in the storage form, on a device not measured to store `0x7fc00000` in
    /// a 16-bit float storage image as Metal's `0x7e00` (`DriverDeviation::MetalFloatStorageNans`):
    /// the blend gives every NaN as `0x7fc00000`, and the device's store converts it (ANV stores
    /// `0xfe00`). Also recorded by a `float` read of a heap's texture buffer on a device without
    /// Metal's sampler: Metal hands back a half-float texel's NaN as `0x7fc00000`, and the handle
    /// carries no class to convert it by. Validation derives no tolerance and no explanation from
    /// it, as for `DeviceDenormals`.
    DeviceNanEncoding,
    /// The device compares a depth texture's texels (a sample or a gather with comparison the
    /// shader does not make itself) on a device without `VK_EXT_depth_range_unrestricted` enabled
    /// whose driver is not measured to keep depth values outside `[0, 1]` without it (narrowed to
    /// comparisons, and to the device's). Metal's `Depth32Float` texels hold any
    /// `float`, and the device's comparison reads them from a depth image: the texture's own, or
    /// the depth twin of a texture the consumer holds in an `R32_SFLOAT` image. Vulkan leaves a
    /// value outside `[0, 1]` uploaded there undefined without the extension
    /// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and a device may store it clamped (RADV on a
    /// Radeon 780M and ANV read 1000.0 back as 1.0, measured) or keep it (MoltenVK and lavapipe
    /// keep it, measured), so a comparison whose texel and reference both lie outside `[0, 1]` on
    /// the same side can differ. Values within `[0, 1]`, and every texel on a device with the
    /// extension enabled or a driver measured to keep such values without it
    /// (`VkProperty::DepthImageKeepsValuesOutsideUnitRange`, which such a module requires
    /// instead), are exact. A function that reads, samples or gathers a depth texture
    /// without comparing, or compares the texels it read without comparison itself (a nearest
    /// comparison, a comparison gather, a linear comparison the emulation reaches), records
    /// nothing: its interface's `TextureSlot::access` states the use, and a consumer that uploads a
    /// value into a depth image its device may clamp announces that upload (docs/API.md section 5).
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
    /// Not recorded where the module takes Metal's footprint itself through a constexpr sampler
    /// (`lower::sampler::anisotropic`, on a device without Metal's own sampler and on one whose
    /// stage's cube-gradient route is not measured): Metal's level, and an anisotropic sampler's
    /// taps (`PrecisionNote::AnisotropicFilter`). Where the route is not measured
    /// (`CubeGradientRoute::Unmeasured`), every other sample hands the isotropic form, which
    /// every measured route reads alike but on a `y` face under macOS 27, through a sampler made
    /// isotropic for that use (`metal::sampler::SamplerUse::cube_gradient`): an anisotropic
    /// sampler's footprint is then Metal's isotropic one, its major axis's, but where a runtime
    /// sampler slot's variant takes Metal's taps.
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
    /// yet. On macOS 26.7's identity (`CubeGradientRoute::AppleSwizzled`) the route hands that GPU
    /// its own footprint of that release, which counted the normal and the tangent `(a + 2) mod 3`.
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
    /// The module reads, gathers, or samples a texture with a `half` result. Metal converts the
    /// texel to `half` by the bound format: measured for the 32-bit float, half-float, and 8- and
    /// 16-bit normalized formats, which a texture slot converts exactly by its `TexelFormat`
    /// specialization (`metal::texel_format`); a read, nearest sample, or gather of a slot bound in
    /// an 8-bit sRGB format decodes the raw code exactly instead, and an emulated linear or a
    /// bicubic sample sums the raw codes' steps exactly, under the slot's `SrgbTexture` variant,
    /// and records nothing. A texture buffer converts by its bound format the same way (Metal reads
    /// a texture buffer's texels as a texture's: the P2a probe). A texture bound in another format
    /// (sRGB otherwise, packed, depth, and the rest) and a heap element keep the exact rounding of
    /// the device's `f32` texel to nearest even (`narrow::round_texel`), which Metal's conversion
    /// may not be: Metal rounds a 32-bit float sample toward zero, and an 8- or 16-bit normalized
    /// one onto its format's grid first. What a filtered sample converts is the filter's value
    /// ([`PrecisionNote::HalfSampleFilter`]). Recorded wherever such an access is lowered;
    /// validation derives no tolerance from it.
    ///
    /// Native device sampling (a texel conversion): under the default
    /// `TexelConversionPolicy::Device`, off Apple's GPUs (`VkProperty::MetalSampler` keeps the
    /// conversion), every slot's access with a `half` result takes that rounding too, and records
    /// this note; under `TexelConversionPolicy::Exact` a sample or gather that native device
    /// sampling (`SamplingPolicy::NativeDevice`) answers does. The recorded bound: the device's
    /// `float` texel (within [`PrecisionNote::DeviceTexelConversion`]'s bound of Metal's) rounded
    /// to nearest even is within one `half` ulp of Metal's `half` result (Metal rounds a 32-bit
    /// float sample toward zero, a half-float linear sum with ties away from zero, and a normalized
    /// value onto its grid first), but for a 32-bit float texel's sample past `±65504`, which Metal
    /// saturates to `±65504` and the rounding takes to the infinity of its sign; a NaN is the
    /// device's decode ([`PrecisionNote::DeviceNanEncoding`]).
    ///
    /// On a device that samples through Metal's own sampler under
    /// [`YCbCrFilterPolicy::Device`](crate::device::YCbCrFilterPolicy), a sample with a `half`
    /// result through a runtime sampler slot bound to a two-plane YCbCr texture on the grid `8192`
    /// (the `_sRGB` twins, the unpacked 12-bit and the extended-range formats): where its word
    /// reads one texel, the texel's step rounds to nearest even, where Metal rounds it with ties
    /// away from zero, one `half` unit in the last place apart at a tie. The runtime announces it
    /// at a draw or dispatch that binds such a texture there.
    HalfTexelConversion,
    /// The module samples a texture with a `half` result through a filter (linear minification
    /// or magnification, or a runtime sampler) on a device that does not sample through Metal's
    /// own sampler (one that does, `VkProperty::MetalSampler`, filters as Metal does: the module
    /// requires it instead). Metal's linear filter (the footprint's 1/256 fraction rounded half up,
    /// the taps weighed `(256 - k) / 256` and `k / 256`, each format's sum rounded as
    /// `metal::texel_format` states, a 32-bit float or half-float format's through Metal's linear
    /// accumulator: each term truncated toward zero onto a grid 38 or 17 bits below the largest
    /// term's tap and weight exponents, then summed exactly) is emulated by fetches of the base
    /// level (`lower::sampler::linear`) for a constexpr sampler with linear minification and
    /// magnification, no mip filter, no anisotropy, and weighted-average reduction, sampling a 1D
    /// or 2D texture slot (an array or not), or a 3D one (Metal's 3D weights, rounded to 16 bits,
    /// and its truncated normalized terms), with a constant texel offset or none, no
    /// `min_lod_clamp`, and no observed residency, under every address mode. It is exact for a
    /// bound format whose conversion is measured (`metal::texel_format`; the accumulator on every
    /// lane of the P2a and P2c probes; a normalized format's 2D sum truncates each term first, onto
    /// `2^-8` of a 16-bit code and `2^-4` of an sRGB step) but where it is known not to be:
    /// - a constant offset on a repeating or mirrored axis of a view whose first level lies above
    ///   its parent texture's base level: Metal drops the offset on such an axis of level size 2
    ///   to 7 where the parent's base-level size is not a power of two, and the emulation reads
    ///   the view's own base level, so at a view size of 2 or 4 it adds the offset Metal drops
    ///   (on every other texture the emulation places it as Metal does);
    /// - in 3D, a half-float format's result, half a `half` ulp off on about 0.3% of channels, and
    ///   a 32-bit float format's on about one lane in 180,000 (a cancellation).
    ///
    /// A slot bound in an 8-bit sRGB format takes the emulation under its `SrgbTexture` variant: it
    /// fetches the raw codes and sums their table steps exactly on the grids `1/4095` (color) and
    /// `1/(16 · 255)` (alpha), rounded half up, then to `half` to nearest even.
    ///
    /// Every other filtered `half` sample is the device's filter, then Metal's conversion of its
    /// value: through a runtime sampler; with a mip filter, anisotropy above 1, min or max
    /// reduction, or minification and magnification filters that differ; of a cube texture or an
    /// argument-buffer (heap) element; with a `min_lod_clamp` or a dynamic texel offset;
    /// with an observed residency; with one axis clamping to zero beside another clamping to a
    /// colored border; and of a slot bound in a format whose conversion is not measured
    /// (`TexelFormat::OTHER`: the packed and other formats, and an sRGB one outside the
    /// emulation's reach, which also records [`PrecisionNote::SrgbTexelDecode`]). There the
    /// device's footprint,
    /// weights, and sum (Vulkan bounds them only by its precision rules, such as
    /// `subTexelPrecisionBits`) may differ from Metal's, and the conversion rounds another value.
    /// A sample through a constexpr sampler with nearest minification and magnification, no mip
    /// filter or a nearest one, no anisotropy, and weighted-average reduction, and a gather, are
    /// not filtered and do not carry it (a linear mip filter, anisotropy, or a min or max
    /// reduction filters even with nearest minification and magnification). The `float` result's
    /// counterpart is [`PrecisionNote::FloatSampleFilter`]. Validation derives no tolerance from
    /// it: the class is unbounded.
    HalfSampleFilter,
    /// The module reads, samples, or gathers a texture with a `half` or `float` result through
    /// the device's sRGB decode, or decodes the device's filter of its codes, where the texture may
    /// be bound in an 8-bit sRGB format. Metal
    /// decodes such a texel's color channels through its twelve-bit table (`k / 4095` for the
    /// code's step `k`) and its alpha as `c / 255`, which the device's decode (Vulkan's transfer
    /// function, within its precision rules) does not reproduce. A texture slot the shape binds in
    /// such a format (`metal::texel_format::srgb_raw`) joins the plan's variant
    /// (`RuntimeFact::SrgbTexture`), under which its reads, nearest samples (a constexpr sampler
    /// with nearest filters, no mip filter or a nearest one, no anisotropy, and weighted-average
    /// reduction), pixel-coordinate fetches, and gathers (a `float` one's where the device does not
    /// sample through Metal's own sampler) decode the raw code exactly as Metal does, and its
    /// bicubic samples, and its linear samples with a `half` result that the emulation reaches
    /// ([`PrecisionNote::HalfSampleFilter`]), sum the raw codes' steps exactly as Metal does, as do
    /// its linear samples with a `float` result that the emulation reaches
    /// ([`PrecisionNote::FloatSampleFilter`]). Recorded where an access keeps the device's decode
    /// on a device that does not sample through Metal's own sampler: under that variant, a sample
    /// of the slot through a runtime sampler or another filter (a runtime sampler's filter is not
    /// known when the function compiles); and any such access of an argument-buffer (heap) element,
    /// whose format is never known. On a device that does (`VkProperty::MetalSampler`: MoltenVK on
    /// Apple's GPU) the device's decode is Metal's, and the module requires the property instead.
    /// Where the device cannot sample the bound format, whose texture it holds as the format's
    /// UNORM twin (`LoweringInputs::srgb_twins`), such an access samples the twin's codes and
    /// decodes the device's filtered value through the table: Metal's value where the filter reads
    /// one texel (a nearest sampler's), Metal's sum where it blends two adjacent codes, otherwise
    /// the decode of the codes' blend, where Metal blends the decoded texels; recorded on every
    /// device under that variant. Validation derives no tolerance from it: the class is unbounded.
    SrgbTexelDecode,
    /// The module samples a texture with a `float` result through a filter on a device that does
    /// not sample through Metal's own sampler (one that does, `VkProperty::MetalSampler`, filters
    /// as Metal does: the module requires it instead). A sample is filtered unless it goes through
    /// a constexpr sampler with nearest minification and magnification, no mip filter or a nearest
    /// one, and no anisotropy (a minimum or maximum reduction reads the weighted average, as
    /// Metal's does on Apple's GPU): linear minification or magnification, a linear mip filter,
    /// anisotropy, and a runtime sampler (whose filter is not known when the function compiles) all
    /// filter, a depth texture's sample and a comparison's included. Metal's linear filter (the
    /// footprint's 1/256 fraction rounded half up, the taps weighed `(256 - k) / 256` and `k /
    /// 256`, each format's sum rounded as `metal::texel_format` states: an 8- or 16-bit normalized
    /// or 8-bit sRGB format's exactly (a 16-bit format's 2D terms truncated onto `2^-8` of a code
    /// and an sRGB format's onto `2^-4` of a step first), rounded half up onto its grid, then `n /
    /// D` to `f32`; a 32-bit float format's rounded to `f32` with ties away from zero; a half-float
    /// format's to `half` with ties away from zero, widened; both through Metal's linear
    /// accumulator) is emulated by fetches where the `half` result's is
    /// ([`PrecisionNote::HalfSampleFilter`]), for every measured format, with the same exceptions:
    /// a constant offset on a repeating or mirrored axis of a view whose first level lies above its
    /// parent's base level, and in 3D a half-float format's half-ulp residue and a rare 32-bit
    /// float cancellation. Every other filtered sample is the device's filter, whose footprint,
    /// weights, and sum Vulkan bounds only by its precision rules (such as
    /// `subTexelPrecisionBits`): a device whose weights or sum differ answers another value, even
    /// for a blend at weight 1/2. Validation derives no tolerance from it: the class is unbounded.
    FloatSampleFilter,
    /// The module leaves a gather to the device's footprint on a device that does not sample
    /// through Metal's own sampler (one that does, `VkProperty::MetalSampler`, gathers as Metal
    /// does: the module requires it instead). Metal's 2D footprint (per axis the position `u * n`
    /// rounded once to `f32` with ties away from zero, a repeat or mirrored-repeat coordinate
    /// wrapped into its period first, then `q = floor(256 x - 127.5)` and the texels `q >> 8` and
    /// the one after it), measured at odd sizes on Apple M3 Max and M4 alike, is read by gathering
    /// at the footprint's corner, where every device reads Metal's 2x2 block, for a 2D or 2D-array
    /// gather (a comparison's included) through a constexpr sampler with normalized coordinates,
    /// and emulated by fetches for a cube gather (a depth cube's comparison included) through such
    /// a sampler. A 2D or 2D-array gather through a runtime or argument-buffer sampler snaps behind
    /// the sampler's word onto Metal's block by the address mode the word states on each axis,
    /// every mode measured, and records nothing. Recorded for every other gather: of a cube texture
    /// bound in a format whose corner rule is not measured (the module's gather branches to the
    /// device's there; its face at a tie of major axes is handed to the device). Such a
    /// gather is the device's, which Vulkan bounds only by `subTexelPrecisionBits`: a lane a few
    /// ulps from a texel edge can read the next 2x2 block. Validation derives no tolerance from it:
    /// the class is unbounded.
    DeviceGatherFootprint,
    /// The module hands a nearest cube sample the centre of the texel Metal reads, on a device
    /// that does not sample through Metal's own sampler (one that does, `VkProperty::MetalSampler`,
    /// is Metal's texture unit itself). Metal's position on a face axis of size `n` is
    /// `floor(RN(RN(1 + sc * r) / 2 * n))` (the cube rounding probes, Apple M3 Max and M4 alike),
    /// with `r` its texture unit's reciprocal of the major magnitude, which is not correctly
    /// rounded: on about 10% of majors (208 of 2,001 measured), those whose reciprocal lies within
    /// about 0.2 ulp of a rounding midpoint, it lies one or two ulps from `RN(1 / |ma|)`, and no
    /// exact form of it is known. The module computes the position with the correctly rounded
    /// reciprocal, so where the position lies within about `n * 2^-23` of a texel edge on such a
    /// major, the sample reads the next texel on that axis. A linear cube sample the module
    /// emulates by fetches takes the same position, so where it lies within about `n * 2^-23` of a
    /// 1/256 quantization step on such a major, the sample's fraction is one step off. Recorded for
    /// every nearest cube sample the module hands its texel and every linear cube sample it
    /// emulates. Validation derives no tolerance from it: the class is unbounded.
    CubeFaceReciprocal,
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
    /// The module leaves a depth comparison (a sample or gather with comparison) to the device, on
    /// a device that does not sample through Metal's own sampler (one that does,
    /// `VkProperty::MetalSampler`, compares as Metal does: the module requires it instead). Metal
    /// compares a `Depth16Unorm` texel's code with the reference's (the reference clamped into
    /// `[0, 1]`, its exact product with 65535 rounded to nearest), and a float format's texel and
    /// reference as given, denormals and values outside `[0, 1]` included, `-0` equal to `+0`; a
    /// NaN fails every ordered test and passes `not_equal`. The module makes that comparison
    /// itself, on texels it fetches exactly for a bound `Depth16Unorm` or `Depth32Float` texture: a
    /// linear comparison through a constexpr sampler the emulation reaches, a cube's comparison
    /// gather, and a 2D comparison gather or nearest comparison through a constexpr sampler at a
    /// level the function knows, and a runtime sampler's comparison gather and nearest comparison
    /// without a mip filter. Elsewhere it compares texels the device reads: that read is
    /// [`PrecisionNote::DeviceTexelDenormals`]'s (a device's sample or gather may read a denormal
    /// as zero), and the comparison is Metal's.
    ///
    /// Recorded where the device compares (a heap texture, an argument buffer's sampler, a filter
    /// the emulation does not reach, a runtime sampler's sample whose bound sampler filters
    /// otherwise): Vulkan pins only a normalized format's clamp of the reference. Of 9,024
    /// `Depth16Unorm` comparisons the device made (`depth_compares`), RADV (Radeon 780M) differed
    /// from Metal on 71 (a NaN reference compares as 0), ANV (Iris Xe) on 584 (its rounding of the
    /// reference to a code), the RTX 3090 on 474, x86 lavapipe on 1,203 and arm64 lavapipe on 1,191
    /// (both compare the clamped reference with the texel as floats); x86 lavapipe's float
    /// comparison also flushes a denormal `Depth32Float` value.
    ///
    /// Not this note's: a depth image that answers a denormal as zero (RADV's and ANV's), which
    /// the runtime replaces by a colour image wherever the driver is not measured to keep one; the
    /// device's read of the texels the module compares ([`PrecisionNote::DeviceTexelDenormals`]);
    /// and [`PrecisionNote::RestrictedDepthRange`]'s values outside `[0, 1]`.
    ///
    /// Validation keeps the row open (C3): no exact path replaces the device's comparison there
    /// yet.
    DeviceDepthComparison,
    /// A `float` sample or gather, a comparison's included, whose texels the device reads, on a
    /// device that does not sample through Metal's own sampler. Metal's sample and gather hand back
    /// a denormal 32-bit float texel (`R32Float`, `RG32Float`, `RGBA32Float`, `Depth32Float`) as it
    /// is. A device's sample or gather may read it as the zero of its sign
    /// (`runtime_lane::depth_compares`):
    /// - RADV (Radeon 780M): gathers and nearest samples;
    /// - NVIDIA (RTX 3090): gathers;
    /// - ANV (Iris Xe): nearest samples;
    /// - lavapipe: x86's float comparison.
    ///
    /// Each device's `OpImageFetch` and `OpImageRead` keep it. The module fetches the texels
    /// itself, behind the slot's `TexelFormat` specialization, where Metal's footprint and level
    /// are known:
    /// - a plain gather, or a nearest sample at a level the function knows, through a constexpr
    ///   sampler;
    /// - a comparison's;
    /// - a linear sample the emulation reaches;
    /// - a cube gather's.
    ///
    /// Recorded for every other such sample or gather:
    /// - an implicit level under a nearest mip filter in a fragment function;
    /// - a filtered sample no emulation reaches;
    /// - a `min_lod_clamp`, a dynamic offset, or axes clamping to zero and to a colored border;
    /// - an observed residency;
    /// - a cube's sample;
    /// - a heap texture;
    /// - a runtime or argument buffer's sampler.
    ///
    /// Not this note's: a depth image (a `Depth32Float` texture's own, or its twin) that a device's
    /// comparison reads, which RADV holds a denormal in as zero
    /// ([`PrecisionNote::DeviceDepthComparison`]).
    ///
    /// Validation derives no tolerance and no explanation from it, as for `DeviceDenormals`.
    DeviceTexelDenormals,
    /// The module is compiled for a device whose driver release is newer than the last one
    /// measured with a deviation the profile takes (`ProfileFields::deviations_beyond`, which
    /// names each with that release): where the specifications leave the answer to the driver,
    /// the module relies on its answer being Metal's (a simdgroup's lockstep, Metal's own `float`
    /// functions, its canonical NaN, its sampler, its cube-gradient route), as measured on the
    /// earlier release. The lane that measured each runs at every master gate on the venues of
    /// that driver, so a venue upgrade that changes one fails there; a release no venue runs is
    /// taken on that measurement alone. Recorded by every module compiled for such a profile,
    /// since the deviations shape `float` arithmetic, NaNs, sampling and simdgroups broadly.
    ///
    /// Validation derives no tolerance and no explanation from it, as for
    /// `FunctionStorageAssumed`: a difference stays a mismatch for review.
    BeyondMeasuredDeviation,
    /// The module places a texel offset on an axis that may repeat or mirror as Metal does in an
    /// access of an argument buffer's texture: on a device that does not sample through Metal's
    /// own sampler (one that does, `VkProperty::MetalSampler`, is Metal's texture unit itself), in
    /// a snapped gather or nearest sample, fetched texels, or an emulated linear filter, and on
    /// every device in an emulated raw sample (`heap_raw`). Metal drops the offset on a repeating
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
    /// A fragment function writes a float to a color channel whose conversion the device makes,
    /// not Metal's rule (`ChannelWrite::Device`): an attachment that blends with factors other
    /// than one and zero, the alpha of attachment 0 under alpha-to-coverage, a format whose
    /// conversion Metal was not measured with (the packed 16-bit unorm formats, `A8Unorm`, the
    /// packed floats). (A `bfloat` output is refused, as Metal's compiler rejects one.) Near a tie
    /// between two codes the device may store the other one (every Vulkan driver measured but
    /// MoltenVK rounds the `f32` product with ties to even and encodes sRGB by its own
    /// approximation, where Metal rounds the exact product and encodes through its twelve-bit
    /// table). Every other float color write rounds to Metal's code in the shader. A 32-bit float
    /// attachment that so blends stores the device's blend arithmetic (NVIDIA's 615.71.09 one `f32`
    /// ulp from Metal's in some source-over sums, a NaN as `0x7fffffff` where Metal stores
    /// `0x7fc00000`).
    ///
    /// Also: a function writes a `float` texel through an argument buffer's texture buffer (a heap
    /// texture buffer, which carries no texel class) that may hold a normalized format the device
    /// converts otherwise than Metal, which stores each channel's exact product rounded with a tie
    /// away from zero (a device without `DriverDeviation::MetalNormStorageWrites`, or with it where
    /// such a buffer may hold a 1-bit alpha). Every other `float` texel write to a normalized
    /// storage texture rounds to Metal's code in the shader.
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
    /// An access through an argument buffer's sampler (not a 2D gather, which snaps behind the
    /// sampler's word) whose word the specialized module cannot read may hand the device a
    /// pixel coordinate scaled by the texture's size: where the sampler states pixel coordinates
    /// and a linear filter or a comparison, on a device without `VkProperty::MetalSampler` (a
    /// filtered sample of a class the emulation does not reach), or a comparison on one with it.
    /// Measured exact: a nearest or linear sample of `RGBA8Unorm` and `RGBA16Float` through such a
    /// pixel sampler, clamped to the edge, to zero, to a border or mirrored once, on MoltenVK and
    /// lavapipe, so a device with `MetalSampler` records it only for a comparison. It is its own
    /// note, apart from [`PrecisionNote::PixelSamplerScaling`], because only the bound states
    /// decide it: the runtime announces it at the first draw or dispatch of a pipeline that records
    /// it while the device lists a sampler state of pixel coordinates with a linear filter or a
    /// comparison, never at the pipeline's making.
    ///
    /// Validation derives no tolerance from it: the class is unbounded.
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
    /// ANV and lavapipe, Mesa 26.2.4, the 14 blends Metal measured over the two sweeps); and 1 or 2
    /// codes of an 8-bit or extended-range format. MoltenVK on the M3 Max stores Metal's bytes in
    /// every one of the draws.
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
    /// `_sRGB` twin) where Metal may round the blend's source otherwise than the function's
    /// computed blend: under fast math a lane of the output is not a load; or the module names no
    /// math mode (a module without fast math, safe or relaxed math, too previously, which computes
    /// the blend Metal rounds there). Metal's compiler folds the blend into the function, so how it
    /// rounds the source follows how the function computed it
    /// (`lower::arith::output_precision::note_extended_range_sources`): a blended channel may store
    /// a value one code apart from Metal's. A `float` output records
    /// [`PrecisionNote::ExtendedRangeFloatBlendSource`] instead (this note covered both before), so
    /// a function of both records both.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
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
    /// of AIR's `float`): blended into an extended-range attachment where Metal may round the
    /// blend's source otherwise than the function's computed blend, under fast math a lane of the
    /// output that is not a load, or in a module that names no math mode. A blended channel may
    /// store a value one code apart from Metal's. Measured on an Apple M4 under macOS 27.0.1
    /// under fast math: a product of loads differs by a code under six of 21 blends (on 35 to 432
    /// of 4,096 texels), and of 26 computed `float4` shapes Metal's compiler rounds a third
    /// otherwise again; the consumer's measurement of computed `float4` outputs under fast math
    /// found a one-code difference under 17 of 66 blends. Its own note, apart from the `half`
    /// output's, so a consumer can tell the two apart.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    ExtendedRangeFloatBlendSource,
    /// The module samples a texture through an anisotropic sampler with Metal's anisotropic
    /// filter emulated by the device's samples at explicit levels, on a device that does not
    /// sample through Metal's own sampler (`lower::sampler::anisotropic`): a sample without
    /// comparison of a 2D texture or 2D array at the fragment's implicit level or with explicit
    /// gradients, or of a cube or cube array with explicit gradients, through a constexpr sampler
    /// of `maxAnisotropy` above 1, or through a runtime sampler slot whose word states anisotropy
    /// (under that word alone). A cube's sample is emulated on Metal's own sampler too where the
    /// stage's cube-gradient route is not measured (`CubeGradientRoute::Unmeasured`), its taps
    /// Metal's own samples at explicit levels; and so is an anisotropic sampler's on a device
    /// without `samplerAnisotropy` whatever its sampling, every Vulkan sampler being isotropic
    /// there, each tap the device's isotropic filter. Metal's footprint (its gradient lengths,
    /// piecewise-linear `log2`, level, tap count, tap weights and direction) is reproduced as
    /// measured on an Apple M4 (the M3 Max answering identically), with this residue:
    /// - the level is off by `1/256` of a level on about 1% of anisotropic samples;
    /// - the tap spacing is Metal's at the ratio cap and for two taps (but where the ratio lies
    ///   within `0.07/256` of a rounding tie, on about 5% of two-tap footprints, by up to 1.8/256
    ///   of a texel), and a fitted approximation for three or more taps below the cap (exact on
    ///   1,021 of 1,674 measured windows; up to about 18/256 of a texel at the outer tap of 10 or
    ///   more taps);
    /// - each tap is the device's own filter at the level the shader hands it, so its footprint
    ///   and sum are the device's (`HalfSampleFilter`, `FloatSampleFilter`);
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
    /// device that does not sample through Metal's own sampler: a cube, a 3D texture, a
    /// comparison, a sample whose residency the function observes, or a texture of another
    /// class than its declared one, at an implicit level or with explicit gradients, through a
    /// constexpr sampler of `maxAnisotropy` above 1 or a runtime sampler slot whose word states
    /// anisotropy (under that word alone). The device filters with its own anisotropic rule,
    /// whose footprint, level, tap count and weights Vulkan leaves to the implementation: on
    /// `rgba8Unorm` 2D textures at `maxAnisotropy` 8 with a linear filter, ANV (Iris Xe), RADV
    /// (Radeon 780M) and an RTX 3090 miss Metal by up to 28, 60 and 62 codes (mean 6.6, 6.2 and
    /// 12.4); other shapes are not measured. A sampler with a nearest min or mag filter is handed
    /// no anisotropy off Metal's sampler, so there the device filters isotropically, and so does
    /// every sampler on a device without `samplerAnisotropy`, whatever its sampling (the device's
    /// own sampling recording it there too). An argument buffer's sampler, whose anisotropy the
    /// function does not know, records `HalfSampleFilter` or `FloatSampleFilter` alone, and is
    /// refused anisotropic without `samplerAnisotropy`. Validation derives no tolerance from it.
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
    /// The module hands an 8-bit unsigned normalized texture's linear sample to the device's
    /// filter, at the coordinate of Metal's footprint step (`lower::sampler::native`), on a
    /// device measured to filter that class within one code of Metal
    /// (`DriverDeviation::Unorm8LinearFilter`, which the module requires as
    /// `VkProperty::Unorm8LinearFilter`): a sample with a `float` or `half` result of a 2D texture
    /// or 2D array slot bound in `R8Unorm`, `RG8Unorm`, `RGBA8Unorm` or `BGRA8Unorm`, without
    /// comparison, offset or pixel coordinates, through a constexpr sampler Metal's linear rule
    /// answers or a runtime sampler slot whose word states one (under that class and word
    /// alone). Elsewhere the sample keeps Metal's filter, emulated exactly
    /// ([`PrecisionNote::FloatSampleFilter`]). The device reads Metal's taps and weights there;
    /// its blend of them is its own. Measured against an Apple M4 (the deviation probe: 4,096
    /// coordinates, a third near the 1/256 grid, under clamp to edge, repeat, mirrored repeat and
    /// clamp to zero), the largest difference of any channel is 0.849 of a code (1/255) on
    /// NVIDIA's 615.71.09 (RTX 3090, RTX 5080 Laptop GPU), 0.031 on RADV and ANV (Mesa 26.2.4,
    /// Radeon 780M, Iris Xe of RPL-P, UHD 770 of RPL-S), and just under 1 on lavapipe (Mesa
    /// 26.2.4, whose filter answers whole codes), for a `float` result; for a `half` one, 0.809,
    /// 0.125 and 0.996. The owner accepted the one-code bound
    /// (2026-10-10). Validation derives no tolerance from it.
    DeviceUnorm8LinearFilter,
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
    /// Max against plain Metal over every blended draw of the sweep; the device's `float` blend
    /// modelled on the same draws, three units). A consumer may relay it as a known difference.
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
    /// Native device sampling: the module hands a sample, gather or comparison to the device's
    /// own sampler where a device whose sampler is not Metal's would emulate Metal's
    /// (`SamplingPolicy::NativeDevice`, the default; the module requires
    /// `VkProperty::DeviceSampling`). Its filter, footprint, level selection, anisotropy, decode
    /// and comparison are the device's. [`SamplingPolicy::Exact`](crate::device::SamplingPolicy)
    /// keeps Metal's, emulated exactly; a device that samples through Metal's own sampler
    /// (MoltenVK on Apple's GPU) never records it.
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
    /// A linear sample of a two-plane YCbCr texture is the device's filter of its converted texels
    /// under [`YCbCrFilterPolicy::Device`](crate::device::YCbCrFilterPolicy), the default: the
    /// texels hold Metal's reads exactly, but Metal sums their steps on the format's grid at its own
    /// weights, each term truncated. Against that sum (the slot's emulation under
    /// `YCbCrFilterPolicy::Exact`, which keeps it), at 65,536 random coordinates on each of five
    /// `YCBCR8_420_2P` surfaces of both ranges and three matrices, `float` and `half` results, the
    /// largest difference of any channel, in codes of 1/255, is 1.183 on NVIDIA's 615.71.09 (RTX
    /// 3090), 0.877 on RADV (Mesa 26.0.8, Radeon 780M and RX 7900 XTX), 1.058 on ANV (Mesa 26.2.4,
    /// Iris Xe of RPL-P) and 0.872 on lavapipe (Mesa 26.2.4); at coordinates on the 1/256 grid,
    /// 0.498, 0.125, 0.125 and 0.125.
    ///
    /// Validation derives no tolerance from it.
    DeviceSampling,
    /// Native device maths: the module computes a `float` or `half` math-library function by the
    /// device's own `GLSL.std.450` instruction (or Metal's composition of the device's `Exp2` and
    /// `Log2`), not by the translation's routine of Metal's answer, on a device whose functions
    /// are not Metal's own, under `FloatMathsPolicy::Native` (the default; the owner's ruling of
    /// 2026-10-11); and `fast::sqrt` the device's `Sqrt`, under `RoundingOrderPolicy::Native`.
    /// **Recorded bound:** Vulkan's precision table for the instruction, where Metal promises 4 to
    /// 6 ulp precise and its fast-math table fast (MSL Tables 8.1, 8.2): `Exp2` and `Exp` 3 + 2·|x|
    /// ulp; `Log2` and `Log` 3 ulp outside `[0.5, 2]` and `2^-21` absolute inside; `Pow` inherited
    /// from `Exp2(y · Log2(x))`; `Sqrt` inherited from `1 / InverseSqrt` (`fast::sqrt`'s too, where
    /// Metal's is `x · rsqrt(x)`); `InverseSqrt` 2 ulp; `Sin` and `Cos` `2^-11` absolute in `[-π,
    /// π]` (both answers in `[-1, 1]` beyond); `Tan` inherited from `Sin / Cos`; `Asin`, `Acos`,
    /// `Atan` and `Atan2` 4096 ulp; the hyperbolic functions inherited from their `Exp` and `Log`
    /// formulas. A denormal operand and a NaN's encoding are the device's (`DeviceDenormals`,
    /// `DeviceNanEncoding` where recorded). Validation derives no tolerance from it.
    NativeTranscendentals,
    /// Native device maths: a precise `float` or `half` division, or Metal's fast division or
    /// reciprocal, is the device's `OpFDiv`, not the translation's correctly rounded routine or
    /// its `RN(1 / y)`, on a device whose division is not Metal's own, under
    /// `FloatMathsPolicy::Native` (the default). **Recorded bound:** Vulkan's 2.5 ulp of the
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
    /// (on the `half` operands held in `float`, its result rounded to `half`),
    /// not the translation's correctly rounded routine, under `FloatMathsPolicy::Native` (the
    /// default). **Recorded bound:** a `half` result within one unit in its last place of
    /// Metal's (the product of two `half` values is exact in `float`, so only the two roundings
    /// differ); a `float` one within half a unit in the last place of the product plus half a
    /// unit in the last place of the result (Vulkan lets `Fma` round the product and then the
    /// sum). Validation derives no tolerance from it.
    NativeFma,
    /// Native device sampling (a texel conversion): the module hands a read, sample or gather
    /// with a `float` result the device's own conversion of the texel's codes, where Metal's is
    /// emulated exactly: a read's under `TexelConversionPolicy::Exact`, a sample's or gather's
    /// under it with `SamplingPolicy::Exact`. The default off Apple's GPUs
    /// (`VkProperty::MetalSampler` keeps Metal's value). Recorded for every such access of a
    /// texture slot or a heap texture (not a heap's texture buffer,
    /// [`PrecisionNote::DeviceNanEncoding`]). The recorded bound, of a texel whose access reads
    /// one texel of one level:
    /// - an 8- or 16-bit normalized code `c`: Metal's `c / (2^b - 1)` correctly rounded; the
    ///   device's within one `f32` ulp of it where measured (lavapipe multiplies by the
    ///   reciprocal: some 8-bit codes one ulp high, a 16-bit code and 512 `Depth16Unorm` codes one
    ///   ulp low);
    /// - a 5- or 6-bit channel of a packed format: the device may widen it by replicating its
    ///   bits (ANV, NVIDIA), within 0.18 of a code (`1 / (2^b - 1)`) where it replicates to 8
    ///   bits; a 4-bit channel's replication is exact;
    /// - a half-float or small-float texel's NaN: the device's encoding, where Metal answers
    ///   `0x7fc00000`;
    /// - every other class: the device's value, as under either policy.
    ///
    /// A filtered sample's value is the filter's under either policy. Validation derives no
    /// tolerance from it.
    DeviceTexelConversion,
    /// Native half rounding (a conversion): the module rounds a `half` result it computes in
    /// `float` (a `half` operation's, a function's, a conversion from `float`'s, a texel's) by
    /// the device's conversion of the `float` to 16 bits and its widening (`OpFConvert` of the
    /// value itself, with no barrier), not by the translation's integer rounding, on a device with
    /// `shaderFloat16` whose conversion is neither measured to be Metal's `half(float)` (MoltenVK
    /// on an Apple GPU) nor made Metal's by its float controls (`RoundingModeRTE`,
    /// `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits, under which ANV and RADV
    /// round through the conversion exactly and record no note), under
    /// `HalfRoundingPolicy::Native` (the default): NVIDIA, whose driver folds the conversion's
    /// round trip even under those controls (`DriverWorkaround::HalfConversionPairFolded`),
    /// lavapipe, and any device without them. Without them Vulkan leaves the conversion's
    /// rounding mode and its 16-bit denormals to the device. The recorded
    /// bound, against Metal's `half(float)` (to nearest even on the half grid, half denormals kept,
    /// infinity from 65520 up, `0x7E00` for a NaN):
    /// - a finite result within one unit in the last place of the `half`: NVIDIA's 615.71.09
    ///   rounds to nearest even on the RTX 5080 Laptop GPU (every `float` pattern, loaded and
    ///   computed) and toward zero on the RTX 3090; lavapipe (Mesa 26.0.8) a loaded value toward
    ///   zero;
    /// - or no rounding at all: NVIDIA's 615.71.09 on the RTX 3090 drops the conversion pair, with
    ///   or without a `NoContraction` barrier; lavapipe drops it in arithmetic on Mesa 26.2.4 with
    ///   LLVM 23.1.1 and for a loaded texel on Mesa 26.0.8 with LLVM 21.1.8, barrier or not, whose
    ///   arithmetic measured rounded. The result is then the `float` computation itself: its
    ///   differences from Metal's compound across operations, and a value past the `half`'s range
    ///   stays finite until a `half` store saturates it to infinity. The RTX 5080 Laptop GPU on the
    ///   same driver rounds here (8192 of 8192 sums Metal's);
    /// - on a device that rounds toward zero, a finite value from 65520 up is 65504 where
    ///   Metal's is infinity;
    /// - a result below the smallest normal `half` (`2^-14`) may be the zero of its sign on a
    ///   device that flushes 16-bit denormals (the RTX 5080 keeps them);
    /// - a NaN's encoding and the sign of a zero result are the device's: on the RTX 5080 a
    ///   computed `half` NaN is `0x7FFF` (where 18 C3 cases differed in NaNs, signs of zero,
    ///   `atan2` and `pow` at special operands, and `half` texel and pack conversions).
    ///
    /// Under the same policy a `half` the module reads from its bits (a load, a reinterpretation,
    /// a texel's bits) where no loop of the finished module holds the read is widened by the
    /// device (`UnpackHalf2x16`), not by the translation's integer decode, and a module that does
    /// so records this note too: its bound is the last two items', a half denormal read as the zero
    /// of its sign on a device that flushes 16-bit denormals and a NaN's payload the device's;
    /// every other `half` widens to its exact value.
    ///
    /// Validation derives no tolerance from it.
    NativeHalfConversion,
    /// Native half rounding (a quantize): the module rounds a `half` result it computes in
    /// `float` by the device's `OpQuantizeToF16`, a result the quantize flushes to zero rounded
    /// by the translation to the half denormal grid, not by its integer rounding, on a device
    /// without `shaderFloat16` whose quantize is not measured to round as Metal's conversion
    /// does, under `HalfRoundingPolicy::Native` (the default). Vulkan leaves the quantize's
    /// rounding mode to the device. The recorded bound, against Metal's `half(float)`: a finite
    /// result within one unit in the last place of the `half`; a value from 65504 up to 65520 may
    /// be infinity where Metal's is 65504 (NVIDIA's 615.71.09 rounds `0x477FEFFF` to infinity); a
    /// NaN's encoding is the device's. Validation derives no tolerance from it.
    NativeHalfQuantize,
    /// Native device maths: a `float` or `half` operation's NaN result is the device's, its sign
    /// and payload unselected, on a device whose arithmetic is not measured to answer Metal's
    /// canonical NaN, under `NanMinMaxPolicy::Native` (the default). **Recorded bound:** where
    /// Metal answers a NaN the device answers a NaN (the module keeps
    /// `SignedZeroInfNanPreserve` where Metal's entry keeps NaNs), its sign and payload the
    /// device's own (llvmpipe: `0xffc00000`), where Metal answers its canonical `0x7fc00000`
    /// (`0x7e00` as `half`), an operand's bits (`round`, `ldexp`, `fast::tan`), or the canonical
    /// NaN with an operand's sign (`asin`, `atan`, `asinh`, `atanh`). Every non-NaN result is as
    /// without it. Also recorded by a module that reads a `half` from its bits (a load, a
    /// reinterpretation, a texel's bits) by the device's widening under the float controls its
    /// `half` rounding declares (ANV, RADV): every `half` but a NaN widens to its value, and a NaN
    /// to a NaN whose bits are the device's, where Metal widens every `half` NaN to `0x7fc00000`; a
    /// `half` stored back unchanged keeps its bits. Validation derives no tolerance from it.
    NativeNanEncoding,
    /// Native device maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its
    /// `clamp(x, +0, 1)`) on `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or
    /// `NClamp`, not the translation's selects of Metal's operand, on a device not measured to
    /// answer Metal's, under `NanMinMaxPolicy::Native` (the default). **Recorded bound:** the
    /// result is one of the operands, the one Metal answers but at three ties: of two NaNs a NaN
    /// (its bits the device's), where Metal answers the second operand's; of `+0` and `-0` either
    /// zero, where Metal answers the second; and an `f32` denormal beside a zero or another
    /// denormal compared by its value where the device keeps denormals, where Metal compares it as
    /// a zero and answers the second. Validation derives no tolerance from it.
    NativeMinMax,
    /// Native device maths: Metal's `dot` on `float`, or on `half` where its fused multiply-add is
    /// not the measured 16-bit one, is the device's `OpDot` (a `half` one summed in `float` and
    /// rounded once to `half`), not Metal's fused chain of its products, under
    /// `RoundingOrderPolicy::Native` (the default). **Recorded bound:** a `float` `dot` of `n`
    /// lanes within `(3n - 1) * 2^-24 * sum |a_i * b_i|` of Metal's chain, plus `n * 2^-126` where
    /// a product is an `f32` denormal the device keeps (Vulkan's `OpDot` inherits `OpFMul`'s and
    /// `OpFAdd`'s rounding, `2n - 1` roundings, where Metal's chain rounds `n` times); a `half`
    /// one within `(n + 3) * 2^-11 * sum |a_i * b_i| + (n + 2) * 2^-25` of Metal's chain of `half`
    /// fused multiply-adds. Validation derives no tolerance from it.
    NativeDot,
    /// Native device maths: Metal's `mix(x, y, t)` on `float`, or on `half` where its fused
    /// multiply-add is not the measured 16-bit one, is the device's `FMix` (a `half` one computed
    /// in `float` and rounded once to `half`), not Metal's `fma(t, y, fma(-t, x, x))`, under
    /// `RoundingOrderPolicy::Native` (the default). **Recorded bound:** a `float` result within
    /// `2^-21 * (|x * (1 - t)| + |y * t|)` of Metal's (Vulkan's `FMix` inherits
    /// `x * (1 - t) + y * t`'s rounding, four roundings, where Metal's rounds twice); a `half`
    /// one within `3 * 2^-10 * (|x * (1 - t)| + |y * t|) + 2^-23`. Validation derives no
    /// tolerance from it.
    NativeMix,
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
    /// A sample whose level the device's level arithmetic chooses, on a device that samples neither
    /// through Metal's own sampler nor natively (which records [`PrecisionNote::DeviceSampling`]),
    /// of a texture other than a cube (whose level [`PrecisionNote::CubeGradientFootprint`]
    /// records), without comparison (whose level [`PrecisionNote::DeviceDepthComparison`] records
    /// where the device compares): a fragment function's implicit-level sample, its bias Metal's
    /// own reading `B` within the device's `maxSamplerLodBias` (the `Bias` operand), a sample with
    /// explicit gradients, a fragment function's LOD query, and a sample at an explicit level (or a
    /// kernel's bias) that Metal's level rule is not handed on (`lower::sampler::level`: an integer
    /// texture's, and a constexpr sampler's with a level bias of its own or a clamp the rule does
    /// not take), but a constant level the device reads as Metal does (a whole level a constexpr
    /// sampler without a level bias of its own clamps to a whole level, a float texture's whole
    /// level through a runtime sampler, or one a constexpr sampler's snap hands on unchanged). The
    /// device takes the level of detail by Vulkan's rules, whose scale factor may be approximated,
    /// and a nearest mip filter reads its level `ceil(λ + 1/2) - 1`, where Metal reads the level
    /// rounded to `half` with its fraction truncated onto sixty-fourths (measured for explicit
    /// levels); Metal's level of detail from derivatives is not measured. Recorded for an LOD
    /// query, and for a sample through a sampler that chooses a level or a filter by it (a mip
    /// filter, or minification and magnification filters that differ) where no sample-filter note
    /// records the sample: through a constexpr sampler with nearest filters and a nearest mip
    /// filter, through a runtime sampler where its word states those, and through any such sampler
    /// of an integer texture (an argument buffer's sampler, whose state the function does not know,
    /// always). An extended-range texture's fragment sample at an implicit level and its sample
    /// with explicit gradients (`BGR10_XR`, `BGRA10_XR` and their sRGB twins), which read texels
    /// in the shader on every device, take Vulkan's level of detail computed in the shader and
    /// then Metal's level rule: recorded through a constexpr sampler with a mip filter, and through
    /// a runtime sampler where its word states one. A bias past the limit and hoisted derivatives
    /// record [`PrecisionNote::GradFromBias`] instead. **Residual:** the sample may read another level, or
    /// another filter, than Metal's near a boundary, and the query answers the device's level.
    /// Validation derives no tolerance from it.
    DeviceSampleLevel,
    /// A render pipeline covers by alpha (`alphaToCoverageEnabled`) under
    /// `TextureWriteRoundingMode::NearestEven`, and its fragment function writes a `float` alpha
    /// to a half-float attachment 0 that stores it as written: Metal stores the alpha rounded to
    /// the nearest-even half and reads the coverage from the alpha before rounding (the coverage
    /// of 16,384 alphas the same under either mode on an Apple M3 Max), which no device
    /// separates, so the function rounds the alpha and the coverage reads the rounded one. The
    /// stored alpha is Metal's; a fragment whose alpha lies within half a `half` step of a
    /// coverage threshold may cover one sample more or fewer (2 of the 16,384 alphas on the
    /// M3 Max, each just past the midpoint between two halves). The exact path, Metal's coverage
    /// computed in the function, needs Metal's coverage pattern, which is not modelled.
    ///
    /// Validation derives no tolerance and no explanation from it: a difference stays a
    /// mismatch for review, deferred in C3 until an exact path exists.
    CoverageReadsRoundedAlpha,
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
