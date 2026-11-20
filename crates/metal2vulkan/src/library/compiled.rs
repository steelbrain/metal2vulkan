//! `CompiledFunction` and `FunctionCompileKey`: per-function compilation, cached by the caller.
//!
//! A `CompiledFunction` is the only per-function state that outlives a compile: relocatable words
//! plus plain metadata, never decoded AIR, a plan, or a typed module. The crate keeps no cache; the
//! consumer bounds and evicts. `Device::compile_function` produces it; its [`FunctionArtifact`]
//! carries the profile it was compiled for and its kernel facts. Its byte encoding
//! (`to_bytes`/`from_bytes`) is in `cache`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::cache::{write_hex, Hash256};
use crate::device::requirements::Requirements;
use crate::device::subgroup::SubgroupRequirement;
use crate::pipeline::artifact::{ModuleStats, PrecisionNote, Reconvergence, StrategyCensus};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{AttributeFetch, SlotInterface, StageInterface};
use crate::pipeline::lower_inputs::VertexFetch;
use crate::pipeline::stage_io::IoRole;
use crate::pipeline::sym::{SlotRef, Stage};
use crate::spirv::serialize::RelocatableModule;

/// The memo key of a per-function compile: H(`SEMANTICS_VERSION`, air hash, entry name, constant
/// values, profile fingerprint (policy included), `LoweringInputs`).
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionCompileKey(pub Hash256);

impl FunctionCompileKey {
    /// The key with this digest (for keys the consumer stored itself).
    pub const fn new(hash: Hash256) -> FunctionCompileKey {
        FunctionCompileKey(hash)
    }

    /// The digest.
    pub const fn digest(&self) -> &Hash256 {
        &self.0
    }
}

impl fmt::Debug for FunctionCompileKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, "FunctionCompileKey", &self.0)
    }
}

/// A compiled function: opaque, `Send + Sync`, cheap to clone. Pass it to a pipeline descriptor as
/// `StageSource::Compiled` to link without decoding or lowering again.
///
/// A clone shares both the artifact and the relocation slot, so every pipeline a consumer's memo
/// hands this function to relocates against the same slot: a sweep of pipeline states over one
/// function relocates its module once rather than once a state.
#[non_exhaustive]
#[derive(Clone)]
pub struct CompiledFunction(Arc<FunctionArtifact>, Arc<LastRelocation>);

/// The last relocation of one compiled function's module, kept beside it: **one slot,
/// last-wins**.
///
/// Relocation writes only the literals the serializer reserved, so the final words are a pure
/// function of the module and the value link assigned to each reserved slot
/// (`pipeline::relocate`). A pipeline whose link assigns the same values therefore wants the same
/// words, and a sweep of pipeline states over one function pair -- a blend sweep, where the state
/// reaches no relocation slot -- assigns the same values on every creation. Without the slot each
/// creation clones and patches the whole module again and everything keyed on its content hashes
/// the result afresh, over words that are byte-identical to the previous creation's.
///
/// **The bound is the slot.** One entry per compiled function, replaced rather than accumulated,
/// so the extra retention is one relocated word array per live function: at most what the
/// function's own `RelocatableModule` already holds, and it cannot grow with the creations, the
/// pipelines, or the distinct links a run makes. A run that varies something a relocation slot
/// does carry never hits, and still holds one entry. The slot dies with the function, so nothing
/// evicts it.
#[derive(Default)]
pub(crate) struct LastRelocation(std::sync::Mutex<Option<Relocated>>);

/// One relocation: what link assigned, and the words that came out.
struct Relocated {
    /// The value link assigned to each of the module's relocation slots, in `relocs` order: the
    /// whole of what the words depend on besides the module itself.
    assigned: Vec<u32>,
    words: Arc<[u32]>,
}

impl LastRelocation {
    /// The words of the relocation that assigned `assigned`, if that is the one the slot holds.
    pub(crate) fn words_for(&self, assigned: &[u32]) -> Option<Arc<[u32]>> {
        let slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let kept = slot.as_ref()?;
        (kept.assigned == assigned).then(|| Arc::clone(&kept.words))
    }

    /// Keep `words` as the relocation that assigned `assigned`, replacing what the slot held.
    pub(crate) fn keep(&self, assigned: Vec<u32>, words: &Arc<[u32]>) {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some(Relocated {
            assigned,
            words: Arc::clone(words),
        });
    }

}

/// A clone of a compiled function shares its slot, so the `Debug`, equality and hashing of a
/// compiled function are its artifact's alone: the slot is a memo of a value the artifact and the
/// link already determine, never a fact about the function.
impl fmt::Debug for CompiledFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompiledFunction").field(&self.0).finish()
    }
}

impl PartialEq for CompiledFunction {
    fn eq(&self, other: &CompiledFunction) -> bool {
        self.0 == other.0
    }
}

impl Eq for CompiledFunction {}

impl CompiledFunction {
    /// Wrap a finished per-function compile (`pipeline::compile`, or a cache load), with an empty
    /// relocation slot.
    pub(crate) fn new(artifact: Arc<FunctionArtifact>) -> CompiledFunction {
        CompiledFunction(artifact, Arc::new(LastRelocation::default()))
    }

    /// The function's relocation slot ([`LastRelocation`]), which `pipeline::relocate` reads and
    /// fills.
    pub(crate) fn last_relocation(&self) -> &LastRelocation {
        &self.1
    }

    /// The key the function was compiled under.
    pub fn key(&self) -> &FunctionCompileKey {
        &self.0.key
    }

    /// The per-function artifact (internal).
    pub(crate) fn artifact(&self) -> &FunctionArtifact {
        &self.0
    }

    /// The stage the function compiled as.
    pub fn stage(&self) -> Stage {
        self.0.interface.stage
    }

    /// What the function's module needs of the device: a pipeline that links it needs at least
    /// this ([`DeviceProfile::check`](crate::device::DeviceProfile::check) says whether a profile
    /// provides it).
    pub fn requirements(&self) -> &Requirements {
        &self.0.requirements
    }

    /// The near-equivalents of Metal the function's lowering recorded. A pipeline's stage of it
    /// states these and those the pipeline's own state adds (`StageModule::precision_notes`):
    /// for a fragment function, its color attachments' writes; for a vertex function, the
    /// formats the pipeline's fixed-function vertex input converts
    /// (`PrecisionNote::DeviceVertexConversion`, `PrecisionNote::DeviceNanEncoding`).
    pub fn precision_notes(&self) -> &[PrecisionNote] {
        &self.0.precision
    }

    /// The lowering strategies the function used and the words each emitted, as its linked
    /// stage states them (`StageModule::strategies`).
    pub fn strategies(&self) -> &StrategyCensus {
        &self.0.strategies
    }

    /// The reconvergence the function's module relies on (`StageModule::reconvergence`).
    pub fn reconvergence(&self) -> Reconvergence {
        self.0.reconvergence
    }

    /// What the function's module contains: its functions, blocks, instructions, structured
    /// constructs and loops, as its linked stage states them (`StageModule::stats`). Linking the
    /// function resolves its symbols and adds no instruction, so these are the counts of the
    /// module a pipeline of it hands the driver.
    pub fn module_stats(&self) -> ModuleStats {
        self.0.stats
    }

    /// The subgroup size the function's simd tier pins, if any (its stage's
    /// `requiredSubgroupSize`, `StageModule::required_subgroup_size`).
    pub fn required_subgroup_size(&self) -> Option<u32> {
        self.0.subgroup.size
    }

    /// Whether the function's simd tier requires full subgroups (its stage's
    /// `REQUIRE_FULL_SUBGROUPS`).
    pub fn requires_full_subgroups(&self) -> bool {
        self.0.subgroup.full_subgroups
    }

    /// Whether the function, a kernel, observes its threadgroup partition: it may compute
    /// otherwise in another partition of its grid into threadgroups, as it holds threadgroup
    /// memory, a barrier, a fence or atomic of threadgroup or simdgroup scope, a simdgroup or quad
    /// function, or reads a builtin of the partition. Decided from its AIR alone. A kernel that
    /// does not runs any grid in threadgroups of a consumer's choosing (`LoweringInputs`'s
    /// `flat_grid`); one that does runs a grid that is not a whole number of threadgroups on its
    /// partial-threadgroups variant (`partial_threadgroups`) or split. `true` for a function that
    /// is not a kernel.
    pub fn observes_threadgroup_partition(&self) -> bool {
        !self.0.kernel.partition_oblivious
    }

    /// Whether the function, a kernel, is image-only: it samples, gathers or reads a texture, and
    /// loads, stores and updates atomically nothing in device memory, through no load, store,
    /// atomic or call but a texture or sampler operation on a pointer into the `device` address
    /// space. Loads of `constant` memory and texture writes are allowed. Decided from its AIR
    /// alone. A runtime may run such a kernel that does not observe its threadgroup partition
    /// ([`observes_threadgroup_partition`](Self::observes_threadgroup_partition)) in square
    /// workgroups. `false` for a function that is not a kernel.
    pub fn is_image_only(&self) -> bool {
        self.0.kernel.image_only
    }

    /// Where a fragment function linked after this vertex function pushes its fields on
    /// `Device::from_queries`' profile (`LoweringInputs::push_base`): this function's push end, a
    /// multiple of 4; where it pushes nothing, 24 for a post-tessellation function (past its
    /// pipeline's control stage's push range) and 0 for any other. Each fragment field is placed
    /// at the next offset aligned to it, so a four-byte field may take the bytes below an
    /// eight-byte one. A consumer compiling a render pipeline's fragment function on its own
    /// compiles it under this offset.
    pub fn fragment_push_base(&self) -> u32 {
        let interface = &self.0.interface;
        crate::lower::push::fragment_base(&interface.push, interface.tessellation.is_some())
    }

    /// Whether the function is a vertex function whose module writes its `[[point_size]]`
    /// output, read from its stage's built-in outputs: Metal refuses such a function in a render
    /// pipeline of the line or triangle input primitive topology class.
    pub fn writes_point_size(&self) -> bool {
        self.0.interface.writes_point_size()
    }

    /// The samplers a pipeline state of the function numbers for it, in Metal's order
    /// ([`NumberedSampler`](crate::pipeline::NumberedSampler)): Metal's pipeline creation gives
    /// each a plain sampler resource ID unless an equal one of the device holds one.
    pub fn numbered_samplers(&self) -> &[crate::pipeline::NumberedSampler] {
        &self.0.interface.numbered_samplers
    }

    /// Every Metal argument slot the function uses, with how it uses it: a buffer's access,
    /// window and form, a texture's or a sampler's.
    pub fn slots(&self) -> &BTreeMap<SlotRef, SlotInterface> {
        &self.0.interface.slots
    }

    /// How the function fetches each attribute it reads, by Metal attribute index: a vertex
    /// function's `[[attribute(n)]]` inputs, by fixed-function vertex input or in the shader, and a
    /// kernel's `[[stage_in]]` attributes, in the shader. Empty for a function that reads none.
    pub fn attribute_fetches(&self) -> BTreeMap<u32, AttributeFetch> {
        let io = &self.0.interface.io;
        let fixed = io.inputs.iter().filter_map(|variable| match variable.role {
            IoRole::Attribute { index, .. } => Some((index, AttributeFetch::FixedFunction)),
            _ => None,
        });
        let pulled = io.pulled.iter().filter_map(|input| match input.fetch {
            VertexFetch::Pulled { buffer_index, .. } => {
                Some((input.index, AttributeFetch::Pulled { buffer_index }))
            }
            VertexFetch::FixedFunction | VertexFetch::SignedFixedFunction => None,
        });
        let stage_in = io.stage_in.iter().map(|input| {
            (
                input.index,
                AttributeFetch::Pulled {
                    buffer_index: input.fetch.buffer_index,
                },
            )
        });
        fixed.chain(pulled).chain(stage_in).collect()
    }

    /// Whether the function writes memory another invocation or a later command can read: it
    /// stores to a buffer or a texture, or updates one atomically. A render pipeline whose colors
    /// the device draws in more than one group refuses such a function, which each group's draw
    /// would run again ([`ColorGroup`](crate::pipeline::ColorGroup)).
    pub fn writes_memory(&self) -> bool {
        self.0.interface.writes_memory
    }

    /// Whether the function reads the resident root table: it has a device-address slot or a
    /// dynamic root, so the binder builds the table with an entry for each such slot.
    pub fn reads_resident_roots(&self) -> bool {
        self.0.interface.needs_pointer_translation
    }

    /// Whether the function looks texture or sampler handles up in the handle table.
    pub fn resolves_handles(&self) -> bool {
        self.0.interface.needs_handle_translation
    }
}

/// What one per-function compile retains, with its provenance: the profile it was compiled for and
/// its kernel facts. Internal, behind [`CompiledFunction`]. `pipeline::compile` produces it; no
/// decoded AIR, plan, or typed module is retained.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionArtifact {
    pub key: FunctionCompileKey,
    /// SPIR-V words with their link-assigned literals still symbolic.
    pub module: RelocatableModule,
    /// The function's slots, push placement, and declared link symbols.
    pub interface: StageInterface,
    /// Runtime facts the function depends on.
    pub facts: Vec<RuntimeFact>,
    pub subgroup: SubgroupRequirement,
    pub reconvergence: Reconvergence,
    pub precision: Vec<PrecisionNote>,
    pub strategies: StrategyCensus,
    /// What the finalize verifier counted of the module: what its linked stage states as
    /// `StageModule::stats`.
    pub stats: ModuleStats,
    /// Constant data the function reads through `WriteSource::ConstantBlob`.
    pub blob: Option<Arc<[u8]>>,
    pub requirements: Requirements,
    /// Specialization and variant bits the function can take.
    pub variant_bits: u32,
    /// The fingerprint of the profile it was compiled for: a pipeline links it only for that
    /// profile (`DescriptorError::ForeignCompiledFunction`), and a cache load only on a device of
    /// that profile.
    pub profile: Hash256,
    /// The entry's facts behind Metal's per-pipeline compute queries.
    pub kernel: KernelFacts,
    /// What a pipeline's reflection states for the stage, from the entry the function compiled
    /// (`reflection::bindings` over it, as the pipeline channel answers), so a pipeline of the
    /// compiled function answers its reflection without its AIR. `None` where a binding's answer
    /// is unmeasured (the reflection refuses it).
    pub(crate) reflection: Option<crate::reflection::StageReflection>,
    /// What the stage may store through each slot its arguments bind, from the entry the function
    /// compiled (`reflection::entry_slot_writes` over it), so a pipeline of the compiled function
    /// answers it without its AIR. Never absent: the answer reads each parameter's role, declared
    /// access and slot alone and derives no type description, so no unmeasured argument type can
    /// take it away, and a per-slot gap is one undetermined answer among the others
    /// (`SlotWritability::Unknown`).
    pub(crate) slot_writes: Vec<crate::reflection::SlotWrite>,
    /// The notes of `precision` that hold only under some specializations, with their
    /// conditions: what its linked stage states as `StageModule::specialized_notes`.
    pub specialized: Vec<crate::pipeline::artifact::SpecializedNote>,
}

/// What a kernel's `ComputePipelineMetalInfo`, and an object or mesh function's
/// `MeshPipelineMetalInfo`, need from its compile. A vertex or fragment function has the default:
/// no attribute, not flattened, no threadgroup memory.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelFacts {
    /// `[[max_total_threads_per_threadgroup(n)]]` (`air.max_work_group_size`).
    pub max_total_threads: Option<u32>,
    /// OpenCL C's `reqd_work_group_size(x, y, z)` (`air.work_group_size`): Metal caps the
    /// pipeline's `maxTotalThreadsPerThreadgroup` at `x * y * z`.
    pub work_group_size: Option<[u32; 3]>,
    /// The Direct or Cluster32 tier flattened the local size.
    pub flat_local_size: bool,
    /// Bytes of the static part of Metal's threadgroup layout: the static objects, and with
    /// `[[threadgroup(n)]]` regions the 16-byte-aligned start of the first region. A static object
    /// held as a typed `Workgroup` variable (a kernel without regions) counts at its place there,
    /// whatever the variable's own bytes.
    pub static_threadgroup_bytes: u64,
    /// The kernel does not observe its threadgroup partition: it computes the same in every
    /// partition of its grid into threadgroups, so a dispatch may run in threadgroups the runtime
    /// chooses. Decided from the entry's AIR alone (`pipeline::compile`'s `partition`): no
    /// threadgroup memory, no barrier, no fence or atomic of threadgroup or simdgroup scope, no
    /// simdgroup or quad function, and no builtin of the partition.
    pub partition_oblivious: bool,
    /// The kernel observes its threadgroup partition through its simdgroups alone: its only
    /// observations are barriers of simdgroup execution scope, fences and atomics of simdgroup
    /// scope, simdgroup and quad functions, `is_uniform`, the simdgroup size, and the builtins
    /// `thread_index_in_simdgroup`, `thread_index_in_quadgroup` and `threads_per_simdgroup`. It
    /// computes the same in any partition that keeps its threadgroups, so a one-dimensional
    /// dispatch of the call's threadgroups may run flat (`LoweringInputs::call_threadgroups`).
    /// `false` for a kernel that does not observe its partition
    /// ([`partition_oblivious`](Self::partition_oblivious)). Decided as that one is.
    pub partition_simdgroups: bool,
    /// The kernel reads an image and loads, stores and updates atomically nothing in device
    /// memory (`pipeline::compile`'s `images`): it samples, gathers or reads a texture, and no
    /// load, store or atomic, and no call but a texture or sampler operation, goes through a
    /// pointer into the `device` address space. Loads of `constant` memory and texture writes are
    /// allowed. Decided from the entry's AIR alone; `false` for a function that is not a kernel.
    pub image_only: bool,
}
