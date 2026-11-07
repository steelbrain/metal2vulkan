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

/// One relocation: what link assigned, the words that came out, and a digest of those words a
/// caller asked for.
struct Relocated {
    /// The value link assigned to each of the module's relocation slots, in `relocs` order: the
    /// whole of what the words depend on besides the module itself.
    assigned: Vec<u32>,
    words: Arc<[u32]>,
    /// The digest of `words`, with the function that derived it: a caller that hashes words some
    /// other way derives again rather than reading a digest its own function did not make
    /// ([`CompiledFunction::relocated_words_digest`]). Cleared with the words it describes, so
    /// it can never name another relocation's.
    digest: Option<(fn(&[u32]) -> Hash256, Hash256)>,
}

/// What a relocation slot can say about a digest of some words by some function
/// ([`LastRelocation::digest_of`]).
enum DigestOf {
    /// It holds those very words and that function's digest of them.
    Kept(Hash256),
    /// It holds those very words and no digest of them by that function, so one derived now
    /// belongs in it.
    Wanted,
    /// The words are not the ones it holds, so it has nothing to say about them.
    Other,
}

impl LastRelocation {
    /// The words of the relocation that assigned `assigned`, if that is the one the slot holds.
    pub(crate) fn words_for(&self, assigned: &[u32]) -> Option<Arc<[u32]>> {
        let slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let kept = slot.as_ref()?;
        (kept.assigned == assigned).then(|| Arc::clone(&kept.words))
    }

    /// Keep `words` as the relocation that assigned `assigned`, replacing what the slot held
    /// (and the digest of the words it held).
    pub(crate) fn keep(&self, assigned: Vec<u32>, words: &Arc<[u32]>) {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some(Relocated {
            assigned,
            words: Arc::clone(words),
            digest: None,
        });
    }

    /// What the slot has for a digest of `words` by `derive` ([`DigestOf`]). The words are the
    /// slot's only if they are the same allocation: a relocation hands back the very `Arc` it
    /// kept, so an equal copy of the words is a different caller's and answers `Other`.
    fn digest_of(&self, words: &Arc<[u32]>, derive: fn(&[u32]) -> Hash256) -> DigestOf {
        let slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(kept) = slot.as_ref() else {
            return DigestOf::Other;
        };
        if !Arc::ptr_eq(&kept.words, words) {
            return DigestOf::Other;
        }
        match kept.digest {
            Some((made_by, digest)) if std::ptr::fn_addr_eq(made_by, derive) => {
                DigestOf::Kept(digest)
            }
            _ => DigestOf::Wanted,
        }
    }

    /// Keep `digest`, which `derive` produced over `words`, if the slot still holds those very
    /// words; drop it if the slot has moved on, since a digest only ever describes the words
    /// beside it.
    fn keep_digest(&self, words: &Arc<[u32]>, derive: fn(&[u32]) -> Hash256, digest: Hash256) {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(kept) = slot.as_mut() {
            if Arc::ptr_eq(&kept.words, words) {
                kept.digest = Some((derive, digest));
            }
        }
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

    /// The digest `derive` gives for `words`, where `words` are the final SPIR-V of this
    /// function's last relocation -- the very `Arc` a pipeline's stage took from it -- and
    /// `None` where they are not.
    ///
    /// A pipeline state that names a stage's module by a digest of its words asks here first: a
    /// sweep of states over one function pair takes the same words on every creation (the
    /// relocation slot gives back the same allocation), so without this each creation hashes
    /// words byte-identical to the previous creation's. Pass the words a stage of the pipeline
    /// holds and the function that would have hashed them.
    ///
    /// **A miss costs no `derive` call at all.** `None` is answered from the slot alone: no
    /// hash is taken, and the caller is free to derive or not. That is what a caller in a hot
    /// path needs to know, and it is also why the caller's own function stays the only
    /// derivation of a stage's words that ever runs -- nothing here hashes words any other way,
    /// so nothing keyed on such a digest can take a different value from this.
    ///
    /// **The digest a caller gets is always one its own `derive` produced.** The slot keeps the
    /// function beside the digest and answers only where that function is this call's, so two
    /// callers that hash differently never read each other's digest: a mismatch derives again
    /// and replaces what the slot held.
    ///
    /// **On comparing `fn` pointers**, which is not guaranteed stable: identical-code folding
    /// may merge two functions into one address, and one `fn` item may have several addresses
    /// across codegen units. Both directions are harmless here. Two addresses for one function
    /// make a mismatch, which costs a hash and gives the same digest. The dangerous direction --
    /// two *different* hash functions reading as one -- cannot occur, because folding merges
    /// only identical machine code and identical machine code cannot produce different digests.
    ///
    /// `derive` is the caller's, so it is **never called while the slot is locked**: one that
    /// reached back into this compiled function would otherwise deadlock. The slot is locked and
    /// read, released, `derive` is called, and the digest is stored only if the slot still holds
    /// those words. Two threads deriving at once is benign -- the same words give the same
    /// digest, and the last store wins.
    pub fn relocated_words_digest(
        &self,
        words: &Arc<[u32]>,
        derive: fn(&[u32]) -> crate::cache::Hash256,
    ) -> Option<crate::cache::Hash256> {
        match self.1.digest_of(words, derive) {
            DigestOf::Kept(digest) => Some(digest),
            DigestOf::Other => None,
            // Derived with the slot released: `derive` is the caller's and may reach anywhere.
            DigestOf::Wanted => {
                let digest = derive(words);
                self.1.keep_digest(words, derive, digest);
                Some(digest)
            }
        }
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
            VertexFetch::FixedFunction => None,
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
}
