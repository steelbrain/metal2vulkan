//! What a render pipeline state's creation works out from its descriptor before its compile,
//! kept by the descriptor's key, in the device's memory and in the cache directory.
//!
//! A creation decoded and specialized its stages' AIR to answer Metal's linked-use refusals
//! (`RenderPipelineDescriptor::{vertex_returns_value, fragment_writes_depth_unattached,
//! stages_using_direct_arguments, unwritten_fragment_inputs}`), to learn the key its pipeline
//! is found by (`Device::render_variant_key`), and, where asked, its reflection: 18 to 23 ms of
//! a creation of the consumer's heaviest states on the RTX 5080 Laptop GPU's host made again in
//! the same process, and most of the 33 to 39 ms of one in a new process over warm caches, where
//! Metal answers in under 1 ms.
//!
//! - **Keyed by everything the answers read.** Each answer is a function of the descriptor and
//!   the device alone: `Device::render_descriptor_key` digests every field of the descriptor
//!   (each `Function` stage by its compile key: AIR hash, entry, constants, linked functions and
//!   dynamic libraries) with the device's profile fingerprint and `SEMANTICS_VERSION`. A
//!   variant's keys are kept per variant key too ([`VariantStages`]): its pipeline's, and its
//!   stages' functions', which a later creation finds and links with no stage decoded. The
//!   creation's budget decides only whether an
//!   answer comes at all: a refused answer (a deadline, a cancellation, a decode's error) is not
//!   kept, so the next creation asks again.
//! - **And by more than they read, which costs a sweep a trip each.** The key is every field,
//!   where the four answers read only some: the stages, `rasterization_enabled`,
//!   `support_indirect_command_buffers`, the two call-stack depths, whether `depth_format` names
//!   an attachment, and each color attachment's format, write mask, `blending_enabled` and four
//!   blend factors (`reflection::linked`'s `fragment_output` and `writes_depth_unattached`). So a
//!   sweep that moves a field none of them reads -- a vertex descriptor's formats, strides or
//!   step functions, the topology class, the sample state -- asks the directory for a sibling's
//!   answers and works them out again, for an answer identical to the previous sibling's. A
//!   sweep that moves a write mask or a blend factor is **not** such a case: a fragment color
//!   output is kept or dropped by its attachment's mask, format and blending, so those answers
//!   really can differ between siblings and the key must hold them. Narrowing the key is the
//!   translator's to do, beside `render_descriptor_key` and under the same discipline (a field
//!   added to the descriptor must not compile until the key decides about it); doing it here,
//!   from the runtime's reading of which fields the translator's analysis happens to touch,
//!   would break silently the next time it touches one more.
//! - **In memory first, then the directory.** A lookup asks the device's maps, then the
//!   directory's entry (kept in the maps when found); what a creation works out goes to both.
//!   The entries live in the build's directory like the artifacts, so a build never reads
//!   another translator's, under names of their kind, key, profile and the runtime's
//!   `RUNTIME_VERSION` ([`record_name`]): they are the runtime's derivation from a descriptor,
//!   which the build's identity (the translator's) does not cover, so a runtime whose version
//!   moved reads none of them. The answers and the reflection are
//!   the translator's frames (`LinkedUseAnswers::to_bytes`,
//!   `RenderPipelineReflection::to_bytes`: versions, kind, key, profile and checksum checked); a
//!   variant's keys are a record of its own name and the keys under a SHA-256 checksum. A
//!   lookup counts in no statistic but a refused or unreadable entry, and a write in none but a
//!   failed one: they are not translations found, compiled or written. The write itself is the
//!   device's cache thread's, as an artifact's is (`store.rs`, `worker.rs`).
//! - **Bounded.** Each map keeps at most [`ENTRIES`] entries, the least recently used going first
//!   ([`Lru`]); the directory's entries go as every entry does, past its cap. They hold digests,
//!   small answers and reflections, and no AIR, library, or Vulkan object.

use super::store::{entry_name, Lookup};
use super::{EntryKind, ShaderCache};
use crate::device::lock;
use crate::pipeline::memo::Lru;
use metal2vulkan::cache::{Hash256, PipelineKey, VariantKey};
use metal2vulkan::device::Device as CompileDevice;
use metal2vulkan::library::FunctionCompileKey;
use metal2vulkan::reflection::{LinkedUseAnswers, RenderPipelineReflection};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

/// The most entries each of the memo's maps keeps.
pub(crate) const ENTRIES: usize = 256;

/// The tag a creation record's key hashes first, before the runtime's version ([`record_name`]).
const RECORD_KEY_TAG: &[u8] = b"metal2vulkan-runtime creation record key v1\0";

/// The tag a variant's record name hashes first.
const VARIANT_NAME_TAG: &[u8] = b"metal2vulkan-runtime variant pipeline name v2\0";

/// The first bytes of a variant's record.
const VARIANT_MAGIC: &[u8; 8] = b"M2VVPIP\x02";

/// A variant record's length: its magic, its name, the pipeline key, the vertex stage's key, a
/// byte saying whether a fragment stage's key follows, that key (zeros where none), and the
/// checksum.
const VARIANT_RECORD: usize = 8 + 32 + 32 + 32 + 1 + 32 + 32;

/// What a creation learned of a render descriptor's variant (`Device::render_variant_key`): the
/// key its pipeline is found by, and the compile keys its stages' functions are found by. A
/// creation that finds both functions links them, as one that computed the keys links what it
/// found or compiled under them, and decodes no stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VariantStages {
    pub(crate) pipeline: PipelineKey,
    pub(crate) vertex: FunctionCompileKey,
    pub(crate) fragment: Option<FunctionCompileKey>,
}

/// The variant records this process gave the directory, which `tests::render_stage_digests`'
/// count reads: a sweep must give it one per creation. Carried
/// only where that device test is compiled, since nothing else reads it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
static RECORDS_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`RECORDS_WRITTEN`].
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn variant_records_written() -> u64 {
    RECORDS_WRITTEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count a variant record given to the directory.
fn note_variant_record_written() {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    RECORDS_WRITTEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The device's memo of what its render creations worked out before their compiles (the module
/// docs).
pub(crate) struct CreationMemo {
    linked: Mutex<Lru<PipelineKey, LinkedUseAnswers>>,
    pipelines: Mutex<Lru<(PipelineKey, VariantKey), VariantStages>>,
    reflections: Mutex<Lru<PipelineKey, RenderPipelineReflection>>,
    /// How many reflections the device has worked out, for the check that every state of one
    /// descriptor shares one ([`ShaderCache::reflections_derived`]). Working one out decodes the
    /// descriptor's stages' AIR, and a count of that is deterministic where a time is not; it is
    /// of this device and its cache directory, so a test that reads it gives the device a
    /// directory of its own.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    derived: std::sync::atomic::AtomicU64,
}

impl CreationMemo {
    /// An empty memo.
    pub(crate) fn new() -> CreationMemo {
        CreationMemo {
            linked: Mutex::new(Lru::counted(ENTRIES)),
            pipelines: Mutex::new(Lru::counted(ENTRIES)),
            reflections: Mutex::new(Lru::counted(ENTRIES)),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            derived: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl ShaderCache {
    /// The linked-use answers a creation of the render descriptor of `key` on `compile` kept:
    /// the memo's, else the directory's.
    pub(crate) fn linked_answers(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
    ) -> Option<LinkedUseAnswers> {
        if let Some(answers) = lock(&self.creation.linked).get(key) {
            return Some(answers.clone());
        }
        let answers =
            self.read_answer(EntryKind::LinkedAnswers, key.digest(), compile, |bytes| {
                LinkedUseAnswers::from_bytes_for_key(bytes, key, compile)
            })?;
        lock(&self.creation.linked).insert(*key, answers.clone(), 1);
        Some(answers)
    }

    /// Keep `answers` as the linked-use answers of the render descriptor of `key` on `compile`,
    /// in the memo and the directory.
    pub(crate) fn keep_linked_answers(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        answers: LinkedUseAnswers,
    ) {
        self.keep_answer(EntryKind::LinkedAnswers, key.digest(), compile, || {
            answers.to_bytes(key, compile)
        });
        lock(&self.creation.linked).insert(*key, answers, 1);
    }

    /// The keys of the variant of the render descriptor of `key` on `compile` under `variant`,
    /// as a creation learned them: the memo's, else the directory's.
    pub(crate) fn variant_pipeline(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        variant: &VariantKey,
    ) -> Option<VariantStages> {
        let at = (*key, variant.clone());
        if let Some(stages) = lock(&self.creation.pipelines).get(&at) {
            return Some(*stages);
        }
        let name = variant_name(key, variant);
        let stages = self.read_answer(EntryKind::VariantPipeline, &name, compile, |bytes| {
            variant_record_stages(bytes, &name)
        })?;
        lock(&self.creation.pipelines).insert(at, stages, 1);
        Some(stages)
    }

    /// Keep `stages` as the keys of the variant of the render descriptor of `key` under
    /// `variant` in the memo, which is what spares another creation of the same descriptor in
    /// this process its [`Device::render_variant_key`]; whether the memo did not already hold
    /// them, and so whether the directory's record of them is one this creation would write
    /// ([`record_variant_pipeline`](Self::record_variant_pipeline)).
    ///
    /// [`Device::render_variant_key`]: metal2vulkan::device::Device::render_variant_key
    pub(crate) fn remember_variant_pipeline(
        &self,
        key: &PipelineKey,
        variant: &VariantKey,
        stages: &VariantStages,
    ) -> bool {
        let at = (*key, variant.clone());
        let mut pipelines = lock(&self.creation.pipelines);
        if pipelines.get(&at) == Some(stages) {
            return false;
        }
        pipelines.insert(at, *stages, 1);
        true
    }

    /// Write the directory's record that the variant of the render descriptor of `key` on
    /// `compile` under `variant` has the keys `stages`.
    ///
    /// **A record spares a later process its stages' decodes.** It hands that process the key to
    /// look the pipeline up by and, where the directory does not hold the pipeline, the keys of
    /// the stages' functions to link it from, so `render_variant_key`, which decodes and
    /// specializes every `Function` stage's AIR to learn them, runs only where neither is found.
    /// A sweep of states over one function pair stores no pipeline artifact where the directory
    /// holds its stages (`Sources::linked_render`), and each sibling's record is what lets the
    /// next process link it from those stages without decoding them.
    pub(crate) fn record_variant_pipeline(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        variant: &VariantKey,
        stages: &VariantStages,
    ) {
        let name = variant_name(key, variant);
        note_variant_record_written();
        self.keep_answer(EntryKind::VariantPipeline, &name, compile, || {
            variant_record(&name, stages)
        });
    }

    /// The reflection of the render descriptor of `key` on `compile`: the memo's, else the
    /// directory's, else `work`'s, kept in both where it answers.
    pub(crate) fn render_reflection<E>(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        work: impl FnOnce() -> Result<RenderPipelineReflection, E>,
    ) -> Result<RenderPipelineReflection, E> {
        if let Some(reflection) = self.kept_render_reflection(compile, key) {
            return Ok(reflection);
        }
        let made = work()?;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.creation
            .derived
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.keep_render_reflection(compile, key, made))
    }

    /// How many reflections the device has worked out through
    /// [`render_reflection`](Self::render_reflection) ([`CreationMemo::derived`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn reflections_derived(&self) -> u64 {
        self.creation
            .derived
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The reflection of the render descriptor of `key` on `compile` the memo keeps, else the
    /// directory's, then kept in the memo: [`render_reflection`](Self::render_reflection)'s
    /// answer where it needs no work.
    pub(crate) fn kept_render_reflection(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
    ) -> Option<RenderPipelineReflection> {
        if let Some(reflection) = lock(&self.creation.reflections).get(key) {
            return Some(reflection.clone());
        }
        let reflection = self.read_answer(
            EntryKind::RenderReflection,
            key.digest(),
            compile,
            |bytes| RenderPipelineReflection::from_bytes_for_key(bytes, key, compile),
        )?;
        lock(&self.creation.reflections).insert(*key, reflection.clone(), 1);
        Some(reflection)
    }

    /// Keep `reflection`, worked out for the render descriptor of `key` on `compile`, in the memo
    /// and the directory; returned.
    pub(crate) fn keep_render_reflection(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        reflection: RenderPipelineReflection,
    ) -> RenderPipelineReflection {
        self.keep_answer(EntryKind::RenderReflection, key.digest(), compile, || {
            reflection.to_bytes(key, compile)
        });
        lock(&self.creation.reflections).insert(*key, reflection.clone(), 1);
        reflection
    }

    /// Write what `encode` encodes as the directory's entry of `kind` under `key` for
    /// `compile`'s profile, as an artifact is written (the bytes counted against the directory's
    /// cap, the least recently used entries going past it), but counted as no entry written (it
    /// is no translation); a failed write is counted and said as an artifact's is. A device with
    /// no directory encodes nothing and counts it as not written (the module docs).
    pub(super) fn keep_answer(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &CompileDevice,
        encode: impl FnOnce() -> Vec<u8>,
    ) {
        if !self.store.has_directory() {
            self.store.write_failed();
            return;
        }
        let name = record_name(
            self.store.build(),
            kind,
            key,
            &compile.profile_fingerprint(),
            metal2vulkan::cache::RUNTIME_VERSION,
        );
        if self.store.write_soon(&name, encode()).is_err() {
            self.store.write_failed();
        }
        if self.store.over_capacity() {
            self.worker.evict();
        }
    }

    /// The directory's entry of `kind` under `key` for `compile`'s profile, as `decode` accepts
    /// it: counted as a hit or a miss by no statistic (it is no translation), a refused or
    /// unreadable entry counted as the store counts it.
    pub(super) fn read_answer<T, E>(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &CompileDevice,
        decode: impl FnOnce(&[u8]) -> Result<T, E>,
    ) -> Option<T> {
        let name = record_name(
            self.store.build(),
            kind,
            key,
            &compile.profile_fingerprint(),
            metal2vulkan::cache::RUNTIME_VERSION,
        );
        match self.store.load(&name, decode) {
            Lookup::Hit(value) => Some(value),
            Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
        }
    }
}

/// The name a creation's record of `kind` under `key` for `profile` is kept under in `build`'s
/// directory: the entry name of the SHA-256 of [`RECORD_KEY_TAG`], `runtime` (the runtime's
/// `RUNTIME_VERSION`) and `key`. What a creation works out from its descriptor -- its linked-use
/// answers, its reflection, its variant's keys -- is the runtime's own derivation, which the build
/// directory's identity does not cover (it names the translator, `this_build`), so the runtime's
/// version names it: a runtime change that maps a descriptor to other answers or keys bumps
/// `RUNTIME_VERSION` and reads none of an earlier runtime's records.
pub(crate) fn record_name(
    build: &Hash256,
    kind: EntryKind,
    key: &Hash256,
    profile: &Hash256,
    runtime: u32,
) -> String {
    let mut hash = Sha256::new();
    hash.update(RECORD_KEY_TAG);
    hash.update(runtime.to_le_bytes());
    hash.update(key);
    let key: Hash256 = hash.finalize().into();
    entry_name(build, kind, &key, profile)
}

/// The name a variant's record is kept under: the SHA-256 of the render descriptor's key and the
/// variant key's every field (its derived `Debug` rendering, as the translator's keys render
/// their inputs).
fn variant_name(key: &PipelineKey, variant: &VariantKey) -> Hash256 {
    let mut hash = Sha256::new();
    hash.update(VARIANT_NAME_TAG);
    hash.update(key.digest());
    hash.update(format!("{variant:?}").as_bytes());
    hash.finalize().into()
}

/// A variant's record: its magic, `name`, the digests of `stages` (the fragment's after a byte
/// saying whether there is one), and the SHA-256 of all of them.
fn variant_record(name: &Hash256, stages: &VariantStages) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(VARIANT_RECORD);
    bytes.extend_from_slice(VARIANT_MAGIC);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(stages.pipeline.digest());
    bytes.extend_from_slice(stages.vertex.digest());
    bytes.push(u8::from(stages.fragment.is_some()));
    bytes.extend_from_slice(stages.fragment.as_ref().map_or(&[0; 32], |f| f.digest()));
    let checksum: Hash256 = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// A variant record's refusal: not a record of this kind, of another name, or damaged.
#[derive(Debug)]
struct NotTheRecord;

/// The keys of the record `bytes` named `name`, refused unless they are exactly such a record
/// ([`variant_record`]).
fn variant_record_stages(bytes: &[u8], name: &Hash256) -> Result<VariantStages, NotTheRecord> {
    let sum = VARIANT_RECORD - 32;
    if bytes.len() != VARIANT_RECORD
        || &bytes[..8] != VARIANT_MAGIC
        || &bytes[8..40] != name
        || bytes[sum..] != <[u8; 32]>::from(Sha256::digest(&bytes[..sum]))
    {
        return Err(NotTheRecord);
    }
    let digest = |at: usize| -> Result<Hash256, NotTheRecord> {
        bytes[at..at + 32].try_into().map_err(|_| NotTheRecord)
    };
    let fragment = match bytes[104] {
        0 => None,
        1 => Some(FunctionCompileKey::new(digest(105)?)),
        _ => return Err(NotTheRecord),
    };
    Ok(VariantStages {
        pipeline: PipelineKey::new(digest(40)?),
        vertex: FunctionCompileKey::new(digest(72)?),
        fragment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A variant record gives its keys back under its own name only, with and without a
    /// fragment stage, and a damaged or cut record is refused.
    #[test]
    fn a_variant_record_names_its_keys_under_its_own_name_only() {
        let key = PipelineKey::new([1; 32]);
        let mut variant = VariantKey::new();
        let name = variant_name(&key, &variant);
        variant.emulated_simdgroups = true;
        let other = variant_name(&key, &variant);
        assert_ne!(name, other, "every variant field names the record");
        assert_ne!(
            name,
            variant_name(&PipelineKey::new([2; 32]), &VariantKey::new())
        );
        let stages = VariantStages {
            pipeline: PipelineKey::new([9; 32]),
            vertex: FunctionCompileKey::new([7; 32]),
            fragment: Some(FunctionCompileKey::new([5; 32])),
        };
        let vertex_only = VariantStages {
            fragment: None,
            ..stages
        };
        for stages in [stages, vertex_only] {
            let record = variant_record(&name, &stages);
            assert_eq!(record.len(), VARIANT_RECORD);
            assert_eq!(variant_record_stages(&record, &name).ok(), Some(stages));
            assert!(variant_record_stages(&record, &other).is_err());
            for at in [
                0,
                8,
                40,
                72,
                104,
                105,
                VARIANT_RECORD - 33,
                VARIANT_RECORD - 1,
            ] {
                let mut damaged = record.clone();
                damaged[at] ^= 1;
                assert!(variant_record_stages(&damaged, &name).is_err(), "byte {at}");
            }
            assert!(variant_record_stages(&record[..VARIANT_RECORD - 1], &name).is_err());
        }
    }
}
