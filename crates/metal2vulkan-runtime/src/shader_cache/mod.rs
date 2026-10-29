//! The translated-shader cache on disk: always on, so a second process does not translate again
//! what the first already did, as Metal programs get from the OS.
//!
//! - **The directory.** [`ShaderCacheOptions::directory`], or the platform's per-user cache
//!   directory: `~/Library/Caches/metal2vulkan` on macOS, `$XDG_CACHE_HOME/metal2vulkan`
//!   elsewhere, or `~/.cache/metal2vulkan` when that is unset or not absolute. Never the OS temp
//!   directory. Creating or adopting a device makes it (owner-only) and checks that it takes a
//!   write; if not, the call fails with [`Error::ShaderCache`], naming the path and the I/O error.
//!   After that, a failed read or write is a miss the [statistics](ShaderCacheStatistics) count,
//!   never an error.
//! - **What it holds.** The compile layer's artifacts in their self-checking envelope: translated
//!   functions under their [`FunctionCompileKey`] and compute and render pipelines under their
//!   [`PipelineKey`]; and the driver's `VkPipelineCache` data, one record per family of makings
//!   that convert the same stages, named by what a stage's conversion reads (`driver.rs`); and
//!   what render creations work out from their descriptors before their compiles, under the
//!   descriptor's key (`creation.rs`).
//! - **A build never reads the entries of another translator.** They live under `v1/<build>/`,
//!   where `<build>` is the SHA-256 of the crate's version, both artifact versions, and the
//!   translator's source digest (`metal2vulkan::cache::SOURCE_DIGEST`, its product files), so a
//!   translator edit starts afresh even where no version moved. The runtime's own sources are not
//!   in it: a translated artifact is the translator's answer for the inputs its key and profile
//!   fingerprint hash, so a build that changed only the runtime reads every artifact an earlier
//!   one wrote. What the runtime derives itself from a render descriptor (its linked-use answers,
//!   reflection and variant keys, `creation.rs`) is named by `RUNTIME_VERSION` too, so a runtime
//!   change that bumps it reads none of an earlier runtime's records; a record's format is also
//!   versioned by its own tag and magic.
//! - **Records of content, every build's**. Beside the builds', one directory holds records named
//!   by the content they describe, which hold no translation: the driver's data of each family of
//!   makings (above), and the libraries pipeline records name (`libraries.rs`).
//! - **A wrong entry is never used.** An entry's name is the SHA-256 of its build, kind, key and
//!   profile fingerprint, and a read is a hit only if the envelope's versions, kind, key and
//!   profile equal the request's and its checksum holds (`from_bytes_for_key`). Anything else is
//!   a miss and the file is removed.
//! - **Bounded.** [`ShaderCacheOptions::capacity`] caps the directory (25 GiB by default; 1 GiB
//!   before, which the consumer's boots filled, removing records they needed); the least
//!   recently used entries go first, and at device creation other builds' directories unused for
//!   30 days go (`store.rs`, on the device's cache thread, `worker.rs`). A device's first removal
//!   past the cap and its first failed write are each said once as a warning
//!   (`metal2vulkan-shader-cache-evicted`, and the degrade notice
//!   [`Degradation::ShaderCacheEntryNotWritten`](crate::Degradation::ShaderCacheEntryNotWritten))
//!   naming the directory and its cap; neither ends any work.
//! - **Off the creating thread.** An entry's bytes are encoded by the creation that made it,
//!   which needs their length for the memo's weighing, and the directory's write of them -- the
//!   `mkdir`, the temporary file, the write and the `rename` -- is the device's cache thread's
//!   (`worker.rs`). The queue is bounded at [`QUEUED_WRITES`](worker::QUEUED_WRITES) entries and
//!   [`QUEUED_BYTES`](worker::QUEUED_BYTES) bytes;
//!   a creation that meets either bound writes its entry itself, as every creation once did. A
//!   queued entry answers this device's lookups from the queue, so nothing is compiled again for
//!   want of a file not written yet, and what is queued when the device goes is written before its
//!   drop returns. A process that ends without dropping its devices loses what is still queued:
//!   at most the bound, and each of them an entry the next process compiles again.
//! - **In memory first.** Before the directory, every lookup asks the device's memo
//!   (`pipeline/memo.rs`), which [`ShaderCacheOptions::memory_capacity`] bounds (256 MiB by
//!   default). An entry read or written here is kept there, weighed at its encoded length; one
//!   kept without being encoded, at what it retains (`RenderPipeline::footprint`).
//! - **A question about the directory is kept too.** Whether the directory holds a stage's
//!   compiled function is a fact of that function's key and the device's profile, which a render
//!   variant's link asks once per stage; the answer is kept under exactly that key (`held.rs`),
//!   so a sweep of pipeline states over one function pair asks it once rather than once a state.

mod creation;
mod libraries;
/// The name of a creation's record under a runtime version, which `tests::shader_cache` reads.
#[cfg(test)]
pub(crate) use creation::record_name;
/// The variant records this process gave the directory, which `tests::render_stage_digests`'
/// count reads (`creation.rs`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use creation::variant_records_written;
pub(crate) use creation::VariantStages;
pub(crate) mod driver;
mod held;
pub(super) mod store;
pub(super) mod worker;

use super::error::{invalid, Error, Problem, ShaderCacheError};
use super::pipeline::memo::PipelineMemo;
use super::Device;
use crate::MessageSink;
use ash::vk;
use creation::CreationMemo;
use driver::{LiveCaches, Making, Records};
use metal2vulkan::cache::{Hash256, PipelineArtifact, PipelineKey};
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use store::{count, entry_name, Directory, Looked, Lookup, Store};
use worker::{Asked, Worker};

/// The default cap of a cache directory, in bytes: 25 GiB.
pub const DEFAULT_CAPACITY: u64 = 25 << 30;

/// The default cap of a device's memo of compiled pipelines, in bytes: 256 MiB.
pub const DEFAULT_MEMORY_CAPACITY: u64 = 256 << 20;

/// Where a device keeps its translated shaders, and how much it keeps. No Metal counterpart:
/// Metal's shader cache is the OS's.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderCacheOptions {
    /// The cache directory; `None` is the platform's per-user cache directory (the module's
    /// documentation names it). Devices of different processes and builds may share one.
    pub directory: Option<PathBuf>,
    /// The most bytes the directory holds, every build's entries together; past it the least
    /// recently used entries are removed, the device's first removal said once as a warning.
    /// [`DEFAULT_CAPACITY`] (25 GiB) by default. Not zero.
    pub capacity: u64,
    /// The most bytes of translated functions and compiled pipelines the device keeps in
    /// memory, each counted at its encoded length, or at what it retains where it was kept
    /// without being encoded; past it, or past 4096 of them, the least recently used are
    /// dropped. A live pipeline state keeps what it uses whatever this says. Not zero.
    pub memory_capacity: u64,
}

impl Default for ShaderCacheOptions {
    fn default() -> ShaderCacheOptions {
        ShaderCacheOptions {
            directory: None,
            capacity: DEFAULT_CAPACITY,
            memory_capacity: DEFAULT_MEMORY_CAPACITY,
        }
    }
}

impl ShaderCacheOptions {
    /// The cache in `directory`, with the default caps.
    pub fn in_directory(directory: impl Into<PathBuf>) -> ShaderCacheOptions {
        ShaderCacheOptions {
            directory: Some(directory.into()),
            ..ShaderCacheOptions::default()
        }
    }
}

/// The most stages a graphics pipeline has (vertex, two tessellation stages, geometry,
/// fragment; or task, mesh and fragment): a making reports each one's feedback where its
/// pipeline has no more ([`found_in_cache`]).
pub(crate) const REPORTED_STAGES: usize = 5;

/// What a making's `VK_EXT_pipeline_creation_feedback` says of its pipeline: found in the cache
/// the making handed the driver, `Some(true)`; compiled, `Some(false)`; `None` where the whole
/// pipeline's feedback is not valid, so the driver reported nothing.
///
/// Found where the whole pipeline's feedback says `APPLICATION_PIPELINE_CACHE_HIT`, or where
/// `stages` (each stage's feedback, empty where the making chained none) is not empty and every
/// one is valid and says it. The specification lets a driver say it either way:
/// `APPLICATION_PIPELINE_CACHE_HIT` "specifies that a readily usable pipeline or pipeline stage
/// was found in the `pipelineCache` specified by the application in the pipeline creation
/// command", and "when an implementation sets this bit for the entire pipeline, it may leave it
/// unset for any stage" (`VkPipelineCreationFeedbackFlagBits`). A driver that sets it for the
/// whole pipeline counts as before; MoltenVK 1.4.1 sets it for each stage alone, where it found
/// the stage's converted Metal function in the cache (`MVKPipelineCache::getShaderLibraryImpl`),
/// and leaves the whole pipeline's feedback valid alone (`MVKGraphicsPipeline`,
/// `MVKComputePipeline`).
pub(crate) fn found_in_cache(
    whole: &vk::PipelineCreationFeedback,
    stages: &[vk::PipelineCreationFeedback],
) -> Option<bool> {
    let valid = vk::PipelineCreationFeedbackFlags::VALID;
    let hit = vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT;
    if !whole.flags.contains(valid) {
        return None;
    }
    let every_stage =
        !stages.is_empty() && stages.iter().all(|stage| stage.flags.contains(valid | hit));
    Some(whole.flags.contains(hit) || every_stage)
}

/// What a device's shader cache directory cost it ([`ShaderCache::directory_time`]), which
/// `WorkCounts` reports and [`ShaderCacheStatistics`] does not. Not part of the API.
pub(super) struct DirectoryTime {
    /// `WorkCounts::shader_cache_read_nanos`.
    pub(super) read_nanos: u64,
    /// `WorkCounts::shader_cache_write_nanos`.
    pub(super) write_nanos: u64,
    /// `WorkCounts::shader_cache_off_thread_nanos`.
    pub(super) off_thread_nanos: u64,
    /// `WorkCounts::shader_cache_writes_on_thread`.
    pub(super) writes_on_thread: u64,
}

/// A device's shader cache counts since it was made ([`Device::shader_cache_statistics`]). No
/// Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderCacheStatistics {
    /// Artifacts read and accepted: a translation or pipeline compile not run.
    pub hits: u64,
    /// Artifact lookups that found nothing usable, for any reason; each was compiled.
    pub misses: u64,
    /// Of [`hits`](Self::hits), those of a translated function: a creation asks for one per stage
    /// it compiles. The three kinds are counted apart because only the kind says which artifact a
    /// lookup decoded, and they differ in size and in how often a creation stores one -- a
    /// creation that makes one answered lookup and one unanswered one cannot be read off the
    /// whole counts.
    pub function_hits: u64,
    /// Of [`misses`](Self::misses), those of a translated function: each one a function
    /// translated.
    pub function_misses: u64,
    /// Of [`hits`](Self::hits), those of a compiled compute pipeline: a compute or tile state's
    /// whole artifact, which such a creation stores on every miss
    /// (`pipeline/sources.rs`'s `kernel`).
    pub compute_pipeline_hits: u64,
    /// Of [`misses`](Self::misses), those of a compiled compute pipeline.
    pub compute_pipeline_misses: u64,
    /// Of [`hits`](Self::hits), those of a compiled render pipeline: a render state's whole
    /// artifact, which such a creation stores only where linking a variant again would translate
    /// (`pipeline/sources.rs`'s `linked_render`), so a warm render creation decodes none of
    /// these.
    pub render_pipeline_hits: u64,
    /// Of [`misses`](Self::misses), those of a compiled render pipeline.
    pub render_pipeline_misses: u64,
    /// Entries found under the right name whose contents failed the checks, each removed.
    pub rejected: u64,
    /// Entries that could not be read.
    pub read_failures: u64,
    /// Translations the directory was given: counted when a creation hands them over, which for
    /// all but a write past the queue's bound is before the file exists. A write that then fails
    /// counts in [`write_failures`](Self::write_failures) too.
    pub writes: u64,
    /// Entries and records of every kind -- translations, a creation's answers, the driver's
    /// pipeline cache data -- that the device's cache thread took, and the creating thread did
    /// not write itself.
    pub writes_queued: u64,
    /// Those a thread of the program wrote itself, because the queue was at one of its bounds or
    /// the entry was larger than the whole bound, or because the store has no cache thread. A
    /// creation pays four journalled filesystem calls for each of these and none for a queued
    /// one.
    pub writes_on_thread: u64,
    /// The bytes of every entry and record actually written, the driver's pipeline cache data
    /// among them: what the directory was made to hold, whichever thread wrote it.
    pub written_bytes: u64,
    /// Lookups answered from an entry the cache thread had not written yet: a translation or
    /// compile not run, counted in [`hits`](Self::hits) as a read of the file is.
    pub queued_hits: u64,
    /// Entries that could not be written, by whichever thread wrote them.
    pub write_failures: u64,
    /// Entries removed to keep the directory under its cap.
    pub evictions: u64,
    /// Records whose `VkPipelineCache` started from the record's saved data: one per family of
    /// makings the device makes a cache for, not one per making (it was whether the device's one
    /// driver cache did, 0 or 1).
    pub driver_loads: u64,
    /// Pipelines of the program's states that the driver reports it found in the cache their
    /// making was handed (`VK_EXT_pipeline_creation_feedback`, where the
    /// device lists the extension): the whole pipeline's feedback is valid and says
    /// `APPLICATION_PIPELINE_CACHE_HIT`, or every stage's feedback is valid and says it. Drivers
    /// differ in where they say it: most for the whole pipeline; MoltenVK 1.4.1 for each stage
    /// alone, whose Metal function it found converted in the cache (its whole pipeline's
    /// feedback is only ever valid).
    pub driver_pipeline_hits: u64,
    /// Such pipelines whose valid feedback says no hit, for the whole pipeline or for every
    /// stage: compiled, or a stage converted again.
    pub driver_pipeline_misses: u64,
    /// Records of the driver's pipeline cache data written: one per making that added to its
    /// record's cache (no record, or its pipeline not in it), so a family whose makings convert
    /// the same stages writes its record once. (It was the saves of the device's one driver
    /// cache.)
    pub driver_saves: u64,
    /// Reads of the driver's pipeline cache data refused because the data ends inside the
    /// header it states, at a making's save. Such data is never kept, so a record's cache
    /// starts empty in every process. A driver defect.
    pub driver_data_truncated: u64,
    /// Reads of the driver's pipeline cache data refused because its header is not this
    /// device's version-one header (version, vendor, device or UUID), counted as
    /// [`driver_data_truncated`](Self::driver_data_truncated) is.
    pub driver_data_mismatched: u64,
    /// Reads of the driver's pipeline cache data whose SHA-256 was that of what the cache was
    /// loaded with or last wrote: nothing written. With
    /// [`driver_saves`](Self::driver_saves) and the two refusals,
    /// one read per making the runtime finished while the process was not exiting and did not
    /// count in [`driver_data_found`](Self::driver_data_found).
    pub driver_data_unchanged: u64,
    /// Makings handed a cache loaded from their record that the driver reports found there (the
    /// whole pipeline's `APPLICATION_PIPELINE_CACHE_HIT`, or every stage's), whose data is not
    /// read back:
    /// nothing written. With the reads, one per making the runtime finished while the process
    /// was not exiting.
    pub driver_data_found: u64,
}

/// What an entry holds, in its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum EntryKind {
    Function = 1,
    ComputePipeline = 2,
    // 3 named the device's one driver cache entry, retired.
    RenderPipeline = 4,
    // 5 named a pipeline's kept graphics keys, removed. 6 named a variant's own pipeline cache,
    // retired.
    /// A pipeline making's driver data (`driver.rs`), in the content directory.
    PipelineData = 7,
    /// A render descriptor's linked-use answers (`creation.rs`).
    LinkedAnswers = 8,
    /// A render descriptor's reflection (`creation.rs`).
    RenderReflection = 9,
    /// The pipeline key of a render descriptor under a variant key (`creation.rs`).
    VariantPipeline = 10,
    /// A library a pipeline record names, in the content directory (`libraries.rs`).
    Library = 11,
}

/// The translations this build reads and writes: the SHA-256 of the crate's version, both artifact
/// versions and the translator's source digest. Not the runtime's own sources: every entry under
/// it is the translator's answer for its key (an artifact, a frame of linked-use answers or a
/// reflection) or a record of the translator's keys, under a name the runtime's own format tags
/// version, so a build whose translator and versions did not move reads its entries
/// (the module docs).
pub(crate) fn this_build() -> Hash256 {
    let mut hash = Sha256::new();
    hash.update(b"metal2vulkan shader cache build\0");
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    hash.update([0]);
    hash.update(metal2vulkan::cache::ARTIFACT_FORMAT_VERSION.to_le_bytes());
    hash.update(metal2vulkan::cache::SEMANTICS_VERSION.to_le_bytes());
    hash.update(metal2vulkan::cache::SOURCE_DIGEST.as_bytes());
    hash.finalize().into()
}

/// The platform's per-user cache directory for this crate.
fn default_directory() -> Result<PathBuf, ShaderCacheError> {
    let absolute = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    let home = || {
        absolute(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).ok_or_else(|| {
            ShaderCacheError::new(
                None,
                std::io::ErrorKind::NotFound,
                "no home directory for the default shader cache directory",
            )
        })
    };
    if cfg!(target_os = "macos") {
        return Ok(home()?.join("Library/Caches/metal2vulkan"));
    }
    match absolute("XDG_CACHE_HOME") {
        Some(base) => Ok(base.join("metal2vulkan")),
        None => Ok(home()?.join(".cache/metal2vulkan")),
    }
}

/// A device's shader cache as its options set it up, before the device is made: the directory
/// opened for this build, and the memo's cap.
#[derive(Clone, Debug)]
pub(super) struct CacheSetup {
    directory: Directory,
    memory_capacity: u64,
}

impl CacheSetup {
    /// The cache `options` describe: its directory opened for this build. `call` names the
    /// creating call.
    pub(super) fn of(
        options: &ShaderCacheOptions,
        call: &'static str,
    ) -> Result<CacheSetup, Error> {
        if options.capacity == 0 || options.memory_capacity == 0 {
            return Err(invalid(call, Problem::ShaderCacheCapacity));
        }
        let root = match &options.directory {
            Some(directory) => directory.clone(),
            None => default_directory().map_err(Error::ShaderCache)?,
        };
        let directory =
            Directory::open(root, this_build(), options.capacity).map_err(|failed| {
                Error::ShaderCache(ShaderCacheError::new(
                    Some(failed.path),
                    failed.error.kind(),
                    &failed.error.to_string(),
                ))
            })?;
        Ok(CacheSetup {
            directory,
            memory_capacity: options.memory_capacity,
        })
    }
}

/// One device's shader cache: its memo, its view of the directory, what names its makings'
/// records, the `VkPipelineCache` it keeps for each record, and the cache thread.
pub(super) struct ShaderCache {
    /// The compiled pipelines kept in memory.
    pub(super) memo: PipelineMemo,
    /// What render creations worked out from their descriptors before their compiles, in
    /// memory (`creation.rs`).
    creation: CreationMemo,
    /// What the directory answered about the stage functions this device asked about
    /// (`held.rs`).
    held: held::HeldFunctions,
    store: Arc<Store>,
    worker: Worker,
    /// The libraries pipeline records name that this device knows the directory holds
    /// (`libraries.rs`).
    libraries: libraries::KeptLibraries,
    /// What names the makings' records (`driver.rs`).
    records: Records,
    /// The `VkPipelineCache` the device keeps for each record name, one per family of makings
    /// that convert the same stages (`driver.rs`).
    live: LiveCaches,
    /// Whether a making reports whether the driver found its pipeline in its cache
    /// (`VK_EXT_pipeline_creation_feedback`).
    driver_reports: bool,
}

impl ShaderCache {
    /// The cache `setup` describes, of the device of `properties` and profile fingerprint
    /// `profile`, with the cache thread started (which scans the directory). Its makings' records
    /// are named with `settings` (the digest of its instance's layer settings), and its makings
    /// report whether the driver found their pipelines where `feedback`
    /// (`VK_EXT_pipeline_creation_feedback`). (No driver cache is made or loaded here; the first
    /// making of each record's name makes that record's.) Its warnings go to `sink` (the
    /// instance's message sink), else on stderr.
    pub(super) fn start(
        setup: CacheSetup,
        properties: &vk::PhysicalDeviceProperties,
        profile: Hash256,
        (settings, feedback): (Hash256, bool),
        sink: Option<MessageSink>,
    ) -> Result<ShaderCache, Error> {
        let CacheSetup {
            directory,
            memory_capacity,
        } = setup;
        let asked = Arc::new(Asked::new());
        let store = Arc::new(Store::new(directory, sink, Some(Arc::clone(&asked))));
        let worker = Worker::start(Arc::clone(&store), asked)?;
        Ok(ShaderCache {
            memo: PipelineMemo::new(memory_capacity),
            creation: CreationMemo::new(),
            held: held::HeldFunctions::new(),
            store,
            worker,
            libraries: libraries::KeptLibraries::new(),
            records: Records {
                identity: driver::Identity::of(properties),
                settings,
                profile,
            },
            live: LiveCaches::default(),
            driver_reports: feedback,
        })
    }

    /// The device's view of the directory: what reads and writes its content records (the
    /// device's check of textures over host pages keeps its verdicts there, `in_place.rs`).
    pub(crate) fn content_store(&self) -> &Store {
        &self.store
    }

    /// Destroy the `VkPipelineCache` of every record the device holds, then stop the cache
    /// thread, which writes what is queued first: the device is going.
    ///
    /// # Safety
    ///
    /// The device of `raw` is live and every pipeline of every making is made: the device's
    /// teardown (`DeviceShared::drop`).
    pub(super) unsafe fn close(&self, raw: &ash::Device) {
        // SAFETY: the caller's.
        unsafe { self.live.destroy(raw) };
        self.worker.stop();
    }

    /// What `create` makes with the `VkPipelineCache` it is handed, its `vkCreate*Pipelines`
    /// call on the device of `raw` (`DeviceShared::vk`): the device's cache of the record
    /// `making` names, made by the first making of that name from the record where one loads,
    /// and kept until the device's teardown, whose data is written as that record where the
    /// making added to it (`driver.rs`); null where the driver makes none.
    /// The call holds a turn of the process's budget of makings in flight (`pipeline/budget.rs`):
    /// its own, waited for, or its pool piece's. `create` returns, with what it made, whether the
    /// driver reports its pipeline found in that cache ([`found_in_cache`], `None` where it
    /// reports nothing): one found in a cache loaded from its record reads no data back
    /// (`driver.rs`).
    pub(super) fn make_pipelines<T, E>(
        &self,
        raw: &ash::Device,
        making: Making,
        create: impl FnOnce(vk::PipelineCache) -> (Result<T, E>, Option<bool>),
    ) -> Result<T, E> {
        let _turn = crate::pipeline::budget::program_turn();
        let made = driver::make(
            &self.store,
            &self.records,
            &self.live,
            raw,
            making,
            crate::exit_guard::closed,
            create,
        );
        if self.store.over_capacity() {
            self.worker.evict();
        }
        made
    }

    /// The directory round trips the threads that asked have made: the questions about an entry
    /// (`store::Counters::holds_questions`) and the reads of one
    /// (`store::Counters::entry_reads`), each one filesystem call. What the device tests of a
    /// creation's directory round trips read: the count is a fact of the structure where
    /// `WorkCounts::shader_cache_read_nanos`, which times the very same calls, is a figure of the
    /// host and its load.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn directory_trips(&self) -> (u64, u64) {
        (
            self.store.counters.holds_questions.load(Ordering::Relaxed),
            self.store.counters.entry_reads.load(Ordering::Relaxed),
        )
    }

    /// Makings handed a `VkPipelineCache` the device already held for their record
    /// (`store::Counters::driver_caches_shared`): what the device tests of the sharing read.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn caches_shared(&self) -> u64 {
        self.store
            .counters
            .driver_caches_shared
            .load(Ordering::Relaxed)
    }

    /// Whether a making chains a `VkPipelineCreationFeedbackCreateInfo`, its whole pipeline's
    /// feedback and its stages', which [`driver_reported`](Self::driver_reported) counts.
    pub(crate) fn driver_reports(&self) -> bool {
        self.driver_reports
    }

    /// Count what a making reported of its pipeline, `whole`, and of its stages, `stages` (empty
    /// where it chained none): found in its cache, or compiled ([`found_in_cache`]); nothing
    /// where the driver left it unreported.
    pub(crate) fn driver_reported(
        &self,
        whole: &vk::PipelineCreationFeedback,
        stages: &[vk::PipelineCreationFeedback],
    ) {
        let counters = &self.store.counters;
        if let Some(found) = found_in_cache(whole, stages) {
            count(match found {
                true => &counters.driver_pipeline_hits,
                false => &counters.driver_pipeline_misses,
            });
        }
    }

    /// The function `compile` compiled under `key`, from the cache, and its encoded length.
    pub(super) fn load_function(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &FunctionCompileKey,
    ) -> Option<(CompiledFunction, u64)> {
        self.load(Looked::Function, key.digest(), compile, |bytes| {
            CompiledFunction::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Whether the directory holds the function of `key` for `compile`'s profile: its file is in
    /// place, or its write is queued for the cache thread (`worker.rs`), whose rename has not
    /// happened yet. A question about the directory, so it reads no contents, decodes nothing and
    /// counts in no hit or miss: a hit here is not a lookup answered, and a miss is not a
    /// translation. Its one `stat` is counted in the directory's read time
    /// (`WorkCounts::shader_cache_read_nanos`).
    ///
    /// **Asked of the directory once per function.** Nothing of the pipeline state reaches the
    /// name, so the answer is the function's and the profile's; a `true` is kept under those two
    /// (`held.rs`) and a sweep of states over one function pair asks the directory nothing after
    /// the creation that put its stages there.
    pub(super) fn holds_function(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &FunctionCompileKey,
    ) -> bool {
        let profile = compile.profile_fingerprint();
        self.held
            .holds((key.digest(), &profile), self.store.removals(), || {
                let name = entry_name(
                    self.store.directory().build(),
                    EntryKind::Function,
                    key.digest(),
                    &profile,
                );
                self.store.holds(&name)
            })
    }

    /// Keep `function`, which `compile` compiled; its encoded length.
    pub(super) fn store_function(
        &self,
        compile: &metal2vulkan::device::Device,
        function: &CompiledFunction,
    ) -> u64 {
        self.store(
            EntryKind::Function,
            function.key().digest(),
            compile,
            function.to_bytes(),
        )
    }

    /// The compute pipeline `compile` compiled under `key`, from the cache, and its encoded
    /// length.
    pub(super) fn load_compute_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(ComputePipeline, u64)> {
        self.load(Looked::ComputePipeline, key.digest(), compile, |bytes| {
            ComputePipeline::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Keep `pipeline`, which `compile` compiled; its encoded length.
    pub(super) fn store_compute_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        pipeline: &ComputePipeline,
    ) -> u64 {
        self.store(
            EntryKind::ComputePipeline,
            pipeline.key().digest(),
            compile,
            pipeline.to_bytes(),
        )
    }

    /// The render pipeline `compile` compiled under `key`, from the cache, and its encoded
    /// length.
    pub(super) fn load_render_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(RenderPipeline, u64)> {
        self.load(Looked::RenderPipeline, key.digest(), compile, |bytes| {
            RenderPipeline::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Keep `pipeline`, which `compile` compiled; its encoded length.
    pub(super) fn store_render_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        pipeline: &RenderPipeline,
    ) -> u64 {
        self.store(
            EntryKind::RenderPipeline,
            pipeline.key().digest(),
            compile,
            pipeline.to_bytes(),
        )
    }

    /// The entry of `looked`'s kind under `key` for `compile`'s profile, as `decode` accepts it,
    /// and its length. A miss when absent, unreadable, or refused (then removed). Counted in the
    /// whole hit and miss counts and in `looked`'s own.
    fn load<T, E>(
        &self,
        looked: Looked,
        key: &Hash256,
        compile: &metal2vulkan::device::Device,
        decode: impl FnOnce(&[u8]) -> Result<T, E>,
    ) -> Option<(T, u64)> {
        let counters = &self.store.counters;
        let name = entry_name(
            self.store.directory().build(),
            looked.kind(),
            key,
            &compile.profile_fingerprint(),
        );
        let found = match self.store.load(&name, |bytes| {
            decode(bytes).map(|value| (value, bytes.len() as u64))
        }) {
            Lookup::Hit(found) => Some(found),
            Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
        };
        let (whole, by) = match found.is_some() {
            true => (&counters.hits, &counters.hits_by),
            false => (&counters.misses, &counters.misses_by),
        };
        count(whole);
        count(&by[looked as usize]);
        found
    }

    /// Give `bytes` to the directory as the entry of `kind` under `key` for `compile`'s profile
    /// (the cache thread writes it, `worker.rs`); their length.
    fn store(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &metal2vulkan::device::Device,
        bytes: Vec<u8>,
    ) -> u64 {
        let counters = &self.store.counters;
        let name = entry_name(
            self.store.directory().build(),
            kind,
            key,
            &compile.profile_fingerprint(),
        );
        let length = bytes.len() as u64;
        match self.store.write_soon(&name, bytes) {
            Ok(_) => count(&counters.writes),
            Err(_) => self.store.write_failed(),
        }
        if self.store.over_capacity() {
            self.worker.evict();
        }
        length
    }

    /// What the directory cost this device: its time, the asking threads' apart from the cache
    /// thread's, and the entries an asking thread wrote itself
    /// (`WorkCounts::shader_cache_read_nanos` and its fellows).
    pub(super) fn directory_time(&self) -> DirectoryTime {
        let counters = &self.store.counters;
        let read = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
        DirectoryTime {
            read_nanos: read(&counters.read_nanos),
            write_nanos: read(&counters.write_nanos),
            off_thread_nanos: read(&counters.off_thread_nanos),
            writes_on_thread: read(&counters.writes_on_thread),
        }
    }

    /// The counts so far.
    fn statistics(&self) -> ShaderCacheStatistics {
        let counters = &self.store.counters;
        let read = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
        ShaderCacheStatistics {
            hits: read(&counters.hits),
            misses: read(&counters.misses),
            function_hits: read(&counters.hits_by[Looked::Function as usize]),
            function_misses: read(&counters.misses_by[Looked::Function as usize]),
            compute_pipeline_hits: read(&counters.hits_by[Looked::ComputePipeline as usize]),
            compute_pipeline_misses: read(&counters.misses_by[Looked::ComputePipeline as usize]),
            render_pipeline_hits: read(&counters.hits_by[Looked::RenderPipeline as usize]),
            render_pipeline_misses: read(&counters.misses_by[Looked::RenderPipeline as usize]),
            rejected: read(&counters.rejected),
            read_failures: read(&counters.read_failures),
            writes: read(&counters.writes),
            writes_queued: read(&counters.writes_queued),
            writes_on_thread: read(&counters.writes_on_thread),
            written_bytes: read(&counters.written_bytes),
            queued_hits: read(&counters.queued_hits),
            write_failures: read(&counters.write_failures),
            evictions: read(&counters.evictions),
            driver_loads: read(&counters.driver_loads),
            driver_pipeline_hits: read(&counters.driver_pipeline_hits),
            driver_pipeline_misses: read(&counters.driver_pipeline_misses),
            driver_saves: read(&counters.driver_saves),
            driver_data_truncated: read(&counters.driver_data_truncated),
            driver_data_mismatched: read(&counters.driver_data_mismatched),
            driver_data_unchanged: read(&counters.driver_data_unchanged),
            driver_data_found: read(&counters.driver_data_found),
        }
    }
}

impl Device {
    /// The device's shader cache counts since it was made. No Metal counterpart.
    pub fn shader_cache_statistics(&self) -> ShaderCacheStatistics {
        self.shared().shader_cache.statistics()
    }

    /// The device's shader cache directory. No Metal counterpart.
    pub fn shader_cache_directory(&self) -> &Path {
        self.shared().shader_cache.store.directory().root()
    }
}
