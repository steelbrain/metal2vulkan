//! Kernels past the stage bound: compiled in the consumer's helper process, created here from the
//! helper's cache data alone, never compiled in this process.
//!
//! A kernel whose module weighs more than `metal2vulkan::pipeline::COMPILED_STAGE_WEIGHT`
//! (`ModuleStats::entry_weight`) is one whose compile may take the host's memory, so the process
//! that runs the program never compiles it. Each of its `VkPipeline`s is made from a
//! `VkPipelineCache` made of that pipeline's own entry in the cache directory alone, with
//! `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` ([`made`]), and the cache destroyed
//! once the create returns: the device's own pipeline cache and its blob never hold it. The kernel
//! keeps its module's words and no shader module: each such create is of a shader module made of
//! the words specialized for it alone ([`Create::specialize`]), destroyed with the cache, as Vulkan
//! lets a module go while pipelines made of it are in use, and two makes of one kernel's
//! pipelines at once each make their own. A request carries those words, and the helper compiles
//! them as they are.
//! Where the directory holds no entry for the pipeline, the make (or the dispatch, or
//! [`ComputePipelineState::prepare_dispatch`](crate::ComputePipelineState::prepare_dispatch)) is
//! refused with [`Error::StageUnfilled`], carrying a [`StageRequest`]: the consumer runs its helper
//! process, hands its pipes to [`Device::fill_stage`], which keeps the helper's cache data as the
//! entry, and makes again. The helper's main is [`serve_stage`]. The runtime starts, caps, times
//! and kills no process: the consumer owns the helper, and [`Device::record_stage_breach`] keeps
//! what it ended one for.
//!
//! **The entry.** One file per pipeline under `stages/` in the cache directory, named by the hex of
//! the pipeline's stage key: a digest of the wire version, the device's pipeline cache header, the
//! device's enables, the module's words, its entry point, stage flags, pinned subgroup size and
//! specialization. It holds, framed as the blob is (magic, length, checksum), the key again and
//! either the pipeline's cache data, as the helper's `vkGetPipelineCacheData` gave it, or a breach:
//! what the helper's compile met ([`StageBreach`]). A file that does not check, names another key,
//! or holds data that does not begin with the device's header is a miss and is removed. A hit
//! touches the file's modification time; each write removes the entries past [`STAGE_BOUNDS`],
//! oldest use first. Nothing of an entry stays in memory past the call that reads or writes it.
//!
//! **A breach is final for its driver.** An entry that holds a breach refuses the pipeline
//! ([`Gap::StageSize`]) with no helper asked again, and so does one whose cache data the driver
//! answered compile-required for: that answer is kept as a breach. The key holds the driver's
//! header, so a new driver version asks again. A [`StageBreach::Memory`] entry also holds the
//! memory ceiling of the consumer's helper it was kept under
//! ([`DeviceOptions::stage_memory_ceiling`]), 0 where it holds the breach's byte alone: on a
//! device given a higher ceiling it is a miss, and removed. A [`StageBreach::Ended`] entry, a
//! helper that ended before its reply with no end of the consumer's, holds the [`HELPER_VERSION`]
//! it was kept under: under a later version it is a miss, and removed.
//!
//! **The wire.** [`Device::fill_stage`] writes one request frame to the helper and reads one reply
//! frame back: each the magic `m2vstage`, the wire version, the payload's length and its checksum,
//! then the payload. A reply of another wire version is [`StageFill::Version`].

use crate::cache::{checksum, frame, hex, read_whole, unframe, write_whole};
use crate::counters::{Gauge, Metric};
use crate::device::{Core, Device, DeviceOptions};
use crate::error::{Error, Gap, StageGap};
use crate::instance::Instance;
use crate::times::host_time;
use ash::vk;
use metal2vulkan::pipeline::specialize;
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// The bytes and the entries the `stages/` directory holds at most: 256 MiB and 32 entries,
/// sixteen heavy kernels at two pipelines each at about 7.6 MB, the largest entry measured (an RTX
/// 2080's, NVIDIA 620.70.06).
pub(crate) const STAGE_BOUNDS: StageBounds = StageBounds {
    bytes: 256 << 20,
    entries: 32,
};

/// The bounds of the `stages/` directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StageBounds {
    pub(crate) bytes: u64,
    pub(crate) entries: usize,
}

/// The directory of the entries, under the cache directory.
const STAGES: &str = "stages";
/// The first bytes of a frame on the wire.
const WIRE: &[u8; 8] = b"m2vstage";
/// The wire's version: a frame of another is not read.
const WIRE_VERSION: u32 = 3;
/// The longest payload a frame is read with.
const MAX_PAYLOAD: u64 = 1 << 30;
/// An entry's kinds, after its key.
const DATA: u8 = 0;
const BREACH: u8 = 1;
/// A reply's statuses.
const DONE: u32 = 0;
const COMPILE: u32 = 1;
const NO_DEVICE: u32 = 2;
const ENABLES: u32 = 3;
/// The version of the helper's own code ([`serve_stage`] and what it runs in this module: `serve`,
/// `compile`, `read_request`), which a [`StageBreach::Ended`] entry holds: one kept under a lower
/// version is a miss and removed, so a fix of ours to a helper that ended asks the helper again.
/// Bump it with every change to that code that changes what the helper does; a driver's fix needs
/// none, as the stage key holds the driver's header.
pub(crate) const HELPER_VERSION: u32 = 3;

/// No Metal counterpart: what ended a helper's compile of a kernel past the stage bound, kept as
/// the pipeline's entry ([`Device::record_stage_breach`], [`Device::fill_stage`]) so its makes are
/// refused ([`Gap::StageSize`]) with no helper asked again, until the driver's version changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageBreach {
    /// The helper passed the memory cap the consumer gave it. Kept with the device's
    /// [`DeviceOptions::stage_memory_ceiling`], and refuses only on a device given that ceiling
    /// or a lower one: one given a higher ceiling asks the helper again.
    Memory,
    /// The helper passed the time the consumer gave it.
    Time,
    /// The driver failed the helper's compile (`vkCreateComputePipelines` answered an error).
    Compile,
    /// The driver answered compile-required for the helper's cache data
    /// (`VK_PIPELINE_COMPILE_REQUIRED`): it would compile again in this process.
    CompileRequired,
    /// No Metal counterpart, as Metal compiles in its own compiler service: the helper ended
    /// before its whole reply ([`StageFill::Ended`]) and the consumer did not end it (no cap, no
    /// time limit, no cancel), as a helper whose driver faults in the compile does. Kept with the
    /// version of the runtime's helper code it ended under: a later runtime whose helper code
    /// changed asks the helper again.
    Ended,
}

impl StageBreach {
    const ALL: [StageBreach; 5] = [
        StageBreach::Memory,
        StageBreach::Time,
        StageBreach::Compile,
        StageBreach::CompileRequired,
        StageBreach::Ended,
    ];

    fn byte(self) -> u8 {
        match self {
            StageBreach::Memory => 0,
            StageBreach::Time => 1,
            StageBreach::Compile => 2,
            StageBreach::CompileRequired => 3,
            StageBreach::Ended => 4,
        }
    }
}

/// No Metal counterpart: the helper's own times in one fill ([`Device::fill_stage`]), from its
/// host clock, for the consumer to log beside its own (the helper's start, the wait on its pipe).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageSpans {
    /// The helper's device creation ([`Device::new`]), in nanoseconds.
    pub device_nanos: u64,
    /// The helper's compile: its one `vkCreateComputePipelines` call, through an empty
    /// `VkPipelineCache`, in nanoseconds.
    pub compile_nanos: u64,
}

/// No Metal counterpart: why [`Device::fill_stage`] kept no cache data. Only
/// [`StageFill::Compile`] is kept, as a breach.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageFill {
    /// The helper's reply did not come whole: it ended, was killed (a cap, a time limit, a
    /// cancel) or closed its pipes first. Nothing is kept; the consumer keeps the end with
    /// [`Device::record_stage_breach`]: a cap's as [`StageBreach::Memory`], a time limit's as
    /// [`StageBreach::Time`], and the helper's own end, one the consumer did not cause, as
    /// [`StageBreach::Ended`]. A cancel keeps nothing.
    Ended,
    /// The helper speaks another wire version: a helper of another build of the runtime.
    Version,
    /// The helper found no device of the request's `deviceUUID` and `driverUUID`, could not open
    /// it, or gave cache data of another driver's header.
    NoDevice,
    /// The helper's device enabled other extensions or features than this one: it was opened
    /// with other [`DeviceOptions`].
    Enables,
    /// The driver failed the helper's compile with this result; kept as
    /// [`StageBreach::Compile`].
    Compile(vk::Result),
    /// The cache directory took no entry: it is read-only, or the write failed.
    Unkept,
}

impl fmt::Display for StageFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageFill::Ended => f.write_str("the helper ended before its reply"),
            StageFill::Version => f.write_str("the helper speaks another wire version"),
            StageFill::NoDevice => f.write_str("the helper opened no device of the request's"),
            StageFill::Enables => f.write_str("the helper's device enabled other features"),
            StageFill::Compile(result) => write!(f, "the helper's compile failed: {result}"),
            StageFill::Unkept => f.write_str("the cache directory took no entry"),
        }
    }
}

/// A kernel's module past the stage bound, kept with its kernel in place of a shader module: the
/// words each make of its `VkPipeline`s makes a shader module of its own from ([`made`]), and
/// that a request carries.
pub(crate) struct Heavy {
    /// The module's entry weight.
    weight: u64,
    pub(crate) words: Arc<[u32]>,
    /// The digest of the words, which every stage key holds.
    digest: [u8; 32],
}

/// The entry weight past which a kernel is compiled only in the consumer's helper: the translator's
/// `COMPILED_STAGE_WEIGHT`, or, in a device test, one the test sets (`TEST_BOUND`), so a small
/// kernel passes it. A render stage is weighed against [`RENDER_STAGE_WEIGHT`].
pub(crate) fn bound() -> u64 {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let Some(bound) =
        std::num::NonZeroU64::new(TEST_BOUND.load(std::sync::atomic::Ordering::Relaxed))
    {
        return bound.get();
    }
    metal2vulkan::pipeline::COMPILED_STAGE_WEIGHT
}

/// The heaviest entry weight of a render pipeline's stage this process makes: no helper compiles a
/// render stage, so one that weighs more is refused (`Gap::StageSize`, `StageGap::RenderStage`),
/// and one at or under it compiles in this process with no cap, past the kernels' [`bound`] too.
/// The two numbers measure two things: where a kernel leaves this process, and where a render
/// stage is refused because nothing else can compile it. They are one number again when a helper
/// compiles render stages. Measured cold and alone in a process of its own, the heaviest render
/// stage of a real program, of entry weight 35,063 beside a vertex stage of 431 (39,424 as
/// lavapipe's profile weighs it), grew the process by 316 MiB in 3.8 s on ANV (Mesa 26.2.6, the
/// Iris Xe of RPL-P; job m3-max-66048); no render stage between 39,424 and this weight is measured.
pub(crate) const RENDER_STAGE_WEIGHT: u64 = 99_109;

/// The stage bound a device test sets; none where zero ([`bound`]).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) static TEST_BOUND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Heavy {
    /// The module of `stage` where it weighs more than the stage bound ([`bound`]).
    pub(crate) fn of(stage: &metal2vulkan::pipeline::StageModule) -> Option<Heavy> {
        let weight = stage.stats.entry_weight;
        (weight > bound()).then(|| Heavy {
            weight,
            words: Arc::clone(&stage.spirv),
            digest: words_digest(&stage.spirv),
        })
    }
}

/// What a compute `VkPipeline`'s create gives its one stage besides the module.
pub(crate) struct Create<'a> {
    pub(crate) entry: &'a CStr,
    pub(crate) flags: vk::PipelineShaderStageCreateFlags,
    pub(crate) subgroup: Option<u32>,
    /// The specialization constants' ids, each 4 bytes of `data` in order: what the module is
    /// specialized with ([`Create::specialize`]).
    pub(crate) ids: &'a [u32],
    pub(crate) data: &'a [u8],
}

impl Create<'_> {
    /// `words`, the translator's module, with each specialization constant the fixed constant of
    /// its value here (`metal2vulkan::pipeline::specialize`): the words a shader module of the
    /// create is made of, so no driver is handed a specialization constant, and a kernel's
    /// workgroup size is its entry point's `LocalSize` literals.
    pub(crate) fn specialize(&self, words: &[u32]) -> Result<Vec<u32>, Error> {
        let (data, _) = self.data.as_chunks::<4>();
        let values: Vec<(u32, u64)> = (self.ids.iter().zip(data))
            .map(|(&id, bytes)| (id, u64::from(u32::from_ne_bytes(*bytes))))
            .collect();
        specialize(words, &values).map_err(Error::translation)
    }
}

/// No Metal counterpart: a `VkPipeline` of a kernel past the stage bound whose cache data the
/// device's cache directory does not hold ([`Error::StageUnfilled`]): what the consumer's helper
/// compiles ([`Device::fill_stage`]). Requests of one pipeline compare equal, so a consumer runs
/// one helper for them.
#[derive(Clone)]
pub struct StageRequest(Arc<Request>);

/// What a request holds.
struct Request {
    key: [u8; 32],
    weight: u64,
    header: [u8; 32],
    uuids: [[u8; 16]; 2],
    enables: [u8; 32],
    /// The module's words specialized for the pipeline ([`Create::specialize`]).
    words: Arc<[u32]>,
    entry: CString,
    flags: u32,
    subgroup: Option<u32>,
}

impl StageRequest {
    /// The module's entry weight (`ModuleStats::entry_weight`), past the compiled stage bound.
    pub fn weight(&self) -> u64 {
        self.0.weight
    }
}

impl PartialEq for StageRequest {
    fn eq(&self, other: &StageRequest) -> bool {
        self.0.key == other.0.key
    }
}

impl Eq for StageRequest {}

impl std::hash::Hash for StageRequest {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.key.hash(state);
    }
}

impl fmt::Debug for StageRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageRequest")
            .field("key", &hex(&self.0.key))
            .field("weight", &self.0.weight)
            .finish()
    }
}

/// The SHA-256 of `words`' bytes, each word least significant byte first, read a page at a time,
/// so no copy of the module is made.
fn words_digest(words: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut page = [0u8; 4096];
    for chunk in words.chunks(page.len() / 4) {
        for (bytes, word) in page.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
            *bytes = word.to_le_bytes();
        }
        h.update(&page[..chunk.len() * 4]);
    }
    h.finalize().into()
}

/// The `deviceUUID` and `driverUUID` of `physical`.
fn uuids(instance: &ash::Instance, physical: vk::PhysicalDevice) -> [[u8; 16]; 2] {
    let mut ids = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
    // SAFETY: the instance listed the device, at the floor's Vulkan 1.3, which defines the
    // structure.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    [ids.device_uuid, ids.driver_uuid]
}

/// The stage key of a pipeline of `heavy` made with `create` on a device of `header` and
/// `enables` (the module docs).
fn key(header: &[u8; 32], enables: &[u8; 32], heavy: &Heavy, create: &Create<'_>) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(WIRE);
    h.update(WIRE_VERSION.to_le_bytes());
    h.update(header);
    h.update(enables);
    h.update(heavy.digest);
    let entry = create.entry.to_bytes();
    h.update((entry.len() as u64).to_le_bytes());
    h.update(entry);
    h.update(create.flags.as_raw().to_le_bytes());
    h.update(create.subgroup.unwrap_or(0).to_le_bytes());
    h.update((create.ids.len() as u64).to_le_bytes());
    for id in create.ids {
        h.update(id.to_le_bytes());
    }
    h.update(create.data);
    h.finalize().into()
}

/// The refusal of a pipeline of `weight` for `cause`.
fn gap(weight: u64, cause: StageGap) -> Error {
    Error::Unsupported(Gap::StageSize { weight, cause })
}

/// A compute `VkPipeline` of `heavy` that `make` creates through the cache and of the shader
/// module it is given, with `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` in its
/// create info: through a cache made of the pipeline's entry alone, and of a module made of
/// `words`, `heavy`'s words specialized for this create ([`Create::specialize`]), alone, both
/// destroyed once `make` returns (the module docs). Counted as made, and its time
/// (`Metrics::pipeline_creation_nanos`).
///
/// Err: [`Error::StageUnfilled`] where the directory holds no entry for it; [`Gap::StageSize`]
/// where it holds a breach, where the driver answers compile-required for its data (kept as a
/// breach), on a device with no cache directory or whose driver writes no pipeline cache header;
/// [`Error::Vulkan`] where the cache's, the module's or the pipeline's create fails otherwise.
pub(crate) fn made(
    core: &Core,
    heavy: &Heavy,
    create: &Create<'_>,
    words: Vec<u32>,
    make: impl FnOnce(
        vk::PipelineCache,
        vk::ShaderModule,
    ) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) -> Result<vk::Pipeline, Error> {
    let weight = heavy.weight;
    let dir = (core.caches.dir()).ok_or_else(|| gap(weight, StageGap::NoCacheDirectory))?;
    let dir = dir.join(STAGES);
    let header = (core.caches.header()).ok_or_else(|| gap(weight, StageGap::NoPipelineCache))?;
    let enables = core.enables();
    let key = key(&header, &enables, heavy, create);
    let data = match read_entry(&dir, &key, &header, core.stage_memory_ceiling) {
        None => {
            let (instance, physical) = (core.instance.raw().1, core.physical);
            return Err(Error::StageUnfilled(StageRequest(Arc::new(Request {
                key,
                weight,
                header,
                uuids: uuids(instance, physical),
                enables,
                words: words.into(),
                entry: create.entry.to_owned(),
                flags: create.flags.as_raw(),
                subgroup: create.subgroup,
            }))));
        }
        Some(Entry::Breach(breach, _)) => return Err(gap(weight, StageGap::Breach(breach))),
        Some(Entry::Data(data)) => data,
    };
    let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
    // SAFETY: the data is what the helper's `vkGetPipelineCacheData` gave for a device of this
    // header, whole, as its frame's length and checksum show; the driver ignores data it does not
    // take.
    let cache = unsafe { core.device.create_pipeline_cache(&info, None) }.map_err(|result| {
        Error::Vulkan {
            call: "vkCreatePipelineCache",
            result,
        }
    })?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: the words are the translator's SPIR-V with its specialization constants fixed, for
    // the features device creation enabled.
    let module = match unsafe { core.device.create_shader_module(&info, None) } {
        Ok(module) => module,
        Err(result) => {
            // SAFETY: the cache is the device's, and nothing uses it.
            unsafe { core.device.destroy_pipeline_cache(cache, None) };
            return Err(Error::Vulkan {
                call: "vkCreateShaderModule",
                result,
            });
        }
    };
    let began = host_time();
    let created = make(cache, module);
    core.counters
        .pipeline_creation(host_time().saturating_sub(began));
    // SAFETY: the cache and the module are the device's, and the create that used them has
    // returned; a module may be destroyed while pipelines made of it are in use.
    unsafe {
        core.device.destroy_pipeline_cache(cache, None);
        core.device.destroy_shader_module(module, None);
    }
    match created {
        Ok(pipelines) => {
            core.counters.count(Metric::PipelinesCreated);
            core.counters.count(Metric::PipelinesFromStageFills);
            core.counters.rise(Gauge::Pipelines);
            Ok(pipelines[0])
        }
        Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED)) => {
            let breach = StageBreach::CompileRequired;
            keep_entry(&dir, &key, &Entry::Breach(breach, 0), STAGE_BOUNDS);
            Err(gap(weight, StageGap::Breach(breach)))
        }
        Err((_, result)) => Err(Error::Vulkan {
            call: "vkCreateComputePipelines",
            result,
        }),
    }
}

impl Device {
    /// No Metal counterpart: the cache data of `request`'s pipeline from the consumer's helper,
    /// kept as its entry in the cache directory, so the make refused with
    /// [`Error::StageUnfilled`] makes the pipeline when asked again. Writes the request to
    /// `to_helper`, closes it, and reads the helper's reply ([`serve_stage`]) from `from_helper`,
    /// on the calling thread, until the reply is whole or the helper's end closes the pipe: a
    /// consumer that kills the helper (its cap, its time limit, a cancel) ends the call at once.
    /// Ok: the helper's device creation and compile times ([`StageSpans`]).
    ///
    /// Err: [`StageFill`] for each reply that keeps no cache data; a failed compile is kept as
    /// [`StageBreach::Compile`].
    pub fn fill_stage(
        &self,
        request: &StageRequest,
        mut to_helper: impl Write,
        mut from_helper: impl Read,
    ) -> Result<StageSpans, StageFill> {
        let request = &request.0;
        let sent = write_frame(&mut to_helper, &request_payload(request));
        drop(to_helper);
        sent.map_err(|_| StageFill::Ended)?;
        let payload = match read_frame(&mut from_helper) {
            Ok(payload) => payload,
            Err(Framing::Version) => return Err(StageFill::Version),
            Err(Framing::Cut) => return Err(StageFill::Ended),
        };
        let mut reply = Reader(&payload);
        let (Some(key), Some(status)) = (reply.array::<32>(), reply.u32()) else {
            return Err(StageFill::Ended);
        };
        if key != request.key {
            return Err(StageFill::Ended);
        }
        let core = self.core();
        let dir = core.caches.dir().map(|dir| dir.join(STAGES));
        let keep = |entry: &Entry| {
            (dir.as_deref()).is_some_and(|dir| keep_entry(dir, &request.key, entry, STAGE_BOUNDS))
        };
        match status {
            DONE => {
                let (Some(device_nanos), Some(compile_nanos), Some(data)) =
                    (reply.u64(), reply.u64(), reply.bytes())
                else {
                    return Err(StageFill::Ended);
                };
                if !data.starts_with(&request.header) {
                    return Err(StageFill::NoDevice);
                }
                if !keep(&Entry::Data(data.to_vec())) {
                    return Err(StageFill::Unkept);
                }
                core.counters.count(Metric::StageFills);
                Ok(StageSpans {
                    device_nanos,
                    compile_nanos,
                })
            }
            COMPILE => {
                let raw = reply.u32().ok_or(StageFill::Ended)?;
                keep(&Entry::Breach(StageBreach::Compile, 0));
                Err(StageFill::Compile(vk::Result::from_raw(raw as i32)))
            }
            NO_DEVICE => Err(StageFill::NoDevice),
            ENABLES => Err(StageFill::Enables),
            _ => Err(StageFill::Ended),
        }
    }

    /// No Metal counterpart: keeps `breach` as the entry of `request`'s pipeline, so its makes are
    /// refused ([`Gap::StageSize`]) with no helper asked again, until the driver's version changes:
    /// what the consumer ended its helper for ([`StageBreach::Memory`], [`StageBreach::Time`]),
    /// or the helper's own end before its reply ([`StageBreach::Ended`]), the first such end being
    /// the answer. A memory breach is kept with the device's
    /// [`DeviceOptions::stage_memory_ceiling`], and a device given a higher ceiling asks the
    /// helper again; an ended one is kept with the version of this runtime's helper code, and a
    /// runtime whose helper code changed asks again. A directory that takes no entry keeps
    /// nothing.
    pub fn record_stage_breach(&self, request: &StageRequest, breach: StageBreach) {
        let core = self.core();
        if let Some(dir) = core.caches.dir() {
            let kept = match breach {
                StageBreach::Memory => core.stage_memory_ceiling,
                StageBreach::Ended => u64::from(HELPER_VERSION),
                _ => 0,
            };
            let entry = Entry::Breach(breach, kept);
            keep_entry(&dir.join(STAGES), &request.0.key, &entry, STAGE_BOUNDS);
        }
    }
}

/// No Metal counterpart: the consumer's helper process's main, which compiles one kernel past the
/// stage bound for [`Device::fill_stage`]. Reads one request from `from_parent`; opens the device
/// of `instance` whose `deviceUUID` and `driverUUID` are the request's, with `options` and no cache
/// directory, which must be the options the parent's device was opened with (the parent's
/// [`StageFill::Enables`] otherwise); compiles the pipeline through an empty `VkPipelineCache`; and
/// writes that cache's data to `to_parent`. The consumer makes `instance` as it makes its main
/// process's (the same loader and drivers), and caps, times and kills the process itself.
///
/// Err: the pipes' errors, or a request that does not read whole; every outcome of the compile is
/// the parent's reply.
pub fn serve_stage(
    instance: &Instance,
    mut options: DeviceOptions,
    mut from_parent: impl Read,
    mut to_parent: impl Write,
) -> io::Result<()> {
    let payload = match read_frame(&mut from_parent) {
        Ok(payload) => payload,
        // The reply's own version tells the parent.
        Err(Framing::Version) => return write_frame(&mut to_parent, &[]),
        Err(Framing::Cut) => return Err(ErrorKind::InvalidData.into()),
    };
    let request = read_request(&payload).ok_or(ErrorKind::InvalidData)?;
    options.pipeline_cache_dir = None;
    let mut reply = Vec::new();
    reply.extend_from_slice(&request.key);
    match serve(instance, options, &request) {
        Ok((data, spans)) => {
            reply.extend(DONE.to_le_bytes());
            reply.extend(spans.device_nanos.to_le_bytes());
            reply.extend(spans.compile_nanos.to_le_bytes());
            put_bytes(&mut reply, &data);
        }
        Err(Served::Compile(result)) => {
            reply.extend(COMPILE.to_le_bytes());
            reply.extend((result.as_raw() as u32).to_le_bytes());
        }
        Err(Served::NoDevice) => reply.extend(NO_DEVICE.to_le_bytes()),
        Err(Served::Enables) => reply.extend(ENABLES.to_le_bytes()),
    }
    write_frame(&mut to_parent, &reply)
}

/// What kept the helper's compile from giving cache data.
enum Served {
    NoDevice,
    Enables,
    Compile(vk::Result),
}

/// The helper's compile of `request` on a device of `instance` opened with `options`: the data
/// of an empty `VkPipelineCache` the pipeline was created through, and the times of the device's
/// creation and of the compile.
fn serve(
    instance: &Instance,
    options: DeviceOptions,
    request: &Request,
) -> Result<(Vec<u8>, StageSpans), Served> {
    let raw = instance.raw().1;
    let physical = (instance.physical_devices().map_err(|_| Served::NoDevice)?)
        .into_iter()
        .find(|physical| uuids(raw, physical.raw()) == request.uuids)
        .ok_or(Served::NoDevice)?;
    let began = host_time();
    let device = Device::new(&physical, options).map_err(|_| Served::NoDevice)?;
    let device_nanos = host_time().saturating_sub(began);
    let core = device.core();
    if core.enables() != request.enables {
        return Err(Served::Enables);
    }
    // SAFETY: the words are the parent's translator's SPIR-V for a device of these enables.
    let (data, compile_nanos) = unsafe { compile(core, request) }.map_err(Served::Compile)?;
    let spans = StageSpans {
        device_nanos,
        compile_nanos,
    };
    Ok((data, spans))
}

/// Compiles `request`'s pipeline on `core`'s device through an empty `VkPipelineCache` and gives
/// the cache's data and the create's nanoseconds; everything it made is destroyed.
///
/// # Safety
///
/// `request`'s words are a valid SPIR-V module for the features `core`'s device enabled.
unsafe fn compile(core: &Core, request: &Request) -> Result<(Vec<u8>, u64), vk::Result> {
    let device = &core.device;
    let info = vk::ShaderModuleCreateInfo::default().code(&request.words);
    // SAFETY: the caller's contract.
    let module = unsafe { device.create_shader_module(&info, None) }?;
    // SAFETY: no data.
    let cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) };
    let cache = match cache {
        Ok(cache) => cache,
        Err(result) => {
            // SAFETY: the module is the device's, and nothing uses it.
            unsafe { device.destroy_shader_module(module, None) };
            return Err(result);
        }
    };
    let mut required = (request.subgroup).map(|size| {
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(size)
    });
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::from_raw(request.flags))
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&request.entry);
    if let Some(required) = &mut required {
        stage = stage.push_next(required);
    }
    let mut info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(core.layout);
    // As the device makes the pipeline (`crate::pipeline::kernel_robustness`).
    let mut robustness = crate::pipeline::kernel_robustness(core);
    if let Some(robustness) = &mut robustness {
        info = info.push_next(robustness);
    }
    let began = host_time();
    // SAFETY: the module, the layout and the cache are the device's.
    let made = unsafe { device.create_compute_pipelines(cache, &[info], None) };
    let nanos = host_time().saturating_sub(began);
    let data = match made {
        Ok(pipelines) => {
            // SAFETY: the pipeline is the device's, and nothing uses it.
            unsafe { device.destroy_pipeline(pipelines[0], None) };
            // SAFETY: the cache is the device's.
            unsafe { device.get_pipeline_cache_data(cache) }.map(|data| (data, nanos))
        }
        Err((_, result)) => Err(result),
    };
    // SAFETY: both are the device's, and nothing uses them.
    unsafe {
        device.destroy_pipeline_cache(cache, None);
        device.destroy_shader_module(module, None);
    }
    data
}

/// What an entry holds.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    Data(Vec<u8>),
    /// A breach, with what it was kept under: for a [`StageBreach::Memory`], the memory ceiling
    /// ([`DeviceOptions::stage_memory_ceiling`]), 8 bytes after the breach's byte, read as 0 where
    /// none follow it; for a [`StageBreach::Ended`], the [`HELPER_VERSION`], 4 bytes after it; 0
    /// and nothing after the byte for the others.
    Breach(StageBreach, u64),
}

/// The entry of key `key` in `dir`, where it checks for a device of `header` given the memory
/// ceiling `ceiling`, touched as used; a file that does not check, a memory breach kept under a
/// ceiling below `ceiling`, or an ended one kept under a helper version below [`HELPER_VERSION`],
/// is a miss and removed (the module docs).
pub(crate) fn read_entry(
    dir: &Path,
    key: &[u8; 32],
    header: &[u8; 32],
    ceiling: u64,
) -> Option<Entry> {
    let path = dir.join(hex(key));
    let file = read_whole(&path, None).ok().flatten()?;
    let entry = unframe(&file).and_then(|body| {
        let (held, rest) = body.split_first_chunk::<32>()?;
        let (&kind, payload) = rest.split_first()?;
        match (kind, payload) {
            _ if held != key => None,
            (DATA, data) if data.starts_with(header) => Some(Entry::Data(data.to_vec())),
            (BREACH, [byte, kept @ ..]) => {
                let breach =
                    (StageBreach::ALL.into_iter()).find(|breach| breach.byte() == *byte)?;
                let (kept, holds) = match (breach, kept) {
                    (StageBreach::Memory, []) => (0, ceiling == 0),
                    (StageBreach::Memory, kept) => {
                        let kept = u64::from_le_bytes(kept.try_into().ok()?);
                        (kept, kept >= ceiling)
                    }
                    (StageBreach::Ended, kept) => {
                        let kept = u32::from_le_bytes(kept.try_into().ok()?);
                        (u64::from(kept), kept >= HELPER_VERSION)
                    }
                    (_, []) => (0, true),
                    _ => return None,
                };
                holds.then_some(Entry::Breach(breach, kept))
            }
            _ => None,
        }
    });
    match &entry {
        Some(_) => {
            let touched = File::options().write(true).open(&path);
            let _ = touched.and_then(|file| file.set_modified(SystemTime::now()));
        }
        None => {
            let _ = fs::remove_file(&path);
        }
    }
    entry
}

/// Keeps `entry` as key `key`'s in `dir`, made where it is missing, then removes the entries past
/// `bounds`, oldest use first; whether it was written.
pub(crate) fn keep_entry(dir: &Path, key: &[u8; 32], entry: &Entry, bounds: StageBounds) -> bool {
    let mut body = key.to_vec();
    match entry {
        Entry::Data(data) => {
            body.push(DATA);
            body.extend_from_slice(data);
        }
        Entry::Breach(breach, kept) => {
            body.extend([BREACH, breach.byte()]);
            match breach {
                StageBreach::Memory => body.extend(kept.to_le_bytes()),
                StageBreach::Ended => body.extend(u32::try_from(*kept).unwrap_or(0).to_le_bytes()),
                _ => {}
            }
        }
    }
    let _ = fs::create_dir_all(dir);
    let written = write_whole(&dir.join(hex(key)), &[&frame(&body), &body]).is_ok();
    prune(dir, bounds);
    written
}

/// Removes the entries of `dir` past `bounds`: keeping the most recently used while they fit.
fn prune(dir: &Path, bounds: StageBounds) {
    let Ok(listed) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<(SystemTime, u64, PathBuf)> = (listed.flatten())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            (name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(())?;
            let meta = entry.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), entry.path()))
        })
        .collect();
    entries.sort_unstable_by_key(|(used, _, _)| std::cmp::Reverse(*used));
    let mut bytes = 0u64;
    for (at, (_, len, path)) in entries.into_iter().enumerate() {
        bytes = bytes.saturating_add(len);
        if at >= bounds.entries || bytes > bounds.bytes {
            let _ = fs::remove_file(path);
        }
    }
}

/// Why a frame was not read.
pub(crate) enum Framing {
    /// It is of another wire version.
    Version,
    /// It ended early, or does not check.
    Cut,
}

/// Writes `payload` as one frame (the module docs).
pub(crate) fn write_frame(to: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let mut head = Vec::with_capacity(28);
    head.extend_from_slice(WIRE);
    head.extend(WIRE_VERSION.to_le_bytes());
    head.extend((payload.len() as u64).to_le_bytes());
    head.extend(checksum(payload).to_le_bytes());
    to.write_all(&head)?;
    to.write_all(payload)?;
    to.flush()
}

/// The payload of one frame read from `from`.
pub(crate) fn read_frame(from: &mut impl Read) -> Result<Vec<u8>, Framing> {
    let mut head = [0u8; 28];
    from.read_exact(&mut head).map_err(|_| Framing::Cut)?;
    let word = |at: usize| u64::from_le_bytes(head[at..at + 8].try_into().expect("8 bytes"));
    if &head[..8] != WIRE {
        return Err(Framing::Cut);
    }
    if u32::from_le_bytes(head[8..12].try_into().expect("4 bytes")) != WIRE_VERSION {
        return Err(Framing::Version);
    }
    let len = word(12);
    if len > MAX_PAYLOAD {
        return Err(Framing::Cut);
    }
    let mut payload = vec![0; len as usize];
    from.read_exact(&mut payload).map_err(|_| Framing::Cut)?;
    (checksum(&payload) == word(20))
        .then_some(payload)
        .ok_or(Framing::Cut)
}

/// `bytes`, after their length.
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend((bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// A request's payload.
fn request_payload(request: &Request) -> Vec<u8> {
    // The fixed fields' 160 bytes, then the variable ones: the payload is made once, at its size.
    let len = 160 + request.entry.as_bytes().len() + request.words.len() * 4;
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(&request.key);
    out.extend(request.weight.to_le_bytes());
    out.extend_from_slice(&request.header);
    out.extend_from_slice(&request.uuids[0]);
    out.extend_from_slice(&request.uuids[1]);
    out.extend_from_slice(&request.enables);
    put_bytes(&mut out, request.entry.to_bytes());
    out.extend(request.flags.to_le_bytes());
    out.extend(request.subgroup.unwrap_or(0).to_le_bytes());
    out.extend((request.words.len() as u64 * 4).to_le_bytes());
    for word in request.words.iter() {
        out.extend(word.to_le_bytes());
    }
    out
}

/// The request of a payload [`request_payload`] wrote.
fn read_request(payload: &[u8]) -> Option<Request> {
    let mut r = Reader(payload);
    let key = r.array::<32>()?;
    let weight = r.u64()?;
    let header = r.array::<32>()?;
    let uuids = [r.array::<16>()?, r.array::<16>()?];
    let enables = r.array::<32>()?;
    let entry = CString::new(r.bytes()?).ok()?;
    let flags = r.u32()?;
    let subgroup = Some(r.u32()?).filter(|&size| size != 0);
    let words = r.bytes()?;
    let (words, []) = words.as_chunks::<4>() else {
        return None;
    };
    let words: Arc<[u32]> = words.iter().map(|word| u32::from_le_bytes(*word)).collect();
    r.0.is_empty().then_some(Request {
        key,
        weight,
        header,
        uuids,
        enables,
        words,
        entry,
        flags,
        subgroup,
    })
}

/// Reads a payload from its front.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }

    fn u32(&mut self) -> Option<u32> {
        self.array::<4>().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.array::<8>().map(u64::from_le_bytes)
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let len = usize::try_from(self.u64()?).ok()?;
        let (bytes, rest) = self.0.split_at_checked(len)?;
        self.0 = rest;
        Some(bytes)
    }
}
