//! Kernels past the stage bound: compiled in the consumer's helper process, created here from the
//! helper's cache data alone, never compiled in this process.
//!
//! A kernel whose module weighs more than `metal2vulkan::pipeline::COMPILED_STAGE_WEIGHT`
//! (`ModuleStats::entry_weight`) is one whose compile may take the host's memory, so the process
//! that runs the program never compiles it. Each of its `VkPipeline`s is made from a
//! `VkPipelineCache` made of that pipeline's own entry in the cache directory alone, with
//! `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` ([`made`]), and the cache destroyed
//! once the create returns: the device's own pipeline cache and its blob never hold it. Where the
//! directory holds no entry for the pipeline, the make (or the dispatch, or
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
//! header, so a new driver version asks again.
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
const WIRE_VERSION: u32 = 1;
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

/// No Metal counterpart: what ended a helper's compile of a kernel past the stage bound, kept as
/// the pipeline's entry ([`Device::record_stage_breach`], [`Device::fill_stage`]) so its makes are
/// refused ([`Gap::StageSize`]) with no helper asked again, until the driver's version changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageBreach {
    /// The helper passed the memory cap the consumer gave it.
    Memory,
    /// The helper passed the time the consumer gave it.
    Time,
    /// The driver failed the helper's compile (`vkCreateComputePipelines` answered an error).
    Compile,
    /// The driver answered compile-required for the helper's cache data
    /// (`VK_PIPELINE_COMPILE_REQUIRED`): it would compile again in this process.
    CompileRequired,
}

impl StageBreach {
    const ALL: [StageBreach; 4] = [
        StageBreach::Memory,
        StageBreach::Time,
        StageBreach::Compile,
        StageBreach::CompileRequired,
    ];

    fn byte(self) -> u8 {
        match self {
            StageBreach::Memory => 0,
            StageBreach::Time => 1,
            StageBreach::Compile => 2,
            StageBreach::CompileRequired => 3,
        }
    }
}

/// No Metal counterpart: why [`Device::fill_stage`] kept no cache data. Only
/// [`StageFill::Compile`] is kept, as a breach.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageFill {
    /// The helper's reply did not come whole: it ended, was killed (a cap, a time limit, a
    /// cancel) or closed its pipes first. Nothing is kept; the consumer keeps a cap's or a time
    /// limit's end with [`Device::record_stage_breach`].
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

/// A kernel's module past the stage bound, kept with its kernel for the requests of its
/// `VkPipeline`s.
pub(crate) struct Heavy {
    /// The module's entry weight.
    weight: u64,
    words: Arc<[u32]>,
    /// The digest of the words, which every stage key holds.
    digest: [u8; 32],
}

/// The entry weight past which a stage is compiled only in the consumer's helper: the translator's
/// `COMPILED_STAGE_WEIGHT`, or, in a device test, one the test sets ([`TEST_BOUND`]), so a small
/// kernel passes it.
pub(crate) fn bound() -> u64 {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let Some(bound) =
        std::num::NonZeroU64::new(TEST_BOUND.load(std::sync::atomic::Ordering::Relaxed))
    {
        return bound.get();
    }
    metal2vulkan::pipeline::COMPILED_STAGE_WEIGHT
}

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
            digest: Sha256::digest(words_bytes(&stage.spirv)).into(),
        })
    }
}

/// What a compute `VkPipeline`'s create gives its one stage besides the module.
pub(crate) struct Create<'a> {
    pub(crate) entry: &'a CStr,
    pub(crate) flags: vk::PipelineShaderStageCreateFlags,
    pub(crate) subgroup: Option<u32>,
    /// The specialization constants' ids, each 4 bytes of `data` in order.
    pub(crate) ids: &'a [u32],
    pub(crate) data: &'a [u8],
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
    words: Arc<[u32]>,
    entry: CString,
    flags: u32,
    subgroup: Option<u32>,
    ids: Vec<u32>,
    data: Vec<u8>,
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

/// The bytes of `words`, least significant byte first.
fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
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

/// A compute `VkPipeline` of `heavy` that `make` creates through the cache it is given, with
/// `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` in its create info: through a cache
/// made of the pipeline's entry alone, destroyed once `make` returns (the module docs). Counted as
/// made, and its time (`Metrics::pipeline_creation_nanos`).
///
/// Err: [`Error::StageUnfilled`] where the directory holds no entry for it; [`Gap::StageSize`]
/// where it holds a breach, where the driver answers compile-required for its data (kept as a
/// breach), on a device with no cache directory or whose driver writes no pipeline cache header;
/// [`Error::Vulkan`] where the create fails otherwise.
pub(crate) fn made(
    core: &Core,
    heavy: &Heavy,
    create: &Create<'_>,
    make: impl FnOnce(vk::PipelineCache) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) -> Result<vk::Pipeline, Error> {
    let weight = heavy.weight;
    let dir = (core.caches.dir()).ok_or_else(|| gap(weight, StageGap::NoCacheDirectory))?;
    let dir = dir.join(STAGES);
    let header = (core.caches.header()).ok_or_else(|| gap(weight, StageGap::NoPipelineCache))?;
    let enables = core.enables();
    let key = key(&header, &enables, heavy, create);
    let data = match read_entry(&dir, &key, &header) {
        None => {
            let (instance, physical) = (core.instance.raw().1, core.physical);
            return Err(Error::StageUnfilled(StageRequest(Arc::new(Request {
                key,
                weight,
                header,
                uuids: uuids(instance, physical),
                enables,
                words: Arc::clone(&heavy.words),
                entry: create.entry.to_owned(),
                flags: create.flags.as_raw(),
                subgroup: create.subgroup,
                ids: create.ids.to_vec(),
                data: create.data.to_vec(),
            }))));
        }
        Some(Entry::Breach(breach)) => return Err(gap(weight, StageGap::Breach(breach))),
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
    let began = host_time();
    let created = make(cache);
    core.counters
        .pipeline_creation(host_time().saturating_sub(began));
    // SAFETY: the cache is the device's, and the create that used it has returned.
    unsafe { core.device.destroy_pipeline_cache(cache, None) };
    match created {
        Ok(pipelines) => {
            core.counters.count(Metric::PipelinesCreated);
            core.counters.count(Metric::PipelinesFromStageFills);
            core.counters.rise(Gauge::Pipelines);
            Ok(pipelines[0])
        }
        Err((_, vk::Result::PIPELINE_COMPILE_REQUIRED)) => {
            let breach = StageBreach::CompileRequired;
            keep_entry(&dir, &key, &Entry::Breach(breach), STAGE_BOUNDS);
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
    ///
    /// Err: [`StageFill`] for each reply that keeps no cache data; a failed compile is kept as
    /// [`StageBreach::Compile`].
    pub fn fill_stage(
        &self,
        request: &StageRequest,
        mut to_helper: impl Write,
        mut from_helper: impl Read,
    ) -> Result<(), StageFill> {
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
                let data = reply.bytes().ok_or(StageFill::Ended)?;
                if !data.starts_with(&request.header) {
                    return Err(StageFill::NoDevice);
                }
                if !keep(&Entry::Data(data.to_vec())) {
                    return Err(StageFill::Unkept);
                }
                core.counters.count(Metric::StageFills);
                Ok(())
            }
            COMPILE => {
                let raw = reply.u32().ok_or(StageFill::Ended)?;
                keep(&Entry::Breach(StageBreach::Compile));
                Err(StageFill::Compile(vk::Result::from_raw(raw as i32)))
            }
            NO_DEVICE => Err(StageFill::NoDevice),
            ENABLES => Err(StageFill::Enables),
            _ => Err(StageFill::Ended),
        }
    }

    /// No Metal counterpart: keeps `breach` as the entry of `request`'s pipeline, so its makes are
    /// refused ([`Gap::StageSize`]) with no helper asked again, until the driver's version changes:
    /// what the consumer ended its helper for ([`StageBreach::Memory`], [`StageBreach::Time`]).
    /// A directory that takes no entry keeps nothing.
    pub fn record_stage_breach(&self, request: &StageRequest, breach: StageBreach) {
        if let Some(dir) = self.core().caches.dir() {
            let entry = Entry::Breach(breach);
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
        Ok(data) => {
            reply.extend(DONE.to_le_bytes());
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
/// of an empty `VkPipelineCache` the pipeline was created through.
fn serve(
    instance: &Instance,
    options: DeviceOptions,
    request: &Request,
) -> Result<Vec<u8>, Served> {
    let raw = instance.raw().1;
    let physical = (instance.physical_devices().map_err(|_| Served::NoDevice)?)
        .into_iter()
        .find(|physical| uuids(raw, physical.raw()) == request.uuids)
        .ok_or(Served::NoDevice)?;
    let device = Device::new(&physical, options).map_err(|_| Served::NoDevice)?;
    let core = device.core();
    if core.enables() != request.enables {
        return Err(Served::Enables);
    }
    // SAFETY: the words are the parent's translator's SPIR-V for a device of these enables.
    unsafe { compile(core, request) }.map_err(Served::Compile)
}

/// Compiles `request`'s pipeline on `core`'s device through an empty `VkPipelineCache` and gives
/// the cache's data; everything it made is destroyed.
///
/// # Safety
///
/// `request`'s words are a valid SPIR-V module for the features `core`'s device enabled.
unsafe fn compile(core: &Core, request: &Request) -> Result<Vec<u8>, vk::Result> {
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
    let entries: Vec<vk::SpecializationMapEntry> = (request.ids.iter().enumerate())
        .map(|(index, &id)| vk::SpecializationMapEntry {
            constant_id: id,
            offset: (index * 4) as u32,
            size: 4,
        })
        .collect();
    let specialization = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&request.data);
    let mut required = (request.subgroup).map(|size| {
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(size)
    });
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::from_raw(request.flags))
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&request.entry)
        .specialization_info(&specialization);
    if let Some(required) = &mut required {
        stage = stage.push_next(required);
    }
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(core.layout);
    // SAFETY: the module, the layout and the cache are the device's, and the specialization gives
    // each constant a 4-byte value, as the parent's create does.
    let made = unsafe { device.create_compute_pipelines(cache, &[info], None) };
    let data = match made {
        Ok(pipelines) => {
            // SAFETY: the pipeline is the device's, and nothing uses it.
            unsafe { device.destroy_pipeline(pipelines[0], None) };
            // SAFETY: the cache is the device's.
            unsafe { device.get_pipeline_cache_data(cache) }
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
    Breach(StageBreach),
}

/// The entry of key `key` in `dir`, where it checks for a device of `header`, touched as used; a
/// file that does not check is removed (the module docs).
pub(crate) fn read_entry(dir: &Path, key: &[u8; 32], header: &[u8; 32]) -> Option<Entry> {
    let path = dir.join(hex(key));
    let file = read_whole(&path, None).ok().flatten()?;
    let entry = unframe(&file).and_then(|body| {
        let (held, rest) = body.split_first_chunk::<32>()?;
        let (&kind, payload) = rest.split_first()?;
        match (kind, payload) {
            _ if held != key => None,
            (DATA, data) if data.starts_with(header) => Some(Entry::Data(data.to_vec())),
            (BREACH, &[byte]) => (StageBreach::ALL.into_iter())
                .find(|breach| breach.byte() == byte)
                .map(Entry::Breach),
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
        Entry::Breach(breach) => body.extend([BREACH, breach.byte()]),
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
    let mut out = Vec::new();
    out.extend_from_slice(&request.key);
    out.extend(request.weight.to_le_bytes());
    out.extend_from_slice(&request.header);
    out.extend_from_slice(&request.uuids[0]);
    out.extend_from_slice(&request.uuids[1]);
    out.extend_from_slice(&request.enables);
    put_bytes(&mut out, request.entry.to_bytes());
    out.extend(request.flags.to_le_bytes());
    out.extend(request.subgroup.unwrap_or(0).to_le_bytes());
    out.extend((request.ids.len() as u64).to_le_bytes());
    for id in &request.ids {
        out.extend(id.to_le_bytes());
    }
    put_bytes(&mut out, &request.data);
    put_bytes(&mut out, &words_bytes(&request.words));
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
    let count = usize::try_from(r.u64()?).ok()?;
    let ids = (0..count).map(|_| r.u32()).collect::<Option<Vec<u32>>>()?;
    let data = r.bytes()?.to_vec();
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
        ids,
        data,
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
