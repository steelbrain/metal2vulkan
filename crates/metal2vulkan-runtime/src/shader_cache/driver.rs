//! The Vulkan driver's pipeline cache data: one record per family of makings that convert the
//! same stages, and one live `VkPipelineCache` per record for the device (it replaced the one
//! entry per device, the caches per variant, and the cache per making).
//!
//! - **One record, one live cache, every making of its name.** A record's `VkPipelineCache` is
//!   made once, by the first `vkCreate*Pipelines` call of the runtime
//!   (`ShaderCache::make_pipelines`) that names it: from the record where one loads, else empty.
//!   The device keeps it ([`LiveCaches`]) and hands it to every later making of that name, until
//!   its teardown destroys it. So a family of makings that differ only in state no stage's
//!   conversion reads converts and compiles its stages once, and no making past the first reads a
//!   file or pays the driver's load of a record. Where a making added to the data (no record, or
//!   its pipeline not in it), the data is written as the record; where it is what the cache was
//!   loaded with or last wrote, nothing is written. A boot reads the records of the families it
//!   makes, and nothing else. A device that holds [`LIVE`] caches already hands a further name's
//!   making a cache of its own, destroyed after the call.
//! - **Why not a cache per making.** MoltenVK answers a stage's lookup out of the shader module's
//!   own library cache when it is handed no pipeline cache, and out of the pipeline cache alone
//!   when it is handed one. A cache made and destroyed around one making therefore carried
//!   nothing to the next and converted every stage of every making: 64 conversions over 32 blend
//!   states of one vertex-and-fragment pair, where no cache at all and one cache shared by the 32
//!   each convert 2, 1.8x to 10.5x of the call by the module's size. Measured on an M3 Max,
//!   macOS 27.0.1, MoltenVK 1.4.2 (`driverVersion` 0x28a2), through the driver's own activity
//!   tracking (`metal2vulkan-conformance/probes/pipeline_cache_across_blend_states.c`); one
//!   machine and one driver, and no other is measured.
//! - **Found, not read back.** A making handed a cache that was loaded from its record, whose
//!   driver reports the pipeline found in that cache (`VK_EXT_pipeline_creation_feedback`'s
//!   `APPLICATION_PIPELINE_CACHE_HIT`, for the whole pipeline or every stage: `found_in_cache`)
//!   reads no data back and writes nothing: the record held what the making needed. MoltenVK
//!   1.4.1's data after such a making is the record's entries in the other order (its shader
//!   library map iterates its modules' keys opposite to their insertion), so the byte comparison
//!   alone wrote every record of two modules again at each warm making. The data's format is
//!   MoltenVK's own, undocumented beyond Vulkan's header, so no comparison here reads its
//!   entries. A making whose driver reports nothing, or reports a compile, or whose cache was
//!   made empty, reads the data and compares its SHA-256 with what the cache was loaded with or
//!   last wrote.
//! - **Named by what a stage's conversion reads** ([`Making`]). A record is named by the SHA-256
//!   of [`VERSION`], the making's description, the driver's identity (vendor, device, driver
//!   version, `pipelineCacheUUID`) and the digest of the layer settings the instance was created
//!   with; and by the device's profile fingerprint (`store::entry_name`). A program's pipeline is
//!   described by its variant's stages (SPIR-V words, entry points, stage flags, required subgroup
//!   sizes, specialization), its layout (its immutable samplers among it), and the state its
//!   making adds. A render making adds the names of the two pipeline parts that hold shaders, its
//!   pre-rasterization and fragment shader libraries (`GraphicsContent::record` in
//!   `pipeline/graphics.rs`), its vertex input, and whether it rasterizes points; it adds no
//!   blend state, no write mask, no fragment output interface and no topology besides the point
//!   class, none of which reaches the driver's conversion of a stage, so a state's sweep over
//!   them is one record and one cache. A compute making adds its specialization and flags, which
//!   do reach it. A helper of the runtime's is named by its SPIR-V and its parameters.
//!   Records live in the content directory, which no build owns (`store.rs`): a new build of the
//!   runtime or the translator that leaves a pipeline's SPIR-V and state as they were finds its
//!   record. The driver keys each pipeline by everything its creation reads, so a description
//!   that covers a making loosely costs a compile, never a wrong pipeline.
//! - **No figure of the driver's to meet.** A record holds one family's data (an Xgc pipeline's
//!   1.7 MB on an RTX 3090), so how much of a large cache's initial data a
//!   driver keeps does not decide which pipelines are kept. The shader cache's cap and its scans,
//!   the least recently used first (`store.rs`), do: a hit refreshes a record's access time.
//! - **One family's libraries, loaded once.** A driver that compiles every shader of a cache's
//!   initial data when the cache is made (`RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad`,
//!   MoltenVK) compiles from a record the family's own stages alone, and once for the device, not
//!   once a making: the record's cache is made one time. A record shared by a family does not grow
//!   with the family either, since MoltenVK's data holds converted libraries and no pipeline
//!   state -- 32 blend states of one pair are the 1522 bytes of one, whose load is 2 of Metal's
//!   compiles and 50 us against the 64 compiles and 2639 us of the 32 records a cache per making
//!   wrote. Host steps and pool pieces take and save a record's cache as the program's threads
//!   do.
//! - **Makings at once.** A record is written to a temporary file renamed over it, by the
//!   device's cache thread (`Store::write_record_soon`, `worker.rs`), so a reader reads the whole
//!   old record or the whole new one, never a mix; a making that reads the record of one still
//!   queued reads the queued bytes. Two processes, or two threads, making the same pipeline at once each write that
//!   pipeline's data from the same driver under the same name: the same bytes, or either one is
//!   fine.
//! - **Cold on purpose.** A program that wants every pipeline compiled (a cold boot measured as
//!   one) names an empty directory in `ShaderCacheOptions::directory`: nothing else of the
//!   runtime's holds the driver's data. A driver's own shader disk cache is the driver's to turn
//!   off (NVIDIA's `__GL_SHADER_DISK_CACHE=0`, Mesa's `MESA_SHADER_CACHE_DISABLE=true`).
//!
//! ```text
//! magic        8 bytes   "M2VPPLC\0"
//! identity     28 bytes  vendorID, deviceID, driverVersion (u32 each), pipelineCacheUUID
//! data length  u64
//! data         vkGetPipelineCacheData's bytes
//! checksum     32 bytes  SHA-256 of every byte before it
//! ```
//!
//! Integers are little-endian. A load takes the data only if the identity equals the device's,
//! the checksum holds, and the data's own header (`VkPipelineCacheHeaderVersionOne`) names the
//! same vendor, device and UUID; anything else is refused and the record removed before the
//! driver sees it. A record the driver refuses is counted, and the making starts empty. A save
//! checks the driver's data the same way and keeps nothing it would refuse to load, counting the
//! reason ([`Refusal`]): some drivers return data that ends inside the header it states, at
//! `VK_SUCCESS`.

use super::store::{self, count, entry_name, Lookup, Store};
use super::EntryKind;
use crate::device::lock;
use crate::exit_guard::{self, Call};
use crate::release;
use ash::vk;
use metal2vulkan::cache::Hash256;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

const MAGIC: [u8; 8] = *b"M2VPPLC\0";
/// The version of what a record's name covers and how a record is framed: a change to either
/// bumps it, so no build opens a record another build names or frames otherwise. A record of an
/// older version is named otherwise, so it is never found and never read: a bump costs a family
/// its records, never a misread.
pub(crate) const VERSION: u32 = 2;

/// The most `VkPipelineCache`es a device keeps live, one per record name. Past it a making of a
/// further name is handed a cache of its own, destroyed once the making returned, so a program
/// that makes pipelines of unboundedly many families keeps a bounded number of the driver's
/// caches.
pub(crate) const LIVE: usize = 256;
/// The identity's bytes.
const IDENTITY: usize = 4 + 4 + 4 + 16;
/// Bytes before the data.
const HEADER: usize = 8 + IDENTITY + 8;
/// Bytes after the data.
const CHECKSUM: usize = 32;
/// `sizeof(VkPipelineCacheHeaderVersionOne)`.
const VULKAN_HEADER: usize = 32;

/// Why the driver's pipeline cache data is not kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The data ends before the end of the header it states, or of a version-one header.
    Truncated,
    /// The header is smaller than a version-one header, or its version, vendor, device or UUID
    /// is not this device's.
    Mismatched,
}

/// A Vulkan device's identity for its pipeline cache data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Identity([u8; IDENTITY]);

impl Identity {
    /// The identity of a device of `properties`.
    pub(super) fn of(properties: &vk::PhysicalDeviceProperties) -> Identity {
        Identity::new(
            properties.vendor_id,
            properties.device_id,
            properties.driver_version,
            properties.pipeline_cache_uuid,
        )
    }

    /// The identity of these values.
    pub(crate) fn new(vendor: u32, device: u32, driver: u32, uuid: [u8; 16]) -> Identity {
        let mut bytes = [0; IDENTITY];
        bytes[..4].copy_from_slice(&vendor.to_le_bytes());
        bytes[4..8].copy_from_slice(&device.to_le_bytes());
        bytes[8..12].copy_from_slice(&driver.to_le_bytes());
        bytes[12..].copy_from_slice(&uuid);
        Identity(bytes)
    }

    /// Whether `data` begins with the `VkPipelineCacheHeaderVersionOne` of this identity, whole:
    /// its size, its version, then vendor, device and UUID (not the driver version, which it does
    /// not hold).
    pub(crate) fn check_vulkan_header(&self, data: &[u8]) -> Result<(), Refusal> {
        let word = |at: usize| {
            data.get(at..at + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
        };
        let size = word(0)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or(Refusal::Truncated)?;
        if size < VULKAN_HEADER {
            return Err(Refusal::Mismatched);
        }
        if data.len() < size {
            return Err(Refusal::Truncated);
        }
        let ours = word(4) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
            && data[8..12] == self.0[..4]
            && data[12..16] == self.0[4..8]
            && data[16..32] == self.0[12..];
        if ours {
            Ok(())
        } else {
            Err(Refusal::Mismatched)
        }
    }
}

/// `data` framed for `identity`.
pub(crate) fn seal(identity: &Identity, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + data.len() + CHECKSUM);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&identity.0);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(data);
    let checksum: Hash256 = Sha256::digest(&out).into();
    out.extend_from_slice(&checksum);
    out
}

/// Where the data of a frame for `identity` lies in `bytes`, or `None` when the frame or the
/// data's own header does not hold.
pub(crate) fn open(identity: &Identity, bytes: &[u8]) -> Option<Range<usize>> {
    if bytes.get(..8) != Some(&MAGIC[..]) || bytes.get(8..8 + IDENTITY) != Some(&identity.0[..]) {
        return None;
    }
    let length = bytes
        .get(HEADER - 8..HEADER)?
        .try_into()
        .ok()
        .map(u64::from_le_bytes)?;
    let end = usize::try_from(length).ok()?.checked_add(HEADER)?;
    if bytes.len() != end.checked_add(CHECKSUM)? {
        return None;
    }
    let checksum: Hash256 = Sha256::digest(&bytes[..end]).into();
    if bytes[end..] != checksum {
        return None;
    }
    identity
        .check_vulkan_header(&bytes[HEADER..end])
        .is_ok()
        .then_some(HEADER..end)
}

/// What a making reads, which names its record (the module docs): a tag naming the code that
/// makes it, then the parts its caller adds, in order.
pub(crate) struct Making(Sha256);

impl Making {
    /// A making by the code `kind` names.
    pub(crate) fn new(kind: &str) -> Making {
        let mut hash = Sha256::new();
        hash.update(b"metal2vulkan pipeline making\0");
        hash.update(VERSION.to_le_bytes());
        Making(hash).part(kind.as_bytes())
    }

    /// And `bytes`, after their length.
    fn part(mut self, bytes: &[u8]) -> Making {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
        self
    }

    /// And the SPIR-V `words`.
    pub(crate) fn words(mut self, words: &[u32]) -> Making {
        self.0.update(((words.len() as u64) * 4).to_le_bytes());
        let mut chunk = [0u8; 4096];
        for words in words.chunks(chunk.len() / 4) {
            for (at, word) in words.iter().enumerate() {
                chunk[at * 4..at * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            self.0.update(&chunk[..words.len() * 4]);
        }
        self
    }

    /// And `value`'s `Debug` text: equal values give equal text, and a build that prints them
    /// otherwise names other records, which costs a compile.
    pub(crate) fn text(self, value: &dyn std::fmt::Debug) -> Making {
        self.part(format!("{value:?}").as_bytes())
    }

    /// And `digest`, of parts described as these are.
    pub(crate) fn digest(self, digest: &Hash256) -> Making {
        self.part(digest)
    }

    /// The digest of the parts so far, to describe the makings that share them at once.
    pub(crate) fn finish(self) -> Hash256 {
        self.0.finalize().into()
    }
}

/// What names a device's records: its driver's identity, its instance's layer settings' digest,
/// and its profile fingerprint.
pub(crate) struct Records {
    pub(crate) identity: Identity,
    pub(crate) settings: Hash256,
    pub(crate) profile: Hash256,
}

impl Records {
    /// What `making` reads, with this device's: the key of its record, which is also what the
    /// device holds its live cache under ([`LiveCaches`]).
    pub(crate) fn key(&self, making: Making) -> Hash256 {
        making
            .digest(&Sha256::digest(self.identity.0).into())
            .digest(&self.settings)
            .finish()
    }

    /// The name of the record of `key`.
    pub(crate) fn name(&self, key: &Hash256) -> String {
        entry_name(
            &store::content(),
            EntryKind::PipelineData,
            key,
            &self.profile,
        )
    }
}

/// What a live cache started from.
struct Seed {
    /// The record's bytes.
    bytes: Vec<u8>,
    /// Where its data lies in them.
    data: Range<usize>,
}

/// One record's `VkPipelineCache`, made once and handed to every making of that record's name
/// (the module docs). It keeps no device functions and has no destroying `Drop`: its device
/// destroys it, at its teardown ([`LiveCaches::destroy`]) or, for a cache past [`LIVE`] that no
/// map holds, after the one making that was handed it.
pub(crate) struct Live {
    cache: vk::PipelineCache,
    /// The record's name in the store.
    name: String,
    /// Whether the cache was made from the record's data: only then does a making whose driver
    /// reports the pipeline found in it read no data back (the module docs).
    loaded: bool,
    /// The SHA-256 and the length of the data the cache was made from, or of what a making last
    /// wrote as the record; `None` until one of the two happened.
    kept: Mutex<Option<(Hash256, u64)>>,
}

impl Live {
    /// Destroy the cache.
    ///
    /// # Safety
    ///
    /// The device is live, and no making holds the cache any more: its device's teardown, after
    /// every pipeline was made, or the return of the one making handed a cache of its own.
    unsafe fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the caller's.
        unsafe { release::object(raw, self.cache) };
    }
}

/// The `VkPipelineCache`es a device keeps, one per record name, at most [`LIVE`] of them (the
/// module docs).
#[derive(Default)]
pub(crate) struct LiveCaches {
    caches: Mutex<HashMap<Hash256, Arc<Live>>>,
}

impl LiveCaches {
    /// The device's cache of the record of `key`, and whether the device holds it (so a later
    /// making of the name is handed this one and its caller destroys nothing). A cache is made
    /// here, from the record where one loads, when the device holds none of that name; `None`
    /// where the driver made none, when the making is handed a null cache.
    ///
    /// The record is read and the cache made with no lock of the map held, so one making's load
    /// does not hold up another's: two makings of one name at once may each make a cache, and the
    /// one that is second to the map destroys its own and takes the first's.
    fn of(
        &self,
        store: &Store,
        records: &Records,
        raw: &ash::Device,
        key: &Hash256,
    ) -> Option<(Arc<Live>, bool)> {
        if let Some(live) = lock(&self.caches).get(key) {
            count(&store.counters.driver_caches_shared);
            return Some((Arc::clone(live), true));
        }
        let name = records.name(key);
        let made = Arc::new(make_live(store, &records.identity, raw, name)?);
        let mut caches = lock(&self.caches);
        if let Some(live) = caches.get(key) {
            let live = Arc::clone(live);
            drop(caches);
            // SAFETY: nothing was ever handed this cache, and the device is live.
            unsafe { made.destroy(raw) };
            count(&store.counters.driver_caches_shared);
            return Some((live, true));
        }
        if caches.len() >= LIVE {
            return Some((made, false));
        }
        caches.insert(*key, Arc::clone(&made));
        Some((made, true))
    }

    /// Destroy every cache: the device's teardown, with the device live and every pipeline of
    /// every making made (`DeviceShared::drop`).
    ///
    /// # Safety
    ///
    /// As [`Live::destroy`]'s.
    pub(crate) unsafe fn destroy(&self, raw: &ash::Device) {
        for (_, live) in lock(&self.caches).drain() {
            // SAFETY: the caller's.
            unsafe { live.destroy(raw) };
        }
    }
}

/// The `VkPipelineCache` of the record `name` on the device of `raw`, made from that record's
/// data where it loads and the driver takes it, else empty; `None` where the driver made none.
fn make_live(store: &Store, identity: &Identity, raw: &ash::Device, name: String) -> Option<Live> {
    let seed = match store.load_content(&name, |bytes| {
        let data = open(identity, &bytes)?;
        Some(Seed { bytes, data })
    }) {
        Lookup::Hit(seed) => Some(seed),
        Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
    };
    let seeded = seed.as_ref().and_then(|seed| {
        let data = &seed.bytes[seed.data.clone()];
        let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        // SAFETY: the device is live and `info` lives for the call.
        let made = unsafe { raw.create_pipeline_cache(&info, None) };
        exit_guard::called(Call::CreatePipelineCache);
        match made {
            Ok(cache) => {
                count(&store.counters.driver_loads);
                Some((cache, (Sha256::digest(data).into(), data.len() as u64)))
            }
            Err(_) => {
                // The driver refused data that passed the checks; a making's save replaces it.
                count(&store.counters.rejected);
                None
            }
        }
    });
    match seeded {
        Some((cache, kept)) => Some(Live {
            cache,
            name,
            loaded: true,
            kept: Mutex::new(Some(kept)),
        }),
        None => {
            let info = vk::PipelineCacheCreateInfo::default();
            // SAFETY: as above.
            let made = unsafe { raw.create_pipeline_cache(&info, None) };
            exit_guard::called(Call::CreatePipelineCache);
            made.ok().map(|cache| Live {
                cache,
                name,
                loaded: false,
                kept: Mutex::new(None),
            })
        }
    }
}

/// Make with `create`, handed the device's `VkPipelineCache` of the record `making` names on the
/// device of `raw` (the module docs), made from that record where one loads, and write the data
/// the making added as the record; `create` is handed a null cache where the driver makes none.
/// `raw` is the device's checked accessor's (`DeviceShared::vk`).
///
/// `create` returns, with what it made, what the driver reported of the making: whether it found
/// the pipeline in the cache it was handed ([`found_in_cache`]'s answer: found, compiled, or
/// `None`, unreported). A making handed a cache that was loaded from its record, which the driver
/// reports found, reads none of its cache's data back and writes nothing (counted,
/// `driver_data_found`: the module docs); every other one reads its data and keeps it where it
/// is not what the cache was loaded with or last wrote.
///
/// [`found_in_cache`]: super::found_in_cache
///
/// Once the process is exiting (`exiting`: the exit guard's [`closed`](exit_guard::closed), or a
/// test's own), the making's own `vkCreate*Pipelines` is the one Vulkan call it makes: a pool
/// piece's section, begun before the close, runs on past it (`exit_guard`), and what the making
/// would add to it (a cache's creation, its data's read for the record) is left out. Asked before
/// the cache is taken, where `create` is then handed a null cache, and again once `create`
/// returned, where the cache is left to the device's teardown or the process's end, unread.
pub(crate) fn make<T, E>(
    store: &Store,
    records: &Records,
    live: &LiveCaches,
    raw: &ash::Device,
    making: Making,
    exiting: impl Fn() -> bool,
    create: impl FnOnce(vk::PipelineCache) -> (Result<T, E>, Option<bool>),
) -> Result<T, E> {
    if exiting() {
        return create(vk::PipelineCache::null()).0;
    }
    let key = records.key(making);
    let Some((held, shared)) = live.of(store, records, raw, &key) else {
        return create(vk::PipelineCache::null()).0;
    };
    // The creation log's split names whether the cache the making was handed holds its record.
    crate::pipeline::creation_log::record_loaded(held.loaded);
    let (made, found) = create(held.cache);
    let exited = exiting();
    if made.is_ok() && !exited {
        match (held.loaded, found) {
            (true, Some(true)) => count(&store.counters.driver_data_found),
            _ => keep(store, &records.identity, raw, &held),
        }
    }
    if !shared && !exited {
        // SAFETY: the making returned, and no map holds this cache, so nothing else can be
        // handed it.
        unsafe { held.destroy(raw) };
    }
    made
}

/// Write `cache`'s data as its record unless it is what the cache was loaded with or last wrote,
/// by its SHA-256 (counted), or the checks refuse it (counted). The read is a section of the exit
/// guard ([`exit_guard`]): on the program's own thread, once the process is exiting, nothing is
/// read or written. On a warm pass's threads, the bytes the data grew by since the cache was
/// loaded or last written count against the pass's budget (`warm::added_pipeline_bytes`).
///
/// `vkGetPipelineCacheData` names no externally synchronized parameter, and the Vulkan
/// specification's pipeline cache chapter states that the use of a pipeline cache object in the
/// creation commands "is internally synchronized, and the same pipeline cache object can be used
/// in multiple threads simultaneously", so this read needs no lock against another thread's
/// making from the same cache. The per-record lock below is held only over the comparison and the
/// write, so two makings of one record never write the same bytes twice.
fn keep(store: &Store, identity: &Identity, raw: &ash::Device, live: &Live) {
    let Some(busy) = exit_guard::enter() else {
        return;
    };
    // SAFETY: the cache is live, and the call is internally synchronized against the makings of
    // other threads (above).
    let read = unsafe { raw.get_pipeline_cache_data(live.cache) };
    exit_guard::called(Call::GetPipelineCacheData);
    drop(busy);
    let Ok(data) = read else {
        store.write_failed();
        return;
    };
    let digest: Hash256 = Sha256::digest(&data).into();
    let length = data.len() as u64;
    let mut kept = lock(&live.kept);
    if kept.is_some_and(|(kept, _)| kept == digest) {
        count(&store.counters.driver_data_unchanged);
        return;
    }
    if let Err(refusal) = identity.check_vulkan_header(&data) {
        store.counters.refused(refusal);
        return;
    }
    let before = kept.map_or(0, |(_, length)| length);
    crate::warm::added_pipeline_bytes(length.saturating_sub(before));
    match store.write_record_soon(&live.name, seal(identity, &data)) {
        Ok(_) => {
            *kept = Some((digest, length));
            count(&store.counters.driver_saves);
        }
        Err(_) => store.write_failed(),
    }
}
