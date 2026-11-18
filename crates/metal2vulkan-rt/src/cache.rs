//! The device's caches on disk, under [`DeviceOptions::pipeline_cache_dir`]: a kernel lowered and
//! a `VkPipeline` compiled in one run cost little in the next. Compute and render pipelines.
//!
//! - `translations/`: one file per lowered function (a kernel, or a render pipeline's vertex or
//!   fragment function), named by the hex of its `FunctionCompileKey`, which covers everything
//!   that changes the lowering's output: the function's AIR, entry, constants and linked
//!   functions, its lowering inputs, the device profile's fingerprint and the translator's
//!   semantics version. The bytes are the translator's
//!   self-checking encoding (`CompiledFunction::to_bytes`), and bytes that do not decode for the
//!   key are a miss. A partial-threadgroups variant the translator refused is kept too, as a
//!   marker.
//! - `pipelines/<pipelineCacheUUID>/`: one file per `VkPipeline`, named by the hex of a SHA-256
//!   over everything its creation takes ([`pipeline_key`]): the `VkPipelineCache` data of a cache
//!   that made only that pipeline. A `VkPipeline` is made through a `VkPipelineCache` of its own,
//!   started from its file where there is one; the reaper then reads the cache's data and writes
//!   it as the file, unless the data is no longer than the file's (the pipeline came from it),
//!   and destroys the cache. At most [`PENDING`] caches wait for the reaper: past it a pipeline's
//!   cache is destroyed at once and its file not written, a miss at its next creation.
//!
//! Both kinds share one cap, the consumer's ([`DeviceOptions::pipeline_cache_cap`]): past it the
//! least recently used files go first. A file's use is its time on disk, moved forward at a use
//! when it is older than [`TOUCH`], so the order holds across runs and processes. Every write is
//! a temporary file renamed over its name, and every file checks itself (the translator's
//! encoding, the pipeline data's checksum), so a file cut short by a crash is a miss. A failed
//! read or write is a miss too: the caches never fail a call. Devices open on one directory,
//! in one process or several, share its files, each keeping the cap over the files it knows:
//! those there when it opened, and those it wrote or read since; a file another removed is a
//! miss.
//!
//! [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
//! [`DeviceOptions::pipeline_cache_cap`]: crate::DeviceOptions::pipeline_cache_cap

use crate::counters::Counters;
use crate::device::lock;
use ash::vk;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

/// The cap of a cache directory where the consumer names none.
pub(crate) const DEFAULT_CAP: u64 = 512 << 20;
/// The most pipeline caches waiting for the reaper to write their data.
pub(crate) const PENDING: usize = 64;
/// How old a file's time is before a use moves it forward.
const TOUCH: Duration = Duration::from_secs(3600);
/// How old a temporary file left by a run that ended mid-write is before it is removed.
const STALE: Duration = Duration::from_secs(3600);
/// The first bytes of a pipeline data file, before its length and its checksum.
const MAGIC: &[u8; 8] = b"m2vpipe1";
/// What a translation file holds where the translator refused the lowering: bytes no lowering
/// encodes to, and that a file cut short is not.
const REFUSED: &[u8] = b"m2v-refused";
/// The translations' directory, under the cache directory.
const TRANSLATIONS: &str = "translations";
/// The pipelines' directory, under the cache directory: one directory per `pipelineCacheUUID`.
const PIPELINES: &str = "pipelines";

/// The files of a cache directory, `translations/<name>` and `pipelines/<uuid>/<name>`, at most
/// `cap` bytes in all, the least recently used going first. Its reads and writes run outside its
/// lock, which guards only what it knows of the files.
pub(crate) struct Store {
    root: PathBuf,
    cap: u64,
    known: Mutex<Known>,
}

struct Known {
    /// Each file by its path under the root: its length and when it was last used.
    files: BTreeMap<String, (u64, SystemTime)>,
    total: u64,
}

impl Store {
    /// The store in `root`, made where it is missing, with the files a run before left, each used
    /// at its time on disk, evicted down to `cap`.
    pub(crate) fn open(root: PathBuf, cap: u64) -> Store {
        let _ = fs::create_dir_all(root.join(TRANSLATIONS));
        let _ = fs::create_dir_all(root.join(PIPELINES));
        let mut known = Known {
            files: BTreeMap::new(),
            total: 0,
        };
        let now = SystemTime::now();
        known.scan(&root, TRANSLATIONS, now);
        let uuids = fs::read_dir(root.join(PIPELINES)).into_iter().flatten();
        for entry in uuids.flatten() {
            let (Ok(kind), Ok(uuid)) = (entry.file_type(), entry.file_name().into_string()) else {
                continue;
            };
            if kind.is_dir() && is_uuid(&uuid) {
                known.scan(&root, &format!("{PIPELINES}/{uuid}"), now);
            }
        }
        known.evict(&root, cap);
        Store {
            root,
            cap,
            known: Mutex::new(known),
        }
    }

    /// The bytes of `file`, a path under the root, where the directory holds it: one this store
    /// wrote or found when it opened, or one another device on the directory wrote since. A use
    /// moves an old file's time on disk forward.
    pub(crate) fn get(&self, file: &str) -> Option<Vec<u8>> {
        let path = self.root.join(file);
        let Ok(bytes) = fs::read(&path) else {
            lock(&self.known).forget(file);
            return None;
        };
        let now = SystemTime::now();
        let old = (fs::metadata(&path).and_then(|meta| meta.modified()))
            .is_ok_and(|written| now.duration_since(written).is_ok_and(|age| age > TOUCH));
        if old {
            let _ = fs::File::options()
                .write(true)
                .open(&path)
                .and_then(|opened| opened.set_modified(now));
        }
        lock(&self.known).used(&self.root, self.cap, file, bytes.len() as u64);
        Some(bytes)
    }

    /// Keeps `bytes` as `file`, evicting the least recently used files past the cap. False where
    /// they were not written: past the cap alone, or a failed write.
    pub(crate) fn put(&self, file: &str, bytes: &[u8]) -> bool {
        let len = bytes.len() as u64;
        if len > self.cap || write_whole(&self.root.join(file), bytes).is_err() {
            return false;
        }
        lock(&self.known).used(&self.root, self.cap, file, len);
        true
    }

    /// Whether the directory holds `bytes` as `file`, reading it only where its length is
    /// theirs.
    pub(crate) fn holds(&self, file: &str, bytes: &[u8]) -> bool {
        fs::metadata(self.root.join(file)).is_ok_and(|meta| meta.len() == bytes.len() as u64)
            && self.get(file).is_some_and(|held| held == bytes)
    }

    /// The bytes the store holds.
    #[cfg(test)]
    pub(crate) fn total(&self) -> u64 {
        lock(&self.known).total
    }
}

impl Known {
    /// Notes the files of `dir`, a directory under `root`, and removes its stale temporary files.
    fn scan(&mut self, root: &Path, dir: &str, now: SystemTime) {
        for entry in fs::read_dir(root.join(dir)).into_iter().flatten().flatten() {
            let (Ok(meta), Ok(name)) = (entry.metadata(), entry.file_name().into_string()) else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let written = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            if is_name(&name) {
                self.note(format!("{dir}/{name}"), meta.len(), written);
            } else if now.duration_since(written).is_ok_and(|age| age > STALE) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// Notes `file` used now, `len` bytes long, evicting the least recently used files past
    /// `cap`.
    fn used(&mut self, root: &Path, cap: u64, file: &str, len: u64) {
        self.note(file.to_owned(), len, SystemTime::now());
        self.evict(root, cap);
    }

    fn note(&mut self, file: String, len: u64, used: SystemTime) {
        self.forget(&file);
        self.total += len;
        self.files.insert(file, (len, used));
    }

    fn evict(&mut self, root: &Path, cap: u64) {
        while self.total > cap {
            let Some(oldest) = self
                .files
                .iter()
                .min_by_key(|(_, &(_, used))| used)
                .map(|(file, _)| file.clone())
            else {
                break;
            };
            let _ = fs::remove_file(root.join(&oldest));
            self.forget(&oldest);
        }
    }

    fn forget(&mut self, file: &str) {
        if let Some((len, _)) = self.files.remove(file) {
            self.total -= len;
        }
    }
}

/// A file name the store keeps: 64 lowercase hex digits, a SHA-256's.
fn is_name(name: &str) -> bool {
    name.len() == 64 && is_hex(name)
}

/// A pipelines directory's name: 32 lowercase hex digits, a `pipelineCacheUUID`'s.
fn is_uuid(name: &str) -> bool {
    name.len() == 2 * vk::UUID_SIZE && is_hex(name)
}

fn is_hex(name: &str) -> bool {
    name.bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Writes `bytes` to `path` through a temporary file beside it, named for this process and this
/// write, then renamed over it; its directory is made where it is missing.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Relaxed)
    ));
    let temp = PathBuf::from(temp);
    let created = fs::File::create(&temp).or_else(|error| match (error.kind(), path.parent()) {
        (std::io::ErrorKind::NotFound, Some(dir)) => {
            fs::create_dir_all(dir).and_then(|()| fs::File::create(&temp))
        }
        _ => Err(error),
    });
    let written = created.and_then(|mut file| file.write_all(bytes));
    match written.and_then(|()| fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(error)
        }
    }
}

/// FNV-1a over `bytes`: the pipeline data file's check that it was written whole.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The pipeline data a file holds: its bytes after the magic, their length and their checksum,
/// where all three check.
pub(crate) fn unframe(file: &[u8]) -> Option<&[u8]> {
    let (magic, rest) = file.split_first_chunk::<8>()?;
    let (len, rest) = rest.split_first_chunk::<8>()?;
    let (sum, data) = rest.split_first_chunk::<8>()?;
    (magic == MAGIC
        && u64::from_le_bytes(*len) == data.len() as u64
        && u64::from_le_bytes(*sum) == checksum(data))
    .then_some(data)
}

/// `data` framed for its file.
pub(crate) fn frame(data: &[u8]) -> Vec<u8> {
    let mut file = Vec::with_capacity(24 + data.len());
    file.extend_from_slice(MAGIC);
    file.extend_from_slice(&(data.len() as u64).to_le_bytes());
    file.extend_from_slice(&checksum(data).to_le_bytes());
    file.extend_from_slice(data);
    file
}

/// The key of a `VkPipeline`'s file: a SHA-256 over what its creation takes, `module` (a compute
/// module's digest, [`module_digest`], or a render content's, over its modules' digests and its
/// fixed-function state) and `values` (its specialization's, after a render pipeline's topology
/// class), so two creations with one key make one pipeline.
pub(crate) fn pipeline_key(module: &[u8; 32], values: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(module);
    hash.update((values.len() as u64).to_le_bytes());
    hash.update(values);
    hash.finalize().into()
}

/// The digest of a module's part of its pipelines' creation: its SPIR-V, its entry point, its
/// stage's flags and required subgroup size, the specialization's constant IDs, and the pipeline
/// layout's, which is the runtime's own (`layout`, the words that describe it).
pub(crate) fn module_digest(
    spirv: &[u32],
    entry: &[u8],
    flags: vk::PipelineShaderStageCreateFlags,
    subgroup: Option<u32>,
    constant_ids: &[u32],
    layout: &[u32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    let words = |hash: &mut Sha256, words: &[u32]| {
        hash.update((words.len() as u64).to_le_bytes());
        for word in words {
            hash.update(word.to_le_bytes());
        }
    };
    words(&mut hash, spirv);
    hash.update((entry.len() as u64).to_le_bytes());
    hash.update(entry);
    hash.update(flags.as_raw().to_le_bytes());
    hash.update(subgroup.map_or([0; 5], |size| {
        let mut bytes = [1; 5];
        bytes[1..].copy_from_slice(&size.to_le_bytes());
        bytes
    }));
    words(&mut hash, constant_ids);
    words(&mut hash, layout);
    hash.finalize().into()
}

/// A device's caches.
pub(crate) struct Caches {
    store: Option<Store>,
    /// The device's pipelines directory under the root: `pipelines/<pipelineCacheUUID>`.
    pipelines: String,
    /// The pipeline caches given to the reaper and not written yet.
    pending: AtomicUsize,
}

/// A `VkPipeline`'s cache, for the reaper to write the data of and destroy.
pub(crate) struct KeepPipeline {
    pub(crate) cache: vk::PipelineCache,
    /// The pipeline's key.
    pub(crate) key: [u8; 32],
    /// The length of the data the cache started from.
    pub(crate) seeded: usize,
}

impl Caches {
    /// The caches of a device whose pipeline data is keyed by `uuid`, in `dir` where there is
    /// one, at most `cap` bytes in all.
    pub(crate) fn new(dir: Option<&Path>, cap: u64, uuid: [u8; vk::UUID_SIZE]) -> Caches {
        let pipelines = format!("{PIPELINES}/{}", hex(&uuid));
        let store = dir.map(|dir| {
            let _ = fs::create_dir_all(dir.join(&pipelines));
            Store::open(dir.to_owned(), cap)
        });
        Caches {
            store,
            pipelines,
            pending: AtomicUsize::new(0),
        }
    }

    /// Whether the device has a cache directory.
    pub(crate) fn on_disk(&self) -> bool {
        self.store.is_some()
    }

    fn translation_file(key: &[u8]) -> String {
        format!("{TRANSLATIONS}/{}", hex(key))
    }

    /// The bytes kept for a lowered kernel of key digest `key`.
    pub(crate) fn translation(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.store.as_ref()?.get(&Caches::translation_file(key))
    }

    /// Keeps the bytes of a lowered kernel of key digest `key`.
    pub(crate) fn keep_translation(&self, key: &[u8], bytes: &[u8], counters: &Counters) {
        if let Some(store) = &self.store {
            if store.put(&Caches::translation_file(key), bytes) {
                counters.cache_written(bytes.len() as u64);
            }
        }
    }

    /// Whether the translator refused the lowering of key digest `key` on a device before.
    pub(crate) fn refused(&self, key: &[u8]) -> bool {
        self.store
            .as_ref()
            .is_some_and(|store| store.holds(&Caches::translation_file(key), REFUSED))
    }

    /// Keeps the translator's refusal of the lowering of key digest `key`.
    pub(crate) fn keep_refusal(&self, key: &[u8], counters: &Counters) {
        self.keep_translation(key, REFUSED, counters);
    }

    fn pipeline_file(&self, key: &[u8; 32]) -> String {
        format!("{}/{}", self.pipelines, hex(key))
    }

    /// The data kept for the `VkPipeline` of `key`, where its file checks.
    pub(crate) fn pipeline_data(&self, key: &[u8; 32]) -> Option<Vec<u8>> {
        let file = self.store.as_ref()?.get(&self.pipeline_file(key))?;
        unframe(&file).map(<[u8]>::to_vec)
    }

    /// Takes a place among the caches waiting for the reaper: false past [`PENDING`], when the
    /// pipeline's cache is to be destroyed at once.
    pub(crate) fn take_pending(&self) -> bool {
        self.pending
            .try_update(Relaxed, Relaxed, |pending| {
                (pending < PENDING).then_some(pending + 1)
            })
            .is_ok()
    }

    /// On the reaper: writes the data of `kept`'s cache as its pipeline's file where it is longer
    /// than the data the cache started from, then destroys the cache.
    ///
    /// # Safety
    ///
    /// The cache is the device's, and no pipeline is being made from it.
    pub(crate) unsafe fn keep_pipeline(
        &self,
        device: &ash::Device,
        kept: KeepPipeline,
        counters: &Counters,
    ) {
        // SAFETY: the caller's contract.
        let data = unsafe { device.get_pipeline_cache_data(kept.cache) };
        // SAFETY: the caller's contract; the data is read.
        unsafe { device.destroy_pipeline_cache(kept.cache, None) };
        self.pending.fetch_sub(1, Relaxed);
        let (Some(store), Ok(data)) = (&self.store, data) else {
            return;
        };
        if data.len() > kept.seeded {
            let file = frame(&data);
            if store.put(&self.pipeline_file(&kept.key), &file) {
                counters.cache_written(file.len() as u64);
            }
        }
    }
}
