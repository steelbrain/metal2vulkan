//! The device's caches on disk, under [`DeviceOptions::pipeline_cache_dir`]: a kernel lowered and
//! a `VkPipeline` compiled in one run cost little in the next. Compute pipelines only.
//!
//! - `translations/`: one file per lowered kernel, named by the hex of its `FunctionCompileKey`,
//!   which covers everything that changes the lowering's output: the kernel's AIR, entry,
//!   constants and linked functions, its lowering inputs (the variant), the device profile's
//!   fingerprint and the translator's semantics version. The bytes are the translator's
//!   self-checking encoding (`CompiledFunction::to_bytes`), and bytes that do not decode for the
//!   key are a miss. A partial-threadgroups variant the translator refused is kept too, as a
//!   marker. The files take at most [`TRANSLATION_BYTES`]; past it the least recently used
//!   go.
//! - `pipelines-<pipelineCacheUUID>`: the device's `VkPipelineCache` data, which the driver keys by
//!   each pipeline's module and specialization. It is loaded with the device, unless it is past
//!   [`PIPELINE_BYTES`] or does not check, when the cache starts empty; it is written on the reaper
//!   thread once a `VkPipeline` was made and [`SAVE_INTERVAL`] has passed since the last write, so
//!   a run that is killed loses at most that, and at the device's end.
//!
//! Every write is a temporary file renamed over its name, and every file checks itself (the
//! translator's encoding, the pipeline data's checksum), so a file cut short by a crash is a miss.
//! A failed read or write is a miss too: the caches never fail a call. Devices open on one
//! directory share its files, each keeping the cap over the files it knows.
//!
//! [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir

use crate::device::lock;
use ash::vk;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// The most bytes of lowered kernels a cache directory keeps.
pub(crate) const TRANSLATION_BYTES: u64 = 256 << 20;
/// The most bytes of pipeline data loaded and written.
pub(crate) const PIPELINE_BYTES: u64 = 256 << 20;
/// The least time between two writes of the pipeline data.
pub(crate) const SAVE_INTERVAL: Duration = Duration::from_secs(10);
/// How old a temporary file left by a run that ended mid-write is before it is removed.
const STALE: Duration = Duration::from_secs(3600);
/// The first bytes of a pipeline data file, before its length and its checksum.
const MAGIC: &[u8; 8] = b"m2vpipe1";
/// What a translation file holds where the translator refused the lowering: bytes no lowering
/// encodes to, and that a file cut short is not.
const REFUSED: &[u8] = b"m2v-refused";

/// A directory of files of at most `cap` bytes in all, the least recently used going first.
pub(crate) struct Store {
    dir: PathBuf,
    cap: u64,
    /// Each file by name: its length and when it was last used, in `clock`'s ticks.
    files: BTreeMap<String, (u64, u64)>,
    total: u64,
    clock: u64,
}

impl Store {
    /// The store in `dir`, made where it is missing, with the files a run before left, the
    /// least recently written first, evicted down to `cap`.
    pub(crate) fn open(dir: PathBuf, cap: u64) -> Store {
        let _ = fs::create_dir_all(&dir);
        let now = SystemTime::now();
        let mut found = Vec::new();
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let (Ok(meta), Ok(name)) = (entry.metadata(), entry.file_name().into_string()) else {
                continue;
            };
            let written = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            if !meta.is_file() {
                continue;
            }
            if is_name(&name) {
                found.push((written, name, meta.len()));
            } else if now.duration_since(written).is_ok_and(|age| age > STALE) {
                let _ = fs::remove_file(entry.path());
            }
        }
        found.sort();
        let mut store = Store {
            dir,
            cap,
            files: BTreeMap::new(),
            total: 0,
            clock: 0,
        };
        for (_, name, len) in found {
            store.clock += 1;
            store.total += len;
            store.files.insert(name, (len, store.clock));
        }
        store.evict();
        store
    }

    /// The bytes of `name`, where the directory holds it: one this store wrote or found when it
    /// opened, or one another device on the directory wrote since.
    pub(crate) fn get(&mut self, name: &str) -> Option<Vec<u8>> {
        let read = fs::read(self.dir.join(name));
        self.forget(name);
        let bytes = read.ok()?;
        self.clock += 1;
        self.files
            .insert(name.to_owned(), (bytes.len() as u64, self.clock));
        self.total += bytes.len() as u64;
        self.evict();
        Some(bytes)
    }

    /// Keeps `bytes` as `name`, evicting the least recently used files past the cap.
    pub(crate) fn put(&mut self, name: &str, bytes: &[u8]) {
        let len = bytes.len() as u64;
        if len > self.cap || write_whole(&self.dir.join(name), bytes).is_err() {
            return;
        }
        self.forget(name);
        self.clock += 1;
        self.files.insert(name.to_owned(), (len, self.clock));
        self.total += len;
        self.evict();
    }

    /// Whether the directory holds `bytes` as `name`, reading the file only where its length is
    /// theirs.
    pub(crate) fn holds(&mut self, name: &str, bytes: &[u8]) -> bool {
        fs::metadata(self.dir.join(name)).is_ok_and(|meta| meta.len() == bytes.len() as u64)
            && self.get(name).is_some_and(|held| held == bytes)
    }

    /// The bytes the store holds.
    #[cfg(test)]
    pub(crate) fn total(&self) -> u64 {
        self.total
    }

    fn evict(&mut self) {
        while self.total > self.cap {
            let Some(oldest) = self
                .files
                .iter()
                .min_by_key(|(_, &(_, used))| used)
                .map(|(name, _)| name.clone())
            else {
                break;
            };
            let _ = fs::remove_file(self.dir.join(&oldest));
            self.forget(&oldest);
        }
    }

    fn forget(&mut self, name: &str) {
        if let Some((len, _)) = self.files.remove(name) {
            self.total -= len;
        }
    }
}

/// A file name the store keeps: 64 lowercase hex digits, a SHA-256's.
fn is_name(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Writes `bytes` to `path` through a temporary file beside it, named for this process and this
/// write, then renamed over it.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Relaxed)
    ));
    let temp = PathBuf::from(temp);
    let written = fs::File::create(&temp).and_then(|mut file| file.write_all(bytes));
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

/// A device's caches.
pub(crate) struct Caches {
    translations: Option<Mutex<Store>>,
    /// The pipeline data's file, where there is a directory.
    pipelines: Option<PathBuf>,
    /// Every `VkPipeline` of the device is made through it.
    pub(crate) pipeline_cache: vk::PipelineCache,
    save: Mutex<Save>,
}

/// When the pipeline data was last written, and whether a `VkPipeline` was made since.
struct Save {
    last: Instant,
    made: bool,
}

impl Caches {
    /// The caches of a device whose pipeline data is keyed by `uuid`, in `dir` where there is
    /// one: its `VkPipelineCache` made from the data a run before wrote.
    ///
    /// # Safety
    ///
    /// `device` is live.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        dir: Option<&Path>,
        uuid: [u8; vk::UUID_SIZE],
    ) -> Result<Caches, vk::Result> {
        let translations =
            dir.map(|dir| Mutex::new(Store::open(dir.join("translations"), TRANSLATION_BYTES)));
        let pipelines = dir.map(|dir| dir.join(format!("pipelines-{}", hex(&uuid))));
        let loaded = pipelines
            .as_ref()
            .filter(|path| fs::metadata(path).is_ok_and(|meta| meta.len() <= PIPELINE_BYTES + 24))
            .and_then(|path| fs::read(path).ok());
        let data = loaded.as_deref().and_then(unframe).unwrap_or_default();
        let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        // SAFETY: the caller's contract; `data` is empty or what `vkGetPipelineCacheData` gave,
        // whole, as its checksum shows.
        let pipeline_cache = unsafe { device.create_pipeline_cache(&info, None) }?;
        Ok(Caches {
            translations,
            pipelines,
            pipeline_cache,
            save: Mutex::new(Save {
                last: Instant::now(),
                made: false,
            }),
        })
    }

    /// The bytes kept for a lowered kernel of key digest `key`.
    pub(crate) fn translation(&self, key: &[u8]) -> Option<Vec<u8>> {
        lock(self.translations.as_ref()?).get(&hex(key))
    }

    /// Keeps the bytes of a lowered kernel of key digest `key`.
    pub(crate) fn keep_translation(&self, key: &[u8], bytes: &[u8]) {
        if let Some(store) = &self.translations {
            lock(store).put(&hex(key), bytes);
        }
    }

    /// Whether the translator refused the lowering of key digest `key` on a device before.
    pub(crate) fn refused(&self, key: &[u8]) -> bool {
        self.translations
            .as_ref()
            .is_some_and(|store| lock(store).holds(&hex(key), REFUSED))
    }

    /// Keeps the translator's refusal of the lowering of key digest `key`.
    pub(crate) fn keep_refusal(&self, key: &[u8]) {
        self.keep_translation(key, REFUSED);
    }

    /// Notes a `VkPipeline` made: true where it is the first since the pipeline data's last
    /// write, when the reaper is to be woken to time the next.
    pub(crate) fn made(&self) -> bool {
        if self.pipelines.is_none() {
            return false;
        }
        let mut save = lock(&self.save);
        let first = !save.made;
        save.made = true;
        first
    }

    /// How long until the pipeline data is due, where a `VkPipeline` was made since its last
    /// write.
    pub(crate) fn due_in(&self) -> Option<Duration> {
        let save = lock(&self.save);
        save.made
            .then(|| SAVE_INTERVAL.saturating_sub(save.last.elapsed()))
    }

    /// Writes the pipeline data, where a `VkPipeline` was made since its last write.
    pub(crate) fn save(&self, device: &ash::Device) {
        let Some(path) = &self.pipelines else {
            return;
        };
        {
            let mut save = lock(&self.save);
            if !save.made {
                return;
            }
            save.made = false;
            save.last = Instant::now();
        }
        // SAFETY: the cache is the device's, and stays until the device's end, which writes it
        // last.
        let Ok(data) = (unsafe { device.get_pipeline_cache_data(self.pipeline_cache) }) else {
            return;
        };
        if data.len() as u64 <= PIPELINE_BYTES {
            let _ = write_whole(path, &frame(&data));
        }
    }

    /// Writes the pipeline data a last time and destroys the cache.
    ///
    /// # Safety
    ///
    /// No pipeline is being made from the cache, and none will be.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        self.save(device);
        // SAFETY: the caller's contract.
        unsafe { device.destroy_pipeline_cache(self.pipeline_cache, None) };
    }
}
