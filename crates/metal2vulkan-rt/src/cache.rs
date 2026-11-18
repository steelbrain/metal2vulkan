//! The device's caches: a kernel lowered and a `VkPipeline` compiled in one run cost little in
//! the next. Compute and render pipelines.
//!
//! **The pipeline cache.** A device makes every `VkPipeline` through one `VkPipelineCache` of its
//! own, internally synchronized, which threads that create at once share. With a cache directory
//! ([`DeviceOptions::pipeline_cache_dir`]) the cache starts from the device's blob there: the
//! cache's data as `vkGetPipelineCacheData` gave it, after a magic, its length and its checksum,
//! named `pipelines-` and the hex of the whole 32-byte pipeline cache header the device's driver
//! writes, opaque bytes: a device or driver version whose header differs in any byte, its
//! `pipelineCacheUUID` included, has a blob of its own. A directory keeps at most
//! [`BLOBS_KEPT`] blobs: a device's seed removes the oldest past that, by last write, its own
//! counted among the kept. The reaper
//! makes an empty cache first when it starts, which a creation that comes before it waits for
//! (microseconds); then it reads the blob and makes a cache from it, merges into that what
//! creations compiled in the empty cache meanwhile, and makes it the cache creations take. So no
//! creation waits for the blob's read or parse. A blob that does not check, whose data does not
//! begin with the device's header, or that the driver refuses, is removed, and the empty cache
//! stays.
//!
//! The reaper writes the cache's data over the blob, a temporary file renamed over it and never
//! appended to, once creation is quiet (no creation running and none for [`QUIET_NANOS`]) after a
//! creation: at most once a quiet period, and only where the size the driver gives grew by more
//! than an eighth since the last write, so a run that makes nothing new writes nothing and a
//! device's writes over its life come to a few times the blob's size. The device's drop writes it
//! once more where the size changed at all. A crash between writes loses only the pipelines made
//! since the last. No guest call waits for a write or a size query.
//!
//! The cache holds at most [`PIPELINE_CACHE_BOUND`]: a cache past it when the reaper looks is
//! replaced by a fresh, empty one that creations take from then on, and destroyed once no creation
//! holds it; the next write replaces the blob with the fresh cache's data, so the blob holds at
//! most the bound too, with no second file. [`Gauges::pipeline_cache_bytes`] is the cache's size
//! at the reaper's last look. Without a directory the cache is the same, and nothing is on disk.
//!
//! **The lowerings.** The store keeps one file per lowered function (a kernel, or a render
//! pipeline's vertex or fragment function) under `translations/`, named by the hex of its
//! `FunctionCompileKey`, which covers everything that changes the lowering's output: the
//! function's AIR, entry, constants and linked functions, its lowering inputs, the device
//! profile's fingerprint and the translator's semantics version. The bytes are the translator's
//! self-checking encoding (`CompiledFunction::to_bytes`), and bytes that do not decode for the key
//! are a miss. A partial-threadgroups variant the translator refused is kept too, as a marker.
//!
//! The files live in two generations, `g<N>/` (young) and `g<N-1>/` (old), which the consumer's cap
//! ([`DeviceOptions::pipeline_cache_cap`]) bounds at half of it each. A lookup tries young, then
//! old; a hit in old is renamed into young, which is its mark of use. Generations start at 1, and
//! the first has no old: a lookup there tries young alone. A write goes into young, made by the
//! reaper: a state hands its lowering over ([`keep_translation`]) and returns, and a lowering that
//! would take the ones waiting past [`PENDING_BYTES`] is dropped and counted, never waited for. The
//! bytes young holds are counted in `state`, 24 bytes beside the generations: young's number, its
//! bytes and a checksum of both, read and written under a lock on the file, which every store on
//! the directory, in this process or another, holds open for its life. The write that takes young
//! past half the cap makes the next generation young: young becomes old, and the old one is dead. A
//! store keeps young's number as it last read or wrote the state, so a lookup reads no state: a
//! move another store made is seen at this store's next write, and until then its lookups try the
//! generations it knew, a miss at worst. A thread of the store's own removes the dead generation
//! and ends, so no thread that submits, waits for or completes work removes files. So a file stays
//! at least half the cap and at most the cap of other writes and moves after its last use, opening
//! reads one file whatever the store holds, and nothing per file is kept in memory. The same thread
//! removes the `pipelines/` directories under young and old in which an earlier version kept one
//! file of pipeline data per `VkPipeline`; their bytes stay counted in young until it moves on.
//!
//! Every write is a temporary file renamed over its name, counted before it is written and given
//! back where it fails, so the count is never below what young holds. Every file checks itself
//! (the translator's encoding, the blob's checksum): a file cut short, by a crash or a power loss,
//! is a miss and is removed, and nothing unchecked reaches the driver. A failed read or write is
//! a miss too: the caches never fail a call. A state file that does not check, or is missing, is
//! rebuilt from the generations' names, with young counted full, so the next write moves on and
//! the removal takes the rest. A removal cut short by the process's end is found by the next
//! open, which looks for the generation below old.
//!
//! [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
//! [`DeviceOptions::pipeline_cache_cap`]: crate::DeviceOptions::pipeline_cache_cap
//! [`Gauges::pipeline_cache_bytes`]: crate::Gauges::pipeline_cache_bytes

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::times::host_time;
use ash::vk;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime};

/// The cap of the lowerings in a cache directory where the consumer names none: half of it, what
/// the store is sure to keep, is about seven times the 9.16 MB a long recorded session's 282
/// lowerings fill.
pub(crate) const DEFAULT_CAP: u64 = 128 << 20;
/// The most bytes of data a device's `VkPipelineCache` holds when the reaper looks before it is
/// replaced by a fresh one, and so the most its blob holds: 32 MiB, about five times the 6.25 MB
/// of pipeline data a long recorded session made.
pub(crate) const PIPELINE_CACHE_BOUND: u64 = 32 << 20;
/// How long pipeline creation stays quiet, no creation running and none ending, before the
/// reaper looks at the cache: 20 ms, in nanoseconds of [`host_time`].
const QUIET_NANOS: u64 = 20_000_000;
/// The first bytes of a blob, before its data's length and checksum.
const MAGIC: &[u8; 8] = b"m2vblob1";
/// The start of a device's blob's name, under the cache directory.
const BLOB: &str = "pipelines";
/// The most blobs a cache directory keeps: four kinds of device or driver versions.
pub(crate) const BLOBS_KEPT: usize = 4;
/// The most bytes of lowerings waiting for the reaper to write them: about the 9.16 MB a long
/// recorded session's 282 lowerings fill. A lowering past it is not kept.
pub(crate) const PENDING_BYTES: u64 = 8 << 20;
/// What a translation file holds where the translator refused the lowering: bytes no lowering
/// encodes to, and that a file cut short is not.
const REFUSED: &[u8] = b"m2v-refused";
/// The translations' directory, under a generation.
const TRANSLATIONS: &str = "translations";
/// The directory, under a generation, an earlier version kept a file of pipeline data per
/// `VkPipeline` in: removed where it is found.
const PIPELINES: &str = "pipelines";
/// The state file, under the cache directory.
const STATE: &str = "state";

/// Young's number and the bytes counted in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct State {
    pub(crate) young: u64,
    pub(crate) bytes: u64,
}

/// The lowerings of a cache directory, `translations/<name>` under two generations of at most
/// half of `cap` bytes each (the module's doc).
pub(crate) struct Store {
    root: PathBuf,
    /// Half the cap: the most bytes young counts.
    half: u64,
    /// The state file, open for the store's life; none where it could not be opened, and every
    /// lookup misses.
    state: Option<File>,
    /// Whether the state file is open for writing: a read-only directory serves hits only.
    writable: bool,
    /// Held around the state file's lock, which threads sharing the file share.
    writer: Mutex<()>,
    /// Set while the store's removal thread runs.
    removing: AtomicBool,
    /// Young's number as this store last read or wrote the state file, 0 before it has: what a
    /// lookup tries, without reading the file. It only grows.
    young: AtomicU64,
    /// The device's counters, whose thread gauge holds the removal thread while it runs.
    counters: Counters,
    /// The removal thread last started, joined by the next start or at the device's teardown.
    removal: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The times the store listed the cache directory.
    #[cfg(test)]
    pub(crate) listings: AtomicUsize,
    /// The times the store read the state file outside its lock.
    #[cfg(test)]
    pub(crate) state_reads: AtomicUsize,
}

impl Store {
    /// The store in `root`, made where it is missing; it starts the removal of a generation a
    /// removal before left, and of an earlier version's pipeline data.
    pub(crate) fn open(root: PathBuf, cap: u64, counters: Counters) -> Arc<Store> {
        let _ = fs::create_dir_all(&root);
        let path = root.join(STATE);
        let opened = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path);
        let (state, writable) = match opened {
            Ok(file) => (Some(file), true),
            Err(_) => (File::open(&path).ok(), false),
        };
        let store = Arc::new(Store {
            root,
            half: cap / 2,
            state,
            writable,
            writer: Mutex::new(()),
            removing: AtomicBool::new(false),
            young: AtomicU64::new(0),
            counters,
            removal: Mutex::new(None),
            #[cfg(test)]
            listings: AtomicUsize::new(0),
            #[cfg(test)]
            state_reads: AtomicUsize::new(0),
        });
        if let Some(state) = store.current() {
            store.young.fetch_max(state.young, Relaxed);
            let young = state.young;
            let dead = young >= 2 && store.generation(young - 2).exists();
            if dead || !store.pipeline_dirs(young).is_empty() {
                store.remove_dead();
            }
        }
        store
    }

    /// Whether the state file is open for writing: a read-only directory serves hits only.
    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    /// The directories of an earlier version's pipeline data under young, `young`, and old.
    fn pipeline_dirs(&self, young: u64) -> Vec<PathBuf> {
        (young.saturating_sub(1)..=young)
            .map(|number| self.generation(number).join(PIPELINES))
            .filter(|dir| dir.exists())
            .collect()
    }

    /// What `decode` makes of the bytes of `file`, a path under a generation, where the
    /// directory holds it and they decode; bytes that do not are removed. A hit in old moves the
    /// file into young.
    pub(crate) fn get<T>(
        self: &Arc<Self>,
        file: &str,
        decode: impl FnOnce(&[u8]) -> Option<T>,
    ) -> Option<T> {
        self.lookup(file, None, decode, true)
    }

    /// Whether the directory holds `bytes` as `file`, reading it only where its length is
    /// theirs; a file that differs stays.
    pub(crate) fn holds(self: &Arc<Self>, file: &str, bytes: &[u8]) -> bool {
        let len = Some(bytes.len() as u64);
        self.lookup(file, len, |held| (held == bytes).then_some(()), false)
            .is_some()
    }

    /// Keeps `bytes` as `file` in young. False where they were not written: more than half the
    /// cap alone, or a failed write.
    pub(crate) fn put(self: &Arc<Self>, file: &str, bytes: &[u8]) -> bool {
        let len = bytes.len() as u64;
        if len > self.half {
            return false;
        }
        let Some(young) = self.reserve(len) else {
            return false;
        };
        if self
            .place(young, file, |path| write_whole(path, &[bytes]))
            .is_ok()
        {
            return true;
        }
        self.give_back(young, len);
        false
    }

    /// The state, read without the lock where it checks, else under it, rebuilt where it is
    /// missing or does not check.
    pub(crate) fn current(&self) -> Option<State> {
        #[cfg(test)]
        self.state_reads.fetch_add(1, Relaxed);
        read_state(self.state.as_ref()?).or_else(|| self.update(|state| Some(*state)))
    }

    /// Young's number as this store last read or wrote it, read from the state file only where
    /// it has not been yet.
    fn young(&self) -> Option<u64> {
        match self.young.load(Relaxed) {
            0 => {
                let young = self.current()?.young;
                self.young.fetch_max(young, Relaxed);
                Some(young)
            }
            young => Some(young),
        }
    }

    /// Generation `number`'s directory.
    pub(crate) fn generation(&self, number: u64) -> PathBuf {
        self.root.join(format!("g{number}"))
    }

    /// Waits for the removal thread to end.
    #[cfg(test)]
    pub(crate) fn settle(&self) {
        let start = std::time::Instant::now();
        while self.removing.load(SeqCst) {
            assert!(start.elapsed().as_secs() < 60, "the removal ends");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Looks `file` up in young, then old: `decode`'s answer for its bytes where they are `len`
    /// long (any length with none). Bytes that do not decode are removed where `torn`.
    fn lookup<T>(
        self: &Arc<Self>,
        file: &str,
        len: Option<u64>,
        decode: impl FnOnce(&[u8]) -> Option<T>,
        torn: bool,
    ) -> Option<T> {
        let young = self.young()?;
        let mut decode = Some(decode);
        for (number, old) in [(young, false), (young.saturating_sub(1), true)] {
            let path = self.generation(number).join(file);
            let bytes = match read_whole(&path, len) {
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                Ok(Some(bytes)) => bytes,
                Ok(None) | Err(_) => return None,
            };
            let Some(value) = decode.take().and_then(|decode| decode(&bytes)) else {
                if torn {
                    let _ = fs::remove_file(&path);
                }
                return None;
            };
            if old {
                self.promote(&path, file, bytes.len() as u64);
            }
            return Some(value);
        }
        None
    }

    /// Moves `from`, `len` bytes, into young as `file`.
    fn promote(self: &Arc<Self>, from: &Path, file: &str, len: u64) {
        let Some(young) = self.reserve(len) else {
            return;
        };
        if self.place(young, file, |to| fs::rename(from, to)).is_err() {
            self.give_back(young, len);
        }
    }

    /// Counts `len` bytes into young, moving on to the next generation first where they take it
    /// past half the cap; young's number.
    fn reserve(self: &Arc<Self>, len: u64) -> Option<u64> {
        let (young, moved) = self.update(|state| {
            if state.bytes.saturating_add(len) <= self.half {
                state.bytes += len;
                return Some((state.young, false));
            }
            let next = state.young + 1;
            make_dir(&self.generation(next)).ok()?;
            *state = State {
                young: next,
                bytes: len,
            };
            Some((next, true))
        })?;
        if moved {
            self.remove_dead();
        }
        Some(young)
    }

    /// Takes `len` bytes a failed write or move counted back out of young, where it is still
    /// `young`.
    fn give_back(&self, young: u64, len: u64) {
        self.update(|state| {
            if state.young == young {
                state.bytes = state.bytes.saturating_sub(len);
            }
            Some(())
        });
    }

    /// Runs `change` on the state under the lock, and writes the state back where it answers.
    fn update<T>(&self, change: impl FnOnce(&mut State) -> Option<T>) -> Option<T> {
        let file = self.state.as_ref().filter(|_| self.writable)?;
        let _writer = lock(&self.writer);
        file.lock().ok()?;
        let changed = (|| {
            let mut state = read_state(file).or_else(|| self.rebuild(file))?;
            let answer = change(&mut state)?;
            write_state(file, state).ok()?;
            self.young.fetch_max(state.young, Relaxed);
            Some(answer)
        })();
        let _ = file.unlock();
        changed
    }

    /// Under the lock, a state for a state file that is missing or does not check: young is the
    /// highest generation there, counted full, or a first generation where there is none.
    fn rebuild(&self, file: &File) -> Option<State> {
        #[cfg(test)]
        self.listings.fetch_add(1, Relaxed);
        let state = match generations(&self.root).into_iter().max() {
            Some(young) => State {
                young: young.max(1),
                bytes: self.half,
            },
            None => State { young: 1, bytes: 0 },
        };
        make_dir(&self.generation(state.young)).ok()?;
        write_state(file, state).ok()?;
        Some(state)
    }

    /// Runs `op` on `file`'s path in generation `young`, and where a directory is missing, once
    /// more after making the file's directories under the generation's. Never the generation's
    /// own: a removal may have taken it, and the write is then a miss.
    fn place(
        &self,
        young: u64,
        file: &str,
        op: impl Fn(&Path) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut dir = self.generation(young);
        let path = dir.join(file);
        match op(&path) {
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            done => return done,
        }
        for part in Path::new(file)
            .parent()
            .into_iter()
            .flat_map(Path::components)
        {
            dir.push(part);
            make_dir(&dir)?;
        }
        op(&path)
    }

    /// Starts the removal thread, where none runs: it removes every generation below old, and an
    /// earlier version's pipeline data.
    fn remove_dead(self: &Arc<Self>) {
        if !self.writable || self.removing.swap(true, SeqCst) {
            return;
        }
        let store = Arc::clone(self);
        match crate::device::start("m2v-store-remove", &self.counters, move || store.remove()) {
            // The thread before it has ended its pass, since `removing` was clear: joined here,
            // where it at most finishes looking for a dead generation.
            Ok(started) => {
                let ended = lock(&self.removal).replace(started);
                if let Some(ended) = ended {
                    let _ = ended.join();
                }
            }
            Err(_) => self.removing.store(false, SeqCst),
        }
    }

    /// At the device's teardown: waits for the removal thread, which removes a finite set of
    /// directories, so no thread of the device outlives it.
    pub(crate) fn join_removal(&self) {
        let removal = lock(&self.removal).take();
        if let Some(removal) = removal {
            let _ = removal.join();
        }
    }

    /// The removal thread: makes the state durable, then removes the dead generations and an
    /// earlier version's pipeline data, until a pass finds none it did not try.
    fn remove(&self) {
        loop {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            std::thread::sleep(Duration::from_millis(REMOVAL_PAUSE.load(Relaxed)));
            if let Some(file) = &self.state {
                let _ = file.sync_all();
            }
            let tried = self.dead();
            for dir in &tried {
                let _ = fs::remove_dir_all(dir);
            }
            self.removing.store(false, SeqCst);
            // A move to the next generation during the pass found the thread running and started
            // none: its dead generation is taken here. One the pass could not remove waits for
            // the next move or open.
            let new = self.dead().iter().any(|dir| !tried.contains(dir));
            if !new || self.removing.swap(true, SeqCst) {
                return;
            }
        }
    }

    /// The generations below old, and an earlier version's pipeline data under young and old.
    fn dead(&self) -> Vec<PathBuf> {
        let Some(young) = self.current().map(|state| state.young) else {
            return Vec::new();
        };
        let mut dead: Vec<PathBuf> = (generations(&self.root).into_iter())
            .filter(|&number| number + 1 < young)
            .map(|number| self.generation(number))
            .collect();
        dead.extend(self.pipeline_dirs(young));
        dead
    }
}

/// A pause at the start of each removal pass, in milliseconds, for a test that tears a device
/// down during one.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) static REMOVAL_PAUSE: AtomicU64 = AtomicU64::new(0);

/// Makes directory `dir` where it is missing; not its parent.
fn make_dir(dir: &Path) -> io::Result<()> {
    match fs::create_dir(dir) {
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(()),
        made => made,
    }
}

/// The generations in `root`, by number.
fn generations(root: &Path) -> Vec<u64> {
    let entries = fs::read_dir(root).into_iter().flatten().flatten();
    entries
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let digits = name.strip_prefix('g')?;
            (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| digits.parse().ok())?
        })
        .collect()
}

/// The state `file` holds, where it checks.
fn read_state(file: &File) -> Option<State> {
    let mut record = [0; 24];
    read_at(file, &mut record).ok()?;
    let word = |at: usize| u64::from_le_bytes(record[at..at + 8].try_into().expect("8 bytes"));
    let state = State {
        young: word(0),
        bytes: word(8),
    };
    (state.young >= 1 && word(16) == checksum(&record[..16])).then_some(state)
}

/// Writes `state` to `file`, with its checksum.
fn write_state(file: &File, state: State) -> io::Result<()> {
    let mut record = [0; 24];
    record[..8].copy_from_slice(&state.young.to_le_bytes());
    record[8..16].copy_from_slice(&state.bytes.to_le_bytes());
    let sum = checksum(&record[..16]);
    record[16..].copy_from_slice(&sum.to_le_bytes());
    write_at(file, &record)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, 0)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, 0)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8]) -> io::Result<()> {
    match std::os::windows::fs::FileExt::seek_read(file, buf, 0)? {
        read if read == buf.len() => Ok(()),
        _ => Err(ErrorKind::UnexpectedEof.into()),
    }
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8]) -> io::Result<()> {
    match std::os::windows::fs::FileExt::seek_write(file, buf, 0)? {
        written if written == buf.len() => Ok(()),
        _ => Err(ErrorKind::WriteZero.into()),
    }
}

/// The bytes of the file at `path`, where they are `len` long (any length with none); `None`
/// where they are not.
fn read_whole(path: &Path, len: Option<u64>) -> io::Result<Option<Vec<u8>>> {
    let mut opened = File::open(path)?;
    let held = opened.metadata()?.len();
    if len.is_some_and(|len| len != held) {
        return Ok(None);
    }
    let mut bytes = vec![0; usize::try_from(held).map_err(|_| ErrorKind::OutOfMemory)?];
    opened.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}

/// The lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The pipeline cache header `data` begins with, the device's `vkGetPipelineCacheData` of an empty
/// cache: its first 32 bytes, where they are a `VkPipelineCacheHeaderVersionOne`, whose size field
/// says 32 or more and whose version field says one (both least significant byte first, whatever
/// the host's byte order). None otherwise: a driver may cache nothing, and lavapipe (Mesa 26.2.4)
/// gives 20 bytes for an empty cache, shorter than a header. A device whose data begins with no
/// header keeps no blob; its cache lives with the device.
pub(crate) fn usable_header(data: &[u8]) -> Option<[u8; 32]> {
    let header = data.first_chunk::<32>()?;
    let field = |at: usize| u32::from_le_bytes([0, 1, 2, 3].map(|i| header[at + i]));
    (field(0) >= 32 && field(4) == 1).then_some(*header)
}

/// The name of the blob of a device whose driver writes the pipeline cache header `header`: the
/// whole header in hex, opaque bytes.
pub(crate) fn blob_name(header: &[u8; 32]) -> String {
    format!("{BLOB}-{}", hex(header))
}

/// Removes the blobs in `dir` past the [`BLOBS_KEPT`] most recently written, the blob named `own`
/// counted among the kept
/// whether or not it is there yet. A temporary file, and anything not named as a blob, stays.
pub(crate) fn prune_blobs(dir: &Path, own: &str) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut others: Vec<(SystemTime, PathBuf)> = (entries.flatten())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let digits = name.strip_prefix(BLOB)?.strip_prefix('-')?;
            let blob = digits.len() == 64 && digits.bytes().all(|byte| byte.is_ascii_hexdigit());
            (blob && name != own).then_some(())?;
            let time = entry.metadata().and_then(|meta| meta.modified()).ok()?;
            Some((time, entry.path()))
        })
        .collect();
    others.sort_unstable_by_key(|(time, _)| std::cmp::Reverse(*time));
    for (_, path) in others.into_iter().skip(BLOBS_KEPT - 1) {
        let _ = fs::remove_file(path);
    }
}

/// Writes `parts`, one after another, to `path` through a temporary file beside it, named for this
/// process and this write, then renamed over it.
fn write_whole(path: &Path, parts: &[&[u8]]) -> io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Relaxed)
    ));
    let temp = PathBuf::from(temp);
    let written = File::create(&temp)
        .and_then(|mut file| parts.iter().try_for_each(|part| file.write_all(part)));
    match written.and_then(|()| fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(error)
        }
    }
}

/// A hash of `bytes`, eight at a time in four lanes, each step a bijection of its lane, so any one
/// changed word changes it: the state's and the blob's check that they were written whole, fast
/// enough to check a 32 MiB blob in a few milliseconds. Not a defence against a writer that means
/// harm.
pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let step = |lane: u64, word: u64| (lane ^ word).wrapping_mul(K).rotate_left(29);
    let mut lanes = [K, K ^ 1, K ^ 2, K ^ 3];
    let (chunks, tail) = bytes.as_chunks::<32>();
    for chunk in chunks {
        for (lane, word) in lanes.iter_mut().zip(chunk.as_chunks::<8>().0) {
            *lane = step(*lane, u64::from_le_bytes(*word));
        }
    }
    let hash = lanes.into_iter().fold(bytes.len() as u64, step);
    tail.iter()
        .fold(hash, |hash, &byte| step(hash, u64::from(byte)))
}

/// The blob's data, after its magic, its length and its checksum, where all three check.
pub(crate) fn unframe(file: &[u8]) -> Option<&[u8]> {
    let (magic, rest) = file.split_first_chunk::<8>()?;
    let (len, rest) = rest.split_first_chunk::<8>()?;
    let (sum, data) = rest.split_first_chunk::<8>()?;
    (magic == MAGIC
        && u64::from_le_bytes(*len) == data.len() as u64
        && u64::from_le_bytes(*sum) == checksum(data))
    .then_some(data)
}

/// The bytes before `data` in its blob: the magic, its length and its checksum.
pub(crate) fn frame(data: &[u8]) -> [u8; 24] {
    let mut head = [0; 24];
    head[..8].copy_from_slice(MAGIC);
    head[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    head[16..].copy_from_slice(&checksum(data).to_le_bytes());
    head
}

/// A device's caches: the lowerings in the store, and the `VkPipelineCache` every `VkPipeline`
/// is made through, with its blob.
pub(crate) struct Caches {
    store: Option<Arc<Store>>,
    /// The cache directory, where there is one.
    dir: Option<PathBuf>,
    /// The device's blob in it, named once the reaper has read the device's header.
    blob: OnceLock<PathBuf>,
    pipelines: Mutex<Pipelines>,
    /// Signalled once the reaper made the cache.
    made: Condvar,
    /// Creations through the cache that have not ended.
    creating: AtomicUsize,
    /// When the last creation ended, in nanoseconds of [`host_time`].
    created: AtomicU64,
    /// Set by a creation's end, cleared by the reaper's look at the cache.
    grown: AtomicBool,
    /// [`PIPELINE_CACHE_BOUND`], lower in a test.
    bound: AtomicU64,
    /// The lowerings and refusals waiting for the reaper to write them.
    pending: Mutex<Pending>,
    /// Whether the reaper's writes wait, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    writes_stalled: AtomicBool,
    /// Whether the reaper has seeded the cache from the blob, or found none to take.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    seeded: AtomicBool,
    /// The reaper's looks at the cache.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    looks: AtomicUsize,
    /// The blob's writes.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    writes: AtomicUsize,
    /// Whether the seed waits before its merge, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    merge_held: AtomicBool,
    /// Whether the seed is waiting before its merge.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    merge_waiting: AtomicBool,
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
thread_local! {
    /// Whether the caches made next on this thread hold their seed before its merge, for a test.
    static HOLD_NEXT_MERGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The `VkPipelineCache` creations take, and those a fresh one replaced.
struct Pipelines {
    /// Whether the reaper made the cache, or found the driver makes none.
    made: bool,
    /// Whether the reaper is reading and taking in the device's blob, creations going through
    /// the empty cache meanwhile.
    seeding: bool,
    /// The cache creations take; none where the driver made none.
    current: Option<Arc<PipelineCache>>,
    /// The caches a fresh one replaced, each destroyed once no creation holds it.
    retired: Vec<Arc<PipelineCache>>,
    /// The size of the current cache's data when the blob was last written from it, or when it
    /// was made: what the next write must grow past.
    written: u64,
}

/// A `VkPipelineCache` of the device's, held by each creation through it.
pub(crate) struct PipelineCache(vk::PipelineCache);

/// Hands the bytes of a lowered function of key digest `key` to the reaper, which writes them to
/// the store, off every guest thread. Where the lowerings waiting for it hold [`PENDING_BYTES`]
/// already, they are dropped and counted (`Metrics::cache_writes_dropped`), never waited for: a
/// lowering is only a cache, and a later state lowers it again.
pub(crate) fn keep_translation(core: &Core, key: &[u8], bytes: Vec<u8>) {
    if core.caches.store.is_none() {
        return;
    }
    let queued = lock(&core.caches.pending).push(Caches::translation_file(key), bytes);
    match queued {
        true => core.reaper.poke(),
        false => core.counters.count(Metric::CacheWritesDropped),
    }
}

/// Hands the translator's refusal of the lowering of key digest `key` to the reaper, as
/// [`keep_translation`] does a lowering.
pub(crate) fn keep_refusal(core: &Core, key: &[u8]) {
    keep_translation(core, key, REFUSED.to_vec());
}

/// The files waiting for the reaper to write them, in order, at most `bound` bytes of them.
pub(crate) struct Pending {
    files: VecDeque<(String, Vec<u8>)>,
    bytes: u64,
    bound: u64,
}

impl Pending {
    /// An empty queue of at most `bound` bytes.
    pub(crate) fn new(bound: u64) -> Pending {
        Pending {
            files: VecDeque::new(),
            bytes: 0,
            bound,
        }
    }

    /// Queues `bytes` under `file`; false, queuing nothing, where they would take the queue past
    /// its bound.
    pub(crate) fn push(&mut self, file: String, bytes: Vec<u8>) -> bool {
        let len = bytes.len() as u64;
        if self.bytes.saturating_add(len) > self.bound {
            return false;
        }
        self.bytes += len;
        self.files.push_back((file, bytes));
        true
    }

    /// The first file queued, taken off the queue.
    pub(crate) fn pop(&mut self) -> Option<(String, Vec<u8>)> {
        let (file, bytes) = self.files.pop_front()?;
        self.bytes -= bytes.len() as u64;
        Some((file, bytes))
    }
}

/// A creation through the device's `VkPipelineCache`, from [`creation`] until it is dropped.
pub(crate) struct Creation<'a> {
    core: &'a Core,
    cache: Option<Arc<PipelineCache>>,
}

impl Creation<'_> {
    /// The cache to make the pipeline through; null where the driver made none.
    pub(crate) fn handle(&self) -> vk::PipelineCache {
        self.cache
            .as_ref()
            .map_or(vk::PipelineCache::null(), |cache| cache.0)
    }
}

impl Drop for Creation<'_> {
    fn drop(&mut self) {
        let caches = &self.core.caches;
        caches.created.fetch_max(host_time(), Relaxed);
        caches.creating.fetch_sub(1, SeqCst);
        // The first creation since the reaper's last look tells it to look again once creation
        // is quiet.
        if !caches.grown.swap(true, SeqCst) {
            self.core.reaper.poke();
        }
    }
}

/// A creation through `core`'s `VkPipelineCache`: where the reaper has not made the cache yet,
/// which it does first when it starts, the calling thread waits for it.
pub(crate) fn creation(core: &Core) -> Creation<'_> {
    let caches = &core.caches;
    caches.creating.fetch_add(1, SeqCst);
    let mut pipelines = lock(&caches.pipelines);
    while !pipelines.made {
        pipelines = (caches.made.wait(pipelines)).unwrap_or_else(PoisonError::into_inner);
    }
    if pipelines.seeding {
        core.counters.count(Metric::CreationsBeforeSeed);
    }
    let cache = pipelines.current.clone();
    Creation { core, cache }
}

/// A `VkPipelineCache` of `device` started from `data`, internally synchronized.
///
/// # Safety
///
/// `data` is empty, or what `vkGetPipelineCacheData` gave, whole.
unsafe fn make_cache(device: &ash::Device, data: &[u8]) -> Option<vk::PipelineCache> {
    let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
    // SAFETY: the caller's contract; data of another device or version is ignored by the driver,
    // and a refusal is answered by the caller.
    unsafe { device.create_pipeline_cache(&info, None) }.ok()
}

/// The size of `cache`'s data, `vkGetPipelineCacheData` with no buffer; 0 where the driver
/// fails it.
///
/// # Safety
///
/// The cache is `device`'s.
unsafe fn data_size(device: &ash::Device, cache: vk::PipelineCache) -> u64 {
    let mut size = 0;
    // SAFETY: the caller's contract; no buffer is given, so only the size is written.
    let result = unsafe {
        (device.fp_v1_0().get_pipeline_cache_data)(
            device.handle(),
            cache,
            &mut size,
            std::ptr::null_mut(),
        )
    };
    match result {
        vk::Result::SUCCESS => size as u64,
        _ => 0,
    }
}

impl Caches {
    /// A device's caches, in `dir` where there is one, the lowerings at most `cap` bytes; its
    /// removal thread counts in `counters`' thread gauge.
    pub(crate) fn new(dir: Option<&Path>, cap: u64, counters: &Counters) -> Caches {
        Caches {
            store: dir.map(|dir| Store::open(dir.to_owned(), cap, counters.clone())),
            dir: dir.map(Path::to_owned),
            blob: OnceLock::new(),
            pipelines: Mutex::new(Pipelines {
                made: false,
                seeding: false,
                current: None,
                retired: Vec::new(),
                written: 0,
            }),
            made: Condvar::new(),
            creating: AtomicUsize::new(0),
            created: AtomicU64::new(0),
            grown: AtomicBool::new(false),
            bound: AtomicU64::new(PIPELINE_CACHE_BOUND),
            pending: Mutex::new(Pending::new(PENDING_BYTES)),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            writes_stalled: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            seeded: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            looks: AtomicUsize::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            writes: AtomicUsize::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            merge_held: AtomicBool::new(HOLD_NEXT_MERGE.with(std::cell::Cell::take)),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            merge_waiting: AtomicBool::new(false),
        }
    }

    /// Makes the seed of the device made next on this thread wait before its merge, until
    /// [`Caches::release_merge`], for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold_next_merge() {
        HOLD_NEXT_MERGE.with(|hold| hold.set(true));
    }

    /// Whether the seed is waiting before its merge.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn merge_waiting(&self) -> bool {
        (self.merge_waiting).load(std::sync::atomic::Ordering::Acquire)
    }

    /// Lets the seed's merge go on.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn release_merge(&self) {
        (self.merge_held).store(false, std::sync::atomic::Ordering::Release);
    }

    /// On the reaper, before the seed's merge: waits while a test holds the merge.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fn wait_while_merge_held(&self) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        if !self.merge_held.load(Acquire) {
            return;
        }
        self.merge_waiting.store(true, Release);
        while self.merge_held.load(Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.merge_waiting.store(false, Release);
    }

    /// The device's blob, where it has a cache directory and the reaper has seeded the cache.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn blob(&self) -> Option<&Path> {
        self.blob.get().map(PathBuf::as_path)
    }

    /// Makes the reaper's writes of lowerings wait, or go on, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stall_writes(&self, stalled: bool) {
        (self.writes_stalled).store(stalled, std::sync::atomic::Ordering::Release);
    }

    /// Sets the bound of the lowerings waiting to be written, for a test; the bytes waiting.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn set_pending_bound(&self, bytes: u64) -> u64 {
        let mut pending = lock(&self.pending);
        pending.bound = bytes;
        pending.bytes
    }

    /// Sets the cache's bound for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn set_bound(&self, bytes: u64) {
        self.bound.store(bytes, Relaxed);
    }

    /// The caches a fresh one replaced and the reaper has not destroyed yet.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn retired(&self) -> usize {
        lock(&self.pipelines).retired.len()
    }

    /// Whether the reaper has seeded the cache from the blob, or found none to take.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn is_seeded(&self) -> bool {
        self.seeded.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The current cache, held as a creation holds it, without counting as one.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold(&self) -> Option<Arc<PipelineCache>> {
        lock(&self.pipelines).current.clone()
    }

    /// The current cache's data as the driver gives it, for a test; none where it made no cache.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) unsafe fn data(&self, device: &ash::Device) -> Option<Vec<u8>> {
        let current = self.hold()?;
        // SAFETY: the caller's contract; the cache is internally synchronized.
        unsafe { device.get_pipeline_cache_data(current.0) }.ok()
    }

    /// The reaper's looks at the cache, and the blob writes of its looks and of the drop.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn looks_and_writes(&self) -> (usize, usize) {
        (
            self.looks.load(std::sync::atomic::Ordering::Acquire),
            self.writes.load(std::sync::atomic::Ordering::Acquire),
        )
    }

    fn translation_file(key: &[u8]) -> String {
        format!("{TRANSLATIONS}/{}", hex(key))
    }

    /// What `decode` makes of the bytes kept for a lowered kernel of key digest `key`, where they
    /// decode. A file that holds the translator's refusal is no lowering, and stays.
    pub(crate) fn translation<T>(
        &self,
        key: &[u8],
        decode: impl FnOnce(&[u8]) -> Option<T>,
    ) -> Option<T> {
        let file = Caches::translation_file(key);
        let kept = self.store.as_ref()?.get(&file, |bytes| match bytes {
            REFUSED => Some(None),
            lowering => decode(lowering).map(Some),
        });
        kept.flatten()
    }

    /// The next lowering or refusal waiting for the reaper to write it.
    pub(crate) fn next_write(&self) -> Option<(String, Vec<u8>)> {
        lock(&self.pending).pop()
    }

    /// On the reaper: writes `bytes` under `file` in the store, as [`keep_translation`] queued
    /// them.
    pub(crate) fn write_kept(&self, file: &str, bytes: &[u8], counters: &Counters) {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        while self
            .writes_stalled
            .load(std::sync::atomic::Ordering::Acquire)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(store) = &self.store {
            if store.put(file, bytes) {
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

    /// On the reaper, first: makes the device's `VkPipelineCache` empty, which creations waiting
    /// for a cache take at once; then, where the device has a blob that checks, whose data begins
    /// with the header the device's driver writes and that the driver takes, makes a cache from
    /// it, merges into it what creations compiled in the empty cache meanwhile, and makes it the
    /// cache creations take from then on. A blob that is not taken is removed. So no creation
    /// waits for the blob's read or the driver's parse of it, which for a blob of 32 MiB take tens
    /// of milliseconds, or for the merge, which takes hundreds on some drivers: a creation during
    /// it goes through the empty cache. A creation through the empty cache that ends once the
    /// merge has begun is not in this run's blob, and the next start compiles it again.
    ///
    /// The blob is named by the whole header in hex ([`blob_name`]); before its read, the blobs
    /// past the [`BLOBS_KEPT`] most recently written are removed. A device whose cache data
    /// begins with no header ([`usable_header`]) keeps no blob: it reads none and writes none.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    pub(crate) unsafe fn seed(&self, device: &ash::Device, counters: &Counters) {
        // SAFETY: no data.
        let empty = unsafe { make_cache(device, &[]) };
        let header = empty.and_then(|empty| {
            // SAFETY: the cache is the device's.
            let data = unsafe { device.get_pipeline_cache_data(empty) }.ok()?;
            usable_header(&data)
        });
        // SAFETY: the cache is the device's.
        let size = empty.map_or(0, |cache| unsafe { data_size(device, cache) });
        counters.set(Gauge::PipelineCacheBytes, size);
        let mut pipelines = lock(&self.pipelines);
        pipelines.current = empty.map(|cache| Arc::new(PipelineCache(cache)));
        pipelines.written = size;
        pipelines.made = true;
        pipelines.seeding = self.dir.is_some() && header.is_some();
        drop(pipelines);
        self.made.notify_all();
        if let (Some(dir), Some(header)) = (&self.dir, header) {
            let name = blob_name(&header);
            prune_blobs(dir, &name);
            let path = dir.join(name);
            let file = read_whole(&path, None).ok().flatten();
            let data = (file.as_deref())
                .and_then(unframe)
                .filter(|data| data.starts_with(&header));
            // SAFETY: the data is what `vkGetPipelineCacheData` gave for a device of this header,
            // whole, as its length and checksum show.
            let seeded = data.and_then(|data| unsafe { make_cache(device, data) });
            match seeded {
                // SAFETY: the cache is the device's, and this thread's alone.
                Some(seeded) => unsafe { self.take_seeded(device, seeded, counters) },
                None if file.is_some() => {
                    let _ = fs::remove_file(&path);
                }
                None => {}
            }
            let _ = self.blob.set(path);
        }
        lock(&self.pipelines).seeding = false;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.seeded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Makes `seeded`, made from the blob, the cache creations take, once what they compiled in
    /// the empty cache meanwhile is merged into it; the empty cache is destroyed once no creation
    /// holds it. Its size is what the next write must grow past: the blob's, without the merge.
    ///
    /// # Safety
    ///
    /// `seeded` is the device's, and no other thread holds it.
    unsafe fn take_seeded(
        &self,
        device: &ash::Device,
        seeded: vk::PipelineCache,
        counters: &Counters,
    ) {
        // SAFETY: the cache is the device's.
        let written = unsafe { data_size(device, seeded) };
        // Taken in a statement of its own, so the lock is let go before the merge, which takes
        // hundreds of milliseconds on some drivers: a creation meanwhile takes the empty cache and
        // never waits for it.
        let empty = lock(&self.pipelines).current.clone();
        if let Some(empty) = empty {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            self.wait_while_merge_held();
            // SAFETY: both caches are the device's; `seeded`, the destination, is this thread's
            // alone, and the source is internally synchronized against the creations through it.
            // Where the merge fails, `seeded` holds the blob's pipelines alone.
            let _ = unsafe { device.merge_pipeline_caches(seeded, &[empty.0]) };
        }
        // SAFETY: the cache is the device's.
        let size = unsafe { data_size(device, seeded) };
        let mut pipelines = lock(&self.pipelines);
        let empty = pipelines.current.replace(Arc::new(PipelineCache(seeded)));
        pipelines.retired.extend(empty);
        pipelines.written = written;
        pipelines.seeding = false;
        drop(pipelines);
        counters.set(Gauge::PipelineCacheBytes, size);
        // SAFETY: the caller's contract.
        unsafe { self.reap(device) };
    }

    /// When the reaper is to look at the cache ([`Caches::look`]), at `now` in nanoseconds of
    /// [`host_time`]: none where no creation ended since its last look; else once creation has
    /// been quiet, no creation running and none ending, for [`QUIET_NANOS`], zero once it has.
    pub(crate) fn look_in(&self, now: u64) -> Option<Duration> {
        if !self.grown.load(SeqCst) {
            return None;
        }
        if self.creating.load(SeqCst) > 0 {
            return Some(Duration::from_nanos(QUIET_NANOS));
        }
        let quiet = self.created.load(Relaxed).saturating_add(QUIET_NANOS);
        Some(Duration::from_nanos(quiet.saturating_sub(now)))
    }

    /// On the reaper, creation quiet: destroys each replaced cache no creation holds, reads the
    /// cache's size, and past the bound starts a fresh cache, else writes the blob where the
    /// data grew by more than an eighth since the last write.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    pub(crate) unsafe fn look(&self, device: &ash::Device, counters: &Counters) {
        // SAFETY: the caller's contract.
        unsafe { self.replace_or_write(device, counters) };
        // Counted once the look's effects are in place, so a test that waits for it sees them.
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.looks
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// The work of [`Caches::look`].
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    unsafe fn replace_or_write(&self, device: &ash::Device, counters: &Counters) {
        self.grown.store(false, SeqCst);
        // SAFETY: the caller's contract.
        let Some(current) = (unsafe { self.reap(device) }) else {
            return;
        };
        // SAFETY: the cache is the device's.
        let size = unsafe { data_size(device, current.0) };
        counters.set(Gauge::PipelineCacheBytes, size);
        if size > self.bound.load(Relaxed) {
            // SAFETY: no data.
            let Some(fresh) = (unsafe { make_cache(device, &[]) }) else {
                return;
            };
            // SAFETY: the cache is the device's.
            let fresh_size = unsafe { data_size(device, fresh) };
            let mut pipelines = lock(&self.pipelines);
            let old = pipelines.current.replace(Arc::new(PipelineCache(fresh)));
            pipelines.retired.extend(old);
            pipelines.written = fresh_size;
            drop(pipelines);
            counters.set(Gauge::PipelineCacheBytes, fresh_size);
            drop(current);
            // SAFETY: the caller's contract.
            unsafe { self.reap(device) };
            return;
        }
        let written = lock(&self.pipelines).written;
        if size > written + written / 8 {
            // SAFETY: the cache is the device's.
            unsafe { self.write(device, &current, size, counters) };
        }
    }

    /// Destroys each replaced cache no creation holds; the current cache.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    unsafe fn reap(&self, device: &ash::Device) -> Option<Arc<PipelineCache>> {
        let mut pipelines = lock(&self.pipelines);
        // A replaced cache is never handed out again: once the list holds its last reference,
        // no creation holds it.
        let (idle, held) =
            (pipelines.retired.drain(..)).partition(|old| Arc::strong_count(old) == 1);
        pipelines.retired = held;
        let current = pipelines.current.clone();
        drop(pipelines);
        // SAFETY: the caches are the device's, and no creation holds them.
        unsafe { destroy(device, idle) };
        current
    }

    /// Writes `cache`'s data, of size `size` as the driver last gave it, over the device's blob,
    /// where the directory is writable and the data is within the bound.
    ///
    /// # Safety
    ///
    /// The cache is `device`'s.
    unsafe fn write(
        &self,
        device: &ash::Device,
        cache: &PipelineCache,
        size: u64,
        counters: &Counters,
    ) {
        let (Some(path), Some(store)) = (self.blob.get(), &self.store) else {
            return;
        };
        if !store.writable() {
            return;
        }
        // SAFETY: the caller's contract; the cache is internally synchronized.
        let Ok(data) = (unsafe { device.get_pipeline_cache_data(cache.0) }) else {
            return;
        };
        let len = data.len() as u64;
        if len > self.bound.load(Relaxed) {
            return;
        }
        if write_whole(path, &[&frame(&data), &data]).is_ok() {
            counters.cache_written(24 + len);
            lock(&self.pipelines).written = size;
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            self.writes
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// At the device's drop, no creation running: writes the blob where the cache's data changed
    /// since the last write, then destroys every cache.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches, and no creation runs.
    /// At the device's teardown, once nothing writes to the store: waits for its removal thread.
    pub(crate) fn join_removal(&self) {
        if let Some(store) = &self.store {
            store.join_removal();
        }
    }

    pub(crate) unsafe fn close(&self, device: &ash::Device, counters: &Counters) {
        let (current, written) = {
            let mut pipelines = lock(&self.pipelines);
            (pipelines.current.take(), pipelines.written)
        };
        if let Some(current) = &current {
            // SAFETY: the cache is the device's.
            let size = unsafe { data_size(device, current.0) };
            if size != written {
                // SAFETY: the cache is the device's.
                unsafe { self.write(device, current, size, counters) };
            }
        }
        let mut pipelines = lock(&self.pipelines);
        // SAFETY: the caches are the device's, and no creation runs.
        unsafe { destroy(device, pipelines.retired.drain(..).chain(current)) };
        counters.set(Gauge::PipelineCacheBytes, 0);
    }
}

/// Destroys `caches`.
///
/// # Safety
///
/// The caches are `device`'s, and no creation holds them.
unsafe fn destroy(device: &ash::Device, caches: impl IntoIterator<Item = Arc<PipelineCache>>) {
    for cache in caches {
        // SAFETY: the caller's contract.
        unsafe { device.destroy_pipeline_cache(cache.0, None) };
    }
}
