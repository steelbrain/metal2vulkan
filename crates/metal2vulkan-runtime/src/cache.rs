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
//! (microseconds); then it reads the blob, makes a cache from it and makes that the cache
//! creations take. So no creation waits for the blob's read or parse. What creations compiled in
//! the empty cache meanwhile is not in this run's blob: the empty cache retires at the swap and is
//! destroyed once no creation holds it, and a later run compiles those pipelines again, once. A
//! blob that does not check, whose data does not begin with the device's header, or that the
//! driver refuses, is removed, and the empty cache stays.
//!
//! The reaper writes the cache's data over the blob, a temporary file renamed over it and never
//! appended to, once creation is quiet (no creation running and none for [`QUIET_NANOS`]) after a
//! creation: at most once a quiet period, and only where the size the driver gives grew by more
//! than an eighth since the last write, so a run that makes nothing new writes nothing and a
//! device's writes over its life come to a few times the blob's size. The device's drop writes it
//! once more where the size changed at all. A crash between writes loses only the pipelines made
//! since the last. No guest call waits for a write.
//!
//! **Reads exclude creations.** Every creation holds a lock shared ([`Caches::creations`]) from
//! before it takes a cache until it ends, and the reaper's look holds it exclusive around its
//! reads of the cache's size and data, taken by a try: where a creation began after
//! creation was quiet, the look is skipped and counted (`Metrics::pipeline_cache_looks_skipped`),
//! and the reaper looks again once creation is quiet again. So the reaper never waits for a
//! creation, and a creation that begins during a look waits for that look alone, its wait counted
//! (`Metrics::pipeline_cache_lock_wait_nanos`). Vulkan lets a thread read an internally
//! synchronized cache's data while another creates through it, but Intel's Windows driver
//! (101.2145) for the UHD Graphics 630 has faulted inside `vkGetPipelineCacheData` so.
//!
//! The cache holds at most [`PIPELINE_CACHE_BOUND`]: a cache past it when the reaper looks is
//! replaced by a fresh, empty one that creations take from then on, and destroyed once no creation
//! holds it (counted in `Metrics::pipeline_caches_replaced`); the next write replaces the blob with
//! the fresh cache's data, so the blob holds at most the bound too, with no second file. A write
//! of data past the bound, the teardown's of a cache that grew past it after the last look, is
//! skipped (`Metrics::pipeline_cache_writes_skipped`). [`Gauges::pipeline_cache_bytes`] is the
//! cache's size at the reaper's last look. Without a directory the cache is the same, and nothing is on disk.
//!
//! **The lowerings.** The store keeps one file per lowered function (a kernel, or a render
//! pipeline's vertex or fragment function) under `translations/`, named by the hex of its
//! `FunctionCompileKey`, which covers everything that changes the lowering's output: the
//! function's AIR, entry, constants and linked functions, its lowering inputs, the device
//! profile's fingerprint and the translator's semantics version. The bytes are the translator's
//! self-checking encoding (`CompiledFunction::to_bytes`), and bytes that do not decode for the key
//! are a miss. A partial-threadgroups variant the translator refused is kept too, as a marker, and
//! so is the translator's answer that a kernel observes its threadgroup partition, under its
//! variant's key, until that variant's lowering replaces it.
//!
//! **The records.** A render content whose functions a device lowered keeps a record under
//! `renders/`, named by the hex of the translator's key of its descriptor
//! (`Device::queried_render_descriptor_key`), which reads no function: the `FunctionCompileKey` of
//! each of its functions' lowerings ([`Lowerings`]). A state of a content the device does not hold
//! reads its record in place of the translator's key of its compile (`Device::render_variant_key`),
//! which decodes and specializes both functions, and then reads the lowerings the record names. The
//! record's name covers everything that can change either lowering: every field of the descriptor,
//! the fields the specialization reads among them; each function by its library's AIR hash (the
//! library's content), its entry, its constant values as supplied, its linked functions and its
//! dynamic libraries; the device profile's fingerprint; and the translator's semantics version. Its
//! bytes ([`record_bytes`]) hold the translator's semantics and artifact format versions and the
//! descriptor's key again, under a checksum, so a record of another version or of another content,
//! or one cut short, is a miss, and is removed. A record whose lowering the store no longer holds
//! is a miss too, counted (`Metrics::render_records_stale`): the state takes the path of a content
//! with no record, and keeps the record again. A record is [`RECORD_LEN`] bytes, written and
//! counted as a lowering is, within the store's cap.
//!
//! The files live in two generations, `g<N>/` (young) and `g<N-1>/` (old), which the consumer's cap
//! ([`DeviceOptions::pipeline_cache_cap`]) bounds at half of it each. A lookup tries young, then
//! old; a hit in old is renamed into young, which is its mark of use. Generations start at 1, and
//! the first has no old: a lookup there tries young alone. A write goes into young, made by the
//! reaper: a state hands its lowering over ([`keep_translation`]) and returns, and a lowering that
//! would take the ones waiting past [`PENDING_BYTES`] is dropped and counted, never waited for,
//! except that one larger than that by itself is queued where nothing else waits. The
//! bytes young holds are counted in `state`, 40 bytes beside the generations: young's number, its
//! bytes and a checksum of both, then the newest translator versions a store opened the directory
//! under ([`Versions`]) and a checksum of them, read and written under a lock on the file, which
//! every store on the directory, in this process or another, holds open for its life. The write
//! that takes young past half the cap makes the next generation young: young becomes old, and the
//! old one is dead. A store keeps young's number as it last read or wrote the state, so a lookup
//! reads no state: a move another store made is seen at this store's next write, and until then
//! its lookups try the generations it knew, a miss at worst. A thread of the store's own removes
//! the dead generation and ends, so no thread that submits, waits for or completes work removes
//! files. So a file stays at least half the cap and at most the cap of other writes and moves
//! after its last use, until a newer translator's open (below), opening reads one file whatever
//! the store holds, and nothing per file is kept in memory. The same thread removes the
//! `pipelines/` directories under young and old in which an earlier version kept one file of
//! pipeline data per `VkPipeline`; their bytes stay counted in young until it moves on.
//!
//! **A newer translator.** Every lowering's and record's name holds the translator's semantics
//! version, and a lowering decodes under its own artifact format version alone, so a store under
//! a newer translator reads no file an older one wrote: in both generations they are dead. So a
//! store whose translator is newer, in either version, than the newest the state records retires
//! both at its open: under the lock, young moves two generations on, counted empty, so the young
//! and old it found are both below old and the removal thread takes them, and the state records
//! the higher of each version. A store at the versions the state records, or older ones, retires
//! nothing and records nothing. So builds of two versions that share a directory, at once or in
//! turn, keep each other's files after the newer one's first open: the older one's go into young
//! and stay by use, as every store's do. Versions that are missing or do not check after young's
//! number and bytes that do, as in an earlier version's state file, read as older than every
//! translator's, so the first open of a store that records them retires both generations once.
//! An earlier version writes only the first 24 bytes, at the file's start, and leaves the versions
//! as they are. A store that is still open under an older translator when a newer one retires its
//! generations misses in them until its next write reads the new young.
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
use crate::pipeline::VariantRefusal;
use crate::times::host_time;
use ash::vk;
use metal2vulkan::cache::{Hash256, ARTIFACT_FORMAT_VERSION, SEMANTICS_VERSION};
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{
    Arc, Condvar, Mutex, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
    TryLockError,
};
use std::time::{Duration, SystemTime};

/// The cap of the lowerings in a cache directory where the consumer names none: half of it, what
/// the store is sure to keep, is about seven times the 9.16 MB a long recorded session's 282
/// lowerings fill.
pub(crate) const DEFAULT_CAP: u64 = 128 << 20;
/// The most bytes of data a device's `VkPipelineCache` holds when the reaper looks before it is
/// replaced by a fresh one, and so the most its blob holds: 128 MiB, past the 62.6 MB KosmicKrisp
/// (the Mac host's driver) writes for a long recorded session's 448 pipelines, against 1.5-11.1 MB
/// on NVIDIA's, RADV's and ANV's drivers for the same pipelines.
pub(crate) const PIPELINE_CACHE_BOUND: u64 = 128 << 20;
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
/// recorded session's 282 lowerings fill. A lowering larger than this alone is queued where none
/// waits, so the heaviest kernels, the costliest to lower again, are kept too.
pub(crate) const PENDING_BYTES: u64 = 8 << 20;
/// What a translation file holds where the translator refused the lowering, but for the vote word
/// ([`REFUSED_VOTE`]): bytes no lowering encodes to, and that a file cut short is not.
const REFUSED: &[u8] = b"m2v-refused";
/// What a partial-threadgroups variant's translation file holds where the translator refused it
/// for its predicated walk's vote word ([`VariantRefusal::VoteWord`]): bytes no lowering encodes
/// to, nor [`REFUSED`] or [`OBSERVES`], and that a file cut short is not.
pub(crate) const REFUSED_VOTE: &[u8] = b"m2v-refused-vote";
/// What a partial-threadgroups variant's translation file holds where the translator answered
/// that the kernel observes its threadgroup partition, until the variant is lowered over it:
/// bytes no lowering encodes to, nor [`REFUSED`] or [`REFUSED_VOTE`], and that a file cut short is
/// not. The file is named by the variant's `FunctionCompileKey`, which covers everything that changes the answer:
/// the kernel's AIR, entry, constants and linked functions, its lowering inputs, the device
/// profile's fingerprint and the translator's semantics version, which a change to the answer
/// bumps. It is 12 bytes, written and counted as a lowering is, within the store's cap.
const OBSERVES: &[u8] = b"m2v-observes";
/// The translations' directory, under a generation.
const TRANSLATIONS: &str = "translations";
/// The render contents' records' directory, under a generation.
const RENDERS: &str = "renders";
/// The first bytes of a render content's record: its layout's name and version.
const RECORD: &[u8; 8] = b"m2vrndr1";
/// The length of a render content's record ([`record_bytes`]): its first bytes, two versions, the
/// descriptor's key, two stages of a tag and a digest each, and the checksum.
pub(crate) const RECORD_LEN: usize = 8 + 4 + 4 + 32 + 2 * 33 + 8;
/// The directory, under a generation, an earlier version kept a file of pipeline data per
/// `VkPipeline` in: removed where it is found.
const PIPELINES: &str = "pipelines";
/// The state file, under the cache directory.
const STATE: &str = "state";
/// Where the newest versions the state file records start in it, after young's number, its
/// bytes and their checksum.
const NEWEST_AT: u64 = 24;

/// Young's number and the bytes counted in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct State {
    pub(crate) young: u64,
    pub(crate) bytes: u64,
}

/// The translator's versions a store writes its files under: its semantics version, which every
/// lowering's and record's name holds, and its artifact format version, under which alone a
/// lowering decodes. So a store under newer versions reads no file of older ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Versions {
    pub(crate) semantics: u32,
    pub(crate) format: u32,
}

impl Versions {
    /// This build's translator's.
    pub(crate) const BUILT: Versions = Versions {
        semantics: SEMANTICS_VERSION,
        format: ARTIFACT_FORMAT_VERSION,
    };

    /// Whether either version is above `other`'s.
    fn newer_than(self, other: Versions) -> bool {
        self.semantics > other.semantics || self.format > other.format
    }

    /// The higher of each version.
    fn newest(self, other: Versions) -> Versions {
        Versions {
            semantics: self.semantics.max(other.semantics),
            format: self.format.max(other.format),
        }
    }
}

/// The lowerings of a cache directory, `translations/<name>` under two generations of at most
/// half of `cap` bytes each (the module's doc).
pub(crate) struct Store {
    root: PathBuf,
    /// Half the cap: the most bytes young counts.
    half: u64,
    /// The translator's versions the store writes its files under.
    versions: Versions,
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
    /// The store in `root` under the translator's `versions`, made where it is missing. It
    /// retires both generations where the directory records older versions ([`Store::meet`]),
    /// and starts the removal of the generations that leaves dead, of a generation a removal
    /// before left, and of an earlier version's pipeline data.
    pub(crate) fn open(
        root: PathBuf,
        cap: u64,
        counters: Counters,
        versions: Versions,
    ) -> Arc<Store> {
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
            versions,
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
        store.meet();
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

    /// At the open, under the state file's lock: where the state records versions older than
    /// the store's in either, makes young two generations on, counted empty, so the young and old
    /// it found are both dead (no lookup of the store's reads their files), and records the
    /// higher of each version. A state that records the store's versions or newer ones keeps its
    /// generations and its record: a store of an older build on the directory empties nothing of
    /// a newer one's. Versions that are missing or do not check after a state that does (an
    /// earlier version's state file) read as older than every translator's; a state that does not
    /// check at all, a new directory's among them, is rebuilt (the module's doc) and retires
    /// nothing.
    fn meet(&self) {
        self.locked(|file| {
            let recorded = read_newest(file);
            if recorded.is_some_and(|recorded| !self.versions.newer_than(recorded)) {
                return Some(());
            }
            if recorded.is_some() || read_state(file).is_some() {
                self.change(file, |state| {
                    let next = state.young + 2;
                    make_dir(&self.generation(next)).ok()?;
                    *state = State {
                        young: next,
                        bytes: 0,
                    };
                    Some(())
                })?;
            }
            let newest = recorded.map_or(self.versions, |recorded| recorded.newest(self.versions));
            write_newest(file, newest).ok()
        });
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

    /// Waits for the removal thread to end: its last look for a dead generation too, which reads
    /// the state and rebuilds one that does not check.
    #[cfg(test)]
    pub(crate) fn settle(&self) {
        self.join_removal();
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
        self.locked(|file| self.change(file, change))
    }

    /// Runs `op` on the state file under its lock, where the file is open for writing.
    fn locked<T>(&self, op: impl FnOnce(&File) -> Option<T>) -> Option<T> {
        let file = self.state.as_ref().filter(|_| self.writable)?;
        let _writer = lock(&self.writer);
        file.lock().ok()?;
        let done = op(file);
        let _ = file.unlock();
        done
    }

    /// Under the lock: runs `change` on the state `file` holds, rebuilt where it does not check,
    /// and writes the state back where it answers.
    fn change<T>(&self, file: &File, change: impl FnOnce(&mut State) -> Option<T>) -> Option<T> {
        let mut state = read_state(file).or_else(|| self.rebuild(file))?;
        let answer = change(&mut state)?;
        write_state(file, state).ok()?;
        self.young.fetch_max(state.young, Relaxed);
        Some(answer)
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
            #[cfg(test)]
            std::thread::sleep(Duration::from_millis(LAST_LOOK_PAUSE.load(Relaxed)));
            // A move to the next generation during the pass found the thread running and started
            // none: its dead generation is taken here. One the pass could not remove waits for
            // the next move or open.
            let new = self.dead().iter().any(|dir| !tried.contains(dir));
            if !new || self.removing.swap(true, SeqCst) {
                return;
            }
        }
    }

    /// The generations below old, lowest first, so a removal cut short leaves the one just below
    /// old, which the next open looks for; and an earlier version's pipeline data under young and
    /// old.
    pub(crate) fn dead(&self) -> Vec<PathBuf> {
        let Some(young) = self.current().map(|state| state.young) else {
            return Vec::new();
        };
        let mut numbers = generations(&self.root);
        numbers.sort_unstable();
        let mut dead: Vec<PathBuf> = (numbers.into_iter())
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

/// A pause before the removal thread's last look for a dead generation, in milliseconds, for a
/// test that tears the state once a store is settled.
#[cfg(test)]
pub(crate) static LAST_LOOK_PAUSE: AtomicU64 = AtomicU64::new(0);

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
    let mut record = [0; NEWEST_AT as usize];
    read_at(file, &mut record, 0).ok()?;
    let word = |at: usize| u64::from_le_bytes(record[at..at + 8].try_into().expect("8 bytes"));
    let state = State {
        young: word(0),
        bytes: word(8),
    };
    (state.young >= 1 && word(16) == checksum(&record[..16])).then_some(state)
}

/// Writes `state` to `file`, with its checksum.
fn write_state(file: &File, state: State) -> io::Result<()> {
    let mut record = [0; NEWEST_AT as usize];
    record[..8].copy_from_slice(&state.young.to_le_bytes());
    record[8..16].copy_from_slice(&state.bytes.to_le_bytes());
    let sum = checksum(&record[..16]);
    record[16..].copy_from_slice(&sum.to_le_bytes());
    write_at(file, &record, 0)
}

/// The newest versions `file` records after the state, where they check.
fn read_newest(file: &File) -> Option<Versions> {
    let mut record = [0; 16];
    read_at(file, &mut record, NEWEST_AT).ok()?;
    let word = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().expect("4 bytes"));
    let sum = u64::from_le_bytes(record[8..].try_into().expect("8 bytes"));
    (sum == checksum(&record[..8])).then(|| Versions {
        semantics: word(0),
        format: word(4),
    })
}

/// Writes `versions` to `file` after the state, with their checksum.
fn write_newest(file: &File, versions: Versions) -> io::Result<()> {
    let mut record = [0; 16];
    record[..4].copy_from_slice(&versions.semantics.to_le_bytes());
    record[4..8].copy_from_slice(&versions.format.to_le_bytes());
    let sum = checksum(&record[..8]);
    record[8..].copy_from_slice(&sum.to_le_bytes());
    write_at(file, &record, NEWEST_AT)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], at: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, at)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], at: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, at)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], at: u64) -> io::Result<()> {
    match std::os::windows::fs::FileExt::seek_read(file, buf, at)? {
        read if read == buf.len() => Ok(()),
        _ => Err(ErrorKind::UnexpectedEof.into()),
    }
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], at: u64) -> io::Result<()> {
    match std::os::windows::fs::FileExt::seek_write(file, buf, at)? {
        written if written == buf.len() => Ok(()),
        _ => Err(ErrorKind::WriteZero.into()),
    }
}

/// The bytes of the file at `path`, where they are `len` long (any length with none); `None`
/// where they are not.
pub(crate) fn read_whole(path: &Path, len: Option<u64>) -> io::Result<Option<Vec<u8>>> {
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
pub(crate) fn write_whole(path: &Path, parts: &[&[u8]]) -> io::Result<()> {
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

/// What a render content's record holds: the `FunctionCompileKey` digest of each of its
/// functions' lowerings, `None` for a stage the descriptor gives compiled, or has not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lowerings {
    pub(crate) vertex: Option<Hash256>,
    pub(crate) fragment: Option<Hash256>,
}

/// The record of the render content of descriptor key digest `content` (the module's doc):
/// [`RECORD`], the translator's semantics and artifact format versions, `content`, each stage's
/// tag (1 for a lowering, 0 for none) and digest (zeros for none), and the [`checksum`] of all of
/// them, [`RECORD_LEN`] bytes.
pub(crate) fn record_bytes(content: &Hash256, lowerings: &Lowerings) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_LEN);
    bytes.extend_from_slice(RECORD);
    bytes.extend_from_slice(&SEMANTICS_VERSION.to_le_bytes());
    bytes.extend_from_slice(&ARTIFACT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(content);
    for stage in [lowerings.vertex, lowerings.fragment] {
        bytes.push(u8::from(stage.is_some()));
        bytes.extend_from_slice(&stage.unwrap_or_default());
    }
    let sum = checksum(&bytes);
    bytes.extend_from_slice(&sum.to_le_bytes());
    bytes
}

/// The lowerings `bytes` record for the render content of descriptor key digest `content`, where
/// they are the record [`record_bytes`] writes for it under this build's versions; none for
/// another content's record, another layout or version, or bytes cut short or changed.
pub(crate) fn read_record(bytes: &[u8], content: &Hash256) -> Option<Lowerings> {
    let (body, sum) = bytes.split_last_chunk::<8>()?;
    let (magic, rest) = body.split_first_chunk::<8>()?;
    let (semantics, rest) = rest.split_first_chunk::<4>()?;
    let (format, rest) = rest.split_first_chunk::<4>()?;
    let (key, rest) = rest.split_first_chunk::<32>()?;
    let (vertex, rest) = recorded_stage(rest)?;
    let (fragment, rest) = recorded_stage(rest)?;
    (rest.is_empty()
        && magic == RECORD
        && u32::from_le_bytes(*semantics) == SEMANTICS_VERSION
        && u32::from_le_bytes(*format) == ARTIFACT_FORMAT_VERSION
        && key == content
        && u64::from_le_bytes(*sum) == checksum(body))
    .then_some(Lowerings { vertex, fragment })
}

/// A stage of a record, its tag and digest, and the bytes after it; none for another tag, or a
/// digest that is not zeros beside tag 0.
fn recorded_stage(bytes: &[u8]) -> Option<(Option<Hash256>, &[u8])> {
    let (&tag, rest) = bytes.split_first()?;
    let (digest, rest) = rest.split_first_chunk::<32>()?;
    let stage = match tag {
        1 => Some(*digest),
        0 if *digest == [0; 32] => None,
        _ => return None,
    };
    Some((stage, rest))
}

/// A device's caches: the lowerings in the store, and the `VkPipelineCache` every `VkPipeline`
/// is made through, with its blob.
pub(crate) struct Caches {
    store: Option<Arc<Store>>,
    /// The cache directory, where there is one.
    dir: Option<PathBuf>,
    /// The device's blob in it, named once the reaper has read the device's header.
    blob: OnceLock<PathBuf>,
    /// The pipeline cache header the device's driver writes ([`usable_header`]), set before the
    /// reaper makes the cache; unset where the driver writes none.
    header: OnceLock<[u8; 32]>,
    pipelines: Mutex<Pipelines>,
    /// Signalled once the reaper made the cache.
    made: Condvar,
    /// Held shared by each creation from before it takes a cache until it ends, and exclusive by
    /// the reaper's look around its reads of the cache's size and data, which it takes by a try
    /// (the module's doc). Taken before [`Caches::pipelines`].
    creations: RwLock<()>,
    /// Creations through the cache that have not ended, those waiting for [`Caches::creations`]
    /// among them.
    creating: AtomicUsize,
    /// When the last creation ended, in nanoseconds of [`host_time`].
    created: AtomicU64,
    /// Set by a creation's end, cleared by the reaper's look at the cache.
    grown: AtomicBool,
    /// [`PIPELINE_CACHE_BOUND`], lower in a test.
    bound: AtomicU64,
    /// The lowerings, refusals and records waiting for the reaper to write them.
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
    /// Where the seed waits before it makes the blob's cache the one creations take.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) seed_hold: Hold,
    /// How long the seed waits at its start, before it makes the empty cache and sets
    /// `Gauge::PipelineCacheBytes`, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    seed_delay: Duration,
    /// Where a look waits before it tries [`Caches::creations`].
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) look_hold: Hold,
    /// Where a look waits once it holds [`Caches::creations`] exclusive, before its reads.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) alone_hold: Hold,
    /// Creations holding [`Caches::creations`] shared.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    sharing: AtomicUsize,
    /// Whether a look holds [`Caches::creations`] exclusive.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    alone: AtomicBool,
    /// Creations that held [`Caches::creations`] shared while a look held it exclusive, counted at
    /// the start or the end of either.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    overlaps: AtomicUsize,
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
thread_local! {
    /// Whether the caches made next on this thread hold their seed before it takes the blob's
    /// cache, for a test.
    static HOLD_NEXT_SEED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How long the seed of the caches made next on this thread waits at its start, for a test.
    static DELAY_NEXT_SEED: std::cell::Cell<Duration> = const { std::cell::Cell::new(Duration::ZERO) };
}

/// A point the reaper waits at while a test holds it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) struct Hold {
    held: AtomicBool,
    waiting: AtomicBool,
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Hold {
    /// A point held from the start where `held`.
    fn new(held: bool) -> Hold {
        Hold {
            held: AtomicBool::new(held),
            waiting: AtomicBool::new(false),
        }
    }

    /// Makes the reaper wait at the point when it next comes to it, until [`Hold::release`].
    pub(crate) fn hold(&self) {
        self.held.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether the reaper is waiting at the point.
    pub(crate) fn waiting(&self) -> bool {
        self.waiting.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Lets the reaper go on.
    pub(crate) fn release(&self) {
        self.held.store(false, std::sync::atomic::Ordering::Release);
    }

    /// On the reaper, at the point: waits while it is held.
    fn pass(&self) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        if !self.held.load(Acquire) {
            return;
        }
        self.waiting.store(true, Release);
        while self.held.load(Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.waiting.store(false, Release);
    }
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
    /// The caches a fresh one or the blob's replaced, each destroyed once no creation holds it.
    retired: Vec<Arc<PipelineCache>>,
    /// The size of the current cache's data when the blob was last written from it, or when it
    /// was made: what the next write must grow past.
    written: u64,
}

/// A `VkPipelineCache` of the device's, held by each creation through it.
pub(crate) struct PipelineCache(vk::PipelineCache);

/// Hands the bytes of a lowered function of key digest `key` to the reaper, which writes them to
/// the store, off every guest thread. Where it would take the lowerings waiting for it past
/// [`PENDING_BYTES`], it is dropped and counted (`Metrics::cache_writes_dropped`), never waited
/// for: a lowering is only a cache, and a later state lowers it again. One larger than the bound
/// by itself is queued where none waits.
pub(crate) fn keep_translation(core: &Core, key: &[u8], bytes: Vec<u8>) {
    keep(core, Caches::translation_file(key), bytes);
}

/// Hands the record of the render content of descriptor key digest `content`, the keys of its
/// functions' lowerings, to the reaper, as [`keep_translation`] does a lowering.
pub(crate) fn keep_record(core: &Core, content: &Hash256, lowerings: &Lowerings) {
    keep(
        core,
        Caches::record_file(content),
        record_bytes(content, lowerings),
    );
}

/// Hands `bytes` to the reaper, to write as `file` in the store ([`keep_translation`]).
fn keep(core: &Core, file: String, bytes: Vec<u8>) {
    if core.caches.store.is_none() {
        return;
    }
    let queued = lock(&core.caches.pending).push(file, bytes);
    match queued {
        true => core.reaper.poke(),
        false => core.counters.count(Metric::CacheWritesDropped),
    }
}

/// Hands the translator's refusal of the lowering of key digest `key`, and `why`, to the reaper,
/// as [`keep_translation`] does a lowering.
pub(crate) fn keep_refusal(core: &Core, key: &[u8], why: VariantRefusal) {
    let marker = match why {
        VariantRefusal::VoteWord => REFUSED_VOTE,
        VariantRefusal::Translator => REFUSED,
    };
    keep_translation(core, key, marker.to_vec());
}

/// Hands the translator's answer that a kernel observes its threadgroup partition to the reaper,
/// under the key digest `key` of its partial-threadgroups variant, as [`keep_translation`] does a
/// lowering.
pub(crate) fn keep_observer(core: &Core, key: &[u8]) {
    keep_translation(core, key, OBSERVES.to_vec());
}

/// The files waiting for the reaper to write them, in order: at most `bound` bytes of them, or one
/// file larger than that alone.
pub(crate) struct Pending {
    files: VecDeque<(String, Vec<u8>)>,
    bytes: u64,
    bound: u64,
}

impl Pending {
    /// An empty queue of at most `bound` bytes, or of one file larger than that.
    pub(crate) fn new(bound: u64) -> Pending {
        Pending {
            files: VecDeque::new(),
            bytes: 0,
            bound,
        }
    }

    /// Queues `bytes` under `file`; false, queuing nothing, where they would take a queue that
    /// holds a file past its bound. An empty queue takes a file of any size, so a file larger than
    /// the bound is kept, alone: what it holds is bytes already in memory, made before the push.
    pub(crate) fn push(&mut self, file: String, bytes: Vec<u8>) -> bool {
        let len = bytes.len() as u64;
        if !self.files.is_empty() && self.bytes.saturating_add(len) > self.bound {
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

/// A creation through the device's `VkPipelineCache`, from [`creation`] until it is dropped,
/// holding [`Caches::creations`] shared: let go last, after the cache.
pub(crate) struct Creation<'a> {
    core: &'a Core,
    cache: Option<Arc<PipelineCache>>,
    _shared: RwLockReadGuard<'a, ()>,
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
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        caches.sharing.fetch_sub(1, SeqCst);
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
/// which it does first when it starts, the calling thread waits for it, and where the reaper's
/// look holds [`Caches::creations`], for that look.
pub(crate) fn creation(core: &Core) -> Creation<'_> {
    let caches = &core.caches;
    caches.creating.fetch_add(1, SeqCst);
    let shared = caches.share(&core.counters);
    let mut pipelines = lock(&caches.pipelines);
    while !pipelines.made {
        pipelines = (caches.made.wait(pipelines)).unwrap_or_else(PoisonError::into_inner);
    }
    if pipelines.seeding {
        // Through the empty cache: what it compiles is not in this run's blob.
        core.counters.count(Metric::CreationsBeforeSeed);
    }
    let cache = pipelines.current.clone();
    drop(pipelines);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    caches.shared_by_a_creation();
    Creation {
        core,
        cache,
        _shared: shared,
    }
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
            store: dir
                .map(|dir| Store::open(dir.to_owned(), cap, counters.clone(), Versions::BUILT)),
            dir: dir.map(Path::to_owned),
            blob: OnceLock::new(),
            header: OnceLock::new(),
            pipelines: Mutex::new(Pipelines {
                made: false,
                seeding: false,
                current: None,
                retired: Vec::new(),
                written: 0,
            }),
            made: Condvar::new(),
            creations: RwLock::new(()),
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
            seed_hold: Hold::new(HOLD_NEXT_SEED.with(std::cell::Cell::take)),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            seed_delay: DELAY_NEXT_SEED.with(std::cell::Cell::take),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            look_hold: Hold::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            alone_hold: Hold::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            sharing: AtomicUsize::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            alone: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            overlaps: AtomicUsize::new(0),
        }
    }

    /// Makes the seed of the device made next on this thread wait before it takes the blob's
    /// cache, until its [`Caches::seed_hold`] is released, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold_next_seed() {
        HOLD_NEXT_SEED.with(|hold| hold.set(true));
    }

    /// Makes the seed of the device made next on this thread wait `by` at its start, before it
    /// makes the empty cache and sets `Gauge::PipelineCacheBytes`, for a test.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn delay_next_seed(by: Duration) {
        DELAY_NEXT_SEED.with(|delay| delay.set(by));
    }

    /// Creations that have not ended, those waiting for [`Caches::creations`] among them.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn creating(&self) -> usize {
        self.creating.load(SeqCst)
    }

    /// Creations that held [`Caches::creations`] shared while a look held it exclusive.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn overlaps(&self) -> usize {
        self.overlaps.load(SeqCst)
    }

    /// A creation's start, holding [`Caches::creations`] shared: an overlap where a look holds it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fn shared_by_a_creation(&self) {
        self.sharing.fetch_add(1, SeqCst);
        if self.alone.load(SeqCst) {
            self.overlaps.fetch_add(1, SeqCst);
        }
    }

    /// A look's start or end holding [`Caches::creations`] exclusive (`alone` at its start): an
    /// overlap where a creation holds it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fn held_alone(&self, alone: bool) {
        if alone {
            self.alone.store(true, SeqCst);
        }
        if self.sharing.load(SeqCst) > 0 {
            self.overlaps.fetch_add(1, SeqCst);
        }
        if !alone {
            self.alone.store(false, SeqCst);
        }
    }

    /// The device's blob, where it has a cache directory and the reaper has seeded the cache.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn blob(&self) -> Option<&Path> {
        self.blob.get().map(PathBuf::as_path)
    }

    /// Keeps the lowerings waiting to be written in their queue, or lets the reaper take them, for
    /// a test: the reaper takes none while they are stalled, so the queue holds every one handed
    /// over. A test that lets them go pokes the reaper.
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

    /// The cache directory, where the device has one.
    pub(crate) fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The pipeline cache header the device's driver writes ([`usable_header`]), once the reaper
    /// made the device's cache, which the calling thread waits for (microseconds, at the device's
    /// start); none where the driver writes none.
    pub(crate) fn header(&self) -> Option<[u8; 32]> {
        let mut pipelines = lock(&self.pipelines);
        while !pipelines.made {
            pipelines = (self.made.wait(pipelines)).unwrap_or_else(PoisonError::into_inner);
        }
        drop(pipelines);
        self.header.get().copied()
    }

    fn translation_file(key: &[u8]) -> String {
        format!("{TRANSLATIONS}/{}", hex(key))
    }

    /// The file, under a generation, of the record of the render content of descriptor key digest
    /// `content`.
    pub(crate) fn record_file(content: &Hash256) -> String {
        format!("{RENDERS}/{}", hex(content))
    }

    /// The lowerings the record of the render content of descriptor key digest `content` names,
    /// where the store holds one that reads as its record ([`read_record`]); bytes that do not are
    /// removed, a miss.
    pub(crate) fn record(&self, content: &Hash256) -> Option<Lowerings> {
        let file = Caches::record_file(content);
        (self.store.as_ref()?).get(&file, |bytes| read_record(bytes, content))
    }

    /// What `decode` makes of the bytes kept for a lowered kernel of key digest `key`, where they
    /// decode. A file that holds the translator's refusal, or its answer that the kernel observes
    /// its partition, is no lowering, and stays.
    pub(crate) fn translation<T>(
        &self,
        key: &[u8],
        decode: impl FnOnce(&[u8]) -> Option<T>,
    ) -> Option<T> {
        let file = Caches::translation_file(key);
        let kept = self.store.as_ref()?.get(&file, |bytes| match bytes {
            REFUSED | REFUSED_VOTE | OBSERVES => Some(None),
            lowering => decode(lowering).map(Some),
        });
        kept.flatten()
    }

    /// The next lowering, refusal or record waiting for the reaper to write it; none while a test
    /// stalls the writes, which then stay in the queue.
    pub(crate) fn next_write(&self) -> Option<(String, Vec<u8>)> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if self
            .writes_stalled
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return None;
        }
        lock(&self.pending).pop()
    }

    /// On the reaper: writes `bytes` under `file` in the store, as [`keep_translation`] queued
    /// them.
    pub(crate) fn write_kept(&self, file: &str, bytes: &[u8], counters: &Counters) {
        if let Some(store) = &self.store {
            if store.put(file, bytes) {
                counters.cache_written(bytes.len() as u64);
            }
        }
    }

    /// Why the translator refused the lowering of key digest `key` on a device before, where it
    /// did.
    pub(crate) fn refused(&self, key: &[u8]) -> Option<VariantRefusal> {
        let store = self.store.as_ref()?;
        let file = Caches::translation_file(key);
        if store.holds(&file, REFUSED_VOTE) {
            return Some(VariantRefusal::VoteWord);
        }
        store
            .holds(&file, REFUSED)
            .then_some(VariantRefusal::Translator)
    }

    /// Whether the translator answered on a device before that the kernel whose
    /// partial-threadgroups variant has key digest `key` observes its threadgroup partition.
    pub(crate) fn observed(&self, key: &[u8]) -> bool {
        self.store
            .as_ref()
            .is_some_and(|store| store.holds(&Caches::translation_file(key), OBSERVES))
    }

    /// On the reaper, first: makes the device's `VkPipelineCache` empty, which creations waiting
    /// for a cache take at once; then, where the device has a blob that checks, whose data begins
    /// with the header the device's driver writes and that the driver takes, makes a cache from
    /// it and makes that the cache creations take from then on. A blob that is not taken is
    /// removed. So no creation waits for the blob's read or the driver's parse of it, which for a
    /// blob of 32 MiB take tens of milliseconds: a creation meanwhile goes through the empty
    /// cache, and what it compiles there is not in this run's blob.
    ///
    /// The blob is named by the whole header in hex ([`blob_name`]); before its read, the blobs
    /// past the [`BLOBS_KEPT`] most recently written are removed. A device whose cache data
    /// begins with no header ([`usable_header`]) keeps no blob: it reads none and writes none.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    pub(crate) unsafe fn seed(&self, device: &ash::Device, counters: &Counters) {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        std::thread::sleep(self.seed_delay);
        // SAFETY: no data.
        let empty = unsafe { make_cache(device, &[]) };
        // Read before any creation takes the cache: creations wait for `made`.
        let header = empty.and_then(|empty| {
            // SAFETY: the cache is the device's.
            let data = unsafe { device.get_pipeline_cache_data(empty) }.ok()?;
            usable_header(&data)
        });
        // SAFETY: the cache is the device's.
        let size = empty.map_or(0, |cache| unsafe { data_size(device, cache) });
        counters.set(Gauge::PipelineCacheBytes, size);
        if let Some(header) = header {
            let _ = self.header.set(header);
        }
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
        // Where no blob was taken, the empty cache stays the one creations take.
        lock(&self.pipelines).seeding = false;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.seeded
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Makes `seeded`, made from the blob, the cache creations take; the empty cache retires, and
    /// is destroyed once no creation holds it, with what creations compiled in it. `seeded`'s size
    /// is what the next write must grow past: the blob's.
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
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.seed_hold.pass();
        // SAFETY: the cache is the device's, and no creation takes it yet.
        let written = unsafe { data_size(device, seeded) };
        let mut pipelines = lock(&self.pipelines);
        let empty = pipelines.current.replace(Arc::new(PipelineCache(seeded)));
        pipelines.retired.extend(empty);
        pipelines.written = written;
        pipelines.seeding = false;
        drop(pipelines);
        counters.set(Gauge::PipelineCacheBytes, written);
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

    /// [`Caches::creations`] held shared, for a creation: at once where no look holds it, else
    /// once the look lets it go, the wait counted (`Metrics::pipeline_cache_lock_wait_nanos`).
    fn share(&self, counters: &Counters) -> RwLockReadGuard<'_, ()> {
        match self.creations.try_read() {
            Ok(shared) => shared,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                let start = host_time();
                let shared = (self.creations.read()).unwrap_or_else(PoisonError::into_inner);
                counters.pipeline_cache_lock_wait(host_time().saturating_sub(start));
                shared
            }
        }
    }

    /// [`Caches::creations`] held exclusive, for a look, where no creation holds it: taken by a
    /// try, so the reaper never waits for a creation.
    fn alone(&self) -> Option<RwLockWriteGuard<'_, ()>> {
        match self.creations.try_write() {
            Ok(alone) => Some(alone),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// On the reaper, creation quiet: where no creation began since, holds [`Caches::creations`]
    /// exclusive while it reads the cache's size, and past the bound starts a fresh cache, else
    /// reads its data where it grew by more than an eighth since the last write; then lets
    /// creations go on, writes the blob, and destroys each replaced cache no creation holds. Where
    /// a creation began since creation was quiet, the look is skipped and counted
    /// (`Metrics::pipeline_cache_looks_skipped`), and the reaper looks again once creation is quiet
    /// again.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    pub(crate) unsafe fn look(&self, device: &ash::Device, counters: &Counters) {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.look_hold.pass();
        let Some(alone) = self.alone() else {
            counters.pipeline_cache_look_skipped();
            return;
        };
        self.grown.store(false, SeqCst);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.held_alone(true);
        // SAFETY: the caller's contract; no creation runs while `alone` is held.
        let write = unsafe { self.look_alone(device, counters) };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.held_alone(false);
        drop(alone);
        if let Some((path, data, size)) = write {
            self.write(path, &data, size, counters);
        }
        // SAFETY: the caller's contract.
        unsafe { self.reap(device) };
        // Counted once the look's effects are in place, so a test that waits for it sees them.
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.looks
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// The work of [`Caches::look`] while it holds [`Caches::creations`] exclusive: the blob, the
    /// cache's data and its size where the look writes them.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches, and no creation runs.
    unsafe fn look_alone(
        &self,
        device: &ash::Device,
        counters: &Counters,
    ) -> Option<(&Path, Vec<u8>, u64)> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.alone_hold.pass();
        let current = lock(&self.pipelines).current.clone()?;
        // SAFETY: the cache is the device's.
        let size = unsafe { data_size(device, current.0) };
        counters.set(Gauge::PipelineCacheBytes, size);
        if size > self.bound.load(Relaxed) {
            // SAFETY: no data.
            let fresh = unsafe { make_cache(device, &[]) }?;
            // SAFETY: the cache is the device's.
            let fresh_size = unsafe { data_size(device, fresh) };
            let mut pipelines = lock(&self.pipelines);
            let old = pipelines.current.replace(Arc::new(PipelineCache(fresh)));
            pipelines.retired.extend(old);
            pipelines.written = fresh_size;
            drop(pipelines);
            counters.set(Gauge::PipelineCacheBytes, fresh_size);
            counters.count(Metric::PipelineCachesReplaced);
            return None;
        }
        let written = lock(&self.pipelines).written;
        let path = self
            .blob_to_write()
            .filter(|_| size > written + written / 8)?;
        // SAFETY: the cache is the device's.
        let data = unsafe { device.get_pipeline_cache_data(current.0) }.ok()?;
        Some((path, data, size))
    }

    /// Destroys each replaced cache that no creation holds.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches.
    unsafe fn reap(&self, device: &ash::Device) {
        let mut pipelines = lock(&self.pipelines);
        // A replaced cache is never handed out again: once the list holds its last reference,
        // no creation holds it.
        let (idle, held) =
            (pipelines.retired.drain(..)).partition(|old| Arc::strong_count(old) == 1);
        pipelines.retired = held;
        drop(pipelines);
        // SAFETY: the caches are the device's, and no creation holds them.
        unsafe { destroy(device, idle) };
    }

    /// The device's blob, where it keeps one and its directory is writable.
    fn blob_to_write(&self) -> Option<&Path> {
        let path = self.blob.get()?;
        let store = self.store.as_ref()?;
        store.writable().then_some(path.as_path())
    }

    /// Writes `data`, the cache's data the driver gave at size `size`, over the device's blob at
    /// `path`, where it is within the bound; past it, the write is skipped and counted.
    fn write(&self, path: &Path, data: &[u8], size: u64, counters: &Counters) {
        let len = data.len() as u64;
        if len > self.bound.load(Relaxed) {
            counters.count(Metric::PipelineCacheWritesSkipped);
            return;
        }
        if write_whole(path, &[&frame(data), data]).is_ok() {
            counters.cache_written(24 + len);
            lock(&self.pipelines).written = size;
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            self.writes
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// At the device's teardown, once nothing writes to the store: waits for its removal thread.
    pub(crate) fn join_removal(&self) {
        if let Some(store) = &self.store {
            store.join_removal();
        }
    }

    /// At the device's drop, no creation running: writes the blob where the cache's data changed
    /// since the last write; then destroys every cache.
    ///
    /// # Safety
    ///
    /// `device` is the device of the caches, and no creation runs.
    pub(crate) unsafe fn close(&self, device: &ash::Device, counters: &Counters) {
        let (current, written) = {
            let mut pipelines = lock(&self.pipelines);
            (pipelines.current.take(), pipelines.written)
        };
        if let Some(current) = &current {
            // SAFETY: the cache is the device's.
            let size = unsafe { data_size(device, current.0) };
            if let Some(path) = self.blob_to_write().filter(|_| size != written) {
                // SAFETY: the cache is the device's.
                if let Ok(data) = unsafe { device.get_pipeline_cache_data(current.0) } {
                    self.write(path, &data, size, counters);
                }
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
