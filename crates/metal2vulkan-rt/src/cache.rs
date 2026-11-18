//! The device's caches on disk, under [`DeviceOptions::pipeline_cache_dir`]: a kernel lowered and
//! a `VkPipeline` compiled in one run cost little in the next. Compute and render pipelines.
//!
//! Two kinds of file, each under the generation it was last used in (below):
//!
//! - `translations/`: one file per lowered function (a kernel, or a render pipeline's vertex or
//!   fragment function), named by the hex of its `FunctionCompileKey`, which covers everything
//!   that changes the lowering's output: the function's AIR, entry, constants and linked
//!   functions, its lowering inputs, the device profile's fingerprint and the translator's
//!   semantics version. The bytes are the translator's self-checking encoding
//!   (`CompiledFunction::to_bytes`), and bytes that do not decode for the key are a miss. A
//!   partial-threadgroups variant the translator refused is kept too, as a marker.
//! - `pipelines/<pipelineCacheUUID>/`: one file per `VkPipeline`, named by the hex of a SHA-256
//!   over everything its creation takes ([`pipeline_key`]): the `VkPipelineCache` data of a cache
//!   that made only that pipeline, framed with its length and checksum. A `VkPipeline` is made
//!   through a `VkPipelineCache` of its own, started from its file where there is one; the reaper
//!   then reads the cache's data and writes it as the file, unless the data is no longer than
//!   the file's (the pipeline came from it), and destroys the cache. At most [`PENDING`] caches
//!   wait for the reaper: past it a pipeline's cache is destroyed at once and its file not
//!   written, a miss at its next creation.
//!
//! The files live in two generations, `g<N>/` (young) and `g<N-1>/` (old), which the consumer's
//! cap ([`DeviceOptions::pipeline_cache_cap`]) bounds at half of it each. A lookup tries young,
//! then old; a hit in old is renamed into young, which is its mark of use. A write goes into
//! young. The bytes young holds are counted in `state`, 24 bytes beside the generations: young's
//! number, its bytes and a checksum of both, read and written under a lock on the file, which
//! every store on the directory, in this process or another, holds open for its life. The write
//! that takes young past half the cap makes the next generation young: young becomes old, and the
//! old one is dead. A store keeps young's number as it last read or wrote the state, so a lookup
//! reads no state: a move another store made is seen at this store's next write, and until then
//! its lookups try the generations it knew, a miss at worst. A thread of the store's own removes
//! the dead generation and ends, so no thread that submits, waits for or completes work removes
//! files. So a file stays at least half the cap and at most the cap of other writes and moves
//! after its last use, opening reads one file whatever the store holds, and nothing per file is
//! kept in memory.
//!
//! Every write is a temporary file renamed over its name, counted before it is written and given
//! back where it fails, so the count is never below what young holds. Every file checks itself
//! (the translator's encoding, the pipeline data's checksum): a file cut short, by a crash or a
//! power loss, is a miss and is removed, and nothing unchecked reaches the driver. A failed read
//! or write is a miss too: the caches never fail a call. A state file that does not check, or is
//! missing, is rebuilt from the generations' names, with young counted full, so the next write
//! moves on and the removal takes the rest. A removal cut short by the process's end is found by
//! the next open, which looks for the generation below old.
//!
//! [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
//! [`DeviceOptions::pipeline_cache_cap`]: crate::DeviceOptions::pipeline_cache_cap

use crate::counters::Counters;
use crate::device::lock;
use ash::vk;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::thread;

/// The cap of a cache directory where the consumer names none: half of it, what the store is sure
/// to keep, is about four times the 15.4 MB a long recorded session's pipelines and lowerings
/// fill (379 pipeline files, 6.25 MB; 282 lowerings, 9.16 MB).
pub(crate) const DEFAULT_CAP: u64 = 128 << 20;
/// The most pipeline caches waiting for the reaper to write their data.
pub(crate) const PENDING: usize = 64;
/// The first bytes of a pipeline data file, before its length and its checksum.
const MAGIC: &[u8; 8] = b"m2vpipe1";
/// What a translation file holds where the translator refused the lowering: bytes no lowering
/// encodes to, and that a file cut short is not.
const REFUSED: &[u8] = b"m2v-refused";
/// The translations' directory, under a generation.
const TRANSLATIONS: &str = "translations";
/// The pipelines' directory, under a generation: one directory per `pipelineCacheUUID`.
const PIPELINES: &str = "pipelines";
/// The state file, under the cache directory.
const STATE: &str = "state";

/// Young's number and the bytes counted in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct State {
    pub(crate) young: u64,
    pub(crate) bytes: u64,
}

/// The files of a cache directory, `translations/<name>` and `pipelines/<uuid>/<name>` under
/// two generations of at most half of `cap` bytes each (the module's doc).
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
    /// The times the store listed the cache directory.
    #[cfg(test)]
    pub(crate) listings: AtomicUsize,
    /// The times the store read the state file outside its lock.
    #[cfg(test)]
    pub(crate) state_reads: AtomicUsize,
}

impl Store {
    /// The store in `root`, made where it is missing; it starts the removal of a generation a
    /// removal before left.
    pub(crate) fn open(root: PathBuf, cap: u64) -> Arc<Store> {
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
            #[cfg(test)]
            listings: AtomicUsize::new(0),
            #[cfg(test)]
            state_reads: AtomicUsize::new(0),
        });
        if let Some(state) = store.current() {
            store.young.fetch_max(state.young, Relaxed);
            if state.young >= 2 && store.generation(state.young - 2).exists() {
                store.remove_dead();
            }
        }
        store
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
            .place(young, file, |path| write_whole(path, bytes))
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
            thread::sleep(std::time::Duration::from_millis(1));
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

    /// Starts the removal thread, where none runs: it removes every generation below old.
    fn remove_dead(self: &Arc<Self>) {
        if !self.writable || self.removing.swap(true, SeqCst) {
            return;
        }
        let store = Arc::clone(self);
        let started = thread::Builder::new()
            .name("m2v-store-remove".to_string())
            .spawn(move || store.remove());
        if started.is_err() {
            self.removing.store(false, SeqCst);
        }
    }

    /// The removal thread: makes the state durable, then removes the dead generations, until a
    /// pass finds none it did not try.
    fn remove(&self) {
        loop {
            if let Some(file) = &self.state {
                let _ = file.sync_all();
            }
            let tried = self.dead();
            for &number in &tried {
                let _ = fs::remove_dir_all(self.generation(number));
            }
            self.removing.store(false, SeqCst);
            // A move to the next generation during the pass found the thread running and started
            // none: its dead generation is taken here. One the pass could not remove waits for
            // the next move or open.
            let new = self.dead().iter().any(|number| !tried.contains(number));
            if !new || self.removing.swap(true, SeqCst) {
                return;
            }
        }
    }

    /// The generations below old.
    fn dead(&self) -> Vec<u64> {
        let Some(young) = self.current().map(|state| state.young) else {
            return Vec::new();
        };
        (generations(&self.root).into_iter())
            .filter(|&number| number + 1 < young)
            .collect()
    }
}

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

/// Writes `bytes` to `path` through a temporary file beside it, named for this process and this
/// write, then renamed over it.
fn write_whole(path: &Path, bytes: &[u8]) -> io::Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Relaxed)
    ));
    let temp = PathBuf::from(temp);
    let written = File::create(&temp).and_then(|mut file| file.write_all(bytes));
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
    store: Option<Arc<Store>>,
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
        let store = dir.map(|dir| Store::open(dir.to_owned(), cap));
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
        let file = self.pipeline_file(key);
        (self.store.as_ref()?).get(&file, |bytes| unframe(bytes).map(<[u8]>::to_vec))
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
