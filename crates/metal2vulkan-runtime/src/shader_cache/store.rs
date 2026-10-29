//! The cache directory and its entries: where they live, how one is read and written, and the
//! scans that keep the directory under its cap.
//!
//! ```text
//! <root>/v1/<build>/<first byte of name, hex>/<name, hex>
//! ```
//!
//! - `<build>` names the build ([`Directory::build`]); a build reads only its own directory.
//! - One directory beside the builds', the content directory ([`content`]), holds records no build
//!   owns, named by the content they describe: the driver's pipeline cache data (It was the
//!   build's; one record per pipeline making), named by what each making reads and read as an entry
//!   is, with its counts (`driver.rs`), and the libraries pipeline records name, by their content
//!   hash (`libraries.rs`); the scans bound them with the rest.
//! - An entry's name is the SHA-256 of a domain tag, the build, the entry's kind, its key and the
//!   profile fingerprint ([`entry_name`]). The decoder the caller hands to [`Store::load`] checks
//!   the contents against the request byte for byte; contents it refuses are removed.
//! - A writer writes a temporary file beside the entry (named for the entry, the process and a
//!   sequence number), flushes it and renames it over the entry, so a reader never sees part of
//!   one. A temporary file a crashed writer left is removed by the next scan that finds it old.
//! - A device's writes go to its cache thread ([`Store::write_soon`], `worker.rs`): the same
//!   bytes, the same temporary file and the same `rename`, off the creating thread. The queue is
//!   bounded, an entry queued is visible to this device's lookups at once, and what is queued when
//!   the device goes is written before its drop returns (`worker.rs` states the bound and what
//!   happens at it). A store with no cache thread (the tests') writes on the calling thread.
//! - **The directory's time is counted**, in three figures the asking thread's are apart from the
//!   cache thread's: the reads ([`Counters::read_nanos`]), the writes a thread that asked made
//!   itself ([`Counters::write_nanos`]), and everything the device's cache thread did here
//!   ([`Counters::off_thread_nanos`]). Each is read from the host's clock around the directory's
//!   calls alone, so a lookup the write queue answered, and the encode and decode of an entry's
//!   bytes, are in none of them (`WorkCounts::shader_cache_read_nanos`).
//! - Every scan that finds the directory past its cap removes the least recently used entries,
//!   by their access time, which a hit refreshes, down to [`LOW_WATER`] of the cap. At device
//!   creation the scan also removes other builds' directories that nothing used for
//!   [`UNUSED_BUILD`].

use super::worker::{Asked, Write as Queued, QUEUED_BYTES};
use super::EntryKind;
use crate::{DebugMessage, Degradation, DegradeNotice, MessageKind, MessageSeverity, MessageSink};
use metal2vulkan::cache::Hash256;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// The layout version of the directory's contents.
pub(crate) const LAYOUT: &str = "v1";
/// The largest entry read or written: no artifact is near it, and a larger file is not one.
const MAX_ENTRY: u64 = 256 << 20;
/// A temporary file older than this was left by a writer that died, and a scan removes it.
const STALE_TEMPORARY: Duration = Duration::from_secs(60 * 60);
/// Another build's directory nothing used for this long is removed by the scan at creation.
pub(crate) const UNUSED_BUILD: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// A scan that finds the directory past its cap removes entries down to this share of it (in
/// percent), so the next few writes do not each start another scan.
const LOW_WATER: u64 = 90;

/// The domain tag of an entry's name.
const NAME_TAG: &[u8] = b"metal2vulkan shader cache entry\0";

/// The domain tag of the content directory's name.
const CONTENT_TAG: &[u8] = b"metal2vulkan shader cache content\0";

/// The name the content directory has among the builds' (the module docs): a build's name is a
/// SHA-256 too, so the scans take it as one, and remove it once nothing in it was used for
/// [`UNUSED_BUILD`].
pub(crate) fn content() -> Hash256 {
    Sha256::digest(CONTENT_TAG).into()
}

/// An opened cache directory: its build's directory exists and takes writes.
#[derive(Clone, Debug)]
pub(crate) struct Directory {
    root: PathBuf,
    build: Hash256,
    capacity: u64,
}

impl Directory {
    /// Open `root` for `build` with a cap of `capacity` bytes: create it and the build's
    /// directory (owner-only) and write and remove a probe file there. The error names the
    /// directory that failed.
    pub(crate) fn open(root: PathBuf, build: Hash256, capacity: u64) -> Result<Directory, Failed> {
        let directory = Directory {
            root,
            build,
            capacity,
        };
        let failed = |path: &Path| {
            let path = path.to_path_buf();
            move |error| Failed { path, error }
        };
        create_private_dir_all(&directory.root).map_err(failed(&directory.root))?;
        let build_dir = directory.build_dir();
        create_private_dir_all(&build_dir).map_err(failed(&build_dir))?;
        let probe = build_dir.join(format!(".probe.{}.tmp", unique()));
        let written = new_private_file(&probe).and_then(|mut file| file.write_all(b"probe"));
        let removed = fs::remove_file(&probe);
        written.and(removed).map_err(failed(&directory.root))?;
        Ok(directory)
    }

    /// The directory the consumer named or the platform's default.
    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    /// The build this directory serves.
    pub(super) fn build(&self) -> &Hash256 {
        &self.build
    }

    /// `<root>/v1`.
    fn layout_dir(&self) -> PathBuf {
        self.root.join(LAYOUT)
    }

    /// `<root>/v1/<build>`.
    fn build_dir(&self) -> PathBuf {
        self.layout_dir().join(hex(&self.build))
    }

    /// The path of the entry named `name`.
    pub(crate) fn entry_path(&self, name: &str) -> PathBuf {
        self.build_dir().join(&name[..2]).join(name)
    }

    /// The path of the record named `name` in the content directory.
    pub(crate) fn record_path(&self, name: &str) -> PathBuf {
        self.layout_dir()
            .join(hex(&content()))
            .join(&name[..2])
            .join(name)
    }
}

/// A directory that cannot be used, and why.
#[derive(Debug)]
pub(crate) struct Failed {
    pub(crate) path: PathBuf,
    pub(crate) error: io::Error,
}

/// The name of the entry of `kind` under `key` for `profile`, in `build`: the hex SHA-256 of
/// all of them after a domain tag.
pub(crate) fn entry_name(
    build: &Hash256,
    kind: EntryKind,
    key: &Hash256,
    profile: &Hash256,
) -> String {
    let mut hash = Sha256::new();
    hash.update(NAME_TAG);
    hash.update(build);
    hash.update((kind as u32).to_le_bytes());
    hash.update(key);
    hash.update(profile);
    hex(&hash.finalize().into())
}

/// `bytes` in lowercase hex.
pub(crate) fn hex(bytes: &Hash256) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// What a lookup of an entry gave. Every outcome but a hit is a miss.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Lookup<T> {
    /// The entry's contents, as the decoder accepted them.
    Hit(T),
    /// No entry of that name.
    Absent,
    /// The entry could not be read.
    Unreadable,
    /// The decoder refused the entry's contents: the entry was removed.
    Rejected,
}

/// What a read of an entry found.
enum Found {
    /// No entry of that name.
    Absent,
    /// The entry could not be read.
    Failed,
    /// The entry's bytes, and the open file (to refresh its access time on a hit).
    Bytes(Vec<u8>, fs::File),
    /// The entry's bytes, from a write the cache thread has not made yet: no file to refresh.
    Queued(Vec<u8>),
}

/// The counts behind [`ShaderCacheStatistics`](super::ShaderCacheStatistics).
#[derive(Default)]
pub(crate) struct Counters {
    pub(crate) hits: AtomicU64,
    pub(crate) misses: AtomicU64,
    /// Of `hits`, by the kind of artifact looked up, in [`Looked`]'s order.
    pub(crate) hits_by: [AtomicU64; Looked::COUNT],
    /// Of `misses`, likewise.
    pub(crate) misses_by: [AtomicU64; Looked::COUNT],
    /// Libraries pipeline records name that this device wrote to the content directory
    /// (`libraries.rs`): `WarmStatistics::libraries_kept`.
    pub(crate) libraries_kept: AtomicU64,
    pub(crate) rejected: AtomicU64,
    pub(crate) read_failures: AtomicU64,
    pub(crate) writes: AtomicU64,
    pub(crate) written_bytes: AtomicU64,
    pub(crate) writes_queued: AtomicU64,
    pub(crate) writes_on_thread: AtomicU64,
    pub(crate) queued_hits: AtomicU64,
    pub(crate) write_failures: AtomicU64,
    pub(crate) evictions: AtomicU64,
    pub(crate) driver_loads: AtomicU64,
    pub(crate) driver_pipeline_hits: AtomicU64,
    pub(crate) driver_pipeline_misses: AtomicU64,
    pub(crate) driver_saves: AtomicU64,
    pub(crate) driver_data_truncated: AtomicU64,
    pub(crate) driver_data_mismatched: AtomicU64,
    pub(crate) driver_data_unchanged: AtomicU64,
    pub(crate) driver_data_found: AtomicU64,
    /// Makings handed a `VkPipelineCache` the device already held for their record, made by an
    /// earlier making of the same name (`driver.rs`): each one a stage's conversion and Metal's
    /// compile of it not run again. Not in [`ShaderCacheStatistics`]; the device tests of the
    /// sharing read it here.
    ///
    /// [`ShaderCacheStatistics`]: super::ShaderCacheStatistics
    pub(crate) driver_caches_shared: AtomicU64,
    /// Questions about an entry ([`Store::holds`]) that reached the filesystem: one `stat` each.
    /// A question the write queue answered is not one, since it reached no directory. Not in
    /// [`ShaderCacheStatistics`]; the device tests of a creation's directory round trips read it
    /// here, because a count of round trips is a fact of the structure where
    /// [`read_nanos`](Self::read_nanos) is a figure of the host.
    ///
    /// [`ShaderCacheStatistics`]: super::ShaderCacheStatistics
    pub(crate) holds_questions: AtomicU64,
    /// Reads of an entry's bytes ([`Store::read`]) that reached the filesystem: one `open` each,
    /// whatever it found. One the write queue answered is not one. Counted and read as
    /// [`holds_questions`](Self::holds_questions) is, and apart from it because the two are
    /// different calls at different sites: a question is a `stat` of a name a creation is about
    /// to decide by, a read is an `open` of bytes it wants.
    ///
    /// [`ShaderCacheStatistics`]: super::ShaderCacheStatistics
    pub(crate) entry_reads: AtomicU64,
    /// Nanoseconds the directory's reads took on the threads that asked for an artifact: the
    /// questions about an entry ([`Store::holds`]), the reads of one ([`Store::load`],
    /// [`Store::load_content`]), the refresh of a hit's access time, and the removal of an entry
    /// a decoder refused or that is longer than the directory takes. Not in
    /// [`ShaderCacheStatistics`]; `WorkCounts::shader_cache_read_nanos` reports it.
    ///
    /// [`ShaderCacheStatistics`]: super::ShaderCacheStatistics
    pub(crate) read_nanos: AtomicU64,
    /// Nanoseconds the directory's writes took on the threads that asked for an artifact: the
    /// `mkdir`, the temporary file, its write and flush and the `rename` of each entry a thread
    /// wrote itself because the queue was at a bound or the store has no cache thread
    /// (`writes_on_thread`). Reported as `WorkCounts::shader_cache_write_nanos`.
    pub(crate) write_nanos: AtomicU64,
    /// Nanoseconds the device's cache thread spent in the directory: the writes it took from the
    /// queue (`writes_queued`), and the scans and removals that bound the directory. None of it
    /// is on a creation's path. Reported as `WorkCounts::shader_cache_off_thread_nanos`.
    pub(crate) off_thread_nanos: AtomicU64,
}

/// The kinds of artifact a lookup asks the directory for, which [`Counters::hits_by`] and
/// [`Counters::misses_by`] count apart: one entry kind each, and the three a pipeline state's
/// creation looks up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Looked {
    /// A translated function ([`EntryKind::Function`]).
    Function = 0,
    /// A compiled compute pipeline ([`EntryKind::ComputePipeline`]).
    ComputePipeline = 1,
    /// A compiled render pipeline ([`EntryKind::RenderPipeline`]).
    RenderPipeline = 2,
}

impl Looked {
    /// How many there are.
    pub(crate) const COUNT: usize = 3;

    /// The kind of entry it is looked up as.
    pub(crate) fn kind(self) -> EntryKind {
        match self {
            Looked::Function => EntryKind::Function,
            Looked::ComputePipeline => EntryKind::ComputePipeline,
            Looked::RenderPipeline => EntryKind::RenderPipeline,
        }
    }
}

impl Counters {
    /// Count driver pipeline cache data refused for `refusal`.
    pub(crate) fn refused(&self, refusal: super::driver::Refusal) {
        count(match refusal {
            super::driver::Refusal::Truncated => &self.driver_data_truncated,
            super::driver::Refusal::Mismatched => &self.driver_data_mismatched,
        });
    }
}

/// Run `f`, the directory's calls and nothing else, with the nanoseconds they took on this thread
/// added to `counter` ([`Counters::read_nanos`] and its two fellows). Two clock reads, 27 ns
/// together on an M3 Max, against the microseconds of the filesystem call they bracket.
fn timed<R>(counter: &AtomicU64, f: impl FnOnce() -> R) -> R {
    let entered = crate::gpu_time::host_nanos();
    let done = f();
    let taken = crate::gpu_time::host_nanos().saturating_sub(entered);
    counter.fetch_add(taken, Ordering::Relaxed);
    done
}

/// Add one to `counter`.
pub(super) fn count(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// One device's view of the directory: reads, writes, and the scans that bound it.
pub(crate) struct Store {
    directory: Directory,
    pub(crate) counters: Counters,
    /// The bytes this process takes the directory to hold: the last scan's total plus what it
    /// wrote since. Other processes' writes show at the next scan.
    used: AtomicU64,
    /// A scan has set `used`; before that the cap is not judged.
    scanned: AtomicBool,
    /// The warnings said once each ([`Warnings`]).
    warnings: Warnings,
    /// The device's cache thread, which writes the entries this store is given; `None` in a
    /// store with no thread, which writes them on the calling thread.
    asked: Option<Arc<Asked>>,
}

/// What a device's store says once each, through its instance's message sink, else on stderr:
/// its first removal of an entry past the cap ([`EVICTED`]), and its first entry not written
/// (the degrade notice [`Degradation::ShaderCacheEntryNotWritten`]); later ones are counted
/// alone.
struct Warnings {
    sink: Option<MessageSink>,
    evicted: AtomicBool,
    write_failed: AtomicBool,
}

/// The id name of the warning of a store's first removal past its cap.
const EVICTED: &str = "metal2vulkan-shader-cache-evicted";

impl Store {
    /// The store of `directory`, whose warnings go to `sink`, else on stderr, and whose writes
    /// go to the cache thread `asked` asks (`worker.rs`); with no thread they are written on the
    /// calling thread.
    pub(crate) fn new(
        directory: Directory,
        sink: Option<MessageSink>,
        asked: Option<Arc<Asked>>,
    ) -> Store {
        Store {
            directory,
            counters: Counters::default(),
            used: AtomicU64::new(0),
            scanned: AtomicBool::new(false),
            warnings: Warnings {
                sink,
                evicted: AtomicBool::new(false),
                write_failed: AtomicBool::new(false),
            },
            asked,
        }
    }

    /// Count an entry not written (`ShaderCacheStatistics::write_failures`); the first is
    /// announced as [`Degradation::ShaderCacheEntryNotWritten`], naming the directory and its
    /// cap. Never an error: the work the entry would have saved is the directory's regardless.
    pub(crate) fn write_failed(&self) {
        count(&self.counters.write_failures);
        if !self.warnings.write_failed.swap(true, Ordering::Relaxed) {
            let degradation = Degradation::ShaderCacheEntryNotWritten;
            let root = self.directory.root().display();
            let reason = format!(
                "the directory, a temporary file in it or the rename over the entry failed (cap \
                 {} bytes, an entry at most {} bytes). This device's later failures are counted \
                 alone (`ShaderCacheStatistics::write_failures`).",
                self.directory.capacity,
                self.entry_limit(),
            );
            self.say(
                degradation.code(),
                Some(DegradeNotice {
                    degradation,
                    object: format!("the shader cache directory {root}"),
                    reason: reason.clone(),
                    cost: degradation.cost(),
                    object_id: None,
                    misplaced: None,
                    call: None,
                }),
                format!(
                    "metal2vulkan: an entry of the shader cache directory {root} was not \
                     written: {reason} {}",
                    degradation.cost(),
                ),
            );
        }
    }

    /// Say `text` as the notice `id_name`, carrying `degradation` where it announces one: through
    /// the
    /// sink, else on stderr. Called on a program's thread or on the cache thread; the sink runs
    /// the program's code inside the exit guard's `program` either way.
    fn say(&self, id_name: &str, degradation: Option<DegradeNotice>, text: String) {
        let message = DebugMessage {
            severity: MessageSeverity::Warning,
            kind: MessageKind {
                general: true,
                ..MessageKind::default()
            },
            id_name: id_name.to_owned(),
            id: 0,
            text,
            degradation,
        };
        match &self.warnings.sink {
            Some(sink) => sink.deliver(&message),
            None => eprintln!("{}", message.text),
        }
    }

    pub(crate) fn directory(&self) -> &Directory {
        &self.directory
    }

    /// The largest entry this directory takes.
    fn entry_limit(&self) -> u64 {
        MAX_ENTRY.min(self.directory.capacity)
    }

    /// Whether the entry named `name` is there to be read: its file is in place and of a length
    /// this directory takes, or its write is queued for the cache thread, whose rename has not
    /// happened yet. Reads no contents and counts in no hit or miss -- it is a question about the
    /// directory, not a lookup of an entry -- but its `stat` is in [`Counters::read_nanos`].
    pub(crate) fn holds(&self, name: &str) -> bool {
        self.holds_at(&self.directory.entry_path(name))
    }

    /// [`Store::holds`] of the record named `name` in the content directory: a library a pipeline
    /// record names (`shader_cache/libraries.rs`).
    pub(crate) fn holds_record(&self, name: &str) -> bool {
        self.holds_at(&self.directory.record_path(name))
    }

    /// [`Store::holds`] of the file at `path`.
    fn holds_at(&self, path: &Path) -> bool {
        if self.asked.as_ref().is_some_and(|asked| asked.holds(path)) {
            return true;
        }
        count(&self.counters.holds_questions);
        timed(&self.counters.read_nanos, || {
            fs::metadata(path).is_ok_and(|metadata| metadata.len() <= self.entry_limit())
        })
    }

    /// The entries this store has removed from the directory: the ones it evicted past the cap
    /// and the ones a decoder refused. A removal can take the very file a question about an entry
    /// answered `true` for, so an answer kept under a function's key is kept only while this
    /// figure stands still (`held.rs`).
    pub(crate) fn removals(&self) -> u64 {
        self.counters.evictions.load(Ordering::Relaxed)
            + self.counters.rejected.load(Ordering::Relaxed)
    }

    /// The entry named `name`, as `decode` accepts it. A hit refreshes the entry's access
    /// time; contents `decode` refuses are removed. Counts unreadable and refused entries.
    pub(crate) fn load<T, E>(
        &self,
        name: &str,
        decode: impl FnOnce(&[u8]) -> Result<T, E>,
    ) -> Lookup<T> {
        self.load_at(&self.directory.entry_path(name), decode)
    }

    /// The entry named `name` in the content directory, read and counted as [`Store::load`]
    /// reads and counts a build's entry: a making's pipeline cache data, which no build owns.
    /// `decode` takes the bytes read, which the value may keep.
    pub(crate) fn load_content<T>(
        &self,
        name: &str,
        decode: impl FnOnce(Vec<u8>) -> Option<T>,
    ) -> Lookup<T> {
        let path = self.directory.record_path(name);
        let (bytes, file) = match self.read(&path) {
            Found::Absent => return Lookup::Absent,
            Found::Failed => {
                count(&self.counters.read_failures);
                return Lookup::Unreadable;
            }
            Found::Bytes(bytes, file) => (bytes, Some(file)),
            Found::Queued(bytes) => (bytes, None),
        };
        match decode(bytes) {
            Some(value) => {
                self.accepted(file.as_ref());
                Lookup::Hit(value)
            }
            None => {
                self.refused(&path);
                Lookup::Rejected
            }
        }
    }

    /// [`Store::load`] of the file at `path`.
    fn load_at<T, E>(&self, path: &Path, decode: impl FnOnce(&[u8]) -> Result<T, E>) -> Lookup<T> {
        let (bytes, file) = match self.read(path) {
            Found::Absent => return Lookup::Absent,
            Found::Failed => {
                count(&self.counters.read_failures);
                return Lookup::Unreadable;
            }
            Found::Bytes(bytes, file) => (bytes, Some(file)),
            Found::Queued(bytes) => (bytes, None),
        };
        match decode(&bytes) {
            Ok(value) => {
                self.accepted(file.as_ref());
                Lookup::Hit(value)
            }
            Err(_) => {
                self.refused(path);
                Lookup::Rejected
            }
        }
    }

    /// The contents a decoder accepted: a hit from `file` makes its entry the most recently used;
    /// a hit from the queue (no file) is counted as one the directory had not written yet.
    fn accepted(&self, file: Option<&fs::File>) {
        match file {
            Some(file) => timed(&self.counters.read_nanos, || touch(file)),
            None => count(&self.counters.queued_hits),
        }
    }

    /// The contents a decoder refused, at `path`: the entry is removed, and a write of it the
    /// cache thread has not made is dropped rather than put in its place.
    fn refused(&self, path: &Path) {
        if let Some(asked) = &self.asked {
            asked.discard(path);
        }
        timed(&self.counters.read_nanos, || remove(path));
        count(&self.counters.rejected);
    }

    /// Write `bytes` as the record named `name` in the content directory, as
    /// [`Store::write_soon`] writes an entry: on the cache thread, counted in no statistic but
    /// the bytes the directory holds and, where this thread writes it itself, the time
    /// ([`Counters::write_nanos`]).
    pub(crate) fn write_record_soon(&self, name: &str, bytes: Vec<u8>) -> io::Result<u64> {
        self.write_soon_at(self.directory.record_path(name), name, bytes)
    }

    /// Read the file at `path`, or the bytes of a write of it the cache thread has not made yet
    /// (`worker.rs`): its file is not in place until its `rename`, so the queue answers first. A
    /// file larger than any entry is removed and found absent.
    fn read(&self, path: &Path) -> Found {
        if let Some(bytes) = self.asked.as_ref().and_then(|asked| asked.pending(path)) {
            return Found::Queued(bytes);
        }
        count(&self.counters.entry_reads);
        timed(&self.counters.read_nanos, || {
            let file = match fs::File::open(path) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Found::Absent,
                Err(_) => return Found::Failed,
            };
            let Ok(length) = file.metadata().map(|metadata| metadata.len()) else {
                return Found::Failed;
            };
            let limit = self.entry_limit();
            if length > limit {
                remove(path);
                return Found::Absent;
            }
            let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
            match (&file).take(limit + 1).read_to_end(&mut bytes) {
                Ok(_) if bytes.len() as u64 <= limit => Found::Bytes(bytes, file),
                Ok(_) => {
                    remove(path);
                    Found::Absent
                }
                Err(_) => Found::Failed,
            }
        })
    }

    /// Give `bytes` to the cache thread to write as the entry named `name` (`worker.rs`): the
    /// same file and the same `rename`, off this thread. Their length, as a write of them gives
    /// it; the failure of one this thread wrote itself, where the queue was at a bound. The
    /// failure of a queued write is counted when the thread meets it, not here.
    pub(crate) fn write_soon(&self, name: &str, bytes: Vec<u8>) -> io::Result<u64> {
        self.write_soon_at(self.directory.entry_path(name), name, bytes)
    }

    /// [`Store::write_soon`] of the file at `path`, named `name`.
    fn write_soon_at(&self, path: PathBuf, name: &str, bytes: Vec<u8>) -> io::Result<u64> {
        let length = bytes.len() as u64;
        // An entry the queue could never hold whole is written here, as one past a bound is.
        let takeable = length <= QUEUED_BYTES && length <= self.entry_limit();
        let Some(asked) = self.asked.as_ref().filter(|_| takeable) else {
            count(&self.counters.writes_on_thread);
            return timed(&self.counters.write_nanos, || {
                self.write_at(&path, name, &bytes)
            });
        };
        match asked.queue(Queued::new(path, bytes)) {
            None => {
                count(&self.counters.writes_queued);
                Ok(length)
            }
            Some(write) => {
                count(&self.counters.writes_on_thread);
                timed(&self.counters.write_nanos, || {
                    self.write_at(&write.path, name, &write.bytes)
                })
            }
        }
    }

    /// Write the entry at `path` that the cache thread took, counting a failure as the calling
    /// thread's would be counted (`ShaderCacheStatistics::write_failures`) and its time as the
    /// cache thread's ([`Counters::off_thread_nanos`]).
    pub(crate) fn write_queued(&self, path: &Path, bytes: &[u8]) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("entry");
        let written = timed(&self.counters.off_thread_nanos, || {
            self.write_at(path, name, bytes)
        });
        if written.is_err() {
            self.write_failed();
        }
    }

    /// Write `bytes` as the entry at `path`, named `name`, on the calling thread: a temporary
    /// file beside it, flushed and renamed over it. The bytes written, or the failure (the
    /// temporary file is removed). An entry larger than [`Store::entry_limit`] fails.
    fn write_at(&self, path: &Path, name: &str, bytes: &[u8]) -> io::Result<u64> {
        let length = bytes.len() as u64;
        if length > self.entry_limit() {
            return Err(io::Error::other("the entry is larger than the cache takes"));
        }
        let parent = path.parent().expect("an entry has a directory");
        create_private_dir_all(parent)?;
        let temporary = parent.join(format!(".{name}.{}.tmp", unique()));
        let written = (|| {
            let mut file = new_private_file(&temporary)?;
            file.write_all(bytes)?;
            drop(file);
            fs::rename(&temporary, path)?;
            Ok(length)
        })();
        match &written {
            Ok(bytes) => {
                self.used.fetch_add(*bytes, Ordering::Relaxed);
                self.counters
                    .written_bytes
                    .fetch_add(*bytes, Ordering::Relaxed);
            }
            Err(_) => {
                let _ = fs::remove_file(&temporary);
            }
        }
        written
    }

    /// Whether the bytes this process counts are past the cap (judged only after a scan).
    pub(crate) fn over_capacity(&self) -> bool {
        self.scanned.load(Ordering::Acquire)
            && self.used.load(Ordering::Relaxed) > self.directory.capacity
    }

    /// The scan at device creation: remove other builds' directories nothing used for
    /// [`UNUSED_BUILD`] and stale temporary files, count what remains, and remove the least
    /// recently used entries if that is past the cap. Stops early when `stop` is set. The cache
    /// thread's time ([`Counters::off_thread_nanos`]).
    pub(crate) fn sweep(&self, stop: &AtomicBool) {
        timed(&self.counters.off_thread_nanos, || self.scan(stop, true));
    }

    /// Bring the directory under its cap: count every build's entries again and remove the
    /// least recently used past it. Stops early when `stop` is set.
    pub(crate) fn evict(&self, stop: &AtomicBool) {
        timed(&self.counters.off_thread_nanos, || self.scan(stop, false));
    }

    /// [`Store::sweep`] (with `unused_builds`) or [`Store::evict`].
    fn scan(&self, stop: &AtomicBool, unused_builds: bool) {
        let now = SystemTime::now();
        let own = hex(&self.directory.build);
        let Ok(builds) = fs::read_dir(self.directory.layout_dir()) else {
            return;
        };
        let mut entries = Vec::new();
        for build in builds.flatten() {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let path = build.path();
            if !path.is_dir() {
                continue;
            }
            let start = entries.len();
            let newest = survey(&path, now, &mut entries, stop);
            let unused = now.duration_since(newest).unwrap_or_default() >= UNUSED_BUILD;
            if unused_builds && unused && build.file_name().to_str() != Some(own.as_str()) {
                entries.truncate(start);
                let _ = fs::remove_dir_all(&path);
            }
        }
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let total: u64 = entries.iter().map(|entry| entry.bytes).sum();
        let remaining = if total > self.directory.capacity {
            self.remove_oldest(entries, total)
        } else {
            total
        };
        // Writes since the scan began may be counted twice: the next scan corrects it.
        self.used.store(remaining, Ordering::Relaxed);
        self.scanned.store(true, Ordering::Release);
    }

    /// Remove the least recently used of `entries` (holding `total` bytes) until what remains
    /// is at most [`LOW_WATER`] of the cap; the bytes that remain.
    fn remove_oldest(&self, mut entries: Vec<Entry>, mut total: u64) -> u64 {
        let target = self.directory.capacity / 100 * LOW_WATER;
        entries.sort_by_key(|entry| entry.used);
        for entry in entries {
            if total <= target {
                break;
            }
            if fs::remove_file(&entry.path).is_ok() {
                count(&self.counters.evictions);
                if !self.warnings.evicted.swap(true, Ordering::Relaxed) {
                    self.say(
                        EVICTED,
                        None,
                        format!(
                            "metal2vulkan: the shader cache directory {} reached its cap of {} \
                             bytes (`ShaderCacheOptions::capacity`): its least recently used \
                             entries are removed, down to {}% of the cap, and what they held \
                             (translations, driver pipeline data) is made again when \
                             next needed. A larger cap keeps them. This device's later removals \
                             are counted alone (`ShaderCacheStatistics::evictions`).",
                            self.directory.root().display(),
                            self.directory.capacity,
                            LOW_WATER,
                        ),
                    );
                }
            }
            total = total.saturating_sub(entry.bytes);
        }
        total
    }
}

/// Remove the file at `path` (another process may have removed it first).
fn remove(path: &Path) {
    let _ = fs::remove_file(path);
}

/// An entry a scan found.
struct Entry {
    path: PathBuf,
    bytes: u64,
    used: SystemTime,
}

/// Add the entries of the build directory `build` to `entries`, removing stale temporary
/// files; the most recent use of anything in it: an entry's access time, or a directory's
/// modification time (an entry added or removed). A directory's access time is not a use:
/// listing it, as this scan does, moves it.
fn survey(
    build: &Path,
    now: SystemTime,
    entries: &mut Vec<Entry>,
    stop: &AtomicBool,
) -> SystemTime {
    let modified = |metadata: fs::Metadata| metadata.modified().ok();
    let mut newest = fs::metadata(build)
        .ok()
        .and_then(modified)
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let Ok(groups) = fs::read_dir(build) else {
        return newest;
    };
    for group in groups.flatten() {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Some(time) = group.metadata().ok().and_then(modified) {
            newest = newest.max(time);
        }
        let Ok(files) = fs::read_dir(group.path()) else {
            continue;
        };
        for file in files.flatten() {
            let Ok(metadata) = file.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let used = last_use(&metadata);
            let temporary = file
                .file_name()
                .to_str()
                .is_some_and(|n| n.starts_with('.'));
            if temporary {
                if now.duration_since(used).unwrap_or_default() >= STALE_TEMPORARY {
                    let _ = fs::remove_file(file.path());
                }
                continue;
            }
            newest = newest.max(used);
            entries.push(Entry {
                path: file.path(),
                bytes: metadata.len(),
                used,
            });
        }
    }
    newest
}

/// When the file of `metadata` was last used: its access time, or where the platform keeps
/// none, its modification time.
fn last_use(metadata: &fs::Metadata) -> SystemTime {
    metadata
        .accessed()
        .or_else(|_| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Refresh `file`'s access time: a hit makes its entry the most recently used. Best effort.
fn touch(file: &fs::File) {
    let _ = file.set_times(fs::FileTimes::new().set_accessed(SystemTime::now()));
}

/// A name no other writer uses: this process's id, a per-process sequence number, and the time.
fn unique() -> String {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    format!(
        "{}.{}.{nanos}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

/// Create `dir` and its missing parents, each only its owner may enter.
pub(super) fn create_private_dir_all(dir: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

/// Create the new file `path`, which only its owner may read; fails if it exists.
fn new_private_file(path: &Path) -> io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}
