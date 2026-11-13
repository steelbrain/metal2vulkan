//! The content directory's large records, kept as chunks (`store.rs`): one copy of each run of
//! bytes the records share.
//!
//! ```text
//! <root>/chunks/<first byte of digest, hex>/<digest, hex>
//! <root>/chunks/lock
//! <root>/chunks/state
//! ```
//!
//! - **Why.** The pipeline cache data of NVIDIA's driver (620.70.06, an RTX 3090) held one
//!   1,851,552-byte section, the same bytes, in 3,565 of one node's records, each record a copy
//!   of it: that node's 11,853 NVIDIA records of 8,061 MB are 180 MB once each run of bytes they
//!   share is kept once (a measure of its records at the average below).
//! - **Content-defined chunks.** A record of [`WHOLE`] bytes or more is cut where a rolling hash
//!   of its bytes ([`ends`]: a gear hash, FastCDC's normalized cut between [`MIN`] and [`MAX`]
//!   around an [`AVERAGE`]) meets a mask, so a run two records share is cut alike in both
//!   wherever it lies in each, and each chunk is kept once, named by its SHA-256. The cuts read
//!   the bytes alone: nothing of any driver's format. A smaller record stays whole.
//! - **The manifest** is the record's file, at its path: [`MAGIC`], the record's length, the
//!   chunks' count, each chunk's SHA-256 and length, and the SHA-256 of every byte before. A read
//!   joins the chunks back into the record's bytes, which its decoder checks end to end as it
//!   checks a whole record's (a driver record's checksum, a library's content hash), so a chunk
//!   is not hashed again as it is read. A chunk that is not there, or of another length, makes
//!   the read a miss, and a making writes the record again.
//! - **A manifest never names a chunk that is not there.** A write writes each chunk, a
//!   temporary file renamed over it, before the manifest's own rename; a crash leaves chunks no
//!   manifest names, never a manifest without its chunks. A chunk already there is read and
//!   checked against its digest first, and written again where it does not match. The cleanup
//!   removes only chunks no manifest names: writers hold [`LOCK`] shared from their first look at
//!   a chunk to their manifest's rename, and the cleanup holds it exclusively (`File::lock`, an
//!   advisory lock on the file: other processes' writers and cleanups included). Neither waits:
//!   the cleanup takes it with `try_lock` and is left to a later scan while a writer writes, and a
//!   writer that meets a cleanup writes its record whole.
//! - **Outside the layout directory.** The chunks live beside `<root>/v1`, not in it, so a build
//!   whose scans know nothing of them (they remove a build's directory unused for 30 days and the
//!   least recently used files they find) never removes one.
//! - **A bounded cleanup** ([`Chunks::cleanup`]), in the scans `store.rs` makes already: at the
//!   sweep at a device's creation where none ran yet ([`STATE`] missing) or the chunks hold more
//!   than [`GROWTH`] percent of what the last one kept, and after a scan past the cap removed a
//!   record. It reads the first bytes of each content record (a manifest whole: 36 bytes a
//!   chunk) and lists the chunks once. It also removes the content directory's records of a
//!   retired frame (`driver::RETIRED`: the driver data of an older `driver::VERSION`, which no
//!   build of this one names), so an upgrade does not leave the old records beside the new.

use super::store::{create_private_dir_all, hex, new_private_file, unique};
use metal2vulkan::cache::Hash256;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// The least record kept as chunks; a smaller one stays whole.
pub(crate) const WHOLE: usize = 256 << 10;
/// The chunks' average length. Of the averages measured on the node's records (16, 32 and
/// 64 KiB), the least stored: 64 KiB keeps the chunk across each end of a shared run apart in
/// every record, 16 KiB doubles the files a read opens for 1.3 times the bytes.
pub(crate) const AVERAGE: usize = 32 << 10;
/// The shortest chunk but a record's last.
const MIN: usize = AVERAGE / 4;
/// The longest chunk.
const MAX: usize = AVERAGE * 4;
/// The first bytes of a manifest.
pub(crate) const MAGIC: [u8; 8] = *b"M2VCHNK1";
/// A manifest's bytes before its chunks: the magic, the record's length (u64) and the count
/// (u32).
const HEAD: usize = 8 + 8 + 4;
/// A chunk's bytes in a manifest: its SHA-256 and its length (u32).
const ENTRY: usize = 32 + 4;
/// A manifest's bytes after its chunks: the SHA-256 of every byte before.
const CHECKSUM: usize = 32;
/// The lock file writers hold shared and the cleanup exclusively.
const LOCK: &str = "lock";
/// The file the last cleanup left the bytes it kept in (u64).
const STATE: &str = "state";
/// The share (percent) of the last cleanup's kept bytes past which the chunks bring the sweep's
/// cleanup.
const GROWTH: u64 = 110;

/// The gear hash's table: a value per byte, from SplitMix64.
const GEAR: [u64; 256] = gear();

const fn gear() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut at = 0;
    while at < 256 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        table[at] = z ^ (z >> 31);
        at += 1;
    }
    table
}

/// The ends of `data`'s chunks, in order (the module docs): each cut where the gear hash of the
/// bytes since the chunk's [`MIN`]th meets the harder mask before [`AVERAGE`] and the easier one
/// after it, else at [`MAX`]; the last chunk ends with the data.
pub(crate) fn ends(data: &[u8]) -> Vec<usize> {
    let bits = AVERAGE.trailing_zeros();
    // The masks take the hash's bits from the 16th on, which the 16 bytes before a byte reach too.
    let hard = ((1u64 << (bits + 1)) - 1) << 16;
    let easy = ((1u64 << (bits - 1)) - 1) << 16;
    let mut ends = Vec::with_capacity(data.len() / AVERAGE + 1);
    let mut start = 0;
    while start < data.len() {
        let rest = &data[start..];
        start += match rest.len() <= MIN {
            true => rest.len(),
            false => cut(rest, hard, easy),
        };
        ends.push(start);
    }
    ends
}

/// The length of the chunk `rest` begins with ([`ends`]).
fn cut(rest: &[u8], hard: u64, easy: u64) -> usize {
    let limit = rest.len().min(MAX);
    let normal = rest.len().min(AVERAGE);
    let mut hash = 0u64;
    for (at, &byte) in rest.iter().enumerate().take(limit).skip(MIN) {
        hash = (hash << 1).wrapping_add(GEAR[usize::from(byte)]);
        let mask = if at < normal { hard } else { easy };
        if hash & mask == 0 {
            return at + 1;
        }
    }
    limit
}

/// A record's chunks, as its manifest lists them: each one's SHA-256 and length.
pub(crate) type Listed = Vec<(Hash256, u32)>;

/// The manifest of a record of `length` bytes cut into `listed`.
fn manifest(length: u64, listed: &[(Hash256, u32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEAD + ENTRY * listed.len() + CHECKSUM);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&(listed.len() as u32).to_le_bytes());
    for (digest, chunk) in listed {
        bytes.extend_from_slice(digest);
        bytes.extend_from_slice(&chunk.to_le_bytes());
    }
    let checksum: Hash256 = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The record's length and chunks a manifest's `bytes` list, where they are one whole: its magic,
/// its checksum, its count and the chunks' lengths summing to the record's.
pub(crate) fn listed(bytes: &[u8]) -> Option<(u64, Listed)> {
    if bytes.get(..8)? != MAGIC || bytes.len() < HEAD + CHECKSUM {
        return None;
    }
    let body = bytes.len() - CHECKSUM;
    let checksum: Hash256 = Sha256::digest(&bytes[..body]).into();
    if bytes[body..] != checksum {
        return None;
    }
    let length = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let count = u32::from_le_bytes(bytes[16..HEAD].try_into().ok()?) as usize;
    if body != HEAD + count.checked_mul(ENTRY)? {
        return None;
    }
    let (entries, _) = bytes[HEAD..body].as_chunks::<ENTRY>();
    let listed: Listed = entries
        .iter()
        .map(|entry| {
            let digest: Hash256 = entry[..32].try_into().expect("32 bytes");
            let chunk = u32::from_le_bytes(entry[32..].try_into().expect("4 bytes"));
            (digest, chunk)
        })
        .collect();
    let sum: u64 = listed.iter().map(|(_, chunk)| u64::from(*chunk)).sum();
    (sum == length).then_some((length, listed))
}

/// The chunk directory of a cache root (the module docs).
pub(crate) struct Chunks {
    dir: PathBuf,
}

impl Chunks {
    /// The chunks of the cache directory `root`.
    pub(crate) fn of(root: &Path) -> Chunks {
        Chunks {
            dir: root.join("chunks"),
        }
    }

    /// The path of the chunk of SHA-256 `digest`.
    fn path(&self, digest: &Hash256) -> PathBuf {
        let name = hex(digest);
        self.dir.join(&name[..2]).join(name)
    }

    /// The lock file, opened (made where missing).
    fn lock(&self) -> io::Result<fs::File> {
        create_private_dir_all(&self.dir)?;
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.dir.join(LOCK))
    }

    /// Write `bytes`, a record, as chunks and then its manifest, which `put` writes in its place
    /// (a temporary file renamed over it), holding the lock shared throughout (the module docs);
    /// while a cleanup holds it, `put` writes the record whole instead, so no write waits. The
    /// bytes written: the chunks not there before, and the manifest.
    pub(crate) fn write(
        &self,
        bytes: &[u8],
        put: impl FnOnce(&[u8]) -> io::Result<u64>,
    ) -> io::Result<u64> {
        let lock = self.lock()?;
        match lock.try_lock_shared() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => return put(bytes),
            Err(fs::TryLockError::Error(error)) => return Err(error),
        }
        let mut listed = Vec::new();
        let mut written = 0u64;
        let mut start = 0;
        for end in ends(bytes) {
            let chunk = &bytes[start..end];
            start = end;
            let digest: Hash256 = Sha256::digest(chunk).into();
            let path = self.path(&digest);
            if !holds(&path, &digest) {
                write_chunk(&path, chunk)?;
                written += chunk.len() as u64;
            }
            listed.push((digest, chunk.len() as u32));
        }
        let put = put(&manifest(bytes.len() as u64, &listed))?;
        drop(lock);
        Ok(written + put)
    }

    /// The record `manifest` lists, its chunks joined; `None` where it is no manifest, lists a
    /// record past `limit` bytes, or names a chunk that is not there or of another length.
    pub(crate) fn read(&self, manifest: &[u8], limit: u64) -> Option<Vec<u8>> {
        let (length, listed) = listed(manifest)?;
        if length > limit {
            return None;
        }
        let mut bytes = Vec::with_capacity(usize::try_from(length).ok()?);
        // Every chunk opened first, and on Linux the kernel asked to read each ahead, so the reads
        // of those not in the page cache run together rather than one after another.
        let files: Vec<(fs::File, u32)> = (listed.into_iter())
            .map(|(digest, chunk)| Some((fs::File::open(self.path(&digest)).ok()?, chunk)))
            .collect::<Option<_>>()?;
        #[cfg(target_os = "linux")]
        for (file, _) in &files {
            use std::os::fd::AsRawFd;
            // SAFETY: a file descriptor of an open file, the whole of it (a length of 0); the
            // advice changes no contents.
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED) };
        }
        for (file, chunk) in files {
            let before = bytes.len();
            file.take(u64::from(chunk) + 1)
                .read_to_end(&mut bytes)
                .ok()?;
            if bytes.len() - before != chunk as usize {
                return None;
            }
        }
        Some(bytes)
    }

    /// The bytes the chunks hold, by a listing.
    pub(crate) fn bytes(&self) -> u64 {
        let mut bytes = 0;
        each_chunk(&self.dir, |_, metadata| bytes += metadata.len());
        bytes
    }

    /// Whether the sweep's cleanup is due: none left its state yet, or the chunks hold more than
    /// [`GROWTH`] percent of what the last one kept.
    pub(crate) fn due(&self) -> bool {
        let kept = fs::read(self.dir.join(STATE))
            .ok()
            .and_then(|bytes| bytes.try_into().ok())
            .map(u64::from_le_bytes);
        match kept {
            None => true,
            Some(kept) => self.bytes() > kept.saturating_mul(GROWTH) / 100,
        }
    }

    /// The manifests among the records of the content directory `content`, by path, with the
    /// chunks each lists, the records of a `retired` frame removed on the way; `None` where
    /// `stop` was set before every record was read.
    pub(crate) fn manifests(
        &self,
        content: &Path,
        retired: &[[u8; 8]],
        stop: &AtomicBool,
    ) -> Option<HashMap<PathBuf, Listed>> {
        let mut manifests = HashMap::new();
        let groups = match fs::read_dir(content) {
            Ok(groups) => groups,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Some(manifests),
            Err(_) => return None,
        };
        for group in groups.flatten() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let Ok(files) = fs::read_dir(group.path()) else {
                continue;
            };
            for file in files.flatten() {
                let temporary = file
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with('.'));
                if temporary {
                    continue;
                }
                let path = file.path();
                let Ok(mut opened) = fs::File::open(&path) else {
                    continue;
                };
                let mut head = [0u8; 8];
                if opened.read_exact(&mut head).is_err() {
                    continue;
                }
                if retired.contains(&head) {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                if head != MAGIC {
                    continue;
                }
                let mut bytes = head.to_vec();
                if opened.read_to_end(&mut bytes).is_err() {
                    // A manifest not read whole names chunks that stay: the cleanup gives up.
                    return None;
                }
                if let Some((_, listed)) = listed(&bytes) {
                    manifests.insert(path, listed);
                }
            }
        }
        Some(manifests)
    }

    /// Remove the chunks no manifest of the content directory `content` names, and the
    /// temporary files of writers that died, and the content records of a `retired` frame,
    /// holding the lock exclusively; the bytes the chunks kept hold, left as [`STATE`]. `None`
    /// where a writer holds the lock (nothing is done), or `stop` was set first.
    pub(crate) fn cleanup(
        &self,
        content: &Path,
        retired: &[[u8; 8]],
        stop: &AtomicBool,
    ) -> Option<u64> {
        let lock = self.lock().ok()?;
        lock.try_lock().ok()?;
        let manifests = self.manifests(content, retired, stop)?;
        let named: HashSet<&Hash256> = (manifests.values().flatten())
            .map(|(digest, _)| digest)
            .collect();
        let mut kept = 0;
        each_chunk(&self.dir, |path, metadata| {
            let name = path.file_name().and_then(|name| name.to_str());
            // A temporary file: the writer that made it holds no lock, so it died.
            let temporary = name.is_none_or(|name| name.starts_with('.'));
            let digest = name.and_then(unhex);
            match digest.filter(|digest| named.contains(digest)) {
                Some(_) if !temporary => kept += metadata.len(),
                _ => {
                    let _ = fs::remove_file(path);
                }
            }
        });
        let _ = fs::write(self.dir.join(STATE), kept.to_le_bytes());
        drop(lock);
        Some(kept)
    }
}

/// Run `f` over each file in a group directory of `dir` (the chunks and their temporary files).
fn each_chunk(dir: &Path, mut f: impl FnMut(&Path, &fs::Metadata)) {
    let Ok(groups) = fs::read_dir(dir) else {
        return;
    };
    for group in groups.flatten() {
        if !group.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let Ok(files) = fs::read_dir(group.path()) else {
            continue;
        };
        for file in files.flatten() {
            if let Ok(metadata) = file.metadata() {
                if metadata.is_file() {
                    f(&file.path(), &metadata);
                }
            }
        }
    }
}

/// The digest a chunk's file name spells, in hex.
fn unhex(name: &str) -> Option<Hash256> {
    if name.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (at, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(name.get(2 * at..2 * at + 2)?, 16).ok()?;
    }
    Some(digest)
}

/// Whether the chunk at `path` is there and holds the bytes of SHA-256 `digest`.
fn holds(path: &Path, digest: &Hash256) -> bool {
    fs::read(path).is_ok_and(|bytes| Sha256::digest(&bytes)[..] == digest[..])
}

/// Write `chunk` at `path`: a temporary file beside it, renamed over it.
fn write_chunk(path: &Path, chunk: &[u8]) -> io::Result<()> {
    let parent = path.parent().expect("a chunk has a group directory");
    create_private_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("chunk");
    let temporary = parent.join(format!(".{name}.{}.tmp", unique()));
    let written = (|| {
        let mut file = new_private_file(&temporary)?;
        file.write_all(chunk)?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}
