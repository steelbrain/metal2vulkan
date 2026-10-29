//! The libraries pipeline records name (`warm/record.rs`), kept in the directory's content
//! directory by their content hash, so a record names each by its hash alone and a VM's records
//! over one library hold it once between them.
//!
//! - **Named by content.** A library's record is named by its content hash
//!   (`Library::content_hash`, the SHA-256 of its whole container), in the content directory
//!   every build shares (`store.rs`): a pin move finds it as it finds the driver's data. Its
//!   bytes are [`MAGIC`], the hash, and the library's container bytes.
//! - **Checked as it is read.** A read is a hit only where the magic, the hash and the length are
//!   the record's, the bytes parse as a library (`Library::parse`) and that library's content hash
//!   is the one asked for; anything else is a miss and the file is removed.
//! - **Written once.** A pipeline record's making keeps each library it names here where the
//!   directory does not hold it already, written by the device's cache thread as an entry is
//!   (`worker.rs`); one larger than the queue's whole bound (16 MiB) is written on the creating
//!   thread, once per directory. The device keeps the hashes it knows the directory holds
//!   ([`KeptLibraries`], at most [`KNOWN`]), forgotten at any removal from the directory, so a
//!   later record of the same library asks nothing of the directory.
//! - **Bounded with the rest.** The scans count these records against the directory's cap, and
//!   the least recently used go first (a warm pass's read refreshes a library's use); a pipeline
//!   record whose library went names one the directory no longer holds, which a warm pass counts
//!   unread (`warm/pass.rs`).

use super::store::{count, entry_name, Lookup};
use super::{EntryKind, ShaderCache};
use crate::device::lock;
use metal2vulkan::cache::Hash256;
use metal2vulkan::library::Library;
use std::collections::HashSet;
use std::sync::Mutex;

/// The first bytes of a library's record.
const MAGIC: &[u8; 8] = b"M2VLIBR1";

/// The length of a library record's head: its magic and the content hash.
const HEAD: usize = 8 + 32;

/// The most library hashes a device keeps as held: more than any one guest's libraries.
pub(crate) const KNOWN: usize = 1024;

/// The library hashes this device knows the directory holds, and the store's removal count they
/// were known at (`Store::removals`: a removal may have taken one).
pub(crate) struct KeptLibraries(Mutex<(HashSet<Hash256>, u64)>);

impl KeptLibraries {
    /// None known.
    pub(crate) fn new() -> KeptLibraries {
        KeptLibraries(Mutex::new((HashSet::new(), 0)))
    }
}

/// The name of the record of the library of content hash `hash`.
fn name(hash: &Hash256) -> String {
    entry_name(&super::store::content(), EntryKind::Library, hash, &[0; 32])
}

impl ShaderCache {
    /// Keep `library` in the directory, where this device does not know it holds it already
    /// (the module docs).
    pub(crate) fn keep_library(&self, library: &Library) {
        let hash = *library.content_hash();
        let removals = self.store.removals();
        {
            let mut known = lock(&self.libraries.0);
            if known.1 != removals {
                known.0.clear();
                known.1 = removals;
            }
            if known.0.contains(&hash) {
                return;
            }
        }
        let name = name(&hash);
        if !self.store.holds_record(&name) {
            let mut bytes = Vec::with_capacity(HEAD + library.bytes().len());
            bytes.extend_from_slice(MAGIC);
            bytes.extend_from_slice(&hash);
            bytes.extend_from_slice(library.bytes());
            match self.store.write_record_soon(&name, bytes) {
                Ok(_) => count(&self.store.counters.libraries_kept),
                Err(_) => self.store.write_failed(),
            }
            if self.store.over_capacity() {
                self.worker.evict();
            }
        }
        let mut known = lock(&self.libraries.0);
        if known.0.len() >= KNOWN {
            known.0.clear();
        }
        known.0.insert(hash);
    }

    /// The library of content hash `hash` and container length `length` the directory holds,
    /// parsed and checked (the module docs); `None` where it holds none.
    pub(crate) fn library(&self, hash: &Hash256, length: u64) -> Option<Library> {
        let found = self.store.load_content(&name(hash), |bytes| {
            if bytes.len() as u64 != HEAD as u64 + length
                || &bytes[..8] != MAGIC
                || bytes[8..HEAD] != *hash
            {
                return None;
            }
            let library = Library::parse(&bytes[HEAD..]).ok()?;
            (library.content_hash() == hash).then_some(library)
        });
        match found {
            Lookup::Hit(library) => Some(library),
            Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
        }
    }

    /// The libraries this device wrote to the directory (`WarmStatistics::libraries_kept`).
    pub(crate) fn libraries_kept(&self) -> u64 {
        self.store
            .counters
            .libraries_kept
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
