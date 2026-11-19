//! The read-backs of a stream whose header marks its announced bytes recorded
//! ([`FLAG_ANNOUNCED_BYTES_RECORDED`](crate::replay::FLAG_ANNOUNCED_BYTES_RECORDED), sections 1 and
//! 6 of `docs/REPLAY-STREAM.md`). In such a stream the last write before a `didModifyRange:` that
//! carries each 4 KiB chunk of the range holds the program's CPU copy's bytes there, the GPU's
//! that a `synchronizeResource:` read back included. So a `synchronizeResource:` of a buffer played
//! with two copies (the runtime's Managed buffer with a device copy, or a second copy of the
//! player's own) copies the GPU's copy into an aside buffer of the player's own, not into the
//! CPU's, which keeps the stream's bytes for the announcement to put into the GPU's copy. A check
//! of the buffer reads the aside from the read-back to the buffer's next `didModifyRange:`, as
//! Metal's `contents` answers the bytes read back; then the CPU's copy again. A stream without the
//! flag plays its read-backs into the CPU's copy, as Metal does.

use super::places::Place;
use metal2vulkan_rt as rt;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

/// One buffer's aside, its place, and whether it holds the buffer's last read-back since its last
/// announcement.
struct Aside {
    buffer: rt::Buffer,
    place: Place,
    read_back: bool,
}

#[derive(Default)]
pub(super) struct Asides {
    /// Whether the stream's header marks its announced bytes recorded.
    marked: bool,
    /// The live buffers played with two copies.
    two_copies: HashSet<u64>,
    buffers: HashMap<u64, Aside>,
    /// The read-backs copied into an aside, the asides made and their bytes.
    read_backs: u64,
    made: u64,
    made_bytes: u64,
}

impl Asides {
    /// The stream's header marks its announced bytes recorded.
    pub(super) fn mark(&mut self) {
        self.marked = true;
    }

    /// Buffer `object` made with two copies.
    pub(super) fn two_copies(&mut self, object: u64) {
        self.two_copies.insert(object);
    }

    /// Whether buffer `object` lives with two copies.
    pub(super) fn has_two(&self, object: u64) -> bool {
        self.two_copies.contains(&object)
    }

    /// Whether a read-back of buffer `object` goes into an aside: the stream is marked and the
    /// buffer has two copies.
    pub(super) fn takes(&self, object: u64) -> bool {
        self.marked && self.two_copies.contains(&object)
    }

    /// Buffer `object`'s aside, made by `make` (a buffer of the length given, and its place)
    /// where it has none, then marked as holding its read-back.
    pub(super) fn read_back<E>(
        &mut self,
        object: u64,
        length: u64,
        make: impl FnOnce(u64) -> Result<(rt::Buffer, Place), E>,
    ) -> Result<rt::Buffer, E> {
        let aside = match self.buffers.entry(object) {
            Entry::Occupied(aside) => aside.into_mut(),
            Entry::Vacant(vacant) => {
                let (buffer, place) = make(length)?;
                self.made += 1;
                self.made_bytes += length;
                vacant.insert(Aside {
                    buffer,
                    place,
                    read_back: false,
                })
            }
        };
        self.read_backs += 1;
        aside.read_back = true;
        Ok(aside.buffer.clone())
    }

    /// Buffer `object` announced (`didModifyRange:`): its CPU copy is what a check reads again.
    pub(super) fn announced(&mut self, object: u64) {
        if let Some(aside) = self.buffers.get_mut(&object) {
            aside.read_back = false;
        }
    }

    /// The aside holding buffer `object`'s read-back since its last announcement, where one does.
    pub(super) fn read(&self, object: u64) -> Option<&rt::Buffer> {
        let aside = self.buffers.get(&object)?;
        aside.read_back.then_some(&aside.buffer)
    }

    /// Buffer `object` released: its aside's place, where it had one.
    pub(super) fn remove(&mut self, object: u64) -> Option<Place> {
        self.two_copies.remove(&object);
        self.buffers.remove(&object).map(|aside| aside.place)
    }

    /// The report's line: which bytes an announcement puts into a GPU's copy, and the asides.
    pub(super) fn line(&self) -> String {
        let rule = match self.marked {
            true => "the stream's (header flag bit 0)",
            false => "the CPU copy's, read-backs included (no header flag)",
        };
        format!(
            "announced bytes: {rule}; {} read-backs into {} asides ({} bytes)",
            self.read_backs, self.made, self.made_bytes
        )
    }
}
