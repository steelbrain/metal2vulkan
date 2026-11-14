//! The storage of a command buffer's lists of what its encoders ended with, kept by its device
//! from one command buffer to the next: its ended encoders' records (`State::encoder_ended`), their
//! recordings (`State::end_encoder`), the logs of a group of encoders recorded into one
//! `VkCommandBuffer` its members did not keep (`encoder_fold.rs`'s `Fold`), and those held for its
//! commit (`fold.rs`). A command buffer starts on the latest storage a released one left, emptied
//! with its room kept, so its encoders' ends push into room the lists already had rather than
//! grow them one reallocation at a time over fresh pages; its release empties the lists and gives
//! their storage back ([`SpareRecordStorage`]). What the lists held goes as it went before, at the
//! release; only the room goes on, and only to command buffers of the same device.

use std::sync::Mutex;

use crate::device::lock;
use crate::encoder::Recorded;
use crate::encoder_fold::Stashed;
use crate::used_resources::EncoderRecord;

/// The storages a device keeps, at most: a command buffer made after another's release takes the
/// latest, so a program that makes one as one is released keeps one in use.
const SPARE_RECORD_STORAGES: usize = 2;

/// The room a kept storage's lists hold, at most, in bytes: a command buffer of more encoders
/// than that room holds lets its storage go. A command buffer of a few thousand one-dispatch
/// encoders fits.
const RECORD_STORAGE_BYTES: usize = 8 << 20;

/// The lists a command buffer keeps of what its encoders ended with, whose storage goes from one
/// command buffer of a device to the next (the module docs).
#[derive(Default)]
pub(crate) struct RecordStorage {
    pub(super) ended: Vec<EncoderRecord>,
    pub(super) recorded: Vec<Recorded>,
    /// The storage the command buffer's next group of encoders stashes its members' logs in.
    pub(super) stashed: Vec<Stashed>,
    pub(super) held_logs: Vec<Stashed>,
}

impl RecordStorage {
    /// The room its lists hold, each list's in entries.
    pub(super) fn room(&self) -> Room {
        Room([
            self.ended.capacity(),
            self.recorded.capacity(),
            self.stashed.capacity(),
            self.held_logs.capacity(),
        ])
    }

    /// The room its lists hold, in bytes.
    fn bytes(&self) -> usize {
        self.ended.capacity() * std::mem::size_of::<EncoderRecord>()
            + self.recorded.capacity() * std::mem::size_of::<Recorded>()
            + (self.stashed.capacity() + self.held_logs.capacity()) * std::mem::size_of::<Stashed>()
    }
}

/// The room of each of a [`RecordStorage`]'s lists as a command buffer took it.
#[derive(Clone, Copy)]
pub(super) struct Room([usize; 4]);

impl Room {
    /// How many of `storage`'s lists hold more room than this: those that grew since.
    pub(super) fn grown(self, storage: &RecordStorage) -> u64 {
        let Room(now) = storage.room();
        let grown = (self.0.iter().zip(now)).filter(|&(&was, now)| now > was);
        grown.count() as u64
    }
}

/// A device's storages of the lists of records its released command buffers left, which the
/// command buffers it makes next start on (the module docs): at most [`SPARE_RECORD_STORAGES`],
/// each of at most [`RECORD_STORAGE_BYTES`], the latest given taken first.
#[derive(Default)]
pub(crate) struct SpareRecordStorage(Mutex<Vec<RecordStorage>>);

impl SpareRecordStorage {
    /// The storage a command buffer being made starts on: the latest a released one gave back,
    /// else lists of no room.
    pub(super) fn take(&self) -> RecordStorage {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep `storage`, a released command buffer's, emptied here (what its lists held goes) for
    /// the next command buffer to start on, where its room is within [`RECORD_STORAGE_BYTES`]
    /// and fewer than [`SPARE_RECORD_STORAGES`] are kept; else it goes. Whether it was kept.
    pub(super) fn give(&self, mut storage: RecordStorage) -> bool {
        storage.ended.clear();
        storage.recorded.clear();
        storage.stashed.clear();
        storage.held_logs.clear();
        if storage.bytes() > RECORD_STORAGE_BYTES {
            return false;
        }
        let mut spare = lock(&self.0);
        let keeps = spare.len() < SPARE_RECORD_STORAGES;
        if keeps {
            spare.push(storage);
        }
        keeps
    }
}
