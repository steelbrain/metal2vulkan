//! The states of a device's released command buffers, kept for the command buffers it makes
//! next. A released command buffer's state is reset in place by the thread that releases it
//! ([`State::reset`]): what it held goes as it went before, at the release, and its lists of what
//! its encoders ended with (its ended encoders' records, their recordings, the logs of a group of
//! encoders recorded into one `VkCommandBuffer` its members did not keep, `encoder_fold.rs`'s
//! `Fold`, and those held for its commit, `fold.rs`), its command predication's lists
//! (`predication.rs`) and its groups' order (`FoldOrder`, its map and its resources' lists) are
//! emptied with their room kept. A command buffer being made takes a kept state whole
//! ([`SpareStates::take`]): its making neither allocates nor fills one, and its encoders' ends push
//! into room the lists already had rather than grow them one reallocation at a time over fresh
//! pages. Its release also keeps up to [`STATE_LOGS`] of its compute encoders' logs, cleared, in
//! the state (`State::compute_logs`): the compute encoders of the command buffer that takes it
//! take those, under the lock their open takes, rather than the device's spares' (`replay.rs`),
//! whose lock the queues' handler threads take as they give logs back. A state goes on only to
//! command buffers of the device generation that made it, and the kept ones go with that
//! generation.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use super::State;
use crate::device::lock;
use crate::encoder::Recorded;
use crate::encoder_fold::{FoldOrder, Stashed};
use crate::predication::Predication;
use crate::resource::ResourceKey;
use crate::used_resources::EncoderRecord;

/// The states a device keeps, at most: one per uncompleted command buffer of a queue of Metal's
/// default `maxCommandBufferCount`, so the releases of a queue's whole run of command buffers
/// serve the makings after them.
const SPARE_STATES: usize = 64;

/// The room a kept state's lists hold, at most, in bytes: a command buffer of more encoders than
/// that room holds lets its state go. A command buffer of a few thousand one-dispatch encoders
/// fits.
const STATE_ROOM_BYTES: usize = 8 << 20;

/// The room the kept states' lists hold together, at most, in bytes.
const SPARE_ROOM_BYTES: usize = 2 * STATE_ROOM_BYTES;

/// The cleared compute logs a state keeps, at most (`State::compute_logs`): each holds at most
/// `SPARE_LOG_BYTES` (64 KiB, `replay.rs`), and their room counts toward the state's
/// ([`STATE_ROOM_BYTES`]).
pub(super) const STATE_LOGS: usize = 32;

/// The lists of records a state keeps the room of ([`Room`]).
const LISTS: usize = 12;

/// The room of each of a state's lists of records, in entries (its ended encoders' records, their
/// recordings, its stash of a group's logs, the logs it holds for its commit, its command
/// predication's five lists, and its groups' order's map, boxes and priors), and the room they
/// and the compute logs it keeps hold, in bytes.
#[derive(Clone, Copy, Default)]
pub(super) struct Room {
    lists: [usize; LISTS],
    bytes: usize,
}

impl Room {
    /// The room of a state's lists: of its lists of records `records` (its ended encoders'
    /// records, their recordings, its stash and the logs it holds for its commit), of its command
    /// predication's lists `predication` (`Predication::room`), of its groups' order `order`
    /// (`FoldOrder::room`), each in entries, and of the compute logs it keeps, `logs` bytes.
    pub(super) fn new(
        records: [usize; 4],
        predication: [usize; 5],
        order: [usize; 3],
        logs: usize,
    ) -> Room {
        let [ended, recorded, stashed, held_logs] = records;
        let bytes = ended * std::mem::size_of::<EncoderRecord>()
            + recorded * std::mem::size_of::<Recorded>()
            + (stashed + held_logs) * std::mem::size_of::<Stashed>()
            + Predication::room_bytes(predication)
            + FoldOrder::<ResourceKey>::room_bytes(order)
            + logs;
        let [runs, open_checks, opened, alone, opened_fresh] = predication;
        let [map, boxes, priors] = order;
        let lists = [
            ended,
            recorded,
            stashed,
            held_logs,
            runs,
            open_checks,
            opened,
            alone,
            opened_fresh,
            map,
            boxes,
            priors,
        ];
        Room { lists, bytes }
    }

    /// How many of the lists hold more room in `now` than in this: those that grew since.
    pub(super) fn grown(self, now: Room) -> u64 {
        let grown = (self.lists.iter().zip(now.lists)).filter(|&(&was, now)| now > was);
        grown.count() as u64
    }

    /// The room, in bytes.
    pub(super) fn bytes(self) -> usize {
        self.bytes
    }
}

/// A released command buffer's state, reset, as a device keeps it: the state, the revision its
/// parts that footprints are built from move (`Footprints::revision`), which the command buffer
/// that takes it builds its footprints against, and the room its lists hold, in bytes.
pub(super) struct KeptState {
    pub(super) state: Arc<Mutex<State>>,
    pub(super) revision: Arc<AtomicU64>,
    bytes: usize,
}

impl KeptState {
    /// `state`, reset, whose parts move `revision` and whose lists hold `room`.
    pub(super) fn new(state: Arc<Mutex<State>>, revision: Arc<AtomicU64>, room: Room) -> KeptState {
        KeptState {
            state,
            revision,
            bytes: room.bytes(),
        }
    }

    /// A state built for a command buffer being made, where the device keeps none: lists of no
    /// room, under a revision of its own.
    pub(super) fn fresh() -> KeptState {
        let revision = Arc::default();
        let state = Arc::new(Mutex::new(State::new(&revision)));
        KeptState {
            state,
            revision,
            bytes: 0,
        }
    }
}

/// A device's states of its released command buffers, which the command buffers it makes next
/// take (the module docs): at most [`SPARE_STATES`], whose lists hold at most
/// [`SPARE_ROOM_BYTES`] together and [`STATE_ROOM_BYTES`] each, the latest given taken first.
#[derive(Default)]
pub(crate) struct SpareStates(Mutex<Spare>);

/// What [`SpareStates`] holds.
#[derive(Default)]
struct Spare {
    kept: Vec<KeptState>,
    /// The room the kept states' lists hold, in bytes.
    bytes: usize,
}

impl Spare {
    /// Whether it has a place for a state whose lists hold `bytes`.
    fn fits(&self, bytes: usize) -> bool {
        bytes <= STATE_ROOM_BYTES
            && self.kept.len() < SPARE_STATES
            && self.bytes + bytes <= SPARE_ROOM_BYTES
    }
}

impl SpareStates {
    /// The state a command buffer being made takes: the latest a released one left, if any.
    pub(super) fn take(&self) -> Option<KeptState> {
        let mut spare = lock(&self.0);
        let kept = spare.kept.pop()?;
        spare.bytes -= kept.bytes;
        Some(kept)
    }

    /// Whether it would keep a state whose lists hold `room` now: a released command buffer's
    /// state is reset only where it would.
    pub(super) fn wants(&self, room: Room) -> bool {
        lock(&self.0).fits(room.bytes())
    }

    /// Keep `kept`, a released command buffer's state, reset, where it has a place for it; else
    /// it goes with the command buffer. Whether it was kept.
    pub(super) fn give(&self, kept: KeptState) -> bool {
        let mut spare = lock(&self.0);
        if !spare.fits(kept.bytes) {
            return false;
        }
        spare.bytes += kept.bytes;
        spare.kept.push(kept);
        true
    }

    /// The states it keeps, and the room their lists hold, in bytes: a device test's
    /// (`tests::spare_states`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self) -> (usize, usize) {
        let spare = lock(&self.0);
        (spare.kept.len(), spare.bytes)
    }
}
