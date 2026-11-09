//! What a Metal resource carries beside its memory: its label and its purgeable state
//! (`MTLResource`'s `label` and `setPurgeableState:`).
//!
//! The purgeable state follows Metal's answers as measured (an Apple M3 Max and an M4): a query
//! answers the state, any other request answers the state before it and becomes the state. The
//! runtime never purges a volatile resource: Metal may discard its contents and need not, so this
//! is one of Metal's allowed behaviours, not a measured match. Metal keeps the state only of a
//! resource with an allocation of its own, one whose `allocatedSize` is at least
//! `PURGEABLE_ALLOCATION`: it answers `NonVolatile` to every request on a smaller one and keeps its
//! contents. Emptying a `Shared` or `Managed` buffer discards its bytes (they read zero after); a
//! `Private` buffer and every texture keep theirs, as Metal's did. Each buffer embeds one
//! [`ResourceAttributes`] and forwards its methods (`Buffer` here); a texture keeps its label and
//! state itself (`texture/attributes.rs`).

use super::buffer::Buffer;
use super::device::lock;
use super::error::{invalid, Error, Problem};
use super::values::StorageMode;
use std::sync::Mutex;

/// The least `allocatedSize` of a Metal resource whose purgeable state Metal keeps. A smaller
/// one shares an allocation with others: Metal answers `NonVolatile` to every request on it and
/// keeps its contents (buffers of 1, 4, 8 KiB and 16383 bytes, and textures of up to 12672 bytes,
/// did so on an Apple M3 Max and an M4; buffers of 16384 bytes and more, and textures of 16384
/// bytes and more, kept their state).
pub(super) const PURGEABLE_ALLOCATION: u64 = 16 * 1024;

/// The bytes Metal allocates for a texture of this shape, as the runtime models Apple's layout: for
/// each level, its extent in blocks of `block_extent` texels, each axis rounded up to a power of
/// two (the depth only for a `volume`, a 3D texture), times `block_bytes`; summed over the
/// `levels`, times the `layers` (a cube's faces each) and the `samples`. Compared with
/// `PURGEABLE_ALLOCATION`, it answers as Metal did for each of the 45 texture shapes the probes
/// measured (sizes, formats, mipmaps, arrays, cubes, 3D, 1D, multisample, compressed); Apple's own
/// sizes (`allocatedSize`) differ from it, by compression metadata and alignment.
pub(super) fn texture_allocation(
    block_extent: [u32; 2],
    block_bytes: u64,
    extent: [u32; 3],
    volume: bool,
    levels: u32,
    layers: u32,
    samples: u32,
) -> u64 {
    let blocks = |texels: u32, block: u32| u64::from(texels.div_ceil(block.max(1)).max(1));
    let level_bytes: u64 = (0..levels.max(1))
        .map(|level| {
            let [w, h, d] = extent.map(|axis| (axis >> level.min(31)).max(1));
            let depth = if volume { u64::from(d) } else { 1 };
            blocks(w, block_extent[0]).next_power_of_two()
                * blocks(h, block_extent[1]).next_power_of_two()
                * depth.next_power_of_two()
                * block_bytes
        })
        .sum();
    level_bytes * u64::from(layers.max(1)) * u64::from(samples.max(1))
}

/// `MTLPurgeableState`, with Metal's raw values.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PurgeableState {
    /// `MTLPurgeableStateKeepCurrent`: a query; the state does not change.
    KeepCurrent,
    /// `MTLPurgeableStateNonVolatile`: the contents may not be discarded.
    NonVolatile,
    /// `MTLPurgeableStateVolatile`: the contents may be discarded.
    Volatile,
    /// `MTLPurgeableStateEmpty`: the contents are discarded.
    Empty,
}

impl PurgeableState {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            PurgeableState::KeepCurrent => 1,
            PurgeableState::NonVolatile => 2,
            PurgeableState::Volatile => 3,
            PurgeableState::Empty => 4,
        }
    }

    /// The state of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<PurgeableState, Error> {
        Ok(match raw {
            1 => PurgeableState::KeepCurrent,
            2 => PurgeableState::NonVolatile,
            3 => PurgeableState::Volatile,
            4 => PurgeableState::Empty,
            raw => {
                return Err(invalid(
                    "from_raw",
                    Problem::UndefinedValue {
                        ty: "MTLPurgeableState",
                        raw,
                    },
                ))
            }
        })
    }
}

/// The purgeable state of a resource that is never purged: `set` of `requested` while the state
/// is `current`, as `(the answer, the new state)`. A query answers the state and keeps it; any
/// other request answers the state before it and becomes the state.
///
/// One request does not take: `Volatile` asked of an object that is already `Empty` leaves it
/// `Empty`. Metal's own rule, measured over every ordered pair of the three states on an
/// `MTLHeap` and on an `MTLTexture` on an Apple M3 Max under Metal 27.0.1 — `Empty` then
/// `Volatile` reads back `Empty`, where `Empty` then `NonVolatile` reads back `NonVolatile` and
/// `Volatile` then `Empty` reads back `Empty`. Contents that have been discarded cannot be made
/// discardable again; only `NonVolatile` brings the object back. The answer is still the state
/// before the call, so a caller sees the refusal in the read-back and not in the return.
pub(super) fn set_purgeable(
    current: PurgeableState,
    requested: PurgeableState,
) -> (PurgeableState, PurgeableState) {
    match (current, requested) {
        (current, PurgeableState::KeepCurrent) => (current, current),
        (PurgeableState::Empty, PurgeableState::Volatile) => {
            (PurgeableState::Empty, PurgeableState::Empty)
        }
        (current, requested) => (current, requested),
    }
}

/// A resource's label and purgeable state. A new resource has no label and is non-volatile.
#[derive(Debug)]
pub(super) struct ResourceAttributes {
    label: Mutex<Option<String>>,
    purgeable: Mutex<PurgeableState>,
}

impl ResourceAttributes {
    /// No label, non-volatile.
    pub(super) fn new() -> ResourceAttributes {
        ResourceAttributes {
            label: Mutex::new(None),
            purgeable: Mutex::new(PurgeableState::NonVolatile),
        }
    }

    /// The label and purgeable state now, for the resource carried to its device's next
    /// generation (`device/carry.rs`).
    pub(super) fn carried(&self) -> ResourceAttributes {
        ResourceAttributes {
            label: Mutex::new(lock(&self.label).clone()),
            purgeable: Mutex::new(*lock(&self.purgeable)),
        }
    }

    /// `setLabel:`.
    pub(super) fn set_label(&self, label: &str) {
        *lock(&self.label) = Some(label.to_owned());
    }

    /// `label`.
    pub(super) fn label(&self) -> Option<String> {
        lock(&self.label).clone()
    }

    /// `setPurgeableState:`: the state before the call, or the state now for `KeepCurrent`.
    pub(super) fn set_purgeable_state(&self, state: PurgeableState) -> PurgeableState {
        let mut current = lock(&self.purgeable);
        let (answer, next) = set_purgeable(*current, state);
        *current = next;
        answer
    }
}

impl Buffer {
    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        self.attributes().set_label(label);
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<String> {
        self.attributes().label()
    }

    /// `setPurgeableState:`: the state before the call, or the state now for `KeepCurrent`, as
    /// measured on Metal. The runtime never purges a volatile buffer (Metal may, and need not).
    /// `Empty` discards a `Shared` or `Managed` buffer's bytes, which then read zero, and keeps a
    /// `Private` buffer's, as Metal's did.
    ///
    /// A buffer shorter than `PURGEABLE_ALLOCATION` (16 KiB), whose `allocatedSize` is its
    /// length, shares an allocation on Metal: it answers `NonVolatile` to every call and keeps its
    /// bytes, as Metal's does. So does a no-copy buffer, as Metal's does, and a buffer over the
    /// caller's `VkBuffer`, whose memory the runtime never empties. A texture over the buffer
    /// changes none of its answers. Emptying a buffer the GPU is using leaves what either sees
    /// undefined, as in Metal.
    ///
    /// A buffer placed in a heap, whose state is its heap's, answers `NonVolatile` to every call
    /// and changes nothing: `Empty` keeps its bytes and its heap's state stays as it was, as plain
    /// Metal answers where its validation stops every call there, a query included. Fails only when
    /// zeroing a `Managed` buffer's bytes on a device without unified memory fails.
    pub fn set_purgeable_state(&self, state: PurgeableState) -> Result<PurgeableState, Error> {
        // Its parts held while the call reaches them, across a recovery (`Buffer::held_parts`).
        let _held = self.held_parts();
        if self.heap().is_some() {
            return Ok(PurgeableState::NonVolatile);
        }
        if self.caller_memory().is_some() || self.length() < PURGEABLE_ALLOCATION {
            return Ok(PurgeableState::NonVolatile);
        }
        let answer = self.attributes().set_purgeable_state(state);
        if state == PurgeableState::Empty && self.storage_mode() != StorageMode::Private {
            self.clear()?;
        }
        Ok(answer)
    }
}
