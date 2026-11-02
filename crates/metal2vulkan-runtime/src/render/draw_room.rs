//! What a render encoder's draws gather into, kept by the device from one encoder to the next.
//!
//! A pass's draws gather, into its first instance, the shadowed textures they use and the bytes of
//! the buffers they use (`render/shadows.rs`), into it and the encoder the rectangles they reach
//! (`clear_reach::DrawnArea`), and into the encoder, the bytes its fragment stages wrote and read
//! (`render/pass_start_reads.rs`): lists made at a pass's first draws.
//! Where each was dropped as its pass ended and made again by the next pass's draws, an ended
//! encoder now gives them to the device emptied, with their room, and the next encoder it opens
//! takes them: so a pass of as many buffers as an earlier one's makes and frees none of them.
//! What they gather, and what their pass ends with, is the same either way: each is emptied
//! before it is kept.

use super::pass_start_reads::PassStartReads;
use super::shadows::InstanceShadows;
use crate::clear_reach::DrawnArea;
use crate::device::lock;
use std::sync::Mutex;

/// The most rooms a device keeps: one per encoder open at once beyond it starts with none, as
/// every encoder did before.
const SPARE_ROOMS: usize = 4;

/// The most buffers' and images' entries a kept room holds room for: a room grown past it by a
/// pass that used more goes, so a device keeps no more than a few passes' worth.
const ROOM_ENTRIES: usize = 64;

/// The room of one render encoder's gathering: its first instance's shadowed textures and bytes
/// ([`InstanceShadows`]) and drawn area ([`DrawnArea`]), and its pass's drawn area and fragment
/// writes and reads ([`PassStartReads`]).
#[derive(Default)]
pub(super) struct DrawRoom {
    pub(super) shadows: InstanceShadows,
    pub(super) first_drawn: DrawnArea,
    pub(super) drawn: DrawnArea,
    pub(super) pass_start: PassStartReads,
}

/// The rooms of a device's render encoders whose passes ended, which the next encoders it opens
/// take (the module docs): at most [`SPARE_ROOMS`], the latest given taken first.
#[derive(Default)]
pub(crate) struct SpareDrawRooms(Mutex<Vec<DrawRoom>>);

impl SpareDrawRooms {
    /// The room a render encoder opening gathers into: the latest an encoder gave back, else a
    /// new one.
    pub(super) fn take(&self) -> DrawRoom {
        let kept = lock(&self.0).pop();
        kept.unwrap_or_else(|| {
            #[cfg(test)]
            made::note();
            DrawRoom::default()
        })
    }

    /// Keep `room`, an ended encoder's, emptied, where its room is small enough
    /// ([`ROOM_ENTRIES`]) and fewer than [`SPARE_ROOMS`] are kept; else it goes.
    pub(super) fn give(&self, mut room: DrawRoom) {
        let rooms = [
            room.shadows.room(),
            room.first_drawn.room(),
            room.drawn.room(),
            room.pass_start.room(),
        ];
        if rooms.iter().any(|&entries| entries > ROOM_ENTRIES) {
            return;
        }
        room.shadows.clear();
        room.first_drawn.clear();
        room.drawn.clear();
        room.pass_start.clear();
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_ROOMS {
            spare.push(room);
        }
    }

    /// How many rooms it keeps.
    #[cfg(test)]
    fn kept(&self) -> usize {
        lock(&self.0).len()
    }
}

/// The rooms made new on this thread, where no ended encoder's was kept ([`SpareDrawRooms::take`]):
/// what a test reads to tell that a pass took the room an earlier one left.
#[cfg(test)]
pub(crate) mod made {
    use std::cell::Cell;

    thread_local! {
        static MADE: Cell<u64> = const { Cell::new(0) };
    }

    pub(super) fn note() {
        MADE.with(|made| made.set(made.get() + 1));
    }

    /// The rooms made so far.
    pub(crate) fn so_far() -> u64 {
        MADE.with(Cell::get)
    }
}

#[cfg(test)]
mod tests {
    use super::{made, DrawRoom, SpareDrawRooms, ROOM_ENTRIES, SPARE_ROOMS};
    use crate::resource::ResourceKey;
    use crate::tracker::{Access, Use, UseRange};
    use ash::vk;

    fn read(key: ResourceKey, range: std::ops::Range<u64>) -> Use<ResourceKey> {
        let access = Access::new(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        Use::reads(key, true, UseRange::Bytes(range), access)
    }

    /// A room given back is taken again, emptied, before a new one is made; at most
    /// [`SPARE_ROOMS`] are kept, and one grown past [`ROOM_ENTRIES`] goes.
    #[test]
    fn a_room_given_back_is_taken_again_empty() {
        let spares = SpareDrawRooms::default();
        let before = made::so_far();
        let mut room = spares.take();
        assert_eq!(made::so_far(), before + 1, "none kept: a new one");
        let key = ResourceKey::new();
        room.shadows.add(&[], &[read(key, 0..16)]);
        room.pass_start.add(&[read(key, 0..16)]);
        room.drawn.add(&[vk::Rect2D::default().extent(vk::Extent2D {
            width: 4,
            height: 4,
        })]);
        spares.give(room);
        assert_eq!(spares.kept(), 1);
        let again = spares.take();
        assert_eq!(made::so_far(), before + 1, "the kept one taken");
        assert_eq!(again.shadows.room(), 1, "its list kept");
        assert!(again.shadows.is_empty() && again.drawn.bounds().is_none());
        assert!(!again.pass_start.fragment_wrote());
        for _ in 0..SPARE_ROOMS + 2 {
            spares.give(DrawRoom::default());
        }
        assert_eq!(spares.kept(), SPARE_ROOMS);
        // A room past the entries kept goes.
        let spares = SpareDrawRooms::default();
        let mut grown = DrawRoom::default();
        let keys: Vec<Use<ResourceKey>> = (0..=ROOM_ENTRIES)
            .map(|_| read(ResourceKey::new(), 0..4))
            .collect();
        grown.shadows.add(&[], &keys);
        spares.give(grown);
        assert_eq!(spares.kept(), 0);
    }
}
