//! The records of an indirect command buffer of indexed draws under the identity domains: what a
//! kernel's encoding writes (the lowering of `render_command`'s setters, draw and reset) and what
//! a consumer's execution reads, one `VkIndirectCommandsLayoutEXT` sequence per command.
//!
//! The buffer's memory starts with a header of [`IndirectDrawLayout::HEADER_BYTES`]: its capacity, the vertex and the
//! fragment buffer slots each command holds, the stride (a `u32` each), and the index address a
//! reset command binds (a `u64`). Command `i` starts at `HEADER_BYTES + i * stride`, every field at
//! the offset its [`IndirectDrawLayout`] names: each slot's word (the `SlotGuestBase` form of the
//! buffer's address, the address less 2^31), the index buffer token's data
//! (`VkBindIndexBufferIndirectCommandEXT`: the indices' address, their bytes and their
//! `VkIndexType`), four bytes the sequence index token takes its place in, and the indexed draw's
//! `VkDrawIndexedIndirectCommand`. A buffer's `gpuResourceID` is the address of its header.

/// The index token's data, the sequence index's place and the draw's arguments.
const TAIL_BYTES: u32 = 16 + 4 + 20;

/// Where each field of a command lies, for a buffer of `vertex_slots` vertex and
/// `fragment_slots` fragment buffer slots per command (Metal's `maxVertexBufferBindCount` and
/// `maxFragmentBufferBindCount`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndirectDrawLayout {
    vertex_slots: u32,
    fragment_slots: u32,
}

impl IndirectDrawLayout {
    /// The header's bytes: the capacity at [`Self::CAPACITY_AT`], the vertex slots at
    /// [`Self::VERTEX_SLOTS_AT`], the fragment slots at [`Self::FRAGMENT_SLOTS_AT`], the stride at [`Self::STRIDE_AT`]
    /// (a `u32` each) and the reset index address at [`Self::RESET_INDEX_AT`] (a `u64`).
    pub const HEADER_BYTES: u32 = 24;
    /// The header's capacity: the commands the buffer holds.
    pub const CAPACITY_AT: u32 = 0;
    /// The header's vertex buffer slots per command.
    pub const VERTEX_SLOTS_AT: u32 = 4;
    /// The header's fragment buffer slots per command.
    pub const FRAGMENT_SLOTS_AT: u32 = 8;
    /// The header's stride: the bytes of one command.
    pub const STRIDE_AT: u32 = 12;
    /// The header's reset index address: a valid device address, a multiple of 4, that a reset
    /// command's index buffer token binds with no bytes.
    pub const RESET_INDEX_AT: u32 = 16;
    /// The bytes of a slot's word.
    pub const SLOT_BYTES: u32 = 8;
    /// The most buffer slots of a stage a command holds: Metal's buffer indices below 31.
    pub const MAX_SLOTS: u32 = 31;
    /// `VkIndexType`'s `VK_INDEX_TYPE_UINT16` and `VK_INDEX_TYPE_UINT32`, as the index token holds them.
    pub const INDEX_UINT16: u32 = 0;
    /// See [`Self::INDEX_UINT16`].
    pub const INDEX_UINT32: u32 = 1;

    /// The layout of commands holding `vertex_slots` and `fragment_slots` slots; `None` past
    /// [`Self::MAX_SLOTS`] of either.
    pub fn new(vertex_slots: u32, fragment_slots: u32) -> Option<IndirectDrawLayout> {
        (vertex_slots <= Self::MAX_SLOTS && fragment_slots <= Self::MAX_SLOTS).then_some(
            IndirectDrawLayout {
                vertex_slots,
                fragment_slots,
            },
        )
    }

    /// The vertex buffer slots a command holds.
    pub fn vertex_slots(self) -> u32 {
        self.vertex_slots
    }

    /// The fragment buffer slots a command holds.
    pub fn fragment_slots(self) -> u32 {
        self.fragment_slots
    }

    /// The bytes of one command, a multiple of 8, at most 536.
    pub fn stride(self) -> u32 {
        self.index() + TAIL_BYTES
    }

    /// The offset in a command of vertex slot `slot`'s word.
    pub fn vertex_slot(self, slot: u32) -> u32 {
        Self::SLOT_BYTES * slot
    }

    /// The offset in a command of fragment slot `slot`'s word.
    pub fn fragment_slot(self, slot: u32) -> u32 {
        Self::SLOT_BYTES * (self.vertex_slots + slot)
    }

    /// The offset in a command of the index buffer token's data, a multiple of 8.
    pub fn index(self) -> u32 {
        Self::SLOT_BYTES * (self.vertex_slots + self.fragment_slots)
    }

    /// The offset in a command of the sequence index token's place, which holds nothing.
    pub fn sequence(self) -> u32 {
        self.index() + 16
    }

    /// The offset in a command of the indexed draw's `VkDrawIndexedIndirectCommand`.
    pub fn draw(self) -> u32 {
        self.index() + 20
    }

    /// The offset in the buffer's memory of command `command`.
    pub fn command(self, command: u32) -> u64 {
        u64::from(Self::HEADER_BYTES) + u64::from(command) * u64::from(self.stride())
    }

    /// The bytes of a buffer of `capacity` commands.
    pub fn bytes(self, capacity: u32) -> u64 {
        self.command(capacity)
    }

    /// The header of a buffer of `capacity` commands whose reset commands bind
    /// `reset_index_address`.
    pub fn header(
        self,
        capacity: u32,
        reset_index_address: u64,
    ) -> [u8; Self::HEADER_BYTES as usize] {
        let mut header = [0; Self::HEADER_BYTES as usize];
        let words = [
            capacity,
            self.vertex_slots,
            self.fragment_slots,
            self.stride(),
        ];
        for (at, word) in words.into_iter().enumerate() {
            header[4 * at..4 * at + 4].copy_from_slice(&word.to_le_bytes());
        }
        let at = Self::RESET_INDEX_AT as usize;
        header[at..at + 8].copy_from_slice(&reset_index_address.to_le_bytes());
        header
    }

    /// A command as a reset leaves it: every slot's word `slot_word`, the index token binding
    /// `reset_index_address` with no bytes, and a draw of no indices and no instances, which draws
    /// nothing. A kernel's reset writes the index token and the draw alone.
    pub fn reset_command(self, slot_word: u64, reset_index_address: u64) -> Vec<u8> {
        let mut command = vec![0; self.stride() as usize];
        for slot in 0..self.vertex_slots + self.fragment_slots {
            let at = (Self::SLOT_BYTES * slot) as usize;
            command[at..at + 8].copy_from_slice(&slot_word.to_le_bytes());
        }
        let at = self.index() as usize;
        command[at..at + 8].copy_from_slice(&reset_index_address.to_le_bytes());
        command[at + 12..at + 16].copy_from_slice(&Self::INDEX_UINT32.to_le_bytes());
        command
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seven vertex and seven fragment slots: a command of 152 bytes, its fields in order and
    /// every `u64` 8-aligned.
    #[test]
    fn a_command_places_its_fields_in_order() {
        let layout = IndirectDrawLayout::new(7, 7).expect("a layout");
        assert_eq!(layout.stride(), 152);
        assert_eq!(
            (
                layout.vertex_slot(6),
                layout.fragment_slot(0),
                layout.fragment_slot(6)
            ),
            (48, 56, 104)
        );
        assert_eq!(
            (layout.index(), layout.sequence(), layout.draw()),
            (112, 128, 132)
        );
        assert_eq!(layout.command(0), 24);
        assert_eq!(layout.command(3), 24 + 3 * 152);
        assert_eq!(layout.bytes(2), 24 + 2 * 152);
        let widest =
            IndirectDrawLayout::new(IndirectDrawLayout::MAX_SLOTS, IndirectDrawLayout::MAX_SLOTS)
                .expect("a layout");
        assert_eq!(widest.stride(), 536);
        assert!(IndirectDrawLayout::new(IndirectDrawLayout::MAX_SLOTS + 1, 0).is_none());
        assert!(IndirectDrawLayout::new(0, IndirectDrawLayout::MAX_SLOTS + 1).is_none());
        let none = IndirectDrawLayout::new(0, 0).expect("a layout");
        assert_eq!((none.stride(), none.index(), none.draw()), (40, 0, 20));
    }

    /// The header holds the capacity, the counts, the stride and the reset index address; a
    /// reset command holds the slot word in every slot, the index token at that address with no
    /// bytes, and a draw of nothing.
    #[test]
    fn the_header_and_a_reset_command_hold_their_words() {
        let layout = IndirectDrawLayout::new(2, 1).expect("a layout");
        let header = layout.header(9, 0x1234_5678_9abc_def0);
        let word = |at: u32| u32::from_le_bytes(header[at as usize..][..4].try_into().unwrap());
        assert_eq!(
            (
                word(IndirectDrawLayout::CAPACITY_AT),
                word(IndirectDrawLayout::VERTEX_SLOTS_AT),
                word(IndirectDrawLayout::FRAGMENT_SLOTS_AT),
                word(IndirectDrawLayout::STRIDE_AT)
            ),
            (9, 2, 1, 64)
        );
        let at = IndirectDrawLayout::RESET_INDEX_AT as usize;
        assert_eq!(
            u64::from_le_bytes(header[at..at + 8].try_into().unwrap()),
            0x1234_5678_9abc_def0
        );
        let command = layout.reset_command(0xaaaa_bbbb_cccc_dddd, 0x40);
        assert_eq!(command.len(), 64);
        for slot in 0..3 {
            let at = (8 * slot) as usize;
            assert_eq!(
                u64::from_le_bytes(command[at..at + 8].try_into().unwrap()),
                0xaaaa_bbbb_cccc_dddd
            );
        }
        let at = layout.index() as usize;
        assert_eq!(
            u64::from_le_bytes(command[at..at + 8].try_into().unwrap()),
            0x40
        );
        assert_eq!(&command[at + 8..at + 12], &[0; 4]);
        assert_eq!(
            u32::from_le_bytes(command[at + 12..at + 16].try_into().unwrap()),
            IndirectDrawLayout::INDEX_UINT32
        );
        assert!(command[layout.sequence() as usize..]
            .iter()
            .all(|&b| b == 0));
    }
}
