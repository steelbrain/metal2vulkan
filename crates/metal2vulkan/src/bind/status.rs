//! `ExecutionStatus` buffer layout: word indices and status bits.
//!
//! One 16-byte buffer per command buffer, shared by every plan in it. The consumer initializes it
//! with [`ExecutionStatus::initial`] before the command buffer's first use and reads it back with
//! [`ExecutionStatus::decode`] after the last one (a `SHADER_WRITE -> HOST_READ` barrier first);
//! the resident-set miss counter, the index
//! bounds pass, an absent argument's read, a mesh draw's arguments pass, a sample
//! through a cubic argument sampler, and a sequentially consistent fence write the
//! words named here.

/// The `ExecutionStatus` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionStatus;

impl ExecutionStatus {
    /// Size of the buffer in bytes.
    pub const BYTES: usize = 16;

    /// The buffer's initial bytes: every word 0.
    pub const fn initial() -> [u8; ExecutionStatus::BYTES] {
        [0; ExecutionStatus::BYTES]
    }

    /// What the buffer's bytes report after the command buffer ran. The consumer maps a replay
    /// fault, a mesh grid Metal's GPU
    /// discards or the device cannot launch, and a cubic argument sampler's sample
    /// to a command-buffer execution error, and treats resident-set misses by its own policy.
    pub fn decode(bytes: &[u8; ExecutionStatus::BYTES]) -> StatusReport {
        let word = |w: StatusWord| {
            let at = w.byte_offset() as usize;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let bits = StatusBits(word(StatusWord::Status));
        StatusReport {
            resident_miss: bits.contains(StatusBits::RESIDENT_MISS),
            resident_misses: word(StatusWord::ResidentMisses),
            replay_fault: bits.contains(StatusBits::REPLAY_FAULT),
            replay_faults: ReplayFaults(word(StatusWord::ReplayFaults)),
            cubic_argument_sampler: bits.contains(StatusBits::CUBIC_ARGUMENT_SAMPLER),
            mesh_grid_discarded: bits.contains(StatusBits::MESH_GRID_DISCARDED),
            mesh_grid_past_limits: bits.contains(StatusBits::MESH_GRID_PAST_LIMITS),
        }
    }
}

/// What an `ExecutionStatus` buffer reports.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StatusReport {
    /// An access missed the resident set.
    pub resident_miss: bool,
    /// The number of resident-set misses counted (a `u32` the device increments, wrapping after
    /// `2^32` misses).
    pub resident_misses: u32,
    /// The device replay's preprocess met a command it does not replay: that command dispatched
    /// nothing. [`StatusReport::replay_faults`] says why.
    pub replay_fault: bool,
    /// Why the device replay's preprocess refused commands: the union of their reasons.
    pub replay_faults: ReplayFaults,
    /// A function sampled through an argument buffer's sampler whose min or mag filter is
    /// `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): no result of the command buffer is then Metal's.
    pub cubic_argument_sampler: bool,
    /// An indirect mesh draw's arguments pass met an object grid that Metal's GPU discards (`StatusBits::MESH_GRID_DISCARDED`): that draw drew
    /// nothing, and the command buffer fails as Metal's does (`MTLCommandBufferErrorInternal`).
    pub mesh_grid_discarded: bool,
    /// An indirect mesh draw's arguments pass met a grid that Metal draws and the device cannot launch (`StatusBits::MESH_GRID_PAST_LIMITS`):
    /// that draw drew nothing, and the command buffer fails (not Metal's failure).
    pub mesh_grid_past_limits: bool,
}

/// A word of the `ExecutionStatus` buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatusWord {
    /// Status bits ([`StatusBits`]). Initial 0.
    Status,
    /// Resident-miss count. Initial 0.
    ResidentMisses,
    /// The device replay's fault reasons ([`ReplayFaults`]). Initial 0.
    ReplayFaults,
    /// The word every sequentially consistent fence of the command buffer increments with an
    /// acquire-release atomic at the fence's scope (`lower::sync::barrier::seq_cst_order`), so
    /// that the fences of one command buffer are totally ordered, as Metal's are. Its value means
    /// nothing; the decode does not read it. Initial 0.
    SeqCstFences,
}

impl StatusWord {
    /// The word's index in the buffer.
    pub const fn index(self) -> u32 {
        match self {
            StatusWord::Status => 0,
            StatusWord::ResidentMisses => 1,
            StatusWord::ReplayFaults => 2,
            StatusWord::SeqCstFences => 3,
        }
    }

    /// The word's byte offset in the buffer.
    pub const fn byte_offset(self) -> u32 {
        self.index() * 4
    }
}

/// Bits of [`StatusWord::Status`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StatusBits(pub u32);

impl StatusBits {
    /// An access missed the resident set.
    pub const RESIDENT_MISS: StatusBits = StatusBits(1 << 0);
    /// The device replay's preprocess refused a command; [`StatusWord::ReplayFaults`] holds why.
    pub const REPLAY_FAULT: StatusBits = StatusBits(1 << 1);
    /// A function sampled, gathered, or queried through a sampler an argument buffer holds whose
    /// min or mag filter is `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): its result is the device's through a nearest
    /// placeholder, not Metal's.
    pub const CUBIC_ARGUMENT_SAMPLER: StatusBits = StatusBits(1 << 3);
    /// An indirect mesh draw's arguments pass met a grid that Metal draws and the device cannot
    /// launch: a mesh grid past the mesh stage's limits, an unfolded object grid past the task
    /// stage's, or a folded one of more pieces than `MeshArguments::OBJECT_PIECES` (only under task
    /// limits below Vulkan's required least); the draw drew nothing (`bind::MeshArguments`).
    pub const MESH_GRID_PAST_LIMITS: StatusBits = StatusBits(1 << 4);
    /// An indirect mesh draw's arguments pass met an object grid of more threadgroups than
    /// `bind::METAL_OBJECT_GRID_TOTAL`, which Metal's GPU discards; the draw drew nothing
    /// (`bind::MeshArguments`).
    pub const MESH_GRID_DISCARDED: StatusBits = StatusBits(1 << 5);
    /// Every bit in `other` is set in `self`.
    pub const fn contains(self, other: StatusBits) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Why the device replay's preprocess refused a command, the bits of [`StatusWord::ReplayFaults`].
/// A refused command dispatches nothing; every other command of the range still runs, so a consumer
/// that reads a fault reports the command buffer's execution as failed, as Metal reports an
/// encoding fault, and may replay the range on the host instead (`CommandBuffer::decode`, which
/// names the command).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReplayFaults(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_distinct_and_fill_the_buffer() {
        let words = [
            StatusWord::Status,
            StatusWord::ResidentMisses,
            StatusWord::ReplayFaults,
            StatusWord::SeqCstFences,
        ];
        for (i, w) in words.iter().enumerate() {
            assert_eq!(w.index() as usize, i, "{w:?}");
            assert_eq!(w.byte_offset(), 4 * w.index(), "{w:?}");
        }
        assert_eq!(words.len() * 4, ExecutionStatus::BYTES);
    }

    #[test]
    fn status_bits_are_distinct() {
        let bits = [
            StatusBits::RESIDENT_MISS,
            StatusBits::REPLAY_FAULT,
            StatusBits::CUBIC_ARGUMENT_SAMPLER,
            StatusBits::MESH_GRID_PAST_LIMITS,
            StatusBits::MESH_GRID_DISCARDED,
        ];
        let mut all = 0u32;
        for b in bits {
            assert_eq!(b.0.count_ones(), 1);
            assert_eq!(all & b.0, 0);
            all |= b.0;
        }
        assert!(StatusBits(all).contains(StatusBits::REPLAY_FAULT));
        assert!(!StatusBits::RESIDENT_MISS.contains(StatusBits::REPLAY_FAULT));
    }

    /// The initial bytes are all zero, and `decode` of them reports nothing.
    #[test]
    fn initial_bytes_are_zero_and_decode_clean() {
        let bytes = ExecutionStatus::initial();
        assert!(bytes.iter().all(|&b| b == 0));
        assert_eq!(ExecutionStatus::decode(&bytes), StatusReport::default());
    }

    #[test]
    fn decode_reports_every_bit_and_counter() {
        let mut bytes = [0u8; ExecutionStatus::BYTES];
        let bits = StatusBits::CUBIC_ARGUMENT_SAMPLER.0 | StatusBits::RESIDENT_MISS.0;
        bytes[0..4].copy_from_slice(&bits.to_le_bytes());
        bytes[4..8].copy_from_slice(&7u32.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.cubic_argument_sampler && r.resident_miss);
        assert_eq!(r.resident_misses, 7);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::CUBIC_ARGUMENT_SAMPLER.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.cubic_argument_sampler && !r.resident_miss);
        assert_eq!(r.resident_misses, 0);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::REPLAY_FAULT.0.to_le_bytes());
        bytes[8..12].copy_from_slice(&0x41u32.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.replay_fault && !r.cubic_argument_sampler);
        assert_eq!(r.replay_faults, ReplayFaults(0x41));
        let mut cubic = [0u8; ExecutionStatus::BYTES];
        cubic[0..4].copy_from_slice(&StatusBits::CUBIC_ARGUMENT_SAMPLER.0.to_le_bytes());
        let c = ExecutionStatus::decode(&cubic);
        assert!(c.cubic_argument_sampler && !c.replay_fault);
    }
}
