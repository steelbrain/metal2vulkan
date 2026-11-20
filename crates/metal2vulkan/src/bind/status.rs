//! `ExecutionStatus` buffer layout: word indices and status bits.
//!
//! One 16-byte buffer per command buffer, shared by every plan in it. The consumer initializes it
//! with [`ExecutionStatus::initial`] before the command buffer's first use and reads it back with
//! [`ExecutionStatus::decode`] after the last one (a `SHADER_WRITE -> HOST_READ` barrier first);
//! the resident-set miss counter, the device replay's preprocess, the index
//! bounds pass, an absent argument's read, a mesh draw's arguments pass, a sample
//! through a cubic argument sampler, and a sequentially consistent fence write the
//! words named here.

/// The `ExecutionStatus` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionStatus;

impl ExecutionStatus {
    /// Size of the buffer in bytes.
    pub const BYTES: usize = 16;
    /// Size of the buffer in 32-bit words: up to the last of [`StatusWord::ALL`].
    pub const WORDS: usize = 4;

    /// The buffer's initial bytes: every word 0.
    pub const fn initial() -> [u8; ExecutionStatus::BYTES] {
        [0; ExecutionStatus::BYTES]
    }

    /// What the buffer's bytes report after the command buffer ran. The consumer maps a replay
    /// fault, an index pool overflow, an absent argument read, a mesh grid Metal's GPU
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
            index_pool_overflow: bits.contains(StatusBits::INDEX_POOL_OVERFLOW),
            absent_argument_read: bits.contains(StatusBits::ABSENT_ARGUMENT_READ),
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
    /// An index bounds pass found no room in its command buffer's index pool for a draw's
    /// indices: that draw drew nothing.
    pub index_pool_overflow: bool,
    /// A function read an argument its pipeline's function constants removed, which Metal gives
    /// no value (`RuntimeFact::AbsentArgumentRead`): the read yielded zero, so no result of the
    /// command buffer is Metal's.
    pub absent_argument_read: bool,
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
    /// Every word, in index order.
    pub const ALL: [StatusWord; 4] = [
        StatusWord::Status,
        StatusWord::ResidentMisses,
        StatusWord::ReplayFaults,
        StatusWord::SeqCstFences,
    ];

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
    /// An index bounds pass found no room in the command buffer's index pool for a draw's
    /// indices, and the draw drew nothing (`bind::IndexBoundsPass`).
    pub const INDEX_POOL_OVERFLOW: StatusBits = StatusBits(1 << 2);
    /// A function read an argument its pipeline's function constants removed, a read Metal
    /// gives no value, and took zero for it (`RuntimeFact::AbsentArgumentRead`).
    pub const ABSENT_ARGUMENT_READ: StatusBits = StatusBits(1 << 3);
    /// A function sampled, gathered, or queried through a sampler an argument buffer holds whose
    /// min or mag filter is `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): its result is the device's through a nearest
    /// placeholder, not Metal's.
    pub const CUBIC_ARGUMENT_SAMPLER: StatusBits = StatusBits(1 << 4);
    /// An indirect mesh draw's arguments pass met a grid that Metal draws and the device cannot
    /// launch: a mesh grid past the mesh stage's limits, an unfolded object grid past the task
    /// stage's, or a folded one of more pieces than `MeshArguments::OBJECT_PIECES` (only under task
    /// limits below Vulkan's required least); the draw drew nothing (`bind::MeshArguments`).
    pub const MESH_GRID_PAST_LIMITS: StatusBits = StatusBits(1 << 5);
    /// An indirect mesh draw's arguments pass met an object grid of more threadgroups than
    /// `bind::METAL_OBJECT_GRID_TOTAL`, which Metal's GPU discards; the draw drew nothing
    /// (`bind::MeshArguments`).
    pub const MESH_GRID_DISCARDED: StatusBits = StatusBits(1 << 6);
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

impl ReplayFaults {
    /// The record storage's header holds the producer's sticky encoding fault: an encoder
    /// operation the descriptor policy refused.
    pub const ENCODING: ReplayFaults = ReplayFaults(1 << 0);
    /// The command's action is not one the replay runs: a draw in a replay of compute commands,
    /// or a draw of a primitive type a replay of draws does not record
    /// (`DrawReplay::primitive_types`).
    pub const ACTION: ReplayFaults = ReplayFaults(1 << 1);
    /// The command has no pipeline from its state source, or sets one while it inherits it.
    pub const PIPELINE: ReplayFaults = ReplayFaults(1 << 2);
    /// The command sets a kernel buffer while it inherits buffers, or binds one with a dynamic
    /// attribute stride, which no replayed template reads.
    pub const BUFFERS: ReplayFaults = ReplayFaults(1 << 3);
    /// No registered variant has the command's pipeline and threadgroup lengths and the
    /// threadgroup size of one of its regions: the command's size for its whole threadgroups, an
    /// edge's remaining threads for a thread dispatch whose grid is not a multiple of it. A
    /// threadgroup axis of 0 or above 1,024 has no variant.
    pub const VARIANT: ReplayFaults = ReplayFaults(1 << 4);
    // Bit 5 (the `NON_UNIFORM`, a ragged thread grid) is not reported: such a grid runs its edge
    // regions on the variants of their sizes.
    /// The command dispatches threadgroups whose grid exceeds `2^32 - 1` threads on some axis
    /// (Metal's `threads_per_grid` is 32 bits). A group count past `maxComputeWorkGroupCount`
    /// splits into regions instead. On a flat variant also a grid of more than `2^64 - 1` threads,
    /// or one whose threadgroups of the variant's size no group count within
    /// `maxComputeWorkGroupCount` holds.
    pub const GROUP_COUNT: ReplayFaults = ReplayFaults(1 << 6);
    // Bit 7 (a kernel buffer's address in no allocation of the resident set) is not reported:
    // such a slot is unbound and counted as a resident miss (`StatusBits::RESIDENT_MISS`), and
    // the command runs, as the host's step runs it.
    /// The record is one `CommandRecord::decode` rejects (an action no record encodes, a barrier
    /// flag other than 0 or 1), or sets an imageblock size with one side 0.
    pub const RECORD: ReplayFaults = ReplayFaults(1 << 8);
    /// The command's index is not below the capacity the record storage's header states: the
    /// execution's capacity is not the storage's, and the record is not read. The runtime also
    /// reports it for an execution range read from a buffer that Metal's GPU faults on (a
    /// location past the buffer's size, or a range past `2^32 - 1`), which runs no command.
    pub const RANGE: ReplayFaults = ReplayFaults(1 << 9);
    /// The command has more regions (group-count chunks of at most `maxComputeWorkGroupCount`, and
    /// a thread dispatch's edges) than the execution records per command
    /// (`ReplayExecution::regions`).
    pub const REGIONS: ReplayFaults = ReplayFaults(1 << 10);
    /// The command of a replay of draws sets its barrier: the replay draws every command in one
    /// render pass instance, which no barrier splits on the GPU.
    pub const BARRIER: ReplayFaults = ReplayFaults(1 << 11);
    /// A replay of draws merged into one draw (`DrawReplay::merged`) whose commands' vertices,
    /// over their instances, reach `2^32` times `MERGED_CHUNK` (above `2^62`), which its two
    /// draws cannot hold: no command of the execution draws.
    pub const VERTICES: ReplayFaults = ReplayFaults(1 << 12);

    /// Every bit in `other` is set in `self`.
    pub const fn contains(self, other: ReplayFaults) -> bool {
        self.0 & other.0 == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_distinct_and_inside_the_buffer() {
        for (i, w) in StatusWord::ALL.iter().enumerate() {
            if let Some(next) = StatusWord::ALL.get(i + 1) {
                assert!(w.index() < next.index(), "{w:?}");
            }
            assert!((w.byte_offset() as usize) + 4 <= ExecutionStatus::BYTES);
        }
        let last = StatusWord::ALL[StatusWord::ALL.len() - 1];
        assert_eq!(last.index() as usize + 1, ExecutionStatus::WORDS);
        assert_eq!(ExecutionStatus::WORDS * 4, ExecutionStatus::BYTES);
    }

    #[test]
    fn status_bits_are_distinct() {
        let bits = [
            StatusBits::RESIDENT_MISS,
            StatusBits::REPLAY_FAULT,
            StatusBits::INDEX_POOL_OVERFLOW,
            StatusBits::ABSENT_ARGUMENT_READ,
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
        // The bits are the low ones, none left unused between them.
        assert_eq!(all, (1 << bits.len()) - 1);
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
        let bits = StatusBits::INDEX_POOL_OVERFLOW.0 | StatusBits::RESIDENT_MISS.0;
        bytes[0..4].copy_from_slice(&bits.to_le_bytes());
        bytes[4..8].copy_from_slice(&7u32.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.index_pool_overflow && r.resident_miss);
        assert_eq!(r.resident_misses, 7);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::INDEX_POOL_OVERFLOW.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.index_pool_overflow && !r.resident_miss);
        assert_eq!(r.resident_misses, 0);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::REPLAY_FAULT.0.to_le_bytes());
        let faults = ReplayFaults::VARIANT.0 | ReplayFaults::RECORD.0;
        bytes[8..12].copy_from_slice(&faults.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.replay_fault && !r.index_pool_overflow);
        assert!(r.replay_faults.contains(ReplayFaults::VARIANT));
        assert!(r.replay_faults.contains(ReplayFaults::RECORD));
        assert!(!r.replay_faults.contains(ReplayFaults::ENCODING));
        assert!(!r.index_pool_overflow);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::INDEX_POOL_OVERFLOW.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.index_pool_overflow && !r.replay_fault);
        assert!(!r.absent_argument_read);
        bytes = [0u8; ExecutionStatus::BYTES];
        bytes[0..4].copy_from_slice(&StatusBits::ABSENT_ARGUMENT_READ.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(!r.cubic_argument_sampler);
        let mut cubic = [0u8; ExecutionStatus::BYTES];
        cubic[0..4].copy_from_slice(&StatusBits::CUBIC_ARGUMENT_SAMPLER.0.to_le_bytes());
        let c = ExecutionStatus::decode(&cubic);
        assert!(c.cubic_argument_sampler && !c.absent_argument_read);
        assert!(r.absent_argument_read && !r.index_pool_overflow);
    }

    #[test]
    fn replay_fault_bits_are_distinct() {
        let bits = [
            ReplayFaults::ENCODING,
            ReplayFaults::ACTION,
            ReplayFaults::PIPELINE,
            ReplayFaults::BUFFERS,
            ReplayFaults::VARIANT,
            ReplayFaults::GROUP_COUNT,
            ReplayFaults::RECORD,
            ReplayFaults::RANGE,
            ReplayFaults::REGIONS,
        ];
        let mut all = 0u32;
        for b in bits {
            assert_eq!(b.0.count_ones(), 1);
            assert_eq!(all & b.0, 0);
            all |= b.0;
        }
    }
}
