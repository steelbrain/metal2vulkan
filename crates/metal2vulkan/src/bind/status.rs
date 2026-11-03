//! `ExecutionStatus` buffer layout: word indices and status bits.
//!
//! One 32-byte buffer per command buffer, shared by every plan in it. The consumer initializes it
//! with [`ExecutionStatus::initial`] before the command buffer's first use and reads it back with
//! [`ExecutionStatus::decode`] after the last one (a `SHADER_WRITE -> HOST_READ` barrier first);
//! the resident-set miss counter, the indirect clamp, the device replay's preprocess, the index
//! bounds pass, an absent argument's read, the mesh and tessellation scans, a sample through a
//! cubic argument sampler, and a sequentially consistent fence write the words
//! named here.

/// The `ExecutionStatus` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionStatus;

impl ExecutionStatus {
    /// Size of the buffer in bytes.
    pub const BYTES: usize = 32;
    /// Size of the buffer in 32-bit words: [`StatusWord::ALL`].
    pub const WORDS: usize = 8;

    /// The buffer's initial bytes: every word 0.
    pub const fn initial() -> [u8; ExecutionStatus::BYTES] {
        [0; ExecutionStatus::BYTES]
    }

    /// What the buffer's bytes report after the command buffer ran. The consumer maps `overflow`, a
    /// replay fault, an index pool overflow, an absent argument read, a mesh records fault, a mesh
    /// grid Metal's GPU discards or the device cannot launch, and a cubic argument sampler's sample
    /// to a command-buffer execution error, and treats resident-set misses by its own policy.
    pub fn decode(bytes: &[u8; ExecutionStatus::BYTES]) -> StatusReport {
        let word = |w: StatusWord| {
            let at = w.byte_offset() as usize;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let bits = StatusBits(word(StatusWord::Status));
        let tessellation_first = bits.contains(StatusBits::TESSELLATION_RECORDS_OVERFLOW)
            && word(StatusWord::MeshFaultObjects) == 0;
        StatusReport {
            overflow: bits.contains(StatusBits::OVERFLOW),
            resident_miss: bits.contains(StatusBits::RESIDENT_MISS),
            resident_misses: word(StatusWord::ResidentMisses),
            replay_fault: bits.contains(StatusBits::REPLAY_FAULT),
            replay_faults: ReplayFaults(word(StatusWord::ReplayFaults)),
            index_pool_overflow: bits.contains(StatusBits::INDEX_POOL_OVERFLOW),
            absent_argument_read: bits.contains(StatusBits::ABSENT_ARGUMENT_READ),
            cubic_argument_sampler: bits.contains(StatusBits::CUBIC_ARGUMENT_SAMPLER),
            mesh_grid_discarded: bits.contains(StatusBits::MESH_GRID_DISCARDED),
            mesh_grid_past_limits: bits.contains(StatusBits::MESH_GRID_PAST_LIMITS),
            mesh_records_overflow: (bits.contains(StatusBits::MESH_RECORDS_OVERFLOW)).then(|| {
                // The fault words are the first records fault's: a tessellation draw's when its
                // object threadgroups word is 0 (a scanned mesh draw's is at least 1), which
                // names no mesh draw.
                match tessellation_first {
                    false => MeshRecordsFault {
                        draw: word(StatusWord::MeshFaultDraw),
                        object_threadgroups: word(StatusWord::MeshFaultObjects),
                        mesh_threadgroups: word(StatusWord::MeshFaultGroups),
                    },
                    true => MeshRecordsFault::default(),
                }
            }),
            tessellation_records_overflow: bits
                .contains(StatusBits::TESSELLATION_RECORDS_OVERFLOW)
                .then(|| match tessellation_first {
                    true => TessellationRecordsFault {
                        draw: word(StatusWord::MeshFaultDraw),
                        records: word(StatusWord::MeshFaultGroups),
                    },
                    false => TessellationRecordsFault::default(),
                }),
        }
    }
}

/// What an `ExecutionStatus` buffer reports.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StatusReport {
    /// The indirect clamp reduced a GPU-written group count.
    pub overflow: bool,
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
    /// A mesh scan found no room in the command buffer's records pool for an emulated mesh
    /// draw's records, or the draw's vertices passed a `u32` (`bind::MeshScanModule`), or an
    /// object arguments pass none for an emulated indirect draw's object slots
    /// (`bind::mesh_object_arguments_module`, no mesh threadgroups yet): such draws drew nothing.
    /// The first one, as the pass saw it, where it was the command buffer's first records fault;
    /// a fault of all zeros where a tessellation draw's was.
    pub mesh_records_overflow: Option<MeshRecordsFault>,
    /// A tessellation scan found no room in the command buffer's records pool for an emulated
    /// tessellation draw's records, or the draw's records passed a `u32`
    /// (`bind::TessellationScanModule`): such draws drew nothing. The first one, as the scan saw
    /// it, where it was the command buffer's first records fault; a fault of all zeros where a
    /// mesh draw's was.
    pub tessellation_records_overflow: Option<TessellationRecordsFault>,
    /// A function sampled through an argument buffer's sampler whose min or mag filter is
    /// `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): no result of the command buffer is then Metal's.
    pub cubic_argument_sampler: bool,
    /// An indirect mesh draw's arguments pass (or an emulated one's object arguments pass) met an
    /// object grid that Metal's GPU discards (`StatusBits::MESH_GRID_DISCARDED`): that draw drew
    /// nothing, and the command buffer fails as Metal's does (`MTLCommandBufferErrorInternal`).
    pub mesh_grid_discarded: bool,
    /// An indirect mesh draw's arguments pass (or an emulated one's object arguments pass) met a
    /// grid that Metal draws and the device cannot launch (`StatusBits::MESH_GRID_PAST_LIMITS`):
    /// that draw drew nothing, and the command buffer fails (not Metal's failure).
    pub mesh_grid_past_limits: bool,
}

/// The first emulated mesh draw of a command buffer that drew nothing because its records did
/// not fit (`StatusReport::mesh_records_overflow`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshRecordsFault {
    /// The consumer's number for the draw (the work buffer's `DRAW_ID` word).
    pub draw: u32,
    /// The draw's object threadgroups (1 without an object function).
    pub object_threadgroups: u32,
    /// The mesh threadgroups the draw's object threadgroups launched, saturated at `u32::MAX`; 0
    /// where its object slots did not fit, before any launched.
    pub mesh_threadgroups: u32,
}

/// The first emulated tessellation draw of a command buffer that drew nothing because its records
/// did not fit (`StatusReport::tessellation_records_overflow`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TessellationRecordsFault {
    /// The consumer's number for the draw (the work buffer's `DRAW_ID` word).
    pub draw: u32,
    /// The records the draw needed (`patchCount · instanceCount`), saturated at `u32::MAX`.
    pub records: u32,
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
    /// The words of the command buffer's mesh records pool its scans reserved
    /// (`bind::MeshScanModule`). Initial 0.
    MeshRecordsCursor,
    /// The consumer's number for the command buffer's first emulated mesh or tessellation draw
    /// whose records did not fit ([`StatusBits::RECORDS_FAULTS`]). Initial 0.
    MeshFaultDraw,
    /// That draw's object threadgroups, 0 for a tessellation draw. Initial 0.
    MeshFaultObjects,
    /// The mesh threadgroups, or the tessellation records, that draw needed, saturated. Initial
    /// 0.
    MeshFaultGroups,
    /// The word every sequentially consistent fence of the command buffer increments with an
    /// acquire-release atomic at the fence's scope (`lower::sync::barrier::seq_cst_order`), so
    /// that the fences of one command buffer are totally ordered, as Metal's are. Its value means
    /// nothing; the decode does not read it. Initial 0.
    SeqCstFences,
}

impl StatusWord {
    /// Every word, in index order.
    pub const ALL: [StatusWord; 8] = [
        StatusWord::Status,
        StatusWord::ResidentMisses,
        StatusWord::ReplayFaults,
        StatusWord::MeshRecordsCursor,
        StatusWord::MeshFaultDraw,
        StatusWord::MeshFaultObjects,
        StatusWord::MeshFaultGroups,
        StatusWord::SeqCstFences,
    ];

    /// The word's index in the buffer.
    pub const fn index(self) -> u32 {
        match self {
            StatusWord::Status => 0,
            StatusWord::ResidentMisses => 1,
            StatusWord::ReplayFaults => 2,
            StatusWord::MeshRecordsCursor => 3,
            StatusWord::MeshFaultDraw => 4,
            StatusWord::MeshFaultObjects => 5,
            StatusWord::MeshFaultGroups => 6,
            StatusWord::SeqCstFences => 7,
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
    /// The indirect clamp reduced a group count.
    pub const OVERFLOW: StatusBits = StatusBits(1 << 0);
    /// An access missed the resident set.
    pub const RESIDENT_MISS: StatusBits = StatusBits(1 << 1);
    /// The device replay's preprocess refused a command; [`StatusWord::ReplayFaults`] holds why.
    pub const REPLAY_FAULT: StatusBits = StatusBits(1 << 2);
    /// An index bounds pass found no room in the command buffer's index pool for a draw's
    /// indices, and the draw drew nothing (`bind::IndexBoundsPass`).
    pub const INDEX_POOL_OVERFLOW: StatusBits = StatusBits(1 << 3);
    /// A function read an argument its pipeline's function constants removed, a read Metal
    /// gives no value, and took zero for it (`RuntimeFact::AbsentArgumentRead`).
    pub const ABSENT_ARGUMENT_READ: StatusBits = StatusBits(1 << 4);
    /// A mesh scan found no room in the command buffer's records pool for an emulated mesh
    /// draw's records, or the draw's vertices passed a `u32`, or an object arguments pass none for
    /// an emulated indirect draw's object slots, and the draw drew nothing
    /// (`bind::MeshScanModule`, `bind::mesh_object_arguments_module`); the fault words name the
    /// command buffer's first records fault of either kind.
    pub const MESH_RECORDS_OVERFLOW: StatusBits = StatusBits(1 << 5);
    /// A function sampled, gathered, or queried through a sampler an argument buffer holds whose
    /// min or mag filter is `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): its result is the device's through a nearest
    /// placeholder, not Metal's.
    pub const CUBIC_ARGUMENT_SAMPLER: StatusBits = StatusBits(1 << 6);
    /// A tessellation scan found no room in the command buffer's records pool for an emulated
    /// tessellation draw's records, or the draw's records passed a `u32`, and the draw drew
    /// nothing (`bind::TessellationScanModule`); the fault words name the command buffer's first
    /// records fault of either kind.
    pub const TESSELLATION_RECORDS_OVERFLOW: StatusBits = StatusBits(1 << 8);
    /// An indirect mesh draw's arguments pass met a grid that Metal draws and the device cannot
    /// launch: a mesh grid past the mesh stage's limits, an unfolded object grid past the task
    /// stage's, or a folded one of more pieces than `MeshArguments::OBJECT_PIECES` (only under task
    /// limits below Vulkan's required least); or an emulated indirect draw's object arguments pass
    /// met an object grid past the device's compute limits (`bind::mesh_object_arguments_module`);
    /// the draw drew nothing
    /// (`bind::MeshArguments`).
    pub const MESH_GRID_PAST_LIMITS: StatusBits = StatusBits(1 << 9);
    /// An indirect mesh draw's arguments pass met an object grid of more threadgroups than
    /// `bind::METAL_OBJECT_GRID_TOTAL`, which Metal's GPU discards (or an emulated draw's object
    /// arguments pass did); the draw drew nothing (`bind::MeshArguments`,
    /// `bind::mesh_object_arguments_module`).
    pub const MESH_GRID_DISCARDED: StatusBits = StatusBits(1 << 10);
    /// Either records fault: the command buffer's first one of either kind writes the fault
    /// words.
    pub const RECORDS_FAULTS: StatusBits = StatusBits(
        StatusBits::MESH_RECORDS_OVERFLOW.0 | StatusBits::TESSELLATION_RECORDS_OVERFLOW.0,
    );

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
            assert_eq!(w.index() as usize, i, "{w:?}");
            assert!((w.byte_offset() as usize) + 4 <= ExecutionStatus::BYTES);
        }
        assert_eq!(ExecutionStatus::WORDS * 4, ExecutionStatus::BYTES);
    }

    #[test]
    fn status_bits_are_distinct() {
        let bits = [
            StatusBits::OVERFLOW,
            StatusBits::RESIDENT_MISS,
            StatusBits::REPLAY_FAULT,
            StatusBits::INDEX_POOL_OVERFLOW,
            StatusBits::ABSENT_ARGUMENT_READ,
            StatusBits::MESH_RECORDS_OVERFLOW,
            StatusBits::CUBIC_ARGUMENT_SAMPLER,
        ];
        let mut all = 0u32;
        for b in bits {
            assert_eq!(b.0.count_ones(), 1);
            assert_eq!(all & b.0, 0);
            all |= b.0;
        }
        assert!(StatusBits(all).contains(StatusBits::OVERFLOW));
        assert!(!StatusBits::RESIDENT_MISS.contains(StatusBits::OVERFLOW));
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
        let mut bytes = [0u8; 32];
        let bits = StatusBits::OVERFLOW.0 | StatusBits::RESIDENT_MISS.0;
        bytes[0..4].copy_from_slice(&bits.to_le_bytes());
        bytes[4..8].copy_from_slice(&7u32.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.overflow && r.resident_miss);
        assert_eq!(r.resident_misses, 7);
        bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&StatusBits::OVERFLOW.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.overflow && !r.resident_miss);
        assert_eq!(r.resident_misses, 0);
        bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&StatusBits::REPLAY_FAULT.0.to_le_bytes());
        let faults = ReplayFaults::VARIANT.0 | ReplayFaults::RECORD.0;
        bytes[8..12].copy_from_slice(&faults.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.replay_fault && !r.overflow);
        assert!(r.replay_faults.contains(ReplayFaults::VARIANT));
        assert!(r.replay_faults.contains(ReplayFaults::RECORD));
        assert!(!r.replay_faults.contains(ReplayFaults::ENCODING));
        assert!(!r.index_pool_overflow);
        bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&StatusBits::INDEX_POOL_OVERFLOW.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(r.index_pool_overflow && !r.overflow && !r.replay_fault);
        assert!(!r.absent_argument_read);
        bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&StatusBits::ABSENT_ARGUMENT_READ.0.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert!(!r.cubic_argument_sampler);
        let mut cubic = [0u8; 32];
        cubic[0..4].copy_from_slice(&StatusBits::CUBIC_ARGUMENT_SAMPLER.0.to_le_bytes());
        let c = ExecutionStatus::decode(&cubic);
        assert!(c.cubic_argument_sampler && !c.absent_argument_read && !c.overflow);
        assert!(r.absent_argument_read && !r.index_pool_overflow && !r.overflow);
        assert_eq!(r.mesh_records_overflow, None);
        bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&StatusBits::MESH_RECORDS_OVERFLOW.0.to_le_bytes());
        bytes[12..16].copy_from_slice(&40u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&3u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        let r = ExecutionStatus::decode(&bytes);
        assert_eq!(
            r.mesh_records_overflow,
            Some(MeshRecordsFault {
                draw: 3,
                object_threadgroups: 2,
                mesh_threadgroups: u32::MAX,
            })
        );
        assert!(!r.index_pool_overflow && !r.overflow);
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
