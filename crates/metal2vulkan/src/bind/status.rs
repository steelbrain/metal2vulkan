//! `ExecutionStatus` buffer layout: word indices and status bits.
//!
//! One 16-byte buffer per command buffer, shared by every pipeline in it, which the consumer
//! zeroes before the command buffer's first use; the resident-set miss counter, an absent
//! argument's read, a sample through a cubic argument sampler, and a sequentially consistent
//! fence write the words named here.

/// A word of the `ExecutionStatus` buffer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatusWord {
    /// Status bits ([`StatusBits`]). Initial 0.
    Status,
    /// Resident-miss count. Initial 0.
    ResidentMisses,
    /// The word every sequentially consistent fence of the command buffer increments with an
    /// acquire-release atomic at the fence's scope (`lower::sync::barrier::seq_cst_order`), so
    /// that the fences of one command buffer are totally ordered, as Metal's are. Its value means
    /// nothing; the decode does not read it. Initial 0. It is the buffer's last word, after one
    /// that nothing writes.
    SeqCstFences,
}

impl StatusWord {
    /// The word's index in the buffer.
    pub const fn index(self) -> u32 {
        match self {
            StatusWord::Status => 0,
            StatusWord::ResidentMisses => 1,
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
    /// A function sampled, gathered, or queried through a sampler an argument buffer holds whose
    /// min or mag filter is `Cubic`, whose bicubic filter it does not emulate there
    /// (`RuntimeFact::CubicArgumentSampler`): its result is the device's through a nearest
    /// placeholder, not Metal's.
    pub const CUBIC_ARGUMENT_SAMPLER: StatusBits = StatusBits(1 << 3);
}
