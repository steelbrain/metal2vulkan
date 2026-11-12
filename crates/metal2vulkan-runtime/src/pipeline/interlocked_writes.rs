//! What a stage's SPIR-V words state about its set 0 that decides whether a variant pushes it on
//! a driver whose pushed sets lose their buffers' sizes (`objects::pushes_set_zero`): whether the
//! stage runs in a fragment shader interlock, whether it declares a storage buffer of set 0 it may
//! write, and whether it reads a buffer's length. Whether a fragment stage interlocks also decides
//! whether its pipeline is made whole on a driver whose fast-linked pipelines leave the interlock
//! unordered (`graphics.rs`'s `GraphicsContent::links`).
//!
//! The words are a module's binary form (the SPIR-V specification, "Physical Layout of a SPIR-V
//! Module and Instruction"): a header of five words, then the instructions, each opening with its
//! word count in the high 16 bits and its opcode in the low 16. The translator runs an interlocked
//! fragment function's whole body inside the critical section (`lower/fragment/mod.rs`,
//! `lower/fragment/rog.rs`), so every storage buffer such a module declares is reached inside its
//! interlock; and it decorates `NonWritable` every storage resource a module cannot write
//! (`spirv/serialize`'s `read_only_resources`), so an undecorated one may be written.

/// `OpExecutionMode`.
const EXECUTION_MODE: u32 = 16;
/// `OpVariable`.
const VARIABLE: u32 = 59;
/// `OpArrayLength`.
const ARRAY_LENGTH: u32 = 68;
/// `OpDecorate`.
const DECORATE: u32 = 71;
/// `OpUntypedArrayLengthKHR`.
const UNTYPED_ARRAY_LENGTH: u32 = 4425;
/// The interlock execution modes, `PixelInterlockOrderedEXT` (5366) to
/// `ShadingRateInterlockUnorderedEXT` (5371).
const INTERLOCKS: std::ops::RangeInclusive<u32> = 5366..=5371;
/// The `StorageBuffer` storage class.
const STORAGE_BUFFER: u32 = 12;
/// The `NonWritable` decoration.
const NON_WRITABLE: u32 = 24;
/// The `DescriptorSet` decoration.
const DESCRIPTOR_SET: u32 = 34;

/// What one stage's words state ([`facts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StageFacts {
    /// It declares an interlock execution mode.
    pub(crate) interlocks: bool,
    /// It declares a `StorageBuffer` variable of descriptor set 0 not decorated `NonWritable`.
    pub(crate) writable_set_zero_buffer: bool,
    /// It reads a buffer's length (`OpArrayLength`, `OpUntypedArrayLengthKHR`), or holds an
    /// instruction whose word count is 0 or runs past its words, which ends the walk: a module
    /// the walk cannot read is taken as reading one.
    pub(crate) reads_lengths: bool,
}

/// What the module of `words` states (the module docs).
pub(crate) fn facts(words: &[u32]) -> StageFacts {
    let mut facts = StageFacts::default();
    // Each variable of the StorageBuffer class, with whether set 0 holds it and whether it is
    // NonWritable: decorations come before the variables they name.
    let mut set_zero = Vec::new();
    let mut non_writable = Vec::new();
    let mut at = words.len().min(5);
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let Some(instruction) = words.get(at..at + count).filter(|_| count > 0) else {
            facts.reads_lengths = true;
            break;
        };
        match (instruction[0] & 0xffff, instruction) {
            (EXECUTION_MODE, [_, _, mode, ..]) if INTERLOCKS.contains(mode) => {
                facts.interlocks = true;
            }
            (DECORATE, [_, target, DESCRIPTOR_SET, 0, ..]) => set_zero.push(*target),
            (DECORATE, [_, target, NON_WRITABLE, ..]) => non_writable.push(*target),
            (VARIABLE, [_, _, id, STORAGE_BUFFER, ..])
                if set_zero.contains(id) && !non_writable.contains(id) =>
            {
                facts.writable_set_zero_buffer = true;
            }
            (ARRAY_LENGTH | UNTYPED_ARRAY_LENGTH, _) => facts.reads_lengths = true,
            _ => {}
        }
        at += count;
    }
    facts
}
