//! Command predication's step: the GPU half of Metal's private compute encoder calls
//! `encodeStartIf:offset:comparison:referenceValue:`, `encodeStartElse` and `encodeEndIf` (and, for
//! a killed block, the ends of the loop calls). Metal reads each block's condition when the GPU
//! reaches it, so the consumer records one invocation of the crate-owned module
//! [`predication_step_module`] at each record, which reads the condition word where it lies and
//! moves a skip counter, then gates the dispatches up to the next record.
//!
//! - **The counter** (one `u32` per unit of predication, in the consumer's memory, 0 at the
//!   unit's start): 0 while the unit's commands run, above 0 while they are skipped. A record's
//!   step moves it by its [`PredicationOp`]: an If inside a skipped region adds 1, and otherwise
//!   leaves 0 where its condition holds and sets 1 where not; an Else turns 0 into 1 and 1 into 0,
//!   leaving a deeper count alone; an End takes 1 off a count above 0. A block a misused record
//!   killed ([`PredicationOp::Dead`]) adds 2 at its start, so its own Else (which sees 2 or more)
//!   turns nothing back on, and takes 2 off at its End ([`PredicationOp::EndDead`]). The steps of
//!   a range an unbalanced record dropped are rewritten: the range's first as
//!   [`PredicationOp::Drop`] (the counter becomes 1), every later one as
//!   [`PredicationOp::Hold`] (the counter stays), and a [`PredicationOp::Resume`] (the counter
//!   becomes 0) goes before the record that ends the range. [`PredicationOp::UnitEnd`] leaves the
//!   counter and writes whether it is above 0 into its check word.
//! - **The halt word** (the `u32` 4 bytes past the counter, 0 at the unit's start): a loop's step
//!   past its last copy ([`PredicationOp::WhilePast`], [`PredicationOp::DoPast`]) sets it to the
//!   loop's id where the loop would run another pass. While it is not 0, every step leaves the
//!   counter, its loop's flag and its check word as they are and closes its gates, but the start
//!   of the first copy of the loop of that id ([`PredicationOp::WhileFirst`],
//!   [`PredicationOp::DoFirst`]), which sets it back to 0 and runs: so the consumer can run the
//!   recorded steps again from the halted loop's first copy on, and the loop goes on where it
//!   stopped, as many passes again as it has copies. A first copy's start takes its loop's flag
//!   as clear, so a loop laid down again runs afresh.
//! - **The condition** (an If's): `word OP reference`, unsigned, the word on the left
//!   ([`PredicationCompare`], Metal's codes 0 to 5; any other code compares `==`, as Metal's do).
//!   The word is the `u32` at the record's word address, which the consumer points at Metal's
//!   offset rounded down to a multiple of 4. Every step reads its word, and only an If's
//!   condition is used: a step of another op names its own record as its word.
//! - **The gates** (the dispatches a step's record runs until the next one): each gate is
//!   [`PredicationStep::GATE_BYTES`] of the consumer's memory: a `VkDispatchIndirectCommand` the
//!   consumer wrote (the dispatch's group counts), and at [`PredicationStep::GATE_OUT_OFFSET`] the
//!   one the dispatch runs by `vkCmdDispatchIndirect`, which the step writes: the group counts
//!   where the counter it leaves is 0, else zero. The record's table names them
//!   ([`PredicationStep::table`]), and then the record's own sink gate (at
//!   [`PredicationStep::SINK_GATE_OFFSET`]); a record's own empty table (at
//!   [`PredicationStep::EMPTY_TABLE_OFFSET`], naming only the sink) is its table until the
//!   consumer points it at one.
//! - **Every read lands in memory the record names**, in a branch not taken and in a loop's pass
//!   past its count too: llvmpipe (lavapipe) makes a uniform load under control flow whether or
//!   not an invocation runs it, and a step faulted there reading address 0 (a scalar `u32` load of
//!   address 0 in its JIT code, `SEGV_MAPERR`). So the word is read whatever the op, and the gate
//!   loop's pass past the table's count reads the sink, whose command the step writes, if at all,
//!   where no dispatch reads it.
//!
//! **The record** ([`PredicationStep::record`], [`PredicationStep::RECORD_BYTES`], 16-aligned): the
//! counter's device address, the word's, the table's, the op, the comparison and the reference, an
//! empty table, and the sink gate. The consumer may rewrite the op ([`PredicationStep::OP_OFFSET`],
//! [`PredicationOp::word`]) and the table's address ([`PredicationStep::TABLE_OFFSET`]) until the
//! step runs.
//!
//! **The module** (one per device, in the device's [`PredicationStepForm`]): one `GLCompute` entry
//! `main` of local size `(1, 1, 1)`, dispatched `(1, 1, 1)`.
//!
//! - **Addressed** (a device with device addresses and `shaderInt64`): no descriptor, and
//!   [`PredicationStepModule::PUSH_BYTES`] of `COMPUTE` push constants: the record's device
//!   address ([`PredicationStepModule::push_constants`]). It reaches every place by device
//!   address.
//! - **Bound** (every other device): no push constants, and set 0 of
//!   [`PredicationStepModule::BINDINGS`] storage buffers, which the consumer binds per step: the
//!   record's region ([`PredicationStepModule::RECORD_BINDING`], the record at its byte 0), the
//!   counter's (with its unit's halt word), Metal's word's (bound at an offset the device aligns),
//!   the loop's flag's and the check word's. The record is [`PredicationStep::record`] of `at` =
//!   0, whose address fields hold, in their low halves, each place's byte offset in its binding:
//!   a step that names its own record for a place (the word of a step that tests none, the flag
//!   of a step not of a loop, the check word of a step not a unit's end) has that binding bound
//!   to its record's region. Its gates lie in the record's region, the first at [`PredicationStep::BOUND_GATES_OFFSET`], their
//!   count (a `u32`) at [`PredicationStep::BOUND_COUNT_OFFSET`] (0 until the consumer writes it),
//!   and the region holds one gate more ([`PredicationStep::bound_bytes`]), which the gate loop's
//!   pass past the count reads. No table.
//!
//! **Recording** (consumer): a barrier from everything before the step (Metal's word is read
//! after the work before the record, written with or without a barrier); the step; a barrier
//! from its writes to the indirect reads and compute reads after it. The counter, the table and
//! the gates are read and written by the steps and the gated dispatches alone.

use std::sync::Arc;

use super::{
    at, count_loop, count_words, k32, k64, load32, load64, no_binding, store32, wide, Types,
};
use crate::bind::visibility_combine::words_buffer;
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, MemAccess, Type, TypedModule, UnOp};

/// What a predication record's step does to its unit's skip counter (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicationOp {
    /// `encodeStartIf:...`: inside a skipped region, one deeper; else 0 where the condition
    /// holds and 1 where not.
    If,
    /// `encodeStartElse`: 0 becomes 1 and 1 becomes 0; a deeper count stays.
    Else,
    /// `encodeEndIf` (or another block's end): one less, down to 0.
    End,
    /// The start of a block a misused record killed: two deeper, so nothing inside it runs and
    /// its own Else turns nothing back on.
    Dead,
    /// The end of a killed block: two less.
    EndDead,
    /// A copy's start of `encodeStartWhile:...`'s loop (laid down as copies): inside a
    /// skipped region one deeper; else 1 where the loop has exited (its flag set), 0 where the
    /// condition holds, and 1 where not, which sets the flag.
    While,
    /// The start of the first copy of `encodeStartWhile:...`'s loop: a [`While`](Self::While)
    /// whose flag is taken as clear; where the halt word is the loop's id, it clears it first.
    WhileFirst,
    /// A copy's start of `encodeStartDoWhile`'s loop: inside a skipped region one deeper; else 1
    /// where the loop has exited, and 0 where not.
    Do,
    /// The start of the first copy of `encodeStartDoWhile`'s loop: a [`Do`](Self::Do) whose flag
    /// is taken as clear; where the halt word is the loop's id, it clears it first.
    DoFirst,
    /// A copy's end of a DoWhile loop (`encodeEndDoWhile:...`): where the copy ran (the counter
    /// 0) and the condition fails, the flag is set; then one less, as [`End`](Self::End).
    DoEnd,
    /// After a While loop's last copy: where the loop would run another (the counter 0, the
    /// flag clear, the condition holding), the halt word becomes the loop's id; the counter stays.
    WhilePast,
    /// After a DoWhile loop's last copy: where the loop would run another (the counter 0, the
    /// flag clear), the halt word becomes the loop's id; the counter stays.
    DoPast,
    /// The first step of a range an unbalanced record dropped: the counter becomes 1, so nothing
    /// up to the range's end runs.
    Drop,
    /// A later step of a dropped range: the counter stays.
    Hold,
    /// The end of a dropped range, before the record after it: the counter becomes 0.
    Resume,
    /// The end of a compute encoder whose unit has a block open: the counter stays, and the check
    /// word ([`PredicationStep::checking`]) becomes 1 where the counter is above 0 (the unit's
    /// commands skip there), else 0.
    UnitEnd,
}

impl PredicationOp {
    /// The op's `u32` in a record, little-endian (written at [`PredicationStep::OP_OFFSET`]).
    pub fn word(self) -> [u8; 4] {
        self.code().to_le_bytes()
    }

    fn code(self) -> u32 {
        match self {
            PredicationOp::If => OP_IF,
            PredicationOp::Else => OP_ELSE,
            PredicationOp::End => OP_END,
            PredicationOp::Dead => OP_DEAD,
            PredicationOp::EndDead => OP_END_DEAD,
            PredicationOp::While => OP_WHILE,
            PredicationOp::WhileFirst => OP_WHILE_FIRST,
            PredicationOp::Do => OP_DO,
            PredicationOp::DoFirst => OP_DO_FIRST,
            PredicationOp::DoEnd => OP_DO_END,
            PredicationOp::WhilePast => OP_WHILE_PAST,
            PredicationOp::DoPast => OP_DO_PAST,
            PredicationOp::Drop => OP_DROP,
            PredicationOp::Hold => OP_HOLD,
            PredicationOp::Resume => OP_RESUME,
            PredicationOp::UnitEnd => OP_UNIT_END,
        }
    }

    /// Whether the step reads Metal's word.
    fn tests(self) -> bool {
        matches!(
            self,
            PredicationOp::If
                | PredicationOp::While
                | PredicationOp::WhileFirst
                | PredicationOp::DoEnd
                | PredicationOp::WhilePast
        )
    }

    /// Whether the step is of a loop: it reads and writes its loop's flag, and names its loop's id.
    fn loops(self) -> bool {
        matches!(
            self,
            PredicationOp::While
                | PredicationOp::WhileFirst
                | PredicationOp::Do
                | PredicationOp::DoFirst
                | PredicationOp::DoEnd
                | PredicationOp::WhilePast
                | PredicationOp::DoPast
        )
    }
}

const OP_IF: u32 = 0;
const OP_ELSE: u32 = 1;
const OP_END: u32 = 2;
const OP_DEAD: u32 = 3;
const OP_END_DEAD: u32 = 4;
const OP_WHILE: u32 = 5;
const OP_DO: u32 = 6;
const OP_DO_END: u32 = 7;
const OP_WHILE_PAST: u32 = 8;
const OP_DO_PAST: u32 = 9;
const OP_DROP: u32 = 10;
const OP_HOLD: u32 = 11;
const OP_RESUME: u32 = 12;
const OP_UNIT_END: u32 = 13;
const OP_WHILE_FIRST: u32 = 14;
const OP_DO_FIRST: u32 = 15;

/// An If's comparison of Metal's word with its reference, unsigned, the word on the left: Metal's
/// `comparison` argument, which is not an `MTLCompareFunction` (measured on an M4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicationCompare {
    /// 0: `word == reference`.
    Equal,
    /// 1: `word != reference`.
    NotEqual,
    /// 2: `word > reference`.
    Greater,
    /// 3: `word >= reference`.
    GreaterEqual,
    /// 4: `word < reference`.
    Less,
    /// 5: `word <= reference`.
    LessEqual,
}

impl PredicationCompare {
    /// Metal's `comparison` argument: 0 to 5 as the variants state, and any other value
    /// (6, 255, `2^32 + 2`, `2^64 - 1`, all measured) [`Equal`](Self::Equal), as Metal compares.
    pub fn from_metal(comparison: u64) -> PredicationCompare {
        match comparison {
            1 => PredicationCompare::NotEqual,
            2 => PredicationCompare::Greater,
            3 => PredicationCompare::GreaterEqual,
            4 => PredicationCompare::Less,
            5 => PredicationCompare::LessEqual,
            _ => PredicationCompare::Equal,
        }
    }

    fn code(self) -> u32 {
        match self {
            PredicationCompare::Equal => 0,
            PredicationCompare::NotEqual => 1,
            PredicationCompare::Greater => 2,
            PredicationCompare::GreaterEqual => 3,
            PredicationCompare::Less => 4,
            PredicationCompare::LessEqual => 5,
        }
    }

    /// Whether `word` compares so with `reference`.
    pub fn holds(self, word: u32, reference: u32) -> bool {
        match self {
            PredicationCompare::Equal => word == reference,
            PredicationCompare::NotEqual => word != reference,
            PredicationCompare::Greater => word > reference,
            PredicationCompare::GreaterEqual => word >= reference,
            PredicationCompare::Less => word < reference,
            PredicationCompare::LessEqual => word <= reference,
        }
    }
}

/// One predication record (the module docs): the step's inputs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PredicationStep {
    /// The device address of the unit's skip counter (a `u32`, 8-aligned), and 4 bytes past it of
    /// its halt word (the module docs).
    pub counter: u64,
    /// What the step does.
    pub op: PredicationOp,
    /// The device address of Metal's condition word (4-aligned), which an
    /// [`If`](PredicationOp::If), a [`While`](PredicationOp::While), a
    /// [`DoEnd`](PredicationOp::DoEnd) and a [`WhilePast`](PredicationOp::WhilePast) test; 0 for
    /// the others, whose record names itself instead ([`PredicationStep::record`]).
    pub word: u64,
    /// The condition's comparison.
    pub compare: PredicationCompare,
    /// The condition's reference value.
    pub reference: u32,
    /// A loop's step: the device address of the loop's flag (a `u32`, 0 until the loop exits;
    /// one per laying down of the loop). 0 for the others, whose record names a word of its own.
    pub flag: u64,
    /// A loop's step: the loop's id (not 0; one per laying down of the loop within its unit),
    /// which its past step writes into the halt word and its first copy's start answers to. 0 for
    /// the others.
    pub id: u32,
    /// A [`UnitEnd`](PredicationOp::UnitEnd)'s check word (a `u32`), at this device address. 0
    /// for the others, whose record names a word of its own.
    pub check: u64,
}

// The record's fields.
const R_COUNTER: u64 = 0;
const R_WORD: u64 = 8;
const R_COMPARE: u64 = 28;
const R_REFERENCE: u64 = 32;
/// A word of the record's own that a step names as its flag and check word where it has none,
/// which it writes back unchanged.
const R_SCRATCH: u64 = 36;
const R_FLAG: u64 = 40;
const R_CHECK: u64 = 48;
const R_ID: u64 = 72;

impl PredicationStep {
    /// Bytes of a record.
    pub const RECORD_BYTES: u64 = 112;
    /// The record's table address (`u64`): the record's own empty table until the consumer
    /// points it at one ([`PredicationStep::table`]).
    pub const TABLE_OFFSET: u64 = 16;
    /// The record's op (`u32`, [`PredicationOp::word`]).
    pub const OP_OFFSET: u64 = 24;
    /// The record's own empty table: a count of 0, and the sink's entry.
    pub const EMPTY_TABLE_OFFSET: u64 = 56;
    /// The record's sink gate ([`PredicationStep::GATE_BYTES`]), which every table names after
    /// its gates (the module docs).
    pub const SINK_GATE_OFFSET: u64 = 80;
    /// Bytes of a gate: the dispatch's `VkDispatchIndirectCommand`, then the one it runs.
    pub const GATE_BYTES: u64 = 32;
    /// The gate's command the dispatch runs (12 bytes), which the step writes.
    pub const GATE_OUT_OFFSET: u64 = 16;
    /// The bound form: the byte of the record's binding where its first gate starts, past the
    /// record; each next one is [`PredicationStep::GATE_BYTES`] further.
    pub const BOUND_GATES_OFFSET: u64 = PredicationStep::RECORD_BYTES;
    /// The bound form: the record's count of gates (a `u32`), at its own empty table's count.
    pub const BOUND_COUNT_OFFSET: u64 = PredicationStep::EMPTY_TABLE_OFFSET;

    /// The bound form: the bytes of a record's binding that holds `gates` gates, and one more
    /// past them, which a pass of the gate loop past the count reads (the module docs).
    pub const fn bound_bytes(gates: u64) -> u64 {
        PredicationStep::BOUND_GATES_OFFSET + (gates + 1) * PredicationStep::GATE_BYTES
    }

    /// The step of `op` on the counter at device address `counter`: for an
    /// [`If`](PredicationOp::If) use [`PredicationStep::condition`].
    pub fn new(counter: u64, op: PredicationOp) -> PredicationStep {
        PredicationStep {
            counter,
            op,
            word: 0,
            compare: PredicationCompare::Equal,
            reference: 0,
            flag: 0,
            id: 0,
            check: 0,
        }
    }

    /// The If on the counter at device address `counter` of Metal's word at device address
    /// `word`, compared so with `reference`.
    pub fn condition(
        counter: u64,
        word: u64,
        compare: PredicationCompare,
        reference: u32,
    ) -> PredicationStep {
        PredicationStep {
            counter,
            op: PredicationOp::If,
            word,
            compare,
            reference,
            flag: 0,
            id: 0,
            check: 0,
        }
    }

    /// This step testing Metal's word at device address `word`, compared so with `reference`
    /// (a [`While`](PredicationOp::While), [`DoEnd`](PredicationOp::DoEnd) or
    /// [`WhilePast`](PredicationOp::WhilePast)).
    pub fn testing(self, word: u64, compare: PredicationCompare, reference: u32) -> Self {
        PredicationStep {
            word,
            compare,
            reference,
            ..self
        }
    }

    /// This loop step on the loop's flag at device address `flag`, of the loop of id `id`.
    pub fn in_loop(self, flag: u64, id: u32) -> Self {
        PredicationStep { flag, id, ..self }
    }

    /// This [`UnitEnd`](PredicationOp::UnitEnd) step writing its check word (a `u32`) at device
    /// address `word`. Its flag is the record's own word.
    pub fn checking(self, word: u64) -> Self {
        PredicationStep {
            check: word,
            ..self
        }
    }

    /// The record's bytes where it lies at device address `at`, its table its own empty one; a
    /// step that tests no word names the record itself as its word, and one with no flag or check
    /// word a word of its own for each, so no step reads or writes address 0.
    pub fn record(&self, at: u64) -> [u8; 112] {
        let mut bytes = [0u8; 112];
        let mut put = |offset: u64, word: &[u8]| {
            let offset = offset as usize;
            bytes[offset..offset + word.len()].copy_from_slice(word);
        };
        put(R_COUNTER, &self.counter.to_le_bytes());
        let word = if self.op.tests() { self.word } else { at };
        put(R_WORD, &word.to_le_bytes());
        let (flag, check) = match self.op {
            PredicationOp::UnitEnd => (at + R_SCRATCH, self.check),
            op if op.loops() => (self.flag, at + R_SCRATCH),
            _ => (at + R_SCRATCH, at + R_SCRATCH),
        };
        put(R_FLAG, &flag.to_le_bytes());
        put(R_CHECK, &check.to_le_bytes());
        put(R_ID, &self.id.to_le_bytes());
        put(
            PredicationStep::TABLE_OFFSET,
            &(at + PredicationStep::EMPTY_TABLE_OFFSET).to_le_bytes(),
        );
        put(PredicationStep::OP_OFFSET, &self.op.word());
        put(R_COMPARE, &self.compare.code().to_le_bytes());
        put(R_REFERENCE, &self.reference.to_le_bytes());
        let empty = PredicationStep::table(at, &[]);
        put(PredicationStep::EMPTY_TABLE_OFFSET, &empty);
        bytes
    }

    /// The table, for the record at device address `at`, of the gates at device addresses
    /// `gates` (each 16-aligned): their count (a `u32`), 4 bytes of padding, then each address
    /// (a `u64`), and last the record's sink gate's (the module docs).
    pub fn table(at: u64, gates: &[u64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + 8 * gates.len());
        bytes.extend_from_slice(&(gates.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        let sink = at + PredicationStep::SINK_GATE_OFFSET;
        for gate in gates.iter().chain([&sink]) {
            bytes.extend_from_slice(&gate.to_le_bytes());
        }
        bytes
    }

    /// A gate's bytes for a dispatch of `group_count`: the command, and a zero one for the step
    /// to write.
    pub fn gate(group_count: [u32; 3]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (i, n) in group_count.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&n.to_le_bytes());
        }
        bytes
    }
}

/// How a device's step module reaches its memory (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicationStepForm {
    /// By device address: the record's in the push constants, every other place's in the
    /// record. A device with device addresses and `shaderInt64`.
    Addressed,
    /// Through set 0's storage buffers, the record naming each place by its byte offset in its
    /// binding. Every other device.
    Bound,
}

impl PredicationStepForm {
    /// The form of `device`'s step module.
    pub fn of(device: &Device) -> PredicationStepForm {
        match device.profile().fields().buffer_access {
            BufferAccess::Addressed => PredicationStepForm::Addressed,
            _ => PredicationStepForm::Bound,
        }
    }
}

/// The crate-owned predication step module (the module docs), validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicationStepModule {
    /// SPIR-V words, entry point `main`.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64` in the
    /// addressed form).
    pub requirements: Requirements,
    /// How it reaches its memory, which sets its pipeline layout: in the
    /// [`Addressed`](PredicationStepForm::Addressed) form no descriptor set and one `COMPUTE`
    /// push-constant range of [`PredicationStepModule::PUSH_BYTES`] at offset 0; in the
    /// [`Bound`](PredicationStepForm::Bound) form no push constants and set 0 of
    /// [`PredicationStepModule::BINDINGS`] storage buffers, bindings `0..5` in the order of the
    /// `*_BINDING` constants.
    pub form: PredicationStepForm,
}

impl PredicationStepModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0 of the addressed form: the record's
    /// device address.
    pub const PUSH_BYTES: u32 = 8;
    /// The bound form's storage buffers in set 0.
    pub const BINDINGS: u32 = 5;
    /// The bound form's binding of the record, at its byte 0, and its gates after it.
    pub const RECORD_BINDING: u32 = 0;
    /// The bound form's binding of the unit's skip counter.
    pub const COUNTER_BINDING: u32 = 1;
    /// The bound form's binding of Metal's condition word (the record's, where the step tests
    /// none).
    pub const WORD_BINDING: u32 = 2;
    /// The bound form's binding of the loop's flag (the record's, where the step is not of a
    /// loop).
    pub const FLAG_BINDING: u32 = 3;
    /// The bound form's binding of a [`UnitEnd`](PredicationOp::UnitEnd)'s check word (the
    /// record's, for any other step).
    pub const CHECK_BINDING: u32 = 4;

    /// The push bytes for the record at device address `record` (the addressed form).
    pub fn push_constants(record: u64) -> [u8; 8] {
        record.to_le_bytes()
    }
}

/// The predication step module of `device` (the module docs), in the device's form
/// ([`PredicationStepForm::of`]): built with the typed builder, checked by the owned verifier
/// against the device's profile (requirements included), and serialized.
pub fn predication_step_module(device: &Device) -> Result<PredicationStepModule, Error> {
    let profile = device.profile();
    let form = PredicationStepForm::of(device);
    let m = match form {
        PredicationStepForm::Addressed => typed_module()?,
        PredicationStepForm::Bound => bound_typed_module()?,
    };
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the predication step module needs {c:?}"),
        ))
    })?;
    let words = match form {
        PredicationStepForm::Addressed => crate::spirv::serialize::relocate(&r, &no_binding)?,
        PredicationStepForm::Bound => crate::spirv::serialize::relocate(&r, &bound_binding)?,
    };
    Ok(PredicationStepModule {
        spirv: words.into(),
        requirements: r.requirements,
        form,
    })
}

/// The bound form's binding symbols, in binding order.
pub(super) const BOUND_SLOTS: [ResourceSlot; 5] = [
    bound_slot(PredicationStepModule::RECORD_BINDING),
    bound_slot(PredicationStepModule::COUNTER_BINDING),
    bound_slot(PredicationStepModule::WORD_BINDING),
    bound_slot(PredicationStepModule::FLAG_BINDING),
    bound_slot(PredicationStepModule::CHECK_BINDING),
];

/// The bound form's symbol of binding `index`.
const fn bound_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// The bound form's binding numbers: set 0, its symbols in order.
fn bound_binding(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s) => (0..).zip(BOUND_SLOTS).find(|(_, b)| b == s).map(|(i, _)| i),
        _ => None,
    }
}

/// How the step reaches its memory: by device address ([`Addressed`]) or through the bound
/// buffers ([`Bound`]).
trait Memory: Copy {
    /// A place it reads or writes.
    type Place: Copy;

    /// The place `offset` bytes past `place`.
    fn at(
        self,
        fb: &mut FunctionBuilder<'_>,
        place: Self::Place,
        offset: u64,
    ) -> BResult<Self::Place>;

    /// The `u32` at `place` (4-aligned).
    fn load(self, fb: &mut FunctionBuilder<'_>, place: Self::Place) -> BResult<Value>;

    /// Store `value` (a `u32`) at `place` (4-aligned).
    fn store(self, fb: &mut FunctionBuilder<'_>, place: Self::Place, value: Value) -> BResult<()>;

    /// The place the record's field at byte `offset` names, in the bound form in binding
    /// `binding`.
    fn named(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: Self::Place,
        offset: u64,
        binding: u32,
    ) -> BResult<Self::Place>;

    /// `each` of the record's gates.
    fn gates<'a>(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: Self::Place,
        each: impl FnOnce(&mut FunctionBuilder<'_>, Self::Place) -> BResult<()> + 'a,
    ) -> BResult<()>;
}

/// The addressed form: every place a device address (a `u64`).
#[derive(Clone, Copy)]
struct Addressed(Types);

impl Memory for Addressed {
    type Place = Value;

    fn at(self, fb: &mut FunctionBuilder<'_>, place: Value, offset: u64) -> BResult<Value> {
        at(fb, place, offset)
    }

    fn load(self, fb: &mut FunctionBuilder<'_>, place: Value) -> BResult<Value> {
        load32(fb, self.0, place)
    }

    fn store(self, fb: &mut FunctionBuilder<'_>, place: Value, value: Value) -> BResult<()> {
        store32(fb, self.0, place, value)
    }

    fn named(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: Value,
        offset: u64,
        _: u32,
    ) -> BResult<Value> {
        let field = at(fb, record, offset)?;
        load64(fb, self.0, field)
    }

    /// The table the record names: its count, then each gate's device address.
    fn gates<'a>(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: Value,
        each: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
    ) -> BResult<()> {
        let t = self.0;
        let table = self.named(fb, record, PredicationStep::TABLE_OFFSET, 0)?;
        let n = load32(fb, t, table)?;
        count_loop(fb, t, n, move |fb, i| {
            let i64_ = wide(fb, t, i)?;
            let eight = k64(fb, 8)?;
            let rel = fb.binary(BinOp::IMul, i64_, eight)?;
            let entries = at(fb, table, 8)?;
            let entry = fb.binary(BinOp::IAdd, entries, rel)?;
            let gate = load64(fb, t, entry)?;
            each(fb, gate)
        })
    }
}

/// The bound form: every place a byte offset (a `u32`) of one of the bound buffers.
#[derive(Clone, Copy)]
struct Bound {
    u32_: TypeId,
    buffers: [Value; 5],
}

impl Bound {
    /// The pointer to the word at `at` of bound `buffer`.
    fn word(fb: &mut FunctionBuilder<'_>, (buffer, at): (Value, Value)) -> BResult<Value> {
        let two = k32(fb, 2)?;
        let index = fb.binary(BinOp::ShiftRightLogical, at, two)?;
        let zero = k32(fb, 0)?;
        fb.access_chain(buffer, &[zero, index])
    }
}

impl Memory for Bound {
    type Place = (Value, Value);

    fn at(
        self,
        fb: &mut FunctionBuilder<'_>,
        (buffer, at): (Value, Value),
        offset: u64,
    ) -> BResult<(Value, Value)> {
        if offset == 0 {
            return Ok((buffer, at));
        }
        let d = k32(fb, offset as u32)?;
        Ok((buffer, fb.binary(BinOp::IAdd, at, d)?))
    }

    fn load(self, fb: &mut FunctionBuilder<'_>, place: (Value, Value)) -> BResult<Value> {
        let p = Bound::word(fb, place)?;
        fb.load(p, MemAccess::default())
    }

    fn store(
        self,
        fb: &mut FunctionBuilder<'_>,
        place: (Value, Value),
        value: Value,
    ) -> BResult<()> {
        let p = Bound::word(fb, place)?;
        fb.store(p, value, MemAccess::default())
    }

    /// The field's low half: the place's byte offset in its binding.
    fn named(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: (Value, Value),
        offset: u64,
        binding: u32,
    ) -> BResult<(Value, Value)> {
        let field = self.at(fb, record, offset)?;
        Ok((self.buffers[binding as usize], self.load(fb, field)?))
    }

    /// The gates after the record ([`PredicationStep::BOUND_GATES_OFFSET`]), their count in its
    /// own table's.
    fn gates<'a>(
        self,
        fb: &mut FunctionBuilder<'_>,
        record: (Value, Value),
        each: impl FnOnce(&mut FunctionBuilder<'_>, (Value, Value)) -> BResult<()> + 'a,
    ) -> BResult<()> {
        let count = self.at(fb, record, PredicationStep::BOUND_COUNT_OFFSET)?;
        let n = self.load(fb, count)?;
        count_words(fb, self.u32_, n, move |fb, i| {
            let size = k32(fb, PredicationStep::GATE_BYTES as u32)?;
            let rel = fb.binary(BinOp::IMul, i, size)?;
            let first = k32(fb, PredicationStep::BOUND_GATES_OFFSET as u32)?;
            let at = fb.binary(BinOp::IAdd, first, rel)?;
            each(fb, (record.0, at))
        })
    }
}

/// The predication step module as a typed module, before serialization (what
/// [`predication_step_module`] serializes for the addressed form, and what the CPU interpreter
/// runs). Its push field is `ReplayRegionAddress`, the record's address.
pub(super) fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let m_record = b.push_member(PushField::ReplayRegionAddress, u64_, 0)?;
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;
    let push_value = fb.module().global_value(push)?;
    let i = k32(fb, m_record.index)?;
    let p = fb.access_chain(push_value, &[i])?;
    let record = fb.load(p, MemAccess::default())?;
    step_body(fb, Addressed(t), record)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The bound form of the predication step module as a typed module, before serialization (what
/// [`predication_step_module`] serializes for the bound form, and what the CPU interpreter runs).
/// Its bindings are the symbols [`BOUND_SLOTS`], in binding order.
pub(super) fn bound_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let mut globals: Vec<GlobalId> = Vec::with_capacity(BOUND_SLOTS.len());
    for slot in BOUND_SLOTS {
        globals.push(words_buffer(&mut b, slot, u32_)?);
    }
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;
    let zero = k32(fb, 0)?;
    let mut buffers = [zero; 5];
    for (buffer, global) in buffers.iter_mut().zip(globals) {
        *buffer = fb.module().global_value(global)?;
    }
    let record = (
        buffers[PredicationStepModule::RECORD_BINDING as usize],
        zero,
    );
    step_body(fb, Bound { u32_, buffers }, record)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The step of the record at `record`, in either form: its counter moved, its loop's flag, its
/// check word and its unit's halt word written, and its gates.
fn step_body<M: Memory>(fb: &mut FunctionBuilder<'_>, memory: M, record: M::Place) -> BResult<()> {
    let field = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = memory.at(fb, record, offset)?;
        memory.load(fb, a)
    };
    let counter_at = memory.named(
        fb,
        record,
        R_COUNTER,
        PredicationStepModule::COUNTER_BINDING,
    )?;
    let c = memory.load(fb, counter_at)?;
    let op = field(fb, PredicationStep::OP_OFFSET)?;
    let compare = field(fb, R_COMPARE)?;
    let reference = field(fb, R_REFERENCE)?;

    // The word: an If's condition word, else the record itself (`PredicationStep::record`).
    let word_at = memory.named(fb, record, R_WORD, PredicationStepModule::WORD_BINDING)?;
    let word = memory.load(fb, word_at)?;

    // The condition: Metal's six comparisons, `==` for any other code.
    let mut holds = fb.binary(BinOp::IEqual, word, reference)?;
    for (code, cmp) in [
        (1, BinOp::INotEqual),
        (2, BinOp::UGreaterThan),
        (3, BinOp::UGreaterThanEqual),
        (4, BinOp::ULessThan),
        (5, BinOp::ULessThanEqual),
    ] {
        let k = k32(fb, code)?;
        let is = fb.binary(BinOp::IEqual, compare, k)?;
        let v = fb.binary(cmp, word, reference)?;
        holds = fb.select(is, v, holds)?;
    }

    // The counter each op leaves.
    let zero = k32(fb, 0)?;
    let one = k32(fb, 1)?;
    let two = k32(fb, 2)?;
    let skipping = fb.binary(BinOp::INotEqual, c, zero)?;
    let up = fb.binary(BinOp::IAdd, c, one)?;
    let fresh = fb.select(holds, zero, one)?;
    let after_if = fb.select(skipping, up, fresh)?;
    let at_one = fb.binary(BinOp::IEqual, c, one)?;
    let flipped = fb.select(at_one, zero, c)?;
    let after_else = fb.select(skipping, flipped, one)?;
    let down = fb.binary(BinOp::ISub, c, one)?;
    let after_end = fb.select(skipping, down, zero)?;
    let after_dead = fb.binary(BinOp::IAdd, c, two)?;
    let deep = fb.binary(BinOp::UGreaterThanEqual, c, two)?;
    let down_two = fb.binary(BinOp::ISub, c, two)?;
    let after_end_dead = fb.select(deep, down_two, zero)?;

    // A loop's flag (set once the loop exits) and a UnitEnd's check word; a step with neither
    // names a word of its record for each, and writes back what it read.
    let flag_at = memory.named(fb, record, R_FLAG, PredicationStepModule::FLAG_BINDING)?;
    let check_at = memory.named(fb, record, R_CHECK, PredicationStepModule::CHECK_BINDING)?;
    let flag = memory.load(fb, flag_at)?;
    let check = memory.load(fb, check_at)?;
    // The halt word, 4 bytes past the counter, and the loop's id.
    let halt_at = memory.at(fb, counter_at, 4)?;
    let halt = memory.load(fb, halt_at)?;
    let id = field(fb, R_ID)?;
    let exited = fb.binary(BinOp::INotEqual, flag, zero)?;
    let fails = fb.unary(UnOp::LogicalNot, holds)?;
    let running = fb.unary(UnOp::LogicalNot, skipping)?;
    // While: as an If, but 1 once the loop exited, and a failing test sets the flag.
    let tested = fb.select(exited, one, fresh)?;
    let after_while = fb.select(skipping, up, tested)?;
    let exits = fb.binary(BinOp::LogicalAnd, running, fails)?;
    let flag_while = fb.select(exits, one, flag)?;
    // WhileFirst: a While on a clear flag.
    let after_while_first = fb.select(skipping, up, fresh)?;
    let flag_while_first = fb.select(exits, one, zero)?;
    // Do: 1 once the loop exited, else 0; DoFirst: 0, its flag cleared.
    let entered = fb.select(exited, one, zero)?;
    let after_do = fb.select(skipping, up, entered)?;
    let after_do_first = fb.select(skipping, up, zero)?;
    // DoEnd: a copy that ran sets the flag where the test fails; then as an End.
    let flag_do_end = fb.select(exits, one, flag)?;
    // WhilePast and DoPast: a loop still running halts the unit at its id; the counter stays.
    let staying = fb.unary(UnOp::LogicalNot, exited)?;
    let live = fb.binary(BinOp::LogicalAnd, running, staying)?;
    let past_while = fb.binary(BinOp::LogicalAnd, live, holds)?;
    let halt_while = fb.select(past_while, id, halt)?;
    let halt_do = fb.select(live, id, halt)?;
    // UnitEnd: the check word says whether the unit's commands skip where it ends.
    let unit_check = fb.select(skipping, one, zero)?;

    let mut next = c;
    let mut flag_next = flag;
    let mut check_next = check;
    let mut halt_next = halt;
    for (code, value, flag_value, check_value, halt_value) in [
        (OP_IF, after_if, flag, check, halt),
        (OP_ELSE, after_else, flag, check, halt),
        (OP_END, after_end, flag, check, halt),
        (OP_DEAD, after_dead, flag, check, halt),
        (OP_END_DEAD, after_end_dead, flag, check, halt),
        (OP_WHILE, after_while, flag_while, check, halt),
        (
            OP_WHILE_FIRST,
            after_while_first,
            flag_while_first,
            check,
            zero,
        ),
        (OP_DO, after_do, flag, check, halt),
        (OP_DO_FIRST, after_do_first, zero, check, zero),
        (OP_DO_END, after_end, flag_do_end, check, halt),
        (OP_WHILE_PAST, c, flag, check, halt_while),
        (OP_DO_PAST, c, flag, check, halt_do),
        (OP_DROP, one, flag, check, halt),
        (OP_HOLD, c, flag, check, halt),
        (OP_RESUME, zero, flag, check, halt),
        (OP_UNIT_END, c, flag, unit_check, halt),
    ] {
        let k = k32(fb, code)?;
        let is = fb.binary(BinOp::IEqual, op, k)?;
        next = fb.select(is, value, next)?;
        flag_next = fb.select(is, flag_value, flag_next)?;
        check_next = fb.select(is, check_value, check_next)?;
        halt_next = fb.select(is, halt_value, halt_next)?;
    }
    // A halted unit: every step leaves what it read, but its loop's first copy's start.
    let halted = fb.binary(BinOp::INotEqual, halt, zero)?;
    let while_first = k32(fb, OP_WHILE_FIRST)?;
    let do_first = k32(fb, OP_DO_FIRST)?;
    let is_while_first = fb.binary(BinOp::IEqual, op, while_first)?;
    let is_do_first = fb.binary(BinOp::IEqual, op, do_first)?;
    let first = fb.binary(BinOp::LogicalOr, is_while_first, is_do_first)?;
    let ours = fb.binary(BinOp::IEqual, halt, id)?;
    let resumes = fb.binary(BinOp::LogicalAnd, first, ours)?;
    let resumed = fb.unary(UnOp::LogicalNot, resumes)?;
    let holds_all = fb.binary(BinOp::LogicalAnd, halted, resumed)?;
    let next = fb.select(holds_all, c, next)?;
    let flag_next = fb.select(holds_all, flag, flag_next)?;
    let check_next = fb.select(holds_all, check, check_next)?;
    let halt_next = fb.select(holds_all, halt, halt_next)?;
    memory.store(fb, counter_at, next)?;
    memory.store(fb, flag_at, flag_next)?;
    memory.store(fb, check_at, check_next)?;
    memory.store(fb, halt_at, halt_next)?;

    // The gates: each runs its command where the counter is 0 and the unit is not halted, else
    // nothing.
    let counted = fb.binary(BinOp::IEqual, next, zero)?;
    let going = fb.binary(BinOp::IEqual, halt_next, zero)?;
    let runs = fb.binary(BinOp::LogicalAnd, counted, going)?;
    memory.gates(fb, record, move |fb, gate| {
        let zero = k32(fb, 0)?;
        for axis in 0..3 {
            let source = memory.at(fb, gate, 4 * axis)?;
            let count = memory.load(fb, source)?;
            let count = fb.select(runs, count, zero)?;
            let out = memory.at(fb, gate, PredicationStep::GATE_OUT_OFFSET + 4 * axis)?;
            memory.store(fb, out, count)?;
        }
        Ok(())
    })?;
    fb.ret(None)
}
