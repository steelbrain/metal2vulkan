//! Command predication: Metal's private compute-encoder calls
//! `encodeStartIf:offset:comparison:referenceValue:`, `encodeStartElse` and `encodeEndIf`, and its
//! loops, `encodeStartWhile:offset:comparison:referenceValue:` to `encodeEndWhile` and
//! `encodeStartDoWhile` to `encodeEndDoWhile:offset:comparison:referenceValue:`, as Metal runs
//! them (measured on an M3 Max under macOS 27.0.1, the release every device answers as,
//! `metal_release.rs`; an M4 under macOS 27.0.1 gave the same answers for every stream both
//! ran).
//!
//! # What Metal does
//!
//! - **A unit** of predication is a run of a command buffer's compute encoders of one dispatch
//!   type with nothing between them. A blit, render or resource state encoder, an event wait or
//!   signal, an encoder of the other dispatch type, the end of an encoder that updated a fence
//!   (`updateFence:`), or the command buffer's end ends it; a scheduled handler, a fence wait and
//!   a memory barrier do not (nor does a binary semaphore wait or signal, which Metal has not). A
//!   block may span its unit's encoders, a loop's included; a record after the unit's end is the
//!   next unit's.
//! - **The condition** is `word OP reference` (`PredicationCompare`, Metal's codes 0 to 5, and
//!   `==` for any other code), the word the `u32` at the offset rounded down to a multiple of 4,
//!   read when the GPU reaches the record: after the work before it, with or without a barrier,
//!   and after a CPU write made before then.
//! - **A skipped region** runs none of its dispatches, of any form, and none of its executions
//!   of an indirect command buffer. Setters, barriers, resource uses and fences take effect in
//!   encode order whatever runs (a fence updated inside a skipped block or a dropped range is
//!   updated).
//! - **Nesting** has no limit (2^20 levels of Ifs measured, six of loops).
//! - **A loop** runs its commands again and again: a While while its condition holds, tested
//!   before each pass, a DoWhile once and again while its End's condition holds, tested after
//!   each pass; each word read when the GPU reaches the test. Each pass runs every command with
//!   the encoder's state it was encoded with (its bindings and bytes), and the state the body's
//!   last setters left stands after the loop. Metal bounds no loop (16 million passes ran).
//! - **An End** of the open block answers YES.
//! - **A misused record kills** (an End of another kind of block, a second Else or an Else on a
//!   loop, a Start or an `encodeEndDoWhile` on a nil buffer inside a block): the outermost open
//!   block, with every block in it, runs nothing from its start on, the commands of the unit's
//!   earlier encoders included; the record answers NO. Every End then answers NO but the one
//!   that closes that outermost block, which answers YES, and the unit runs again after it. A
//!   further misuse inside it changes nothing. A dispatch encoded in a killed block after the
//!   kill is not one Metal emits (below).
//! - **An unbalanced record** (an End or Else with no block open, a Start or an
//!   `encodeEndDoWhile` on a nil buffer with none) answers NO and drops a range of its unit's
//!   work: from its anchor, the start of the unit's last block opened with no block open (or the
//!   unit's start where none was), to the next record (or the unit's end). Every dispatch in the
//!   range runs nothing, the ones encoded before the record included; the next record runs as
//!   though the range were not there. Where the range holds no dispatch Metal emitted (setters,
//!   resource uses, barriers and fences are none), the block the next record opens is killed
//!   from its start instead (as a misused record kills it). Further unbalanced records before the
//!   next record widen the range.
//! - **A block open where its unit ends**: a killed one is harmless; a live one fails the command
//!   buffer in a GPU recovery where the unit's commands skip there (a failed If or While, a held
//!   If's Else, any block inside one that skips), and is harmless where they run (a held If or
//!   While, which then ran once, a DoWhile, a failed If's Else). The failure is Metal's
//!   `MTLCommandBufferErrorInternal` (`PageFault` in some runs), and how much else of the command
//!   buffer ran varies run to run.
//! - **A word past its buffer's end** is what Metal's address space holds there (the buffer's
//!   GPU address plus the offset, which does not wrap past 2^32): zero within the 256 bytes Metal
//!   rounds a lone buffer up to; past them another buffer's bytes where Metal placed one, else
//!   zero; the heap's bytes past a buffer placed in a heap, zero past the heap's end. Only a
//!   record that tests its word reads it: an `encodeEndDoWhile:...` that is unbalanced, misused
//!   or closes a killed DoWhile reads no word, and answers as on any word.
//!
//! # What the runtime builds
//!
//! - **A skip counter** per unit, in the command buffer's upload arena, and **a step** at each
//!   record that moves it: one invocation of the crate's predication step module
//!   (`bind::predication_step_module`), behind a barrier from everything before it and with one
//!   to the commands after it. On a device with device addresses (and `shaderInt64`) the step
//!   reaches its record by the address it is pushed, and every other place by the address the
//!   record names; on any other device (`bind::PredicationStepForm::Bound`) it binds its record's
//!   region, its unit's counter, Metal's word (its buffer at the word's offset aligned down to
//!   `minStorageBufferOffsetAlignment`), its loop's flag and a `UnitEnd`'s check word as five
//!   storage buffers of a set of its own, and the record names each by its byte offset there.
//! - **Gated dispatches.** A dispatch encoded inside a block, or after the unit's first block
//!   opened with none open (so a later unbalanced record can drop it), runs by
//!   `vkCmdDispatchIndirect` from a gate the step before it writes: its group counts where the
//!   counter is 0, else zero. A gate lies in the arena, named in the step's table, or in the
//!   bound form in the step's own record, which holds 16 (as many as one dispatch needs, where
//!   more): a dispatch past them first records a further step, a `PredicationOp::Hold`, which
//!   leaves the counter as it is and holds the next gates. A dispatch on the flat path gates its
//!   preprocess, whose command then stays zero. A dispatch or an execution a host step records reads the counter there, once the
//!   work before it completed, and records nothing where it is not 0; an execution inside a block
//!   is always recorded so (`Degradation::HostReplayedIndirectCommands`).
//! - **A word past its buffer's end** ([`Word`]) is read as Metal reads it: the heap's bytes
//!   past a buffer placed in a heap, a zero word of the command buffer's within the 256 bytes
//!   Metal rounds a lone buffer up to; elsewhere, where Metal reads memory unrelated to the
//!   buffer, the zero word too, announced (`Degradation::PredicationWordPastBuffer`).
//! - **A kill** rewrites the outermost open block's start step as `PredicationOp::Dead`, whose
//!   End is then `PredicationOp::EndDead`; this model gives Metal's answers.
//! - **A dropped range** rewrites its anchor's step as `PredicationOp::Drop` and every step since
//!   as `PredicationOp::Hold` (or, anchored at the unit's start, records a `Drop` step at the
//!   record), so its gated dispatches run nothing; the next record first records a
//!   `PredicationOp::Resume`. The unit's dispatches before its first record run ungated, so a
//!   program that never records pays nothing for predication; where that first record is
//!   unbalanced and such dispatches came before it, what the unit's encoders recorded before the
//!   record is left out of the command buffer's work instead: its earlier encoders' parts and
//!   what the open encoder recorded so far (its `VkCommandBuffer`s and the work it left apart)
//!   are never submitted, as a failed encoder's are, their fences' waits kept, and the open
//!   encoder forgets the textures it shadows unread and goes on, from the record's `Drop` step,
//!   in a part of its own (`EncoderCore::leave_out_recorded`).
//! - **A block open where its unit ends**: each compute encoder that ends with a live block open
//!   records a `PredicationOp::UnitEnd` step, which writes whether the counter skips into the
//!   unit's check word; where the unit ended so, a set word fails the command buffer at its
//!   completion (`CommandBufferError::PredicationOpenAtUnitEnd`, Metal's code, with the index of
//!   the record that opened the unit's outermost open block), before its
//!   completed handlers run. The rest of the command buffer runs.
//! - **A loop** is laid down as copies of its commands: its first copy is what the encoders
//!   encode between its Start and its End, which they note as they go (each command's public call
//!   with the encoder's state it ran with, `Captures`, carried from an encoder to the next of its
//!   unit, the boundary a barrier and the ended encoder's residency); at the End the encoder
//!   encodes the further copies ([`copies_of`]): 64 in all, or 16 inside another loop, but no more
//!   than keep the steps and gated dispatches the loop lays down within [`LAID_BUDGET`] (counting
//!   the first copy's, inner loops' copies included), and at least one, so neither a long body
//!   nor nesting multiplies a loop's commands past what a driver takes in one command buffer;
//!   each behind a copy start step and closed by a copy end step, and then a step past them. The
//!   copies share a flag the first failing test sets, after which every copy skips; the first
//!   copy's start takes the flag as clear (`PredicationOp::WhileFirst`, `PredicationOp::DoFirst`).
//!   A step past the copies halts the unit where the loop would run another: it sets the unit's
//!   halt word (beside its counter) to the loop's id, and every later step of the unit then leaves
//!   its words and runs nothing (a host step's command too). A loop inside another's copy is laid
//!   down again in each, with an id of its own. A killed loop gets no copies.
//! - **A loop's laying runs again** until the loop ends, as Metal runs a loop on without bound
//!   (`command_buffer/rerun.rs`): the outermost loop's laying goes in command buffers of its own,
//!   cut at its start (and at each later encoder of its unit it spans) and ended by its check, a
//!   host step that reads the halt word once the laying completed and, while it is set, submits
//!   the laying again; the halted loop's first copy's start clears it, so the loop goes on where
//!   it stopped, and nothing encoded before the loop runs again. Work the encoders left apart in
//!   the laying (an indirect dispatch or an indirect command buffer's execution a host step
//!   records, or a dispatch whose variant records in a command buffer of its own) splits it: each
//!   run again records it again where it lies, as its first run did. A dispatch inside a laying
//!   whose pipelines are being made waits for them, so the laying is not split for it.
//! - **Encoded again at a rebind** (`command_buffer/reencode.rs`), as Metal resolves what a
//!   command names as it submits the command buffer: a compute encoder whose unit laid no loop
//!   and left no work out, and of whose unit's predication nothing reached in over its open and
//!   nothing of its own reaches out over its end ([`Predication::alone`]), logs its Ifs, Elses
//!   and Ends among its calls (`replay.rs`), and is encoded again in its unit begun anew as it
//!   opened ([`Predication::replaying`]: the next record's index and the run of compute encoders
//!   as they were), over a counter, steps, gates and a check word of its own; the first
//!   encoding's go unsubmitted with its recording. So an encoder of a unit that recorded no
//!   predication at all, every encoder of a unit before the one that laid its first step, and
//!   that one where its unit ended with it. Any other compute encoder of a command buffer that
//!   used predication keeps no log — the commands of a unit's encoders after its first step run
//!   from gates that step's encoder's recording writes, which an encoding again replaces — and a
//!   loop's record ends the log.

use super::buffer::Buffer;
use super::command_buffer::rerun::{LoopCheck, Piece};
use super::compute::{ComputeCommandEncoder, EncoderState};
use super::deferred::Deferred;
use super::degradation::Degradation;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use super::replay::ComputeCall;
use super::residency::Residency;
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::{Upload, Uploads};
use super::values::Size;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    PredicationCompare, PredicationOp, PredicationStep, PredicationStepForm, PredicationStepModule,
};
use metal2vulkan::metal::{BarrierScope, DispatchType};
use std::ops::Range;
use std::sync::Arc;

/// A command buffer's command predication (the module docs): its current unit, the gated runs of
/// its records, and the check words of the units that ended with a live block open.
#[derive(Default)]
pub(super) struct Predication {
    unit: Option<Unit>,
    /// Each step's record and the gates of the dispatches it runs, until the commit writes
    /// their tables.
    runs: Vec<Run>,
    /// The check words of the units that ended with a live block open, each set where the unit's
    /// commands skipped there, with the index of the record that opened the unit's outermost
    /// open block: the command buffer fails at its completion
    /// (`CommandBufferError::PredicationOpenAtUnitEnd`).
    open_checks: Vec<(Upload, u32)>,
    /// The command buffer's command-predication calls that returned `Ok` so far: the index of
    /// the next (the calls a loop's copies replay not counted).
    records: u32,
    /// The loops laid down so far: the next one's id ([`PredicationStep::id`]) is one more.
    loops: u32,
    /// The laying of the unit's outermost open loop, from its start (the module docs).
    laying: Option<Laying>,
    /// The steps and gated dispatches recorded so far: what a loop's copies lay down is counted
    /// from it ([`copies_of`]).
    laid: u64,
    /// An unbalanced record just dropped what its unit's encoders recorded before it, from the
    /// encoder at this place among the command buffer's ended encoders on, which the open encoder
    /// leaves out of the command buffer's work ([`take_left_out`](Self::take_left_out)).
    left_out: Option<usize>,
    /// What each compute encoder found as it opened, in order ([`Opened`]).
    opened: Vec<Opened>,
    /// The places among the command buffer's ended encoders of the compute encoders that encode
    /// again, in their unit begun anew, exactly what they first encoded
    /// ([`alone`](Self::alone)): a rebind may encode one again
    /// ([`replaying`](Self::replaying)).
    alone: Vec<usize>,
}

/// What a compute encoder found as it opened: its place among the command buffer's ended
/// encoders, the event waits and signals encoded before it, its dispatch type, and the index of
/// the command buffer's next record.
struct Opened {
    place: usize,
    events: (usize, usize),
    dispatch_type: DispatchType,
    records: u32,
}

/// The laying of a unit's outermost loop: its pieces in encoding order (`command_buffer/rerun.rs`),
/// the `VkCommandBuffer`s its encoders recorded it into, each begun to be submitted again, from
/// the one begun at the loop's start, and the work they left apart between them; whether a loop
/// in it was laid down as copies, which may halt the unit; and the command buffer's parts
/// recorded before its first, after which its own lie.
pub(super) struct Laying {
    pieces: Vec<Piece>,
    copied: bool,
    from: usize,
}

impl Laying {
    /// Its pieces, in order, and the parts before its first.
    pub(super) fn into_parts(self) -> (Vec<Piece>, usize) {
        (self.pieces, self.from)
    }
}

/// The copies of a loop no other loop holds, at most.
const FIRST_COPIES: u64 = 64;

/// The copies of a loop inside another, at most.
const NESTED_COPIES: u64 = 16;

/// The steps and gated dispatches a loop's copies lay down, at most, where the first copy alone
/// does not pass it: MoltenVK 1.4.2 on an Apple M3 Max ended the process (Metal's
/// `IOGPUMetalDeviceShmem` allocation failed) at a loop of six nested loops laid as 64, 16 and 16
/// copies of the outer three (some 400,000 commands in one command buffer).
const LAID_BUDGET: u64 = 16_384;

/// The copies laid down in all of a loop with `depth` loops open around it whose first copy laid
/// down `first` steps and gated dispatches: [`FIRST_COPIES`] or [`NESTED_COPIES`], but no more than
/// keep them within [`LAID_BUDGET`], and at least one.
fn copies_of(depth: usize, first: u64) -> u64 {
    let most = if depth == 0 {
        FIRST_COPIES
    } else {
        NESTED_COPIES
    };
    (LAID_BUDGET / first.max(1)).clamp(1, most)
}

/// A record's condition: the `u32` Metal reads, compared so with `reference`: at `place`, its
/// device address, or in the bound form its byte offset in `binding` (the `VkBuffer`, the offset
/// and the range a step binds it at, the offset aligned down as the device asks).
#[derive(Clone)]
pub(super) struct Test {
    /// The buffer the word lies in and its byte there: the record's buffer at Metal's offset
    /// rounded down to a multiple of 4, or past its end the heap it lies in; `None` where the
    /// step reads a zero word of the command buffer's ([`Word`]).
    word: Option<(Buffer, u64)>,
    place: u64,
    binding: Option<(vk::Buffer, u64, u64)>,
    compare: PredicationCompare,
    reference: u32,
}

/// Where Metal reads a record's word: the `u32` at the buffer's GPU address plus its offset rounded
/// down to a multiple of 4, in Metal's address space (measured on an M3 Max under macOS 27.0.1).
enum Word {
    /// In the buffer, or past its end in the heap it lies in: those bytes.
    In(Buffer, u64),
    /// Past a lone buffer's end within the 256 bytes Metal rounds its length up to (Metal places
    /// small buffers 256 bytes apart): zero.
    Zero,
    /// Past those, or past its heap's end: what Metal's address space holds there (another
    /// buffer's bytes, measured, or zero, at offsets up to 2^36 and past 2^32, which do not wrap),
    /// memory unrelated to the buffer. Zero, announced
    /// (`Degradation::PredicationWordPastBuffer`).
    Unrelated,
}

impl Word {
    /// Where Metal reads the word at byte `at` of `buffer`.
    fn of(buffer: &Buffer, at: u64) -> Word {
        let within = |end: u64| at.checked_add(4).is_some_and(|word| word <= end);
        if within(buffer.length()) {
            return Word::In(buffer.clone(), at);
        }
        if let Some((heap, start)) = buffer.heap_bytes() {
            return match start.checked_add(at) {
                Some(at) if at.checked_add(4).is_some_and(|end| end <= heap.length()) => {
                    Word::In(heap.clone(), at)
                }
                _ => Word::Unrelated,
            };
        }
        match buffer.length().checked_next_multiple_of(256) {
            Some(slot) if within(slot) => Word::Zero,
            _ => Word::Unrelated,
        }
    }
}

impl Test {
    /// The condition of a record on `buffer` at `offset`, of Metal's `comparison` code and
    /// `reference`, on `device`: a word past the buffer's end reads as Metal reads it ([`Word`]),
    /// a zero word made in `uploads` where it reads zero.
    fn new(
        device: &Device,
        uploads: &mut Uploads,
        buffer: &Buffer,
        offset: u64,
        (comparison, reference): (u64, u32),
    ) -> Result<Test, Error> {
        let word = match Word::of(buffer, offset & !3) {
            Word::In(buffer, at) => Some((buffer, at)),
            Word::Zero => None,
            Word::Unrelated => {
                let length = buffer.length();
                device.degrade(Degradation::PredicationWordPastBuffer, || {
                    (
                        format!(
                            "a command predication record's word at offset {offset} of a buffer \
                             of {length} bytes"
                        ),
                        "Metal reads memory unrelated to the buffer there".to_owned(),
                    )
                });
                None
            }
        };
        let (place, binding) = match &word {
            Some((buffer, at)) => match bound(device) {
                false => {
                    let address = buffer
                        .device_address()
                        .ok_or_else(|| internal("a condition word without a device address"))?;
                    (address + at, None)
                }
                true => {
                    let (raw, start) = buffer.at(*at);
                    let limits = &device.properties().description.limits;
                    let within = start % limits.min_storage_buffer_offset_alignment.max(4);
                    (within, Some((raw, start - within, within + 4)))
                }
            },
            None => {
                let zero = uploads.upload(device, &[], 16)?;
                match bound(device) {
                    false => (addressed(&zero)?, None),
                    true => (0, Some((zero.raw(), zero.offset(), 4))),
                }
            }
        };
        Ok(Test {
            word,
            place,
            binding,
            compare: PredicationCompare::from_metal(comparison),
            reference,
        })
    }

    /// `step` testing this condition.
    fn of(&self, step: PredicationStep) -> PredicationStep {
        step.testing(self.place, self.compare, self.reference)
    }
}

/// Whether `device` runs the bound form of the step module
/// (`bind::PredicationStepForm::Bound`): no device addresses, so each step binds its places.
fn bound(device: &Device) -> bool {
    PredicationStepForm::of(device.compile_device()) == PredicationStepForm::Bound
}

/// Where a step's record names `upload` on `device`: its device address, or in the bound form
/// byte 0 of its binding (each region is bound at its own offset).
fn place(device: &Device, upload: &Upload) -> Result<u64, Error> {
    match bound(device) {
        true => Ok(0),
        false => addressed(upload),
    }
}

/// A step to record: its record, and in the bound form what it binds besides its record and
/// Metal's word.
#[derive(Clone)]
pub(super) struct Step {
    record: Upload,
    bound: Option<Places>,
}

/// What a bound step binds besides its record and Metal's word: its unit's counter (with its halt
/// word), and its loop's flag or a `UnitEnd`'s check word, where it has one (its record where
/// not).
#[derive(Clone)]
struct Places {
    counter: Upload,
    flag: Option<Upload>,
    check: Option<Upload>,
}

/// The gates a bound step's record holds before a dispatch needs a further step's
/// (`Predication::gates`).
const BOUND_GATES: u64 = 16;

/// A record's condition as called: the word at `offset` of `buffer`, compared by Metal's
/// `comparison` code with `reference`, read (made a [`Test`]) only where the record tests it, as
/// Metal reads it only there.
#[derive(Clone, Copy)]
pub(super) struct Condition<'a> {
    buffer: &'a Buffer,
    offset: u64,
    comparison: u64,
    reference: u32,
}

impl Condition<'_> {
    /// Its [`Test`] on `device`, a zero word made in `uploads` where it reads one.
    fn test(self, device: &Device, uploads: &mut Uploads) -> Result<Test, Error> {
        let arguments = (self.comparison, self.reference);
        Test::new(device, uploads, self.buffer, self.offset, arguments)
    }
}

/// One unit of predication.
struct Unit {
    /// The place among the command buffer's ended encoders its first compute encoder takes.
    first: usize,
    /// The place among the command buffer's ended encoders its last compute encoder takes.
    place: usize,
    /// The event waits and signals encoded before it.
    events: (usize, usize),
    dispatch_type: DispatchType,
    /// Its skip counter, once a record needs it.
    counter: Option<Upload>,
    /// Its open blocks, the outermost first.
    stack: Vec<Block>,
    /// A misuse killed the outermost open block (the module docs).
    dead: bool,
    /// The run (in [`Predication::runs`]) of its last step, which gates the dispatches encoded
    /// now.
    run: Option<usize>,
    /// What an unbalanced record would drop, or drops (the module docs).
    range: Droppable,
    /// Its open encoder updated a fence: the unit ends with that encoder.
    closing: bool,
    /// The word its encoders' `UnitEnd` steps write, made at the first that ended with a live
    /// block open.
    check: Option<Upload>,
    /// The bodies of the loops its last encoder ended with open, which its next encoder carries
    /// on (`Captures`).
    carried: Vec<Vec<(EncoderState, Event)>>,
    /// A loop was laid in it.
    looped: bool,
    /// An unbalanced record left out what its encoders recorded before it.
    left: bool,
    /// The places among the command buffer's ended encoders of its compute encoders that opened
    /// with no step of its laid ([`Unit::fresh`]), in order: its first encoder's, and each
    /// later one's until one laid a step, so a prefix of its encoders
    /// ([`alone`](Predication::alone)).
    opened_fresh: Vec<usize>,
}

/// The range of a unit an unbalanced record drops (the module docs): from the anchor to the next
/// record.
#[derive(Default)]
struct Droppable {
    /// The start step of the unit's last block opened with none open; `None`: the unit's start.
    anchor: Option<Upload>,
    /// Every step recorded since the anchor, which a drop holds.
    since: Vec<Upload>,
    /// A dispatch Metal emitted since the anchor (or the unit's start): one encoded outside a
    /// killed block.
    work: bool,
    /// An unbalanced record dropped from the anchor on, and no record came since.
    dropping: bool,
}

/// An open block.
struct Block {
    /// Its start step's record, whose op a kill rewrites.
    start: Upload,
    /// The index of the call that opened it among its command buffer's command-predication calls
    /// ([`Predication::records`]); a block a loop's copy opens has its loop End's.
    record: u32,
    kind: Kind,
}

/// What an open block is.
#[derive(Clone)]
enum Kind {
    /// An If, and whether it has had its Else.
    If { elsed: bool },
    /// A loop's copy.
    Loop(Looping),
}

/// A loop as its copies are laid down: a While's condition, or a DoWhile's (`None`: its End names
/// it), the flag its copies share, which its first failing test sets, and its id, at which its
/// step past its copies halts its unit.
#[derive(Clone)]
pub(super) struct Looping {
    test: Option<Test>,
    flag: Upload,
    id: u32,
    /// The command buffer's steps and gated dispatches laid down before it ([`Predication::laid`]).
    laid_at: u64,
}

/// An End of a loop: Metal's answer, the step to record, if any, whether it closed the loop
/// (whose body's capture then ends), and, where the loop is live (no misused record killed it),
/// the loop, with a DoWhile's condition, whose further copies the encoder lays down.
struct LoopEnd {
    answer: bool,
    record: Option<Step>,
    closed: bool,
    copies: Option<(Looping, Option<Test>)>,
}

/// A step's record and the gates of the dispatches it runs: their device addresses, or in the
/// bound form their byte offsets in its record's region, which holds `capacity` of them.
struct Run {
    record: Upload,
    gates: Vec<u64>,
    capacity: Option<u64>,
}

impl Unit {
    fn new(place: usize, events: (usize, usize), dispatch_type: DispatchType) -> Unit {
        Unit {
            first: place,
            place,
            events,
            dispatch_type,
            counter: None,
            stack: Vec::new(),
            dead: false,
            run: None,
            range: Droppable::default(),
            closing: false,
            check: None,
            carried: Vec::new(),
            looped: false,
            left: false,
            opened_fresh: vec![place],
        }
    }

    /// Whether no step of its has been laid: its skip counter is made with its first step
    /// ([`Predication::step_holding`]), and every other field a record leaves — its open blocks,
    /// its kill, its run, its droppable range's anchor and held steps, its check word and its
    /// carried loop bodies — is set only through one, which the assertion below holds it to. Its
    /// range's `work` alone is not, so a fresh unit can hold it where a unit begun anew would
    /// not: a dispatch Metal emitted sets it. Its one reader,
    /// [`resume`](Predication::resume), is an encoder whose own encoding laid the work it reads,
    /// or one of a unit [`alone`](Predication::alone) excludes, which is the argument there.
    fn fresh(&self) -> bool {
        let fresh = self.counter.is_none();
        // What `alone` rests on: a unit with no step laid carries nothing a unit begun anew
        // would not hold, so an encoder that opened on one encodes again exactly.
        debug_assert!(
            !fresh
                || (self.stack.is_empty()
                    && !self.dead
                    && self.run.is_none()
                    && self.range.anchor.is_none()
                    && self.range.since.is_empty()
                    && !self.range.dropping
                    && self.check.is_none()
                    && self.carried.is_empty()
                    && !self.looped),
            "a unit of predication with no step laid carries predication state"
        );
        fresh
    }

    /// Kill the outermost open block (the module docs), once.
    fn kill(&mut self) {
        if self.dead {
            return;
        }
        if let Some(outermost) = self.stack.first() {
            self.dead = true;
            outermost
                .start
                .write_at(PredicationStep::OP_OFFSET, &PredicationOp::Dead.word());
        }
    }

    /// Whether its dispatches run from gates: inside a block, after its first block opened with
    /// none open, or in a dropped range.
    fn gated(&self) -> bool {
        !self.stack.is_empty() || self.range.anchor.is_some() || self.range.dropping
    }

    /// Whether a block is open that no misuse killed.
    fn live_open(&self) -> bool {
        !self.stack.is_empty() && !self.dead
    }

    /// The open encoder's dispatches since the last record (`worked`): Metal emitted them where
    /// no misuse killed the open block.
    fn note_work(&mut self, worked: bool) {
        if worked && !self.dead {
            self.range.work = true;
        }
    }
}

impl Predication {
    /// A compute encoder of `dispatch_type` opened at `place` among the ended encoders, after
    /// `events` event waits and signals: it joins the current unit, taking the bodies of the loops
    /// open across its start, or that one ends and it starts the next.
    pub(super) fn enter(
        &mut self,
        place: usize,
        events: (usize, usize),
        dispatch_type: DispatchType,
    ) -> Vec<Vec<(EncoderState, Event)>> {
        self.opened.push(Opened {
            place,
            events,
            dispatch_type,
            records: self.records,
        });
        if let Some(unit) = &mut self.unit {
            if !unit.closing
                && unit.place + 1 == place
                && unit.events == events
                && unit.dispatch_type == dispatch_type
            {
                if unit.fresh() {
                    unit.opened_fresh.push(place);
                }
                unit.place = place;
                return std::mem::take(&mut unit.carried);
            }
        }
        self.end_unit();
        self.unit = Some(Unit::new(place, events, dispatch_type));
        Vec::new()
    }

    /// The open compute encoder updated a fence: its unit ends with it.
    pub(super) fn fence_updated(&mut self) {
        if let Some(unit) = &mut self.unit {
            unit.closing = true;
        }
    }

    /// The current unit ends: one with a live block open keeps its check word for the completion,
    /// with the index of the record that opened its outermost open block. Its encoders that a
    /// rebind may encode again alone are noted ([`alone`](Self::alone)).
    fn end_unit(&mut self) {
        // A loop open where its unit ends is closed by no End: its laying is checked by none.
        self.laying = None;
        if let Some(unit) = self.unit.take() {
            if !unit.looped && !unit.left {
                // Each encoder that opened with no step of the unit laid and laid none itself:
                // the unit was never gated while it was open, so no command of its reads a gate
                // and it leaves no step for a later encoder of the unit to read. Those are
                // every one of `opened_fresh` but its last where the unit laid a step, and every
                // one where it laid none.
                let stepped = !unit.fresh();
                // Where the unit laid a step, the last of `opened_fresh` is the encoder that
                // laid its first: that one gates the commands of the unit's later encoders, so
                // it encodes again alone only where the unit ended with it.
                let first_step_at = unit.opened_fresh.len().saturating_sub(1);
                for (at, &place) in unit.opened_fresh.iter().enumerate() {
                    if !stepped || at < first_step_at || place == unit.place {
                        self.alone.push(place);
                    }
                }
            }
            if let (true, Some(check), Some(outermost)) =
                (unit.live_open(), unit.check, unit.stack.first())
            {
                self.open_checks.push((check, outermost.record));
            }
        }
    }

    /// A command-predication call returned `Ok`: the next call's index is one more.
    pub(super) fn counted(&mut self) {
        self.records += 1;
    }

    /// Whether the compute encoder at `place` among the command buffer's ended encoders encodes
    /// again, in its unit begun anew ([`replaying`](Self::replaying)), exactly what it first
    /// encoded, as its commit found: a rebind may encode it again
    /// (`command_buffer/reencode.rs`). It does where its unit laid no loop and left no work out
    /// and, of the unit's predication, nothing reached in over its open and nothing of its own
    /// reaches out over its end:
    ///
    /// - **Nothing reached in**: no step of the unit was laid before it opened ([`Unit::fresh`]),
    ///   so a unit begun anew holds what the unit held then — its counter, open blocks, kill,
    ///   run, droppable range, check word and carried loop bodies all come of a step.
    /// - **Nothing reaches out**: the unit laid no step while the encoder was open, so it was
    ///   never gated (`Unit::gated`) and left the unit as it found it; or the encoder laid the
    ///   unit's first step and the unit ended with it, so its steps, gates and check word are
    ///   read by its own commands alone.
    ///
    /// So every encoder of a unit that recorded no predication at all, every encoder before the
    /// one that laid its unit's first step, and that one where its unit ended with it. A unit
    /// whose first step lies in an earlier encoder than its last keeps none of the encoders from
    /// that step on: the commands of the unit's later encoders run from gates the step before
    /// them writes, which lie in that encoder's recording, and an encoding again lays steps and
    /// gates of its own, leaving those written by nothing. A gate holds the dispatch's group
    /// counts where the step writes them and nothing where it does not
    /// (`PredicationStep::GATE_OUT_OFFSET`, which only a step fills), so such a command would
    /// run nothing where Metal runs it. Encoding a run of a unit's encoders again together,
    /// which that shape needs, is not done.
    pub(super) fn alone(&self, place: usize) -> bool {
        self.alone.contains(&place)
    }

    /// The predication the compute encoder at `place` encodes again with, where it encodes again
    /// exactly ([`alone`](Self::alone)): its unit begun anew as it opened and the next record's
    /// index as it was then. Its records and dispatches then step
    /// and gate as the first encoding's did, over a counter, steps and gates of their own; the
    /// first encoding's go with its recording, never submitted.
    pub(super) fn replaying(&self, place: usize) -> Option<Predication> {
        if !self.alone(place) {
            return None;
        }
        let opened = self.opened.iter().find(|opened| opened.place == place)?;
        Some(Predication {
            unit: Some(Unit::new(place, opened.events, opened.dispatch_type)),
            records: opened.records,
            ..Predication::default()
        })
    }

    /// Take `replayed`'s check words, an encoder's encoded again in place of its first
    /// encoding's ([`replaying`](Self::replaying)), committed: the command buffer fails at its
    /// completion where one is set. The first encoding's, which its `UnitEnd` steps no longer
    /// write, read zero.
    pub(super) fn take_checks(&mut self, replayed: Predication) {
        let Predication { open_checks, .. } = replayed;
        self.open_checks.extend(open_checks);
    }

    /// The command buffer commits: its last unit ends, and each run's table is written into its
    /// record. Refused as the arena refuses a table.
    pub(super) fn commit(&mut self, uploads: &mut Uploads, device: &Device) -> Result<(), Error> {
        self.end_unit();
        // A bound step's gates lie in its record, their count written with them.
        let tabled = |run: &&Run| run.capacity.is_none() && !run.gates.is_empty();
        for run in self.runs.iter().filter(tabled) {
            let bytes = PredicationStep::table(addressed(&run.record)?, &run.gates);
            let table = uploads.upload(device, &bytes, bytes.len() as u64)?;
            let address = addressed(&table)?;
            run.record
                .write_at(PredicationStep::TABLE_OFFSET, &address.to_le_bytes());
        }
        self.runs.clear();
        Ok(())
    }

    /// The open encoder's unit, with the encoder's dispatches since the last record (`worked`)
    /// noted.
    fn unit(&mut self) -> Result<&mut Unit, Error> {
        self.unit
            .as_mut()
            .ok_or_else(|| internal("a predication record outside a unit"))
    }

    /// A step of `step` on the unit's counter (made now where it has none), whose loop's flag or
    /// check word is `places` where it has one: what to record, which
    /// gates the dispatches encoded after it until the next step. `step` takes the counter's
    /// place ([`place`]).
    fn step(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        places: (Option<&Upload>, Option<&Upload>),
        step: impl FnOnce(u64) -> PredicationStep,
    ) -> Result<Step, Error> {
        self.step_holding(uploads, device, (places, BOUND_GATES), step)
    }

    /// [`step`](Self::step), whose record holds `gates` gates in the bound form.
    fn step_holding(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        ((flag, check), gates): ((Option<&Upload>, Option<&Upload>), u64),
        step: impl FnOnce(u64) -> PredicationStep,
    ) -> Result<Step, Error> {
        let counter = match self.unit()?.counter.clone() {
            Some(counter) => counter,
            None => uploads.upload(device, &[], 16)?,
        };
        let bound = bound(device);
        let bytes = match bound {
            true => PredicationStep::bound_bytes(gates),
            false => PredicationStep::RECORD_BYTES,
        };
        let record = uploads.upload(device, &[], bytes)?;
        self.laid += 1;
        let at = place(device, &record)?;
        record.write(&step(place(device, &counter)?).record(at));
        let run = self.runs.len();
        self.runs.push(Run {
            record: record.clone(),
            gates: Vec::new(),
            capacity: bound.then_some(gates),
        });
        let unit = self.unit()?;
        unit.counter = Some(counter.clone());
        unit.run = Some(run);
        if unit.range.anchor.is_some() {
            unit.range.since.push(record.clone());
        }
        let places = bound.then(|| Places {
            counter,
            flag: flag.cloned(),
            check: check.cloned(),
        });
        Ok(Step {
            record,
            bound: places,
        })
    }

    /// Where a unit ended with a live block open where its commands skipped, as Metal's GPU
    /// recovery fails the command buffer, the index of the record that opened the first such
    /// unit's outermost open block: read once the command buffer completed.
    pub(super) fn open_at_unit_end(&self) -> Option<u32> {
        (self.open_checks.iter())
            .find(|(check, _)| u32::from_le_bytes(check.read::<4>()) != 0)
            .map(|(_, record)| *record)
    }

    /// An unbalanced record (the module docs): its range drops from the anchor on; the step to
    /// record, where the range starts at the unit's start. Where the unit's dispatches before the
    /// record ran ungated, what its encoders recorded so far is left out
    /// ([`take_left_out`](Self::take_left_out)).
    fn unbalanced(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<Option<Step>, Error> {
        let range = &mut self.unit()?.range;
        if range.dropping {
            return Ok(None);
        }
        if let Some(anchor) = &range.anchor {
            anchor.write_at(PredicationStep::OP_OFFSET, &PredicationOp::Drop.word());
            for step in &range.since {
                step.write_at(PredicationStep::OP_OFFSET, &PredicationOp::Hold.word());
            }
            range.dropping = true;
            return Ok(None);
        }
        let record = self.step(uploads, device, (None, None), |counter| {
            PredicationStep::new(counter, PredicationOp::Drop)
        })?;
        let unit = self.unit()?;
        unit.range.dropping = true;
        unit.left |= unit.range.work;
        let left_out = unit.range.work.then_some(unit.first);
        self.left_out = left_out.or(self.left_out);
        Ok(Some(record))
    }

    /// Where the last record dropped what its unit's encoders recorded before it, dispatches the
    /// runtime ran ungated among it (the module docs): the place of the unit's first encoder
    /// among the command buffer's ended encoders. The open encoder leaves out of the command
    /// buffer's work every part of those encoders and what it recorded itself so far, before the
    /// record's step.
    pub(super) fn take_left_out(&mut self) -> Option<usize> {
        self.left_out.take()
    }

    /// A Start with no block open, before its own step: where a range is dropping, the `Resume`
    /// step to record first, and whether the block it opens is killed from its start (the range
    /// held no work Metal emitted).
    fn resume(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<(Option<Step>, bool), Error> {
        let unit = self.unit()?;
        if !unit.stack.is_empty() || !unit.range.dropping {
            return Ok((None, false));
        }
        // The range runs from its anchor to this record, so the work it holds is what Metal
        // emitted before the record that dropped it and what the encoders since laid down: a
        // dispatch encoded after that record lies in the range too, and is one Metal emits. An
        // encoder [`alone`](Self::alone) reads only work its own encoding laid down: the work
        // from before the drop set [`left`](Unit::left), which `alone` excludes
        // ([`unbalanced`](Self::unbalanced) sets the two together), and the work since was laid
        // by the encoder that laid the `Drop` step or a later one of the unit, of which `alone`
        // keeps only the encoder its unit ended with.
        let dead = !unit.range.work;
        let record = self.step(uploads, device, (None, None), |counter| {
            PredicationStep::new(counter, PredicationOp::Resume)
        })?;
        self.unit()?.range.dropping = false;
        Ok((Some(record), dead))
    }

    /// Open a block of `kind` started by `record`: one with none open is the unit's anchor, and is
    /// killed from its start where `dead` (the module docs).
    fn open(&mut self, record: &Step, kind: Kind, dead: bool) -> Result<(), Error> {
        let index = self.records;
        let unit = self.unit()?;
        if unit.stack.is_empty() {
            unit.range = Droppable {
                anchor: Some(record.record.clone()),
                ..Droppable::default()
            };
        }
        unit.stack.push(Block {
            start: record.record.clone(),
            record: index,
            kind,
        });
        if dead {
            unit.kill();
        }
        Ok(())
    }

    /// A Start or an `encodeEndDoWhile` on a nil buffer: inside a block it kills, answering NO;
    /// with none it is unbalanced: the step to record, if any.
    fn nil(&mut self, uploads: &mut Uploads, device: &Device) -> Result<Option<Step>, Error> {
        let unit = self.unit()?;
        if unit.stack.is_empty() {
            return self.unbalanced(uploads, device);
        }
        unit.kill();
        Ok(None)
    }

    /// `encodeStartIf:...` of `test`: the steps to record, a `Resume` first where a range drops.
    fn start_if(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        test: &Test,
    ) -> Result<(Option<Step>, Step), Error> {
        let (resume, dead) = self.resume(uploads, device)?;
        let record = self.step(uploads, device, (None, None), |counter| {
            test.of(PredicationStep::new(counter, PredicationOp::If))
        })?;
        self.open(&record, Kind::If { elsed: false }, dead)?;
        Ok((resume, record))
    }

    /// `encodeStartElse`: the step to record, if any. An Else on a loop, or an If's second,
    /// kills; with no block open it is unbalanced.
    fn start_else(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<Option<Step>, Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return self.unbalanced(uploads, device);
        };
        if !matches!(top.kind, Kind::If { elsed: false }) {
            unit.kill();
            return Ok(None);
        }
        let record = self.step(uploads, device, (None, None), |counter| {
            PredicationStep::new(counter, PredicationOp::Else)
        })?;
        if let Some(top) = self.unit()?.stack.last_mut() {
            top.kind = Kind::If { elsed: true };
        }
        Ok(Some(record))
    }

    /// Close the unit's top block with a step of `op` (inside a killed block, the outermost
    /// block's closes as `EndDead`): Metal's answer, which inside a killed block only the End
    /// that closes it gives YES, the unit running again after it; and the step's record.
    fn close(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        op: PredicationOp,
    ) -> Result<(bool, Step), Error> {
        let unit = self.unit()?;
        let op = match unit.dead && unit.stack.len() == 1 {
            true => PredicationOp::EndDead,
            false => op,
        };
        let record = self.step(uploads, device, (None, None), |counter| {
            PredicationStep::new(counter, op)
        })?;
        let unit = self.unit()?;
        unit.stack.pop();
        let answer = match unit.dead {
            true => {
                let closed = unit.stack.is_empty();
                unit.dead = !closed;
                closed
            }
            false => true,
        };
        Ok((answer, record))
    }

    /// `encodeEndIf`: Metal's answer, and the step to record, if any. An End of a loop kills
    /// and answers NO; with no block open it is unbalanced, answering NO.
    fn end_if(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<(bool, Option<Step>), Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return Ok((false, self.unbalanced(uploads, device)?));
        };
        if !matches!(top.kind, Kind::If { .. }) {
            unit.kill();
            return Ok((false, None));
        }
        let (answer, record) = self.close(uploads, device, PredicationOp::End)?;
        Ok((answer, Some(record)))
    }

    /// The outermost loop's laying begins, in `raw`, after `from` parts of the command buffer.
    pub(super) fn lay_from(&mut self, raw: vk::CommandBuffer, from: usize) {
        self.laying = Some(Laying {
            pieces: vec![Piece::Raw(raw)],
            copied: false,
            from,
        });
    }

    /// Whether the outermost open loop's laying goes on.
    pub(super) fn laying(&self) -> bool {
        self.laying.is_some()
    }

    /// The laying goes on in `raw`, begun to be submitted again.
    pub(super) fn laid_into(&mut self, raw: vk::CommandBuffer) {
        if let Some(laying) = &mut self.laying {
            laying.pieces.push(Piece::Raw(raw));
        }
    }

    /// The encoder left `work` apart inside the laying, before the `VkCommandBuffer` it went on
    /// in, the laying's last piece ([`laid_into`](Self::laid_into)).
    pub(super) fn laid_apart(&mut self, work: &Deferred) {
        if let Some(laying) = &mut self.laying {
            let before = laying.pieces.len().saturating_sub(1);
            laying.pieces.insert(before, Piece::Apart(work.clone()));
        }
    }

    /// A loop inside the laying was laid down as copies: its step past them may halt the unit.
    fn copies_laid(&mut self) {
        if let Some(laying) = &mut self.laying {
            laying.copied = true;
        }
    }

    /// The outermost loop closed: its laying, where a loop in it was laid down as copies, with
    /// the unit's counter, whose halt word its check reads.
    fn end_laying(&mut self) -> Result<Option<(Laying, Upload)>, Error> {
        let Some(laying) = self.laying.take() else {
            return Ok(None);
        };
        if !laying.copied {
            return Ok(None);
        }
        let counter = (self.unit.as_ref())
            .and_then(|unit| unit.counter.clone())
            .ok_or_else(|| internal("a loop's laying without its unit's counter"))?;
        Ok(Some((laying, counter)))
    }

    /// How many loops the unit has open.
    fn loops_open(&self) -> usize {
        self.unit.as_ref().map_or(0, |unit| {
            let loops = |block: &&Block| matches!(block.kind, Kind::Loop(_));
            unit.stack.iter().filter(loops).count()
        })
    }

    /// `encodeStartWhile:...` of `test`, or `encodeStartDoWhile` (`None`): the steps to record, a
    /// `Resume` first where a range drops, then its first copy's start.
    fn start_loop(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        test: Option<&Test>,
    ) -> Result<(Option<Step>, Step), Error> {
        let (resume, dead) = self.resume(uploads, device)?;
        self.unit()?.looped = true;
        self.loops = (self.loops.checked_add(1))
            .ok_or_else(|| internal("a command buffer's loops past 2^32"))?;
        let looping = Looping {
            test: test.cloned(),
            flag: uploads.upload(device, &[], 16)?,
            id: self.loops,
            laid_at: self.laid,
        };
        let record = self.copy_step(uploads, device, &looping, true)?;
        self.open(&record, Kind::Loop(looping), dead)?;
        Ok((resume, record))
    }

    /// The start of a copy of `looping`, its first where `first`: its step's record.
    fn copy_step(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        first: bool,
    ) -> Result<Step, Error> {
        let op = match (&looping.test, first) {
            (Some(_), true) => PredicationOp::WhileFirst,
            (Some(_), false) => PredicationOp::While,
            (None, true) => PredicationOp::DoFirst,
            (None, false) => PredicationOp::Do,
        };
        self.loop_step(uploads, device, looping, (op, looping.test.as_ref()))
    }

    /// The start of a further copy of `looping`: its step's record, its block open.
    fn copy_start(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
    ) -> Result<Step, Error> {
        let record = self.copy_step(uploads, device, looping, false)?;
        let index = self.records;
        self.unit()?.stack.push(Block {
            start: record.record.clone(),
            record: index,
            kind: Kind::Loop(looping.clone()),
        });
        Ok(record)
    }

    /// A step of `op` on a copy of `looping`, testing `test` where `op` tests: its record.
    fn loop_step(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        (op, test): (PredicationOp, Option<&Test>),
    ) -> Result<Step, Error> {
        let flag = place(device, &looping.flag)?;
        let places = (Some(&looping.flag), None);
        self.step(uploads, device, places, |counter| {
            let step = PredicationStep::new(counter, op).in_loop(flag, looping.id);
            match test {
                Some(test) => test.of(step),
                None => step,
            }
        })
    }

    /// The end of a copy of `looping`, a DoWhile's testing `test`: its step's record, its block
    /// closed.
    fn copy_end(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        test: Option<&Test>,
    ) -> Result<Step, Error> {
        let op = match looping.test {
            Some(_) => PredicationOp::End,
            None => PredicationOp::DoEnd,
        };
        let record = self.loop_step(uploads, device, looping, (op, test))?;
        self.unit()?.stack.pop();
        Ok(record)
    }

    /// After the last copy of `looping`, a DoWhile's testing `test`: the step's record that halts
    /// the unit where the loop would run another.
    fn past(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        test: Option<&Test>,
    ) -> Result<Step, Error> {
        let op = match looping.test {
            Some(_) => PredicationOp::WhilePast,
            None => PredicationOp::DoPast,
        };
        let test = looping.test.as_ref().or(test);
        self.loop_step(uploads, device, looping, (op, test))
    }

    /// `encodeEndWhile` (`None`) or `encodeEndDoWhile:...` of `condition`: Metal's answer, the
    /// step to record, if any, and the loop to lay further copies of where it is live. An End of
    /// another block kills and answers NO; with no block open it is unbalanced, answering NO.
    /// Only an End that closes a live DoWhile reads its condition (refused as [`Test::new`]
    /// refuses).
    fn end_loop(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        condition: Option<Condition<'_>>,
    ) -> Result<LoopEnd, Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return Ok(LoopEnd {
                answer: false,
                record: self.unbalanced(uploads, device)?,
                closed: false,
                copies: None,
            });
        };
        let looping = match &top.kind {
            Kind::Loop(looping) if looping.test.is_some() == condition.is_none() => looping.clone(),
            _ => {
                unit.kill();
                return Ok(LoopEnd {
                    answer: false,
                    record: None,
                    closed: false,
                    copies: None,
                });
            }
        };
        if unit.dead {
            let (answer, record) = self.close(uploads, device, PredicationOp::End)?;
            return Ok(LoopEnd {
                answer,
                record: Some(record),
                closed: true,
                copies: None,
            });
        }
        // Only a live DoWhile's End reads its word.
        let test = condition.map(|c| c.test(device, uploads)).transpose()?;
        let record = self.copy_end(uploads, device, &looping, test.as_ref())?;
        Ok(LoopEnd {
            answer: true,
            record: Some(record),
            closed: true,
            copies: Some((looping, test)),
        })
    }

    /// The open encoder ends, with `worked` its dispatches since its last record and `open` the
    /// bodies of the loops it leaves open, which its unit's next encoder carries on: where its
    /// unit has a live block open, the `UnitEnd` step to record.
    fn encoder_ends(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        (worked, open): (bool, Vec<Vec<(EncoderState, Event)>>),
    ) -> Result<Option<Step>, Error> {
        let unit = self.unit()?;
        unit.note_work(worked);
        unit.carried = open;
        if !unit.live_open() {
            return Ok(None);
        }
        let check = match &unit.check {
            Some(check) => check.clone(),
            None => uploads.upload(device, &[], 16)?,
        };
        let word = place(device, &check)?;
        let record = self.step(uploads, device, (None, Some(&check)), |counter| {
            PredicationStep::new(counter, PredicationOp::UnitEnd).checking(word)
        })?;
        self.unit()?.check = Some(check);
        Ok(Some(record))
    }

    /// Whether the open encoder's unit records anything at the encoder's end: it has a block
    /// open, or loops whose bodies the encoder carries on.
    pub(super) fn ends_with_open_blocks(&self) -> bool {
        self.unit
            .as_ref()
            .is_some_and(|unit| !unit.stack.is_empty())
    }

    /// The gates of a dispatch the open encoder records now, one per command of `counts` (its
    /// group counts), where its unit gates its dispatches (`Unit::gated`); none where not, and it
    /// runs as recorded. In the bound form they lie in the last step's record, or where it has no
    /// room for them in a further step's, which leaves the counter as it is (`Hold`): that step,
    /// to record before the dispatch.
    pub(super) fn gates(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        counts: &[[u32; 3]],
    ) -> Result<(Option<Step>, Vec<Upload>), Error> {
        let Some(unit) = &self.unit else {
            return Ok((None, Vec::new()));
        };
        if !unit.gated() {
            return Ok((None, Vec::new()));
        }
        let mut run = unit
            .run
            .ok_or_else(|| internal("a gated predication unit without its step"))?;
        let mut further = None;
        if let Some(capacity) = self.runs[run].capacity {
            if (self.runs[run].gates.len() + counts.len()) as u64 > capacity {
                let holding = BOUND_GATES.max(counts.len() as u64);
                let hold = |counter| PredicationStep::new(counter, PredicationOp::Hold);
                further =
                    Some(self.step_holding(uploads, device, ((None, None), holding), hold)?);
                run = self.runs.len() - 1;
            }
        }
        let record = self.runs[run].record.clone();
        let bound = self.runs[run].capacity.is_some();
        let first = self.runs[run].gates.len() as u64;
        self.laid += counts.len() as u64;
        let mut gates = Vec::with_capacity(counts.len());
        let mut places = Vec::with_capacity(counts.len());
        for (i, &count) in (0..).zip(counts) {
            let bytes = PredicationStep::gate(count);
            let gate = match bound {
                true => {
                    let at = PredicationStep::BOUND_GATES_OFFSET
                        + (first + i) * PredicationStep::GATE_BYTES;
                    let gate = record.part(at, PredicationStep::GATE_BYTES);
                    gate.write(&bytes);
                    places.push(at);
                    gate
                }
                false => {
                    let gate = uploads.upload(device, &bytes, PredicationStep::GATE_BYTES)?;
                    places.push(addressed(&gate)?);
                    gate
                }
            };
            gates.push(gate);
        }
        let run = &mut self.runs[run];
        run.gates.extend(places);
        if bound {
            let count = run.gates.len() as u32;
            record.write_at(PredicationStep::BOUND_COUNT_OFFSET, &count.to_le_bytes());
        }
        Ok((further, gates))
    }

    /// The check of a command the open encoder leaves to a host step, where its unit gates its
    /// dispatches.
    pub(super) fn host_gate(&self) -> Option<HostGate> {
        let unit = self.unit.as_ref()?;
        if !unit.gated() {
            return None;
        }
        unit.counter.clone().map(HostGate)
    }
}

/// The device address of `upload`, which every predication region has in the addressed form.
fn addressed(upload: &Upload) -> Result<u64, Error> {
    upload
        .device_address()
        .ok_or_else(|| internal("a predication region without a device address"))
}

/// The check a host step makes of a command inside a block: its unit's skip counter and halt
/// word.
#[derive(Clone)]
pub(super) struct HostGate(Upload);

impl HostGate {
    /// Whether the command runs nothing: the counter the steps before it left is not 0, or a loop
    /// halted the unit. Read at the command's host step, once the work before it completed.
    pub(super) fn skips(&self) -> bool {
        self.0.read::<8>() != [0; 8]
    }
}

/// Record the step of `record` into `core`, reading the word of `test`, where it tests one: a
/// barrier from everything before, the step, and a barrier from its writes to the commands after
/// it.
fn record_step(
    core: &mut EncoderCore,
    kernel: OwnedKernel,
    step: &Step,
    test: Option<&Test>,
) -> Result<(), Error> {
    // The addressed form pushes the record's address; the bound one binds its places.
    let (push, binds) = match &step.bound {
        None => {
            let address = addressed(&step.record)?;
            (Some(PredicationStepModule::push_constants(address)), None)
        }
        Some(places) => {
            let region = |upload: &Upload| (upload.raw(), upload.offset(), upload.len());
            let record = region(&step.record);
            let or_record = |upload: &Option<Upload>| upload.as_ref().map_or(record, region);
            let set = OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: vec![
                    record,
                    region(&places.counter),
                    test.and_then(|test| test.binding).unwrap_or(record),
                    or_record(&places.flag),
                    or_record(&places.check),
                ],
            };
            (None, Some(OwnedBinds::new(core.device(), &kernel, &[set])?))
        }
    };
    let uses: Vec<Use<ResourceKey>> = (test.and_then(|test| test.word.as_ref()).into_iter())
        .map(|(buffer, at)| {
            let (identity, range) = buffer.tracked(*at..*at + 4);
            let read = Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
        })
        .collect();
    core.barrier(Barrier::everything());
    core.command(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        &uses,
        |raw, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass, and only
            // this encoder records it; the kernel is the device's, and the record lies in the
            // command buffer's arena, which lives until it completes.
            unsafe {
                raw.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                if let Some(push) = &push {
                    raw.cmd_push_constants(
                        command_buffer,
                        kernel.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push,
                    );
                }
                if let Some(binds) = &binds {
                    binds.bind(raw, command_buffer, &kernel, 0);
                }
                raw.cmd_dispatch(command_buffer, 1, 1, 1);
                Barrier::between_dispatches().record(raw, command_buffer);
            }
        },
    );
    if let Some(binds) = binds {
        core.retain(binds);
    }
    if let Some((buffer, _)) = test.and_then(|test| test.word.as_ref()) {
        core.retain(buffer.clone());
    }
    Ok(())
}

/// Record `step`, a step [`Predication::gates`] made for a dispatch's gates, into `core` before
/// the dispatch.
pub(super) fn record_further(core: &mut EncoderCore, step: &Step) -> Result<(), Error> {
    let kernel = core.device().owned_kernel(OwnedModule::PredicationStep)?;
    record_step(core, kernel, step, None)
}

/// Record `steps` (their records and tests, each where there is one) into `core` in order.
fn record_steps(
    core: &mut EncoderCore,
    kernel: OwnedKernel,
    steps: [(Option<&Step>, Option<&Test>); 2],
) -> Result<(), Error> {
    for (record, test) in steps {
        if let Some(record) = record {
            record_step(core, kernel, record, test)?;
        }
    }
    Ok(())
}

/// A command a compute encoder encoded inside a loop, as its copies replay it: the public call's
/// arguments.
#[derive(Clone)]
pub(super) enum Event {
    Threadgroups(Size, Size),
    Threads(Size, Size),
    ThreadgroupsIndirect(Buffer, u64, Size),
    ThreadsIndirect(Buffer, u64),
    /// A memory barrier, of a scope or of resources: one global barrier.
    Barrier,
    /// The end of the loop's encoder and the start of the next of its unit: one global barrier,
    /// as Metal orders an encoder's commands after the ones before, and what the ended encoder
    /// made resident, which its commands ran with.
    Encoder(Residency),
    Execute(IndirectCommandBuffer, Range<u64>),
    ExecuteIndirect(IndirectCommandBuffer, Buffer, u64),
    StartIf(Option<Buffer>, u64, u64, u32),
    Else,
    EndIf,
    StartWhile(Option<Buffer>, u64, u64, u32),
    StartDoWhile,
    EndWhile,
    EndDoWhile(Option<Buffer>, u64, u64, u32),
}

/// A compute encoder's captures of the bodies of the loops open in it (opened in it, or carried
/// from its unit's encoder before), the innermost last: each command with the encoder's state it
/// ran with; and whether it dispatched since its last record.
#[derive(Default)]
pub(super) struct Captures {
    open: Vec<Vec<(EncoderState, Event)>>,
    /// The captures below it take no command: those a loop's further copies replay.
    floor: usize,
    /// It encoded a dispatch or an execution since its last record (`Unit::note_work`).
    worked: bool,
    /// It is laying a loop's copies down, whose replayed calls are not the command buffer's
    /// records (`Predication::records`).
    replaying: bool,
}

impl Captures {
    /// The captures of an encoder that carries on `open`, the bodies of the loops its unit's
    /// encoder before left open.
    pub(super) fn carried(open: Vec<Vec<(EncoderState, Event)>>) -> Captures {
        Captures {
            open,
            ..Captures::default()
        }
    }
}

impl ComputeCommandEncoder {
    /// The predication step kernel, in the device's form.
    fn predication_kernel(&mut self) -> Result<OwnedKernel, Error> {
        self.core()
            .device()
            .owned_kernel(OwnedModule::PredicationStep)
    }

    /// Run `record` on the command buffer's predication with its arena, under its lock, the
    /// encoder's dispatches since its last record noted first. Where it dropped what the unit's
    /// encoders recorded before it, dispatches that ran ungated among it, that is left out of the
    /// command buffer's work (`Predication::take_left_out`), before what it returns is recorded.
    fn predication<T>(
        &mut self,
        record: impl FnOnce(&mut Predication, &mut Uploads, &Device) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let worked = self.captures().worked;
        self.captures().worked = false;
        let core = self.core();
        let device = core.device().clone();
        let (recorded, left_out) = {
            let mut state = core.command_buffer().state();
            let state = &mut *state;
            state.predicated = true;
            let (predication, uploads) = state.predication_and_uploads();
            if let Some(unit) = &mut predication.unit {
                unit.note_work(worked);
            }
            let recorded = record(predication, uploads, &device)?;
            let left_out = state.predication_mut().take_left_out();
            if let Some(first) = left_out {
                state.leave_out_ended(first);
            }
            (recorded, left_out.is_some())
        };
        if left_out {
            self.core().leave_out_recorded()?;
        }
        Ok(recorded)
    }

    /// The encoder dispatched, or executed an indirect command buffer's commands: work Metal
    /// emits for its unit where no misuse killed the open block (the module docs).
    pub(super) fn worked(&mut self) {
        self.captures().worked = true;
    }

    /// A command-predication call of the encoder returned `Ok`: it is a record of its command
    /// buffer's, unless a loop's copy replayed it.
    fn counted<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_ok() && !self.captures().replaying {
            self.core()
                .command_buffer()
                .state()
                .predication_mut()
                .counted();
        }
        result
    }

    /// The encoder updated a fence: its unit of predication ends with it (the module docs).
    pub(super) fn fence_updated(&mut self) {
        let mut state = self.core().command_buffer().state();
        state.predication_mut().fence_updated();
    }

    /// As the encoder ends: its unit takes its dispatches since its last record, and the bodies
    /// of the loops it leaves open, each closed by the encoder's boundary (`Event::Encoder`),
    /// which its unit's next encoder carries on; where its unit has a live block open, it records
    /// the `UnitEnd` step (the module docs). Nothing where the command buffer used no
    /// predication.
    pub(super) fn end_predication(&mut self) -> Result<(), Error> {
        let worked = self.captures().worked;
        self.captures().worked = false;
        let open_blocks = {
            let mut state = self.core().command_buffer().state();
            if !state.predicated {
                // No record yet: only its dispatches, which a later unbalanced record of its unit
                // would find ran ungated.
                if let Some(unit) = &mut state.predication_mut().unit {
                    unit.note_work(worked);
                }
                return Ok(());
            }
            state.predication_mut().ends_with_open_blocks()
        };
        let mut open = std::mem::take(&mut self.captures().open);
        if !open.is_empty() {
            let boundary = (self.encoder_state(), Event::Encoder(self.residency_now()));
            for body in &mut open {
                body.push(boundary.clone());
            }
        }
        let kernel = match open_blocks {
            true => Some(self.predication_kernel()?),
            false => None,
        };
        let record = self.predication(|predication, uploads, device| {
            predication.encoder_ends(uploads, device, (worked, open))
        })?;
        match (record, kernel) {
            (Some(record), Some(kernel)) => record_step(self.core(), kernel, &record, None),
            (None, _) => Ok(()),
            (Some(_), None) => Err(internal("a predication unit end without its kernel")),
        }
    }

    /// The condition of a record of `call` on `buffer` (`None`: a nil buffer), with the step
    /// kernel. Refused: a buffer of another device (`Invalid`).
    fn condition<'a>(
        &mut self,
        call: &'static str,
        buffer: Option<&'a Buffer>,
        offset: u64,
        (comparison, reference): (u64, u32),
    ) -> Result<(OwnedKernel, Option<Condition<'a>>), Error> {
        if let Some(buffer) = buffer {
            self.core().check_device(call, buffer.device())?;
        }
        let kernel = self.predication_kernel()?;
        let condition = buffer.map(|buffer| Condition {
            buffer,
            offset,
            comparison,
            reference,
        });
        Ok((kernel, condition))
    }

    /// Note `event`, encoded now, in the captures of the open loops' bodies above the floor.
    pub(super) fn capture(&mut self, event: impl FnOnce() -> Event) {
        let floor = self.captures().floor;
        if self.captures().open.len() <= floor {
            return;
        }
        let captured = (self.encoder_state(), event());
        for body in &mut self.captures().open[floor..] {
            body.push(captured.clone());
        }
    }

    /// Encode `event` again, as its public call.
    fn replay(&mut self, event: &Event) -> Result<(), Error> {
        match event {
            Event::Threadgroups(groups, threads) => self.dispatch_threadgroups(*groups, *threads),
            Event::Threads(threads, per_group) => self.dispatch_threads(*threads, *per_group),
            Event::ThreadgroupsIndirect(buffer, offset, threads) => {
                self.dispatch_threadgroups_indirect(buffer, *offset, *threads)
            }
            Event::ThreadsIndirect(buffer, offset) => {
                self.dispatch_threads_indirect(buffer, *offset)
            }
            Event::Barrier => {
                self.memory_barrier_with_scope(BarrierScope::BUFFERS);
                Ok(())
            }
            // What the ended encoder made resident joined this one's before the copies.
            Event::Encoder(_) => {
                self.core().barrier(Barrier::everything());
                Ok(())
            }
            Event::Execute(icb, range) => self.execute_commands_in_buffer(icb, range.clone()),
            Event::ExecuteIndirect(icb, buffer, offset) => {
                self.execute_commands_in_buffer_indirect(icb, buffer, *offset)
            }
            Event::StartIf(buffer, offset, comparison, reference) => {
                self.encode_start_if(buffer.as_ref(), *offset, *comparison, *reference)
            }
            Event::Else => self.encode_start_else(),
            Event::EndIf => self.encode_end_if().map(drop),
            Event::StartWhile(buffer, offset, comparison, reference) => {
                self.encode_start_while(buffer.as_ref(), *offset, *comparison, *reference)
            }
            Event::StartDoWhile => self.encode_start_do_while(),
            Event::EndWhile => self.encode_end_while().map(drop),
            Event::EndDoWhile(buffer, offset, comparison, reference) => self
                .encode_end_do_while(buffer.as_ref(), *offset, *comparison, *reference)
                .map(drop),
        }
    }

    /// Open a loop of `test` (`None`: a DoWhile) by the call of `event`, recording its first
    /// copy's start. Its body is captured from here.
    fn open_loop(
        &mut self,
        kernel: OwnedKernel,
        test: Option<&Test>,
        event: impl FnOnce() -> Event,
    ) -> Result<(), Error> {
        // The outermost loop's laying goes in command buffers of its own, from here to its check
        // (the module docs).
        if self.predication(|predication, _, _| Ok(predication.loops_open() == 0))? {
            let raw = self.core().cut_reusable()?;
            let from = self.core().command_buffer().state().parts();
            self.predication(|predication, _, _| {
                predication.lay_from(raw, from);
                Ok(())
            })?;
        }
        let (resume, record) = self.predication(|predication, uploads, device| {
            predication.start_loop(uploads, device, test)
        })?;
        record_steps(
            self.core(),
            kernel,
            [(resume.as_ref(), None), (Some(&record), test)],
        )?;
        self.capture(event);
        self.captures().open.push(Vec::new());
        Ok(())
    }

    /// Close the open loop by `encodeEndWhile` or `encodeEndDoWhile:...` of `condition`: its
    /// first copy's end, then, where it is live, its further copies (`lay_copies`). Metal's
    /// answer.
    fn close_loop(
        &mut self,
        kernel: OwnedKernel,
        condition: Option<Condition<'_>>,
        event: impl FnOnce() -> Event,
    ) -> Result<bool, Error> {
        let end = self.predication(|predication, uploads, device| {
            predication.end_loop(uploads, device, condition)
        })?;
        let Some((looping, test)) = end.copies else {
            // A killed loop's End, a misused or an unbalanced one: no copies.
            if let Some(record) = &end.record {
                record_step(self.core(), kernel, record, None)?;
            }
            if end.closed {
                self.captures().open.pop();
            }
            self.capture(event);
            self.end_laying()?;
            return Ok(end.answer);
        };
        let record = end
            .record
            .ok_or_else(|| internal("a live loop's End without its step"))?;
        record_step(self.core(), kernel, &record, test.as_ref())?;
        let body = (self.captures().open.pop())
            .ok_or_else(|| internal("a live loop's End without its body"))?;
        self.capture(event);
        self.lay_copies(kernel, (&looping, test.as_ref()), &body)?;
        self.predication(|predication, _, _| {
            predication.copies_laid();
            Ok(())
        })?;
        self.end_laying()?;
        Ok(end.answer)
    }

    /// A loop closed: where it was the outermost one and a loop in its laying was laid down as
    /// copies, the laying ends with its check (the module docs).
    fn end_laying(&mut self) -> Result<(), Error> {
        let ended = self.predication(|predication, _, _| match predication.loops_open() {
            0 => predication.end_laying(),
            _ => Ok(None),
        })?;
        let Some((laying, counter)) = ended else {
            return Ok(());
        };
        let (pieces, from) = laying.into_parts();
        let check = LoopCheck::new(counter, pieces, from);
        self.core().end_laying(Deferred::Loop(Arc::new(check)))
    }

    /// Lay down the further copies of `looping` (with a DoWhile's `test`), each its start, its
    /// `body` replayed with the encoder's state each command ran with, and its end; then the step
    /// that finds whether it would run past them. The encoder's state is then as before.
    fn lay_copies(
        &mut self,
        kernel: OwnedKernel,
        (looping, test): (&Looping, Option<&Test>),
        body: &[(EncoderState, Event)],
    ) -> Result<(), Error> {
        let copies = self.predication(|predication, _, _| {
            let first = predication.laid - looping.laid_at;
            Ok(copies_of(predication.loops_open(), first))
        })?;
        // The commands of the unit's earlier encoders replay here with what those made resident.
        for (_, event) in body {
            if let Event::Encoder(residency) = event {
                self.carry_residency(residency);
            }
        }
        let state = self.encoder_state();
        let (floor, replaying) = (self.captures().floor, self.captures().replaying);
        self.captures().floor = self.captures().open.len();
        self.captures().replaying = true;
        let laid = self.copies(kernel, (looping, test), body, copies);
        self.captures().floor = floor;
        self.captures().replaying = replaying;
        self.restore_state(state);
        laid
    }

    /// [`lay_copies`](Self::lay_copies)'s copies 2 to `copies`, and its past step.
    fn copies(
        &mut self,
        kernel: OwnedKernel,
        (looping, test): (&Looping, Option<&Test>),
        body: &[(EncoderState, Event)],
        copies: u64,
    ) -> Result<(), Error> {
        for _ in 1..copies {
            let record = self.predication(|predication, uploads, device| {
                predication.copy_start(uploads, device, looping)
            })?;
            record_step(self.core(), kernel, &record, looping.test.as_ref())?;
            for (state, event) in body {
                self.restore_state(state.clone());
                self.replay(event)?;
            }
            let record = self.predication(|predication, uploads, device| {
                predication.copy_end(uploads, device, looping, test)
            })?;
            record_step(self.core(), kernel, &record, test)?;
        }
        let record = self.predication(|predication, uploads, device| {
            predication.past(uploads, device, looping, test)
        })?;
        record_step(self.core(), kernel, &record, looping.test.as_ref().or(test))
    }

    /// `encodeStartIf:offset:comparison:referenceValue:` (SPI): the commands encoded after it, up
    /// to its `encodeStartElse` or `encodeEndIf`, run where the `u32` at `offset` (rounded down
    /// to a multiple of 4) of `buffer` compares so with `reference_value` when the GPU reaches the
    /// call (`comparison`: Metal's codes, `metal2vulkan::bind::PredicationCompare`), and those
    /// after its Else where not (the module docs). A nil `buffer` inside a block kills the
    /// outermost open block, as Metal's does; with none open it is an unbalanced record, which
    /// drops its range of the unit's work, as Metal's does.
    ///
    /// A word past the buffer's end reads as Metal reads it: the heap's bytes past a buffer
    /// placed in a heap, zero within the 256 bytes Metal rounds a lone buffer up to; elsewhere
    /// Metal reads memory unrelated to the buffer, which the runtime reads as zero, announced
    /// (`Degradation::PredicationWordPastBuffer`).
    ///
    /// Refused, recording nothing: a buffer of another device (`Invalid`).
    pub fn encode_start_if(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::StartIf(buffer.cloned(), offset, comparison, reference_value),
            |this| {
                let result =
                    this.encode_start_if_uncounted(buffer, offset, comparison, reference_value);
                this.counted(result)
            },
        )
    }

    /// [`encode_start_if`](Self::encode_start_if), not yet counted among the command buffer's records.
    fn encode_start_if_uncounted(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        let arguments = (comparison, reference_value);
        let (kernel, condition) = self.condition("encode_start_if", buffer, offset, arguments)?;
        let test = self.predication(|_, uploads, device| {
            condition.map(|c| c.test(device, uploads)).transpose()
        })?;
        match &test {
            Some(test) => {
                let (resume, record) = self.predication(|predication, uploads, device| {
                    predication.start_if(uploads, device, test)
                })?;
                record_steps(
                    self.core(),
                    kernel,
                    [(resume.as_ref(), None), (Some(&record), Some(test))],
                )?;
            }
            None => {
                let record = self
                    .predication(|predication, uploads, device| predication.nil(uploads, device))?;
                record_steps(self.core(), kernel, [(record.as_ref(), None), (None, None)])?;
            }
        }
        self.capture(|| Event::StartIf(buffer.cloned(), offset, comparison, reference_value));
        Ok(())
    }

    /// `encodeStartElse` (SPI): the commands encoded after it, up to the open If's
    /// `encodeEndIf`, run where its condition failed (the module docs). A second Else in one If, or
    /// an Else on a loop, kills the outermost open block, as Metal's does; with no block open it is
    /// an unbalanced record, as Metal's is.
    pub fn encode_start_else(&mut self) -> Result<(), Error> {
        self.logged(
            |_| ComputeCall::StartElse,
            |this| {
                let result = this.encode_start_else_uncounted();
                this.counted(result)
            },
        )
    }

    /// [`encode_start_else`](Self::encode_start_else), not yet counted among the command buffer's records.
    fn encode_start_else_uncounted(&mut self) -> Result<(), Error> {
        let kernel = self.predication_kernel()?;
        let record = self
            .predication(|predication, uploads, device| predication.start_else(uploads, device))?;
        if let Some(record) = record {
            record_step(self.core(), kernel, &record, None)?;
        }
        self.capture(|| Event::Else);
        Ok(())
    }

    /// `encodeEndIf` (SPI): closes the open If, answering Metal's `YES`; on a loop it kills the
    /// outermost open block and answers `NO`; inside a block a misused record killed, `NO` but
    /// for the End that closes that block; with no block open it is an unbalanced record,
    /// answering `NO` (the module docs).
    pub fn encode_end_if(&mut self) -> Result<bool, Error> {
        let _for = self.core().notices_for();
        let at = self.log(|_| ComputeCall::EndIf(false));
        let result = self.encode_end_if_uncounted();
        let result = self.counted(result);
        self.answered(at, &result, ComputeCall::EndIf);
        result
    }

    /// [`encode_end_if`](Self::encode_end_if), not yet counted among the command buffer's records.
    fn encode_end_if_uncounted(&mut self) -> Result<bool, Error> {
        let kernel = self.predication_kernel()?;
        let (answer, record) =
            self.predication(|predication, uploads, device| predication.end_if(uploads, device))?;
        if let Some(record) = record {
            record_step(self.core(), kernel, &record, None)?;
        }
        self.capture(|| Event::EndIf);
        Ok(answer)
    }

    /// `encodeStartWhile:offset:comparison:referenceValue:` (SPI): the commands encoded after it,
    /// up to its `encodeEndWhile`, run again and again while the condition (read as
    /// [`encode_start_if`](Self::encode_start_if) reads it, before each run) holds, each run with
    /// the encoder's state each command was encoded with (the module docs), the commands of its
    /// unit's later encoders included. A nil `buffer` is taken as
    /// [`encode_start_if`](Self::encode_start_if)'s.
    ///
    /// The runtime lays the loop down as copies of its commands (64, or 16 inside another loop,
    /// fewer where the copies would lay down more than 16384 steps and gated dispatches, and at
    /// least one); where the loop runs past them, the runtime submits its copies again from where
    /// it stopped, each time after a round trip to the host, until it ends, as Metal runs it on: a
    /// loop without bound runs until the process ends, or its device is lost.
    ///
    /// Refused, recording nothing: as [`encode_start_if`](Self::encode_start_if) refuses.
    pub fn encode_start_while(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        let _for = self.core().notices_for();
        self.stop_logging();
        let result = self.encode_start_while_uncounted(buffer, offset, comparison, reference_value);
        self.counted(result)
    }

    /// [`encode_start_while`](Self::encode_start_while), not yet counted among the command buffer's records.
    fn encode_start_while_uncounted(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        let arguments = (comparison, reference_value);
        let (kernel, condition) =
            self.condition("encode_start_while", buffer, offset, arguments)?;
        let event = || Event::StartWhile(buffer.cloned(), offset, comparison, reference_value);
        let test = self.predication(|_, uploads, device| {
            condition.map(|c| c.test(device, uploads)).transpose()
        })?;
        match &test {
            Some(test) => self.open_loop(kernel, Some(test), event),
            None => {
                let record = self
                    .predication(|predication, uploads, device| predication.nil(uploads, device))?;
                record_steps(self.core(), kernel, [(record.as_ref(), None), (None, None)])?;
                self.capture(event);
                Ok(())
            }
        }
    }

    /// `encodeStartDoWhile` (SPI): the commands encoded after it, up to its `encodeEndDoWhile`,
    /// run, and again while that End's condition holds after each run, as
    /// [`encode_start_while`](Self::encode_start_while)'s run, laid down as there.
    pub fn encode_start_do_while(&mut self) -> Result<(), Error> {
        let _for = self.core().notices_for();
        self.stop_logging();
        let result = self.encode_start_do_while_uncounted();
        self.counted(result)
    }

    /// [`encode_start_do_while`](Self::encode_start_do_while), not yet counted among the command buffer's records.
    fn encode_start_do_while_uncounted(&mut self) -> Result<(), Error> {
        let kernel = self.predication_kernel()?;
        self.open_loop(kernel, None, || Event::StartDoWhile)
    }

    /// `encodeEndWhile` (SPI): closes the open While loop, answering Metal's `YES`; on another
    /// block it kills the outermost open block and answers `NO`; with no block open it is an
    /// unbalanced record, answering `NO` (the module docs).
    pub fn encode_end_while(&mut self) -> Result<bool, Error> {
        let _for = self.core().notices_for();
        self.stop_logging();
        let result = self.encode_end_while_uncounted();
        self.counted(result)
    }

    /// [`encode_end_while`](Self::encode_end_while), not yet counted among the command buffer's records.
    fn encode_end_while_uncounted(&mut self) -> Result<bool, Error> {
        let kernel = self.predication_kernel()?;
        self.close_loop(kernel, None, || Event::EndWhile)
    }

    /// `encodeEndDoWhile:offset:comparison:referenceValue:` (SPI): closes the open DoWhile loop,
    /// whose body runs again where its condition (read as
    /// [`encode_start_if`](Self::encode_start_if) reads it, after each run) holds, answering
    /// Metal's `YES`; on another block, or for a nil `buffer` inside a block, it kills the
    /// outermost open block and answers `NO`; with no block open it is an unbalanced record,
    /// answering `NO` (the module docs).
    ///
    /// Only an End that closes a live DoWhile reads its word, as only there Metal reads it.
    ///
    /// Refused, recording nothing: as [`encode_start_if`](Self::encode_start_if) refuses.
    pub fn encode_end_do_while(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<bool, Error> {
        let _for = self.core().notices_for();
        self.stop_logging();
        let result =
            self.encode_end_do_while_uncounted(buffer, offset, comparison, reference_value);
        self.counted(result)
    }

    /// [`encode_end_do_while`](Self::encode_end_do_while), not yet counted among the command buffer's records.
    fn encode_end_do_while_uncounted(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<bool, Error> {
        let arguments = (comparison, reference_value);
        let (kernel, condition) =
            self.condition("encode_end_do_while", buffer, offset, arguments)?;
        let event = || Event::EndDoWhile(buffer.cloned(), offset, comparison, reference_value);
        match condition {
            Some(condition) => self.close_loop(kernel, Some(condition), event),
            None => {
                let record = self
                    .predication(|predication, uploads, device| predication.nil(uploads, device))?;
                record_steps(self.core(), kernel, [(record.as_ref(), None), (None, None)])?;
                self.capture(event);
                Ok(false)
            }
        }
    }
}
