//! Visible-function tables loaded from memory.
//!
//! Devirtualization (`air::canon::devirt`) carries a `visible_function_table` loaded from an
//! argument buffer as its 8-byte resource id (`gpuResourceID`) and reads its words through
//! [`FunctionTable::Word`]: word `w` of the table whose resource id is `id`, in
//! `pipeline::table`'s layout (1 for a table, its size, then its entries).
//!
//! - **The directory.** The stage's function-table directory (`Synthetic(FunctionTableDirectory)`)
//!   is the handle table's layout (`memory::handles`) over the tables the pipeline state made:
//!   16-byte entries `{resource_id: u64, element: u32, flags: u32}` sorted by id after the null
//!   entry, searched by the module's one search of it (`handles::Lookup::Element`). Metal reads a
//!   table through an argument buffer without `useResource` (measured on an M3 Max and an M4), so
//!   the directory lists every live table of the state, resident or not.
//! - **Containment.** A word past a table's words (an entry at or past its size) is undefined in
//!   Metal; both forms below check every read and yield 0 outside the table (a null entry),
//!   whatever the device's robustness. A null id and every id the directory misses read a null
//!   table of size 0 (Metal's answer for a null id; an id of a released table, or of another
//!   state's, is undefined there).
//! - **The snapshot**. The read takes the tables' words from the stage's function-table snapshot
//!   (`Synthetic(FunctionTableSnapshot)`): one storage buffer the binder writes per command as the
//!   command buffer commits
//!   (`bind::FunctionTableSnapshot`), word 0 the null table's record (a word count of 0) and then
//!   each listed table's record, its word count and its words. The directory gives each listed
//!   id its record's word offset `e` in place of an element, so a miss and the null id read the
//!   null table. Word `w` is checked against the record's count, always: `snapshot[e + 1 + w]`
//!   below it and 0 at or past it (a null entry), so no read leaves the record. It needs no
//!   feature past the directory's own storage buffer, on every device.
//! - **Addressed**. Under `LoweringInputs::addressed_function_tables`, the variant a shape whose
//!   listed tables may pass `maxStorageBufferRange` asks for, the directory's entry of an id holds
//!   the device address of the table's words in its element and flags words (low, then high), and
//!   the search yields the entry's index (`handles::Lookup::Entry`; the null entry, address 0, for
//!   the null id and every miss). Word `w` of a live address is read at that address, checked
//!   against the table's size word (word 1): a header word, or an entry below the size, and 0 past
//!   it, so no read leaves the table's words, whatever their length. The null entry (address 0)
//!   reads 0, its loads taken at the directory's last entry's address (a listed table's) and
//!   dropped, so no load takes the null address. It declares no snapshot, and needs device
//!   addresses.

use std::borrow::Cow;

use crate::air::intrinsics::{AirIntrinsic, FunctionTable};
use crate::air::ir::{Callee, Inst, InstKind, ValueId};
use crate::device::profile::BufferAccess;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::arith::legal;
use crate::lower::emit::EmitCx;
use crate::lower::select::{DescriptorCounts, Plan, SelectCx};
use crate::memory::handles::{self, EntryField};
use crate::memory::leaf::{k32, WordMem};
use crate::memory::views::{self, View};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::FuncId;
use crate::spirv::ir::{BinOp, ConvOp, Type as SType};

use super::handle::HandleIds;
use super::psb::{k64, PsbMem};

/// Where a function's table reads find the tables' words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableWords {
    /// The stage's function-table snapshot, at the word offset the directory gives an id.
    Snapshot,
    /// The device address the directory's entry of an id holds, under
    /// `LoweringInputs::addressed_function_tables`.
    Addressed,
}

/// How a function reads tables by their resource ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableReads {
    /// How the directory's ids are held: one `u64`, or two words on a profile without
    /// `shaderInt64`.
    pub ids: HandleIds,
    /// Where the words are.
    pub words: TableWords,
}

/// Word `word` of the table whose resource id is `id` ([`FunctionTable::Word`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableWord {
    /// The `i64` resource id.
    pub id: ValueId,
    /// The `i32` word number.
    pub word: ValueId,
    /// How the id is held and where the words are: a snapshot read is checked against its
    /// record's count, an addressed read against its table's size word.
    pub reads: TableReads,
}

impl TableWord {
    /// The census name.
    pub fn name(&self) -> Cow<'static, str> {
        match self.reads.words {
            TableWords::Snapshot => "access.function_table.word.snapshot".into(),
            TableWords::Addressed => "access.function_table.word.addressed".into(),
        }
    }

    /// An upper estimate of the SPIR-V words one read emits: the id's two words and the search's
    /// call, then a snapshot read's two loads, its count's compare, two adds and two selects; or an
    /// addressed read's four entry fields (its entry's and the last entry's), the table's
    /// length, the addresses' widening, tests and select, the branch around the reads, the size's
    /// and the word's loads at their address, and the checks' compares and selects. The search is
    /// the module's routine, counted once ([`routine_words`]).
    pub fn words(&self) -> u32 {
        let read = 64;
        match self.reads.words {
            TableWords::Snapshot => read + 40,
            TableWords::Addressed => read + 4 * handles::ENTRY_FIELD_WORDS + 160,
        }
    }
}

/// The stage's function-table directory with the module's search of it, and where the tables'
/// words are.
#[derive(Clone, Copy, Debug)]
pub struct FunctionTables {
    /// `fn(lo: u32, hi: u32) -> u32`, the snapshot offset (or the directory entry of an addressed
    /// read) of a table's resource id ([`handles::define_lookup`]).
    pub lookup: FuncId,
    /// The tables' words.
    pub words: DeclaredWords,
}

/// The declared home of the tables' words.
#[derive(Clone, Copy, Debug)]
pub enum DeclaredWords {
    /// The stage's function-table snapshot, a word view.
    Snapshot(View),
    /// The directory itself, whose entries hold the tables' device addresses.
    Addressed(handles::Table),
}

/// The binding symbol of the function-table directory of `stage`.
pub fn directory_symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::FunctionTableDirectory),
        index: 0,
    }
}

/// The binding symbol of the function-table snapshot of `stage`.
pub fn snapshot_symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::FunctionTableSnapshot),
        index: 0,
    }
}

/// Select the read of `inst` when it is a [`FunctionTable::Word`] call, `None` otherwise. The
/// first plans the stage's directory and snapshot (two storage descriptors of the stage), or under
/// `LoweringInputs::addressed_function_tables` its directory alone, and states
/// `RuntimeFact::FunctionTableRange`.
pub fn select(cx: &mut SelectCx<'_>, inst: &Inst) -> Result<Option<TableWord>, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Air {
        op: AirIntrinsic::FunctionTable(FunctionTable::Word),
        ..
    } = call.callee
    else {
        return Ok(None);
    };
    let [id, word] = call.args[..] else {
        return Err(cx.invariant("a table word read of another arity"));
    };
    let ids = match legal::holds_wide(cx.profile) {
        true => HandleIds::Words,
        false => HandleIds::U64,
    };
    let addressed = cx.inputs.addressed_function_tables;
    if addressed && cx.profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::BufferRange,
            vec![cx.profile.addressed_needs()],
        )
        .into());
    }
    let (words, storage_buffers) = match addressed {
        true => (TableWords::Addressed, 1),
        false => (TableWords::Snapshot, 2),
    };
    let reads = TableReads { ids, words };
    if cx.families.access.function_tables.is_none() {
        cx.families.access.function_tables = Some(reads);
        cx.descriptors = cx.descriptors.plus(&DescriptorCounts {
            storage_buffers,
            ..DescriptorCounts::default()
        });
        cx.facts
            .insert(RuntimeFact::FunctionTableRange { addressed });
    }
    let value = |v| {
        cx.values
            .id(v)
            .ok_or_else(|| cx.invariant("an operand out of the value range"))
    };
    Ok(Some(TableWord {
        id: value(id)?,
        word: value(word)?,
        reads,
    }))
}

/// The words of the directory's search, counted once per module, with the census name of the reads
/// that call it.
pub fn routine_words(tables: Option<TableReads>) -> impl Iterator<Item = (&'static str, u64)> {
    tables
        .map(|reads| {
            let words =
                handles::lookup_routine_words(reads.ids == HandleIds::U64, lookup_form(reads));
            ("access.function_table.word", u64::from(words))
        })
        .into_iter()
}

/// The directory search's form for `reads`: an addressed read takes the entry's index, whose
/// element and flags words hold the address; a snapshot read the element, its record's offset.
fn lookup_form(reads: TableReads) -> handles::Lookup {
    match reads.words {
        TableWords::Addressed => handles::Lookup::Entry,
        TableWords::Snapshot => handles::Lookup::Element,
    }
}

/// Declare the stage's function-table directory (an entry array, or a word view on a profile
/// without `shaderInt64`), the module's search of it, and the stage's snapshot (none for an
/// addressed read), when the plan reads a table by its resource id.
pub fn declare(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<Option<FunctionTables>> {
    let Some(reads) = plan.families.access.function_tables else {
        return Ok(None);
    };
    let directory = handles::declare_table(
        mb,
        directory_symbol(plan.stage),
        reads.ids == HandleIds::U64,
    )?;
    let lookup = handles::define_lookup(mb, directory, lookup_form(reads))?;
    if reads.words == TableWords::Addressed {
        return Ok(Some(FunctionTables {
            lookup,
            words: DeclaredWords::Addressed(directory),
        }));
    }
    let view = views::declare_at(mb, snapshot_symbol(plan.stage), views::ViewKind::Storage)?;
    Ok(Some(FunctionTables {
        lookup,
        words: DeclaredWords::Snapshot(view),
    }))
}

/// Emit read `s`, defining value `vid`: the id's record offset `e` in the directory, then, below the record's count `snapshot[e]`, the snapshot's word `e + 1 + w`, and 0 past it;
/// or the address the id's directory entry holds, then its word `w` below the table's words (the
/// size word's 2 more), and 0 past them or for the null address ([`addressed_word`]).
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &TableWord,
) -> BResult<()> {
    let Some(tables) = cx.decl.families.access.function_tables else {
        return Err(cx.fail("a table word read without its directory"));
    };
    let id = cx.operand(fb, s.id)?;
    let (lo, hi) = match s.reads.ids {
        HandleIds::Words => (
            fb.composite_extract(id, &[0])?,
            fb.composite_extract(id, &[1])?,
        ),
        HandleIds::U64 => {
            let u32_ty = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let lo = fb.convert(ConvOp::UConvert, id, u32_ty)?;
            let shift = k32(fb, 32)?;
            let high = fb.binary(BinOp::ShiftRightLogical, id, shift)?;
            (lo, fb.convert(ConvOp::UConvert, high, u32_ty)?)
        }
    };
    let Some(element) = fb.call(tables.lookup, &[lo, hi])? else {
        return Err(cx.fail("a directory search without a result"));
    };
    let word = cx.operand(fb, s.word)?;
    let zero = k32(fb, 0)?;
    let w = match tables.words {
        DeclaredWords::Addressed(directory) => {
            let Some(w) = addressed_word(fb, directory, element, word)? else {
                return Err(cx.fail("an addressed table read without its merged word"));
            };
            w
        }
        DeclaredWords::Snapshot(view) => {
            // The record's count, then its word `w` below it: every read stays in the record.
            let count = views::read_word(fb, view, element)?;
            let inside = fb.binary(BinOp::ULessThan, word, count)?;
            let one = k32(fb, 1)?;
            let first = fb.binary(BinOp::IAdd, element, one)?;
            let at = fb.binary(BinOp::IAdd, first, word)?;
            let at = fb.select(inside, at, zero)?;
            let w = views::read_word(fb, view, at)?;
            fb.select(inside, w, zero)?
        }
    };
    cx.define(vid, w);
    Ok(())
}

/// The device address directory entry `entry` holds: its element word low, its flags word high.
fn entry_address(
    fb: &mut FunctionBuilder<'_>,
    directory: handles::Table,
    entry: crate::spirv::ids::Value,
) -> BResult<crate::spirv::ids::Value> {
    let lo = handles::entry_field(fb, directory, entry, EntryField::Element)?;
    let hi = handles::entry_field(fb, directory, entry, EntryField::Flags)?;
    let u64_ty = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    let lo = fb.convert(ConvOp::UConvert, lo, u64_ty)?;
    let hi = fb.convert(ConvOp::UConvert, hi, u64_ty)?;
    let shift = k64(fb, 32)?;
    let hi = fb.binary(BinOp::ShiftLeftLogical, hi, shift)?;
    fb.binary(BinOp::BitwiseOr, lo, hi)
}

/// Word `word` of the table whose words lie at the address directory entry `entry` holds: 0 for the
/// null entry (address 0) and for a word past the table's, whose count is its size word's (word 1)
/// plus the two header words. The loads never take the null address: the null entry's reads take
/// the directory's last entry's address, a listed table's, and their words are dropped. A branch
/// around them did not keep a driver from reading the null address: lavapipe faulted at address 0
/// on the null and unknown ids with the loads behind `address != 0`. The
/// binder refuses an addressed directory that lists no table (`bind::heap`), and the branch skips
/// the loads where the last entry's address is null too. `None` only where the branch merged no
/// value.
fn addressed_word(
    fb: &mut FunctionBuilder<'_>,
    directory: handles::Table,
    entry: crate::spirv::ids::Value,
    word: crate::spirv::ids::Value,
) -> BResult<Option<crate::spirv::ids::Value>> {
    let address = entry_address(fb, directory, entry)?;
    let last = handles::last_entry(fb, directory)?;
    let parked = entry_address(fb, directory, last)?;
    let null = k64(fb, 0)?;
    let live = fb.binary(BinOp::INotEqual, address, null)?;
    let at_address = fb.select(live, address, parked)?;
    let readable = fb.binary(BinOp::INotEqual, at_address, null)?;
    let read: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let mut words = PsbMem::at(at_address);
        let one = k32(fb, 1)?;
        let two = k32(fb, 2)?;
        let zero = k32(fb, 0)?;
        let size = words.read(fb, one)?;
        // A header word, or an entry below the size: `w - 2` wraps past every size for a header
        // word, which the first compare takes.
        let header = fb.binary(BinOp::ULessThan, word, two)?;
        let entry = fb.binary(BinOp::ISub, word, two)?;
        let below = fb.binary(BinOp::ULessThan, entry, size)?;
        let inside = fb.binary(BinOp::LogicalOr, header, below)?;
        let at = fb.select(inside, word, zero)?;
        let w = words.read(fb, at)?;
        let kept = fb.binary(BinOp::LogicalAnd, inside, live)?;
        Ok(ArmEnd::FallThrough(vec![fb.select(kept, w, zero)?]))
    });
    let none: ArmFn<'_, ()> = Box::new(|fb, _| Ok(ArmEnd::FallThrough(vec![k32(fb, 0)?])));
    let merged = fb.if_(&mut (), readable, read, Some(none))?;
    Ok(merged.0.first().copied())
}
