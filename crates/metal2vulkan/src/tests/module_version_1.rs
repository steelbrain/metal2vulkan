//! Bitcode module version 1 reads as version 2 does.
//!
//! Version 1 has no string table: its `GLOBALVAR` and `FUNCTION` records carry no name fields, and
//! the module-level `VALUE_SYMTAB` names each global and function by value id (`VST_ENTRY`, and
//! `VST_FNENTRY` with the word offset of a defined function's block, after the function blocks
//! where `VSTOFFSET` says). [`to_version_1`] re-emits a version-2 module (what `llvm-as` and
//! `xcrun metal-as` write) in that layout, as LLVM's version-1 writer (LLVM 4) laid it out, and
//! the tests hold every public fixture to it: each decodes to the module and ABI its version-2
//! bitcode does (or is refused alike), and every engine fixture's entry points translate to the
//! same SPIR-V, which passes L3 (`spirv-val`), or are refused alike.
//!
//! The layout is that writer's: the module opens with `VERSION` and a `BLOCKINFO` of LLVM's four
//! symbol table abbreviations; `FUNCTION` records have version 1's 15 fields and `GLOBALVAR`
//! records 6 or 12 (a field version 2 added is kept where it is not 0); no `SYNC_SCOPE_NAMES`
//! block (where it names only the built-in scopes); every symbol table record is spelled through
//! the abbreviation LLVM's writer chose for its characters, the module's `VST_FNENTRY` through
//! three the module symbol table defines; the producer and epoch through their own. Every other
//! record is written unabbreviated but those with a blob and `VSTOFFSET`, which take an
//! abbreviation of their own block (a reader keeps a record's fields, not its spelling).

use std::path::{Path, PathBuf};

use crate::air::bitstream::{Entry, Reader};
use crate::air::container::bitcode_stream;
use crate::budget::{Budget, Meter};
use crate::error::Error;
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, metal_assemble,
    AssembleError,
};
use crate::spirv::testing::committed_profiles;
use crate::tests::golden_hashes::parallel;
use crate::tests::public_fixtures_engine::refusal;

// The block ids and record codes of LLVM's `LLVMBitCodes.h` (and `BLOCKINFO`'s of
// `BitCodeEnums.h`) the re-emission writes.
const BLOCKINFO_BLOCK: u32 = 0;
const BLOCKINFO_SETBID: u32 = 1;
const MODULE_BLOCK: u32 = 8;
const FUNCTION_BLOCK: u32 = 12;
const IDENTIFICATION_BLOCK: u32 = 13;
const VALUE_SYMTAB_BLOCK: u32 = 14;
const STRTAB_BLOCK: u32 = 23;
const SYMTAB_BLOCK: u32 = 25;
const SYNC_SCOPE_NAMES_BLOCK: u32 = 26;
const IDENTIFICATION_STRING: u32 = 1;
const IDENTIFICATION_EPOCH: u32 = 2;
const MODULE_VERSION: u32 = 1;
const MODULE_GLOBALVAR: u32 = 7;
const MODULE_FUNCTION: u32 = 8;
const MODULE_VSTOFFSET: u32 = 13;
/// Module records that name a value by string-table offset or take a value id, which the
/// re-emission does not rewrite (none of the fixtures holds one): ALIAS_OLD, COMDAT, ALIAS, IFUNC.
const NOT_REWRITTEN: [u32; 4] = [9, 12, 14, 15];
const VST_ENTRY: u32 = 1;
const VST_BBENTRY: u32 = 2;
const VST_FNENTRY: u32 = 3;
/// The abbrev width of every block written: abbreviations 4 to 63.
const WIDTH: u32 = 6;

/// An abbreviation operand, as `DEFINE_ABBREV` spells it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Literal(u64),
    Fixed(u32),
    Vbr(u32),
    /// The rest of the record, each field spelled by the next operand.
    Array,
    Char6,
    Blob,
}

/// LLVM's `BLOCKINFO` abbreviations for `VALUE_SYMTAB` (ids 4 to 7): an entry (or a basic
/// block's) of 8-bit characters, its code a 3-bit field; an entry of 7-bit characters; an entry
/// of char6 ones; a basic block's entry of char6 ones.
const VST_ABBREVS: [&[Op]; 4] = [
    &[Op::Fixed(3), Op::Vbr(8), Op::Array, Op::Fixed(8)],
    &[
        Op::Literal(VST_ENTRY as u64),
        Op::Vbr(8),
        Op::Array,
        Op::Fixed(7),
    ],
    &[
        Op::Literal(VST_ENTRY as u64),
        Op::Vbr(8),
        Op::Array,
        Op::Char6,
    ],
    &[
        Op::Literal(VST_BBENTRY as u64),
        Op::Vbr(8),
        Op::Array,
        Op::Char6,
    ],
];

/// The module symbol table's own abbreviations of `VST_FNENTRY` (ids 8 to 10, after
/// [`VST_ABBREVS`]): 8-bit, 7-bit and char6 characters.
const FNENTRY_ABBREVS: [&[Op]; 3] = [
    &[
        Op::Literal(VST_FNENTRY as u64),
        Op::Vbr(8),
        Op::Vbr(8),
        Op::Array,
        Op::Fixed(8),
    ],
    &[
        Op::Literal(VST_FNENTRY as u64),
        Op::Vbr(8),
        Op::Vbr(8),
        Op::Array,
        Op::Fixed(7),
    ],
    &[
        Op::Literal(VST_FNENTRY as u64),
        Op::Vbr(8),
        Op::Vbr(8),
        Op::Array,
        Op::Char6,
    ],
];

/// The character width LLVM's writer gives a name: 8 bits where a byte has its high bit set,
/// char6 where every byte is one, 7 bits otherwise; as the index of its abbreviation in
/// [`FNENTRY_ABBREVS`].
fn name_width(name: &[u8]) -> usize {
    if name.iter().any(|&b| b >= 0x80) {
        0
    } else if name.iter().all(|&b| char6(b).is_some()) {
        2
    } else {
        1
    }
}

fn char6(b: u8) -> Option<u64> {
    Some(u64::from(match b {
        b'a'..=b'z' => b - b'a',
        b'A'..=b'Z' => b - b'A' + 26,
        b'0'..=b'9' => b - b'0' + 52,
        b'.' => 62,
        b'_' => 63,
        _ => return None,
    }))
}

/// An LSB-first bitstream writer.
struct BitWriter {
    bytes: Vec<u8>,
    bit: u64,
    width: u32,
    /// Open blocks: the parent's abbrev width, the byte offset of the length word, and the
    /// abbreviations the block has (its `BLOCKINFO` ones, then those defined in it), the first
    /// with id 4.
    open: Vec<(u32, usize, Vec<Vec<Op>>)>,
    /// The `BLOCKINFO` abbreviations written, by block id.
    blockinfo: Vec<(u32, Vec<Vec<Op>>)>,
}

impl BitWriter {
    fn new() -> BitWriter {
        let mut w = BitWriter {
            bytes: Vec::new(),
            bit: 0,
            width: 2,
            open: Vec::new(),
            blockinfo: Vec::new(),
        };
        for b in *b"BC\xC0\xDE" {
            w.fixed(u64::from(b), 8);
        }
        w
    }

    fn fixed(&mut self, v: u64, n: u32) {
        for i in 0..n {
            if self.bit.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (v >> i) & 1 != 0 {
                *self.bytes.last_mut().unwrap() |= 1 << (self.bit % 8);
            }
            self.bit += 1;
        }
    }

    fn vbr(&mut self, mut v: u64, n: u32) {
        let high = 1u64 << (n - 1);
        while v >= high {
            self.fixed((v & (high - 1)) | high, n);
            v >>= n - 1;
        }
        self.fixed(v, n);
    }

    fn align32(&mut self) {
        while !self.bit.is_multiple_of(32) {
            self.fixed(0, 1);
        }
    }

    /// The word the next entry starts at, which must start one.
    fn word(&self) -> u64 {
        assert!(self.bit.is_multiple_of(32), "an entry off a word boundary");
        self.bit / 32
    }

    fn enter(&mut self, id: u32) {
        self.fixed(1, self.width);
        self.vbr(u64::from(id), 8);
        self.vbr(u64::from(WIDTH), 4);
        self.align32();
        let at = self.bytes.len();
        self.fixed(0, 32);
        let inherited = self
            .blockinfo
            .iter()
            .find(|(block, _)| *block == id)
            .map(|(_, abbrevs)| abbrevs.clone())
            .unwrap_or_default();
        self.open.push((self.width, at, inherited));
        self.width = WIDTH;
    }

    fn end(&mut self) {
        self.fixed(0, self.width);
        self.align32();
        let (width, at, _) = self.open.pop().expect("an open block");
        let words = u32::try_from((self.bytes.len() - at - 4) / 4).expect("a block length");
        self.bytes[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = width;
    }

    fn record(&mut self, code: u32, ops: &[u64]) {
        self.fixed(3, self.width);
        self.vbr(u64::from(code), 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// `DEFINE_ABBREV` of `ops`, which the caller files.
    fn define(&mut self, ops: &[Op]) {
        self.fixed(2, self.width);
        self.vbr(ops.len() as u64, 5);
        for &op in ops {
            let (encoding, width) = match op {
                Op::Literal(v) => {
                    self.fixed(1, 1);
                    self.vbr(v, 8);
                    continue;
                }
                Op::Fixed(n) => (1, Some(n)),
                Op::Vbr(n) => (2, Some(n)),
                Op::Array => (3, None),
                Op::Char6 => (4, None),
                Op::Blob => (5, None),
            };
            self.fixed(0, 1);
            self.fixed(encoding, 3);
            if let Some(n) = width {
                self.vbr(u64::from(n), 5);
            }
        }
    }

    /// The id of the innermost block's abbreviation `ops`, defined here at its first use.
    fn abbrev(&mut self, ops: &[Op]) -> u64 {
        let defined = &self.open.last().expect("an open block").2;
        if let Some(i) = defined.iter().position(|o| o == ops) {
            return 4 + i as u64;
        }
        self.define(ops);
        let defined = &mut self.open.last_mut().expect("an open block").2;
        defined.push(ops.to_vec());
        let id = 3 + defined.len() as u64;
        assert!(id < 1 << WIDTH, "too many abbreviations in one block");
        id
    }

    /// `BLOCKINFO` with `abbrevs` for block `id`, which every later block `id` starts with.
    fn blockinfo(&mut self, id: u32, abbrevs: &[&[Op]]) {
        self.enter(BLOCKINFO_BLOCK);
        self.record(BLOCKINFO_SETBID, &[u64::from(id)]);
        for ops in abbrevs {
            self.define(ops);
        }
        self.end();
        let abbrevs = abbrevs.iter().map(|ops| ops.to_vec()).collect();
        self.blockinfo.push((id, abbrevs));
    }

    fn scalar(&mut self, op: Op, v: u64) {
        match op {
            Op::Literal(l) => assert_eq!(v, l, "a field its literal does not spell"),
            Op::Fixed(n) => {
                assert!(n == 64 || v >> n == 0, "{v} does not fit {n} bits");
                self.fixed(v, n);
            }
            Op::Vbr(n) => self.vbr(v, n),
            Op::Char6 => {
                let c = u8::try_from(v)
                    .ok()
                    .and_then(char6)
                    .expect("a char6 character");
                self.fixed(c, 6);
            }
            Op::Array | Op::Blob => panic!("{op:?} is not a scalar"),
        }
    }

    /// Record `code` with `ops` (and `blob`) through the innermost block's abbreviation `id`.
    fn abbreviated(&mut self, id: u64, code: u32, ops: &[u64], blob: Option<&[u8]>) {
        let abbrev = self.open.last().expect("an open block").2[(id - 4) as usize].clone();
        self.fixed(id, self.width);
        let fields: Vec<u64> = std::iter::once(u64::from(code))
            .chain(ops.iter().copied())
            .collect();
        let mut fields = fields.into_iter();
        let mut spelled = abbrev.iter();
        while let Some(&op) = spelled.next() {
            match op {
                Op::Array => {
                    let element = *spelled.next().expect("an array's element");
                    let rest: Vec<u64> = fields.by_ref().collect();
                    self.vbr(rest.len() as u64, 6);
                    for v in rest {
                        self.scalar(element, v);
                    }
                }
                Op::Blob => {
                    let blob = blob.expect("a blob");
                    self.vbr(blob.len() as u64, 6);
                    self.align32();
                    for &b in blob {
                        self.fixed(u64::from(b), 8);
                    }
                    self.align32();
                }
                op => self.scalar(op, fields.next().expect("a field per operand")),
            }
        }
        assert!(fields.next().is_none(), "fields past the abbreviation");
    }

    fn blob_record(&mut self, code: u32, ops: &[u64], blob: &[u8]) {
        let mut abbrev = vec![Op::Literal(u64::from(code))];
        abbrev.extend(ops.iter().map(|_| Op::Vbr(6)));
        abbrev.push(Op::Blob);
        let id = self.abbrev(&abbrev);
        self.abbreviated(id, code, ops, Some(blob));
    }

    /// `code` with one 32-bit field, so its value moves nothing after it (`VSTOFFSET`, as LLVM
    /// writes it).
    fn fixed32_record(&mut self, code: u32, value: u64) {
        let id = self.abbrev(&[Op::Literal(u64::from(code)), Op::Fixed(32)]);
        self.abbreviated(id, code, &[value], None);
    }

    fn finish(self) -> Vec<u8> {
        assert!(self.open.is_empty());
        assert!(self.bit.is_multiple_of(32));
        self.bytes
    }
}

/// One entry read, its record's operands copied out of the reader.
enum Item<'a> {
    Block(u32),
    End,
    Record(u32, Vec<u64>, Option<&'a [u8]>),
    Stream,
}

fn next<'a>(r: &mut Reader<'a>, m: &mut Meter) -> Item<'a> {
    match r.advance(m).expect("a well-formed stream") {
        Entry::SubBlock { id } => Item::Block(id),
        Entry::EndBlock { .. } => Item::End,
        Entry::Record(rec) => Item::Record(rec.code, rec.ops.to_vec(), rec.blob),
        Entry::End => Item::Stream,
    }
}

fn write_record(w: &mut BitWriter, code: u32, ops: &[u64], blob: Option<&[u8]>) {
    match blob {
        Some(blob) => w.blob_record(code, ops, blob),
        None => w.record(code, ops),
    }
}

/// A symbol table record as LLVM's writer spells it: `VST_ENTRY` and `VST_BBENTRY` through
/// [`VST_ABBREVS`] by their characters' width (a basic block's 7-bit name takes the 8-bit one),
/// `VST_FNENTRY` through [`FNENTRY_ABBREVS`].
fn vst_record(w: &mut BitWriter, code: u32, ops: &[u64]) {
    let chars = if code == VST_FNENTRY { 2 } else { 1 };
    let name: Vec<u8> = ops[chars..]
        .iter()
        .map(|&c| u8::try_from(c).expect("a byte"))
        .collect();
    let width = name_width(&name);
    let id = match code {
        VST_ENTRY => [4, 5, 6][width],
        VST_BBENTRY => [4, 4, 7][width],
        VST_FNENTRY => [8, 9, 10][width],
        _ => return w.record(code, ops),
    };
    w.abbreviated(id, code, ops, None);
}

/// Copy the rest of block `id`, just entered, as read (a function's symbol table spelled as
/// LLVM's writer spells it).
fn copy_block(r: &mut Reader<'_>, w: &mut BitWriter, id: u32, m: &mut Meter) {
    w.enter(id);
    loop {
        match next(r, m) {
            Item::Block(id) => copy_block(r, w, id, m),
            Item::End => break,
            Item::Record(code, ops, None) if id == VALUE_SYMTAB_BLOCK => vst_record(w, code, &ops),
            Item::Record(code, ops, blob) => write_record(w, code, &ops, blob),
            Item::Stream => panic!("the stream ends inside block {id}"),
        }
    }
    w.end();
}

/// The string table's blob.
fn string_table<'a>(stream: &'a [u8], m: &mut Meter) -> &'a [u8] {
    let mut r = Reader::new(stream).expect("a bitstream");
    let mut table = None;
    loop {
        match next(&mut r, m) {
            Item::Block(STRTAB_BLOCK) => loop {
                match next(&mut r, m) {
                    Item::Record(_, _, blob) => table = blob,
                    Item::End => break,
                    _ => panic!("a string table holds one blob"),
                }
            },
            Item::Block(_) => r.skip_block(m).unwrap(),
            Item::Stream => return table.expect("a version-2 module has a string table"),
            _ => panic!("a record at the top level"),
        }
    }
}

/// A global or function of the module, by value id: its name, and for a function whether it has a
/// body (`None` for a global).
type Values = Vec<(Vec<u8>, Option<bool>)>;

/// The module symbol table of version 1, as LLVM's version-1 writer wrote it: a name for each
/// named global and function, by value id, with the word of its block for a defined function
/// (`VST_FNENTRY`, through the abbreviations it defines first). An unnamed value has no entry.
/// LLVM wrote the entries in its symbol table's hash order; they are written last value first.
fn symbol_table(w: &mut BitWriter, values: &Values, blocks: &[u64]) {
    w.enter(VALUE_SYMTAB_BLOCK);
    for (i, ops) in FNENTRY_ABBREVS.iter().enumerate() {
        assert_eq!(w.abbrev(ops), 8 + i as u64, "after the four of BLOCKINFO");
    }
    let mut bodies = blocks.iter();
    let mut entries = Vec::new();
    for (id, (name, defined)) in values.iter().enumerate() {
        let body = (*defined == Some(true)).then(|| *bodies.next().expect("a block per body"));
        if name.is_empty() {
            continue;
        }
        let chars = name.iter().map(|&b| u64::from(b));
        entries.push(match body {
            Some(word) => (
                VST_FNENTRY,
                [id as u64, word].into_iter().chain(chars).collect(),
            ),
            None => (VST_ENTRY, std::iter::once(id as u64).chain(chars).collect()),
        });
    }
    assert!(bodies.next().is_none(), "a block per body");
    for (code, ops) in entries.iter().rev() {
        vst_record(w, *code, ops);
    }
    w.end();
}

/// What a re-emission wrote in LLVM's version-1 forms, beyond the names.
#[derive(Default, Clone, Copy)]
struct Shape {
    /// `FUNCTION` records cut to version 1's 15 fields.
    functions_cut: usize,
    /// `FUNCTION` records kept longer, a later field not 0.
    functions_long: usize,
    /// `SYNC_SCOPE_NAMES` blocks left out, holding only LLVM's two built-in scopes.
    scopes_dropped: usize,
}

/// A `GLOBALVAR` or `FUNCTION` record's fields as LLVM's version-1 writer wrote them: a function
/// has 15 (to `personalityfn`), a global 6 or 12 (to `comdat`); a later field version 2 added is
/// kept where it is not 0, and so are the fields before it.
fn version_1_fields(code: u32, fields: &[u64], shape: &mut Shape) -> Vec<u64> {
    let lengths: &[usize] = if code == MODULE_FUNCTION {
        &[15]
    } else {
        &[6, 12]
    };
    let cut = lengths
        .iter()
        .copied()
        .find(|&n| fields.iter().skip(n).all(|&f| f == 0))
        .unwrap_or(fields.len())
        .min(fields.len());
    if code == MODULE_FUNCTION {
        if cut == 15 {
            shape.functions_cut += 1;
        } else {
            shape.functions_long += 1;
        }
    }
    fields[..cut].to_vec()
}

/// The rest of a `SYNC_SCOPE_NAMES` block just entered, written unless it names only LLVM's two
/// built-in scopes (`singlethread`, then the system's ``): version 1 wrote no such block, and its
/// scope ids 0 and 1 are those two.
fn sync_scopes(r: &mut Reader<'_>, w: &mut BitWriter, shape: &mut Shape, m: &mut Meter) {
    let mut names = Vec::new();
    loop {
        match next(r, m) {
            Item::Record(code, ops, None) => names.push((code, ops)),
            Item::End => break,
            _ => panic!("a sync scope table holds records"),
        }
    }
    let built_in = [
        (1, b"singlethread".iter().map(|&b| u64::from(b)).collect()),
        (1, Vec::new()),
    ];
    if names == built_in {
        shape.scopes_dropped += 1;
        return;
    }
    w.enter(SYNC_SCOPE_NAMES_BLOCK);
    for (code, ops) in names {
        w.record(code, &ops);
    }
    w.end();
}

/// The rest of the `IDENTIFICATION` block just entered, written as LLVM's writer does (the
/// producer through a char6 abbreviation where it is char6, the epoch through its own), with
/// `producer` in place of the one read when given.
fn identification(r: &mut Reader<'_>, w: &mut BitWriter, producer: Option<&[u8]>, m: &mut Meter) {
    let (mut read, mut epoch) = (None, None);
    loop {
        match next(r, m) {
            Item::Record(IDENTIFICATION_STRING, ops, None) => read = Some(ops),
            Item::Record(IDENTIFICATION_EPOCH, ops, None) => epoch = Some(ops),
            Item::End => break,
            _ => panic!("an identification block holds a producer and an epoch"),
        }
    }
    let chars = match producer {
        Some(p) => p.iter().map(|&b| u64::from(b)).collect(),
        None => read.expect("a producer"),
    };
    w.enter(IDENTIFICATION_BLOCK);
    let code = u64::from(IDENTIFICATION_STRING);
    let id = w.abbrev(&[Op::Literal(code), Op::Array, Op::Char6]);
    if chars
        .iter()
        .all(|&c| u8::try_from(c).ok().and_then(char6).is_some())
    {
        w.abbreviated(id, IDENTIFICATION_STRING, &chars, None);
    } else {
        w.record(IDENTIFICATION_STRING, &chars);
    }
    let id = w.abbrev(&[Op::Literal(u64::from(IDENTIFICATION_EPOCH)), Op::Vbr(6)]);
    w.abbreviated(id, IDENTIFICATION_EPOCH, &epoch.expect("an epoch"), None);
    w.end();
}

/// Rewrite the module block just entered: `VERSION = 1` and LLVM's `BLOCKINFO` of symbol table
/// abbreviations, the global and function records without their string-table fields (and cut to
/// version 1's fields), `VSTOFFSET` as `vst`, no `SYNC_SCOPE_NAMES` of the built-in scopes, and
/// the symbol table naming every value. Returns the word the symbol table starts at (0 when the
/// module has none).
fn module(
    r: &mut Reader<'_>,
    w: &mut BitWriter,
    strtab: &[u8],
    vst: u64,
    shape: &mut Shape,
    m: &mut Meter,
) -> u64 {
    w.enter(MODULE_BLOCK);
    let mut values: Values = Vec::new();
    let mut blocks = Vec::new();
    let mut table = None;
    loop {
        match next(r, m) {
            Item::Record(MODULE_VERSION, ops, _) => {
                assert_eq!(ops, [2], "a version-2 module");
                w.record(MODULE_VERSION, &[1]);
                w.blockinfo(VALUE_SYMTAB_BLOCK, &VST_ABBREVS);
            }
            Item::Record(code @ (MODULE_GLOBALVAR | MODULE_FUNCTION), ops, None) => {
                let (at, len) = (ops[0] as usize, ops[1] as usize);
                let defined = (code == MODULE_FUNCTION).then(|| ops[4] == 0);
                values.push((strtab[at..at + len].to_vec(), defined));
                w.record(code, &version_1_fields(code, &ops[2..], shape));
            }
            Item::Record(MODULE_VSTOFFSET, _, _) => w.fixed32_record(MODULE_VSTOFFSET, vst),
            Item::Record(code, ..) if NOT_REWRITTEN.contains(&code) => {
                panic!("module record {code} is not rewritten")
            }
            Item::Record(code, ops, blob) => write_record(w, code, &ops, blob),
            Item::Block(FUNCTION_BLOCK) => {
                blocks.push(w.word());
                copy_block(r, w, FUNCTION_BLOCK, m);
            }
            Item::Block(VALUE_SYMTAB_BLOCK) => {
                r.skip_block(m).unwrap();
                table = Some(w.word());
                symbol_table(w, &values, &blocks);
            }
            Item::Block(SYNC_SCOPE_NAMES_BLOCK) => sync_scopes(r, w, shape, m),
            Item::Block(id) => copy_block(r, w, id, m),
            Item::End => break,
            Item::Stream => panic!("the stream ends inside the module"),
        }
    }
    // Version 2 writes no module symbol table without a function body; version 1 names the
    // globals and declarations in one.
    if table.is_none() && values.iter().any(|(name, _)| !name.is_empty()) {
        assert!(blocks.is_empty());
        symbol_table(w, &values, &blocks);
    }
    w.end();
    table.unwrap_or(0)
}

/// `v2` (raw or wrapped) re-emitted in module version 1, with `VSTOFFSET` written as `vst` and
/// `producer` in place of its own when given, the word its symbol table starts at, and its
/// version-1 forms.
fn rewrite(v2: &[u8], vst: u64, producer: Option<&[u8]>) -> (Vec<u8>, u64, Shape) {
    let (_, stream) = bitcode_stream(v2).expect("a bitcode stream");
    let mut m = Meter::new(&Budget::default());
    let strtab = string_table(stream, &mut m);
    let mut r = Reader::new(stream).expect("a bitstream");
    let mut w = BitWriter::new();
    let (mut table, mut shape) = (0, Shape::default());
    loop {
        match next(&mut r, &mut m) {
            Item::Block(IDENTIFICATION_BLOCK) => identification(&mut r, &mut w, producer, &mut m),
            Item::Block(MODULE_BLOCK) => {
                table = module(&mut r, &mut w, strtab, vst, &mut shape, &mut m)
            }
            Item::Block(STRTAB_BLOCK | SYMTAB_BLOCK) => r.skip_block(&mut m).unwrap(),
            Item::Block(id) => panic!("top-level block {id}"),
            Item::Stream => break,
            _ => panic!("a record at the top level"),
        }
    }
    (w.finish(), table, shape)
}

/// `v2` in module version 1 (`producer` in place of its own when given), and its version-1
/// forms: written twice, the second time with `VSTOFFSET` naming the word the first found the
/// symbol table at (its fixed width moves nothing).
fn to_version_1(v2: &[u8], producer: Option<&[u8]>) -> (Vec<u8>, Shape) {
    let (_, table, _) = rewrite(v2, 0, producer);
    let (bytes, again, shape) = rewrite(v2, table, producer);
    assert_eq!(table, again);
    (bytes, shape)
}

/// The `VERSION` a module states, and whether a top-level string table follows it.
fn version_and_strtab(bytes: &[u8]) -> (u64, bool) {
    let (_, stream) = bitcode_stream(bytes).expect("a bitcode stream");
    let mut m = Meter::new(&Budget::default());
    let mut r = Reader::new(stream).expect("a bitstream");
    let (mut version, mut strtab) = (None, false);
    loop {
        match next(&mut r, &mut m) {
            Item::Block(MODULE_BLOCK) => {
                if let Item::Record(MODULE_VERSION, ops, _) = next(&mut r, &mut m) {
                    version = ops.first().copied();
                }
                r.skip_block(&mut m).unwrap();
            }
            Item::Block(id) => {
                strtab |= id == STRTAB_BLOCK;
                r.skip_block(&mut m).unwrap();
            }
            Item::Stream => return (version.expect("a VERSION record"), strtab),
            _ => panic!("a record at the top level"),
        }
    }
}

/// `.ll` files directly in `dir`: the opaque-pointer ones (`llvm-as`) and the typed-pointer
/// `.air.ll` ones (`xcrun metal-as`).
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect();
    paths.sort();
    paths
}

fn is_typed(path: &Path) -> bool {
    path.to_string_lossy().ends_with(".air.ll")
}

fn assemble(path: &Path) -> Result<Vec<u8>, AssembleError> {
    if is_typed(path) {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    }
}

/// A refusal without its input location, which differs between the two spellings; anything but a
/// typed refusal is a translator bug.
fn typed(e: &Error, what: &Path) -> String {
    refusal(e).unwrap_or_else(|| panic!("{what:?}: translator bug: {e}"))
}

fn decode(bytes: &[u8]) -> Result<crate::air::Decoded, Error> {
    crate::air::decode(bytes, &mut Meter::new(&Budget::default())).map_err(Error::from)
}

/// Every public fixture, re-emitted in module version 1, decodes to the module and ABI of its
/// version-2 bitcode, or is refused alike (a legacy top-level fixture whose hand-written metadata
/// is not the ABI grammar). The typed-pointer fixtures need `xcrun metal-as` and skip visibly
/// without it.
#[test]
fn every_public_fixture_decodes_alike_in_module_version_1() {
    let (mut compared, mut refused, mut rejected) = (0, 0, 0);
    let mut shapes = Shape::default();
    let mut missing: Option<(String, usize)> = None;
    for path in ll_files(&fixtures_dir())
        .into_iter()
        .chain(ll_files(&engine_dir()))
    {
        let v2 = match assemble(&path) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                missing.get_or_insert((why, 0)).1 += 1;
                continue;
            }
            // A legacy top-level fixture this assembler does not take (the inventory's
            // `unassembled`); the engine fixtures always assemble.
            Err(AssembleError::Rejected(why)) => {
                assert!(!path.starts_with(engine_dir()), "{path:?}: {why}");
                rejected += 1;
                continue;
            }
        };
        let (v1, shape) = to_version_1(&v2, None);
        shapes.functions_cut += shape.functions_cut;
        shapes.functions_long += shape.functions_long;
        shapes.scopes_dropped += shape.scopes_dropped;
        assert_eq!(version_and_strtab(&v2), (2, true), "{path:?}");
        assert_eq!(version_and_strtab(&v1), (1, false), "{path:?}");
        match (decode(&v2), decode(&v1)) {
            (Ok(a), Ok(b)) => {
                assert!(a == b, "{path:?}: version 1 decodes to another module");
                compared += 1;
            }
            (Err(a), Err(b)) => {
                assert_eq!(typed(&a, &path), typed(&b, &path), "{path:?}");
                refused += 1;
            }
            (a, b) => panic!(
                "{path:?}: version 2 {:?}, version 1 {:?}",
                a.map(|_| ()),
                b.map(|_| ())
            ),
        }
    }
    if let Some((why, n)) = missing {
        eprintln!("SKIPPED {n} fixtures: {why}");
    }
    eprintln!("{compared} decoded alike, {refused} refused alike, {rejected} unassembled");
    eprintln!(
        "{} FUNCTION records cut to 15 fields, {} kept longer; {} SYNC_SCOPE_NAMES left out",
        shapes.functions_cut, shapes.functions_long, shapes.scopes_dropped
    );
    assert!(compared > 400, "only {compared} fixtures compared");
    assert!(
        shapes.functions_cut > 400 && shapes.scopes_dropped > 400,
        "too few fixtures in version 1's own forms"
    );
}

/// The profile [`engine_fixtures_translate_alike_in_module_version_1`] compiles for.
const PROFILE: &str = "lunarg-desktop-baseline-2023";

/// Every entry point of every opaque-pointer engine fixture, compiled from its module-version-1
/// re-emission with an empty producer (as Apple's compiler writes it), translates to the SPIR-V
/// words its version-2 bitcode does, and those pass L3 (`spirv-val`); or both are refused alike.
#[test]
fn engine_fixtures_translate_alike_in_module_version_1() {
    let profiles = committed_profiles().expect("committed profiles");
    let (_, p) = profiles
        .iter()
        .find(|(stem, _)| stem == PROFILE)
        .unwrap_or_else(|| panic!("no committed profile {PROFILE}"));
    let paths: Vec<PathBuf> = ll_files(&engine_dir())
        .into_iter()
        .filter(|path| !is_typed(path))
        .collect();
    let counts = parallel(paths.len(), |i| {
        let path = &paths[i];
        let v2 = match llvm_assemble(path) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => return Err(why),
            Err(AssembleError::Rejected(why)) => panic!("{path:?}: {why}"),
        };
        let (v1, _) = to_version_1(&v2, Some(b""));
        let entries = decode(&v2)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"))
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect::<Vec<_>>();
        let (mut ok, mut refused) = (0, 0);
        for entry in entries {
            match (compile(&v2, &entry, p), compile(&v1, &entry, p)) {
                (Ok(a), Ok(b)) => {
                    let (_, a) = link_and_validate(&a, p)
                        .unwrap_or_else(|why| panic!("{path:?} {entry}: L3 failed: {why}"));
                    let (_, b) = link_and_validate(&b, p).unwrap_or_else(|why| {
                        panic!("{path:?} {entry}: version 1: L3 failed: {why}")
                    });
                    assert!(a == b, "{path:?} {entry}: version 1 translates otherwise");
                    ok += 1;
                }
                (Err(a), Err(b)) => {
                    assert_eq!(typed(&a, path), typed(&b, path), "{path:?} {entry}");
                    refused += 1;
                }
                (a, b) => panic!(
                    "{path:?} {entry}: version 2 {:?}, version 1 {:?}",
                    a.map(|_| ()),
                    b.map(|_| ())
                ),
            }
        }
        Ok((ok, refused))
    });
    let (mut ok, mut refused) = (0, 0);
    for c in counts {
        match c {
            Ok((o, r)) => {
                ok += o;
                refused += r;
            }
            Err(why) => {
                eprintln!("SKIP engine_fixtures_translate_alike_in_module_version_1: {why}");
                return;
            }
        }
    }
    eprintln!("{ok} entry points translated alike on {PROFILE}, {refused} refused alike");
    assert!(ok > 300, "only {ok} entry points translated");
}
