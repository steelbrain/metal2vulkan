//! Bitcode module version 1 reads as version 2 does.
//!
//! Version 1 has no string table: its `GLOBALVAR` and `FUNCTION` records carry no name fields, and
//! the module-level `VALUE_SYMTAB` names each global and function by value id (`VST_ENTRY`, and
//! `VST_FNENTRY` with the word offset of a defined function's block, after the function blocks
//! where `VSTOFFSET` says). [`to_version_1`] re-emits a version-2 module (what `llvm-as` and
//! `xcrun metal-as` write) in that layout, as LLVM's version-1 writer laid it out, and the tests
//! hold every public fixture to it: each decodes to the module and ABI its version-2 bitcode does
//! (or is refused alike), and every engine fixture's entry points translate to the same SPIR-V,
//! which passes L3 (`spirv-val`), or are refused alike.
//!
//! The re-emission writes every record unabbreviated but those with a blob and `VSTOFFSET`, which
//! take an abbreviation of their own block (a reader keeps a record's fields, not its spelling),
//! and no `BLOCKINFO`.

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

// The block ids and record codes of LLVM's `LLVMBitCodes.h` the re-emission rewrites.
const MODULE_BLOCK: u32 = 8;
const FUNCTION_BLOCK: u32 = 12;
const IDENTIFICATION_BLOCK: u32 = 13;
const VALUE_SYMTAB_BLOCK: u32 = 14;
const STRTAB_BLOCK: u32 = 23;
const SYMTAB_BLOCK: u32 = 25;
const MODULE_VERSION: u32 = 1;
const MODULE_GLOBALVAR: u32 = 7;
const MODULE_FUNCTION: u32 = 8;
const MODULE_VSTOFFSET: u32 = 13;
/// Module records that name a value by string-table offset or take a value id, which the
/// re-emission does not rewrite (none of the fixtures holds one): ALIAS_OLD, COMDAT, ALIAS, IFUNC.
const NOT_REWRITTEN: [u32; 4] = [9, 12, 14, 15];
const VST_ENTRY: u32 = 1;
const VST_FNENTRY: u32 = 3;
/// The abbrev width of every block written: abbreviations 4 to 63.
const WIDTH: u32 = 6;

/// An LSB-first bitstream writer.
struct BitWriter {
    bytes: Vec<u8>,
    bit: u64,
    width: u32,
    /// Open blocks: the parent's abbrev width, the byte offset of the length word, and the
    /// abbreviations defined in the block (literal code and operand encodings).
    open: Vec<(u32, usize, Vec<(u32, Vec<(u64, u64)>)>)>,
}

impl BitWriter {
    fn new() -> BitWriter {
        let mut w = BitWriter {
            bytes: Vec::new(),
            bit: 0,
            width: 2,
            open: Vec::new(),
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
        self.open.push((self.width, at, Vec::new()));
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

    /// The id of the innermost block's abbreviation of literal `code` then `ops` (`(encoding,
    /// width)`: 1 fixed, 2 VBR, 5 blob), defined here at its first use.
    fn abbrev(&mut self, code: u32, ops: &[(u64, u64)]) -> u64 {
        let defined = &self.open.last().expect("an open block").2;
        if let Some(i) = defined.iter().position(|(c, o)| *c == code && o == ops) {
            return 4 + i as u64;
        }
        self.fixed(2, self.width);
        self.vbr(1 + ops.len() as u64, 5);
        self.fixed(1, 1);
        self.vbr(u64::from(code), 8);
        for &(encoding, width) in ops {
            self.fixed(0, 1);
            self.fixed(encoding, 3);
            if encoding == 1 || encoding == 2 {
                self.vbr(width, 5);
            }
        }
        let defined = &mut self.open.last_mut().expect("an open block").2;
        defined.push((code, ops.to_vec()));
        let id = 3 + defined.len() as u64;
        assert!(id < 1 << WIDTH, "too many abbreviations in one block");
        id
    }

    fn blob_record(&mut self, code: u32, ops: &[u64], blob: &[u8]) {
        let mut encodings = vec![(2, 6); ops.len()];
        encodings.push((5, 0));
        let id = self.abbrev(code, &encodings);
        self.fixed(id, self.width);
        for &op in ops {
            self.vbr(op, 6);
        }
        self.vbr(blob.len() as u64, 6);
        self.align32();
        for &b in blob {
            self.fixed(u64::from(b), 8);
        }
        self.align32();
    }

    /// `code` with one 32-bit field, so its value moves nothing after it (`VSTOFFSET`, as LLVM
    /// writes it).
    fn fixed32_record(&mut self, code: u32, value: u64) {
        let id = self.abbrev(code, &[(1, 32)]);
        self.fixed(id, self.width);
        self.fixed(value, 32);
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

/// Copy the rest of block `id`, just entered, as read.
fn copy_block(r: &mut Reader<'_>, w: &mut BitWriter, id: u32, m: &mut Meter) {
    w.enter(id);
    loop {
        match next(r, m) {
            Item::Block(id) => copy_block(r, w, id, m),
            Item::End => break,
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
/// named global and function, by value id, with the word of its block for a defined function. An
/// unnamed value has no entry.
fn symbol_table(w: &mut BitWriter, values: &Values, blocks: &[u64]) {
    w.enter(VALUE_SYMTAB_BLOCK);
    let mut bodies = blocks.iter();
    for (id, (name, defined)) in values.iter().enumerate() {
        let body = (*defined == Some(true)).then(|| *bodies.next().expect("a block per body"));
        if name.is_empty() {
            continue;
        }
        let chars = name.iter().map(|&b| u64::from(b));
        match body {
            Some(word) => {
                let ops: Vec<u64> = [id as u64, word].into_iter().chain(chars).collect();
                w.record(VST_FNENTRY, &ops);
            }
            None => {
                let ops: Vec<u64> = std::iter::once(id as u64).chain(chars).collect();
                w.record(VST_ENTRY, &ops);
            }
        }
    }
    assert!(bodies.next().is_none(), "a block per body");
    w.end();
}

/// Rewrite the module block just entered: `VERSION = 1`, the global and function records without
/// their string-table fields, `VSTOFFSET` as `vst`, and the symbol table naming every value.
/// Returns the word the symbol table starts at (0 when the module has none).
fn module(r: &mut Reader<'_>, w: &mut BitWriter, strtab: &[u8], vst: u64, m: &mut Meter) -> u64 {
    w.enter(MODULE_BLOCK);
    let mut values: Values = Vec::new();
    let mut blocks = Vec::new();
    let mut table = None;
    loop {
        match next(r, m) {
            Item::Record(MODULE_VERSION, ops, _) => {
                assert_eq!(ops, [2], "a version-2 module");
                w.record(MODULE_VERSION, &[1]);
            }
            Item::Record(code @ (MODULE_GLOBALVAR | MODULE_FUNCTION), ops, None) => {
                let (at, len) = (ops[0] as usize, ops[1] as usize);
                let defined = (code == MODULE_FUNCTION).then(|| ops[4] == 0);
                values.push((strtab[at..at + len].to_vec(), defined));
                w.record(code, &ops[2..]);
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

/// `v2` (raw or wrapped) re-emitted in module version 1, with `VSTOFFSET` written as `vst`, and
/// the word its symbol table starts at.
fn rewrite(v2: &[u8], vst: u64) -> (Vec<u8>, u64) {
    let (_, stream) = bitcode_stream(v2).expect("a bitcode stream");
    let mut m = Meter::new(&Budget::default());
    let strtab = string_table(stream, &mut m);
    let mut r = Reader::new(stream).expect("a bitstream");
    let mut w = BitWriter::new();
    let mut table = 0;
    loop {
        match next(&mut r, &mut m) {
            Item::Block(IDENTIFICATION_BLOCK) => {
                copy_block(&mut r, &mut w, IDENTIFICATION_BLOCK, &mut m)
            }
            Item::Block(MODULE_BLOCK) => table = module(&mut r, &mut w, strtab, vst, &mut m),
            Item::Block(STRTAB_BLOCK | SYMTAB_BLOCK) => r.skip_block(&mut m).unwrap(),
            Item::Block(id) => panic!("top-level block {id}"),
            Item::Stream => break,
            _ => panic!("a record at the top level"),
        }
    }
    (w.finish(), table)
}

/// `v2` in module version 1: written twice, the second time with `VSTOFFSET` naming the word
/// the first found the symbol table at (its fixed width moves nothing).
fn to_version_1(v2: &[u8]) -> Vec<u8> {
    let (_, table) = rewrite(v2, 0);
    let (bytes, again) = rewrite(v2, table);
    assert_eq!(table, again);
    bytes
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
        let v1 = to_version_1(&v2);
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
    assert!(compared > 400, "only {compared} fixtures compared");
}

/// The profile [`engine_fixtures_translate_alike_in_module_version_1`] compiles for.
const PROFILE: &str = "lunarg-desktop-baseline-2023";

/// Every entry point of every opaque-pointer engine fixture, compiled from its module-version-1
/// re-emission, translates to the SPIR-V words its version-2 bitcode does, and those pass L3
/// (`spirv-val`); or both are refused alike.
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
        let v1 = to_version_1(&v2);
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
