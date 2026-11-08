//! The top level and the `MODULE` block: identification, string table, module records, and the
//! module-level sub-blocks.
//!
//! Both module versions with relative value ids are read. Version 2 keeps global and function
//! names in the `STRTAB` block after the module, and its `GLOBALVAR` and `FUNCTION` records start
//! with each name's offset and size there. Version 1 has no string table: its records carry no
//! name fields, and the module-level `VALUE_SYMTAB` names each global and function by value id
//! (`VST_ENTRY`, and `VST_FNENTRY` with the word offset of a defined function's block). LLVM writes
//! that table after the function blocks (with `VSTOFFSET`) or, before LLVM 3.8, ahead of them.
//! Version 0 (absolute value ids) is refused.
//!
//! Decoding makes two passes over the top level: the first reads `IDENTIFICATION` and `STRTAB`,
//! and in a version-1 module the module-level records and that symbol table, skipping every other
//! block by its length word; the second decodes the module, its records named as they are read.
//! Both are linear.
//!
//! Records that identify symbols to LLVM's own tooling rather than describe the program are
//! checked and dropped: `HASH`, and LLVM 23's `GUIDLIST`.

use super::attrs::{self, AttrList, Groups};
use super::codes::*;
use super::constants::{self, ConstCx};
use super::enum_attrs::EnumAttrTable;
use super::function::{self, FnCx};
use super::markers;
use super::metadata::{self, MdCx, MdKinds, MdValue};
use super::stream::{Item, Stream};
use super::types::{self, grow, BitcodeTypes};
use super::{DecodeOptions, ModuleTables, ModuleValue};
use crate::bitstream::Reader;
use crate::budget::Meter;
use crate::container::BITCODE_MAGIC;
use crate::error::{BitcodeError, ContainerError, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature};
use crate::intrinsics::{self, DeclClass};
use crate::ir::{
    AddrSpace, Align, CallConv, Const, ConstId, Constant, DataLayout, FuncId, Function, Global,
    GlobalId, Linkage, ModuleParts, Param, Section, SyncScope, Type, UnnamedAddr, Visibility,
};

/// Relative value ids, names in the module-level `VALUE_SYMTAB`.
const MODULE_VERSION_1: u64 = 1;
/// Relative value ids, names in `STRTAB`.
const MODULE_VERSION_2: u64 = 2;
/// `CallingConv::C` and `CallingConv::Fast`.
const CC_C: u64 = 0;
const CC_FAST: u64 = 8;

/// What the first pass finds at the top level.
struct TopLevel<'a> {
    producer: Option<Vec<u8>>,
    strtab: Option<&'a [u8]>,
    /// A version-1 module's symbol table (empty for any other module).
    names: Names,
}

/// A version-1 module's names: the entries of its module-level `VALUE_SYMTAB`.
#[derive(Debug, Default)]
struct Names {
    /// Every name's bytes, back to back.
    bytes: Vec<u8>,
    /// The entries, sorted by value id (each value id at most once).
    entries: Vec<NameEntry>,
}

#[derive(Clone, Copy, Debug)]
struct NameEntry {
    /// The module value id the entry names.
    value: u64,
    /// `VST_FNENTRY`'s word offset of the function's block; `None` for `VST_ENTRY`.
    offset: Option<u64>,
    /// The name's range in [`Names::bytes`].
    start: usize,
    end: usize,
}

impl Names {
    /// Read the module-level `VALUE_SYMTAB` just entered: `VST_ENTRY: [valueid, namechar x N]`
    /// and `VST_FNENTRY: [valueid, offset, namechar x N]`. A name holds no NUL (LLVM's reader
    /// refuses one), and a value is named once.
    fn read(s: &mut Stream<'_>, m: &mut Meter) -> Result<Names, Error> {
        const B: u32 = VALUE_SYMTAB_BLOCK;
        let mut names = Names::default();
        let Names { bytes, entries } = &mut names;
        s.records(m, B, |s, code, _, m| {
            let (offset, first) = match code {
                VST_ENTRY => (None, 1),
                VST_FNENTRY => (Some(s.op(B, code, 1)?), 2),
                // Block names belong to a function's symbol table.
                VST_BBENTRY => return Err(s.malformed(B, code)),
                VST_COMBINED_ENTRY => {
                    return Err(s.unsupported(Feature::BitcodeRecord { block: B, code }))
                }
                _ => return Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
            };
            let value = s.op(B, code, 0)?;
            let chars = s.ops().get(first..).unwrap_or(&[]);
            let start = bytes.len();
            m.grow(bytes, chars.len())?;
            for &c in chars {
                match u8::try_from(c) {
                    Ok(b) if b != 0 => bytes.push(b),
                    _ => return Err(s.malformed(B, code)),
                }
            }
            grow(m, entries)?;
            entries.push(NameEntry {
                value,
                offset,
                start,
                end: bytes.len(),
            });
            Ok(())
        })?;
        entries.sort_unstable_by_key(|e| e.value);
        if entries
            .windows(2)
            .any(|w| matches!(w, [a, b] if a.value == b.value))
        {
            return Err(s.malformed(B, VST_ENTRY));
        }
        Ok(names)
    }

    /// The entry naming module value `value`.
    fn get(&self, value: u64) -> Option<(NameEntry, &[u8])> {
        let i = self
            .entries
            .binary_search_by_key(&value, |e| e.value)
            .ok()?;
        let e = *self.entries.get(i)?;
        Some((e, self.bytes.get(e.start..e.end)?))
    }
}

/// Decode the module-level tables of a bitcode stream (starting with `BC 0xC0DE`), keeping what
/// `opts` asks for beyond what translation reads.
pub(crate) fn decode(
    stream: &[u8],
    opts: DecodeOptions,
    m: &mut Meter,
) -> Result<ModuleTables, Error> {
    let top = top_level(stream, m)?;
    let mut s = Stream::new(Reader::new(stream)?);
    loop {
        match s.next(m)? {
            Item::SubBlock(MODULE_BLOCK) => break,
            Item::SubBlock(_) => s.skip_block(m)?,
            // The first pass proved exactly one module block exists.
            Item::Record { .. } | Item::EndBlock | Item::End => {
                return Err(s.err(BitcodeError::MalformedRecord {
                    block: MODULE_BLOCK,
                    code: 0,
                }))
            }
        }
    }
    let mut d = ModuleDecoder {
        opts,
        enum_attrs: EnumAttrTable::default(),
        parts: ModuleParts::default(),
        strtab: top.strtab,
        names: top.names,
        named: 0,
        version: None,
        bc_types: None,
        groups: Groups::default(),
        attr_lists: Vec::new(),
        sections: Vec::new(),
        values: Vec::new(),
        pending_inits: Vec::new(),
        md_kinds: MdKinds::default(),
        sync_scopes: Vec::new(),
        defined: Vec::new(),
        intrinsics: Vec::new(),
        marker_consts: Vec::new(),
        function_blocks: 0,
        fn_entries: Vec::new(),
        block_bits: Vec::new(),
        inst_total: 0,
        vst_offset: None,
        vst_bit: None,
        guid_entries: 0,
        guid_values: Vec::new(),
    };
    if let Some(p) = top.producer {
        d.parts.symbols.intern(&p, m)?;
    }
    d.module_block(&mut s, m)?;
    d.finish(m)
}

/// First pass: identification, string table, and exactly one module block.
fn top_level<'a>(stream: &'a [u8], m: &mut Meter) -> Result<TopLevel<'a>, Error> {
    let reader = Reader::new(stream)?;
    if reader.magic() != BITCODE_MAGIC {
        return Err(InputError::new(
            InputErrorKind::Container(ContainerError::BadMagic),
            Location::bitstream(0, &[], None),
        )
        .into());
    }
    let mut s = Stream::new(reader);
    let mut top = TopLevel {
        producer: None,
        strtab: None,
        names: Names::default(),
    };
    let mut modules = 0u32;
    loop {
        match s.next(m)? {
            Item::SubBlock(IDENTIFICATION_BLOCK) => {
                if modules > 0 || top.producer.is_some() {
                    return Err(s.err(BitcodeError::BadIdentification));
                }
                top.producer = Some(identification(&mut s, m)?);
            }
            Item::SubBlock(MODULE_BLOCK) => {
                modules += 1;
                if modules > 1 {
                    return Err(s.err(BitcodeError::UnknownBlock {
                        block: MODULE_BLOCK,
                    }));
                }
                top.names = module_names(&mut s, m)?;
            }
            Item::SubBlock(STRTAB_BLOCK) => {
                if top.strtab.is_some() {
                    return Err(s.err(BitcodeError::BadStringTable));
                }
                top.strtab = Some(strtab(&mut s, m)?);
            }
            Item::SubBlock(SYMTAB_BLOCK) => s.skip_block(m)?,
            Item::SubBlock(block) => return Err(s.err(BitcodeError::UnknownBlock { block })),
            Item::End => break,
            // Only blocks live at the top level (the reader enforces it).
            Item::Record { .. } | Item::EndBlock => {
                return Err(s.err(BitcodeError::MalformedRecord {
                    block: MODULE_BLOCK,
                    code: 0,
                }))
            }
        }
    }
    if modules == 0 {
        return Err(InputError::new(
            InputErrorKind::Bitcode(BitcodeError::MalformedRecord {
                block: MODULE_BLOCK,
                code: 0,
            }),
            Location::module(),
        )
        .into());
    }
    Ok(top)
}

/// The first pass over the module block just entered: a version-1 module's symbol table, read
/// ahead of the records it names (LLVM writes it after the function blocks). The module-level
/// records are read past and every other block skipped by its length word; a module that does not
/// open with `VERSION = 1` is skipped whole, and the second pass checks its version.
fn module_names(s: &mut Stream<'_>, m: &mut Meter) -> Result<Names, Error> {
    match s.next(m)? {
        Item::Record {
            code: MODULE_VERSION,
            ..
        } if s.ops().first() == Some(&MODULE_VERSION_1) => {}
        Item::SubBlock(_) => {
            s.skip_block(m)?;
            s.skip_block(m)?;
            return Ok(Names::default());
        }
        Item::Record { .. } => {
            s.skip_block(m)?;
            return Ok(Names::default());
        }
        Item::EndBlock => return Ok(Names::default()),
        Item::End => return Err(s.malformed(MODULE_BLOCK, 0)),
    }
    let mut names = None;
    loop {
        match s.next(m)? {
            Item::SubBlock(VALUE_SYMTAB_BLOCK) => {
                if names.is_some() {
                    return Err(s.malformed(VALUE_SYMTAB_BLOCK, 0));
                }
                names = Some(Names::read(s, m)?);
            }
            Item::SubBlock(_) => s.skip_block(m)?,
            Item::Record { .. } => {}
            Item::EndBlock => return Ok(names.unwrap_or_default()),
            Item::End => return Err(s.malformed(MODULE_BLOCK, 0)),
        }
    }
}

/// `IDENTIFICATION`: the producer string and an epoch that must be 0.
fn identification(s: &mut Stream<'_>, m: &mut Meter) -> Result<Vec<u8>, Error> {
    const B: u32 = IDENTIFICATION_BLOCK;
    let mut producer = Vec::new();
    let mut epoch = None;
    s.records(m, B, |s, code, _, m| {
        match code {
            IDENTIFICATION_STRING => producer = s.chars(m, B, code, 0)?,
            IDENTIFICATION_EPOCH => epoch = Some(s.op(B, code, 0)?),
            _ => return Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        }
        Ok(())
    })?;
    if epoch != Some(0) {
        return Err(s.err(BitcodeError::BadIdentification));
    }
    Ok(producer)
}

/// `STRTAB`: one blob.
fn strtab<'a>(s: &mut Stream<'a>, m: &mut Meter) -> Result<&'a [u8], Error> {
    const B: u32 = STRTAB_BLOCK;
    let mut table = None;
    s.records(m, B, |s, code, blob, _| {
        if code != STRTAB_BLOB || table.is_some() {
            return Err(s.err(BitcodeError::BadStringTable));
        }
        table = Some(blob.ok_or_else(|| s.err(BitcodeError::BadStringTable))?);
        Ok(())
    })?;
    table.ok_or_else(|| s.err(BitcodeError::BadStringTable))
}

struct ModuleDecoder<'a> {
    opts: DecodeOptions,
    /// The uninterpreted enum attributes, when `opts` asks for them.
    enum_attrs: EnumAttrTable,
    parts: ModuleParts,
    strtab: Option<&'a [u8]>,
    /// A version-1 module's symbol table, read by the first pass.
    names: Names,
    /// The entries of `names` a global or function took.
    named: usize,
    version: Option<u64>,
    bc_types: Option<BitcodeTypes>,
    groups: Groups,
    attr_lists: Vec<AttrList>,
    sections: Vec<Section>,
    values: Vec<ModuleValue>,
    /// `(global, initializer value id)`, resolved once every module constant exists.
    pending_inits: Vec<(GlobalId, u64)>,
    md_kinds: MdKinds,
    sync_scopes: Vec<Option<SyncScope>>,
    defined: Vec<FuncId>,
    /// Every declaration named `air.*`/`llvm.*` with its class, in function id order.
    intrinsics: Vec<(FuncId, DeclClass)>,
    /// With `opts.drop_unused_markers`, every constant operand of a dropped marker call.
    marker_consts: Vec<ConstId>,
    function_blocks: u32,
    fn_entries: Vec<(FuncId, u64)>,
    /// The bit position of each function block's `ENTER_SUBBLOCK`, in stream order.
    block_bits: Vec<u64>,
    /// Instructions decoded so far (`Limit::InstructionCount`).
    inst_total: u64,
    /// `MODULE_CODE_VSTOFFSET`: the word offset of the module-level `VALUE_SYMTAB`.
    vst_offset: Option<u64>,
    /// The bit position of the module-level `VALUE_SYMTAB` block.
    vst_bit: Option<u64>,
    /// `GUIDLIST` entries read so far (one per module value id).
    guid_entries: u64,
    /// The value ids a `GUIDLIST` assigns a non-zero GUID, checked by `finish`.
    guid_values: Vec<u64>,
}

/// What [`ModuleDecoder::name`] reads of a `GLOBALVAR` or `FUNCTION` record.
struct Named {
    name: crate::ir::Sym,
    /// The operand index of the record's `type` field.
    base: usize,
    /// A version-1 `VST_FNENTRY`'s word offset of the function's block.
    offset: Option<u64>,
}

impl<'a> ModuleDecoder<'a> {
    fn module_block(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        const B: u32 = MODULE_BLOCK;
        loop {
            let item = s.next(m)?;
            if self.version.is_none()
                && !matches!(
                    item,
                    Item::Record {
                        code: MODULE_VERSION,
                        ..
                    }
                )
            {
                // The module must open with its VERSION record; nothing declared a version.
                return Err(s.malformed(B, MODULE_VERSION));
            }
            match item {
                Item::EndBlock => return Ok(()),
                Item::End => return Err(s.malformed(B, 0)),
                Item::SubBlock(block) => self.sub_block(s, block, m)?,
                Item::Record { code, .. } => self.record(s, code, m)?,
            }
        }
    }

    fn sub_block(&mut self, s: &mut Stream<'a>, block: u32, m: &mut Meter) -> Result<(), Error> {
        match block {
            TYPE_BLOCK => {
                if self.bc_types.is_some() {
                    return Err(s.malformed(TYPE_BLOCK, 0));
                }
                let t = types::decode(s, &mut self.parts.types, &mut self.parts.symbols, m)?;
                self.bc_types = Some(t);
            }
            PARAMATTR_GROUP_BLOCK => {
                attrs::decode_groups(s, &mut self.groups, &mut self.parts.symbols, m)?
            }
            PARAMATTR_BLOCK => attrs::decode_lists(s, &mut self.groups, &mut self.attr_lists, m)?,
            CONSTANTS_BLOCK => self.constants(s, m)?,
            METADATA_BLOCK => self.metadata(s, m)?,
            METADATA_KIND_BLOCK => metadata::decode_kinds(s, &mut self.md_kinds, m)?,
            VALUE_SYMTAB_BLOCK => {
                if self.vst_bit.is_some() {
                    return Err(s.malformed(VALUE_SYMTAB_BLOCK, 0));
                }
                self.vst_bit = Some(s.entry_bit());
                if self.version == Some(MODULE_VERSION_1) {
                    // Read by the first pass.
                    s.skip_block(m)?
                } else {
                    self.symtab(s, m)?
                }
            }
            SYNC_SCOPE_NAMES_BLOCK => self.sync_scopes(s, m)?,
            FUNCTION_BLOCK => self.function_block(s, m)?,
            // Use-list order is an optimizer hint (see `function` for this deviation).
            OPERAND_BUNDLE_TAGS_BLOCK | USELIST_BLOCK => s.skip_block(m)?,
            _ => return Err(s.err(BitcodeError::UnknownBlock { block })),
        }
        Ok(())
    }

    fn record(&mut self, s: &Stream<'a>, code: u32, m: &mut Meter) -> Result<(), Error> {
        const B: u32 = MODULE_BLOCK;
        match code {
            MODULE_VERSION => {
                let v = s.op(B, code, 0)?;
                if self.version.is_some() {
                    return Err(s.malformed(B, code));
                }
                if v != MODULE_VERSION_1 && v != MODULE_VERSION_2 {
                    return Err(s.err(BitcodeError::UnsupportedBitcodeVersion { version: v }));
                }
                self.version = Some(v);
            }
            MODULE_TRIPLE => {
                let t = s.chars(m, B, code, 0)?;
                self.parts.triple = Some(self.parts.symbols.intern(&t, m)?);
            }
            MODULE_DATALAYOUT => {
                let dl = s.chars(m, B, code, 0)?;
                let dl =
                    std::str::from_utf8(&dl).map_err(|_| s.err(BitcodeError::BadDataLayout))?;
                self.parts.data_layout = DataLayout::parse(dl).map_err(|e| match e {
                    Error::InvalidInput(_) => s.err(BitcodeError::BadDataLayout),
                    other => other,
                })?;
            }
            MODULE_SOURCE_FILENAME => {
                let f = s.chars(m, B, code, 0)?;
                self.parts.source_filename = Some(self.parts.symbols.intern(&f, m)?);
            }
            MODULE_SECTIONNAME => {
                let name = s.chars(m, B, code, 0)?;
                // The two AIR ABI sections are stable `air.*` names.
                let section = match name.as_slice() {
                    b"air.fc_initializer" => Section::FcInitializer,
                    b"air.static_init" => Section::StaticInit,
                    _ => Section::Other(self.parts.symbols.intern(&name, m)?),
                };
                grow(m, &mut self.sections)?;
                self.sections.push(section);
            }
            MODULE_GLOBALVAR => self.global(s, m)?,
            MODULE_FUNCTION => self.function(s, m)?,
            MODULE_VSTOFFSET => {
                if self.vst_offset.is_some() {
                    return Err(s.malformed(B, code));
                }
                self.vst_offset = Some(s.op(B, code, 0)?);
            }
            MODULE_HASH => {}
            MODULE_GUIDLIST => self.guid_list(s, m)?,
            c if MODULE_UNSUPPORTED.contains(&c) => {
                return Err(s.unsupported(Feature::ModuleRecord { code }))
            }
            _ => return Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        }
        Ok(())
    }

    fn types(&self, s: &Stream<'_>) -> Result<&BitcodeTypes, Error> {
        self.bc_types
            .as_ref()
            .ok_or_else(|| s.err(BitcodeError::BadType))
    }

    /// The name of the global or function a `GLOBALVAR` or `FUNCTION` record declares, the index
    /// of its first field after the name, and the word offset of its block that a version-1
    /// `VST_FNENTRY` gives. Version 2 starts the record with the name's `(offset, size)` in the
    /// string table. Version 1 names the record's value id (the next module value) in the symbol
    /// table, or leaves the value unnamed, as version 2's empty name does.
    fn name(&mut self, s: &Stream<'_>, code: u32, m: &mut Meter) -> Result<Named, Error> {
        if self.version == Some(MODULE_VERSION_1) {
            let value = self.values.len() as u64;
            let (offset, bytes): (_, &[u8]) = match self.names.get(value) {
                Some((e, bytes)) => {
                    self.named += 1;
                    (e.offset, bytes)
                }
                None => (None, b""),
            };
            let name = self.parts.symbols.intern(bytes, m)?;
            return Ok(Named {
                name,
                base: 0,
                offset,
            });
        }
        let offset = s.op(MODULE_BLOCK, code, 0)?;
        let size = s.op(MODULE_BLOCK, code, 1)?;
        let bytes = self
            .strtab
            .and_then(|t| {
                let start = usize::try_from(offset).ok()?;
                let end = start.checked_add(usize::try_from(size).ok()?)?;
                t.get(start..end)
            })
            .ok_or_else(|| s.err(BitcodeError::BadStringTable))?;
        Ok(Named {
            name: self.parts.symbols.intern(bytes, m)?,
            base: 2,
            offset: None,
        })
    }

    fn section(&self, s: &Stream<'_>, code: u32, raw: u64) -> Result<Option<Section>, Error> {
        match raw.checked_sub(1) {
            None => Ok(None),
            Some(i) => usize::try_from(i)
                .ok()
                .and_then(|i| self.sections.get(i))
                .copied()
                .map(Some)
                .ok_or_else(|| s.malformed(MODULE_BLOCK, code)),
        }
    }

    fn push_value(
        &mut self,
        c: Constant,
        bc_type: Option<u32>,
        object_bc_type: u32,
        m: &mut Meter,
    ) -> Result<ConstId, Error> {
        let id = self.parts.consts.intern(c, m)?;
        grow(m, &mut self.values)?;
        self.values.push(ModuleValue {
            id: Some(id),
            bc_type,
            object_bc_type: Some(object_bc_type),
        });
        Ok(id)
    }

    /// `GLOBALVAR: [strtab offset, size (version 2), type, isconst|explicit<<1|addrspace<<2,
    /// initid, linkage, alignment, section, visibility, threadlocal, unnamed_addr,
    /// externally_initialized, ...]`. Fields are indexed from `type`.
    fn global(&mut self, s: &Stream<'_>, m: &mut Meter) -> Result<(), Error> {
        const C: u32 = MODULE_GLOBALVAR;
        let Named { name, base, offset } = self.name(s, C, m)?;
        if offset.is_some() {
            // `VST_FNENTRY` names a function.
            return Err(s.malformed(VALUE_SYMTAB_BLOCK, VST_FNENTRY));
        }
        let op = |i| s.op(MODULE_BLOCK, C, base + i);
        let ty_idx = op(0)?;
        let flags = op(1)?;
        let types = self.types(s)?;
        let bad_type = || s.err(BitcodeError::BadType);
        let (value_ty, space, bc_type, object_bc_type) = if flags & 2 != 0 {
            // The explicit-type spelling gives the global's value type.
            let object = u32::try_from(ty_idx).map_err(|_| bad_type())?;
            let ty = types.ty(ty_idx).ok_or_else(bad_type)?;
            let raw = u32::try_from(flags >> 2).map_err(|_| s.malformed(MODULE_BLOCK, C))?;
            let space = AddrSpace::from_raw(raw).map_err(|u| s.unsupported(u.feature))?;
            (ty, space, None, object)
        } else {
            // The older spelling gives the global's (typed) pointer type.
            let own = u32::try_from(ty_idx).map_err(|_| bad_type())?;
            let object = types.pointee(ty_idx).ok_or_else(bad_type)?;
            let pointee = types.ty(u64::from(object)).ok_or_else(bad_type)?;
            let space = types
                .ty(ty_idx)
                .and_then(|t| self.parts.types.ptr_space(t))
                .ok_or_else(bad_type)?;
            (pointee, space, Some(own), object)
        };
        let init = op(2)?;
        let linkage = linkage(op(3)?).ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
        let align = align(op(4)?).ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
        let section = self.section(s, C, op(5)?)?;
        let opt = |i: usize| s.ops().get(base + i).copied().unwrap_or(0);
        // Thread-local storage and comdats are rejected at decode.
        if opt(7) != 0 || opt(11) != 0 {
            return Err(s.unsupported(Feature::ModuleRecord { code: C }));
        }
        let global = Global {
            name,
            value_ty,
            space,
            constant: flags & 1 != 0,
            init: None,
            externally_initialized: opt(9) != 0,
            linkage,
            visibility: visibility(linkage, opt(6)),
            unnamed_addr: unnamed_addr(opt(8)),
            align,
            section,
        };
        let g = self.parts.globals.push(global, m)?;
        if let Some(v) = init.checked_sub(1) {
            grow(m, &mut self.pending_inits)?;
            self.pending_inits.push((g, v));
        }
        let ptr = self.parts.types.intern(Type::Ptr { space }, m)?;
        self.push_value(
            Constant {
                ty: ptr,
                value: Const::GlobalAddr(g),
            },
            bc_type,
            object_bc_type,
            m,
        )?;
        Ok(())
    }

    /// `FUNCTION: [strtab offset, size (version 2), type, callingconv, isproto, linkage,
    /// paramattrs, alignment, section, visibility, gc, unnamed_addr, prologuedata,
    /// dllstorageclass, comdat, prefixdata, personalityfn, preemption, addrspace, partition offset,
    /// partition size, prefalign]`. Fields are indexed from `type`.
    fn function(&mut self, s: &Stream<'_>, m: &mut Meter) -> Result<(), Error> {
        const C: u32 = MODULE_FUNCTION;
        let Named { name, base, offset } = self.name(s, C, m)?;
        let op = |i| s.op(MODULE_BLOCK, C, base + i);
        let opt = |i: usize| s.ops().get(base + i).copied().unwrap_or(0);
        let types = self.types(s)?;
        let record_ty = op(0)?;
        let record_idx = u32::try_from(record_ty).map_err(|_| s.err(BitcodeError::BadType))?;
        // The oldest spelling names a pointer to the function type: that pointer is the value's
        // own type, and its pointee the function type.
        let (own_bc_type, fn_bc_type) = match types.pointee(record_ty) {
            Some(p) => (Some(record_idx), p),
            None => (None, record_idx),
        };
        let ty_idx = u64::from(fn_bc_type);
        let ty = types
            .ty(ty_idx)
            .ok_or_else(|| s.err(BitcodeError::BadType))?;
        let arity = match self.parts.types.get(ty) {
            Some(Type::Function { params, .. }) => params.len(),
            _ => return Err(s.err(BitcodeError::BadType)),
        };
        let cconv = match op(1)? {
            CC_C => CallConv::C,
            CC_FAST => CallConv::Fast,
            other => {
                let cc = u32::try_from(other).unwrap_or(u32::MAX);
                return Err(s.unsupported(Feature::CallingConvention(cc)));
            }
        };
        let is_proto = op(2)? != 0;
        if is_proto && offset.is_some() {
            // `VST_FNENTRY` locates a body, which a declaration has none of.
            return Err(s.malformed(VALUE_SYMTAB_BLOCK, VST_FNENTRY));
        }
        let linkage = linkage(op(3)?).ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
        let attr_list = op(4)?;
        // LLVM 23's preferred alignment is a code-placement hint ("treated as a hint" by LangRef),
        // which a shader has no use for: checked like `align` and dropped.
        if let Some(&raw) = s.ops().get(base + 19) {
            align(raw).ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
        }
        let align = align(op(5)?).ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
        let section = self.section(s, C, op(6)?)?;
        // GC names, prologue/prefix data, comdats, and personality functions are rejected.
        if opt(8) != 0 || opt(10) != 0 || opt(12) != 0 || opt(13) != 0 || opt(14) != 0 {
            return Err(s.unsupported(Feature::ModuleRecord { code: C }));
        }
        let program = self.parts.data_layout.program_space;
        let raw_space = match s.ops().get(base + 16) {
            Some(&v) => u32::try_from(v).map_err(|_| s.malformed(MODULE_BLOCK, C))?,
            None => program,
        };
        if raw_space != program {
            return Err(s.unsupported(Feature::AddressSpace(raw_space)));
        }
        let space = AddrSpace::from_raw(raw_space).map_err(|u| s.unsupported(u.feature))?;
        let resolved = match attr_list.checked_sub(1) {
            None => None,
            Some(i) => {
                let list = usize::try_from(i)
                    .ok()
                    .and_then(|i| self.attr_lists.get(i))
                    .ok_or_else(|| s.malformed(MODULE_BLOCK, C))?;
                let params = types.params(ty_idx).unwrap_or(&[]);
                let pointee = |p: usize| params.get(p).and_then(|&t| types.pointee(u64::from(t)));
                let r = list.resolve(types, arity, pointee, self.opts.enum_attrs, m)?;
                Some(r.ok_or_else(|| s.malformed(MODULE_BLOCK, C))?)
            }
        };
        let (attrs, ret_attrs, mut param_attrs, enums) = match resolved {
            Some(r) => (r.func, r.ret, r.params, r.enums),
            None => Default::default(),
        };
        let mut params = Vec::new();
        m.reserve(&mut params, arity)?;
        let missing = arity.saturating_sub(param_attrs.len());
        m.reserve(&mut param_attrs, missing)?;
        param_attrs.resize(arity, Default::default());
        params.extend(
            param_attrs
                .into_iter()
                .map(|attrs| Param { attrs, name: None }),
        );
        let f = self.parts.funcs.push(
            Function {
                name,
                ty,
                linkage,
                visibility: visibility(linkage, opt(7)),
                unnamed_addr: unnamed_addr(opt(9)),
                cconv,
                attrs,
                ret_attrs,
                params: params.into_boxed_slice(),
                body: None,
                section,
                align,
            },
            m,
        )?;
        if !enums.is_empty() {
            m.charge(std::mem::size_of_val(&enums))?;
            self.enum_attrs.set_func(f, enums);
        }
        if let Some(words) = offset {
            grow(m, &mut self.fn_entries)?;
            self.fn_entries.push((f, words));
        }
        if !is_proto {
            grow(m, &mut self.defined)?;
            self.defined.push(f);
        } else if let Some(class) = self
            .parts
            .symbols
            .get(name)
            .and_then(|n| intrinsics::classify(n, name, ty, &self.parts.types))
        {
            // Classified once per declaration; calls only read their immediates.
            grow(m, &mut self.intrinsics)?;
            self.intrinsics.push((f, class));
        }
        let ptr = self.parts.types.intern(Type::Ptr { space }, m)?;
        self.push_value(
            Constant {
                ty: ptr,
                value: Const::FuncAddr(f),
            },
            own_bc_type,
            fn_bc_type,
            m,
        )?;
        Ok(())
    }

    /// A `FUNCTION` block: the body of the next defined function in record order (the order
    /// LLVM writes bodies in; `finish` checks it against the symbol table's offsets).
    fn function_block(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        let bit = s.entry_bit();
        let index = self.function_blocks as usize;
        let func = *self
            .defined
            .get(index)
            .ok_or_else(|| s.malformed(FUNCTION_BLOCK, 0))?;
        self.function_blocks = self.function_blocks.saturating_add(1);
        grow(m, &mut self.block_bits)?;
        self.block_bits.push(bit);
        let bc_types = self
            .bc_types
            .as_ref()
            .ok_or_else(|| s.err(BitcodeError::BadType))?;
        let ModuleParts {
            symbols,
            types,
            consts,
            funcs,
            metadata,
            data_layout,
            ..
        } = &mut self.parts;
        let mut cx = FnCx {
            bc_types,
            types,
            consts,
            symbols,
            module_md: metadata,
            funcs,
            values: &self.values,
            attr_lists: &self.attr_lists,
            md_kinds: self.md_kinds,
            sync_scopes: &self.sync_scopes,
            intrinsics: &self.intrinsics,
            alloca_space: data_layout.alloca_space,
            inst_total: &mut self.inst_total,
            keep_enums: self.opts.enum_attrs,
            keep_marker_operands: self.opts.drop_unused_markers,
        };
        let decoded = function::decode(s, &mut cx, func, m)?;
        for (call, enums) in decoded.call_enums {
            self.enum_attrs.set_call(func, call, enums);
        }
        for c in decoded.marker_consts {
            grow(m, &mut self.marker_consts)?;
            self.marker_consts.push(c);
        }
        let f = self
            .parts
            .funcs
            .get_mut(func)
            .ok_or_else(|| s.err(BitcodeError::BadValueRef))?;
        for (p, name) in f.params.iter_mut().zip(decoded.param_names) {
            p.name = name;
        }
        f.body = Some(decoded.body);
        Ok(())
    }

    /// Cross-check the function blocks against the defined functions and the module symbol
    /// table: one block per defined function, and, when the module declares `VSTOFFSET`, the
    /// symbol table at that word offset naming each defined function's block by its word
    /// offset. LLVM associates bodies with functions by those offsets and writes them in record
    /// order; a module that orders them otherwise is refused rather than misread.
    ///
    /// Version 2 gives every defined function an offset. Version 1 gives one to each named
    /// defined function when it declares `VSTOFFSET` (LLVM 3.8 on) and none before; a function it
    /// gives none is found by record order alone, as LLVM's reader finds it.
    fn check_bodies(&self, m: &mut Meter) -> Result<(), Error> {
        let legacy = self.version == Some(MODULE_VERSION_1);
        let at = |kind| -> Error {
            InputError::new(InputErrorKind::Bitcode(kind), Location::module()).into()
        };
        if self.function_blocks as usize != self.defined.len() {
            return Err(at(BitcodeError::MalformedRecord {
                block: FUNCTION_BLOCK,
                code: 0,
            }));
        }
        let vst_malformed = || {
            at(BitcodeError::MalformedRecord {
                block: MODULE_BLOCK,
                code: MODULE_VSTOFFSET,
            })
        };
        match (self.vst_offset, self.vst_bit) {
            (None, None) => {
                return if self.fn_entries.is_empty() {
                    Ok(())
                } else {
                    Err(vst_malformed())
                }
            }
            (None, Some(_)) if legacy => {
                return if self.fn_entries.is_empty() {
                    Ok(())
                } else {
                    Err(vst_malformed())
                }
            }
            (Some(words), Some(bit)) if words.checked_mul(32) == Some(bit) => {}
            _ => return Err(vst_malformed()),
        }
        let entry_malformed = || {
            at(BitcodeError::MalformedRecord {
                block: VALUE_SYMTAB_BLOCK,
                code: VST_FNENTRY,
            })
        };
        let entries_fit = if legacy {
            self.fn_entries.len() <= self.defined.len()
        } else {
            self.fn_entries.len() == self.defined.len()
        };
        if !entries_fit {
            return Err(entry_malformed());
        }
        // Sorted copies keep the check O(n log n) in the number of functions.
        let mut entries = Vec::new();
        m.reserve(&mut entries, self.fn_entries.len())?;
        entries.extend_from_slice(&self.fn_entries);
        entries.sort_unstable();
        if entries
            .windows(2)
            .any(|w| matches!(w, [a, b] if a.0 == b.0))
        {
            return Err(entry_malformed());
        }
        // Blocks are recorded in stream order, so their positions increase.
        let blocks = &self.block_bits;
        let mut matched = 0;
        for (&f, &bit) in self.defined.iter().zip(blocks) {
            m.tick(1)?;
            let words = entries
                .binary_search_by_key(&f, |&(g, _)| g)
                .ok()
                .and_then(|i| entries.get(i))
                .map(|&(_, w)| w);
            let words = match words {
                Some(w) => w,
                None if legacy => continue,
                None => return Err(entry_malformed()),
            };
            matched += 1;
            match words.checked_mul(32) {
                Some(b) if b == bit => {}
                Some(b) if blocks.binary_search(&b).is_ok() => {
                    return Err(InputError::new(
                        InputErrorKind::Unsupported(Feature::BitcodeRecord {
                            block: VALUE_SYMTAB_BLOCK,
                            code: VST_FNENTRY,
                        }),
                        Location::module(),
                    )
                    .into())
                }
                _ => return Err(entry_malformed()),
            }
        }
        if matched != entries.len() {
            return Err(entry_malformed());
        }
        Ok(())
    }

    /// `GUIDLIST: [n x (high 32 bits, low 32 bits)]` (LLVM 23): the GUID LLVM assigned to each
    /// global value, by module value id, 0 where none is. GUIDs identify symbols to ThinLTO and
    /// profile summaries and carry no program semantics, so they are checked and dropped: the
    /// operands pair 32-bit halves, later records continue the numbering (as LLVM's reader does),
    /// and `finish` checks that every value id given a GUID names a global or function.
    fn guid_list(&mut self, s: &Stream<'_>, m: &mut Meter) -> Result<(), Error> {
        let ops = s.ops();
        if !ops.len().is_multiple_of(2) || ops.iter().any(|&w| w > u64::from(u32::MAX)) {
            return Err(s.malformed(MODULE_BLOCK, MODULE_GUIDLIST));
        }
        for pair in ops.as_chunks::<2>().0 {
            m.tick(1)?;
            if pair.iter().any(|&w| w != 0) {
                grow(m, &mut self.guid_values)?;
                self.guid_values.push(self.guid_entries);
            }
            self.guid_entries = self.guid_entries.saturating_add(1);
        }
        Ok(())
    }

    /// Every value id a `GUIDLIST` gives a GUID names a global or a function.
    fn check_guids(&self, m: &mut Meter) -> Result<(), Error> {
        for &v in &self.guid_values {
            m.tick(1)?;
            let global = usize::try_from(v)
                .ok()
                .and_then(|i| self.values.get(i))
                .and_then(|v| self.parts.consts.get(v.id?))
                .is_some_and(|c| matches!(c.value, Const::GlobalAddr(_) | Const::FuncAddr(_)));
            if !global {
                return Err(InputError::new(
                    InputErrorKind::Bitcode(BitcodeError::MalformedRecord {
                        block: MODULE_BLOCK,
                        code: MODULE_GUIDLIST,
                    }),
                    Location::module(),
                )
                .into());
            }
        }
        Ok(())
    }

    fn constants(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        let Some(bc_types) = self.bc_types.as_ref() else {
            return Err(s.err(BitcodeError::BadType));
        };
        let values = &self.values;
        let prior = |id: u64| -> Option<ConstId> {
            usize::try_from(id)
                .ok()
                .and_then(|i| values.get(i))
                .and_then(|v| v.id)
        };
        let mut cx = ConstCx {
            bc_types,
            types: &self.parts.types,
            consts: &mut self.parts.consts,
        };
        let base = values.len() as u64;
        let decoded = constants::decode(s, &mut cx, base, &prior, m)?;
        m.reserve(&mut self.values, decoded.len())?;
        self.values.extend(decoded.into_iter().map(|c| ModuleValue {
            id: Some(c.id),
            bc_type: Some(c.bc_type),
            object_bc_type: None,
        }));
        Ok(())
    }

    fn metadata(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        let ModuleParts {
            symbols,
            consts,
            metadata,
            named_md,
            ..
        } = &mut self.parts;
        let values = &self.values;
        let bc_types = self.bc_types.as_ref();
        let consts: &crate::ir::ConstTable = consts;
        let value = |ty: u64, id: u64| -> MdValue {
            let found = (|| {
                let v = values.get(usize::try_from(id).ok()?)?;
                let want = bc_types?.ty(ty)?;
                let id = v.id?;
                (consts.ty(id)? == want).then_some(id)
            })();
            found.map_or(MdValue::Bad, MdValue::Const)
        };
        let global_object = |id: u64| -> bool {
            usize::try_from(id)
                .ok()
                .and_then(|i| values.get(i))
                .and_then(|v| consts.get(v.id?))
                .is_some_and(|c| matches!(c.value, Const::GlobalAddr(_) | Const::FuncAddr(_)))
        };
        let mut cx = MdCx {
            symbols,
            metadata,
            base: 0,
            named: named_md,
        };
        metadata::decode(s, &mut cx, &value, &global_object, m)
    }

    /// Version 2's module-level `VALUE_SYMTAB`: with a string table it holds only function
    /// offsets.
    fn symtab(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        const B: u32 = VALUE_SYMTAB_BLOCK;
        let values = &self.values;
        let consts = &self.parts.consts;
        let entries = &mut self.fn_entries;
        s.records(m, B, |s, code, _, m| match code {
            VST_FNENTRY => {
                let v = s.op(B, code, 0)?;
                let offset = s.op(B, code, 1)?;
                let func = usize::try_from(v)
                    .ok()
                    .and_then(|i| values.get(i))
                    .and_then(|v| match consts.get(v.id?)?.value {
                        Const::FuncAddr(f) => Some(f),
                        _ => None,
                    })
                    .ok_or_else(|| s.err(BitcodeError::BadValueRef))?;
                grow(m, entries)?;
                entries.push((func, offset));
                Ok(())
            }
            // Names come from the string table in module version 2.
            VST_ENTRY | VST_BBENTRY => Err(s.malformed(B, code)),
            VST_COMBINED_ENTRY => Err(s.unsupported(Feature::BitcodeRecord { block: B, code })),
            _ => Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        })
    }

    fn sync_scopes(&mut self, s: &mut Stream<'a>, m: &mut Meter) -> Result<(), Error> {
        const B: u32 = SYNC_SCOPE_NAMES_BLOCK;
        let scopes = &mut self.sync_scopes;
        s.records(m, B, |s, code, _, m| {
            if code != SYNC_SCOPE_NAME {
                return Err(s.err(BitcodeError::UnknownRecord { block: B, code }));
            }
            let name = s.chars(m, B, code, 0)?;
            // LLVM's two built-in scope names; any other scope is refused where it is used.
            let scope = match name.as_slice() {
                b"singlethread" => Some(SyncScope::SingleThread),
                b"" => Some(SyncScope::System),
                _ => None,
            };
            grow(m, scopes)?;
            scopes.push(scope);
            Ok(())
        })
    }

    fn finish(mut self, m: &mut Meter) -> Result<ModuleTables, Error> {
        if self.named != self.names.entries.len() {
            // A version-1 symbol-table entry that names no global or function.
            return Err(InputError::new(
                InputErrorKind::Bitcode(BitcodeError::MalformedRecord {
                    block: VALUE_SYMTAB_BLOCK,
                    code: VST_ENTRY,
                }),
                Location::module(),
            )
            .into());
        }
        self.check_bodies(m)?;
        self.check_guids(m)?;
        let bad = || -> Error {
            InputError::new(
                InputErrorKind::Bitcode(BitcodeError::BadValueRef),
                Location::module(),
            )
            .into()
        };
        for &(g, v) in &self.pending_inits {
            m.tick(1)?;
            let c = usize::try_from(v)
                .ok()
                .and_then(|i| self.values.get(i))
                .and_then(|v| v.id)
                .ok_or_else(bad)?;
            let global = self.parts.globals.get_mut(g).ok_or_else(bad)?;
            global.init = Some(c);
        }
        let tables = ModuleTables {
            parts: self.parts,
            types: self.bc_types.unwrap_or_default(),
            values: self.values,
            attr_lists: self.attr_lists,
            md_kinds: self.md_kinds,
            sync_scopes: self.sync_scopes,
            defined: self.defined,
            function_blocks: self.function_blocks,
            fn_entries: self.fn_entries,
            enum_attrs: self.enum_attrs,
        };
        if self.opts.drop_unused_markers {
            markers::drop_unused(tables, &self.intrinsics, &self.marker_consts, m)
        } else {
            Ok(tables)
        }
    }
}

/// LLVM's bitcode linkage encoding (`getDecodedLinkage`), including the obsolete values it still
/// maps. `None` for a value outside the table.
fn linkage(v: u64) -> Option<Linkage> {
    Some(match v {
        0 | 5 | 6 | 15 => Linkage::External,
        2 => Linkage::Appending,
        3 => Linkage::Internal,
        7 => Linkage::ExternalWeak,
        8 => Linkage::Common,
        9 | 13 | 14 => Linkage::Private,
        12 => Linkage::AvailableExternally,
        1 | 16 => Linkage::WeakAny,
        10 | 17 => Linkage::WeakOdr,
        4 | 18 => Linkage::LinkOnceAny,
        11 | 19 => Linkage::LinkOnceOdr,
        _ => return None,
    })
}

/// LLVM's visibility encoding as its reader applies it (`getDecodedVisibility`): an unknown value
/// is the default, and a symbol with local linkage keeps the default whatever the field says.
fn visibility(linkage: Linkage, v: u64) -> Visibility {
    if matches!(linkage, Linkage::Internal | Linkage::Private) {
        return Visibility::Default;
    }
    match v {
        1 => Visibility::Hidden,
        2 => Visibility::Protected,
        _ => Visibility::Default,
    }
}

/// LLVM's `unnamed_addr` encoding as its reader applies it (`getDecodedUnnamedAddrType`): an
/// unknown value is a significant address.
fn unnamed_addr(v: u64) -> UnnamedAddr {
    match v {
        1 => UnnamedAddr::Global,
        2 => UnnamedAddr::Local,
        _ => UnnamedAddr::Significant,
    }
}

/// A `log2 + 1` alignment field (0 = none). `None` for an out-of-range value.
fn align(v: u64) -> Option<Option<Align>> {
    match v.checked_sub(1) {
        None => Some(None),
        Some(log2) => Align::from_log2(u8::try_from(log2).ok()?).map(Some),
    }
}
