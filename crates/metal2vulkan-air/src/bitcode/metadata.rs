//! `METADATA` and `METADATA_KIND` blocks.
//!
//! Metadata ids follow record order: every string of a `METADATA_STRINGS` record, then every
//! node, value, and debug-info record, gets the next id. Nodes may refer to later nodes, so
//! references are checked once the block ends. A function block may hold its own `METADATA`
//! block (LLVM writes metadata reachable from only one function there): its ids continue after
//! the module's, it is decoded into a separate arena with that base, and it is dropped with the
//! function once its attachments are typed. Debug-info records are kept as [`Md::Debug`]
//! leaves only so numbering stays aligned with the bitcode. Instruction attachment kinds are
//! resolved from the two stable LLVM kind names the IR keeps (`llvm.loop`, `range`).
//!
//! The module's `METADATA` block may also hold the records older LLVM writers wrote there (Apple's
//! AIR 1.8 among them), each read as LLVM's `MetadataLoader` reads it
//! (`llvm/lib/Bitcode/Reader/MetadataLoader.cpp` at `llvmorg-23.1.2`):
//!
//! - `STRING_OLD: [values]` (line 2424): one `MDString` of the operands as bytes, the next id.
//!   A string beginning `llvm.vectorizer.` also makes LLVM rename such keys in `!llvm.loop`
//!   attachments (`llvm/lib/IR/AutoUpgrade.cpp:6979-7026`); the IR keeps neither spelling of
//!   those keys (only `llvm.loop.mustprogress` and `llvm.loop.unroll.*`), so its hints are the
//!   same.
//! - `KIND: [n, name]` (line 2457): an attachment kind, read as the `METADATA_KIND` block's record
//!   is (line 2631); it takes no id.
//! - `OLD_NODE: [n x (type, value)]` (line 1401): a uniqued node, the next id. An operand of
//!   metadata type names metadata by id, one of `void` type is null, and any other is the
//!   module value of that type and id, as `ValueAsMetadata`. LLVM gives such a value no id; the
//!   arena keeps it as an [`Md::Value`] entry, one per value, appended once the block's records
//!   are checked, so every id LLVM numbers after the block is that many entries further on
//!   ([`MdIds`]).
//!
//! In a function's block, `KIND` and `OLD_NODE` are refused (`Unsupported`), as is
//! `OLD_FN_NODE` (the function-local value node) anywhere, as is an `OLD_NODE` value in a module
//! block after one whose old nodes held values: no writer this reader has seen writes them.
//!
//! Deviation (which lists only `INDEX_OFFSET`, `INDEX`, and debug info as skippable here):
//! `GLOBAL_DECL_ATTACHMENT` records are checked and then dropped, because the IR has no global or
//! declaration attachments and LLVM writes them for any module whose globals carry metadata (e.g.
//! debug info, which the spec already tolerates). The check requires `[value id, (kind id, node
//! id)+]`, the value to be a module global or function, and each attached id to name a metadata
//! node once the block ends. Kind ids are not checked: the IR keeps no attachment kinds on globals.

use std::collections::{HashMap, HashSet};

use super::codes::*;
use super::stream::Stream;
use super::types::grow;
use crate::budget::Meter;
use crate::error::BitcodeError;
use crate::error::{Error, Feature, Limit};
use crate::ir::{Arena, ConstId, Md, MdId, NamedMd, Symbols};

/// The attachment kinds the IR keeps, by their module-local kind id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MdKinds {
    /// The id of `llvm.loop` (typed as `LoopHints`).
    pub loop_hints: Option<u64>,
    /// The id of `range` (typed as `RangeMd`).
    pub range: Option<u64>,
}

/// The module's attachment kinds as `KIND` records state them, wherever they sit.
#[derive(Debug, Default)]
pub(crate) struct KindTable {
    pub kinds: MdKinds,
    /// Every kind id stated so far: LLVM refuses a second record for one.
    seen: HashSet<u64>,
}

/// How LLVM's metadata ids map to the arena's: one to one, except that from `first` on each id is
/// `extra` entries further on, past the [`Md::Value`] entries the arena keeps for the module's
/// old-node values, which LLVM numbers none of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MdIds {
    first: u32,
    extra: u32,
}

impl MdIds {
    /// The arena id of LLVM's metadata id `raw`.
    pub(crate) fn arena(self, raw: u64) -> Option<MdId> {
        let raw = u32::try_from(raw).ok()?;
        let id = if raw < self.first {
            raw
        } else {
            raw.checked_add(self.extra)?
        };
        Some(MdId::new(id))
    }
}

/// What the metadata decoder writes.
pub(crate) struct MdCx<'c> {
    pub symbols: &'c mut Symbols,
    /// The arena this block appends to; its entry `i` has metadata id `base + i`.
    pub metadata: &'c mut Arena<MdId, Md>,
    /// The id of `metadata`'s first entry: 0 for the module, the module's count in a function.
    /// Ids below it name module metadata, which is complete when a function block is read.
    pub base: u32,
    pub named: &'c mut Vec<NamedMd>,
    /// LLVM's ids to the arena's; a module block that keeps old-node values updates it.
    pub ids: MdIds,
    /// The module's kinds, for a module block; `None` in a function's.
    pub kinds: Option<&'c mut KindTable>,
}

/// What a `METADATA_VALUE` operand resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MdValue {
    /// A constant of the requested type.
    Const(ConstId),
    /// A function-local argument or instruction (`LocalAsMetadata`), which the IR does not model.
    Local,
    /// No value of that type.
    Bad,
}

/// What an `OLD_NODE` operand's bitcode type makes of its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandType {
    /// `metadata`: the value is a metadata id.
    Metadata,
    /// `void`: the operand is null.
    Void,
    /// Any other type: the value is a value id of that type.
    Value,
}

/// Decode a `METADATA` block that was just entered. `value(ty, id)` resolves a
/// `METADATA_VALUE` operand (bitcode type index, value id); `operand_type(ty)` classifies an
/// `OLD_NODE` operand's bitcode type (`None`: no such type); `global_object(id)` says whether
/// module value `id` is a global or a function.
pub(crate) fn decode(
    s: &mut Stream<'_>,
    cx: &mut MdCx<'_>,
    value: &dyn Fn(u64, u64) -> MdValue,
    operand_type: &dyn Fn(u64) -> Option<OperandType>,
    global_object: &dyn Fn(u64) -> bool,
    m: &mut Meter,
) -> Result<(), Error> {
    const B: u32 = METADATA_BLOCK;
    let first = cx.metadata.len();
    let first_named = cx.named.len();
    let mut pending_name: Option<crate::ir::Sym> = None;
    // Nodes named by `GLOBAL_DECL_ATTACHMENT` records, checked once the block is complete.
    let mut attached: Vec<MdId> = Vec::new();
    // `OLD_NODE` value operands: (the node's arena id, the operand's position, the value), held
    // as null until the block's records are checked.
    let mut old_values: Vec<(MdId, usize, ConstId)> = Vec::new();
    let ids = cx.ids;
    let md_id = |s: &Stream<'_>, code: u32, raw: u64| -> Result<MdId, Error> {
        ids.arena(raw).ok_or_else(|| s.malformed(B, code))
    };
    s.records(m, B, |s, code, blob, m| {
        if pending_name.is_some() && code != METADATA_NAMED_NODE {
            return Err(s.malformed(B, code));
        }
        match code {
            METADATA_STRINGS => strings(s, cx, blob, m),
            METADATA_STRING_OLD => {
                let chars = s.chars(m, B, code, 0)?;
                let sym = cx.symbols.intern(&chars, m)?;
                push(s, cx, Md::String(sym), m)
            }
            METADATA_KIND => match cx.kinds.as_deref_mut() {
                Some(kinds) => kind(s, B, kinds, m),
                None => Err(s.unsupported(Feature::BitcodeRecord { block: B, code })),
            },
            METADATA_OLD_NODE => {
                if cx.kinds.is_none() {
                    return Err(s.unsupported(Feature::BitcodeRecord { block: B, code }));
                }
                let ops = s.ops();
                if !ops.len().is_multiple_of(2) {
                    return Err(s.malformed(B, code));
                }
                let node = MdId::new(
                    cx.base
                        .checked_add(
                            u32::try_from(cx.metadata.len()).map_err(|_| s.malformed(B, code))?,
                        )
                        .ok_or_else(|| s.malformed(B, code))?,
                );
                let mut refs = Vec::new();
                m.reserve(&mut refs, ops.len() / 2)?;
                for (pos, &[ty, v]) in ops.as_chunks::<2>().0.iter().enumerate() {
                    m.tick(1)?;
                    refs.push(
                        match operand_type(ty).ok_or_else(|| s.err(BitcodeError::BadType))? {
                            OperandType::Metadata => Some(md_id(s, code, v)?),
                            OperandType::Void => None,
                            OperandType::Value => {
                                let c = match value(ty, v) {
                                    MdValue::Const(c) => c,
                                    MdValue::Local => {
                                        return Err(s.unsupported(Feature::BitcodeRecord {
                                            block: B,
                                            code,
                                        }))
                                    }
                                    MdValue::Bad => return Err(s.err(BitcodeError::BadValueRef)),
                                };
                                // Kept only once per module (see the module docs).
                                if ids.extra != 0 {
                                    return Err(
                                        s.unsupported(Feature::BitcodeRecord { block: B, code })
                                    );
                                }
                                grow(m, &mut old_values)?;
                                old_values.push((node, pos, c));
                                None
                            }
                        },
                    );
                }
                push(
                    s,
                    cx,
                    Md::Node {
                        distinct: false,
                        ops: refs.into_boxed_slice(),
                    },
                    m,
                )
            }
            METADATA_NAME => {
                let name = s.chars(m, B, code, 0)?;
                pending_name = Some(cx.symbols.intern(&name, m)?);
                Ok(())
            }
            METADATA_NAMED_NODE => {
                let name = pending_name.take().ok_or_else(|| s.malformed(B, code))?;
                let ops = s.ops();
                let mut nodes = Vec::new();
                m.reserve(&mut nodes, ops.len())?;
                for &op in ops {
                    nodes.push(md_id(s, code, op)?);
                }
                grow(m, cx.named)?;
                cx.named.push(NamedMd {
                    name,
                    ops: nodes.into_boxed_slice(),
                });
                Ok(())
            }
            METADATA_NODE | METADATA_DISTINCT_NODE => {
                let ops = s.ops();
                let mut refs = Vec::new();
                m.reserve(&mut refs, ops.len())?;
                for &op in ops {
                    // Operand ids are `id + 1`; 0 is null.
                    refs.push(match op.checked_sub(1) {
                        None => None,
                        Some(id) => Some(md_id(s, code, id)?),
                    });
                }
                push(
                    s,
                    cx,
                    Md::Node {
                        distinct: code == METADATA_DISTINCT_NODE,
                        ops: refs.into_boxed_slice(),
                    },
                    m,
                )
            }
            METADATA_VALUE => {
                let ty = s.op(B, code, 0)?;
                let v = s.op(B, code, 1)?;
                match value(ty, v) {
                    MdValue::Const(c) => push(s, cx, Md::Value(c), m),
                    MdValue::Local => Err(s.unsupported(Feature::BitcodeRecord { block: B, code })),
                    MdValue::Bad => Err(s.err(BitcodeError::BadValueRef)),
                }
            }
            c if METADATA_DEBUG.contains(&c) => push(s, cx, Md::Debug, m),
            // Lazy-loading indexes carry nothing the IR keeps.
            METADATA_INDEX_OFFSET | METADATA_INDEX => Ok(()),
            // Global attachments are checked and dropped (see the module docs).
            // Only the module-level block attaches metadata to globals.
            METADATA_GLOBAL_DECL_ATTACHMENT if cx.base != 0 => Err(s.malformed(B, code)),
            METADATA_GLOBAL_DECL_ATTACHMENT => {
                let ops = s.ops();
                let (&v, pairs) = ops.split_first().ok_or_else(|| s.malformed(B, code))?;
                if pairs.is_empty() || pairs.len() % 2 != 0 {
                    return Err(s.malformed(B, code));
                }
                if !global_object(v) {
                    return Err(s.err(BitcodeError::BadValueRef));
                }
                m.reserve(&mut attached, pairs.len() / 2)?;
                for &node in pairs.iter().skip(1).step_by(2) {
                    attached.push(md_id(s, code, node)?);
                }
                Ok(())
            }
            c if METADATA_UNSUPPORTED.contains(&c) => {
                Err(s.unsupported(Feature::BitcodeRecord { block: B, code }))
            }
            _ => Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        }
    })?;
    if pending_name.is_some() {
        return Err(s.err(BitcodeError::BadMetadata));
    }
    // Every reference must name metadata that exists once the block is complete.
    let len = cx.base as usize + cx.metadata.len();
    let in_range = |id: &MdId| (id.raw() as usize) < len;
    for md in cx.metadata.values().skip(first) {
        m.tick(1)?;
        if let Md::Node { ops, .. } = md {
            if !ops.iter().flatten().all(in_range) {
                return Err(s.err(BitcodeError::BadMetadata));
            }
        }
    }
    for n in cx.named.iter().skip(first_named) {
        if !n.ops.iter().all(in_range) {
            return Err(s.err(BitcodeError::BadMetadata));
        }
    }
    for &id in &attached {
        m.tick(1)?;
        if !matches!(cx.metadata.get(id), Some(Md::Node { .. })) {
            return Err(s.err(BitcodeError::BadMetadata));
        }
    }
    keep_old_values(s, cx, &old_values, m)
}

/// Append the `OLD_NODE` value operands of the block just checked as [`Md::Value`] entries, one
/// per value (`ValueAsMetadata::get` is one per value), fill them into their nodes, and move every
/// id LLVM numbers from here on past them.
fn keep_old_values(
    s: &Stream<'_>,
    cx: &mut MdCx<'_>,
    old_values: &[(MdId, usize, ConstId)],
    m: &mut Meter,
) -> Result<(), Error> {
    if old_values.is_empty() {
        return Ok(());
    }
    let numbered = cx.base as usize + cx.metadata.len();
    let mut entries: HashMap<ConstId, MdId> = HashMap::new();
    for &(node, pos, c) in old_values {
        m.tick(1)?;
        let id = match entries.get(&c) {
            Some(&id) => id,
            None => {
                let id = MdId::new(
                    u32::try_from(cx.base as usize + cx.metadata.len())
                        .map_err(|_| Error::LimitExceeded(Limit::MetadataNodes))?,
                );
                push(s, cx, Md::Value(c), m)?;
                m.charge(std::mem::size_of::<(ConstId, MdId)>().saturating_mul(2))?;
                entries.insert(c, id);
                id
            }
        };
        let op = match cx.metadata.get_mut(MdId::new(node.raw() - cx.base)) {
            Some(Md::Node { ops, .. }) => ops.get_mut(pos),
            _ => None,
        };
        *op.ok_or_else(|| s.err(BitcodeError::BadMetadata))? = Some(id);
    }
    let first = u32::try_from(numbered).map_err(|_| Error::LimitExceeded(Limit::MetadataNodes))?;
    let extra =
        u32::try_from(entries.len()).map_err(|_| Error::LimitExceeded(Limit::MetadataNodes))?;
    cx.ids = MdIds { first, extra };
    Ok(())
}

/// `KIND: [n, name]`, in the `METADATA_KIND` block or, before LLVM 4.0, the `METADATA` block:
/// at least the id and one character, and each id stated once (`MetadataLoader.cpp:2631-2643` at
/// `llvmorg-23.1.2`).
fn kind(s: &Stream<'_>, block: u32, table: &mut KindTable, m: &mut Meter) -> Result<(), Error> {
    let code = METADATA_KIND;
    if s.ops().len() < 2 {
        return Err(s.malformed(block, code));
    }
    let id = s.op(block, code, 0)?;
    let name = s.chars(m, block, code, 1)?;
    m.charge(std::mem::size_of::<u64>().saturating_mul(2))?;
    if !table.seen.insert(id) {
        return Err(s.malformed(block, code));
    }
    // Stable LLVM attachment-kind names (ABI, not guest identifiers).
    match name.as_slice() {
        b"llvm.loop" => table.kinds.loop_hints = Some(id),
        b"range" => table.kinds.range = Some(id),
        _ => {}
    }
    Ok(())
}

fn push(s: &Stream<'_>, cx: &mut MdCx<'_>, md: Md, m: &mut Meter) -> Result<(), Error> {
    m.cap(
        Limit::MetadataNodes,
        u64::from(cx.base) + cx.metadata.len() as u64 + 1,
    )?;
    cx.metadata.push(md, m).map(|_| ()).map_err(|e| match e {
        Error::Internal(_) => s.err(BitcodeError::BadMetadata),
        other => other,
    })
}

/// `METADATA_STRINGS [count, offset]` with a blob of VBR6 lengths (a nested bitstream) followed,
/// at `offset`, by the concatenated characters.
fn strings(
    s: &Stream<'_>,
    cx: &mut MdCx<'_>,
    blob: Option<&[u8]>,
    m: &mut Meter,
) -> Result<(), Error> {
    const C: u32 = METADATA_STRINGS;
    let bad = || s.malformed(METADATA_BLOCK, C);
    let count = s.op(METADATA_BLOCK, C, 0)?;
    let offset = usize::try_from(s.op(METADATA_BLOCK, C, 1)?).map_err(|_| bad())?;
    let blob = blob.ok_or_else(bad)?;
    let lengths = blob.get(..offset).ok_or_else(bad)?;
    let mut chars = blob.get(offset..).ok_or_else(bad)?;
    // Each length takes at least 6 bits, so the count is bounded by the lengths' size.
    if count > (lengths.len() as u64).saturating_mul(8) / 6 {
        return Err(bad());
    }
    m.cap(
        Limit::MetadataNodes,
        (u64::from(cx.base) + cx.metadata.len() as u64).saturating_add(count),
    )?;
    let mut bits = Vbr6 {
        bytes: lengths,
        pos: 0,
    };
    for _ in 0..count {
        m.tick(1)?;
        let len = bits.next().ok_or_else(bad)?;
        let len = usize::try_from(len).map_err(|_| bad())?;
        if len > chars.len() {
            return Err(bad());
        }
        let (head, rest) = chars.split_at(len);
        chars = rest;
        let sym = cx.symbols.intern(head, m)?;
        push(s, cx, Md::String(sym), m)?;
    }
    Ok(())
}

/// A VBR6 reader over a little-endian, LSB-first bit sequence.
struct Vbr6<'b> {
    bytes: &'b [u8],
    pos: u64,
}

impl Vbr6<'_> {
    fn bit(&mut self) -> Option<u64> {
        let byte = self.bytes.get(usize::try_from(self.pos / 8).ok()?)?;
        let b = (byte >> (self.pos % 8)) & 1;
        self.pos += 1;
        Some(u64::from(b))
    }

    fn chunk(&mut self) -> Option<u64> {
        let mut v = 0;
        for i in 0..6 {
            v |= self.bit()? << i;
        }
        Some(v)
    }

    fn next(&mut self) -> Option<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let chunk = self.chunk()?;
            let payload = chunk & 0x1F;
            let part = payload.checked_shl(shift)?;
            if part >> shift != payload {
                return None;
            }
            value |= part;
            if chunk & 0x20 == 0 {
                return Some(value);
            }
            shift += 5;
        }
    }
}

/// Decode a `METADATA_KIND` block that was just entered.
pub(crate) fn decode_kinds(
    s: &mut Stream<'_>,
    table: &mut KindTable,
    m: &mut Meter,
) -> Result<(), Error> {
    const B: u32 = METADATA_KIND_BLOCK;
    s.records(m, B, |s, code, _, m| {
        if code != METADATA_KIND {
            return Err(s.err(BitcodeError::UnknownRecord { block: B, code }));
        }
        kind(s, B, table, m)
    })
}
