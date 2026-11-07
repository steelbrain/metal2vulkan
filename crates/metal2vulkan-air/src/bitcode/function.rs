//! `FUNCTION` blocks: one function body per block.
//!
//! # Value numbering
//!
//! A function continues the module's value numbering: its arguments, then the constants of its
//! local `CONSTANTS` block, then one value per non-void instruction in record order. Void
//! instructions (stores, fences, terminators, void calls) take no value number, but every
//! instruction takes the next [`InstId`], so an [`InstId`] is its record index less the dropped
//! markers before it (see *Calls*). Module version 2
//! encodes most operands relative to the next value number (32-bit wrapping, as LLVM's reader
//! does), PHI operands as signed relative values, and switch case values and alloca sizes as
//! absolute ids.
//!
//! # Forward references
//!
//! An operand may name an instruction value later in the function (PHIs in loops, and any use in
//! a block laid out before its definition). Operands whose type the record does not fix carry an
//! explicit type then. A forward reference is decoded to a placeholder
//! `Value::Inst(InstId(u32::MAX - k))` for its entry `k` in a side table of `(value id, type)`;
//! placeholders lie above every real instruction id and are replaced once the function's last
//! record is read. A reference that is never defined, or whose definition has a different type,
//! is `BadValueRef`. No placeholder survives into the returned body.
//!
//! # Records
//!
//! Every record of the closed instruction set is decoded exactly as LLVM's reader lays it out.
//! Codes LLVM defines outside the set are `Unsupported(Instruction { code })`, as are spellings
//! the IR cannot represent: typed-pointer forms without an explicit type (pre-3.7 writers),
//! vector GEPs, `inalloca`/`swifterror` allocas, label or metadata call arguments, and calling
//! conventions other than C and fast. `DEBUG_LOC`, `DEBUG_LOC_AGAIN`, `BLOCKADDR_USERS`, and the
//! debug records are skipped. Poison-generating flags the IR does not keep (`nneg`, `trunc
//! nuw/nsw`, `icmp samesign`, `getelementptr nusw/nuw`) and fast-math flags on casts are dropped,
//! which only removes poison and so refines the program; an unknown flag bit is malformed.
//! A call's calling convention is checked (C or fast) and not stored: the callee's is.
//!
//! # Calls
//!
//! A call to a declaration named `air.*`/`llvm.*` is decoded by `air::intrinsics`: each such
//! declaration is classified once when its module record is read, and each call binds its immediate
//! operands. A declaration whose own type is not the intrinsic's signature binds no call:
//! [`IntrinsicError::DeclaredSignatureMismatch`] names that declaration, at the function whose
//! body calls it, so the report says which spelling of the operation was read and where. Calls to the markers `llvm.lifetime.*`, `llvm.assume`, and
//! `llvm.experimental.noalias.scope.decl` are dropped: their operands are read and checked (the
//! last one's `metadata` operand must name metadata), they take a record number, which
//! attachments refer to, but no instruction. So is a call of the one inline assembly MSL accepts,
//! `asm sideeffect "", ""` of type `void ()`, which assembles to nothing (`executes_nothing`); a
//! call of any other inline assembly stays an indirect call through its constant. Metadata
//! operands of any other call are refused.
//!
//! # Metadata
//!
//! A function's own `METADATA` block is decoded with the module-half decoder into a scratch
//! arena whose ids continue after the module's. Attachments keep only `!llvm.loop` (as
//! [`LoopHints`], on terminators) and `!range` (as [`RangeMd`], on loads and calls); every other
//! kind is dropped. `USELIST` blocks are skipped: use-list order is an optimizer hint with no
//! semantics (a deviation, which lists no `USELIST` disposition; `llvm-as` writes them by default).

use std::collections::HashMap;

use super::attrs::AttrList;
use super::codes::*;
use super::constants::{self, binop, int_flags, ConstCx};
use super::enum_attrs::SiteEnums;
use super::metadata::{self, MdCx, MdKinds, MdValue};
use super::stream::{signed_vbr, Item, Stream};
use super::types::{grow, BitcodeTypes};
use super::ModuleValue;
use crate::budget::Meter;
use crate::error::{BitcodeError, DiagName, InputError, InputErrorKind, IntrinsicError, Location};
use crate::error::{Error, Feature, Limit};
use crate::intrinsics::{self, Bound, DeclClass};
use crate::ir::{
    AddrSpace, Align, Arena, Atomic, AtomicRmwOp, Block, BlockId, Body, Call, CallAttrs, Callee,
    CastOp, Const, ConstId, ConstTable, FastMath, FloatPredicate, FuncId, Function, Inst, InstId,
    InstKind, IntFlags, IntPredicate, LoopHints, Md, MdId, Ordering, RangeMd, Sym, Symbols,
    SyncScope, TailKind, Type, TypeId, TypeTable, Unroll, Value,
};

const B: u32 = FUNCTION_BLOCK;

/// `CALL` `CCInfo` bits (`CallMarkersFlags`).
const CALL_TAIL: u64 = 1 << 0;
const CALL_CCONV_SHIFT: u32 = 1;
const CALL_CCONV_MASK: u64 = 0x3FF << CALL_CCONV_SHIFT;
const CALL_MUSTTAIL: u64 = 1 << 14;
const CALL_EXPLICIT_TYPE: u64 = 1 << 15;
const CALL_NOTAIL: u64 = 1 << 16;
const CALL_FMF: u64 = 1 << 17;
const CALL_KNOWN: u64 =
    CALL_TAIL | CALL_CCONV_MASK | CALL_MUSTTAIL | CALL_EXPLICIT_TYPE | CALL_NOTAIL | CALL_FMF;
/// `CallingConv::C` and `CallingConv::Fast`.
const CC_C: u64 = 0;
const CC_FAST: u64 = 8;
/// The marker of the pre-3.x switch record with case ranges.
const SWITCH_INST_MAGIC: u64 = 0x4B5;
/// Alloca alignment record fields (`AllocaPackedValues`).
const ALLOCA_ALIGN_LOWER: u64 = 0x1F;
const ALLOCA_INALLOCA: u64 = 1 << 5;
const ALLOCA_EXPLICIT_TYPE: u64 = 1 << 6;
const ALLOCA_SWIFTERROR: u64 = 1 << 7;
const ALLOCA_ALIGN_UPPER_SHIFT: u32 = 8;
const ALLOCA_ALIGN_UPPER: u64 = 0x7;

/// Read-only module state a function body is decoded against, plus the module tables it may
/// extend (types, constants, symbols).
pub(crate) struct FnCx<'c> {
    pub bc_types: &'c BitcodeTypes,
    pub types: &'c mut TypeTable,
    pub consts: &'c mut ConstTable,
    pub symbols: &'c mut Symbols,
    pub module_md: &'c Arena<MdId, Md>,
    pub funcs: &'c Arena<FuncId, Function>,
    pub values: &'c [ModuleValue],
    pub attr_lists: &'c [AttrList],
    pub md_kinds: MdKinds,
    pub sync_scopes: &'c [Option<SyncScope>],
    /// Every `air.*`/`llvm.*` declaration with its class, sorted by function.
    pub intrinsics: &'c [(FuncId, DeclClass)],
    pub alloca_space: u32,
    /// Instructions decoded so far in the module, for `Limit::InstructionCount`.
    pub inst_total: &'c mut u64,
    /// Collect each call's uninterpreted enum attributes ([`DecodedBody::call_enums`]).
    pub keep_enums: bool,
    /// Collect the constants each dropped marker call named ([`DecodedBody::marker_consts`]).
    pub keep_marker_operands: bool,
}

/// A decoded body and the argument names from the function's symbol table.
#[derive(Debug)]
pub(crate) struct DecodedBody {
    pub body: Body,
    pub param_names: Vec<Option<Sym>>,
    /// With [`FnCx::keep_enums`], each call with uninterpreted enum attributes and those
    /// attributes by position.
    pub call_enums: Vec<(InstId, SiteEnums)>,
    /// With [`FnCx::keep_marker_operands`], every constant operand of a dropped marker call, in
    /// call order (repeats included).
    pub marker_consts: Vec<ConstId>,
}

/// Decode the `FUNCTION` block just entered, for function `func`.
pub(crate) fn decode(
    s: &mut Stream<'_>,
    cx: &mut FnCx<'_>,
    func: FuncId,
    m: &mut Meter,
) -> Result<DecodedBody, Error> {
    let f = cx
        .funcs
        .get(func)
        .ok_or_else(|| s.err(BitcodeError::BadValueRef))?;
    let (ret, params) = match cx.types.get(f.ty) {
        Some(Type::Function { ret, params, .. }) => (*ret, params.clone()),
        _ => return Err(s.err(BitcodeError::BadType)),
    };
    let module_count =
        u32::try_from(cx.values.len()).map_err(|_| s.err(BitcodeError::BadValueRef))?;
    let mut d = FnDecoder {
        module_count,
        next: module_count,
        local: Vec::new(),
        insts: Vec::new(),
        blocks: Vec::new(),
        declared: None,
        block_start: 0,
        forward: Vec::new(),
        local_md: Arena::new(),
        bb_names: Vec::new(),
        param_names: Vec::new(),
        consts_seen: false,
        records: 0,
        dropped: Vec::new(),
        call_enums: Vec::new(),
        marker_consts: Vec::new(),
        ret,
        func,
    };
    m.reserve(&mut d.local, params.len())?;
    for (i, &ty) in params.iter().enumerate() {
        let a = crate::ir::ArgIdx::new(u32::try_from(i).map_err(|_| s.err(BitcodeError::BadType))?);
        d.local.push((Value::Arg(a), ty));
    }
    d.next = d
        .next
        .checked_add(u32::try_from(params.len()).map_err(|_| s.err(BitcodeError::BadType))?)
        .ok_or(Error::LimitExceeded(Limit::InstructionCount))?;
    m.reserve(&mut d.param_names, params.len())?;
    d.param_names.resize(params.len(), None);
    loop {
        match s.next(m)? {
            Item::EndBlock => break,
            Item::End => return Err(s.malformed(B, 0)),
            Item::SubBlock(block) => d.sub_block(s, cx, block, m)?,
            Item::Record { code, .. } => d.record(s, cx, code, m)?,
        }
    }
    d.finish(s, m)
}

struct FnDecoder {
    module_count: u32,
    /// The next value number.
    next: u32,
    /// Function-local values (arguments, constants, instruction results) from `module_count`.
    local: Vec<(Value, TypeId)>,
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    declared: Option<u32>,
    block_start: u32,
    /// Forward references: `(value id, expected type)`, by placeholder index.
    forward: Vec<(u32, TypeId)>,
    /// The function's own metadata, ids from the module's metadata count.
    local_md: Arena<MdId, Md>,
    bb_names: Vec<(u32, Sym)>,
    param_names: Vec<Option<Sym>>,
    consts_seen: bool,
    /// Instruction records read, dropped markers included (the numbering attachments use).
    records: u32,
    /// The record numbers of the dropped markers, ascending.
    dropped: Vec<u32>,
    /// Calls with uninterpreted enum attributes, when the context keeps them.
    call_enums: Vec<(InstId, SiteEnums)>,
    /// The constant operands of the dropped marker calls, when the context keeps them.
    marker_consts: Vec<ConstId>,
    ret: TypeId,
    /// The function this block defines, to name it where a declaration mismatch is reported.
    func: FuncId,
}

/// A cursor over the operands of one record.
struct Ops<'r> {
    ops: &'r [u64],
    i: usize,
    code: u32,
}

impl Ops<'_> {
    fn next(&mut self, s: &Stream<'_>) -> Result<u64, Error> {
        let v = self
            .ops
            .get(self.i)
            .copied()
            .ok_or_else(|| s.malformed(B, self.code))?;
        self.i += 1;
        Ok(v)
    }

    fn next32(&mut self, s: &Stream<'_>) -> Result<u32, Error> {
        u32::try_from(self.next(s)?).map_err(|_| s.malformed(B, self.code))
    }

    /// The next operand, if any is left.
    fn opt(&mut self) -> Option<u64> {
        let v = self.ops.get(self.i).copied();
        if v.is_some() {
            self.i += 1;
        }
        v
    }

    fn left(&self) -> usize {
        self.ops.len().saturating_sub(self.i)
    }

    /// Require every operand to be consumed.
    fn end(&self, s: &Stream<'_>) -> Result<(), Error> {
        if self.left() == 0 {
            Ok(())
        } else {
            Err(s.malformed(B, self.code))
        }
    }
}

impl FnDecoder {
    fn sub_block(
        &mut self,
        s: &mut Stream<'_>,
        cx: &mut FnCx<'_>,
        block: u32,
        m: &mut Meter,
    ) -> Result<(), Error> {
        match block {
            CONSTANTS_BLOCK => self.constants(s, cx, m),
            METADATA_BLOCK => self.metadata(s, cx, m),
            METADATA_ATTACHMENT_BLOCK => self.attachments(s, cx, m),
            VALUE_SYMTAB_BLOCK => self.symtab(s, cx, m),
            USELIST_BLOCK => s.skip_block(m),
            _ => Err(s.err(BitcodeError::UnknownBlock { block })),
        }
    }

    /// The function-local constants, numbered after the arguments. LLVM writes one block before
    /// the first instruction; any other placement would break the numbering.
    fn constants(
        &mut self,
        s: &mut Stream<'_>,
        cx: &mut FnCx<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        if self.consts_seen || !self.insts.is_empty() {
            return Err(s.malformed(CONSTANTS_BLOCK, 0));
        }
        self.consts_seen = true;
        let values = cx.values;
        let prior = |id: u64| -> Option<ConstId> {
            usize::try_from(id)
                .ok()
                .and_then(|i| values.get(i))
                .and_then(|v| v.id)
        };
        let mut ccx = ConstCx {
            bc_types: cx.bc_types,
            types: cx.types,
            consts: cx.consts,
        };
        let decoded = constants::decode(s, &mut ccx, u64::from(self.next), &prior, m)?;
        m.reserve(&mut self.local, decoded.len())?;
        for c in decoded {
            let ty = cx
                .consts
                .ty(c.id)
                .ok_or_else(|| s.err(BitcodeError::BadValueRef))?;
            self.local.push((Value::Const(c.id), ty));
            self.next = self
                .next
                .checked_add(1)
                .ok_or(Error::LimitExceeded(Limit::InstructionCount))?;
        }
        Ok(())
    }

    /// A function-level `METADATA` block, decoded into the function's arena. Its ids continue
    /// after the module's and after any earlier function-level block.
    fn metadata(
        &mut self,
        s: &mut Stream<'_>,
        cx: &mut FnCx<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let base = u32::try_from(cx.module_md.len())
            .map_err(|_| Error::LimitExceeded(Limit::MetadataNodes))?;
        let mut named = Vec::new();
        let FnDecoder {
            local,
            module_count,
            local_md,
            ..
        } = self;
        let (local, module_count) = (&*local, *module_count);
        let consts = &*cx.consts;
        let (bc_types, values) = (cx.bc_types, cx.values);
        let value = |ty: u64, id: u64| -> MdValue {
            let Some(want) = bc_types.ty(ty) else {
                return MdValue::Bad;
            };
            let Ok(id) = u32::try_from(id) else {
                return MdValue::Bad;
            };
            match lookup(values, consts, module_count, local, id) {
                Some((Value::Const(c), t)) if t == want => MdValue::Const(c),
                Some((Value::Const(_), _)) | None => MdValue::Bad,
                Some(_) => MdValue::Local,
            }
        };
        let mut mcx = MdCx {
            symbols: cx.symbols,
            metadata: local_md,
            base,
            named: &mut named,
        };
        metadata::decode(s, &mut mcx, &value, &|_| false, m)?;
        if !named.is_empty() {
            return Err(s.malformed(METADATA_BLOCK, METADATA_NAMED_NODE));
        }
        Ok(())
    }

    /// The metadata entry `id` (module or function-local).
    fn md<'a>(&'a self, cx: &'a FnCx<'_>, id: MdId) -> Option<&'a Md> {
        let module = cx.module_md.len();
        match (id.raw() as usize).checked_sub(module) {
            None => cx.module_md.get(id),
            Some(local) => self.local_md.get(MdId::new(u32::try_from(local).ok()?)),
        }
    }

    fn attachments(
        &mut self,
        s: &mut Stream<'_>,
        cx: &mut FnCx<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        const AB: u32 = METADATA_ATTACHMENT_BLOCK;
        let kinds = cx.md_kinds;
        loop {
            match s.next(m)? {
                Item::EndBlock => return Ok(()),
                Item::End => return Err(s.malformed(AB, 0)),
                Item::SubBlock(block) => return Err(s.err(BitcodeError::UnknownBlock { block })),
                Item::Record { code, .. } => {
                    if code != METADATA_ATTACHMENT {
                        return Err(s.err(BitcodeError::UnknownRecord { block: AB, code }));
                    }
                    let ops = s.ops();
                    // Odd: `[instruction, (kind, node)*]`; even: function attachments.
                    let (inst, pairs) = if !ops.len().is_multiple_of(2) {
                        let (&i, rest) = ops.split_first().ok_or_else(|| s.malformed(AB, code))?;
                        (Some(i), rest)
                    } else {
                        (None, ops)
                    };
                    for &[kind, node] in pairs.as_chunks::<2>().0 {
                        m.tick(1)?;
                        let node =
                            MdId::new(u32::try_from(node).map_err(|_| s.malformed(AB, code))?);
                        if self.md(cx, node).is_none() {
                            return Err(s.err(BitcodeError::BadMetadata));
                        }
                        let Some(i) = inst else {
                            // The IR keeps no function attachments.
                            continue;
                        };
                        let idx = match self.inst_of_record(i) {
                            None => return Err(s.malformed(AB, code)),
                            // An attachment on a marker decode dropped goes with it.
                            Some(None) => continue,
                            Some(Some(idx)) => idx,
                        };
                        if Some(kind) == kinds.loop_hints {
                            let hints = self
                                .loop_hints(cx, node)
                                .ok_or_else(|| s.err(BitcodeError::BadMetadata))?;
                            let target = self
                                .insts
                                .get_mut(idx)
                                .ok_or_else(|| s.malformed(AB, code))?;
                            match &mut target.kind {
                                InstKind::Br { hints: h, .. }
                                | InstKind::CondBr { hints: h, .. }
                                | InstKind::Switch { hints: h, .. } => *h = Some(hints),
                                _ => return Err(s.malformed(AB, code)),
                            }
                        } else if Some(kind) == kinds.range {
                            let range = self
                                .range(cx, node, m)?
                                .ok_or_else(|| s.err(BitcodeError::BadMetadata))?;
                            let target = self
                                .insts
                                .get_mut(idx)
                                .ok_or_else(|| s.malformed(AB, code))?;
                            match &mut target.kind {
                                InstKind::Load { range: r, .. } => *r = Some(range),
                                InstKind::Call(call) => call.range = Some(range),
                                _ => return Err(s.malformed(AB, code)),
                            }
                        }
                    }
                }
            }
        }
    }

    /// Decode a `!llvm.loop` node: its property nodes named by stable `llvm.loop.*` keys. Other
    /// properties (vectorizer, distribution, debug locations) are advisory and ignored.
    fn loop_hints(&self, cx: &FnCx<'_>, node: MdId) -> Option<LoopHints> {
        let Some(Md::Node { ops, .. }) = self.md(cx, node) else {
            return None;
        };
        let mut hints = LoopHints::default();
        for op in ops.iter().skip(1).flatten() {
            let Some(Md::Node { ops: prop, .. }) = self.md(cx, *op) else {
                continue;
            };
            let Some(Some(key)) = prop.first() else {
                continue;
            };
            let Some(Md::String(key)) = self.md(cx, *key) else {
                continue;
            };
            match cx.symbols.get(*key)? {
                b"llvm.loop.mustprogress" => hints.must_progress = true,
                b"llvm.loop.unroll.disable" => hints.unroll = Unroll::Disable,
                b"llvm.loop.unroll.enable" => hints.unroll = Unroll::Enable,
                b"llvm.loop.unroll.full" => hints.unroll = Unroll::Full,
                b"llvm.loop.unroll.count" => {
                    let Some(Some(v)) = prop.get(1) else {
                        return None;
                    };
                    let Some(Md::Value(c)) = self.md(cx, *v) else {
                        return None;
                    };
                    let Some(Const::Int(n)) = cx.consts.get(*c).map(|c| &c.value) else {
                        return None;
                    };
                    hints.unroll = Unroll::Count(u32::try_from(*n).ok()?);
                }
                _ => {}
            }
        }
        Some(hints)
    }

    /// Decode a `!range` node: pairs of integer constants.
    fn range(&self, cx: &FnCx<'_>, node: MdId, m: &mut Meter) -> Result<Option<RangeMd>, Error> {
        let Some(Md::Node { ops, .. }) = self.md(cx, node) else {
            return Ok(None);
        };
        if ops.is_empty() || !ops.len().is_multiple_of(2) {
            return Ok(None);
        }
        let mut ranges = Vec::new();
        m.reserve(&mut ranges, ops.len() / 2)?;
        let int = |op: &Option<MdId>| -> Option<u64> {
            let Some(Md::Value(c)) = self.md(cx, (*op)?) else {
                return None;
            };
            match cx.consts.get(*c)?.value {
                Const::Int(v) => Some(v),
                _ => None,
            }
        };
        for [lo, hi] in ops.as_chunks::<2>().0 {
            let (Some(lo), Some(hi)) = (int(lo), int(hi)) else {
                return Ok(None);
            };
            ranges.push((lo, hi));
        }
        Ok(Some(RangeMd {
            ranges: ranges.into_boxed_slice(),
        }))
    }

    /// The function-level `VALUE_SYMTAB`: argument and block names. Instruction and constant
    /// names are dropped (the IR keeps none).
    fn symtab(
        &mut self,
        s: &mut Stream<'_>,
        cx: &mut FnCx<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        const VB: u32 = VALUE_SYMTAB_BLOCK;
        let module_count = u64::from(self.module_count);
        let next = u64::from(self.next);
        let nparams = self.param_names.len() as u64;
        loop {
            match s.next(m)? {
                Item::EndBlock => return Ok(()),
                Item::End => return Err(s.malformed(VB, 0)),
                Item::SubBlock(block) => return Err(s.err(BitcodeError::UnknownBlock { block })),
                Item::Record { code, .. } => match code {
                    VST_ENTRY => {
                        let id = s.op(VB, code, 0)?;
                        let name = s.chars(m, VB, code, 1)?;
                        match id.checked_sub(module_count) {
                            Some(a) if a < nparams => {
                                let sym = cx.symbols.intern(&name, m)?;
                                let slot = usize::try_from(a)
                                    .ok()
                                    .and_then(|a| self.param_names.get_mut(a))
                                    .ok_or_else(|| s.malformed(VB, code))?;
                                *slot = Some(sym);
                            }
                            Some(_) if id < next => {}
                            _ => return Err(s.err(BitcodeError::BadValueRef)),
                        }
                    }
                    VST_BBENTRY => {
                        let bb = s.op32(VB, code, 0)?;
                        let name = s.chars(m, VB, code, 1)?;
                        if self.declared.is_none_or(|n| bb >= n) {
                            return Err(s.malformed(VB, code));
                        }
                        let sym = cx.symbols.intern(&name, m)?;
                        grow(m, &mut self.bb_names)?;
                        self.bb_names.push((bb, sym));
                    }
                    VST_FNENTRY => return Err(s.malformed(VB, code)),
                    VST_COMBINED_ENTRY => {
                        return Err(s.unsupported(Feature::BitcodeRecord { block: VB, code }))
                    }
                    _ => return Err(s.err(BitcodeError::UnknownRecord { block: VB, code })),
                },
            }
        }
    }

    /// The value numbered `id` and its type, if defined.
    fn get(&self, values: &[ModuleValue], consts: &ConstTable, id: u32) -> Option<(Value, TypeId)> {
        lookup(values, consts, self.module_count, &self.local, id)
    }

    /// A forward reference to value `id` of type `ty`.
    fn forward(&mut self, id: u32, ty: TypeId, m: &mut Meter) -> Result<Value, Error> {
        let k = u32::try_from(self.forward.len())
            .ok()
            .filter(|&k| k < u32::MAX / 2)
            .ok_or(Error::LimitExceeded(Limit::InstructionCount))?;
        grow(m, &mut self.forward)?;
        self.forward.push((id, ty));
        Ok(Value::Inst(InstId::new(u32::MAX - k)))
    }

    /// The value `id` required to have type `ty` (a forward reference when not yet defined).
    fn value_at(
        &mut self,
        s: &Stream<'_>,
        cx: &FnCx<'_>,
        id: u32,
        ty: TypeId,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        if id < self.next {
            match self.get(cx.values, cx.consts, id) {
                Some((v, t)) if t == ty => Ok(v),
                _ => Err(s.err(BitcodeError::BadValueRef)),
            }
        } else {
            self.forward(id, ty, m)
        }
    }

    /// A relative operand whose type the record fixes.
    fn typed(
        &mut self,
        s: &Stream<'_>,
        cx: &FnCx<'_>,
        r: &mut Ops<'_>,
        ty: TypeId,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let rel = r.next32(s)?;
        let id = self.next.wrapping_sub(rel);
        self.value_at(s, cx, id, ty, m)
    }

    /// A relative operand followed, when it is a forward reference, by its type.
    fn pair(
        &mut self,
        s: &Stream<'_>,
        cx: &FnCx<'_>,
        r: &mut Ops<'_>,
        m: &mut Meter,
    ) -> Result<(Value, TypeId), Error> {
        let rel = r.next32(s)?;
        let id = self.next.wrapping_sub(rel);
        if id < self.next {
            return self
                .get(cx.values, cx.consts, id)
                .ok_or_else(|| s.err(BitcodeError::BadValueRef));
        }
        let ty = bc_ty(s, cx, r.next(s)?)?;
        Ok((self.forward(id, ty, m)?, ty))
    }

    /// A defined value by absolute id (switch cases, alloca sizes).
    fn absolute(&self, s: &Stream<'_>, cx: &FnCx<'_>, id: u64, ty: TypeId) -> Result<Value, Error> {
        u32::try_from(id)
            .ok()
            .filter(|&id| id < self.next)
            .and_then(|id| self.get(cx.values, cx.consts, id))
            .filter(|&(_, t)| t == ty)
            .map(|(v, _)| v)
            .ok_or_else(|| s.err(BitcodeError::BadValueRef))
    }

    /// Block operand `raw`.
    fn block(&self, s: &Stream<'_>, code: u32, raw: u64) -> Result<BlockId, Error> {
        match (u32::try_from(raw), self.declared) {
            (Ok(b), Some(n)) if b < n => Ok(BlockId::new(b)),
            _ => Err(s.malformed(B, code)),
        }
    }

    /// Append an instruction to the open block; a non-void result takes the next value number
    /// and a terminator closes the block.
    fn push(
        &mut self,
        s: &Stream<'_>,
        cx: &mut FnCx<'_>,
        code: u32,
        ty: TypeId,
        kind: InstKind,
        m: &mut Meter,
    ) -> Result<(), Error> {
        self.count_record(s, cx, code, m)?;
        let id = InstId::new(
            u32::try_from(self.insts.len())
                .map_err(|_| Error::LimitExceeded(Limit::InstructionCount))?,
        );
        let terminator = kind.is_terminator();
        grow(m, &mut self.insts)?;
        self.insts.push(Inst { ty, kind });
        if !matches!(cx.types.get(ty), Some(Type::Void)) {
            grow(m, &mut self.local)?;
            self.local.push((Value::Inst(id), ty));
            self.next = self
                .next
                .checked_add(1)
                .ok_or(Error::LimitExceeded(Limit::InstructionCount))?;
        }
        if terminator {
            let end = InstId::new(id.raw() + 1);
            grow(m, &mut self.blocks)?;
            self.blocks.push(Block {
                start: InstId::new(self.block_start),
                end,
                name: None,
            });
            self.block_start = end.raw();
        }
        Ok(())
    }

    /// Count one instruction record in the open block.
    fn count_record(
        &mut self,
        s: &Stream<'_>,
        cx: &mut FnCx<'_>,
        code: u32,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let open = self
            .declared
            .is_some_and(|n| (self.blocks.len() as u64) < u64::from(n));
        if !open {
            return Err(s.malformed(B, code));
        }
        *cx.inst_total = cx.inst_total.saturating_add(1);
        m.cap(Limit::InstructionCount, *cx.inst_total)?;
        self.records = self
            .records
            .checked_add(1)
            .ok_or(Error::LimitExceeded(Limit::InstructionCount))?;
        Ok(())
    }

    /// Record a call to a marker decode drops: it takes a record number, which attachments refer
    /// to, but no instruction id and no value number (it returns void).
    fn drop_call(&mut self, s: &Stream<'_>, cx: &mut FnCx<'_>, m: &mut Meter) -> Result<(), Error> {
        let record = self.records;
        self.count_record(s, cx, FUNC_CALL, m)?;
        grow(m, &mut self.dropped)?;
        self.dropped.push(record);
        Ok(())
    }

    /// The instruction index of instruction record `record`: `None` when out of range,
    /// `Some(None)` for a dropped marker.
    fn inst_of_record(&self, record: u64) -> Option<Option<usize>> {
        let record = u32::try_from(record).ok().filter(|&r| r < self.records)?;
        if self.dropped.binary_search(&record).is_ok() {
            return Some(None);
        }
        let before = self.dropped.partition_point(|&d| d < record);
        Some(Some((record as usize).saturating_sub(before)))
    }

    fn intern(&self, cx: &mut FnCx<'_>, ty: Type, m: &mut Meter) -> Result<TypeId, Error> {
        cx.types.intern(ty, m)
    }

    fn record(
        &mut self,
        s: &Stream<'_>,
        cx: &mut FnCx<'_>,
        code: u32,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let mut r = Ops {
            ops: s.ops(),
            i: 0,
            code,
        };
        let bad = || s.malformed(B, code);
        let void = self.intern(cx, Type::Void, m)?;
        match code {
            FUNC_DECLAREBLOCKS => {
                if self.declared.is_some() {
                    return Err(bad());
                }
                let n = r.next32(s)?;
                r.end(s)?;
                if n == 0 {
                    return Err(bad());
                }
                self.declared = Some(n);
                Ok(())
            }
            FUNC_BINOP => {
                let (a, ty) = self.pair(s, cx, &mut r, m)?;
                let b = self.typed(s, cx, &mut r, ty, m)?;
                let float = cx.types.float_bits(ty).is_some();
                let op = binop(r.next(s)?, float).ok_or_else(bad)?;
                let (mut flags, mut fmf) = (IntFlags::NONE, FastMath::NONE);
                if let Some(raw) = r.opt() {
                    if op.is_float() {
                        fmf = fast_math(raw).ok_or_else(bad)?;
                    } else {
                        flags = int_flags(op, raw).ok_or_else(bad)?;
                    }
                }
                r.end(s)?;
                let kind = InstKind::BinOp {
                    op,
                    flags,
                    fmf,
                    a,
                    b,
                };
                self.push(s, cx, code, ty, kind, m)
            }
            FUNC_CAST => {
                let (v, _) = self.pair(s, cx, &mut r, m)?;
                let to = bc_ty(s, cx, r.next(s)?)?;
                let op = match r.next(s)? {
                    0 => CastOp::Trunc,
                    1 => CastOp::ZExt,
                    2 => CastOp::SExt,
                    3 => CastOp::FPToUI,
                    4 => CastOp::FPToSI,
                    5 => CastOp::UIToFP,
                    6 => CastOp::SIToFP,
                    7 => CastOp::FPTrunc,
                    8 => CastOp::FPExt,
                    9 => CastOp::PtrToInt,
                    10 => CastOp::IntToPtr,
                    11 => CastOp::Bitcast,
                    12 => CastOp::AddrSpaceCast,
                    // CAST_PTRTOADDR: an LLVM cast the IR has no form for.
                    13 => return Err(s.unsupported(Feature::Instruction { code })),
                    _ => return Err(bad()),
                };
                if let Some(raw) = r.opt() {
                    // nneg (zext, uitofp), nuw/nsw (trunc), or fast-math flags (FP casts): all
                    // dropped (see the module docs). LLVM 23 gives `uitofp` fast-math flags too,
                    // above its `nneg` bit.
                    let known = match op {
                        CastOp::ZExt => 1,
                        CastOp::UIToFP => 0x1FF,
                        CastOp::Trunc => 3,
                        CastOp::FPTrunc
                        | CastOp::FPExt
                        | CastOp::FPToUI
                        | CastOp::FPToSI
                        | CastOp::SIToFP => 0xFF,
                        _ => 0,
                    };
                    if raw & !known != 0 {
                        return Err(bad());
                    }
                }
                r.end(s)?;
                self.push(s, cx, code, to, InstKind::Cast { op, v }, m)
            }
            FUNC_UNOP => {
                let (v, ty) = self.pair(s, cx, &mut r, m)?;
                if r.next(s)? != 0 {
                    return Err(bad());
                }
                let fmf = match r.opt() {
                    Some(raw) => fast_math(raw).ok_or_else(bad)?,
                    None => FastMath::NONE,
                };
                r.end(s)?;
                self.push(s, cx, code, ty, InstKind::FNeg { v, fmf }, m)
            }
            FUNC_EXTRACTELT => {
                let (vec, vt) = self.pair(s, cx, &mut r, m)?;
                let (idx, _) = self.pair(s, cx, &mut r, m)?;
                r.end(s)?;
                let elem = match cx.types.get(vt) {
                    Some(Type::Vector { elem, .. }) => *elem,
                    _ => return Err(bad()),
                };
                self.push(s, cx, code, elem, InstKind::ExtractElement { vec, idx }, m)
            }
            FUNC_INSERTELT => {
                let (vec, vt) = self.pair(s, cx, &mut r, m)?;
                let elem = match cx.types.get(vt) {
                    Some(Type::Vector { elem, .. }) => *elem,
                    _ => return Err(bad()),
                };
                let elt = self.typed(s, cx, &mut r, elem, m)?;
                let (idx, _) = self.pair(s, cx, &mut r, m)?;
                r.end(s)?;
                self.push(
                    s,
                    cx,
                    code,
                    vt,
                    InstKind::InsertElement { vec, elt, idx },
                    m,
                )
            }
            FUNC_SHUFFLEVEC => {
                let (a, vt) = self.pair(s, cx, &mut r, m)?;
                let b = self.typed(s, cx, &mut r, vt, m)?;
                let (mask, mt) = self.pair(s, cx, &mut r, m)?;
                r.end(s)?;
                let elem = match cx.types.get(vt) {
                    Some(Type::Vector { elem, .. }) => *elem,
                    _ => return Err(bad()),
                };
                let lanes = cx.types.lanes(mt).ok_or_else(bad)?;
                let mask = shuffle_mask(cx.consts, mask, lanes, m)?.ok_or_else(bad)?;
                let ty = self.intern(cx, Type::Vector { elem, len: lanes }, m)?;
                self.push(s, cx, code, ty, InstKind::ShuffleVector { a, b, mask }, m)
            }
            FUNC_CMP | FUNC_CMP2 => {
                let (a, ty) = self.pair(s, cx, &mut r, m)?;
                let b = self.typed(s, cx, &mut r, ty, m)?;
                let pred = r.next(s)?;
                let flags = r.opt();
                r.end(s)?;
                let i1 = self.intern(cx, Type::Int(1), m)?;
                let res = match cx.types.lanes(ty) {
                    Some(len) => self.intern(cx, Type::Vector { elem: i1, len }, m)?,
                    None => i1,
                };
                let kind = if cx.types.float_bits(ty).is_some() {
                    let pred = usize::try_from(pred)
                        .ok()
                        .and_then(|p| FloatPredicate::ALL.get(p))
                        .copied()
                        .ok_or_else(bad)?;
                    let fmf = match flags {
                        Some(raw) => fast_math(raw).ok_or_else(bad)?,
                        None => FastMath::NONE,
                    };
                    InstKind::FCmp { pred, fmf, a, b }
                } else {
                    let pred = pred
                        .checked_sub(32)
                        .and_then(|p| usize::try_from(p).ok())
                        .and_then(|p| IntPredicate::ALL.get(p))
                        .copied()
                        .ok_or_else(bad)?;
                    // `samesign` (bit 0) is dropped.
                    if flags.is_some_and(|f| f & !1 != 0) {
                        return Err(bad());
                    }
                    InstKind::ICmp { pred, a, b }
                };
                self.push(s, cx, code, res, kind, m)
            }
            FUNC_VSELECT => {
                let (t, ty) = self.pair(s, cx, &mut r, m)?;
                let f = self.typed(s, cx, &mut r, ty, m)?;
                let (c, _) = self.pair(s, cx, &mut r, m)?;
                let fmf = match r.opt() {
                    Some(raw) => fast_math(raw).ok_or_else(bad)?,
                    None => FastMath::NONE,
                };
                r.end(s)?;
                self.push(s, cx, code, ty, InstKind::Select { c, t, f, fmf }, m)
            }
            FUNC_EXTRACTVAL => {
                let (agg, at) = self.pair(s, cx, &mut r, m)?;
                let indices = indices(s, &mut r, m)?;
                let leaf = aggregate_leaf(cx.types, at, &indices).ok_or_else(bad)?;
                self.push(
                    s,
                    cx,
                    code,
                    leaf,
                    InstKind::ExtractValue { agg, indices },
                    m,
                )
            }
            FUNC_INSERTVAL => {
                let (agg, at) = self.pair(s, cx, &mut r, m)?;
                let (val, _) = self.pair(s, cx, &mut r, m)?;
                let indices = indices(s, &mut r, m)?;
                aggregate_leaf(cx.types, at, &indices).ok_or_else(bad)?;
                let kind = InstKind::InsertValue { agg, val, indices };
                self.push(s, cx, code, at, kind, m)
            }
            FUNC_FREEZE => {
                let (v, ty) = self.pair(s, cx, &mut r, m)?;
                r.end(s)?;
                self.push(s, cx, code, ty, InstKind::Freeze(v), m)
            }
            FUNC_RET => {
                let v = if r.left() == 0 {
                    None
                } else {
                    let (v, ty) = self.pair(s, cx, &mut r, m)?;
                    if ty != self.ret {
                        return Err(s.err(BitcodeError::BadValueRef));
                    }
                    Some(v)
                };
                r.end(s)?;
                self.push(s, cx, code, void, InstKind::Ret(v), m)
            }
            FUNC_BR => {
                let dest = self.block(s, code, r.next(s)?)?;
                let kind = if r.left() == 0 {
                    InstKind::Br { dest, hints: None }
                } else {
                    let f = self.block(s, code, r.next(s)?)?;
                    let i1 = self.intern(cx, Type::Int(1), m)?;
                    let c = self.typed(s, cx, &mut r, i1, m)?;
                    r.end(s)?;
                    InstKind::CondBr {
                        c,
                        t: dest,
                        f,
                        hints: None,
                    }
                };
                self.push(s, cx, code, void, kind, m)
            }
            FUNC_SWITCH => {
                let opty_raw = r.next(s)?;
                if opty_raw >> 16 == SWITCH_INST_MAGIC {
                    return Err(s.unsupported(Feature::Instruction { code }));
                }
                let opty = bc_ty(s, cx, opty_raw)?;
                let v = self.typed(s, cx, &mut r, opty, m)?;
                let default = self.block(s, code, r.next(s)?)?;
                if !r.left().is_multiple_of(2) {
                    return Err(bad());
                }
                let mut cases = Vec::new();
                m.reserve(&mut cases, r.left() / 2)?;
                while r.left() > 0 {
                    m.tick(1)?;
                    let c = self.absolute(s, cx, r.next(s)?, opty)?;
                    let bits = match c {
                        Value::Const(c) => match cx.consts.get(c).map(|c| &c.value) {
                            Some(Const::Int(v)) => *v,
                            _ => return Err(bad()),
                        },
                        _ => return Err(bad()),
                    };
                    let dest = self.block(s, code, r.next(s)?)?;
                    cases.push((bits, dest));
                }
                let kind = InstKind::Switch {
                    v,
                    default,
                    cases: cases.into_boxed_slice(),
                    hints: None,
                };
                self.push(s, cx, code, void, kind, m)
            }
            FUNC_UNREACHABLE => {
                r.end(s)?;
                self.push(s, cx, code, void, InstKind::Unreachable, m)
            }
            FUNC_PHI => self.phi(s, cx, &mut r, m),
            FUNC_ALLOCA => {
                if !(4..=5).contains(&r.ops.len()) {
                    return Err(bad());
                }
                let ty_raw = r.next(s)?;
                let opty = bc_ty(s, cx, r.next(s)?)?;
                let count = self.absolute(s, cx, r.next(s)?, opty)?;
                let rec = r.next(s)?;
                if rec & (ALLOCA_INALLOCA | ALLOCA_SWIFTERROR) != 0
                    || rec & ALLOCA_EXPLICIT_TYPE == 0
                {
                    return Err(s.unsupported(Feature::Instruction { code }));
                }
                let upper = rec >> ALLOCA_ALIGN_UPPER_SHIFT;
                if upper & !ALLOCA_ALIGN_UPPER != 0 {
                    return Err(bad());
                }
                let align = align((rec & ALLOCA_ALIGN_LOWER) | (upper << 5)).ok_or_else(bad)?;
                let ty = bc_ty(s, cx, ty_raw)?;
                let raw_space = match r.opt() {
                    Some(sp) => u32::try_from(sp).map_err(|_| bad())?,
                    None => cx.alloca_space,
                };
                let space = AddrSpace::from_raw(raw_space).map_err(|u| s.unsupported(u.feature))?;
                let ptr = self.intern(cx, Type::Ptr { space }, m)?;
                self.push(s, cx, code, ptr, InstKind::Alloca { ty, count, align }, m)
            }
            FUNC_LOAD | FUNC_LOADATOMIC => {
                let (ptr, _) = self.pair(s, cx, &mut r, m)?;
                let fixed = if code == FUNC_LOAD { 2 } else { 4 };
                if r.left() == fixed {
                    // The typed-pointer spelling without an explicit type.
                    return Err(s.unsupported(Feature::Instruction { code }));
                }
                if r.left() != fixed + 1 {
                    return Err(bad());
                }
                let ty = bc_ty(s, cx, r.next(s)?)?;
                let align = align(r.next(s)?).ok_or_else(bad)?;
                let volatile = flag(s, code, r.next(s)?)?;
                let atomic = if code == FUNC_LOADATOMIC {
                    Some(self.atomic(s, cx, code, &mut r)?)
                } else {
                    None
                };
                let kind = InstKind::Load {
                    ptr,
                    align,
                    volatile,
                    atomic,
                    range: None,
                };
                self.push(s, cx, code, ty, kind, m)
            }
            FUNC_STORE | FUNC_STOREATOMIC => {
                let (ptr, _) = self.pair(s, cx, &mut r, m)?;
                let (val, _) = self.pair(s, cx, &mut r, m)?;
                let fixed = if code == FUNC_STORE { 2 } else { 4 };
                if r.left() != fixed {
                    return Err(bad());
                }
                let align = align(r.next(s)?).ok_or_else(bad)?;
                let volatile = flag(s, code, r.next(s)?)?;
                let atomic = if code == FUNC_STOREATOMIC {
                    Some(self.atomic(s, cx, code, &mut r)?)
                } else {
                    None
                };
                let kind = InstKind::Store {
                    val,
                    ptr,
                    align,
                    volatile,
                    atomic,
                };
                self.push(s, cx, code, void, kind, m)
            }
            FUNC_FENCE => {
                let a = self.atomic(s, cx, code, &mut r)?;
                r.end(s)?;
                self.push(s, cx, code, void, InstKind::Fence(a), m)
            }
            FUNC_ATOMICRMW => {
                let (ptr, _) = self.pair(s, cx, &mut r, m)?;
                let (val, ty) = self.pair(s, cx, &mut r, m)?;
                if !(4..=5).contains(&r.left()) {
                    return Err(bad());
                }
                let op = rmw_op(r.next(s)?)
                    .ok_or_else(|| s.unsupported(Feature::Instruction { code }))?;
                let volatile = flag(s, code, r.next(s)?)?;
                let atomic = self.atomic(s, cx, code, &mut r)?;
                let align = match r.opt() {
                    Some(a) => align(a).ok_or_else(bad)?,
                    None => None,
                };
                let kind = InstKind::AtomicRmw {
                    op,
                    ptr,
                    val,
                    atomic,
                    volatile,
                    align,
                };
                self.push(s, cx, code, ty, kind, m)
            }
            FUNC_CMPXCHG => {
                let (ptr, _) = self.pair(s, cx, &mut r, m)?;
                let (cmp, ty) = self.pair(s, cx, &mut r, m)?;
                let new = self.typed(s, cx, &mut r, ty, m)?;
                if !(3..=6).contains(&r.left()) {
                    return Err(bad());
                }
                let volatile = flag(s, code, r.next(s)?)?;
                let success = self.atomic(s, cx, code, &mut r)?;
                let failure = match r.opt() {
                    Some(o) => ordering(o).ok_or_else(bad)?,
                    None => strongest_failure(success.ordering),
                };
                let weak = match r.opt() {
                    Some(w) => flag(s, code, w)?,
                    None => false,
                };
                let align = match r.opt() {
                    Some(a) => align(a).ok_or_else(bad)?,
                    None => None,
                };
                let i1 = self.intern(cx, Type::Int(1), m)?;
                let res = self.intern(
                    cx,
                    Type::Struct {
                        fields: Box::new([ty, i1]),
                        packed: false,
                        name: None,
                    },
                    m,
                )?;
                let kind = InstKind::CmpXchg {
                    ptr,
                    cmp,
                    new,
                    success,
                    failure,
                    weak,
                    volatile,
                    align,
                };
                self.push(s, cx, code, res, kind, m)
            }
            FUNC_GEP => {
                let flags = r.next(s)?;
                // inbounds (bit 0); nusw and nuw (bits 1, 2) are dropped.
                if flags & !7 != 0 {
                    return Err(bad());
                }
                let src_ty = bc_ty(s, cx, r.next(s)?)?;
                let (base, bt) = self.pair(s, cx, &mut r, m)?;
                let mut idx = Vec::new();
                m.reserve(&mut idx, r.left())?;
                while r.left() > 0 {
                    m.tick(1)?;
                    let (v, t) = self.pair(s, cx, &mut r, m)?;
                    if cx.types.lanes(t).is_some() {
                        return Err(s.unsupported(Feature::Instruction { code }));
                    }
                    idx.push(v);
                }
                let space = match cx.types.get(bt) {
                    Some(Type::Ptr { space }) => *space,
                    // A vector of pointers: a vector GEP.
                    Some(Type::Vector { .. }) => {
                        return Err(s.unsupported(Feature::Instruction { code }))
                    }
                    _ => return Err(bad()),
                };
                let ptr = self.intern(cx, Type::Ptr { space }, m)?;
                let kind = InstKind::Gep {
                    src_ty,
                    base,
                    indices: idx.into_boxed_slice(),
                    inbounds: flags & 1 != 0,
                };
                self.push(s, cx, code, ptr, kind, m)
            }
            FUNC_CALL => self.call(s, cx, &mut r, m),
            FUNC_DEBUG_LOC | FUNC_DEBUG_LOC_AGAIN | FUNC_BLOCKADDR_USERS => Ok(()),
            c if FUNC_DEBUG_RECORDS.contains(&c) => Ok(()),
            c if FUNC_UNSUPPORTED.contains(&c) => Err(s.unsupported(Feature::Instruction { code })),
            _ => Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        }
    }

    /// `[ordering, sync scope]` at the cursor.
    fn atomic(
        &self,
        s: &Stream<'_>,
        cx: &FnCx<'_>,
        code: u32,
        r: &mut Ops<'_>,
    ) -> Result<Atomic, Error> {
        let ordering = ordering(r.next(s)?).ok_or_else(|| s.malformed(B, code))?;
        let scope = match r.next(s)? {
            0 => SyncScope::SingleThread,
            1 => SyncScope::System,
            other => match usize::try_from(other)
                .ok()
                .and_then(|i| cx.sync_scopes.get(i))
            {
                Some(Some(scope)) => *scope,
                Some(None) => return Err(s.unsupported(Feature::SyncScope)),
                None => return Err(s.malformed(B, code)),
            },
        };
        Ok(Atomic { ordering, scope })
    }

    /// `PHI: [ty, (value, block)*, fmf?]`, values signed-relative. A block named twice reuses the
    /// value of its first entry, as LLVM's reader does.
    fn phi(
        &mut self,
        s: &Stream<'_>,
        cx: &mut FnCx<'_>,
        r: &mut Ops<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let code = FUNC_PHI;
        let bad = || s.malformed(B, code);
        let ty = bc_ty(s, cx, r.next(s)?)?;
        let n = r.left() / 2;
        let fmf = if !r.left().is_multiple_of(2) {
            let raw = r.ops.last().copied().ok_or_else(bad)?;
            if cx.types.float_bits(ty).is_none() {
                return Err(bad());
            }
            fast_math(raw).ok_or_else(bad)?
        } else {
            FastMath::NONE
        };
        let mut incoming = Vec::new();
        m.reserve(&mut incoming, n)?;
        let mut seen: HashMap<BlockId, Value> = HashMap::new();
        for _ in 0..n {
            m.tick(1)?;
            let raw = r.next(s)?;
            let bb = self.block(s, code, r.next(s)?)?;
            if let Some(&v) = seen.get(&bb) {
                incoming.push((v, bb));
                continue;
            }
            let rel = signed_vbr(raw);
            let id = i64::from(self.next)
                .checked_sub(rel)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| s.err(BitcodeError::BadValueRef))?;
            let v = self.value_at(s, cx, id, ty, m)?;
            m.charge(std::mem::size_of::<(BlockId, Value)>().saturating_mul(2))?;
            seen.insert(bb, v);
            incoming.push((v, bb));
        }
        let kind = InstKind::Phi {
            incoming: incoming.into_boxed_slice(),
            fmf,
        };
        self.push(s, cx, code, ty, kind, m)
    }

    /// `CALL: [paramattrs, cc, fmf?, fnty, callee, args...]`.
    fn call(
        &mut self,
        s: &Stream<'_>,
        cx: &mut FnCx<'_>,
        r: &mut Ops<'_>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let code = FUNC_CALL;
        let bad = || s.malformed(B, code);
        let attr_list = r.next(s)?;
        let cc_info = r.next(s)?;
        if cc_info & !CALL_KNOWN != 0 {
            return Err(bad());
        }
        let fmf = if cc_info & CALL_FMF != 0 {
            let f = fast_math(r.next(s)?).ok_or_else(bad)?;
            if f.is_empty() {
                return Err(bad());
            }
            f
        } else {
            FastMath::NONE
        };
        if cc_info & CALL_EXPLICIT_TYPE == 0 {
            // The typed-pointer spelling that derives the function type from the callee.
            return Err(s.unsupported(Feature::Instruction { code }));
        }
        let fn_bc = r.next(s)?;
        let fn_ty = bc_ty(s, cx, fn_bc)?;
        let (ret, params, vararg) = match cx.types.get(fn_ty) {
            Some(Type::Function {
                ret,
                params,
                vararg,
            }) => (*ret, params.clone(), *vararg),
            _ => return Err(s.err(BitcodeError::BadType)),
        };
        match (cc_info & CALL_CCONV_MASK) >> CALL_CCONV_SHIFT {
            CC_C | CC_FAST => {}
            other => {
                let cc = u32::try_from(other).unwrap_or(u32::MAX);
                return Err(s.unsupported(Feature::CallingConvention(cc)));
            }
        }
        let tail = match (
            cc_info & CALL_TAIL != 0,
            cc_info & CALL_MUSTTAIL != 0,
            cc_info & CALL_NOTAIL != 0,
        ) {
            (false, false, false) => TailKind::None,
            (true, false, false) => TailKind::Tail,
            (true, true, false) | (false, true, false) => TailKind::MustTail,
            (false, false, true) => TailKind::NoTail,
            _ => return Err(bad()),
        };
        let (callee, _) = self.pair(s, cx, r, m)?;
        let target = self.callee(cx, callee, fn_ty);
        let dropped = matches!(target, Target::Intrinsic(_, DeclClass::Drop(_)))
            || executes_nothing(cx.consts, callee, (ret, &params, vararg), cx.types);
        let mut args = Vec::new();
        m.reserve(&mut args, params.len())?;
        for &p in params.iter() {
            m.tick(1)?;
            match cx.types.get(p) {
                // A dropped marker's metadata operand (`noalias.scope.decl`) is checked to name
                // metadata and not kept.
                Some(Type::Metadata) if dropped => self.metadata_operand(s, cx, r)?,
                Some(Type::Label | Type::Metadata) => {
                    return Err(s.unsupported(Feature::Instruction { code }))
                }
                _ => args.push(self.typed(s, cx, r, p, m)?),
            }
        }
        if vararg {
            while r.left() > 0 {
                m.tick(1)?;
                let (v, _) = self.pair(s, cx, r, m)?;
                grow(m, &mut args)?;
                args.push(v);
            }
        }
        r.end(s)?;
        let list = match attr_list.checked_sub(1) {
            None => None,
            Some(i) => Some(
                usize::try_from(i)
                    .ok()
                    .and_then(|i| cx.attr_lists.get(i))
                    .ok_or_else(bad)?,
            ),
        };
        if dropped {
            if cx.keep_marker_operands {
                for a in &args {
                    if let Value::Const(c) = a {
                        grow(m, &mut self.marker_consts)?;
                        self.marker_consts.push(*c);
                    }
                }
            }
            return self.drop_call(s, cx, m);
        }
        let mut enums = SiteEnums::default();
        let attrs = match list {
            None => None,
            Some(list) => {
                let bc_params = cx.bc_types.params(fn_bc).unwrap_or(&[]);
                let bc_types = cx.bc_types;
                let pointee = |p: usize| {
                    bc_params
                        .get(p)
                        .and_then(|&t| bc_types.pointee(u64::from(t)))
                };
                let resolved = list
                    .resolve(cx.bc_types, args.len(), pointee, cx.keep_enums, m)?
                    .ok_or_else(bad)?;
                enums = resolved.enums;
                // The lists were charged as they were resolved; the box is the call's.
                m.charge(std::mem::size_of::<CallAttrs>())?;
                Some(Box::new(CallAttrs {
                    func: resolved.func,
                    ret: resolved.ret,
                    params: resolved.params.into_boxed_slice(),
                }))
            }
        };
        let callee = match target {
            Target::Callee(c) => c,
            Target::Intrinsic(f, class) => match intrinsics::bind(class, &args, cx.consts) {
                Ok(Bound::Air(op)) => Callee::Air { decl: f, op },
                Ok(Bound::Llvm(op)) => Callee::Llvm { decl: f, op },
                Ok(Bound::Unknown) => Callee::UnknownIntrinsic(f),
                // A declaration typed unlike the intrinsic's signature is another Metal
                // compiler's spelling of the operation, so the report names the declaration and
                // the function that calls it rather than a bitstream position alone.
                Err(IntrinsicError::SignatureMismatch) => {
                    let kind =
                        InputErrorKind::Intrinsic(IntrinsicError::DeclaredSignatureMismatch {
                            name: self.symbol_name(cx, f),
                        });
                    let at = Location::function(self.symbol_name(cx, self.func), None, None);
                    return Err(InputError::new(kind, at).into());
                }
                // An immediate this function computes, such as one of its parameters (OpenCL
                // C's `barrier(flags)` out of line): specialization binds the call where inlining
                // and constant propagation make it each call site's constant.
                Err(e) => match (e, class.template()) {
                    (IntrinsicError::ImmediateNotConstant, Some(template)) => {
                        Callee::Unbound { decl: f, template }
                    }
                    (e, _) => {
                        return Err(
                            InputError::new(InputErrorKind::Intrinsic(e), s.location()).into()
                        )
                    }
                },
            },
        };
        let call = Call {
            callee,
            args: args.into_boxed_slice(),
            fmf,
            tail,
            attrs,
            range: None,
        };
        // The arguments were charged as they were read; the payload's box is charged here.
        m.charge(std::mem::size_of::<Call>())?;
        if !enums.is_empty() {
            let id = InstId::new(
                u32::try_from(self.insts.len())
                    .map_err(|_| Error::LimitExceeded(Limit::InstructionCount))?,
            );
            grow(m, &mut self.call_enums)?;
            self.call_enums.push((id, enums));
        }
        self.push(s, cx, code, ret, InstKind::Call(Box::new(call)), m)
    }

    /// A relative metadata operand (`metadata !N` as a call argument): it must name module or
    /// function metadata.
    fn metadata_operand(
        &self,
        s: &Stream<'_>,
        cx: &FnCx<'_>,
        r: &mut Ops<'_>,
    ) -> Result<(), Error> {
        let rel = r.next32(s)?;
        let id = MdId::new(self.next.wrapping_sub(rel));
        match self.md(cx, id) {
            Some(_) => Ok(()),
            None => Err(s.err(BitcodeError::BadMetadata)),
        }
    }

    /// Classify a callee: a function whose type matches the call is direct, or an intrinsic
    /// when it is a declaration named with the stable `air.`/`llvm.` ABI prefix (classified
    /// once per declaration by `air::intrinsics`). Any other callee is a call through a pointer
    /// with the call's function type.
    /// The symbol of function `f`, for a diagnostic; `?` where the module has no such function
    /// or its name is not interned.
    fn symbol_name(&self, cx: &FnCx<'_>, f: FuncId) -> DiagName {
        let bytes = cx
            .funcs
            .get(f)
            .and_then(|f| cx.symbols.get(f.name))
            .unwrap_or(b"?");
        DiagName::new(String::from_utf8_lossy(bytes))
    }

    fn callee(&self, cx: &FnCx<'_>, callee: Value, fn_ty: TypeId) -> Target {
        let func = match callee {
            Value::Const(c) => match cx.consts.get(c).map(|c| &c.value) {
                Some(Const::FuncAddr(f)) => Some(*f),
                _ => None,
            },
            _ => None,
        };
        let indirect = Target::Callee(Callee::Indirect { callee, fn_ty });
        let Some((f, decl)) = func.and_then(|f| Some((f, cx.funcs.get(f)?))) else {
            return indirect;
        };
        if decl.ty != fn_ty {
            return indirect;
        }
        match cx.intrinsics.binary_search_by_key(&f, |&(g, _)| g) {
            Ok(i) => match cx.intrinsics.get(i) {
                Some(&(_, class)) => Target::Intrinsic(f, class),
                None => Target::Callee(Callee::Direct(f)),
            },
            Err(_) => Target::Callee(Callee::Direct(f)),
        }
    }

    /// Check the block structure, resolve forward references, and assemble the body.
    fn finish(mut self, s: &Stream<'_>, m: &mut Meter) -> Result<DecodedBody, Error> {
        let bad_ref = || s.err(BitcodeError::BadValueRef);
        let declared = self
            .declared
            .ok_or_else(|| s.malformed(B, FUNC_DECLAREBLOCKS))?;
        if self.blocks.len() as u64 != u64::from(declared)
            || self.block_start as usize != self.insts.len()
        {
            return Err(s.malformed(B, FUNC_DECLAREBLOCKS));
        }
        // Resolve forward references; the first failure wins.
        let limit = self.insts.len();
        let mut failed = false;
        {
            let FnDecoder {
                insts,
                forward,
                local,
                module_count,
                ..
            } = &mut self;
            for inst in insts.iter_mut() {
                m.tick(1)?;
                inst.kind.for_each_value_mut(|v| {
                    let Value::Inst(i) = *v else {
                        return;
                    };
                    if (i.raw() as usize) < limit {
                        return;
                    }
                    let k = (u32::MAX - i.raw()) as usize;
                    let resolved = forward.get(k).and_then(|&(id, ty)| {
                        let (value, t) = *local.get(id.checked_sub(*module_count)? as usize)?;
                        (t == ty && matches!(value, Value::Inst(_))).then_some(value)
                    });
                    match resolved {
                        Some(value) => *v = value,
                        None => failed = true,
                    }
                });
            }
        }
        if failed {
            return Err(bad_ref());
        }
        for &(bb, name) in &self.bb_names {
            if let Some(b) = self.blocks.get_mut(bb as usize) {
                b.name = Some(name);
            }
        }
        let mut blocks = Arena::new();
        for b in self.blocks {
            blocks.push(b, m)?;
        }
        // Each instruction's lists were charged as its record was read (and a call's boxes with
        // them), so the arena charges only its slots here, not `Body::push_inst`'s payload.
        let mut insts = Arena::new();
        for i in self.insts {
            insts.push(i, m)?;
        }
        Ok(DecodedBody {
            body: Body { blocks, insts },
            param_names: self.param_names,
            call_enums: self.call_enums,
            marker_consts: self.marker_consts,
        })
    }
}

/// What a call's callee operand names, before its arguments are read.
enum Target {
    /// A direct or indirect call.
    Callee(Callee),
    /// A call to a classified `air.*`/`llvm.*` declaration.
    Intrinsic(FuncId, DeclClass),
}

/// Whether a call of `callee` with signature `(ret, params, vararg)` is the one inline assembly MSL
/// accepts, `call void asm sideeffect "", ""()`: an empty template with an empty constraint string,
/// so no operand, result or clobber, and no flag but `sideeffect`. It assembles to no instruction
/// and names nothing it reads or writes. Its `sideeffect` only keeps an optimizer from removing it
/// or moving memory accesses across it (a shader generator writes it into loop bodies so that a
/// loop is kept): no value a race-free program computes depends on it. Decode drops the call, as it
/// drops the markers; any other inline assembly is kept, and refused where an entry reaches it.
fn executes_nothing(
    consts: &ConstTable,
    callee: Value,
    (ret, params, vararg): (TypeId, &[TypeId], bool),
    types: &TypeTable,
) -> bool {
    let empty = consts.inline_asm(callee).is_some_and(|a| {
        a.template.is_empty()
            && a.constraints.is_empty()
            && a.side_effects
            && !a.align_stack
            && !a.intel_dialect
            && !a.can_throw
    });
    empty && params.is_empty() && !vararg && matches!(types.get(ret), Some(Type::Void))
}

/// The value numbered `id` and its type: a module value below `module_count`, else a local one.
fn lookup(
    values: &[ModuleValue],
    consts: &ConstTable,
    module_count: u32,
    local: &[(Value, TypeId)],
    id: u32,
) -> Option<(Value, TypeId)> {
    match id.checked_sub(module_count) {
        None => {
            let c = values.get(id as usize)?.id?;
            Some((Value::Const(c), consts.ty(c)?))
        }
        Some(l) => local.get(l as usize).copied(),
    }
}

/// The erased IR type of bitcode type `idx`.
fn bc_ty(s: &Stream<'_>, cx: &FnCx<'_>, idx: u64) -> Result<TypeId, Error> {
    cx.bc_types
        .ty(idx)
        .ok_or_else(|| s.err(BitcodeError::BadType))
}

/// The remaining operands as aggregate indices (at least one).
fn indices(s: &Stream<'_>, r: &mut Ops<'_>, m: &mut Meter) -> Result<Box<[u32]>, Error> {
    if r.left() == 0 {
        return Err(s.malformed(B, r.code));
    }
    let mut out = Vec::new();
    m.reserve(&mut out, r.left())?;
    while r.left() > 0 {
        out.push(r.next32(s)?);
    }
    Ok(out.into_boxed_slice())
}

/// The type an aggregate index path reaches.
fn aggregate_leaf(types: &TypeTable, agg: TypeId, indices: &[u32]) -> Option<TypeId> {
    let mut cur = agg;
    for &i in indices {
        cur = match types.get(cur)? {
            Type::Struct { fields, .. } => *fields.get(i as usize)?,
            Type::Array { elem, len } if u64::from(i) < *len => *elem,
            _ => return None,
        };
    }
    Some(cur)
}

/// A shuffle mask constant of `lanes` lanes; `None` lanes are poison (an `undef` lane is poison
/// in a mask, as in LLVM).
fn shuffle_mask(
    consts: &ConstTable,
    mask: Value,
    lanes: u32,
    m: &mut Meter,
) -> Result<Option<Box<[Option<u32>]>>, Error> {
    let Value::Const(c) = mask else {
        return Ok(None);
    };
    let Some(c) = consts.get(c) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    m.reserve(&mut out, lanes as usize)?;
    match &c.value {
        Const::Null => out.resize(lanes as usize, Some(0)),
        Const::Undef | Const::Poison => out.resize(lanes as usize, None),
        Const::Aggregate(elems) if elems.len() == lanes as usize => {
            for &e in elems.iter() {
                let lane = match consts.get(e).map(|e| &e.value) {
                    Some(Const::Int(v)) => match u32::try_from(*v) {
                        Ok(v) => Some(v),
                        Err(_) => return Ok(None),
                    },
                    Some(Const::Undef | Const::Poison) => None,
                    _ => return Ok(None),
                };
                out.push(lane);
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(out.into_boxed_slice()))
}

/// LLVM's bitcode fast-math flags (`getDecodedFastMathFlags`). The legacy "unsafe algebra" bit
/// sets every flag. `None` for an unknown bit.
pub(crate) fn fast_math(raw: u64) -> Option<FastMath> {
    if raw & !0xFF != 0 {
        return None;
    }
    let bit = |n: u32, f: FastMath| {
        if raw & (1 << n) != 0 {
            f
        } else {
            FastMath::NONE
        }
    };
    let all = FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |acc, &(f, _)| acc.union(f));
    Some(
        bit(0, all)
            .union(bit(1, FastMath::NNAN))
            .union(bit(2, FastMath::NINF))
            .union(bit(3, FastMath::NSZ))
            .union(bit(4, FastMath::ARCP))
            .union(bit(5, FastMath::CONTRACT))
            .union(bit(6, FastMath::AFN))
            .union(bit(7, FastMath::REASSOC)),
    )
}

/// An atomic ordering (`NotAtomic` is not one).
fn ordering(raw: u64) -> Option<Ordering> {
    Some(match raw {
        1 => Ordering::Unordered,
        2 => Ordering::Monotonic,
        3 => Ordering::Acquire,
        4 => Ordering::Release,
        5 => Ordering::AcqRel,
        6 => Ordering::SeqCst,
        _ => return None,
    })
}

/// `AtomicCmpXchgInst::getStrongestFailureOrdering`.
fn strongest_failure(success: Ordering) -> Ordering {
    match success {
        Ordering::Release | Ordering::Monotonic | Ordering::Unordered => Ordering::Monotonic,
        Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
        Ordering::SeqCst => Ordering::SeqCst,
    }
}

/// An `atomicrmw` operator (`RMW_*`); `None` for one the IR does not model (`usub_cond`,
/// `usub_sat`, `fmaximum`, `fminimum`, …).
fn rmw_op(raw: u64) -> Option<AtomicRmwOp> {
    Some(match raw {
        0 => AtomicRmwOp::Xchg,
        1 => AtomicRmwOp::Add,
        2 => AtomicRmwOp::Sub,
        3 => AtomicRmwOp::And,
        4 => AtomicRmwOp::Nand,
        5 => AtomicRmwOp::Or,
        6 => AtomicRmwOp::Xor,
        7 => AtomicRmwOp::Max,
        8 => AtomicRmwOp::Min,
        9 => AtomicRmwOp::UMax,
        10 => AtomicRmwOp::UMin,
        11 => AtomicRmwOp::FAdd,
        12 => AtomicRmwOp::FSub,
        13 => AtomicRmwOp::FMax,
        14 => AtomicRmwOp::FMin,
        15 => AtomicRmwOp::UIncWrap,
        16 => AtomicRmwOp::UDecWrap,
        _ => return None,
    })
}

/// A `log2 + 1` alignment field (0 = none). `None` for an out-of-range value.
fn align(v: u64) -> Option<Option<Align>> {
    match v.checked_sub(1) {
        None => Some(None),
        Some(log2) => Align::from_log2(u8::try_from(log2).ok()?).map(Some),
    }
}

/// A 0/1 flag operand.
fn flag(s: &Stream<'_>, code: u32, v: u64) -> Result<bool, Error> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(s.malformed(B, code)),
    }
}
