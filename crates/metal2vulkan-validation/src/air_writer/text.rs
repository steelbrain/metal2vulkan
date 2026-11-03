//! The typed-pointer LLVM text spelling.
//!
//! The syntax is the LLVM 14 dialect `xcrun metal-as` parses: typed pointers
//! (`float addrspace(1)*`), and the pre-`memory(...)` spellings of memory effects (`readonly`,
//! `argmemonly`, ...). Every call spells its callee's full function type, so vararg and
//! pointer-returning callees need no special case. Values with a name in the IR keep it; the rest
//! are numbered in LLVM's order (unnamed parameters, then per block its label and its non-void
//! instructions).

use std::fmt::Write as _;

use metal2vulkan_air::bitcode::{EnumAttrTable, EnumAttrs, SiteEnums};
use metal2vulkan_air::ir::{
    Align, Atomic, CallAttrs, CallConv, Callee, Const, ConstExpr, ConstId, DataLayout, FastMath,
    FnAttrs, FuncId, Function, Idx, Inst, InstId, InstKind, IntFlags, Linkage, LoopHints, Md, MdId,
    MemoryEffects, ModRef, Module, ParamAttrs, RangeMd, Section, Sym, SyncScope, TailKind, Type,
    TypeId, UnnamedAddr, Unroll, Value, Visibility,
};

use super::infer::{CUse, Op, Typing};
use super::terms::{Terms, N};
use super::{Unspellable, WriteError};

/// `"..."` with LLVM's `\XX` escapes.
pub(super) fn quoted(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 2);
    out.push('"');
    for &b in bytes {
        if (b.is_ascii_graphic() || b == b' ') && b != b'"' && b != b'\\' {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "\\{b:02X}");
        }
    }
    out.push('"');
    out
}

pub(super) struct Printer<'a, 'm> {
    pub m: &'m Module,
    /// The enum attributes the IR does not carry, by function and call site.
    pub enums: &'a EnumAttrTable,
    pub t: &'a mut Terms<'m>,
    pub typing: &'a Typing,
    pub out: String,
    /// Metadata nodes generated for instruction attachments, printed after the module's.
    pub extra_md: Vec<String>,
    /// The next free metadata number.
    pub next_md: u32,
}

/// Append formatted text; writing to a `String` cannot fail.
macro_rules! w {
    ($out:expr, $($arg:tt)*) => {{
        let _ = write!($out, $($arg)*);
    }};
}

/// The names of one function's values.
struct Names {
    func: FuncId,
    args: Vec<String>,
    blocks: Vec<String>,
    insts: Vec<Option<String>>,
}

fn bad(what: impl Into<String>) -> WriteError {
    WriteError::Malformed(what.into())
}

impl Printer<'_, '_> {
    pub fn module(&mut self) -> Result<(), WriteError> {
        let parts = self.m.parts();
        self.out
            .push_str("; Written by metal2vulkan-validation air_writer (typed pointers).\n");
        if let Some(s) = parts.source_filename {
            let s = self.sym(s)?;
            w!(self.out, "source_filename = {s}\n");
        }
        w!(
            self.out,
            "target datalayout = \"{}\"\n",
            data_layout(&parts.data_layout)
        );
        if let Some(s) = parts.triple {
            let s = self.sym(s)?;
            w!(self.out, "target triple = {s}\n");
        }
        self.out.push('\n');
        for (id, ty) in self.m.types().iter() {
            if matches!(
                ty,
                Type::Struct { name: Some(_), .. } | Type::OpaqueStruct { name: Some(_) }
            ) {
                let name = self.t.struct_name(id)?;
                let body = self.t.struct_definition(id)?;
                w!(self.out, "{name} = type {body}\n");
            }
        }
        for (id, g) in parts.globals.iter() {
            let name = self.sym(g.name)?;
            if name == "\"\"" {
                return Err(WriteError::Unsupported(Unspellable::UnnamedGlobal));
            }
            w!(self.out, "@{name} = ");
            if g.init.is_none() && g.linkage == Linkage::External {
                self.out.push_str("external ");
            } else {
                linkage(&mut self.out, g.linkage);
            }
            symbol_flags(&mut self.out, g.visibility, g.unnamed_addr);
            if g.space.raw() != 0 {
                w!(self.out, "addrspace({}) ", g.space.raw());
            }
            if g.externally_initialized {
                self.out.push_str("externally_initialized ");
            }
            let node = *self
                .typing
                .globals
                .get(id.index())
                .ok_or_else(|| bad("global"))?;
            let value = self.t.pointee(node)?;
            let ty = self.t.text(value)?;
            let kind = if g.constant { "constant" } else { "global" };
            w!(self.out, "{kind} {ty}");
            if let (Some(c), Some(Some(cu))) = (g.init, self.typing.inits.get(id.index())) {
                let v = self.cvalue(c, cu)?;
                w!(self.out, " {v}");
            }
            self.section(g.section, ",")?;
            align(&mut self.out, g.align);
            self.out.push('\n');
        }
        for (id, f) in parts.funcs.iter() {
            self.function(id, f)?;
        }
        self.out.push('\n');
        for n in &parts.named_md {
            let name = self.m.symbols().get(n.name).ok_or_else(|| bad("md name"))?;
            let mut ops = Vec::with_capacity(n.ops.len());
            for &op in n.ops.iter() {
                ops.push(self.md_ref(op)?);
            }
            w!(self.out, "!{} = !{{{}}}\n", md_name(name), ops.join(", "));
        }
        for id in self.reachable_md()? {
            let Some(Md::Node { distinct, ops }) = self.m.md(id) else {
                continue;
            };
            let mut text = Vec::with_capacity(ops.len());
            for op in ops.iter() {
                text.push(match op {
                    Some(op) => self.md_ref(*op)?,
                    None => "null".to_owned(),
                });
            }
            let d = if *distinct { "distinct " } else { "" };
            w!(self.out, "!{} = {d}!{{{}}}\n", id.raw(), text.join(", "));
        }
        for line in &self.extra_md {
            self.out.push_str(line);
            self.out.push('\n');
        }
        Ok(())
    }

    /// Every metadata node reachable from named metadata, in id order.
    fn reachable_md(&self) -> Result<Vec<MdId>, WriteError> {
        let mut seen = vec![false; self.m.parts().metadata.len()];
        let mut stack: Vec<MdId> = self
            .m
            .named_md()
            .iter()
            .flat_map(|n| n.ops.iter().copied())
            .collect();
        while let Some(id) = stack.pop() {
            let slot = seen.get_mut(id.index()).ok_or_else(|| bad("metadata id"))?;
            if *slot {
                continue;
            }
            *slot = true;
            match self.m.md(id) {
                Some(Md::Node { ops, .. }) => stack.extend(ops.iter().flatten().copied()),
                Some(Md::Debug) => return Err(WriteError::Unsupported(Unspellable::DebugMetadata)),
                _ => {}
            }
        }
        Ok(seen
            .iter()
            .enumerate()
            .filter(|(_, s)| **s)
            .filter_map(|(i, _)| MdId::from_index(i))
            .collect())
    }

    fn md_ref(&mut self, id: MdId) -> Result<String, WriteError> {
        match self.m.md(id) {
            Some(Md::String(s)) => {
                let bytes = self.m.symbols().get(*s).ok_or_else(|| bad("md string"))?;
                Ok(format!("!{}", quoted(bytes)))
            }
            Some(Md::Value(c)) => {
                let cu = self
                    .typing
                    .md_values
                    .get(id.index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| bad("md value"))?;
                self.typed_cvalue(*c, cu)
            }
            Some(Md::Node { .. }) => Ok(format!("!{}", id.raw())),
            Some(Md::Debug) => Err(WriteError::Unsupported(Unspellable::DebugMetadata)),
            None => Err(bad("metadata id out of range")),
        }
    }

    fn sym(&self, s: Sym) -> Result<String, WriteError> {
        self.m
            .symbols()
            .get(s)
            .map(quoted)
            .ok_or_else(|| bad(format!("symbol {s:?}")))
    }

    /// `section "..."`, after `sep` (`,` for a global, nothing for a function).
    fn section(&mut self, s: Option<Section>, sep: &str) -> Result<(), WriteError> {
        let name = match s {
            Some(Section::FcInitializer) => quoted(b"air.fc_initializer"),
            Some(Section::StaticInit) => quoted(b"air.static_init"),
            Some(Section::Other(s)) => self.sym(s)?,
            None => return Ok(()),
        };
        w!(self.out, "{sep} section {name}");
        Ok(())
    }

    fn names(&self, id: FuncId, f: &Function) -> Result<Names, WriteError> {
        let mut next = 0u32;
        let mut take = |name: Option<Sym>| -> Result<String, WriteError> {
            match name {
                Some(n) => self.sym(n),
                None => {
                    let s = next.to_string();
                    next += 1;
                    Ok(s)
                }
            }
        };
        let mut args = Vec::with_capacity(f.params.len());
        for p in f.params.iter() {
            args.push(take(p.name)?);
        }
        let mut blocks = Vec::new();
        let mut insts = Vec::new();
        if let Some(body) = &f.body {
            for (_, block) in body.blocks.iter() {
                blocks.push(take(block.name)?);
                for i in block.inst_ids() {
                    let inst = body.insts.get(i).ok_or_else(|| bad("instruction"))?;
                    let void = matches!(self.m.types().get(inst.ty), Some(Type::Void));
                    insts.push(if void { None } else { Some(take(None)?) });
                }
            }
        }
        Ok(Names {
            func: id,
            args,
            blocks,
            insts,
        })
    }

    fn function(&mut self, id: FuncId, f: &Function) -> Result<(), WriteError> {
        let name = self.sym(f.name)?;
        let names = self.names(id, f)?;
        let node = *self
            .typing
            .funcs
            .get(id.index())
            .ok_or_else(|| bad("function"))?;
        let (ret, params, vararg) = self.t.signature(node)?;
        let kw = if f.body.is_some() {
            "define"
        } else {
            "declare"
        };
        w!(self.out, "\n{kw} ");
        linkage(&mut self.out, f.linkage);
        symbol_flags(&mut self.out, f.visibility, UnnamedAddr::Significant);
        if f.cconv == CallConv::Fast {
            self.out.push_str("fastcc ");
        }
        let empty = SiteEnums::default();
        let enums = self.enums.func(id).unwrap_or(&empty);
        let ret_attrs = self.param_attrs(&f.ret_attrs, enums.ret, None)?;
        let ret_ty = self.t.text(ret)?;
        w!(self.out, "{ret_attrs}{ret_ty} @{name}(");
        for (i, &p) in params.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let mut text = self.t.text(p)?;
            if let Some(param) = f.params.get(i) {
                let attrs = self.param_attrs(&param.attrs, enums.param(i), Some(p))?;
                if !attrs.is_empty() {
                    text.push(' ');
                    text.push_str(attrs.trim_end());
                }
            }
            if f.body.is_some() {
                let n = names.args.get(i).ok_or_else(|| bad("argument"))?;
                w!(text, " %{n}");
            }
            self.out.push_str(&text);
        }
        if vararg {
            self.out
                .push_str(if params.is_empty() { "..." } else { ", ..." });
        }
        self.out.push(')');
        let unnamed = f.unnamed_addr.keyword();
        if !unnamed.is_empty() {
            w!(self.out, " {unnamed}");
        }
        let attrs = self.fn_attrs(&f.attrs, enums.func)?;
        self.out.push_str(&attrs);
        self.section(f.section, "")?;
        if let Some(a) = f.align {
            w!(self.out, " align {}", a.bytes());
        }
        let Some(body) = &f.body else {
            self.out.push('\n');
            return Ok(());
        };
        self.out.push_str(" {\n");
        let typing = self
            .typing
            .bodies
            .get(id.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| bad("body typing"))?;
        for (b, block) in body.blocks.iter() {
            if b.index() > 0 || block.name.is_some() {
                let label = names.blocks.get(b.index()).ok_or_else(|| bad("block"))?;
                w!(self.out, "\n{label}:\n");
            }
            for i in block.inst_ids() {
                let inst = body.insts.get(i).ok_or_else(|| bad("instruction"))?;
                let ops = typing.ops.get(i.index()).ok_or_else(|| bad("operands"))?;
                let res = typing.results.get(i.index()).copied().flatten();
                let callee = typing.callee.get(i.index()).copied().flatten();
                let line = self.inst(inst, ops, res, callee, &names, i.index())?;
                w!(self.out, "  {line}\n");
            }
        }
        self.out.push_str("}\n");
        Ok(())
    }

    fn fn_attrs(&mut self, a: &FnAttrs, enums: EnumAttrs) -> Result<String, WriteError> {
        let mut out = String::new();
        if a.convergent {
            out.push_str(" convergent");
        }
        if let Some(m) = a.memory {
            for word in memory_spelling(m)? {
                w!(out, " {word}");
            }
        }
        for word in enum_spelling(enums)? {
            w!(out, " {word}");
        }
        for &(k, v) in a.strings.iter() {
            w!(out, " {}", self.string_attr(k, v)?);
        }
        Ok(out)
    }

    fn string_attr(&self, k: Sym, v: Sym) -> Result<String, WriteError> {
        let value = self.m.symbols().get(v).ok_or_else(|| bad("attr value"))?;
        let key = self.sym(k)?;
        Ok(if value.is_empty() {
            key
        } else {
            format!("{key}={}", quoted(value))
        })
    }

    /// Parameter or return attributes, each followed by a space. `param` is the parameter's
    /// term, whose pointee spells `byval`/`sret`.
    fn param_attrs(
        &mut self,
        a: &ParamAttrs,
        enums: EnumAttrs,
        param: Option<N>,
    ) -> Result<String, WriteError> {
        let mut out = String::new();
        if let Some(al) = a.align {
            w!(out, "align {} ", al.bytes());
        }
        if let Some(n) = a.dereferenceable {
            w!(out, "dereferenceable({n}) ");
        }
        if let Some(n) = a.dereferenceable_or_null {
            w!(out, "dereferenceable_or_null({n}) ");
        }
        if a.noalias {
            out.push_str("noalias ");
        }
        if let Some(m) = a.access {
            out.push_str(match m {
                ModRef::NoModRef => "readnone ",
                ModRef::Ref => "readonly ",
                ModRef::Mod => "writeonly ",
                ModRef::ModRef => "",
            });
        }
        for (kw, ty) in [("byval", a.byval), ("sret", a.sret)] {
            if ty.is_some() {
                let p = param.ok_or(WriteError::Unsupported(Unspellable::ReturnTypeAttribute))?;
                let pointee = self.t.pointee(p)?;
                w!(out, "{kw}({}) ", self.t.text(pointee)?);
            }
        }
        if a.nocapture {
            out.push_str("nocapture ");
        }
        for word in enum_spelling(enums)? {
            w!(out, "{word} ");
        }
        for &(k, v) in a.strings.iter() {
            w!(out, "{} ", self.string_attr(k, v)?);
        }
        Ok(out)
    }

    fn value(&mut self, v: Value, op: &Op, names: &Names) -> Result<String, WriteError> {
        match v {
            Value::Arg(a) => names
                .args
                .get(a.index())
                .map(|n| format!("%{n}"))
                .ok_or_else(|| bad("argument")),
            Value::Inst(i) => names
                .insts
                .get(i.index())
                .cloned()
                .flatten()
                .map(|n| format!("%{n}"))
                .ok_or_else(|| bad("instruction value")),
            Value::Const(c) => {
                let cu = op.cuse.as_ref().ok_or_else(|| bad("constant use"))?;
                self.cvalue(c, cu)
            }
        }
    }

    /// `<type> <value>` of operand `k`.
    fn typed(
        &mut self,
        v: Value,
        ops: &[Op],
        k: usize,
        names: &Names,
    ) -> Result<String, WriteError> {
        let op = ops.get(k).ok_or_else(|| bad("operand index"))?;
        let ty = self.t.text(op.node)?;
        let v = self.value(v, op, names)?;
        Ok(format!("{ty} {v}"))
    }

    fn typed_cvalue(&mut self, c: ConstId, cu: &CUse) -> Result<String, WriteError> {
        let ty = self.t.text(cu.node)?;
        let v = self.cvalue(c, cu)?;
        Ok(format!("{ty} {v}"))
    }

    fn cvalue(&mut self, c: ConstId, cu: &CUse) -> Result<String, WriteError> {
        let k = self.m.consts().get(c).ok_or_else(|| bad("constant"))?;
        let types = self.m.types();
        Ok(match &k.value {
            Const::Int(v) => int_text(types.get(k.ty), *v)?,
            Const::Float(bits) => float_text(types.get(k.ty), *bits)?,
            Const::Null => match types.get(k.ty) {
                Some(Type::Ptr { .. }) => "null".into(),
                _ => "zeroinitializer".into(),
            },
            Const::Undef => "undef".into(),
            Const::Poison => "poison".into(),
            // A call's callee: `asm <flags> "template", "constraints"`.
            Const::InlineAsm(a) => a.spelling(),
            Const::GlobalAddr(g) => {
                let g = self.m.global(*g).ok_or_else(|| bad("global"))?;
                format!("@{}", self.sym(g.name)?)
            }
            Const::FuncAddr(f) => {
                let f = self.m.func(*f).ok_or_else(|| bad("function"))?;
                format!("@{}", self.sym(f.name)?)
            }
            Const::Aggregate(elems) => {
                let mut parts = Vec::with_capacity(elems.len());
                for (i, &e) in elems.iter().enumerate() {
                    parts.push(match cu.kids.get(i) {
                        Some(kid) => self.typed_cvalue(e, kid)?,
                        None => {
                            let ty = self.m.consts().ty(e).ok_or_else(|| bad("constant"))?;
                            let ty = self.t.erased_text(ty)?;
                            let leaf = CUse {
                                node: cu.node,
                                kids: Box::new([]),
                            };
                            format!("{ty} {}", self.cvalue(e, &leaf)?)
                        }
                    });
                }
                let body = parts.join(", ");
                match types.get(k.ty) {
                    Some(Type::Vector { .. }) => format!("<{body}>"),
                    Some(Type::Array { .. }) => format!("[{body}]"),
                    Some(Type::Struct { packed: true, .. }) => format!("<{{ {body} }}>"),
                    Some(Type::Struct { .. }) => format!("{{ {body} }}"),
                    _ => return Err(bad("aggregate of a scalar type")),
                }
            }
            Const::Expr(ConstExpr::Cast { op, ops: [o] }) => {
                let kid = cu.kids.first().ok_or_else(|| bad("cast operand"))?;
                let from = self.typed_cvalue(*o, kid)?;
                let to = self.t.text(cu.node)?;
                format!("{} ({from} to {to})", op.mnemonic())
            }
            Const::Expr(ConstExpr::BinOp { op, flags, ops }) => {
                let mut parts = Vec::with_capacity(2);
                for (i, &o) in ops.iter().enumerate() {
                    let kid = cu.kids.get(i).ok_or_else(|| bad("binop operand"))?;
                    parts.push(self.typed_cvalue(o, kid)?);
                }
                format!(
                    "{}{} ({})",
                    op.mnemonic(),
                    int_flags(*flags),
                    parts.join(", ")
                )
            }
            Const::Expr(ConstExpr::Gep { inbounds, ops, .. }) => {
                let base = cu.kids.first().ok_or_else(|| bad("GEP base"))?;
                let src = self.t.pointee(base.node)?;
                let src = self.t.text(src)?;
                let mut parts = vec![src];
                for (i, &o) in ops.iter().enumerate() {
                    let kid = cu.kids.get(i).ok_or_else(|| bad("GEP operand"))?;
                    parts.push(self.typed_cvalue(o, kid)?);
                }
                let ib = if *inbounds { " inbounds" } else { "" };
                format!("getelementptr{ib} ({})", parts.join(", "))
            }
        })
    }

    fn block(&self, names: &Names, b: metal2vulkan_air::ir::BlockId) -> Result<String, WriteError> {
        names
            .blocks
            .get(b.index())
            .map(|n| format!("label %{n}"))
            .ok_or_else(|| bad("block"))
    }

    fn inst(
        &mut self,
        inst: &Inst,
        ops: &[Op],
        res: Option<N>,
        callee: Option<N>,
        names: &Names,
        index: usize,
    ) -> Result<String, WriteError> {
        let mut out = String::new();
        if let Some(Some(n)) = names.insts.get(index) {
            w!(out, "%{n} = ");
        }
        let res_ty = |p: &mut Self| -> Result<String, WriteError> {
            p.t.text(res.ok_or_else(|| bad("result"))?)
        };
        match &inst.kind {
            InstKind::BinOp {
                op,
                flags,
                fmf,
                a,
                b,
            } => {
                let a = self.typed(*a, ops, 0, names)?;
                let b = self.value(*b, ops.get(1).ok_or_else(|| bad("operand"))?, names)?;
                w!(
                    out,
                    "{}{}{} {a}, {b}",
                    op.mnemonic(),
                    int_flags(*flags),
                    fmf_text(*fmf)
                );
            }
            InstKind::FNeg { v, fmf } => {
                let v = self.typed(*v, ops, 0, names)?;
                w!(out, "fneg{} {v}", fmf_text(*fmf));
            }
            InstKind::Cast { op, v } => {
                let v = self.typed(*v, ops, 0, names)?;
                w!(out, "{} {v} to {}", op.mnemonic(), res_ty(self)?);
            }
            InstKind::ICmp { pred, a, b } => {
                let a = self.typed(*a, ops, 0, names)?;
                let b = self.value(*b, ops.get(1).ok_or_else(|| bad("operand"))?, names)?;
                w!(out, "icmp {} {a}, {b}", pred.mnemonic());
            }
            InstKind::FCmp { pred, fmf, a, b } => {
                let a = self.typed(*a, ops, 0, names)?;
                let b = self.value(*b, ops.get(1).ok_or_else(|| bad("operand"))?, names)?;
                w!(out, "fcmp{} {} {a}, {b}", fmf_text(*fmf), pred.mnemonic());
            }
            InstKind::Select { c, t, f, fmf } => {
                let c = self.typed(*c, ops, 0, names)?;
                let t = self.typed(*t, ops, 1, names)?;
                let f = self.typed(*f, ops, 2, names)?;
                w!(out, "select{} {c}, {t}, {f}", fmf_text(*fmf));
            }
            InstKind::Freeze(v) => {
                let v = self.typed(*v, ops, 0, names)?;
                w!(out, "freeze {v}");
            }
            InstKind::ExtractElement { vec, idx } => {
                let vec = self.typed(*vec, ops, 0, names)?;
                let idx = self.typed(*idx, ops, 1, names)?;
                w!(out, "extractelement {vec}, {idx}");
            }
            InstKind::InsertElement { vec, elt, idx } => {
                let vec = self.typed(*vec, ops, 0, names)?;
                let elt = self.typed(*elt, ops, 1, names)?;
                let idx = self.typed(*idx, ops, 2, names)?;
                w!(out, "insertelement {vec}, {elt}, {idx}");
            }
            InstKind::ShuffleVector { a, b, mask } => {
                let a = self.typed(*a, ops, 0, names)?;
                let b = self.typed(*b, ops, 1, names)?;
                let lanes: Vec<String> = mask
                    .iter()
                    .map(|m| match m {
                        Some(m) => format!("i32 {m}"),
                        None => "i32 undef".to_owned(),
                    })
                    .collect();
                w!(
                    out,
                    "shufflevector {a}, {b}, <{} x i32> <{}>",
                    mask.len(),
                    lanes.join(", ")
                );
            }
            InstKind::ExtractValue { agg, indices } => {
                let agg = self.typed(*agg, ops, 0, names)?;
                w!(out, "extractvalue {agg}{}", index_list(indices));
            }
            InstKind::InsertValue { agg, val, indices } => {
                let agg = self.typed(*agg, ops, 0, names)?;
                let val = self.typed(*val, ops, 1, names)?;
                w!(out, "insertvalue {agg}, {val}{}", index_list(indices));
            }
            InstKind::Alloca {
                count, align: a, ..
            } => {
                let r = res.ok_or_else(|| bad("alloca result"))?;
                let pointee = self.t.pointee(r)?;
                let ty = self.t.text(pointee)?;
                let count = self.typed(*count, ops, 0, names)?;
                w!(out, "alloca {ty}, {count}");
                align(&mut out, *a);
                let space = self.m.data_layout().alloca_space;
                if space != 0 {
                    w!(out, ", addrspace({space})");
                }
            }
            InstKind::Load {
                ptr,
                align: a,
                volatile,
                atomic,
                range,
            } => {
                let ty = res_ty(self)?;
                let ptr = self.typed(*ptr, ops, 0, names)?;
                out.push_str("load");
                atomic_prefix(&mut out, *atomic, *volatile);
                w!(out, " {ty}, {ptr}");
                atomic_suffix(&mut out, *atomic);
                align(&mut out, *a);
                if let Some(r) = range {
                    let md = self.range_md(r, inst.ty)?;
                    w!(out, ", !range {md}");
                }
            }
            InstKind::Store {
                val,
                ptr,
                align: a,
                volatile,
                atomic,
            } => {
                let val = self.typed(*val, ops, 0, names)?;
                let ptr = self.typed(*ptr, ops, 1, names)?;
                out.push_str("store");
                atomic_prefix(&mut out, *atomic, *volatile);
                w!(out, " {val}, {ptr}");
                atomic_suffix(&mut out, *atomic);
                align(&mut out, *a);
            }
            InstKind::Gep {
                base,
                indices,
                inbounds,
                ..
            } => {
                let base_op = ops.first().ok_or_else(|| bad("GEP base"))?;
                let src = self.t.pointee(base_op.node)?;
                let src = self.t.text(src)?;
                let mut parts = vec![src, self.typed(*base, ops, 0, names)?];
                for (k, &i) in indices.iter().enumerate() {
                    parts.push(self.typed(i, ops, k + 1, names)?);
                }
                let ib = if *inbounds { " inbounds" } else { "" };
                w!(out, "getelementptr{ib} {}", parts.join(", "));
            }
            InstKind::AtomicRmw {
                op,
                ptr,
                val,
                atomic,
                volatile,
                align: a,
            } => {
                let ptr = self.typed(*ptr, ops, 0, names)?;
                let val = self.typed(*val, ops, 1, names)?;
                out.push_str("atomicrmw");
                if *volatile {
                    out.push_str(" volatile");
                }
                w!(out, " {} {ptr}, {val}", op.mnemonic());
                atomic_suffix(&mut out, Some(*atomic));
                align(&mut out, *a);
            }
            InstKind::CmpXchg {
                ptr,
                cmp,
                new,
                success,
                failure,
                weak,
                volatile,
                align: a,
            } => {
                let ptr = self.typed(*ptr, ops, 0, names)?;
                let cmp = self.typed(*cmp, ops, 1, names)?;
                let new = self.typed(*new, ops, 2, names)?;
                out.push_str("cmpxchg");
                if *weak {
                    out.push_str(" weak");
                }
                if *volatile {
                    out.push_str(" volatile");
                }
                w!(out, " {ptr}, {cmp}, {new}");
                atomic_suffix(&mut out, Some(*success));
                w!(out, " {}", failure.mnemonic());
                align(&mut out, *a);
            }
            InstKind::Fence(a) => {
                out.push_str("fence");
                atomic_suffix(&mut out, Some(*a));
            }
            InstKind::Call(call) => {
                let f = callee.ok_or_else(|| bad("call typing"))?;
                match call.tail {
                    TailKind::None => {}
                    TailKind::Tail => out.push_str("tail "),
                    TailKind::MustTail => out.push_str("musttail "),
                    TailKind::NoTail => out.push_str("notail "),
                }
                w!(out, "call{}", fmf_text(call.fmf));
                let (callee_text, first_arg) = match &call.callee {
                    Callee::Indirect { callee, .. } => {
                        let op = ops.first().ok_or_else(|| bad("callee"))?;
                        (self.value(*callee, op, names)?, 1)
                    }
                    other => {
                        let decl = other.func().ok_or_else(|| bad("callee"))?;
                        let decl = self.m.func(decl).ok_or_else(|| bad("callee"))?;
                        if decl.cconv == CallConv::Fast {
                            out.push_str(" fastcc");
                        }
                        (format!("@{}", self.sym(decl.name)?), 0)
                    }
                };
                let no_attrs = CallAttrs::default();
                let attrs = call.attrs.as_deref().unwrap_or(&no_attrs);
                let empty = SiteEnums::default();
                let enums = u32::try_from(index)
                    .ok()
                    .and_then(|i| self.enums.call(names.func, InstId::new(i)))
                    .unwrap_or(&empty);
                let r = self.param_attrs(&attrs.ret, enums.ret, None)?;
                if !r.is_empty() {
                    w!(out, " {}", r.trim_end());
                }
                let fn_ty = self.t.text(f)?;
                w!(out, " {fn_ty} {callee_text}(");
                for (i, &a) in call.args.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let op = ops.get(first_arg + i).ok_or_else(|| bad("argument"))?;
                    let ty = self.t.text(op.node)?;
                    let none = ParamAttrs::default();
                    let pa = attrs.params.get(i).unwrap_or(&none);
                    let spelled = self.param_attrs(pa, enums.param(i), Some(op.node))?;
                    let v = self.value(a, op, names)?;
                    w!(out, "{ty} {spelled}{v}");
                }
                out.push(')');
                let fa = self.fn_attrs(&attrs.func, enums.func)?;
                out.push_str(&fa);
                if let Some(r) = &call.range {
                    let md = self.range_md(r, inst.ty)?;
                    w!(out, ", !range {md}");
                }
            }
            InstKind::Phi { incoming, fmf } => {
                let ty = res_ty(self)?;
                w!(out, "phi{} {ty} ", fmf_text(*fmf));
                let mut parts = Vec::with_capacity(incoming.len());
                for (k, &(v, b)) in incoming.iter().enumerate() {
                    let op = ops.get(k).ok_or_else(|| bad("incoming"))?;
                    let v = self.value(v, op, names)?;
                    let b = names.blocks.get(b.index()).ok_or_else(|| bad("block"))?;
                    parts.push(format!("[ {v}, %{b} ]"));
                }
                out.push_str(&parts.join(", "));
            }
            InstKind::Ret(v) => match v {
                Some(v) => {
                    let v = self.typed(*v, ops, 0, names)?;
                    w!(out, "ret {v}");
                }
                None => out.push_str("ret void"),
            },
            InstKind::Br { dest, hints } => {
                w!(out, "br {}", self.block(names, *dest)?);
                self.hints(&mut out, hints);
            }
            InstKind::CondBr { c, t, f, hints } => {
                let c = self.typed(*c, ops, 0, names)?;
                w!(
                    out,
                    "br {c}, {}, {}",
                    self.block(names, *t)?,
                    self.block(names, *f)?
                );
                self.hints(&mut out, hints);
            }
            InstKind::Switch {
                v,
                default,
                cases,
                hints,
            } => {
                let op = ops.first().ok_or_else(|| bad("switch operand"))?;
                let ty = self.t.text(op.node)?;
                let vt = self.typed(*v, ops, 0, names)?;
                w!(out, "switch {vt}, {} [", self.block(names, *default)?);
                let int = self
                    .m
                    .value_ty(names.func, *v)
                    .and_then(|t| self.m.types().get(t))
                    .ok_or_else(|| bad("switch operand type"))?;
                for &(c, b) in cases.iter() {
                    w!(
                        out,
                        " {ty} {}, {}",
                        int_text(Some(int), c)?,
                        self.block(names, b)?
                    );
                }
                out.push_str(" ]");
                self.hints(&mut out, hints);
            }
            InstKind::Unreachable => out.push_str("unreachable"),
        }
        Ok(out)
    }

    fn hints(&mut self, out: &mut String, hints: &Option<LoopHints>) {
        let Some(h) = hints else {
            return;
        };
        let id = self.next_md;
        self.next_md += 1;
        let mut props = vec![format!("!{id}")];
        let mut prop = |p: &mut Self, body: String| {
            let n = p.next_md;
            p.next_md += 1;
            p.extra_md.push(format!("!{n} = !{{{body}}}"));
            props.push(format!("!{n}"));
        };
        if h.must_progress {
            prop(self, "!\"llvm.loop.mustprogress\"".into());
        }
        match h.unroll {
            Unroll::Default => {}
            Unroll::Disable => prop(self, "!\"llvm.loop.unroll.disable\"".into()),
            Unroll::Enable => prop(self, "!\"llvm.loop.unroll.enable\"".into()),
            Unroll::Full => prop(self, "!\"llvm.loop.unroll.full\"".into()),
            Unroll::Count(n) => prop(self, format!("!\"llvm.loop.unroll.count\", i32 {n}")),
        }
        self.extra_md
            .push(format!("!{id} = distinct !{{{}}}", props.join(", ")));
        w!(out, ", !llvm.loop !{id}");
    }

    fn range_md(&mut self, r: &RangeMd, ty: TypeId) -> Result<String, WriteError> {
        let types = self.m.types();
        let scalar = types.scalar(ty);
        let t = types.get(scalar);
        let name = self.t.erased_text(scalar)?;
        let mut parts = Vec::with_capacity(r.ranges.len() * 2);
        for &(lo, hi) in r.ranges.iter() {
            parts.push(format!("{name} {}", int_text(t, lo)?));
            parts.push(format!("{name} {}", int_text(t, hi)?));
        }
        let n = self.next_md;
        self.next_md += 1;
        self.extra_md
            .push(format!("!{n} = !{{{}}}", parts.join(", ")));
        Ok(format!("!{n}"))
    }
}

fn index_list(indices: &[u32]) -> String {
    indices.iter().map(|i| format!(", {i}")).collect()
}

fn linkage(out: &mut String, l: Linkage) {
    let kw = l.keyword();
    if !kw.is_empty() {
        w!(out, "{kw} ");
    }
}

/// A global's or function's visibility and (for a global, which spells it before the type)
/// `unnamed_addr`, each followed by a space.
fn symbol_flags(out: &mut String, v: Visibility, u: UnnamedAddr) {
    for kw in [v.keyword(), u.keyword()] {
        if !kw.is_empty() {
            w!(out, "{kw} ");
        }
    }
}

/// The LLVM keyword of each enum attribute kind (`AttributeKindCodes`) the dialect `metal-as`
/// parses spells, other than the ones the IR interprets (written from their own fields). The
/// table is checked kind by kind against `metal-as` and the decoder
/// (`tests::enum_attribute_keywords_round_trip`).
pub(super) const ENUM_KEYWORDS: &[(u32, &str)] = &[
    (2, "alwaysinline"),
    (4, "inlinehint"),
    (5, "inreg"),
    (6, "minsize"),
    (7, "naked"),
    (10, "nobuiltin"),
    (12, "noduplicate"),
    (13, "noimplicitfloat"),
    (14, "noinline"),
    (15, "nonlazybind"),
    (16, "noredzone"),
    (17, "noreturn"),
    (18, "nounwind"),
    (19, "optsize"),
    (22, "returned"),
    (23, "returns_twice"),
    (24, "signext"),
    (26, "ssp"),
    (27, "sspreq"),
    (28, "sspstrong"),
    (30, "sanitize_address"),
    (31, "sanitize_thread"),
    (32, "sanitize_memory"),
    // Not 33: `uwtable` is an integer attribute in the dialect `metal-as` parses, so an enum-form
    // `uwtable` (older bitcode) would not read back as written; it is refused.
    (34, "zeroext"),
    (35, "builtin"),
    (36, "cold"),
    (37, "optnone"),
    (39, "nonnull"),
    (40, "jumptable"),
    (44, "safestack"),
    (48, "norecurse"),
    (53, "speculatable"),
    (54, "strictfp"),
    (55, "sanitize_hwaddress"),
    (56, "nocf_check"),
    (57, "optforfuzzing"),
    (58, "shadowcallstack"),
    (59, "speculative_load_hardening"),
    (60, "immarg"),
    (61, "willreturn"),
    (62, "nofree"),
    (63, "nosync"),
    (64, "sanitize_memtag"),
    (66, "nomerge"),
    (67, "null_pointer_is_valid"),
    (68, "noundef"),
    (70, "mustprogress"),
    (71, "nocallback"),
    (72, "hot"),
    (73, "noprofile"),
    (76, "nosanitize_coverage"),
    (78, "disable_sanitizer_instrumentation"),
    (79, "nosanitize_bounds"),
    (80, "allocalign"),
];

/// The keywords of `enums`, in kind order; a kind without one is refused.
fn enum_spelling(enums: EnumAttrs) -> Result<Vec<&'static str>, WriteError> {
    enums
        .kinds()
        .map(|kind| {
            ENUM_KEYWORDS
                .iter()
                .find(|&&(k, _)| k == kind)
                .map(|&(_, kw)| kw)
                .ok_or(WriteError::Unsupported(Unspellable::AttributeKind(kind)))
        })
        .collect()
}

fn align(out: &mut String, a: Option<Align>) {
    if let Some(a) = a {
        w!(out, ", align {}", a.bytes());
    }
}

fn atomic_prefix(out: &mut String, atomic: Option<Atomic>, volatile: bool) {
    if atomic.is_some() {
        out.push_str(" atomic");
    }
    if volatile {
        out.push_str(" volatile");
    }
}

fn atomic_suffix(out: &mut String, atomic: Option<Atomic>) {
    if let Some(a) = atomic {
        if a.scope == SyncScope::SingleThread {
            out.push_str(" syncscope(\"singlethread\")");
        }
        w!(out, " {}", a.ordering.mnemonic());
    }
}

fn int_flags(flags: IntFlags) -> String {
    IntFlags::ALL
        .iter()
        .filter(|(f, _)| flags.contains(*f))
        .map(|(_, n)| format!(" {}", n.to_ascii_lowercase()))
        .collect()
}

fn fmf_text(fmf: FastMath) -> String {
    FastMath::ALL
        .iter()
        .filter(|(f, _)| fmf.contains(*f))
        .map(|(_, n)| format!(" {}", n.to_ascii_lowercase()))
        .collect()
}

/// A named-metadata name: bare when every byte is an identifier byte, else escaped.
fn md_name(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for (i, &b) in bytes.iter().enumerate() {
        let ok = b.is_ascii_alphabetic()
            || matches!(b, b'.' | b'_' | b'-' | b'$')
            || (i > 0 && b.is_ascii_digit());
        if ok {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "\\{b:02X}");
        }
    }
    out
}

/// An integer constant of `ty` from its zero-extended bits, as LLVM spells it.
fn int_text(ty: Option<&Type>, v: u64) -> Result<String, WriteError> {
    let bits = match ty {
        Some(Type::Int(b)) => u32::from(*b),
        _ => return Err(bad("an integer constant of a non-integer type")),
    };
    if bits == 1 {
        return Ok(if v & 1 == 1 { "true" } else { "false" }.into());
    }
    let shift = 64 - bits;
    Ok(((v << shift) as i64 >> shift).to_string())
}

/// A floating-point constant from its bit pattern, as LLVM's hexadecimal spellings (a `float` is
/// spelled as the `double` of the same value, which LLVM requires).
fn float_text(ty: Option<&Type>, bits: u64) -> Result<String, WriteError> {
    Ok(match ty {
        Some(Type::Half) => format!("0xH{:04X}", bits & 0xFFFF),
        Some(Type::BFloat) => format!("0xR{:04X}", bits & 0xFFFF),
        Some(Type::Float) => format!("0x{:016X}", f32_bits_as_f64(bits as u32)),
        Some(Type::Double) => format!("0x{bits:016X}"),
        _ => return Err(bad("a float constant of a non-float type")),
    })
}

/// The `double` bit pattern of the `float` with bits `b`, exactly, NaN payloads included.
pub(super) fn f32_bits_as_f64(b: u32) -> u64 {
    let sign = u64::from(b >> 31) << 63;
    let exp = (b >> 23) & 0xFF;
    let man = u64::from(b & 0x7F_FFFF);
    if exp == 0xFF {
        return sign | (0x7FF << 52) | (man << 29);
    }
    if exp == 0 {
        if man == 0 {
            return sign;
        }
        // A denormal: man * 2^-149 with its top set bit at position p.
        let p = 63 - man.leading_zeros();
        let exp64 = u64::from(p) + 1023 - 149;
        let frac = (man & !(1 << p)) << (52 - p);
        return sign | (exp64 << 52) | frac;
    }
    sign | ((u64::from(exp) + 1023 - 127) << 52) | (man << 29)
}

/// `memory(...)` in the pre-`memory` spellings `metal-as` parses; `None` restriction for
/// `readwrite` everywhere, which is no attribute.
fn memory_spelling(m: MemoryEffects) -> Result<Vec<&'static str>, WriteError> {
    let access = |mr: ModRef| match mr {
        ModRef::NoModRef => Some("readnone"),
        ModRef::Ref => Some("readonly"),
        ModRef::Mod => Some("writeonly"),
        ModRef::ModRef => None,
    };
    let MemoryEffects {
        argmem,
        inaccessible,
        other,
    } = m;
    let (loc, mr) = if argmem == inaccessible && inaccessible == other {
        (None, argmem)
    } else if other == ModRef::NoModRef && inaccessible == ModRef::NoModRef {
        (Some("argmemonly"), argmem)
    } else if other == ModRef::NoModRef && argmem == ModRef::NoModRef {
        (Some("inaccessiblememonly"), inaccessible)
    } else if other == ModRef::NoModRef && argmem == inaccessible {
        (Some("inaccessiblemem_or_argmemonly"), argmem)
    } else {
        return Err(WriteError::Unsupported(Unspellable::MemoryEffects));
    };
    Ok(loc.into_iter().chain(access(mr)).collect())
}

/// The `target datalayout` string: every entry that differs from LLVM's defaults.
pub(super) fn data_layout(dl: &DataLayout) -> String {
    let d = DataLayout::default();
    let mut parts: Vec<String> = Vec::new();
    parts.push(
        match dl.endian {
            metal2vulkan_air::ir::layout::Endian::Little => "e",
            metal2vulkan_air::ir::layout::Endian::Big => "E",
        }
        .into(),
    );
    if let Some(m) = dl.mangling {
        parts.push(format!("m:{}", char::from(m)));
    }
    for (&space, p) in &dl.pointers {
        if d.pointers.get(&space) == Some(p) {
            continue;
        }
        let sp = if space == 0 {
            String::new()
        } else {
            space.to_string()
        };
        let mut s = format!("p{sp}:{}:{}:{}", p.size_bits, p.abi_bits, p.pref_bits);
        if p.index_bits != p.size_bits {
            w!(s, ":{}", p.index_bits);
        }
        parts.push(s);
    }
    for (head, table, defaults) in [
        ('i', &dl.ints, &d.ints),
        ('f', &dl.floats, &d.floats),
        ('v', &dl.vectors, &d.vectors),
    ] {
        for (&size, spec) in table {
            if defaults.get(&size) != Some(spec) {
                parts.push(format!("{head}{size}:{}:{}", spec.abi_bits, spec.pref_bits));
            }
        }
    }
    if dl.aggregate != d.aggregate {
        parts.push(format!(
            "a:{}:{}",
            dl.aggregate.abi_bits, dl.aggregate.pref_bits
        ));
    }
    if !dl.native_ints.is_empty() {
        let n: Vec<String> = dl.native_ints.iter().map(u32::to_string).collect();
        parts.push(format!("n{}", n.join(":")));
    }
    if !dl.non_integral.is_empty() {
        let n: Vec<String> = dl.non_integral.iter().map(u32::to_string).collect();
        parts.push(format!("ni:{}", n.join(":")));
    }
    if let Some(s) = dl.stack_align_bits {
        parts.push(format!("S{s}"));
    }
    if dl.alloca_space != 0 {
        parts.push(format!("A{}", dl.alloca_space));
    }
    if dl.program_space != 0 {
        parts.push(format!("P{}", dl.program_space));
    }
    if dl.globals_space != 0 {
        parts.push(format!("G{}", dl.globals_space));
    }
    match dl.fn_ptr_align {
        Some(metal2vulkan_air::ir::layout::FnPtrAlign::Independent(n)) => {
            parts.push(format!("Fi{n}"))
        }
        Some(metal2vulkan_air::ir::layout::FnPtrAlign::MultipleOfFunctionAlign(n)) => {
            parts.push(format!("Fn{n}"))
        }
        None => {}
    }
    parts.join("-")
}
