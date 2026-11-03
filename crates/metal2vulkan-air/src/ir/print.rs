//! A deterministic, LLVM-like diagnostic printer. Never parsed.
//!
//! The output depends only on the module, so identical modules print identical bytes (golden
//! and determinism tests rely on it). Scalars and vectors print inline; arrays, structs, and
//! function types print as `%tN` references defined in a leading type section, and aggregate and
//! expression constants as `cN` references defined in a constant section. Every section is a
//! single forward sweep with no recursion, so printing is linear in the module size.

use std::fmt::Write;

use super::attrs::ParamAttrs;
use super::constant::{Const, ConstExpr};
use super::id::{ConstId, Idx, InstId, Sym, TypeId};
use super::inst::{Atomic, Callee, FastMath, Inst, InstKind, IntFlags, LoopHints, Value};
use super::metadata::Md;
use super::module::{Function, Section};
use super::types::Type;
use super::verify::Module;

/// Print `module`.
pub fn print(module: &Module) -> String {
    let mut p = Printer {
        m: module,
        out: String::new(),
    };
    p.module();
    p.out
}

/// Print one instruction of a function body of `module` exactly as [`print()`] prints it inside
/// that body, without the indent and the newline (for diagnostics and census tooling). Operands
/// name the module's sections (`%tN`, `cN`, `@fN`) as the full printout does.
pub fn print_inst(module: &Module, id: InstId, inst: &Inst) -> String {
    let mut p = Printer {
        m: module,
        out: String::new(),
    };
    p.inst(id, inst);
    let line = p.out.strip_suffix('\n').unwrap_or(&p.out);
    line.strip_prefix("  ").unwrap_or(line).to_owned()
}

struct Printer<'a> {
    m: &'a Module,
    out: String,
}

/// Append formatted text; writing to a `String` cannot fail.
macro_rules! w {
    ($p:expr, $($arg:tt)*) => {{
        let _ = write!($p.out, $($arg)*);
    }};
}

impl Printer<'_> {
    fn module(&mut self) {
        let parts = self.m.parts();
        if let Some(t) = parts.triple {
            w!(self, "target triple = {}\n", self.sym(t));
        }
        if let Some(s) = parts.source_filename {
            w!(self, "source_filename = {}\n", self.sym(s));
        }
        w!(self, "; datalayout {:?}\n", parts.data_layout);
        for (id, ty) in self.m.types().iter() {
            if self.is_inline(ty) {
                continue;
            }
            w!(self, "%t{} = type ", id.raw());
            self.type_body(ty);
            if let Type::Struct { name: Some(n), .. } | Type::OpaqueStruct { name: Some(n) } = ty {
                w!(self, " ; {}", self.sym(*n));
            }
            self.out.push('\n');
        }
        for (id, c) in self.m.consts().iter() {
            if self.const_inline(&c.value).is_some() {
                continue;
            }
            w!(self, "c{} = {} ", id.raw(), self.ty(c.ty));
            match &c.value {
                Const::Aggregate(elems) => {
                    self.out.push('[');
                    self.const_list(elems);
                    self.out.push(']');
                }
                Const::Expr(ConstExpr::Cast { op, ops: [v] }) => {
                    w!(self, "{} ({})", op.mnemonic(), self.const_ref(*v));
                }
                Const::Expr(ConstExpr::BinOp { op, flags, ops }) => {
                    w!(self, "{}", op.mnemonic());
                    self.int_flags(*flags);
                    self.out.push_str(" (");
                    self.const_list(ops);
                    self.out.push(')');
                }
                Const::Expr(ConstExpr::Gep {
                    src_ty,
                    inbounds,
                    ops,
                }) => {
                    let ib = if *inbounds { " inbounds" } else { "" };
                    w!(self, "getelementptr{ib} ({}, ", self.ty(*src_ty));
                    self.const_list(ops);
                    self.out.push(')');
                }
                Const::InlineAsm(a) => self.out.push_str(&a.spelling()),
                Const::Int(_)
                | Const::Float(_)
                | Const::Null
                | Const::Undef
                | Const::Poison
                | Const::GlobalAddr(_)
                | Const::FuncAddr(_) => {}
            }
            self.out.push('\n');
        }
        for (id, g) in parts.globals.iter() {
            w!(self, "@g{} {} = ", id.raw(), self.sym(g.name));
            self.linkage(g.linkage.keyword());
            if g.externally_initialized {
                self.out.push_str("externally_initialized ");
            }
            let kind = if g.constant { "constant" } else { "global" };
            w!(self, "{} {kind} {}", g.space, self.ty(g.value_ty));
            if let Some(init) = g.init {
                w!(self, " {}", self.const_ref(init));
            }
            if let Some(a) = g.align {
                w!(self, ", align {}", a.bytes());
            }
            self.section(g.section);
            self.out.push('\n');
        }
        for (id, f) in parts.funcs.iter() {
            self.function(id.raw(), f);
        }
        for (id, md) in parts.metadata.iter() {
            w!(self, "!md{} = ", id.raw());
            match md {
                Md::String(s) => w!(self, "!{}", self.sym(*s)),
                Md::Value(c) => w!(self, "{} {}", self.const_ty(*c), self.const_ref(*c)),
                Md::Node { distinct, ops } => {
                    if *distinct {
                        self.out.push_str("distinct ");
                    }
                    self.out.push_str("!{");
                    for (i, op) in ops.iter().enumerate() {
                        if i > 0 {
                            self.out.push_str(", ");
                        }
                        match op {
                            Some(m) => w!(self, "!md{}", m.raw()),
                            None => self.out.push_str("null"),
                        }
                    }
                    self.out.push('}');
                }
                Md::Debug => self.out.push_str("debug"),
            }
            self.out.push('\n');
        }
        for n in &parts.named_md {
            w!(self, "!{} = !{{", self.sym(n.name));
            for (i, m) in n.ops.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                w!(self, "!md{}", m.raw());
            }
            self.out.push_str("}\n");
        }
    }

    fn function(&mut self, id: u32, f: &Function) {
        let kw = if f.body.is_some() {
            "define"
        } else {
            "declare"
        };
        w!(self, "\n{kw} ");
        self.linkage(f.linkage.keyword());
        w!(self, "{} @f{id} {}", self.ty(f.ty), self.sym(f.name));
        if f.cconv == super::CallConv::Fast {
            self.out.push_str(" fastcc");
        }
        if f.attrs.convergent {
            self.out.push_str(" convergent");
        }
        if let Some(m) = f.attrs.memory {
            w!(
                self,
                " memory({:?}, {:?}, {:?})",
                m.argmem,
                m.inaccessible,
                m.other
            );
        }
        self.strings(&f.attrs.strings);
        self.param_attrs("ret", &f.ret_attrs);
        for (i, p) in f.params.iter().enumerate() {
            self.param_attrs(&format!("%a{i}"), &p.attrs);
            if let Some(n) = p.name {
                w!(self, " %a{i}={}", self.sym(n));
            }
        }
        if let Some(a) = f.align {
            w!(self, ", align {}", a.bytes());
        }
        self.section(f.section);
        let Some(body) = &f.body else {
            self.out.push('\n');
            return;
        };
        self.out.push_str(" {\n");
        for (b, block) in body.blocks.iter() {
            w!(self, "bb{}:", b.raw());
            if let Some(n) = block.name {
                w!(self, " ; {}", self.sym(n));
            }
            self.out.push('\n');
            for i in block.inst_ids() {
                if let Some(inst) = body.insts.get(i) {
                    self.inst(i, inst);
                }
            }
        }
        self.out.push_str("}\n");
    }

    fn inst(&mut self, id: InstId, inst: &Inst) {
        self.out.push_str("  ");
        if !matches!(self.m.types().get(inst.ty), Some(Type::Void)) {
            w!(self, "%i{} = ", id.raw());
        }
        let ty = self.ty(inst.ty);
        match &inst.kind {
            InstKind::BinOp {
                op,
                flags,
                fmf,
                a,
                b,
            } => {
                w!(self, "{}", op.mnemonic());
                self.int_flags(*flags);
                self.fmf(*fmf);
                w!(self, " {ty} {}, {}", self.v(*a), self.v(*b));
            }
            InstKind::FNeg { v, fmf } => {
                self.out.push_str("fneg");
                self.fmf(*fmf);
                w!(self, " {ty} {}", self.v(*v));
            }
            InstKind::Cast { op, v } => {
                w!(self, "{} {} to {ty}", op.mnemonic(), self.v(*v));
            }
            InstKind::ICmp { pred, a, b } => {
                w!(
                    self,
                    "icmp {} {}, {}",
                    pred.mnemonic(),
                    self.v(*a),
                    self.v(*b)
                );
            }
            InstKind::FCmp { pred, fmf, a, b } => {
                self.out.push_str("fcmp");
                self.fmf(*fmf);
                w!(self, " {} {}, {}", pred.mnemonic(), self.v(*a), self.v(*b));
            }
            InstKind::Select { c, t, f, fmf } => {
                self.out.push_str("select");
                self.fmf(*fmf);
                w!(self, " {ty} {}, {}, {}", self.v(*c), self.v(*t), self.v(*f));
            }
            InstKind::Freeze(v) => w!(self, "freeze {ty} {}", self.v(*v)),
            InstKind::ExtractElement { vec, idx } => {
                w!(self, "extractelement {}, {}", self.v(*vec), self.v(*idx));
            }
            InstKind::InsertElement { vec, elt, idx } => {
                let (vec, elt, idx) = (self.v(*vec), self.v(*elt), self.v(*idx));
                w!(self, "insertelement {vec}, {elt}, {idx}");
            }
            InstKind::ShuffleVector { a, b, mask } => {
                w!(self, "shufflevector {ty} {}, {}, <", self.v(*a), self.v(*b));
                for (i, m) in mask.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    match m {
                        Some(m) => w!(self, "{m}"),
                        None => self.out.push_str("poison"),
                    }
                }
                self.out.push('>');
            }
            InstKind::ExtractValue { agg, indices } => {
                w!(self, "extractvalue {ty} {}", self.v(*agg));
                self.indices(indices);
            }
            InstKind::InsertValue { agg, val, indices } => {
                w!(self, "insertvalue {} {}", self.v(*agg), self.v(*val));
                self.indices(indices);
            }
            InstKind::Alloca {
                ty: allocated,
                count,
                align,
            } => {
                w!(self, "alloca {}, {}", self.ty(*allocated), self.v(*count));
                if let Some(a) = align {
                    w!(self, ", align {}", a.bytes());
                }
            }
            InstKind::Load {
                ptr,
                align,
                volatile,
                atomic,
                range,
            } => {
                self.out.push_str("load");
                self.atomic_prefix(*atomic, *volatile);
                w!(self, " {ty}, {}", self.v(*ptr));
                self.atomic_suffix(*atomic);
                if let Some(a) = align {
                    w!(self, ", align {}", a.bytes());
                }
                if let Some(r) = range {
                    w!(self, ", !range {:?}", r.ranges);
                }
            }
            InstKind::Store {
                val,
                ptr,
                align,
                volatile,
                atomic,
            } => {
                self.out.push_str("store");
                self.atomic_prefix(*atomic, *volatile);
                w!(self, " {}, {}", self.v(*val), self.v(*ptr));
                self.atomic_suffix(*atomic);
                if let Some(a) = align {
                    w!(self, ", align {}", a.bytes());
                }
            }
            InstKind::Gep {
                src_ty,
                base,
                indices,
                inbounds,
            } => {
                let ib = if *inbounds { " inbounds" } else { "" };
                w!(
                    self,
                    "getelementptr{ib} {}, {}",
                    self.ty(*src_ty),
                    self.v(*base)
                );
                for &i in indices.iter() {
                    w!(self, ", {}", self.v(i));
                }
            }
            InstKind::AtomicRmw {
                op,
                ptr,
                val,
                atomic,
                volatile,
                align,
            } => {
                self.out.push_str("atomicrmw");
                if *volatile {
                    self.out.push_str(" volatile");
                }
                w!(
                    self,
                    " {} {}, {}",
                    op.mnemonic(),
                    self.v(*ptr),
                    self.v(*val)
                );
                self.atomic_suffix(Some(*atomic));
                if let Some(a) = align {
                    w!(self, ", align {}", a.bytes());
                }
            }
            InstKind::CmpXchg {
                ptr,
                cmp,
                new,
                success,
                failure,
                weak,
                volatile,
                align,
            } => {
                self.out.push_str("cmpxchg");
                if *weak {
                    self.out.push_str(" weak");
                }
                if *volatile {
                    self.out.push_str(" volatile");
                }
                let (ptr, cmp, new) = (self.v(*ptr), self.v(*cmp), self.v(*new));
                w!(self, " {ptr}, {cmp}, {new}");
                self.atomic_suffix(Some(*success));
                w!(self, " {}", failure.mnemonic());
                if let Some(a) = align {
                    w!(self, ", align {}", a.bytes());
                }
            }
            InstKind::Fence(a) => {
                self.out.push_str("fence");
                self.atomic_suffix(Some(*a));
            }
            InstKind::Call(call) => {
                let tail = match call.tail {
                    super::TailKind::None => "",
                    super::TailKind::Tail => "tail ",
                    super::TailKind::MustTail => "musttail ",
                    super::TailKind::NoTail => "notail ",
                };
                w!(self, "{tail}call");
                self.fmf(call.fmf);
                w!(self, " {ty} ");
                match &call.callee {
                    Callee::Direct(f) => w!(self, "@f{}", f.raw()),
                    Callee::Indirect { callee, fn_ty } => {
                        w!(self, "{} {}", self.ty(*fn_ty), self.v(*callee));
                    }
                    Callee::Air { decl, op } => w!(self, "@f{} [air {op:?}]", decl.raw()),
                    Callee::Llvm { decl, op } => w!(self, "@f{} [llvm {op:?}]", decl.raw()),
                    Callee::UnknownIntrinsic(f) => w!(self, "@f{} [unknown]", f.raw()),
                }
                self.out.push('(');
                for (i, &a) in call.args.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    let a = self.v(a);
                    self.out.push_str(&a);
                }
                self.out.push(')');
                if let Some(attrs) = &call.attrs {
                    if attrs.func.convergent {
                        self.out.push_str(" convergent");
                    }
                    self.strings(&attrs.func.strings);
                    self.param_attrs("ret", &attrs.ret);
                    for (i, p) in attrs.params.iter().enumerate() {
                        self.param_attrs(&format!("arg{i}"), p);
                    }
                }
                if let Some(r) = &call.range {
                    w!(self, ", !range {:?}", r.ranges);
                }
            }
            InstKind::Phi { incoming, fmf } => {
                self.out.push_str("phi");
                self.fmf(*fmf);
                w!(self, " {ty}");
                for (i, &(v, b)) in incoming.iter().enumerate() {
                    let sep = if i > 0 { "," } else { "" };
                    w!(self, "{sep} [{}, bb{}]", self.v(v), b.raw());
                }
            }
            InstKind::Ret(v) => match v {
                Some(v) => w!(self, "ret {}", self.v(*v)),
                None => self.out.push_str("ret void"),
            },
            InstKind::Br { dest, hints } => {
                w!(self, "br bb{}", dest.raw());
                self.hints(hints);
            }
            InstKind::CondBr { c, t, f, hints } => {
                w!(self, "br {}, bb{}, bb{}", self.v(*c), t.raw(), f.raw());
                self.hints(hints);
            }
            InstKind::Switch {
                v,
                default,
                cases,
                hints,
            } => {
                w!(self, "switch {}, bb{} [", self.v(*v), default.raw());
                for &(c, b) in cases.iter() {
                    w!(self, " {c}: bb{}", b.raw());
                }
                self.out.push_str(" ]");
                self.hints(hints);
            }
            InstKind::Unreachable => self.out.push_str("unreachable"),
        }
        self.out.push('\n');
    }

    fn hints(&mut self, hints: &Option<LoopHints>) {
        if let Some(h) = hints {
            w!(self, ", !llvm.loop {h:?}");
        }
    }

    fn atomic_prefix(&mut self, atomic: Option<Atomic>, volatile: bool) {
        if atomic.is_some() {
            self.out.push_str(" atomic");
        }
        if volatile {
            self.out.push_str(" volatile");
        }
    }

    fn atomic_suffix(&mut self, atomic: Option<Atomic>) {
        if let Some(a) = atomic {
            if a.scope == super::SyncScope::SingleThread {
                self.out.push_str(" syncscope(\"singlethread\")");
            }
            w!(self, " {}", a.ordering.mnemonic());
        }
    }

    fn indices(&mut self, indices: &[u32]) {
        for i in indices {
            w!(self, ", {i}");
        }
    }

    fn int_flags(&mut self, flags: IntFlags) {
        for &(f, name) in IntFlags::ALL {
            if flags.contains(f) {
                w!(self, " {}", name.to_ascii_lowercase());
            }
        }
    }

    fn fmf(&mut self, fmf: FastMath) {
        for &(f, name) in FastMath::ALL {
            if fmf.contains(f) {
                w!(self, " {}", name.to_ascii_lowercase());
            }
        }
    }

    fn linkage(&mut self, kw: &str) {
        if !kw.is_empty() {
            w!(self, "{kw} ");
        }
    }

    fn section(&mut self, s: Option<Section>) {
        match s {
            Some(Section::FcInitializer) => self.out.push_str(", section air.fc_initializer"),
            Some(Section::StaticInit) => self.out.push_str(", section air.static_init"),
            Some(Section::Other(s)) => w!(self, ", section {}", self.sym(s)),
            None => {}
        }
    }

    fn strings(&mut self, strings: &[(Sym, Sym)]) {
        for &(k, v) in strings {
            w!(self, " {}={}", self.sym(k), self.sym(v));
        }
    }

    fn param_attrs(&mut self, what: &str, a: &ParamAttrs) {
        if *a == ParamAttrs::default() {
            return;
        }
        w!(self, " {what}(");
        if let Some(al) = a.align {
            w!(self, " align {}", al.bytes());
        }
        if let Some(n) = a.dereferenceable {
            w!(self, " dereferenceable({n})");
        }
        if let Some(n) = a.dereferenceable_or_null {
            w!(self, " dereferenceable_or_null({n})");
        }
        if a.noalias {
            self.out.push_str(" noalias");
        }
        if let Some(m) = a.access {
            w!(self, " {m:?}");
        }
        if let Some(t) = a.byval {
            w!(self, " byval({})", self.ty(t));
        }
        if let Some(t) = a.sret {
            w!(self, " sret({})", self.ty(t));
        }
        if a.nocapture {
            self.out.push_str(" nocapture");
        }
        self.strings(&a.strings);
        self.out.push_str(" )");
    }

    fn const_list(&mut self, ids: &[ConstId]) {
        for (i, &c) in ids.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let c = self.const_ref(c);
            self.out.push_str(&c);
        }
    }

    /// Scalars and vectors print inline; the rest are `%tN` references.
    fn is_inline(&self, ty: &Type) -> bool {
        !matches!(
            ty,
            Type::Array { .. }
                | Type::Struct { .. }
                | Type::OpaqueStruct { .. }
                | Type::Function { .. }
        )
    }

    fn ty(&self, id: TypeId) -> String {
        match self.m.types().get(id) {
            Some(Type::Vector { elem, len }) => format!("<{len} x {}>", self.scalar_ty(*elem)),
            Some(ty) if self.is_inline(ty) => self.scalar_ty(id),
            Some(_) => format!("%t{}", id.raw()),
            None => format!("<bad type {}>", id.raw()),
        }
    }

    fn scalar_ty(&self, id: TypeId) -> String {
        match self.m.types().get(id) {
            Some(Type::Void) => "void".into(),
            Some(Type::Label) => "label".into(),
            Some(Type::Metadata) => "metadata".into(),
            Some(Type::Int(w)) => format!("i{w}"),
            Some(Type::Half) => "half".into(),
            Some(Type::BFloat) => "bfloat".into(),
            Some(Type::Float) => "float".into(),
            Some(Type::Double) => "double".into(),
            Some(Type::Ptr { space }) => format!("ptr {space}"),
            _ => format!("%t{}", id.raw()),
        }
    }

    fn type_body(&mut self, ty: &Type) {
        match ty {
            Type::Array { elem, len } => w!(self, "[{len} x {}]", self.ty(*elem)),
            Type::Struct { fields, packed, .. } => {
                let (open, close) = if *packed { ("<{", "}>") } else { ("{", "}") };
                self.out.push_str(open);
                for (i, &f) in fields.iter().enumerate() {
                    let sep = if i > 0 { ", " } else { " " };
                    w!(self, "{sep}{}", self.ty(f));
                }
                w!(self, " {close}");
            }
            Type::OpaqueStruct { .. } => self.out.push_str("opaque"),
            Type::Function {
                ret,
                params,
                vararg,
            } => {
                w!(self, "{} (", self.ty(*ret));
                for (i, &p) in params.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    let p = self.ty(p);
                    self.out.push_str(&p);
                }
                if *vararg {
                    self.out
                        .push_str(if params.is_empty() { "..." } else { ", ..." });
                }
                self.out.push(')');
            }
            _ => {}
        }
    }

    fn const_ty(&self, c: ConstId) -> String {
        self.m
            .consts()
            .ty(c)
            .map_or_else(|| "<bad>".into(), |t| self.ty(t))
    }

    fn const_inline(&self, c: &Const) -> Option<String> {
        Some(match c {
            Const::Int(v) => format!("{v}"),
            Const::Float(bits) => format!("0x{bits:X}"),
            Const::Null => "zeroinitializer".into(),
            Const::Undef => "undef".into(),
            Const::Poison => "poison".into(),
            Const::GlobalAddr(g) => format!("@g{}", g.raw()),
            Const::FuncAddr(f) => format!("@f{}", f.raw()),
            Const::Aggregate(_) | Const::Expr(_) | Const::InlineAsm(_) => return None,
        })
    }

    fn const_ref(&self, id: ConstId) -> String {
        self.m
            .consts()
            .get(id)
            .and_then(|c| self.const_inline(&c.value))
            .unwrap_or_else(|| format!("c{}", id.raw()))
    }

    fn v(&self, v: Value) -> String {
        match v {
            Value::Arg(a) => format!("%a{}", a.index()),
            Value::Inst(i) => format!("%i{}", i.raw()),
            Value::Const(c) => format!("{} {}", self.const_ty(c), self.const_ref(c)),
        }
    }

    /// A symbol as an LLVM-style quoted string with `\XX` escapes.
    fn sym(&self, s: Sym) -> String {
        let bytes = self.m.symbols().get(s).unwrap_or(b"<bad symbol>");
        let mut out = String::with_capacity(bytes.len() + 2);
        out.push('"');
        for &b in bytes {
            if b.is_ascii_graphic() && b != b'"' && b != b'\\' || b == b' ' {
                out.push(char::from(b));
            } else {
                let _ = write!(out, "\\{b:02X}");
            }
        }
        out.push('"');
        out
    }
}
