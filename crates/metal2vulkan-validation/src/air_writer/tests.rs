//! Writer tests. The round trips run `xcrun metal-as` (and `xcrun metal` for MSL fixtures) and
//! skip with a visible message where `xcrun` is absent. CPU only.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use metal2vulkan::Budget;
use metal2vulkan_air::bitcode::{Decoded, EnumAttrs, SiteEnums};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::ir::{
    CallAttrs, Callee, Const, ConstExpr, ConstId, FnAttrs, FuncId, Global, InstId, InstKind, Md,
    MdId, MemoryEffects, ModRef, ParamAttrs, Section, Sym, Type, TypeId, UnnamedAddr, Value,
    Visibility,
};

use super::text::{f32_bits_as_f64, ENUM_KEYWORDS};
use super::{decode_as_written, metal_as, write_decoded, AssembleError};
use crate::gpu::compile::xcrun;
use crate::ScratchDir;

// ---- An id-free rendering of everything the IR keeps ----

/// The structure of a decoded module with every arena id replaced by what it denotes: types and
/// constants by structure, globals and functions by name and position, metadata by the tree
/// reachable from named metadata. Two decodings are "structurally equal" when these agree.
///
/// Two spellings with no meaning are normalized: an attribute list present but empty (the
/// decoder keeps `Some` for a list whose attributes it all drops) and trailing default
/// parameter attributes, and `memory(readwrite)`, which restricts nothing. The uninterpreted
/// function attributes of an `llvm.*` declaration are not compared: the toolchain gives an
/// intrinsic the attributes of its own intrinsic table when it reads the module (`metal-as`
/// drops the `mustprogress` the newer front end writes on `llvm.lifetime.start`), so they are
/// the reader's, not the module's.
fn structure(d: &Decoded, with_pointees: bool) -> Vec<String> {
    let r = Render { d };
    let m = &d.module;
    let parts = m.parts();
    let mut out = vec![
        format!("triple {:?}", parts.triple.map(|s| r.sym(s))),
        format!("source {:?}", parts.source_filename.map(|s| r.sym(s))),
        format!("layout {:?}", parts.data_layout),
    ];
    for g in parts.globals.values() {
        out.push(r.global(g));
    }
    for (id, f) in parts.funcs.iter() {
        let intrinsic = f.body.is_none() && r.sym(f.name).starts_with("llvm.");
        let empty = SiteEnums::default();
        let enums = d.enum_attrs.func(id).unwrap_or(&empty);
        let fn_enums = if intrinsic {
            EnumAttrs::default()
        } else {
            enums.func
        };
        out.push(format!(
            "fn {} {} {:?} {:?} {:?} {:?} {} ret({}) {} {:?}",
            r.sym(f.name),
            r.ty(f.ty),
            f.linkage,
            f.visibility,
            f.unnamed_addr,
            f.cconv,
            r.fn_attrs(&f.attrs, fn_enums),
            r.param_attrs(&f.ret_attrs, enums.ret),
            r.section(f.section),
            f.align,
        ));
        for (i, p) in f.params.iter().enumerate() {
            out.push(format!(
                "  param {} {:?}",
                r.param_attrs(&p.attrs, enums.param(i)),
                p.name.map(|n| r.sym(n))
            ));
        }
        if with_pointees {
            let ps = d.pointees.params(id).unwrap_or(&[]);
            let ps: Vec<String> = ps.iter().map(|p| r.opt_ty(*p)).collect();
            out.push(format!(
                "  pointees ret {} params [{}]",
                r.opt_ty(d.pointees.ret(id)),
                ps.join(", ")
            ));
        }
        let Some(body) = &f.body else {
            continue;
        };
        for (b, block) in body.blocks.iter() {
            out.push(format!(
                "  bb{} [{}..{}) {:?}",
                b.raw(),
                block.start.raw(),
                block.end.raw(),
                block.name.map(|n| r.sym(n))
            ));
        }
        for (i, inst) in body.insts.iter() {
            out.push(format!(
                "  %{} = {} {}",
                i.raw(),
                r.ty(inst.ty),
                r.inst(&inst.kind, id, i)
            ));
        }
    }
    for n in m.named_md() {
        let ops: Vec<String> = n.ops.iter().map(|&o| r.md(o, &mut Vec::new())).collect();
        out.push(format!("!{} = [{}]", r.sym(n.name), ops.join(", ")));
    }
    out
}

struct Render<'d> {
    d: &'d Decoded,
}

impl Render<'_> {
    fn sym(&self, s: Sym) -> String {
        String::from_utf8_lossy(self.d.module.symbols().get(s).unwrap()).into_owned()
    }

    fn ty(&self, t: TypeId) -> String {
        match self.d.module.types().get(t).unwrap() {
            Type::Vector { elem, len } => format!("<{len} x {}>", self.ty(*elem)),
            Type::Array { elem, len } => format!("[{len} x {}]", self.ty(*elem)),
            Type::Struct {
                fields,
                packed,
                name,
            } => {
                let f: Vec<_> = fields.iter().map(|&f| self.ty(f)).collect();
                let n = name.map(|n| self.sym(n)).unwrap_or_default();
                format!(
                    "{n}{}{{{}}}",
                    if *packed { "packed" } else { "" },
                    f.join(", ")
                )
            }
            Type::OpaqueStruct { name } => format!("opaque {:?}", name.map(|n| self.sym(n))),
            Type::Function {
                ret,
                params,
                vararg,
            } => {
                let p: Vec<_> = params.iter().map(|&p| self.ty(p)).collect();
                format!(
                    "{}({}{})",
                    self.ty(*ret),
                    p.join(", "),
                    if *vararg { ", ..." } else { "" }
                )
            }
            other => format!("{other:?}"),
        }
    }

    fn opt_ty(&self, t: Option<TypeId>) -> String {
        t.map_or_else(|| "-".into(), |t| self.ty(t))
    }

    fn konst(&self, c: ConstId) -> String {
        let k = self.d.module.consts().get(c).unwrap();
        let v = match &k.value {
            Const::Aggregate(elems) => {
                let e: Vec<_> = elems.iter().map(|&e| self.konst(e)).collect();
                format!("[{}]", e.join(", "))
            }
            Const::GlobalAddr(g) => {
                format!("@{}", self.sym(self.d.module.global(*g).unwrap().name))
            }
            Const::FuncAddr(f) => format!("@{}", self.sym(self.d.module.func(*f).unwrap().name)),
            Const::Expr(ConstExpr::Cast { op, ops: [o] }) => {
                format!("{op:?}({})", self.konst(*o))
            }
            Const::Expr(ConstExpr::BinOp { op, flags, ops }) => format!(
                "{op:?}{flags:?}({}, {})",
                self.konst(ops[0]),
                self.konst(ops[1])
            ),
            Const::Expr(ConstExpr::Gep {
                src_ty,
                inbounds,
                ops,
            }) => {
                let o: Vec<_> = ops.iter().map(|&o| self.konst(o)).collect();
                format!("gep{inbounds}({}; {})", self.ty(*src_ty), o.join(", "))
            }
            other => format!("{other:?}"),
        };
        format!("{} {v}", self.ty(k.ty))
    }

    fn value(&self, v: Value) -> String {
        match v {
            Value::Arg(a) => format!("%a{}", a.raw()),
            Value::Inst(i) => format!("%{}", i.raw()),
            Value::Const(c) => self.konst(c),
        }
    }

    fn section(&self, s: Option<Section>) -> String {
        match s {
            Some(Section::Other(s)) => format!("section {}", self.sym(s)),
            other => format!("{other:?}"),
        }
    }

    fn strings(&self, s: &[(Sym, Sym)]) -> String {
        let s: Vec<_> = s
            .iter()
            .map(|&(k, v)| format!("{}={}", self.sym(k), self.sym(v)))
            .collect();
        s.join(" ")
    }

    fn fn_attrs(&self, a: &FnAttrs, enums: EnumAttrs) -> String {
        let memory = a
            .memory
            .filter(|m| *m != MemoryEffects::uniform(ModRef::ModRef));
        format!(
            "{{conv {} {memory:?} {} {:?}}}",
            a.convergent,
            self.strings(&a.strings),
            enums.kinds().collect::<Vec<_>>()
        )
    }

    fn param_attrs(&self, a: &ParamAttrs, enums: EnumAttrs) -> String {
        format!(
            "{{{:?} {:?} {:?} {} {:?} {} {} {} {} {:?}}}",
            a.align,
            a.dereferenceable,
            a.dereferenceable_or_null,
            a.noalias,
            a.access,
            self.opt_ty(a.byval),
            self.opt_ty(a.sret),
            a.nocapture,
            self.strings(&a.strings),
            enums.kinds().collect::<Vec<_>>()
        )
    }

    fn call_attrs(&self, a: &Option<Box<CallAttrs>>, enums: Option<&SiteEnums>) -> String {
        let no_attrs = CallAttrs::default();
        let a = a.as_deref().unwrap_or(&no_attrs);
        let no_enums = SiteEnums::default();
        let enums = enums.unwrap_or(&no_enums);
        let positions = a.params.len().max(enums.params.len());
        let none = ParamAttrs::default();
        let mut params: Vec<_> = (0..positions)
            .map(|i| self.param_attrs(a.params.get(i).unwrap_or(&none), enums.param(i)))
            .collect();
        let empty = self.param_attrs(&none, EnumAttrs::default());
        while params.last() == Some(&empty) {
            params.pop();
        }
        let s = format!(
            "{} ret {} [{}]",
            self.fn_attrs(&a.func, enums.func),
            self.param_attrs(&a.ret, enums.ret),
            params.join(", ")
        );
        let none = format!(
            "{} ret {} []",
            self.fn_attrs(&FnAttrs::default(), EnumAttrs::default()),
            empty
        );
        if s == none {
            "-".into()
        } else {
            s
        }
    }

    fn global(&self, g: &Global) -> String {
        format!(
            "global {} {} {:?} const={} init={} ext={} {:?} {:?} {:?} {:?} {}",
            self.sym(g.name),
            self.ty(g.value_ty),
            g.space,
            g.constant,
            g.init.map(|c| self.konst(c)).unwrap_or_default(),
            g.externally_initialized,
            g.linkage,
            g.visibility,
            g.unnamed_addr,
            g.align,
            self.section(g.section)
        )
    }

    /// Instruction `at` of function `func`.
    fn inst(&self, k: &InstKind, func: FuncId, at: InstId) -> String {
        let mut ops = Vec::new();
        k.for_each_value(|v| ops.push(self.value(v)));
        let ops = ops.join(", ");
        let detail = match k {
            InstKind::BinOp { op, flags, fmf, .. } => format!("{op:?} {flags:?} {fmf:?}"),
            InstKind::FNeg { fmf, .. } => format!("fneg {fmf:?}"),
            InstKind::Cast { op, .. } => format!("{op:?}"),
            InstKind::ICmp { pred, .. } => format!("icmp {pred:?}"),
            InstKind::FCmp { pred, fmf, .. } => format!("fcmp {pred:?} {fmf:?}"),
            InstKind::Select { fmf, .. } => format!("select {fmf:?}"),
            InstKind::Freeze(_) => "freeze".into(),
            InstKind::ExtractElement { .. } => "extractelement".into(),
            InstKind::InsertElement { .. } => "insertelement".into(),
            InstKind::ShuffleVector { mask, .. } => format!("shufflevector {mask:?}"),
            InstKind::ExtractValue { indices, .. } => format!("extractvalue {indices:?}"),
            InstKind::InsertValue { indices, .. } => format!("insertvalue {indices:?}"),
            InstKind::Alloca { ty, align, .. } => format!("alloca {} {align:?}", self.ty(*ty)),
            InstKind::Load {
                align,
                volatile,
                atomic,
                range,
                ..
            } => format!("load {align:?} {volatile} {atomic:?} {range:?}"),
            InstKind::Store {
                align,
                volatile,
                atomic,
                ..
            } => format!("store {align:?} {volatile} {atomic:?}"),
            InstKind::Gep {
                src_ty, inbounds, ..
            } => format!("gep {} {inbounds}", self.ty(*src_ty)),
            InstKind::AtomicRmw {
                op,
                atomic,
                volatile,
                align,
                ..
            } => format!("atomicrmw {op:?} {atomic:?} {volatile} {align:?}"),
            InstKind::CmpXchg {
                success,
                failure,
                weak,
                volatile,
                align,
                ..
            } => format!("cmpxchg {success:?} {failure:?} {weak} {volatile} {align:?}"),
            InstKind::Fence(a) => format!("fence {a:?}"),
            InstKind::Call(call) => {
                let callee = match &call.callee {
                    Callee::Direct(f) => format!("direct @{}", self.fname(*f)),
                    Callee::Indirect { fn_ty, .. } => format!("indirect {}", self.ty(*fn_ty)),
                    Callee::Air { decl, op } => format!("air @{} {op:?}", self.fname(*decl)),
                    Callee::Llvm { decl, .. } => format!("llvm @{}", self.fname(*decl)),
                    Callee::UnknownIntrinsic(f) => format!("unknown @{}", self.fname(*f)),
                    Callee::Unbound { decl, template } => {
                        format!("unbound @{} {template:?}", self.fname(*decl))
                    }
                };
                format!(
                    "call {callee} {:?} {:?} {} {:?}",
                    call.fmf,
                    call.tail,
                    self.call_attrs(&call.attrs, self.d.enum_attrs.call(func, at)),
                    call.range
                )
            }
            InstKind::Phi { incoming, fmf } => {
                let b: Vec<_> = incoming.iter().map(|(_, b)| b.raw()).collect();
                format!("phi {fmf:?} {b:?}")
            }
            InstKind::Ret(_) => "ret".into(),
            InstKind::Br { dest, hints } => format!("br {} {hints:?}", dest.raw()),
            InstKind::CondBr { t, f, hints, .. } => {
                format!("condbr {} {} {hints:?}", t.raw(), f.raw())
            }
            InstKind::Switch {
                default,
                cases,
                hints,
                ..
            } => {
                let c: Vec<_> = cases.iter().map(|(v, b)| (v, b.raw())).collect();
                format!("switch {} {c:?} {hints:?}", default.raw())
            }
            InstKind::Unreachable => "unreachable".into(),
        };
        format!("{detail} ({ops})")
    }

    fn fname(&self, f: metal2vulkan_air::ir::FuncId) -> String {
        self.sym(self.d.module.func(f).unwrap().name)
    }

    fn md(&self, id: MdId, stack: &mut Vec<MdId>) -> String {
        if stack.contains(&id) {
            return "<self>".into();
        }
        match self.d.module.md(id).unwrap() {
            Md::String(s) => format!("!{:?}", self.sym(*s)),
            Md::Value(c) => self.konst(*c),
            Md::Debug => "debug".into(),
            Md::Node { distinct, ops } => {
                stack.push(id);
                let o: Vec<_> = ops
                    .iter()
                    .map(|o| o.map_or_else(|| "null".into(), |o| self.md(o, stack)))
                    .collect();
                stack.pop();
                format!(
                    "{}!{{{}}}",
                    if *distinct { "distinct " } else { "" },
                    o.join(", ")
                )
            }
        }
    }
}

// ---- Tools and fixtures ----

/// Decode as the oracle's writer does: keeping the enum attributes the IR does not carry.
fn decode_bytes(bytes: &[u8]) -> Result<Decoded, String> {
    decode_as_written(bytes, &mut Meter::new(&Budget::default())).map_err(|e| format!("{e:?}"))
}

fn have_xcrun_metal() -> bool {
    cfg!(target_os = "macos") && crate::gpu::compile::has_tool("metal-as")
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public")
}

fn files(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_str().is_some_and(|s| s.ends_with(suffix)))
        .collect();
    out.sort();
    out
}

/// Typed AIR of an MSL source, compiled as `scripts/regen-air-fixtures` compiles a public
/// fixture's: `xcrun metal -c`, with `-std=` from a `// metal-std:` line. A source with
/// `// air-rename: <declared> <intrinsic>` lines declares functions it never defines that stand
/// for intrinsics MSL cannot spell (`llvm.agx3.yield`); its AIR is `xcrun metal-as` of its typed
/// text (`-S -emit-llvm`) with every call `@<declared>(` renamed `@<intrinsic>(`, the module its
/// `.air.ll` companion records, since the plain compile leaves the declaration unresolved and
/// `xcrun metallib` refuses it.
pub(crate) fn compile_msl(src: &Path, scratch: &ScratchDir) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(src).map_err(|e| e.to_string())?;
    let directive =
        |key: &'static str| text.lines().filter_map(move |l| l.trim().strip_prefix(key));
    let std = directive("// metal-std:")
        .next()
        .map(|s| format!("-std={}", s.trim()));
    let renames = directive("// air-rename:")
        .map(|r| match r.split_whitespace().collect::<Vec<_>>()[..] {
            [from, to] => Ok((format!("@{from}("), format!("@{to}("))),
            _ => Err(format!(
                "{}: air-rename needs <declared> <intrinsic>: {r}",
                src.display()
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (mode, out): (&[&str], _) = if renames.is_empty() {
        (&["-c"], scratch.path().join("msl.air"))
    } else {
        (&["-S", "-emit-llvm"], scratch.path().join("msl.ll"))
    };
    let args = mode
        .iter()
        .map(OsString::from)
        .chain(std.map(OsString::from))
        .chain([src.into(), "-o".into(), out.clone().into()]);
    let result = xcrun("metal", args).map_err(|e| e.to_string())?;
    if !result.success {
        return Err(result.stderr);
    }
    if renames.is_empty() {
        return std::fs::read(&out).map_err(|e| e.to_string());
    }
    let mut typed = std::fs::read_to_string(&out).map_err(|e| e.to_string())?;
    for (from, to) in &renames {
        typed = typed.replace(from, to);
    }
    metal_as(&typed).map_err(|e| e.to_string())
}

fn assemble_file(air_ll: &Path) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(air_ll).map_err(|e| e.to_string())?;
    metal_as(&text).map_err(|e| e.to_string())
}

/// decode → write → `metal-as` → decode, compared structurally (pointees included). Returns the
/// reassembled AIR.
fn round_trip(original: &Decoded) -> Result<Vec<u8>, String> {
    let text = write_decoded(original).map_err(|e| format!("write: {e}"))?;
    let bytes = metal_as(&text).map_err(|e| format!("{e}\n--- text ---\n{text}"))?;
    let again =
        decode_bytes(&bytes).map_err(|e| format!("re-decode: {e}\n--- text ---\n{text}"))?;
    let (a, b) = (structure(original, true), structure(&again, true));
    if a != b {
        let first = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        return Err(format!(
            "structure differs at line {first}:\n  original:   {:?}\n  round trip: {:?}\n--- text ---\n{text}",
            a.get(first),
            b.get(first)
        ));
    }
    Ok(bytes)
}

/// Package AIR into a metallib with `xcrun metallib`, as the Metal oracle does (CPU only).
fn metallib(air: &[u8], scratch: &ScratchDir) -> Result<(), String> {
    let (input, output) = (
        scratch.path().join("w.air"),
        scratch.path().join("w.metallib"),
    );
    std::fs::write(&input, air).map_err(|e| e.to_string())?;
    let result = xcrun(
        "metallib",
        [input.as_os_str(), "-o".as_ref(), output.as_os_str()],
    )
    .map_err(|e| e.to_string())?;
    if result.success {
        Ok(())
    } else {
        Err(format!("metallib: {}", result.stderr))
    }
}

// ---- Tests ----

#[test]
fn float_constants_widen_exactly() {
    for x in [
        0.0f32,
        -0.0,
        1.0,
        0.1,
        -3.0e38,
        f32::MIN_POSITIVE,
        1e-45,
        -7.0e-42,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(
            f32_bits_as_f64(x.to_bits()),
            f64::from(x).to_bits(),
            "{x:e}"
        );
    }
    // NaN payloads survive: quiet and signaling alike.
    assert_eq!(f32_bits_as_f64(0x7FC0_0001), 0x7FF8_0000_2000_0000);
    assert_eq!(f32_bits_as_f64(0xFF80_0001), 0xFFF0_0000_2000_0000);
}

/// Round trip every input. An input the decoder itself refuses is outside the writer's domain
/// (the writer spells decoded IR): it is listed visibly, not failed.
fn round_trip_all(
    what: &str,
    inputs: &[PathBuf],
    bitcode: impl Fn(&Path) -> Result<Vec<u8>, String>,
    package: Option<&ScratchDir>,
) {
    assert!(!inputs.is_empty(), "{what}: no inputs");
    let (mut failures, mut undecodable, mut written) = (Vec::new(), Vec::new(), 0);
    for path in inputs {
        let original = match bitcode(path) {
            Ok(bytes) => decode_bytes(&bytes),
            Err(e) => {
                failures.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let original = match original {
            Ok(d) => d,
            Err(e) => {
                undecodable.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let result = round_trip(&original)
            .and_then(|air| package.map_or(Ok(()), |scratch| metallib(&air, scratch)));
        match result {
            Ok(()) => written += 1,
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    for u in &undecodable {
        eprintln!("{what}: SKIP (the decoder refuses the input): {u}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(written > 0, "{what}: nothing round-tripped");
    eprintln!("{what}: {written} of {} inputs round-trip", inputs.len());
}

#[test]
fn typed_fixtures_round_trip() {
    if !have_xcrun_metal() {
        eprintln!("SKIP typed_fixtures_round_trip: xcrun metal-as not found");
        return;
    }
    let mut inputs = files(&fixtures(), ".air.ll");
    inputs.extend(files(&fixtures().join("engine"), ".air.ll"));
    // The written AIR also packages into a metallib, the oracle's next step.
    let scratch = ScratchDir::new("air-writer-typed").unwrap();
    round_trip_all(
        "typed_fixtures_round_trip",
        &inputs,
        assemble_file,
        Some(&scratch),
    );
}

#[test]
fn msl_fixtures_round_trip() {
    if !have_xcrun_metal() {
        eprintln!("SKIP msl_fixtures_round_trip: xcrun metal not found");
        return;
    }
    let scratch = ScratchDir::new("air-writer-msl").unwrap();
    let mut inputs = files(&fixtures(), ".metal");
    inputs.extend(files(&fixtures().join("engine"), ".metal"));
    let compile = |p: &Path| compile_msl(p, &scratch);
    round_trip_all("msl_fixtures_round_trip", &inputs, compile, None);
}

/// `llvm-as`: the `METAL2VULKAN_LLVM_AS` override, else `PATH`, else the Homebrew LLVM prefixes.
fn llvm_as() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("METAL2VULKAN_LLVM_AS") {
        return Some(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(
            ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
                .iter()
                .map(PathBuf::from),
        )
        .map(|d| d.join("llvm-as"))
        .find(|p| p.is_file())
}

/// Opaque-pointer inputs are outside the writer's contract (they carry no pointees, and one
/// pointer may serve several pointee types), but the writer still infers what it can. This
/// surveys every public opaque `.ll` fixture (`llvm-as`) and prints the outcomes. It asserts
/// that the writer never reports a malformed module and that some fixtures round-trip; it does
/// not require structural equality, because `metal-as` auto-upgrades what a typed-pointer LLVM
/// reads differently (it remangles opaque `llvm.memcpy.p0.p1` names, for one).
#[test]
fn opaque_fixtures_survey() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP opaque_fixtures_survey: llvm-as not found");
        return;
    };
    if !have_xcrun_metal() {
        eprintln!("SKIP opaque_fixtures_survey: xcrun metal-as not found");
        return;
    }
    let scratch = ScratchDir::new("air-writer-opaque").unwrap();
    let mut outcomes: std::collections::BTreeMap<String, usize> = Default::default();
    for path in files(&fixtures(), ".ll") {
        if path.to_str().is_some_and(|s| s.ends_with(".air.ll")) {
            continue;
        }
        let out = scratch.path().join("opaque.bc");
        let ok = Command::new(&tool)
            .arg(&path)
            .arg("-o")
            .arg(&out)
            .output()
            .is_ok_and(|o| o.status.success());
        let outcome = if !ok {
            "llvm-as rejects".to_owned()
        } else {
            match decode_bytes(&std::fs::read(&out).unwrap()) {
                Err(_) => "decoder refuses".to_owned(),
                Ok(d) => match write_decoded(&d) {
                    Err(super::WriteError::PointeeConflict { .. }) => {
                        "writer: pointee conflict".to_owned()
                    }
                    Err(super::WriteError::Unsupported(u)) => format!("writer: {u:?}"),
                    Err(e @ super::WriteError::Malformed(_)) => {
                        panic!("{}: {e}", path.display())
                    }
                    Ok(text) => match metal_as(&text).map(|b| decode_bytes(&b)) {
                        Err(_) => "metal-as rejects".to_owned(),
                        Ok(Err(_)) => "re-decode refuses".to_owned(),
                        Ok(Ok(again)) => {
                            let (a, b) = (structure(&d, false), structure(&again, false));
                            match a.iter().zip(&b).position(|(x, y)| x != y) {
                                None if a.len() == b.len() => "round-trips".to_owned(),
                                at => {
                                    let at = at.unwrap_or(a.len().min(b.len()));
                                    eprintln!(
                                        "DIFF {}\n  {:?}\n  {:?}",
                                        path.display(),
                                        a.get(at),
                                        b.get(at)
                                    );
                                    "structure differs".to_owned()
                                }
                            }
                        }
                    },
                },
            }
        };
        *outcomes.entry(outcome).or_default() += 1;
    }
    eprintln!("opaque_fixtures_survey: {outcomes:?}");
    assert!(outcomes.get("round-trips").is_some_and(|&n| n > 0));
}

/// Typed-pointer text that exercises what the fixtures do not: a loop with loop hints, PHIs of
/// pointers, a self-referential struct, pointers loaded from argument-buffer-like structs,
/// atomics, fences, a switch, aggregates, `freeze`, a `fastcc` callee, memory effects, `!range`,
/// constant GEP and `bitcast` expressions in initializers, a threadgroup global, and a
/// pointer-changing `bitcast` instruction.
const SYNTHETIC: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Node = type { i32, %struct.Node addrspace(1)*, [2 x float] }
%struct.Args = type { float addrspace(1)*, i32 addrspace(1)* }

@data = internal addrspace(2) constant [4 x i32] [i32 1, i32 2, i32 3, i32 -1], align 4
@f = internal addrspace(2) constant float 0x3FB99999A0000000, align 4
@table = internal addrspace(2) constant [2 x i32 addrspace(2)*] [i32 addrspace(2)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(2)* @data, i64 0, i64 1), i32 addrspace(2)* bitcast (float addrspace(2)* @f to i32 addrspace(2)*)], align 8
@tg = internal addrspace(3) global [8 x half] undef, align 2

define internal fastcc i32 @helper(%struct.Node addrspace(1)* %n, i32 %k) argmemonly readonly {
entry:
  br label %loop
loop:
  %p = phi %struct.Node addrspace(1)* [ %n, %entry ], [ %next, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc2, %body ]
  %i = phi i32 [ 0, %entry ], [ %i2, %body ]
  %done = icmp uge i32 %i, %k
  br i1 %done, label %exit, label %body, !llvm.loop !0
body:
  %vp = getelementptr inbounds %struct.Node, %struct.Node addrspace(1)* %p, i64 0, i32 0
  %v = load i32, i32 addrspace(1)* %vp, align 4, !range !3
  %acc2 = add nsw i32 %acc, %v
  %np = getelementptr inbounds %struct.Node, %struct.Node addrspace(1)* %p, i64 0, i32 1
  %next = load %struct.Node addrspace(1)*, %struct.Node addrspace(1)* addrspace(1)* %np, align 8
  %i2 = add nuw i32 %i, 1
  br label %loop
exit:
  ret i32 %acc
}

define void @k(%struct.Args addrspace(2)* noalias nocapture readonly %args, %struct.Node addrspace(1)* %list, i32 %sel, <2 x float> %vv) {
  %slot = alloca [2 x float], align 4
  %fp_p = getelementptr inbounds %struct.Args, %struct.Args addrspace(2)* %args, i64 0, i32 0
  %fp = load float addrspace(1)*, float addrspace(1)* addrspace(2)* %fp_p, align 8
  %ip_p = getelementptr inbounds %struct.Args, %struct.Args addrspace(2)* %args, i64 0, i32 1
  %ip = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(2)* %ip_p, align 8
  %h = tail call fastcc i32 @helper(%struct.Node addrspace(1)* %list, i32 %sel)
  %old = atomicrmw add i32 addrspace(1)* %ip, i32 %h monotonic, align 4
  %pair = cmpxchg weak i32 addrspace(1)* %ip, i32 %old, i32 7 seq_cst monotonic, align 4
  %ok = extractvalue { i32, i1 } %pair, 1
  %sw = select i1 %ok, i32 1, i32 2
  fence syncscope("singlethread") acquire
  switch i32 %sw, label %d [ i32 1, label %a
                             i32 -1, label %b ]
a:
  %e0 = extractelement <2 x float> %vv, i32 0
  %s0 = getelementptr inbounds [2 x float], [2 x float]* %slot, i64 0, i64 0
  store float %e0, float* %s0, align 4
  %agg = insertvalue { float, i32 } undef, float %e0, 0
  %fr = freeze { float, i32 } %agg
  %x = extractvalue { float, i32 } %fr, 0
  store atomic volatile float %x, float addrspace(1)* %fp seq_cst, align 4
  br label %d
b:
  %tp = getelementptr inbounds [8 x half], [8 x half] addrspace(3)* @tg, i64 0, i64 3
  store half 0xH3C00, half addrspace(3)* %tp, align 2
  %tv = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(2)* getelementptr inbounds ([2 x i32 addrspace(2)*], [2 x i32 addrspace(2)*] addrspace(2)* @table, i64 0, i64 1), align 8
  %t = load i32, i32 addrspace(2)* %tv, align 4
  %c = bitcast i32 addrspace(1)* %ip to <2 x i16> addrspace(1)*
  %sh = shufflevector <2 x float> %vv, <2 x float> undef, <4 x i32> <i32 1, i32 undef, i32 0, i32 0>
  store i32 %t, i32 addrspace(1)* %ip, align 4
  %cv = load <2 x i16>, <2 x i16> addrspace(1)* %c, align 4
  br label %d
d:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.count", i32 4}
!3 = !{i32 0, i32 100}
"#;

#[test]
fn synthetic_typed_module_round_trips() {
    if !have_xcrun_metal() {
        eprintln!("SKIP synthetic_typed_module_round_trips: xcrun metal-as not found");
        return;
    }
    let d = decode_bytes(&metal_as(SYNTHETIC).unwrap()).unwrap();
    // Not packaged: the module carries no AIR metadata, which `metallib` requires.
    round_trip(&d).unwrap();
    let text = write_decoded(&d).unwrap();
    // Pointees no parameter records are recovered from uses.
    assert!(
        text.contains(
            "load %\"struct.Node\" addrspace(1)*, %\"struct.Node\" addrspace(1)* addrspace(1)*"
        ),
        "{text}"
    );
    assert!(
        text.contains("load float addrspace(1)*, float addrspace(1)* addrspace(2)*"),
        "{text}"
    );
}

/// Assemble opaque-pointer `text` with `llvm-as` and decode it; `None` without `llvm-as`.
fn decode_opaque(text: &str) -> Option<Decoded> {
    let tool = llvm_as()?;
    let scratch = ScratchDir::new("air-writer-opaque-unit").unwrap();
    let (ll, bc) = (scratch.path().join("m.ll"), scratch.path().join("m.bc"));
    std::fs::write(&ll, text).unwrap();
    let out = Command::new(tool)
        .arg(&ll)
        .arg("-o")
        .arg(&bc)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(decode_bytes(&std::fs::read(&bc).unwrap()).unwrap())
}

#[test]
fn one_pointer_at_two_pointee_types_is_a_conflict() {
    let Some(d) = decode_opaque(
        "define float @f(ptr addrspace(1) %p) {\n  %a = load i32, ptr addrspace(1) %p\n  %b = load float, ptr addrspace(1) %p\n  ret float %b\n}\n",
    ) else {
        eprintln!("SKIP one_pointer_at_two_pointee_types_is_a_conflict: llvm-as not found");
        return;
    };
    assert!(matches!(
        write_decoded(&d),
        Err(super::WriteError::PointeeConflict { .. })
    ));
}

#[test]
fn memory_effects_without_an_old_spelling_are_refused() {
    let Some(d) = decode_opaque(
        "define void @f() memory(argmem: read, inaccessiblemem: write) {\n  ret void\n}\n",
    ) else {
        eprintln!("SKIP memory_effects_without_an_old_spelling_are_refused: llvm-as not found");
        return;
    };
    assert_eq!(
        write_decoded(&d),
        Err(super::WriteError::Unsupported(
            super::Unspellable::MemoryEffects
        ))
    );
}

#[test]
fn metal_as_never_accepts_garbage() {
    // `metal_as` never panics: it assembles, rejects, or reports the tool missing.
    match metal_as("this is not LLVM text") {
        Err(AssembleError::Rejected(_)) | Err(AssembleError::Unavailable(_)) => {}
        Err(AssembleError::Stopped(why)) => panic!("metal-as was stopped: {why}"),
        Ok(_) => panic!("metal-as accepted garbage"),
    }
}

/// LLVM types `llvm.global_ctors` itself (`[N x { i32, void ()*, i8* }]`), and the Metal compiler
/// emits it empty, so no use constrains its function pointer: the writer takes the pointees from
/// that contract (Q2: `metal-as` refused `[0 x { i32, i8*, i8* }]`), for an empty list and a
/// populated one alike.
#[test]
fn constructor_lists_keep_the_element_type_llvm_requires() {
    let Some(empty) = decode_opaque(
        "@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer\n\
         define void @f() {\n  ret void\n}\n",
    ) else {
        eprintln!("SKIP constructor_lists_keep_the_element_type_llvm_requires: llvm-as not found");
        return;
    };
    let text = write_decoded(&empty).unwrap();
    assert!(
        text.contains(
            "@\"llvm.global_ctors\" = appending global [0 x { i32, void ()*, i8* }] zeroinitializer"
        ),
        "{text}"
    );
    // The module interns no `void ()` of its own: the contract's function type is built.
    let Some(bare) = decode_opaque(
        "@llvm.global_dtors = appending global [0 x { i32, ptr, ptr }] zeroinitializer\n",
    ) else {
        return;
    };
    let text = write_decoded(&bare).unwrap();
    assert!(
        text.contains("[0 x { i32, void ()*, i8* }] zeroinitializer"),
        "{text}"
    );
    if !have_xcrun_metal() {
        eprintln!(
            "SKIP constructor_lists_keep_the_element_type_llvm_requires (metal-as): xcrun \
             metal-as not found"
        );
        return;
    }
    let typed = r#"
@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @init, i8* null }]

define internal void @init() {
  ret void
}
"#;
    let d = decode_bytes(&metal_as(typed).unwrap()).unwrap();
    round_trip(&d).unwrap();
}

// ---- What Metal's compiler relies on ----

/// Where a keyword is spelled in a probe module.
#[derive(Clone, Copy, Debug)]
enum Position {
    Function,
    Param,
    Return,
    CallSite,
}

/// The enum attributes of `position` in the probe module `d` (`@f`, or its call to `@g`).
fn enums_at(d: &Decoded, position: Position) -> EnumAttrs {
    let parts = d.module.parts();
    let (id, f) = parts
        .funcs
        .iter()
        .find(|(_, f)| parts.symbols.get(f.name) == Some(&b"f"[..]))
        .expect("@f");
    let site = d.enum_attrs.func(id).cloned().unwrap_or_default();
    match position {
        Position::Function => site.func,
        Position::Param => site.param(0),
        Position::Return => site.ret,
        Position::CallSite => f
            .body
            .as_ref()
            .and_then(|b| {
                b.insts.iter().find_map(|(i, inst)| match &inst.kind {
                    InstKind::Call(_) => d.enum_attrs.call(id, i).map(|c| c.func),
                    _ => None,
                })
            })
            .unwrap_or_default(),
    }
}

/// Every keyword of the writer's table is one `metal-as` parses, at the kind code the table gives:
/// assembled in the first position it is valid in, it decodes to that kind there, and the module
/// round-trips through the writer. Some keywords are valid only beside another (`optnone` with
/// `noinline`, `jumptable` with `unnamed_addr`) or on an intrinsic's parameter (`immarg`).
#[test]
fn enum_attribute_keywords_round_trip() {
    if !have_xcrun_metal() {
        eprintln!("SKIP enum_attribute_keywords_round_trip: xcrun metal-as not found");
        return;
    }
    let mut failures = Vec::new();
    for &(kind, kw) in ENUM_KEYWORDS {
        let probes = [
            (
                Position::Function,
                format!("define void @f() {kw} {{\n  ret void\n}}\n"),
            ),
            (
                Position::Function,
                format!("define void @f() noinline {kw} {{\n  ret void\n}}\n"),
            ),
            (
                Position::Function,
                format!("define void @f() unnamed_addr {kw} {{\n  ret void\n}}\n"),
            ),
            (
                Position::Param,
                format!("define i32 @f(i32 {kw} %a) {{\n  ret i32 %a\n}}\n"),
            ),
            (
                Position::Param,
                format!("define i8* @f(i8* {kw} %a) {{\n  ret i8* %a\n}}\n"),
            ),
            (
                Position::Param,
                format!(
                    "declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1 {kw})\n\n\
                     define void @f(i8* %p) {{\n  \
                     call void @llvm.memset.p0i8.i64(i8* %p, i8 0, i64 4, i1 false)\n  \
                     ret void\n}}\n"
                ),
            ),
            (
                Position::Return,
                format!("define {kw} i8* @f() {{\n  ret i8* null\n}}\n"),
            ),
            (
                Position::CallSite,
                format!(
                    "declare void @g()\n\ndefine void @f() {{\n  call void @g() {kw}\n  \
                     ret void\n}}\n"
                ),
            ),
        ];
        let assembled = probes
            .iter()
            .find_map(|(position, text)| metal_as(text).ok().map(|b| (*position, b)));
        let Some((position, bytes)) = assembled else {
            failures.push(format!(
                "{kind} {kw}: metal-as accepts it in no probe position"
            ));
            continue;
        };
        let d = match decode_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("{kind} {kw}: decode: {e}"));
                continue;
            }
        };
        // `immarg` sits on the intrinsic's parameter, not `@f`'s.
        let found = if kw == "immarg" {
            d.module.parts().funcs.ids().any(|f| {
                d.enum_attrs
                    .func(f)
                    .is_some_and(|site| site.params.iter().any(|p| p.contains(kind)))
            })
        } else {
            enums_at(&d, position).contains(kind)
        };
        if !found {
            failures.push(format!(
                "{kind} {kw}: decodes without kind {kind} ({position:?})"
            ));
            continue;
        }
        if let Err(e) = round_trip(&d) {
            failures.push(format!("{kind} {kw}: round trip: {e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What a `"deferred-static-alloca-size"` function and its call must keep for Metal's compiler to
/// delete the call once the allocation it sizes is dead (dropping them made pipeline
/// creation fail for the function-constant variants that leave a tensor type unused): the
/// function's `hidden`, `local_unnamed_addr`, `readnone`, `mustprogress nofree nosync
/// speculatable willreturn`, and the call's `nounwind`. The committed fixture `tns_mm_int` has
/// ten such functions, all called from the entry.
fn assert_deferred_sizes_as_written(d: &Decoded, what: &str) {
    let parts = d.module.parts();
    let deferred = |a: &FnAttrs| {
        a.strings
            .iter()
            .any(|&(k, _)| parts.symbols.get(k) == Some(&b"deferred-static-alloca-size"[..]))
    };
    let sizes: Vec<_> = parts
        .funcs
        .iter()
        .filter(|(_, f)| deferred(&f.attrs))
        .collect();
    assert_eq!(sizes.len(), 10, "{what}");
    let promises = [
        EnumAttrs::SPECULATABLE,
        EnumAttrs::WILL_RETURN,
        EnumAttrs::NO_FREE,
        EnumAttrs::NO_SYNC,
        EnumAttrs::MUST_PROGRESS,
    ];
    for &(id, f) in &sizes {
        let enums = d.enum_attrs.func(id).map(|s| s.func).unwrap_or_default();
        assert_eq!(
            (f.visibility, f.unnamed_addr, f.attrs.memory),
            (
                Visibility::Hidden,
                UnnamedAddr::Local,
                Some(MemoryEffects::uniform(ModRef::NoModRef))
            ),
            "{what}"
        );
        assert!(
            promises.iter().all(|&k| enums.contains(k)),
            "{what}: {:?}",
            enums.kinds().collect::<Vec<_>>()
        );
    }
    let mut calls = 0;
    for (caller, f) in parts.funcs.iter() {
        for (i, inst) in f.body.iter().flat_map(|b| b.insts.iter()) {
            let InstKind::Call(c) = &inst.kind else {
                continue;
            };
            let Callee::Direct(callee) = c.callee else {
                continue;
            };
            if sizes.iter().any(|&(id, _)| id == callee) {
                calls += 1;
                let enums = d
                    .enum_attrs
                    .call(caller, i)
                    .map(|s| s.func)
                    .unwrap_or_default();
                assert!(enums.contains(EnumAttrs::NO_UNWIND), "{what}");
            }
        }
    }
    assert_eq!(calls, 10, "{what}");
}

/// The writer keeps what Metal's compiler relies on: decode → write → `metal-as` → decode of a
/// committed fixture with deferred-size calls preserves every attribute and symbol flag above, and
/// so does the oracle's fuel transform, which hands Metal the written module.
#[test]
fn deferred_size_functions_keep_their_attributes_through_the_oracles_writer() {
    if !have_xcrun_metal() {
        eprintln!(
            "SKIP deferred_size_functions_keep_their_attributes_through_the_oracles_writer: xcrun \
             metal-as not found"
        );
        return;
    }
    let air = assemble_file(&fixtures().join("engine/tns_mm_int.air.ll")).unwrap();
    let original = decode_bytes(&air).unwrap();
    assert_deferred_sizes_as_written(&original, "as compiled");
    let written = decode_bytes(&round_trip(&original).unwrap()).unwrap();
    assert_deferred_sizes_as_written(&written, "written");
    let (instrumented, report) = crate::oracle_fuel::instrument_air(
        &air,
        "tns_mm_int",
        &crate::oracle_fuel::FuelConfig::oracle(30),
    )
    .unwrap();
    assert!(report.functions > 0);
    assert_deferred_sizes_as_written(&decode_bytes(&instrumented).unwrap(), "fuel-instrumented");
}
