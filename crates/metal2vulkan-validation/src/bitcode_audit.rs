//! The `bitcode-reader` audit lane: the engine's bitcode reader against `llvm-dis`, row by row.
//!
//! For one retained bitcode blob, the lane decodes and verifies it (including its AIR metadata
//! ABI, so an ABI refusal is a typed refusal too) through `metal2vulkan_air::bitcode::decode`
//! and disassembles the same bytes with `llvm-dis`. It then
//! compares a structural [`Fingerprint`] of each: every function (and, for a definition, its block
//! count), the opcode histogram, the callee histogram, and the global names. The text side is a
//! validation-local scanner of `llvm-dis` output; it never uses the old product parser. A decode
//! refusal is reported with its typed error; whether `llvm-dis` accepted the same bytes tells a
//! reader gap from input LLVM rejects too.
//!
//! `llvm-dis` auto-upgrades LLVM intrinsics when it reads older bitcode: it renames typed-pointer
//! manglings (`llvm.lifetime.end.p0i8` becomes `llvm.lifetime.end.p0`), rewrites call signatures
//! (LLVM 22 dropped `lifetime.start`'s size operand), and keeps the old declarations under new
//! names. The reader keeps the bytes as written. So `llvm.*` declarations are left out of
//! `functions`, and `llvm.*` callees are keyed by their family (the name up to the first
//! type-mangling component). Every other name, every block count, and every opcode compare
//! exactly.
//!
//! The same lane is the intrinsic census: every decoded row reports how many of its calls decoded
//! to `Callee::Air` / `Callee::Llvm` and which callee names stayed `Callee::UnknownIntrinsic`, so
//! decode completeness is measured over the corpus rather than asserted. Each unknown name is
//! reported as `listed` when it belongs to a family the decoder lists as not decoded
//! (`metal2vulkan_air::intrinsics::UNDECODED`), else `unlisted`: a closed census has no unlisted
//! names. It also names every function a row calls that the module declares without defining and
//! that is no intrinsic (`IntrinsicCensus::external`): what the row needs linked, such as a visible
//! function another library defines. The Metal runtime routines Apple's compiler calls for `frexp`
//! and `ilogb` decode as math.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use metal2vulkan::error::Error;
use metal2vulkan::Budget;
use metal2vulkan_air::bitcode::{decode, Decoded};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::intrinsics::dropped_family;
use metal2vulkan_air::ir::{Callee, CastOp, Const, ConstExpr, FuncId, InstKind, Value};
use serde::{Deserialize, Serialize};
use wait_timeout::ChildExt;

/// The per-attempt ceiling (AGENTS.md), applied to the decode's meter and to `llvm-dis`.
pub const ROW_TIMEOUT: Duration = Duration::from_secs(20);

/// The structure both sides must agree on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// Every function by name: `Some(blocks)` for a definition, `None` for a declaration
    /// (`llvm.*` declarations excluded; see the module docs).
    pub functions: BTreeMap<String, Option<usize>>,
    /// Instructions by LLVM opcode mnemonic (`br` covers both branch forms).
    pub opcodes: BTreeMap<String, usize>,
    /// Calls by callee name (`llvm.*` by family); `<indirect>` for a call through a
    /// non-function value.
    pub callees: BTreeMap<String, usize>,
    pub globals: BTreeSet<String>,
}

/// The family of an `llvm.*` intrinsic name: its components up to the first type mangling
/// (`p0i8`, `i64`, `v4f32`, `f16`, …). Other names are returned unchanged.
pub fn intrinsic_family(name: &str) -> &str {
    if !name.starts_with("llvm.") {
        return name;
    }
    let mut end = 0;
    for (i, part) in name.split('.').enumerate() {
        let mut chars = part.chars();
        let mangled = i > 1
            && matches!(chars.next(), Some('p' | 'i' | 'f' | 'v' | 'a'))
            && chars.next().is_some_and(|c| c.is_ascii_digit());
        if mangled {
            break;
        }
        end += part.len() + usize::from(i > 0);
    }
    &name[..end]
}

impl Fingerprint {
    fn declare(&mut self, name: String) {
        if !name.starts_with("llvm.") {
            self.functions.insert(name, None);
        }
    }

    fn call(&mut self, callee: &str) {
        *self
            .callees
            .entry(intrinsic_family(callee).to_string())
            .or_default() += 1;
    }

    /// The components that differ, each with its first difference; `None` when equal.
    pub fn diff(&self, text: &Fingerprint) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(d) = first_difference(&self.functions, &text.functions) {
            parts.push(format!("functions({d})"));
        }
        if let Some(d) = first_difference(&self.opcodes, &text.opcodes) {
            parts.push(format!("opcodes({d})"));
        }
        if let Some(d) = first_difference(&self.callees, &text.callees) {
            parts.push(format!("callees({d})"));
        }
        if self.globals != text.globals {
            let missing = text.globals.difference(&self.globals).next();
            let extra = self.globals.difference(&text.globals).next();
            parts.push(format!("globals(ir-only={extra:?} text-only={missing:?})"));
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

fn first_difference<V: PartialEq + std::fmt::Debug>(
    ir: &BTreeMap<String, V>,
    text: &BTreeMap<String, V>,
) -> Option<String> {
    let keys: BTreeSet<&String> = ir.keys().chain(text.keys()).collect();
    keys.into_iter().find_map(|k| {
        let (a, b) = (ir.get(k), text.get(k));
        (a != b).then(|| format!("{k}: ir={a:?} text={b:?}"))
    })
}

/// The LLVM opcode mnemonic of an instruction.
pub fn mnemonic(kind: &InstKind) -> &'static str {
    match kind {
        InstKind::BinOp { op, .. } => op.mnemonic(),
        InstKind::FNeg { .. } => "fneg",
        InstKind::Cast { op, .. } => op.mnemonic(),
        InstKind::ICmp { .. } => "icmp",
        InstKind::FCmp { .. } => "fcmp",
        InstKind::Select { .. } => "select",
        InstKind::Freeze(_) => "freeze",
        InstKind::ExtractElement { .. } => "extractelement",
        InstKind::InsertElement { .. } => "insertelement",
        InstKind::ShuffleVector { .. } => "shufflevector",
        InstKind::ExtractValue { .. } => "extractvalue",
        InstKind::InsertValue { .. } => "insertvalue",
        InstKind::Alloca { .. } => "alloca",
        InstKind::Load { .. } => "load",
        InstKind::Store { .. } => "store",
        InstKind::Gep { .. } => "getelementptr",
        InstKind::AtomicRmw { .. } => "atomicrmw",
        InstKind::CmpXchg { .. } => "cmpxchg",
        InstKind::Fence(_) => "fence",
        InstKind::Call(_) => "call",
        InstKind::Phi { .. } => "phi",
        InstKind::Ret(_) => "ret",
        InstKind::Br { .. } | InstKind::CondBr { .. } => "br",
        InstKind::Switch { .. } => "switch",
        InstKind::Unreachable => "unreachable",
    }
}

/// The fingerprint of a decoded module.
pub fn ir_fingerprint(d: &Decoded) -> Fingerprint {
    let m = &d.module;
    let name = |sym| String::from_utf8_lossy(m.symbols().get(sym).unwrap_or(b"?")).into_owned();
    let fn_name = |f: FuncId| m.func(f).map_or_else(|| "?".into(), |f| name(f.name));
    let mut out = Fingerprint::default();
    for g in m.parts().globals.values() {
        out.globals.insert(name(g.name));
    }
    for f in m.parts().funcs.values() {
        let Some(body) = &f.body else {
            out.declare(name(f.name));
            continue;
        };
        out.functions.insert(name(f.name), Some(body.blocks.len()));
        for inst in body.insts.values() {
            *out.opcodes
                .entry(mnemonic(&inst.kind).to_string())
                .or_default() += 1;
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            let callee = match &call.callee {
                Callee::Direct(f) | Callee::UnknownIntrinsic(f) => fn_name(*f),
                Callee::Air { decl, .. } | Callee::Llvm { decl, .. } => fn_name(*decl),
                // A call whose type differs from its callee's still names the function, also
                // through typed-pointer bitcasts, which opaque-pointer `llvm-dis` folds away.
                Callee::Indirect {
                    callee: Value::Const(c),
                    ..
                } => {
                    let mut c = *c;
                    loop {
                        match m.consts().get(c).map(|c| &c.value) {
                            Some(Const::FuncAddr(f)) => break fn_name(*f),
                            Some(Const::Expr(ConstExpr::Cast {
                                op: CastOp::Bitcast,
                                ops: [inner],
                            })) => c = *inner,
                            _ => break "<indirect>".into(),
                        }
                    }
                }
                Callee::Indirect { .. } => "<indirect>".into(),
            };
            out.call(&callee);
        }
    }
    out
}

/// The global name at the start of `s` (after `@`): plain, or quoted with `\XX` escapes.
fn global_name(s: &str) -> Option<(String, &str)> {
    if let Some(rest) = s.strip_prefix('"') {
        let end = rest.find('"')?;
        let raw = rest.get(..end)?.as_bytes();
        let mut bytes = Vec::with_capacity(raw.len());
        let mut i = 0;
        while let Some(&b) = raw.get(i) {
            if b == b'\\' {
                let hex = std::str::from_utf8(raw.get(i + 1..i + 3)?).ok()?;
                bytes.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            } else {
                bytes.push(b);
                i += 1;
            }
        }
        return Some((
            String::from_utf8_lossy(&bytes).into_owned(),
            rest.get(end + 1..)?,
        ));
    }
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || "-$._".contains(c)))
        .unwrap_or(s.len());
    (end > 0).then(|| (s[..end].to_string(), &s[end..]))
}

const TERMINATORS: [&str; 4] = ["ret", "br", "switch", "unreachable"];

/// The fingerprint of `llvm-dis` text.
pub fn text_fingerprint(ll: &str) -> Result<Fingerprint, String> {
    let mut out = Fingerprint::default();
    // (function name, blocks) while inside a definition.
    let mut current: Option<(String, usize)> = None;
    let mut in_switch = false;
    for (n, line) in ll.lines().enumerate() {
        let line = line.trim();
        let Some((_, blocks)) = current.as_mut() else {
            if let Some(rest) = line
                .strip_prefix("define ")
                .or_else(|| line.strip_prefix("declare "))
            {
                let at = rest
                    .find('@')
                    .ok_or(format!("line {n}: no function name"))?;
                let (name, _) =
                    global_name(&rest[at + 1..]).ok_or(format!("line {n}: bad function name"))?;
                if line.starts_with("define ") {
                    current = Some((name, 0));
                } else {
                    out.declare(name);
                }
            } else if let Some(rest) = line.strip_prefix('@') {
                let (name, tail) = global_name(rest).ok_or(format!("line {n}: bad global name"))?;
                if tail.trim_start().starts_with('=') {
                    out.globals.insert(name);
                }
            }
            continue;
        };
        if line == "}" {
            let (name, blocks) = current.take().ok_or("unbalanced function")?;
            out.functions.insert(name, Some(blocks));
            continue;
        }
        if in_switch {
            in_switch = line != "]";
            continue;
        }
        let head = line.split(';').next().unwrap_or_default().trim_end();
        // Empty lines, comments, and block labels (`name:` with an optional `; preds` comment).
        if head.is_empty() || (head.ends_with(':') && !head.contains(' ')) {
            continue;
        }
        let body = match head.find(" = ") {
            Some(at) if head.starts_with('%') => &head[at + 3..],
            _ => head,
        };
        let mut words = body.split_whitespace();
        let mut op = words.next().unwrap_or_default();
        if matches!(op, "tail" | "musttail" | "notail") {
            op = words.next().unwrap_or_default();
        }
        if op == "switch" && body.ends_with('[') {
            in_switch = true;
        }
        if TERMINATORS.contains(&op) {
            *blocks += 1;
        }
        if op == "call" {
            // A direct callee is the first `@name` immediately followed by its argument list.
            let callee = body
                .find('@')
                .and_then(|at| global_name(&body[at + 1..]))
                .filter(|(_, rest)| rest.starts_with('('))
                .map_or_else(|| "<indirect>".to_string(), |(name, _)| name);
            // Decode drops these marker calls, so the decoded fingerprint never counts them.
            if dropped_family(&callee).is_some() {
                continue;
            }
            // Decode drops the call of MSL's empty inline assembly too: `asm sideeffect "", ""`
            // at the call type `void ()`, with no flag but `sideeffect` (`llvm-dis` prints a flag
            // or a vararg type before the template, and an operand inside the parentheses).
            let rest = body.split_once("call ").map_or("", |(_, rest)| rest);
            if rest
                .strip_prefix("void asm sideeffect \"\", \"\"()")
                .is_some_and(|tail| tail.is_empty() || tail.starts_with([' ', ',']))
            {
                continue;
            }
            *out.opcodes.entry(op.to_string()).or_default() += 1;
            out.call(&callee);
        } else {
            *out.opcodes.entry(op.to_string()).or_default() += 1;
        }
    }
    if current.is_some() {
        return Err("unterminated function".into());
    }
    Ok(out)
}

/// `llvm-dis`: `METAL2VULKAN_LLVM_DIS`, the Homebrew LLVM prefixes, then `PATH`.
pub fn resolve_llvm_dis() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("METAL2VULKAN_LLVM_DIS") {
        return Ok(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
        .iter()
        .map(PathBuf::from)
        .chain(std::env::split_paths(&path))
        .map(|d| d.join("llvm-dis"))
        .find(|p| p.is_file())
        .ok_or_else(|| "llvm-dis not found (set METAL2VULKAN_LLVM_DIS)".into())
}

/// Disassemble `bytes` through `llvm-dis` on pipes, within [`ROW_TIMEOUT`].
pub fn llvm_dis(tool: &Path, bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new(tool)
        .args(["-", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", tool.display()))?;
    let (Some(mut stdin), Some(mut stdout), Some(mut stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        return Err("llvm-dis pipes unavailable".into());
    };
    std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(bytes));
        let reader = scope.spawn(move || {
            let mut text = String::new();
            stdout.read_to_string(&mut text).map(|_| text)
        });
        let errors = scope.spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        let status = match child.wait_timeout(ROW_TIMEOUT) {
            Ok(Some(status)) => status,
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("llvm-dis timed out".into());
            }
            Err(e) => return Err(format!("wait for llvm-dis: {e}")),
        };
        // A write error means llvm-dis stopped reading; its status says why.
        let _ = writer.join();
        let text = reader
            .join()
            .map_err(|_| "llvm-dis reader panicked".to_string())?
            .map_err(|e| format!("read llvm-dis output: {e}"))?;
        let errors = errors.join().unwrap_or_default();
        if status.success() {
            Ok(text)
        } else {
            let first = errors.lines().next().unwrap_or_default();
            Err(format!("llvm-dis exited {status}: {first}"))
        }
    })
}

/// How one row came out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Decoded, verified, and structurally equal to the `llvm-dis` text.
    Match,
    /// Decoded and verified, but the fingerprints differ.
    Mismatch(String),
    /// The reader refused the row with a typed error.
    Refused {
        /// `malformed`, `unsupported`, `limit`, `budget`, or `internal`.
        class: &'static str,
        detail: String,
        /// Whether `llvm-dis` accepted the same bytes.
        llvm_accepts: bool,
    },
    /// The reader decoded the row but `llvm-dis` did not.
    DisassemblyFailed(String),
    /// The row retains no bitcode blob.
    NoBlob,
    /// The row retains a blob that is not valid base64, so no bitcode reaches the reader.
    InvalidBlob(String),
    /// The worker boundary stopped the reader: the stop's outcome key (`timeout`, `oom`,
    /// `worker_exit`, ...) and its message.
    Stopped { key: String, detail: String },
}

impl Outcome {
    /// A stable summary key for histograms.
    pub fn shape(&self) -> String {
        match self {
            Outcome::Match => "match".into(),
            Outcome::Mismatch(d) => format!("mismatch {d}"),
            Outcome::Refused {
                class,
                detail,
                llvm_accepts,
            } => format!(
                "{class} {detail} llvm={}",
                if *llvm_accepts { "accepts" } else { "rejects" }
            ),
            Outcome::DisassemblyFailed(e) => format!("llvm-dis-failed {e}"),
            Outcome::NoBlob => "no-blob".into(),
            Outcome::InvalidBlob(e) => format!("invalid-blob {e}"),
            Outcome::Stopped { key, .. } => format!("worker-stopped {key}"),
        }
    }
}

/// The intrinsic calls of one decoded module.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntrinsicCensus {
    /// Calls decoded to an `AirIntrinsic`.
    pub air: usize,
    /// Calls decoded to an `LlvmIntrinsic`.
    pub llvm: usize,
    /// Calls kept as `Callee::UnknownIntrinsic`, by callee name.
    pub unknown: BTreeMap<String, usize>,
    /// Direct calls to a function the module declares but does not define and that is no
    /// intrinsic, by callee name: what the module needs linked (a visible function's, which the
    /// engine refuses as `Linking` where one survives specialization).
    #[serde(default)]
    pub external: BTreeMap<String, usize>,
}

/// The intrinsic census of a decoded module.
pub fn intrinsic_census(d: &Decoded) -> IntrinsicCensus {
    let m = &d.module;
    let name = |f: FuncId| {
        m.func(f)
            .and_then(|f| m.symbols().get(f.name))
            .map_or_else(|| "?".into(), |n| String::from_utf8_lossy(n).into_owned())
    };
    let mut out = IntrinsicCensus::default();
    for f in m.parts().funcs.values() {
        let Some(body) = &f.body else {
            continue;
        };
        for inst in body.insts.values() {
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            match &call.callee {
                Callee::Air { .. } => out.air += 1,
                Callee::Llvm { .. } => out.llvm += 1,
                Callee::UnknownIntrinsic(decl) => {
                    *out.unknown.entry(name(*decl)).or_default() += 1;
                }
                Callee::Direct(callee) if m.func(*callee).is_some_and(|f| f.body.is_none()) => {
                    *out.external.entry(name(*callee)).or_default() += 1;
                }
                Callee::Direct(_) | Callee::Indirect { .. } => {}
            }
        }
    }
    out
}

/// One audited row.
#[derive(Clone, Debug)]
pub struct RowResult {
    pub outcome: Outcome,
    /// The intrinsic census, when the reader decoded the row.
    pub census: Option<IntrinsicCensus>,
    /// Decode and verification wall time.
    pub decode: Duration,
    /// The meter's ledger peak (bytes) and work units.
    pub ledger_peak: u64,
    pub work: u64,
    /// What the parent measured around the reader's worker ([`audit_row_in_worker`]), stopped or
    /// not; `None` in process or when no worker ran.
    pub worker: Option<crate::worker::Measured>,
}

/// The reader's class of a typed refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefusalClass {
    Malformed,
    Unsupported,
    Limit,
    Budget,
    Internal,
}

impl RefusalClass {
    /// The class as the lane's shapes spell it.
    pub fn as_str(self) -> &'static str {
        match self {
            RefusalClass::Malformed => "malformed",
            RefusalClass::Unsupported => "unsupported",
            RefusalClass::Limit => "limit",
            RefusalClass::Budget => "budget",
            RefusalClass::Internal => "internal",
        }
    }
}

/// The reader's typed refusal of a blob.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    pub class: RefusalClass,
    pub detail: String,
}

/// A module the reader decoded: its structural fingerprint and its intrinsic census.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadModule {
    pub fingerprint: Fingerprint,
    pub census: IntrinsicCensus,
}

/// What the engine's reader made of one blob under the per-attempt meter: the part of a row
/// that runs the product, which the audit runs in a killable worker
/// ([`crate::worker::audit::read`]); `llvm-dis` and the comparison ([`judge`]) stay in the parent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reading {
    /// The decoded module, or the reader's refusal.
    pub read: Result<ReadModule, Refusal>,
    /// Decode and verification wall time.
    pub decode: Duration,
    /// The meter's ledger peak (bytes) and work units.
    pub ledger_peak: u64,
    pub work: u64,
}

fn classify(e: &Error) -> Refusal {
    let (class, detail) = match e {
        Error::InvalidInput(i) => (RefusalClass::Malformed, format!("{:?}", i.kind)),
        Error::Unsupported(u) => (RefusalClass::Unsupported, format!("{:?}", u.feature)),
        Error::LimitExceeded(l) => (RefusalClass::Limit, format!("{l:?}")),
        Error::Budget(b) => (RefusalClass::Budget, format!("{b:?}")),
        other => (RefusalClass::Internal, format!("{other:?}")),
    };
    Refusal { class, detail }
}

/// A row with no reading: no blob, a blob that is not base64, or a stop at the worker boundary.
fn not_read(outcome: Outcome, worker: Option<crate::worker::Measured>) -> RowResult {
    RowResult {
        outcome,
        census: None,
        decode: Duration::ZERO,
        ledger_peak: 0,
        work: 0,
        worker,
    }
}

/// Audit one indexed row's retained blob (`blob_b64`) against `llvm-dis` at `tool`, in process. A
/// row without a blob and a row whose blob is not valid base64 are reported apart, so a corrupt
/// shard row is never counted as a source that simply retains no bitcode.
pub fn audit_row(blob_b64: Option<&str>, tool: &Path) -> RowResult {
    match blob_b64.map(|b| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b)) {
        Some(Ok(bytes)) => audit_blob(&bytes, tool),
        Some(Err(e)) => not_read(Outcome::InvalidBlob(e.to_string()), None),
        None => not_read(Outcome::NoBlob, None),
    }
}

/// [`audit_row`] with the reader in a killable worker of `program`: a stop at the boundary is the
/// row's [`Outcome::Stopped`], measured like a reading.
pub fn audit_row_in_worker(program: &Path, blob_b64: Option<&str>, tool: &Path) -> RowResult {
    let Some(blob_b64) = blob_b64 else {
        return not_read(Outcome::NoBlob, None);
    };
    let bytes = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, blob_b64) {
        Ok(bytes) => bytes,
        Err(e) => return not_read(Outcome::InvalidBlob(e.to_string()), None),
    };
    match crate::worker::audit::read(program, blob_b64) {
        Ok((reading, measured)) => RowResult {
            worker: Some(measured),
            ..judge(reading, llvm_dis(tool, &bytes))
        },
        Err(failure) => not_read(
            Outcome::Stopped {
                key: failure.outcome().key(),
                detail: failure.message(),
            },
            failure.measured(),
        ),
    }
}

/// Audit one bitcode blob (raw or wrapped) against `llvm-dis` at `tool`, in process.
pub fn audit_blob(bytes: &[u8], tool: &Path) -> RowResult {
    let reading = read_blob(bytes);
    judge(reading, llvm_dis(tool, bytes))
}

/// The reader's [`Reading`] of `bytes`, under the per-attempt meter (a panic is an `internal`
/// refusal).
pub fn read_blob(bytes: &[u8]) -> Reading {
    let mut meter = Meter::new(&Budget::with_deadline(
        Instant::now().checked_add(ROW_TIMEOUT),
    ));
    let start = Instant::now();
    let decoded =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(bytes, &mut meter)));
    let elapsed = start.elapsed();
    let read = match decoded {
        Err(_) => Err(Refusal {
            class: RefusalClass::Internal,
            detail: "decoder panicked".into(),
        }),
        Ok(Err(e)) => Err(classify(&metal2vulkan::error::Error::from(e))),
        Ok(Ok(d)) => Ok(ReadModule {
            fingerprint: ir_fingerprint(&d),
            census: intrinsic_census(&d),
        }),
    };
    Reading {
        read,
        decode: elapsed,
        ledger_peak: meter.ledger_peak(),
        work: meter.work(),
    }
}

/// The row's result from the reader's `reading` and `llvm-dis`'s `text` of the same bytes.
pub fn judge(reading: Reading, text: Result<String, String>) -> RowResult {
    let Reading {
        read,
        decode,
        ledger_peak,
        work,
    } = reading;
    let (outcome, census) = match read {
        Err(Refusal { class, detail }) => (
            Outcome::Refused {
                class: class.as_str(),
                detail,
                llvm_accepts: text.is_ok(),
            },
            None,
        ),
        Ok(ReadModule {
            fingerprint,
            census,
        }) => {
            let outcome = match text {
                Err(e) => Outcome::DisassemblyFailed(e),
                Ok(text) => match text_fingerprint(&text) {
                    Err(e) => Outcome::DisassemblyFailed(format!("unscannable text: {e}")),
                    Ok(t) => match fingerprint.diff(&t) {
                        None => Outcome::Match,
                        Some(diff) => Outcome::Mismatch(diff),
                    },
                },
            };
            (outcome, Some(census))
        }
    };
    RowResult {
        outcome,
        census,
        decode,
        ledger_peak,
        work,
        worker: None,
    }
}

/// The process's peak resident set, in bytes.
pub fn peak_rss_bytes() -> u64 {
    // SAFETY: `getrusage` fills the zeroed struct it is given and has no other effect.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    let max = u64::try_from(usage.ru_maxrss).unwrap_or(0);
    // macOS reports bytes, Linux kibibytes.
    if cfg!(target_os = "macos") {
        max
    } else {
        max.saturating_mul(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"
@g = internal addrspace(2) constant [2 x i32] [i32 1, i32 2], align 4
@"quoted\22name" = external global i32

define void @k(ptr addrspace(1) %0, i32 %1) {
  %3 = tail call i32 @air.x(i32 %1) #1
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi i32 [ %3, %2 ], [ %6, %4 ]
  %6 = add nsw i32 %5, 3
  switch i32 %6, label %4 [
    i32 0, label %7
    i32 1, label %7
  ]

7:                                                ; preds = %4
  call void %0(i32 %6)
  store i32 %6, ptr addrspace(1) @g, align 4
  ret void
}

declare i32 @air.x(i32)
"#;

    #[test]
    fn the_text_scanner_reads_llvm_dis_output() {
        let f = text_fingerprint(TEXT).unwrap();
        let functions: Vec<_> = f.functions.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(functions, vec![("air.x", None), ("k", Some(3))]);
        let ops: Vec<_> = f.opcodes.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(
            ops,
            vec![
                ("add", 1),
                ("br", 1),
                ("call", 2),
                ("phi", 1),
                ("ret", 1),
                ("store", 1),
                ("switch", 1)
            ]
        );
        let callees: Vec<_> = f.callees.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(callees, vec![("<indirect>", 1), ("air.x", 1)]);
        let globals: Vec<_> = f.globals.iter().map(String::as_str).collect();
        assert_eq!(globals, vec!["g", "quoted\"name"]);
        assert!(text_fingerprint("define void @k() {\n  ret void\n").is_err());
    }

    /// The call of MSL's empty inline assembly, which decode drops, is not counted; any other
    /// inline assembly stays an indirect call.
    #[test]
    fn the_text_scanner_skips_the_empty_inline_assembly_decode_drops() {
        let text = r#"define void @k(i32 %0) {
  tail call void asm sideeffect "", ""() #3, !srcloc !34
  call void asm sideeffect "", ""(), !srcloc !35
  call void asm sideeffect alignstack "", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "", "r"(i32 %0)
  call void asm "", ""()
  ret void
}
"#;
        let f = text_fingerprint(text).unwrap();
        assert_eq!(f.opcodes.get("call"), Some(&4));
        let callees: Vec<_> = f.callees.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(callees, vec![("<indirect>", 4)]);
    }

    #[test]
    fn intrinsics_compare_by_family() {
        assert_eq!(
            intrinsic_family("llvm.lifetime.end.p0i8"),
            "llvm.lifetime.end"
        );
        assert_eq!(
            intrinsic_family("llvm.lifetime.end.p0"),
            "llvm.lifetime.end"
        );
        assert_eq!(intrinsic_family("llvm.memcpy.p1i8.p0i8.i64"), "llvm.memcpy");
        assert_eq!(intrinsic_family("llvm.fmuladd.v4f32"), "llvm.fmuladd");
        assert_eq!(
            intrinsic_family("llvm.agx3.load.with.emask.global.v4i32"),
            "llvm.agx3.load.with.emask.global"
        );
        assert_eq!(
            intrinsic_family("air.convert.f.f32.u.i32"),
            "air.convert.f.f32.u.i32"
        );
        let mut f = Fingerprint::default();
        f.declare("llvm.lifetime.end.i64".into());
        f.declare("air.x".into());
        let declared: Vec<_> = f.functions.keys().map(String::as_str).collect();
        assert_eq!(declared, vec!["air.x"]);
    }

    #[test]
    fn differences_name_their_component() {
        let a = text_fingerprint(TEXT).unwrap();
        assert_eq!(a.diff(&a), None);
        let mut b = a.clone();
        *b.opcodes.get_mut("add").unwrap() += 1;
        b.globals.insert("h".into());
        let d = a.diff(&b).unwrap();
        assert!(
            d.starts_with("opcodes(add: ir=Some(1) text=Some(2))"),
            "{d}"
        );
        assert!(
            d.contains("globals(ir-only=None text-only=Some(\"h\"))"),
            "{d}"
        );
    }

    /// A call through a (chain of) typed-pointer bitcast(s) of a function names that function,
    /// as opaque-pointer `llvm-dis` prints it; any other constant callee stays `<indirect>`.
    #[test]
    fn bitcast_callees_name_their_function() {
        use metal2vulkan_air::bitcode::PointeeTable;
        use metal2vulkan_air::ir::build::ModuleBuilder;
        use metal2vulkan_air::ir::{AddrSpace, Call, FastMath, TailKind};
        let mut b = ModuleBuilder::new();
        let void = b.void();
        let fn_ty = b.fn_ty(void, &[]);
        let target = b.declare("target", fn_ty);
        let ptr = b.ptr(AddrSpace::Thread);
        let addr = b.constant(ptr, Const::FuncAddr(target));
        let cast = |b: &mut ModuleBuilder, c| {
            b.constant(
                ptr,
                Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast,
                    ops: [c],
                }),
            )
        };
        let once = cast(&mut b, addr);
        let twice = cast(&mut b, once);
        let int = b.int(64);
        let other = b.constant(int, Const::Int(1));
        let not_fn = b.constant(
            ptr,
            Const::Expr(ConstExpr::Cast {
                op: CastOp::IntToPtr,
                ops: [other],
            }),
        );
        let mut f = b.define("k", fn_ty);
        for c in [twice, not_fn] {
            f.push_void(InstKind::Call(Box::new(Call {
                callee: Callee::Indirect {
                    callee: Value::Const(c),
                    fn_ty,
                },
                args: Box::new([]),
                fmf: FastMath::NONE,
                tail: TailKind::None,
                attrs: None,
                range: None,
            })));
        }
        f.ret(None);
        f.finish();
        let module = b.finish().unwrap();
        let pointees = PointeeTable::default();
        let mut meter = Meter::new(&Budget::default());
        let abi = metal2vulkan_air::abi::decode(&module, &pointees, &mut meter).unwrap();
        let d = Decoded {
            module,
            abi,
            pointees,
            enum_attrs: Default::default(),
        };
        let fp = ir_fingerprint(&d);
        let callees: Vec<_> = fp.callees.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(callees, vec![("<indirect>", 1), ("target", 1)]);
    }

    /// The census counts a direct call to a declared, undefined function that is no intrinsic (a
    /// linked function) by name, apart from the intrinsics; a call to a
    /// defined function is no external call.
    #[test]
    fn the_census_counts_calls_to_undefined_functions() {
        use metal2vulkan_air::bitcode::PointeeTable;
        use metal2vulkan_air::ir::build::ModuleBuilder;
        use metal2vulkan_air::ir::{Call, FastMath, TailKind};
        let mut b = ModuleBuilder::new();
        let void = b.void();
        let fn_ty = b.fn_ty(void, &[]);
        let linked = b.declare("linked_visible_function", fn_ty);
        let unknown = b.declare("air.some_future_op", fn_ty);
        let mut h = b.define("helper", fn_ty);
        h.ret(None);
        let helper = h.finish();
        let call = |callee| {
            InstKind::Call(Box::new(Call {
                callee,
                args: Box::new([]),
                fmf: FastMath::NONE,
                tail: TailKind::None,
                attrs: None,
                range: None,
            }))
        };
        let mut f = b.define("k", fn_ty);
        for callee in [
            Callee::Direct(linked),
            Callee::Direct(linked),
            Callee::Direct(helper),
            Callee::UnknownIntrinsic(unknown),
        ] {
            f.push_void(call(callee));
        }
        f.ret(None);
        f.finish();
        let module = b.finish().unwrap();
        let pointees = PointeeTable::default();
        let mut meter = Meter::new(&Budget::default());
        let abi = metal2vulkan_air::abi::decode(&module, &pointees, &mut meter).unwrap();
        let d = Decoded {
            module,
            abi,
            pointees,
            enum_attrs: Default::default(),
        };
        let census = intrinsic_census(&d);
        let external: Vec<_> = census
            .external
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        assert_eq!(external, vec![("linked_visible_function", 2)]);
        let unknown: Vec<_> = census
            .unknown
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        assert_eq!(unknown, vec![("air.some_future_op", 1)]);
    }

    fn llvm_as() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("METAL2VULKAN_LLVM_AS") {
            return Some(PathBuf::from(p));
        }
        let path = std::env::var_os("PATH").unwrap_or_default();
        ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
            .iter()
            .map(PathBuf::from)
            .chain(std::env::split_paths(&path))
            .map(|d| d.join("llvm-as"))
            .find(|p| p.is_file())
    }

    /// Every public fixture `llvm-as` accepts audits as a match (the lane's end-to-end path on
    /// committed inputs; the corpus sweep is developer-local).
    #[test]
    fn public_fixtures_audit_as_matches() {
        let (Some(as_tool), Ok(dis_tool)) = (llvm_as(), resolve_llvm_dis()) else {
            eprintln!("SKIP public_fixtures_audit_as_matches: llvm-as or llvm-dis not found");
            return;
        };
        let scratch = crate::ScratchDir::new("bitcode-audit").unwrap();
        let out = scratch.path().join("fixture.bc");
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public");
        let mut fixtures: Vec<PathBuf> = [root.clone(), root.join("engine")]
            .iter()
            .flat_map(|d| std::fs::read_dir(d).unwrap())
            .map(|e| e.unwrap().path())
            .filter(|p| {
                let n = p.to_string_lossy();
                n.ends_with(".ll") && !n.ends_with(".air.ll")
            })
            .collect();
        fixtures.sort();
        let mut matched = 0;
        let mut abi_refused = Vec::new();
        let mut air_calls = 0;
        let mut unknown = BTreeMap::<String, usize>::new();
        for ll in fixtures {
            let ok = Command::new(&as_tool)
                .arg(&ll)
                .arg("-o")
                .arg(&out)
                .output()
                .is_ok_and(|o| o.status.success());
            if !ok {
                // A legacy top-level fixture this llvm-as rejects (see tests/air_input.rs).
                continue;
            }
            let bytes = std::fs::read(&out).unwrap();
            let r = audit_blob(&bytes, &dis_tool);
            // Decode includes the metadata ABI, so a legacy top-level fixture whose hand-written
            // metadata is not the AIR ABI grammar is refused (tests/air_abi.rs pins which). A
            // engine fixture always matches.
            if let Outcome::Refused {
                class: "malformed",
                detail,
                llvm_accepts: true,
            } = &r.outcome
            {
                if detail.starts_with("Abi(") && !ll.starts_with(root.join("engine")) {
                    abi_refused.push(ll);
                    continue;
                }
            }
            assert_eq!(r.outcome, Outcome::Match, "{ll:?}");
            let census = r.census.as_ref().expect("a decoded row has a census");
            air_calls += census.air;
            for (name, calls) in &census.unknown {
                *unknown.entry(name.clone()).or_default() += calls;
            }
            assert!(r.ledger_peak > 0 && r.work > 0);
            matched += 1;
        }
        assert!(matched > 100, "only {matched} fixtures audited");
        if !abi_refused.is_empty() {
            eprintln!("legacy fixtures refused for their ABI metadata: {abi_refused:?}");
        }
        // The census sees the decoded calls, and every public intrinsic call decodes or belongs
        // to a family the decoder lists as not decoded.
        assert!(air_calls > 800, "only {air_calls} air.* calls decoded");
        let unlisted: Vec<_> = unknown
            .keys()
            .filter(|name| metal2vulkan_air::intrinsics::undecoded_family(name).is_none())
            .collect();
        assert!(
            unlisted.is_empty(),
            "unlisted unknown intrinsics: {unlisted:?}"
        );
        // Bytes neither side accepts are refused with a typed error.
        let r = audit_blob(b"not bitcode", &dis_tool);
        assert!(
            matches!(
                r.outcome,
                Outcome::Refused {
                    class: "malformed",
                    llvm_accepts: false,
                    ..
                }
            ),
            "{:?}",
            r.outcome
        );
    }

    /// A row whose retained blob is not valid base64 is reported as such, never as a row that
    /// retains no blob; neither reaches the reader or `llvm-dis`.
    #[test]
    fn a_corrupt_blob_is_not_reported_as_a_missing_one() {
        let tool = Path::new("/nonexistent/llvm-dis");
        assert_eq!(audit_row(None, tool).outcome, Outcome::NoBlob);
        let r = audit_row(Some("not*base64"), tool);
        assert!(
            matches!(&r.outcome, Outcome::InvalidBlob(e) if !e.is_empty()),
            "{:?}",
            r.outcome
        );
        assert!(r.outcome.shape().starts_with("invalid-blob "));
        assert!(r.census.is_none() && r.work == 0);
    }
}
