//! LLVM's bit-manipulation intrinsics: `llvm.bitreverse`, `llvm.bswap`, `llvm.ctpop`,
//! `llvm.ctlz`, `llvm.cttz`, `llvm.fshl` and `llvm.fshr`.
//!
//! The owned fixture `engine/kernel_bit_manipulation` holds every one Metal's compiler emits from
//! MSL (clang's builtins and the shift-and-mask idioms LLVM recognizes): it translates on every
//! committed profile, and runs on the CPU interpreter against its definition over edge operands.
//! Synthetic modules cover the whole family besides: each intrinsic on `i8`, `i16`, `i32` and
//! `i64`, scalar and on 2-, 3- and 4-lane vectors, `ctlz` and `cttz` with the zero-poison flag
//! false (a zero operand answers the width) and true (no zero operand), and the funnel shifts on
//! two distinct operands with amounts past the width. Each runs on `moltenvk-m3` with
//! `shaderInt64` (64-bit containers) and without it (two 32-bit words), the two lowerings a 64-bit
//! operand takes; `i8` and `i16` are held in 32-bit containers on every profile. `llvm-as` and `xcrun
//! metal-as` assemble; a missing tool skips visibly. No GPU is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text, metal_assemble,
    typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const STEM: &str = "kernel_bit_manipulation";

/// The committed `moltenvk-m3` description with `shaderInt64` (64-bit containers), and without
/// it (two 32-bit words): the two lowerings a 64-bit operand takes.
fn profiles() -> [(&'static str, DeviceProfile); 2] {
    [("int64", true), ("two-word", false)].map(|(label, int64)| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        if !int64 {
            assert!(desc.features.remove(&VkFeature::ShaderInt64));
        }
        let p = Device::new(desc, DevicePolicy::default())
            .expect("device")
            .profile()
            .clone();
        assert_eq!(p.fields().leaf.int64, int64);
        (label, p)
    })
}

fn mask(n: u32) -> u64 {
    if n == 64 {
        u64::MAX
    } else {
        (1 << n) - 1
    }
}

fn rev(x: u64, n: u32) -> u64 {
    x.reverse_bits() >> (64 - n)
}

fn bswap(x: u64, n: u32) -> u64 {
    x.swap_bytes() >> (64 - n)
}

fn ctlz(x: u64, n: u32) -> u64 {
    if x == 0 {
        u64::from(n)
    } else {
        u64::from(x.leading_zeros() - (64 - n))
    }
}

fn cttz(x: u64, n: u32) -> u64 {
    if x == 0 {
        u64::from(n)
    } else {
        u64::from(x.trailing_zeros())
    }
}

/// `llvm.fshl`: the high half of `a:b` shifted left by `s` modulo the width.
fn fshl(a: u64, b: u64, s: u64, n: u32) -> u64 {
    let k = (s % u64::from(n)) as u32;
    if k == 0 {
        a
    } else {
        ((a << k) | (b >> (n - k))) & mask(n)
    }
}

/// `llvm.fshr`: the low half of `a:b` shifted right by `s` modulo the width.
fn fshr(a: u64, b: u64, s: u64, n: u32) -> u64 {
    let k = (s % u64::from(n)) as u32;
    if k == 0 {
        b
    } else {
        ((b >> k) | (a << (n - k))) & mask(n)
    }
}

/// One workgroup of `threads` threads running `m` with `buffers` bound at slots 0, 1, ...: the
/// bytes of every buffer afterwards, identical in both scheduling orders.
fn run_kernel(
    what: &str,
    plan: &Plan,
    m: &TypedModule,
    threads: u32,
    buffers: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, threads), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let res = Resources {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, bytes)| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: i as u32,
                }),
                address: None,
                bytes: bytes.clone(),
            })
            .collect(),
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{what} {order:?}: {e:?}"));
        outs.push(out.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

// ---- The fixture ----

/// The fixture under each available assembler: `(spelling, bitcode)`.
fn spellings() -> Vec<(&'static str, Vec<u8>)> {
    let mut out = Vec::new();
    for (what, assembled) in [
        (
            "llvm-as",
            llvm_assemble(&engine_dir().join(format!("{STEM}.ll"))),
        ),
        (
            "metal-as",
            metal_assemble(&engine_dir().join(format!("{STEM}.air.ll"))),
        ),
    ] {
        match assembled {
            Ok(bc) => out.push((what, bc)),
            Err(AssembleError::Missing(why)) => eprintln!("SKIPPED {what}: {why}"),
            Err(AssembleError::Rejected(why)) => panic!("{what} rejected {STEM}: {why}"),
        }
    }
    out
}

/// The fixture's words: x[t], s[t] for each of its 16 threads.
const X: [u64; 16] = [
    0,
    u64::MAX,
    1 << 63,
    0x80,
    0x8000,
    0x8000_0000,
    1,
    0x0123_4567_89AB_CDEF,
    0xF0E1_D2C3_B4A5_9687,
    0x0000_0000_FFFF_FFFF,
    0xFFFF_FFFF_0000_0000,
    0x5555_5555_5555_5555,
    0x8000_0000_8000_8080,
    0x7F,
    0x7FFF,
    0x1_0000_0000,
];
const S: [u32; 16] = [
    0,
    1,
    7,
    8,
    15,
    16,
    31,
    32,
    33,
    63,
    64,
    65,
    3,
    100,
    255,
    u32::MAX,
];
const FILL: u32 = 0xABAB_ABAB;

/// What `bit_ops` writes to thread t's 48 words, from the MSL.
fn bit_ops_words(w: u64, n: u32) -> [u32; 48] {
    let (a, b) = (w & 0xFFFF_FFFF, w >> 32);
    let (h, c) = (w & 0xFFFF, w & 0xFF);
    let n64 = u64::from(n);
    let lo = |v: u64| v as u32;
    let mut r = [FILL; 48];
    let (rw, sw) = (rev(w, 64), bswap(w, 64));
    let (lw, rr) = (fshl(w, w, n64, 64), fshr(w, w, n64, 64));
    let words = [
        rev(c, 8),
        rev(h, 16),
        rev(a, 32),
        rw & 0xFFFF_FFFF,
        rw >> 32,
        bswap(h, 16),
        bswap(a, 32),
        sw & 0xFFFF_FFFF,
        sw >> 32,
        fshl(c, c, n64 & 0xFF, 8),
        fshl(h, h, n64 & 0xFFFF, 16),
        fshl(a, a, n64, 32),
        lw & 0xFFFF_FFFF,
        lw >> 32,
        fshr(c, c, n64 & 0xFF, 8),
        fshr(h, h, n64 & 0xFFFF, 16),
        fshr(a, a, n64, 32),
        rr & 0xFFFF_FFFF,
        rr >> 32,
        u64::from(a.count_ones()),
        u64::from(w.count_ones()),
        ctlz(a, 32),
        cttz(a, 32),
        ctlz(w, 64),
        cttz(w, 64),
        ctlz(h | 1, 16),
        cttz(h | 0x8000, 16),
        ctlz(a | 1, 32),
        cttz(a | 0x8000_0000, 32),
        ctlz(w | 1, 64),
        cttz(w | 1 << 63, 64),
    ];
    for (i, v) in words.iter().enumerate() {
        r[i] = lo(*v);
    }
    let v = [a, b, a ^ b, !a & 0xFFFF_FFFF];
    for l in 0..4 {
        r[31 + l] = lo(rev(v[l], 32));
        r[35 + l] = lo(bswap(v[l], 32));
        r[39 + l] = lo(fshl(v[l], v[l], v[(l + 1) % 4] & 31, 32));
    }
    r
}

/// `bit_ops` translates, links and validates on every committed profile, in both spellings,
/// with no precision note: every answer is exact, from 32-bit operations where the width is
/// another.
#[test]
fn the_bit_manipulation_fixture_translates_everywhere() {
    let profiles = committed_profiles().expect("profiles");
    for (what, bc) in spellings() {
        for (stem, p) in &profiles {
            let a =
                compile(&bc, "bit_ops", p).unwrap_or_else(|e| panic!("{what} on {stem}: {e:?}"));
            assert_eq!(a.precision, [], "{what} on {stem}");
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
        }
    }
}

/// `bit_ops` over its 16 edge words answers every word as LLVM defines each intrinsic, with and
/// without `shaderInt64`, in both spellings; the words it does not write keep their fill.
#[test]
fn the_bit_manipulation_fixture_answers_its_definition() {
    let x: Vec<u8> = X.iter().flat_map(|v| v.to_le_bytes()).collect();
    let s: Vec<u8> = S.iter().flat_map(|v| v.to_le_bytes()).collect();
    let want: Vec<u32> = X
        .iter()
        .zip(S)
        .flat_map(|(&w, n)| bit_ops_words(w, n))
        .collect();
    for (what, bc) in spellings() {
        for (stem, p) in profiles() {
            let (plan, m) =
                typed_module(&bc, "bit_ops", &p).unwrap_or_else(|e| panic!("{what} {stem}: {e:?}"));
            let fill = FILL.to_le_bytes().repeat(16 * 48);
            let out = run_kernel(
                &format!("{what} {stem}"),
                &plan,
                &m,
                16,
                &[x.clone(), s.clone(), fill],
            );
            let got: Vec<u32> = out[2]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{what} {stem}: thread {} word {} (x {:#x}, s {})",
                    i / 48,
                    i % 48,
                    X[i / 48],
                    S[i / 48]
                );
            }
        }
    }
}

// ---- The whole family, synthetic ----

/// An intrinsic of the family, with the operands its definition reads.
#[derive(Clone, Copy, Debug)]
enum Op {
    BitReverse,
    Bswap,
    Ctpop,
    /// `llvm.ctlz` with its zero-poison flag.
    Ctlz(bool),
    /// `llvm.cttz` with its zero-poison flag.
    Cttz(bool),
    Fshl,
    Fshr,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::BitReverse => "bitreverse",
            Op::Bswap => "bswap",
            Op::Ctpop => "ctpop",
            Op::Ctlz(_) => "ctlz",
            Op::Cttz(_) => "cttz",
            Op::Fshl => "fshl",
            Op::Fshr => "fshr",
        }
    }

    fn funnel(self) -> bool {
        matches!(self, Op::Fshl | Op::Fshr)
    }

    fn eval(self, a: u64, b: u64, s: u64, n: u32) -> u64 {
        match self {
            Op::BitReverse => rev(a, n),
            Op::Bswap => bswap(a, n),
            Op::Ctpop => u64::from(a.count_ones()),
            Op::Ctlz(_) => ctlz(a, n),
            Op::Cttz(_) => cttz(a, n),
            Op::Fshl => fshl(a, b, s, n),
            Op::Fshr => fshr(a, b, s, n),
        }
    }
}

/// The edge operands of width `n`: zero, all ones, the top bit alone and with the low one, one,
/// the alternating and nibble patterns, two irregular words, and the largest signed value. A
/// flag-true count reads no zero (its answer there is poison).
fn edges(n: u32, op: Op) -> Vec<u64> {
    let m = mask(n);
    let top = 1u64 << (n - 1);
    let all = [
        0,
        m,
        top,
        top | 1,
        1,
        0x5555_5555_5555_5555 & m,
        0xAAAA_AAAA_AAAA_AAAA & m,
        0x0F0F_0F0F_0F0F_0F0F & m,
        0x1234_5678_9ABC_DEF0 & m,
        0xFEDC_BA98_7654_3211 & m,
        top >> 1,
        m >> 1,
    ];
    let poison_at_zero = matches!(op, Op::Ctlz(true) | Op::Cttz(true));
    all.into_iter()
        .filter(|&v| !(poison_at_zero && v == 0))
        .collect()
}

/// The funnel shifts' amounts for width `n`: zero, one, past the width and every residue class
/// the edge cases sit in.
fn amounts(n: u32) -> Vec<u64> {
    let n = u64::from(n);
    [0, 1, n - 1, n, n + 1, 2 * n - 1, 3, n / 2, u64::MAX]
        .into_iter()
        .map(|v| v & mask(n as u32))
        .collect()
}

/// The kernel `name` applying `op` to `lanes`-lane vectors (a scalar for 1) of `i<n>`: its block
/// j gathers operands `j * lanes + l` (modulo `count`) of the input buffer (slot 1: the first
/// operands, then for a funnel shift the second operands and the amounts, `count` each), applies
/// the intrinsic and stores lane l at output index `j * lanes + l` (slot 0).
fn kernel(name: &str, op: Op, n: u32, lanes: u32, count: usize) -> String {
    let e = format!("i{n}");
    let ty = if lanes == 1 {
        e.clone()
    } else {
        format!("<{lanes} x {e}>")
    };
    let suffix = if lanes == 1 {
        e.clone()
    } else {
        format!("v{lanes}{e}")
    };
    let blocks = count.div_ceil(lanes as usize);
    let mut body = String::new();
    let mut v = 0usize;
    let mut fresh = || {
        v += 1;
        format!("%v{v}")
    };
    for j in 0..blocks {
        let mut operands = Vec::new();
        for part in 0..(if op.funnel() { 3 } else { 1 }) {
            let mut acc = "undef".to_string();
            let mut scalar = String::new();
            for l in 0..lanes as usize {
                let index = part * count + (j * lanes as usize + l) % count;
                let p = fresh();
                let x = fresh();
                body.push_str(&format!(
                    "  {p} = getelementptr {e}, ptr addrspace(1) %1, i64 {index}\n  {x} = load {e}, ptr addrspace(1) {p}, align 1\n"
                ));
                if lanes == 1 {
                    scalar = x;
                } else {
                    let next = fresh();
                    body.push_str(&format!(
                        "  {next} = insertelement {ty} {acc}, {e} {x}, i32 {l}\n"
                    ));
                    acc = next;
                }
            }
            operands.push(if lanes == 1 { scalar } else { acc });
        }
        let args: Vec<String> = operands.iter().map(|o| format!("{ty} {o}")).collect();
        let flag = match op {
            Op::Ctlz(f) | Op::Cttz(f) => format!(", i1 {f}"),
            _ => String::new(),
        };
        let r = fresh();
        body.push_str(&format!(
            "  {r} = call {ty} @llvm.{}.{suffix}({}{flag})\n",
            op.name(),
            args.join(", ")
        ));
        for l in 0..lanes as usize {
            let lane = if lanes == 1 {
                r.clone()
            } else {
                let x = fresh();
                body.push_str(&format!("  {x} = extractelement {ty} {r}, i32 {l}\n"));
                x
            };
            let p = fresh();
            body.push_str(&format!(
                "  {p} = getelementptr {e}, ptr addrspace(1) %0, i64 {}\n  store {e} {lane}, ptr addrspace(1) {p}, align 1\n",
                j * lanes as usize + l
            ));
        }
    }
    format!(
        "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1) {{\nentry:\n{body}  ret void\n}}\n\n"
    )
}

/// The declaration of `op` on `lanes`-lane vectors of `i<n>`.
fn declaration(op: Op, n: u32, lanes: u32) -> String {
    let e = format!("i{n}");
    let (ty, suffix) = if lanes == 1 {
        (e.clone(), e.clone())
    } else {
        (format!("<{lanes} x {e}>"), format!("v{lanes}{e}"))
    };
    let params = match op {
        Op::Fshl | Op::Fshr => format!("{ty}, {ty}, {ty}"),
        Op::Ctlz(_) | Op::Cttz(_) => format!("{ty}, i1"),
        _ => ty.clone(),
    };
    format!("declare {ty} @llvm.{}.{suffix}({params})\n", op.name())
}

/// A module of one kernel per `(n, lanes)` of `cells`, applying `op`.
fn module(op: Op, cells: &[(u32, u32, String, usize)]) -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for (n, lanes, name, count) in cells {
        s.push_str(&kernel(name, op, *n, *lanes, *count));
    }
    let mut decls: Vec<String> = cells
        .iter()
        .map(|(n, lanes, _, _)| declaration(op, *n, *lanes))
        .collect();
    decls.dedup();
    for d in decls {
        s.push_str(&d);
    }
    let refs: Vec<String> = (0..cells.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12, !13}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"out\"}\n!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"in\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, (_, _, name, _)) in cells.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{name}, !10, !11}}\n", 20 + i));
    }
    s
}

/// `op` at every width it is defined on and every lane count from 1 to 4 answers its definition
/// over the edge operands, with and without `shaderInt64` ([`profiles`]).
fn family(op: Op) {
    let widths: &[u32] = match op {
        Op::Bswap => &[16, 32, 64],
        _ => &[8, 16, 32, 64],
    };
    let mut cells = Vec::new();
    for &n in widths {
        for lanes in 1..=4u32 {
            let count = edges(n, op)
                .len()
                .max(if op.funnel() { amounts(n).len() } else { 0 });
            cells.push((n, lanes, format!("k_i{n}_x{lanes}"), count));
        }
    }
    let bc = match llvm_assemble_text(&module(op, &cells)) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the {op:?} module: {why}"),
    };
    for (stem, p) in profiles() {
        for (n, lanes, name, count) in &cells {
            let (n, count) = (*n, *count);
            let bytes = (n / 8) as usize;
            let cycle = |v: Vec<u64>| -> Vec<u64> { (0..count).map(|i| v[i % v.len()]).collect() };
            let a = cycle(edges(n, op));
            // The second operands are the first ones five places on.
            let b: Vec<u64> = (0..count).map(|i| a[(i + 5) % count]).collect();
            let s = cycle(amounts(n));
            let parts: &[&[u64]] = if op.funnel() { &[&a, &b, &s] } else { &[&a] };
            let mut input: Vec<u8> = parts
                .iter()
                .flat_map(|part| part.iter())
                .flat_map(|v| v.to_le_bytes()[..bytes].to_vec())
                .collect();
            // Both buffers are whole words long, as every buffer the runtime binds is.
            input.resize(input.len().next_multiple_of(4), 0);
            let outputs = count.div_ceil(*lanes as usize) * *lanes as usize;
            let (plan, m) = typed_module(&bc, name, &p)
                .unwrap_or_else(|e| panic!("{op:?} {name} on {stem}: {e:?}"));
            let out = run_kernel(
                &format!("{op:?} {name} on {stem}"),
                &plan,
                &m,
                1,
                &[vec![0xCD; (outputs * bytes).next_multiple_of(4)], input],
            );
            for i in 0..outputs {
                let k = i % count;
                let mut word = [0u8; 8];
                word[..bytes].copy_from_slice(&out[0][i * bytes..(i + 1) * bytes]);
                let got = u64::from_le_bytes(word);
                let want = op.eval(a[k], b[k], s[k], n);
                assert_eq!(
                    got, want,
                    "{op:?} {name} on {stem}: operand {k}: a {:#x} b {:#x} s {}",
                    a[k], b[k], s[k]
                );
            }
        }
    }
}

#[test]
fn bitreverse_answers_its_definition_at_every_width() {
    family(Op::BitReverse);
}

#[test]
fn bswap_answers_its_definition_at_every_width() {
    family(Op::Bswap);
}

#[test]
fn ctpop_answers_its_definition_at_every_width() {
    family(Op::Ctpop);
}

#[test]
fn ctlz_answers_its_definition_at_every_width_with_either_flag() {
    family(Op::Ctlz(false));
    family(Op::Ctlz(true));
}

#[test]
fn cttz_answers_its_definition_at_every_width_with_either_flag() {
    family(Op::Cttz(false));
    family(Op::Cttz(true));
}

#[test]
fn fshl_answers_its_definition_at_every_width() {
    family(Op::Fshl);
}

#[test]
fn fshr_answers_its_definition_at_every_width() {
    family(Op::Fshr);
}
