//! Float arithmetic is emitted in the AIR's own order, each operation decorated by its own
//! fast-math flags: an operation carrying both `contract` and `reassoc` is left undecorated, so the
//! device's compiler groups and fuses it as those flags allow, and any other is `NoContraction`,
//! which keeps it apart and in place (`lower::arith::contraction`).
//!
//! One module holds the forms, each a kernel of one thread that reads its operands from a device
//! buffer and stores its result:
//!
//! - `fast_sum`, `((d + c) + b) + a` under `fast`, its leaves loaded in the order `a, b, c, d`;
//! - `precise_sum`, the same sum with no flag;
//! - `fast_mul_add`, `a * a + c` under `fast`;
//! - `contract_mul_add`, the same pair under `contract` alone;
//! - `fast_product`, `(splat(u) * v) * splat(3)` under `fast`, on two lanes.
//!
//! The operands make each grouping its own answer: in the leaves' load order the sum is `2^-23`
//! and as written `0`; fused, the pair is `2^-24` and unfused `0`; as written the product's first
//! lane is `0x40400003` and paired as `v * (u * 3)` `0x40400004`. The interpreter rounds each
//! instruction on its own and fuses nothing, so its answer names the order the module states.
//!
//! A second module holds `half` chains, `(a + b) + c` under `fast` and with no flag, and a `half`
//! scaled by the constant 2. With `a = 1` and `b = c = 2^-11` (half the `half` unit at 1, a tie)
//! each rounded sum is `1` (`0x3C00`); the chain held at `float` and rounded once is `1 + 2^-10`
//! (`0x3C01`). Where a profile takes the device's 16-bit arithmetic, each operation reads its
//! operands as 16-bit floats with no conversion between them: a 16-bit float widened and narrowed
//! back is that float (`FunctionBuilder::convert`), and a narrowed constant is a 16-bit constant.
//! Without `shaderFloat16` the `fast` chain is held at `float` precision under
//! `HalfChainPolicy::AtExits` (`lower::arith::half_chain`), and rounded at each operation under
//! `HalfChainPolicy::EachOperation`.
//!
//! `llvm-as` assembles the modules; without it the tests skip visibly.

use std::collections::BTreeSet;

use crate::device::HalfChainPolicy;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, exact_maths, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
    Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::committed_profiles;

const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @fast_sum(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load float, ptr addrspace(1) %in, align 4
  %pb = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %b = load float, ptr addrspace(1) %pb, align 4
  %pc = getelementptr inbounds float, ptr addrspace(1) %in, i32 2
  %c = load float, ptr addrspace(1) %pc, align 4
  %pd = getelementptr inbounds float, ptr addrspace(1) %in, i32 3
  %d = load float, ptr addrspace(1) %pd, align 4
  %s1 = fadd fast float %d, %c
  %s2 = fadd fast float %s1, %b
  %s3 = fadd fast float %s2, %a
  store float %s3, ptr addrspace(1) %out, align 4
  ret void
}

define void @precise_sum(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load float, ptr addrspace(1) %in, align 4
  %pb = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %b = load float, ptr addrspace(1) %pb, align 4
  %pc = getelementptr inbounds float, ptr addrspace(1) %in, i32 2
  %c = load float, ptr addrspace(1) %pc, align 4
  %pd = getelementptr inbounds float, ptr addrspace(1) %in, i32 3
  %d = load float, ptr addrspace(1) %pd, align 4
  %s1 = fadd float %d, %c
  %s2 = fadd float %s1, %b
  %s3 = fadd float %s2, %a
  store float %s3, ptr addrspace(1) %out, align 4
  ret void
}

define void @fast_mul_add(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load float, ptr addrspace(1) %in, align 4
  %pc = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %c = load float, ptr addrspace(1) %pc, align 4
  %p = fmul fast float %a, %a
  %r = fadd fast float %p, %c
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

define void @contract_mul_add(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load float, ptr addrspace(1) %in, align 4
  %pc = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %c = load float, ptr addrspace(1) %pc, align 4
  %p = fmul contract float %a, %a
  %r = fadd contract float %p, %c
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

define void @fast_product(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %u = load float, ptr addrspace(1) %in, align 4
  %pv = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %v = load <2 x float>, ptr addrspace(1) %pv, align 4
  %us = insertelement <2 x float> poison, float %u, i64 0
  %uu = shufflevector <2 x float> %us, <2 x float> poison, <2 x i32> zeroinitializer
  %p = fmul fast <2 x float> %uu, %v
  %r = fmul fast <2 x float> %p, <float 3.000000e+00, float 3.000000e+00>
  store <2 x float> %r, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!20, !21, !22, !23, !24}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!9 = !{!4, !5}
!20 = !{ptr @fast_sum, !3, !9}
!21 = !{ptr @precise_sum, !3, !9}
!22 = !{ptr @fast_mul_add, !3, !9}
!23 = !{ptr @contract_mul_add, !3, !9}
!24 = !{ptr @fast_product, !3, !9}
"#;

/// The forms: entry, operands (bits), the first word the AIR's order stores, the word the other
/// grouping or the fusion stores (which must differ), and whether every float operation carries
/// both `contract` and `reassoc` (undecorated) or none of them does (`NoContraction`).
const FORMS: [(&str, [u32; 4], u32, u32, bool); 5] = [
    // ((1 + 2^-24) + 2^-24) - 1 is 0; ((-1 + 2^-24) + 2^-24) + 1, the load order, is 2^-23.
    (
        "fast_sum",
        [0xBF80_0000, 0x3380_0000, 0x3380_0000, 0x3F80_0000],
        0,
        0x3400_0000,
        true,
    ),
    (
        "precise_sum",
        [0xBF80_0000, 0x3380_0000, 0x3380_0000, 0x3F80_0000],
        0,
        0x3400_0000,
        false,
    ),
    // (1 + 2^-12)^2 - (1 + 2^-11): 0 rounded twice, the product's rounding error 2^-24 fused.
    (
        "fast_mul_add",
        [0x3F80_0800, 0xBF80_1000, 0, 0],
        0,
        0x3380_0000,
        true,
    ),
    (
        "contract_mul_add",
        [0x3F80_0800, 0xBF80_1000, 0, 0],
        0,
        0x3380_0000,
        false,
    ),
    // u = v.x = 1 + 2^-23: (u * v.x) * 3 is 0x40400003, v.x * (u * 3) 0x40400004.
    (
        "fast_product",
        [0x3F80_0001, 0x3F80_0001, 0x3F80_0002, 0],
        0x4040_0003,
        0x4040_0004,
        true,
    ),
];

const HALF_MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @fast_half_chain(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load half, ptr addrspace(1) %in, align 2
  %pb = getelementptr inbounds half, ptr addrspace(1) %in, i32 1
  %b = load half, ptr addrspace(1) %pb, align 2
  %pc = getelementptr inbounds half, ptr addrspace(1) %in, i32 2
  %c = load half, ptr addrspace(1) %pc, align 2
  %s1 = fadd fast half %a, %b
  %s2 = fadd fast half %s1, %c
  store half %s2, ptr addrspace(1) %out, align 2
  ret void
}

define void @strict_half_chain(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load half, ptr addrspace(1) %in, align 2
  %pb = getelementptr inbounds half, ptr addrspace(1) %in, i32 1
  %b = load half, ptr addrspace(1) %pb, align 2
  %pc = getelementptr inbounds half, ptr addrspace(1) %in, i32 2
  %c = load half, ptr addrspace(1) %pc, align 2
  %s1 = fadd half %a, %b
  %s2 = fadd half %s1, %c
  store half %s2, ptr addrspace(1) %out, align 2
  ret void
}

define void @half_scale(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load half, ptr addrspace(1) %in, align 2
  %r = fmul half %a, 0xH4000
  store half %r, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!20, !21, !22}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!9 = !{!4, !5}
!20 = !{ptr @fast_half_chain, !3, !9}
!21 = !{ptr @strict_half_chain, !3, !9}
!22 = !{ptr @half_scale, !3, !9}
"#;

/// The `half` chains' operands, two halves to a word: `a = 1`, `b = c = 2^-11`.
const HALF_INPUT: [u32; 4] = [0x1000_3C00, 0x0000_1000, 0, 0];

/// The rounded chain's `half`.
const ROUNDED: u32 = 0x3C00;

/// The chain's `half` held at `float` and rounded once.
const HELD: u32 = 0x3C01;

fn assemble() -> Option<Vec<u8>> {
    assemble_text(MODULE)
}

fn assemble_text(text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-fast-math-order");
    let ll = scratch.path().join("k.ll");
    std::fs::write(&ll, text).expect("write");
    match llvm_assemble(&ll) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// The float arithmetic of `words`: the results of every `OpFAdd`, `OpFSub` and `OpFMul`, of
/// every `OpFmaKHR` and every `GLSL.std.450 Fma` whose addend is not a `-0.0` constant (a fused
/// multiply-add; with a `-0.0` addend it is a product rounded on its own), and the results
/// decorated `NoContraction`.
struct Arith {
    sums_and_products: Vec<u32>,
    fused: Vec<u32>,
    no_contraction: BTreeSet<u32>,
}

fn arith(words: &[u32]) -> Arith {
    const GLSL_FMA: u32 = 50;
    let mut out = Arith {
        sums_and_products: Vec::new(),
        fused: Vec::new(),
        no_contraction: BTreeSet::new(),
    };
    let mut negative_zeros = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let op = |o: crate::spv::Op| opcode == o as u32;
        if op(crate::spv::Op::Decorate)
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            out.no_contraction.insert(words[i + 1]);
        }
        if op(crate::spv::Op::Constant) && len == 4 && words[i + 3] == 0x8000_0000 {
            negative_zeros.push(words[i + 2]);
        }
        if op(crate::spv::Op::FAdd) || op(crate::spv::Op::FSub) || op(crate::spv::Op::FMul) {
            out.sums_and_products.push(words[i + 2]);
        }
        if op(crate::spv::Op::FmaKHR) {
            out.fused.push(words[i + 2]);
        }
        if op(crate::spv::Op::ExtInst) && words[i + 4] == GLSL_FMA {
            if negative_zeros.contains(&words[i + 7]) {
                out.sums_and_products.push(words[i + 2]);
            } else {
                out.fused.push(words[i + 2]);
            }
        }
        i += len;
    }
    out
}

/// The words `entry` stores on the interpreter, for `p`, from operands `input`.
fn interpret(
    bc: &[u8],
    entry: &str,
    p: &crate::device::profile::DeviceProfile,
    input: [u32; 4],
) -> [u32; 2] {
    let (plan, module) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let slot = |index: u32| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(slot(index)),
        address: None,
        bytes,
    };
    let mut spec = std::collections::BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let resources = Resources {
        buffers: vec![
            buffer(0, input.iter().flat_map(|w| w.to_le_bytes()).collect()),
            buffer(1, vec![0xEE; 8]),
        ],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let cfg = Config {
        workgroups: [1, 1, 1],
        order: Order::Forward,
        ..Config::default()
    };
    let o = run(&module, &cfg, resources).unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let w = |k: usize| u32::from_le_bytes(o.buffers[1][4 * k..4 * k + 4].try_into().expect("word"));
    [w(0), w(1)]
}

/// On every committed profile each form's float arithmetic is the AIR's own: one instruction per
/// AIR operation and no fused multiply-add, each `fast` one undecorated and each one without
/// `contract` and `reassoc` `NoContraction`; every module validates; and the interpreter stores the
/// AIR's grouping, not the leaves' load order or the pair fused.
#[test]
fn float_arithmetic_keeps_the_airs_order_and_flags() {
    let Some(bc) = assemble() else {
        return;
    };
    // The operands tell the groupings apart.
    for (entry, _, ours, other, _) in FORMS {
        assert_ne!(ours, other, "{entry}: the operands name no grouping");
    }
    let ops = |entry: &str| match entry {
        "fast_sum" | "precise_sum" => 3,
        _ => 2,
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, input, ours, _, fast) in FORMS {
            let what = format!("{stem} {entry}");
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let found = arith(&words);
            assert!(found.fused.is_empty(), "{what}: fused {:?}", found.fused);
            assert_eq!(
                found.sums_and_products.len(),
                ops(entry),
                "{what}: one instruction per AIR operation"
            );
            for id in &found.sums_and_products {
                assert_eq!(
                    found.no_contraction.contains(id),
                    !fast,
                    "{what}: %{id} NoContraction"
                );
            }
            assert_eq!(interpret(&bc, entry, &p, input)[0], ours, "{what}: stored");
        }
    }
}

/// The note follows the flags: a `fast` sum chained into a `fast` sum, a `fast` product into a
/// `fast` product, and a `contract` product into a `contract` addition (which Metal's compiler may
/// fuse where the device does not) each note `FastMathRewrites`; the sum without a flag does not.
#[test]
fn chained_fast_math_notes_fast_math_rewrites() {
    let Some(bc) = assemble() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, _, _, _, _) in FORMS {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::FastMathRewrites),
                entry != "precise_sum",
                "{stem} {entry}: {:?}",
                a.precision
            );
        }
    }
}

/// The `OpFConvert`s of `words` that narrow to 16 bits a value the module widened from 16 bits, or
/// a constant; and the count of 16-bit `OpFAdd`, `OpFSub` and `OpFMul`. A widening of a narrowing
/// is a rounding's second half, which the builder keeps whole (a `half` store's narrowing reads
/// it), so it is no widened 16-bit value here.
fn narrowings_of_halves(words: &[u32]) -> (Vec<u32>, usize) {
    use crate::spv::Op;
    let mut width = std::collections::BTreeMap::new();
    let mut vector = std::collections::BTreeMap::new();
    let mut ty_of = std::collections::BTreeMap::new();
    let mut constants = BTreeSet::new();
    let mut converts = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let op = |o: Op| opcode == o as u32;
        if op(Op::TypeFloat) {
            width.insert(words[i + 1], words[i + 2]);
        } else if op(Op::TypeVector) {
            vector.insert(words[i + 1], words[i + 2]);
        } else if op(Op::Constant) || op(Op::ConstantComposite) || op(Op::ConstantNull) {
            constants.insert(words[i + 2]);
        } else if op(Op::FConvert) {
            converts.push((words[i + 1], words[i + 2], words[i + 3]));
        }
        if len > 2
            && (op(Op::FAdd) || op(Op::FSub) || op(Op::FMul) || op(Op::FConvert) || op(Op::ExtInst))
        {
            ty_of.insert(words[i + 2], words[i + 1]);
        }
        i += len;
    }
    let bits = |ty: u32| width.get(vector.get(&ty).unwrap_or(&ty)).copied();
    let narrowed: BTreeSet<u32> = converts
        .iter()
        .filter(|(ty, _, _)| bits(*ty) == Some(16))
        .map(|(_, id, _)| *id)
        .collect();
    let widened: BTreeSet<u32> = converts
        .iter()
        .filter(|(ty, _, x)| {
            bits(*ty) == Some(32)
                && ty_of.get(x).and_then(|t| bits(*t)) == Some(16)
                && !narrowed.contains(x)
        })
        .map(|(_, id, _)| *id)
        .collect();
    let bad = converts
        .iter()
        .filter(|(ty, _, x)| {
            bits(*ty) == Some(16) && (widened.contains(x) || constants.contains(x))
        })
        .map(|(_, id, _)| *id)
        .collect();
    let mut sixteen = 0;
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        let arith = [
            crate::spv::Op::FAdd,
            crate::spv::Op::FSub,
            crate::spv::Op::FMul,
        ]
        .iter()
        .any(|o| opcode == *o as u32);
        if arith && bits(words[i + 1]) == Some(16) {
            sixteen += 1;
        }
        i += len;
    }
    (bad, sixteen)
}

/// Where a profile takes the device's 16-bit arithmetic, no conversion to 16 bits reads a value
/// the module widened from 16 bits or a constant: the operations read their 16-bit operands
/// directly. On every committed profile each module validates and computes the rounded chain, but
/// for the `fast` chain held at `float` without `shaderFloat16`; and some profile takes the 16-bit
/// arithmetic, so the check is not vacuous.
#[test]
fn half_operations_read_their_16_bit_operands_directly() {
    let Some(bc) = assemble_text(HALF_MODULE) else {
        return;
    };
    let mut sixteen_bit = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let held = crate::lower::arith::narrow::quantize_form(&p)
            == crate::spirv::builder::QuantizeForm::Device;
        for entry in ["fast_half_chain", "strict_half_chain", "half_scale"] {
            let what = format!("{stem} {entry}");
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (bad, sixteen) = narrowings_of_halves(&words);
            assert!(
                bad.is_empty(),
                "{what}: narrowings of 16-bit values {bad:?}"
            );
            sixteen_bit += usize::from(sixteen > 0);
            let stored = interpret(&bc, entry, &p, HALF_INPUT)[0] & 0xFFFF;
            let want = match entry {
                "half_scale" => 0x4000,
                "fast_half_chain" if held => HELD,
                _ => ROUNDED,
            };
            assert_eq!(stored, want, "{what}: stored");
        }
    }
    assert!(
        sixteen_bit > 0,
        "no committed profile takes 16-bit arithmetic"
    );
}

/// Without `shaderFloat16` the `fast` chain's first sum is held at `float` under
/// `HalfChainPolicy::AtExits`, the default: one quantize, at the chain's exit, and the chain's
/// sum rounded once; under `HalfChainPolicy::EachOperation` (`exact_answers`) each sum rounds, two
/// quantizes and the rounded chain. The chain without flags rounds each sum under both, and a
/// profile with `shaderFloat16` takes no quantize under either. Each `fast` chain notes
/// `FastMathRewrites`, and the two values make two profile fingerprints.
#[test]
fn half_chains_without_float16_round_at_their_exits_by_default() {
    let Some(bc) = assemble_text(HALF_MODULE) else {
        return;
    };
    assert_eq!(
        crate::device::DevicePolicy::UNTRUSTED.half_chains,
        HalfChainPolicy::AtExits
    );
    assert_eq!(exact_maths().half_chains, HalfChainPolicy::EachOperation);
    let quantizes = |words: &[u32]| {
        let mut n = 0;
        let mut i = 5;
        while i < words.len() {
            let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
            n += usize::from(opcode == crate::spv::Op::QuantizeToF16 as u32);
            i += len;
        }
        n
    };
    let mut emulated = 0;
    for (stem, fast_p) in committed_profiles().expect("profiles") {
        let exact_p = profile(&stem, exact_maths()).expect("profile");
        // The switch changes emitted code, so the profile's fingerprint (every cache key) covers it.
        assert_ne!(
            fast_p.fingerprint(),
            exact_p.fingerprint(),
            "{stem}: the half chains' policy keys no profile"
        );
        let device = crate::lower::arith::narrow::quantize_form(&fast_p)
            == crate::spirv::builder::QuantizeForm::Device;
        emulated += usize::from(device);
        for (entry, fast) in [("fast_half_chain", true), ("strict_half_chain", false)] {
            for (policy, p) in [("AtExits", &fast_p), ("EachOperation", &exact_p)] {
                let what = format!("{stem} {entry} {policy}");
                let a = compile(&bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let (_, words) =
                    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let held = device && fast && policy == "AtExits";
                let want = match (device, held) {
                    (false, _) => 0,
                    (true, true) => 1,
                    (true, false) => 2,
                };
                assert_eq!(quantizes(&words), want, "{what}: OpQuantizeToF16");
                let stored = interpret(&bc, entry, p, HALF_INPUT)[0] & 0xFFFF;
                assert_eq!(stored, if held { HELD } else { ROUNDED }, "{what}: stored");
                assert_eq!(
                    a.precision.contains(&PrecisionNote::FastMathRewrites),
                    fast,
                    "{what}: {:?}",
                    a.precision
                );
            }
        }
    }
    assert!(emulated > 0, "no committed profile lacks shaderFloat16");
}
