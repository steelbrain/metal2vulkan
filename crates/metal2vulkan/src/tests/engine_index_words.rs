//! Index words end to end (Profile F needs no `Int64` for index arithmetic).
//!
//! - `i64` arithmetic (`add`, `sub`, `mul`, `shl` by a constant below 32, `and`, `or`, `xor`)
//!   over index extensions and constants, whose only uses are index operands of GEPs on
//!   `u32`-offset pointers, is held as its low word: kernels built on it compile on `vk12-floor`
//!   (no `shaderInt64`) and `moltenvk-m3`, in both containment encodings, pass L3, select
//!   `arith.index_word`, and read and write exactly the element the 64-bit index names, including
//!   an index whose 64-bit computation wraps through 2³².
//! - An index word used in another block crosses it as a `u32`.
//! - `i64` arithmetic with any other use, a shift right, or a shift by 32 or more keeps its
//!   64-bit form: `moltenvk-m3` as a `u64`, `vk12-floor` as two words.
//! - An index of a wide pointer (a loaded pointer) keeps its 64-bit form on Profile A: its offset
//!   is 64-bit, so the low word would not do.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air;
use crate::air::ir::Value;
use crate::budget::{Budget, Meter, LEDGER_BYTES};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Capability, Error, Feature, Limit, Needs};
use crate::lower::access::index::index_values;
use crate::lower::select::Plan;
use crate::memory::static_root::analyze_memory;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, typed_module, AssembleError,
};
use crate::spirv::ids::{DemoteTy, ScalarTy};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Committed profile `stem`, optionally with buffer robustness disabled (the in-shader encoding).
fn profile_with(stem: &str, robust: bool) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if !robust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A profile without (`vk12-floor`, Profile F) and one with `shaderInt64` (`moltenvk-m3`,
/// Profile A), each in both containment encodings.
fn encodings() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in ["vk12-floor", "moltenvk-m3"] {
        for robust in [true, false] {
            out.push((
                format!("{stem} robust={robust}"),
                profile_with(stem, robust),
            ));
        }
    }
    out
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-index-words-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

/// A kernel `name` over `device` buffers `data` (0, read-write), `out` (1, read-write), and
/// `params` (2, read, element `params_ty` of `params_size` bytes). The body starts in block
/// `entry` and must end every block with a terminator.
fn kernel(name: &str, body: &str, params: (&str, u32)) -> String {
    let (params_ty, params_size) = params;
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2) {{\nentry:\n{body}}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"data\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"uchar\", !\"air.arg_name\", !\"out\"}}\n\
         !14 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {params_size}, !\"air.arg_type_align_size\", i32 {params_size}, !\"air.arg_type_name\", !\"{params_ty}\", !\"air.arg_name\", !\"params\"}}\n\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A buffer bound at its start over `data` padded with zeros to a multiple of 16.
fn bound(index: u32, data: &[u8]) -> Buffer {
    let mut bytes = data.to_vec();
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    Buffer {
        slot: Some(kslot(index)),
        address: None,
        bytes,
    }
}

/// Push bytes of `plan` with zero residuals (every field is a bind residual in the push window).
fn push_bytes(plan: &Plan) -> Vec<u8> {
    for f in &plan.push.fields {
        assert!(
            matches!(f.loc, PushLoc::Push { .. }),
            "{:?} in the argument table",
            f.field
        );
        assert!(
            matches!(f.field, PushField::SlotBindOffset(_)),
            "{:?}",
            f.field
        );
    }
    vec![0u8; 128]
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(11) ^ ((i >> 8) as u8))
        .collect()
}

/// The shared tail of an index-word kernel: `%t` indexes the `i32` elements of `data` from byte
/// 64; the element is copied to `out[0]` and overwritten in `data` with `-1`.
const TAIL: &str = "  %mid = getelementptr i8, ptr addrspace(1) %0, i64 64\n  %p = getelementptr inbounds i32, ptr addrspace(1) %mid, i64 %t\n  %v = load i32, ptr addrspace(1) %p, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n  store i32 -1, ptr addrspace(1) %p, align 4\n  ret void\n";

/// An index-word kernel: the IR computing `%t` from the loaded `i32` `%o`, and the 64-bit index
/// `%t` of each tested `%o`.
struct WordKernel {
    name: &'static str,
    prep: &'static str,
    index: fn(i32) -> i64,
    cases: &'static [i32],
}

fn word_kernels() -> Vec<WordKernel> {
    vec![
        WordKernel {
            name: "add_to_sext",
            prep: "  %s = sext i32 %o to i64\n  %t = add nsw i64 %s, 1\n",
            index: |o| i64::from(o) + 1,
            cases: &[-16, -9, -1, 0, 1, 7, 14],
        },
        WordKernel {
            name: "sub_from_constant",
            prep: "  %s = zext i32 %o to i64\n  %t = sub i64 20, %s\n",
            index: |o| 20 - i64::from(o as u32),
            cases: &[5, 12, 20, 21, 35],
        },
        WordKernel {
            name: "shl_or_mul_sub",
            prep: "  %s = zext i32 %o to i64\n  %a = shl i64 %s, 1\n  %b = or disjoint i64 %a, 1\n  %c = mul i64 %b, 3\n  %t = add i64 %c, -16\n",
            index: |o| ((i64::from(o as u32) << 1) | 1) * 3 - 16,
            cases: &[0, 1, 2, 3, 4],
        },
        WordKernel {
            name: "xor_and_of_sext",
            prep: "  %s = sext i32 %o to i64\n  %a = xor i64 %s, 7\n  %t = and i64 %a, 15\n",
            index: |o| (i64::from(o) ^ 7) & 15,
            cases: &[i32::MIN, -100, -1, 0, 3, 99, i32::MAX],
        },
        // The 64-bit index is `%o - 2³² + 6`: for `%o` near 2³² it is small, and only its low
        // word is the low word of the sum.
        WordKernel {
            name: "wraps_through_two_to_the_32",
            prep: "  %s = zext i32 %o to i64\n  %t = add i64 %s, -4294967290\n",
            index: |o| i64::from(o as u32) - 4_294_967_290,
            cases: &[-16, -10, -7, -1],
        },
    ]
}

/// Run `bc`'s entry `name` on `p` with `params` = `o`, and check the element at `index` was read
/// and overwritten exactly.
fn run_word(bc: &[u8], name: &str, p: &DeviceProfile, what: &str, o: i32, index: i64) {
    let (plan, m) = typed_module(bc, name, p).expect("module");
    let data = pattern(128);
    let res = Resources {
        buffers: vec![
            bound(0, &data),
            bound(1, &[0; 16]),
            bound(2, &o.to_le_bytes()),
        ],
        push: push_bytes(&plan),
        spec: Default::default(),
    };
    let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what} {o}: {e:?}"));
    let at = usize::try_from(64 + 4 * index).expect("in range");
    assert_eq!(out.buffers[1][..4], data[at..at + 4], "{what} o={o}");
    let mut want = data.clone();
    want[at..at + 4].copy_from_slice(&[0xFF; 4]);
    assert_eq!(out.buffers[0][..128], want[..], "{what} o={o}");
}

/// Every index-word kernel compiles without `shaderInt64`, selects `arith.index_word`, passes L3,
/// and reads and writes exactly the element its 64-bit index names.
#[test]
fn index_words_are_exact_without_int64() {
    for k in word_kernels() {
        let body = format!(
            "  %o = load i32, ptr addrspace(1) %2, align 4\n{}{TAIL}",
            k.prep
        );
        let Some(bc) = assemble_text(k.name, &kernel(k.name, &body, ("int", 4))) else {
            return;
        };
        for (what, p) in encodings() {
            let what = format!("{what} {}", k.name);
            let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let names: Vec<&String> = a.strategies.entries.keys().collect();
            assert!(
                names.iter().any(|n| *n == "arith.index_word"),
                "{what}: {names:?}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for &o in k.cases {
                let index = (k.index)(o);
                assert!((-16..16).contains(&index), "{what}: case {o} -> {index}");
                run_word(&bc, k.name, &p, &what, o, index);
            }
        }
    }
}

/// An index word defined in the entry block and used by GEPs in two other blocks crosses them as a
/// `u32`; no 64-bit value is demoted.
#[test]
fn an_index_word_used_in_other_blocks_crosses_as_a_word() {
    let name = "word_across_blocks";
    let body = format!(
        "  %o = load i32, ptr addrspace(1) %2, align 4\n  %s = sext i32 %o to i64\n  %t = add nsw i64 %s, 3\n  %c = icmp slt i32 %o, 0\n  br i1 %c, label %neg, label %pos\nneg:\n{TAIL}pos:\n  %omid = getelementptr i8, ptr addrspace(1) %0, i64 64\n  %q = getelementptr inbounds i32, ptr addrspace(1) %omid, i64 %t\n  %w = load i32, ptr addrspace(1) %q, align 4\n  store i32 %w, ptr addrspace(1) %1, align 4\n  store i32 -1, ptr addrspace(1) %q, align 4\n  ret void\n"
    );
    let Some(bc) = assemble_text(name, &kernel(name, &body, ("int", 4))) else {
        return;
    };
    for (what, p) in encodings() {
        let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            a.strategies.entries.contains_key("arith.index_word"),
            "{what}"
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, _) = typed_module(&bc, name, &p).expect("module");
        assert!(
            plan.cross
                .iter()
                .all(|(_, t)| !matches!(t, DemoteTy::Scalar(ScalarTy::Int { bits: 64 }))),
            "{what}: {:?}",
            plan.cross
        );
        assert!(
            plan.cross
                .iter()
                .any(|(_, t)| *t == DemoteTy::Scalar(ScalarTy::Int { bits: 32 })),
            "{what}: {:?}",
            plan.cross
        );
        for o in [-16, -5, -3, 0, 4, 12] {
            run_word(&bc, name, &p, &what, o, i64::from(o) + 3);
        }
    }
}

/// `i64` arithmetic that is not an index word keeps its 64-bit form: `moltenvk-m3` compiles it as a
/// native `u64`, and `vk12-floor` (no `shaderInt64`) as two words; neither uses `arith.index_word`,
/// and both pass L3.
#[test]
fn index_arithmetic_with_other_uses_or_operators_keeps_64_bits() {
    let cases = [
        (
            "word_also_stored",
            "  %s = sext i32 %o to i64\n  %t = add i64 %s, 1\n  %sp = getelementptr i8, ptr addrspace(1) %1, i64 8\n  store i64 %t, ptr addrspace(1) %sp, align 8\n",
        ),
        (
            "word_also_selected",
            "  %s = sext i32 %o to i64\n  %a = add i64 %s, 1\n  %c = icmp slt i32 %o, 4\n  %t = select i1 %c, i64 %a, i64 0\n",
        ),
        (
            "shift_right",
            "  %s = sext i32 %o to i64\n  %t = ashr i64 %s, 1\n",
        ),
        (
            "shift_left_by_40",
            "  %s = zext i32 %o to i64\n  %a = shl i64 %s, 40\n  %t = add i64 %a, 2\n",
        ),
        (
            "a_loaded_i64_operand",
            "  %wp = getelementptr i8, ptr addrspace(1) %2, i64 8\n  %w = load i64, ptr addrspace(1) %wp, align 8\n  %s = sext i32 %o to i64\n  %t = add i64 %s, %w\n",
        ),
    ];
    for (name, prep) in cases {
        let body = format!("  %o = load i32, ptr addrspace(1) %2, align 4\n{prep}{TAIL}");
        let Some(bc) = assemble_text(name, &kernel(name, &body, ("long", 8))) else {
            return;
        };
        for stem in ["vk12-floor", "moltenvk-m3"] {
            let p = profile_with(stem, true);
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{name} {stem}: {e}"));
            assert!(
                !a.strategies.entries.contains_key("arith.index_word"),
                "{name} {stem}"
            );
            let words = a
                .strategies
                .entries
                .keys()
                .any(|n| n.starts_with("arith.wide"));
            assert_eq!(words, !p.fields().leaf.int64, "{name} {stem}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} {stem}: L3: {e}"));
        }
    }
}

/// The index of a wide pointer (a pointer loaded from memory) is not an index word: its offset is
/// 64-bit. Profile A computes it in 64 bits; Profile F refuses the dereference.
#[test]
fn a_wide_pointer_index_keeps_its_64_bit_form() {
    let name = "wide_pointer_index";
    let body = "  %o = load i32, ptr addrspace(1) %2, align 4\n  %pp = getelementptr i8, ptr addrspace(1) %2, i64 8\n  %q = load ptr addrspace(1), ptr addrspace(1) %pp, align 8\n  %s = zext i32 %o to i64\n  %t = add i64 %s, 1\n  %p = getelementptr inbounds i32, ptr addrspace(1) %q, i64 %t\n  %v = load i32, ptr addrspace(1) %p, align 4\n  store i32 %v, ptr addrspace(1) %1, align 4\n  ret void\n";
    let Some(bc) = assemble_text(name, &kernel(name, body, ("long", 8))) else {
        return;
    };
    let p = profile_with("moltenvk-m3", true);
    let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{e}"));
    assert!(a.interface.needs_pointer_translation);
    assert!(!a.strategies.entries.contains_key("arith.index_word"));
    assert!(a.strategies.entries.contains_key("arith.int"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("L3: {e}"));
    match compile(&bc, name, &profile_with("vk12-floor", true)) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::DereferenceLoadedPointer);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![
                    Capability::Feature(VkFeature::BufferDeviceAddress),
                    Capability::Feature(VkFeature::ShaderInt64),
                ]])
            );
        }
        other => panic!("expected DereferenceLoadedPointer, got {other:?}"),
    }
}

/// The index-value scan is metered and linear. A kernel of `n` chains, each an index extension, an
/// index word, and a GEP load: the scan finds all `n` of each, ticks each of its two passes once
/// per instruction and its worklist once per item (at most `2·insts + candidates + 2·operands`
/// units), charges its tables and releases them, and a work cap one unit below what it ticks stops
/// it with `LimitExceeded(WorkUnits)`.
#[test]
fn the_index_value_scan_is_metered_and_linear() {
    let p = profile_with("vk12-floor", true);
    for n in [16usize, 64, 256] {
        let name = format!("index_chains_{n}");
        let mut body = String::from("  %o = load i32, ptr addrspace(1) %2, align 4\n");
        for i in 0..n {
            body += &format!(
                "  %s{i} = sext i32 %o to i64\n  %t{i} = add nsw i64 %s{i}, {i}\n  %p{i} = getelementptr inbounds i8, ptr addrspace(1) %0, i64 %t{i}\n  %v{i} = load i8, ptr addrspace(1) %p{i}, align 1\n  %q{i} = getelementptr inbounds i8, ptr addrspace(1) %1, i64 {i}\n  store i8 %v{i}, ptr addrspace(1) %q{i}, align 1\n"
            );
        }
        body += "  ret void\n";
        let Some(bc) = assemble_text(&name, &kernel(&name, &body, ("int", 4))) else {
            return;
        };
        let m = &mut meter();
        let decoded = air::decode(&bc, m).expect("decode");
        let req = air::SpecRequest {
            call_stack_depth: 1,
            entry: name.clone(),
            ..air::SpecRequest::default()
        };
        let spec = air::specialize(decoded, &req, m).expect("specialize");
        let memory = analyze_memory(&spec, &p, m).expect("analyze");
        let func_body = spec
            .module
            .func(spec.func)
            .and_then(|f| f.body.as_ref())
            .expect("body");
        let insts = func_body.insts.len() as u64;
        let mut operands = 0u64;
        for inst in func_body.insts.values() {
            inst.kind
                .for_each_value(|v| operands += u64::from(matches!(v, Value::Inst(_))));
        }

        let mut scan = meter();
        let values =
            index_values(&spec.module, spec.func, func_body, &memory, &mut scan).expect("scan");
        assert_eq!(values.extensions.len(), n, "{name}");
        assert_eq!(values.words.len(), n, "{name}");
        let ticked = scan.work();
        let candidates = 2 * n as u64;
        assert!(ticked >= 2 * insts, "{name}: {ticked} < 2·{insts}");
        assert!(
            ticked <= 2 * insts + candidates + 2 * operands,
            "{name}: {ticked} work units for {insts} instructions and {operands} operands"
        );
        assert_eq!(scan.ledger(), 0, "{name}: the scan's tables are released");
        assert!(
            scan.ledger_peak() > 0,
            "{name}: the scan's tables are charged"
        );

        let mut capped = Meter::with_caps(&Budget::default(), ticked - 1, LEDGER_BYTES);
        assert_eq!(
            index_values(&spec.module, spec.func, func_body, &memory, &mut capped),
            Err(Error::LimitExceeded(Limit::WorkUnits)),
            "{name}"
        );
    }
}
