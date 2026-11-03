//! Inline assembly: the owned fixture `engine/kernel_inline_asm` and synthetic modules.
//!
//! An MSL `asm volatile("")`, the one inline assembly MSL accepts, compiles to a call of
//! `asm sideeffect "", ""` of type `void ()`, whose callee is a function-local INLINEASM constant
//! (CONSTANTS_BLOCK record 30). It assembles to nothing, and decode drops the call, so every entry
//! of the fixture translates: `asm_in_loop` (the call first in a loop body), `asm_in_helper` (in a
//! helper the entry inlines) and `asm_free` (none of its own). Both spellings are checked: the
//! opaque-pointer `.ll` (`llvm-as`) and the typed-pointer `.air.ll` (`xcrun metal-as`, which writes
//! the record as Apple's compiler does). `asm_in_loop` runs on the CPU interpreter: every iteration
//! stores. Any other inline assembly a reached call names is refused as `InlineAsm`, by name. A
//! missing tool skips visibly. No GPU is used.

use std::collections::BTreeMap;

use crate::air::error::DiagName;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs};
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text, metal_assemble,
    profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::committed_profiles;

const STEM: &str = "kernel_inline_asm";

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

/// Every entry translates, links and validates on every committed profile, in both spellings:
/// the empty assembly's calls are gone, in the loop body and in the inlined helper alike.
#[test]
fn every_entry_of_the_empty_asm_fixture_translates() {
    let profiles = committed_profiles().expect("profiles");
    for (what, bc) in spellings() {
        for entry in ["asm_in_loop", "asm_in_helper", "asm_free"] {
            for (stem, p) in &profiles {
                let a = compile(&bc, entry, p)
                    .unwrap_or_else(|e| panic!("{what} {entry} on {stem}: {e:?}"));
                link_and_validate(&a, p)
                    .unwrap_or_else(|e| panic!("{what} {entry} on {stem}: {e}"));
            }
        }
    }
}

/// `asm_in_loop` with `n = 6` over one workgroup of four threads: thread t stores its partial
/// sums of `i ^ t` to words `8 + 8t + i`, its sum to word `2t` and its iteration count, 6, to word
/// `2t + 1`, so the loop whose body began with the assembly ran every iteration. Words it does
/// not store keep their sentinel. In both scheduling orders.
#[test]
fn the_loop_that_held_the_empty_asm_runs_every_iteration() {
    const N: u32 = 6;
    let mut want = vec![0xABAB_ABABu32; 40];
    for t in 0..4u32 {
        let mut acc = 0;
        for i in 0..N {
            acc += i ^ t;
            want[(8 + 8 * t + i) as usize] = acc;
        }
        want[(2 * t) as usize] = acc;
        want[(2 * t + 1) as usize] = N;
    }
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for (what, bc) in spellings() {
        let (plan, m) = typed_module(&bc, "asm_in_loop", &p).unwrap_or_else(|e| panic!("{e:?}"));
        let mut spec = BTreeMap::new();
        for (axis, v) in [(Axis::X, 4), (Axis::Y, 1), (Axis::Z, 1)] {
            spec.insert(SpecKey::LocalSize(axis), v);
        }
        spec.insert(SpecKey::FlatLocalSize, 4);
        let slot = |index| ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        };
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes: [0xAB; 160].to_vec(),
                },
                Buffer {
                    slot: Some(slot(1)),
                    address: None,
                    bytes: N.to_le_bytes().to_vec(),
                },
            ],
            push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
            spec,
        };
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                order,
                ..Config::default()
            };
            let out = run(&m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let words: Vec<u32> = out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            assert_eq!(words, want, "{what} {order:?}");
        }
    }
}

/// A kernel calling the assembly `asm` (its flags and strings) of type `ret ()`, then storing 1.
fn asm_module(ret: &str, asm: &str) -> String {
    let call = if ret == "void" {
        format!("call void {asm}()")
    } else {
        format!("%r = call {ret} {asm}()")
    };
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @k(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
  {call}
  store i32 1, ptr addrspace(1) %0, align 4
  ret void
}}

!llvm.module.flags = !{{!0}}
!air.kernel = !{{!20}}
!air.version = !{{!17}}
!air.language_version = !{{!18}}

!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}
!10 = !{{}}
!11 = !{{!12}}
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}
!17 = !{{i32 2, i32 8, i32 0}}
!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
!20 = !{{ptr @k, !10, !11}}
"
    )
}

/// Inline assembly other than the empty `asm sideeffect "", ""` of type `void ()` is refused
/// where the entry reaches it, as `InlineAsm`, not implemented, naming the expression: a
/// template, a clobber, a result, the empty template without `sideeffect`, and with `alignstack`.
/// The empty one translates.
#[test]
fn any_other_inline_asm_is_refused_by_name() {
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for (ret, asm) in [
        ("void", r#"asm sideeffect "nop", """#),
        ("void", r#"asm sideeffect "", "~{memory}""#),
        ("i32", r#"asm sideeffect "", "=r""#),
        ("void", r#"asm "", """#),
        ("void", r#"asm sideeffect alignstack "", """#),
    ] {
        let bc = match llvm_assemble_text(&asm_module(ret, asm)) {
            Ok(bc) => bc,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {asm}: {why}"),
        };
        match compile(&bc, "k", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::InlineAsm, "{asm}: {u:?}");
                assert_eq!(u.needs, Needs::NotImplemented, "{asm}");
                assert_eq!(u.symbol, Some(DiagName::new(asm)), "{asm}");
            }
            other => panic!("{asm}: {other:?}"),
        }
    }
    if let Ok(bc) = llvm_assemble_text(&asm_module("void", r#"asm sideeffect "", """#)) {
        let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{e:?}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{e}"));
    }
}
