//! Atomics on places, and the MSL MoltenVK compiles from them.
//!
//! SPIRV-Cross writes an atomic's pointer as `&expr`; MSL takes the address only of a place, so an
//! atomic on a bare `OpConvertUToPtr` result fails MoltenVK pipeline creation ("cannot take the
//! address of an rvalue", Q1's three atomic-subtraction cases). Every device-address pointer is
//! therefore the member of a one-member `Block` at the address (`FunctionBuilder::device_place`).
//!
//! - The committed `kernel_atomic_xor_and_subtract` fixture (a native float add through a device
//!   address on `moltenvk-m3`) has every atomic of its final words on a place, on every committed
//!   profile.
//! - Its `moltenvk-m3` module, a PSB leaf's word atomics and sub-word store, and a dynamic root's
//!   store with its resident-miss count convert to MSL as MoltenVK converts them and compile with
//!   `xcrun metal` (compile only; nothing runs on a GPU).
//! - In each of those modules every atomic's place is within the builder's documented walk
//!   (at most 5 instructions, the corpus maximum), which the fixture's contained threadgroup
//!   atomics reach.
//!
//! `llvm-as`, `spirv-cross`, and the Metal compiler are external; without them the tests skip
//! visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::SPIRV_WORDS;
use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
use crate::pipeline::testing::{
    atomic_place_scan_steps, atomics_on_places, fixtures_dir, link_and_validate, llvm_assemble,
    meter, moltenvk_msl_compile, profile, AssembleError,
};
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of `text`, or `None` (skipped visibly) without `llvm-as`.
fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-places-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    skip_missing(r, name)
}

fn skip_missing<T>(r: Result<T, AssembleError>, what: &str) -> Option<T> {
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{what}: {why}"),
    }
}

fn fixture() -> Option<Vec<u8>> {
    skip_missing(
        llvm_assemble(&fixtures_dir().join("kernel_atomic_xor_and_subtract.ll")),
        "llvm-as",
    )
}

/// The final (linked, relocated, L3-validated) words of `entry` under `inputs`.
fn final_words(bc: &[u8], entry: &str, p: &DeviceProfile, inputs: &LowerInputs) -> Vec<u32> {
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{entry}: compile: {e}"));
    link_and_validate(&a, p)
        .unwrap_or_else(|e| panic!("{entry}: L3: {e}"))
        .1
}

/// The longest walk of the builder's place check over the whole local corpus on three profiles: its
/// documented bound.
const MAX_PLACE_SCAN: usize = 5;

/// Every atomic of `words` has its place within the builder's documented walk.
fn places_within_the_scan_bound(words: &[u32], what: &str) {
    let steps = atomic_place_scan_steps(words).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        steps <= MAX_PLACE_SCAN,
        "{what}: a place {steps} instructions from its atomic"
    );
}

/// The module's atomics (at least `min`) are on places within the builder's walk, and MoltenVK's
/// MSL of it compiles.
fn compiles_as_moltenvk_msl(words: &[u32], what: &str, min: usize) {
    let atomics = atomics_on_places(words).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(atomics >= min, "{what}: {atomics} atomics");
    places_within_the_scan_bound(words, what);
    skip_missing(moltenvk_msl_compile(words), what);
}

/// Whether the module `words` holds an instruction with opcode `op`.
fn has_opcode(words: &[u32], op: crate::spv::Op) -> bool {
    let mut at = 5;
    while let Some(w) = words.get(at) {
        if w & 0xffff == op as u32 {
            return true;
        }
        at += ((w >> 16) as usize).max(1);
    }
    false
}

/// The fixture's six atomics (device xor, signed and float subtractions, threadgroup store,
/// subtraction, and load) are on places in its final words on every committed profile.
#[test]
fn the_atomic_fixture_has_every_atomic_on_a_place_on_every_profile() {
    let Some(bc) = fixture() else {
        return;
    };
    let profiles = committed_profiles().expect("profiles");
    assert!(profiles.iter().any(|(stem, _)| stem == "moltenvk-m3"));
    for (stem, p) in profiles {
        let words = final_words(
            &bc,
            "kernel_atomic_xor_and_subtract",
            &p,
            &LowerInputs::new(),
        );
        let atomics = atomics_on_places(&words).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(atomics >= 6, "{stem}: {atomics} atomics");
        places_within_the_scan_bound(&words, &stem);
    }
}

/// Q1's failure: on `moltenvk-m3` the fixture's `air.atomic.global.sub.f32` is a native float add
/// through a device address, and MoltenVK's MSL of the module compiles.
#[test]
fn the_atomic_fixture_compiles_as_moltenvk_msl() {
    let Some(bc) = fixture() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let words = final_words(
        &bc,
        "kernel_atomic_xor_and_subtract",
        &p,
        &LowerInputs::new(),
    );
    assert!(
        has_opcode(&words, crate::spv::Op::AtomicFAddEXT),
        "the float subtraction is a native float add on moltenvk-m3"
    );
    compiles_as_moltenvk_msl(&words, "kernel_atomic_xor_and_subtract", 6);
}

const PSB_LEAF: &str = r#"define void @psb_leaf(ptr addrspace(1) %buf, i32 %t) {
entry:
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %t
  %o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 5, i32 0, i32 2, i1 true)
  %b = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 %o
  store i8 7, ptr addrspace(1) %b, align 1
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @psb_leaf, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A slot reached through its device address (the PSB leaf): the atomic's word place, the byte
/// store's `uchar` place (`moltenvk-m3` has `storageBuffer8BitAccess`, so it is one narrow store,
/// not the sub-word store's two word atomics), and the root-table entries.
#[test]
fn a_psb_leaf_compiles_as_moltenvk_msl() {
    let Some(bc) = assemble("psb_leaf", &format!("{HEADER}{PSB_LEAF}")) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 0,
    });
    let words = final_words(&bc, "psb_leaf", &p, &inputs);
    // The add; the byte store is a narrow store.
    compiles_as_moltenvk_msl(&words, "psb_leaf", 1);
}

const DYNAMIC_STORE: &str = r#"define void @dynamic_store(ptr addrspace(1) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1) {
entry:
  %k = load i32, ptr addrspace(1) %1, align 4
  %pp = getelementptr ptr addrspace(1), ptr addrspace(1) %0, i32 %k
  %p = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  store i16 -21829, ptr addrspace(1) %p, align 1
  ret void
}

!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}
!9 = !{ptr @dynamic_store, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"ptrs"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"params"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A dynamic root (a pointer loaded from a buffer): the root-table search, the sub-word store's
/// word atomics through the wide pointer, and the resident-miss count's status words.
#[test]
fn a_dynamic_root_compiles_as_moltenvk_msl() {
    let Some(bc) = assemble("dynamic_store", &format!("{HEADER}{DYNAMIC_STORE}")) else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let words = final_words(&bc, "dynamic_store", &p, &LowerInputs::new());
    // The miss count and bit, and the halfword store's clear and set (of one or two words).
    compiles_as_moltenvk_msl(&words, "dynamic_store", 4);
}
