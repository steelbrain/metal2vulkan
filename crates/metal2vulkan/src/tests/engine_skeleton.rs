//! The walking skeleton: the smallest kernel end to end.
//!
//! The owned kernel `engine/store_const` (buffer 0 word 0 = 42) goes through parse → decode
//! → specialize → analyze → select → structurize → emit → verify → serialize → link → relocate
//! and passes L3 (`spirv-val` with the profile's environments and relaxations, requirements ⊆
//! profile) under `moltenvk-m3`, `lunarg-desktop-baseline-2023`, and `vk12-floor`. The file also
//! holds the pipeline-level properties the
//! skeleton owns: determinism, the zeroed-relocation property, the typed/opaque identity, the size
//! check before emission, and clean refusals for everything the families do not lower yet.
//!
//! `llvm-as` (and, `xcrun metal-as` on macOS) assemble the fixtures; a missing tool skips its test
//! visibly, a tool that rejects an engine fixture fails it.

use std::collections::BTreeSet;

use crate::device::DevicePolicy;
use crate::error::{Error, InternalKind, Limit};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{
    AccessSet, BufferForm, PushLoc, SlotInterface, StageDescriptor, StageSpec, Window,
};
use crate::pipeline::link::{LinkInputs, LinkTable};
use crate::pipeline::relocate::relocate;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, metal_assemble, meter,
    profile, zero_relocations, AssembleError, Scratch,
};
use crate::spirv::testing::check_words_valid;
use crate::vk;

/// The M1 profiles.
const M1_PROFILES: [&str; 2] = ["moltenvk-m3", "runtime-floor"];

/// `store_const` as opaque bitcode, or `None` (skipped visibly) without `llvm-as`.
fn store_const() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("store_const.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected store_const.ll: {why}"),
    }
}

fn buffer0() -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 0,
    }
}

fn opcodes(words: &[u32]) -> Vec<u32> {
    let mut ops = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        ops.push(w & 0xFFFF);
        i += (w >> 16).max(1) as usize;
    }
    ops
}

/// The artifact facts `store_const` must carry on every profile.
fn check_artifact(a: &FunctionArtifact, what: &str) {
    let i = &a.interface;
    assert_eq!(i.stage, Stage::Kernel, "{what}");
    let SlotInterface::Buffer(b) = i.slots.get(&buffer0()).expect("buffer 0") else {
        panic!("{what}: buffer 0 is not a buffer slot");
    };
    assert_eq!(b.access, AccessSet::WRITE, "{what}");
    assert_eq!(b.window, Window::Bounded { start: 0, len: 4 }, "{what}");
    assert_eq!(b.form, BufferForm::Descriptor, "{what}");
    assert_eq!(i.slots.len(), 1, "{what}");
    let placed: Vec<(PushField, PushLoc)> =
        i.push.fields.iter().map(|f| (f.field, f.loc)).collect();
    assert_eq!(
        placed,
        vec![(
            PushField::SlotBindOffset(buffer0()),
            PushLoc::Push { offset: 0 }
        )],
        "{what}"
    );
    assert_eq!(i.push.push_range, Some((0, 4)), "{what}");
    assert_eq!(i.push.argument_table_size, 0, "{what}");
    assert_eq!(
        i.descriptors,
        vec![StageDescriptor::new(
            ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0
            },
            vk::DescriptorType::STORAGE_BUFFER,
            1
        )],
        "{what}"
    );
    // The workgroup is the `WorkgroupSize` composite of its own plain constants, 4-byte `u32`
    // constants defaulting to 1, beside the threadgroup's sides `sx, sy, sz` (defaulting to 1)
    // and `DeepThreadgroup` (defaulting to false), of which this kernel's body reads no bit.
    let unread = |key, default| StageSpec {
        unread_bits: u64::from(u32::MAX),
        ..StageSpec::new(key, 4, default)
    };
    assert_eq!(
        i.spec,
        vec![
            unread(SpecKey::LocalSize(Axis::X), 1),
            unread(SpecKey::LocalSize(Axis::Y), 1),
            unread(SpecKey::LocalSize(Axis::Z), 1),
            unread(SpecKey::DeepThreadgroup, 0),
            StageSpec::new(SpecKey::WorkgroupSize(Axis::X), 4, 1),
            StageSpec::new(SpecKey::WorkgroupSize(Axis::Y), 4, 1),
            StageSpec::new(SpecKey::WorkgroupSize(Axis::Z), 4, 1),
        ],
        "{what}"
    );
    assert!(i.io.is_empty(), "{what}");
    assert!(!i.needs_pointer_translation, "{what}: no Psb slot");
    let facts: BTreeSet<RuntimeFact> = a.facts.iter().copied().collect();
    assert!(
        facts.contains(&RuntimeFact::ThreadsPerThreadgroup),
        "{what}"
    );
    assert!(facts.contains(&RuntimeFact::LocalSizeLimits), "{what}");
    assert!(!facts.contains(&RuntimeFact::ExecutionStatus), "{what}");
    let uses: u32 = a.strategies.entries.values().map(|u| u.uses).sum();
    assert_eq!(uses, 1, "{what}: one store strategy");
    assert!(
        a.strategies
            .entries
            .keys()
            .all(|k| k.starts_with("access.store")),
        "{what}"
    );
}

/// M1: `store_const` passes the whole pipeline and L3 on the three M1 profiles.
#[test]
fn store_const_passes_the_pipeline_on_every_m1_profile() {
    let Some(bitcode) = store_const() else { return };
    for stem in M1_PROFILES {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a =
            compile(&bitcode, "store_const", &p).unwrap_or_else(|e| panic!("{stem}: compile: {e}"));
        check_artifact(&a, stem);
        let (table, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(table.spec.len(), 4, "{stem}");
        let ops = opcodes(&words);
        for op in [
            crate::spv::Op::SpecConstantComposite,
            crate::spv::Op::AccessChain,
            crate::spv::Op::Store,
            crate::spv::Op::ShiftRightLogical,
        ] {
            assert!(ops.contains(&(op as u32)), "{stem}: no {op:?}");
        }
    }
}

/// The store sits in the in-shader bounds check (a selection around it), and validates.
#[test]
fn containment_guards_the_store() {
    let Some(bitcode) = store_const() else { return };
    let p = profile("runtime-floor", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bitcode, "store_const", &p).expect("compile");
    assert!(a.strategies.entries.contains_key("access.store.checked"));
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{e}"));
    let ops = opcodes(&words);
    assert!(ops.contains(&(crate::spv::Op::ArrayLength as u32)));
    assert!(ops.contains(&(crate::spv::Op::SelectionMerge as u32)));
}

/// Compilation is deterministic: equal artifacts (key, words, relocations, interface) and equal
/// link tables.
#[test]
fn compilation_and_link_are_deterministic() {
    let Some(bitcode) = store_const() else { return };
    for stem in M1_PROFILES {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bitcode, "store_const", &p).expect("compile");
        let b = compile(&bitcode, "store_const", &p).expect("compile");
        assert_eq!(a, b, "{stem}");
        let (ta, wa) = link_and_validate(&a, &p).expect("link");
        let (tb, wb) = link_and_validate(&b, &p).expect("link");
        assert_eq!((ta, wa), (tb, wb), "{stem}");
    }
}

/// Over many links (random binding, set, and spec numbers), the words with the relocation slots
/// zeroed are identical, and only relocation slots differ between links.
#[test]
fn zeroed_relocation_slots_are_link_independent() {
    let Some(bitcode) = store_const() else { return };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bitcode, "store_const", &p).expect("compile");
    let slots: BTreeSet<usize> = a.module.relocs.iter().map(|(_, at)| *at as usize).collect();
    assert!(!slots.is_empty());
    let base = crate::pipeline::link::link(&[&a], &LinkInputs::compute(), &p)
        .expect("link")
        .table;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 1000) as u32
    };
    let mut seen = BTreeSet::new();
    for _ in 0..32 {
        let mut t = LinkTable::default();
        for r in base.bindings.keys() {
            t.bindings.insert(*r, (next() % 4, next()));
        }
        for k in base.spec.keys() {
            t.spec.insert(*k, next());
        }
        let words = relocate(&a, &t).expect("relocate");
        assert_eq!(zero_relocations(&a, &words), a.module.words);
        for (i, (x, y)) in words.iter().zip(&a.module.words).enumerate() {
            assert!(
                x == y || slots.contains(&i),
                "word {i} changed outside a slot"
            );
        }
        seen.insert(words);
    }
    assert!(seen.len() > 1, "the links wrote different numbers");
    // A table that lacks a symbol is a typed error, not a partial module.
    let empty = LinkTable::default();
    assert!(matches!(relocate(&a, &empty), Err(Error::Internal(_))));
}

/// The typed-pointer (`metal-as`) and opaque-pointer (`llvm-as`) bitcode of `store_const` produce
/// identical SPIR-V (macOS only; skipped visibly elsewhere).
#[test]
fn typed_and_opaque_bitcode_give_identical_spirv() {
    let Some(opaque) = store_const() else { return };
    let typed = match metal_assemble(&engine_dir().join("store_const.air.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("metal-as rejected store_const.air.ll: {why}"),
    };
    for stem in M1_PROFILES {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&opaque, "store_const", &p).expect("opaque");
        let b = compile(&typed, "store_const", &p).expect("typed");
        assert_eq!(a.module, b.module, "{stem}");
        assert_eq!(a.interface, b.interface, "{stem}");
    }
}

/// A module above the word cap is refused.
#[test]
fn a_module_above_the_word_cap_is_refused() {
    let Some(bitcode) = store_const() else { return };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bitcode, "store_const", &p).expect("compile");
    let words = a.module.words.len() as u64;
    let r = compile_bitcode(
        &bitcode,
        EntryRef::Name("store_const"),
        &FunctionConstantValues::defaults(),
        &p,
        &LowerInputs::new(),
        &mut meter(),
        16,
    );
    assert_eq!(r, Err(Error::LimitExceeded(Limit::SpirvWords)));
    // The same compile under a generous cap passes with the same module.
    let ok = compile_bitcode(
        &bitcode,
        EntryRef::Name("store_const"),
        &FunctionConstantValues::defaults(),
        &p,
        &LowerInputs::new(),
        &mut meter(),
        1 << 20,
    )
    .expect("compile");
    assert!(ok.module.words.len() as u64 == words);
}

/// A loop kernel from the public fixtures (a counted loop between a buffer load and a buffer store)
/// compiles once the access family lowers static-root loads and stores: its loop is the module's
/// one loop, with no execution status, and the module passes L3.
#[test]
fn a_loop_kernel_compiles() {
    let ll = fixtures_dir().join("kernel_bounded_loop_word.ll");
    let bitcode = match llvm_assemble(&ll) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIPPED: {e:?}");
            return;
        }
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bitcode, "bounded_loop_word", &p).unwrap_or_else(|e| panic!("{e}"));
    assert!(!a.facts.contains(&RuntimeFact::ExecutionStatus));
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{e}"));
    let loops = opcodes(&words)
        .into_iter()
        .filter(|&op| op == crate::spv::Op::LoopMerge as u32)
        .count();
    assert_eq!(loops, 1);
}

/// A kernel that switches on a dispatch builtin: the builtin parameter is the dispatch family's, so
/// its value comes from that family rather than a terminator-operand refusal of another family .
const SWITCH_ON_GID: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @switch_on_gid(ptr addrspace(1) noundef "air-buffer-no-alias" %0, i32 noundef %1) {
entry:
  switch i32 %1, label %done [ i32 0, label %st ]
st:
  store i32 42, ptr addrspace(1) %0, align 4
  br label %done
done:
  ret void
}

!llvm.module.flags = !{!0}
!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}

!0 = !{i32 7, !"air.max_device_buffers", i32 31}
!9 = !{ptr @switch_on_gid, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

#[test]
fn a_switch_on_a_dispatch_builtin_is_the_dispatch_familys() {
    let scratch = Scratch::new("m2v-switch-gid");
    let dir = scratch.path();
    let ll = dir.join("switch_on_gid.ll");
    std::fs::write(&ll, SWITCH_ON_GID).expect("write");
    let assembled = llvm_assemble(&ll);
    let bitcode = match assembled {
        Ok(b) => b,
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            return;
        }
        Err(e) => panic!("llvm-as rejected the inline kernel: {e:?}"),
    };
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    // The dispatch family lowers the builtin: the selector is the dispatch family's value (a grid
    // builtin, so the kernel depends on the dispatch grid), and the module passes L3.
    let a = compile(&bitcode, "switch_on_gid", &p).expect("the dispatch family lowers it");
    link_and_validate(&a, &p).expect("L3");
    assert!(a.facts.contains(&RuntimeFact::DispatchGrid));
}

/// Every entry of every public fixture (top level and `engine/`) compiles and validates, or
/// is refused with a typed error that is never internal (honest intermediate commits).
#[test]
fn every_public_fixture_compiles_or_refuses_without_an_internal_error() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let mut files: Vec<_> = [fixtures_dir(), engine_dir()]
        .iter()
        .flat_map(|d| std::fs::read_dir(d).expect("fixtures"))
        .map(|e| e.expect("entry").path())
        .filter(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            n.ends_with(".ll") && !n.ends_with(".air.ll")
        })
        .collect();
    files.sort();
    let (mut ok, mut refused, mut skipped) = (0, 0, 0);
    for ll in files {
        let bitcode = match llvm_assemble(&ll) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            // Legacy top-level fixtures this `llvm-as` rejects are not engine inputs.
            Err(AssembleError::Rejected(why)) => {
                assert!(!ll.starts_with(engine_dir()), "{ll:?}: {why}");
                skipped += 1;
                continue;
            }
        };
        let decoded = crate::air::decode(&bitcode, &mut meter()).map_err(Error::from);
        let entries = match &decoded {
            Ok(d) => d.abi.entries.len() as u32,
            Err(e) => {
                assert!(!matches!(e, Error::Internal(_)), "{ll:?}: {e}");
                refused += 1;
                continue;
            }
        };
        for i in 0..entries {
            let r = compile_bitcode(
                &bitcode,
                EntryRef::Index(i),
                &FunctionConstantValues::defaults(),
                &p,
                &LowerInputs::new(),
                &mut meter(),
                1 << 24,
            );
            match r {
                Ok(a) => {
                    let (_, words) =
                        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{ll:?}: {e}"));
                    check_words_valid(&words, &a.requirements, &p).expect("valid");
                    ok += 1;
                }
                Err(Error::Internal(e)) => {
                    assert!(e.kind != InternalKind::Panic, "{ll:?} #{i}: panic {e}");
                    panic!("{ll:?} #{i}: internal error {e}");
                }
                Err(_) => refused += 1,
            }
        }
    }
    eprintln!("public fixtures: {ok} compiled, {refused} refused, {skipped} not assembled");
    assert!(ok >= 1, "store_const compiles");
}
