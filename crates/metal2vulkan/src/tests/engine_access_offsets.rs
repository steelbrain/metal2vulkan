//! Under the identity domains no access's device address is scaled in 64 bits: each unit's byte
//! offset from the address an access is reached from is its index shifted in 32 bits, widened
//! once at the add (`lower::access::psb`'s `Reach::Below` and `Reach::Direct`). The runtime makes
//! no buffer longer than 2 GiB, so the 32-bit product is exact. The kernels: a `uchar` and a
//! `ushort` buffer read and written by 4-byte accesses at a byte offset the kernel loads, reached
//! from the word below each access; `buffer_end_half_load`, a `ushort` buffer by `i16` loads; and
//! the vector word-phase fixture's `uint` and `float4` buffers, reached from their addresses as
//! they are. Each is compiled on every identity profile and passes L3, and no 64-bit
//! `OpShiftLeftLogical` or `OpIMul` reaches the integer an `OpConvertUToPtr` converts.
//!
//! `llvm-as` assembles the kernels; without it the test skips visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use super::engine_word_phases::identity_profiles;
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv_binary::{parse_bytes, Operand};
use crate::spv::Op;
use metal2vulkan_devtools::vector_word_phases::{FIXTURE, VEC16, WORDS};

/// A kernel whose buffer `data` is held to `align` bytes (a `uchar` for 1, a `ushort` for 2): it
/// loads a word at the byte offset in its `constant` buffer `n` and stores it plus one 8 bytes
/// further, both accesses claiming `align`.
fn word_kernel(align: u32) -> String {
    let name = match align {
        1 => "uchar",
        _ => "ushort",
    };
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @words_below(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(2) noundef %1) {{\n\
entry:\n\
  %n = load i32, ptr addrspace(2) %1, align 4\n\
  %p = getelementptr i8, ptr addrspace(1) %0, i32 %n\n\
  %v = load i32, ptr addrspace(1) %p, align {align}\n\
  %s = add i32 %v, 1\n\
  %m = add i32 %n, 8\n\
  %q = getelementptr i8, ptr addrspace(1) %0, i32 %m\n\
  store i32 %s, ptr addrspace(1) %q, align {align}\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @words_below, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {align}, !\"air.arg_type_align_size\", i32 {align}, !\"air.arg_type_name\", !\"{name}\", !\"air.arg_name\", !\"data\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `text` assembled in a scratch directory of its own, removed once `llvm-as` is done; `None`
/// (the test skips) without `llvm-as`.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-offsets-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    assembled(name, r)
}

fn assembled(name: &str, r: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(e) => panic!("{name}: {e:?}"),
    }
}

/// The 64-bit `OpShiftLeftLogical`s and `OpIMul`s of serialized module `words` that reach the
/// integer operand of an `OpConvertUToPtr` through 64-bit integer results, and the number of
/// `OpConvertUToPtr`s.
fn wide_scales(words: &[u32]) -> (usize, usize) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = parse_bytes(&bytes).expect("parse");
    let wide: BTreeSet<u32> = (module.instructions.iter())
        .filter(|i| i.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::Word(64)))
        .filter_map(|i| i.result_id)
        .collect();
    let defs: BTreeMap<u32, _> = (module.instructions.iter())
        .filter_map(|i| Some((i.result_id?, i)))
        .collect();
    let mut seen = BTreeSet::new();
    let mut scales = BTreeSet::new();
    let mut stack: Vec<u32> = Vec::new();
    let mut converts = 0;
    for i in &module.instructions {
        if i.opcode != Op::ConvertUToPtr {
            continue;
        }
        converts += 1;
        if let Some(Operand::IdRef(v)) = i.operands.first() {
            stack.push(*v);
        }
    }
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(def) = defs.get(&id) else {
            continue;
        };
        if !def.result_type.is_some_and(|t| wide.contains(&t)) {
            continue;
        }
        if matches!(def.opcode, Op::ShiftLeftLogical | Op::IMul) {
            scales.insert(id);
        }
        stack.extend(def.operands.iter().filter_map(|o| match o {
            Operand::IdRef(v) => Some(*v),
            _ => None,
        }));
    }
    (scales.len(), converts)
}

#[test]
fn no_access_address_is_scaled_in_64_bits() {
    let mut kernels: Vec<(String, Vec<u8>, &str)> = Vec::new();
    for align in [1, 2] {
        let Some(bc) = assemble_text("words_below", &word_kernel(align)) else {
            return;
        };
        kernels.push((format!("words_below at {align}"), bc, "words_below"));
    }
    for (fixture, entry) in [
        ("buffer_end_half_load", "buffer_end_half_load"),
        (FIXTURE, WORDS),
        (FIXTURE, VEC16),
    ] {
        let path = engine_dir().join(format!("{fixture}.ll"));
        let Some(bc) = assembled(fixture, llvm_assemble(&path)) else {
            return;
        };
        kernels.push((entry.to_owned(), bc, entry));
    }
    for (what, bc, entry) in &kernels {
        for (stem, p) in &identity_profiles() {
            let what = format!("{what} on {stem}");
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (scales, converts) = wide_scales(&a.module.words);
            assert!(converts > 0, "{what}: no device-address access");
            assert_eq!(scales, 0, "{what}: an access address scaled in 64 bits");
        }
    }
}

/// A kernel over a `ulong` buffer `buf` and a `float4` buffer `vec` (argument alignments 8 and
/// 16): a 64-bit `atomic_max` on `buf`'s first word, or, where `wide`, on the word at the address
/// `buf`'s first word holds (a pointer made from an integer, which may reach either buffer), and
/// `vec[t] = -vec[t]`.
fn held_kernel(wide: bool) -> String {
    let target = match wide {
        true => {
            "  %w = load i64, ptr addrspace(1) %0, align 8\n  %a = inttoptr i64 %w to ptr addrspace(1)\n"
        }
        false => "  %a = getelementptr i8, ptr addrspace(1) %0, i32 0\n",
    };
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @held(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, i32 noundef %t) {{\n\
entry:\n\
{target}\
  %v = zext i32 %t to i64\n\
  call void @air.atomic.global.max.u.i64(ptr addrspace(1) %a, i64 %v, i32 0, i32 2, i1 true)\n\
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i32 %t\n\
  %x = load <4 x float>, ptr addrspace(1) %p, align 16\n\
  %y = fneg <4 x float> %x\n\
  store <4 x float> %y, ptr addrspace(1) %p, align 16\n\
  ret void\n\
}}\n\n\
declare void @air.atomic.global.max.u.i64(ptr addrspace(1), i64, i32, i32, i1)\n\n\
!air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
!9 = !{{ptr @held, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13, !14}}\n\
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"buf\"}}\n\
!13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"vec\"}}\n\
!14 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
!17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The alignment each buffer slot states its address is held to (`RuntimeFact::AddressAlignment`)
/// is its argument's, 8 and 16, and stays so for a slot bound at its alignment. Under
/// `LoweringInputs::word_bound_buffers`, both slots bound at a multiple of 4 below their
/// alignment, it is what the slot's accesses need: the `float4` buffer's 4, which plain Metal
/// runs bound at any multiple of 4, and the `ulong` buffer's 8, where a 64-bit atomic reaches it,
/// as Vulkan's 64-bit atomic needs its natural alignment. Where the atomic goes through a pointer
/// made from an integer, which may reach any buffer, both state 8. Each module passes L3.
#[test]
fn a_word_bound_slot_is_held_to_what_its_accesses_need() {
    use crate::lower::select::LowerInputs;
    use crate::pipeline::sym::{SlotClass, SlotRef, Stage};
    let mut bound = LowerInputs::new();
    bound.word_bound_buffers = (0..2)
        .map(|index| SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index,
        })
        .collect();
    let plain = LowerInputs::new();
    for (wide, inputs, want) in [
        (false, &plain, [(0, 8), (1, 16)]),
        (true, &plain, [(0, 8), (1, 16)]),
        (false, &bound, [(0, 8), (1, 4)]),
        (true, &bound, [(0, 8), (1, 8)]),
    ] {
        let Some(bc) = assemble_text("held", &held_kernel(wide)) else {
            return;
        };
        let bound = !inputs.word_bound_buffers.is_empty();
        let mut compiled = 0;
        for (stem, p) in &identity_profiles() {
            let what = format!("held (wide {wide}, word-bound {bound}) on {stem}");
            let a = match compile_with(&bc, "held", p, inputs) {
                Ok(a) => a,
                // A profile without 64-bit buffer atomics refuses the kernel.
                Err(e) if e.to_string().contains("Int64") => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let stated: Vec<(u32, u32)> = (a.facts.iter())
                .filter_map(|f| match f {
                    crate::pipeline::facts::RuntimeFact::AddressAlignment { slot, align } => {
                        Some((slot.index, *align))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(stated, want, "{what}: stated alignments");
            compiled += 1;
        }
        assert!(
            compiled > 0,
            "held (wide {wide}, word-bound {bound}): no profile compiled it"
        );
    }
}
