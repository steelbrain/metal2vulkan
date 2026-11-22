//! Buffer-table access end to end.
//!
//! A `[[buffer(1)]]` table of three `device uint *` elements (an `array_ref`, whose LLVM
//! parameter is a thread-space pointer to the element pointers) is read one element pointer at a
//! time; each element is its own slot `1 + k`:
//!
//! - constant element indices fold to the element's static fat pointer (`access.buffer_table.load`)
//!   and read and write exactly the buffer bound at that slot;
//! - a dynamic element index (a loaded `uint`, extended to `i64`) selects the element by its
//!   clamped index (`access.buffer_table.load.clamped`): in-range indices are exact, and an
//!   out-of-range or negative index is clamped into the table (contained: it reaches the last
//!   element and nothing else changes).
//!
//! Every case compiles on `runtime-floor` and `moltenvk-m3`, passes L3, and runs byte-exact on the
//! CPU interpreter. `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The table's first Metal slot and element count.
const FIRST: u32 = 1;
const COUNT: u32 = 3;

/// `runtime-floor` and `moltenvk-m3`, under the resident-table domains.
fn profiles() -> Vec<(String, DeviceProfile)> {
    ["runtime-floor", "moltenvk-m3"]
        .into_iter()
        .map(|stem| {
            let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("committed profile");
            (stem.to_owned(), p)
        })
        .collect()
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-buffer-table-{}-{}-{name}",
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

/// A kernel `name` over `out` (`[[buffer(0)]]`, a `device uint *`) and the table `tab`
/// (`[[buffer(1..4)]]`, parameter `%1`, a thread-space pointer to three device pointers).
fn kernel(name: &str, body: &str) -> String {
    format!(
        "{HEADER}define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr noundef %1) {{\nentry:\n{body}  ret void\n}}\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
         !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 {FIRST}, i32 {COUNT}, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tab\"}}\n\
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

fn bound(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(kslot(index)),
        address: None,
        bytes,
    }
}

/// Push bytes of `plan` with zero residuals (every field is in the push window).
fn push_bytes(plan: &Plan) -> Vec<u8> {
    for f in &plan.push.fields {
        assert!(
            matches!(f.loc, PushLoc::Push { .. }),
            "{:?} in the argument table",
            f.field
        );
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "{:?}",
            f.field
        );
    }
    vec![0u8; 128]
}

/// 16 distinct bytes for element `k`.
fn element_bytes(k: u32) -> Vec<u8> {
    (0..16u8)
        .map(|i| i.wrapping_mul(17).wrapping_add(0x30 + 0x40 * k as u8))
        .collect()
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("word"))
}

/// The buffers of a run: `out` holding `input` in word 0, then the table's elements.
fn buffers(input: u32) -> Vec<Buffer> {
    let mut out = vec![0u8; 16];
    out[..4].copy_from_slice(&input.to_le_bytes());
    let mut b = vec![bound(0, out)];
    for k in 0..COUNT {
        b.push(bound(FIRST + k, element_bytes(k)));
    }
    b
}

/// Constant indices: `out[1] = tab[0][0] + tab[2][1]` and `tab[2][0] = tab[0][0]`; each load folds
/// to its element's fat pointer and reaches exactly slots 1 and 3.
#[test]
fn constant_table_indices_reach_their_elements_exactly() {
    let body = "  %e0 = load ptr addrspace(1), ptr %1, align 8\n  %t2 = getelementptr inbounds ptr addrspace(1), ptr %1, i64 2\n  %e2 = load ptr addrspace(1), ptr %t2, align 8\n  %a = load i32, ptr addrspace(1) %e0, align 4\n  %q = getelementptr inbounds i32, ptr addrspace(1) %e2, i64 1\n  %b = load i32, ptr addrspace(1) %q, align 4\n  %s = add i32 %a, %b\n  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %s, ptr addrspace(1) %o, align 4\n  store i32 %a, ptr addrspace(1) %e2, align 4\n";
    let Some(bc) = assemble_text("table_const", &kernel("table_const", body)) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "table_const", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let names: Vec<&String> = a.strategies.entries.keys().collect();
        assert!(
            names.iter().any(|n| *n == "access.buffer_table.load"),
            "{what}: {names:?}"
        );
        assert!(
            !names
                .iter()
                .any(|n| *n == "access.buffer_table.load.clamped"),
            "{what}: {names:?}"
        );
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "table_const", &p).expect("module");
        let slots: Vec<u32> = plan.slots.iter().flatten().map(|c| c.slot.index).collect();
        assert_eq!(slots, vec![0, FIRST, FIRST + 2], "{what}");
        let res = Resources {
            buffers: buffers(0),
            push: push_bytes(&plan),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (e0, e2) = (element_bytes(0), element_bytes(2));
        let sum = word(&e0, 0).wrapping_add(word(&e2, 4));
        assert_eq!(word(&out.buffers[0], 4), sum, "{what}");
        assert_eq!(out.buffers[1], e0, "{what}: element 0 unchanged");
        assert_eq!(
            out.buffers[2],
            element_bytes(1),
            "{what}: element 1 unchanged"
        );
        let mut want = e2.clone();
        want[..4].copy_from_slice(&e0[..4]);
        assert_eq!(out.buffers[3], want, "{what}: element 2 written");
    }
}

/// A dynamic index: `k = out[0]` (`zext` or `sext` to `i64`), `out[1] = tab[k][0]`, and
/// `tab[k][1] = k`. Indices inside the table are exact; outside it (3, 1000, `-1`) the index is
/// clamped to the last element (contained).
#[test]
fn a_dynamic_table_index_is_exact_and_clamped_into_the_table() {
    for ext in ["zext", "sext"] {
        let name = format!("table_dynamic_{ext}");
        let body = format!(
            "  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = {ext} i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %el = load ptr addrspace(1), ptr %tp, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n  %w = getelementptr inbounds i32, ptr addrspace(1) %el, i64 1\n  store i32 %k, ptr addrspace(1) %w, align 4\n"
        );
        let Some(bc) = assemble_text(&name, &kernel(&name, &body)) else {
            return;
        };
        for (what, p) in profiles() {
            let what = format!("{what} {name}");
            let a = compile(&bc, &name, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let names: Vec<&String> = a.strategies.entries.keys().collect();
            assert!(
                names
                    .iter()
                    .any(|n| *n == "access.buffer_table.load.clamped"),
                "{what}: {names:?}"
            );
            assert!(
                names.iter().any(|n| *n == "arith.index_extension"),
                "{what}: {names:?}"
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, &name, &p).expect("module");
            let slots: Vec<u32> = plan.slots.iter().flatten().map(|c| c.slot.index).collect();
            assert_eq!(slots, vec![0, 1, 2, 3], "{what}");
            for (input, element) in [
                (0u32, 0u32),
                (1, 1),
                (2, 2),
                (3, 2),
                (1000, 2),
                (u32::MAX, 2),
            ] {
                let res = Resources {
                    buffers: buffers(input),
                    push: push_bytes(&plan),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res)
                    .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
                let e = element_bytes(element);
                assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
                for k in 0..COUNT {
                    let mut want = element_bytes(k);
                    if k == element {
                        want[4..8].copy_from_slice(&input.to_le_bytes());
                    }
                    assert_eq!(
                        out.buffers[1 + k as usize],
                        want,
                        "{what} {input}: element {k}"
                    );
                }
            }
        }
    }
}

/// A table pointer computed in one block and loaded through in another is demoted as its `u32`
/// offset; the element it names is exact (and the index extension crosses too).
#[test]
fn a_table_pointer_crossing_blocks_is_exact() {
    let body = "  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %c = icmp ult i32 %k, 7\n  br i1 %c, label %read, label %done\nread:\n  %el = load ptr addrspace(1), ptr %tp, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n  br label %done\ndone:\n";
    let Some(bc) = assemble_text("table_cross", &kernel("table_cross", body)) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "table_cross", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "table_cross", &p).expect("module");
        assert!(
            plan.cross
                .iter()
                .any(|(_, t)| *t == crate::spirv::ids::DemoteTy::ArenaOffset),
            "{what}: {:?}",
            plan.cross
        );
        for input in 0..COUNT {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            let e = element_bytes(input);
            assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
        }
    }
}

/// A table pointer parked in a private cell and read back is its `u32` offset there; the element it
/// names is exact, and an out-of-range index is still clamped into the table. The cell is written
/// twice: one store's value would be read directly (`canon::forward`), leaving no cell.
#[test]
fn a_table_pointer_parked_in_a_private_cell_is_exact() {
    let body = "  %cell = alloca ptr, align 8\n  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  store ptr %tp, ptr %cell, align 8\n  store ptr %tp, ptr %cell, align 8\n  %back = load ptr, ptr %cell, align 8\n  %el = load ptr addrspace(1), ptr %back, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n";
    let Some(bc) = assemble_text("table_cell", &kernel("table_cell", body)) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "table_cell", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let names: Vec<&String> = a.strategies.entries.keys().collect();
        for want in [
            "access.cell.store",
            "access.cell.load",
            "access.buffer_table.load.clamped",
        ] {
            assert!(names.iter().any(|n| *n == want), "{what}: {names:?}");
        }
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "table_cell", &p).expect("module");
        for (input, element) in [(0u32, 0u32), (1, 1), (2, 2), (9, 2)] {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            let e = element_bytes(element);
            assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
        }
    }
}

/// A dynamic element load aligned below the pointer size could read parts of two element
/// pointers: it is refused as not implemented, never guessed.
#[test]
fn an_underaligned_dynamic_table_load_is_refused() {
    let body = "  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %el = load ptr addrspace(1), ptr %tp, align 4\n  %v = load i32, ptr addrspace(1) %el, align 4\n  store i32 %v, ptr addrspace(1) %0, align 4\n";
    let Some(bc) = assemble_text("table_underaligned", &kernel("table_underaligned", body)) else {
        return;
    };
    for (what, p) in profiles() {
        match compile(&bc, "table_underaligned", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Lowering(LowerFamily::Access), "{what}");
                assert_eq!(u.needs, Needs::NotImplemented, "{what}");
            }
            other => panic!("{what}: expected a refusal, got {other:?}"),
        }
    }
}

/// A table pointer and a buffer pointer carried by value in a struct (MPS's `_MPSKernelInOut` form,
/// `{ { ptr, i64 }, ptr addrspace(1) }`, built by `insertvalue` and read by `extractvalue` in
/// another block) are forwarded: each extraction is its inserted pointer, so `out[1] = tab[k][0]`
/// through the struct's table pointer and `out` through its buffer pointer, with the length member
/// beside them, are exact, and an out-of-range index is still clamped into the table.
#[test]
fn table_and_buffer_pointers_carried_in_a_struct_are_forwarded() {
    let body = "  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %r0 = insertvalue { ptr, i64 } poison, ptr %tp, 0\n  %r1 = insertvalue { ptr, i64 } %r0, i64 3, 1\n  %s0 = insertvalue { { ptr, i64 }, ptr addrspace(1) } poison, { ptr, i64 } %r1, 0\n  %s1 = insertvalue { { ptr, i64 }, ptr addrspace(1) } %s0, ptr addrspace(1) %0, 1\n  br label %use\nuse:\n  %back = extractvalue { { ptr, i64 }, ptr addrspace(1) } %s1, 0, 0\n  %n = extractvalue { { ptr, i64 }, ptr addrspace(1) } %s1, 0, 1\n  %out = extractvalue { { ptr, i64 }, ptr addrspace(1) } %s1, 1\n  %el = load ptr addrspace(1), ptr %back, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n  %n32 = trunc i64 %n to i32\n  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2\n  store i32 %n32, ptr addrspace(1) %o2, align 4\n";
    let Some(bc) = assemble_text("table_struct", &kernel("table_struct", body)) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "table_struct", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let forwards = a
            .strategies
            .entries
            .get("arith.aggregate.forward")
            .map_or(0, |u| u.uses);
        assert_eq!(forwards, 3, "{what}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "table_struct", &p).expect("module");
        for (input, element) in [(0u32, 0u32), (1, 1), (2, 2), (9, 2)] {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            let e = element_bytes(element);
            assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
            assert_eq!(word(&out.buffers[0], 8), 3, "{what} {input}");
            for k in 0..COUNT {
                assert_eq!(
                    out.buffers[1 + k as usize],
                    element_bytes(k),
                    "{what} {input}: element {k} unchanged"
                );
            }
        }
    }
}

/// MPS's `_MPSNDArrays` shape. The table pointer and a buffer pointer are stored into a private
/// struct beside data, which is `memcpy`'d whole into a second struct and `memmove`'d on into a
/// third: each copy moves the pointer slots, so the table pointer stays parked in forwarding cells
/// (its `u32` offset, never an escape) and `out[1] = tab[k][0]` read through the third struct is
/// exact, as are the buffer pointer and the data words read through the second and third; an
/// out-of-range index is still clamped into the table.
#[test]
fn a_table_pointer_copied_between_cells_is_exact() {
    let body = "  %a = alloca { { ptr, i64 }, ptr addrspace(1), i32 }, align 8\n  %b = alloca { { ptr, i64 }, ptr addrspace(1), i32 }, align 8\n  %d = alloca { { ptr, i64 }, ptr addrspace(1), i32 }, align 8\n  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %a0 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %a, i64 0, i32 0, i32 0\n  store ptr %tp, ptr %a0, align 8\n  %a1 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %a, i64 0, i32 0, i32 1\n  store i64 3, ptr %a1, align 8\n  %a2 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %a, i64 0, i32 1\n  store ptr addrspace(1) %0, ptr %a2, align 8\n  %a3 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %a, i64 0, i32 2\n  store i32 5, ptr %a3, align 8\n  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %b, ptr align 8 %a, i64 32, i1 false)\n  call void @llvm.memmove.p0.p0.i64(ptr align 8 %d, ptr align 8 %b, i64 32, i1 false)\n  %back = load ptr, ptr %d, align 8\n  %el = load ptr addrspace(1), ptr %back, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %b2 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %b, i64 0, i32 1\n  %out = load ptr addrspace(1), ptr %b2, align 8\n  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n  %d1 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %d, i64 0, i32 0, i32 1\n  %n = load i64, ptr %d1, align 8\n  %n32 = trunc i64 %n to i32\n  %d3 = getelementptr inbounds { { ptr, i64 }, ptr addrspace(1), i32 }, ptr %d, i64 0, i32 2\n  %f = load i32, ptr %d3, align 8\n  %nf = add i32 %n32, %f\n  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2\n  store i32 %nf, ptr addrspace(1) %o2, align 4\n";
    let text = format!(
        "{}\ndeclare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\ndeclare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)\n",
        kernel("table_copied", body)
    );
    let Some(bc) = assemble_text("table_copied", &text) else {
        return;
    };
    for (what, p) in profiles() {
        let a = compile(&bc, "table_copied", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let uses = |name: &str| a.strategies.entries.get(name).map_or(0, |u| u.uses);
        assert_eq!(
            (uses("access.cell.store"), uses("access.cell.load")),
            (2, 2),
            "{what}"
        );
        assert_eq!(uses("access.buffer_table.load.clamped"), 1, "{what}");
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "table_copied", &p).expect("module");
        for (input, element) in [(0u32, 0u32), (1, 1), (2, 2), (9, 2)] {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            let e = element_bytes(element);
            assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
            assert_eq!(word(&out.buffers[0], 8), 8, "{what} {input}");
        }
    }
}

/// `{ { ptr, i64 }, ptr addrspace(1) }`: MPS's `_MPSKernelInOut` reduced to a table pointer
/// with its length and a buffer pointer.
const INOUT: &str = "{ { ptr, i64 }, ptr addrspace(1) }";

/// The `inout_phi` kernel: `_MPSKernelInOut` built in two branches and merged by a phi; then
/// `carry` (a statement sequence from `%s` in block `m` that leaves the aggregate in `%c2` and
/// ends in the block `use`) rewrites its length; then `out[1] = tab[..][0]` through the table
/// pointer member and `out[2] = length`.
fn inout_phi(carry: &str) -> String {
    let body = format!(
        "  %k = load i32, ptr addrspace(1) %0, align 4\n  %x = zext i32 %k to i64\n  %odd = trunc i32 %k to i1\n  br i1 %odd, label %a, label %b\n\
         a:\n  %tp = getelementptr inbounds ptr addrspace(1), ptr %1, i64 %x\n  %ra = insertvalue {{ ptr, i64 }} poison, ptr %tp, 0\n  %ra1 = insertvalue {{ ptr, i64 }} %ra, i64 3, 1\n  %sa = insertvalue {INOUT} poison, {{ ptr, i64 }} %ra1, 0\n  %sa1 = insertvalue {INOUT} %sa, ptr addrspace(1) %0, 1\n  br label %m\n\
         b:\n  %sb = insertvalue {INOUT} poison, ptr %1, 0, 0\n  %sb1 = insertvalue {INOUT} %sb, i64 7, 0, 1\n  %sb2 = insertvalue {INOUT} %sb1, ptr addrspace(1) %0, 1\n  br label %m\n\
         m:\n  %s = phi {INOUT} [ %sa1, %a ], [ %sb2, %b ]\n{carry}\
         use:\n  %back = extractvalue {INOUT} %c2, 0, 0\n  %n = extractvalue {INOUT} %c2, 0, 1\n  %out = extractvalue {INOUT} %c2, 1\n  %el = load ptr addrspace(1), ptr %back, align 8\n  %v = load i32, ptr addrspace(1) %el, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1\n  store i32 %v, ptr addrspace(1) %o, align 4\n  %n32 = trunc i64 %n to i32\n  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2\n  store i32 %n32, ptr addrspace(1) %o2, align 4\n  br label %end\n\
         end:\n"
    );
    kernel("inout_phi", &body)
}

/// Compile `inout_phi` with `carry` on `profiles`, check its member extractions (`members`),
/// and run it: `(k, element, length)` for each input `k`.
fn check_inout_phi(
    carry: &str,
    profiles: &[(String, DeviceProfile)],
    members: u32,
    runs: &[(u32, u32, u32)],
) {
    let Some(bc) = assemble_text("inout_phi", &inout_phi(carry)) else {
        return;
    };
    for (what, p) in profiles {
        let a = compile(&bc, "inout_phi", p).unwrap_or_else(|e| panic!("{what}: {e}"));
        let uses = |name: &str| a.strategies.entries.get(name).map_or(0, |u| u.uses);
        assert_eq!(uses("arith.aggregate.member"), members, "{what}");
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "inout_phi", p).expect("module");
        for &(input, element, length) in runs {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            let e = element_bytes(element);
            assert_eq!(word(&out.buffers[0], 4), word(&e, 0), "{what} {input}");
            assert_eq!(word(&out.buffers[0], 8), length, "{what} {input}");
        }
    }
}

/// A phi of pointer aggregates (MPS's `_MPSKernelInOut` rebuilt in two branches and merged,
/// `ndArrayPooling_1dpass`) is a member-wise composite: each edge assembles the incoming insertion
/// chain member by member, and each extraction is a composite extract of the phi's value. A second
/// phi merges the first with a copy whose length an insertion rewrote (that edge's assembly
/// extracts the pointer members from the first phi), and the extractions read it through an
/// insertion chain whose base is that phi. An odd `k` takes `tab[k]` (clamped into the table) with
/// length 3, an even one `tab[0]` with length 7, and `k = 0` adds 2.
#[test]
fn a_phi_of_pointer_aggregates_is_member_wise() {
    let carry = format!(
        "  %zero = icmp eq i32 %k, 0\n  br i1 %zero, label %bump, label %join\n\
         bump:\n  %len = extractvalue {INOUT} %s, 0, 1\n  %len1 = add i64 %len, 2\n  %s2 = insertvalue {INOUT} %s, i64 %len1, 0, 1\n  br label %join\n\
         join:\n  %c = phi {INOUT} [ %s2, %bump ], [ %s, %m ]\n  %c2 = insertvalue {INOUT} %c, ptr addrspace(1) %0, 1\n  br label %use\n"
    );
    check_inout_phi(
        &carry,
        &profiles(),
        3,
        &[(0, 0, 9), (1, 1, 3), (2, 0, 7), (9, 2, 3)],
    );
}

/// The same aggregate carried around a loop whose phi takes the merge's value and its own copy with
/// the length rewritten: the back edge's assembly extracts the pointer members from the loop's phi
/// itself. The loop adds 1 twice.
#[test]
fn a_loop_carried_phi_of_pointer_aggregates_is_member_wise() {
    let carry = format!(
        "  br label %loop\n\
         loop:\n  %i = phi i32 [ 0, %m ], [ %i1, %loop ]\n  %c = phi {INOUT} [ %s, %m ], [ %c1, %loop ]\n  %len = extractvalue {INOUT} %c, 0, 1\n  %len1 = add i64 %len, 1\n  %c1 = insertvalue {INOUT} %c, i64 %len1, 0, 1\n  %i1 = add i32 %i, 1\n  %done = icmp eq i32 %i1, 2\n  br i1 %done, label %exit, label %loop\n\
         exit:\n  %c2 = insertvalue {INOUT} %c1, ptr addrspace(1) %0, 1\n  br label %use\n"
    );
    check_inout_phi(
        &carry,
        &profiles(),
        2,
        &[(0, 0, 9), (1, 1, 5), (2, 0, 9), (9, 2, 5)],
    );
}

/// A phi member that is a null private pointer on one edge has no representation (an arena offset
/// has no null value), so the phi is refused as not implemented, never guessed; with the member
/// undefined instead (a `poison` base), the same kernel compiles.
#[test]
fn a_phi_member_with_a_null_private_pointer_is_refused() {
    let kernel_with = |name: &str, other: &str| {
        let body = format!(
            "  %cell = alloca i32, align 4\n  store i32 7, ptr %cell, align 4\n  %k = load i32, ptr addrspace(1) %0, align 4\n  %odd = trunc i32 %k to i1\n  br i1 %odd, label %a, label %b\n\
             a:\n  %sa = insertvalue {{ ptr, i64 }} poison, ptr %cell, 0\n  %sa1 = insertvalue {{ ptr, i64 }} %sa, i64 3, 1\n  br label %m\n\
             b:\n  %sb = insertvalue {{ ptr, i64 }} poison, ptr {other}, 0\n  %sb1 = insertvalue {{ ptr, i64 }} %sb, i64 5, 1\n  br label %m\n\
             m:\n  %s = phi {{ ptr, i64 }} [ %sa1, %a ], [ %sb1, %b ]\n  %n = extractvalue {{ ptr, i64 }} %s, 1\n  %n32 = trunc i64 %n to i32\n  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n  store i32 %n32, ptr addrspace(1) %o, align 4\n"
        );
        assemble_text(name, &kernel(name, &body))
    };
    let (Some(null), Some(undef)) = (
        kernel_with("member_null", "null"),
        kernel_with("member_undef", "poison"),
    ) else {
        return;
    };
    for (what, p) in profiles() {
        match compile(&null, "member_null", &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Lowering(LowerFamily::Arith), "{what}");
                assert_eq!(u.needs, Needs::NotImplemented, "{what}");
            }
            other => panic!("{what}: expected a refusal, got {other:?}"),
        }
        let a = compile(&undef, "member_undef", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&undef, "member_undef", &p).expect("module");
        for (input, n) in [(1u32, 3u32), (2, 5)] {
            let res = Resources {
                buffers: buffers(input),
                push: push_bytes(&plan),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res)
                .unwrap_or_else(|e| panic!("{what} {input}: {e:?}"));
            assert_eq!(word(&out.buffers[0], 4), n, "{what} {input}");
        }
    }
}
