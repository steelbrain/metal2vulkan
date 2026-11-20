//! Pointer cells end to end.
//!
//! A pointer stored into an alloca whose every access is visible and loaded back is one of the
//! pointers stored at its slot: the cell holds the pointer's lowered representation, so buffer
//! pointers stay static fat pointers (no root table, and Profile F compiles them), private and
//! threadgroup pointers stay arena offsets, and texture handles stay handle indices. Each kernel
//! passes L3 and runs on the CPU interpreter on Profile A and Profile F profiles; an alloca whose
//! pointer slot is also read as data stays an open cell (guest addresses), and a `memcpy` or
//! `memmove` between two cells moves their pointer slots unless it splits one. A volatile cell
//! access is the plain one whose words carry `Volatile`. Each refusal is pinned against a control
//! kernel without the refused construct. A kernel that writes a storage texture compiles under its
//! format on a profile that asks for it (the floor, without formatless storage), as the runtime
//! compiles the variant of the bound texture's format.
//!
//! A slot one store writes, ahead of every read of it, is read as the value stored before the plan
//! sees it (`canon::forward`), and leaves no cell. So each kernel here that reads back a slot it
//! writes once writes it twice, the same value: no one store decides the read, and the cell stays.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::lower::select::Plan;
use crate::metal::PixelFormat;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, compile_given_storage_format, link_and_validate, llvm_assemble, profile, typed_module,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::refuses_threadgroup_memory;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Profile A (device addresses) and Profile F (none), with and without robust descriptors.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lavapipe",
    "vk12-floor",
    "lunarg-desktop-baseline-2023",
];

const THREADS: u32 = 32;

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("profile")
}

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-pointer-cells-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
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

/// Kernel metadata for `name`: `buffers` read-write `device` buffers of `uint`, then the thread's
/// index in its threadgroup.
fn meta(name: &str, buffers: u32) -> String {
    let mut s = format!(
        "!air.kernel = !{{!0}}\n!air.version = !{{!90}}\n!air.language_version = !{{!91}}\n\
         !0 = !{{ptr @{name}, !1, !2}}\n!1 = !{{}}\n!2 = !{{"
    );
    let args: Vec<String> = (0..=buffers).map(|i| format!("!{}", 10 + i)).collect();
    s.push_str(&args.join(", "));
    s.push_str("}\n");
    for i in 0..buffers {
        s.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b{i}\"}}\n",
            10 + i
        ));
    }
    s.push_str(&format!(
        "!{} = !{{i32 {buffers}, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n\
         !90 = !{{i32 2, i32 8, i32 0}}\n!91 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        10 + buffers
    ));
    s
}

/// A kernel `name(b0, .., b{buffers-1}, t)`: `types`, then the function with `body`.
fn kernel(name: &str, buffers: u32, types: &str, body: &str) -> String {
    let params: Vec<String> = (0..buffers)
        .map(|i| format!("ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}"))
        .collect();
    format!(
        "{types}\ndefine void @{name}({}, i32 %t) {{\n{body}}}\n\n{}",
        params.join(", "),
        meta(name, buffers)
    )
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The resources of `plan` with buffer `i` holding `bufs[i]`, each bound at its start.
fn resources(plan: &Plan, bufs: &[Vec<u32>]) -> Resources {
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(THREADS));
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(THREADS));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            // A bound buffer's Metal address: nonzero, so it never compares equal to null.
            (PushField::SlotGuestBase(slot), PushLoc::Push { offset }) => {
                let base = 0x1_0000_0000u64 + u64::from(slot.index) * 0x10_0000;
                push[offset as usize..offset as usize + 8].copy_from_slice(&base.to_le_bytes())
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let mut buffers: Vec<Buffer> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.iter().flat_map(|w| w.to_le_bytes()).collect(),
        })
        .collect();
    // The constant blob, when the function reads it.
    if let Some(b) = &plan.families.access.blob {
        let mut bytes = b.bytes.to_vec();
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        buffers.push(Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
                index: 0,
            }),
            address: None,
            bytes,
        });
    }
    Resources {
        buffers,
        push,
        spec,
    }
}

/// Run in both scheduling orders (which must agree); the words of every buffer.
fn run_both(m: &TypedModule, res: &Resources) -> Vec<Vec<u32>> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        outs.push(out.buffers.iter().map(|b| words(b)).collect::<Vec<_>>());
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

fn count(plan: &Plan, name: &str) -> u32 {
    plan.census().get(name).map_or(0, |(n, _)| *n)
}

/// Compile `name` on every profile that takes it (a function with threadgroup memory on those
/// with zeroed workgroup memory), check it passes L3, reads no root table, and moves its pointers
/// through `cells` cell stores and loads; run it over `bufs` and return each profile's buffers.
fn run_everywhere(
    bc: &[u8],
    name: &str,
    bufs: &[Vec<u32>],
    cells: (u32, u32),
) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        let prof = p(stem);
        let a = match compile(bc, name, &prof) {
            // A profile without zeroed workgroup memory takes no threadgroup memory.
            Err(e) if refuses_threadgroup_memory(&e, &prof) => continue,
            a => a.unwrap_or_else(|e| panic!("{stem}: {e}")),
        };
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(bc, name, &prof).expect("module");
        assert!(
            !plan
                .push
                .fields
                .iter()
                .any(|f| f.field == PushField::RootTableAddress),
            "{stem}: a forwarded pointer needs no root table: {:?}",
            plan.census()
        );
        assert_eq!(
            (
                count(&plan, "access.cell.store"),
                count(&plan, "access.cell.load")
            ),
            cells,
            "{stem}: {:?}",
            plan.census()
        );
        out.push((stem, run_both(&m, &resources(&plan, bufs))));
    }
    out
}

fn input(seed: u32) -> Vec<u32> {
    (0..THREADS)
        .map(|i| seed.wrapping_mul(i + 7) ^ 0x5a5a)
        .collect()
}

/// Two device pointers parked in a private struct (the `access` race program's shape) are read
/// back as the buffers they are: static fat pointers, on Profile F too.
#[test]
fn buffer_pointers_parked_in_a_struct_read_back_statically() {
    let text = kernel(
        "parked",
        2,
        "%S = type { ptr addrspace(1), ptr addrspace(1) }",
        r#"entry:
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr addrspace(1) %b1, ptr %f0, align 8
  store ptr addrspace(1) %b1, ptr %f0, align 8
  %f1 = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr addrspace(1) %b0, ptr %f1, align 8
  store ptr addrspace(1) %b0, ptr %f1, align 8
  %i = zext i32 %t to i64
  %src = load ptr addrspace(1), ptr %f0, align 8
  %sp = getelementptr inbounds i32, ptr addrspace(1) %src, i64 %i
  %v = load i32, ptr addrspace(1) %sp, align 4
  %dst = load ptr addrspace(1), ptr %f1, align 8
  %dp = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %i
  %w = add i32 %v, 1
  store i32 %w, ptr addrspace(1) %dp, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("parked", &text) else {
        return;
    };
    let inp = input(3);
    let want: Vec<u32> = inp.iter().map(|v| v.wrapping_add(1)).collect();
    for (stem, bufs) in run_everywhere(&bc, "parked", &[vec![0; THREADS as usize], inp], (4, 2)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A slot written in three branches, with either of two buffers or with null, reads back as a
/// nullable pointer over both: a store through it goes to the buffer stored, and through null
/// it is dropped (the switch's null arm).
#[test]
fn a_cell_written_with_null_reads_back_nullable() {
    let text = kernel(
        "nullable",
        2,
        "%S = type { i32, ptr addrspace(1) }",
        r#"entry:
  %s = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %even
even:
  %two = and i32 %t, 2
  %second = icmp ne i32 %two, 0
  br i1 %second, label %b, label %n
a:
  store ptr addrspace(1) %b0, ptr %f, align 8
  br label %c
b:
  store ptr addrspace(1) %b1, ptr %f, align 8
  br label %c
n:
  store ptr addrspace(1) null, ptr %f, align 8
  br label %c
c:
  %p = load ptr addrspace(1), ptr %f, align 8
  %i = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  store i32 7, ptr addrspace(1) %q, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("nullable", &text) else {
        return;
    };
    let zero = vec![0; THREADS as usize];
    let want0: Vec<u32> = (0..THREADS)
        .map(|t| if t % 2 == 1 { 7 } else { 0 })
        .collect();
    let want1: Vec<u32> = (0..THREADS)
        .map(|t| if t % 4 == 2 { 7 } else { 0 })
        .collect();
    for (stem, bufs) in run_everywhere(&bc, "nullable", &[zero.clone(), zero], (3, 1)) {
        assert_eq!((&bufs[0], &bufs[1]), (&want0, &want1), "{stem}");
    }
}

/// An array of pointer cells indexed at run time: each load reads any slot of the cell, so its
/// pointer may be either buffer (a switch on the fat pointer's root).
#[test]
fn a_dynamically_indexed_pointer_array_reads_either_buffer() {
    let text = kernel(
        "table",
        3,
        "",
        r#"entry:
  %a = alloca [2 x ptr addrspace(1)], align 8
  %e0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %a, i64 0, i64 0
  store ptr addrspace(1) %b1, ptr %e0, align 8
  %e1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %a, i64 0, i64 1
  store ptr addrspace(1) %b2, ptr %e1, align 8
  %k = and i32 %t, 1
  %kk = zext i32 %k to i64
  %e = getelementptr inbounds [2 x ptr addrspace(1)], ptr %a, i64 0, i64 %kk
  %src = load ptr addrspace(1), ptr %e, align 8
  %i = zext i32 %t to i64
  %sp = getelementptr inbounds i32, ptr addrspace(1) %src, i64 %i
  %v = load i32, ptr addrspace(1) %sp, align 4
  %dp = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %v, ptr addrspace(1) %dp, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("table", &text) else {
        return;
    };
    let (x, y) = (input(5), input(11));
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| if t % 2 == 1 { y[t] } else { x[t] })
        .collect();
    for (stem, bufs) in run_everywhere(&bc, "table", &[vec![0; THREADS as usize], x, y], (2, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A private pointer parked next to data in a struct (the SceneKit lighting-contribution shape)
/// reads back as its arena offset; the array it points to is written directly and read through
/// the loaded pointer.
#[test]
fn a_private_pointer_parked_in_a_struct_reads_back_its_array() {
    let text = kernel(
        "private",
        1,
        "%S = type { <4 x float>, float, ptr }",
        r#"entry:
  %arr = alloca [4 x i32], align 4
  %s = alloca %S, align 16
  %d = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store float 1.5, ptr %d, align 16
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 2
  store ptr %arr, ptr %f, align 8
  store ptr %arr, ptr %f, align 8
  %a0 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 0
  store i32 %t, ptr %a0, align 4
  %a1 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 1
  %t2 = mul i32 %t, 2
  store i32 %t2, ptr %a1, align 4
  %a2 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 2
  %t3 = mul i32 %t, 3
  store i32 %t3, ptr %a2, align 4
  %a3 = getelementptr inbounds [4 x i32], ptr %arr, i64 0, i64 3
  %t4 = mul i32 %t, 4
  store i32 %t4, ptr %a3, align 4
  %p = load ptr, ptr %f, align 8
  %k = and i32 %t, 3
  %kk = zext i32 %k to i64
  %q = getelementptr inbounds i32, ptr %p, i64 %kk
  %v = load i32, ptr %q, align 4
  %x = load float, ptr %d, align 16
  %xi = fptoui float %x to i32
  %r = add i32 %v, %xi
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("private", &text) else {
        return;
    };
    let want: Vec<u32> = (0..THREADS).map(|t| t * ((t & 3) + 1) + 1).collect();
    for (stem, bufs) in run_everywhere(&bc, "private", &[vec![0; THREADS as usize]], (2, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A threadgroup global's address parked in a cell is its arena offset (a constant store).
#[test]
fn a_threadgroup_address_parked_in_a_cell_is_its_arena_offset() {
    let text = kernel(
        "shared",
        1,
        "@pad = internal addrspace(3) global [8 x i32] undef, align 4\n\
         @g = internal addrspace(3) global [32 x i32] undef, align 4",
        r#"entry:
  %s = alloca ptr addrspace(3), align 8
  store ptr addrspace(3) @g, ptr %s, align 8
  store ptr addrspace(3) @g, ptr %s, align 8
  %i = zext i32 %t to i64
  %pad = getelementptr inbounds [8 x i32], ptr addrspace(3) @pad, i64 0, i64 0
  store i32 9, ptr addrspace(3) %pad, align 4
  %p = load ptr addrspace(3), ptr %s, align 8
  %q = getelementptr inbounds i32, ptr addrspace(3) %p, i64 %i
  %v = mul i32 %t, 5
  store i32 %v, ptr addrspace(3) %q, align 4
  %d = getelementptr inbounds [32 x i32], ptr addrspace(3) @g, i64 0, i64 %i
  %w = load i32, ptr addrspace(3) %d, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("shared", &text) else {
        return;
    };
    let want: Vec<u32> = (0..THREADS).map(|t| t * 5).collect();
    for (stem, bufs) in run_everywhere(&bc, "shared", &[vec![0; THREADS as usize]], (2, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A texture handle parked in a struct (the `median3x3` helper shape) reads back as its handle:
/// one candidate, so the write binds the texture directly.
#[test]
fn a_texture_handle_parked_in_a_struct_reads_back_as_its_handle() {
    let text = r#"%T = type { ptr addrspace(1) }
%H = type <{ %T, <2 x i16>, i8, [3 x i8] }>
define void @tex(ptr addrspace(1) %tex, <2 x i16> %pos) {
entry:
  %h = alloca %H, align 8
  %f = getelementptr inbounds %H, ptr %h, i64 0, i32 0, i32 0
  store ptr addrspace(1) %tex, ptr %f, align 8
  store ptr addrspace(1) %tex, ptr %f, align 8
  %g = getelementptr inbounds %H, ptr %h, i64 0, i32 1
  store <2 x i16> %pos, ptr %g, align 8
  %c = load <2 x i16>, ptr %g, align 8
  %u = load ptr addrspace(1), ptr %f, align 8
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %u, <2 x i16> %c, <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, i16 0, i32 2)
  ret void
}
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @tex, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("tex", text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile_given_storage_format(&bc, "tex", &prof, PixelFormat::RGBA16Float)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let uses = |n: &str| a.strategies.entries.get(n).map_or(0, |u| u.uses);
        assert_eq!(
            (uses("access.cell.store"), uses("access.cell.load")),
            (2, 1),
            "{stem}"
        );
        assert!(
            a.strategies.entries.keys().all(|k| !k.contains("heap")),
            "{stem}: a static handle: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
    }
}

/// A pointer slot also read as an integer is an open cell: its pointer is a guest address, so
/// Profile A reads the root table and Profile F refuses it.
#[test]
fn a_pointer_slot_read_as_data_stays_an_open_cell() {
    let text = kernel(
        "punned",
        2,
        "",
        r#"entry:
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %b1, ptr %s, align 8
  store ptr addrspace(1) %b1, ptr %s, align 8
  %bits = load i64, ptr %s, align 8
  %lo = trunc i64 %bits to i32
  %p = load ptr addrspace(1), ptr %s, align 8
  %i = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %r = xor i32 %v, %lo
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("punned", &text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match typed_module(&bc, "punned", &prof) {
            Ok((plan, _)) => {
                assert!(
                    prof.fields().buffer_access == crate::device::profile::BufferAccess::Addressed
                );
                assert_eq!(count(&plan, "access.cell.load"), 0, "{stem}");
                assert!(
                    plan.push
                        .fields
                        .iter()
                        .any(|f| f.field == PushField::RootTableAddress),
                    "{stem}: a guest address resolves through the root table"
                );
            }
            Err(Error::Unsupported(u)) => {
                assert!(
                    matches!(
                        u.feature,
                        Feature::DereferenceLoadedPointer | Feature::PointerToInteger
                    ),
                    "{stem}: {u:?}"
                );
            }
            Err(e) => panic!("{stem}: {e}"),
        }
    }
}

/// A `memcpy` into the data beside a pointer slot is a data access that keeps clear of the slot:
/// the cell still forwards, and the copied words read back. (A data access at an offset that is
/// not a translation-time constant could reach the slot's bytes, and opens the cell.)
#[test]
fn a_copy_beside_a_pointer_slot_keeps_the_cell() {
    let text = kernel(
        "copied",
        2,
        "%S = type { [4 x i32], ptr addrspace(1) }\n\
         declare void @llvm.memcpy.p0.p1.i64(ptr, ptr addrspace(1), i64, i1)",
        r#"entry:
  %s = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr addrspace(1) %b1, ptr %f, align 8
  store ptr addrspace(1) %b1, ptr %f, align 8
  %d = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  call void @llvm.memcpy.p0.p1.i64(ptr align 8 %d, ptr addrspace(1) align 4 %b1, i64 16, i1 false)
  %i = zext i32 %t to i64
  %dk = getelementptr inbounds [4 x i32], ptr %d, i64 0, i64 2
  %x = load i32, ptr %dk, align 4
  %p = load ptr addrspace(1), ptr %f, align 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %r = add i32 %v, %x
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("copied", &text) else {
        return;
    };
    let inp = input(5);
    let want: Vec<u32> = inp.iter().map(|v| v.wrapping_add(inp[2])).collect();
    for (stem, bufs) in run_everywhere(&bc, "copied", &[vec![0; THREADS as usize], inp], (2, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A `memcpy` of a whole struct between two cells is a slot copy: it moves the pointer slot to the
/// same place in the copy, so both cells forward, and the pointer read from the copy is the buffer
/// stored into the original (a static fat pointer, on Profile F too). A `memmove` onward to a third
/// cell carries it further, and the data beside the slot is copied as data.
#[test]
fn a_copy_of_a_whole_struct_moves_its_pointer_slot() {
    let text = kernel(
        "moved",
        2,
        "%S = type { ptr addrspace(1), i32 }\n\
         declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n\
         declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)",
        r#"entry:
  %s = alloca %S, align 8
  %c = alloca %S, align 8
  %d = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr addrspace(1) %b1, ptr %f, align 8
  %n = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store i32 %t, ptr %n, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c, ptr align 8 %s, i64 16, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %d, ptr align 8 %c, i64 16, i1 false)
  %p = load ptr addrspace(1), ptr %d, align 8
  %dn = getelementptr inbounds %S, ptr %d, i64 0, i32 1
  %k = load i32, ptr %dn, align 8
  %i = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %r = add i32 %v, %k
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("moved", &text) else {
        return;
    };
    let inp = input(7);
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| inp[t].wrapping_add(t as u32))
        .collect();
    for (stem, bufs) in run_everywhere(&bc, "moved", &[vec![0; THREADS as usize], inp], (1, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A `memcpy` that splits a pointer slot (its bytes end inside the slot) copies part of the
/// pointer's bytes as data: both cells are open, so the pointer read from the copy is a guest
/// address, which Profile A resolves through the root table and Profile F refuses.
#[test]
fn a_copy_that_splits_a_pointer_slot_opens_both_cells() {
    let text = kernel(
        "split",
        2,
        "%S = type { i32, ptr addrspace(1) }\n\
         declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)",
        r#"entry:
  %s = alloca %S, align 8
  %c = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr addrspace(1) %b1, ptr %f, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c, ptr align 8 %s, i64 12, i1 false)
  %g = getelementptr inbounds %S, ptr %c, i64 0, i32 1
  %p = load ptr addrspace(1), ptr %g, align 8
  %i = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("split", &text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match typed_module(&bc, "split", &prof) {
            Ok((plan, _)) => {
                assert!(
                    prof.fields().buffer_access == crate::device::profile::BufferAccess::Addressed
                );
                assert_eq!(
                    (
                        count(&plan, "access.cell.store"),
                        count(&plan, "access.cell.load")
                    ),
                    (0, 0),
                    "{stem}"
                );
                assert!(
                    plan.push
                        .fields
                        .iter()
                        .any(|f| f.field == PushField::RootTableAddress),
                    "{stem}: a guest address resolves through the root table"
                );
            }
            Err(Error::Unsupported(u)) => {
                assert!(
                    matches!(
                        u.feature,
                        Feature::DereferenceLoadedPointer | Feature::PointerToInteger
                    ),
                    "{stem}: {u:?}"
                );
            }
            Err(e) => panic!("{stem}: {e}"),
        }
    }
}

// ---- Refusals ----

/// `b0[t] = zext %n` (the compare result `%n`), then `ret`.
const STORE_FLAG: &str = "  %z = zext i1 %n to i32
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %z, ptr addrspace(1) %o, align 4
  ret void
";

/// The kernel `name` with `body` is refused as `Lowering(Access)`, `NotImplemented` on every
/// profile, while the same kernel with `control` (the body without the refused construct)
/// compiles and passes L3, so the refusal is that construct's.
fn refused_unlike(name: &str, types: &str, body: &str, control: &str) {
    let Some(bc) = assemble(name, &kernel(name, 1, types, body)) else {
        return;
    };
    let Some(ok) = assemble(name, &kernel(name, 1, types, control)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match compile(&bc, name, &prof) {
            Err(Error::Unsupported(u)) => {
                assert!(
                    u.feature == Feature::Lowering(LowerFamily::Access)
                        && u.needs == Needs::NotImplemented,
                    "{stem} {name}: {u:?}"
                );
            }
            other => panic!(
                "{stem} {name}: expected a refusal, got {:?}",
                other.map(|_| ())
            ),
        }
        let a = compile(&ok, name, &prof).unwrap_or_else(|e| panic!("{stem} {name} control: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem} {name} control: L3: {e}"));
    }
}

/// A private cell written with null on one path and an array's address on the other: the
/// pointer read back may be null.
const NULL_OR_ARRAY: &str = "entry:
  %x = alloca [4 x i32], align 4
  %s = alloca ptr, align 8
  store ptr null, ptr %s, align 8
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %c
a:
  store ptr %x, ptr %s, align 8
  br label %c
c:
  %p = load ptr, ptr %s, align 8
";

/// A pointer that may be null compared with an object and with null. The private arena keeps its
/// first word for null, so the array lies past offset 0 and null is offset 0: each compare is of
/// offsets and answers as Metal's (odd threads read the array's address back, even ones null).
#[test]
fn a_possibly_null_arena_pointer_compares_by_offset() {
    for (name, rhs, odd) in [
        ("nullable_vs_object", "%x", 1),
        ("nullable_vs_null", "null", 0),
    ] {
        let body = format!("{NULL_OR_ARRAY}  %n = icmp eq ptr %p, {rhs}\n{STORE_FLAG}");
        let Some(bc) = assemble(name, &kernel(name, 1, "", &body)) else {
            return;
        };
        let want: Vec<u32> = (0..THREADS)
            .map(|t| if t % 2 == 1 { odd } else { 1 - odd })
            .collect();
        for (stem, bufs) in run_everywhere(&bc, name, &[vec![0; THREADS as usize]], (2, 1)) {
            assert_eq!(bufs[0], want, "{stem} {name}");
        }
    }
}

/// A signed compare with null depends on the address's sign, which the arena offset does not carry:
/// refused. The unsigned compare folds.
#[test]
fn a_signed_compare_with_null_is_refused() {
    let body = format!(
        "entry:\n  %x = alloca [4 x i32], align 4\n  %n = icmp sgt ptr %x, null\n{STORE_FLAG}"
    );
    let control = body.replace("icmp sgt", "icmp ugt");
    refused_unlike("signed_null", "", &body, &control);
}

/// The kernel `name` with `body` compiles and passes L3 on every profile.
fn compiles(name: &str, types: &str, body: &str) {
    let Some(bc) = assemble(name, &kernel(name, 1, types, body)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile(&bc, name, &prof).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem} {name}: L3: {e}"));
    }
}

/// An undefined pointer compared with null names no home: refused (a pointer to an object compared
/// with null folds). Null compared with null never reaches the access family: constant propagation
/// folds it.
#[test]
fn an_undefined_pointer_compared_with_null_is_refused() {
    let body = format!(
        "entry:\n  %x = alloca [4 x i32], align 4\n  %n = icmp eq ptr undef, null\n{STORE_FLAG}"
    );
    let control = body.replace("icmp eq ptr undef, null", "icmp eq ptr %x, null");
    refused_unlike("undef_null", "", &body, &control);
    compiles(
        "null_null",
        "",
        &body.replace("icmp eq ptr undef, null", "icmp ult ptr null, null"),
    );
}

/// Two objects compared are both placed in the private arena (a cross-root compare keeps them out
/// of the typed form), so the compare is one home's offsets, and distinct objects are unequal. The
/// only other way to meet two homes in one space, a threadgroup address cast into the thread space,
/// is refused at the cast.
#[test]
fn compared_objects_share_one_home() {
    let body = format!(
        "entry:\n  %x = alloca [4 x i32], align 4\n  %y = alloca [4 x i32], align 4\n  \
         store i32 %t, ptr %x, align 4\n  store i32 %t, ptr %y, align 4\n  \
         %n = icmp eq ptr %y, %x\n{STORE_FLAG}"
    );
    let Some(bc) = assemble("two_objects", &kernel("two_objects", 1, "", &body)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile(&bc, "two_objects", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "two_objects", &prof).expect("module");
        assert_eq!(count(&plan, "access.private.pointer_compare"), 1, "{stem}");
        assert_eq!(count(&plan, "access.typed.alloca"), 0, "{stem}");
        let bufs = run_both(&m, &resources(&plan, &[vec![7; THREADS as usize]]));
        assert_eq!(bufs[0], vec![0; THREADS as usize], "{stem}");
    }
    let types = "@g = internal addrspace(3) global [4 x i32] undef, align 4";
    let cast = body.replace(
        "  %n = icmp eq ptr %y, %x\n",
        "  %gc = addrspacecast ptr addrspace(3) @g to ptr\n  %n = icmp eq ptr %gc, %x\n",
    );
    refused_unlike("cast_home", types, &cast, &body);
}

/// A device pointer parked in a private cell and read back through `b0[t]`.
const PARKED_B0: &str = "entry:
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %b0, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  store i32 1, ptr addrspace(1) %o, align 4
  ret void
";

/// A volatile store or load of a cell slot is the plain slot access whose words carry `Volatile`;
/// the pointer still forwards (the slot is the invocation's own).
#[test]
fn volatile_cell_accesses_carry_volatile() {
    use crate::spirv::ir::Inst;
    for (name, from, to) in [
        (
            "volatile_cell_store",
            "store ptr addrspace(1) %b0, ptr %s, align 8",
            "store volatile ptr addrspace(1) %b0, ptr %s, align 8",
        ),
        (
            "volatile_cell_load",
            "%p = load ptr addrspace(1), ptr %s, align 8",
            "%p = load volatile ptr addrspace(1), ptr %s, align 8",
        ),
    ] {
        let text = kernel(name, 1, "", &PARKED_B0.replace(from, to));
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for (stem, bufs) in run_everywhere(&bc, name, &[vec![0; THREADS as usize]], (1, 1)) {
            assert_eq!(bufs[0], vec![1; THREADS as usize], "{stem} {name}");
            let (_, m) = typed_module(&bc, name, &p(stem)).expect("module");
            let volatile = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|n| {
                    matches!(&n.inst, Inst::Load { access, .. } | Inst::Store { access, .. }
                        if access.volatile)
                })
                .count();
            assert!(volatile >= 1, "{stem} {name}");
        }
    }
}

/// An atomic access of a cell slot is refused (the plain one forwards).
#[test]
fn atomic_cell_accesses_are_refused() {
    refused_unlike(
        "atomic_cell_store",
        "",
        &PARKED_B0.replace(
            "store ptr addrspace(1) %b0, ptr %s, align 8",
            "store atomic ptr addrspace(1) %b0, ptr %s monotonic, align 8",
        ),
        PARKED_B0,
    );
}

/// A constant array the kernel reads through the pointer `%q` to element `t & 3`.
const TABLE: &str =
    "@k = internal addrspace(2) constant [4 x i32] [i32 3, i32 5, i32 7, i32 11], align 4";

/// The values of [`TABLE`].
const TABLE_WORDS: [u32; 4] = [3, 5, 7, 11];

/// The kernel `name(b0, b1, t)` of [`kernel`] with `b1` a `constant uint*` buffer instead of a
/// `device` one.
fn constant_b1_kernel(name: &str, types: &str, body: &str) -> String {
    kernel(name, 2, types, body)
        .replace(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %b1",
            "ptr addrspace(2) noundef \"air-buffer-no-alias\" %b1",
        )
        .replace(
            "!11 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1,",
            "!11 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2,",
        )
}

/// A pointer into the constant blob parked in a cell whose loads are all blob pointers is its blob
/// offset there, a constant global's address (a stored global leaves the `Private` form) as well as
/// a computed element pointer.
#[test]
fn a_blob_pointer_parked_in_a_cell_is_its_offset() {
    let text = kernel(
        "blob_cell",
        1,
        &format!("{TABLE}\n%S = type {{ ptr addrspace(2), ptr addrspace(2) }}"),
        r#"entry:
  %i = zext i32 %t to i64
  %j = and i64 %i, 3
  %q = getelementptr inbounds [4 x i32], ptr addrspace(2) @k, i64 0, i64 %j
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr addrspace(2) %q, ptr %f0, align 8
  store ptr addrspace(2) %q, ptr %f0, align 8
  %f1 = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr addrspace(2) @k, ptr %f1, align 8
  store ptr addrspace(2) @k, ptr %f1, align 8
  %r = load ptr addrspace(2), ptr %f0, align 8
  %v = load i32, ptr addrspace(2) %r, align 4
  %g = load ptr addrspace(2), ptr %f1, align 8
  %g3 = getelementptr inbounds i32, ptr addrspace(2) %g, i64 3
  %w = load i32, ptr addrspace(2) %g3, align 4
  %x = add i32 %v, %w
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("blob_cell", &text) else {
        return;
    };
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| TABLE_WORDS[t & 3] + TABLE_WORDS[3])
        .collect();
    for (stem, bufs) in run_everywhere(&bc, "blob_cell", &[vec![0; THREADS as usize]], (4, 2)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A slot written with a constant global's address on odd threads and a `constant` buffer on even
/// ones reads back a static fat pointer over both, which names the blob `BLOB_ROOT`: the blob
/// address is stored as `(BLOB_ROOT, offset)`. A `memcpy` of the cell onward (a slot copy) carries
/// it to the copy's loads, which read the same pair.
#[test]
fn a_blob_pointer_parked_beside_a_buffer_is_a_fat_pointer() {
    let text = constant_b1_kernel(
        "blob_or_buffer",
        &format!("{TABLE}\ndeclare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)"),
        r#"entry:
  %s = alloca ptr addrspace(2), align 8
  %c = alloca ptr addrspace(2), align 8
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %b
a:
  store ptr addrspace(2) @k, ptr %s, align 8
  br label %m
b:
  store ptr addrspace(2) %b1, ptr %s, align 8
  br label %m
m:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c, ptr align 8 %s, i64 8, i1 false)
  %p = load ptr addrspace(2), ptr %c, align 8
  %i = zext i32 %t to i64
  %j = and i64 %i, 3
  %q = getelementptr inbounds i32, ptr addrspace(2) %p, i64 %j
  %v = load i32, ptr addrspace(2) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("blob_or_buffer", &text) else {
        return;
    };
    let inp = input(13);
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| {
            if t % 2 == 1 {
                TABLE_WORDS[t & 3]
            } else {
                inp[t & 3]
            }
        })
        .collect();
    let bufs = [vec![0; THREADS as usize], inp.clone()];
    for (stem, out) in run_everywhere(&bc, "blob_or_buffer", &bufs, (2, 1)) {
        assert_eq!(out[0], want, "{stem}");
    }
}

/// A slot one load reads as a blob pointer (its exact slot holds only the constant global's
/// address) and another as a fat pointer (it reads either slot, one of them a buffer's) holds no
/// one encoding, so its stores are refused. With the second load alone, the blob address is stored
/// as the fat pointer `(BLOB_ROOT, offset)`.
#[test]
fn a_blob_slot_read_in_two_encodings_is_refused() {
    let both = r#"entry:
  %a = alloca [2 x ptr addrspace(2)], align 8
  %e0 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %a, i64 0, i64 0
  store ptr addrspace(2) @k, ptr %e0, align 8
  %e1 = getelementptr inbounds [2 x ptr addrspace(2)], ptr %a, i64 0, i64 1
  store ptr addrspace(2) %b1, ptr %e1, align 8
  %i = zext i32 %t to i64
  %h = and i64 %i, 1
  %e = getelementptr inbounds [2 x ptr addrspace(2)], ptr %a, i64 0, i64 %h
  %p = load ptr addrspace(2), ptr %e, align 8
  %v = load i32, ptr addrspace(2) %p, align 4
  %z = load ptr addrspace(2), ptr %e0, align 8
  %w = load i32, ptr addrspace(2) %z, align 4
  %x = add i32 %v, %w
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#;
    let one = both
        .replace("  %z = load ptr addrspace(2), ptr %e0, align 8\n", "")
        .replace("  %w = load i32, ptr addrspace(2) %z, align 4\n", "")
        .replace("%x = add i32 %v, %w", "%x = add i32 %v, 0");
    let Some(bc) = assemble(
        "two_encodings",
        &constant_b1_kernel("two_encodings", TABLE, both),
    ) else {
        return;
    };
    let Some(ok) = assemble(
        "two_encodings",
        &constant_b1_kernel("two_encodings", TABLE, &one),
    ) else {
        return;
    };
    let inp = input(17);
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| if t % 2 == 1 { inp[0] } else { TABLE_WORDS[0] })
        .collect();
    for stem in PROFILES {
        let prof = p(stem);
        match compile(&bc, "two_encodings", &prof) {
            Err(Error::Unsupported(u)) => assert!(
                u.feature == Feature::Lowering(LowerFamily::Access)
                    && u.needs == Needs::NotImplemented,
                "{stem}: {u:?}"
            ),
            other => panic!("{stem}: expected a refusal, got {:?}", other.map(|_| ())),
        }
        let a = compile(&ok, "two_encodings", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(&ok, "two_encodings", &prof).expect("module");
        let bufs = run_both(
            &m,
            &resources(&plan, &[vec![0; THREADS as usize], inp.clone()]),
        );
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A `select` of private arena pointers on an undefined condition is refused; on a thread's bit it
/// selects the offsets. (A literal `true` or `false` condition never reaches the access family:
/// constant propagation folds the select.)
#[test]
fn a_select_on_an_undefined_condition_is_refused() {
    let body = "entry:
  %x = alloca [4 x i32], align 4
  %y = alloca [4 x i32], align 4
  %odd = trunc i32 %t to i1
  %s = select i1 undef, ptr %x, ptr %y
  store i32 %t, ptr %s, align 4
  %v = load i32, ptr %x, align 4
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
";
    refused_unlike(
        "undef_select",
        "",
        body,
        &body.replace("select i1 undef", "select i1 %odd"),
    );
    refused_unlike(
        "poison_select",
        "",
        &body.replace("select i1 undef", "select i1 poison"),
        &body.replace("select i1 undef", "select i1 %odd"),
    );
    compiles(
        "true_select",
        "",
        &body.replace("select i1 undef", "select i1 true"),
    );
}

/// A store through a pointer into two cells joins their slot groups. Cell `%a` alone holds the
/// constant global's address (its load reads a blob offset) and cell `%b` also a `constant` buffer
/// (its load reads a fat pointer): the store of `@k` through a `select` of both meets both loads,
/// so its group reads two encodings and the store is refused. Without that store each cell is its
/// own group, and both compile.
#[test]
fn a_store_into_two_cells_joins_their_groups() {
    let body = r#"entry:
  %a = alloca ptr addrspace(2), align 8
  %b = alloca ptr addrspace(2), align 8
  store ptr addrspace(2) @k, ptr %a, align 8
  store ptr addrspace(2) %b1, ptr %b, align 8
  %odd = trunc i32 %t to i1
  %s = select i1 %odd, ptr %a, ptr %b
  store ptr addrspace(2) @k, ptr %s, align 8
  %p = load ptr addrspace(2), ptr %a, align 8
  %v = load i32, ptr addrspace(2) %p, align 4
  %q = load ptr addrspace(2), ptr %b, align 8
  %w = load i32, ptr addrspace(2) %q, align 4
  %x = add i32 %v, %w
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#;
    let control = body.replace("  store ptr addrspace(2) @k, ptr %s, align 8\n", "");
    let Some(bc) = assemble(
        "joined_groups",
        &constant_b1_kernel("joined_groups", TABLE, body),
    ) else {
        return;
    };
    let Some(ok) = assemble(
        "joined_groups",
        &constant_b1_kernel("joined_groups", TABLE, &control),
    ) else {
        return;
    };
    let inp = input(19);
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|_| TABLE_WORDS[0].wrapping_add(inp[0]))
        .collect();
    for stem in PROFILES {
        let prof = p(stem);
        match compile(&bc, "joined_groups", &prof) {
            Err(Error::Unsupported(u)) => assert!(
                u.feature == Feature::Lowering(LowerFamily::Access)
                    && u.needs == Needs::NotImplemented,
                "{stem}: {u:?}"
            ),
            other => panic!("{stem}: expected a refusal, got {:?}", other.map(|_| ())),
        }
        let a = compile(&ok, "joined_groups", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, m) = typed_module(&ok, "joined_groups", &prof).expect("module");
        let bufs = run_both(
            &m,
            &resources(&plan, &[vec![0; THREADS as usize], inp.clone()]),
        );
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// A computed pointer into the constant blob (an element at a run-time index) stored where static
/// fat pointers read it is the pair `(BLOB_ROOT, offset)` with the computed offset, not only a
/// constant global's.
#[test]
fn a_computed_blob_pointer_parked_beside_a_buffer_is_a_fat_pointer() {
    let text = constant_b1_kernel(
        "computed_blob_or_buffer",
        TABLE,
        r#"entry:
  %s = alloca ptr addrspace(2), align 8
  %i = zext i32 %t to i64
  %j = and i64 %i, 3
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %b
a:
  %q = getelementptr inbounds [4 x i32], ptr addrspace(2) @k, i64 0, i64 %j
  store ptr addrspace(2) %q, ptr %s, align 8
  br label %m
b:
  %r = getelementptr inbounds i32, ptr addrspace(2) %b1, i64 %j
  store ptr addrspace(2) %r, ptr %s, align 8
  br label %m
m:
  %p = load ptr addrspace(2), ptr %s, align 8
  %v = load i32, ptr addrspace(2) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("computed_blob_or_buffer", &text) else {
        return;
    };
    let inp = input(23);
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| {
            if t % 2 == 1 {
                TABLE_WORDS[t & 3]
            } else {
                inp[t & 3]
            }
        })
        .collect();
    let bufs = [vec![0; THREADS as usize], inp.clone()];
    for (stem, out) in run_everywhere(&bc, "computed_blob_or_buffer", &bufs, (2, 1)) {
        assert_eq!(out[0], want, "{stem}");
    }
}

/// A slot written with a constant global's address on odd threads and with null on even ones reads
/// back a static fat pointer over the blob and null: the address is stored as `(BLOB_ROOT, offset)`
/// and null as zeros. The odd threads read the table through it; the even ones never dereference
/// it.
#[test]
fn a_blob_pointer_or_null_parked_in_a_cell_is_a_fat_pointer() {
    let text = kernel(
        "blob_or_null",
        1,
        TABLE,
        r#"entry:
  %s = alloca ptr addrspace(2), align 8
  %i = zext i32 %t to i64
  %j = and i64 %i, 3
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %b
a:
  store ptr addrspace(2) @k, ptr %s, align 8
  br label %m
b:
  store ptr addrspace(2) null, ptr %s, align 8
  br label %m
m:
  %p = load ptr addrspace(2), ptr %s, align 8
  br i1 %odd, label %d, label %z
d:
  %q = getelementptr inbounds i32, ptr addrspace(2) %p, i64 %j
  %v = load i32, ptr addrspace(2) %q, align 4
  br label %z
z:
  %x = phi i32 [ %v, %d ], [ 99, %m ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("blob_or_null", &text) else {
        return;
    };
    let want: Vec<u32> = (0..THREADS as usize)
        .map(|t| if t % 2 == 1 { TABLE_WORDS[t & 3] } else { 99 })
        .collect();
    for (stem, out) in run_everywhere(&bc, "blob_or_null", &[vec![0; THREADS as usize]], (2, 1)) {
        assert_eq!(out[0], want, "{stem}");
    }
}

/// A pointer's bits stored into a cell (a `ptrtoint` of it whose every use is the value of a store
/// into a forwarding cell) are a store of the pointer: the slot holds its lowered representation, a
/// pointer load reads the buffer back statically on Profile A and Profile F, and the integer is
/// never formed.
#[test]
fn a_pointer_stored_as_its_bits_reads_back_as_the_pointer() {
    let text = kernel(
        "bits",
        2,
        "%S = type { i32, ptr addrspace(1) }",
        r#"entry:
  %s = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  %bits = ptrtoint ptr addrspace(1) %b1 to i64
  store i64 %bits, ptr %f, align 8
  %i = zext i32 %t to i64
  %p = load ptr addrspace(1), ptr %f, align 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %w = add i32 %v, 3
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("bits", &text) else {
        return;
    };
    let inp = input(5);
    let want: Vec<u32> = inp.iter().map(|v| v.wrapping_add(3)).collect();
    for (stem, bufs) in run_everywhere(&bc, "bits", &[vec![0; THREADS as usize], inp], (1, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
    for stem in PROFILES {
        let (plan, _) = typed_module(&bc, "bits", &p(stem)).expect("module");
        assert_eq!(count(&plan, "access.cell.bits"), 1, "{stem}");
    }
}

/// A texture's bits stored into a struct field (RealityKit's and SceneKit's shape) are a store of
/// its handle: read back as a pointer, the write binds the texture, and the integer is never formed
/// (it was a texture `ptrtoint`, refused). Never read back, the store goes with what nothing
/// observes.
#[test]
fn a_texture_stored_as_its_bits_is_its_handle() {
    let text = r#"%T = type { ptr addrspace(1) }
%H = type { <2 x i16>, %T, %T }
define void @texbits(ptr addrspace(1) %tex, <2 x i16> %pos) {
entry:
  %h = alloca %H, align 8
  %f = getelementptr inbounds %H, ptr %h, i64 0, i32 1, i32 0
  %bits = ptrtoint ptr addrspace(1) %tex to i64
  store i64 %bits, ptr %f, align 8
  %dead = getelementptr inbounds %H, ptr %h, i64 0, i32 2, i32 0
  %again = ptrtoint ptr addrspace(1) %tex to i64
  store i64 %again, ptr %dead, align 8
  %u = load ptr addrspace(1), ptr %f, align 8
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %u, <2 x i16> %pos, <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, i16 0, i32 2)
  ret void
}
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @texbits, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("texbits", text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile_given_storage_format(&bc, "texbits", &prof, PixelFormat::RGBA16Float)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let uses = |n: &str| a.strategies.entries.get(n).map_or(0, |u| u.uses);
        assert_eq!(
            (
                uses("access.cell.bits"),
                uses("access.cell.store"),
                uses("access.cell.load")
            ),
            (1, 1, 1),
            "{stem}"
        );
        assert!(
            a.strategies.entries.keys().all(|k| !k.contains("heap")),
            "{stem}: a static handle: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
    }
}

/// The control: pointer bits that are also observed as an integer (here xor-ed into the output) are
/// an integer, so the slot is data and the cell stays open: Profile A reads the root table and
/// Profile F refuses.
#[test]
fn pointer_bits_observed_as_an_integer_stay_an_integer() {
    let text = kernel(
        "seen_bits",
        2,
        "%S = type { i32, ptr addrspace(1) }",
        r#"entry:
  %s = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  %bits = ptrtoint ptr addrspace(1) %b1 to i64
  store i64 %bits, ptr %f, align 8
  %lo = trunc i64 %bits to i32
  %i = zext i32 %t to i64
  %p = load ptr addrspace(1), ptr %f, align 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %w = xor i32 %v, %lo
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("seen_bits", &text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match typed_module(&bc, "seen_bits", &prof) {
            Ok((plan, _)) => {
                assert_eq!(
                    (
                        count(&plan, "access.cell.bits"),
                        count(&plan, "access.cell.load")
                    ),
                    (0, 0),
                    "{stem}"
                );
                assert!(
                    plan.push
                        .fields
                        .iter()
                        .any(|f| f.field == PushField::RootTableAddress),
                    "{stem}: a guest address resolves through the root table"
                );
            }
            Err(Error::Unsupported(u)) => assert!(
                matches!(
                    u.feature,
                    Feature::DereferenceLoadedPointer | Feature::PointerToInteger
                ),
                "{stem}: {u:?}"
            ),
            Err(e) => panic!("{stem}: {e}"),
        }
    }
}

/// An integer load of a pointer slot whose every use is an `inttoptr` back to a pointer (the round
/// trip RealityKit's `fs*UV` shaders take) reads the slot's pointer: the load holds its lowered
/// representation and the `inttoptr` is that pointer, here read in another block. With the slot
/// written through the pointer's own bits, no integer remains at all.
#[test]
fn a_pointer_read_back_through_its_bits_is_the_pointer() {
    for (name, store) in [
        (
            "word",
            "  store ptr addrspace(1) %b1, ptr %f, align 8\n  store ptr addrspace(1) %b1, ptr %f, \
             align 8\n",
        ),
        (
            "word_of_bits",
            "  %bits = ptrtoint ptr addrspace(1) %b1 to i64\n  store i64 %bits, ptr %f, align 8\n  \
             store i64 %bits, ptr %f, align 8\n",
        ),
    ] {
        let text = kernel(
            name,
            2,
            "%S = type { i32, ptr addrspace(1) }",
            &format!(
                r#"entry:
  %s = alloca %S, align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
{store}  %w = load i64, ptr %f, align 8
  %i = zext i32 %t to i64
  %odd = trunc i32 %t to i1
  br i1 %odd, label %a, label %z
a:
  %p = inttoptr i64 %w to ptr addrspace(1)
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  br label %z
z:
  %x = phi i32 [ %v, %a ], [ 77, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#
            ),
        );
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        let inp = input(9);
        let want: Vec<u32> = (0..THREADS as usize)
            .map(|t| if t % 2 == 1 { inp[t] } else { 77 })
            .collect();
        for (stem, bufs) in run_everywhere(&bc, name, &[vec![0; THREADS as usize], inp], (2, 1)) {
            assert_eq!(bufs[0], want, "{name} {stem}");
        }
        for stem in PROFILES {
            let (plan, _) = typed_module(&bc, name, &p(stem)).expect("module");
            assert_eq!(count(&plan, "access.cell.word"), 1, "{name} {stem}");
        }
    }
}

/// Textures parked in a private array and read back at a run-time index through an integer (`load
/// i64` then `inttoptr`), the `fsNormalsUV` shape: each read is a cell word over both handles, so
/// the sample switches between the two textures statically.
#[test]
fn textures_read_back_through_their_bits_at_a_dynamic_index_are_handles() {
    let text = r#"%T = type { ptr addrspace(1) }
define void @texarray(ptr addrspace(1) %t0, ptr addrspace(1) %t1, <2 x i16> %pos) {
entry:
  %a = alloca [2 x %T], align 8
  %e0 = getelementptr inbounds [2 x %T], ptr %a, i64 0, i64 0, i32 0
  store ptr addrspace(1) %t0, ptr %e0, align 8
  %e1 = getelementptr inbounds [2 x %T], ptr %a, i64 0, i64 1, i32 0
  store ptr addrspace(1) %t1, ptr %e1, align 8
  %x = extractelement <2 x i16> %pos, i64 0
  %k = and i16 %x, 1
  %kk = zext i16 %k to i64
  %e = getelementptr inbounds [2 x %T], ptr %a, i64 0, i64 %kk
  %w = load i64, ptr %e, align 8
  br label %use
use:
  %u = inttoptr i64 %w to ptr addrspace(1)
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %u, <2 x i16> %pos, <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, i16 0, i32 2)
  ret void
}
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @texarray, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t0"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("texarray", text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile_given_storage_format(&bc, "texarray", &prof, PixelFormat::RGBA16Float)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let uses = |n: &str| a.strategies.entries.get(n).map_or(0, |u| u.uses);
        assert_eq!(
            (
                uses("access.cell.store"),
                uses("access.cell.load"),
                uses("access.cell.word")
            ),
            (2, 1, 1),
            "{stem}"
        );
        assert!(
            a.strategies.entries.keys().all(|k| !k.contains("heap")),
            "{stem}: static handles: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
    }
}

/// A pointer slot copied to another cell through an integer (`load i64` stored as is, the
/// `fsSFBCompose*` shape) is a slot copy of its 8 bytes, as a `memcpy` of them is: the
/// destination's pointer load reads the buffer back statically, through a chain of two copies.
#[test]
fn a_pointer_copied_through_its_bits_is_a_slot_copy() {
    let text = kernel(
        "bitcopy",
        2,
        "%S = type { i32, ptr addrspace(1) }",
        r#"entry:
  %s = alloca %S, align 8
  %d = alloca %S, align 8
  %e = alloca ptr addrspace(1), align 8
  %f = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr addrspace(1) %b1, ptr %f, align 8
  %w = load i64, ptr %f, align 8
  %g = getelementptr inbounds %S, ptr %d, i64 0, i32 1
  store i64 %w, ptr %g, align 8
  %w2 = load i64, ptr %g, align 8
  store i64 %w2, ptr %e, align 8
  %i = zext i32 %t to i64
  %p = load ptr addrspace(1), ptr %e, align 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %x = mul i32 %v, 3
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("bitcopy", &text) else {
        return;
    };
    let inp = input(11);
    let want: Vec<u32> = inp.iter().map(|v| v.wrapping_mul(3)).collect();
    for (stem, bufs) in run_everywhere(&bc, "bitcopy", &[vec![0; THREADS as usize], inp], (1, 1)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
}

/// The control: pointer bits copied out of the cell into device memory are data, so the source slot
/// is read as data and its cell stays open.
#[test]
fn pointer_bits_copied_out_of_the_cells_are_data() {
    let text = kernel(
        "bitleak",
        2,
        "",
        r#"entry:
  %s = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %b1, ptr %s, align 8
  store ptr addrspace(1) %b1, ptr %s, align 8
  %w = load i64, ptr %s, align 8
  store i64 %w, ptr addrspace(1) %b0, align 8
  %i = zext i32 %t to i64
  %p = load ptr addrspace(1), ptr %s, align 8
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  store i32 1, ptr addrspace(1) %q, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("bitleak", &text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match typed_module(&bc, "bitleak", &prof) {
            Ok((plan, _)) => {
                assert_eq!(count(&plan, "access.cell.load"), 0, "{stem}");
                assert!(
                    plan.push
                        .fields
                        .iter()
                        .any(|f| f.field == PushField::RootTableAddress),
                    "{stem}: a guest address resolves through the root table"
                );
            }
            Err(Error::Unsupported(u)) => assert!(
                matches!(
                    u.feature,
                    Feature::DereferenceLoadedPointer | Feature::PointerToInteger
                ),
                "{stem}: {u:?}"
            ),
            Err(e) => panic!("{stem}: {e}"),
        }
    }
}

/// A null device pointer inserted into a struct (`fsSFBCompose*` builds its texture struct with
/// null members) and extracted is the null pointer in the extraction's representation, as such a
/// member of a member-wise phi is; parked in a cell, it reads back as null and compares equal to
/// null, where the other member does not.
#[test]
fn a_null_pointer_member_is_the_null_pointer() {
    let text = kernel(
        "nullmember",
        1,
        "%A = type { ptr addrspace(1), ptr addrspace(1) }",
        r#"entry:
  %g0 = insertvalue %A poison, ptr addrspace(1) %b0, 0
  %g1 = insertvalue %A %g0, ptr addrspace(1) null, 1
  br label %body
body:
  %a = alloca %A, align 8
  %x0 = extractvalue %A %g1, 0
  %e0 = getelementptr inbounds %A, ptr %a, i64 0, i32 0
  store ptr addrspace(1) %x0, ptr %e0, align 8
  store ptr addrspace(1) %x0, ptr %e0, align 8
  %x1 = extractvalue %A %g1, 1
  %e1 = getelementptr inbounds %A, ptr %a, i64 0, i32 1
  store ptr addrspace(1) %x1, ptr %e1, align 8
  store ptr addrspace(1) %x1, ptr %e1, align 8
  %p = load ptr addrspace(1), ptr %e0, align 8
  %q = load ptr addrspace(1), ptr %e1, align 8
  %c = and i32 %t, 1
  %odd = icmp ne i32 %c, 0
  %r = select i1 %odd, ptr addrspace(1) %p, ptr addrspace(1) %q
  %z = icmp eq ptr addrspace(1) %r, null
  %v = select i1 %z, i32 7, i32 9
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("nullmember", &text) else {
        return;
    };
    let want: Vec<u32> = (0..THREADS)
        .map(|t| if t % 2 == 1 { 9 } else { 7 })
        .collect();
    for (stem, bufs) in run_everywhere(&bc, "nullmember", &[vec![0; THREADS as usize]], (4, 2)) {
        assert_eq!(bufs[0], want, "{stem}");
    }
    for stem in PROFILES {
        let (plan, _) = typed_module(&bc, "nullmember", &p(stem)).expect("module");
        assert_eq!(count(&plan, "arith.aggregate.null"), 1, "{stem}");
    }
}

/// The `fsSFBCompose*` shape: a struct of textures whose other members are null or never inserted
/// (the chain's `poison` base), each member extracted and spilled into a private struct. Those
/// members are the null handle in their cells: a write through each, read back, is a switch with no
/// texture to write (`texture.null.switch`). The texture read back from its own slot is its static
/// handle. (Read back, the null members' stores are observed; unread, they would go with what
/// nothing observes.)
#[test]
fn null_texture_members_spilled_to_a_cell_are_null_handles() {
    let text = r#"%T = type { ptr addrspace(1) }
%A = type { %T, %T, %T }
define void @nulltex(ptr addrspace(1) %t0, <2 x i16> %pos) {
entry:
  %g0 = insertvalue %A poison, ptr addrspace(1) %t0, 0, 0
  %g1 = insertvalue %A %g0, ptr addrspace(1) null, 1, 0
  br label %body
body:
  %a = alloca %A, align 8
  %x0 = extractvalue %A %g1, 0, 0
  %e0 = getelementptr inbounds %A, ptr %a, i64 0, i32 0, i32 0
  store ptr addrspace(1) %x0, ptr %e0, align 8
  store ptr addrspace(1) %x0, ptr %e0, align 8
  %x1 = extractvalue %A %g1, 1, 0
  %e1 = getelementptr inbounds %A, ptr %a, i64 0, i32 1, i32 0
  store ptr addrspace(1) %x1, ptr %e1, align 8
  store ptr addrspace(1) %x1, ptr %e1, align 8
  %x2 = extractvalue %A %g1, 2, 0
  %e2 = getelementptr inbounds %A, ptr %a, i64 0, i32 2, i32 0
  store ptr addrspace(1) %x2, ptr %e2, align 8
  store ptr addrspace(1) %x2, ptr %e2, align 8
  %u = load ptr addrspace(1), ptr %e0, align 8
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %u, <2 x i16> %pos, <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, i16 0, i32 2)
  %w1 = load ptr addrspace(1), ptr %e1, align 8
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %w1, <2 x i16> %pos, <4 x half> <half 2.0, half 2.0, half 2.0, half 2.0>, i16 0, i32 2)
  %w2 = load ptr addrspace(1), ptr %e2, align 8
  call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %w2, <2 x i16> %pos, <4 x half> <half 3.0, half 3.0, half 3.0, half 3.0>, i16 0, i32 2)
  ret void
}
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
!air.kernel = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @nulltex, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t0"}
!5 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pos"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("nulltex", text) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile_given_storage_format(&bc, "nulltex", &prof, PixelFormat::RGBA16Float)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let uses = |n: &str| a.strategies.entries.get(n).map_or(0, |u| u.uses);
        assert_eq!(
            (
                uses("arith.aggregate.handle"),
                uses("access.cell.store"),
                uses("access.cell.load"),
                uses("texture.null.switch"),
            ),
            (3, 6, 3, 2),
            "{stem}: {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>()
        );
    }
}
