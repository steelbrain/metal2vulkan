//! Memory intrinsics and the constant blob end to end.
//!
//! - The committed memcpy/memset fixtures (buffer ↔ private arena, constant buffer → private arena
//!   with a partial memset, threadgroup → threadgroup) compute their documented results on the CPU
//!   interpreter and pass L3 on every profile, with robust descriptors and without.
//! - A copy and a fill longer than the unroll threshold are counted loops (statically bounded)
//!   and exact; `memmove` of overlapping ranges is exact.
//! - Byte-aligned copies write only their bytes: the neighbours of the destination survive.
//! - Memory intrinsics of a dynamic length are counted loops with guarded tail bytes; overlapping
//!   `memmove`s longer than the unroll threshold or of a dynamic length within one buffer copy in
//!   the direction the offsets require; a `volatile` `memcpy` lowers as a plain one; a dynamic
//!   `memmove` between two slots compares their Metal addresses.
//! - `i64` lengths (Metal's `size_t`), loaded or scaled from a `uint` count, are exact on every
//!   profile, with `shaderInt64` and without it (the length is two words there); one of 2³² or more
//!   counts as `u32::MAX` bytes and every access it makes is contained (the CPU interpreter's step
//!   limit stops the run; no execution bound does). The Metal-compiled fixture
//!   `engine/kernel_copy_runtime_ulong_length` pins the same on a runtime `ulong` length.
//! - A punned or copied constant global lives in the constant blob: the artifact carries its
//!   bytes, the module binds `Synthetic(ConstantBlob)`, and loads through it are exact.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::library::FunctionConstantValues;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::compile::{plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, meter, profile, typed_module,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, InterpError, Order, Resources, TrapKind};
use crate::spirv::ir::{GlobalKind, TypedModule};
use crate::spirv::testing::zeroes_threadgroup_memory;

const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "runtime-floor",
    "lavapipe",
    "nvidia-pascal-linux",
];

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// Every profile of [`PROFILES`] as committed, and without buffer robustness (so containment is
/// checked in the shader).
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        out.push((
            stem.to_owned(),
            profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile"),
        ));
        out.push((format!("{stem} unrobust"), variant(stem, true, false)));
    }
    out
}

/// `stem` without buffer robustness (`unrobust`) and without `shaderInt64` (`no_int64`).
fn variant(stem: &str, unrobust: bool, no_int64: bool) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if unrobust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    if no_int64 {
        desc.features.remove(&VkFeature::ShaderInt64);
    }
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-copy-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
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

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

/// Compile entry `entry` of `bc` for `prof`, pass L3, and check that the plan's size estimate
/// bounds the emitted words.
fn l3(bc: &[u8], entry: &str, prof: &DeviceProfile, what: &str) {
    let a = compile(bc, entry, prof).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    let plan = plan_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        prof,
        &LowerInputs::new(),
        &mut meter(),
    )
    .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
    let words = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= words,
        "{what}: estimate {} < emitted {words}",
        plan.estimate.words
    );
}

fn to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Resources of a kernel with `threads` threads, `[[buffer(i)]]` holding `init[i]` (bytes),
/// `spec` besides the local size, and the blob when the plan reads one.
fn resources(plan: &Plan, init: &[Vec<u8>], threads: u32, spec: &[(SpecKey, u64)]) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), u64::from(threads));
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, u64::from(threads));
    s.extend(spec.iter().copied());
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            // A single-region dispatch: the region is unused.
            (PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let mut buffers: Vec<Buffer> = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
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
        spec: s,
    }
}

/// Run in both scheduling orders; both must agree. Returns the buffers' bytes.
fn run_both(m: &TypedModule, res: &Resources) -> Result<Vec<Vec<u8>>, InterpError> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone())?.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    Ok(outs.swap_remove(0))
}

/// The plan's memory intrinsics (`access.mem*`) and their uses. Each side is named by what it
/// reaches: `.to_arena` for a destination in the threadgroup or private arena (checked in the
/// shader in every leg, or provably in bounds), `.from_arena` for such a source; a descriptor view
/// (under the profile's encoding) adds nothing.
fn intrinsics(plan: &Plan) -> Vec<(String, u32)> {
    plan.census()
        .into_iter()
        .filter(|(name, _)| name.starts_with("access.mem"))
        .map(|(name, (uses, _))| (name, uses))
        .collect()
}

/// `(name, uses)` pairs as [`intrinsics`] lists them.
fn named(pairs: &[(&str, u32)]) -> Vec<(String, u32)> {
    pairs.iter().map(|&(n, u)| (n.to_owned(), u)).collect()
}

fn fl(x: f32) -> u32 {
    x.to_bits()
}

/// `blobs[1].payload = blobs[0].payload` through a private copy (two `memcpy`s of 12 bytes).
#[test]
fn copy_a_float_run_is_exact() {
    let Some(bc) = fixture("kernel_copy_a_float_run") else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "copy_a_float_run", &prof, &what);
        let (plan, m) = typed_module(&bc, "copy_a_float_run", &prof).expect("module");
        assert_eq!(
            intrinsics(&plan),
            named(&[
                ("access.memcpy.from_arena", 1),
                ("access.memcpy.to_arena", 1)
            ]),
            "{what}"
        );
        let init = [7, fl(1.5), fl(-2.25), fl(3.0), 9, 0, 0, 0];
        let res = resources(&plan, &[to_bytes(&init)], 1, &[]);
        let out = run_both(&m, &res).expect("run");
        assert_eq!(
            words(&out[0]),
            vec![
                7,
                fl(1.5),
                fl(-2.25),
                fl(3.0),
                9,
                fl(1.5),
                fl(-2.25),
                fl(3.0)
            ],
            "{what}"
        );
    }
}

/// `items[0]` and `items[1]` swap through a private copy (three 16-byte `memcpy`s, 8-aligned).
#[test]
fn swap_corners_through_a_local_is_exact() {
    let Some(bc) = fixture("kernel_swap_corners_through_a_local") else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "swap_corners_through_a_local", &prof, &what);
        let (plan, m) = typed_module(&bc, "swap_corners_through_a_local", &prof).expect("module");
        assert_eq!(
            intrinsics(&plan),
            named(&[
                ("access.memcpy", 1),
                ("access.memcpy.from_arena", 1),
                ("access.memcpy.to_arena", 1),
            ]),
            "{what}"
        );
        let init = [1, 2, 3, 4, 5, 6, 7, 8];
        let res = resources(&plan, &[to_bytes(&init)], 1, &[]);
        let out = run_both(&m, &res).expect("run");
        assert_eq!(words(&out[0]), vec![5, 6, 7, 8, 1, 2, 3, 4], "{what}");
    }
}

/// Two copies of a constant struct, one with its first 4 bytes and one with its first 12 bytes
/// cleared by `memset`, and a 16-byte `memset` of the output.
#[test]
fn zero_a_struct_prefix_is_exact() {
    let Some(bc) = fixture("kernel_zero_a_struct_prefix") else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "zero_a_struct_prefix", &prof, &what);
        let (plan, m) = typed_module(&bc, "zero_a_struct_prefix", &prof).expect("module");
        assert_eq!(
            intrinsics(&plan),
            named(&[
                ("access.memcpy.to_arena", 2),
                ("access.memset", 1),
                ("access.memset.to_arena", 2),
            ]),
            "{what}"
        );
        let src = [fl(1.0), fl(2.0), fl(3.0), fl(4.0)];
        let res = resources(&plan, &[to_bytes(&src), to_bytes(&[u32::MAX; 12])], 1, &[]);
        let out = run_both(&m, &res).expect("run");
        let z = fl(0.0);
        assert_eq!(
            words(&out[1]),
            vec![z, fl(2.0), fl(3.0), fl(4.0), z, z, z, fl(4.0), z, z, z, z],
            "{what}"
        );
    }
}

/// A 16-byte record copied between threadgroup slots by `memcpy`, then read member by member.
#[test]
fn threadgroup_record_copy_is_exact() {
    let Some(bc) = fixture("kernel_threadgroup_flat_leaf") else {
        return;
    };
    let entry = "kernel_threadgroup_flat_leaf";
    let profiles = profiles()
        .into_iter()
        .filter(|(_, p)| zeroes_threadgroup_memory(p));
    for (what, prof) in profiles {
        l3(&bc, entry, &prof, &what);
        let (plan, m) = typed_module(&bc, entry, &prof).expect("module");
        assert_eq!(
            intrinsics(&plan),
            named(&[("access.memcpy.to_arena.from_arena", 1)]),
            "{what}"
        );
        let input: Vec<u32> = (0..8u32).map(|t| 100 * t + 1).collect();
        let res = resources(
            &plan,
            &[vec![0; 32], to_bytes(&input)],
            8,
            &[(SpecKey::TgLength(0), 256)],
        );
        let out = run_both(&m, &res).expect("run");
        let want: Vec<u32> = (0..8u32)
            .map(|t| {
                let v = input[((t + 1) & 7) as usize];
                v + (v + 1) * 2 + (v + 2) * 4 + (v + 3) * 8
            })
            .collect();
        assert_eq!(words(&out[0]), want, "{what}");
    }
}

const BULK: &str = r#"define void @bulk(ptr addrspace(1) %dst, ptr addrspace(1) %src) {
entry:
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i64 1206, i1 false)
  %f = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 1216
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %f, i8 -91, i64 1030, i1 false)
  %a = getelementptr inbounds i8, ptr addrspace(1) %src, i64 2
  %b = getelementptr inbounds i8, ptr addrspace(1) %src, i64 5
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 1 %b, ptr addrspace(1) align 1 %a, i64 9, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @bulk, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
"#;

/// A 1206-byte word copy (301 words: a counted loop, then two tail bytes), a 1030-byte fill (257
/// words: a loop, then two tail bytes), and a 9-byte overlapping byte `memmove`; every byte around
/// them survives.
#[test]
fn long_copies_loop_and_overlapping_moves_are_exact() {
    let Some(bc) = assemble("bulk", BULK) else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "bulk", &prof, &what);
        let a = compile(&bc, "bulk", &prof).expect("compile");
        assert!(!a.facts.contains(&RuntimeFact::ExecutionStatus), "{what}");
        let (plan, m) = typed_module(&bc, "bulk", &prof).expect("module");
        assert_eq!(
            m.loops().len(),
            2,
            "{what}: the copy's and the fill's loops"
        );
        let census = plan.census();
        assert_eq!(census["access.memcpy.loop"].0, 1, "{what}");
        assert_eq!(census["access.memset.loop"].0, 1, "{what}");
        assert_eq!(census["access.memmove"].0, 1, "{what}");
        let src: Vec<u8> = (0..1216u32).map(|i| (i * 7 + 3) as u8).collect();
        let dst: Vec<u8> = vec![0x11; 2256];
        let res = resources(&plan, &[dst.clone(), src.clone()], 1, &[]);
        let out = run_both(&m, &res).expect("run");
        let mut want = dst;
        want[..1206].copy_from_slice(&src[..1206]);
        want[1216..2246].fill(0xA5);
        assert_eq!(out[0], want, "{what}: dst");
        let mut moved = src.clone();
        moved.copy_within(2..11, 5);
        assert_eq!(out[1], moved, "{what}: memmove");
    }
}

const BLOB: &str = r#"@table = internal addrspace(2) constant [8 x i32] [i32 16909060, i32 84281096, i32 151653132, i32 219025168, i32 -1, i32 7, i32 -559038737, i32 305419896], align 4

define void @blob(ptr addrspace(1) %out, i32 %t) {
entry:
  %tmp = alloca [8 x i32], align 4
  call void @llvm.memcpy.p0.p2.i64(ptr align 4 %tmp, ptr addrspace(2) align 4 @table, i64 32, i1 false)
  %k = and i32 %t, 7
  %p = getelementptr inbounds [8 x i32], ptr %tmp, i32 0, i32 %k
  %w = load i32, ptr %p, align 4
  %b = getelementptr inbounds i8, ptr addrspace(2) @table, i32 %t
  %byte = load i8, ptr addrspace(2) %b, align 1
  %bz = zext i8 %byte to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  %i = add i32 %t, 32
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i
  store i32 %bz, ptr addrspace(1) %o2, align 4
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @blob, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A constant global that is copied and punned lives in the blob: the artifact carries its 32
/// bytes, the module binds the blob, and a copy into a private array and byte loads through the
/// global are exact (32 threads: every byte of the table).
#[test]
fn a_copied_constant_global_is_read_from_the_blob() {
    let Some(bc) = assemble("blob", BLOB) else {
        return;
    };
    let table: [u32; 8] = [
        16909060,
        84281096,
        151653132,
        219025168,
        0xFFFF_FFFF,
        7,
        0xDEAD_BEEF,
        305419896,
    ];
    let table_bytes = to_bytes(&table);
    for (what, prof) in profiles() {
        l3(&bc, "blob", &prof, &what);
        let a = compile(&bc, "blob", &prof).expect("compile");
        assert_eq!(a.blob.as_deref(), Some(&table_bytes[..]), "{what}");
        let (plan, m) = typed_module(&bc, "blob", &prof).expect("module");
        assert!(
            plan.census()
                .keys()
                .any(|k| k.starts_with("access.blob.load")),
            "{what}"
        );
        let res = resources(&plan, &[vec![0; 4 * 64]], 32, &[]);
        let out = run_both(&m, &res).expect("run");
        let w = words(&out[0]);
        for t in 0..32usize {
            assert_eq!(w[t], table[t & 7], "{what}: word {t}");
            assert_eq!(w[32 + t], u32::from(table_bytes[t]), "{what}: byte {t}");
        }
    }
}

const DYNAMIC: &str = r#"define void @dynamic(ptr addrspace(1) %dst, ptr addrspace(1) %src, ptr addrspace(1) %lens) {
entry:
  %n = load i32, ptr addrspace(1) %lens, align 4
  %pm = getelementptr inbounds i32, ptr addrspace(1) %lens, i32 1
  %m = load i32, ptr addrspace(1) %pm, align 4
  %pk = getelementptr inbounds i32, ptr addrspace(1) %lens, i32 2
  %k = load i32, ptr addrspace(1) %pk, align 4
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i32 %n, i1 false)
  %f = getelementptr inbounds i8, ptr addrspace(1) %dst, i32 512
  call void @llvm.memset.p1.i32(ptr addrspace(1) align 1 %f, i8 90, i32 %m, i1 false)
  %a = getelementptr inbounds i8, ptr addrspace(1) %src, i32 2
  %b = getelementptr inbounds i8, ptr addrspace(1) %src, i32 5
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 1 %b, ptr addrspace(1) align 1 %a, i32 %k, i1 false)
  %c = getelementptr inbounds i8, ptr addrspace(1) %src, i32 304
  %d = getelementptr inbounds i8, ptr addrspace(1) %src, i32 292
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %d, ptr addrspace(1) align 4 %c, i32 %k, i1 false)
  %e = getelementptr inbounds i8, ptr addrspace(1) %src, i32 600
  %g = getelementptr inbounds i8, ptr addrspace(1) %src, i32 612
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %g, ptr addrspace(1) align 4 %e, i32 1100, i1 false)
  %v = getelementptr inbounds i8, ptr addrspace(1) %dst, i32 700
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) align 4 %v, ptr addrspace(1) align 4 %src, i32 10, i1 true)
  ret void
}

declare void @llvm.memcpy.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)
declare void @llvm.memset.p1.i32(ptr addrspace(1), i8, i32, i1)
declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @dynamic, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lens"}
"#;

/// Memory intrinsics of a dynamic length (counted loops, with the tail bytes guarded), overlapping
/// dynamic `memmove`s in both directions within one buffer, a constant `memmove` longer than 256
/// units (a directed loop), and a `volatile` `memcpy`, for several lengths each; every byte around
/// them survives.
#[test]
fn dynamic_lengths_and_long_moves_are_exact() {
    let Some(bc) = assemble("dynamic", DYNAMIC) else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "dynamic", &prof, &what);
        let (plan, m) = typed_module(&bc, "dynamic", &prof).expect("module");
        let census = plan.census();
        assert_eq!(census["access.memcpy.dynamic"].0, 1, "{what}");
        assert_eq!(census["access.memset.dynamic"].0, 1, "{what}");
        assert_eq!(census["access.memmove.dynamic"].0, 2, "{what}");
        assert_eq!(census["access.memmove.loop"].0, 1, "{what}");
        assert_eq!(census["access.memcpy"].0, 1, "{what}");
        for (n, mm, k) in [
            (0u32, 0u32, 0u32),
            (7, 1, 3),
            (301, 130, 45),
            (256, 255, 64),
        ] {
            let src: Vec<u8> = (0..1728u32).map(|i| (i * 7 + 3) as u8).collect();
            let dst: Vec<u8> = vec![0x11; 1024];
            let lens: Vec<u8> = to_bytes(&[n, mm, k, 0]);
            let res = resources(&plan, &[dst.clone(), src.clone(), lens], 1, &[]);
            let out = run_both(&m, &res).expect("run");
            let mut want = dst;
            want[..n as usize].copy_from_slice(&src[..n as usize]);
            want[512..512 + mm as usize].fill(90);
            let mut moved = src.clone();
            moved.copy_within(2..2 + k as usize, 5);
            moved.copy_within(304..304 + k as usize, 292);
            moved.copy_within(600..1700, 612);
            want[700..710].copy_from_slice(&moved[..10]);
            assert_eq!(out[1], moved, "{what} {n} {mm} {k}: memmove");
            assert_eq!(out[0], want, "{what} {n} {mm} {k}: dst");
        }
    }
}

const ALIASED_MOVE: &str = r#"define void @aliased(ptr addrspace(1) %dst, ptr addrspace(1) %src, i32 %n) {
entry:
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i32 %n, i1 false)
  ret void
}

declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @aliased, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}
"#;

/// A dynamic `memmove` between two buffer slots (possibly one buffer bound twice, with no
/// comparable offsets) compares the endpoints' Metal addresses (each slot's `SlotGuestBase`), never
/// a guessed direction: it passes L3 where the profile has `shaderInt64` for those `u64` addresses
/// and is refused naming it elsewhere (`engine_pointer_memory` executes it).
#[test]
fn a_dynamic_move_between_slots_compares_metal_addresses() {
    let Some(bc) = assemble("aliased", ALIASED_MOVE) else {
        return;
    };
    for (what, prof) in profiles() {
        if prof.fields().leaf.int64 {
            l3(&bc, "aliased", &prof, &what);
            let (plan, _) = typed_module(&bc, "aliased", &prof).expect("module");
            assert!(
                plan.push
                    .fields
                    .iter()
                    .any(|f| matches!(f.field, PushField::SlotGuestBase(_))),
                "{what}"
            );
            continue;
        }
        let e = compile(&bc, "aliased", &prof).expect_err("refused");
        let Error::Unsupported(u) = e else {
            panic!("{what}: {e:?}");
        };
        let want = Unsupported::needs(
            Feature::PointerCompare,
            Capability::Feature(VkFeature::ShaderInt64),
        );
        assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{what}");
    }
}

const MIXED: &str = r#"@tbl = internal addrspace(2) constant [4 x i32] [i32 101, i32 202, i32 303, i32 404], align 4

define void @mixed(ptr addrspace(2) %c, ptr addrspace(1) %out, i32 %t) {
entry:
  %odd = and i32 %t, 1
  %sel = icmp eq i32 %odd, 1
  %p = select i1 %sel, ptr addrspace(2) @tbl, ptr addrspace(2) %c
  %i = lshr i32 %t, 1
  %q = getelementptr inbounds i32, ptr addrspace(2) %p, i32 %i
  %v = load i32, ptr addrspace(2) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mixed, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"c"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A `constant` pointer selected between a program-scope constant (in the blob) and a buffer
/// parameter is a static fat pointer whose blob candidate has `BLOB_ROOT`; its loads read the blob
/// or the slot by root, exact for every thread on every profile.
#[test]
fn a_pointer_mixing_the_blob_and_a_slot_reads_both() {
    let Some(bc) = assemble("mixed", MIXED) else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&bc, "mixed", &prof, &what);
        let a = compile(&bc, "mixed", &prof).expect("compile");
        assert_eq!(
            a.blob.as_deref(),
            Some(&to_bytes(&[101, 202, 303, 404])[..]),
            "{what}"
        );
        let (plan, m) = typed_module(&bc, "mixed", &prof).expect("module");
        let c: Vec<u32> = (0..4).map(|k| 1000 + k).collect();
        let res = resources(&plan, &[to_bytes(&c), vec![0; 4 * 8]], 8, &[]);
        let out = run_both(&m, &res).expect("run");
        let w = words(&out[1]);
        for t in 0..8usize {
            let want = if t % 2 == 1 {
                [101, 202, 303, 404][t / 2]
            } else {
                c[t / 2]
            };
            assert_eq!(w[t], want, "{what}: thread {t}");
        }
    }
}

const MIXED_COPY: &str = r#"@tbl = internal addrspace(2) constant [8 x i32] [i32 101, i32 202, i32 303, i32 404, i32 505, i32 606, i32 707, i32 808], align 4

define void @mixed_copy(ptr addrspace(2) %c, ptr addrspace(1) %out, i32 %t) {
entry:
  %odd = and i32 %t, 1
  %sel = icmp eq i32 %odd, 1
  %p = select i1 %sel, ptr addrspace(2) @tbl, ptr addrspace(2) %c
  %i = lshr i32 %t, 1
  %q = getelementptr inbounds i32, ptr addrspace(2) %p, i32 %i
  %o = getelementptr inbounds [2 x i32], ptr addrspace(1) %out, i32 %t
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 4 %o, ptr addrspace(2) align 4 %q, i64 8, i1 false)
  %n = add i32 %t, 1
  %ob = getelementptr inbounds i8, ptr addrspace(1) %out, i32 64
  %d = getelementptr inbounds [3 x i8], ptr addrspace(1) %ob, i32 %t
  call void @llvm.memcpy.p1.p2.i32(ptr addrspace(1) align 1 %d, ptr addrspace(2) align 1 %q, i32 3, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1), ptr addrspace(2), i64, i1)
declare void @llvm.memcpy.p1.p2.i32(ptr addrspace(1), ptr addrspace(2), i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @mixed_copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"c"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A `memcpy` whose source is selected between a program-scope constant (in the blob) and a buffer
/// parameter reads the blob or the slot by root, per unit: word and byte copies are exact for every
/// thread on every profile.
#[test]
fn a_memcpy_from_a_pointer_mixing_the_blob_and_a_slot_is_exact() {
    let Some(bc) = assemble("mixed_copy", MIXED_COPY) else {
        return;
    };
    let tbl: Vec<u32> = (1..=8).map(|k| 101 * k).collect();
    for (what, prof) in profiles() {
        l3(&bc, "mixed_copy", &prof, &what);
        let (plan, m) = typed_module(&bc, "mixed_copy", &prof).expect("module");
        assert_eq!(plan.census()["access.memcpy"].0, 2, "{what}");
        let c: Vec<u32> = (0..8).map(|k| 1000 + k).collect();
        let res = resources(&plan, &[to_bytes(&c), vec![0; 64 + 3 * 8]], 8, &[]);
        let out = run_both(&m, &res).expect("run");
        for t in 0..8usize {
            let src = if t % 2 == 1 { &tbl } else { &c };
            let bytes = to_bytes(&src[t / 2..t / 2 + 2]);
            assert_eq!(out[1][8 * t..8 * t + 8], bytes[..], "{what}: words {t}");
            assert_eq!(
                out[1][64 + 3 * t..67 + 3 * t],
                bytes[..3],
                "{what}: bytes {t}"
            );
        }
    }
}

/// `i64` lengths as Metal spells them (`size_t` on 64-bit pointers).
const WIDE: &str = r#"define void @wide(ptr addrspace(1) %dst, ptr addrspace(1) %src, ptr addrspace(1) %lens) {
entry:
  %n = load i64, ptr addrspace(1) %lens, align 8
  %pm = getelementptr inbounds i64, ptr addrspace(1) %lens, i32 1
  %m = load i64, ptr addrspace(1) %pm, align 8
  %pk = getelementptr inbounds i64, ptr addrspace(1) %lens, i32 2
  %k = load i64, ptr addrspace(1) %pk, align 8
  %pj = getelementptr inbounds i64, ptr addrspace(1) %lens, i32 3
  %j = load i64, ptr addrspace(1) %pj, align 8
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i64 %n, i1 false)
  %f = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 512
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %f, i8 90, i64 %m, i1 false)
  %c = getelementptr inbounds i8, ptr addrspace(1) %src, i64 304
  %d = getelementptr inbounds i8, ptr addrspace(1) %src, i64 292
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 %d, ptr addrspace(1) align 4 %c, i64 %k, i1 false)
  %a = getelementptr inbounds i8, ptr addrspace(1) %src, i64 2
  %b = getelementptr inbounds i8, ptr addrspace(1) %src, i64 5
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 1 %b, ptr addrspace(1) align 1 %a, i64 %j, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @wide, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"lens"}
"#;

/// `i64` lengths computed from a `uint` count (`count * sizeof(uint)` as the Metal front end spells
/// it), whose `shl` reaches 2³² and beyond for counts of 2³⁰ or more.
const WIDE_COUNT: &str = r#"define void @wide_count(ptr addrspace(1) %dst, ptr addrspace(1) %src, ptr addrspace(1) %lens) {
entry:
  %x = load i32, ptr addrspace(1) %lens, align 4
  %n = zext i32 %x to i64
  %n4 = shl nuw nsw i64 %n, 2
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %dst, ptr addrspace(1) align 4 %src, i64 %n4, i1 false)
  %f = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 512
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %f, i8 90, i64 %n, i1 false)
  %c = getelementptr inbounds i8, ptr addrspace(1) %src, i64 304
  %d = getelementptr inbounds i8, ptr addrspace(1) %src, i64 292
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 %d, ptr addrspace(1) align 4 %c, i64 %n4, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @wide_count, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"lens"}
"#;

/// The interpreter's lane-step limit for a run of a length of 2³² or more (`u32::MAX` bytes): its
/// loop runs thousands of words past every binding long before the limit stops it.
const HUGE_STEPS: u64 = 1 << 18;

/// Every profile of [`PROFILES`] with and without buffer robustness, as committed and without
/// `shaderInt64`: `(what, profile, unrobust)`.
fn int64_profiles() -> Vec<(String, DeviceProfile, bool)> {
    let mut out = Vec::new();
    for stem in PROFILES {
        for unrobust in [false, true] {
            for no_int64 in [false, true] {
                let what = format!(
                    "{stem}{}{}",
                    if unrobust { " unrobust" } else { "" },
                    if no_int64 { " no-int64" } else { "" }
                );
                out.push((what, variant(stem, unrobust, no_int64), unrobust));
            }
        }
    }
    out
}

/// Run `m` once per scheduling order on the kernel's three buffers; returns them.
fn run_wide(m: &TypedModule, plan: &Plan, init: &[Vec<u8>]) -> Vec<Vec<u8>> {
    run_both(m, &resources(plan, init, 1, &[])).expect("run")
}

/// Run `m` once per scheduling order with a length of 2³² or more in `init`: the interpreter
/// traps any access outside a binding, so a step-limit trap (and no other) shows that every
/// access the run made, thousands of words past each binding, was contained.
fn assert_huge_run_is_contained(m: &TypedModule, plan: &Plan, init: &[Vec<u8>], case: &str) {
    let res = resources(plan, init, 1, &[]);
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            step_limit: HUGE_STEPS,
            ..Config::default()
        };
        let err = run(m, &cfg, res.clone()).expect_err("the huge run outruns the step limit");
        assert_eq!(
            err.trap_kind(),
            Some(&TrapKind::StepLimit(HUGE_STEPS)),
            "{case} {order:?}: {err}"
        );
    }
}

/// `memcpy`, `memset`, and directed `memmove`s (forward within one buffer, and backward over an
/// overlap) of `i64` lengths loaded from a buffer are exact for several lengths on every profile
/// with `shaderInt64`, in both containment encodings. A length of 2³² or more counts as `u32::MAX`
/// bytes, and on the in-shader encoding (which the interpreter executes; it traps on any access
/// outside a binding) every access its loop makes is contained until the interpreter's step limit
/// stops the run (no execution bound does). Without `shaderInt64` the length is two words, whose
/// nonzero high word saturates the same way.
#[test]
fn i64_lengths_are_exact_and_contained() {
    let Some(bc) = assemble("wide", WIDE) else {
        return;
    };
    const HUGE: u64 = (1 << 32) + 5;
    let src: Vec<u8> = (0..1728u32).map(|i| (i * 7 + 3) as u8).collect();
    let dst: Vec<u8> = vec![0x11; 1024];
    let (mut two_words, mut contained) = (0, 0);
    for (what, prof, unrobust) in int64_profiles() {
        two_words += u32::from(!prof.fields().leaf.int64);
        l3(&bc, "wide", &prof, &what);
        let (plan, m) = typed_module(&bc, "wide", &prof).expect("module");
        let census = plan.census();
        assert_eq!(census["access.memcpy.dynamic"].0, 1, "{what}");
        assert_eq!(census["access.memset.dynamic"].0, 1, "{what}");
        assert_eq!(census["access.memmove.dynamic"].0, 2, "{what}");
        for (n, mm, k, j) in [
            (0u64, 0u64, 0u64, 0u64),
            (7, 1, 3, 2),
            (301, 130, 45, 64),
            (1024, 512, 1300, 300),
        ] {
            let lens = [n, mm, k, j].iter().flat_map(|v| v.to_le_bytes()).collect();
            let out = run_wide(&m, &plan, &[dst.clone(), src.clone(), lens]);
            let mut want = dst.clone();
            want[..n as usize].copy_from_slice(&src[..n as usize]);
            want[512..512 + mm as usize].fill(90);
            let mut moved = src.clone();
            moved.copy_within(304..304 + k as usize, 292);
            moved.copy_within(2..2 + j as usize, 5);
            assert_eq!(out[1], moved, "{what} {n} {mm} {k} {j}: memmove");
            assert_eq!(out[0], want, "{what} {n} {mm} {k} {j}: dst");
        }
        if !unrobust {
            // Robust descriptors contain the huge copies in hardware, which the interpreter does
            // not model; the in-shader encoding below is the executed evidence.
            continue;
        }
        // One huge length per run, each call's loop in turn.
        for huge in 0..4 {
            let mut l = [0u64; 4];
            l[huge] = HUGE;
            let lens: Vec<u8> = l.iter().flat_map(|v| v.to_le_bytes()).collect();
            let init = [dst.clone(), src.clone(), lens];
            assert_huge_run_is_contained(&m, &plan, &init, &format!("{what} huge #{huge}"));
        }
        contained += 1;
    }
    assert!(two_words > 0 && contained > 0, "{two_words} {contained}");
}

/// `i64` lengths that are a `uint` count scaled by `shl` (the length Metal emits for `count *
/// sizeof(uint)`) are exact for counts below 2³⁰; a count of 2³⁰ or more makes the scaled length
/// 2³² or more, which counts as `u32::MAX` bytes whose every access is contained on the in-shader
/// encoding, with `shaderInt64` and as two words without it.
#[test]
fn scaled_i64_counts_are_exact_and_contained() {
    let Some(bc) = assemble("wide_count", WIDE_COUNT) else {
        return;
    };
    let src: Vec<u8> = (0..1728u32).map(|i| (i * 5 + 1) as u8).collect();
    let dst: Vec<u8> = vec![0x22; 1024];
    let (mut two_words, mut contained) = (0, 0);
    for (what, prof, unrobust) in int64_profiles() {
        two_words += u32::from(!prof.fields().leaf.int64);
        l3(&bc, "wide_count", &prof, &what);
        let (plan, m) = typed_module(&bc, "wide_count", &prof).expect("module");
        for x in [0u32, 1, 75, 128, 256] {
            let lens = to_bytes(&[x]);
            let out = run_wide(&m, &plan, &[dst.clone(), src.clone(), lens]);
            let (n, n4) = (x as usize, 4 * x as usize);
            let mut want = dst.clone();
            want[..n4].copy_from_slice(&src[..n4]);
            want[512..512 + n].fill(90);
            let mut moved = src.clone();
            moved.copy_within(304..304 + n4, 292);
            assert_eq!(out[0], want, "{what} {x}: dst");
            assert_eq!(out[1], moved, "{what} {x}: memmove");
        }
        if !unrobust {
            continue;
        }
        // 4 * (2^30 + 1) = 2^32 + 4: the memcpy counts u32::MAX bytes.
        let init = [dst.clone(), src.clone(), to_bytes(&[(1 << 30) + 1])];
        assert_huge_run_is_contained(&m, &plan, &init, &format!("{what} huge"));
        contained += 1;
    }
    assert!(two_words > 0 && contained > 0, "{two_words} {contained}");
}

/// The committed Metal-compiled fixture `kernel_copy_runtime_ulong_length` (`__builtin_memcpy` of a
/// runtime `ulong` length between word-aligned buffers, then a `__builtin_memset` of a `uint` count
/// widened to `i64`) is exact on every profile, with `shaderInt64` and without it, where each `i64`
/// length is two words. The lengths have a high word of 0 and run the tail bytes alone, whole words
/// alone, and both. A copy length with a nonzero high word (2³² + 5) counts `u32::MAX` bytes: on
/// the in-shader encoding every access its loop makes is contained until the interpreter's step
/// limit stops the run.
#[test]
fn a_runtime_ulong_length_is_exact_in_two_words() {
    let entry = "kernel_copy_runtime_ulong_length";
    let Some(bc) = fixture(&format!("engine/{entry}")) else {
        return;
    };
    let src: Vec<u8> = (0..1024u32).map(|i| (i * 11 + 5) as u8).collect();
    let dst: Vec<u8> = vec![0x33; 1024];
    // `struct Lengths { ulong copy; uint fill; }`: 16 bytes.
    let lengths = |copy: u64, fill: u32| -> Vec<u8> {
        let mut b = copy.to_le_bytes().to_vec();
        b.extend(fill.to_le_bytes());
        b.extend([0; 4]);
        b
    };
    let (mut two_words, mut contained) = (0, 0);
    for (what, prof, unrobust) in int64_profiles() {
        two_words += u32::from(!prof.fields().leaf.int64);
        l3(&bc, entry, &prof, &what);
        let (plan, m) = typed_module(&bc, entry, &prof).expect("module");
        let census = plan.census();
        assert_eq!(census["access.memcpy.dynamic"].0, 1, "{what}");
        assert_eq!(census["access.memset.dynamic"].0, 1, "{what}");
        for (copy, fill) in [
            (0u64, 0u32),
            (1, 1),
            (3, 2),
            (4, 3),
            (301, 130),
            (512, 511),
            (1024, 512),
        ] {
            let case = format!("{what} copy {copy} fill {fill}");
            let out = run_wide(&m, &plan, &[src.clone(), dst.clone(), lengths(copy, fill)]);
            let mut want = dst.clone();
            want[..copy as usize].copy_from_slice(&src[..copy as usize]);
            want[512..512 + fill as usize].fill(0x5a);
            assert_eq!(out[1], want, "{case}: dst");
            assert_eq!(out[0], src, "{case}: src");
        }
        if !unrobust {
            // Robust descriptors contain the huge copy in hardware, which the interpreter does
            // not model; the in-shader encoding is the executed evidence.
            continue;
        }
        let init = [src.clone(), dst.clone(), lengths((1 << 32) + 5, 7)];
        assert_huge_run_is_contained(&m, &plan, &init, &format!("{what} huge"));
        contained += 1;
    }
    assert!(two_words > 0 && contained > 0, "{two_words} {contained}");
}

/// A kernel copying `len` bytes from `a + 4` to `b` (two private `[4 x i32]`, the copy declared
/// `align`), after filling `a` from `in`; thread `t` then writes `b[t % 3]` to `out[t]`.
fn local_copy(align: u32, len: u32) -> String {
    format!(
        r#"define void @local_copy(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %t) {{
entry:
  %a = alloca [4 x i32], align 4
  %b = alloca [4 x i32], align 4
  %i0 = load i32, ptr addrspace(1) %in, align 4
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %i1 = load i32, ptr addrspace(1) %q1, align 4
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 2
  %i2 = load i32, ptr addrspace(1) %q2, align 4
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 3
  %i3 = load i32, ptr addrspace(1) %q3, align 4
  %t0 = add i32 %t, 0
  %s0 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %t0
  store i32 %i0, ptr %s0, align 4
  %t1 = add i32 %t0, 1
  %s1 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %t1
  store i32 %i1, ptr %s1, align 4
  %t2 = add i32 %t0, 2
  %s2 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %t2
  store i32 %i2, ptr %s2, align 4
  %t3 = add i32 %t0, 3
  %s3 = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %t3
  store i32 %i3, ptr %s3, align 4
  %src = getelementptr inbounds i8, ptr %a, i32 4
  call void @llvm.memcpy.p0.p0.i64(ptr align {align} %b, ptr align {align} %src, i64 {len}, i1 false)
  %k = urem i32 %t, 3
  %pb = getelementptr inbounds [4 x i32], ptr %b, i32 0, i32 %k
  %w = load i32, ptr %pb, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!air.kernel = !{{!0}}
!0 = !{{ptr @local_copy, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}}
!5 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
"#
    )
}

/// A copy between private objects at constant offsets copies words although it is declared `align
/// 1` (both offsets are multiples of 4, and the arena starts at a word), so it emits the module of
/// the same copy declared `align 4`; its accesses provably lie inside the arena, so they add no
/// containment construct (a 12-byte copy has as many selection merges as an empty one); and it is
/// exact (thread 0 stores into `a[0..4]`, so `b = a[1..4]`).
#[test]
fn a_copy_at_constant_arena_offsets_copies_words_without_checks() {
    const SELECTION_MERGE: u32 = 247;
    let merges = |words: &[u32]| {
        let mut n = 0;
        let mut i = 5;
        while let Some(w) = words.get(i) {
            n += usize::from(w & 0xFFFF == SELECTION_MERGE);
            i += ((w >> 16) as usize).max(1);
        }
        n
    };
    let (Some(narrow), Some(wide), Some(empty)) = (
        assemble("local_copy", &local_copy(1, 12)),
        assemble("local_copy", &local_copy(4, 12)),
        assemble("local_copy", &local_copy(1, 0)),
    ) else {
        return;
    };
    for (what, prof) in profiles() {
        l3(&narrow, "local_copy", &prof, &what);
        let a = compile(&narrow, "local_copy", &prof).expect("compile");
        let b = compile(&wide, "local_copy", &prof).expect("compile");
        let e = compile(&empty, "local_copy", &prof).expect("compile");
        assert_eq!(
            a.module.words, b.module.words,
            "{what}: align 1 copies words"
        );
        assert_eq!(
            merges(&a.module.words),
            merges(&e.module.words),
            "{what}: the copy adds no containment construct"
        );
        let (plan, m) = typed_module(&narrow, "local_copy", &prof).expect("module");
        let input: [u32; 4] = [0x0102_0304, 0xA0B0_C0D0, 0xDEAD_BEEF, 0x1234_5678];
        let res = resources(&plan, &[vec![0; 4], to_bytes(&input)], 1, &[]);
        let out = run_both(&m, &res).expect("run");
        assert_eq!(words(&out[0]), vec![input[1]], "{what}");
    }
}

/// Copies whose declared alignment is `align`: 48 bytes `src → dst`, 64 bytes from the `constant`
/// slot `k` into a private object and from it to `dst + 64`, a 1040-byte (a loop) and a
/// 1024-byte (unrolled) `src → dst` copy, then within `dst` an overlapping 48-byte `memmove` and
/// two overlapping 1040-byte `memmove`s, upward then downward. Three slots fit every profile's
/// storage-buffer limit.
fn aligned_copies(align: u32) -> String {
    format!(
        r#"define void @aligned_copies(ptr addrspace(1) noundef "air-buffer-no-alias" %dst, ptr addrspace(1) noundef "air-buffer-no-alias" %src, ptr addrspace(2) noundef "air-buffer-no-alias" %k) {{
entry:
  %a = alloca [16 x i32], align 16
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align {align} %dst, ptr addrspace(1) align {align} %src, i64 48, i1 false)
  call void @llvm.memcpy.p0.p2.i64(ptr align {align} %a, ptr addrspace(2) align {align} %k, i64 64, i1 false)
  %d1 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 64
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align {align} %d1, ptr align {align} %a, i64 64, i1 false)
  %s2 = getelementptr inbounds i8, ptr addrspace(1) %src, i64 48
  %d2 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 128
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align {align} %d2, ptr addrspace(1) align {align} %s2, i64 1040, i1 false)
  %s3 = getelementptr inbounds i8, ptr addrspace(1) %src, i64 1088
  %d3 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 1168
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align {align} %d3, ptr addrspace(1) align {align} %s3, i64 1024, i1 false)
  %m0 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 2192
  %m1 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 2208
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align {align} %m1, ptr addrspace(1) align {align} %m0, i64 48, i1 false)
  %n0 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 2256
  %n16 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 2272
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align {align} %n16, ptr addrspace(1) align {align} %n0, i64 1040, i1 false)
  %n32 = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 2288
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align {align} %n0, ptr addrspace(1) align {align} %n32, i64 1040, i1 false)
  ret void
}}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1), ptr, i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{{!0}}
!air.version = !{{!8}}
!air.language_version = !{{!9}}
!0 = !{{ptr @aligned_copies, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"dst"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"src"}}
!5 = !{{i32 2, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 64, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Four", !"air.arg_name", !"k"}}
!8 = !{{i32 2, i32 8, i32 0}}
!9 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The resource views of kernel buffer `index` in `m`.
fn views_of(m: &TypedModule, index: u32) -> usize {
    m.globals()
        .iter()
        .filter(|g| {
            g.kind
                == GlobalKind::Resource(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index,
                })
        })
        .count()
}

/// A `memcpy` or `memmove` of a constant multiple of 16 bytes whose buffer endpoints are declared
/// `align 16` moves 16-byte units, each one element of the slot's `uvec4` view (a private endpoint
/// moves its four words): every copy and move is exact (overlapping ones too, short buffered and
/// long directed), the copies unroll or loop exactly as their word copies (`align 8`) do, each
/// storage slot gains its `uvec4` view, and the module is smaller than the word copies'.
#[test]
fn sixteen_aligned_copies_move_uvec4_units() {
    let (Some(vec4), Some(word)) = (
        assemble("aligned_copies", &aligned_copies(16)),
        assemble("aligned_copies", &aligned_copies(8)),
    ) else {
        return;
    };
    let entry = "aligned_copies";
    let dst: Vec<u8> = (0..3360u32).map(|i| (i * 13 + 5) as u8).collect();
    let src: Vec<u8> = (0..2144u32).map(|i| (i * 7 + 3) as u8).collect();
    let k: Vec<u8> = (0..64u32).map(|i| (i * 11 + 1) as u8).collect();
    let mut want = dst.clone();
    want[..48].copy_from_slice(&src[..48]);
    want[64..128].copy_from_slice(&k);
    want[128..1168].copy_from_slice(&src[48..1088]);
    want[1168..2192].copy_from_slice(&src[1088..2112]);
    want.copy_within(2192..2240, 2208);
    want.copy_within(2256..3296, 2272);
    want.copy_within(2288..3328, 2256);
    for (what, prof) in profiles() {
        let mut census = Vec::new();
        let mut words = Vec::new();
        for (bc, align) in [(&vec4, 16), (&word, 8)] {
            let what = format!("{what} align {align}");
            l3(bc, entry, &prof, &what);
            let a = compile(bc, entry, &prof).expect("compile");
            words.push(a.module.words.len());
            let (plan, module) = typed_module(bc, entry, &prof).expect("module");
            let counts: BTreeMap<String, u32> = plan
                .census()
                .into_iter()
                .map(|(k, (n, _))| (k, n))
                .collect();
            census.push(counts);
            let res = resources(&plan, &[dst.clone(), src.clone(), k.clone()], 1, &[]);
            let out = run_both(&module, &res).expect("run");
            assert_eq!(out[0], want, "{what}: dst");
            assert_eq!(out[1], src, "{what}: src");
            let views = if align == 16 { 2 } else { 1 };
            for slot in [0, 1] {
                assert_eq!(views_of(&module, slot), views, "{what}: views of {slot}");
            }
        }
        assert_eq!(census[0], census[1], "{what}: the units unroll as words");
        assert_eq!(census[0]["access.memcpy"], 2, "{what}");
        assert_eq!(census[0]["access.memcpy.to_arena"], 1, "{what}");
        assert_eq!(census[0]["access.memcpy.from_arena"], 1, "{what}");
        assert_eq!(census[0]["access.memcpy.loop"], 1, "{what}");
        assert_eq!(census[0]["access.memmove"], 1, "{what}");
        assert_eq!(census[0]["access.memmove.loop"], 2, "{what}");
        assert!(
            words[0] < words[1],
            "{what}: {} >= {} words",
            words[0],
            words[1]
        );
    }
}

/// A copy declared `align 16` whose length is not a multiple of 16 moves words, and so does one
/// with an endpoint declared below 16; neither declares a `uvec4` view.
#[test]
fn a_copy_not_in_whole_sixteen_byte_units_moves_words() {
    const BODY: &str = r#"define void @ragged(ptr addrspace(1) noundef "air-buffer-no-alias" %dst, ptr addrspace(1) noundef "air-buffer-no-alias" %src) {
entry:
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src, i64 40, i1 false)
  %d = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 48
  %s = getelementptr inbounds i8, ptr addrspace(1) %src, i64 48
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %d, ptr addrspace(1) align 8 %s, i64 32, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}
!0 = !{ptr @ragged, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"src"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
"#;
    let Some(bc) = assemble("ragged", BODY) else {
        return;
    };
    let dst: Vec<u8> = (0..96u32).map(|i| (i * 13 + 5) as u8).collect();
    let src: Vec<u8> = (0..96u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut want = dst.clone();
    want[..40].copy_from_slice(&src[..40]);
    want[48..80].copy_from_slice(&src[48..80]);
    for (what, prof) in profiles() {
        l3(&bc, "ragged", &prof, &what);
        let (plan, module) = typed_module(&bc, "ragged", &prof).expect("module");
        let res = resources(&plan, &[dst.clone(), src.clone()], 1, &[]);
        let out = run_both(&module, &res).expect("run");
        assert_eq!(out[0], want, "{what}");
        assert_eq!(views_of(&module, 0), 1, "{what}: views of dst");
        assert_eq!(views_of(&module, 1), 1, "{what}: views of src");
    }
}
