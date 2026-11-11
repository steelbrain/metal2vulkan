//! Threadgroup and private arenas end to end.
//!
//! - The 32768-byte static threadgroup array (`engine/threadgroup_32k`) compiles for
//!   `moltenvk-m3` and `lavapipe` (whose `maxComputeSharedMemorySize` is exactly 32768), passes
//!   L3, and computes its documented result on the CPU interpreter with 1024 threads; one byte
//!   more is refused naming the limit, as is 32768 on a 16 KiB profile.
//! - Reading threadgroup memory nobody wrote traps on the interpreter (it poisons `Workgroup`
//!   bytes).
//! - `[[threadgroup(n)]]` regions follow the static globals at offsets that are specialization
//!   sums: results are exact for every specialized length, and accesses past the arena's
//!   specialized end are contained (a load is 0, a store is dropped, nothing traps).
//! - Byte stores into threadgroup memory by neighbouring invocations of different subgroups
//!   preserve each other (two word atomics).
//! - An alloca whose accesses are punned lives in the private arena, word-composed and
//!   byte-exact.
//! - Out-of-bounds threadgroup indices are contained; with containment off no check is emitted.
//! - Typed allocas and closed constant globals are `Private` arrays indexed by access chains; their
//!   dynamic indices are contained under `GuestIsolation::On` (a typed alloca's out-of-range load
//!   is 0 and its out-of-range store is dropped; a constant global's out-of-range load reads the
//!   level's last element) and unclamped without it.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::{Budget, Meter, LEDGER_BYTES, SPIRV_WORDS, WORK_UNITS};
use crate::device::profile::{DeviceProfile, GuestIsolation};
use crate::device::requirements::VkLimit;
use crate::device::{ContainmentPolicy, DevicePolicy};
use crate::error::{Capability, Error, Feature, Limit, LowerFamily, Needs, Unsupported};
use crate::library::FunctionConstantValues;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage, SyntheticBinding,
};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{
    run, Buffer, Config, InterpError, Order, Resources, SubgroupSize, TrapKind,
};
use crate::spirv::ir::TypedModule;

/// Profiles with and without 64-bit integers, 16 and 32 KiB of shared memory.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "lavapipe",
];

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("profile")
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-arena-{}-{}-{name}",
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

fn out_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// The resources of a one-output kernel with `threads` threads: output buffer `index` of
/// `words` zero words, the local size, and `spec` besides.
fn resources(
    plan: &Plan,
    index: u32,
    words: usize,
    threads: u32,
    spec: &[(SpecKey, u64)],
) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), u64::from(threads));
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, u64::from(threads));
    s.extend(spec.iter().copied());
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            // Zero residuals: the output is bound at its start.
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(out_slot(index)),
            address: None,
            bytes: vec![0; words * 4],
        }],
        push,
        spec: s,
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run in both scheduling orders; both must agree. Returns the output words.
fn run_both(m: &TypedModule, res: &Resources) -> Result<Vec<u32>, InterpError> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        outs.push(words(&run(m, &cfg, res.clone())?.buffers[0]));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    Ok(outs.swap_remove(0))
}

fn l3(bc: &[u8], entry: &str, prof: &DeviceProfile, what: &str) {
    let a = compile(bc, entry, prof).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
}

/// The 32768-byte static threadgroup array is exactly the arena, so it fits
/// `maxComputeSharedMemorySize = 32768` on `moltenvk-m3` and `lavapipe`, and computes
/// `out[t] = (8191 - t) * 3` with 1024 threads.
#[test]
fn a_32768_byte_threadgroup_array_compiles_for_moltenvk_and_lavapipe() {
    let bc = match llvm_assemble(&engine_dir().join("threadgroup_32k.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("{why}"),
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let prof = p(stem);
        assert_eq!(prof.fields().limits.max_compute_shared_memory_size, 32768);
        l3(&bc, "threadgroup_32k", &prof, stem);
        let (plan, m) = typed_module(&bc, "threadgroup_32k", &prof).expect("module");
        let tg = plan.families.access.local.tg.as_ref().expect("an arena");
        assert_eq!((tg.static_size, tg.regions.len()), (32768, 0), "{stem}");
        let res = resources(&plan, 0, 1024, 1024, &[]);
        let out = run_both(&m, &res).expect("run");
        let want: Vec<u32> = (0..1024u32).map(|t| (8191 - t) * 3).collect();
        assert_eq!(out, want, "{stem}");
    }
}

fn big_array_kernel(words: u32) -> String {
    format!(
        r#"@scratch = internal addrspace(3) global [{words} x i32] undef, align 4

define void @big(ptr addrspace(1) %out, i32 %t) {{
  %p = getelementptr inbounds [{words} x i32], ptr addrspace(3) @scratch, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p, align 4
  %v = load i32, ptr addrspace(3) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @big, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
"#
    )
}

/// One word past 32768 bytes is refused naming `maxComputeSharedMemorySize`, and 32768 is refused
/// on a profile with 16 KiB; 16 KiB fits there.
#[test]
fn the_static_arena_is_checked_against_max_compute_shared_memory_size() {
    let refusal = |required: u64, available: u64| -> Error {
        Unsupported::needs(
            Feature::ThreadgroupMemorySize,
            Capability::Limit {
                limit: VkLimit::MaxComputeSharedMemorySize,
                required,
                available,
            },
        )
        .into()
    };
    let strip = |e: Error| match e {
        Error::Unsupported(mut u) => {
            u.location = None;
            Error::Unsupported(u)
        }
        other => other,
    };
    let Some(over) = assemble("over", &big_array_kernel(8193)) else {
        return;
    };
    let Some(full) = assemble("full", &big_array_kernel(8192)) else {
        return;
    };
    let Some(half) = assemble("half", &big_array_kernel(4096)) else {
        return;
    };
    let e = compile(&over, "big", &p("moltenvk-m3")).expect_err("over the limit");
    assert_eq!(strip(e), refusal(32772, 32768));
    let e = compile(&full, "big", &p("vk12-floor")).expect_err("over 16 KiB");
    assert_eq!(strip(e), refusal(32768, 16384));
    l3(&full, "big", &p("moltenvk-m3"), "moltenvk-m3 32768");
    l3(&half, "big", &p("vk12-floor"), "vk12-floor 16384");
}

/// A threadgroup word read before any invocation wrote it traps on the interpreter.
#[test]
fn an_uninitialized_threadgroup_read_traps_on_the_interpreter() {
    let Some(bc) = assemble(
        "uninit",
        r#"@g = internal addrspace(3) global [64 x i32] undef, align 4

define void @uninit(ptr addrspace(1) %out, i32 %t) {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @g, i32 0, i32 %t
  %v = load i32, ptr addrspace(3) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @uninit, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#,
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "uninit", &prof, stem);
        let (plan, m) = typed_module(&bc, "uninit", &prof).expect("module");
        let res = resources(&plan, 0, 64, 64, &[]);
        let r = run(&m, &Config::default(), res);
        match r {
            Err(InterpError::Trap(t)) => assert!(
                matches!(t.kind, TrapKind::UninitializedRead { .. }),
                "{stem}: {t:?}"
            ),
            other => panic!("{stem}: {other:?}"),
        }
    }
}

const REGIONS: &str = r#"@s = internal addrspace(3) global [3 x i32] undef, align 4

define void @regions(ptr addrspace(3) %a, ptr addrspace(1) %out, ptr addrspace(3) %b, i32 %t) {
  %pa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %t
  store i32 %t, ptr addrspace(3) %pa, align 4
  %t2 = mul i32 %t, 2
  %pb = getelementptr inbounds i32, ptr addrspace(3) %b, i32 %t
  store i32 %t2, ptr addrspace(3) %pb, align 4
  %first = icmp eq i32 %t, 0
  br i1 %first, label %seed, label %sync

seed:
  store i32 7, ptr addrspace(3) @s, align 4
  %p1 = getelementptr inbounds [3 x i32], ptr addrspace(3) @s, i32 0, i32 1
  store i32 7, ptr addrspace(3) %p1, align 4
  %ps = getelementptr inbounds [3 x i32], ptr addrspace(3) @s, i32 0, i32 2
  store i32 1000, ptr addrspace(3) %ps, align 4
  br label %sync

sync:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 63, %t
  %qa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %r
  %va = load i32, ptr addrspace(3) %qa, align 4
  %qb = getelementptr inbounds i32, ptr addrspace(3) %b, i32 %r
  %vb = load i32, ptr addrspace(3) %qb, align 4
  %qs = getelementptr inbounds [3 x i32], ptr addrspace(3) @s, i32 0, i32 2
  %vs = load i32, ptr addrspace(3) %qs, align 4
  %x = add i32 %va, %vb
  %y = add i32 %x, %vs
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %y, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @regions, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!6 = !{i32 3, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Two `[[threadgroup(n)]]` regions after a 12-byte static global start at 16 and at `16 +
/// round_up(len_0, 16)`, from the specialized lengths. With both regions long enough the result is
/// exact; with region 1 specialized to 16 bytes the accesses past the arena's end are contained:
/// its stores are dropped and its loads read 0, and nothing traps (a contained load reads word 0,
/// which the kernel initializes: the interpreter traps on reading poison even when the value is
/// then discarded).
#[test]
fn threadgroup_regions_follow_their_specialized_lengths() {
    let Some(bc) = assemble("regions", REGIONS) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "regions", &prof, stem);
        let a = compile(&bc, "regions", &prof).expect("compile");
        for index in [0, 1] {
            assert!(
                a.facts
                    .contains(&RuntimeFact::ThreadgroupMemoryLength { index }),
                "{stem}: region {index}"
            );
        }
        let (plan, m) = typed_module(&bc, "regions", &prof).expect("module");
        let tg = plan.families.access.local.tg.as_ref().expect("an arena");
        assert_eq!(tg.static_size, 12);
        assert_eq!(tg.regions, vec![(0, 0), (2, 1)]);
        let full = resources(
            &plan,
            0,
            64,
            64,
            &[(SpecKey::TgLength(0), 256), (SpecKey::TgLength(1), 256)],
        );
        let want: Vec<u32> = (0..64u32).map(|t| (63 - t) * 3 + 1000).collect();
        assert_eq!(run_both(&m, &full).expect("run"), want, "{stem}");
        // Region 0 needs 256 bytes; region 1 gets 16 (4 words) and the arena ends there.
        let short = resources(
            &plan,
            0,
            64,
            64,
            &[(SpecKey::TgLength(0), 256), (SpecKey::TgLength(1), 16)],
        );
        let want: Vec<u32> = (0..64u32)
            .map(|t| {
                let r = 63 - t;
                r + if r < 4 { 2 * r } else { 0 } + 1000
            })
            .collect();
        assert_eq!(run_both(&m, &short).expect("run"), want, "{stem} short");
    }
}

/// 64 invocations (two subgroups) store neighbouring bytes of shared words; each byte keeps its own
/// value.
#[test]
fn neighbouring_threadgroup_byte_stores_preserve_each_other() {
    let Some(bc) = assemble(
        "bytes",
        r#"@g = internal addrspace(3) global [64 x i8] undef, align 4

define void @bytes(ptr addrspace(1) %out, i32 %t) {
  %p = getelementptr inbounds [64 x i8], ptr addrspace(3) @g, i32 0, i32 %t
  %m = mul i32 %t, 7
  %v = trunc i32 %m to i8
  store i8 %v, ptr addrspace(3) %p, align 1
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 63, %t
  %q = getelementptr inbounds [64 x i8], ptr addrspace(3) @g, i32 0, i32 %r
  %b = load i8, ptr addrspace(3) %q, align 1
  %w = zext i8 %b to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @bytes, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#,
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "bytes", &prof, stem);
        let (plan, m) = typed_module(&bc, "bytes", &prof).expect("module");
        let res = resources(&plan, 0, 64, 64, &[]);
        let want: Vec<u32> = (0..64u32).map(|t| ((63 - t) * 7) & 0xFF).collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// An alloca read through a punned byte pointer is not typed, so it lives in the private arena,
/// word-composed: byte loads and a byte store are exact.
#[test]
fn a_punned_alloca_lives_in_the_private_arena() {
    let Some(bc) = assemble(
        "punned",
        r#"define void @punned(ptr addrspace(1) %out, i32 %t) {
  %a = alloca [2 x i32], align 4
  %a0 = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 0
  store i32 287454020, ptr %a0, align 4
  %a1 = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 1
  store i32 %t, ptr %a1, align 4
  %k = and i32 %t, 7
  %pb = getelementptr inbounds i8, ptr %a, i32 %k
  %b = load i8, ptr %pb, align 1
  %bw = zext i8 %b to i32
  %j = and i32 %t, 3
  %pj = getelementptr inbounds i8, ptr %a, i32 %j
  store i8 -85, ptr %pj, align 1
  %w = load i32, ptr %a0, align 4
  %i0 = shl i32 %t, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i0
  store i32 %bw, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %i0, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %w, ptr addrspace(1) %o1, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @punned, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#,
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "punned", &prof, stem);
        let (plan, m) = typed_module(&bc, "punned", &prof).expect("module");
        assert_eq!(plan.families.access.local.private, Some(8), "{stem}");
        assert!(
            plan.census().contains_key("access.private.load.checked"),
            "{stem}"
        );
        let res = resources(&plan, 0, 64, 32, &[]);
        let want: Vec<u32> = (0..32u32)
            .flat_map(|t| {
                let mut bytes = 0x1122_3344u32.to_le_bytes().to_vec();
                bytes.extend_from_slice(&t.to_le_bytes());
                let b = u32::from(bytes[(t & 7) as usize]);
                let mut w = 0x1122_3344u32.to_le_bytes();
                w[(t & 3) as usize] = 0xAB;
                [b, u32::from_le_bytes(w)]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// A punned private alloca accessed at constant byte offsets decides each check at translation
/// time: the stores and loads inside it run unchecked and exactly, a load past it reads 0, and a
/// store past it changes nothing; the dynamic accesses beside them stay checked.
#[test]
fn constant_private_offsets_decide_their_checks() {
    let Some(bc) = assemble(
        "konst",
        r#"define void @konst(ptr addrspace(1) %out, i32 %t) {
  %a = alloca [2 x i32], align 4
  %a0 = getelementptr inbounds [2 x i32], ptr %a, i32 0, i32 0
  store i32 287454020, ptr %a0, align 4
  %a1 = getelementptr inbounds i8, ptr %a, i32 4
  store i32 %t, ptr %a1, align 4
  %b2 = getelementptr inbounds i8, ptr %a, i32 2
  store i8 -85, ptr %b2, align 1
  %past = getelementptr inbounds i8, ptr %a, i32 8
  store i32 -1, ptr %past, align 4
  %r = load i32, ptr %past, align 4
  %k = and i32 %t, 7
  %pk = getelementptr inbounds i8, ptr %a, i32 %k
  %b = load i8, ptr %pk, align 1
  %bw = zext i8 %b to i32
  %w = load i32, ptr %a0, align 4
  %i0 = mul i32 %t, 3
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i0
  store i32 %bw, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %i0, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %w, ptr addrspace(1) %o1, align 4
  %i2 = add i32 %i0, 2
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i2
  store i32 %r, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @konst, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#,
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "konst", &prof, stem);
        let (plan, m) = typed_module(&bc, "konst", &prof).expect("module");
        assert!(
            plan.census().contains_key("access.private.load.checked"),
            "{stem}"
        );
        let res = resources(&plan, 0, 96, 32, &[]);
        let want: Vec<u32> = (0..32u32)
            .flat_map(|t| {
                let mut bytes = 0x1122_3344u32.to_le_bytes().to_vec();
                bytes[2] = 0xAB;
                bytes.extend_from_slice(&t.to_le_bytes());
                let b = u32::from(bytes[(t & 7) as usize]);
                let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                [b, w, 0]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

const OOB: &str = r#"@g = internal addrspace(3) global [16 x i32] undef, align 4

define void @oob(ptr addrspace(1) %out, i32 %t) {
  %p = getelementptr inbounds [16 x i32], ptr addrspace(3) @g, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %v = load i32, ptr addrspace(3) %p, align 4
  %pb = getelementptr inbounds i8, ptr addrspace(3) @g, i32 1
  %b = load i8, ptr addrspace(3) %pb, align 1
  %bz = zext i8 %b to i32
  %b0 = and i32 %bz, 0
  %vb = or i32 %v, %b0
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %vb, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @oob, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Indices past a 64-byte threadgroup array in the arena (a byte read of its words keeps it there)
/// are contained (stores dropped, loads 0), and with containment off (`UnsafeOff`, isolation off)
/// no check is emitted.
#[test]
fn out_of_bounds_threadgroup_indices_are_contained() {
    let Some(bc) = assemble("oob", OOB) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "oob", &prof, stem);
        let (plan, m) = typed_module(&bc, "oob", &prof).expect("module");
        assert!(plan.families.access.local.typed.is_empty(), "{stem}");
        let census = plan.census();
        assert!(census.contains_key("access.threadgroup.store.checked"));
        assert!(census.contains_key("access.threadgroup.load.checked"));
        let res = resources(&plan, 0, 64, 64, &[]);
        let want: Vec<u32> = (0..64u32).map(|t| if t < 16 { t } else { 0 }).collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");

        let unsafe_off = DevicePolicy {
            containment: ContainmentPolicy::UnsafeOff,
            isolation: GuestIsolation::Off,
            ..Default::default()
        };
        let prof = profile(stem, unsafe_off).expect("profile");
        l3(&bc, "oob", &prof, stem);
        let (plan, _) = typed_module(&bc, "oob", &prof).expect("module");
        let census = plan.census();
        assert!(
            census.contains_key("access.threadgroup.store"),
            "{census:?}"
        );
        assert!(census.contains_key("access.threadgroup.load"), "{census:?}");
    }
}

const KERNEL_MD: &str = r#"!air.kernel = !{!0}
!0 = !{ptr @NAME, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A kernel `@name(out, t)` running `body`, which writes `out[2t]` and `out[2t + 1]` from `%v0`
/// and `%v1`.
fn two_word_kernel(name: &str, globals: &str, body: &str) -> String {
    format!(
        r#"{globals}
define void @{name}(ptr addrspace(1) %out, i32 %t) {{
{body}
  %i0 = shl i32 %t, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i0
  store i32 %v0, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %i0, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %v1, ptr addrspace(1) %o1, align 4
  ret void
}}

{}"#,
        KERNEL_MD.replace("NAME", name)
    )
}

/// The body of the typed-alloca kernel: `a: [8 x i32]` initialized to `10 + i`, a store of 999
/// at `t + 4` (out of range for `t >= 4`), a store of `100 t` at `t & 7`, then `v0 = a[(t & 7) ^
/// 5]` and `x = a[t]` (out of range for `t >= 8`); `b: [4 x [3 x i32]]` (a 12-byte stride) is
/// initialized to its flat index and `v1 = x + b[t & 3][2]`.
fn typed_alloca_body() -> String {
    let mut s =
        String::from("  %a = alloca [8 x i32], align 4\n  %b = alloca [4 x [3 x i32]], align 4\n");
    for i in 0..8 {
        s += &format!(
            "  %ia{i} = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 {i}\n  store i32 {}, ptr %ia{i}, align 4\n",
            10 + i
        );
    }
    for i in 0..4 {
        for j in 0..3 {
            s += &format!(
                "  %ib{i}{j} = getelementptr inbounds [4 x [3 x i32]], ptr %b, i32 0, i32 {i}, i32 {j}\n  store i32 {}, ptr %ib{i}{j}, align 4\n",
                3 * i + j
            );
        }
    }
    s += r#"  %t4 = add i32 %t, 4
  %p4 = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %t4
  store i32 999, ptr %p4, align 4
  %k = and i32 %t, 7
  %pk = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %k
  %h = mul i32 %t, 100
  store i32 %h, ptr %pk, align 4
  %j = xor i32 %k, 5
  %pj = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %j
  %v0 = load i32, ptr %pj, align 4
  %pt = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %t
  %x = load i32, ptr %pt, align 4
  %m = and i32 %t, 3
  %pm = getelementptr inbounds [4 x [3 x i32]], ptr %b, i32 0, i32 %m, i32 2
  %y = load i32, ptr %pm, align 4
  %v1 = add i32 %x, %y"#;
    s
}

/// A typed alloca is a `Private` array indexed by access chains; its dynamic indices are checked
/// under isolation (an out-of-range load is 0, an out-of-range store is dropped) and exact in
/// range, including a 12-byte (non-power-of-two) stride. Without isolation no check is emitted.
#[test]
fn typed_alloca_indices_are_clamped_under_isolation() {
    let Some(bc) = assemble("typed", &two_word_kernel("typed", "", &typed_alloca_body())) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "typed", &prof, stem);
        let (plan, m) = typed_module(&bc, "typed", &prof).expect("module");
        let local = &plan.families.access.local;
        assert_eq!(local.typed.len(), 2, "{stem}");
        assert_eq!(local.private, None, "{stem}: no private arena");
        let census = plan.census();
        for name in [
            "access.typed.load.clamped",
            "access.typed.store.clamped",
            "access.typed.store",
        ] {
            assert!(census.contains_key(name), "{stem}: {name} in {census:?}");
        }
        let res = resources(&plan, 0, 64, 32, &[]);
        let want: Vec<u32> = (0..32u32)
            .flat_map(|t| {
                let mut a: Vec<u32> = (10..18).collect();
                if t + 4 < 8 {
                    a[(t + 4) as usize] = 999;
                }
                a[(t & 7) as usize] = t * 100;
                let v0 = a[((t & 7) ^ 5) as usize];
                let x = if t < 8 { a[t as usize] } else { 0 };
                [v0, x + 3 * (t & 3) + 2]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");

        let off = DevicePolicy {
            isolation: GuestIsolation::Off,
            ..Default::default()
        };
        let prof = profile(stem, off).expect("profile");
        l3(&bc, "typed", &prof, stem);
        let (plan, _) = typed_module(&bc, "typed", &prof).expect("module");
        let census = plan.census();
        assert!(census.contains_key("access.typed.load"), "{census:?}");
        assert!(
            !census.contains_key("access.typed.load.clamped"),
            "{census:?}"
        );
        assert!(
            !census.contains_key("access.typed.store.clamped"),
            "{census:?}"
        );
    }
}

/// The body of the nested-levels kernel: `c: [3 x [5 x i32]]` (strides 20 and 4) initialized to
/// `100 + 5i + j`, a store of `900 + t` at `c[t % 3][7t % 5]`, a store of 777 at `c[t][0]` (out
/// of range from `t = 3`), `x = c[(t + 1) % 3][3t % 5]` and `z = c[t][t]` (out of range from `t
/// = 3`), `v0 = x + z`; `d: [2 x {i32, [3 x i32]}]` (the inner array 4 bytes into a 16-byte
/// element) with `d[i].1[j] = 60 + 3i + j`, and `e: [2 x [2 x <2 x i32>]]` with `e[i][j] = <i,
/// 10 + 2i + j>`; `v1 = d[t & 1].1[t % 3] + e[t & 1][(t >> 1) & 1].y`.
fn nested_levels_body() -> String {
    let mut s = String::from(
        "  %c = alloca [3 x [5 x i32]], align 4\n  %d = alloca [2 x { i32, [3 x i32] }], align 4\n  %e = alloca [2 x [2 x <2 x i32>]], align 8\n",
    );
    for i in 0..3 {
        for j in 0..5 {
            s += &format!(
                "  %ic{i}{j} = getelementptr inbounds [3 x [5 x i32]], ptr %c, i32 0, i32 {i}, i32 {j}\n  store i32 {}, ptr %ic{i}{j}, align 4\n",
                100 + 5 * i + j
            );
        }
    }
    for i in 0..2 {
        s += &format!(
            "  %id{i} = getelementptr inbounds [2 x {{ i32, [3 x i32] }}], ptr %d, i32 0, i32 {i}, i32 0\n  store i32 {}, ptr %id{i}, align 4\n",
            50 + i
        );
        for j in 0..3 {
            s += &format!(
                "  %id{i}{j} = getelementptr inbounds [2 x {{ i32, [3 x i32] }}], ptr %d, i32 0, i32 {i}, i32 1, i32 {j}\n  store i32 {}, ptr %id{i}{j}, align 4\n",
                60 + 3 * i + j
            );
        }
        for j in 0..2 {
            s += &format!(
                "  %ie{i}{j} = getelementptr inbounds [2 x [2 x <2 x i32>]], ptr %e, i32 0, i32 {i}, i32 {j}\n  store <2 x i32> <i32 {i}, i32 {}>, ptr %ie{i}{j}, align 8\n",
                10 + 2 * i + j
            );
        }
    }
    s += r#"  %ci = urem i32 %t, 3
  %t7 = mul i32 %t, 7
  %cj = urem i32 %t7, 5
  %pc = getelementptr inbounds [3 x [5 x i32]], ptr %c, i32 0, i32 %ci, i32 %cj
  %w = add i32 %t, 900
  store i32 %w, ptr %pc, align 4
  %po = getelementptr inbounds [3 x [5 x i32]], ptr %c, i32 0, i32 %t, i32 0
  store i32 777, ptr %po, align 4
  %t1 = add i32 %t, 1
  %xi = urem i32 %t1, 3
  %t3 = mul i32 %t, 3
  %xj = urem i32 %t3, 5
  %px = getelementptr inbounds [3 x [5 x i32]], ptr %c, i32 0, i32 %xi, i32 %xj
  %x = load i32, ptr %px, align 4
  %pz = getelementptr inbounds [3 x [5 x i32]], ptr %c, i32 0, i32 %t, i32 %t
  %z = load i32, ptr %pz, align 4
  %v0 = add i32 %x, %z
  %di = and i32 %t, 1
  %dj = urem i32 %t, 3
  %pd = getelementptr inbounds [2 x { i32, [3 x i32] }], ptr %d, i32 0, i32 %di, i32 1, i32 %dj
  %y = load i32, ptr %pd, align 4
  %t2 = lshr i32 %t, 1
  %ej = and i32 %t2, 1
  %pe = getelementptr inbounds [2 x [2 x <2 x i32>]], ptr %e, i32 0, i32 %di, i32 %ej
  %ev = load <2 x i32>, ptr %pe, align 8
  %e1 = extractelement <2 x i32> %ev, i32 1
  %v1 = add i32 %y, %e1"#;
    s
}

/// An alloca indexed dynamically at several nested array levels (non-power-of-two strides, a struct
/// member between the levels, vector leaves) is typed: an access chain whose indices are the
/// accesses' own GEP indices, exact in range, and checked at every level under isolation (an
/// out-of-range load is 0, an out-of-range store is dropped). Without isolation no check is
/// emitted.
#[test]
fn typed_allocas_index_nested_levels_dynamically() {
    let Some(bc) = assemble(
        "nested",
        &two_word_kernel("nested", "", &nested_levels_body()),
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "nested", &prof, stem);
        let (plan, m) = typed_module(&bc, "nested", &prof).expect("module");
        let local = &plan.families.access.local;
        assert_eq!(local.typed.len(), 3, "{stem}");
        assert_eq!(local.private, None, "{stem}: no private arena");
        let census = plan.census();
        for name in ["access.typed.load.clamped", "access.typed.store.clamped"] {
            assert!(census.contains_key(name), "{stem}: {name} in {census:?}");
        }
        let res = resources(&plan, 0, 64, 32, &[]);
        let want: Vec<u32> = (0..32u32)
            .flat_map(|t| {
                let mut c: Vec<Vec<u32>> = (0..3)
                    .map(|i| (0..5).map(|j| 100 + 5 * i + j).collect())
                    .collect();
                c[(t % 3) as usize][((7 * t) % 5) as usize] = 900 + t;
                if t < 3 {
                    c[t as usize][0] = 777;
                }
                let x = c[((t + 1) % 3) as usize][((3 * t) % 5) as usize];
                let z = if t < 3 { c[t as usize][t as usize] } else { 0 };
                let (di, dj, ej) = (t & 1, t % 3, (t >> 1) & 1);
                [x + z, 60 + 3 * di + dj + 10 + 2 * di + ej]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");

        let off = DevicePolicy {
            isolation: GuestIsolation::Off,
            ..Default::default()
        };
        let prof = profile(stem, off).expect("profile");
        l3(&bc, "nested", &prof, stem);
        let (plan, _) = typed_module(&bc, "nested", &prof).expect("module");
        assert_eq!(plan.families.access.local.private, None, "{stem}");
        let census = plan.census();
        assert!(census.contains_key("access.typed.load"), "{census:?}");
        assert!(
            !census.contains_key("access.typed.load.clamped"),
            "{census:?}"
        );
    }
}

/// An alloca of `simdgroup_matrix`-like structs (`[2 x { <8 x float> }]`, whose eight-lane member
/// is a chunked vector no typed object holds) is accessed only by plain loads and stores, yet has
/// its place in the private arena, whose leaves hold a chunked vector: a store of a vector built
/// from `t` into element 0, of constants into element 1, and a load of element `t & 1` read back
/// lanes 3 and 6 exactly.
#[test]
fn an_alloca_of_chunked_vectors_lives_in_the_private_arena() {
    let body = r#"  %a = alloca [2 x { <8 x float> }], align 32
  %f = uitofp i32 %t to float
  %v = insertelement <8 x float> <float 0.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0>, float %f, i32 3
  %p0 = getelementptr inbounds [2 x { <8 x float> }], ptr %a, i32 0, i32 0, i32 0
  store <8 x float> %v, ptr %p0, align 32
  %p1 = getelementptr inbounds [2 x { <8 x float> }], ptr %a, i32 0, i32 1, i32 0
  store <8 x float> <float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0, float 17.0>, ptr %p1, align 32
  %k = and i32 %t, 1
  %pk = getelementptr inbounds [2 x { <8 x float> }], ptr %a, i32 0, i32 %k, i32 0
  %w = load <8 x float>, ptr %pk, align 32
  %e3 = extractelement <8 x float> %w, i32 3
  %e6 = extractelement <8 x float> %w, i32 6
  %v0 = bitcast float %e3 to i32
  %v1 = bitcast float %e6 to i32"#;
    let Some(bc) = assemble("chunked", &two_word_kernel("chunked", "", body)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "chunked", &prof, stem);
        let (plan, m) = typed_module(&bc, "chunked", &prof).expect("module");
        let local = &plan.families.access.local;
        assert!(local.typed.is_empty(), "{stem}");
        assert_eq!(local.private, Some(64), "{stem}");
        let res = resources(&plan, 0, 64, 32, &[]);
        let want: Vec<u32> = (0..32u32)
            .flat_map(|t| {
                let (e3, e6) = if t & 1 == 0 {
                    (t as f32, 6.0f32)
                } else {
                    (13.0, 16.0)
                };
                [e3.to_bits(), e6.to_bits()]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

const TABLES: &str = r#"@tab = internal addrspace(2) constant [4 x i32] [i32 7, i32 11, i32 13, i32 17], align 4
@pairs = internal addrspace(2) constant [3 x { i32, i32 }] [{ i32, i32 } { i32 1, i32 2 }, { i32, i32 } { i32 3, i32 4 }, { i32, i32 } { i32 5, i32 6 }], align 4
"#;

/// A closed constant global is a `Private` variable with its initializer; a dynamic index into it
/// is clamped under isolation (out of range reads the level's last element), including a member
/// below the dynamic level.
#[test]
fn constant_private_arrays_are_clamped_under_isolation() {
    let body = r#"  %p = getelementptr inbounds [4 x i32], ptr addrspace(2) @tab, i32 0, i32 %t
  %v0 = load i32, ptr addrspace(2) %p, align 4
  %q = getelementptr inbounds [3 x { i32, i32 }], ptr addrspace(2) @pairs, i32 0, i32 %t, i32 1
  %v1 = load i32, ptr addrspace(2) %q, align 4"#;
    let Some(bc) = assemble("tables", &two_word_kernel("tables", TABLES, body)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "tables", &prof, stem);
        let (plan, m) = typed_module(&bc, "tables", &prof).expect("module");
        assert_eq!(plan.families.access.local.typed.len(), 2, "{stem}");
        let census = plan.census();
        assert!(
            census.contains_key("access.constant.load.clamped"),
            "{stem}: {census:?}"
        );
        let res = resources(&plan, 0, 32, 16, &[]);
        let tab = [7u32, 11, 13, 17];
        let pairs = [2u32, 4, 6];
        let want: Vec<u32> = (0..16usize)
            .flat_map(|t| [tab[t.min(3)], pairs[t.min(2)]])
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// Pointers into one arena or one typed object compare by offset; an object pointer is never null.
/// A loop whose pointer walks a private array until it meets the end pointer runs its four
/// iterations.
#[test]
fn arena_pointers_compare_by_offset() {
    let globals = "@g = internal addrspace(3) global [8 x i32] undef, align 4\n\
                   declare void @air.wg.barrier(i32, i32)";
    let body = r#"  %a = alloca [4 x i32], align 4
  %b = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 0
  %e = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 4
  br label %loop
loop:
  %p = phi ptr [ %b, %0 ], [ %n, %loop ]
  %k = phi i32 [ 0, %0 ], [ %k1, %loop ]
  %kt = mul i32 %k, %t
  store i32 %kt, ptr %p, align 4
  %n = getelementptr inbounds i32, ptr %p, i32 1
  %k1 = add i32 %k, 1
  %done = icmp eq ptr %n, %e
  br i1 %done, label %exit, label %loop
exit:
  %j = and i32 %t, 3
  %q = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %j
  %w = load i32, ptr %q, align 4
  %three = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 3
  %c3 = icmp eq ptr %q, %three
  %nn = icmp ne ptr %q, null
  %lt = icmp ult ptr %b, %q
  %z3 = zext i1 %c3 to i32
  %zn = zext i1 %nn to i32
  %zl = zext i1 %lt to i32
  %s1 = shl i32 %zn, 1
  %s2 = shl i32 %zl, 2
  %f0 = or i32 %z3, %s1
  %f1 = or i32 %f0, %s2
  %w8 = shl i32 %w, 8
  %v0 = or i32 %w8, %f1
  %gt = getelementptr inbounds [8 x i32], ptr addrspace(3) @g, i32 0, i32 %t
  %g5 = getelementptr inbounds [8 x i32], ptr addrspace(3) @g, i32 0, i32 5
  %c5 = icmp eq ptr addrspace(3) %gt, %g5
  %v1 = zext i1 %c5 to i32"#;
    let Some(bc) = assemble(
        "compares",
        &two_word_kernel("compares", globals, &body.replace("%0", "%entry0")).replace(
            "define void @compares(ptr addrspace(1) %out, i32 %t) {\n",
            "define void @compares(ptr addrspace(1) %out, i32 %t) {\nentry0:\n",
        ),
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "compares", &prof, stem);
        let (plan, m) = typed_module(&bc, "compares", &prof).expect("module");
        let census = plan.census();
        assert!(
            census.contains_key("access.private.pointer_compare")
                && census.contains_key("access.threadgroup.pointer_compare")
                && census.contains_key("access.private.null_compare"),
            "{stem}: {census:?}"
        );
        let res = resources(&plan, 0, 32, 16, &[]);
        let want: Vec<u32> = (0..16u32)
            .flat_map(|t| {
                let j = t & 3;
                let flags = u32::from(j == 3) | 2 | (u32::from(j > 0) << 2);
                [(j * t) << 8 | flags, u32::from(t == 5)]
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// A `select` between two threadgroup globals' addresses selects their arena offsets.
#[test]
fn a_select_of_threadgroup_globals_selects_their_offsets() {
    let globals = "@g1 = internal addrspace(3) global [16 x i32] undef, align 4\n\
                   @g2 = internal addrspace(3) global [16 x i32] undef, align 4";
    let body = r#"  %odd = trunc i32 %t to i1
  %s1 = select i1 %odd, ptr addrspace(3) @g1, ptr addrspace(3) @g2
  %s2 = select i1 %odd, ptr addrspace(3) @g2, ptr addrspace(3) @g1
  %q1 = getelementptr inbounds [16 x i32], ptr addrspace(3) %s1, i32 0, i32 %t
  %a = add i32 %t, 100
  store i32 %a, ptr addrspace(3) %q1, align 4
  %q2 = getelementptr inbounds [16 x i32], ptr addrspace(3) %s2, i32 0, i32 %t
  %b = add i32 %t, 200
  store i32 %b, ptr addrspace(3) %q2, align 4
  %r1 = getelementptr inbounds [16 x i32], ptr addrspace(3) @g1, i32 0, i32 %t
  %v0 = load i32, ptr addrspace(3) %r1, align 4
  %r2 = getelementptr inbounds [16 x i32], ptr addrspace(3) @g2, i32 0, i32 %t
  %v1 = load i32, ptr addrspace(3) %r2, align 4"#;
    let Some(bc) = assemble("tg_select", &two_word_kernel("tg_select", globals, body)) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "tg_select", &prof, stem);
        let (plan, m) = typed_module(&bc, "tg_select", &prof).expect("module");
        let res = resources(&plan, 0, 32, 16, &[]);
        let want: Vec<u32> = (0..16u32)
            .flat_map(|t| {
                if t % 2 == 1 {
                    [t + 100, t + 200]
                } else {
                    [t + 200, t + 100]
                }
            })
            .collect();
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// Constant `getelementptr`s address their exact displacement from their global: into a threadgroup
/// global placed before a second one (in the arena, where a byte read of its words keeps it),
/// through a nested aggregate, with a negative index, and into a closed constant global in its
/// `Private` form.
#[test]
fn constant_geps_address_their_displacement() {
    let globals = r#"@pad = internal addrspace(3) global [3 x i32] undef, align 4
@g = internal addrspace(3) global [2 x { i32, [4 x i32] }] undef, align 4
@tab = internal addrspace(2) constant [4 x <2 x i32>] [<2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>, <2 x i32> <i32 5, i32 6>, <2 x i32> <i32 7, i32 8>], align 8
declare void @air.wg.barrier(i32, i32)"#;
    let body = r#"  %small = icmp ult i32 %t, 10
  br i1 %small, label %fill, label %wait
fill:
  %o = udiv i32 %t, 5
  %i = urem i32 %t, 5
  %isz = icmp eq i32 %i, 0
  %im = sub i32 %i, 1
  %v = mul i32 %t, 3
  br i1 %isz, label %head, label %body
head:
  %ph = getelementptr inbounds [2 x { i32, [4 x i32] }], ptr addrspace(3) @g, i32 0, i32 %o, i32 0
  store i32 %v, ptr addrspace(3) %ph, align 4
  br label %wait
body:
  %pb = getelementptr inbounds [2 x { i32, [4 x i32] }], ptr addrspace(3) @g, i32 0, i32 %o, i32 1, i32 %im
  store i32 %v, ptr addrspace(3) %pb, align 4
  br label %wait
wait:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %x = load i32, ptr addrspace(3) getelementptr inbounds ([2 x { i32, [4 x i32] }], ptr addrspace(3) @g, i32 0, i32 1, i32 1, i32 2), align 4
  %y = load i32, ptr addrspace(3) getelementptr inbounds (i32, ptr addrspace(3) getelementptr inbounds ([2 x { i32, [4 x i32] }], ptr addrspace(3) @g, i32 0, i32 1, i32 0), i32 -2), align 4
  %xy = shl i32 %x, 8
  %v0 = or i32 %xy, %y
  %c = load <2 x i32>, ptr addrspace(2) getelementptr inbounds ([4 x <2 x i32>], ptr addrspace(2) @tab, i32 0, i32 2), align 8
  %d = load i32, ptr addrspace(2) getelementptr inbounds (i32, ptr addrspace(2) getelementptr inbounds ([4 x <2 x i32>], ptr addrspace(2) @tab, i32 0, i32 3), i32 -1), align 4
  %c1 = extractelement <2 x i32> %c, i32 1
  %cd = shl i32 %c1, 8
  %v1p = or i32 %cd, %d
  %gb = load i8, ptr addrspace(3) @g, align 4
  %gz = zext i8 %gb to i32
  %g0 = and i32 %gz, 0
  %v1 = or i32 %v1p, %g0"#;
    let Some(bc) = assemble("const_geps", &two_word_kernel("const_geps", globals, body)) else {
        return;
    };
    // `@g` is at arena offset 0 (Metal's layout places the last-declared global first; `@pad`
    // follows at 40): element 1 starts at byte 20 of it, member 1 at 24. `x` is g[1].1[2] = thread
    // 5 * 1 + 3 = 8's value 24; `y` is the word 2 below g[1].0 (thread 5's), which is g[0].1[2] =
    // thread 3's value 9.
    let (x, y) = (8 * 3, 3 * 3);
    // `c` is tab[2] = <5, 6>, `d` the word before tab[3] = tab[2][1] = 6.
    let want: Vec<u32> = (0..16u32)
        .flat_map(|_| [(x << 8) | y, (6 << 8) | 6])
        .collect();
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "const_geps", &prof, stem);
        let (plan, m) = typed_module(&bc, "const_geps", &prof).expect("module");
        assert_eq!(
            plan.families.access.local.typed.len(),
            1,
            "{stem}: `@tab` is a typed constant"
        );
        let res = resources(&plan, 0, 32, 16, &[]);
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// How the pointers between a typed alloca and its accesses are made ([`pointer_levels`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Levels {
    /// Each level reads the last twice: `%p{i+1} = select %c, %p{i}, %q{i}`, a DAG of shared
    /// operands with `2^depth` paths.
    Shared,
    /// Each level reads the last once: `%p{i+1} = getelementptr i8, %q{i}, 0`, one path.
    Chain,
}

/// A kernel whose typed alloca `a: [8 x i32]` is addressed through `depth` levels of pointers
/// from `%p0 = &a[t & 7]`, each level `%q{i} = getelementptr i8, %p{i}, 0` and `%p{i+1}` as
/// `levels` says. `stores` stores and one load go through `%p{depth}` when `through_end`, else
/// through `%p0`; `v1` compares the two, so every level is live either way.
fn pointer_levels(levels: Levels, depth: u32, stores: u32, through_end: bool) -> String {
    let mut s = String::from(
        "  %a = alloca [8 x i32], align 4\n  %k = and i32 %t, 7\n  %p0 = getelementptr inbounds [8 x i32], ptr %a, i32 0, i32 %k\n  %c = icmp ult i32 %t, 16\n",
    );
    for i in 0..depth {
        let n = i + 1;
        s += &format!("  %q{i} = getelementptr inbounds i8, ptr %p{i}, i32 0\n");
        s += &match levels {
            Levels::Shared => format!("  %p{n} = select i1 %c, ptr %p{i}, ptr %q{i}\n"),
            Levels::Chain => format!("  %p{n} = getelementptr inbounds i8, ptr %q{i}, i32 0\n"),
        };
    }
    let at = if through_end { depth } else { 0 };
    for j in 0..stores {
        s += &format!("  store i32 {j}, ptr %p{at}, align 4\n");
    }
    s += &format!(
        "  %v0 = load i32, ptr %p{at}, align 4\n  %e = icmp eq ptr %p{depth}, %p0\n  %v1 = zext i1 %e to i32"
    );
    two_word_kernel("levels", "", &s)
}

/// The work units of compiling `pointer_levels(...)` for `lavapipe` under a meter capped at
/// `cap` work units, or its refusal.
fn levels_work(
    levels: Levels,
    depth: u32,
    stores: u32,
    through_end: bool,
    cap: u64,
) -> Option<Result<u64, Error>> {
    let name = format!("levels_{levels:?}_{depth}_{stores}_{through_end}");
    let bc = assemble(&name, &pointer_levels(levels, depth, stores, through_end))?;
    let mut m = Meter::with_caps(&Budget::default(), cap, LEDGER_BYTES);
    Some(
        compile_bitcode(
            &bc,
            EntryRef::Name("levels"),
            &FunctionConstantValues::defaults(),
            &p("lavapipe"),
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        )
        .map(|_| m.work()),
    )
}

/// The offset-shape walk of a typed access is metered by the definitions it reads. An access
/// through the end of a 100-level chain reads its 200 pointer definitions, so it charges at least
/// that much more work than the same kernel accessing through the chain's start (every level live
/// in both). Previously the walk ticked nothing and the two charged the same.
#[test]
fn a_typed_access_shape_walk_is_metered_by_the_definitions_it_reads() {
    const DEPTH: u32 = 100;
    let work = |through_end| levels_work(Levels::Chain, DEPTH, 1, through_end, WORK_UNITS);
    let (Some(end), Some(start)) = (work(true), work(false)) else {
        return;
    };
    let (end, start) = (end.expect("translates"), start.expect("translates"));
    assert!(
        end >= start + u64::from(2 * DEPTH),
        "{end} work units through the end against {start} through the start: the walk over \
         {} pointer definitions is not charged",
        2 * DEPTH
    );
}

/// Each definition's shape is read once per function. 32 more stores through the end of a 100-level
/// chain cost what 32 more stores through the end of a 1-level chain do (within a unit each): every
/// access after the first finds the pointer's shape read. Read per access, each would charge the
/// 200-definition walk again. (Both ends are `getelementptr i8` steps, so both chains' accesses
/// are emitted alike.)
#[test]
fn a_pointer_shape_is_read_once_per_function() {
    let work = |depth, stores| {
        levels_work(Levels::Chain, depth, stores, true, WORK_UNITS).map(|w| w.expect("translates"))
    };
    let (Some(deep1), Some(deep33), Some(short1), Some(short33)) =
        (work(100, 1), work(100, 33), work(1, 1), work(1, 33))
    else {
        return;
    };
    let (deep, short) = (deep33 - deep1, short33 - short1);
    assert!(
        deep <= short + 32,
        "32 more stores cost {deep} work units through a 100-level chain's end and {short} \
         through a 1-level chain's: the walk is repeated per access"
    );
}

/// A DAG of selects whose arms share their operands (`2^40` paths through its 80 pointer
/// definitions) translates under a cap of twice the work of the 40-level chain, which has as many
/// definitions on one path: the walk reads each definition once. Previously the walk followed every
/// path: unmetered, it did not finish (a 22-level DAG took 12 s in a test build, four times more
/// per two levels); metered without reading each definition once, it would pass any cap here.
#[test]
fn a_shared_pointer_dag_is_read_in_linear_work() {
    const DEPTH: u32 = 40;
    let Some(chain) = levels_work(Levels::Chain, DEPTH, 1, true, WORK_UNITS) else {
        return;
    };
    let chain = chain.expect("the chain translates");
    let shared = levels_work(Levels::Shared, DEPTH, 1, true, 2 * chain).expect("assembled");
    assert!(
        shared.is_ok(),
        "the {DEPTH}-level DAG under twice the chain's {chain} work units: {shared:?}"
    );
}

/// The walk's bound (`SHAPE_WALK`, 256 definitions open at once) exactly where it was: the deepest
/// path of a 127-level DAG holds 256 definitions (the 254 levels, `%p0`, and the alloca) and
/// translates; a 128-level DAG's holds 258 and is refused as an access the family does not lower,
/// though every definition on that path was read before on a shallower one. An access through `%p0`
/// of the same 128-level DAG translates (the bound is the walk's).
#[test]
fn the_shape_walk_bound_is_unchanged_by_reading_each_definition_once() {
    for levels in [Levels::Shared, Levels::Chain] {
        let Some(deepest) = levels_work(levels, 127, 1, true, WORK_UNITS) else {
            return;
        };
        assert!(deepest.is_ok(), "{levels:?} 127: {deepest:?}");
        match levels_work(levels, 128, 1, true, WORK_UNITS).expect("assembled") {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::Lowering(LowerFamily::Access),
                    "{levels:?}"
                );
                assert_eq!(u.needs, Needs::NotImplemented, "{levels:?}");
            }
            other => panic!("{levels:?} 128: expected a refusal, got {other:?}"),
        }
        let start = levels_work(levels, 128, 1, false, WORK_UNITS).expect("assembled");
        assert!(start.is_ok(), "{levels:?} 128 through the start: {start:?}");
    }
}

/// Whether a type holds a pointer is the type table's, kept per type at interning, so planning an
/// alloca of a type whose struct members share types costs one lookup. `%e{k}` is two `%e{k-1}`
/// (empty at the bottom, so nothing is placed): 40 levels are `2^40` paths through 41 types.
/// Previously the memory plan walked every path to classify the alloca (and every `insertvalue` or
/// `extractvalue` of such a type), unmetered: it did not finish.
#[test]
fn an_alloca_of_a_shared_struct_dag_is_planned_in_linear_work() {
    const DEPTH: u32 = 40;
    let mut types = String::from("%e0 = type {}\n");
    for k in 1..=DEPTH {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    let body = format!(
        "  %a = alloca %e{DEPTH}, align 4\n  %n = icmp eq ptr %a, null\n  %v0 = zext i1 %n to i32\n  %v1 = add i32 %t, 1"
    );
    let Some(bc) = assemble("struct_dag", &two_word_kernel("struct_dag", &types, &body)) else {
        return;
    };
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
    let compiled = compile_bitcode(
        &bc,
        EntryRef::Name("struct_dag"),
        &FunctionConstantValues::defaults(),
        &p("lavapipe"),
        &LowerInputs::new(),
        &mut m,
        SPIRV_WORDS,
    );
    assert!(compiled.is_ok(), "{compiled:?}");
}

/// The shapes the walk records are released when the instruction walk ends. An access through the
/// end of a 100-level chain records 202 shapes, one through its start 2, and the two compiles end
/// with the same bytes on the ledger: the memo's charge is gone. Kept to the end of the compile,
/// the 200 more entries would stay charged (25,600 bytes here).
#[test]
fn the_recorded_shapes_are_released_when_selection_ends() {
    const DEPTH: u32 = 100;
    let left = |through_end| -> Option<u64> {
        let name = format!("levels_released_{through_end}");
        let bc = assemble(&name, &pointer_levels(Levels::Chain, DEPTH, 1, through_end))?;
        let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
        compile_bitcode(
            &bc,
            EntryRef::Name("levels"),
            &FunctionConstantValues::defaults(),
            &p("lavapipe"),
            &LowerInputs::new(),
            &mut m,
            SPIRV_WORDS,
        )
        .expect("translates");
        Some(m.ledger())
    };
    let (Some(end), Some(start)) = (left(true), left(false)) else {
        return;
    };
    assert_eq!(
        end, start,
        "{end} ledger bytes are left after compiling through the chain's end and {start} through \
         its start: the recorded shapes stay charged after selection"
    );
}

/// A kernel whose alloca `a` is a struct DAG `depth` levels deep: `%e1 = { i32, i32 }` and
/// `%e{k} = { %e{k-1}, %e{k-1} }`, `2^depth` leaves through `depth + 1` types. It stores `t` at
/// byte 0 and 7 at byte 4, then `v0` reads byte 0 and `v1` byte 4. With `constant`, `v0` also
/// adds word 1 of `@g`, a closed constant of the same type, all zero, read at its byte address
/// (SCCP reads an integer field at a typed address as its constant, which would leave no load of
/// `@g` to plan). Every access is a plain `i32` load or store at a constant offset, so `a` and
/// `@g` are typed objects within the function's room of unshared type size.
fn struct_dag_kernel(depth: u32, constant: bool) -> String {
    let mut types = String::from("%e1 = type { i32, i32 }\n");
    for k in 2..=depth {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    let read = if constant {
        types +=
            &format!("@g = internal addrspace(2) constant %e{depth} zeroinitializer, align 4\n");
        "  %x = load i32, ptr %a, align 4\n  %gq = getelementptr inbounds i8, ptr addrspace(2) @g, i32 4\n  %y = load i32, ptr addrspace(2) %gq, align 4\n  %v0 = add i32 %x, %y"
    } else {
        "  %v0 = load i32, ptr %a, align 4"
    };
    let body = format!(
        r#"  %a = alloca %e{depth}, align 4
  store i32 %t, ptr %a, align 4
  %q = getelementptr inbounds i8, ptr %a, i32 4
  store i32 7, ptr %q, align 4
{read}
  %v1 = load i32, ptr %q, align 4"#
    );
    two_word_kernel("struct_dag", &types, &body)
}

/// The work units and ledger peak of compiling `struct_dag_kernel(depth, false)` for `lavapipe`,
/// and its outcome.
fn struct_dag_cost(depth: u32) -> Option<(u64, u64, Result<(), Error>)> {
    let bc = assemble(
        &format!("struct_dag_{depth}"),
        &struct_dag_kernel(depth, false),
    )?;
    let mut m = Meter::with_caps(&Budget::default(), WORK_UNITS, LEDGER_BYTES);
    let compiled = compile_bitcode(
        &bc,
        EntryRef::Name("struct_dag"),
        &FunctionConstantValues::defaults(),
        &p("lavapipe"),
        &LowerInputs::new(),
        &mut m,
        SPIRV_WORDS,
    );
    Some((m.work(), m.ledger_peak(), compiled.map(|_| ())))
}

/// A typed object's type is built once per distinct AIR type, so planning an alloca of a struct DAG
/// costs work and ledger bytes linear in its levels however many leaves it has. At 30, 35, and 40
/// levels (`2^30` to `2^40` leaves, 4 GiB and up) the alloca is past the function's typed room, so
/// it has its place in the private arena, which a `u32` offset cannot address: each is
/// `LimitExceeded(PrivateArenaBytes)`, five more levels costing what five more do (within a few
/// units). Previously the shape was a tree of every path, unmetered and uncharged: 22 levels took
/// 11.2 s and 408 MB resident in a test build against a 44 KB ledger peak, and each level doubles
/// both. Metered without being built once per type, 30 levels pass the ledger cap.
#[test]
fn a_struct_dag_is_planned_in_linear_work() {
    let (Some(c30), Some(c35), Some(c40)) = (
        struct_dag_cost(30),
        struct_dag_cost(35),
        struct_dag_cost(40),
    ) else {
        return;
    };
    for (depth, (_, _, outcome)) in [(30, &c30), (35, &c35), (40, &c40)] {
        assert_eq!(
            outcome,
            &Err(Error::LimitExceeded(Limit::PrivateArenaBytes)),
            "{depth} levels: expected the arena's bound"
        );
    }
    let ((w30, l30, _), (w35, l35, _), (w40, l40, _)) = (c30, c35, c40);
    assert!(
        w40 - w35 <= w35 - w30 + 8,
        "work units {w30}, {w35}, {w40} at 30, 35, 40 levels: not linear in the levels"
    );
    assert!(
        l40 - l35 <= l35 - l30 + 1024,
        "ledger peaks {l30}, {l35}, {l40} at 30, 35, 40 levels: not linear in the levels"
    );
}

/// A struct DAG within the function's room is typed, and keeps its values: `a`'s two words read
/// back what was stored, and `@g` (all zero) adds nothing. At 14 levels `@g` (planned first) takes
/// the room and `a` has its place in the private arena; at 15 levels neither fits, so `@g` is in
/// the constant blob too. Each passes L3, and the values are the same typed (3 levels) and untyped
/// (15 levels). The CPU interpreter walks a struct type's every path to lay it out, so it does not
/// run the 14-level typed `@g`. A profile whose Function storage is less than the arena refuses it,
/// needing that much.
#[test]
fn a_struct_dag_keeps_its_values_typed_or_untyped() {
    // (levels, typed objects, private arena bytes, blob bytes, run on the interpreter)
    let cases = [
        (3u32, 2usize, None, None, true),
        (14, 1, Some(4u64 << 14), None, false),
        (15, 0, Some(4 << 15), Some(4usize << 15), true),
    ];
    for (depth, typed, private, blob, interpret) in cases {
        let Some(bc) = assemble(
            &format!("struct_dag_values_{depth}"),
            &struct_dag_kernel(depth, true),
        ) else {
            return;
        };
        for stem in PROFILES {
            let prof = p(stem);
            let storage = prof.fields().function_storage.bytes();
            if private.is_some_and(|b| b > storage) {
                match compile(&bc, "struct_dag", &prof) {
                    Err(Error::Unsupported(u)) if u.feature == Feature::PrivateMemory => {}
                    other => panic!("{depth}, {stem}: {:?}", other.map(|_| ())),
                }
                continue;
            }
            l3(&bc, "struct_dag", &prof, stem);
            let (plan, m) = typed_module(&bc, "struct_dag", &prof).expect("module");
            let local = &plan.families.access.local;
            assert_eq!(local.typed.len(), typed, "{depth}, {stem}");
            assert_eq!(local.private, private, "{depth}, {stem}");
            let constant = plan.families.access.blob.as_ref();
            assert_eq!(constant.map(|b| b.bytes.len()), blob, "{depth}, {stem}");
            if !interpret {
                continue;
            }
            let mut res = resources(&plan, 0, 64, 32, &[]);
            if let Some(b) = constant {
                res.buffers.push(Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
                        index: 0,
                    }),
                    address: None,
                    bytes: b.bytes.to_vec(),
                });
            }
            let want: Vec<u32> = (0..32u32).flat_map(|t| [t, 7]).collect();
            assert_eq!(run_both(&m, &res).expect("run"), want, "{depth}, {stem}");
        }
    }
}

/// A kernel reading and writing `%big = { %e13, %e13, i32 x extra }`, `%e13` the 13-level struct
/// DAG of `struct_dag_kernel`. Its unshared type size is the DAG's 32,752 (`i32` and `%e1` to
/// `%e13`) and `%big`'s `32,767 + extra`: with 17 extra members exactly `alloca::MAX_UNSHARED`
/// (65,536), with 18 one more. With `constant`, `%big` is the type of `@g`, a closed constant whose
/// DAG halves are zero and whose extra member `i` is `i + 1`, and `v0 = t + @g.2`, `v1 = @g.18`,
/// each read at its byte address (65,536 and 65,600: SCCP reads an integer field at a typed
/// address as its constant, which would leave no load of `@g` to plan); otherwise of the alloca
/// `a`, which stores `t` at byte 0 and 7 in member 2 (byte 65,536) and reads them back as `v0` and
/// `v1`.
fn room_kernel(extra: u32, constant: bool) -> String {
    let mut types = String::from("%e1 = type { i32, i32 }\n");
    for k in 2..=13 {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    types += &format!(
        "%big = type {{ %e13, %e13{} }}\n",
        ", i32".repeat(extra as usize)
    );
    let body = if constant {
        let ints: String = (1..=extra).map(|i| format!(", i32 {i}")).collect();
        types += &format!(
            "@g = internal addrspace(2) constant %big {{ %e13 zeroinitializer, %e13 zeroinitializer{ints} }}, align 4\n"
        );
        r#"  %p2 = getelementptr inbounds i8, ptr addrspace(2) @g, i32 65536
  %g2 = load i32, ptr addrspace(2) %p2, align 4
  %v0 = add i32 %t, %g2
  %q = getelementptr inbounds i8, ptr addrspace(2) @g, i32 65600
  %v1 = load i32, ptr addrspace(2) %q, align 4"#
            .to_string()
    } else {
        r#"  %a = alloca %big, align 4
  store i32 %t, ptr %a, align 4
  %q = getelementptr inbounds %big, ptr %a, i32 0, i32 2
  store i32 7, ptr %q, align 4
  %v0 = load i32, ptr %a, align 4
  %v1 = load i32, ptr %q, align 4"#
            .to_string()
    };
    two_word_kernel("room", &types, &body)
}

/// The room's boundary. An object whose type is exactly `alloca::MAX_UNSHARED` (65,536) of unshared
/// type size is a typed object; one more (an `i32` member added) and an alloca has its place in the
/// private arena, a closed constant in the blob. Each passes L3 and gives the same values on the
/// interpreter (one invocation: the interpreter lays out a struct by walking its every path, per
/// access), for the alloca and for the constant. The profile is MoltenVK on an M3 Max, whose
/// Function storage (192 KiB) holds the 64 KiB arena.
#[test]
fn the_typed_room_s_boundary_keeps_the_values() {
    for constant in [false, true] {
        let want = if constant { vec![1, 17] } else { vec![0, 7] };
        for extra in [17, 18] {
            let name = format!("room_{extra}_{constant}");
            let Some(bc) = assemble(&name, &room_kernel(extra, constant)) else {
                return;
            };
            let prof = p("moltenvk-m3");
            l3(&bc, "room", &prof, &name);
            let (plan, m) = typed_module(&bc, "room", &prof).expect("module");
            let access = &plan.families.access;
            let typed = extra == 17;
            assert_eq!(access.local.typed.len(), usize::from(typed), "{name}");
            let untyped = if constant {
                access.blob.is_some()
            } else {
                access.local.private.is_some()
            };
            assert_eq!(untyped, !typed, "{name}");
            let mut res = resources(&plan, 0, 2, 1, &[]);
            if let Some(b) = &access.blob {
                res.buffers.push(Buffer {
                    slot: Some(ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
                        index: 0,
                    }),
                    address: None,
                    bytes: b.bytes.to_vec(),
                });
            }
            assert_eq!(run_both(&m, &res).expect("run"), want, "{name}");
        }
    }
}

/// A kernel whose alloca `a` is `%big = { %e29, %e28, ..., %e1, i32 x extra }`, `%e{k}` the
/// `k`-level struct DAG of `struct_dag_kernel` (`2^(k+2)` bytes). The DAGs are `2^32 - 8` bytes, so
/// with one extra member `a` is `2^32 - 4` bytes, the largest 4-aligned size within `u32::MAX`, and
/// with two `2^32`. Its unshared type size is past the typed room, so `a` has its place in the
/// private arena. It stores `t` at byte 0 and 7 at byte 4 and reads them back as `v0` and `v1`.
fn arena_bound_kernel(extra: u32) -> String {
    let mut types = String::from("%e1 = type { i32, i32 }\n");
    for k in 2..=29 {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    let dags: Vec<String> = (1..=29).rev().map(|k| format!("%e{k}")).collect();
    types += &format!(
        "%big = type {{ {}{} }}\n",
        dags.join(", "),
        ", i32".repeat(extra as usize)
    );
    let body = r#"  %a = alloca %big, align 4
  store i32 %t, ptr %a, align 4
  %q = getelementptr inbounds i8, ptr %a, i32 4
  store i32 7, ptr %q, align 4
  %v0 = load i32, ptr %a, align 4
  %v1 = load i32, ptr %q, align 4"#;
    two_word_kernel("arena_bound", &types, body)
}

/// The private arena's bound is its `u32` offsets. An alloca of `2^32 - 4` bytes has its place in
/// an arena of that size, which no device's Function storage holds: it is
/// `Unsupported{PrivateMemory}` needing that much. One of `2^32` bytes is
/// `LimitExceeded(PrivateArenaBytes)`, a deterministic cap of the input on every device, not a
/// lowering left to implement, ahead of the device's storage. (A Metal device refuses a compute
/// function's thread memory far below it.) The interpreter would allocate the arena, so it does not
/// run.
#[test]
fn the_private_arena_s_bound_is_its_u32_offsets() {
    for extra in [1u32, 2] {
        let name = format!("arena_bound_{extra}");
        let Some(bc) = assemble(&name, &arena_bound_kernel(extra)) else {
            return;
        };
        for stem in PROFILES {
            let prof = p(stem);
            if extra == 1 {
                let available = prof.fields().function_storage.bytes();
                assert_eq!(
                    compile(&bc, "arena_bound", &prof).err(),
                    Some(Error::Unsupported(Unsupported::needs(
                        Feature::PrivateMemory,
                        Capability::Limit {
                            limit: VkLimit::FunctionStorageBytesPerThread,
                            required: (1u64 << 32) - 4,
                            available,
                        },
                    ))),
                    "{name}, {stem}"
                );
            } else {
                assert_eq!(
                    compile(&bc, "arena_bound", &prof).err(),
                    Some(Error::LimitExceeded(Limit::PrivateArenaBytes)),
                    "{name}, {stem}"
                );
            }
        }
    }
}

/// A vector constant in the blob has its lanes packed at their width, as LLVM lays it out in
/// memory, and no write passes its own bytes. `@v`, a `<5 x i24>`, is 15 bytes (lanes 3 apart) in
/// its 16; `@z`, a `[2 x [2 x i16]]` placed right after it, keeps its zero first half; `@w`, a `<4
/// x i24>`, is 12 bytes. Each is read a byte at a time at a dynamic index, so all three are in the
/// blob. Writing lanes at their allocation size (4 bytes apart) put `@v`'s fifth lane over `@z`'s
/// first word, silently, and spread `@w`'s lanes into its padding.
#[test]
fn a_vector_constant_s_lanes_are_packed_in_the_blob() {
    let globals = "@v = internal addrspace(2) constant <5 x i24> <i24 1, i24 2, i24 3, i24 4, i24 5>, align 16\n\
         @z = internal addrspace(2) constant [2 x [2 x i16]] [[2 x i16] zeroinitializer, [2 x i16] [i16 9, i16 9]], align 4\n\
         @w = internal addrspace(2) constant <4 x i24> <i24 197121, i24 394500, i24 591879, i24 789258>, align 16\n";
    let body = r#"  %tv = urem i32 %t, 15
  %pv = getelementptr inbounds i8, ptr addrspace(2) @v, i32 %tv
  %bv = load i8, ptr addrspace(2) %pv, align 1
  %xv = zext i8 %bv to i32
  %tz = and i32 %t, 7
  %pz = getelementptr inbounds i8, ptr addrspace(2) @z, i32 %tz
  %bz = load i8, ptr addrspace(2) %pz, align 1
  %xz = zext i8 %bz to i32
  %sz = shl i32 %xz, 8
  %v0 = or i32 %xv, %sz
  %tw = urem i32 %t, 12
  %pw = getelementptr inbounds i8, ptr addrspace(2) @w, i32 %tw
  %bw = load i8, ptr addrspace(2) %pw, align 1
  %v1 = zext i8 %bw to i32"#;
    let Some(bc) = assemble(
        "packed_lanes",
        &two_word_kernel("packed_lanes", globals, body),
    ) else {
        return;
    };
    let v: [u32; 15] = [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0];
    let z: [u32; 8] = [0, 0, 0, 0, 9, 0, 9, 0];
    let mut blob = vec![1u8, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 0];
    blob.extend([0, 0, 0, 0, 9, 0, 9, 0]);
    blob.extend([0; 8]);
    blob.extend(1..=12u8);
    blob.extend([0; 4]);
    let want: Vec<u32> = (0..32usize)
        .flat_map(|t| [v[t % 15] | z[t & 7] << 8, (t % 12) as u32 + 1])
        .collect();
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "packed_lanes", &prof, stem);
        let (plan, m) = typed_module(&bc, "packed_lanes", &prof).expect("module");
        let constant = plan.families.access.blob.as_ref().expect("a blob");
        assert_eq!(constant.bytes.to_vec(), blob, "{stem}");
        let mut res = resources(&plan, 0, 64, 32, &[]);
        res.buffers.push(Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Synthetic(SyntheticBinding::ConstantBlob),
                index: 0,
            }),
            address: None,
            bytes: constant.bytes.to_vec(),
        });
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// The globals of the integer-address kernels: `@x` (64 bytes) declared before `@y` (64 bytes),
/// so Metal's layout puts `@y` at 0 and `@x` at 64.
const LITERAL_GLOBALS: &str = "@x = internal addrspace(3) global [16 x i32] undef, align 4\n\
     @y = internal addrspace(3) global [4 x <4 x i32>] undef, align 16\n\
     declare void @air.wg.barrier(i32, i32)\n\
     declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)";

/// An integer threadgroup address names a byte of Metal's layout, which places the static globals
/// in the reverse of the order of their last references from 0: here `@y` is read by name before
/// `@x`, so `@x` lies at 0 and `@y` at 64 (the reverse of the module order would put `@y` at 0).
/// Thread 0 stores `0xC0FFEE` at byte 16 (`x[4]`) and an atomic adds 1000 at byte 72 (word 2 of
/// `y`); every thread then reads word `t` of `y` and `x[t]` by name. Previously both literal
/// accesses were refused `DereferenceLoadedPointer`.
#[test]
fn an_integer_threadgroup_address_reaches_metals_layout() {
    let body = r#"  %fx = getelementptr inbounds [16 x i32], ptr addrspace(3) @x, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %fx, align 4
  %fy = getelementptr inbounds i32, ptr addrspace(3) @y, i32 %t
  %w = lshr i32 %t, 2
  %yv = add i32 %w, 100
  store i32 %yv, ptr addrspace(3) %fy, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %first = icmp eq i32 %t, 0
  br i1 %first, label %poke, label %read
poke:
  store i32 12648430, ptr addrspace(3) inttoptr (i64 16 to ptr addrspace(3)), align 16
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) inttoptr (i64 72 to ptr addrspace(3)), i32 1000, i32 0, i32 1, i1 true)
  br label %read
read:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %ry = getelementptr inbounds i32, ptr addrspace(3) @y, i32 %t
  %v1 = load i32, ptr addrspace(3) %ry, align 4
  %rx = getelementptr inbounds [16 x i32], ptr addrspace(3) @x, i32 0, i32 %t
  %v0 = load i32, ptr addrspace(3) %rx, align 4"#;
    let Some(bc) = assemble(
        "tg_literal",
        &two_word_kernel("tg_literal", LITERAL_GLOBALS, body),
    ) else {
        return;
    };
    let want: Vec<u32> = (0..16u32)
        .flat_map(|t| {
            let x = if t == 4 { 0xC0FFEE } else { t };
            let y = if t == 2 { 1100 } else { 100 + t / 4 };
            [x, y]
        })
        .collect();
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "tg_literal", &prof, stem);
        let (plan, m) = typed_module(&bc, "tg_literal", &prof).expect("module");
        let res = resources(&plan, 0, 32, 16, &[]);
        assert_eq!(run_both(&m, &res).expect("run"), want, "{stem}");
    }
}

/// An integer threadgroup address outside the measured extent stays refused `NotRepresentable`
/// `DereferenceLoadedPointer`: a store at the static size (128, past Metal's static objects), an
/// 8-byte store at byte 124 (its last four bytes past them), a `getelementptr` of a literal (its
/// displacement of the address is not measured), a store at byte 16 in a module that declares a
/// function constant (Metal lays out the specialized function), and one beside a `select` of the
/// two globals' addresses (an instruction naming both: no measurement orders them).
#[test]
fn an_integer_threadgroup_address_outside_the_static_objects_is_refused() {
    let reads = r#"  %rx = getelementptr inbounds [16 x i32], ptr addrspace(3) @x, i32 0, i32 %t
  %v0 = load i32, ptr addrspace(3) %rx, align 4
  %ry = getelementptr inbounds i32, ptr addrspace(3) @y, i32 %t
  %v1 = load i32, ptr addrspace(3) %ry, align 4"#;
    let cases = [
        (
            "past_static",
            "  store i32 1, ptr addrspace(3) inttoptr (i64 128 to ptr addrspace(3)), align 4",
        ),
        (
            "straddling",
            "  store <2 x i32> <i32 1, i32 1>, ptr addrspace(3) inttoptr (i64 124 to ptr addrspace(3)), align 4",
        ),
        (
            "displaced",
            "  %q = getelementptr inbounds i32, ptr addrspace(3) inttoptr (i64 16 to ptr addrspace(3)), i32 %t\n  store i32 1, ptr addrspace(3) %q, align 4",
        ),
        (
            "specialized",
            "  store i32 1, ptr addrspace(3) inttoptr (i64 16 to ptr addrspace(3)), align 16",
        ),
        (
            "unordered",
            "  store i32 1, ptr addrspace(3) inttoptr (i64 16 to ptr addrspace(3)), align 16\n  %odd = trunc i32 %t to i1\n  %s = select i1 %odd, ptr addrspace(3) @x, ptr addrspace(3) @y\n  store i32 2, ptr addrspace(3) %s, align 4",
        ),
    ];
    // A bool constant at index 0 that nothing reads.
    let constant = "@k.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section \"air.fc_initializer\", align 1";
    let constant_md = "!air.function_constants = !{!9}\n!9 = !{ptr addrspace(2) @k.MTL_FC_INIT_0_b, !\"bool\", !\"k\", i32 0}\n";
    for (name, poke) in cases {
        let text = two_word_kernel(name, LITERAL_GLOBALS, &format!("{poke}\n{reads}"));
        let text = if name == "specialized" {
            format!("{constant}\n{text}{constant_md}")
        } else {
            text
        };
        let Some(bc) = assemble(name, &text) else {
            return;
        };
        for stem in PROFILES {
            match compile(&bc, name, &p(stem)) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(
                        u.feature,
                        Feature::DereferenceLoadedPointer,
                        "{name} {stem}"
                    );
                    assert_eq!(u.needs, Needs::NotRepresentable, "{name} {stem}");
                }
                other => panic!("{name} {stem}: expected a refusal, got {other:?}"),
            }
        }
    }
}
