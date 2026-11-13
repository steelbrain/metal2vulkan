//! Memory atomics, fences, and the memory model end to end.
//!
//! - Every integer and float atomic operation on a buffer slot is exact on the CPU interpreter
//!   under 64 contending invocations in both scheduling orders, and passes L3 on every profile
//!   (the float read-modify-writes are compare-and-swap loops).
//! - The committed `kernel_atomic_xor_and_subtract` fixture (device and threadgroup atomics, an
//!   `f32` subtract) computes its documented result.
//! - A threadgroup compare-exchange writes the old value back through its `expected` pointer and
//!   exactly one invocation wins; threadgroup loads and stores are atomic.
//! - An out-of-bounds atomic is contained (no trap, result 0, memory untouched), through a view
//!   and through a device address (the sink).
//! - LLVM `i8`/`i16` atomics on fields sharing buffer or threadgroup words are exact under
//!   contention, and contained out of bounds.
//! - 64-bit atomics refuse by the profile's int64-atomic features (the matrix per profile).
//! - LLVM `fence` instructions: a system fence is `OpMemoryBarrier`, a single-thread one nothing.
//! - Coherent accesses take the Vulkan memory model where the profile has it with its device
//!   scope, and `GLSL450` with `Coherent` storage buffers elsewhere, exact either way.
//! - A coherent access beside a storage image is `NotImplemented{Lowering(MemoryModel)}` (its
//!   image accesses would need `NonPrivateTexel` under the Vulkan model).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air;
use crate::budget::SPIRV_WORDS;
use crate::device::profile::{DeviceProfile, GuestIsolation};
use crate::device::requirements::VkFeature;
use crate::device::{ContainmentPolicy, Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::library::FunctionConstantValues;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs, Plan};
use crate::lower::simd::WidestPin;
use crate::memory::static_root::analyze_memory;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, meter, profile, status_reach,
    status_reach_on, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, InterpError, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::verify::verify;

/// Profiles with robust descriptors and without, with and without 64-bit integers.
const PROFILES: [&str; 5] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "lavapipe",
    "nvidia-pascal-linux",
];

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("profile")
}

/// Committed profile `stem` without buffer robustness: containment is checked in the shader.
fn unrobust(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    desc.features.remove(&VkFeature::RobustBufferAccess);
    desc.features.remove(&VkFeature::RobustBufferAccess2);
    Device::new(desc, DevicePolicy::default())
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
        "m2v-atomics-{}-{}-{name}",
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

fn buffer_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
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

/// The device address of `[[buffer(i)]]` in [`resources`].
fn data_address(i: usize) -> u64 {
    0x0000_0050_0000_0000 + ((i as u64) << 32)
}

const TABLE_AT: u64 = 0x0000_0040_0000_0000;
const SINK_AT: u64 = 0x0000_0041_0000_0000;
/// The command buffer's `ExecutionStatus` buffer (the `seq_cst` order).
const STATUS_AT: u64 = 0x0000_0042_0000_0000;

/// Resources of a kernel with `threads` threads and buffers `[[buffer(i)]]` holding `init[i]`,
/// bound at their starts. Every buffer is also at a device address with
/// its root-table entry, for a plan that reaches slots by device address (typed atomics).
fn resources(plan: &Plan, init: &[Vec<u32>], threads: u32) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), u64::from(threads));
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let (at, value) = match (f.field, f.loc) {
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => (offset, None),
            (PushField::RootTableAddress, PushLoc::Push { offset }) => (offset, Some(TABLE_AT)),
            (PushField::ExecutionStatusAddress, PushLoc::Push { offset }) => {
                (offset, Some(STATUS_AT))
            }
            other => panic!("unexpected push field {other:?}"),
        };
        let at = at as usize;
        match value {
            None => push[at..at + 4].fill(0),
            Some(v) => push[at..at + 8].copy_from_slice(&v.to_le_bytes()),
        }
    }
    let mut table = vec![0u8; 32 * (2 + init.len())];
    table[32..40].copy_from_slice(&SINK_AT.to_le_bytes());
    table[48..56].copy_from_slice(&64u64.to_le_bytes());
    let mut buffers: Vec<Buffer> = init
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let e = 32 * (2 + i);
            let (a, n) = (data_address(i), 4 * w.len() as u64);
            table[e..e + 8].copy_from_slice(&a.to_le_bytes());
            table[e + 8..e + 16].copy_from_slice(&a.to_le_bytes());
            table[e + 16..e + 24].copy_from_slice(&n.to_le_bytes());
            Buffer {
                slot: Some(buffer_slot(i as u32)),
                address: Some(a),
                bytes: to_bytes(w),
            }
        })
        .collect();
    buffers.push(Buffer {
        slot: None,
        address: Some(SINK_AT),
        bytes: vec![0; 64],
    });
    buffers.push(Buffer {
        slot: None,
        address: Some(TABLE_AT),
        bytes: table,
    });
    // At its address, and at the kernel's binding of it where the device does not address
    // buffers.
    buffers.push(Buffer {
        slot: Some(crate::lower::access::absent::symbol(Stage::Kernel)),
        address: Some(STATUS_AT),
        bytes: vec![0; crate::bind::status::ExecutionStatus::BYTES],
    });
    Resources {
        buffers,
        push,
        spec: s,
    }
}

/// Run in both scheduling orders; returns each order's buffers as words.
fn run_orders(m: &TypedModule, res: &Resources) -> Result<Vec<Vec<Vec<u32>>>, InterpError> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone())?;
        outs.push(out.buffers.iter().map(|b| words(b)).collect());
    }
    Ok(outs)
}

const COUNTERS: &str = r#"define void @counters(ptr addrspace(1) %buf, i32 %t) {
entry:
  %o0 = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %buf, i32 1, i32 0, i32 2, i1 true)
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 1
  %o1 = call i32 @air.atomic.global.max.s.i32(ptr addrspace(1) %p1, i32 %t, i32 0, i32 2, i1 true)
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 2
  %t5 = add i32 %t, 5
  %o2 = call i32 @air.atomic.global.min.u.i32(ptr addrspace(1) %p2, i32 %t5, i32 0, i32 2, i1 true)
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 3
  %sh = and i32 %t, 31
  %bit = shl i32 1, %sh
  %o3 = call i32 @air.atomic.global.or.u.i32(ptr addrspace(1) %p3, i32 %bit, i32 0, i32 2, i1 true)
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 4
  %o4 = call i32 @air.atomic.global.xor.u.i32(ptr addrspace(1) %p4, i32 %t, i32 0, i32 2, i1 true)
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 5
  %lo = and i32 %t, 15
  %lbit = shl i32 1, %lo
  %keep = xor i32 %lbit, -1
  %o5 = call i32 @air.atomic.global.and.u.i32(ptr addrspace(1) %p5, i32 %keep, i32 0, i32 2, i1 true)
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 6
  %o6 = call i32 @air.atomic.global.sub.s.i32(ptr addrspace(1) %p6, i32 %t, i32 0, i32 2, i1 true)
  %p7 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 7
  %o7 = call i32 @air.atomic.global.min.s.i32(ptr addrspace(1) %p7, i32 %t, i32 0, i32 2, i1 true)
  %p8 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 8
  %f8 = call float @air.atomic.global.add.f32(ptr addrspace(1) %p8, float 5.000000e-01, i32 0, i32 2, i1 true)
  %tf = uitofp i32 %t to float
  %p9 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 9
  %f9 = call float @air.atomic.global.max.f32(ptr addrspace(1) %p9, float %tf, i32 0, i32 2, i1 true)
  %tf1 = fadd float %tf, 1.000000e+00
  %p10 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 10
  %f10 = call float @air.atomic.global.min.f32(ptr addrspace(1) %p10, float %tf1, i32 0, i32 2, i1 true)
  %p11 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 11
  %f11 = call float @air.atomic.global.sub.f32(ptr addrspace(1) %p11, float 2.500000e-01, i32 0, i32 2, i1 true)
  %p12 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 12
  %x12 = call float @air.atomic.global.xchg.f32(ptr addrspace(1) %p12, float %tf, i32 0, i32 2, i1 true)
  %p13 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 13
  call void @air.atomic.global.store.i32(ptr addrspace(1) %p13, i32 77, i32 0, i32 2, i1 true)
  %p14 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 14
  %l14 = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p14, i32 0, i32 2, i1 true)
  %i0 = add i32 %t, 16
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i0
  store i32 %o0, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 80
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i1
  %b8 = bitcast float %f8 to i32
  store i32 %b8, ptr addrspace(1) %q1, align 4
  %i2 = add i32 %t, 144
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i2
  store i32 %l14, ptr addrspace(1) %q2, align 4
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.max.s.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.min.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.or.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.xor.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.and.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.sub.s.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.min.s.i32(ptr addrspace(1), i32, i32, i32, i1)
declare float @air.atomic.global.add.f32(ptr addrspace(1), float, i32, i32, i1)
declare float @air.atomic.global.max.f32(ptr addrspace(1), float, i32, i32, i1)
declare float @air.atomic.global.min.f32(ptr addrspace(1), float, i32, i32, i1)
declare float @air.atomic.global.sub.f32(ptr addrspace(1), float, i32, i32, i1)
declare float @air.atomic.global.xchg.f32(ptr addrspace(1), float, i32, i32, i1)
declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @counters, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// The uses of the strategy named `name`, the memory it reaches included (`.psb`,
/// `.psb_and_view`), in either robustness encoding: `name` and `name.checked`.
fn uses(plan: &Plan, name: &str) -> u32 {
    plan.census()
        .iter()
        .filter(|(k, _)| k.strip_suffix(".checked").unwrap_or(k) == name)
        .map(|(_, (n, _))| *n)
        .sum()
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Whether `p` lowers a buffer's `f32` add natively: the float-add feature and device addresses.
fn native_float_add(p: &DeviceProfile) -> bool {
    p.fields().leaf.buffer_float32_atomic_add
        && p.fields().buffer_access == crate::device::profile::BufferAccess::Addressed
}

/// Every operation, 64 contending invocations, both orders.
#[test]
fn buffer_atomics_are_exact_under_contention() {
    let Some(bc) = assemble("counters", COUNTERS) else {
        return;
    };
    const N: u32 = 64;
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "counters", &prof, stem);
        let a = compile(&bc, "counters", &prof).expect("compile");
        // Float add and sub are native through the slot's device address where the profile has
        // the float-add feature; min, max, and elsewhere add and sub are compare-and-swap loops,
        // which need no status buffer.
        let native = native_float_add(&prof);
        let loops = if native { 2 } else { 4 };
        assert!(!a.facts.contains(&RuntimeFact::ExecutionStatus), "{stem}");
        let psb = a
            .facts
            .iter()
            .any(|f| matches!(f, RuntimeFact::DeviceAddresses { .. }));
        assert_eq!(psb, native, "{stem}");
        let (plan, m) = typed_module(&bc, "counters", &prof).expect("module");
        // Every atomic reaches the one slot, through its device address where the float add is
        // native.
        let reach = |name: &str| {
            if native {
                format!("{name}.psb")
            } else {
                name.to_owned()
            }
        };
        assert_eq!(
            uses(&plan, &reach("access.atomic.buffer.float_cas")),
            loops,
            "{stem}"
        );
        assert_eq!(
            uses(&plan, &reach("access.atomic.buffer.float_add")),
            4 - loops,
            "{stem}"
        );
        // Eight integer operations and the float exchange.
        assert_eq!(uses(&plan, &reach("access.atomic.buffer.rmw")), 9, "{stem}");
        let mut init = vec![0u32; 16 + 3 * N as usize];
        init[2] = u32::MAX;
        init[5] = u32::MAX;
        init[7] = 1000;
        init[9] = f(-1.0);
        init[10] = f(1000.0);
        init[11] = f(100.0);
        init[12] = f(-5.0);
        init[14] = 0xABCD;
        let res = resources(&plan, &[init], N);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            assert_eq!(w[0], N, "{stem}: add");
            assert_eq!(w[1], N - 1, "{stem}: max.s");
            assert_eq!(w[2], 5, "{stem}: min.u");
            assert_eq!(w[3], u32::MAX, "{stem}: or");
            assert_eq!(w[4], (0..N).fold(0, |x, t| x ^ t), "{stem}: xor");
            assert_eq!(w[5], 0xFFFF_0000, "{stem}: and");
            assert_eq!(w[6], 0u32.wrapping_sub((0..N).sum()), "{stem}: sub.s");
            assert_eq!(w[7], 0, "{stem}: min.s");
            assert_eq!(w[8], f(0.5 * N as f32), "{stem}: float add");
            assert_eq!(w[9], f((N - 1) as f32), "{stem}: float max");
            assert_eq!(w[10], f(1.0), "{stem}: float min");
            assert_eq!(w[11], f(100.0 - 0.25 * N as f32), "{stem}: float sub");
            assert!(
                (0..N).any(|t| w[12] == f(t as f32)),
                "{stem}: float xchg {:#x}",
                w[12]
            );
            assert_eq!(w[13], 77, "{stem}: store");
            // The old values of the add are a permutation of 0..N, and of the float add the
            // multiples of 0.5.
            let mut olds: Vec<u32> = w[16..16 + N as usize].to_vec();
            olds.sort_unstable();
            assert_eq!(olds, (0..N).collect::<Vec<_>>(), "{stem}: add olds");
            let mut fl: Vec<u32> = w[80..80 + N as usize].to_vec();
            fl.sort_unstable_by(|a, b| f32::from_bits(*a).total_cmp(&f32::from_bits(*b)));
            let want: Vec<u32> = (0..N).map(|k| f(0.5 * k as f32)).collect();
            assert_eq!(fl, want, "{stem}: float add olds");
            assert!(
                w[144..144 + N as usize].iter().all(|x| *x == 0xABCD),
                "{stem}: load"
            );
        }
    }
}

/// The committed fixture's documented result: `unsigned_words[0]` is the xor of
/// `t * 0x9E3779B1`, `signed_total` and `float_total` subtract `t + 1`, and the threadgroup total
/// (initialized by thread 0 before a barrier) is stored to `unsigned_words[1]`.
#[test]
fn the_atomic_xor_and_subtract_fixture_is_exact() {
    let bc = match llvm_assemble(&fixtures_dir().join("kernel_atomic_xor_and_subtract.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("{why}"),
    };
    const N: u32 = 64;
    let entry = "kernel_atomic_xor_and_subtract";
    let sum: u32 = (1..=N).sum();
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, entry, &prof, stem);
        let (plan, m) = typed_module(&bc, entry, &prof).expect("module");
        let res = resources(&plan, &[vec![0; 2], vec![0; 1], vec![f(10.0)]], N);
        for out in run_orders(&m, &res).expect("run") {
            let x = (0..N).fold(0u32, |x, t| x ^ t.wrapping_mul(0x9E37_79B1));
            assert_eq!(out[0][0], x, "{stem}: xor");
            assert_eq!(out[0][1], 0u32.wrapping_sub(sum), "{stem}: threadgroup sub");
            assert_eq!(out[1][0], 0u32.wrapping_sub(sum), "{stem}: signed sub");
            assert_eq!(out[2][0], f(10.0 - sum as f32), "{stem}: float sub");
        }
    }
}

const TG_CMPXCHG: &str = r#"@c = internal addrspace(3) global i32 undef, align 4

define void @tg_cmpxchg(ptr addrspace(1) %out, i32 %t) {
entry:
  %e = alloca i32, align 4
  %first = icmp eq i32 %t, 0
  br i1 %first, label %init, label %go

init:
  call void @air.atomic.local.store.i32(ptr addrspace(3) @c, i32 0, i32 0, i32 1, i1 true)
  br label %go

go:
  call void @air.wg.barrier(i32 2, i32 1)
  store i32 0, ptr %e, align 4
  %t1 = add i32 %t, 1
  %r = call i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3) @c, ptr %e, i32 %t1, i32 0, i32 0, i32 1, i1 true)
  %ev = load i32, ptr %e, align 4
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %r, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %ev, ptr addrspace(1) %q1, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %v = call i32 @air.atomic.local.load.i32(ptr addrspace(3) @c, i32 0, i32 1, i1 true)
  %i2 = add i32 %t, 128
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i2
  store i32 %v, ptr addrspace(1) %q2, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)
declare i32 @air.atomic.local.cmpxchg.weak.i32(ptr addrspace(3), ptr, i32, i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @tg_cmpxchg, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Each invocation tries once to swap 0 for `t + 1`: exactly one wins (its old value is 0), every
/// other sees the winner's value, `expected` holds the old value afterwards, and every invocation
/// then loads the winner's value.
#[test]
fn a_threadgroup_compare_exchange_writes_back_its_expected_value() {
    let Some(bc) = assemble("tg_cmpxchg", TG_CMPXCHG) else {
        return;
    };
    const N: usize = 64;
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "tg_cmpxchg", &prof, stem);
        let (plan, m) = typed_module(&bc, "tg_cmpxchg", &prof).expect("module");
        assert_eq!(
            uses(&plan, "access.atomic.threadgroup.cmpxchg"),
            1,
            "{stem}"
        );
        let res = resources(&plan, &[vec![0; 3 * N]], N as u32);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            let winners: Vec<usize> = (0..N).filter(|t| w[*t] == 0).collect();
            assert_eq!(winners.len(), 1, "{stem}: one winner");
            let won = winners[0] as u32 + 1;
            for t in 0..N {
                if t != winners[0] {
                    assert_eq!(w[t], won, "{stem}: loser {t}");
                }
                assert_eq!(w[N + t], w[t], "{stem}: expected written back");
                assert_eq!(w[2 * N + t], won, "{stem}: final load");
            }
        }
    }
}

const OOB: &str = r#"define void @oob(ptr addrspace(1) %buf, i32 %t) {
entry:
  %i = mul i32 %t, 1000
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i
  %o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 5, i32 0, i32 2, i1 true)
  %j = add i32 %t, 1
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %j
  store i32 %o, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @oob, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Invocation 0 adds 5 to word 0; the others' atomics are 1000·t words out: contained (no trap, old
/// value 0, nothing written) by the in-shader check (robustness removed; the interpreter traps an
/// unchecked out-of-bounds access, so the robust profiles pass L3 only).
#[test]
fn an_out_of_bounds_buffer_atomic_is_contained() {
    let Some(bc) = assemble("oob", OOB) else {
        return;
    };
    for stem in PROFILES {
        l3(&bc, "oob", &p(stem), stem);
        let prof = unrobust(stem);
        l3(&bc, "oob", &prof, stem);
        let (plan, m) = typed_module(&bc, "oob", &prof).expect("module");
        assert!(
            plan.census()
                .contains_key("access.atomic.buffer.rmw.checked"),
            "{stem}"
        );
        let res = resources(&plan, &[vec![0; 8]], 4);
        for out in run_orders(&m, &res).expect("run") {
            assert_eq!(out[0], vec![5, 0, 0, 0, 0, 0, 0, 0], "{stem}");
        }
    }
}

/// SwiftShader's `robustBufferAccess` bounds an atomic only through `OpImageTexelPointer`, so its
/// profile carries `DriverWorkaround::BufferAtomicsUncontained`, keyed on its driver identity: a
/// storage-buffer atomic through a descriptor view is checked in the shader there (contained: no
/// trap, old value 0, nothing written) while its loads and stores keep the robust encoding. No
/// other committed profile has the workaround, and containment off checks nothing.
#[test]
fn swiftshader_checks_buffer_atomics_under_robustness() {
    use crate::device::profile::DriverWorkaround;
    use crate::memory::contain::{atomic_encoding, encoding, Check};
    let Some(bc) = assemble("oob", OOB) else {
        return;
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut stems: Vec<String> = std::fs::read_dir(&dir)
        .expect("profiles")
        .filter_map(|e| {
            let path = e.expect("entry").path();
            (path.extension()? == "json").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    stems.sort();
    assert!(stems.iter().any(|s| s == "swiftshader"), "{stems:?}");
    for stem in &stems {
        let prof = p(stem);
        let swiftshader = stem == "swiftshader";
        assert_eq!(
            prof.fields()
                .workarounds
                .contains(&DriverWorkaround::BufferAtomicsUncontained),
            swiftshader,
            "{stem}"
        );
        if encoding(&prof) == Check::InShader {
            assert_eq!(atomic_encoding(&prof), Check::InShader, "{stem}");
            continue;
        }
        assert_eq!(
            atomic_encoding(&prof),
            if swiftshader {
                Check::InShader
            } else {
                Check::None
            },
            "{stem}"
        );
    }
    let prof = p("swiftshader");
    assert_eq!(encoding(&prof), Check::None, "loads and stores stay robust");
    l3(&bc, "oob", &prof, "swiftshader");
    let (plan, m) = typed_module(&bc, "oob", &prof).expect("module");
    assert!(
        plan.census()
            .contains_key("access.atomic.buffer.rmw.checked"),
        "{:?}",
        plan.census()
    );
    let res = resources(&plan, &[vec![0; 8]], 4);
    for out in run_orders(&m, &res).expect("run") {
        assert_eq!(out[0], vec![5, 0, 0, 0, 0, 0, 0, 0]);
    }
    // lavapipe's robustness contains buffer atomics: no check.
    let (plan, _) = typed_module(&bc, "oob", &p("lavapipe")).expect("module");
    assert!(plan.census().contains_key("access.atomic.buffer.rmw"));
    // Containment off checks nothing, the workaround's atomics included.
    let unsafe_off = DevicePolicy {
        containment: ContainmentPolicy::UnsafeOff,
        isolation: GuestIsolation::Off,
        ..Default::default()
    };
    let off = profile("swiftshader", unsafe_off).expect("profile");
    assert_eq!(atomic_encoding(&off), Check::None);
    let (plan, _) = typed_module(&bc, "oob", &off).expect("module");
    assert!(plan.census().contains_key("access.atomic.buffer.rmw"));
}

fn int64_kernel(space: &str) -> String {
    let (ptr, meta) = if space == "global" {
        ("ptr addrspace(1)", "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"buf\"}")
    } else {
        ("ptr addrspace(3)", "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"buf\"}")
    };
    format!(
        r#"define void @wide({ptr} %buf, i32 %t) {{
entry:
  %v = zext i32 %t to i64
  %o = call i64 @air.atomic.{space}.max.u.i64({ptr} %buf, i64 %v, i32 0, i32 2, i1 true)
  ret void
}}

declare i64 @air.atomic.{space}.max.u.i64({ptr}, i64, i32, i32, i1)

!air.kernel = !{{!0}}
!0 = !{{ptr @wide, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
{meta}
!4 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
"#
    )
}

/// A 64-bit atomic needs the int64-atomic feature of its memory (named in the refusal); where the
/// profile has it, a buffer's is lowered through a typed device address and a threadgroup one is
/// not implemented (no `u64` view of the arena).
#[test]
fn int64_atomics_refuse_by_the_profiles_features() {
    for (space, shared) in [("global", false), ("local", true)] {
        let Some(bc) = assemble(space, &int64_kernel(space)) else {
            return;
        };
        for stem in PROFILES {
            let prof = p(stem);
            let leaf = &prof.fields().leaf;
            let (have, feature) = if shared {
                (
                    leaf.shared_int64_atomics,
                    VkFeature::ShaderSharedInt64Atomics,
                )
            } else {
                (
                    leaf.buffer_int64_atomics,
                    VkFeature::ShaderBufferInt64Atomics,
                )
            };
            let addressed =
                prof.fields().buffer_access == crate::device::profile::BufferAccess::Addressed;
            if have && !shared && addressed {
                l3(&bc, "wide", &prof, stem);
                continue;
            }
            let e = compile(&bc, "wide", &prof).expect_err("refused");
            let Error::Unsupported(u) = e else {
                panic!("{stem} {space}: {e:?}");
            };
            // Without Int64 the `zext` is held as two words, so the refusal is the atomic's own: a
            // 64-bit atomic needs its int64-atomic feature, which such a profile lacks.
            if !prof.fields().leaf.int64 {
                assert!(!have, "{stem} {space}");
            }
            let want = if have {
                Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access))
            } else {
                Unsupported::needs(Feature::Int64Atomics, Capability::Feature(feature))
            };
            assert_eq!(
                (u.feature, u.needs),
                (want.feature, want.needs),
                "{stem} {space}"
            );
        }
    }
}

const WIDE: &str = r#"define void @wide64(ptr addrspace(1) %buf, i32 %t) {
entry:
  %v = zext i32 %t to i64
  %big = shl i64 %v, 40
  %o0 = atomicrmw add ptr addrspace(1) %buf, i64 %big monotonic, align 8
  %p1 = getelementptr inbounds i64, ptr addrspace(1) %buf, i32 1
  %o1 = call i64 @air.atomic.global.max.u.i64(ptr addrspace(1) %p1, i64 %big, i32 0, i32 2, i1 true)
  %p2 = getelementptr inbounds i64, ptr addrspace(1) %buf, i32 2
  %t1 = add i64 %v, 1
  %cx = cmpxchg ptr addrspace(1) %p2, i64 0, i64 %t1 monotonic monotonic, align 8
  %p3 = getelementptr inbounds i64, ptr addrspace(1) %buf, i32 3
  %o3 = atomicrmw nand ptr addrspace(1) %p3, i64 -1 monotonic, align 8
  %old = extractvalue { i64, i1 } %cx, 0
  %i = add i32 %t, 4
  %q = getelementptr inbounds i64, ptr addrspace(1) %buf, i32 %i
  store i64 %old, ptr addrspace(1) %q, align 8
  ret void
}

declare i64 @air.atomic.global.max.u.i64(ptr addrspace(1), i64, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @wide64, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// 64-bit buffer atomics (`air` and LLVM, a compare-and-swap-loop `nand` among them) go through the
/// slot's device address as `u64` atomics and are exact under 64 contending invocations on every
/// profile with `shaderBufferInt64Atomics` and device addresses.
#[test]
fn int64_buffer_atomics_are_exact_through_device_addresses() {
    let Some(bc) = assemble("wide64", WIDE) else {
        return;
    };
    const N: usize = 64;
    for stem in [
        "lavapipe",
        "nvidia-pascal-linux",
        "amd-rdna3-radv",
        "intel-arl-anv",
    ] {
        let prof = p(stem);
        l3(&bc, "wide64", &prof, stem);
        let (plan, m) = typed_module(&bc, "wide64", &prof).expect("module");
        assert_eq!(uses(&plan, "access.atomic.buffer.cas.psb"), 1, "{stem}");
        let res = resources(&plan, &[vec![0; 2 * (4 + N)]], N as u32);
        for out in run_orders(&m, &res).expect("run") {
            let w: Vec<u64> = out[0]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                .collect();
            let sum: u64 = (0..N as u64).map(|t| t << 40).sum();
            assert_eq!(w[0], sum, "{stem}: add");
            assert_eq!(w[1], (N as u64 - 1) << 40, "{stem}: max");
            assert_eq!(w[3], 0, "{stem}: nand of 0 toggled an even number of times");
            let winners: Vec<usize> = (0..N).filter(|t| w[4 + t] == 0).collect();
            assert_eq!(winners.len(), 1, "{stem}: one winner");
            assert_eq!(w[2], winners[0] as u64 + 1, "{stem}: cmpxchg");
        }
    }
}

const ULONG_MIN_MAX: &str = r#"define void @ulong_min_max(ptr addrspace(1) %buf, i32 %t) {
entry:
  %v = zext i32 %t to i64
  %big = shl i64 %v, 40
  %low = add i64 %big, 7
  call void @air.atomic.global.max.u.i64(ptr addrspace(1) %buf, i64 %low, i32 0, i32 2, i1 true)
  %p1 = getelementptr inbounds i64, ptr addrspace(1) %buf, i32 1
  call void @air.atomic.global.min.u.i64(ptr addrspace(1) %p1, i64 %low, i32 0, i32 2, i1 true)
  ret void
}

declare void @air.atomic.global.max.u.i64(ptr addrspace(1), i64, i32, i32, i1)
declare void @air.atomic.global.min.u.i64(ptr addrspace(1), i64, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @ulong_min_max, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Metal's 64-bit `atomic_max_explicit`/`atomic_min_explicit` on a `device atomic_ulong` return
/// nothing, and their result-less `air.atomic.global.{max,min}.u.i64` lower as the fetching forms
/// do, exact under 64 contending invocations on every profile with `shaderBufferInt64Atomics` and
/// device addresses.
#[test]
fn result_less_ulong_min_max_atomics_are_exact() {
    let Some(bc) = assemble("ulong_min_max", ULONG_MIN_MAX) else {
        return;
    };
    const N: usize = 64;
    for stem in [
        "lavapipe",
        "nvidia-pascal-linux",
        "amd-rdna3-radv",
        "intel-arl-anv",
    ] {
        let prof = p(stem);
        l3(&bc, "ulong_min_max", &prof, stem);
        let (plan, m) = typed_module(&bc, "ulong_min_max", &prof).expect("module");
        // Word 1 starts at the largest value, so every invocation's minimum competes.
        let mut init = vec![0u32; 4];
        init[2] = u32::MAX;
        init[3] = u32::MAX;
        let res = resources(&plan, &[init], N as u32);
        for out in run_orders(&m, &res).expect("run") {
            let w: Vec<u64> = out[0]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                .collect();
            assert_eq!(w[0], ((N as u64 - 1) << 40) + 7, "{stem}: max");
            assert_eq!(w[1], 7, "{stem}: min");
        }
    }
}

const ADDRESSED_AND_VIEWED: &str = r#"define void @addressed_and_viewed(ptr addrspace(1) %wide, ptr addrspace(1) %words, i32 %t) {
entry:
  %v = zext i32 %t to i64
  %o0 = atomicrmw add ptr addrspace(1) %wide, i64 %v monotonic, align 8
  %o1 = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %words, i32 1, i32 0, i32 2, i1 true)
  %c = icmp eq i32 %t, 0
  %p = select i1 %c, ptr addrspace(1) %wide, ptr addrspace(1) %words
  %x = load i32, ptr addrspace(1) %p, align 4
  %i = add i32 %t, 2
  %q = getelementptr inbounds i32, ptr addrspace(1) %words, i32 %i
  store i32 %x, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @addressed_and_viewed, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"wide"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"words"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A slot a typed atomic reaches is read by device address and checked in the shader on every
/// profile (robust buffer access does not cover it), while another slot is a descriptor view under
/// the profile's encoding, so the census names each access by the form it takes: the `u64` add
/// `access.atomic.buffer.rmw.psb.checked` with robust descriptors and without them, the `u32` add
/// through the view `access.atomic.buffer.rmw` or `.checked` as the profile's encoding says, and a
/// load whose pointer may be either `access.load.switch.psb_and_view.checked`. With containment off
/// nothing is checked, the device address included.
#[test]
fn a_device_address_access_is_named_apart_from_a_descriptor_access() {
    let Some(bc) = assemble("addressed_and_viewed", ADDRESSED_AND_VIEWED) else {
        return;
    };
    let entry = "addressed_and_viewed";
    let names = |prof: &DeviceProfile| {
        let (plan, _) = typed_module(&bc, entry, prof).expect("module");
        let census = plan.census();
        let names = census
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        (plan, names)
    };
    let expect = |names: &std::collections::BTreeSet<String>, want: &[&str], what: &str| {
        for name in want {
            assert!(names.contains(*name), "{what}: {name} not in {names:?}");
        }
    };
    let robust = p("lavapipe");
    l3(&bc, entry, &robust, "lavapipe");
    let (plan, robust_names) = names(&robust);
    assert!(plan.families.access.psb_contained);
    let forms = plan
        .families
        .access
        .slots
        .iter()
        .flatten()
        .map(|l| matches!(l, crate::lower::access::SlotLowering::Psb { .. }))
        .collect::<Vec<_>>();
    assert_eq!(
        forms,
        [true, false],
        "the typed slot alone is a device address"
    );
    expect(
        &robust_names,
        &[
            "access.atomic.buffer.rmw.psb.checked",
            "access.atomic.buffer.rmw",
            "access.load.switch.psb_and_view.checked",
            "access.store",
        ],
        "lavapipe",
    );
    let checked = unrobust("lavapipe");
    l3(&bc, entry, &checked, "lavapipe without robustness");
    let (_, checked_names) = names(&checked);
    expect(
        &checked_names,
        &[
            "access.atomic.buffer.rmw.psb.checked",
            "access.atomic.buffer.rmw.checked",
            "access.load.switch.psb_and_view.checked",
            "access.store.checked",
        ],
        "lavapipe without robustness",
    );
    // The device-address access is one name in both encodings; the descriptor access is not.
    assert_eq!(
        robust_names
            .intersection(&checked_names)
            .filter(|name| name.starts_with("access.atomic.buffer"))
            .collect::<Vec<_>>(),
        ["access.atomic.buffer.rmw.psb.checked"]
    );
    let unsafe_off = DevicePolicy {
        containment: ContainmentPolicy::UnsafeOff,
        isolation: GuestIsolation::Off,
        ..Default::default()
    };
    let off = profile("lavapipe", unsafe_off).expect("profile");
    let (plan, off_names) = names(&off);
    assert!(!plan.families.access.psb_contained);
    expect(
        &off_names,
        &[
            "access.atomic.buffer.rmw.psb",
            "access.atomic.buffer.rmw",
            "access.load.switch.psb_and_view",
            "access.store",
        ],
        "lavapipe with containment off",
    );
}

const ADDRESSED_COPIES: &str = r#"define void @addressed_copies(ptr addrspace(1) %wide, ptr addrspace(1) %words, i32 %t) {
entry:
  %o0 = atomicrmw add ptr addrspace(1) %wide, i64 1 monotonic, align 8
  %w8 = getelementptr inbounds i8, ptr addrspace(1) %wide, i64 8
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %words, ptr addrspace(1) align 4 %w8, i64 16, i1 false)
  %w24 = getelementptr inbounds i8, ptr addrspace(1) %wide, i64 24
  %d16 = getelementptr inbounds i8, ptr addrspace(1) %words, i64 16
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %w24, ptr addrspace(1) align 4 %d16, i64 16, i1 false)
  %w40 = getelementptr inbounds i8, ptr addrspace(1) %wide, i64 40
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %w40, i8 7, i64 8, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @addressed_copies, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"wide"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"words"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A memory intrinsic names each side by what it reaches. `[[buffer(0)]]` is a device address (a
/// typed atomic reaches it) and `[[buffer(1)]]` a descriptor view: a copy from the device address
/// to the view loads checked in the shader and stores under the profile's encoding, the reverse
/// copy the other way round, so they are `access.memcpy.from_psb` and `access.memcpy.to_psb`, never
/// one name for both; a fill of the device address is `access.memset.to_psb`. The names are the
/// same with robust descriptors and without them (the view side follows the module's buffer
/// encoding, the coverage key).
#[test]
fn a_memory_intrinsic_names_each_side_by_what_it_reaches() {
    let Some(bc) = assemble("addressed_copies", ADDRESSED_COPIES) else {
        return;
    };
    let entry = "addressed_copies";
    for (what, prof) in [
        ("lavapipe", p("lavapipe")),
        ("lavapipe without robustness", unrobust("lavapipe")),
    ] {
        l3(&bc, entry, &prof, what);
        let (plan, _) = typed_module(&bc, entry, &prof).expect("module");
        let intrinsics = plan
            .census()
            .into_iter()
            .filter(|(name, _)| name.starts_with("access.mem"))
            .map(|(name, (uses, _))| (name, uses))
            .collect::<Vec<_>>();
        assert_eq!(
            intrinsics,
            [
                ("access.memcpy.from_psb".to_owned(), 1),
                ("access.memcpy.to_psb".to_owned(), 1),
                ("access.memset.to_psb".to_owned(), 1),
            ],
            "{what}"
        );
    }
}

const FENCES: &str = r#"define void @fences(ptr addrspace(1) %buf, i32 %t) {
entry:
  store i32 %t, ptr addrspace(1) %buf, align 4
  fence seq_cst
  fence syncscope("singlethread") acquire
  fence release
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @fences, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A system-scope LLVM fence is a device-scope `OpMemoryBarrier`; a single-thread fence is nothing;
/// both pass L3 on every profile under `GLSL450`. The `seq_cst` one is also ordered through the
/// command buffer's status word (`memory_model.fence.seq_cst`), reached by address where the
/// profile addresses buffers and through the kernel's binding of it elsewhere.
#[test]
fn llvm_fences_lower_to_memory_barriers() {
    let Some(bc) = assemble("fences", FENCES) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "fences", &prof, stem);
        let a = compile(&bc, "fences", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(status_reach(&a), status_reach_on(&prof), "{stem}");
        let (plan, m) = typed_module(&bc, "fences", &prof).expect("module");
        let census = plan.census();
        assert_eq!(census["memory_model.fence"].0, 1, "{stem}");
        assert_eq!(census["memory_model.fence.seq_cst"].0, 1, "{stem}");
        assert_eq!(census["memory_model.fence.none"].0, 1, "{stem}");
        assert!(plan.families.memory_model.device_scope, "{stem}");
        let res = resources(&plan, &[vec![0; 1]], 1);
        for out in run_orders(&m, &res).expect("run") {
            assert_eq!(out[0], vec![0], "{stem}");
        }
    }
}

/// The typed module of `entry` under `inputs`.
fn typed_module_with(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> (Plan, TypedModule) {
    let m = &mut meter();
    let decoded = air::decode(bc, m).expect("decode");
    let req = air::SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..air::SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, m).expect("specialize");
    let memory = analyze_memory(&spec, p, m).expect("analyze");
    let plan = select(&spec, &memory, p, (inputs, WidestPin::Weighed), m).expect("select");
    let module = emit_module(&spec, &plan, m).expect("emit").module;
    verify(&module, p, m).expect("verify");
    (plan, module)
}

const TABLE: u64 = 0x0000_0010_0000_0000;
const DATA: u64 = 0x0000_0020_0000_0000;
const SINK: u64 = 0x0000_0030_0000_0000;

/// Through a device address (a `BufferRangeLimit` variant), in-bounds atomics are exact and an
/// out-of-bounds one operates on the sink: result 0, the allocation untouched.
#[test]
fn device_address_atomics_are_exact_and_contained() {
    let Some(bc) = assemble("oob", OOB) else {
        return;
    };
    let prof = p("moltenvk-m3");
    let mut inputs = LowerInputs::new();
    inputs.psb_slots.insert(SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index: 0,
    });
    let a = compile_bitcode(
        &bc,
        EntryRef::Name("oob"),
        &FunctionConstantValues::defaults(),
        &prof,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .expect("compile");
    link_and_validate(&a, &prof).expect("L3");
    let (plan, m) = typed_module_with(&bc, "oob", &prof, &inputs);
    let size = 32u64;
    let mut table = vec![0u8; 32 * 3];
    table[32..40].copy_from_slice(&SINK.to_le_bytes());
    table[48..56].copy_from_slice(&64u64.to_le_bytes());
    table[64..72].copy_from_slice(&DATA.to_le_bytes());
    table[72..80].copy_from_slice(&DATA.to_le_bytes());
    table[80..88].copy_from_slice(&size.to_le_bytes());
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for fl in &plan.push.fields {
        match (fl.field, fl.loc) {
            (PushField::RootTableAddress, PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 8].copy_from_slice(&TABLE.to_le_bytes())
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), 4);
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, 4);
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: None,
                address: Some(DATA),
                bytes: vec![0; size as usize],
            },
            Buffer {
                slot: None,
                address: Some(SINK),
                bytes: vec![0; 64],
            },
            Buffer {
                slot: None,
                address: Some(TABLE),
                bytes: table,
            },
        ],
        push,
        spec: s,
    };
    for out in run_orders(&m, &res).expect("run") {
        assert_eq!(out[0], vec![5, 0, 0, 0, 0, 0, 0, 0]);
    }
}

const LLVM_RMW: &str = r#"define void @llvm_rmw(ptr addrspace(1) %buf, i32 %t) {
entry:
  %o0 = atomicrmw add ptr addrspace(1) %buf, i32 1 monotonic, align 4
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 1
  %tm = sub i32 %t, 32
  %o1 = atomicrmw max ptr addrspace(1) %p1, i32 %tm seq_cst, align 4
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 2
  %t5 = add i32 %t, 5
  %o2 = atomicrmw umin ptr addrspace(1) %p2, i32 %t5 acq_rel, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 3
  %o3 = atomicrmw nand ptr addrspace(1) %p3, i32 -1 monotonic, align 4
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 4
  %o4 = atomicrmw uinc_wrap ptr addrspace(1) %p4, i32 10 monotonic, align 4
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 5
  %o5 = atomicrmw udec_wrap ptr addrspace(1) %p5, i32 10 monotonic, align 4
  %p6 = getelementptr inbounds float, ptr addrspace(1) %buf, i32 6
  %f6 = atomicrmw fadd ptr addrspace(1) %p6, float 5.000000e-01 monotonic, align 4
  %p7 = getelementptr inbounds float, ptr addrspace(1) %buf, i32 7
  %f7 = atomicrmw fsub ptr addrspace(1) %p7, float 2.500000e-01 monotonic, align 4
  %tf = uitofp i32 %t to float
  %p8 = getelementptr inbounds float, ptr addrspace(1) %buf, i32 8
  %f8 = atomicrmw fmax ptr addrspace(1) %p8, float %tf monotonic, align 4
  %tf1 = fadd float %tf, 1.000000e+00
  %p9 = getelementptr inbounds float, ptr addrspace(1) %buf, i32 9
  %f9 = atomicrmw fmin ptr addrspace(1) %p9, float %tf1 monotonic, align 4
  %p10 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 10
  %o10 = atomicrmw xchg ptr addrspace(1) %p10, i32 %t monotonic, align 4
  %p11 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 11
  store atomic i32 77, ptr addrspace(1) %p11 release, align 4
  %p12 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 12
  %l12 = load atomic i32, ptr addrspace(1) %p12 acquire, align 4
  %p13 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 13
  %t1 = add i32 %t, 1
  %cx = cmpxchg ptr addrspace(1) %p13, i32 0, i32 %t1 seq_cst seq_cst, align 4
  %cold = extractvalue { i32, i1 } %cx, 0
  %cok = extractvalue { i32, i1 } %cx, 1
  %cokw = zext i1 %cok to i32
  %i0 = add i32 %t, 16
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i0
  store i32 %o0, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 80
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i1
  store i32 %cold, ptr addrspace(1) %q1, align 4
  %i2 = add i32 %t, 144
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i2
  store i32 %cokw, ptr addrspace(1) %q2, align 4
  %i3 = add i32 %t, 208
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i3
  store i32 %l12, ptr addrspace(1) %q3, align 4
  %i4 = add i32 %t, 272
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i4
  store i32 %o3, ptr addrspace(1) %q4, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @llvm_rmw, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// LLVM `atomicrmw` (every operator, the ones SPIR-V lacks as compare-and-swap loops), `cmpxchg`
/// (its `{old, success}` pair), and atomic `load`/`store` on a buffer slot are exact under 64
/// contending invocations in both scheduling orders, and pass L3 on every profile.
#[test]
fn llvm_atomics_are_exact_under_contention() {
    let Some(bc) = assemble("llvm_rmw", LLVM_RMW) else {
        return;
    };
    const N: u32 = 64;
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "llvm_rmw", &prof, stem);
        let (plan, m) = typed_module(&bc, "llvm_rmw", &prof).expect("module");
        let native = native_float_add(&prof);
        let loops = if native { 2 } else { 4 };
        // Every atomic reaches the one slot, through its device address where the float add is
        // native.
        let reach = |name: &str| {
            if native {
                format!("{name}.psb")
            } else {
                name.to_owned()
            }
        };
        assert_eq!(
            uses(&plan, &reach("access.atomic.buffer.float_cas")),
            loops,
            "{stem}"
        );
        assert_eq!(
            uses(&plan, &reach("access.atomic.buffer.float_add")),
            4 - loops,
            "{stem}"
        );
        assert_eq!(uses(&plan, &reach("access.atomic.buffer.cas")), 3, "{stem}");
        assert_eq!(uses(&plan, &reach("access.atomic.buffer.rmw")), 4, "{stem}");
        assert_eq!(
            uses(&plan, &reach("access.atomic.buffer.cmpxchg")),
            1,
            "{stem}"
        );
        let mut init = vec![0u32; 16 + 5 * N as usize];
        init[2] = u32::MAX;
        init[3] = 0x1234_5678;
        init[5] = 3;
        init[7] = f(100.0);
        init[8] = f(-1.0);
        init[9] = f(1000.0);
        init[12] = 0xABCD;
        let res = resources(&plan, &[init], N);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            assert_eq!(w[0], N, "{stem}: add");
            assert_eq!(w[1], N - 1 - 32, "{stem}: max");
            assert_eq!(w[2], 5, "{stem}: umin");
            assert_eq!(
                w[3], 0x1234_5678,
                "{stem}: nand toggles an even number of times"
            );
            assert_eq!(w[4], N % 11, "{stem}: uinc_wrap");
            assert_eq!(w[5], 5, "{stem}: udec_wrap");
            assert_eq!(w[6], f(0.5 * N as f32), "{stem}: fadd");
            assert_eq!(w[7], f(100.0 - 0.25 * N as f32), "{stem}: fsub");
            assert_eq!(w[8], f((N - 1) as f32), "{stem}: fmax");
            assert_eq!(w[9], f(1.0), "{stem}: fmin");
            assert!(w[10] < N, "{stem}: xchg");
            assert_eq!(w[11], 77, "{stem}: store");
            let mut olds: Vec<u32> = w[16..16 + N as usize].to_vec();
            olds.sort_unstable();
            assert_eq!(olds, (0..N).collect::<Vec<_>>(), "{stem}: add olds");
            let won: Vec<usize> = (0..N as usize).filter(|t| w[144 + t] == 1).collect();
            assert_eq!(won.len(), 1, "{stem}: one cmpxchg winner");
            for t in 0..N as usize {
                let want = if t == won[0] { 0 } else { won[0] as u32 + 1 };
                assert_eq!(w[80 + t], want, "{stem}: cmpxchg old {t}");
                assert_eq!(w[208 + t], 0xABCD, "{stem}: load");
            }
            let toggled = w[272..272 + N as usize]
                .iter()
                .filter(|x| **x == !0x1234_5678u32)
                .count();
            assert_eq!(toggled, N as usize / 2, "{stem}: nand olds");
        }
    }
}

const LLVM_TG: &str = r#"@c = internal addrspace(3) global i32 undef, align 4

define void @llvm_tg(ptr addrspace(1) %out, i32 %t) {
entry:
  %first = icmp eq i32 %t, 0
  br i1 %first, label %init, label %go

init:
  store atomic i32 0, ptr addrspace(3) @c monotonic, align 4
  br label %go

go:
  call void @air.wg.barrier(i32 2, i32 1)
  %o = atomicrmw add ptr addrspace(3) @c, i32 2 monotonic, align 4
  %cx = cmpxchg weak ptr addrspace(3) @c, i32 1, i32 1 acquire monotonic, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %v = load atomic i32, ptr addrspace(3) @c monotonic, align 4
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %o, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %v, ptr addrspace(1) %q1, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @llvm_tg, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// LLVM atomics on the threadgroup arena: the adds are exact (old values a permutation of the even
/// numbers below `2N`), a compare-exchange against an odd value never matches, and every invocation
/// loads the total after the barrier.
#[test]
fn llvm_threadgroup_atomics_are_exact() {
    let Some(bc) = assemble("llvm_tg", LLVM_TG) else {
        return;
    };
    const N: u32 = 64;
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "llvm_tg", &prof, stem);
        let (plan, m) = typed_module(&bc, "llvm_tg", &prof).expect("module");
        assert_eq!(
            uses(&plan, "access.atomic.threadgroup.cmpxchg"),
            1,
            "{stem}"
        );
        let res = resources(&plan, &[vec![0; 2 * N as usize]], N);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            let mut olds: Vec<u32> = w[..N as usize].to_vec();
            olds.sort_unstable();
            assert_eq!(olds, (0..N).map(|k| 2 * k).collect::<Vec<_>>(), "{stem}");
            assert!(w[N as usize..].iter().all(|v| *v == 2 * N), "{stem}");
        }
    }
}

const NARROW: &str = r#"define void @narrow(ptr addrspace(1) %buf, i32 %t) {
entry:
  %t8 = trunc i32 %t to i8
  %t16 = trunc i32 %t to i16
  %o0 = atomicrmw add ptr addrspace(1) %buf, i8 1 monotonic, align 1
  %b1 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 1
  %o1 = atomicrmw add ptr addrspace(1) %b1, i8 3 monotonic, align 1
  %b2 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 2
  %o2 = atomicrmw umax ptr addrspace(1) %b2, i8 %t8 monotonic, align 1
  %b3 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 3
  %tm = sub i8 %t8, 32
  %o3 = atomicrmw min ptr addrspace(1) %b3, i8 %tm monotonic, align 1
  %h2 = getelementptr inbounds i16, ptr addrspace(1) %buf, i32 2
  %o4 = atomicrmw sub ptr addrspace(1) %h2, i16 5 seq_cst, align 2
  %h3 = getelementptr inbounds i16, ptr addrspace(1) %buf, i32 3
  %sh = and i16 %t16, 15
  %bit = shl i16 1, %sh
  %o5 = atomicrmw or ptr addrspace(1) %h3, i16 %bit monotonic, align 2
  %b8 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 8
  %o6 = atomicrmw xor ptr addrspace(1) %b8, i8 %t8 monotonic, align 1
  %b9 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 9
  %lo = and i8 %t8, 7
  %lbit = shl i8 1, %lo
  %keep = xor i8 %lbit, -1
  %o7 = atomicrmw and ptr addrspace(1) %b9, i8 %keep monotonic, align 1
  %b10 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 10
  %o8 = atomicrmw xchg ptr addrspace(1) %b10, i8 %t8 monotonic, align 1
  %b11 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 11
  %o9 = atomicrmw nand ptr addrspace(1) %b11, i8 -1 monotonic, align 1
  %h6 = getelementptr inbounds i16, ptr addrspace(1) %buf, i32 6
  %t1 = add i16 %t16, 1
  %cx = cmpxchg ptr addrspace(1) %h6, i16 0, i16 %t1 seq_cst seq_cst, align 2
  %b14 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 14
  store atomic i8 77, ptr addrspace(1) %b14 release, align 1
  %b15 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 15
  %l = load atomic i8, ptr addrspace(1) %b15 acquire, align 1
  %b16 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 16
  %o10 = atomicrmw uinc_wrap ptr addrspace(1) %b16, i8 10 monotonic, align 1
  %b17 = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 17
  %o11 = atomicrmw udec_wrap ptr addrspace(1) %b17, i8 10 monotonic, align 1
  %h9 = getelementptr inbounds i16, ptr addrspace(1) %buf, i32 9
  %t5 = add i16 %t16, 5
  %o12 = atomicrmw umin ptr addrspace(1) %h9, i16 %t5 monotonic, align 2
  %cold = extractvalue { i16, i1 } %cx, 0
  %cok = extractvalue { i16, i1 } %cx, 1
  %w0 = zext i8 %o0 to i32
  %i0 = add i32 %t, 16
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i0
  store i32 %w0, ptr addrspace(1) %q0, align 4
  %w1 = zext i16 %cold to i32
  %i1 = add i32 %t, 80
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i1
  store i32 %w1, ptr addrspace(1) %q1, align 4
  %w2 = zext i1 %cok to i32
  %i2 = add i32 %t, 144
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i2
  store i32 %w2, ptr addrspace(1) %q2, align 4
  %w3 = zext i8 %l to i32
  %i3 = add i32 %t, 208
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i3
  store i32 %w3, ptr addrspace(1) %q3, align 4
  %w4 = zext i8 %o9 to i32
  %i4 = add i32 %t, 272
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i4
  store i32 %w4, ptr addrspace(1) %q4, align 4
  %w5 = zext i16 %o4 to i32
  %i5 = add i32 %t, 336
  %q5 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i5
  store i32 %w5, ptr addrspace(1) %q5, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// LLVM atomics on `i8` and `i16` fields that share words with each other are exact under 64
/// contending invocations in both scheduling orders, and pass L3 on every profile: `and`, `or`,
/// `xor`, and `load` are one word atomic each, every other operation a compare-and-swap loop on the
/// containing word that leaves the neighbouring fields as the other invocations wrote them.
#[test]
fn narrow_atomics_are_exact_under_contention() {
    let Some(bc) = assemble("narrow", NARROW) else {
        return;
    };
    const N: u32 = 64;
    for stem in PROFILES {
        for prof in [p(stem), unrobust(stem)] {
            l3(&bc, "narrow", &prof, stem);
            let (plan, m) = typed_module(&bc, "narrow", &prof).expect("module");
            // or, xor, and, load.
            assert_eq!(uses(&plan, "access.atomic.buffer.narrow"), 4, "{stem}");
            // add ×2, umax, min, sub, xchg, nand, cmpxchg, store, uinc_wrap, udec_wrap, umin.
            assert_eq!(uses(&plan, "access.atomic.buffer.narrow_cas"), 12, "{stem}");
            let mut init = vec![0u32; 16 + 6 * N as usize];
            init[1] = 1000;
            init[2] = 0x1200_FF5A;
            init[3] = 0xAB00_0000;
            init[4] = 0xFFFF_0300;
            let res = resources(&plan, &[init], N);
            for out in run_orders(&m, &res).expect("run") {
                let w = &out[0];
                let b = |i: usize| (w[i / 4] >> (8 * (i % 4))) & 0xFF;
                let h = |i: usize| (w[i / 2] >> (16 * (i % 2))) & 0xFFFF;
                assert_eq!(b(0), N, "{stem}: add");
                assert_eq!(b(1), (3 * N) & 0xFF, "{stem}: add wraps at 8 bits");
                assert_eq!(b(2), N - 1, "{stem}: umax");
                assert_eq!(b(3), 0xE0, "{stem}: signed min (-32)");
                assert_eq!(h(2), 1000 - 5 * N, "{stem}: sub");
                assert_eq!(h(3), 0xFFFF, "{stem}: or");
                assert_eq!(b(8), 0x5A, "{stem}: xor of 0..64");
                assert_eq!(b(9), 0, "{stem}: and");
                assert!(b(10) < N, "{stem}: xchg");
                assert_eq!(b(11), 0x12, "{stem}: nand toggled an even number of times");
                assert_eq!(b(14), 77, "{stem}: store");
                assert_eq!(b(15), 0xAB, "{stem}: load leaves the byte");
                assert_eq!(b(16), N % 11, "{stem}: uinc_wrap");
                assert_eq!(b(17), 5, "{stem}: udec_wrap");
                assert_eq!(h(9), 5, "{stem}: umin");
                let mut olds: Vec<u32> = w[16..16 + N as usize].to_vec();
                olds.sort_unstable();
                assert_eq!(olds, (0..N).collect::<Vec<_>>(), "{stem}: add olds");
                let won: Vec<usize> = (0..N as usize).filter(|t| w[144 + t] == 1).collect();
                assert_eq!(won.len(), 1, "{stem}: one cmpxchg winner");
                assert_eq!(h(6), won[0] as u32 + 1, "{stem}: cmpxchg");
                for t in 0..N as usize {
                    let want = if t == won[0] { 0 } else { won[0] as u32 + 1 };
                    assert_eq!(w[80 + t], want, "{stem}: cmpxchg old {t}");
                    assert_eq!(w[208 + t], 0xAB, "{stem}: load {t}");
                    assert!(w[272 + t] == 0x12 || w[272 + t] == 0xED, "{stem}: nand old");
                }
                let mut subs: Vec<u32> = w[336..336 + N as usize].to_vec();
                subs.sort_unstable();
                let want: Vec<u32> = (0..N).rev().map(|k| 1000 - 5 * (k + 1) + 5).collect();
                assert_eq!(subs, want, "{stem}: sub olds");
            }
        }
    }
}

const NARROW_TG: &str = r#"@c = internal addrspace(3) global i32 undef, align 4

define void @narrow_tg(ptr addrspace(1) %out, i32 %t) {
entry:
  %first = icmp eq i32 %t, 0
  br i1 %first, label %init, label %go

init:
  store atomic i32 0, ptr addrspace(3) @c monotonic, align 4
  br label %go

go:
  call void @air.wg.barrier(i32 2, i32 1)
  %k = and i32 %t, 3
  %b = getelementptr inbounds i8, ptr addrspace(3) @c, i32 %k
  %o = atomicrmw add ptr addrspace(3) %b, i8 1 monotonic, align 1
  call void @air.wg.barrier(i32 2, i32 1)
  %v = load atomic i32, ptr addrspace(3) @c monotonic, align 4
  %h = getelementptr inbounds i16, ptr addrspace(3) @c, i32 1
  %hv = load atomic i16, ptr addrspace(3) %h monotonic, align 2
  %ow = zext i8 %o to i32
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %ow, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %v, ptr addrspace(1) %q1, align 4
  %hw = zext i16 %hv to i32
  %i2 = add i32 %t, 128
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i2
  store i32 %hw, ptr addrspace(1) %q2, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @narrow_tg, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Byte atomics on the four fields of one threadgroup word are exact (each field's old values a
/// permutation of `0..N/4`), and a word and a half-word atomic load read the fields.
#[test]
fn narrow_threadgroup_atomics_are_exact() {
    let Some(bc) = assemble("narrow_tg", NARROW_TG) else {
        return;
    };
    const N: u32 = 64;
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "narrow_tg", &prof, stem);
        let (plan, m) = typed_module(&bc, "narrow_tg", &prof).expect("module");
        assert_eq!(
            uses(&plan, "access.atomic.threadgroup.narrow_cas"),
            1,
            "{stem}"
        );
        assert_eq!(uses(&plan, "access.atomic.threadgroup.narrow"), 1, "{stem}");
        let res = resources(&plan, &[vec![0; 3 * N as usize]], N);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            for k in 0..4 {
                let mut olds: Vec<u32> = (0..N as usize)
                    .filter(|t| t % 4 == k)
                    .map(|t| w[t])
                    .collect();
                olds.sort_unstable();
                assert_eq!(olds, (0..N / 4).collect::<Vec<_>>(), "{stem}: field {k}");
            }
            assert!(w[64..128].iter().all(|v| *v == 0x1010_1010), "{stem}");
            assert!(w[128..192].iter().all(|v| *v == 0x1010), "{stem}");
        }
    }
}

const NARROW_OOB: &str = r#"define void @narrow_oob(ptr addrspace(1) %buf, i32 %t) {
entry:
  %i = mul i32 %t, 4001
  %p = getelementptr inbounds i8, ptr addrspace(1) %buf, i32 %i
  %o = atomicrmw add ptr addrspace(1) %p, i8 5 monotonic, align 1
  %x = atomicrmw or ptr addrspace(1) %p, i8 8 monotonic, align 1
  %ow = zext i8 %o to i32
  %xw = zext i8 %x to i32
  %j = add i32 %t, 1
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %j
  store i32 %ow, ptr addrspace(1) %q, align 4
  %k = add i32 %t, 5
  %r = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %k
  store i32 %xw, ptr addrspace(1) %r, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow_oob, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// Invocation 0's byte atomics act on byte 0; the others' are 4001·t bytes out (at every field
/// position): contained (no trap, old value 0, nothing written) by the in-shader check.
#[test]
fn an_out_of_bounds_narrow_atomic_is_contained() {
    let Some(bc) = assemble("narrow_oob", NARROW_OOB) else {
        return;
    };
    for stem in PROFILES {
        l3(&bc, "narrow_oob", &p(stem), stem);
        let prof = unrobust(stem);
        l3(&bc, "narrow_oob", &prof, stem);
        let (plan, m) = typed_module(&bc, "narrow_oob", &prof).expect("module");
        assert!(
            plan.census()
                .contains_key("access.atomic.buffer.narrow_cas.checked"),
            "{stem}"
        );
        let res = resources(&plan, &[vec![0; 12]], 4);
        for out in run_orders(&m, &res).expect("run") {
            assert_eq!(out[0], vec![13, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0], "{stem}");
        }
    }
}

const COHERENT: &str = r#"@tg = internal addrspace(3) global [64 x i32] undef, align 4

define void @coherent(ptr addrspace(1) %buf, i32 %t) {
entry:
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %t
  %t3 = mul i32 %t, 3
  call void @air.store.device_coherent.i32.p1i32(i32 %t3, ptr addrspace(1) %p)
  %g = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %g, align 4
  call void @air.wg.barrier(i32 3, i32 1)
  %n = add i32 %t, 1
  %m = and i32 %n, 63
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %m
  %v = call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %q)
  %h = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %m
  %w = load i32, ptr addrspace(3) %h, align 4
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i1
  store i32 %v, ptr addrspace(1) %q1, align 4
  %i2 = add i32 %t, 128
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %i2
  store i32 %w, ptr addrspace(1) %q2, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))
declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @coherent, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A function with `*_coherent` accesses selects the Vulkan memory model where the profile has
/// `vulkanMemoryModel` and its device scope. Under it, coherent stores carry
/// `MakePointerAvailable` and coherent loads `MakePointerVisible` (both `NonPrivatePointer`,
/// `Device` scope), and every other buffer and threadgroup access is `NonPrivatePointer` (so the
/// barrier orders it as under `GLSL450`). Elsewhere (the floor) it keeps `GLSL450`: no access
/// carries a memory-model operand, and every storage buffer is decorated `Coherent`. Either way the
/// kernel's neighbour exchange through the barrier is exact in both scheduling orders, and each
/// form is exercised.
#[test]
fn coherent_accesses_select_the_vulkan_memory_model() {
    use crate::spirv::ir::Inst;
    let Some(bc) = assemble("coherent", COHERENT) else {
        return;
    };
    const N: usize = 64;
    let mut forms = [0; 2];
    for stem in PROFILES {
        let prof = p(stem);
        let fields = prof.fields();
        let vulkan = fields.vulkan_memory_model
            && fields
                .enabled
                .features
                .contains(&VkFeature::VulkanMemoryModelDeviceScope);
        forms[usize::from(vulkan)] += 1;
        if !vulkan {
            // Every buffer access goes through a `Coherent` variable (the interlock's check).
            let a = compile(&bc, "coherent", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let (_, words) =
                link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
            let coherent = crate::pipeline::testing::interlocked_coherence(&words)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(!coherent.vulkan, "{stem}: the memory model");
            assert!(coherent.buffer >= 4, "{stem}: {coherent:?}");
        }
        l3(&bc, "coherent", &prof, stem);
        let (plan, m) = typed_module(&bc, "coherent", &prof).expect("module");
        assert_eq!(
            plan.families.memory_model.coherent_resources, !vulkan,
            "{stem}"
        );
        assert!(plan.families.memory_model.device_scope, "{stem}");
        let caps = &m.requirements().spirv_capabilities;
        assert_eq!(
            caps.iter().any(|c| format!("{c:?}") == "VulkanMemoryModel"),
            vulkan,
            "{stem}: {caps:?}"
        );
        let (mut available, mut visible, mut plain_shared, mut volatile) = (0, 0, 0, 0);
        for b in m.blocks() {
            for n in &b.insts {
                match &n.inst {
                    Inst::Store { access, .. } if access.make_available.is_some() => {
                        assert!(access.non_private, "{stem}");
                        available += 1;
                    }
                    Inst::Load { access, .. } if access.make_visible.is_some() => {
                        assert!(access.non_private && access.volatile, "{stem}");
                        visible += 1;
                    }
                    Inst::Load { access, .. } | Inst::Store { access, .. }
                        if access.non_private =>
                    {
                        plain_shared += 1;
                    }
                    Inst::Load { access, .. } if access.volatile => volatile += 1,
                    _ => {}
                }
            }
        }
        if vulkan {
            assert!(available >= 1 && visible >= 1, "{stem}");
            // The threadgroup store and load, and the two plain buffer stores.
            assert!(plain_shared >= 4, "{stem}: {plain_shared}");
        } else {
            assert_eq!((available, visible, plain_shared), (0, 0, 0), "{stem}");
            // The volatile coherent load keeps its `Volatile` operand.
            assert!(volatile >= 1, "{stem}");
        }
        let res = resources(&plan, &[vec![0; 3 * N]], N as u32);
        for out in run_orders(&m, &res).expect("run") {
            let w = &out[0];
            for t in 0..N {
                let n = (t + 1) % N;
                assert_eq!(w[t], 3 * t as u32, "{stem}: coherent store {t}");
                assert_eq!(w[N + t], 3 * n as u32, "{stem}: coherent load {t}");
                assert_eq!(w[2 * N + t], n as u32, "{stem}: threadgroup load {t}");
            }
        }
    }
    assert!(forms.iter().all(|&n| n > 0), "{forms:?}");
}

/// A `texture2d<uint, access::read_write>` read and written at the thread's position, and a word
/// stored to a buffer by `store` (`@air.store.device_coherent...` or a plain `store`).
fn coherent_image_kernel(store: &str) -> String {
    format!(
        r#"define void @coherent_image(ptr addrspace(1) %tex, ptr addrspace(1) %buf, <2 x i32> %gid) {{
entry:
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ <4 x i32>, i8 }} @air.read_texture_2d.u.v4i32(ptr addrspace(1) %tex, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue {{ <4 x i32>, i8 }} %r, 0
  %w = add <4 x i32> %v, <i32 1, i32 2, i32 3, i32 4>
  call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %tex, <2 x i32> %gid, <4 x i32> %w, i32 0, i32 3)
  %x = extractelement <2 x i32> %gid, i32 0
  %e = extractelement <4 x i32> %v, i32 0
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i32 %x
  {store}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x i32>, i8 }} @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)
declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))

!air.kernel = !{{!0}}
!0 = !{{ptr @coherent_image, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"tex"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

/// Under the Vulkan memory model a storage image's accesses would need `NonPrivateTexel` to stay
/// ordered by barriers, which is not emitted, so a function with a coherent access and a storage
/// image is `NotImplemented{Lowering(MemoryModel)}` on every profile with `vulkanMemoryModel`, and
/// on the others, whose `GLSL450` form keeps the same rule. The same kernel with a plain store
/// instead of the coherent one compiles and passes L3 everywhere.
#[test]
fn a_coherent_access_beside_a_storage_image_is_refused() {
    use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
    let Some(coherent) = assemble(
        "coherent_image",
        &coherent_image_kernel(
            "call void @air.store.device_coherent.i32.p1i32(i32 %e, ptr addrspace(1) %p)",
        ),
    ) else {
        return;
    };
    let Some(plain) = assemble(
        "plain_image",
        &coherent_image_kernel("store i32 %e, ptr addrspace(1) %p, align 4"),
    ) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    inputs.storage_images.insert(
        SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Texture,
            index: 0,
        },
        StorageImageChoice::new(crate::vk::Format::R32_UINT, StorageImageClass::Identity),
    );
    let with = |bc: &[u8], prof: &DeviceProfile| {
        compile_bitcode(
            bc,
            EntryRef::Name("coherent_image"),
            &FunctionConstantValues::defaults(),
            prof,
            &inputs,
            &mut meter(),
            SPIRV_WORDS,
        )
    };
    let want = Unsupported::not_implemented(Feature::Lowering(LowerFamily::MemoryModel));
    let mut modelled = 0;
    for stem in PROFILES {
        let prof = p(stem);
        let a = with(&plain, &prof).unwrap_or_else(|e| panic!("{stem}: plain: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: plain: L3: {e}"));
        let e = with(&coherent, &prof).expect_err("refused");
        let Error::Unsupported(u) = e else {
            panic!("{stem}: {e:?}");
        };
        assert_eq!(
            (u.feature, u.needs),
            (want.feature.clone(), want.needs.clone()),
            "{stem}"
        );
        if prof.fields().vulkan_memory_model {
            modelled += 1;
        }
    }
    assert!(modelled > 0, "no profile has vulkanMemoryModel");
}

const SPINS: &str = r#"@flag = internal addrspace(3) global i32 undef, align 4

define void @spins(ptr addrspace(1) %buf, i32 %t) {
entry:
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 1
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 2
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 3
  %first = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %buf, i32 0, i32 2, i1 true)
  store i32 %first, ptr addrspace(1) %p3, align 4
  %most = call float @air.atomic.global.max.f32(ptr addrspace(1) %p3, float 1.000000e+00, i32 0, i32 2, i1 true)
  br label %data

data:
  %d = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p1, i32 0, i32 2, i1 true)
  %dz = icmp eq i32 %d, 0
  br i1 %dz, label %data, label %ctrl

ctrl:
  %c = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p2, i32 0, i32 2, i1 true)
  %c1 = icmp eq i32 %c, 1
  br i1 %c1, label %test, label %latch

test:
  %big = icmp ugt i32 %t, 3
  br i1 %big, label %poll, label %latch

latch:
  br label %ctrl

poll:
  %lead = icmp eq i32 %t, 0
  br i1 %lead, label %read, label %wait

read:
  %f = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p2, i32 0, i32 2, i1 true)
  store i32 %f, ptr addrspace(3) @flag, align 4
  br label %wait

wait:
  call void @air.wg.barrier(i32 2, i32 1)
  %g = load i32, ptr addrspace(3) @flag, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %gz = icmp eq i32 %g, 0
  br i1 %gz, label %poll, label %count

count:
  %j = phi i32 [ 0, %wait ], [ %j1, %count ]
  %v = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p3, i32 0, i32 2, i1 true)
  %w = call i32 @air.atomic.global.load.i32(ptr addrspace(1) %p1, i32 2, i32 2, i1 true)
  %vw = add i32 %v, %w
  store i32 %vw, ptr addrspace(1) %buf, align 4
  %j1 = add i32 %j, 1
  %jd = icmp eq i32 %j1, 4
  br i1 %jd, label %exit, label %count

exit:
  ret void
}

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)
declare float @air.atomic.global.max.f32(ptr addrspace(1), float, i32, i32, i1)
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @spins, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// A relaxed atomic load acquires its memory exactly where an exit of a loop holding it depends on
/// it: through its value (`data`), through the branch that decides whether the iteration reaches
/// the exit test (`ctrl`, whose exit tests only `t`), and through a threadgroup word a store of the
/// loop writes (`poll`); a load outside every loop, one a counted loop repeats only to store
/// (`count`), and an acquire load keep what they ask. The read of the compare-and-swap loop a float
/// `max` is, which that loop exits on, acquires. Every profile, whatever its robustness: the rule
/// reads the function's structure only.
#[test]
fn a_relaxed_load_acquires_where_its_loops_exit_depends_on_it() {
    use crate::spirv::ir::{Inst, MemoryOrder, Scope};
    let Some(bc) = assemble("spins", SPINS) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "spins", &prof, stem);
        let (plan, m) = typed_module(&bc, "spins", &prof).expect("module");
        assert_eq!(uses(&plan, "access.atomic.buffer.spin_load"), 3, "{stem}");
        assert_eq!(uses(&plan, "access.atomic.buffer.load"), 3, "{stem}");
        assert_eq!(uses(&plan, "access.atomic.buffer.float_cas"), 1, "{stem}");
        // The buffer's atomic loads (`Device` scope; the arena's own word loads are `Workgroup`).
        let (mut acquire, mut relaxed) = (0, 0);
        for b in m.blocks() {
            for n in &b.insts {
                if let Inst::AtomicLoad { sem, scope, .. } = &n.inst {
                    if scope.scope != Scope::Device {
                        continue;
                    }
                    match sem.sem.order {
                        MemoryOrder::Acquire => {
                            assert!(sem.sem.uniform_memory, "{stem}: {:?}", sem.sem);
                            acquire += 1;
                        }
                        _ => relaxed += 1,
                    }
                }
            }
        }
        // The three spin reads, the acquire load and the loop's read acquire; the other two stay
        // relaxed.
        assert_eq!((acquire, relaxed), (5, 2), "{stem}");
    }
}
