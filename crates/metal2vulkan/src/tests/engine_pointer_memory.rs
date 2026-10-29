//! Atomics and stored pointers through dynamic roots end to end.
//!
//! - Atomics through a wide (dynamic-root) pointer act on the allocation its root entry names:
//!   every operation kind is exact under 64 contending invocations in both scheduling orders
//!   (`air` and LLVM atomics, a compare-and-swap loop, a native or looped `f32` add, and `u64`
//!   atomics where the profile has them); an out-of-bounds one acts on the sink and yields 0,
//!   and one through root 0 is counted under isolation. `i8`/`i16` atomics act on their fields
//!   of the containing words, exact and contained the same way.
//! - Coherent loads and stores through a wide pointer select the Vulkan memory model and carry
//!   the coherent memory operands.
//! - Memory intrinsics through wide pointers are exact and contained, and a `memmove` between
//!   endpoints that may alias compares their Metal addresses.
//! - Pointers stored into buffers and threadgroup cells are guest addresses that resolve back;
//!   stored and loaded volatile, their words are volatile accesses.
//! - A private struct zero-filled by `memset` before its pointers are stored stays a forwarding
//!   cell, and a pointer slot the fill covers reads null, directly and through a copy; a fill over
//!   part of a slot opens the cell.
//!
//! Every module passes L3 on the tested profiles and runs on the CPU interpreter. `llvm-as`
//! assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::status::{ExecutionStatus, StatusWord};
use crate::budget::SPIRV_WORDS;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::FunctionConstantValues;
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::interface::{BufferForm, PushLoc, Window};
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, meter, typed_module_with,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, TrapKind};
use crate::spirv::ir::{Inst, StorageClass, Type, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Device addresses of the interpreter's address-backed buffers.
const TABLE: u64 = 0x0000_0010_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;

/// Host (device) and guest (Metal) addresses of `[[buffer(i)]]` (distinct domains, so a mix-up
/// of the two faults the interpreter or reads the wrong bytes).
const fn slot_host(i: u64) -> u64 {
    0x0000_0020_0000_0000 + i * 0x1_0000_0000
}
const fn slot_guest(i: u64) -> u64 {
    0x0000_0001_0000_0000 + i * 0x1_0000
}
/// Host and guest addresses of the extra resident allocation `k`.
const fn alloc_host(k: u64) -> u64 {
    0x0000_0040_0000_0000 + k * 0x1_0000_0000
}
const fn alloc_guest(k: u64) -> u64 {
    0x0000_0003_0000_0000 + k * 0x1_0000
}

fn profile(stem: &str, robust: bool) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if !robust {
        desc.features.remove(&VkFeature::RobustBufferAccess);
        desc.features.remove(&VkFeature::RobustBufferAccess2);
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Profile-A profiles in both descriptor encodings, with and without `u64` buffer atomics and
/// the native `f32` add.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = Vec::new();
    for stem in ["moltenvk-m3", "lavapipe", "amd-rdna2-apu-radv"] {
        for robust in [true, false] {
            out.push((format!("{stem} robust={robust}"), profile(stem, robust)));
        }
    }
    out
}

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-pointer-memory-{}-{}-{name}",
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

/// Kernel metadata for `name`: `buffers` read-write `device` buffers of element type `uint`
/// (`air.arg_type_size` 4), then the thread's index in its threadgroup.
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

/// A kernel `name(b0, .., b{buffers-1}, t)` with `body` and the declarations `decls`.
fn kernel(name: &str, buffers: u32, body: &str, decls: &str) -> String {
    let params: Vec<String> = (0..buffers)
        .map(|i| format!("ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}"))
        .collect();
    format!(
        "define void @{name}({}, i32 %t) {{\nentry:\n{body}  ret void\n}}\n\n{decls}\n{}",
        params.join(", "),
        meta(name, buffers)
    )
}

fn kslot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
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

fn le_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The binder stand-in's memory: the kernel's buffers (each bound at its start, reachable by
/// device address with its root entry, and resident at its guest address) and extra resident
/// allocations.
struct Memory {
    slots: Vec<Vec<u8>>,
    allocs: Vec<Vec<u8>>,
}

/// `[[buffer(slot)]]` bound at byte `offset` of the extra allocation `alloc` instead of at the
/// start of its own buffer (whose bytes are then unused): its root entry names the allocation
/// with that bind offset, and its `SlotGuestBase` is the allocation's guest address plus the
/// offset. Only a device-address (`BufferForm::Psb`) slot reads its bytes through the entry.
#[derive(Clone, Copy)]
struct Bind {
    slot: usize,
    alloc: usize,
    offset: u64,
}

/// The buffers of one run, in the order the output lists them: the slots, the extra
/// allocations, the root table, the sink, and the status buffer.
struct Layout {
    slots: usize,
    allocs: usize,
}

impl Layout {
    fn slot(&self, i: usize) -> usize {
        i
    }
    fn alloc(&self, k: usize) -> usize {
        self.slots + k
    }
    fn sink(&self) -> usize {
        self.slots + self.allocs + 1
    }
    fn status(&self) -> usize {
        self.slots + self.allocs + 2
    }
}

/// The root table: null, the sink with the resident range, `[[buffer(i)]]` at root `2 + i` (at its
/// `binds` entry if it has one), and the resident set (every slot and allocation) sorted by guest
/// address.
fn root_table(mem: &Memory, binds: &[Bind]) -> Vec<u8> {
    let statics = mem.slots.len();
    let mut resident: Vec<(u64, u64, u64)> = Vec::new();
    for (i, b) in mem.slots.iter().enumerate() {
        resident.push((slot_host(i as u64), slot_guest(i as u64), b.len() as u64));
    }
    for (k, b) in mem.allocs.iter().enumerate() {
        resident.push((alloc_host(k as u64), alloc_guest(k as u64), b.len() as u64));
    }
    resident.sort_by_key(|r| r.1);
    let first = 2 + statics as u64;
    let n = first as usize + resident.len();
    let mut t = vec![0u8; 32 * n];
    let mut put = |root: usize, field: usize, v: u64| {
        let at = 32 * root + 8 * field;
        t[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    put(1, 0, SINK);
    put(1, 1, first);
    put(1, 2, resident.len() as u64);
    for (i, b) in mem.slots.iter().enumerate() {
        match binds.iter().find(|b| b.slot == i) {
            Some(b) => {
                put(2 + i, 0, alloc_host(b.alloc as u64));
                put(2 + i, 1, alloc_guest(b.alloc as u64));
                put(2 + i, 2, mem.allocs[b.alloc].len() as u64);
                put(2 + i, 3, b.offset);
            }
            None => {
                put(2 + i, 0, slot_host(i as u64));
                put(2 + i, 1, slot_guest(i as u64));
                put(2 + i, 2, b.len() as u64);
            }
        }
    }
    for (i, (h, g, s)) in resident.iter().enumerate() {
        let r = first as usize + i;
        put(r, 0, *h);
        put(r, 1, *g);
        put(r, 2, *s);
    }
    t
}

/// The resources of `plan` over `mem` for `threads` invocations in one threadgroup.
fn resources(plan: &Plan, mem: &Memory, threads: u32) -> (Resources, Layout) {
    resources_bound(plan, mem, threads, &[])
}

/// [`resources`] with the slots of `binds` bound inside extra allocations.
fn resources_bound(plan: &Plan, mem: &Memory, threads: u32, binds: &[Bind]) -> (Resources, Layout) {
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), u64::from(threads));
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("{:?} spilled to the ArgumentTable", f.field);
        };
        let at = offset as usize;
        match f.field {
            PushField::RootTableAddress => push[at..at + 8].copy_from_slice(&TABLE.to_le_bytes()),
            PushField::ExecutionStatusAddress => {
                push[at..at + 8].copy_from_slice(&STATUS.to_le_bytes())
            }
            // A descriptor view reads a bound slot from its own buffer, not the allocation.
            PushField::SlotBindOffset(s) if binds.iter().any(|b| b.slot == s.index as usize) => {
                panic!("bound slot {} is a descriptor view", s.index)
            }
            // Zero residuals (every other buffer is bound at its start) and a dispatch at the
            // origin.
            PushField::SlotBindOffset(_) | PushField::DispatchRegion => {
                push[at..at + f.size as usize].fill(0)
            }
            // The bound buffer's Metal address plus its bind offset.
            PushField::SlotGuestBase(s) => {
                let i = s.index as usize;
                let guest = match binds.iter().find(|b| b.slot == i) {
                    Some(b) => alloc_guest(b.alloc as u64) + b.offset,
                    None => slot_guest(i as u64),
                };
                push[at..at + 8].copy_from_slice(&guest.to_le_bytes())
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let mut buffers: Vec<Buffer> = mem
        .slots
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(kslot(i as u32)),
            address: Some(slot_host(i as u64)),
            bytes: b.clone(),
        })
        .collect();
    for (k, b) in mem.allocs.iter().enumerate() {
        buffers.push(Buffer {
            slot: None,
            address: Some(alloc_host(k as u64)),
            bytes: b.clone(),
        });
    }
    buffers.push(Buffer {
        slot: None,
        address: Some(TABLE),
        bytes: root_table(mem, binds),
    });
    buffers.push(Buffer {
        slot: None,
        address: Some(SINK),
        bytes: vec![0; 64],
    });
    // The status buffer, reached by its device address only.
    buffers.push(Buffer {
        slot: None,
        address: Some(STATUS),
        bytes: ExecutionStatus::initial().to_vec(),
    });
    (
        Resources {
            buffers,
            push,
            spec,
        },
        Layout {
            slots: mem.slots.len(),
            allocs: mem.allocs.len(),
        },
    )
}

/// Compile entry `entry` of `bc` for `p` under `inputs`, pass L3, check that the plan's size
/// estimate bounds the emitted words, and return the plan and the typed module the interpreter
/// runs.
fn checked(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
    what: &str,
) -> (Plan, TypedModule) {
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    let (plan, m) = typed_module_with(bc, entry, p, inputs).expect("module");
    let words = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= words,
        "{what}: estimate {} < emitted {words}",
        plan.estimate.words
    );
    (plan, m)
}

/// Run `m` in both scheduling orders: each order's output buffers.
fn run_orders(m: &TypedModule, res: &Resources, what: &str) -> Vec<Vec<Vec<u8>>> {
    [Order::Forward, Order::Reverse]
        .into_iter()
        .map(|order| {
            let cfg = Config {
                order,
                ..Config::default()
            };
            run(m, &cfg, res.clone())
                .unwrap_or_else(|e| panic!("{what} {order:?}: {e}"))
                .buffers
        })
        .collect()
}

/// The uses of the strategy named `name`, the memory it reaches included (`.psb`,
/// `.psb_and_view`), in either robustness encoding: `name` and `name.checked`.
fn uses(plan: &Plan, name: &str) -> u32 {
    plan.census()
        .iter()
        .filter(|(k, _)| k.strip_suffix(".checked").unwrap_or(k) == name)
        .map(|(_, (n, _))| *n)
        .sum()
}

/// The plan's memory intrinsics (`access.mem*`) and their uses, each side named by what it
/// reaches (`.to_psb` for a destination by device address, `.from_psb` for such a source).
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

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Every thread loads the pointer `b0[0]` (a guest address) and applies one atomic of each kind
/// to the words it points at; thread `t` also tries an atomic 4096·(t + 1) bytes past the
/// pointer (out of bounds) and one through the null pointer `b0[1]`. The old values are
/// stored to `b1`.
const WIDE_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %np = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  %n = load ptr addrspace(1), ptr addrspace(1) %np, align 8
  %o0 = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 1, i32 0, i32 2, i1 true)
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %o1 = call i32 @air.atomic.global.max.s.i32(ptr addrspace(1) %p1, i32 %t, i32 0, i32 2, i1 true)
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 2
  %t1 = add i32 %t, 1
  %cx = cmpxchg ptr addrspace(1) %p2, i32 0, i32 %t1 monotonic monotonic, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 3
  %f3 = call float @air.atomic.global.add.f32(ptr addrspace(1) %p3, float 5.000000e-01, i32 0, i32 2, i1 true)
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 4
  %o4 = atomicrmw nand ptr addrspace(1) %p4, i32 -1 monotonic, align 4
  %tf = uitofp i32 %t to float
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 5
  %f5 = call float @air.atomic.global.max.f32(ptr addrspace(1) %p5, float %tf, i32 0, i32 2, i1 true)
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 6
  store atomic i32 77, ptr addrspace(1) %p6 monotonic, align 4
  %p7 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 7
  %l7 = load atomic i32, ptr addrspace(1) %p7 acquire, align 4
  %tz = zext i32 %t to i64
  %tz1 = add i64 %tz, 1
  %far = mul i64 %tz1, 4096
  %pf = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %far
  %of = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %pf, i32 1, i32 0, i32 2, i1 true)
  %on = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %n, i32 1, i32 0, i32 2, i1 true)
  %old = extractvalue { i32, i1 } %cx, 0
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t
  store i32 %o0, ptr addrspace(1) %q0, align 4
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i1
  store i32 %old, ptr addrspace(1) %q1, align 4
  %i2 = add i32 %t, 128
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i2
  store i32 %of, ptr addrspace(1) %q2, align 4
  %i3 = add i32 %t, 192
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i3
  store i32 %on, ptr addrspace(1) %q3, align 4
  %i4 = add i32 %t, 256
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i4
  %b3 = bitcast float %f3 to i32
  store i32 %b3, ptr addrspace(1) %q4, align 4
  %i5 = add i32 %t, 320
  %q5 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i5
  store i32 %l7, ptr addrspace(1) %q5, align 4
";

const WIDE_DECLS: &str =
    "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.max.s.i32(ptr addrspace(1), i32, i32, i32, i1)
declare float @air.atomic.global.add.f32(ptr addrspace(1), float, i32, i32, i1)
declare float @air.atomic.global.max.f32(ptr addrspace(1), float, i32, i32, i1)
";

/// Atomics through a loaded pointer are exact under contention, contained out of bounds (old value
/// 0, the allocation untouched), and counted through root 0.
#[test]
fn atomics_through_a_loaded_pointer_are_exact_and_contained() {
    const N: u32 = 64;
    let Some(bc) = assemble(
        "wide_atomics",
        &kernel("wide_atomics", 2, WIDE_BODY, WIDE_DECLS),
    ) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "wide_atomics", &p, &LowerInputs::new(), &what);
        let native = p.fields().leaf.buffer_float32_atomic_add;
        // add, max, the far add, and the null add.
        assert_eq!(uses(&plan, "access.atomic.wide.rmw"), 4, "{what}");
        assert_eq!(uses(&plan, "access.atomic.wide.cmpxchg"), 1, "{what}");
        assert_eq!(uses(&plan, "access.atomic.wide.cas"), 1, "{what}");
        assert_eq!(
            uses(&plan, "access.atomic.wide.float_add"),
            u32::from(native),
            "{what}"
        );
        assert_eq!(
            uses(&plan, "access.atomic.wide.float_cas"),
            2 - u32::from(native),
            "{what}"
        );
        let mut target = vec![0u32; 16];
        target[1] = (-5i32) as u32;
        target[5] = f(-1.0);
        target[7] = 0xABCD;
        let mut ptrs = vec![0u8; 16];
        ptrs[..8].copy_from_slice(&alloc_guest(0).to_le_bytes());
        let mem = Memory {
            slots: vec![ptrs, vec![0; 4 * 6 * N as usize]],
            allocs: vec![le_words(&target)],
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[at.alloc(0)]);
            assert_eq!(w[0], N, "{what}: add");
            assert_eq!(w[1], N - 1, "{what}: max.s");
            let o = words(&out[at.slot(1)]);
            let winners: Vec<u32> = (0..N).filter(|t| o[64 + *t as usize] == 0).collect();
            assert_eq!(winners.len(), 1, "{what}: one cmpxchg winner");
            assert_eq!(w[2], winners[0] + 1, "{what}: cmpxchg");
            assert_eq!(w[3], f(0.5 * N as f32), "{what}: float add");
            assert_eq!(w[4], 0, "{what}: nand toggled an even number of times");
            assert_eq!(w[5], f((N - 1) as f32), "{what}: float max");
            assert_eq!(w[6], 77, "{what}: store");
            assert_eq!(&w[8..], &[0; 8], "{what}: bytes past the objects");
            let mut olds: Vec<u32> = o[..N as usize].to_vec();
            olds.sort_unstable();
            assert_eq!(olds, (0..N).collect::<Vec<_>>(), "{what}: add olds");
            let mut fl: Vec<u32> = o[256..256 + N as usize].to_vec();
            fl.sort_unstable_by(|a, b| f32::from_bits(*a).total_cmp(&f32::from_bits(*b)));
            let want: Vec<u32> = (0..N).map(|k| f(0.5 * k as f32)).collect();
            assert_eq!(fl, want, "{what}: float add olds");
            // The far atomics are out of bounds: 0, and the allocation holds only the N adds.
            assert!(o[128..192].iter().all(|x| *x == 0), "{what}: contained");
            assert!(o[192..256].iter().all(|x| *x == 0), "{what}: null");
            assert!(o[320..384].iter().all(|x| *x == 0xABCD), "{what}: load");
            let status = words(&out[at.status()]);
            assert_eq!(
                status[StatusWord::ResidentMisses.index() as usize],
                N,
                "{what}: one miss per null atomic"
            );
            assert!(
                words(&out[at.sink()]).iter().any(|x| *x != 0),
                "{what}: sink"
            );
        }
    }
}

/// Every thread applies `u64` atomics through the loaded pointer `b0[0]`.
const WIDE64_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %tz = zext i32 %t to i64
  %big = shl i64 %tz, 40
  %o0 = atomicrmw add ptr addrspace(1) %p, i64 %big monotonic, align 8
  %p1 = getelementptr inbounds i64, ptr addrspace(1) %p, i64 1
  %o1 = call i64 @air.atomic.global.max.u.i64(ptr addrspace(1) %p1, i64 %big, i32 0, i32 2, i1 true)
  %p2 = getelementptr inbounds i64, ptr addrspace(1) %p, i64 2
  %t1 = add i64 %tz, 1
  %cx = cmpxchg ptr addrspace(1) %p2, i64 0, i64 %t1 monotonic monotonic, align 8
  %old = extractvalue { i64, i1 } %cx, 0
  %q = getelementptr inbounds i64, ptr addrspace(1) %b1, i32 %t
  store i64 %old, ptr addrspace(1) %q, align 8
";

/// `u64` atomics through a loaded pointer are exact where the profile has
/// `shaderBufferInt64Atomics`, and refused naming it elsewhere.
#[test]
fn int64_atomics_through_a_loaded_pointer_follow_the_profile() {
    const N: u32 = 64;
    let decls = "declare i64 @air.atomic.global.max.u.i64(ptr addrspace(1), i64, i32, i32, i1)\n";
    let Some(bc) = assemble("wide64", &kernel("wide64", 2, WIDE64_BODY, decls)) else {
        return;
    };
    for (what, p) in profiles() {
        if !p.fields().leaf.buffer_int64_atomics {
            let e = compile(&bc, "wide64", &p).expect_err("refused");
            let crate::error::Error::Unsupported(u) = e else {
                panic!("{what}: {e:?}");
            };
            assert_eq!(
                (u.feature, u.needs),
                (
                    crate::error::Feature::Int64Atomics,
                    crate::error::Needs::AnyOf(vec![vec![crate::error::Capability::Feature(
                        VkFeature::ShaderBufferInt64Atomics
                    )]])
                ),
                "{what}"
            );
            continue;
        }
        let (plan, m) = checked(&bc, "wide64", &p, &LowerInputs::new(), &what);
        let mut ptrs = vec![0u8; 8];
        ptrs.copy_from_slice(&alloc_guest(0).to_le_bytes());
        let mem = Memory {
            slots: vec![ptrs, vec![0; 8 * N as usize]],
            allocs: vec![vec![0; 32]],
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            let w: Vec<u64> = out[at.alloc(0)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            let sum: u64 = (0..u64::from(N)).map(|t| t << 40).sum();
            assert_eq!(w[0], sum, "{what}: add");
            assert_eq!(w[1], u64::from(N - 1) << 40, "{what}: max");
            let olds: Vec<u64> = out[at.slot(1)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            let winners: Vec<usize> = (0..N as usize).filter(|t| olds[*t] == 0).collect();
            assert_eq!(winners.len(), 1, "{what}: one winner");
            assert_eq!(w[2], winners[0] as u64 + 1, "{what}: cmpxchg");
        }
    }
}

/// Every thread applies `i8` and `i16` atomics through the loaded pointer `b0[0]`: a byte add
/// on field `t & 3` of word 0, a half-word compare-exchange and signed max in word 1, and a byte
/// load of word 2; an `or` 4096·(t + 1) + 1 bytes past the pointer (out of bounds) and an add
/// through the null pointer `b0[1]`. The old values are stored to `b1`.
const NARROW_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %np = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  %n = load ptr addrspace(1), ptr addrspace(1) %np, align 8
  %k = and i32 %t, 3
  %kz = zext i32 %k to i64
  %pb = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %kz
  %o0 = atomicrmw add ptr addrspace(1) %pb, i8 1 monotonic, align 1
  %t16 = trunc i32 %t to i16
  %t1 = add i16 %t16, 1
  %ph = getelementptr inbounds i16, ptr addrspace(1) %p, i64 2
  %cx = cmpxchg ptr addrspace(1) %ph, i16 0, i16 %t1 monotonic monotonic, align 2
  %ts = sub i16 %t16, 100
  %pm = getelementptr inbounds i16, ptr addrspace(1) %p, i64 3
  %o2 = atomicrmw max ptr addrspace(1) %pm, i16 %ts monotonic, align 2
  %pl = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  %l = load atomic i8, ptr addrspace(1) %pl acquire, align 1
  %tz = zext i32 %t to i64
  %tz1 = add i64 %tz, 1
  %far0 = mul i64 %tz1, 4096
  %far = add i64 %far0, 1
  %pf = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %far
  %of = atomicrmw or ptr addrspace(1) %pf, i8 1 monotonic, align 1
  %on = atomicrmw add ptr addrspace(1) %n, i8 1 monotonic, align 1
  %old = extractvalue { i16, i1 } %cx, 0
  %w0 = zext i8 %o0 to i32
  %q0 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t
  store i32 %w0, ptr addrspace(1) %q0, align 4
  %w1 = zext i16 %old to i32
  %i1 = add i32 %t, 64
  %q1 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i1
  store i32 %w1, ptr addrspace(1) %q1, align 4
  %w2 = zext i8 %of to i32
  %i2 = add i32 %t, 128
  %q2 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i2
  store i32 %w2, ptr addrspace(1) %q2, align 4
  %w3 = zext i8 %on to i32
  %i3 = add i32 %t, 192
  %q3 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i3
  store i32 %w3, ptr addrspace(1) %q3, align 4
  %w4 = zext i8 %l to i32
  %i4 = add i32 %t, 256
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %i4
  store i32 %w4, ptr addrspace(1) %q4, align 4
";

/// Narrow atomics through a loaded pointer operate on the fields of their containing words at the
/// root entry's device address: exact under contention, contained out of bounds, and counted
/// through root 0.
#[test]
fn narrow_atomics_through_a_loaded_pointer_are_exact_and_contained() {
    const N: u32 = 64;
    let Some(bc) = assemble("wide_narrow", &kernel("wide_narrow", 2, NARROW_BODY, "")) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "wide_narrow", &p, &LowerInputs::new(), &what);
        // The load and the far or; the add, cmpxchg, max, and null add.
        assert_eq!(uses(&plan, "access.atomic.wide.narrow"), 2, "{what}");
        assert_eq!(uses(&plan, "access.atomic.wide.narrow_cas"), 4, "{what}");
        let mut target = vec![0u32; 4];
        target[1] = 0x8000_0000;
        target[2] = 0x0000_00AB;
        let mut ptrs = vec![0u8; 16];
        ptrs[..8].copy_from_slice(&alloc_guest(0).to_le_bytes());
        let mem = Memory {
            slots: vec![ptrs, vec![0; 4 * 5 * N as usize]],
            allocs: vec![le_words(&target)],
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[at.alloc(0)]);
            assert_eq!(w[0], 0x1010_1010, "{what}: byte adds");
            let o = words(&out[at.slot(1)]);
            let winners: Vec<u32> = (0..N).filter(|t| o[64 + *t as usize] == 0).collect();
            assert_eq!(winners.len(), 1, "{what}: one cmpxchg winner");
            // max(-32768, t - 100) over t < 64 is -37 (0xFFDB).
            assert_eq!(
                w[1],
                0xFFDB_0000 | (winners[0] + 1),
                "{what}: cmpxchg and max"
            );
            assert_eq!(
                &w[2..],
                &[0xAB, 0],
                "{what}: the load and the bytes past the fields"
            );
            for k in 0..4 {
                let mut olds: Vec<u32> = (0..N as usize)
                    .filter(|t| t % 4 == k)
                    .map(|t| o[t])
                    .collect();
                olds.sort_unstable();
                assert_eq!(olds, (0..N / 4).collect::<Vec<_>>(), "{what}: field {k}");
            }
            assert!(o[128..192].iter().all(|x| *x == 0), "{what}: contained");
            assert!(o[192..256].iter().all(|x| *x == 0), "{what}: null");
            assert!(o[256..320].iter().all(|x| *x == 0xAB), "{what}: load");
            let status = words(&out[at.status()]);
            assert_eq!(
                status[StatusWord::ResidentMisses.index() as usize],
                N,
                "{what}: one miss per null atomic"
            );
        }
    }
}

/// Every thread stores `3t` coherently to word `t` of the loaded pointer `b0[0]`'s allocation,
/// meets the others at a device barrier, and loads its neighbour's word coherently into `b1`.
const COHERENT_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %pt = getelementptr inbounds i32, ptr addrspace(1) %p, i32 %t
  %t3 = mul i32 %t, 3
  call void @air.store.device_coherent.i32.p1i32(i32 %t3, ptr addrspace(1) %pt)
  call void @air.wg.barrier(i32 1, i32 1)
  %n = add i32 %t, 1
  %m = and i32 %n, 63
  %pm = getelementptr inbounds i32, ptr addrspace(1) %p, i32 %m
  %v = call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %pm)
  %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t
  store i32 %v, ptr addrspace(1) %q, align 4
";

const COHERENT_DECLS: &str = "declare void @air.wg.barrier(i32, i32)
declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))
declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))
";

/// Coherent accesses through a loaded pointer carry `MakePointerAvailable` (stores) and
/// `MakePointerVisible` (loads) under the Vulkan memory model where the profile has it, and the
/// neighbour exchange is exact; elsewhere they refuse naming `vulkanMemoryModel`.
#[test]
fn coherent_accesses_through_a_loaded_pointer_are_exact() {
    use crate::error::{Capability, Error, Feature, Unsupported};
    use crate::spirv::ir::Inst;
    const N: usize = 64;
    let Some(bc) = assemble(
        "wide_coherent",
        &kernel("wide_coherent", 2, COHERENT_BODY, COHERENT_DECLS),
    ) else {
        return;
    };
    for (what, p) in profiles() {
        if !p.fields().vulkan_memory_model {
            let e = compile(&bc, "wide_coherent", &p).expect_err("refused");
            let want = Unsupported::needs(
                Feature::CoherentMemory,
                Capability::Feature(VkFeature::VulkanMemoryModel),
            );
            let Error::Unsupported(u) = e else {
                panic!("{what}: {e:?}");
            };
            assert_eq!((u.feature, u.needs), (want.feature, want.needs), "{what}");
            continue;
        }
        let (plan, m) = checked(&bc, "wide_coherent", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.store"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.load"), 1, "{what}");
        let (mut available, mut visible) = (0, 0);
        for b in m.blocks() {
            for n in &b.insts {
                match &n.inst {
                    Inst::Store { access, .. } if access.make_available.is_some() => {
                        assert!(access.non_private, "{what}");
                        available += 1;
                    }
                    Inst::Load { access, .. } if access.make_visible.is_some() => {
                        assert!(access.non_private && access.volatile, "{what}");
                        visible += 1;
                    }
                    _ => {}
                }
            }
        }
        assert!(available >= 1 && visible >= 1, "{what}");
        let mut ptrs = vec![0u8; 8];
        ptrs.copy_from_slice(&alloc_guest(0).to_le_bytes());
        let mem = Memory {
            slots: vec![ptrs, vec![0; 4 * N]],
            allocs: vec![vec![0; 4 * N]],
        };
        let (res, at) = resources(&plan, &mem, N as u32);
        for out in run_orders(&m, &res, &what) {
            let w = words(&out[at.alloc(0)]);
            let o = words(&out[at.slot(1)]);
            for t in 0..N {
                assert_eq!(w[t], 3 * t as u32, "{what}: coherent store {t}");
                assert_eq!(o[t], 3 * ((t + 1) % N) as u32, "{what}: coherent load {t}");
            }
        }
    }
}

/// The public fixture `kernel_device_address_field_atomic`, without its `llvm.lifetime` markers:
/// both buffer pointers are parked in a private struct, and the (inlined) helper reads one back
/// and runs an integer compare-exchange on a float slot through it, returning `expected`.
const FIELD_ATOMIC: &str = "%struct.cache = type { ptr addrspace(1), ptr addrspace(1) }

define void @field_atomic(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %2, i32 noundef %3) {
  %5 = alloca %struct.cache, align 8
  %7 = getelementptr inbounds %struct.cache, ptr %5, i64 0, i32 0
  store ptr addrspace(1) %0, ptr %7, align 8
  %8 = getelementptr inbounds %struct.cache, ptr %5, i64 0, i32 1
  store ptr addrspace(1) %1, ptr %8, align 8
  %9 = call fastcc i32 @claim(ptr noundef nonnull align 8 dereferenceable(16) %5, i32 noundef %3)
  %10 = zext i32 %3 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %10
  store i32 %9, ptr addrspace(1) %11, align 4
  ret void
}

define internal fastcc i32 @claim(ptr noundef nonnull readonly align 8 dereferenceable(16) %0, i32 noundef %1) {
  %3 = alloca i32, align 4
  %4 = getelementptr inbounds %struct.cache, ptr %0, i64 0, i32 0
  %5 = load ptr addrspace(1), ptr %4, align 8
  %6 = zext i32 %1 to i64
  %7 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %6
  %8 = add i32 %1, 100
  store i32 0, ptr %3, align 4
  %11 = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %7, ptr nonnull %3, i32 %8, i32 0, i32 0, i32 2, i1 true)
  %12 = load i32, ptr %3, align 4
  ret i32 %12
}

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)

!air.kernel = !{!9}
!air.version = !{!20}
!air.language_version = !{!21}
!9 = !{ptr @field_atomic, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"slots\"}
!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"aux\"}
!14 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}
!15 = !{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!\"Metal\", i32 4, i32 0, i32 0}
";

/// Run `field_atomic` (or its open-cell variant) compiled as `plan`/`m` over 32 threads: the
/// compare-exchange through the pointer read back succeeds on the even threads' zero slots
/// (writing `i + 100`) and fails on the odd threads' 3.5 slots, `expected` holding the value
/// observed either way (the fixture's documented result), and no resident-set search misses.
fn check_field_atomic(plan: &Plan, m: &TypedModule, what: &str) {
    const N: u32 = 32;
    let slots: Vec<u32> = (0..N)
        .map(|i| if i % 2 == 0 { 0 } else { f(3.5) })
        .collect();
    let mem = Memory {
        slots: vec![le_words(&slots), vec![0; 16], vec![0; 4 * N as usize]],
        allocs: Vec::new(),
    };
    let (res, at) = resources(plan, &mem, N);
    for out in run_orders(m, &res, what) {
        let s = words(&out[at.slot(0)]);
        let o = words(&out[at.slot(2)]);
        for i in 0..N as usize {
            if i % 2 == 0 {
                assert_eq!((s[i], o[i]), (i as u32 + 100, 0), "{what}: thread {i}");
            } else {
                assert_eq!((s[i], o[i]), (f(3.5), f(3.5)), "{what}: thread {i}");
            }
        }
        assert_eq!(words(&out[at.slot(1)]), vec![0; 4], "{what}: aux untouched");
        let status = words(&out[at.status()]);
        assert_eq!(status[StatusWord::ResidentMisses.index() as usize], 0);
    }
}

/// A pointer parked in a private struct whose every access is visible is a forwarding cell: the
/// struct holds its pointer as its fat pointer, and the one read back is exactly the pointer stored
/// at its field (`slots`). The other field, which nothing reads back, is not stored at all.
#[test]
fn a_pointer_parked_in_a_private_struct_is_exact() {
    let Some(bc) = assemble("field_atomic", FIELD_ATOMIC) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "field_atomic", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.cell.store"), 1, "{what}");
        assert_eq!(uses(&plan, "access.cell.load"), 1, "{what}");
        assert_eq!(uses(&plan, "access.atomic.buffer.cmpxchg"), 1, "{what}");
        check_field_atomic(&plan, &m, &what);
    }
}

/// `FIELD_ATOMIC` with the struct copied whole by `memcpy` into `%copy` before the helper reads
/// the copy, plus `extra` after the copy.
fn field_atomic_copied(extra: &str) -> String {
    let text = FIELD_ATOMIC
        .replace(
            "  %5 = alloca %struct.cache, align 8\n",
            "  %5 = alloca %struct.cache, align 8\n  %copy = alloca %struct.cache, align 8\n",
        )
        .replace(
            "  %9 = call fastcc i32 @claim(ptr noundef nonnull align 8 dereferenceable(16) %5,",
            &format!(
                "  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %copy, ptr align 8 %5, i64 16, i1 false)\n{extra}  \
                 %9 = call fastcc i32 @claim(ptr noundef nonnull align 8 dereferenceable(16) %copy,"
            ),
        )
        .replace(
            "declare i32 @air.atomic.global.cmpxchg.weak.i32(",
            "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\ndeclare i32 @air.atomic.global.cmpxchg.weak.i32(",
        );
    assert!(text.contains("%copy,") && text.contains("@llvm.memcpy"));
    text
}

/// The same struct copied whole by `memcpy` before the helper reads it: the copy moves both pointer
/// slots, so both allocas stay forwarding cells, and the pointer read back from the copy is exactly
/// the one stored into the original, a static fat pointer.
#[test]
fn a_pointer_copied_through_a_private_struct_is_exact() {
    let Some(bc) = assemble("field_atomic", &field_atomic_copied("")) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "field_atomic", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.cell.store"), 2, "{what}");
        assert_eq!(uses(&plan, "access.cell.load"), 1, "{what}");
        assert_eq!(uses(&plan, "access.atomic.buffer.cmpxchg"), 1, "{what}");
        check_field_atomic(&plan, &m, &what);
    }
}

/// The copy's second pointer slot is also read as data, so the copy is an open cell, and the copy
/// into it is a data access over the original's slots: both allocas are open cells. Each pointer is
/// stored as its guest address, the one read back from the copy resolves through the resident set
/// to its allocation, and the compare-exchange through that wide pointer is exact. The data read is
/// observed (a branch tests it against an address no allocation has, so no thread takes it), or it
/// would go with what nothing observes.
#[test]
fn a_pointer_copied_through_an_open_private_struct_is_its_guest_address() {
    let peek = "  %peek.at = getelementptr inbounds %struct.cache, ptr %copy, i64 0, i32 1\n  \
                %peek = load i64, ptr %peek.at, align 8\n  %peek.ones = icmp eq i64 %peek, -1\n  \
                br i1 %peek.ones, label %peek.never, label %peek.go\npeek.never:\n  store i32 1, \
                ptr addrspace(1) %1, align 4\n  br label %peek.go\npeek.go:\n";
    let Some(bc) = assemble("field_atomic", &field_atomic_copied(peek)) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "field_atomic", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.cell.store"), 0, "{what}");
        assert_eq!(uses(&plan, "access.cell.load"), 0, "{what}");
        assert_eq!(
            uses(&plan, "access.dynamic.store_pointer.arena"),
            2,
            "{what}"
        );
        assert_eq!(
            uses(&plan, "access.dynamic.load_pointer.arena"),
            1,
            "{what}"
        );
        assert_eq!(uses(&plan, "access.atomic.wide.cmpxchg"), 1, "{what}");
        check_field_atomic(&plan, &m, &what);
    }
}

/// `FIELD_ATOMIC` with its struct zero-filled by a `memset` of `len` bytes before its fields are
/// stored, as a zero-initialized struct is.
fn field_atomic_zeroed(len: u32) -> String {
    let text = FIELD_ATOMIC
        .replace(
            "  %5 = alloca %struct.cache, align 8\n",
            &format!(
                "  %5 = alloca %struct.cache, align 8\n  \
                 call void @llvm.memset.p0.i64(ptr align 8 %5, i8 0, i64 {len}, i1 false)\n"
            ),
        )
        .replace(
            "declare i32 @air.atomic.global.cmpxchg.weak.i32(",
            "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)\ndeclare i32 @air.atomic.global.cmpxchg.weak.i32(",
        );
    assert!(text.contains("@llvm.memset.p0.i64(ptr align 8 %5"));
    text
}

/// A zero fill of the whole struct before its pointers are stored writes null over both slots, not
/// data: the struct stays a forwarding cell, the pointer read back is the one stored or null, and
/// the compare-exchange through it is exact.
#[test]
fn a_pointer_parked_in_a_zero_filled_private_struct_is_exact() {
    let Some(bc) = assemble("field_atomic", &field_atomic_zeroed(16)) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "field_atomic", &p, &LowerInputs::new(), &what);
        let census = plan.census();
        assert_eq!(uses(&plan, "access.cell.load"), 1, "{what}: {census:?}");
        assert_eq!(
            uses(&plan, "access.dynamic.load_pointer.arena"),
            0,
            "{what}: {census:?}"
        );
        check_field_atomic(&plan, &m, &what);
    }
}

/// A zero fill over part of a pointer slot (4 bytes, the first slot's, which the helper reads back)
/// is a data access of a pointer's bytes: the struct is an open cell, the pointer read back its
/// guest address, and the result the same.
#[test]
fn a_zero_fill_over_part_of_a_pointer_slot_opens_the_cell() {
    let Some(bc) = assemble("field_atomic", &field_atomic_zeroed(4)) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "field_atomic", &p, &LowerInputs::new(), &what);
        let census = plan.census();
        assert_eq!(uses(&plan, "access.cell.load"), 0, "{what}: {census:?}");
        assert_eq!(
            uses(&plan, "access.dynamic.load_pointer.arena"),
            1,
            "{what}: {census:?}"
        );
        check_field_atomic(&plan, &m, &what);
    }
}

/// A private array `s` of two buffer pointers zero-filled by `memset`, the even threads storing
/// `b0` at `s[0]`, then copied whole into `c`: `c[0]` reads `b0` or the fill's null, carried by
/// the copy, and `s[1]`, never stored, the fill's null. `b1[t]` receives bit 0 for a null `c[0]`
/// and bit 1 for a null `s[1]`.
const ZERO_FILLED: &str = "  %s = alloca [2 x ptr addrspace(1)], align 8
  %c = alloca [2 x ptr addrspace(1)], align 8
  call void @llvm.memset.p0.i64(ptr align 8 %s, i8 0, i64 16, i1 false)
  %odd = and i32 %t, 1
  %even = icmp eq i32 %odd, 0
  br i1 %even, label %put, label %go
put:
  store ptr addrspace(1) %b0, ptr %s, align 8
  br label %go
go:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %c, ptr align 8 %s, i64 16, i1 false)
  %p = load ptr addrspace(1), ptr %c, align 8
  %s1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %s, i64 0, i64 1
  %q = load ptr addrspace(1), ptr %s1, align 8
  %np = icmp eq ptr addrspace(1) %p, null
  %nq = icmp eq ptr addrspace(1) %q, null
  %xp = zext i1 %np to i32
  %xq = zext i1 %nq to i32
  %hq = shl i32 %xq, 1
  %x = or i32 %xp, %hq
  %tz = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %tz
  store i32 %x, ptr addrspace(1) %o, align 4
";

/// [`ZERO_FILLED`]'s cells stay forwarding cells, and each load reads the fill's null where no
/// store wrote: the odd threads' `c[0]` and every thread's `s[1]`.
#[test]
fn a_zero_filled_pointer_slot_reads_null() {
    const N: usize = 32;
    let decls = "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)\n\
                 declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n";
    let Some(bc) = assemble("zero_filled", &kernel("zero_filled", 2, ZERO_FILLED, decls)) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "zero_filled", &p, &LowerInputs::new(), &what);
        let census = plan.census();
        assert_eq!(
            uses(&plan, "access.dynamic.load_pointer.arena"),
            0,
            "{what}: {census:?}"
        );
        let mem = Memory {
            slots: vec![vec![0; 4 * N], vec![0; 4 * N]],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, N as u32);
        for out in run_orders(&m, &res, &what) {
            let o = words(&out[at.slot(1)]);
            for (t, &x) in o.iter().enumerate() {
                assert_eq!(x, if t % 2 == 0 { 2 } else { 3 }, "{what}: thread {t}");
            }
        }
    }
}

/// Pointer round trips through buffers and threadgroup memory: `b1[t]` receives the guest
/// address of `&b0[t]` (`ptrtoint`) and `b2[2t]` the pointer itself; thread 0 parks `&b0[1]` in a
/// threadgroup cell, and after a barrier every thread adds 1 through the pointer it reads back
/// from the cell and 2 through the pointer it reads back from `b2`. A null pointer stored and
/// read back is root 0: its load is contained and counted.
const ROUND_TRIP: &str = "@cell = internal addrspace(3) global ptr addrspace(1) undef, align 8

define void @round_trip(ptr addrspace(1) noundef \"air-buffer-no-alias\" %b0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %b1, ptr addrspace(1) noundef \"air-buffer-no-alias\" %b2, ptr addrspace(1) noundef \"air-buffer-no-alias\" %b3, i32 %t) {
entry:
  %tz = zext i32 %t to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %tz
  %g = ptrtoint ptr addrspace(1) %p to i64
  %q1 = getelementptr inbounds i64, ptr addrspace(1) %b1, i64 %tz
  store i64 %g, ptr addrspace(1) %q1, align 8
  %t2 = shl i64 %tz, 1
  %q2 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %b2, i64 %t2
  store ptr addrspace(1) %p, ptr addrspace(1) %q2, align 8
  %q3 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %q2, i64 1
  store ptr addrspace(1) null, ptr addrspace(1) %q3, align 8
  %first = icmp eq i32 %t, 0
  br i1 %first, label %park, label %go

park:
  %one = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 1
  store ptr addrspace(1) %one, ptr addrspace(3) @cell, align 8
  br label %go

go:
  call void @air.wg.barrier(i32 2, i32 1)
  %c = load ptr addrspace(1), ptr addrspace(3) @cell, align 8
  %o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %c, i32 1, i32 0, i32 2, i1 true)
  %back = load ptr addrspace(1), ptr addrspace(1) %q2, align 8
  %o2 = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %back, i32 2, i32 0, i32 2, i1 true)
  %nul = load ptr addrspace(1), ptr addrspace(1) %q3, align 8
  %v = load i32, ptr addrspace(1) %nul, align 4
  %q4 = getelementptr inbounds i32, ptr addrspace(1) %b3, i64 %tz
  store i32 %v, ptr addrspace(1) %q4, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!air.version = !{!90}
!air.language_version = !{!91}
!0 = !{ptr @round_trip, !1, !2}
!1 = !{}
!2 = !{!10, !11, !12, !13, !14}
!10 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b0\"}
!11 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"b1\"}
!12 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"b2\"}
!13 = !{i32 3, !\"air.buffer\", !\"air.location_index\", i32 3, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b3\"}
!14 = !{i32 4, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}
!90 = !{i32 2, i32 8, i32 0}
!91 = !{!\"Metal\", i32 4, i32 0, i32 0}
";

/// Stored and converted pointers are guest addresses (`guest + bind_offset + offset` of the root
/// entry), and a pointer read back from a buffer or a threadgroup cell reaches its allocation
/// through the resident set. With every pointer store and load volatile, the guest addresses' words
/// are volatile accesses of the slot and the threadgroup arena, and nothing else is.
#[test]
fn stored_pointers_are_guest_addresses_that_resolve_back() {
    let volatile = ROUND_TRIP
        .replace(
            "  store ptr addrspace(1) ",
            "  store volatile ptr addrspace(1) ",
        )
        .replace(
            " = load ptr addrspace(1), ",
            " = load volatile ptr addrspace(1), ",
        );
    assert_eq!(volatile.matches(" volatile ptr").count(), 6);
    for (form, text, want_volatile) in [
        ("plain", ROUND_TRIP, (vec![], vec![])),
        (
            "volatile",
            volatile.as_str(),
            (vec![(WG, 2), (SB, 4)], vec![(WG, 2), (SB, 4)]),
        ),
    ] {
        let Some(bc) = assemble("round_trip", text) else {
            return;
        };
        check_round_trip(&bc, form, &want_volatile);
    }
}

fn check_round_trip(bc: &[u8], form: &str, want_volatile: &(ByClass, ByClass)) {
    const N: u32 = 32;
    for (what, p) in profiles() {
        let what = format!("{what} {form}");
        let (plan, m) = checked(bc, "round_trip", &p, &LowerInputs::new(), &what);
        assert_eq!(&volatiles(&m), want_volatile, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.ptrtoint"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.store_pointer"), 2, "{what}");
        assert_eq!(
            uses(&plan, "access.dynamic.store_pointer.arena"),
            1,
            "{what}"
        );
        assert_eq!(
            uses(&plan, "access.dynamic.load_pointer.arena"),
            1,
            "{what}"
        );
        let mem = Memory {
            slots: vec![
                vec![0; 4 * N as usize],
                vec![0; 8 * N as usize],
                vec![0; 16 * N as usize],
                vec![0xEE; 4 * N as usize],
            ],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            let g: Vec<u64> = out[at.slot(1)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            let want: Vec<u64> = (0..u64::from(N)).map(|t| slot_guest(0) + 4 * t).collect();
            assert_eq!(g, want, "{what}: ptrtoint");
            let stored: Vec<u64> = out[at.slot(2)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            for t in 0..N as usize {
                assert_eq!(stored[2 * t], want[t], "{what}: stored pointer {t}");
                assert_eq!(stored[2 * t + 1], 0, "{what}: stored null {t}");
            }
            let w = words(&out[at.slot(0)]);
            for (t, x) in w.iter().enumerate() {
                let cell = if t == 1 { N } else { 0 };
                assert_eq!(*x, 2 + cell, "{what}: b0[{t}]");
            }
            assert_eq!(words(&out[at.slot(3)]), vec![0; N as usize], "{what}: null");
            let status = words(&out[at.status()]);
            assert_eq!(
                status[StatusWord::ResidentMisses.index() as usize],
                N,
                "{what}: one miss per null load"
            );
        }
    }
}

/// A pointer stored into and loaded back from a slot reached by device address is named by that
/// reach. A `u64` atomic on the counter `b1[0]` lowers `b1` by device address, so thread `t`'s
/// store of `&b0[t]` into `b1[1 + t]` and its load back are `access.dynamic.store_pointer.psb` and
/// `access.dynamic.load_pointer.psb`, never the descriptor view's names; the pointer read back
/// reaches `b0[t]`, which the thread increments, and the cell holds `b0[t]`'s guest address. A
/// profile without `u64` buffer atomics refuses naming them.
#[test]
fn pointers_through_a_slot_reached_by_device_address_are_named_psb() {
    const N: u32 = 32;
    let body = "  %tz = zext i32 %t to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %tz
  %i = add i64 %tz, 1
  %cell = getelementptr inbounds i64, ptr addrspace(1) %b1, i64 %i
  store ptr addrspace(1) %p, ptr addrspace(1) %cell, align 8
  %back = load ptr addrspace(1), ptr addrspace(1) %cell, align 8
  %v = load i32, ptr addrspace(1) %back, align 4
  %v1 = add i32 %v, 1
  store i32 %v1, ptr addrspace(1) %back, align 4
  %n = atomicrmw add ptr addrspace(1) %b1, i64 1 monotonic, align 8
";
    let Some(bc) = assemble("psb_cells", &kernel("psb_cells", 2, body, "")) else {
        return;
    };
    for (what, p) in profiles() {
        if !p.fields().leaf.buffer_int64_atomics {
            let e = compile(&bc, "psb_cells", &p).expect_err("refused");
            let crate::error::Error::Unsupported(u) = e else {
                panic!("{what}: {e:?}");
            };
            assert_eq!(u.feature, crate::error::Feature::Int64Atomics, "{what}");
            continue;
        }
        let (plan, m) = checked(&bc, "psb_cells", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.store_pointer.psb"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.load_pointer.psb"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.store_pointer"), 0, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.load_pointer"), 0, "{what}");
        let b1 = plan
            .slots
            .iter()
            .flatten()
            .find(|c| c.slot.index == 1)
            .expect("planned slot");
        assert_eq!(b1.form, BufferForm::Psb, "{what}");
        let mem = Memory {
            slots: vec![le_words(&[40; N as usize]), vec![0; 8 * (N as usize + 1)]],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            assert_eq!(words(&out[at.slot(0)]), vec![41; N as usize], "{what}: b0");
            let cells: Vec<u64> = out[at.slot(1)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            assert_eq!(cells[0], u64::from(N), "{what}: counter");
            for t in 0..N as usize {
                assert_eq!(
                    cells[1 + t],
                    slot_guest(0) + 4 * t as u64,
                    "{what}: cell {t}"
                );
            }
            let status = words(&out[at.status()]);
            assert_eq!(
                status[StatusWord::ResidentMisses.index() as usize],
                0,
                "{what}: misses"
            );
        }
    }
}

/// Counts by storage class.
type ByClass = Vec<(StorageClass, u32)>;

const SB: StorageClass = StorageClass::StorageBuffer;
const WG: StorageClass = StorageClass::Workgroup;

/// The volatile loads and stores of `m` by the storage class of their pointer.
fn volatiles(m: &TypedModule) -> (ByClass, ByClass) {
    let class = |ptr| match m.value(ptr).and_then(|d| m.types().get(d.ty)) {
        Ok(Type::Pointer { class, .. }) => *class,
        other => panic!("an access through {other:?}"),
    };
    let (mut loads, mut stores) = (BTreeMap::new(), BTreeMap::new());
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        match &n.inst {
            Inst::Load { ptr, access } if access.volatile => {
                *loads.entry(class(*ptr)).or_insert(0) += 1
            }
            Inst::Store { ptr, access, .. } if access.volatile => {
                *stores.entry(class(*ptr)).or_insert(0) += 1
            }
            _ => {}
        }
    }
    (loads.into_iter().collect(), stores.into_iter().collect())
}

/// A slot bound at byte 48 of a larger allocation (its root entry's `bind_offset`): the Metal
/// address its pointer converts to and is stored as includes the bind offset (`guest +
/// bind_offset + offset`), an access through the slot reaches `host + bind_offset + offset`, and
/// the stored pointer read back resolves through the resident set to the same bytes, where an
/// atomic and a load act.
#[test]
fn a_slot_bound_inside_an_allocation_stores_its_metal_address() {
    let body = "  %p = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 1
  %g = ptrtoint ptr addrspace(1) %p to i64
  store i64 %g, ptr addrspace(1) %b1, align 8
  %q = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 8
  store ptr addrspace(1) %p, ptr addrspace(1) %q, align 8
  store i32 7, ptr addrspace(1) %b0, align 4
  %back = load ptr addrspace(1), ptr addrspace(1) %q, align 8
  %o = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %back, i32 5, i32 0, i32 2, i1 true)
  %v = load i32, ptr addrspace(1) %back, align 4
  %r = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 16
  store i32 %v, ptr addrspace(1) %r, align 4
";
    let decls = "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)\n";
    let Some(bc) = assemble("bound", &kernel("bound", 2, body, decls)) else {
        return;
    };
    let binds = [Bind {
        slot: 0,
        alloc: 0,
        offset: 48,
    }];
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "bound", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.ptrtoint"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.store_pointer"), 1, "{what}");
        let b0 = plan
            .slots
            .iter()
            .flatten()
            .find(|c| c.slot.index == 0)
            .expect("planned slot");
        assert_eq!(b0.form, BufferForm::Psb, "{what}");
        let mem = Memory {
            slots: vec![vec![0; 16], vec![0; 24]],
            allocs: vec![vec![0x11; 128]],
        };
        let (res, at) = resources_bound(&plan, &mem, 1, &binds);
        let g = alloc_guest(0) + 48 + 4;
        let mut alloc = vec![0x11u8; 128];
        alloc[48..52].copy_from_slice(&7u32.to_le_bytes());
        alloc[52..56].copy_from_slice(&0x1111_1116u32.to_le_bytes());
        let mut b1 = Vec::new();
        b1.extend_from_slice(&g.to_le_bytes());
        b1.extend_from_slice(&g.to_le_bytes());
        b1.extend_from_slice(&le_words(&[0x1111_1116, 0]));
        for out in run_orders(&m, &res, &what) {
            assert_eq!(out[at.alloc(0)], alloc, "{what}: allocation");
            assert_eq!(out[at.slot(1)], b1, "{what}: b1");
            assert_eq!(out[at.slot(0)], vec![0; 16], "{what}: b0's own buffer");
            let status = words(&out[at.status()]);
            assert_eq!(
                status[StatusWord::ResidentMisses.index() as usize],
                0,
                "{what}"
            );
        }
    }
}

/// `ptrtoint` to a narrower integer is the guest address truncated to that width, held as the value
/// families hold it: an `i1` is a `bool` (the address's low bit) that `zext` and `select` take, and
/// `i8`..`i32` drop the address's high word.
#[test]
fn ptrtoint_truncates_the_guest_address_to_each_width() {
    let body = "  %p4 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 4
  %p5 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 5
  %a = ptrtoint ptr addrspace(1) %p4 to i1
  %b = ptrtoint ptr addrspace(1) %p5 to i1
  %c = ptrtoint ptr addrspace(1) %p5 to i8
  %d = ptrtoint ptr addrspace(1) %p5 to i16
  %e = ptrtoint ptr addrspace(1) %p5 to i32
  %f = ptrtoint ptr addrspace(1) %p5 to i64
  %az = zext i1 %a to i32
  store i32 %az, ptr addrspace(1) %b1, align 4
  %bz = zext i1 %b to i32
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 1
  store i32 %bz, ptr addrspace(1) %o1, align 4
  %s = select i1 %b, i32 7, i32 9
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 2
  store i32 %s, ptr addrspace(1) %o2, align 4
  %cz = zext i8 %c to i32
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 3
  store i32 %cz, ptr addrspace(1) %o3, align 4
  %dz = zext i16 %d to i32
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 4
  store i32 %dz, ptr addrspace(1) %o4, align 4
  %o5 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 5
  store i32 %e, ptr addrspace(1) %o5, align 4
  %o6 = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 6
  store i64 %f, ptr addrspace(1) %o6, align 8
";
    let Some(bc) = assemble("to_narrow", &kernel("to_narrow", 2, body, "")) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "to_narrow", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.ptrtoint"), 6, "{what}");
        let mem = Memory {
            slots: vec![vec![0; 16], vec![0xEE; 32]],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, 1);
        let g = slot_guest(0) + 5;
        let mut want = le_words(&[0, 1, 7, 5, 5, g as u32]);
        want.extend_from_slice(&g.to_le_bytes());
        for out in run_orders(&m, &res, &what) {
            assert_eq!(out[at.slot(1)], want, "{what}");
        }
    }
}

/// One thread moves bytes through the loaded pointers `p = b0[0]` (the allocation `a0`), `q =
/// b0[1]` (8 bytes into `b1`, so it aliases the static `b1`), and the null `b0[2]`, with the
/// `u32` lengths `b2[0..3]`: a constant word copy into `p`, a byte fill, a dynamic copy out of
/// `b1`, three `memmove`s between `b1` and `q` over their overlap (backward, forward from a
/// static source above the destination, backward into a static destination above the source:
/// the last two copy in the wrong direction unless the static side's Metal address includes its
/// `SlotGuestBase`), a dynamic copy through null, and a copy running past `a0`'s end.
const MEM_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %qp = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  %q = load ptr addrspace(1), ptr addrspace(1) %qp, align 8
  %np = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 16
  %n = load ptr addrspace(1), ptr addrspace(1) %np, align 8
  %l0 = load i32, ptr addrspace(1) %b2, align 4
  %l1p = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 1
  %l1 = load i32, ptr addrspace(1) %l1p, align 4
  %l2p = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 2
  %l2 = load i32, ptr addrspace(1) %l2p, align 4
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %p, ptr addrspace(1) align 4 %b1, i64 16, i1 false)
  %p16 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 17
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %p16, i8 90, i64 7, i1 false)
  %p32 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 32
  %s64 = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 64
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) align 1 %p32, ptr addrspace(1) align 1 %s64, i32 %l0, i1 false)
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %q, ptr addrspace(1) align 4 %b1, i32 %l1, i1 false)
  %s16 = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 16
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %q, ptr addrspace(1) align 4 %s16, i32 %l1, i1 false)
  %d120 = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 120
  %q104 = getelementptr inbounds i8, ptr addrspace(1) %q, i64 104
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %d120, ptr addrspace(1) align 4 %q104, i32 %l1, i1 false)
  call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) align 1 %n, ptr addrspace(1) align 1 %b1, i32 %l2, i1 false)
  %p60 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 60
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %p60, ptr addrspace(1) align 4 %b1, i64 16, i1 false)
";

const MEM_DECLS: &str =
    "declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memcpy.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)
declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
";

/// Memory intrinsics through loaded pointers are exact: each unit is contained against the root
/// entry (the bytes past `a0`'s end are dropped), a `memmove` between a static buffer and a wide
/// pointer into it compares their Metal addresses and copies in the direction the overlap needs,
/// and a zero-length copy through null touches nothing and counts no miss.
#[test]
fn memory_intrinsics_through_loaded_pointers_are_exact() {
    let Some(bc) = assemble("wide_mem", &kernel("wide_mem", 3, MEM_BODY, MEM_DECLS)) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "wide_mem", &p, &LowerInputs::new(), &what);
        // Copies into a wide pointer reach a device address at their destination, the copy from
        // `q + 104` at its source; `b1` stays a descriptor view.
        assert_eq!(
            intrinsics(&plan),
            named(&[
                ("access.memcpy.dynamic.to_psb", 2),
                ("access.memcpy.to_psb", 2),
                ("access.memmove.dynamic.from_psb", 1),
                ("access.memmove.dynamic.to_psb", 2),
                ("access.memset.to_psb", 1),
            ]),
            "{what}"
        );
        let mut ptrs = vec![0u8; 24];
        ptrs[..8].copy_from_slice(&alloc_guest(0).to_le_bytes());
        ptrs[8..16].copy_from_slice(&(slot_guest(1) + 8).to_le_bytes());
        let data: Vec<u8> = (0..256u32).map(|i| (i * 7 + 3) as u8).collect();
        for (l0, l1, l2) in [(0u32, 0u32, 0u32), (5, 40, 0), (29, 64, 0)] {
            let lens = le_words(&[l0, l1, l2]);
            let mem = Memory {
                slots: vec![ptrs.clone(), data.clone(), lens],
                allocs: vec![vec![0x11; 64]],
            };
            let (res, at) = resources(&plan, &mem, 1);
            for out in run_orders(&m, &res, &what) {
                let case = format!("{what} {l0} {l1} {l2}");
                let mut a0 = vec![0x11u8; 64];
                a0[..16].copy_from_slice(&data[..16]);
                a0[17..24].fill(90);
                a0[32..32 + l0 as usize].copy_from_slice(&data[64..64 + l0 as usize]);
                a0[60..64].copy_from_slice(&data[..4]);
                assert_eq!(out[at.alloc(0)], a0, "{case}: a0");
                let mut b1 = data.clone();
                b1.copy_within(0..l1 as usize, 8);
                b1.copy_within(16..16 + l1 as usize, 8);
                b1.copy_within(112..112 + l1 as usize, 120);
                assert_eq!(out[at.slot(1)], b1, "{case}: b1");
                let status = words(&out[at.status()]);
                assert_eq!(
                    status[StatusWord::ResidentMisses.index() as usize],
                    0,
                    "{case}: no miss"
                );
            }
        }
    }
}

/// One thread moves bytes between the loaded pointers `p = b0[0]` (the allocation `a0`) and `q =
/// b0[1]` (the allocation `a1`) with the `i64` lengths `b1[0..3]`: a `memcpy` from `q` to `p`, a
/// `memset` at `p + 64`, and a `memmove` within `a1` from `q` to `q + 8` (backward over the
/// overlap). Every endpoint is wide.
const MEM64_BODY: &str = "  %p = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %qp = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  %q = load ptr addrspace(1), ptr addrspace(1) %qp, align 8
  %l0 = load i64, ptr addrspace(1) %b1, align 8
  %l1p = getelementptr inbounds i64, ptr addrspace(1) %b1, i64 1
  %l1 = load i64, ptr addrspace(1) %l1p, align 8
  %l2p = getelementptr inbounds i64, ptr addrspace(1) %b1, i64 2
  %l2 = load i64, ptr addrspace(1) %l2p, align 8
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %p, ptr addrspace(1) align 4 %q, i64 %l0, i1 false)
  %p64 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 64
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %p64, i8 90, i64 %l1, i1 false)
  %q8 = getelementptr inbounds i8, ptr addrspace(1) %q, i64 8
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 %q8, ptr addrspace(1) align 4 %q, i64 %l2, i1 false)
";

const MEM64_DECLS: &str =
    "declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
";

/// An `i64` length between wide endpoints only is counted exactly in 64 bits (a wide allocation's
/// root entry has a `u64` size, so a length of 2³² or more may lie inside it): each of the four
/// loops (the `memcpy`, the `memset`, and both directions of the `memmove`) runs a `u64` counter.
/// The copies are exact for lengths that run the tail bytes alone, whole words alone, and both. A
/// length of 2³² + 5 runs past both allocations with every access contained until the interpreter's
/// step limit stops it (no execution bound does).
#[test]
fn i64_lengths_between_wide_endpoints_count_in_64_bits() {
    /// Lane-steps a 2³²-byte copy runs before the interpreter stops it: thousands of words past
    /// both allocations.
    const HUGE_STEPS: u64 = 1 << 18;
    let Some(bc) = assemble(
        "wide_mem64",
        &kernel("wide_mem64", 2, MEM64_BODY, MEM64_DECLS),
    ) else {
        return;
    };
    let mut ptrs = vec![0u8; 16];
    ptrs[..8].copy_from_slice(&alloc_guest(0).to_le_bytes());
    ptrs[8..].copy_from_slice(&alloc_guest(1).to_le_bytes());
    let a1: Vec<u8> = (0..256u32).map(|i| (i * 13 + 1) as u8).collect();
    let lengths = |l: [u64; 3]| -> Vec<u8> { l.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "wide_mem64", &p, &LowerInputs::new(), &what);
        assert_eq!(
            intrinsics(&plan),
            named(&[
                ("access.memcpy.dynamic.to_psb.from_psb", 1),
                ("access.memmove.dynamic.to_psb.from_psb", 1),
                ("access.memset.dynamic.to_psb", 1),
            ]),
            "{what}"
        );
        let counters = m
            .vars()
            .iter()
            .filter(|v| matches!(m.types().get(v.pointee), Ok(Type::Int { width: 64, .. })))
            .count();
        // The four loops' counters, and the base and length the module's one resident-set
        // search carries through its loop past its unrolled steps.
        assert_eq!(counters, 4 + 2, "{what}: u64 loop counters");
        for (l0, l1, l2) in [
            (0u64, 0u64, 0u64),
            (1, 1, 1),
            (3, 2, 7),
            (4, 5, 8),
            (61, 63, 101),
            (64, 64, 248),
        ] {
            let case = format!("{what} {l0} {l1} {l2}");
            let mem = Memory {
                slots: vec![ptrs.clone(), lengths([l0, l1, l2])],
                allocs: vec![vec![0x11; 128], a1.clone()],
            };
            let (res, at) = resources(&plan, &mem, 1);
            let mut a0 = vec![0x11u8; 128];
            a0[..l0 as usize].copy_from_slice(&a1[..l0 as usize]);
            a0[64..64 + l1 as usize].fill(90);
            let mut moved = a1.clone();
            moved.copy_within(0..l2 as usize, 8);
            for out in run_orders(&m, &res, &case) {
                assert_eq!(out[at.alloc(0)], a0, "{case}: a0");
                assert_eq!(out[at.alloc(1)], moved, "{case}: a1");
                let status = words(&out[at.status()]);
                assert_eq!(status[StatusWord::Status.index() as usize], 0, "{case}");
            }
        }
        let huge = lengths([(1 << 32) + 5, 0, 0]);
        let mem = Memory {
            slots: vec![ptrs.clone(), huge.clone()],
            allocs: vec![vec![0x11; 128], a1.clone()],
        };
        // The interpreter traps any access outside a bound buffer, so a step-limit trap (and no
        // other) shows every access the copy made past both allocations was contained.
        let (res, _) = resources(&plan, &mem, 1);
        for order in [Order::Forward, Order::Reverse] {
            let cfg = Config {
                order,
                step_limit: HUGE_STEPS,
                ..Config::default()
            };
            let err = run(&m, &cfg, res.clone()).expect_err("the copy outruns the step limit");
            assert_eq!(
                err.trap_kind(),
                Some(&TrapKind::StepLimit(HUGE_STEPS)),
                "{what} huge {order:?}: {err}"
            );
        }
    }
}

/// A dynamic `memmove` between two static slots (which one buffer bound twice would alias) compares
/// their Metal addresses (`SlotGuestBase` plus the displacement) and is exact; a profile without
/// `shaderInt64` refuses it, naming it.
#[test]
fn a_memmove_between_static_slots_compares_metal_addresses() {
    let body = "  %l = load i32, ptr addrspace(1) %b2, align 4
  %d = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 8
  call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) align 4 %d, ptr addrspace(1) align 4 %b1, i32 %l, i1 false)
";
    let decls =
        "declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)\n";
    let Some(bc) = assemble("slot_move", &kernel("slot_move", 3, body, decls)) else {
        return;
    };
    let mut all = profiles();
    for stem in ["vk12-floor", "swiftshader", "lunarg-desktop-baseline-2023"] {
        all.push((stem.to_owned(), profile(stem, false)));
    }
    let mut refused = 0;
    for (what, p) in all {
        if !p.fields().leaf.int64 {
            let e = compile(&bc, "slot_move", &p).expect_err("refused");
            let crate::error::Error::Unsupported(u) = e else {
                panic!("{what}: {e:?}");
            };
            assert_eq!(
                (u.feature, u.needs),
                (
                    crate::error::Feature::PointerCompare,
                    crate::error::Needs::AnyOf(vec![vec![crate::error::Capability::Feature(
                        VkFeature::ShaderInt64
                    )]])
                ),
                "{what}"
            );
            refused += 1;
            continue;
        }
        let (plan, m) = checked(&bc, "slot_move", &p, &LowerInputs::new(), &what);
        assert_eq!(
            intrinsics(&plan),
            named(&[("access.memmove.dynamic", 1)]),
            "{what}"
        );
        let src: Vec<u8> = (0..64u32).map(|i| (i * 5 + 1) as u8).collect();
        for l in [0u32, 13, 56] {
            let mem = Memory {
                slots: vec![vec![0x22; 64], src.clone(), le_words(&[l])],
                allocs: Vec::new(),
            };
            let (res, at) = resources(&plan, &mem, 1);
            for out in run_orders(&m, &res, &what) {
                let mut want = vec![0x22u8; 64];
                want[8..8 + l as usize].copy_from_slice(&src[..l as usize]);
                assert_eq!(out[at.slot(0)], want, "{what} {l}");
            }
        }
    }
    assert!(refused > 0, "no profile without shaderInt64");
}

/// A long `memmove` between slots aliasing one allocation compares their Metal addresses.
/// `[[buffer(0)]]` and `[[buffer(1)]]` are bound at bytes 512 and 704 of one allocation, both
/// through device addresses. The destination is 64 bytes below whichever of them `b2[0]`
/// selects (so both windows start at -64, below the bind offset) and the source is `b0`:
/// selecting `b0` needs a forward copy and selecting `b1` a backward one, each over a 300-byte
/// overlap. The direction is right only when the static address takes the selected
/// candidate's `SlotGuestBase` and recovers the negative displacement from the window start.
#[test]
fn a_memmove_between_slots_aliasing_one_allocation_is_exact() {
    let body = "  %c = load i32, ptr addrspace(1) %b2, align 4
  %first = icmp ne i32 %c, 0
  %s = select i1 %first, ptr addrspace(1) %b0, ptr addrspace(1) %b1
  %d = getelementptr inbounds i8, ptr addrspace(1) %s, i64 -64
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 1 %d, ptr addrspace(1) align 1 %b0, i64 300, i1 false)
";
    let decls =
        "declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)\n";
    let Some(bc) = assemble("alias_move", &kernel("alias_move", 3, body, decls)) else {
        return;
    };
    let mut inputs = LowerInputs::new();
    for index in [0, 1] {
        inputs.psb_slots.insert(SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index,
        });
    }
    let binds = [
        Bind {
            slot: 0,
            alloc: 0,
            offset: 512,
        },
        Bind {
            slot: 1,
            alloc: 0,
            offset: 704,
        },
    ];
    let data: Vec<u8> = (0..1024u32).map(|i| (i * 7 + 3) as u8).collect();
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "alias_move", &p, &inputs, &what);
        assert_eq!(
            intrinsics(&plan),
            named(&[("access.memmove.loop.to_psb.from_psb", 1)]),
            "{what}"
        );
        for index in [0, 1] {
            let c = plan
                .slots
                .iter()
                .flatten()
                .find(|c| c.slot.index == index)
                .expect("planned slot");
            assert_eq!(c.form, BufferForm::Psb, "{what}: b{index}");
            assert!(
                matches!(c.window, Window::Bounded { start: -64, .. }),
                "{what}: b{index} {:?}",
                c.window
            );
        }
        for (first, dst) in [(1u32, 512 - 64), (0, 704 - 64)] {
            let mem = Memory {
                slots: vec![vec![0; 4], vec![0; 4], le_words(&[first])],
                allocs: vec![data.clone()],
            };
            let (res, at) = resources_bound(&plan, &mem, 1, &binds);
            let mut want = data.clone();
            want.copy_within(512..812, dst);
            for out in run_orders(&m, &res, &what) {
                assert_eq!(out[at.alloc(0)], want, "{what}: into b{}", 1 - first);
            }
        }
    }
}

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// The public fixture `kernel_escaping_pointer_integers` subtracts the integer addresses of a
/// buffer and of a pointer 12 (device) or 28 (constant) bytes into it, as `i64` and as `i32`: every
/// difference is exact.
#[test]
fn the_escaping_pointer_integers_fixture_is_exact() {
    let Some(bc) = fixture("kernel_escaping_pointer_integers") else {
        return;
    };
    let entry = "escaping_pointer_integers";
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, entry, &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.ptrtoint"), 8, "{what}");
        let mem = Memory {
            slots: vec![vec![0; 16], vec![0; 32], vec![0xEE; 40]],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, 1);
        for out in run_orders(&m, &res, &what) {
            let w: Vec<u64> = out[at.slot(2)]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            assert_eq!(&w[1..], &[12, 28, 12, 28], "{what}");
        }
    }
}

/// The public fixture `kernel_buffer_address_round_trip` adds a host-supplied byte offset to the
/// integer address of a `constant` buffer and reads 6-byte rows back through the pointer it makes
/// (resolved through the resident set): every row is exact.
#[test]
fn the_buffer_address_round_trip_fixture_is_exact() {
    const N: u32 = 8;
    let Some(bc) = fixture("kernel_buffer_address_round_trip") else {
        return;
    };
    let entry = "buffer_address_round_trip";
    // Rows of three `ushort`s; the kernel starts one row in.
    let rows: Vec<u8> = (0..(N + 1) * 3)
        .flat_map(|i| (0x0101u16.wrapping_mul(i as u16 + 1)).to_le_bytes())
        .chain([0; 10])
        .collect();
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, entry, &p, &LowerInputs::new(), &what);
        let mem = Memory {
            slots: vec![
                rows.clone(),
                le_words(&[6, 0, 0, 0]),
                vec![0; 4 * N as usize],
            ],
            allocs: Vec::new(),
        };
        let (res, at) = resources(&plan, &mem, N);
        for out in run_orders(&m, &res, &what) {
            let o = words(&out[at.slot(2)]);
            for (g, x) in o.iter().enumerate() {
                let field = |k: usize| {
                    let at = 6 * (g + 1) + 2 * k;
                    u32::from(u16::from_le_bytes([rows[at], rows[at + 1]]))
                };
                let want = field(0) | (field(1) << 8) | (field(2) << 16);
                assert_eq!(*x, want, "{what}: row {g}");
            }
        }
    }
}

/// The shapes of a pointer's bits through a private cell, over a wide pointer: odd threads take the
/// pointer loaded from `b1` (the guest address of `b3`), even threads `b2` itself, so the pointer
/// parked in `%s` is wide on Profile A with a dynamic and a static root. Its bits are stored
/// (`ptrtoint` + `store i64`) and read back as a pointer, stored as a pointer and read back through
/// a pointer word (`load i64` + `inttoptr`), or both; each thread then writes three times the word
/// it reads through the pointer at its index.
fn wide_bits_body(shape: &str) -> String {
    let park = match shape {
        "bits" => {
            "  %bits = ptrtoint ptr addrspace(1) %pp to i64
  store i64 %bits, ptr %s, align 8
  %p = load ptr addrspace(1), ptr %s, align 8
"
        }
        "word" => {
            "  store ptr addrspace(1) %pp, ptr %s, align 8
  %w = load i64, ptr %s, align 8
  %p = inttoptr i64 %w to ptr addrspace(1)
"
        }
        "bits_word" => {
            "  %bits = ptrtoint ptr addrspace(1) %pp to i64
  store i64 %bits, ptr %s, align 8
  %w = load i64, ptr %s, align 8
  %p = inttoptr i64 %w to ptr addrspace(1)
"
        }
        other => panic!("no shape {other}"),
    };
    format!(
        "  %s = alloca ptr addrspace(1), align 8
  %ld = load ptr addrspace(1), ptr addrspace(1) %b1, align 8
  %c = and i32 %t, 1
  %odd = icmp ne i32 %c, 0
  %pp = select i1 %odd, ptr addrspace(1) %ld, ptr addrspace(1) %b2
{park}  %i = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = load i32, ptr addrspace(1) %q, align 4
  %x = mul i32 %v, 3
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %x, ptr addrspace(1) %o, align 4
"
    )
}

/// A wide pointer's cell bytes are its guest address, which is the integer itself, so its bits
/// stored into a forwarding cell, and a pointer word read from one, take the integer path (the
/// guest address stored and loaded as data, `ptrtoint` and `inttoptr` through the root table):
/// exact on Profile A in both scheduling orders; Profile F refuses the loaded pointer.
#[test]
fn a_wide_pointers_bits_through_a_cell_are_its_guest_address() {
    const N: u32 = 32;
    for shape in ["bits", "word", "bits_word"] {
        let name = format!("wide_{shape}");
        let Some(bc) = assemble(&name, &kernel(&name, 4, &wide_bits_body(shape), "")) else {
            return;
        };
        let b2: Vec<u32> = (0..N).map(|i| i * 7 + 1).collect();
        let b3: Vec<u32> = (0..N).map(|i| i * 11 + 1000).collect();
        let want: Vec<u32> = (0..N as usize)
            .map(|t| 3 * if t % 2 == 1 { b3[t] } else { b2[t] })
            .collect();
        for (what, p) in profiles() {
            let what = format!("{shape} {what}");
            let (plan, m) = checked(&bc, &name, &p, &LowerInputs::new(), &what);
            assert_eq!(uses(&plan, "access.cell.bits"), 0, "{what}");
            assert_eq!(uses(&plan, "access.cell.word"), 0, "{what}");
            let mem = Memory {
                slots: vec![
                    vec![0; 4 * N as usize],
                    slot_guest(3).to_le_bytes().to_vec(),
                    le_words(&b2),
                    le_words(&b3),
                ],
                allocs: Vec::new(),
            };
            let (res, at) = resources(&plan, &mem, N);
            for out in run_orders(&m, &res, &what) {
                assert_eq!(words(&out[at.slot(0)]), want, "{what}");
                let status = words(&out[at.status()]);
                assert_eq!(status[StatusWord::ResidentMisses.index() as usize], 0);
            }
        }
        for stem in ["vk12-floor", "lunarg-desktop-baseline-2023"] {
            let p = profile(stem, true);
            match typed_module_with(&bc, &name, &p, &LowerInputs::new()) {
                Err(crate::error::Error::Unsupported(u)) => assert_eq!(
                    u.feature,
                    crate::error::Feature::DereferenceLoadedPointer,
                    "{shape} {stem}: {u:?}"
                ),
                Err(e) => panic!("{shape} {stem}: {e}"),
                Ok(_) => panic!("{shape} {stem}: a loaded pointer on Profile F"),
            }
        }
    }
}

/// A loop that loads through a pointer read from a buffer, stores through it, and stores to a
/// static slot each iteration (the store that may alias anything, as far as a driver knows).
const LOOP_ACCESS: &str = "define void @loop_access(ptr addrspace(1) noundef \"air-buffer-no-alias\" %b0, ptr addrspace(1) noundef \"air-buffer-no-alias\" %b1, i32 %t) {
entry:
  %d = load ptr addrspace(1), ptr addrspace(1) %b0, align 8
  %tz = zext i32 %t to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %tz
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]
  %k = mul i64 %i, 64
  %k2 = add i64 %k, %tz
  %p = getelementptr inbounds i32, ptr addrspace(1) %d, i64 %k2
  %v = load i32, ptr addrspace(1) %p, align 4
  %acc1 = xor i32 %acc, %v
  store i32 %acc1, ptr addrspace(1) %q, align 4
  %w = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  store i32 %acc1, ptr addrspace(1) %w, align 4
  %i1 = add i64 %i, 1
  %done = icmp eq i64 %i1, 16
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

!air.kernel = !{!0}
!air.version = !{!90}
!air.language_version = !{!91}
!0 = !{ptr @loop_access, !1, !2}
!1 = !{}
!2 = !{!10, !11, !12}
!10 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"b0\"}
!11 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b1\"}
!12 = !{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}
!90 = !{i32 2, i32 8, i32 0}
!91 = !{!\"Metal\", i32 4, i32 0, i32 0}
";

/// An access through a wide pointer reads no root entry: the pointer carries its entry's `host`
/// and `size` from where it was formed (the resident-set search, before the loop), and the sink's
/// address is read once at entry. So a loop that stores each iteration, which a driver cannot
/// prove leaves the root table alone, has no `u64` device-address load in its blocks at all
/// (the old form read `host`, `bind_offset`, `size` and the sink's `host` per access: 8 here).
#[test]
fn accesses_through_a_loaded_pointer_read_no_root_entry_in_a_loop() {
    let Some(bc) = assemble("loop_access", LOOP_ACCESS) else {
        return;
    };
    for (what, p) in profiles() {
        let (plan, m) = checked(&bc, "loop_access", &p, &LowerInputs::new(), &what);
        assert_eq!(uses(&plan, "access.dynamic.load"), 1, "{what}");
        assert_eq!(uses(&plan, "access.dynamic.store"), 1, "{what}");
        let headers: Vec<(usize, usize)> = m
            .blocks()
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b.merge {
                Some(crate::spirv::ir::Merge::Loop { merge, .. }) => Some((i, merge.index())),
                _ => None,
            })
            .collect();
        assert!(!headers.is_empty(), "{what}: no loop");
        let is_entry_load = |ptr| match m.value(ptr).and_then(|d| m.types().get(d.ty)) {
            Ok(Type::Pointer {
                class: StorageClass::PhysicalStorageBuffer,
                pointee,
            }) => matches!(m.types().get(*pointee), Ok(Type::Int { width: 64, .. })),
            _ => false,
        };
        let is_memory = |ptr| {
            matches!(
                m.value(ptr).and_then(|d| m.types().get(d.ty)),
                Ok(Type::Pointer {
                    class: StorageClass::PhysicalStorageBuffer | StorageClass::StorageBuffer,
                    ..
                })
            )
        };
        let mut kernel_loops = 0;
        for (header, merge) in headers {
            // The loop's blocks: those reached from its header short of its merge block.
            let mut seen = vec![header];
            let mut work = vec![header];
            while let Some(b) = work.pop() {
                let succ: Vec<usize> = match &m.blocks()[b].term {
                    crate::spirv::ir::Terminator::Branch(t) => vec![t.index()],
                    crate::spirv::ir::Terminator::CondBranch { t, f, .. } => {
                        vec![t.index(), f.index()]
                    }
                    crate::spirv::ir::Terminator::Switch { default, cases, .. } => {
                        let mut v = vec![default.index()];
                        v.extend(cases.iter().map(|(_, c)| c.index()));
                        v
                    }
                    _ => Vec::new(),
                };
                for s in succ {
                    if s != merge && !seen.contains(&s) {
                        seen.push(s);
                        work.push(s);
                    }
                }
            }
            let insts = || seen.iter().flat_map(|&b| &m.blocks()[b].insts);
            // The kernel's loop, which stores to memory (the resident-set search's own loop over
            // the table, which stores only its variables, is not one).
            if !insts().any(|n| matches!(&n.inst, Inst::Store { ptr, .. } if is_memory(*ptr))) {
                continue;
            }
            kernel_loops += 1;
            let loads = insts()
                .filter(|n| matches!(&n.inst, Inst::Load { ptr, .. } if is_entry_load(*ptr)))
                .count();
            assert_eq!(loads, 0, "{what}: root-entry loads in the loop");
        }
        assert_eq!(kernel_loops, 1, "{what}: the kernel's loop");
    }
}
