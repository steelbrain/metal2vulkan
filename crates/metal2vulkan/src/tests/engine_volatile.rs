//! Volatile loads and stores end to end.
//!
//! A volatile access is the plain access whose every word load and store carries SPIR-V's
//! `Volatile` memory operand: through buffer slots (word and sub-word leaves), the threadgroup
//! arena, a typed threadgroup object, a typed alloca, the private arena, volatile `memcpy`, `memmove`, and `memset` (buffer
//! and arena endpoints), a heap texture handle's resource id, and next to a coherent access under
//! the Vulkan memory model. Each kernel passes L3 on every profile, keeps the function's memory
//! model (`GLSL450` unless a coherent access selects Vulkan), emits exactly the loads and stores
//! of its non-volatile twin, the volatile ones counted per storage class, and runs exactly on the
//! CPU interpreter in both scheduling orders. A handle-array or buffer-table element load, which
//! reads no memory at the access, is refused volatile against its plain twin.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::{Inst, StorageClass, Type, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Profile A (device addresses) and Profile F (none), with and without robust descriptors,
/// with and without the Vulkan memory model.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lavapipe",
    "vk12-floor",
    "lunarg-desktop-baseline-2023",
];

const THREADS: u32 = 32;

const SB: StorageClass = StorageClass::StorageBuffer;
const WG: StorageClass = StorageClass::Workgroup;
const PRIV: StorageClass = StorageClass::Private;

fn p(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("profile")
}

fn assemble(name: &str, text: &str) -> Option<Vec<u8>> {
    assemble_text(name, &format!("{HEADER}{text}"))
}

/// Assemble the whole module `text`.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-volatile-{}-{}-{name}",
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
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: bufs
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
            .collect(),
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

/// The module's loads and stores, and the volatile ones by the storage class of their pointer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Accesses {
    loads: u32,
    stores: u32,
    volatile: Volatiles,
}

/// Volatile loads and stores by the storage class of their pointer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Volatiles {
    loads: BTreeMap<StorageClass, u32>,
    stores: BTreeMap<StorageClass, u32>,
}

impl Volatiles {
    fn of(loads: &[(StorageClass, u32)], stores: &[(StorageClass, u32)]) -> Volatiles {
        Volatiles {
            loads: loads.iter().copied().collect(),
            stores: stores.iter().copied().collect(),
        }
    }

    fn none() -> Volatiles {
        Volatiles::default()
    }
}

fn accesses(m: &TypedModule) -> Accesses {
    let class = |ptr| match m.value(ptr).and_then(|d| m.types().get(d.ty)) {
        Ok(Type::Pointer { class, .. }) => *class,
        other => panic!("an access through {other:?}"),
    };
    let mut a = Accesses::default();
    for n in m.blocks().iter().flat_map(|b| &b.insts) {
        match &n.inst {
            Inst::Load { ptr, access } => {
                a.loads += 1;
                if access.volatile {
                    *a.volatile.loads.entry(class(*ptr)).or_default() += 1;
                }
            }
            Inst::Store { ptr, access, .. } => {
                a.stores += 1;
                if access.volatile {
                    *a.volatile.stores.entry(class(*ptr)).or_default() += 1;
                }
            }
            _ => {}
        }
    }
    a
}

/// Whether the module declares the Vulkan memory model.
fn vulkan_model(m: &TypedModule) -> bool {
    m.requirements()
        .spirv_capabilities
        .iter()
        .any(|c| format!("{c:?}") == "VulkanMemoryModel")
}

/// How many strategies of `plan` have a census name starting with `prefix`.
fn count(plan: &Plan, prefix: &str) -> u32 {
    plan.census()
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(_, (n, _))| *n)
        .sum()
}

/// The non-volatile twin of kernel `text`: without `volatile`, and its memory intrinsics' volatile
/// flags cleared.
fn twin(text: &str) -> String {
    text.replace(" volatile ", " ")
        .replace(", i1 true)", ", i1 false)")
}

/// Compile the volatile kernel `text` and its non-volatile twin on every profile: both pass L3
/// and emit the same loads and stores, the twin none volatile and the kernel exactly `want` of
/// the profile, in
/// the `GLSL450` model; run the kernel over `bufs`. Returns each profile's buffers and plan.
fn run_everywhere(
    name: &str,
    text: &str,
    bufs: &[Vec<u32>],
    want: impl Fn(&DeviceProfile) -> Volatiles,
) -> Vec<(&'static str, Plan, Vec<Vec<u32>>)> {
    let Some(bc) = assemble(name, text) else {
        return Vec::new();
    };
    let Some(plain) = assemble(name, &twin(text)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile(&bc, name, &prof).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem} {name}: L3: {e}"));
        let (plan, m) = typed_module(&bc, name, &prof).expect("module");
        let (_, twin) = typed_module(&plain, name, &prof).expect("twin");
        let (got, base) = (accesses(&m), accesses(&twin));
        assert_eq!(
            (got.loads, got.stores),
            (base.loads, base.stores),
            "{stem} {name}: the volatile form is the same accesses"
        );
        assert_eq!(base.volatile, Volatiles::none(), "{stem} {name}");
        assert_eq!(got.volatile, want(&prof), "{stem} {name}");
        assert!(
            !vulkan_model(&m),
            "{stem} {name}: volatile keeps the GLSL450 model"
        );
        let bufs = run_both(&m, &resources(&plan, bufs));
        out.push((stem, plan, bufs));
    }
    out
}

fn input(seed: u32) -> Vec<u32> {
    (0..THREADS)
        .map(|i| seed.wrapping_mul(i + 7) ^ 0x5a5a)
        .collect()
}

/// Volatile word and sub-word accesses of buffer slots: every word loaded or stored carries
/// `Volatile` (a sub-word store's word atomics are unchanged), and so does the store of a half
/// through the slot's `u16[]` view where the profile has one; the values are exact.
#[test]
fn volatile_buffer_accesses_are_exact() {
    let text = kernel(
        "buffers",
        3,
        "",
        r#"entry:
  %i = zext i32 %t to i64
  %s = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i
  %v = load volatile i32, ptr addrspace(1) %s, align 4
  %w = add i32 %v, 1
  %d = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store volatile i32 %w, ptr addrspace(1) %d, align 4
  %h = trunc i32 %v to i16
  %e = getelementptr inbounds i16, ptr addrspace(1) %b2, i64 %i
  store volatile i16 %h, ptr addrspace(1) %e, align 2
  ret void
"#,
    );
    let inp = input(3);
    let want0: Vec<u32> = inp.iter().map(|v| v.wrapping_add(1)).collect();
    let want2: Vec<u32> = inp
        .chunks(2)
        .map(|c| (c[0] & 0xffff) | (c[1] << 16))
        .collect();
    let bufs = [
        vec![0; THREADS as usize],
        inp,
        vec![0; THREADS as usize / 2],
    ];
    for (stem, _, out) in run_everywhere("buffers", &text, &bufs, |p: &DeviceProfile| {
        // The half's store is a plain store of a `u16[]` view where the profile has one.
        let leaf = &p.fields().leaf;
        let half = u32::from(leaf.storage_buffer_16bit && leaf.int16);
        Volatiles::of(&[(SB, 1)], &[(SB, 1 + half)])
    }) {
        assert_eq!(out[0], want0, "{stem}");
        assert_eq!(out[2], want2, "{stem}");
    }
}

/// The finite half `0x3C00 + 4t + k` (lane `k` of thread `t`'s `<4 x half>`), as a pair of words.
fn halves(t: u32) -> [u32; 2] {
    let h = |k: u32| 0x3C00 + 4 * t + k;
    [h(0) | (h(1) << 16), h(2) | (h(3) << 16)]
}

/// The corpus's threadgroup shapes: a `<4 x half>` stored volatile into a threadgroup array and
/// read back volatile by a neighbour after a barrier, and volatile byte stores into shared words
/// (the arena's sub-word atomics) read back as whole words. The `half` array is a typed
/// `Workgroup` object (`memory::plan`), so each of its volatile accesses is one volatile load or
/// store of the held vector; read back through a punned `<2 x i32>` instead, it stays in the arena,
/// where each of the vector's words carries `Volatile`. Both forms answer the same words.
#[test]
fn volatile_threadgroup_accesses_are_exact() {
    let halves_read = r#"  %x = load volatile <4 x half>, ptr addrspace(3) %h, align 8
  %d = getelementptr inbounds <4 x half>, ptr addrspace(1) %b0, i64 %i
  store <4 x half> %x, ptr addrspace(1) %d, align 8
"#;
    let words_read = r#"  %x = load volatile <2 x i32>, ptr addrspace(3) %h, align 8
  %d = getelementptr inbounds <2 x i32>, ptr addrspace(1) %b0, i64 %i
  store <2 x i32> %x, ptr addrspace(1) %d, align 8
"#;
    // (name, the neighbour's read, whether the array is typed, the volatile accesses)
    let forms = [
        (
            "threadgroup",
            halves_read,
            true,
            Volatiles::of(&[(SB, 2), (WG, 2)], &[(WG, 1)]),
        ),
        (
            "threadgroup_arena",
            words_read,
            false,
            Volatiles::of(&[(SB, 2), (WG, 3)], &[(WG, 2)]),
        ),
    ];
    let inp: Vec<u32> = (0..THREADS).flat_map(halves).collect();
    let want0: Vec<u32> = (0..THREADS)
        .flat_map(|t| halves((t + 1) % THREADS))
        .collect();
    let want2: Vec<u32> = (0..THREADS)
        .map(|t| {
            let k = 4 * (t / 4);
            k | ((k + 1) << 8) | ((k + 2) << 16) | ((k + 3) << 24)
        })
        .collect();
    let bufs = [
        vec![0; 2 * THREADS as usize],
        inp,
        vec![0; THREADS as usize],
    ];
    for (name, read, typed, want) in forms {
        let text = kernel(
            name,
            3,
            "@tg = internal addrspace(3) global [32 x <4 x half>] undef, align 8\n\
             @tb = internal addrspace(3) global [32 x i8] undef, align 4\n\
             declare void @air.wg.barrier(i32, i32)",
            &format!(
                r#"entry:
  %i = zext i32 %t to i64
  %s = getelementptr inbounds <4 x half>, ptr addrspace(1) %b1, i64 %i
  %v = load volatile <4 x half>, ptr addrspace(1) %s, align 8
  %g = getelementptr inbounds [32 x <4 x half>], ptr addrspace(3) @tg, i64 0, i64 %i
  store volatile <4 x half> %v, ptr addrspace(3) %g, align 8
  %c = getelementptr inbounds [32 x i8], ptr addrspace(3) @tb, i64 0, i64 %i
  %byte = trunc i32 %t to i8
  store volatile i8 %byte, ptr addrspace(3) %c, align 1
  call void @air.wg.barrier(i32 2, i32 1)
  %n = add i32 %t, 1
  %m = and i32 %n, 31
  %j = zext i32 %m to i64
  %h = getelementptr inbounds [32 x <4 x half>], ptr addrspace(3) @tg, i64 0, i64 %j
{read}  %q = lshr i64 %i, 2
  %wp = getelementptr inbounds i32, ptr addrspace(3) @tb, i64 %q
  %w = load volatile i32, ptr addrspace(3) %wp, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
"#
            ),
        );
        for (stem, plan, out) in
            run_everywhere(name, &text, &bufs, |_: &DeviceProfile| want.clone())
        {
            let local = &plan.families.access.local;
            assert_eq!(local.typed.len(), usize::from(typed), "{stem} {name}");
            if typed {
                assert_eq!(
                    count(&plan, "access.threadgroup.typed_store"),
                    1,
                    "{stem} {name}"
                );
                assert_eq!(
                    count(&plan, "access.threadgroup.typed_load"),
                    1,
                    "{stem} {name}"
                );
            } else {
                assert!(
                    count(&plan, "access.threadgroup.store") >= 2,
                    "{stem} {name}"
                );
            }
            assert_eq!(out[0], want0, "{stem} {name}");
            assert_eq!(out[2], want2, "{stem} {name}");
        }
    }
}

/// Volatile accesses of private memory: a typed alloca (the corpus's `float` slot) and the private
/// arena (an alloca a `memset` reaches), with a volatile byte store (the arena's plain
/// read-modify-write, both of whose words carry `Volatile`). The slot is written twice: a plain
/// slot one store writes is read as the value stored (`canon::forward`), which would leave the
/// twin without the slot's accesses.
#[test]
fn volatile_private_accesses_are_exact() {
    let text = kernel(
        "private",
        3,
        "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)",
        r#"entry:
  %f = alloca float, align 4
  %a = alloca [4 x i32], align 4
  %i = zext i32 %t to i64
  %s = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %i
  %x = load float, ptr addrspace(1) %s, align 4
  store volatile float %x, ptr %f, align 4
  store volatile float %x, ptr %f, align 4
  %y = load volatile float, ptr %f, align 4
  %z = fadd float %y, 1.0
  %d = getelementptr inbounds float, ptr addrspace(1) %b0, i64 %i
  store float %z, ptr addrspace(1) %d, align 4
  call void @llvm.memset.p0.i64(ptr align 4 %a, i8 0, i64 16, i1 false)
  %k = and i64 %i, 15
  %e = getelementptr inbounds i8, ptr %a, i64 %k
  %byte = trunc i32 %t to i8
  store volatile i8 %byte, ptr %e, align 1
  %q = lshr i64 %k, 2
  %wp = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 %q
  %w = load volatile i32, ptr %wp, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let inp: Vec<u32> = (0..THREADS).map(|t| (t as f32 * 0.5).to_bits()).collect();
    let want0: Vec<u32> = (0..THREADS)
        .map(|t| (t as f32 * 0.5 + 1.0).to_bits())
        .collect();
    let want2: Vec<u32> = (0..THREADS).map(|t| t << (8 * (t % 4))).collect();
    let bufs = [vec![0; THREADS as usize], inp, vec![0; THREADS as usize]];
    for (stem, plan, out) in run_everywhere("private", &text, &bufs, |_: &DeviceProfile| {
        Volatiles::of(&[(PRIV, 3)], &[(PRIV, 3)])
    }) {
        assert_eq!(count(&plan, "access.typed.store"), 2, "{stem}");
        assert!(count(&plan, "access.private.store") >= 1, "{stem}");
        assert_eq!(out[0], want0, "{stem}");
        assert_eq!(out[2], want2, "{stem}");
    }
}

/// A volatile `memcpy`'s units are volatile loads and stores.
#[test]
fn a_volatile_memcpy_is_volatile_units() {
    let text = kernel(
        "copy",
        2,
        "declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)",
        r#"entry:
  %i = zext i32 %t to i64
  %s = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i
  %d = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %d, ptr addrspace(1) align 4 %s, i64 4, i1 true)
  ret void
"#,
    );
    let inp = input(9);
    let bufs = [vec![0; THREADS as usize], inp.clone()];
    for (stem, _, out) in run_everywhere("copy", &text, &bufs, |_: &DeviceProfile| {
        Volatiles::of(&[(SB, 1)], &[(SB, 1)])
    }) {
        assert_eq!(out[0], inp, "{stem}");
    }
}

/// A volatile `memset` stores volatile words of its pattern, and a volatile `memmove` between
/// buffers loads and stores volatile units.
#[test]
fn a_volatile_memset_and_memmove_are_volatile_units() {
    let text = kernel(
        "fill_move",
        3,
        "declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)\n\
         declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)",
        r#"entry:
  %i = zext i32 %t to i64
  %d = getelementptr inbounds [2 x i32], ptr addrspace(1) %b0, i64 %i
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %d, i8 -85, i64 8, i1 true)
  %s = getelementptr inbounds [2 x i32], ptr addrspace(1) %b1, i64 %i
  %e = getelementptr inbounds [2 x i32], ptr addrspace(1) %b2, i64 %i
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 %e, ptr addrspace(1) align 4 %s, i64 8, i1 true)
  ret void
"#,
    );
    let inp: Vec<u32> = (0..2).flat_map(|k| input(11 + k)).collect();
    let n = 2 * THREADS as usize;
    let bufs = [vec![0; n], inp.clone(), vec![0; n]];
    for (stem, plan, out) in run_everywhere("fill_move", &text, &bufs, |_: &DeviceProfile| {
        Volatiles::of(&[(SB, 2)], &[(SB, 4)])
    }) {
        assert_eq!(count(&plan, "access.memset"), 1, "{stem}");
        assert_eq!(count(&plan, "access.memmove"), 1, "{stem}");
        assert_eq!(out[0], vec![0xABAB_ABAB; n], "{stem}");
        assert_eq!(out[2], inp, "{stem}");
    }
}

/// Volatile `memcpy`s whose endpoints are arenas: a buffer row into a private arena object, that
/// object into the threadgroup arena, and a neighbour's threadgroup row back into a buffer after a
/// barrier. Every unit's words carry `Volatile` on both sides.
#[test]
fn volatile_copies_between_arenas_are_volatile_units() {
    let text = kernel(
        "arena_copy",
        2,
        "@tg = internal addrspace(3) global [32 x [4 x i32]] undef, align 4\n\
         declare void @llvm.memcpy.p0.p1.i64(ptr, ptr addrspace(1), i64, i1)\n\
         declare void @llvm.memcpy.p3.p0.i64(ptr addrspace(3), ptr, i64, i1)\n\
         declare void @llvm.memcpy.p1.p3.i64(ptr addrspace(1), ptr addrspace(3), i64, i1)\n\
         declare void @air.wg.barrier(i32, i32)",
        r#"entry:
  %a = alloca [4 x i32], align 4
  %i = zext i32 %t to i64
  %s = getelementptr inbounds [4 x i32], ptr addrspace(1) %b1, i64 %i
  call void @llvm.memcpy.p0.p1.i64(ptr align 4 %a, ptr addrspace(1) align 4 %s, i64 16, i1 true)
  %g = getelementptr inbounds [32 x [4 x i32]], ptr addrspace(3) @tg, i64 0, i64 %i
  call void @llvm.memcpy.p3.p0.i64(ptr addrspace(3) align 4 %g, ptr align 4 %a, i64 16, i1 true)
  call void @air.wg.barrier(i32 2, i32 1)
  %n = add i32 %t, 1
  %m = and i32 %n, 31
  %j = zext i32 %m to i64
  %h = getelementptr inbounds [32 x [4 x i32]], ptr addrspace(3) @tg, i64 0, i64 %j
  %d = getelementptr inbounds [4 x i32], ptr addrspace(1) %b0, i64 %i
  call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1) align 4 %d, ptr addrspace(3) align 4 %h, i64 16, i1 true)
  ret void
"#,
    );
    let inp: Vec<u32> = (0..4).flat_map(|k| input(17 + k)).collect();
    let want: Vec<u32> = (0..THREADS as usize)
        .flat_map(|t| {
            let r = 4 * ((t + 1) % THREADS as usize);
            inp[r..r + 4].to_vec()
        })
        .collect();
    let bufs = [vec![0; 4 * THREADS as usize], inp];
    for (stem, plan, out) in run_everywhere("arena_copy", &text, &bufs, |_: &DeviceProfile| {
        Volatiles::of(
            &[(SB, 4), (PRIV, 4), (WG, 4)],
            &[(SB, 4), (PRIV, 4), (WG, 4)],
        )
    }) {
        assert_eq!(count(&plan, "access.memcpy"), 3, "{stem}");
        assert_eq!(out[0], want, "{stem}");
    }
}

/// Under the Vulkan memory model (a coherent store elsewhere selects it where the profile has the
/// model and its device scope): a volatile access keeps the function's `NonPrivatePointer` and
/// gains `Volatile`, with no availability or visibility operand of its own; the coherent volatile
/// load keeps both. Elsewhere (the floor) the function keeps `GLSL450` and its storage buffers are
/// decorated `Coherent`: every volatile access, the coherent one included, gains only `Volatile`.
/// Both forms run exactly.
#[test]
fn volatile_under_the_vulkan_model_adds_only_volatile() {
    let text = kernel(
        "model",
        2,
        "declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))\n\
         declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))",
        r#"entry:
  %i = zext i32 %t to i64
  %s = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i
  %v = load volatile i32, ptr addrspace(1) %s, align 4
  %d = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  call void @air.store.device_coherent.i32.p1i32(i32 %v, ptr addrspace(1) %d)
  %c = call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %d)
  %w = add i32 %c, 1
  store volatile i32 %w, ptr addrspace(1) %s, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("model", &text) else {
        return;
    };
    let inp = input(5);
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
        let a = compile(&bc, "model", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        if !vulkan {
            let coherent = crate::pipeline::testing::interlocked_coherence(&words)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(!coherent.vulkan, "{stem}: the memory model");
            assert!(coherent.buffer >= 2, "{stem}: {coherent:?}");
        }
        let (plan, m) = typed_module(&bc, "model", &prof).expect("module");
        assert_eq!(vulkan_model(&m), vulkan, "{stem}");
        let (mut plain_volatile, mut coherent_volatile) = (0, 0);
        for b in m.blocks() {
            for n in &b.insts {
                let (Inst::Load { access, .. } | Inst::Store { access, .. }) = &n.inst else {
                    continue;
                };
                if !access.volatile {
                    continue;
                }
                assert_eq!(access.non_private, vulkan, "{stem}: {access:?}");
                if access.make_visible.is_some() {
                    coherent_volatile += 1;
                } else {
                    assert!(access.make_available.is_none(), "{stem}: {access:?}");
                    plain_volatile += 1;
                }
            }
        }
        let want_counts = if vulkan { (2, 1) } else { (3, 0) };
        assert_eq!(
            (plain_volatile, coherent_volatile),
            want_counts,
            "{stem}: the volatile load and store, and the coherent volatile load"
        );
        let out = run_both(
            &m,
            &resources(&plan, &[vec![0; THREADS as usize], inp.clone()]),
        );
        let want: Vec<u32> = inp.iter().map(|v| v.wrapping_add(1)).collect();
        assert_eq!((&out[0], &out[1]), (&inp, &want), "{stem}");
    }
    assert!(
        forms.iter().all(|&n| n > 0),
        "both memory-model forms run: {forms:?}"
    );
}

/// A volatile load of read-only memory (a typed constant object, and the constant blob a copied
/// global lives in) is the plain load: the module is its twin's, word for word.
#[test]
fn volatile_loads_of_read_only_memory_are_the_plain_loads() {
    let text = kernel(
        "readonly",
        1,
        "@k = internal addrspace(2) constant [4 x i32] [i32 3, i32 5, i32 7, i32 11], align 4\n\
         @table = internal addrspace(2) constant [4 x i32] [i32 1, i32 2, i32 4, i32 8], align 4\n\
         declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)",
        r#"entry:
  %tmp = alloca [4 x i32], align 4
  call void @llvm.memcpy.p0.p2.i64(ptr align 4 %tmp, ptr addrspace(2) align 4 @table, i64 16, i1 false)
  %i = zext i32 %t to i64
  %j = and i64 %i, 3
  %q = getelementptr inbounds [4 x i32], ptr addrspace(2) @k, i64 0, i64 %j
  %a = load volatile i32, ptr addrspace(2) %q, align 4
  %r = getelementptr inbounds [4 x i32], ptr addrspace(2) @table, i64 0, i64 %j
  %b = load volatile i32, ptr addrspace(2) %r, align 4
  %u = getelementptr inbounds [4 x i32], ptr %tmp, i64 0, i64 %j
  %c = load i32, ptr %u, align 4
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  %o = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i
  store i32 %abc, ptr addrspace(1) %o, align 4
  ret void
"#,
    );
    let Some(bc) = assemble("readonly", &text) else {
        return;
    };
    let Some(plain) = assemble("readonly", &text.replace(" volatile ", " ")) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        let a = compile(&bc, "readonly", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        let (plan, _) = typed_module(&bc, "readonly", &prof).expect("module");
        assert!(count(&plan, "access.constant.load") >= 1, "{stem}");
        assert!(count(&plan, "access.blob.load") >= 1, "{stem}");
        let twin = compile(&plain, "readonly", &prof).expect("twin");
        assert!(
            a.module == twin.module,
            "{stem}: the loads are the plain ones"
        );
    }
}

/// A volatile load of an element pointer of a buffer table (an `array_ref` of device pointers),
/// which yields the element's static fat pointer and reads no memory at the access, is refused; the
/// plain load compiles and passes L3.
#[test]
fn a_volatile_buffer_table_load_is_refused() {
    let table = |name: &str, body: &str| {
        format!(
            "define void @{name}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr noundef %1) {{\nentry:\n{body}  ret void\n}}\n\n\
             !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
             !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{!12, !13}}\n\
             !12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
             !13 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 3, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tab\"}}\n\
             !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
        )
    };
    let body = "  %t2 = getelementptr inbounds ptr addrspace(1), ptr %1, i64 2\n  %e2 = load volatile ptr addrspace(1), ptr %t2, align 8\n  %a = load i32, ptr addrspace(1) %e2, align 4\n  store i32 %a, ptr addrspace(1) %0, align 4\n";
    let Some(bc) = assemble("table", &table("table", body)) else {
        return;
    };
    let Some(plain) = assemble("table", &table("table", &body.replace(" volatile ", " "))) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match compile(&bc, "table", &prof) {
            Err(e) => assert!(not_implemented_access(&e), "{stem}: {e:?}"),
            Ok(_) => panic!("{stem}: a volatile table load compiled"),
        }
        let a = compile(&plain, "table", &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// The `engine/handles` fixture with each `(from, to)` edit, and the same with the load
/// `line` of kernel `entry` made volatile: `(volatile, plain)`. An edit and the load are found in
/// `entry`'s body (its `define` through its closing brace), where each must match once, since the
/// fixture's other kernels repeat some of its lines; an edit matching nothing there must match
/// once in the rest of the fixture. `first_argument` edits `entry`'s own copy of its first
/// argument's metadata ([`detach_first_argument`]).
fn handles_with_volatile(
    entry: &str,
    line: &str,
    edits: &[(&str, &str)],
    first_argument: Option<(&str, &str)>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let path = engine_dir().join("handles.ll");
    let mut text = std::fs::read_to_string(&path).expect("fixture");
    if let Some((from, to)) = first_argument {
        text = detach_first_argument(&text, entry, from, to);
    }
    let start = text
        .find(&format!("define void @{entry}("))
        .unwrap_or_else(|| panic!("the fixture defines {entry}"));
    let end = start + text[start..].find("\n}\n").expect("its closing brace") + 3;
    let (mut head, mut body, mut tail) = (
        text[..start].to_string(),
        text[start..end].to_string(),
        text[end..].to_string(),
    );
    for (from, to) in edits {
        if body.matches(from).count() == 1 {
            body = body.replace(from, to);
            continue;
        }
        let outside = head.matches(from).count() + tail.matches(from).count();
        assert_eq!(
            (body.matches(from).count(), outside),
            (0, 1),
            "the fixture has one {from}"
        );
        head = head.replace(from, to);
        tail = tail.replace(from, to);
    }
    assert_eq!(body.matches(line).count(), 1, "{entry} has one {line}");
    let volatile = body.replace(line, &line.replacen(" load ", " load volatile ", 1));
    let plain = assemble_text("handles", &format!("{head}{body}{tail}"))?;
    let volatile = assemble_text("handles_volatile", &format!("{head}{volatile}{tail}"))?;
    Some((volatile, plain))
}

/// `text` with `entry`'s `!air.kernel` argument list and its first argument's node given their
/// own copies at fresh metadata ids, `(from, to)` applied to the copied argument node. LLVM
/// uniques metadata, so kernels with the same signature share these nodes; editing the shared
/// node would change the other kernels' argument ABI too.
fn detach_first_argument(text: &str, entry: &str, from: &str, to: &str) -> String {
    // The body (`!{...}`) of numbered node `id`, from its definition line.
    let body_of = |id: &str| -> &str {
        let key = format!("\n{id} = ");
        let start = text
            .find(&key)
            .unwrap_or_else(|| panic!("the fixture defines {id}"))
            + key.len();
        &text[start..start + text[start..].find('\n').expect("a line")]
    };
    let next = text
        .lines()
        .filter_map(|l| l.strip_prefix('!')?.split(' ').next()?.parse::<u32>().ok())
        .max()
        .expect("numbered metadata")
        + 1;
    let (list, arg) = (format!("!{next}"), format!("!{}", next + 1));
    let kernel = text
        .find(&format!("!{{ptr @{entry}, "))
        .unwrap_or_else(|| panic!("{entry} is a kernel"));
    let kernel_end = kernel + text[kernel..].find('}').expect("its node");
    let old_list = text[kernel..kernel_end]
        .rsplit(", ")
        .next()
        .expect("its arguments");
    let list_body = body_of(old_list);
    let old_arg = list_body
        .trim_start_matches("!{")
        .split(", ")
        .next()
        .expect("a first argument")
        .trim_end_matches('}');
    let arg_body = body_of(old_arg);
    assert_eq!(
        arg_body.matches(from).count(),
        1,
        "{old_arg} has one {from}"
    );
    let new_list = list_body.replacen(old_arg, &arg, 1);
    let new_arg = arg_body.replace(from, to);
    let mut out = String::with_capacity(text.len() + new_list.len() + new_arg.len() + 16);
    out.push_str(&text[..kernel_end - old_list.len()]);
    out.push_str(&list);
    out.push_str(&text[kernel_end..]);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&format!("{list} = {new_list}\n{arg} = {new_arg}\n"));
    out
}

/// `argument_buffer_read` with its argument buffer in `device` memory instead of `constant`.
const DEVICE_ARGUMENT_BUFFER: [(&str, &str); 3] = [
    (
        "ptr addrspace(2) noundef readonly align 8 captures(none) dereferenceable(8) \"air-buffer-no-alias\" %0",
        "ptr addrspace(1) noundef align 8 dereferenceable(8) \"air-buffer-no-alias\" %0",
    ),
    (
        "getelementptr inbounds %struct.Resources, ptr addrspace(2) %0",
        "getelementptr inbounds %struct.Resources, ptr addrspace(1) %0",
    ),
    (
        "%6 = load ptr addrspace(1), ptr addrspace(2) %5",
        "%6 = load ptr addrspace(1), ptr addrspace(1) %5",
    ),
];

/// `argument_buffer_read`'s argument-buffer metadata for a `device` argument buffer.
const DEVICE_ARGUMENT_METADATA: (&str, &str) = (
    "!\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !39",
    "!\"air.read_write\", !\"air.address_space\", i32 1, !\"air.struct_type_info\", !39",
);

/// A volatile load of a heap texture handle (an argument-buffer member) is its plain lookup with
/// the resource id's words volatile: where the plain kernel compiles, the volatile one compiles
/// with the same loads and passes L3; where the plain kernel is refused (no descriptor indexing),
/// the volatile one is refused alike. A `device` argument buffer's two id words carry `Volatile`; a
/// `constant` one is read through its uniform view, whose read-only words stay plain, so its module
/// is the twin's.
#[test]
fn a_volatile_heap_texture_handle_load_is_its_lookup() {
    let entry = "argument_buffer_read";
    let mut compiled = 0;
    for (space, edits, first_argument, line, want) in [
        (
            "constant",
            &[][..],
            None,
            "%6 = load ptr addrspace(1), ptr addrspace(2) %5, align 8",
            Volatiles::none(),
        ),
        (
            "device",
            &DEVICE_ARGUMENT_BUFFER[..],
            Some(DEVICE_ARGUMENT_METADATA),
            "%6 = load ptr addrspace(1), ptr addrspace(1) %5, align 8",
            Volatiles::of(&[(SB, 2)], &[]),
        ),
    ] {
        let Some((bc, plain)) = handles_with_volatile(entry, line, edits, first_argument) else {
            return;
        };
        for stem in PROFILES {
            let what = format!("{stem} {space}");
            let prof = p(stem);
            let twin = match compile(&plain, entry, &prof) {
                Ok(twin) => twin,
                Err(e) => {
                    let refused = matches!(&e, Error::Unsupported(u)
                        if u.feature == Feature::ArgumentBufferTextures);
                    assert!(refused, "{what}: {e}");
                    assert_eq!(compile(&bc, entry, &prof).err(), Some(e), "{what}");
                    continue;
                }
            };
            let a = compile(&bc, entry, &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (plan, m) = typed_module(&bc, entry, &prof).expect("module");
            let (_, base) = typed_module(&plain, entry, &prof).expect("twin");
            let (got, base) = (accesses(&m), accesses(&base));
            assert_eq!((got.loads, got.stores), (base.loads, base.stores), "{what}");
            assert_eq!(base.volatile, Volatiles::none(), "{what}");
            assert_eq!(count(&plan, "access.handle.lookup"), 1, "{what}");
            assert_eq!(got.volatile, want, "{what}");
            assert_eq!(
                a.module == twin.module,
                want == Volatiles::none(),
                "{what}: the module is the twin's exactly when no word is volatile"
            );
            compiled += 1;
        }
    }
    assert!(compiled > 0, "no profile has descriptor indexing");
}

/// A volatile load of a texture-array element, which yields the element's static handle and reads
/// no memory at the access, is refused; the plain load compiles and passes L3.
#[test]
fn a_volatile_texture_array_element_load_is_refused() {
    let entry = "array_read";
    let Some((bc, plain)) = handles_with_volatile(
        entry,
        "%9 = load ptr addrspace(1), ptr %8, align 8",
        &[],
        None,
    ) else {
        return;
    };
    for stem in PROFILES {
        let prof = p(stem);
        match compile(&bc, entry, &prof) {
            Err(e) => assert!(not_implemented_access(&e), "{stem}: {e:?}"),
            Ok(_) => panic!("{stem}: a volatile texture-array element load compiled"),
        }
        let a = compile(&plain, entry, &prof).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
    }
}

/// A refusal of `Lowering(Access)`, `NotImplemented`.
fn not_implemented_access(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u)
        if u.feature == Feature::Lowering(LowerFamily::Access) && u.needs == Needs::NotImplemented)
}
