//! A kernel's threadgroup memory on `Device::from_queries`' profiles
//! (`ThreadgroupMemory::Native`).
//!
//! - Where every access of the arena is a whole, aligned `float` word (a scalar, a `float2` or a
//!   `float4`), the arena is a `Workgroup` `float` array of its own words: one access chain and one
//!   typed load or store per lane, with no check. A static arena and one of `[[threadgroup(n)]]`
//!   regions both. Every answer is exact, in both scheduling orders.
//! - No threadgroup access is contained: a composed arena's (an `int` word's) and a typed
//!   threadgroup object's carry no check.
//! - The array is Metal's allocation: a region filling `maxComputeSharedMemorySize` is planned, one
//!   byte more is refused, and a static arena's array is its own words. An arena filling the limit
//!   answers every word.
//!
//! An access past threadgroup memory is undefined, in Metal as here; no test reads one.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, BoundBuffer, StageTable};
use crate::budget::Budget;
use crate::device::profile::DeviceProfile;
use crate::device::{Device, DevicePolicy};
use crate::error::{BindError, Feature};
use crate::library::Library;
use crate::lower::select::Plan;
use crate::memory::arena::TgWords;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, description, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// Profiles of four drivers.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lavapipe",
    "amd-780m-radv",
    "nvidia-rtx50-linux",
];

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// `stem` through `Device::from_queries`.
fn queried_device(stem: &str) -> Device {
    let desc = description(stem).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

fn queried(stem: &str) -> DeviceProfile {
    queried_device(stem).profile().clone()
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-tg-floats-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{body}"),
    }
}

/// The body over the 64 `float`s at `%base`, with 64 threads:
///
/// - thread `t` stores `t` at word `t`, and the `float4` of words `4 k` to `4 k + 3`'s indices at
///   word `4 k`, `k = t & 15`: each word holds its index, whatever the order of the stores;
/// - after a barrier, `out[t]` is the word `63 - t`, `out[64 + t]` the sum of the `float4` at
///   `4 (t & 15)` and `out[128 + t]` the sum of the `float2` at `2 (t & 31)`.
const BODY: &str = r#"  %w = and i32 %t, 63
  %fw = uitofp i32 %w to float
  %pa = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %w
  store float %fw, ptr addrspace(3) %pa, align 4
  %k = and i32 %t, 15
  %w4 = shl i32 %k, 2
  %f4 = uitofp i32 %w4 to float
  %v0 = insertelement <4 x float> poison, float %f4, i32 0
  %f41 = fadd float %f4, 1.000000e+00
  %v1 = insertelement <4 x float> %v0, float %f41, i32 1
  %f42 = fadd float %f4, 2.000000e+00
  %v2 = insertelement <4 x float> %v1, float %f42, i32 2
  %f43 = fadd float %f4, 3.000000e+00
  %v3 = insertelement <4 x float> %v2, float %f43, i32 3
  %p4 = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %w4
  store <4 x float> %v3, ptr addrspace(3) %p4, align 16
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 63, %t
  %q = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %r
  %x = load float, ptr addrspace(3) %q, align 4
  %xo = fptoui float %x to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %xo, ptr addrspace(1) %o, align 4
  %j = and i32 %t, 15
  %j4 = shl i32 %j, 2
  %q4 = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %j4
  %y = load <4 x float>, ptr addrspace(3) %q4, align 16
  %y0 = extractelement <4 x float> %y, i32 0
  %y1 = extractelement <4 x float> %y, i32 1
  %y2 = extractelement <4 x float> %y, i32 2
  %y3 = extractelement <4 x float> %y, i32 3
  %ya = fadd float %y0, %y1
  %yb = fadd float %y2, %y3
  %ys = fadd float %ya, %yb
  %yo = fptoui float %ys to i32
  %t64 = add i32 %t, 64
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t64
  store i32 %yo, ptr addrspace(1) %o4, align 4
  %h = and i32 %t, 31
  %h2 = shl i32 %h, 1
  %q2 = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %h2
  %z = load <2 x float>, ptr addrspace(3) %q2, align 8
  %z0 = extractelement <2 x float> %z, i32 0
  %z1 = extractelement <2 x float> %z, i32 1
  %zs = fadd float %z0, %z1
  %zo = fptoui float %zs to i32
  %t128 = add i32 %t, 128
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t128
  store i32 %zo, ptr addrspace(1) %o2, align 4
  ret void
"#;

const TAIL: &str = r#"
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!8}
!air.language_version = !{!9}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!0 = !{ptr @NAME, !1, !2}
!1 = !{}
"#;

const OUT_MD: &str = r#"!{i32 IX, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}"#;

const TID_MD: &str = r#"!{i32 IX, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}"#;

/// `BODY` over a static `[words x float]` global, whose `float4` and `float2` loads keep it in the
/// arena (they are not its element).
fn static_kernel(words: u32) -> String {
    static_of(words, BODY)
}

/// `body` over a static `[words x float]` global.
fn static_of(words: u32, body: &str) -> String {
    format!(
        "@g = internal addrspace(3) global [{words} x float] undef, align 16\n\n\
define void @tg_static(ptr addrspace(1) %out, i32 %t) {{\n{}}}\n{}\
!2 = !{{!3, !4}}\n!3 = {}\n!4 = {}\n",
        body.replace("BASE", "@g"),
        TAIL.replace("NAME", "tg_static"),
        OUT_MD.replace("IX", "0"),
        TID_MD.replace("IX", "1"),
    )
}

/// `BODY` over a `[[threadgroup(0)]]` `float` region, the arena's only part.
fn region_kernel() -> String {
    region_of(BODY)
}

/// `body` over a `[[threadgroup(0)]]` `float` region, the arena's only part.
fn region_of(body: &str) -> String {
    let region = r#"!{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}"#;
    format!(
        "define void @tg_region(ptr addrspace(3) %a, ptr addrspace(1) %out, i32 %t) {{\n{}}}\n{}\
!2 = !{{!3, !4, !5}}\n!3 = {region}\n!4 = {}\n!5 = {}\n",
        body.replace("BASE", "%a"),
        TAIL.replace("NAME", "tg_region"),
        OUT_MD.replace("IX", "1"),
        TID_MD.replace("IX", "2"),
    )
}

fn out_slot() -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: 0,
    }
}

/// Every word of an arena of `WORDS` `float`s over the `float`s at `BASE`, with 64 threads: each
/// thread stores each of its words' indices; after a barrier, `out[k]` is word `WORDS - 1 - k`
/// and `out[WORDS + j]` the sum of the `float4` at word `4 j`, `16 j + 6`.
const FILL: &str = r#"entry:
  br label %fill

fill:
  %i = phi i32 [ %t, %entry ], [ %i2, %fill ]
  %f = uitofp i32 %i to float
  %pa = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %i
  store float %f, ptr addrspace(3) %pa, align 4
  %i2 = add i32 %i, 64
  %more = icmp ult i32 %i2, WORDS
  br i1 %more, label %fill, label %sync

sync:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %back

back:
  %k = phi i32 [ %t, %sync ], [ %k2, %back ]
  %r = sub i32 LAST, %k
  %q = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %r
  %x = load float, ptr addrspace(3) %q, align 4
  %xo = fptoui float %x to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %k
  store i32 %xo, ptr addrspace(1) %o, align 4
  %k2 = add i32 %k, 64
  %more2 = icmp ult i32 %k2, WORDS
  br i1 %more2, label %back, label %vectors

vectors:
  %j = phi i32 [ %t, %back ], [ %j2, %vectors ]
  %j4 = shl i32 %j, 2
  %q4 = getelementptr inbounds float, ptr addrspace(3) BASE, i32 %j4
  %y = load <4 x float>, ptr addrspace(3) %q4, align 16
  %y0 = extractelement <4 x float> %y, i32 0
  %y1 = extractelement <4 x float> %y, i32 1
  %y2 = extractelement <4 x float> %y, i32 2
  %y3 = extractelement <4 x float> %y, i32 3
  %ya = fadd float %y0, %y1
  %yb = fadd float %y2, %y3
  %ys = fadd float %ya, %yb
  %yo = fptoui float %ys to i32
  %at = add i32 WORDS, %j
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %at
  store i32 %yo, ptr addrspace(1) %o4, align 4
  %j2 = add i32 %j, 64
  %more3 = icmp ult i32 %j2, QUADS
  br i1 %more3, label %vectors, label %done

done:
  ret void
"#;

/// [`FILL`] over `words` words: a static `[words x float]` global, or (`region`) a
/// `[[threadgroup(0)]]` `float` region.
fn fill_kernel(words: u32, region: bool) -> String {
    let body = FILL
        .replace("WORDS", &words.to_string())
        .replace("LAST", &(words - 1).to_string())
        .replace("QUADS", &(words / 4).to_string());
    if region {
        region_of(&body)
    } else {
        static_of(words, &body)
    }
}

/// Output buffer 0 of 192 zero words, 64 threads, and `spec` besides.
fn resources(plan: &Plan, spec: &[(SpecKey, u64)]) -> Resources {
    resources_of(plan, spec, 192)
}

/// Output buffer 0 of `out_words` zero words, 64 threads, and `spec` besides.
fn resources_of(plan: &Plan, spec: &[(SpecKey, u64)], out_words: usize) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), 64);
    s.insert(SpecKey::LocalSize(Axis::Y), 1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, 64);
    s.extend(spec.iter().copied());
    // Zero push bytes: the output is bound at its start, and the dispatch is one region at the
    // grid's origin.
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(
            matches!(
                (f.field, f.loc),
                (
                    PushField::SlotBindOffset(_) | PushField::DispatchRegion,
                    PushLoc::Push { .. }
                )
            ),
            "unexpected push field {f:?}"
        );
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(out_slot()),
            address: None,
            bytes: vec![0; out_words * 4],
        }],
        push,
        spec: s,
    }
}

/// The output words, in both scheduling orders (which must agree).
fn run_both(m: &TypedModule, res: &Resources) -> Vec<u32> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        outs.push(
            out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

/// The answers of `BODY` over its 64-word arena: each word its index.
fn want() -> Vec<u32> {
    // The words an access of `lanes` lanes at word `b` reads.
    let lanes = |b: u32, lanes: u32| -> u32 { (b..b + lanes).sum() };
    let mut out: Vec<u32> = (0..64u32).map(|t| lanes(63 - t, 1)).collect();
    out.extend((0..64u32).map(|t| lanes(4 * (t & 15), 4)));
    out.extend((0..64u32).map(|t| lanes(2 * (t & 31), 2)));
    out
}

/// What the module's words say of its `Workgroup` variables and their accesses.
#[derive(Debug, Default)]
struct Shape {
    /// Each `Workgroup` variable's element: `Some(32)` for a 32-bit `float` array, `None` for any
    /// other.
    float_arrays: Vec<Option<u32>>,
    /// The constant length of each `Workgroup` `float` array where it is an `OpConstant`.
    lengths: Vec<u32>,
    /// `OpAccessChain`s into a `Workgroup` `float` array.
    chains: usize,
    labels: usize,
    merges: usize,
    bitcasts: usize,
}

fn shape(words: &[u32]) -> Shape {
    const TYPE_FLOAT: u32 = 22;
    const TYPE_ARRAY: u32 = 28;
    const TYPE_POINTER: u32 = 32;
    const CONSTANT: u32 = 43;
    const VARIABLE: u32 = 59;
    const ACCESS_CHAIN: u32 = 65;
    const BITCAST: u32 = 124;
    const SELECTION_MERGE: u32 = 247;
    const LABEL: u32 = 248;
    const WORKGROUP: u32 = 4;
    let mut floats = BTreeMap::new();
    let mut arrays = BTreeMap::new();
    let mut pointers = BTreeMap::new();
    let mut constants = BTreeMap::new();
    let mut vars = Vec::new();
    let mut s = Shape::default();
    let mut chains = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (n, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        let w = &words[i..i + n.max(1)];
        match op {
            TYPE_FLOAT => {
                floats.insert(w[1], w[2]);
            }
            TYPE_ARRAY => {
                arrays.insert(w[1], (w[2], w[3]));
            }
            TYPE_POINTER => {
                pointers.insert(w[1], (w[2], w[3]));
            }
            CONSTANT if n == 4 => {
                constants.insert(w[2], w[3]);
            }
            VARIABLE if w[3] == WORKGROUP => vars.push((w[1], w[2])),
            ACCESS_CHAIN => chains.push(w[3]),
            BITCAST => s.bitcasts += 1,
            SELECTION_MERGE => s.merges += 1,
            LABEL => s.labels += 1,
            _ => {}
        }
        i += n.max(1);
    }
    let mut float_vars = Vec::new();
    for (ty, id) in vars {
        let array = pointers.get(&ty).and_then(|(_, p)| arrays.get(p));
        let elem = array.and_then(|(e, _)| floats.get(e)).copied();
        s.float_arrays.push(elem);
        if let (Some(32), Some((_, len))) = (elem, array) {
            float_vars.push(id);
            if let Some(len) = constants.get(len) {
                s.lengths.push(*len);
            }
        }
    }
    s.chains = chains.iter().filter(|b| float_vars.contains(b)).count();
    s
}

/// A kernel's bitcode, its entry, and the specialization of a 64-word arena.
type Case<'a> = (&'a [u8], &'a str, Spec<'a>);

/// Specialization constants.
type Spec<'a> = &'a [(SpecKey, u64)];

/// Each kernel through `from_queries` is of `float` words: one `Workgroup` `float` array (of 64
/// words for the static 64), 12 access chains into it (one a lane: 1 + 4 stored, 1 + 4 + 2
/// loaded), the `.floats` census names, no threadgroup check, and the exact answers.
#[test]
fn an_arena_of_float_accesses_is_a_float_array_through_from_queries() {
    let Some(st) = assemble("tg_static", &static_kernel(64)) else {
        return;
    };
    let Some(rg) = assemble("tg_region", &region_kernel()) else {
        return;
    };
    // The kernel and the specialization of a 64-word arena.
    let short = [(SpecKey::TgLength(0), 256)];
    let cases: [Case<'_>; 2] = [(&st, "tg_static", &[]), (&rg, "tg_region", &short)];
    for stem in PROFILES {
        let prof = queried(stem);
        for (bc, entry, spec) in cases {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            // Metal's static length.
            let metal = if entry == "tg_static" { 256 } else { 0 };
            assert_eq!(a.kernel.static_threadgroup_bytes, metal, "{what}");
            let s = shape(&a.module.words);
            assert_eq!(s.float_arrays, vec![Some(32)], "{what}: {s:?}");
            if entry == "tg_static" {
                assert_eq!(s.lengths, vec![64], "{what}: {s:?}");
            }
            assert_eq!(s.chains, 12, "{what}: {s:?}");
            let (plan, m) = typed_module(bc, entry, &prof).expect("module");
            let tg = plan.families.access.local.tg.as_ref().expect("an arena");
            assert_eq!(tg.words, TgWords::Floats, "{what}");
            let census = plan.census();
            for name in [
                "access.threadgroup.store.floats",
                "access.threadgroup.load.floats",
            ] {
                assert!(census.contains_key(name), "{what}: {name} in {census:?}");
            }
            assert!(!contains_threadgroup(&census), "{what}: {census:?}");
            assert_eq!(run_both(&m, &resources(&plan, spec)), want(), "{what}");
        }
    }
}

/// Whether `census` names a contained threadgroup access: a checked arena access or atomic, or a
/// clamped typed threadgroup object's.
fn contains_threadgroup(census: &BTreeMap<String, (u32, u64)>) -> bool {
    census.keys().any(|k| {
        k.starts_with("access.threadgroup") && (k.ends_with(".checked") || k.ends_with(".clamped"))
            || k.starts_with("access.atomic.threadgroup") && k.ends_with(".checked")
    })
}

/// Thread `t` of 64 stores `t` as an `int` at word `t` of a 64-word `[[threadgroup(0)]]` region
/// (the arena composed: an `int` is not a `float` word), adds 1 to word 0 by an atomic, and after
/// a barrier `out[t]` is word `63 - t`; and stores `t` at element `t` of a typed threadgroup
/// `[64 x half]` object, `out[64 + t]` its element `63 - t`.
const COMPOSED: &str = r#"@h = internal addrspace(3) global [64 x half] undef, align 2

define void @tg_composed(ptr addrspace(3) %a, ptr addrspace(1) %out, i32 %t) {
  %w = and i32 %t, 63
  %pa = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %w
  store i32 %w, ptr addrspace(3) %pa, align 4
  %fh = uitofp i32 %w to half
  %ph = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 %w
  store half %fh, ptr addrspace(3) %ph, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %a, i32 1, i32 0, i32 1, i1 true)
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 63, %t
  %q = getelementptr inbounds i32, ptr addrspace(3) %a, i32 %r
  %x = load i32, ptr addrspace(3) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %o, align 4
  %qh = getelementptr inbounds [64 x half], ptr addrspace(3) @h, i32 0, i32 %r
  %y = load half, ptr addrspace(3) %qh, align 2
  %yo = fptoui half %y to i32
  %t64 = add i32 %t, 64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t64
  store i32 %yo, ptr addrspace(1) %o2, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!air.version = !{!8}
!air.language_version = !{!9}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!0 = !{ptr @tg_composed, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// No threadgroup access is contained through `from_queries` ([`COMPOSED`]): its composed arena's
/// loads, stores and atomic, and its typed threadgroup object's accesses, carry no check; word 0
/// is the 64 threads' atomic adds.
#[test]
fn no_threadgroup_access_is_contained_through_from_queries() {
    let Some(bc) = assemble("tg_composed", COMPOSED) else {
        return;
    };
    let mut want: Vec<u32> = (0..64u32).map(|t| 63 - t).collect();
    want[63] += 64;
    want.extend((0..64u32).map(|t| 63 - t));
    let length = [(SpecKey::TgLength(0), 256)];
    for stem in PROFILES {
        let prof = queried(stem);
        let what = format!("tg_composed on {stem}");
        let a = compile(&bc, "tg_composed", &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let (plan, m) = typed_module(&bc, "tg_composed", &prof).expect("module");
        let tg = plan.families.access.local.tg.as_ref().expect("an arena");
        assert_eq!(tg.words, TgWords::Composed, "{what}");
        let census = plan.census();
        assert!(!contains_threadgroup(&census), "{what}: {census:?}");
        assert_eq!(
            run_both(&m, &resources_of(&plan, &length, 128)),
            want,
            "{what}"
        );
    }
}

/// The array is Metal's allocation: through `from_queries`, a region filling `maxComputeSharedMemorySize` is planned and one 16 bytes longer is refused
/// (`ThreadgroupMemorySize`). A static arena two words short of the limit is an array of its own
/// words.
#[test]
fn the_float_array_is_metals_allocation() {
    let Some(bc) = assemble("tg_region", &region_kernel()) else {
        return;
    };
    for stem in PROFILES {
        let dev = queried_device(stem);
        let what = stem;
        let lib = Library::from_air(&bc).expect("library");
        let f = lib.function("tg_region").expect("function");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)));
        let pipe = dev
            .compile_compute_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(
            pipe.metal_info.static_threadgroup_memory_length, 0,
            "{what}"
        );
        let state = |len: u32| {
            let mut t = StageTable::default();
            t.buffers.insert(
                0,
                BoundBuffer::Buffer {
                    length: 192 * 4,
                    offset: 0,
                    gpu_address: 0,
                    device_address: None,
                    attribute_stride: None,
                },
            );
            t.threadgroup_lengths.insert(0, len);
            let mut s = BindState::default();
            s.stages.insert(Stage::Kernel, t);
            s
        };
        let max = dev.profile().fields().limits.max_compute_shared_memory_size;
        assert!(
            pipe.plan(&dev, &state(max).shape()).is_ok(),
            "{what}: exactly full"
        );
        match pipe.plan(&dev, &state(max + 16).shape()) {
            Err(BindError::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::ThreadgroupMemorySize, "{what}");
            }
            other => panic!("{what}: {other:?}"),
        }
        let prof = queried(stem);
        let words = prof.fields().limits.max_compute_shared_memory_size / 4;
        let what = format!("a static arena of {} words on {stem}", words - 2);
        let Some(big) = assemble("tg_static", &static_kernel(words - 2)) else {
            return;
        };
        let a = compile(&big, "tg_static", &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
        link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let s = shape(&a.module.words);
        assert_eq!(s.float_arrays, vec![Some(32)], "{what}: {s:?}");
        assert_eq!(s.lengths, vec![words - 2], "{what}: {s:?}");
    }
}

/// An arena filling the workgroup limit answers every word in range: a `[[threadgroup(0)]]`
/// region of the limit's bytes, and a static arena two words short of it, every word stored and
/// loaded back through `from_queries` ([`FILL`]), as a scalar and as a `float4`.
#[test]
fn an_arena_filling_the_limit_answers_every_word_in_range() {
    for stem in PROFILES {
        let prof = queried(stem);
        let words = prof.fields().limits.max_compute_shared_memory_size / 4;
        for (n, region) in [(words, true), (words - 2, false)] {
            let entry = if region { "tg_region" } else { "tg_static" };
            let what = format!("{entry} of {n} words on {stem}");
            let Some(bc) = assemble(entry, &fill_kernel(n, region)) else {
                return;
            };
            let a = compile(&bc, entry, &prof).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let s = shape(&a.module.words);
            assert_eq!(s.float_arrays, vec![Some(32)], "{what}: {s:?}");
            if !region {
                assert_eq!(s.lengths, vec![n], "{what}: {s:?}");
            }
            let (plan, m) = typed_module(&bc, entry, &prof).expect("module");
            let length = [(SpecKey::TgLength(0), u64::from(4 * n))];
            let spec: &[(SpecKey, u64)] = if region { &length } else { &[] };
            let want: Vec<u32> = (0..n)
                .map(|k| n - 1 - k)
                .chain((0..n / 4).map(|j| 16 * j + 6))
                .collect();
            let got = run_both(&m, &resources_of(&plan, spec, want.len()));
            let first = got.iter().zip(&want).position(|(got, want)| got != want);
            assert_eq!(first, None, "{what}: the first word that differs");
        }
    }
}
