//! Typed objects' access chains and their containment (`memory::alloca`'s clamps,
//! `lower::access::local`).
//!
//! - A static `float` threadgroup array is a typed `Workgroup` object, and every in-range element
//!   moves as its bits whichever way the pointer is formed: through a `getelementptr` that indexes
//!   each level itself (its own operands are the chain's indices), a pointer to a row indexed
//!   again, a `float *` carried across the rows (a flattened index), or the pointer's own index
//!   over the rows. The first and the last line up; the other two recover their indices from the
//!   offset, and every enclosing array level is recovered with them.
//! - An out-of-range dynamic index, from a buffer (past the end, and -1), on a load and a store of
//!   a threadgroup object and a load of a constant object, reaches its level's last element: a load
//!   reads it and a store writes it. On a typed alloca the access is checked: the load reads zero
//!   and the store is dropped. The neighbouring object is untouched, and an in-range index on each
//!   kind reads every element as stored.
//! - The tiled matrix product of a kernel's two `float[16][16]` threadgroup tiles, every access
//!   lined up, answers the CPU's fused multiply-adds in the kernel's order bit for bit.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::access::local::{LocalStrategy, TypedAccess};
use crate::lower::access::AccessStrategy;
use crate::lower::select::Plan;
use crate::lower::strategy::Strategy;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

const PROFILES: [&str; 3] = ["moltenvk-m3", "vk12-floor", "lavapipe"];

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
        "m2v-typed-clamp-{}-{}-{name}",
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

/// The metadata of kernel `name(out, inp, t)`: two read-write `uint` buffers and the thread's
/// index in its threadgroup.
fn kernel_md(name: &str) -> String {
    format!(
        r#"declare void @air.wg.barrier(i32, i32)

!air.kernel = !{{!0}}
!air.version = !{{!90}}
!air.language_version = !{{!91}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"inp"}}
!5 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!90 = !{{i32 2, i32 8, i32 0}}
!91 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// `threads` threads along x (`local` the workgroup's shape), and buffer `i` holding `bufs[i]`.
fn resources(plan: &Plan, local: (u64, u64), bufs: &[Vec<u32>]) -> Resources {
    let mut s = BTreeMap::new();
    s.insert(SpecKey::LocalSize(Axis::X), local.0);
    s.insert(SpecKey::LocalSize(Axis::Y), local.1);
    s.insert(SpecKey::LocalSize(Axis::Z), 1);
    s.insert(SpecKey::FlatLocalSize, local.0 * local.1);
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
                slot: Some(slot(i as u32)),
                address: None,
                bytes: b.iter().flat_map(|w| w.to_le_bytes()).collect(),
            })
            .collect(),
        push,
        spec: s,
    }
}

/// Buffer `index`'s words after a run in both scheduling orders (which must agree).
fn run_both(m: &TypedModule, res: &Resources, index: usize) -> Vec<u32> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            subgroup: SubgroupSize::W32,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run");
        outs.push(
            out.buffers[index]
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

fn l3(bc: &[u8], entry: &str, prof: &DeviceProfile, what: &str) {
    let a = compile(bc, entry, prof).unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, prof).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
}

/// The plan's typed loads and stores.
fn typed_accesses(plan: &Plan) -> Vec<(&'static str, &TypedAccess)> {
    plan.strategy
        .iter()
        .flatten()
        .filter_map(|s| match s {
            Strategy::Access(AccessStrategy::Local(LocalStrategy::TypedLoad(a))) => {
                Some(("load", a))
            }
            Strategy::Access(AccessStrategy::Local(LocalStrategy::TypedStore(a))) => {
                Some(("store", a))
            }
            _ => None,
        })
        .collect()
}

/// The words thread `t` stores: spread over the 32 bits, NaN payloads, infinities, denormals and
/// zeros among them, so a move that is not of the bits shows.
fn word(t: u32) -> u32 {
    let w = t.wrapping_mul(0x9E37_79B9) ^ 0x7F80_0001;
    match t % 8 {
        0 => 0x7FC0_0000 | (t << 4),
        1 => 0xFF80_0000 | (t << 2) | 1,
        2 => t,
        3 => 0x8000_0000 | t,
        _ => w,
    }
}

/// `@a: [8 x [8 x float]]`: thread `t` (row `y = t / 8`, column `x = t % 8`) stores word `t` at
/// `a[y][x]`, and after a barrier reads `a[x][y]` through the pointer `read` forms (`%q`) into
/// `out[t]`.
fn transposed(name: &str, read: &str) -> String {
    format!(
        r#"@a = internal addrspace(3) global [8 x [8 x float]] undef, align 4

define void @{name}(ptr addrspace(1) %out, ptr addrspace(1) %inp, i32 %t) {{
  %y = lshr i32 %t, 3
  %x = and i32 %t, 7
  %yi = zext i32 %y to i64
  %xi = zext i32 %x to i64
  %ip = getelementptr inbounds i32, ptr addrspace(1) %inp, i32 %t
  %bits = load i32, ptr addrspace(1) %ip, align 4
  %v = bitcast i32 %bits to float
  %p = getelementptr inbounds [8 x [8 x float]], ptr addrspace(3) @a, i64 0, i64 %yi, i64 %xi
  store float %v, ptr addrspace(3) %p, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
{read}
  %r = load float, ptr addrspace(3) %q, align 4
  %rb = bitcast float %r to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %rb, ptr addrspace(1) %o, align 4
  ret void
}}

{}"#,
        kernel_md(name)
    )
}

/// Every in-range element of a `float[8][8]` threadgroup object moves as its bits whichever way
/// the read's pointer is formed; the indices line up where the GEP indexes each level itself
/// (its own levels, or the pointer's index over the rows), and are recovered from the offset for a
/// row pointer indexed again and for a flattened `float *` index that crosses the rows.
#[test]
fn every_in_range_element_moves_as_its_bits_however_the_pointer_is_formed() {
    // (name, the read's pointer `%q` to a[x][y], whether the read's indices line up)
    let reads = [
        (
            "lined",
            "  %q = getelementptr inbounds [8 x [8 x float]], ptr addrspace(3) @a, i64 0, i64 %xi, i64 %yi",
            true,
        ),
        (
            "rows",
            "  %q = getelementptr inbounds [8 x float], ptr addrspace(3) @a, i64 %xi, i64 %yi",
            true,
        ),
        (
            "row_pointer",
            "  %row = getelementptr inbounds [8 x [8 x float]], ptr addrspace(3) @a, i64 0, i64 %xi\n  \
             %q = getelementptr inbounds [8 x float], ptr addrspace(3) %row, i64 0, i64 %yi",
            false,
        ),
        (
            "flattened",
            "  %f = shl i32 %x, 3\n  %fi = add i32 %f, %y\n  %fz = zext i32 %fi to i64\n  \
             %q = getelementptr inbounds float, ptr addrspace(3) @a, i64 %fz",
            false,
        ),
    ];
    let inp: Vec<u32> = (0..64).map(word).collect();
    let want: Vec<u32> = (0..64u32).map(|t| word((t % 8) * 8 + t / 8)).collect();
    for (name, read, lined) in reads {
        let Some(bc) = assemble(name, &transposed(name, read)) else {
            return;
        };
        for stem in PROFILES {
            let prof = p(stem);
            let what = format!("{name} on {stem}");
            l3(&bc, name, &prof, &what);
            let (plan, m) = typed_module(&bc, name, &prof).expect("module");
            let local = &plan.families.access.local;
            assert_eq!(local.tg, None, "{what}: no arena");
            assert_eq!(local.typed.len(), 1, "{what}");
            let accesses = typed_accesses(&plan);
            assert_eq!(accesses.len(), 2, "{what}");
            for (op, a) in accesses {
                assert!(a.clamped, "{what}: {op}");
                assert_eq!(a.path.dynamic().count(), 2, "{what}: {op}: both levels");
                let lined_up = op == "store" || lined;
                assert_eq!(a.indices.is_some(), lined_up, "{what}: {op}");
            }
            let res = resources(&plan, (64, 1), &[vec![0; 64], inp.clone()]);
            assert_eq!(run_both(&m, &res, 0), want, "{what}");
        }
    }
}

/// The words out-of-range and in-range accesses answer in the containment kernels: `out[t]` the
/// out-of-range load, `out[64 + t]` element `t % 8` of the first object, `out[128 + t]` of the
/// second (`inp[0]` the store's index, `inp[1]` the load's).
fn contained_answers(stem: &str, name: &str, text: &str, objects: usize, writes: bool) -> Vec<u32> {
    let Some(bc) = assemble(name, text) else {
        return Vec::new();
    };
    let prof = p(stem);
    let what = format!("{name} on {stem}");
    l3(&bc, name, &prof, &what);
    let (plan, m) = typed_module(&bc, name, &prof).expect("module");
    let local = &plan.families.access.local;
    assert_eq!(local.typed.len(), objects, "{what}: typed objects");
    let accesses = typed_accesses(&plan);
    assert!(
        accesses.iter().any(|(op, _)| *op == "store") == writes,
        "{what}"
    );
    for (op, a) in &accesses {
        if a.path.dynamic().count() > 0 {
            assert!(a.clamped, "{what}: {op}");
        }
    }
    // Index -1 for the store and 1000 for the load: both past the end, one a huge unsigned.
    let res = resources(&plan, (64, 1), &[vec![0; 192], vec![u32::MAX, 1000]]);
    run_both(&m, &res, 0)
}

/// The tail every containment kernel shares: the out-of-range load of the first object (`%pa`
/// its base) at `inp[1]` into `out[t]`, and elements `t % 8` of `%pa` and `%pb` into
/// `out[64 + t]` and `out[128 + t]`.
const CONTAINED_TAIL: &str = r#"  %l1p = getelementptr inbounds i32, ptr addrspace(1) %inp, i32 1
  %l1 = load i32, ptr addrspace(1) %l1p, align 4
  %l1i = sext i32 %l1 to i64
  %pl = getelementptr inbounds [8 x float], ptr ADDR %pa, i64 0, i64 %l1i
  %lv = load float, ptr ADDR %pl, align 4
  %lb = bitcast float %lv to i32
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t
  store i32 %lb, ptr addrspace(1) %o0, align 4
  %ea = getelementptr inbounds [8 x float], ptr ADDR %pa, i64 0, i64 %ki
  %eav = load float, ptr ADDR %ea, align 4
  %eab = bitcast float %eav to i32
  %t64 = add i32 %t, 64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t64
  store i32 %eab, ptr addrspace(1) %o1, align 4
  %eb = getelementptr inbounds [8 x float], ptr ADDR %pb, i64 0, i64 %ki
  %ebv = load float, ptr ADDR %eb, align 4
  %ebb = bitcast float %ebv to i32
  %t128 = add i32 %t, 128
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %t128
  store i32 %ebb, ptr addrspace(1) %o2, align 4
  ret void
"#;

/// The first object's element `k` (a normal `float`'s bits), the second's, and the word the
/// out-of-range store writes.
const FIRST: u32 = 0x4100_0000;
const SECOND: u32 = 0x4200_0000;
const STORED: u32 = 0x4300_0000;

/// An out-of-range dynamic index never leaves its object. A threadgroup object's load and store,
/// and a constant object's load, reach the level's last element; a typed alloca's are checked, the
/// load reading zero and the store dropped, so the alloca's last element keeps its value. The
/// neighbouring object keeps its elements, and every in-range element reads back as stored.
#[test]
fn an_out_of_range_index_stays_inside_its_object() {
    // Threadgroup objects: threads 0..7 fill both, thread 0 stores out of range after a barrier.
    let threadgroup = format!(
        r#"@ga = internal addrspace(3) global [8 x float] undef, align 4
@gb = internal addrspace(3) global [8 x float] undef, align 4

define void @tg(ptr addrspace(1) %out, ptr addrspace(1) %inp, i32 %t) {{
entry:
  %pa = getelementptr inbounds [8 x float], ptr addrspace(3) @ga, i64 0, i64 0
  %pb = getelementptr inbounds [8 x float], ptr addrspace(3) @gb, i64 0, i64 0
  %k = and i32 %t, 7
  %ki = zext i32 %k to i64
  %wa = add i32 %k, FIRST
  %fa = bitcast i32 %wa to float
  %qa = getelementptr inbounds [8 x float], ptr addrspace(3) @ga, i64 0, i64 %ki
  store float %fa, ptr addrspace(3) %qa, align 4
  %wb = add i32 %k, SECOND
  %fb = bitcast i32 %wb to float
  %qb = getelementptr inbounds [8 x float], ptr addrspace(3) @gb, i64 0, i64 %ki
  store float %fb, ptr addrspace(3) %qb, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %first = icmp eq i32 %t, 0
  br i1 %first, label %st, label %join
st:
  %s0 = load i32, ptr addrspace(1) %inp, align 4
  %s0i = sext i32 %s0 to i64
  %ps = getelementptr inbounds [8 x float], ptr addrspace(3) @ga, i64 0, i64 %s0i
  store float bitcast (i32 STORED to float), ptr addrspace(3) %ps, align 4
  br label %join
join:
  tail call void @air.wg.barrier(i32 2, i32 1)
{tail}}}

{md}"#,
        tail = CONTAINED_TAIL.replace("ADDR", "addrspace(3)"),
        md = kernel_md("tg")
    )
    .replace("FIRST", &FIRST.to_string())
    .replace("SECOND", &SECOND.to_string())
    .replace("STORED", &STORED.to_string());
    // Two typed allocas of the thread's own, filled by constant-index stores.
    let mut fill = String::new();
    for k in 0..8 {
        fill.push_str(&format!(
            "  %ca{k} = getelementptr inbounds [8 x float], ptr %pa, i64 0, i64 {k}\n  \
             store float bitcast (i32 {} to float), ptr %ca{k}, align 4\n  \
             %cb{k} = getelementptr inbounds [8 x float], ptr %pb, i64 0, i64 {k}\n  \
             store float bitcast (i32 {} to float), ptr %cb{k}, align 4\n",
            FIRST + k,
            SECOND + k
        ));
    }
    let alloca = format!(
        r#"define void @al(ptr addrspace(1) %out, ptr addrspace(1) %inp, i32 %t) {{
entry:
  %pa = alloca [8 x float], align 4
  %pb = alloca [8 x float], align 4
  %k = and i32 %t, 7
  %ki = zext i32 %k to i64
{fill}  %s0 = load i32, ptr addrspace(1) %inp, align 4
  %s0i = sext i32 %s0 to i64
  %ps = getelementptr inbounds [8 x float], ptr %pa, i64 0, i64 %s0i
  store float bitcast (i32 STORED to float), ptr %ps, align 4
{tail}}}

{md}"#,
        tail = CONTAINED_TAIL.replace("ptr ADDR", "ptr"),
        md = kernel_md("al")
    )
    .replace("STORED", &STORED.to_string());
    // Two constant objects, filled as the others.
    let elems = |base: u32| {
        (0..8)
            .map(|k| format!("float bitcast (i32 {} to float)", base + k))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let constant = format!(
        r#"@ca = internal addrspace(2) constant [8 x float] [{a}], align 4
@cb = internal addrspace(2) constant [8 x float] [{b}], align 4

define void @co(ptr addrspace(1) %out, ptr addrspace(1) %inp, i32 %t) {{
entry:
  %pa = getelementptr inbounds [8 x float], ptr addrspace(2) @ca, i64 0, i64 0
  %pb = getelementptr inbounds [8 x float], ptr addrspace(2) @cb, i64 0, i64 0
  %k = and i32 %t, 7
  %ki = zext i32 %k to i64
{tail}}}

{md}"#,
        a = elems(FIRST),
        b = elems(SECOND),
        tail = CONTAINED_TAIL.replace("ADDR", "addrspace(2)"),
        md = kernel_md("co")
    );
    for stem in PROFILES {
        for (name, text, objects, writes, checked) in [
            ("tg", &threadgroup, 2, true, false),
            ("al", &alloca, 2, true, true),
            ("co", &constant, 2, false, false),
        ] {
            let got = contained_answers(stem, name, text, objects, writes);
            if got.is_empty() {
                return;
            }
            // A clamped out-of-range store wrote the first object's last element, which the
            // clamped out-of-range load reads; a checked store was dropped and a checked load read
            // zero. The second object is untouched.
            let last = if writes && !checked {
                STORED
            } else {
                FIRST + 7
            };
            let loaded = if checked { 0 } else { last };
            let mut want: Vec<u32> = (0..64).map(|_| loaded).collect();
            want.extend((0..64u32).map(|t| if t % 8 == 7 { last } else { FIRST + t % 8 }));
            want.extend((0..64u32).map(|t| SECOND + t % 8));
            assert_eq!(got, want, "{name} on {stem}");
        }
    }
}

/// The authored case of the public fixture `ob2_sized_array` ("loads and stores past a
/// thread-local array"): eight threads, each with its own `Frame { before[4], array[4], after[8] }`,
/// store `0xC0000000 | t` to `array[s]` and then load `array[l]`, `l` and `s` the case's signed
/// indices (`idx[t]` and `idx[8 + t]`), several past the array on either side. The array's words
/// after the store are Metal's: a store inside the array writes its element, and one past it
/// changes none of the array's words (Metal's lands past them). A load inside the array reads its
/// element; one past it reads zero, the value the case's checker allows there. No word before or
/// past the array changes, and each thread's last word is `0xF0000000 | t`.
#[test]
fn a_store_past_a_thread_local_array_changes_none_of_its_words() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/ob2_sized_array.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected ob2_sized_array: {why}"),
    };
    let loads: [i32; 8] = [0, 1, 2, 3, 4, 11, -1, -4];
    let stores: [i32; 8] = [2, 3, 4, 11, -1, -4, 7, 0];
    let idx: Vec<u32> = loads.iter().chain(&stores).map(|&i| i as u32).collect();
    let inside = |i: i32| usize::try_from(i).ok().filter(|&i| i < 4);
    let mut want = Vec::new();
    for t in 0..8u32 {
        let mut array: Vec<u32> = (0..4).map(|k| 0xA000_0000 | k).collect();
        if let Some(s) = inside(stores[t as usize]) {
            array[s] = 0xC000_0000 | t;
        }
        let loaded = inside(loads[t as usize]).map_or(0, |l| array[l]);
        want.push(loaded);
        want.extend(&array);
        want.extend([0, 0, 0xF000_0000 | t]);
    }
    for stem in ["amd-780m-radv", "lavapipe", "moltenvk-m3", "vk12-floor"] {
        let prof = p(stem);
        l3(&bc, "ob2_sized_array", &prof, stem);
        let (plan, m) = typed_module(&bc, "ob2_sized_array", &prof).expect("module");
        assert!(
            typed_accesses(&plan)
                .iter()
                .any(|(op, a)| *op == "store" && a.checked()),
            "{stem}: the store into the array is checked"
        );
        let res = resources(&plan, (8, 1), &[idx.clone(), vec![0xABAB_ABAB; 64]]);
        assert_eq!(run_both(&m, &res, 1), want, "{stem}");
    }
}

const MATMUL_N: usize = 32;

/// `C = A · B` over `N = 32`, one 16x16 threadgroup computing the block at row 0, column 0 through
/// two `float[16][16]` threadgroup tiles (m2vbench's `cs_matmul` with `N` a constant): each thread
/// loads one element of each tile per step, and after a barrier accumulates 16 fused
/// multiply-adds of its row of `As` and column of `Bs`.
fn matmul() -> String {
    r#"@As = internal addrspace(3) global [16 x [16 x float]] undef, align 4
@Bs = internal addrspace(3) global [16 x [16 x float]] undef, align 4
declare void @air.wg.barrier(i32, i32)
declare float @air.fma.f32(float, float, float)

define void @mm(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, <2 x i32> %lid) {
entry:
  %lx = extractelement <2 x i32> %lid, i64 0
  %ly = extractelement <2 x i32> %lid, i64 1
  %yi = zext i32 %ly to i64
  %xi = zext i32 %lx to i64
  %pa = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @As, i64 0, i64 %yi, i64 %xi
  %pb = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @Bs, i64 0, i64 %yi, i64 %xi
  %rown = mul i32 %ly, 32
  br label %tile
tile:
  %t = phi i32 [ 0, %entry ], [ %tn, %tile_end ]
  %acc0 = phi float [ 0.000000e+00, %entry ], [ %acck, %tile_end ]
  %ai = add i32 %rown, %t
  %ai2 = add i32 %ai, %lx
  %aiz = zext i32 %ai2 to i64
  %ap = getelementptr inbounds float, ptr addrspace(1) %A, i64 %aiz
  %av = load float, ptr addrspace(1) %ap, align 4
  store float %av, ptr addrspace(3) %pa, align 4
  %bi = add i32 %t, %ly
  %bi2 = mul i32 %bi, 32
  %bi3 = add i32 %bi2, %lx
  %biz = zext i32 %bi3 to i64
  %bp = getelementptr inbounds float, ptr addrspace(1) %B, i64 %biz
  %bv = load float, ptr addrspace(1) %bp, align 4
  store float %bv, ptr addrspace(3) %pb, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %inner
inner:
  %k = phi i32 [ 0, %tile ], [ %kn, %inner ]
  %acc = phi float [ %acc0, %tile ], [ %acck, %inner ]
  %kz = zext i32 %k to i64
  %qa = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @As, i64 0, i64 %yi, i64 %kz
  %x1 = load float, ptr addrspace(3) %qa, align 4
  %qb = getelementptr inbounds [16 x [16 x float]], ptr addrspace(3) @Bs, i64 0, i64 %kz, i64 %xi
  %x2 = load float, ptr addrspace(3) %qb, align 4
  %acck = tail call fast float @air.fma.f32(float %x1, float %x2, float %acc)
  %kn = add nuw nsw i32 %k, 1
  %kd = icmp eq i32 %kn, 16
  br i1 %kd, label %tile_end, label %inner
tile_end:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %tn = add i32 %t, 16
  %td = icmp ult i32 %tn, 32
  br i1 %td, label %tile, label %done
done:
  %ci = add i32 %rown, %lx
  %ciz = zext i32 %ci to i64
  %cp = getelementptr inbounds float, ptr addrspace(1) %C, i64 %ciz
  store float %acck, ptr addrspace(1) %cp, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!90}
!air.language_version = !{!91}
!0 = !{ptr @mm, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"B"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"C"}
!6 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint2", !"air.arg_name", !"lid"}
!90 = !{i32 2, i32 8, i32 0}
!91 = !{!"Metal", i32 4, i32 0, i32 0}
"#
    .to_string()
}

/// The tiled matrix product's tiles are typed objects whose every access lines up, and its block
/// is the CPU's fused multiply-adds in the kernel's order, bit for bit.
#[test]
fn a_tiled_matrix_product_through_typed_tiles_is_exact() {
    let Some(bc) = assemble("mm", &matmul()) else {
        return;
    };
    let n = MATMUL_N;
    // Values whose products round: the low mantissa bits vary.
    let value = |i: usize, salt: u32| {
        let w = (i as u32).wrapping_mul(0x9E37_79B9) ^ salt;
        f32::from_bits(0x3F00_0000 | (w >> 9)) - 0.75
    };
    let a: Vec<f32> = (0..n * n).map(|i| value(i, 0x1234_5678)).collect();
    let b: Vec<f32> = (0..n * n).map(|i| value(i, 0x0BAD_F00D)).collect();
    let mut want = vec![0u32; n * n];
    for r in 0..16 {
        for c in 0..16 {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc = a[r * n + k].mul_add(b[k * n + c], acc);
            }
            want[r * n + c] = acc.to_bits();
        }
    }
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    for stem in PROFILES {
        let prof = p(stem);
        l3(&bc, "mm", &prof, stem);
        let (plan, m) = typed_module(&bc, "mm", &prof).expect("module");
        let local = &plan.families.access.local;
        assert_eq!(local.tg, None, "{stem}: no arena");
        assert_eq!(local.typed.len(), 2, "{stem}");
        let accesses = typed_accesses(&plan);
        assert_eq!(accesses.len(), 4, "{stem}");
        for (op, a) in accesses {
            assert!(a.indices.is_some(), "{stem}: every {op} lines up");
        }
        let res = resources(&plan, (16, 16), &[bits(&a), bits(&b), vec![0; n * n]]);
        assert_eq!(run_both(&m, &res, 2), want, "{stem}");
    }
}
