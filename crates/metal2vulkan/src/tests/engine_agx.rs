//! The AGX lane intrinsics end to end.
//!
//! MSL has no spelling for `llvm.agx*`, so the kernels are hand-written AIR (`llvm-as`
//! assembles them; a missing tool skips visibly). Each runs on the CPU interpreter against
//! expectations computed here from the device-measured contracts the old engine recorded:
//!
//! - `llvm.agx3.edgecheck` over signed and unsigned edge triples (negative lanes, `INT_MAX`
//!   overflow, a nonzero `low`, count 0 and `u32::MAX`);
//! - the emask loads and stores in both spaces and every element width, under every mask:
//!   an inactive lane is never written and reads zero;
//! - `llvm.agx2.cluster.num` is 0 in every thread;
//! - the four `agx2` matrix multiply-adds over the distributed 8x8 tile, in the `Direct` tier
//!   at 32 lanes and the `Cluster32` tier at 64 lanes, in both scheduling orders, with several
//!   simdgroups per threadgroup.
//!
//! On every committed profile each kernel compiles, links, and passes L3, or (the
//! multiply-adds only) is refused exactly as its tier is. Unmodeled AGX operations and a shape
//! mask the backend would reject are refused as `AgxUnmodeled`; `llvm.agx3.yield`, a delay with no
//! data effect, compiles to the kernel without it. No GPU is used.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::subgroup::{self, OpBits, SimdClass, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature, LowerFamily, Needs};
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::lower::select::Plan;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::spirv::testing::threadgroup_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-agx-{}-{}-{name}",
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

/// A kernel `name` over device buffers `%b0..` (`[[buffer(i)]]`) and `%t`, the thread index in
/// the threadgroup; `globals` and `decls` go before and after the function.
fn kernel(name: &str, buffers: u32, globals: &str, body: &str, decls: &str) -> String {
    let mut params = String::new();
    let mut md = String::new();
    let mut list = Vec::new();
    for i in 0..buffers {
        params.push_str(&format!(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}, "
        ));
        md.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b{i}\"}}\n",
            20 + i
        ));
        list.push(format!("!{}", 20 + i));
    }
    params.push_str("i32 noundef %t");
    md.push_str(&format!(
        "!{} = !{{i32 {buffers}, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n",
        20 + buffers
    ));
    list.push(format!("!{}", 20 + buffers));
    format!(
        "{HEADER}{globals}define void @{name}({params}) {{\nentry:\n{body}  ret void\n}}\n\n{decls}\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{md}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        list.join(", ")
    )
}

fn moltenvk() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

/// A wave64-only compute device: the committed RDNA3 description with its subgroup size range
/// narrowed to 64 (the `Cluster32` tier).
fn wave64() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/amd-rdna3-radv.json");
    let text = std::fs::read_to_string(&path).expect("read the RDNA3 description");
    let narrowed = text.replace("\"minSubgroupSize\": 32", "\"minSubgroupSize\": 64");
    assert_ne!(narrowed, text, "the description names minSubgroupSize");
    let desc = DeviceDescription::from_khronos_json(&narrowed, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Buffers `bufs` (bound at slots 0..) after one workgroup of `local` threads ran `m`, identical
/// in both scheduling orders.
fn run_all(
    plan: &Plan,
    m: &TypedModule,
    local: u32,
    lanes: SubgroupSize,
    bufs: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([local, 1, 1]) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(local));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let res = Resources {
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
                bytes: b.clone(),
            })
            .collect(),
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone()).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_of(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bit(m: u32, i: u32) -> bool {
    (m >> i) & 1 == 1
}

// ---- edgecheck -------------------------------------------------------------------------------

const EDGE_DECL: &str = "declare i16 @llvm.agx3.edgecheck(i32, i32, i32)\n";

fn edge_kernel() -> String {
    let body =
        "  %i32 = mul i32 %t, 3\n  %i = zext i32 %i32 to i64\n  %t64 = zext i32 %t to i64\n  \
                %p0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %i\n  \
                %base = load i32, ptr addrspace(1) %p0, align 4\n  \
                %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1\n  \
                %low = load i32, ptr addrspace(1) %p1, align 4\n  \
                %p2 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 2\n  \
                %count = load i32, ptr addrspace(1) %p2, align 4\n  \
                %m = tail call i16 @llvm.agx3.edgecheck(i32 %base, i32 %low, i32 %count)\n  \
                %z = zext i16 %m to i32\n  \
                %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %t64\n  \
                store i32 %z, ptr addrspace(1) %q, align 4\n";
    kernel("edge", 2, "", body, EDGE_DECL)
}

/// The device-measured contract: bit `i` when `base + i >= 0` (signed) and
/// `(unsigned)(base + i - low) < count`.
fn edgecheck(base: i32, low: i32, count: u32) -> u32 {
    (0..4i32)
        .filter(|&i| {
            i64::from(base) + i64::from(i) >= 0
                && (base.wrapping_add(i).wrapping_sub(low) as u32) < count
        })
        .fold(0u32, |a, i| a | (1 << i))
}

fn edge_triples() -> Vec<(i32, i32, u32)> {
    let bases = [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        5,
        i32::MAX - 2,
        i32::MAX,
        i32::MIN,
        i32::MIN + 2,
    ];
    let lows = [0, 1, 3, -2];
    let counts = [0, 1, 3, 4, 5, u32::MAX];
    let mut out = Vec::new();
    for b in bases {
        for l in lows {
            for c in counts {
                out.push((b, l, c));
            }
        }
    }
    out
}

/// `edgecheck` matches the device-measured contract on every triple, including the negative-lane
/// rule and a lane index that overflows `i32`.
#[test]
fn edgecheck_matches_the_measured_contract() {
    let Some(bc) = assemble_text("edge", &edge_kernel()) else {
        return;
    };
    let p = moltenvk();
    let triples = edge_triples();
    let n = triples.len() as u32;
    let input: Vec<u32> = triples
        .iter()
        .flat_map(|&(b, l, c)| [b as u32, l as u32, c])
        .collect();
    let (plan, m) = typed_module(&bc, "edge", &p).expect("compile");
    let out = run_all(
        &plan,
        &m,
        n,
        SubgroupSize::W32,
        &[bytes_of(&input), vec![0; n as usize * 4]],
    );
    let got = words(&out[1]);
    for (t, &(b, l, c)) in triples.iter().enumerate() {
        assert_eq!(got[t], edgecheck(b, l, c), "edgecheck({b}, {l}, {c})");
    }
    // The measured examples of the old engine's note.
    assert_eq!(edgecheck(0, 1, 3), 0b1110);
    assert_eq!(edgecheck(-2, 0, 4), 0b1100);
}

// ---- emask accesses --------------------------------------------------------------------------

const SENTINEL: u8 = 0xA5;

/// A global emask kernel on `<lanes x iN>`: lane-masked load of `in` by `t`'s low mask bits,
/// then lane-masked store into `out` by the next mask bits.
fn emask_global_kernel(name: &str, lanes: u32, bits: u32) -> String {
    let ty = format!("<{lanes} x i{bits}>");
    let tag = format!("v{lanes}i{bits}");
    let shape = (1u32 << lanes) - 1;
    let size = bits / 8;
    let body = format!(
        "  %o32 = mul i32 %t, {lanes}\n  %o = zext i32 %o32 to i64\n  \
         %p = getelementptr inbounds i{bits}, ptr addrspace(1) %b0, i64 %o\n  \
         %t16 = trunc i32 %t to i16\n  %lm = and i16 %t16, {shape}\n  \
         %v = tail call {ty} @llvm.agx3.load.with.emask.global.{tag}(ptr addrspace(1) %p, i16 %lm, i16 {shape}, i16 {size})\n  \
         %s0 = lshr i16 %t16, {lanes}\n  %sm = and i16 %s0, {shape}\n  \
         %q = getelementptr inbounds i{bits}, ptr addrspace(1) %b1, i64 %o\n  \
         tail call void @llvm.agx3.store.with.emask.global.{tag}(ptr addrspace(1) %q, {ty} %v, i16 %sm, i16 {shape}, i16 {size})\n"
    );
    let decls = format!(
        "declare {ty} @llvm.agx3.load.with.emask.global.{tag}(ptr addrspace(1), i16, i16, i16)\n\
         declare void @llvm.agx3.store.with.emask.global.{tag}(ptr addrspace(1), {ty}, i16, i16, i16)\n"
    );
    kernel(name, 2, "", &body, &decls)
}

/// A threadgroup emask kernel on `<4 x i8>`: the thread's four bytes of threadgroup memory are
/// filled with the sentinel, lane-masked stored from `in`, lane-masked loaded, and written out.
fn emask_local_kernel() -> String {
    let body = "  %o32 = shl i32 %t, 2\n  %o = zext i32 %o32 to i64\n  \
                %g = getelementptr inbounds i8, ptr addrspace(3) @tg, i64 %o\n  \
                store i32 -1515870811, ptr addrspace(3) %g, align 4\n  \
                %pi = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 %o\n  \
                %v = load <4 x i8>, ptr addrspace(1) %pi, align 4\n  \
                %t16 = trunc i32 %t to i16\n  %sm = and i16 %t16, 15\n  \
                tail call void @llvm.agx3.store.with.emask.local.v4i8(ptr addrspace(3) %g, <4 x i8> %v, i16 %sm, i16 15, i16 1)\n  \
                %l0 = lshr i16 %t16, 4\n  %lm = and i16 %l0, 15\n  \
                %r = tail call <4 x i8> @llvm.agx3.load.with.emask.local.v4i8(ptr addrspace(3) %g, i16 %lm, i16 15, i16 1)\n  \
                %po = getelementptr inbounds i8, ptr addrspace(1) %b1, i64 %o\n  \
                store <4 x i8> %r, ptr addrspace(1) %po, align 4\n";
    let globals = "@tg = internal unnamed_addr addrspace(3) global [1024 x i8] undef, align 4\n\n";
    let decls = "declare <4 x i8> @llvm.agx3.load.with.emask.local.v4i8(ptr addrspace(3), i16, i16, i16)\n\
                 declare void @llvm.agx3.store.with.emask.local.v4i8(ptr addrspace(3), <4 x i8>, i16, i16, i16)\n";
    kernel("emask_local", 2, globals, body, decls)
}

/// Distinct nonzero input bytes.
/// The plan's lane-masked accesses (`agx.emask*`) and their uses, each named by what its
/// endpoint reaches.
fn emasks(plan: &Plan) -> Vec<(String, u32)> {
    plan.census()
        .into_iter()
        .filter(|(name, _)| name.starts_with("agx.emask"))
        .map(|(name, (uses, _))| (name, uses))
        .collect()
}

fn input_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11) | 1)
        .collect()
}

/// Global emask loads and stores of every element width and lane count, under every combination of
/// load and store masks: an inactive loaded lane reads zero, an inactive stored lane leaves memory
/// unchanged.
#[test]
fn global_emask_accesses_touch_exactly_their_active_lanes() {
    let p = moltenvk();
    for (lanes, bits) in [(4u32, 32u32), (2, 32), (4, 16), (2, 16), (4, 8), (2, 8)] {
        let name = format!("emask_v{lanes}i{bits}");
        let Some(bc) = assemble_text(&name, &emask_global_kernel(&name, lanes, bits)) else {
            return;
        };
        let (plan, m) = typed_module(&bc, &name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            emasks(&plan),
            [("agx.emask_load", 1), ("agx.emask_store", 1)].map(|(n, u)| (n.to_owned(), u)),
            "{name}"
        );
        let threads = 1u32 << (2 * lanes);
        let size = (bits / 8) as usize;
        let len = threads as usize * lanes as usize * size;
        // Whole words: the views are word arrays.
        let padded = len.div_ceil(4) * 4;
        let mut input = input_bytes(padded);
        input.truncate(padded);
        let out = run_all(
            &plan,
            &m,
            threads,
            SubgroupSize::W32,
            &[input.clone(), vec![SENTINEL; padded]],
        );
        let shape = (1u32 << lanes) - 1;
        for t in 0..threads {
            let (lm, sm) = (t & shape, (t >> lanes) & shape);
            for i in 0..lanes {
                let at = (t * lanes + i) as usize * size;
                for b in 0..size {
                    let want = if !bit(sm, i) {
                        SENTINEL
                    } else if bit(lm, i) {
                        input[at + b]
                    } else {
                        0
                    };
                    assert_eq!(out[1][at + b], want, "{name} thread {t} lane {i} byte {b}");
                }
            }
        }
    }
}

/// Threadgroup-space emask accesses behave the same, with sub-word lanes composed into threadgroup
/// words.
#[test]
fn local_emask_accesses_touch_exactly_their_active_lanes() {
    let Some(bc) = assemble_text("emask_local", &emask_local_kernel()) else {
        return;
    };
    let p = moltenvk();
    let (plan, m) = typed_module(&bc, "emask_local", &p).expect("compile");
    // Threadgroup memory is an arena, checked in the shader whatever the buffer encoding: named
    // apart from the descriptor views of the global accesses.
    assert_eq!(
        emasks(&plan),
        [("agx.emask_load.arena", 1), ("agx.emask_store.arena", 1)].map(|(n, u)| (n.to_owned(), u))
    );
    let input = input_bytes(1024);
    let out = run_all(
        &plan,
        &m,
        256,
        SubgroupSize::W32,
        &[input.clone(), vec![0; 1024]],
    );
    for t in 0..256u32 {
        let (sm, lm) = (t & 15, (t >> 4) & 15);
        for i in 0..4 {
            let at = (4 * t + i) as usize;
            let want = if !bit(lm, i) {
                0
            } else if bit(sm, i) {
                input[at]
            } else {
                SENTINEL
            };
            assert_eq!(out[1][at], want, "thread {t} lane {i}");
        }
    }
}

/// An emask load on one side of a divergent branch, merged by a phi: the per-lane selections nest
/// inside the structurized branch (the old engine's block-split concern).
#[test]
fn a_divergent_emask_load_merges_through_a_phi() {
    let body = "  %o32 = shl i32 %t, 2\n  %o = zext i32 %o32 to i64\n  \
                %odd = and i32 %t, 1\n  %c = icmp ne i32 %odd, 0\n  br i1 %c, label %yes, label %no\n\
                yes:\n  %p = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %o\n  \
                %t16 = trunc i32 %t to i16\n  %m = lshr i16 %t16, 1\n  %mm = and i16 %m, 15\n  \
                %v = tail call <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(ptr addrspace(1) %p, i16 %mm, i16 15, i16 4)\n  \
                br label %join\n\
                no:\n  br label %join\n\
                join:\n  %r = phi <4 x i32> [ %v, %yes ], [ <i32 7, i32 7, i32 7, i32 7>, %no ]\n  \
                %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %o\n  \
                store <4 x i32> %r, ptr addrspace(1) %q, align 4\n";
    let text = kernel(
        "emask_branch",
        2,
        "",
        body,
        "declare <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(ptr addrspace(1), i16, i16, i16)\n",
    );
    let Some(bc) = assemble_text("emask_branch", &text) else {
        return;
    };
    let p = moltenvk();
    let a = compile(&bc, "emask_branch", &p).expect("compile");
    link_and_validate(&a, &p).expect("valid");
    let (plan, m) = typed_module(&bc, "emask_branch", &p).expect("compile");
    let input: Vec<u32> = (0..256).map(|i| 5000 + i).collect();
    let out = run_all(
        &plan,
        &m,
        64,
        SubgroupSize::W32,
        &[bytes_of(&input), vec![0; 1024]],
    );
    let got = words(&out[1]);
    for t in 0..64u32 {
        for i in 0..4 {
            let at = (4 * t + i) as usize;
            let want = if t & 1 == 0 {
                7
            } else if bit((t >> 1) & 15, i) {
                input[at]
            } else {
                0
            };
            assert_eq!(got[at], want, "thread {t} lane {i}");
        }
    }
}

/// The public fixture `kernel_agx_emask_edge_lanes` (an `edgecheck` mask over a count read from a
/// `constant` buffer, driving an emask load and an emask store) on the interpreter: with `base =
/// gid - 2` and `count = 5`, lane `i` of thread `g` is active when `0 <= g - 2 + i < 5`.
#[test]
fn the_public_edge_lanes_fixture_runs() {
    let path = fixtures_dir().join("kernel_agx_emask_edge_lanes.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let p = moltenvk();
    let (plan, m) = typed_module(&bc, "agx_emask_edge_lanes", &p).expect("compile");
    let input: Vec<u32> = (0..64).map(|i| 1000 + i).collect();
    let sentinel = 0xDEAD_BEEF;
    let out = run_all(
        &plan,
        &m,
        8,
        SubgroupSize::W32,
        &[
            bytes_of(&input),
            bytes_of(&[sentinel; 64]),
            bytes_of(&[5, 0, 0, 0]),
        ],
    );
    let got = words(&out[1]);
    for g in 0..8u32 {
        for i in 0..4 {
            let lane = g as i32 - 2 + i as i32;
            let active = (0..5).contains(&lane);
            let a = (4 * g + i) as usize;
            assert_eq!(got[a], if active { input[a] } else { 0 }, "load, {g} {i}");
            let b = a + 32;
            assert_eq!(
                got[b],
                if active { input[b] } else { sentinel },
                "store, {g} {i}"
            );
        }
    }
}

// ---- cluster.num -----------------------------------------------------------------------------

/// The cluster number is 0 in every thread (uniform, a value the hardware gives).
#[test]
fn cluster_number_is_zero_everywhere() {
    let body = "  %c = tail call i32 @llvm.agx2.cluster.num()\n  %t64 = zext i32 %t to i64\n  \
                %q = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %t64\n  \
                store i32 %c, ptr addrspace(1) %q, align 4\n";
    let text = kernel(
        "cluster",
        1,
        "",
        body,
        "declare i32 @llvm.agx2.cluster.num()\n",
    );
    let Some(bc) = assemble_text("cluster", &text) else {
        return;
    };
    let p = moltenvk();
    let (plan, m) = typed_module(&bc, "cluster", &p).expect("compile");
    assert_eq!(plan.simd.1, Tier::NoSubgroup, "no subgroup operation");
    let out = run_all(&plan, &m, 64, SubgroupSize::W32, &[vec![0xFF; 256]]);
    assert!(words(&out[0]).iter().all(|&w| w == 0));
}

// ---- matrix multiply-adds --------------------------------------------------------------------

/// The four `agx2` multiply-adds: (name, `half`, dimension).
const MATMADS: [(&str, bool, u32); 4] = [
    ("f32matmad8x8", false, 8),
    ("f32matmad4x4", false, 4),
    ("f16matmad8x8", true, 8),
    ("f16matmad4x4", true, 4),
];

fn matmad_kernel(op: &str, half: bool) -> String {
    let (s, v, align) = if half {
        ("half", "<2 x half>", 4)
    } else {
        ("float", "<2 x float>", 8)
    };
    let suffix = if half { "v2f16" } else { "v2f32" };
    let body = format!(
        "  %o32 = shl i32 %t, 1\n  %o = zext i32 %o32 to i64\n  \
         %pa = getelementptr inbounds {s}, ptr addrspace(1) %b0, i64 %o\n  %a = load {v}, ptr addrspace(1) %pa, align {align}\n  \
         %pb = getelementptr inbounds {s}, ptr addrspace(1) %b1, i64 %o\n  %b = load {v}, ptr addrspace(1) %pb, align {align}\n  \
         %pc = getelementptr inbounds {s}, ptr addrspace(1) %b2, i64 %o\n  %c = load {v}, ptr addrspace(1) %pc, align {align}\n  \
         %r = tail call {v} @llvm.agx2.{op}.{suffix}({v} %a, {v} %b, {v} %c)\n  \
         %pr = getelementptr inbounds {s}, ptr addrspace(1) %b3, i64 %o\n  store {v} %r, ptr addrspace(1) %pr, align {align}\n"
    );
    let decls = format!("declare {v} @llvm.agx2.{op}.{suffix}({v}, {v}, {v})\n");
    kernel(op, 4, "", &body, &decls)
}

/// The device-measured storage mapping: lane `l` owns row `row(l)` and columns `col(l)`,
/// `col(l) + 1`.
fn row(l: u32) -> u32 {
    ((l >> 1) & 3) | ((l >> 4) << 2)
}

fn col(l: u32) -> u32 {
    ((l >> 3) & 1) * 4 + (l & 1) * 2
}

/// The expected cells of every thread: per simdgroup, `c + a·b` over the tile (8x8) or over
/// each quadrant (4x4), accumulated with fused multiply-adds in increasing `k`.
fn matmad_expected(a: &[f32], b: &[f32], c: &[f32], threads: u32, dim: u32) -> Vec<f32> {
    let mut out = vec![0.0; threads as usize * 2];
    for s in 0..threads / 32 {
        let mut ta = [[0f32; 8]; 8];
        let mut tb = [[0f32; 8]; 8];
        let mut tc = [[0f32; 8]; 8];
        for l in 0..32 {
            for j in 0..2 {
                let at = ((32 * s + l) * 2 + j) as usize;
                let (r, q) = (row(l) as usize, (col(l) + j) as usize);
                ta[r][q] = a[at];
                tb[r][q] = b[at];
                tc[r][q] = c[at];
            }
        }
        for l in 0..32 {
            for j in 0..2 {
                let (r, q) = (row(l), col(l) + j);
                let (r0, q0) = if dim == 4 { (r & 4, q & 4) } else { (0, 0) };
                let mut acc = tc[r as usize][q as usize];
                for k in 0..dim {
                    acc = ta[r as usize][(q0 + k) as usize]
                        .mul_add(tb[(r0 + k) as usize][q as usize], acc);
                }
                out[((32 * s + l) * 2 + j) as usize] = acc;
            }
        }
    }
    out
}

/// Small integers: every product and sum is exact in `half` and `float`, so the expected cells
/// do not depend on the accumulation's rounding.
fn operand(threads: u32, seed: u32) -> Vec<f32> {
    (0..threads * 2)
        .map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) % 9) as f32 - 4.0)
        .collect()
}

fn encode(v: &[f32], half: bool) -> Vec<u8> {
    if half {
        v.iter()
            .flat_map(|&x| f64_to_f16(f64::from(x)).to_le_bytes())
            .collect()
    } else {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

fn decode(b: &[u8], half: bool) -> Vec<f32> {
    if half {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32)
            .collect()
    } else {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn run_matmads(p: &DeviceProfile, tier: Tier, lanes: SubgroupSize, locals: &[u32]) {
    for (op, half, dim) in MATMADS {
        let Some(bc) = assemble_text(op, &matmad_kernel(op, half)) else {
            return;
        };
        let (plan, m) = typed_module(&bc, op, p).unwrap_or_else(|e| panic!("{op}: {e}"));
        assert_eq!(plan.simd.1, tier, "{op}");
        assert_eq!(
            plan.simd.0,
            SimdClass::CrossLane(OpBits::SHUFFLE),
            "{op}: the class names the shuffles"
        );
        for &n in locals {
            let (a, b, c) = (operand(n, 1), operand(n, 2), operand(n, 3));
            let out = run_all(
                &plan,
                &m,
                n,
                lanes,
                &[
                    encode(&a, half),
                    encode(&b, half),
                    encode(&c, half),
                    vec![0; n as usize * if half { 4 } else { 8 }],
                ],
            );
            let got = decode(&out[3], half);
            let want = matmad_expected(&a, &b, &c, n, dim);
            assert_eq!(got, want, "{op} local {n}");
        }
    }
}

/// The multiply-adds in the `Direct` tier at 32 lanes, for one, two, and three simdgroups.
#[test]
fn matmads_run_direct_at_32_lanes() {
    run_matmads(&moltenvk(), Tier::Direct, SubgroupSize::W32, &[32, 64, 96]);
}

/// The multiply-adds in the `Cluster32` tier at 64 lanes: every shuffle stays in its own 32-lane
/// cluster.
#[test]
fn matmads_run_cluster32_at_64_lanes() {
    run_matmads(&wave64(), Tier::Cluster32, SubgroupSize::W64, &[64, 128]);
}

/// The multiply-adds on emulated simdgroups (`lavapipe` cannot pin 32 lanes) at 8 and 16 lanes: the
/// tile's cells are exchanged through workgroup memory.
#[test]
fn matmads_run_emulated_at_8_and_16_lanes() {
    let p =
        crate::pipeline::testing::profile("lavapipe", DevicePolicy::default()).expect("profile");
    for lanes in [SubgroupSize::W8, SubgroupSize::W16] {
        run_matmads(&p, Tier::Emulated, lanes, &[32, 64, 96]);
    }
}

/// The multiply-adds record the device's accumulation precision; `half` also its
/// rounding through `f32`.
#[test]
fn matmads_record_their_precision() {
    let p = moltenvk();
    for (op, half, _) in MATMADS {
        let Some(bc) = assemble_text(op, &matmad_kernel(op, half)) else {
            return;
        };
        let a = compile(&bc, op, &p).expect("compile");
        assert!(a.precision.contains(&PrecisionNote::ReductionOrder), "{op}");
        assert!(
            a.precision.contains(&PrecisionNote::DeviceArithmetic),
            "{op}"
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::F16FmaDoubleRounding),
            half,
            "{op}"
        );
    }
}

// ---- every profile ---------------------------------------------------------------------------

/// Every AGX kernel compiles, links, relocates, and passes L3 on every committed profile; a
/// multiply-add exactly where a cross-lane shuffle class has a compute tier (a subgroup tier, or
/// emulated simdgroups), and elsewhere it is refused as that tier is. The plan's size estimate
/// bounds the emitted words.
#[test]
fn agx_kernels_compile_or_refuse_with_the_tier() {
    let mut kernels = vec![
        ("edge".to_string(), edge_kernel(), false),
        ("emask_local".to_string(), emask_local_kernel(), false),
    ];
    for (lanes, bits) in [(4u32, 32u32), (2, 16), (4, 8)] {
        let name = format!("emask_v{lanes}i{bits}");
        let text = emask_global_kernel(&name, lanes, bits);
        kernels.push((name, text, false));
    }
    for (op, half, _) in MATMADS {
        kernels.push((op.to_string(), matmad_kernel(op, half), true));
    }
    let mut assembled = Vec::new();
    for (name, text, shuffles) in kernels {
        let Some(bc) = assemble_text(&name, &text) else {
            return;
        };
        assembled.push((name, bc, shuffles));
    }
    let (mut ok, mut refused, mut emulated) = (0, 0, 0);
    for (stem, p) in threadgroup_profiles().expect("profiles") {
        let tier = subgroup::tier(
            &SimdClass::CrossLane(OpBits::SHUFFLE),
            Stage::Kernel,
            &p.fields().subgroup,
        );
        for (name, bc, shuffles) in &assembled {
            match compile(bc, name, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
                    assert!(!shuffles || tier.is_ok(), "{stem} {name}");
                    if *shuffles && tier == Ok(Tier::Emulated) {
                        emulated += 1;
                    }
                    let (plan, _) = typed_module(bc, name, &p).expect("plan");
                    let words = a.module.words.len() as u64;
                    assert!(
                        plan.estimate.words >= words,
                        "{stem} {name}: estimate {} < emitted {words}",
                        plan.estimate.words
                    );
                    ok += 1;
                }
                Err(Error::Unsupported(u)) if *shuffles => {
                    let want = tier.clone().expect_err("a refused tier");
                    assert_eq!(u.feature, want.feature, "{stem} {name}");
                    refused += 1;
                }
                Err(e) => panic!("{stem} {name}: {e}"),
            }
        }
    }
    assert!(
        ok > 0 && emulated > 0,
        "{ok} compiled ({emulated} emulated), {refused} refused"
    );
}

/// AGX operations without a modeled contract, and an emask shape mask other than the vector's
/// (which the backend rejects), are refused as `AgxUnmodeled`. An unmodeled operation with a
/// pointer operand (a `wide` emask load) is refused one step earlier, by the memory analysis, which
/// knows no access contract for it.
#[test]
fn unmodeled_agx_operations_are_refused() {
    let permute = "  %r = tail call i32 @llvm.agx2.lane.permute(i32 %t)\n  \
                 store i32 %r, ptr addrspace(1) %b0, align 4\n";
    let permute = kernel(
        "permute",
        1,
        "",
        permute,
        "declare i32 @llvm.agx2.lane.permute(i32)\n",
    );
    let shape = "  %v = tail call <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(ptr addrspace(1) %b0, i16 15, i16 7, i16 4)\n  \
                 store <4 x i32> %v, ptr addrspace(1) %b0, align 4\n";
    let shape = kernel(
        "shape",
        1,
        "",
        shape,
        "declare <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(ptr addrspace(1), i16, i16, i16)\n",
    );
    let wide = "  %v = tail call <8 x i32> @llvm.agx3.wide.load.with.emask.global.v8i32(ptr addrspace(1) %b0, i16 255, i16 7, i16 4)\n  \
                %e = extractelement <8 x i32> %v, i32 0\n  store i32 %e, ptr addrspace(1) %b0, align 4\n";
    let wide = kernel(
        "wide",
        1,
        "",
        wide,
        "declare <8 x i32> @llvm.agx3.wide.load.with.emask.global.v8i32(ptr addrspace(1), i16, i16, i16)\n",
    );
    let p = moltenvk();
    for (name, text, feature) in [
        ("permute", permute, Feature::AgxUnmodeled),
        ("shape", shape, Feature::AgxUnmodeled),
        ("wide", wide, Feature::Lowering(LowerFamily::Access)),
    ] {
        let Some(bc) = assemble_text(name, &text) else {
            return;
        };
        match compile(&bc, name, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, feature, "{name}");
                assert_eq!(u.needs, Needs::NotImplemented, "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// `llvm.agx3.yield` delays the calling thread and changes no word (measured on the M3 Max and the
/// M4), so decode drops it. A kernel that yields with constant and run-time operands between its
/// arithmetic, stores and loads, and in a loop, compiles on every committed profile to exactly the
/// words of the same kernel without the calls, and runs on the CPU interpreter to the words the
/// arithmetic gives.
#[test]
fn a_yield_compiles_to_the_same_kernel_without_it() {
    let body = |yields: bool| {
        let call = |op: &str| {
            if yields {
                format!("  tail call void @llvm.agx3.yield(i16 noundef zeroext {op})\n")
            } else {
                String::new()
            }
        };
        format!(
            "  %v = mul i32 %t, -1640531535\n{}  %p = getelementptr inbounds i32, ptr addrspace(1) %b0, i32 %t\n  \
             store i32 %v, ptr addrspace(1) %p, align 4\n  %t16 = trunc i32 %t to i16\n{}  \
             %w = load i32, ptr addrspace(1) %p, align 4\n  %t32 = zext i16 %t16 to i32\n  \
             %x = add i32 %w, %t32\n{}  br label %loop\nloop:\n  \
             %k = phi i32 [ 0, %entry ], [ %k1, %loop ]\n  %a = phi i32 [ %x, %entry ], [ %a1, %loop ]\n{}  \
             %a1 = mul i32 %a, 1664525\n  %k1 = add i32 %k, 1\n  %done = icmp eq i32 %k1, 4\n  \
             br i1 %done, label %exit, label %loop\nexit:\n  \
             %q = getelementptr inbounds i32, ptr addrspace(1) %b1, i32 %t\n  \
             store i32 %a1, ptr addrspace(1) %q, align 4\n",
            call("0"),
            call("%t16"),
            call("-1"),
            call("8"),
        )
    };
    let decl = "declare void @llvm.agx3.yield(i16 noundef zeroext) #0\n\
                attributes #0 = { convergent nobuiltin nounwind \"no-builtins\" }\n";
    let (Some(yields), Some(plain)) = (
        assemble_text("yields", &kernel("k", 2, "", &body(true), decl)),
        assemble_text("plain", &kernel("k", 2, "", &body(false), "")),
    ) else {
        return;
    };
    let mut compiled = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let with = compile(&yields, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let without = compile(&plain, "k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(with.module.words, without.module.words, "{stem}");
        link_and_validate(&with, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        compiled += 1;
    }
    assert!(compiled > 0, "no committed profile");
    let (plan, m) = typed_module(&yields, "k", &moltenvk()).expect("module");
    let local = 32;
    let out = run_all(
        &plan,
        &m,
        local,
        SubgroupSize::W32,
        &[vec![0u8; 4 * local as usize], vec![0u8; 4 * local as usize]],
    );
    for t in 0..local {
        let v = t.wrapping_mul(0x9e37_79b1);
        let a = (v.wrapping_add(t)).wrapping_mul(1_664_525u32.wrapping_pow(4));
        assert_eq!(words(&out[0])[t as usize], v, "b0[{t}]");
        assert_eq!(words(&out[1])[t as usize], a, "b1[{t}]");
    }
}
