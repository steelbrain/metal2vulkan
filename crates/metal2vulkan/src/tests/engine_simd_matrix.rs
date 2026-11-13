//! Simdgroup matrices end to end.
//!
//! The owned fixture `engine/simd_matrix` and the public fixture
//! `kernel_simdgroup_matrix_transposed_and_offset_block` run on the CPU interpreter, in the
//! `Direct` tier at 32 lanes, the `Cluster32` tier at 64 lanes, and the `Emulated` tier at 8 and 16
//! lanes (the public fixture in the `Emulated` tier at every width but on moltenvk-m3, which its
//! lockstep order needs where the device does not run a simdgroup in lockstep), against
//! expectations computed here from the MSL meaning of each operation:
//!
//! - `thread_elements()` element `j` of lane `L` is cell `(row(L), col(L) + j)` of Apple's
//!   distributed 8x8 layout (the `agx2` storage mapping), and a block store writes every cell where
//!   the MSL formula puts it;
//! - `simdgroup_load`/`simdgroup_store` at a leading dimension, an origin, and transposed, in
//!   device and threadgroup memory;
//! - products `c + a·b` in every `half`/`float` mix the frontend spells, `bfloat` factors, and
//!   8-bit float factors (hand-written AIR: MSL has no spelling for them), with filled and
//!   diagonal matrices;
//! - blocks whose leading dimension and origin are runtime values, with and without `shaderInt64`,
//!   a block and plain accesses at constant `getelementptr`s into a threadgroup array, and the
//!   memory facts of block accesses: a block load reads and a block store writes exactly the hull
//!   of its cells.
//!
//! The data are small integers (or 8-bit floats that are multiples of 1/4 no larger than 4), so
//! every product and sum is exact and the expectations do not depend on the accumulation's
//! rounding. On every committed profile each kernel compiles, links, and passes L3 exactly where a
//! cross-lane shuffle class has a compute tier. 16x16x16 products are refused as
//! `SimdgroupMatrix`. No GPU is used.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::air::intrinsics::Fp8;
use crate::device::requirements::VkFeature;
use crate::device::subgroup::{self, OpBits, SimdClass, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature, Needs};
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::lower::arith::narrow::Fp8Layout;
use crate::lower::select::Plan;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::interface::{AccessSet, Window};
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, metal_assemble, profile,
    typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const ENTRIES: [&str; 3] = [
    "matrix_thread_elements",
    "matrix_mixed",
    "matrix_threadgroup_block",
];

fn assembled(r: Result<Vec<u8>, AssembleError>, what: &str) -> Option<Vec<u8>> {
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

fn owned() -> Option<Vec<u8>> {
    assembled(
        llvm_assemble(&engine_dir().join("simd_matrix.ll")),
        "simd_matrix",
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
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The tiers: (profile, tier, subgroup size, threadgroup sizes). `lavapipe` and `swiftshader`,
/// which cannot pin 32 lanes, emulate their simdgroups through workgroup memory, here on 8- and
/// 16-lane subgroups.
fn tiers() -> [(DeviceProfile, Tier, SubgroupSize, &'static [u32]); 4] {
    let committed = |stem| profile(stem, DevicePolicy::default()).expect("profile");
    [
        (moltenvk(), Tier::Direct, SubgroupSize::W32, &[32, 64]),
        (wave64(), Tier::Cluster32, SubgroupSize::W64, &[64]),
        (
            committed("lavapipe"),
            Tier::Emulated,
            SubgroupSize::W8,
            &[32, 64],
        ),
        (
            committed("swiftshader"),
            Tier::Emulated,
            SubgroupSize::W16,
            &[32],
        ),
    ]
}

/// The committed `moltenvk-m3` description without `shaderInt64`: 64-bit integers are held as
/// two words.
fn moltenvk_words() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(&path).expect("read the MoltenVK description");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    assert!(desc.features.remove(&VkFeature::ShaderInt64));
    let p = Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    assert!(!p.fields().leaf.int64);
    p
}

/// The two tiers and the 32-lane tier without `shaderInt64`.
fn tiers_and_words() -> Vec<(DeviceProfile, Tier, SubgroupSize, &'static [u32])> {
    let mut out: Vec<_> = tiers().into_iter().collect();
    out.push((moltenvk_words(), Tier::Direct, SubgroupSize::W32, &[32]));
    out
}

/// A buffer slot's index, access set, and proven window.
type SlotFact = (u32, AccessSet, Window);

/// Each accessed buffer slot's facts, by index.
fn slot_facts(plan: &Plan) -> Vec<SlotFact> {
    let mut out: Vec<_> = plan
        .slots
        .iter()
        .flatten()
        .map(|s| (s.slot.index, s.access, s.window))
        .collect();
    out.sort_by_key(|(i, _, _)| *i);
    out
}

fn bounded(start: i64, len: u64) -> Window {
    Window::Bounded { start, len }
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
    let mut out = outs.swap_remove(0);
    out.truncate(bufs.len());
    out
}

/// The distributed layout: lane `l` holds row `row(l)`, columns `col(l)` and `col(l) + 1`.
fn row(l: u32) -> usize {
    (((l >> 1) & 3) | ((l >> 4) << 2)) as usize
}

fn col(l: u32) -> usize {
    (((l >> 3) & 1) * 4 + (l & 1) * 2) as usize
}

type M8 = [[f64; 8]; 8];

/// `c + a·b`.
fn mad(a: &M8, b: &M8, c: &M8) -> M8 {
    let mut r = *c;
    for (i, ri) in r.iter_mut().enumerate() {
        for (j, x) in ri.iter_mut().enumerate() {
            for k in 0..8 {
                *x += a[i][k] * b[k][j];
            }
        }
    }
    r
}

/// Row-major block of 8x8 at `m[r][c] = src[r * 8 + c]`.
fn block(src: &[f64]) -> M8 {
    let mut m = [[0.0; 8]; 8];
    for (r, mr) in m.iter_mut().enumerate() {
        for (c, x) in mr.iter_mut().enumerate() {
            *x = src[r * 8 + c];
        }
    }
    m
}

fn flat(m: &M8) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

/// Small integers in `-4..=4`.
fn ints(n: usize, seed: u32) -> Vec<f64> {
    (0..n as u32)
        .map(|i| {
            f64::from(((i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) % 9) as i32 - 4)
        })
        .collect()
}

fn f32s(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()
}

fn f16s(v: &[f64]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f64_to_f16(x).to_le_bytes())
        .collect()
}

fn bf16s(v: &[f64]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| (((x as f32).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn read_f32(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn read_f16(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn check_plan(plan: &Plan, tier: Tier, what: &str) {
    assert_eq!(plan.simd.1, tier, "{what}");
    assert_eq!(
        plan.simd.0,
        SimdClass::CrossLane(OpBits::BASIC.union(OpBits::SHUFFLE)),
        "{what}: a product shuffles, the fragments read the lane"
    );
}

// ---- the owned fixture ---------------------------------------------------------------------

/// Lane `L`'s `thread_elements()` are cells `(row(L), col(L) + j)`: the product by the diagonal 2
/// plus the filled 0.5, read back through `thread_elements()` and through a block store, agrees
/// cell by cell.
#[test]
fn thread_elements_are_the_distributed_cells() {
    let Some(bc) = owned() else { return };
    let entry = "matrix_thread_elements";
    for (p, tier, lanes, locals) in tiers() {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        check_plan(&plan, tier, entry);
        for &n in locals {
            let input = ints(64, 5);
            let out = run_all(
                &plan,
                &m,
                n,
                lanes,
                &[f32s(&input), vec![0xFF; 256], vec![0xFF; 256]],
            );
            let mut a = [[0.0; 8]; 8];
            for l in 0..32u32 {
                for j in 0..2 {
                    a[row(l)][col(l) + j] = input[(2 * l) as usize + j];
                }
            }
            let mut two = [[0.0; 8]; 8];
            for (i, r) in two.iter_mut().enumerate() {
                r[i] = 2.0;
            }
            let want = mad(&a, &two, &[[0.5; 8]; 8]);
            let elems = read_f32(&out[1]);
            for l in 0..32u32 {
                for j in 0..2 {
                    assert_eq!(
                        elems[(2 * l) as usize + j],
                        want[row(l)][col(l) + j],
                        "lane {l} element {j}, local {n}"
                    );
                }
            }
            assert_eq!(read_f32(&out[2]), flat(&want), "block store, local {n}");
        }
    }
}

/// Products in every `half`/`float` mix and of `bfloat` factors, from and to device blocks.
#[test]
fn mixed_products_compute_the_matrix_product() {
    let Some(bc) = owned() else { return };
    let entry = "matrix_mixed";
    let (fa, ha, ba) = (ints(64, 1), ints(64, 2), ints(64, 3));
    let (f, h, b) = (block(&fa), block(&ha), block(&ba));
    let mut want_f = flat(&mad(&f, &h, &f));
    want_f.extend(flat(&mad(&h, &h, &f)));
    want_f.extend(flat(&mad(&b, &b, &f)));
    let mut want_h = flat(&mad(&h, &h, &h));
    want_h.extend(flat(&mad(&f, &h, &h)));
    for (p, tier, lanes, locals) in tiers() {
        let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        check_plan(&plan, tier, entry);
        for &n in locals {
            let out = run_all(
                &plan,
                &m,
                n,
                lanes,
                &[
                    f32s(&fa),
                    f16s(&ha),
                    bf16s(&ba),
                    vec![0xFF; 192 * 4],
                    vec![0xFF; 128 * 2],
                ],
            );
            assert_eq!(read_f32(&out[3]), want_f, "float results, local {n}");
            assert_eq!(read_f16(&out[4]), want_h, "half results, local {n}");
        }
    }
}

/// Threadgroup blocks at a leading dimension of 16, one loaded transposed at an origin, the product
/// stored transposed at another origin; cells outside the stored block keep their bytes. Also from
/// the typed-pointer companion, whose block pointer is the constant expression `getelementptr ([256
/// x float], @t, 0, 0)` rather than the global (canonicalized to it).
#[test]
fn threadgroup_blocks_follow_stride_origin_and_transpose() {
    let Some(opaque) = owned() else { return };
    let mut sources = vec![opaque];
    match metal_assemble(&engine_dir().join("simd_matrix.air.ll")) {
        Ok(b) => sources.push(b),
        Err(AssembleError::Missing(why)) => eprintln!("SKIPPED (typed pointers): {why}"),
        Err(AssembleError::Rejected(why)) => panic!("metal-as rejected simd_matrix: {why}"),
    }
    let entry = "matrix_threadgroup_block";
    let input = ints(256, 7);
    let t = |r: usize, c: usize| input[r * 16 + c];
    let mut a = [[0.0; 8]; 8];
    let mut b = [[0.0; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            a[r][c] = t(r, c);
            // simdgroup_load(b, t, 16, ulong2(3, 2), true): origin column 3, row 2, transposed.
            b[r][c] = t(2 + c, 3 + r);
        }
    }
    let prod = mad(&a, &b, &[[0.0; 8]; 8]);
    // simdgroup_store(r, out, 16, ulong2(5, 1), true).
    let sentinel = f64::from(f32::from_bits(0x7F7F_7F7F));
    let mut want = vec![sentinel; 160];
    for (r, pr) in prod.iter().enumerate() {
        for (c, x) in pr.iter().enumerate() {
            want[(1 + c) * 16 + 5 + r] = *x;
        }
    }
    for bc in &sources {
        for (p, tier, lanes, locals) in tiers() {
            let (plan, m) = typed_module(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
            check_plan(&plan, tier, entry);
            for &n in locals {
                let out = run_all(&plan, &m, n, lanes, &[f32s(&input), vec![0x7F; 160 * 4]]);
                assert_eq!(read_f32(&out[1]), want, "local {n}");
            }
        }
    }
}

/// The public fixture's loads and stores at every stride/origin/transpose spelling agree with the
/// MSL formulas cell by cell.
#[test]
fn public_block_fixture_matches_the_msl_formulas() {
    let stem = "kernel_simdgroup_matrix_transposed_and_offset_block";
    let Some(bc) = assembled(
        llvm_assemble(&fixtures_dir().join(format!("{stem}.ll"))),
        stem,
    ) else {
        return;
    };
    let input = ints(256, 9);
    let t = |r: usize, c: usize| input[r * 16 + c];
    let cells = |f: &dyn Fn(usize, usize) -> f64| -> M8 {
        std::array::from_fn(|r| std::array::from_fn(|c| f(r, c)))
    };
    let ms: [M8; 4] = [
        cells(&|r, c| t(r, c)),
        cells(&|r, c| t(c, r)),
        cells(&|r, c| t(2 + r, 3 + c)),
        cells(&|r, c| t(2 + c, 3 + r)),
    ];
    let sentinel = f64::from(f32::from_bits(0x7F7F_7F7F));
    let mut want = vec![sentinel; 768];
    for (i, m) in ms.iter().enumerate() {
        want[64 * i..64 * (i + 1)].copy_from_slice(&flat(m));
    }
    for r in 0..8 {
        for c in 0..8 {
            let x = ms[0][r][c];
            want[256 + c * 16 + r] = x;
            want[432 + (2 + r) * 16 + 3 + c] = x;
            want[608 + (2 + c) * 16 + 3 + r] = x;
        }
    }
    for (p, tier, lanes, locals) in tiers() {
        let (plan, m) = typed_module(&bc, stem, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        // The copy loop's stores at `tid + 32 i` may meet between threads of a simdgroup that
        // share `thread_position_in_threadgroup` (a threadgroup narrower than 32 threads), where
        // no barrier every thread of the simdgroup reaches orders them: the kernel takes the
        // `Emulated` tier, but on moltenvk-m3 (the `Direct` row), whose device runs a simdgroup's
        // 32 threads in lockstep itself and keeps the kernel's tier, and on the 64-lane RDNA3 row
        // (`Cluster32`), whose entry reconverges maximally: a subgroup barrier where the threads
        // diverge orders the threads that run it together, the lanes Apple's simdgroup runs it for,
        // so the kernel keeps its tier. The cells below are Metal's on every row.
        let (tier, key) = match tier {
            Tier::Direct => (tier, "barrier.lockstep.device"),
            Tier::Cluster32 => {
                assert!(p.fields().subgroup.maximal_reconvergence, "{stem}");
                (tier, "barrier.lockstep.simdgroup")
            }
            _ => (Tier::Emulated, "barrier.lockstep.emulated.simdgroup"),
        };
        assert_eq!(plan.simd.1, tier, "{stem}");
        let lockstep = plan.census();
        assert!(lockstep.contains_key(key), "{stem}: {lockstep:?}");
        for &n in locals {
            let out = run_all(&plan, &m, n, lanes, &[f32s(&input), vec![0x7F; 768 * 4]]);
            assert_eq!(read_f32(&out[1]), want, "local {n}");
        }
    }
}

/// A block load reads and a block store writes (a read-only buffer is not reported written), each
/// covering exactly the hull of its cells: 8x8 blocks at a leading dimension of 8 from and to the
/// buffer start and at constant element offsets, and a transposed store at origin (5, 1) of a
/// 16-wide array, cells `(1 + c) · 16 + 5 + r`, elements 21 to 140. Plain accesses at a
/// lane-dependent index keep the whole buffer.
#[test]
fn block_accesses_are_exact_memory_facts() {
    let Some(bc) = owned() else { return };
    let p = moltenvk();
    let (r, w) = (AccessSet::READ, AccessSet::WRITE);
    let cases: [(&str, Vec<SlotFact>); 3] = [
        (
            "matrix_thread_elements",
            vec![
                (0, r, Window::Whole),
                (1, w, Window::Whole),
                (2, w, bounded(0, 256)),
            ],
        ),
        (
            "matrix_mixed",
            vec![
                (0, r, bounded(0, 256)),
                (1, r, bounded(0, 128)),
                (2, r, bounded(0, 128)),
                (3, w, bounded(0, 768)),
                (4, w, bounded(0, 256)),
            ],
        ),
        (
            "matrix_threadgroup_block",
            vec![(0, r, Window::Whole), (1, w, bounded(84, 480))],
        ),
    ];
    for (entry, want) in cases {
        let (plan, _) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(slot_facts(&plan), want, "{entry}");
    }
}

// ---- hand-written AIR ----------------------------------------------------------------------

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-matrix-{}-{}-{name}",
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
/// its simdgroup; `decls` go after the function.
fn kernel(name: &str, buffers: u32, body: &str, decls: &str) -> String {
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
        "!{} = !{{i32 {buffers}, !\"air.thread_index_in_simdgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}\n",
        20 + buffers
    ));
    list.push(format!("!{}", 20 + buffers));
    format!(
        "{HEADER}define void @{name}({params}) {{\nentry:\n{body}  ret void\n}}\n\n{decls}\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{md}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        list.join(", ")
    )
}

const FP8: [(&str, Fp8); 3] = [
    ("f8e4m3", Fp8::E4M3),
    ("f8e4m3fn", Fp8::E4M3Fn),
    ("f8e5m2", Fp8::E5M2),
];

/// `b0` holds the `a` bytes (two per lane) then the `b` bytes; each lane builds its elements of
/// both factors, multiplies them with a zero accumulator, and writes its two `float` results to
/// `b1`.
fn fp8_kernel(name: &str, fmt: &str) -> String {
    let op =
        format!("air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64{fmt}.v64{fmt}.v64f32");
    let body = format!(
        "  %o32 = shl i32 %t, 1\n  %o = zext i32 %o32 to i64\n  %o1 = add i64 %o, 1\n  \
         %pa0 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 %o\n  %a0 = load i8, ptr addrspace(1) %pa0, align 1\n  \
         %pa1 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 %o1\n  %a1 = load i8, ptr addrspace(1) %pa1, align 1\n  \
         %ob = add i64 %o, 64\n  %ob1 = add i64 %ob, 1\n  \
         %pb0 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 %ob\n  %c0 = load i8, ptr addrspace(1) %pb0, align 1\n  \
         %pb1 = getelementptr inbounds i8, ptr addrspace(1) %b0, i64 %ob1\n  %c1 = load i8, ptr addrspace(1) %pb1, align 1\n  \
         %va0 = insertelement <64 x i8> undef, i8 %a0, i64 0\n  %va = insertelement <64 x i8> %va0, i8 %a1, i64 1\n  \
         %vb0 = insertelement <64 x i8> undef, i8 %c0, i64 0\n  %vb = insertelement <64 x i8> %vb0, i8 %c1, i64 1\n  \
         %r = tail call <64 x float> @{op}(<64 x i8> %va, <64 x i8> %vb, <64 x float> zeroinitializer)\n  \
         %r0 = extractelement <64 x float> %r, i64 0\n  %r1 = extractelement <64 x float> %r, i64 1\n  \
         %q0 = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %o\n  store float %r0, ptr addrspace(1) %q0, align 4\n  \
         %q1 = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %o1\n  store float %r1, ptr addrspace(1) %q1, align 4\n"
    );
    let decls = format!("declare <64 x float> @{op}(<64 x i8>, <64 x i8>, <64 x float>)\n");
    kernel(name, 2, &body, &decls)
}

/// 64 bytes of format `l` decoding to multiples of 1/4 no larger than 4 in magnitude, so every
/// product sum is exact in `f32`.
fn fp8_bytes(l: Fp8Layout, seed: u32) -> Vec<u8> {
    let ok: Vec<u8> = (0..=255u8)
        .filter(|&b| {
            let v = l.value(b);
            v.is_finite() && v.abs() <= 4.0 && (v * 4.0).fract() == 0.0
        })
        .collect();
    (0..64u32)
        .map(|i| ok[((i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) as usize) % ok.len()])
        .collect()
}

/// 8-bit float factors decode exactly and multiply as the logical matrices their
/// `thread_elements()` place: by the integer decode, and on a device with its own 8-bit floats
/// (`shaderFloat8`: the RTX 50 profile) through the device's type where it has the format's
/// encoding (`f8e4m3fn`, `f8e5m2`: the module declares `Float8EXT`), not for the IEEE-like
/// `f8e4m3`.
#[test]
fn fp8_factors_multiply_exactly() {
    for (fmt, f) in FP8 {
        let name = format!("mac_{fmt}");
        let Some(bc) = assemble_text(&name, &fp8_kernel(&name, fmt)) else {
            return;
        };
        let l = Fp8Layout::of(f);
        let (ab, bb) = (fp8_bytes(l, 11), fp8_bytes(l, 12));
        let mut a = [[0.0; 8]; 8];
        let mut b = [[0.0; 8]; 8];
        for lane in 0..32u32 {
            for j in 0..2 {
                let i = (2 * lane) as usize + j;
                a[row(lane)][col(lane) + j] = l.value(ab[i]);
                b[row(lane)][col(lane) + j] = l.value(bb[i]);
            }
        }
        let want = mad(&a, &b, &[[0.0; 8]; 8]);
        let mut input = ab.clone();
        input.extend(&bb);
        let rtx50 = profile("nvidia-rtx50-linux", DevicePolicy::default()).expect("profile");
        let one: &'static [u32] = &[32];
        let runs = tiers()
            .into_iter()
            .map(|(p, tier, lanes, locals)| (p, Some(tier), lanes, locals))
            .chain([(rtx50, None, SubgroupSize::W32, one)]);
        let float8 =
            crate::device::requirements::SpirvCapability::of(crate::spv::Capability::Float8EXT);
        let mut native = 0;
        for (p, tier, lanes, locals) in runs {
            let (plan, m) = typed_module(&bc, &name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            if let Some(tier) = tier {
                assert_eq!(plan.simd.1, tier, "{name}");
            }
            let a = compile(&bc, &name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: L3: {e}"));
            let declared = a.requirements.spirv_capabilities.contains(&float8);
            assert_eq!(
                declared,
                p.fields().leaf.float8 && f != Fp8::E4M3,
                "{name}: {:?}",
                a.requirements.spirv_capabilities
            );
            native += usize::from(declared);
            for &n in locals {
                let out = run_all(&plan, &m, n, lanes, &[input.clone(), vec![0xFF; 256]]);
                let got = read_f32(&out[1]);
                for lane in 0..32u32 {
                    for j in 0..2 {
                        assert_eq!(
                            got[(2 * lane) as usize + j],
                            want[row(lane)][col(lane) + j],
                            "{name} lane {lane} element {j}, local {n}"
                        );
                    }
                }
            }
        }
        assert_eq!(native, usize::from(f != Fp8::E4M3), "{name}");
    }
}

/// A 16x16x16 product.
fn refused_kernels() -> Vec<(&'static str, String)> {
    let op16 = "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f32.v8f32.v8f32";
    let body16 = format!(
        "  %r = tail call <8 x float> @{op16}(<8 x float> zeroinitializer, i1 false, <8 x float> zeroinitializer, i1 false, <8 x float> zeroinitializer)\n  \
         %e = extractelement <8 x float> %r, i64 0\n  store float %e, ptr addrspace(1) %b0, align 4\n"
    );
    let decl16 =
        format!("declare <8 x float> @{op16}(<8 x float>, i1, <8 x float>, i1, <8 x float>)\n");
    // An AIR 2.7 block load whose transpose flag is no constant stays in that form.
    let legacy = format!(
        "  %w = load i32, ptr addrspace(1) %b0, align 4\n  %f = icmp ne i32 %w, 0\n  \
         %a = tail call <64 x float> @{LOAD1}(ptr addrspace(1) %b0, i64 8, <2 x i64> zeroinitializer, i1 %f)\n  \
         %e = extractelement <64 x float> %a, i64 0\n  store float %e, ptr addrspace(1) %b0, align 4\n"
    );
    let legacy_decl =
        format!("declare <64 x float> @{LOAD1}(ptr addrspace(1), i64, <2 x i64>, i1)\n");
    vec![
        ("mac16", kernel("mac16", 1, &body16, &decl16)),
        (
            "legacy_runtime_transpose",
            kernel("legacy_runtime_transpose", 1, &legacy, &legacy_decl),
        ),
    ]
}

/// The unmeasured 16x16x16 fragment layout is refused as `Unsupported{SimdgroupMatrix,
/// NotImplemented}`, and so is an AIR 2.7 block load whose transpose flag is no constant.
#[test]
fn unmeasured_forms_are_refused() {
    let p = moltenvk();
    for (name, text) in refused_kernels() {
        let Some(bc) = assemble_text(name, &text) else {
            return;
        };
        match compile(&bc, name, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::SimdgroupMatrix, "{name}");
                assert_eq!(u.needs, Needs::NotImplemented, "{name}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

const LOAD1: &str = "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
const LOAD3: &str = "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
const STORE1: &str = "air.simdgroup_matrix_8x8_store.v64f32.p1f32";

fn block_decls() -> String {
    format!(
        "declare <64 x float> @{LOAD1}(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)\n\
         declare <64 x float> @{LOAD3}(ptr addrspace(3), <2 x i64>, <2 x i64>, <2 x i64>)\n\
         declare void @{STORE1}(<64 x float>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)\n\
         declare void @air.wg.barrier(i32, i32)\n"
    )
}

/// `b2` holds the words `n`, `m`, `ox`, `oy`. The block at leading dimension `n` and origin
/// `(ox, oy)` of `b0` is loaded (the frontend's spelling of a runtime leading dimension: the
/// `zext` of a 32-bit value inserted into `<1, poison>`) and stored transposed at leading
/// dimension `m` and the constant origin `(2, 1)` into `b1`.
fn runtime_descriptor_kernel(name: &str) -> String {
    let body = format!(
        "  %n = load i32, ptr addrspace(1) %b2, align 4\n  \
         %pm = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 1\n  %m = load i32, ptr addrspace(1) %pm, align 4\n  \
         %pox = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 2\n  %ox32 = load i32, ptr addrspace(1) %pox, align 4\n  \
         %poy = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 3\n  %oy32 = load i32, ptr addrspace(1) %poy, align 4\n  \
         %nz = zext i32 %n to i64\n  %mz = zext i32 %m to i64\n  %ox = zext i32 %ox32 to i64\n  %oy = zext i32 %oy32 to i64\n  \
         %ext = insertelement <2 x i64> <i64 poison, i64 8>, i64 %nz, i64 0\n  \
         %ls = insertelement <2 x i64> <i64 1, i64 poison>, i64 %nz, i64 1\n  \
         %o0 = insertelement <2 x i64> poison, i64 %ox, i64 0\n  %o = insertelement <2 x i64> %o0, i64 %oy, i64 1\n  \
         %a = tail call <64 x float> @{LOAD1}(ptr addrspace(1) %b0, <2 x i64> %ext, <2 x i64> %ls, <2 x i64> %o)\n  \
         %ss = insertelement <2 x i64> <i64 poison, i64 1>, i64 %mz, i64 0\n  \
         tail call void @{STORE1}(<64 x float> %a, ptr addrspace(1) %b1, <2 x i64> <i64 8, i64 8>, <2 x i64> %ss, <2 x i64> <i64 2, i64 1>)\n"
    );
    kernel(name, 3, &body, &block_decls())
}

/// `b0`'s 256 floats are copied into the threadgroup array `@tg` (each lane eight), then the
/// block at leading dimension `b2[0]` whose pointer is the constant
/// `getelementptr ([256 x float], @tg, 0, 35)` is stored at leading dimension 8 into `b1`, and
/// each lane adds `tg[200]`, read through a constant `getelementptr`, and `tg[100 + lane]`, read
/// through a GEP of one, into `b1[64 + lane]`.
fn threadgroup_offset_kernel(name: &str) -> String {
    let mut body = String::from("  %base = shl i32 %t, 3\n");
    for k in 0..8 {
        body.push_str(&format!(
            "  %i{k} = add i32 %base, {k}\n  %z{k} = zext i32 %i{k} to i64\n  \
             %p{k} = getelementptr inbounds float, ptr addrspace(1) %b0, i64 %z{k}\n  \
             %v{k} = load float, ptr addrspace(1) %p{k}, align 4\n  \
             %q{k} = getelementptr inbounds [256 x float], ptr addrspace(3) @tg, i64 0, i64 %z{k}\n  \
             store float %v{k}, ptr addrspace(3) %q{k}, align 4\n"
        ));
    }
    body.push_str(&format!(
        "  tail call void @air.wg.barrier(i32 2, i32 1)\n  \
         %n = load i32, ptr addrspace(1) %b2, align 4\n  %nz = zext i32 %n to i64\n  \
         %ls = insertelement <2 x i64> <i64 1, i64 poison>, i64 %nz, i64 1\n  \
         %a = tail call <64 x float> @{LOAD3}(ptr addrspace(3) getelementptr inbounds ([256 x float], ptr addrspace(3) @tg, i64 0, i64 35), <2 x i64> <i64 8, i64 8>, <2 x i64> %ls, <2 x i64> zeroinitializer)\n  \
         tail call void @{STORE1}(<64 x float> %a, ptr addrspace(1) %b1, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)\n  \
         %x = load float, ptr addrspace(3) getelementptr inbounds ([256 x float], ptr addrspace(3) @tg, i64 0, i64 200), align 4\n  \
         %tz = zext i32 %t to i64\n  \
         %py = getelementptr inbounds float, ptr addrspace(3) getelementptr inbounds ([256 x float], ptr addrspace(3) @tg, i64 0, i64 100), i64 %tz\n  \
         %y = load float, ptr addrspace(3) %py, align 4\n  %s = fadd float %x, %y\n  \
         %oi = add i32 %t, 64\n  %oz = zext i32 %oi to i64\n  \
         %po = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %oz\n  store float %s, ptr addrspace(1) %po, align 4\n"
    ));
    let decls = format!(
        "@tg = internal addrspace(3) global [256 x float] undef, align 4\n{}",
        block_decls()
    );
    kernel(name, 3, &body, &decls)
}

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A runtime leading dimension and origin address the cells the MSL formula names, natively and
/// with 64-bit integers held as two words; the blocks' buffers are read and written as a whole (no
/// constant extent), the descriptor words as loaded.
#[test]
fn runtime_descriptors_address_the_formula_cells() {
    let name = "runtime_descriptors";
    let Some(bc) = assemble_text(name, &runtime_descriptor_kernel(name)) else {
        return;
    };
    let (n, m, ox, oy) = (12usize, 10usize, 3usize, 2usize);
    let input = ints(128, 13);
    let sentinel = f64::from(f32::from_bits(0x7F7F_7F7F));
    let mut want = vec![sentinel; 100];
    for r in 0..8 {
        for c in 0..8 {
            want[(r + 1) + (c + 2) * m] = input[(r + oy) * n + (c + ox)];
        }
    }
    let desc = words(&[n as u32, m as u32, ox as u32, oy as u32]);
    for (p, tier, lanes, locals) in tiers_and_words() {
        let (plan, module) = typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(plan.simd.1, tier, "{name}");
        assert_eq!(
            slot_facts(&plan),
            vec![
                (0, AccessSet::READ, Window::Whole),
                (1, AccessSet::WRITE, Window::Whole),
                (2, AccessSet::READ, bounded(0, 16)),
            ],
            "{name}"
        );
        for &l in locals {
            let out = run_all(
                &plan,
                &module,
                l,
                lanes,
                &[f32s(&input), vec![0x7F; 100 * 4], desc.clone()],
            );
            assert_eq!(read_f32(&out[1]), want, "local {l}");
        }
    }
}

/// Constant `getelementptr`s with a nonzero offset into a threadgroup array address the arena at
/// the global's offset plus their displacement: a block pointer (with a runtime leading dimension),
/// a plain load, and the base of a GEP.
#[test]
fn constant_geps_into_the_threadgroup_arena_keep_their_displacement() {
    let name = "threadgroup_offsets";
    let Some(bc) = assemble_text(name, &threadgroup_offset_kernel(name)) else {
        return;
    };
    let input = ints(256, 17);
    let mut want = Vec::new();
    for r in 0..8 {
        for c in 0..8 {
            want.push(input[35 + r * 16 + c]);
        }
    }
    for l in 0..32 {
        want.push(input[200] + input[100 + l]);
    }
    for (p, tier, lanes, locals) in tiers_and_words() {
        let (plan, module) = typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(plan.simd.1, tier, "{name}");
        for &l in locals {
            let out = run_all(
                &plan,
                &module,
                l,
                lanes,
                &[f32s(&input), vec![0xFF; 96 * 4], words(&[16])],
            );
            assert_eq!(read_f32(&out[1]), want, "local {l}");
        }
    }
}

/// The products record the device's accumulation precision, and an all-`half` product its
/// per-step rounding through `f32`.
#[test]
fn products_record_their_precision() {
    let Some(bc) = owned() else { return };
    let p = moltenvk();
    let a = compile(&bc, "matrix_mixed", &p).expect("compile");
    for note in [
        PrecisionNote::ReductionOrder,
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::F16FmaDoubleRounding,
    ] {
        assert!(a.precision.contains(&note), "{note:?}");
    }
    let a = compile(&bc, "matrix_thread_elements", &p).expect("compile");
    assert!(a.precision.contains(&PrecisionNote::ReductionOrder));
    assert!(!a.precision.contains(&PrecisionNote::F16FmaDoubleRounding));
}

/// Every matrix kernel compiles, links, relocates, and passes L3 on every committed profile
/// exactly where its cross-lane class (a shuffle with a product, the lane alone for blocks) has
/// a compute tier, and elsewhere is refused as that tier is (`matrix_mixed`'s five buffers fit
/// `vk12-floor`'s four storage buffers per stage through a uniform view). The plan's size estimate
/// bounds the emitted words.
#[test]
fn matrix_kernels_compile_or_refuse_with_the_tier() {
    // Each kernel with its operations' class: a product shuffles, the blocks alone read the lane.
    let product = OpBits::BASIC.union(OpBits::SHUFFLE);
    let mut kernels: Vec<(String, Vec<u8>, OpBits)> = Vec::new();
    let Some(bc) = owned() else { return };
    for e in ENTRIES {
        kernels.push((e.to_string(), bc.clone(), product));
    }
    for (fmt, _) in FP8 {
        let name = format!("mac_{fmt}");
        let Some(bc) = assemble_text(&name, &fp8_kernel(&name, fmt)) else {
            return;
        };
        kernels.push((name, bc, product));
    }
    for (name, text) in [
        (
            "runtime_descriptors",
            runtime_descriptor_kernel("runtime_descriptors"),
        ),
        (
            "threadgroup_offsets",
            threadgroup_offset_kernel("threadgroup_offsets"),
        ),
    ] {
        let Some(bc) = assemble_text(name, &text) else {
            return;
        };
        kernels.push((name.to_string(), bc, OpBits::BASIC));
    }
    let (mut ok, mut refused) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        for (name, bc, ops) in &kernels {
            let tier = subgroup::tier(
                &SimdClass::CrossLane(*ops),
                Stage::Kernel,
                &p.fields().subgroup,
            );
            match compile(bc, name, &p) {
                Ok(a) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
                    assert!(tier.is_ok(), "{stem} {name}");
                    let (plan, _) = typed_module(bc, name, &p).expect("plan");
                    let words = a.module.words.len() as u64;
                    assert!(
                        plan.estimate.words >= words,
                        "{stem} {name}: estimate {} < emitted {words}",
                        plan.estimate.words
                    );
                    ok += 1;
                }
                Err(Error::Unsupported(u)) => {
                    let want = tier.clone().expect_err("a refused tier");
                    assert_eq!(u.feature, want.feature, "{stem} {name}");
                    refused += 1;
                }
                Err(e) => panic!("{stem} {name}: {e}"),
            }
        }
    }
    assert!(ok > 0, "{ok} compiled, {refused} refused");
}

/// A block load or store is named by what its endpoint reaches: a descriptor view
/// (`simd.matrix_load`, under the profile's buffer encoding), threadgroup memory
/// (`simd.matrix_load.arena`, an arena checked in the shader in every leg), or a device address
/// (`simd.matrix_load.psb`, here every slot of a `BufferRangeLimit` variant).
#[test]
fn a_block_is_named_by_the_memory_it_reaches() {
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{plan_bitcode, EntryRef};
    use crate::pipeline::sym::{SlotClass, SlotRef};
    use crate::pipeline::testing::meter;

    let Some(bc) = owned() else {
        return;
    };
    let p = moltenvk();
    let blocks = |entry: &str, inputs: &LowerInputs| {
        let plan = plan_bitcode(
            &bc,
            EntryRef::Name(entry),
            &FunctionConstantValues::defaults(),
            &p,
            inputs,
            &mut meter(),
        )
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
        plan.census()
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with("simd.matrix_load") || name.starts_with("simd.matrix_store")
            })
            .map(|(name, (uses, _))| (name, uses))
            .collect::<Vec<_>>()
    };
    let named = |pairs: &[(&str, u32)]| {
        pairs
            .iter()
            .map(|&(n, u)| (n.to_owned(), u))
            .collect::<Vec<_>>()
    };
    let views = LowerInputs::new();
    assert_eq!(
        blocks("matrix_mixed", &views),
        named(&[("simd.matrix_load", 3), ("simd.matrix_store", 5)])
    );
    assert_eq!(
        blocks("matrix_threadgroup_block", &views),
        named(&[("simd.matrix_load.arena", 2), ("simd.matrix_store", 1)])
    );
    let mut addressed = LowerInputs::new();
    addressed.psb_slots = (0..8)
        .map(|index| SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index,
        })
        .collect();
    assert_eq!(
        blocks("matrix_mixed", &addressed),
        named(&[("simd.matrix_load.psb", 3), ("simd.matrix_store.psb", 5)])
    );
}

// ---- matrices held as fragments -----------------------------------------------

const INIT_DIAG: &str = "air.simdgroup_matrix_8x8_init_diag.v64f32.f32";
const MAC_F32: &str = "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32";

/// How `fragment_kernel`'s `insertelement` at 1 spells its index.
#[derive(Clone, Copy, Debug)]
enum Index {
    /// The constant 1.
    Const,
    /// A `phi` of 1 and a value the loop carries, `admitted` by the fragment proof or not.
    Carried { admitted: bool },
}

/// The kernel `name` over an `n` x `n` private array of matrices (each the frontend's
/// `simdgroup_matrix` structs around `<64 x float>`), indexed at run time: matrix `k` (row-major)
/// is first `init_diag(k)`, then `a·a` plus itself (`a` the block of `b0`) with 1 added to
/// `thread_elements()[1]` (an `insertelement` at 1 and a `shufflevector` taking elements 2..63
/// from the product, or from the inserted value for a carried `index`), then matrix 0 is
/// replaced by the last such product, read through a `phi` and a `select` against
/// `zeroinitializer`, and each matrix `k` is stored as a block at `b1[64 k]`. With `spoil`, each product is stored with element 5 set to 1: the array holds
/// more than fragments. `index` is how the `insertelement` at 1 spells its index.
fn fragment_kernel(name: &str, n: u32, spoil: bool, index: Index) -> String {
    let nn = n * n;
    let arr = format!("[{n} x [{n} x {{ {{ <64 x float> }} }}]]");
    let ext = "<2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer";
    // Elements 2..63 from the product, or, for a carried index, from the inserted value itself
    // (so the stored value is a fragment only when the insertion is proven one).
    let from = match index {
        Index::Const => 64,
        Index::Carried { .. } => 0,
    };
    let rest: Vec<String> = (2..64).map(|j| format!("i32 {}", from + j)).collect();
    let mask = format!("<64 x i32> <i32 0, i32 1, {}>", rest.join(", "));
    // The index is 1 whenever it runs: the loop's first `phi` value, then a value the loop
    // carries that is 1 at run time (`j | 1` reduced to its low bit, or the lane's `t != 99`,
    // which a lane index below 32 makes true), spelled so the fragment proof admits it, or
    // through a `udiv` it does not admit.
    let (phi, carried, idx) = match index {
        Index::Const => (String::new(), String::new(), "i64 1".to_string()),
        Index::Carried { admitted } => {
            let bit = if admitted {
                "and i32 %io, 1"
            } else {
                "udiv i32 %io, %io"
            };
            (
                "  %ix = phi i32 [ 1, %fill ], [ %ixn, %mul ]\n".to_string(),
                format!(
                    "%io = or i32 %j, 1\n  %ia = {bit}\n  %k0 = icmp ne i32 %t, 99\n  \
                     %ib = zext i1 %k0 to i32\n  %ixs = select i1 %k0, i32 %ia, i32 %ib\n  \
                     %ixt = trunc i32 %ixs to i16\n  %ixn = zext i16 %ixt to i32\n"
                ),
                "i32 %ix".to_string(),
            )
        }
    };
    let stored = if spoil {
        "  %y3 = insertelement <64 x float> %y2, float 1.000000e+00, i64 5\n  \
         store <64 x float> %y3, ptr %q, align 256\n"
    } else {
        "  store <64 x float> %y2, ptr %q, align 256\n"
    };
    let body = format!(
        "  %acc = alloca {arr}, align 256\n  \
         %a = tail call <64 x float> @{LOAD1}(ptr addrspace(1) %b0, {ext})\n  \
         br label %fill\n\
         fill:\n  \
         %i = phi i32 [ 0, %entry ], [ %i1, %fill ]\n  \
         %r = udiv i32 %i, {n}\n  %c = urem i32 %i, {n}\n  %v = uitofp i32 %i to float\n  \
         %d = tail call <64 x float> @{INIT_DIAG}(float %v)\n  \
         %p = getelementptr inbounds {arr}, ptr %acc, i32 0, i32 %r, i32 %c, i32 0\n  \
         store <64 x float> %d, ptr %p, align 256\n  \
         %i1 = add nuw i32 %i, 1\n  %fd = icmp eq i32 %i1, {nn}\n  \
         br i1 %fd, label %mul, label %fill\n\
         mul:\n  \
         %j = phi i32 [ 0, %fill ], [ %j1, %mul ]\n\
         {phi}  \
         %r2 = udiv i32 %j, {n}\n  %c2 = urem i32 %j, {n}\n  \
         %q = getelementptr inbounds {arr}, ptr %acc, i32 0, i32 %r2, i32 %c2, i32 0\n  \
         %x = load <64 x float>, ptr %q, align 256\n  \
         %y = tail call <64 x float> @{MAC_F32}(<64 x float> %a, <64 x float> %a, <64 x float> %x)\n  \
         %e1 = extractelement <64 x float> %y, i64 1\n  %e1b = fadd float %e1, 1.000000e+00\n  \
         {carried}  %y1 = insertelement <64 x float> %y, float %e1b, {idx}\n  \
         %y2 = shufflevector <64 x float> %y1, <64 x float> %y, {mask}\n\
         {stored}  \
         %j1 = add nuw i32 %j, 1\n  %md = icmp eq i32 %j1, {nn}\n  \
         br i1 %md, label %post, label %mul\n\
         post:\n  \
         %last = phi <64 x float> [ %y2, %mul ]\n  \
         %keep = icmp ne i32 %t, 99\n  \
         %sel = select i1 %keep, <64 x float> %last, <64 x float> zeroinitializer\n  \
         %p0 = getelementptr inbounds {arr}, ptr %acc, i32 0, i32 0, i32 0, i32 0\n  \
         store <64 x float> %sel, ptr %p0, align 256\n  \
         br label %out\n\
         out:\n  \
         %k = phi i32 [ 0, %post ], [ %k1, %out ]\n  \
         %r3 = udiv i32 %k, {n}\n  %c3 = urem i32 %k, {n}\n  \
         %s = getelementptr inbounds {arr}, ptr %acc, i32 0, i32 %r3, i32 %c3, i32 0\n  \
         %z = load <64 x float>, ptr %s, align 256\n  \
         %off = mul i32 %k, 64\n  %oz = zext i32 %off to i64\n  \
         %po = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %oz\n  \
         tail call void @{STORE1}(<64 x float> %z, ptr addrspace(1) %po, {ext})\n  \
         %k1 = add nuw i32 %k, 1\n  %od = icmp eq i32 %k1, {nn}\n  \
         br i1 %od, label %done, label %out\n\
         done:\n"
    );
    let decls = format!(
        "declare <64 x float> @{INIT_DIAG}(float)\n\
         declare <64 x float> @{MAC_F32}(<64 x float>, <64 x float>, <64 x float>)\n{}",
        block_decls()
    );
    kernel(name, 2, &body, &decls)
}

/// What [`fragment_kernel`] stores for the block `a` and `n` x `n` matrices: matrix `k` is
/// `a·a + diag(k)` with 1 added to every odd column (lane `L`'s element 1 is cell
/// `(row(L), col(L) + 1)`, and `col(L)` is even), and matrix 0 is the last one.
fn fragment_want(a: &M8, n: usize) -> Vec<f64> {
    let product = |k: usize| {
        let mut d = [[0.0; 8]; 8];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = k as f64;
        }
        let mut m = mad(a, a, &d);
        for row in m.iter_mut() {
            for x in row.iter_mut().skip(1).step_by(2) {
                *x += 1.0;
            }
        }
        flat(&m)
    };
    let nn = n * n;
    let mut out = product(nn - 1);
    for k in 1..nn {
        out.extend(product(k));
    }
    out
}

/// An array of simdgroup matrices indexed at run time, which holds only fragments, is a typed
/// object holding each matrix's two elements (no private arena), and answers as the MSL meaning
/// says on every tier; the same array holding a matrix whose element 5 is set lives whole in the
/// private arena and answers the same (element 5 is no cell). So does an array whose
/// `thread_elements()[i]` writes take `i` from a loop-carried `phi` whose every value is 0 or 1 (an
/// `and` with 1, an `i1` widened, their `select`, `trunc` and `zext`); one whose index passes
/// through a `udiv` is not proven, and lives whole.
#[test]
fn an_array_of_matrices_holding_fragments_holds_two_elements_each() {
    let n = 3usize;
    let input = ints(64, 7);
    let want = fragment_want(&block(&input), n);
    for (name, spoil, index) in [
        ("matrices_fragments", false, Index::Const),
        ("matrices_whole", true, Index::Const),
        (
            "matrices_carried_index",
            false,
            Index::Carried { admitted: true },
        ),
        (
            "matrices_unproven_index",
            false,
            Index::Carried { admitted: false },
        ),
    ] {
        let Some(bc) = assemble_text(name, &fragment_kernel(name, n as u32, spoil, index)) else {
            return;
        };
        let whole = spoil || matches!(index, Index::Carried { admitted: false });
        for (p, tier, lanes, locals) in tiers() {
            let (plan, module) =
                typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_plan(&plan, tier, name);
            let local = &plan.families.access.local;
            let fragments = local.typed.iter().any(|o| o.ty.has_fragments());
            assert_eq!(fragments, !whole, "{name}: typed as fragments");
            assert_eq!(local.private.is_some(), whole, "{name}: the private arena");
            for &l in locals {
                let out = run_all(
                    &plan,
                    &module,
                    l,
                    lanes,
                    &[f32s(&input), vec![0xFF; n * n * 64 * 4]],
                );
                assert_eq!(read_f32(&out[1]), want, "{name}, local {l}");
            }
        }
    }
}

/// How [`through_memory_kernel`] adds 1 to a product's element 1: through memory, as MSL's
/// `thread_elements()` does, after storing the product.
#[derive(Clone, Copy, Debug)]
enum Through {
    /// A `<2 x float>` load and store at the matrix's start (both held elements).
    Pair,
    /// A `float` load and store of element `i` (`getelementptr` into the `<64 x float>`); `i`
    /// past 1 reaches an element the fragment does not hold.
    Lane(u32),
}

/// [`fragment_kernel`]'s kernel (`Index::Const`, unspoiled) with its product's element 1 raised
/// by 1 through memory instead of an `insertelement`: the product is stored, `through` loads,
/// raises, and stores back element 1 (and element 0 unchanged, for a pair; for a lane past 1,
/// element `i` raised by 0), and the matrix is loaded again for what follows.
fn through_memory_kernel(name: &str, n: u32, through: Through) -> String {
    let text = fragment_kernel(name, n, false, Index::Const);
    let start = text
        .find("%e1 = extractelement <64 x float> %y, i64 1")
        .expect("the insertion");
    let tail = "store <64 x float> %y2, ptr %q, align 256\n";
    let end = text[start..].find(tail).expect("the store") + start + tail.len();
    let update = match through {
        Through::Pair => "%pe = load <2 x float>, ptr %q, align 256\n  \
             %pe1 = extractelement <2 x float> %pe, i64 1\n  \
             %pf1 = fadd float %pe1, 1.000000e+00\n  \
             %pf = insertelement <2 x float> %pe, float %pf1, i64 1\n  \
             store <2 x float> %pf, ptr %q, align 256\n"
            .to_string(),
        Through::Lane(i) => {
            // Element `i` raised by 1 if it is 1; an element past 1 by 0 (it holds zero, and
            // keeps it), and then element 1 by 1.
            let lane = |i: u32, by: &str| {
                format!(
                    "%q{i} = getelementptr inbounds <64 x float>, ptr %q, i32 0, i32 {i}\n  \
                     %l{i} = load float, ptr %q{i}, align 4\n  %f{i} = fadd float %l{i}, {by}\n  \
                     store float %f{i}, ptr %q{i}, align 4\n"
                )
            };
            match i {
                1 => lane(1, "1.000000e+00"),
                _ => format!("{}  {}", lane(i, "0.000000e+00"), lane(1, "1.000000e+00")),
            }
        }
    };
    format!(
        "{}store <64 x float> %y, ptr %q, align 256\n  {update}  \
         %y2 = load <64 x float>, ptr %q, align 256\n{}",
        &text[..start],
        &text[end..]
    )
}

/// An array of simdgroup matrices whose held elements a function reads and writes through memory
/// (MSL's `thread_elements()`: a `<2 x float>` at the matrix's start, or one `float` of element 0
/// or 1) still holds only fragments: a typed object of two elements a matrix, no private arena,
/// answering as the MSL meaning says on every tier. An access of element 2 reaches past the
/// fragment, so that array lives whole in the private arena, and answers the same.
#[test]
fn held_elements_accessed_through_memory_keep_the_fragment() {
    let n = 3usize;
    let input = ints(64, 7);
    let want = fragment_want(&block(&input), n);
    for (name, through) in [
        ("matrices_pair_through_memory", Through::Pair),
        ("matrices_lane_through_memory", Through::Lane(1)),
        ("matrices_past_the_fragment", Through::Lane(2)),
    ] {
        let Some(bc) = assemble_text(name, &through_memory_kernel(name, n as u32, through)) else {
            return;
        };
        let whole = matches!(through, Through::Lane(i) if i > 1);
        for (p, tier, lanes, locals) in tiers() {
            let (plan, module) =
                typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_plan(&plan, tier, name);
            let local = &plan.families.access.local;
            let fragments = local.typed.iter().any(|o| o.ty.has_fragments());
            assert_eq!(fragments, !whole, "{name}: typed as fragments");
            assert_eq!(local.private.is_some(), whole, "{name}: the private arena");
            for &l in locals {
                let out = run_all(
                    &plan,
                    &module,
                    l,
                    lanes,
                    &[f32s(&input), vec![0xFF; n * n * 64 * 4]],
                );
                assert_eq!(read_f32(&out[1]), want, "{name}, local {l}");
            }
        }
    }
}

/// A `shufflevector` whose result is a simdgroup matrix's fragment moves only its elements 0 and 1,
/// and the rest are zero. [`fragment_kernel`]'s `%y2` takes elements 0 and 1 from the raised
/// product and the rest from the product (or, with a carried index, from the raised product
/// itself): a fragment wherever the insertion is proven one, lowered as the fragment shuffle in
/// under 64 words where the lane-by-lane move of 64 lanes takes hundreds. With an index the proof
/// does not admit, the raised product is no fragment, so neither is `%y2`: the lane-by-lane move.
/// Every tier answers as the MSL meaning says either way.
#[test]
fn a_shuffle_into_a_fragment_moves_only_its_two_elements() {
    let n = 2usize;
    let input = ints(64, 11);
    let want = fragment_want(&block(&input), n);
    for (name, index, fragment) in [
        ("fragment_shuffle", Index::Const, true),
        (
            "fragment_shuffle_carried",
            Index::Carried { admitted: true },
            true,
        ),
        ("whole_shuffle", Index::Carried { admitted: false }, false),
    ] {
        let Some(bc) = assemble_text(name, &fragment_kernel(name, n as u32, false, index)) else {
            return;
        };
        for (p, tier, lanes, locals) in tiers() {
            let (plan, module) =
                typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_plan(&plan, tier, name);
            let census = plan.census();
            let moved = census.get("arith.shuffle.fragment").copied();
            let whole = census.get("arith.shuffle.chunked").copied();
            if fragment {
                let (uses, words) = moved.unwrap_or_else(|| panic!("{name}: {census:?}"));
                assert_eq!(uses, 1, "{name}: {census:?}");
                assert!(words < 64, "{name}: {words} words");
                assert_eq!(whole, None, "{name}: {census:?}");
            } else {
                assert_eq!(moved, None, "{name}: {census:?}");
                let (_, words) = whole.unwrap_or_else(|| panic!("{name}: {census:?}"));
                assert!(words > 64 * 4, "{name}: {words} words");
            }
            for &l in locals {
                let out = run_all(
                    &plan,
                    &module,
                    l,
                    lanes,
                    &[f32s(&input), vec![0xFF; n * n * 64 * 4]],
                );
                assert_eq!(read_f32(&out[1]), want, "{name}, local {l}");
            }
        }
    }
}

/// The private memory a thread holds counts a matrix held as its fragment as its two elements. On
/// the RTX 5080's measured 32 KiB a thread, a 16 x 16 array of `float` matrices compiles as
/// fragments (2 KiB); held whole it is 64 KiB, refused.
#[test]
fn matrices_held_as_fragments_count_their_two_elements() {
    let p = profile("nvidia-rtx5080-laptop", DevicePolicy::default()).expect("profile");
    for spoil in [false, true] {
        let name = if spoil {
            "matrices_whole"
        } else {
            "matrices_fragments"
        };
        let Some(bc) = assemble_text(name, &fragment_kernel(name, 16, spoil, Index::Const)) else {
            return;
        };
        let r = compile(&bc, name, &p);
        if !spoil {
            r.unwrap_or_else(|e| panic!("{name}: {e}"));
            continue;
        }
        match r {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::PrivateMemory, "{name}");
                assert!(format!("{u}").contains("required: 65536"), "{name}: {u}");
            }
            other => panic!("{name}: {:?}", other.map(|_| ())),
        }
    }
}

/// The kernel of `simdgroup_load` and `simdgroup_store` this file's AIR 2.7 test compiles, over
/// `b0` (read), `b1` (written) and `b2`, whose words are the elements per row `s` and the origin
/// `(ox, oy)`, and the threadgroup array `@tg`: `m = load(b0, s, (ox, oy), transposed)`,
/// `n = load(b0, 8)`, `r = m·n + 1`, `store(r, tg, 8)`, `q = load(tg, 8, (1, 2), transposed)`,
/// `store(q, b1, s, (ox, oy))` and `store(r, b1 + 64, 16, (3, 1), transposed)`. With `legacy`,
/// each block operation is the AIR 2.7 call `xcrun metal` 32023.921 writes for macOS 15
/// (`-std=metal3.0`); without, the AIR 2.8 operands it writes for macOS 26.
fn legacy_pair_kernel(name: &str, legacy: bool) -> String {
    let p1 = "ptr addrspace(1)";
    let p3 = "ptr addrspace(3)";
    let v2 = "<2 x i64>";
    let mut body = String::from(
        "  %s32 = load i32, ptr addrspace(1) %b2, align 4\n  %s = zext i32 %s32 to i64\n  \
         %pox = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 1\n  %ox32 = load i32, ptr addrspace(1) %pox, align 4\n  \
         %poy = getelementptr inbounds i32, ptr addrspace(1) %b2, i64 2\n  %oy32 = load i32, ptr addrspace(1) %poy, align 4\n  \
         %ox = zext i32 %ox32 to i64\n  %oy = zext i32 %oy32 to i64\n  \
         %o0 = insertelement <2 x i64> poison, i64 %ox, i64 0\n  %o = insertelement <2 x i64> %o0, i64 %oy, i64 1\n",
    );
    if legacy {
        body.push_str(&format!(
            "  %m = tail call <64 x float> @{LOAD1}({p1} %b0, i64 %s, {v2} %o, i1 true)\n  \
             %n = tail call <64 x float> @{LOAD1}({p1} %b0, i64 8, {v2} zeroinitializer, i1 false)\n"
        ));
    } else {
        body.push_str(&format!(
            "  %e1 = insertelement {v2} <i64 8, i64 poison>, i64 %s, i64 1\n  \
             %d1 = insertelement {v2} <i64 poison, i64 1>, i64 %s, i64 0\n  \
             %o1 = shufflevector {v2} %o, {v2} poison, <2 x i32> <i32 1, i32 0>\n  \
             %m = tail call <64 x float> @{LOAD1}({p1} %b0, {v2} %e1, {v2} %d1, {v2} %o1)\n  \
             %n = tail call <64 x float> @{LOAD1}({p1} %b0, {v2} <i64 8, i64 8>, {v2} <i64 1, i64 8>, {v2} zeroinitializer)\n"
        ));
    }
    body.push_str(&format!(
        "  %r0 = tail call <64 x float> @{INIT_DIAG}(float 1.0)\n  \
         %r = tail call <64 x float> @{MAC_F32}(<64 x float> %m, <64 x float> %n, <64 x float> %r0)\n"
    ));
    if legacy {
        body.push_str(&format!(
            "  tail call void @{STORE3}(<64 x float> %r, {p3} @tg, i64 8, {v2} zeroinitializer, i1 false)\n  \
             %q = tail call <64 x float> @{LOAD3}({p3} @tg, i64 8, {v2} <i64 1, i64 2>, i1 true)\n  \
             tail call void @{STORE1}(<64 x float> %q, {p1} %b1, i64 %s, {v2} %o, i1 false)\n  \
             %b64 = getelementptr inbounds float, {p1} %b1, i64 64\n  \
             tail call void @{STORE1}(<64 x float> %r, {p1} %b64, i64 16, {v2} <i64 3, i64 1>, i1 true)\n"
        ));
    } else {
        body.push_str(&format!(
            "  tail call void @{STORE3}(<64 x float> %r, {p3} @tg, {v2} <i64 8, i64 8>, {v2} <i64 1, i64 8>, {v2} zeroinitializer)\n  \
             %q = tail call <64 x float> @{LOAD3}({p3} @tg, {v2} <i64 8, i64 8>, {v2} <i64 8, i64 1>, {v2} <i64 2, i64 1>)\n  \
             %e2 = insertelement {v2} <i64 poison, i64 8>, i64 %s, i64 0\n  \
             %d2 = insertelement {v2} <i64 1, i64 poison>, i64 %s, i64 1\n  \
             tail call void @{STORE1}(<64 x float> %q, {p1} %b1, {v2} %e2, {v2} %d2, {v2} %o)\n  \
             %b64 = getelementptr inbounds float, {p1} %b1, i64 64\n  \
             tail call void @{STORE1}(<64 x float> %r, {p1} %b64, {v2} <i64 8, i64 16>, {v2} <i64 16, i64 1>, {v2} <i64 1, i64 3>)\n"
        ));
    }
    let blocks = if legacy {
        format!(
            "declare <64 x float> @{LOAD1}({p1}, i64, {v2}, i1)\n\
             declare <64 x float> @{LOAD3}({p3}, i64, {v2}, i1)\n\
             declare void @{STORE1}(<64 x float>, {p1}, i64, {v2}, i1)\n\
             declare void @{STORE3}(<64 x float>, {p3}, i64, {v2}, i1)\n"
        )
    } else {
        format!(
            "declare <64 x float> @{LOAD1}({p1}, {v2}, {v2}, {v2})\n\
             declare <64 x float> @{LOAD3}({p3}, {v2}, {v2}, {v2})\n\
             declare void @{STORE1}(<64 x float>, {p1}, {v2}, {v2}, {v2})\n\
             declare void @{STORE3}(<64 x float>, {p3}, {v2}, {v2}, {v2})\n"
        )
    };
    let decls = format!(
        "@tg = internal addrspace(3) global [128 x float] undef, align 4\n{blocks}\
         declare <64 x float> @{INIT_DIAG}(float)\n\
         declare <64 x float> @{MAC_F32}(<64 x float>, <64 x float>, <64 x float>)\n"
    );
    let text = kernel(name, 3, &body, &decls).replace(
        "!18 = !{!\"Metal\", i32 4, i32 0, i32 0}",
        "!18 = !{!\"Metal\", i32 3, i32 0, i32 0}",
    );
    if legacy {
        text.replace(
            "air64_v28-apple-macosx26.0.0",
            "air64_v27-apple-macosx15.0.0",
        )
        .replace(
            "!17 = !{i32 2, i32 8, i32 0}",
            "!17 = !{i32 2, i32 7, i32 0}",
        )
    } else {
        text
    }
}

const STORE3: &str = "air.simdgroup_matrix_8x8_store.v64f32.p3f32";

/// The AIR 2.7 block loads and stores (runtime and constant elements per row and origins,
/// transposed or not, device and threadgroup) compile on every committed profile to the words of
/// the AIR 2.8 operands the front end writes for the same MSL, or are refused as those are.
#[test]
fn air27_block_operations_compile_to_their_air28_words() {
    let name = "legacy_blocks";
    let (Some(old), Some(new)) = (
        assemble_text("legacy", &legacy_pair_kernel(name, true)),
        assemble_text("modern", &legacy_pair_kernel(name, false)),
    ) else {
        return;
    };
    let mut compiled = Vec::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let words = |bc: &[u8]| {
            compile(bc, name, &p).map(|a| {
                link_and_validate(&a, &p)
                    .map(|(_, w)| w)
                    .unwrap_or_else(|e| panic!("{stem}: L3: {e}"))
            })
        };
        match (words(&old), words(&new)) {
            (Ok(o), Ok(n)) => {
                assert!(o == n, "{stem}: the forms' words differ");
                compiled.push(stem);
            }
            (Err(o), Err(n)) => assert_eq!(o.to_string(), n.to_string(), "{stem}"),
            (o, n) => panic!("{stem}: AIR 2.7 {:?}, AIR 2.8 {:?}", o.err(), n.err()),
        }
    }
    assert!(
        compiled.iter().any(|s| s == "moltenvk-m3"),
        "compiled on {compiled:?}"
    );
}
