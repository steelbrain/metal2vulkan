//! A simdgroup multiply-add on simdgroups whose lanes may not all take part.
//!
//! The owned fixtures `engine/dsm_*` run an 8x8 `simdgroup_multiply_accumulate`, in float and
//! in half, in a block `lower::simd::uniform` cannot prove workgroup uniform (`dsm_uniform`: in
//! workgroup-uniform control flow). Where the whole simdgroup takes part the product is the exact
//! `c + a·b`, [`exact`] below, against which the literal readings of the authored cases and of our
//! probe's uncommitted ones (Apple M3 Max, macOS 26.7) are checked. Where only part of it does
//! (not live, or past the threadgroup's thread count), MSL makes the multiply-add undefined (MSL
//! 4.1 section 2.4: a simdgroup matrix operation outside uniform control flow within the
//! simdgroup), so its cells are not promised. On every profile whose kernels emulate their
//! simdgroups (`lavapipe`, `runtime-floor`) each fixture compiles, passes L3, and runs on the CPU
//! interpreter at 8, 16, and 32 subgroup lanes in both scheduling orders:
//!
//! - whole simdgroups (a branch on the simdgroup index, or on a device word equal across the
//!   simdgroup): the exact `c + a·b`;
//! - part of a simdgroup live (lanes 0-15, the odd lanes; cells loaded before the branch, built
//!   from plain loads, or loaded inside it), and a threadgroup below 32 threads, in divergent and
//!   in uniform control flow: the run completes and every running thread writes its cells;
//! - the same for the AGX `f32matmad8x8` (hand-written AIR: MSL cannot spell it).
//!
//! `moltenvk-m3`'s emulated-simdgroups variant, which a dispatch with a partial simdgroup runs,
//! does the same.
//!
//! On the subgroup tiers (`moltenvk-m3`'s `Direct` at 32 lanes, a wave64 device's `Cluster32`
//! at 64) a multiply-add in a block the uniformity analysis cannot prove uniform ballots its live
//! lanes and reads a lane that does not take part through the calling lane, without a shuffle
//! naming it: the interpreter traps a shuffle from an inactive lane, so these runs fail without it.
//! `engine/dsm2_collective_branch` puts the multiply-add under a branch on a simdgroup
//! collective's result after a loop whose exit differs by lane, which the workgroup analysis
//! reads as divergent.
//!
//! The owned no-fast-math fixtures `engine/dsm2_zero_sign*` give a whole simdgroup's signed zeros
//! exactly on every tier: with a `-0` accumulator and every other term `-0`, Metal's results show
//! the signs of the exact sum.
//!
//! Lanes that do not run the multiply-add write nothing. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::subgroup::{OpBits, SimdClass, Tier};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::lower::select::Plan;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile_kept, engine_dir, link_and_validate, llvm_assemble, profile, typed_module_with,
    AssembleError,
};
use crate::pipeline::LoweringInputs;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// The committed profiles whose kernels' simdgroups are emulated: one with device addresses and
/// one without.
const PROFILES: [&str; 2] = ["lavapipe", "runtime-floor"];

/// The poison every output word starts as.
const POISON: u32 = 0xABAB_ABAB;

const WIDTHS: [SubgroupSize; 3] = [SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32];

fn assemble_path(path: &std::path::Path, what: &str) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

fn fixture(stem: &str) -> Option<Vec<u8>> {
    assemble_path(&engine_dir().join(format!("{stem}.ll")), stem)
}

fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-partly-live-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = assemble_path(&ll, name);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// The distributed layout: lane `l` holds row `row(l)`, columns `col(l)` and `col(l) + 1`.
fn row(l: u32) -> usize {
    (((l >> 1) & 3) | ((l >> 4) << 2)) as usize
}

fn col(l: u32) -> usize {
    (((l >> 3) & 1) * 4 + (l & 1) * 2) as usize
}

type M8 = [[f64; 8]; 8];

/// Which lanes (or threads) take part in, or run, a multiply-add.
type Lanes<'a> = &'a dyn Fn(u32) -> bool;

/// The probes' inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Input {
    /// `a` the row rotation `a[r][(r + 4) & 7] = 1`, `b[k][j] = 8k + j + 1`,
    /// `c[r][j] = 256 + 8r + j`: every cell, product, and sum an integer exact in half.
    A,
    /// A with the rotation's 1 made 2 in columns 0, 1, 4, 5 (the even lanes' columns) and 64
    /// added to `b` rows 4-7 (lanes 16-31's cells).
    B,
    /// A with `a[0][4] = +inf` (lane 8's cell), whose step reads `b` row 4 (lanes 16-23's).
    Infinity,
    /// The signed-zero input of `dsm2_zero_sign`: `a` is +1 except -1 in rows 2-3, columns 4-7; `b`
    /// rows 0-3 are -0 and rows 4-7 (lanes 16-31's cells) are `rows`; `c` is -0.
    Signed { rows: f64 },
    /// The `a`-side signed-zero input: `a` is +1 in columns 0, 1, 4, 5 (the even lanes' cells) and
    /// -0 elsewhere, `b[k][j]` is -1 for `k` in {0, 1, 4, 5} and `j` in {6, 7}, else +1, `c` is -0.
    Odd,
}

fn inputs(which: Input) -> [M8; 3] {
    match which {
        Input::Signed { rows } => {
            let mut a = [[1.0; 8]; 8];
            for row in a.iter_mut().take(4).skip(2) {
                for x in row.iter_mut().skip(4) {
                    *x = -1.0;
                }
            }
            let mut b = [[-0.0; 8]; 8];
            for r in b.iter_mut().skip(4) {
                *r = [rows; 8];
            }
            return [a, b, [[-0.0; 8]; 8]];
        }
        Input::Odd => {
            let even = |k: usize| matches!(k, 0 | 1 | 4 | 5);
            let mut a = [[-0.0; 8]; 8];
            let mut b = [[1.0; 8]; 8];
            for r in 0..8 {
                for k in 0..8 {
                    if even(k) {
                        a[r][k] = 1.0;
                    }
                    if even(r) && k >= 6 {
                        b[r][k] = -1.0;
                    }
                }
            }
            return [a, b, [[-0.0; 8]; 8]];
        }
        _ => {}
    }
    let other = which == Input::B;
    let mut a = [[0.0; 8]; 8];
    let mut b = [[0.0; 8]; 8];
    let mut c = [[0.0; 8]; 8];
    for r in 0..8 {
        let k = (r + 4) & 7;
        a[r][k] = if other && matches!(k, 0 | 1 | 4 | 5) {
            2.0
        } else {
            1.0
        };
        for j in 0..8 {
            b[r][j] = (8 * r + j + 1 + if other && r >= 4 { 64 } else { 0 }) as f64;
            c[r][j] = (256 + 8 * r + j) as f64;
        }
    }
    if which == Input::Infinity {
        a[0][4] = f64::INFINITY;
    }
    [a, b, c]
}

/// The cells of lane `l` of a whole simdgroup's multiply-add: the exact `c + a·b`.
fn exact(m: &[M8; 3], l: u32) -> [f64; 2] {
    tile(m, l, 8)
}

/// [`exact`] over dimension `dim` (8, or 4: each of the tile's quadrants multiplied on its own,
/// the AGX `matmad4x4`). Each cell accumulates from `c` in increasing `k`; the products here are
/// exact, so a sum of signed zeros is `-0` exactly when every term is.
fn tile(m: &[M8; 3], l: u32, dim: usize) -> [f64; 2] {
    let [a, b, c] = m;
    let r = row(l);
    let cell = |q: usize| {
        let (r0, q0) = if dim == 4 { (r & 4, q & 4) } else { (0, 0) };
        (0..dim).fold(c[r][q], |acc, k| acc + a[r][q0 + k] * b[r0 + k][q])
    };
    [cell(col(l)), cell(col(l) + 1)]
}

/// Words with every NaN made one: Metal's NaN is `0x7fc00000`, and Vulkan does not specify a
/// NaN's bits.
fn canon(w: [u32; 4]) -> [u32; 4] {
    w.map(|x| {
        if f32::from_bits(x).is_nan() {
            0x7fc0_0000
        } else {
            x
        }
    })
}

/// The literal readings our probes took on Metal of whole simdgroups (both passes
/// byte-identical): `(case, input, lane, float cells)`. The half cells are the same values.
#[test]
fn the_model_is_metals_reading() {
    let inf = f64::INFINITY;
    let readings: [(&str, Input, u32, [f64; 2]); 2] = [
        // 434ee829: a whole simdgroup.
        ("434ee829", Input::A, 0, [289.0, 291.0]),
        // Our probe's uncommitted case: a whole simdgroup reads +inf.
        ("150d4c95", Input::Infinity, 8, [inf, inf]),
    ];
    for (case, input, lane, want) in readings {
        let got = exact(&inputs(input), lane);
        assert_eq!(
            canon(written(got)),
            canon(written(want)),
            "{case} lane {lane}"
        );
    }
}

fn f32s(v: impl IntoIterator<Item = f64>) -> Vec<u8> {
    v.into_iter()
        .flat_map(|x| (x as f32).to_le_bytes())
        .collect()
}

fn f16s(v: impl IntoIterator<Item = f64>) -> Vec<u8> {
    v.into_iter()
        .flat_map(|x| f64_to_f16(x).to_le_bytes())
        .collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn emulating(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile")
}

/// `runtime-floor` with 4-lane subgroups in the vertex, fragment and compute stages and the basic,
/// vote, arithmetic, ballot, shuffle, relative shuffle and quad operations (the shape of a
/// software rasterizer's subgroups): its vertex and fragment functions take the
/// `PartialFragment` tier, its kernels emulate their simdgroups.
fn narrow_graphics() -> DeviceProfile {
    use crate::vk::{ShaderStageFlags as S, SubgroupFeatureFlags as O};
    let mut desc = crate::pipeline::testing::description("runtime-floor").expect("description");
    desc.subgroup.size = 4;
    desc.subgroup.supported_stages = S::VERTEX | S::FRAGMENT | S::COMPUTE;
    desc.subgroup.supported_operations =
        O::BASIC | O::VOTE | O::ARITHMETIC | O::BALLOT | O::SHUFFLE | O::SHUFFLE_RELATIVE | O::QUAD;
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// The buffers of a `dsm_*` kernel over `n` threads: `in`, `hin` (a, b, c row-major), the
/// poisoned `out`, and `sel` when the kernel binds it.
fn dsm_buffers(m: &[M8; 3], n: u32, sel: Option<[u32; 3]>) -> Vec<Vec<u8>> {
    let flat = || m.iter().flat_map(|x| x.iter().flatten().copied());
    let mut bufs = vec![f32s(flat()), f16s(flat()), vec![0xAB; 16 * n as usize]];
    if let Some(s) = sel {
        bufs.push(s.iter().flat_map(|w| w.to_le_bytes()).collect());
    }
    bufs
}

/// The buffers `bufs` (bound at slots 0..) after one workgroup of `local` threads ran `m`,
/// identical in both scheduling orders. Push fields are zero residuals.
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
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        // Single-region dispatch: the region is not read. Zero residuals.
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "unexpected push field {f:?}"
        );
    }
    let buffers = bufs
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
    let res = Resources {
        buffers,
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

/// The plan and module of `kernel` on profile `p`: emulated simdgroups, L3 passed, no runtime
/// fact beyond the emulation's.
fn emulated(bc: &[u8], kernel: &str, p: &DeviceProfile) -> (Plan, TypedModule) {
    let (a, plan, m) = compile_kept(bc, kernel, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    assert_eq!(plan.simd.1, Tier::Emulated, "{kernel}");
    assert!(
        !plan.facts.contains(&RuntimeFact::ExecutionStatus),
        "{kernel}: nothing to report"
    );
    (plan, m)
}

/// The float and half cells of thread `g` in `out` (a grid of `n` threads), as words.
fn cells(out: &[u8], n: u32, g: u32) -> [u32; 4] {
    let w = words(out);
    let (g, n) = (g as usize, n as usize);
    [
        w[2 * g],
        w[2 * g + 1],
        w[2 * n + 2 * g],
        w[2 * n + 2 * g + 1],
    ]
}

/// Cells `x` as the kernel writes them: float, and half widened.
fn written(x: [f64; 2]) -> [u32; 4] {
    let h = |v: f64| (f16_to_f64(f64_to_f16(v)) as f32).to_bits();
    [
        (x[0] as f32).to_bits(),
        (x[1] as f32).to_bits(),
        h(x[0]),
        h(x[1]),
    ]
}

/// One dispatch of a `dsm_*` case: the fixture, the grid (one threadgroup), the input, `sel`,
/// and which threads run the multiply-add.
type Dispatch<'a> = (&'a str, u32, Input, Option<[u32; 3]>, Lanes<'a>);

/// Run each dispatch on each emulating profile and check every thread ([`check_on`]).
fn check(dispatches: &[Dispatch<'_>]) {
    for stem in PROFILES {
        let p = emulating(stem);
        check_on(stem, dispatches, &WIDTHS, &|bc, entry| {
            Some(emulated(bc, entry, &p))
        });
    }
}

/// A wave64-only compute device: the committed RDNA3 description with its subgroup size range
/// narrowed to 64, whose kernels' simdgroups are 32-lane clusters (`Cluster32`).
fn wave64() -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/amd-780m-radv.json");
    let text = std::fs::read_to_string(&path).expect("read the RDNA3 description");
    let narrowed = text.replace("\"minSubgroupSize\": 32", "\"minSubgroupSize\": 64");
    assert_ne!(narrowed, text, "the description names minSubgroupSize");
    let desc = DeviceDescription::from_khronos_json(&narrowed, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

/// The subgroup tiers: `moltenvk-m3`'s `Direct` at 32 lanes, and a wave64 device's `Cluster32`
/// at 64.
fn subgroup_tiers() -> [(&'static str, DeviceProfile, Tier, SubgroupSize); 2] {
    [
        (
            "moltenvk-m3",
            emulating("moltenvk-m3"),
            Tier::Direct,
            SubgroupSize::W32,
        ),
        ("wave64", wave64(), Tier::Cluster32, SubgroupSize::W64),
    ]
}

/// The plan and module of `kernel` on profile `p`, whose tier is `tier`: L3 passed, and the
/// SPIR-V words.
fn on_tier(
    bc: &[u8],
    kernel: &str,
    p: &DeviceProfile,
    tier: Tier,
) -> (Plan, TypedModule, Vec<u32>) {
    let (a, plan, m) = compile_kept(bc, kernel, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    assert_eq!(plan.simd.1, tier, "{kernel}");
    (plan, m, words)
}

/// Run each dispatch (a grid a multiple of the subgroup) on each subgroup tier and check every
/// thread ([`check_on`]).
fn check_subgroups(dispatches: &[Dispatch<'_>]) {
    for (stem, p, tier, lanes) in subgroup_tiers() {
        let n = match lanes {
            SubgroupSize::W64 => 64,
            _ => 32,
        };
        let fitted: Vec<Dispatch<'_>> = dispatches
            .iter()
            .filter(|d| d.1 % n == 0)
            .copied()
            .collect();
        assert!(!fitted.is_empty(), "{stem}: no dispatch fits {n} lanes");
        check_on(stem, &fitted, &[lanes], &|bc, entry| {
            let (plan, m, _) = on_tier(bc, entry, &p, tier);
            Some((plan, m))
        });
    }
}

/// How a check compiles a fixture's entry: its plan and module (`None`: skipped).
type Compiled<'a> = &'a dyn Fn(&[u8], &str) -> Option<(Plan, TypedModule)>;

/// Run each dispatch on the module `compiled` gives, at each subgroup width of `widths`, and
/// check every thread: one whose whole simdgroup runs the multiply-add holds the exact cells
/// ([`exact`]), one whose simdgroup runs it in part writes cells that are not promised, and one
/// that does not run it writes nothing.
fn check_on(
    what: &str,
    dispatches: &[Dispatch<'_>],
    widths: &[SubgroupSize],
    compiled: Compiled<'_>,
) {
    for &(entry, n, which, sel, runs) in dispatches {
        let Some(bc) = fixture(entry) else { return };
        let Some((plan, m)) = compiled(&bc, entry) else {
            return;
        };
        let input = inputs(which);
        for &lanes in widths {
            let out = run_all(&plan, &m, n, lanes, &dsm_buffers(&input, n, sel));
            for g in 0..n {
                let got = canon(cells(&out[2], n, g));
                let at = format!("{what} {entry} n={n} {which:?} {lanes:?} thread {g}");
                if !runs(g) {
                    assert_eq!(got, [POISON; 4], "{at}");
                } else if whole(n, runs, g) {
                    assert_eq!(got, canon(written(exact(&input, g & 31))), "{at}");
                } else {
                    assert!(got.iter().all(|&w| w != POISON), "{at}: written");
                }
            }
        }
    }
}

/// Whether every lane of thread `g`'s simdgroup exists in a threadgroup of `n` threads and runs
/// the multiply-add.
fn whole(n: u32, runs: Lanes<'_>, g: u32) -> bool {
    let base = g & !31;
    (base..base + 32).all(|u| u < n && runs(u))
}

/// A multiply-add that only whole simdgroups run is the exact `c + a·b`: simdgroup 1 of 2 (a branch
/// on the simdgroup index), simdgroups 0 and 2 of 3 (a device word per simdgroup), and the one
/// simdgroup of a threadgroup of 32.
#[test]
fn whole_simdgroups_run_the_exact_multiply_add() {
    check(&[
        ("dsm_whole_by_simdgroup", 64, Input::A, None, &|g| g >= 32),
        ("dsm_whole_by_buffer", 96, Input::A, Some([1, 0, 1]), &|g| {
            g / 32 != 1
        }),
        (
            "dsm_whole_by_buffer",
            32,
            Input::A,
            Some([1, 0, 1]),
            &|_| true,
        ),
    ]);
}

/// Where only some lanes of a simdgroup run the multiply-add (every lane loaded its cells before
/// the branch, the cells are built from plain loads, or the running lanes load them inside it; in
/// one and in two simdgroups, and with both inputs), the run completes and each running lane
/// writes its cells, which are not promised.
#[test]
fn a_partly_live_simdgroup_runs_its_multiply_add() {
    let half = |g: u32| g % 32 < 16;
    let odd = |g: u32| g % 2 == 1;
    check(&[
        ("dsm_half_live_loaded", 32, Input::A, None, &half),
        ("dsm_half_live_loaded", 32, Input::B, None, &half),
        ("dsm_half_live_loaded", 64, Input::A, None, &half),
        ("dsm_half_live_built", 32, Input::B, None, &half),
        ("dsm_half_live_inside", 32, Input::A, None, &half),
        ("dsm_odd_live_loaded", 32, Input::A, None, &odd),
        ("dsm_odd_live_loaded", 32, Input::B, None, &odd),
        ("dsm_half_live_loaded", 32, Input::Infinity, None, &half),
    ]);
}

/// In a threadgroup below 32 threads every thread of which runs the multiply-add (in a block the
/// translator cannot prove uniform), the run completes and each thread writes its cells, which are
/// not promised; at 48 threads the first simdgroup is whole and exact, and at 32 every thread is.
#[test]
fn a_partial_simdgroup_runs_its_multiply_add() {
    let sel = Some([1, 1, 1]);
    check(&[
        ("dsm_whole_by_buffer", 16, Input::A, sel, &|_| true),
        ("dsm_whole_by_buffer", 16, Input::B, sel, &|_| true),
        ("dsm_whole_by_buffer", 8, Input::A, sel, &|_| true),
        ("dsm_whole_by_buffer", 1, Input::A, sel, &|_| true),
        ("dsm_whole_by_buffer", 48, Input::A, sel, &|_| true),
        ("dsm_whole_by_buffer", 16, Input::Infinity, sel, &|_| true),
        ("dsm_whole_by_buffer", 32, Input::Infinity, sel, &|_| true),
    ]);
}

/// In workgroup-uniform control flow: a threadgroup whose last simdgroup is partial runs and
/// writes its cells, which are not promised, and a whole simdgroup is exact.
#[test]
fn uniform_flow_runs_a_partial_simdgroups_multiply_add() {
    let all: Lanes<'_> = &|_| true;
    check(&[
        ("dsm_uniform", 32, Input::A, None, all),
        ("dsm_uniform", 16, Input::A, None, all),
        ("dsm_uniform", 16, Input::B, None, all),
        ("dsm_uniform", 16, Input::Infinity, None, all),
        ("dsm_uniform", 8, Input::A, None, all),
        ("dsm_uniform", 1, Input::A, None, all),
        ("dsm_uniform", 48, Input::A, None, all),
    ]);
}

/// Metal's readings of the owned no-fast-math cases of whole simdgroups (two passes,
/// byte-identical), `(case, input, lane, float cells)`; the half cells are the same values. With a
/// `-0` accumulator and every other term `-0`, each cell shows the sign of the exact sum.
#[test]
fn a_whole_simdgroups_signed_zeros_are_metals_reading() {
    let minus = Input::Signed { rows: -0.0 };
    let plus = Input::Signed { rows: 0.0 };
    let (p, n) = (0.0, -0.0);
    let readings: [(&str, Input, u32, [f64; 2]); 7] = [
        ("dd0e539e", minus, 0, [n, n]),
        ("dd0e539e", minus, 4, [p, p]),
        ("dd0e539e", minus, 16, [n, n]),
        ("6bd772f3", plus, 0, [p, p]),
        ("6bd772f3", plus, 4, [n, n]),
        ("7b445bf4", Input::Odd, 0, [4.0, 4.0]),
        ("7b445bf4", Input::Odd, 9, [-4.0, -4.0]),
    ];
    for (case, input, lane, want) in readings {
        let got = exact(&inputs(input), lane);
        assert_eq!(written(got), written(want), "{case} lane {lane}");
    }
}

/// The signed-zero inputs' live masks (`dsm2_zero_sign`'s `live[0]`) and running threads.
fn zero_sign_dispatches(n: u32) -> Vec<Dispatch<'static>> {
    let minus = Input::Signed { rows: -0.0 };
    let plus = Input::Signed { rows: 0.0 };
    let all: Lanes<'static> = &|_| true;
    let half: Lanes<'static> = &|g| g % 32 < 16;
    let odd: Lanes<'static> = &|g| g % 2 == 1;
    let mask = |m: u32| Some([m, 0, 0]);
    vec![
        ("dsm2_zero_sign", n, minus, mask(u32::MAX), all),
        ("dsm2_zero_sign", n, plus, mask(u32::MAX), all),
        ("dsm2_zero_sign", n, minus, mask(0xFFFF), half),
        ("dsm2_zero_sign", n, plus, mask(0xFFFF), half),
        ("dsm2_zero_sign", n, Input::Odd, mask(0xAAAA_AAAA), odd),
        ("dsm2_zero_sign", n, Input::Odd, mask(u32::MAX), all),
        ("dsm2_zero_sign_uniform", n, minus, None, all),
    ]
}

/// The zero-sign fixtures on every emulating profile, at 32 threads and in threadgroups that leave
/// the simdgroup partial (16, 8; 16 with lanes 0-7 live), in divergent and in uniform flow: a
/// whole simdgroup gives Metal's bits, and a partial or partly live one runs and writes its cells.
#[test]
fn the_emulated_tier_runs_the_zero_sign_fixtures() {
    let minus = Input::Signed { rows: -0.0 };
    let plus = Input::Signed { rows: 0.0 };
    let all: Lanes<'_> = &|_| true;
    let mut dispatches = zero_sign_dispatches(32);
    dispatches.extend([
        ("dsm2_zero_sign", 16, minus, Some([u32::MAX, 0, 0]), all),
        ("dsm2_zero_sign", 16, minus, Some([0xFF, 0, 0]), &|g| g < 8),
        ("dsm2_zero_sign", 8, minus, Some([u32::MAX, 0, 0]), all),
        ("dsm2_zero_sign_uniform", 16, minus, None, all),
        ("dsm2_zero_sign_uniform", 16, plus, None, all),
    ]);
    check(&dispatches);
}

/// The zero-sign fixtures on the subgroup tiers, whose threadgroups hold whole subgroups: the
/// whole-simdgroup controls give Metal's bits, and lanes 0-15 live and the odd lanes live run and
/// write their cells.
#[test]
fn the_subgroup_tiers_run_the_zero_sign_fixtures() {
    let mut dispatches = zero_sign_dispatches(32);
    dispatches.extend(zero_sign_dispatches(64));
    check_subgroups(&dispatches);
}

/// On the subgroup tiers whole simdgroups under a divergent branch, and in a uniform block, run
/// the exact multiply-add.
#[test]
fn whole_simdgroups_run_the_exact_multiply_add_on_the_subgroup_tiers() {
    let all: Lanes<'_> = &|_| true;
    check_subgroups(&[
        ("dsm_whole_by_simdgroup", 64, Input::A, None, &|g| g >= 32),
        ("dsm_whole_by_buffer", 64, Input::A, Some([1, 0, 1]), &|g| {
            g / 32 != 1
        }),
        ("dsm_whole_by_buffer", 32, Input::B, Some([1, 0, 1]), all),
        ("dsm_uniform", 64, Input::A, None, all),
    ]);
}

/// On the subgroup tiers a multiply-add some lanes of a simdgroup do not run (lanes 0-15, the odd
/// lanes; cells loaded before the branch, built from plain loads, or loaded inside it) reads them
/// through the ballot of the lanes that run it and never shuffles from one of them (the
/// interpreter traps that): the run completes and each running lane writes its cells.
#[test]
fn the_subgroup_tiers_run_a_partly_live_multiply_add_without_naming_a_lane_that_takes_no_part() {
    let half: Lanes<'_> = &|g| g % 32 < 16;
    let odd: Lanes<'_> = &|g| g % 2 == 1;
    check_subgroups(&[
        ("dsm_half_live_loaded", 32, Input::A, None, half),
        ("dsm_half_live_loaded", 64, Input::B, None, half),
        ("dsm_half_live_loaded", 64, Input::Infinity, None, half),
        ("dsm_half_live_built", 64, Input::B, None, half),
        ("dsm_half_live_inside", 64, Input::A, None, half),
        ("dsm_odd_live_loaded", 32, Input::A, None, odd),
        ("dsm_odd_live_loaded", 64, Input::B, None, odd),
    ]);
}

/// A multiply-add under a branch on a simdgroup collective's result, after a loop whose exit
/// differs by lane (`engine/dsm2_collective_branch`: lane `L` leaves holding `simd_sum(1) =
/// 32 - L`, and lanes 0-15 run the multiply-add). The decision reads the workgroup analysis, in
/// which a collective's result is divergent, so the branch is divergent and every tier runs it as
/// a partly live multiply-add; a reading that took the branch for uniform would shuffle from
/// inactive lanes on the subgroup tiers, which the interpreter traps.
#[test]
fn a_multiply_add_under_a_branch_on_a_collective_is_divergent() {
    let half: Lanes<'_> = &|g| g % 32 < 16;
    let dispatches: [Dispatch<'_>; 2] = [
        ("dsm2_collective_branch", 32, Input::A, None, half),
        ("dsm2_collective_branch", 64, Input::B, None, half),
    ];
    check(&dispatches);
    check_subgroups(&dispatches);
}

fn opcodes(words: &[u32], op: crate::spv::Op) -> usize {
    let mut at = 5;
    let mut n = 0;
    while let Some(w) = words.get(at) {
        if w & 0xffff == op as u32 {
            n += 1;
        }
        at += ((w >> 16) as usize).max(1);
    }
    n
}

/// On the subgroup tiers only a multiply-add in a block the uniformity analysis cannot prove
/// uniform takes the live-lane form: `dsm_half_live_loaded`'s two are planned divergent, raise the
/// class with `Ballot`, and emit one ballot each and two bit extracts per step (the `a` and the `b`
/// owner); `dsm_uniform`'s two keep their plain shuffles.
#[test]
fn only_divergent_multiply_adds_ballot_their_lanes() {
    let (Some(divergent), Some(uniform)) =
        (fixture("dsm_half_live_loaded"), fixture("dsm_uniform"))
    else {
        return;
    };
    let ballot = |c: &SimdClass| matches!(c, SimdClass::CrossLane(o) if o.contains(OpBits::BALLOT));
    for (stem, p, tier, _) in subgroup_tiers() {
        let (plan, _, words) = on_tier(&divergent, "dsm_half_live_loaded", &p, tier);
        assert_eq!(plan.families.simd.partly_live.len(), 2, "{stem}");
        assert!(ballot(&plan.simd.0), "{stem}: {:?}", plan.simd.0);
        assert_eq!(
            opcodes(&words, crate::spv::Op::GroupNonUniformBallot),
            2,
            "{stem}"
        );
        assert_eq!(
            opcodes(&words, crate::spv::Op::GroupNonUniformBallotBitExtract),
            2 * 2 * 8,
            "{stem}"
        );
        let (plan, _, words) = on_tier(&uniform, "dsm_uniform", &p, tier);
        assert!(plan.families.simd.partly_live.is_empty(), "{stem}");
        assert!(!ballot(&plan.simd.0), "{stem}: {:?}", plan.simd.0);
        assert_eq!(
            opcodes(&words, crate::spv::Op::GroupNonUniformBallot),
            0,
            "{stem}"
        );
    }
}

/// On `moltenvk-m3`, whose kernels run whole simdgroups on subgroups, a dispatch with a partial
/// simdgroup runs the kernel's emulated-simdgroups variant, in uniform and in divergent control
/// flow: a whole simdgroup is exact, and a partial or partly live one runs and writes its cells.
#[test]
fn the_emulated_variant_runs_partial_simdgroups() {
    let p = emulating("moltenvk-m3");
    let mut inputs = LoweringInputs::new();
    inputs.emulated_simdgroups = true;
    let variant = |bc: &[u8], entry: &str| {
        let (plan, m) =
            typed_module_with(bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, Tier::Emulated, "{entry}");
        assert!(
            plan.facts.contains(&RuntimeFact::EmulatedSimdgroups),
            "{entry}"
        );
        Some((plan, m))
    };
    let all: Lanes<'_> = &|_| true;
    let half: Lanes<'_> = &|g| g % 32 < 16;
    check_on(
        "moltenvk-m3 variant",
        &[
            ("dsm_uniform", 16, Input::A, None, all),
            ("dsm_uniform", 48, Input::Infinity, None, all),
            ("dsm_whole_by_buffer", 16, Input::B, Some([1, 1, 1]), all),
            ("dsm_half_live_loaded", 48, Input::A, None, half),
            (
                "dsm2_zero_sign_uniform",
                16,
                Input::Signed { rows: -0.0 },
                None,
                all,
            ),
            (
                "dsm2_zero_sign",
                16,
                Input::Signed { rows: -0.0 },
                Some([0xFF, 0, 0]),
                &|g| g < 8,
            ),
        ],
        &WIDTHS,
        &variant,
    );
}

const AGX_HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A hand-written AIR kernel: thread `t` loads its two cells of `a`, `b`, `c` from buffers 0-2
/// (`2t`, `2t + 1`) and, when `%go` (which the instructions `cond` define from `%t`) holds,
/// writes `llvm.agx2.f32matmad<dim>x<dim>` of them to buffer 3.
fn agx_kernel(name: &str, cond: &str, dim: u32) -> String {
    let op = format!("llvm.agx2.f32matmad{dim}x{dim}.v2f32");
    let mut params = String::new();
    let mut md = String::new();
    let mut list = Vec::new();
    for i in 0..4 {
        params.push_str(&format!(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}, "
        ));
        md.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"b{i}\"}}\n",
            20 + i
        ));
        list.push(format!("!{}", 20 + i));
    }
    params.push_str("i32 noundef %t");
    md.push_str("!24 = !{i32 4, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}\n");
    list.push("!24".to_string());
    format!(
        "{AGX_HEADER}define void @{name}({params}) {{\nentry:\n  \
         %o32 = shl i32 %t, 1\n  %o = zext i32 %o32 to i64\n  \
         %pa = getelementptr inbounds float, ptr addrspace(1) %b0, i64 %o\n  %a = load <2 x float>, ptr addrspace(1) %pa, align 8\n  \
         %pb = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %o\n  %b = load <2 x float>, ptr addrspace(1) %pb, align 8\n  \
         %pc = getelementptr inbounds float, ptr addrspace(1) %b2, i64 %o\n  %c = load <2 x float>, ptr addrspace(1) %pc, align 8\n  \
         {cond}\n  br i1 %go, label %mad, label %done\n\n\
         mad:\n  %r = tail call <2 x float> @{op}(<2 x float> %a, <2 x float> %b, <2 x float> %c)\n  \
         %pr = getelementptr inbounds float, ptr addrspace(1) %b3, i64 %o\n  store <2 x float> %r, ptr addrspace(1) %pr, align 8\n  \
         br label %done\n\n\
         done:\n  ret void\n}}\n\n\
         declare <2 x float> @{op}(<2 x float>, <2 x float>, <2 x float>)\n\n\
         !air.kernel = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{}}\n!11 = !{{{}}}\n{md}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        list.join(", ")
    )
}

/// The hand-written AGX kernels: whole (simdgroup 1 of 2), lanes 0-15 live, the odd lanes live,
/// and every thread of a threadgroup of `small` (lanes past it absent), with the threads each
/// dispatch runs.
fn agx_kernels(small: u32) -> [(&'static str, &'static str, u32, Lanes<'static>); 4] {
    [
        ("agx_whole", "%go = icmp uge i32 %t, 32", 64, &|t| t >= 32),
        (
            "agx_half_live",
            "%lane = and i32 %t, 31\n  %go = icmp ult i32 %lane, 16",
            64,
            &|t| t % 32 < 16,
        ),
        (
            "agx_odd_live",
            "%low = and i32 %t, 1\n  %go = icmp ne i32 %low, 0",
            64,
            &|t| t % 2 == 1,
        ),
        ("agx_all", "%go = icmp ult i32 %t, 4096", small, &|_| true),
    ]
}

/// How a test compiles an AGX kernel: its plan and module.
type CompiledAgx<'a> = &'a dyn Fn(&[u8], &str) -> (Plan, TypedModule);

/// Run AGX kernel `name` (`dim`, the threads that run it `runs`, of `n`) compiled by
/// `compiled` at each of `widths`, and check every thread as [`check_on`] does ([`tile`]).
fn check_agx(
    what: &str,
    (name, cond, n, runs): (&str, &str, u32, Lanes<'_>),
    dim: u32,
    widths: &[SubgroupSize],
    compiled: CompiledAgx<'_>,
) {
    let stem = format!("{name}_{dim}x{dim}");
    let Some(bc) = assemble_text(&stem, &agx_kernel(&stem, cond, dim)) else {
        return;
    };
    let (plan, m) = compiled(&bc, &stem);
    let input = inputs(Input::B);
    let of = |x: &M8| {
        f32s((0..n).flat_map(|t| {
            let l = t % 32;
            [x[row(l)][col(l)], x[row(l)][col(l) + 1]]
        }))
    };
    for &lanes in widths {
        let bufs = [
            of(&input[0]),
            of(&input[1]),
            of(&input[2]),
            vec![0xAB; 8 * n as usize],
        ];
        let out = run_all(&plan, &m, n, lanes, &bufs);
        let got = words(&out[3]);
        for t in 0..n {
            let cell = [got[2 * t as usize], got[2 * t as usize + 1]];
            let at = format!("{what} {stem} {lanes:?} thread {t}");
            if !runs(t) {
                assert_eq!(cell, [POISON; 2], "{at}");
            } else if whole(n, runs, t) {
                let x = tile(&input, t & 31, dim as usize);
                assert_eq!(
                    cell,
                    [(x[0] as f32).to_bits(), (x[1] as f32).to_bits()],
                    "{at}"
                );
            } else {
                assert!(cell.iter().all(|&w| w != POISON), "{at}: written");
            }
        }
    }
}

/// The rule for the AGX multiply-adds, which share the matrix one's lowering (Metal's AGX reading
/// is unmeasured), on every emulating profile, for the 8x8 form and the 4x4 one (each quadrant on
/// its own): run by simdgroup 1 of 2 (`t >= 32`) it is exact; run by lanes 0-15 or by the odd
/// lanes, or in a threadgroup of 16 every thread of which runs it, the run completes and each
/// running lane writes its cells.
#[test]
fn the_agx_multiply_adds_follow_the_matrix_one() {
    for stem in PROFILES {
        let p = emulating(stem);
        for dim in [8, 4] {
            for k in agx_kernels(16) {
                check_agx(stem, k, dim, &WIDTHS, &|bc, name| emulated(bc, name, &p));
            }
        }
    }
}

/// The same on the subgroup tiers, whose threadgroups hold whole subgroups (the all-threads kernel
/// at 64): a lane that does not run the AGX multiply-add is read through the ballot of the lanes
/// that run it, in the 8x8 and the 4x4 form.
#[test]
fn the_subgroup_tiers_agx_multiply_adds_follow_the_matrix_one() {
    for (stem, p, tier, lanes) in subgroup_tiers() {
        for dim in [8, 4] {
            for k in agx_kernels(64) {
                if lanes == SubgroupSize::W64 && k.2 % 64 != 0 {
                    continue;
                }
                check_agx(stem, k, dim, &[lanes], &|bc, name| {
                    let (plan, m, _) = on_tier(bc, name, &p, tier);
                    (plan, m)
                });
            }
        }
    }
}

/// Whether `words` declares `OpExecutionMode <entry> SignedZeroInfNanPreserve <width>`.
fn declares_signed_zero_inf_nan(words: &[u32], width: u32) -> bool {
    let mut i = 5;
    while let Some(w) = words.get(i) {
        let len = (w >> 16) as usize;
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if w & 0xFFFF == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::SignedZeroInfNanPreserve as u32
            && words[i + 3] == width
        {
            return true;
        }
        i += len;
    }
    false
}

/// Every step of a matrix or AGX multiply-add is an `f32` `Fma`. Without all of `nnan`, `ninf`, and
/// `nsz` on the call LLVM keeps their signed zeros, so the entry declares `SignedZeroInfNanPreserve
/// 32` where the profile supports it and notes `DeviceSignedZeroInfNan` where it does not; a `fast`
/// call needs neither. Without the mode a device may fold `fma(a, 0, c)` to `c`, which turns
/// Metal's `+0` for a `-0` accumulator into `-0` (RADV did, on the no-fast-math `dsm2_zero_sign*`
/// cases below 32 threads).
#[test]
fn multiply_adds_without_fast_flags_keep_their_signed_zeros() {
    let mut kernels: Vec<(String, Vec<u8>, bool)> = Vec::new();
    for (stem, needs) in [
        ("dsm2_zero_sign", true),
        ("dsm2_zero_sign_uniform", true),
        ("dsm2_collective_branch", true),
        ("dsm_uniform", false),
        ("dsm_half_live_loaded", false),
    ] {
        let Some(bc) = fixture(stem) else {
            return;
        };
        kernels.push((stem.to_string(), bc, needs));
    }
    for dim in [8, 4] {
        let cond = "%lane = and i32 %t, 31\n  %go = icmp ult i32 %lane, 16";
        let plain = agx_kernel("agx_half_live", cond, dim);
        let flagged = |f: &str| {
            plain.replace(
                "tail call <2 x float>",
                &format!("tail call {f}<2 x float>"),
            )
        };
        let fast = flagged("fast ");
        assert_ne!(fast, plain, "the AGX call takes flags");
        // A call missing exactly one of `nnan`, `ninf`, and `nsz` still keeps signed zeros.
        let one_short = flagged("nnan ninf ");
        for (tag, text, needs) in [
            ("plain", plain, true),
            ("fast", fast, false),
            ("nnan-ninf", one_short, true),
        ] {
            let Some(bc) = assemble_text("agx_half_live", &text) else {
                return;
            };
            kernels.push((format!("agx_half_live {dim} {tag}"), bc, needs));
        }
    }
    let (mut declared, mut noted) = (0, 0);
    for stem in ["lavapipe", "runtime-floor", "moltenvk-m3", "amd-780m-radv"] {
        let p = emulating(stem);
        let supported = p.fields().float_controls.f32.signed_zero_inf_nan_preserve;
        for (name, bc, needs) in &kernels {
            let kernel = name.split(' ').next().expect("a kernel name");
            let (a, _, _) =
                compile_kept(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
            let mode = declares_signed_zero_inf_nan(&words, 32);
            let note = a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan);
            assert_eq!(mode, *needs && supported, "{stem} {name}: the mode");
            assert_eq!(note, *needs && !supported, "{stem} {name}: the note");
            declared += usize::from(mode);
            noted += usize::from(note);
        }
    }
    assert!(
        declared > 0 && noted > 0,
        "{declared} declared, {noted} noted"
    );
}

/// A hand-written AIR vertex function: vertex `vid` loads its two cells of `a`, `b`, `c` from
/// buffers 0-2 (`2·vid`, `2·vid + 1`) and writes `llvm.agx2.f32matmad<dim>x<dim>` of them to
/// buffer 3, then returns a zero position.
fn agx_vertex(name: &str, dim: u32) -> String {
    let op = format!("llvm.agx2.f32matmad{dim}x{dim}.v2f32");
    let mut params = String::new();
    let mut md = String::new();
    let mut list = Vec::new();
    for i in 0..4 {
        params.push_str(&format!(
            "ptr addrspace(1) noundef \"air-buffer-no-alias\" %b{i}, "
        ));
        md.push_str(&format!(
            "!{} = !{{i32 {i}, !\"air.buffer\", !\"air.location_index\", i32 {i}, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"b{i}\"}}\n",
            20 + i
        ));
        list.push(format!("!{}", 20 + i));
    }
    params.push_str("i32 noundef %vid");
    md.push_str("!24 = !{i32 4, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"vid\"}\n");
    list.push("!24".to_string());
    format!(
        "{AGX_HEADER}define <{{ <4 x float> }}> @{name}({params}) {{\nentry:\n  \
         %o32 = shl i32 %vid, 1\n  %o = zext i32 %o32 to i64\n  \
         %pa = getelementptr inbounds float, ptr addrspace(1) %b0, i64 %o\n  %a = load <2 x float>, ptr addrspace(1) %pa, align 8\n  \
         %pb = getelementptr inbounds float, ptr addrspace(1) %b1, i64 %o\n  %b = load <2 x float>, ptr addrspace(1) %pb, align 8\n  \
         %pc = getelementptr inbounds float, ptr addrspace(1) %b2, i64 %o\n  %c = load <2 x float>, ptr addrspace(1) %pc, align 8\n  \
         %r = tail call <2 x float> @{op}(<2 x float> %a, <2 x float> %b, <2 x float> %c)\n  \
         %pr = getelementptr inbounds float, ptr addrspace(1) %b3, i64 %o\n  store <2 x float> %r, ptr addrspace(1) %pr, align 8\n  \
         %out = insertvalue <{{ <4 x float> }}> undef, <4 x float> zeroinitializer, 0\n  \
         ret <{{ <4 x float> }}> %out\n}}\n\n\
         declare <2 x float> @{op}(<2 x float>, <2 x float>, <2 x float>)\n\n\
         !air.vertex = !{{!9}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\
         !9 = !{{ptr @{name}, !10, !11}}\n!10 = !{{!30}}\n\
         !30 = !{{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}}\n\
         !11 = !{{{}}}\n{md}\
         !17 = !{{i32 2, i32 8, i32 0}}\n!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        list.join(", ")
    )
}

/// An AGX multiply-add outside a kernel always takes the ballot form, and on the `PartialFragment`
/// tier (a vertex or fragment subgroup narrower than 32 lanes, here `runtime-floor` given 4-lane
/// vertex and fragment subgroups with every non-clustered operation: [`narrow_graphics`]) the
/// Metal lanes at or above the subgroup's size are permanently inactive: neither the ballot's bit
/// extract nor the shuffle names them (both are poison there; the interpreter traps them). Each
/// vertex is a subgroup of its own on the interpreter, so every other owner lane is either not
/// live (below the size) or beyond it (8- and 16-lane subgroups): the run completes and each
/// vertex writes its cells, which are not promised.
#[test]
fn a_narrow_subgroups_permanently_inactive_lanes_are_named_by_no_instruction() {
    let p = narrow_graphics();
    let input = inputs(Input::B);
    for dim in [8, 4] {
        let name = format!("agx_vertex_{dim}x{dim}");
        let Some(bc) = assemble_text(&name, &agx_vertex(&name, dim)) else {
            return;
        };
        let (a, plan, m) = compile_kept(&bc, &name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(plan.simd.1, Tier::PartialFragment, "{name}");
        assert_eq!(plan.families.simd.partly_live.len(), 1, "{name}");
        let n = 6u32;
        let lane0 = |x: &M8| f32s((0..n).flat_map(|_| [x[row(0)][col(0)], x[row(0)][col(0) + 1]]));
        let bufs = [
            lane0(&input[0]),
            lane0(&input[1]),
            lane0(&input[2]),
            vec![0xAB; 8 * n as usize],
        ];
        let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
        let buffers: Vec<Buffer> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Vertex,
                    kind: ResourceKind::Buffer,
                    index: i as u32,
                }),
                address: None,
                bytes: b.clone(),
            })
            .collect();
        let res = Resources {
            buffers,
            push,
            spec: BTreeMap::new(),
        };
        for lanes in [SubgroupSize::W8, SubgroupSize::W16] {
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    workgroups: [n, 1, 1],
                    subgroup: lanes,
                    order,
                    ..Config::default()
                };
                let out = run(&m, &cfg, res.clone())
                    .unwrap_or_else(|e| panic!("{name} {lanes:?}: {e}"))
                    .buffers;
                let got = words(&out[3]);
                for v in 0..n as usize {
                    assert!(
                        [got[2 * v], got[2 * v + 1]].iter().all(|&w| w != POISON),
                        "{name} {lanes:?} vertex {v}: written"
                    );
                }
            }
        }
    }
}
