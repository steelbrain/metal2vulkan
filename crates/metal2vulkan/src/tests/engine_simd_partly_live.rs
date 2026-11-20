//! A simdgroup multiply-add on emulated simdgroups whose lanes do not all take part.
//!
//! The owned fixtures `engine/dsm_*` run an 8x8 `simdgroup_multiply_accumulate`, in float and
//! in half, in a block `lower::simd::uniform` cannot prove workgroup uniform (`dsm_uniform`: in
//! workgroup-uniform control flow). Metal (Apple M3 Max,
//! macOS 26.7) computes such a product deterministically across runs, processes, inputs, and
//! dispatch shapes, with the cells of every lane that does not take part (not live, or past the
//! threadgroup's thread count) read as zero, so an infinite cell meeting one makes NaN: that is
//! [`metal`] below, and the literal readings of the authored cases and of our probe's
//! uncommitted ones are checked against it. On every profile whose kernels emulate their
//! simdgroups (`lavapipe`, `vk12-floor`) each fixture compiles, passes L3, and runs on the CPU
//! interpreter at 8, 16, and 32 subgroup lanes in both scheduling orders, giving that answer:
//!
//! - whole simdgroups (a branch on the simdgroup index, or on a device word equal across the
//!   simdgroup): the exact `c + a·b`;
//! - part of a simdgroup live (lanes 0-15, the odd lanes; cells loaded before the branch, built
//!   from plain loads, or loaded inside it), and a threadgroup below 32 threads, in divergent and
//!   in uniform control flow: the product with the other lanes' cells zero;
//! - the same for the AGX `f32matmad8x8` (hand-written AIR: MSL cannot spell it).
//!
//! `moltenvk-m3`'s emulated-simdgroups variant, which a dispatch with a partial simdgroup runs,
//! gives the same answers.
//!
//! On the subgroup tiers (`moltenvk-m3`'s `Direct` at 32 lanes, a wave64 device's `Cluster32`
//! at 64) a multiply-add in a block the uniformity analysis cannot prove uniform ballots its live
//! lanes and reads a lane that does not take part as zero cells without a shuffle naming it: the
//! interpreter traps a shuffle from an inactive lane, so these runs fail without it.
//! `engine/dsm2_collective_branch` puts the multiply-add under a branch on a simdgroup
//! collective's result after a loop whose exit differs by lane, which the workgroup analysis
//! reads as divergent.
//!
//! The zero cell is `+0` (the owned no-fast-math fixtures `engine/dsm2_zero_sign*`): with a
//! `-0` accumulator and every other term `-0`, Metal's results show the sign of a not-live lane's
//! and an absent lane's zero, on the `a` side and the `b` side, and every tier gives those bits.
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
const PROFILES: [&str; 2] = ["lavapipe", "vk12-floor"];

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
    /// rows 0-3 are -0 and rows 4-7 (lanes 16-31's cells) are `rows`; `c` is -0. Where lanes 16-31
    /// do not take part, rows 0-1 read their zero `z` and rows 2-3 `-z`.
    Signed { rows: f64 },
    /// The `a`-side signed-zero input: `a` is +1 in columns 0, 1, 4, 5 (the even lanes' cells) and
    /// -0 elsewhere, `b[k][j]` is -1 for `k` in {0, 1, 4, 5} and `j` in {6, 7}, else +1, `c` is -0.
    /// Where the even lanes do not take part, columns 2-3 read their zero `z` and columns 6-7 `-z`.
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

/// Metal's cells for lane `l` of a simdgroup whose lanes `part` take part in the multiply-add:
/// `c + a·b` with the `a` and `b` cells of every other lane `+0`, so an infinite cell whose step
/// meets another lane's zero cell makes NaN.
fn metal(m: &[M8; 3], part: Lanes<'_>, l: u32) -> [f64; 2] {
    tile(m, part, l, 8, 0.0)
}

/// [`metal`] over dimension `dim` (8, or 4: each of the tile's quadrants multiplied on its own,
/// the AGX `matmad4x4`), with `zero` the cell of a lane that does not take part. Each cell
/// accumulates from `c` in increasing `k`; the products here are exact, so a sum of signed
/// zeros is `-0` exactly when every term is.
fn tile(m: &[M8; 3], part: Lanes<'_>, l: u32, dim: usize, zero: f64) -> [f64; 2] {
    let [a, b, c] = m;
    let mut ta = [[zero; 8]; 8];
    let mut tb = [[zero; 8]; 8];
    for o in (0..32).filter(|&o| part(o)) {
        for j in 0..2 {
            ta[row(o)][col(o) + j] = a[row(o)][col(o) + j];
            tb[row(o)][col(o) + j] = b[row(o)][col(o) + j];
        }
    }
    let r = row(l);
    let cell = |q: usize| {
        let (r0, q0) = if dim == 4 { (r & 4, q & 4) } else { (0, 0) };
        (0..dim).fold(c[r][q], |acc, k| acc + ta[r][q0 + k] * tb[r0 + k][q])
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

/// The literal readings our probes took on Metal (both passes byte-identical): `(case, input, part,
/// lane, float cells)`. The half cells are the same values.
#[test]
fn the_model_is_metals_reading() {
    let half = |l: u32| l < 16;
    let odd = |l: u32| l % 2 == 1;
    let all = |_: u32| true;
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let readings: [(&str, Input, Lanes<'_>, u32, [f64; 2]); 14] = [
        // 434ee829: a whole simdgroup.
        ("434ee829", Input::A, &all, 0, [289.0, 291.0]),
        // e2474ecb, ce3f96fb: lanes 0-15 live, inputs B and A alike.
        ("e2474ecb", Input::B, &half, 0, [256.0, 257.0]),
        ("ce3f96fb", Input::A, &half, 9, [262.0, 263.0]),
        // 46c2ad06: the odd lanes, input B.
        ("46c2ad06", Input::B, &odd, 1, [258.0, 259.0]),
        ("46c2ad06", Input::B, &odd, 5, [389.0, 391.0]),
        ("46c2ad06", Input::B, &odd, 17, [290.0, 291.0]),
        ("46c2ad06", Input::B, &odd, 21, [325.0, 327.0]),
        // 2b60dd21: a threadgroup of 16.
        ("2b60dd21", Input::A, &half, 15, [286.0, 287.0]),
        // b577c71f: a threadgroup of 48, simdgroup 0 whole.
        ("b577c71f", Input::A, &all, 31, [349.0, 351.0]),
        // Our probe's uncommitted cases: a whole simdgroup reads +inf; lanes 0-15 live, and
        // a threadgroup of 16, read NaN where the infinite cell meets a zero one.
        ("150d4c95", Input::Infinity, &all, 8, [inf, inf]),
        ("a62e77f4", Input::Infinity, &half, 0, [nan, nan]),
        ("a62e77f4", Input::Infinity, &half, 9, [nan, nan]),
        ("a62e77f4", Input::Infinity, &half, 2, [264.0, 265.0]),
        ("56e1b000", Input::Infinity, &half, 1, [nan, nan]),
    ];
    for (case, input, part, lane, want) in readings {
        let got = metal(&inputs(input), part, lane);
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
    profile(stem, DevicePolicy::default()).expect("profile")
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

/// Run each dispatch on each emulating profile and check every thread against [`metal`].
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
/// thread against [`metal`].
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
/// check every thread against [`metal`]: a thread that runs the multiply-add holds Metal's cells
/// for the lanes of its simdgroup that run it, any other writes nothing.
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
                let base = g & !31;
                let part = |l: u32| base + l < n && runs(base + l);
                let want = if runs(g) {
                    written(metal(&input, &part, g & 31))
                } else {
                    [POISON; 4]
                };
                assert_eq!(
                    canon(cells(&out[2], n, g)),
                    canon(want),
                    "{what} {entry} n={n} {which:?} {lanes:?} thread {g}"
                );
            }
        }
    }
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

/// Where only some lanes of a simdgroup run the multiply-add, the other lanes' cells are zero,
/// whether every lane loaded its cells before the branch, the cells are built from plain loads, or
/// the running lanes load them inside it; in one and in two simdgroups, and with both inputs.
#[test]
fn a_partly_live_simdgroup_reads_the_other_lanes_as_zero() {
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
/// translator cannot prove uniform), the lanes past the thread count take part with zero cells; at
/// 48 threads the first simdgroup is whole.
#[test]
fn absent_lanes_take_part_with_zero_cells() {
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

/// In workgroup-uniform control flow (not the calling lane's read): a threadgroup whose last
/// simdgroup is partial takes the absent lanes' cells as zero, and a whole one is exact.
#[test]
fn uniform_flow_takes_absent_lanes_as_zero_cells() {
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

/// One Metal reading: the case, its input, the lanes that run the multiply-add, a lane, its
/// float cells, and whether the case separates the zero's signs.
type Reading<'a> = (&'a str, Input, Lanes<'a>, u32, [f64; 2], bool);

/// Metal's readings of the owned no-fast-math cases (two passes, byte-identical), `(case, input,
/// part, lane, float cells)`; the half cells are the same values. With a `-0` accumulator and every
/// other term `-0`, a lane that does not take part shows its zero cell's sign: `+0`, not live or
/// absent, on the `b` side (rows 0-1 read `z`, rows 2-3 `-z`) and on the `a` side (columns 2-3 read
/// `z`, columns 6-7 `-z`). The cases marked `true` separate the signs: a `-0` zero cell would give
/// the opposite bits.
#[test]
fn the_zero_cell_is_plus_zero_in_metals_readings() {
    let all = |_: u32| true;
    let half = |l: u32| l < 16;
    let odd = |l: u32| l % 2 == 1;
    let eight = |l: u32| l < 8;
    let minus = Input::Signed { rows: -0.0 };
    let plus = Input::Signed { rows: 0.0 };
    let (p, n) = (0.0, -0.0);
    let readings: [Reading<'_>; 17] = [
        // Whole simdgroups: the signed zeros of an exact multiply-add.
        ("dd0e539e", minus, &all, 0, [n, n], false),
        ("dd0e539e", minus, &all, 4, [p, p], false),
        ("dd0e539e", minus, &all, 16, [n, n], false),
        ("6bd772f3", plus, &all, 0, [p, p], false),
        ("6bd772f3", plus, &all, 4, [n, n], false),
        ("7b445bf4", Input::Odd, &all, 0, [4.0, 4.0], false),
        ("7b445bf4", Input::Odd, &all, 9, [-4.0, -4.0], false),
        // Lanes 0-15 live, whatever lanes 16-31 loaded: the b side's zero is +0.
        ("4bd03887", minus, &half, 0, [p, p], true),
        ("4bd03887", minus, &half, 4, [n, n], true),
        ("b579838e", plus, &half, 0, [p, p], true),
        ("b579838e", plus, &half, 4, [n, n], true),
        // The odd lanes live: the a side's zero is +0.
        ("702c2457", Input::Odd, &odd, 1, [p, p], true),
        ("702c2457", Input::Odd, &odd, 9, [n, n], true),
        // A threadgroup of 16 (divergent and uniform flow): the absent lanes' zero is +0.
        ("041b4bad", minus, &half, 4, [n, n], true),
        ("101f0995", minus, &half, 0, [p, p], true),
        // A not-live zero times an absent zero, and two absent zeros: +0.
        ("258dd494", minus, &eight, 4, [p, p], false),
        ("df34a66f", minus, &eight, 4, [p, p], false),
    ];
    for (case, input, part, lane, want, separates) in readings {
        let m = inputs(input);
        let got = tile(&m, part, lane, 8, 0.0);
        assert_eq!(written(got), written(want), "{case} lane {lane}");
        if separates {
            let other = tile(&m, part, lane, 8, -0.0);
            assert_ne!(
                written(other),
                written(want),
                "{case} lane {lane} separates"
            );
        }
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

/// The zero-sign fixtures give Metal's bits on every emulating profile, at 32 threads and in
/// threadgroups that leave the simdgroup partial (16, 8; 16 with lanes 0-7 live), in divergent and
/// in uniform flow.
#[test]
fn the_emulated_tier_gives_the_zero_signs_metal_does() {
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

/// The zero-sign fixtures give Metal's bits on the subgroup tiers, whose threadgroups hold whole
/// subgroups: lanes 0-15 live, the odd lanes live, and their whole-simdgroup controls.
#[test]
fn the_subgroup_tiers_give_the_zero_signs_metal_does() {
    let mut dispatches = zero_sign_dispatches(32);
    dispatches.extend(zero_sign_dispatches(64));
    check_subgroups(&dispatches);
}

/// On the subgroup tiers a multiply-add some lanes of a simdgroup do not run (lanes 0-15, the odd
/// lanes; cells loaded before the branch, built from plain loads, or loaded inside it) reads their
/// cells as zero, with the ballot of the lanes that run it, and never shuffles from one of them
/// (the interpreter traps that); whole simdgroups under a divergent branch, and a uniform block,
/// are exact.
#[test]
fn the_subgroup_tiers_read_lanes_that_do_not_take_part_as_zero_cells() {
    let half: Lanes<'_> = &|g| g % 32 < 16;
    let odd: Lanes<'_> = &|g| g % 2 == 1;
    let all: Lanes<'_> = &|_| true;
    check_subgroups(&[
        ("dsm_whole_by_simdgroup", 64, Input::A, None, &|g| g >= 32),
        ("dsm_whole_by_buffer", 64, Input::A, Some([1, 0, 1]), &|g| {
            g / 32 != 1
        }),
        ("dsm_whole_by_buffer", 32, Input::B, Some([1, 0, 1]), all),
        ("dsm_half_live_loaded", 32, Input::A, None, half),
        ("dsm_half_live_loaded", 64, Input::B, None, half),
        ("dsm_half_live_loaded", 64, Input::Infinity, None, half),
        ("dsm_half_live_built", 64, Input::B, None, half),
        ("dsm_half_live_inside", 64, Input::A, None, half),
        ("dsm_odd_live_loaded", 32, Input::A, None, odd),
        ("dsm_odd_live_loaded", 64, Input::B, None, odd),
        ("dsm_uniform", 64, Input::A, None, all),
    ]);
}

/// A multiply-add under a branch on a simdgroup collective's result, after a loop whose exit
/// differs by lane (`engine/dsm2_collective_branch`: lane `L` leaves holding `simd_sum(1) =
/// 32 - L`, and lanes 0-15 run the multiply-add). The decision reads the workgroup analysis, in
/// which a collective's result is divergent, so the branch is divergent and every tier reads lanes
/// 16-31 as zero cells; a reading that took the branch for uniform would shuffle from inactive
/// lanes on the subgroup tiers.
#[test]
fn a_multiply_add_under_a_branch_on_a_collective_reads_the_lanes_that_left_as_zero() {
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
/// simdgroup runs the kernel's emulated-simdgroups variant, which gives Metal's answer in uniform
/// and in divergent control flow.
#[test]
fn the_emulated_variant_gives_metals_answer() {
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
/// `compiled` at each of `widths`, and check every thread against the rule ([`tile`]).
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
            let base = t & !31;
            let part = |l: u32| base + l < n && runs(base + l);
            let want = if runs(t) {
                let x = tile(&input, &part, t & 31, dim as usize, 0.0);
                [(x[0] as f32).to_bits(), (x[1] as f32).to_bits()]
            } else {
                [POISON; 2]
            };
            assert_eq!(cell, want, "{what} {stem} {lanes:?} thread {t}");
        }
    }
}

/// The rule for the AGX multiply-adds, which share the matrix one's lowering (Metal's AGX reading
/// is unmeasured), on every emulating profile, for the 8x8 form and the 4x4 one (each quadrant on
/// its own): run by simdgroup 1 of 2 (`t >= 32`) it is exact; run by lanes 0-15 or by the odd
/// lanes, the other lanes' cells are zero; in a threadgroup of 16 every thread of which runs it, so
/// are the absent lanes'.
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
/// at 64): a lane that does not run the AGX multiply-add reads as zero cells through the ballot of
/// the lanes that run it, in the 8x8 and the 4x4 form.
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
    for stem in ["lavapipe", "vk12-floor", "moltenvk-m3", "amd-rdna3-radv"] {
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
/// tier (a vertex or fragment subgroup narrower than 32 lanes, here SwiftShader's 4-lane vertex
/// stage) the Metal lanes at or above the subgroup's size are permanently inactive: they read as
/// `+0` cells, and neither the ballot's bit extract nor the shuffle names them (both are poison
/// there; the interpreter traps them). Each vertex is a subgroup of its own on the interpreter, so
/// every other owner lane is either not live (below the size) or beyond it (8- and 16-lane
/// subgroups), and the vertex's cells are the product of its own cells alone.
#[test]
fn a_narrow_subgroups_permanently_inactive_lanes_read_as_zero_cells() {
    let p = emulating("swiftshader");
    let input = inputs(Input::B);
    let alone: Lanes<'_> = &|l| l == 0;
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
        let x = tile(&input, alone, 0, dim as usize, 0.0);
        let want = [(x[0] as f32).to_bits(), (x[1] as f32).to_bits()];
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
                    assert_eq!(
                        [got[2 * v], got[2 * v + 1]],
                        want,
                        "{name} {lanes:?} vertex {v}"
                    );
                }
            }
        }
    }
}
