//! `matmul2d` lowered as Metal was measured to compute it, on the CPU interpreter.
//!
//! The owned fixtures `tns_mm_int` and `tns_mm_float` call MetalPerformancePrimitives' `matmul2d`
//! through every form the lowering models: each variant (function constant 0) is compiled for a
//! profile of each simdgroup tier, passes L3, and runs on the interpreter in both scheduling
//! orders, and every word of its output must be the reference's:
//!
//! - integers: `D = C + sum_{k < K} A(m, k) * B(k, n)` (or without `C` in `multiply` mode), with
//!   an element of `A` or `B` past its tensor's extents read as zero and every tile element
//!   written (18 cases);
//! - floats: `acc = C`, then `acc = fma(A(m, k), B(k, n), acc)` for `k` ascending, one rounding
//!   per step (56,064 elements in float, half, and bfloat); the same with relaxed precision, and
//!   from `acc = +0` in `multiply` mode, the fixture's descriptors respelled.
//!
//! The owned fixture `tns_mm_v2` is `tns_mm_float` built for macOS 27, whose header spells every
//! run in its `_v2` form, with a transposed B also as a cooperative right input and in threadgroup
//! memory: its float sums are the same reference's.
//!
//! The owned fixture `tns_mm_left` (built for macOS 27) gives the product a cooperative left input
//! filled by coordinate, over 16 x 32 x 32 and 32 x 32 x 32 with B transposed or not, in device or
//! threadgroup memory or a cooperative right input: its float sums are the same reference's.
//!
//! The owned fixture `tns_mm_fp8` (built for macOS 27) multiplies 8-bit floats (e4m3 and e5m2)
//! into float over 32 x 32 x 32 on one simdgroup, B transposed or not, accumulating C or not, A or
//! B staged in threadgroup memory: each element decodes exactly, subnormals kept (an e4m3 `0x7F`
//! reads 480, as measured on the M3 Max), and its float sums are the same reference's, on every
//! profile, one that lists `shaderFloat8` (the RTX 50 profile) included: the module declares no
//! `Float8EXT`.
//!
//! The owned fixtures `tns_init` and `tns_init_left` (`tns_mm_left`'s second kernel) construct a
//! cooperative tensor (a destination, a right input, or a left input) in a loop's body: each
//! construction zeroes the elements the thread holds, the second after the first set them, as on
//! Metal (and the gather kernels whose destinations Metal starts from zero in every round).
//!
//! The extents give whole tiles, partial tiles, and tiles that pass the tensors; the values are
//! seeded pseudo-random, the floats over a wide exponent range so that a sum in any other order
//! or with any other rounding differs. `llvm-as` assembles the fixtures; without it the tests skip
//! visibly.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::requirements::SpirvCapability;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs};
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::lower::select::Plan;
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_specialized,
    AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// Profiles and the subgroup widths their devices run: 32-lane subgroups (MoltenVK on Apple
/// silicon; RADV pinned to 32 lanes), and lavapipe's 8, where a kernel with cross-lane operations
/// emulates its simdgroups.
const MOLTENVK: (&str, &[SubgroupSize]) = ("moltenvk-m3", &[SubgroupSize::W32]);
const RADV: (&str, &[SubgroupSize]) = ("amd-rdna2-apu-radv", &[SubgroupSize::W32]);
const LAVAPIPE: (&str, &[SubgroupSize]) = ("lavapipe", &[SubgroupSize::W8]);
/// A device with its own 8-bit floats (`shaderFloat8`), on 32-lane subgroups.
const RTX50: (&str, &[SubgroupSize]) = ("nvidia-rtx50-linux", &[SubgroupSize::W32]);

#[test]
fn integer_products_are_metals_on_moltenvk() {
    integer_products_are_metals(MOLTENVK);
}

#[test]
fn integer_products_are_metals_on_radv() {
    integer_products_are_metals(RADV);
}

#[test]
fn integer_products_are_metals_on_lavapipe() {
    integer_products_are_metals(LAVAPIPE);
}

#[test]
fn float_products_are_metals_sequential_fused_sums_on_moltenvk() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Plain, MOLTENVK);
}

#[test]
fn float_products_are_metals_sequential_fused_sums_on_radv() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Plain, RADV);
}

#[test]
fn float_products_are_metals_sequential_fused_sums_on_lavapipe() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Plain, LAVAPIPE);
}

#[test]
fn v2_float_products_are_metals_sequential_fused_sums_on_moltenvk() {
    fused_sums_of(V2, FloatMode::Plain, MOLTENVK);
}

#[test]
fn v2_float_products_are_metals_sequential_fused_sums_on_radv() {
    fused_sums_of(V2, FloatMode::Plain, RADV);
}

#[test]
fn v2_float_products_are_metals_sequential_fused_sums_on_lavapipe() {
    fused_sums_of(V2, FloatMode::Plain, LAVAPIPE);
}

#[test]
fn left_cooperative_float_products_are_metals_sequential_fused_sums_on_moltenvk() {
    left_cooperative_fused_sums(MOLTENVK);
}

#[test]
fn left_cooperative_float_products_are_metals_sequential_fused_sums_on_radv() {
    left_cooperative_fused_sums(RADV);
}

#[test]
fn left_cooperative_float_products_are_metals_sequential_fused_sums_on_lavapipe() {
    left_cooperative_fused_sums(LAVAPIPE);
}

#[test]
fn relaxed_float_products_are_metals_sequential_fused_sums_on_moltenvk() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Relaxed, MOLTENVK);
}

#[test]
fn relaxed_float_products_are_metals_sequential_fused_sums_on_lavapipe() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Relaxed, LAVAPIPE);
}

#[test]
fn float_multiply_mode_sums_from_positive_zero_on_moltenvk() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Multiply, MOLTENVK);
}

#[test]
fn float_multiply_mode_sums_from_positive_zero_on_lavapipe() {
    float_products_are_metals_sequential_fused_sums(FloatMode::Multiply, LAVAPIPE);
}

#[test]
fn construction_zeroes_cooperative_tensors_on_moltenvk() {
    construction_zeroes_cooperative_tensors(MOLTENVK);
}

#[test]
fn construction_zeroes_cooperative_tensors_on_radv() {
    construction_zeroes_cooperative_tensors(RADV);
}

#[test]
fn construction_zeroes_cooperative_tensors_on_lavapipe() {
    construction_zeroes_cooperative_tensors(LAVAPIPE);
}

#[test]
fn construction_zeroes_cooperative_left_inputs_on_moltenvk() {
    construction_zeroes_cooperative_left_inputs(MOLTENVK);
}

#[test]
fn construction_zeroes_cooperative_left_inputs_on_lavapipe() {
    construction_zeroes_cooperative_left_inputs(LAVAPIPE);
}

#[test]
fn fp8_products_are_metals_sequential_fused_sums_on_moltenvk() {
    fp8_fused_sums(MOLTENVK);
}

#[test]
fn fp8_products_are_metals_sequential_fused_sums_on_radv() {
    fp8_fused_sums(RADV);
}

#[test]
fn fp8_products_are_metals_sequential_fused_sums_on_lavapipe() {
    fp8_fused_sums(LAVAPIPE);
}

#[test]
fn fp8_products_are_metals_sequential_fused_sums_on_rtx50() {
    fp8_fused_sums(RTX50);
}

fn bitcode(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

fn variant(v: u32) -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, v.to_le_bytes()),
    )
}

/// A seeded pseudo-random sequence (a 64-bit LCG's high words).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

/// Kernel `entry` of `bc` specialized to variant `v` for profile `stem`, checked through L3: its
/// plan and typed module.
fn kernel(bc: &[u8], entry: &str, v: u32, stem: &str) -> Kernel {
    let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
    let consts = variant(v);
    let what = format!("{entry} variant {v} on {stem}");
    let a = compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &consts,
        &p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
    let (plan, module) =
        typed_module_specialized(bc, entry, &p, &consts).unwrap_or_else(|e| panic!("{what}: {e}"));
    Kernel {
        what,
        plan,
        module,
        artifact: a,
    }
}

struct Kernel {
    what: String,
    plan: Plan,
    module: TypedModule,
    artifact: Arc<FunctionArtifact>,
}

impl Kernel {
    /// One threadgroup of `threads` threads with buffers `init`, at each subgroup width in
    /// `widths` and in both scheduling orders, which must agree: the final buffers.
    fn run(&self, widths: &[SubgroupSize], threads: u32, init: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut outs: Vec<Vec<Vec<u8>>> = Vec::new();
        for &w in widths {
            for order in [Order::Forward, Order::Reverse] {
                let cfg = Config {
                    subgroup: w,
                    order,
                    ..Config::default()
                };
                let out = run(&self.module, &cfg, resources(&self.plan, threads, init))
                    .unwrap_or_else(|e| panic!("{} {w:?}: {e:?}", self.what));
                outs.push(out.buffers);
            }
        }
        for o in &outs[1..] {
            assert!(o == &outs[0], "{}: the runs disagree", self.what);
        }
        outs.swap_remove(0)
    }
}

fn resources(plan: &Plan, threads: u32, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([threads, 1, 1]));
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
    let buffers = init
        .iter()
        .zip(0u32..)
        .map(|(b, index)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

fn le_bytes<T: Copy, const N: usize>(xs: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    xs.iter().flat_map(|&x| f(x)).collect()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A descriptor: `(m, n, k, transpose_left, transpose_right, accumulate)`.
type Desc = (u32, u32, u32, bool, bool, bool);

/// `tns_mm_int`'s variants: the descriptor and the scope's simdgroups.
const INT_VARIANTS: [(Desc, u32); 11] = [
    ((16, 16, 32, false, false, true), 1),
    ((16, 16, 32, false, true, true), 1),
    ((16, 16, 32, true, false, true), 1),
    ((16, 16, 32, true, true, true), 1),
    ((16, 16, 32, false, false, false), 1),
    ((16, 16, 32, false, false, true), 1),
    ((16, 16, 32, false, false, true), 1),
    ((16, 16, 32, false, false, true), 1),
    ((16, 16, 32, false, false, true), 1),
    ((32, 32, 32, false, true, true), 1),
    ((16, 64, 16, false, false, true), 2),
];

/// `tns_mm_int`'s output stride.
const STRIDE: usize = 64;

/// The integer reference: `out` as `tns_mm_int` leaves it.
fn int_reference(
    (m, n, k, tl, tr, acc): Desc,
    a: &[i8],
    b: &[i8],
    c0: &[i32],
    e: [i32; 4],
    out: &mut [i32],
) {
    // Tensor index (i0, i1) of extents (e0, e1) reads element i1 * e0 + i0, or zero past them.
    let at = |t: &[i8], (e0, e1): (i32, i32), (i0, i1): (u32, u32)| -> i32 {
        if (i0 as i32) < e0 && (i1 as i32) < e1 {
            i32::from(t[(i1 as i32 * e0 + i0 as i32) as usize])
        } else {
            0
        }
    };
    for row in 0..m {
        for col in 0..n {
            let mut d = if acc {
                c0[row as usize * STRIDE + col as usize]
            } else {
                0
            };
            for kk in 0..k {
                let x = at(a, (e[0], e[1]), if tl { (row, kk) } else { (kk, row) });
                let y = at(b, (e[2], e[3]), if tr { (kk, col) } else { (col, kk) });
                d = d.wrapping_add(x * y);
            }
            out[row as usize * STRIDE + col as usize] = d;
        }
    }
}

/// `int8_t` products into `int32_t` are Metal's on `profile` for every form the fixture has, over
/// whole tiles, partial tiles, tiles that pass the tensors, and an inner extent other than `k`.
fn integer_products_are_metals((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode("tns_mm_int") else {
        return;
    };
    let mut rng = Rng(0x7a5_1e5);
    for (v, &(d, simdgroups)) in INT_VARIANTS.iter().enumerate() {
        let kernels = [(kernel(&bc, "tns_mm_int", v as u32, stem), widths)];
        let (m, n, k, tl, tr, _) = d;
        // The tensors' extents: whole tiles, partial tiles, tiles past the tensors, and an inner
        // extent other than k. Each tensor's element count is a whole number of words: the
        // fixture copies only a tensor's own bytes to threadgroup memory, and an `int8_t` load
        // there reads its word, whose other bytes the interpreter traps on reading uninitialized.
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let shapes = [
            (m, k, n, k),
            (m - 4, k - 9, n - 4, k - 9),
            (m + 4, k + 6, n + 2, k + 6),
            (m, k - 4, n, k - 12),
        ];
        for (sm, sk, sn, skb) in shapes {
            // A is M x K (extents (K, M)), or K x M transposed (extents (M, K)).
            let ea = if tl { (sm, sk) } else { (sk, sm) };
            let eb = if tr { (skb, sn) } else { (sn, skb) };
            let e = [ea.0, ea.1, eb.0, eb.1];
            for (x, y) in [ea, eb] {
                assert!(x * y <= 2048 && x * y % 4 == 0, "{ea:?} {eb:?}");
            }
            let a: Vec<i8> = (0..2048).map(|_| rng.next() as i8).collect();
            let b: Vec<i8> = (0..2048).map(|_| rng.next() as i8).collect();
            let c0: Vec<i32> = (0..STRIDE * STRIDE)
                .map(|_| rng.next() as i32 >> 8)
                .collect();
            let fill = vec![0x5a5a_5a5a_i32; STRIDE * STRIDE];
            let mut want = fill.clone();
            int_reference(d, &a, &b, &c0, e, &mut want);
            let init = [
                le_bytes(&a, |x: i8| x.to_le_bytes()),
                le_bytes(&b, |x: i8| x.to_le_bytes()),
                le_bytes(&c0, i32::to_le_bytes),
                le_bytes(&fill, i32::to_le_bytes),
                le_bytes(&e, i32::to_le_bytes),
            ];
            for (kernel, widths) in &kernels {
                let out = kernel.run(widths, 32 * simdgroups, &init);
                let got: Vec<i32> = words(&out[3]).into_iter().map(|w| w as i32).collect();
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g,
                        w,
                        "{} extents {e:?}: out[{i}] (row {}, column {})",
                        kernel.what,
                        i / STRIDE,
                        i % STRIDE
                    );
                }
            }
        }
    }
}

/// An element format of `tns_mm_float`'s inputs.
#[derive(Clone, Copy, Debug)]
enum Format {
    Float,
    Half,
    BFloat,
}

impl Format {
    fn bytes(self) -> usize {
        match self {
            Format::Float => 4,
            Format::Half | Format::BFloat => 2,
        }
    }

    /// A random finite, normal value's bits, its exponent within `spread` binades of 1.
    fn random(self, rng: &mut Rng, spread: i32) -> u32 {
        let r = rng.next();
        let sign = r >> 31;
        let e = (r >> 16) as i32 % (2 * spread + 1) - spread;
        let frac = rng.next();
        match self {
            Format::Float => (sign << 31) | (((127 + e) as u32) << 23) | (frac >> 9),
            Format::Half => (sign << 15) | (((15 + e) as u32) << 10) | (frac >> 22),
            Format::BFloat => (sign << 15) | (((127 + e) as u32) << 7) | (frac >> 25),
        }
    }

    /// The value of bits `x` as a float (exactly).
    fn widen(self, x: u32) -> f32 {
        match self {
            Format::Float => f32::from_bits(x),
            Format::BFloat => f32::from_bits(x << 16),
            Format::Half => {
                let sign = if x >> 15 == 1 { -1.0 } else { 1.0 };
                let e = ((x >> 10) & 0x1f) as i32;
                let f = (x & 0x3ff) as f32;
                assert!(e != 0x1f, "a half infinity or NaN");
                if e == 0 {
                    sign * f * 2f32.powi(-24)
                } else {
                    sign * (1.0 + f / 1024.0) * 2f32.powi(e - 15)
                }
            }
        }
    }

    fn store(self, x: u32) -> Vec<u8> {
        match self {
            Format::Float => x.to_le_bytes().to_vec(),
            Format::Half | Format::BFloat => (x as u16).to_le_bytes().to_vec(),
        }
    }

    /// The bits of `-0`.
    fn negative_zero(self) -> u32 {
        match self {
            Format::Float => 0x8000_0000,
            Format::Half | Format::BFloat => 0x8000,
        }
    }
}

/// The descriptor flags `tns_mm_float` runs with: as compiled, or with its descriptors respelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatMode {
    /// `multiply_accumulate`, as compiled.
    Plain,
    /// `multiply_accumulate` with relaxed precision: Metal's sums are the plain ones.
    Relaxed,
    /// `multiply`: the sum starts from `+0`, not from `C`.
    Multiply,
}

impl FloatMode {
    /// The fixture's descriptor spellings (`transpose_left, transpose_right, relaxed_precision,
    /// mode`) respelled for this mode, the plain and the transposed descriptor alike.
    fn respellings(self) -> &'static [(&'static str, &'static str)] {
        match self {
            FloatMode::Plain => &[],
            FloatMode::Relaxed => &[
                ("i8 0, i8 0, i8 0, i32 1", "i8 0, i8 0, i8 1, i32 1"),
                ("i8 0, i8 1, i8 0, i32 1", "i8 0, i8 1, i8 1, i32 1"),
            ],
            FloatMode::Multiply => &[
                ("i8 0, i8 0, i8 0, i32 1", "i8 0, i8 0, i8 0, i32 0"),
                ("i8 0, i8 1, i8 0, i32 1", "i8 0, i8 1, i8 0, i32 0"),
            ],
        }
    }
}

/// `tns_mm_float`'s variants: format, transposed B.
const FLOAT_VARIANTS: [(Format, bool); 10] = [
    (Format::Float, false),
    (Format::Half, false),
    (Format::BFloat, false),
    (Format::Float, true),
    (Format::Half, true),
    (Format::BFloat, true),
    (Format::Float, false),
    (Format::Half, false),
    (Format::BFloat, false),
    (Format::Float, false),
];

/// `tns_mm_v2`'s variants: `tns_mm_float`'s, then a transposed B as a cooperative right input and
/// in threadgroup memory, each in float, half, and bfloat.
const V2_VARIANTS: [(Format, bool); 16] = [
    (Format::Float, false),
    (Format::Half, false),
    (Format::BFloat, false),
    (Format::Float, true),
    (Format::Half, true),
    (Format::BFloat, true),
    (Format::Float, false),
    (Format::Half, false),
    (Format::BFloat, false),
    (Format::Float, false),
    (Format::Float, true),
    (Format::Half, true),
    (Format::BFloat, true),
    (Format::Float, true),
    (Format::Half, true),
    (Format::BFloat, true),
];

/// A float-sums fixture: its name (and its kernel's) and its variants.
type FloatFixture = (&'static str, &'static [(Format, bool)]);

const FLOAT: FloatFixture = ("tns_mm_float", &FLOAT_VARIANTS);
const V2: FloatFixture = ("tns_mm_v2", &V2_VARIANTS);

/// Float, half, and bfloat products into float are Metal's sequential fused sums on `profile`, with
/// B in device or threadgroup memory or a cooperative right input, in `mode`. In `multiply` mode
/// row 0 of A is `-0`, so its sums start from `+0` only if they end `+0`.
fn float_products_are_metals_sequential_fused_sums(
    mode: FloatMode,
    profile: (&str, &[SubgroupSize]),
) {
    fused_sums_of(FLOAT, mode, profile);
}

/// [`float_products_are_metals_sequential_fused_sums`] over `fixture`'s variants.
fn fused_sums_of(
    (fixture, variants): FloatFixture,
    mode: FloatMode,
    (stem, widths): (&str, &[SubgroupSize]),
) {
    let bc = if mode == FloatMode::Plain {
        bitcode(fixture)
    } else {
        respelled(fixture, mode.respellings())
    };
    let Some(bc) = bc else {
        return;
    };
    let mut rng = Rng(0xf1_0a7);
    for (v, &(format, tr)) in variants.iter().enumerate() {
        let kernels = [(kernel(&bc, fixture, v as u32, stem), widths)];
        for spread in [3, 12] {
            // One block each: A 16 x 32, B 32 x 16 (or 16 x 32 transposed), C 16 x 16.
            let mut a: Vec<u32> = (0..512).map(|_| format.random(&mut rng, spread)).collect();
            if mode == FloatMode::Multiply {
                a[..32].fill(format.negative_zero());
            }
            let b: Vec<u32> = (0..512).map(|_| format.random(&mut rng, spread)).collect();
            let c: Vec<f32> = (0..256)
                .map(|_| f32::from_bits(Format::Float.random(&mut rng, spread)))
                .collect();
            let mut want = vec![0u32; 256];
            for row in 0..16 {
                for col in 0..16 {
                    let mut acc = if mode == FloatMode::Multiply {
                        0.0
                    } else {
                        c[row * 16 + col]
                    };
                    for k in 0..32 {
                        let x = format.widen(a[row * 32 + k]);
                        let y = format.widen(if tr { b[col * 32 + k] } else { b[k * 16 + col] });
                        acc = x.mul_add(y, acc);
                    }
                    want[row * 16 + col] = acc.to_bits();
                }
            }
            let init = [
                a.iter().flat_map(|&x| format.store(x)).collect::<Vec<u8>>(),
                b.iter().flat_map(|&x| format.store(x)).collect(),
                le_bytes(&c, f32::to_le_bytes),
                vec![0xa5; 256 * 4],
                le_bytes(&[0u32, 0, 0, 0], u32::to_le_bytes),
            ];
            assert_eq!(init[0].len(), 512 * format.bytes());
            for (kernel, widths) in &kernels {
                let out = kernel.run(widths, 32, &init);
                let got = words(&out[3]);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g,
                        w,
                        "{} ({mode:?}, {format:?}, spread {spread}): out[{i}] {} != {}",
                        kernel.what,
                        f32::from_bits(*g),
                        f32::from_bits(*w)
                    );
                }
            }
        }
    }
}

/// `tns_mm_left`'s variant `v`: the format of A and B, M, and whether B is transposed (B's place,
/// device or threadgroup memory or a cooperative right input, `v % 12 / 4`, reads the same).
fn left_variant(v: u32) -> (Format, usize, bool) {
    let format = [Format::Float, Format::Half, Format::BFloat][(v / 12) as usize];
    let w = v % 12;
    (format, if w % 4 < 2 { 16 } else { 32 }, w % 2 == 1)
}

/// Float, half, and bfloat products into float with a cooperative left input are Metal's
/// sequential fused sums on `profile`, B in device or threadgroup memory or a cooperative right
/// input, transposed or not, for M of 16 and 32 (`tns_mm_left`, one tile).
fn left_cooperative_fused_sums((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode("tns_mm_left") else {
        return;
    };
    let mut rng = Rng(0x1ef7);
    for v in 0..36 {
        let (format, m, tr) = left_variant(v);
        let kernel = kernel(&bc, "tns_mm_left", v, stem);
        for spread in [3, 12] {
            // One block each: A m x 32, B 32 x 32 (its rows B's columns when transposed), C m x 32.
            let a: Vec<u32> = (0..m * 32)
                .map(|_| format.random(&mut rng, spread))
                .collect();
            let b: Vec<u32> = (0..1024).map(|_| format.random(&mut rng, spread)).collect();
            let c: Vec<f32> = (0..m * 32)
                .map(|_| f32::from_bits(Format::Float.random(&mut rng, spread)))
                .collect();
            let mut want = vec![0u32; m * 32];
            for row in 0..m {
                for col in 0..32 {
                    let mut acc = c[row * 32 + col];
                    for k in 0..32 {
                        let x = format.widen(a[row * 32 + k]);
                        let y = format.widen(if tr { b[col * 32 + k] } else { b[k * 32 + col] });
                        acc = x.mul_add(y, acc);
                    }
                    want[row * 32 + col] = acc.to_bits();
                }
            }
            let init = [
                a.iter().flat_map(|&x| format.store(x)).collect::<Vec<u8>>(),
                b.iter().flat_map(|&x| format.store(x)).collect(),
                le_bytes(&c, f32::to_le_bytes),
                vec![0xa5; m * 32 * 4],
                le_bytes(&[0u32, 0, 0, 0], u32::to_le_bytes),
            ];
            let out = kernel.run(widths, 32, &init);
            let got = words(&out[3]);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{} ({format:?}, spread {spread}): out[{i}] {} != {}",
                    kernel.what,
                    f32::from_bits(*g),
                    f32::from_bits(*w)
                );
            }
        }
    }
}

/// An 8-bit float format: `metal_fp8_e4m3_format` (4 exponent bits, bias 7, no infinity or NaN)
/// or `metal_fp8_e5m2_format` (5 exponent bits, bias 15).
#[derive(Clone, Copy, Debug)]
enum Fp8 {
    E4M3,
    E5M2,
}

impl Fp8 {
    /// A random finite byte: any e4m3 byte (Metal reads `0x7F` and `0xFF` as ±480, measured), an
    /// e5m2 byte below the all-ones exponent.
    fn random(self, rng: &mut Rng) -> u8 {
        loop {
            let b = (rng.next() >> 24) as u8;
            if matches!(self, Fp8::E4M3) || (b >> 2) & 0x1f != 0x1f {
                return b;
            }
        }
    }

    /// The value of byte `b` as a float, exactly, subnormals kept: as Metal reads it (measured on
    /// the M3 Max).
    fn widen(self, b: u8) -> f32 {
        let (mantissa, bias) = match self {
            Fp8::E4M3 => (3, 7),
            Fp8::E5M2 => (2, 15),
        };
        let sign = if b >> 7 == 1 { -1.0 } else { 1.0 };
        let e = i32::from((b & 0x7f) >> mantissa);
        let scale = (1u32 << mantissa) as f32;
        let m = f32::from(b & ((1u8 << mantissa) - 1)) / scale;
        match e {
            0 => sign * m * 2f32.powi(1 - bias),
            _ => sign * (1.0 + m) * 2f32.powi(e - bias),
        }
    }
}

/// `tns_mm_fp8`'s variant `v`: its format, whether B is transposed, and whether the product
/// accumulates C (`multiply_accumulate`) or sums from `+0` (`multiply`).
fn fp8_variant(v: u32) -> (Fp8, bool, bool) {
    let format = if v < 7 { Fp8::E4M3 } else { Fp8::E5M2 };
    let (transposed, accumulates) = match v % 7 {
        0 | 4 => (false, true),
        2 => (false, false),
        3 => (true, false),
        _ => (true, true),
    };
    (format, transposed, accumulates)
}

/// Products of 8-bit floats into float are Metal's sequential fused sums on `profile`, of the
/// exact decode of each byte, A and B in device memory or either in threadgroup memory
/// (`tns_mm_fp8`, one 32 x 32 x 32 tile), by the integer decode on every profile: no module
/// declares `Float8EXT`, and no sum is a NaN.
fn fp8_fused_sums((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode("tns_mm_fp8") else {
        return;
    };
    let float8 = SpirvCapability::of(crate::spv::Capability::Float8EXT);
    let mut rng = Rng(0xf8e4);
    let mut nans = 0;
    for v in 0..14 {
        let (format, transposed, accumulates) = fp8_variant(v);
        let kernel = kernel(&bc, "tns_mm_fp8", v, stem);
        let artifact = &kernel.artifact;
        assert!(
            !artifact.requirements.spirv_capabilities.contains(&float8),
            "{}: {:?}",
            kernel.what,
            artifact.requirements.spirv_capabilities
        );
        let a: Vec<u8> = (0..1024).map(|_| format.random(&mut rng)).collect();
        let b: Vec<u8> = (0..1024).map(|_| format.random(&mut rng)).collect();
        let c: Vec<f32> = (0..1024)
            .map(|_| f32::from_bits(Format::Float.random(&mut rng, 6)))
            .collect();
        let mut want = vec![0u32; 1024];
        for row in 0..32 {
            for col in 0..32 {
                let mut acc = if accumulates { c[row * 32 + col] } else { 0.0 };
                for k in 0..32 {
                    let x = format.widen(a[row * 32 + k]);
                    let at = if transposed {
                        col * 32 + k
                    } else {
                        k * 32 + col
                    };
                    acc = x.mul_add(format.widen(b[at]), acc);
                }
                want[row * 32 + col] = acc.to_bits();
            }
        }
        let init = [
            a,
            b,
            le_bytes(&c, f32::to_le_bytes),
            vec![0xa5; 4096],
            le_bytes(&[0u32, 0, 0, 0], u32::to_le_bytes),
        ];
        let out = kernel.run(widths, 32, &init);
        let got = words(&out[3]);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if f32::from_bits(*w).is_nan() {
                assert!(
                    f32::from_bits(*g).is_nan(),
                    "{} ({format:?}): out[{i}] {} is no NaN",
                    kernel.what,
                    f32::from_bits(*g)
                );
                nans += 1;
                continue;
            }
            assert_eq!(
                g,
                w,
                "{} ({format:?}): out[{i}] {} != {}",
                kernel.what,
                f32::from_bits(*g),
                f32::from_bits(*w)
            );
        }
    }
    // Metal's e4m3 has no NaN, and the finite inputs make none.
    assert_eq!(nans, 0, "{stem}: {nans} NaN sums");
}

/// `fixture` with every spelling of a descriptor's fields `from` respelled `to`, for each pair (in
/// the descriptor constants and in the helpers' by-value arguments alike), assembled; `None`
/// without `llvm-as`.
fn respelled(fixture: &str, pairs: &[(&str, &str)]) -> Option<Vec<u8>> {
    let mut text =
        std::fs::read_to_string(engine_dir().join(format!("{fixture}.ll"))).expect("fixture");
    for (from, to) in pairs {
        assert!(text.contains(from), "{from}");
        text = text.replace(from, to);
    }
    let what = pairs
        .iter()
        .map(|(_, to)| *to)
        .collect::<Vec<_>>()
        .join(" | ");
    let scratch = Scratch::new(&format!("m2v-matmul2d-{}", what.len()));
    let dir = scratch.path();
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{what}: {why}"),
    }
}

/// What no probe measured is refused as `Unsupported{Tensors, NotImplemented}`, naming it: a tile
/// shape whose layout is not measured.
#[test]
fn unmeasured_forms_are_refused_naming_them() {
    let cases = [(
        "i32 16, i32 16, i32 32",
        "i32 16, i32 64, i32 32",
        "16x64x32",
    )];
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    for (from, to, why) in cases {
        let Some(bc) = respelled("tns_mm_float", &[(from, to)]) else {
            return;
        };
        let e = compile_bitcode(
            &bc,
            EntryRef::Name("tns_mm_float"),
            &variant(0),
            &p,
            &LoweringInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        )
        .err()
        .unwrap_or_else(|| panic!("{to}: translated"));
        let Error::Unsupported(u) = &e else {
            panic!("{to}: {e}");
        };
        assert_eq!(u.feature, Feature::Tensors, "{to}: {e}");
        assert_eq!(u.needs, Needs::NotImplemented, "{to}: {e}");
        let named = u
            .symbol
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        assert!(named.contains(why), "{to}: {named}");
        assert!(u.location.is_some(), "{to}: {e}");
    }
}

/// `tns_init`'s variants (destinations, then right inputs) and their scopes' simdgroups.
const INIT_VARIANTS: [(u32, u32); 8] = [
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 4),
    (5, 1),
    (6, 1),
    (7, 1),
];

/// Constructing a cooperative tensor, a destination or a right input, zeroes the elements the
/// thread holds, as Metal does (`tns_init`, the M3 Max and an M4): a tensor constructed in a loop's
/// body reads zero in every round, the second after the first set each element `i` to `i + 1`, and
/// reads back what the round set.
fn construction_zeroes_cooperative_tensors((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode("tns_init") else {
        return;
    };
    constructions_zero(&bc, "tns_init", &INIT_VARIANTS, (stem, widths));
}

/// The same of a cooperative left input, 16 x 32 x 32 and 32 x 32 x 32 in float, half, and bfloat,
/// as Metal does (`tns_init_left`, the M3 Max).
fn construction_zeroes_cooperative_left_inputs((stem, widths): (&str, &[SubgroupSize])) {
    let Some(bc) = bitcode("tns_mm_left") else {
        return;
    };
    let variants: Vec<(u32, u32)> = (0..6).map(|v| (v, 1)).collect();
    constructions_zero(&bc, "tns_init_left", &variants, (stem, widths));
}

/// Kernel `entry` of `bc`, `tns_init`'s or its like, constructs zeroed tensors in every one of its
/// `variants` (each with its scope's simdgroups) on `stem`.
fn constructions_zero(
    bc: &[u8],
    entry: &str,
    variants: &[(u32, u32)],
    (stem, widths): (&str, &[SubgroupSize]),
) {
    const ROUNDS: u32 = 2;
    // `tns_init`'s words per thread and round.
    const HELD: usize = 64;
    const FILL: u32 = 0xa5a5_a5a5;
    for &(v, simdgroups) in variants {
        let kernel = kernel(bc, entry, v, stem);
        let threads = 32 * simdgroups;
        let rows = (ROUNDS * threads) as usize;
        let init = [
            le_bytes(&vec![FILL; rows * (HELD + 1)], u32::to_le_bytes),
            le_bytes(&[ROUNDS], u32::to_le_bytes),
        ];
        let out = words(&kernel.run(widths, threads, &init)[0]);
        for r in 0..ROUNDS {
            for t in 0..threads {
                let at = (r * threads + t) as usize;
                let held = &out[at * HELD..][..HELD];
                let cap = held.iter().take_while(|&&w| w != FILL).count();
                let what = format!("{} round {r} thread {t}", kernel.what);
                assert!(cap > 0, "{what}: no element");
                assert!(held[cap..].iter().all(|&w| w == FILL), "{what}: {held:x?}");
                assert!(held[..cap].iter().all(|&w| w == 0), "{what}: {held:x?}");
                let want: u32 = (0..cap as u32).map(|i| r * 32 + i + 1).sum();
                assert_eq!(out[rows * HELD + at], want, "{what}: the sum read back");
            }
        }
    }
}
