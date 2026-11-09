//! MSL's integer `median3`, which Metal's compiler spells `air.median3.<s|u>.<type>`.
//!
//! The owned fixture `engine/kernel_integer_median3` calls it on every integer width, signed and
//! unsigned (`i8`, `i16`, `i32`, `i64`), and on `<4 x i8>`, `<2 x i16>`, `<4 x i32>` and
//! `<2 x i64>`: it translates on every committed profile with no precision note, and runs on the
//! CPU interpreter against the middle one of the three operands over 64 threads of edge words
//! (every order of a width's `INT_MIN`, `INT_MAX` and all ones, ties, all three alike) and mixed
//! ones, on `moltenvk-m3` with `shaderInt64` (64-bit containers) and without it (two 32-bit
//! words), the two lowerings a 64-bit operand takes. The authored cases `integer-median3-*` run
//! the same kernel on Metal and on Vulkan devices over the same words. `llvm-as` and `xcrun
//! metal-as` assemble; a missing tool skips visibly. No GPU is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, metal_assemble, typed_module,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

const STEM: &str = "kernel_integer_median3";
const ENTRY: &str = "median3_ops";
const THREADS: usize = 64;
const WORDS: usize = 32;

/// The committed `moltenvk-m3` description with `shaderInt64` (64-bit containers), and without
/// it (two 32-bit words).
fn profiles() -> [(&'static str, DeviceProfile); 2] {
    [("int64", true), ("two-word", false)].map(|(label, int64)| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        if !int64 {
            assert!(desc.features.remove(&VkFeature::ShaderInt64));
        }
        let p = Device::new(desc, DevicePolicy::default())
            .expect("device")
            .profile()
            .clone();
        assert_eq!(p.fields().leaf.int64, int64);
        (label, p)
    })
}

/// splitmix64's output for `x`.
fn splitmix(x: u64) -> u64 {
    let z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Thread t's three words `(a, b, c)`, the authored cases' too. For each width `w` of 8, 16, 32
/// and 64, with `lo` its `INT_MIN` (`1 << (w - 1)`), `hi` its `INT_MAX` (`lo - 1`) and `all` all
/// ones (`-1`, `UINT_MAX`): threads `6i .. 6i + 6` hold the six orders of `(lo, hi, all)` for
/// width `i`, and threads `24 + 4i .. 28 + 4i` the ties `(lo, lo, hi)`, `(hi, all, hi)`,
/// `(all, lo, all)` and `(lo, lo, lo)`; threads 40 to 43 hold `(0, 0, 0)`, `(all, all, all)`,
/// `(0, 1, all)` and `(1, 0, 0)` of 64 bits; threads 44 to 63 hold splitmix64 of
/// `0x1234567 (3t + k + 1)` for `k` of 0, 1 and 2.
fn triple(t: usize) -> [u64; 3] {
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let all = u64::MAX;
    let edges = |i: usize| {
        let lo = 1u64 << ([8, 16, 32, 64][i] - 1);
        [lo, lo - 1, all]
    };
    match t {
        0..24 => {
            let e = edges(t / 6);
            ORDERS[t % 6].map(|k| e[k])
        }
        24..40 => {
            let [lo, hi, all] = edges((t - 24) / 4);
            [[lo, lo, hi], [hi, all, hi], [all, lo, all], [lo, lo, lo]][(t - 24) % 4]
        }
        40..44 => [[0, 0, 0], [all, all, all], [0, 1, all], [1, 0, 0]][t - 40],
        _ => [0, 1, 2].map(|k| splitmix(0x123_4567 * (3 * t as u64 + k + 1))),
    }
}

/// The middle one of the `bits`-wide fields at `shift` of the three words, read signed or
/// unsigned, as those `bits` bits.
fn median(t: [u64; 3], shift: u32, bits: u32, signed: bool) -> u64 {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    };
    let mut v = t.map(|x| {
        let f = (x >> shift) & mask;
        if signed {
            i128::from(((f << (64 - bits)) as i64) >> (64 - bits))
        } else {
            i128::from(f)
        }
    });
    v.sort_unstable();
    (v[1] as u64) & mask
}

/// `x`, `bits` wide, sign-extended to 32 bits.
fn sext32(x: u64, bits: u32) -> u32 {
    (((x << (64 - bits)) as i64) >> (64 - bits)) as u32
}

/// What `median3_ops` writes to a thread's 32 words, from the MSL.
fn median3_words(t: [u64; 3]) -> [u32; WORDS] {
    let not = t.map(|x| !x);
    let mut r = [0u32; WORDS];
    r[0] = sext32(median(t, 0, 8, true), 8);
    r[1] = median(t, 0, 8, false) as u32;
    r[2] = sext32(median(t, 0, 16, true), 16);
    r[3] = median(t, 0, 16, false) as u32;
    r[4] = median(t, 0, 32, true) as u32;
    r[5] = median(t, 0, 32, false) as u32;
    for (at, signed) in [(6, true), (8, false)] {
        let m = median(t, 0, 64, signed);
        r[at] = m as u32;
        r[at + 1] = (m >> 32) as u32;
    }
    for l in 0..4 {
        r[10 + l] = sext32(median(t, 8 * l as u32, 8, true), 8);
    }
    for l in 0..2 {
        r[14 + l] = median(t, 16 * l as u32, 16, false) as u32;
    }
    for (at, signed) in [(16, true), (20, false)] {
        r[at] = median(t, 0, 32, signed) as u32;
        r[at + 1] = median(t, 32, 32, signed) as u32;
        r[at + 2] = median(not, 0, 32, signed) as u32;
        r[at + 3] = median(not, 32, 32, signed) as u32;
    }
    for (at, signed) in [(24, true), (28, false)] {
        for (k, words) in [t, not].into_iter().enumerate() {
            let m = median(words, 0, 64, signed);
            r[at + 2 * k] = m as u32;
            r[at + 2 * k + 1] = (m >> 32) as u32;
        }
    }
    r
}

/// One workgroup of `threads` threads running `m` with `buffers` bound at slots 0, 1, ...: the
/// bytes of every buffer afterwards, identical in both scheduling orders.
fn run_kernel(
    what: &str,
    plan: &Plan,
    m: &TypedModule,
    threads: u32,
    buffers: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, threads), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let res = Resources {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, bytes)| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: i as u32,
                }),
                address: None,
                bytes: bytes.clone(),
            })
            .collect(),
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{what} {order:?}: {e:?}"));
        outs.push(out.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs.swap_remove(0)
}

/// The fixture under each available assembler: `(spelling, bitcode)`.
fn spellings() -> Vec<(&'static str, Vec<u8>)> {
    let mut out = Vec::new();
    for (what, assembled) in [
        (
            "llvm-as",
            llvm_assemble(&engine_dir().join(format!("{STEM}.ll"))),
        ),
        (
            "metal-as",
            metal_assemble(&engine_dir().join(format!("{STEM}.air.ll"))),
        ),
    ] {
        match assembled {
            Ok(bc) => out.push((what, bc)),
            Err(AssembleError::Missing(why)) => eprintln!("SKIPPED {what}: {why}"),
            Err(AssembleError::Rejected(why)) => panic!("{what} rejected {STEM}: {why}"),
        }
    }
    out
}

/// `median3_ops` translates, links and validates on every committed profile, in both spellings,
/// with no precision note: every answer is exact.
#[test]
fn the_integer_median3_fixture_translates_everywhere() {
    let profiles = committed_profiles().expect("profiles");
    for (what, bc) in spellings() {
        for (stem, p) in &profiles {
            let a = compile(&bc, ENTRY, p).unwrap_or_else(|e| panic!("{what} on {stem}: {e:?}"));
            assert_eq!(a.precision, [], "{what} on {stem}");
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
        }
    }
}

/// `median3_ops` over its 64 threads' words answers the middle one of the three operands in
/// every word, signed and unsigned, at every width and lane, with and without `shaderInt64`, in
/// both spellings.
#[test]
fn the_integer_median3_fixture_answers_the_middle_one() {
    let x: Vec<u8> = (0..THREADS)
        .flat_map(triple)
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let want: Vec<u32> = (0..THREADS)
        .flat_map(|t| median3_words(triple(t)))
        .collect();
    for (what, bc) in spellings() {
        for (stem, p) in profiles() {
            let (plan, m) =
                typed_module(&bc, ENTRY, &p).unwrap_or_else(|e| panic!("{what} {stem}: {e:?}"));
            let out = run_kernel(
                &format!("{what} {stem}"),
                &plan,
                &m,
                THREADS as u32,
                &[x.clone(), vec![0xAB; THREADS * WORDS * 4]],
            );
            let got: Vec<u32> = out[1]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{what} {stem}: thread {} word {} ({:#x?})",
                    i / WORDS,
                    i % WORDS,
                    triple(i / WORDS)
                );
            }
        }
    }
}
