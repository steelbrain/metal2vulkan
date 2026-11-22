//! Metal's `float` multiply flushes a product whose exact value lies below `2^-126` to the zero of
//! its sign before rounding it (tininess before rounding), on vector and scalar lanes, with and
//! without fast math, and as the first product of its `dot` (the module's `dot` is the device's
//! `OpDot`): the owned fixtures
//! `tiny_products_flush` and `tiny_products_flush_fast`, whose words on an Apple M3 Max under
//! macOS 27.0.1 are `tiny_products_m3.txt` (the authored cases of the fixtures,
//! `float-products-flushed-before-rounding` and its fast-math twin).
//! A Vulkan device rounds first, so an exact product within half a unit below `2^-126` answers
//! `2^-126` there, and the module takes the device's multiply (announced once per device,
//! `float-tininess-after-rounding`).
//!
//! The CPU SPIR-V interpreter rounds first and keeps denormals, as a device may, so on each
//! committed profile (one per fused multiply-add route) it shows what the device answers. `llvm-as` assembles the fixtures; without
//! it the tests skip visibly.

use crate::device::{DevicePolicy, DeviceProfile};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles_with;

/// Metal's words (`tiny_products_m3.txt`).
const WORDS: &str = include_str!("tiny_products_m3.txt");

/// The fixtures' entries: without and with fast math.
const ENTRIES: [&str; 2] = ["tiny_products_flush", "tiny_products_flush_fast"];

/// A thread past the authored cases', its words measured on the same M3 Max under macOS 27.0.1 by
/// the fixtures' own source: `(2 - 2^-23) · 2^-126`, a normal `float`, times `±0.5` on each lane,
/// whose exact product `±(2^-126 - 2^-150)` a device that rounds first ties to `±2^-126`, where
/// Metal answers the zero of its sign.
const HALF_A_NORMAL: &str = concat!(
    "00ffffff 80ffffff 3f000000 00ffffff 3f000000 3f000000 00ffffff bf000000 ",
    "00000000 80000000 00000000 80000000 00000000 80000000 ",
    "00000000 80000000 80000000 80000000 80000000 80000000 ",
    "00000000 80000000 00000000 80000000 00000000 80000000 ",
    "00000000 80000000 80000000 80000000 80000000 80000000",
);

/// One thread's operands (`a` then `b`, four lanes each) and Metal's twelve words for each entry:
/// `a * b`, the scalar products, and the four `dot`s.
struct Thread {
    input: [u32; 8],
    metal: [[u32; 12]; 2],
}

/// The threads of `tiny_products_m3.txt`, then [`HALF_A_NORMAL`].
fn threads() -> Vec<Thread> {
    let hex = |w: &str| u32::from_str_radix(w, 16).expect("hex word");
    let threads: Vec<Thread> = WORDS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .chain([HALF_A_NORMAL])
        .map(|l| {
            let w: Vec<u32> = l.split_whitespace().map(hex).collect();
            assert_eq!(w.len(), 32, "a thread's line: {l}");
            let mut t = Thread {
                input: [0; 8],
                metal: [[0; 12]; 2],
            };
            t.input.copy_from_slice(&w[..8]);
            t.metal[0].copy_from_slice(&w[8..20]);
            t.metal[1].copy_from_slice(&w[20..]);
            t
        })
        .collect();
    assert_eq!(threads.len(), 65, "the fixtures' threads and HALF_A_NORMAL");
    threads
}

/// Fixture `entry` assembled, or `None` (skipped visibly) without `llvm-as`.
fn assemble(entry: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{entry}.ll"));
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {entry}: {e}"),
    }
}

/// The words `entry` stores for `threads` on `p`, run on the interpreter, a thread a workgroup.
fn interpret(bc: &[u8], entry: &str, p: &DeviceProfile, threads: &[Thread]) -> Vec<[u32; 12]> {
    let (plan, m) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let input: Vec<u8> = threads
        .iter()
        .flat_map(|t| t.input.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: vec![0xab; 48 * threads.len()],
            },
        ],
        push: vec![0; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let config = Config {
        workgroups: [u32::try_from(threads.len()).expect("threads"), 1, 1],
        ..Config::default()
    };
    let out = run(&m, &config, res).unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    out.buffers[1]
        .as_chunks::<48>()
        .0
        .iter()
        .map(|t| {
            let mut words = [0; 12];
            for (w, c) in words.iter_mut().zip(t.as_chunks::<4>().0) {
                *w = u32::from_le_bytes(*c);
            }
            words
        })
        .collect()
}

/// The first committed profile (every one's fused multiply-add is the device's `Fma`): each device
/// may keep denormals, as the interpreter does.
fn keeping() -> Vec<(String, DeviceProfile)> {
    committed_profiles_with(DevicePolicy::RESIDENT_TABLES)
        .expect("profiles")
        .into_iter()
        .take(1)
        .collect()
}

/// Whether `x` is an `f32` denormal.
fn denormal(x: u32) -> bool {
    x & 0x7F80_0000 == 0 && x & 0x007F_FFFF != 0
}

/// On a device that may keep a denormal each product is the device's (here the interpreter's IEEE
/// product, rounded first, denormals kept), which differs from Metal's word only where Metal
/// flushes: a product that rounds up to `2^-126` of its sign where Metal answers the zero of its
/// sign (`float-tininess-after-rounding`; some lanes do), or a denormal operand or result
/// (`PrecisionNote::DeviceDenormals`). Every committed profile compiles the fixtures and passes L3.
#[test]
fn a_tiny_product_is_the_device_s() {
    let threads = threads();
    for (i, entry) in ENTRIES.into_iter().enumerate() {
        let Some(bc) = assemble(entry) else {
            return;
        };
        for (stem, p) in committed_profiles_with(DevicePolicy::RESIDENT_TABLES).expect("profiles") {
            let what = format!("{entry} on {stem}");
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
        let profiles = keeping();
        assert!(!profiles.is_empty(), "no profile keeps denormals");
        for (stem, p) in profiles {
            let got = interpret(&bc, entry, &p, &threads);
            let mut rounded_first = 0;
            for (t, (thread, words)) in threads.iter().zip(&got).enumerate() {
                // The vector products, then the scalar ones: lane `l` of each is `a[l] * b[l]`.
                let products = words.iter().zip(&thread.metal[i]).take(8);
                for (w, (&got, &metal)) in products.enumerate() {
                    let l = w % 4;
                    let (a, b) = (thread.input[l], thread.input[4 + l]);
                    let device = (f32::from_bits(a) * f32::from_bits(b)).to_bits();
                    assert_eq!(
                        got, device,
                        "{entry} on {stem}: thread {t} word {w}, Metal {metal:08x}"
                    );
                    if device == metal || denormal(a) || denormal(b) || denormal(device) {
                        continue;
                    }
                    assert_eq!(
                        (device & 0x7FFF_FFFF, device & 0x8000_0000),
                        (0x0080_0000, metal),
                        "{entry} on {stem}: thread {t} word {w}: only a tiny product differs"
                    );
                    rounded_first += 1;
                }
            }
            assert!(
                rounded_first > 0,
                "{entry} on {stem}: no product rounded up to 2^-126"
            );
        }
    }
}

/// The fast entry declares no `SignedZeroInfNanPreserve`, at 32 bits or 64, on any committed
/// profile: nothing in it answers a zero's sign.
#[test]
fn the_fast_entry_keeps_no_zero_signs() {
    use super::engine_arith::declares_signed_zero_inf_nan;
    let entry = ENTRIES[1];
    let Some(bc) = assemble(entry) else {
        return;
    };
    for (stem, p) in committed_profiles_with(DevicePolicy::RESIDENT_TABLES).expect("profiles") {
        let what = format!("{entry} on {stem}");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        for width in [32, 64] {
            assert!(
                !declares_signed_zero_inf_nan(&words, width),
                "{what}: mode {width}"
            );
        }
    }
}
