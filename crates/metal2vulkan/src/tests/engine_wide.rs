//! Wide integers without `shaderInt64` end to end.
//!
//! The public fixtures whose 64-bit integers are genuine (loads and stores of `i64` and
//! `<N x i64>`, `air.convert` to and from `i64`, `udiv i64`, 64-bit bit-field functions, `i64`
//! allocas and unions) compile on the committed `moltenvk-m3` description twice: as committed
//! (native `u64`) and with `shaderInt64` removed (two words per value). Both typed modules run on
//! the CPU interpreter over the same input buffers, and every output byte must agree; the word
//! form also passes L3 without the `Int64` capability. The inputs are small positive words
//! (denormal as floats), so no conversion, division, or bit field is out of its defined range.
//!
//! `llvm-as` assembles the fixtures; without it the test skips visibly.

use std::path::Path;

use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, fixtures_dir, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// `(fixture, entry)` of the fixtures with genuine 64-bit integers.
const FIXTURES: [(&str, &str); 8] = [
    (
        "kernel_bitfield_and_wide_counts",
        "kernel_bitfield_and_wide_counts",
    ),
    ("kernel_divide_shapes", "divshapes"),
    (
        "kernel_integer_width_conversions",
        "kernel_integer_width_conversions",
    ),
    (
        "kernel_thread_local_narrowing_store",
        "kernel_thread_local_narrowing_store",
    ),
    ("kernel_union_member_store", "kernel_union_member_store"),
    (
        "kernel_narrow_vector_converts",
        "kernel_narrow_vector_converts",
    ),
    (
        "kernel_bfloat_convert_rounding",
        "kernel_bfloat_convert_rounding",
    ),
    (
        "kernel_exact_convert_and_fma",
        "kernel_exact_convert_and_fma",
    ),
];

/// The committed `moltenvk-m3` description, with or without `shaderInt64`.
fn moltenvk(int64: bool) -> DeviceProfile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    if !int64 {
        assert!(desc.features.remove(&VkFeature::ShaderInt64));
    }
    let p = Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    assert_eq!(p.fields().leaf.int64, int64);
    p
}

fn assemble(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// Eight buffers of small words (1 to 12: denormal as floats, in range as bit-field offsets and
/// counts; an `i64` read from two of them has a nonzero high word).
fn inputs() -> Vec<Buffer> {
    (0..8u32)
        .map(|index| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index,
            }),
            address: None,
            bytes: (0..1024u32)
                .flat_map(|i| (1 + (i * 5 + index * 3) % 12).to_le_bytes())
                .collect(),
        })
        .collect()
}

/// The buffers after one invocation of `entry` on `p`.
fn outputs(bc: &[u8], entry: &str, p: &DeviceProfile) -> Vec<Vec<u8>> {
    let (_, m) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let res = Resources {
        buffers: inputs(),
        push: vec![0; 256],
        spec: Default::default(),
    };
    run(&m, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"))
        .buffers
}

#[test]
fn wide_fixtures_compute_what_native_64_bit_integers_do() {
    let (native, words) = (moltenvk(true), moltenvk(false));
    for (stem, entry) in FIXTURES {
        let Some(bc) = assemble(stem) else {
            return;
        };
        // L3 without shaderInt64: no Int64 capability.
        let a = compile(&bc, entry, &words).unwrap_or_else(|e| panic!("{entry}: {e}"));
        link_and_validate(&a, &words).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
        assert!(
            a.strategies
                .entries
                .keys()
                .any(|n| n.starts_with("arith.wide") || n.starts_with("access.")),
            "{entry}: {:?}",
            a.strategies.entries
        );
        let (want, got) = (outputs(&bc, entry, &native), outputs(&bc, entry, &words));
        assert_eq!(want.len(), got.len(), "{entry}");
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            if let Some(at) = w.iter().zip(g).position(|(x, y)| x != y) {
                panic!(
                    "{entry}: buffer {i} differs at byte {at}: native {:02x?}, words {:02x?}",
                    &w[at..(at + 8).min(w.len())],
                    &g[at..(at + 8).min(g.len())]
                );
            }
        }
    }
}
