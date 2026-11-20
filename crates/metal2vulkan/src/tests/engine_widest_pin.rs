//! The widest-pin rule (`lower::simd::widest_pin`) on the owned fixture `engine/wps_shuffles`: ten
//! shuffle-only kernels, `wps_<float|uint>_k<K>_s<S>`, each K live accumulators over S steps whose
//! only cross-lane operations are `simd_shuffle_xor`s in uniform flow.
//!
//! On a profile whose kernel stage pins a size between a `minSubgroupSize` below 32 and a
//! `maxSubgroupSize` of 32 (the committed Intel profiles), the four large kernels of each form take
//! the `Emulated` tier (`simd.emulated.widest-pin`) and pin nothing, and `k8_s48`, under the
//! rule's bounds, pins 32 lanes with full subgroups. On a profile whose smallest size is 32 (RADV,
//! NVIDIA) every kernel pins 32 lanes with full subgroups. Where the kernel stage has no 32-lane
//! subgroup (lavapipe, swiftshader) or pins no size (moltenvk-m3) the rule does not apply and no
//! kernel pins. The pick follows the device facts alone; the rule's measured reason (the pinned
//! form of the large kernels is slower to create and to run on Intel GPUs, and the emulated one is
//! no faster elsewhere) is the device test's
//! (`metal2vulkan-validation`'s `widest_pin` under `--cfg m2v_gpu`).

use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::library::Library;
use crate::lower::simd::WIDEST_PIN;
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, LoweringInputs};

/// The kernels over the rule's bounds, then the one under them.
const LARGE: [&str; 4] = ["k32_s64", "k48_s16", "k40_s32", "k24_s32"];
const SMALL: &str = "k8_s48";

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("wps_shuffles.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected wps_shuffles.ll: {why}"),
    }
}

/// How a profile's kernel stage meets the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// It pins sizes from below 32 up to 32: the rule applies.
    Widest,
    /// Its smallest size is 32: every kernel pins 32 lanes.
    Pins,
    /// It has no 32-lane subgroup or pins no size: no kernel pins.
    Neither,
}

/// On each profile, each kernel's tier is the one the rule picks (the module docs).
#[test]
fn the_widest_pin_rule_picks_each_shuffle_kernel_s_tier_per_profile() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    for (stem, stage) in [
        ("intel-irisxe-rplp-anv", Stage::Widest),
        ("intel-irisxe-anv", Stage::Widest),
        ("intel-uhd770-rpls-anv", Stage::Widest),
        ("intel-arl-anv", Stage::Widest),
        ("amd-rdna3-radv", Stage::Pins),
        ("amd-780m-radv", Stage::Pins),
        ("nvidia-rtx50-linux", Stage::Pins),
        ("nvidia-pascal-linux", Stage::Pins),
        ("lavapipe", Stage::Neither),
        ("swiftshader", Stage::Neither),
        ("moltenvk-m3", Stage::Neither),
    ] {
        let dev = Device::from_queries(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        for form in ["float", "uint"] {
            for size in LARGE.into_iter().chain([SMALL]) {
                let name = format!("wps_{form}_{size}");
                let what = format!("{stem} {name}");
                let compiled = dev
                    .compile_function(
                        &FunctionDescriptor::new(library.function(&name).expect("entry")),
                        &LoweringInputs::default(),
                        &Budget::standard(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let emulated = stage == Stage::Widest && size != SMALL;
                assert_eq!(
                    compiled.strategies().entries.contains_key(WIDEST_PIN),
                    emulated,
                    "{what}: the widest pin's emulated tier"
                );
                match stage {
                    Stage::Widest | Stage::Pins if !emulated => {
                        assert_eq!(compiled.required_subgroup_size(), Some(32), "{what}");
                        assert!(compiled.requires_full_subgroups(), "{what}");
                    }
                    _ => assert_eq!(compiled.required_subgroup_size(), None, "{what}"),
                }
            }
        }
    }
}
