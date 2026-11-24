//! A kernel's partial-threadgroups variant (`LoweringInputs::partial_threadgroups`) and its
//! reconvergence. Its invocations past the grid return before the body, and SPIR-V states
//! reconvergence over all invocations, returned ones included (SPIR-V 1.6 section 2.2.5, "Dynamic
//! Instance": "Invocations may only resume executing the same dynamic instances when all
//! invocations reach the same static instruction"). Only `MaximallyReconvergesKHR` lets an
//! invocation that returned escape reconvergence (`SPV_KHR_maximal_reconvergence`,
//! "Reconvergence"). So the variant's tangled instructions take the profile's mode even where the
//! plain module's are reached together (`lower::simd::together`): on a profile with maximal
//! reconvergence both are `Guaranteed` with no note, and on one without it the variant records
//! `ImplementationDefined` with `DeviceReconvergence` while the plain module is `Guaranteed`.
//!
//! The kernel is the engine's `a27_a4_simdgroup_barrier` (a simdgroup barrier in the entry block,
//! loop-free), whose plain module needs no mode.
//!
//! `llvm-as` assembles the fixture; without it the test skips visibly.

use crate::lower::select::LowerInputs;
use crate::pipeline::artifact::{PrecisionNote, Reconvergence};
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::committed_profiles;

/// The simdgroup barrier's plain module is `Guaranteed` with no note on every committed profile;
/// its partial-threadgroups variant is too where the profile has maximal reconvergence, and
/// records `ImplementationDefined` with `DeviceReconvergence` where it does not.
#[test]
fn a_simdgroup_barrier_s_variant_is_guaranteed_only_under_maximal_reconvergence() {
    let ll = engine_dir().join("a27_a4_simdgroup_barrier.air.ll");
    let bc = match llvm_assemble(&ll) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let (mut with, mut without) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let plain = compile_with(&bc, "a4_simdgroup_barrier", &p, &LowerInputs::new());
        let mut inputs = LowerInputs::new();
        inputs.partial_threadgroups = true;
        let variant = compile_with(&bc, "a4_simdgroup_barrier", &p, &inputs);
        let (Ok(plain), Ok(variant)) = (plain, variant) else {
            continue;
        };
        let state = |a: &crate::library::FunctionArtifact| {
            (
                a.reconvergence,
                a.precision.contains(&PrecisionNote::DeviceReconvergence),
            )
        };
        let guaranteed = (Reconvergence::Guaranteed, false);
        assert_eq!(state(&*plain), guaranteed, "{stem}: the plain module");
        if p.fields().subgroup.maximal_reconvergence {
            assert_eq!(state(&*variant), guaranteed, "{stem}: the variant");
            with += 1;
        } else {
            assert_eq!(
                state(&*variant),
                (Reconvergence::ImplementationDefined, true),
                "{stem}: the variant"
            );
            without += 1;
        }
    }
    assert!(
        with > 0 && without > 0,
        "profiles compiled: {with} with maximal reconvergence, {without} without"
    );
}
