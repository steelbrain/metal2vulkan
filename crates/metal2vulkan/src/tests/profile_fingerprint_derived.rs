//! A device derives its profile's fingerprint **once**, and every compile on it takes that one
//! ([`DeviceProfile::fingerprint`](crate::device::DeviceProfile::fingerprint)).
//!
//! # What this guards
//!
//! The fingerprint is a SHA-256 over the derived `Debug` rendering of every field of the profile.
//! A profile with a format table renders tens of kilobytes, so one derivation is 46 to 64 us over
//! the committed profiles on an M3 Max -- not a rounding error inside a translation that costs one
//! to seven milliseconds.
//!
//! `Device::profile_fingerprint` has held a memo of it since the device was first built, but the
//! two call sites that mattered could not reach it: a stage's compile takes the fingerprint for
//! the compiled function's key (`pipeline::compile`'s `compile_one`, through `key::function_key`)
//! and again to stamp the artifact with the profile it was compiled for, and both hold a
//! `&DeviceProfile` rather than the device. So every translation re-rendered the whole profile
//! twice, and nothing counted it: the SPIR-V, the keys and the artifacts all came out identical,
//! so no golden table, no artifact check and no existing count could see the work.
//!
//! The derivation now happens where the profile is built (`DeviceProfile::of`), which is the
//! only place a profile comes from, and `fingerprint()` is a read.
//!
//! # Why this is a count
//!
//! The number of renderings a run costs is a fact of where the derivation sits, not of the host:
//! it holds under any load and needs no quiet machine. It is held in **both** directions. Over
//! the count is a caller that derives again; under it is a device that derives none, which would
//! mean the fingerprint came from somewhere other than its fields -- so the device's own making
//! is required to derive exactly one, which is what keeps the compiles' zero from being a
//! vacuous zero.
//!
//! The second test is the half a count catches on the low side and the one that matters for
//! correctness: a memo on a value that is supposed to vary must still vary. Two profiles that
//! differ must carry different fingerprints, and a profile edited through
//! `DeviceProfile::edit` must carry the edited fields' fingerprint and not the original's.
//! A stale fingerprint would make two different devices share cache entries, which is wrong and
//! not merely slow.

use crate::budget::{Budget, Meter};
use crate::device::profile::fingerprints_derived;
use crate::device::profile::GuestIsolation;
use crate::device::{ContainmentPolicy, Device, DeviceDescription, DevicePolicy};
use crate::library::FunctionConstantValues;
use crate::library::Library;
use crate::pipeline::compile::{compile_stage, InputsFrom};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::FunctionDescriptor;
use std::path::Path;

/// The kernel the compiles below translate, and the fixture that holds it: a kernel with two
/// device pointers and a threadgroup array, so it compiles under the default lowering inputs.
const FIXTURE: &str = "a27_ob1_threadgroup";
const ENTRY: &str = "ob1_threadgroup";

/// The compiles the run makes after the device is built.
const RUN: u64 = 8;

/// The committed profile the devices are made from.
fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("committed profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// The fixture's library, or `None` where `llvm-as` is absent.
fn library() -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{FIXTURE}.ll"))) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the fixture parses")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {FIXTURE}.ll: {why}"),
    }
}

/// `RUN` translations of one kernel on one device derive the profile's fingerprint **not once
/// each but not at all**: the device's own making derived the one they all take.
#[test]
fn a_run_of_compiles_derives_no_profile_fingerprint() {
    let Some(library) = library() else { return };
    let start = fingerprints_derived();
    let device =
        Device::from_queries(description("moltenvk-m3"), DevicePolicy::default()).expect("device");
    let made = fingerprints_derived() - start;
    // The fingerprint every compile takes, read as often as anything reads it.
    for _ in 0..RUN {
        let _ = device.profile_fingerprint();
        let _ = device.profile().fingerprint();
    }
    let before = fingerprints_derived();
    let function = FunctionDescriptor::new(library.function(ENTRY).expect("the entry"));
    for _ in 0..RUN {
        let mut meter = Meter::new(&Budget::standard());
        let compiled = compile_stage(
            &function.function,
            &FunctionConstantValues::defaults(),
            (&function.linked, &function.libraries),
            device.profile(),
            InputsFrom::Given(&LoweringInputs::default()),
            &mut meter,
        )
        .expect("the kernel compiles");
        // Both call sites are inside one compile: the key the artifact carries and the profile it
        // is stamped with.
        assert_eq!(
            compiled.artifact().profile,
            device.profile_fingerprint(),
            "a compile stamps its artifact with this device's profile"
        );
    }
    // Asserted before the device's own count, so a tree that derives per call fails on the
    // reading that says so: the translations, and the reads beside them, must cost none.
    assert_eq!(
        fingerprints_derived() - before,
        0,
        "{RUN} translations on one device, and {RUN} reads of its fingerprint either way, must \
         derive none: each takes the profile's"
    );
    assert_eq!(
        made, 1,
        "a device must derive its profile's fingerprint exactly once as it is built: a zero here \
         would mean the fingerprint came from somewhere other than its fields, and would make the \
         zero above vacuous"
    );
}

/// The memo must not outlive the fields it is of: two profiles that differ carry different
/// fingerprints, and an edited profile carries the edit's.
#[test]
fn profiles_that_differ_carry_their_own_fingerprints() {
    let m3 =
        Device::from_queries(description("moltenvk-m3"), DevicePolicy::default()).expect("device");
    let pipe =
        Device::from_queries(description("lavapipe"), DevicePolicy::default()).expect("device");
    assert_ne!(
        m3.profile_fingerprint(),
        pipe.profile_fingerprint(),
        "two devices whose profiles differ must not share a fingerprint: a shared one would let \
         each read the other's cache entries"
    );
    // The derivation is of the fields the profile holds, whatever built them.
    let uncontained = DevicePolicy {
        containment: ContainmentPolicy::UnsafeOff,
        isolation: GuestIsolation::Off,
        ..DevicePolicy::default()
    };
    let uncontained = Device::from_queries(description("moltenvk-m3"), uncontained)
        .expect("device under the uncontained policy");
    assert_ne!(
        m3.profile_fingerprint(),
        uncontained.profile_fingerprint(),
        "the policy is part of the profile's fields, so it is part of the fingerprint"
    );
    let mut edited = m3.profile().clone();
    edited.edit(|x| x.dynamic_vertex_stride = !x.dynamic_vertex_stride);
    assert_ne!(
        m3.profile().fingerprint(),
        edited.fingerprint(),
        "an edited profile must carry the edited fields' fingerprint, not the original's"
    );
    let mut back = edited.clone();
    back.edit(|x| x.dynamic_vertex_stride = !x.dynamic_vertex_stride);
    assert_eq!(
        m3.profile().fingerprint(),
        back.fingerprint(),
        "the derivation is a function of the fields alone, so an edit and its inverse come back \
         to the same fingerprint"
    );
}
