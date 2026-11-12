//! MSL 4.1's generic pointers, as compute pipelines of the owned fixture `engine/generic_pointers`
//! on committed profiles: `generic_spaces` (generic pointers into the device, thread and
//! threadgroup spaces, each of one space, read, written and asked their space) and `generic_null`
//! (generic pointers the run may make null) compile; `generic_mixed` (one generic pointer into the
//! thread or the device space as the run decides) is refused as a generic pointer of two spaces.

use crate::air::ir::print::print;
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature};
use crate::library::{FunctionConstantValues, Library};
use crate::pipeline::testing::{
    description, engine_dir, llvm_assemble, meter, profile, specialized_entry, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("generic_pointers.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected generic_pointers.ll: {why}"),
    }
}

const PROFILES: [&str; 3] = ["lavapipe", "amd-rdna3-radv", "moltenvk-m3"];

/// Kernel `entry`'s pipeline on profile `stem`.
fn compiled(library: &Library, stem: &str, entry: &str) -> Result<(), Error> {
    let dev = Device::new(
        description(stem).expect("committed description"),
        DevicePolicy::default(),
    )
    .expect("device");
    let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        library.function(entry).expect("entry"),
    )));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .map(|_| ())
}

/// Kernel `entry` of `bc` as the compile specializes it for profile `stem`, printed: the module a
/// refusal's location names an instruction of.
fn specialized(bc: &[u8], stem: &str, entry: &str) -> String {
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    let none = FunctionConstantValues::defaults();
    match specialized_entry(bc, entry, &p, &none, &[], &mut meter()) {
        Ok(e) => print(&e.module),
        Err(e) => format!("not specialized: {e}"),
    }
}

/// Generic pointers each of one space, the null pointer among them, compile on every profile.
#[test]
fn generic_pointers_of_one_space_compile() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    for stem in PROFILES {
        for entry in ["generic_spaces", "generic_null"] {
            if let Err(e) = compiled(&library, stem, entry) {
                panic!("{stem} {entry}: {e}\n{}", specialized(&bc, stem, entry));
            }
        }
    }
}

/// A generic pointer into the thread or the device space as the run decides is refused, naming
/// the generic space.
#[test]
fn a_generic_pointer_of_two_spaces_is_refused() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    for stem in PROFILES {
        match compiled(&library, stem, "generic_mixed") {
            Err(Error::Unsupported(u)) if u.feature == Feature::AddressSpace(8) => {}
            other => panic!("{stem}: {other:?}"),
        }
    }
}
