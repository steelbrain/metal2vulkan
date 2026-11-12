//! A texture unit a `switch` picks (a PHI of constants) and a bit test of the unit that every unit
//! it can pick passes, the shape GL's AIR takes where it emulates a cube texture as a 2D array: the
//! owned fixture `kernel_texture_unit_bit_test` compiles on committed profiles, its cube sample of
//! a `texture2d_array` (on the way no unit takes) deleted by SCCP's values of a few constants.

use crate::air::ir::print::print;
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::library::{FunctionConstantValues, Library};
use crate::pipeline::testing::{
    description, fixtures_dir, llvm_assemble, meter, profile, specialized_entry, AssembleError,
};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

const ENTRY: &str = "kernel_texture_unit_bit_test";

const PROFILES: [&str; 3] = ["lavapipe", "amd-rdna3-radv", "moltenvk-m3"];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("{ENTRY}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {ENTRY}.ll: {why}"),
    }
}

/// The fixture's kernel as the compile specializes it for profile `stem`, printed.
fn specialized(bc: &[u8], stem: &str) -> String {
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    let none = FunctionConstantValues::defaults();
    match specialized_entry(bc, ENTRY, &p, &none, &[], &mut meter()) {
        Ok(e) => print(&e.module),
        Err(e) => format!("not specialized: {e}"),
    }
}

/// The bit test is true for every unit the `switch` picks, so the cube sample is deleted and the
/// kernel's pipeline compiles on every profile.
#[test]
fn a_cube_sample_no_picked_unit_reaches_is_deleted() {
    let Some(bc) = fixture() else { return };
    let library = Library::from_air(&bc).expect("library");
    for stem in PROFILES {
        let text = specialized(&bc, stem);
        let dev = Device::new(
            description(stem).expect("committed description"),
            DevicePolicy::default(),
        )
        .expect("device");
        let d = ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            library.function(ENTRY).expect("entry"),
        )));
        if let Err(e) = dev.compile_compute_pipeline(&d, &Budget::standard()) {
            panic!("{stem}: {e}\n{text}");
        }
        assert!(!text.contains("sample_texture_cube"), "{stem}: {text}");
        assert!(text.contains("sample_texture_2d_array"), "{stem}: {text}");
    }
}
