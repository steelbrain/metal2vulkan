//! A pass's color attachments its render pipeline lacks (`VariantKey::zeroed_colors`), over the
//! owned fixture `engine/stage_io` (a fragment function writing colors 0 and 2): a function that
//! writes a color it would zero stores the zero there in place of its own value, as plain Metal
//! does (a Metal probe): its one output at that color is the zero's.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::collections::BTreeMap;
use std::path::Path;

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::metal::PixelFormat;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};

/// The bitcode of the owned fixture `stem` (`engine/<stem>.ll`), or `None` (printed) where
/// `llvm-as` is missing.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// A fragment function writing a color it would zero (`stage_io_fragment` writes colors 0 and
/// 2, color 2 a `uint4`) stores the zero there in place of its own value, as plain Metal does: its
/// one output at color 2 is the zero's, of the zeroed format's numeric class (a `float` one too,
/// beside the `uint4` the function writes), and color 0 keeps its own output.
#[test]
fn a_function_writing_a_color_it_would_zero_stores_the_zero_there() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let dev = Device::from_queries(description("lavapipe"), DevicePolicy::RESIDENT_TABLES)
        .expect("device");
    let p = dev.profile();
    for (format, numeric) in [
        (PixelFormat::RGBA32Uint, IoNumeric::Uint),
        (PixelFormat::RGBA8Unorm, IoNumeric::Float),
    ] {
        let mut inputs = LoweringInputs::new();
        inputs.zeroed_colors = BTreeMap::from([(2, format)]);
        let a = compile_with(&bc, "stage_io_fragment", p, &inputs)
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let at = |color| {
            a.interface
                .io
                .outputs
                .iter()
                .filter(|o| {
                    o.role
                        == IoRole::Color {
                            attachment: color,
                            index: 0,
                        }
                })
                .collect::<Vec<_>>()
        };
        let zero = at(2);
        assert!(
            matches!(zero.as_slice(), [o] if o.ty.numeric == numeric && o.ty.components == 4),
            "{format:?}: {zero:?}"
        );
        let own = at(0);
        assert!(
            matches!(own.as_slice(), [o] if o.ty.numeric == IoNumeric::Float),
            "{format:?}: {own:?}"
        );
        assert!(a.interface.io.unattached_colors.is_empty(), "{format:?}");
    }
}
