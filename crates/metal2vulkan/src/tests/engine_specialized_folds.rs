//! Specializations that fold to the size of the same shader with its constants written in, as
//! the owned fixtures `engine/fold_*` (compiled for `nvidia-rtx5080-laptop`):
//!
//! - `fold_config_struct` reads its configuration from a program-scope `constant` struct its
//!   static initializer fills from function constants; specialized, its fragment function comes
//!   within a small margin of `fold_config_direct`, the same shader reading the constants
//!   directly, because SCCP reads the struct's integer fields as their constants.
//! - `fold_loop_extract` feeds a `uint4` of opcodes, indexed by the counters of a 4x4 loop, to a
//!   colour switch; specialized, its fragment function comes within a small margin of
//!   `fold_loop_extract_baked`, the same shader with those constants written in, because the
//!   loops are unrolled and each copy's switch folds.
//! - `fold_static_init_widen` widens a `uchar4` function constant to a `uint4` in its static
//!   initializer (`air.convert.u.v4i32.u.v4i8`); its kernel compiles, and the value it adds is the
//!   widened constant's.
//!
//! The fragment functions sample and compute in `float`: those operations stay in the module, and
//! each pair records the same precision notes. `llvm-as` assembles the fixtures; without it the
//! tests skip visibly.

use crate::device::{Device, DevicePolicy};
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::DataType;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use crate::pipeline::testing::AssembleError;
use crate::pipeline::testing::{description, engine_dir, link_standalone, llvm_assemble};
use crate::pipeline::{FunctionDescriptor, LoweringInputs, StageSource};
use crate::vk;
use crate::Budget;

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

fn device() -> Device {
    let description = description("nvidia-rtx5080-laptop").expect("committed description");
    Device::new(description, DevicePolicy::default()).expect("device")
}

/// A function constant's value: `ty` with `lanes` as its little-endian lanes of `width` bytes.
fn value(ty: DataType, width: usize, lanes: &[u32]) -> FunctionConstantValue {
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|l| l.to_le_bytes().into_iter().take(width))
        .collect();
    FunctionConstantValue::new(ty, bytes)
}

fn with(constants: &[(u32, FunctionConstantValue)]) -> FunctionConstantValues {
    constants
        .iter()
        .cloned()
        .fold(FunctionConstantValues::defaults(), |all, (i, v)| {
            all.with_index(i, v)
        })
}

/// The kernel `k` of `fold_static_init_widen` compiled with `constants`: its words, linked alone.
fn kernel(constants: &[(u32, FunctionConstantValue)]) -> Option<Vec<u32>> {
    let bitcode = fixture("fold_static_init_widen")?;
    let library = Library::from_air(&bitcode).expect("library");
    let mut descriptor = FunctionDescriptor::new(library.function("k").expect("entry"));
    descriptor.constants = with(constants);
    let device = device();
    let compiled = device
        .compile_function(&descriptor, &LoweringInputs::new(), &Budget::standard())
        .unwrap_or_else(|e| panic!("k: {e}"));
    let words = link_standalone(compiled.artifact(), device.profile())
        .unwrap_or_else(|e| panic!("k: {e}"))
        .1;
    Some(words)
}

/// The render pipeline of the fixture `stem`'s `vertex` and `fragment` (compiled with
/// `constants`) to a `BGRA8Unorm` attachment 0: its fragment stage's words and precision notes.
fn fragment(
    stem: &str,
    (vertex, fragment): (&str, &str),
    constants: &[(u32, FunctionConstantValue)],
) -> Option<(usize, Vec<PrecisionNote>)> {
    let bitcode = fixture(stem)?;
    let library = Library::from_air(&bitcode).expect("library");
    let function = |name| FunctionDescriptor::new(library.function(name).expect("entry"));
    let mut shaded = function(fragment);
    shaded.constants = with(constants);
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(vertex)));
    d.fragment = Some(StageSource::Function(shaded));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let pipeline = device()
        .compile_render_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let stage = pipeline
        .stages
        .iter()
        .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
        .expect("a fragment stage");
    Some((stage.spirv.len(), stage.precision_notes.clone()))
}

/// `struct_words` is within an eighth of `direct_words`.
fn near(struct_words: usize, direct_words: usize) -> bool {
    struct_words <= direct_words + direct_words / 8
}

/// The vertex and fragment functions of `fold_config_struct` and `fold_config_direct`.
const ENTRIES: (&str, &str) = ("cfg_vert", "cfg_frag");

/// The vertex and fragment functions of `fold_loop_extract` and `fold_loop_extract_baked`.
const UBER: (&str, &str) = ("uber_vert", "uber_frag");

/// The configuration of `fold_config_struct`: both layers' records (layer 0 present, sampled with
/// texture function 1 and colour opcodes 3 and 5; layer 1 absent), the fetch on with blend 1 and
/// destination 1, one layer.
fn configuration() -> Vec<(u32, FunctionConstantValue)> {
    vec![
        (0, value(DataType::Bool, 1, &[1])),
        (1, value(DataType::Bool, 1, &[0])),
        (2, value(DataType::Bool, 1, &[1])),
        (3, value(DataType::UChar, 1, &[1])),
        (4, value(DataType::UChar, 1, &[0])),
        (5, value(DataType::UChar, 1, &[1])),
        (6, value(DataType::UChar, 1, &[1])),
        (7, value(DataType::UChar4, 1, &[3, 5, 0, 0])),
        (8, value(DataType::UChar4, 1, &[0, 0, 0, 0])),
        (9, value(DataType::UChar, 1, &[1])),
    ]
}

#[test]
fn a_configuration_struct_folds_to_the_direct_reads() {
    let constants = configuration();
    let Some((direct, direct_notes)) = fragment("fold_config_direct", ENTRIES, &constants) else {
        return;
    };
    let Some((folded, notes)) = fragment("fold_config_struct", ENTRIES, &constants) else {
        return;
    };
    eprintln!("config: struct {} words, direct {}", folded, direct);
    assert!(
        near(folded, direct),
        "the struct's specialization is {} words, the direct reads' {}",
        folded,
        direct
    );
    assert_eq!(notes, direct_notes);
}

#[test]
fn a_loop_over_a_constant_vector_folds_to_its_baked_form() {
    let constants = vec![
        (0, value(DataType::Bool, 1, &[1])),
        (1, value(DataType::Bool, 1, &[0])),
        (2, value(DataType::Bool, 1, &[0])),
        (3, value(DataType::UInt, 4, &[3])),
        (4, value(DataType::UInt, 4, &[0])),
        (5, value(DataType::UInt, 4, &[0])),
        (6, value(DataType::UInt4, 4, &[0x03_02_01, 0, 0, 0])),
    ];
    let Some((baked, baked_notes)) = fragment("fold_loop_extract_baked", UBER, &[]) else {
        return;
    };
    let Some((folded, notes)) = fragment("fold_loop_extract", UBER, &constants) else {
        return;
    };
    eprintln!("loop: specialized {} words, baked {}", folded, baked);
    assert!(
        near(folded, baked),
        "the loop's specialization is {} words, the baked form's {}",
        folded,
        baked
    );
    assert_eq!(notes, baked_notes);
}

#[test]
fn a_static_initializer_widening_a_vector_constant_compiles() {
    let widened = |ops: [u32; 4], sel| {
        let constants = [
            (0, value(DataType::UChar4, 1, &ops)),
            (1, value(DataType::UChar, 1, &[sel])),
        ];
        kernel(&constants)
    };
    let Some(a) = widened([1, 200, 3, 255], 7) else {
        return;
    };
    let Some(b) = widened([1, 200, 3, 254], 7) else {
        return;
    };
    // The widened lanes are the module's constants: another value is other words.
    assert_ne!(a, b);
}
