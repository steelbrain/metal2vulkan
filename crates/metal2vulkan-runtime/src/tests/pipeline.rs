//! The checks a compute pipeline state's descriptor passes before its compile
//! (`pipeline::duplicate_linked_name`), the imageblock length it answers
//! (`pipeline::imageblock_memory_length`), the specialization values a failed draw's
//! pipeline names (`pipeline::specialization_values`), and which form of a key a driver's facts
//! choose (`pipeline::views_read_at_bind`).
//!
//! The check reads only the linked functions' names, so any library function stands in for a
//! visible one. The runtime reaches only the public compile-and-plan layer (`api_surface`), its
//! tests too, so the functions come from committed AIR bitcode read as it is
//! (`validation/fixtures/public/engine/*.air`, compiled by `scripts/regen-air-fixtures`):
//! no tool runs and no `METAL2VULKAN_*` knob is read.

use super::super::pipeline::{
    duplicate_linked_name, imageblock_memory_length, specialization_values, views_read_at_bind,
};
use super::super::values::Size;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::device::{DeviceDescription, DriverFact};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues, Library};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{LinkedFunction, LinkedFunctions, SpecKey};
use metal2vulkan::vk::SpecializationMapEntry;
use std::path::Path;

/// The one function of committed AIR fixture `stem`, as a linked function.
fn linked(stem: &str) -> LinkedFunction {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{stem}.air"));
    let bitcode = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let library = Library::from_air(&bitcode).expect("library");
    LinkedFunction::new(library.function(stem).expect("function"))
}

/// A name `functions` lists twice, or `binary_functions` twice, or both lists hold for two
/// different functions, is a descriptor Metal's creation fails on or whose answer was not measured,
/// named as its first repeat; one function both lists hold is taken, as plain Metal takes it; a
/// name only `private_functions` or a group repeats is left to the compile, as is one listed once.
#[test]
fn a_name_the_numbered_lists_repeat_is_found() {
    let (first, second, third) = (
        linked("ob3_stage_input"),
        linked("ob3_index_buffer"),
        linked("ob3_texture_array"),
    );
    let once = LinkedFunctions::new()
        .with_function(first.clone())
        .with_function(second.clone())
        .with_private_function(third.clone())
        .with_private_function(third.clone());
    assert_eq!(duplicate_linked_name(&once), None);
    let mut grouped = once.clone();
    grouped
        .groups
        .insert("calls".to_owned(), vec![first.clone(), first.clone()]);
    assert_eq!(duplicate_linked_name(&grouped), None);
    let twice = LinkedFunctions::new()
        .with_function(second.clone())
        .with_function(first.clone())
        .with_function(second.clone())
        .with_function(first.clone());
    assert_eq!(
        duplicate_linked_name(&twice),
        Some("ob3_index_buffer".to_owned())
    );
    // One function in both lists: plain Metal takes it, with one handle.
    let mut both = once.clone();
    both.binary_functions
        .push(first.clone().with_precompiled(true));
    assert_eq!(duplicate_linked_name(&both), None);
    let mut binary_twice = once.clone();
    binary_twice.binary_functions.extend([
        third.clone().with_precompiled(true),
        third.clone().with_precompiled(true),
    ]);
    assert_eq!(
        duplicate_linked_name(&binary_twice),
        Some("ob3_texture_array".to_owned())
    );
    // Another function of the same name (its constant values differ) in the other list.
    let mut other = first.clone().with_precompiled(true);
    other.constants = FunctionConstantValues::defaults()
        .with_index(0, FunctionConstantValue::new(DataType::Bool, [1u8]));
    let mut different = once;
    different.binary_functions.push(other);
    assert_eq!(
        duplicate_linked_name(&different),
        Some("ob3_stage_input".to_owned())
    );
}

/// `imageblockMemoryLengthForDimensions:` rounds each side up to a power of two and ignores the
/// depth, as Metal does on an M3 and an M4 (probe 6), and is none without an imageblock or a side.
/// A cell rounds up to 8 bytes, as an M4's does (an M3's to 4), and the answer is raised to the
/// cells rounded up to 16 bytes, what the binder reserves, where that is more.
#[test]
fn an_imageblock_takes_power_of_two_sides_of_its_cells() {
    let length = |cell, width, height, depth| {
        imageblock_memory_length(cell, Size::new(width, height, depth))
    };
    assert_eq!(length(None, 8, 8, 1), 0);
    assert_eq!(length(Some(16), 0, 8, 1), 0);
    assert_eq!(length(Some(16), 8, 0, 1), 0);
    // An M3's and an M4's answers alike for 16-byte cells.
    for ((width, height), bytes) in [
        ((1, 1), 16),
        ((2, 1), 32),
        ((3, 5), 512),
        ((5, 3), 512),
        ((7, 1), 128),
        ((9, 9), 4096),
        ((33, 2), 2048),
        ((48, 40), 65536),
        ((16, 16), 4096),
    ] {
        assert_eq!(
            length(Some(16), width, height, 1),
            bytes,
            "{width}x{height}"
        );
    }
    assert_eq!(length(Some(16), 16, 16, 2), 4096);
    // An M4's answers for 1-, 6-, 12- and 20-byte cells (probe 6).
    for (cell, (width, height), bytes) in [
        (1, (2, 1), 16),
        (1, (8, 8), 512),
        (6, (2, 1), 16),
        (6, (3, 5), 256),
        (12, (1, 1), 16),
        (12, (2, 1), 32),
        (12, (16, 16), 4096),
        (20, (2, 1), 48),
        (20, (9, 9), 6144),
    ] {
        assert_eq!(
            length(Some(cell), width, height, 1),
            bytes,
            "{cell}: {width}x{height}"
        );
    }
    // One cell whose rounded size is an odd multiple of 8: the 16 bytes the binder reserves.
    assert_eq!(length(Some(1), 1, 1, 1), 16);
    assert_eq!(length(Some(6), 1, 1, 1), 16);
    assert_eq!(length(Some(20), 1, 1, 1), 32);
    assert_eq!(length(Some(16), u64::MAX, 1, 1), u64::MAX);
}

/// A failed draw's pipeline names each of its stages' specialization constants once, by
/// meaning, in id order: the value the key gives it, read as its low bytes (4 or 8 of them), else
/// the module's default. A key entry for an id no stage declares specializes nothing and is left
/// out; one whose bytes lie past the key's data leaves the default.
#[test]
fn a_pipelines_constants_take_the_keys_values_else_their_defaults() {
    let entry = |constant_id, offset, size| SpecializationMapEntry {
        constant_id,
        offset,
        size,
    };
    let mut key = SpecializationKey::new();
    key.data = [4u32.to_le_bytes(), 7u32.to_le_bytes(), 1u32.to_le_bytes()].concat();
    key.data.extend(0x1_0000_0002u64.to_le_bytes());
    key.entries = vec![
        entry(2, 0, 4),
        entry(9, 4, 4),
        entry(5, 8, 4),
        entry(7, 12, 8),
        entry(0, 100, 4),
    ];
    // Two stages: the vertex stage's, then the fragment stage's, sharing the sample count.
    let constants: [(u32, SpecKey, u64); 6] = [
        (2, SpecKey::SampleCount, 1),
        (0, SpecKey::AmplificationCount, 1),
        (7, SpecKey::TgLength(1), 0),
        (5, SpecKey::MultiRegion, 0),
        (2, SpecKey::SampleCount, 1),
        (3, SpecKey::FlatLocalSize, 32),
    ];
    let expected: [(SpecKey, u64); 5] = [
        (SpecKey::AmplificationCount, 1),
        (SpecKey::SampleCount, 4),
        (SpecKey::FlatLocalSize, 32),
        (SpecKey::MultiRegion, 1),
        (SpecKey::TgLength(1), 0x1_0000_0002),
    ];
    assert_eq!(specialization_values(constants, &key), expected);
    assert!(specialization_values(std::iter::empty(), &key).is_empty());
}

/// Which `VkPipeline` a state's draws take is the driver's **fact**, never its name
/// (`pipeline/render/residual.rs`): a driver measured to load a uniform read at a constant offset
/// ahead of the shader (`DriverFact::ConstantOffsetUniformReadsPreloaded`) reads a view at its
/// bind offset alone, a second pipeline per state; every other driver reads every view at its
/// residual, the one pipeline its creation made.
///
/// No committed profile carries the fact, so each of them -- MoltenVK's, RADV's, ANV's, NVIDIA's,
/// lavapipe's and SwiftShader's alike -- takes the residual form. That is the check: a decision
/// that told one driver apart would answer otherwise for its profile than for the rest, and a
/// `MEASURED` row written without a measurement would show here as a profile that reads at its
/// bind offset.
#[test]
fn every_committed_driver_reads_its_uniform_views_at_their_residuals() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut read = 0;
    for entry in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = entry.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let description =
            DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let facts = description.driver_facts();
        assert!(
            !facts.contains(&DriverFact::ConstantOffsetUniformReadsPreloaded),
            "{} is measured with a fact no measurement took",
            path.display()
        );
        assert!(
            !views_read_at_bind(&facts),
            "{} reads a uniform view at its bind offset alone",
            path.display()
        );
        read += 1;
    }
    assert!(read > 1, "profiles read: {read}");
    // The other form is kept and reachable: the fact alone chooses it.
    assert!(views_read_at_bind(
        &[DriverFact::ConstantOffsetUniformReadsPreloaded]
            .into_iter()
            .collect()
    ));
}
