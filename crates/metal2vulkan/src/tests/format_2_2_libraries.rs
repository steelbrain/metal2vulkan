//! A library in format 2.2's layout, an older toolchain's, through the public API.
//!
//! Format 2.2's entries carry no `MDSZ` and it has no extended header (`container`'s `mtlb` docs);
//! `metal2vulkan-devtools` writes the container in that layout here, around owned fixtures'
//! modules padded to a multiple of 16 bytes, as Apple's toolchain pads each module. Metal takes a
//! library without `MDSZ` and runs its functions (the conformance crate's
//! `probes/metallib_without_entry_tags.m`, an M3 Max under macOS 27.0.1). Each function holds its
//! module to the next one's start, so it has the content key, and compiles to the words, that the
//! same module has in today's layout. `llvm-as` assembles the fixtures; without it the test skips
//! visibly. No GPU is used.

use metal2vulkan_devtools::metallib::{metallib, metallib_2_2, wrap, MtlbFunction};

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::library::{FunctionStage, Library};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::testing::{fixtures_dir, llvm_assemble, AssembleError};
use crate::pipeline::FunctionDescriptor;

/// Owned fixture `stem`'s module, wrapped and padded to a multiple of 16 bytes, or `None`
/// (skipped visibly) without `llvm-as`.
fn module(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&fixtures_dir().join(format!("{stem}.ll"))) {
        Ok(bc) => {
            let mut m = wrap(&bc);
            m.resize(m.len().next_multiple_of(16), 0);
            Some(m)
        }
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

/// The bytes a bitcode wrapper's offset and size cover.
fn wrapper_end(module: &[u8]) -> usize {
    let word =
        |at: usize| u32::from_le_bytes(module[at..at + 4].try_into().expect("word")) as usize;
    word(8) + word(12)
}

fn device() -> Device {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/runtime-floor.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

#[test]
fn a_format_2_2_library_compiles_as_today_s_layout() {
    let stages = [
        ("vertex_varying_triangle", 0u8, FunctionStage::Vertex),
        ("fragment_varying_color", 1, FunctionStage::Fragment),
        ("fragment_constant_color", 1, FunctionStage::Fragment),
    ];
    let mut modules = Vec::new();
    for (stem, _, _) in stages {
        let Some(m) = module(stem) else {
            return;
        };
        modules.push(m);
    }
    assert!(
        modules.iter().any(|m| wrapper_end(m) < m.len()),
        "a module is padded past its wrapper's end"
    );
    let functions: Vec<MtlbFunction<'_>> = stages
        .iter()
        .zip(&modules)
        .map(|((name, type_code, _), m)| MtlbFunction {
            name: *name,
            type_code: *type_code,
            bitcode: m.clone(),
            constants: Vec::new(),
            public: Vec::new(),
            tags: Vec::new(),
        })
        .collect();
    let old = Library::parse(&metallib_2_2(&functions)).expect("the format 2.2 library");
    let today = Library::parse(&metallib(&functions)).expect("today's library");
    let dev = device();
    for (name, _, stage) in stages {
        let (a, b) = (
            old.function(name).expect("function"),
            today.function(name).expect("function"),
        );
        assert_eq!(a.info().stage, stage, "{name}");
        assert_eq!(a.info().air_hash, b.info().air_hash, "{name}");
        let compile = |f| {
            dev.compile_function(
                &FunctionDescriptor::new(f),
                &LoweringInputs::new(),
                &Budget::standard(),
            )
            .expect(name)
        };
        let (a, b) = (compile(a), compile(b));
        assert_eq!(a.key(), b.key(), "{name}");
        assert_eq!(
            a.artifact().module.words,
            b.artifact().module.words,
            "{name}"
        );
        assert!(a.artifact() == b.artifact(), "{name}");
    }
}
