//! Forms SPIRV-Cross (MoltenVK) converts to MSL that Metal compiles, on every profile. Each holds
//! on every device, since each is exact everywhere:
//!
//! - A `bool` vector member of an aggregate is held as `u32` lanes of 0 and 1: no module declares
//!   a struct with a `bool` vector member. SPIRV-Cross declares every `bool` struct member as a
//!   `short` and assigns a `bool` vector to it without a conversion, which Metal refuses. The
//!   owned fixture `engine/stage_io_tail`'s `tail_vertex_bool` returns such a struct.
//! - Only the entry reads the `SampleMask` input: where the entry wraps the body (an interlock; the
//!   loop over the covered samples), the body reads the entry's copy. SPIRV-Cross declares the
//!   input a scalar in the entry and passes it to a function as an array. The owned fixtures
//!   `engine/msf_fetch_ms`'s `msf_fetch_mask_in` (the interlocked storage form of a framebuffer
//!   fetch) and `engine/rt6_attachmentless`'s `rt6_mark_coverage` (the loop) read it.
//!
//! And one form that keeps MoltenVK's pipeline creation within the ceiling: SPIRV-Cross
//! force-inlines every function it writes, so Metal's back end compiles a routine once per call. A
//! `half` fused multiply-add on a MoltenVK capture is the device's own 16-bit `Fma`, inline, never
//! a call of the `half` routine (one corpus kernel's 800 calls took 6 s to create).
//!
//! On macOS each MoltenVK profile's module is also converted with `spirv-cross` under MoltenVK's
//! options and compiled with `xcrun metal`, as MoltenVK's pipeline creation does
//! (`moltenvk_msl_compile`); a missing tool skips visibly. `llvm-as` assembles the fixtures. No
//! GPU is used.

use std::collections::BTreeSet;
use std::path::Path;

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::testing::{
    engine_dir, link_standalone, llvm_assemble, moltenvk_msl_compile, AssembleError,
};
use crate::pipeline::{
    ColorAttachmentDescriptor, FramebufferFetchInputs, FunctionDescriptor, LoweringInputs,
    RenderPipelineDescriptor, StageSource,
};
use crate::spirv::testing::check_words_valid;
use crate::Budget;

const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_STRUCT: u32 = 30;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const DECORATION_BUILTIN: u32 = 11;
const BUILTIN_SAMPLE_MASK: u32 = 20;
const STORAGE_INPUT: u32 = 1;

fn library(name: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

/// Every committed profile, by stem.
fn devices() -> Vec<(String, Device)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, Device)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("profile directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().expect("stem").to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile");
            let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
            let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
            (stem, dev)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Each instruction of `words` as its opcode and operands (the words after the first).
fn instructions(words: &[u32]) -> Vec<(u32, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        let Some(operands) = words.get(i + 1..i + len.max(1)) else {
            break;
        };
        out.push((w & 0xFFFF, operands));
        if len == 0 {
            break;
        }
        i += len;
    }
    out
}

/// The struct types of `words` with a `bool` vector member.
fn structs_with_bool_vectors(words: &[u32]) -> Vec<u32> {
    let insts = instructions(words);
    let bools: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, _)| *op == OP_TYPE_BOOL)
        .filter_map(|(_, o)| o.first().copied())
        .collect();
    let vectors: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, o)| *op == OP_TYPE_VECTOR && o.get(1).is_some_and(|c| bools.contains(c)))
        .filter_map(|(_, o)| o.first().copied())
        .collect();
    insts
        .iter()
        .filter(|(op, o)| *op == OP_TYPE_STRUCT && o[1..].iter().any(|m| vectors.contains(m)))
        .filter_map(|(_, o)| o.first().copied())
        .collect()
}

/// Whether a function other than the entry point names the `SampleMask` input variable.
fn sample_mask_input_outside_entry(words: &[u32]) -> bool {
    let insts = instructions(words);
    let entry = insts
        .iter()
        .find(|(op, _)| *op == OP_ENTRY_POINT)
        .and_then(|(_, o)| o.get(1).copied());
    let masks: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, o)| {
            *op == OP_DECORATE
                && o.get(1) == Some(&DECORATION_BUILTIN)
                && o.get(2) == Some(&BUILTIN_SAMPLE_MASK)
        })
        .filter_map(|(_, o)| o.first().copied())
        .collect();
    let inputs: BTreeSet<u32> = insts
        .iter()
        .filter(|(op, o)| *op == OP_VARIABLE && o.get(2) == Some(&STORAGE_INPUT))
        .filter_map(|(_, o)| o.get(1).copied())
        .filter(|v| masks.contains(v))
        .collect();
    let mut function = None;
    for (op, o) in &insts {
        match *op {
            OP_FUNCTION => function = o.get(1).copied(),
            OP_FUNCTION_END => function = None,
            _ if function.is_some()
                && function != entry
                && o.iter().any(|w| inputs.contains(w)) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// On macOS, the MSL of a MoltenVK profile's module compiles; elsewhere, or without the tools,
/// the check skips visibly.
fn compiles_as_moltenvk_msl(stem: &str, words: &[u32], what: &str) {
    if !stem.starts_with("moltenvk") {
        return;
    }
    match moltenvk_msl_compile(words) {
        Ok(()) => {}
        Err(AssembleError::Missing(why)) => eprintln!("SKIPPED ({what}): {why}"),
        Err(AssembleError::Rejected(why)) => panic!("{what}: MoltenVK's MSL: {why}"),
    }
}

/// `tail_vertex_bool` returns a struct with a `bool2` member. On every profile its module holds the
/// member as `u32` lanes (no struct has a `bool` vector member), passes L3, and on macOS compiles
/// as MoltenVK's MSL.
#[test]
fn a_bool_vector_member_is_held_as_words() {
    let Some(lib) = library("stage_io_tail") else {
        return;
    };
    for (stem, dev) in devices() {
        let f = FunctionDescriptor::new(lib.function("tail_vertex_bool").expect("vertex"));
        let c = match dev.compile_function(&f, &LoweringInputs::new(), &Budget::standard()) {
            Ok(c) => c,
            Err(Error::Unsupported(u)) => {
                eprintln!("{stem}: refused {u:?}");
                continue;
            }
            Err(e) => panic!("{stem}: {e}"),
        };
        let words = link_standalone(c.artifact(), dev.profile())
            .unwrap_or_else(|e| panic!("{stem}: link: {e}"))
            .1;
        assert_eq!(
            structs_with_bool_vectors(&words),
            Vec::<u32>::new(),
            "{stem}"
        );
        check_words_valid(&words, &c.artifact().requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        compiles_as_moltenvk_msl(&stem, &words, &format!("{stem} tail_vertex_bool"));
    }
}

/// `msf_fetch_mask_in` reads `[[sample_mask]]` beside a framebuffer fetch. Wherever it compiles to
/// the interlocked storage form (the entry calls the body inside the critical section), at one and
/// four samples, only the entry reads the `SampleMask` input; every module passes L3 and, on macOS,
/// compiles as MoltenVK's MSL.
#[test]
fn only_the_entry_reads_the_coverage_of_an_interlocked_body() {
    let Some(lib) = library("msf_fetch_ms") else {
        return;
    };
    let mut wrapped = 0;
    for (stem, dev) in devices() {
        for samples in [1u32, 4] {
            let what = format!("{stem} msf_fetch_mask_in at {samples}");
            let colors = vec![Some(ColorAttachmentDescriptor::new(
                PixelFormat::RGBA32Float,
            ))];
            let mut inputs = LoweringInputs::new();
            inputs.framebuffer_fetch =
                Some(FramebufferFetchInputs::of(0..1, &colors, false, false, 1));
            inputs.multisampled = samples > 1;
            let f = FunctionDescriptor::new(lib.function("msf_fetch_mask_in").expect("fragment"));
            let c = match dev.compile_function(&f, &inputs, &Budget::standard()) {
                Ok(c) => c,
                Err(Error::Unsupported(_) | Error::NeedsInput(_)) => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            let words = link_standalone(c.artifact(), dev.profile())
                .unwrap_or_else(|e| panic!("{what}: link: {e}"))
                .1;
            let functions = instructions(&words)
                .iter()
                .filter(|(op, _)| *op == OP_FUNCTION)
                .count();
            wrapped += usize::from(functions > 1);
            assert!(!sample_mask_input_outside_entry(&words), "{what}");
            check_words_valid(&words, &c.artifact().requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            compiles_as_moltenvk_msl(&stem, &words, &what);
        }
    }
    assert!(wrapped >= 2, "{wrapped} interlocked modules");
}

/// The loop over the covered samples of `rt6_mark_coverage` at four samples reads the coverage in
/// the entry, and the body reads the entry's copy, on every profile where it compiles; on macOS the
/// MoltenVK profiles' modules compile as MoltenVK's MSL.
#[test]
fn only_the_entry_reads_the_coverage_of_a_looped_body() {
    let Some(lib) = library("rt6_attachmentless") else {
        return;
    };
    let mut looped = 0;
    for (stem, dev) in devices() {
        let what = format!("{stem} rt6_mark_coverage at 4");
        let vertex = StageSource::Function(FunctionDescriptor::new(
            lib.function("rt6_cover").expect("vertex"),
        ));
        let mut d = RenderPipelineDescriptor::new(vertex);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            lib.function("rt6_mark_coverage").expect("fragment"),
        )));
        d.raster_sample_count = 4;
        let p = match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(p) => p,
            Err(Error::Unsupported(_)) => continue,
            Err(e) => panic!("{what}: {e}"),
        };
        let words = &p
            .stages
            .iter()
            .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
            .expect("fragment stage")
            .spirv;
        looped += 1;
        assert!(!sample_mask_input_outside_entry(words), "{what}");
        compiles_as_moltenvk_msl(&stem, words, &what);
    }
    assert!(looped >= 2, "{looped} looped modules");
}
