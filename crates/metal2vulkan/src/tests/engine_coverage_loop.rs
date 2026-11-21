//! The loop over the covered samples over the owned fixtures `engine/rt6_attachmentless`,
//! `engine/sample_rate`, `engine/coverage_loop`, `engine/coverage_natural` and
//! `engine/coverage_outputs`:
//!
//! - `rt6_mark_coverage` reads `[[sample_mask]]` and writes memory without a sample index or a
//!   sample-qualified varying. Apple's GPUs were measured to run it once per covered sample, and
//!   documented Metal names no count, so it runs as the device runs it
//!   (`PrecisionNote::DeviceFragmentRuns`): compiled as written, with no loop, no `SampleId` and no
//!   `SampleRateShading`, reading `SampleMask`. Its interface needs no loop
//!   (`CoverageLoop::NotNeeded`), and compiled on its own it links at four samples.
//! - `sample_rate_mask_in` returns a color beside `[[sample_mask]]` and `[[sample_id]]`: Metal runs
//!   it once per covered sample, and compiled for a multisampled pipeline it runs once per pixel,
//!   without sample shading, and runs its body from a loop over the coverage's set bits
//!   (`FindILsb`), its color the last pass's (`PrecisionNote::LoopedSampleOutputs`). Its interface
//!   says so (`CoverageLoop::Looped`); compiled on its own it says `Unlooped`, and link asks for
//!   `multisampled` (`NeedsInput(Multisampled)`) at four samples.
//! - `coverage_outputs`' memory writers return a color, one also a sample mask, without a sample
//!   index: they run as the device runs them, with no loop, recording `DeviceFragmentRuns`.
//! - `rt6_mark_sample` runs per sample without reading the coverage: no loop.
//! - `coverage_loop`'s memory writers read `[[sample_id]]` and loop at four samples wherever the
//!   device makes them; each of
//!   its functions holding one thing the loop does not fit (early fragment tests, a discard, a
//!   derivative, a simdgroup operation, a color attachment read) is refused by that thing; its
//!   sample-qualified varying is interpolated at its samples (`InterpolateAtSample`, no `Sample`
//!   decoration), or refused as needing the interpolation functions where withheld.
//!
//! - `coverage_natural`'s functions read no `[[position]]` beside a sample index or varying: each
//!   that reads one loops, and each that reads neither runs as the device runs it.
//! - The looping functions' sample-qualified varyings are read at every sample a pipeline of the
//!   device may have, by constant, before the loop, where the whole quad runs, on every profile.
//!
//! Every pipeline that compiles passes L3 (`spirv-val` and requirements within the profile).
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs, RuntimeInput, Unsupported};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{CoverageLoop, CoverageLoopRefusal};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;

const COVER: &str = "rt6_cover";
const COVERAGE: &str = "rt6_mark_coverage";
const SAMPLE: &str = "rt6_mark_sample";
const MASK_IN: &str = "sample_rate_mask_in";

/// `coverage_loop`'s vertex functions, the functions the loop fits, and the ones it refuses with
/// what each is refused for.
const COV_COVER: &str = "cov_cover";
const COV_COVER_UV: &str = "cov_cover_uv";
const COV_FITS: [&str; 2] = ["cov_loop_record", "cov_loop_union"];
const COV_VARYING: &str = "cov_loop_varying";
const COV_FETCH: &str = "cov_loop_fetch";
/// `coverage_natural`'s functions, each with the `coverage_loop` vertex function that feeds it and
/// whether it loops: those that read a sample index or a sample-qualified varying.
const NATURAL: [(&str, &str, bool); 4] = [
    ("nat_coverage", COV_COVER, false),
    ("nat_coverage_at_pixel", COV_COVER, false),
    ("nat_sample", COV_COVER, true),
    ("nat_varying", COV_COVER_UV, true),
];
const COV_REFUSED: [(&str, CoverageLoopRefusal); 5] = [
    ("cov_loop_early", CoverageLoopRefusal::EarlyFragmentTests),
    ("cov_loop_discard", CoverageLoopRefusal::Discard),
    ("cov_loop_derivative", CoverageLoopRefusal::QuadOperations),
    ("cov_loop_simdgroup", CoverageLoopRefusal::QuadOperations),
    (COV_FETCH, CoverageLoopRefusal::Attachments),
];

/// SPIR-V opcodes, decorations, builtins, capabilities, and `GLSL.std.450` numbers read here.
const OP_CAPABILITY: u32 = 17;
const OP_EXT_INST: u32 = 12;
const OP_DECORATE: u32 = 71;
const OP_LOOP_MERGE: u32 = 246;
const DECORATION_SAMPLE: u32 = 17;
const DECORATION_BUILTIN: u32 = 11;
const BUILTIN_SAMPLE_ID: u32 = 18;
const BUILTIN_SAMPLE_MASK: u32 = 20;
const BUILTIN_FRAG_COORD: u32 = 15;
const CAPABILITY_SAMPLE_RATE_SHADING: u32 = 35;
const FIND_ILSB: u32 = 73;
const INTERPOLATE_AT_SAMPLE: u32 = 77;
const OP_CONSTANT: u32 = 43;

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

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::default()).expect("device")
}

/// Every committed profile's stem, sorted.
fn stems() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("profile directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().expect("stem").to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

/// The pipeline of `rt6_cover` and `fragment` from `fragments` at `samples`, with no attachment
/// unless `color`.
fn render(
    cover: &Library,
    fragment: StageSource,
    samples: u32,
    color: bool,
) -> RenderPipelineDescriptor {
    let vertex = StageSource::Function(FunctionDescriptor::new(
        cover.function(COVER).expect("vertex"),
    ));
    let mut d = RenderPipelineDescriptor::new(vertex);
    d.fragment = Some(fragment);
    if color {
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    d.raster_sample_count = samples;
    d
}

fn function(lib: &Library, name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        lib.function(name).expect("function"),
    ))
}

/// The fragment module of `p`.
fn fragment_words(p: &RenderPipeline) -> &[u32] {
    &p.stages
        .iter()
        .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
        .expect("fragment stage")
        .spirv
}

/// What the fragment module of a looping function must and must not hold.
#[derive(Debug, Default, PartialEq, Eq)]
struct Shape {
    sample_id: bool,
    sample_mask: bool,
    sample_rate_shading: bool,
    loops: bool,
    find_lsb: bool,
    at_sample: bool,
    sample_decorated: bool,
    frag_coord: bool,
}

fn shape(words: &[u32]) -> Shape {
    let mut s = Shape::default();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        let at = |k: usize| words.get(i + k).copied().unwrap_or(u32::MAX);
        match op {
            OP_DECORATE if at(2) == DECORATION_BUILTIN => {
                s.sample_id |= at(3) == BUILTIN_SAMPLE_ID;
                s.sample_mask |= at(3) == BUILTIN_SAMPLE_MASK;
                s.frag_coord |= at(3) == BUILTIN_FRAG_COORD;
            }
            OP_DECORATE => s.sample_decorated |= at(2) == DECORATION_SAMPLE,
            OP_CAPABILITY => s.sample_rate_shading |= at(1) == CAPABILITY_SAMPLE_RATE_SHADING,
            OP_LOOP_MERGE => s.loops = true,
            OP_EXT_INST => {
                s.find_lsb |= at(4) == FIND_ILSB;
                s.at_sample |= at(4) == INTERPOLATE_AT_SAMPLE;
            }
            _ => {}
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    s
}

/// The sample operand of each `InterpolateAtSample` in `words`, in module order: its value where
/// it is a 32-bit constant, `None` where it is computed.
fn interpolated_samples(words: &[u32]) -> Vec<Option<u32>> {
    let mut constants = std::collections::BTreeMap::new();
    let mut reads = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        let at = |k: usize| words.get(i + k).copied().unwrap_or(u32::MAX);
        match op {
            OP_CONSTANT if len == 4 => {
                constants.insert(at(2), at(3));
            }
            OP_EXT_INST if at(4) == INTERPOLATE_AT_SAMPLE => {
                reads.push(constants.get(&at(6)).copied());
            }
            _ => {}
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    reads
}

/// Whether a stage of `p` records `note`.
fn noted(p: &RenderPipeline, note: PrecisionNote) -> bool {
    p.stages.iter().any(|m| m.precision_notes.contains(&note))
}

/// On every committed profile, the coverage-reading memory writer without a sample index at two
/// and four samples runs as the device runs it: compiled as written, with no loop, no `SampleId`
/// and no `SampleRateShading`, reading `SampleMask`, and recording
/// `PrecisionNote::DeviceFragmentRuns`; or refused for what the device lacks (fragment stores),
/// never as not representable. At one sample it compiles the same, recording nothing of its runs.
#[test]
fn a_coverage_reading_memory_writer_runs_as_the_device_runs_it() {
    let Some(lib) = library("rt6_attachmentless") else {
        return;
    };
    let mut plain = 0;
    for stem in stems() {
        let dev = device(&stem);
        for samples in [2, 4] {
            let what = format!("{stem} at {samples}");
            let d = render(&lib, function(&lib, COVERAGE), samples, false);
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Ok(p) => {
                    plain += 1;
                    assert_eq!(
                        Shape {
                            frag_coord: false,
                            ..shape(fragment_words(&p))
                        },
                        Shape {
                            sample_mask: true,
                            ..Shape::default()
                        },
                        "{what}"
                    );
                    assert!(noted(&p, PrecisionNote::DeviceFragmentRuns), "{what}");
                    for m in &p.stages {
                        check_words_valid(&m.spirv, &p.requirements, dev.profile())
                            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                    }
                }
                Err(Error::Unsupported(u)) => {
                    assert_ne!(
                        u.feature,
                        Feature::PixelCoverageAtSampleRate,
                        "{what}: {u:?}"
                    );
                }
                Err(e) => panic!("{what}: {e}"),
            }
        }
        let d = render(&lib, function(&lib, COVERAGE), 1, false);
        if let Ok(p) = dev.compile_render_pipeline(&d, &Budget::standard()) {
            let s = shape(fragment_words(&p));
            assert!(!s.loops && !s.sample_id, "{stem} at 1: {s:?}");
            assert!(!noted(&p, PrecisionNote::DeviceFragmentRuns), "{stem} at 1");
        }
    }
    assert!(plain >= 6, "{plain} pipelines compiled it as written");
}

/// A coverage reader in a multisampled pipeline records `PrecisionNote::ClippedCoverageSplit` on
/// every device, as no Vulkan query states that a clipped primitive is split into the pieces Metal
/// hands separate runs; at one sample, where the coverage is the one sample's, it records nothing
/// anywhere.
#[test]
fn a_multisampled_coverage_reader_announces_the_split() {
    let Some(lib) = library("rt6_attachmentless") else {
        return;
    };
    let note = PrecisionNote::ClippedCoverageSplit;
    let notes = |p: &RenderPipeline| -> Vec<PrecisionNote> {
        p.stages
            .iter()
            .flat_map(|m| m.precision_notes.iter().copied())
            .collect()
    };
    let mut noted = 0;
    for stem in stems() {
        let dev = device(&stem);
        let d = render(&lib, function(&lib, COVERAGE), 4, false);
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(p) => {
                assert!(notes(&p).contains(&note), "{stem} at 4");
                noted += 1;
            }
            Err(Error::Unsupported(_)) => {}
            Err(e) => panic!("{stem} at 4: {e}"),
        }
        let d = render(&lib, function(&lib, COVERAGE), 1, false);
        if let Ok(p) = dev.compile_render_pipeline(&d, &Budget::standard()) {
            assert!(!notes(&p).contains(&note), "{stem} at 1");
        }
    }
    assert!(noted >= 7, "{noted} pipelines noted the split");
}

/// A function that reads a centroid varying and runs once per pixel (`air_2_1_front_end`'s
/// `a21_blit_fs`) records `PrecisionNote::ClippedCoverageSplit` in a multisampled pipeline on every
/// device, its centroid the whole coverage's where Metal's may be a piece's; at one sample it records
/// nothing, and neither does one that runs per sample (`multisample`'s
/// `multisample_centroid_at_sample`), which reads the varying at its own sample.
#[test]
fn a_multisampled_centroid_reader_at_the_pixel_announces_the_split() {
    let (Some(a21), Some(ms)) = (library("air_2_1_front_end"), library("multisample")) else {
        return;
    };
    let note = PrecisionNote::ClippedCoverageSplit;
    let mut noted = 0;
    for stem in stems() {
        let dev = device(&stem);
        for (lib, vertex, fragment, samples, recorded) in [
            (&a21, "a21_blit_vs", "a21_blit_fs", 4, true),
            (&a21, "a21_blit_vs", "a21_blit_fs", 1, false),
            (
                &ms,
                "multisample_vertex",
                "multisample_centroid_at_sample",
                4,
                false,
            ),
        ] {
            let mut d = RenderPipelineDescriptor::new(function(lib, vertex));
            d.fragment = Some(function(lib, fragment));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            d.raster_sample_count = samples;
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Ok(p) => {
                    let notes: Vec<PrecisionNote> = (p.stages.iter())
                        .flat_map(|m| m.precision_notes.iter().copied())
                        .collect();
                    assert_eq!(
                        notes.contains(&note),
                        recorded,
                        "{stem}: {fragment} at {samples}"
                    );
                    noted += usize::from(recorded);
                }
                Err(Error::Unsupported(_)) => {}
                Err(e) => panic!("{stem}: {fragment} at {samples}: {e}"),
            }
        }
    }
    assert!(noted >= 7, "{noted} pipelines noted the split");
}

/// The interface states the loop: `Looped` under `multisampled`, `Unlooped` without, and link asks
/// a function compiled without it for `multisampled` at four samples; a function that runs per
/// sample without reading the coverage has no loop, and neither has a memory writer that reads the
/// coverage without a sample index, which compiled on its own links at four samples and at one; a
/// looping function with a color output records `PrecisionNote::LoopedSampleOutputs`.
#[test]
fn the_interface_states_the_coverage_loop() {
    let (Some(attachmentless), Some(sample_rate)) =
        (library("rt6_attachmentless"), library("sample_rate"))
    else {
        return;
    };
    let dev = device("amd-780m-radv");
    let mut multisampled = LoweringInputs::new();
    multisampled.multisampled = true;
    let compile = |lib: &Library, name: &str, inputs: &LoweringInputs| {
        let f = FunctionDescriptor::new(lib.function(name).expect("function"));
        dev.compile_function(&f, inputs, &Budget::standard())
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let state = |lib: &Library, name: &str, inputs: &LoweringInputs| {
        compile(lib, name, inputs)
            .artifact()
            .interface
            .io
            .coverage_loop
    };
    let none = LoweringInputs::new();
    for (lib, name, alone, under) in [
        (
            &attachmentless,
            COVERAGE,
            CoverageLoop::NotNeeded,
            CoverageLoop::NotNeeded,
        ),
        (
            &attachmentless,
            SAMPLE,
            CoverageLoop::NotNeeded,
            CoverageLoop::NotNeeded,
        ),
        (
            &sample_rate,
            MASK_IN,
            CoverageLoop::Unlooped,
            CoverageLoop::Looped,
        ),
    ] {
        assert_eq!(state(lib, name, &none), alone, "{name} alone");
        assert_eq!(
            state(lib, name, &multisampled),
            under,
            "{name} multisampled"
        );
    }
    // Compiled on its own, the memory writer links at four samples, as the device runs it.
    let alone = StageSource::Compiled(compile(&attachmentless, COVERAGE, &none));
    let budget = Budget::standard();
    let four = dev
        .compile_render_pipeline(&render(&attachmentless, alone.clone(), 4, false), &budget)
        .expect("four samples");
    assert!(noted(&four, PrecisionNote::DeviceFragmentRuns));
    dev.compile_render_pipeline(&render(&attachmentless, alone, 1, false), &budget)
        .expect("one sample");
    // A color output beside the coverage at sample rate loops, its color the last pass's.
    let alone = StageSource::Compiled(compile(&sample_rate, MASK_IN, &none));
    assert_eq!(
        dev.compile_render_pipeline(&render(&attachmentless, alone, 4, true), &budget)
            .map(|_| ()),
        Err(Error::NeedsInput(RuntimeInput::Multisampled))
    );
    let p = dev
        .compile_render_pipeline(
            &render(&attachmentless, function(&sample_rate, MASK_IN), 4, true),
            &budget,
        )
        .expect("the looping color writer");
    let s = shape(fragment_words(&p));
    assert!(s.loops && s.find_lsb && !s.sample_id, "{s:?}");
    assert!(p.stages.iter().any(|m| m
        .precision_notes
        .contains(&PrecisionNote::LoopedSampleOutputs)));
    assert_eq!(
        CoverageLoopRefusal::EarlyFragmentTests.error(),
        Error::Unsupported(Unsupported::not_representable(
            Feature::PixelCoverageAtSampleRate
        ))
    );
    let Error::Unsupported(u) = CoverageLoopRefusal::Discard.error() else {
        panic!("not a refusal");
    };
    assert_eq!(
        (u.feature, u.needs),
        (Feature::PixelCoverageAtSampleRate, Needs::NotImplemented)
    );
}

/// The pipeline of `coverage_loop`'s `vertex` and `fragment` on `dev` at four samples, with an
/// `R32Uint` color attachment for the function that reads one and none otherwise.
fn coverage_loop_pipeline(
    dev: &Device,
    lib: &Library,
    vertex: &str,
    fragment: &str,
) -> Result<RenderPipeline, Error> {
    let mut d = RenderPipelineDescriptor::new(function(lib, vertex));
    d.fragment = Some(function(lib, fragment));
    if fragment == COV_FETCH {
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    }
    d.raster_sample_count = 4;
    dev.compile_render_pipeline(&d, &Budget::standard())
}

/// Every stage of `p` passes L3 on `dev`.
fn l3(dev: &Device, p: &RenderPipeline, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// Over `coverage_loop` on every committed profile at four samples: where the device makes the
/// memory writers, they loop without sample shading, each function holding one thing the loop
/// does not fit is refused for that thing (or first for a feature the device lacks, as the color
/// attachment read where neither rasterization-order access nor pixel interlock exists, but never
/// made), and the sample-qualified varying loops, read at each
/// pass's sample through `InterpolateAtSample` with no `Sample` decoration, or is refused as
/// needing the interpolation functions on a device that withholds them. A device that does not
/// make the memory writers refuses them for what it lacks, never for the loop, and every other
/// function by its reason.
#[test]
fn the_loop_refuses_each_function_it_does_not_fit_by_its_reason() {
    let Some(lib) = library("coverage_loop") else {
        return;
    };
    let looped = Shape {
        sample_mask: true,
        loops: true,
        find_lsb: true,
        ..Shape::default()
    };
    let (mut fitted, mut varying, mut refused) = (0, 0, [0; COV_REFUSED.len()]);
    for stem in stems() {
        let dev = device(&stem);
        let fits = COV_FITS.map(|f| coverage_loop_pipeline(&dev, &lib, COV_COVER, f));
        if fits.iter().any(Result::is_err) {
            for (f, made) in COV_FITS.iter().zip(&fits) {
                match made {
                    Ok(_) => {}
                    Err(Error::Unsupported(u)) => assert_ne!(
                        u.feature,
                        Feature::PixelCoverageAtSampleRate,
                        "{f} on {stem}: {u:?}"
                    ),
                    Err(e) => panic!("{f} on {stem}: {e}"),
                }
            }
            continue;
        }
        for (f, p) in COV_FITS.iter().zip(&fits) {
            let Ok(p) = p else {
                continue;
            };
            let what = format!("{f} on {stem}");
            assert_eq!(
                Shape {
                    frag_coord: false,
                    ..shape(fragment_words(p))
                },
                looped,
                "{what}"
            );
            l3(&dev, p, &what);
        }
        fitted += 1;
        for (k, (f, refusal)) in COV_REFUSED.into_iter().enumerate() {
            match coverage_loop_pipeline(&dev, &lib, COV_COVER, f).map(|_| ()) {
                Err(e) if e == refusal.error() => refused[k] += 1,
                Err(Error::Unsupported(u))
                    if u.feature != Feature::PixelCoverageAtSampleRate
                        && matches!(u.needs, Needs::AnyOf(_)) => {}
                other => panic!("{f} on {stem}: {other:?}, not {:?}", refusal.error()),
            }
        }
        let what = format!("{COV_VARYING} on {stem}");
        match coverage_loop_pipeline(&dev, &lib, COV_COVER_UV, COV_VARYING) {
            Ok(p) => {
                varying += 1;
                assert_eq!(
                    Shape {
                        frag_coord: false,
                        ..shape(fragment_words(&p))
                    },
                    Shape {
                        at_sample: true,
                        ..looped
                    },
                    "{what}"
                );
                l3(&dev, &p, &what);
            }
            Err(e) => assert!(
                e == CoverageLoopRefusal::InterpolationFunctions.error(),
                "{what}: {e:?}"
            ),
        }
    }
    assert!(fitted >= 6, "{fitted} profiles made the memory writers");
    for ((f, _), n) in COV_REFUSED.iter().zip(refused) {
        assert!(n >= 6, "{f} refused by its reason on {n} profiles");
    }
    assert!(varying >= 6, "{varying} profiles looped the varying");
}

/// `coverage_natural` at 2 and 4 samples on every committed profile: each function that reads a
/// sample index or a sample-qualified varying loops without sample shading, its coverage read, and
/// each that reads neither compiles as written, with no loop; every one, as a memory writer,
/// records `PrecisionNote::DeviceFragmentRuns`. Or it is refused for what the device lacks (the
/// interpolation functions among them), never for the loop.
#[test]
fn the_natural_functions_loop_where_they_read_a_sample() {
    let (Some(natural_lib), Some(vertex_lib)) =
        (library("coverage_natural"), library("coverage_loop"))
    else {
        return;
    };
    let mut compiled = 0;
    for stem in stems() {
        let dev = device(&stem);
        for (f, vertex, loops) in NATURAL {
            for samples in [2, 4] {
                let what = format!("{f} at {samples} on {stem}");
                let mut d = RenderPipelineDescriptor::new(function(&vertex_lib, vertex));
                d.fragment = Some(function(&natural_lib, f));
                d.raster_sample_count = samples;
                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(p) => {
                        let s = shape(fragment_words(&p));
                        assert!(s.sample_mask, "{what}: {s:?}");
                        assert_eq!((s.loops, s.find_lsb), (loops, loops), "{what}: {s:?}");
                        assert!(!s.sample_rate_shading, "{what}: {s:?}");
                        assert!(noted(&p, PrecisionNote::DeviceFragmentRuns), "{what}");
                        compiled += 1;
                        l3(&dev, &p, &what);
                    }
                    Err(e) if e == CoverageLoopRefusal::InterpolationFunctions.error() => {}
                    Err(Error::Unsupported(u)) => {
                        assert_ne!(
                            u.feature,
                            Feature::PixelCoverageAtSampleRate,
                            "{what}: {u:?}"
                        );
                    }
                    Err(e) => panic!("{what}: {e}"),
                }
            }
        }
    }
    assert!(
        compiled >= 6 * NATURAL.len(),
        "{compiled} pipelines compiled"
    );
}

/// A looping function's sample-qualified varying (`coverage_loop`'s `cov_loop_varying`, and
/// `coverage_natural`'s `nat_varying`, at 2 and 4 samples) is read at every sample a pipeline of
/// the device may have, each by its constant index from 0, a power of two of them and at least
/// the pipeline's samples, and at no computed sample, on every profile that loops it: before the
/// loop, which the helper invocations of a partly covered quad leave at once. Each varying is
/// perspective (`sample_perspective`), so every module records
/// `PrecisionNote::ExtrapolatedInterpolation`.
#[test]
fn a_looping_function_reads_its_sample_varying_at_every_sample_before_the_loop() {
    let (Some(loop_lib), Some(natural_lib)) =
        (library("coverage_loop"), library("coverage_natural"))
    else {
        return;
    };
    let mut before = 0;
    for stem in stems() {
        let dev = device(&stem);
        for (lib, f) in [(&loop_lib, COV_VARYING), (&natural_lib, "nat_varying")] {
            for samples in [2, 4] {
                let what = format!("{f} at {samples} on {stem}");
                let mut d = RenderPipelineDescriptor::new(function(&loop_lib, COV_COVER_UV));
                d.fragment = Some(function(lib, f));
                d.raster_sample_count = samples;
                let Ok(p) = dev.compile_render_pipeline(&d, &Budget::standard()) else {
                    continue;
                };
                let words = fragment_words(&p);
                if !shape(words).loops {
                    continue;
                }
                assert!(
                    p.stages.iter().any(|m| m
                        .precision_notes
                        .contains(&PrecisionNote::ExtrapolatedInterpolation)),
                    "{what}: the extrapolation note"
                );
                before += 1;
                let reads = interpolated_samples(words);
                let n = reads.len() as u32;
                assert!(n.is_power_of_two() && n >= samples, "{what}: {reads:?}");
                let every: Vec<Option<u32>> = (0..n).map(Some).collect();
                assert_eq!(reads, every, "{what}");
                l3(&dev, &p, &what);
            }
        }
    }
    assert!(
        before >= 6 * 4,
        "{before} pipelines read it before the loop"
    );
}

/// How many `BuiltIn SampleMask` decorations `words` holds: the input's, and the output's where
/// the module writes one.
fn sample_mask_decorations(words: &[u32]) -> usize {
    let mut n = 0;
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        let at = |k: usize| words.get(i + k).copied().unwrap_or(u32::MAX);
        n += usize::from(
            op == OP_DECORATE && at(2) == DECORATION_BUILTIN && at(3) == BUILTIN_SAMPLE_MASK,
        );
        if len == 0 {
            break;
        }
        i += len;
    }
    n
}

/// `coverage_outputs`' memory writers return a color beside `[[sample_mask]]`, one also a sample
/// mask, and read no sample index: Apple's GPUs were measured to write them per covered sample,
/// each from that sample's run (our probe `co`), and documented Metal names no count. At two and
/// four samples on every committed profile each runs as the device runs it, compiled with no loop
/// and recording `PrecisionNote::DeviceFragmentRuns` and not `LoopedSampleOutputs`, or is refused
/// for what the device lacks, never as the coverage at sample rate; the masked one declares the
/// `SampleMask` output beside the input.
#[test]
fn coverage_reading_color_writers_run_as_the_device_runs_them() {
    let Some(lib) = library("coverage_outputs") else {
        return;
    };
    let mut plain = 0;
    for stem in stems() {
        let dev = device(&stem);
        for (name, masks) in [("cvo_color", 1), ("cvo_masked", 2)] {
            for samples in [2, 4] {
                let what = format!("{name} on {stem} at {samples}");
                let mut d = RenderPipelineDescriptor::new(function(&lib, "cvo_vertex"));
                d.fragment = Some(function(&lib, name));
                d.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Uint));
                d.raster_sample_count = samples;
                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(p) => {
                        let words = fragment_words(&p);
                        let s = shape(words);
                        assert_eq!(
                            (s.loops, s.find_lsb, s.sample_id),
                            (false, false, false),
                            "{what}"
                        );
                        assert_eq!(sample_mask_decorations(words), masks, "{what}");
                        assert!(noted(&p, PrecisionNote::DeviceFragmentRuns), "{what}");
                        assert!(!noted(&p, PrecisionNote::LoopedSampleOutputs), "{what}");
                        for m in &p.stages {
                            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                                .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                        }
                        plain += 1;
                    }
                    Err(Error::Unsupported(u)) => {
                        assert_ne!(
                            u.feature,
                            Feature::PixelCoverageAtSampleRate,
                            "{what}: {u:?}"
                        );
                    }
                    Err(e) => panic!("{what}: {e}"),
                }
            }
        }
    }
    assert!(plain >= 12, "{plain} compiled as written");
}
