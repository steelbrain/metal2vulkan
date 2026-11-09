//! The loop over the covered samples over the owned fixtures `engine/rt6_attachmentless`,
//! `engine/sample_rate`, `engine/coverage_loop`, `engine/coverage_natural` and
//! `engine/coverage_outputs`:
//!
//! - `rt6_mark_coverage` reads `[[sample_mask]]` and writes memory, so Metal runs it once per
//!   covered sample, each run reading the pixel's coverage. Compiled for a multisampled pipeline it
//!   runs once per pixel, without sample shading: the module declares no `SampleId` and no
//!   `SampleRateShading`, reads `SampleMask`, and runs the body from a loop over the coverage's set
//!   bits (`FindILsb`). Its interface says so (`CoverageLoop::Looped`); compiled on its own it says
//!   `Unlooped`, and link asks for `multisampled` (`NeedsInput(Multisampled)`) at four samples
//!   and takes it at one.
//! - `sample_rate_mask_in` returns a color beside `[[sample_mask]]` and `[[sample_id]]`: it loops
//!   as the memory writer does, its color the last pass's (`PrecisionNote::LoopedSampleOutputs`).
//! - `coverage_outputs`' memory writers return a color, one also a sample mask: they loop
//!   wherever the device makes them, recording the note, the masked one storing the union of the
//!   samples its passes keep after the loop; on MoltenVK on Apple's GPU each runs as written,
//!   exactly, without the note.
//! - `rt6_mark_sample` runs per sample without reading the coverage: no loop.
//! - `coverage_loop`'s memory writers loop at four samples wherever the device makes them; each of
//!   its functions holding one thing the loop does not fit (early fragment tests, a discard, a
//!   derivative, a simdgroup operation, a color attachment read) is refused by that thing; its
//!   sample-qualified varying is interpolated at each pass's sample (`InterpolateAtSample`, no
//!   `Sample` decoration), or refused as needing the interpolation functions where withheld.
//!
//! - `coverage_natural`'s functions read no `[[position]]` beside a sample index or varying: on a
//!   driver that runs a memory writer once per covered sample (MoltenVK on Apple's GPU) each runs
//!   as written, without the loop or sample shading, and never declares `FragCoord` beside
//!   `SampleRateShading`; elsewhere each loops.
//! - Where the driver's interpolation at a sample needs the whole quad (RADV), the looping
//!   functions' sample-qualified varyings are read before the loop, at every sample a pipeline of
//!   the device may have, each by a constant; elsewhere once, inside the loop.
//!
//! Every pipeline that compiles passes L3 (`spirv-val` and requirements within the profile).
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy, DriverWorkaround};
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
/// `coverage_natural`'s functions, each with the `coverage_loop` vertex function that feeds it.
const NATURAL: [(&str, &str); 4] = [
    ("nat_coverage", COV_COVER),
    ("nat_coverage_at_pixel", COV_COVER),
    ("nat_sample", COV_COVER),
    ("nat_varying", COV_COVER_UV),
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
    Device::new(description(stem), DevicePolicy::default()).expect("device")
}

/// The refusal of a memory writer on a driver that runs it once per covered sample where that
/// driver would mask its coverage, where `dev`'s profile carries that workaround.
fn per_sample_defect(dev: &Device) -> Option<Error> {
    let defect = DriverWorkaround::MemoryWritingFragmentsRunPerSample;
    dev.profile()
        .fields()
        .workarounds
        .contains(&defect)
        .then(|| Unsupported::driver_defect(Feature::PixelCoverageAtSampleRate, defect).into())
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
/// (as `runtime_lane::attachmentless` draws it) unless `color`.
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

/// The most samples a pipeline of `dev` may rasterize: its largest framebuffer sample count.
fn most_samples(dev: &Device) -> u32 {
    let l = &dev.profile().fields().limits;
    let all = [
        l.framebuffer_color_sample_counts,
        l.framebuffer_integer_color_sample_counts,
        l.framebuffer_depth_sample_counts,
        l.framebuffer_stencil_sample_counts,
        l.framebuffer_no_attachments_sample_counts,
    ]
    .iter()
    .fold(0, |all, c| all | c.as_raw());
    1 << all.max(1).ilog2()
}

/// On every committed profile, the coverage-reading memory writer at two and four samples runs
/// once per pixel and loops over its covered samples, with no sample shading, or is refused for
/// what the device lacks (fragment stores, lavapipe's split of a clipped triangle's fragments),
/// never as not representable. On a driver that runs a memory writer once per covered sample
/// (MoltenVK on Apple's GPU) it is compiled as written, which reads no sample index and so keeps
/// the coverage whole there: no loop, no `SampleId`, no `SampleRateShading`, and no sample shading.
/// At one sample it compiles without the loop.
#[test]
fn a_coverage_reading_memory_writer_loops_over_its_covered_samples() {
    let Some(lib) = library("rt6_attachmentless") else {
        return;
    };
    let mut looped = 0;
    for stem in stems() {
        let dev = device(&stem);
        for samples in [2, 4] {
            let what = format!("{stem} at {samples}");
            let d = render(&lib, function(&lib, COVERAGE), samples, false);
            let driver_runs = per_sample_defect(&dev).is_some();
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Ok(p) => {
                    looped += usize::from(!driver_runs);
                    assert!(!p.graphics.sample_shading, "{what}");
                    assert_eq!(
                        Shape {
                            frag_coord: false,
                            ..shape(fragment_words(&p))
                        },
                        Shape {
                            sample_mask: true,
                            loops: !driver_runs,
                            find_lsb: !driver_runs,
                            ..Shape::default()
                        },
                        "{what}"
                    );
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
        }
    }
    assert!(looped >= 6, "{looped} pipelines looped");
}

/// A coverage reader in a multisampled pipeline records `PrecisionNote::ClippedCoverageSplit` on
/// every device whose rasterizer is not Apple's, where a clipped primitive is not split into the
/// pieces Metal hands separate runs, and not on MoltenVK on Apple's GPU, which rasterizes through
/// Metal; at one sample, where the coverage is the one sample's, it records nothing anywhere.
#[test]
fn a_multisampled_coverage_reader_off_apples_rasterizer_announces_the_split() {
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
    let (mut apple, mut other) = (0, 0);
    for stem in stems() {
        let dev = device(&stem);
        let metals = stem.starts_with("moltenvk-");
        let d = render(&lib, function(&lib, COVERAGE), 4, false);
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(p) => {
                assert_eq!(notes(&p).contains(&note), !metals, "{stem} at 4");
                *if metals { &mut apple } else { &mut other } += 1;
            }
            Err(Error::Unsupported(_)) => {}
            Err(e) => panic!("{stem} at 4: {e}"),
        }
        let d = render(&lib, function(&lib, COVERAGE), 1, false);
        if let Ok(p) = dev.compile_render_pipeline(&d, &Budget::standard()) {
            assert!(!notes(&p).contains(&note), "{stem} at 1");
        }
    }
    assert!(
        apple >= 1 && other >= 6,
        "{apple} on Apple's, {other} on others"
    );
}

/// The interface states the loop: `Looped` under `multisampled`, `Unlooped` without, and link asks
/// a function compiled without it for `multisampled` at four samples and takes it at one; a
/// function that runs per sample without reading the coverage has no loop; one with a color output
/// loops as well, recording `PrecisionNote::LoopedSampleOutputs`; on MoltenVK on Apple's GPU the
/// coverage reader runs as written (`DriverRunsPerSample`).
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
            CoverageLoop::Unlooped,
            CoverageLoop::Looped,
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
    // Compiled on its own, the coverage reader is asked for `multisampled` at four samples.
    let alone = StageSource::Compiled(compile(&attachmentless, COVERAGE, &none));
    let budget = Budget::standard();
    assert_eq!(
        dev.compile_render_pipeline(&render(&attachmentless, alone.clone(), 4, false), &budget)
            .map(|_| ()),
        Err(Error::NeedsInput(RuntimeInput::Multisampled))
    );
    dev.compile_render_pipeline(&render(&attachmentless, alone, 1, false), &budget)
        .expect("one sample");
    // A color output beside the coverage at sample rate loops too, its color the last pass's.
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
    assert!(!p.graphics.sample_shading);
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
    // On a driver that runs a memory writer once per covered sample (MoltenVK on Apple's GPU), the
    // coverage reader runs as written whether or not it was compiled for a multisampled pipeline;
    // link checks its other reads as any per-sample function's (its `[[position]]` asks for
    // `multisampled`) and takes it without sample shading.
    let mvk = device("moltenvk-m3");
    for (inputs, made) in [(&none, false), (&multisampled, true)] {
        let f = FunctionDescriptor::new(attachmentless.function(COVERAGE).expect("function"));
        let compiled = mvk
            .compile_function(&f, inputs, &budget)
            .unwrap_or_else(|e| panic!("{COVERAGE} on moltenvk-m3: {e}"));
        assert_eq!(
            compiled.artifact().interface.io.coverage_loop,
            CoverageLoop::DriverRunsPerSample
        );
        let p = mvk.compile_render_pipeline(
            &render(&attachmentless, StageSource::Compiled(compiled), 4, false),
            &budget,
        );
        match p {
            Ok(p) if made => assert!(!p.graphics.sample_shading),
            Err(Error::NeedsInput(RuntimeInput::Multisampled)) if !made => {}
            other => panic!("multisampled {made}: {:?}", other.map(|_| ())),
        }
    }
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
/// make the memory writers refuses them for what it lacks, never for the loop, except a driver
/// that runs a memory writer once per covered sample (MoltenVK on Apple's GPU), which refuses them,
/// and the varying, as its defect, and every other function by its reason.
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
        let defect = per_sample_defect(&dev);
        if let Some(defect) = &defect {
            for (f, made) in COV_FITS.iter().zip(&fits) {
                assert_eq!(made.as_ref().map(|_| ()), Err(defect), "{f} on {stem}");
            }
        } else if fits.iter().any(Result::is_err) {
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
            assert!(!p.graphics.sample_shading, "{what}");
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
        fitted += usize::from(defect.is_none());
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
                assert!(!p.graphics.sample_shading, "{what}");
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
                e == CoverageLoopRefusal::InterpolationFunctions.error()
                    || Some(&e) == defect.as_ref(),
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

/// `coverage_natural` at 2 and 4 samples on every committed profile. On a driver that runs a memory
/// writer once per covered sample (MoltenVK on Apple's GPU) each function runs as written: no loop,
/// no sample shading, its coverage read, and no `FragCoord` where the module declares
/// `SampleRateShading` (SPIRV-Cross would then mask the coverage with the run's sample bit).
/// Elsewhere each loops without sample shading, or is refused for what the device lacks (the
/// interpolation functions among them), never for the loop.
#[test]
fn a_per_sample_driver_runs_the_natural_functions_as_written() {
    let (Some(natural_lib), Some(vertex_lib)) =
        (library("coverage_natural"), library("coverage_loop"))
    else {
        return;
    };
    let (mut natural, mut looped) = (0, 0);
    for stem in stems() {
        let dev = device(&stem);
        let driver_runs = per_sample_defect(&dev).is_some();
        for (f, vertex) in NATURAL {
            for samples in [2, 4] {
                let what = format!("{f} at {samples} on {stem}");
                let mut d = RenderPipelineDescriptor::new(function(&vertex_lib, vertex));
                d.fragment = Some(function(&natural_lib, f));
                d.raster_sample_count = samples;
                match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(p) => {
                        assert!(!p.graphics.sample_shading, "{what}");
                        let s = shape(fragment_words(&p));
                        assert!(s.sample_mask, "{what}: {s:?}");
                        assert_eq!(s.loops, !driver_runs, "{what}: {s:?}");
                        if driver_runs {
                            natural += 1;
                            assert!(!(s.sample_rate_shading && s.frag_coord), "{what}: {s:?}");
                        } else {
                            looped += 1;
                        }
                        l3(&dev, &p, &what);
                    }
                    Err(e) if e == CoverageLoopRefusal::InterpolationFunctions.error() => {
                        assert!(!driver_runs, "{what}");
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
    assert_eq!(
        natural,
        NATURAL.len() * 2,
        "the MoltenVK profile ran each as written"
    );
    assert!(looped >= 6 * NATURAL.len(), "{looped} pipelines looped");
}

/// Where the driver's interpolation at a sample needs the whole quad (RADV,
/// `DriverWorkaround::InterpolationNeedsTheWholeQuad`), a looping function's sample-qualified
/// varying is read before the loop, at every sample a pipeline of the device may have, each by a
/// constant (`coverage_loop`'s `cov_loop_varying`, and `coverage_natural`'s `nat_varying`, at 2
/// and 4 samples); everywhere else that loops it is read once, at the pass's sample, which the
/// loop computes. Each varying is perspective (`sample_perspective`), so on that driver, which
/// answers a perspective read at a sample by extrapolating from the pixel center
/// (`DriverWorkaround::PerspectiveReadsExtrapolated`), the module records
/// `PrecisionNote::ExtrapolatedInterpolation`, and nowhere else.
#[test]
fn where_interpolation_needs_the_whole_quad_the_varyings_are_read_before_the_loop() {
    let (Some(loop_lib), Some(natural_lib)) =
        (library("coverage_loop"), library("coverage_natural"))
    else {
        return;
    };
    let defect = DriverWorkaround::InterpolationNeedsTheWholeQuad;
    let (mut before, mut inside) = (0, 0);
    for stem in stems() {
        let dev = device(&stem);
        let whole_quad = dev.profile().fields().workarounds.contains(&defect);
        let extrapolates = (dev.profile().fields().workarounds)
            .contains(&DriverWorkaround::PerspectiveReadsExtrapolated);
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
                let reads = interpolated_samples(words);
                let extrapolation_noted = p.stages.iter().any(|m| {
                    m.precision_notes
                        .contains(&PrecisionNote::ExtrapolatedInterpolation)
                });
                assert_eq!(
                    extrapolation_noted, extrapolates,
                    "{what}: the extrapolation note"
                );
                if whole_quad {
                    before += 1;
                    let every: Vec<_> = (0..most_samples(&dev)).map(Some).collect();
                    assert_eq!(reads, every, "{what}");
                } else {
                    inside += 1;
                    assert_eq!(reads, [None], "{what}");
                }
                l3(&dev, &p, &what);
            }
        }
    }
    assert!(
        before >= 4,
        "{before} pipelines read the varying before the loop"
    );
    assert!(
        inside >= 6 * 4,
        "{inside} pipelines read it inside the loop"
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
/// mask, which Metal writes per covered sample, each from that sample's run (our probe `co`). At
/// two and four samples on every committed profile each loops, without sample shading, recording
/// `PrecisionNote::LoopedSampleOutputs`, or is refused for what the device lacks, never as the
/// coverage at sample rate; the masked one declares the `SampleMask` output beside the input. On
/// a driver that runs a memory writer once per covered sample (MoltenVK on Apple's GPU) each runs
/// as written, which is Metal's, without the loop or the note.
#[test]
fn coverage_reading_color_writers_loop_or_run_as_written() {
    let Some(lib) = library("coverage_outputs") else {
        return;
    };
    let (mut looped, mut natural) = (0, 0);
    for stem in stems() {
        let dev = device(&stem);
        let driver_runs = per_sample_defect(&dev).is_some();
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
                        assert!(!p.graphics.sample_shading, "{what}");
                        let words = fragment_words(&p);
                        let s = shape(words);
                        assert_eq!(
                            (s.loops, s.find_lsb, s.sample_id),
                            (!driver_runs, !driver_runs, false),
                            "{what}"
                        );
                        assert_eq!(sample_mask_decorations(words), masks, "{what}");
                        let noted = p.stages.iter().any(|m| {
                            m.precision_notes
                                .contains(&PrecisionNote::LoopedSampleOutputs)
                        });
                        assert_eq!(noted, !driver_runs, "{what}");
                        for m in &p.stages {
                            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                                .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
                        }
                        *if driver_runs {
                            &mut natural
                        } else {
                            &mut looped
                        } += 1;
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
    assert!(
        looped >= 12 && natural >= 4,
        "{looped} looped, {natural} as written"
    );
}
