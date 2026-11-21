//! Multisampled fragment invocations over the owned fixture `engine/multisample`, as render
//! pipelines of `multisample_vertex` and each fragment function at one and four samples on every
//! committed profile:
//!
//! - At four samples a fragment function compiles under `LoweringInputs::multisampled`: it reads
//!   `[[position]]`'s x and y as `floor(FragCoord.xy) + 0.5` and, where the device has
//!   interpolation functions, each center varying through `InterpolateAtOffset` at offset 0, the
//!   pixel center Metal reads in every invocation, at sample rate or not. The pipeline then
//!   requires `VkProperty::InterpolateAtOffsetCentered`, which every device holds, and on a
//!   portability-subset device `shaderSampleRateInterpolationFunctions`. A center varying read at
//!   sample rate on a device without those interpolation functions is refused naming them.
//! - A function that writes memory runs as the device runs it, at pixel rate where nothing else
//!   makes it per-sample, and its fragment stage notes `PrecisionNote::DeviceFragmentRuns` at four
//!   samples: Apple's GPUs were measured to run it once per covered sample, and documented Metal
//!   names no count.
//! - A per-sample function (one that reads `[[sample_id]]` or a sample-qualified varying) reads
//!   what has no pixel form under sample shading (`[[position]]`'s z and w, a
//!   centroid varying) where Vulkan evaluates it for its sample, and notes
//!   `PrecisionNote::MultisamplePosition`, as Metal accepts it.
//! - At one sample every function compiles as without multisampling.
//! - A pixel-rate function of a multisampled pipeline that reads a value Vulkan evaluates at an
//!   implementation-defined position (`[[position]]`'s z, a centroid varying) notes
//!   `PrecisionNote::MultisamplePosition`.
//! - `get_num_samples()` reads the `SampleCount` specialization constant, which the pipeline's
//!   plan sets to its sample count (the owned fixture `engine/spc_num_samples`).
//!
//! Every pipeline that compiles passes L3 (`spirv-val` and requirements within the profile) for
//! each stage module; on `moltenvk-m3` the multisampled fragment modules also convert to MSL and
//! compile as MoltenVK's do
//! (`spirv-cross` and `xcrun metal`, compile only). `llvm-as` assembles the fixtures; a missing
//! tool skips visibly. No GPU is used.

use std::path::Path;

use crate::bind::BindState;
use crate::budget::Budget;
use crate::device::requirements::{VkExtension, VkFeature, VkProperty};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, LowerFamily, Needs, Unsupported};
use crate::library::Library;
use crate::metal::{PixelFormat, VertexFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::sym::SpecKey;
use crate::pipeline::testing::{
    compile, engine_dir, llvm_assemble, moltenvk_msl_compile, AssembleError, Scratch,
};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;

const VERTEX: &str = "multisample_vertex";
const CENTER: &str = "multisample_center_at_sample";
const DEPTH: &str = "multisample_depth_at_sample";
const SAMPLED: &str = "multisample_sampled_varying";
const CENTROID: &str = "multisample_centroid_at_sample";
const COUNTER: &str = "multisample_counter";
const FETCH_VERTEX: &str = "multisample_fetch_vertex";
const FETCH_DEPTH: &str = "multisample_fetch_depth";

/// `GLSL.std.450` instruction numbers.
const FLOOR: u32 = 8;
const INTERPOLATE_AT_OFFSET: u32 = 78;

fn fixture() -> Option<Library> {
    library("multisample")
}

pub(super) fn library(name: &str) -> Option<Library> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(Library::from_air(&b).expect("library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

pub(super) fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

/// Every committed profile's stem, sorted.
pub(super) fn stems() -> Vec<String> {
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

/// The render pipeline of `multisample_vertex` and `fragment` over one `RGBA16Float`
/// attachment at `samples`.
fn render(lib: &Library, fragment: &str, samples: u32) -> RenderPipelineDescriptor {
    render_after(lib, VERTEX, fragment, samples)
}

/// The render pipeline of `vertex` and `fragment` over one `RGBA16Float` attachment at `samples`.
fn render_after(
    lib: &Library,
    vertex: &str,
    fragment: &str,
    samples: u32,
) -> RenderPipelineDescriptor {
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(vertex));
    d.fragment = Some(stage(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    d.raster_sample_count = samples;
    d
}

/// The `stage_io` pair's render pipeline at `samples` (as `src/tests/golden_hashes.rs`, which leaves
/// out the integer attachment at more than one sample): a pixel-rate fragment function that reads
/// `[[position]]`, center, centroid, and flat varyings, front facing, and the input sample mask.
fn stage_io_render(lib: &Library, samples: u32) -> RenderPipelineDescriptor {
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut layout = VertexDescriptor::new();
    layout
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
    for (index, format, offset) in [
        (0, VertexFormat::Float3, 0),
        (1, VertexFormat::Half2, 12),
        (4, VertexFormat::UShort2, 16),
        (7, VertexFormat::UChar4, 20),
    ] {
        layout
            .attributes
            .insert(index, VertexAttributeDescriptor::new(format, offset, 0));
    }
    let mut d = RenderPipelineDescriptor::new(stage("stage_io_vertex"));
    d.fragment = Some(stage("stage_io_fragment"));
    d.vertex_descriptor = Some(layout);
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    if samples == 1 {
        d.color_attachments[2] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    }
    d.depth_format = Some(PixelFormat::Depth32Float);
    d.raster_sample_count = samples;
    d
}

/// The `GLSL.std.450` instruction numbers of every `OpExtInst` in `words`.
pub(super) fn ext_insts(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        if op == 12 {
            out.extend(words.get(i + 4).copied());
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    out
}

/// The fragment module of `p`.
pub(super) fn fragment_words(p: &RenderPipeline) -> &[u32] {
    &p.stages
        .iter()
        .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
        .expect("fragment stage")
        .spirv
}

/// L3 for every stage of `p` on `dev`.
pub(super) fn validate(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// Whether the fragment module of `p` notes `MultisamplePosition`.
fn noted(p: &RenderPipeline) -> bool {
    p.stages
        .iter()
        .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
        .expect("fragment stage")
        .precision_notes
        .contains(&PrecisionNote::MultisamplePosition)
}

/// Whether the fragment module of `p` notes `DeviceFragmentRuns`.
fn noted_runs(p: &RenderPipeline) -> bool {
    p.stages
        .iter()
        .find(|m| m.stage == crate::vk::ShaderStage::FRAGMENT)
        .expect("fragment stage")
        .precision_notes
        .contains(&PrecisionNote::DeviceFragmentRuns)
}

/// The refusal of a per-sample function's center varying on a device that lacks `needs`.
fn needs_pixel_form(needs: Vec<Capability>) -> Error {
    Unsupported::needs_any_of(Feature::SampleRateShading, vec![needs]).into()
}

/// Whether `dev` has the pixel form of a center varying: interpolation functions whose
/// `InterpolateAtOffset` reads from the center.
pub(super) fn pixel_form(dev: &Device) -> bool {
    let fields = dev.profile().fields();
    let withheld = fields
        .enabled
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset)
        && !fields
            .enabled
            .features
            .contains(&VkFeature::ShaderSampleRateInterpolationFunctions);
    fields.fragment.sample_rate_shading
        && !withheld
        && fields
            .enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
}

/// Every function at one and four samples on every committed profile: what compiles passes L3,
/// reads the pixel center where Metal does, and what is refused is refused for its reason. A
/// per-sample center varying compiles on every profile with `sampleRateShading`, through
/// `InterpolateAtOffset` at offset 0, which every device reads from the center. A per-sample
/// `[[position]]` z
/// and w, or centroid varying, compiles on every profile with `sampleRateShading` too, noting
/// `MultisamplePosition`. The function that writes memory compiles at pixel rate on every profile
/// with fragment stores, its center varying read at the pixel where the device has the form, and
/// notes `DeviceFragmentRuns` at four samples only.
#[test]
fn multisampled_fragments_read_the_pixel_center_or_are_refused() {
    let Some(lib) = fixture() else { return };
    let mut checked = 0usize;
    let mut unformed = 0usize;
    let mut writers = 0usize;
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        let fields = dev.profile().fields();
        let rate = fields.fragment.sample_rate_shading;
        let stores = fields.leaf.fragment_stores_and_atomics;
        let portability = fields
            .enabled
            .extensions
            .contains(&VkExtension::KhrPortabilitySubset);
        assert!(
            fields
                .enabled
                .properties
                .contains(&VkProperty::InterpolateAtOffsetCentered),
            "{stem}"
        );
        for fragment in [CENTER, DEPTH, SAMPLED, CENTROID, COUNTER] {
            let sample_rate = fragment != COUNTER;
            let what = format!("{stem} {fragment}");
            // Without the device's sample-rate shading or fragment stores the function never
            // compiles, at any sample count.
            if (sample_rate && !rate) || (fragment == COUNTER && !stores) {
                for samples in [1, 4] {
                    let e = dev
                        .compile_render_pipeline(
                            &render(&lib, fragment, samples),
                            &Budget::standard(),
                        )
                        .expect_err(&what);
                    assert!(
                        matches!(
                            &e,
                            Error::Unsupported(u)
                                if matches!(u.feature, Feature::SampleRateShading | Feature::StageStoresAndAtomics)
                        ),
                        "{what} at {samples}: {e:?}"
                    );
                }
                continue;
            }
            let one = dev
                .compile_render_pipeline(&render(&lib, fragment, 1), &Budget::standard())
                .unwrap_or_else(|e| panic!("{what} at 1: {e}"));
            validate(&one, &dev, &what);
            assert!(
                !ext_insts(fragment_words(&one)).contains(&INTERPOLATE_AT_OFFSET),
                "{what} at 1"
            );
            assert!(!noted_runs(&one), "{what} at 1");
            let four = dev.compile_render_pipeline(&render(&lib, fragment, 4), &Budget::standard());
            match (fragment, four) {
                (_, Err(Error::Unsupported(u))) if u.feature == Feature::RasterSampleCount => {
                    eprintln!("{what}: four samples are not a framebuffer sample count");
                }
                (CENTER | SAMPLED, Ok(p)) => {
                    validate(&p, &dev, &what);
                    // `InterpolateAtOffset` at offset 0 is the pixel form.
                    let ext = ext_insts(fragment_words(&p));
                    assert!(ext.contains(&INTERPOLATE_AT_OFFSET), "{what}: {ext:?}");
                    assert_eq!(ext.contains(&FLOOR), fragment != SAMPLED, "{what}: {ext:?}");
                    assert_eq!(
                        p.requirements
                            .features
                            .contains(&VkFeature::ShaderSampleRateInterpolationFunctions),
                        portability,
                        "{what}"
                    );
                    assert!(
                        p.requirements
                            .properties
                            .contains(&VkProperty::InterpolateAtOffsetCentered),
                        "{what}"
                    );
                    // Per sample through its input, which needs `sampleRateShading`.
                    assert!(!noted_runs(&p), "{what}");
                    assert!(
                        p.requirements
                            .features
                            .contains(&VkFeature::SampleRateShading),
                        "{what}"
                    );
                    checked += 1;
                }
                // Metal accepts a per-sample function beside a read without a pixel form: it
                // reads where Vulkan evaluates it for the sample, which the module notes.
                (DEPTH | CENTROID, Ok(p)) => {
                    validate(&p, &dev, &what);
                    assert!(noted(&p), "{what}");
                    let ext = ext_insts(fragment_words(&p));
                    assert!(!ext.contains(&INTERPOLATE_AT_OFFSET), "{what}: {ext:?}");
                    // `[[sample_id]]` turns sample shading on itself.
                    assert!(!noted_runs(&p), "{what}");
                    assert!(
                        p.requirements
                            .features
                            .contains(&VkFeature::SampleRateShading),
                        "{what}"
                    );
                    unformed += 1;
                }
                // The function that writes memory runs as the device runs it, at pixel rate, and
                // reads its position and center varying as a pixel-rate function does.
                (COUNTER, Ok(p)) => {
                    validate(&p, &dev, &what);
                    let ext = ext_insts(fragment_words(&p));
                    assert!(ext.contains(&FLOOR), "{what}: {ext:?}");
                    assert_eq!(
                        ext.contains(&INTERPOLATE_AT_OFFSET),
                        pixel_form(&dev),
                        "{what}: {ext:?}"
                    );
                    assert!(noted_runs(&p), "{what}");
                    writers += 1;
                }
                (_, other) => panic!("{what} at 4: {other:?}"),
            }
        }
    }
    assert!(
        checked > 0,
        "no profile compiled a multisampled pixel-center read"
    );
    assert!(
        unformed > 0,
        "no profile compiled a per-sample read without a pixel form"
    );
    assert!(
        writers > 0,
        "no profile compiled the memory writer at four samples"
    );
}

/// A pixel-rate function of a multisampled pipeline reads its center varyings at the pixel center
/// too, where the device has the pixel form: Vulkan evaluates an undecorated input at an
/// implementation-defined position in the pixel, Metal at its center. `stage_io_fragment` at four
/// samples reads them through `InterpolateAtOffset` on every committed profile with interpolation
/// functions, requiring the centered offset's property; at one sample nowhere. (`roadmap-2022` and
/// `vk12-floor` lack the pipeline's depth format.)
#[test]
fn a_pixel_rate_center_varying_reads_the_pixel_center_where_the_device_has_the_form() {
    let Some(lib) = library("stage_io") else {
        return;
    };
    let mut with_form = 0usize;
    let mut without_form = Vec::new();
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        for samples in [1, 4] {
            let what = format!("{stem} at {samples}");
            let p = match dev
                .compile_render_pipeline(&stage_io_render(&lib, samples), &Budget::standard())
            {
                // The profile lacks an attachment format, or four samples of one.
                Err(Error::Unsupported(u))
                    if samples == 1 || u.feature == Feature::RasterSampleCount =>
                {
                    eprintln!("{what}: {u}");
                    break;
                }
                other => other.unwrap_or_else(|e| panic!("{what}: {e}")),
            };
            validate(&p, &dev, &what);
            let form = samples == 4 && pixel_form(&dev);
            assert_eq!(
                ext_insts(fragment_words(&p)).contains(&INTERPOLATE_AT_OFFSET),
                form,
                "{what}"
            );
            assert_eq!(
                p.requirements
                    .properties
                    .contains(&VkProperty::InterpolateAtOffsetCentered),
                form,
                "{what}"
            );
            // A pixel-rate function that writes no memory runs as Metal's does.
            assert!(!noted_runs(&p), "{what}");
            with_form += usize::from(form);
            if samples == 4 && !form {
                without_form.push(stem.clone());
            }
        }
    }
    assert!(
        with_form > 0,
        "no profile read a pixel-rate center varying at the pixel"
    );
    assert_eq!(without_form, Vec::<String>::new());
}

/// A pixel-rate fragment function of a multisampled pipeline that reads a value Vulkan evaluates
/// at an implementation-defined position in the fragment notes `MultisamplePosition`:
/// `stage_io_fragment` reads `[[position]]`'s z and a centroid varying, beside a center varying it
/// reads at the pixel through `InterpolateAtOffset` (`moltenvk-m3` and `amd-780m-radv`). The same
/// pipeline at one sample, and the multisampled per-sample pipelines that read only through the
/// pixel forms, note nothing.
#[test]
fn a_pixel_rate_read_at_an_unstated_position_is_noted() {
    let Some(lib) = fixture() else { return };
    let Some(stage_io) = library("stage_io") else {
        return;
    };
    for stem in ["moltenvk-m3", "amd-780m-radv"] {
        let dev = Device::from_queries(description(stem), DevicePolicy::default()).expect("device");
        for samples in [1, 4] {
            let p = dev
                .compile_render_pipeline(&stage_io_render(&stage_io, samples), &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} at {samples}: {e}"));
            validate(&p, &dev, stem);
            assert_eq!(noted(&p), samples == 4, "{stem} at {samples}");
            assert_eq!(
                ext_insts(fragment_words(&p)).contains(&INTERPOLATE_AT_OFFSET),
                samples == 4,
                "{stem} at {samples}"
            );
        }
        for fragment in [CENTER, SAMPLED, COUNTER] {
            let p = dev
                .compile_render_pipeline(&render(&lib, fragment, 4), &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem} {fragment}: {e}"));
            assert!(!noted(&p), "{stem} {fragment}");
        }
    }
}

/// Programmable blending beside `[[position]]`'s w (`multisample_fetch_depth` after
/// `multisample_fetch_vertex`, the owned fixture `engine/multisample_fetch`), the shape of a
/// browser compositor's blend: Metal runs a function that reads a color attachment once per
/// covered sample and reads `[[position]]` as the pixel's in each run. At four samples it compiles
/// on every committed profile with a form of the fetch at sample rate, its per-sample fetch
/// reading `[[sample_id]]` beside `[[position]]`'s w, which reads where Vulkan evaluates it and
/// notes `MultisamplePosition`; a profile without such a form refuses it for the fetch, never for
/// the read.
#[test]
fn programmable_blending_beside_position_w_compiles_noted() {
    let Some(lib) = library("multisample_fetch") else {
        return;
    };
    let mut compiled = Vec::new();
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        let what = format!("{stem} {FETCH_DEPTH}");
        let d = render_after(&lib, FETCH_VERTEX, FETCH_DEPTH, 4);
        match dev.compile_render_pipeline(&d, &Budget::standard()) {
            Ok(p) => {
                validate(&p, &dev, &what);
                assert!(noted(&p), "{what}");
                compiled.push(stem);
            }
            Err(Error::Unsupported(u)) => assert!(
                matches!(
                    u.feature,
                    Feature::FramebufferFetch | Feature::RasterSampleCount
                ),
                "{what}: {u:?}"
            ),
            Err(e) => panic!("{what}: {e:?}"),
        }
    }
    for stem in [
        "amd-780m-radv",
        "intel-uhd770-rpls-anv",
        "nvidia-rtx5080-laptop",
    ] {
        assert!(compiled.iter().any(|s| s == stem), "{stem}: {compiled:?}");
    }
}

/// The interface states what link needs: whether the function was compiled for a multisampled
/// pipeline, reads a center varying at the pixel, and may read `[[position]]`'s z or w (anything
/// but an extract or a shuffle of its x and y lanes, from either shuffle operand), on lavapipe and
/// on `moltenvk-m3`.
#[test]
fn the_interface_states_the_pixel_reads() {
    let Some(lib) = fixture() else { return };
    let mut multisampled = LoweringInputs::new();
    multisampled.multisampled = true;
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = Device::from_queries(description(stem), DevicePolicy::default()).expect("device");
        // (function, reads z or w, reads a center varying)
        for (fragment, depth, center) in [
            (CENTER, false, true),
            (DEPTH, true, false),
            (SAMPLED, false, true),
            (CENTROID, false, false),
            (COUNTER, false, true),
        ] {
            for inputs in [LoweringInputs::new(), multisampled.clone()] {
                let f = FunctionDescriptor::new(lib.function(fragment).expect("function"));
                let c = dev
                    .compile_function(&f, &inputs, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{stem} {fragment}: {e}"));
                let io = &c.artifact().interface.io;
                let what = format!("{stem} {fragment} {}", inputs.multisampled);
                assert_eq!(io.multisampled, inputs.multisampled, "{what}");
                assert_eq!(io.position_depth, depth, "{what}");
                assert_eq!(
                    io.varyings_at_pixel,
                    inputs.multisampled && center,
                    "{what}"
                );
            }
        }
    }
}

/// On `moltenvk-m3`, a portability-subset device, a per-sample function's center varying has a
/// pixel form only where the portability member allows it, and is refused naming the member: with
/// it the pipeline compiles, requires the member and the centered offset, and converts to MSL that
/// Metal compiles; either way, the pipeline of the same function at one sample compiles without
/// either. The memory writer runs at pixel rate: with the member it reads its center varying
/// through the same pixel form, and without it where the device evaluates it, noting
/// `MultisamplePosition`, rather than being refused.
#[test]
fn a_center_varying_at_sample_rate_needs_the_interpolation_functions() {
    let Some(lib) = fixture() else { return };
    let feature = VkFeature::ShaderSampleRateInterpolationFunctions;
    for withheld in [true, false] {
        let mut desc = description("moltenvk-m3");
        if withheld {
            assert!(desc.features.remove(&feature));
        }
        let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
        for fragment in [CENTER, SAMPLED, COUNTER] {
            let what = format!("{fragment} withheld={withheld}");
            let four = dev.compile_render_pipeline(&render(&lib, fragment, 4), &Budget::standard());
            if !withheld {
                let p = four.unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&p, &dev, &what);
                assert!(p.requirements.features.contains(&feature), "{what}");
                assert!(
                    p.requirements
                        .properties
                        .contains(&VkProperty::InterpolateAtOffsetCentered),
                    "{what}"
                );
                // MoltenVK converts the pixel forms to MSL that Metal compiles.
                match moltenvk_msl_compile(fragment_words(&p)) {
                    Ok(()) => {}
                    Err(AssembleError::Missing(why)) => eprintln!("SKIPPED: {why}"),
                    Err(AssembleError::Rejected(why)) => panic!("{what}: MoltenVK's MSL: {why}"),
                }
            } else if fragment == COUNTER {
                let p = four.unwrap_or_else(|e| panic!("{what}: {e}"));
                validate(&p, &dev, &what);
                assert!(!p.requirements.features.contains(&feature), "{what}");
                assert!(noted(&p), "{what}");
            } else {
                assert_eq!(
                    four.map(|_| ()),
                    Err(needs_pixel_form(vec![Capability::Feature(feature)])),
                    "{what}"
                );
            }
            let p = dev
                .compile_render_pipeline(&render(&lib, fragment, 1), &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            validate(&p, &dev, &what);
            assert!(!p.requirements.features.contains(&feature), "{what}");
            assert!(
                !p.requirements
                    .properties
                    .contains(&VkProperty::InterpolateAtOffsetCentered),
                "{what}"
            );
        }
    }
}

/// The value of the planned specialization constant `key` of `p`.
pub(super) fn planned(p: &RenderPipeline, plan: &crate::bind::BindingPlan, key: SpecKey) -> u32 {
    let id = p
        .stages
        .iter()
        .flat_map(|m| m.specialization.constants.iter())
        .find(|c| c.key == key)
        .unwrap_or_else(|| panic!("no {key:?} constant"))
        .id;
    let e = plan
        .specialization
        .entries
        .iter()
        .find(|e| e.constant_id == id)
        .unwrap_or_else(|| panic!("no {key:?} entry"));
    let at = e.offset as usize;
    u32::from_le_bytes(
        plan.specialization.data[at..at + 4]
            .try_into()
            .expect("4 bytes"),
    )
}

/// `get_num_samples()` (`air.get_num_samples.i32` with operand 0, the owned fixture
/// `engine/spc_num_samples`) reads the `SampleCount` specialization constant, default 1,
/// and states `RuntimeFact::SampleCount`; a render pipeline's plan sets it to the pipeline's
/// sample count, at one and at four samples. Another operand has no established meaning and is
/// refused.
#[test]
fn get_num_samples_reads_the_pipeline_sample_count() {
    let Some(ms) = fixture() else { return };
    let path = engine_dir().join("spc_num_samples.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected spc_num_samples.ll: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let mut checked = 0;
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = Device::from_queries(description(stem), DevicePolicy::default()).expect("device");
        let a = compile(&bc, "spc_num_samples", dev.profile()).expect("spc_num_samples");
        assert!(a.facts.contains(&RuntimeFact::SampleCount), "{stem}");
        for samples in [1, 4] {
            let what = format!("{stem} at {samples}");
            let stage = |l: &Library, name: &str| {
                StageSource::Function(FunctionDescriptor::new(l.function(name).expect("function")))
            };
            let mut d = RenderPipelineDescriptor::new(stage(&ms, VERTEX));
            d.fragment = Some(stage(&lib, "spc_num_samples"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            d.raster_sample_count = samples;
            let p = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                    assert!(samples > 1 && stem != "moltenvk-m3", "{what}: {u:?}");
                    eprintln!("{what}: not a framebuffer sample count of an integer attachment");
                    continue;
                }
                p => p.unwrap_or_else(|e| panic!("{what}: {e}")),
            };
            validate(&p, &dev, &what);
            let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
            assert_eq!(planned(&p, &plan, SpecKey::SampleCount), samples, "{what}");
            checked += 1;
        }
    }
    assert_eq!(
        checked, 3,
        "moltenvk-m3 at one and four samples, lavapipe at one"
    );
    // The same call with operand 1.
    let text = std::fs::read_to_string(&path).expect("fixture text");
    let other = text.replace(
        "@air.get_num_samples.i32(i32 0)",
        "@air.get_num_samples.i32(i32 1)",
    );
    assert_ne!(other, text);
    let scratch = Scratch::new("m2v-num-samples");
    let dir = scratch.path();
    let ll = dir.join("num_samples_one.ll");
    std::fs::write(&ll, other).expect("write");
    let bc = llvm_assemble(&ll);
    let bc = bc.expect("assembled");
    let p = Device::from_queries(description("lavapipe"), DevicePolicy::default()).expect("device");
    match compile(&bc, "spc_num_samples", p.profile()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Lowering(LowerFamily::Fragment));
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("operand 1: {:?}", other.map(|_| ())),
    }
}
