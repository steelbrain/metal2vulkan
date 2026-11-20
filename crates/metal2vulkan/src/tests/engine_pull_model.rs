//! Metal's pull-model interpolation over the owned fixture `engine/pull_model_interpolants`, as
//! render pipelines of `pm_place` and each fragment function at one and four samples on every
//! committed profile:
//!
//! - An `interpolant` read at the center alone compiles everywhere, as a center varying is read:
//!   at four samples through its pixel-center form (`InterpolateAtOffset` at offset 0) where the
//!   device has one.
//! - A read at the centroid, a sample or an offset is `GLSL.std.450`'s `InterpolateAtCentroid`,
//!   `InterpolateAtSample` or `InterpolateAtOffset`, which need `sampleRateShading` (and on a
//!   portability-subset device `shaderSampleRateInterpolationFunctions`): a device without them
//!   refuses the function, `Unsupported(PullModelInterpolation)` naming what it lacks.
//! - A sample's index is masked to the pipeline's sample count (`SpecKey::SampleCount`, which the
//!   plan sets), as Metal reads sample `i mod n`.
//! - An offset is moved to the device's origin, a half, and clamped (`NClamp`) to its interpolation
//!   offset range, requiring `InterpolateAtOffsetCentered`. The function records
//!   `PrecisionNote::DeviceInterpolationOffset`.
//! - A function that reads a perspective interpolant at a sample or an offset records
//!   `PrecisionNote::ExtrapolatedInterpolation` on every profile, and no other does.
//! - A centroid read of a multisampled pixel compiles on every profile whose reads it needs.
//! - An offset read past the first block (`pm_offset_after_discard`, behind a branch to a discard)
//!   is not lowered on any profile, `NotImplemented(DivergentInterpolation)`.
//! - An interpolant chosen at run time (a `select` of two) is not lowered,
//!   `NotImplemented(PullModelInterpolation)`.
//!
//! Every pipeline that compiles passes L3 (`spirv-val` and requirements within the profile) for each
//! stage module. `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use super::engine_multisample::{
    description, ext_insts, fragment_words, library, pixel_form, planned, stems, validate,
};
use crate::bind::BindState;
use crate::budget::Budget;
use crate::device::requirements::{VkExtension, VkFeature, VkProperty};
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::sym::SpecKey;
use crate::pipeline::RenderPipeline;
use crate::vk::ShaderStage;

const VERTEX: &str = "pm_place";

/// The fragment function that reads its no-perspective interpolant alone, at a sample and an
/// offset.
const LINEAR: &str = "pm_linear";

/// `GLSL.std.450` instruction numbers.
const INTERPOLATE_AT_CENTROID: u32 = 76;
const INTERPOLATE_AT_SAMPLE: u32 = 77;
const INTERPOLATE_AT_OFFSET: u32 = 78;
const N_CLAMP: u32 = 81;

/// The bits of 7/16, where an interpolation offset range may end.
const SEVEN_SIXTEENTHS: u32 = 0x3EE0_0000;

/// Where a fragment function reads its interpolants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    Center,
    Centroid,
    Sample,
    Offset,
}

/// Each fragment function of the fixture that compiles somewhere, its color attachments' format
/// and count, and where it reads.
const FRAGMENTS: [(&str, PixelFormat, usize, &[At]); 9] = [
    (
        LINEAR,
        PixelFormat::RGBA32Float,
        2,
        &[At::Sample, At::Offset],
    ),
    ("pm_center", PixelFormat::RGBA32Float, 2, &[At::Center]),
    ("pm_centroid", PixelFormat::RGBA32Float, 2, &[At::Centroid]),
    ("pm_sample", PixelFormat::RGBA32Float, 2, &[At::Sample]),
    ("pm_sample_id", PixelFormat::RGBA32Float, 2, &[At::Sample]),
    ("pm_offset", PixelFormat::RGBA32Float, 2, &[At::Offset]),
    (
        "pm_offset_sweep",
        PixelFormat::RGBA32Float,
        2,
        &[At::Offset],
    ),
    (
        "pm_narrow",
        PixelFormat::RGBA32Float,
        4,
        &[At::Center, At::Centroid, At::Sample, At::Offset],
    ),
    (
        "pm_half",
        PixelFormat::RGBA16Float,
        4,
        &[At::Center, At::Centroid, At::Sample, At::Offset],
    ),
];

/// The pipeline of `pm_place` and `fragment` into `count` attachments of `format` at `samples`.
fn render(
    lib: &crate::library::Library,
    fragment: &str,
    (format, count): (PixelFormat, usize),
    samples: u32,
) -> RenderPipelineDescriptor {
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(VERTEX));
    d.fragment = Some(stage(fragment));
    for slot in d.color_attachments.iter_mut().take(count) {
        *slot = Some(ColorAttachmentDescriptor::new(format));
    }
    d.raster_sample_count = samples;
    d
}

/// Whether `dev` has the interpolation functions: `sampleRateShading`, and on a
/// portability-subset device `shaderSampleRateInterpolationFunctions`.
fn interpolation_functions(dev: &Device) -> bool {
    let fields = dev.profile().fields();
    let withheld = fields
        .enabled
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset)
        && !fields
            .enabled
            .features
            .contains(&VkFeature::ShaderSampleRateInterpolationFunctions);
    fields.fragment.sample_rate_shading && !withheld
}

/// Whether `words` hold an `OpConstant` whose value is `bits`.
fn holds_constant(words: &[u32], bits: u32) -> bool {
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let (len, op) = ((w >> 16) as usize, w & 0xFFFF);
        if op == 43 && len == 4 && words.get(i + 3) == Some(&bits) {
            return true;
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    false
}

/// Whether the fragment module of `p` records `note`.
fn noted(p: &RenderPipeline, note: PrecisionNote) -> bool {
    p.stages
        .iter()
        .filter(|m| m.stage == ShaderStage::FRAGMENT)
        .any(|m| m.precision_notes.contains(&note))
}

/// Every fragment function at one and four samples on every committed profile compiles where the
/// device has what its reads need, with the instructions, mask, origin, clamp, note and
/// requirements the module docs state, and passes L3; elsewhere it is refused for its reason.
#[test]
fn interpolants_read_where_their_functions_say_or_are_refused() {
    let Some(lib) = library("pull_model_interpolants") else {
        return;
    };
    let mut compiled = 0usize;
    let mut refused = Vec::new();
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        let functions = interpolation_functions(&dev);
        let fields = dev.profile().fields();
        let origin = VkProperty::InterpolateAtOffsetCentered;
        assert!(fields.enabled.properties.contains(&origin), "{stem}");
        let ends_at_seven = fields.limits.max_interpolation_offset_bits == SEVEN_SIXTEENTHS;
        for (fragment, format, count, reads) in FRAGMENTS {
            let reads_at = |at| reads.contains(&at);
            let beyond_center = reads.iter().any(|&at| at != At::Center);
            for samples in [1, 4] {
                let what = format!("{stem} {fragment} at {samples}");
                let d = render(&lib, fragment, (format, count), samples);
                let result = dev.compile_render_pipeline(&d, &Budget::standard());
                if beyond_center && !functions {
                    let e = result.expect_err(&what);
                    assert!(
                        matches!(&e, Error::Unsupported(u)
                            if u.feature == Feature::PullModelInterpolation),
                        "{what}: {e:?}"
                    );
                    refused.push(what);
                    continue;
                }
                let p = match result {
                    // A sample count the profile's attachments do not take.
                    Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                        assert!(samples > 1, "{what}: {u:?}");
                        continue;
                    }
                    p => p.unwrap_or_else(|e| panic!("{what}: {e}")),
                };
                validate(&p, &dev, &what);
                let words = fragment_words(&p);
                let ext = ext_insts(words);
                let pixel_center = reads_at(At::Center) && samples > 1 && pixel_form(&dev);
                assert_eq!(
                    ext.contains(&INTERPOLATE_AT_CENTROID),
                    reads_at(At::Centroid),
                    "{what}: InterpolateAtCentroid"
                );
                assert_eq!(
                    ext.contains(&INTERPOLATE_AT_SAMPLE),
                    reads_at(At::Sample),
                    "{what}: InterpolateAtSample"
                );
                assert_eq!(
                    ext.contains(&INTERPOLATE_AT_OFFSET),
                    reads_at(At::Offset) || pixel_center,
                    "{what}: InterpolateAtOffset"
                );
                assert_eq!(
                    ext.contains(&N_CLAMP),
                    reads_at(At::Offset),
                    "{what}: the offset's clamp"
                );
                assert_eq!(
                    noted(&p, PrecisionNote::DeviceInterpolationOffset),
                    reads_at(At::Offset),
                    "{what}: the offset note"
                );
                // Every function but `pm_linear` reads its perspective interpolant where it reads.
                let perspective = fragment != LINEAR;
                assert_eq!(
                    noted(&p, PrecisionNote::ExtrapolatedInterpolation),
                    perspective && (reads_at(At::Sample) || reads_at(At::Offset)),
                    "{what}: the extrapolation note"
                );
                assert_eq!(
                    p.requirements.properties.contains(&origin),
                    reads_at(At::Offset) || pixel_center,
                    "{what}: {origin:?}"
                );
                // 7/16 is no device's origin; a range may end there, and then the clamp holds it.
                let seven = holds_constant(words, SEVEN_SIXTEENTHS);
                assert!(!seven || ends_at_seven, "{what}: the 7/16 origin");
                if reads_at(At::Sample) && fragment != "pm_sample_id" {
                    let plan = p.plan(&dev, &BindState::default().shape()).expect("plan");
                    assert_eq!(
                        planned(&p, &plan, SpecKey::SampleCount),
                        samples,
                        "{what}: the sample mask's count"
                    );
                }
                compiled += 1;
            }
        }
    }
    eprintln!(
        "{compiled} pipelines compiled, {} refused: {refused:?}",
        refused.len()
    );
    assert!(compiled > 100, "{compiled}");
    assert!(
        refused.iter().any(|r| r.starts_with("vk12-floor ")),
        "a profile without sampleRateShading refuses: {refused:?}"
    );
}

/// An interpolant chosen at run time, a `select` of two interpolant handles (`pm_selected`), is
/// not lowered on any profile: `NotImplemented(PullModelInterpolation)`.
#[test]
fn an_interpolant_chosen_at_run_time_is_not_lowered() {
    let Some(lib) = library("pull_model_interpolants") else {
        return;
    };
    let mut checked = 0;
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        for samples in [1, 4] {
            let what = format!("{stem} pm_selected at {samples}");
            let d = render(&lib, "pm_selected", (PixelFormat::RGBA32Float, 1), samples);
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                    assert!(samples > 1, "{what}: {u:?}");
                }
                result => {
                    let e = result.expect_err(&what);
                    assert!(
                        matches!(&e, Error::Unsupported(u)
                            if u.feature == Feature::PullModelInterpolation
                                && u.needs == Needs::NotImplemented),
                        "{what}: {e:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 20, "{checked}");
}

/// An offset read past the function's first block (`pm_offset_after_discard`, whose reads follow a
/// branch to a discard), where some of the quad's invocations may have left, is not lowered on
/// any profile, before what the reads need is asked: `NotImplemented(DivergentInterpolation)`.
#[test]
fn an_offset_read_where_the_quad_may_have_left_is_not_lowered() {
    let Some(lib) = library("pull_model_interpolants") else {
        return;
    };
    let mut checked = 0;
    for stem in stems() {
        let dev =
            Device::from_queries(description(&stem), DevicePolicy::default()).expect("device");
        for samples in [1, 4] {
            let what = format!("{stem} pm_offset_after_discard at {samples}");
            let d = render(
                &lib,
                "pm_offset_after_discard",
                (PixelFormat::RGBA32Float, 2),
                samples,
            );
            match dev.compile_render_pipeline(&d, &Budget::standard()) {
                Err(Error::Unsupported(u)) if u.feature == Feature::RasterSampleCount => {
                    assert!(samples > 1, "{what}: {u:?}");
                }
                result => {
                    let e = result.expect_err(&what);
                    assert!(
                        matches!(&e, Error::Unsupported(u)
                            if u.feature == Feature::DivergentInterpolation
                                && u.needs == Needs::NotImplemented),
                        "{what}: {e:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 20, "{checked}");
}
