//! Metal's pull-model interpolation over the owned fixture `engine/pull_model_interpolants`, as
//! render pipelines of `pm_place` and each fragment function at one and four samples on every
//! committed profile:
//!
//! - An `interpolant` read at the center alone compiles everywhere, as a center varying is read:
//!   at four samples through its pixel-center form (`InterpolateAtOffset` at offset 0) where the
//!   device has one and does not read center varyings at the pixel itself.
//! - A read at the centroid, a sample or an offset is `GLSL.std.450`'s `InterpolateAtCentroid`,
//!   `InterpolateAtSample` or `InterpolateAtOffset`, which need `sampleRateShading` (and on a
//!   portability-subset device `shaderSampleRateInterpolationFunctions`): a device without them
//!   refuses the function, `Unsupported(PullModelInterpolation)` naming what it lacks.
//! - A sample's index is masked to the pipeline's sample count (`SpecKey::SampleCount`, which the
//!   plan sets), as Metal reads sample `i mod n`.
//! - An offset is moved to the device's origin and clamped (`NClamp`) to its interpolation offset
//!   range: by 7/16 on MoltenVK, whose SPIRV-Cross adds 7/16 back, requiring
//!   `InterpolateAtOffsetFromSevenSixteenths`; by a half elsewhere, requiring
//!   `InterpolateAtOffsetCentered`. The function records `PrecisionNote::DeviceInterpolationOffset`.
//! - On RADV, whose driver answers a perspective read at a sample or an offset with the pixel
//!   center's value extrapolated along the quad's derivatives
//!   (`DriverWorkaround::PerspectiveReadsExtrapolated`), a function that reads a perspective
//!   interpolant at a sample or an offset records `PrecisionNote::ExtrapolatedInterpolation`, and
//!   one that reads its no-perspective interpolants alone so (`pm_linear`), or reads only at the
//!   center or the centroid, records nothing; no other profile records it.
//! - On RADV (`DriverWorkaround::InterpolationNeedsTheWholeQuad`) an offset read after a discard
//!   is refused, where some of the quad's invocations may have left; on a driver that shades a
//!   triangle it clips as several (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`:
//!   lavapipe) a centroid read of a multisampled pixel is refused as a centroid varying is.
//! - An interpolant chosen at run time (a `select` of two) is not lowered,
//!   `NotImplemented(PullModelInterpolation)`.
//!
//! Every pipeline that compiles passes L3 (`spirv-val` and requirements within the profile) for each
//! stage module. `llvm-as` assembles the fixture; a missing tool skips visibly. No GPU is used.

use super::engine_multisample::{
    description, ext_insts, fragment_words, library, pixel_form, planned, reads_center_at_pixel,
    stems, validate,
};
use crate::bind::BindState;
use crate::budget::Budget;
use crate::device::requirements::{VkExtension, VkFeature, VkProperty};
use crate::device::{Device, DevicePolicy, DriverWorkaround};
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

/// The bits of 7/16, MoltenVK's offset origin.
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
const FRAGMENTS: [(&str, PixelFormat, usize, &[At]); 10] = [
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
        "pm_offset_after_discard",
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
    let mut extrapolations = 0usize;
    let mut refused = Vec::new();
    for stem in stems() {
        let dev = Device::new(description(&stem), DevicePolicy::default()).expect("device");
        let functions = interpolation_functions(&dev);
        let fields = dev.profile().fields();
        let centered = fields
            .enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered);
        let origin = match centered {
            true => VkProperty::InterpolateAtOffsetCentered,
            false => VkProperty::InterpolateAtOffsetFromSevenSixteenths,
        };
        let whole_quad = fields
            .workarounds
            .contains(&DriverWorkaround::InterpolationNeedsTheWholeQuad);
        let extrapolates = fields
            .workarounds
            .contains(&DriverWorkaround::PerspectiveReadsExtrapolated);
        let ends_at_seven = fields.limits.max_interpolation_offset_bits == SEVEN_SIXTEENTHS;
        let split = fields
            .workarounds
            .contains(&DriverWorkaround::ClippedTrianglesSplitPixelFragments);
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
                if fragment == "pm_offset_after_discard" && whole_quad {
                    let e = result.expect_err(&what);
                    let defect =
                        Needs::DriverDefect(DriverWorkaround::InterpolationNeedsTheWholeQuad);
                    assert!(
                        matches!(&e, Error::Unsupported(u)
                            if u.feature == Feature::PullModelInterpolation && u.needs == defect),
                        "{what}: {e:?}"
                    );
                    refused.push(what);
                    continue;
                }
                // A centroid read of a multisampled pixel, refused as a centroid varying is on a
                // driver that shades a triangle it clips as several.
                if reads_at(At::Centroid) && samples > 1 && split {
                    let e = result.expect_err(&what);
                    let defect =
                        Needs::DriverDefect(DriverWorkaround::ClippedTrianglesSplitPixelFragments);
                    assert!(
                        matches!(&e, Error::Unsupported(u)
                            if u.feature == Feature::CentroidVarying && u.needs == defect),
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
                let pixel_center = reads_at(At::Center)
                    && samples > 1
                    && pixel_form(&dev)
                    && !reads_center_at_pixel(&dev);
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
                let perspective_beyond =
                    fragment != LINEAR && (reads_at(At::Sample) || reads_at(At::Offset));
                let extrapolation_noted = noted(&p, PrecisionNote::ExtrapolatedInterpolation);
                assert_eq!(
                    extrapolation_noted,
                    extrapolates && perspective_beyond,
                    "{what}: the extrapolation note"
                );
                extrapolations += usize::from(extrapolation_noted);
                assert_eq!(
                    p.requirements.properties.contains(&origin),
                    reads_at(At::Offset) || (pixel_center && centered),
                    "{what}: {origin:?}"
                );
                // 7/16 is the origin a MoltenVK offset moves by, and no centered device's (whose
                // range may end at 7/16: then the clamp holds it).
                let seven = holds_constant(words, SEVEN_SIXTEENTHS);
                if centered {
                    assert!(!seven || ends_at_seven, "{what}: the 7/16 origin");
                } else {
                    assert_eq!(seven, reads_at(At::Offset), "{what}: the 7/16 origin");
                }
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
    assert!(
        refused
            .iter()
            .any(|r| r.starts_with("amd-780m-radv pm_offset_after_discard")),
        "RADV refuses an offset read after a discard: {refused:?}"
    );
    // pm_sample, pm_sample_id, pm_offset, pm_offset_sweep, pm_narrow and pm_half at 1 and 4
    // samples on the RADV profile.
    assert!(
        extrapolations >= 12,
        "{extrapolations} pipelines recorded the extrapolation"
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
        let dev = Device::new(description(&stem), DevicePolicy::default()).expect("device");
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
