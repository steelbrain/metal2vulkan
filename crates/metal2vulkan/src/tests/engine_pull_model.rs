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

use super::engine_multisample::{description, library, stems};
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::error::{Error, Feature, Needs};
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};

const VERTEX: &str = "pm_place";

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

/// An interpolant chosen at run time, a `select` of two interpolant handles (`pm_selected`), is
/// not lowered on any profile: `NotImplemented(PullModelInterpolation)`.
#[test]
fn an_interpolant_chosen_at_run_time_is_not_lowered() {
    let Some(lib) = library("pull_model_interpolants") else {
        return;
    };
    let mut checked = 0;
    for stem in stems() {
        let dev = Device::from_queries(description(&stem), DevicePolicy::RESIDENT_TABLES)
            .expect("device");
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
        let dev = Device::from_queries(description(&stem), DevicePolicy::RESIDENT_TABLES)
            .expect("device");
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
