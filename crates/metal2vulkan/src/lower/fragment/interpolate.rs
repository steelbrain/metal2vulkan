//! Pull-model interpolation: an `interpolate_at_center`, `_centroid`, `_sample` or `_offset` call
//! of an `interpolant` fragment input, read from the interpolant's variable, which
//! `lower::stage_io` plans (`stage_io::plan`'s `interpolant`), where the call says
//! (`stage_io::interpolate`).
//!
//! Metal's answers, measured on the Apple M3 Max under macOS 27.0.1 (the conformance probe
//! `pull_model_interpolation.m` over the owned fixture `pull_model_interpolants`):
//!
//! - `interpolate_at_offset` reads at the offset from the pixel's upper-left corner, `(0.5, 0.5)`
//!   its center, as given: no range (an offset of `1e9` extrapolates to `1e9`) and no grid (every
//!   step of a 1/512-pixel sweep is a step; 0.999 reads at 0.999); a NaN or infinite offset reads
//!   NaN. Vulkan's `InterpolateAtOffset` reads from the pixel's center, within the device's
//!   `[minInterpolationOffset, maxInterpolationOffset]` and on its `subPixelInterpolationOffsetBits`
//!   grid: the offset is moved by the device's origin (a half; 7/16 on MoltenVK, whose SPIRV-Cross
//!   adds 7/16 before Metal reads it) and clamped to that range, which Vulkan requires, and the
//!   rest is the device's (`PrecisionNote::DeviceInterpolationOffset`).
//! - `interpolate_at_sample(i)` reads sample `i mod n` of a pipeline of `n` samples, every index
//!   (`stage_io::InterpolantAt::Sample`); the positions are Vulkan's standard ones.
//! - `interpolate_at_centroid` reads the pixel's center where the primitive covers the whole pixel,
//!   else the location of its lowest covered sample. Vulkan's centroid is a location in the pixel
//!   and the primitive the device chooses: in a multisampled pipeline the compile layer notes
//!   `PrecisionNote::MultisamplePosition` for it, as for a centroid varying.
//! - `interpolate_at_center` reads the pixel center as a center-sampled varying does, and
//!   `interpolate_at_sample(0)` at one sample equals it bit for bit, while
//!   `interpolate_at_offset(0.5, 0.5)` differs from it in the last bits: the center read is the
//!   center varying's, never an offset.

use crate::air::intrinsics::{Interpolate, InterpolateAt};
use crate::air::ir::Value;
use crate::device::requirements::VkProperty;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::select::SelectCx;
use crate::lower::stage_io::{InterpolantAt, InterpolantRead, Source};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;

use super::FragmentStrategy;

/// Metal's offset origin in the coordinates of `InterpolateAtOffset` on a device that reads from
/// the pixel center: `(0.5, 0.5)` from the corner.
const CENTER_ORIGIN: f32 = 0.5;
/// The same on a device that hands the offset to Metal plus 7/16
/// (`VkProperty::InterpolateAtOffsetFromSevenSixteenths`).
const SEVEN_SIXTEENTHS_ORIGIN: f32 = 0.4375;

/// Select interpolation function `i`, a call with operands `args`: its interpolant must be an
/// interpolant parameter itself, which `lower::stage_io` planned, and its result is of the
/// interpolant's form (every call of one interpolant returns one type, which the plan checked).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    i: &Interpolate,
    args: &[Value],
) -> Result<FragmentStrategy, Error> {
    let refused =
        || -> Error { Unsupported::not_implemented(Feature::PullModelInterpolation).into() };
    if cx.stage != Stage::Fragment {
        return Err(refused());
    }
    let Some(Value::Arg(a)) = args.first() else {
        return Err(refused());
    };
    let param = a.raw();
    let planned = cx
        .families
        .stage_io
        .params
        .iter()
        .find(|p| p.param == param)
        .map(|p| p.source);
    if !matches!(
        planned,
        Some(Source::Interpolant { .. } | Source::Absent(_))
    ) {
        return Err(cx.invariant("an interpolation function of no interpolant"));
    }
    let operand = |cx: &mut SelectCx<'_>| {
        let v = *args
            .get(1)
            .ok_or_else(|| cx.invariant("an interpolation function without its operand"))?;
        let ty = cx
            .value_ty(v)
            .ok_or_else(|| cx.invariant("an interpolation operand without a type"))?;
        cx.lowered_repr(v, ty)?;
        cx.values
            .id(v)
            .ok_or_else(|| cx.invariant("an interpolation operand out of range"))
    };
    let at = match i.at {
        InterpolateAt::Center => InterpolantAt::Center,
        InterpolateAt::Centroid => InterpolantAt::Centroid,
        InterpolateAt::Sample => {
            cx.facts.insert(RuntimeFact::SampleCount);
            InterpolantAt::Sample(operand(cx)?)
        }
        InterpolateAt::Offset => {
            let offset = operand(cx)?;
            let properties = &cx.profile.fields().enabled.properties;
            let origin = if properties.contains(&VkProperty::InterpolateAtOffsetCentered) {
                CENTER_ORIGIN
            } else if properties.contains(&VkProperty::InterpolateAtOffsetFromSevenSixteenths) {
                cx.families.stage_io.offsets_from_seven_sixteenths = true;
                SEVEN_SIXTEENTHS_ORIGIN
            } else {
                return Err(Unsupported::needs_any_of(
                    Feature::PullModelInterpolation,
                    vec![
                        vec![Capability::Property(
                            VkProperty::InterpolateAtOffsetCentered,
                        )],
                        vec![Capability::Property(
                            VkProperty::InterpolateAtOffsetFromSevenSixteenths,
                        )],
                    ],
                )
                .into());
            };
            let limits = &cx.profile.fields().limits;
            let (min, max) = (
                limits.min_interpolation_offset_bits,
                limits.max_interpolation_offset_bits,
            );
            cx.precision
                .insert(PrecisionNote::DeviceInterpolationOffset);
            InterpolantAt::Offset {
                offset,
                origin: origin.to_bits(),
                min,
                max,
            }
        }
    };
    Ok(FragmentStrategy::Interpolate(InterpolantRead { param, at }))
}
