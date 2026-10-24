//! Blending into a color attachment held with Metal's channels rotated.
//!
//! An `ARGB8Unorm` attachment's image is `R8G8B8A8_UNORM`, holding Metal's alpha, red, green and
//! blue in its red, green, blue and alpha (`metal::format::to_rotated_image`). Metal blends the
//! format as `RGBA8Unorm`, channel by channel: its alpha by the alpha operation and factors, which
//! read Metal's alphas, and its red, green and blue by the color ones (eleven blend
//! states, every texel the `RGBA8Unorm` one's rotated). Vulkan's color operation and factors govern
//! the image's red, green and blue (Metal's alpha, red and green), its alpha ones the image's alpha
//! (Metal's blue), and a Vulkan factor that reads an alpha reads the image's (Metal's blue). So
//! [`rotated_blend`] looks for Vulkan factors and operations whose value at each channel of the
//! image is Metal's at the channel the image holds there:
//!
//! - **One operation.** Metal's color and alpha operations must be one, as Vulkan's color
//!   operation governs Metal's alpha and red alike. A minimum or a maximum takes no factor.
//! - **Factors the image answers in place:** zero, one, the source's or destination's own
//!   channel, one minus it, and the blend constant's alpha (Metal's, which the encoder sets
//!   unrotated). An alpha factor of Metal's reads Metal's alpha, which is the channel's own there.
//! - **Factors of a second output.** Where Metal's factors read a source channel the image's
//!   factors cannot reach there (Metal's source alpha in a color factor, as `SourceAlpha` over
//!   `OneMinusSourceAlpha` reads it), the fragment function writes a second output whose channels
//!   hold copies of its first output's channels, or constants
//!   (`LoweringInputs::rotated_factors`), and Vulkan's `SRC1` factors read them: the blend reads
//!   the very value Metal's blend reads, and computes one minus it itself, as Metal's does. This
//!   needs `dualSrcBlend`. A blend of Metal's own second output (`Source1` factors) reads that
//!   output, which the function stores rotated as the image's channels are.
//! - **What no such factor gives is `NotImplemented(Format)`:** Metal's destination alpha in a
//!   color factor, the saturated source alpha, the blend color's red, green or blue, a second
//!   output of Metal's own beside one the rotation needs, and different color and alpha
//!   operations. A one minus a value the function would compute is not offered: it would round
//!   apart from the blend's own.
//!
//! **Measured forms.** Apple's blend does not compute every factor pair alike in its color and
//! alpha arithmetic, and the rotation moves Metal's blue into the image's alpha and Metal's alpha
//! into its red: under `SourceAlpha` over `OneMinusSourceAlpha` Metal's color is `fma(d, 1 - a, s *
//! a)`, where the image's alpha blends a factor and one minus it as `fma(f, s - d, d)`, which
//! rounds one code apart near half-code ties. A swept 440 blend states held 1 to 1,767 differing
//! texels of 65,536 in 40 of them. The blend states [`MEASURED`] lists store Metal's bytes rotated
//! in the forms this module gives them, on the M4 and the M3 Max at 65,536 tie-heavy texels and on
//! every lane device at the rotated-targets lane's; under `SourceAlpha` over `OneMinusSourceAlpha`
//! with the add operation, where the device's blend is Apple's (`VkProperty::MetalColorWrites`),
//! the image's alpha takes one over one minus the second output's alpha and the function writes
//! Metal's blue premultiplied by Metal's alpha (`RotatedBlendForm::Premultiplied`), which the blend
//! then computes as Metal's own `fma(d, 1 - a, s * a)`. Every other blend state is the search's,
//! announced (`RotatedBlendForm::Unmeasured`, `PrecisionNote::RotatedBlendRounding`).

use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::metal::state::{blend_factors, blend_op};
use crate::metal::{BlendFactor, BlendOperation};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::graphics::BlendState;
use crate::pipeline::lower_inputs::{FactorLane, RotatedBlendForm};
use crate::vk;

/// The Metal channel (0 to 3: red, green, blue, alpha) each channel of the image holds: its red
/// holds Metal's alpha, and its green, blue and alpha Metal's red, green and blue.
const HELD: [u8; 4] = [3, 0, 1, 2];

/// Vulkan's blend of an attachment held rotated, which link takes (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RotatedBlend {
    /// Factors that read the image, the constant, or Metal's own second output.
    Direct(BlendState),
    /// Factors that read the second output the function writes for the rotation, whose channels
    /// the lanes give (`LoweringInputs::rotated_factors`).
    Synthesized(BlendState, [FactorLane; 4]),
}

/// One measured blend state (the module docs): Metal's color source and destination factors,
/// its alpha source and destination factors, and its one operation; and, for the straight pair
/// on a device whose blend is Apple's, the premultiplied form's Vulkan factors (color source,
/// color destination, alpha source, alpha destination) and second-output lanes.
struct Measured {
    metal: ([BlendFactor; 4], BlendOperation),
    premultiplied: Option<([vk::BlendFactor; 4], [FactorLane; 4])>,
}

/// The measured blend states (the module docs): the rotated-targets lane's blended cases, and
/// one over zero and one over one, whose arithmetic is the one channel's or one sum.
const MEASURED: [Measured; 11] = {
    use vk::BlendFactor as B;
    use BlendFactor as F;
    use BlendOperation as O;
    use FactorLane::{One, Output, Zero};
    const fn plain(metal: [BlendFactor; 4], op: BlendOperation) -> Measured {
        Measured {
            metal: (metal, op),
            premultiplied: None,
        }
    }
    [
        plain([F::One, F::Zero, F::One, F::Zero], O::Add),
        plain([F::One, F::One, F::One, F::One], O::Add),
        plain(
            [F::DestinationColor, F::Zero, F::DestinationAlpha, F::Zero],
            O::Add,
        ),
        plain(
            [
                F::SourceColor,
                F::OneMinusSourceColor,
                F::SourceAlpha,
                F::OneMinusSourceAlpha,
            ],
            O::Add,
        ),
        plain(
            [
                F::BlendAlpha,
                F::OneMinusBlendAlpha,
                F::BlendAlpha,
                F::OneMinusBlendAlpha,
            ],
            O::Add,
        ),
        // Straight alpha: the image's red, green and blue by the second output's alpha (Metal's
        // source alpha), its alpha (Metal's blue) premultiplied.
        Measured {
            metal: (
                [
                    F::SourceAlpha,
                    F::OneMinusSourceAlpha,
                    F::SourceAlpha,
                    F::OneMinusSourceAlpha,
                ],
                O::Add,
            ),
            premultiplied: Some((
                [
                    B::SRC1_ALPHA,
                    B::ONE_MINUS_SRC1_ALPHA,
                    B::ONE,
                    B::ONE_MINUS_SRC1_ALPHA,
                ],
                [Zero, Zero, Zero, Output(3)],
            )),
        },
        // Straight alpha, Metal's alpha accumulated: the image's red (Metal's alpha) by one.
        Measured {
            metal: (
                [
                    F::SourceAlpha,
                    F::OneMinusSourceAlpha,
                    F::One,
                    F::OneMinusSourceAlpha,
                ],
                O::Add,
            ),
            premultiplied: Some((
                [
                    B::SRC1_COLOR,
                    B::ONE_MINUS_SRC1_ALPHA,
                    B::ONE,
                    B::ONE_MINUS_SRC1_ALPHA,
                ],
                [One, Output(3), Output(3), Output(3)],
            )),
        },
        plain(
            [
                F::One,
                F::OneMinusSourceAlpha,
                F::One,
                F::OneMinusSourceAlpha,
            ],
            O::Add,
        ),
        plain([F::SourceAlpha, F::One, F::Zero, F::One], O::Add),
        plain(
            [F::SourceAlpha, F::One, F::SourceAlpha, F::One],
            O::ReverseSubtract,
        ),
        plain(
            [
                F::One,
                F::OneMinusSourceAlpha,
                F::One,
                F::OneMinusSourceAlpha,
            ],
            O::Subtract,
        ),
    ]
};

/// The measured blend state `a` blends by ([`MEASURED`]), if it is one.
fn measured(a: &ColorAttachmentDescriptor) -> Option<&'static Measured> {
    let factors = [
        a.source_rgb_blend_factor,
        a.destination_rgb_blend_factor,
        a.source_alpha_blend_factor,
        a.destination_alpha_blend_factor,
    ];
    if a.rgb_blend_operation != a.alpha_blend_operation {
        return None;
    }
    MEASURED
        .iter()
        .find(|m| m.metal == (factors, a.rgb_blend_operation))
}

/// The premultiplied form of `a` on `p` (the module docs): a measured straight pair's, where the
/// device's blend is Apple's (`VkProperty::MetalColorWrites`: MoltenVK on the measured Apple GPUs,
/// whose blend is Metal's own).
fn premultiplied(
    a: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Option<([vk::BlendFactor; 4], [FactorLane; 4])> {
    let apple = p
        .fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalColorWrites);
    measured(a)?.premultiplied.filter(|_| apple)
}

/// The form of the blend of attachment `a`, held with its channels rotated, on `p`
/// (`LoweringInputs::rotated_blends`, the module docs): `Premultiplied` for a measured straight
/// pair where the device's blend is Apple's, `Unmeasured` for a blend [`rotated_blend`] admits in
/// none of the measured forms, and `None` without blending, under a minimum or a maximum (which
/// takes no factor), for another measured form, and for a blend no Vulkan state gives (which link
/// refuses).
pub(crate) fn rotated_form(
    a: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Option<RotatedBlendForm> {
    if !matches!(rotated_blend(a, p), Ok(Some(_))) {
        return None;
    }
    // The same resolution [`rotated_blend`] makes, so the form of a blend holding a placeholder is
    // the form of the blend Metal draws with.
    let a = &a.specialized();
    if matches!(
        a.rgb_blend_operation,
        BlendOperation::Min | BlendOperation::Max
    ) {
        return None;
    }
    match measured(a) {
        Some(_) if premultiplied(a, p).is_some() => Some(RotatedBlendForm::Premultiplied),
        Some(_) => None,
        None => Some(RotatedBlendForm::Unmeasured),
    }
}

/// The value a blend factor takes at one channel, in Metal's channels (0 to 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    Zero,
    One,
    Source(u8),
    OneMinusSource(u8),
    Destination(u8),
    OneMinusDestination(u8),
    Second(u8),
    OneMinusSecond(u8),
    Constant(u8),
    OneMinusConstant(u8),
    /// The source alpha saturated, `min(As, 1 - Ad)`.
    Saturated,
}

/// What one channel of the second output holds: [`FactorLane`]'s, or Metal's own second output's
/// channel `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Zero,
    One,
    Output(u8),
    Second(u8),
}

/// The value of Metal's factor `f` at Metal's channel `m`: an alpha factor's (`m` 3) reads alphas
/// only, and the saturated source alpha is one there. `None` for the placeholder.
fn metal_value(f: BlendFactor, m: u8) -> Option<Value> {
    use BlendFactor as F;
    use Value as V;
    Some(match f {
        F::Zero => V::Zero,
        F::One => V::One,
        F::SourceColor => V::Source(m),
        F::OneMinusSourceColor => V::OneMinusSource(m),
        F::SourceAlpha => V::Source(3),
        F::OneMinusSourceAlpha => V::OneMinusSource(3),
        F::DestinationColor => V::Destination(m),
        F::OneMinusDestinationColor => V::OneMinusDestination(m),
        F::DestinationAlpha => V::Destination(3),
        F::OneMinusDestinationAlpha => V::OneMinusDestination(3),
        F::SourceAlphaSaturated if m == 3 => V::One,
        F::SourceAlphaSaturated => V::Saturated,
        F::BlendColor => V::Constant(m),
        F::OneMinusBlendColor => V::OneMinusConstant(m),
        F::BlendAlpha => V::Constant(3),
        F::OneMinusBlendAlpha => V::OneMinusConstant(3),
        F::Source1Color => V::Second(m),
        F::OneMinusSource1Color => V::OneMinusSecond(m),
        F::Source1Alpha => V::Second(3),
        F::OneMinusSource1Alpha => V::OneMinusSecond(3),
        F::Unspecialized => return None,
    })
}

/// The Vulkan factors tried, in order: those the image answers, then those of the second output.
const ORDER: [vk::BlendFactor; 18] = [
    vk::BlendFactor::ZERO,
    vk::BlendFactor::ONE,
    vk::BlendFactor::SRC_COLOR,
    vk::BlendFactor::ONE_MINUS_SRC_COLOR,
    vk::BlendFactor::DST_COLOR,
    vk::BlendFactor::ONE_MINUS_DST_COLOR,
    vk::BlendFactor::SRC_ALPHA,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendFactor::DST_ALPHA,
    vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    vk::BlendFactor::CONSTANT_COLOR,
    vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    vk::BlendFactor::CONSTANT_ALPHA,
    vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
    vk::BlendFactor::SRC1_COLOR,
    vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
    vk::BlendFactor::SRC1_ALPHA,
    vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
];

/// The value of Vulkan's factor `f` at channel `i` of the image, in Metal's channels: the source's
/// and destination's channel there are Metal's `HELD[i]`, an alpha is the image's (Metal's blue),
/// and the blend constant is Metal's as the encoder sets it. `None` for a factor of the second
/// output ([`second_lanes`]) and for the saturated alpha, which reads Metal's blue.
fn vulkan_value(f: vk::BlendFactor, i: usize) -> Option<Value> {
    use vk::BlendFactor as B;
    use Value as V;
    let (own, alpha) = (HELD[i], HELD[3]);
    Some(match f {
        B::ZERO => V::Zero,
        B::ONE => V::One,
        B::SRC_COLOR => V::Source(own),
        B::ONE_MINUS_SRC_COLOR => V::OneMinusSource(own),
        B::DST_COLOR => V::Destination(own),
        B::ONE_MINUS_DST_COLOR => V::OneMinusDestination(own),
        B::SRC_ALPHA => V::Source(alpha),
        B::ONE_MINUS_SRC_ALPHA => V::OneMinusSource(alpha),
        B::DST_ALPHA => V::Destination(alpha),
        B::ONE_MINUS_DST_ALPHA => V::OneMinusDestination(alpha),
        B::CONSTANT_COLOR => V::Constant(i as u8),
        B::ONE_MINUS_CONSTANT_COLOR => V::OneMinusConstant(i as u8),
        B::CONSTANT_ALPHA => V::Constant(3),
        B::ONE_MINUS_CONSTANT_ALPHA => V::OneMinusConstant(3),
        _ => return None,
    })
}

/// The lane a second output holds for an `SRC1` factor to take the value `v` (`one_minus`: for a
/// `ONE_MINUS_SRC1` factor to): a copy of a channel, or a constant.
fn lane_for(v: Value, one_minus: bool) -> Option<Lane> {
    use Value as V;
    Some(match (v, one_minus) {
        (V::Zero, false) | (V::One, true) => Lane::Zero,
        (V::One, false) | (V::Zero, true) => Lane::One,
        (V::Source(m), false) | (V::OneMinusSource(m), true) => Lane::Output(m),
        (V::Second(m), false) | (V::OneMinusSecond(m), true) => Lane::Second(m),
        _ => return None,
    })
}

/// The lanes of the second output Vulkan's factor `f` needs to take the value `wanted[i]` at each
/// channel `i` of `channels`, where `f` reads the second output; `None` where it cannot.
fn second_lanes(
    f: vk::BlendFactor,
    channels: &[usize],
    wanted: &[Value; 4],
) -> Option<[Option<Lane>; 4]> {
    use vk::BlendFactor as B;
    let (one_minus, alpha) = match f {
        B::SRC1_COLOR => (false, false),
        B::ONE_MINUS_SRC1_COLOR => (true, false),
        B::SRC1_ALPHA => (false, true),
        B::ONE_MINUS_SRC1_ALPHA => (true, true),
        _ => return None,
    };
    let mut lanes = [None; 4];
    for &i in channels {
        let lane = lane_for(wanted[i], one_minus)?;
        // An `SRC1_ALPHA` factor reads the second output's alpha at every channel.
        let at = if alpha { 3 } else { i };
        match lanes[at] {
            Some(l) if l != lane => return None,
            _ => lanes[at] = Some(lane),
        }
    }
    Some(lanes)
}

/// The Vulkan factors that take the value `wanted[i]` at each channel `i` of `channels`, in
/// [`ORDER`], each with the lanes of the second output it needs (none for a factor the image
/// answers).
fn candidates(
    channels: &[usize],
    wanted: &[Value; 4],
) -> Vec<(vk::BlendFactor, [Option<Lane>; 4])> {
    ORDER
        .into_iter()
        .filter_map(|f| {
            if channels
                .iter()
                .all(|&i| vulkan_value(f, i) == Some(wanted[i]))
            {
                return Some((f, [None; 4]));
            }
            second_lanes(f, channels, wanted).map(|lanes| (f, lanes))
        })
        .collect()
}

/// `a`'s lanes and `b`'s together, or `None` where they hold different values in one channel.
fn merged(a: [Option<Lane>; 4], b: [Option<Lane>; 4]) -> Option<[Option<Lane>; 4]> {
    let mut out = a;
    for (o, l) in out.iter_mut().zip(b) {
        match (*o, l) {
            (Some(x), Some(y)) if x != y => return None,
            (None, l) => *o = l,
            _ => {}
        }
    }
    Some(out)
}

/// Metal's factor of each channel of the image under the color factor `rgb` and the alpha factor
/// `alpha`: the alpha factor at its red (Metal's alpha), the color factor elsewhere. `None` for
/// a placeholder.
fn metal_values(rgb: BlendFactor, alpha: BlendFactor) -> Option<[Value; 4]> {
    Some([
        metal_value(alpha, HELD[0])?,
        metal_value(rgb, HELD[1])?,
        metal_value(rgb, HELD[2])?,
        metal_value(rgb, HELD[3])?,
    ])
}

/// Vulkan's blend of attachment `a` on an image holding Metal's channels rotated, on `p` (the
/// module docs): `None` without blending, `NotImplemented(Format)` where no Vulkan state is
/// Metal's, `Unsupported{DualSourceBlending, Feature(DualSrcBlend)}` where only a second output
/// gives it and the device has no dual-source blending, and the descriptor's own errors.
pub(crate) fn rotated_blend(
    a: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Result<Option<RotatedBlend>, Error> {
    // Metal 4's blend placeholders resolve to their slot's own default before anything here reads
    // a factor or an operation, so the rotated form of a blend holding one is the rotated form of
    // the blend Metal draws with (`ColorAttachmentDescriptor::specialized`).
    let a = &a.specialized();
    // The descriptor's own errors first: a `Source1` factor without dual-source blending.
    let Some(plain) = blend_factors(a, p)? else {
        return Ok(None);
    };
    let refused = || Error::from(Unsupported::not_implemented(Feature::Format));
    if a.rgb_blend_operation != a.alpha_blend_operation {
        return Err(refused());
    }
    if matches!(
        a.rgb_blend_operation,
        BlendOperation::Min | BlendOperation::Max
    ) {
        return Ok(Some(RotatedBlend::Direct(plain)));
    }
    let op = blend_op(a.rgb_blend_operation)?;
    // A measured straight pair on Apple's blend: its premultiplied form.
    if let Some(([src_color, dst_color, src_alpha, dst_alpha], lanes)) = premultiplied(a, p) {
        if !p.fields().fragment.dual_src_blend {
            return Err(Unsupported::needs(
                Feature::DualSourceBlending,
                Capability::Feature(VkFeature::DualSrcBlend),
            )
            .into());
        }
        let state = BlendState {
            src_color,
            dst_color,
            color_op: op,
            src_alpha,
            dst_alpha,
            alpha_op: op,
        };
        return Ok(Some(RotatedBlend::Synthesized(state, lanes)));
    }
    let src = metal_values(a.source_rgb_blend_factor, a.source_alpha_blend_factor);
    let dst = metal_values(
        a.destination_rgb_blend_factor,
        a.destination_alpha_blend_factor,
    );
    let (Some(src), Some(dst)) = (src, dst) else {
        return Err(refused());
    };
    let (color, alpha): (&[usize], &[usize]) = (&[0, 1, 2], &[3]);
    let [src_colors, dst_colors, src_alphas, dst_alphas] = [
        candidates(color, &src),
        candidates(color, &dst),
        candidates(alpha, &src),
        candidates(alpha, &dst),
    ];
    // Metal's own second output, which the function stores rotated, is the one Vulkan's factors
    // read where Metal's blend reads it; a second output for the rotation otherwise.
    let own_second = plain.reads_second_output();
    let fits = |lanes: &[Option<Lane>; 4]| {
        lanes.iter().enumerate().all(|(i, l)| match l {
            None => true,
            Some(Lane::Second(m)) => own_second && *m == HELD[i],
            Some(_) => !own_second,
        })
    };
    let mut fitting = Vec::new();
    for (s, sl) in &src_colors {
        for (d, dl) in &dst_colors {
            for (sa, sal) in &src_alphas {
                for (da, dal) in &dst_alphas {
                    let lanes = merged(*sl, *dl)
                        .and_then(|l| merged(l, *sal))
                        .and_then(|l| merged(l, *dal));
                    if let Some(lanes) = lanes.filter(fits) {
                        fitting.push(([*s, *d, *sa, *da], lanes));
                    }
                }
            }
        }
    }
    // The first in order, preferring factors the image answers alone.
    let alone =
        |(_, lanes): &&([vk::BlendFactor; 4], [Option<Lane>; 4])| lanes.iter().all(Option::is_none);
    let chosen = fitting.iter().find(alone).or(fitting.first());
    let Some(&([src_color, dst_color, src_alpha, dst_alpha], lanes)) = chosen else {
        return Err(refused());
    };
    let state = BlendState {
        src_color,
        dst_color,
        color_op: op,
        src_alpha,
        dst_alpha,
        alpha_op: op,
    };
    if own_second || lanes.iter().all(Option::is_none) {
        return Ok(Some(RotatedBlend::Direct(state)));
    }
    if !p.fields().fragment.dual_src_blend {
        return Err(Unsupported::needs(
            Feature::DualSourceBlending,
            Capability::Feature(VkFeature::DualSrcBlend),
        )
        .into());
    }
    let lanes = lanes.map(|l| match l {
        Some(Lane::One) => FactorLane::One,
        Some(Lane::Output(m)) => FactorLane::Output(m),
        _ => FactorLane::Zero,
    });
    Ok(Some(RotatedBlend::Synthesized(state, lanes)))
}

/// Vulkan's blend of attachment `a` on an `R8_UNORM` image holding Metal's alpha in red
/// (`metal::format::alpha_in_red`, `A8Unorm`), whose fragment output the function writes rotated
/// (the red of each output, of either dual-source index, Metal's alpha): `None` without blending,
/// and the descriptor's own errors. Metal blends `A8Unorm`'s alpha by its alpha operation and
/// factors alone, each factor reading alphas: `SourceColor` and `SourceAlpha` the source's,
/// `DestinationColor` and `DestinationAlpha` the destination's, `SourceAlphaSaturated` one,
/// `BlendColor` and `BlendAlpha` the blend color's, `Source1Color` and `Source1Alpha` the second
/// output's (probe `a8`, an M3 Max under macOS 27.0.1: every factor as source and as
/// destination, under every operation). Vulkan's color operation and factors govern the image's
/// red, where each of those alphas is that of the red it reads (the source's red, the
/// destination's, the second output's), but the blend color's, which `CONSTANT_ALPHA` reads
/// unrotated, and the saturated alpha's one. The image has no alpha, so its alpha factors are the
/// same as its color factors, read by nothing.
pub(crate) fn alpha_in_red_blend(
    a: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Result<Option<BlendState>, Error> {
    // The descriptor's own errors first: a placeholder, and a `Source1` factor without
    // dual-source blending.
    if blend_factors(a, p)?.is_none() {
        return Ok(None);
    }
    use vk::BlendFactor as B;
    use BlendFactor as F;
    let red = |f: BlendFactor| match f {
        F::Zero => Some(B::ZERO),
        F::One | F::SourceAlphaSaturated => Some(B::ONE),
        F::SourceColor | F::SourceAlpha => Some(B::SRC_COLOR),
        F::OneMinusSourceColor | F::OneMinusSourceAlpha => Some(B::ONE_MINUS_SRC_COLOR),
        F::DestinationColor | F::DestinationAlpha => Some(B::DST_COLOR),
        F::OneMinusDestinationColor | F::OneMinusDestinationAlpha => Some(B::ONE_MINUS_DST_COLOR),
        F::BlendColor | F::BlendAlpha => Some(B::CONSTANT_ALPHA),
        F::OneMinusBlendColor | F::OneMinusBlendAlpha => Some(B::ONE_MINUS_CONSTANT_ALPHA),
        F::Source1Color | F::Source1Alpha => Some(B::SRC1_COLOR),
        F::OneMinusSource1Color | F::OneMinusSource1Alpha => Some(B::ONE_MINUS_SRC1_COLOR),
        F::Unspecialized => None,
    };
    let (Some(src), Some(dst)) = (
        red(a.source_alpha_blend_factor),
        red(a.destination_alpha_blend_factor),
    ) else {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    };
    let op = blend_op(a.alpha_blend_operation)?;
    Ok(Some(BlendState {
        src_color: src,
        dst_color: dst,
        color_op: op,
        src_alpha: src,
        dst_alpha: dst,
        alpha_op: op,
    }))
}

#[cfg(test)]
mod tests;
