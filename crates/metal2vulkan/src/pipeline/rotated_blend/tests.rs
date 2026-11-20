//! [`rotated_blend`] against a numeric model of both blends: Metal's of an `RGBA8Unorm` texel,
//! channel by channel, and Vulkan's of the image holding it rotated, under the chosen state and
//! second output.

use std::path::Path;

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Needs;
use crate::metal::PixelFormat;
use crate::pipeline::testing::profile;

/// One texel's four channels.
type Texel = [f64; 4];

/// The value of Metal's factor `f` at Metal's channel `m` (3: alpha), as Metal's documentation
/// states each factor.
fn metal_factor(f: BlendFactor, m: usize, (s, s1, d, c): (Texel, Texel, Texel, Texel)) -> f64 {
    use BlendFactor as F;
    match f {
        F::Zero => 0.0,
        F::One => 1.0,
        F::SourceColor => s[m],
        F::OneMinusSourceColor => 1.0 - s[m],
        F::SourceAlpha => s[3],
        F::OneMinusSourceAlpha => 1.0 - s[3],
        F::DestinationColor => d[m],
        F::OneMinusDestinationColor => 1.0 - d[m],
        F::DestinationAlpha => d[3],
        F::OneMinusDestinationAlpha => 1.0 - d[3],
        F::SourceAlphaSaturated if m == 3 => 1.0,
        F::SourceAlphaSaturated => s[3].min(1.0 - d[3]),
        F::BlendColor => c[m],
        F::OneMinusBlendColor => 1.0 - c[m],
        F::BlendAlpha => c[3],
        F::OneMinusBlendAlpha => 1.0 - c[3],
        F::Source1Color => s1[m],
        F::OneMinusSource1Color => 1.0 - s1[m],
        F::Source1Alpha => s1[3],
        F::OneMinusSource1Alpha => 1.0 - s1[3],
        F::Unspecialized => unreachable!("no placeholder is drawn"),
    }
}

/// The value of Vulkan's factor `f` at channel `i` of an attachment (3: alpha), as Vulkan's
/// specification states each factor.
fn vulkan_factor(f: vk::BlendFactor, i: usize, (s, s1, d, c): (Texel, Texel, Texel, Texel)) -> f64 {
    use vk::BlendFactor as B;
    match f {
        B::ZERO => 0.0,
        B::ONE => 1.0,
        B::SRC_COLOR => s[i],
        B::ONE_MINUS_SRC_COLOR => 1.0 - s[i],
        B::DST_COLOR => d[i],
        B::ONE_MINUS_DST_COLOR => 1.0 - d[i],
        B::SRC_ALPHA => s[3],
        B::ONE_MINUS_SRC_ALPHA => 1.0 - s[3],
        B::DST_ALPHA => d[3],
        B::ONE_MINUS_DST_ALPHA => 1.0 - d[3],
        B::CONSTANT_COLOR => c[i],
        B::ONE_MINUS_CONSTANT_COLOR => 1.0 - c[i],
        B::CONSTANT_ALPHA => c[3],
        B::ONE_MINUS_CONSTANT_ALPHA => 1.0 - c[3],
        B::SRC1_COLOR => s1[i],
        B::ONE_MINUS_SRC1_COLOR => 1.0 - s1[i],
        B::SRC1_ALPHA => s1[3],
        B::ONE_MINUS_SRC1_ALPHA => 1.0 - s1[3],
        B::SRC_ALPHA_SATURATE if i == 3 => 1.0,
        B::SRC_ALPHA_SATURATE => s[3].min(1.0 - d[3]),
        other => panic!("{other:?} is no blend factor"),
    }
}

fn apply(op: BlendOperation, sf: f64, s: f64, df: f64, d: f64) -> f64 {
    match op {
        BlendOperation::Add => sf * s + df * d,
        BlendOperation::Subtract => sf * s - df * d,
        BlendOperation::ReverseSubtract => df * d - sf * s,
        BlendOperation::Min => s.min(d),
        BlendOperation::Max => s.max(d),
        BlendOperation::Unspecialized => unreachable!("no placeholder is drawn"),
    }
}

fn vk_op(op: vk::BlendOp) -> BlendOperation {
    match op {
        vk::BlendOp::ADD => BlendOperation::Add,
        vk::BlendOp::SUBTRACT => BlendOperation::Subtract,
        vk::BlendOp::REVERSE_SUBTRACT => BlendOperation::ReverseSubtract,
        vk::BlendOp::MIN => BlendOperation::Min,
        vk::BlendOp::MAX => BlendOperation::Max,
        other => panic!("{other:?}"),
    }
}

/// Metal's blend of `a` (source `s`, second output `s1`, destination `d`, blend color `c`).
fn metal_blend(a: &ColorAttachmentDescriptor, v: (Texel, Texel, Texel, Texel)) -> Texel {
    std::array::from_fn(|m| {
        let (sf, df, op) = match m {
            3 => (
                a.source_alpha_blend_factor,
                a.destination_alpha_blend_factor,
                a.alpha_blend_operation,
            ),
            _ => (
                a.source_rgb_blend_factor,
                a.destination_rgb_blend_factor,
                a.rgb_blend_operation,
            ),
        };
        let (s, d) = (v.0[m], v.2[m]);
        apply(op, metal_factor(sf, m, v), s, metal_factor(df, m, v), d)
    })
}

/// Vulkan's blend `b` of an attachment (source `s`, second output `s1`, destination `d`, blend
/// constants `c`).
fn vulkan_blend(b: &BlendState, v: (Texel, Texel, Texel, Texel)) -> Texel {
    std::array::from_fn(|i| {
        let (sf, df, op) = match i {
            3 => (b.src_alpha, b.dst_alpha, b.alpha_op),
            _ => (b.src_color, b.dst_color, b.color_op),
        };
        let (s, d) = (v.0[i], v.2[i]);
        apply(
            vk_op(op),
            vulkan_factor(sf, i, v),
            s,
            vulkan_factor(df, i, v),
            d,
        )
    })
}

fn rotated(t: Texel) -> Texel {
    crate::metal::format::to_rotated_image(t)
}

/// Pseudo-random texels of 256ths, from `seed`.
fn texels(seed: u64) -> impl Iterator<Item = (Texel, Texel, Texel, Texel)> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 33) % 257) as f64 / 256.0
    };
    std::iter::repeat_with(move || {
        let mut t = || [next(), next(), next(), next()];
        (t(), t(), t(), t())
    })
}

/// That Vulkan's blend `chosen` of `a` on the rotated image on `p` stores Metal's blend rotated,
/// over pseudo-random texels: the function's first output rotated, its image alpha (Metal's
/// blue) premultiplied by Metal's alpha where the form says so.
fn check(a: &ColorAttachmentDescriptor, chosen: &RotatedBlend, p: &DeviceProfile) {
    let premultiplied = rotated_form(a, p) == Some(RotatedBlendForm::Premultiplied);
    for v in texels(u64::from(a.source_rgb_blend_factor.as_raw())).take(16) {
        let (s, s1, d, c) = v;
        let mut first = rotated(s);
        if premultiplied {
            first[3] = s[2] * s[3];
        }
        let (state, second) = match *chosen {
            RotatedBlend::Direct(state) => (state, rotated(s1)),
            RotatedBlend::Synthesized(state, lanes) => (
                state,
                lanes.map(|l| match l {
                    FactorLane::Zero => 0.0,
                    FactorLane::One => 1.0,
                    FactorLane::Output(m) => s[usize::from(m)],
                }),
            ),
        };
        let got = vulkan_blend(&state, (first, second, rotated(d), c));
        assert_eq!(got, rotated(metal_blend(a, v)), "{a:?}: {chosen:?}");
    }
}

fn attachment(
    (sr, dr, rgb): (BlendFactor, BlendFactor, BlendOperation),
    (sa, da, alpha): (BlendFactor, BlendFactor, BlendOperation),
) -> ColorAttachmentDescriptor {
    let mut a = ColorAttachmentDescriptor::new(PixelFormat::ARGB8Unorm);
    a.blending_enabled = true;
    a.source_rgb_blend_factor = sr;
    a.destination_rgb_blend_factor = dr;
    a.rgb_blend_operation = rgb;
    a.source_alpha_blend_factor = sa;
    a.destination_alpha_blend_factor = da;
    a.alpha_blend_operation = alpha;
    a
}

fn moltenvk() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

/// Every pair of Metal's color factors, under alpha factors that read nothing, the source's or
/// destination's alpha, or the second output's, and each operation: every blend admitted stores
/// Metal's, and every refusal is `NotImplemented(Format)`.
#[test]
fn every_admitted_blend_stores_metals_rotated() {
    use BlendFactor as F;
    use BlendOperation as O;
    let p = moltenvk();
    let factors: Vec<F> = F::ALL
        .iter()
        .copied()
        .filter(|&f| f != F::Unspecialized)
        .collect();
    let alphas = [
        F::Zero,
        F::One,
        F::SourceAlpha,
        F::OneMinusSourceAlpha,
        F::DestinationAlpha,
        F::Source1Alpha,
    ];
    let (mut admitted, mut synthesized, mut refused) = (0, 0, 0);
    for op in [O::Add, O::Subtract, O::ReverseSubtract, O::Min, O::Max] {
        for &sr in &factors {
            for &dr in &factors {
                for sa in alphas {
                    for da in alphas {
                        let a = attachment((sr, dr, op), (sa, da, op));
                        match rotated_blend(&a, &p) {
                            Ok(Some(chosen)) => {
                                check(&a, &chosen, &p);
                                admitted += 1;
                                synthesized +=
                                    usize::from(matches!(chosen, RotatedBlend::Synthesized(..)));
                            }
                            Ok(None) => panic!("{a:?} blends"),
                            Err(e) => {
                                assert_eq!(
                                    e,
                                    Unsupported::not_implemented(Feature::Format).into(),
                                    "{a:?}"
                                );
                                refused += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("admitted {admitted} ({synthesized} through a second output), refused {refused}");
    assert!(admitted > 0 && synthesized > 0 && refused > 0);
}

/// The common blends: kept channel-wise where the image answers them, and through a second output
/// of copies of the source's alpha where a color factor reads it.
#[test]
fn source_alpha_blends_read_a_second_output_of_the_source_alpha() {
    use vk::BlendFactor as B;
    use BlendFactor as F;
    use BlendOperation as O;
    use FactorLane::{One, Output, Zero};
    let p = moltenvk();
    let state = |s, d, sa, da| BlendState {
        src_color: s,
        dst_color: d,
        color_op: vk::BlendOp::ADD,
        src_alpha: sa,
        dst_alpha: da,
        alpha_op: vk::BlendOp::ADD,
    };
    let add = |s, d| (s, d, O::Add);
    let cases = [
        // Straight alpha, Metal's alpha alike: on MoltenVK the measured premultiplied form, the
        // image's alpha (Metal's blue) by one over one minus the source alpha.
        (
            attachment(
                add(F::SourceAlpha, F::OneMinusSourceAlpha),
                add(F::SourceAlpha, F::OneMinusSourceAlpha),
            ),
            RotatedBlend::Synthesized(
                state(
                    B::SRC1_ALPHA,
                    B::ONE_MINUS_SRC1_ALPHA,
                    B::ONE,
                    B::ONE_MINUS_SRC1_ALPHA,
                ),
                [Zero, Zero, Zero, Output(3)],
            ),
        ),
        // Straight alpha, Metal's alpha accumulated: the image's red (Metal's alpha) takes one.
        (
            attachment(
                add(F::SourceAlpha, F::OneMinusSourceAlpha),
                add(F::One, F::OneMinusSourceAlpha),
            ),
            RotatedBlend::Synthesized(
                state(
                    B::SRC1_COLOR,
                    B::ONE_MINUS_SRC1_ALPHA,
                    B::ONE,
                    B::ONE_MINUS_SRC1_ALPHA,
                ),
                [One, Output(3), Output(3), Output(3)],
            ),
        ),
        // Premultiplied alpha.
        (
            attachment(
                add(F::One, F::OneMinusSourceAlpha),
                add(F::One, F::OneMinusSourceAlpha),
            ),
            RotatedBlend::Synthesized(
                state(
                    B::ONE,
                    B::ONE_MINUS_SRC1_COLOR,
                    B::ONE,
                    B::ONE_MINUS_SRC1_COLOR,
                ),
                [Output(3); 4],
            ),
        ),
        // Additive and multiplicative: the image answers them in place.
        (
            attachment(add(F::One, F::One), add(F::One, F::One)),
            RotatedBlend::Direct(state(B::ONE, B::ONE, B::ONE, B::ONE)),
        ),
        (
            attachment(
                add(F::DestinationColor, F::Zero),
                add(F::DestinationAlpha, F::Zero),
            ),
            RotatedBlend::Direct(state(B::DST_COLOR, B::ZERO, B::DST_COLOR, B::ZERO)),
        ),
        // The blend constant's alpha, which the encoder sets unrotated.
        (
            attachment(
                add(F::BlendAlpha, F::OneMinusBlendAlpha),
                add(F::BlendAlpha, F::OneMinusBlendAlpha),
            ),
            RotatedBlend::Direct(state(
                B::CONSTANT_ALPHA,
                B::ONE_MINUS_CONSTANT_ALPHA,
                B::CONSTANT_COLOR,
                B::ONE_MINUS_CONSTANT_COLOR,
            )),
        ),
    ];
    for (a, want) in cases {
        assert_eq!(rotated_blend(&a, &p), Ok(Some(want)), "{a:?}");
        check(&a, &want, &p);
    }
}

/// What no Vulkan factor gives on the rotated image: Metal's destination alpha or saturated
/// source alpha in a color factor, the blend color's red, green or blue, different operations,
/// and Metal's own second output read where the image's alpha is not its alpha.
#[test]
fn blends_no_factor_gives_are_refused_typed() {
    use BlendFactor as F;
    use BlendOperation as O;
    let p = moltenvk();
    let add = |s, d| (s, d, O::Add);
    for a in [
        attachment(
            add(F::DestinationAlpha, F::OneMinusDestinationAlpha),
            add(F::One, F::Zero),
        ),
        attachment(
            add(F::SourceAlphaSaturated, F::One),
            add(F::SourceAlphaSaturated, F::One),
        ),
        attachment(
            add(F::BlendColor, F::OneMinusBlendColor),
            add(F::BlendAlpha, F::OneMinusBlendAlpha),
        ),
        attachment((F::One, F::One, O::Add), (F::One, F::One, O::Max)),
        attachment(add(F::Source1Alpha, F::Zero), add(F::One, F::Zero)),
    ] {
        assert_eq!(
            rotated_blend(&a, &p),
            Err(Unsupported::not_implemented(Feature::Format).into()),
            "{a:?}"
        );
    }
}

/// A blend that needs the second output on a device without dual-source blending names the
/// feature; one the image answers in place needs none.
#[test]
fn a_second_output_needs_dual_source_blending() {
    use BlendFactor as F;
    use BlendOperation as O;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    assert!(desc.features.remove(&VkFeature::DualSrcBlend));
    let p = Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    let straight = attachment(
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
    );
    let Err(Error::Unsupported(u)) = rotated_blend(&straight, &p) else {
        panic!("straight alpha without dual-source blending");
    };
    assert_eq!(u.feature, Feature::DualSourceBlending);
    assert_eq!(
        u.needs,
        Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::DualSrcBlend)]])
    );
    let additive = attachment((F::One, F::One, O::Add), (F::One, F::One, O::Add));
    assert!(matches!(
        rotated_blend(&additive, &p),
        Ok(Some(RotatedBlend::Direct(_)))
    ));
}

/// Every measured blend state is admitted on MoltenVK and on lavapipe and stores Metal's blend
/// rotated; it takes the premultiplied form exactly where it lists one and the device's blend is
/// Apple's (MoltenVK), and the search's form elsewhere, alike on both devices, unannounced.
#[test]
fn measured_blends_take_their_measured_forms() {
    let mvk = moltenvk();
    let lvp = profile("lavapipe", DevicePolicy::default()).expect("profile");
    let mut premultiplied = 0;
    for m in &MEASURED {
        let ([sr, dr, sa, da], op) = m.metal;
        let a = attachment((sr, dr, op), (sa, da, op));
        let on_mvk = rotated_blend(&a, &mvk).expect("admitted").expect("blended");
        let on_lvp = rotated_blend(&a, &lvp).expect("admitted").expect("blended");
        check(&a, &on_mvk, &mvk);
        check(&a, &on_lvp, &lvp);
        assert_eq!(rotated_form(&a, &lvp), None, "{a:?}");
        match m.premultiplied {
            Some((_, lanes)) => {
                premultiplied += 1;
                assert_eq!(
                    rotated_form(&a, &mvk),
                    Some(RotatedBlendForm::Premultiplied),
                    "{a:?}"
                );
                let RotatedBlend::Synthesized(_, chosen) = on_mvk else {
                    panic!("{a:?}: {on_mvk:?}");
                };
                assert_eq!(chosen, lanes, "{a:?}");
                assert_eq!(lanes[3], FactorLane::Output(3), "{a:?}");
                assert_ne!(on_mvk, on_lvp, "{a:?}");
            }
            None => {
                assert_eq!(rotated_form(&a, &mvk), None, "{a:?}");
                assert_eq!(on_mvk, on_lvp, "{a:?}");
            }
        }
    }
    assert_eq!(premultiplied, 2);
}

/// A blend admitted in none of the measured forms is announced on every device; a minimum or
/// maximum (which takes no factor), a blend that is off, and a blend no Vulkan state gives are
/// not.
#[test]
fn unmeasured_blends_are_announced() {
    use BlendFactor as F;
    use BlendOperation as O;
    let mvk = moltenvk();
    let lvp = profile("lavapipe", DevicePolicy::default()).expect("profile");
    // Metal's color kept and its alpha replaced: in the 440-state sweep, 902 of 65,536 texels
    // apart on the M4.
    let kept = attachment((F::Zero, F::One, O::Add), (F::One, F::Zero, O::Add));
    let straight = (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add);
    let mut off = attachment(straight, straight);
    off.blending_enabled = false;
    for p in [&mvk, &lvp] {
        assert!(matches!(rotated_blend(&kept, p), Ok(Some(_))));
        assert_eq!(
            rotated_form(&kept, p),
            Some(RotatedBlendForm::Unmeasured),
            "{p:?}"
        );
        let max = attachment((F::One, F::One, O::Max), (F::One, F::One, O::Max));
        assert!(matches!(rotated_blend(&max, p), Ok(Some(_))));
        assert_eq!(rotated_form(&max, p), None);
        assert_eq!(rotated_form(&off, p), None);
        let refused = attachment(
            (F::DestinationAlpha, F::OneMinusDestinationAlpha, O::Add),
            (F::One, F::Zero, O::Add),
        );
        assert_eq!(rotated_form(&refused, p), None);
    }
}

/// Every pair of Metal's alpha factors under each operation, beside color factors that would read
/// other values: Vulkan's blend [`alpha_in_red_blend`] gives an `R8_UNORM` image holding Metal's
/// alpha in red, whose function writes its outputs rotated, stores in red Metal's blend of
/// `A8Unorm`'s alpha (probe `a8`, an M3 Max under macOS 27.0.1: each alpha factor reads alphas,
/// the color factors nothing). The image's destination reads red as its own and alpha as one.
#[test]
fn an_alpha_in_red_blend_stores_metals_alpha() {
    use BlendFactor as F;
    use BlendOperation as O;
    let p = moltenvk();
    let factors: Vec<F> = F::ALL
        .iter()
        .copied()
        .filter(|&f| f != F::Unspecialized)
        .collect();
    let mut checked = 0;
    for op in [O::Add, O::Subtract, O::ReverseSubtract, O::Min, O::Max] {
        for &sa in &factors {
            for &da in &factors {
                let mut a = attachment(
                    (F::OneMinusDestinationColor, F::SourceAlphaSaturated, O::Add),
                    (sa, da, op),
                );
                a.pixel_format = PixelFormat::A8Unorm;
                let state = alpha_in_red_blend(&a, &p)
                    .unwrap_or_else(|e| panic!("{a:?}: {e:?}"))
                    .expect("blends");
                for v in texels(u64::from(sa.as_raw()) << 8 | u64::from(da.as_raw())).take(16) {
                    let (s, s1, d, c) = v;
                    let image = [d[3], 0.0, 0.0, 1.0];
                    let got = vulkan_blend(&state, (rotated(s), rotated(s1), image, c));
                    assert_eq!(got[0], metal_blend(&a, v)[3], "{a:?}: {state:?}");
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 5 * factors.len() * factors.len());
    let mut off = attachment((F::One, F::Zero, O::Add), (F::One, F::Zero, O::Add));
    off.pixel_format = PixelFormat::A8Unorm;
    off.blending_enabled = false;
    assert_eq!(alpha_in_red_blend(&off, &p), Ok(None));
}
