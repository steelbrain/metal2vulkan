//! Multisampled textures of the formats a device samples but renders only through the view of
//! their word (`image_plan::word_samples`), over every committed device profile
//! (`validation/profiles/`, read-only), each created as the runtime creates a device
//! (`capabilities::created`) and holding an image the limits it states take
//! (`msaa_32bit::stated_limits`). The device test (validation's `runtime_lane::word_samples_msaa`)
//! renders, reads and resolves such textures on a device.

use super::capabilities::{committed_descriptions, created};
use super::msaa_32bit::{stated_limits, Sampled};
use crate::device::create::withhold_attachments;
use crate::image_plan::{
    check_image_limits, compatible, plan_image, DepthImageFacts, ImageFacts, ImagePlan,
};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout, TextureRules};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy, FormatUsage, VkFeature};
use metal2vulkan::metal::format::{direct_format, vk_format, WordAttachment};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags, SampleCountFlags};
use std::collections::BTreeSet;

/// The formats with a word layout whose 2- and 4-sample textures Metal makes on an M3 Max, for
/// every usage (our probe `probes/msaa_word_formats.m`).
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::R8UnormSRGB,
    PixelFormat::A1BGR5Unorm,
    PixelFormat::ABGR4Unorm,
    PixelFormat::RGB9E5Float,
];

/// How a profile's device holds a multisampled texture of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// The format's own image (or an 8-bit sRGB format's UNORM twin), which the device renders.
    Native,
    /// The word's format (`image_plan::word_samples`).
    Word,
    /// The format's sampled image, its render target usage unavailable: the device renders neither
    /// the format nor a word of it.
    Unrendered,
    /// No image the device's stated limits take.
    Refused,
}

fn layout(pf: PixelFormat, samples: u64) -> TextureLayout {
    let checked = TextureDescriptor {
        texture_type: TextureType::Type2DMultisample,
        width: 16,
        height: 16,
        sample_count: samples,
        ..TextureDescriptor::default()
    }
    .check(TextureRules {
        sample_counts: SampleCountFlags::from_raw(0x7f),
    })
    .expect("a valid descriptor");
    TextureLayout {
        pixel_format: pf,
        ..checked
    }
}

/// The device the profile `stem` describes, created as the runtime creates one, each format of
/// `withheld` no color attachment (`DeviceOptions::withheld_attachment_formats`).
fn device(stem: &str, withheld: &BTreeSet<Format>) -> DeviceDescription {
    let desc = committed_descriptions()
        .into_iter()
        .find(|(name, _)| name == stem)
        .map(|(_, desc)| desc)
        .expect("the profile");
    let (mut enabled, _) = created(stem, &desc);
    withhold_attachments(&mut enabled, withheld);
    enabled
}

/// The route of a texture of `pf` at `samples` with `usage` on the device `enabled` of the profile
/// `stem`, a word's plan checked: the word's image, mutable to the format's sampled view, every
/// shader write unavailable, rendered, read and copied as Metal's texture is; and a memoryless one
/// of it the word's transient attachment.
fn route(
    stem: &str,
    enabled: &DeviceDescription,
    pf: PixelFormat,
    samples: u64,
    usage: TextureUsage,
) -> Route {
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let profile = compile.profile();
    let facts = ImageFacts {
        profile,
        formats: &enabled.formats,
        storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: DepthImageFacts::of(&enabled.driver_facts()),
        linear_attachments: false,
        any_views: &crate::image_plan::AnyViews::default(),
    };
    let sampled = Sampled::of(stem);
    let what = format!("{stem}: {pf:?} x{samples} {usage:?}");
    let l = layout(pf, samples);
    let taken = |plan: &ImagePlan| {
        let limits = stated_limits(enabled, &sampled, plan);
        check_image_limits(plan, TextureType::Type2DMultisample, &limits).is_ok()
    };
    let Ok(plan) = plan_image(&l, usage, false, &facts) else {
        return Route::Refused;
    };
    let word = WordAttachment::of(pf, profile).map(WordAttachment::word_format);
    if Some(plan.format) != word {
        return match taken(&plan) {
            false => Route::Refused,
            true if plan.unavailable.contains(TextureUsage::RENDER_TARGET) => Route::Unrendered,
            true => Route::Native,
        };
    }
    let view = vk_format(pf, &[FormatUsage::SampledImage].into(), profile)
        .expect(&what)
        .format;
    assert_eq!(Some(view), direct_format(pf), "{what}");
    assert!(compatible(plan.format, view, false), "{what}");
    // A pixel-format-view texture takes any compatible view: no list. Else the views are the
    // word's, the format's sampled one where read, and an 8-bit sRGB word's UNORM twin, through
    // which its resolve reads and writes its codes.
    let mutable = plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
    match usage.contains(TextureUsage::PIXEL_FORMAT_VIEW) {
        true => assert!(mutable && plan.view_formats.is_empty(), "{what}: {plan:?}"),
        false => {
            let read = usage.contains(TextureUsage::SHADER_READ);
            let raw = pf == PixelFormat::R8UnormSRGB;
            let others: BTreeSet<Format> = [read.then_some(view), raw.then_some(Format::R8_UNORM)]
                .into_iter()
                .flatten()
                .collect();
            match plan.view_formats.split_first() {
                Some((&first, rest)) => {
                    assert_eq!(first, plan.format, "{what}");
                    assert_eq!(
                        rest.iter().copied().collect::<BTreeSet<_>>(),
                        others,
                        "{what}"
                    );
                    assert_eq!(rest.len(), others.len(), "{what}");
                }
                None => assert!(others.is_empty(), "{what}: {plan:?}"),
            }
            assert_eq!(mutable, !others.is_empty(), "{what}");
        }
    }
    let rendered = usage.contains(TextureUsage::RENDER_TARGET);
    let read = usage.contains(TextureUsage::SHADER_READ);
    for (bit, wanted) in [
        (ImageUsageFlags::COLOR_ATTACHMENT, rendered),
        (ImageUsageFlags::SAMPLED, read),
        (ImageUsageFlags::TRANSFER_SRC, true),
        (ImageUsageFlags::TRANSFER_DST, true),
        (ImageUsageFlags::STORAGE, false),
    ] {
        assert_eq!(plan.usage.contains(bit), wanted, "{what}: {bit:?}");
    }
    assert!(
        profile.supports_format(plan.format, FormatUsage::ColorAttachment),
        "{what}"
    );
    assert_eq!(
        plan.unavailable,
        usage.intersection(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC),
        "{what}"
    );
    let mut made = taken(&plan);
    if rendered {
        let memoryless = plan_image(&l, TextureUsage::RENDER_TARGET, true, &facts).expect(&what);
        assert_eq!(memoryless.format, plan.format, "{what}");
        let transient = ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSIENT_ATTACHMENT;
        assert!(memoryless.usage.contains(transient), "{what}");
        assert!(
            !memoryless
                .usage
                .intersects(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC),
            "{what}"
        );
        made &= taken(&memoryless);
    }
    match made {
        true => Route::Word,
        false => Route::Refused,
    }
}

/// Each format's route at 2 and 4 samples on the device `enabled` of the profile `stem`, as a
/// `RenderTarget | ShaderRead` texture.
fn routes(stem: &str, enabled: &DeviceDescription) -> [[Route; 2]; 4] {
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    FORMATS.map(|pf| [2, 4].map(|samples| route(stem, enabled, pf, samples, usage)))
}

/// Each committed profile's route for the four formats at 2 and 4 samples, pinned with its reason:
/// - **The word** where the device samples the format and renders it only through its word, whose
///   unsigned format renders at the count: every format on the RTX 5080 and NVIDIA's other
///   profiles; `R8Unorm_sRGB` and `RGB9E5Float` on ANV, which renders neither and multisamples
///   integer attachments; and `RGB9E5Float` at 4 on the gate's lavapipe, whose `R32_UINT` takes
///   1, 4 and 8 samples (its own counts, which the profile states).
/// - **Native** where the device renders the format at the count: MoltenVK on the M3 (each at 1,
///   2 and 4, as Metal), RADV, ANV's packed formats, and lavapipe's, swiftshader's and the
///   floor's at 4 (`R8Unorm_sRGB` in its UNORM twin where the device lacks `R8_SRGB`).
/// - **Unrendered** where the device samples only another order of a packed format, which a word
///   cannot view as its own: `vk12-floor`'s `ABGR4Unorm` at 4.
/// - **Refused** where the device's textures take no such count (2 on lavapipe, swiftshader,
///   the floors and `lunarg-desktop-baseline-2023` where it lacks the format; `roadmap-2022`
///   states no sampled image's counts), and where the word's format renders one sample
///   (`framebufferIntegerColorSampleCounts` 1, the floor where a profile states no format's own
///   counts: lavapipe's and swiftshader's `RGB9E5Float`, swiftshader's `R8Unorm_sRGB`).
#[test]
fn each_profile_holds_the_four_formats_as_pinned() {
    use Route::{Native as N, Refused as R, Unrendered as U, Word as W};
    // `R8Unorm_sRGB`, `A1BGR5Unorm`, `ABGR4Unorm`, `RGB9E5Float`, each at 2 then 4 samples.
    let radv = [[N, N], [N, N], [N, N], [N, N]];
    let gate_lavapipe = [[R, N], [R, N], [R, N], [R, W]];
    let anv = [[W, W], [N, N], [N, N], [W, W]];
    let nvidia = [[W, W], [W, W], [W, W], [W, W]];
    let pinned: [(&str, [[Route; 2]; 4]); 19] = [
        ("amd-780m-hawkpoint-radv", radv),
        ("amd-780m-radv", radv),
        ("amd-rdna2-apu-radv", radv),
        ("amd-rdna3-radv", radv),
        ("gate-lavapipe", gate_lavapipe),
        ("gate-lavapipe-x86_64", gate_lavapipe),
        ("intel-arl-anv", anv),
        ("intel-irisxe-anv", anv),
        ("intel-irisxe-rplp-anv", anv),
        ("intel-uhd770-rpls-anv", anv),
        ("lavapipe", [[R, N], [R, N], [R, N], [R, R]]),
        (
            "lunarg-desktop-baseline-2023",
            [[N, N], [R, R], [R, R], [R, R]],
        ),
        ("moltenvk-m3", [[N, N], [N, N], [N, N], [N, N]]),
        ("nvidia-pascal-linux", nvidia),
        ("nvidia-rtx50-linux", nvidia),
        ("nvidia-rtx5080-laptop", nvidia),
        ("roadmap-2022", [[R, R], [R, R], [R, R], [R, R]]),
        ("swiftshader", [[R, R], [R, N], [R, N], [R, R]]),
        ("vk12-floor", [[R, N], [R, R], [R, U], [R, R]]),
    ];
    let stems: Vec<String> = committed_descriptions().into_iter().map(|p| p.0).collect();
    assert_eq!(
        stems,
        pinned.map(|p| p.0.to_string()),
        "a profile was added or removed: pin its routes"
    );
    for (stem, expected) in pinned {
        let enabled = device(stem, &BTreeSet::new());
        assert_eq!(routes(stem, &enabled), expected, "{stem}");
    }
}

/// On the RTX 5080 each usage of the four formats at 4 samples takes the word's image: a texture
/// only read (filled by copies, bitwise the word), only rendered, rendered and viewed in another
/// format (no view list), and one a function would write, which Metal makes and refuses to write.
#[test]
fn every_usage_of_a_word_texture_takes_the_word() {
    let stem = "nvidia-rtx5080-laptop";
    let enabled = device(stem, &BTreeSet::new());
    for pf in FORMATS {
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::RENDER_TARGET,
            TextureUsage::RENDER_TARGET
                | TextureUsage::SHADER_READ
                | TextureUsage::PIXEL_FORMAT_VIEW,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ] {
            assert_eq!(
                route(stem, &enabled, pf, 4, usage),
                Route::Word,
                "{pf:?} {usage:?}"
            );
        }
    }
}

/// A device made with the four formats' attachments withheld (the device test's way of reaching
/// the word on MoltenVK, which renders each) holds every one in its word at 2 and 4 samples.
#[test]
fn a_device_withholding_the_attachments_takes_the_words() {
    let stem = "moltenvk-m3";
    let withheld: BTreeSet<Format> = FORMATS.iter().filter_map(|&pf| direct_format(pf)).collect();
    assert_eq!(withheld.len(), 4);
    let enabled = device(stem, &withheld);
    assert_eq!(routes(stem, &enabled), [[Route::Word; 2]; 4]);
}
