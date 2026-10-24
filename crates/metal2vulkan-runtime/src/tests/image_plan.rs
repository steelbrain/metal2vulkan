//! The texture image mapping, over every pixel format the format table knows and every committed
//! device profile.

use super::super::blit_plan::BufferImageRegion;
use super::super::chroma_pairs::{holds_pairs, CARRIER as PAIRS_CARRIER};
use super::super::copy_plan::{copy_unit, BlitOption, CopyUnit, ImageRegion};
use super::super::driver_defects;
use super::super::image_plan::{
    block_family, check_image_limits, compatible, cube_compatible, holds_depth_in_colour,
    holds_srgb_twin, plan_image, plan_images, sample_count, srgb_twin, DepthImageFacts, ImageFacts,
    ImagePlan, ImageRefusal,
};
use super::super::memory::MemoryClass;
use super::super::pvrtc::{holds_blocks, image_buffer_region, image_region, CARRIERS};
use super::super::sharing::Sharing;
use super::super::texture_descriptor::{
    effective_usage, TextureDescriptor, TextureLayout, TextureRules,
};
use super::super::values::{ResourceOptions, StorageMode};
use super::texture_descriptor::APPLE;
use crate::MetalLevel;
use ash::vk;
use metal2vulkan::device::{
    Device, DeviceDescription, DevicePolicy, FormatUsage, VkExtension, VkFeature, VkProperty,
};
use metal2vulkan::error::{Capability, Feature, Needs, RuntimeDriverDefect};
use metal2vulkan::metal::format::{
    alpha_in_red, alpha_in_red_format, direct_format, is_depth_stencil, srgb_raw_twin,
    storage_image_choice, storage_view_formats, vk_format, FormatEmulation, WordAttachment,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    DriverId, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType,
    ImageUsageFlags, SampleCountFlags,
};
use std::path::PathBuf;

/// Every committed profile's description and compile device (`validation/profiles/`).
pub(super) fn committed() -> Vec<(String, DeviceDescription, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, DeviceDescription, Device)> = std::fs::read_dir(&dir)
        .expect("validation/profiles is readable")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile is readable");
            let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
            let device = Device::new(desc.clone(), DevicePolicy::default()).expect("derives");
            (stem, desc, device)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(out.len() >= 10, "the committed profiles are present");
    out
}

pub(super) fn facts<'a>(desc: &'a DeviceDescription, device: &'a Device) -> ImageFacts<'a> {
    ImageFacts {
        profile: device.profile(),
        formats: &desc.formats,
        storage_multisample: desc.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: DepthImageFacts::of(&desc.driver_facts()),
        linear_attachments: false,
        multisample_depth_twin: false,
    }
}

/// [`facts`] with a multisampled `Depth32Float` texture held in colour
/// (`MultisampleDepthTwinPolicy::Exact`).
fn facts_holding_multisample_depth<'a>(
    desc: &'a DeviceDescription,
    device: &'a Device,
) -> ImageFacts<'a> {
    ImageFacts {
        multisample_depth_twin: true,
        ..facts(desc, device)
    }
}

fn layout(ty: TextureType, pf: PixelFormat, samples: u64, array: u64) -> TextureLayout {
    let (w, h, d) = match ty {
        TextureType::Type1D | TextureType::Type1DArray => (16, 1, 1),
        TextureType::Type3D => (8, 8, 4),
        _ => (8, 8, 1),
    };
    // The descriptor check refuses the placeholder formats; their layout still reaches the plan
    // here, which refuses them too. Every Vulkan sample count is taken: the plan decides what the
    // device makes.
    let checked = TextureDescriptor {
        texture_type: ty,
        width: w,
        height: h,
        depth: d,
        sample_count: samples,
        array_length: array,
        ..TextureDescriptor::default()
    }
    .check(TextureRules {
        sample_counts: SampleCountFlags::from_raw(0x7f),
        level: MetalLevel::NEWEST,
    })
    .expect("a valid descriptor");
    TextureLayout {
        pixel_format: pf,
        ..checked
    }
}

const USAGES: &[TextureUsage] = &[
    TextureUsage::SHADER_READ,
    TextureUsage::SHADER_WRITE,
    TextureUsage::RENDER_TARGET,
    TextureUsage::PIXEL_FORMAT_VIEW,
    TextureUsage::SHADER_ATOMIC,
];

/// Every combination of Metal's usage bits, `Unknown` (empty) included.
fn usage_sets() -> Vec<TextureUsage> {
    (0..1u32 << USAGES.len())
        .map(|mask| {
            USAGES
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .fold(TextureUsage::empty(), |acc, (_, &u)| acc | u)
        })
        .collect()
}

/// The invariants of a plan for a texture of `pf` with `usage`.
fn check_plan(
    name: &str,
    pf: PixelFormat,
    usage: TextureUsage,
    plan: &ImagePlan,
    facts: &ImageFacts<'_>,
) {
    let profile = facts.profile;
    let what = format!("{name}: {pf:?} {usage:?} -> {plan:?}");
    assert!(!plan.usage.is_empty(), "an image of no usage: {what}");
    let unknown = usage.is_empty();
    let usage = effective_usage(usage);
    let mutable = plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
    let extended = plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE);
    // The listed formats start with the image's own, and each is compatible with it and one the
    // device has.
    if !plan.view_formats.is_empty() {
        assert!(mutable, "{what}");
        assert_eq!(plan.view_formats[0], plan.format, "{what}");
        let twin = srgb_twin(pf);
        for &f in &plan.view_formats {
            assert!(compatible(plan.format, f, twin.is_some()), "{what}: {f:?}");
            assert!(
                !facts.optimal(f).is_empty(),
                "{what}: {f:?} is not the device's"
            );
        }
    }
    // A cube-compatible image is square, of six layers or more (VUID-VkImageCreateInfo-flags-08865
    // and -08866).
    if plan.flags.contains(ImageCreateFlags::CUBE_COMPATIBLE) {
        assert_eq!(plan.extent[0], plan.extent[1], "{what}");
        assert!(plan.layers >= 6, "{what}");
    }
    let viewable = |f: Format| {
        f == plan.format
            || (mutable && (plan.view_formats.is_empty() || plan.view_formats.contains(&f)))
    };
    // Each usage is supported by the image's format, or with extended usage by a view's.
    let supports = |usage: FormatUsage| profile.supports_format(plan.format, usage);
    if plan.usage.contains(ImageUsageFlags::SAMPLED) {
        // An 8-bit sRGB format the device cannot sample is sampled through its UNORM twin's
        // view.
        let m = vk_format(pf, &[FormatUsage::SampledImage].into(), profile)
            .or_else(|e| match srgb_raw_twin(pf) {
                Some(twin) => vk_format(twin, &[FormatUsage::SampledImage].into(), profile),
                None => Err(e),
            })
            .expect(&what);
        assert!(viewable(m.format), "{what}: the sampled view {m:?}");
        assert!(supports(FormatUsage::SampledImage) || extended, "{what}");
    }
    if plan.usage.contains(ImageUsageFlags::STORAGE) {
        match storage_image_choice(pf, profile) {
            Ok(c) => {
                let view = if c.alias.is_some() {
                    c.vk_format
                } else {
                    plan.format
                };
                assert!(viewable(view), "{what}: the storage view {c:?}");
            }
            // A format no slot both reads and writes: the written-only binding's views.
            Err(e) => {
                let views = storage_view_formats(pf, profile);
                assert!(!views.is_empty(), "{what}: {e:?}");
                for view in views {
                    assert!(viewable(view), "{what}: the written-only view {view:?}");
                }
            }
        }
        assert!(supports(FormatUsage::StorageImage) || extended, "{what}");
        // The format's own, or the UNORM twin of an 8-bit sRGB format the device holds no image
        // of, which serves its storage alias, or the red image of an `A8Unorm` texture the
        // device holds no `A8_UNORM` image of.
        let own = direct_format(pf);
        let unheld = own.filter(|&f| facts.optimal(f).is_empty());
        let twin = unheld.and(srgb_raw_twin(pf).and_then(direct_format));
        let red = unheld.and(alpha_in_red_format(pf));
        assert!(
            Some(plan.format) == own || Some(plan.format) == twin || Some(plan.format) == red,
            "{what}"
        );
    }
    if plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
        // Or, with extended usage, through the view of the format's word.
        let word = WordAttachment::of(pf, profile).map(|w| w.word_format());
        assert!(
            supports(FormatUsage::ColorAttachment) || (extended && word.is_some_and(&viewable)),
            "{what}"
        );
    }
    // A color attachment is an input attachment exactly where the device's framebuffer fetch
    // reads input attachments.
    assert_eq!(
        plan.usage.contains(ImageUsageFlags::INPUT_ATTACHMENT),
        plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
            && profile.fragment().rasterization_order_color_access,
        "{what}"
    );
    if plan
        .usage
        .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
    {
        assert!(supports(FormatUsage::DepthStencilAttachment), "{what}");
    }
    let optimal = facts.optimal(plan.format);
    if plan.usage.contains(ImageUsageFlags::TRANSFER_SRC) {
        assert!(
            optimal.contains(FormatFeatureFlags2::TRANSFER_SRC),
            "{what}"
        );
    }
    if plan.usage.contains(ImageUsageFlags::TRANSFER_DST) {
        assert!(
            optimal.contains(FormatFeatureFlags2::TRANSFER_DST),
            "{what}"
        );
    }
    // `PixelFormatView` and `Unknown` take any compatible format; a PVRTC texture's carrier
    // has no other view (`pvrtc.rs`).
    // A 4:2:2 texture's carrier likewise (`chroma_pairs.rs`), and an ASTC texture's, whose
    // sampled views present its decoded twin (`astc.rs`), as a PVRTC texture's do, and an ETC2
    // or EAC texture's (`etc.rs`).
    let astc = super::super::astc::decode_of(plan).is_some();
    let etc = super::super::etc::decode_of(plan).is_some();
    let carried = astc || etc || holds_blocks(plan) || holds_pairs(plan);
    if (unknown || usage.contains(TextureUsage::PIXEL_FORMAT_VIEW))
        && !holds_blocks(plan)
        && !holds_pairs(plan)
        && !astc
        && !etc
    {
        assert!(mutable && plan.view_formats.is_empty(), "{what}");
    }
    // What is unavailable was asked for (`Unknown` asks for Metal's effective usage).
    assert!(usage.contains(plan.unavailable), "{what}");
    // Every requested usage is either present or unavailable.
    for (bit, flags) in [
        (TextureUsage::SHADER_READ, ImageUsageFlags::SAMPLED),
        (TextureUsage::SHADER_WRITE, ImageUsageFlags::STORAGE),
        (
            TextureUsage::RENDER_TARGET,
            ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
    ] {
        // An ASTC, PVRTC, ETC2, EAC or 4:2:2 carrier's reads are its decoded twin's.
        if carried && bit == TextureUsage::SHADER_READ {
            assert!(!plan.unavailable.contains(bit), "{what}");
            continue;
        }
        if usage.contains(bit) {
            assert!(
                plan.usage.intersects(flags) != plan.unavailable.contains(bit),
                "{what}: {bit:?}"
            );
        }
    }
    // The sRGB twin's sampled view, where the texture is sampled and the twin's representation is
    // compatible, is viewable without `PixelFormatView` (Metal makes it).
    if let Some(twin) = srgb_twin(pf) {
        if plan.usage.contains(ImageUsageFlags::SAMPLED) {
            if let Ok(m) = vk_format(twin, &[FormatUsage::SampledImage].into(), profile) {
                if compatible(plan.format, m.format, true) {
                    assert!(viewable(m.format), "{what}: the twin's view {m:?}");
                }
            }
        }
    }
}

/// Every format and usage set plans or is refused typed on every profile; Apple's private 0x8
/// alone too, which a texture keeps as a usage other than `Unknown`.
#[test]
fn every_format_and_usage_plans_or_refuses_typed_on_every_profile() {
    let mut planned = 0;
    let mut refused = 0;
    let dropped = TextureUsage::try_from(TextureUsage::DROPPED).unwrap();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for &pf in PixelFormat::ALL {
            let ty = TextureType::Type2D;
            for usage in usage_sets().into_iter().chain([dropped]) {
                let l = layout(ty, pf, 1, 1);
                match plan_image(&l, usage, false, &facts) {
                    Ok(plan) => {
                        planned += 1;
                        check_plan(&name, pf, usage, &plan, &facts);
                    }
                    Err(ImageRefusal::NotAFormat) => {
                        refused += 1;
                        assert!(
                            matches!(pf, PixelFormat::Invalid | PixelFormat::Unspecialized),
                            "{name}: {pf:?}"
                        );
                    }
                    Err(ImageRefusal::Unsupported(u)) => {
                        refused += 1;
                        // A format with no representation the device exposes.
                        assert_eq!(u.feature, Feature::Format, "{name}: {pf:?} {usage:?}");
                    }
                    Err(other) => panic!("{name}: {pf:?} {usage:?}: {other:?}"),
                }
            }
        }
    }
    assert!(
        planned > 10_000 && refused > 0,
        "{planned} planned, {refused} refused"
    );
}

#[test]
fn a_usage_the_device_lacks_is_unavailable_not_a_refusal() {
    // An 8-bit sRGB texture is a storage image itself where the device stores and declares its
    // sRGB format, through its UNORM alias where the device stores the UNORM format (the image's
    // own format where it holds the texture in that twin, unable to sample the sRGB one), and
    // otherwise its write usage is unavailable.
    let (mut own, mut alias, mut lacking) = (0, 0, 0);
    for (pf, srgb, unorm) in [
        (
            PixelFormat::RGBA8UnormSRGB,
            Format::R8G8B8A8_SRGB,
            Format::R8G8B8A8_UNORM,
        ),
        (
            PixelFormat::RG8UnormSRGB,
            Format::R8G8_SRGB,
            Format::R8G8_UNORM,
        ),
        (PixelFormat::R8UnormSRGB, Format::R8_SRGB, Format::R8_UNORM),
    ] {
        for (name, desc, device) in committed() {
            let name = format!("{name} {pf:?}");
            let facts = facts(&desc, &device);
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let plan = plan_image(&l, usage, false, &facts).expect(&name);
            assert!(plan.format == srgb || plan.format == unorm, "{name}");
            match storage_image_choice(pf, device.profile()) {
                Ok(choice) if choice.alias.is_some() => {
                    alias += 1;
                    assert!(plan.usage.contains(ImageUsageFlags::STORAGE), "{name}");
                    assert_eq!(choice.vk_format, unorm, "{name}");
                    assert!(
                        plan.format == unorm || plan.view_formats.contains(&unorm),
                        "{name}"
                    );
                    let extended = !device
                        .profile()
                        .supports_format(plan.format, FormatUsage::StorageImage);
                    assert_eq!(
                        plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE),
                        extended,
                        "{name}"
                    );
                    assert!(plan.unavailable.is_empty(), "{name}");
                }
                Ok(_) => {
                    own += 1;
                    assert!(plan.usage.contains(ImageUsageFlags::STORAGE), "{name}");
                    assert!(
                        !plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE),
                        "{name}"
                    );
                    assert!(plan.unavailable.is_empty(), "{name}");
                }
                Err(_) if storage_view_formats(pf, device.profile()).is_empty() => {
                    lacking += 1;
                    assert!(!plan.usage.contains(ImageUsageFlags::STORAGE), "{name}");
                    assert_eq!(plan.unavailable, TextureUsage::SHADER_WRITE, "{name}");
                }
                Err(_) => {
                    own += 1;
                    assert!(plan.usage.contains(ImageUsageFlags::STORAGE), "{name}");
                    assert!(plan.unavailable.is_empty(), "{name}");
                }
            }
        }
    }
    assert!(
        alias > 0 && own + alias + lacking > 30,
        "{own} {alias} {lacking}"
    );
}

/// A texture of a format no slot both reads and writes on the device, whose written-only
/// binding's view the device stores, takes the storage usage and that view, so a slot that only
/// writes it binds: on ANV, which reads no storage image without a format, a format SPIR-V names
/// no image format for and no alias view serves (the packed 16-bit formats among them), whose own
/// format ANV stores without one for writes alone. Where no written-only view serves either, the
/// write stays unavailable. `BGRA8Unorm_sRGB`, whose `R8G8B8A8_UNORM` view serves both accesses
/// (`StorageImageClass::SrgbEncodeBgra`), is not such a format on any committed profile.
#[test]
fn a_written_only_view_serves_where_no_slot_both_reads_and_writes() {
    let mut written_only = Vec::new();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        for &pf in PixelFormat::ALL {
            let what = format!("{name} {pf:?}");
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let Ok(plan) = plan_image(&l, usage, false, &facts) else {
                continue;
            };
            if storage_image_choice(pf, profile).is_ok() || is_depth_stencil(pf) {
                continue;
            }
            let views = storage_view_formats(pf, profile);
            if views.is_empty() {
                assert!(!plan.usage.contains(ImageUsageFlags::STORAGE), "{what}");
                assert!(
                    plan.unavailable.contains(TextureUsage::SHADER_WRITE),
                    "{what}"
                );
                continue;
            }
            assert!(plan.usage.contains(ImageUsageFlags::STORAGE), "{what}");
            for view in &views {
                assert!(
                    plan.format == *view || plan.view_formats.contains(view),
                    "{what}: {plan:?}"
                );
            }
            assert!(
                !plan.unavailable.contains(TextureUsage::SHADER_WRITE),
                "{what}"
            );
            assert_ne!(pf, PixelFormat::BGRA8UnormSRGB, "{what}");
            written_only.push(what);
        }
    }
    assert!(
        written_only.iter().any(|n| n.contains("-anv ")),
        "an ANV profile takes a written-only view: {written_only:?}"
    );
}

/// An 8-bit sRGB format the device samples only through its UNORM twin is held in the twin's
/// format, sampled, with no usage the device lacks for the sRGB format: lavapipe's `R8Unorm_sRGB`
/// and ANV's and NVIDIA's `RG8Unorm_sRGB`. A device that samples the format keeps its own image.
/// A render target the device renders through its word's view (`WordAttachment`) lists that view.
#[test]
fn an_srgb_format_the_device_cannot_sample_is_held_as_its_unorm_twin() {
    let mut held = Vec::new();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = facts.profile;
        for pf in [
            PixelFormat::R8UnormSRGB,
            PixelFormat::RG8UnormSRGB,
            PixelFormat::RGBA8UnormSRGB,
            PixelFormat::BGRA8UnormSRGB,
        ] {
            let what = format!("{name} {pf:?}");
            let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), profile).is_ok();
            let twin = srgb_raw_twin(pf).expect(&what);
            let twin_sampled = vk_format(twin, &[FormatUsage::SampledImage].into(), profile);
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let plan = match plan_image(&l, TextureUsage::SHADER_READ, false, &facts) {
                Ok(plan) => plan,
                Err(e) => {
                    assert!(!sampled && twin_sampled.is_err(), "{what}: {e:?}");
                    continue;
                }
            };
            assert_eq!(holds_srgb_twin(&plan), !sampled, "{what}: {plan:?}");
            if sampled {
                continue;
            }
            held.push(what.clone());
            let m = twin_sampled.expect(&what);
            assert_eq!(plan.format, m.format, "{what}");
            assert_eq!(Some(m.format), direct_format(twin), "{what}");
            assert!(plan.usage.contains(ImageUsageFlags::SAMPLED), "{what}");
            assert!(plan.unavailable.is_empty(), "{what}");
            check_plan(&name, pf, TextureUsage::SHADER_READ, &plan, &facts);
            // A render target the device lacks for the format stays unavailable, unless the
            // device renders the view of its word, which the twin's image lists.
            let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            let plan = plan_image(&l, usage, false, &facts).expect(&what);
            assert!(plan.usage.contains(ImageUsageFlags::SAMPLED), "{what}");
            let word = WordAttachment::of(pf, profile).map(|w| w.word_format());
            if let Some(word) = word {
                assert!(plan.unavailable.is_empty(), "{what}: {plan:?}");
                assert!(plan.view_formats.contains(&word), "{what}: {plan:?}");
            } else if !profile.supports_format(
                direct_format(pf).expect(&what),
                FormatUsage::ColorAttachment,
            ) {
                assert_eq!(plan.unavailable, TextureUsage::RENDER_TARGET, "{what}");
            }
        }
    }
    for stem in ["lavapipe", "gate-lavapipe"] {
        assert!(
            held.contains(&format!("{stem} R8UnormSRGB")),
            "{stem}: {held:?}"
        );
    }
}

/// An `A8Unorm` texture on a device that holds no `A8_UNORM` image is an `R8_UNORM` image holding
/// Metal's alpha in red (`FormatEmulation::AlphaInRed`), sampled through `(Zero, Zero, Zero, R)`,
/// stored and rendered where the device does those with `R8_UNORM`; where the device holds
/// `A8_UNORM`, that format. A pipeline's attachment agrees (`alpha_in_red`).
#[test]
fn an_a8_texture_without_a8_unorm_holds_its_alpha_in_red() {
    let usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    let mut held = Vec::new();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = facts.profile;
        let what = name.clone();
        let l = layout(TextureType::Type2D, PixelFormat::A8Unorm, 1, 1);
        let plan = plan_image(&l, usage, false, &facts).expect(&what);
        check_plan(&name, PixelFormat::A8Unorm, usage, &plan, &facts);
        if !facts.optimal(Format::A8_UNORM).is_empty() {
            assert_eq!(plan.format, Format::A8_UNORM, "{what}: {plan:?}");
            assert!(!alpha_in_red(PixelFormat::A8Unorm, profile), "{what}");
            continue;
        }
        held.push(name.clone());
        assert_eq!(plan.format, Format::R8_UNORM, "{what}: {plan:?}");
        assert_eq!(
            plan.mapping.emulation,
            FormatEmulation::AlphaInRed,
            "{what}"
        );
        assert_eq!(
            plan.mapping.components.a,
            metal2vulkan::vk::ComponentSwizzle::R,
            "{what}"
        );
        assert!(alpha_in_red(PixelFormat::A8Unorm, profile), "{what}");
        for (bit, flags, need) in [
            (
                TextureUsage::SHADER_READ,
                ImageUsageFlags::SAMPLED,
                FormatUsage::SampledImage,
            ),
            (
                TextureUsage::SHADER_WRITE,
                ImageUsageFlags::STORAGE,
                FormatUsage::StorageImage,
            ),
            (
                TextureUsage::RENDER_TARGET,
                ImageUsageFlags::COLOR_ATTACHMENT,
                FormatUsage::ColorAttachment,
            ),
        ] {
            let has = profile.supports_format(Format::R8_UNORM, need);
            assert_eq!(plan.usage.contains(flags), has, "{what}: {bit:?} {plan:?}");
            assert_eq!(plan.unavailable.contains(bit), !has, "{what}: {bit:?}");
        }
    }
    for stem in [
        "vk12-floor",
        "lavapipe",
        "amd-rdna3-radv",
        "nvidia-pascal-linux",
    ] {
        assert!(held.contains(&stem.to_owned()), "{stem}: {held:?}");
    }
}

#[test]
fn srgb_twins_share_one_mutable_image() {
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let l = layout(TextureType::Type2D, PixelFormat::RGBA8Unorm, 1, 1);
        let plan = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
        assert_eq!(plan.format, Format::R8G8B8A8_UNORM, "{name}");
        assert!(
            plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT),
            "{name}"
        );
        assert_eq!(
            plan.view_formats,
            [Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB],
            "{name}"
        );
        // A format with no twin and one view is not mutable.
        let l = layout(TextureType::Type2D, PixelFormat::R32Float, 1, 1);
        let plan = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
        assert!(plan.view_formats.is_empty(), "{name}");
        assert!(
            !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT),
            "{name}"
        );
        assert_eq!(
            plan.usage,
            ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::TRANSFER_DST,
            "{name}"
        );
    }
}

/// A heap's texture without GPU-optimized contents takes the image of one of `PixelFormatView`
/// usage (`MUTABLE_FORMAT`, no list of formats), as Metal lays out a color texture with the
/// property off; with it on, the image of its usage, an sRGB twin's list kept. A device's own
/// texture takes the image of its usage either way (`TextureDescriptor::own_image_usage`): its
/// layout shows nowhere, and the image keeps the driver's compression. Its own usage stays the
/// descriptor's, `Unknown` taken as Metal takes it.
#[test]
fn textures_without_optimized_contents_take_pixel_format_view_images() {
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::BGRA8Unorm,
            PixelFormat::R32Float,
        ] {
            for usage in [
                TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            ] {
                let l = layout(TextureType::Type2D, pf, 1, 1);
                let on = TextureDescriptor {
                    pixel_format: pf,
                    usage,
                    ..TextureDescriptor::default()
                };
                let off = TextureDescriptor {
                    allow_gpu_optimized_contents: false,
                    ..on
                };
                assert_eq!(on.image_usage(), usage, "{name}");
                assert_eq!(
                    off.image_usage(),
                    usage | TextureUsage::PIXEL_FORMAT_VIEW,
                    "{name}"
                );
                let viewed = plan_image(&l, usage | TextureUsage::PIXEL_FORMAT_VIEW, false, &facts)
                    .expect(&name);
                let plan = plan_image(&l, off.image_usage(), false, &facts).expect(&name);
                assert_eq!(plan, viewed, "{name} {pf:?}");
                assert!(
                    plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT),
                    "{name} {pf:?}"
                );
                assert!(plan.view_formats.is_empty(), "{name} {pf:?}");
                let optimized = plan_image(&l, on.image_usage(), false, &facts).expect(&name);
                assert_eq!(
                    optimized,
                    plan_image(&l, usage, false, &facts).expect(&name),
                    "{name} {pf:?}"
                );
                assert_eq!(off.own_image_usage(), usage, "{name}");
                assert_eq!(on.own_image_usage(), usage, "{name}");
                let own = plan_image(&l, off.own_image_usage(), false, &facts).expect(&name);
                assert_eq!(own, optimized, "{name} {pf:?}");
            }
        }
    }
    let unknown = TextureDescriptor {
        usage: TextureUsage::empty(),
        allow_gpu_optimized_contents: false,
        ..TextureDescriptor::default()
    };
    assert_eq!(
        unknown.image_usage(),
        effective_usage(TextureUsage::empty())
    );
}

/// A render target's sRGB twin is listed as an attachment view only where the device has the
/// twin's format as an attachment: `RG8Unorm`'s `R8G8_SRGB` is on some committed profiles and not
/// on others (ANV makes no image whose format list names it).
#[test]
fn twin_attachment_views_the_device_lacks_are_left_out() {
    let (mut with, mut without) = (0, 0);
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let l = layout(TextureType::Type2D, PixelFormat::RG8Unorm, 1, 1);
        let Ok(plan) = plan_image(&l, TextureUsage::RENDER_TARGET, false, &facts) else {
            continue;
        };
        if device
            .profile()
            .supports_format(Format::R8G8_SRGB, FormatUsage::ColorAttachment)
        {
            with += 1;
            assert_eq!(
                plan.view_formats,
                [Format::R8G8_UNORM, Format::R8G8_SRGB],
                "{name}"
            );
        } else {
            without += 1;
            assert!(plan.view_formats.is_empty(), "{name}: {plan:?}");
            assert!(
                !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT),
                "{name}"
            );
        }
    }
    assert!(
        with > 0 && without > 0,
        "{with} with the twin, {without} without"
    );
}

/// A packed 16-bit render target a device samples but does not render renders through the view of
/// its word (`metal::format::WordAttachment`): its image keeps the format, mutable with extended
/// usage, takes the attachment usage and lists `R16_UINT`; where the device renders the format the
/// image takes neither, and one it does not hold is not planned so.
#[test]
fn packed_render_targets_the_device_cannot_render_take_their_word_view() {
    let (mut with, mut without) = (0, 0);
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for pf in [
            PixelFormat::A1BGR5Unorm,
            PixelFormat::ABGR4Unorm,
            PixelFormat::B5G6R5Unorm,
            PixelFormat::BGR5A1Unorm,
        ] {
            let l = layout(TextureType::Type2D, pf, 1, 1);
            for usage in [
                TextureUsage::RENDER_TARGET,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ] {
                let what = format!("{name}: {pf:?} {usage:?}");
                let Ok(plan) = plan_image(&l, usage, false, &facts) else {
                    continue;
                };
                let word = WordAttachment::of(pf, device.profile());
                match word {
                    Some(word) => {
                        with += 1;
                        assert_eq!(Some(plan.format), direct_format(pf), "{what}");
                        assert!(
                            plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT),
                            "{what}"
                        );
                        assert!(
                            plan.flags.contains(
                                ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE
                            ),
                            "{what}"
                        );
                        assert!(
                            plan.view_formats.contains(&word.word_format()),
                            "{what}: {plan:?}"
                        );
                        assert!(
                            !plan.unavailable.contains(TextureUsage::RENDER_TARGET),
                            "{what}"
                        );
                    }
                    None => {
                        without += 1;
                        assert!(!plan.view_formats.contains(&Format::R16_UINT), "{what}");
                    }
                }
            }
        }
    }
    assert!(
        with > 0 && without > 0,
        "{with} with the word, {without} without"
    );
}

/// A `ulong` texture (`RG32Uint`) made for atomics takes the `R64_UINT` view its atomics update
/// on every profile with that view's storage atomics, and only there; without `ShaderAtomic` it
/// keeps its one format.
#[test]
fn ulong_textures_made_for_atomics_take_the_r64_view() {
    let mut with = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let l = layout(TextureType::Type2D, PixelFormat::RG32Uint, 1, 1);
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let atomic = plan_image(&l, usage | TextureUsage::SHADER_ATOMIC, false, &facts);
        let plain = plan_image(&l, usage, false, &facts).expect(&name);
        assert!(!plain.view_formats.contains(&Format::R64_UINT), "{name}");
        let Ok(atomic) = atomic else { continue };
        let r64 = device
            .profile()
            .supports_format(Format::R64_UINT, FormatUsage::StorageImageAtomic);
        let storage = atomic.usage.contains(ImageUsageFlags::STORAGE);
        assert_eq!(
            atomic.view_formats.contains(&Format::R64_UINT),
            r64 && storage,
            "{name}"
        );
        if r64 && storage {
            with += 1;
            assert_eq!(atomic.format, Format::R32G32_UINT, "{name}");
            assert!(
                atomic.flags.contains(ImageCreateFlags::MUTABLE_FORMAT),
                "{name}"
            );
        }
    }
    assert!(
        with > 0,
        "no committed profile has R64_UINT storage atomics"
    );
}

#[test]
fn texture_types_take_their_image_types_and_cube_flags() {
    let (name, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&desc, &device);
    let usage = TextureUsage::SHADER_READ;
    let plan = |ty, samples, array| {
        let l = layout(ty, PixelFormat::RGBA8Unorm, samples, array);
        (l, plan_image(&l, usage, false, &facts).expect(&name))
    };
    let cube = ImageCreateFlags::CUBE_COMPATIBLE;
    let (l, p) = plan(TextureType::Cube, 1, 1);
    assert!(cube_compatible(&l) && p.flags.contains(cube));
    assert_eq!((p.image_type, p.layers), (ImageType::TYPE_2D, 6));
    let (_, p) = plan(TextureType::CubeArray, 1, 2);
    assert!(p.flags.contains(cube) && p.layers == 12);
    // A square 2D array of six layers is viewed as a cube by Metal; of five it is not.
    let (_, p) = plan(TextureType::Type2DArray, 1, 6);
    assert!(p.flags.contains(cube));
    let (_, p) = plan(TextureType::Type2DArray, 1, 5);
    assert!(!p.flags.contains(cube));
    let (_, p) = plan(TextureType::Type2D, 1, 1);
    assert!(!p.flags.contains(cube));
    let (_, p) = plan(TextureType::Type3D, 1, 1);
    assert_eq!(
        (p.image_type, p.extent, p.layers),
        (ImageType::TYPE_3D, [8, 8, 4], 1)
    );
    // A 3D render target's planes are 2D views of it, so its image is 2D-array compatible; a
    // texture nothing renders into is not.
    let compatible = ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE;
    assert!(!p.flags.contains(compatible));
    let l = layout(TextureType::Type3D, PixelFormat::RGBA8Unorm, 1, 1);
    let rendered = usage | TextureUsage::RENDER_TARGET;
    let p = plan_image(&l, rendered, false, &facts).expect(&name);
    assert!(p.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) && p.flags.contains(compatible));
    let (_, p) = plan(TextureType::Type1DArray, 1, 3);
    assert_eq!((p.image_type, p.layers), (ImageType::TYPE_1D, 3));
    let (_, p) = plan(TextureType::Type2DMultisample, 4, 1);
    assert_eq!(p.samples, SampleCountFlags::COUNT_4);
    assert!(!p.flags.contains(cube));
}

#[test]
fn depth_render_targets_are_depth_attachments() {
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let l = layout(TextureType::Type2D, PixelFormat::Depth32Float, 1, 1);
        let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let plan = plan_image(&l, usage, false, &facts).expect(&name);
        assert_eq!(plan.format, Format::D32_SFLOAT, "{name}");
        assert_eq!(plan.mapping.aspects, ImageAspectFlags::DEPTH, "{name}");
        let profile = device.profile();
        for (need, flag, metal) in [
            (
                FormatUsage::DepthStencilAttachment,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                TextureUsage::RENDER_TARGET,
            ),
            (
                FormatUsage::SampledImage,
                ImageUsageFlags::SAMPLED,
                TextureUsage::SHADER_READ,
            ),
        ] {
            let has = profile.supports_format(Format::D32_SFLOAT, need);
            assert_eq!(plan.usage.contains(flag), has, "{name}: {need:?}");
            assert_eq!(plan.unavailable.contains(metal), !has, "{name}: {need:?}");
        }
        assert!(
            !plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT),
            "{name}"
        );
        // Metal has no shader writes to depth textures.
        let plan = plan_image(&l, TextureUsage::SHADER_WRITE, false, &facts).expect(&name);
        assert_eq!(plan.unavailable, TextureUsage::SHADER_WRITE, "{name}");
    }
}

/// A multisampled depth or depth/stencil render target, memoryless too, is made for custom sample
/// positions on every committed profile that takes them at its count with
/// `variableSampleLocations`, so a pass with positions may attach it; a single-sampled one, a
/// stencil-only one, a color one and a depth texture nothing renders into are not.
#[test]
fn multisampled_depth_targets_take_custom_sample_positions() {
    use PixelFormat as P;
    let flag = ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT;
    let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let mut flagged = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let properties = &device.profile().enabled().properties;
        for samples in [1u64, 2, 4, 8] {
            let located = samples > 1
                && properties.contains(&VkProperty::VariableSampleLocations)
                && properties.contains(&VkProperty::SampleLocationSampleCount(samples as u32));
            let ty = match samples {
                1 => TextureType::Type2D,
                _ => TextureType::Type2DMultisample,
            };
            for (pf, depth) in [
                (P::Depth32Float, true),
                (P::Depth16Unorm, true),
                (P::Depth32FloatStencil8, true),
                (P::Stencil8, false),
                (P::RGBA8Unorm, false),
            ] {
                for (usage, memoryless) in [(target, false), (TextureUsage::RENDER_TARGET, true)] {
                    let l = layout(ty, pf, samples, 1);
                    let Ok(plan) = plan_image(&l, usage, memoryless, &facts) else {
                        continue;
                    };
                    let attached = plan
                        .usage
                        .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
                    // A stencil-only format a device holds in a combined image has a depth
                    // aspect; `S8_UINT` has none, and Vulkan flags no such image.
                    let has_depth = plan.format.block().is_some_and(|b| b.depth.is_some());
                    let expected = located && attached && has_depth;
                    assert!(!expected || depth || pf == P::Stencil8, "{name}: {pf:?}");
                    assert_eq!(
                        plan.flags.contains(flag),
                        expected,
                        "{name}: {pf:?} x{samples} memoryless {memoryless}"
                    );
                    flagged += usize::from(expected);
                }
            }
            let l = layout(ty, P::Depth32Float, samples, 1);
            let read = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
            assert!(
                !read.flags.contains(flag),
                "{name}: read-only depth x{samples}"
            );
        }
    }
    assert!(
        flagged > 0,
        "some committed profile takes custom positions with depth"
    );
}

/// A `Depth32Float` texture the program only reads or samples holds its texels in an `R32_SFLOAT`
/// image beside a `D32_SFLOAT` twin of its shape on every committed profile whose depth image its
/// driver is not measured to keep every `float` in: without `VK_EXT_depth_range_unrestricted` and
/// the driver's fact that its depth images keep values outside `[0, 1]`, the Xe-LP ANV ones among
/// them, or without its fact that they keep denormals, RADV's among them; and in its depth
/// image, as `plan_image` plans it, elsewhere: NVIDIA's 615 and lavapipe, with the extension, and
/// MoltenVK without it. A render target (`Unknown` usage counts as one), a memoryless or
/// multisampled texture and another depth format have no twin.
#[test]
fn depth_twins_hold_the_texels_in_colour_where_a_depth_image_may_not() {
    use PixelFormat as P;
    let read = TextureUsage::SHADER_READ;
    let mut twinned = Vec::new();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let unrestricted = device
            .profile()
            .enabled()
            .extensions
            .contains(&VkExtension::ExtDepthRangeUnrestricted);
        let kept = facts.depth_images;
        let exact = (unrestricted || kept.outside_unit_range) && kept.denormals;
        assert_eq!(facts.depth_loss().is_none(), exact, "{name}");
        for (ty, array) in [
            (TextureType::Type2D, 1),
            (TextureType::Type2DArray, 3),
            (TextureType::Cube, 1),
        ] {
            let l = layout(ty, P::Depth32Float, 1, array);
            let (plan, twin) = plan_images(&l, read, false, &facts).expect(&name);
            let Some(twin) = twin else {
                assert!(exact, "{name}: {ty:?}");
                assert_eq!(plan, plan_image(&l, read, false, &facts).unwrap(), "{name}");
                continue;
            };
            assert!(!exact, "{name}: {ty:?}");
            assert_eq!(
                (plan.pixel_format, plan.format, plan.mapping.aspects),
                (P::Depth32Float, Format::R32_SFLOAT, ImageAspectFlags::COLOR),
                "{name}: {ty:?}"
            );
            let copies = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
            assert!(
                plan.usage.contains(ImageUsageFlags::SAMPLED | copies),
                "{name}: {ty:?}"
            );
            assert_eq!(
                (twin.pixel_format, twin.format, twin.mapping.aspects),
                (P::Depth32Float, Format::D32_SFLOAT, ImageAspectFlags::DEPTH),
                "{name}: {ty:?}"
            );
            assert_eq!(
                (
                    twin.image_type,
                    twin.flags,
                    twin.extent,
                    twin.levels,
                    twin.layers,
                    twin.samples
                ),
                (
                    plan.image_type,
                    plan.flags,
                    plan.extent,
                    plan.levels,
                    plan.layers,
                    plan.samples
                ),
                "{name}: {ty:?}"
            );
            assert!(
                twin.usage
                    .contains(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST),
                "{name}: {ty:?}"
            );
            if ty == TextureType::Type2D {
                twinned.push(name.clone());
            }
        }
        let flat = layout(TextureType::Type2D, P::Depth32Float, 1, 1);
        for (what, l, usage, memoryless) in [
            (
                "a render target",
                flat,
                read | TextureUsage::RENDER_TARGET,
                false,
            ),
            ("Unknown usage", flat, TextureUsage::empty(), false),
            ("memoryless", flat, TextureUsage::RENDER_TARGET, true),
            (
                "multisampled",
                layout(TextureType::Type2DMultisample, P::Depth32Float, 4, 1),
                read,
                false,
            ),
            (
                "Depth16Unorm",
                layout(TextureType::Type2D, P::Depth16Unorm, 1, 1),
                read,
                false,
            ),
        ] {
            if let Ok((_, twin)) = plan_images(&l, usage, memoryless, &facts) {
                assert_eq!(twin, None, "{name}: {what}");
            }
        }
    }
    // Lavapipe 26.2.3 (`lavapipe`) is no measured identity, so it takes the twin too.
    for stem in [
        "intel-irisxe-rplp-anv",
        "amd-780m-radv",
        "amd-rdna3-radv",
        "lavapipe",
    ] {
        assert!(twinned.iter().any(|n| n == stem), "{stem}: {twinned:?}");
    }
    // Drivers measured to keep every `float` in a depth image: NVIDIA's and lavapipe with the
    // extension, MoltenVK without it.
    for stem in [
        "nvidia-rtx5080-laptop",
        "gate-lavapipe",
        "moltenvk-m3",
        "moltenvk-m4",
    ] {
        assert!(!twinned.iter().any(|n| n == stem), "{stem}: {twinned:?}");
    }
}

/// A memoryless texture is a transient attachment, on lavapipe (whose framebuffer fetch reads
/// input attachments) an input attachment too.
#[test]
fn memoryless_textures_are_transient_attachments() {
    let (name, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&desc, &device);
    let l = layout(TextureType::Type2D, PixelFormat::BGRA8Unorm, 1, 1);
    let plan = plan_image(&l, TextureUsage::RENDER_TARGET, true, &facts).expect(&name);
    assert_eq!(
        plan.usage,
        ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::INPUT_ATTACHMENT
            | ImageUsageFlags::TRANSIENT_ATTACHMENT
    );
    assert!(plan.unavailable.is_empty());
    let plan = plan_image(
        &l,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        true,
        &facts,
    )
    .expect(&name);
    assert_eq!(plan.unavailable, TextureUsage::SHADER_READ);
    // A memoryless texture of a format the device cannot render to is refused, naming it.
    let l = layout(TextureType::Type2D, PixelFormat::BC1RGBA, 1, 1);
    match plan_image(&l, TextureUsage::RENDER_TARGET, true, &facts) {
        Err(ImageRefusal::Unsupported(u)) => assert_eq!(u.feature, Feature::Format),
        other => panic!("{other:?}"),
    }
}

/// A memoryless texture's image takes lazily allocated memory and a stored texture's
/// device-local memory, on every committed profile, but where the driver crashes on a 2D view of
/// a 3D image in lazily allocated memory (`RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews`:
/// MoltenVK's profiles): there a memoryless 3D texture, 2D-array compatible for its depth planes'
/// views, takes device-local memory; and where a view of some of an image's levels or layers there
/// has no texture (`RuntimeDriverDefect::LazilyAllocatedSubresourceViews`: MoltenVK's profiles), so
/// does one of more than one level or layer (six per cube), mipmapped or an array. A texture of
/// one level and layer, whose view is the texture itself, keeps lazily allocated memory.
#[test]
fn a_memoryless_texture_takes_device_memory_where_its_views_fail() {
    let mut crashing = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let defects = driver_defects::of(&desc);
        let crashes = desc.driver_id == Some(DriverId::MOLTENVK);
        assert_eq!(
            defects.contains(&RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews),
            crashes,
            "{name}"
        );
        assert_eq!(
            defects.contains(&RuntimeDriverDefect::LazilyAllocatedSubresourceViews),
            crashes,
            "{name}"
        );
        for (ty, array) in [
            (TextureType::Type1D, 1),
            (TextureType::Type1DArray, 2),
            (TextureType::Type2D, 1),
            (TextureType::Type2DArray, 2),
            (TextureType::Cube, 1),
            (TextureType::CubeArray, 2),
            (TextureType::Type3D, 1),
        ] {
            for levels in [1, 2] {
                // Metal makes no mipmapped 1D texture.
                if levels > 1 && matches!(ty, TextureType::Type1D | TextureType::Type1DArray) {
                    continue;
                }
                let l = TextureLayout {
                    levels,
                    ..layout(ty, PixelFormat::RGBA8Unorm, 1, array)
                };
                let plan = plan_image(&l, TextureUsage::RENDER_TARGET, true, &facts)
                    .unwrap_or_else(|e| panic!("{name}: {ty:?}: {e:?}"));
                let viewed = plan
                    .flags
                    .contains(ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE);
                assert_eq!(viewed, ty == TextureType::Type3D, "{name}: {ty:?}");
                let parts = plan.layers > 1 || plan.levels > 1;
                assert_eq!(
                    parts,
                    levels > 1 || array > 1 || matches!(ty, TextureType::Cube),
                    "{name}: {ty:?} of {levels} levels"
                );
                let want = if crashes && (viewed || parts) {
                    crashing += 1;
                    MemoryClass::Private
                } else {
                    MemoryClass::Memoryless
                };
                let class = plan.memory_class(StorageMode::Memoryless, &defects);
                assert_eq!(class, want, "{name}: {ty:?} of {levels} levels");
                for storage in [StorageMode::Private, StorageMode::Shared] {
                    let class = plan.memory_class(storage, &defects);
                    assert_eq!(class, MemoryClass::Private, "{name}: {ty:?}");
                }
            }
        }
    }
    assert!(crashing >= 2, "MoltenVK's M3 Max and M4 profiles");
}

/// A multisample array plans wherever the device has `multisampleArrayImage` or no portability
/// subset, and is refused, naming the feature, where the subset withholds it, memoryless or not.
#[test]
fn multisample_arrays_need_the_portability_subsets_feature() {
    let l = layout(
        TextureType::Type2DMultisampleArray,
        PixelFormat::RGBA8Unorm,
        4,
        2,
    );
    let usage = TextureUsage::RENDER_TARGET;
    for (name, mut desc, device) in committed() {
        for memoryless in [false, true] {
            let plan = plan_image(&l, usage, memoryless, &facts(&desc, &device)).expect(&name);
            assert_eq!(
                (plan.samples, plan.layers),
                (SampleCountFlags::COUNT_4, 2),
                "{name}"
            );
        }
        if !desc
            .extensions
            .contains(&metal2vulkan::device::VkExtension::KhrPortabilitySubset)
        {
            continue;
        }
        assert!(
            desc.features.remove(&VkFeature::MultisampleArrayImage),
            "{name}"
        );
        let withheld = Device::new(desc.clone(), DevicePolicy::default()).expect(&name);
        for memoryless in [false, true] {
            match plan_image(&l, usage, memoryless, &facts(&desc, &withheld)) {
                Err(ImageRefusal::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::TextureSampleCount, "{name}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Feature(
                            VkFeature::MultisampleArrayImage
                        )]]),
                        "{name}"
                    );
                }
                other => panic!("{name}: {other:?}"),
            }
        }
    }
}

#[test]
fn texture_buffers_have_no_image() {
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let facts = facts(&desc, &device);
    let l = TextureDescriptor::texture_buffer(
        PixelFormat::R32Float,
        64,
        ResourceOptions::shared(),
        TextureUsage::SHADER_READ,
    )
    .check(APPLE)
    .unwrap();
    assert_eq!(
        plan_image(&l, TextureUsage::SHADER_READ, false, &facts),
        Err(ImageRefusal::TextureBuffer)
    );
}

#[test]
fn sample_counts_are_vulkans_powers_of_two() {
    assert_eq!(sample_count(1), Some(SampleCountFlags::COUNT_1));
    assert_eq!(sample_count(8), Some(SampleCountFlags::COUNT_8));
    assert_eq!(sample_count(64), Some(SampleCountFlags::COUNT_64));
    assert_eq!(sample_count(3), None);
    assert_eq!(sample_count(128), None);
}

#[test]
fn compatibility_follows_vulkans_classes() {
    type F = Format;
    assert!(compatible(F::R8G8B8A8_UNORM, F::R32_UINT, false));
    assert!(compatible(
        F::R8G8B8A8_UNORM,
        F::B10G11R11_UFLOAT_PACK32,
        false
    ));
    assert!(!compatible(F::R8G8B8A8_UNORM, F::R16_UINT, false));
    assert!(!compatible(F::D32_SFLOAT, F::R32_SFLOAT, false));
    assert!(!compatible(F::D32_SFLOAT_S8_UINT, F::D32_SFLOAT, false));
    assert!(compatible(
        F::BC1_RGBA_UNORM_BLOCK,
        F::BC1_RGBA_SRGB_BLOCK,
        true
    ));
    assert!(!compatible(
        F::BC1_RGBA_UNORM_BLOCK,
        F::BC4_UNORM_BLOCK,
        false
    ));
    assert!(!compatible(F::BC1_RGBA_UNORM_BLOCK, F::R32G32_UINT, true));
    // One texel size, another class: `A8_UNORM` is "8-bit alpha", a 4-pack 10-bit format "64-bit
    // R10G10B10A10", a 4:2:2 format "32-bit G8B8G8R8"; `A1B5G5R5_UNORM_PACK16` is "16-bit".
    assert!(!compatible(F::A8_UNORM, F::R8_UINT, false));
    assert!(!compatible(F::R8_UNORM, F::A8_UNORM, false));
    assert!(!compatible(
        F::R10X6G10X6B10X6A10X6_UNORM_4PACK16,
        F::R16G16B16A16_UNORM,
        false
    ));
    assert!(!compatible(F::G8B8G8R8_422_UNORM, F::R32_UINT, false));
    assert!(compatible(F::A1B5G5R5_UNORM_PACK16, F::R16_UINT, false));
    for pf in PixelFormat::ALL {
        if let (Some(twin), Some(a)) = (srgb_twin(*pf), direct_format(*pf)) {
            assert_eq!(srgb_twin(twin), Some(*pf));
            if let Some(b) = direct_format(twin) {
                assert!(compatible(a, b, true), "{pf:?}");
            }
        }
    }
}

#[test]
fn image_limits_name_the_device_limit_or_sample_count() {
    let (name, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&desc, &device);
    let l = layout(
        TextureType::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        4,
        1,
    );
    let plan = plan_image(&l, TextureUsage::RENDER_TARGET, false, &facts).expect(&name);
    let roomy = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 16384,
            height: 16384,
            depth: 1,
        },
        max_mip_levels: 15,
        max_array_layers: 2048,
        sample_counts: vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_4,
        max_resource_size: 1 << 31,
    };
    assert_eq!(
        check_image_limits(&plan, TextureType::Type2DMultisample, &roomy),
        Ok(())
    );
    let narrow = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 4,
            height: 16384,
            depth: 1,
        },
        ..roomy
    };
    let refused = check_image_limits(&plan, TextureType::Type2DMultisample, &narrow).unwrap_err();
    assert_eq!(refused.feature, Feature::TextureSize);
    assert_eq!(
        refused.needs,
        Needs::AnyOf(vec![vec![Capability::Limit {
            limit: metal2vulkan::device::VkLimit::MaxImageDimension2D,
            required: 8,
            available: 4,
        }]])
    );
    let single = vk::ImageFormatProperties {
        sample_counts: vk::SampleCountFlags::TYPE_1,
        ..roomy
    };
    let refused = check_image_limits(&plan, TextureType::Type2DMultisample, &single).unwrap_err();
    assert_eq!(refused.feature, Feature::TextureSampleCount);
    assert_eq!(
        refused.needs,
        Needs::AnyOf(vec![vec![Capability::Property(
            metal2vulkan::device::VkProperty::FramebufferColorSampleCount(4)
        )]])
    );
}

#[test]
fn the_create_info_carries_the_plan_and_its_format_list() {
    // `ash` values have no `Debug`: they are compared as the crate's own.
    let (name, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&desc, &device);
    let l = layout(TextureType::CubeArray, PixelFormat::RGBA8Unorm, 1, 2);
    let plan = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
    let formats = plan.ash_view_formats();
    let listed: Vec<Format> = formats.iter().map(|&f| f.into()).collect();
    assert_eq!(listed, [Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB]);
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = plan.create_info(Sharing::new(&[0]), &formats, &mut list);
    assert_eq!(ImageType::from(info.image_type), ImageType::TYPE_2D);
    assert_eq!(Format::from(info.format), Format::R8G8B8A8_UNORM);
    assert_eq!(
        (info.extent.width, info.extent.height, info.extent.depth),
        (8, 8, 1)
    );
    assert_eq!((info.mip_levels, info.array_layers), (1, 12));
    assert_eq!(
        SampleCountFlags::from(info.samples),
        SampleCountFlags::COUNT_1
    );
    assert!(info.tiling == vk::ImageTiling::OPTIMAL);
    assert!(info.sharing_mode == vk::SharingMode::EXCLUSIVE);
    assert!(info.initial_layout == vk::ImageLayout::UNDEFINED);
    // On a device of several queue families the image is concurrent among all of them.
    let families = [0, 1, 2, 3];
    let mut several_list = vk::ImageFormatListCreateInfo::default();
    let several = plan.create_info(Sharing::new(&families), &formats, &mut several_list);
    assert!(several.sharing_mode == vk::SharingMode::CONCURRENT);
    assert_eq!(several.queue_family_index_count, 4);
    assert_eq!(
        ImageCreateFlags::from(info.flags),
        ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::CUBE_COMPATIBLE
    );
    assert_eq!(ImageUsageFlags::from(info.usage), plan.usage);
    assert!(!info.p_next.is_null());
    // Without a list, nothing is chained.
    let l = layout(TextureType::Type2D, PixelFormat::R32Float, 1, 1);
    let plan = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
    let formats = plan.ash_view_formats();
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = plan.create_info(Sharing::new(&[0]), &formats, &mut list);
    assert!(info.p_next.is_null() && formats.is_empty());
}

#[test]
fn compressed_formats_are_compatible_only_within_their_family() {
    type F = Format;
    // PVRTC blocks count one texel in Vulkan's registry; they are still compressed.
    assert!(!compatible(
        F::PVRTC1_2BPP_UNORM_BLOCK_IMG,
        F::R32G32_UINT,
        true
    ));
    assert!(compatible(
        F::PVRTC1_2BPP_UNORM_BLOCK_IMG,
        F::PVRTC1_2BPP_SRGB_BLOCK_IMG,
        true
    ));
    assert!(compatible(F::BC4_UNORM_BLOCK, F::BC4_SNORM_BLOCK, true));
    assert!(!compatible(F::BC4_UNORM_BLOCK, F::BC4_SNORM_BLOCK, false));
    assert_eq!(block_family(PixelFormat::BC4RSnorm), PixelFormat::BC4RUnorm);
    assert_eq!(
        block_family(PixelFormat::ASTC6x6HDR),
        PixelFormat::ASTC6x6LDR
    );
    assert_eq!(
        block_family(PixelFormat::ASTC6x6SRGB),
        PixelFormat::ASTC6x6LDR
    );
    assert_eq!(block_family(PixelFormat::BC1RGBASRGB), PixelFormat::BC1RGBA);
    assert_eq!(
        block_family(PixelFormat::RGBA8UnormSRGB),
        PixelFormat::RGBA8Unorm
    );
    assert_eq!(block_family(PixelFormat::R32Float), PixelFormat::R32Float);
}

/// A texture of ASTC's 4x2 or 8x4 formats, which Metal makes and no Vulkan block holds, is
/// refused `NotImplemented(Format)` on every committed device, sampled or rendered to.
#[test]
fn astc_4x2_and_8x4_textures_are_refused_typed() {
    use PixelFormat as P;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for pf in [
            P::ASTC4x2LDR,
            P::ASTC4x2SRGB,
            P::ASTC4x2HDR,
            P::ASTC8x4LDR,
            P::ASTC8x4SRGB,
            P::ASTC8x4HDR,
        ] {
            let l = layout(TextureType::Type2D, pf, 1, 1);
            for usage in [TextureUsage::SHADER_READ, TextureUsage::RENDER_TARGET] {
                match plan_image(&l, usage, false, &facts) {
                    Err(ImageRefusal::Unsupported(u)) => {
                        assert_eq!(u.feature, Feature::Format, "{name}: {pf:?}");
                        assert_eq!(u.needs, Needs::NotImplemented, "{name}: {pf:?}: {u:?}");
                    }
                    other => panic!("{name}: {pf:?} {usage:?}: {other:?}"),
                }
            }
        }
    }
}

/// Apple's extended-range formats are held raw in the unsigned integer format of their bytes,
/// sampled, stored, and since item 10 rendered to: a render target's image is a color attachment of
/// the raw format, and a memoryless texture of one is a transient attachment of it. `ARGB8Unorm` is
/// `R8G8B8A8_UNORM` read through a rotating view, and the two-plane `RGB10A8_2P_XR10` formats hold
/// both planes side by side in `R32G32_UINT`. The bfloat formats' render targets are color
/// attachments of their raw formats.
#[test]
fn extended_range_textures_hold_their_bytes_raw() {
    let usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    let mut planned = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for (pf, raw) in [
            (PixelFormat::BGR10XR, Format::R32_UINT),
            (PixelFormat::BGR10XRSRGB, Format::R32_UINT),
            (PixelFormat::BGRA10XR, Format::R16G16B16A16_UINT),
            (PixelFormat::BGRA10XRSRGB, Format::R16G16B16A16_UINT),
            (PixelFormat::RGB10A82PXR10, Format::R32G32_UINT),
            (PixelFormat::RGB10A82PXR10SRGB, Format::R32G32_UINT),
            // The bfloat formats' render targets, which the consumer's passes clear.
            (PixelFormat::R16Bfloat, Format::R16_UINT),
            (PixelFormat::RG16Bfloat, Format::R16G16_UINT),
            (PixelFormat::RGBA16Bfloat, Format::R16G16B16A16_UINT),
        ] {
            let what = format!("{name}: {pf:?}");
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let Ok(plan) = plan_image(&l, usage, false, &facts) else {
                continue;
            };
            planned += 1;
            check_plan(&name, pf, usage, &plan, &facts);
            assert_eq!(plan.format, raw, "{what}");
            assert!(plan.usage.contains(ImageUsageFlags::SAMPLED), "{what}");
            let renders = facts
                .optimal(raw)
                .contains(FormatFeatureFlags2::COLOR_ATTACHMENT);
            // MoltenVK renders to all three raw formats on Apple's GPUs, as Metal does.
            if name.starts_with("moltenvk") {
                assert!(renders, "{what}");
            }
            assert_eq!(
                plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT),
                renders,
                "{what}"
            );
            assert_eq!(
                plan.unavailable.contains(TextureUsage::RENDER_TARGET),
                !renders,
                "{what}"
            );
            // A memoryless texture: a transient attachment of the raw format, as the consumer's
            // guest makes one of `RGB10A8_2P_XR10`.
            for target in [
                TextureUsage::RENDER_TARGET,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ] {
                let memoryless = plan_images(&l, target, true, &facts);
                match renders {
                    true => {
                        let (plan, twin) = memoryless.expect(&what);
                        assert!(twin.is_none(), "{what}");
                        assert_eq!(plan.format, raw, "{what}");
                        assert!(
                            plan.usage.contains(
                                ImageUsageFlags::COLOR_ATTACHMENT
                                    | ImageUsageFlags::TRANSIENT_ATTACHMENT
                            ),
                            "{what}"
                        );
                    }
                    false => assert!(memoryless.is_err(), "{what}"),
                }
            }
        }
        // A memoryless `ARGB8Unorm` texture: a transient attachment of the image holding its
        // bytes, which a fragment function writes rotated.
        let l = layout(TextureType::Type2D, PixelFormat::ARGB8Unorm, 1, 1);
        if facts
            .optimal(Format::R8G8B8A8_UNORM)
            .contains(FormatFeatureFlags2::COLOR_ATTACHMENT)
        {
            let (plan, twin) = plan_images(&l, TextureUsage::RENDER_TARGET, true, &facts)
                .unwrap_or_else(|e| panic!("{name}: a memoryless ARGB8Unorm texture: {e:?}"));
            assert!(twin.is_none(), "{name}");
            assert_eq!(plan.format, Format::R8G8B8A8_UNORM, "{name}");
            assert!(
                plan.usage.contains(
                    ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSIENT_ATTACHMENT
                ),
                "{name}"
            );
        }
        if let Ok(plan) = plan_image(&l, TextureUsage::SHADER_READ, false, &facts) {
            assert_eq!(plan.format, Format::R8G8B8A8_UNORM, "{name}");
            let m = vk_format(
                PixelFormat::ARGB8Unorm,
                &[FormatUsage::SampledImage].into(),
                device.profile(),
            )
            .expect(&name);
            use metal2vulkan::vk::{ComponentMapping, ComponentSwizzle as S};
            assert_eq!(
                m.components,
                ComponentMapping {
                    r: S::G,
                    g: S::B,
                    b: S::A,
                    a: S::R,
                },
                "{name}"
            );
        }
    }
    assert!(planned > 0);
}

/// A PVRTC texture a device has no format for holds its blocks in the first carrier the device
/// copies, which copies alone reach: its reads are its decoded twin's, so no usage it asks for is
/// unavailable (none a shader write or a pass needs is asked for here). A 4-bit
/// texture's image is the texture's extent; a 2-bit texture's is half as wide, rounded up, its 8
/// x 4 blocks each over 4 x 4 carrier texels. A device with `VK_IMG_format_pvrtc` keeps its RGBA
/// formats' own images, 2- and 4-bit; no device refuses a PVRTC format where a carrier copies.
#[test]
fn a_pvrtc_texture_without_a_format_holds_its_blocks_in_a_carrier() {
    let mut carried = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let pvrtc = desc.extensions.contains(&VkExtension::ImgFormatPvrtc);
        let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
        let carrier = CARRIERS
            .into_iter()
            .find(|&f| facts.optimal(f).contains(copies));
        for (pf, width) in [
            (PixelFormat::PVRTCRGB4BPP, 16),
            (PixelFormat::PVRTCRGB4BPPSRGB, 16),
            (PixelFormat::PVRTCRGBA4BPP, 16),
            (PixelFormat::PVRTCRGBA4BPPSRGB, 16),
            (PixelFormat::PVRTCRGB2BPP, 8),
            (PixelFormat::PVRTCRGB2BPPSRGB, 8),
            (PixelFormat::PVRTCRGBA2BPP, 8),
            (PixelFormat::PVRTCRGBA2BPPSRGB, 8),
        ] {
            let what = format!("{name} {pf:?}");
            let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
            let mut l = layout(TextureType::Type2D, pf, 1, 1);
            l.extent = [16, 16, 1];
            l.levels = 5;
            let rgba = matches!(
                pf,
                PixelFormat::PVRTCRGBA4BPP
                    | PixelFormat::PVRTCRGBA4BPPSRGB
                    | PixelFormat::PVRTCRGBA2BPP
                    | PixelFormat::PVRTCRGBA2BPPSRGB
            );
            let plan = plan_image(&l, usage, false, &facts);
            if rgba && pvrtc {
                let plan = plan.expect(&what);
                assert!(!holds_blocks(&plan), "{what}");
                continue;
            }
            let Some(carrier) = carrier else {
                assert!(plan.is_err(), "{what}");
                continue;
            };
            let plan = plan.expect(&what);
            carried += 1;
            assert!(holds_blocks(&plan), "{what}");
            assert_eq!(plan.format, carrier, "{what}");
            assert_eq!((plan.extent, plan.levels), ([width, 16, 1], 5), "{what}");
            assert_eq!(
                plan.usage,
                ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
                "{what}"
            );
            assert_eq!(plan.unavailable, TextureUsage::empty(), "{what}");
            check_plan(&name, pf, usage, &plan, &facts);
            // A cube's carrier holds its faces as layers, of no create flags: no view of it is
            // made, and a 2-bit one's half-width carrier is no square.
            for (ty, array) in [(TextureType::Cube, 1), (TextureType::CubeArray, 2)] {
                let mut l = layout(ty, pf, 1, array);
                l.extent = [16, 16, 1];
                let plan = plan_image(&l, usage, false, &facts).expect(&what);
                assert!(holds_blocks(&plan), "{what} {ty:?}");
                assert_eq!(plan.flags, ImageCreateFlags::empty(), "{what} {ty:?}");
                assert_eq!(
                    (plan.extent, plan.layers),
                    ([width, 16, 1], 6 * array as u32),
                    "{what} {ty:?}"
                );
                check_plan(&name, pf, usage, &plan, &facts);
            }
        }
    }
    assert!(carried >= 40, "{carried}");
}

/// An ETC2 or EAC texture a device has no format for holds its blocks in the first carrier of
/// its blocks' bytes the device copies (`etc.rs`): of the texture's extent, levels and layers
/// (a cube's faces as layers, of no create flags), which copies alone reach, its reads its
/// decoded twin's. A device with the formats keeps each texture's own image.
#[test]
fn an_etc_texture_without_a_format_holds_its_blocks_in_a_carrier() {
    use super::super::etc::{decode_of, CARRIERS_16, CARRIERS_8};
    use metal2vulkan::bind::EtcDecode;
    let mut carried = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
        for &pf in PixelFormat::ALL {
            let Some(decode) = EtcDecode::of(pf) else {
                continue;
            };
            let what = format!("{name} {pf:?}");
            let usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
            let mut l = layout(TextureType::Type2D, pf, 1, 1);
            l.extent = [20, 12, 1];
            l.levels = 3;
            let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), device.profile());
            let plan = plan_image(&l, usage, false, &facts);
            // A device that samples neither the format nor its twin's view refuses it.
            let Ok(sampled) = sampled else {
                assert!(plan.is_err(), "{what}");
                continue;
            };
            if sampled.emulation == metal2vulkan::metal::format::FormatEmulation::Direct {
                let plan = plan.expect(&what);
                assert_eq!(decode_of(&plan), None, "{what}");
                continue;
            }
            let carriers: &[Format] = match decode.block_bytes() {
                8 => &CARRIERS_8,
                _ => &CARRIERS_16,
            };
            let Some(&carrier) = carriers
                .iter()
                .find(|&&f| facts.optimal(f).contains(copies))
            else {
                assert!(plan.is_err(), "{what}");
                continue;
            };
            let plan = plan.expect(&what);
            carried += 1;
            assert_eq!(decode_of(&plan), Some(decode), "{what}");
            assert_eq!(plan.format, carrier, "{what}");
            assert_eq!((plan.extent, plan.levels), ([20, 12, 1], 3), "{what}");
            assert_eq!(
                plan.usage,
                ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
                "{what}"
            );
            assert_eq!(plan.unavailable, TextureUsage::empty(), "{what}");
            check_plan(&name, pf, usage, &plan, &facts);
            let mut l = layout(TextureType::Cube, pf, 1, 1);
            l.extent = [16, 16, 1];
            let plan = plan_image(&l, usage, false, &facts).expect(&what);
            assert_eq!(decode_of(&plan), Some(decode), "{what} cube");
            assert_eq!(plan.flags, ImageCreateFlags::empty(), "{what} cube");
            assert_eq!((plan.extent, plan.layers), ([16, 16, 1], 6), "{what} cube");
            check_plan(&name, pf, usage, &plan, &facts);
        }
    }
    // Ten formats on each committed profile without ETC2 (lavapipe, RADV, NVIDIA).
    assert!(carried >= 50, "{carried}");
}

/// Each level of a 2-bit PVRTC texture's half-width carrier holds the blocks the texture's level
/// holds, at every power-of-two side from 1 to 4096 and every level: `ceil(w / 8)` across and
/// `ceil(h / 4)` down, one block once the level is no larger than one; a region of whole blocks
/// (x a multiple of 8, the width 8 a block or to the level's edge) is the carrier's region of
/// the same blocks, and a buffer row of `n` blocks is `4n` carrier texels.
#[test]
fn a_two_bit_pvrtc_carrier_holds_each_level_s_blocks() {
    let (name, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    let facts = facts(&desc, &device);
    for side in (0..13).map(|k| 1u32 << k) {
        let mut l = layout(TextureType::Type2D, PixelFormat::PVRTCRGB2BPP, 1, 1);
        l.extent = [side, side, 1];
        l.levels = side.ilog2() + 1;
        let plan = plan_image(&l, TextureUsage::SHADER_READ, false, &facts).expect(&name);
        assert!(holds_blocks(&plan), "{side}");
        for level in 0..l.levels {
            let w = (side >> level).max(1);
            let carrier = (plan.extent[0] >> level).max(1);
            assert_eq!(carrier.div_ceil(4), w.div_ceil(8), "{side} level {level}");
            for (x, width) in [(0, w), (0, w.min(8)), (8, w.saturating_sub(8))] {
                if x >= w || width == 0 {
                    continue;
                }
                let region = ImageRegion {
                    aspect: ImageAspectFlags::COLOR,
                    level,
                    layer: 0,
                    layers: 1,
                    origin: [x, 0, 0],
                    size: [width, 4.min(w), 1],
                };
                let image = image_region(&plan, region);
                // The carrier's region starts at its first block's texel and covers its blocks.
                assert_eq!(image.origin[0], x / 8 * 4, "{side} level {level} at {x}");
                let (first, last) = (x / 8, (x + width).div_ceil(8));
                assert_eq!(
                    (
                        image.origin[0] / 4,
                        (image.origin[0] + image.size[0]).div_ceil(4)
                    ),
                    (first, last),
                    "{side} level {level} at {x}"
                );
                assert!(image.origin[0] + image.size[0] <= carrier, "{side} {level}");
            }
        }
        let row = BufferImageRegion {
            buffer_offset: 0,
            row_length: 3 * 8,
            image_height: 0,
            image: ImageRegion {
                aspect: ImageAspectFlags::COLOR,
                level: 0,
                layer: 0,
                layers: 1,
                origin: [0, 0, 0],
                size: [side.min(8), side.min(4), 1],
            },
        };
        assert_eq!(image_buffer_region(&plan, row).row_length, 3 * 4);
    }
}

/// A 4:2:2 texture holds its 2-byte texels in an `R8G8_UNORM` image of its extent, for copies, on
/// every committed profile (each copies the format), its reads and samples its decoded twin's:
/// every other usage a shader or a pass needs is unavailable, whatever its storage and usage. Its
/// twin is an `R8G8B8A8_UNORM` image of the texture's extent and level, its copies' unit an
/// `RGBA8Unorm` texel, and its decode the format's.
#[test]
fn a_four_two_two_texture_holds_its_texels_in_a_carrier() {
    use super::super::chroma_pairs::decode_of;
    use super::super::texture::copy_units;
    use super::super::texture::decoded_twin::{twin_plan, TwinDecode};
    use metal2vulkan::bind::ChromaDecode;
    let mut carried = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
            for usage in [
                TextureUsage::SHADER_READ,
                TextureUsage::SHADER_WRITE,
                TextureUsage::RENDER_TARGET,
                TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
                TextureUsage::empty(),
            ] {
                let what = format!("{name} {pf:?} {usage:?}");
                let mut l = layout(TextureType::Type2D, pf, 1, 1);
                l.extent = [18, 6, 1];
                let plan = plan_image(&l, usage, false, &facts).expect(&what);
                assert!(holds_pairs(&plan), "{what}");
                assert_eq!(plan.format, PAIRS_CARRIER, "{what}");
                assert_eq!((plan.extent, plan.levels), ([18, 6, 1], 1), "{what}");
                assert_eq!(
                    plan.usage,
                    ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
                    "{what}"
                );
                assert_eq!(
                    plan.unavailable,
                    effective_usage(usage)
                        .intersection(TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET),
                    "{what}"
                );
                let decode = decode_of(&plan).expect(&what);
                assert_eq!(Some(decode), ChromaDecode::of(pf), "{what}");
                for routed in [false, true] {
                    let twin = twin_plan(&plan, &l, TwinDecode::Chroma(decode), routed);
                    assert_eq!(
                        (twin.pixel_format, twin.format),
                        (PixelFormat::RGBA8Unorm, Format::R8G8B8A8_UNORM),
                        "{what}"
                    );
                    let unit = (ImageAspectFlags::COLOR, 4, [1, 1]);
                    assert_eq!(copy_units(twin.format, twin.pixel_format), [unit], "{what}");
                    assert!(twin.usage.contains(ImageUsageFlags::SAMPLED), "{what}");
                    assert_eq!((twin.extent, twin.levels), ([18, 6, 1], 1), "{what}");
                }
                check_plan(&name, pf, usage, &plan, &facts);
                carried += 1;
            }
        }
    }
    assert!(carried >= 40, "{carried}");
}

/// A texture of Apple's private `BlockWritesOnly` usage is written as one of `ShaderWrite` (plain
/// Metal stores every write form into it, and its validation takes them), so its
/// image is the image of the texture with `ShaderWrite` in its place, on every committed profile
/// and for every pixel format: CoreAnimation's blur textures' `ShaderRead | RenderTarget |
/// BlockWritesOnly` plans as `ShaderRead | RenderTarget | ShaderWrite` does, the storage usage
/// included where the device stores the format. Where it does not, the texture is not refused:
/// `ShaderWrite` is not a usage it asked for, and a binding takes the usage shadow; there the
/// `ShaderWrite` colour texture's image, which Metal keeps uncompressed and the `BlockWritesOnly`
/// one's not, is mutable to any format.
#[test]
fn a_block_writes_only_texture_s_image_is_a_shader_write_one_s() {
    let blur =
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET | TextureUsage::BLOCK_WRITES_ONLY;
    let written =
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE;
    let mut stored = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for &pf in PixelFormat::ALL {
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let what = format!("{name}: {pf:?}");
            match (
                plan_image(&l, blur, false, &facts),
                plan_image(&l, written, false, &facts),
            ) {
                (Ok(b), Ok(w)) => {
                    let any_format = w.flags.contains(ImageCreateFlags::MUTABLE_FORMAT)
                        && w.view_formats.is_empty()
                        && !w.usage.contains(ImageUsageFlags::STORAGE)
                        && !metal2vulkan::metal::format::is_depth_stencil(pf);
                    let (flags, views) = match any_format {
                        true => (b.flags | ImageCreateFlags::MUTABLE_FORMAT, Vec::new()),
                        false => (b.flags, b.view_formats.clone()),
                    };
                    assert_eq!(
                        (b.format, b.usage, flags, &views),
                        (w.format, w.usage, w.flags, &w.view_formats),
                        "{what}"
                    );
                    assert_eq!(
                        b.unavailable,
                        w.unavailable.difference(TextureUsage::SHADER_WRITE),
                        "{what}"
                    );
                    if b.usage.contains(ImageUsageFlags::STORAGE) {
                        stored += 1;
                    }
                }
                // A format the device holds no image of is refused for both, typed alike (each
                // naming the first usage the device lacks for it).
                (Err(b), Err(w)) => match (&b, &w) {
                    (ImageRefusal::Unsupported(bu), ImageRefusal::Unsupported(wu)) => {
                        assert_eq!(bu.feature, wu.feature, "{what}")
                    }
                    _ => assert_eq!(
                        std::mem::discriminant(&b),
                        std::mem::discriminant(&w),
                        "{what}: {b:?} against {w:?}"
                    ),
                },
                (b, w) => panic!("{what}: {b:?} against {w:?}"),
            }
        }
        let l = layout(TextureType::Type2D, PixelFormat::RGBA16Float, 1, 1);
        let plan = plan_image(&l, blur, false, &facts).expect(&name);
        assert!(plan.usage.contains(ImageUsageFlags::STORAGE), "{name}");
    }
    assert!(stored > 100, "{stored}");
}

/// A texture of `R10Unorm_X6` or `RG10Unorm_X12` (or of their `_sRGB` twins) holds its words raw
/// in `R16_UINT` or `R16G16_UINT` for every usage, on every committed device, in memory of its own
/// as much as over memory it is given: one plan for both, as Metal makes it from either
/// (`metal::format::padded_words`).
#[test]
fn padded_formats_are_held_raw_from_every_creator() {
    use PixelFormat as P;
    let usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for (pf, raw) in [
            (P::R10UnormX6, Format::R16_UINT),
            (P::RG10UnormX12, Format::R16G16_UINT),
            (P::R10UnormX6SRGB, Format::R16_UINT),
            (P::RG10UnormX12SRGB, Format::R16G16_UINT),
        ] {
            let l = layout(TextureType::Type2D, pf, 1, 1);
            let plan = plan_image(&l, usage, false, &facts).expect(&name);
            assert_eq!((plan.pixel_format, plan.format), (pf, raw), "{name}");
            // Every usage on the Apple venues and lavapipe; a device without the raw format's
            // storage use plans the image without it.
            let wanted = match name.starts_with("moltenvk") || name == "lavapipe" {
                true => {
                    ImageUsageFlags::SAMPLED
                        | ImageUsageFlags::STORAGE
                        | ImageUsageFlags::COLOR_ATTACHMENT
                }
                false => ImageUsageFlags::SAMPLED,
            };
            assert!(
                plan.usage.contains(wanted),
                "{name} {pf:?}: {:?}",
                plan.usage
            );
            let (images, twin) = plan_images(&l, usage, false, &facts).expect(&name);
            assert_eq!((images, twin), (plan, None), "{name}");
        }
    }
}

/// A decoded twin's plan holds its texels' own format, so its copies move texels (not the BC
/// image's blocks, `copy_units`); a twin routed as its BC image's CPU copies go copies both
/// ways, as the staging queue's images do; and its start, its zero blocks' decode, is BC1's
/// opaque black and every other decode's zero.
#[test]
fn decoded_twins_are_planned_in_their_texels_format() {
    use super::super::texture::copy_units;
    use super::super::texture::decoded_twin::{twin_plan, TwinDecode};
    use metal2vulkan::bind::BlockDecode;
    let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    let mut planned = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for &pf in PixelFormat::ALL {
            let Some(decode) = BlockDecode::of(pf) else {
                continue;
            };
            let texel = decode.texel_pixel_format();
            assert_eq!(direct_format(texel), Some(decode.texel_format()), "{pf:?}");
            let zero = decode.decode(&vec![0; decode.block_bytes() as usize], [1, 1, 1]);
            let start = match decode == BlockDecode::Bc1 {
                true => vec![0, 0, 0, 255],
                false => vec![0; decode.texel_bytes() as usize],
            };
            assert_eq!(zero, start, "{pf:?}");
            let l = layout(TextureType::Type2DArray, pf, 1, 3);
            let Ok((plan, _)) = plan_images(&l, TextureUsage::SHADER_READ, false, &facts) else {
                continue;
            };
            planned += 1;
            for routed in [false, true] {
                let twin = twin_plan(&plan, &l, TwinDecode::Block(decode), routed);
                let what = format!("{name}: {pf:?}, routed {routed}");
                assert_eq!(
                    (twin.pixel_format, twin.format),
                    (texel, decode.texel_format())
                );
                let unit = (ImageAspectFlags::COLOR, decode.texel_bytes(), [1, 1]);
                assert_eq!(copy_units(twin.format, twin.pixel_format), [unit], "{what}");
                assert!(twin.usage.contains(ImageUsageFlags::SAMPLED), "{what}");
                assert!(twin.usage.contains(ImageUsageFlags::TRANSFER_DST), "{what}");
                assert_eq!(twin.usage.contains(both), routed, "{what}");
                let shape = |p: &ImagePlan| (p.image_type, p.extent, p.levels, p.layers, p.samples);
                assert_eq!(shape(&twin), shape(&plan), "{what}");
            }
        }
    }
    assert!(planned > 0);
}

/// A decode pass's bands of a region cover each of its texels once, and each band's blocks and
/// texels take at most the budget together, so a pass in the device's decode scratch
/// (`PassScratch::Device`, of the budget and an alignment) holds any band; only a single block
/// row of one image past the budget is a band alone. Regions of BC decodes, of ASTC ones
/// (blocks of 12 x 10 and 4 x 4, texels of 16 and 4 bytes), PVRTC ones, ETC2 and EAC ones
/// (texels of 2, 8 and 4 bytes) and 4:2:2 ones (pairs of 2 x 1), square and narrow, of sides off a
/// multiple of the block, of layers and of depth slices, under the band budget and two small ones
/// that make them band.
#[test]
fn decode_bands_cover_each_texel_once_within_the_budget() {
    use super::super::texture::decoded_twin::{bands, TwinDecode, BAND_BYTES};
    use metal2vulkan::bind::{
        AstcDecode, AstcTexels, BlockDecode, ChromaDecode, EtcDecode, EtcTexels, PvrtcDecode,
    };
    let shapes: [([u32; 3], u32); 7] = [
        ([2048, 2048, 1], 1),
        ([16384, 16, 1], 1),
        ([1, 16384, 1], 6),
        ([33, 17, 1], 3),
        ([512, 512, 1], 64),
        ([64, 64, 32], 1),
        ([3, 1, 1], 1),
    ];
    let astc = |pf| AstcDecode::of(pf).expect("an ASTC decode");
    let decodes = [
        TwinDecode::Block(BlockDecode::Bc1),
        TwinDecode::Block(BlockDecode::Bc3),
        TwinDecode::Block(BlockDecode::Bc5Unorm),
        TwinDecode::Astc(astc(PixelFormat::ASTC12x10LDR), AstcTexels::Float),
        TwinDecode::Astc(astc(PixelFormat::ASTC4x4SRGB), AstcTexels::Unorm8),
        TwinDecode::Pvrtc(PvrtcDecode::of(PixelFormat::PVRTCRGBA2BPP).expect("a PVRTC decode")),
        TwinDecode::Pvrtc(PvrtcDecode::of(PixelFormat::PVRTCRGB4BPP).expect("a PVRTC decode")),
        TwinDecode::Etc(EtcDecode::R11Unorm, EtcTexels::Compact),
        TwinDecode::Etc(EtcDecode::Rg11Snorm, EtcTexels::Float),
        TwinDecode::Etc(EtcDecode::Rgba8, EtcTexels::Compact),
        TwinDecode::Chroma(ChromaDecode::Gbgr),
    ];
    for decode in decodes {
        let (texel, block) = (
            u64::from(decode.texel_bytes()),
            u64::from(decode.block_bytes()),
        );
        let [bw, bh] = decode.block_extent().map(u64::from);
        for (size, layers) in shapes {
            for most in [BAND_BYTES, 1 << 20, 64 << 10] {
                let region = ImageRegion {
                    aspect: ImageAspectFlags::COLOR,
                    level: 0,
                    layer: 2,
                    layers,
                    origin: [4, 8, 0],
                    size,
                };
                let what = format!("{decode:?} {size:?} x{layers} in {most}");
                let mut covered = std::collections::BTreeMap::new();
                for band in bands(&region, decode, most) {
                    let [w, h, d] = band.size.map(u64::from);
                    // A whole-level decode's band reads all its images' blocks, of any rows.
                    let (block_rows, row) = match decode.whole_levels() {
                        true => (u64::from(size[1]).div_ceil(bh), 1),
                        false => (h.div_ceil(bh), bh),
                    };
                    let bytes = w.div_ceil(bw) * block_rows * d * u64::from(band.layers) * block
                        + w * h * d * u64::from(band.layers) * texel;
                    let one_row = h <= row && d * u64::from(band.layers) == 1;
                    assert!(bytes <= most || one_row, "{what}: {band:?} takes {bytes}");
                    assert_eq!(band.size[0], size[0], "{what}");
                    assert_eq!(band.origin[0], region.origin[0], "{what}");
                    for layer in band.layer..band.layer + band.layers {
                        for z in band.origin[2]..band.origin[2] + band.size[2] {
                            for y in band.origin[1]..band.origin[1] + band.size[1] {
                                *covered.entry((layer, z, y)).or_insert(0) += 1;
                            }
                        }
                    }
                }
                let rows = u64::from(size[1]) * u64::from(size[2]) * u64::from(layers);
                assert_eq!(covered.len() as u64, rows, "{what}");
                assert!(covered.values().all(|&n| n == 1), "{what}");
                let inside = covered.keys().all(|&(layer, z, y)| {
                    (2..2 + layers).contains(&layer) && z < size[2] && (8..8 + size[1]).contains(&y)
                });
                assert!(inside, "{what}");
            }
        }
    }
}

/// On every committed device without ASTC, every ASTC LDR format's texture, of every usage Metal
/// takes, holds its blocks in a carrier (`astc.rs`) whose every level keeps the blocks the
/// texture's level holds, at sides no multiple of the block too: a whole level's region (and one
/// starting at its second block) maps to the carrier's region of the same blocks, inside the
/// carrier's level, and back. A profile that lists no carrier's copies refuses the texture: the
/// decoded twin's view holds no blocks.
#[test]
fn astc_carriers_hold_each_level_s_blocks() {
    use super::super::astc::{decode_of, texture_region, CARRIERS as ASTC_CARRIERS};
    use metal2vulkan::bind::AstcDecode;
    use metal2vulkan::metal::format::FormatEmulation;
    let mut carried = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        for &pf in PixelFormat::ALL {
            let Some(decode) = AstcDecode::of(pf) else {
                continue;
            };
            let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), profile);
            if sampled.map(|m| m.emulation) != Ok(FormatEmulation::Decoded) {
                continue;
            }
            let [bw, bh] = decode.block_extent();
            let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
            let carriers = (ASTC_CARRIERS.iter()).any(|&f| facts.optimal(f).contains(copies));
            for (width, height) in [(1, 1), (7, 3), (16, 16), (37, 23), (100, 257)] {
                let what = format!("{name} {pf:?} {width}x{height}");
                let mut l = layout(TextureType::Type2D, pf, 1, 1);
                l.extent = [width, height, 1];
                l.levels = width.max(height).ilog2() + 1;
                let usage = TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE
                    | TextureUsage::RENDER_TARGET;
                let planned = plan_image(&l, usage, false, &facts);
                if !carriers {
                    assert!(planned.is_err(), "{what}: {planned:?}");
                    continue;
                }
                let plan = planned.expect(&what);
                assert_eq!(decode_of(&plan), Some(decode), "{what}");
                assert!(ASTC_CARRIERS.contains(&plan.format), "{what}");
                carried += 1;
                for level in 0..l.levels {
                    let [w, h] = [width, height].map(|side| (side >> level).max(1));
                    let carrier = [0, 1].map(|axis| (plan.extent[axis] >> level).max(1));
                    let blocks = [w.div_ceil(bw), h.div_ceil(bh)];
                    assert!(carrier[0].div_ceil(4) >= blocks[0], "{what} level {level}");
                    assert!(carrier[1].div_ceil(4) >= blocks[1], "{what} level {level}");
                    let starts = [(0, 0), (bw.min(w - 1) / bw * bw, 0)];
                    for (x, y) in starts {
                        let region = ImageRegion {
                            aspect: ImageAspectFlags::COLOR,
                            level,
                            layer: 0,
                            layers: 1,
                            origin: [x, y, 0],
                            size: [w - x, h - y, 1],
                        };
                        let image = image_region(&plan, region);
                        let end = [0, 1].map(|axis| image.origin[axis] + image.size[axis]);
                        assert!(end[0] <= carrier[0] && end[1] <= carrier[1], "{what}");
                        let across = image.size[0].div_ceil(4);
                        assert_eq!(across, (w - x).div_ceil(bw), "{what} level {level}");
                        assert_eq!(image.size[1].div_ceil(4), blocks[1], "{what}");
                        let back = texture_region(&plan, l.extent, image);
                        assert_eq!(back, Some(region), "{what} level {level} at {x}");
                    }
                }
            }
        }
    }
    assert!(carried > 0, "no committed device holds ASTC in a carrier");
}

/// A multisampled `Depth32Float` texture the program only reads is held in its depth image under
/// the default, as before, and in a multisampled `R32_SFLOAT` image of its shape under
/// `MultisampleDepthTwinPolicy::Exact` on every committed profile whose depth image its driver is
/// not measured to keep every `float` in; in its depth image elsewhere. It takes no `D32_SFLOAT`
/// twin beside it under either value: Metal reads a `depth2d_ms` and compares nothing, and no copy
/// writes a multisampled depth image from a colour one. A render target, a memoryless texture, a
/// single-sampled one (which has the twin of its own row) and another depth format are held as
/// before under both values.
#[test]
fn the_switch_holds_a_multisampled_depth_texture_s_texels_in_colour() {
    use PixelFormat as P;
    let read = TextureUsage::SHADER_READ;
    let mut held = Vec::new();
    for (name, desc, device) in committed() {
        let default = facts(&desc, &device);
        let switched = facts_holding_multisample_depth(&desc, &device);
        let exact = default.depth_loss().is_none();
        for samples in [2, 4] {
            let l = layout(TextureType::Type2DMultisample, P::Depth32Float, samples, 1);
            // Under the default: the depth image, as `plan_image` plans it.
            let Ok((plan, twin)) = plan_images(&l, read, false, &default) else {
                continue;
            };
            assert!(twin.is_none(), "{name}: {samples} samples, default");
            assert_eq!(
                plan,
                plan_image(&l, read, false, &default).unwrap(),
                "{name}: {samples} samples, default"
            );
            assert_eq!(
                (plan.pixel_format, plan.format),
                (P::Depth32Float, Format::D32_SFLOAT),
                "{name}: {samples} samples, default"
            );
            // Under the switch: the colour image where the device's depth images may lose the
            // texels and it holds a multisampled `R32_SFLOAT` it samples and copies.
            let (switched_plan, switched_twin) =
                plan_images(&l, read, false, &switched).expect(&name);
            assert!(switched_twin.is_none(), "{name}: {samples} samples, switch");
            let colour = switched_plan.format == Format::R32_SFLOAT;
            if !colour {
                assert_eq!(
                    switched_plan, plan,
                    "{name}: {samples} samples: no colour image, so the plan is the default's"
                );
                continue;
            }
            assert!(
                !exact,
                "{name}: {samples} samples: a colour image without a loss"
            );
            assert_eq!(
                (
                    switched_plan.pixel_format,
                    switched_plan.mapping.aspects,
                    switched_plan.extent,
                    switched_plan.levels,
                    switched_plan.layers,
                    switched_plan.samples
                ),
                (
                    P::Depth32Float,
                    ImageAspectFlags::COLOR,
                    plan.extent,
                    plan.levels,
                    plan.layers,
                    plan.samples
                ),
                "{name}: {samples} samples"
            );
            assert!(
                switched_plan.usage.contains(ImageUsageFlags::SAMPLED),
                "{name}: {samples} samples"
            );
            assert!(
                !switched_plan
                    .usage
                    .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                "{name}: {samples} samples: a read-only texture keeps no attachment usage"
            );
            // The plan every blit and binding reads the choice by (`Texture::holds_depth_in_colour`
            // for the facts and `blit_texture::held` for the images a copy moves).
            assert!(
                holds_depth_in_colour(&switched_plan) && !holds_depth_in_colour(&plan),
                "{name}: {samples} samples"
            );
            // Every copy of the texture's texels moves them in colour, 4 bytes a texel, as a
            // texture with a depth twin's are: a fill of its samples
            // (`blit_fill::sample_fill` through `blit_texture::unit`) and a copy between textures
            // (`blit_plan::pair_unit`) alike.
            assert_eq!(
                copy_unit(P::Depth32Float, &switched_plan.mapping, BlitOption::None).ok(),
                Some(CopyUnit {
                    bytes: 4,
                    extent: [1, 1],
                    aspect: ImageAspectFlags::COLOR
                }),
                "{name}: {samples} samples"
            );
            assert_eq!(
                copy_unit(P::Depth32Float, &plan.mapping, BlitOption::None)
                    .map(|unit| unit.aspect)
                    .ok(),
                Some(ImageAspectFlags::DEPTH),
                "{name}: {samples} samples, default"
            );
            held.push(format!("{name} {samples}"));
        }
        // A render target, a memoryless texture and another depth format are held as before.
        for (usage, memoryless, pf) in [
            (TextureUsage::RENDER_TARGET | read, false, P::Depth32Float),
            (read, true, P::Depth32Float),
            (read, false, P::Depth32FloatStencil8),
        ] {
            let l = layout(TextureType::Type2DMultisample, pf, 4, 1);
            let before = plan_images(&l, usage, memoryless, &default);
            let after = plan_images(&l, usage, memoryless, &switched);
            assert_eq!(
                before.ok(),
                after.ok(),
                "{name}: {pf:?} {usage:?} memoryless {memoryless}"
            );
        }
    }
    assert!(
        !held.is_empty(),
        "no committed profile holds a multisampled depth texture in colour under the switch"
    );
}
