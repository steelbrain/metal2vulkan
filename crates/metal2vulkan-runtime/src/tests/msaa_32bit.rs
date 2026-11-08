//! `supports32BitMSAA` (`families::formats::msaa_32bit`) over every committed device profile
//! (`validation/profiles/`, read-only).
//!
//! A profile has no physical device to ask for an image's limits, so each image here takes any
//! size, level and layer count, and the sample counts the profile states the image takes
//! ([`stated_limits`]): for each of its usages, the format's own color attachment counts where a
//! venue profile states them (`m2v_color_attachment_sample_counts`, its capture's
//! `vkGetPhysicalDeviceImageFormatProperties2`), else the device-wide counts Vulkan floors every
//! format of the kind at. The device test (validation's `runtime_lane::msaa_32bit`) answers with
//! the device's own limits.

use super::capabilities::{committed_descriptions, created, profile_counts};
use crate::families::formats::{
    answers, msaa_32bit, FormatAnswers, FormatCapability, Missing, NotMsaa32, MSAA_32BIT_FLOAT,
    MSAA_32BIT_INTEGER,
};
use crate::image_plan::{DepthImageFacts, ImageFacts, ImagePlan};
use crate::RuntimeGap;
use ash::vk;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy, VkFeature, VkProperty};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{
    ImageCreateFlags, ImageType, ImageUsageFlags, NumericType, SampleCountFlags,
};

/// The sample counts a profile states for each kind of sampled image (`sampledImage*SampleCounts`,
/// which a description does not hold).
pub(super) struct Sampled {
    color: SampleCountFlags,
    integer: SampleCountFlags,
    depth: SampleCountFlags,
    stencil: SampleCountFlags,
}

impl Sampled {
    pub(super) fn of(stem: &str) -> Sampled {
        Sampled {
            color: profile_counts(stem, "sampledImageColorSampleCounts"),
            integer: profile_counts(stem, "sampledImageIntegerSampleCounts"),
            depth: profile_counts(stem, "sampledImageDepthSampleCounts"),
            stencil: profile_counts(stem, "sampledImageStencilSampleCounts"),
        }
    }
}

/// The limits the profile of `enabled` and `sampled` states for an image of `plan`: any size,
/// level and layer count; one sample for an image that is not a 2D one or is cube compatible
/// (`VkImageFormatProperties::sampleCounts`' rules); else the counts every usage of it takes,
/// each the format's own color attachment counts where the profile states them, or Vulkan's floor
/// for the kind of format (an integer color format's `framebufferIntegerColorSampleCounts` and
/// `sampledImageIntegerSampleCounts`, another color format's `framebufferColorSampleCounts` and
/// `sampledImageColorSampleCounts`, depth's and stencil's, and `storageImageSampleCounts`).
pub(super) fn stated_limits(
    enabled: &DeviceDescription,
    sampled: &Sampled,
    plan: &ImagePlan,
) -> vk::ImageFormatProperties {
    let l = &enabled.limits;
    let block = plan.format.block();
    let depth = block.as_ref().is_some_and(|b| b.depth.is_some());
    let stencil = block.as_ref().is_some_and(|b| b.stencil.is_some());
    let integer = block.as_ref().is_some_and(|b| {
        b.color
            .is_some_and(|c| matches!(c.numeric_type(), NumericType::Uint | NumericType::Sint))
    });
    let own = enabled
        .formats
        .formats
        .get(&plan.format)
        .map(|p| p.color_attachment_sample_counts)
        .filter(|counts| !counts.is_empty());
    let one = SampleCountFlags::from_raw(1);
    let mut counts = SampleCountFlags::from_raw(0x7f);
    if plan.image_type != ImageType::TYPE_2D
        || plan.flags.contains(ImageCreateFlags::CUBE_COMPATIBLE)
    {
        counts = one;
    }
    if plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
        counts &= match own {
            Some(own) => own,
            None if integer => l.framebuffer_integer_color_sample_counts,
            None => l.framebuffer_color_sample_counts,
        };
    }
    if plan
        .usage
        .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
    {
        if depth {
            counts &= l.framebuffer_depth_sample_counts;
        }
        if stencil {
            counts &= l.framebuffer_stencil_sample_counts;
        }
    }
    if plan.usage.contains(ImageUsageFlags::SAMPLED) {
        counts &= match (depth, stencil, integer) {
            (true, _, _) | (_, true, _) => {
                let mut both = SampleCountFlags::from_raw(0x7f);
                if depth {
                    both &= sampled.depth;
                }
                if stencil {
                    both &= sampled.stencil;
                }
                both
            }
            (_, _, true) => sampled.integer,
            _ => sampled.color,
        };
    }
    if plan.usage.contains(ImageUsageFlags::STORAGE) {
        counts &= l.storage_image_sample_counts;
    }
    vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: u32::MAX,
            height: u32::MAX,
            depth: u32::MAX,
        },
        max_mip_levels: 32,
        max_array_layers: u32::MAX,
        sample_counts: vk::SampleCountFlags::from_raw((counts | one).as_raw()),
        max_resource_size: u64::MAX,
    }
}

/// The answer on the device the profile `stem` describes, created as the runtime creates one
/// (`capabilities::created`), with the limits the profile states; and the nine formats' answers.
fn answer(
    stem: &str,
    desc: &DeviceDescription,
) -> (Result<(), NotMsaa32>, Vec<(PixelFormat, FormatAnswers)>) {
    let (enabled, properties) = created(stem, desc);
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let image = ImageFacts {
        profile: compile.profile(),
        formats: &enabled.formats,
        storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: DepthImageFacts::of(&enabled.driver_facts()),
        linear_attachments: false,
        any_views: &crate::image_plan::AnyViews::default(),
    };
    let sampled = Sampled::of(stem);
    let limits = |plan: &ImagePlan| stated_limits(&enabled, &sampled, plan);
    let counts = properties.texture_rules().sample_counts;
    let formats = MSAA_32BIT_INTEGER
        .into_iter()
        .chain(MSAA_32BIT_FLOAT)
        .map(|pf| {
            let sparse = Err(Missing::Gap(RuntimeGap::SparseHeaps));
            (pf, answers(pf, &image, &limits, sparse))
        })
        .collect();
    (msaa_32bit(&image, &limits, counts), formats)
}

/// A refusal of a texture sample count the device property `property` would give.
fn lacks(property: VkProperty) -> Missing {
    Missing::Refused(Unsupported::needs(
        Feature::TextureSampleCount,
        Capability::Property(property),
    ))
}

/// Each committed profile's answer, pinned with its reason:
/// - **Yes** on every venue's profile, each of which states its formats' own color attachment
///   counts: MoltenVK on the M3 (1, 2 and 4 samples, as Metal's own, which answers yes on an M3
///   Max), RADV on the 780M (whose `framebufferIntegerColorSampleCounts` is 1, but whose 32-bit
///   integer formats each take 1 to 8), lavapipe (each takes 1, 4 and 8, its texture counts 1 and
///   4), ANV and NVIDIA; and on the other profiles whose integer formats Vulkan floors at every
///   count their textures take (Intel's and NVIDIA's other profiles).
/// - **No** where a profile states only Vulkan's floor and that floor holds one sample for an
///   integer color attachment (`framebufferIntegerColorSampleCounts` 1): `R32Uint`, the first
///   of the nine, is refused at the profile's lowest count above 1, naming the property. On such
///   a device the answer follows its own limits, which may take more. `roadmap-2022` states no
///   sampled image's counts, so its textures take one sample and no 4.
#[test]
fn every_committed_profile_answers_as_pinned() {
    let int = VkProperty::FramebufferIntegerColorSampleCount;
    let pinned: [(&str, Result<(), NotMsaa32>); 19] = [
        ("amd-780m-hawkpoint-radv", Ok(())),
        ("amd-780m-radv", Ok(())),
        (
            "amd-rdna2-apu-radv",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 2,
                missing: lacks(int(2)),
            }),
        ),
        (
            "amd-rdna3-radv",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 2,
                missing: lacks(int(2)),
            }),
        ),
        ("gate-lavapipe", Ok(())),
        ("gate-lavapipe-x86_64", Ok(())),
        ("intel-arl-anv", Ok(())),
        ("intel-irisxe-anv", Ok(())),
        ("intel-irisxe-rplp-anv", Ok(())),
        ("intel-uhd770-rpls-anv", Ok(())),
        (
            "lavapipe",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 4,
                missing: lacks(int(4)),
            }),
        ),
        (
            "lunarg-desktop-baseline-2023",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 2,
                missing: lacks(int(2)),
            }),
        ),
        ("moltenvk-m3", Ok(())),
        ("nvidia-pascal-linux", Ok(())),
        ("nvidia-rtx50-linux", Ok(())),
        ("nvidia-rtx5080-laptop", Ok(())),
        ("roadmap-2022", Err(NotMsaa32::NoFourSamples)),
        (
            "swiftshader",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 4,
                missing: lacks(int(4)),
            }),
        ),
        (
            "vk12-floor",
            Err(NotMsaa32::Texture {
                format: PixelFormat::R32Uint,
                samples: 4,
                missing: lacks(int(4)),
            }),
        ),
    ];
    let profiles = committed_descriptions();
    let stems: Vec<&str> = profiles.iter().map(|(s, _)| s.as_str()).collect();
    let listed: Vec<&str> = pinned.iter().map(|(s, _)| *s).collect();
    assert_eq!(listed, stems, "every committed profile is pinned, in order");
    for (stem, desc) in &profiles {
        let (got, _) = answer(stem, desc);
        let want = &pinned.iter().find(|(s, _)| s == stem).unwrap().1;
        assert_eq!(&got, want, "{stem}");
    }
}

/// Wherever the answer is yes, the nine formats' capabilities agree with it: each of them has
/// `Msaa` (a 4-sample render target), and each float format `Resolve`
/// ([`Device::pixel_format_capabilities`](crate::Device::pixel_format_capabilities)'s answers,
/// with the same limits, which the presented Apple GPU's column caps: Apple9 lists both), while
/// no integer format resolves, as in Metal.
#[test]
fn a_yes_agrees_with_the_formats_capabilities() {
    let mut yes = 0;
    for (stem, desc) in &committed_descriptions() {
        let (got, formats) = answer(stem, desc);
        if got.is_err() {
            continue;
        }
        yes += 1;
        for (pf, answers) in formats {
            assert_eq!(
                answers.get(FormatCapability::Msaa),
                Ok(()),
                "{stem}: {pf:?}"
            );
            if MSAA_32BIT_FLOAT.contains(&pf) {
                assert_eq!(
                    answers.get(FormatCapability::Resolve),
                    Ok(()),
                    "{stem}: {pf:?}"
                );
            } else {
                assert!(
                    answers.get(FormatCapability::Resolve).is_err(),
                    "{stem}: {pf:?} resolves, as Metal resolves no integer format"
                );
            }
        }
    }
    assert!(yes >= 10, "{yes} profiles answer yes");
}

/// The answer needs 4 samples, and every count above 1 the textures take: a device whose
/// textures take 1 and 2 answers no before any format, and one whose integer formats take 1 and 4
/// where its textures take 1, 2 and 4 answers no at 2.
#[test]
fn the_answer_needs_four_samples_and_every_count_the_textures_take() {
    let (stem, desc) = committed_descriptions()
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .unwrap();
    let (enabled, _) = created(&stem, &desc);
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let image = ImageFacts {
        profile: compile.profile(),
        formats: &enabled.formats,
        storage_multisample: enabled.has(VkFeature::ShaderStorageImageMultisample),
        depth_images: DepthImageFacts::of(&enabled.driver_facts()),
        linear_attachments: false,
        any_views: &crate::image_plan::AnyViews::default(),
    };
    let sampled = Sampled::of(&stem);
    let limits = |plan: &ImagePlan| stated_limits(&enabled, &sampled, plan);
    let counts = |raw: u32| SampleCountFlags::from_raw(raw);
    assert_eq!(msaa_32bit(&image, &limits, counts(0b111)), Ok(()));
    assert_eq!(
        msaa_32bit(&image, &limits, counts(0b011)),
        Err(NotMsaa32::NoFourSamples)
    );
    // The integer formats take 1 and 4 samples alone.
    let no_two = |plan: &ImagePlan| {
        let mut l = stated_limits(&enabled, &sampled, plan);
        if plan.pixel_format == PixelFormat::R32Sint {
            l.sample_counts = vk::SampleCountFlags::from_raw(0b101);
        }
        l
    };
    assert_eq!(
        msaa_32bit(&image, &no_two, counts(0b111)),
        Err(NotMsaa32::Texture {
            format: PixelFormat::R32Sint,
            samples: 2,
            missing: lacks(VkProperty::FramebufferIntegerColorSampleCount(2)),
        })
    );
    // Where the textures take 1 and 4 alone, 2 is not asked.
    assert_eq!(msaa_32bit(&image, &no_two, counts(0b101)), Ok(()));
    // A float format whose single-sample render target the device does not make resolves
    // nowhere.
    let no_single = |plan: &ImagePlan| {
        let mut l = stated_limits(&enabled, &sampled, plan);
        if plan.pixel_format == PixelFormat::RG32Float && plan.samples.as_raw() == 1 {
            l.max_extent = vk::Extent3D::default();
        }
        l
    };
    assert!(
        matches!(
            msaa_32bit(&image, &no_single, counts(0b111)),
            Err(NotMsaa32::Resolve {
                format: PixelFormat::RG32Float,
                ..
            })
        ),
        "{:?}",
        msaa_32bit(&image, &no_single, counts(0b111))
    );
}
