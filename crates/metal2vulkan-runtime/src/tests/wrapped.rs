//! Textures over the caller's images (`texture/wrapped.rs`): what an image must be for a descriptor
//! on every committed profile, each contradiction refused by name, the layout contract and its
//! transitions, the table of moving images, and the ownership rule.

use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use crate::image_barrier::ImageDst;
use crate::image_plan::{plan_image, ImagePlan};
use crate::resource::ResourceKey;
use crate::sharing::Sharing;
use crate::texture::wrapped::{check_layouts, wrap_plan, Transition, Wrapped, WrappedImage};
use crate::texture::Owner;
use crate::texture_descriptor::{texture_usage, TextureDescriptor};
use crate::tracker::{Access, UseRange};
use crate::{VkImageLayouts, WrappedImageProblem};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    AccessFlags, Format, ImageAspectFlags, ImageCreateFlags, ImageLayout, ImageTiling, ImageType,
    ImageUsageFlags, SampleCountFlags,
};
use std::sync::Arc;

const FAMILY: u32 = 0;

/// The runtime's plan for a texture of `texture_type`, `format` and `usage` on `profile`, at the
/// descriptor's sizes.
fn plan_on(
    profile: &str,
    texture_type: TextureType,
    format: PixelFormat,
    usage: TextureUsage,
    samples: u64,
) -> Option<ImagePlan> {
    let (_, desc, device) = committed().into_iter().find(|(n, ..)| n == profile)?;
    plan_of(&facts(&desc, &device), texture_type, format, usage, samples)
}

fn plan_of(
    facts: &crate::image_plan::ImageFacts<'_>,
    texture_type: TextureType,
    format: PixelFormat,
    usage: TextureUsage,
    samples: u64,
) -> Option<ImagePlan> {
    let descriptor = TextureDescriptor {
        texture_type,
        pixel_format: format,
        width: 16,
        height: if texture_type == TextureType::Type1D {
            1
        } else {
            16
        },
        depth: if texture_type == TextureType::Type3D {
            4
        } else {
            1
        },
        sample_count: samples,
        array_length: if texture_type == TextureType::Type2DArray {
            3
        } else {
            1
        },
        mipmap_level_count: if samples > 1 { 1 } else { 2 },
        usage,
        ..TextureDescriptor::default()
    };
    let layout = descriptor.check(APPLE).expect("a valid descriptor");
    plan_image(
        &layout,
        texture_usage(usage, descriptor.swizzle),
        false,
        facts,
    )
    .ok()
}

fn lavapipe(texture_type: TextureType, format: PixelFormat, usage: TextureUsage) -> ImagePlan {
    plan_on("lavapipe", texture_type, format, usage, 1).expect("lavapipe plans it")
}

/// Wrap an image made with `plan`'s own create info, changed by `change`.
fn wrap_changed(
    plan: &ImagePlan,
    change: impl FnOnce(vk::ImageCreateInfo<'_>) -> vk::ImageCreateInfo<'_>,
) -> Result<ImagePlan, WrappedImageProblem> {
    let formats = plan.ash_view_formats();
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = change(plan.create_info(Sharing::new(&[FAMILY]), &formats, &mut list));
    wrap_plan(plan.clone(), &info, &[], Sharing::new(&[FAMILY]))
}

#[test]
fn an_image_made_as_the_runtime_would_wraps_on_every_profile() {
    let cases = [
        (
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            1,
        ),
        (
            TextureType::Type2D,
            PixelFormat::BGRA8Unorm,
            TextureUsage::RENDER_TARGET,
            1,
        ),
        (
            TextureType::Type2DArray,
            PixelFormat::R32Float,
            TextureUsage::SHADER_WRITE,
            1,
        ),
        (
            TextureType::Cube,
            PixelFormat::RGBA16Float,
            TextureUsage::SHADER_READ,
            1,
        ),
        (
            TextureType::Type3D,
            PixelFormat::RGBA8Uint,
            TextureUsage::SHADER_READ,
            1,
        ),
        (
            TextureType::Type2D,
            PixelFormat::Depth32Float,
            TextureUsage::RENDER_TARGET,
            1,
        ),
        (
            TextureType::Type2DMultisample,
            PixelFormat::RGBA8Unorm,
            TextureUsage::RENDER_TARGET,
            4,
        ),
    ];
    let mut wrapped = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for &(texture_type, format, usage, samples) in &cases {
            let Some(plan) = plan_of(&facts, texture_type, format, usage, samples) else {
                continue;
            };
            let got = wrap_changed(&plan, |info| info).expect(&name);
            assert_eq!(got, plan, "{name} {format:?}");
            wrapped += 1;
        }
    }
    assert!(wrapped > 50, "{wrapped}");
}

#[test]
fn each_contradiction_is_refused_by_name() {
    use WrappedImageProblem as P;
    let plan = lavapipe(
        TextureType::Type2D,
        PixelFormat::RGBA8Unorm,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    );
    let refused = |change: fn(vk::ImageCreateInfo<'_>) -> vk::ImageCreateInfo<'_>| {
        wrap_changed(&plan, change).expect_err("refused")
    };
    assert_eq!(
        refused(|i| i.image_type(vk::ImageType::TYPE_3D)),
        P::ImageType {
            image: ImageType::TYPE_3D
        }
    );
    assert_eq!(
        refused(|i| i.format(vk::Format::B8G8R8A8_UNORM)),
        P::Format {
            image: Format::B8G8R8A8_UNORM,
            texture: Format::R8G8B8A8_UNORM
        }
    );
    assert_eq!(
        refused(|i| i.extent(vk::Extent3D {
            width: 16,
            height: 8,
            depth: 1
        })),
        P::Extent {
            image: [16, 8, 1],
            texture: [16, 16, 1]
        }
    );
    assert_eq!(
        refused(|i| i.mip_levels(1)),
        P::Levels {
            image: 1,
            texture: 2
        }
    );
    assert_eq!(
        refused(|i| i.array_layers(2)),
        P::Layers {
            image: 2,
            texture: 1
        }
    );
    assert_eq!(
        refused(|i| i.samples(vk::SampleCountFlags::TYPE_4)),
        P::Samples {
            image: SampleCountFlags::COUNT_4,
            texture: SampleCountFlags::COUNT_1
        }
    );
    assert_eq!(
        refused(|i| i.tiling(vk::ImageTiling::LINEAR)),
        P::Tiling {
            image: ImageTiling::LINEAR
        }
    );
    assert_eq!(
        refused(|i| {
            let flags = i.flags | vk::ImageCreateFlags::PROTECTED;
            i.flags(flags)
        }),
        P::Protected
    );
    // Metal copies every texture with blits: the transfer usages are the texture's too.
    let missing = match refused(|i| {
        i.usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
    }) {
        P::Usage { missing } => missing,
        other => panic!("{other:?}"),
    };
    // Lavapipe's color render target takes input attachment usage, which an image may lack.
    assert!(plan.usage.contains(ImageUsageFlags::INPUT_ATTACHMENT));
    assert_eq!(
        missing,
        plan.usage.difference(
            ImageUsageFlags::COLOR_ATTACHMENT
                | ImageUsageFlags::SAMPLED
                | ImageUsageFlags::INPUT_ATTACHMENT
        )
    );
    assert!(missing.contains(ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST));
    // More usage than the texture needs is the image's own business.
    let more = wrap_changed(&plan, |i| {
        let usage = i.usage | vk::ImageUsageFlags::STORAGE;
        i.usage(usage)
    });
    assert_eq!(more, Ok(plan.clone()));
    // A concurrent image shared with the device's family wraps; one without it does not.
    let formats = plan.ash_view_formats();
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = plan
        .create_info(Sharing::new(&[FAMILY]), &formats, &mut list)
        .sharing_mode(vk::SharingMode::CONCURRENT);
    assert_eq!(
        wrap_plan(plan.clone(), &info, &[1, 2], Sharing::new(&[FAMILY])),
        Err(P::QueueFamily)
    );
    assert_eq!(
        wrap_plan(plan.clone(), &info, &[2, FAMILY], Sharing::new(&[FAMILY])),
        Ok(plan.clone())
    );
    // On a device of several families an exclusive image is refused, as the runtime transfers
    // no ownership between them, and a concurrent one must name every family.
    let several = Sharing::new(&[FAMILY, 1, 2]);
    let exclusive = info.sharing_mode(vk::SharingMode::EXCLUSIVE);
    assert_eq!(
        wrap_plan(plan.clone(), &exclusive, &[], several),
        Err(P::QueueFamily)
    );
    assert_eq!(
        wrap_plan(plan.clone(), &info, &[2, FAMILY], several),
        Err(P::QueueFamily)
    );
    assert_eq!(
        wrap_plan(plan.clone(), &info, &[2, 1, FAMILY], several),
        Ok(plan.clone())
    );
}

/// A color render target's input attachment usage, which the runtime gives it where framebuffer
/// fetch reads input attachments (lavapipe), is the only usage an image may lack: the texture is
/// wrapped without it, and a pass that fetches it renders into a stand-in.
#[test]
fn a_render_target_wraps_without_input_attachment_usage() {
    let mut planned = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(plan) = plan_of(
            &facts,
            TextureType::Type2D,
            PixelFormat::RGBA8Unorm,
            usage,
            1,
        ) else {
            continue;
        };
        let without = plan.usage.difference(ImageUsageFlags::INPUT_ATTACHMENT);
        let got = wrap_changed(&plan, |i| i.usage(without.into())).expect(&name);
        assert_eq!(got.usage, without, "{name}");
        assert_eq!(
            ImagePlan {
                usage: plan.usage,
                ..got
            },
            plan,
            "{name}: only the usage changes"
        );
        planned += usize::from(plan.usage.contains(ImageUsageFlags::INPUT_ATTACHMENT));
    }
    assert!(
        planned > 0,
        "a committed profile fetches through input attachments"
    );
}

/// A multisampled depth render target's custom sample location flag, which the runtime gives it
/// where the device takes custom positions at its count, may be missing too: the texture is
/// wrapped without it, and a pass at custom positions renders it through a stand-in made with it.
#[test]
fn a_multisampled_depth_target_wraps_without_the_sample_location_flag() {
    let located = ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT;
    let mut planned = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let Some(plan) = plan_of(
            &facts,
            TextureType::Type2DMultisample,
            PixelFormat::Depth32Float,
            TextureUsage::RENDER_TARGET,
            4,
        ) else {
            continue;
        };
        let without = plan.flags.difference(located);
        let got = wrap_changed(&plan, |i| i.flags(without.into())).expect(&name);
        assert_eq!(got.flags, without, "{name}");
        assert_eq!(
            ImagePlan {
                flags: plan.flags,
                ..got
            },
            plan,
            "{name}: only the flags change"
        );
        assert_eq!(wrap_changed(&plan, |i| i).expect(&name), plan, "{name}");
        planned += usize::from(plan.flags.contains(located));
    }
    assert!(
        planned > 0,
        "a committed profile takes custom positions at 4 samples"
    );
}

#[test]
fn a_cube_needs_cube_compatibility() {
    let plan = lavapipe(
        TextureType::Cube,
        PixelFormat::RGBA8Unorm,
        TextureUsage::SHADER_READ,
    );
    let got = wrap_changed(&plan, |i| {
        let flags = i.flags & !vk::ImageCreateFlags::CUBE_COMPATIBLE;
        i.flags(flags)
    });
    assert_eq!(
        got,
        Err(WrappedImageProblem::Flags {
            missing: ImageCreateFlags::CUBE_COMPATIBLE
        })
    );
}

#[test]
fn an_image_not_mutable_is_viewed_in_its_own_format_only() {
    // The runtime's own image of RGBA8Unorm is mutable, for its sRGB twin's views.
    let plan = lavapipe(
        TextureType::Type2D,
        PixelFormat::RGBA8Unorm,
        TextureUsage::SHADER_READ,
    );
    assert!(plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT));
    assert_eq!(
        plan.view_formats,
        [Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB]
    );
    let got = wrap_changed(&plan, |i| {
        let flags = i.flags & !vk::ImageCreateFlags::MUTABLE_FORMAT;
        i.flags(flags)
    })
    .expect("an immutable image wraps");
    assert!(!got.flags.contains(ImageCreateFlags::MUTABLE_FORMAT));
    assert!(got.view_formats.is_empty());
    assert_eq!(
        ImagePlan {
            flags: plan.flags.difference(ImageCreateFlags::MUTABLE_FORMAT),
            view_formats: Vec::new(),
            ..plan.clone()
        },
        got
    );
}

#[test]
fn a_storage_view_in_another_format_needs_extended_usage() {
    // Where the device stores `RGBA8Unorm_sRGB` only through its UNORM alias, the image is
    // extended-usage and mutable: an image without either cannot be that texture.
    let mut checked = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let Some(plan) = plan_of(
            &facts,
            TextureType::Type2D,
            PixelFormat::RGBA8UnormSRGB,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            1,
        ) else {
            continue;
        };
        if !plan.flags.contains(ImageCreateFlags::EXTENDED_USAGE) {
            continue;
        }
        let got = wrap_changed(&plan, |i| {
            i.flags(
                i.flags
                    & !(vk::ImageCreateFlags::EXTENDED_USAGE
                        | vk::ImageCreateFlags::MUTABLE_FORMAT),
            )
        });
        assert_eq!(
            got,
            Err(WrappedImageProblem::Flags {
                missing: ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE
            }),
            "{name}"
        );
        checked += 1;
    }
    assert!(checked > 0, "a committed profile stores sRGB through UNORM");
}

#[test]
fn the_layout_contract_takes_layouts_the_image_allows() {
    use ImageLayout as L;
    let color = ImageAspectFlags::COLOR;
    let depth = ImageAspectFlags::DEPTH;
    let attachment = ImageUsageFlags::COLOR_ATTACHMENT
        | ImageUsageFlags::TRANSFER_SRC
        | ImageUsageFlags::TRANSFER_DST;
    let ok = |initial, home, usage, aspects| {
        check_layouts(VkImageLayouts::new(initial, home), usage, aspects)
    };
    let refused = |layout| Err(WrappedImageProblem::Layout { layout });
    // A swapchain image: new, then presented between batches.
    assert_eq!(
        ok(L::UNDEFINED, L::PRESENT_SRC_KHR, attachment, color),
        Ok(())
    );
    assert_eq!(ok(L::PREINITIALIZED, L::GENERAL, attachment, color), Ok(()));
    assert_eq!(
        ok(L::GENERAL, L::GENERAL, ImageUsageFlags::empty(), color),
        Ok(())
    );
    for home in [
        L::COLOR_ATTACHMENT_OPTIMAL,
        L::TRANSFER_SRC_OPTIMAL,
        L::TRANSFER_DST_OPTIMAL,
    ] {
        assert_eq!(ok(home, home, attachment, color), Ok(()), "{home:?}");
    }
    // `UNDEFINED` and `PREINITIALIZED` only start the contract.
    assert_eq!(
        ok(L::GENERAL, L::UNDEFINED, attachment, color),
        refused(L::UNDEFINED)
    );
    assert_eq!(
        ok(L::GENERAL, L::PREINITIALIZED, attachment, color),
        refused(L::PREINITIALIZED)
    );
    // Each layout needs the usage or aspect Vulkan's barriers require of it.
    assert_eq!(
        ok(L::UNDEFINED, L::SHADER_READ_ONLY_OPTIMAL, attachment, color),
        refused(L::SHADER_READ_ONLY_OPTIMAL)
    );
    assert_eq!(
        ok(
            L::UNDEFINED,
            L::TRANSFER_SRC_OPTIMAL,
            ImageUsageFlags::SAMPLED,
            color
        ),
        refused(L::TRANSFER_SRC_OPTIMAL)
    );
    assert_eq!(
        ok(L::UNDEFINED, L::PRESENT_SRC_KHR, attachment, depth),
        refused(L::PRESENT_SRC_KHR)
    );
    assert_eq!(
        ok(
            L::UNDEFINED,
            L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth
        ),
        Ok(())
    );
    assert_eq!(
        ok(
            L::COLOR_ATTACHMENT_OPTIMAL,
            L::GENERAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth
        ),
        refused(L::COLOR_ATTACHMENT_OPTIMAL)
    );
    // A layout the runtime does not move images through.
    assert_eq!(
        ok(L::UNDEFINED, L::SHARED_PRESENT_KHR, attachment, color),
        refused(L::SHARED_PRESENT_KHR)
    );
}

fn image(layouts: VkImageLayouts) -> WrappedImage {
    WrappedImage::new(
        ResourceKey::new(),
        vk::Image::null(),
        ImageAspectFlags::COLOR,
        2,
        3,
        layouts,
        "the test's image".to_owned(),
        None,
    )
}

fn transition(old: ImageLayout, new: ImageLayout) -> Option<Transition> {
    Some(Transition {
        image: vk::Image::null(),
        aspects: ImageAspectFlags::COLOR,
        old,
        new,
    })
}

#[test]
fn each_batch_moves_the_image_to_general_and_back_home() {
    use ImageLayout as L;
    let swapchain = image(VkImageLayouts::new(L::UNDEFINED, L::PRESENT_SRC_KHR));
    // The first batch finds it in `initial`.
    assert_eq!(
        swapchain.transitions(),
        (
            transition(L::UNDEFINED, L::GENERAL),
            transition(L::GENERAL, L::PRESENT_SRC_KHR)
        )
    );
    // Planning a batch changes nothing until it is submitted.
    assert_eq!(
        swapchain.transitions().0,
        transition(L::UNDEFINED, L::GENERAL)
    );
    swapchain.submitted();
    for _ in 0..2 {
        assert_eq!(
            swapchain.transitions(),
            (
                transition(L::PRESENT_SRC_KHR, L::GENERAL),
                transition(L::GENERAL, L::PRESENT_SRC_KHR)
            )
        );
        swapchain.submitted();
    }
    // An image that starts in `GENERAL` and goes home elsewhere moves only out at first.
    let read_back = image(VkImageLayouts::new(L::GENERAL, L::TRANSFER_SRC_OPTIMAL));
    assert_eq!(
        read_back.transitions(),
        (None, transition(L::GENERAL, L::TRANSFER_SRC_OPTIMAL))
    );
    // A home in `GENERAL` moves only in, once.
    let adopted = image(VkImageLayouts::new(L::TRANSFER_DST_OPTIMAL, L::GENERAL));
    assert_eq!(
        adopted.transitions(),
        (transition(L::TRANSFER_DST_OPTIMAL, L::GENERAL), None)
    );
    adopted.submitted();
    assert_eq!(adopted.transitions(), (None, None));
    assert!(!VkImageLayouts::general().moves());
    assert!(VkImageLayouts::new(L::GENERAL, L::PRESENT_SRC_KHR).moves());
    assert!(VkImageLayouts::new(L::UNDEFINED, L::GENERAL).moves());
}

#[test]
fn an_image_resting_out_of_general_names_its_home_in_its_notice() {
    use ImageLayout as L;
    let swapchain = image(VkImageLayouts::new(L::UNDEFINED, L::PRESENT_SRC_KHR));
    assert!(swapchain.rests_out_of_general());
    let (object, reason) = swapchain.notice();
    assert_eq!(object, "the test's image");
    assert!(
        reason.contains("PRESENT_SRC_KHR") && reason.contains("2 whole-image barriers per batch"),
        "{reason}"
    );
    // Moved into `GENERAL` once and kept there: no notice.
    let adopted = image(VkImageLayouts::new(L::TRANSFER_DST_OPTIMAL, L::GENERAL));
    assert!(!adopted.rests_out_of_general());
}

#[test]
fn a_transition_is_one_barrier_over_the_whole_image() {
    let family = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    let barrier = transition(ImageLayout::UNDEFINED, ImageLayout::GENERAL)
        .unwrap()
        .barrier(family);
    assert_eq!(
        ImageLayout::from(barrier.old_layout),
        ImageLayout::UNDEFINED
    );
    assert_eq!(ImageLayout::from(barrier.new_layout), ImageLayout::GENERAL);
    assert_eq!(
        AccessFlags::from(barrier.src_access_mask),
        AccessFlags::MEMORY_WRITE
    );
    assert_eq!(
        barrier.dst_access_mask.as_raw(),
        ImageDst::Every.access(family).as_raw()
    );
    assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    assert_eq!(barrier.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    let range = barrier.subresource_range;
    assert_eq!(
        ImageAspectFlags::from(range.aspect_mask),
        ImageAspectFlags::COLOR
    );
    assert_eq!(
        (range.base_mip_level, range.level_count),
        (0, vk::REMAINING_MIP_LEVELS)
    );
    assert_eq!(
        (range.base_array_layer, range.layer_count),
        (0, vk::REMAINING_ARRAY_LAYERS)
    );
}

#[test]
fn a_transition_is_a_tracked_write_of_every_subresource() {
    let wrapped = image(VkImageLayouts::new(
        ImageLayout::UNDEFINED,
        ImageLayout::PRESENT_SRC_KHR,
    ));
    let used = wrapped.transition_use();
    assert!(used.tracked);
    assert_eq!(
        used.range,
        UseRange::Image {
            aspects: vk::ImageAspectFlags::COLOR,
            levels: 0..2,
            layers: 0..3,
        }
    );
    assert_eq!(used.read, Access::NONE);
    assert_eq!(
        used.write,
        Access::new(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE
        )
    );
}

#[test]
fn the_table_finds_the_moving_images_among_a_batchs_keys() {
    let table = Wrapped::default();
    let first = Arc::new(image(VkImageLayouts::new(
        ImageLayout::UNDEFINED,
        ImageLayout::PRESENT_SRC_KHR,
    )));
    let second = Arc::new(image(VkImageLayouts::new(
        ImageLayout::GENERAL,
        ImageLayout::TRANSFER_SRC_OPTIMAL,
    )));
    let other = ResourceKey::new();
    assert!(table.among([first.key, other]).is_empty());
    table.insert(&first);
    table.insert(&second);
    let found = table.among([second.key, other, first.key, second.key]);
    let keys: Vec<ResourceKey> = found.iter().map(|image| image.key).collect();
    let mut expected = vec![first.key, second.key];
    expected.sort();
    assert_eq!(keys, expected);
    table.forget(&first);
    let found = table.among([first.key, second.key]);
    assert_eq!(found.len(), 1);
    assert!(Arc::ptr_eq(&found[0], &second));
}

#[test]
fn a_wrapped_image_is_never_destroyed_or_freed() {
    let caller = Owner::Caller(Arc::new(image(VkImageLayouts::general())));
    let (destroyed, freed) = caller.releases();
    assert!(!destroyed && freed.is_none());
}

/// A texture over one plane of the caller's multi-planar image
/// (`Device::new_texture_with_vk_image_plane`): planned in the descriptor's format at the plane's
/// extent (the registry's divisors), with the plane's aspect and the image's shader usages alone,
/// mutable and of extended usage; each contradiction refused by name.
#[test]
fn a_plane_of_a_multi_planar_image_is_planned_by_its_plane() {
    use crate::texture::wrapped::{plane_plan, planes_of};
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .expect("the lavapipe profile");
    let facts = facts(&desc, &device);
    let plan = |format: PixelFormat, side: u64, usage: TextureUsage| {
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format: format,
            width: side,
            height: side,
            usage,
            ..TextureDescriptor::default()
        };
        let layout = descriptor.check(APPLE).expect("a valid descriptor");
        plan_image(
            &layout,
            texture_usage(usage, descriptor.swizzle),
            false,
            &facts,
        )
        .unwrap()
    };
    let read = TextureUsage::SHADER_READ;
    let image = |format: vk::Format| {
        vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
    };
    let sharing = Sharing::new(&[FAMILY]);
    let nv12 = image(vk::Format::G8_B8R8_2PLANE_420_UNORM);
    let (luma, aspect) = plane_plan(
        plan(PixelFormat::R8Unorm, 16, read),
        &nv12,
        0,
        read,
        &[],
        sharing,
    )
    .unwrap();
    assert_eq!(
        (luma.format, luma.extent, aspect),
        (Format::R8_UNORM, [16, 16, 1], ImageAspectFlags::PLANE_0)
    );
    assert_eq!(luma.usage, ImageUsageFlags::SAMPLED);
    assert_eq!(
        luma.flags,
        ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE
    );
    assert!(luma.view_formats.is_empty());
    let chroma = plan(PixelFormat::RG8Unorm, 8, read);
    let (planned, aspect) = plane_plan(chroma.clone(), &nv12, 1, read, &[], sharing).unwrap();
    assert_eq!(
        (planned.format, planned.extent, aspect),
        (Format::R8G8_UNORM, [8, 8, 1], ImageAspectFlags::PLANE_1)
    );
    // P010's chroma plane, viewed as `RG16Unorm`: one compatibility class.
    let p010 = image(vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16);
    let rg16 = plan(PixelFormat::RG16Unorm, 8, read);
    assert!(plane_plan(rg16, &p010, 1, read, &[], sharing).is_ok());
    use WrappedImageProblem as P;
    let refused = |plan: ImagePlan, info: &vk::ImageCreateInfo<'_>, plane, usage| {
        plane_plan(plan, info, plane, usage, &[], sharing).unwrap_err()
    };
    assert_eq!(
        refused(chroma.clone(), &nv12, 2, read),
        P::Plane {
            plane: 2,
            planes: 2
        }
    );
    let rgba = image(vk::Format::R8G8B8A8_UNORM);
    assert_eq!(
        refused(chroma.clone(), &rgba, 0, read),
        P::Plane {
            plane: 0,
            planes: 0
        }
    );
    assert_eq!(
        refused(plan(PixelFormat::R8Unorm, 8, read), &nv12, 1, read),
        P::PlaneFormat {
            plane: Format::R8G8_UNORM,
            texture: Format::R8_UNORM
        }
    );
    assert_eq!(
        refused(plan(PixelFormat::RG8Unorm, 16, read), &nv12, 1, read),
        P::Extent {
            image: [8, 8, 1],
            texture: [16, 16, 1]
        }
    );
    let fixed = nv12.flags(vk::ImageCreateFlags::MUTABLE_FORMAT);
    assert_eq!(
        refused(chroma.clone(), &fixed, 1, read),
        P::Flags {
            missing: ImageCreateFlags::EXTENDED_USAGE
        }
    );
    let write = read | TextureUsage::SHADER_WRITE;
    assert_eq!(
        refused(plan(PixelFormat::RG8Unorm, 8, write), &nv12, 1, write),
        P::Usage {
            missing: ImageUsageFlags::STORAGE
        }
    );
    let linear = nv12.tiling(vk::ImageTiling::LINEAR);
    assert_eq!(
        refused(chroma.clone(), &linear, 1, read),
        P::Tiling {
            image: ImageTiling::LINEAR
        }
    );
    // A concurrent image that leaves out one of the device's families.
    let concurrent = nv12.sharing_mode(vk::SharingMode::CONCURRENT);
    assert_eq!(
        plane_plan(
            chroma,
            &concurrent,
            1,
            read,
            &[1, 2],
            Sharing::new(&[FAMILY, 1])
        ),
        Err(P::QueueFamily)
    );
    // The table: a two-plane format's name says two planes, a three-plane one's three.
    for raw in 1_000_156_000..1_000_156_034 {
        let format = Format::from(vk::Format::from_raw(raw));
        let name = format!("{format:?}");
        let planes = planes_of(format).len();
        let named = [("_2PLANE_", 2), ("_3PLANE_", 3)]
            .into_iter()
            .find(|(part, _)| name.contains(part))
            .map_or(0, |(_, count)| count);
        assert_eq!(planes, named, "{name}");
    }
}
