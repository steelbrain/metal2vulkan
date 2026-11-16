//! [`Texture::view_placement`] against the views made: for textures of several formats and usages
//! (every pair a sweep of view formats meets, Metal's refusals among them), the query's answer is
//! what [`Texture::new_texture_view`] then makes, read by what the making moved and hands out: a
//! view shadow counts in `WorkCounts::view_shadows`, a view through a swizzle in
//! `WorkCounts::views_through_swizzle`, a Vulkan view of the texture's image counts in neither and
//! hands out the texture's `VkImage`; a refused view is refused as the query was. The query itself
//! moves no count. The consumer's cases are named: a render target's views in its own format, its
//! sRGB twin and its channel-order sibling are views of its image, and another format of its class
//! without `PixelFormatView` usage is a shadow, for the image's list. A texture over a buffer's
//! bytes whose image takes no view of the format answers storage over the bytes, no shadow of the
//! image.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{
    Device, Error, ResourceOptions, Texture, TextureDescriptor, ViewPlacement, ViewShadowReason,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::collections::BTreeMap;

use PixelFormat as P;

/// The textures' formats.
const TEXTURES: [PixelFormat; 10] = [
    P::BGRA8Unorm,
    P::BGRA8UnormSRGB,
    P::RGBA8Unorm,
    P::R8Unorm,
    P::A8Unorm,
    P::RG8Unorm,
    P::R16Float,
    P::R32Float,
    P::RGB10A2Unorm,
    P::RGBA16Float,
];

/// The views' formats: of each size the textures take, color formats of several classes.
const VIEWS: [PixelFormat; 24] = [
    P::R8Unorm,
    P::A8Unorm,
    P::R8Uint,
    P::R8Snorm,
    P::RG8Unorm,
    P::R16Float,
    P::R16Uint,
    P::B5G6R5Unorm,
    P::RGBA8Unorm,
    P::RGBA8UnormSRGB,
    P::BGRA8Unorm,
    P::BGRA8UnormSRGB,
    P::RGBA8Uint,
    P::R32Uint,
    P::R32Float,
    P::RG16Float,
    P::RGB10A2Unorm,
    P::RG11B10Float,
    P::RGB9E5Float,
    P::BGR10A2Unorm,
    P::RGBA16Float,
    P::RG32Float,
    P::RGBA16Uint,
    P::RGBA32Float,
];

/// The textures' usages.
fn usages() -> [TextureUsage; 5] {
    let read = TextureUsage::SHADER_READ;
    let render = TextureUsage::RENDER_TARGET;
    let views = TextureUsage::PIXEL_FORMAT_VIEW;
    [
        read,
        render | read,
        read | views,
        render | read | views,
        read | TextureUsage::SHADER_WRITE | views,
    ]
}

/// An 8x8 `Private` texture of `pf` and `usage`.
fn descriptor(pf: PixelFormat, usage: TextureUsage) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(pf, 8, 8, false);
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// The counts a view's making moves: its view shadows and its views through a swizzle.
fn counts(device: &Device) -> [u64; 2] {
    let counts = device.work_counts();
    [counts.view_shadows, counts.views_through_swizzle]
}

/// What the view in `pf` of `texture` was made as, by the counts its making moved and the image
/// it hands out, a refusal as its error's text. A shadow's reason is the query's, by construction,
/// so a made shadow names none (`NotListed` stands for any).
fn made(device: &Device, texture: &Texture, pf: PixelFormat) -> Result<ViewPlacement, String> {
    let before = counts(device);
    let view = texture
        .new_texture_view(pf)
        .map_err(|error| format!("{error:?}"))?;
    let after = counts(device);
    let moved = [after[0] - before[0], after[1] - before[1]];
    Ok(match moved {
        [1, 0] => ViewPlacement::Shadow(ViewShadowReason::NotListed),
        [0, 1] => ViewPlacement::ImageThroughSwizzle,
        [0, 0] => {
            // SAFETY: the images stay the textures'; nothing uses them.
            let (own, viewed) = unsafe { (texture.vk_image(), view.vk_image()) };
            assert!(
                own.is_some(),
                "{pf:?}: a private texture hands out no image"
            );
            match own == viewed {
                true => ViewPlacement::Image,
                false => ViewPlacement::OverBytes,
            }
        }
        moved => panic!("{pf:?}: a view's making moved the counts by {moved:?}"),
    })
}

/// The query's answer in `made`'s terms.
fn as_made(answer: &Result<ViewPlacement, Error>) -> Result<ViewPlacement, String> {
    match answer {
        Ok(ViewPlacement::Shadow(_)) => Ok(ViewPlacement::Shadow(ViewShadowReason::NotListed)),
        Ok(placement) => Ok(*placement),
        Err(error) => Err(format!("{error:?}")),
    }
}

/// `texture`'s answer for a view in `pf`, checked to move no count and to be what the view made
/// then is; the answer.
fn answered(device: &Device, texture: &Texture, pf: PixelFormat) -> Result<ViewPlacement, Error> {
    let before = counts(device);
    let answer = texture.view_placement(pf);
    assert_eq!(counts(device), before, "{pf:?}: the query moved a count");
    let made = made(device, texture, pf);
    assert_eq!(
        as_made(&answer),
        made,
        "{pf:?}: the query's answer {answer:?}"
    );
    answer
}

#[test]
fn a_view_s_placement_answer_is_the_view_made_for_every_format_and_usage() {
    let scratch = Scratch::new("view-placement");
    let device = device(&scratch.0);
    let mut seen: BTreeMap<String, u64> = BTreeMap::new();
    for pf in TEXTURES {
        for usage in usages() {
            for view in VIEWS {
                // A fresh texture per view: a handed-out image, or a view made, changes nothing
                // of the next view's answer. A texture the device does not make for the usage
                // has no views to ask of.
                let Ok(texture) = device.new_texture(&descriptor(pf, usage)) else {
                    break;
                };
                let answer = answered(&device, &texture, view);
                let kind = match &answer {
                    Ok(placement) => format!("{placement:?}"),
                    Err(error) => format!("refused {}", error_kind(error)),
                };
                *seen.entry(kind).or_default() += 1;
            }
        }
    }
    eprintln!("VIEW-PLACEMENT: answers met {seen:?}");
    for kind in ["Image", "ImageThroughSwizzle", "Shadow(NotListed)"] {
        assert!(seen.contains_key(kind), "no view answered {kind}: {seen:?}");
    }
}

/// The variant of `error`, without its text.
fn error_kind(error: &Error) -> &'static str {
    match error {
        Error::Invalid(_) => "Invalid",
        Error::Nil(_) => "Nil",
        Error::Unsupported(_) => "Unsupported",
        Error::NotImplemented(_) => "NotImplemented",
        _ => "other",
    }
}

#[test]
fn a_plane_cache_s_views_are_views_of_its_image_and_another_format_of_its_class_a_shadow() {
    let scratch = Scratch::new("view-placement-plane");
    let device = device(&scratch.0);
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let cases = [
        (P::BGRA8Unorm, usage, P::BGRA8Unorm, ViewPlacement::Image),
        (
            P::BGRA8Unorm,
            usage,
            P::BGRA8UnormSRGB,
            ViewPlacement::Image,
        ),
        (P::BGRA8Unorm, usage, P::RGBA8Unorm, ViewPlacement::Image),
        (
            P::BGRA8Unorm,
            usage,
            P::RGBA8UnormSRGB,
            ViewPlacement::Image,
        ),
        (
            P::BGRA8Unorm,
            usage,
            P::R32Uint,
            ViewPlacement::Shadow(ViewShadowReason::NotListed),
        ),
        (
            P::RGBA8Unorm,
            TextureUsage::SHADER_READ,
            P::BGRA8Unorm,
            ViewPlacement::Shadow(ViewShadowReason::NotListed),
        ),
        (
            P::RGBA8Unorm,
            usage | TextureUsage::PIXEL_FORMAT_VIEW,
            P::R32Uint,
            ViewPlacement::Image,
        ),
        (
            P::R8Unorm,
            TextureUsage::SHADER_READ,
            P::A8Unorm,
            ViewPlacement::ImageThroughSwizzle,
        ),
    ];
    for (pf, usage, view, want) in cases {
        let texture = device.new_texture(&descriptor(pf, usage)).unwrap();
        let what = format!("a {view:?} view of a {pf:?} texture of {usage:?}");
        assert_eq!(answered(&device, &texture, view), Ok(want), "{what}");
    }
    // Over a buffer's bytes, a view its image takes no view of is storage over the bytes.
    let buffer = device
        .new_buffer(8 * 256, ResourceOptions::shared())
        .unwrap();
    let mut over = descriptor(P::RGBA8Unorm, TextureUsage::SHADER_READ);
    over.resource_options = ResourceOptions::shared();
    let texture = buffer.new_texture(&over, 0, 256).unwrap();
    let before = counts(&device);
    assert_eq!(
        texture.view_placement(P::R32Uint),
        Ok(ViewPlacement::OverBytes)
    );
    texture.new_texture_view(P::R32Uint).unwrap();
    assert_eq!(
        counts(&device),
        before,
        "a view over the bytes is no view shadow"
    );
}
