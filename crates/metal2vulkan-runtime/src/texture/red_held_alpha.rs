//! An `A8Unorm` texture over a buffer's bytes whose image is an `R8Unorm` texture's, read
//! through `(Zero, Zero, Zero, Red)`: where the device's linear `A8_UNORM` image lacks what its
//! optimal image does for the texture's usage, and its linear `R8_UNORM` image lacks nothing.
//!
//! - **Why.** A texture over a buffer's bytes is a linear image over them where the device makes
//!   one that does all its optimal image does (`linear_plan::linear_lack`), else a shadow the
//!   encoders copy the bytes into and back (`shadow.rs`). NVIDIA's drivers report `A8_UNORM`'s
//!   optimal tiling with sampling, linear filtering and transfers, and its linear tiling (and its
//!   `DRM_FORMAT_MOD_LINEAR` modifier) with none of them, where `R8_UNORM`'s linear tiling has
//!   them all (`vkGetPhysicalDeviceFormatProperties2`'s `linearTilingFeatures` and
//!   `optimalTilingFeatures`): every glyph-sized `A8Unorm` texture over a guest's `Shared`
//!   buffers was a shadow there.
//! - **What.** The texture's storage is the one an `R8Unorm` texture of the same descriptor makes
//!   over the same bytes (`Device::storage_over_buffer`): the image in place over host pages, an
//!   image aliasing the buffer's memory, or a shadow where that texture would be one. Its record
//!   reads that image in `R8Unorm`'s format through the swizzle `metal::format::swizzle_over`
//!   composes with the texture's own (`ViewRecord::read_as`), as an `A8Unorm` view of an
//!   `R8Unorm` texture reads its image (`texture/view_shadow.rs`): the same byte a texel, sampled
//!   through the same filter, the channels moved by the view's component mapping, which Vulkan
//!   applies to gathered and border texels as to any other ("Texel Input Operations"), so its
//!   reads, samples and gathers are Metal's `(0, 0, 0, a)`. Its copies and CPU access move the
//!   bytes, which are the same in either format. Counted:
//!   `WorkCounts::alpha_textures_held_in_red`.
//! - **Which.** A 2D texture (or array) of one level and one sample that only reads: no
//!   `RenderTarget`, `ShaderWrite`, `ShaderAtomic` or `BlockWritesOnly` usage (`Unknown` usage
//!   takes them all), its alpha not read as one and not rotated, on a device that takes a view of
//!   the composed mapping (`swizzle_over`). A texture that renders or is written keeps the
//!   `A8_UNORM` route. A device that holds no `A8_UNORM` image at all already holds every
//!   `A8Unorm` texture in `R8_UNORM` (`metal::format::alpha_in_red`), and its plan lacks nothing
//!   here.

use super::super::image_plan::{plan_image, ImageFacts};
use super::super::linear_plan::linear_lack;
use super::super::texture_descriptor::{texture_usage, TextureDescriptor, TextureLayout};
use metal2vulkan::metal::format::{alpha_in_red_format, direct_format, swizzle_over};
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels, TextureType, TextureUsage};

/// An `A8Unorm` texture made over a buffer as an `R8Unorm` texture's storage: that texture's
/// descriptor and layout, and the format and swizzle its bindings read the image through.
pub(super) struct RedHeld {
    pub(super) descriptor: TextureDescriptor,
    pub(super) layout: TextureLayout,
    pub(super) read_as: (PixelFormat, TextureSwizzleChannels),
}

/// The usages that write a texture: the texture keeps its own format's route.
const WRITES: TextureUsage = TextureUsage::RENDER_TARGET
    .union(TextureUsage::SHADER_WRITE)
    .union(TextureUsage::SHADER_ATOMIC)
    .union(TextureUsage::BLOCK_WRITES_ONLY);

/// Whether a texture of `descriptor` (checked as `layout`) over a buffer's bytes is made as the
/// format its alpha is held in the red of (the module docs), on the device of `facts`: that
/// texture's descriptor and layout and the read it takes; `None` where it keeps its own route.
pub(super) fn red_held_alpha(
    descriptor: &TextureDescriptor,
    layout: &TextureLayout,
    facts: &ImageFacts<'_>,
) -> Option<RedHeld> {
    let pf = layout.pixel_format;
    let red = alpha_in_red_format(pf)?;
    let usage = texture_usage(descriptor.usage, descriptor.swizzle);
    let one = matches!(
        layout.texture_type,
        TextureType::Type2D | TextureType::Type2DArray
    ) && (layout.levels, layout.samples) == (1, 1);
    if !one || usage.intersects(WRITES) || descriptor.alpha_reads_one || descriptor.rotation != 0 {
        return None;
    }
    let own =
        (PixelFormat::ALL.iter().copied()).find(|&other| direct_format(other) == Some(red))?;
    // The texture's own plan's linear image lacks what its optimal one does, and the red
    // format's lacks nothing.
    let asked = plan_image(layout, usage, false, facts).ok()?;
    linear_lack(&asked, facts)?;
    let held = TextureLayout {
        pixel_format: own,
        ..*layout
    };
    let plan = plan_image(&held, usage, false, facts).ok()?;
    if linear_lack(&plan, facts).is_some() {
        return None;
    }
    let over = swizzle_over(pf, descriptor.swizzle, own, facts.profile)?;
    Some(RedHeld {
        descriptor: TextureDescriptor {
            pixel_format: own,
            swizzle: TextureSwizzleChannels::DEFAULT,
            ..*descriptor
        },
        layout: held,
        read_as: (own, over),
    })
}
