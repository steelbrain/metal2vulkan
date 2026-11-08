//! The shadow log: a default-off diagnostic, `METAL2VULKAN_LOG_SHADOWS` (`metal2vulkan::env_vars`),
//! that names each shadow a device makes of a texture, and why no image over its truth serves.
//!
//! - **What it prints.** One line per shadow as it is made, through the device's message sink at
//!   `Info` severity (id name `metal2vulkan-shadows`), else on stderr: the Metal call that made
//!   it (`new_texture_with_iosurface_plane` for an IOSurface plane's, the binding's or the pass's
//!   call for a shadow made at its first use), the texture's format, size, levels and usage, and
//!   its truth: a buffer's bytes (the buffer's storage mode, length and object ID,
//!   `Buffer::object_id`, and the texels' offset and row pitch) or the texture's own image; then
//!   what made it a shadow. Two textures over the same bytes name the same object ID and offset.
//! - **Which shadows.** A texture over a buffer's bytes made a shadow (with the reason its notice
//!   gives, `shadowed-texture`, announced once a device); a usage shadow, for a binding beyond
//!   the texture's usage or for a draw that samples a color attachment of its own render pass
//!   instance (`texture/usage_shadow.rs`), naming whether the texture's own image aliases the
//!   bytes and took the feedback loop usage; a rotated texture's image of its stored frame and
//!   its twin (`texture/rotation.rs`); a two-plane extended-range texture's
//!   (`texture/surface_two_planes.rs`); a depth or stencil texture's tiled twin over a surface
//!   plane (`texture/surface_depth.rs`); and a YCbCr texture's converted image and its render
//!   shadow (`texture/ycbcr_image.rs`; its write shadow is a usage shadow). A device copy is no shadow it names, nor is a heap texture's
//!   (its own notice names it), a view's shadow of an image or a texture buffer's staging.
//! - **Off, it costs nothing.** The switch is read once; no line is formatted.

use super::buffer_texture::Texels;
use super::driver_facts::inform_through;
use super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::Device;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::OnceLock;

/// The id name of the log's messages.
const ID_NAME: &str = "metal2vulkan-shadows";

/// Whether the log is on: read once.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_shadows)
}

/// What a line names of a shadow's texture: its format, width and height, levels and usage.
pub(super) struct Shadowed {
    pub(super) pixel_format: PixelFormat,
    pub(super) extent: [u32; 2],
    pub(super) levels: u32,
    pub(super) usage: TextureUsage,
}

impl Shadowed {
    /// A texture of `descriptor`, checked as `layout`.
    pub(super) fn of(descriptor: &TextureDescriptor, layout: &TextureLayout) -> Shadowed {
        let [width, height, _] = layout.extent;
        Shadowed {
            pixel_format: descriptor.pixel_format,
            extent: [width, height],
            levels: layout.levels,
            usage: descriptor.usage,
        }
    }
}

/// Where the log is on, say that `call` made a shadow of a texture of `descriptor` (checked as
/// `layout`) over `texels`, for `why` (the module docs).
pub(super) fn made(
    device: &Device,
    call: &str,
    (descriptor, layout): (&TextureDescriptor, &TextureLayout),
    texels: &Texels,
    why: &str,
) {
    made_of(
        device,
        call,
        &Shadowed::of(descriptor, layout),
        Some(texels),
        why,
    );
}

/// Where the log is on, say that `call` made a shadow of `texture`, over `texels` where its truth
/// is a buffer's bytes, else over the texture's own image, for `why` (the module docs).
pub(super) fn made_of(
    device: &Device,
    call: &str,
    texture: &Shadowed,
    texels: Option<&Texels>,
    why: &str,
) {
    if !enabled() {
        return;
    }
    let [width, height] = texture.extent;
    let truth = match texels {
        Some(texels) => {
            let buffer = &texels.buffer;
            let storage = match buffer.resource_options().storage_mode() {
                Ok(mode) => format!("{mode:?}"),
                Err(_) => "unknown".to_owned(),
            };
            format!(
                "a {storage} buffer, object {}, of {} bytes at offset {} with rows of {} bytes",
                buffer.object_id().get(),
                buffer.length(),
                texels.layout.offset,
                texels.layout.row_pitch,
            )
        }
        None => "its own image".to_owned(),
    };
    let text = format!(
        "metal2vulkan: shadow: {call} made a {:?} texture of {width}x{height} and {} levels, \
         usage {:?}, over {truth}, a shadow: {why}",
        texture.pixel_format, texture.levels, texture.usage,
    );
    inform_through(device.shared().instance.sink(), ID_NAME, text);
}
