//! The shadow log: a default-off diagnostic, `METAL2VULKAN_LOG_SHADOWS` (`metal2vulkan::env_vars`),
//! that names each shadow a device makes of a texture over a buffer's bytes, and why its route
//! took no image over them.
//!
//! - **What it prints.** One line per such shadow as it is made, through the device's message
//!   sink at `Info` severity (id name `metal2vulkan-shadows`), else on stderr: the Metal call that
//!   made it (`new_texture_with_iosurface_plane` for an IOSurface plane's), the texture's format,
//!   size, levels and usage, the buffer's storage mode, length and object ID
//!   (`Buffer::object_id`), the texels' offset and row pitch, and the reason the texture's notice
//!   gives (`shadowed-texture`, announced once a device). Two textures over the same bytes name
//!   the same object ID and offset. A device copy is no shadow it names.
//! - **Off, it costs nothing.** The switch is read once; no line is formatted.

use super::buffer_texture::Texels;
use super::driver_facts::inform_through;
use super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::Device;
use std::sync::OnceLock;

/// The id name of the log's messages.
const ID_NAME: &str = "metal2vulkan-shadows";

/// Whether the log is on: read once.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_shadows)
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
    if !enabled() {
        return;
    }
    let [width, height, _] = layout.extent;
    let buffer = &texels.buffer;
    let storage = match buffer.resource_options().storage_mode() {
        Ok(mode) => format!("{mode:?}"),
        Err(_) => "unknown".to_owned(),
    };
    let text = format!(
        "metal2vulkan: shadow: {call} made a {:?} texture of {width}x{height} and {} levels, \
         usage {:?}, over a {storage} buffer, object {}, of {} bytes at offset {} with rows of {} \
         bytes, a shadow: {why}",
        descriptor.pixel_format,
        layout.levels,
        descriptor.usage,
        buffer.object_id().get(),
        buffer.length(),
        texels.layout.offset,
        texels.layout.row_pitch,
    );
    inform_through(device.shared().instance.sink(), ID_NAME, text);
}
