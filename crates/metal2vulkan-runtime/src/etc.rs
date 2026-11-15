//! ETC2 and EAC textures on a device without the formats: their blocks held as raw bytes in a
//! carrier, which copies move, and their samples read a decoded twin.
//!
//! - **Metal's answers** (an Apple M3 Max under macOS 27.0.1, our own probe: 4096 random blocks
//!   of each of the ten formats, every texel read as `float`): Metal reads them as
//!   `bind::EtcDecode` decodes them, an 8-bit channel `c / 255`, an sRGB form's colour through its
//!   twelve-bit table, an 11-bit channel `c / 2047` (signed `c / 1023`).
//! - **The store.** A device without `textureCompressionETC2` has none of the formats. Such a
//!   texture holds its blocks in an image of a *carrier*: a compressed format whose blocks are
//!   also 4 x 4 texels and of the same bytes (8, or 16 for `EAC_RGBA8` and the `RG11` formats),
//!   the first of [`CARRIERS_8`] or [`CARRIERS_16`] the device copies. Vulkan copies a compressed
//!   image's blocks as bytes, so the carrier keeps every block a copy writes, whatever its bits
//!   would mean to the carrier's decoder; its image is the texture's shape, so every region a copy
//!   names is the texture's.
//! - **What it serves.** Copies between buffers and the texture and between textures (fills
//!   included), and the CPU's copies, move its blocks through the carrier, so `getBytes` and a
//!   blit out of it hand back the blocks written. Its sampled views, of its linear and sRGB forms
//!   alike (the same codes), present its decoded twin (`texture/decoded_twin.rs`): Metal's decode
//!   of the blocks, decoded after each write as an ASTC texture's twin is, on the GPU
//!   (`bind::etc_decode_module`), or a host write's on the CPU. An 8-bit format's twin holds its
//!   codes, exact (an sRGB form's views decode them as `R8G8B8A8_SRGB`, the device's sRGB decode,
//!   `metal::texel_format::srgb_coded`); an 11-bit format's holds 16-bit normalized codes
//!   (`bind::EtcTexels::Compact`, announced `decoded-texel-precision`), or Metal's values exactly
//!   where the device does not take those codes' format (`EtcTexels::Float`). Announced
//!   `carried-decoded-twin`.
//!   Refused: a view of another decode (an `R11` or `RG11` format's other signedness;
//!   `Feature::TextureView`) and a sparse heap's texture (`RuntimeGap::DecodedHeapTexture`). A
//!   heap's texture whose image shadows the heap's bytes keeps a twin, decoded after each copy in
//!   that moves its blocks (`texture/heap.rs`). A texture without a twin (a heap's over the
//!   caller's host pages) holds its blocks for copies alone, and a command that binds it for a
//!   read is refused. A shader write and an attachment take no such image: Metal's functions write no
//!   compressed texture, and no pass renders into one.

use super::image_plan::{sample_count, ImageFacts, ImagePlan};
use super::texture_descriptor::{effective_usage, TextureLayout};
use metal2vulkan::bind::EtcDecode;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{image_type, vk_format, FormatEmulation, FormatMapping};
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageUsageFlags,
};

/// The compressed formats of 8-byte blocks over 4 x 4 texels other than ETC2's and EAC's, in the
/// order an ETC2 or EAC texture of such blocks takes the first the device copies.
pub(super) const CARRIERS_8: [Format; 2] = [Format::BC1_RGBA_UNORM_BLOCK, Format::BC4_UNORM_BLOCK];

/// The compressed formats of 16-byte blocks over 4 x 4 texels other than ETC2's and EAC's, in the
/// order an `EAC_RGBA8` or `RG11` texture takes the first the device copies: BC7, BC3, BC2 and
/// BC5, then ASTC's 4x4 (a device with neither BC nor ETC2).
pub(super) const CARRIERS_16: [Format; 5] = [
    Format::BC7_UNORM_BLOCK,
    Format::BC3_UNORM_BLOCK,
    Format::BC2_UNORM_BLOCK,
    Format::BC5_UNORM_BLOCK,
    Format::ASTC_4X4_UNORM_BLOCK,
];

/// The carriers of `decode`'s blocks.
fn carriers(decode: EtcDecode) -> &'static [Format] {
    match decode.block_bytes() {
        8 => &CARRIERS_8,
        _ => &CARRIERS_16,
    }
}

/// The decode of the ETC2 or EAC texture whose blocks `plan` holds in a carrier (an sRGB form's
/// its unorm form's: the same codes); `None` for any other image.
pub(super) fn decode_of(plan: &ImagePlan) -> Option<EtcDecode> {
    EtcDecode::of(plan.pixel_format).filter(|&decode| carriers(decode).contains(&plan.format))
}

/// The image of an ETC2 or EAC texture of `layout` with Metal `usage` on a device with `facts`
/// that holds its blocks in a carrier (the module docs): `None` for a texture of another format,
/// or of one the device holds (its format, or no decoded twin's view: its refusal stands), or
/// where no carrier is copied (the device's refusal of the format stands).
pub(super) fn plan_blocks(
    layout: &TextureLayout,
    usage: TextureUsage,
    facts: &ImageFacts<'_>,
) -> Option<ImagePlan> {
    let pf = layout.pixel_format;
    let decode = EtcDecode::of(pf)?;
    let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), facts.profile).ok()?;
    if sampled.emulation != FormatEmulation::Decoded {
        return None;
    }
    let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    let carrier = (carriers(decode).iter())
        .copied()
        .find(|&f| facts.optimal(f).contains(copies))?;
    // No create flags, never cube compatible: copies alone reach the carrier and no view of it is
    // made, so a cube texture's faces are its layers.
    let (image_type, _) = image_type(layout.texture_type)?;
    Some(ImagePlan {
        pixel_format: pf,
        flags: ImageCreateFlags::empty(),
        image_type,
        format: carrier,
        extent: layout.extent,
        levels: layout.levels,
        layers: layout.layers,
        samples: sample_count(layout.samples)?,
        usage: ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(carrier, ImageAspectFlags::COLOR),
        // Its decoded twin serves its samples and reads. (`BlockWritesOnly` plans as
        // `ShaderWrite` does, which Metal's functions never use on a compressed texture.)
        unavailable: effective_usage(usage).intersection(
            TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC | TextureUsage::RENDER_TARGET,
        ),
    })
}
