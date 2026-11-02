//! ASTC textures (LDR, sRGB and HDR) on a device without their ASTC block: their blocks held as
//! raw bytes in a carrier, which copies move, and their samples read a decoded twin.
//!
//! - **Metal's answers** (an Apple M3 Max under macOS 27.0.1, our own probes; the HDR formats'
//!   `conformance/probes/astc_hdr_textures.m`): every ASTC format, LDR, sRGB and HDR, makes a
//!   texture, `Shared`, `Private` and memoryless, of every type but 1D and multisampled (which
//!   end the process), of `ShaderWrite` and `RenderTarget` usage too, and views as its other
//!   forms of the footprint (another footprint's view ends the process). `getBytes` and a blit out
//!   of the texture hand back the blocks a write put there; a new texture's blocks are zeros,
//!   which read magenta (an error block; an HDR texture's NaN). Its reads and samples are
//!   `bind::AstcDecode`'s. An HDR format's render pipeline ends the process; a pass storing into
//!   such a texture completes and leaves its blocks as they were, and a kernel's write completes
//!   and leaves bytes no format defines: Metal's validation stops both.
//! - **The store.** Vulkan copies a compressed image's blocks as bytes. Such a texture holds its
//!   blocks in an image of a *carrier*: a compressed format whose blocks are also 16 bytes, over
//!   4 x 4 texels, the first of [`CARRIERS`] the device copies (every Vulkan device has BC or ETC2,
//!   which hold one). Each of the texture's blocks lies over 4 x 4 carrier texels, so a region of
//!   the texture's blocks is a region of the carrier's ([`image_region`]). A level of the carrier
//!   holds at least the blocks the texture's level holds ([`carrier_side`]): the carrier's side is
//!   the least whose every level, halved as Vulkan halves it, keeps them (at most about the
//!   texture's side in texels, 1 byte a texel), and a copy addresses the first of them.
//! - **What it serves.** Copies between buffers and the texture and between textures (fills
//!   included), and the CPU's copies, move its blocks through the carrier, so `getBytes` and a
//!   blit out of it hand back the blocks written. Its sampled views present its decoded twin
//!   (`texture/decoded_twin.rs`), Metal's decode of the blocks as 8-bit codes (an LDR form's the
//!   code nearest Metal's 12-bit value, announced `decoded-texel-precision`; an sRGB form's its own
//!   codes, which its slots read through the twin's `R8G8B8A8_SRGB` view, the device's decode,
//!   announced `srgb-texel-decode`) or an HDR form's halves, Metal's exactly
//!   (`R16G16B16A16_SFLOAT`), decoded after each write as a BC
//!   texture's twin is: on the GPU (`bind::astc_decode_module`), or a host write's on the CPU.
//!   A view in the texture's other colour space (its `_sRGB` or LDR form) presents a second twin
//!   of that form's decode, made at the first such view. Refused: a heap's or a sparse heap's
//!   texture (`RuntimeGap::DecodedHeapTexture`) and a view of another decode (an LDR or sRGB
//!   texture's HDR form, an HDR texture's LDR or sRGB form, or another footprint;
//!   `Feature::TextureView`). A shader write and an attachment take no such image (no more than
//!   the device's own ASTC block, which Vulkan neither stores into nor renders into): a pass
//!   rendering into one and a kernel's binding for writes are refused, typed, where Metal's
//!   validation stops each.

use super::copy_plan::ImageRegion;
use super::image_plan::{sample_count, ImageFacts, ImagePlan};
use super::texture_descriptor::{effective_usage, TextureLayout};
use metal2vulkan::bind::AstcDecode;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{image_type, vk_format, FormatEmulation, FormatMapping};
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageUsageFlags,
};

/// The compressed formats whose blocks are 16 bytes over 4 x 4 texels, in the order an ASTC
/// texture's blocks take the first the device copies: BC7, BC3, BC2 and BC5 (desktop GPUs), then
/// ETC2's and EAC's (mobile GPUs).
pub(super) const CARRIERS: [Format; 6] = [
    Format::BC7_UNORM_BLOCK,
    Format::BC3_UNORM_BLOCK,
    Format::BC2_UNORM_BLOCK,
    Format::BC5_UNORM_BLOCK,
    Format::ETC2_R8G8B8A8_UNORM_BLOCK,
    Format::EAC_R11G11_UNORM_BLOCK,
];

/// The decode of the ASTC texture whose blocks `plan` holds in a carrier; `None` for any other
/// image.
pub(super) fn decode_of(plan: &ImagePlan) -> Option<AstcDecode> {
    AstcDecode::of(plan.pixel_format).filter(|_| CARRIERS.contains(&plan.format))
}

/// The least carrier side, in carrier texels, whose `levels` levels each hold the blocks of
/// `block` texels a side that a texture of `side` texels holds at that level: the carrier's level
/// `l`, `max(1, c >> l)` wide, holds `ceil(max(1, c >> l) / 4)` blocks, at least `n` where
/// `c >= (4 n - 3) << l`. The last level's term is at least `2^(levels - 1)`, so the carrier takes
/// every level.
pub(super) fn carrier_side(side: u32, block: u32, levels: u32) -> u32 {
    (0..levels)
        .map(|l| {
            let blocks = (side >> l).max(1).div_ceil(block);
            (4 * blocks - 3) << l
        })
        .max()
        .unwrap_or(1)
}

/// The image of an ASTC texture of `layout` with Metal `usage` on a device with `facts` that
/// holds its blocks in a carrier (the module docs): `None` for a texture of another format, or of
/// an ASTC format the device holds (its block, or no decoded twin: its refusal stands), or where
/// no carrier is copied (the device's refusal of the format stands).
pub(super) fn plan_blocks(
    layout: &TextureLayout,
    usage: TextureUsage,
    facts: &ImageFacts<'_>,
) -> Option<ImagePlan> {
    let pf = layout.pixel_format;
    let decode = AstcDecode::of(pf)?;
    let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), facts.profile).ok()?;
    if sampled.emulation != FormatEmulation::Decoded {
        return None;
    }
    let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    let carrier = CARRIERS
        .into_iter()
        .find(|&f| facts.optimal(f).contains(copies))?;
    // No create flags, never cube compatible: copies alone reach the carrier and no view of it is
    // made, so a cube texture's faces are its layers.
    let (image_type, _) = image_type(layout.texture_type)?;
    let [width, height, depth] = layout.extent;
    let [bw, bh] = decode.block_extent();
    Some(ImagePlan {
        pixel_format: pf,
        flags: ImageCreateFlags::empty(),
        image_type,
        format: carrier,
        extent: [
            carrier_side(width, bw, layout.levels),
            carrier_side(height, bh, layout.levels),
            depth,
        ],
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

/// `region`, in the texels of the ASTC texture whose carrier `plan` plans (a region of whole
/// blocks, or ending at the level's edge), in the carrier's texels: each block 4 x 4 of them, the
/// region's last block's ending at the carrier level's edge where that comes first. `None` for an
/// image of another kind.
pub(super) fn image_region(plan: &ImagePlan, region: ImageRegion) -> Option<ImageRegion> {
    let decode = decode_of(plan)?;
    let [bw, bh] = decode.block_extent();
    let [x, y, z] = region.origin;
    let [width, height, depth] = region.size;
    let level = |axis: usize| (plan.extent[axis] >> region.level.min(31)).max(1);
    let span = |at: u32, size: u32, block: u32, axis: usize| {
        let first = at / block;
        let end = (at + size).div_ceil(block);
        (4 * first, (4 * end).min(level(axis)) - 4 * first)
    };
    let (x, width) = span(x, width, bw, 0);
    let (y, height) = span(y, height, bh, 1);
    Some(ImageRegion {
        origin: [x, y, z],
        size: [width, height, depth],
        ..region
    })
}

/// [`image_region`]'s inverse: `region` of the carrier `plan` plans, in the texture's texels, its
/// last block's ending at the texture level's edge (`extent`, the texture's level 0). `None` for
/// an image of another kind.
pub(super) fn texture_region(
    plan: &ImagePlan,
    extent: [u32; 3],
    region: ImageRegion,
) -> Option<ImageRegion> {
    let decode = decode_of(plan)?;
    let [bw, bh] = decode.block_extent();
    let [x, y, z] = region.origin;
    let [width, height, depth] = region.size;
    let level = |axis: usize| (extent[axis] >> region.level.min(31)).max(1);
    let span = |at: u32, size: u32, block: u32, axis: usize| {
        let first = at / 4;
        let end = (at + size).div_ceil(4);
        (
            block * first,
            (block * end).min(level(axis)) - block * first,
        )
    };
    let (x, width) = span(x, width, bw, 0);
    let (y, height) = span(y, height, bh, 1);
    Some(ImageRegion {
        origin: [x, y, z],
        size: [width, height, depth],
        ..region
    })
}

/// A buffer row length or image height of `texels` texture texels in carrier texels: its blocks
/// of `block` texels, 4 carrier texels each (0, tightly packed, stays 0).
pub(super) fn carrier_texels(texels: u32, block: u32) -> u32 {
    texels / block * 4
}
