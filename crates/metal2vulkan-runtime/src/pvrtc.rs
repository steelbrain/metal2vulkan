//! PVRTC textures on a device without a format for them: their blocks held as raw bytes, which
//! copies move, and their samples read a decoded twin.
//!
//! - **Metal's answers** (an M4 under macOS 27.0.1, alike plain twice and under API validation
//!   where it passes): every PVRTC format makes a texture, `Private` and `Shared`, square with a
//!   power-of-two side down to 4 texels, of levels down to 1 x 1, and a shareable one opens from
//!   its handle. Each level holds its blocks, one block a level once the level is no larger than a
//!   block. With `MTLBlitOptionRowLinearPVRTC` and a `bytesPerRow`, a copy between a buffer and the
//!   texture moves the blocks row by row, as every other compressed format's (whole levels, regions
//!   of whole blocks, from any offset, rows padded apart), and a copy between two PVRTC textures
//!   moves their blocks. Without the option and with a nonzero `bytesPerRow`, plain Metal copies as
//!   with it (its validation stops it: our probe on an M4 under macOS 27.0.1, both directions, 4-
//!   and 2-bit RGBA, 16 and 32 texels square, padded rows and a 2 x 2-block region; the reims-vgpu
//!   consumer's run on an M3 Max under macOS 26.7, 16 x 16 4-bit RGBA at 32 bytes a row). With a
//!   `bytesPerRow` of 0, with the option or without it, plain Metal lays the blocks out in the
//!   region's own Morton order, the x bit lowest, a whole level's and any region's of whole blocks
//!   alike (its validation stops both: our probe on an M4, the consumer's
//!   report of a whole level on an M3 Max); `replaceRegion` and `getBytes` of a whole level take
//!   them in the Z order with the y bit lowest, at a `bytesPerRow` of 0 (which its validation asks
//!   for, with a whole level) and of the level's row, every level of a chain alike. Of a region
//!   other than a whole level, plain Metal ignores the region's origin and places the blocks by no
//!   rule the runtime has found (the probe's 21 regions).
//! - **The store.** Vulkan has no PVRTC RGB format, and its PVRTC RGBA formats need
//!   `VK_IMG_format_pvrtc`, which only MoltenVK exposes. A PVRTC texture of a format the device
//!   has no representation of holds its blocks in an image of a *carrier*: a compressed format
//!   whose blocks are also 8 bytes, over 4 x 4 texels, the first of [`CARRIERS`] the device copies
//!   (`TRANSFER_SRC` and `TRANSFER_DST` in optimal tiling). Vulkan copies a compressed image's
//!   blocks as bytes, so the carrier keeps every block a copy writes, whatever its bits would mean
//!   to the carrier's decoder. A 4-bit texture's image is the texture's extent; a 2-bit
//!   texture's 8 x 4 blocks each lie over 4 x 4 carrier texels, so its image is half as wide
//!   (rounded up), and every region a copy names is the texture's with its x and width halved
//!   (the width rounded up, where the region ends at the level's edge; [`image_region`]). Either
//!   way each level holds the blocks Metal's level holds, one block once the level is no larger
//!   than one: a PVRTC texture is square with a power-of-two side (`texture_descriptor`), so the
//!   half-width image's level `l`, `max(1, w / 2^(l+1))` wide, holds `ceil(max(1, w / 2^l) / 8)`
//!   blocks across, as the texture's does.
//! - **What it serves.** Copies between buffers and the texture, row by row and in block order,
//!   and copies between textures (`blit_texture.rs`), and the CPU's copies of whole levels in
//!   their order (`Texture::replace_region` and `get_bytes`, through rows, [`rows_of_cpu_order`]).
//!   Its sampled views, of the format's linear and sRGB forms alike, present its decoded twin
//!   (`texture/decoded_twin.rs`): Metal's decode of the blocks (`bind::PvrtcDecode`, fitted on an
//!   Apple M3 Max) as 8-bit codes, exact, decoded after each write into it, whole levels at a time
//!   (each texel's colors come from the blocks around it, wrapping): on the GPU
//!   (`bind::pvrtc_decode_module`), or a host write's on the CPU. Announced
//!   `carried-decoded-twin` (the twin's memory and the decodes). A shader write and an attachment
//!   take no such image: Metal's functions write no compressed texture, and no pass renders into
//!   one. A texture without a twin (memoryless, sparse, or a heap's over the caller's host pages or
//!   laid out apart from the heap's bytes) holds its blocks for copies alone, and a command that
//!   binds it for a read is refused (the twin's format's sampled usage, which its image lacks).
//! - **Through handles.** A function that reads texture handles from memory may reach a texture
//!   without a twin made resident, or bound in a stage that reads handles, and the runtime does not
//!   know which of the textures it reaches the function reads: the handle's element is the neutral
//!   texture, nothing is refused, and the command announces it once
//!   ([`Degradation::UnreadableHandleTexture`], `degrade_unreadable`).
//! - **The CPU's whole level** takes any `bytesPerRow` of 0 or of the level's row of blocks and
//!   above: plain Metal moves the level's bytes contiguously there and ignores the pitch, so a
//!   padded one moves its padding bytes as block data (measured on an M3 Max, `texture.rs`).
//! - **Not yet:** the CPU's copies of a region other than a whole level, which plain Metal does
//!   not place by the region, and of a whole level at a nonzero `bytesPerRow` below its row,
//!   where the driver asserts and the call moves nothing (`NotImplemented(PvrtcBlockOrder)`). A
//!   blit's at a `bytesPerRow` of 0 of a region off its blocks copies the blocks it is rounded
//!   to.

use super::blit_plan::{block_order_position, BufferImageRegion};
use super::copy_plan::{block_extent, ImageRegion};
use super::degradation::Degradation;
use super::device::Device;
use super::image_plan::{block_family, sample_count, ImageFacts, ImagePlan};
use super::residency::Residency;
use super::texture::Texture;
use super::texture_descriptor::{effective_usage, TextureLayout};
use metal2vulkan::bind::PvrtcDecode;
use metal2vulkan::metal::format::{image_type, vk_format, FormatEmulation, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageUsageFlags,
};
use std::collections::BTreeSet;

/// The compressed formats whose blocks are 8 bytes over 4 x 4 texels, in the order a PVRTC
/// texture's blocks take the first the device copies: BC1 (every desktop GPU and Apple's), then
/// ETC2 and EAC (mobile GPUs and Apple's), then BC4.
pub(super) const CARRIERS: [Format; 4] = [
    Format::BC1_RGBA_UNORM_BLOCK,
    Format::ETC2_R8G8B8_UNORM_BLOCK,
    Format::EAC_R11_UNORM_BLOCK,
    Format::BC4_UNORM_BLOCK,
];

/// The bytes of a PVRTC block, of every PVRTC format.
pub(super) const BLOCK_BYTES: u32 = 8;

/// Whether `pf` is a PVRTC format (RGB or RGBA, 2 or 4 bits a pixel, linear or sRGB).
pub(super) fn is_pvrtc(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        block_family(pf),
        P::PVRTCRGB2BPP | P::PVRTCRGB4BPP | P::PVRTCRGBA2BPP | P::PVRTCRGBA4BPP
    )
}

/// Whether `pf` is a PVRTC format of 8 x 4 blocks (2 bits a pixel).
fn two_bits(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(block_family(pf), P::PVRTCRGB2BPP | P::PVRTCRGBA2BPP)
}

/// The texture's texels across one texel of the image `plan` plans: 2 for a 2-bit PVRTC
/// texture's carrier, whose 4 x 4 texels hold each 8 x 4 block (the module docs), 1 for every
/// other image.
fn texels_across(plan: &ImagePlan) -> u32 {
    if holds_blocks(plan) && two_bits(plan.pixel_format) {
        2
    } else {
        1
    }
}

/// `region`, in the texels of a texture whose image `plan` plans, in the image's texels: a 2-bit
/// PVRTC texture's carrier halves x and the width, the width rounded up (a region of whole
/// blocks starts at a multiple of 8 and is 8 texels a block wide, or ends at the level's edge,
/// which rounding up keeps at the image level's); an ASTC texture's carrier holds each block in
/// 4 x 4 of its texels (`astc::image_region`); every other image's region is the texture's.
pub(super) fn image_region(plan: &ImagePlan, region: ImageRegion) -> ImageRegion {
    if let Some(carried) = super::astc::image_region(plan, region) {
        return carried;
    }
    let n = texels_across(plan);
    let [x, y, z] = region.origin;
    let [width, height, depth] = region.size;
    ImageRegion {
        origin: [x / n, y, z],
        size: [width.div_ceil(n), height, depth],
        ..region
    }
}

/// [`image_region`] of `region`'s image region, and its buffer row length (texels a row) in the
/// image's texels.
pub(super) fn image_buffer_region(
    plan: &ImagePlan,
    region: BufferImageRegion,
) -> BufferImageRegion {
    // An ASTC texture's carrier: rows and images of its blocks, 4 x 4 carrier texels each.
    if let Some(decode) = super::astc::decode_of(plan) {
        let [bw, bh] = decode.block_extent();
        return BufferImageRegion {
            row_length: super::astc::carrier_texels(region.row_length, bw),
            image_height: super::astc::carrier_texels(region.image_height, bh),
            image: image_region(plan, region.image),
            ..region
        };
    }
    BufferImageRegion {
        row_length: region.row_length / texels_across(plan),
        image: image_region(plan, region.image),
        ..region
    }
}

/// The position of block `(x, y)` of a PVRTC level `blocks` across and down in the order plain
/// Metal's CPU copies (`replaceRegion`, `getBytes`) take a whole level's blocks: Z order with the
/// y bit lowest, the larger axis's extra bits on top (PowerVR's own order), the blit's block order
/// ([`block_order_position`]) with the axes swapped. A level's blocks across and down are powers of
/// two, so its blocks take every position below their count.
pub(super) fn cpu_order_position(x: u64, y: u64, [across, down]: [u64; 2]) -> u64 {
    block_order_position(y, x, [down, across])
}

/// A whole level's blocks, `blocks` across and down, laid out in plain Metal's CPU order in
/// `level` (at least their bytes), as rows: the tightly packed layout the runtime copies through.
pub(super) fn rows_of_cpu_order(level: &[u8], blocks: [u64; 2]) -> Vec<u8> {
    let n = BLOCK_BYTES as usize;
    let mut rows = vec![0; (blocks[0] * blocks[1]) as usize * n];
    for (i, block) in rows.chunks_exact_mut(n).enumerate() {
        let (x, y) = (i as u64 % blocks[0], i as u64 / blocks[0]);
        let at = cpu_order_position(x, y, blocks) as usize * n;
        block.copy_from_slice(&level[at..at + n]);
    }
    rows
}

/// [`rows_of_cpu_order`] reversed: the rows `rows` of a whole level's blocks, `blocks` across and
/// down, into `level` (at least their bytes) in plain Metal's CPU order.
pub(super) fn cpu_order_of_rows(rows: &[u8], blocks: [u64; 2], level: &mut [u8]) {
    let n = BLOCK_BYTES as usize;
    for (i, block) in rows.chunks_exact(n).enumerate() {
        let (x, y) = (i as u64 % blocks[0], i as u64 / blocks[0]);
        let at = cpu_order_position(x, y, blocks) as usize * n;
        level[at..at + n].copy_from_slice(block);
    }
}

/// The block extent, in texels of an image of `format` holding a texture of `pf`, of the blocks
/// Vulkan copies: a carrier's 4 x 4 (a 2-bit PVRTC texture's 8 x 4 blocks and an ASTC texture's
/// blocks lie over 4 x 4 of its carrier's texels), else `pf`'s own.
pub(super) fn image_block_extent(format: Format, pf: PixelFormat) -> Option<[u32; 2]> {
    let astc = metal2vulkan::bind::AstcDecode::of(pf).is_some();
    if (is_pvrtc(pf) && CARRIERS.contains(&format))
        || (astc && super::astc::CARRIERS.contains(&format))
    {
        Some([4, 4])
    } else {
        block_extent(pf)
    }
}

/// The image of a texture of `layout` with Metal `usage` on a device with `facts` that holds a
/// PVRTC texture's blocks in a carrier (the module docs): `None` for a texture of another format,
/// of a PVRTC format the device represents, or of one no carrier holds (its refusal stands).
pub(super) fn plan_blocks(
    layout: &TextureLayout,
    usage: TextureUsage,
    facts: &ImageFacts<'_>,
) -> Option<ImagePlan> {
    let pf = layout.pixel_format;
    let direct = vk_format(pf, &BTreeSet::new(), facts.profile)
        .is_ok_and(|m| m.emulation == FormatEmulation::Direct);
    if !is_pvrtc(pf) || direct {
        return None;
    }
    let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    let carrier = CARRIERS
        .into_iter()
        .find(|&f| facts.optimal(f).contains(copies))?;
    // No create flags, never cube compatible: copies alone reach the carrier and no view of it
    // is made, so a cube texture's faces are its layers; and a 2-bit texture's half-width
    // carrier is no square.
    let (image_type, _) = image_type(layout.texture_type)?;
    let usage = effective_usage(usage);
    Some(ImagePlan {
        pixel_format: pf,
        flags: ImageCreateFlags::empty(),
        image_type,
        format: carrier,
        // A 2-bit texture's 8 x 4 blocks each lie over 4 x 4 carrier texels.
        extent: match two_bits(pf) {
            true => {
                let [width, height, depth] = layout.extent;
                [width.div_ceil(2), height, depth]
            }
            false => layout.extent,
        },
        levels: layout.levels,
        layers: layout.layers,
        samples: sample_count(layout.samples)?,
        usage: ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(carrier, ImageAspectFlags::COLOR),
        // Its decoded twin serves its samples and reads (`texture/decoded_twin.rs`).
        unavailable: usage.intersection(
            TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC | TextureUsage::RENDER_TARGET,
        ),
    })
}

/// Whether `plan` holds a PVRTC texture's blocks in a carrier (the module docs).
pub(super) fn holds_blocks(plan: &ImagePlan) -> bool {
    is_pvrtc(plan.pixel_format) && CARRIERS.contains(&plan.format)
}

/// The decode of the PVRTC texture whose blocks `plan` holds in a carrier, its linear form's (an
/// sRGB form's codes are the same); `None` for any other image.
pub(super) fn decode_of(plan: &ImagePlan) -> Option<PvrtcDecode> {
    PvrtcDecode::of(plan.pixel_format)
        .filter(|_| holds_blocks(plan))
        .map(PvrtcDecode::linear)
}

/// The carrier's format of `texture`, whose bytes a carrier holds for copies alone: a PVRTC,
/// ETC2 or EAC texture's blocks, or a 4:2:2 texture's texels (`chroma_pairs.rs`), without a
/// decoded twin (the module docs, `etc.rs`); `None` for any other texture.
pub(super) fn held_for_copies(texture: &Texture) -> Option<Format> {
    let plan = texture.image_plan().ok()?;
    let carried = holds_blocks(plan)
        || super::etc::decode_of(plan).is_some()
        || super::chroma_pairs::holds_pairs(plan);
    (carried && !texture.has_decoded_twin()).then_some(plan.format)
}

/// The first texture a residency made resident whose bytes a carrier holds for copies alone
/// ([`held_for_copies`]), kept by an encoder between its commands and found again only where the
/// residency's generation moved since (`Residency::generation`): whether a carrier holds a
/// texture's bytes, and whether it has a decoded twin, are fixed at its making.
#[derive(Default)]
pub(super) struct ResidentUnreadable {
    found_at: Option<u64>,
    texture: Option<Texture>,
}

impl ResidentUnreadable {
    /// That texture among those `residency`, of `device`, made resident, found again where it may
    /// have changed.
    pub(super) fn of(&mut self, device: &Device, residency: &Residency) -> Option<&Texture> {
        let generation = residency.generation();
        if self.found_at != Some(generation) {
            self.texture = (residency.textures())
                .find(|texture| held_for_copies(texture).is_some())
                .cloned();
            self.found_at = Some(generation);
            device.shared().counts.resident_textures_looked_at();
        }
        self.texture.as_ref()
    }

    /// Hold none: an ended encoder's, kept for the next (`compute/live.rs`).
    pub(super) fn let_go(&mut self) {
        self.found_at = None;
        self.texture = None;
    }
}

/// Announce once ([`Degradation::UnreadableHandleTexture`]) that `command` (named), encoded, whose
/// functions read texture handles from memory, reaches a texture among `textures` (those made
/// resident, and those bound in a stage that reads handles) whose bytes a carrier holds for
/// copies alone ([`held_for_copies`]).
pub(super) fn degrade_unreadable<'a>(
    device: &Device,
    textures: impl IntoIterator<Item = &'a Texture>,
    command: &str,
) {
    let Some((texture, carrier)) = textures
        .into_iter()
        .find_map(|texture| Some((texture, held_for_copies(texture)?)))
    else {
        return;
    };
    device.degrade(Degradation::UnreadableHandleTexture, || {
        (
            format!(
                "{:?} texture at resource ID {}",
                texture.pixel_format(),
                texture.gpu_resource_id().0
            ),
            format!(
                "{command}: a function reading texture handles from memory reaches the texture, \
                 whose bytes a {carrier:?} image holds for copies alone"
            ),
        )
    });
}
