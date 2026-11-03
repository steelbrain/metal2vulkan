//! 4:2:2 textures (`GBGR422`, `BGRG422`): two texels share their chroma, 4 bytes a pair (`G0 B G1
//! R`, `B G0 R G1`), held as 2-byte texels.
//!
//! - **Metal's answers** (an M4 under macOS 27.0.1, alike plain twice and under
//!   API validation where it passes): `newTextureWithDescriptor:` makes either format's 2D
//!   texture (of an even width, `texture_descriptor`), `Private` and `Shared`, with `ShaderRead`,
//!   `ShaderWrite` or `RenderTarget` usage, at 2 bytes a texel (a 16 x 16 texture allocates 512
//!   bytes). `replaceRegion` and `getBytes`, and blits between a buffer and the texture, move its
//!   bytes unchanged.
//! - **The store.** Vulkan's 4:2:2 formats need a sampler Y′CbCr conversion for any view of them
//!   (VUID-VkImageViewCreateInfo-format-06415), which neither the binder nor the lowering models
//!   (`metal::format`). A 4:2:2 texture's bytes lie in an image of the *carrier*, [`CARRIER`], of
//!   the texture's extent, each texel's 2 bytes one carrier texel: a copy's regions and buffer
//!   rows are the texture's own, and Vulkan's copies move Metal's bytes.
//! - **What it serves.** Copies: between buffers and the texture, between textures, and the CPU's
//!   (`copy_plan.rs`). A storage slot whose function only writes the texture binds the slot's
//!   neutral (the binding plan's `IncompatibleBinding` note): plain Metal completes the writes
//!   and stores nothing. Refused: a render pipeline state of the format,
//!   whose descriptor Metal's validation aborts plain too, and a render pass into the texture.
//! - **Reads and samples** present a decoded twin (`texture/decoded_twin.rs`, announced
//!   `carried-decoded-twin`): an `R8G8B8A8_UNORM` image of the texels as Metal reads them
//!   (`bind::ChromaDecode`: each texel's own green, its red and blue interpolated with the pair
//!   beside it, alpha 1; our probe on an Apple M3 Max under macOS 27.0.1 read, sampled (nearest,
//!   linear and bicubic) and gathered both formats bit for bit as such an `RGBA8Unorm` texture),
//!   exact, decoded after
//!   each write as an ETC2 texture's twin is, on the GPU (`bind::chroma_decode_module`) or a host
//!   write's on the CPU. Where the device converts the format itself with Metal's reconstruction
//!   (`ycbcr_device.rs`: a sampler Y′CbCr conversion of `G8B8G8R8_422_UNORM` or
//!   `B8G8R8G8_422_UNORM` at the midpoint through a linear chroma filter, MoltenVK), the twin holds
//!   the device's conversion of the pairs instead, every write's decode a GPU pass that samples
//!   the pairs through it (`bind::chroma_sample_module`), announced `decoded-texel-precision`
//!   where a command first reads the twin. A texel reads the pair beside it, so a write's decode
//!   is of whole rows (a host write of part of a row decodes its rows as the texture then holds
//!   them). Refused: a
//!   sparse heap's texture (`RuntimeGap::DecodedHeapTexture`). Without a twin (a heap's texture
//!   over the caller's host pages or laid out apart from the heap's bytes) a command that binds
//!   it for a read is refused.
//! - **Through handles.** A function that reads texture handles from memory reaches such a
//!   texture without a twin as a PVRTC carrier's (`pvrtc.rs`): the neutral texture, announced
//!   once ([`Degradation::UnreadableHandleTexture`](super::degradation::Degradation)).

use super::image_plan::{sample_count, ImageFacts, ImagePlan};
use super::texture_descriptor::{effective_usage, TextureLayout};
use metal2vulkan::bind::ChromaDecode;
use metal2vulkan::metal::format::{image_type, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageAspectFlags, ImageUsageFlags};

/// The format of the image holding a 4:2:2 texture's texels, 2 bytes each (the module docs).
pub(super) const CARRIER: Format = Format::R8G8_UNORM;

/// The bytes of a 4:2:2 texel.
pub(super) const TEXEL_BYTES: u32 = 2;

/// Whether `pf` is a 4:2:2 format.
pub(super) fn is_four_two_two(pf: PixelFormat) -> bool {
    matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422)
}

/// The image of a 4:2:2 texture of `layout` with Metal `usage` on a device with `facts`: the
/// carrier's (the module docs), for copies, its reads and samples its decoded twin's; `None` for a
/// texture of another format, or where the device does not copy the carrier (the device's refusal
/// of the format stands).
pub(super) fn plan_pairs(
    layout: &TextureLayout,
    usage: TextureUsage,
    facts: &ImageFacts<'_>,
) -> Option<ImagePlan> {
    let pf = layout.pixel_format;
    let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    if !is_four_two_two(pf) || !facts.optimal(CARRIER).contains(copies) {
        return None;
    }
    let (image_type, flags) = image_type(layout.texture_type)?;
    Some(ImagePlan {
        pixel_format: pf,
        flags,
        image_type,
        format: CARRIER,
        extent: layout.extent,
        levels: layout.levels,
        layers: layout.layers,
        samples: sample_count(layout.samples)?,
        usage: ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(CARRIER, ImageAspectFlags::COLOR),
        // Its decoded twin serves its reads and samples.
        unavailable: effective_usage(usage).intersection(
            TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC | TextureUsage::RENDER_TARGET,
        ),
    })
}

/// Whether `plan` holds a 4:2:2 texture's texels in the carrier (the module docs).
pub(super) fn holds_pairs(plan: &ImagePlan) -> bool {
    is_four_two_two(plan.pixel_format) && plan.format == CARRIER
}

/// The decode of the 4:2:2 texture whose texels `plan` holds in the carrier; `None` for any other
/// image.
pub(super) fn decode_of(plan: &ImagePlan) -> Option<ChromaDecode> {
    ChromaDecode::of(plan.pixel_format).filter(|_| holds_pairs(plan))
}
