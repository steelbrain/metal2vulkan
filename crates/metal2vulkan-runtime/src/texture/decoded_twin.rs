//! Decoded twins of carried textures: an ASTC (LDR, sRGB or HDR), PVRTC, ETC2 or EAC texture on a
//! device without its format, and a 4:2:2 texture on every device.
//!
//! **Why.** The device holds no image of such a texture's format that samples as Metal's does, so
//! its blocks lie in a carrier image of a format whose blocks are as many bytes, which every copy,
//! blit and CPU access reads and writes as Metal's texture's bytes. Beside it, where its image is
//! its own for good (in memory of its own, or bound in a heap's own memory;
//! [`Device::with_decoded_twin`]), the texture keeps a decoded twin: an image of the texture's
//! shape, levels and layers, in memory of its own, holding each texel of Metal's decode of the
//! blocks (`TwinDecode::texel_format`). Every sampled view a binding takes of the texture, or of a
//! view of it of the same decode, presents the twin ([`Texture::decoded_view`]), so reads, samples
//! and gathers read Metal's texels, which the device filters as it filters the twin's format. The
//! twin's format must take sampling, its linear filter and copies on the device, else the texture
//! is refused (no conformant device lacks them).
//!
//! **Kept in step.** After every write into the carrier (`replace_region`, staged or submitted
//! now; a blit from a buffer, `fillTexture` included; a copy from a texture), the written region
//! is decoded into the twin by the decode's kernel (`TwinDecode::module`; [`DecodePass`]): its
//! blocks copied into a scratch buffer, decoded, and the texels copied into the twin, in bands of
//! whole block rows of at most [`BAND_BYTES`] of blocks and texels, which bound the scratch: the
//! pass's own, or for a pass the device's next submission takes, the device's ([`PassScratch`]). A
//! new texture's twin starts as its zero blocks decode (the module's `zero_clear`); a heap's
//! texture, which holds what its bytes held, has its whole twin decoded before any command reads
//! it (a staged pass, `TexelCopy`).
//!
//! **Host writes decode on the CPU** (off the caller's thread). A `replace_region` whose blocks go
//! on the host or the staging queue queues the region's decode on the CPU
//! (`TwinDecode::decode_into`, the kernel's rules) for the device's decode thread, which copies
//! its texels into the twin the same way ([`Image::decode_written`], `twin_decode.rs`, announced
//! `cpu-block-decode`). A batch that may read the texture waits at its submission for that; one
//! submitted before the call and not yet complete reads the twin as the thread's copies stand when
//! it runs, as Metal's unsynchronized CPU write leaves the ordering to the program. For that the
//! twin is made as its carrier's CPU copies go: with host-transfer usage, or shared with the
//! staging queue, and moved to `GENERAL` there alone as it is made, its texels started as the zero
//! blocks' decode the same way, a constant fill queued with the texture's making
//! ([`Image::start_decoded_twin`]). The runtime stages no GPU decode of such a twin of its own:
//! one in flight could read the old blocks before a later write and copy their texels into the
//! twin after that write's CPU decode. A blit's decode (`decode_twin`) is the caller's command's,
//! and covers only what it wrote, as Metal's write of those texels would.
//!
//! **ASTC** (`astc.rs`): its twin holds Metal's decode as 8-bit codes (`TwinDecode::Astc`; an LDR
//! form's the code nearest Metal's 12-bit value, an sRGB form's its own codes, which its slots read
//! through the twin's sRGB view, the device's decode), or an HDR form's halves, Metal's exactly
//! (`R16G16B16A16_SFLOAT`), kept in step as above (`bind::astc_decode_module` on the
//! GPU, `AstcDecode::decode_into` on the CPU): a pass's bands are of the texture's texels, each
//! band's blocks copied out of the carrier's texels over them. A new one's zero blocks read
//! magenta (an error block; an HDR one's NaN, the twin started cleared to it).
//!
//! **An ASTC texture's other colour space.** Metal views an ASTC texture as its `_sRGB` or LDR
//! form over the same blocks, whatever its usage, and the view reads what a texture of its own
//! format over those blocks reads (our probe on an M3 Max, macOS 27.0.1: reads and linear samples
//! bit for bit, footprints 4x4, 6x5 and 12x12). The two forms widen the endpoints apart
//! (`bind::AstcDecode`), so neither's texels follow from the other's. An HDR texture's LDR or sRGB
//! view, and an LDR or sRGB texture's HDR view, decode in another profile: a carrier refuses them
//! (`Feature::TextureView`). Within the LDR profile the first view of the
//! texture in its other colour space makes the carrier a second twin ([`Image::make_second_twin`]),
//! of that form's decode in 8-bit codes, in memory of its own, which the CPU
//! never writes. Its decodes are the GPU's, each recorded by a batch's submission
//! ([`SecondDecode`]): its whole decode as it is made, and a host write's, queued for the
//! device's next batch with work, which records them before its work, their uses joined to its
//! first part's, so the tracker orders them after earlier writes of the texture
//! ([`Device::take_second_decodes`]); a blit's write's, made as the blit is encoded where the
//! twin is there then, recorded by its batch after the blit's encoder ([`TwinWrite`]). A write
//! encoded before the twin was made has no decode: its batch leaves the texture's queued decodes,
//! the whole decode among them, to a later batch, which reads the carrier as the write left it.
//! Without a second twin a submission records nothing for such writes (counted:
//! `WorkCounts::second_twins`, `second_twin_checks`, `second_twin_decodes`).
//!
//! **PVRTC** (`pvrtc.rs`): its twin holds the blocks' codes (`TwinDecode::Pvrtc`,
//! `R8G8B8A8_UNORM`; an sRGB form's slots read them through its `R8G8B8A8_SRGB` view, the
//! device's decode),
//! which serve the views of the format's linear and sRGB forms alike. Each texel's colors come
//! from the four blocks around it, wrapping at the level's edges, so a write's decode is of whole
//! levels (`twin_region`): on the GPU (`bind::pvrtc_decode_module`), each band of a level's rows
//! with all of its blocks, or a host write's on the CPU (`PvrtcDecode::decode_rows`), a host write
//! being of whole levels. A new one's zero blocks read transparent black (an RGB format's opaque).
//!
//! **ETC2 and EAC** (`etc.rs`): the twin holds Metal's decode (`TwinDecode::Etc`): an 8-bit
//! format's codes (`R8G8B8A8_UNORM`; an sRGB form's slots read them raw and decode them through
//! Metal's table), which serve the views of its linear and sRGB forms alike, and an 11-bit
//! format's 16-bit normalized codes, or its values where the device does not take those
//! (`bind::EtcTexels`). Each is kept in step as above (`bind::etc_decode_module` on the GPU,
//! `EtcDecode::decode_into` on the CPU). A new one's zero blocks read grey of codes 2
//! (`ETC2_RGB8`, and `EAC_RGBA8`'s colour, its alpha 0), opaque black (`RGB8A1`), or an 11-bit
//! value of 1 (signed -3), not zeros: the twin starts cleared to them.
//!
//! **4:2:2** (`chroma_pairs.rs`): a `GBGR422` or `BGRG422` texture's texels lie in a carrier of
//! 2 bytes a texel, the texture's shape, and its twin holds Metal's texels (`TwinDecode::Chroma`,
//! `R8G8B8A8_UNORM` codes, exact), kept in step as an ETC2 texture's twin is
//! (`bind::chroma_decode_module` on the GPU, `ChromaDecode::decode_into` on the CPU). A texel's red
//! and blue read the pair beside it, so a write's decode is of whole rows (`twin_region`; a host
//! write of part of a row decodes its rows as the carrier then holds them, read back first). A new
//! one's zero pairs read opaque black: the twin starts cleared to it.
//!
//! **4:2:2 through the device's own conversion.** Where the device converts the pairs' format
//! itself with Metal's reconstruction (`ycbcr_device.rs`, [`YCbCrSampler::pairs`]), the twin holds
//! the device's conversion of the pairs instead (`decoded-texel-precision`, announced as
//! **Reads** says; counted `WorkCounts::chroma_device_twins`): each decode pass's band, its pairs
//! copied out of the carrier into the scratch, is copied into the texture's pairs' image
//! ([`PairsSampling`]) and sampled through the conversion into the band's texels
//! (`bind::chroma_sample_module`), which are copied into the twin. Every write's decode is such a
//! pass: the carrier's CPU copies go through
//! submissions there (no host or staging copies), each followed by its pass, so no texel of the
//! twin is the CPU's (Metal's) decode.
//!
//! **Cost**, announced once per device (`carried-decoded-twin`, performance): the carrier and the
//! twin's memory, and a decode pass after each write. A `VkImage` handed out (`Texture::vk_image`)
//! is the carrier: what the caller writes into it outside the runtime's calls leaves the twin as
//! it was.
//!
//! **Reads.** Only a sampled view reads the twin ([`Texture::decoded_view`]); copies, blits, fills
//! and CPU accesses move the carrier's blocks, and a decode pass writes it. Where its texels are
//! not Metal's decode exactly (an ASTC LDR form's 8-bit codes, an EAC 11-bit format's 16-bit ones,
//! a 4:2:2 texture's device conversion: [`TwinRead`]), `decoded-texel-precision` is announced,
//! once per device, where a command may first read it, and not where the texture is made: at a
//! command's sampled slot binding of the texture or of a view of it (`Texture::bound_view`, a
//! draw's, a dispatch's, a tile dispatch's), at a command's binding of a heap entry that names it
//! (`binding.rs`), at the first command that binds a heap set holding the twin's view, where a
//! function may read it through its resource ID with no encoder naming it (`heap_sets.rs`), and
//! as the view is placed in a heap set a command buffer that bound the set and has not completed
//! may read through.
//!
//! A BC texture keeps no twin: the device samples its own BC image through its own decode,
//! announced `compressed-texel-decode` where its driver is not measured to decode the format as
//! Metal does ([`Device::degrade_block_decode`]).

use super::super::astc;
use super::super::blit::{BlitCommandEncoder, TRANSFER};
use super::super::chroma_pairs;
use super::super::copy_plan::ImageRegion;
use super::super::degradation::{Degradation, Told};
use super::super::device::{lock, DebugMessage, MessageKind, MessageSeverity};
use super::super::error::{internal, Error};
use super::super::etc;
use super::super::exit_guard::{self, Call};
use super::super::image_plan::ImagePlan;
use super::super::owned_kernel::{OwnedModule, OwnedSet, OwnedSets};
use super::super::pvrtc;
use super::super::resource::Identity;
use super::super::resource::ResourceKey;
use super::super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::super::tracker::{Access, Barrier, Use, UseRange};
use super::super::trails::Trail;
use super::super::values::{HazardTrackingMode, ResourceOptions};
use super::super::ycbcr_device::{PairsSampling, YCbCrSampler};
use super::super::{Buffer, Device};
use super::first_move::{FirstMove, Pattern, Patterned, TexelCopy};
use super::twin_decode::{self, Job};
use super::{
    binding_view, buffer_image_copy, texture_aspects, view_error, view_shadow, Image, Owner, Start,
    Texture,
};
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    AstcDecode, AstcTexels, BlockDecodeModule, ChromaDecode, EtcDecode, EtcTexels, ImageViewDesc,
    PvrtcDecode,
};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};
use std::sync::{Arc, Mutex, Weak};

/// What a decoded twin's format must take on the device: sampling, its linear filter and copies
/// into it.
const TWIN_FEATURES: FormatFeatureFlags2 = FormatFeatureFlags2::SAMPLED_IMAGE
    .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
    .union(FormatFeatureFlags2::TRANSFER_DST);

/// The most bytes of texels one band of a decode pass holds: its scratch's texels (and fewer
/// blocks), within `maxStorageBufferRange`.
pub(crate) const BAND_BYTES: u64 = 32 << 20;

/// How a decoded twin's blocks, held in a carrier, decode: an ASTC format's (`astc.rs`) into 8-bit
/// codes or an HDR form's halves, a PVRTC format's (`pvrtc.rs`) into its codes (its linear form's decode, which its
/// sRGB form's is too), an ETC2 or EAC format's (`etc.rs`) into its codes or an 11-bit format's
/// texels (`Device::etc_texels`; its unorm form's decode, which its sRGB form's is too), or a
/// 4:2:2 format's pairs (`chroma_pairs.rs`) into its codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TwinDecode {
    Astc(AstcDecode, AstcTexels),
    Pvrtc(PvrtcDecode),
    Etc(EtcDecode, EtcTexels),
    Chroma(ChromaDecode),
}

impl TwinDecode {
    /// The texels across and down of a block.
    pub(crate) fn block_extent(self) -> [u32; 2] {
        match self {
            TwinDecode::Astc(decode, _) => decode.block_extent(),
            TwinDecode::Pvrtc(decode) => decode.block_extent(),
            TwinDecode::Etc(decode, _) => decode.block_extent(),
            TwinDecode::Chroma(decode) => decode.block_extent(),
        }
    }

    /// Whether a write's decode is of whole levels: a PVRTC texel's colors come from the blocks
    /// around it, wrapping at the level's edges.
    pub(crate) fn whole_levels(self) -> bool {
        matches!(self, TwinDecode::Pvrtc(_))
    }

    /// Whether a write's decode is of whole rows: a 4:2:2 texel's red and blue read the pair
    /// beside it.
    pub(crate) fn whole_rows(self) -> bool {
        matches!(self, TwinDecode::Chroma(_))
    }

    /// The bytes of a block.
    pub(crate) fn block_bytes(self) -> u32 {
        match self {
            TwinDecode::Astc(..) => AstcDecode::BLOCK_BYTES,
            TwinDecode::Pvrtc(_) => PvrtcDecode::BLOCK_BYTES,
            TwinDecode::Etc(decode, _) => decode.block_bytes(),
            TwinDecode::Chroma(_) => ChromaDecode::BLOCK_BYTES,
        }
    }

    /// The bytes of a decoded texel.
    pub(crate) fn texel_bytes(self) -> u32 {
        match self {
            TwinDecode::Astc(decode, texels) => decode.texel_bytes(texels),
            TwinDecode::Pvrtc(_) => PvrtcDecode::TEXEL_BYTES,
            TwinDecode::Etc(decode, texels) => decode.texel_bytes(texels),
            TwinDecode::Chroma(_) => ChromaDecode::TEXEL_BYTES,
        }
    }

    /// The Vulkan format of the decoded texels.
    pub(crate) fn texel_format(self) -> Format {
        match self {
            TwinDecode::Astc(decode, texels) => decode.texel_format(texels),
            TwinDecode::Pvrtc(decode) => decode.texel_format(),
            TwinDecode::Etc(decode, texels) => decode.texel_format(texels),
            TwinDecode::Chroma(decode) => decode.texel_format(),
        }
    }

    /// The Metal pixel format whose texels the decoded ones are.
    fn texel_pixel_format(self) -> PixelFormat {
        match self {
            TwinDecode::Astc(decode, texels) => decode.texel_pixel_format(texels),
            TwinDecode::Pvrtc(decode) => decode.texel_pixel_format(),
            TwinDecode::Etc(decode, texels) => decode.texel_pixel_format(texels),
            TwinDecode::Chroma(decode) => decode.texel_pixel_format(),
        }
    }

    /// The decode of a texture of `pf`, whose texels it decodes as the twin's: the same ASTC
    /// format (an sRGB form is another decode), the same PVRTC, ETC2 or EAC blocks (an sRGB form's
    /// codes are its linear form's), or the same 4:2:2 pairs.
    fn decodes(self, pf: PixelFormat) -> bool {
        match self {
            TwinDecode::Astc(decode, _) => AstcDecode::of(pf) == Some(decode),
            TwinDecode::Pvrtc(decode) => {
                PvrtcDecode::of(pf).map(PvrtcDecode::linear) == Some(decode)
            }
            TwinDecode::Etc(decode, _) => EtcDecode::of(pf) == Some(decode),
            TwinDecode::Chroma(decode) => ChromaDecode::of(pf) == Some(decode),
        }
    }

    /// [`decode_into`](Self::decode_into) into a new buffer.
    pub(crate) fn decode(self, blocks: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let mut texels = vec![0; width * height * depth * self.texel_bytes() as usize];
        self.decode_into(blocks, size, &mut texels);
        texels
    }

    /// The texels of a region `size` whose `blocks` lie as a copy of the region out of the
    /// blocks' image lays them, into `texels`, laid out as a copy into the twin reads them (a
    /// PVRTC region's images whole levels, [`whole_levels`](Self::whole_levels)).
    pub(crate) fn decode_into(self, blocks: &[u8], size: [u32; 3], texels: &mut [u8]) {
        match self {
            TwinDecode::Astc(decode, form) => decode.decode_into(form, blocks, size, texels),
            TwinDecode::Pvrtc(decode) => texels.copy_from_slice(&decode.decode(blocks, size)),
            TwinDecode::Etc(decode, form) => decode.decode_into(form, blocks, size, texels),
            TwinDecode::Chroma(decode) => decode.decode_into(blocks, size, texels),
        }
    }

    /// The texels of rows `first` to `first + rows` of one image `width` x `height` of a region
    /// whose `blocks` (all of the image's) lie as for [`decode_into`](Self::decode_into), into
    /// `texels`: an ASTC, ETC2, EAC or 4:2:2 decode's rows start at a block row (`first` a
    /// multiple of the block's height); a PVRTC image's rows read all its blocks.
    pub(crate) fn decode_rows(
        self,
        blocks: &[u8],
        [width, height]: [u32; 2],
        [first, rows]: [u32; 2],
        texels: &mut [u8],
    ) {
        if let TwinDecode::Pvrtc(decode) = self {
            decode.decode_rows(blocks, [width, height], [first, rows], texels);
            return;
        }
        let [bw, bh] = self.block_extent();
        let row = (width.div_ceil(bw) * self.block_bytes()) as usize;
        let at = (first / bh) as usize * row;
        self.decode_into(&blocks[at..], [width, rows, 1], texels);
    }

    /// The module of `device` whose dispatch decodes its blocks into the twin's texels on the
    /// GPU ([`Image::decode_pass`]), each decode's own kernel on `BlockDecodeModule`'s bindings:
    /// `bind::astc_decode_module`'s, `bind::etc_decode_module`'s and
    /// `bind::chroma_decode_module`'s of its shape, `bind::pvrtc_decode_module`'s of
    /// `PvrtcDecode::shape`.
    pub(crate) fn module(
        self,
        device: &metal2vulkan::device::Device,
    ) -> Result<BlockDecodeModule, metal2vulkan::error::Error> {
        match self {
            TwinDecode::Astc(decode, texels) => {
                metal2vulkan::bind::astc_decode_module(device, decode, texels)
            }
            TwinDecode::Pvrtc(decode) => metal2vulkan::bind::pvrtc_decode_module(device, decode),
            TwinDecode::Etc(decode, texels) => {
                metal2vulkan::bind::etc_decode_module(device, decode, texels)
            }
            TwinDecode::Chroma(decode) => metal2vulkan::bind::chroma_decode_module(device, decode),
        }
    }
}

/// An ASTC, PVRTC, ETC2, EAC or 4:2:2 carrier's decoded twin (the module docs).
pub(super) struct Decoded {
    /// The twin: `decode`'s texels, of the texture's shape, levels and layers.
    pub(super) image: Image,
    /// How the blocks decode.
    pub(super) decode: TwinDecode,
    /// It was made moved alone (`Start::Moved`): its texels start as the CPU's decode of the zero
    /// blocks, written as the texture is made ([`Image::start_decoded_twin`]).
    started_on_the_cpu: bool,
    /// The layout and descriptor of the texture it was made for, by which an ASTC carrier's
    /// second twin is made ([`Image::make_second_twin`]).
    made_for: (TextureLayout, TextureDescriptor),
    /// A 4:2:2 twin's pairs' image, where it holds the device's own conversion of its pairs (the
    /// module docs); `None` where it holds Metal's decode.
    pairs: Option<PairsSampling>,
    /// What a read of it announces, where its texels are not Metal's decode exactly.
    read: Option<TwinRead>,
}

/// What a read of a decoded twin whose texels are not Metal's decode exactly announces
/// (`decoded-texel-precision`; the module docs' **Reads**): the texture, as its notice names it,
/// and why.
#[derive(Clone, Debug)]
pub(crate) struct TwinRead {
    object: Arc<str>,
    why: &'static str,
}

impl TwinRead {
    /// What a read of a twin of `decode` of a texture of `layout` announces (`pairs`: a 4:2:2
    /// twin holds the device's own conversion of its pairs): an ASTC LDR form's 8-bit codes, each
    /// nearest Metal's 12-bit value; an EAC 11-bit format's 16-bit normalized codes, each nearest
    /// Metal's 11-bit value; a 4:2:2 texture's device conversion. `None` for any other twin, whose
    /// texels are Metal's exactly: a PVRTC texture's codes, an ASTC sRGB texture's, an 8-bit ETC2
    /// or EAC texture's, an ASTC HDR texture's halves and a 4:2:2 texture's of Metal's decode.
    fn of(layout: &TextureLayout, decode: TwinDecode, pairs: bool) -> Option<TwinRead> {
        let why = match decode {
            TwinDecode::Chroma(_) if pairs => {
                "its decoded twin holds the device's own Y′CbCr conversion of its pairs (Vulkan's \
                 midpoint reconstruction through a linear chroma filter, at the device's \
                 precision), each channel as the nearest 8-bit code with ties to even, where \
                 Metal's 3:1 sum of the codes rounds its ties up"
            }
            TwinDecode::Astc(decode, AstcTexels::Unorm8) if !decode.srgb => {
                "its decoded twin holds each channel as the 8-bit code nearest Metal's 12-bit \
                 value, at most 1/510 off it"
            }
            TwinDecode::Etc(decode, EtcTexels::Compact) if decode.eleven_bit() => {
                "its decoded twin holds each channel as the 16-bit normalized code nearest \
                 Metal's 11-bit value, at most half a 16-bit step off it"
            }
            _ => return None,
        };
        Some(TwinRead {
            object: carried_object(layout).into(),
            why,
        })
    }

    /// Announce it on `device`: once per device (but at each occurrence where its options ask
    /// for that), where a command may read the twin.
    pub(crate) fn announce(&self, device: &Device) {
        device.degrade(Degradation::DecodedTexelPrecision, || {
            (self.object.to_string(), self.why.to_owned())
        });
    }

    /// Announce it on `device` once per device, whatever its options ask: as the twin's view is
    /// placed in a heap set a pending command buffer may read through, which a submission may do
    /// on a runtime thread (`heap_sets.rs`).
    pub(crate) fn announce_once(&self, device: &Device) {
        device.degrade_once(Degradation::DecodedTexelPrecision, || {
            (self.object.to_string(), self.why.to_owned())
        });
    }
}

/// How a carried texture of `layout` is named in its notices.
fn carried_object(layout: &TextureLayout) -> String {
    let [width, height, depth] = layout.extent;
    format!(
        "a {:?} {:?} texture of {width}x{height}x{depth}",
        layout.pixel_format, layout.texture_type
    )
}

impl Decoded {
    /// The twin's format for a binding's view of the carrier in `format`: the twin's own, or an
    /// 8-bit form's view of its codes; `None` for a format whose view reads the carrier.
    fn view_format(&self, format: Format) -> Option<Format> {
        use Format as F;
        // An ASTC texture's binding views name its candidate's format (its `_sRGB` form's
        // `R8G8B8A8_SRGB`, whose codes its twin holds, `R8G8B8A8_UNORM`, or its `_HDR` form's
        // `R16G16B16A16_SFLOAT`): the twin's own view of it, or a twin's exact values.
        // A PVRTC texture's name the same formats, and the twin holds their codes; so does an
        // ETC2 or EAC 8-bit format's, and an 11-bit one's name its compact texels' format. A
        // 4:2:2 texture's name the codes' `R8G8B8A8_UNORM` alone (it has no sRGB form).
        let named = matches!(format, F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB);
        match self.decode {
            TwinDecode::Astc(astc, texels) => {
                let named = match astc.hdr {
                    true => format == F::R16G16B16A16_SFLOAT,
                    false => named,
                };
                named.then_some(match astc.form(texels) {
                    AstcTexels::Unorm8 => format,
                    _ => self.decode.texel_format(),
                })
            }
            TwinDecode::Pvrtc(_) => named.then_some(format),
            TwinDecode::Etc(etc, texels) if etc.eleven_bit() => {
                let compact = etc.texel_format(EtcTexels::Compact);
                (format == compact).then_some(etc.texel_format(texels))
            }
            TwinDecode::Etc(..) => named.then_some(format),
            TwinDecode::Chroma(_) => (format == F::R8G8B8A8_UNORM).then_some(format),
        }
    }

    /// The bytes of memory placed for the twin.
    fn placed(&self) -> Option<u64> {
        placed(&self.image)
    }

    /// The twin's texels' bytes and its blocks', every level and layer.
    fn bytes(&self) -> (u64, u64) {
        let plan = self.image.plan();
        let decode = self.decode;
        let [bw, bh] = decode.block_extent().map(u64::from);
        (0..plan.levels)
            .map(|level| {
                let [w, h, d] = plan.extent.map(|axis| u64::from((axis >> level).max(1)));
                let layers = u64::from(plan.layers);
                let texels = w * h * d * layers * u64::from(decode.texel_bytes());
                let blocks = w.div_ceil(bw) * h.div_ceil(bh) * d * layers;
                (texels, blocks * u64::from(decode.block_bytes()))
            })
            .fold((0, 0), |(t, b), (texels, blocks)| (t + texels, b + blocks))
    }
}

/// The bytes of memory the runtime placed `image` in, where it is memory of its own.
fn placed(image: &Image) -> Option<u64> {
    match &image.owner {
        Owner::Runtime { placement, .. } => Some(placement.size()),
        _ => None,
    }
}

/// The plan of the decoded twin of the carrier of `plan`, a texture of `layout`, whose blocks
/// decode by `decode`: the texture's type, shape (its extent, which an ASTC carrier's is not),
/// levels, layers and cube compatibility, of the decoded texels' format (mutable for an sRGB
/// view of 8-bit codes; `TwinDecode::texel_pixel_format`'s), sampled and copied into. `routed`:
/// the blocks' CPU copies go on the host or the staging queue, which the twin starts on and takes
/// the CPU's decodes through ([`Image::decode_written`]), copied both ways as the staging queue's
/// images are (`Device::stages`).
pub(crate) fn twin_plan(
    plan: &ImagePlan,
    layout: &TextureLayout,
    decode: TwinDecode,
    routed: bool,
) -> ImagePlan {
    let format = decode.texel_format();
    // A cube texture's twin is cube compatible, as its image would be (a carrier is not).
    let wanted = super::super::image_plan::cube_flags(layout);
    let (flags, view_formats) = match format {
        Format::R8G8B8A8_UNORM => (
            wanted | ImageCreateFlags::MUTABLE_FORMAT,
            vec![Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB],
        ),
        _ => (wanted, Vec::new()),
    };
    let copies = match routed {
        true => ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        false => ImageUsageFlags::TRANSFER_DST,
    };
    ImagePlan {
        // Its copies' units are its texels' (`copy_units`), not the carrier's blocks.
        pixel_format: decode.texel_pixel_format(),
        flags,
        image_type: plan.image_type,
        format,
        extent: layout.extent,
        levels: plan.levels,
        layers: plan.layers,
        samples: plan.samples,
        usage: ImageUsageFlags::SAMPLED | copies,
        view_formats,
        mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
        unavailable: plan.unavailable,
    }
}

impl Device {
    /// The texels of the decoded twin of an ASTC texture of `decode` on this device: an LDR or
    /// sRGB form's 8-bit codes, an LDR form's each the code nearest Metal's 12-bit value
    /// (`decoded-texel-precision`), an sRGB form's its own, which its `R8G8B8A8_SRGB` view
    /// decodes, the device's sRGB decode (`metal::texel_format::srgb_coded`); an HDR form's
    /// halves, Metal's values exactly. Refused where the device does not take their format so (no
    /// conformant device).
    fn astc_texels(&self, decode: AstcDecode) -> Result<AstcTexels, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let texels = decode.form(AstcTexels::Unorm8);
        if facts
            .optimal(decode.texel_format(texels))
            .contains(TWIN_FEATURES)
        {
            return Ok(texels);
        }
        Err(Error::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: decode.texel_format(texels),
                usage: FormatUsage::SampledImageFilterLinear,
            },
        )))
    }

    /// The texels of the decoded twin of an ETC2 or EAC texture of `decode` on this device: an
    /// 8-bit format's codes; an 11-bit format's 16-bit normalized codes where the device samples,
    /// filters and copies into their format, else its values exactly. Refused where the device
    /// takes neither so.
    fn etc_texels(&self, decode: EtcDecode) -> Result<EtcTexels, Error> {
        if !decode.eleven_bit() {
            return Ok(EtcTexels::Compact);
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let takes = |texels: EtcTexels| {
            facts
                .optimal(decode.texel_format(texels))
                .contains(TWIN_FEATURES)
        };
        let order = [EtcTexels::Compact, EtcTexels::Float];
        order
            .into_iter()
            .find(|&texels| takes(texels))
            .ok_or_else(|| {
                Error::Unsupported(Unsupported::needs(
                    Feature::Format,
                    Capability::Format {
                        format: decode.texel_format(order[0]),
                        usage: FormatUsage::SampledImageFilterLinear,
                    },
                ))
            })
    }

    /// `image`, a new texture's image of `layout` made with `descriptor`, its own for good, with
    /// its decoded twin where it is a carrier (the module docs), announced
    /// (`carried-decoded-twin`); any other image as it is, a BC texture's announced
    /// `compressed-texel-decode` where the device's decode is not measured
    /// ([`degrade_block_decode`](Device::degrade_block_decode)). Refused where the device does
    /// not take the twin's format (no conformant device). `fresh`: the image holds the zeros of a
    /// new texture, which the twin starts as (where the image takes host or staging copies, by
    /// the decode the caller stages once the image is shared, [`Image::start_decoded_twin`]);
    /// else the caller decodes the whole twin once the image is shared
    /// ([`Image::decode_whole`]).
    pub(super) fn with_decoded_twin(
        &self,
        mut image: Image,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        fresh: bool,
    ) -> Result<Image, Error> {
        // An ASTC, PVRTC, ETC2, EAC or 4:2:2 carrier's texels are read through its twin alone
        // (`astc.rs`, `pvrtc.rs`, `etc.rs`, `chroma_pairs.rs`).
        let plan = image.plan();
        let carried = if let Some(decode) = astc::decode_of(plan) {
            Some(TwinDecode::Astc(decode, self.astc_texels(decode)?))
        } else if let Some(decode) = etc::decode_of(plan) {
            Some(TwinDecode::Etc(decode, self.etc_texels(decode)?))
        } else {
            (pvrtc::decode_of(plan).map(TwinDecode::Pvrtc))
                .or_else(|| chroma_pairs::decode_of(plan).map(TwinDecode::Chroma))
        };
        let Some(decode) = carried else {
            self.degrade_block_decode(layout);
            return Ok(image);
        };
        // A 4:2:2 twin holds the device's own conversion of the pairs where it converts their
        // format with Metal's reconstruction (the module docs). Every decode is then a GPU pass:
        // the carrier's CPU copies go through submissions, each followed by its pass.
        let pairs = match decode {
            TwinDecode::Chroma(chroma) => {
                let [width, height, _] = layout.extent;
                YCbCrSampler::pairs(self, chroma, [width, height])
                    .map(|sampler| PairsSampling::new(self, sampler, [width, height]))
                    .transpose()?
            }
            _ => None,
        };
        if pairs.is_some() {
            image.host_copies = false;
            image.staging_copies = false;
        }
        // A twin starts, and takes the CPU's decodes, as the carrier's CPU copies go: on the
        // host or the staging queue where they do (then made as a `Shared` or `Managed`
        // texture's image is), else through submissions.
        let routed = image.host_copies || image.staging_copies;
        let plan = twin_plan(image.plan(), layout, decode, routed);
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        if !facts.optimal(plan.format).contains(TWIN_FEATURES) {
            return Err(Error::Unsupported(Unsupported::needs(
                Feature::Format,
                Capability::Format {
                    format: plan.format,
                    usage: FormatUsage::SampledImageFilterLinear,
                },
            )));
        }
        let mut own = *descriptor;
        if !routed {
            own.resource_options = ResourceOptions::private();
        }
        // Its uses are the image's (the twin shares its resource identity): its own goes unused.
        // A routed twin is moved to `GENERAL` alone as it is made, so a CPU decode may write it
        // from then on; its texels start as the CPU's decode of the zero blocks, written on the
        // host or the staging queue as the texture is made (`Image::start_decoded_twin`): 6 to 10 ms for
        // 2048 x 2048. No GPU decode the runtime stages of its own is in flight to race a later
        // write's CPU decode.
        let identity = Identity::new(HazardTrackingMode::Default, false);
        let start = match routed {
            true => Start::Moved,
            false => Start::Zero,
        };
        let twin =
            self.new_image_starting(plan, layout, &own, identity, start, Device::runtime_owner)?;
        let moved = twin.host_copies || twin.staging_copies;
        // A twin that takes neither copy while its blocks take one could take no CPU decode
        // behind work in flight ([`Image::decode_written`]): the blocks' CPU copies then go
        // through submissions too, in order with the twin's decodes. No measured device makes
        // such a twin (its formats take the copies its blocks take).
        if routed && !moved {
            image.host_copies = false;
            image.staging_copies = false;
        }
        // A new texture's zero blocks: an ASTC texture's error blocks of magenta, an ETC2 or EAC
        // texture's grey, opaque black or 11-bit 1, a 4:2:2 texture's opaque black; a PVRTC
        // texture's decode to zero, which the twin's first move writes.
        if let Some(color) = zero_clear(decode).filter(|_| fresh && !routed) {
            let plan = twin.plan();
            let patterned = Patterned::new(Pattern::Color(color), plan.levels, plan.levels);
            let first = FirstMove::new(
                twin.raw.get(),
                plan.format,
                vk::ImageLayout::UNDEFINED,
                None,
            );
            *lock(&twin.first_move) = Some(first.patterned(patterned));
        }
        let read = TwinRead::of(layout, decode, pairs.is_some());
        let decoded = Decoded {
            image: twin,
            decode,
            started_on_the_cpu: routed && fresh,
            made_for: (*layout, *descriptor),
            pairs,
            read,
        };
        self.degrade_carried_twin(layout, &decoded, placed(&image));
        image.decoded = Some(Box::new(decoded));
        Ok(image)
    }

    /// Announce, for an ASTC, PVRTC, ETC2, EAC or 4:2:2 texture of `layout` made with `decoded`:
    /// `carried-decoded-twin`, its cost (the twin's texels' bytes and memory beside its blocks'
    /// carrier). A twin whose texels are not Metal's decode exactly announces
    /// `decoded-texel-precision` where a command may first read it, not here ([`TwinRead`]).
    fn degrade_carried_twin(&self, layout: &TextureLayout, decoded: &Decoded, image: Option<u64>) {
        self.degrade(Degradation::CarriedDecodedTwin, || {
            let (texels, blocks) = decoded.bytes();
            let placed =
                |bytes: Option<u64>| bytes.map_or_else(String::new, |n| format!(", {n} placed"));
            let held = match decoded.decode {
                TwinDecode::Chroma(_) if decoded.pairs.is_some() => {
                    "Vulkan views a 4:2:2 format only through a sampler Y′CbCr conversion, which \
                     no pipeline's layout bakes, so its twin holds the device's conversion of its \
                     pairs"
                        .to_owned()
                }
                TwinDecode::Chroma(_) => "the device does not convert its 4:2:2 format with \
                                          Metal's reconstruction (its chroma at the midpoint \
                                          through a linear filter)"
                    .to_owned(),
                _ => format!("the device has no {:?} format", layout.pixel_format),
            };
            let why = format!(
                "{held}: its blocks ({blocks} bytes) lie in a carrier image{}, which copies \
                 move, and its samples read a decoded twin of {texels} bytes of {:?} texels{}, \
                 decoded after each write into it",
                placed(image),
                decoded.decode.texel_format(),
                placed(decoded.placed()),
            );
            (carried_object(layout), why)
        });
    }
}

impl Texture {
    /// The `VkImageView` of `desc` (a sampled view; `call`'s) of the texture's decoded twin, where
    /// its image (`image`) has one whose decode the texture's format takes (the module docs; an
    /// ASTC carrier's second twin for a view in the texture's other colour space): made at its
    /// first bind, then kept with the twin. With what a read through it announces, where the
    /// twin's texels are not Metal's decode exactly ([`TwinRead`]): the caller announces it where
    /// a command may read the view. `None` elsewhere (an image with one presents no stand-in: its
    /// detach is refused, `stand_in.rs`).
    pub(super) fn decoded_view<'a>(
        &self,
        call: &'static str,
        image: &'a Image,
        desc: &ImageViewDesc,
    ) -> Result<Option<(vk::ImageView, Option<&'a TwinRead>)>, Error> {
        let Some(decoded) = image.twin_of(self.0.record.pixel_format) else {
            return Ok(None);
        };
        let Some(format) = decoded.view_format(desc.format) else {
            return Ok(None);
        };
        let mut twin = *desc;
        twin.format = format;
        let view = binding_view(decoded.image.plan(), &self.0.record, &twin, false)
            .map_err(|refusal| view_error(call, refusal))?;
        let view = decoded.image.view(&view)?;
        Ok(Some((view, decoded.read.as_ref())))
    }

    /// Whether the texture's texels are held in an image with a decoded twin
    /// ([`decoding_image`](Self::decoding_image)).
    pub(crate) fn has_decoded_twin(&self) -> bool {
        self.decoding_image().is_some()
    }

    /// The image the texture's texels are held in where it has a decoded twin: its image, or a
    /// shadowed view's texture's (`view_shadow.rs`), whose twin a write through the view decodes
    /// into once its copy back is made. `None` for a texture without one.
    fn decoding_image(&self) -> Option<Arc<Image>> {
        let image = view_shadow::truth(self.image().ok()?);
        image.decoded.is_some().then(|| Arc::clone(image))
    }
}

/// The texel every texel of a zero block decodes to, as `decode` writes it: `None` where they
/// differ (no format so far), when the zero blocks are decoded instead.
fn zero_texel(decode: TwinDecode) -> Option<Vec<u8>> {
    let [bw, bh] = decode.block_extent();
    let texels = decode.decode(&vec![0; decode.block_bytes() as usize], [bw, bh, 1]);
    let size = decode.texel_bytes() as usize;
    let first = texels.get(..size)?;
    texels
        .chunks_exact(size)
        .all(|texel| texel == first)
        .then(|| first.to_vec())
}

/// The clear color of a twin whose zero blocks decode to texels other than zeros: an ASTC
/// texture's magenta (an error block; an HDR one's NaN), an ETC2 texture's grey of codes 2, an EAC
/// 11-bit texture's value of 1 (signed -3) or its compact code; `None` where they decode to
/// zeros, which a new image's first move writes. The clear converts each value back to the code
/// it came from: a normalized code over its largest is within half a step of it.
fn zero_clear(decode: TwinDecode) -> Option<vk::ClearColorValue> {
    let texel = zero_texel(decode)?;
    if texel.iter().all(|&byte| byte == 0) {
        return None;
    }
    let mut float32 = [0.0; 4];
    match decode.texel_format() {
        Format::R8G8B8A8_UNORM => {
            for (value, &code) in float32.iter_mut().zip(&texel) {
                *value = f32::from(code) / 255.0;
            }
        }
        Format::R32G32B32A32_SFLOAT | Format::R32G32_SFLOAT | Format::R32_SFLOAT => {
            for (value, bytes) in float32.iter_mut().zip(texel.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*bytes);
            }
        }
        // An ASTC HDR texture's halves: its zero blocks' NaN.
        Format::R16G16B16A16_SFLOAT => {
            let TwinDecode::Astc(astc, _) = decode else {
                return None;
            };
            let halves = texel
                .as_chunks::<2>()
                .0
                .iter()
                .map(|h| u16::from_le_bytes(*h));
            let halves: Vec<u16> = halves.collect();
            float32 = astc.metal_value(halves.try_into().ok()?);
        }
        Format::R16_UNORM | Format::R16G16_UNORM => {
            for (value, bytes) in float32.iter_mut().zip(texel.as_chunks::<2>().0) {
                *value = f32::from(u16::from_le_bytes(*bytes)) / 65535.0;
            }
        }
        Format::R16_SNORM | Format::R16G16_SNORM => {
            for (value, bytes) in float32.iter_mut().zip(texel.as_chunks::<2>().0) {
                *value = (f32::from(i16::from_le_bytes(*bytes)) / 32767.0).max(-1.0);
            }
        }
        _ => return None,
    }
    Some(vk::ClearColorValue { float32 })
}

/// The copies of `band`'s texels, laid out tight in rows of `row` bytes, in parts of whole block
/// rows of at least `twin_decode::ROWS_PER_PART` texel rows, at most `twin_decode::THREADS`, each
/// reading its rows at its offset: disjoint texels, which host copies may write at once.
fn host_parts(band: &ImageRegion, row: usize) -> Vec<vk::BufferImageCopy> {
    let height = band.size[1];
    let parts = (height / twin_decode::ROWS_PER_PART).clamp(1, twin_decode::THREADS as u32);
    let per = height.div_ceil(parts).next_multiple_of(4);
    (0..height)
        .step_by(per as usize)
        .map(|y| {
            let part = ImageRegion {
                origin: [band.origin[0], band.origin[1] + y, band.origin[2]],
                size: [band.size[0], per.min(height - y), band.size[2]],
                ..*band
            };
            vk::BufferImageCopy {
                buffer_offset: (y as usize * row) as u64,
                ..buffer_image_copy(&part)
            }
        })
        .collect()
}

/// The part of `region` that is texel rows `y` to `y + height` of its image `image` (its layer,
/// or its depth slice in a volume).
fn band_region(region: &ImageRegion, image: u32, y: u32, height: u32) -> ImageRegion {
    let [x, top, z] = region.origin;
    let (layer, z) = match region.layers > 1 {
        true => (region.layer + image, z),
        false => (region.layer, z + image),
    };
    ImageRegion {
        layer,
        layers: 1,
        origin: [x, top + y, z],
        size: [region.size[0], height, 1],
        ..*region
    }
}

/// Every texel of an image of `plan`: one region a level, of all its layers.
fn whole_regions(plan: &ImagePlan) -> Vec<ImageRegion> {
    let volume = plan.image_type == ImageType::TYPE_3D;
    (0..plan.levels)
        .map(|level| {
            let [w, h, d] = plan.extent.map(|axis| (axis >> level).max(1));
            ImageRegion {
                aspect: ImageAspectFlags::COLOR,
                level,
                layer: 0,
                layers: plan.layers,
                origin: [0, 0, 0],
                size: [w, h, if volume { d } else { 1 }],
            }
        })
        .collect()
}

impl Image {
    /// Decode the whole image, just made and shared, into its decoded twin before any command
    /// reads it: a heap's texture's, which holds what its bytes held. Staged for the device's
    /// next submission as a host call's copy is (`TexelCopy`), or past the bound made now; before
    /// its texture is made, so the texture takes its resource ID only once its making cannot
    /// fail (`resource_ids.rs`). Nothing for an image without one.
    pub(super) fn decode_whole(self: &Arc<Self>) -> Result<(), Error> {
        if self.decoded.is_none() {
            return Ok(());
        }
        self.stage_decode(&whole_regions(self.plan()))
    }

    /// Start the decoded twin of the image, just made and shared, where it was made moved alone
    /// (`Start::Moved`, [`Device::with_decoded_twin`]): the decode of a new texture's zero
    /// blocks, every level's, queued for the device's decode thread (`twin_decode.rs`), which
    /// fills the twin with it on the host or the staging queue as its copies go; before its
    /// texture is made, so the texture takes its resource ID only once its making cannot fail
    /// (`resource_ids.rs`). Nothing for any other image.
    pub(super) fn start_decoded_twin(self: &Arc<Self>) -> Result<(), Error> {
        if !self
            .decoded
            .as_deref()
            .is_some_and(|d| d.started_on_the_cpu)
        {
            return Ok(());
        }
        for region in whole_regions(self.plan()) {
            self.queue_twin_decode(&region, None)?;
        }
        Ok(())
    }

    /// Stage the decode of `regions` of the image into its decoded twin for the device's next
    /// submission (`TexelCopy::decode_only`), what the image holds there already: a heap's
    /// bytes. Past the bound, made now. Nothing for an image without a twin. (no longer a host
    /// call's copy's, which the CPU decodes, [`decode_written`](Self::decode_written).)
    pub(super) fn stage_decode(self: &Arc<Self>, regions: &[ImageRegion]) -> Result<(), Error> {
        let firsts = &self.device.shared().textures.first_moves;
        let Some(pass) = self.decode_pass(regions, PassScratch::Device)? else {
            return Ok(());
        };
        let staged = TexelCopy::decode_only(pass);
        let keeper: Weak<Image> = Arc::downgrade(self);
        if firsts.register_texels(keeper, staged) {
            self.submit_copy(|_, _| {})?;
        }
        Ok(())
    }

    /// The decode into the image's decoded twin of `region`, which a host call just wrote into
    /// the image on the host or the staging queue as the bytes `packed` makes (its blocks, laid
    /// out as [`buffer_image_copy`] reads them, made only for an image with a twin): queued for the device's decode thread, which decodes it
    /// on the CPU and copies the texels into the twin as the twin's copies go
    /// (`twin_decode.rs`; announced `cpu-block-decode`). The first batch that may read the
    /// texture waits at its submission for it (every batch, where the texture is untracked);
    /// the call waits for nothing: a batch submitted before it reads the twin as the thread's
    /// copies stand when it runs, as Metal's unsynchronized CPU write leaves the ordering to the
    /// program. Always on the CPU: a decode staged for the GPU could run while a later write's CPU
    /// decode copies into the twin, read the old blocks before it and write their texels after it
    /// (lavapipe under a lane's load, master `7d426aff`). An ASTC carrier's second twin's decode
    /// goes to the device's next batch ([`queue_second_decode`](Self::queue_second_decode)).
    /// A 4:2:2 carrier's write of part of its rows decodes the whole rows as the image then holds
    /// them ([`written_rows`](Self::written_rows)). Nothing for an image without a twin.
    pub(super) fn decode_written(
        self: &Arc<Self>,
        region: &ImageRegion,
        packed: impl FnOnce() -> Vec<u8>,
    ) -> Result<(), Error> {
        let Some(decoded) = self.decoded.as_deref() else {
            return Ok(());
        };
        let packed = packed();
        let host = decoded.image.takes_host_copies();
        let (region, packed) = self.written_rows(decoded, region, packed)?;
        self.queue_twin_decode(&region, Some(packed))?;
        self.queue_second_decode(std::slice::from_ref(&region))?;
        let device = &self.device;
        let object = || {
            let plan = self.plan();
            let [width, height, depth] = plan.extent;
            format!(
                "a {:?} texture of {width}x{height}x{depth}",
                plan.pixel_format
            )
        };
        device.degrade(Degradation::CpuBlockDecode, || {
            let route = match host {
                true => "on the host",
                false => "on the staging queue",
            };
            let why = format!(
                "a host call's write into it decodes the written blocks on the CPU, on the \
                 device's decode thread, and copies the texels into its decoded twin {route}; \
                 the first command buffer that may read it waits at its submission for that"
            );
            (object(), why)
        });
        Ok(())
    }

    /// `region` and its blocks `packed`, a host call's write into the image, as its twin's decode
    /// reads them: for a decode of whole rows ([`TwinDecode::whole_rows`], a 4:2:2 carrier's) of a
    /// region short of its rows, the rows' region and their bytes as the image holds them after
    /// the write, read back on the host or the staging queue as the write went (the caller waits
    /// for that read); else themselves.
    fn written_rows(
        &self,
        decoded: &Decoded,
        region: &ImageRegion,
        packed: Vec<u8>,
    ) -> Result<(ImageRegion, Vec<u8>), Error> {
        let rows = self.twin_region(decoded, region);
        if !decoded.decode.whole_rows() || rows == *region {
            return Ok((*region, packed));
        }
        let decode = decoded.decode;
        let [bw, bh] = decode.block_extent();
        let [width, height, depth] = rows.size.map(|axis| axis as usize);
        let image = width.div_ceil(bw as usize)
            * height.div_ceil(bh as usize)
            * decode.block_bytes() as usize;
        let mut bytes = vec![0; image * depth * rows.layers as usize];
        let copy = buffer_image_copy(&rows);
        let read = match self.takes_host_copies() {
            true => self.host_read(copy, &mut bytes)?,
            false => {
                let len = bytes.len() as u64;
                self.staging_read(copy, len, |staged| bytes.copy_from_slice(staged))?
            }
        };
        if !read {
            return Err(internal(
                "a 4:2:2 texture's rows that took no CPU copy its write took",
            ));
        }
        Ok((rows, bytes))
    }

    /// Queue the decode of `region` into the image's twin for the device's decode thread
    /// (`blocks`, the written blocks; `None`, a new texture's zero blocks): the scheduler holds
    /// each batch that may read the texture until it finished (every batch, where the texture is
    /// untracked: no footprint names it). The call waits for nothing but room in the queue.
    fn queue_twin_decode(
        self: &Arc<Self>,
        region: &ImageRegion,
        blocks: Option<Vec<u8>>,
    ) -> Result<(), Error> {
        let shared = self.device.shared();
        let decodes = &shared.textures.twin_decodes;
        decodes.start()?;
        let (key, tracked) = (self.identity.key, self.identity.tracked);
        shared.scheduler.decode_queued(key, tracked);
        decodes.queue(Job {
            image: Arc::clone(self),
            region: *region,
            blocks,
        });
        Ok(())
    }

    /// On the device's decode thread: decode `blocks` (`region`'s; `None`, the zero blocks) into
    /// the twin ([`write_decoded`](Self::write_decoded)) and count a write's decode;
    /// [`twin_decoded`](Self::twin_decoded) then lets the batches that wait for it go. A failure
    /// is said on the device's message sink as an error (`metal2vulkan-twin-decode-failed`), the
    /// device's first, and each later one counted (`WorkCounts::twin_decode_failed_withheld`): the
    /// call that queued it has returned.
    pub(super) fn finish_twin_decode(
        self: &Arc<Self>,
        region: &ImageRegion,
        blocks: Option<&[u8]>,
    ) {
        let shared = self.device.shared();
        let result = match self.decoded.as_deref() {
            Some(decoded) => self.write_decoded(decoded, region, blocks),
            None => Ok(false),
        };
        match result {
            Ok(_) if blocks.is_some() => shared.counts.cpu_block_decode(),
            Ok(_) => {}
            Err(error) => self.device.tell(Told::TwinDecodeFailed, || {
                let plan = self.plan();
                twin_decode_failed(plan.pixel_format, plan.extent, &error)
            }),
        }
    }

    /// On the device's decode thread, a job decoded, or a start a later queued write overwrites
    /// whole skipped (its texels are never read): the batches that wait for its decodes may go
    /// (those of a skipped start once that write finished too). The trail opened just before
    /// (`trails.rs`), which the thread closes once it let go of the job: from here a batch may run
    /// and complete while the job still holds this image, and the device with it.
    pub(super) fn twin_decoded(&self) -> Trail {
        let shared = self.device.shared();
        let trail = shared.trails.open();
        let identity = &self.identity;
        shared
            .scheduler
            .decode_finished(identity.key, identity.tracked);
        trail
    }

    /// Decode `blocks`, `region`'s (`None`: the zero blocks, a constant fill), on the CPU and copy
    /// the texels into the twin of `decoded` (the image's) as the twin's copies go, in bands of at
    /// most `twin_decode::BAND_BYTES` of texels, each decoded on up to `twin_decode::THREADS`
    /// threads: whether on the host (else on the staging queue). An internal error where the twin
    /// takes neither copy, or took none.
    fn write_decoded(
        self: &Arc<Self>,
        decoded: &Decoded,
        region: &ImageRegion,
        blocks: Option<&[u8]>,
    ) -> Result<bool, Error> {
        let twin = &decoded.image;
        let host = twin.takes_host_copies();
        // The image's CPU copies go on the host or the staging queue only where its twin's do
        // ([`Device::with_decoded_twin`]), and a twin presents no stand-in.
        if !host && !twin.takes_staging_copies() {
            return Err(internal(
                "a decoded twin that takes no CPU copy its blocks took",
            ));
        }
        let decode = decoded.decode;
        let block = decode.block_bytes() as usize;
        let texel = decode.texel_bytes() as usize;
        // The twin's region: a carrier's region in the texture's texels (a PVRTC one's whole
        // levels, a 4:2:2 one's whole rows, whose blocks `decode_written` read).
        let region = &self.twin_region(decoded, region);
        let [bw, bh] = decode.block_extent();
        let [width, height, depth] = region.size;
        let image_blocks = width.div_ceil(bw) as usize * height.div_ceil(bh) as usize * block;
        // Texel rows a band, of `width` texels: whole block rows, but for a whole-level decode,
        // each of whose bands reads all its image's blocks.
        let fit = u32::try_from(twin_decode::BAND_BYTES / (width as usize * texel))
            .unwrap_or(u32::MAX)
            .max(1);
        let rows = match decode.whole_levels() {
            true => fit,
            false => (fit / bh).max(1) * bh,
        };
        let fill = blocks.is_none().then(|| zero_texel(decode));
        for image in 0..depth * region.layers {
            for y in (0..height).step_by(rows as usize) {
                let h = rows.min(height - y);
                let mut texels = vec![0; width as usize * h as usize * texel];
                match (blocks, &fill) {
                    (Some(blocks), _) => {
                        let at = image as usize * image_blocks;
                        let image = &blocks[at..at + image_blocks];
                        let size = [width, height];
                        twin_decode::decode_band(decode, image, size, [y, h], &mut texels);
                    }
                    (None, Some(Some(texel))) => {
                        for out in texels.chunks_exact_mut(texel.len()) {
                            out.copy_from_slice(texel);
                        }
                    }
                    (None, _) => {
                        let zeros = vec![0; image_blocks];
                        let size = [width, height];
                        twin_decode::decode_band(decode, &zeros, size, [y, h], &mut texels);
                    }
                }
                let band = band_region(region, image, y, h);
                let written = match host {
                    true => {
                        let row = width as usize * texel;
                        let copies = host_parts(&band, row);
                        twin.host_write_disjoint(&copies, &texels, twin_decode::THREADS)?
                    }
                    false => {
                        let (copy, len) = (buffer_image_copy(&band), texels.len() as u64);
                        let fill = |staged: &mut [u8]| staged.copy_from_slice(&texels);
                        twin.staging_write(self, copy, len, fill)?
                    }
                };
                if !written {
                    return Err(internal(
                        "a decoded twin that took no CPU copy its blocks took",
                    ));
                }
            }
        }
        Ok(host)
    }

    /// The twin's region of `region` of the image: a PVRTC carrier's whole levels of the
    /// region's images, whose texels each write changes around it ([`TwinDecode::whole_levels`]);
    /// a 4:2:2 carrier's whole rows of the region ([`TwinDecode::whole_rows`]); an ASTC carrier's
    /// region in the texture's texels (`astc::texture_region`); any other image's itself.
    fn twin_region(&self, decoded: &Decoded, region: &ImageRegion) -> ImageRegion {
        let extent = decoded.image.plan().extent;
        let level = |axis: usize| (extent[axis] >> region.level.min(31)).max(1);
        if decoded.decode.whole_levels() {
            return ImageRegion {
                origin: [0, 0, region.origin[2]],
                size: [level(0), level(1), region.size[2]],
                ..*region
            };
        }
        if decoded.decode.whole_rows() {
            return ImageRegion {
                origin: [0, region.origin[1], region.origin[2]],
                size: [level(0), region.size[1], region.size[2]],
                ..*region
            };
        }
        astc::texture_region(self.plan(), extent, *region).unwrap_or(*region)
    }

    /// The plan of the step of a decode pass that decodes `band` (of the twin's texels, one of
    /// [`bands`]): the image's region its blocks are copied out of and their bytes, its shape and
    /// its workgroups. A PVRTC band's blocks are its images' whole levels' (`pvrtc::image_region`
    /// of them), its shape `PvrtcDecode::shape`; any other band's blocks lie over it (an ASTC
    /// carrier's region over it, `astc::image_region`).
    fn band_plan(&self, decode: TwinDecode, band: &ImageRegion) -> BandPlan {
        let [w, h, _] = band.size;
        let images = band.size[2] * band.layers;
        let local = BlockDecodeModule::LOCAL_SIZE;
        let groups = [w.div_ceil(local), h.div_ceil(local), images];
        if let TwinDecode::Pvrtc(pvrtc) = decode {
            let height = (self.decoded_extent(1) >> band.level.min(31)).max(1);
            let whole = ImageRegion {
                origin: [0, 0, band.origin[2]],
                size: [w, height, band.size[2]],
                ..*band
            };
            return BandPlan {
                blocks: pvrtc::image_region(self.plan(), whole),
                block_bytes: (pvrtc.level_bytes([w, height]) * images as usize) as u64,
                shape: PvrtcDecode::shape([w, height], [band.origin[1], h], images),
                groups,
            };
        }
        let [bw, bh] = decode.block_extent();
        let blocks = u64::from(w.div_ceil(bw)) * u64::from(h.div_ceil(bh)) * u64::from(images);
        BandPlan {
            blocks: astc::image_region(self.plan(), *band).unwrap_or(*band),
            block_bytes: blocks * u64::from(decode.block_bytes()),
            shape: BlockDecodeModule::shape([w, h, images]),
            groups,
        }
    }

    /// The decoded twin's level-0 extent along `axis` (the texture's), 1 for an image without one.
    fn decoded_extent(&self, axis: usize) -> u32 {
        self.decoded
            .as_deref()
            .map_or(1, |decoded| decoded.image.plan().extent[axis])
    }

    /// The pass that decodes `regions` (of the image's levels and layers, in its texels: a
    /// carrier's) of the image into its decoded twin (the module docs), its scratch (as `scratch`
    /// says) and sets made; `None` for an image without one, or regions of no texel. Its steps are
    /// bands of the twin's regions (a carrier's in the texture's texels, a PVRTC one's whole
    /// levels, `twin_region`), each band's blocks copied out of the image's region over them (a
    /// PVRTC band's, its images' whole levels; [`band_plan`](Self::band_plan)).
    pub(super) fn decode_pass(
        &self,
        regions: &[ImageRegion],
        scratch: PassScratch,
    ) -> Result<Option<DecodePass>, Error> {
        match self.decoded.as_deref() {
            Some(decoded) => self.decode_pass_into(decoded, regions, scratch),
            None => Ok(None),
        }
    }

    /// [`decode_pass`](Self::decode_pass) into `decoded`, one of the image's twins.
    fn decode_pass_into(
        &self,
        decoded: &Decoded,
        regions: &[ImageRegion],
        scratch: PassScratch,
    ) -> Result<Option<DecodePass>, Error> {
        let decode = decoded.decode;
        let device = &self.device;
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(16);
        let most = u64::from(limits.max_storage_buffer_range).min(BAND_BYTES);
        let texel = u64::from(decode.texel_bytes());
        // A whole-level decode's regions of one level and its images are one region.
        let mut twins: Vec<ImageRegion> = (regions.iter())
            .map(|region| self.twin_region(decoded, region))
            .collect();
        twins.dedup();
        let steps: Vec<ImageRegion> = (twins.iter())
            .flat_map(|region| bands(region, decode, most))
            .collect();
        if steps.is_empty() {
            return Ok(None);
        }
        // A step's texels in whole words: an `R11` format's compact texels are 2 bytes, two to
        // a word, which the kernel writes whole (`bind::etc_decode_module`).
        let texels_of = |r: &ImageRegion| {
            let [w, h, _] = r.size.map(u64::from);
            (w * h * u64::from(r.size[2]) * u64::from(r.layers) * texel).next_multiple_of(4)
        };
        let mut plans: Vec<BandPlan> = steps
            .iter()
            .map(|step| self.band_plan(decode, step))
            .collect();
        // A 4:2:2 twin of the device's conversion: each band's rows sampled out of the pairs'
        // image (the module docs), its own shape and workgroups.
        if let Some(pairs) = &decoded.pairs {
            for (plan, step) in plans.iter_mut().zip(&steps) {
                (plan.shape, plan.groups) = pairs.step(step.origin[1], step.size[1]);
            }
        }
        // Each step's blocks at the scratch's start and its texels at the alignment after them:
        // at most `most` bytes together (`bands`), so within `most` and the alignment.
        let layouts: Vec<(u64, u64, u64)> = (steps.iter().zip(&plans))
            .map(|(step, plan)| {
                let blocks = plan.block_bytes;
                (blocks, blocks.div_ceil(align) * align, texels_of(step))
            })
            .collect();
        let needed = (layouts.iter())
            .map(|&(_, texels_at, texels)| texels_at + texels)
            .max()
            .unwrap_or(0);
        let (scratch, raw_scratch, base, bytes) = match scratch {
            PassScratch::Own => {
                let own = device.own_buffer(needed, ResourceOptions::private())?;
                let (raw, base) = own.at(0);
                (Some(own), raw, base, needed)
            }
            PassScratch::Device => {
                if needed > most + align {
                    return Err(internal(
                        "a block decode's step past the device's decode scratch",
                    ));
                }
                let firsts = &device.shared().textures.first_moves;
                (None, firsts.decode_scratch(device, most + align)?, 0, 0)
            }
        };
        // Each step's shape, written by the host at its slot of the shapes' buffer: a shape's
        // bytes rounded up to the alignment.
        let shape_bytes = u64::from(BlockDecodeModule::SHAPE_WORDS) * 4;
        let slot = shape_bytes.div_ceil(align) * align;
        let shapes = device.write_combined_buffer(steps.len() as u64 * slot)?;
        let (raw_shapes, shapes_base) = shapes.at(0);
        if !base.is_multiple_of(align) || !shapes_base.is_multiple_of(align) {
            return Err(internal("a block decode's scratch off its alignment"));
        }
        let contents = (shapes.contents())
            .ok_or_else(|| internal("a Shared block decode shapes' buffer without contents"))?;
        for (at, plan) in plans.iter().enumerate() {
            // SAFETY: the shapes' buffer holds `steps.len()` slots of `slot` bytes, each at least
            // a shape's, in host-coherent memory nothing else uses yet.
            unsafe {
                let to = contents.as_ptr().add(at * slot as usize);
                std::ptr::copy_nonoverlapping(plan.shape.as_ptr(), to, plan.shape.len());
            }
        }
        let (kernel, sets) = match &decoded.pairs {
            None => {
                let kernel = device.owned_kernel(OwnedModule::TwinDecode(decode))?;
                let ranges: Vec<[(vk::Buffer, u64, u64); 3]> = (layouts.iter())
                    .zip(0u64..)
                    .map(|(&(blocks, texels_at, texels), at)| {
                        [
                            (raw_shapes, shapes_base + at * slot, shape_bytes),
                            (raw_scratch, base, blocks),
                            (raw_scratch, base + texels_at, texels),
                        ]
                    })
                    .collect();
                (kernel, OwnedSets::write(device, &kernel, &ranges)?)
            }
            // The pairs' image, then the shape and the texels (`bind::ChromaSampleModule`).
            Some(pairs) => {
                let owned: Vec<OwnedSet> = (layouts.iter())
                    .zip(0u64..)
                    .map(|(&(_, texels_at, texels), at)| OwnedSet {
                        images: vec![pairs.view],
                        storage_images: Vec::new(),
                        buffers: vec![
                            (raw_shapes, shapes_base + at * slot, shape_bytes),
                            (raw_scratch, base + texels_at, texels),
                        ],
                    })
                    .collect();
                let kernels = vec![pairs.kernel; owned.len()];
                device.shared().counts.chroma_device_decode();
                (
                    pairs.kernel,
                    OwnedSets::write_each(device, &kernels, &owned)?,
                )
            }
        };
        let steps = steps
            .iter()
            .zip(&plans)
            .zip(&layouts)
            .zip(&sets.sets)
            .map(|(((region, plan), &(_, texels_at, _)), &set)| Step {
                blocks: vk::BufferImageCopy {
                    buffer_offset: base,
                    ..buffer_image_copy(&plan.blocks)
                },
                // The band's pairs, tight rows of the texture's width out of the scratch, into
                // the same rows of the pairs' image.
                pairs: decoded.pairs.as_ref().map(|_| vk::BufferImageCopy {
                    buffer_offset: base,
                    ..buffer_image_copy(&plan.blocks)
                }),
                texels: vk::BufferImageCopy {
                    buffer_offset: base + texels_at,
                    ..buffer_image_copy(region)
                },
                set,
                groups: plan.groups,
            })
            .collect();
        device.shared().counts.block_decode_pass();
        Ok(Some(DecodePass {
            image: self.raw.get(),
            twin: decoded.image.raw.get(),
            pairs: decoded
                .pairs
                .as_ref()
                .map_or(vk::Image::null(), |pairs| pairs.image),
            scratch: raw_scratch,
            pipeline: kernel.pipeline,
            layout: kernel.layout,
            steps,
            bytes,
            held: Held {
                _scratch: scratch,
                _shapes: shapes,
                _sets: sets,
            },
        }))
    }

    /// The image's decoded twin that decodes a texture of `pf` (its first, or an ASTC carrier's
    /// second): `None` where neither does.
    fn twin_of(&self, pf: PixelFormat) -> Option<&Decoded> {
        (self.decoded.as_deref().into_iter())
            .chain(self.second.get().map(Box::as_ref))
            .find(|decoded| decoded.decode.decodes(pf))
    }

    /// Make the image's second decoded twin, an ASTC carrier's, for a view of the texture as `pf`
    /// (its other colour space; the module docs), where it has none:
    /// moved by the device's next submission, its texels the carrier's whole decode, which the
    /// device's next batch with work records before that work ([`Device::take_second_decodes`]).
    /// Refused (`Feature::TextureView`, not implemented) where the device takes no twin of the
    /// decode's texels.
    pub(super) fn make_second_twin(self: &Arc<Self>, pf: PixelFormat) -> Result<(), Error> {
        if self.second.get().is_some() {
            return Ok(());
        }
        // A carrier without a twin (the device takes none of its texels) reads none either way.
        let (Some(first), Some(decode)) = (self.decoded.as_deref(), AstcDecode::of(pf)) else {
            return Ok(());
        };
        let device = &self.device;
        let (layout, descriptor) = first.made_for;
        let decode = TwinDecode::Astc(decode, device.astc_texels(decode)?);
        let plan = twin_plan(self.plan(), &layout, decode, false);
        let facts = (device.shared().textures).facts(device.compile_device().profile());
        if !facts.optimal(plan.format).contains(TWIN_FEATURES) {
            return Err(Error::Unsupported(Unsupported::not_implemented(
                Feature::TextureView,
            )));
        }
        // The kernel that keeps it in step, made now rather than by a submission.
        device.owned_kernel(OwnedModule::TwinDecode(decode))?;
        // Its uses are the image's, as the first twin's; it takes no CPU copy, so only the GPU
        // writes it, in submission order.
        let mut own = descriptor;
        own.resource_options = ResourceOptions::private();
        let identity = Identity::new(HazardTrackingMode::Default, false);
        let start = Start::Moved;
        let twin = device.new_image_starting(
            plan,
            &layout,
            &own,
            identity,
            start,
            Device::runtime_owner,
        )?;
        let mut viewed = layout;
        viewed.pixel_format = pf;
        let second = Decoded {
            image: twin,
            decode,
            started_on_the_cpu: false,
            made_for: (layout, own),
            pairs: None,
            read: TwinRead::of(&viewed, decode, false),
        };
        device.degrade_carried_twin(&viewed, &second, None);
        // Made for another view meanwhile: that one stands, and this one goes unused.
        if self.second.set(Box::new(second)).is_err() {
            return Ok(());
        }
        device.shared().counts.second_twin();
        self.register_first_move();
        self.queue_second_decode(&whole_regions(self.plan()))
    }

    /// Queue the decode of `regions` (of the image's levels and layers, in its texels: a
    /// carrier's) into its second decoded twin, where it has one, for the device's next batch with
    /// work ([`Device::take_second_decodes`]): its pass made now, by the calling thread. After a
    /// host call wrote the regions, or as the twin is made.
    pub(super) fn queue_second_decode(
        self: &Arc<Self>,
        regions: &[ImageRegion],
    ) -> Result<(), Error> {
        let Some(SecondDecode { pass, uses, .. }) = self.second_decode(regions)? else {
            return Ok(());
        };
        // Held weakly: an image gone takes its queued decodes with it.
        let queued = QueuedDecode {
            image: Arc::downgrade(self),
            key: self.identity.key,
            pass,
            uses,
        };
        lock(&self.device.shared().textures.second_decodes).push(queued);
        Ok(())
    }

    /// The decode of `regions` (as for [`queue_second_decode`](Self::queue_second_decode)) into
    /// the image's second decoded twin, for a batch to record: `None` for an image without one,
    /// or regions of no texel.
    fn second_decode(
        self: &Arc<Self>,
        regions: &[ImageRegion],
    ) -> Result<Option<SecondDecode>, Error> {
        let Some(second) = self.second.get() else {
            return Ok(None);
        };
        let Some(pass) = self.decode_pass_into(second, regions, PassScratch::Own)? else {
            return Ok(None);
        };
        Ok(Some(SecondDecode {
            pass,
            uses: self.decode_uses(regions),
            _image: Arc::clone(self),
        }))
    }

    /// The uses of a command that decodes `regions` of the image into its twin: a write of each
    /// (the twin shares the image's resource identity), as a copy reads and writes.
    fn decode_uses(&self, regions: &[ImageRegion]) -> Vec<Use<ResourceKey>> {
        let access = Access::new(
            TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        let volume = self.plan().image_type == ImageType::TYPE_3D;
        regions
            .iter()
            .map(|region| {
                let layers = match volume {
                    true => 0..1,
                    false => region.layer..region.layer + region.layers,
                };
                let range = UseRange::Image {
                    aspects: texture_aspects(self.plan()),
                    levels: region.level..region.level + 1,
                    layers,
                };
                Use::writes(self.identity.key, self.identity.tracked, range, access)
            })
            .collect()
    }
}

/// The error message of a decoded twin's failed CPU decode (`metal2vulkan-twin-decode-failed`),
/// of a `pixel_format` texture of `extent`: the twin kept its earlier texels.
pub(crate) fn twin_decode_failed(
    pixel_format: PixelFormat,
    extent: [u32; 3],
    error: &Error,
) -> DebugMessage {
    DebugMessage {
        severity: MessageSeverity::Error,
        kind: MessageKind {
            general: true,
            ..MessageKind::default()
        },
        id_name: "metal2vulkan-twin-decode-failed".to_owned(),
        id: 0,
        text: format!(
            "metal2vulkan: the decoded twin of a {pixel_format:?} texture of {extent:?} kept its \
             earlier texels: its CPU decode failed ({error})"
        ),
        degradation: None,
    }
}

/// `region` (of the twin's texels, in `decode`'s blocks) in steps whose blocks and texels take at
/// most `most` bytes together (before, its texels alone): whole, else runs of its images (layers
/// or depth slices), else each image in bands of whole block rows (of a whole-level decode's
/// region, bands of rows beside all the image's blocks, [`TwinDecode::whole_levels`]). A block
/// row past `most` (16384 texels of 16 bytes and their blocks are 1 MiB and 64 KiB) is a step
/// alone.
pub(crate) fn bands(region: &ImageRegion, decode: TwinDecode, most: u64) -> Vec<ImageRegion> {
    let (texel, block) = (
        u64::from(decode.texel_bytes()),
        u64::from(decode.block_bytes()),
    );
    let [bw, bh] = decode.block_extent();
    let [w, h, d] = region.size;
    let count = u64::from(d) * u64::from(region.layers);
    let across = u64::from(w).div_ceil(bw.into());
    // A row of blocks and its rows of texels.
    let row = across * block + u64::from(bh) * u64::from(w) * texel;
    let image = across * u64::from(h.div_ceil(bh)) * block + u64::from(w) * u64::from(h) * texel;
    if count == 0 || w == 0 || h == 0 {
        return Vec::new();
    }
    if image * count <= most {
        return vec![*region];
    }
    // The `i`-th image alone, of `n` from it.
    let images = |i: u32, n: u32| {
        let mut one = *region;
        match d > 1 {
            true => {
                one.origin[2] += i;
                one.size[2] = n;
            }
            false => {
                one.layer += i;
                one.layers = n;
            }
        }
        one
    };
    let total = d.max(region.layers);
    if image <= most {
        let per = u32::try_from((most / image).max(1)).unwrap_or(u32::MAX);
        return (0..total)
            .step_by(per as usize)
            .map(|i| images(i, per.min(total - i)))
            .collect();
    }
    // Rows a band: whole block rows, but for a whole-level decode, each of whose bands reads all
    // its image's blocks.
    let rows = match decode.whole_levels() {
        true => {
            let blocks = image - u64::from(w) * u64::from(h) * texel;
            most.saturating_sub(blocks) / (u64::from(w) * texel)
        }
        false => (most / row).max(1) * u64::from(bh),
    };
    let rows = u32::try_from(rows.max(1)).unwrap_or(u32::MAX);
    (0..total)
        .flat_map(|i| {
            let one = images(i, 1);
            (0..h).step_by(rows as usize).map(move |y| {
                let mut band = one;
                band.origin[1] += y;
                band.size[1] = rows.min(h - y);
                band
            })
        })
        .collect()
}

/// The plan of one band of a decode pass ([`Image::band_plan`]): the image's region its blocks
/// are copied out of and their bytes, its shape and its workgroups.
struct BandPlan {
    blocks: ImageRegion,
    block_bytes: u64,
    shape: [u8; BlockDecodeModule::SHAPE_WORDS as usize * 4],
    groups: [u32; 3],
}

/// One band of a decode pass: its blocks copied out of the image, a 4:2:2 twin's of the device's
/// conversion copied on into the pairs' image, its set, its workgroups and its texels copied into
/// the twin.
struct Step {
    blocks: vk::BufferImageCopy,
    pairs: Option<vk::BufferImageCopy>,
    texels: vk::BufferImageCopy,
    set: vk::DescriptorSet,
    groups: [u32; 3],
}

/// What a decode pass holds until the command that records it completed: its own scratch
/// (`None` in the device's, which the device keeps), its shapes and its sets.
struct Held {
    _scratch: Option<Buffer>,
    _shapes: Buffer,
    _sets: OwnedSets,
}

/// A decode into an ASTC carrier's second decoded twin (the module docs) that a batch records
/// (`command_buffer/submit.rs`): its pass and its uses, and the image, held until the batch
/// completed.
pub(crate) struct SecondDecode {
    pass: DecodePass,
    uses: Vec<Use<ResourceKey>>,
    _image: Arc<Image>,
}

impl SecondDecode {
    /// The uses of its command (the image's, `Image::decode_uses`).
    pub(crate) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record it ([`DecodePass::record`]).
    ///
    /// # Safety
    ///
    /// `raw` is recording, outside a render pass, on a queue family with compute; the decode lives
    /// until it completes.
    pub(crate) unsafe fn record(&self, device: &Vk, raw: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the decode holds the image and its twin, both in the
        // `GENERAL` layout (the twin moved by a submission no later than the one that records
        // this, `Image::make_second_twin`).
        unsafe { self.pass.record(device, raw) }
    }
}

/// A blit encoder's write into an ASTC carrier's blocks (`BlitCommandEncoder::decode_twin`),
/// which its batch's submission decodes into the image's second decoded twin: by the decode made
/// as the command was encoded, where the image had a second twin then. Without one, the decodes
/// queued for the image wait past the batch ([`Device::take_second_decodes`]): a second twin made
/// since reads the carrier as the batch leaves it.
pub(crate) struct TwinWrite {
    key: ResourceKey,
    decode: Option<SecondDecode>,
}

impl TwinWrite {
    /// The image's resource key, where the write made no decode.
    pub(crate) fn without_decode(&self) -> Option<ResourceKey> {
        self.decode.is_none().then_some(self.key)
    }

    /// Its decode, where it made one.
    pub(crate) fn into_decode(self) -> Option<SecondDecode> {
        self.decode
    }
}

/// A decode queued for an image's second decoded twin ([`Image::queue_second_decode`]): the
/// image, held weakly, its key, and the decode's pass and uses.
pub(crate) struct QueuedDecode {
    image: Weak<Image>,
    key: ResourceKey,
    pass: DecodePass,
    uses: Vec<Use<ResourceKey>>,
}

/// The decodes queued for second decoded twins ([`Image::queue_second_decode`]), which the
/// device's next batch with work records before that work ([`Device::take_second_decodes`]).
pub(crate) type SecondDecodes = Mutex<Vec<QueuedDecode>>;

impl Device {
    /// The decodes queued for second decoded twins (the module docs), taken for the batch now
    /// submitted to record before its work, in the order they were queued, but those of an image
    /// for which `held` answers true (a write of the batch's made no decode), which stay queued
    /// for a later batch, and those of an image gone, dropped.
    pub(crate) fn take_second_decodes(
        &self,
        held: impl Fn(ResourceKey) -> bool,
    ) -> Vec<SecondDecode> {
        let mut queued = lock(&self.shared().textures.second_decodes);
        if queued.is_empty() {
            return Vec::new();
        }
        let (kept, taken): (Vec<QueuedDecode>, Vec<QueuedDecode>) =
            queued.drain(..).partition(|decode| held(decode.key));
        queued.extend(kept);
        drop(queued);
        taken
            .into_iter()
            .filter_map(
                |QueuedDecode {
                     image, pass, uses, ..
                 }| {
                    let image = image.upgrade()?;
                    Some(SecondDecode {
                        pass,
                        uses,
                        _image: image,
                    })
                },
            )
            .collect()
    }
}

/// The scratch buffer a decode pass's bands go through ([`Image::decode_pass`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassScratch {
    /// One of its own, held until the command that records it completed: a pass a command
    /// buffer records, or an internal submission's own.
    Own,
    /// The device's (`FirstMoves::decode_scratch`), made once: a pass a submission takes from the
    /// device's staged work, which runs after every such pass before it. It is no staged byte
    /// (the bound), so a texture's start (`Image::start_decoded_twin`) allocates nothing.
    Device,
}

/// The decode of written regions of a carrier into its decoded twin (the module docs), made by
/// [`Image::decode_pass`] and recorded by [`DecodePass::record`] after the write.
pub(crate) struct DecodePass {
    image: vk::Image,
    twin: vk::Image,
    /// A 4:2:2 twin's pairs' image, where it holds the device's conversion; null elsewhere.
    pairs: vk::Image,
    scratch: vk::Buffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    steps: Vec<Step>,
    /// The bytes of its own scratch (none in the device's).
    pub(super) bytes: u64,
    held: Held,
}

impl DecodePass {
    /// Record the pass: after every write before it, each band's blocks copied into the
    /// scratch, decoded, and its texels copied into the twin, each step past a barrier, then a
    /// barrier to everything after. Only entry points of the exit guard's table (a staged
    /// copy's runs inside a submission's section).
    ///
    /// # Safety
    ///
    /// `raw` is recording, outside a render pass, on a queue family with compute; the image, the
    /// twin and the pass live until it completes; both images are in the `GENERAL` layout.
    pub(crate) unsafe fn record(&self, device: &Vk, raw: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        let compute = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's contract; each copy lies in its image and in the scratch, and
        // each dispatch reads and writes within its set's ranges (`decode_pass`).
        unsafe {
            Barrier::everything().record(device, raw);
            device.cmd_bind_pipeline(raw, compute, self.pipeline);
            for step in &self.steps {
                let blocks = std::slice::from_ref(&step.blocks);
                device.cmd_copy_image_to_buffer(raw, self.image, general, self.scratch, blocks);
                Barrier::everything().record(device, raw);
                if let Some(pairs) = &step.pairs {
                    // The pairs' image's earlier texels dropped (a step samples only the rows
                    // it copies), the band's pairs copied in, then sampled.
                    let whole = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let into = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(general)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.pairs)
                        .subresource_range(whole);
                    device.cmd_pipeline_barrier(
                        raw,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[into],
                    );
                    let pairs = std::slice::from_ref(pairs);
                    device.cmd_copy_buffer_to_image(raw, self.scratch, self.pairs, general, pairs);
                    Barrier::everything().record(device, raw);
                }
                let set = std::slice::from_ref(&step.set);
                device.cmd_bind_descriptor_sets(raw, compute, self.layout, 0, set, &[]);
                let [x, y, z] = step.groups;
                device.cmd_dispatch(raw, x, y, z);
                Barrier::everything().record(device, raw);
                let texels = std::slice::from_ref(&step.texels);
                device.cmd_copy_buffer_to_image(raw, self.scratch, self.twin, general, texels);
                Barrier::everything().record(device, raw);
            }
        }
        for call in [
            Call::CmdPipelineBarrier,
            Call::CmdBindPipeline,
            Call::CmdCopyImageToBuffer,
            Call::CmdBindDescriptorSets,
            Call::CmdDispatch,
            Call::CmdCopyBufferToImage,
        ] {
            exit_guard::called(call);
        }
    }
}

impl BlitCommandEncoder {
    /// After the encoder's last command wrote `regions` (of the texture's levels and layers, in
    /// its texels) of `texture`'s texels: their decode into its decoded twin (the module docs),
    /// one command of the encoder. Nothing for a texture without one.
    pub(crate) fn decode_twin(
        &mut self,
        texture: &Texture,
        regions: &[ImageRegion],
    ) -> Result<(), Error> {
        let Some(image) = texture.decoding_image() else {
            return Ok(());
        };
        // A carrier's regions: the carrier's texels over the texture's.
        let regions: Vec<ImageRegion> = regions
            .iter()
            .map(|region| pvrtc::image_region(image.plan(), *region))
            .collect();
        let Some(pass) = image.decode_pass(&regions, PassScratch::Own)? else {
            return Ok(());
        };
        let uses = image.decode_uses(&regions);
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording, outside a render pass; the
            // image (its twin with it) and the pass are retained until it completes; both images
            // rest in the `GENERAL` layout.
            unsafe { pass.record(device, raw) }
        })?;
        self.core().retain(pass.held);
        // An ASTC carrier's write, for its batch's submission to decode into the image's second
        // twin, by the decode made now where it has one (the module docs).
        if matches!(image.decoded.as_deref(), Some(d) if matches!(d.decode, TwinDecode::Astc(..))) {
            let decode = image.second_decode(&regions)?;
            let key = image.identity.key;
            self.core().wrote_twin(TwinWrite { key, decode });
        }
        self.core().retain(image);
        Ok(())
    }
}
