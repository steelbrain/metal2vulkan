//! Metal's private texture fills: `fillTexture:level:slice:region:bytes:length:`
//! ([`BlitCommandEncoder::fill_texture`]) and `fillTexture:level:slice:region:color:` with and
//! without `pixelFormat:` ([`BlitCommandEncoder::fill_texture_color`]), Apple SPI the runtime
//! takes as plain Metal answers it.
//!
//! **Metal's answer** (an Apple M4 under macOS 27.0.1, our own probes; every byte of every level,
//! slice and aspect read back after one fill over bytes of `0xEE`, 112 formats, one case a
//! process; and an Apple M3 Max under macOS 27.0.1, our probes `surface.m`, `tp.m`, `ds.m` and
//! `misc.m`, every byte of the surface, buffer or texture read back). A fill writes the pattern,
//! cut at a unit's bytes or repeated to them, into every unit of the region, and leaves every
//! other byte:
//! - a **compressed** format's unit is a block: the region's origin and size count blocks, cut at
//!   the level's blocks (74 formats, PVRTC at square power-of-two extents);
//! - every other format's unit is a texel, and the region counts texels, cut at the level: an
//!   ordinary color or depth format's texel (31 formats); a 4:2:2 format's 2 bytes (`GBGR422`,
//!   `BGRG422`, at bytes `2x` and `2x + 1` of a row, which the 4:2:2 copies move); a stencil
//!   format's 1 byte (`Stencil8`, `X32Stencil8`, an `X32_Stencil8` view's too); `Depth32Float_Stencil8`'s 4 depth bytes, its
//!   stencil kept;
//! - a **two-plane YCbCr** texture over a surface's plane (every one of the 24 formats the runtime
//!   makes, over plane 0 of `420v` to `x444` and plane 1 of the 4:2:2 and 4:4:4 surfaces) has
//!   the plane's code as its unit, 1 byte or 2 (a 10-bit code's word), at the plane's rows, the
//!   other plane kept, as its blits move it; a **packed** one's (`YCBCR10_420_2P_PACKED` to
//!   `YCBCR10_444_2P_PACKED_sRGB`, over `p420` to `p444`) every fill writes one zero byte at each
//!   texel's `x` of its row of the plane, whatever the pattern or color;
//! - a **padded 10-bit** texture (`R10Unorm_X6`, `RG10Unorm_X12` and their `_sRGB` twins, each
//!   code in the top bits of its own 16-bit word) has its 2 or 4 bytes as its unit, at its rows;
//! - a **two-plane extended-range** texel (`RGB10A8_2P_XR10` and its sRGB twin) is its color word
//!   in plane 0, 4 bytes, its alpha kept. Metal writes row `y` of the region `⌊4 w / 16⌋ · 16`
//!   bytes apart (16 below a width of 4), its texel `x` at byte `4 x` of it, `w` level 0's width,
//!   at any level, the region cut at the level's extent: where a heap's bytes hold the texture's
//!   first level (a 1D, 2D, 1D or 2D array or 3D texture of any level count,
//!   `texture/heap_two_planes.rs`), the fill writes those bytes there, rows overlapping included,
//!   an array's slice `align(h · align(4 w, 16), 128)` bytes past the texture's first, a 3D
//!   texture's depth images as its first, but a mipmapped one's past the first not at all (an
//!   Apple M3 Max under macOS 27.0.1: every level of 2D textures 5 to 255 texels wide of 3 to 6
//!   levels, whole and inset regions; slices of 1D and 2D arrays; every depth image of 3D
//!   textures of 1 to 4 levels; a one-level 3D texture's fill past its first depth image wrote
//!   its first image's bytes, or in some runs none). A texture of its own memory, and a heap's
//!   the runtime holds apart from the heap's bytes, takes the same bytes of level 0 of the slice
//!   at any level, the region cut at the level's extent (3D: image 0): each
//!   word lands on the texel of level 0's rows (`align(4 w, 16)` bytes apart) it meets, or between
//!   them, which Metal's blits with a buffer read back and the runtime holds apart for a texture
//!   of one level (`two_planes/buffer_rows.rs`; of more, not) (`getBytes` of every texel after fills of 1 to 37
//!   texels wide, 1 to 3 levels, arrays of 2, 1D and 3D, of its own memory and a heap's; a
//!   texture's levels and slices past the first lie in bytes that overlap, which Metal's own
//!   copies show and the runtime holds apart). A texture over a surface's planes is written at
//!   its GPU rows (plane 0's `bytesPerRow` rounded down to 16, the rows Metal's blits address it
//!   at) at any width, its alpha plane and every byte between the rows kept (`RGB10A8_2P_XR10`
//!   over 7 surfaces and the sRGB twin over one, 16, 33, 37, 61, 64 and the consumer's 384 texels
//!   wide, a region, a whole texture with a 3-byte pattern and a region past the level, every
//!   byte of each surface read back).
//!
//! A **color fill** packs the color in `pixelFormat:` (the texture's own format without it) by
//! the fill's own rules (`fill_value.rs`), which are not a `Clear` load action's, and fills that
//! pattern. The runtime packs it so, on the host, for every fill with `pixelFormat:` (a capability:
//! Metal's packing in another format). A **plain** fill of a texture of a format whose fill
//! converts a `float` (normalized, sRGB, half, float, small float, shared exponent;
//! `fill_value::device_converts`) and that the device renders, its image holding Metal's bytes,
//! is the device's own conversion of the color instead, as the native-arithmetic rule asks: the
//! device's clear of a color value (`vkCmdClearColorImage`). A fill of whole subresources (every
//! texel of a level's layer, or of a 3D level's depth images) whose image is the texture's own (not
//! a buffer's bytes), made with transfer-destination usage in a format that takes transfer writes
//! in its tiling, is the clear of that image there, and makes nothing. Any other region copies a
//! **tile** the device cleared to the color, of the texture's image format, samples and dimension,
//! into the region of each image the fill writes, side by side from the region's origin, as a
//! blit copy between the textures moves it. The encoder keeps its last 4 tiles for its later
//! fills (each side the power of two at least the region's and 16 texels, no larger than the
//! texture's first level, the longer one halved to hold 1 MiB at most): a fill of a color a kept
//! tile holds copies it; one of another color makes a tile while fewer are kept and clears the
//! least recently used one again past that. So a burst of fills of a few colors makes one tile a
//! color, which the device's recycler gives where an earlier command buffer's went. The clear is
//! announced where the device's conversion may store another texel than Metal's packing
//! (`fill_value::device_apart`, `device-color-conversion`; a half or small-float NaN,
//! `device-nan-encoding`). A texture buffer and every other format are packed on the host. Metal
//! fills the packed texel's bytes, repeated
//! or cut at the texture's texel (12 pairs of formats, 256 colors each: 128 colors packed in each
//! of 70 formats into textures of 18 formats of 1 to 16 bytes; on the M3 Max the 88 formats Metal
//! packs a color in, 1,144 colors each, into `RGBA32Uint` texels). In every other format (213: compressed, 4:2:2, depth, stencil,
//! two-plane, YCbCr, padded, `Invalid` and the private formats) Metal packs nothing and fills
//! bytes of the calling thread's stack, the texel's bytes of a 16-byte buffer the packing writes:
//! a probe that set the stack below its call to one byte read that byte back in every texel, and
//! two consecutive color fills in one encoder find the first's packed texel there. Whether calls
//! between them keep it depends on the calling code's stack: in our probe `slot.m` an encoder's
//! end, a blit or a byte fill between them left other bytes (pointers of the process, differing
//! between processes), while the consumer's process found the first's packed texel across the
//! encoders of one command buffer. The runtime holds that buffer per command buffer, zeros at its
//! start, each color fill's packed texel over its first bytes, and fills its bytes there, as
//! Metal's consecutive fills and the consumer's encoders do; announced once
//! (`Degradation::UnpackedFillColor`). Metal's
//! validation stops such a fill ("must be a color format", "is not compatible with the texture
//! format").
//!
//! **Placement** (31 cases of a byte fill of `RGBA8Unorm` textures of each type,
//! every byte of every slice and level read back): a level past the texture's writes nothing; a 1D,
//! 2D or 3D texture ignores the slice, a cube takes the slice's face modulo 6, and an array (1D, 2D
//! or cube) writes nothing at a slice past its own; a 3D texture writes the region's depth images,
//! cut at the level's depth, and every other type one image, ignoring the region's `z` and depth. A
//! 2D multisample texture ignores the slice, as a 2D one does, and a multisample array writes
//! nothing at a slice past its own.
//!
//! **Texture buffers** (the M3 Max, our probe `tb.m`: 1,493 cases of 5 formats of 1 to 16 bytes
//! a texel, patterns of 1 to 256 bytes and packed colors, over a 12-texel texture at offsets 0
//! to 68 of a 512-byte buffer, every byte read back, the consumer's cases among them). A fill
//! writes bytes of the buffer, not texels: the `w · b` bytes from the texture's offset plus
//! `x · b` (`b` the texel's bytes), past the texture's width too, cut at the buffer's end;
//! the level, slice, row, height and depth change nothing (a height or depth of 0 writes the row),
//! and a width of 0 writes nothing. The pattern of `n` bytes runs across the region's bytes: byte
//! `j` of the region is `pattern[j & (n - 1)]` (the pattern repeated where `n` is a power of two;
//! `n` 3 writes `p0 p0 p2 p2`), except in each 16-byte chunk of the buffer the region covers
//! whole, which takes 16 bytes of a block from its byte `(j & (n - 1)) & !3`, `j` the chunk's
//! first. The block is the power of two of bytes at least `n` and 16, holding the pattern's whole
//! copies from its start and, past them and past the block, bytes of the calling thread's stack:
//! zeros in every probe of ours, a 9-byte fill after an 8-byte one in the same encoder too, while
//! the consumer's process read an earlier fill's bytes there. The runtime writes zeros there,
//! announced once where a chunk reads them (`Degradation::TextureBufferFillLeftover`). The rule
//! holds byte for byte for long patterns too (our probe `tblf.m`: 18 fills of patterns of 16 to
//! 32752 bytes into `R8Uint`, `R32Uint` and `RGBA32Uint` texture buffers at offsets 4 to 64, up
//! to 100,000 texels wide). A pattern of more than 32752 bytes ends Metal's process in the call,
//! in the blit's dispatch, whatever the format (`R8Uint`, `R32Uint`, `RGBA32Uint`) and region
//! (of 0 texels, past the buffer, up to 100,000 texels; our probes `tblf.m` and `tbz.m`); the
//! runtime refuses it (`Problem::LongTextureBufferFillPattern`). A fill of any other texture
//! takes a pattern of any length (up to 1 MiB in our probe `tl.m`) and reads 16 bytes of it at
//! most.
//!
//! **Multisample textures** (the M3 Max, our probe `ms.m`: 32 formats at 4 samples and 11 at 2,
//! every sample of a 5 x 3 texture or 2-slice array read back through an unsigned view, as its
//! depth's bits or as its stencil): a fill writes every sample of the region's texels and no
//! other, each sample `s` its own `b` bytes (`b` the texel's, or its aspect's: the depth of
//! `Depth32Float_Stencil8`, its stencil kept). Where the `n · b` bytes of a texel's `n` samples
//! are 16 or fewer, the pattern is repeated from its start and cut at `n · b`, and sample `s`
//! takes its bytes from `s · b` (`RGBA8Unorm` at 4 samples a 16-byte pattern's 4 bytes each,
//! `R8Unorm` its first 4 bytes one each); past 16, every sample takes the pattern cut at `b`
//! (`RGBA16Float` at 4 samples, `RGBA32Float` at 2 and 4). A color fill's pattern is the packed
//! texel, which every sample then takes; packed in no format, the 16-byte buffer, which the
//! samples take a part each (`Depth32Float` at 2 and 4). Every value lands as given, a depth's
//! NaN, 2.0, -1.0 and denormal too.
//!
//! An empty pattern Metal's API validation layer stops; plain Metal takes the fill, the layer off,
//! and writes the region all the same, from bytes the call does not give: an ordinary texture's
//! texels a unit read from the pattern's address, a texture buffer's region the fill's leftover
//! bytes, Metal's own (the M3 Max, our probe `emptyfill.m`). The runtime holds neither, so it
//! refuses the fill its own gap (`NotImplemented(RuntimeGap::EmptyFillPattern)`,
//! [`check_pattern`]).
//!
//! **`fillBuffer:range:pattern4:`** ([`BlitCommandEncoder::fill_buffer_pattern4`]; 10 ranges of a
//! 256-byte buffer) writes the 4-byte pattern from the range's start, restarting it at each
//! multiple of 16 bytes of the buffer: byte `k` of the range takes the pattern's byte `k - start`
//! modulo 4 before the first such multiple, and byte `k` modulo 4 from it on.
//!
//! **How.** A plain color fill the device clears is the clear above, of the texture's image or of a
//! tile copied from. Every other fill is copies the encoder already makes: a staging buffer of the
//! pattern's units
//! (4 MiB at most, copied again for a larger region) and `copy_from_buffer_to_texture` into the
//! region's units of each image, so shadows, twins, carriers and two planes take their own paths;
//! a heap's two-plane texture, one copy of the color words into the heap's bytes; a two-plane
//! texture held apart, the plane words' copy (`two_planes.rs`) into the rectangles of texels the
//! words land on; a texture buffer (its bytes before the first whole chunk, its chunks' period
//! repeated, its bytes after the last: units of at most the block's 32 KiB, read from the pattern
//! as they are made, whatever the region's length) and a packed YCbCr texture, copies into the
//! buffer's bytes. A multisample texture's samples: the sample planes' merge writes each sample's
//! bits into a multisampled image of the unsigned-integer format of the texel's size
//! (`sample_planes::BitsCopy::fill`), and a copy between multisampled textures moves the region
//! into the texture (`blit_texture.rs`): a color image's bits by an image copy, or by the sample
//! planes' copies where the device's own converts float NaNs; a depth or stencil aspect by
//! `sample_copy.rs`'s copies or draws, exact or announced as those are (`drawn-depth-copy` into a
//! 32-bit float depth whose driver's fragment depth changes values). A pattern4 fill copies its
//! bytes from a staging buffer.

use super::blit::{bytes_in, reads, writes, BlitCommandEncoder, TRANSFER};
use super::blit_plan::BufferLayout;
use super::blit_plan::ImageCopyRegion;
use super::blit_texture;
use super::chroma_pairs::{is_four_two_two, TEXEL_BYTES};
use super::clear_value;
use super::clear_value::has_channels;
use super::compressed_blit::degrade_compressed_blit;
use super::copy_plan::{
    block_bytes, block_extent, level_extent, BlitOption, CopyUnit, ImageRegion,
};
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::fill_value::{device_apart, device_converts, device_nan, fill_texel};
use super::heap::Heap;
use super::image_plan::{sample_count, ImagePlan};
use super::render_values::ClearColor;
use super::replay::BlitCall;
use super::sample_planes::{BitsCopy, BITS_COPY_STAGES};
use super::shadow::Covered;
use super::sparse::emulated::Written;
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::Access;
use super::two_planes;
use super::values::ResourceOptions;
use super::values::{Origin, Region, Size};
use super::Device;
use super::{Buffer, Degradation};
use ash::vk;
use metal2vulkan::bind::{PlanesCopy, YCbCrCodes};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{
    byte_format, padded_words, ycbcr_format, ClearRoute, ExtendedRange, FormatMapping,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};
use std::ops::Range;

/// The most bytes a fill stages at once: a larger region copies them again.
const STAGING_BYTES: u64 = 4 << 20;

/// What a fill writes into one unit of a format (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FillUnit {
    /// A compressed format's block of `bytes`, `extent` texels: the region counts blocks.
    Blocks { bytes: u32, extent: [u32; 2] },
    /// A texel's `bytes`, which a copy with `option` writes: the region counts texels.
    Texels { bytes: u32, option: BlitOption },
    /// A two-plane extended-range texel's color word, its alpha kept.
    PlaneWords,
    /// A packed two-plane YCbCr texel (`YCBCR10_420_2P_PACKED`, ...): one zero byte at the
    /// texel's `x` of its row of the plane, whatever the pattern.
    PackedCodes,
}

/// The fill unit of `pf`, `None` for a format the runtime makes no texture of (`Invalid`, the
/// private formats it holds none of, the 24-bit depth formats, which Apple GPUs make none of).
pub(super) fn fill_unit(pf: PixelFormat) -> Option<FillUnit> {
    use PixelFormat as P;
    let texels = |bytes, option| Some(FillUnit::Texels { bytes, option });
    if two_planes::holds_planes(pf) {
        return Some(FillUnit::PlaneWords);
    }
    if let Some(extent) = block_extent(pf) {
        return Some(FillUnit::Blocks {
            bytes: block_bytes(pf)?,
            extent,
        });
    }
    if is_four_two_two(pf) {
        return texels(TEXEL_BYTES, BlitOption::None);
    }
    // A two-plane YCbCr texel is its plane's code, 1 or 2 bytes, as Metal's blits move it.
    if let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) {
        if codes.packed {
            return Some(FillUnit::PackedCodes);
        }
        return texels(codes.bytes, BlitOption::None);
    }
    // A padded 10-bit texel is its 2 or 4 bytes, the words holding its codes.
    if padded_words(pf) {
        return texels(block_bytes(pf)?, BlitOption::None);
    }
    match pf {
        P::Stencil8 => texels(1, BlitOption::None),
        P::X32Stencil8 => texels(1, BlitOption::StencilFromDepthStencil),
        P::Depth32FloatStencil8 => texels(4, BlitOption::DepthFromDepthStencil),
        P::Depth16Unorm => texels(2, BlitOption::None),
        P::Depth32Float => texels(4, BlitOption::None),
        _ if has_channels(pf).is_some() => texels(block_bytes(pf)?, BlitOption::None),
        // Every other extended-range texel: a bfloat or 10-bit one's words (color fills of
        // single texels and of 4 x 1), `RGB10A2Unorm_sRGB`'s 4 bytes (byte fills, the M3 Max).
        _ if ExtendedRange::of(pf).is_some() => texels(block_bytes(pf)?, BlitOption::None),
        _ => None,
    }
}

/// `pattern` repeated from its start and cut at `bytes`.
pub(super) fn unit_of(pattern: &[u8], bytes: u32) -> Vec<u8> {
    pattern
        .iter()
        .copied()
        .cycle()
        .take(bytes as usize)
        .collect()
}

/// The bytes of `pattern` a fill of a texture of type `ty` reads, which the encoder's log keeps
/// (`replay.rs`): a texture buffer's fill the whole pattern, up to one byte past
/// [`TEXTURE_BUFFER_PATTERN_BYTES`] (a longer one, refused, is refused again where the call is
/// encoded again); another texture's 16 bytes at most, a unit's or a texel's samples'
/// ([`unit_of`], [`sample_units`]).
pub(super) fn read_pattern(ty: TextureType, pattern: &[u8]) -> &[u8] {
    let read = match ty {
        TextureType::TextureBuffer => TEXTURE_BUFFER_PATTERN_BYTES + 1,
        _ => 16,
    };
    &pattern[..pattern.len().min(read)]
}

/// The bits each sample of a multisampled texel of `bytes` bytes and `samples` samples takes
/// from `pattern` (the module docs): where the samples' bytes are 16 or fewer, the pattern
/// repeated from its start and cut at them, sample `s` its bytes from `s · bytes`; else each
/// sample the pattern cut at `bytes`.
pub(super) fn sample_units(pattern: &[u8], bytes: u32, samples: u32) -> Vec<Vec<u8>> {
    let all = bytes * samples;
    let unit = unit_of(pattern, if all <= 16 { all } else { bytes });
    let bytes = bytes as usize;
    (0..samples as usize)
        .map(|sample| {
            let at = sample * bytes % unit.len();
            unit[at..at + bytes].to_vec()
        })
        .collect()
}

/// The texel `unit` (1, 2, 4, 8 or 16 bytes) as the words of a clear value of the
/// unsigned-integer format of its size: one zero-extended word up to 4 bytes, else its words, low
/// bytes first.
pub(super) fn words(unit: &[u8]) -> [u32; 4] {
    let mut out = [0; 4];
    for (index, byte) in unit.iter().enumerate() {
        out[index / 4] |= u32::from(*byte) << (8 * (index % 4));
    }
    out
}

/// The longest pattern a texture buffer's fill takes: Metal's process ends in the fill of a longer
/// one (the module docs).
pub(super) const TEXTURE_BUFFER_PATTERN_BYTES: usize = 32752;

/// A fill's refusal for the length of its `pattern` over a texture of type `ty`, in Metal's own
/// order (the module docs). The two lengths fall on either side of what misuse is:
///
/// - **no bytes at all** is input plain Metal takes, so it is the runtime's own gap
///   ([`RuntimeGap::EmptyFillPattern`]): Metal's API validation layer stops it, and plain Metal
///   writes bytes the call never gave, which the runtime holds none of;
/// - **more than [`TEXTURE_BUFFER_PATTERN_BYTES`] into a texture buffer** is misuse
///   ([`Problem::LongTextureBufferFillPattern`]): Metal's own process ends in the call, whatever
///   the region. A fill of any other texture takes a pattern of any length.
pub(super) fn check_pattern(
    call: &'static str,
    ty: TextureType,
    pattern: &[u8],
) -> Result<(), Error> {
    if pattern.is_empty() {
        return Err(Error::NotImplemented(RuntimeGap::EmptyFillPattern));
    }
    if ty == TextureType::TextureBuffer && pattern.len() > TEXTURE_BUFFER_PATTERN_BYTES {
        let length = pattern.len() as u64;
        return Err(invalid(
            call,
            Problem::LongTextureBufferFillPattern { length },
        ));
    }
    Ok(())
}

/// The bytes Metal's fill of a texture buffer writes into its buffer's bytes `start..end` (the
/// region's texels from the texture's offset), from `pattern` of `n` bytes (the module docs):
/// byte `j` of the region (from its first) is `pattern[j & (n - 1)]`, but a 16-byte chunk of the
/// buffer the region covers whole, `j` its first byte's, takes 16 bytes of the fill's block from
/// byte `(j & (n - 1)) & !3`. The block is the power of two of bytes at least `n` and 16, holding
/// the pattern's whole copies from its start; its other bytes, and those after it, are Metal's
/// stack's, zeros in our probes, which the runtime writes. The block is read from the pattern
/// where a byte of it is asked for, never made.
pub(super) struct StreamFill<'a> {
    pattern: &'a [u8],
    /// The bytes of the pattern's whole copies at the block's start.
    copies: u64,
}

impl<'a> StreamFill<'a> {
    /// The fill of `pattern` (not empty).
    pub(super) fn new(pattern: &'a [u8]) -> StreamFill<'a> {
        let n = pattern.len() as u64;
        StreamFill {
            pattern,
            copies: StreamFill::block_of(n) / n * n,
        }
    }

    /// The bytes of the block of a pattern of `n` bytes: the power of two at least `n` and 16.
    fn block_of(n: u64) -> u64 {
        n.next_power_of_two().max(16)
    }

    /// The bytes of this fill's block, which are also the period of its chunks' bytes:
    /// `j & (n - 1)` repeats every power of two past `n - 1`.
    fn block(&self) -> u64 {
        StreamFill::block_of(self.pattern.len() as u64)
    }

    /// The mask of a byte's index into the pattern.
    fn mask(&self) -> u64 {
        self.pattern.len() as u64 - 1
    }

    /// Byte `at` of the block and of the bytes after it: the pattern's whole copies, then zeros.
    fn block_byte(&self, at: u64) -> u8 {
        match at < self.copies {
            true => self.pattern[(at % self.pattern.len() as u64) as usize],
            false => 0,
        }
    }

    /// The whole 16-byte chunks of the buffer that `bytes` covers.
    fn chunks(bytes: &Range<u64>) -> Range<u64> {
        let first = bytes.start.next_multiple_of(16);
        let last = bytes.end / 16 * 16;
        if first < last {
            first..last
        } else {
            bytes.end..bytes.end
        }
    }

    /// Byte `at` of the buffer, of the fill of `bytes`.
    pub(super) fn byte(&self, bytes: &Range<u64>, at: u64) -> u8 {
        let chunks = StreamFill::chunks(bytes);
        let j = at - bytes.start;
        if chunks.contains(&at) {
            let chunk = at / 16 * 16;
            let from = ((chunk - bytes.start) & self.mask()) & !3;
            self.block_byte(from + at % 16)
        } else {
            self.pattern[(j & self.mask()) as usize]
        }
    }

    /// Whether the fill of `bytes` writes a byte of the block past the pattern's whole copies.
    pub(super) fn reads_leftover(&self, bytes: Range<u64>) -> bool {
        let chunks = StreamFill::chunks(&bytes);
        (chunks.start..chunks.end.min(chunks.start + self.block()))
            .step_by(16)
            .any(|chunk| {
                let from = ((chunk - bytes.start) & self.mask()) & !3;
                from + 16 > self.copies
            })
    }

    /// The fill of `bytes` as spans of the buffer, each with the unit it holds repeated from its
    /// first byte: the bytes before the first whole chunk (fewer than 16, or fewer than 32 where
    /// no chunk is whole), the chunks (a period of them, the block's bytes, at most 32 KiB for a
    /// pattern of [`TEXTURE_BUFFER_PATTERN_BYTES`]), and the bytes after the last (fewer than
    /// 16): nothing of the region's length.
    pub(super) fn spans(&self, bytes: Range<u64>) -> Vec<(Range<u64>, Vec<u8>)> {
        let chunks = StreamFill::chunks(&bytes);
        let explicit = |span: Range<u64>| {
            let unit: Vec<u8> = span.clone().map(|at| self.byte(&bytes, at)).collect();
            (span, unit)
        };
        if chunks.is_empty() {
            return vec![explicit(bytes.clone())];
        }
        let first = chunks.start..(chunks.start + self.block()).min(chunks.end);
        let unit: Vec<u8> = first.map(|at| self.byte(&bytes, at)).collect();
        [
            explicit(bytes.start..chunks.start),
            (chunks.clone(), unit),
            explicit(chunks.end..bytes.end),
        ]
        .into_iter()
        .filter(|(span, _)| !span.is_empty())
        .collect()
    }
}

/// The units `from..from + count` of an axis of `units`, cut there.
fn cut(from: u64, count: u64, units: u64) -> Range<u64> {
    from.min(units)..from.saturating_add(count).min(units)
}

/// The texels of a two-plane extended-range level `width` texels wide that a fill of the units
/// `x` by `y` writes (the module docs): each row `y` written at [`two_planes::buffer_blit_rows`] apart, its
/// words landing on the texels of the level's rows `align(4 w, 16)` bytes apart, those past a row's
/// texels in the bytes between the rows. As rectangles `(x, y)` of the level, in rows that
/// ascend.
fn plane_rectangles(width: u64, x: Range<u64>, y: Range<u64>) -> Vec<(Range<u64>, Range<u64>)> {
    let rows = two_planes::buffer_blit_rows(width, true);
    let pitch = (4 * width).next_multiple_of(16);
    // The words written, as spans of bytes of the level that ascend and do not meet.
    let mut spans: Vec<Range<u64>> = Vec::new();
    for row in y {
        let span = row * rows + 4 * x.start..row * rows + 4 * x.end;
        match spans.last_mut() {
            Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
            _ => spans.push(span),
        }
    }
    // Each row's runs of texels, in order.
    let mut runs: Vec<(u64, Range<u64>)> = Vec::new();
    for span in spans {
        let mut at = span.start;
        while at < span.end {
            let row = at / pitch;
            let row_end = ((row + 1) * pitch).min(span.end);
            let texels = (at - row * pitch) / 4..((row_end - row * pitch) / 4).min(width);
            if !texels.is_empty() {
                match runs.last_mut() {
                    Some((last, run)) if *last == row && texels.start <= run.end => {
                        run.end = run.end.max(texels.end);
                    }
                    _ => runs.push((row, texels)),
                }
            }
            at = row_end;
        }
    }
    // Consecutive rows of one run each, the same, make one rectangle.
    let mut out: Vec<(Range<u64>, Range<u64>)> = Vec::new();
    for (i, (row, run)) in runs.iter().enumerate() {
        let alone = |j: usize| runs.iter().filter(|(r, _)| *r == runs[j].0).count() == 1;
        if let Some((columns, rows)) = out.last_mut() {
            if alone(i) && rows.end == *row && columns == run && i > 0 && alone(i - 1) {
                rows.end += 1;
                continue;
            }
        }
        out.push((run.clone(), *row..row + 1));
    }
    out
}

/// Where a fill of `region` at `level` and `slice` of `texture` writes, as plain Metal places it
/// (the module docs): the slice and the depth images, `None` where it writes nothing.
fn placement(
    texture: &Texture,
    level: u64,
    slice: u64,
    region: Region,
) -> Result<Option<(u64, Range<u64>)>, Error> {
    let record = texture.record();
    let Some(level) = u32::try_from(level)
        .ok()
        .filter(|&level| level < record.level_count)
    else {
        return Ok(None);
    };
    let slices = u64::from(record.layer_count);
    let (slice, images) = match texture.texture_type() {
        TextureType::Type1D | TextureType::Type2D | TextureType::Type2DMultisample => (0, 0..1),
        TextureType::Type3D => {
            let depth = u64::from(level_extent(record, level)[2]);
            (0, cut(region.origin.z, region.size.depth, depth))
        }
        TextureType::Cube => (slice % 6, 0..1),
        TextureType::Type1DArray
        | TextureType::Type2DArray
        | TextureType::CubeArray
        | TextureType::Type2DMultisampleArray
        | TextureType::TextureBuffer => {
            if slice >= slices {
                return Ok(None);
            }
            (slice, 0..1)
        }
    };
    Ok((!images.is_empty()).then_some((slice, images)))
}

/// Announce the device's clear of a plain color fill of `pf` to `color` where it may store another
/// texel than Metal's fill packs ([`device_apart`], `device-color-conversion`), and where it
/// converts a half or small-float NaN as the device likes ([`device_nan`], `device-nan-encoding`).
fn degrade_fill_conversion(device: &Device, pf: PixelFormat, color: [f64; 4]) {
    if device_apart(pf, color) {
        device.degrade(Degradation::DeviceColorConversion, || {
            (
                format!("a color fill of a {pf:?} texture to {color:?}"),
                "the device clears the region by its own conversion of the color, which Vulkan \
                 asks it to round to nearest, leaving a tie, an sRGB encode and a float's \
                 rounding to it, where Metal's fill packs it by rules of its own (a tie rounded \
                 half up, an sRGB channel the code whose decoded value is nearest): a channel can \
                 store a code next to Metal's"
                    .to_owned(),
            )
        });
    }
    if device_nan(pf, color) {
        device.degrade(Degradation::DeviceNanEncoding, || {
            (
                format!("a {pf:?} texture's color fill to a NaN"),
                "the device converts it to the format's NaN as it likes, where Metal's fill keeps \
                 a half NaN's sign and payload and makes a small float's all ones"
                    .to_owned(),
            )
        });
    }
}

/// The most bytes of a plain color fill's tile (the module docs): a larger region takes copies of
/// it side by side.
const TILE_BYTES: u64 = 1 << 20;

/// The most tiles an encoder keeps for its later plain color fills (the module docs).
const KEPT_TILES: usize = 4;

/// The least side of a plain color fill's tile, where its texture's first level is as large.
const TILE_SIDE: u32 = 16;

/// What a plain color fill's tile is made as: the pixel format, image format, samples and
/// dimension (a 1D image's, or 2D) of the textures it fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TileShape {
    pixel_format: PixelFormat,
    format: Format,
    samples: u32,
    one_row: bool,
}

/// A tile the device cleared to a color, which plain color fills copy from (the module docs).
pub(super) struct FillTile {
    shape: TileShape,
    extent: [u32; 2],
    /// The color it holds, its channels' bits.
    color: [u64; 4],
    texture: Texture,
}

/// A kept tile, taken for a fill.
enum Taken {
    /// One holding the fill's color.
    Cleared(FillTile),
    /// One of another color, for the fill to clear again.
    Other(FillTile),
}

/// The tiles an encoder keeps for its later plain color fills (the module docs), the one used
/// latest last: at most [`KEPT_TILES`].
#[derive(Default)]
pub(super) struct FillTiles(Vec<FillTile>);

impl FillTiles {
    /// The kept tile of `shape` at least `width` by `height` texels for a fill of `color`, taken
    /// out until it is kept again: one holding the color where there is one; else, where it keeps
    /// [`KEPT_TILES`] already, the one used least recently, to clear again; else none, and the
    /// fill makes one (so as many colors in turn clear each tile once).
    fn take(
        &mut self,
        shape: TileShape,
        [width, height]: [u32; 2],
        color: [u64; 4],
    ) -> Option<Taken> {
        let fits = |tile: &FillTile| {
            tile.shape == shape && tile.extent[0] >= width && tile.extent[1] >= height
        };
        if let Some(at) = (self.0.iter()).position(|tile| fits(tile) && tile.color == color) {
            return Some(Taken::Cleared(self.0.remove(at)));
        }
        if self.0.len() < KEPT_TILES {
            return None;
        }
        let at = self.0.iter().position(fits)?;
        Some(Taken::Other(self.0.remove(at)))
    }

    /// Keep `tile`, the one used latest: past [`KEPT_TILES`] the one used least recently goes
    /// (its command buffer holds it until it completes all the same).
    fn keep(&mut self, tile: FillTile) {
        self.0.push(tile);
        if self.0.len() > KEPT_TILES {
            self.0.remove(0);
        }
    }
}

/// The extent of the tile that a plain color fill of `width` by `height` texels, of `bytes` bytes
/// each (a texel's samples' together), copies from, in a texture whose first level is `limit`
/// texels: each side the power of two at least the region's and [`TILE_SIDE`], no larger than the
/// first level's (where the device made an image), then the longer side halved (rounded up) until
/// it holds [`TILE_BYTES`] at most. Fills of near sizes take one tile, and an encoder's fills of
/// growing regions make few.
pub(super) fn tile_extent(
    [width, height]: [u32; 2],
    bytes: u64,
    [wide, high]: [u32; 2],
) -> [u32; 2] {
    let side = |side: u32, limit: u32| {
        let rounded = side.max(TILE_SIDE).checked_next_power_of_two();
        rounded
            .map_or(limit, |rounded| rounded.min(limit))
            .max(side)
            .max(1)
    };
    let mut extent = [side(width, wide), side(height, high)];
    while u64::from(extent[0]) * u64::from(extent[1]) * bytes > TILE_BYTES && extent != [1, 1] {
        let longer = usize::from(extent[1] > extent[0]);
        extent[longer] = extent[longer].div_ceil(2);
    }
    extent
}

/// The copies out of a tile of `tile` texels that write the `size` texels from `origin` (its
/// depth image's third) of the image's `level` and `layer`: the region cut into parts of the
/// tile's extent from its origin, each copied from the tile's origin.
pub(super) fn tile_copies(
    tile: [u32; 2],
    origin: [u32; 3],
    size: [u32; 2],
    (level, layer): (u32, u32),
) -> Vec<ImageCopyRegion> {
    let at = |level, layer, origin, [width, height]: [u32; 2]| ImageRegion {
        aspect: ImageAspectFlags::COLOR,
        level,
        layer,
        layers: 1,
        origin,
        size: [width, height, 1],
    };
    let mut copies = Vec::new();
    for top in (0..size[1]).step_by(tile[1].max(1) as usize) {
        for left in (0..size[0]).step_by(tile[0].max(1) as usize) {
            let part = [tile[0].min(size[0] - left), tile[1].min(size[1] - top)];
            let to = [origin[0] + left, origin[1] + top, origin[2]];
            copies.push(ImageCopyRegion {
                src: at(0, 0, [0, 0, 0], part),
                dst: at(level, layer, to, part),
            });
        }
    }
    copies
}

/// Whether a plain color fill of every texel of `texture`'s image at its `level` (`extent`
/// texels) is the device's clear of that image there (the module docs): an image of its own (not
/// a buffer's bytes) of that extent at the level, made with transfer-destination usage
/// ("`image` must have been created with the `VK_IMAGE_USAGE_TRANSFER_DST_BIT` usage flag set",
/// `VUID-vkCmdClearColorImage-image-00002`), whose format takes transfer writes in the image's
/// tiling ("The format features of `image` must contain `VK_FORMAT_FEATURE_TRANSFER_DST_BIT`",
/// `VUID-vkCmdClearColorImage-image-01993`). The fill's formats are color formats, neither
/// compressed nor YCbCr nor of 64-bit channels (`VUID-vkCmdClearColorImage-image-00007`,
/// `-01545`, `-09678`), and every image rests in `GENERAL`
/// (`VUID-vkCmdClearColorImage-imageLayout-01394`: "`imageLayout` must be
/// `VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR`, `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` or
/// `VK_IMAGE_LAYOUT_GENERAL`").
fn in_place(texture: &Texture, level: u32, extent: [u32; 3]) -> Result<bool, Error> {
    if texture.over_buffer() {
        return Ok(false);
    }
    let plan = texture.image_plan()?;
    let planned = plan.extent.map(|axis| (axis >> level.min(31)).max(1));
    if planned != extent || !plan.usage.contains(ImageUsageFlags::TRANSFER_DST) {
        return Ok(false);
    }
    let device = texture.device();
    let profile = device.compile_device().profile();
    let facts = device.shared().textures.facts(profile);
    let features = match texture.linear_image()? {
        true => facts.linear(plan.format),
        false => facts.optimal(plan.format),
    };
    Ok(features.contains(FormatFeatureFlags2::TRANSFER_DST))
}

impl BlitCommandEncoder {
    /// `fillTexture:level:slice:region:bytes:length:` (Apple SPI): every unit of `region` of the
    /// texture's `level` and `slice` becomes `pattern`, repeated from its start and cut at the
    /// unit's bytes, as plain Metal writes it (`blit_fill.rs`). A compressed format's unit is a
    /// block, and the region's origin and size count blocks; every other format's is a texel (a
    /// 4:2:2 format's 2 bytes, a stencil format's 1, the depth of `Depth32Float_Stencil8`, a
    /// two-plane YCbCr texture's plane code, a two-plane extended-range texel's color word). A
    /// packed two-plane YCbCr texture's texels each take one zero byte, whatever the pattern. The
    /// region is cut at the level, and an empty one writes nothing, but a texture buffer's (below).
    ///
    /// Placed as plain Metal places it (the module docs): a level past the texture's, or a slice
    /// past an array's, writes nothing; a 1D, 2D or 3D texture ignores the slice and a cube takes
    /// it modulo 6; a 3D texture writes the region's depth images, every other type one image; a
    /// texture buffer's fill writes its buffer's bytes from the region's first texel, as many as
    /// its texels', past the texture's width and cut at the buffer's end, whatever the level,
    /// slice, row, height and depth, the pattern running across them, not started again at each
    /// texel (`texture-buffer-fill-leftover` where Metal's bytes are its stack's). A two-plane
    /// extended-range texture's rows are Metal's fill rows, and a level past its first writes level
    /// 0, as Metal's fills do. A multisample texture's samples each take their own bytes of the
    /// pattern where a texel's samples hold 16 bytes or fewer, else the pattern cut at the texel
    /// (the module docs).
    ///
    /// Refused: a texture of another device (`Invalid`), and a texture buffer's fill of a pattern
    /// of more than 32752 bytes (`Invalid(LongTextureBufferFillPattern)`), in which Metal's process
    /// ends, whatever the region; another texture's fill takes a pattern of any length and reads 16
    /// bytes of it at most. An empty pattern is the runtime's own gap
    /// (`NotImplemented(RuntimeGap::EmptyFillPattern)`): Metal's API validation layer stops it, and
    /// plain Metal takes the fill and writes bytes the call never gave, a unit read from the
    /// pattern's address or, for a texture buffer, the fill's leftovers. A multisample
    /// texture's fill is refused, `Unsupported`, where the device cannot run the sample planes'
    /// merge (sample-rate shading, the sample count among its integer color attachments') or the
    /// copy into the texture (`copy_from_texture`'s refusals of copies between multisampled
    /// textures).
    pub fn fill_texture(
        &mut self,
        texture: &Texture,
        level: u64,
        slice: u64,
        region: Region,
        pattern: &[u8],
    ) -> Result<(), Error> {
        self.logged(
            |log| {
                BlitCall::FillTexture(
                    Box::new((texture.clone(), level, slice, region)),
                    log.byte_span(read_pattern(texture.texture_type(), pattern)),
                )
            },
            |this| this.fill("fill_texture", texture, (level, slice), region, pattern),
        )
    }

    /// `fillTexture:level:slice:region:color:` (`pixel_format` `None`) and
    /// `fillTexture:level:slice:region:color:pixelFormat:` (Apple SPI): [`fill_texture`] of the
    /// texel `color` packs to in `pixel_format`, the texture's own format without it, as Metal's
    /// fills pack it (`fill_value.rs`), which is not how a `Clear` load action converts it: an sRGB
    /// channel takes the code whose decoded value is nearest, `RG11B10Float` rounds to nearest
    /// even, an integer channel's NaN is 0 or the extreme by its kind, and the extended-range
    /// formats are packed.
    ///
    /// A plain fill (`pixel_format` `None`) of a texture of a normalized, sRGB, half, float or
    /// small-float format the device renders is the device's clear of the color over the region
    /// (the module docs), placed as the fill: a texel can stand a code from Metal's packing where
    /// its conversion is in question, announced (`Degradation::DeviceColorConversion`; a half or
    /// small-float NaN `Degradation::DeviceNanEncoding`).
    ///
    /// A color packed in a format Metal packs no color in (compressed, 4:2:2, depth, stencil,
    /// two-plane, YCbCr, padded 10-bit, `Invalid`, the private formats) fills the command buffer's
    /// last color fill's packed texel, over zeros at its start (the module docs): Metal fills bytes
    /// of the calling thread's stack there, which hold that texel only where no other call used
    /// them since. The device announces `Degradation::UnpackedFillColor` once.
    ///
    /// Refused as [`fill_texture`] refuses.
    ///
    /// [`fill_texture`]: BlitCommandEncoder::fill_texture
    pub fn fill_texture_color(
        &mut self,
        texture: &Texture,
        level: u64,
        slice: u64,
        region: Region,
        color: ClearColor,
        pixel_format: Option<PixelFormat>,
    ) -> Result<(), Error> {
        const CALL: &str = "fill_texture_color";
        // Refused before the command buffer's fill slot takes the texel, as its log is.
        self.refuse_aborted()?;
        let packed_in = pixel_format.unwrap_or(texture.pixel_format());
        let channels = [color.red, color.green, color.blue, color.alpha];
        // The texel, which the command buffer's slot takes; else the slot's bytes.
        let texel = fill_texel(packed_in, channels);
        let slot = self.core().command_buffer().fill_slot(texel.as_deref());
        // A plain fill of a texture of a format the device renders: the device's clear of the
        // color (the module docs). Logged as the call, which encoded again clears again.
        let cleared = match pixel_format {
            None => self.device_fill(texture)?.and_then(|format| {
                clear_value::held_color(packed_in, format, channels, ClearRoute::ThroughHalf)
            }),
            Some(_) => None,
        };
        if let Some(value) = cleared {
            return self.logged(
                |_| {
                    BlitCall::FillTextureColor(Box::new((
                        texture.clone(),
                        level,
                        slice,
                        region,
                        color,
                    )))
                },
                |this| this.cleared_fill(CALL, texture, (level, slice), region, (value, channels)),
            );
        }
        let pattern = texel.clone().unwrap_or(slot.to_vec());
        // Logged as the bytes it fills, which a call encoded again fills again.
        self.logged(
            |log| {
                BlitCall::FillTexture(
                    Box::new((texture.clone(), level, slice, region)),
                    log.byte_span(&pattern),
                )
            },
            |this| {
                this.fill(CALL, texture, (level, slice), region, &pattern)?;
                if texel.is_none() {
                    let format = texture.pixel_format();
                    this.core()
                        .device()
                        .degrade(Degradation::UnpackedFillColor, || {
                            (
                        format!("a color fill of a {format:?} texture, packed in {packed_in:?}"),
                        "Metal packs no color there and fills bytes of the calling thread's \
                         stack, which hold the last color fill's packed texel only where no \
                         other call used them since; the runtime fills the command buffer's last \
                         color fill's packed texels, over zeros"
                            .to_owned(),
                    )
                        });
                }
                Ok(())
            },
        )
    }

    /// `fillBuffer:range:pattern4:` (Apple SPI): the bytes of `range` take `pattern`'s 4 bytes,
    /// its low byte first, as plain Metal writes them (the module docs): from the range's start,
    /// the pattern restarting at each multiple of 16 bytes of the buffer, so byte `k` of the buffer
    /// takes the pattern's byte `k` modulo 4 from the range's first such multiple on.
    ///
    /// An empty range fills nothing. Refused (`Invalid`): a range past the buffer, a buffer of
    /// another device.
    pub fn fill_buffer_pattern4(
        &mut self,
        buffer: &Buffer,
        range: Range<u64>,
        pattern: u32,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::FillBufferPattern4(buffer.clone(), range.clone(), pattern),
            |this| {
                const CALL: &str = "fill_buffer_pattern4";
                this.core().check_device(CALL, buffer.device())?;
                let size = range.end.saturating_sub(range.start);
                if size == 0 {
                    return Ok(());
                }
                let range = bytes_in(CALL, range.start, size, buffer.length())?;
                let pattern = pattern.to_le_bytes();
                // The bytes before the first multiple of 16, the pattern from the range's start;
                // then the pattern from that multiple, in copies of at most `STAGING_BYTES` (a
                // multiple of 4, so each starts the pattern again).
                let restart = range.start.next_multiple_of(16).min(range.end);
                let head = restart - range.start;
                let body = range.end - restart;
                let chunk = body.min(STAGING_BYTES);
                let bytes: Vec<u8> = (0..head)
                    .map(|k| pattern[(k % 4) as usize])
                    .chain(pattern.iter().copied().cycle().take(chunk as usize))
                    .collect();
                let staging = this.staged(&bytes)?;
                let (source, from) = staging.at(0);
                // Read as a blit's (`host_residency.rs`): where the range starts in the `VkBuffer`,
                // and each byte past it as far on.
                let (destination, origin) = this.core().blit_at(buffer, range.start)?;
                let mut regions = Vec::new();
                if head > 0 {
                    regions.push(vk::BufferCopy {
                        src_offset: from,
                        dst_offset: origin,
                        size: head,
                    });
                }
                let mut at = restart;
                while at < range.end {
                    let size = (range.end - at).min(chunk);
                    regions.push(vk::BufferCopy {
                        src_offset: from + head,
                        dst_offset: origin + (at - range.start),
                        size,
                    });
                    at += size;
                }
                let length = bytes.len() as u64;
                let uses = [reads(&staging, 0..length), writes(buffer, range)];
                this.command(&uses, |device, raw| {
                    // SAFETY: the encoder's command buffer is recording; the staging buffer and the
                    // buffer are live and retained until it completes; each region lies inside
                    // both, and the buffer's regions are disjoint.
                    unsafe { device.cmd_copy_buffer(raw, source, destination, &regions) };
                })?;
                this.core().retain(staging);
                this.core().retain(buffer.clone());
                Ok(())
            },
        )
    }

    /// The Vulkan format in which a plain color fill of `texture` is the device's clear of the
    /// color (the module docs): its image's, where that holds Metal's bytes of the texture's format
    /// (`metal::format::byte_format`), the format's fill converts a `float`
    /// ([`device_converts`]), the device renders the format and copies images of it both ways,
    /// and the texture is not a texture buffer, whose fill writes its buffer's bytes. `None` where
    /// the fill packs the color.
    fn device_fill(&mut self, texture: &Texture) -> Result<Option<Format>, Error> {
        let pf = texture.pixel_format();
        let texels = matches!(
            fill_unit(pf),
            Some(FillUnit::Texels {
                option: BlitOption::None,
                ..
            })
        );
        if texture.texture_type() == TextureType::TextureBuffer || !texels || !device_converts(pf) {
            return Ok(None);
        }
        let Some(format) = byte_format(pf) else {
            return Ok(None);
        };
        if texture
            .image_plan()
            .map_or(true, |plan| plan.format != format)
        {
            return Ok(None);
        }
        let device = self.core().device();
        let profile = device.compile_device().profile();
        let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
        let renders = profile.supports_format(format, FormatUsage::ColorAttachment)
            && device
                .shared()
                .textures
                .facts(profile)
                .optimal(format)
                .contains(copies);
        Ok(renders.then_some(format))
    }

    /// The device's clear of `color` (its clear value `value`) into `region` of `texture`'s `level`
    /// and `slice`, placed as a fill (the module docs). A fill of whole subresources whose image
    /// takes a transfer write is the clear of the texture's own image there
    /// ([`clear_in_place`](Self::clear_in_place)); any other copies a tile the device cleared to
    /// the color ([`cleared_tile`](Self::cleared_tile)) into the region of each depth image the
    /// fill writes, side by side from the region's origin, as a blit copy between the two textures
    /// moves it. Announced where the device's conversion may store another texel than Metal's
    /// fill packs.
    fn cleared_fill(
        &mut self,
        call: &'static str,
        texture: &Texture,
        (level, slice): (u64, u64),
        region: Region,
        (value, color): (vk::ClearColorValue, [f64; 4]),
    ) -> Result<(), Error> {
        self.core().check_device(call, texture.device())?;
        texture.check_plane_copied()?;
        self.see(texture);
        let Size {
            width,
            height,
            depth,
        } = region.size;
        if width == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        let Some((slice, images)) = placement(texture, level, slice, region)? else {
            return Ok(());
        };
        let level = u32::try_from(level).map_err(|_| internal("a fill's level"))?;
        let record = texture.record();
        let extent = level_extent(record, level);
        let [w, h, d] = extent.map(u64::from);
        let x = cut(region.origin.x, width, w);
        let y = cut(region.origin.y, height, h);
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }
        let narrow = |v: u64| u32::try_from(v).map_err(|_| internal("a fill's region"));
        let [left, top] = [narrow(x.start)?, narrow(y.start)?];
        let size = [narrow(x.end - x.start)?, narrow(y.end - y.start)?];
        let (level, layer) = (
            record.base_level + level,
            record.base_layer + narrow(slice)?,
        );
        // Every texel of the level's layer, or of a 3D level's depth images.
        let whole = x == (0..w) && y == (0..h) && images == (0..d);
        if whole && in_place(texture, level, extent)? {
            self.clear_in_place(call, texture, (level, layer), extent, value)?;
        } else {
            let (tile, held) = self.cleared_tile(texture, size, (value, color))?;
            let mut copies = Vec::new();
            for z in images {
                copies.extend(tile_copies(
                    held,
                    [left, top, narrow(z)?],
                    size,
                    (level, layer),
                ));
            }
            self.texture_copies(call, &tile, texture, &copies)?;
        }
        degrade_fill_conversion(self.core().device(), texture.pixel_format(), color);
        Ok(())
    }

    /// The device's clear of `value` into every texel of `texture`'s own image at `level` and
    /// `layer` (the image's, `extent` texels), one `vkCmdClearColorImage`: nothing is made. Written
    /// as a blit's copy into the subresource writes it (a heap shadow's every texel, a sparse
    /// texture's scrub), the note of a texture Metal keeps compressed announced as there.
    fn clear_in_place(
        &mut self,
        call: &'static str,
        texture: &Texture,
        (level, layer): (u32, u32),
        [width, height, depth]: [u32; 3],
        value: vk::ClearColorValue,
    ) -> Result<(), Error> {
        let image = texture.raw_image()?;
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let uses = [texture.image_use(level..level + 1, layer..layer + 1, write, true)?];
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: level,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        };
        let covered = [Covered {
            image,
            subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: level,
                base_array_layer: layer,
                layer_count: 1,
            },
            offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width,
                height,
                depth,
            },
        }];
        self.command_reading(&uses, (&[], &covered), |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the image
            // is live and retained until it completes, rests in `GENERAL` and was made with
            // transfer-destination usage, its format (a color format, neither compressed nor
            // YCbCr, nor of 64-bit channels) taking transfer writes in its tiling (`in_place`);
            // the range is one of its levels and layers (a 3D image's one).
            unsafe {
                device.cmd_clear_color_image(raw, image, vk::ImageLayout::GENERAL, &value, &[range])
            };
        })?;
        self.core().retain(texture.clone());
        let written = ImageRegion {
            aspect: ImageAspectFlags::COLOR,
            level,
            layer,
            layers: 1,
            origin: [0, 0, 0],
            size: [width, height, depth],
        };
        self.decode_twin(texture, &[written])?;
        self.scrub(
            texture,
            vec![(vk::ImageAspectFlags::COLOR, Written::of(&written))],
        )?;
        degrade_compressed_blit(call, texture)
    }

    /// A tile the device cleared to `color` (its clear value `value`), of `texture`'s image
    /// format, samples and dimension (a 1D image's, or 2D), to copy the region of `size` texels
    /// from (the module docs), and its extent: the encoder's tile of that shape and color where
    /// it keeps one as large as [`tile_extent`] asks; else, where it keeps [`KEPT_TILES`], its
    /// least recently used such tile of another color, cleared again; else a new one, which its
    /// command buffer holds until it completes. A tile is cleared whole by the device
    /// (`vkCmdClearColorImage`, the device's conversion of a clear value). Kept by the encoder for
    /// its later fills ([`FillTiles`]).
    fn cleared_tile(
        &mut self,
        texture: &Texture,
        size: [u32; 2],
        (value, color): (vk::ClearColorValue, [f64; 4]),
    ) -> Result<(Texture, [u32; 2]), Error> {
        let pf = texture.pixel_format();
        let (format, one_row) = {
            let plan = texture.image_plan()?;
            (plan.format, plan.image_type == ImageType::TYPE_1D)
        };
        let samples = texture.record().samples;
        let shape = TileShape {
            pixel_format: pf,
            format,
            samples,
            one_row,
        };
        let texel = block_bytes(pf).ok_or_else(|| internal("a fill tile of no texel's bytes"))?;
        let first = level_extent(texture.record(), 0);
        let extent = tile_extent(
            size,
            u64::from(texel) * u64::from(samples),
            [first[0], first[1]],
        );
        let color = color.map(f64::to_bits);
        let tile = match self.fill_tiles().take(shape, extent, color) {
            Some(Taken::Cleared(tile)) => tile,
            Some(Taken::Other(tile)) => {
                self.clear_tile(&tile.texture, value)?;
                FillTile { color, ..tile }
            }
            None => {
                let texture = self.new_tile(shape, extent)?;
                self.clear_tile(&texture, value)?;
                FillTile {
                    shape,
                    extent,
                    color,
                    texture,
                }
            }
        };
        let made = (tile.texture.clone(), tile.extent);
        self.fill_tiles().keep(tile);
        Ok(made)
    }

    /// A tile of `shape` over `width` by `height` texels: a texture the command buffer holds
    /// until it completes.
    fn new_tile(&mut self, shape: TileShape, [width, height]: [u32; 2]) -> Result<Texture, Error> {
        let TileShape {
            pixel_format: pf,
            format,
            samples,
            one_row,
        } = shape;
        let texture_type = match (one_row, samples > 1) {
            (true, _) => TextureType::Type1D,
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DMultisample,
        };
        let descriptor = TextureDescriptor {
            texture_type,
            pixel_format: pf,
            width: width.into(),
            height: height.into(),
            sample_count: samples.into(),
            usage: TextureUsage::RENDER_TARGET,
            resource_options: ResourceOptions::private(),
            // Not a texture Metal keeps compressed, whose blits rewrite texels: the copy out of
            // it moves the clear's bits (`compressed_blit.rs`).
            allow_gpu_optimized_contents: false,
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format: pf,
            flags: ImageCreateFlags::empty(),
            image_type: match one_row {
                true => ImageType::TYPE_1D,
                false => ImageType::TYPE_2D,
            },
            format,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: sample_count(samples)
                .ok_or_else(|| internal("a fill of a sample count Vulkan lacks"))?,
            usage: ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE
                | TextureUsage::SHADER_ATOMIC
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        Device::new_planned_texture(self.core().command_buffer(), &descriptor, plan)
    }

    /// The device's clear of the whole `tile` to `value` (`vkCmdClearColorImage`), after its
    /// earlier copies out read it.
    fn clear_tile(&mut self, tile: &Texture, value: vk::ClearColorValue) -> Result<(), Error> {
        let image = tile.raw_image()?;
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let uses = [tile.image_use(0..1, 0..1, write, true)?];
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        self.command(&uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the image
            // is live until it completes (the command buffer holds the tile), rests in `GENERAL`
            // and has transfer-destination usage; the range is its one level and layer, and its
            // format, one the device renders, takes transfer writes.
            unsafe {
                device.cmd_clear_color_image(raw, image, vk::ImageLayout::GENERAL, &value, &[range])
            };
        })
    }

    /// The fill of `pattern` into `region` of `texture`'s `level` and `slice` (the module docs).
    fn fill(
        &mut self,
        call: &'static str,
        texture: &Texture,
        (level, slice): (u64, u64),
        region: Region,
        pattern: &[u8],
    ) -> Result<(), Error> {
        self.core().check_device(call, texture.device())?;
        texture.check_plane_copied()?;
        self.see(texture);
        check_pattern(call, texture.texture_type(), pattern)?;
        let unit = fill_unit(texture.pixel_format())
            .ok_or_else(|| internal("a fill of a format the runtime makes no texture of"))?;
        // A texture buffer's fill writes one row whatever the region's height and depth.
        if texture.texture_type() == TextureType::TextureBuffer {
            return self.texture_buffer_fill(texture, unit, region, pattern);
        }
        let Size {
            width,
            height,
            depth,
        } = region.size;
        if width == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        let Some((slice, images)) = placement(texture, level, slice, region)? else {
            return Ok(());
        };
        let level_u32 = u32::try_from(level).map_err(|_| internal("a fill's level"))?;
        let [w, h, _] = level_extent(texture.record(), level_u32).map(u64::from);
        let (bytes, extent, option) = match unit {
            FillUnit::Blocks { bytes, extent } => (bytes, extent, BlitOption::None),
            FillUnit::Texels { bytes, option } => (bytes, [1, 1], option),
            FillUnit::PlaneWords => (4, [1, 1], BlitOption::None),
            FillUnit::PackedCodes => (1, [1, 1], BlitOption::None),
        };
        let [bw, bh] = extent.map(u64::from);
        let x = cut(region.origin.x, width, w.div_ceil(bw));
        let y = cut(region.origin.y, height, h.div_ceil(bh));
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }
        if texture.record().samples > 1 {
            return self.sample_fill(call, texture, (level_u32, slice), option, (x, y), pattern);
        }
        let unit_bytes = unit_of(pattern, bytes);
        if unit == FillUnit::PackedCodes {
            return self.packed_code_fill(texture, (x, y));
        }
        if unit == FillUnit::PlaneWords && !texture.over_buffer() {
            // Every level's fill writes the heap's bytes of level 0 of the slice, at its rows; a
            // mipmapped 3D texture's past its first depth image, none (the module docs).
            if let Some((heap, at, layout)) = texture.heap_two_planes() {
                if texture.texture_type() == TextureType::Type3D
                    && layout.levels > 1
                    && images.start > 0
                {
                    return Ok(());
                }
                let at = at + slice * layout.color_slice;
                return self.plane_heap_fill(
                    texture,
                    (heap, at),
                    layout.width,
                    (x, y),
                    &unit_bytes,
                );
            }
            // Every level is filled as level 0, at Metal's fill rows (the module docs).
            return self.plane_texel_fill(texture, slice, (x, y), &unit_bytes);
        }
        self.fill_rows(
            texture,
            (level, slice, images),
            option,
            (x, y),
            ([bw, bh], [w, h]),
            &unit_bytes,
        )
    }

    /// The fill of `pattern` into the texels `x` by `y` of the multisampled `texture`'s `level`
    /// and `slice`, in the aspect `option` names (the module docs): each sample its own bits
    /// ([`sample_units`]), written by the sample planes' merge into a multisampled image of the
    /// unsigned-integer format of the texel's size ([`BitsCopy::fill`]), which a blit copy then
    /// moves into the texture's region as a copy between multisampled textures moves it
    /// (`blit_texture.rs`: an image copy, the planes' copies where the device's copy converts
    /// float NaNs, a depth or stencil aspect's copies or draws, `sample_copy.rs`, exact or
    /// announced as there).
    fn sample_fill(
        &mut self,
        call: &'static str,
        texture: &Texture,
        (level, slice): (u32, u64),
        option: BlitOption,
        (x, y): (Range<u64>, Range<u64>),
        pattern: &[u8],
    ) -> Result<(), Error> {
        let record = texture.record();
        let samples = record.samples;
        let CopyUnit { bytes, aspect, .. } = blit_texture::unit(call, texture, option)?;
        let copy = match bytes {
            1 => PlanesCopy::R8,
            2 => PlanesCopy::R16,
            4 => PlanesCopy::R32,
            8 => PlanesCopy::Rg32,
            16 => PlanesCopy::Rgba32,
            _ => {
                return Err(internal(
                    "a multisampled texel of no unsigned format's size",
                ))
            }
        };
        let values: Vec<[u32; 4]> = sample_units(pattern, bytes, samples)
            .iter()
            .map(|unit| words(unit))
            .collect();
        let narrow = |range: &Range<u64>| -> Result<[u32; 2], Error> {
            let start = u32::try_from(range.start).map_err(|_| internal("a fill's origin"))?;
            let end = u32::try_from(range.end).map_err(|_| internal("a fill's extent"))?;
            Ok([start, end - start])
        };
        let ([left, width], [top, height]) = (narrow(&x)?, narrow(&y)?);
        let (scratch, fill) = BitsCopy::fill(
            self.core().command_buffer(),
            copy,
            &values,
            [left, top],
            [width, height],
        )?;
        self.command_in(BITS_COPY_STAGES, &fill.uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the
            // scratch image and the fill's planes and objects are live until it completes
            // (retained below).
            unsafe { fill.record(device, raw) };
        })?;
        self.core().retain(vec![fill]);
        let slice = u32::try_from(slice).map_err(|_| internal("a fill's slice"))?;
        let region = |aspect, level, layer| ImageRegion {
            aspect,
            level,
            layer,
            layers: 1,
            origin: [left, top, 0],
            size: [width, height, 1],
        };
        let copy = ImageCopyRegion {
            src: region(ImageAspectFlags::COLOR, 0, 0),
            dst: region(aspect, record.base_level + level, record.base_layer + slice),
        };
        self.texture_copies(call, &scratch, texture, &[copy])
    }

    /// Copy `unit` into the units `x` by `y` of each depth image `images` of `texture`'s `level`
    /// and `slice`, `block` texels each (cut at the level's `size`), through `option`: copies
    /// from one staging buffer of whole rows of units, in rows of at most `STAGING_BYTES`.
    fn fill_rows(
        &mut self,
        texture: &Texture,
        (level, slice, images): (u64, u64, Range<u64>),
        option: BlitOption,
        (x, y): (Range<u64>, Range<u64>),
        (block, size): ([u64; 2], [u64; 2]),
        unit: &[u8],
    ) -> Result<(), Error> {
        let row_bytes = (x.end - x.start) * unit.len() as u64;
        let per_copy = (STAGING_BYTES / row_bytes).clamp(1, y.end - y.start);
        let staging = self.staging(unit, row_bytes * per_copy)?;
        let left = x.start * block[0];
        let width = (x.end * block[0]).min(size[0]) - left;
        for z in images {
            let mut row = y.start;
            while row < y.end {
                let end = (row + per_copy).min(y.end);
                let top = row * block[1];
                let height = (end * block[1]).min(size[1]) - top;
                self.unlogged(|this| {
                    this.copy_from_buffer_to_texture(
                        &staging,
                        0,
                        row_bytes,
                        row_bytes * (end - row),
                        Size::new(width, height, 1),
                        texture,
                        slice,
                        level,
                        Origin::new(left, top, z),
                        option,
                    )
                })?;
                row = end;
            }
        }
        Ok(())
    }

    /// Write `word` at each texel `x` by `y` of a level of the two-plane `texture` whose first
    /// level, `width` texels wide, has the slice filled lying in `heap`'s bytes from byte `at`, at
    /// Metal's fill rows of that first level: the texels' color words, rows that meet merged, in
    /// one copy into the heap's bytes. Its alpha bytes are not written.
    fn plane_heap_fill(
        &mut self,
        texture: &Texture,
        (heap, at): (&Heap, u64),
        width: u64,
        (x, y): (Range<u64>, Range<u64>),
        word: &[u8],
    ) -> Result<(), Error> {
        let Some(bytes) = heap.bytes() else {
            return Err(internal("a two-plane texture's bytes in a heap of none"));
        };
        let rows = two_planes::buffer_blit_rows(width, true);
        // Each row's words, from byte `at + row * rows + 4 x`; rows ascend, so a row that meets
        // the one before extends it.
        let mut spans: Vec<Range<u64>> = Vec::new();
        for row in y {
            let span = at + row * rows + 4 * x.start..at + row * rows + 4 * x.end;
            match spans.last_mut() {
                Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
                _ => spans.push(span),
            }
        }
        self.fill_spans(bytes, &spans, word)?;
        self.core().retain(texture.clone());
        Ok(())
    }

    /// Write `unit` repeated from each span's first byte into the bytes `spans` of `buffer`
    /// (ascending, disjoint), in one command of copies from a staging buffer.
    fn fill_spans(
        &mut self,
        buffer: &Buffer,
        spans: &[Range<u64>],
        unit: &[u8],
    ) -> Result<(), Error> {
        let (Some(first), Some(last)) = (spans.first(), spans.last()) else {
            return Ok(());
        };
        let written = first.start..last.end;
        // A staging buffer of whole units, so each copy from its start begins the unit again.
        let longest = spans.iter().map(|s| s.end - s.start).max().unwrap_or(0);
        let whole = unit.len() as u64;
        let length = longest.min((STAGING_BYTES / whole).max(1) * whole);
        let staging = self.staging(unit, length)?;
        let (source, from) = staging.at(0);
        // Read as a blit's (`host_residency.rs`).
        let (destination, base) = self.core().blit_at(buffer, written.start)?;
        let mut regions = Vec::new();
        for span in spans {
            let mut start = span.start;
            while start < span.end {
                let size = (span.end - start).min(length);
                regions.push(vk::BufferCopy {
                    src_offset: from,
                    dst_offset: base + (start - written.start),
                    size,
                });
                start += size;
            }
        }
        let uses = [reads(&staging, 0..length), writes(buffer, written)];
        self.command(&uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording; the staging buffer and the
            // buffer are live and retained until it completes; each region lies inside both,
            // and the buffer's regions are disjoint.
            unsafe { device.cmd_copy_buffer(raw, source, destination, &regions) };
        })?;
        self.core().retain(staging);
        self.core().retain(buffer.clone());
        Ok(())
    }

    /// Write `pattern` into the bytes of the texture buffer `texture`'s buffer that the texels `x`
    /// to `x + width` of `region` cover from its offset, past its width too (cut at the buffer's
    /// end), whatever the level, slice, row, height and depth, as Metal writes them
    /// ([`StreamFill`]; the module docs): a width of 0 writes nothing. Its pattern's length is
    /// already checked ([`check_pattern`]).
    fn texture_buffer_fill(
        &mut self,
        texture: &Texture,
        unit: FillUnit,
        region: Region,
        pattern: &[u8],
    ) -> Result<(), Error> {
        let FillUnit::Texels { bytes, .. } = unit else {
            return Err(internal("a texture buffer of no texel format"));
        };
        let Some(texels) = texture.buffer_texels() else {
            return Err(internal("a texture buffer over no buffer"));
        };
        let buffer = texels.buffer.clone();
        let bytes = u64::from(bytes);
        let length = buffer.length();
        let start = texels
            .layout
            .offset
            .saturating_add(region.origin.x.saturating_mul(bytes))
            .min(length);
        let end = start
            .saturating_add(region.size.width.saturating_mul(bytes))
            .min(length);
        if start >= end {
            return Ok(());
        }
        let stream = StreamFill::new(pattern);
        for (span, unit) in stream.spans(start..end) {
            self.fill_spans(&buffer, &[span], &unit)?;
        }
        if stream.reads_leftover(start..end) {
            let format = texture.pixel_format();
            self.core()
                .device()
                .degrade(Degradation::TextureBufferFillLeftover, || {
                    (
                        format!(
                            "a fill of a {format:?} texture buffer with a pattern of {} bytes",
                            pattern.len()
                        ),
                        "Metal reads bytes of its stack past the pattern's whole copies there, \
                         zeros in our probes and other bytes in other processes; the runtime \
                         reads zeros"
                            .to_owned(),
                    )
                });
        }
        self.core().retain(texture.clone());
        Ok(())
    }

    /// Fill the texels `x` by `y` of a packed two-plane YCbCr `texture` over a surface's plane:
    /// one zero byte at each texel's `x` of its row of the plane (the module docs).
    fn packed_code_fill(
        &mut self,
        texture: &Texture,
        (x, y): (Range<u64>, Range<u64>),
    ) -> Result<(), Error> {
        let Some(texels) = texture.buffer_texels() else {
            return Err(internal("a packed two-plane texture over no surface"));
        };
        let (offset, pitch) = (texels.layout.offset, texels.layout.row_pitch);
        let spans: Vec<Range<u64>> = y
            .map(|row| offset + row * pitch)
            .map(|start| start + x.start..start + x.end)
            .collect();
        let buffer = texels.buffer.clone();
        self.fill_spans(&buffer, &spans, &[0])?;
        self.core().retain(texture.clone());
        Ok(())
    }

    /// Write `word` at the texels of level 0 of `slice` of the two-plane extended-range `texture`
    /// (of its own memory, or a heap's held apart) that a fill of the units `x` by `y` of any of
    /// its levels writes (the module docs): copies of the color words alone, at Metal's fill
    /// rows mapped onto the texels' rows.
    fn plane_texel_fill(
        &mut self,
        texture: &Texture,
        slice: u64,
        (x, y): (Range<u64>, Range<u64>),
        word: &[u8],
    ) -> Result<(), Error> {
        let [width, height, _] = level_extent(texture.record(), 0).map(u64::from);
        let layer = u32::try_from(slice).map_err(|_| internal("a fill's slice"))?;
        // The words a fill writes between the level's rows: held where its blits with a buffer
        // read them back (one level, `two_planes::rows_held`), as the rows of plane 0 Metal's
        // fill writes, rows that meet merged; else not held, noted, so a blit that reads them
        // back announces it (`two_planes.rs`).
        let size = [x.end - x.start, y.end - y.start];
        let spans = two_planes::row_spans(width, [x.start, y.start], size, true);
        if spans
            .iter()
            .flatten()
            .any(|span| two_planes::between_rows(span, width))
        {
            if two_planes::rows_held(texture) {
                let held = self.held_between_rows(texture, [width, height])?;
                let base = slice * height * (4 * width).next_multiple_of(16);
                let mut merged: Vec<Range<u64>> = Vec::new();
                for span in spans.into_iter().flatten() {
                    let span = base + span.start..base + span.end;
                    match merged.last_mut() {
                        Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
                        _ => merged.push(span),
                    }
                }
                self.fill_spans(&held, &merged, word)?;
            } else {
                texture.keep_outside();
            }
        }
        for (columns, rows) in plane_rectangles(width, x, y) {
            let w = columns.end - columns.start;
            let per_copy = (STAGING_BYTES / (4 * w)).clamp(1, rows.end - rows.start);
            let staging = self.staging(word, 4 * w * per_copy)?;
            let mut row = rows.start;
            while row < rows.end {
                let end = (row + per_copy).min(rows.end);
                let region = ImageRegion {
                    aspect: ImageAspectFlags::COLOR,
                    level: 0,
                    layer,
                    layers: 1,
                    origin: [columns.start, row, 0].map(|v| v as u32),
                    size: [w, end - row, 1].map(|v| v as u32),
                };
                let layout = BufferLayout {
                    offset: 0,
                    bytes_per_row: 4 * w,
                    bytes_per_image: 4 * w * (end - row),
                };
                self.plane_words_copy(&staging, texture, &region, &layout, true)?;
                row = end;
            }
            self.core().retain(staging);
        }
        Ok(())
    }

    /// A `Shared` buffer of `length` bytes of `unit` repeated, the source of a fill's copies.
    fn staging(&mut self, unit: &[u8], length: u64) -> Result<Buffer, Error> {
        let bytes: Vec<u8> = unit.iter().copied().cycle().take(length as usize).collect();
        self.staged(&bytes)
    }

    /// A `Shared` buffer of `bytes`, written through its mapping (seen by the device at the
    /// commit's submission).
    fn staged(&mut self, bytes: &[u8]) -> Result<Buffer, Error> {
        let buffer = self
            .core()
            .device()
            .write_combined_buffer(bytes.len() as u64)?;
        let Some(contents) = buffer.contents() else {
            return Err(Error::Vulkan(super::error::VkResultCode(
                vk::Result::ERROR_MEMORY_MAP_FAILED.as_raw(),
            )));
        };
        // SAFETY: the mapping covers the buffer's `bytes.len()` bytes, which nothing else uses
        // yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        Ok(buffer)
    }
}
