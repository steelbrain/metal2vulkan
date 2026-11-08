//! Metal's private pixel format description, `MTLPixelFormatGetInfoForDevice` (Apple SPI).
//!
//! `MTLPixelFormatGetInfoForDevice(device, pixelFormat)` returns Metal's `MTLPixelFormatInfo`, a
//! 0x38-byte struct: the format's name (a C string), a flags word, a cast class, and four words
//! that hold the format's block (for a compressed format: its block's bytes, width, height and
//! depth). Private Metal is Metal: a program that asks it gets Metal's answer.
//!
//! The answers are our own measurements on an Apple M3 Max at the release every device answers as
//! (`metal_release.rs`; every raw value from 0 to 2047, plain Metal): 264 values described. An
//! Apple M4 at that release answers the M3 Max's every one of those, and also describes the 16
//! values from 630 to 645, which the runtime does not (`docs/FAST-PATHS.md`, known gaps). A value
//! the table does not describe answers [`PixelFormatInfo::INVALID`], as Metal answers it:
//! `MTLPixelFormatInvalid`, every word 0.
//!
//! The answer is Metal's description of the format on an Apple GPU, which every device answers
//! as. Its flags are Apple's, whose bits no document names: they state what Metal does with the
//! format on that GPU, not what this Vulkan device does. The device's own answers are its typed
//! queries ([`Device::pixel_format_capabilities`], the creators' refusals).
//!
//! `describes` reads it: Metal describes a format exactly where the table has a row for it.
//!
//! `MTLGetTextureLevelInfoForDeviceWithOptions`, Metal's layout of one level of a texture, is
//! computed from the description ([`Device::texture_level_info`]).

use crate::Device;
use metal2vulkan::metal::PixelFormat;
use PixelFormat as P;

/// What `MTLPixelFormatGetInfoForDevice` answers for a raw pixel format value: Metal's
/// `MTLPixelFormatInfo` (Apple SPI), field for field.
///
/// Metal's struct holds a union from its fourth word on; [`words`](Self::words) keeps its four
/// words raw, as Metal returns them. For a format of blocks (flags bit `0x400`: the compressed
/// formats, and `GBGR422` and `BGRG422`, whose blocks are 2 by 1) they are its block's bytes,
/// width, height and depth; for another format they are the texel's bytes, a packed word, and two
/// more words Metal does not name.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelFormatInfo {
    /// `name`: Metal's enumerator name of the format, such as `MTLPixelFormatBGRA8Unorm`;
    /// `MTLPixelFormatInvalid` where Metal describes no format.
    pub name: &'static str,
    /// `flags`: Apple's flags word.
    pub flags: u64,
    /// `castClass`: the class of formats a view may reinterpret the format as.
    pub cast_class: u64,
    /// The four words from offset 0x18.
    pub words: [u64; 4],
}

impl PixelFormatInfo {
    /// Metal's answer for a value it describes no format for: `MTLPixelFormatInvalid`, every word
    /// 0.
    pub const INVALID: PixelFormatInfo = PixelFormatInfo {
        name: "MTLPixelFormatInvalid",
        flags: 0,
        cast_class: 0,
        words: [0; 4],
    };

    /// Whether Metal describes a format here (any answer but [`INVALID`](Self::INVALID)).
    pub fn describes_a_format(&self) -> bool {
        *self != PixelFormatInfo::INVALID
    }

    /// Whether Metal describes a depth or a stencil format here ([`DEPTH_OR_STENCIL`]).
    pub(crate) fn depth_or_stencil(&self) -> bool {
        self.flags & DEPTH_OR_STENCIL != 0
    }
}

/// `MTLPixelFormatGetInfoForDevice(device, raw)` as Metal answers it on its Apple GPU (the module
/// docs): the measured row for `raw`, else [`PixelFormatInfo::INVALID`].
pub(crate) fn pixel_format_info(raw: u64) -> PixelFormatInfo {
    let Ok(index) = DESCRIBED.binary_search_by_key(&raw, |row| u64::from(row.format.as_raw()))
    else {
        return PixelFormatInfo::INVALID;
    };
    let row = &DESCRIBED[index];
    PixelFormatInfo {
        name: row.format.name(),
        flags: u64::from(row.flags),
        cast_class: u64::from(row.cast_class),
        words: row.words.map(u64::from),
    }
}

/// Whether Metal describes `pf`: `MTLPixelFormatGetInfoForDevice` answers a format for it
/// ([`pixel_format_info`], the measured table, which holds no row for `Invalid`). Plain Metal ends
/// its process over every texture of a format it describes none of, and its alignment queries
/// assert on one, which the runtime answers alike (`Invalid`, `TextureProblem::NotAFormat` and
/// `BufferTextureProblem::NotLinear`), and so does its render pipeline descriptor's validation for
/// an attachment of one ("is not a valid MTLPixelFormat", `Problem::AttachmentNotAFormat`). An
/// Apple GPU, whose Metal every device answers as, describes neither `Depth24Unorm_Stencil8` nor
/// `X24_Stencil8` (an M4 stops a texture, a depth and a stencil attachment of each as of no
/// format; our probe and the consumer's M3 run), so `isDepth24Stencil8PixelFormatSupported`
/// answers no.
pub(crate) fn describes(pf: PixelFormat) -> bool {
    pixel_format_info(u64::from(pf.as_raw())).describes_a_format()
}

impl Device {
    /// `MTLPixelFormatGetInfoForDevice(device, raw)` (Apple SPI): Metal's description of the raw
    /// pixel format value `raw` on an Apple GPU at the release every device answers as (the
    /// module docs). Any value is taken, as Metal takes any: one it describes no format for
    /// answers [`PixelFormatInfo::INVALID`]. The flags are those of Apple's GPU, not this
    /// device's.
    pub fn pixel_format_info(&self, raw: u64) -> PixelFormatInfo {
        self.current().pixel_format_info_now(raw)
    }

    /// [`Device::pixel_format_info`] on the generation that serves (`device/line.rs`).
    pub(crate) fn pixel_format_info_now(&self, raw: u64) -> PixelFormatInfo {
        pixel_format_info(raw)
    }

    /// `MTLGetTextureLevelInfoForDeviceWithOptions(device, ...)` (Apple SPI): the layout Metal
    /// gives one level of a texture on an Apple GPU, computed from the format's description
    /// ([`Device::pixel_format_info`]) by the rule our probes fit with no exception, alike on an
    /// Apple M3 Max under macOS 26.7 and an Apple M4 under 27.0.1 (on each: every described
    /// format over 20 sizes; 16 formats over levels 0 to 17, 31, 32, 63, 64, 1000 and 2^32 by
    /// six plane values, eight shapes each; every value `PixelFormat` names by plane values 0 to
    /// 7 at four shapes; values no format has; zero sizes; 34, 306 and 34, 626 answers):
    ///
    /// - the level's size is the texture's shifted right by the level's low 6 bits (Metal shifts
    ///   by the level modulo 64), each at least 1 (a 0 is 1);
    /// - a block is the block of a format whose flags hold bit `0x400` (the description's second
    ///   and third words), else one texel; the blocks cover the size rounded up, as
    ///   `(extent + block - 1) / block` with an add that wraps (a 4-wide block's count at a width
    ///   of `u64::MAX` is 0);
    /// - a block's bytes are the description's first word (0 for a value it describes no format
    ///   of), but where the depth-and-stencil formats answer by the plane argument (`planes`:
    ///   `Depth32Float_Stencil8` and `Depth24Unorm_Stencil8` 0 for 0, 4 for 1, 1 for 2;
    ///   `X24_Stencil8` 1 for 0 and 2), times the array length;
    /// - a row is the blocks across times a block's bytes, an image the rows, the level the
    ///   images times its depth, and the size the blocks cover the blocks times the block, all
    ///   in 64-bit arithmetic that wraps.
    pub fn texture_level_info(&self, query: &TextureLevelQuery) -> TextureLevelInfo {
        self.current().texture_level_info_now(query)
    }

    /// [`Device::texture_level_info`] on the generation that serves (`device/line.rs`).
    pub(crate) fn texture_level_info_now(&self, query: &TextureLevelQuery) -> TextureLevelInfo {
        texture_level_info(query)
    }
}

/// The arguments of `MTLGetTextureLevelInfoForDeviceWithOptions` after the device (Apple SPI), in
/// Metal's order. Metal's argument list, so it does not grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureLevelQuery {
    /// The raw pixel format value (any value, as for [`Device::pixel_format_info`]).
    pub pixel_format: u64,
    /// The texture's width.
    pub width: u64,
    /// The texture's height.
    pub height: u64,
    /// The texture's depth.
    pub depth: u64,
    /// The texture's array length.
    pub array_length: u64,
    /// The mipmap level (the sixth argument).
    pub level: u64,
    /// Which plane of a depth-and-stencil format (the seventh argument): 1 its depth, 2 its
    /// stencil ([`Device::texture_level_info`] says how each format answers it).
    pub planes: u64,
}

impl TextureLevelQuery {
    /// A query of level 0 of a 2D texture of `width` by `height`, depth 1, array length 1, plane
    /// argument 0.
    pub fn plane_2d(pixel_format: u64, width: u64, height: u64) -> TextureLevelQuery {
        TextureLevelQuery {
            pixel_format,
            width,
            height,
            depth: 1,
            array_length: 1,
            level: 0,
            planes: 0,
        }
    }
}

/// What `MTLGetTextureLevelInfoForDeviceWithOptions` answers: Metal's `MTLTextureLevelInfo`, 13
/// words (0x68 bytes) that lay out one level ([`words`](Self::words) gives them in Metal's
/// order). Fields are named by what they hold in our probes; Metal names none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureLevelInfo {
    /// The level's width (word 0): the texture's shifted right by the level, at least 1.
    pub width: u64,
    /// The level's height (word 1).
    pub height: u64,
    /// The level's depth (word 2).
    pub depth: u64,
    /// The level's blocks across (word 3): its width over the block's, rounded up (a compressed
    /// format's block; one texel otherwise).
    pub blocks_across: u64,
    /// The level's blocks down (word 4).
    pub blocks_down: u64,
    /// The level's blocks deep (word 5): its depth.
    pub blocks_deep: u64,
    /// The width the blocks cover (word 6).
    pub aligned_width: u64,
    /// The height the blocks cover (word 7).
    pub aligned_height: u64,
    /// The depth the blocks cover (word 8).
    pub aligned_depth: u64,
    /// A block's bytes times the array length (word 9).
    pub bytes_per_block: u64,
    /// A row of blocks' bytes (word 10).
    pub bytes_per_row: u64,
    /// One image's bytes (word 11): the rows of blocks.
    pub bytes_per_image: u64,
    /// The level's bytes (word 12): its images, one a depth slice.
    pub bytes: u64,
}

impl TextureLevelInfo {
    /// The 13 words in Metal's order.
    pub fn words(&self) -> [u64; 13] {
        [
            self.width,
            self.height,
            self.depth,
            self.blocks_across,
            self.blocks_down,
            self.blocks_deep,
            self.aligned_width,
            self.aligned_height,
            self.aligned_depth,
            self.bytes_per_block,
            self.bytes_per_row,
            self.bytes_per_image,
            self.bytes,
        ]
    }
}

/// A block's bytes where Metal answers them by the plane argument rather than by the format's
/// description (`described`, its first word): measured alike on an Apple M3 Max under macOS 26.7
/// and an Apple M4 under 27.0.1 (plane values 0 to 12, 16, 256, 2^31 and all ones; every value
/// `PixelFormat` names by 0 to 7). The combined formats answer their depth plane (a
/// 32-bit float, 4) for 1 and their stencil plane (1) for 2, and nothing for 0;
/// `X24_Stencil8` answers its stencil byte for 0 and 2. Every other plane value, and every other
/// format whatever the value (`X32_Stencil8`, `Depth32Float` and `Stencil8` among them), answers
/// the description's bytes, which are 0 for the 24-bit formats an Apple GPU describes none of.
fn plane_bytes(format: Option<PixelFormat>, planes: u64, described: u64) -> u64 {
    let combined = matches!(
        format,
        Some(P::Depth32FloatStencil8 | P::Depth24UnormStencil8)
    );
    match (planes, combined, format == Some(P::X24Stencil8)) {
        (0, true, _) => 0,
        (1, true, _) => 4,
        (2, true, _) | (0 | 2, _, true) => 1,
        _ => described,
    }
}

/// `MTLGetTextureLevelInfoForDeviceWithOptions` as Metal answers it on its Apple GPU
/// ([`Device::texture_level_info`] says how).
pub(crate) fn texture_level_info(query: &TextureLevelQuery) -> TextureLevelInfo {
    let info = pixel_format_info(query.pixel_format);
    let shift = query.level & 63;
    let at_level = |extent: u64| (extent >> shift).max(1);
    let (width, height, depth) = (
        at_level(query.width),
        at_level(query.height),
        at_level(query.depth),
    );
    let (block_width, block_height) = if info.flags & BLOCKS != 0 {
        (info.words[1].max(1), info.words[2].max(1))
    } else {
        (1, 1)
    };
    let format = u32::try_from(query.pixel_format)
        .ok()
        .and_then(|raw| PixelFormat::try_from(raw).ok());
    let block_bytes = plane_bytes(format, query.planes, info.words[0]);
    let blocks_across = blocks(width, block_width);
    let blocks_down = blocks(height, block_height);
    let bytes_per_block = block_bytes.wrapping_mul(query.array_length);
    let bytes_per_row = blocks_across.wrapping_mul(bytes_per_block);
    let bytes_per_image = bytes_per_row.wrapping_mul(blocks_down);
    TextureLevelInfo {
        width,
        height,
        depth,
        blocks_across,
        blocks_down,
        blocks_deep: depth,
        aligned_width: blocks_across.wrapping_mul(block_width),
        aligned_height: blocks_down.wrapping_mul(block_height),
        aligned_depth: depth,
        bytes_per_block,
        bytes_per_row,
        bytes_per_image,
        bytes: bytes_per_image.wrapping_mul(depth),
    }
}

/// The blocks of `block` texels (at least 1) that cover `extent`, as Metal counts them, with an
/// add that wraps: `(extent + block - 1) / block`. An extent past `2^64 - block` wraps to less than
/// `block`, so it has no blocks: a width of `u64::MAX` has 0 blocks of 4, and 0 of 2.
fn blocks(extent: u64, block: u64) -> u64 {
    extent.wrapping_add(block - 1) / block
}

/// The flags bit Metal sets on exactly the formats of blocks in the table (every BC, ETC2, EAC,
/// ASTC and PVRTC format, and the 4:2:2 formats `GBGR422` and `BGRG422`, whose blocks are 2 by 1),
/// whose description's words are then the block's.
const BLOCKS: u64 = 0x400;

/// The flags bits Metal sets on exactly its depth (`0x20`) and stencil (`0x40`) formats in the
/// table: `Depth16Unorm`, `Depth32Float`, `Stencil8`, `Depth32Float_Stencil8`, `X32_Stencil8`,
/// `AnyDepth` and `AnyStencil`.
const DEPTH_OR_STENCIL: u64 = 0x60;

/// One format's measured description: its four words, its flags and its cast class.
struct Described {
    format: PixelFormat,
    words: [u16; 4],
    flags: u32,
    cast_class: u8,
}

/// A row of [`DESCRIBED`].
const fn d(format: PixelFormat, words: [u16; 4], flags: u32, cast_class: u8) -> Described {
    Described {
        format,
        words,
        flags,
        cast_class,
    }
}

/// Every format Metal describes, by raw value (an Apple M3 Max at macOS 27.0.1, build 26A434,
/// whose answers are an Apple M4's but for the 16 values from 630 to 645), each as
/// `d(format, words, flags, castClass)`. Sorted by raw value (`pixel_format_info` searches it).
#[rustfmt::skip]
const DESCRIBED: &[Described] = &[
    d(P::A8Unorm, [1, 257, 1, 1], 0x0403019f, 1),
    d(P::R8Unorm, [1, 257, 1, 1], 0x0703019f, 1),
    d(P::R8UnormSRGB, [1, 257, 1, 1], 0x0703099f, 1),
    d(P::R8Snorm, [1, 257, 1, 1], 0x0403019f, 1),
    d(P::R8Uint, [1, 257, 1, 1], 0x04280113, 1),
    d(P::R8Sint, [1, 257, 1, 1], 0x04140113, 1),
    d(P::R16Unorm, [2, 513, 2, 2], 0x0703019f, 2),
    d(P::R16Snorm, [2, 513, 2, 2], 0x0703019f, 2),
    d(P::R16Uint, [2, 513, 2, 2], 0x07280113, 2),
    d(P::R16Sint, [2, 513, 2, 2], 0x07140113, 2),
    d(P::R16Float, [2, 513, 2, 2], 0x0703019f, 2),
    d(P::R16Bfloat, [2, 513, 2, 2], 0x00030013, 2),
    d(P::RG8Unorm, [2, 258, 2, 2], 0x0503019f, 2),
    d(P::RG8UnormSRGB, [2, 258, 2, 2], 0x0503099f, 2),
    d(P::RG8Snorm, [2, 258, 2, 2], 0x0503019f, 2),
    d(P::RG8Uint, [2, 258, 2, 2], 0x05280113, 2),
    d(P::RG8Sint, [2, 258, 2, 2], 0x05140113, 2),
    d(P::B5G6R5Unorm, [2, 515, 6, 6], 0x0003019d, 2),
    d(P::A1BGR5Unorm, [2, 516, 8, 8], 0x0003019d, 2),
    d(P::ABGR4Unorm, [2, 516, 8, 8], 0x0003019d, 2),
    d(P::BGR5A1Unorm, [2, 516, 8, 8], 0x0003019d, 2),
    d(P::BGRA4Unorm, [2, 516, 8, 8], 0x0003019d, 2),
    d(P::R32Uint, [4, 1025, 4, 4], 0x04080113, 4),
    d(P::R32Sint, [4, 1025, 4, 4], 0x04040113, 4),
    d(P::R32Float, [4, 1025, 4, 4], 0x0403019f, 4),
    d(P::RG16Unorm, [4, 514, 4, 4], 0x0703019f, 4),
    d(P::RG16Snorm, [4, 514, 4, 4], 0x0703019f, 4),
    d(P::RG16Uint, [4, 514, 4, 4], 0x07280113, 4),
    d(P::RG16Sint, [4, 514, 4, 4], 0x07140113, 4),
    d(P::RG16Float, [4, 514, 4, 4], 0x0703019f, 4),
    d(P::RG16Bfloat, [4, 514, 4, 4], 0x00030013, 4),
    d(P::RGBA8Unorm, [4, 260, 4, 4], 0x0703019f, 4),
    d(P::RGBA8UnormSRGB, [4, 260, 4, 4], 0x0703099f, 4),
    d(P::RGBA8Snorm, [4, 260, 4, 4], 0x0703019f, 4),
    d(P::RGBA8Uint, [4, 260, 4, 4], 0x07280113, 4),
    d(P::RGBA8Sint, [4, 260, 4, 4], 0x07140113, 4),
    d(P::BGRA8Unorm, [4, 260, 4, 4], 0x0703019f, 4),
    d(P::BGRA8UnormSRGB, [4, 260, 4, 4], 0x0703099f, 4),
    d(P::ARGB8Unorm, [4, 260, 4, 4], 0x0003019f, 4),
    d(P::ARGB8UnormSRGB, [4, 260, 4, 4], 0x0003099f, 4),
    d(P::RGB10A2Unorm, [4, 1028, 4, 4], 0x0703019f, 4),
    d(P::RGB10A2Uint, [4, 516, 8, 8], 0x07280113, 4),
    d(P::RG11B10Float, [4, 1027, 4, 4], 0x0003019f, 4),
    d(P::RGB9E5Float, [4, 1027, 4, 4], 0x0003819f, 4),
    d(P::BGR10A2Unorm, [4, 1028, 4, 4], 0x0703019f, 4),
    d(P::RG32Uint, [8, 1026, 8, 8], 0x24080113, 6),
    d(P::RG32Sint, [8, 1026, 8, 8], 0x04040113, 6),
    d(P::RG32Float, [8, 1026, 8, 8], 0x0403019f, 6),
    d(P::RGBA16Unorm, [8, 516, 8, 8], 0x0703019f, 6),
    d(P::RGBA16Snorm, [8, 516, 8, 8], 0x0703019f, 6),
    d(P::RGBA16Uint, [8, 516, 8, 8], 0x07280113, 6),
    d(P::RGBA16Sint, [8, 516, 8, 8], 0x07140113, 6),
    d(P::RGBA16Float, [8, 516, 8, 8], 0x0703019f, 6),
    d(P::RGBA16Bfloat, [8, 516, 8, 8], 0x00030013, 6),
    d(P::RGBA32Uint, [16, 1028, 16, 16], 0x04080113, 8),
    d(P::RGBA32Sint, [16, 1028, 16, 16], 0x04040113, 8),
    d(P::RGBA32Float, [16, 1028, 16, 16], 0x0403019f, 8),
    d(P::BC1RGBA, [8, 4, 4, 1], 0x00430405, 9),
    d(P::BC1RGBASRGB, [8, 4, 4, 1], 0x00430c05, 9),
    d(P::BC2RGBA, [16, 4, 4, 1], 0x00430405, 10),
    d(P::BC2RGBASRGB, [16, 4, 4, 1], 0x00430c05, 10),
    d(P::BC3RGBA, [16, 4, 4, 1], 0x00430405, 11),
    d(P::BC3RGBASRGB, [16, 4, 4, 1], 0x00430c05, 11),
    d(P::BC4RUnorm, [8, 4, 4, 1], 0x00430405, 12),
    d(P::BC4RSnorm, [8, 4, 4, 1], 0x00430405, 12),
    d(P::BC5RGUnorm, [16, 4, 4, 1], 0x00430405, 13),
    d(P::BC5RGSnorm, [16, 4, 4, 1], 0x00430405, 13),
    d(P::BC6HRGBFloat, [16, 4, 4, 1], 0x00430405, 14),
    d(P::BC6HRGBUfloat, [16, 4, 4, 1], 0x00430405, 14),
    d(P::BC7RGBAUnorm, [16, 4, 4, 1], 0x00430405, 15),
    d(P::BC7RGBAUnormSRGB, [16, 4, 4, 1], 0x00430c05, 15),
    d(P::PVRTCRGB2BPP, [8, 8, 4, 1], 0x00032405, 16),
    d(P::PVRTCRGB2BPPSRGB, [8, 8, 4, 1], 0x00032c05, 16),
    d(P::PVRTCRGB4BPP, [8, 4, 4, 1], 0x00032405, 17),
    d(P::PVRTCRGB4BPPSRGB, [8, 4, 4, 1], 0x00032c05, 17),
    d(P::PVRTCRGBA2BPP, [8, 8, 4, 1], 0x00032405, 18),
    d(P::PVRTCRGBA2BPPSRGB, [8, 8, 4, 1], 0x00032c05, 18),
    d(P::PVRTCRGBA4BPP, [8, 4, 4, 1], 0x00032405, 19),
    d(P::PVRTCRGBA4BPPSRGB, [8, 4, 4, 1], 0x00032c05, 19),
    d(P::EACR11Unorm, [8, 4, 4, 1], 0x00030405, 20),
    d(P::EACR11Snorm, [8, 4, 4, 1], 0x00030405, 20),
    d(P::EACRG11Unorm, [16, 4, 4, 1], 0x00030405, 21),
    d(P::EACRG11Snorm, [16, 4, 4, 1], 0x00030405, 21),
    d(P::EACRGBA8, [16, 4, 4, 1], 0x00030405, 22),
    d(P::EACRGBA8SRGB, [16, 4, 4, 1], 0x00030c05, 22),
    d(P::ETC2RGB8, [8, 4, 4, 1], 0x00030405, 23),
    d(P::ETC2RGB8SRGB, [8, 4, 4, 1], 0x00030c05, 23),
    d(P::ETC2RGB8A1, [8, 4, 4, 1], 0x00030405, 24),
    d(P::ETC2RGB8A1SRGB, [8, 4, 4, 1], 0x00030c05, 24),
    d(P::ASTC4x2SRGB, [16, 4, 2, 1], 0x00031c05, 25),
    d(P::ASTC4x4SRGB, [16, 4, 4, 1], 0x00031c05, 26),
    d(P::ASTC5x4SRGB, [16, 5, 4, 1], 0x00031c05, 27),
    d(P::ASTC5x5SRGB, [16, 5, 5, 1], 0x00031c05, 28),
    d(P::ASTC6x5SRGB, [16, 6, 5, 1], 0x00031c05, 29),
    d(P::ASTC6x6SRGB, [16, 6, 6, 1], 0x00031c05, 30),
    d(P::ASTC8x4SRGB, [16, 8, 4, 1], 0x00031c05, 31),
    d(P::ASTC8x5SRGB, [16, 8, 5, 1], 0x00031c05, 32),
    d(P::ASTC8x6SRGB, [16, 8, 6, 1], 0x00031c05, 33),
    d(P::ASTC8x8SRGB, [16, 8, 8, 1], 0x00031c05, 34),
    d(P::ASTC10x5SRGB, [16, 10, 5, 1], 0x00031c05, 35),
    d(P::ASTC10x6SRGB, [16, 10, 6, 1], 0x00031c05, 36),
    d(P::ASTC10x8SRGB, [16, 10, 8, 1], 0x00031c05, 37),
    d(P::ASTC10x10SRGB, [16, 10, 10, 1], 0x00031c05, 38),
    d(P::ASTC12x10SRGB, [16, 12, 10, 1], 0x00031c05, 39),
    d(P::ASTC12x12SRGB, [16, 12, 12, 1], 0x00031c05, 40),
    d(P::ASTC4x2LDR, [16, 4, 2, 1], 0x00031405, 25),
    d(P::ASTC4x4LDR, [16, 4, 4, 1], 0x00031405, 26),
    d(P::ASTC5x4LDR, [16, 5, 4, 1], 0x00031405, 27),
    d(P::ASTC5x5LDR, [16, 5, 5, 1], 0x00031405, 28),
    d(P::ASTC6x5LDR, [16, 6, 5, 1], 0x00031405, 29),
    d(P::ASTC6x6LDR, [16, 6, 6, 1], 0x00031405, 30),
    d(P::ASTC8x4LDR, [16, 8, 4, 1], 0x00031405, 31),
    d(P::ASTC8x5LDR, [16, 8, 5, 1], 0x00031405, 32),
    d(P::ASTC8x6LDR, [16, 8, 6, 1], 0x00031405, 33),
    d(P::ASTC8x8LDR, [16, 8, 8, 1], 0x00031405, 34),
    d(P::ASTC10x5LDR, [16, 10, 5, 1], 0x00031405, 35),
    d(P::ASTC10x6LDR, [16, 10, 6, 1], 0x00031405, 36),
    d(P::ASTC10x8LDR, [16, 10, 8, 1], 0x00031405, 37),
    d(P::ASTC10x10LDR, [16, 10, 10, 1], 0x00031405, 38),
    d(P::ASTC12x10LDR, [16, 12, 10, 1], 0x00031405, 39),
    d(P::ASTC12x12LDR, [16, 12, 12, 1], 0x00031405, 40),
    d(P::ASTC4x2HDR, [16, 4, 2, 1], 0x00031405, 25),
    d(P::ASTC4x4HDR, [16, 4, 4, 1], 0x00031405, 26),
    d(P::ASTC5x4HDR, [16, 5, 4, 1], 0x00031405, 27),
    d(P::ASTC5x5HDR, [16, 5, 5, 1], 0x00031405, 28),
    d(P::ASTC6x5HDR, [16, 6, 5, 1], 0x00031405, 29),
    d(P::ASTC6x6HDR, [16, 6, 6, 1], 0x00031405, 30),
    d(P::ASTC8x4HDR, [16, 8, 4, 1], 0x00031405, 31),
    d(P::ASTC8x5HDR, [16, 8, 5, 1], 0x00031405, 32),
    d(P::ASTC8x6HDR, [16, 8, 6, 1], 0x00031405, 33),
    d(P::ASTC8x8HDR, [16, 8, 8, 1], 0x00031405, 34),
    d(P::ASTC10x5HDR, [16, 10, 5, 1], 0x00031405, 35),
    d(P::ASTC10x6HDR, [16, 10, 6, 1], 0x00031405, 36),
    d(P::ASTC10x8HDR, [16, 10, 8, 1], 0x00031405, 37),
    d(P::ASTC10x10HDR, [16, 10, 10, 1], 0x00031405, 38),
    d(P::ASTC12x10HDR, [16, 12, 10, 1], 0x00031405, 39),
    d(P::ASTC12x12HDR, [16, 12, 12, 1], 0x00031405, 40),
    d(P::GBGR422, [4, 2, 1, 1], 0x00034405, 0),
    d(P::BGRG422, [4, 2, 1, 1], 0x00034405, 0),
    d(P::Depth16Unorm, [2, 513, 2, 2], 0x000301a5, 0),
    d(P::Depth32Float, [4, 1025, 4, 4], 0x000301a5, 0),
    d(P::Stencil8, [1, 257, 1, 1], 0x002801c1, 0),
    d(P::Depth32FloatStencil8, [8, 1026, 8, 8], 0x008303e5, 42),
    d(P::X32Stencil8, [1, 257, 1, 1], 0x00a80141, 42),
    d(P::Unspecialized, [4, 260, 4, 4], 0x203f0019, 4),
    d(P::RGBX8Unorm, [4, 260, 3, 3], 0x0003019f, 4),
    d(P::RGBX8UnormSRGB, [4, 260, 3, 3], 0x0003098f, 4),
    d(P::RGBX8Snorm, [4, 260, 3, 3], 0x0003018f, 4),
    d(P::RGBX8Uint, [4, 260, 3, 3], 0x00280103, 4),
    d(P::RGBX8Sint, [4, 260, 3, 3], 0x00140103, 4),
    d(P::RGBX16Uint, [8, 516, 6, 6], 0x00280103, 6),
    d(P::RGBX16Sint, [8, 516, 6, 6], 0x00140103, 6),
    d(P::RGBX16Float, [8, 516, 6, 6], 0x0003019f, 6),
    d(P::RGBX32Uint, [16, 1028, 12, 12], 0x00080103, 8),
    d(P::RGBX32Sint, [16, 1028, 12, 12], 0x00040103, 8),
    d(P::RGBX32Float, [16, 1028, 12, 12], 0x00030197, 8),
    d(P::YCBCR84202P, [1, 515, 6, 6], 0x0783001d, 46),
    d(P::YCBCR84221P, [1, 515, 6, 6], 0x0003001d, 47),
    d(P::YCBCR84222P, [1, 515, 6, 6], 0x0783001d, 48),
    d(P::YCBCR84442P, [1, 515, 6, 6], 0x0783001f, 49),
    d(P::YCBCR104441P, [2, 515, 6, 6], 0x0703001f, 50),
    d(P::YCBCR104202P, [2, 515, 6, 6], 0x0783001d, 51),
    d(P::YCBCR104222P, [2, 515, 6, 6], 0x0783001d, 52),
    d(P::YCBCR104442P, [2, 515, 6, 6], 0x0783001f, 53),
    d(P::YCBCR104202PPACKED, [1, 515, 8, 8], 0x07830015, 54),
    d(P::YCBCR104222PPACKED, [1, 515, 8, 8], 0x07830015, 55),
    d(P::YCBCR104442PPACKED, [1, 515, 8, 8], 0x07830017, 56),
    d(P::YCBCR84202PSRGB, [1, 515, 6, 6], 0x0783081d, 46),
    d(P::YCBCR84221PSRGB, [1, 515, 6, 6], 0x0003081d, 47),
    d(P::YCBCR84222PSRGB, [1, 515, 6, 6], 0x0783081d, 48),
    d(P::YCBCR84442PSRGB, [1, 515, 6, 6], 0x0783081f, 49),
    d(P::YCBCR104441PSRGB, [2, 515, 6, 6], 0x0703081f, 50),
    d(P::YCBCR104202PSRGB, [2, 515, 6, 6], 0x0783081d, 51),
    d(P::YCBCR104222PSRGB, [2, 515, 6, 6], 0x0783081d, 52),
    d(P::YCBCR104442PSRGB, [2, 515, 6, 6], 0x0783081f, 53),
    d(P::YCBCR104202PPACKEDSRGB, [1, 515, 8, 8], 0x07830815, 54),
    d(P::YCBCR104222PPACKEDSRGB, [1, 515, 8, 8], 0x07830815, 55),
    d(P::YCBCR104442PPACKEDSRGB, [1, 515, 8, 8], 0x07830815, 56),
    d(P::RGB84202P, [1, 515, 6, 6], 0x0783001d, 0),
    d(P::RGB84222P, [1, 515, 6, 6], 0x0783001d, 0),
    d(P::RGB84442P, [1, 515, 6, 6], 0x0783001f, 0),
    d(P::RGB104202P, [2, 515, 6, 6], 0x0783001d, 0),
    d(P::RGB104222P, [2, 515, 6, 6], 0x0783001d, 0),
    d(P::RGB104442P, [2, 515, 6, 6], 0x0783001f, 0),
    d(P::RGB104202PPACKED, [1, 515, 8, 8], 0x07830015, 0),
    d(P::RGB104222PPACKED, [1, 515, 8, 8], 0x07830015, 0),
    d(P::RGB104442PPACKED, [1, 515, 8, 8], 0x07830015, 0),
    d(P::RGB10A82PXR10, [2, 516, 8, 8], 0x0783001f, 45),
    d(P::RGB10A82PXR10SRGB, [2, 516, 8, 8], 0x0783081f, 45),
    d(P::BGRA10XR, [8, 516, 8, 8], 0x0703019f, 43),
    d(P::BGRA10XRSRGB, [8, 516, 8, 8], 0x0703099f, 43),
    d(P::BGR10XR, [4, 515, 6, 6], 0x0703019f, 44),
    d(P::BGR10XRSRGB, [4, 515, 6, 6], 0x0703099f, 44),
    d(P::RGBA16FloatXR, [8, 516, 8, 8], 0x0003019f, 6),
    d(P::YCBCRA84441P, [4, 516, 8, 8], 0x0703001f, 0),
    d(P::YCBCR84221PREV, [1, 515, 6, 6], 0x0703001d, 0),
    d(P::YCBCR104441PPQ, [2, 515, 6, 6], 0x0f03001f, 0),
    d(P::YCBCR104202PPQ, [2, 515, 6, 6], 0x0f83001d, 0),
    d(P::YCBCR104222PPQ, [2, 515, 6, 6], 0x0f83001d, 0),
    d(P::YCBCR104442PPQ, [2, 515, 6, 6], 0x0f83001f, 0),
    d(P::YCBCR104202PPACKEDPQ, [1, 515, 6, 6], 0x0f830015, 0),
    d(P::YCBCR104222PPACKEDPQ, [1, 515, 6, 6], 0x0f830015, 0),
    d(P::YCBCR104442PPACKEDPQ, [1, 515, 6, 6], 0x0f830015, 0),
    d(P::YCBCR124202P, [2, 515, 6, 6], 0x0783001d, 0),
    d(P::YCBCR124222P, [2, 515, 6, 6], 0x0783001d, 0),
    d(P::YCBCR124442P, [2, 515, 6, 6], 0x0783001f, 0),
    d(P::YCBCR124202PPQ, [2, 515, 6, 6], 0x0f83001d, 0),
    d(P::YCBCR124222PPQ, [2, 515, 6, 6], 0x0f83001d, 0),
    d(P::YCBCR124442PPQ, [2, 515, 6, 6], 0x0f83001f, 0),
    d(P::R10UnormX6, [2, 513, 2, 2], 0x0703001f, 57),
    d(P::R10UnormX6SRGB, [2, 513, 2, 2], 0x0703081f, 57),
    d(P::RG10UnormX12, [4, 514, 4, 4], 0x0303001f, 58),
    d(P::RG10UnormX12SRGB, [4, 514, 4, 4], 0x0303081f, 58),
    d(P::YCBCR124202PPACKED, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124222PPACKED, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124442PPACKED, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124202PPACKEDPQ, [2, 259, 2, 2], 0x0f83001d, 0),
    d(P::YCBCR124222PPACKEDPQ, [2, 259, 2, 2], 0x0f83001d, 0),
    d(P::YCBCR124442PPACKEDPQ, [2, 259, 2, 2], 0x0f83001d, 0),
    d(P::RGB10A2UnormSRGB, [4, 516, 8, 8], 0x0703099d, 4),
    d(P::RGB10A2UnormPQ, [4, 516, 8, 8], 0x0f03019d, 4),
    d(P::R10UnormPACKED, [1, 513, 2, 2], 0x0703001f, 0),
    d(P::RG10UnormPACKED, [2, 514, 4, 4], 0x0303001f, 0),
    d(P::YCBCR104441PXR, [2, 259, 2, 2], 0x0703001f, 0),
    d(P::YCBCR104202PXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR104222PXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR104442PXR, [2, 259, 2, 2], 0x0783001f, 0),
    d(P::YCBCR104202PPACKEDXR, [1, 259, 1, 1], 0x07830015, 0),
    d(P::YCBCR104222PPACKEDXR, [1, 259, 1, 1], 0x07830015, 0),
    d(P::YCBCR104442PPACKEDXR, [1, 259, 1, 1], 0x07830015, 0),
    d(P::YCBCR124202PXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124222PXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124442PXR, [2, 259, 2, 2], 0x0783001f, 0),
    d(P::YCBCR124202PPACKEDXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124222PPACKEDXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::YCBCR124442PPACKEDXR, [2, 259, 2, 2], 0x0783001d, 0),
    d(P::R12UnormX4, [2, 513, 2, 2], 0x0703001d, 59),
    d(P::R12UnormX4PQ, [2, 513, 2, 2], 0x0f03001d, 59),
    d(P::RG12UnormX8, [4, 514, 4, 4], 0x0703001d, 60),
    d(P::R10UnormX6PQ, [2, 513, 2, 2], 0x0f03001d, 57),
    d(P::YCBCR104441PHLG, [2, 515, 6, 6], 0x1703001f, 0),
    d(P::YCBCR104202PHLG, [2, 515, 6, 6], 0x1783001d, 0),
    d(P::YCBCR104222PHLG, [2, 515, 6, 6], 0x1783001d, 0),
    d(P::YCBCR104442PHLG, [2, 515, 6, 6], 0x1783001f, 0),
    d(P::YCBCR104202PPACKEDHLG, [1, 515, 6, 6], 0x17830015, 0),
    d(P::YCBCR104222PPACKEDHLG, [1, 515, 6, 6], 0x17830015, 0),
    d(P::YCBCR104442PPACKEDHLG, [1, 515, 6, 6], 0x17830015, 0),
    d(P::YCBCR124202PHLG, [2, 515, 6, 6], 0x1783001d, 0),
    d(P::YCBCR124222PHLG, [2, 515, 6, 6], 0x1783001d, 0),
    d(P::YCBCR124442PHLG, [2, 515, 6, 6], 0x1783001f, 0),
    d(P::YCBCR124202PPACKEDHLG, [2, 259, 2, 2], 0x1783001d, 0),
    d(P::YCBCR124222PPACKEDHLG, [2, 259, 2, 2], 0x1783001d, 0),
    d(P::YCBCR124442PPACKEDHLG, [2, 259, 2, 2], 0x1783001d, 0),
    d(P::R10UnormX6HLG, [2, 513, 2, 2], 0x1703001d, 0),
    d(P::R12UnormX4HLG, [2, 513, 2, 2], 0x1703001d, 0),
    d(P::RGB10A2UnormHLG, [4, 516, 8, 8], 0x1703019d, 4),
    d(P::RGB10A82P, [2, 516, 8, 8], 0x0783001f, 45),
    d(P::RGB10A82PSRGB, [2, 516, 8, 8], 0x0783081f, 45),
    d(P::RGB10A82PPQ, [2, 516, 8, 8], 0x0f83001f, 45),
    d(P::RGB10A82PHLG, [2, 516, 8, 8], 0x1783001f, 45),
    d(P::BGR10A2UnormSRGB, [4, 516, 8, 8], 0x0703099d, 4),
    d(P::BGR10A2UnormPQ, [4, 516, 8, 8], 0x0f03019d, 4),
    d(P::AnyDepth, [4, 1025, 4, 4], 0x00000021, 0),
    d(P::AnyStencil, [1, 257, 1, 1], 0x00000041, 0),
];
