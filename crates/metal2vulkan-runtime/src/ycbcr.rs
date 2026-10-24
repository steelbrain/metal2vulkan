//! Two-plane YCbCr textures over an IOSurface's planes: `YCBCR8_420_2P` (format 500), the texture
//! CoreVideo's `420v` and `420f` surfaces are sampled through, `RGB8_420_2P` (540), the 4:2:2 and
//! 4:4:4 formats `YCBCR8_422_2P`, `YCBCR8_444_2P`, `RGB8_422_2P` and `RGB8_444_2P` (502, 503,
//! 541, 542), `YCBCR10_420_2P` and `RGB10_420_2P` (505 and 543), `YCBCR10_422_2P`,
//! `YCBCR10_444_2P`, `RGB10_422_2P` and `RGB10_444_2P` (506, 507, 544, 545), the packed
//! `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED` and `YCBCR10_444_2P_PACKED` (508, 509, 510),
//! and the `_sRGB` twins of the converted formats (520, 522, 523, 525, 526, 527, 528, 529, 530),
//! each over any plane of a surface as **Surfaces** below says, which Metal makes only over a
//! surface (`newTextureWithDescriptor:iosurface:plane:`; plain creation answers nil,
//! `TextureProblem::NoTextureOfFormat`), or as a view of its converted format.
//!
//! - **What Metal reads** (an Apple M4 under macOS 27.0.1; for 4:2:2 and 4:4:4): each texel from
//!   the luma code at its place and the chroma pairs around it at the format's subsampling: 4:2:0
//!   9/3/3/1 over the four nearest pairs at the middle of each 2x2 block, 4:2:2 3/1 over the two
//!   nearest across, its row's down, 4:4:4 its own pair (the chroma plane of a surface of finer
//!   chroma read as the format's at its rows), each code a byte, the high 10 bits of a 16-bit word,
//!   or one of three 10-bit codes of a 32-bit word, packed. The converted formats read the color
//!   converted by the surface's range (its OSType: `4..v` and `x4..` video, `4..f` and `xf..` full)
//!   and the matrix it held when the texture was made (`kIOSurfaceYCbCrMatrix`: BT.601 or none,
//!   BT.709, BT.2020 (measured too), and SMPTE 240M, read as BT.709): every channel a
//!   multiple of 1/2048, and alpha 1. The code formats (540 to 545) read the codes, red Cr, green
//!   Y, blue Cb, each `code / 1023` or `code / 255`, alpha 1 (for 540, for 541, 542, 544 and 545).
//!   The `_sRGB` twins read the converted sum through the sRGB curve, every channel a multiple of
//!   1/8192 (every one of the curve's 8193 steps), and a `half` read of it rounds with ties away
//!   from zero (`metal2vulkan::bind::ycbcr_read` states the arithmetic). `read()`, a nearest sample
//!   and a gather read those texels; a 4:4:4 texture samples repeat, mirrored repeat and
//!   mirror-clamp-to-edge as those modes, a 4:2:0 or 4:2:2 one as clamp-to-edge (for the code
//!   formats), and the borders read alpha 1, but for a texture read as codes, whose clamp-to-zero
//!   and clamp-to-border-color read as clamp-to-edge too.
//! - **The image.** The texture is an image of those texels, which holds every one exactly
//!   (`R16G16B16A16_SFLOAT` of the converted values, `R32G32B32A32_SFLOAT` of the sRGB twins',
//!   16 bytes a texel where a half holds no step of 1/8192 above 1/4,
//!   `A2B10G10R10_UNORM_PACK32` or `R8G8B8A8_UNORM` of the codes, written through an `R32_UINT`
//!   view), and a shadow
//!   of the surface's bytes (`shadow.rs`, `ShadowSource::Converted`): its copy in is a
//!   conversion, the planes' bytes brought into a staging buffer of the texture's (**When it
//!   converts** below) and one dispatch of the conversion kernel
//!   (`bind::ycbcr_convert_module`, `owned_kernel.rs`) writing every texel. The texture reads
//!   alpha 1 (`alpha_reads_one`) and maps those modes to clamp-to-edge (`address_mode_map`), so
//!   a nearest sample and a border read are Metal's exactly.
//! - **When it converts.** Before an encoder's first command that uses the texture, and again
//!   after a command of the encoder that writes the planes' bytes (through the buffer, a texture
//!   over the planes such as a decoder's `R8Unorm` and `RG8Unorm` ones, or a blit): a use of the
//!   bytes like any other, which the tracker orders after the writes of the same and earlier
//!   command buffers. The surface's bytes are the CPU's to write unseen, so the runtime cannot
//!   tell at recording whether they changed since the last conversion; the device tells. The
//!   copy in is the conversion's refresh (`bind::ycbcr_refresh_module`): one dispatch compares
//!   every word of the planes with the staging buffer's copy (the bytes the last conversion
//!   read), stores each word that differs into the copy, and sets the conversion's indirect
//!   arguments to its whole grid where a word differed or the last conversion did not complete
//!   (the staging buffer's completed word, cleared before the copy changes and set after the
//!   conversion). The conversion is dispatched through those arguments: unchanged planes run no
//!   workgroup of it, and changed ones convert every texel from the copy, as a whole conversion
//!   does. A texture's staging buffer is its own and starts zeroed, no conversion completed, so
//!   its first use converts every texel, and a texture made again over the same surface trusts
//!   no other texture's copy. Where the refresh is not made (planes not whole words or not at a
//!   word, past one storage range, or past the device's workgroup counts) or the planes' buffer
//!   is not the one its set binds, the conversion copies the planes and converts whole
//!   (`WorkCounts::ycbcr_refreshes`, `ycbcr_whole_conversions`).
//! - **The texels Metal's copies move** are the luma plane's bytes, a code's bytes a texel
//!   (for 4:2:2 and 4:4:4 and, for the code formats): `get_bytes`, `replace_region` and a blit to
//!   or from a buffer move the luma rows (Metal keeps the chroma plane), the texture's `Texels`
//!   (`buffer_texture.rs`). A packed texture's `get_bytes` and `replace_region` move nothing, at
//!   any bytes per row (the bytes and both planes as they were, 48 calls of each), and its blits
//!   with a buffer, which move bytes at places that follow no rule found, are refused.
//! - **Plane 1** (each of the 18 unpacked formats above over plane 1 of `422v`,
//!   `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44` surfaces, without a matrix and with
//!   BT.709, 7 x 5 to 64 x 64; `read()`, a `half` read and nearest samples, 0 texels off; and
//!   **Surfaces** above). Over plane 1 of a two-plane surface Metal reads plane 1's codes as the
//!   luma plane, at plane 1's rows (past the plane where it is shorter than the texture), with
//!   chroma codes 0, converted as over plane 0 (the surface's range and matrix); its copies move
//!   plane 1's codes. The runtime converts the same: the luma plane is plane 1, and the chroma
//!   plane one row of zeros the conversion's staging buffer holds past the plane's bytes, every
//!   row read from it. Over plane 1 of a three-plane surface (`y420`, `v0a8`) Metal reads plane
//!   2's pairs as chroma, which the facts do not name: the runtime reads chroma codes 0 there
//!   (docs/FAST-PATHS.md's known gaps).
//! - **Views.** A converted texture's view as its `_sRGB` twin (Metal makes it with
//!   `ShaderRead` usage alone) is a texture of the twin over the same planes, with a conversion
//!   of its own: Metal's view reads what a texture of the twin over the surface reads, bit for
//!   bit (`texture/ycbcr_image.rs`). A view in the texture's own format as a 2D array reads the
//!   luma codes alone, `(Y, 0, 0, 1)`, or `(0, Y, 0, 1)` of a format read as codes, its linear
//!   sample the luma of the code-reading format's (the M3 Max under macOS 27.0.1, bit for bit):
//!   an image of the codes over the same planes read through a swizzle
//!   (`Texture::luma_view`); an `_sRGB` twin's reads no defined value on Metal and reads as its
//!   converted format's here.
//! - **Surfaces** (an Apple M3 Max under macOS 27.0.1; our probe `ycr`: each of the 24 formats
//!   above over each plane of CoreVideo surfaces of 35 OSTypes, 16 x 8, random bytes, every texel
//!   of `read()`, 1320 textures; and textures of other sizes and matrices). Metal makes such a
//!   texture over any plane of a surface with planes and over plane 0 of one without (a later
//!   plane of one without asserts), at any size. It reads the luma codes from the asked plane at
//!   its offset and rows (a surface without planes: its base and rows) and the chroma pairs from
//!   the plane after it at that plane's rows, each in the format's own layout and subsampling
//!   whatever the surface's, or chroma codes 0 where no plane follows; bytes past the surface's
//!   allocation read as zeros. It converts by full range over the full-range OSTypes (`420f`,
//!   `422f`, `444f`, `xf20`, `xf22`, `xf44`, `pf20`, `pf22`, `pf44`) and by video range over
//!   every other. No matrix, and a value other than those named, read as BT.601. `DCI_P3` and
//!   `P3_D65` read each code alone: a converted format's red, green and blue are its luma, Cb and
//!   Cr codes over `2^b - 1` (a code format reads as under any matrix), and a pass stores by
//!   BT.601. A render target of a 4:2:0 or 4:2:2 format of an odd width or height is nil. The
//!   runtime reads the same (`surface_planes`): the luma plane the call's, the chroma plane the
//!   facts' plane 1 under plane 0 of a surface with planes and a row of zeros otherwise, the bytes
//!   past its buffer zeros the conversion holds; a texture over a P3 surface is an image of the
//!   codes of the code-reading format, read through a swizzle (`texture/ycbcr_image.rs`).
//! - **Refused** (`Unsupported`, `NotImplemented`, `Feature::Format`, naming what): a surface of
//!   a lossless compressed OSType (`&8v0`, `&xv2` and their kin: Metal reads compressed blocks), a
//!   plane past 0 of a surface without planes, plane 1 at other bytes than the surface's facts
//!   name, an `_sRGB` twin over a P3 surface (its reads follow no rule found), a load of a pass
//!   into a P3 texture, a write of one, and a view of one but in its own format and type (none
//!   measured), planes at offsets or rows off 16 bytes, and planes past one storage range; the
//!   writes and passes of a texture whose reads reach past its buffer or over plane 1, or whose
//!   rows of codes overlap; a packed texture's blit with a buffer (above); a copy between two such
//!   textures (Metal copies both planes), a view in another format than the twin (a twin's view
//!   as its converted format among them, unmeasured), and a pass rendering into one but as
//!   **Passes** below say (Metal converts back into the planes). The runtime's gaps besides
//!   (`NotImplemented`, `linear_plan.rs`): luma rows of codes that overlap (a 10-bit format over
//!   rows under twice its width, `OverlappingSurfaceRows`) and luma rows past the buffer (a
//!   texture taller than its plane at the surface's end, `TexelsPastBufferLength`). A
//!   function's writes to a 4:2:0, 4:2:2 or packed format are dropped, as Metal drops them (the
//!   packed 4:4:4 ones on the M3 Max under macOS 27.0.1, where an M4 stores them;
//!   `metal::format::writes_dropped`); a 4:4:4 `_sRGB` twin's are refused (the twins' encoding is
//!   not found), as are a texture's over plane 1 (not measured) and a slot's that reads the
//!   texture too.
//! - **Writes** (on an Apple M4 under macOS 27.0.1, on an Apple M3 Max under macOS 26.7, byte for
//!   byte alike). Metal stores a function's writes to `YCBCR8_444_2P`, `YCBCR10_444_2P`,
//!   `RGB8_444_2P` and `RGB10_444_2P` (503, 507, 542, 545) into both planes: each written texel's
//!   luma code at its place and its Cb and Cr pair at its own, the codes
//!   `metal2vulkan::bind::ycbcr_write` states (the converted formats' by
//!   the surface's range and matrix; the code formats' green, blue and red as codes), and leaves
//!   every other byte. The function writes such a texture through its write shadow (`shadow.rs`,
//!   `ShadowSource::YCbCrWrite`; `texture/ycbcr_image.rs`): an `RGBA32Uint` image of each written
//!   texel's bits and a mark, cleared before an encoder's first command that writes it and again
//!   after the planes' bytes are written another way; its copy back ([`YCbCrWriteBack`]) copies the
//!   planes' bytes into a staging buffer of the texture's, dispatches the encode kernel
//!   (`bind::ycbcr_write_module`), which stores the codes of the marked texels, and copies the
//!   planes' rows of codes back. The texture's converted texels then convert again at their next
//!   use, as after any write of the planes. Exact, so announced nowhere.
//! - **Passes** (on an Apple M4 under macOS 27.0.1, and on an Apple M3 Max under macOS 26.7 and
//!   27.0.1, byte for byte alike). Metal's tile holds a color attachment of `YCBCR8_420_2P`,
//!   `YCBCR8_422_2P`, `YCBCR8_444_2P` or their 10-bit formats (500 to 507), or of the code formats
//!   `RGB8_420_2P` to `RGB10_444_2P` (540 to 545), over plane 0 as halves of red, green and blue,
//!   and its store converts the tile back into both planes: every texel's
//!   luma code, drawn or not, and each chroma pair's codes of the block of texels it covers, the
//!   codes `metal2vulkan::bind::ycbcr_store` states (`metal::format::ycbcr_renders`). The pass
//!   renders into the texture's render shadow (`texture/ycbcr_image.rs`), an
//!   `R32G32B32A32_SFLOAT` image of the texture's size; its copy back is the write-back above
//!   storing every texel ([`YCbCrWriteBack`] of a store), and the texture's converted texels
//!   convert again at their next use. A pass that clears the attachment, or leaves it to the pass
//!   (`DontCare`), clears the shadow at its copy in and stores every texel, exact. A pass that
//!   loads it loads each texel as the texture reads it ([`YCbCrLoad`], the conversion kernel
//!   writing floats), and Metal stores only the 32x32 tiles its draws reach, every texel of each
//!   re-encoded, leaving the others' bytes (a load with no draw stores nothing): the copy back
//!   stores the tiles of the pass's drawn area (`render/shadows.rs`'s `DrawnArea`, the scissored
//!   bounds of its draws), so a tile inside that area that no primitive reaches is re-encoded
//!   where Metal keeps its bytes, announced (`ycbcr-load-tiles`). A `DontCare` store, and a store
//!   left `Unknown`, store nothing, as Metal's do. A write mask keeps the masked channels' cleared
//!   or loaded values, exact. A blend blends the shadow's floats where Metal blends its halves (an
//!   8-bit format's in half arithmetic, a 10-bit one's in `float` rounded toward zero), within a
//!   code of Metal's, announced at the draw (`ycbcr-blend-rounding`); on a device that does not
//!   blend `R32G32B32A32_SFLOAT` such a pipeline is refused, naming the need. A resolve is refused
//!   (`NotImplemented(Format)`, naming the format, at the pass), as are a texture over plane 1 and
//!   one whose size the chroma blocks do not tile, and the other formats' passes. A packed 10-bit
//!   format's pipeline is made, as Metal makes it, and its pass refused at its start, naming the
//!   format (`metal::format::ycbcr_pass_refused`): Metal fails the pass's command buffer, its error
//!   and the luma bytes it leaves varying from run to run, so no exact form exists.
//! - **Linear filtering** is Metal's, exact: the function emulates Metal's filter of the texels on
//!   their grid of 1/2048 or 1/8192 (`metal2vulkan::metal::texel_format::PackedTexels::YCbCr2048`,
//!   `YCbCr8192`), under MoltenVK through the slot's YCbCr variant, and announces, naming the slot,
//!   any sample the emulation does not reach (`ycbcr-filter-not-exact`). A function that samples
//!   textures through handles reads its sampler from memory and is refused where it can reach the
//!   texture (`MappedTextureSampling`, as the texture's address-mode map is).

use super::buffer::Buffer;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::texture_descriptor::TextureLayout;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    YCbCrCodes, YCbCrConvertModule, YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrRange,
    YCbCrRefreshModule, YCbCrWriteModule,
};
use metal2vulkan::error::{DiagName, Feature, Unsupported};
use metal2vulkan::metal::format::ycbcr_format;
use metal2vulkan::metal::PixelFormat;
use std::ops::Range;

/// The OSTypes of the full-range surfaces (the module docs): a texture over a surface of any other
/// OSType reads its codes as video range.
const FULL_RANGE: [&[u8; 4]; 9] = [
    b"420f", b"422f", b"444f", b"xf20", b"xf22", b"xf44", b"pf20", b"pf22", b"pf44",
];
/// The first byte of the OSTypes of CoreVideo's lossless compressed surfaces (`'&8v0'`, `'&xv2'`
/// and their kin), whose planes hold compressed blocks, which Metal reads as nothing found here.
const COMPRESSED: u8 = b'&';
/// What the planes' offsets and rows are multiples of: Metal's GPU reads a plane from its offset
/// rounded down to 16, rows `bytesPerRow` rounded down to 16 apart (`linear_plan.rs`), which
/// the conversion is measured at only where the two agree.
const PLANE_ALIGNMENT: u64 = 16;
/// Where the planes' bytes start in the staging buffer, after the shape and the refresh's state:
/// a multiple of every `minStorageBufferOffsetAlignment`.
const BYTES_AT: u64 = 256;
/// Where the refresh's state lies in the staging buffer, after the shape (`YCbCrRefresh`): the
/// conversion's `VkDispatchIndirectCommand` and the completed word the refresh reads
/// (`YCbCrRefreshModule::STATE_WORDS`), then [`COMPLETED_AT`]'s word.
const STATE_AT: u64 = 128;
/// The word that says the image holds the conversion of the staging buffer's bytes, 1, or that
/// no conversion completed since they last changed, 0: cleared before any of them changes, set
/// once the conversion that reads them is recorded (`YCbCrConversion::record`).
const COMPLETED_AT: u64 = STATE_AT + 4 * YCbCrRefreshModule::STATE_WORDS as u64;

/// The facts of an IOSurface a two-plane YCbCr texture over it reads, and a two-plane
/// extended-range one its plane 1 from
/// ([`IoSurfacePlane::ycbcr`](crate::IoSurfacePlane::ycbcr)): Metal's
/// `newTextureWithDescriptor:iosurface:plane:` reads them from the `IOSurfaceRef`, the
/// runtime's stand-in from here.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoSurfaceYCbCr {
    /// `IOSurfaceGetPixelFormat`: the surface's OSType, as a big-endian four-character code
    /// (`'420v'` is `0x3432_3076`).
    pub pixel_format: u32,
    /// The plane the texture is asked of (`newTextureWithDescriptor:iosurface:plane:`'s
    /// `plane`).
    pub plane: u64,
    /// `IOSurfaceGetWidth` and `IOSurfaceGetHeight`: the luma plane's size. A two-plane YCbCr
    /// texture reads its rows whatever its size against it, as Metal's does.
    pub size: [u64; 2],
    /// Plane 1's first byte in the buffer that stands for the surface:
    /// `IOSurfaceGetBaseAddressOfPlane(surface, 1)` less `IOSurfaceGetBaseAddress`. A two-plane
    /// extended-range texture reads its alpha plane there.
    pub chroma_offset: u64,
    /// `IOSurfaceGetBytesPerRowOfPlane(surface, 1)`.
    pub chroma_bytes_per_row: u64,
    /// The surface's `kIOSurfaceYCbCrMatrix` as the texture is made, `None` without one.
    pub matrix: Option<SurfaceYCbCrMatrix>,
}

impl IoSurfaceYCbCr {
    /// A surface of OSType `pixel_format` and `size`, the texture asked of `plane`, its chroma
    /// plane at `chroma_offset` with rows `chroma_bytes_per_row` apart, without a matrix.
    pub const fn new(
        pixel_format: u32,
        plane: u64,
        size: [u64; 2],
        chroma_offset: u64,
        chroma_bytes_per_row: u64,
    ) -> IoSurfaceYCbCr {
        IoSurfaceYCbCr {
            pixel_format,
            plane,
            size,
            chroma_offset,
            chroma_bytes_per_row,
            matrix: None,
        }
    }

    /// These facts with the surface's `kIOSurfaceYCbCrMatrix`.
    pub const fn with_matrix(mut self, matrix: SurfaceYCbCrMatrix) -> IoSurfaceYCbCr {
        self.matrix = Some(matrix);
        self
    }
}

/// An IOSurface's `kIOSurfaceYCbCrMatrix` attachment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceYCbCrMatrix {
    /// `ITU_R_601_4`.
    Itu601,
    /// `ITU_R_709_2`.
    Itu709,
    /// `ITU_R_2020`.
    Itu2020,
    /// `SMPTE_240M_1995`, which Metal reads as BT.709.
    Smpte240M,
    /// `DCI_P3`, which Metal reads as each code alone, not converted, and stores by BT.601 (the
    /// module docs).
    DciP3,
    /// `P3_D65`, which Metal reads and stores as `DCI_P3`.
    P3D65,
    /// Any other value, which Metal reads as BT.601.
    Other,
}

impl SurfaceYCbCrMatrix {
    /// The matrix of the attachment's string value (`kCVImageBufferYCbCrMatrix_*`).
    pub fn from_attachment(value: &str) -> SurfaceYCbCrMatrix {
        match value {
            "ITU_R_601_4" => SurfaceYCbCrMatrix::Itu601,
            "ITU_R_709_2" => SurfaceYCbCrMatrix::Itu709,
            "ITU_R_2020" => SurfaceYCbCrMatrix::Itu2020,
            "SMPTE_240M_1995" => SurfaceYCbCrMatrix::Smpte240M,
            "DCI_P3" => SurfaceYCbCrMatrix::DciP3,
            "P3_D65" => SurfaceYCbCrMatrix::P3D65,
            _ => SurfaceYCbCrMatrix::Other,
        }
    }
}

/// A refusal of a texture of `pf` over a surface: `NotImplemented`, naming `what`.
fn refused(pf: PixelFormat, what: String) -> Error {
    let named = format!("{} {what}", pf.name());
    Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming(DiagName::new(named)))
}

/// The planes of a two-plane YCbCr texture of `layout` (checked; a format `metal::format::ycbcr`
/// holds) over `surface`, its luma plane at `luma` (offset, bytes per row) of a buffer of
/// `length` bytes on `device`: the planes as the conversion reads them from the span's first
/// byte, and the span, the bytes the planes lie in. Refused as the module docs say.
pub(super) fn surface_planes(
    layout: &TextureLayout,
    (luma, planar): ([u64; 2], bool),
    surface: &IoSurfaceYCbCr,
    length: u64,
    device: &Device,
) -> Result<(YCbCrPlanes, Range<u64>), Error> {
    let pf = layout.pixel_format;
    let refused = |what: String| refused(pf, what);
    let mut codes = ycbcr_format(pf)
        .and_then(YCbCrCodes::of)
        .ok_or_else(|| internal("a YCbCr texture of a layout the conversion does not read"))?;
    let code = || {
        surface
            .pixel_format
            .to_be_bytes()
            .map(|b| b.escape_ascii().to_string())
            .concat()
    };
    if surface.pixel_format.to_be_bytes()[0] == COMPRESSED {
        return Err(refused(format!("over a surface of OSType '{}'", code())));
    }
    let full = FULL_RANGE
        .iter()
        .any(|&code| u32::from_be_bytes(*code) == surface.pixel_format);
    let range = match full {
        true => YCbCrRange::Full,
        false => YCbCrRange::Video,
    };
    // The chroma codes are plane 1's, of a texture over plane 0 of a surface with planes, and 0
    // of any other.
    let chroma_zero = match surface.plane {
        0 => !planar,
        _ if !planar => return Err(refused(format!("of plane {}", surface.plane))),
        _ => true,
    };
    let matrix = match surface.matrix {
        None | Some(SurfaceYCbCrMatrix::Itu601 | SurfaceYCbCrMatrix::Other) => YCbCrMatrix::Bt601,
        Some(SurfaceYCbCrMatrix::Itu709 | SurfaceYCbCrMatrix::Smpte240M) => YCbCrMatrix::Bt709,
        Some(SurfaceYCbCrMatrix::Itu2020) => YCbCrMatrix::Bt2020,
        // A P3 matrix: a converted format reads each code alone, its luma, Cb and Cr as red,
        // green and blue, which the codes read (the texture presents them swizzled,
        // `texture/ycbcr_image.rs`), and stores by BT.601; an `_sRGB` twin's reads follow no rule
        // found.
        Some(p3 @ (SurfaceYCbCrMatrix::DciP3 | SurfaceYCbCrMatrix::P3D65)) => {
            match codes.output {
                YCbCrOutput::Srgb => return Err(refused(format!("of YCbCr matrix {p3:?}"))),
                _ => codes.output = YCbCrOutput::Codes,
            }
            YCbCrMatrix::Bt601
        }
    };
    let [width, height, _] = layout.extent.map(u64::from);
    if layout.levels != 1 || layout.layers != 1 {
        return Err(refused(format!(
            "of {} levels, {} slices",
            layout.levels, layout.layers
        )));
    }
    let halved = codes.subsampling.halved();
    let [cw, ch] =
        [(width, halved[0]), (height, halved[1])]
            .map(|(n, halved)| if halved { n.div_ceil(2) } else { n });
    let chroma = [surface.chroma_offset, surface.chroma_bytes_per_row];
    let read = luma
        .iter()
        .chain(if chroma_zero { &[] } else { &chroma[..] });
    let aligned = read.into_iter().all(|v| v.is_multiple_of(PLANE_ALIGNMENT));
    // A row of `n` codes is `units(n)` units of `bytes` bytes (three packed codes to a unit).
    let unit = u64::from(codes.bytes);
    let [luma_bytes, chroma_bytes] = [width, 2 * cw].map(|n| unit * codes.units(n));
    // Over plane 1, the call's plane is the surface's plane 1, and its codes the luma.
    if !aligned || (surface.plane == 1 && planar && luma != chroma) {
        return Err(refused(format!(
            "with planes at {luma:?} and {chroma:?} (offset, bytes per row)"
        )));
    }
    // The bytes the planes' rows reach, read whether or not the surface's planes hold them;
    // the buffer's from its first, the rest past its end read as zeros (`YCbCrConversion`).
    let end = |[offset, row]: [u64; 2], rows: u64, bytes: u64| offset + row * (rows - 1) + bytes;
    let reach = match chroma_zero {
        true => luma[0]..end(luma, height, luma_bytes),
        false => {
            luma[0].min(chroma[0])..end(luma, height, luma_bytes).max(end(chroma, ch, chroma_bytes))
        }
    };
    if reach.start >= length {
        return Err(refused(format!(
            "with planes from byte {} of a buffer of {length}",
            reach.start
        )));
    }
    let span = reach.start..reach.end.min(length);
    let limit = u64::from(
        device
            .properties()
            .description
            .limits
            .max_storage_buffer_range,
    )
    .min(u64::from(u32::MAX));
    // With chroma codes 0, the conversion holds a row of zeros past the reach too.
    let zeros = if chroma_zero { chroma_bytes + 4 } else { 0 };
    if reach.end - reach.start + zeros > limit - BYTES_AT {
        return Err(refused(format!(
            "with planes over bytes {reach:?}, past one storage range"
        )));
    }
    // The reach is under `u32::MAX`, so every offset and row in it is.
    let word = |v: u64| v as u32;
    // Chroma codes 0: one row of zeros past the reach, which the conversion holds
    // (`YCbCrConversion::new`), every row read from it.
    let chroma = match chroma_zero {
        true => [(reach.end - reach.start).next_multiple_of(4), 0],
        false => [chroma[0] - reach.start, chroma[1]],
    };
    let planes = YCbCrPlanes::new(
        [word(width), word(height)],
        [word(luma[0] - span.start), word(luma[1])],
        [word(chroma[0]), word(chroma[1])],
        (range, matrix),
        codes,
    );
    Ok((planes, span))
}

/// The end of the bytes `planes` read, from their first: the end of the luma plane's last row
/// of codes, or of the chroma plane's.
fn planes_end(planes: &YCbCrPlanes) -> u64 {
    let [w, h] = planes.size.map(u64::from);
    let [cw, ch] = planes.chroma_size().map(u64::from);
    // A row of `n` codes: as many units, or a third of them packed.
    let row = |n: u64| planes.codes.units(n) * u64::from(planes.codes.bytes);
    let end = |[at, pitch]: [u32; 2], rows: u64, bytes: u64| {
        u64::from(at) + u64::from(pitch) * (rows.max(1) - 1) + bytes
    };
    end(planes.luma, h, row(w)).max(end(planes.chroma, ch, row(2 * cw)))
}

/// The conversion of a two-plane YCbCr texture's planes into its image (the module docs): the
/// copy in of its shadow (`ShadowSource::Converted`).
pub(super) struct YCbCrConversion {
    /// The planes it converts, and the bytes of their buffer they lie in, which a view as the
    /// texture's `_sRGB` twin converts too (`texture/ycbcr_image.rs`).
    source: (YCbCrPlanes, Range<u64>),
    /// The shape at its start, the planes' bytes from `BYTES_AT`; held for the texture's life.
    staging: Buffer,
    /// The staging buffer's identity and its bytes.
    staging_use: (ResourceKey, Range<u64>),
    /// Where the shape lies in the staging buffer's `VkBuffer`.
    shape_at: u64,
    shape: [u8; YCbCrConvertModule::SHAPE_WORDS as usize * 4],
    /// The planes' bytes from the surface's buffer into the staging buffer.
    into: vk::BufferCopy,
    /// Where the zeros a texture over plane 1 reads as its chroma lie in the staging buffer's
    /// `VkBuffer`, and their bytes (0 for every other texture).
    zeros: (u64, u64),
    kernel: OwnedKernel,
    sets: OwnedSets,
    groups: [u32; 2],
    /// The refresh that converts only where the planes changed, where the device makes it
    /// ([`YCbCrRefresh`]); every conversion is whole without it.
    refresh: Option<YCbCrRefresh>,
}

/// The refresh of a conversion's staging copy (the module docs' **When it converts**): one
/// dispatch of the refresh kernel (`bind::ycbcr_refresh_module`) compares the planes' words with
/// the copy, stores each changed one over it, and writes the conversion's group count into the
/// indirect arguments where any changed or no conversion completed from the copy; the conversion
/// is then dispatched through them. Its set binds the planes in the `VkBuffer` the texture was
/// made over: a conversion recorded over another (a rebind of a no-copy buffer's pages) takes
/// the whole conversion.
struct YCbCrRefresh {
    kernel: OwnedKernel,
    sets: OwnedSets,
    /// The `VkBuffer` its set binds the planes in.
    planes: vk::Buffer,
    shape: [u8; YCbCrRefreshModule::PUSH_BYTES as usize],
    groups: [u32; 2],
}

impl YCbCrRefresh {
    /// The refresh of the `span` bytes of planes at `at` of `planes`, a `VkBuffer`, into the
    /// staging buffer `staging` (its `VkBuffer` and the copy's offset there), for a conversion of
    /// `convert_groups` workgroups across; `None` where the planes' bytes are not whole words or
    /// lie past what one storage binding reaches, or the device does not make the kernel.
    fn new(
        device: &Device,
        (planes, at, span): (vk::Buffer, u64, u64),
        (staging, start): (vk::Buffer, u64),
        convert_groups: u32,
    ) -> Option<YCbCrRefresh> {
        if span == 0 || !span.is_multiple_of(4) || !at.is_multiple_of(4) {
            return None;
        }
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let offset = at - at % align;
        let range = at + span - offset;
        let first = (at - offset) / 4;
        let words = u32::try_from(span / 4).ok()?;
        if range > u64::from(limits.max_storage_buffer_range)
            || first > u64::from(YCbCrRefreshModule::MOST_FIRST_WORD)
        {
            return None;
        }
        let groups = YCbCrRefreshModule::groups(words);
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return None;
        }
        let kernel = device.owned_kernel(OwnedModule::YCbCrRefresh).ok()?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: Vec::new(),
            buffers: vec![
                (planes, offset, range),
                (staging, start + BYTES_AT, span),
                (
                    staging,
                    start + STATE_AT,
                    4 * u64::from(YCbCrRefreshModule::STATE_WORDS),
                ),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set]).ok()?;
        Some(YCbCrRefresh {
            kernel,
            sets,
            planes,
            shape: YCbCrRefreshModule::shape([first as u32, 0], words, convert_groups),
            groups,
        })
    }
}

impl YCbCrConversion {
    /// The conversion of `planes`, whose bytes lie in `bytes` of their buffer, at `at` of its
    /// `VkBuffer`, into `view`, a storage view of the texture's image, on `device`. Refused as
    /// `new_buffer` refuses the staging buffer and as the kernel's objects are refused.
    pub(super) fn new(
        device: &Device,
        (planes, bytes_in): (&YCbCrPlanes, Range<u64>),
        (source, at): (vk::Buffer, u64),
        view: vk::ImageView,
    ) -> Result<YCbCrConversion, Error> {
        let span = bytes_in.end - bytes_in.start;
        let bytes = span.next_multiple_of(4);
        // The bytes the planes read past their span, zeros (the chroma of a texture whose chroma
        // codes are 0, and rows past the end of the buffer), filled at every conversion.
        let zeros = planes_end(planes).saturating_sub(bytes).next_multiple_of(4);
        let held = bytes + zeros;
        let staging = device.own_buffer(BYTES_AT + held, ResourceOptions::private())?;
        // One of these a texture, the planes' whole byte range
        // (`WorkCounts::ycbcr_conversion_scratch_buffers`).
        device
            .shared()
            .counts
            .ycbcr_conversion_scratch(BYTES_AT + held);
        let (raw, start) = staging.at(0);
        let (identity, range) = staging.tracked(0..BYTES_AT + held);
        let shape = YCbCrConvertModule::shape(planes);
        let kernel = device.owned_kernel(OwnedModule::YCbCrConvert(planes.codes.output, false))?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (raw, start, shape.len() as u64),
                (raw, start + BYTES_AT, held),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let groups = planes
            .size
            .map(|n| n.div_ceil(YCbCrConvertModule::LOCAL_SIZE));
        let limits = &device.properties().description.limits;
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return Err(internal(
                "a YCbCr texture past the device's workgroup counts",
            ));
        }
        let refresh = YCbCrRefresh::new(device, (source, at, span), (raw, start), groups[0]);
        Ok(YCbCrConversion {
            source: (*planes, bytes_in),
            staging_use: (identity.key, range),
            shape_at: start,
            shape,
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start + BYTES_AT,
                size: span,
            },
            zeros: (start + BYTES_AT + bytes, zeros),
            staging,
            kernel,
            sets,
            groups,
            refresh,
        })
    }

    /// The planes it converts, and the bytes of their buffer they lie in.
    pub(super) fn source(&self) -> (YCbCrPlanes, Range<u64>) {
        (self.source.0, self.source.1.clone())
    }

    /// The staging buffer, and where the refresh's state lies in it: the conversion's
    /// `VkDispatchIndirectCommand`, the completed word the refresh read, then the completed word
    /// (five words); `None` where the conversion has no refresh.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn refresh_state(&self) -> Option<(&Buffer, u64)> {
        self.refresh.as_ref().map(|_| (&self.staging, STATE_AT))
    }

    /// The conversion's workgroups across, which the refresh's indirect arguments take where it
    /// converts.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn groups_across(&self) -> u32 {
        self.groups[0]
    }

    /// Whether the texture reads zeros the staging buffer holds past the planes' bytes: chroma
    /// codes 0 (over plane 1 of its surface, or a surface without planes), or rows past the end
    /// of its buffer. Its writes and passes, and its retyped views, are refused.
    pub(super) fn reads_zeros(&self) -> bool {
        self.zeros.1 > 0
    }

    /// The conversion's uses: `truth`, the read of the planes' bytes, then the staging buffer's
    /// writes and read, and the write of every texel of the image of `copies`.
    pub(super) fn uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let (key, range) = self.staging_use.clone();
        vec![
            truth,
            Use {
                key,
                tracked: copies.tracked,
                range: UseRange::Bytes(range),
                read: Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT
                        | vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ
                        | vk::AccessFlags2::TRANSFER_READ,
                ),
                write: Access::new(
                    vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            },
            Use::writes(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ),
        ]
    }

    /// Record the conversion of the planes in `source` (the surface's buffer's `VkBuffer` now)
    /// into `command_buffer`: through the refresh where the conversion has one over `source`
    /// ([`YCbCrRefresh`]), converting only where the planes' bytes changed since the conversion
    /// its image holds, else whole. The completed word is cleared before any byte of the staging
    /// buffer's copy changes and set after the conversion's dispatch, so a command buffer that
    /// ends between the two, or never runs, leaves the next conversion whole.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `source`, the staging buffer and
    /// the image are live until it completes, and the image is in the `GENERAL` layout.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
    ) {
        let owner = self.staging.device();
        let staging = self.staging.at(0).0;
        let state_at = self.shape_at + STATE_AT;
        let completed_at = self.shape_at + COMPLETED_AT;
        let word = |w: u32| w.to_le_bytes();
        let refresh = self
            .refresh
            .as_ref()
            .filter(|refresh| refresh.planes == source);
        // SAFETY: the caller's contract; the shape, the state, the copy and the zeros lie in the
        // staging buffer, the copy and the planes' words in `source` (checked when the texture was
        // made), and the sets bind them.
        unsafe {
            device.cmd_update_buffer(command_buffer, staging, self.shape_at, &self.shape);
            let (zeros_at, zeros) = self.zeros;
            crate::word_fill::record(
                self.staging.device(),
                command_buffer,
                staging,
                zeros_at..zeros_at + zeros,
                0,
            );
            match refresh {
                Some(refresh) => {
                    // The arguments of no workgroup across, which the refresh raises where it
                    // must; the completed word, kept for the refresh, then cleared.
                    let arguments: Vec<u8> =
                        [0, self.groups[1], 1].into_iter().flat_map(word).collect();
                    device.cmd_update_buffer(command_buffer, staging, state_at, &arguments);
                    let kept = vk::BufferCopy {
                        src_offset: completed_at,
                        dst_offset: state_at + 4 * u64::from(YCbCrRefreshModule::COMPLETED),
                        size: 4,
                    };
                    device.cmd_copy_buffer(command_buffer, staging, staging, &[kept]);
                    Barrier::everything().record(device, command_buffer);
                    device.cmd_update_buffer(command_buffer, staging, completed_at, &word(0));
                    Barrier::everything().record(device, command_buffer);
                    let point = vk::PipelineBindPoint::COMPUTE;
                    device.cmd_bind_pipeline(command_buffer, point, refresh.kernel.pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        point,
                        refresh.kernel.layout,
                        0,
                        &refresh.sets.sets,
                        &[],
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        refresh.kernel.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &refresh.shape,
                    );
                    device.cmd_dispatch(command_buffer, refresh.groups[0], refresh.groups[1], 1);
                    owner.shared().counts.ycbcr_refreshed();
                }
                None => {
                    device.cmd_update_buffer(command_buffer, staging, completed_at, &word(0));
                    Barrier::everything().record(device, command_buffer);
                    device.cmd_copy_buffer(command_buffer, source, staging, &[self.into]);
                    owner.shared().counts.ycbcr_converted_whole();
                }
            }
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &self.sets.sets,
                &[],
            );
            match refresh {
                Some(_) => device.cmd_dispatch_indirect(command_buffer, staging, state_at),
                None => device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1),
            }
            Barrier::everything().record(device, command_buffer);
            device.cmd_update_buffer(command_buffer, staging, completed_at, &word(1));
        }
    }
}

/// The write-back of a 4:4:4 two-plane YCbCr texture's written texels into its planes (the
/// module docs): the copy back of its write shadow (`ShadowSource::YCbCrWrite`). The planes'
/// bytes go into a staging buffer of the texture's, the encode kernel (`bind::ycbcr_write_module`,
/// `owned_kernel.rs`) stores the codes of every texel the shadow marks written, and the planes'
/// rows of codes go back; the bytes between the rows stay as they were. A render shadow's
/// write-back stores every texel, of the tiles a pass changed alone where it knows them
/// ([`YCbCrWriteBack::record`]), and loads the shadow too ([`YCbCrLoad`]).
pub(super) struct YCbCrWriteBack {
    /// The shape at its start, the planes' bytes from `BYTES_AT`, and a load's shape after them;
    /// held for the texture's life.
    staging: Buffer,
    /// The staging buffer's identity and its bytes.
    staging_use: (ResourceKey, Range<u64>),
    /// Where the shape lies in the staging buffer's `VkBuffer`.
    shape_at: u64,
    shape: [u8; YCbCrWriteModule::SHAPE_WORDS as usize * 4],
    /// The planes' bytes from the surface's buffer into the staging buffer.
    into: vk::BufferCopy,
    /// Where the planes' bytes start in the staging buffer's `VkBuffer`, and in the surface's
    /// buffer's.
    bytes_at: [u64; 2],
    /// The planes, whose rows of codes go back.
    planes: YCbCrPlanes,
    kernel: OwnedKernel,
    sets: OwnedSets,
    groups: [u32; 2],
    /// The load of a render shadow, `None` for a write shadow.
    load: Option<YCbCrLoad>,
}

/// The load of a render shadow (the module docs' **Passes**): the copy in of a pass that loads
/// the attachment, which converts the planes' bytes into the shadow as floats through the
/// conversion kernel (`bind::ycbcr_convert_floats_module`), each texel what the texture reads.
/// It shares the write-back's staging buffer, its shape past the planes' bytes.
struct YCbCrLoad {
    /// Where the shape lies in the staging buffer's `VkBuffer`.
    shape_at: u64,
    shape: [u8; YCbCrConvertModule::SHAPE_WORDS as usize * 4],
    kernel: OwnedKernel,
    sets: OwnedSets,
    groups: [u32; 2],
}

/// The side of the tiles Metal loads and stores a render pass's attachment by
/// (`bind::tile_reach`): a pass that loads a two-plane YCbCr attachment stores every texel of
/// each tile it reaches, its codes re-encoded from what the tile loaded.
const TILE: u32 = 32;

impl YCbCrWriteBack {
    /// The write-back into `planes` (a layout `bind::ycbcr_write` encodes, over plane 0), whose
    /// bytes lie in `bytes` of their buffer, at `at` of its `VkBuffer`, of the written texels in
    /// `view`, a storage view of the write shadow, on `device`; or, `all`, a pass's store of
    /// every texel of its render shadow (a layout `bind::ycbcr_store` encodes), the view's texels'
    /// bits. Refused as `new_buffer` refuses the staging buffer and as the kernel's objects are
    /// refused.
    pub(super) fn new(
        device: &Device,
        (planes, bytes_in): (&YCbCrPlanes, Range<u64>),
        at: u64,
        (view, all): (vk::ImageView, bool),
        load_view: Option<vk::ImageView>,
    ) -> Result<YCbCrWriteBack, Error> {
        let span = bytes_in.end - bytes_in.start;
        // The kernel reads and writes whole words: a row's last may reach past the span's end.
        let held = span.next_multiple_of(4);
        // A load's shape past the planes' bytes, at a multiple of every
        // `minStorageBufferOffsetAlignment`.
        let load_shape_at = BYTES_AT + held.next_multiple_of(BYTES_AT);
        let length = match load_view {
            Some(_) => load_shape_at + 4 * u64::from(YCbCrConvertModule::SHAPE_WORDS),
            None => BYTES_AT + held,
        };
        let staging = device.own_buffer(length, ResourceOptions::private())?;
        // As the conversion's, one a texture that takes writes.
        device.shared().counts.ycbcr_conversion_scratch(length);
        let (raw, start) = staging.at(0);
        let (identity, range) = staging.tracked(0..length);
        let shape = YCbCrWriteModule::shape(planes, all);
        let kernel = device.owned_kernel(OwnedModule::YCbCrWrite)?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (raw, start, shape.len() as u64),
                (raw, start + BYTES_AT, held),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let groups = YCbCrWriteModule::groups(planes);
        let limits = &device.properties().description.limits;
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return Err(internal(
                "a YCbCr texture past the device's workgroup counts",
            ));
        }
        let load = match load_view {
            Some(view) => Some(YCbCrLoad::new(
                device,
                planes,
                (raw, start, load_shape_at),
                held,
                view,
            )?),
            None => None,
        };
        Ok(YCbCrWriteBack {
            staging_use: (identity.key, range),
            shape_at: start,
            shape,
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start + BYTES_AT,
                size: span,
            },
            bytes_at: [start + BYTES_AT, at],
            planes: *planes,
            staging,
            kernel,
            sets,
            groups,
            load,
        })
    }

    /// Whether it loads a render shadow ([`YCbCrLoad`]): its copy in converts the planes.
    pub(super) fn loads(&self) -> bool {
        self.load.is_some()
    }

    /// The rows of codes of both planes it copies back from the staging buffer: of every texel,
    /// or, `area` (`[x0, y0, x1, y1)` of the texture's texels), of the tiles of [`TILE`] texels
    /// a side from the texture's origin that it meets, clipped to the texture; each chroma pair
    /// of the luma texels it covers.
    fn back(&self, area: Option<[u32; 4]>) -> Vec<vk::BufferCopy> {
        let planes = &self.planes;
        let [width, height] = planes.size;
        let [x0, y0, x1, y1] = match area {
            Some([x0, y0, x1, y1]) => [
                x0 - x0 % TILE,
                y0 - y0 % TILE,
                x1.next_multiple_of(TILE).min(width),
                y1.next_multiple_of(TILE).min(height),
            ],
            None => [0, 0, width, height],
        };
        if x0 >= x1 || y0 >= y1 {
            return Vec::new();
        }
        let unit = u64::from(planes.codes.bytes);
        let [from, to] = self.bytes_at;
        let rows = |[offset, row]: [u32; 2], rows: Range<u32>, columns: Range<u32>, pair: u64| {
            rows.map(move |y| {
                let first = u64::from(offset) + u64::from(y) * u64::from(row);
                let first = first + pair * unit * u64::from(columns.start);
                vk::BufferCopy {
                    src_offset: from + first,
                    dst_offset: to + first,
                    size: pair * unit * u64::from(columns.end - columns.start),
                }
            })
        };
        let [across, down] = planes.codes.subsampling.halved();
        let halve = |v: u32, halved: bool| if halved { v.div_ceil(2) } else { v };
        let luma = rows(planes.luma, y0..y1, x0..x1, 1);
        let chroma = rows(
            planes.chroma,
            halve(y0, down)..halve(y1, down),
            halve(x0, across)..halve(x1, across),
            2,
        );
        luma.chain(chroma).collect()
    }

    /// The write-back's uses: `truth`, the read and the write of the planes' bytes, then the
    /// staging buffer's writes and reads, and the read of every texel of the shadow of `copies`.
    pub(super) fn uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let (key, range) = self.staging_use.clone();
        let both = vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        vec![
            truth,
            Use {
                key,
                tracked: copies.tracked,
                range: UseRange::Bytes(range),
                read: Access::new(
                    both,
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                write: Access::new(
                    both,
                    vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            },
            Use::reads(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            ),
        ]
    }

    /// Record the write-back into the planes in `source` (the surface's buffer's `VkBuffer`
    /// now) into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `source`, the staging buffer and
    /// the shadow are live until it completes, and the shadow is in the `GENERAL` layout.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
        area: Option<[u32; 4]>,
    ) {
        let staging = self.staging.at(0).0;
        let back = self.back(area);
        if back.is_empty() {
            return;
        }
        // SAFETY: the caller's contract; the shape, the copy in and the rows lie in the staging
        // buffer and in `source` (checked when the texture was made), and the set binds them.
        unsafe {
            device.cmd_update_buffer(command_buffer, staging, self.shape_at, &self.shape);
            device.cmd_copy_buffer(command_buffer, source, staging, &[self.into]);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &self.sets.sets,
                &[],
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
            device.cmd_copy_buffer(command_buffer, staging, source, &back);
        }
    }

    /// The load's uses ([`YCbCrLoad`]): `truth`, the read of the planes' bytes, then the staging
    /// buffer's writes and reads, and the write of every texel of the shadow of `copies`.
    pub(super) fn load_uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let mut uses = self.uses(truth, copies);
        if let Some(shadow) = uses.last_mut() {
            *shadow = Use::writes(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            );
        }
        uses
    }

    /// Record the load of the render shadow ([`YCbCrLoad`]) from the planes in `source` (the
    /// surface's buffer's `VkBuffer` now) into `command_buffer`: nothing for a write shadow.
    ///
    /// # Safety
    ///
    /// As [`record`](Self::record)'s.
    pub(super) unsafe fn record_load(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
    ) {
        let Some(load) = &self.load else {
            return;
        };
        let staging = self.staging.at(0).0;
        // SAFETY: the caller's contract; the shape and the copy in lie in the staging buffer and
        // in `source` (checked when the texture was made), and the load's set binds them.
        unsafe {
            device.cmd_update_buffer(command_buffer, staging, load.shape_at, &load.shape);
            device.cmd_copy_buffer(command_buffer, source, staging, &[self.into]);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                load.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                load.kernel.layout,
                0,
                &load.sets.sets,
                &[],
            );
            device.cmd_dispatch(command_buffer, load.groups[0], load.groups[1], 1);
        }
    }
}

impl YCbCrLoad {
    /// The load of `planes` into `view`, a float storage view of the render shadow, through the
    /// staging buffer `staging` (its `VkBuffer`, its first byte there, and the load's shape's
    /// offset in it), whose planes' bytes are `held` bytes from `BYTES_AT`.
    fn new(
        device: &Device,
        planes: &YCbCrPlanes,
        (raw, start, shape_at): (vk::Buffer, u64, u64),
        held: u64,
        view: vk::ImageView,
    ) -> Result<YCbCrLoad, Error> {
        let shape = YCbCrConvertModule::shape(planes);
        let kernel = device.owned_kernel(OwnedModule::YCbCrConvert(planes.codes.output, true))?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (raw, start + shape_at, shape.len() as u64),
                (raw, start + BYTES_AT, held),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let groups = planes
            .size
            .map(|n| n.div_ceil(YCbCrConvertModule::LOCAL_SIZE));
        Ok(YCbCrLoad {
            shape_at: start + shape_at,
            shape,
            kernel,
            sets,
            groups,
        })
    }
}
