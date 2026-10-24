//! Texture copies: Metal's blit copies between buffers and textures and between textures, and
//! `replaceRegion` and `getBytes`, as Vulkan copy regions.
//!
//! A copy moves Metal's **copy units**: a pixel of an uncompressed format, a block of a compressed
//! one, and of a combined depth/stencil format the aspect the blit option names (a depth texel as
//! a 4-byte float, a stencil texel as a byte). Vulkan's buffer layout for the same aspect of the
//! image's format is the same bytes for every format the runtime holds exactly, so a copy is a
//! byte copy in both APIs, whatever view emulation the image uses for sampling.
//!
//! Metal's rules (API validation, measured on an Apple M3 Max and M4):
//!
//! - **Regions** stay in the level, a compressed region in the level's blocks: a level that ends
//!   inside a block holds that whole block, so a region may name it past the level's texels (4 x 4
//!   of a 4 x 2 level). Metal's validation takes such a region out of a texture, between textures
//!   and of the CPU, and stops it into a texture from a buffer, which plain Metal copies all the
//!   same (an Apple M3 Max under macOS 26.7, a probe). A compressed region starts on a block and
//!   covers whole blocks, except where it reaches the level's edge. Plain Metal copies a region off
//!   its blocks as the blocks it starts in: the origin rounded down to its block and the size
//!   rounded up to whole blocks, each on its own, so does the runtime, ending
//!   at the level's edge where the blocks pass it, in every copy between buffers and textures,
//!   between textures, and of the CPU.
//! - **Buffer layouts:** the offset and `bytesPerRow` are multiples of the copy unit;
//!   `bytesPerRow` covers a row when the region has more than one; `bytesPerImage` is a multiple
//!   of the copy unit covering the region's rows when the region has more than one image (one
//!   image ignores it); the bytes stay in the buffer. A blit of an offset, `bytesPerRow` or
//!   `bytesPerImage` off them copies where plain Metal places its rows (`blit_plan.rs`).
//! - **Blit options:** a combined depth/stencil texture is copied by one aspect, named by
//!   `DepthFromDepthStencil` or `StencilFromDepthStencil`; no other texture takes an option. The
//!   stencil view of a combined one (`X32Stencil8`) copies its stencil whatever the option, as
//!   plain Metal does (a probe; its validation stops `DepthFromDepthStencil`). Plain Metal copies
//!   the other forms its validation stops, and so does the runtime (an M4 under macOS 27.0.1): a
//!   combined depth/stencil texture without an option copies its depth, and an option that names an
//!   aspect the format lacks is ignored (a colour format copies its texels, `Depth32Float` its
//!   depth, `Stencil8` its stencil). `RowLinearPVRTC` lays a PVRTC texture's blocks out row by row,
//!   and another format ignores it, as plain Metal does (its validation stops it on another
//!   compressed format).
//! - **Texture to texture:** sample counts match, and the formats are equal (macOS 27.0.1's
//!   validation) or of one copy unit (macOS 26.7's; both copy the bytes without validation). The
//!   runtime takes macOS 26.7's rule. A depth-only or stencil-only format, or the stencil view
//!   `X32Stencil8`, has its one aspect's unit (2 bytes for `Depth16Unorm`, 4 for `Depth32Float`,
//!   1 for stencil), and Metal copies its bytes to and from colour formats of that unit and
//!   between stencil formats (probes on an M3 Max and an M4: every value, NaN payloads, denormals
//!   and depth outside `[0, 1]` included); a combined depth/stencil format copies only into its own
//!   format, both aspects. A copy of levels and slices needs equal sizes at each pair of levels.
//! - **CPU access:** `replaceRegion` and `getBytes` refuse a private texture and a combined
//!   depth/stencil format (a depth-only or stencil-only format is copied), and take the buffer
//!   layout rules above without an offset. Plain Metal copies a combined format's depth, as a
//!   blit without an option does, the stencil left as it is, and so does the runtime. A
//!   `bytesPerRow` or `bytesPerImage` off the rules is taken as plain Metal takes it
//!   ([`plan_host_copy`]).
//!
//! `replaceRegion` and `getBytes` copy through a tightly packed staging layout, packed on the
//! host ([`HostLayout`]); the blit encoder's copies are `blit_plan.rs`'s.

use super::chroma_pairs::{is_four_two_two, TEXEL_BYTES};
use super::host_bytes::HostBytes;
use super::pvrtc::{is_pvrtc, BLOCK_BYTES};
use super::view_plan::ViewRecord;
use metal2vulkan::bind::YCbCrCodes;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{
    aspects, byte_format, is_depth_stencil, ycbcr_format, FormatMapping,
};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::{Format, ImageAspectFlags};

/// `MTLBlitOption`, as far as copies take one.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlitOption {
    /// `MTLBlitOptionNone`.
    #[default]
    None,
    /// `MTLBlitOptionDepthFromDepthStencil`: the depth of a combined depth/stencil texture.
    DepthFromDepthStencil,
    /// `MTLBlitOptionStencilFromDepthStencil`: the stencil of a combined depth/stencil texture.
    StencilFromDepthStencil,
    /// `MTLBlitOptionRowLinearPVRTC`: a PVRTC texture's blocks lie in the buffer row by row, as
    /// every other compressed format's do (Metal lays them out in block order without it).
    /// Another format ignores it, as plain Metal does.
    RowLinearPVRTC,
}

/// A copy Metal's API validation stops.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CopyProblem {
    /// `replaceRegion` or `getBytes` of a private texture. Plain Metal answers by the texture's
    /// private layout, which no descriptor rule predicts (an M4 under macOS 27.0.1): a texture of
    /// up to 8 KiB copies as a shared one, most of 16 KiB and more end the process, a 64x64 one
    /// with `ShaderWrite` of 4 bytes a texel copies nothing, and a combined depth/stencil one
    /// copies nothing at 2x2.
    CpuAccessPrivate,
    /// A level or slice past the texture, or a region past its level (a compressed region past
    /// the level's last block).
    Region,
    /// A buffer offset that is not a multiple of the copy unit, of a blit whose plain Metal
    /// answer is not measured: a depth or stencil format other than `Depth32Float`, or a
    /// texture over a buffer (other blits copy where plain Metal places the rows).
    Offset,
    /// A `bytesPerRow` that is not a multiple of the copy unit, or that does not cover a row,
    /// where plain Metal's answer is undefined or not measured: a blit out of a texture whose
    /// rows overlap (Metal's writes race), Metal's aligned path into a texture at a pitch below a
    /// row (it reads garbage at 0) or from an offset of 16 or more, a depth or stencil format
    /// other than `Depth32Float`, a texture over a buffer, and a CPU copy of a texture over a
    /// buffer or of a pitch of 0 for more than one row and image (other layouts copy where plain
    /// Metal places the rows).
    BytesPerRow,
    /// Bytes past the end of the buffer: of a no-copy buffer read from, past the end of its
    /// host pages, which plain Metal reads up to. Past that, and past another buffer's length,
    /// plain Metal reads zeros, bytes a write past the length left, or another resource's bytes (an
    /// M4 under macOS 27.0.1).
    BufferRange,
    /// Two textures whose formats a copy does not take. Plain Metal copies the source's bytes
    /// in the GPU's own texel layout, which differs between formats of different units (an M4
    /// under macOS 27.0.1): no defined answer.
    Format,
    /// Two textures of different sample counts. Plain Metal copies the bytes of the GPU's own
    /// sample layout: no defined answer.
    SampleCount,
    /// A copy of whole textures with no levels of one size, where plain Metal aborts; or a copy
    /// between compressed levels of different sizes, not measured. (Other levels of different
    /// sizes copy their common region, and whole textures the levels and slices they share,
    /// paired by size, as plain Metal copies them.)
    Size,
}

/// Why a copy cannot be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CopyRefusal {
    Invalid(CopyProblem),
    Unsupported(Unsupported),
}

impl From<CopyProblem> for CopyRefusal {
    fn from(problem: CopyProblem) -> CopyRefusal {
        CopyRefusal::Invalid(problem)
    }
}

/// What one Metal copy unit is: its bytes, its extent in texels, and the Vulkan aspect it copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CopyUnit {
    pub(super) bytes: u32,
    pub(super) extent: [u32; 2],
    pub(super) aspect: ImageAspectFlags,
}

/// The texel extent of a block of the compressed format `pf` (`None` for an uncompressed
/// format).
///
/// The translator owns the fact ([`metal2vulkan::metal::format::block_extent`]), since it is a
/// property of the `MTLPixelFormat` and not of a copy: this is the name this module reads it by.
pub(super) use metal2vulkan::metal::format::block_extent;

/// The bytes of a block (a texel, uncompressed) of the colour format `pf`: its byte format's
/// (`metal::format::byte_format`, the format whose image holds Metal's bytes), and a PVRTC
/// block's 8, which the device may hold in a carrier of no PVRTC format (`pvrtc.rs`).
pub(super) fn block_bytes(pf: PixelFormat) -> Option<u32> {
    if is_pvrtc(pf) {
        return Some(BLOCK_BYTES);
    }
    // Plain Metal copies a 4:2:2 texture's 2-byte texels at any x, width, offset and pitch (its
    // validation asks for 2 x 1 blocks of 4 bytes), which its carrier holds
    // one to an image texel (`chroma_pairs.rs`).
    if is_four_two_two(pf) {
        return Some(TEXEL_BYTES);
    }
    // A two-plane YCbCr texture's texels are its surface's luma codes, a code's bytes a texel (1
    // for `YCBCR8_420_2P`, 2 for the 10-bit formats), which Metal's blits and CPU calls move
    // (`ycbcr.rs`). A packed one's blits with a buffer move bytes at places that follow no rule
    // found, by the buffer's bytes per row: no block, so they are refused
    // (`NotImplemented(Format)`).
    if let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) {
        return (!codes.packed).then_some(codes.bytes);
    }
    byte_format(pf)
        .and_then(|f| f.block())
        .map(|block| block.size)
}

/// The bytes of the depth Vulkan copies of `format`'s depth aspect, when they are Metal's:
/// a 4-byte float for 32-bit float depth, 2 bytes for 16-bit normalized depth.
pub(super) fn depth_bytes(format: Format) -> Option<u32> {
    match format {
        Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT => Some(4),
        Format::D16_UNORM | Format::D16_UNORM_S8_UINT => Some(2),
        _ => None,
    }
}

/// The unit a depth or stencil texture of `pf` over a plane of a surface with planes keeps in
/// the plane, which its blits with a buffer and its shadow's copies move (`linear_plan.rs`): its
/// depth where the format has one (`Depth16Unorm`'s 2 bytes, a 32-bit float depth's 4), else its
/// stencil (a byte), as Metal's GPU lays them in the plane (an M4 under macOS 27.0.1). `None` for a
/// format of neither aspect or of a 24-bit depth.
pub(super) fn surface_depth_unit(pf: PixelFormat) -> Option<CopyUnit> {
    let held = aspects(pf)?;
    let (aspect, bytes) = match pf {
        PixelFormat::Depth16Unorm => (ImageAspectFlags::DEPTH, 2),
        PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8 => {
            (ImageAspectFlags::DEPTH, 4)
        }
        _ if held == ImageAspectFlags::STENCIL => (ImageAspectFlags::STENCIL, 1),
        _ => return None,
    };
    Some(CopyUnit {
        bytes,
        extent: [1, 1],
        aspect,
    })
}

/// Metal's copy unit of a texture of `pf`, a format that is not depth or stencil, with any blit
/// option (plain Metal ignores one): its texel block, whatever holds it (an image, or a buffer's
/// bytes for a texture buffer). Refused as [`copy_unit`] refuses it.
pub(super) fn colour_unit(pf: PixelFormat) -> Result<CopyUnit, CopyRefusal> {
    let bytes = block_bytes(pf).ok_or(CopyRefusal::Unsupported(
        Unsupported::not_implemented(Feature::Format).naming_format(pf),
    ))?;
    Ok(CopyUnit {
        bytes,
        extent: block_extent(pf).unwrap_or([1, 1]),
        aspect: ImageAspectFlags::COLOR,
    })
}

/// Metal's copy unit of a texture of `pf` held as `mapping`, copied with `option`: plain
/// Metal's aspect, the stencil where the option names it and the format has one, else the depth
/// where the format has one, else the stencil.
///
/// Refusals: `NotImplemented(Format)` for depth whose Metal bytes Vulkan's buffer layout of the
/// image does not give (24-bit depth).
pub(super) fn copy_unit(
    pf: PixelFormat,
    mapping: &FormatMapping,
    option: BlitOption,
) -> Result<CopyUnit, CopyRefusal> {
    if !is_depth_stencil(pf) {
        return colour_unit(pf);
    }
    // A texture holding a depth format's texels in a colour image
    // (`image_plan::holds_depth_in_colour`: one with a depth twin, and a multisampled
    // `Depth32Float` one under `MultisampleDepthTwinPolicy::Exact`) holds them in an `R32_SFLOAT`
    // image, Metal's 4 bytes a texel in its colour aspect; `Depth32Float` has no stencil to name.
    if mapping.aspects == ImageAspectFlags::COLOR {
        if pf != PixelFormat::Depth32Float || mapping.format != Format::R32_SFLOAT {
            return Err(CopyRefusal::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(pf),
            ));
        }
        return Ok(CopyUnit {
            bytes: 4,
            extent: [1, 1],
            aspect: ImageAspectFlags::COLOR,
        });
    }
    // The stencil view of a combined depth/stencil texture (`X32Stencil8` of
    // `Depth32FloatStencil8`) copies the stencil of the image it shares, a byte a texel, whatever
    // the option: plain Metal's answer on an M3 Max and an M4 (a probe; their API validation stops
    // `DepthFromDepthStencil`).
    let both = ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL;
    if aspects(pf) == Some(ImageAspectFlags::STENCIL) && mapping.aspects.contains(both) {
        return Ok(CopyUnit {
            bytes: 1,
            extent: [1, 1],
            aspect: ImageAspectFlags::STENCIL,
        });
    }
    // Plain Metal's aspect (an M4 under macOS 27.0.1; its validation stops
    // all but the named aspect of a combined format): a combined format without an option copies
    // its depth, and an option naming an aspect the format lacks is ignored.
    let has = |a: ImageAspectFlags| mapping.aspects.contains(a);
    let aspect = match option {
        BlitOption::StencilFromDepthStencil if has(ImageAspectFlags::STENCIL) => {
            ImageAspectFlags::STENCIL
        }
        _ if has(ImageAspectFlags::DEPTH) => ImageAspectFlags::DEPTH,
        _ => ImageAspectFlags::STENCIL,
    };
    let bytes = if aspect == ImageAspectFlags::STENCIL {
        1
    } else {
        // Metal's depth bytes are its format's (Depth16Unorm 2, 32-bit float depth 4); the image
        // must hold that depth for Vulkan to copy the same bytes.
        let metal = match pf {
            PixelFormat::Depth16Unorm => Some(2),
            PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8 => Some(4),
            _ => None,
        };
        match (metal, depth_bytes(mapping.format)) {
            (Some(m), Some(v)) if m == v => m,
            _ => {
                return Err(CopyRefusal::Unsupported(
                    Unsupported::not_implemented(Feature::Format).naming_format(pf),
                ))
            }
        }
    };
    Ok(CopyUnit {
        bytes,
        extent: [1, 1],
        aspect,
    })
}

/// A region of one level and slice of a texture, in the texture's own levels and slices
/// (`MTLRegion` with a level and slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TextureRegion {
    pub(super) level: u32,
    pub(super) slice: u32,
    pub(super) origin: [u32; 3],
    pub(super) size: [u32; 3],
}

/// A checked region in the image's levels and layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ImageRegion {
    pub(super) aspect: ImageAspectFlags,
    pub(super) level: u32,
    pub(super) layer: u32,
    pub(super) layers: u32,
    pub(super) origin: [u32; 3],
    pub(super) size: [u32; 3],
}

/// The extent of the record's `level` (counted from its first).
pub(super) fn level_extent(record: &ViewRecord, level: u32) -> [u32; 3] {
    let e = record.extent();
    let mut out = e.map(|axis| (axis >> level.min(31)).max(1));
    if record.texture_type != TextureType::Type3D {
        out[2] = 1;
    }
    out
}

/// Metal's checks of `region` of a texture presenting `record`, copied in units of `unit`; the
/// region in the image's levels and layers, as the blocks it starts in.
///
/// A compressed region off its blocks is copied as plain Metal copies it (an M4 under macOS
/// 27.0.1; its validation stops it): the origin rounded down to its block and
/// the size rounded up to whole blocks, each on its own (an origin of 5 and a size of 4 copy the
/// one block from 4), ending at the level's edge where the blocks pass it. The region's blocks, and
/// so its buffer layout ([`shape`]), are the same.
///
/// A compressed region is bounded by the level's blocks, not its texels: a level that ends inside
/// a block holds that whole block, and a region of it past the level's texels (4 x 4 of a 4 x 2
/// level, or 4 x 2 from y = 2) copies the block, ending at the level's edge, as plain Metal copies
/// it (an Apple M3 Max under macOS 26.7, a probe; Metal's validation takes a whole block of such a
/// level out of a texture, between textures and of the CPU). Its blocks, and so its buffer layout,
/// are the whole block's. Vulkan bounds a copy by the level's texels, hence the edge; between two
/// textures where only one region ends at its level's edge inside a block, the two regions differ
/// in size and cover the same blocks (`blit_texture.rs` copies them through a buffer).
pub(super) fn check_region(
    record: &ViewRecord,
    region: &TextureRegion,
    unit: &CopyUnit,
) -> Result<ImageRegion, CopyProblem> {
    if region.level >= record.level_count || region.slice >= record.layer_count {
        return Err(CopyProblem::Region);
    }
    let extent = level_extent(record, region.level);
    let block = [unit.extent[0], unit.extent[1], 1];
    let blocks = [0, 1, 2].map(|axis| extent[axis].div_ceil(block[axis]) * block[axis]);
    let axes = || region.origin.iter().zip(&region.size).zip(&blocks);
    if axes().any(|((&o, &s), &b)| u64::from(o) + u64::from(s) > u64::from(b)) {
        return Err(CopyProblem::Region);
    }
    let (mut origin, mut size) = (region.origin, region.size);
    for axis in 0..2 {
        let b = block[axis];
        origin[axis] -= origin[axis] % b;
        // Only an empty region starts past the level's edge (at the blocks' end).
        size[axis] = (size[axis].div_ceil(b) * b).min(extent[axis].saturating_sub(origin[axis]));
    }
    Ok(ImageRegion {
        aspect: unit.aspect,
        level: record.base_level + region.level,
        layer: record.base_layer + region.slice,
        layers: 1,
        origin,
        size,
    })
}

/// The rows and images of a region, in copy units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) row_bytes: u64,
    pub(super) rows: u64,
    pub(super) images: u64,
}

pub(super) fn shape(size: [u32; 3], unit: &CopyUnit) -> Shape {
    let blocks = |texels: u32, block: u32| u64::from(texels.div_ceil(block));
    Shape {
        row_bytes: blocks(size[0], unit.extent[0]) * u64::from(unit.bytes),
        rows: blocks(size[1], unit.extent[1]),
        images: u64::from(size[2]),
    }
}

/// A host buffer's layout for `replaceRegion` and `getBytes`, against the tightly packed staging
/// layout the runtime copies through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostLayout {
    pub(super) row_bytes: u64,
    pub(super) rows: u64,
    pub(super) images: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
}

impl HostLayout {
    /// The bytes of the tightly packed staging layout.
    pub(super) fn packed_len(&self) -> u64 {
        self.row_bytes * self.rows * self.images
    }

    /// Whether the host buffer's layout is the tightly packed one: each row at its packed
    /// offset, so the packed bytes are the host buffer's first [`packed_len`](Self::packed_len),
    /// and a copy may read or write them in place.
    pub(super) fn is_packed(&self) -> bool {
        (self.rows <= 1 || self.bytes_per_row == self.row_bytes)
            && (self.images <= 1 || self.bytes_per_image == self.row_bytes * self.rows)
    }

    /// The bytes of the host buffer the layout reads or writes; `None` past `u64::MAX`, more
    /// than any buffer holds. Rows lie in order, so the last ends past every other: a host buffer
    /// of these bytes holds every row [`rows`](Self::rows) gives.
    pub(super) fn host_len(&self) -> Option<u64> {
        if self.row_bytes == 0 || self.rows == 0 || self.images == 0 {
            return Some(0);
        }
        let images = (self.images - 1).checked_mul(self.bytes_per_image)?;
        let rows = (self.rows - 1).checked_mul(self.bytes_per_row)?;
        images.checked_add(rows)?.checked_add(self.row_bytes)
    }

    /// Each row: its offset in the host buffer and in the packed layout, of a layout whose
    /// [`host_len`](Self::host_len) a host buffer holds.
    pub(super) fn rows(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.images).flat_map(move |z| {
            (0..self.rows).map(move |y| {
                (
                    (z * self.bytes_per_image + y * self.bytes_per_row) as usize,
                    ((z * self.rows + y) * self.row_bytes) as usize,
                )
            })
        })
    }

    /// Packs the host bytes `host` (at least [`host_len`](Self::host_len)) into `packed` (at
    /// least [`packed_len`](Self::packed_len)).
    pub(super) fn pack(&self, host: &[u8], packed: &mut [u8]) {
        let n = self.row_bytes as usize;
        for (h, p) in self.rows() {
            packed[p..p + n].copy_from_slice(&host[h..h + n]);
        }
    }

    /// [`pack`](Self::pack) from bytes at a pointer (at least [`host_len`](Self::host_len)),
    /// each row read once, straight into `packed`.
    pub(super) fn pack_from(&self, host: HostBytes<'_>, packed: &mut [u8]) {
        let n = self.row_bytes as usize;
        for (h, p) in self.rows() {
            host.copy_into(h, &mut packed[p..p + n]);
        }
    }

    /// Whether the host bytes `host` of 32-bit float texels (a depth's) hold a nonzero
    /// denormal in a row.
    pub(super) fn holds_denormal(&self, host: &[u8]) -> bool {
        let n = self.row_bytes as usize;
        self.rows().any(|(h, _)| {
            host[h..h + n].as_chunks::<4>().0.iter().any(|word| {
                let bits = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0
            })
        })
    }

    /// Whether the host bytes `host` of 32-bit float texels (a depth's) hold a negative value other
    /// than a zero or a denormal, or a NaN of either sign, in a row.
    pub(super) fn holds_negative_or_nan(&self, host: &[u8]) -> bool {
        let n = self.row_bytes as usize;
        self.rows().any(|(h, _)| {
            host[h..h + n].as_chunks::<4>().0.iter().any(|word| {
                let bits = u32::from_le_bytes(*word);
                let exponent = bits & 0x7f80_0000;
                let nan = exponent == 0x7f80_0000 && bits & 0x007f_ffff != 0;
                nan || (bits & 0x8000_0000 != 0 && exponent != 0)
            })
        })
    }

    /// Whether the host bytes `host` of 32-bit float texels (a depth's) hold, in a row, a word
    /// whose magnitude stand-in (`bind::DepthMagnitudeModule::stand_in`, a denormal's the smallest
    /// normal where `flush`) is not itself: a negative value, -0 included, a NaN, or where `flush`
    /// a denormal; a depth test of the device's compares such a value otherwise than Metal's.
    pub(super) fn holds_unlike_its_stand_in(&self, host: &[u8], flush: bool) -> bool {
        let n = self.row_bytes as usize;
        self.rows().any(|(h, _)| {
            host[h..h + n].as_chunks::<4>().0.iter().any(|word| {
                let bits = u32::from_le_bytes(*word);
                metal2vulkan::bind::DepthMagnitudeModule::stand_in(bits, flush) != bits
            })
        })
    }

    /// Unpacks `packed` into the host bytes `host`, leaving the bytes between rows untouched;
    /// rows that overlap are written in order, image by image and row by row, so the later
    /// row's bytes stay.
    pub(super) fn unpack(&self, packed: &[u8], host: &mut [u8]) {
        let n = self.row_bytes as usize;
        for (h, p) in self.rows() {
            host[h..h + n].copy_from_slice(&packed[p..p + n]);
        }
    }
}

/// The plan of `replaceRegion` or `getBytes` of `region` of a texture presenting `record`, in
/// units of `unit`, with the host's `bytes_per_row` and `bytes_per_image`; `private` for a
/// texture in private storage, `over_buffer` for a texture over a buffer. `None`: the call
/// copies nothing.
///
/// A `bytes_per_row` off Metal's rules is the host's pitch as plain Metal takes it (an M4 under
/// macOS 27.0.1; its validation stops it): any pitch of a row or more exactly, off the copy unit
/// too; 0, every row at the image's first byte (`replaceRegion` reads each row there, `getBytes`
/// writes each there in order, so the last row stays); between 0 and a row, nothing copied,
/// whatever the rows (the AGX driver logs a failed assertion and goes on). Refused (`BytesPerRow`,
/// not measured): a pitch off Metal's rules of a texture over a buffer, and a pitch of 0 for a
/// region of more than one row and more than one image.
///
/// `bytes_per_image` is taken as plain Metal takes it (an M4 under macOS 27.0.1; its validation
/// stops all but a multiple of the copy unit that covers the region's
/// rows): a region of one image ignores it; the images of a region of more than one are exactly
/// `bytes_per_image` apart, off the copy unit and below an image too, and `getBytes` writes them in
/// order, a later image's bytes over an earlier one's ([`HostLayout::unpack`]).
pub(super) fn plan_host_copy(
    record: &ViewRecord,
    region: &TextureRegion,
    unit: &CopyUnit,
    bytes_per_row: u64,
    bytes_per_image: u64,
    private: bool,
    over_buffer: bool,
) -> Result<Option<(ImageRegion, HostLayout)>, CopyProblem> {
    if private {
        return Err(CopyProblem::CpuAccessPrivate);
    }
    let image = check_region(record, region, unit)?;
    let s = shape(region.size, unit);
    let bytes = u64::from(unit.bytes);
    let below_row = bytes_per_row < s.row_bytes;
    let metal_rules = bytes_per_row.is_multiple_of(bytes) && !(s.rows > 1 && below_row);
    if !metal_rules && (over_buffer || (bytes_per_row == 0 && s.images > 1)) {
        return Err(CopyProblem::BytesPerRow);
    }
    if !over_buffer && bytes_per_row != 0 && below_row {
        return Ok(None);
    }
    Ok(Some((
        image,
        HostLayout {
            row_bytes: s.row_bytes,
            rows: s.rows,
            images: s.images,
            bytes_per_row,
            bytes_per_image,
        },
    )))
}
