//! The blit encoder's texture copies: `copyFromBuffer:...toTexture:`,
//! `copyFromTexture:...toBuffer:` and the texture-to-texture copies, as Vulkan copy regions, with
//! Metal's rules as `copy_plan.rs` states them.
//!
//! Vulkan takes a buffer layout in texels per row and rows per image, at offsets aligned to the
//! texel block (and to 4 bytes for a depth/stencil image). A layout Vulkan cannot state that way
//! is copied through a tightly packed staging layout ([`Staging`]).
//!
//! A layout off Metal's rules (an offset or `bytesPerRow` off the copy unit, rows of more than
//! one that overlap, or a `bytesPerImage` of more than one image off the copy unit or below the
//! region's rows) is copied where plain Metal copies it (an M4 under macOS 27.0.1; its validation
//! stops it): each row at `offset + z * bytesPerImage + y * bytesPerRow` rounded down to the copy
//! unit or to 4 bytes, whichever is smaller, and a copy into a texture of one image of more than
//! one row, from offset 0, with `bytesPerRow` rounded down to the copy unit a multiple of 16, at
//! that rounded pitch (Metal's aligned path). Out of a texture, rows and images that overlap are
//! written in turn, a later one's bytes over an earlier one's. A region of one image ignores
//! `bytesPerImage`. [`placed_rows`] states what stays refused.

use super::chroma_pairs::is_four_two_two;
use super::copy_plan::{
    block_bytes, block_extent, check_region, colour_unit, copy_unit, level_extent, shape,
    BlitOption, CopyProblem, CopyRefusal, CopyUnit, ImageRegion, Shape, TextureRegion,
};
use super::error::RuntimeGap;
use super::image_plan::ImagePlan;
use super::linear_plan::{plane_rows_share_bytes, BufferLayout as TexelsLayout};
use super::pvrtc::is_pvrtc;
use super::two_planes;
use super::view_plan::ViewRecord;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{is_depth_stencil, ycbcr};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::ImageAspectFlags;
use std::collections::BTreeMap;
use std::ops::Range;

/// A buffer's side of a copy: `sourceOffset`, `sourceBytesPerRow`, `sourceBytesPerImage` (or the
/// destination's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BufferLayout {
    pub(super) offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) bytes_per_image: u64,
}

impl BufferLayout {
    /// The offset of row `y` of image `z`, `offset + z * bytesPerImage + y * bytesPerRow`; `None`
    /// past `u64::MAX`, where no buffer's byte lies. The layout's values are the caller's, so
    /// every row the runtime places is found through this.
    pub(super) fn row_at(&self, z: u64, y: u64) -> Option<u64> {
        let image = z.checked_mul(self.bytes_per_image)?;
        let row = y.checked_mul(self.bytes_per_row)?;
        self.offset.checked_add(image)?.checked_add(row)
    }

    /// The end of the last row of a region of shape `s`, one past its last byte; `None` past
    /// `u64::MAX`. Rows lie in order, so every row of the region ends there or before.
    fn end(&self, s: &Shape) -> Option<u64> {
        let last = self.row_at(s.images.saturating_sub(1), s.rows.saturating_sub(1))?;
        last.checked_add(s.row_bytes)
    }
}

/// Metal's checks of a buffer layout for a region of `size` in units of `unit`, in a buffer of
/// `buffer_length` bytes; the bytes it spans from its offset. A layout of more than one image
/// reaches it within Metal's `bytesPerImage` rule ([`plan_buffer_copy`] places one off it first;
/// a texture over a buffer, and a depth or stencil format, has one image).
fn check_layout(
    layout: &BufferLayout,
    size: [u32; 3],
    unit: &CopyUnit,
    buffer_length: u64,
) -> Result<u64, CopyProblem> {
    let s = shape(size, unit);
    let bytes = u64::from(unit.bytes);
    if !layout.offset.is_multiple_of(bytes) {
        return Err(CopyProblem::Offset);
    }
    if !layout.bytes_per_row.is_multiple_of(bytes)
        || (s.rows > 1 && layout.bytes_per_row < s.row_bytes)
    {
        return Err(CopyProblem::BytesPerRow);
    }
    if s.row_bytes == 0 || s.rows == 0 || s.images == 0 {
        return Ok(0);
    }
    // A layout whose last row ends past `u64::MAX` runs past every buffer.
    match layout.end(&s) {
        Some(end) if end <= buffer_length => Ok(end - layout.offset),
        _ => Err(CopyProblem::BufferRange),
    }
}

/// One Vulkan buffer-image copy region (`VkBufferImageCopy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BufferImageRegion {
    pub(super) buffer_offset: u64,
    /// Texels per row, or 0 for tightly packed rows.
    pub(super) row_length: u32,
    /// Texel rows per image, or 0 for tightly packed images.
    pub(super) image_height: u32,
    pub(super) image: ImageRegion,
}

/// One buffer-to-buffer copy (`VkBufferCopy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BufferRange {
    pub(super) src: u64,
    pub(super) dst: u64,
    pub(super) size: u64,
}

/// A copy through a tightly packed staging buffer: `rows` move each row (or run of contiguous
/// rows) between the Metal buffer's layout and the staging buffer's, whose `length` bytes the
/// staging buffer holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Staging {
    pub(super) length: u64,
    /// From the Metal buffer to staging for a copy into the texture; the other way for a copy
    /// out of it (`src` is then the staging offset).
    pub(super) rows: Vec<BufferRange>,
}

/// How a buffer-texture copy runs: buffer-image regions on the Metal buffer itself, or on a
/// staging buffer filled from it (or emptied into it). The regions' and rows' offsets in the
/// Metal buffer are its `VkBuffer`'s (from the plan's `base`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BufferCopyPlan {
    pub(super) regions: Vec<BufferImageRegion>,
    pub(super) staging: Option<Staging>,
    /// The Metal buffer's bytes the copy reads or writes, padding between rows included.
    pub(super) span: Range<u64>,
}

/// Whether `layout` is off Metal's rules for a region of shape `s` in `unit`s: an offset or a
/// `bytesPerRow` off the copy unit, rows (of more than one) that overlap, or, of more than one
/// image, a `bytesPerImage` off the copy unit or below the region's rows.
fn off_rules(layout: &BufferLayout, s: &Shape, unit: &CopyUnit) -> bool {
    let bytes = u64::from(unit.bytes);
    // Images below the region's rows: an image of rows past `u64::MAX` is above any.
    let below_rows =
        (layout.bytes_per_row.checked_mul(s.rows)).is_none_or(|rows| layout.bytes_per_image < rows);
    let images_off = s.images > 1 && (!layout.bytes_per_image.is_multiple_of(bytes) || below_rows);
    !layout.offset.is_multiple_of(bytes)
        || !layout.bytes_per_row.is_multiple_of(bytes)
        || (s.rows > 1 && layout.bytes_per_row < s.row_bytes)
        || images_off
}

/// The buffer offset of each row of a region of shape `s` (image by image, row by row) where
/// plain Metal's blit reads it (`into_texture`) or writes it, for a layout off Metal's rules
/// (every offset from 0 to 7 and 4 to 12, every `bytesPerRow` from 0 to 38,
/// units of 1 to 16 bytes, `BC1_RGBA`, `Depth32Float`, and images of a 3D texture).
///
/// Each row is at `offset + z * bytesPerImage + y * bytesPerRow` rounded down to the copy unit or
/// to 4 bytes, whichever is smaller, images of a 3D texture at any `bytesPerImage` too (0, 4, 12
/// and 18 under a 16-byte image). A copy into a texture of one image of
/// more than one row whose `bytesPerRow` rounded down to the copy unit is a multiple of 16 takes
/// Metal's aligned path from offset 0: its rows at that rounded pitch, one below a row too, each
/// row read whole from there so that the rows overlap in the buffer (an Apple M3 Max under macOS
/// 27.0.1, `probes/short_row_blits.m`: every row of a 32 x 8 texture, 64 x 8 for `R8Unorm`, at
/// every pitch of the texel from 16 bytes below the row and at half the row plus a texel, in
/// eight formats of 1 to 16 bytes a texel, `YCBCR8_422_1P` and `YCBCR10_444_1P` among them,
/// `Private` and `Shared`).
///
/// A copy out of a texture at a pitch below a row writes each row at its own offset, the rows
/// overlapping in the buffer: Metal writes them in no one order, and the runtime writes them in
/// turn, the later row's bytes over the earlier's (`plan_placed_copy`), one of Metal's own answers
/// (the same probe: copies out at half the row and below, rows `y x bytesPerRow` apart and nothing
/// past their span written, the later rows' bytes standing in about a fifth of its shapes).
///
/// Refused (`BytesPerRow`): on the aligned path, a pitch of 0 (Metal reads garbage) and an offset
/// of 16 or more (not measured). Refused (`BufferRange`): a row past `u64::MAX`, which lies past
/// every buffer.
fn placed_rows(
    layout: &BufferLayout,
    s: &Shape,
    unit: &CopyUnit,
    into_texture: bool,
) -> Result<Vec<u64>, CopyProblem> {
    let bytes = u64::from(unit.bytes);
    let word = bytes.min(4);
    let bpr = layout.bytes_per_row;
    let pitch = bpr - bpr % bytes;
    let aligned = into_texture && s.images == 1 && s.rows > 1 && pitch.is_multiple_of(16);
    if aligned && layout.offset.is_multiple_of(16) {
        if layout.offset != 0 || pitch == 0 {
            return Err(CopyProblem::BytesPerRow);
        }
        return (0..s.rows)
            .map(|y| y.checked_mul(pitch).ok_or(CopyProblem::BufferRange))
            .collect();
    }
    let mut rows = Vec::with_capacity((s.images * s.rows) as usize);
    for z in 0..s.images {
        for y in 0..s.rows {
            let at = layout.row_at(z, y).ok_or(CopyProblem::BufferRange)?;
            rows.push(at - at % word);
        }
    }
    Ok(rows)
}

/// Whether a layout off Metal's rules copies `pf` where plain Metal places its rows
/// ([`placed_rows`]): a colour format or `Depth32Float` (other depth and stencil formats are not
/// measured, and keep Metal's validation's refusal, but for [`rows_below_row`]).
fn placed(pf: PixelFormat) -> bool {
    !is_depth_stencil(pf) || pf == PixelFormat::Depth32Float
}

/// Whether `layout` is off Metal's rules for a region of shape `s` in `unit`s by a `bytesPerRow`
/// below the region's row alone: rows of more than one, of one image, from an offset on the copy
/// unit, at a nonzero pitch on the copy unit below the row. Plain Metal's blit copies such a
/// layout of each kind of single-sample texture measured, each row whole at
/// `offset + y x bytesPerRow`: read so into the texture, and written so out of it, in no one
/// order where the rows overlap (an Apple M3 Max under macOS 27.0.1, `probes/short_row_kinds.m`:
/// `Depth16Unorm`, `Stencil8`, both aspects of `Depth32Float_Stencil8`, an `RGBA8Unorm` texture
/// over a buffer and `RGB10A8_2P_XR10`, 32 x 8, from offset 0 at half the row, half plus a unit
/// and the row less 16, both ways, twice; the colour formats' rows lie so from other offsets on
/// the unit too, [`placed_rows`]). So a depth or stencil format copies such a layout where
/// [`placed_rows`] places it, and a texture over a buffer as [`plan_rows_copy`] does.
pub(super) fn rows_below_row(layout: &BufferLayout, s: &Shape, unit: &CopyUnit) -> bool {
    let bytes = u64::from(unit.bytes);
    let bpr = layout.bytes_per_row;
    s.rows > 1
        && s.images == 1
        && bpr > 0
        && bpr < s.row_bytes
        && bpr.is_multiple_of(bytes)
        && layout.offset.is_multiple_of(bytes)
}

/// The rows of a region of shape `s` laid out as `layout`, image by image and row by row, each
/// `s.row_bytes` long; `None` where one ends past `u64::MAX`.
fn layout_rows(layout: &BufferLayout, s: &Shape) -> Option<Vec<Range<u64>>> {
    let mut rows = Vec::with_capacity((s.images * s.rows) as usize);
    for z in 0..s.images {
        for y in 0..s.rows {
            let start = layout.row_at(z, y)?;
            rows.push(start..start.checked_add(s.row_bytes)?);
        }
    }
    Some(rows)
}

/// Whether Vulkan takes `offset` as a buffer-image copy's buffer offset for `unit` (a multiple of
/// the texel block, and of 4 for a depth/stencil aspect).
fn vulkan_offset(offset: u64, unit: &CopyUnit) -> bool {
    let align = if unit.aspect == ImageAspectFlags::COLOR {
        u64::from(unit.bytes)
    } else {
        4
    };
    offset.is_multiple_of(align)
}

/// The plan of a copy between a buffer of `buffer_length` bytes laid out as `layout` and
/// `region` of a texture presenting `record`, in units of `unit` (`into_texture` from the buffer
/// into the texture; either way the plan's staging rows run buffer to staging, and the caller
/// reverses them for a copy out of the texture). The buffer's bytes start `base` bytes into its
/// `VkBuffer` (`Buffer::at`): Metal's rules hold of the layout, Vulkan's of the `VkBuffer`'s
/// offsets, so a layout Metal takes is staged where its offset there misses the texel block. A
/// layout off Metal's rules is staged where plain Metal places its rows ([`placed_rows`]), for a
/// colour format or `Depth32Float` (other depth and stencil formats are not measured, and keep
/// Metal's validation's refusal).
pub(super) fn plan_buffer_copy(
    record: &ViewRecord,
    region: &TextureRegion,
    unit: &CopyUnit,
    layout: &BufferLayout,
    buffer_length: u64,
    base: u64,
    into_texture: bool,
) -> Result<BufferCopyPlan, CopyProblem> {
    let image = check_region(record, region, unit)?;
    let s = shape(region.size, unit);
    let placed = placed(record.pixel_format) || rows_below_row(layout, &s, unit);
    if placed && off_rules(layout, &s, unit) {
        return plan_placed_copy(layout, &s, unit, image, buffer_length, base, into_texture);
    }
    // Every row ends within the buffer (`check_layout`), so no offset below passes `u64::MAX`.
    let span = check_layout(layout, region.size, unit, buffer_length)?;
    let span = layout.offset..layout.offset + span;
    let start = base + layout.offset;
    if s.row_bytes == 0 || s.rows == 0 || s.images == 0 {
        return Ok(BufferCopyPlan {
            regions: Vec::new(),
            staging: None,
            span,
        });
    }
    let texels_per_row = |bytes_per_row: u64| {
        let texels = (bytes_per_row / u64::from(unit.bytes)).checked_mul(u64::from(unit.extent[0]));
        texels.and_then(|texels| u32::try_from(texels).ok())
    };
    // Rows: the pitch in texels, or tight when there is one row.
    let row_length = if s.rows == 1 {
        Some(0)
    } else {
        texels_per_row(layout.bytes_per_row)
    };
    let images_fit = s.images == 1
        || (layout.bytes_per_row != 0
            && layout.bytes_per_image.is_multiple_of(layout.bytes_per_row));
    let image_height = if s.images == 1 {
        Some(0)
    } else if images_fit {
        let rows = layout.bytes_per_image / layout.bytes_per_row;
        (rows.checked_mul(u64::from(unit.extent[1]))).and_then(|rows| u32::try_from(rows).ok())
    } else {
        None
    };
    if let (Some(row_length), true) = (row_length, vulkan_offset(start, unit)) {
        if let Some(image_height) = image_height {
            return Ok(BufferCopyPlan {
                regions: vec![BufferImageRegion {
                    buffer_offset: start,
                    row_length,
                    image_height,
                    image,
                }],
                staging: None,
                span,
            });
        }
        // One region per image, where every image's offset is one Vulkan takes.
        if vulkan_offset(layout.bytes_per_image, unit) {
            let regions = (0..s.images)
                .map(|z| BufferImageRegion {
                    buffer_offset: start + z * layout.bytes_per_image,
                    row_length,
                    image_height: 0,
                    image: ImageRegion {
                        origin: [image.origin[0], image.origin[1], image.origin[2] + z as u32],
                        size: [image.size[0], image.size[1], 1],
                        ..image
                    },
                })
                .collect();
            return Ok(BufferCopyPlan {
                regions,
                staging: None,
                span,
            });
        }
    }
    // A tightly packed staging layout: rows of `row_bytes`, images of `rows` rows.
    let mut rows = Vec::new();
    for z in 0..s.images {
        let src_image = start + z * layout.bytes_per_image;
        let dst_image = z * s.rows * s.row_bytes;
        if layout.bytes_per_row == s.row_bytes || s.rows == 1 {
            rows.push(BufferRange {
                src: src_image,
                dst: dst_image,
                size: s.rows * s.row_bytes,
            });
        } else {
            rows.extend((0..s.rows).map(|y| BufferRange {
                src: src_image + y * layout.bytes_per_row,
                dst: dst_image + y * s.row_bytes,
                size: s.row_bytes,
            }));
        }
    }
    Ok(BufferCopyPlan {
        regions: vec![BufferImageRegion {
            buffer_offset: 0,
            row_length: 0,
            image_height: 0,
            image,
        }],
        staging: Some(Staging {
            length: s.images * s.rows * s.row_bytes,
            rows,
        }),
        span,
    })
}

/// A copy of a layout off Metal's rules (shape `s`, of the checked `image`) through a tightly
/// packed staging layout, each row where plain Metal places it ([`placed_rows`]); the rows stay
/// in the buffer. Out of the texture, rows that overlap are written in turn: images closer than
/// the region's rows as plain Metal writes them (in order in every run of 3D textures up to
/// 256x64x4), and rows closer than a row, which Metal writes in no one order, as one of its
/// answers (the later row's bytes standing). Each row's bytes that a later row writes are left
/// out of it ([`later_rows_win`]), so the regions of the one copy into the buffer are disjoint.
fn plan_placed_copy(
    layout: &BufferLayout,
    s: &Shape,
    unit: &CopyUnit,
    image: ImageRegion,
    buffer_length: u64,
    base: u64,
    into_texture: bool,
) -> Result<BufferCopyPlan, CopyProblem> {
    let starts = placed_rows(layout, s, unit, into_texture)?;
    let (Some(&first), Some(&last)) = (starts.iter().min(), starts.iter().max()) else {
        return Ok(BufferCopyPlan {
            regions: Vec::new(),
            staging: None,
            span: layout.offset..layout.offset,
        });
    };
    let end = (last.checked_add(s.row_bytes))
        .filter(|&end| end <= buffer_length)
        .ok_or(CopyProblem::BufferRange)?;
    let rows: Vec<BufferRange> = starts
        .iter()
        .enumerate()
        .map(|(i, &start)| BufferRange {
            src: base + start,
            dst: i as u64 * s.row_bytes,
            size: s.row_bytes,
        })
        .collect();
    let rows = if into_texture {
        rows
    } else {
        later_rows_win(&rows)
    };
    Ok(BufferCopyPlan {
        regions: vec![BufferImageRegion {
            buffer_offset: 0,
            row_length: 0,
            image_height: 0,
            image,
        }],
        staging: Some(Staging {
            length: s.images * s.rows * s.row_bytes,
            rows,
        }),
        span: first..end,
    })
}

/// `rows` (from the Metal buffer at `src`, in the order Metal writes them) without the bytes a
/// later row writes: each row's pieces that no later row covers, so the regions of one
/// `vkCmdCopyBuffer` into the buffer are disjoint and the buffer ends as Metal's in-order writes
/// leave it. A row wholly covered by later ones copies nothing.
pub(super) fn later_rows_win(rows: &[BufferRange]) -> Vec<BufferRange> {
    // The buffer's bytes later rows write: disjoint ranges, keyed by start.
    let mut covered: BTreeMap<u64, u64> = BTreeMap::new();
    let mut kept = Vec::with_capacity(rows.len());
    for row in rows.iter().rev() {
        let (start, end) = (row.src, row.src + row.size);
        // The covered ranges that meet or touch the row, in order.
        let before = covered
            .range(..=start)
            .next_back()
            .filter(|&(_, &e)| e >= start)
            .map(|(&s, &e)| (s, e));
        let within: Vec<(u64, u64)> = covered
            .range(start + 1..end)
            .map(|(&s, &e)| (s, e))
            .collect();
        let mut at = start;
        for &(s, e) in before.iter().chain(&within) {
            if s > at {
                kept.push(BufferRange {
                    src: at,
                    dst: row.dst + (at - start),
                    size: s - at,
                });
            }
            at = at.max(e);
        }
        if at < end {
            kept.push(BufferRange {
                src: at,
                dst: row.dst + (at - start),
                size: end - at,
            });
        }
        let merged_start = before.map_or(start, |(s, _)| s);
        let merged_end = before
            .iter()
            .chain(&within)
            .map(|&(_, e)| e)
            .fold(end, u64::max);
        for (s, _) in before.iter().chain(&within) {
            covered.remove(s);
        }
        covered.insert(merged_start, merged_end);
    }
    kept.reverse();
    kept
}

/// One Vulkan image-image copy region (`VkImageCopy`). The two regions cover the same blocks; a
/// compressed pair differs in size where one ends at its level's edge inside a block and the
/// other does not (a whole block of a 4 x 2 level and of a larger one). No one extent states
/// such a pair: Vulkan bounds each region by its level's texels and asks each for whole blocks
/// except at its image's edge, so 4 x 2 is a part block inside the larger level and 4 x 4 passes
/// the smaller (`blit_texture.rs` copies such a pair through a buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ImageCopyRegion {
    pub(super) src: ImageRegion,
    pub(super) dst: ImageRegion,
}

/// Metal's copy unit of `pf` between textures, as bytes and a block extent: its texel block; of a
/// depth-only or stencil-only format, or a stencil view, its one aspect (`Depth16Unorm` 2 bytes,
/// `Depth32Float` 4, `Stencil8` and `X32Stencil8` 1). `None` for a combined depth/stencil format,
/// which copies only into its own format.
fn texture_unit(pf: PixelFormat) -> Option<(u32, [u32; 2])> {
    use PixelFormat as P;
    match pf {
        P::Depth16Unorm => Some((2, [1, 1])),
        P::Depth32Float => Some((4, [1, 1])),
        P::Stencil8 | P::X32Stencil8 => Some((1, [1, 1])),
        _ if is_depth_stencil(pf) => None,
        _ => block_bytes(pf).map(|bytes| (bytes, block_extent(pf).unwrap_or([1, 1]))),
    }
}

/// Whether Metal copies between textures of `a` and `b` (macOS 26.7's rule): the same format, or
/// two formats of one copy unit, whose bytes it copies, a depth or stencil format's among them
/// (probes), or a compressed format and an uncompressed one whose texel is the bytes of its block
/// (each block into one texel and back, [`source_region`]); formats of units of different bytes
/// are refused.
///
/// The runtime's gaps: a PVRTC format with an uncompressed one, whose blocks plain Metal moves in
/// an order of its own (our probe on an Apple M3 Max under macOS 27.0.1: neither the rows nor the
/// CPU's Z order), and two compressed formats of one block's bytes and different extents (ASTC
/// 6x5 and BC7), not measured.
fn check_formats(a: PixelFormat, b: PixelFormat) -> Result<(), CopyRefusal> {
    // Between two `YCBCR8_420_2P` textures Metal copies both of the surfaces' planes, where the
    // runtime's copy moves the luma bytes its texels are: the runtime's
    // gap, as is a copy between one and another format, which is not measured.
    if ycbcr(a) || ycbcr(b) {
        return Err(CopyRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(a),
        ));
    }
    if a == b {
        return Ok(());
    }
    let not_implemented =
        || CopyRefusal::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(a));
    match (texture_unit(a), texture_unit(b)) {
        (Some(x), Some(y)) if x == y => {}
        (Some((x, from)), Some((y, to))) if x == y => {
            let texel = |extent: [u32; 2]| extent == [1, 1];
            if texel(from) == texel(to) || is_pvrtc(a) || is_pvrtc(b) {
                return Err(not_implemented());
            }
        }
        _ => return Err(CopyProblem::Format.into()),
    }
    // Into a 4:2:2 texture from another format of its unit plain Metal writes `0xff` over part
    // of each row (`RG8Unorm` into `GBGR422`), no copy of the bytes: the
    // runtime's gap. Out of one, and between the two orders, it copies them.
    if is_four_two_two(b) && !is_four_two_two(a) {
        return Err(CopyRefusal::Unsupported(Unsupported::not_implemented(
            Feature::Format,
        )));
    }
    Ok(())
}

/// The two textures of a texture-to-texture copy: what each presents and its image, `None` for
/// a texture buffer, which has none (its format is a colour format: its copy unit is its
/// texel).
#[derive(Clone, Copy, Debug)]
pub(super) struct CopyPair<'a> {
    pub(super) src: &'a ViewRecord,
    pub(super) src_image: Option<&'a ImagePlan>,
    pub(super) dst: &'a ViewRecord,
    pub(super) dst_image: Option<&'a ImagePlan>,
}

impl CopyPair<'_> {
    /// Metal's checks of the pair as a whole, and each texture's copy unit: one size, in each
    /// texture's own aspect (a depth or stencil format's, else colour).
    fn check(&self) -> Result<(CopyUnit, CopyUnit), CopyRefusal> {
        if self.src.samples != self.dst.samples {
            return Err(CopyProblem::SampleCount.into());
        }
        // A two-plane extended-range texture copies both words of its texels into another of the
        // pair, which plain Metal copies into (its validation stops the other format); a copy
        // between one and a texture of another format is not measured (`two_planes.rs`).
        let (src, dst) = (self.src.pixel_format, self.dst.pixel_format);
        match (two_planes::holds_planes(src), two_planes::holds_planes(dst)) {
            (true, true) => return Ok((two_planes::TEXEL_UNIT, two_planes::TEXEL_UNIT)),
            (true, false) => return Err(CopyRefusal::Unsupported(two_planes::unmeasured(src))),
            (false, true) => return Err(CopyRefusal::Unsupported(two_planes::unmeasured(dst))),
            (false, false) => {}
        }
        check_formats(src, dst)?;
        Ok((
            pair_unit(self.src, self.src_image)?,
            pair_unit(self.dst, self.dst_image)?,
        ))
    }
}

/// The copy unit of one texture of a texture-to-texture copy: a combined depth/stencil format's
/// both aspects (it copies only into its own format), else its unit as a buffer copy without an
/// option takes it.
fn pair_unit(record: &ViewRecord, image: Option<&ImagePlan>) -> Result<CopyUnit, CopyRefusal> {
    let Some(image) = image else {
        return colour_unit(record.pixel_format);
    };
    let both = ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL;
    if image.mapping.aspects.contains(both) && record.pixel_format == image.pixel_format {
        let block = image
            .mapping
            .format
            .block()
            .ok_or(CopyRefusal::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(record.pixel_format),
            ))?;
        return Ok(CopyUnit {
            bytes: block.size,
            extent: [1, 1],
            aspect: both,
        });
    }
    copy_unit(record.pixel_format, &image.mapping, BlitOption::None)
}

/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:...`: `region` of
/// the source to the same size at `dst_origin` of `dst_level` and `dst_slice` of the destination,
/// each side as the blocks it starts in ([`check_region`]). Where those end at one side's level
/// edge inside a block and not at the other's, the sides differ in size and cover the same blocks,
/// which plain Metal copies whole (an Apple M3 Max under macOS 26.7, a probe: a whole block of a
/// 4 x 2 level into one of an 8 x 8 level and back, which its validation takes, and 4 x 2 of
/// either, which it stops). Between a compressed and an uncompressed format the region is in the
/// destination's texels ([`source_region`]).
pub(super) fn plan_region_copy(
    pair: &CopyPair<'_>,
    region: &TextureRegion,
    dst_level: u32,
    dst_slice: u32,
    dst_origin: [u32; 3],
) -> Result<Vec<ImageCopyRegion>, CopyRefusal> {
    let (src_unit, dst_unit) = pair.check()?;
    let source = source_region(region, &src_unit, &dst_unit)?;
    let src = check_region(pair.src, &source, &src_unit)?;
    let dst_region = TextureRegion {
        level: dst_level,
        slice: dst_slice,
        origin: dst_origin,
        size: region.size,
    };
    let dst = check_region(pair.dst, &dst_region, &dst_unit)?;
    if region.size.contains(&0) {
        return Ok(Vec::new());
    }
    Ok(vec![ImageCopyRegion { src, dst }])
}

/// The source's region of a region copy of `region` between textures of units `src` and `dst`:
/// `region` itself between units of one extent. Between a compressed format and an uncompressed
/// one (`check_formats`) every coordinate of the copy is in the destination's texels, the source
/// addressed as if it held the destination's format (our probe on an Apple M3 Max under macOS
/// 27.0.1: BC1 with `RG32Uint` and `RGBA16Uint`, BC7, `EAC_RGBA8` and ASTC 6x5 with
/// `RGBA32Uint`, `ETC2_RGB8` with `RG32Uint`, both ways): out of a compressed texture the origin
/// and size count its blocks (an origin of 4 is the fifth block), into one they count the
/// source's texels as the destination's, the origin rounded down to a block's texel and the size
/// up to whole blocks, each on its own (an origin of 1 and a size of 4 copy texel 0 alone).
/// Refused (`Region`) where the source's texels overflow.
fn source_region(
    region: &TextureRegion,
    src: &CopyUnit,
    dst: &CopyUnit,
) -> Result<TextureRegion, CopyProblem> {
    let mut out = *region;
    for axis in 0..2 {
        let (from, to) = (src.extent[axis], dst.extent[axis]);
        if from == to {
            continue;
        }
        let (origin, size) = (region.origin[axis], region.size[axis]);
        (out.origin[axis], out.size[axis]) = match to {
            1 => (
                origin.checked_mul(from).ok_or(CopyProblem::Region)?,
                size.checked_mul(from).ok_or(CopyProblem::Region)?,
            ),
            _ => (origin / to, size.div_ceil(to)),
        };
    }
    Ok(out)
}

/// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:
/// sliceCount:levelCount:`, and `copyFromTexture:toTexture:` ([`plan_whole_copy`]: every level
/// and slice two textures share). Each level pair copies the copy units both levels hold, from
/// their origins. Between a compressed and an uncompressed format (`check_formats`) the runtime's
/// gap (`NotImplemented(Format)`): which level's blocks Metal copies into which level's texels is
/// not measured.
pub(super) fn plan_levels_copy(
    pair: &CopyPair<'_>,
    src_slice: u32,
    src_level: u32,
    dst_slice: u32,
    dst_level: u32,
    slices: u32,
    levels: u32,
) -> Result<Vec<ImageCopyRegion>, CopyRefusal> {
    let (src_unit, dst_unit) = pair.check()?;
    if src_unit.extent != dst_unit.extent {
        return Err(CopyRefusal::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(pair.src.pixel_format),
        ));
    }
    let (src, dst) = (pair.src, pair.dst);
    let in_range = |start: u32, count: u32, limit: u32| {
        start.checked_add(count).is_some_and(|end| end <= limit)
    };
    if !in_range(src_level, levels, src.level_count)
        || !in_range(dst_level, levels, dst.level_count)
        || !in_range(src_slice, slices, src.layer_count)
        || !in_range(dst_slice, slices, dst.layer_count)
    {
        return Err(CopyProblem::Region.into());
    }
    let mut out = Vec::with_capacity(levels as usize);
    for i in 0..levels {
        let (from, to) = (
            level_extent(src, src_level + i),
            level_extent(dst, dst_level + i),
        );
        // Levels of different sizes copy the copy units both levels hold, from their origins,
        // whole: the lesser count of units on each axis, and the rest of the larger level kept,
        // as plain Metal copies them. Of a compressed format that is the lesser count of blocks,
        // the block that passes the smaller level's edge included (an Apple M3 Max under macOS
        // 27.0.1, our probe: BC1, BC3, BC4, BC7, `EAC_RGBA8`, `ETC2_RGB8` and ASTC 6 x 5, each
        // into a level of fewer blocks, of more, and of fewer on one axis and more on the other,
        // and out of a level that ends inside a block, every one block for block). Vulkan bounds
        // a copy by the level's texels, so where the last unit passes the smaller level's edge
        // the two sides differ in size and cover the same units, which `blit_texture.rs` copies
        // through a buffer.
        let block = [src_unit.extent[0], src_unit.extent[1], 1];
        let units = [0, 1, 2].map(|axis| {
            from[axis]
                .div_ceil(block[axis])
                .min(to[axis].div_ceil(block[axis]))
        });
        let whole = |extent: [u32; 3]| {
            [0, 1, 2].map(|axis| units[axis].saturating_mul(block[axis]).min(extent[axis]))
        };
        let (src_size, dst_size) = (whole(from), whole(to));
        if slices == 0 {
            continue;
        }
        let at = |record: &ViewRecord, unit: &CopyUnit, level: u32, slice: u32, size: [u32; 3]| {
            ImageRegion {
                aspect: unit.aspect,
                level: record.base_level + level,
                layer: record.base_layer + slice,
                layers: slices,
                origin: [0, 0, 0],
                size,
            }
        };
        out.push(ImageCopyRegion {
            src: at(src, &src_unit, src_level + i, src_slice, src_size),
            dst: at(dst, &dst_unit, dst_level + i, dst_slice, dst_size),
        });
    }
    Ok(out)
}

/// `copyFromTexture:toTexture:`: the levels the two textures share, paired by extent
/// ([`aligned_levels`]), and the slices they share, as plain Metal copies them (an M4 under macOS
/// 27.0.1): an 8x8 texture of 4 levels into a 4x4 one of 3 copies its levels 1 to 3, a 2D texture
/// into a 2D array its one slice. The paired levels may lie past the textures' own, since Metal
/// pairs them by the textures' extents shifted and not by their level counts, and then each side
/// copies the levels it has from its paired one: none, where that side's paired level is the one
/// past its last (an Apple M3 Max under macOS 27.0.1, our probe `wh-bc1-16-to-8`,
/// `wh-plain-16-to-8` and `wh-bc1-16-to-32`: a 16 by 16 texture of one level into an 8 by 8 one
/// of one, and into a 32 by 32 one of one, each complete with no error and the destination's own
/// bytes kept).
///
/// Refused (`Size`): two textures whose extents pair at no level, where plain Metal aborts, and a
/// paired level past the end of its texture's own levels, where plain Metal reads or writes a
/// level the texture does not hold.
pub(super) fn plan_whole_copy(pair: &CopyPair<'_>) -> Result<Vec<ImageCopyRegion>, CopyRefusal> {
    let (src, dst) = (pair.src, pair.dst);
    let (src_level, dst_level) = aligned_levels(src, dst).ok_or(CopyProblem::Size)?;
    if src_level > src.level_count || dst_level > dst.level_count {
        return Err(CopyProblem::Size.into());
    }
    let levels = (src.level_count - src_level).min(dst.level_count - dst_level);
    let slices = src.layer_count.min(dst.layer_count);
    plan_levels_copy(pair, 0, src_level, 0, dst_level, slices, levels)
}

/// The first levels a whole copy pairs: both first levels where they are one extent, else the
/// source's level `k` and the destination's first, or the source's first and the destination's
/// level `k`, at the least `k` where every axis of the two extents is equal. Plain Metal pairs
/// them so (`sz_whole_8l4_to_4l3` pairs the source's level 1 with the destination's 0,
/// `sz_whole_4l3_to_8l4` the source's 0 with the destination's 1, `sz_whole_6l3_to_3l2` and
/// `sz_whole_5l3_to_2l2` its level 1 with 0), and aborts where the extents meet at no level
/// (`sz_whole_8_to_3`, `sz_whole_8x4_to_4x8`, `sz_whole_8x4_to_4x4`). `None` there, and where
/// both directions would pair (unmeasured).
///
/// The shifts a pair may take are the extent's own, which the texture's level count does not
/// bound: Metal shifts the texture's width, height and depth and asks that one of the two shifts
/// be none (`MTLPickLargestMip`'s postcondition, which its abort names), so a 16 by 16 texture of
/// one level pairs its level 1 with an 8 by 8 destination's level 0 and copies no level at all
/// (our probe's `wh-bc1-16-to-8`). An extent is 1 on every axis by the 31st shift, so the search
/// ends there.
fn aligned_levels(src: &ViewRecord, dst: &ViewRecord) -> Option<(u32, u32)> {
    if level_extent(src, 0) == level_extent(dst, 0) {
        return Some((0, 0));
    }
    let first = |of: &ViewRecord, to: [u32; 3]| (1..32).find(|&k| level_extent(of, k) == to);
    match (
        first(src, level_extent(dst, 0)),
        first(dst, level_extent(src, 0)),
    ) {
        (Some(k), None) => Some((k, 0)),
        (None, Some(k)) => Some((0, k)),
        _ => None,
    }
}

/// A copy of rows of bytes, each from `src`'s row to `dst`'s of the same index: a blit between a
/// buffer's bytes and a texture over a buffer, or between two textures over buffers, which runs on
/// the buffers' bytes at Metal's layout. Offsets are the buffers' own (from their first byte), and
/// rows contiguous in both are one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RowsCopy {
    pub(super) rows: Vec<BufferRange>,
    /// The source's bytes from its first row's start to its last row's end.
    pub(super) src_span: Range<u64>,
    /// The destination's, likewise.
    pub(super) dst_span: Range<u64>,
}

impl RowsCopy {
    /// Each row of `src` into the same row of `dst` (as many rows, each as long). Destination
    /// rows overlap in two shapes alone, both a buffer's: rows whose images meet, at a
    /// `bytesPerImage` below the region's rows, which `check_layout` takes as plain Metal does,
    /// and rows at a `bytesPerRow` below the region's row ([`plan_rows_copy`]). There the later
    /// row keeps the bytes they share, as plain Metal writes overlapping images in turn (an M4
    /// under macOS 27.0.1, the `ctb.race.*` cases) and as one of the orders it writes such rows
    /// in ([`plan_placed_copy`] states the same rule for a layout off Metal's rules): an earlier
    /// row copies only the bytes no later row writes ([`later_rows_win`]). No texture's own rows
    /// overlap as the destination: a plane's shared
    /// bytes, which have no one Metal order, are refused wherever they would be written
    /// (`blit_texture.rs`), and a rotated texture's rows share no byte
    /// (`buffer_texture::turned_plane`). A plane's rows do overlap as the source, where they are
    /// read: each destination row takes the plane's bytes at its own GPU row, so a shared byte is
    /// read once for each row that holds it, and such a row joins neither neighbour into one
    /// range.
    pub(super) fn new(src: &[Range<u64>], dst: &[Range<u64>]) -> RowsCopy {
        debug_assert_eq!(src.len(), dst.len());
        // Each row as `later_rows_win` takes it: by the bytes it writes.
        let mut written: Vec<BufferRange> = src
            .iter()
            .zip(dst)
            .map(|(s, d)| {
                debug_assert_eq!(s.end - s.start, d.end - d.start);
                BufferRange {
                    src: d.start,
                    dst: s.start,
                    size: d.end - d.start,
                }
            })
            .collect();
        if !dst.windows(2).all(|two| two[0].end <= two[1].start) {
            written = later_rows_win(&written);
            written.sort_by_key(|row| row.src);
        }
        let mut rows: Vec<BufferRange> = Vec::with_capacity(written.len());
        for row in written {
            let (s, d) = (row.dst..row.dst + row.size, row.src..row.src + row.size);
            match rows.last_mut() {
                Some(last)
                    if last.src + last.size == s.start && last.dst + last.size == d.start =>
                {
                    last.size += s.end - s.start;
                }
                _ => rows.push(BufferRange {
                    src: s.start,
                    dst: d.start,
                    size: s.end - s.start,
                }),
            }
        }
        let span = |r: &[Range<u64>]| {
            let start = r.iter().map(|r| r.start).min().unwrap_or(0);
            start..r.iter().map(|r| r.end).max().unwrap_or(start)
        };
        RowsCopy {
            rows,
            src_span: span(src),
            dst_span: span(dst),
        }
    }

    /// Whether a source row and a destination row share a byte, for a copy within one buffer:
    /// Vulkan's copy forbids it, where Metal moves the bytes as a copy through a temporary does (a
    /// Metal probe on the M3 Max and an M4).
    pub(super) fn meets(&self) -> bool {
        let mut src: Vec<Range<u64>> = self.rows.iter().map(|r| r.src..r.src + r.size).collect();
        let mut dst: Vec<Range<u64>> = self.rows.iter().map(|r| r.dst..r.dst + r.size).collect();
        src.sort_by_key(|r| r.start);
        dst.sort_by_key(|r| r.start);
        let (mut i, mut j) = (0, 0);
        while i < src.len() && j < dst.len() {
            if src[i].start < dst[j].end && dst[j].start < src[i].end {
                return true;
            }
            if src[i].end <= dst[j].end {
                i += 1;
            } else {
                j += 1;
            }
        }
        false
    }

    /// The bytes the rows hold.
    pub(super) fn length(&self) -> u64 {
        self.rows.iter().map(|r| r.size).sum()
    }

    /// This copy cut where a side's bytes pass `src_cut` or `dst_cut` (a texture's whose texels
    /// run past its buffer's length, whose bytes past it lie in its tail, `buffer_texture.rs`):
    /// up to four copies, each `(whether its source rows lie past the source's cut, whether its
    /// destination rows lie past the destination's, the copy)`, every row in exactly one of
    /// them, cut where it crosses either cut. A side with no cut lies before it whole.
    pub(super) fn split(
        &self,
        src_cut: Option<u64>,
        dst_cut: Option<u64>,
    ) -> Vec<(bool, bool, RowsCopy)> {
        let (src_cut, dst_cut) = (src_cut.unwrap_or(u64::MAX), dst_cut.unwrap_or(u64::MAX));
        let mut parts: Vec<(bool, bool, Vec<BufferRange>)> = Vec::new();
        for row in &self.rows {
            // The row's cuts, from its start, where either side's bytes pass their cut.
            let mut cuts = vec![0, row.size];
            for (start, cut) in [(row.src, src_cut), (row.dst, dst_cut)] {
                if start < cut && cut < start + row.size {
                    cuts.push(cut - start);
                }
            }
            cuts.sort_unstable();
            cuts.dedup();
            for piece in cuts.windows(2) {
                let part = BufferRange {
                    src: row.src + piece[0],
                    dst: row.dst + piece[0],
                    size: piece[1] - piece[0],
                };
                let side = (part.src >= src_cut, part.dst >= dst_cut);
                match parts.iter_mut().find(|(s, d, _)| (*s, *d) == side) {
                    Some((.., rows)) => rows.push(part),
                    None => parts.push((side.0, side.1, vec![part])),
                }
            }
        }
        parts
            .into_iter()
            .map(|(src_past, dst_past, rows)| {
                let span = |at: fn(&BufferRange) -> u64| {
                    let start = rows.iter().map(at).min().unwrap_or(0);
                    let end = rows.iter().map(|r| at(r) + r.size).max().unwrap_or(start);
                    start..end
                };
                let (src_span, dst_span) = (span(|r| r.src), span(|r| r.dst));
                let copy = RowsCopy {
                    rows,
                    src_span,
                    dst_span,
                };
                (src_past, dst_past, copy)
            })
            .collect()
    }
}

/// The rows of `region` (a checked region, in `unit`s of one texel or one block) of a texture
/// over a buffer whose texels lie at `texels` (Metal's layout): one range of the buffer's bytes
/// per row, slice by slice. A block-compressed texture's rows are rows of its blocks (a BC texture
/// over an IOSurface plane's rows, `linear_plan.rs`), the region's texels counted in blocks.
pub(super) fn texel_rows(
    texels: &TexelsLayout,
    unit: &CopyUnit,
    region: &ImageRegion,
) -> Vec<Range<u64>> {
    let bytes = u64::from(unit.bytes);
    let ([x, y], [across, down]) = in_units(unit, region);
    let width = across * bytes;
    let mut rows = Vec::with_capacity(region.layers as usize * down as usize);
    for layer in region.layer..region.layer + region.layers {
        for row in y..y + down {
            let start =
                texels.offset + u64::from(layer) * texels.slice_pitch + row * texels.row_pitch;
            rows.push(start + x * bytes..start + x * bytes + width);
        }
    }
    rows
}

/// The origin and size of a checked `region`, across and down, in `unit`s: its texels, or its
/// blocks for a block-compressed unit (the region starts on a block, and its last block of a
/// row or column may cover its texels alone).
fn in_units(unit: &CopyUnit, region: &ImageRegion) -> ([u64; 2], [u64; 2]) {
    let [w, h] = unit.extent.map(u64::from);
    let origin = [region.origin[0], region.origin[1]].map(u64::from);
    let size = [region.size[0], region.size[1]].map(u64::from);
    (
        [origin[0] / w, origin[1] / h],
        [size[0].div_ceil(w), size[1].div_ceil(h)],
    )
}

/// The row copy of `copies` between two textures over buffers, each copy's source rows of
/// `from`'s texels and its destination rows of `to`'s, in `unit`s of one texel. Each side is its
/// texture's two layouts, `(gpu, cpu)`: where Metal's GPU finds its texels and where the CPU does
/// (`linear_plan::Placed`).
///
/// `Err(RuntimeGap::OverlappingSurfaceRows)` where either side's GPU rows share bytes (an
/// IOSurface plane's at a pitch off 16 or wider than its pitch, [`plane_rows_share_bytes`]). The
/// shared bytes take
/// whichever of the two rows Metal's command wrote last, an order that follows the command, the
/// GPU pitch and the row count and that one shape of `copyFromTexture:toTexture:` does not settle
/// from run to run (`linear_plan.rs`'s **Rows that overlap**), so no order a row copy writes them
/// in is Metal's. A source plane alone would be read, not written, as a copy out of a plane into a
/// buffer is (`blit_texture::copy_from_texture_to_buffer`), and so has no such order to miss; what
/// Metal reads out of one between two textures over buffers is unmeasured, and the source stays
/// refused with the destination until it is.
pub(super) fn plan_texels_rows_copy(
    from: (&TexelsLayout, &TexelsLayout),
    to: (&TexelsLayout, &TexelsLayout),
    unit: &CopyUnit,
    copies: &[ImageCopyRegion],
) -> Result<RowsCopy, RuntimeGap> {
    for (gpu, _) in [from, to] {
        if plane_rows_share_bytes(gpu) {
            return Err(RuntimeGap::OverlappingSurfaceRows);
        }
    }
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for copy in copies {
        src.extend(texel_rows(from.0, unit, &copy.src));
        dst.extend(texel_rows(to.0, unit, &copy.dst));
    }
    Ok(RowsCopy::new(&src, &dst))
}

/// The layout of slice `layer` (of the image's) of `region` of a texture over a buffer at
/// `texels`, as a buffer's side of a copy with a texture: its first texel's (or block's) offset,
/// and the texels' row pitch (of a row of blocks, a block-compressed texture's).
pub(super) fn texel_layout(
    texels: &TexelsLayout,
    unit: &CopyUnit,
    region: &ImageRegion,
    layer: u32,
) -> BufferLayout {
    let ([x, y], _) = in_units(unit, region);
    BufferLayout {
        offset: texels.offset
            + u64::from(layer) * texels.slice_pitch
            + y * texels.row_pitch
            + x * u64::from(unit.bytes),
        bytes_per_row: texels.row_pitch,
        bytes_per_image: 0,
    }
}

/// A copy between a buffer of `buffer_length` bytes laid out as `layout` and `region` of a
/// texture over a buffer presenting `record`, whose texels lie at `texels`, in `unit`s of one
/// texel: Metal's checks of both sides (as [`plan_buffer_copy`] makes them), then the rows, from
/// the buffer to the texture (the caller swaps them for a copy out of the texture). A
/// `bytesPerRow` below the row alone ([`rows_below_row`]) is taken as plain Metal takes it, each
/// row at `y x bytesPerRow`: rows that overlap as the source are each read whole, and as the
/// destination are written in turn, the later row's bytes standing ([`RowsCopy::new`]), one of
/// the orders plain Metal's blit leaves.
pub(super) fn plan_rows_copy(
    record: &ViewRecord,
    region: &TextureRegion,
    unit: &CopyUnit,
    layout: &BufferLayout,
    buffer_length: u64,
    texels: &TexelsLayout,
) -> Result<(Vec<Range<u64>>, Vec<Range<u64>>), CopyProblem> {
    let image = check_region(record, region, unit)?;
    let s = shape(region.size, unit);
    match rows_below_row(layout, &s, unit) {
        // The last row ends furthest: rows lie in order.
        true => match layout.end(&s) {
            Some(end) if end <= buffer_length => {}
            _ => return Err(CopyProblem::BufferRange),
        },
        false => {
            check_layout(layout, region.size, unit, buffer_length)?;
        }
    }
    Ok((
        layout_rows(layout, &s).ok_or(CopyProblem::BufferRange)?,
        texel_rows(texels, unit, &image),
    ))
}

/// The position, in blocks from the layout's offset, at which plain Metal lays block (`x`, `y`)
/// of a region `blocks` wide and high out at a `bytesPerRow` of 0 (an M4 under
/// macOS 27.0.1; the reims-vgpu consumer's report of a whole 16 x 16 level on an M3 Max under
/// macOS 26.7): the region's own Morton order, counted from its corner, the x bit lowest. Each axis
/// takes as many bits as its block count needs, and the larger axis's extra bits go on top, so a
/// count that is not a power of two leaves positions no block takes.
pub(super) fn block_order_position(x: u64, y: u64, blocks: [u64; 2]) -> u64 {
    let bits = |n: u64| u64::BITS - n.saturating_sub(1).leading_zeros();
    let (nx, ny) = (bits(blocks[0]), bits(blocks[1]));
    let (mut position, mut bit) = (0, 0);
    for i in 0..nx.max(ny) {
        if i < nx {
            position |= ((x >> i) & 1) << bit;
            bit += 1;
        }
        if i < ny {
            position |= ((y >> i) & 1) << bit;
            bit += 1;
        }
    }
    position
}

/// The plan of a copy between a buffer of `buffer_length` bytes laid out as `layout` (a
/// `bytesPerRow` of 0) and `region` of a PVRTC texture presenting `record`, in units of `unit`,
/// with the blocks in plain Metal's block order ([`block_order_position`]; `base` as
/// [`plan_buffer_copy`] takes it). Either way the plan's staging rows run buffer to staging, and
/// the caller reverses them for a copy out of the texture. Each 2 x 2 blocks the order lays out in turn is one
/// region whose rows are two blocks long, a region one block wide or high is one region, and each
/// other block is a region of its own; a position no block takes is neither read nor written.
/// Where the offset in the `VkBuffer` misses the block the copy runs through staging, which moves
/// only the positions the blocks take.
///
/// A region off its blocks lays out the blocks [`check_region`] rounds it to (its origin rounded
/// down to its block and its size up to whole blocks, each on its own), in their own Morton
/// order, as plain Metal's copy does (its validation stops it).
///
/// Refused as Metal's validation refuses it: a region past the level and an offset off the block
/// (`CopyProblem`), and a layout past the buffer (`BufferRange`). A PVRTC texture has no 3D type
/// (`texture_descriptor`), so a region within its level is one image.
pub(super) fn plan_block_order_copy(
    record: &ViewRecord,
    region: &TextureRegion,
    unit: &CopyUnit,
    layout: &BufferLayout,
    buffer_length: u64,
    base: u64,
) -> Result<BufferCopyPlan, CopyProblem> {
    let image = check_region(record, region, unit)?;
    let bytes = u64::from(unit.bytes);
    if !layout.offset.is_multiple_of(bytes) {
        return Err(CopyProblem::Offset);
    }
    let s = shape(region.size, unit);
    let blocks = [s.row_bytes / bytes, s.rows];
    let [bw, bh] = unit.extent.map(u64::from);
    // A region of blocks `[x, y]` of the grid, `[w, h]` blocks large, at `position`.
    let part = |[x, y]: [u64; 2], [w, h]: [u64; 2], position: u64| {
        let texels = |at: u64, n: u64, axis: usize, block: u64| {
            let start = at * block;
            let size = (n * block).min(u64::from(image.size[axis]) - start);
            (image.origin[axis] + start as u32, size as u32)
        };
        let ((ox, sx), (oy, sy)) = (texels(x, w, 0, bw), texels(y, h, 1, bh));
        let row_length = if h > 1 && w > 1 { (w * bw) as u32 } else { 0 };
        let image_height = if h > 1 && w > 1 { (h * bh) as u32 } else { 0 };
        (
            position,
            row_length,
            image_height,
            ImageRegion {
                origin: [ox, oy, image.origin[2]],
                size: [sx, sy, 1],
                ..image
            },
        )
    };
    let mut parts = Vec::new();
    if blocks[0] == 1 || blocks[1] == 1 {
        // One axis takes no bit: the order is the region's rows, tightly packed.
        parts.push(part([0, 0], blocks, 0));
    } else {
        for y in (0..blocks[1]).step_by(2) {
            for x in (0..blocks[0]).step_by(2) {
                if x + 1 < blocks[0] && y + 1 < blocks[1] {
                    parts.push(part([x, y], [2, 2], block_order_position(x, y, blocks)));
                    continue;
                }
                for (bx, by) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                    if bx < blocks[0] && by < blocks[1] {
                        let position = block_order_position(bx, by, blocks);
                        parts.push(part([bx, by], [1, 1], position));
                    }
                }
            }
        }
    }
    // The positions each part takes, in order: a 2 x 2 part four, a row or column its blocks.
    let taken = |p: &(u64, u32, u32, ImageRegion)| {
        let n = u64::from(p.3.size[0].div_ceil(unit.extent[0]))
            * u64::from(p.3.size[1].div_ceil(unit.extent[1]));
        p.0..p.0 + n
    };
    let end = parts.iter().map(|p| taken(p).end).max().unwrap_or(0) * bytes;
    match layout.offset.checked_add(end) {
        Some(last) if last <= buffer_length => {}
        _ => return Err(CopyProblem::BufferRange),
    }
    let span = layout.offset..layout.offset + end;
    let start = base + layout.offset;
    let staged = !vulkan_offset(start, unit);
    let regions = parts
        .iter()
        .map(
            |&(position, row_length, image_height, image)| BufferImageRegion {
                buffer_offset: (if staged { 0 } else { start }) + position * bytes,
                row_length,
                image_height,
                image,
            },
        )
        .collect();
    let staging = staged.then(|| {
        let mut runs: Vec<Range<u64>> = parts.iter().map(taken).collect();
        runs.sort_by_key(|r| r.start);
        let mut rows = Vec::<BufferRange>::new();
        for run in runs {
            match rows.last_mut() {
                Some(last) if last.dst + last.size == run.start * bytes => {
                    last.size += (run.end - run.start) * bytes;
                }
                _ => rows.push(BufferRange {
                    src: start + run.start * bytes,
                    dst: run.start * bytes,
                    size: (run.end - run.start) * bytes,
                }),
            }
        }
        Staging { length: end, rows }
    });
    Ok(BufferCopyPlan {
        regions,
        staging,
        span,
    })
}
