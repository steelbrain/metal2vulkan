//! Blits between a buffer and a two-plane extended-range texture of one level whose width is not a
//! multiple of 4, at the rows Metal's blit addresses.
//!
//! **Metal's answers** (an Apple M3 Max under macOS 27.0.1, our probe `bp.m`: 2D textures and
//! slices of arrays of their own memory, 3 to 101 texels wide, regions at origins; each texel's
//! color word set by `replaceRegion`, the texels read by `getBytes`; alike in every run):
//!
//! - Out of the texture, row `r` of the region reads the `4 w` bytes of plane 0 from
//!   `(y + r) · ⌊4 W / 16⌋ · 16 + 4 x` (`W` the level's width), where the level's rows lie
//!   `align(4 W, 16)` bytes apart: texels of the row it starts in and of the next, and the bytes
//!   between the rows, 0 in a new texture and what a fill or a blit wrote there after. Below a
//!   width of 4 the level's first row reads as it lies and every later row as 0.
//! - Into the texture, row `r` writes from `(y + r) · max(16, ⌊4 W / 16⌋ · 16) + 4 x`, the rows
//!   a fill addresses (`blit_fill.rs`). Rows `r` and `r + 1` of the region meet where `w`
//!   passes `⌊4 W / 16⌋ · 4` texels (probe `bpm.m`: 5 to 101 texels wide, 2 to 128 rows, origins
//!   0 to 2 across and 0 to 64 down, every meeting, run 2 to 5 times; alike for a buffer of
//!   other rows and offsets, a slice of an array, a heap's texture, a `Private` one and the sRGB
//!   twin). Below a width of 32 Metal's GPU writes the region's rows in groups of 32, and in each
//!   in pairs where the origin's `x` and `y` are even, else in fours: within a pair or four the
//!   earlier row's bytes stand where two meet, and between them the later row's, alike in every
//!   run. Between groups of 32 rows, and at every meeting 32 texels wide or wider, the order
//!   varies from run to run (the earlier row's bytes at 556 of 560 meetings between groups, and
//!   at most meetings of the wider ones, the later row's at most meetings 3 or 5 rows high).
//!
//! **Held:** the level's rows the blit reaches go through a scratch buffer of the encoder laid
//! out as Metal's plane 0, rows `align(4 W, 16)` apart, one blit command a step: the scratch
//! zeroed, then the texels copied into it (`plane_words_copy`; for a heap's texture whose planes
//! lie in the heap's bytes, those bytes). Out of the texture, the region's rows are copied from
//! the scratch at Metal's addresses into the buffer (a row Metal reads as 0 from the scratch's
//! zeroed tail). Into it, the buffer's rows are copied into the scratch at Metal's addresses, the
//! row whose bytes Metal's stand where two meet ([`later_row_stands`]; the earlier row's where
//! Metal's order varies), and the scratch back into the texels (or the heap's bytes). A texture of the device's memory holds the bytes between its rows apart
//! (`Texture::between_rows`, plane 0 of each slice as Metal lays it out, made zeroed at the first
//! write there): a blit into the texture whose rows reach them, and a fill whose words land
//! there (`blit_fill.rs`), write them, and the scratch of a blit out of it starts from them. A
//! copy from another texture that writes bytes there does not (`two_planes.rs`): it notes them
//! (`Texture::keep_outside`), and a later blit out of the texture that reads them reads them as
//! the held bytes, announced (`two-plane-buffer-blit`), as is a blit into it 32 texels wide or
//! wider whose rows meet; one narrower whose rows meet between groups of 32 rows announces
//! `two-plane-buffer-blit-order`, the earlier row's bytes there, with Metal's order elsewhere, being
//! Metal's most frequent whole outcome.

use super::super::blit::{reads, writes, BlitCommandEncoder};
use super::super::blit_plan::BufferLayout;
use super::super::copy_plan::{level_extent, ImageRegion};
use super::super::degradation::Degradation;
use super::super::error::Error;
use super::super::texture::Texture;
use super::super::values::ResourceOptions;
use super::super::Buffer;
use super::heap_bytes::BytePlanes;
use ash::vk;
use std::ops::Range;

/// The bytes apart that Metal's blit addresses plane 0's rows of a level `width` texels wide at:
/// out of the texture (0 below a width of 4), or into it (`into_texture`).
pub(in super::super) fn buffer_blit_rows(width: u64, into_texture: bool) -> u64 {
    let rows = 4 * width / 16 * 16;
    if into_texture {
        rows.max(16)
    } else {
        rows
    }
}

/// The bytes of plane 0 that each row of a blit of `size` texels (width, height) at `origin`
/// addresses in a level `width` texels wide, in order: `None` for a row Metal's blit out of the
/// texture reads as 0.
pub(in super::super) fn row_spans(
    width: u64,
    [x, y]: [u64; 2],
    [w, h]: [u64; 2],
    into_texture: bool,
) -> Vec<Option<Range<u64>>> {
    let rows = buffer_blit_rows(width, into_texture);
    (y..y + h)
        .map(|row| {
            let start = row * rows + 4 * x;
            (rows > 0 || row == 0).then_some(start..start + 4 * w)
        })
        .collect()
}

/// Where rows `r` and `r + 1` of a blit into a level `width` texels wide from `origin` (`x`, `y`)
/// meet, whether the later row's bytes stand as Metal's GPU writes them (the module docs):
/// `Some(true)` between pairs of rows (an even origin) or fours (another) in a group of 32 rows,
/// `Some(false)` within them, `None` where Metal's order varies from run to run (between groups,
/// or 32 texels wide or wider).
pub(in super::super) fn later_row_stands(width: u64, [x, y]: [u64; 2], r: u64) -> Option<bool> {
    if width >= 32 || r % 32 == 31 {
        return None;
    }
    let rows = if x % 2 == 0 && y % 2 == 0 { 2 } else { 4 };
    Some(r % rows == rows - 1)
}

/// Whether `span` of plane 0 of a level `width` texels wide holds bytes between its rows, which
/// lie `align(4 width, 16)` bytes apart.
pub(in super::super) fn between_rows(span: &Range<u64>, width: u64) -> bool {
    let pitch = (4 * width).next_multiple_of(16);
    if span.is_empty() || pitch == 4 * width {
        return false;
    }
    (span.start / pitch..=(span.end - 1) / pitch).any(|row| {
        let padding = row * pitch + 4 * width..(row + 1) * pitch;
        padding.start.max(span.start) < padding.end.min(span.end)
    })
}

/// Whether a texture's blits with a buffer at a width off 4 go through Metal's rows: a 2D
/// texture or array of one level, not a view, whose texels are not a surface's bytes. Metal's
/// answer for another (more levels: its blits read back bytes that vary from run to run) is not
/// measured.
pub(in super::super) fn rows_held(texture: &Texture) -> bool {
    use metal2vulkan::metal::TextureType;
    !texture.over_buffer()
        && texture.parent_texture().is_none()
        && texture.record().level_count == 1
        && matches!(
            texture.texture_type(),
            TextureType::Type2D | TextureType::Type2DArray
        )
}

impl BlitCommandEncoder {
    /// A blit between `buffer`, laid out as `layout`, and `region` (one slice) of a two-plane
    /// extended-range texture of one level ([`rows_held`]) whose width is not a multiple of 4, at
    /// Metal's rows (`into_texture` from the buffer; the module docs).
    pub(in super::super) fn plane_rows_copy(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
        region: &ImageRegion,
        layout: &BufferLayout,
        into_texture: bool,
    ) -> Result<(), Error> {
        let [width, height, _] = level_extent(texture.record(), region.level).map(u64::from);
        let [x, y, _] = region.origin.map(u64::from);
        let [w, h, _] = region.size.map(u64::from);
        if w == 0 || h == 0 {
            return Ok(());
        }
        let pitch = (4 * width).next_multiple_of(16);
        let row_bytes = 4 * w;
        let spans = row_spans(width, [x, y], [w, h], into_texture);
        let first = spans.iter().flatten().map(|s| s.start / pitch).min();
        let last = spans.iter().flatten().map(|s| (s.end - 1) / pitch).max();
        // The level's rows `first..first + rows` in the scratch, then a zeroed row of the region.
        let (first, rows) = match (first, last) {
            (Some(first), Some(last)) => (first, last + 1 - first),
            _ => (0, 0),
        };
        let bytes = rows * pitch;
        let total = bytes + row_bytes;
        let device = self.core().device().clone();
        let scratch = device.own_buffer(total, ResourceOptions::private())?;
        let (raw_scratch, at) = scratch.at(0);
        self.command(&[writes(&scratch, 0..total)], |device, raw| {
            // SAFETY: the command buffer is recording, outside a render pass; the scratch is
            // retained until it completes; the fill covers its bytes, from a 4-byte multiple.
            unsafe { device.cmd_fill_buffer(raw, raw_scratch, at, total, 0) };
        })?;
        self.core().retain(scratch.clone());
        // Plane 0's bytes in a heap's, where they lie there; else the image's texels, over the
        // bytes between its rows that fills and blits wrote (held from the slice's plane 0 on,
        // `Texture::between_rows`).
        let level = [width, height].map(|v| v as u32);
        let planes = BytePlanes::of(texture, level).filter(|planes| !planes.surface);
        let slice_at = u64::from(region.layer) * height * pitch + first * pitch;
        if let (None, Some(held), true) = (&planes, texture.between_rows(), rows > 0) {
            let held = held.clone();
            self.byte_copies(&held, &scratch, &[(slice_at, 0, bytes)])?;
        }
        let level_rows = ImageRegion {
            origin: [0, first as u32, 0],
            size: [width as u32, rows as u32, 1],
            layers: 1,
            ..*region
        };
        let scratch_layout = BufferLayout {
            offset: 0,
            bytes_per_row: pitch,
            bytes_per_image: bytes,
        };
        if rows > 0 {
            match &planes {
                Some(planes) => {
                    let from = planes.color + first * pitch;
                    self.byte_copies(planes.bytes, &scratch, &[(from, 0, bytes)])?;
                }
                None => {
                    self.plane_words_copy(&scratch, texture, &level_rows, &scratch_layout, false)?
                }
            }
        }
        let pf = texture.pixel_format();
        let announce_as = |kind: Degradation, reason: &'static str| {
            device.degrade(kind, || {
                let object = format!(
                    "a blit between a buffer and a {} texture {width} texels wide",
                    pf.name()
                );
                (object, reason.to_owned())
            });
        };
        let announce = |reason| announce_as(Degradation::TwoPlaneBufferBlit, reason);
        let reaches_between = spans.iter().flatten().any(|s| between_rows(s, width));
        if !into_texture {
            let copies: Vec<(u64, u64, u64)> = spans
                .iter()
                .enumerate()
                .map(|(r, span)| {
                    let from = span.as_ref().map_or(bytes, |s| s.start - first * pitch);
                    (
                        from,
                        layout.offset + r as u64 * layout.bytes_per_row,
                        row_bytes,
                    )
                })
                .collect();
            self.byte_copies(&scratch, buffer, &copies)?;
            if planes.is_none() && reaches_between && texture.kept_outside() {
                announce(
                    "out of the texture it reads the bytes between the texture's rows that a \
                     copy from another texture wrote as 0; Metal's blit reads them back",
                );
            }
            return Ok(());
        }
        // Each row cut where it meets the next and the later row's bytes stand, and from where
        // the row before it ends where the earlier row's do (the earlier row's where Metal's
        // order varies). Into the texture every row has a span, and only neighbours meet.
        let spans: Vec<Range<u64>> = spans.into_iter().flatten().collect();
        let stands = |r: usize| {
            let meets = spans
                .get(r + 1)
                .is_some_and(|next| next.start < spans[r].end);
            meets.then(|| later_row_stands(width, [x, y], r as u64))
        };
        let mut copies = Vec::with_capacity(spans.len());
        let mut varies = false;
        for (r, span) in spans.iter().enumerate() {
            let start = match r.checked_sub(1).and_then(stands) {
                Some(Some(true)) | None => span.start,
                Some(Some(false) | None) => span.start.max(spans[r - 1].end),
            };
            let end = match stands(r) {
                Some(Some(true)) => span.end.min(spans[r + 1].start),
                Some(None) => {
                    varies = true;
                    span.end
                }
                Some(Some(false)) | None => span.end,
            };
            if start < end {
                let from = layout.offset + r as u64 * layout.bytes_per_row + (start - span.start);
                copies.push((from, start - first * pitch, end - start));
            }
        }
        self.byte_copies(buffer, &scratch, &copies)?;
        match &planes {
            Some(planes) => {
                let to = planes.color + first * pitch;
                self.byte_copies(&scratch, planes.bytes, &[(0, to, bytes)])?;
            }
            None => {
                self.plane_words_copy(&scratch, texture, &level_rows, &scratch_layout, true)?;
                if reaches_between {
                    let held = self.held_between_rows(texture, [width, height])?;
                    self.byte_copies(&scratch, &held, &[(0, slice_at, bytes)])?;
                }
            }
        }
        if varies && width >= 32 {
            announce(
                "into the texture, where its rows meet, the earlier row's bytes stand; Metal's GPU \
                 writes such rows of a texture 32 texels wide or wider in an order that varies \
                 from run to run",
            );
        } else if varies {
            announce_as(
                Degradation::TwoPlaneBufferBlitOrder,
                "into the texture, where its rows meet between Metal's groups of 32 rows, the \
                 earlier row's bytes stand, Metal's most frequent; Metal's GPU writes such rows \
                 in an order that varies from run to run",
            );
        }
        Ok(())
    }

    /// The bytes between the rows of plane 0 of every slice of `texture`, a two-plane
    /// extended-range texture of one level `level` (width, height) texels ([`rows_held`]), laid
    /// out as Metal's plane 0 (`Texture::between_rows`): made, and zeroed in the command buffer,
    /// at the first write that lands there.
    pub(in super::super) fn held_between_rows(
        &mut self,
        texture: &Texture,
        [width, height]: [u64; 2],
    ) -> Result<Buffer, Error> {
        let pitch = (4 * width).next_multiple_of(16);
        let length = u64::from(texture.record().layer_count) * height * pitch;
        let device = self.core().device().clone();
        let (held, made) =
            texture.hold_between_rows(|| device.own_buffer(length, ResourceOptions::private()))?;
        if made {
            let (raw, at) = held.at(0);
            self.command(&[writes(&held, 0..length)], |device, command_buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; the buffer is
                // the texture's, live while it is; the fill covers its bytes, a multiple of 16.
                unsafe { device.cmd_fill_buffer(command_buffer, raw, at, length, 0) };
            })?;
        }
        self.core().retain(held.clone());
        Ok(held)
    }

    /// Copy each `(from, to, bytes)` of `regions` from `source`'s bytes into `destination`'s, in
    /// one command (the destinations disjoint; `source` and `destination` other buffers).
    pub(in super::super) fn byte_copies(
        &mut self,
        source: &Buffer,
        destination: &Buffer,
        regions: &[(u64, u64, u64)],
    ) -> Result<(), Error> {
        let regions: Vec<(u64, u64, u64)> = regions.iter().copied().filter(|r| r.2 > 0).collect();
        let span = |at: fn(&(u64, u64, u64)) -> u64| -> Range<u64> {
            let start = regions.iter().map(at).min().unwrap_or(0);
            let end = regions.iter().map(|r| at(r) + r.2).max().unwrap_or(0);
            start..end
        };
        let (read, written) = (span(|r| r.0), span(|r| r.1));
        if read.is_empty() {
            return Ok(());
        }
        // Read as a blit's (`host_residency.rs`).
        let (raw_source, source_base) = self.core().blit_at(source, 0)?;
        let (raw_destination, destination_base) = self.core().blit_at(destination, 0)?;
        let copies: Vec<vk::BufferCopy> = regions
            .iter()
            .map(|&(from, to, size)| vk::BufferCopy {
                src_offset: source_base + from,
                dst_offset: destination_base + to,
                size,
            })
            .collect();
        let uses = [reads(source, read), writes(destination, written)];
        self.command(&uses, |device, raw| {
            // SAFETY: the command buffer is recording, outside a render pass; both buffers are
            // live and retained until it completes; every region lies inside both, and the two
            // are other buffers.
            unsafe { device.cmd_copy_buffer(raw, raw_source, raw_destination, &copies) };
        })?;
        Ok(())
    }
}
