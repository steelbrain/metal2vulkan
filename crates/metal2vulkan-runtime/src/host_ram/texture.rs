//! Copies between a guest RAM and a texture, by the GPU, from the device's imports of the RAM's
//! chunks (a runtime extension: Metal has none): [`BlitCommandEncoder::copy_texture_from_host_ram`]
//! fills one 2D region of a slice and level of a texture from a surface laid out in the RAM, and
//! [`BlitCommandEncoder::copy_texture_to_host_ram`] copies the region back out into it. A VMM
//! whose guest's surface is a scatter of guest pages, cached in an optimal texture, copies it with
//! no view of the pages and no staging buffer of its own.
//!
//! - **The surface** is its bytes in order, given as host runs laid one after another from its
//!   first byte (`runs`: a surface's pages, each run starting on a 4-byte word, as
//!   `copy_from_host_ram` takes them), and the region's place in them as a buffer's in Metal's
//!   `copyFromBuffer:sourceOffset:sourceBytesPerRow:...`: row `y` of the region from surface byte
//!   `offset + y × bytes_per_row`, its texels one after another. Metal's checks of that layout
//!   are a buffer's (`blit_plan.rs`'s `check_layout`, the surface's bytes its length:
//!   `Problem::Copy`), and of the region the texture's (`copy_plan.rs`'s `check_region`).
//! - **One region per run of rows, straight on the imports.** Each row's texel bytes are cut
//!   where the runs' host bytes break (runs that continue one another in host memory are one) and
//!   where a chunk ends (`copy.rs`'s `pieces`), and the pieces of one import with the same first
//!   texel and texels in rows one after another, each row's bytes the surface's pitch past the row
//!   before's, are one region of a `vkCmdCopyBufferToImage` (or `vkCmdCopyImageToBuffer`) on that
//!   import (the chunk's, a range's alone, a list's: `RamOnDevice::copy_imports`, taken as the
//!   buffer forms take them, `joined_rows`): its `bufferOffset` the first row's byte in the
//!   import, its `imageExtent` the pieces' texels by its rows, its `bufferRowLength` the pitch in
//!   texels (0 for a region of one row) and its `bufferImageHeight` 0 (one image). A row that
//!   crosses a break or a chunk's end keeps its pieces, each a region of its own unless the rows
//!   around it break at the same texel. The pitch is a whole number of texels, at least a row's
//!   (`check_layout`: `VUID-VkBufferImageCopy-bufferRowLength-09101`); a pitch past 2^31 - 1 bytes
//!   joins no rows (`VUID-vkCmdCopyBufferToImage-bufferRowLength-09108`,
//!   `VUID-vkCmdCopyImageToBuffer-bufferRowLength-09108`). The regions of one import are one
//!   command, at most `MOST_REGIONS` each. A region reads or writes its rows' texel bytes alone
//!   (the spec's buffer addressing, "Copying Data Between Buffers and Images"): no staging copy and
//!   no other byte of the pages read or written, the bytes between the rows and around the surface
//!   stay as they were. A piece must start on a texel of its row and of host memory and hold
//!   whole texels, so that a region's `bufferOffset` is a multiple of the texel's bytes
//!   (`VUID-vkCmdCopyBufferToImage-dstImage-07975`, `VUID-vkCmdCopyImageToBuffer-srcImage-07975`;
//!   the imports start on host pages; the formats are uncompressed colour, so no block or depth
//!   or stencil rule applies); its texels lie inside the level (`check_region`:
//!   `VUID-vkCmdCopyBufferToImage-imageSubresource-07971` and `-07972`, the same for
//!   `vkCmdCopyImageToBuffer`), and no location is both a region's source and another's
//!   destination (the image and the imports are apart: `VUID-vkCmdCopyBufferToImage-pRegions-00173`,
//!   `VUID-vkCmdCopyImageToBuffer-pRegions-00184`). A piece that is not so (a run that splits a
//!   texel, a surface whose texels lie off their size in host memory) is refused
//!   (`Problem::HostRam`): the caller copies through a view of the pages. A copy out of the
//!   texture whose pieces name a host byte twice takes a round of commands for each
//!   (`copy.rs`'s `rounds`; rows join only within a round), the later piece's bytes kept, so no
//!   location is written by two regions of one command
//!   (`VUID-vkCmdCopyImageToBuffer-pRegions-12485`). Counted: `WorkCounts::host_ram_copies`,
//!   `host_ram_copy_bytes`, `host_ram_copy_regions` (the regions, joined rows one).
//! - **Ordered and seen as the buffer forms are** (`copy.rs`'s module docs): the chunks' bytes the
//!   pieces touch by the chunks' keys (a read into the texture, a write out of it, reported to the
//!   RAM's log of dirty pages where it has one), and the texture's region as a blit's copy with a
//!   buffer uses it, the texture held by the command buffer until it completes. First touch and a
//!   chunk that takes no import while it runs are staged through the CPU as the buffer forms
//!   stage them (`StagedGather`, `StagedScatter`), the GPU's copy between the arena's region and
//!   the texture one `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` of the region's rows at
//!   the surface's pitch (`staged_regions`: one region of them all, or one a row, `bufferRowLength`
//!   0, where the pitch is past 2^31 - 1 bytes and joins no rows); an encoding again stages them
//!   again as the buffer forms do.
//! - **The texture's contract**: an image of its own (not a texture over a buffer, a texture
//!   buffer or a texture whose blits take a shadow's copies), one sample, a colour format of
//!   uncompressed texels the image holds one to a texel (no block-compressed, depth, stencil,
//!   two-plane or carried format), and a region of one image (a depth of 1). Any other is refused
//!   (`Problem::HostRam`). A texture Metal keeps losslessly compressed is written as Metal's blit
//!   writes it where the runtime rewrites such copies (`extended_canonical.rs`), else the bytes
//!   are kept and announced (`compressed_blit.rs`), as a copy with a buffer is.

use super::copy::{
    chunk_ranges, count_waited, in_imports, pieces, rounds, touched_bytes, touched_ranges,
    used_ram, Piece, Run, Staging, Target, MOST_RANGES, MOST_REGIONS, WORD,
};
use super::{CopyImports, HostRam, RamOnDevice};
use crate::blit::{BlitCommandEncoder, TRANSFER};
use crate::blit_plan::{check_layout, BufferImageRegion, BufferLayout};
use crate::blit_texture::{buffer_image_region, copy_size, subresources, unit};
use crate::compressed_blit::{self, degrade_compressed_blit};
use crate::copy_plan::{check_region, shape, BlitOption, ImageRegion};
use crate::error::{invalid, Error, Problem};
use crate::fresh_unwritten::FreshUnwritten;
use crate::pvrtc::{image_buffer_region, image_region};
use crate::replay::BlitCall;
use crate::sparse::emulated::Written;
use crate::texture::Texture;
use crate::tracker::{Access, Use, UseRange};
use crate::two_planes;
use crate::values::{Origin, Size};
use ash::vk;
use metal2vulkan::bind::PagesCopy;
use metal2vulkan::vk::ImageAspectFlags;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

/// A copy between a guest RAM's surface and a texture's region as its call gave it
/// ([`BlitCall::HostRamTextureCopy`]), which an encoding again runs again (`replay.rs`).
pub(crate) struct HostRamTextureCall {
    /// `Gather` into the texture, `Scatter` out of it.
    copy: PagesCopy,
    /// The device's use of the RAM; `None` where it uses none such.
    ram: Option<Arc<RamOnDevice>>,
    /// The surface's bytes in order, as host runs laid one after another.
    runs: Box<[Range<u64>]>,
    /// The region's first byte of the surface and the surface's bytes per row.
    layout: (u64, u64),
    texture: Texture,
    /// The region: its slice, level, origin and size.
    region: (u64, u64, Origin, Size),
    /// How its first encoding staged it through the CPU, where it did: an encoding again stages
    /// it so again where [`Staging::again`] says.
    staged: OnceLock<Staging>,
}

impl HostRamTextureCall {
    /// The call's name, as its refusals give it.
    fn name(&self) -> &'static str {
        match self.copy {
            PagesCopy::Gather => "copy_texture_from_host_ram",
            PagesCopy::Scatter => "copy_texture_to_host_ram",
        }
    }

    /// The bytes it holds of its own, besides its size: its runs.
    pub(crate) fn extra(&self) -> usize {
        std::mem::size_of_val(&*self.runs)
    }
}

/// The runs of `runs`, a surface's bytes in order as host runs laid one after another, that the
/// surface's bytes `spans` lie in, cut to them, each at its byte of the surface, the runs of a
/// span that continue one another in host memory one: in one pass, `spans` ascending and apart
/// (a region's rows), their bytes inside the runs' total.
fn laid_over(runs: &[Range<u64>], spans: impl IntoIterator<Item = Range<u64>>) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    let (mut next, mut at) = (0usize, 0u64);
    for span in spans {
        let first = out.len();
        while let Some(run) = runs.get(next) {
            let end = at + (run.end - run.start);
            let (from, to) = (span.start.max(at), span.end.min(end));
            if from < to {
                let host = run.start + (from - at)..run.start + (to - at);
                // The span's bytes run on from the last run's, in the surface: its host bytes
                // too where they start at its end.
                let in_span = out.len() > first;
                match out.last_mut() {
                    Some((_, last)) if in_span && last.end == host.start => last.end = host.end,
                    _ => out.push((from, host)),
                }
            }
            if end > span.end {
                // The run goes on past the span, into the next one's bytes.
                break;
            }
            (next, at) = (next + 1, end);
        }
    }
    out
}

/// Rows of a copy's region that are one region of a copy on one import: `rows` rows of `texels`
/// texels from texel `(x, y)` of the region, its first row's bytes from byte `offset` of import
/// `slot`, each next row's a pitch on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rows {
    pub(crate) slot: usize,
    pub(crate) offset: u64,
    pub(crate) x: u64,
    pub(crate) y: u64,
    pub(crate) texels: u64,
    pub(crate) rows: u64,
}

/// `pieces` (each a row's texels in one import, one row: in order, by row and in a row by first
/// texel) as regions on the imports: a piece that continues a region whose last row is the row
/// before it (the same import, first texel and texels, its bytes `pitch` past that row's) joins
/// it as one more row; any other starts a region. With no `pitch`, each piece is a region of its
/// own.
pub(crate) fn joined_rows(pieces: &[Rows], pitch: Option<u64>) -> Vec<Rows> {
    let mut out: Vec<Rows> = Vec::with_capacity(pieces.len());
    // The regions whose last row is the row before the piece's, and those whose last row is the
    // piece's, by first texel (as their pieces came).
    let mut before: Vec<usize> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let (mut row, mut next) = (None, 0usize);
    for piece in pieces {
        if row != Some(piece.y) {
            let follows = row.is_some_and(|y: u64| y + 1 == piece.y);
            before.clear();
            if follows {
                before.extend_from_slice(&current);
            }
            current.clear();
            (row, next) = (Some(piece.y), 0);
        }
        // Of the row before's regions, those that start left of the piece end before it.
        while before.get(next).is_some_and(|&at| out[at].x < piece.x) {
            next += 1;
        }
        let joins = pitch.and_then(|pitch| {
            let &at = before.get(next)?;
            let region = &out[at];
            let continues = region.slot == piece.slot
                && (region.x, region.texels) == (piece.x, piece.texels)
                && (region.rows.checked_mul(pitch))
                    .is_some_and(|step| piece.offset.checked_sub(region.offset) == Some(step));
            continues.then_some(at)
        });
        match joins {
            Some(at) => {
                out[at].rows += 1;
                current.push(at);
                next += 1;
            }
            None => {
                out.push(*piece);
                current.push(out.len() - 1);
            }
        }
    }
    out
}

/// The pitch at which a region's `rows` rows, `bytes_per_row` bytes apart, are one region of a
/// copy, its `bufferRowLength` the pitch's texels: a region of more than one row, its pitch of at
/// most 2^31 - 1 bytes (`VUID-vkCmdCopyBufferToImage-bufferRowLength-09108`,
/// `VUID-vkCmdCopyImageToBuffer-bufferRowLength-09108`), a whole number of texels at least a
/// row's (`check_layout`: `VUID-VkBufferImageCopy-bufferRowLength-09101`). `None`: each row is a
/// region of its own, `bufferRowLength` 0.
fn joining_pitch(rows: u64, bytes_per_row: u64) -> Option<u64> {
    (rows > 1 && bytes_per_row <= i32::MAX as u64).then_some(bytes_per_row)
}

/// A staged copy's regions between the arena's region and `image` (a region of one image of
/// `texel`-byte texels, one to a texel), its rows `bytes_per_row` apart from the arena region's
/// first byte: one region of them all where they join (`joining_pitch`), else one a row at its
/// row's byte, `bufferRowLength` 0. The same bytes either way.
pub(crate) fn staged_regions(
    image: ImageRegion,
    bytes_per_row: u64,
    texel: u64,
) -> Vec<BufferImageRegion> {
    let [x, y, z] = image.origin;
    let [width, height, depth] = image.size;
    match joining_pitch(u64::from(height), bytes_per_row) {
        Some(pitch) => vec![BufferImageRegion {
            buffer_offset: 0,
            row_length: (pitch / texel) as u32,
            image_height: 0,
            image,
        }],
        None => (0..height)
            .map(|row| BufferImageRegion {
                buffer_offset: u64::from(row) * bytes_per_row,
                row_length: 0,
                image_height: 0,
                image: ImageRegion {
                    origin: [x, y + row, z],
                    size: [width, 1, depth],
                    ..image
                },
            })
            .collect(),
    }
}

impl BlitCommandEncoder {
    /// Copy one 2D region of `destination` from a surface laid out in the guest RAM `ram`, by the
    /// GPU, from the device's imports of the RAM's chunks: a runtime extension, Metal has none (a
    /// program copies from a no-copy buffer over the pages, `copy_from_buffer_to_texture`). The
    /// surface is `runs`, its bytes in order as host runs inside the RAM's ranges laid one after
    /// another from its first byte, each starting on a 4-byte word (as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) takes them); `source_size` texels of it,
    /// row `y` from surface byte `source_offset + y × source_bytes_per_row`, land in
    /// `destination_level` and `destination_slice` at `destination_origin`, as
    /// `copy_from_buffer_to_texture` would land them from a buffer of the surface's bytes. The
    /// rows' texels in one chunk's import whose host bytes lie the surface's pitch apart (pages in
    /// a row) are one region of a `vkCmdCopyBufferToImage` straight from that import
    /// (`bufferRowLength` the pitch's
    /// texels; a row across a chunk's end or a break in the runs' host bytes a region for each
    /// piece: the module docs of `host_ram/texture.rs`): no staging copy, and no byte of the
    /// pages but the region's texels read. Ordered and seen as
    /// [`copy_from_host_ram`](Self::copy_from_host_ram) is (the CPU's writes before the commit
    /// seen, hazard tracking over the host bytes), and the texture written as by any blit; first
    /// touch through the CPU as that call says.
    ///
    /// A size of 0 copies nothing.
    ///
    /// Refused: a device that imports no host memory (`Unsupported`, naming
    /// `VK_EXT_external_memory_host`); a RAM the device does not use
    /// ([`Device::use_host_ram`](crate::Device::use_host_ram)), a run that is empty, starts off a
    /// 4-byte word or lies outside the RAM's ranges, a row's piece in a run or chunk that does not
    /// start on a texel of host memory or hold whole texels, and a texture that is not one
    /// image of its own of one sample and uncompressed colour texels one to a texel, or a
    /// region of more than one image (a texture over a buffer, a texture buffer, a multisampled,
    /// block-compressed, depth, stencil, two-plane or carried format: `Invalid`,
    /// `Problem::HostRam`); a region past the level, a `source_offset` or `source_bytes_per_row`
    /// off the texel, rows that overlap, or rows past the surface's bytes (`Invalid`,
    /// `Problem::Copy`, as `copy_from_buffer_to_texture` refuses them); a texture of another
    /// device (`Invalid`); and a chunk the device cannot import (the import's error). A caller
    /// falls back to its own copy.
    pub fn copy_texture_from_host_ram(
        &mut self,
        ram: &HostRam,
        runs: &[Range<u64>],
        source_offset: u64,
        source_bytes_per_row: u64,
        source_size: Size,
        destination: &Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: Origin,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let call = HostRamTextureCall {
            copy: PagesCopy::Gather,
            ram: used_ram(&device, ram),
            runs: runs.into(),
            layout: (source_offset, source_bytes_per_row),
            texture: destination.clone(),
            region: (
                destination_slice,
                destination_level,
                destination_origin,
                source_size,
            ),
            staged: OnceLock::new(),
        };
        self.host_ram_texture_copy(&Arc::new(call))
    }

    /// Copy one 2D region of `source` out into a surface laid out in the guest RAM `ram`, by the
    /// GPU, into the device's imports of the RAM's chunks: the copy
    /// [`copy_texture_from_host_ram`](Self::copy_texture_from_host_ram) makes the other way, a
    /// runtime extension (Metal has none). `source_size` texels of `source_level` and
    /// `source_slice` from `source_origin` land in the surface `runs` (its bytes in order, as
    /// that call takes them), row `y` from surface byte `destination_offset + y ×
    /// destination_bytes_per_row`, by regions of a `vkCmdCopyImageToBuffer` straight into the
    /// imports that hold them, as that call's regions read them. Only the region's texels
    /// are written: the bytes between the rows and the rest of the pages keep what they held.
    /// Ordered and seen as [`copy_to_host_ram`](Self::copy_to_host_ram) is: its bytes are in the
    /// pages once the command buffer completed, and where the RAM came with a log of its pages'
    /// writes (`HostRam::with_dirty_pages`) they are reported to it as the GPU's writes in place.
    /// Runs that name a host byte twice leave it as the later row's texels wrote it.
    ///
    /// Refused as [`copy_texture_from_host_ram`](Self::copy_texture_from_host_ram), `source` in
    /// place of the destination.
    pub fn copy_texture_to_host_ram(
        &mut self,
        source: &Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: Origin,
        source_size: Size,
        ram: &HostRam,
        runs: &[Range<u64>],
        destination_offset: u64,
        destination_bytes_per_row: u64,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let call = HostRamTextureCall {
            copy: PagesCopy::Scatter,
            ram: used_ram(&device, ram),
            runs: runs.into(),
            layout: (destination_offset, destination_bytes_per_row),
            texture: source.clone(),
            region: (source_slice, source_level, source_origin, source_size),
            staged: OnceLock::new(),
        };
        self.host_ram_texture_copy(&Arc::new(call))
    }

    /// Run `call`, logged first where the encoder logs (`replay.rs`).
    pub(crate) fn host_ram_texture_copy(
        &mut self,
        call: &Arc<HostRamTextureCall>,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::HostRamTextureCopy(Arc::clone(call)),
            |this| this.record_host_ram_texture_copy(call),
        )
    }

    /// Check `call` and record it (the module docs).
    fn record_host_ram_texture_copy(&mut self, call: &HostRamTextureCall) -> Result<(), Error> {
        let name = call.name();
        let into = call.copy == PagesCopy::Gather;
        let texture = &call.texture;
        self.check_texture(name, texture)?;
        let device = self.core().device().clone();
        let shared = device.shared();
        let importer = match &shared.host_import {
            Ok(importer) => importer,
            Err(unsupported) => return Err(Error::Unsupported(unsupported.clone())),
        };
        let refused = || invalid(name, Problem::HostRam);
        let ram = (call.ram.as_ref())
            .filter(|ram| shared.host_ram.uses(ram))
            .ok_or_else(refused)?;
        texture.check_plane_copied()?;
        if self.fails_on_memoryless(&[texture]) {
            return Ok(());
        }
        texture.check_rotated_reach()?;
        let (slice, level, origin, size) = call.region;
        let Some(size) = copy_size(size, &[texture.texture_type()]) else {
            return Ok(());
        };
        // The texture's contract (the module docs): one image of its own, one sample, a region of
        // one image of uncompressed colour texels one to a texel.
        let own_image = !texture.over_buffer() && texture.shadow().is_none();
        let planes = two_planes::holds_planes(texture.pixel_format());
        if !own_image || planes || texture.record().samples > 1 || size.depth != 1 {
            return Err(refused());
        }
        let unit = unit(name, texture, BlitOption::None)?;
        if unit.extent != [1, 1] || unit.aspect != ImageAspectFlags::COLOR {
            return Err(refused());
        }
        let region = crate::blit_texture::region(name, origin, size, level, slice)?;
        let image = check_region(texture.record(), &region, &unit)
            .map_err(|problem| invalid(name, Problem::Copy(problem)))?;
        let plan = texture.image_plan()?;
        if image_region(plan, image) != image {
            return Err(refused());
        }
        // The surface's bytes, and Metal's checks of the region's layout over them.
        let mut surface: u64 = 0;
        for run in call.runs.iter() {
            if run.start >= run.end || !run.start.is_multiple_of(WORD) {
                return Err(refused());
            }
            surface = (surface.checked_add(run.end - run.start)).ok_or_else(refused)?;
        }
        let (offset, bytes_per_row) = call.layout;
        let layout = BufferLayout {
            offset,
            bytes_per_row,
            bytes_per_image: 0,
        };
        let spanned = check_layout(&layout, region.size, &unit, surface)
            .map_err(|problem| invalid(name, Problem::Copy(problem)))?;
        let rows = shape(region.size, &unit);
        // Each row's texel bytes, in the runs, cut where a chunk ends: within the surface's bytes
        // (`check_layout`), so no row's byte passes `u64`.
        let row_bytes = (0..rows.rows).map(|y| {
            let start = offset + y * bytes_per_row;
            start..start + rows.row_bytes
        });
        let laid = laid_over(&call.runs, row_bytes);
        let chunks = ram.chunks();
        let spans = &chunks.spans;
        let (pieces, total) = pieces(&laid, &chunks).ok_or_else(refused)?;
        let texel = u64::from(unit.bytes);
        let whole = |piece: &Piece| {
            let host = spans[piece.slot].start + piece.offset;
            piece.at.is_multiple_of(texel)
                && piece.length.is_multiple_of(texel)
                && host.is_multiple_of(texel)
        };
        if !pieces.iter().all(whole) {
            return Err(refused());
        }
        // The chunks' bytes the pieces touch, and how the copy takes their chunks: as the buffer
        // forms take them (`copy.rs`).
        let (ranges, host_twice) = touched_ranges(&pieces, spans);
        if !into {
            for &slot in ranges.keys() {
                ram.copy_wrote(slot);
            }
        }
        let again = (self.core().encodes_again())
            .then(|| {
                call.staged
                    .get()
                    .and_then(|staging| staging.again(call.copy))
            })
            .flatten();
        let takes = self.takes(ram, importer, &ranges);
        let written = || into && ram.written_over(&device, &touched_bytes(&pieces, spans));
        let (levels, layers) = subresources(&image);
        let access = match into {
            true => Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            false => Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
        };
        let image_use = texture.image_use(levels, layers, access, into)?;
        // The host bytes' use: the other side of the copy's.
        let host_access = match into {
            true => Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
            false => Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
        };
        let raw_image = texture.raw_image()?;
        if let Some(staging) = again.or_else(|| takes.staging(written)) {
            // The first encoding's staging, which an encoding again finds set.
            let _ = call.staged.set(staging);
            // The region's rows at the surface's pitch in the arena's region, from its first.
            let regions = (staged_regions(image, bytes_per_row, texel).into_iter())
                .map(|staged| buffer_image_region(&image_buffer_region(plan, staged)))
                .collect();
            let target = Target::Image {
                image: raw_image,
                regions,
            };
            let from_first: Vec<Piece> = (pieces.iter())
                .map(|piece| Piece {
                    at: piece.at - offset,
                    ..*piece
                })
                .collect();
            let side = (target, spanned, image_use);
            match into {
                true => self.record_staged(ram, (&from_first, spans), side, total, staging)?,
                false => self.record_staged_scatter(ram, (&from_first, spans), side, total)?,
            }
            return self.texture_written(name, texture, into, image);
        }
        let CopyImports {
            imports,
            of_chunk,
            waited,
        } = ram.copy_imports(&device, importer, &ranges)?;
        count_waited(&device, waited);
        let in_imports = in_imports(&imports, &of_chunk, &pieces, spans)?;
        // A copy out of the texture whose pieces name a host byte twice goes by rounds, no byte
        // written by two regions of one command, the later piece's kept.
        let by_round = match (into, host_twice) {
            (false, true) => {
                let host: Vec<Range<u64>> = (pieces.iter())
                    .map(|piece| {
                        let start = spans[piece.slot].start + piece.offset;
                        start..start + piece.length
                    })
                    .collect();
                rounds(&host)
            }
            _ => vec![(0..pieces.len()).collect()],
        };
        // A copy out of the texture reads its region alone (`fresh_unwritten.rs`).
        let fresh_reads = match into {
            true => Vec::new(),
            false => {
                let read = [(image.aspect.into(), Written::of(&image))];
                FreshUnwritten::reads(&device, texture, &read)?
            }
        };
        // Rows of one region at the surface's pitch, `bufferRowLength` its texels.
        let pitch = joining_pitch(rows.rows, bytes_per_row);
        let mut regions_recorded = 0u64;
        for round in &by_round {
            // The round's pieces, each a row's texels from its byte of its import, in order.
            let mut row_pieces = Vec::with_capacity(round.len());
            for &index in round {
                let piece = &in_imports[index];
                if !piece.offset.is_multiple_of(texel) {
                    return Err(refused());
                }
                let from_first = piece.at - offset;
                let y = match rows.rows {
                    1 => 0,
                    _ => from_first / bytes_per_row,
                };
                row_pieces.push(Rows {
                    slot: piece.slot,
                    offset: piece.offset,
                    x: (from_first - y * bytes_per_row) / texel,
                    y,
                    texels: piece.length / texel,
                    rows: 1,
                });
            }
            // The round's regions, by import, in order: the pieces of one import, one first texel
            // and texels in rows one after another at the pitch one region of their rows.
            let joined = joined_rows(&row_pieces, pitch);
            let mut by_import: BTreeMap<usize, Vec<vk::BufferImageCopy>> = BTreeMap::new();
            for region in &joined {
                // Inside the region (`check_region`), whose extent is a `u32`'s.
                let texels = ImageRegion {
                    origin: [
                        image.origin[0] + region.x as u32,
                        image.origin[1] + region.y as u32,
                        image.origin[2],
                    ],
                    size: [region.texels as u32, region.rows as u32, 1],
                    ..image
                };
                let row_length = match (region.rows, pitch) {
                    (2.., Some(pitch)) => (pitch / texel) as u32,
                    _ => 0,
                };
                let copy = BufferImageRegion {
                    buffer_offset: region.offset,
                    row_length,
                    image_height: 0,
                    image: texels,
                };
                let copy = buffer_image_region(&image_buffer_region(plan, copy));
                by_import.entry(region.slot).or_default().push(copy);
            }
            // The round's uses: the chunks' bytes it reads or writes, by their keys, and the
            // texture's region.
            let round_pieces: Vec<Piece> = round.iter().map(|&index| pieces[index]).collect();
            let mut uses = Vec::new();
            for (slot, ranges) in chunk_ranges(&round_pieces, MOST_RANGES) {
                let key = ram.chunk_key(&device, slot);
                for range in ranges {
                    let range = UseRange::Bytes(range);
                    uses.push(match into {
                        true => Use::reads(key, true, range, host_access),
                        false => Use::writes(key, true, range, host_access),
                    });
                }
            }
            uses.push(image_use.clone());
            regions_recorded += joined.len() as u64;
            let imports = &imports;
            self.command_reading(&uses, (&fresh_reads, &[]), |functions, raw| {
                let general = vk::ImageLayout::GENERAL;
                for (slot, regions) in &by_import {
                    let chunk = imports[*slot].raw;
                    for part in regions.chunks(MOST_REGIONS) {
                        // SAFETY: the encoder's command buffer is recording, outside a render
                        // pass; the imports and the texture are live until it completes (held
                        // below); each region's bytes, its rows' alone, lie inside its import
                        // (each row a piece: `pieces`, the imports made over them) from a
                        // multiple of the texel's bytes, its texels inside the level
                        // (`check_region`), the image is in the `GENERAL` layout, and no byte is
                        // written by two regions of one command (`rounds`, the rows of a region
                        // pieces of one round).
                        unsafe {
                            match into {
                                true => functions
                                    .cmd_copy_buffer_to_image(raw, chunk, raw_image, general, part),
                                false => functions
                                    .cmd_copy_image_to_buffer(raw, raw_image, general, chunk, part),
                            }
                        };
                    }
                }
            })?;
        }
        shared.counts.host_ram_copied(total, 0);
        shared.counts.host_ram_copy_regions(regions_recorded);
        self.core().retain(imports);
        self.texture_written(name, texture, into, image)
    }

    /// After a copy `call` recorded between the guest RAM and `region` of `texture`: the texture
    /// held until the command buffer completes; into it (`into`), its region written
    /// (`BlitCommandEncoder::scrub`), Metal's encoding of the texels where it keeps the texture
    /// losslessly compressed and the runtime rewrites them, else the note that the bytes are kept,
    /// and its decoded twin's texels; out of it, that note.
    fn texture_written(
        &mut self,
        call: &'static str,
        texture: &Texture,
        into: bool,
        region: ImageRegion,
    ) -> Result<(), Error> {
        self.core().retain_shared(texture.held());
        if !into {
            return degrade_compressed_blit(call, texture);
        }
        self.scrub(texture, vec![(region.aspect.into(), Written::of(&region))])?;
        let written = [region];
        if !(compressed_blit::canonical(texture) && self.canonical_texels(texture, &written)?) {
            degrade_compressed_blit(call, texture)?;
        }
        self.decode_twin(texture, &written)
    }
}
