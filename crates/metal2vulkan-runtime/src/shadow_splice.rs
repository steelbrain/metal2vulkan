//! A shadowed texture's copy back into bytes the CPU may write beside its texels, spliced: the
//! GPU writes only the lines of memory its texels fill, and the CPU writes the rest; and its merge
//! there, on the host: the CPU stores the bytes the commands changed.
//!
//! A device may write back a line of memory a store of its ends in whole (`in_place::RowEnds`: a
//! line is `nonCoherentAtomSize` bytes at a multiple of it, "the size and alignment in bytes that
//! bounds concurrent access to host-mapped device memory", vkspec, Limits): on an Iris Xe under
//! ANV, a copy back of a 34x34 `RGBA16Float` shadow into rows 512 bytes apart over host pages put
//! back older values over the CPU's writes to the padding in the line that holds each row's last
//! texels, a few thousand bytes in 300 command buffers, and so did a merge's stores
//! (`tests/shadow_write_lines.rs`). Metal keeps every byte the CPU writes beside a texture's
//! texels (`crates/metal2vulkan-conformance/probes/row_padding_writes.m`: the bytes between an
//! IOSurface plane's rows under clears, stores, draws and kernels on every Mac it ran on).
//!
//! **Where.** A copy back ([`Step::Back`](crate::shadow::Step)) of a 2D texture's image, or of a
//! texture buffer's staging buffer, into a no-copy buffer's host pages (not a copy of them,
//! `copied.rs`), on a device whose check of such pages ([`PagesClass::DmaBuf`] where they are
//! imported as dma-bufs, else [`PagesClass::Pointer`]) failed, or has not answered, the copies or
//! the shader writes of texels that share a line with bytes beside them ([`Access::Copies`],
//! [`Access::Writes`] of [`RowEnds::InLines`], [`splices`]); a device where both passed copies
//! back as before. Recorded by the encoder's own copies (`EncoderCore::shadow_copies`), outside a
//! render pass it holds and outside a segment's recording, where a host step may follow.
//!
//! **How** ([`image`], [`staged`]). Each row of the copy's texels (each region's, each slice's)
//! is cut at the first and last line boundaries it holds: the columns between them, whose lines
//! hold that row's texels alone, are copied by the GPU straight into the buffer, one region for
//! every row of a slice where its rows are a multiple of a line apart (each row's boundaries then
//! fall at the same columns), else one a row; the columns before the first boundary and after
//! the last are copied by the GPU into a region of the command buffer's arena (`upload.rs`), its
//! own lines whole ([`region`]), and from there by the CPU into the pages, by a host step of a
//! batch of its own once that copy completed ([`ShadowSplices::drain`], `Deferred::ShadowSplice`):
//! a store of the CPU changes the bytes it stores alone, so its own writes beside them stay. A
//! row whose texels hold no whole line goes to the CPU whole. A texel the boundary cuts goes to
//! the CPU with its row's edge. The encoder notes the CPU's writes of the truth's bytes after the
//! step, so the commands after it order after them; and the work after it goes in a later batch,
//! which the step precedes.
//!
//! **Merges** ([`MergeDrain`]). A merge stores the words its commands changed (`shadow_merge.rs`):
//! a GPU store into a line writes it back whole, the bytes the commands left in it, which the CPU
//! may write meanwhile, among them, wherever the rows end. So over such pages, on such a device, a
//! merge in place goes on the host: its keep and its copy of the texels the commands left go
//! into a `Shared` scratch, as two parts, and the host step compares them, eight bytes at a
//! time, storing each byte that differs into the pages (`WorkCounts::shadow_merges_on_host`).
//!
//! **Where the copy stays.** A copy back or merge between a render pass's instances or before its
//! tile dispatch (`render/pass_steps.rs`), into a pass's held command buffer
//! (`render/inline_splits.rs`), or at a segment's end (`EncoderCore::end_segment`), where no host
//! step may follow before the commands after it that read the bytes; a rotated texture's, a
//! heap's two-plane or tiled texture's, a heap's depth or stencil texture's; a merge through a
//! copy (`shadow_merge.rs`: a texture's first byte inside a word, through a staging buffer, or
//! with bytes in its tail); and a copy back or merge into the runtime's own memory (a `Shared`
//! buffer's): each goes as before (`docs/FAST-PATHS.md`, known gaps). A shadowed texture whose
//! merges go on the host opens no run of passes (`pass_runs.rs`): each pass merges as its encoder
//! ends.

use super::buffer::Buffer;
use super::encoder::RecordedSegment;
use super::error::{internal, Error};
use super::in_place::{Access, InPlaceChecks, PagesClass, RowEnds, Verdict};
use super::shadow_merge::ShadowMerge;
use super::upload::{Upload, Uploads};
use super::Device;
use ash::vk;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// The alignment of each edge's bytes in the region: a multiple of every texel's size a copy
/// into it takes (1, 2, 4, 8 or 16 bytes), as `bufferOffset` must be.
const EDGE_ALIGNMENT: u64 = 16;

/// Whether a copy back into host pages of `class` is spliced, by the device's check: where the
/// copies or the shader writes of texels that share a line with bytes beside them failed, or have
/// no answer yet.
pub(crate) fn splices(checks: &InPlaceChecks, class: PagesClass) -> bool {
    [Access::Copies, Access::Writes].into_iter().any(|access| {
        matches!(
            checks.decided(class, access, RowEnds::InLines),
            Verdict::Failed | Verdict::Pending
        )
    })
}

/// The class of host pages a no-copy buffer's bytes lie in, as the device's check measures them:
/// `None` for a buffer over no host pages, or over a copy of them (`copied.rs`), which the CPU
/// does not write while a command buffer runs.
pub(super) fn class(buffer: &Buffer) -> Option<PagesClass> {
    if buffer.host_pages().is_none() || buffer.copied() {
        return None;
    }
    Some(match buffer.over_dma_buf() {
        true => PagesClass::DmaBuf,
        false => PagesClass::Pointer,
    })
}

/// The columns of a row of `width` texels of `texel` bytes, whose first byte lies `phase` bytes
/// past a line of `line` bytes, that fill whole lines: from the first texel at or past the first
/// line boundary to the last texel that ends at or before the last. `(0, 0)` where none does.
fn columns(phase: u64, width: u32, texel: u64, line: u64) -> (u32, u32) {
    let (first, last) = boundaries(phase, u64::from(width) * texel, line);
    let (c0, c1) = (first.div_ceil(texel), last / texel);
    match c1 > c0 {
        // At most the row's texels: the last boundary lies at or before its end.
        true => (c0 as u32, c1 as u32),
        false => (0, 0),
    }
}

/// The first and last line boundaries of `bytes` bytes whose first lies `phase` bytes past a line
/// of `line` bytes, counted from that first byte; the last before the first where no line lies
/// whole among them.
fn boundaries(phase: u64, bytes: u64, line: u64) -> (u64, u64) {
    let first = (line - phase % line) % line;
    let end = phase % line + bytes;
    let last = (end - end % line).saturating_sub(phase % line);
    (first, last)
}

/// A copy back cut for its splice: the GPU's copies `R` of whole lines into the buffer
/// (`whole`) and of the rest into the region (`edges`, their offsets from its start), and the
/// CPU's pieces: each one's host address, length and byte of the region.
#[derive(Debug, Default)]
pub(crate) struct Cut<R> {
    pub(crate) whole: Vec<R>,
    pub(crate) edges: Vec<R>,
    pub(crate) pieces: Vec<(u64, u64, u64)>,
    /// The bytes of the region the edges take.
    pub(crate) length: u64,
}

impl<R> Cut<R> {
    /// Take `bytes` bytes of the region for an edge: where they start.
    fn take(&mut self, bytes: u64) -> u64 {
        let at = self.length.next_multiple_of(EDGE_ALIGNMENT);
        self.length = at + bytes;
        at
    }
}

/// `regions` of a copy back from a 2D texture's image into a `VkBuffer` whose byte `v` lies at
/// the host address `v + base` (wrapping), `texel` bytes a texel, cut for lines of `line` bytes
/// (the module docs' **How**).
pub(crate) fn image(
    regions: &[vk::BufferImageCopy],
    texel: u64,
    base: u64,
    line: u64,
) -> Cut<vk::BufferImageCopy> {
    let mut cut = Cut::default();
    for region in regions {
        let extent = region.image_extent;
        let layers = region.image_subresource.layer_count;
        if extent.width == 0 || extent.height == 0 || extent.depth == 0 || layers == 0 {
            continue;
        }
        let row = match region.buffer_row_length {
            0 => extent.width,
            row => row,
        };
        let pitch = u64::from(row) * texel;
        let height = match region.buffer_image_height {
            0 => extent.height,
            height => height,
        };
        let slice_pitch = u64::from(height) * pitch;
        // Each slice apart: a layer of an array's (whose depth is 1), or a depth slice.
        for slice in 0..layers.max(extent.depth) {
            let (layer, z) = match layers > 1 {
                true => (slice, 0),
                false => (0, slice),
            };
            let start = region.buffer_offset + u64::from(slice) * slice_pitch;
            let place = Place {
                region,
                layer,
                z,
                row,
                pitch,
                texel,
            };
            match pitch.is_multiple_of(line) {
                true => cut.band(&place, (0, extent.height), start, (base, line)),
                false => {
                    for y in 0..extent.height {
                        let at = start + u64::from(y) * pitch;
                        cut.band(&place, (y, 1), at, (base, line));
                    }
                }
            }
        }
    }
    cut
}

/// Where a band of a region's rows lies: the region, its slice's layer and depth, its rows'
/// length in texels and their pitch, and a texel's bytes.
struct Place<'a> {
    region: &'a vk::BufferImageCopy,
    layer: u32,
    z: u32,
    row: u32,
    pitch: u64,
    texel: u64,
}

impl Cut<vk::BufferImageCopy> {
    /// Cut `rows` rows of `place` from row `y`, whose first byte is the `VkBuffer`'s `at`, each
    /// cut at the same columns (one row, or rows a multiple of a line apart).
    fn band(
        &mut self,
        place: &Place<'_>,
        (y, rows): (u32, u32),
        at: u64,
        (base, line): (u64, u64),
    ) {
        let region = place.region;
        let width = region.image_extent.width;
        let phase = at.wrapping_add(base) % line;
        let (c0, c1) = columns(phase, width, place.texel, line);
        let subresource = vk::ImageSubresourceLayers {
            base_array_layer: region.image_subresource.base_array_layer + place.layer,
            layer_count: 1,
            ..region.image_subresource
        };
        let offset = |x: u32| vk::Offset3D {
            x: region.image_offset.x + x as i32,
            y: region.image_offset.y + y as i32,
            z: region.image_offset.z + place.z as i32,
        };
        if c1 > c0 {
            self.whole.push(vk::BufferImageCopy {
                buffer_offset: at + u64::from(c0) * place.texel,
                buffer_row_length: place.row,
                buffer_image_height: 0,
                image_subresource: subresource,
                image_offset: offset(c0),
                image_extent: vk::Extent3D {
                    width: c1 - c0,
                    height: rows,
                    depth: 1,
                },
            });
        }
        let (head, tail) = match c1 > c0 {
            true => ((0, c0), (c1, width)),
            false => ((0, 0), (0, width)),
        };
        for (x0, x1) in [head, tail].into_iter().filter(|(x0, x1)| x1 > x0) {
            let bytes = u64::from(x1 - x0) * place.texel;
            let into = self.take(bytes * u64::from(rows));
            self.edges.push(vk::BufferImageCopy {
                buffer_offset: into,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: subresource,
                image_offset: offset(x0),
                image_extent: vk::Extent3D {
                    width: x1 - x0,
                    height: rows,
                    depth: 1,
                },
            });
            for r in 0..u64::from(rows) {
                let from = at + r * place.pitch + u64::from(x0) * place.texel;
                self.pieces
                    .push((from.wrapping_add(base), bytes, into + r * bytes));
            }
        }
    }
}

/// `back`, the copies of a staging buffer's rows into a `VkBuffer` whose byte `v` lies at the
/// host address `v + base` (wrapping), cut for lines of `line` bytes: each copy's bytes between
/// its first and last line boundaries go straight in, and the rest through the region.
pub(crate) fn staged(back: &[vk::BufferCopy], base: u64, line: u64) -> Cut<vk::BufferCopy> {
    let mut cut = Cut::default();
    for copy in back.iter().filter(|copy| copy.size > 0) {
        let phase = copy.dst_offset.wrapping_add(base) % line;
        let (first, last) = boundaries(phase, copy.size, line);
        let edges = match last > first {
            true => {
                cut.whole.push(vk::BufferCopy {
                    src_offset: copy.src_offset + first,
                    dst_offset: copy.dst_offset + first,
                    size: last - first,
                });
                [(0, first), (last, copy.size)]
            }
            false => [(0, 0), (0, copy.size)],
        };
        for (a, b) in edges.into_iter().filter(|(a, b)| b > a) {
            let into = cut.take(b - a);
            cut.edges.push(vk::BufferCopy {
                src_offset: copy.src_offset + a,
                dst_offset: into,
                size: b - a,
            });
            let from = copy.dst_offset + a;
            cut.pieces.push((from.wrapping_add(base), b - a, into));
        }
    }
    cut
}

/// A region of `length` bytes of `uploads`, its first byte at a line of `line` bytes and its
/// length a multiple of one: no line holds a byte of another region, so the GPU's copies into it
/// write back no line another region's bytes lie in, as a host step writes those of a later batch
/// while the GPU runs this one's.
pub(super) fn region(
    uploads: &mut Uploads,
    device: &Device,
    length: u64,
    line: u64,
) -> Result<Upload, Error> {
    let rounded = length.next_multiple_of(line);
    let region = uploads.unset(device, rounded + line)?;
    let (chunk, offset) = region.buffer();
    let Some(at) = chunk.mapped_contents() else {
        return Err(internal("an arena chunk the CPU does not map"));
    };
    let host = (at.as_ptr().addr() as u64).wrapping_add(offset);
    let pad = (line - host % line) % line;
    let region = region.part(pad, rounded);
    // Its offset in its `VkBuffer` takes every texel's size too: a chunk's `VkBuffer` lies from
    // the start of its memory, which the CPU maps from a page.
    if !region.offset().is_multiple_of(EDGE_ALIGNMENT) {
        return Err(internal(
            "an arena region a line apart from its offset's alignment",
        ));
    }
    Ok(region)
}

/// A copy back's splice, cut and its region taken: what the copy records.
pub(super) struct Splice {
    pub(super) copies: SpliceCopies,
    /// The truth's `VkBuffer`, as the copy is recorded.
    pub(super) truth: vk::Buffer,
    /// The region's `VkBuffer`.
    pub(super) region: vk::Buffer,
    /// What the host step copies.
    pub(super) drain: SpliceDrain,
}

/// The GPU's copies of a splice, their offsets in the region its `VkBuffer`'s.
pub(super) enum SpliceCopies {
    /// From a 2D texture's image.
    Image {
        whole: Vec<vk::BufferImageCopy>,
        edges: Vec<vk::BufferImageCopy>,
    },
    /// From the staging buffer: a texture buffer's, or a 2D texture's after its image's copy
    /// into it.
    Staged {
        whole: Vec<vk::BufferCopy>,
        edges: Vec<vk::BufferCopy>,
    },
}

/// A copy back cut for its splice, its region not yet taken: the truth's buffer and `VkBuffer`,
/// and the cut.
pub(super) struct SplicePlan {
    pub(super) buffer: Buffer,
    pub(super) truth: vk::Buffer,
    pub(super) cut: PlannedCut,
}

/// A [`SplicePlan`]'s cut, of either form.
pub(super) enum PlannedCut {
    Image(Cut<vk::BufferImageCopy>),
    Staged(Cut<vk::BufferCopy>),
}

impl SplicePlan {
    /// The bytes of the region its edges take.
    pub(super) fn length(&self) -> u64 {
        match &self.cut {
            PlannedCut::Image(cut) => cut.length,
            PlannedCut::Staged(cut) => cut.length,
        }
    }

    /// The splice, its edges in `region`.
    pub(super) fn place(self, region: Upload) -> Splice {
        let into = region.offset();
        let (copies, pieces) = match self.cut {
            PlannedCut::Image(mut cut) => {
                for edge in &mut cut.edges {
                    edge.buffer_offset += into;
                }
                let copies = SpliceCopies::Image {
                    whole: cut.whole,
                    edges: cut.edges,
                };
                (copies, cut.pieces)
            }
            PlannedCut::Staged(mut cut) => {
                for edge in &mut cut.edges {
                    edge.dst_offset += into;
                }
                let copies = SpliceCopies::Staged {
                    whole: cut.whole,
                    edges: cut.edges,
                };
                (copies, cut.pieces)
            }
        };
        Splice {
            copies,
            truth: self.truth,
            region: region.raw(),
            drain: SpliceDrain {
                _buffer: self.buffer,
                region,
                pieces,
            },
        }
    }
}

/// The CPU's part of one copy back's splice: the pieces it copies from the region into the
/// truth's pages.
pub(super) struct SpliceDrain {
    /// Kept until the CPU's copy, whose command buffer keeps its pages until it completes.
    _buffer: Buffer,
    region: Upload,
    /// Each piece's host address, its length and its byte of the region.
    pieces: Vec<(u64, u64, u64)>,
}

/// The CPU's part of a shadow's merge on the host (`shadow_merge.rs`, **On the host**): the
/// texels the commands left and those kept before them, in the merge's scratch, compared row by
/// row, and each byte that differs stored into the truth's pages.
pub(super) struct MergeDrain {
    /// Kept until the CPU's merge, whose command buffer keeps its pages until it completes.
    pub(super) _buffer: Buffer,
    /// The merge, which holds its scratch.
    pub(super) _merge: Arc<ShadowMerge>,
    /// The host addresses of the kept and the changed parts' first bytes.
    pub(super) kept: u64,
    pub(super) changed: u64,
    /// Each row's host address in the pages, its byte of each part, and its bytes.
    pub(super) rows: Vec<(u64, u64, u64)>,
}

impl MergeDrain {
    /// Store each byte of each row whose changed part differs from its kept part, words of eight
    /// compared first: the bytes it stores alone change, so the CPU's writes to every other byte
    /// meanwhile stay.
    fn run(&self) {
        for &(host, at, length) in &self.rows {
            let length = length as usize;
            // SAFETY: both parts hold `length` bytes from `at` (the merge's scratch, a `Shared`
            // buffer the merge holds, mapped while it lives), which the GPU's copies wrote before
            // the batches this step follows completed, and which no command writes any more.
            let (kept, changed) = unsafe {
                (
                    std::slice::from_raw_parts((self.kept + at) as *const u8, length),
                    std::slice::from_raw_parts((self.changed + at) as *const u8, length),
                )
            };
            // The bytes of each word of eight that differs, then those past the last word.
            let ((kept_words, _), (changed_words, _)) =
                (kept.as_chunks::<8>(), changed.as_chunks::<8>());
            let candidates = (kept_words.iter().zip(changed_words).enumerate())
                .filter(|(_, (kept, changed))| kept != changed)
                .flat_map(|(word, _)| word * 8..word * 8 + 8)
                .chain(kept_words.len() * 8..length);
            for k in candidates.filter(|&k| kept[k] != changed[k]) {
                // SAFETY: the row's host bytes lie in the truth's pages, which stay mapped while
                // the command buffer runs (its segment's uses name the buffer, whose pages it
                // holds until it completes); another thread may write them meanwhile, so the
                // store is an atomic byte's.
                let byte = unsafe { AtomicU8::from_ptr((host + k as u64) as *mut u8) };
                byte.store(changed[k], Ordering::Relaxed);
            }
        }
    }
}

/// What a host step writes into the truth's pages for one copy of a shadow.
pub(super) enum HostWrite {
    /// A copy back's spliced edges.
    Splice(SpliceDrain),
    /// A merge's changed bytes.
    Merge(MergeDrain),
}

/// The CPU's parts of an encoder's copies back spliced and merges on the host at one place
/// (`Deferred::ShadowSplice`): held by its segment until its command buffer completes.
pub(super) struct ShadowSplices(pub(super) Vec<HostWrite>);

impl ShadowSplices {
    /// The host step, once every batch submitted before its own completed, the GPU's copies into
    /// the regions and scratches among them (each batch ends with the barrier to the host): each
    /// splice's pieces copied from its region into the pages, and each merge's changed bytes
    /// stored there; each of the CPU's stores changes the bytes it stores alone. It records
    /// nothing; the merges' time is counted on `device`.
    pub(super) fn drain(&self, device: &Device) -> RecordedSegment {
        let started = Instant::now();
        let mut merges = 0;
        for write in &self.0 {
            match write {
                HostWrite::Splice(drain) => {
                    for &(host, length, at) in &drain.pieces {
                        // SAFETY: the piece's host bytes lie in the truth's pages (its texels'
                        // bytes, `image` and `staged`), which stay mapped while the command
                        // buffer runs (its segment's uses name the buffer, whose pages it holds
                        // until it completes); the region is memory of the runtime's own, apart
                        // from them, and no command writes it any more.
                        unsafe { drain.region.copy_out(at, host as *mut u8, length) };
                    }
                }
                HostWrite::Merge(drain) => {
                    drain.run();
                    merges += 1;
                }
            }
        }
        if merges > 0 {
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            device.shared().counts.shadow_merged_on_host(merges, nanos);
        }
        RecordedSegment::submitted(Vec::new())
    }
}
