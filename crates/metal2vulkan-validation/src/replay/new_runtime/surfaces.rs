//! Textures over a surface's plane (`newTextureWithDescriptor:iosurface:plane:`), played as the
//! product's per-shape form makes them: one texture of the runtime's for each plane and shape, made
//! for views (`rt::Device::new_texture_for_views`) with the usage Metal gives a texture whose use
//! it does not know, and every recorded surface texture, and every view of one, a view of it at
//! the texture's own index. A plane is the receiver buffer, which stands for the surface's pages,
//! the plane's index, and its first byte and row pitch there; a shape is a texture's width, height
//! and bytes a texel, so a texture of another pixel format of the same bytes a texel is a view of
//! the same shape, and whether shaders may write it. A plain shape, of a texture whose usage has
//! neither shader write nor shader atomic (a storage image's operation) and is not Metal's unknown,
//! is made with shader read and render-target usage alone, so it views as its own format and that
//! format's sRGB sibling; a texture or view of another format over it is refused, counted. A shape
//! shaders may write is made with the usage Metal gives a texture whose use it does not know, with
//! shader atomic where the texture declares it, and views as every format of its bytes a texel; it
//! serves a later texture of its extent that does not declare it, and a texture that declares it
//! over a plane whose shape of its extent has none is another shape of the plane. Textures that
//! declare pixel-format-view usage over a plain shape are counted. Textures over the same pages at
//! another byte or pitch are another plane, and nothing moves between two planes but through the
//! buffer, at the recorded copies.
//!
//! The shapes hold the texels; the receiver buffer holds the pages' bytes as the stream wrote
//! them. Every shape is current over its whole extent, by the product's three rules:
//! - A new shape takes the buffer's bytes before anything reads it, once the dirty shapes are
//!   copied into the buffer, by blits in a command buffer of the play's own committed then: after
//!   every earlier commit, with no wait. The same shape made again is a view and costs nothing.
//! - A shape is dirty once a committed command buffer may have written it, until it is copied into
//!   the buffer. A copy back to the guest's RAM first copies every dirty shape of the buffer, and
//!   every shape its own command buffer may have written before, into the buffer, in the copy's
//!   own blit encoder.
//! - At the end of an encoder that may have written a shape, the write is copied into each other
//!   shape of its plane that a live texture views or that is dirty, through the buffer at the
//!   plane's byte and pitch, in blit encoders of the same command buffer; the plane's other shapes
//!   go. Every two shapes of a plane overlap from its first texel; a texel the write's rows reach
//!   only in part keeps its bytes, counted. An encoder that may have written two shapes of one
//!   plane is counted, and the shape noted last is taken as the plane's write.
//!
//! An encoder may have written a shape through a pass's attachment, a blit's destination, a
//! texture bound at a slot the bound pipeline state's interface writes
//! (`rt::RenderPipelineState::texture_accesses`), and, where no interface is at hand (a slot of a
//! stage it does not answer, a skipped state, a `useResource:usage:`), a texture whose usage lets a
//! shader write it, counted. A write through an argument buffer with no use named is not seen.
//!
//! A shape no texture views stays while it is one of the buffer's four most recently used such
//! shapes, or dirty. Besides:
//! - A copy from the guest's RAM into the buffer also copies the bytes into every shape they
//!   reach, in the copy's own blit encoder: the guest's bytes win over a shape's write not yet in
//!   the buffer, as the CPU's bytes win in the product's fill of the buffer from the pages.
//! - A recorded CPU write into the buffer reaches the texels it covers of every shape through the
//!   runtime's upload, which orders it as Metal orders a CPU write.
//! - `m2v.externalWriteDone` on a receiver buffer, another API's write landed in it (a decoded
//!   picture): every shape of the buffer owes a fill from it, its own write lost, as the product's
//!   shapes owe one after a video command buffer. The first command buffer whose encoders use a
//!   texture of an owing shape's buffer fills, before its commit, every owing shape of the buffers
//!   it uses, by blits in a command buffer of the play's own committed then; a picture no command
//!   buffer used before the next one landed is never copied, counted. From the first such record,
//!   the recorded CPU writes into that buffer, which stand for the other API's, reach its shapes
//!   only this way.
//! - Any other command that names a receiver buffer as a buffer is refused: the buffer does not
//!   hold the texels.
//! - A texture of a format that reads across a surface's planes (a two-plane YCbCr or XR10 format
//!   over a surface of two planes) is skipped and counted.

use super::super::covers::SlotStage;
use super::{failed, whole_slices, NewRuntime, Object};
use crate::replay::args::{refuse, Fields, SURFACE_FACT_KEYS, SURFACE_PLANE_KEYS};
use crate::replay::player::{Backend, Made, PlayError};
use crate::replay::stream::Call;
use crate::replay::value::Value;
use metal2vulkan_rt as rt;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;

/// How many shapes no texture views a receiver buffer keeps, the most recently used.
const UNREFERENCED_KEPT: usize = 4;

/// The commands that name a receiver buffer and are played through its shapes.
const COPIES: [&str; 3] = [
    "m2v.copyFromGuestRam",
    "m2v.copyFromGuestRamAt",
    "m2v.copyToGuestRam",
];

/// The first texel of a shape, where every copy of one starts.
const ORIGIN: rt::TexturePlace = rt::TexturePlace {
    slice: 0,
    level: 0,
    origin: [0; 3],
};

/// A plane: the receiver buffer, the plane's index, and its first byte and row pitch there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Span {
    buffer: u64,
    plane: u64,
    offset: u64,
    bytes_per_row: u64,
}

/// A shape: its width, height and bytes a texel, and whether shaders may write it (its texture of
/// the runtime's has a storage view).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Shape {
    width: u32,
    height: u32,
    texel: u64,
    storage: bool,
}

impl Span {
    /// The bytes of the buffer the texels of `shape` lie in.
    fn bytes(&self, shape: Shape) -> Range<u64> {
        let rows = u64::from(shape.height.max(1)) - 1;
        let end = self.offset + rows * self.bytes_per_row + u64::from(shape.width) * shape.texel;
        self.offset..end
    }

    /// `rows` rows of texels laid out in the buffer, as a blit copy names them.
    fn layout(&self, rows: u32) -> rt::BufferLayout {
        rt::BufferLayout {
            offset: self.offset,
            bytes_per_row: self.bytes_per_row,
            bytes_per_image: self.bytes_per_row * u64::from(rows),
        }
    }

    /// The texels of `shape` that bytes `written` of the buffer reach, as at most three regions (a
    /// first row in part, whole rows, a last row in part), each with the bytes of the buffer its
    /// rows lie in.
    fn reached(&self, shape: Shape, written: &Range<u64>) -> Vec<(rt::Region, Range<u64>)> {
        let (pitch, texel) = (self.bytes_per_row, shape.texel);
        let row = u64::from(shape.width) * texel;
        let bytes = self.bytes(shape);
        if pitch == 0 || texel == 0 || written.end <= bytes.start || bytes.end <= written.start {
            return Vec::new();
        }
        // From the plane's first byte.
        let start = written.start.max(bytes.start) - self.offset;
        let end = written.end.min(bytes.end) - self.offset;
        let columns = |r: u64| {
            let (first, last) = (start.max(r * pitch), end.min(r * pitch + row));
            (first < last).then(|| (first - r * pitch) / texel..(last - r * pitch).div_ceil(texel))
        };
        let piece = |rows: Range<u64>, columns: Range<u64>| {
            let region = rt::Region {
                origin: [columns.start as u32, rows.start as u32, 0],
                size: [
                    (columns.end - columns.start) as u32,
                    (rows.end - rows.start) as u32,
                    1,
                ],
            };
            let from = self.offset + rows.start * pitch + columns.start * texel;
            let to = self.offset + (rows.end - 1) * pitch + columns.end * texel;
            (region, from..to)
        };
        let (first, last) = (start / pitch, (end - 1) / pitch);
        let mut pieces = Vec::new();
        pieces.extend(columns(first).map(|c| piece(first..first + 1, c)));
        if last > first + 1 {
            pieces.push(piece(first + 1..last, 0..u64::from(shape.width)));
        }
        if last > first {
            pieces.extend(columns(last).map(|c| piece(last..last + 1, c)));
        }
        pieces
    }
}

impl Shape {
    fn size(self) -> [u32; 3] {
        [self.width, self.height, 1]
    }

    /// The bytes of its texels.
    fn bytes(self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * self.texel
    }

    /// The bytes of each row, and the rows, of its plane that it and `other` both cover: from the
    /// plane's first texel, as wide as the narrower one's rows and as high as the lower one.
    fn overlap(self, other: Shape) -> (u64, u32) {
        let row = |s: Shape| u64::from(s.width) * s.texel;
        (row(self).min(row(other)), self.height.min(other.height))
    }

    /// Of `bytes` of each of `rows` rows of a write through another shape of its plane: its
    /// texels the write covers whole, and whether a texel at the rows' end it covers only in part.
    fn covered(self, (bytes, rows): (u64, u32)) -> ([u32; 3], bool) {
        let width = bytes / self.texel;
        ([width as u32, rows, 1], width * self.texel < bytes)
    }
}

/// A texture of the runtime's that a plane's textures of one shape are views of.
struct Entry {
    shape: Shape,
    texture: rt::Texture,
    /// How many surface textures and views of it live.
    live: usize,
    /// Written since it was last copied into the buffer, by a committed command buffer.
    dirty: bool,
    /// Owes a fill from the buffer: another API wrote the buffer since its last fill.
    owed: bool,
    /// When a texture of it was last made or let go (`Surfaces::tick`).
    used: u64,
}

/// An open encoder's uses of surface textures.
struct Notes {
    /// Its command buffer, the stream's.
    command_buffer: u64,
    pipeline: Option<u64>,
    /// The surface textures its slots hold, by stage and index.
    slots: HashMap<(SlotStage, u64), u64>,
    /// The shapes it may have written, in the order noted.
    written: Vec<(Span, Shape)>,
}

/// The play's surface textures and their shapes.
#[derive(Default)]
pub(super) struct Surfaces {
    /// Each plane's shapes.
    ranges: HashMap<Span, Vec<Entry>>,
    /// Each live receiver buffer's planes.
    receivers: HashMap<u64, Vec<Span>>,
    /// Each live surface texture's and view's plane and shape.
    views: HashMap<u64, (Span, Shape)>,
    /// Of those, the ones whose usage lets a shader write them.
    writable: HashSet<u64>,
    /// The open encoders' uses.
    encoders: HashMap<u64, Notes>,
    /// Each command buffer's shapes its ended encoders may have written, dirty at its commit.
    pending: HashMap<u64, Vec<(Span, Shape)>>,
    /// Each command buffer's receiver buffers whose textures its encoders use, filled where
    /// owing at its commit.
    used: HashMap<u64, HashSet<u64>>,
    /// Each receiver buffer's pictures landed since its shapes were last filled.
    landed: HashMap<u64, u64>,
    /// The receiver buffers another API writes (`m2v.externalWriteDone`).
    external: HashSet<u64>,
    /// The queue of the play's own that the fills at a making and the refills go on.
    queue: Option<rt::CommandQueue>,
    /// The refills' command buffers not yet seen complete (`timed`).
    refilling: VecDeque<rt::CommandBuffer>,
    /// A count of makings and lettings go, for the shapes' recency.
    tick: u64,
    counts: Counts,
}

#[derive(Default)]
struct Counts {
    /// Surface textures made as views, and views of them.
    views: u64,
    /// Shapes made, each filled at its making, and textures made of a shape the plane held.
    made: u64,
    found: u64,
    /// Shapes shaders may write made over a plane that held a shape of their extent shaders may
    /// not write (the order case).
    order: u64,
    /// Shapes no texture views that went past the four kept, or at a write of another shape of
    /// their plane.
    dropped: u64,
    /// Dirty shapes copied into the buffer, at a new shape's making or a copy back to the guest's
    /// RAM, and their bytes.
    written_back: u64,
    written_back_bytes: u64,
    /// Shapes that took another shape's write at the end of the encoder that wrote it, the bytes
    /// they took, and the texels the write covered only in part, which kept their bytes.
    overlap_copies: u64,
    overlap_bytes: u64,
    straddled: u64,
    /// Encoders that may have written two shapes of one plane.
    unordered: u64,
    /// Uses taken as writes by the texture's usage alone, with no interface at hand, and draws'
    /// and dispatches' slots whose interface declares a write.
    usage_writes: u64,
    declared_writes: u64,
    /// Recorded CPU writes into a buffer after a shape's making.
    cpu_writes: u64,
    /// `m2v.externalWriteDone` records; the shapes filled for them at a use, and the bytes of
    /// those; the pictures a fill took, and those another landed over before any use.
    external_writes: u64,
    refills: u64,
    refill_bytes: u64,
    pictures_filled: u64,
    pictures_unfilled: u64,
    /// The GPU time of the refills' command buffers seen complete, and the longest one's.
    refill_nanos: u64,
    refill_max_nanos: u64,
    /// Copies from the guest's RAM into a shape, and copies back to the guest's RAM that read a
    /// buffer with shapes.
    copies_in: u64,
    copies_out: u64,
    /// The bytes of the texels the fills, write-backs, overlap copies and copies moved.
    bytes: u64,
    /// Textures of a format that reads across a surface's planes, skipped.
    across_planes: u64,
    /// Textures and views refused over a plain shape for a format neither its own nor that
    /// format's sRGB sibling, and textures over a plain shape that declare pixel-format-view usage.
    format_refusals: u64,
    plain_format_views: u64,
}

impl Surfaces {
    pub(super) fn is_empty(&self) -> bool {
        self.receivers.is_empty() && self.views.is_empty()
    }

    pub(super) fn is_receiver(&self, buffer: u64) -> bool {
        self.receivers.contains_key(&buffer)
    }

    /// Whether `encoder` is an open encoder the play notes surface uses in.
    pub(super) fn notes(&self, encoder: u64) -> bool {
        self.encoders.contains_key(&encoder)
    }

    /// Every shape of buffer `buffer` whose texels' bytes meet one of `spans`.
    fn meeting(&self, buffer: u64, spans: &[Range<u64>]) -> Vec<(Span, Shape, rt::Texture)> {
        let meets = |span: &Span, shape| {
            let bytes = span.bytes(shape);
            (spans.iter()).any(|s| s.start < bytes.end && bytes.start < s.end)
        };
        self.shapes_of(buffer, |span, entry| meets(span, entry.shape))
    }

    /// The shapes of buffer `buffer` that `pick` picks.
    fn shapes_of(
        &self,
        buffer: u64,
        pick: impl Fn(&Span, &Entry) -> bool,
    ) -> Vec<(Span, Shape, rt::Texture)> {
        let spans = self.receivers.get(&buffer).into_iter().flatten();
        (spans.flat_map(|span| {
            let entries = self.ranges.get(span).into_iter().flatten();
            entries
                .filter(|entry| pick(span, entry))
                .map(|entry| (*span, entry.shape, entry.texture.clone()))
                .collect::<Vec<_>>()
        }))
        .collect()
    }

    fn entry(&mut self, span: &Span, shape: Shape) -> Option<&mut Entry> {
        let entries = self.ranges.get_mut(span)?;
        entries.iter_mut().find(|entry| entry.shape == shape)
    }

    /// Marks shapes `shapes` clean: copied into the buffer, or refilled from it.
    fn clean(&mut self, shapes: &[(Span, Shape, rt::Texture)]) {
        for (span, shape, _) in shapes {
            if let Some(entry) = self.entry(span, *shape) {
                entry.dirty = false;
            }
        }
    }

    /// `object` released: a receiver buffer's shapes go with it, and a surface texture's or
    /// view's shape, no texture viewing it, may go (`sweep`).
    pub(super) fn released(&mut self, object: u64) {
        if let Some(ranges) = self.receivers.remove(&object) {
            for span in ranges {
                self.ranges.remove(&span);
            }
        }
        self.external.remove(&object);
        self.writable.remove(&object);
        self.encoders.remove(&object);
        self.pending.remove(&object);
        self.used.remove(&object);
        if let Some(landed) = self.landed.remove(&object) {
            self.counts.pictures_unfilled += landed;
        }
        let Some((span, shape)) = self.views.remove(&object) else {
            return;
        };
        self.tick += 1;
        let tick = self.tick;
        if let Some(entry) = self.entry(&span, shape) {
            entry.live = entry.live.saturating_sub(1);
            entry.used = tick;
        }
        self.sweep(span.buffer);
    }

    /// The shapes of buffer `buffer` no texture views go, but for the most recently used four and
    /// the dirty ones.
    fn sweep(&mut self, buffer: u64) {
        let Some(spans) = self.receivers.get(&buffer) else {
            return;
        };
        let mut idle: Vec<(u64, Span, Shape, bool)> = (spans.iter())
            .flat_map(|span| {
                let entries = self.ranges.get(span).into_iter().flatten();
                let idle = entries.filter(|entry| entry.live == 0);
                idle.map(|entry| (entry.used, *span, entry.shape, entry.dirty))
                    .collect::<Vec<_>>()
            })
            .collect();
        idle.sort_unstable_by_key(|shape| std::cmp::Reverse(shape.0));
        for (_, span, shape, dirty) in idle.into_iter().skip(UNREFERENCED_KEPT) {
            if !dirty {
                self.drop_shape(&span, shape);
            }
        }
    }

    fn drop_shape(&mut self, span: &Span, shape: Shape) {
        if let Some(entries) = self.ranges.get_mut(span) {
            entries.retain(|entry| entry.shape != shape);
            self.counts.dropped += 1;
        }
    }

    /// Command buffer `command_buffer` committed: the shapes its encoders may have written are
    /// dirty.
    pub(super) fn committed(&mut self, command_buffer: u64) {
        for (span, shape) in self.pending.remove(&command_buffer).unwrap_or_default() {
            if let Some(entry) = self.entry(&span, shape) {
                entry.dirty = true;
            }
        }
    }

    /// Notes that the open encoder `encoder` may have written surface texture `id`.
    fn note(&mut self, encoder: u64, id: u64) {
        let Some(&written) = self.views.get(&id) else {
            return;
        };
        if let Some(notes) = self.encoders.get_mut(&encoder) {
            notes.written.push(written);
        }
    }

    /// Adds the GPU time of the refills' command buffers that completed since the last call.
    pub(super) fn timed(&mut self) {
        while let Some(front) = self.refilling.front() {
            match front.status() {
                rt::CommandBufferStatus::Completed => {
                    let (start, end) = (front.gpu_start_time(), front.gpu_end_time());
                    let nanos = end.saturating_sub(start);
                    self.counts.refill_nanos += nanos;
                    self.counts.refill_max_nanos = self.counts.refill_max_nanos.max(nanos);
                }
                rt::CommandBufferStatus::Error => {}
                _ => return,
            }
            self.refilling.pop_front();
        }
    }

    /// The textures skipped: those of a format that reads across a surface's planes.
    pub(super) fn across_planes(&self) -> u64 {
        self.counts.across_planes
    }

    /// The executed counters.
    pub(super) fn counted(&self) -> [(&'static str, u64); 26] {
        let c = &self.counts;
        [
            ("surface_external_writes", c.external_writes),
            ("surface_refills", c.refills),
            ("surface_refill_bytes", c.refill_bytes),
            ("surface_refill_gpu_nanos", c.refill_nanos),
            ("surface_pictures_filled", c.pictures_filled),
            ("surface_pictures_unfilled", c.pictures_unfilled),
            ("surface_textures", c.views),
            ("surface_shapes", c.made),
            ("surface_shapes_found", c.found),
            ("surface_order_shapes", c.order),
            ("surface_shapes_dropped", c.dropped),
            ("surface_written_back", c.written_back),
            ("surface_written_back_bytes", c.written_back_bytes),
            ("surface_overlap_copies", c.overlap_copies),
            ("surface_overlap_bytes", c.overlap_bytes),
            ("surface_straddled", c.straddled),
            ("surface_unordered_writes", c.unordered),
            ("surface_usage_writes", c.usage_writes),
            ("surface_declared_writes", c.declared_writes),
            ("surface_cpu_writes", c.cpu_writes),
            ("surface_copies_in", c.copies_in),
            ("surface_copies_out", c.copies_out),
            ("surface_bytes", c.bytes),
            ("surface_format_refusals", c.format_refusals),
            ("surface_plain_format_views", c.plain_format_views),
            ("skipped_surfaces_across_planes", c.across_planes),
        ]
    }

    /// The report's `surfaces` line.
    pub(super) fn line(&self) -> String {
        let c = &self.counts;
        format!(
            "surfaces {} textures and views of {} shapes, each filled at its making ({} made again \
             as views, {} order-case shapes, {} dropped); {} overlap copies ({} bytes), {} straddled texels, {} \
             unordered writes, {} usage writes and {} declared slot writes (writes through an \
             argument buffer with no use named unseen, not counted); {} dirty shapes written back ({} bytes); {} CPU writes, \
             {} external writes, {} of them filled at a use and {} landed over before any (and \
             {} not used yet), filling {} shapes ({} bytes, {} ns of GPU time, the longest \
             command buffer {} ns), {} copies from the guest's RAM into a shape and {} back, {} \
             bytes in all; {} refused for a format a plain shape does not view as, {} declaring \
             pixel-format-view usage over a plain shape; {} across planes skipped",
            c.views,
            c.made,
            c.found,
            c.order,
            c.dropped,
            c.overlap_copies,
            c.overlap_bytes,
            c.straddled,
            c.unordered,
            c.usage_writes,
            c.declared_writes,
            c.written_back,
            c.written_back_bytes,
            c.cpu_writes,
            c.external_writes,
            c.pictures_filled,
            c.pictures_unfilled,
            self.landed.values().sum::<u64>(),
            c.refills,
            c.refill_bytes,
            c.refill_nanos,
            c.refill_max_nanos,
            c.copies_in,
            c.copies_out,
            c.bytes,
            c.format_refusals,
            c.plain_format_views,
            c.across_planes,
        )
    }
}

/// Whether a texture of `format` over a surface of two planes reads across them: a two-plane
/// YCbCr or XR10 format (`docs/REPLAY-STREAM.md`, section 7).
fn across_planes(format: rt::PixelFormat) -> bool {
    metal2vulkan::metal::format::ycbcr(format)
        || matches!(
            format,
            rt::PixelFormat::RGB10A82PXR10 | rt::PixelFormat::RGB10A82PXR10SRGB
        )
}

/// The stage whose slots a texture binding's selector names, and whether it binds a range.
fn binding(selector: &str) -> Option<(Option<SlotStage>, bool)> {
    let stage = match selector {
        "setTexture:atIndex:" | "setTextures:withRange:" => Some(SlotStage::Kernel),
        "setVertexTexture:atIndex:" | "setVertexTextures:withRange:" => Some(SlotStage::Vertex),
        "setFragmentTexture:atIndex:" | "setFragmentTextures:withRange:" => {
            Some(SlotStage::Fragment)
        }
        _ if selector.starts_with("set") && selector.contains("Texture") => None,
        _ => return None,
    };
    Some((stage, selector.contains("withRange:")))
}

impl NewRuntime {
    /// `newTextureWithDescriptor:iosurface:plane:` on receiver buffer `buffer`: a view of the
    /// shape of its plane, at the texture's own index.
    pub(super) fn surface_texture(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let d = self.descriptor_or_map(call, 0)?;
        let f = Fields::arg(call, 1, &SURFACE_PLANE_KEYS)?;
        let plane = f.required("plane", Fields::uint)?;
        let span = Span {
            buffer: call.receiver,
            plane,
            offset: f.required("offset", Fields::uint)?,
            bytes_per_row: f.required("bytesPerRow", Fields::uint)?,
        };
        if let Some(facts) = f.fields("facts", &SURFACE_FACT_KEYS)? {
            let planes = facts.required("planeCount", Fields::uint)?;
            if (planes == 0 && plane != 0) || (planes != 0 && plane >= planes) {
                return Err(refuse(call, format!("plane {plane} of {planes}")));
            }
            if planes >= 2 && across_planes(d.pixel_format) {
                self.surfaces.counts.across_planes += 1;
                return Ok(Some((Object::Skipped, Made::default())));
            }
        }
        let raw = u64::from(d.pixel_format.as_raw());
        let query = rt::TextureLevelQuery::plane_2d(raw, d.width.into(), d.height.into());
        let texel = self.device.texture_level_info(&query).bytes_per_block;
        // Storage: shader write, shader atomic (a storage image's operation) or Metal's unknown.
        let writes = d.usage.is_empty()
            || d.usage.contains(rt::TextureUsage::SHADER_WRITE)
            || d.usage.contains(rt::TextureUsage::SHADER_ATOMIC);
        let shape = Shape {
            width: d.width,
            height: d.height,
            texel,
            storage: writes,
        };
        let (entry, shape) = self.surface_entry(call, buffer, span, shape, &d)?;
        if !shape.storage && d.usage.contains(rt::TextureUsage::PIXEL_FORMAT_VIEW) {
            self.surfaces.counts.plain_format_views += 1;
        }
        let (levels, slices) = (0..d.mipmap_level_count, whole_slices(&d));
        let make = |index| {
            let (levels, slices) = (levels.clone(), slices.clone());
            let (format, kind) = (d.pixel_format, d.texture_type);
            entry.new_texture_view_with_swizzle(format, kind, levels, slices, d.swizzle, index)
        };
        let gaps = self.substitutes.gaps;
        let made = self.texture_or_substitute(call, d.usage, &d, make)?;
        self.surfaces.counts.views += 1;
        if self.substitutes.gaps == gaps {
            self.surface_viewed(call.result, (span, shape), writes);
        }
        Ok(made)
    }

    /// Texture `id` made a view of shape `shape` of plane `span`.
    fn surface_viewed(&mut self, id: u64, (span, shape): (Span, Shape), writes: bool) {
        let surfaces = &mut self.surfaces;
        surfaces.views.insert(id, (span, shape));
        if writes {
            surfaces.writable.insert(id);
        }
        surfaces.tick += 1;
        let tick = surfaces.tick;
        if let Some(entry) = surfaces.entry(&span, shape) {
            entry.live += 1;
            entry.used = tick;
        }
    }

    /// The shape a texture of `shape` over plane `span` is a view of, and that shape: the plane's
    /// where it holds one, a shape shaders may write serving a texture that does not, else made
    /// and filled from the buffer, after the buffer's dirty shapes are copied into it
    /// (`Surfaces`). A shape is made with shader read, render-target and pixel-format-view usage,
    /// and shader write only for a texture whose usage has it or is Metal's unknown.
    fn surface_entry(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
        span: Span,
        shape: Shape,
        descriptor: &rt::TextureDescriptor,
    ) -> Result<(rt::Texture, Shape), PlayError> {
        let wider = Shape {
            storage: true,
            ..shape
        };
        for shape in [wider, shape] {
            if let Some(entry) = self.surfaces.entry(&span, shape) {
                let texture = entry.texture.clone();
                if !shape.storage {
                    self.surface_plain_format(call, &texture, descriptor.pixel_format)?;
                }
                self.surfaces.counts.found += 1;
                return Ok((texture, shape));
            }
        }
        let narrower = Shape {
            storage: false,
            ..shape
        };
        if shape.storage && self.surfaces.entry(&span, narrower).is_some() {
            self.surfaces.counts.order += 1;
        }
        let mut d = *descriptor;
        d.usage = rt::TextureUsage::SHADER_READ | rt::TextureUsage::RENDER_TARGET;
        if shape.storage {
            // The usage Metal gives a texture whose use it does not know.
            d.usage |= rt::TextureUsage::SHADER_WRITE | rt::TextureUsage::PIXEL_FORMAT_VIEW;
        }
        if descriptor.usage.contains(rt::TextureUsage::SHADER_ATOMIC) {
            d.usage |= rt::TextureUsage::SHADER_ATOMIC;
        }
        d.swizzle = rt::TextureSwizzleChannels::DEFAULT;
        let texture = (self.device.new_texture_for_views(&d)).map_err(failed(call))?;
        let dirty = (self.surfaces).shapes_of(span.buffer, |_, entry| entry.dirty);
        let queue = self.surface_queue(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        for (span, shape, entry) in &dirty {
            let layout = span.layout(shape.height);
            (blit.copy_from_texture_to_buffer(entry, ORIGIN, shape.size(), buffer, layout))
                .map_err(failed(call))?;
        }
        let layout = span.layout(shape.height);
        (blit.copy_from_buffer(buffer, layout, shape.size(), &texture, ORIGIN))
            .map_err(failed(call))?;
        blit.end_encoding();
        self.commit(&queue, command_buffer, 0)
            .map_err(failed(call))?;
        let surfaces = &mut self.surfaces;
        surfaces.clean(&dirty);
        for (_, shape, _) in &dirty {
            surfaces.counts.written_back += 1;
            surfaces.counts.written_back_bytes += shape.bytes();
            surfaces.counts.bytes += shape.bytes();
        }
        surfaces.counts.made += 1;
        surfaces.counts.bytes += shape.bytes();
        surfaces.tick += 1;
        (surfaces.ranges.entry(span).or_default()).push(Entry {
            shape,
            texture: texture.clone(),
            live: 0,
            dirty: false,
            owed: false,
            used: surfaces.tick,
        });
        let ranges = surfaces.receivers.entry(span.buffer).or_default();
        if !ranges.contains(&span) {
            ranges.push(span);
        }
        surfaces.sweep(span.buffer);
        Ok((texture, shape))
    }

    /// Refused, counted, where a texture or view of `format` would view plain shape `texture`,
    /// which views as its own format and that format's sRGB sibling alone.
    fn surface_plain_format(
        &mut self,
        call: &Call,
        texture: &rt::Texture,
        format: rt::PixelFormat,
    ) -> Result<(), PlayError> {
        let own = texture.descriptor().pixel_format;
        if metal2vulkan::metal::format::views_without_usage(own, format) {
            return Ok(());
        }
        self.surfaces.counts.format_refusals += 1;
        let why = format!("a plain surface shape of {own:?} viewed as {format:?}");
        Err(refuse(call, why))
    }

    /// Before `call` makes a view of `format` of the texture it is sent to: refused, counted,
    /// where that is a surface texture of a plain shape that does not view as `format`.
    pub(super) fn surface_view(
        &mut self,
        call: &Call,
        format: rt::PixelFormat,
    ) -> Result<(), PlayError> {
        let Some(&(span, shape)) = self.surfaces.views.get(&call.receiver) else {
            return Ok(());
        };
        let Some(entry) = (!shape.storage)
            .then(|| self.surfaces.entry(&span, shape))
            .flatten()
        else {
            return Ok(());
        };
        let texture = entry.texture.clone();
        self.surface_plain_format(call, &texture, format)
    }

    /// The queue of the play's own that fills and refills go on.
    fn surface_queue(&mut self, call: &Call) -> Result<rt::CommandQueue, PlayError> {
        if let Some(queue) = &self.surfaces.queue {
            return Ok(queue.clone());
        }
        let queue = self.device.new_command_queue(64).map_err(failed(call))?;
        Ok(self.surfaces.queue.insert(queue).clone())
    }

    /// `m2v.externalWriteDone` on buffer `buffer`: another API's write landed in it, and every
    /// shape of its planes owes a fill from it, its own write lost, made at the first commit
    /// that uses one (`surface_fill_owed`). The one place the record's position decides.
    pub(super) fn external_write_done(
        &mut self,
        call: &Call,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let surfaces = &mut self.surfaces;
        surfaces.counts.external_writes += 1;
        surfaces.external.insert(call.receiver);
        let spans = surfaces
            .receivers
            .get(&call.receiver)
            .cloned()
            .unwrap_or_default();
        let mut any = false;
        for span in spans {
            for entry in surfaces.ranges.get_mut(&span).into_iter().flatten() {
                entry.owed = true;
                entry.dirty = false;
                any = true;
            }
        }
        if any {
            *surfaces.landed.entry(call.receiver).or_default() += 1;
        }
        Ok(None)
    }

    /// Before command buffer `call.receiver` commits: every owing shape of the receiver buffers
    /// its encoders used is filled from its buffer, by blits in a command buffer of the play's own
    /// committed now, ahead of it.
    pub(super) fn surface_fill_owed(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(used) = self.surfaces.used.remove(&call.receiver) else {
            return Ok(());
        };
        let mut owing = Vec::new();
        for id in used {
            let shapes = (self.surfaces).shapes_of(id, |_, entry| entry.owed);
            if !shapes.is_empty() {
                owing.push((id, self.buffer_of(call, id)?, shapes));
            }
        }
        if owing.is_empty() {
            return Ok(());
        }
        let queue = self.surface_queue(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        for (_, buffer, shapes) in &owing {
            for (span, shape, entry) in shapes {
                let layout = span.layout(shape.height);
                (blit.copy_from_buffer(buffer, layout, shape.size(), entry, ORIGIN))
                    .map_err(failed(call))?;
            }
        }
        blit.end_encoding();
        self.surfaces.refilling.push_back(command_buffer.clone());
        self.commit(&queue, command_buffer, 0)
            .map_err(failed(call))?;
        let surfaces = &mut self.surfaces;
        for (id, _, shapes) in &owing {
            for (span, shape, _) in shapes {
                if let Some(entry) = surfaces.entry(span, *shape) {
                    entry.owed = false;
                }
                surfaces.counts.refills += 1;
                surfaces.counts.refill_bytes += shape.bytes();
                surfaces.counts.bytes += shape.bytes();
            }
            if let Some(landed) = surfaces.landed.remove(id) {
                surfaces.counts.pictures_filled += 1;
                surfaces.counts.pictures_unfilled += landed.saturating_sub(1);
            }
        }
        Ok(())
    }

    /// Before `call` plays: refused where it names a receiver buffer as a buffer, other than a
    /// copy between the guest's RAM and it, or is sent to one.
    pub(super) fn surface_names(&mut self, call: &Call, named: &[u64]) -> Result<(), PlayError> {
        let copy = COPIES.contains(&&*call.selector);
        if let Some(buffer) = named.iter().find(|id| self.surfaces.is_receiver(**id)) {
            if !copy {
                let why = format!("#{buffer} is a surface's receiver buffer, named as a buffer");
                return Err(refuse(call, why));
            }
        }
        let own = matches!(
            &*call.selector,
            "newTextureWithDescriptor:iosurface:plane:"
                | "didModifyRange:"
                | "m2v.externalWriteDone"
        );
        if self.surfaces.is_receiver(call.receiver) && !own {
            let why = "a call on a surface's receiver buffer, which does not hold the texels";
            return Err(refuse(call, why));
        }
        Ok(())
    }

    /// After `call` made `object`: an encoder's uses of surface textures are noted from here, a
    /// render pass's attachments as written; a view of a surface texture is a view of its shape.
    pub(super) fn surface_made(&mut self, call: &Call, object: &Object) {
        match object {
            Object::Render(_) | Object::Compute(_) | Object::Blit(_) => {
                let mut notes = Notes {
                    command_buffer: call.receiver,
                    pipeline: None,
                    slots: HashMap::new(),
                    written: Vec::new(),
                };
                if let (Object::Render(_), Some(Value::Object(pass))) = (object, call.args.first())
                {
                    if let Some(Object::Pass(pass)) = self.objects.get(pass) {
                        let views = &self.surfaces.views;
                        let attached = pass.textures().filter_map(|id| views.get(&id).copied());
                        notes.written.extend(attached);
                        let used = self.surfaces.used.entry(call.receiver).or_default();
                        used.extend(notes.written.iter().map(|(span, _)| span.buffer));
                    }
                }
                self.surfaces.encoders.insert(call.result, notes);
            }
            Object::Texture(_) if call.selector.starts_with("newTextureView") => {
                if let Some(&of) = self.surfaces.views.get(&call.receiver) {
                    let writes = self.surfaces.writable.contains(&call.receiver);
                    self.surface_viewed(call.result, of, writes);
                    self.surfaces.counts.views += 1;
                }
            }
            _ => {}
        }
    }

    /// Before `call` plays on an open encoder the play notes: the surface textures it binds, and
    /// the shapes it may write.
    pub(super) fn surface_encoded(&mut self, call: &Call) {
        let (selector, encoder) = (&*call.selector, call.receiver);
        let object = |at: usize| call.args.get(at).and_then(Value::as_object);
        if let Some(notes) = self.surfaces.encoders.get(&encoder) {
            let mut named = Vec::new();
            super::super::covers::named(&call.args, &mut named);
            let views = &self.surfaces.views;
            let buffers: Vec<u64> = (named.iter().filter_map(|id| views.get(id)))
                .map(|(span, _)| span.buffer)
                .collect();
            if !buffers.is_empty() {
                let command_buffer = notes.command_buffer;
                (self.surfaces.used.entry(command_buffer).or_default()).extend(buffers);
            }
        }
        if let Some((stage, ranged)) = binding(selector) {
            let bound: Vec<(u64, Option<u64>)> = match ranged {
                false => {
                    let index = call.args.get(1).and_then(Value::as_uint);
                    index.map(|index| (index, object(0))).into_iter().collect()
                }
                true => {
                    let ids = call.args.first().and_then(Value::as_list).unwrap_or(&[]);
                    let first = (call.args.get(1).and_then(Value::as_map))
                        .and_then(|range| range.iter().find(|(key, _)| &**key == "location"))
                        .and_then(|(_, at)| at.as_uint())
                        .unwrap_or(0);
                    (ids.iter().enumerate())
                        .map(|(i, id)| (first + i as u64, id.as_object()))
                        .collect()
                }
            };
            for (index, id) in bound {
                let view = id.filter(|id| self.surfaces.views.contains_key(id));
                match (stage, view) {
                    (Some(stage), Some(id)) => {
                        if let Some(notes) = self.surfaces.encoders.get_mut(&encoder) {
                            notes.slots.insert((stage, index), id);
                        }
                    }
                    (Some(stage), None) => {
                        if let Some(notes) = self.surfaces.encoders.get_mut(&encoder) {
                            notes.slots.remove(&(stage, index));
                        }
                    }
                    // A stage whose interface the runtime does not answer: by the usage.
                    (None, Some(id)) => {
                        if self.surfaces.writable.contains(&id) {
                            self.surfaces.counts.usage_writes += 1;
                            self.surfaces.note(encoder, id);
                        }
                    }
                    (None, None) => {}
                }
            }
            return;
        }
        match selector {
            "setRenderPipelineState:" | "setComputePipelineState:" => {
                if let Some(notes) = self.surfaces.encoders.get_mut(&encoder) {
                    notes.pipeline = object(0);
                }
            }
            "generateMipmapsForTexture:" => {
                if let Some(id) = object(0) {
                    self.surfaces.note(encoder, id);
                }
            }
            _ if selector.starts_with("useResource") => {
                let mut named = Vec::new();
                super::super::covers::named(&call.args, &mut named);
                for id in named {
                    if self.surfaces.writable.contains(&id) {
                        self.surfaces.counts.usage_writes += 1;
                        self.surfaces.note(encoder, id);
                    }
                }
            }
            _ if selector.contains("toTexture:") => {
                let mut named = Vec::new();
                super::super::covers::named(&call.args, &mut named);
                if let Some(&id) = named.last() {
                    self.surfaces.note(encoder, id);
                }
            }
            _ if selector.starts_with("draw") => {
                self.surface_slots_written(encoder, SlotStage::Vertex);
                self.surface_slots_written(encoder, SlotStage::Fragment);
            }
            _ if selector.starts_with("dispatch") => {
                self.surface_slots_written(encoder, SlotStage::Kernel);
            }
            _ => {}
        }
    }

    /// At a draw or dispatch of `encoder`: the surface textures at the slots of `stage` that its
    /// pipeline state's interface writes, or, where it answers none, whose usage lets a shader
    /// write them, noted written.
    fn surface_slots_written(&mut self, encoder: u64, stage: SlotStage) {
        let Some(notes) = self.surfaces.encoders.get(&encoder) else {
            return;
        };
        let bound: Vec<(u64, u64)> = (notes.slots.iter())
            .filter(|((at, _), _)| *at == stage)
            .map(|(&(_, index), &id)| (index, id))
            .collect();
        if bound.is_empty() {
            return;
        }
        let writes = notes
            .pipeline
            .and_then(|state| self.texture_writes(state, stage));
        for (index, id) in bound {
            let written = match &writes {
                Some(writes) => {
                    let declared = writes.contains(&index);
                    self.surfaces.counts.declared_writes += u64::from(declared);
                    declared
                }
                None => {
                    let writable = self.surfaces.writable.contains(&id);
                    self.surfaces.counts.usage_writes += u64::from(writable);
                    writable
                }
            };
            if written {
                self.surfaces.note(encoder, id);
            }
        }
    }

    /// After `endEncoding` of an encoder the play notes: each plane's write copied into its other
    /// shapes that a texture views or that are dirty, through the buffer, in blit encoders of the
    /// encoder's command buffer; the plane's other shapes go.
    pub(super) fn surface_ended(&mut self, call: &Call) -> Result<(), PlayError> {
        let Some(notes) = self.surfaces.encoders.remove(&call.receiver) else {
            return Ok(());
        };
        if notes.written.is_empty() {
            return Ok(());
        }
        // Each plane's write: the shape noted last.
        let mut planes: Vec<(Span, Shape, bool)> = Vec::new();
        for &(span, shape) in &notes.written {
            match planes.iter_mut().find(|(at, _, _)| *at == span) {
                Some(plane) => {
                    plane.2 |= plane.1 != shape;
                    plane.1 = shape;
                }
                None => planes.push((span, shape, false)),
            }
        }
        let surfaces = &mut self.surfaces;
        surfaces.counts.unordered += u64::from(planes.iter().any(|plane| plane.2));
        (surfaces.pending.entry(notes.command_buffer).or_default())
            .extend(planes.iter().map(|&(span, shape, _)| (span, shape)));
        let command_buffer = match self.objects.get(&notes.command_buffer) {
            Some(Object::CommandBuffer(command_buffer)) => command_buffer.clone(),
            _ => return Ok(()),
        };
        for (span, source, _) in planes {
            let Some(entries) = self.surfaces.ranges.get_mut(&span) else {
                continue;
            };
            let gone = |entry: &Entry| entry.shape != source && entry.live == 0 && !entry.dirty;
            let dropped = entries.iter().filter(|&entry| gone(entry)).count();
            entries.retain(|entry| !gone(entry));
            self.surfaces.counts.dropped += dropped as u64;
            let from = entries.iter().find(|entry| entry.shape == source);
            let Some(from) = from.map(|entry| entry.texture.clone()) else {
                continue;
            };
            let into: Vec<(Shape, rt::Texture)> = (entries.iter())
                .filter(|entry| entry.shape != source)
                .map(|entry| (entry.shape, entry.texture.clone()))
                .collect();
            let Some((bytes, rows)) = (into.iter().map(|(shape, _)| source.overlap(*shape)))
                .reduce(|(b, r), (bytes, rows)| (b.max(bytes), r.max(rows)))
            else {
                continue;
            };
            let Some(Object::Buffer(buffer, _)) = self.objects.get(&span.buffer) else {
                continue;
            };
            let size = [bytes.div_ceil(source.texel) as u32, rows, 1];
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(failed(call))?;
            (blit.copy_from_texture_to_buffer(&from, ORIGIN, size, buffer, span.layout(rows)))
                .map_err(failed(call))?;
            blit.end_encoding();
            let mut blit = command_buffer
                .blit_command_encoder()
                .map_err(failed(call))?;
            for (shape, texture) in into {
                let (size, straddled) = shape.covered(source.overlap(shape));
                self.surfaces.counts.straddled += u64::from(straddled);
                if size[0] == 0 || size[1] == 0 {
                    continue;
                }
                let layout = span.layout(size[1]);
                (blit.copy_from_buffer(buffer, layout, size, &texture, ORIGIN))
                    .map_err(failed(call))?;
                let bytes = u64::from(size[0]) * u64::from(size[1]) * shape.texel;
                self.surfaces.counts.overlap_copies += 1;
                self.surfaces.counts.overlap_bytes += bytes;
                self.surfaces.counts.bytes += bytes;
            }
            blit.end_encoding();
        }
        Ok(())
    }

    /// After a copy from the guest's RAM into bytes `spans` of buffer `buffer` (stream id `id`) in
    /// blit encoder `encoder`: each shape those bytes meet copied from the buffer, and clean.
    pub(super) fn surface_copied_in(
        &mut self,
        call: &Call,
        encoder: u64,
        (id, buffer): (u64, &rt::Buffer),
        spans: &[Range<u64>],
    ) -> Result<(), PlayError> {
        let shapes = self.surfaces.meeting(id, spans);
        for (span, shape, entry) in &shapes {
            let blit = self.blit(encoder);
            let layout = span.layout(shape.height);
            (blit.copy_from_buffer(buffer, layout, shape.size(), entry, ORIGIN))
                .map_err(failed(call))?;
            self.surfaces.counts.copies_in += 1;
            self.surfaces.counts.bytes += shape.bytes();
        }
        self.surfaces.clean(&shapes);
        Ok(())
    }

    /// Before a copy back to the guest's RAM of bytes `spans` of buffer `buffer` (stream id `id`)
    /// in blit encoder `encoder`: where the bytes meet a shape, every shape of the buffer that is
    /// dirty, or that the encoder's command buffer may have written before, copied into it.
    pub(super) fn surface_copying_out(
        &mut self,
        call: &Call,
        encoder: u64,
        (id, buffer): (u64, &rt::Buffer),
        spans: &[Range<u64>],
    ) -> Result<(), PlayError> {
        if self.surfaces.meeting(id, spans).is_empty() {
            return Ok(());
        }
        self.surfaces.counts.copies_out += 1;
        let own = self.owners.get(&encoder).copied();
        let pending = own.and_then(|own| self.surfaces.pending.get(&own));
        let written = |span: &Span, entry: &Entry| {
            entry.dirty || pending.is_some_and(|p| p.contains(&(*span, entry.shape)))
        };
        let shapes = self.surfaces.shapes_of(id, written);
        for (span, shape, entry) in &shapes {
            let blit = self.blit(encoder);
            let layout = span.layout(shape.height);
            (blit.copy_from_texture_to_buffer(entry, ORIGIN, shape.size(), buffer, layout))
                .map_err(failed(call))?;
            self.surfaces.counts.written_back += 1;
            self.surfaces.counts.written_back_bytes += shape.bytes();
            self.surfaces.counts.bytes += shape.bytes();
        }
        self.surfaces.clean(&shapes);
        Ok(())
    }

    /// A recorded CPU write of bytes `written` into receiver buffer `buffer` (stream id `id`),
    /// already in its mapping: the texels it reaches of every shape, uploaded from the mapping
    /// (`rt::Texture::replace_region`), ahead of every later commit.
    pub(super) fn surface_written(
        &mut self,
        (id, buffer): (u64, &rt::Buffer),
        written: Range<u64>,
    ) -> Result<(), PlayError> {
        let failed = |error: rt::Error| PlayError::Runtime {
            selector: "m2v.write".into(),
            error: error.to_string(),
        };
        let unmapped = || PlayError::Invalid(format!("#{id}'s texels lie past its mapping"));
        if self.surfaces.external.contains(&id) {
            return Ok(());
        }
        for (span, shape, entry) in self.surfaces.meeting(id, std::slice::from_ref(&written)) {
            let contents = buffer.contents().ok_or_else(unmapped)?;
            for (region, bytes) in span.reached(shape, &written) {
                if bytes.end > buffer.length() {
                    return Err(unmapped());
                }
                // SAFETY: `bytes` lies in the buffer's mapped bytes (checked above), which the
                // write just wrote.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        contents.as_ptr().add(bytes.start as usize),
                        (bytes.end - bytes.start) as usize,
                    )
                };
                let uploaded =
                    entry.replace_region(region, 0, 0, bytes.into(), span.bytes_per_row, 0);
                uploaded.map_err(failed)?;
                let [width, height, _] = region.size;
                self.surfaces.counts.bytes += u64::from(width) * u64::from(height) * shape.texel;
            }
            self.surfaces.counts.cpu_writes += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{binding, Shape, SlotStage, Span};

    /// The texels of a 4 by 3 texture of 4-byte texels, 32 bytes a row from byte 64, that writes
    /// reach: a run within one row (its last byte in a third texel), a run over three rows from the
    /// middle of the first, a run wholly in a row's padding, and one before the range.
    #[test]
    fn a_write_reaches_the_texels_under_its_bytes() {
        let span = Span {
            buffer: 1,
            plane: 0,
            offset: 64,
            bytes_per_row: 32,
        };
        let shape = Shape {
            width: 4,
            height: 3,
            texel: 4,
            storage: false,
        };
        let reached = |written: std::ops::Range<u64>| {
            (span.reached(shape, &written).into_iter())
                .map(|(region, bytes)| (region.origin, region.size, bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(reached(66..73), [([0, 0, 0], [3, 1, 1], 64..76)]);
        assert_eq!(
            reached(72..140),
            [
                ([2, 0, 0], [2, 1, 1], 72..80),
                ([0, 1, 0], [4, 1, 1], 96..112),
                ([0, 2, 0], [3, 1, 1], 128..140),
            ]
        );
        assert!(reached(84..96).is_empty());
        assert!(reached(0..64).is_empty());
    }

    /// Two shapes of one plane overlap from its first texel, as wide as the narrower one's rows and
    /// as high as the lower one; a write through a shape of 4-byte texels 5 wide covers 2 whole
    /// texels of 8 bytes and half of a third, which keeps its bytes.
    #[test]
    fn two_shapes_of_a_plane_overlap_from_its_first_texel() {
        let shape = |width, height, texel| Shape {
            width,
            height,
            texel,
            storage: false,
        };
        let (wide, narrow) = (shape(24, 42, 8), shape(18, 42, 8));
        assert_eq!(wide.overlap(narrow), (144, 42));
        assert_eq!(narrow.covered(wide.overlap(narrow)), ([18, 42, 1], false));
        let (four, eight) = (shape(5, 2, 4), shape(4, 3, 8));
        assert_eq!(four.overlap(eight), (20, 2));
        assert_eq!(eight.covered(four.overlap(eight)), ([2, 2, 1], true));
        assert_eq!(four.covered(eight.overlap(four)), ([5, 2, 1], false));
    }

    /// The stages a texture binding names: a kernel's, a vertex function's and a fragment
    /// function's slots, one or a range; a stage the runtime answers no interface for; and no
    /// binding.
    #[test]
    fn a_binding_names_its_stage() {
        assert_eq!(
            binding("setTexture:atIndex:"),
            Some((Some(SlotStage::Kernel), false))
        );
        assert_eq!(
            binding("setFragmentTextures:withRange:"),
            Some((Some(SlotStage::Fragment), true))
        );
        assert_eq!(
            binding("setVertexTexture:atIndex:"),
            Some((Some(SlotStage::Vertex), false))
        );
        assert_eq!(binding("setTileTexture:atIndex:"), Some((None, false)));
        assert_eq!(binding("setRenderPipelineState:"), None);
    }
}
