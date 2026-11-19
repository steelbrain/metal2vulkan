//! Textures over a surface's plane (`newTextureWithDescriptor:iosurface:plane:`), played as the
//! product makes them: one texture of the runtime's for each receiver range and shape, made for
//! views (`rt::Device::new_texture_for_views`) with the usage Metal gives a texture whose use it
//! does not know, and every recorded surface texture a view of it at the texture's own index. A
//! receiver range is the receiver buffer, which stands for the surface's pages, the plane, and the
//! plane's first byte and row pitch there; a shape is the texture's width, height and bytes a
//! texel, so a texture of another pixel format of the same bytes a texel is a view of the same
//! entry.
//!
//! The entries hold the texels; the receiver buffer holds the pages' bytes as the stream wrote
//! them, and takes the texels where the stream reads the pages:
//! - At its making an entry is filled from the receiver buffer at the range's offset and pitch, by
//!   a blit in a command buffer of the play's own committed then: after every earlier commit, with
//!   no wait.
//! - A new shape over a range that has an entry first downloads the range's newest entry into the
//!   receiver buffer at that entry's extent, then fills the new one, in the same command buffer:
//!   bytes map by byte offset, whatever the extents. An older shape's entry goes once no texture of
//!   it lives; the newest stays while its receiver buffer lives, so the same shape made again is a
//!   view and costs nothing. Older shapes are not kept current: a use the play sees of an older
//!   shape's texture after a newer shape over its range is counted, and one through an argument
//!   buffer is not seen.
//! - A copy from the guest's RAM into a range also copies the range into its newest entry, and a
//!   copy back to the guest's RAM first copies the newest entry into the range, both in the copy's
//!   own blit encoder, where the GPU's order holds them: the runtime's upload (`replace_region`)
//!   would land ahead of the encoders already in the command buffer, and its read-back
//!   (`get_bytes`) would wait where the old runtime's play does not.
//! - A recorded CPU write into a range after its entry's making reaches the texels it covers of the
//!   newest entry through the runtime's upload, which orders it as Metal orders a CPU write.
//! - `m2v.externalWriteDone` on a receiver buffer, another API's write landed in it (a decoded
//!   picture), refills every entry of the buffer from it, by a blit in a command buffer of the
//!   play's own committed then. From the first such record, the recorded CPU writes into that
//!   buffer, which stand for the other API's, reach its entries only this way.
//! - Any other command that names a receiver buffer as a buffer is refused: the buffer does not
//!   hold the texels.
//! - A texture of a format that reads across a surface's planes (a two-plane YCbCr or XR10 format
//!   over a surface of two planes) is skipped and counted.

use super::{failed, whole_slices, NewRuntime, Object};
use crate::replay::args::{refuse, Fields, SURFACE_FACT_KEYS, SURFACE_PLANE_KEYS};
use crate::replay::player::{Made, PlayError};
use crate::replay::stream::Call;
use metal2vulkan_rt as rt;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// How many older-shape uses the report places.
const NAMED: usize = 8;

/// The commands that name a receiver buffer and are played through its entries.
const COPIES: [&str; 3] = [
    "m2v.copyFromGuestRam",
    "m2v.copyFromGuestRamAt",
    "m2v.copyToGuestRam",
];

/// The first texel of an entry, where every copy of one starts.
const ORIGIN: rt::TexturePlace = rt::TexturePlace {
    slice: 0,
    level: 0,
    origin: [0; 3],
};

/// A receiver range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Span {
    buffer: u64,
    plane: u64,
    offset: u64,
    bytes_per_row: u64,
}

/// An entry's shape: its width, height and bytes a texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    width: u32,
    height: u32,
    texel: u64,
}

impl Span {
    /// The bytes of the buffer the texels of `shape` lie in.
    fn bytes(&self, shape: Shape) -> Range<u64> {
        let rows = u64::from(shape.height.max(1)) - 1;
        let end = self.offset + rows * self.bytes_per_row + u64::from(shape.width) * shape.texel;
        self.offset..end
    }

    /// The texels of `shape` laid out in the buffer, as a blit copy names them.
    fn layout(&self, shape: Shape) -> rt::BufferLayout {
        rt::BufferLayout {
            offset: self.offset,
            bytes_per_row: self.bytes_per_row,
            bytes_per_image: self.bytes_per_row * u64::from(shape.height),
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
        // From the range's first byte.
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
}

/// A texture of the runtime's under a range's surface textures.
struct Entry {
    shape: Shape,
    texture: rt::Texture,
    /// How many surface textures of it live.
    live: usize,
}

/// The play's surface textures and their entries.
#[derive(Default)]
pub(super) struct Surfaces {
    /// Each receiver range's entries, the newest last.
    ranges: HashMap<Span, Vec<Entry>>,
    /// Each live receiver buffer's ranges.
    receivers: HashMap<u64, Vec<Span>>,
    /// Each live surface texture's range and shape.
    views: HashMap<u64, (Span, Shape)>,
    /// The receiver buffers another API writes (`m2v.externalWriteDone`).
    external: HashSet<u64>,
    /// The queue of the play's own that the fills at a making and the refills go on.
    queue: Option<rt::CommandQueue>,
    counts: Counts,
}

#[derive(Default)]
struct Counts {
    /// Surface textures made as views.
    views: u64,
    /// Entries made, and the fills at their making.
    entries: u64,
    fills: u64,
    /// Fills at a new shape over a range that had an entry: each after a download.
    shape_changes: u64,
    /// Recorded CPU writes into a range after its entry's making.
    cpu_writes: u64,
    /// `m2v.externalWriteDone` records, and the entries they refilled.
    external_writes: u64,
    refills: u64,
    /// Copies from the guest's RAM into an entry, and from an entry back to it.
    copies_in: u64,
    copies_out: u64,
    /// The bytes of the texels those fills, downloads and copies moved.
    bytes: u64,
    /// Textures of a format that reads across a surface's planes, skipped.
    across_planes: u64,
    /// Uses of an older shape's texture after a newer shape over its range, and the frame and
    /// call of the first few.
    older_uses: u64,
    older_at: Vec<(u64, u64)>,
}

impl Surfaces {
    pub(super) fn is_empty(&self) -> bool {
        self.receivers.is_empty() && self.views.is_empty()
    }

    pub(super) fn is_receiver(&self, buffer: u64) -> bool {
        self.receivers.contains_key(&buffer)
    }

    /// The newest entry of each range of buffer `buffer` whose texels' bytes meet one of `spans`.
    fn meeting(&self, buffer: u64, spans: &[Range<u64>]) -> Vec<(Span, Shape, rt::Texture)> {
        let Some(ranges) = self.receivers.get(&buffer) else {
            return Vec::new();
        };
        let meets = |span: &Span, shape| {
            let bytes = span.bytes(shape);
            (spans.iter()).any(|s| s.start < bytes.end && bytes.start < s.end)
        };
        (ranges.iter())
            .filter_map(|span| {
                let newest = self.ranges.get(span)?.last()?;
                let met = meets(span, newest.shape);
                met.then(|| (*span, newest.shape, newest.texture.clone()))
            })
            .collect()
    }

    /// `object` released: a receiver buffer's entries go with it, and an older shape's entry with
    /// its last texture.
    pub(super) fn released(&mut self, object: u64) {
        if let Some(ranges) = self.receivers.remove(&object) {
            for span in ranges {
                self.ranges.remove(&span);
            }
        }
        self.external.remove(&object);
        let Some((span, shape)) = self.views.remove(&object) else {
            return;
        };
        let Some(entries) = self.ranges.get_mut(&span) else {
            return;
        };
        let newest = entries.len() - 1;
        if let Some(at) = entries.iter().position(|entry| entry.shape == shape) {
            entries[at].live = entries[at].live.saturating_sub(1);
            if entries[at].live == 0 && at != newest {
                entries.remove(at);
            }
        }
    }

    /// Counts the uses of `named` that are surface textures of an older shape than their range's
    /// newest entry, at `at` (frame and call).
    fn used(&mut self, named: &[u64], at: (u64, u64)) {
        for object in named {
            let Some((span, shape)) = self.views.get(object) else {
                continue;
            };
            let newest = self.ranges.get(span).and_then(|entries| entries.last());
            if newest.is_some_and(|newest| newest.shape != *shape) {
                self.counts.older_uses += 1;
                if self.counts.older_at.len() < NAMED {
                    self.counts.older_at.push(at);
                }
            }
        }
    }

    /// The textures skipped: those of a format that reads across a surface's planes.
    pub(super) fn across_planes(&self) -> u64 {
        self.counts.across_planes
    }

    /// The executed counters.
    pub(super) fn counted(&self) -> [(&'static str, u64); 12] {
        let c = &self.counts;
        [
            ("surface_external_writes", c.external_writes),
            ("surface_refills", c.refills),
            ("surface_textures", c.views),
            ("surface_entries", c.entries),
            ("surface_fills", c.fills),
            ("surface_shape_changes", c.shape_changes),
            ("surface_cpu_writes", c.cpu_writes),
            ("surface_copies_in", c.copies_in),
            ("surface_copies_out", c.copies_out),
            ("surface_bytes", c.bytes),
            ("skipped_surfaces_across_planes", c.across_planes),
            ("surface_older_uses", c.older_uses),
        ]
    }

    /// The report's `surfaces` line.
    pub(super) fn line(&self) -> String {
        let c = &self.counts;
        let at: Vec<String> = (c.older_at.iter())
            .map(|(frame, call)| format!("frame {frame} call {call}"))
            .collect();
        format!(
            "surfaces {} textures as views of {} entries: {} fills at a making, {} of them at a new \
             shape after a download; {} CPU writes, {} external writes refilling {} entries, {} \
             copies from the guest's RAM into an entry and {} back, {} bytes; {} across planes \
             skipped; {} older-shape surface uses (those through argument buffers unseen){}{}",
            c.views,
            c.entries,
            c.fills,
            c.shape_changes,
            c.cpu_writes,
            c.external_writes,
            c.refills,
            c.copies_in,
            c.copies_out,
            c.bytes,
            c.across_planes,
            c.older_uses,
            if at.is_empty() { "" } else { ", the first at " },
            at.join(", "),
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

impl NewRuntime {
    /// `newTextureWithDescriptor:iosurface:plane:` on receiver buffer `buffer`: a view of the
    /// entry of its range and shape, at the texture's own index.
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
        let shape = Shape {
            width: d.width,
            height: d.height,
            texel,
        };
        let entry = self.surface_entry(call, buffer, span, shape, &d)?;
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
            self.surfaces.views.insert(call.result, (span, shape));
            let entries = self.surfaces.ranges.get_mut(&span).into_iter().flatten();
            if let Some(entry) = entries.filter(|entry| entry.shape == shape).last() {
                entry.live += 1;
            }
        }
        Ok(made)
    }

    /// The entry under a surface texture of `shape` over `span`: the range's newest where it has
    /// that shape, else made newest and filled (`Surfaces`).
    fn surface_entry(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
        span: Span,
        shape: Shape,
        descriptor: &rt::TextureDescriptor,
    ) -> Result<rt::Texture, PlayError> {
        let surfaces = &mut self.surfaces;
        let entries = surfaces.ranges.entry(span).or_default();
        let newest = entries
            .last()
            .map(|entry| (entry.texture.clone(), entry.shape));
        if let Some((texture, _)) = newest.as_ref().filter(|(_, was)| *was == shape) {
            return Ok(texture.clone());
        }
        let entry = match entries.iter().position(|entry| entry.shape == shape) {
            Some(at) => entries.remove(at),
            None => {
                let mut d = *descriptor;
                d.usage = rt::TextureUsage::SHADER_READ
                    | rt::TextureUsage::SHADER_WRITE
                    | rt::TextureUsage::RENDER_TARGET
                    | rt::TextureUsage::PIXEL_FORMAT_VIEW;
                d.swizzle = rt::TextureSwizzleChannels::DEFAULT;
                let texture = self.device.new_texture_for_views(&d);
                surfaces.counts.entries += 1;
                Entry {
                    shape,
                    texture: texture.map_err(failed(call))?,
                    live: 0,
                }
            }
        };
        let texture = entry.texture.clone();
        entries.push(entry);
        // An older shape's entry goes once no texture of it lives.
        let newest_at = entries.len() - 1;
        let mut at = 0;
        entries.retain(|entry| {
            let keep = at == newest_at || entry.live > 0;
            at += 1;
            keep
        });
        let ranges = surfaces.receivers.entry(span.buffer).or_default();
        if !ranges.contains(&span) {
            ranges.push(span);
        }
        surfaces.counts.fills += 1;
        surfaces.counts.bytes += shape.bytes();
        if let Some((_, was)) = &newest {
            surfaces.counts.shape_changes += 1;
            surfaces.counts.bytes += was.bytes();
        }
        let queue = self.surface_queue(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        if let Some((older, was)) = newest {
            let layout = span.layout(was);
            (blit.copy_from_texture_to_buffer(&older, ORIGIN, was.size(), buffer, layout))
                .map_err(failed(call))?;
        }
        (blit.copy_from_buffer(buffer, span.layout(shape), shape.size(), &texture, ORIGIN))
            .map_err(failed(call))?;
        blit.end_encoding();
        self.commit(&queue, command_buffer, 0)
            .map_err(failed(call))?;
        Ok(texture)
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
    /// entry of its ranges takes its bytes again, by one blit encoder in a command buffer of the
    /// play's own committed now. The one place the record's position decides.
    pub(super) fn external_write_done(
        &mut self,
        call: &Call,
        buffer: &rt::Buffer,
    ) -> Result<Option<(Object, Made)>, PlayError> {
        let surfaces = &mut self.surfaces;
        surfaces.counts.external_writes += 1;
        surfaces.external.insert(call.receiver);
        let ranges = surfaces.receivers.get(&call.receiver).into_iter().flatten();
        let entries: Vec<(Span, Shape, rt::Texture)> = (ranges.flat_map(|span| {
            let entries = surfaces.ranges.get(span).into_iter().flatten();
            entries.map(|entry| (*span, entry.shape, entry.texture.clone()))
        }))
        .collect();
        if entries.is_empty() {
            return Ok(None);
        }
        let queue = self.surface_queue(call)?;
        let command_buffer = queue.command_buffer().map_err(failed(call))?;
        let mut blit = command_buffer
            .blit_command_encoder()
            .map_err(failed(call))?;
        for (span, shape, entry) in entries {
            (blit.copy_from_buffer(buffer, span.layout(shape), shape.size(), &entry, ORIGIN))
                .map_err(failed(call))?;
            self.surfaces.counts.refills += 1;
            self.surfaces.counts.bytes += shape.bytes();
        }
        blit.end_encoding();
        self.commit(&queue, command_buffer, 0)
            .map_err(failed(call))?;
        Ok(None)
    }

    /// Before `call` plays: refused where it names a receiver buffer as a buffer, other than a
    /// copy between the guest's RAM and it; each surface texture of an older shape it names,
    /// counted.
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
        self.surfaces.used(named, self.at);
        Ok(())
    }

    /// After a copy from the guest's RAM into bytes `spans` of buffer `buffer` (stream id `id`) in
    /// blit encoder `encoder`: each range those bytes meet copied into its newest entry.
    pub(super) fn surface_copied_in(
        &mut self,
        call: &Call,
        encoder: u64,
        (id, buffer): (u64, &rt::Buffer),
        spans: &[Range<u64>],
    ) -> Result<(), PlayError> {
        for (span, shape, entry) in self.surfaces.meeting(id, spans) {
            let blit = self.blit(encoder);
            (blit.copy_from_buffer(buffer, span.layout(shape), shape.size(), &entry, ORIGIN))
                .map_err(failed(call))?;
            self.surfaces.counts.copies_in += 1;
            self.surfaces.counts.bytes += shape.bytes();
        }
        Ok(())
    }

    /// Before a copy back to the guest's RAM of bytes `spans` of buffer `buffer` (stream id `id`)
    /// in blit encoder `encoder`: each range those bytes meet takes its newest entry's texels.
    pub(super) fn surface_copying_out(
        &mut self,
        call: &Call,
        encoder: u64,
        (id, buffer): (u64, &rt::Buffer),
        spans: &[Range<u64>],
    ) -> Result<(), PlayError> {
        for (span, shape, entry) in self.surfaces.meeting(id, spans) {
            let blit = self.blit(encoder);
            let layout = span.layout(shape);
            (blit.copy_from_texture_to_buffer(&entry, ORIGIN, shape.size(), buffer, layout))
                .map_err(failed(call))?;
            self.surfaces.counts.copies_out += 1;
            self.surfaces.counts.bytes += shape.bytes();
        }
        Ok(())
    }

    /// A recorded CPU write of bytes `written` into receiver buffer `buffer` (stream id `id`),
    /// already in its mapping: the texels it reaches of each range's newest entry, uploaded from
    /// the mapping (`rt::Texture::replace_region`), ahead of every later commit.
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
                let uploaded = entry.replace_region(region, 0, 0, bytes, span.bytes_per_row, 0);
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
    use super::{Shape, Span};

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
}
