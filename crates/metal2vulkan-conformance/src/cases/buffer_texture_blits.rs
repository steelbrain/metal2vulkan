//! Blits that name a texture laid over a buffer.
//!
//! The family's question is where a copy puts its bytes when one side of it is a texture whose
//! memory is a buffer's. Such a texture's rows lie `bytesPerRow` apart, which here is wider than a
//! row of texels, so every copy has two answers to give: the texels it moved, and the bytes
//! between the rows that it must not have touched. The buffer is read back whole after each copy,
//! which is the only place a byte written between the rows would show.
//!
//! Each row states its texture's row in bytes rather than deriving it from the pixel format.
//! `MTLTexture` answers no bytes-per-texel, and a format table of ours would be a second spelling
//! of Metal's to keep in step; a wrong row width here is what the comparison would quietly stop
//! covering.
//!
//! The bytes are chosen so that a byte in the wrong place names where it came from. The buffer
//! holds `7i + 3` at byte `i`, and a texel holds its own coordinates and its set, so a texel the
//! copy put one row or one texel off is read back as the texel it actually is rather than as a
//! number that could have come from anywhere.
//!
//! **Write the destination before the copy.** The ordinary texture a copy reads from or writes to
//! is always written first, with a set of its own. An unwritten destination's untouched texels
//! compare against whatever the texture happened to hold, which in practice means zeros, and a
//! copy that wrote zeros where it should have written nothing passes such a comparison. The lane
//! this came from left its destination unwritten for exactly that reason and could not have caught
//! it. The fix costs one `replaceRegion:` per program: write a distinctive set first and the same
//! comparison becomes a real one, because an untouched texel is then a texel this program put
//! there. `out-of-buffer-part` is where it shows — columns 2 and 3 still hold set 2 after the
//! copy, which is only a check because set 2 was put there first.
//!
//! **What Metal does where a copy's source and destination are the same bytes**, measured on an
//! Apple M3 Max (macOS 27.0.1, build 26A434) rather than modelled: every destination texel holds
//! the texel the source held **before the copy**, as a copy through a temporary does. In
//! `onto-itself-overlapping`, a 3x3 region from (0, 0) to (1, 1), the texel at (1, 2) reads
//! `t01` — the source's original (0, 1) — and not the `t00` that a forward row-by-row copy would
//! have propagated out of the row it had just written. Both `within-buffer-a-row-on` rows agree,
//! in both copy directions: destination row N holds source row N with no propagation. The next
//! author to implement an overlapping copy will reach for the row-by-row form, which is wrong
//! here; this is the sentence that should stop them.
//!
//! Nothing here faults a GPU, hangs one or trips a watchdog: each program is a handful of blits
//! over shared memory of a kilobyte, committed once and read back.
//!
//! What the lane this came from also covers and these cases do not yet: buffers made by
//! `newBufferWithBytesNoCopy:` (the vocabulary has no such resource, and whether a device imports
//! host memory at all is not Metal's answer to give), the whole-texture
//! `copyFromTexture:toTexture:` selector (a copy that names no region is a different call and
//! wants a step of its own), linear array slices and views of them, texture buffers, and the
//! IOSurface planes.

use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Rect, Resource, Step, Storage as WorkStorage, TextureKind, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The case names, as the report and the recordings spell them.
pub const TO_AND_FROM: &str = "buffer-texture-blits-to-and-from";
pub const OVERLAPPING: &str = "buffer-texture-blits-overlapping";

/// The buffer's bytes. A kilobyte: enough for the texture at [`OFFSET`], a second one at
/// [`ELSEWHERE`], and bytes on either side of both that no copy should reach.
const LEN: u64 = 1024;
/// Where the texture's first row starts in the buffer.
const OFFSET: u64 = 256;
/// Where a second texture over the same buffer starts, far enough from the first that their rows
/// cannot overlap.
const ELSEWHERE: u64 = 640;
/// `bytesPerRow`: the rows lie this far apart.
const PITCH: u64 = 32;
/// The texture is this many texels on a side.
const SIDE: u64 = 4;
/// One row of the texture in bytes, **stated** rather than derived from the format: four
/// `RGBA8Unorm` texels. Half of [`PITCH`], so every row is followed by as many bytes again that no
/// copy may touch.
const ROW_BYTES: u64 = 16;
/// The raw `MTLPixelFormat` every texture here is of.
const FORMAT: u32 = PixelFormat::RGBA8Unorm.as_raw();
/// The usage every texture here is made with, stated whole: a texture a blit reads and writes,
/// which a shader may also read and write.
const USAGE: u32 = TextureUsage::SHADER_READ
    .union(TextureUsage::SHADER_WRITE)
    .as_raw();

/// The buffer's bytes before anything runs: byte `i` holds `7i + 3`.
///
/// It is not the stripe pattern the other buffer cases use, because this one is written by the
/// host into a buffer it then lays textures over, and a byte of it that turns up inside a texel is
/// easier to place when the two patterns differ.
fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(7).wrapping_add(3)) as u8)
        .collect()
}

/// Texel `(x, y)` of set `s`: four bytes that name the texel and the set it came from, so a texel
/// the copy put one row or one texel off is read back as itself.
fn texel(x: u64, y: u64, s: u64) -> [u8; 4] {
    [
        0x10 | x as u8,
        0x20 | y as u8,
        (0x40 + 16 * s + 4 * y + x) as u8,
        (0xC0 + 4 * y + x) as u8,
    ]
}

/// Set `s`'s texels of a `SIDE` by `SIDE` texture, packed into rows of [`ROW_BYTES`].
fn texels(s: u64) -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).flat_map(move |x| texel(x, y, s)))
        .collect()
}

/// The buffer's bytes with set `s`'s texels written at `offset`, its rows [`PITCH`] apart, as the
/// host writes them before any command runs.
fn with_texels(mut bytes: Vec<u8>, offset: u64, s: u64) -> Vec<u8> {
    for y in 0..SIDE {
        let row = (0..SIDE).flat_map(|x| texel(x, y, s));
        let at = (offset + y * PITCH) as usize;
        for (i, byte) in row.enumerate() {
            bytes[at + i] = byte;
        }
    }
    bytes
}

/// Which copy a row runs, and so which Metal call it is.
#[derive(Clone, Copy, Debug)]
enum Blit {
    /// An ordinary texture's region into the texture over the buffer.
    IntoBuffer,
    /// The texture over the buffer's region into an ordinary texture.
    OutOfBuffer,
    /// Between two textures over the **same** buffer, the second one at [`ELSEWHERE`] or at an
    /// offset whose rows overlap the first's. `reverse` copies the second into the first.
    WithinBuffer { other: u64, reverse: bool },
    /// The texture over the buffer into itself.
    OntoItself,
}

/// One copy the case runs, and the program it stands for.
struct Row {
    /// How the row is named, which is the site its answers sit under.
    key: &'static str,
    blit: Blit,
    /// The copy's source origin, its size in texels, and its destination origin.
    from: [u64; 2],
    size: [u64; 2],
    to: [u64; 2],
}

impl Row {
    /// Whether the row's program needs the buffer's texels written before it runs: every row whose
    /// copy reads out of the buffer does.
    fn reads_the_buffer(&self) -> bool {
        !matches!(self.blit, Blit::IntoBuffer)
    }

    /// Whether the row makes an ordinary texture beside the one over the buffer.
    fn has_an_ordinary_texture(&self) -> bool {
        matches!(self.blit, Blit::IntoBuffer | Blit::OutOfBuffer)
    }

    /// Whether the row's copy reads and writes **the same bytes of the buffer**, which is what
    /// `overlapping-texel-copy` names: Vulkan's copy forbids it, so our side takes the rows through
    /// a scratch buffer.
    ///
    /// This is the shape that decides the row's `overlapping-texel-copy` declaration, and nothing
    /// else. It is read off the row's own numbers rather than its key, so a row added to a table,
    /// or an offset moved, changes what the case declares by itself.
    ///
    /// A copy with an ordinary texture on one side shares nothing: that texture is not over the
    /// buffer. Between two textures over the one buffer, the bytes are shared where the textures'
    /// own byte ranges meet. Within one texture, a row of it holds its texels and nothing else, so
    /// the bytes are shared only where the source and destination rectangles share a row **and** a
    /// column: `onto-itself-disjoint` reads rows 0..2 and writes rows 2..4 and shares none.
    fn copies_shared_bytes(&self) -> bool {
        // The half-open byte range a texture at `at` covers: its last row's texels end it, not its
        // pitch, so the bytes past the last row belong to neither texture.
        let span = |at: u64| at..at + (SIDE - 1) * PITCH + ROW_BYTES;
        match self.blit {
            Blit::IntoBuffer | Blit::OutOfBuffer => false,
            Blit::WithinBuffer { other, .. } => {
                let (one, two) = (span(OFFSET), span(other));
                one.start < two.end && two.start < one.end
            }
            Blit::OntoItself => {
                let meet = |from: u64, to: u64, size: u64| from < to + size && to < from + size;
                meet(self.from[1], self.to[1], self.size[1])
                    && meet(self.from[0], self.to[0], self.size[0])
            }
        }
    }

    /// The buffer's bytes before the row's program runs.
    fn buffer_bytes(&self) -> Vec<u8> {
        let bytes = pattern(LEN as usize);
        if self.reads_the_buffer() {
            // Set 0 at the texture's own offset. A row that copies between two textures over the
            // one buffer reads this one and writes the other, so only the source is written: the
            // destination's bytes are the pattern, and a byte the copy left there is visible.
            with_texels(bytes, OFFSET, 0)
        } else {
            bytes
        }
    }

    /// The row's program: the buffer, the texture or textures over it, the ordinary texture where
    /// the row has one, the copy, and the reads.
    fn work(&self) -> Work {
        let mut work = Work::new();
        let buffer = work.buffer(self.buffer_bytes());
        let over = |work: &mut Work, offset: u64| {
            work.add(Resource::TextureOverBuffer {
                buffer,
                spec: spec(WorkStorage::Shared),
                offset,
                bytes_per_row: PITCH,
            })
        };
        let texture = over(&mut work, OFFSET);
        // The ordinary texture is always written before the copy, with a set of its own, so a
        // texel the copy did not move is a texel this program put there rather than whatever the
        // texture happened to hold.
        let ordinary = self.has_an_ordinary_texture().then(|| {
            let ordinary = work.add(Resource::Texture {
                spec: spec(WorkStorage::Shared),
            });
            work.step(Step::ReplaceRegion {
                texture: ordinary,
                level: 0,
                slice: 0,
                region: Rect::new(0, 0, SIDE, SIDE),
                bytes: texels(2),
                rows: SIDE,
                bytes_per_row: ROW_BYTES,
                bytes_per_image: 0,
            });
            ordinary
        });
        let (from, to) = match self.blit {
            Blit::IntoBuffer => (ordinary.expect("an ordinary texture"), texture),
            Blit::OutOfBuffer => (texture, ordinary.expect("an ordinary texture")),
            Blit::WithinBuffer { other, reverse } => {
                let second = over(&mut work, other);
                if reverse {
                    (second, texture)
                } else {
                    (texture, second)
                }
            }
            Blit::OntoItself => (texture, texture),
        };
        work.step(Step::CopyTextureToTexture {
            from,
            from_slice: 0,
            from_level: 0,
            from_origin: [self.from[0], self.from[1], 0],
            size: [self.size[0], self.size[1], 1],
            to,
            to_slice: 0,
            to_level: 0,
            to_origin: [self.to[0], self.to[1], 0],
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: LEN as usize,
        });
        if let Some(ordinary) = ordinary {
            work.step(Step::GetBytes {
                texture: ordinary,
                level: 0,
                slice: 0,
                region: Rect::new(0, 0, SIDE, SIDE),
                bytes_per_row: ROW_BYTES,
                bytes_per_image: 0,
                length: (SIDE * ROW_BYTES) as usize,
                // This read is at the region's own pitch, so it has no padding of its own to ask
                // about; the padding question is the buffer's, and the buffer read asks it.
                before: 0,
            });
        }
        work
    }

    /// The words the case's input fingerprint covers for this row: the call, its geometry, and the
    /// numbers that place the texture in the buffer.
    fn words(&self) -> Vec<u8> {
        let mut words = self.key.as_bytes().to_vec();
        words.extend(b"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:");
        words.extend(match self.blit {
            Blit::IntoBuffer => b"into-buffer".to_vec(),
            Blit::OutOfBuffer => b"out-of-buffer".to_vec(),
            Blit::WithinBuffer { other, reverse } => {
                let mut bytes = b"within-buffer".to_vec();
                bytes.extend(other.to_le_bytes());
                bytes.push(u8::from(reverse));
                bytes
            }
            Blit::OntoItself => b"onto-itself".to_vec(),
        });
        for axis in self.from.iter().chain(&self.size).chain(&self.to) {
            words.extend(axis.to_le_bytes());
        }
        words.extend(OFFSET.to_le_bytes());
        words.extend(PITCH.to_le_bytes());
        words.extend(ROW_BYTES.to_le_bytes());
        words.extend(FORMAT.to_le_bytes());
        words
    }
}

/// Every texture here, of `storage`: a `SIDE` by `SIDE` plain-layout 2D texture of [`FORMAT`] and
/// [`USAGE`].
///
/// The layout is stated plain rather than left to the GPU, because a texture over a buffer holds
/// the buffer's own bytes and a case that read one of GPU-optimized contents would be asking a
/// different question of each side.
fn spec(storage: WorkStorage) -> TextureSpec {
    TextureSpec {
        kind: TextureKind::Type2D,
        storage,
        gpu_optimized: false,
        usage: Some(USAGE),
        ..TextureSpec::texture_2d(FORMAT, SIDE, SIDE)
    }
}

/// The whole buffer as a site's layout, divided into rows of [`PITCH`].
///
/// Every byte is an element, so the bytes between the texture's rows compare like any other: a
/// copy that wrote between the rows is visible here and nowhere else. The texture's first byte is
/// the start of one of these rows, so a difference names the row and the byte inside it.
fn buffer_layout() -> ByteLayout {
    let pitch = u32::try_from(PITCH).expect("a pitch in a u32");
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: LEN,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: pitch,
        row_pitch: pitch,
        rows: u32::try_from(LEN / PITCH).expect("the rows in a u32"),
        image_pitch: u32::try_from(LEN).expect("the length in a u32"),
        images: 1,
        // A row's elements are its whole pitch here: this view divides the buffer up, it does not
        // lay a texture over it.
        gaps: Gaps::Ignored,
    }
}

/// An ordinary texture's texels as a site's layout: its rows, at its own pitch.
fn ordinary_layout() -> ByteLayout {
    let row = u32::try_from(ROW_BYTES).expect("a row in a u32");
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: SIDE * ROW_BYTES,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row,
        row_pitch: row,
        rows: u32::try_from(SIDE).expect("the rows in a u32"),
        image_pitch: u32::try_from(SIDE * ROW_BYTES).expect("the bytes in a u32"),
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The rows of the whole-and-part case: an ordinary texture into the buffer's texture and back,
/// each of them whole and then in part.
///
/// The part rows move their region to a different origin, so a copy that ignored one of the two
/// origins lands somewhere this case reads.
const TO_AND_FROM_ROWS: [Row; 4] = [
    Row {
        key: "into-buffer-whole",
        blit: Blit::IntoBuffer,
        from: [0, 0],
        size: [SIDE, SIDE],
        to: [0, 0],
    },
    Row {
        key: "into-buffer-part",
        blit: Blit::IntoBuffer,
        from: [1, 1],
        size: [2, 2],
        to: [2, 1],
    },
    Row {
        key: "out-of-buffer-whole",
        blit: Blit::OutOfBuffer,
        from: [0, 0],
        size: [SIDE, SIDE],
        to: [0, 0],
    },
    Row {
        key: "out-of-buffer-part",
        blit: Blit::OutOfBuffer,
        from: [1, 0],
        size: [2, 3],
        to: [0, 1],
    },
];

/// The rows of the overlapping case: copies between textures over one buffer, and of the texture
/// onto itself, where the source and the destination bytes are the same bytes or lie a row apart.
///
/// These are the rows a model of the copy would not settle, and they are what measured the rule
/// this module's docs state: Metal reads every texel before it writes any, so a destination texel
/// holds what the source held before the copy. They stay here to hold that rule rather than to
/// confirm it — a release that moved to a row-by-row copy would pass every other row of this
/// family and fail these.
const OVERLAPPING_ROWS: [Row; 5] = [
    Row {
        key: "within-buffer-disjoint",
        blit: Blit::WithinBuffer {
            other: ELSEWHERE,
            reverse: false,
        },
        from: [0, 0],
        size: [SIDE, SIDE],
        to: [0, 0],
    },
    Row {
        // The second texture starts one row on, so its rows are the first's rows shifted by a
        // pitch: every row the copy writes is a row it has already read or is about to read.
        key: "within-buffer-a-row-on",
        blit: Blit::WithinBuffer {
            other: OFFSET + PITCH,
            reverse: false,
        },
        from: [0, 0],
        size: [SIDE, SIDE],
        to: [0, 0],
    },
    Row {
        key: "within-buffer-a-row-on-reversed",
        blit: Blit::WithinBuffer {
            other: OFFSET + PITCH,
            reverse: true,
        },
        from: [0, 0],
        size: [SIDE, SIDE],
        to: [0, 0],
    },
    Row {
        key: "onto-itself-disjoint",
        blit: Blit::OntoItself,
        from: [0, 0],
        size: [2, 2],
        to: [2, 2],
    },
    Row {
        key: "onto-itself-overlapping",
        blit: Blit::OntoItself,
        from: [0, 0],
        size: [3, 3],
        to: [1, 1],
    },
];

/// The cases of this module, as the case set holds them.
pub fn all() -> Vec<Box<dyn Case>> {
    table()
        .into_iter()
        .map(|case| Box::new(case) as _)
        .collect()
}

/// The cases of this module as themselves, which the tests read their tables out of.
fn table() -> Vec<Blits> {
    vec![
        Blits {
            name: TO_AND_FROM,
            rows: &TO_AND_FROM_ROWS,
        },
        Blits {
            name: OVERLAPPING,
            rows: &OVERLAPPING_ROWS,
        },
    ]
}

/// A case over a table of blits, each one its own program and its own sites.
pub struct Blits {
    name: &'static str,
    rows: &'static [Row],
}

impl Case for Blits {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // Every byte of every one of these programs compares exactly, Metal's against ours. What
        // differs is the announcing: a texture whose rows lie further apart than a row is wide is
        // kept as a shadow image, and a copy whose source and destination texels are the same
        // bytes is done in a way the device announces. Metal announces neither, because neither
        // is a thing Metal has to do, so the notices site differs and the notice itself is what
        // justifies it.
        //
        // One declaration per code, and a code is declared only where a row of this case can
        // announce it. A declaration fires on its own code, so a code a case announces and declares
        // nowhere is a divergence for an undeclared notice, and a code it declares but no row can
        // announce is an allowance that fires on no venue — the noise a real difference hides in.
        // Both halves matter, which is why the second code is read off the rows rather than named
        // for the whole family.
        //
        // `shadowed-texture` is every row's: each row's pitch is wider than its row, so the texture
        // over the buffer is kept as a shadow image.
        //
        // `overlapping-texel-copy` is only the rows whose copy reads and writes the same bytes of
        // the buffer ([`Row::copies_shared_bytes`]), which Vulkan's copy forbids and our side takes
        // through a scratch buffer. The `to-and-from` table has no such row — both its sides are an
        // ordinary texture and a texture over the buffer, which share nothing — so that case
        // declares only the first code. Measured on Apple M3 Max: `buffer-texture-blits-overlapping`
        // announced `overlapping-texel-copy` and declared it nowhere, while `to-and-from` compared
        // justified in the same run.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for both rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::ShadowedTexture.code(),
            ),
        ];
        if self.rows.iter().any(Row::copies_shared_bytes) {
            named.push(crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::OverlappingTexelCopy.code(),
            ));
        }
        named
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every row and the order they run in, keyed by the call each one makes and its numbers: a
        // table that gains, loses or re-points a row is a change to the case's inputs rather than
        // a quiet pass against another table's answers.
        let mut words = Vec::new();
        for row in self.rows {
            words.extend(row.words());
        }
        Ok(vec![
            Input::bytes("buffer-bytes", &pattern(LEN as usize)),
            Input::bytes("rows", &words),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for row in self.rows {
            // Each row is a program of its own, so one row the side refuses is a refusal at that
            // row's site and the other rows stay readable.
            let done = call(sites, row.key, side.work(&row.work()), |_| Value::Absent)?;
            let buffer = done.reads.first().cloned().unwrap_or_default();
            sites.push(Node::leaf(
                format!("{}/buffer", row.key),
                Observed::bytes(buffer, buffer_layout()),
            ));
            if row.has_an_ordinary_texture() {
                let texels = done.reads.get(1).cloned().unwrap_or_default();
                sites.push(Node::leaf(
                    format!("{}/ordinary", row.key),
                    Observed::bytes(texels, ordinary_layout()),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of every case has a name of its own, so no row's answers are read as another's.
    #[test]
    fn every_row_has_a_name_of_its_own() {
        let mut seen = std::collections::BTreeSet::new();
        for case in table() {
            for row in case.rows {
                assert!(seen.insert(row.key), "two rows are named {}", row.key);
            }
        }
    }

    /// Every row's copy lies inside both of its textures, and the texture lies inside the buffer.
    ///
    /// A region that ran off the edge of a texture is a call Metal refuses, so such a row would be
    /// recording a refusal rather than where a copy puts its bytes. Its failure mode was exercised
    /// while writing it: a size of `[SIDE, SIDE]` from an origin of `[1, 1]` fails here, naming
    /// the row and the edge it passes.
    #[test]
    fn every_rows_copy_lies_inside_its_textures_and_its_buffer() {
        for case in table() {
            for row in case.rows {
                for (what, at) in [("source", row.from), ("destination", row.to)] {
                    for (axis, (origin, size)) in at.iter().zip(&row.size).enumerate() {
                        let edge = origin + size;
                        assert!(
                            edge <= SIDE,
                            "{}: the {what}'s axis {axis} reaches texel {edge} of a texture {SIDE} \
                             texels on a side",
                            row.key
                        );
                    }
                }
                let offsets = match row.blit {
                    Blit::WithinBuffer { other, .. } => vec![OFFSET, other],
                    _ => vec![OFFSET],
                };
                for offset in offsets {
                    let reaches = offset + (SIDE - 1) * PITCH + ROW_BYTES;
                    assert!(
                        reaches <= LEN,
                        "{}: a texture at {offset} reaches byte {reaches} of a buffer of {LEN}",
                        row.key
                    );
                }
            }
        }
    }

    /// The rows of the overlapping case that say they overlap do, and the ones that say they do not
    /// do not — as [`Row::copies_shared_bytes`] reads them, which is what decides the case's
    /// `overlapping-texel-copy` declaration.
    ///
    /// The point of those rows is that the source and destination bytes are the same bytes, so a
    /// row whose offsets had drifted apart would look like a passing overlap case while asking
    /// nothing. Its failure mode was exercised while writing it: pointing the second texture at
    /// `ELSEWHERE` instead of one pitch on fails here.
    ///
    /// The expectation is computed **independently of the method**, which is the whole value of the
    /// test: the method intersects ranges, while this enumerates the actual buffer byte each texel
    /// of the source and destination rectangles occupies and intersects the two sets. A test that
    /// called the method to work out what to expect would pass whatever the method did. Every
    /// variant of [`Blit`] is covered, not only the two textures over one buffer, because a copy
    /// with an ordinary texture on one side must read as sharing nothing for the `to-and-from` case
    /// to declare only the one code.
    #[test]
    fn the_overlapping_rows_bytes_overlap_and_the_disjoint_ones_do_not() {
        // The buffer bytes a texel of the texture at `at` occupies, as the row pitch lays them.
        let texel_bytes = |at: u64, x: u64, y: u64| {
            let start = at + y * PITCH + x * (ROW_BYTES / SIDE);
            start..start + ROW_BYTES / SIDE
        };
        let rect = |at: u64, origin: [u64; 2], size: [u64; 2]| {
            let mut bytes = std::collections::BTreeSet::new();
            for y in origin[1]..origin[1] + size[1] {
                for x in origin[0]..origin[0] + size[0] {
                    bytes.extend(texel_bytes(at, x, y));
                }
            }
            bytes
        };
        for row in OVERLAPPING_ROWS.iter().chain(&TO_AND_FROM_ROWS) {
            // What the row's two sides really occupy. Only a side that is a texture **over the
            // buffer** occupies any of it, so a side that is an ordinary texture contributes none.
            let (read, written) = match row.blit {
                Blit::IntoBuffer => (
                    std::collections::BTreeSet::new(),
                    rect(OFFSET, row.to, row.size),
                ),
                Blit::OutOfBuffer => (
                    rect(OFFSET, row.from, row.size),
                    std::collections::BTreeSet::new(),
                ),
                Blit::WithinBuffer { other, reverse } => {
                    let (from_at, to_at) = match reverse {
                        false => (OFFSET, other),
                        true => (other, OFFSET),
                    };
                    (
                        rect(from_at, row.from, row.size),
                        rect(to_at, row.to, row.size),
                    )
                }
                Blit::OntoItself => (
                    rect(OFFSET, row.from, row.size),
                    rect(OFFSET, row.to, row.size),
                ),
            };
            let shares = read.intersection(&written).next().is_some();
            assert_eq!(
                row.copies_shared_bytes(),
                shares,
                "{}: the copy's bytes {} really shared, which is not what the row reads as, so the \
                 case would declare the wrong set of notices",
                row.key,
                if shares { "are" } else { "are not" }
            );
        }
        // And the family is not all of one kind, or the comparison above would hold vacuously.
        let sharing = OVERLAPPING_ROWS
            .iter()
            .chain(&TO_AND_FROM_ROWS)
            .filter(|row| row.copies_shared_bytes())
            .count();
        assert!(
            sharing > 0 && sharing < OVERLAPPING_ROWS.len() + TO_AND_FROM_ROWS.len(),
            "{sharing} rows share bytes: the tables must hold both kinds for this to check anything"
        );
    }

    /// Each case declares exactly the notice codes a row of it can announce: both, where a row's
    /// copy reads and writes the same bytes of the buffer, and the shadow alone where none does.
    ///
    /// The second direction is the one that costs a run its meaning. A declaration fires on its own
    /// code, so a code declared where no row can announce it fires on no venue and names nothing
    /// the run justifies — an allowance that sits in every report as noise a real difference can
    /// hide in. So this asserts the whole set, equal, rather than that the codes are present.
    #[test]
    fn each_case_declares_exactly_the_notices_a_row_of_it_can_announce() {
        use crate::allowance::Justification;
        use metal2vulkan_runtime::Degradation;
        for case in table() {
            let codes: Vec<String> = case
                .allowances()
                .iter()
                .map(|allowance| {
                    assert_eq!(
                        allowance.path,
                        format!("case/{}/*", crate::replay::NOTICES_SITE),
                        "{}: an allowance outside the notices site",
                        case.name()
                    );
                    match &allowance.justified_by {
                        Justification::Degradation(code) => code.clone(),
                        other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                    }
                })
                .collect();
            // The venue's own notice leads the set: both cases declare it, because both read
            // their answer back through a `Shared` buffer and a card without unified memory holds
            // that buffer in system memory.
            let mut wanted = vec![
                Degradation::SharedInSystemMemory.code().to_owned(),
                Degradation::ShadowedTexture.code().to_owned(),
            ];
            if case.rows.iter().any(Row::copies_shared_bytes) {
                wanted.push(Degradation::OverlappingTexelCopy.code().to_owned());
            }
            assert_eq!(codes, wanted, "{}", case.name());
        }
        // The two cases must not declare the same set, or the row-shaped test above is decoration.
        let sets: Vec<usize> = table().iter().map(|case| case.allowances().len()).collect();
        assert_eq!(
            sets,
            vec![2, 3],
            "one case declares the venue's notice and the shadow image, the other those two and \
             the overlapping copy"
        );
    }

    /// Every texel of every set names itself, so no two texels of a program share a value.
    ///
    /// A copy that moved the right number of bytes from the wrong place is only visible if the
    /// bytes differ, and the sets a program mixes are the buffer's set 0 and the ordinary
    /// texture's set 2.
    #[test]
    fn no_two_texels_of_a_program_hold_the_same_bytes() {
        let mut seen = std::collections::BTreeMap::new();
        for set in [0, 2] {
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let was = seen.insert(texel(x, y, set), (x, y, set));
                    assert_eq!(was, None, "texel {x},{y} of set {set} repeats {was:?}");
                }
            }
        }
    }

    /// Every site a row emits has a layout that covers the read it lays over.
    #[test]
    fn every_sites_layout_covers_the_read_it_lays_over() {
        assert_eq!(buffer_layout().extent(), LEN as usize);
        assert_eq!(ordinary_layout().extent(), (SIDE * ROW_BYTES) as usize);
    }

    /// The buffer's own pattern never holds a texel's bytes at a texel's place, so a texel read
    /// back out of the buffer came from a copy rather than from the pattern.
    #[test]
    fn the_buffers_pattern_is_not_mistakable_for_a_texel() {
        let bytes = pattern(LEN as usize);
        for set in [0, 2] {
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = (OFFSET + y * PITCH + x * 4) as usize;
                    assert_ne!(
                        bytes[at..at + 4],
                        texel(x, y, set),
                        "the pattern holds texel {x},{y} of set {set} where that texel belongs"
                    );
                }
            }
        }
    }
}
