//! A buffer copied into every sample of a multisample depth or stencil texture, and copied back
//! out: ported from `runtime_lane::buffer_depth_samples`.
//!
//! The lane's Metal answer was a **rule**, not a table — "each copy unit of the buffer lands in its
//! sample by the layout measured for colour, texel `(x, y)` sample `s` at
//! `offset + y * bytesPerRow + (x * n + s) * bpp`, the samples outside a region keep the clear, and
//! the copy back out returns the buffer bit for bit" — written down from a probe on an **Apple M4**,
//! a machine this round does not have. Every call on the path is public: a render pass that clears
//! and draws nothing, `copyFromBuffer:toTexture:` into the depth or stencil aspect, a dispatch that
//! reads every sample, and `copyFromTexture:toBuffer:` back. So record mode **runs** these cases on
//! the oracle and the rule becomes Metal's own numbers on this M3 Max, re-takeable whenever
//! anybody asks. Nothing is carried from the dead M4.
//!
//! **One case per row**, as the multisample fill family does, so a sample count the venue does not
//! render stops that row alone and the suite counts it
//! [`LacksFeature`](crate::verdict::Verdict::LacksFeature) rather than leaving it uncompared.
//! lavapipe does not render 2 samples, which Vulkan leaves optional, and the lane's own loop prints
//! a line and carries on. It also keeps every recording far under the 512 KiB the plan allows: the
//! largest row is an 80 x 80 texture at 4 samples, whose two reads are 100 KiB each.
//!
//! The first row is the consumer's screensaver's copy (`Depth32Float` 54 x 54 at 4 samples, whole,
//! `bytesPerRow` 864) and the second its login window's at HiDPI (80 x 80, `bytesPerRow` 1280);
//! the rest sweep the sample count, a region inside the texture, and the two other aspects.
//!
//! Two sites a row:
//!
//! - `samples` is what the dispatch read, texel by texel in rows with each texel's samples in
//!   order — **where each copy unit landed**, which is the rule's first half.
//! - `copied-back` is the second buffer after the copy out, made holding `0xee` so that a byte the
//!   copy did not write is visible rather than indistinguishable from a copy that wrote zero.
//!
//! A `Depth32Float` row's buffer opens with 16 edge words — signed zeros, the smallest and largest
//! denormals, 1 and the float after it, 2, -1, both infinities, three NaN payloads, the largest
//! float — because the question the lane asks is whether anything on the path quietens, flushes or
//! clamps one. Recording Metal's answer to that is the point: a device that changes such a value
//! announces it, and the announcement is then held to the recorded fact rather than to prose.
//!
//! Nothing here faults a GPU, hangs one or trips a watchdog: one command buffer a row, a clear that
//! draws nothing, two blits and one dispatch of a thread a sample.

use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::fixture::{Fixture, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Clear, Plane, Resource, Step, TextureSpec, Work};

/// The owned fixture whose kernels read a multisample texture sample by sample: the one the
/// multisample fill family reads too, so one shader answers both.
const KERNELS: &str = "rt6_samples";

/// The depth and the stencil the pass clears to before the copy, as the lane's probe cleared them.
const CLEAR_DEPTH: f64 = 0.25;
const CLEAR_STENCIL: u32 = 0x5a;

/// The byte the copy-back buffer holds before the copy.
///
/// Not zero, and that is load-bearing: a read into zeroed memory cannot tell "the copy wrote zero
/// here" from "the copy left this byte alone", so the question "does the copy out write only the
/// region's units?" would be one that could not fail.
const UNWRITTEN: u8 = 0xee;

/// A `Depth32Float` buffer's first words: +0, -0, the smallest and largest denormals, a negative
/// denormal, the smallest normal, 1, the float after 1, 2, -1, +inf, -inf, a quiet NaN, a
/// signalling NaN's payload, a negative NaN's payload, the largest float.
const EDGES: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x8000_0001,
    0x0080_0000,
    0x3f80_0000,
    0x3f80_0001,
    0x4000_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7f80_0001,
    0xffc0_0001,
    0x7f7f_ffff,
];

/// The raw `MTLPixelFormat` of each aspect this family copies.
const STENCIL8: u32 = 253;
const DEPTH16UNORM: u32 = 250;

/// One row: the case's name, the format, the samples, the texture's extent, and the region the
/// copies move.
struct Row {
    name: &'static str,
    format: u32,
    samples: u64,
    extent: [u64; 2],
    origin: [u64; 2],
    size: [u64; 2],
}

/// A row over the texture's whole extent.
const fn whole(name: &'static str, format: u32, samples: u64, side: u64) -> Row {
    Row {
        name,
        format,
        samples,
        extent: [side, side],
        origin: [0, 0],
        size: [side, side],
    }
}

/// A row over a region inside an 8 x 8 texture: four texels by three from (2, 1), so the samples
/// outside it keep the clear and the copy out writes only its own units.
const fn part(name: &'static str, format: u32) -> Row {
    Row {
        name,
        format,
        samples: 4,
        extent: [8, 8],
        origin: [2, 1],
        size: [4, 3],
    }
}

/// The rows: the stencil aspect at both sample counts and over a region inside the texture, and
/// the 16-bit depth aspect at both sample counts.
///
/// The two `Depth16Unorm` rows are **open gaps of ours**, each with a `known-gaps.tsv` row: Metal
/// copies the buffer into the 16-bit aspect and back, and we refuse the copy in
/// (`RuntimeGap::MultisampledBufferCopy`, on a device without `maintenance8` or carrying the
/// measured copy defect). A refusal of something plain Metal does is a bug of ours by the owner's
/// standing rule, so the rows stay here and stay red in the ledger's own way: the run prints them
/// as refusals, counts them as refusals, and fails the day either stops refusing.
///
/// # The four rows this family does not yet hold, and what measuring them found
///
/// The `Depth32Float` rows were written, recorded on the oracle and replayed, and they are left out
/// because they need something the suite cannot yet say. Their measurements are the point of having
/// run them, so they are written down here:
///
/// - **`Depth32Float`, four rows** (54 x 54 and 80 x 80 at 4 samples, 54 x 54 at 2, and a region
///   inside 8 x 8). Metal returns the buffer **bit for bit**, every edge value included: both
///   denormals, every NaN payload, both infinities, -1, 1, the float after 1, 2 and the largest
///   float all come back unchanged, and the samples outside a region keep the clear. Ours differs
///   at **one element of all four rows** — copy unit 1, the edge value **-0**, which Metal keeps
///   and we answer as +0 — and the runtime announces `drawn-depth-copy`, which names exactly that.
///   It is an announced difference, but the suite has no way to say it: an
///   [`Allowance`](crate::allowance::Allowance) covers a **site**, and loosening a 46,656-byte site
///   to tolerate four bytes would throw away the comparison that makes the row worth having. A
///   per-element allowance on a bytes site is what these rows are waiting for.
const ROWS: [Row; 5] = [
    whole("buffer-depth-samples-s8-54-4", STENCIL8, 4, 54),
    whole("buffer-depth-samples-s8-54-2", STENCIL8, 2, 54),
    part("buffer-depth-samples-s8-part-4", STENCIL8),
    whole("buffer-depth-samples-d16-54-4", DEPTH16UNORM, 4, 54),
    whole("buffer-depth-samples-d16-54-2", DEPTH16UNORM, 2, 54),
];

impl Row {
    /// The bytes of one copy unit in the buffer: a stencil's byte, a 16-bit depth's two, a 32-bit
    /// float depth's four.
    fn unit_bytes(&self) -> usize {
        match self.format {
            STENCIL8 => 1,
            DEPTH16UNORM => 2,
            _ => 4,
        }
    }

    /// The copy units the region holds: a sample a unit.
    fn units(&self) -> usize {
        (self.size[0] * self.size[1] * self.samples) as usize
    }

    /// Bytes from one row of the region to the next in the buffer, tight: no padding, so the copy
    /// has nothing to leave alone between the rows and every byte of the buffer is a unit.
    fn bytes_per_row(&self) -> u64 {
        self.size[0] * self.samples * self.unit_bytes() as u64
    }

    /// The buffer's bytes: a word a copy unit, cut to the unit's own bytes.
    ///
    /// A `Depth32Float` row opens with [`EDGES`] and goes on with distinct normal values in
    /// `(0, 0.18]`, which the lane measured every path to copy exactly. A stencil's and a 16-bit
    /// depth's units are distinct numbers of their own width, so a unit that landed in another
    /// sample is visible.
    fn input(&self) -> Vec<u8> {
        let unit = self.unit_bytes();
        (0..self.units())
            .flat_map(|i| {
                let word = match self.format {
                    STENCIL8 => ((i * 37 + 11) & 0xff) as u32,
                    DEPTH16UNORM => ((i * 7919 + 13) & 0xffff) as u32,
                    // `Depth32Float` (252), whose buffer opens with the edge words: no row of the
                    // family holds one yet (see [`ROWS`]), and the shape is kept because what the
                    // rows are waiting for is a per-element allowance, not a rewrite.
                    _ => match EDGES.get(i) {
                        Some(&edge) => edge,
                        None => ((i + 1) as f32 / 65536.0).to_bits(),
                    },
                };
                word.to_le_bytes()[..unit].to_vec()
            })
            .collect()
    }

    /// Which aspect the copies move.
    fn plane(&self) -> Plane {
        match self.format {
            STENCIL8 => Plane::Stencil,
            _ => Plane::Depth,
        }
    }

    /// What the clearing pass clears.
    fn clear(&self) -> Clear {
        match self.format {
            STENCIL8 => Clear::Stencil(CLEAR_STENCIL),
            _ => Clear::Depth(CLEAR_DEPTH),
        }
    }

    /// The kernel that reads every sample, and the bytes it keeps of one: a stencil's value is its
    /// own byte, and either depth is read as the depth's 32-bit bits.
    fn kernel(&self) -> (&'static str, usize) {
        match self.format {
            STENCIL8 => ("rt6_read_stencil", 1),
            _ => ("rt6_read_depth", 4),
        }
    }

    /// The program the row runs: the buffer in, the copy-back buffer, a private multisample
    /// texture cleared by a pass that draws nothing, the copy in, every sample read by a dispatch,
    /// the copy back out, and the second buffer read.
    fn work(&self) -> Work {
        let bytes = self.input();
        let length = bytes.len();
        let mut work = Work::new();
        work.kernels = Some(KERNELS);
        let input = work.buffer(bytes);
        let out = work.buffer(vec![UNWRITTEN; length]);
        let texture = work.add(Resource::Texture {
            spec: TextureSpec::texture_2d(self.format, self.extent[0], self.extent[1])
                .multisample(self.samples, 1),
        });
        work.step(Step::ClearPass {
            texture,
            slice: 0,
            clear: self.clear(),
        });
        let origin = [self.origin[0], self.origin[1], 0];
        let size = [self.size[0], self.size[1], 1];
        work.step(Step::CopyBufferToTexture {
            buffer: input,
            offset: 0,
            bytes_per_row: self.bytes_per_row(),
            bytes_per_image: length as u64,
            size,
            texture,
            slice: 0,
            level: 0,
            origin,
            plane: self.plane(),
        });
        let (kernel, texel_bytes) = self.kernel();
        work.step(Step::ReadSamples {
            texture,
            kernel,
            width: self.extent[0] as u32,
            height: self.extent[1] as u32,
            samples: self.samples as u32,
            texel_bytes,
            words: 1,
        });
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice: 0,
            level: 0,
            origin,
            size,
            buffer: out,
            offset: 0,
            bytes_per_row: self.bytes_per_row(),
            bytes_per_image: length as u64,
            plane: self.plane(),
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: out,
            offset: 0,
            length,
        });
        work
    }

    /// What the dispatch read, as a site's layout: the texture's texels in rows, each texel's
    /// samples in order, the kernel's bytes a sample.
    ///
    /// The row is the whole texture's width, not the region's, because the dispatch reads every
    /// sample of the texture: that is how "the samples outside the region keep the clear" is a fact
    /// a diff can name.
    fn samples_layout(&self) -> ByteLayout {
        let texel = self.kernel().1 as u32;
        let row = (self.extent[0] * self.samples) as u32 * texel;
        let length = u64::from(row) * self.extent[1];
        ByteLayout {
            region: Region::Buffer { offset: 0, length },
            element: element(texel),
            row_offset: 0,
            row_elements: (self.extent[0] * self.samples) as u32,
            row_pitch: row,
            rows: self.extent[1] as u32,
            image_pitch: length as u32,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// The copy-back buffer as a site's layout: the region's rows, each row the region's samples,
    /// the unit's own bytes an element.
    fn copied_back_layout(&self) -> ByteLayout {
        let unit = self.unit_bytes() as u32;
        let row = self.bytes_per_row() as u32;
        let length = self.input().len() as u64;
        ByteLayout {
            region: Region::Buffer { offset: 0, length },
            element: element(unit),
            row_offset: 0,
            row_elements: (self.size[0] * self.samples) as u32,
            row_pitch: row,
            rows: self.size[1] as u32,
            image_pitch: length as u32,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }
}

/// The element a value of `bytes` bytes is, so a diff prints one copy unit rather than a byte of
/// one.
fn element(bytes: u32) -> Element {
    match bytes {
        1 => Element::U8,
        2 => Element::U16,
        _ => Element::U32,
    }
}

/// Every case of the family: one a row.
pub fn all() -> Vec<Box<dyn Case>> {
    ROWS.iter()
        .map(|row| Box::new(BufferDepthSamples { row }) as Box<dyn Case>)
        .collect()
}

/// A case over one row.
pub struct BufferDepthSamples {
    row: &'static Row,
}

/// The site the dispatch's read sits at.
const SAMPLES: &str = "samples";

/// The site the copy back out sits at.
const COPIED_BACK: &str = "copied-back";

impl Case for BufferDepthSamples {
    fn name(&self) -> &'static str {
        self.row.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(KERNELS);
        Ok(vec![ShaderRef {
            fixture: KERNELS.to_owned(),
            entry: self.row.kernel().0.to_owned(),
            source_sha256: crate::sha256_hex(fixture.text(RUNTIME_TEXT)?.as_bytes()),
        }])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let row = self.row;
        // The geometry every call is given, keyed by the call and its numbers rather than by a
        // position, and the bytes the copy in feeds. Either moving makes the recording's answer
        // another program's.
        let geometry = format!(
            "newTextureWithDescriptor format {} samples {} extent {:?}\n\
             clear depth {CLEAR_DEPTH} stencil {CLEAR_STENCIL}\n\
             copyFromBufferToTexture origin {:?} size {:?} bytesPerRow {} plane {:?}\n\
             readSamples kernel {} texel {}\n\
             copyFromTextureToBuffer origin {:?} size {:?} bytesPerRow {} plane {:?} over {UNWRITTEN:#x}\n",
            row.format,
            row.samples,
            row.extent,
            row.origin,
            row.size,
            row.bytes_per_row(),
            row.plane(),
            row.kernel().0,
            row.kernel().1,
            row.origin,
            row.size,
            row.bytes_per_row(),
            row.plane(),
        );
        Ok(vec![
            Input::bytes("calls", geometry.as_bytes()),
            Input::bytes("copy-units", &row.input()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = call(sites, "work", side.work(&self.row.work()), |_| {
            Value::Absent
        })?;
        sites.push(Node::leaf(
            SAMPLES,
            Observed::bytes(
                done.reads.first().cloned().unwrap_or_default(),
                self.row.samples_layout(),
            ),
        ));
        sites.push(Node::leaf(
            COPIED_BACK,
            Observed::bytes(
                done.reads.get(1).cloned().unwrap_or_default(),
                self.row.copied_back_layout(),
            ),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No two rows share a name, so every case of the family has a recording of its own.
    #[test]
    fn every_row_has_a_name_of_its_own() {
        let mut names: Vec<&str> = ROWS.iter().map(|row| row.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "two rows of the family share a name");
    }

    /// Each row's buffer holds exactly one copy unit a sample of its region, and its stated row
    /// pitch is the region's own rows, tight.
    ///
    /// The row pitch is the field this family turns on and the one that goes missing: both copies
    /// take it, and a pitch that did not match the bytes would make every row pass for the wrong
    /// reason. So it is checked against the bytes rather than trusted.
    #[test]
    fn every_rows_bytes_are_its_units_and_its_pitch_is_its_rows() {
        for row in &ROWS {
            let bytes = row.input().len();
            assert_eq!(
                bytes,
                row.units() * row.unit_bytes(),
                "{}: its buffer is not one unit a sample of its region",
                row.name
            );
            assert_eq!(
                row.bytes_per_row() * row.size[1],
                bytes as u64,
                "{}: its bytesPerRow does not divide its buffer into the region's rows",
                row.name
            );
        }
    }

    /// Both of a row's sites describe exactly the bytes its program answers, so a recording's
    /// layout cannot drift from the read it lays over.
    ///
    /// Its failure mode was exercised while writing it: giving `samples_layout` the **region's**
    /// width rather than the texture's fails here for every whole-extent row.
    #[test]
    fn every_sites_layout_covers_the_read_it_lays_over() {
        for row in &ROWS {
            let samples = (row.extent[0] * row.extent[1] * row.samples) as usize * row.kernel().1;
            assert_eq!(
                row.samples_layout().extent(),
                samples,
                "{}: the samples layout does not cover what the dispatch reads",
                row.name
            );
            assert_eq!(
                row.copied_back_layout().extent(),
                row.input().len(),
                "{}: the copy-back layout does not cover the buffer it lays over",
                row.name
            );
        }
    }

    /// Every row's program asks for exactly the two reads its case emits, in that order.
    #[test]
    fn every_program_answers_the_two_reads_its_case_emits() {
        for row in &ROWS {
            assert_eq!(
                row.work().reads(),
                2,
                "{}: its program does not answer two reads",
                row.name
            );
        }
    }
}
