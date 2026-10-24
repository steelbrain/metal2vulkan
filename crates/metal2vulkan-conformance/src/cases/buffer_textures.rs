//! Textures laid over a buffer's bytes: which shapes Metal makes one of, what the made texture
//! answers of itself, and where its texels are in the buffer.
//!
//! Ported from `runtime_lane::buffer_textures`. Metal's whole path here is **public** —
//! `newTextureWithDescriptor:offset:bytesPerRow:` on an `MTLBuffer`, then `getBytes:`,
//! `replaceRegion:` and the buffer's own `contents` — so record mode runs every case of this
//! family on the oracle rather than carrying a table, and the lane's numbers become Metal's own
//! answers, re-takeable whenever anybody asks.
//!
//! Three cases, because the family asks three different kinds of question:
//!
//! - `buffer-textures-made` records, for each shape, whether Metal lays a texture over the buffer
//!   at all and what the texture then answers of itself, field by field
//!   ([`crate::buffer_texture::BufferTextureFacts`]). The subject is the **rule table**: a texture
//!   buffer takes any row that covers its texels and any offset; a 2D texture's rows lie
//!   `bytesPerRow` apart; and the made texture reports **the buffer's** resource options whatever
//!   its own descriptor named.
//! - `buffer-textures-cpu-access` records the bytes: the host reads and writes the texture, and
//!   every byte of the buffer comes back, so Metal says where it laid the texels and what it left
//!   alone between the rows.
//! - `buffer-textures-stride-below-the-row` is the one shape of the family that **must never run
//!   on Metal**: see below.
//!
//! # The shape nothing here runs
//!
//! A `bytesPerRow` **below the texture's own row** fails the AGX driver's assertion
//! `bytes_per_row >= used_bytes_per_row` and ends the process. So that row is
//! [`MetalSource::did_not_return`]: its recording is written by hand, record mode refuses it, and
//! [`crate::buffer_texture::ends_metals_process`] refuses the call on the Metal side as the second
//! line of defence. What the case pins is the one fact there is — Metal did not come back — against
//! our typed refusal (`BufferTextureProblem::RowPitch`) at the same site, so a release where we
//! stop refusing it reads as a divergence rather than as a pass.
//!
//! # What is deliberately not ported, and why
//!
//! The lane also checks five other refusals, each of which the runtime answers
//! `Error::Invalid`: a texture whose texels end past the buffer's length, a block-compressed
//! format, a 3D descriptor, a 2D descriptor at an offset off the format's alignment, and a 2D
//! array through this call. `Error::Invalid` is, by the runtime's own documented model, "Metal
//! stops this call" — as against `Error::Nil`, "plain Metal answers nil and the process goes on" —
//! so handing any of them to Metal risks exactly what the stride row does. Unlike the stride row,
//! no probe of this tree has recorded *how* Metal stops them, so there is nothing to state and the
//! suite would be guessing. They stay `runtime_lane` assertions until somebody measures them
//! safely, and the honest place for that is the refusal surface rather than a case here.
//!
//! The lane's remaining three tests are elsewhere by subject: the shareable texture buffer and its
//! handle, the unbound texture-buffer slots reading the device's neutrals (what Metal reads there
//! is undefined, so there is nothing to compare), and the texture buffer at an offset off 16 bytes
//! whose *functions* read from the offset rounded down, which is a dispatch case rather than a
//! texture-making one.
//!
//! Nothing here faults a GPU, hangs one or trips a watchdog: the making call allocates no texels of
//! its own, and the bytes case runs one blit-free program of host calls over shared memory.

use crate::buffer_texture::BufferTextureRequest as Request;
use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::measured::stripes;
use crate::replay::NOTICES_SITE;
use crate::texture_query::Storage;
use crate::transcript::{
    ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Transcript, Value,
};
use crate::work::{Rect, Resource, Step, Storage as WorkStorage, TextureKind, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The case names, as the report and the recordings spell them.
pub const MADE: &str = "buffer-textures-made";
pub const CPU_ACCESS: &str = "buffer-textures-cpu-access";
pub const STRIDE_BELOW_THE_ROW: &str = "buffer-textures-stride-below-the-row";

/// The usage a texture buffer of this family carries, as the lane's descriptor does: read and
/// written by shaders.
const READ_WRITE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);

/// The buffer every making row is laid over: 4096 bytes, as the lane's is.
const BUFFER: u64 = 4096;

/// What the two recorded cases loosen, and what justifies it.
///
/// A 2D texture over a buffer whose rows the device's own linear image does not lay out as Metal
/// does is an image the encoders copy the texels into and back — a **shadow** — and the runtime
/// announces that once per device (`shadowed-texture`). Metal announces nothing, ever, so the
/// notices site differs for every case of this family that holds a 2D row, and the notice is what
/// justifies the difference. Measured on this M3 Max: the texture buffers alias their bytes and
/// announce nothing, and it is the 2D rows that take the shadow.
///
/// Nothing else is loosened. Every fact and every byte of both cases compares exactly.
fn shadow_allowances() -> Vec<crate::allowance::Allowance> {
    // Every row reads its answer back through a `Shared` buffer, so on a device without
    // unified memory the runtime announces `shared-in-system-memory`
    // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
    // notices site alone, so every value of this family still compares against Metal's
    // exactly.
    //
    // Measured on the RTX 5080 Laptop (615.71.09): it fired for
    // all 3 rows of the family, and each was divergent on that field and
    // nothing else (replay-nvidia.log). On MoltenVK, where
    // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
    // recording of this suite declares it.
    vec![
        crate::replay::shared_memory_notice(),
        crate::replay::notice_allowance(metal2vulkan_runtime::Degradation::ShadowedTexture.code()),
    ]
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(made()),
        Box::new(cpu_access()),
        Box::new(stride_below_the_row()),
    ]
}

/// The shapes the making case asks about, in the order it asks them.
///
/// Every row is one the runtime answers — a texture or Metal's own nil — and a device test of this
/// module holds the table to that, because a row the runtime answers `Invalid` for is a row
/// record mode would hand to Metal's assertion.
fn requests() -> Vec<Request> {
    let mut rows = Vec::new();
    // A texture buffer of 64 `R32Uint` texels: its row is 256 bytes.
    let texels =
        Request::texture_buffer(BUFFER, PixelFormat::R32Uint, 64, 4, 0, 256).used(READ_WRITE);
    rows.push(texels);
    // A row longer than the texels, and one as long as the whole buffer: a texture buffer takes
    // any row that covers its texels.
    rows.push(Request {
        bytes_per_row: 300,
        ..texels
    });
    rows.push(Request {
        bytes_per_row: BUFFER,
        ..texels
    });
    // An offset off 16 bytes, which Metal's own validation stops and plain Metal takes.
    for offset in [4u64, 8, 16, 20, 33, 64] {
        rows.push(Request { offset, ..texels });
    }
    // The descriptor naming a storage mode other than the buffer's, and the buffer in each other
    // mode: the made texture reports **the buffer's** options whatever its descriptor said, which
    // is the fact these rows record. The shared-and-shared pair is the base row above, so it is
    // not repeated.
    for storage in [Storage::Managed, Storage::Private] {
        rows.push(texels.descriptor_storage(storage));
        rows.push(texels.over(storage));
    }
    // Texture buffers of other texel sizes: a one-byte texel whose row is far shorter than the
    // pitch, and a four-byte one at an offset.
    rows.push(Request::texture_buffer(BUFFER, PixelFormat::R8Uint, 12, 1, 0, 256).used(READ_WRITE));
    rows.push(
        Request::texture_buffer(BUFFER, PixelFormat::RGBA8Uint, 16, 4, 512, 64).used(READ_WRITE),
    );
    rows.push(
        Request::texture_buffer(BUFFER, PixelFormat::RGBA32Float, 8, 16, 0, 256).used(READ_WRITE),
    );
    // 2D textures over the buffer: rows exactly as wide as the texels, wider, and at an offset.
    let flat =
        Request::texture_2d(BUFFER, PixelFormat::R32Uint, [16, 4], 4, 0, 64).used(READ_WRITE);
    rows.push(flat);
    rows.push(Request {
        bytes_per_row: 256,
        ..flat
    });
    rows.push(Request {
        offset: 1024,
        ..flat
    });
    rows.push(Request {
        offset: 1024,
        bytes_per_row: 256,
        ..flat
    });
    // Three texels by two rows at a pitch of 32: the shape whose two rows the bytes case reads.
    rows.push(
        Request::texture_2d(BUFFER, PixelFormat::R32Uint, [3, 2], 4, 256, 32).used(READ_WRITE),
    );
    // A one-byte format at a pitch far wider than its row. Measured while writing this table: a
    // pitch of 8 — the row exactly — is refused `Invalid` by the runtime, which by its own model
    // means Metal stops the call, so the row that reached Metal is the padded one. Which pitches a
    // 2D linear texture takes is Metal's own rule and belongs in the refusal surface rather than
    // here, where it cannot be asked safely.
    rows.push(
        Request::texture_2d(BUFFER, PixelFormat::R8Unorm, [8, 8], 1, 0, 256).used(READ_WRITE),
    );
    rows.push(
        Request::texture_2d(BUFFER, PixelFormat::RGBA8Unorm, [8, 8], 4, 0, 32).used(READ_WRITE),
    );
    rows
}

/// The making case: for each shape, whether Metal laid a texture over the buffer and what it
/// answers of itself.
pub fn made() -> MadeOverBuffer {
    MadeOverBuffer {
        name: MADE,
        requests: requests(),
    }
}

/// A case over one table of shapes.
pub struct MadeOverBuffer {
    name: &'static str,
    requests: Vec<Request>,
}

impl Case for MadeOverBuffer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        shadow_allowances()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every shape and the order they are asked in: a table that gains, loses or re-points a
        // row is a change to the case's inputs, so a recording cannot decay into the answers of
        // shapes nobody asked about.
        let mut words = Vec::new();
        for request in &self.requests {
            words.extend(request.words());
        }
        Ok(vec![Input::bytes("requests", &words)])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for request in &self.requests {
            // Each shape is a site of its own and a call that did not answer is recorded at its
            // site, the sweep going on: one shape the runtime refuses compares as a refusal there
            // and the other rows stay readable instead of being lost behind an abort.
            let node = match side.buffer_texture(request) {
                Ok(facts) => facts.node(request.site()),
                Err(failure) => Node::leaf(request.site(), Observed::value(failure.recorded())),
            };
            sites.push(node);
        }
        Ok(())
    }
}

/// The shape whose row pitch is below the texture's own row: 63 bytes under a 16-texel `RGBA8Uint`
/// texture buffer's row of 64.
///
/// It is a `const` of its own, and the one place the number lives, because three things must say
/// the same thing about it: the case's inputs, the guard that refuses it
/// ([`crate::buffer_texture::ends_metals_process`]), and the module's own test of that guard.
const BELOW: Request = Request::texture_buffer(BUFFER, PixelFormat::RGBA8Uint, 16, 4, 0, 63);

/// The site the stride row's one answer sits at.
///
/// `'static`, because a recording's provenance names the call Metal did not return from and must
/// hold the same words the site does.
const BELOW_SITE: &str = "newTextureWithDescriptor:offset:bytesPerRow:/pitch-below-the-row";

/// Where the assertion is written down.
const ASSERTED_IN: &str = "AGX's own assertion text, which this suite's own whole-set replay \
                           prints from the driver (`AGX: Texture read/write assertion failed: \
                           bytes_per_row >= used_bytes_per_row`), and the refusal-surface stream's \
                           finding of 2026-10-23 that Metal ends the process on it";

/// How Metal goes away, in the words of the finding.
const ASSERTED_SAYS: &str =
    "a texture over a buffer whose bytesPerRow is below the texture's own row fails AGX's \
     assertion `bytes_per_row >= used_bytes_per_row`, which ends Metal's process in the call; the \
     runtime refuses it typed (BufferTextureProblem::RowPitch) and that refusal is what the row \
     pins";

/// The runs behind the answer: the finding met the assertion once and the process ended with it.
const ASSERTED_RUNS: u32 = 1;

/// The stride case.
pub fn stride_below_the_row() -> StrideBelowTheRow {
    StrideBelowTheRow {
        name: STRIDE_BELOW_THE_ROW,
    }
}

/// The case over the one shape Metal does not come back from.
pub struct StrideBelowTheRow {
    name: &'static str,
}

impl StrideBelowTheRow {
    /// The transcript the finding states: the one site, holding the negative fact.
    ///
    /// Present and negative, rather than absent: "Metal never returned" and "nobody looked" must
    /// not print the same, and a run of ours that *answers* here is a divergence.
    pub fn stated_transcript(&self) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.push(crate::handler_waits::never_returned(BELOW_SITE));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }

    /// The whole answer the case's recording holds, built the only way such an answer may be
    /// built ([`crate::record::never_returned`]).
    pub fn stated_answer(&self) -> Result<crate::recording::Answer, String> {
        Ok(crate::record::never_returned(
            metal2vulkan_runtime::MetalLevel::NEWEST,
            ASSERTED_IN,
            BELOW_SITE,
            crate::recording::ORACLE_CHIP,
            ASSERTED_RUNS,
            self.stated_transcript()?,
        ))
    }
}

impl Case for StrideBelowTheRow {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::did_not_return(ASSERTED_IN, BELOW_SITE, ASSERTED_SAYS)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![Input::bytes("request", &BELOW.words())])
    }

    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(Some((ASSERTED_RUNS, self.stated_transcript()?)))
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // Both arms emit the one site, because the site is what the comparison settles on: a typed
        // error there is the row's pass, and any answer at all is a divergence.
        match side.buffer_texture(&BELOW) {
            Ok(facts) => sites.push(facts.node(BELOW_SITE)),
            Err(failure) => sites.push(Node::leaf(BELOW_SITE, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}

/// One row of the bytes case: a texture over a striped buffer, the host's reads and writes of it,
/// and every byte of the buffer afterwards.
struct Bytes {
    /// How the row is named, which is the site its answers sit under.
    key: &'static str,
    /// The texture's kind and extent.
    kind: TextureKind,
    format: u32,
    width: u64,
    height: u64,
    /// The texture's first byte in the buffer, and the bytes between its rows.
    offset: u64,
    bytes_per_row: u64,
    /// The host's reads.
    reads: &'static [Read],
    /// The host's write: a region, the pitch it writes at, and the bytes it writes.
    write: ([u64; 4], u64, usize),
}

/// One read the host makes of a row's texture.
struct Read {
    /// The region read, as `x, y, width, height`.
    region: [u64; 4],
    /// The pitch the destination is read at, which may be wider than a row of the region.
    pitch: u64,
    /// The bytes of one row of the region, which the case **states**: `MTLTexture` answers no
    /// bytes-per-texel, and a format table of ours would be a second spelling of Metal's to keep
    /// in step. A read whose pitch is wider than this has padding between its rows, and that
    /// padding is what [`Read::before`] asks about.
    row_bytes: u64,
    /// The bytes of the destination.
    length: usize,
    /// The byte every one of those `length` bytes holds before the call.
    ///
    /// A read at a pitch wider than its rows asks two questions at once — the bytes the call
    /// wrote, and whether it left the padding between the rows alone. Zero answers the second
    /// badly, because a call that wrote zeros there and a call that wrote nothing read alike, so
    /// such a read states a byte no stripe of the buffer holds.
    before: u8,
}

/// The bytes rows: a texture buffer whose texels are the buffer's bytes, and a 2D texture whose
/// rows lie `bytesPerRow` apart with the bytes between them untouched.
const BYTES_ROWS: [Bytes; 2] = [
    // 16 `RGBA8Uint` texels from byte 512, as the lane reads them: the whole row, then four texels
    // from the fifth, then four texels written from the third.
    Bytes {
        key: "texture-buffer-rgba8uint-16",
        kind: TextureKind::TextureBuffer,
        format: PixelFormat::RGBA8Uint.as_raw(),
        width: 16,
        height: 1,
        offset: 512,
        bytes_per_row: 64,
        reads: &[
            Read {
                region: [0, 0, 16, 1],
                pitch: 64,
                row_bytes: 64,
                length: 64,
                before: 0,
            },
            Read {
                region: [4, 0, 4, 1],
                pitch: 16,
                row_bytes: 16,
                length: 16,
                before: 0,
            },
        ],
        write: ([2, 0, 4, 1], 16, 16),
    },
    // Three `R32Uint` texels by two rows from byte 256 at a pitch of 32: the rows are 12 bytes and
    // lie 32 apart, so the 20 bytes between them are what the read and the write must leave alone.
    //
    // Its reads ask that of both sides of the call. The first reads the region at its own pitch,
    // where the texture's padding must not reach the destination; the second reads it at a pitch
    // of 32, where the destination has padding of its own and `getBytes:` must leave it as it
    // found it.
    Bytes {
        key: "texture-2d-r32uint-3x2",
        kind: TextureKind::Type2D,
        format: PixelFormat::R32Uint.as_raw(),
        width: 3,
        height: 2,
        offset: 256,
        bytes_per_row: 32,
        reads: &[
            Read {
                region: [0, 0, 3, 2],
                pitch: 12,
                row_bytes: 12,
                length: 24,
                before: 0,
            },
            // The same region into a destination whose rows lie 32 apart: 12 bytes of texels and
            // 20 of padding, twice, and the last row's padding is past the bytes the call needs,
            // so the destination is the 44 bytes the call reaches. `0xfd` is 253, which no stripe
            // of the buffer holds (they are `i * 37 % 251`), so a padding byte the call wrote from
            // the buffer and one it left alone cannot read alike.
            Read {
                region: [0, 0, 3, 2],
                pitch: 32,
                row_bytes: 12,
                length: 44,
                before: 0xfd,
            },
        ],
        write: ([0, 0, 3, 2], 12, 24),
    },
];

/// The bytes a row's write puts in: `n` bytes counting up from `seed`, which no stripe of the
/// buffer holds in that order, so a write that landed elsewhere is visible.
fn written(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

impl Bytes {
    /// The texture's descriptor as the program states it.
    fn spec(&self) -> TextureSpec {
        TextureSpec {
            kind: self.kind,
            storage: WorkStorage::Shared,
            gpu_optimized: false,
            usage: Some(READ_WRITE.as_raw()),
            ..TextureSpec::texture_2d(self.format, self.width, self.height)
        }
    }

    /// The seed the row's write counts up from: the row's own, so two rows' writes differ.
    fn seed(&self) -> u8 {
        match self.kind {
            TextureKind::TextureBuffer => 99,
            _ => 77,
        }
    }
}

/// The program the bytes case runs: each row's buffer of stripes, its texture over them, its
/// reads, its write, and every byte of the buffer read back.
fn bytes_work() -> Work {
    let mut work = Work::new();
    for row in &BYTES_ROWS {
        let buffer = work.buffer(stripes(BUFFER as usize));
        let texture = work.add(Resource::TextureOverBuffer {
            buffer,
            spec: row.spec(),
            offset: row.offset,
            bytes_per_row: row.bytes_per_row,
        });
        for read in row.reads {
            let [x, y, width, height] = read.region;
            work.step(Step::GetBytes {
                texture,
                level: 0,
                slice: 0,
                region: Rect::new(x, y, width, height),
                bytes_per_row: read.pitch,
                bytes_per_image: 0,
                length: read.length,
                before: read.before,
            });
        }
        let ([x, y, width, height], pitch, length) = row.write;
        work.step(Step::ReplaceRegion {
            texture,
            level: 0,
            slice: 0,
            region: Rect::new(x, y, width, height),
            bytes: written(length, row.seed()),
            // An uncompressed format's rows of bytes are its texel rows, which is the region's
            // height: what this call always meant before a block-compressed family needed the
            // difference named.
            rows: height,
            bytes_per_row: pitch,
            bytes_per_image: 0,
        });
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: BUFFER as usize,
        });
    }
    work
}

/// The whole buffer as a site's layout, divided into the texture's own rows: a difference names
/// the row and the byte inside it rather than a flat offset, and the texture's first byte lies at
/// the start of one of them (512 is row 8 of 64, 256 is row 8 of 32).
///
/// **Every byte of the buffer is an element**, so the bytes the texture does not cover — the 20
/// bytes between a 12-byte row and the next row's start among them — compare like any other. That
/// is the whole point of reading the buffer back rather than reading the texture: a write that
/// landed between the rows is visible here and nowhere else.
fn buffer_layout(row: &Bytes) -> ByteLayout {
    let pitch = u32::try_from(row.bytes_per_row).unwrap_or(u32::MAX).max(1);
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: BUFFER,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: pitch,
        row_pitch: pitch,
        rows: BUFFER as u32 / pitch,
        image_pitch: BUFFER as u32,
        images: 1,
        // Nothing lies between the rows of this view: a row's elements are its whole pitch.
        gaps: Gaps::Ignored,
    }
}

/// A read the host made, as a site's layout: the rows of the region it read, at the pitch it read
/// them at, with the padding between them compared where the pitch leaves any.
///
/// The elements are the row's own bytes, not the whole pitch, so a difference names the row and
/// the byte inside it. Where the pitch is wider, [`Gaps::Compared`] puts the padding in the
/// comparison as well: a `getBytes:` that wrote where it should not have is a difference here, and
/// that is the question the row's [`Read::before`] sets up.
fn read_layout(read: &Read) -> ByteLayout {
    let pitch = u32::try_from(read.pitch).unwrap_or(u32::MAX).max(1);
    // A row no wider than the pitch, so the elements of one row never reach into the next.
    let row = u32::try_from(read.row_bytes)
        .unwrap_or(u32::MAX)
        .clamp(1, pitch);
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: read.length as u64,
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: row,
        row_pitch: pitch,
        rows: u32::try_from(read.region[3]).unwrap_or(1).max(1),
        image_pitch: read.length as u32,
        images: 1,
        gaps: if row < pitch {
            Gaps::Compared
        } else {
            Gaps::Ignored
        },
    }
}

/// The bytes case.
pub fn cpu_access() -> CpuAccess {
    CpuAccess { name: CPU_ACCESS }
}

/// The case over the bytes a texture over a buffer reads and writes.
pub struct CpuAccess {
    name: &'static str,
}

impl Case for CpuAccess {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        shadow_allowances()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The buffer's bytes, and the geometry of every call the program makes, keyed by the call
        // and its numbers: either moving makes the recording's answer another program's.
        let mut steps = Vec::new();
        for row in &BYTES_ROWS {
            steps.extend(row.key.as_bytes());
            steps.extend(row.spec().format.to_le_bytes());
            steps.extend(row.offset.to_le_bytes());
            steps.extend(row.bytes_per_row.to_le_bytes());
            for read in row.reads {
                steps.extend(b"getBytes");
                for axis in read.region {
                    steps.extend(axis.to_le_bytes());
                }
                steps.extend(read.pitch.to_le_bytes());
                steps.extend(read.row_bytes.to_le_bytes());
                steps.extend((read.length as u64).to_le_bytes());
                // The byte the destination holds first is part of the answer wherever the pitch
                // leaves padding, so it is part of what the case asked.
                steps.push(read.before);
            }
            let (region, pitch, length) = row.write;
            steps.extend(b"replaceRegion");
            for axis in region {
                steps.extend(axis.to_le_bytes());
            }
            steps.extend(pitch.to_le_bytes());
            steps.extend(written(length, row.seed()));
        }
        Ok(vec![
            Input::bytes("buffer-bytes", &stripes(BUFFER as usize)),
            Input::bytes("calls", &steps),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = call(sites, "work", side.work(&bytes_work()), |_| Value::Absent)?;
        let mut at = 0;
        for row in &BYTES_ROWS {
            for (which, read) in row.reads.iter().enumerate() {
                let bytes = done.reads.get(at).cloned().unwrap_or_default();
                sites.push(Node::leaf(
                    format!("{}/get-bytes/{which}", row.key),
                    Observed::bytes(bytes, read_layout(read)),
                ));
                at += 1;
            }
            let read = done.reads.get(at).cloned().unwrap_or_default();
            sites.push(Node::leaf(
                format!("{}/buffer", row.key),
                Observed::bytes(read, buffer_layout(row)),
            ));
            at += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard refuses a pitch one byte below the texture's own row, and refuses nothing at or
    /// above it. No device is needed: the decision is the request's own shape.
    ///
    /// Its failure mode was exercised while writing it: changing the guard's comparison to `<=`
    /// makes the second half fail, naming the pitch it refused.
    #[test]
    fn the_guard_refuses_a_pitch_below_the_row_and_nothing_at_or_above_it() {
        let row = BELOW.row_bytes;
        for pitch in 0..row {
            let request = Request {
                bytes_per_row: pitch,
                ..BELOW
            };
            assert!(
                crate::buffer_texture::ends_metals_process(&request).is_some(),
                "a pitch of {pitch} under a row of {row} is not refused, so record mode would \
                 hand it to Metal's assertion"
            );
        }
        for pitch in [row, row + 1, row * 2, BUFFER] {
            let request = Request {
                bytes_per_row: pitch,
                ..BELOW
            };
            assert_eq!(
                crate::buffer_texture::ends_metals_process(&request),
                None,
                "a pitch of {pitch} at or above a row of {row} is refused, which would make the \
                 guard refuse shapes Metal takes"
            );
        }
    }

    /// Every read's layout describes exactly the bytes its case hands the call: the rows of its
    /// region, at its pitch, with the row's own bytes as the elements.
    ///
    /// A read states its row's bytes rather than deriving them from the format, so a wrong number
    /// here would leave part of the destination outside the comparison and every byte of it would
    /// pass unread. Its failure mode was exercised while writing it: a row stating 8 bytes instead
    /// of 12 fails here, naming the read and both counts.
    #[test]
    fn every_reads_layout_covers_the_bytes_it_states() {
        for row in &BYTES_ROWS {
            for (which, read) in row.reads.iter().enumerate() {
                let at = format!("{}/get-bytes/{which}", row.key);
                assert!(
                    read.row_bytes <= read.pitch,
                    "{at} reads rows of {} bytes at a pitch of {}, so one row's bytes reach into \
                     the next",
                    read.row_bytes,
                    read.pitch
                );
                let rows = read.region[3];
                let needed = (rows - 1) * read.pitch + read.row_bytes;
                assert_eq!(
                    read.length as u64, needed,
                    "{at} hands the call {} bytes, but {rows} rows of {} at a pitch of {} reach {}",
                    read.length, read.row_bytes, read.pitch, needed
                );
                assert_eq!(
                    read_layout(read).extent(),
                    read.length,
                    "{at}'s layout and its destination are different sizes, so the comparison \
                     covers the wrong bytes"
                );
            }
        }
    }

    /// One read lays its region into a destination whose rows lie further apart than the rows are
    /// wide, and that read's padding is compared, from a byte no stripe of the buffer holds.
    ///
    /// Without such a read the case cannot ask whether `getBytes:` leaves the bytes past each
    /// row's texels alone: a destination whose rows are its pitch has nowhere for the answer to
    /// show. Its failure mode was exercised while writing it: making [`read_layout`] answer
    /// [`Gaps::Ignored`] throughout fails here.
    #[test]
    fn a_read_at_a_wider_pitch_compares_the_padding_between_its_rows() {
        let padded: Vec<&Read> = BYTES_ROWS
            .iter()
            .flat_map(|row| row.reads.iter())
            .filter(|read| read.row_bytes < read.pitch && read.region[3] > 1)
            .collect();
        assert!(
            !padded.is_empty(),
            "no read lays its rows into a wider destination, so nothing asks whether getBytes: \
             leaves the padding between them alone"
        );
        for read in padded {
            assert_eq!(
                read_layout(read).gaps,
                Gaps::Compared,
                "a read at a pitch of {} with rows of {} leaves padding out of its comparison",
                read.pitch,
                read.row_bytes
            );
            assert!(
                !stripes(BUFFER as usize).contains(&read.before),
                "the destination starts at {:#x}, which the buffer holds somewhere, so a padding \
                 byte the call copied and one it left alone can read alike",
                read.before
            );
        }
        // A read at its region's own pitch has no padding, and compares none.
        let tight = BYTES_ROWS[0].reads.first().expect("a first read");
        assert_eq!(read_layout(tight).gaps, Gaps::Ignored);
    }

    /// Every shape of the recorded table has a row pitch at or above its texture's own row, so no
    /// row of it can reach Metal's assertion. The one row that cannot say that is the stride case,
    /// which record mode refuses outright.
    #[test]
    fn no_recorded_shape_would_end_metals_process() {
        let refused: Vec<String> = requests()
            .iter()
            .filter_map(|request| {
                crate::buffer_texture::ends_metals_process(request)
                    .map(|why| format!("{}: {why}", request.site()))
            })
            .collect();
        assert!(
            refused.is_empty(),
            "these shapes of a recorded table would end Metal's process; take them out:\n{}",
            refused.join("\n")
        );
        assert!(
            crate::buffer_texture::ends_metals_process(&BELOW).is_some(),
            "the stride case's own shape is not the shape the guard refuses"
        );
    }

    /// No two shapes of the table share a site, so every answer of a recording has one shape
    /// behind it.
    #[test]
    fn every_shape_has_a_site_of_its_own() {
        let mut sites: Vec<String> = requests().iter().map(Request::site).collect();
        let total = sites.len();
        sites.sort();
        sites.dedup();
        assert_eq!(sites.len(), total, "two shapes of the table share a site");
    }

    /// Every shape the bytes case lays over a buffer is also a row of the making table, whose
    /// device guard has shown the runtime answers it.
    ///
    /// Without this, the bytes case's safety would be a coincidence: its program makes its
    /// textures as resources, and a resource Metal stops would end the recorder's process before
    /// the first read. Sharing the shapes with the guarded table is what makes it safe by
    /// construction rather than by inspection.
    #[test]
    fn every_shape_the_bytes_case_lays_over_a_buffer_is_a_guarded_one() {
        let guarded = requests();
        for row in &BYTES_ROWS {
            let spec = row.spec();
            let found = guarded.iter().any(|request| {
                request.descriptor.format.as_raw() == spec.format
                    && request.descriptor.size[0] as u64 == row.width
                    && request.descriptor.size[1] as u64 == row.height
                    && request.offset == row.offset
                    && request.bytes_per_row == row.bytes_per_row
            });
            assert!(
                found,
                "{}: its shape is not a row of the making table, so nothing has shown the runtime \
                 answers it and record mode would hand it to Metal unguarded",
                row.key
            );
        }
    }

    /// The committed recording of the stride row is exactly what the case states.
    ///
    /// The file is written by hand because the shape cannot be run, which is precisely why it
    /// needs tying down: without this a hand-written file could say anything, and a change to the
    /// case's site or inputs would leave it stale in silence.
    #[test]
    fn the_stride_rows_recording_is_the_answer_its_case_states() {
        let case = stride_below_the_row();
        let stated = crate::recording::Recording::new(
            case.name(),
            Vec::new(),
            vec![case.stated_answer().unwrap()],
        );
        let committed = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        // Metal's answer alone: the cost and timing halves of the file are our own measurements on
        // a venue, taken by another leg, and a case states nothing about either.
        assert_eq!(
            committed.without_costs().without_times(),
            stated,
            "{}: its committed recording is not the answer the case states",
            case.name()
        );
    }
}

/// The safety guard this family rests on, run on a device.
///
/// Record mode hands every shape of the making table to Metal. A shape Metal **stops** ends its
/// process, and the runtime's model is what tells a stop from a nil: `Error::Invalid` is "Metal
/// stops this call", while `Error::Nil` is "plain Metal answers nil and the process goes on". So a
/// table that makes the runtime answer `Invalid` anywhere is a table that must never be recorded,
/// and this is the check that says so before anybody runs `conformance record`.
///
/// Its failure mode was exercised while writing it: adding a 2D row at an offset of 8 — which the
/// runtime refuses `Invalid(OffsetAlignment)` — fails here, naming the shape and the variant.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod device_tests {
    use super::*;
    use crate::device::{self, Notices, ScratchCache, Venues};
    use crate::runtime_side::RuntimeSide;
    use crate::transcript::Value;

    /// The venue the guard opens: a GPU where the tests are built for one, else a CPU ICD.
    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    /// The variant a failure names, where it names one: `Error::Invalid` and the rest.
    fn variant(value: &Value) -> String {
        match value {
            Value::Variant { name, payload } if name == "Err" => match payload.as_ref() {
                Value::Variant { name, .. } => name.clone(),
                other => format!("{other:?}"),
            },
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn no_shape_of_the_making_table_is_one_metal_stops() {
        let mut notices = Notices::default();
        let cache = ScratchCache::new().unwrap();
        let runtime = device::open(VENUE, &notices, &cache).unwrap();
        let mut side = RuntimeSide::new(
            &runtime,
            notices.open("no_shape_of_the_making_table_is_one_metal_stops"),
        );
        let mut stopped = Vec::new();
        for request in requests() {
            if let Err(failure) = side.buffer_texture(&request) {
                let named = variant(&failure.recorded());
                if named.contains("Invalid") {
                    stopped.push(format!("{}: {named}", request.site()));
                }
            }
        }
        assert!(
            stopped.is_empty(),
            "these shapes are ones the runtime says Metal stops, so record mode would end Metal's \
             process; take them out of the table:\n{}",
            stopped.join("\n")
        );
    }
}
