//! Metal's 4:2:2 chroma-pair textures (`GBGR422`, `BGRG422`): what a copy moves, and what a
//! kernel's `read()` and nearest sample of one answer.
//!
//! Ported from `runtime_lane/chroma_pairs.rs`. Both of these formats are **public** Metal pixel
//! formats, and every call the lane makes of them — `newTextureWithDescriptor:`,
//! `replaceRegion:`, `getBytes:`, the four blit copies and a kernel's `read()` — is a public
//! call. So nothing here is carried: record mode runs both families on Metal and the recordings
//! hold Metal's own answers, taken on this M3 Max.
//!
//! **What the port changes.** The lane's copy table was measured on an Apple M4 and written into
//! assertions. Its *texel* table is not a Metal run at all: `four_two_two_texels_read_as_metal_s`
//! compares our reads with `metal2vulkan::bind::ChromaDecode`'s model, so both sides of its
//! comparison are ours and nothing in the lane can tell a wrong model from a wrong
//! implementation.
//!
//! The model itself **was** measured, and this port must not imply otherwise: `bind/chroma_decode`
//! records a probe on an Apple M3 Max under macOS 27.0.1 that found every `float` and `half`
//! `read()`, nearest, linear and bicubic sample, and gather of both formats bit for bit the one of
//! an `RGBA8Unorm` texture holding the decoded codes, over 16 x 4 texels of random bytes. So
//! Metal's `read()` of these formats is a defined operation, not a guess, and the model's rule
//! stands on a real measurement.
//!
//! What that probe does not leave behind is a re-runnable artifact: it was thrown away, it covered
//! 64 texels rather than the lane's own 256, and nothing mechanical ties the lane's expectation to
//! it. These cases take Metal's answer over the lane's own pattern and commit it, so a new macOS
//! is one `conformance record` from saying whether Apple changed it.
//!
//! # The three families
//!
//! [`ChromaCopies`] is the lane's copy sweep as one declarative program ([`crate::work`]): a whole blit
//! in and out, the host's whole `replaceRegion:`/`getBytes:`, seven blits in of regions whose x
//! and width fall off the 2-texel pairs at buffer offsets and pitches of 34 and 30 bytes, six
//! blits out of such regions, four host copies in and four host reads out, three copies between
//! two textures of the format off the pairs, and four copies into a texture of `GBGR422`,
//! `BGRG422`, `RG8Unorm` and `R16Uint`. Each read-back's bytes are recorded whole under a layout
//! that names the row and the texel, so a diff says which texel of which sub-case moved. Every
//! byte compares exactly; the case's allowances are on the notices site alone, for the kinds the
//! runtime announces while it keeps a 4:2:2 texture's decoded twin beside the bytes it moves.
//!
//! [`ChromaTexels`] is the lane's texel table, over a 16 x 16 texture of the format holding the
//! lane's pattern, reached two ways through the owned fixture `fpv_swizzled_reads`:
//! `read()` at the 256 integer texel positions, and `sample()` at the 256 texel centres through a
//! nearest clamp-to-edge sampler. Both answer 256 `float4`s, recorded whole. The two are separate
//! cases because Metal defines them differently: a chroma-pair format is **filterable**, so
//! sampling is the operation its feature tables grant it and a nearest sample at a centre is
//! defined to be that texel, while whether Metal defines `read()` for these formats is what the
//! recording answers rather than assumes. Comparison is exact and each case's allowances are on the
//! notices site alone: the runtime reaches these texels through the texture's decoded twin and
//! announces `carried-decoded-twin`, and where the device converts the format itself the twin holds
//! the device's own conversion and it announces `decoded-texel-precision` too. Metal announces
//! neither, and those notices are what justify the notices site differing. Every texel still
//! compares bit for bit.
//!
//! # Why nothing here can fault a GPU
//!
//! Every program is a handful of 16 x 16 textures, blits of regions inside them, and one compute
//! dispatch of 256 threads, with at most one sampler state. The two descriptors the lane finds
//! Metal's **validation** aborts plain too — a mipmapped 4:2:2 texture, and a view of one in another format — are left out by
//! construction: the runtime spells that outcome `Error::Invalid`, "Metal's validation stops it,
//! ending the process", and the suite's standing rule is that no case may end Metal's process. So
//! is the render-target half of the lane, whose pipeline Metal's validation aborts plain too.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Bindings, Case, Failure, Input, MetalSource, SamplerSpec, Side, TextureRole,
    TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, TextureSpec as WorkTexture, Work};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan_runtime::Degradation;

/// The textures' side, as the lane makes them.
const SIDE: u64 = 16;

/// Bytes a row of a [`SIDE`]-wide 4:2:2 texture takes: two a texel.
const ROW: u64 = 2 * SIDE;

/// Bytes the whole of such a texture takes.
const WHOLE: u64 = ROW * SIDE;

/// The fixture whose kernel reads the texels, and its entry: one `float4` a position, from a
/// buffer of positions, so one kernel reads a texture of any size.
const READ_FIXTURE: &str = "fpv_swizzled_reads";
const READ_ENTRY: &str = "fpv_swz_read";

/// The same fixture's sampling kernel: one `float4` a position, from a buffer of `float2`
/// coordinates, through the sampler state bound at index 0.
const SAMPLE_ENTRY: &str = "fpv_swz_sample";

/// The threads in a threadgroup of the reading dispatch.
const GROUP: u32 = 64;

/// The byte the read-back buffer holds before the dispatch.
const BLANK: u8 = 0xab;

/// `n` bytes from `base`, byte `i` of them `base + i` (wrapping): the lane's `seq`.
fn seq(n: u64, base: u8) -> Vec<u8> {
    (0..n).map(|i| base.wrapping_add(i as u8)).collect()
}

/// The 512 bytes the lane fills a whole texture with.
fn pattern() -> Vec<u8> {
    seq(WHOLE, 0)
}

/// The byte a texture holds before a copy of a region, so the bytes outside the region are the
/// ones the copy must have left alone.
const BEFORE: u8 = 0xee;

/// The two formats, and the word a case's name carries for each.
const FORMATS: [(PixelFormat, &str); 2] = [
    (PixelFormat::GBGR422, "gbgr422"),
    (PixelFormat::BGRG422, "bgrg422"),
];

/// The case names, written out so each is a `&'static str` the report and `conformance replay`
/// can name.
const COPY_NAMES: [&str; 2] = ["chroma-pair-copies-gbgr422", "chroma-pair-copies-bgrg422"];
const READ_NAMES: [&str; 2] = [
    "chroma-pair-texel-reads-gbgr422",
    "chroma-pair-texel-reads-bgrg422",
];
const SAMPLE_NAMES: [&str; 2] = [
    "chroma-pair-texel-samples-gbgr422",
    "chroma-pair-texel-samples-bgrg422",
];

/// Every case of the family: the copy sweep, the texel reads and the texel samples, of each
/// format.
pub fn all() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for (at, (format, word)) in FORMATS.into_iter().enumerate() {
        cases.push(Box::new(ChromaCopies {
            name: COPY_NAMES[at],
            format,
            word,
        }));
    }
    for (at, (format, word)) in FORMATS.into_iter().enumerate() {
        for (name, access) in [
            (READ_NAMES[at], Access::Read),
            (SAMPLE_NAMES[at], Access::NearestSample),
        ] {
            cases.push(Box::new(ChromaTexels {
                name,
                format,
                word,
                access,
            }));
        }
    }
    cases
}

/// How a case reaches the texture's texels.
///
/// Both exist because Metal defines them differently. A chroma-pair texture is a **filterable**
/// format: `sample` is the operation Metal's own feature tables grant it, and a nearest sample at
/// a texel's centre is defined to be that texel. `read()` is what the runtime lane uses and what a
/// decoder does, and whether Metal defines it for these formats is exactly the question a
/// recording answers rather than assumes — so the two are separate cases, and the sampling one is
/// the control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    /// `t.read(position)` at every texel's integer position.
    Read,
    /// `t.sample(s, uv)` at every texel's centre, through a nearest clamp-to-edge sampler of
    /// normalized coordinates.
    NearestSample,
}

impl Access {
    /// The fixture's entry point this access runs.
    fn entry(self) -> &'static str {
        match self {
            Access::Read => READ_ENTRY,
            Access::NearestSample => SAMPLE_ENTRY,
        }
    }

    /// The word the case's site carries.
    fn word(self) -> &'static str {
        match self {
            Access::Read => "reads",
            Access::NearestSample => "samples",
        }
    }

    /// The positions the kernel reads: integer texel positions, or the texel centres as
    /// normalized coordinates.
    fn positions(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TEXELS * 8);
        for y in 0..SIDE as u32 {
            for x in 0..SIDE as u32 {
                match self {
                    Access::Read => {
                        bytes.extend(x.to_le_bytes());
                        bytes.extend(y.to_le_bytes());
                    }
                    Access::NearestSample => {
                        let centre = |index: u32| (index as f32 + 0.5) / SIDE as f32;
                        bytes.extend(centre(x).to_le_bytes());
                        bytes.extend(centre(y).to_le_bytes());
                    }
                }
            }
        }
        bytes
    }

    /// The sampler state the access binds, where it binds one.
    fn sampler(self) -> Option<SamplerSpec> {
        match self {
            Access::Read => None,
            Access::NearestSample => Some(
                SamplerSpec::filtered(SamplerMinMagFilter::Nearest)
                    .addressed(SamplerAddressMode::ClampToEdge),
            ),
        }
    }
}

/// One blit of a region of a buffer into a texture, as the lane sweeps them: the texel the region
/// starts at, its texels and rows, the buffer offset the bytes come from, and the row pitch.
const BLITS_IN: [(u64, u64, u64, u64, u64); 7] = [
    (1, 2, 1, 0, ROW),
    (2, 3, 1, 0, ROW),
    (3, 1, 1, 0, ROW),
    (14, 2, 2, 0, ROW),
    (0, SIDE, 2, 2, ROW),
    (0, SIDE, 4, 0, 34),
    (0, SIDE, 2, 0, 30),
];

/// One blit of a region of a texture out into a buffer: the texel it starts at, its texels, the
/// buffer offset it lands at, and the row pitch. Every one is two rows tall.
const BLITS_OUT: [(u64, u64, u64, u64); 6] = [
    (1, 2, 0, ROW),
    (2, 3, 0, ROW),
    (3, 1, 0, ROW),
    (0, SIDE, 2, ROW),
    (0, SIDE, 0, 34),
    (2, 2, 0, 4),
];

/// One host `replaceRegion:` of a region: the texel it starts at, its texels and rows, and the
/// row pitch the host's bytes are at.
const HOST_PUTS: [(u64, u64, u64, u64); 4] =
    [(1, 2, 1, 4), (2, 3, 1, 6), (3, 1, 2, 2), (0, SIDE, 2, 34)];

/// One host `getBytes:` of a region, two rows tall: the texel it starts at, its texels, and the
/// row pitch it writes at.
const HOST_GETS: [(u64, u64, u64); 4] = [(1, 2, 4), (2, 3, 6), (3, 1, 2), (0, SIDE, 34)];

/// One copy between two textures of the format: the source texel, the destination texel, and the
/// texels copied. Every one is two rows tall.
const TEXTURE_COPIES: [(u64, u64, u64); 3] = [(1, 4, 2), (0, 3, 2), (2, 6, 3)];

/// The bytes a blit out writes into, so the bytes the copy did not reach are visible.
const OUT_BUFFER: u64 = 1024;

/// The bytes a host read writes into.
const GET_BUFFER: usize = 128;

/// The copy sweep of one format.
pub struct ChromaCopies {
    name: &'static str,
    format: PixelFormat,
    /// The format's word, which the sub-case sites carry.
    word: &'static str,
}

impl ChromaCopies {
    /// A texture of the case's format, [`SIDE`] square, read by shaders.
    fn texture(&self, private: bool) -> Resource {
        let spec = WorkTexture::texture_2d(self.format.as_raw(), SIDE, SIDE);
        Resource::Texture {
            spec: if private { spec } else { spec.shared() },
        }
    }

    /// The program the case runs, and the name of each read it asks for, in order.
    ///
    /// The two are built together because a read's meaning is its place in the program: naming
    /// them apart is how a port drifts into comparing the wrong row.
    fn program(&self) -> (Work, Vec<ReadSite>) {
        let mut work = Work::new();
        let mut reads = Vec::new();
        let pattern = pattern();
        let before = vec![BEFORE; WHOLE as usize];
        let source = seq(4096, 0x40);
        let put = seq(4096, 0xa0);
        let b_pattern = work.buffer(pattern.clone());
        let b_before = work.buffer(before.clone());
        let b_source = work.buffer(source.clone());
        let whole = Rect::new(0, 0, SIDE, SIDE);

        // A whole blit in and a whole blit out: the texels move unchanged.
        let private = work.add(self.texture(true));
        let b_whole_out = work.buffer(before.clone());
        work.step(buffer_to_texture(
            b_pattern,
            0,
            ROW,
            whole,
            private,
            [0, 0, 0],
        ));
        work.step(texture_to_buffer(private, whole, b_whole_out, 0, ROW));
        work.step(Step::Commit);
        reads.push(self.read(&mut work, "blit-whole", b_whole_out, WHOLE));

        // The host's whole copy in and read out.
        let shared = work.add(self.texture(false));
        work.step(replace(shared, whole, pattern.clone(), ROW));
        reads.push(self.get_bytes(&mut work, "host-whole", shared, whole, ROW, WHOLE as usize));

        // Blits in of regions off the texel pairs, at buffer offsets and row pitches of their
        // own. Each gets a texture of its own, as the lane makes one, so the bytes outside the
        // region are the ones the fill left.
        for (x, texels, rows, offset, pitch) in BLITS_IN {
            let target = work.add(self.texture(true));
            let out = work.buffer(before.clone());
            work.step(buffer_to_texture(
                b_before,
                0,
                ROW,
                whole,
                target,
                [0, 0, 0],
            ));
            work.step(buffer_to_texture(
                b_source,
                offset,
                pitch,
                Rect::new(0, 0, texels, rows),
                target,
                [x, 0, 0],
            ));
            work.step(texture_to_buffer(target, whole, out, 0, ROW));
            work.step(Step::Commit);
            let site =
                format!("blit-in/x-{x}/texels-{texels}/rows-{rows}/offset-{offset}/pitch-{pitch}");
            reads.push(self.read(&mut work, &site, out, WHOLE));
        }

        // Blits out of such regions, into a buffer of `BEFORE`, so the bytes the copy did not
        // reach show as themselves.
        for (x, texels, offset, pitch) in BLITS_OUT {
            let out = work.buffer(vec![BEFORE; OUT_BUFFER as usize]);
            work.step(Step::CopyTextureToBuffer {
                texture: private,
                slice: 0,
                level: 0,
                origin: [x, 0, 0],
                size: [texels, 2, 1],
                buffer: out,
                offset,
                bytes_per_row: pitch,
                bytes_per_image: pitch * 2,
                plane: Plane::Whole,
            });
            work.step(Step::Commit);
            let site = format!("blit-out/x-{x}/texels-{texels}/offset-{offset}/pitch-{pitch}");
            reads.push(self.read(&mut work, &site, out, OUT_BUFFER));
        }

        // The host's copies of such regions, each over a texture reset to `BEFORE` first.
        for (x, texels, rows, pitch) in HOST_PUTS {
            work.step(replace(shared, whole, before.clone(), ROW));
            work.step(replace(
                shared,
                Rect::new(x, 0, texels, rows),
                put.clone(),
                pitch,
            ));
            let site = format!("host-put/x-{x}/texels-{texels}/rows-{rows}/pitch-{pitch}");
            reads.push(self.get_bytes(&mut work, &site, shared, whole, ROW, WHOLE as usize));
        }

        // The host's reads of such regions, out of the texture holding the pattern.
        work.step(replace(shared, whole, pattern.clone(), ROW));
        for (x, texels, pitch) in HOST_GETS {
            let site = format!("host-get/x-{x}/texels-{texels}/pitch-{pitch}");
            reads.push(self.get_bytes(
                &mut work,
                &site,
                shared,
                Rect::new(x, 0, texels, 2),
                pitch,
                GET_BUFFER,
            ));
        }

        // Copies between two textures of the format, off the pairs on both sides.
        for (from, to, texels) in TEXTURE_COPIES {
            let target = work.add(self.texture(true));
            let out = work.buffer(before.clone());
            work.step(buffer_to_texture(
                b_before,
                0,
                ROW,
                whole,
                target,
                [0, 0, 0],
            ));
            work.step(Step::CopyTextureToTexture {
                from: private,
                from_slice: 0,
                from_level: 0,
                from_origin: [from, 0, 0],
                size: [texels, 2, 1],
                to: target,
                to_slice: 0,
                to_level: 0,
                to_origin: [to, 0, 0],
            });
            work.step(texture_to_buffer(target, whole, out, 0, ROW));
            work.step(Step::Commit);
            let site = format!("texture-copy/{from}-to-{to}/texels-{texels}");
            reads.push(self.read(&mut work, &site, out, WHOLE));
        }

        // The whole texture copied into a texture of each format whose texel is the same two
        // bytes: the other chroma pair, `RG8Unorm` and `R16Uint`.
        for (into, named) in [
            (PixelFormat::GBGR422, "gbgr422"),
            (PixelFormat::BGRG422, "bgrg422"),
            (PixelFormat::RG8Unorm, "rg8unorm"),
            (PixelFormat::R16Uint, "r16uint"),
        ] {
            let target = work.add(Resource::Texture {
                spec: WorkTexture::texture_2d(into.as_raw(), SIDE, SIDE),
            });
            let out = work.buffer(before.clone());
            work.step(Step::CopyTextureToTexture {
                from: private,
                from_slice: 0,
                from_level: 0,
                from_origin: [0, 0, 0],
                size: [SIDE, SIDE, 1],
                to: target,
                to_slice: 0,
                to_level: 0,
                to_origin: [0, 0, 0],
            });
            work.step(texture_to_buffer(target, whole, out, 0, ROW));
            work.step(Step::Commit);
            reads.push(self.read(&mut work, &format!("into/{named}"), out, WHOLE));
        }
        (work, reads)
    }

    /// Add a read of `buffer`'s first `length` bytes, named `site`, and answer what the read is.
    fn read(&self, work: &mut Work, site: &str, buffer: usize, length: u64) -> ReadSite {
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: length as usize,
        });
        ReadSite {
            name: format!("{}/{site}", self.word),
            layout: buffer_layout(length),
        }
    }

    /// Add a host read of `texture`'s `region` at `pitch` bytes a row into `length` bytes, and
    /// answer what the read is.
    fn get_bytes(
        &self,
        work: &mut Work,
        site: &str,
        texture: usize,
        region: Rect,
        pitch: u64,
        length: usize,
    ) -> ReadSite {
        work.step(Step::GetBytes {
            texture,
            level: 0,
            slice: 0,
            region,
            bytes_per_row: pitch,
            // Metal's host calls take 0 for a 2D texture's image pitch, which is what the lane
            // passes; a 2D region has one image, so there is nothing for it to say.
            bytes_per_image: 0,
            length,
            // The read starts from zero, as this case's reads always did: it asks
            // nothing about the padding between its rows.
            before: 0,
        });
        ReadSite {
            name: format!("{}/{site}", self.word),
            layout: buffer_layout(length as u64),
        }
    }
}

/// One read the program asks for: the site it is emitted at, and how its bytes are laid out.
struct ReadSite {
    name: String,
    layout: ByteLayout,
}

/// A buffer's `length` bytes, read as the 16-bit words a 4:2:2 texel is, so a diff names the word
/// rather than a byte offset. One row, because a read-back buffer's bytes are one run: the row
/// structure a copy wrote into them is what the case is asking about, and flattening it is what
/// makes a diff able to say "word 37 moved" whatever pitch wrote it.
fn buffer_layout(length: u64) -> ByteLayout {
    ByteLayout::buffer(0, length, Element::U16)
}

/// `replaceRegion:` of `region` from `bytes` at `pitch` bytes a row.
fn replace(texture: usize, region: Rect, bytes: Vec<u8>, pitch: u64) -> Step {
    Step::ReplaceRegion {
        texture,
        level: 0,
        slice: 0,
        region,
        bytes,
        // An uncompressed format's rows are its texel rows.
        rows: region.size[1],
        bytes_per_row: pitch,
        // 0, as Metal's host call takes for a 2D texture and as the lane passes.
        bytes_per_image: 0,
    }
}

/// A blit of `buffer` from `offset` at `pitch` bytes a row into `texture`'s `region` at `origin`.
fn buffer_to_texture(
    buffer: usize,
    offset: u64,
    pitch: u64,
    region: Rect,
    texture: usize,
    origin: [u64; 3],
) -> Step {
    Step::CopyBufferToTexture {
        buffer,
        offset,
        bytes_per_row: pitch,
        bytes_per_image: pitch * region.size[1],
        size: region.size,
        texture,
        slice: 0,
        level: 0,
        origin,
        plane: Plane::Whole,
    }
}

/// A blit of `texture`'s `region` out into `buffer` at `offset`, `pitch` bytes a row.
fn texture_to_buffer(texture: usize, region: Rect, buffer: usize, offset: u64, pitch: u64) -> Step {
    Step::CopyTextureToBuffer {
        texture,
        slice: 0,
        level: 0,
        origin: region.origin,
        size: region.size,
        buffer,
        offset,
        bytes_per_row: pitch,
        bytes_per_image: pitch * region.size[1],
        plane: Plane::Whole,
    }
}

impl Case for ChromaCopies {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every byte of every read-back compares exactly; what differs is the notices site, which
        // Metal records empty. A device whose own driver does not decode 4:2:2 texels as Metal
        // does reaches them through the texture's decoded twin, decodes a host copy's rows on the
        // CPU where it does not run a pass for them, and stages a copy between two such textures
        // -- three kinds, each announced once per device. Each is named here, so the difference is
        // justified only while that notice really fires; on a device that needs none of them, none
        // of the three fires and the report counts them as unused.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // both copy rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        //
        // Where the device converts the format itself with Metal's reconstruction, the twin holds
        // the device's own Y′CbCr conversion of the pairs, its native answer (AGENTS.md rule 1),
        // and the runtime announces `decoded-texel-precision` where it makes such a texture. A
        // copy moves the carrier's bytes, never the twin's, so the notice loosens nothing here:
        // it sits at the notices site alone. Measured on the M3 Max (MoltenVK 1.4.2, which samples
        // through Metal's own `GBGR422` and `BGRG422` texture unit): it fired for both copy rows,
        // and each was divergent on that field and nothing else. Where the twin holds Metal's
        // decode it does not fire, which the report counts.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                [
                    Degradation::CarriedDecodedTwin,
                    Degradation::CpuBlockDecode,
                    Degradation::StagedTextureCopy,
                    Degradation::DecodedTexelPrecision,
                ]
                .into_iter()
                .map(|kind| notice_allowance(kind.code())),
            )
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The tables and the bytes they move, named one by one, so a diff of a failed input
        // fingerprint says which of them changed rather than "the program".
        let mut tables = String::new();
        for (x, texels, rows, offset, pitch) in BLITS_IN {
            tables.push_str(&format!("in {x} {texels} {rows} {offset} {pitch}\n"));
        }
        for (x, texels, offset, pitch) in BLITS_OUT {
            tables.push_str(&format!("out {x} {texels} {offset} {pitch}\n"));
        }
        for (x, texels, rows, pitch) in HOST_PUTS {
            tables.push_str(&format!("put {x} {texels} {rows} {pitch}\n"));
        }
        for (x, texels, pitch) in HOST_GETS {
            tables.push_str(&format!("get {x} {texels} {pitch}\n"));
        }
        for (from, to, texels) in TEXTURE_COPIES {
            tables.push_str(&format!("copy {from} {to} {texels}\n"));
        }
        // The reads' names, in the order the program asks for them: a read that moves is a read
        // compared against another row's answer, which the fingerprint has to catch.
        let (_, reads) = self.program();
        let mut names = String::new();
        for read in &reads {
            names.push_str(&read.name);
            names.push('\n');
        }
        Ok(vec![
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("tables", tables.as_bytes()),
            Input::bytes("texture:pattern", &pattern()),
            Input::bytes("buffer:source", &seq(4096, 0x40)),
            Input::bytes("buffer:put", &seq(4096, 0xa0)),
            Input::bytes("before", &[BEFORE]),
            Input::bytes("reads", names.as_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let (work, reads) = self.program();
        let done = made(sites, "work", side.work(&work))?;
        if done.reads.len() != reads.len() {
            return Err(Abort {
                at: "work".to_owned(),
                failure: Failure::named(
                    "ReadCount",
                    format!("{} reads of {}", done.reads.len(), reads.len()),
                ),
            });
        }
        sites.value(
            "statuses",
            Value::List(
                done.statuses
                    .iter()
                    .map(|status| Value::unit_variant(status.clone()))
                    .collect(),
            ),
        );
        let mut output = Node::branch("output", Vec::new());
        for (at, read) in reads.into_iter().enumerate() {
            let bytes = done.reads[at].clone();
            let digest = Digest::sha256(&bytes);
            output = output.with(Node::leaf(
                read.name,
                Observed::bytes(bytes, read.layout).digested(digest),
            ));
        }
        sites.push(output);
        Ok(())
    }
}

/// Every texel of one format, reached one way: `read()` of its position, or a nearest sample at
/// its centre.
pub struct ChromaTexels {
    name: &'static str,
    format: PixelFormat,
    word: &'static str,
    access: Access,
}

/// The texels a [`SIDE`]-square texture holds.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The layout a read's `float4`s are compared under: one row a texel of four lanes, so a diff
/// names the texel and the channel.
fn texel_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: (TEXELS * 16) as u64,
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: TEXELS as u32,
        image_pitch: (TEXELS * 16) as u32,
        images: 1,
        // A texel's four lanes fill its row, so there is nothing between them.
        gaps: Gaps::Ignored,
    }
}

impl ChromaTexels {
    /// The fixture this case runs.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: READ_FIXTURE,
            entry: self.access.entry(),
        }
    }
}

impl Case for ChromaTexels {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{READ_FIXTURE}{extension}"),
                entry: self.access.entry().to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn allowances(&self) -> Vec<Allowance> {
        // On a device whose own driver does not decode these texels as Metal does, the runtime
        // reads them through the texture's decoded twin and announces it once. Metal announces
        // nothing, so the notices site differs and that notice is what justifies it; every texel
        // still compares exactly. On a device that needs no twin the allowance simply does not
        // fire, which the report counts.
        //
        // The host copy that fills the texture (`replaceRegion`) is decoded on the CPU into that
        // twin, which announces `cpu-block-decode` -- the same kind, for the same reason, that
        // this file's blit case names. It is a declaration and not a tolerance: every texel the
        // case reads still compares exactly.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 4 texel rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        //
        // Where the device converts the format itself with Metal's reconstruction, the twin holds
        // the device's own Y′CbCr conversion of the pairs, its native answer (AGENTS.md rule 1):
        // each chroma channel the nearest 8-bit code to the device's 3:1 sum of the pair's code
        // and the code beside it, ties to even, where Metal's `(3 c + n + 2) >> 2` rounds them up.
        // The runtime announces `decoded-texel-precision` where it makes such a texture. It is
        // declared at the notices site alone and loosens no texel. Measured on the M3 Max
        // (MoltenVK 1.4.2, which samples through Metal's own `GBGR422` and `BGRG422` texture
        // unit): it fired for all 4 texel rows, and each was divergent on that field and nothing
        // else, every texel Metal's bit for bit. The lane's pattern holds no tie to round: a
        // pair's chroma code and the one beside it are 4 apart (mod 256), or equal at a row's
        // clamped edge, so every `3 c + n` is a multiple of 4. Where the twin holds Metal's decode
        // it does not fire, which the report counts.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                [
                    Degradation::CarriedDecodedTwin,
                    Degradation::CpuBlockDecode,
                    Degradation::DecodedTexelPrecision,
                ]
                .into_iter()
                .map(|kind| notice_allowance(kind.code())),
            )
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("entry", fixture.entry.as_bytes()),
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("texture:bytes", &pattern()),
            Input::bytes("positions", &self.access.positions()),
            Input::bytes("buffer:blank", &[BLANK]),
        ];
        if let Some(sampler) = self.access.sampler() {
            inputs.push(Input::bytes("sampler", &sampler.words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = self.fixture();
        let source = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&source.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, self.access.entry()),
        )?;
        // The texture is made with its bytes, which is the whole of what the case feeds it: the
        // lane's own fresh-texture row (a new texture reads opaque black) is a separate question
        // from what Metal's decode does with known codes, and this case asks the second.
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(
                &TextureSpec::new(
                    self.format,
                    [SIDE as u32, SIDE as u32],
                    ROW as u32,
                    // Shader-read usage alone, as the lane's read test makes it: these formats are
                    // sampled and read, and asking for write usage as well would be asking a
                    // different question of Metal.
                    TextureRole::Read,
                ),
                &pattern(),
            ),
        )?;
        let positions = made(sites, "positions", side.buffer(&self.access.positions()))?;
        let output = made(sites, "output", side.buffer(&vec![BLANK; TEXELS * 16]))?;
        let mut samplers = Vec::new();
        if let Some(spec) = self.access.sampler() {
            samplers.push(made(sites, "sampler", side.sampler(&spec))?);
        }
        let textures = [texture];
        let buffers = [output, positions];
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers).sampled(&samplers),
                &[],
                [TEXELS as u32, 1, 1],
                [GROUP, 1, 1],
            ),
        )?;
        let status = made(sites, "status", side.wait(command_buffer))?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let bytes = made(sites, "read-back", side.read_buffer(output, 0, TEXELS * 16))?;
        if bytes.len() != TEXELS * 16 {
            return Err(Abort {
                at: "read-back".to_owned(),
                failure: Failure::named(
                    "ShortReadBack",
                    format!("{} bytes of {}", bytes.len(), TEXELS * 16),
                ),
            });
        }
        let digest = Digest::sha256(&bytes);
        sites.push(Node::leaf(
            format!("texels/{}/{}", self.word, self.access.word()),
            Observed::bytes(bytes, texel_layout()).digested(digest),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Six cases, each of a different question, and each name says which format it is of.
    #[test]
    fn the_cases_name_their_format() {
        let cases = all();
        assert_eq!(cases.len(), 6);
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name()).collect();
        assert_eq!(names.len(), 6);
        for name in names {
            assert!(
                name.ends_with("gbgr422") || name.ends_with("bgrg422"),
                "{name}"
            );
        }
    }

    /// The copy program asks for one read a sub-case of the lane's tables, and every read the
    /// program asks for has a site: a read with no name would compare against the wrong row.
    #[test]
    fn every_read_of_the_copy_program_has_a_site() {
        for (format, word) in FORMATS {
            let case = ChromaCopies {
                name: "unused",
                format,
                word,
            };
            let (work, reads) = case.program();
            assert_eq!(work.reads(), reads.len(), "{word}");
            // The lane's rows: two whole copies, seven blits in, six out, four host puts, four
            // host gets, three texture copies and four copies into another format.
            assert_eq!(reads.len(), 2 + 7 + 6 + 4 + 4 + 3 + 4);
            let names: BTreeSet<&str> = reads.iter().map(|read| read.name.as_str()).collect();
            assert_eq!(names.len(), reads.len(), "{word}: two reads share a site");
            for read in &reads {
                assert!(read.name.starts_with(word), "{}", read.name);
                assert!(read.layout.extent() > 0, "{}", read.name);
            }
        }
    }

    /// The program fills every texture whose region a copy writes before the copy, so the bytes
    /// outside the region are the ones the fill left rather than whatever the allocator held.
    #[test]
    fn every_region_copy_runs_over_a_known_texture() {
        let case = ChromaCopies {
            name: "unused",
            format: PixelFormat::GBGR422,
            word: "gbgr422",
        };
        let (work, _) = case.program();
        let mut primed: BTreeSet<usize> = BTreeSet::new();
        for step in &work.steps {
            match step {
                Step::CopyBufferToTexture {
                    texture,
                    size,
                    origin,
                    ..
                } => {
                    let whole = *size == [SIDE, SIDE, 1] && *origin == [0, 0, 0];
                    if whole {
                        primed.insert(*texture);
                    } else {
                        assert!(primed.contains(texture), "texture {texture} was not primed");
                    }
                }
                Step::CopyTextureToTexture {
                    to,
                    to_origin,
                    size,
                    ..
                } => {
                    let whole = *size == [SIDE, SIDE, 1] && *to_origin == [0, 0, 0];
                    assert!(whole || primed.contains(to), "texture {to} was not primed");
                    primed.insert(*to);
                }
                Step::ReplaceRegion {
                    texture, region, ..
                } => {
                    if region.size == [SIDE, SIDE, 1] && region.origin == [0, 0, 0] {
                        primed.insert(*texture);
                    } else {
                        assert!(primed.contains(texture), "texture {texture} was not primed");
                    }
                }
                _ => {}
            }
        }
    }

    /// Both accesses walk every texel once, row by row, which is what makes result `i` texel `i`:
    /// `read()` at the texel's integer position, and the nearest sample at its centre.
    #[test]
    fn both_accesses_walk_every_texel_in_order() {
        let read = Access::Read.positions();
        let sampled = Access::NearestSample.positions();
        assert_eq!(read.len(), TEXELS * 8);
        assert_eq!(sampled.len(), TEXELS * 8);
        let word = |bytes: &[u8], at: usize| {
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
        };
        for texel in 0..TEXELS {
            let (x, y) = (
                (texel % SIDE as usize) as u32,
                (texel / SIDE as usize) as u32,
            );
            assert_eq!(word(&read, texel * 8), x, "texel {texel} x");
            assert_eq!(word(&read, texel * 8 + 4), y, "texel {texel} y");
            let centre = |index: u32| (index as f32 + 0.5) / SIDE as f32;
            assert_eq!(
                f32::from_bits(word(&sampled, texel * 8)),
                centre(x),
                "texel {texel} u"
            );
            assert_eq!(
                f32::from_bits(word(&sampled, texel * 8 + 4)),
                centre(y),
                "texel {texel} v"
            );
        }
        // The sampling access binds a nearest clamp-to-edge sampler; the reading one binds none,
        // because `read()` takes no sampler state.
        assert_eq!(Access::Read.sampler(), None);
        let spec = Access::NearestSample.sampler().expect("a sampler state");
        assert_eq!(spec.min_filter, SamplerMinMagFilter::Nearest);
        assert_eq!(spec.s_address_mode, SamplerAddressMode::ClampToEdge);
        assert!(spec.normalized_coordinates);
    }

    /// The layout covers every texel and names the texel as its row, so a diff over 256 texels
    /// says which one moved.
    #[test]
    fn the_texel_layout_names_the_texel() {
        let layout = texel_layout();
        assert_eq!(layout.extent(), TEXELS * 16);
        assert_eq!(layout.at(0, 255, 3), 255 * 16 + 12);
    }
}
