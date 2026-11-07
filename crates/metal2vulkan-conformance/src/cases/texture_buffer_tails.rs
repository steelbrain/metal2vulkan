//! A texture buffer over a buffer whose texels run past the buffer's length: what each path reads
//! and writes there.
//!
//! Plain Metal makes such a texture buffer (`newTextureWithDescriptor:offset:bytesPerRow:` of a
//! `MTLTextureTypeTextureBuffer` descriptor) wherever its texels end inside the buffer's
//! `allocatedSize`, and takes the texels past the length as one memory every path shares, as it
//! does a 2D texture's ([`super::buffer_tails`]): a kernel's `read()` and `write()`, a blit from
//! and into it, `getBytes:` and `replaceRegion:` each read what the others wrote
//! (`probes/texture_buffer_past_length.m`, an Apple M3 Max under macOS 27.0.1). Over a 4096-byte
//! buffer, whose `allocatedSize` is its length, that memory is the buffer allocated next, which no
//! case can state; over a 20000-byte buffer, whose `allocatedSize` is 32768, it is the rest of the
//! buffer's own allocation, and it reads zero. The runtime keeps those texels in a tail of the
//! texture buffer's own, zero when made, which its functions view. So the case asks the second
//! shape: 1024 `R32Uint` texels from byte 16384 of a 20000-byte buffer, the first 904 inside the
//! length and the last 120 past it.
//!
//! `buffer-tails-texture-buffer` reads texels through `ob1_texel_buffer` (the owned fixture whose
//! thread `t` of eight reads texel `idx[t]` of its first texture buffer and writes `0xA0000000 |
//! t` to texel `idx[8 + t]` of its second) and the whole texture buffer through `getBytes:`, as
//! made, after the kernel wrote texels 900 to 907 (across the length) and after `replaceRegion:`
//! wrote every texel, with the buffer's 20000 bytes after each write; and, in a program of blits,
//! reads it into a buffer as made, writes texels 880 to 943 by a blit from a buffer, and reads it
//! back by a blit and by `getBytes:`, with the buffer's bytes.
//!
//! Every byte compares exactly. What differs is the announcing: the runtime's functions view such
//! a texture buffer's tail as a staging buffer and it announces `staged-texture-buffer`, which
//! Metal never does, and a device without unified memory announces `shared-in-system-memory` for
//! the `Shared` buffers the answers are read back through. Nothing here can fault a GPU: one
//! small texture buffer, eight threads, no loop.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::TextureRequest;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{
    Plane, Rect, Resource, Step, Storage as WorkStorage, TextureKind, TextureSpec, Work,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The case's name, as the report and the recordings spell it.
pub const TEXTURE_BUFFER: &str = "buffer-tails-texture-buffer";

/// The buffer's length: under its `allocatedSize` on the oracle (32768), so the texels past it lie
/// in the buffer's own allocation, where Metal reads zero.
const LEN: u64 = 20000;
/// The texture buffer's first byte in the buffer, a multiple of 16.
const OFFSET: u64 = 16384;
/// Its texels, 4 bytes each: they end at byte 20480, 480 bytes past the length.
const TEXELS: u64 = 1024;
/// Its bytes.
const BYTES: u64 = 4 * TEXELS;
const FORMAT: PixelFormat = PixelFormat::R32Uint;

/// The fixture whose kernel reads and writes texture buffers: our own MSL.
const FIXTURE: &str = "ob1_texel_buffer";

/// The byte a buffer the blits read into holds before they run, which no write here makes.
const BEFORE: u8 = 0xEE;

/// The buffer's bytes before anything runs: byte `i` holds `i % 251 + 1`, never zero.
fn pattern(n: u64) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8 + 1).collect()
}

/// The bytes `replaceRegion:` and the blits write: never zero, and unlike [`pattern`]'s.
fn written(n: u64, set: u64) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 13 + set * 101) % 253) as u8 + 2)
        .collect()
}

/// The texels each dispatch reads, as made, after the kernel's write and after the host's, around
/// the length (texel 904 is the first past it) and at both ends.
const MADE_READS: [u32; 8] = [0, 903, 904, 905, 950, 1023, 500, 1];
const WRITES: [u32; 8] = [900, 901, 902, 903, 904, 905, 906, 907];
const REPLACED_READS: [u32; 8] = [0, 903, 904, 1023, 2, 3, 950, 999];

/// Rows of `row` bytes over `length` bytes from byte 0 of a read, as a site's layout.
fn rows(length: u64, row: u64, element: Element) -> ByteLayout {
    let pitch = u32::try_from(row).expect("a row in a u32");
    ByteLayout {
        region: Region::Buffer { offset: 0, length },
        element,
        row_offset: 0,
        row_elements: pitch / element.size(),
        row_pitch: pitch,
        rows: u32::try_from(length / row).expect("the rows in a u32"),
        image_pitch: u32::try_from(length).expect("the length in a u32"),
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The texture buffer's texels as a read gives them: rows of 16, so a difference names the texel.
fn texel_layout() -> ByteLayout {
    rows(BYTES, 64, Element::U32)
}

/// The buffer's bytes, in rows of 32 (which divide its length).
fn buffer_layout() -> ByteLayout {
    rows(LEN, 32, Element::U8)
}

/// The usage every texture buffer here takes: read and written by shaders.
const READ_WRITE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);

/// The case.
pub struct TextureBufferTails;

impl TextureBufferTails {
    fn fixture() -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// A texture buffer of [`TEXELS`] texels over `buffer` from `offset`.
    fn over(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        (buffer, offset): (Handle, u64),
    ) -> Result<Handle, Abort> {
        let request = TextureRequest {
            usage: READ_WRITE,
            ..TextureRequest::texel_buffer(FORMAT.as_raw(), TEXELS as u32, buffer, offset, BYTES)
        };
        made(sites, at, side.texture(&request))
    }

    /// One dispatch of the fixture: texels `reads` of `from` read into the site `at`,
    /// `0xA0000000 | t` written to texel `writes[t]` of `into`.
    fn dispatch(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        pipeline: Handle,
        (from, reads): (Handle, [u32; 8]),
        (into, writes): (Handle, [u32; 8]),
    ) -> Result<(), Abort> {
        let idx: Vec<u8> = reads
            .iter()
            .chain(&writes)
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let idx = made(sites, &format!("{at}/idx"), side.buffer(&idx))?;
        let out = made(sites, &format!("{at}/out"), side.zeroed_buffer(32))?;
        let (textures, buffers) = ([from, into], [idx, out]);
        let dispatched = side.dispatch_threads(
            pipeline,
            &Bindings::new(&textures, &buffers),
            &[],
            [8, 1, 1],
            [8, 1, 1],
        );
        let dispatched = made(sites, &format!("{at}/dispatch"), dispatched)?;
        call(
            sites,
            &format!("{at}/status"),
            side.wait(dispatched),
            Clone::clone,
        )?;
        let bytes = made(
            sites,
            &format!("{at}/read-back"),
            side.read_buffer(out, 0, 32),
        )?;
        sites.push(Node::leaf(
            at,
            Observed::bytes(bytes, rows(32, 32, Element::U32)),
        ));
        Ok(())
    }

    /// The texture buffer's bytes as `getBytes:` reads them, at `at`.
    fn get_bytes(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        texture: Handle,
    ) -> Result<(), Abort> {
        let bytes = made(sites, &format!("{at}/call"), {
            side.read_texture(texture, [TEXELS as u32, 1], BYTES as u32)
        })?;
        sites.push(Node::leaf(at, Observed::bytes(bytes, texel_layout())));
        Ok(())
    }

    /// The buffer's bytes through its `contents`, at `at`.
    fn buffer(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        buffer: Handle,
    ) -> Result<(), Abort> {
        let bytes = made(
            sites,
            &format!("{at}/call"),
            side.read_buffer(buffer, 0, LEN as usize),
        )?;
        sites.push(Node::leaf(at, Observed::bytes(bytes, buffer_layout())));
        Ok(())
    }

    /// The program of blits: the texture buffer read into a buffer as made, texels 880 to 943
    /// written from a buffer, and every texel read back by a blit and by `getBytes:`, with the
    /// buffer's bytes.
    fn blits() -> Work {
        let spec = TextureSpec {
            kind: TextureKind::TextureBuffer,
            storage: WorkStorage::Shared,
            gpu_optimized: false,
            usage: Some(READ_WRITE.as_raw()),
            ..TextureSpec::texture_2d(FORMAT.as_raw(), TEXELS, 1)
        };
        let mut work = Work::new();
        let buffer = work.buffer(pattern(LEN));
        let texture = work.add(Resource::TextureOverBuffer {
            buffer,
            spec,
            offset: OFFSET,
            bytes_per_row: BYTES,
        });
        let source = work.buffer(written(64 * 4, 1));
        let as_made = work.buffer(vec![BEFORE; BYTES as usize]);
        let at_end = work.buffer(vec![BEFORE; BYTES as usize]);
        let into_buffer = |work: &mut Work, into: usize| {
            work.step(Step::CopyTextureToBuffer {
                texture,
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                size: [TEXELS, 1, 1],
                buffer: into,
                offset: 0,
                bytes_per_row: BYTES,
                bytes_per_image: 0,
                plane: Plane::Whole,
            });
            work.step(Step::Commit);
        };
        into_buffer(&mut work, as_made);
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: 64 * 4,
            bytes_per_image: 0,
            size: [64, 1, 1],
            texture,
            slice: 0,
            level: 0,
            origin: [880, 0, 0],
            plane: Plane::Whole,
        });
        work.step(Step::Commit);
        into_buffer(&mut work, at_end);
        for read in [as_made, at_end] {
            work.step(Step::ReadBuffer {
                buffer: read,
                offset: 0,
                length: BYTES as usize,
            });
        }
        work.step(Step::GetBytes {
            texture,
            level: 0,
            slice: 0,
            region: Rect::new(0, 0, TEXELS, 1),
            bytes_per_row: BYTES,
            bytes_per_image: 0,
            length: BYTES as usize,
            before: BEFORE,
        });
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: LEN as usize,
        });
        work
    }
}

/// The sites of the program of blits' reads, in its order.
const BLIT_READS: [&str; 4] = [
    "blits/made",
    "blits/written",
    "blits/get-bytes",
    "blits/buffer",
];

impl Case for TextureBufferTails {
    fn name(&self) -> &'static str {
        TEXTURE_BUFFER
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The runtime's functions view a texture buffer whose texels run past its buffer's length
        // through its tail, a staging buffer, and it announces `staged-texture-buffer`, which
        // Metal never does. Every answer is read back through `Shared` buffers, so a device
        // without unified memory announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice`).
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(
                metal2vulkan_runtime::Degradation::StagedTextureBuffer.code(),
            ),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: FIXTURE.to_owned(),
                source_sha256: crate::sha256_hex(Self::fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        let mut shape = Vec::new();
        for number in [LEN, OFFSET, TEXELS, u64::from(FORMAT.as_raw())] {
            shape.extend(number.to_le_bytes());
        }
        let mut texels = Vec::new();
        for t in MADE_READS.iter().chain(&WRITES).chain(&REPLACED_READS) {
            texels.extend(t.to_le_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("shape", &shape),
            Input::bytes("buffer-bytes", &pattern(LEN)),
            Input::bytes("texels", &texels),
            Input::bytes("replaced", &written(BYTES, 0)),
            Input::bytes("blit-source", &written(64 * 4, 1)),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, FIXTURE))?;
        let buffer = made(sites, "buffer", side.buffer(&pattern(LEN)))?;
        let texture = Self::over(side, sites, "texture", (buffer, OFFSET))?;
        let other_bytes = made(sites, "other/buffer", side.zeroed_buffer(BYTES as usize))?;
        let other = Self::over(side, sites, "other", (other_bytes, 0))?;
        // As made: the buffer's words inside the length, the allocation's zeros past it.
        Self::dispatch(
            side,
            sites,
            "made/kernel",
            pipeline,
            (texture, MADE_READS),
            (other, [0; 8]),
        )?;
        Self::get_bytes(side, sites, "made/get-bytes", texture)?;
        // The kernel writes texels 900 to 907: four inside the length, four past it.
        Self::dispatch(
            side,
            sites,
            "write",
            pipeline,
            (other, [0; 8]),
            (texture, WRITES),
        )?;
        Self::dispatch(
            side,
            sites,
            "written/kernel",
            pipeline,
            (texture, WRITES),
            (other, [0; 8]),
        )?;
        Self::get_bytes(side, sites, "written/get-bytes", texture)?;
        Self::buffer(side, sites, "written/buffer", buffer)?;
        // `replaceRegion:` writes every texel.
        let replaced = written(BYTES, 0);
        let size = [TEXELS as u32, 1];
        let at = "replaced/call";
        call(
            sites,
            at,
            side.replace_region(texture, size, BYTES as u32, &replaced),
            |_| Value::Absent,
        )?;
        Self::dispatch(
            side,
            sites,
            "replaced/kernel",
            pipeline,
            (texture, REPLACED_READS),
            (other, [0; 8]),
        )?;
        Self::get_bytes(side, sites, "replaced/get-bytes", texture)?;
        Self::buffer(side, sites, "replaced/buffer", buffer)?;
        // The blits, a program of their own over a buffer of their own.
        let done = call(sites, "blits", side.work(&Self::blits()), |_| Value::Absent)?;
        for (index, at) in BLIT_READS.into_iter().enumerate() {
            let bytes = done.reads.get(index).cloned().unwrap_or_default();
            let layout = match at {
                "blits/buffer" => buffer_layout(),
                _ => texel_layout(),
            };
            sites.push(Node::leaf(at, Observed::bytes(bytes, layout)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The texels inside the length.
    const INSIDE: u32 = ((LEN - OFFSET) / 4) as u32;

    /// The shape is the one the module states: 904 texels inside the length and 120 past it, the
    /// whole inside the oracle's allocation, and every read and write in the texture buffer.
    #[test]
    fn the_texture_buffer_crosses_the_length_where_the_module_says() {
        assert_eq!(INSIDE, 904);
        const { assert!(OFFSET + BYTES > LEN && OFFSET + BYTES <= 32768) }
        assert!(OFFSET.is_multiple_of(16));
        for t in MADE_READS.iter().chain(&WRITES).chain(&REPLACED_READS) {
            assert!(u64::from(*t) < TEXELS);
        }
        assert!(WRITES.iter().any(|&t| t < INSIDE) && WRITES.iter().any(|&t| t >= INSIDE));
        const { assert!(880 < INSIDE && 880 + 64 > INSIDE) }
        assert!(!pattern(LEN).contains(&0));
    }
}
