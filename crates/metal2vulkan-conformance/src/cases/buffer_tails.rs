//! A 2D texture over a buffer whose texels run past the buffer's length: what each path reads and
//! writes there.
//!
//! Plain Metal makes such a texture (`newTextureWithDescriptor:offset:bytesPerRow:`) wherever its
//! rows end inside the buffer's `allocatedSize`, and takes the texels past the length as one
//! memory every path shares: a kernel's `read()` and `write()`, a blit from and into it,
//! `getBytes:` and `replaceRegion:` each read what the others wrote
//! (`probes/texels_past_buffer_length.m`, an Apple M3 Max under macOS 27.0.1). What that memory
//! holds before anything writes it is the allocation's: over a buffer of 4096 bytes, whose
//! `allocatedSize` is its length, it is the buffer allocated next, which no case can state; over
//! a buffer of 20000 bytes, whose `allocatedSize` is 32768, it is the rest of the buffer's own
//! allocation, and it reads zero. The runtime keeps those texels in a tail of the texture's own
//! (`metal2vulkan-runtime`'s `buffer_texture.rs`), zero when made, so this family asks the second
//! shape, where Metal's answer is one a case can compare: a 16 by 256 `RGBA8Uint` texture at
//! offset 4096, its rows 64 bytes apart, over 20000 bytes. Its rows 0 to 247 lie inside the
//! length, row 248 crosses it 32 bytes (8 texels) in, and rows 249 to 255 lie past it.
//!
//! - `buffer-tails-kernels` reads the texture through the format scout's `scout_read_u` and
//!   `getBytes:` as made, after `scout_write_u` wrote every texel, and after `replaceRegion:`
//!   wrote every texel, and the buffer's 20000 bytes after each write.
//! - `buffer-tails-blits` reads it through a blit into a buffer as made, writes rows 244 to 251
//!   by a blit from a buffer (across the length) and rows 252 to 255 by a blit from a texture
//!   (wholly past it), copies rows 240 to 255 into a texture of its own, and reads every one of
//!   those back, with the buffer's bytes.
//!
//! Every byte compares exactly. What differs is the announcing: the runtime keeps a texture with
//! a tail as a shadow image and announces `shadowed-texture`, which Metal never does, and a
//! device without unified memory announces `shared-in-system-memory` for the `Shared` buffers the
//! answers are read back through. Nothing here can fault a GPU: one small texture, one thread a
//! texel, no loop.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, PlaneTexture, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, Storage as WorkStorage, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The case names, as the report and the recordings spell them.
pub const KERNELS: &str = "buffer-tails-kernels";
pub const BLITS: &str = "buffer-tails-blits";

/// The buffer's length: under its `allocatedSize` on the oracle (32768), so the texels past it lie
/// in the buffer's own allocation, where Metal reads zero.
const LEN: u64 = 20000;
/// The texture's first byte in the buffer.
const OFFSET: u64 = 4096;
/// The texture's rows lie this many bytes apart: its own row, 16 texels of 4 bytes.
const PITCH: u64 = 64;
/// The texture's width: the grid `format_scout`'s kernels are written for (`out[g.y * 16 + g.x]`).
const WIDTH: u64 = 16;
/// The texture's height: its last row ends at byte 20480, 480 bytes past the length.
const HEIGHT: u64 = 256;
/// The texture's bytes, packed.
const TEXTURE_BYTES: u64 = PITCH * HEIGHT;
/// The bytes a kernel's read of every texel gives: a `uint4` a texel.
const KERNEL_BYTES: u64 = 16 * WIDTH * HEIGHT;
const FORMAT: PixelFormat = PixelFormat::RGBA8Uint;

/// The fixture whose kernels read and write the texture: our own MSL.
const SCOUT: &str = "format_scout";
const READ_ENTRY: &str = "scout_read_u";
const WRITE_ENTRY: &str = "scout_write_u";

/// The byte a buffer the blits read into holds before they run, which no write here makes, so a
/// read that never happened is not read as zeros.
const BEFORE: u8 = 0xEE;

/// The buffer's bytes before anything runs: byte `i` holds `i % 251 + 1`, which is never zero, so
/// a texel that reads zero is one past the length and never the buffer's.
fn pattern(n: u64) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8 + 1).collect()
}

/// The bytes `replaceRegion:` and the blits write: never zero, and unlike [`pattern`]'s.
fn written(n: u64, set: u64) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 13 + set * 101) % 253) as u8 + 2)
        .collect()
}

/// What both cases loosen: the notices, and nothing else.
fn allowances() -> Vec<Allowance> {
    // The runtime keeps a texture whose texels run past its buffer's length as a shadow image of
    // the buffer's bytes and its tail, and announces `shadowed-texture`, which Metal never does.
    // Every case reads its answers back through `Shared` buffers, so a device without unified
    // memory announces `shared-in-system-memory` (`crate::replay::shared_memory_notice`).
    vec![
        crate::replay::shared_memory_notice(),
        crate::replay::notice_allowance(metal2vulkan_runtime::Degradation::ShadowedTexture.code()),
    ]
}

/// The numbers that place the texture in the buffer, for both cases' inputs.
fn shape_words() -> Vec<u8> {
    let mut words = Vec::new();
    for number in [
        LEN,
        OFFSET,
        PITCH,
        WIDTH,
        HEIGHT,
        u64::from(FORMAT.as_raw()),
    ] {
        words.extend(number.to_le_bytes());
    }
    words
}

/// Rows of `row` bytes over `length` bytes from byte 0 of a read, as a site's layout, so a
/// difference names the row and the byte or word inside it.
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

/// A kernel's read of every texel: a row of the texture is 16 `uint4`s.
fn kernel_layout() -> ByteLayout {
    rows(KERNEL_BYTES, 16 * WIDTH, Element::U32)
}

/// The texture's bytes as `getBytes:` or a blit reads them: rows of [`PITCH`].
fn texture_layout() -> ByteLayout {
    rows(TEXTURE_BYTES, PITCH, Element::U8)
}

/// The buffer's bytes, in rows of 32 (which divide its length): the texture's row `y` is rows
/// `128 + 2y` and `129 + 2y`.
fn buffer_layout() -> ByteLayout {
    rows(LEN, 32, Element::U8)
}

/// Both cases.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(Kernels), Box::new(Blits)]
}

/// The texture read and written by kernels and by the host.
pub struct Kernels;

impl Kernels {
    fn fixture(entry: &'static str) -> Fixture {
        Fixture { name: SCOUT, entry }
    }

    /// Every texel as `scout_read_u` reads it, into a buffer of its own, at `at`.
    fn kernel_read(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        (read, texture): (Handle, Handle),
    ) -> Result<(), Abort> {
        let out = made(
            sites,
            &format!("{at}/out"),
            side.zeroed_buffer(KERNEL_BYTES as usize),
        )?;
        let (textures, buffers) = ([texture], [out]);
        let grid = [WIDTH as u32, HEIGHT as u32, 1];
        let dispatched = side.dispatch_threads(
            read,
            &Bindings::new(&textures, &buffers),
            &[],
            grid,
            [WIDTH as u32, 4, 1],
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
            side.read_buffer(out, 0, KERNEL_BYTES as usize),
        )?;
        sites.push(Node::leaf(at, Observed::bytes(bytes, kernel_layout())));
        Ok(())
    }

    /// The texture's bytes as `getBytes:` reads them, at `at`.
    fn get_bytes(
        side: &mut dyn Side,
        sites: &mut Sites,
        at: &str,
        texture: Handle,
    ) -> Result<(), Abort> {
        let size = [WIDTH as u32, HEIGHT as u32];
        let bytes = made(sites, &format!("{at}/call"), {
            side.read_texture(texture, size, PITCH as u32)
        })?;
        sites.push(Node::leaf(at, Observed::bytes(bytes, texture_layout())));
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
}

impl Case for Kernels {
    fn name(&self) -> &'static str {
        KERNELS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        allowances()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for entry in [READ_ENTRY, WRITE_ENTRY] {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{SCOUT}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(READ_ENTRY);
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("shape", &shape_words()),
            Input::bytes("buffer-bytes", &pattern(LEN)),
            Input::bytes("replaced", &written(TEXTURE_BYTES, 0)),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture(READ_ENTRY).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let read = made(
            sites,
            "pipeline/read",
            side.compute_pipeline(library, READ_ENTRY),
        )?;
        let write = made(
            sites,
            "pipeline/write",
            side.compute_pipeline(library, WRITE_ENTRY),
        )?;
        let buffer = made(sites, "buffer", side.buffer(&pattern(LEN)))?;
        let over = PlaneTexture {
            ostype: None,
            format: FORMAT.as_raw(),
            size: [WIDTH as u32, HEIGHT as u32],
            plane: 0,
            planes: [(OFFSET, PITCH), (0, 0)],
        };
        let texture = made(sites, "texture", side.plane_texture(buffer, &over))?;
        // As made: the buffer's bytes inside the length, the allocation's zeros past it.
        Self::kernel_read(side, sites, "made/kernel", (read, texture))?;
        Self::get_bytes(side, sites, "made/get-bytes", texture)?;
        // `scout_write_u` writes `(x, y, 3, 5)` into every texel, inside the length and past it.
        let only = [texture];
        let grid = [WIDTH as u32, HEIGHT as u32, 1];
        let dispatched = side.dispatch_threads(
            write,
            &Bindings::textures(&only),
            &[],
            grid,
            [WIDTH as u32, 4, 1],
        );
        let dispatched = made(sites, "write/dispatch", dispatched)?;
        call(sites, "write/status", side.wait(dispatched), Clone::clone)?;
        Self::kernel_read(side, sites, "written/kernel", (read, texture))?;
        Self::get_bytes(side, sites, "written/get-bytes", texture)?;
        Self::buffer(side, sites, "written/buffer", buffer)?;
        // `replaceRegion:` writes every texel.
        let replaced = written(TEXTURE_BYTES, 0);
        let size = [WIDTH as u32, HEIGHT as u32];
        let at = "replaced/call";
        call(
            sites,
            at,
            side.replace_region(texture, size, PITCH as u32, &replaced),
            |_| Value::Absent,
        )?;
        Self::kernel_read(side, sites, "replaced/kernel", (read, texture))?;
        Self::get_bytes(side, sites, "replaced/get-bytes", texture)?;
        Self::buffer(side, sites, "replaced/buffer", buffer)?;
        Ok(())
    }
}

/// The texture read and written by blits.
pub struct Blits;

/// Every texture of [`Blits`]: [`FORMAT`], [`WIDTH`] by [`HEIGHT`], plain, `Shared`, read and
/// written by shaders.
fn spec() -> TextureSpec {
    TextureSpec {
        storage: WorkStorage::Shared,
        gpu_optimized: false,
        usage: Some(
            TextureUsage::SHADER_READ
                .union(TextureUsage::SHADER_WRITE)
                .as_raw(),
        ),
        ..TextureSpec::texture_2d(FORMAT.as_raw(), WIDTH, HEIGHT)
    }
}

impl Blits {
    /// The program: the copies, each in a command buffer of its own, then every read.
    fn work() -> Work {
        let mut work = Work::new();
        let buffer = work.buffer(pattern(LEN));
        let texture = work.add(Resource::TextureOverBuffer {
            buffer,
            spec: spec(),
            offset: OFFSET,
            bytes_per_row: PITCH,
        });
        let ordinary = work.add(Resource::Texture { spec: spec() });
        let source = work.buffer(written(8 * PITCH, 1));
        let as_made = work.buffer(vec![BEFORE; TEXTURE_BYTES as usize]);
        let at_end = work.buffer(vec![BEFORE; TEXTURE_BYTES as usize]);
        let whole = Rect::new(0, 0, WIDTH, HEIGHT);
        work.step(Step::ReplaceRegion {
            texture: ordinary,
            level: 0,
            slice: 0,
            region: whole,
            bytes: written(TEXTURE_BYTES, 2),
            rows: HEIGHT,
            bytes_per_row: PITCH,
            bytes_per_image: 0,
        });
        let into_buffer = |work: &mut Work, buffer: usize| {
            work.step(Step::CopyTextureToBuffer {
                texture,
                slice: 0,
                level: 0,
                origin: [0, 0, 0],
                size: [WIDTH, HEIGHT, 1],
                buffer,
                offset: 0,
                bytes_per_row: PITCH,
                bytes_per_image: 0,
                plane: Plane::Whole,
            });
            work.step(Step::Commit);
        };
        // As made, through a blit.
        into_buffer(&mut work, as_made);
        // Rows 244 to 251 from a buffer: 244 to 247 inside the length, 248 across it, 249 to 251
        // past it.
        work.step(Step::CopyBufferToTexture {
            buffer: source,
            offset: 0,
            bytes_per_row: PITCH,
            bytes_per_image: 0,
            size: [WIDTH, 8, 1],
            texture,
            slice: 0,
            level: 0,
            origin: [0, 244, 0],
            plane: Plane::Whole,
        });
        work.step(Step::Commit);
        // Rows 252 to 255 from the ordinary texture's: wholly past the length.
        work.step(Step::CopyTextureToTexture {
            from: ordinary,
            from_slice: 0,
            from_level: 0,
            from_origin: [0, 252, 0],
            size: [WIDTH, 4, 1],
            to: texture,
            to_slice: 0,
            to_level: 0,
            to_origin: [0, 252, 0],
        });
        work.step(Step::Commit);
        // Rows 240 to 255, across the length, into the ordinary texture's rows 0 to 15.
        work.step(Step::CopyTextureToTexture {
            from: texture,
            from_slice: 0,
            from_level: 0,
            from_origin: [0, 240, 0],
            size: [WIDTH, 16, 1],
            to: ordinary,
            to_slice: 0,
            to_level: 0,
            to_origin: [0, 0, 0],
        });
        work.step(Step::Commit);
        into_buffer(&mut work, at_end);
        for read in [as_made, at_end] {
            work.step(Step::ReadBuffer {
                buffer: read,
                offset: 0,
                length: TEXTURE_BYTES as usize,
            });
        }
        for read in [texture, ordinary] {
            work.step(Step::GetBytes {
                texture: read,
                level: 0,
                slice: 0,
                region: whole,
                bytes_per_row: PITCH,
                bytes_per_image: 0,
                length: TEXTURE_BYTES as usize,
                before: BEFORE,
            });
        }
        work.step(Step::ReadBuffer {
            buffer,
            offset: 0,
            length: LEN as usize,
        });
        work
    }
}

/// The sites of [`Blits`]'s reads, in the program's order.
const BLIT_READS: [&str; 5] = [
    "made/blit",
    "written/blit",
    "written/get-bytes",
    "ordinary/get-bytes",
    "written/buffer",
];

impl Case for Blits {
    fn name(&self) -> &'static str {
        BLITS
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        allowances()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut steps = Vec::new();
        for set in 0..3 {
            steps.extend(written(16, set));
        }
        Ok(vec![
            Input::bytes("shape", &shape_words()),
            Input::bytes("buffer-bytes", &pattern(LEN)),
            Input::bytes("written-sets", &steps),
            Input::bytes(
                "copies",
                b"rows 244+8 from a buffer; rows 252+4 from a texture; rows 240+16 into a texture",
            ),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let done = call(sites, "work", side.work(&Self::work()), |_| Value::Absent)?;
        for (index, at) in BLIT_READS.into_iter().enumerate() {
            let bytes = done.reads.get(index).cloned().unwrap_or_default();
            let layout = match at {
                "written/buffer" => buffer_layout(),
                _ => texture_layout(),
            };
            sites.push(Node::leaf(at, Observed::bytes(bytes, layout)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape is the one the module states: rows 0 to 247 inside the length, row 248 across it
    /// 32 bytes in, rows 249 to 255 past it, and the whole inside the oracle's allocation.
    #[test]
    fn the_texture_crosses_the_length_where_the_module_says() {
        let row = |y: u64| OFFSET + y * PITCH;
        assert!(row(248) < LEN && row(249) > LEN);
        assert_eq!(LEN - row(248), 32);
        assert!(row(HEIGHT) > LEN && row(HEIGHT) <= 32768);
        assert_eq!(LEN % 32, 0);
        assert!(!pattern(LEN).contains(&0));
        assert!(!written(TEXTURE_BYTES, 2).contains(&0));
    }
}
