//! A 2D texture over a buffer whose texels run past the buffer's length (`buffer_texture.rs`'s
//! **Tails**): plain Metal makes it, and every path of Metal's takes the texels past the length
//! as one memory, which each reads what each wrote to (an Apple M3 Max under macOS 27.0.1,
//! `metal2vulkan-conformance/probes/texels_past_buffer_length.m`). The runtime makes it too: its
//! texels are the buffer's memory as far as that holds them (a buffer of 16 KiB or more holds its
//! allocation, its length rounded up to 16 KiB, zero when made; a heap's buffer its heap to its
//! end), and those past it lie in the texture's tail, zero when made.
//!
//! Each case writes the texels through one path and reads them back through every other: a
//! kernel (`invert_texels`, the owned fixture `validation/fixtures/public/examples/
//! runtime_examples.ll`, which writes each texel of its first texture inverted, its alpha kept,
//! into its second), a blit, and the CPU (`replace_region`, `get_bytes`). The buffer's bytes are
//! read through its `contents`, or by a blit of the buffer for a `Private` one. Buffers made
//! around the texture's are filled with a pattern no write here makes, and keep it: no write
//! past the length reaches memory but the tail and the buffer's own. The shapes are reims-vgpu's
//! (64 by 17 at rows of 256 over 4096 bytes, and 64 by 16 from offset 2048), one whose last row
//! crosses the length (64 by 80 over 20000 bytes, whose row 78 holds 32 of the buffer's bytes,
//! which its allocation holds whole), and one wholly past it (64 by 1 from offset 4096).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::draw_again::{device as hearing_device, Heard};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, BufferTextureProblem, CommandQueue, ComputePipelineState, Degradation,
    Device, Error, HeapDescriptor, HeapType, HostMemory, IoSurfacePlane, Nil, Origin,
    PipelineOptions, Problem, Region, ResourceOptions, Size, Texture, TextureDescriptor,
    TexturePlacement, TextureShadowReason,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ptr::NonNull;

const WIDTH: u64 = 64;
const ROW: u64 = WIDTH * 4;
/// What the buffers made around the texture's hold, which no write here makes.
const AROUND: u8 = 0xA5;

/// A byte of the buffer's pattern: its offset's, never [`AROUND`] nor zero.
fn pattern(at: u64) -> u8 {
    (at % 251) as u8 + 1
}

/// A texel row's bytes `value(texel, channel)`.
fn row(value: impl Fn(u64, u64) -> u8) -> Vec<u8> {
    (0..ROW).map(|at| value(at / 4, at % 4)).collect()
}

/// `bytes` inverted as the kernel inverts a texel: each color channel, the alpha kept.
fn inverted(bytes: &[u8]) -> Vec<u8> {
    let channel = |(at, &byte): (usize, &u8)| if at % 4 == 3 { byte } else { 255 - byte };
    bytes.iter().enumerate().map(channel).collect()
}

/// The texture's world: its buffer and the buffers around it, the texture, the kernel and the
/// plain textures it reads and writes.
struct Tail {
    queue: CommandQueue,
    invert: ComputePipelineState,
    buffer: Buffer,
    around: Vec<Buffer>,
    texture: Texture,
    height: u64,
    offset: u64,
    /// The texture's `bytesPerRow`.
    pitch: u64,
    /// A plain texture of the texture's extent: the kernel's other side.
    plain: Texture,
}

impl Tail {
    /// A `height`-row texture at `offset` over a buffer of `length` bytes of `options`, which
    /// holds [`pattern`] (but a `Private` one, which holds zero), among buffers holding
    /// [`AROUND`].
    fn new(
        device: &Device,
        (length, options): (u64, ResourceOptions),
        offset: u64,
        height: u64,
    ) -> Tail {
        let made = |buffer: &Buffer, descriptor: &TextureDescriptor| {
            buffer.new_texture(descriptor, offset, ROW).unwrap()
        };
        let target = || device.new_buffer(length, options).unwrap();
        Tail::made(device, target, (offset, height, ROW), made)
    }

    /// As [`new`](Self::new), over the buffer `target` makes, its rows `pitch` apart.
    fn over(
        device: &Device,
        target: impl FnOnce() -> Buffer,
        (offset, height, pitch): (u64, u64, u64),
    ) -> Tail {
        let made = |buffer: &Buffer, descriptor: &TextureDescriptor| {
            buffer.new_texture(descriptor, offset, pitch).unwrap()
        };
        Tail::made(device, target, (offset, height, pitch), made)
    }

    /// As [`new`](Self::new), the texture a 2D one of a plane at `offset` of a surface with
    /// planes, rows [`ROW`] apart, which the buffer stands for.
    fn over_plane(device: &Device, length: u64, offset: u64, height: u64) -> Tail {
        let made = |buffer: &Buffer, descriptor: &TextureDescriptor| {
            let plane = IoSurfacePlane::new(offset, ROW, true);
            (buffer.new_texture_with_iosurface_plane(descriptor, plane)).unwrap()
        };
        let target = || {
            device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap()
        };
        Tail::made(device, target, (offset, height, ROW), made)
    }

    /// The world of a `height`-row texture at `offset`, its rows `pitch` apart, over the buffer
    /// `target` makes, `made` over the buffer.
    fn made(
        device: &Device,
        target: impl FnOnce() -> Buffer,
        (offset, height, pitch): (u64, u64, u64),
        made: impl FnOnce(&Buffer, &TextureDescriptor) -> Texture,
    ) -> Tail {
        let around_buffer = || {
            let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            // SAFETY: the buffer's 4096 mapped bytes, which no command uses yet.
            unsafe { std::ptr::write_bytes(buffer.contents().unwrap().as_ptr(), AROUND, 4096) };
            buffer
        };
        let mut around: Vec<Buffer> = (0..8).map(|_| around_buffer()).collect();
        let buffer = target();
        let (length, options) = (buffer.length(), buffer.resource_options());
        around.extend((0..8).map(|_| around_buffer()));
        if let Some(contents) = buffer.contents() {
            let bytes: Vec<u8> = (0..length).map(pattern).collect();
            // SAFETY: the buffer's `length` mapped bytes, which no command uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
        }
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, height, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor.resource_options = options;
        let texture = made(&buffer, &descriptor);
        let examples = library(device, "examples/runtime_examples.ll");
        let kernel = examples.new_function_with_name("invert_texels").unwrap();
        let pipeline = PipelineOptions::default();
        let invert = device
            .new_compute_pipeline_state_with_function(&kernel, &pipeline)
            .unwrap();
        descriptor.resource_options = ResourceOptions::shared();
        let plain = device.new_texture(&descriptor).unwrap();
        Tail {
            queue: device.new_command_queue().unwrap(),
            invert,
            buffer,
            around,
            texture,
            height,
            offset,
            pitch,
            plain,
        }
    }

    /// The kernel run once, from `from` into `into`, every texel.
    fn invert(&self, from: &Texture, into: &Texture) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.invert).unwrap();
        encoder.set_texture(Some(from), 0).unwrap();
        encoder.set_texture(Some(into), 1).unwrap();
        let grid = Size::new(WIDTH, self.height, 1);
        (encoder.dispatch_threads(grid, Size::new(8, 1, 1))).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// Row `y`'s bytes of `texture` as the CPU reads them.
    fn cpu_row(&self, texture: &Texture, y: u64) -> Vec<u8> {
        let mut bytes = vec![0; ROW as usize];
        let region = Region::new_2d(0, y, WIDTH, 1);
        texture.get_bytes(&mut bytes, ROW, 0, region, 0, 0).unwrap();
        bytes
    }

    /// Row `y` of the texture as the kernel reads it, inverted back.
    fn kernel_row(&self, y: u64) -> Vec<u8> {
        self.invert(&self.texture, &self.plain);
        inverted(&self.cpu_row(&self.plain, y))
    }

    /// Row `y` of the texture as a blit into a buffer copies it.
    fn blit_row(&self, y: u64) -> Vec<u8> {
        let device = self.queue.device();
        let read = device.new_buffer(ROW, ResourceOptions::shared()).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, y, 0), Size::new(WIDTH, 1, 1));
        let copy = blit.copy_from_texture_to_buffer(
            &self.texture,
            0,
            0,
            origin,
            size,
            &read,
            0,
            ROW,
            ROW,
            BlitOption::None,
        );
        copy.unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: the read buffer's `ROW` mapped bytes, which no command uses now.
        unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), ROW as usize) }
            .to_vec()
    }

    /// Row `y` of the texture, `bytes`, written by a blit from a buffer.
    fn blit_write(&self, y: u64, bytes: &[u8]) {
        let device = self.queue.device();
        let source = device.new_buffer(ROW, ResourceOptions::shared()).unwrap();
        // SAFETY: the source's `ROW` mapped bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                source.contents().unwrap().as_ptr(),
                bytes.len(),
            )
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let copy = blit.copy_from_buffer_to_texture(
            &source,
            0,
            ROW,
            ROW,
            Size::new(WIDTH, 1, 1),
            &self.texture,
            0,
            0,
            Origin::new(0, y, 0),
            BlitOption::None,
        );
        copy.unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    /// Row `y` written by the kernel: `bytes` placed in the plain texture by the CPU, inverted
    /// into the texture, every row (`rows` holding each row's bytes).
    fn kernel_write(&self, rows: &[Vec<u8>]) {
        for (y, bytes) in rows.iter().enumerate() {
            let region = Region::new_2d(0, y as u64, WIDTH, 1);
            let source = inverted(bytes);
            (self.plain.replace_region(region, 0, 0, &source, ROW, 0)).unwrap();
        }
        self.invert(&self.plain, &self.texture);
    }

    /// The buffer's bytes `range`, through its `contents`.
    fn buffer_bytes(&self, range: std::ops::Range<u64>) -> Vec<u8> {
        let contents = self.buffer.contents().unwrap().as_ptr();
        // SAFETY: `range` lies in the buffer's mapped bytes, which no command uses now.
        let all = unsafe { std::slice::from_raw_parts(contents, self.buffer.length() as usize) };
        all[range.start as usize..range.end as usize].to_vec()
    }

    /// Whether every buffer made around the texture's still holds [`AROUND`].
    fn around_kept(&self) -> bool {
        self.around.iter().all(|buffer| {
            let contents = buffer.contents().unwrap().as_ptr();
            // SAFETY: the buffer's 4096 mapped bytes, which no command uses.
            let bytes = unsafe { std::slice::from_raw_parts(contents, 4096) };
            bytes.iter().all(|&byte| byte == AROUND)
        })
    }

    /// The bytes `range` of the buffer's memory from its start, through its `contents`: past its
    /// length, those its allocation or its heap holds there.
    fn memory_bytes(&self, range: std::ops::Range<u64>) -> Vec<u8> {
        let contents = self.buffer.contents().unwrap().as_ptr();
        let len = (range.end - range.start) as usize;
        // SAFETY: `range` lies in the memory mapped from `contents` (the buffer's allocation, or
        // its heap from its start), which no command uses now.
        unsafe { std::slice::from_raw_parts(contents.add(range.start as usize), len) }.to_vec()
    }

    /// The last byte of the texture's texels, past which none of the buffer's is the texture's.
    fn end(&self) -> u64 {
        self.offset + (self.height - 1) * self.pitch + ROW
    }

    /// Row `y`'s bytes as the buffer's memory holds them before `kept`, those from it on zero.
    fn row_kept(&self, y: u64, kept: u64) -> Vec<u8> {
        let start = self.offset + y * self.pitch;
        let inside = kept.clamp(start, start + ROW);
        let mut bytes = match inside > start {
            true => self.memory_bytes(start..inside),
            false => Vec::new(),
        };
        bytes.resize(ROW as usize, 0);
        bytes
    }

    /// Row `y`'s bytes as the buffer holds them, those past its length zero.
    fn made_row(&self, y: u64) -> Vec<u8> {
        let start = self.offset + y * self.pitch;
        let length = self.buffer.length();
        (start..start + ROW)
            .map(|at| if at < length { pattern(at) } else { 0 })
            .collect()
    }
}

/// Every path reads the texels inside the length as the buffer's bytes and those past it as
/// zero, as made; and what each path writes past the length every path reads, while the
/// buffer's bytes and every other buffer's keep theirs: reims-vgpu's 64 by 17 over 4096 bytes,
/// a `Shared` buffer (whose shadow merges its copies back).
#[test]
fn a_texture_past_its_buffer_s_length_keeps_those_texels_in_its_tail() {
    let scratch = Scratch::new("buffer-tail");
    let device = device(&scratch.0);
    let tail = Tail::new(&device, (4096, ResourceOptions::shared()), 0, 17);
    let texels = tail.texture.buffer_texels().unwrap();
    assert!(texels.past().is_some(), "a buffer under 16 KiB: a tail");
    // As made: rows 0 to 15 the buffer's bytes, row 16 zero, on every path.
    for y in [0, 7, 15, 16] {
        let want = tail.made_row(y);
        assert_eq!(tail.cpu_row(&tail.texture, y), want, "get_bytes row {y}");
        assert_eq!(tail.blit_row(y), want, "blit row {y}");
        assert_eq!(tail.kernel_row(y), want, "kernel row {y}");
    }
    // The kernel writes every row: rows 0 to 15 reach the buffer, row 16 the tail.
    let written: Vec<Vec<u8>> = (0..17)
        .map(|y| row(|x, c| (y * 13 + x * 3 + c * 7) as u8 | 0x80))
        .collect();
    tail.kernel_write(&written);
    assert_eq!(tail.buffer_bytes(15 * ROW..16 * ROW), written[15]);
    for y in [3, 15, 16] {
        assert_eq!(
            tail.cpu_row(&tail.texture, y),
            written[y as usize],
            "get_bytes {y}"
        );
        assert_eq!(tail.blit_row(y), written[y as usize], "blit {y}");
        assert_eq!(tail.kernel_row(y), written[y as usize], "kernel {y}");
    }
    // The CPU writes row 16: every path reads it, the buffer's bytes as they were.
    let by_cpu = row(|x, c| (x * 5 + c) as u8 | 0x40);
    let region = Region::new_2d(0, 16, WIDTH, 1);
    (tail.texture.replace_region(region, 0, 0, &by_cpu, ROW, 0)).unwrap();
    assert_eq!(tail.kernel_row(16), by_cpu, "kernel after replace_region");
    assert_eq!(tail.blit_row(16), by_cpu, "blit after replace_region");
    assert_eq!(tail.buffer_bytes(15 * ROW..16 * ROW), written[15]);
    // A blit writes row 16: every path reads it.
    let by_blit = row(|x, c| (x * 9 + c * 3) as u8 | 0x20);
    tail.blit_write(16, &by_blit);
    assert_eq!(tail.kernel_row(16), by_blit, "kernel after the blit");
    assert_eq!(
        tail.cpu_row(&tail.texture, 16),
        by_blit,
        "get_bytes after the blit"
    );
    assert_eq!(tail.buffer_bytes(15 * ROW..16 * ROW), written[15]);
    assert!(tail.around_kept(), "no write reaches another buffer");
}

/// The other shapes: 64 by 16 from offset 2048 (8 rows past), 64 by 80 over 20000 bytes (row 78
/// crosses the length 32 bytes in, and its first 8 texels are the buffer's; the buffer's
/// allocation holds it whole), and 64 by 1 wholly past a 4096-byte buffer. Each reads as made on
/// every path, and a kernel's write of every row reads back on every path, the bytes inside the
/// length in the buffer.
#[test]
fn every_shape_past_the_length_reads_its_tail_and_its_buffer() {
    let scratch = Scratch::new("buffer-tail-shapes");
    let device = device(&scratch.0);
    let shapes = [
        (4096, 2048, 16, [0, 7, 8, 15]),
        (20000, 0, 80, [0, 77, 78, 79]),
    ];
    for (length, offset, height, rows) in shapes {
        let tail = Tail::new(&device, (length, ResourceOptions::shared()), offset, height);
        for y in rows {
            let want = tail.made_row(y);
            assert_eq!(
                tail.cpu_row(&tail.texture, y),
                want,
                "{length} get_bytes {y}"
            );
            assert_eq!(tail.blit_row(y), want, "{length} blit {y}");
            assert_eq!(tail.kernel_row(y), want, "{length} kernel {y}");
        }
        let written: Vec<Vec<u8>> = (0..height)
            .map(|y| row(|x, c| (y * 11 + x * 3 + c) as u8 | 0x80))
            .collect();
        tail.kernel_write(&written);
        for y in rows {
            let want = &written[y as usize];
            assert_eq!(
                &tail.cpu_row(&tail.texture, y),
                want,
                "{length} get_bytes {y}"
            );
            assert_eq!(&tail.blit_row(y), want, "{length} blit {y}");
            assert_eq!(&tail.kernel_row(y), want, "{length} kernel {y}");
        }
        // The bytes inside the length the kernel wrote: the buffer's own, to its last byte.
        for y in [rows[1], rows[2]] {
            let start = offset + y * ROW;
            let inside = length.min(start + ROW);
            if inside > start {
                let want = &written[y as usize][..(inside - start) as usize];
                assert_eq!(
                    tail.buffer_bytes(start..inside),
                    want,
                    "{length} buffer row {y}"
                );
            }
        }
        assert!(
            tail.around_kept(),
            "{length}: no write reaches another buffer"
        );
    }
    let past = Tail::new(&device, (4096, ResourceOptions::shared()), 4096, 1);
    assert_eq!(past.cpu_row(&past.texture, 0), vec![0; ROW as usize]);
    let by_cpu = row(|x, c| (x + c) as u8 | 0x10);
    let region = Region::new_2d(0, 0, WIDTH, 1);
    (past.texture.replace_region(region, 0, 0, &by_cpu, ROW, 0)).unwrap();
    assert_eq!(past.kernel_row(0), by_cpu, "wholly past: kernel");
    assert_eq!(past.blit_row(0), by_cpu, "wholly past: blit");
    assert_eq!(
        past.buffer_bytes(0..4096),
        (0..4096).map(pattern).collect::<Vec<u8>>()
    );
    assert!(
        past.around_kept(),
        "wholly past: no write reaches another buffer"
    );
}

/// A 2D texture of a plane of an IOSurface with planes whose rows run past the buffer standing
/// for the surface (`linear_plan.rs`'s **Beyond the plane's rows**): Metal makes it and lays the
/// rows past the surface in no one place (an Apple M3 Max under macOS 27.0.1,
/// `metal2vulkan-conformance/probes/surface_texture_envelope.m`'s `bgra1.64x65` and
/// `bgra1.64x128`: its blits and kernels read zeros there, its `getBytes` zeros and other memory,
/// and its writes there change no byte of the surface), where the runtime keeps them in the
/// texture's tail, zero when made, which every path reads and writes alike, announced once as
/// `surface-texels-tail`. The rows inside the surface are its bytes on every path, as Metal's
/// are, and what the kernel writes there reaches them. Two shapes: 64 by 17 over 4096 bytes (row
/// 16 past), and 64 by 16 from offset 2048 (rows 8 to 15 past); every buffer around keeps its
/// bytes.
#[test]
fn a_plane_past_its_surface_keeps_those_texels_in_its_tail() {
    let scratch = Scratch::new("buffer-tail-plane");
    let heard = Heard::default();
    let device = hearing_device(&scratch.0, &[], &heard);
    for (offset, height, rows) in [(0, 17, [0, 15, 16]), (2048, 16, [0, 7, 8])] {
        let tail = Tail::over_plane(&device, 4096, offset, height);
        for y in rows {
            let want = tail.made_row(y);
            assert_eq!(
                tail.cpu_row(&tail.texture, y),
                want,
                "{offset} get_bytes {y}"
            );
            assert_eq!(tail.blit_row(y), want, "{offset} blit {y}");
            assert_eq!(tail.kernel_row(y), want, "{offset} kernel {y}");
        }
        let written: Vec<Vec<u8>> = (0..height)
            .map(|y| row(|x, c| (y * 13 + x * 3 + c * 7) as u8 | 0x80))
            .collect();
        tail.kernel_write(&written);
        let inside = rows[1];
        let start = offset + inside * ROW;
        assert_eq!(
            tail.buffer_bytes(start..start + ROW),
            written[inside as usize],
            "{offset}: the kernel's row {inside} reaches the surface's bytes"
        );
        for y in rows {
            let want = &written[y as usize];
            assert_eq!(
                &tail.cpu_row(&tail.texture, y),
                want,
                "{offset} get_bytes {y}"
            );
            assert_eq!(&tail.blit_row(y), want, "{offset} blit {y}");
            assert_eq!(&tail.kernel_row(y), want, "{offset} kernel {y}");
        }
        let past = rows[2];
        let by_cpu = row(|x, c| (x * 5 + c) as u8 | 0x40);
        let region = Region::new_2d(0, past, WIDTH, 1);
        (tail.texture.replace_region(region, 0, 0, &by_cpu, ROW, 0)).unwrap();
        assert_eq!(
            tail.kernel_row(past),
            by_cpu,
            "{offset}: kernel after replace_region"
        );
        assert_eq!(
            tail.blit_row(past),
            by_cpu,
            "{offset}: blit after replace_region"
        );
        assert_eq!(
            tail.buffer_bytes(start..start + ROW),
            written[inside as usize],
            "{offset}: the surface's bytes as the kernel left them"
        );
        assert!(
            tail.around_kept(),
            "{offset}: no write reaches another buffer"
        );
    }
    let notices = heard.lock().unwrap();
    let tails = notices
        .iter()
        .filter(|notice| notice.degradation == Degradation::SurfaceTexelsTail)
        .count();
    assert_eq!(tails, 1, "announced once");
}

/// A colour 2D texture of two levels, a 3D texture and a 2D multisample one over a plane of an
/// IOSurface with planes (`linear_plan.rs`'s **Beyond the plane's rows**): Metal makes each and
/// lays its texels out in no one place there, its first level's and slice's among them (an Apple
/// M3 Max under macOS 27.0.1, `metal2vulkan-conformance/probes/surface_texture_envelope.m`'s
/// `bgra1x2.mips2`, `bgra1x2.3d.d2` and `bgra1x2.2dms4.private`: its GPU reads them from other
/// bytes than its writes and its CPU put them), where the runtime makes each in memory of its
/// own, zero when made, announced once as `surface-texels-apart`: its first level reads zero
/// where the plane holds other bytes, what the CPU writes into a level or a slice the CPU and a
/// blit read back, and no byte of the surface changes.
#[test]
fn a_plane_s_texture_metal_lays_in_no_one_place_lies_apart() {
    let scratch = Scratch::new("buffer-tail-apart");
    let heard = Heard::default();
    let device = hearing_device(&scratch.0, &[], &heard);
    const LENGTH: u64 = 32768;
    let surface = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let bytes: Vec<u8> = (0..LENGTH).map(pattern).collect();
    // SAFETY: the buffer's `LENGTH` mapped bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            surface.contents().unwrap().as_ptr(),
            bytes.len(),
        )
    };
    let plane = IoSurfacePlane::new(0, ROW, true);
    let queue = device.new_command_queue().unwrap();
    // A blit of `texture`'s first `width`-texel row of `level` and `slice` into a buffer.
    let blit_row = |texture: &Texture, level: u64, slice: u64, origin: Origin, width: u64| {
        let read = device
            .new_buffer(width * 4, ResourceOptions::shared())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let copy = blit.copy_from_texture_to_buffer(
            texture,
            slice,
            level,
            origin,
            Size::new(width, 1, 1),
            &read,
            0,
            width * 4,
            width * 4,
            BlitOption::None,
        );
        copy.unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: the read buffer's mapped bytes, which no command uses now.
        let all = unsafe {
            std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), (width * 4) as usize)
        };
        all.to_vec()
    };
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let mut levels = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, 64, false);
    levels.mipmap_level_count = 2;
    levels.usage = usage;
    levels.resource_options = ResourceOptions::shared();
    let mips = (surface.new_texture_with_iosurface_plane(&levels, plane)).unwrap();
    let mut first = vec![0xEE; ROW as usize];
    let region = Region::new_2d(0, 0, WIDTH, 1);
    (mips.get_bytes(&mut first, ROW, 0, region, 0, 0)).unwrap();
    assert_eq!(
        first,
        vec![0; ROW as usize],
        "level 0 made zero, not the plane's"
    );
    let half = row(|x, c| (x * 7 + c) as u8 | 0x80)[..ROW as usize / 2].to_vec();
    let level_one = Region::new_2d(0, 0, WIDTH / 2, 1);
    (mips.replace_region(level_one, 1, 0, &half, ROW / 2, 0)).unwrap();
    let mut back = vec![0; ROW as usize / 2];
    (mips.get_bytes(&mut back, ROW / 2, 0, level_one, 1, 0)).unwrap();
    assert_eq!(back, half, "level 1 by the CPU");
    assert_eq!(
        blit_row(&mips, 1, 0, Origin::new(0, 0, 0), WIDTH / 2),
        half,
        "level 1 by a blit"
    );
    let mut deep = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, 32, false);
    deep.texture_type = metal2vulkan::metal::TextureType::Type3D;
    deep.depth = 2;
    deep.usage = usage;
    deep.resource_options = ResourceOptions::shared();
    let volume = (surface.new_texture_with_iosurface_plane(&deep, plane)).unwrap();
    let by_cpu = row(|x, c| (x * 3 + c * 5) as u8 | 0x40);
    let second = Region::new_3d(0, 0, 1, WIDTH, 1, 1);
    (volume.replace_region(second, 0, 0, &by_cpu, ROW, ROW)).unwrap();
    assert_eq!(
        blit_row(&volume, 0, 0, Origin::new(0, 0, 1), WIDTH),
        by_cpu,
        "slice 1 of the 3D texture by a blit"
    );
    let mut samples = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, 64, false);
    samples.texture_type = metal2vulkan::metal::TextureType::Type2DMultisample;
    samples.sample_count = 4;
    samples.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    samples.resource_options = ResourceOptions::private();
    (surface.new_texture_with_iosurface_plane(&samples, plane)).unwrap();
    // SAFETY: the surface's `LENGTH` mapped bytes, which no command uses now.
    let now = unsafe {
        std::slice::from_raw_parts(surface.contents().unwrap().as_ptr(), LENGTH as usize)
    };
    assert_eq!(now, &bytes[..], "no byte of the surface changes");
    let notices = heard.lock().unwrap();
    let apart = notices
        .iter()
        .filter(|notice| notice.degradation == Degradation::SurfaceTexelsApart)
        .count();
    assert_eq!(apart, 1, "announced once");
}

/// A 2D array of two slices over a plane of an IOSurface with planes (`linear_plan.rs`'s
/// **Beyond the plane's rows**, 64 by 32 `RGBA8Unorm` at rows of 256 over a 32768-byte surface,
/// as the probe's `bgra1x2.2darray.a2`): Metal keeps its first slice over the plane on every path
/// and lays its second in no one place (an Apple M3 Max under macOS 27.0.1,
/// `metal2vulkan-conformance/probes/surface_texture_envelope.m`), where the runtime keeps the
/// second in the texture's tail, zero when made, announced once as `surface-texels-tail`. Slice 0
/// reads the plane's bytes on every path and what a kernel writes there reaches them; slice 1
/// reads zero as made, and what the CPU, a blit and a kernel write there every path reads back,
/// while no byte of the surface past slice 0 changes and no other buffer's does.
#[test]
fn an_array_over_a_plane_keeps_its_second_slice_in_its_tail() {
    let scratch = Scratch::new("buffer-tail-plane-array");
    let heard = Heard::default();
    let device = hearing_device(&scratch.0, &[], &heard);
    const HEIGHT: u64 = 32;
    const IMAGE: u64 = ROW * HEIGHT;
    // A world of a 64 by 32 2D texture over the plane, whose buffer the array shares.
    let tail = Tail::over_plane(&device, 32768, 0, HEIGHT);
    let plane = IoSurfacePlane::new(0, ROW, true);
    let mut slices = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    slices.texture_type = metal2vulkan::metal::TextureType::Type2DArray;
    slices.array_length = 2;
    slices.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    slices.resource_options = ResourceOptions::shared();
    let array = (tail.buffer.new_texture_with_iosurface_plane(&slices, plane)).unwrap();
    // Each slice as a 2D texture, which the kernel reads and writes.
    let view = |slice: u64| {
        let ty = metal2vulkan::metal::TextureType::Type2D;
        (array.new_texture_view_with_levels_slices(
            PixelFormat::RGBA8Unorm,
            ty,
            0..1,
            slice..slice + 1,
        ))
        .unwrap()
    };
    let views = [view(0), view(1)];
    let cpu_row = |slice: u64, y: u64| {
        let mut bytes = vec![0xEE; ROW as usize];
        let region = Region::new_2d(0, y, WIDTH, 1);
        (array.get_bytes(&mut bytes, ROW, 0, region, 0, slice)).unwrap();
        bytes
    };
    let blit_row = |slice: u64, y: u64| {
        let read = device.new_buffer(ROW, ResourceOptions::shared()).unwrap();
        let command_buffer = tail.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, y, 0), Size::new(WIDTH, 1, 1));
        let copy = blit.copy_from_texture_to_buffer(
            &array,
            slice,
            0,
            origin,
            size,
            &read,
            0,
            ROW,
            ROW,
            BlitOption::None,
        );
        copy.unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: the read buffer's `ROW` mapped bytes, which no command uses now.
        unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), ROW as usize) }
            .to_vec()
    };
    let kernel_row = |slice: u64, y: u64| {
        tail.invert(&views[slice as usize], &tail.plain);
        inverted(&tail.cpu_row(&tail.plain, y))
    };
    // As made: slice 0 the plane's bytes, slice 1 zero, on every path.
    for y in [0, 31] {
        let want = tail.made_row(y);
        assert_eq!(cpu_row(0, y), want, "slice 0 get_bytes {y}");
        assert_eq!(blit_row(0, y), want, "slice 0 blit {y}");
        assert_eq!(kernel_row(0, y), want, "slice 0 kernel {y}");
        let zero = vec![0; ROW as usize];
        assert_eq!(cpu_row(1, y), zero, "slice 1 get_bytes {y}");
        assert_eq!(blit_row(1, y), zero, "slice 1 blit {y}");
        assert_eq!(kernel_row(1, y), zero, "slice 1 kernel {y}");
    }
    // The kernel writes every row of each slice: slice 0's reach the plane's bytes, slice 1's
    // the tail, read back by every path.
    for (slice, mark) in [(0u64, 0x80u8), (1, 0x40)] {
        let written: Vec<Vec<u8>> = (0..HEIGHT)
            .map(|y| row(|x, c| (y * 13 + x * 3 + c * 7 + slice) as u8 | mark))
            .collect();
        for (y, bytes) in written.iter().enumerate() {
            let region = Region::new_2d(0, y as u64, WIDTH, 1);
            (tail
                .plain
                .replace_region(region, 0, 0, &inverted(bytes), ROW, 0))
            .unwrap();
        }
        tail.invert(&tail.plain, &views[slice as usize]);
        for y in [0, 17, 31] {
            let want = &written[y as usize];
            assert_eq!(&cpu_row(slice, y), want, "slice {slice} get_bytes {y}");
            assert_eq!(&blit_row(slice, y), want, "slice {slice} blit {y}");
            assert_eq!(&kernel_row(slice, y), want, "slice {slice} kernel {y}");
        }
        if slice == 0 {
            assert_eq!(
                tail.buffer_bytes(17 * ROW..18 * ROW),
                written[17],
                "slice 0 in place"
            );
        }
    }
    // The CPU writes a row of slice 1: every path reads it, the plane's bytes past slice 0 as
    // they were made.
    let by_cpu = row(|x, c| (x * 5 + c) as u8 | 0x20);
    let region = Region::new_2d(0, 9, WIDTH, 1);
    (array.replace_region(region, 0, 1, &by_cpu, ROW, 0)).unwrap();
    assert_eq!(blit_row(1, 9), by_cpu, "slice 1 blit after replace_region");
    assert_eq!(
        kernel_row(1, 9),
        by_cpu,
        "slice 1 kernel after replace_region"
    );
    let past: Vec<u8> = (IMAGE..2 * IMAGE).map(pattern).collect();
    assert_eq!(
        tail.buffer_bytes(IMAGE..2 * IMAGE),
        past,
        "the plane past slice 0"
    );
    assert!(tail.around_kept(), "no write reaches another buffer");
    let notices = heard.lock().unwrap();
    let tails = notices
        .iter()
        .filter(|notice| notice.degradation == Degradation::SurfaceTexelsTail)
        .count();
    assert_eq!(tails, 1, "announced once");
}

/// Over a `Private` buffer, whose shadow copies back rather than merging, a blit's write past the
/// length reads back through the kernel, and the kernel's through a blit.
#[test]
fn a_private_buffer_s_texture_past_its_length_keeps_its_tail() {
    let scratch = Scratch::new("buffer-tail-private");
    let device = device(&scratch.0);
    let tail = Tail::new(&device, (4096, ResourceOptions::private()), 0, 17);
    assert_eq!(
        tail.blit_row(16),
        vec![0; ROW as usize],
        "made zero past the length"
    );
    let by_blit = row(|x, c| (x * 7 + c) as u8 | 0x40);
    tail.blit_write(16, &by_blit);
    assert_eq!(tail.kernel_row(16), by_blit, "kernel after the blit");
    let written: Vec<Vec<u8>> = (0..17)
        .map(|y| row(|x, c| (y * 5 + x + c * 2) as u8 | 0x80))
        .collect();
    tail.kernel_write(&written);
    for y in [0, 15, 16] {
        assert_eq!(tail.blit_row(y), written[y as usize], "blit {y}");
    }
    assert!(tail.around_kept(), "no write reaches another buffer");
}

/// A 2D array whose second slice lies wholly past its buffer's length (`R32Uint`, 4 by 3 at rows
/// of 16 over 128 bytes, its slices 128 bytes apart, as Metal strides them): its first slice is
/// the buffer's bytes, its second the tail's, zero when made, and what the CPU writes there a
/// blit reads, the buffer's bytes and every other buffer's as they were.
#[test]
fn an_array_s_slice_past_the_length_lies_in_its_tail() {
    let scratch = Scratch::new("buffer-tail-array");
    let device = device(&scratch.0);
    const LENGTH: u64 = 128;
    const SLICE: u64 = 48;
    let tail = Tail::new(&device, (LENGTH, ResourceOptions::shared()), 0, 1);
    let mut rows = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 4, 3, false);
    rows.texture_type = metal2vulkan::metal::TextureType::Type2DArray;
    rows.array_length = 2;
    rows.resource_options = ResourceOptions::shared();
    let array = tail.buffer.new_linear_texture(&rows, 0, 16, 0).unwrap();
    let slice = |slice: u64| {
        let mut bytes = vec![0; SLICE as usize];
        let region = Region::new_2d(0, 0, 4, 3);
        array
            .get_bytes(&mut bytes, 16, 0, region, 0, slice)
            .unwrap();
        bytes
    };
    assert_eq!(
        slice(0),
        (0..SLICE).map(pattern).collect::<Vec<u8>>(),
        "slice 0"
    );
    assert_eq!(slice(1), vec![0; SLICE as usize], "slice 1, as made");
    let written: Vec<u8> = (0..SLICE).map(|at| at as u8 | 0x80).collect();
    let region = Region::new_2d(0, 0, 4, 3);
    (array.replace_region(region, 0, 1, &written, 16, 0)).unwrap();
    assert_eq!(slice(1), written, "slice 1, as written");
    let read = device.new_buffer(SLICE, ResourceOptions::shared()).unwrap();
    let command_buffer = tail.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let (origin, size) = (Origin::new(0, 0, 0), Size::new(4, 3, 1));
    let copy = blit.copy_from_texture_to_buffer(
        &array,
        1,
        0,
        origin,
        size,
        &read,
        0,
        16,
        SLICE,
        BlitOption::None,
    );
    copy.unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    // SAFETY: the read buffer's `SLICE` mapped bytes, which no command uses now.
    let blitted =
        unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), SLICE as usize) };
    assert_eq!(blitted, &written[..], "slice 1 through a blit");
    assert_eq!(
        tail.buffer_bytes(0..LENGTH),
        (0..LENGTH).map(pattern).collect::<Vec<u8>>()
    );
    assert!(tail.around_kept(), "no write reaches another buffer");
}

/// The texels of texture buffers: `ob1_texel_buffer` (the owned fixture
/// `validation/fixtures/public/engine/ob1_texel_buffer.ll`), whose thread `t` of eight reads
/// texel `idx[t]` of its first texture buffer into `out[t]` and writes `0xA0000000 | t` to texel
/// `idx[8 + t]` of its second.
pub(super) struct TexelKernel {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    device: Device,
}

impl TexelKernel {
    pub(super) fn new(device: &Device) -> TexelKernel {
        let library = library(device, "engine/ob1_texel_buffer.ll");
        let function = library.new_function_with_name("ob1_texel_buffer").unwrap();
        let options = PipelineOptions::default();
        TexelKernel {
            queue: device.new_command_queue().unwrap(),
            pipeline: device
                .new_compute_pipeline_state_with_function(&function, &options)
                .unwrap(),
            device: device.clone(),
        }
    }

    /// A texture buffer of `width` `R32Uint` texels at `offset` of `buffer`, read and written by
    /// shaders.
    pub(super) fn over(buffer: &Buffer, width: u64, offset: u64) -> Texture {
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let descriptor = TextureDescriptor::texture_buffer(
            PixelFormat::R32Uint,
            width,
            buffer.resource_options(),
            usage,
        );
        buffer.new_texture(&descriptor, offset, width * 4).unwrap()
    }

    /// One dispatch: texels `reads` of `from` into the answer, `0xA0000000 | t` into texel
    /// `writes[t]` of `into`.
    pub(super) fn run(
        &self,
        (from, reads): (&Texture, [u32; 8]),
        (into, writes): (&Texture, [u32; 8]),
    ) -> Vec<u32> {
        let words: Vec<u8> = reads
            .iter()
            .chain(&writes)
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let shared = ResourceOptions::shared();
        let idx = self.device.new_buffer(64, shared).unwrap();
        let out = self.device.new_buffer(32, shared).unwrap();
        // SAFETY: the buffer's 64 mapped bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), idx.contents().unwrap().as_ptr(), 64)
        };
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.pipeline).unwrap();
        encoder.set_buffer(Some(&idx), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_texture(Some(from), 0).unwrap();
        encoder.set_texture(Some(into), 1).unwrap();
        (encoder.dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: the answer's 32 mapped bytes, which no command uses now.
        let bytes = unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), 32) };
        bytes
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }

    /// Every texel of `texture`, `width` of them, through `get_bytes` and through a blit.
    fn cpu_and_blit(&self, texture: &Texture, width: u64) -> (Vec<u32>, Vec<u32>) {
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes
                .chunks(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect()
        };
        let mut cpu = vec![0; width as usize * 4];
        let row = Region::new_1d(0, width);
        texture
            .get_bytes(&mut cpu, width * 4, 0, row, 0, 0)
            .unwrap();
        let read = self
            .device
            .new_buffer(width * 4, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(width, 1, 1));
        let copy = blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            &read,
            0,
            width * 4,
            width * 4,
            BlitOption::None,
        );
        copy.unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: the read buffer's mapped bytes, which no command uses now.
        let blitted = unsafe {
            std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), width as usize * 4)
        };
        (words(&cpu), words(blitted))
    }
}

/// A texture buffer whose texels run past its buffer's length, as Metal makes one (64 `R32Uint`
/// texels from 3968 of a 4096-byte buffer, the last 32 past it; and from 4096, every one past
/// it): its texels inside the length are the buffer's words, those past it its tail's, zero when
/// made, and what a kernel, the CPU and a blit write there each path reads, the buffer's words
/// and every other buffer's as they were.
#[test]
fn a_texture_buffer_past_its_buffer_s_length_keeps_those_texels_in_its_tail() {
    let scratch = Scratch::new("buffer-tail-texel-buffer");
    let device = device(&scratch.0);
    let kernel = TexelKernel::new(&device);
    let word = |at: u64| 0x5000_0000 | at as u32;
    for (offset, inside) in [(3968u64, 32usize), (4096, 0)] {
        let tail = Tail::new(&device, (4096, ResourceOptions::shared()), 0, 1);
        let words: Vec<u8> = (0..1024u64)
            .flat_map(|at| word(at * 4).to_le_bytes())
            .collect();
        // SAFETY: the buffer's 4096 mapped bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                tail.buffer.contents().unwrap().as_ptr(),
                4096,
            )
        };
        let texels = TexelKernel::over(&tail.buffer, 64, offset);
        let other = TexelKernel::over(
            &device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            64,
            0,
        );
        let made: Vec<u32> = (0..64)
            .map(|t| {
                if t < inside {
                    word(offset + 4 * t as u64)
                } else {
                    0
                }
            })
            .collect();
        let (cpu, blit) = kernel.cpu_and_blit(&texels, 64);
        assert_eq!(
            (&cpu, &blit),
            (&made, &made),
            "{offset}: as made, CPU and blit"
        );
        let reads = [0, 31, 32, 63, 1, 30, 33, 62];
        let got = kernel.run((&texels, reads), (&other, [0; 8]));
        let want: Vec<u32> = reads.iter().map(|&t| made[t as usize]).collect();
        assert_eq!(got, want, "{offset}: as made, a kernel");
        // A kernel writes texels 28 to 35, across the length where it lies inside the texels.
        let writes = [28, 29, 30, 31, 32, 33, 34, 35];
        kernel.run((&other, [0; 8]), (&texels, writes));
        let mut written = made.clone();
        for (t, &at) in writes.iter().enumerate() {
            written[at as usize] = 0xA000_0000 | t as u32;
        }
        let (cpu, blit) = kernel.cpu_and_blit(&texels, 64);
        assert_eq!(
            (&cpu, &blit),
            (&written, &written),
            "{offset}: the kernel's, CPU and blit"
        );
        let got = kernel.run((&texels, writes), (&other, [0; 8]));
        assert_eq!(got, &written[28..36], "{offset}: the kernel's, a kernel");
        // The CPU writes texels 40 to 47, past the length: a kernel and a blit read them.
        let by_cpu: Vec<u8> = (0..8u32)
            .flat_map(|t| (0x7000_0000 | t).to_le_bytes())
            .collect();
        (texels.replace_region(Region::new_1d(40, 8), 0, 0, &by_cpu, 32, 0)).unwrap();
        for t in 0..8 {
            written[40 + t] = 0x7000_0000 | t as u32;
        }
        let got = kernel.run(
            (&texels, [40, 41, 42, 43, 44, 45, 46, 47]),
            (&other, [0; 8]),
        );
        assert_eq!(got, &written[40..48], "{offset}: the CPU's, a kernel");
        let (cpu, blit) = kernel.cpu_and_blit(&texels, 64);
        assert_eq!(
            (&cpu, &blit),
            (&written, &written),
            "{offset}: the CPU's, CPU and blit"
        );
        // The buffer's words: the texels' inside the length as written, every other as it was.
        let mut buffer_words: Vec<u32> = (0..1024u64).map(|at| word(at * 4)).collect();
        for t in 0..inside {
            buffer_words[(offset / 4) as usize + t] = written[t];
        }
        let now = tail.buffer_bytes(0..4096);
        let now: Vec<u32> = now
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        assert_eq!(now, buffer_words, "{offset}: the buffer's words");
        assert!(
            tail.around_kept(),
            "{offset}: no write reaches another buffer"
        );
    }
}

/// Every path reads `tail`'s texture as made (`rows` among its rows): its bytes before `kept`, where
/// the buffer's memory holding the texels ends, as that memory holds them, and those past it zero;
/// and a kernel's write of every row reads back on every path, its texels before `kept` landing in
/// the buffer's memory; the CPU's write of the buffer's bytes under the first row, through its
/// `contents`, every path then reads through the texture; and no write reaches another buffer.
/// Texels past `kept` lie in a tail, the texture a shadow image whose copies in and back go
/// through it (`buffer_texture.rs`'s **Tails**), counted in `shadow_copies`; a texture that ends
/// at `kept` or short of it keeps none. The kernel's rows, as written.
fn coherent_past_length(
    device: &Device,
    tail: &Tail,
    rows: [u64; 4],
    (what, kept): (&str, u64),
) -> Vec<Vec<u8>> {
    let before = device.work_counts().shadow_copies;
    for y in rows {
        let want = tail.row_kept(y, kept);
        let cpu = tail.cpu_row(&tail.texture, y);
        assert_eq!(cpu, want, "{what}: as made, get_bytes {y}");
        assert_eq!(tail.blit_row(y), want, "{what}: as made, blit {y}");
        assert_eq!(tail.kernel_row(y), want, "{what}: as made, kernel {y}");
    }
    let written: Vec<Vec<u8>> = (0..tail.height)
        .map(|y| row(|x, c| (y * 7 + x * 3 + c) as u8 | 0x80))
        .collect();
    tail.kernel_write(&written);
    for y in rows {
        let want = &written[y as usize];
        let cpu = tail.cpu_row(&tail.texture, y);
        assert_eq!(&cpu, want, "{what}: the kernel's, get_bytes {y}");
        assert_eq!(&tail.blit_row(y), want, "{what}: the kernel's, blit {y}");
        assert_eq!(
            &tail.kernel_row(y),
            want,
            "{what}: the kernel's, kernel {y}"
        );
    }
    // The texels before `kept` the kernel wrote: the buffer's memory's bytes.
    for (y, written) in (0..tail.height).zip(&written) {
        let start = tail.offset + y * tail.pitch;
        let inside = kept.min(start + ROW);
        if inside > start {
            let want = &written[..(inside - start) as usize];
            let bytes = tail.memory_bytes(start..inside);
            assert_eq!(bytes, want, "{what}: the buffer's memory of row {y}");
        }
    }
    // The CPU writes the buffer's bytes under row 0, inside the length: the texture reads them.
    let by_cpu = row(|x, c| (x * 5 + c * 3) as u8 | 0x40);
    let at = tail
        .buffer
        .contents()
        .unwrap()
        .as_ptr()
        .wrapping_add(tail.offset as usize);
    // SAFETY: row 0's `ROW` bytes lie inside the buffer's length, mapped, which no command uses
    // now.
    unsafe { std::ptr::copy_nonoverlapping(by_cpu.as_ptr(), at, by_cpu.len()) };
    let cpu = tail.cpu_row(&tail.texture, 0);
    assert_eq!(cpu, by_cpu, "{what}: the buffer's CPU write, get_bytes");
    assert_eq!(
        tail.blit_row(0),
        by_cpu,
        "{what}: the buffer's CPU write, blit"
    );
    assert_eq!(
        tail.kernel_row(0),
        by_cpu,
        "{what}: the buffer's CPU write, kernel"
    );
    let copies = device.work_counts().shadow_copies - before;
    let texels = tail.texture.buffer_texels().unwrap();
    match tail.end() > kept {
        true => {
            let past = texels.past().map(|past| past.length);
            assert_eq!(
                past,
                Some(kept),
                "{what}: its texels past {kept} in its tail"
            );
            assert!(copies > 0, "{what}: a shadow, its copies through its tail");
        }
        false => assert!(
            texels.past().is_none(),
            "{what}: no tail, its texels in the buffer's memory"
        ),
    }
    assert!(
        tail.around_kept(),
        "{what}: no write reaches another buffer"
    );
    written
}

/// Metal's nil for a texture over a buffer that ends at `end`, past where the buffer's
/// allocation ends at `allocation` (`linear_plan.rs`'s **Length**).
fn past_allocation(end: u64, allocation: u64) -> Error {
    let problem = BufferTextureProblem::PastAllocation { end, allocation };
    Error::Nil(Nil::new("new_texture", Problem::BufferTexture(problem)))
}

/// `tail`'s texture's descriptor, `height` rows tall.
fn shaped(tail: &Tail, height: u64) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = tail.buffer.resource_options();
    descriptor
}

/// `tail`'s texture again, `height` rows tall, at its offset and pitch.
fn taller(tail: &Tail, height: u64) -> Result<Texture, Error> {
    (tail.buffer).new_texture(&shaped(tail, height), tail.offset, tail.pitch)
}

/// Where `tail`'s texture lies, as its making answers it: over a buffer `long` bytes long, which
/// holds it inside its length, where `long` is `Some`; over its own buffer where `None`.
fn placement(device: &Device, tail: &Tail, long: Option<u64>) -> TexturePlacement {
    let shape = shaped(tail, tail.height);
    let buffer = match long {
        Some(long) => device
            .new_buffer(long, tail.buffer.resource_options())
            .unwrap(),
        None => tail.buffer.clone(),
    };
    (buffer.texture_placement(&shape, tail.offset, tail.pitch)).unwrap()
}

/// A 2D texture whose last texel ends at its buffer's allocation, its last row's padding past it
/// (`linear_plan.rs`'s **Length**: 64 by 64 from offset 256, rows 512 apart, over 20000 bytes,
/// the last texel at 32768 and the rows' end at 33024), is made as an Apple M3 Max under macOS
/// 27.0.1 makes it, in the buffer's allocation, which holds it whole: no tail, and where it lies
/// as it would over a buffer holding it inside its length. A row more ends past the allocation,
/// Metal's nil.
#[test]
fn a_texture_whose_last_texel_ends_at_the_allocation_is_made() {
    let scratch = Scratch::new("buffer-tail-last-texel");
    let device = device(&scratch.0);
    let target = || device.new_buffer(20000, ResourceOptions::shared()).unwrap();
    let tail = Tail::over(&device, target, (256, 64, 512));
    let inside = placement(&device, &tail, Some(33024));
    assert_eq!(
        placement(&device, &tail, None),
        inside,
        "as inside the length"
    );
    coherent_past_length(&device, &tail, [0, 38, 39, 63], ("last texel", 32768));
    assert_eq!(taller(&tail, 65).err(), Some(past_allocation(33280, 32768)));
}

/// A texture over a buffer of 16 KiB or more that the device made, whose texels run past its
/// length into its allocation (64 by 128 over 20000 bytes, packed rows, ending at 32768): Metal's
/// texels there are the allocation's, zero when made, which every path reads and writes alike, and
/// the runtime's lie there too, in the buffer's memory, its `VkBuffer` its length rounded up to
/// 16 KiB (`Device::placed_new_buffer`). It lies as it would over a buffer holding it inside its
/// length, with no tail; where that is an image over the buffer's memory, no copy is taken. The
/// buffer's own reach stays its length: a blit of its bytes past the length is refused.
#[test]
fn a_buffer_s_texture_to_its_allocation_lies_in_its_memory() {
    let scratch = Scratch::new("buffer-tail-allocation");
    let device = device(&scratch.0);
    let before = device.work_counts();
    let target = || device.new_buffer(20000, ResourceOptions::shared()).unwrap();
    let tail = Tail::over(&device, target, (0, 128, ROW));
    let padding = tail.memory_bytes(20000..32768);
    assert!(padding.iter().all(|&byte| byte == 0), "zero when made");
    let inside = placement(&device, &tail, Some(32768));
    let asked = placement(&device, &tail, None);
    assert_eq!(asked, inside, "as inside the length");
    let past = TexturePlacement::Shadow(TextureShadowReason::PastLength);
    assert_ne!(asked, past, "no tail");
    let rows = [0, 77, 78, 127];
    coherent_past_length(&device, &tail, rows, ("allocation", 32768));
    let after = device.work_counts();
    let aliases = tail.texture.aliases_buffer();
    eprintln!(
        "BUFFER-TAIL on {}: a texture to its buffer's allocation: asked {asked:?}, an image over \
         the buffer's memory {aliases}, {} shadowed, {} copies",
        device.name(),
        after.shadowed_textures - before.shadowed_textures,
        after.shadow_copies - before.shadow_copies,
    );
    if aliases {
        assert_eq!(
            after.shadow_copies - before.shadow_copies,
            0,
            "in place: no copy"
        );
    }
    let read = device.new_buffer(32, ResourceOptions::shared()).unwrap();
    let command_buffer = tail.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let copy = blit.copy_from_buffer_to_buffer(&tail.buffer, 20000 - 16, &read, 0, 32);
    assert!(copy.is_err(), "a blit past the buffer's length: {copy:?}");
    blit.end_encoding().unwrap();
}

/// A buffer of a heap takes a texture to its heap's size from its own start, as an Apple M3 Max
/// under macOS 27.0.1 makes one (`linear_plan.rs`'s **Length**): 64 by 256 over a 16384-byte
/// buffer at the start of a 64 KiB placement heap, ending 65536 bytes from the buffer's start.
/// Its texels past the length are the heap's bytes, as Metal's are: those of the heap's other
/// buffer, at 32768, read through the texture as that buffer holds them, and the kernel's writes
/// over them land in it. No tail. A row more is Metal's nil.
#[test]
fn a_heap_s_buffer_takes_a_texture_to_its_heap_s_size() {
    const HEAP: u64 = 64 << 10;
    let scratch = Scratch::new("buffer-tail-heap");
    let device = device(&scratch.0);
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::shared());
    let heap = device.new_heap(&descriptor).unwrap();
    assert_eq!(heap.size(), HEAP);
    let shared = ResourceOptions::shared();
    let target = || heap.new_buffer_at_offset(16384, shared, 0).unwrap();
    let tail = Tail::over(&device, target, (0, HEAP / ROW, ROW));
    let at = 32768;
    let neighbour = heap.new_buffer_at_offset(4096, shared, at).unwrap();
    let neighbour_at = neighbour.contents().unwrap().as_ptr();
    // SAFETY: the neighbour's 4096 mapped bytes, which no command uses.
    unsafe { std::ptr::write_bytes(neighbour_at, AROUND, 4096) };
    let rows = [0, 63, at / ROW, HEAP / ROW - 1];
    let written = coherent_past_length(&device, &tail, rows, ("heap", HEAP));
    // SAFETY: as above.
    let neighbour_bytes = unsafe { std::slice::from_raw_parts(neighbour_at, 4096) };
    let texels = written.concat();
    let at = at as usize;
    assert_eq!(
        neighbour_bytes,
        &texels[at..at + 4096],
        "the heap's other buffer holds the texels written over its bytes"
    );
    let nil = past_allocation(HEAP + ROW, HEAP);
    assert_eq!(taller(&tail, HEAP / ROW + 1).err(), Some(nil));
}

/// A buffer of a heap past its start takes a texture to the heap's size from its own start, past
/// the heap's end (64 by 256 over a 16384-byte buffer at 16384 of a 64 KiB placement heap, ending
/// 65536 bytes from that buffer's start): its texels inside the heap are the heap's bytes, and those
/// past its end lie in the texture's tail, zero when made, which no other memory shares: the
/// heap's first buffer keeps its bytes.
#[test]
fn a_texture_past_its_heap_s_end_keeps_those_texels_in_its_tail() {
    const HEAP: u64 = 64 << 10;
    let scratch = Scratch::new("buffer-tail-heap-end");
    let device = device(&scratch.0);
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::shared());
    let heap = device.new_heap(&descriptor).unwrap();
    let shared = ResourceOptions::shared();
    let first = heap.new_buffer_at_offset(16384, shared, 0).unwrap();
    let first_at = first.contents().unwrap().as_ptr();
    // SAFETY: the first buffer's 16384 mapped bytes, which no command uses.
    unsafe { std::ptr::write_bytes(first_at, AROUND, 16384) };
    let start = 16384;
    let target = || heap.new_buffer_at_offset(16384, shared, start).unwrap();
    let tail = Tail::over(&device, target, (0, HEAP / ROW, ROW));
    let kept = HEAP - start;
    let rows = [0, 63, kept / ROW - 1, kept / ROW];
    coherent_past_length(&device, &tail, rows, ("past the heap's end", kept));
    // SAFETY: as above.
    let first_bytes = unsafe { std::slice::from_raw_parts(first_at, 16384) };
    assert!(
        first_bytes.iter().all(|&byte| byte == AROUND),
        "no write reaches the heap's first buffer"
    );
}

/// A no-copy buffer takes a texture to its pages from its own start, its pointer's offset in its
/// 16 KiB page plus its length rounded up to 16 KiB, as an Apple M3 Max under macOS 27.0.1 makes
/// one (`linear_plan.rs`'s **Length**): 64 by 128 over 16384 bytes starting 4096 bytes short of
/// a 16 KiB page's end, ending 32768 bytes from the buffer's start, its rows past the length in
/// its tail, as the pages past it are the caller's: no write reaches the caller's bytes around
/// the buffer. A row more is Metal's nil.
#[test]
fn a_no_copy_buffer_takes_a_texture_to_its_pages() {
    const MAPPING: usize = 64 << 10;
    const PAGE: usize = 16 << 10;
    // Declared before the device, so unmapped after every buffer over it.
    let mapping = Mapping::new(MAPPING);
    let scratch = Scratch::new("buffer-tail-no-copy");
    let device = device(&scratch.0);
    let start = mapping.0.as_ptr();
    let lead = start.addr().next_multiple_of(PAGE) - start.addr() + PAGE - 4096;
    let pointer = NonNull::new(start.wrapping_add(lead)).unwrap();
    let target = || {
        // SAFETY: the mapping's bytes, which outlive the buffer (declared before the device).
        let memory = unsafe { HostMemory::new(pointer) };
        let options = ResourceOptions::shared();
        // SAFETY: as above, 16384 bytes of the mapping from `lead`, mapped read and write.
        unsafe { device.new_buffer_with_bytes_no_copy(memory, 16384, options, None) }.unwrap()
    };
    let tail = Tail::over(&device, target, (0, 128, ROW));
    coherent_past_length(&device, &tail, [0, 63, 64, 127], ("no-copy", 16384));
    // SAFETY: the mapping's bytes, which no command uses now.
    let all = unsafe { std::slice::from_raw_parts(start, MAPPING) };
    let around = all[..lead].iter().chain(&all[lead + 16384..]);
    assert!(
        around.into_iter().all(|&byte| byte == AROUND),
        "no write reaches the caller's bytes around the buffer"
    );
    assert_eq!(
        taller(&tail, 129).err(),
        Some(past_allocation(33024, 32768))
    );
}

/// Anonymous private pages of the test's own, each byte [`AROUND`], unmapped when dropped.
struct Mapping(NonNull<u8>, usize);

impl Mapping {
    fn new(length: usize) -> Mapping {
        let start = super::host_os::private(length);
        // SAFETY: the mapping's bytes, which nothing uses yet.
        unsafe { std::ptr::write_bytes(start.as_ptr(), AROUND, length) };
        Mapping(start, length)
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // The pages `new` made, which no object lies in any more.
        super::host_os::release(self.0, self.1);
    }
}
