//! A 2D texture over a buffer whose texels run past the buffer's length (`buffer_texture.rs`'s
//! **Tails**): plain Metal makes it, and every path of Metal's takes the texels past the length
//! as one memory, which each reads what each wrote to (an Apple M3 Max under macOS 27.0.1,
//! `metal2vulkan-conformance/probes/texels_past_buffer_length.m`). The runtime makes it too: its
//! texels inside the length are the buffer's bytes, and those past it lie in the texture's tail,
//! zero when made.
//!
//! Each case writes the texels through one path and reads them back through every other: a
//! kernel (`invert_texels`, the owned fixture `validation/fixtures/public/examples/
//! runtime_examples.ll`, which writes each texel of its first texture inverted, its alpha kept,
//! into its second), a blit, and the CPU (`replace_region`, `get_bytes`). The buffer's bytes are
//! read through its `contents`, or by a blit of the buffer for a `Private` one. Buffers made
//! around the texture's are filled with a pattern no write here makes, and keep it: no write
//! past the length reaches memory but the tail. The shapes are reims-vgpu's (64 by 17 at rows of
//! 256 over 4096 bytes, and 64 by 16 from offset 2048), one whose last row crosses the length
//! (64 by 80 over 20000 bytes, whose row 78 holds 32 of the buffer's bytes), and one wholly past
//! it (64 by 1 from offset 4096).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    BlitOption, Buffer, CommandQueue, ComputePipelineState, Device, Origin, PipelineOptions,
    Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

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
        let around_buffer = || {
            let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            // SAFETY: the buffer's 4096 mapped bytes, which no command uses yet.
            unsafe { std::ptr::write_bytes(buffer.contents().unwrap().as_ptr(), AROUND, 4096) };
            buffer
        };
        let mut around: Vec<Buffer> = (0..8).map(|_| around_buffer()).collect();
        let buffer = device.new_buffer(length, options).unwrap();
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
        let texture = buffer.new_texture(&descriptor, offset, ROW).unwrap();
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

    /// Row `y`'s bytes as the buffer holds them, those past its length zero.
    fn made_row(&self, y: u64) -> Vec<u8> {
        let start = self.offset + y * ROW;
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
/// crosses the length 32 bytes in, and its first 8 texels are the buffer's), and 64 by 1 wholly
/// past a 4096-byte buffer. Each reads as made on every path, and a kernel's write of every row
/// reads back on every path, the bytes inside the length in the buffer.
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
