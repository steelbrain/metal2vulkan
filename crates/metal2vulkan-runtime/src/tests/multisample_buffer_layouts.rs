//! Blits between a buffer and a multisampled texture at layouts off the pixel (a texel's samples'
//! bytes) copy the bytes plain Metal copies: each row where Metal places it
//! (`sample_interleave::placed_rows`), through a staging buffer where a row's offset in the
//! buffer's `VkBuffer` is off a sample, announced as `staged-texture-copy`. Every case's rows are
//! what plain Metal placed on an Apple M3 Max under macOS 27.0.1, for its format, sample count,
//! width, offset and `bytesPerRow`, four rows tall, measured again by
//! `metal2vulkan-conformance/probes/multisample_buffer_copies.m` at the texture's usage here
//! (render target and shader read) and with `PixelFormatView`.
//!
//! Into the texture, the copy back out at the pixel's own layout gives each placed row's bytes
//! (rows that overlap included); out of it, the buffer holds the texture's rows at the placed
//! offsets and its other bytes as they were. A no-copy buffer over a pointer 4 bytes into a page
//! (its byte 0 4 bytes into its `VkBuffer`) copies the same. A layout Metal places by a rule the
//! runtime does not hold is its typed gap (`MultisampledBufferLayout`), and a pitch below the
//! pixel from an offset that is a multiple of 16, from which Metal reads bytes the buffer does
//! not hold, is refused (`BytesPerRow`).
//!
//! Each case prints a `MS-LAYOUT` line. Device tests: compiled only under the validation crate's
//! GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, CopyProblem, Degradation, Device,
    DeviceKind, DeviceOptions, Error, HostMemory, Invalid, MessageSink, Origin, Problem,
    ResourceOptions, RuntimeGap, ShaderCacheOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::alloc::Layout;
use std::collections::BTreeSet;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The rows of every case.
const ROWS: u64 = 4;

/// The buffers' length.
const LENGTH: u64 = 4096;

/// A case: the format and its sample's bytes, the sample count, the width, the layout as
/// `(offset, bytesPerRow)`, the rows' offsets plain Metal placed, and whether the copy stages.
struct Case {
    format: (PixelFormat, u64),
    samples: u64,
    width: u64,
    layout: (u64, u64),
    rows: [u64; ROWS as usize],
    staged: bool,
}

impl Case {
    fn row(&self) -> u64 {
        self.width * self.samples * self.format.1
    }
}

/// Copies into the texture.
const INTO: [Case; 5] = [
    // Rows 16 bytes apart, each 32 long: they overlap.
    Case {
        format: (PixelFormat::R32Uint, 4),
        samples: 4,
        width: 2,
        layout: (5, 20),
        rows: [4, 20, 36, 52],
        staged: false,
    },
    Case {
        format: (PixelFormat::RG32Uint, 8),
        samples: 2,
        width: 5,
        layout: (28, 143),
        rows: [28, 156, 284, 412],
        staged: true,
    },
    Case {
        format: (PixelFormat::RGBA32Uint, 16),
        samples: 4,
        width: 1,
        layout: (4, 64),
        rows: [4, 68, 132, 196],
        staged: true,
    },
    Case {
        format: (PixelFormat::RGBA32Uint, 16),
        samples: 2,
        width: 1,
        layout: (7, 32),
        rows: [6, 38, 70, 102],
        staged: true,
    },
    // A pitch below the pixel from an offset off 16: every row at the first.
    Case {
        format: (PixelFormat::R16Uint, 2),
        samples: 4,
        width: 1,
        layout: (4, 4),
        rows: [4, 4, 4, 4],
        staged: false,
    },
];

/// Copies out of the texture.
const OUT: [Case; 4] = [
    Case {
        format: (PixelFormat::RG32Uint, 8),
        samples: 2,
        width: 4,
        layout: (7, 64),
        rows: [4, 68, 132, 196],
        staged: true,
    },
    Case {
        format: (PixelFormat::RGBA32Uint, 16),
        samples: 2,
        width: 1,
        layout: (22, 32),
        rows: [22, 54, 86, 118],
        staged: true,
    },
    Case {
        format: (PixelFormat::RGBA32Uint, 16),
        samples: 4,
        width: 1,
        layout: (21, 64),
        rows: [20, 84, 148, 212],
        staged: true,
    },
    Case {
        format: (PixelFormat::R32Uint, 4),
        samples: 4,
        width: 2,
        layout: (0, 54),
        rows: [0, 48, 96, 144],
        staged: false,
    },
];

/// The device the cfg selects, its shader cache in `cache`, announcing `staged-texture-copy` at
/// every occurrence, and the count of those it announced.
fn device(cache: &Path) -> (Device, Arc<AtomicUsize>) {
    let staged = Arc::new(AtomicUsize::new(0));
    let heard = Arc::clone(&staged);
    let sink = MessageSink::new(move |message: &crate::DebugMessage| {
        let notice = message.degradation.as_ref();
        if notice.is_some_and(|notice| notice.degradation == Degradation::StagedTextureCopy) {
            heard.fetch_add(1, Ordering::SeqCst);
        }
    });
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(sink),
        announce_each: BTreeSet::from([Degradation::StagedTextureCopy]),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), staged)
}

/// `length` bytes that name their place and `seed`.
fn pattern(seed: u8, length: u64) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8) ^ (i >> 8) as u8 ^ seed.wrapping_mul(29))
        .collect()
}

/// A buffer's bytes, read back with no command buffer using it.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with it; no command buffer
    // uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A shared buffer of `bytes`.
fn shared(device: &Device, bytes: &[u8]) -> Buffer {
    (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
}

/// A `samples`-sample texture of `format`, `width` by [`ROWS`]; `None` where the device makes
/// none.
fn texture(device: &Device, format: PixelFormat, samples: u64, width: u64) -> Option<Texture> {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, ROWS, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = samples;
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).ok()
}

/// Copy `buffer`'s bytes laid out as `(offset, bytesPerRow)` into `texture`, `width` wide, or
/// out of it (`into_texture` false), in a command buffer of `queue`'s run to completion.
fn copy(
    queue: &CommandQueue,
    (buffer, offset, pitch): (&Buffer, u64, u64),
    texture: &Texture,
    width: u64,
    into_texture: bool,
) -> Result<(), Error> {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(width, ROWS, 1);
    let origin = Origin::new(0, 0, 0);
    let copied = match into_texture {
        true => blit.copy_from_buffer_to_texture(
            buffer,
            offset,
            pitch,
            0,
            size,
            texture,
            0,
            0,
            origin,
            BlitOption::None,
        ),
        false => blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            origin,
            size,
            buffer,
            offset,
            pitch,
            0,
            BlitOption::None,
        ),
    };
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    copied
}

/// What a case did.
#[derive(Debug, PartialEq)]
enum Outcome {
    Ran,
    /// The device runs no multisampled buffer copy (`MultisampledBufferCopy`), or makes no such
    /// texture.
    Skipped,
}

/// A copy into the texture at the case's layout from `source`, then back out at the pixel's own
/// layout: each row holds the bytes at its placed offset.
fn into(queue: &CommandQueue, device: &Device, case: &Case, source: &Buffer) -> Outcome {
    let (format, _) = case.format;
    let Some(texture) = texture(device, format, case.samples, case.width) else {
        return Outcome::Skipped;
    };
    let (offset, pitch) = case.layout;
    match copy(queue, (source, offset, pitch), &texture, case.width, true) {
        Ok(()) => {}
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => return Outcome::Skipped,
        Err(other) => panic!("{format:?} x{} into: {other}", case.samples),
    }
    let row = case.row();
    let out = shared(device, &vec![0; (row * ROWS) as usize]);
    copy(queue, (&out, 0, row), &texture, case.width, false).unwrap();
    let held = bytes(source);
    let want: Vec<u8> = (case.rows.iter())
        .flat_map(|&at| held[at as usize..(at + row) as usize].to_vec())
        .collect();
    assert_eq!(bytes(&out), want, "{format:?} x{} into", case.samples);
    Outcome::Ran
}

/// A copy out of the texture, filled at the pixel's own layout, into `destination` at the
/// case's layout: the texture's rows at the placed offsets, the other bytes as they were.
fn out(queue: &CommandQueue, device: &Device, case: &Case, destination: &Buffer) -> Outcome {
    let (format, _) = case.format;
    let Some(texture) = texture(device, format, case.samples, case.width) else {
        return Outcome::Skipped;
    };
    let row = case.row();
    let rows = pattern(7, row * ROWS);
    match copy(
        queue,
        (&shared(device, &rows), 0, row),
        &texture,
        case.width,
        true,
    ) {
        Ok(()) => {}
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)) => return Outcome::Skipped,
        Err(other) => panic!("{format:?} x{} fill: {other}", case.samples),
    }
    let mut want = bytes(destination);
    let (offset, pitch) = case.layout;
    copy(
        queue,
        (destination, offset, pitch),
        &texture,
        case.width,
        false,
    )
    .unwrap();
    for (y, &at) in case.rows.iter().enumerate() {
        let from = y * row as usize;
        want[at as usize..(at + row) as usize].copy_from_slice(&rows[from..from + row as usize]);
    }
    assert_eq!(bytes(destination), want, "{format:?} x{} out", case.samples);
    Outcome::Ran
}

/// Run `case` (`into_texture` from the buffer) on a fresh buffer of `make`'s, checking that a
/// staging case announced `staged-texture-copy` and no other did.
fn run(
    (queue, device, staged): (&CommandQueue, &Device, &AtomicUsize),
    case: &Case,
    into_texture: bool,
    make: &dyn Fn(&[u8]) -> Buffer,
) -> Outcome {
    let before = staged.load(Ordering::SeqCst);
    let buffer = make(&pattern(case.samples as u8, LENGTH));
    let outcome = match into_texture {
        true => into(queue, device, case, &buffer),
        false => out(queue, device, case, &buffer),
    };
    let announced = staged.load(Ordering::SeqCst) - before;
    if outcome == Outcome::Ran {
        assert_eq!(announced, usize::from(case.staged), "{:?}", case.format.0);
    }
    eprintln!(
        "MS-LAYOUT: {:?} x{} {}x{ROWS} {} {:?}: {outcome:?}, staged {announced}",
        case.format.0,
        case.samples,
        case.width,
        if into_texture { "into" } else { "out of" },
        case.layout,
    );
    outcome
}

/// Each measured case, both ways, in a shared buffer.
#[test]
fn off_pixel_layouts_copy_the_rows_plain_metal_places() {
    let scratch = Scratch::new("ms-layouts");
    let (device, staged) = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let make = |bytes: &[u8]| shared(&device, bytes);
    let at = (&queue, &device, &*staged);
    let mut ran = 0;
    for case in &INTO {
        ran += usize::from(run(at, case, true, &make) == Outcome::Ran);
    }
    for case in &OUT {
        ran += usize::from(run(at, case, false, &make) == Outcome::Ran);
    }
    eprintln!("MS-LAYOUT: {ran} of {} cases ran", INTO.len() + OUT.len());
}

/// Host memory of 64 KiB, aligned to 64 KiB, freed when dropped: declared before the device that
/// uses it, so dropped after it.
struct Pages(NonNull<u8>);

impl Pages {
    const LAYOUT: Layout = match Layout::from_size_align(64 << 10, 64 << 10) {
        Ok(layout) => layout,
        Err(_) => panic!("a layout"),
    };

    fn new() -> Pages {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Pages::LAYOUT) };
        Pages(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Pages::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Pages::LAYOUT) };
    }
}

/// A no-copy buffer over a pointer 4 bytes into a page, whose byte 0 lies 4 bytes into its
/// `VkBuffer`: a copy at the pixel's own layout stages, both ways, and copies Metal's rows.
#[test]
fn a_no_copy_buffer_off_its_vk_buffer_s_alignment_copies_through_staging() {
    let pages = Pages::new();
    let scratch = Scratch::new("ms-layouts-no-copy");
    let (device, staged) = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let make = |bytes: &[u8]| {
        assert_eq!(bytes.len() as u64, LENGTH);
        // SAFETY: 4 bytes into the 64 KiB allocation.
        let pointer = unsafe { pages.0.add(4) };
        // SAFETY: `LENGTH` bytes 4 into the 64 KiB allocation, which outlives the device and
        // every buffer over it; only the buffer's commands use them while it lives, and the test
        // reads and writes them only with no command buffer pending.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr(), bytes.len());
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pointer),
                LENGTH,
                ResourceOptions::shared(),
                None,
            )
        }
    };
    if let Err(error) = make(&[0; LENGTH as usize]) {
        eprintln!("MS-LAYOUT: skipped: the device imports no such host memory: {error:?}");
        return;
    }
    let make = |bytes: &[u8]| make(bytes).unwrap();
    let case = Case {
        format: (PixelFormat::RG32Uint, 8),
        samples: 2,
        width: 2,
        layout: (0, 32),
        rows: [0, 32, 64, 96],
        staged: true,
    };
    let at = (&queue, &device, &*staged);
    let ran = [true, false].map(|into_texture| run(at, &case, into_texture, &make));
    eprintln!("MS-LAYOUT: no-copy {ran:?}");
}

/// The layouts Metal places by a rule the runtime does not hold are its gap, and a pitch below
/// the pixel from an offset that is a multiple of 16 is refused, at the call: the encoder and
/// its command buffer run on.
#[test]
fn layouts_without_a_held_placement_are_refused() {
    let scratch = Scratch::new("ms-layouts-refused");
    let (device, _) = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let Some(texture) = texture(&device, PixelFormat::R8Uint, 2, 1) else {
        eprintln!("MS-LAYOUT: skipped: no 2-sample R8Uint texture here");
        return;
    };
    let source = shared(&device, &pattern(1, LENGTH));
    let gap = copy(&queue, (&source, 0, 0), &texture, 1, true);
    assert_eq!(
        gap,
        Err(Error::NotImplemented(RuntimeGap::MultisampledBufferLayout))
    );
    let below = copy(&queue, (&source, 0, 1), &texture, 1, true);
    let refused = Invalid::new(
        "copy_from_buffer_to_texture",
        Problem::Copy(CopyProblem::BytesPerRow),
    );
    assert_eq!(below, Err(Error::Invalid(refused)));
    eprintln!("MS-LAYOUT: refused");
}
