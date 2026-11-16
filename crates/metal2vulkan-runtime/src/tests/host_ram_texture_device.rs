//! Copies between a guest RAM's surface and a texture on a device (`host_ram/texture.rs`):
//! `BlitCommandEncoder::copy_texture_from_host_ram` and `copy_texture_to_host_ram`.
//!
//! - **Both ways, texel for texel, straight from the imports.** A 300 by 200 `BGRA8Unorm`
//!   `Private` texture fills from a surface of 63 pages scattered over the RAM's two chunks
//!   (two of them side by side across the first chunk's end, one run), its rows 1,280 bytes apart
//!   from byte 48: every texel its row's bytes, one region for each row's bytes in a run and a
//!   chunk (a row across the chunk's end two) and no dispatch or staging. Copied back out into
//!   the pages, rewritten since, only the texels' bytes change: the bytes between the rows and
//!   every other byte of the RAM stay, and the writes are told to the RAM's log.
//! - **Ordered as a blit is**, in one command buffer: a fill of part of the texture after a render
//!   pass cleared it leaves the clear around the region and the surface's texels in it, as a
//!   blit's read after it reads; a copy out after a render pass cleared the texture writes the
//!   clear's texels, as a gather from the RAM after it reads.
//! - **First touch through the CPU**, on a device whose thread imports ahead: a fill from, and a
//!   copy out into, pages of chunks not imported yet are staged (one copy of the region's rows
//!   between the arena and the texture), and move the same texels.
//! - **Refused**: a RAM the device does not use, a run off a word, a run that splits a texel, a
//!   region past the level, a pitch off the texel, rows past the surface, and textures off the
//!   copies' contract (multisampled, depth, block-compressed).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::host_ram_copy_device::{
    ahead_ram, blit, bytes, contents, device, new_ram, pattern, scattered_pages, write_pattern,
    Log, PAGE, RAM_BYTES, SECOND, WAIT,
};
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CopyProblem, Device, Error,
    LoadAction, Origin, Problem, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::ops::Range;
use std::sync::Arc;

const WIDTH: u64 = 300;
const HEIGHT: u64 = 200;
const TEXEL: u64 = 4;
/// The surface's first texel's byte and its bytes per row: past a row's 1,200.
const OFFSET: u64 = 48;
const PITCH: u64 = 1280;
/// The surface's pages: its bytes, `OFFSET + HEIGHT × PITCH`, rounded up.
const SURFACE_PAGES: u64 = (OFFSET + HEIGHT * PITCH).div_ceil(PAGE);

/// The RAM's page surface page `i` lies in: apart from each other in the first chunk, then two
/// side by side across its end, then apart in the second.
fn surface_page(i: u64) -> u64 {
    match i {
        0..30 => 100 + 2 * i,
        30 => SECOND - 1,
        31 => SECOND,
        _ => SECOND + 10 + 3 * (i - 32),
    }
}

/// The host address of byte `at` of the surface.
fn host(start: u64, at: u64) -> u64 {
    start + surface_page(at / PAGE) * PAGE + at % PAGE
}

/// The surface's bytes in order as runs of the RAM, pages side by side one run.
fn surface_runs(start: u64) -> Vec<Range<u64>> {
    let mut runs: Vec<Range<u64>> = Vec::new();
    for i in 0..SURFACE_PAGES {
        let at = start + surface_page(i) * PAGE;
        match runs.last_mut() {
            Some(last) if last.end == at => last.end = at + PAGE,
            _ => runs.push(at..at + PAGE),
        }
    }
    runs
}

/// The regions of a copy of `rows` rows of `width` texels from surface byte `offset` at `PITCH`:
/// one a row and one more for each page's end inside it (each is a run's or the chunk's end).
fn row_pieces(offset: u64, (width, rows): (u64, u64)) -> u64 {
    (0..rows)
        .map(|y| {
            let first = offset + y * PITCH;
            let last = first + width * TEXEL - 1;
            1 + last / PAGE - first / PAGE
        })
        .sum()
}

/// A `Private` `BGRA8Unorm` texture of `WIDTH` by `HEIGHT`, a render target.
fn bgra_texture(device: &Device) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A `Shared` buffer of `length` bytes.
fn shared(device: &Device, length: u64) -> Buffer {
    device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap()
}

/// Encode in `command_buffer` a blit of the whole of `texture` into `out`, rows one after
/// another.
fn read_into(command_buffer: &CommandBuffer, texture: &Texture, out: &Buffer) {
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder
        .copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            out,
            0,
            WIDTH * TEXEL,
            WIDTH * TEXEL * HEIGHT,
            BlitOption::None,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// Encode in `command_buffer` a render pass that clears `texture` to `color` and stores it.
fn clear(command_buffer: &CommandBuffer, texture: &Texture, color: [f64; 4]) {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    let [r, g, b, a] = color;
    attachment.clear_color = ClearColor::new(r, g, b, a);
    attachment.store_action = StoreAction::Store;
    let encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.end_encoding().unwrap();
}

/// Commit `command_buffer` and wait for it.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The texture's texels as a blit reads them, rows one after another.
fn read_back(device: &Device, texture: &Texture) -> Vec<u8> {
    let out = shared(device, WIDTH * TEXEL * HEIGHT);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    read_into(&command_buffer, texture, &out);
    complete(&command_buffer);
    contents(&out)
}

/// The surface's texels of a region of `width` by `rows` from surface byte `offset`, as the
/// CPU reads them, rows one after another.
fn surface_texels(start: u64, offset: u64, (width, rows): (u64, u64)) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..rows {
        for b in 0..width * TEXEL {
            let address = host(start, offset + y * PITCH + b);
            out.extend(bytes(address..address + 1));
        }
    }
    out
}

/// The counts a closure's work moved: regions, dispatches, staged gathers and scatters.
fn counted(device: &Device, work: impl FnOnce()) -> [u64; 4] {
    let before = device.work_counts();
    work();
    let after = device.work_counts();
    [
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        after.host_ram_scatters_staged - before.host_ram_scatters_staged,
    ]
}

#[test]
fn a_texture_fills_from_a_surface_s_rows_and_copies_back_into_their_texel_bytes_alone() {
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(Log::default());
    let (ram, start) = new_ram(Some(Arc::clone(&log)));
    let _used = device.use_host_ram(&ram).unwrap();
    let runs = surface_runs(start);
    let end = start + SECOND * PAGE;
    assert!(
        (runs.iter()).any(|run| run.start < end && end < run.end),
        "a run across the first chunk's end"
    );
    let crossing = (0..HEIGHT).any(|y| {
        let row = OFFSET + y * PITCH..OFFSET + y * PITCH + WIDTH * TEXEL;
        row.contains(&(31 * PAGE))
    });
    assert!(crossing, "a row across the chunk's end");
    for run in &runs {
        write_pattern(run.clone(), 21);
    }
    let texture = bgra_texture(&device);
    let whole = Size::new(WIDTH, HEIGHT, 1);
    let origin = Origin::new(0, 0, 0);
    let fill = counted(&device, || {
        blit(&device, |encoder| {
            (encoder.copy_texture_from_host_ram(
                &ram, &runs, OFFSET, PITCH, whole, &texture, 0, 0, origin,
            ))
            .unwrap();
        });
    });
    assert!(
        read_back(&device, &texture) == surface_texels(start, OFFSET, (WIDTH, HEIGHT)),
        "every texel from its row's bytes"
    );
    let pieces = row_pieces(OFFSET, (WIDTH, HEIGHT));
    eprintln!(
        "HOST-RAM-TEXTURE {}: fill regions {}, dispatches {}, staged {}",
        device.name(),
        fill[0],
        fill[1],
        fill[2]
    );
    assert_eq!(
        fill,
        [pieces, 0, 0, 0],
        "a region a row's piece straight from its import, no dispatch, no staging"
    );
    // Out again into the pages, rewritten since: the texels' bytes alone take the texture's.
    for run in &runs {
        write_pattern(run.clone(), 22);
    }
    let mut want = bytes(start..start + RAM_BYTES);
    for y in 0..HEIGHT {
        for b in 0..WIDTH * TEXEL {
            let address = host(start, OFFSET + y * PITCH + b);
            want[(address - start) as usize] = pattern(address, 21);
        }
    }
    let out = counted(&device, || {
        blit(&device, |encoder| {
            (encoder.copy_texture_to_host_ram(
                &texture, 0, 0, origin, whole, &ram, &runs, OFFSET, PITCH,
            ))
            .unwrap();
        });
    });
    let now = bytes(start..start + RAM_BYTES);
    assert!(
        now == want,
        "the texels' bytes alone, and no other byte of the RAM"
    );
    assert_eq!(out, [pieces, 0, 0, 0], "a region a row's piece, out");
    // Each row's texels told to the RAM's log as the GPU's writes in place.
    for y in [0, 99, HEIGHT - 1] {
        let first = host(start, OFFSET + y * PITCH);
        assert!(
            log.heard(&(first..first + 64), 0),
            "row {y}'s writes told to the RAM's log"
        );
    }
}

#[test]
fn texture_copies_are_ordered_against_a_render_pass_before_and_a_read_after() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let runs = surface_runs(start);
    for run in &runs {
        write_pattern(run.clone(), 23);
    }
    let texture = bgra_texture(&device);
    let queue = device.new_command_queue().unwrap();
    // A fill of part of the texture after a clear of all of it, read back after it.
    let (x, y, width, rows) = (20, 30, 260, 150);
    let offset = OFFSET + 8 * TEXEL;
    let out = shared(&device, WIDTH * TEXEL * HEIGHT);
    let command_buffer = queue.command_buffer().unwrap();
    clear(&command_buffer, &texture, [0.0, 1.0, 0.0, 1.0]);
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let size = Size::new(width, rows, 1);
    let at = Origin::new(x, y, 0);
    (encoder.copy_texture_from_host_ram(&ram, &runs, offset, PITCH, size, &texture, 0, 0, at))
        .unwrap();
    encoder.end_encoding().unwrap();
    read_into(&command_buffer, &texture, &out);
    complete(&command_buffer);
    let got = contents(&out);
    let texels = surface_texels(start, offset, (width, rows));
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let at = ((row * WIDTH + column) * TEXEL) as usize;
            let inside = (x..x + width).contains(&column) && (y..y + rows).contains(&row);
            let want: &[u8] = match inside {
                true => {
                    let from = (((row - y) * width + column - x) * TEXEL) as usize;
                    &texels[from..from + TEXEL as usize]
                }
                // BGRA: the clear's green, opaque.
                false => &[0, 255, 0, 255],
            };
            assert_eq!(
                &got[at..at + TEXEL as usize],
                want,
                "texel ({column}, {row})"
            );
        }
    }
    // A copy out after a clear, gathered from the RAM after it: the clear's texels.
    let gathered = shared(&device, SURFACE_PAGES * PAGE);
    let command_buffer = queue.command_buffer().unwrap();
    clear(&command_buffer, &texture, [1.0, 0.0, 0.0, 1.0]);
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let whole = Size::new(WIDTH, HEIGHT, 1);
    let origin = Origin::new(0, 0, 0);
    (encoder.copy_texture_to_host_ram(&texture, 0, 0, origin, whole, &ram, &runs, OFFSET, PITCH))
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder
        .copy_from_host_ram(&ram, &runs, &gathered, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    let got = contents(&gathered);
    for row in 0..HEIGHT {
        let first = (OFFSET + row * PITCH) as usize;
        for texel in got[first..first + (WIDTH * TEXEL) as usize].chunks(TEXEL as usize) {
            assert_eq!(
                texel,
                [0, 0, 255, 255],
                "row {row}: the clear's red, opaque"
            );
        }
    }
}

#[test]
fn first_texture_copies_over_chunks_the_thread_imports_are_staged_through_the_cpu() {
    // The surface over pages of chunks not imported yet: 40 pages, so a region of 100 rows.
    let rows = 100;
    let size = Size::new(WIDTH, rows, 1);
    let origin = Origin::new(0, 0, 0);
    let laid = |start: u64, at: u64| {
        let runs = scattered_pages(start);
        let page = &runs[(at / PAGE) as usize];
        page.start + at % PAGE
    };
    let texels = |start: u64| {
        let mut out = Vec::new();
        for y in 0..rows {
            for b in 0..WIDTH * TEXEL {
                let address = laid(start, OFFSET + y * PITCH + b);
                out.extend(bytes(address..address + 1));
            }
        }
        out
    };
    // A fill, staged.
    let Some((device, used, ram, start, _)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 31);
    }
    let texture = bgra_texture(&device);
    let fill = counted(&device, || {
        blit(&device, |encoder| {
            (encoder.copy_texture_from_host_ram(
                &ram, &runs, OFFSET, PITCH, size, &texture, 0, 0, origin,
            ))
            .unwrap();
        });
    });
    assert_eq!(fill, [0, 0, 1, 0], "the fill staged through the CPU");
    let got = read_back(&device, &texture);
    assert!(
        got[..(WIDTH * TEXEL * rows) as usize] == texels(start)[..],
        "the staged fill's texels"
    );
    drop((used, ram));
    // A copy out, staged, on a device of its own.
    let Some((device, used, ram, start, _)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 32);
    }
    let texture = bgra_texture(&device);
    let source = shared(&device, WIDTH * TEXEL * rows);
    let filled: Vec<u8> = (0..WIDTH * TEXEL * rows)
        .map(|at| pattern(at, 33))
        .collect();
    let contents_at = source.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(filled.as_ptr(), contents_at.as_ptr(), filled.len());
    }
    blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_texture(
                &source,
                0,
                WIDTH * TEXEL,
                WIDTH * TEXEL * rows,
                size,
                &texture,
                0,
                0,
                origin,
                BlitOption::None,
            )
            .unwrap();
    });
    // The surface's pages as they are, then with the texels' bytes the texture's.
    let mut want: Vec<Vec<u8>> = runs.iter().map(|run| bytes(run.clone())).collect();
    for y in 0..rows {
        for b in 0..WIDTH * TEXEL {
            let at = OFFSET + y * PITCH + b;
            want[(at / PAGE) as usize][(at % PAGE) as usize] =
                filled[(y * WIDTH * TEXEL + b) as usize];
        }
    }
    let out = counted(&device, || {
        blit(&device, |encoder| {
            (encoder.copy_texture_to_host_ram(
                &texture, 0, 0, origin, size, &ram, &runs, OFFSET, PITCH,
            ))
            .unwrap();
        });
    });
    assert_eq!(out, [0, 0, 0, 1], "the copy out staged through the CPU");
    for (page, run) in runs.iter().enumerate() {
        assert!(
            bytes(run.clone()) == want[page],
            "the staged copy out's texels alone, in the surface's page {page}"
        );
    }
    drop((used, ram));
}

/// Whether `result` is the refusal `Invalid` of `problem`.
fn refused(result: Result<(), Error>, problem: &Problem) -> bool {
    matches!(result, Err(Error::Invalid(invalid)) if invalid.problem == *problem)
}

#[test]
fn texture_copies_off_their_contract_are_refused() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let runs = surface_runs(start);
    let texture = bgra_texture(&device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let whole = Size::new(WIDTH, HEIGHT, 1);
    let origin = Origin::new(0, 0, 0);
    let host_ram = Problem::HostRam;
    // A RAM the device does not use yet.
    let result = encoder
        .copy_texture_from_host_ram(&ram, &runs, OFFSET, PITCH, whole, &texture, 0, 0, origin);
    assert!(refused(result, &host_ram));
    let used = device.use_host_ram(&ram).unwrap();
    // A run off a word, and one that splits a texel (a run of 4,098 bytes, the next row's texel
    // across its end).
    let mut off = runs.clone();
    off[1].start += 2;
    let mut split = runs.clone();
    split[0].end += 2;
    for runs in [&off, &split] {
        let into = encoder
            .copy_texture_from_host_ram(&ram, runs, OFFSET, PITCH, whole, &texture, 0, 0, origin);
        assert!(refused(into, &host_ram), "{:x?}", &runs[..2]);
        let out = encoder
            .copy_texture_to_host_ram(&texture, 0, 0, origin, whole, &ram, runs, OFFSET, PITCH);
        assert!(refused(out, &host_ram), "{:x?}", &runs[..2]);
    }
    // A region past the level, a pitch or an offset off the texel, rows past the surface.
    let copy = |problem: CopyProblem| Problem::Copy(problem);
    let past = Size::new(WIDTH, HEIGHT, 1);
    let shifted = Origin::new(1, 0, 0);
    for (layout, size, at, problem) in [
        ((OFFSET, PITCH), past, shifted, copy(CopyProblem::Region)),
        (
            (OFFSET, PITCH + 2),
            whole,
            origin,
            copy(CopyProblem::BytesPerRow),
        ),
        (
            (OFFSET + 2, PITCH),
            whole,
            origin,
            copy(CopyProblem::Offset),
        ),
        (
            (OFFSET + PAGE * 8, PITCH),
            whole,
            origin,
            copy(CopyProblem::BufferRange),
        ),
    ] {
        let (offset, pitch) = layout;
        let into = encoder
            .copy_texture_from_host_ram(&ram, &runs, offset, pitch, size, &texture, 0, 0, at);
        assert!(refused(into, &problem), "{layout:?} {size:?} {at:?}");
        let out =
            encoder.copy_texture_to_host_ram(&texture, 0, 0, at, size, &ram, &runs, offset, pitch);
        assert!(refused(out, &problem), "{layout:?} {size:?} {at:?}");
    }
    // Textures off the copies' contract: multisampled, depth, block-compressed.
    let mut multisampled =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    multisampled.texture_type = TextureType::Type2DMultisample;
    multisampled.sample_count = 4;
    multisampled.usage = TextureUsage::RENDER_TARGET;
    multisampled.resource_options = ResourceOptions::private();
    let mut depth = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, WIDTH, HEIGHT, false);
    depth.usage = TextureUsage::RENDER_TARGET;
    depth.resource_options = ResourceOptions::private();
    let mut compressed = TextureDescriptor::texture_2d(PixelFormat::BC1RGBA, WIDTH, HEIGHT, false);
    compressed.resource_options = ResourceOptions::private();
    for descriptor in [multisampled, depth, compressed] {
        let other = match device.new_texture(&descriptor) {
            Ok(other) => other,
            Err(error) => {
                eprintln!(
                    "HOST-RAM-TEXTURE: no {:?} here: {error}",
                    descriptor.pixel_format
                );
                continue;
            }
        };
        let into = encoder
            .copy_texture_from_host_ram(&ram, &runs, OFFSET, PITCH, whole, &other, 0, 0, origin);
        assert!(refused(into, &host_ram), "{:?}", descriptor.pixel_format);
        let out = encoder
            .copy_texture_to_host_ram(&other, 0, 0, origin, whole, &ram, &runs, OFFSET, PITCH);
        assert!(refused(out, &host_ram), "{:?}", descriptor.pixel_format);
    }
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    drop((used, ram));
}
