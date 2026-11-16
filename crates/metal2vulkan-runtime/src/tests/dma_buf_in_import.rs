//! A texture in place over a no-copy buffer's dma-buf pages whose image is bound in the import the
//! buffer lies in (`texture/in_place_image.rs`, `PagesClass::DmaBufInImport`), on the device the
//! cfg selects made without `VK_EXT_external_memory_host`, so that its no-copy buffers are
//! imported as dma-bufs of their pages (`tests/no_pointer_import.rs`): a 64x32 `BGRA8Unorm`
//! texture that only reads, rows 512 bytes apart, 512 bytes into the second page of a no-copy
//! buffer over a `memfd`'s pages mapped shared, whose source makes `udmabuf`s of them and counts
//! its calls.
//!
//! - **In the import.** Where the device's check passed such images for the texture's reads, the
//!   texture is in place with its image in the buffer's import
//!   (`WorkCounts::dma_buf_images_in_import` + 1): no dma-buf image (`dma_buf_images` + 0), no
//!   import of its own (`dma_buf_image_memories_live` + 0), and no call of the source. A blit
//!   reads its texels from the pages, then the CPU's later bytes there; a blit's copy into it
//!   writes its texels in the pages and leaves every other byte of them alone (the 256 bytes
//!   between its rows among them).
//! - **Freed with its last holder.** The image binding in the import keeps it imported for good
//!   (`host_residency.rs`: never released while idle), not past its holders: once the textures
//!   and the buffer went, the buffer's import is freed (`WorkCounts::host_imports_live` and
//!   `dma_buf_memories_live` back to what they were before the buffer).
//! - **Today's image, the negative control.** With that class's verdicts failed, the same texture
//!   takes the image in a dma-buf import of its own (`dma_buf_images` + 1, the source asked once,
//!   `dma_buf_images_in_import` + 0) where the device's check passed those images, and reads what
//!   the image in the import reads.
//! - **Elsewhere** (a device whose check leaves the class unchecked: RADV, whose
//!   `DRM_FORMAT_MOD_LINEAR` images for dma-bufs require a dedicated allocation), no texture's
//!   image is bound in the import (`dma_buf_images_in_import` + 0), and the test says what the
//!   texture took and passes.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, or the device lists no dma-buf route or makes no
//! `DRM_FORMAT_MOD_LINEAR` images, it says so and passes without running.

use super::host_residency::{
    counting, host_page, own_dma_buf_images, udmabuf_openable, Ram, Scratch,
};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::in_place::{Access, PagesClass, RowEnds, Verdict};
use crate::linear_plan::BufferLayout;
use crate::{
    BlitOption, Buffer, CommandQueue, HostMemory, Origin, ResourceOptions, Size, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "IN-IMPORT";
/// The texture's width and height, and the bytes from a row's first texel to the next row's: a
/// multiple of the 256 RADV lays a plane's rows at, with 256 bytes between rows.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 32;
const ROW: u64 = 512;
const ROW_BYTES: u64 = 4 * WIDTH;
const CHECK_WAIT: Duration = Duration::from_secs(120);
/// The longest the test waits for a free on another thread (a completed command buffer's).
const SETTLE: Duration = Duration::from_secs(20);

/// The texture's texels, copied out by a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = (queue.device())
        .new_buffer(ROW_BYTES * HEIGHT, ResourceOptions::shared())
        .unwrap();
    blit(queue, |encoder| {
        (encoder.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            &out,
            0,
            ROW_BYTES,
            ROW_BYTES * HEIGHT,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, ROW_BYTES * HEIGHT)
}

/// The texels of a texture `at` bytes into `pages`, row after row.
fn texels_in(pages: &[u8], at: u64) -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|row| &pages[(at + row * ROW) as usize..][..ROW_BYTES as usize])
        .copied()
        .collect()
}

/// What making a texture changed: its images bound in an import, its dma-buf images, the live
/// imports of such images, and the source's calls.
fn made(before: &WorkCounts, after: &WorkCounts, calls: usize) -> (u64, u64, u64, usize) {
    (
        after.dma_buf_images_in_import - before.dma_buf_images_in_import,
        after.dma_buf_images - before.dma_buf_images,
        after.dma_buf_image_memories_live - before.dma_buf_image_memories_live,
        calls,
    )
}

/// The checks the module's docs list.
#[test]
fn a_texture_over_a_dma_buf_buffer_s_pages_binds_in_its_import() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    let name = device.name();
    if !lists_dma_buf_route(&device) || !device.makes_modifier_images() {
        println!(
            "{TEST}: {name} lists no dma-buf route or makes no DRM_FORMAT_MOD_LINEAR images: not \
             run"
        );
        return;
    }
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    for line in device.in_place_check_report() {
        if line.starts_with("DmaBuf") {
            println!("{TEST} on {name}: the check: {line}");
        }
    }
    let queue = device.new_command_queue().unwrap();
    let page = host_page();
    let offset = page as u64 + 512;
    let length = offset + ROW * HEIGHT;
    let file = (length as usize).next_multiple_of(page);
    let ram = Ram::new(file, 0, file);
    ram.write_file(0, &counting(file, 7));
    let calls = Arc::new(AtomicUsize::new(0));
    let asked = || calls.load(Ordering::Acquire);
    let unmade = device.work_counts();
    // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
    let buffer: Buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(ram.pointer(0), ram.source(&calls)),
            length,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    // The verdicts the texture's route asks: its reads, for texels that lie in lines as its do.
    let layout = BufferLayout {
        offset,
        row_pitch: ROW,
        slice_pitch: ROW * HEIGHT,
        row_bytes: ROW_BYTES,
        rows: HEIGHT,
        slices: 1,
    };
    let start = ram.pointer(0).as_ptr().addr() as u64;
    let rows = RowEnds::of(start, &layout, device.shared().in_place.line());
    let in_import = device.in_place_verdict(PagesClass::DmaBufInImport, Access::Reads, rows);
    let own = device.in_place_verdict(PagesClass::DmaBuf, Access::Reads, rows);

    let before = device.work_counts();
    let by_buffer = asked();
    let texture = buffer.new_texture(&descriptor, offset, ROW).unwrap();
    let after = device.work_counts();
    // SAFETY: the handle is only compared, while the texture lives.
    let in_place = unsafe { texture.vk_image() }.is_some();
    let first = made(&before, &after, asked() - by_buffer);
    assert!(
        texels_of(&queue, &texture) == texels_in(&counting(file, 7), offset),
        "{TEST}: the texture's texels"
    );
    if in_import != Verdict::Passed {
        assert_eq!(
            first.0, 0,
            "{TEST}: no image in the import where its check did not pass ({in_import:?})"
        );
        println!(
            "{TEST} on {name}: the check of images in the import answered {in_import:?}, of \
             images in imports of their own {own:?}; the texture in place {in_place}: in the \
             import {}, dma-buf images {}, their imports live {}, source calls {}",
            first.0, first.1, first.2, first.3
        );
        return;
    }
    assert!(in_place, "{TEST}: in place");
    assert_eq!(
        first,
        (1, 0, 0, 0),
        "{TEST}: in the buffer's import, with no dma-buf image, import or source call (in the \
         import, dma-buf images, their imports live, source calls)"
    );
    // The CPU's later bytes.
    ram.write_file(0, &counting(file, 101));
    assert!(
        texels_of(&queue, &texture) == texels_in(&counting(file, 101), offset),
        "{TEST}: the CPU's later bytes"
    );
    // A copy into it writes its texels in the pages, and no other byte.
    let fresh = counting((ROW_BYTES * HEIGHT) as usize, 211);
    let source = (device.new_buffer(ROW_BYTES * HEIGHT, ResourceOptions::shared())).unwrap();
    // SAFETY: inside the buffer's length, which no command uses now.
    unsafe {
        let contents = source.contents().unwrap().as_ptr();
        std::ptr::copy_nonoverlapping(fresh.as_ptr(), contents, fresh.len());
    }
    blit(&queue, |encoder| {
        (encoder.copy_from_buffer_to_texture(
            &source,
            0,
            ROW_BYTES,
            ROW_BYTES * HEIGHT,
            Size::new(WIDTH, HEIGHT, 1),
            &texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        ))
        .unwrap()
    });
    let mut want = counting(file, 101);
    for row in 0..HEIGHT {
        let at = (offset + row * ROW) as usize;
        let from = (row * ROW_BYTES) as usize;
        want[at..][..ROW_BYTES as usize].copy_from_slice(&fresh[from..][..ROW_BYTES as usize]);
    }
    assert!(
        ram.read_file(0, file) == want,
        "{TEST}: the copy wrote the texels alone"
    );
    let in_the_import = texels_of(&queue, &texture);
    assert!(in_the_import == fresh, "{TEST}: the copied texels read");

    // Today's image, the negative control.
    own_dma_buf_images(&device);
    let before = device.work_counts();
    let by_first = asked();
    let control = buffer.new_texture(&descriptor, offset, ROW).unwrap();
    let after = device.work_counts();
    let second = made(&before, &after, asked() - by_first);
    // SAFETY: as above.
    let control_in_place = unsafe { control.vk_image() }.is_some();
    assert_eq!(
        second.0, 0,
        "{TEST}: no image in the import once its check failed"
    );
    if own == Verdict::Passed {
        assert!(control_in_place, "{TEST}: the control in place");
        assert_eq!(
            (second.1, second.3),
            (1, 1),
            "{TEST}: the control's image in a dma-buf import of its own, the source asked once"
        );
    }
    assert!(
        texels_of(&queue, &control) == in_the_import,
        "{TEST}: the control reads what the image in the import reads"
    );
    println!(
        "{TEST} on {name}: in the import {} (dma-buf images {}, their imports live {}, source \
         calls {}); the control, its check of images in imports of their own {own:?}, in place \
         {control_in_place}: in the import {}, dma-buf images {}, source calls {}",
        first.0, first.1, first.2, first.3, second.0, second.1, second.3
    );
    drop(control);
    drop(texture);
    drop(buffer);
    // Freed with its last holder.
    let live = |counts: &WorkCounts| (counts.host_imports_live, counts.dma_buf_memories_live);
    let start = std::time::Instant::now();
    let gone = loop {
        let counts = device.work_counts();
        let freed = live(&counts).0 <= live(&unmade).0 && live(&counts).1 <= live(&unmade).1;
        if freed || start.elapsed() > SETTLE {
            break counts;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(
        live(&gone).0 <= live(&unmade).0 && live(&gone).1 <= live(&unmade).1,
        "{TEST}: the buffer's import freed once the textures and the buffer went (imports live, \
         dma-buf memories live): {:?}, {:?} before the buffer",
        live(&gone),
        live(&unmade)
    );
    drop(queue);
    drop(device);
    drop(ram);
}

/// The rounds of [`in_import_first_use_bench`], each arm once in each, and the first ones not
/// counted.
const ROUNDS: usize = 4;
const WARM: usize = 1;

/// What a texture over a no-copy buffer's dma-buf pages costs at its making and first use, its
/// image in the buffer's import against in a dma-buf import of its own (the class's verdicts
/// failed, `own_dma_buf_images`), on the device the cfg selects made without
/// `VK_EXT_external_memory_host`: a `BGRA8Unorm` `ShaderRead` texture over a whole no-copy buffer
/// at rows of its width, 1920x1080 (7.9 MiB) and 4096x5376 (84 MiB), each made, then read one
/// texel by a blit that is waited for. Each arm runs once a round, the arms' order turned by one
/// each round, [`ROUNDS`] rounds, the first [`WARM`] not counted; every texture is kept until its
/// shape is done, so none parks for the next. Printed (`IN-IMPORT-AB`), per shape and arm, the
/// median of `new_texture`'s wall time, of the first use's (the blit's commit and wait), and of
/// both, and the counts that show each arm's path.
///
/// Ignored: run it in release under `--cfg m2v_gpu` on Linux where `/dev/udmabuf` opens, on a
/// device whose check passed images in the import, `cargo nextest run --release -p
/// metal2vulkan-runtime --lib --run-ignored all in_import_first_use_bench --no-capture`.
#[test]
#[ignore]
fn in_import_first_use_bench() {
    const BENCH: &str = "IN-IMPORT-AB";
    if !udmabuf_openable(BENCH) {
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0, None);
    let name = device.name();
    if !lists_dma_buf_route(&device) || !device.makes_modifier_images() {
        println!("{BENCH}: {name} lists no dma-buf route or makes no modifier images: not run");
        return;
    }
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{BENCH}: the check"
    );
    let checks = &device.shared().in_place;
    let class = PagesClass::DmaBufInImport;
    let saved: Vec<(Access, RowEnds, Verdict)> = (Access::ALL.into_iter())
        .flat_map(|access| RowEnds::ALL.map(|rows| (access, rows)))
        .map(|(access, rows)| (access, rows, checks.verdict(class, access, rows)))
        .collect();
    if !saved
        .iter()
        .any(|&(_, _, verdict)| verdict == Verdict::Passed)
    {
        println!("{BENCH} on {name}: the check passed no image in the import: not run");
        return;
    }
    let restore = || {
        for &(access, rows, verdict) in &saved {
            checks.set(class, access, rows, verdict);
        }
    };
    let queue = device.new_command_queue().unwrap();
    let out = (device.new_buffer(16, ResourceOptions::shared())).unwrap();
    let page = host_page();
    let median = |mut times: Vec<Duration>| {
        times.sort();
        times[times.len() / 2]
    };
    for (width, height) in [(1920u64, 1080u64), (4096, 5376)] {
        let row = 4 * width;
        let length = row * height;
        let file = (length as usize).next_multiple_of(page);
        let ram = Ram::new(file, 0, file);
        ram.write_file(0, &counting(file, 3));
        let calls = Arc::new(AtomicUsize::new(0));
        // SAFETY: the view outlives the buffer, and the source makes dma-bufs of its pages.
        let buffer: Buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::with_dma_buf(ram.pointer(0), ram.source(&calls)),
                length,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let mut kept = Vec::new();
        // Per arm: the making's, the first use's and both times, and the arm's paths.
        let mut times: [Vec<(Duration, Duration, Duration)>; 2] = [Vec::new(), Vec::new()];
        let mut paths = [(0u64, 0u64, 0usize); 2];
        for round in 0..ROUNDS {
            for turn in 0..2 {
                let arm = (round + turn) % 2;
                match arm {
                    0 => restore(),
                    _ => own_dma_buf_images(&device),
                }
                let before = device.work_counts();
                let asked = calls.load(Ordering::Acquire);
                let start = std::time::Instant::now();
                let texture = buffer.new_texture(&descriptor, 0, row).unwrap();
                let made = start.elapsed();
                blit(&queue, |encoder| {
                    (encoder.copy_from_texture_to_buffer(
                        &texture,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(1, 1, 1),
                        &out,
                        0,
                        4,
                        4,
                        BlitOption::None,
                    ))
                    .unwrap()
                });
                let both = start.elapsed();
                let after = device.work_counts();
                paths[arm].0 += after.dma_buf_images_in_import - before.dma_buf_images_in_import;
                paths[arm].1 += after.dma_buf_images - before.dma_buf_images;
                paths[arm].2 += calls.load(Ordering::Acquire) - asked;
                if round >= WARM {
                    times[arm].push((made, both - made, both));
                }
                kept.push(texture);
            }
        }
        restore();
        for (arm, words) in ["in the import", "an import of its own"]
            .into_iter()
            .enumerate()
        {
            let arm_times = &times[arm];
            let made = median(arm_times.iter().map(|t| t.0).collect());
            let used = median(arm_times.iter().map(|t| t.1).collect());
            let both = median(arm_times.iter().map(|t| t.2).collect());
            let (in_import, images, asked) = paths[arm];
            println!(
                "{BENCH} on {name}: {width}x{height} {words}: new_texture {made:?}, first use \
                 {used:?}, both {both:?} (medians of {}); {ROUNDS} textures: in the import \
                 {in_import}, dma-buf images {images}, source calls {asked}",
                arm_times.len()
            );
        }
        drop(kept);
        drop(buffer);
        drop(ram);
    }
}
