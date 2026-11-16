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
    drop(queue);
    drop(device);
    drop(ram);
}
