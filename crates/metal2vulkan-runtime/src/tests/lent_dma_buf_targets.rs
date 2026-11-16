//! A render target over a no-copy buffer's dma-buf pages whose rows' padding is free
//! (`in_place::Padding::Free`), on the device the cfg selects made without
//! `VK_EXT_external_memory_host`, so that its no-copy buffers are imported as dma-bufs of their
//! pages (`tests/no_pointer_import.rs`): a 34x80 `RGBA16Float` texture that reads and renders, rows
//! 512 bytes apart from the first byte of a no-copy buffer of 80 such rows (40960 bytes, the last
//! row to its pitch) over a `memfd`'s pages mapped shared, whose source makes `udmabuf`s of them.
//! A game engine's renderer makes its targets so, over a linear image's memory.
//!
//! The dma-buf classes' verdicts for texels that share a line with bytes beside them are set by
//! the test: reads passed, unobserved renders passed, renders failed (a pass changed a byte of the
//! rows' padding), copies unchecked; the pointer class's copies failed, its reads passed; and the
//! fresh class's copies unchecked (the only other class of host pages).
//!
//! - **Lent through memory of one type.** With every class's check pages noted in memory of the
//!   type the buffer's pages lie in, the texture takes the pointer class's copies' failure
//!   (`in_place::copies_verdict`): it is its image in place (`WorkCounts::dma_buf_images_in_import`
//!   or `dma_buf_images` + 1, `host_texture_shadows_check_failed` + 0), announced once as
//!   `line-write-back`, and reads the pages' bytes.
//! - **Not across memory types, the negative control.** With the pointer class's check pages noted
//!   in memory of other property flags (`HOST_CACHED` turned), the same texture is a shadow
//!   (`host_texture_shadows_check_failed` + 1), whose `shadowed-texture` notice says the check
//!   made no copy into such rows over those pages and no class in memory of the same type found
//!   one changing a byte.
//! - **The device's own verdicts.** Before both, the check's verdicts as it measured them, and
//!   each class's memory, are printed (`LENT-TARGETS`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, the device lists no dma-buf route or makes no
//! `DRM_FORMAT_MOD_LINEAR` images, or holds no such target in place whose renders passed, it says
//! so and passes without running.

use super::draw_again::Heard;
use super::host_residency::{counting, host_page, udmabuf_openable, Ram, Scratch};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::lists_dma_buf_route;
use crate::device::lock;
use crate::in_place::{Access, PagesClass, RowEnds, Verdict};
use crate::linear_plan::BufferLayout;
use crate::{
    BlitOption, Buffer, CommandQueue, Degradation, Device, DeviceKind, DeviceOptions, HostMemory,
    MessageSink, Origin, ResourceOptions, ShaderCacheOptions, Size, Texture, TextureDescriptor,
    WorkCounts,
};
use ash::vk;
use metal2vulkan::device::VkExtension;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

const TEST: &str = "LENT-TARGETS";
/// The texture's width and height, the bytes of a row of its texels, and the bytes from a row's
/// first texel to the next row's.
const WIDTH: u64 = 34;
const HEIGHT: u64 = 80;
const ROW_BYTES: u64 = 8 * WIDTH;
const ROW: u64 = 512;
const CHECK_WAIT: Duration = Duration::from_secs(120);
/// The classes a texture over a no-copy buffer's dma-buf pages takes, and the one whose copies'
/// failure it takes.
const DMA_BUF: [PagesClass; 2] = [PagesClass::DmaBufInImport, PagesClass::DmaBuf];
const MEASURED: [PagesClass; 3] = [
    PagesClass::DmaBufInImport,
    PagesClass::DmaBuf,
    PagesClass::Pointer,
];

/// The device `no_pointer_import::device` makes, announcing `line-write-back` and
/// `shadowed-texture` at every occurrence, its notices into `heard`.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        withheld_extensions: [VkExtension::ExtExternalMemoryHost].into_iter().collect(),
        announce_each: [Degradation::LineWriteBack, Degradation::ShadowedTexture]
            .into_iter()
            .collect(),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(crate::tests::SETTLE),
        "the submission checks never answered"
    );
    device
}

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

/// The texels of the texture in `pages`, row after row.
fn texels_in(pages: &[u8]) -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|row| &pages[(row * ROW) as usize..][..ROW_BYTES as usize])
        .copied()
        .collect()
}

/// What making a texture changed: its dma-buf images (in the buffer's import, and in imports of
/// their own), and its shadows for a failed verdict.
fn made(before: &WorkCounts, after: &WorkCounts) -> (u64, u64) {
    let images = |counts: &WorkCounts| counts.dma_buf_images_in_import + counts.dma_buf_images;
    (
        images(after) - images(before),
        after.host_texture_shadows_check_failed - before.host_texture_shadows_check_failed,
    )
}

/// The checks the module's docs list.
#[test]
fn a_render_target_takes_the_copies_failure_of_pages_in_memory_of_the_same_type() {
    if !udmabuf_openable(TEST) {
        return;
    }
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
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
    let queue = device.new_command_queue().unwrap();
    let page = host_page();
    let length = ROW * HEIGHT;
    let file = (length as usize).next_multiple_of(page);
    let ram = Ram::new(file, 0, file);
    ram.write_file(0, &counting(file, 7));
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
        TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let layout = BufferLayout {
        offset: 0,
        row_pitch: ROW,
        slice_pitch: ROW * HEIGHT,
        row_bytes: ROW_BYTES,
        rows: HEIGHT,
        slices: 1,
    };
    let checks = &device.shared().in_place;
    let start = ram.pointer(0).as_ptr().addr() as u64;
    let rows = RowEnds::of(start, &layout, checks.line());
    assert_eq!(
        rows,
        RowEnds::InLines,
        "{TEST}: the rows' last texels share a line"
    );

    // The device's own verdicts.
    for class in MEASURED {
        let access = |access| checks.verdict(class, access, rows);
        println!(
            "{TEST} on {name}: {class:?}: reads {:?}, renders {:?}, unobserved renders {:?}, \
             copies {:?} (decided {:?}); memory {:#x?}",
            access(Access::Reads),
            access(Access::Renders),
            access(Access::UnobservedRenders),
            access(Access::Copies),
            checks.decided(class, Access::Copies, rows),
            checks.memory(class).map(vk::MemoryPropertyFlags::as_raw)
        );
    }
    let set = |verdicts: [(Access, Verdict); 4]| {
        for class in DMA_BUF {
            for (access, verdict) in verdicts {
                checks.set(class, access, rows, verdict);
            }
        }
    };
    let index = buffer.pages_type_index();
    let flags = (lock(&device.shared().memory).type_flags(index)).unwrap();
    for class in DMA_BUF {
        checks.note_memory(class, flags);
    }
    checks.set(PagesClass::Pointer, Access::Reads, rows, Verdict::Passed);
    checks.set(PagesClass::Pointer, Access::Copies, rows, Verdict::Failed);
    checks.set(PagesClass::Fresh, Access::Copies, rows, Verdict::Unchecked);
    let notices = |kind: Degradation| {
        let heard = heard.lock().unwrap();
        let mine = heard.iter().filter(|notice| {
            notice.degradation == kind
                && notice.object.contains(&format!("{WIDTH}x{HEIGHT}"))
                && notice.object.contains(&format!("rows of {ROW} bytes"))
        });
        mine.map(|notice| notice.reason.clone()).collect::<Vec<_>>()
    };
    let make = || {
        let before = device.work_counts();
        let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
        let after = device.work_counts();
        // SAFETY: the handle is only compared, while the texture lives.
        let in_place = unsafe { texture.vk_image() }.is_some();
        (texture, in_place, made(&before, &after))
    };

    // A target whose renders passed: whether the device holds the shape in place at all.
    set([
        (Access::Reads, Verdict::Passed),
        (Access::Renders, Verdict::Passed),
        (Access::UnobservedRenders, Verdict::Passed),
        (Access::Copies, Verdict::Unchecked),
    ]);
    checks.note_memory(PagesClass::Pointer, flags);
    let (rendered, in_place, counts) = make();
    if !in_place {
        println!(
            "{TEST} on {name}: a target whose renders passed is no image in place (dma-buf \
             images, shadows for a failed verdict: {counts:?}): not run"
        );
        return;
    }
    drop(rendered);

    // Lent through memory of one type.
    set([
        (Access::Reads, Verdict::Passed),
        (Access::Renders, Verdict::Failed),
        (Access::UnobservedRenders, Verdict::Passed),
        (Access::Copies, Verdict::Unchecked),
    ]);
    let announced = notices(Degradation::LineWriteBack).len();
    let (lent, in_place, counts) = make();
    assert!(
        in_place && counts == (1, 0),
        "{TEST}: the target over pages whose class's copies another class in memory of the same \
         type failed is its image in place (in place {in_place}; dma-buf images, shadows for a \
         failed verdict: {counts:?})"
    );
    assert_eq!(
        notices(Degradation::LineWriteBack).len(),
        announced + 1,
        "{TEST}: announced once as line-write-back"
    );
    assert!(
        texels_of(&queue, &lent) == texels_in(&counting(file, 7)),
        "{TEST}: the image in place reads the pages' bytes"
    );

    // Not across memory types, the negative control.
    let other = flags ^ vk::MemoryPropertyFlags::HOST_CACHED;
    checks.note_memory(PagesClass::Pointer, other);
    let shadows = notices(Degradation::ShadowedTexture).len();
    let (control, in_place, counts) = make();
    let reasons = notices(Degradation::ShadowedTexture);
    assert!(
        !in_place && counts == (0, 1),
        "{TEST}: the control, the pointer class's pages in memory of other flags, a shadow for its \
         failed renders (in place {in_place}; dma-buf images, shadows for a failed verdict: \
         {counts:?})"
    );
    assert!(
        reasons.len() == shadows + 1
            && reasons[shadows].contains("made no copy of a texture's texels into such rows"),
        "{TEST}: the control's notice names the copies' verdict: {:?}",
        &reasons[shadows..]
    );
    println!(
        "{TEST} on {name}: lent through memory {:#x} in place; through {:#x} a shadow: {}",
        flags.as_raw(),
        other.as_raw(),
        reasons[shadows]
    );
    drop(control);
    drop(lent);
    drop(buffer);
    drop(queue);
    drop(device);
    drop(ram);
}
