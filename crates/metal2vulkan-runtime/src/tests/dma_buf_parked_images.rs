//! Images in place over a no-copy buffer's dma-buf pages, parked as their texture goes and taken
//! by the next texture over the same pages (`dma_buf/parked.rs`), on the device the cfg selects
//! made without `VK_EXT_external_memory_host`, so that its no-copy buffers are imported as
//! dma-bufs of their pages (`tests/no_pointer_import.rs`): a 64x32 `BGRA8Unorm` texture that only
//! reads, rows 256 bytes apart, at offset 0 of a no-copy buffer over a `memfd`'s pages mapped
//! shared, whose source makes `udmabuf`s of them and counts its calls. Each texture's texels are
//! read by a blit into a buffer of the device's own.
//!
//! - **Taken after a use.** The first texture, made anew, read, then dropped, parks its image
//!   (its first move made); the CPU then writes other bytes into the pages; a second texture over
//!   the same bytes takes the image: one dma-buf image made for the two
//!   (`WorkCounts::dma_buf_images`), one taken (`dma_buf_images_taken`), the source asked for
//!   neither the second texture nor its first use, and its blit reads the CPU's new bytes. A third
//!   texture made while the second lives finds none parked and is made anew
//!   (`dma_buf_images_missed`), the source asked once: it reads what the second reads, and so
//!   do both after the CPU writes the pages again.
//! - **Taken before a use.** Every parked image released (`Device::release_idle_host_imports`),
//!   a texture made and dropped unused parks an image no submission moved; the next texture takes
//!   it, and its first use, which makes the acquire a new image's makes, reads the pages.
//! - **Not parked** while the device parks none (`Device::park_dma_buf_images_at_most(0)`, as
//!   past its most): the texture's image goes with its import as the texture goes.
//! - **Released** untaken by a remap of the buffer's bytes reported
//!   (`Device::host_range_remapped`); by the residency thread's next pass where the device's held
//!   imports cost every submission (the import tax forced `Holds`), where a pass within its idle
//!   window keeps it while they cost nothing (forced `Fails`); and as the buffer's import goes,
//!   which leaves the device's dma-buf image imports as they were before the textures.
//!
//! A negative control, run by hand: with the take disabled (`Device::take_parked` answering
//! `None`), the second texture is made anew and the test fails at its counts.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, the device lists no dma-buf route or makes no
//! `DRM_FORMAT_MOD_LINEAR` images, or the texture is a shadow (the device's check did not pass
//! for such pages), it says so and passes without running.

use super::host_residency::{
    counting, host_page, own_dma_buf_images, udmabuf_openable, Ram, Scratch,
};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::submit_check::{Check, Verdict};
use crate::{
    BlitOption, Buffer, CommandQueue, Device, HostMemory, Origin, ResourceOptions, Size, Texture,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const TEST: &str = "PARKED-IMAGES";
/// The texture's width and height, and the bytes from a row's first texel to the next row's: its
/// rows' bytes, a multiple of the 256 RADV lays a plane's rows at.
const WIDTH: u64 = 64;
const HEIGHT: u64 = 32;
const ROW: u64 = 4 * WIDTH;
const CHECK_WAIT: Duration = Duration::from_secs(120);
/// The longest the test waits for a release on another thread (a completed command buffer's).
const SETTLE: Duration = Duration::from_secs(20);

/// The texture's texels, copied out by a blit.
fn texels_of(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let out = (queue.device())
        .new_buffer(ROW * HEIGHT, ResourceOptions::shared())
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
            ROW,
            ROW * HEIGHT,
            BlitOption::None,
        ))
        .unwrap()
    });
    read(&out, ROW * HEIGHT)
}

/// The device's counts once `until` holds of them, or after [`SETTLE`].
fn counts_until(device: &Device, until: impl Fn(&WorkCounts) -> bool) -> WorkCounts {
    let start = Instant::now();
    loop {
        let counts = device.work_counts();
        if until(&counts) || start.elapsed() > SETTLE {
            return counts;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The checks the module's docs list.
#[test]
fn a_texture_over_the_pages_of_one_that_went_takes_its_image() {
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
    // The images in dma-buf imports of their own, which park, whatever the check of images in
    // the buffer's import answered.
    own_dma_buf_images(&device);
    let queue = device.new_command_queue().unwrap();
    let length = ROW * HEIGHT;
    let file = (length as usize).next_multiple_of(host_page());
    let ram = Ram::new(file, 0, file);
    let bytes = |seed: usize| counting(length as usize, seed);
    ram.write_file(0, &bytes(7));
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
    let by_buffer = asked();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;

    // Taken after a use.
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let made = device.work_counts();
    if made.dma_buf_textures_in_place == before.dma_buf_textures_in_place {
        println!(
            "{TEST} on {name}: the texture is a shadow (the check: {:?}): not run",
            device.in_place_check_report()
        );
        return;
    }
    assert_eq!(
        (
            made.dma_buf_images_missed - before.dma_buf_images_missed,
            texture.first_move_waits()
        ),
        (1, true),
        "{TEST}: the first texture's image made anew, its acquire waiting"
    );
    assert!(
        texels_of(&queue, &texture) == bytes(7),
        "{TEST}: the first texture's texels"
    );
    drop(texture);
    let parked = counts_until(&device, |counts| {
        counts.dma_buf_images_parked > made.dma_buf_images_parked
    });
    assert_eq!(
        parked.dma_buf_images_parked - made.dma_buf_images_parked,
        1,
        "{TEST}: the first texture's image parked"
    );
    ram.write_file(0, &bytes(101));
    let asked_before = asked();
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let taken = device.work_counts();
    assert!(
        !texture.first_move_waits(),
        "{TEST}: the image taken after its move rests in GENERAL, no move noted"
    );
    let second = texels_of(&queue, &texture);
    let used = device.work_counts();
    assert_eq!(
        (
            taken.dma_buf_images - before.dma_buf_images,
            taken.dma_buf_images_taken - before.dma_buf_images_taken,
        ),
        (1, 1),
        "{TEST}: one image made for the two textures, one taken"
    );
    assert_eq!(
        (asked(), used.dma_buf_images),
        (asked_before, taken.dma_buf_images),
        "{TEST}: the source asked for neither the second texture nor its first use"
    );
    assert!(
        second == bytes(101),
        "{TEST}: the second texture's texels, the CPU's later bytes"
    );
    let fresh = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let anew = device.work_counts();
    assert_eq!(
        (
            anew.dma_buf_images_missed - taken.dma_buf_images_missed,
            asked()
        ),
        (1, asked_before + 1),
        "{TEST}: the third texture, made while the second lives, made anew"
    );
    assert!(
        texels_of(&queue, &fresh) == second,
        "{TEST}: a new image reads what the taken one reads"
    );
    ram.write_file(0, &bytes(211));
    assert!(
        texels_of(&queue, &texture) == bytes(211) && texels_of(&queue, &fresh) == bytes(211),
        "{TEST}: the taken image and the new one, after the CPU's next bytes"
    );
    drop(fresh);
    drop(texture);
    let both = counts_until(&device, |counts| {
        counts.dma_buf_images_parked >= anew.dma_buf_images_parked + 2
    });
    assert_eq!(
        both.dma_buf_images_parked - anew.dma_buf_images_parked,
        2,
        "{TEST}: both images parked"
    );

    // Taken before a use.
    device.release_idle_host_imports();
    let released = device.work_counts();
    assert_eq!(
        released.dma_buf_images_parked_released - both.dma_buf_images_parked_released,
        2,
        "{TEST}: both released"
    );
    drop(buffer.new_texture(&descriptor, 0, ROW).unwrap());
    let unmoved = counts_until(&device, |counts| {
        counts.dma_buf_images_parked > released.dma_buf_images_parked
    });
    ram.write_file(0, &bytes(59));
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let again = device.work_counts();
    assert_eq!(
        (
            again.dma_buf_images_taken - unmoved.dma_buf_images_taken,
            texture.first_move_waits()
        ),
        (1, true),
        "{TEST}: the unused texture's image taken, its acquire waiting as a new one's"
    );
    assert!(
        texels_of(&queue, &texture) == bytes(59),
        "{TEST}: the first use of an image no submission moved reads the pages"
    );
    drop(texture);

    // Released by a remap reported, and as the buffer's import goes.
    let reparked = counts_until(&device, |counts| {
        counts.dma_buf_images_parked > again.dma_buf_images_parked
    });
    device.host_range_remapped(ram.pointer(0), length);
    let remapped = device.work_counts();
    assert_eq!(
        (remapped.dma_buf_images_parked_released) - reparked.dma_buf_images_parked_released,
        1,
        "{TEST}: released by the remap"
    );
    // Not parked while the device parks none.
    device.park_dma_buf_images_at_most(0);
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(texels_of(&queue, &texture) == bytes(59), "{TEST}");
    let unparked = device.work_counts();
    drop(texture);
    let gone_unparked = counts_until(&device, |counts| {
        counts.dma_buf_image_memories_live < unparked.dma_buf_image_memories_live
    });
    device.park_dma_buf_images_at_most(crate::host_residency::PARKED_MOST);
    assert_eq!(
        (
            gone_unparked.dma_buf_images_parked,
            gone_unparked.dma_buf_image_memories_live + 1
        ),
        (
            remapped.dma_buf_images_parked,
            unparked.dma_buf_image_memories_live
        ),
        "{TEST}: none parked while the device parks none, the image's import freed"
    );
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(texels_of(&queue, &texture) == bytes(59), "{TEST}");
    drop(texture);
    let parked = counts_until(&device, |counts| {
        counts.dma_buf_images_parked > remapped.dma_buf_images_parked
    });
    // Kept by a pass within its window where held imports cost nothing, released by the next
    // pass where they cost every submission.
    assert!(
        device.settle_submit_checks(CHECK_WAIT),
        "{TEST}: the submission checks"
    );
    let checks = &device.shared().submit_checks;
    let measured = checks.verdict(Check::ImportTax);
    checks.force(Check::ImportTax, Verdict::Fails);
    device.shared().residency.sweep(false);
    let kept = device.work_counts();
    checks.force(Check::ImportTax, Verdict::Holds);
    device.shared().residency.sweep(false);
    let swept = device.work_counts();
    checks.force(Check::ImportTax, measured);
    let released = |counts: &WorkCounts| {
        counts.dma_buf_images_parked_released - parked.dma_buf_images_parked_released
    };
    assert_eq!(
        (released(&kept), released(&swept)),
        (0, 1),
        "{TEST}: kept by a pass within its window where held imports cost nothing, released by \
         the next where they cost every submission"
    );
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    assert!(texels_of(&queue, &texture) == bytes(59), "{TEST}");
    drop(texture);
    let last = counts_until(&device, |counts| {
        counts.dma_buf_images_parked > swept.dma_buf_images_parked
    });
    drop(buffer);
    let gone = counts_until(&device, |counts| {
        counts.dma_buf_images_parked_released > last.dma_buf_images_parked_released
            && counts.dma_buf_image_memories_live <= before.dma_buf_image_memories_live
    });
    assert_eq!(
        gone.dma_buf_images_parked_released - last.dma_buf_images_parked_released,
        1,
        "{TEST}: released as the buffer's import went"
    );
    assert!(
        gone.dma_buf_image_memories_live <= before.dma_buf_image_memories_live,
        "{TEST}: the images' imports all freed: {} live, {} before",
        gone.dma_buf_image_memories_live,
        before.dma_buf_image_memories_live
    );
    println!(
        "{TEST} on {name}: images made {}, parked {}, taken {}, missed {}, released {}; the \
         source asked {} times for 8 textures",
        gone.dma_buf_images - before.dma_buf_images,
        gone.dma_buf_images_parked - before.dma_buf_images_parked,
        gone.dma_buf_images_taken - before.dma_buf_images_taken,
        gone.dma_buf_images_missed - before.dma_buf_images_missed,
        gone.dma_buf_images_parked_released - before.dma_buf_images_parked_released,
        asked() - by_buffer,
    );
    drop(queue);
    drop(device);
    drop(ram);
}
