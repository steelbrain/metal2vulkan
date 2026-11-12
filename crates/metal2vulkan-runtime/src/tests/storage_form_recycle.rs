//! A texture's first image, which its storage form swapped out, leaves the image recycler's book
//! of live images as it is destroyed (`recycle.rs`, "The book"; `texture/storage_form.rs`).
//!
//! The book holds each live image it may park by its `VkImage` handle, and a driver gives a
//! destroyed image's handle to a later one (MoltenVK's handle is the image object's address). A
//! storage form's first image was destroyed with its handle still in the book: the next image the
//! driver made at that handle, of a kind the book does not note (a `Shared` texture's host-copy
//! image), was parked as the first image's making as it went, and the next texture of that making
//! took an image of another format than its own.
//!
//! A tracked `Private` `BGRA8Unorm` render target whose image lacks `STORAGE`
//! (`DeviceShared::storage_costs_forced`) takes its storage form at the commit of its first pass
//! where the device takes one (`self_read_storage.rs`'s passes: a reader through a snapshot, a
//! fetch through the pixel interlock). Once that command buffer completed and the switch ended,
//! the book no longer holds the first image, which was counted as it left
//! (`WorkCounts::object_churn.swapped_out_images_forgotten`). Then `Shared` `RGBA16Float` textures are made, each
//! kept, until one's image takes the first image's handle where the driver gives it again
//! (printed, not asserted: whether the driver gives it again is the driver's), which is released;
//! a texture of the target's descriptor made after takes no parked image (`images_recycled`), and
//! its passes render and read what a new texture's do. A device that takes no storage form for
//! the target says so and checks nothing more.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use super::self_read_storage::{pass_over, red_after, settled, state, state_of, target, SIDE};
use crate::device::lock;
use crate::{ResourceOptions, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// The most `Shared` `RGBA16Float` textures made while looking for one at the first image's
/// handle.
const TRIES: usize = 64;

/// How long the first image may take to go once the command buffer that swapped it out completed
/// and the switch ended: it goes as the last of the two lets it go.
const GONE_WITHIN: Duration = Duration::from_secs(5);

#[test]
fn a_storage_forms_first_image_leaves_the_recycler_as_it_goes() {
    let cache = Scratch::new("storage-form-recycle");
    let device = device(&cache.0);
    device
        .shared()
        .storage_costs_forced
        .store(true, Ordering::Relaxed);
    let queue = device.new_command_queue().unwrap();
    let fetch = state_of(&device, "engine/fragment_fetch.ll", "fetch_ordered").ok();
    let (fill, read) = (state(&device, "tob_solid"), state(&device, "tob_read"));
    let states = (&fill, &read, fetch.as_ref());
    // The pipelines, made by the same passes over an untracked target, which never takes its
    // storage form, and which lives to the end: its image is never parked for the targets'
    // making.
    let warm = target(&device, false);
    for pass in 0..2 {
        pass_over(&queue, &warm, states, pass);
    }
    settled(&device);
    let first_target = target(&device, true);
    let first = first_target.raw_image().unwrap();
    let in_book = || lock(&device.shared().textures.recycled).holds(first);
    assert!(
        in_book(),
        "the target's first image is one the recycler may park"
    );
    let before = device.work_counts();
    pass_over(&queue, &first_target, states, 0);
    // A command buffer made once the switch rode on a commit waits it out
    // (`Device::storage_form_wait`).
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let forms = device.work_counts().storage_forms_taken - before.storage_forms_taken;
    if forms == 0 {
        eprintln!(
            "STORAGE-FORM-RECYCLE {}: the target took no storage form (fetch {}): nothing to check",
            device.name(),
            fetch.is_some()
        );
        return;
    }
    assert_ne!(
        first_target.raw_image().unwrap(),
        first,
        "the storage form's image swapped in"
    );
    let until = Instant::now() + GONE_WITHIN;
    while in_book() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(1));
    }
    let forgotten = !in_book();
    let left = device
        .work_counts()
        .object_churn
        .swapped_out_images_forgotten
        - before.object_churn.swapped_out_images_forgotten;
    // Kept, so the driver gives none of their handles again; the one at the first image's handle
    // goes, through the recycler.
    let mut kept = Vec::with_capacity(TRIES);
    let mut given_again = false;
    for _ in 0..TRIES {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, 64, 64, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&d).unwrap();
        if texture.raw_image().unwrap() == first {
            given_again = true;
            drop(texture);
            break;
        }
        kept.push(texture);
    }
    let recycled = device.work_counts().object_churn.images_recycled;
    let second = target(&device, true);
    let took = device.work_counts().object_churn.images_recycled - recycled;
    eprintln!(
        "STORAGE-FORM-RECYCLE {}: storage forms {forms}, the first image left the book \
         {forgotten} (swapped-out images forgotten {left}), its handle given again {given_again} \
         ({} tries), parked images the next target took {took}",
        device.name(),
        kept.len() + usize::from(given_again)
    );
    for pass in 0..2 {
        let (reds, _) = pass_over(&queue, &second, states, pass);
        for (i, &red) in reds.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            assert_eq!(
                red,
                red_after(pass, fetch.is_some(), x, y),
                "the next target's pass {pass}: pixel ({x}, {y})"
            );
        }
    }
    assert!(
        forgotten,
        "the first image, destroyed, is still in the recycler's book of live images"
    );
    assert!(
        left >= 1,
        "the first image left the book as it was destroyed (`swapped_out_images_forgotten`)"
    );
    assert_eq!(
        took, 0,
        "the next texture of the target's descriptor took a parked image: no image of that making \
         was parked"
    );
    drop((kept, warm, first_target));
}
