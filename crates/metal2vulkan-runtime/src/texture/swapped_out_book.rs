//! A `VkImage` the image recycler's book holds as live leaves it as the objects it was swapped out
//! with go (`ImageObjects`; `recycle.rs`, "The book"), checked on the objects themselves, so it
//! holds on every device, whether or not the device's passes take a storage form
//! (`texture/storage_form.rs`, whose swapped-out first image goes so).
//!
//! An image is made as a `Private` `BGRA8Unorm` render target's is (`Device::recycled_image`,
//! which notes it in the book under the target's making), then goes as a storage form's
//! swapped-out first image does: in `ImageObjects` that destroy it. The book no longer holds its
//! handle, and it was counted as it left (`WorkCounts::object_churn.swapped_out_images_forgotten`).
//! Then `Shared` `RGBA16Float` textures are made, each kept, until one's image takes that handle
//! where the driver gives it again (printed, not asserted: whether the driver gives it again is the
//! driver's), which is released: where the device copies on the host, such a texture's image is
//! one the book does not note (`Device::host_placed_image`). A texture of the target's descriptor
//! made after takes no parked image (`images_recycled`): a handle left in the book would have
//! parked the `RGBA16Float` image as the target's making, and handed it to that texture.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::first_move_lifetimes::{device, Scratch};
use super::{ImageObjects, RawGoes};
use crate::device::lock;
use crate::{ResourceOptions, TextureDescriptor};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The most `Shared` `RGBA16Float` textures made while looking for one at the destroyed image's
/// handle.
const TRIES: usize = 64;

/// The textures' side, in texels.
const SIDE: u64 = 64;

/// A `Private` `BGRA8Unorm` render target's descriptor, a guest framebuffer's format.
fn target() -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    d
}

#[test]
fn a_swapped_out_image_leaves_the_recycler_s_book_as_it_goes() {
    let cache = Scratch::new("swapped-out-book");
    let device = device(&cache.0);
    let shared = device.shared();
    let descriptor = target();
    // Kept to the end: its plan is the making's, and its image is never parked for it.
    let kept_target = device.new_texture(&descriptor).unwrap();
    let plan = kept_target.image_plan().unwrap().clone();
    let storage = descriptor.resource_options.storage_mode().unwrap();
    let class = plan.memory_class(storage, &device.properties().driver_defects);
    let (raw, placement) = device
        .recycled_image(&plan, TextureType::Type2D, class, shared.sharing())
        .unwrap();
    let in_book = || lock(&shared.textures.recycled).holds(raw);
    assert!(in_book(), "an image made as a texture's is in the book");
    let before = device.work_counts().object_churn;
    drop(ImageObjects {
        device: device.clone(),
        raw,
        views: Vec::new(),
        raw_goes: RawGoes::Destroyed,
        import: None,
    });
    lock(&shared.memory).free(shared.vk(), &placement);
    let forgotten = !in_book();
    let left = device
        .work_counts()
        .object_churn
        .swapped_out_images_forgotten
        - before.swapped_out_images_forgotten;
    // Kept, so the driver gives none of their handles again; the one at the destroyed image's
    // handle goes, through the recycler.
    let mut kept = Vec::with_capacity(TRIES);
    let mut given_again = false;
    for _ in 0..TRIES {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let texture = device.new_texture(&d).unwrap();
        if texture.raw_image().unwrap() == raw {
            given_again = true;
            drop(texture);
            break;
        }
        kept.push(texture);
    }
    let recycled = device.work_counts().object_churn.images_recycled;
    let next = device.new_texture(&descriptor).unwrap();
    let took = device.work_counts().object_churn.images_recycled - recycled;
    eprintln!(
        "SWAPPED-OUT-BOOK {}: the destroyed image left the book {forgotten} (counted {left}), its \
         handle given again {given_again} ({} tries), parked images the next target took {took}",
        device.name(),
        kept.len() + usize::from(given_again)
    );
    assert!(
        forgotten,
        "the image, destroyed, is still in the recycler's book of live images"
    );
    assert_eq!(
        left, 1,
        "the image counted as it left the book (`swapped_out_images_forgotten`)"
    );
    assert_eq!(
        took, 0,
        "the next texture of the target's descriptor took a parked image: no image of that making \
         was parked"
    );
    drop((next, kept, kept_target));
}
