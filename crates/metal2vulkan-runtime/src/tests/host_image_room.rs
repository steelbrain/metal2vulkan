//! `Shared` depth textures past the room of the heap their host-transfer images lie in
//! (`memory::host_image_types`, `texture/host_copy.rs`). Such an image takes its class's own memory
//! types alone, never one the class overflows into: where none of them has room, the texture's
//! image is made without host-transfer usage, in device-local memory where that image has room.
//! An RTX 3090 without resizable BAR (NVIDIA 620.70.06) places a host-transfer image in its
//! 246 MiB BAR heap alone, which every process's devices share, and the image without the usage
//! in its device-local heap of 24 GiB; the types it would overflow into are system memory
//! (`device-memory-overflow`).
//!
//! Where the first texture's image takes the usage in a device-local type whose heap holds at most
//! [`FILLED_AT_MOST`], textures of [`SIDE`] squared `Depth32Float` texels are made until their
//! bytes pass the heap's size by two blocks of images: none overflows, each lies in device-local
//! memory, and the first and the last read back the depths their CPU copies wrote. Elsewhere
//! (no host copies, or the usage's heap is larger) the test says why and makes nothing more. The
//! textures made with and without the usage are printed (`HOST-IMAGE-ROOM`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch};
use crate::device::lock;
use crate::memory::{BLOCK_BYTES, IMAGE_BLOCK_SLACK};
use crate::{Region, ResourceOptions, Texture, TextureDescriptor};
use ash::vk::MemoryPropertyFlags;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::vk::ImageUsageFlags;
use std::collections::BTreeSet;

/// The side of each texture: 16 MiB of `Depth32Float`, a quarter of a block of images.
const SIDE: u64 = 2048;

/// The largest heap of host-transfer images the test fills.
const FILLED_AT_MOST: u64 = 1 << 30;

/// Whether `texture`'s image takes host-transfer usage.
fn hosted(texture: &Texture) -> bool {
    let usage = texture.image_plan().unwrap().usage;
    usage.contains(ImageUsageFlags::HOST_TRANSFER)
}

/// `depth` in each of the region's 4 x 4 texels at the texture's corner, written by
/// `replace_region` and read back by `get_bytes`.
fn round_trip(texture: &Texture, depth: f32) -> Vec<f32> {
    let region = Region::new_2d(0, 0, 4, 4);
    let bytes: Vec<u8> = (0..16).flat_map(|_| depth.to_le_bytes()).collect();
    texture.replace_region(region, 0, 0, &bytes, 16, 0).unwrap();
    let mut read = vec![0u8; bytes.len()];
    texture.get_bytes(&mut read, 16, 0, region, 0, 0).unwrap();
    (read.chunks_exact(4))
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

#[test]
fn shared_depth_textures_past_their_host_transfer_heap_lie_in_device_local_memory() {
    let scratch = Scratch::new("host-image-room");
    let device = device(&scratch.0);
    let name = device.name();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let first = device.new_texture(&descriptor).unwrap();
    let kind = |texture: &Texture| {
        let index = texture.placed_type().unwrap();
        lock(&device.shared().memory).type_and_heap(index)
    };
    let first_type = first.placed_type().unwrap();
    let (flags, heap) = kind(&first);
    if !hosted(&first) || !flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
        println!(
            "HOST-IMAGE-ROOM {name}: the first texture takes no host-transfer usage in \
             device-local memory (usage {}, type flags {:#x}): nothing to fill",
            hosted(&first),
            flags.as_raw()
        );
        return;
    }
    if heap > FILLED_AT_MOST {
        println!(
            "HOST-IMAGE-ROOM {name}: host-transfer images lie in a heap of {heap} bytes, past \
             the {FILLED_AT_MOST} the test fills"
        );
        return;
    }
    let bytes = SIDE * SIDE * 4;
    let blocks = 2 * (BLOCK_BYTES + BLOCK_BYTES / IMAGE_BLOCK_SLACK);
    let count = (heap + blocks).div_ceil(bytes) + 1;
    let before = device.work_counts().device_memory_overflows;
    let mut textures = vec![first];
    for _ in 1..count {
        textures.push(device.new_texture(&descriptor).unwrap());
    }
    let with = textures.iter().filter(|texture| hosted(texture)).count();
    let apart: BTreeSet<u32> = (textures.iter())
        .filter(|texture| !hosted(texture))
        .filter_map(Texture::placed_type)
        .collect();
    let overflowed = textures
        .iter()
        .filter(|texture| texture.overflowed())
        .count();
    println!(
        "HOST-IMAGE-ROOM {name}: host-transfer images in type {first_type} (flags {:#x}), a heap \
         of {heap} bytes; {} textures of {bytes} bytes: {with} with host-transfer usage, {} \
         without (in types {apart:?}), {overflowed} overflowed",
        flags.as_raw(),
        textures.len(),
        textures.len() - with
    );
    assert_eq!(
        overflowed, 0,
        "{name}: textures lay past device-local memory"
    );
    assert_eq!(device.work_counts().device_memory_overflows, before);
    for texture in &textures {
        let (flags, _) = kind(texture);
        assert!(
            flags.contains(MemoryPropertyFlags::DEVICE_LOCAL),
            "{name}: a texture lies in memory of flags {:#x}",
            flags.as_raw()
        );
    }
    let last = textures.last().unwrap();
    assert_eq!(round_trip(&textures[0], 0.25), [0.25; 16]);
    assert_eq!(round_trip(last, 0.75), [0.75; 16]);
}
