//! A device whose making of a block of plain memory took longer than `memory::MAKING_BOUND` makes
//! its later pooled blocks of plain memory of the largest power of two that making's rate makes
//! within the bound with a sixteenth of it besides (`memory::PLAIN_BLOCK_SLACK`), the power of
//! two down to `memory::MIN_PLAIN_BLOCK_BYTES` (`Memory::plain_block_made`): on RADV such a
//! making holds every submission of the process for as long as it runs. A placement larger than
//! such a block, up to half of `memory::BLOCK_BYTES`, takes a block of the least power of two so
//! made that holds it, and the blocks of exportable buffers keep their size.
//!
//! The first two tests are of the memory's bookkeeping alone. The third makes textures on a device
//! told of a slow making and reads the sizes of the blocks of images it made: device tests compile
//! only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use crate::memory::{pooled_block_bytes, Memory, BLOCK_BYTES, MAKING_BOUND, MIN_PLAIN_BLOCK_BYTES};
use ash::vk;
use std::time::Duration;

const MIB: u64 = 1 << 20;

/// The memory of a device of one host-visible memory type, whose largest allocation is
/// `max_allocation` bytes.
fn memory_of(max_allocation: u64) -> Memory {
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        memory_heap_count: 1,
        ..Default::default()
    };
    properties.memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        heap_index: 0,
    };
    properties.memory_heaps[0] = vk::MemoryHeap {
        size: 8 << 30,
        flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
    };
    Memory::new(
        &properties,
        max_allocation,
        false,
        true,
        vec![0],
        None,
        None,
    )
}

#[test]
fn a_making_past_the_bound_makes_the_later_plain_blocks_of_what_its_rate_makes_within_it() {
    let ms = |micros: u64| Duration::from_micros(micros);
    let mut memory = memory_of(u64::MAX);
    assert_eq!(memory.plain_block_bytes(), BLOCK_BYTES);
    // A making within the bound leaves the size.
    memory.plain_block_made(BLOCK_BYTES, MAKING_BOUND);
    assert_eq!(memory.plain_block_bytes(), BLOCK_BYTES);
    // 64 MiB in 9.6 ms (a Radeon 780M's slowest on a quiet host, the kernel's page pools empty)
    // makes 53 MiB in the bound: blocks of 32 MiB and 2 MiB.
    memory.plain_block_made(BLOCK_BYTES, ms(9_600));
    assert_eq!(memory.plain_block_bytes(), 32 * MIB);
    // A fast making never makes them larger again, nor a slow one whose rate makes more.
    memory.plain_block_made(32 * MIB, Duration::ZERO);
    assert_eq!(memory.plain_block_bytes(), 32 * MIB);
    memory.plain_block_made(4 * BLOCK_BYTES, MAKING_BOUND * 2);
    assert_eq!(memory.plain_block_bytes(), 32 * MIB);
    // A dedicated block of an odd size: 48 MiB in twice the bound makes 24 MiB in it.
    memory.plain_block_made(48 * MIB, MAKING_BOUND * 2);
    assert_eq!(memory.plain_block_bytes(), 16 * MIB);
    // A power of two the rate makes within the bound whose sixteenth besides it does not: 64 MiB
    // in twice the bound makes 32 MiB in it, and a block of 32 MiB and 2 MiB would pass it.
    let mut edge = memory_of(u64::MAX);
    edge.plain_block_made(BLOCK_BYTES, MAKING_BOUND * 2);
    assert_eq!(edge.plain_block_bytes(), 16 * MIB);
    // Never below the least.
    memory.plain_block_made(MIN_PLAIN_BLOCK_BYTES, MAKING_BOUND * 100);
    assert_eq!(memory.plain_block_bytes(), MIN_PLAIN_BLOCK_BYTES);
    // 64 MiB in 23.1 ms (the 780M's median with its free memory in single pages) makes 22 MiB in
    // the bound: one such making takes the blocks to 16 MiB.
    let mut fragmented = memory_of(u64::MAX);
    fragmented.plain_block_made(BLOCK_BYTES, ms(23_100));
    assert_eq!(fragmented.plain_block_bytes(), 16 * MIB);
    // A device whose largest allocation is below the least keeps its blocks of that size.
    let mut small = memory_of(4 * MIB);
    small.plain_block_made(4 * MIB, MAKING_BOUND * 100);
    assert_eq!(small.plain_block_bytes(), 4 * MIB);
}

#[test]
fn a_pooled_block_holds_its_placement_and_exportable_blocks_keep_their_size() {
    let plain = Some(16 * MIB);
    // Blocks of 16 MiB and a sixteenth: four images of 4 MiB of texels and up to 256 KiB of
    // metadata and alignment each.
    assert_eq!(pooled_block_bytes(BLOCK_BYTES, plain, 0), 17 * MIB);
    assert_eq!(pooled_block_bytes(BLOCK_BYTES, plain, MIB), 17 * MIB);
    assert_eq!(pooled_block_bytes(BLOCK_BYTES, plain, 17 * MIB), 17 * MIB);
    assert_eq!(
        pooled_block_bytes(BLOCK_BYTES, plain, 17 * MIB + 1),
        34 * MIB
    );
    assert_eq!(
        pooled_block_bytes(BLOCK_BYTES, plain, BLOCK_BYTES / 2),
        34 * MIB
    );
    // A largest allocation that is no power of two bounds the block.
    assert_eq!(pooled_block_bytes(48 * MIB, plain, 20 * MIB), 34 * MIB);
    assert_eq!(pooled_block_bytes(33 * MIB, plain, 20 * MIB), 33 * MIB);
    assert_eq!(
        pooled_block_bytes(48 * MIB, Some(48 * MIB), 20 * MIB),
        48 * MIB
    );
    // Blocks of plain memory not made smaller keep the block's size, with no sixteenth besides.
    assert_eq!(
        pooled_block_bytes(BLOCK_BYTES, Some(BLOCK_BYTES), MIB),
        BLOCK_BYTES
    );
    // The blocks of exportable buffers.
    assert_eq!(pooled_block_bytes(BLOCK_BYTES, None, MIB), BLOCK_BYTES);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod device {
    use super::{BLOCK_BYTES, MAKING_BOUND, MIB, MIN_PLAIN_BLOCK_BYTES};
    use crate::memory::Holds;
    use crate::tests::freed_memory::device;
    use crate::{Device, ResourceOptions, Texture, TextureDescriptor};
    use metal2vulkan::metal::{PixelFormat, TextureUsage};

    /// The textures each run makes at most before the image pool makes the block it waits for.
    const MOST: usize = 256;

    /// A pooled block of plain memory of the least size: 16 MiB and a sixteenth.
    const SMALL_BLOCK: u64 = MIN_PLAIN_BLOCK_BYTES + MIN_PLAIN_BLOCK_BYTES / 16;

    /// A `Private` `RGBA8Unorm` texture of `width` by `height`, sampled and rendered to.
    fn texture(device: &Device, width: u64, height: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        device.new_texture(&descriptor).unwrap()
    }

    /// The device's pooled blocks of images, by size.
    fn image_blocks(device: &Device) -> Vec<u64> {
        let mut sizes =
            crate::device::lock(&device.shared().memory).pooled_block_sizes(Holds::Images);
        sizes.sort_unstable();
        sizes
    }

    /// The blocks in `after` that `before` does not hold, as many times as they are more.
    fn new_blocks(before: &[u64], after: &[u64]) -> Vec<u64> {
        let mut left = before.to_vec();
        (after.iter())
            .filter(|size| match left.iter().position(|kept| kept == *size) {
                Some(at) => {
                    left.swap_remove(at);
                    false
                }
                None => true,
            })
            .copied()
            .collect()
    }

    /// Make textures of `width` by `height` until the device made a block of images of more than
    /// `past` bytes: the blocks made meanwhile.
    fn until_a_block(
        device: &Device,
        (width, height): (u64, u64),
        past: u64,
        held: &mut Vec<Texture>,
    ) -> Vec<u64> {
        let before = image_blocks(device);
        for _ in 0..MOST {
            held.push(texture(device, width, height));
            let made = new_blocks(&before, &image_blocks(device));
            if made.iter().any(|&size| size > past) {
                return made;
            }
        }
        panic!("{MOST} textures of {width} x {height} made no block of images past {past} bytes");
    }

    #[test]
    fn textures_after_a_slow_making_lie_in_smaller_blocks_or_one_that_holds_them() {
        let scratch = std::env::temp_dir().join(format!("m2v-plain-blocks-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        {
            // A pooled block's making that took four times the bound.
            let mut memory = crate::device::lock(&device.shared().memory);
            memory.plain_block_made(BLOCK_BYTES, MAKING_BOUND * 4);
            assert_eq!(memory.plain_block_bytes(), MIN_PLAIN_BLOCK_BYTES);
        }
        let mut held = Vec::new();
        // 1 MiB textures: blocks of the least size, 16 MiB and a sixteenth.
        let small = until_a_block(&device, (512, 512), 0, &mut held);
        // 20 MiB ones: past that size, a block of 32 MiB and a sixteenth each.
        let large = until_a_block(&device, (2560, 2048), SMALL_BLOCK, &mut held);
        println!(
            "PLAIN-BLOCKS {}: after a slow making, 1 MiB textures made blocks of images of {:?} \
             MiB, 20 MiB ones {:?} MiB",
            device.name(),
            small.iter().map(|size| size / MIB).collect::<Vec<_>>(),
            large.iter().map(|size| size / MIB).collect::<Vec<_>>(),
        );
        assert!(
            small.iter().all(|&size| size == SMALL_BLOCK),
            "1 MiB textures made blocks of {small:?} bytes"
        );
        assert!(
            large.contains(&(34 * MIB)) && large.iter().all(|&size| size <= 34 * MIB),
            "20 MiB textures made blocks of {large:?} bytes"
        );
        drop(held);
        drop(device);
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
