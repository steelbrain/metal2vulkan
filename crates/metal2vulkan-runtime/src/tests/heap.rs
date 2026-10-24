//! `heap`: Metal's measured heap answers that need no device: an automatic heap's placement, the
//! placement alignment against every committed profile's offset alignments, the largest heap of
//! each, and the descriptor's defaults; and how a heap numbers the textures it makes, which decides
//! what an encoder's commands reach.

use super::texture_descriptor::APPLE;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::buffer::LINEAR_IMAGE_ALIGNMENT;
use crate::heap::{heap_size, Numbered, PLACEMENT_ALIGNMENT};
use crate::range::RangeAllocator;
use crate::{
    Error, HeapDescriptor, HeapType, MemoryKind, ResourceOptions, RuntimeGap, StorageMode,
};
use metal2vulkan::device::DeviceDescription;

/// Every committed profile's description: (file stem, description), sorted by stem.
fn committed_descriptions() -> Vec<(String, DeviceDescription)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, DeviceDescription)> = std::fs::read_dir(&dir)
        .expect("validation/profiles is readable")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile is readable");
            let desc = DeviceDescription::from_khronos_json(&text, None)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            (stem, desc)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// An automatic heap of `size` bytes, as the runtime keeps one.
fn automatic(size: u64) -> RangeAllocator {
    RangeAllocator::new(0, size)
}

/// Place a buffer of `length` bytes in `heap` as `Heap::new_buffer` does.
fn place(heap: &mut RangeAllocator, length: u64) -> Option<u64> {
    heap.allocate(length, PLACEMENT_ALIGNMENT)
}

/// Metal's automatic heap (probe 2's `automatic_layout`, an M3 Max and an M4): buffers of 1, 255,
/// 256, 257, 16 and 4096 bytes in a 64-KiB heap land at 0, 256, 512, 768, 1280 and 1536, the used
/// size sums the lengths, and the longest buffer at a multiple of 16 runs from the last one's end
/// rounded up to 16.
#[test]
fn an_automatic_heap_packs_buffers_at_256_byte_steps_as_metal_does() {
    let mut heap = automatic(65536);
    let measured = [
        (1, 0, 1, 65520),
        (255, 256, 256, 65024),
        (256, 512, 512, 64768),
        (257, 768, 769, 64496),
        (16, 1280, 785, 64240),
        (4096, 1536, 4881, 59904),
    ];
    for (length, at, used, available16) in measured {
        assert_eq!(place(&mut heap, length), Some(at), "{length} bytes");
        assert_eq!(heap.used(), used, "used after {length} bytes");
        assert_eq!(
            heap.longest(16),
            available16,
            "available after {length} bytes"
        );
    }
}

/// Metal's automatic heap places first fit at the lowest offset with room, and coalesces the ranges
/// freed (probe 3's `automatic_fit`, the same on the M3 Max and the M4).
#[test]
fn an_automatic_heap_places_first_fit_into_its_holes_as_metal_does() {
    /// An automatic heap and where each named buffer is.
    struct Named {
        heap: RangeAllocator,
        at: BTreeMap<&'static str, (u64, u64)>,
    }
    impl Named {
        fn make(&mut self, name: &'static str, length: u64) -> u64 {
            let offset = place(&mut self.heap, length).unwrap();
            self.at.insert(name, (offset, length));
            offset
        }
        fn free(&mut self, name: &'static str) {
            let (offset, length) = self.at.remove(name).unwrap();
            self.heap.free(offset, length);
        }
    }
    let mut named = Named {
        heap: automatic(65536),
        at: BTreeMap::new(),
    };
    assert_eq!(named.make("a", 1024), 0);
    assert_eq!(named.make("b", 256), 1024);
    assert_eq!(named.make("c", 512), 1280);
    assert_eq!(named.make("d", 256), 1792);
    assert_eq!(named.make("e", 4096), 2048);
    assert_eq!((named.heap.used(), named.heap.longest(16)), (6144, 59392));
    named.free("a");
    named.free("c");
    assert_eq!((named.heap.used(), named.heap.longest(256)), (4608, 59392));
    // First fit, not best fit: the 256 bytes go to the 1024-byte hole at 0, not the 512-byte one.
    assert_eq!(named.make("f", 256), 0);
    assert_eq!(named.make("g", 600), 256);
    assert_eq!(named.make("h", 512), 1280);
    assert_eq!(named.heap.used(), 5976);
    named.free("e");
    assert_eq!((named.heap.used(), named.heap.longest(16)), (1880, 63488));
    assert_eq!(named.make("i", 1), 2048);
    assert_eq!((named.heap.used(), named.heap.longest(16)), (1881, 63472));
    named.free("b");
    named.free("d");
    let heap = &named.heap;
    assert_eq!(
        (heap.used(), heap.longest(16), heap.longest(256)),
        (1369, 63472, 63232)
    );
    // The holes from 856 and from 1792 hold no 700 bytes at a multiple of 256.
    assert_eq!(named.make("j", 700), 2304);
    assert_eq!(named.heap.used(), 2069);
}

/// Metal's automatic heap is full where no free range holds the buffer at a multiple of 256 (probe
/// 1's `automatic_basic`: fifteen 4096-byte buffers after two of 100 bytes in 64 KiB), and the
/// longest buffer it then takes at a multiple of 4096 is none.
#[test]
fn an_automatic_heap_fills_as_metal_s_does() {
    let mut heap = automatic(65536);
    assert_eq!(place(&mut heap, 100), Some(0));
    assert_eq!(place(&mut heap, 100), Some(256));
    assert_eq!((heap.longest(16), heap.longest(4096)), (65168, 61440));
    let filled = (0..)
        .take_while(|_| place(&mut heap, 4096).is_some())
        .count();
    assert_eq!(filled, 15);
    assert_eq!(heap.used(), 61640);
    assert_eq!((heap.longest(16), heap.longest(4096)), (3584, 0));
    // An empty heap's alignments past its size (probe 3's `available_*`).
    let mut heap = automatic(65536);
    assert_eq!(place(&mut heap, 100), Some(0));
    assert_eq!(heap.longest(1), 65436);
    assert_eq!(heap.longest(256), 65280);
    assert_eq!(heap.longest(65536), 0);
    assert_eq!(heap.longest(131072), 0);
}

/// Every committed profile's buffer offset alignments, and the runtime's own (the 16 bytes and the
/// linear-image alignment buffers are placed at), divide the placement alignment, so a heap buffer
/// at a multiple of it is at a multiple of each: Vulkan caps each at 256, as powers of two.
#[test]
fn every_device_s_offset_alignments_divide_the_placement_alignment() {
    let descriptions = committed_descriptions();
    assert!(descriptions.len() > 10, "the committed profiles were read");
    for (stem, description) in descriptions {
        let limits = &description.limits;
        for (name, alignment) in [
            (
                "minStorageBufferOffsetAlignment",
                limits.min_storage_buffer_offset_alignment,
            ),
            (
                "minUniformBufferOffsetAlignment",
                limits.min_uniform_buffer_offset_alignment,
            ),
            (
                "minTexelBufferOffsetAlignment",
                limits.min_texel_buffer_offset_alignment,
            ),
        ] {
            assert!(
                alignment.is_power_of_two() && alignment <= 256,
                "{stem}: {name} {alignment} is past what Vulkan allows"
            );
            assert!(
                PLACEMENT_ALIGNMENT.is_multiple_of(alignment),
                "{stem}: {name} {alignment}"
            );
        }
    }
    assert_eq!(PLACEMENT_ALIGNMENT, 256);
    assert!(PLACEMENT_ALIGNMENT.is_multiple_of(16));
    assert!(PLACEMENT_ALIGNMENT.is_multiple_of(LINEAR_IMAGE_ALIGNMENT));
}

/// A heap's size rounds up to 16 KiB (probe 1's `heapsize_*`), and the largest heap is the lower of
/// one allocation and one `VkBuffer` of the device: past the allocation it is out of memory, past
/// only the `VkBuffer` (ANV among the committed profiles) a typed gap naming `maxBufferSize`.
#[test]
fn a_heap_rounds_to_16_kib_and_fits_one_allocation_and_one_buffer() {
    let descriptions = committed_descriptions();
    let mut past_buffer = Vec::new();
    for (stem, description) in &descriptions {
        let limits = &description.allocation;
        for (asked, size) in [(1, 16384), (4095, 16384), (4096, 16384), (65537, 81920)] {
            assert_eq!(heap_size(asked, limits), Ok(size), "{stem}: {asked} bytes");
        }
        assert_eq!(heap_size(1_048_577, limits), Ok(1_064_960), "{stem}");
        let allocation = limits.max_memory_allocation_size;
        let buffer = match limits.max_buffer_size {
            0 => u64::MAX,
            size => size,
        };
        let largest = allocation.min(buffer) / 16384 * 16384;
        assert_eq!(
            heap_size(largest, limits),
            Ok(largest),
            "{stem}: the largest heap"
        );
        let over = heap_size(largest + 1, limits);
        if buffer < allocation {
            past_buffer.push(stem.clone());
            assert_eq!(
                over,
                Err(Error::NotImplemented(RuntimeGap::HeapPastMaxBufferSize {
                    size: largest + 16384,
                    max_buffer_size: buffer,
                })),
                "{stem}"
            );
        } else {
            assert_eq!(over, Err(Error::OutOfMemory(MemoryKind::Device)), "{stem}");
        }
        assert_eq!(
            heap_size(u64::MAX, limits),
            Err(Error::OutOfMemory(MemoryKind::Device)),
            "{stem}"
        );
    }
    assert!(
        past_buffer.iter().any(|stem| stem.ends_with("-anv")),
        "ANV's maxBufferSize is below its maxMemoryAllocationSize: {past_buffer:?}"
    );
}

/// A heap's buffer takes whole units of the heap's base alignment in a pooled block, and its own
/// size in a block of its own (`buffer.rs`'s `placed`), so the largest heap of every committed
/// profile asks one allocation of at most `maxMemoryAllocationSize`, and its block's buffer of
/// addresses, as long as the block, at most `maxBufferSize`. In whole units of the base, RADV's
/// largest heap (4 GiB less 16 KiB) would ask 4 GiB of an allocation of at most 4 GiB less 4 bytes,
/// and MoltenVK's 128 KiB past its largest allocation.
#[test]
fn the_largest_heap_asks_one_allocation_of_its_device() {
    use crate::buffer::placed;
    use crate::memory::BLOCK_BYTES;
    use crate::texture::heap::HEAP_BASE_ALIGNMENT;
    use ash::vk::MemoryRequirements;
    let heap = |size| MemoryRequirements {
        size,
        alignment: HEAP_BASE_ALIGNMENT,
        memory_type_bits: 1,
    };
    let half = BLOCK_BYTES / 2;
    assert_eq!(placed(heap(16384)).size, HEAP_BASE_ALIGNMENT);
    assert_eq!(placed(heap(half - 16384)).size, half);
    assert_eq!(placed(heap(half + 16384)).size, half + 16384);
    for (stem, description) in committed_descriptions() {
        let limits = &description.allocation;
        let buffer = match limits.max_buffer_size {
            0 => u64::MAX,
            size => size,
        };
        let largest = limits.max_memory_allocation_size.min(buffer) / 16384 * 16384;
        assert_eq!(heap_size(largest, limits), Ok(largest), "{stem}");
        let asked = placed(heap(largest));
        assert_eq!(asked.alignment, HEAP_BASE_ALIGNMENT, "{stem}");
        assert!(
            asked.size <= limits.max_memory_allocation_size && asked.size <= buffer,
            "{stem}: the largest heap, {largest} bytes, asks {} bytes of an allocation of at most \
             {} and a buffer of at most {buffer}",
            asked.size,
            limits.max_memory_allocation_size
        );
    }
}

/// The descriptor's defaults are Metal's (probe 1's `defaults`): an automatic heap of size 0 in
/// `Private` storage, options 32; heap types carry Metal's raw values.
#[test]
fn the_heap_descriptor_defaults_to_metal_s() {
    let descriptor = HeapDescriptor::default();
    assert_eq!(descriptor.heap_type, HeapType::Automatic);
    assert_eq!(descriptor.size, 0);
    assert_eq!(descriptor.resource_options, ResourceOptions::private());
    assert_eq!(descriptor.resource_options.raw(), 32);
    assert_eq!(
        descriptor.resource_options.storage_mode(),
        Ok(StorageMode::Private)
    );
    for (heap_type, raw) in [
        (HeapType::Automatic, 0),
        (HeapType::Placement, 1),
        (HeapType::Sparse, 2),
    ] {
        assert_eq!(heap_type.raw(), raw);
        assert_eq!(HeapType::from_raw(raw), Ok(heap_type));
    }
    assert!(HeapType::from_raw(3).is_err());
}

/// A heap lies in memory its textures' optimal images take where its storage class has such a
/// type, else in memory linear images take, and a texture is an optimal image where that image
/// takes every type of the class the heap may lie in, else a linear one where that takes them. On
/// the RTX 3090 (NVIDIA 615.71.09), whose optimal images take memory types 0
/// and 1 (neither host-visible) and linear color images types 0, 1, 3, 4 and 5, a `Shared` heap's
/// textures are linear and a `Private` heap's optimal; on a device whose depth images take no
/// host-visible memory but its color images do (MoltenVK), a `Shared` heap's color textures are
/// optimal and its depth textures linear.
#[test]
fn a_heap_s_textures_are_linear_where_their_optimal_images_take_none_of_its_memory() {
    use crate::memory::{Memory, MemoryClass};
    use crate::texture::heap::{takes_heap, HeapTypes};
    use ash::vk;
    use vk::MemoryPropertyFlags as F;
    let flags = [
        F::empty(),
        F::DEVICE_LOCAL,
        F::DEVICE_LOCAL,
        F::HOST_VISIBLE | F::HOST_COHERENT,
        F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
        F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT,
    ];
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: flags.len() as u32,
        ..Default::default()
    };
    for (slot, &property_flags) in properties.memory_types.iter_mut().zip(&flags) {
        *slot = vk::MemoryType {
            property_flags,
            heap_index: u32::from(!property_flags.contains(F::DEVICE_LOCAL)),
        };
    }
    let memory = Memory::new(&properties, 1 << 32, true, false, vec![0], None, None);
    // The tiling a texture whose optimal image takes `optimal` gets in a heap of `class`.
    let tiling = |types: HeapTypes, optimal: u32, class| {
        let heap = types.heap_memory(&memory, types.buffer, class);
        if takes_heap(&memory, heap, optimal, class) {
            "optimal"
        } else if takes_heap(&memory, heap, types.linear, class) {
            "linear"
        } else {
            "none"
        }
    };
    let nvidia = HeapTypes {
        buffer: 0x3f,
        color: 0x3,
        depth: 0x3,
        linear: 0x3b,
    };
    assert_eq!(nvidia.heap_memory(&memory, 0x3f, MemoryClass::Shared), 0x3b);
    assert_eq!(tiling(nvidia, nvidia.color, MemoryClass::Shared), "linear");
    assert_eq!(
        tiling(nvidia, nvidia.color, MemoryClass::Private),
        "optimal"
    );
    // Color images take host-visible memory, depth images do not (MoltenVK's shape).
    let split = HeapTypes {
        color: 0x3f,
        depth: 0x7,
        ..nvidia
    };
    assert_eq!(split.heap_memory(&memory, 0x3f, MemoryClass::Shared), 0x3f);
    assert_eq!(tiling(split, split.color, MemoryClass::Shared), "optimal");
    assert_eq!(tiling(split, split.depth, MemoryClass::Shared), "linear");
    assert_eq!(tiling(split, split.depth, MemoryClass::Private), "optimal");
    // An image that takes only some of the types of the class the heap may lie in does not bind
    // there: the heap may be given any of them.
    assert_eq!(tiling(split, 0x2f, MemoryClass::Shared), "linear");
    // A device whose linear images take no host-visible memory either.
    let none = HeapTypes {
        linear: 0x3,
        ..nvidia
    };
    assert_eq!(none.heap_memory(&memory, 0x3f, MemoryClass::Shared), 0x3f);
    assert_eq!(tiling(none, none.color, MemoryClass::Shared), "none");
}

/// A heap's texture buffer takes its row of texels at a multiple of 16 bytes, Metal's answer for
/// every format, width, usage and storage mode a Metal heap probe asked (an M3 Max and an M4
/// alike): `heapTextureSizeAndAlignWithDescriptor:` answered 400 and 16 for a 100-texel
/// `RGBA8Unorm` texture buffer, 1 and 16 for one `R8Unorm` texel.
#[test]
fn a_heap_s_texture_buffer_takes_its_row_at_16_bytes_as_metal_s_does() {
    use crate::texture::heap::texture_buffer_size_and_align;
    use crate::{SizeAndAlign, TextureDescriptor};
    use metal2vulkan::metal::{PixelFormat, TextureUsage};
    let formats = [
        (PixelFormat::R8Unorm, 1),
        (PixelFormat::RG16Float, 4),
        (PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::R32Uint, 4),
        (PixelFormat::RGBA32Float, 16),
    ];
    let usages = [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    ];
    for options in [ResourceOptions::shared(), ResourceOptions::private()] {
        for (pixel_format, texel) in formats {
            for width in [1, 3, 64, 100, 1000, 4096, 65536] {
                for usage in usages {
                    let descriptor =
                        TextureDescriptor::texture_buffer(pixel_format, width, options, usage);
                    let layout = descriptor.check(APPLE).unwrap();
                    assert_eq!(
                        texture_buffer_size_and_align(&layout).unwrap(),
                        SizeAndAlign {
                            size: width * texel,
                            align: 16
                        },
                        "{pixel_format:?} x {width}, {usage:?}"
                    );
                }
            }
        }
    }
}

/// A texture with a depth twin takes the twin's bytes of a heap past its image's, at the twin's
/// alignment, inside the answered size, so a texture at any multiple of the answered alignment has
/// each image at a multiple of its own and the next texture's bytes after both; a texture without
/// one takes its image's, in whole granularity pages; on a device whose driver answers a compressed
/// image's size short of its compression data (ANV on Xe-LP GPUs), each image and the texture also
/// take whole 256-byte chunks.
#[test]
fn a_depth_twin_lies_past_its_image_inside_the_heap_texture_s_bytes() {
    use crate::texture::heap::heap_extent;
    use crate::texture::ImagePlacement;
    use crate::SizeAndAlign;
    use ash::vk::MemoryRequirements;
    use metal2vulkan::error::RuntimeDriverDefect;
    use std::collections::BTreeSet;
    let requirements = |size, alignment| MemoryRequirements {
        size,
        alignment,
        memory_type_bits: 1,
    };
    let pages = |granularity| ImagePlacement::new(granularity, &BTreeSet::new());
    let chunks = |granularity| {
        let short = RuntimeDriverDefect::ImageSizeShortOfCompressionData;
        ImagePlacement::new(granularity, &BTreeSet::from([short]))
    };
    let answer = |size, align| SizeAndAlign { size, align };
    // RADV's shape on the Radeon 780M: 64 KiB alignment for the larger images, 4 KiB for the small.
    let image = requirements(16512, 4096);
    let twin = requirements(16384, 65536);
    let (got, offset) = heap_extent(&image, Some(&twin), pages(1024));
    assert_eq!(offset, Some(65536));
    assert_eq!(got, answer(65536 + 16384, 65536));
    // The twin's start meets its alignment wherever the texture lies.
    for at in [0, 1, 7].map(|n| n * got.align) {
        assert!((at + offset.unwrap()).is_multiple_of(twin.alignment));
    }
    // A granularity coarser than both alignments rounds the twin's offset and the size.
    let (small, twin_small) = (requirements(1000, 256), requirements(900, 256));
    for placement in [pages(4096), chunks(4096)] {
        let (got, offset) = heap_extent(&small, Some(&twin_small), placement);
        assert_eq!((got, offset), (answer(8192, 4096), Some(4096)));
    }
    // Without a twin: the image alone, in whole pages.
    let lone = requirements(16512, 128);
    let (got, offset) = heap_extent(&lone, None, pages(1024));
    assert_eq!((got, offset), (answer(17408, 1024), None));
    // ANV's compressed 4x4 `RGBA8Unorm` image on an Xe-LP GPU: 4112 bytes at 64 KiB, in whole
    // chunks there, and with a twin past it.
    let compressed = requirements(4112, 65536);
    let (got, offset) = heap_extent(&compressed, None, chunks(1));
    assert_eq!((got, offset), (answer(4352, 65536), None));
    let (got, offset) = heap_extent(&compressed, Some(&requirements(4112, 65536)), chunks(1));
    assert_eq!((got, offset), (answer(65536 + 4352, 65536), Some(65536)));
    assert_eq!(
        heap_extent(&compressed, None, pages(1)).0,
        answer(4112, 65536)
    );
}

/// A heap texture's footprint is Metal's answer, not the running driver's granularity. A heap's
/// own buffer covers its memory from end to end, so no `bufferImageGranularity` padding can put a
/// heap's image in a granularity page no buffer lies in: the padding separates nothing there and
/// would only widen the number a program places heap resources by. `ImagePlacement::in_heap` drops
/// it and keeps the aux map's chunks, which are the image's own compression data past its reported
/// size.
///
/// Driven by the shape the first conformance replay on an RTX 5080 Laptop GPU (NVIDIA 615.71.09)
/// found, whose `bufferImageGranularity` is 1024 bytes where Metal answers 128 for every
/// descriptor of the family: an image that asks Metal's own bytes at Metal's own alignment answers
/// Metal's, where the padded placement answers 1024; an image that asks a coarser alignment or
/// more bytes than Metal's answer does not fit it ([`images_fit`]), and `Device::heap_plan` holds
/// such a texture to Metal's answer by a shadow of its bytes instead.
#[test]
fn a_heap_texture_s_footprint_takes_no_granularity_padding() {
    use crate::texture::heap::{heap_extent, images_fit};
    use crate::texture::ImagePlacement;
    use crate::SizeAndAlign;
    use ash::vk::MemoryRequirements;
    use metal2vulkan::error::RuntimeDriverDefect;
    use std::collections::BTreeSet;
    let requirements = |size, alignment| MemoryRequirements {
        size,
        alignment,
        memory_type_bits: 1,
    };
    let answer = |size, align| SizeAndAlign { size, align };
    let nvidia = ImagePlacement::new(1024, &BTreeSet::new());
    // Metal's answer for a 64x64 `RGBA8Unorm` texture a shader writes, which its driver's image
    // asks exactly: the answer is Metal's, where the padded placement answers a 1024 alignment.
    let metal = answer(16384, 128);
    let image = requirements(16384, 128);
    assert_eq!(heap_extent(&image, None, nvidia).0, answer(16384, 1024));
    assert_eq!(heap_extent(&image, None, nvidia.in_heap()).0, metal);
    assert!(images_fit(
        heap_extent(&image, None, nvidia.in_heap()).0,
        metal
    ));
    // The 64x64 `RGBA8Unorm` 7-level texture the replay found at 23552 bytes where Metal answers
    // 22528: its image asks 22784 (llvmpipe answered that one for the same descriptor), which the
    // padding carries to a whole page. Without the padding the footprint is the image's own, 256
    // bytes past Metal's answer, which does not fit it: a shadow of the texture's bytes holds it.
    let seven = requirements(22784, 128);
    let metal_seven = answer(22528, 128);
    assert_eq!(heap_extent(&seven, None, nvidia).0, answer(23552, 1024));
    assert_eq!(
        heap_extent(&seven, None, nvidia.in_heap()).0,
        answer(22784, 128)
    );
    assert!(!images_fit(
        heap_extent(&seven, None, nvidia.in_heap()).0,
        metal_seven
    ));
    // An image whose own alignment is coarser than Metal's, and one that asks more bytes (the
    // 5000-texel `Type1D` the replay found at 160768 bytes where Metal answers 32768), do not fit
    // Metal's answer: a shadow of the texture's bytes holds those to it.
    let coarse = heap_extent(&requirements(16384, 1024), None, nvidia.in_heap()).0;
    assert_eq!(coarse, answer(16384, 1024));
    assert!(!images_fit(coarse, metal));
    let wide = heap_extent(&requirements(160768, 128), None, nvidia.in_heap()).0;
    assert!(!images_fit(wide, answer(32768, 128)));
    // A depth twin still lies past the image inside the texture's bytes, at its own alignment,
    // with no page of the granularity's between them.
    let (got, offset) = heap_extent(&image, Some(&requirements(16384, 256)), nvidia.in_heap());
    assert_eq!((got, offset), (answer(32768, 256), Some(16384)));
    // The aux map's chunks survive: ANV's compressed 4x4 `RGBA8Unorm` image on an Xe-LP GPU asks
    // 4112 bytes and writes compression data to the end of the chunk it falls in.
    let short = RuntimeDriverDefect::ImageSizeShortOfCompressionData;
    let anv = ImagePlacement::new(64, &BTreeSet::from([short]));
    let compressed = requirements(4112, 65536);
    assert_eq!(
        heap_extent(&compressed, None, anv.in_heap()).0,
        answer(4352, 65536)
    );
}

/// A heap texture takes Metal's answer, which is the number a program sizes a heap and places its
/// resources by, in one of two ways: its image bound at the texture's offset where the images lie
/// inside that answer, or a shadow of the texture's bytes with the image in memory of its own
/// where they do not. Only where neither holds does the answer run past Metal's, and the device
/// announces `heap-texture-larger-than-metal` there.
///
/// The whole decision over the plan's own numbers, every shape the plan can hand it. An image
/// already in memory of its own takes none of the heap's bytes, so it is held whatever it needs; a
/// 4:2:2 texture is never held by its image, because its bytes lie where Metal's linear rows do
/// and not where its image lays them; a texture held with a depth twin is held by neither, since
/// no shadow of the texture's bytes moves the twin's; a multisample texture is held by a shadow
/// whether or not a shadow of rows fits it, because its image lies in memory of its own either
/// way.
#[test]
fn a_heap_texture_is_held_to_metal_s_answer_by_its_image_or_by_a_shadow() {
    use crate::texture::heap::{held_to_metal, HeapTextureShape, HeldToMetal};
    use crate::SizeAndAlign;
    let answer = |size, align| SizeAndAlign { size, align };
    // A plain single-sample texture whose image binds in the heap's memory, and a shadow of its
    // bytes fits its planes.
    let plain = HeapTextureShape {
        apart: false,
        four_two_two: false,
        twin: false,
        multisample: false,
        shadowable: true,
    };
    let metal = answer(16384, 128);
    // Inside Metal's answer, in both fields and exactly: the image.
    for images in [answer(16384, 128), answer(1024, 64), answer(128, 128)] {
        assert_eq!(held_to_metal(images, metal, plain), HeldToMetal::Images);
    }
    // Past it in either field: a shadow, which answers Metal's all the same. The NVIDIA shapes
    // the first replay found: a coarser alignment (1024 where Metal answers 128), and more bytes
    // (160768 where Metal answers 32768 for a 5000-texel `Type1D`).
    for images in [answer(16384, 1024), answer(22784, 128), answer(160768, 128)] {
        assert_eq!(held_to_metal(images, metal, plain), HeldToMetal::Shadow);
    }
    // No shadow fits its planes: neither, so the answer runs past Metal's, announced.
    let unshadowable = HeapTextureShape {
        shadowable: false,
        ..plain
    };
    let past = answer(16384, 1024);
    assert_eq!(
        held_to_metal(past, metal, unshadowable),
        HeldToMetal::Nothing
    );
    // A multisample texture is held by a shadow either way: its image lies in memory of its own.
    for shadowable in [false, true] {
        let shape = HeapTextureShape {
            multisample: true,
            shadowable,
            ..plain
        };
        assert_eq!(held_to_metal(past, metal, shape), HeldToMetal::Shadow);
        assert_eq!(held_to_metal(metal, metal, shape), HeldToMetal::Images);
    }
    // A depth twin's bytes lie past the image's inside the texture's and no shadow moves them: a
    // texture whose images do not fit is held by neither, whatever else its shape says.
    for (multisample, shadowable) in [(false, false), (false, true), (true, false), (true, true)] {
        let shape = HeapTextureShape {
            twin: true,
            multisample,
            shadowable,
            ..plain
        };
        assert_eq!(held_to_metal(past, metal, shape), HeldToMetal::Nothing);
        assert_eq!(held_to_metal(metal, metal, shape), HeldToMetal::Images);
    }
    // An image in memory of its own takes none of the heap's bytes: held whatever it needs, and a
    // depth twin never goes with one.
    let apart = HeapTextureShape {
        apart: true,
        ..plain
    };
    for images in [answer(160768, 65536), metal] {
        assert_eq!(held_to_metal(images, metal, apart), HeldToMetal::Images);
    }
    // A 4:2:2 texture is a shadow of the heap's bytes even where its image would fit: Metal lays
    // its rows out linearly there, which its image does not.
    let chroma = HeapTextureShape {
        four_two_two: true,
        ..plain
    };
    assert_eq!(held_to_metal(metal, metal, chroma), HeldToMetal::Shadow);
    assert_eq!(
        held_to_metal(
            metal,
            metal,
            HeapTextureShape {
                apart: true,
                ..chroma
            }
        ),
        HeldToMetal::Images
    );
}

/// A heap's memory starts at a multiple of every image alignment measured, so a heap texture of
/// such an image is placed, not refused: RADV's on an RX 7900 XTX (Mesa 26.0.8), 128 KiB for a
/// `D16`, `D32F` or `D32FS8` image of 8 x 8 texels or more and 256 KiB for a multisample image from
/// about 128 x 128 texels, and the Radeon 780M's 64 KiB under the same driver (the alignments alone
/// were measured; the sizes here stand in). An alignment past the base is refused typed.
#[test]
fn every_image_alignment_measured_fits_a_heap_s_base() {
    use crate::error::{Error, RuntimeGap};
    use crate::memory::BLOCK_BYTES;
    use crate::texture::heap::{check_heap_base, heap_extent, HEAP_BASE_ALIGNMENT};
    use crate::texture::ImagePlacement;
    use ash::vk::MemoryRequirements;
    use std::collections::BTreeSet;
    let pages = ImagePlacement::new(1, &BTreeSet::new());
    for alignment in [64 << 10, 128 << 10, 256 << 10] {
        let image = MemoryRequirements {
            size: 4 * alignment,
            alignment,
            memory_type_bits: 1,
        };
        let (answer, _) = heap_extent(&image, None, pages);
        assert_eq!(answer.align, alignment);
        assert_eq!(check_heap_base(answer.align), Ok(()), "{alignment:#x}");
    }
    let past = 2 * HEAP_BASE_ALIGNMENT;
    assert_eq!(
        check_heap_base(past),
        Err(Error::NotImplemented(RuntimeGap::HeapTextureAlignment {
            alignment: past
        }))
    );
    // A heap's buffer takes whole units of the base (`buffer.rs`'s `placed`), so rounding never
    // carries one past the size that takes a block of its own (half a pooled block).
    assert!((BLOCK_BYTES / 2).is_multiple_of(HEAP_BASE_ALIGNMENT));
}

/// A heap's depth or stencil texture held as a shadow of the heap's bytes keeps each aspect in
/// planes of its own, from the texture's first byte: the depth's levels in order, then the
/// stencil's, each level one region of all its layers' texels tightly packed (`Depth16Unorm` 2
/// bytes a texel, `Depth32Float` 4, a stencil 1), at a multiple of 4 bytes. `Depth32Float_Stencil8`
/// takes two regions, its stencil after the depth's 1024 bytes; a `Depth16Unorm` level of 9 texels
/// pushes the next to byte 20; a `Stencil8` array's region holds both layers.
#[test]
fn a_heap_shadow_keeps_each_aspect_in_planes_at_multiples_of_four_bytes() {
    use crate::texture::heap::heap_planes;
    use ash::vk;
    use metal2vulkan::metal::PixelFormat as P;
    use metal2vulkan::vk::Format as F;
    // (aspect, level, layers, offset, width, height) of each region, and the bytes spanned.
    let planes = |format, pf, extent, levels, layers| {
        let (regions, length) = heap_planes(format, pf, extent, levels, layers);
        let regions: Vec<_> = regions
            .iter()
            .map(|r| {
                let s = r.image_subresource;
                assert_eq!((r.buffer_row_length, r.buffer_image_height), (0, 0));
                assert_eq!(s.base_array_layer, 0);
                assert_eq!(r.buffer_offset % 4, 0);
                let e = r.image_extent;
                assert_eq!(e.depth, 1);
                let at = r.buffer_offset;
                (
                    s.aspect_mask.as_raw(),
                    s.mip_level,
                    s.layer_count,
                    at,
                    e.width,
                    e.height,
                )
            })
            .collect();
        (regions, length)
    };
    let (depth, stencil) = (
        vk::ImageAspectFlags::DEPTH.as_raw(),
        vk::ImageAspectFlags::STENCIL.as_raw(),
    );
    assert_eq!(
        planes(F::D32_SFLOAT, P::Depth32Float, [16, 16, 1], 1, 1),
        (vec![(depth, 0, 1, 0, 16, 16)], 1024)
    );
    assert_eq!(
        planes(F::D16_UNORM, P::Depth16Unorm, [3, 3, 1], 2, 1),
        (vec![(depth, 0, 1, 0, 3, 3), (depth, 1, 1, 20, 1, 1)], 22)
    );
    assert_eq!(
        planes(F::S8_UINT, P::Stencil8, [5, 5, 1], 1, 2),
        (vec![(stencil, 0, 2, 0, 5, 5)], 50)
    );
    assert_eq!(
        planes(
            F::D32_SFLOAT_S8_UINT,
            P::Depth32FloatStencil8,
            [16, 16, 1],
            1,
            1
        ),
        (
            vec![(depth, 0, 1, 0, 16, 16), (stencil, 0, 1, 1024, 16, 16)],
            1280
        )
    );
    // A cube's six faces in each level; the stencil's first level after the depth's last.
    assert_eq!(
        planes(
            F::D32_SFLOAT_S8_UINT,
            P::Depth32FloatStencil8,
            [2, 2, 1],
            2,
            6
        ),
        (
            vec![
                (depth, 0, 6, 0, 2, 2),
                (depth, 1, 6, 96, 1, 1),
                (stencil, 0, 6, 120, 2, 2),
                (stencil, 1, 6, 144, 1, 1),
            ],
            150
        )
    );
}

/// A heap shadow of a colour, a compressed and a 3D texture (a no-copy heap's texture that no
/// image over the heap's memory serves) lays each level's images out tightly, a level at a
/// multiple of 4 bytes and of its block's bytes, and `HeapPlane::row` finds a row of blocks
/// from a column: an `RGBA8Unorm` 5x3 level 0 holds 60 bytes, its level 1 starts at byte 60; a
/// `BC1_RGBA` 10x6 texture holds 3x2 blocks of 8 bytes; an `R8Unorm` 3x2x4 3D texture's level 1
/// (1x1x2) starts at byte 24, rounded to 4, and its regions copy each level's depth slices.
#[test]
fn a_heap_shadow_lays_colour_compressed_and_3d_levels_out_tightly() {
    use crate::texture::heap::{heap_plane_layout, heap_planes};
    use metal2vulkan::metal::PixelFormat as P;
    use metal2vulkan::vk::Format as F;
    let (planes, end) = heap_plane_layout(F::R8G8B8A8_UNORM, P::RGBA8Unorm, [5, 3, 1], 2, 1);
    let shape: Vec<_> = planes
        .iter()
        .map(|p| (p.level, p.offset, p.blocks, p.images, p.unit, p.block))
        .collect();
    assert_eq!(
        shape,
        vec![(0, 0, [5, 3], 1, 4, [1, 1]), (1, 60, [2, 1], 1, 4, [1, 1])]
    );
    assert_eq!(end, 68);
    assert_eq!(planes[0].row(0, 2, 1), 44);
    assert_eq!(planes[1].row(0, 0, 1), 64);

    let (planes, end) = heap_plane_layout(F::BC1_RGBA_UNORM_BLOCK, P::BC1RGBA, [10, 6, 1], 1, 2);
    assert_eq!(
        (
            planes[0].blocks,
            planes[0].images,
            planes[0].unit,
            planes[0].block
        ),
        ([3, 2], 2, 8, [4, 4])
    );
    assert_eq!(end, 96);
    assert_eq!(planes[0].row(1, 1, 2), 48 + 40);

    let (planes, end) = heap_plane_layout(F::R8_UNORM, P::R8Unorm, [3, 2, 4], 2, 1);
    let shape: Vec<_> = planes
        .iter()
        .map(|p| (p.level, p.offset, p.extent, p.images))
        .collect();
    assert_eq!(shape, vec![(0, 0, [3, 2, 4], 4), (1, 24, [1, 1, 2], 2)]);
    assert_eq!(end, 26);
    assert_eq!(planes[0].row(3, 1, 2), 23);
    let (regions, length) = heap_planes(F::R8_UNORM, P::R8Unorm, [3, 2, 4], 2, 1);
    let copies: Vec<_> = regions
        .iter()
        .map(|r| {
            (
                r.buffer_offset,
                r.image_extent.depth,
                r.image_subresource.layer_count,
            )
        })
        .collect();
    assert_eq!(copies, vec![(0, 4, 1), (24, 2, 1)]);
    assert_eq!(length, 26);
}

/// A heap's one-level 2D two-plane texture lays its planes out as an Apple M4 does: plane 0's rows
/// at a multiple of 16 bytes, plane 1 from a multiple of 16 KiB in rows at a multiple of 16,
/// spanning the size Metal's heap query answers for each size measured
/// (16576 for 16 x 12, 16613 for 37 x 5, 20480 for 64 x 64, 16385 for 1 x 1, 119580 for
/// 300 x 70, 118797 for 333 x 61); its CPU copies move plane 0's words at those rows.
#[test]
fn a_heap_two_plane_texture_lays_its_planes_out_as_metal_does() {
    use crate::texture::heap_two_planes::TwoPlaneHeapLayout;
    use metal2vulkan::metal::TextureType as T;
    for ([w, h], color, alpha, alpha_row, span) in [
        ([16, 12], 64, 16384, 16, 16576),
        ([37, 5], 160, 16384, 48, 16613),
        ([64, 64], 256, 16384, 64, 20480),
        ([1, 1], 16, 16384, 16, 16385),
        ([300, 70], 1200, 98304, 304, 119580),
        ([333, 61], 1344, 98304, 336, 118797),
    ] {
        let layout = TwoPlaneHeapLayout::of([w, h]);
        assert_eq!(
            (
                layout.color_pitch,
                layout.alpha_at,
                layout.alpha_pitch,
                layout.span()
            ),
            (color, alpha, alpha_row, span),
            "{w} x {h}"
        );
        // A mipmapped texture's first level, and a 3D one's first depth slice, lie alike; Metal's
        // blits address a one-level texture's alone at its blit rows.
        for ty in [T::Type2D, T::Type3D] {
            assert_eq!(TwoPlaneHeapLayout::of_texture(ty, [w, h], 1, 1), layout);
            let mipmapped = TwoPlaneHeapLayout::of_texture(ty, [w, h], 1, 3);
            assert_eq!(
                TwoPlaneHeapLayout {
                    levels: 1,
                    ..mipmapped
                },
                layout
            );
            assert!(
                layout.blit_rows() && !mipmapped.blit_rows(),
                "{ty:?} {w} x {h}"
            );
        }
        let plane = layout.color_plane();
        assert_eq!((plane.unit, plane.pitch, plane.images), (4, color, 1));
        assert_eq!(plane.row(0, h - 1, w - 1), (h - 1) * color + 4 * (w - 1));
    }
}

/// A heap's two-plane array keeps its first level's slices' color words in the heap's bytes as an
/// Apple M3 Max under macOS 27.0.1 lays them (CPU copies and GPU writes alike): each slice's rows
/// as a 2D texture's, the slices `align(h · align(4 w, 16), 128)` bytes apart (0x380 for 37 x 5,
/// 0x200 for 16 x 8, 0x100 for a 1D array of 37), and no alpha plane, which Metal's GPU neither
/// writes nor reads (its alpha reads 0). A 1D texture lays its row out as a 2D one of one row,
/// plane 1 at 16 KiB.
#[test]
fn a_heap_two_plane_array_lays_its_slices_out_as_metal_does() {
    use crate::texture::heap_two_planes::TwoPlaneHeapLayout;
    use metal2vulkan::metal::TextureType as T;
    for (ty, [w, h], slices, slice, span) in [
        (T::Type2DArray, [37, 5], 2, 0x380, 0x380 + 4 * 160 + 148),
        (T::Type2DArray, [16, 8], 3, 0x200, 2 * 0x200 + 7 * 64 + 64),
        (T::Type1DArray, [37, 1], 3, 0x100, 2 * 0x100 + 148),
    ] {
        let layout = TwoPlaneHeapLayout::of_texture(ty, [w, h], slices, 1);
        assert!(
            !layout.alpha_kept && !layout.blit_rows(),
            "{ty:?} {w} x {h}"
        );
        assert_eq!((layout.color_slice, layout.span()), (slice, span), "{ty:?}");
        let plane = layout.color_plane();
        assert_eq!((plane.images, plane.image_pitch), (slices, slice));
        assert_eq!(plane.row(slices - 1, h - 1, w - 1), span - 4);
    }
    let one = TwoPlaneHeapLayout::of_texture(T::Type1D, [37, 1], 1, 1);
    assert_eq!(
        (one.alpha_at, one.span(), one.blit_rows()),
        (0x4000, 0x4025, true)
    );
}

/// A heap numbers its textures in the order it makes them: an encoder that made the textures
/// before number `n` resident takes those from `n` on as it encodes its next command.
#[test]
fn a_heap_numbers_its_textures_and_finds_those_made_after_a_mark() {
    use std::sync::{Arc, Weak};
    let mut held: Numbered<u32> = Numbered::default();
    assert_eq!(held.made(), 0);
    let first = Arc::new(10);
    held.hold(Arc::downgrade(&first));
    let mark = held.made();
    assert_eq!(mark, 1);
    assert_eq!(held.since(mark).count(), 0, "nothing made after the mark");
    let second = Arc::new(20);
    let third = Arc::new(30);
    held.hold(Arc::downgrade(&second));
    held.hold(Arc::downgrade(&third));
    assert_eq!(held.made(), 3);
    let after: Vec<u32> = held
        .since(mark)
        .filter_map(Weak::upgrade)
        .map(|v| *v)
        .collect();
    assert_eq!(after, [20, 30], "those made after the mark, in order");
    drop(second);
    drop(third);
    // Holding forgets the gone ones, and numbers on.
    let fourth = Arc::new(40);
    held.hold(Arc::downgrade(&fourth));
    assert_eq!(held.made(), 4);
    let all: Vec<u32> = held
        .since(0)
        .filter_map(Weak::upgrade)
        .map(|v| *v)
        .collect();
    assert_eq!(all, [10, 40]);
    assert_eq!(held.since(4).count(), 0);
}

/// An image takes whole granularity pages, at a multiple of the larger of its alignment and the
/// granularity; on a device whose driver answers a compressed image's size short of its
/// compression data (ANV on Xe-LP GPUs), also whole 256-byte chunks, so no other resource lies in
/// the chunk the image's compression data ends in: ANV's compressed 4x4 `RGBA8Unorm` image (4112
/// bytes at 64 KiB) takes 4352 bytes there and keeps its size elsewhere.
#[test]
fn an_image_takes_whole_chunks_of_compression_data_where_the_driver_needs_them() {
    use crate::texture::ImagePlacement;
    use ash::vk::MemoryRequirements;
    use metal2vulkan::error::RuntimeDriverDefect;
    use std::collections::BTreeSet;
    let placed = |size, alignment, granularity, short: bool| {
        let defects = if short {
            BTreeSet::from([RuntimeDriverDefect::ImageSizeShortOfCompressionData])
        } else {
            BTreeSet::new()
        };
        let placed = ImagePlacement::new(granularity, &defects).place(MemoryRequirements {
            size,
            alignment,
            memory_type_bits: 1,
        });
        (placed.size, placed.alignment)
    };
    // Whole chunks, and whole pages of a coarser granularity.
    assert_eq!(placed(4112, 65536, 1, true), (4352, 65536));
    assert_eq!(placed(65536 + 272, 65536, 1, true), (65536 + 512, 65536));
    assert_eq!(placed(4352, 65536, 1, true), (4352, 65536));
    assert_eq!(placed(100, 4, 1, true), (256, 4));
    assert_eq!(placed(1000, 256, 4096, true), (4096, 4096));
    // Whole pages; a granularity that is no power of two is rounded up to one.
    assert_eq!(placed(4112, 65536, 1, false), (4112, 65536));
    assert_eq!(
        placed(65536 + 16, 65536, 1024, false),
        (65536 + 1024, 65536)
    );
    assert_eq!(placed(256, 4, 1, false), (256, 4));
    assert_eq!(placed(1000, 256, 4096, false), (4096, 4096));
    assert_eq!(placed(16512, 128, 1000, false), (17408, 1024));
}
