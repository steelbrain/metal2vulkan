//! What a device's recovery carries of a heap over the caller's host memory
//! (`Device::new_heap_with_bytes_no_copy`, `device/carry.rs`): the heap, made again over the same
//! pages on the new `VkDevice`, with the buffers and textures placed in it, which read the pages'
//! bytes there as they did before, and the heap places more there.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_no_copy::{fill, import_page, mapping, read, unmap, BYTES};
use crate::{
    BlitOption, CommandBufferStatus, CommandQueue, HeapDescriptor, HeapType, HostMemory, Origin,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ptr::NonNull;

/// The heap's bytes: the caller's pages it lies over.
const HEAP: u64 = 1 << 20;

/// Where the placements the test makes after the recovery start in the heap.
const LATER: u64 = HEAP / 2;

/// The texels a row of each texture the test places.
const WIDTH: u64 = 64;

/// The rows of each texture the test places.
const HEIGHT: u64 = 16;

/// The bytes of a row of `RGBA8Unorm` texels.
const ROW: u64 = 4 * WIDTH;

/// Set `length` bytes `offset` into the pages at `at` to `byte` from the host, while no command
/// buffer runs.
fn host_set(at: NonNull<u8>, offset: u64, length: u64, byte: u8) {
    // SAFETY: the test's mapping, which no command buffer uses now.
    unsafe { std::ptr::write_bytes(at.as_ptr().add(offset as usize), byte, length as usize) };
}

/// The first byte of the pages at `at`, read by the host, while no command buffer runs.
fn host_first(at: NonNull<u8>) -> u8 {
    // SAFETY: as above.
    unsafe { *at.as_ptr() }
}

/// A `Shared` `RGBA8Unorm` texture of [`WIDTH`] by [`HEIGHT`] texels, which a shader reads.
fn texture_descriptor() -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    descriptor
}

/// `texture`'s texels as `queue`'s blit copies them into a new `Shared` buffer, completed.
#[track_caller]
fn texels(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let length = ROW * HEIGHT;
    let target = (queue.device())
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(WIDTH, HEIGHT, 1),
        &target,
        0,
        ROW,
        length,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let contents = target.contents().unwrap();
    // SAFETY: the target's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// A placement heap over the caller's pages, with a buffer and a texture placed in it and a
/// texture over the buffer, made before a recovery, is carried with them, as the notice counts:
/// the heap is the new generation's, the buffer and the textures read the pages' bytes there, the
/// GPU's write of the buffer reaches the pages and the texture over it, and a buffer and a
/// texture placed in the heap after the recovery read the pages' bytes too.
#[test]
fn a_recovery_carries_a_no_copy_heap_with_its_buffer_and_texture() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    if import_page(&device).is_none() {
        eprintln!("NO-COPY-HEAP-CARRY: the device imports no host memory here; nothing to test");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let texture = texture_descriptor();
    let taken = device.heap_texture_size_and_align(&texture).unwrap();
    let placed_at = BYTES.next_multiple_of(taken.align);
    let later_at = (LATER + BYTES).next_multiple_of(taken.align);
    assert!(placed_at + taken.size <= LATER && later_at + taken.size <= HEAP);
    let at = mapping(HEAP);
    host_set(at, 0, BYTES, 5);
    host_set(at, placed_at, taken.size, 0x33);
    host_set(at, LATER, BYTES, 0x44);
    host_set(at, later_at, taken.size, 0x55);
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::shared());
    // SAFETY: the test's mapping, unmapped after the heap and everything placed in it went.
    let heap =
        unsafe { device.new_heap_with_bytes_no_copy(HostMemory::new(at), &descriptor, None, None) }
            .unwrap();
    let buffer = (heap.new_buffer_at_offset(BYTES, ResourceOptions::shared(), 0)).unwrap();
    let placed = heap.new_texture_at_offset(&texture, placed_at).unwrap();
    let over = buffer.new_texture(&texture, 0, ROW).unwrap();
    assert_eq!(read(&device, &queue, &buffer), vec![5; BYTES as usize]);
    let texel_bytes = (ROW * HEIGHT) as usize;
    assert_eq!(texels(&queue, &placed), vec![0x33; texel_bytes]);
    assert_eq!(texels(&queue, &over), vec![5; texel_bytes]);
    let address = buffer.gpu_address();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(", and 1 no-copy heaps with the 1 buffers placed in them")
            && said[0].contains(
                " 1 textures over the buffers carried, 1 textures placed in the \
                 no-copy heaps carried "
            )
            && !said[0].contains("heaps' textures")
            && !said[0].contains("heaps' buffers"),
        "{said:?}"
    );
    for serves in [
        heap.device().generation_serves(),
        buffer.device().generation_serves(),
        placed.device().generation_serves(),
        over.device().generation_serves(),
    ] {
        assert!(serves);
    }
    assert_eq!(buffer.gpu_address(), address);
    // The pages' bytes, read by the GPU there; its writes reach the pages.
    assert_eq!(read(&device, &queue, &buffer), vec![5; BYTES as usize]);
    assert_eq!(texels(&queue, &placed), vec![0x33; texel_bytes]);
    fill(&queue, &buffer, 9);
    assert_eq!(host_first(at), 9);
    assert_eq!(texels(&queue, &over), vec![9; texel_bytes]);
    // The heap places more there, over the same pages.
    let later = (heap.new_buffer_at_offset(BYTES, ResourceOptions::shared(), LATER)).unwrap();
    assert_eq!(read(&device, &queue, &later), vec![0x44; BYTES as usize]);
    let later_texture = heap.new_texture_at_offset(&texture, later_at).unwrap();
    assert_eq!(texels(&queue, &later_texture), vec![0x55; texel_bytes]);
    drop((later_texture, later, over, placed, buffer, heap));
    unmap(at, HEAP);
}
