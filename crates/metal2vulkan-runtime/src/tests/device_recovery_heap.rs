//! What a device's recovery carries of its heaps (`Heap::carry`, `device/carry.rs`): a `Private`
//! heap goes on as the same heap, its bytes new memory that reads zero, with the buffers and the
//! textures placed in it before (a texture's contents lost, as a new texture's are); buffers and
//! textures made in it after the recovery serve.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use crate::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Device, Error, HeapDescriptor, HeapType,
    Origin, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The bytes of each heap the test makes.
const HEAP: u64 = 64 << 10;

/// The side of the square `RGBA8Unorm` textures the test makes: [`BYTES`] bytes of texels.
const SIDE: u64 = 32;

/// A `Private` `RGBA8Unorm` texture of [`SIDE`] by [`SIDE`] texels.
fn descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// `source`'s first [`BYTES`] bytes copied into `texture` and back out into a new buffer, as
/// `queue`'s GPU work reads them, completed. Refused as the copy's encoding refuses `texture`.
fn through(
    device: &Device,
    queue: &CommandQueue,
    source: &Buffer,
    texture: &Texture,
) -> Result<Vec<u8>, Error> {
    let target = device.new_buffer(BYTES, ResourceOptions::private())?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    let (size, origin) = (Size::new(SIDE, SIDE, 1), Origin::new(0, 0, 0));
    let row = 4 * SIDE;
    blit.copy_from_buffer_to_texture(
        source,
        0,
        row,
        0,
        size,
        texture,
        0,
        0,
        origin,
        BlitOption::None,
    )?;
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        origin,
        size,
        &target,
        0,
        row,
        0,
        BlitOption::None,
    )?;
    blit.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    read(device, queue, &target)
}

/// A recovery carries a `Private` placement heap and a `Private` automatic heap, with the buffer
/// placed in each before it and the texture placed in the automatic one: each heap, buffer and
/// texture the same object on the next generation (its `ObjectId`, Metal address and label), its
/// bytes zero and the texture's contents lost, as its notice counts. The texture made before it
/// takes copies there, and a use of its heap makes it resident. Buffers and textures made in
/// them after it serve, at the automatic heap's free ranges as they were. A carried buffer's
/// release gives its range back once, and the lost `VkDevice` goes at its rest, the texture's
/// parts there let go of while it lives.
#[test]
fn a_recovery_carries_private_heaps_with_their_buffers_and_textures() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let private = ResourceOptions::private();
    let placement =
        (device.new_heap(&HeapDescriptor::new(HeapType::Placement, HEAP, private))).unwrap();
    placement.set_label("placement");
    let automatic =
        (device.new_heap(&HeapDescriptor::new(HeapType::Automatic, HEAP, private))).unwrap();
    let placed = placement
        .new_buffer_at_offset(BYTES, private, BYTES)
        .unwrap();
    let taken = automatic.new_buffer(BYTES, private).unwrap();
    for (buffer, byte) in [(&placed, 7), (&taken, 9)] {
        fill(&queue, buffer, byte);
        assert_eq!(
            read(&device, &queue, buffer).unwrap(),
            vec![byte; BYTES as usize]
        );
    }
    let before = automatic.new_texture(&descriptor()).unwrap();
    let identity = [
        (placement.object_id(), placement.gpu_address()),
        (automatic.object_id(), automatic.gpu_address()),
        (placed.object_id(), placed.gpu_address()),
        (taken.object_id(), taken.gpu_address()),
    ];
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(&format!(
            ", and 2 `Private` heaps ({} bytes) with the 2 buffers placed in them, whose \
             contents were lost too, and 1 textures whose contents were lost too (they read as \
             new ones: 0 `Shared`, 0 `Managed`, 1 `Private` or memoryless), 0 textures over the \
             buffers carried, 0 textures placed in the no-copy heaps carried and 0 texture \
             views; left refused",
            2 * HEAP
        )) && !said[0].contains("heaps' textures"),
        "{said:?}"
    );
    // The heaps and the buffers placed in them before: the same objects on the next generation,
    // their bytes zero.
    for device in [
        placement.generation(),
        automatic.generation(),
        placed.generation(),
        taken.generation(),
    ] {
        assert!(device.generation_serves());
    }
    assert_eq!(
        [
            (placement.object_id(), placement.gpu_address()),
            (automatic.object_id(), automatic.gpu_address()),
            (placed.object_id(), placed.gpu_address()),
            (taken.object_id(), taken.gpu_address()),
        ],
        identity
    );
    assert_eq!(placement.label().as_deref(), Some("placement"));
    for buffer in [&placed, &taken] {
        assert_eq!(
            read(&device, &queue, buffer).unwrap(),
            vec![0; BYTES as usize]
        );
    }
    fill(&queue, &placed, 11);
    assert_eq!(
        read(&device, &queue, &placed).unwrap(),
        vec![11; BYTES as usize]
    );
    // Buffers and textures made in them after it serve: a placement heap's at its offset, an
    // automatic heap's past what it placed before.
    let later = placement
        .new_buffer_at_offset(BYTES, private, 2 * BYTES)
        .unwrap();
    let next = automatic.new_buffer(BYTES, private).unwrap();
    assert_ne!(next.gpu_address(), taken.gpu_address());
    for (buffer, byte) in [(&later, 13), (&next, 15)] {
        fill(&queue, buffer, byte);
        assert_eq!(
            read(&device, &queue, buffer).unwrap(),
            vec![byte; BYTES as usize]
        );
    }
    assert_eq!(
        read(&device, &queue, &taken).unwrap(),
        vec![0; BYTES as usize]
    );
    let texture = automatic.new_texture(&descriptor()).unwrap();
    assert_eq!(
        through(&device, &queue, &placed, &texture).unwrap(),
        vec![11; BYTES as usize]
    );
    // The texture made before is the next generation's, in the heap there: it takes copies.
    assert!(before.device().generation_serves());
    assert_eq!(
        through(&device, &queue, &later, &before).unwrap(),
        vec![13; BYTES as usize]
    );
    // A use of the heap on the next generation makes its textures resident, the one made before
    // among them.
    let command_buffer = queue.command_buffer().unwrap();
    let mut compute = command_buffer.compute_command_encoder().unwrap();
    compute.use_heap(&automatic).unwrap();
    compute.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    // A carried buffer's release gives its range back once.
    let used = automatic.used_size();
    drop(taken);
    assert_eq!(automatic.used_size(), used - BYTES);
    // The lost `VkDevice` went at its rest, the carried texture's parts there let go of.
    assert_eq!(live_vk_devices(&device, 1), 1);
    drop(before);
}
