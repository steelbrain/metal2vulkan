//! What a device's recovery carries of its exportable buffers (`exported_memory.rs`): a `Private`
//! buffer in memory another API imports is the same buffer on the new `VkDevice`, over its own
//! memory imported again, as the recovery's notice counts. Its exports after the recovery answer
//! the allocation size, offset and memory type of one before; another API's import of the export
//! taken before the recovery still writes the buffer's memory, which the buffer and a texture over
//! it read once `Buffer::external_write_done` announces it; and the GPU's writes to the buffer on
//! the new `VkDevice` read back through the texture. The descriptor's file is not compared: the
//! recovery exports the memory again from the lost `VkDevice`, Vulkan does not say that two
//! exports of one memory answer one file, and NVIDIA's driver answers a new dma-buf to each.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! Each handle type the device exports runs; one it does not is named and skipped.

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::exported_memory::Writer;
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBufferStatus, CommandQueue, Error,
    ExportedMemory, MemoryExport, Origin, ResourceOptions, RuntimeGap, Size, Texture,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's texels a row.
const WIDTH: u64 = 64;

/// The texture's rows.
const HEIGHT: u64 = 16;

/// The bytes of a row of `RGBA8Unorm` texels, which is a multiple of 256: the texture over the
/// buffer may be an image in its memory.
const ROW: u64 = 4 * WIDTH;

/// The buffer's bytes, the texture's whole.
const LENGTH: u64 = ROW * HEIGHT;

/// An export of `buffer`'s memory as `export`, and what it answers: the allocation's size, the
/// buffer's offset in it and its memory type.
fn exported(buffer: &Buffer, export: MemoryExport) -> (ExportedMemory, (u64, u64, u32)) {
    let memory = match export {
        MemoryExport::OpaqueFd => buffer.export_memory_fd(),
        _ => buffer.export_memory_dma_buf(),
    }
    .unwrap();
    let answers = (
        memory.allocation_size,
        memory.offset,
        memory.memory_type_index,
    );
    (memory, answers)
}

/// `queue`'s blit of `value` into every byte of `buffer`, completed.
fn fill(queue: &CommandQueue, buffer: &Buffer, value: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..LENGTH, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
}

/// What a read copies from.
enum Source<'a> {
    /// A texture's texels.
    Texels(&'a Texture),
    /// A buffer's bytes.
    Bytes(&'a Buffer),
}

/// What `queue`'s blit copies into a `Shared` buffer of [`LENGTH`] bytes from `from`,
/// completed.
fn read(queue: &CommandQueue, from: Source) -> Vec<u8> {
    let device = queue.device();
    let read = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    match from {
        Source::Texels(texture) => blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(WIDTH, HEIGHT, 1),
            &read,
            0,
            ROW,
            LENGTH,
            BlitOption::None,
        ),
        Source::Bytes(buffer) => blit.copy_from_buffer_to_buffer(buffer, 0, &read, 0, LENGTH),
    }
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    // SAFETY: the read buffer's mapped bytes, which the completed copy wrote and nothing else uses.
    unsafe { std::slice::from_raw_parts(read.contents().unwrap().as_ptr(), LENGTH as usize) }
        .to_vec()
}

/// A `Private` buffer exportable as each handle type the device exports, with a texture over it,
/// made before a recovery, is carried over its own memory: the notice leaves no buffer in
/// exportable memory refused and counts the texture among those over carried buffers; an export
/// after the recovery answers the allocation size, offset and memory type of one before; a fill of
/// the buffer on the new `VkDevice` reads back through the texture; and a second `VkDevice`'s
/// write through its import of the export taken before the recovery, announced, reads back
/// through the buffer and the texture.
#[test]
fn a_recovery_carries_a_private_exportable_buffer_over_its_own_memory() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let name = device.name().to_owned();
    let queue = device.new_command_queue().unwrap();
    let mut carried = 0;
    for export in [MemoryExport::OpaqueFd, MemoryExport::DmaBuf] {
        let descriptor =
            BufferDescriptor::new(LENGTH, ResourceOptions::private(), BufferContents::Zero)
                .with_exported_as(export);
        let buffer = match device.new_buffer_with_descriptor(descriptor) {
            Ok(buffer) => buffer,
            Err(Error::NotImplemented(
                gap @ (RuntimeGap::ExportableMemory | RuntimeGap::DmaBufExport),
            )) => {
                eprintln!("EXPORT-CARRY on {name}: {export:?} not exported: {gap:?}");
                continue;
            }
            Err(error) => panic!("{name} {export:?}: {error:?}"),
        };
        let mut texture =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
        texture.usage = TextureUsage::SHADER_READ;
        texture.resource_options = ResourceOptions::private();
        let texture = buffer.new_texture(&texture, 0, ROW).unwrap();
        fill(&queue, &buffer, 0x11);
        assert_eq!(
            read(&queue, Source::Texels(&texture)),
            [0x11; LENGTH as usize],
            "{export:?}"
        );
        let usage = crate::buffer::usage(device.shared().device_address);
        let (memory, before) = exported(&buffer, export);
        let writer = Writer::import(&device, memory, usage, export);
        let from = heard.lock().unwrap().len();
        device.recover_for_test().unwrap();
        let said = recovery_notices(&heard, from, "the device");
        assert_eq!(said.len(), 1, "{export:?}: {said:?}");
        // The buffers carried with their contents are not counted: a read buffer let go of may
        // be among them. The export after the recovery shows this one's.
        assert!(
            said[0].contains(" 1 textures over the buffers carried,")
                && !said[0].contains("exportable memory")
                && !said[0].contains("opaque-fd"),
            "{export:?}: {said:?}"
        );
        let (_, after) = exported(&buffer, export);
        eprintln!(
            "EXPORT-CARRY on {name}: {export:?}: before (allocation, offset, type) {before:?}, \
             after {after:?}"
        );
        assert_eq!(after, before, "{export:?}: the export's memory");
        fill(&queue, &buffer, 0x5a);
        assert_eq!(
            read(&queue, Source::Texels(&texture)),
            [0x5a; LENGTH as usize],
            "{export:?}"
        );
        // The other API's import of the export taken before the recovery reaches the carried
        // buffer's memory: its write, released to the family the runtime acquires from and
        // announced, is what the buffer and the texture over it hold.
        writer.fill(0xc3);
        buffer.external_write_done().unwrap();
        assert_eq!(
            read(&queue, Source::Bytes(&buffer)),
            [0xc3; LENGTH as usize],
            "{export:?}: the buffer after the import's write"
        );
        assert_eq!(
            read(&queue, Source::Texels(&texture)),
            [0xc3; LENGTH as usize],
            "{export:?}: the texture after the import's write"
        );
        eprintln!(
            "EXPORT-CARRY on {name}: {export:?}: the import before the recovery wrote the \
             carried buffer"
        );
        carried += 1;
    }
    assert!(carried > 0, "{name} exports no buffer's memory");
}
