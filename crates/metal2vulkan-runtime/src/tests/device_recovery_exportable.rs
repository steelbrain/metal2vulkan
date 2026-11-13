//! What a device's recovery carries of its exportable buffers (`exported_memory.rs`): a `Private`
//! buffer in memory another API imports is the same buffer on the new `VkDevice`, over its own
//! memory imported again, as the recovery's notice counts. Its exports after the recovery answer
//! the same memory (the same file, allocation size, offset and memory type), a texture over it
//! is carried over it, and the GPU's writes to the buffer there read back through the texture.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! Each handle type the device exports runs; one it does not is named and skipped.

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBufferStatus, CommandQueue, Error,
    MemoryExport, Origin, ResourceOptions, RuntimeGap, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::os::unix::fs::MetadataExt;

/// The texture's texels a row.
const WIDTH: u64 = 64;

/// The texture's rows.
const HEIGHT: u64 = 16;

/// The bytes of a row of `RGBA8Unorm` texels, which is a multiple of 256: the texture over the
/// buffer may be an image in its memory.
const ROW: u64 = 4 * WIDTH;

/// The buffer's bytes, the texture's whole.
const LENGTH: u64 = ROW * HEIGHT;

/// What an export of `buffer`'s memory as `export` answers: the allocation's size, the buffer's
/// offset in it, its memory type, and the descriptor's file's inode, which every descriptor of
/// one memory shares.
fn exported(buffer: &Buffer, export: MemoryExport) -> (u64, u64, u32, u64) {
    let memory = match export {
        MemoryExport::OpaqueFd => buffer.export_memory_fd(),
        _ => buffer.export_memory_dma_buf(),
    }
    .unwrap();
    let file = std::fs::File::from(memory.fd);
    let inode = file.metadata().unwrap().ino();
    (
        memory.allocation_size,
        memory.offset,
        memory.memory_type_index,
        inode,
    )
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

/// `texture`'s texels as `queue`'s blit copies them into a `Shared` buffer, completed.
fn read(queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
    let device = queue.device();
    let read = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
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
    )
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
/// after the recovery answers the same file, allocation size, offset and memory type as one
/// before; and a fill of the buffer on the new `VkDevice` reads back through the texture.
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
            read(&queue, &texture),
            [0x11; LENGTH as usize],
            "{export:?}"
        );
        let before = exported(&buffer, export);
        let from = heard.lock().unwrap().len();
        device.recover_for_test().unwrap();
        let said = recovery_notices(&heard, from, "the device");
        assert_eq!(said.len(), 1, "{export:?}: {said:?}");
        // The buffers carried with their contents are not counted: a read buffer let go of may
        // be among them. The export after the recovery shows this one's.
        assert!(
            said[0].contains(" 1 textures over the buffers carried ")
                && !said[0].contains("exportable memory")
                && !said[0].contains("opaque-fd"),
            "{export:?}: {said:?}"
        );
        let after = exported(&buffer, export);
        eprintln!(
            "EXPORT-CARRY on {name}: {export:?}: before (allocation, offset, type, inode) \
             {before:?}, after {after:?}"
        );
        assert_eq!(after, before, "{export:?}: the export's memory");
        fill(&queue, &buffer, 0x5a);
        assert_eq!(
            read(&queue, &texture),
            [0x5a; LENGTH as usize],
            "{export:?}"
        );
        carried += 1;
    }
    assert!(carried > 0, "{name} exports no buffer's memory");
}
