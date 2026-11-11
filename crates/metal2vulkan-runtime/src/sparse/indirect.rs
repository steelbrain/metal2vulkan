//! Indirect mapping updates: `updateTextureMapping:mode:indirectBuffer: indirectBufferOffset:`,
//! whose regions are GPU memory the work before the update may write.
//!
//! The update's batch waits for a host step (`CommandBufferShared::host_step`), which runs once
//! everything submitted before it on any queue completed: the host reads the arguments then, as
//! Metal's update reads them when it runs, and the step replaces the update with the mapping
//! they state, which the batch's binding applies in order. A `Shared` or `Managed` buffer is read
//! through its mapping (host-coherent memory, which every batch's end made the device's writes
//! visible to); a `Private` one through a copy into a `Shared` buffer that the device's internal
//! command buffer makes and waits for. The device announces the round trip once
//! (`Degradation::HostReadMappingArguments`).

use super::encoder::SparseUpdate;
use super::mapping::UnitUpdate;
use crate::buffer::Buffer;
use crate::degradation::Degradation;
use crate::error::{internal, Error};
use crate::texture::Texture;
use crate::values::{Region, ResourceOptions};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::TileMapping;

/// The words of one `MTLMapIndirectArguments`: the region's origin and size in tiles, the level
/// and the slice.
const ARGUMENT_WORDS: u64 = 8;

/// The updates an indirect update of `texture` states now: its arguments at `offset` of
/// `buffer`, a count and that many regions, mapped or unmapped by `mode` as one update of them
/// all, through a view naming its texture's own tiles. Arguments Metal's validation stops map as
/// plain Metal maps them (`UnitGrid::clipped_region`), and the regions the count names past the
/// buffer map nothing (the M3 Max).
pub(crate) fn resolve(
    device: &Device,
    texture: &Texture,
    mode: TileMapping,
    buffer: &Buffer,
    offset: u64,
) -> Result<Vec<SparseUpdate>, Error> {
    let sparse = texture
        .sparse_image()
        .ok_or_else(|| internal("an indirect update of a texture that is not sparse"))?;
    let length = buffer.length();
    let count = read_words(device, buffer, offset, 1)?
        .first()
        .copied()
        .map_or(0, u64::from);
    // The regions inside the buffer (the encoder refused a count word past it).
    let room = length.saturating_sub(offset + 4) / (4 * ARGUMENT_WORDS);
    let count = count.min(room);
    let words = read_words(device, buffer, offset + 4, count * ARGUMENT_WORDS)?;
    // Through a view, the slices are the view's and the levels its texture's, as a direct
    // update's (the encoder refused a view whose tiles are not its texture's).
    let first = u64::from(texture.record().base_layer);
    let mut units = Vec::new();
    for arguments in words.as_chunks::<{ ARGUMENT_WORDS as usize }>().0.iter() {
        let word = |i: usize| u64::from(arguments[i]);
        let region = Region::new_3d(word(0), word(1), word(2), word(3), word(4), word(5));
        units.extend(
            sparse
                .grid
                .clipped_region(word(6), word(7) + first, &region),
        );
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![SparseUpdate::Units {
        texture: texture.clone(),
        update: UnitUpdate { mode, units },
    }])
}

/// Announce on `device`, once, that the host reads mapping arguments.
pub(crate) fn degrade(device: &Device) {
    device.degrade(Degradation::HostReadMappingArguments, || {
        (
            "`update_texture_mapping_indirect` of a sparse texture".to_owned(),
            "its regions are GPU memory the work before it may write, which the host reads once \
             that work completed"
                .to_owned(),
        )
    });
}

/// `count` little-endian 32-bit words of `buffer` from `offset`, which lie in it, as they stand
/// now: everything that wrote them completed.
fn read_words(
    device: &Device,
    buffer: &Buffer,
    offset: u64,
    count: u64,
) -> Result<Vec<u32>, Error> {
    let bytes = count * 4;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let mut read = vec![0u8; bytes as usize];
    if let Some(contents) = buffer.device_contents() {
        // SAFETY: the buffer's bytes `offset..offset + bytes` lie in it, host-coherent memory
        // (`Shared` or `Managed` without a host copy) whose device writes completed, made visible
        // to the host by their batch's end; nothing writes them now (the work after waits for
        // this step).
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr().wrapping_add(offset as usize),
                read.as_mut_ptr(),
                read.len(),
            );
        }
    } else {
        let copy = device.own_buffer(bytes, ResourceOptions::shared())?;
        let (from, at) = buffer.at(offset);
        let (into, start) = copy.at(0);
        let region = vk::BufferCopy {
            src_offset: at,
            dst_offset: start,
            size: bytes,
        };
        device.submit_internal(|raw, command_buffer| {
            // SAFETY: the internal command buffer is recording; both buffers are live until the
            // call returns, which waits for the submission, and the region lies in both.
            unsafe {
                raw.cmd_copy_buffer(command_buffer, from, into, &[region]);
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                raw.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        })?;
        let contents = copy
            .mapped_contents()
            .ok_or_else(|| internal("a Shared copy without host bytes"))?;
        // SAFETY: the copy holds `bytes` host-coherent bytes, which the waited-for copy wrote and
        // its barrier made visible to the host.
        unsafe {
            std::ptr::copy_nonoverlapping(contents.as_ptr(), read.as_mut_ptr(), read.len());
        }
    }
    Ok(read
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}
