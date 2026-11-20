//! What the runtime's `heapTextureSizeAndAlignWithDescriptor:` answer is held to.
//!
//! The runtime answers the device's own size and alignment, not Metal's (`docs/API.md` section 10,
//! "The runtime's heap texture size and alignment are the device's own"), so the cases that
//! ask it do not compare the two numbers with Metal's on its run
//! ([`crate::texture_query::DEVICE_OWN_SIZES`]). This side holds each answer to what a consumer
//! places by before it answers it, and answers the failure where it does not hold:
//!
//! 1. the alignment is a power of two, for every query;
//! 2. for a Private query, a heap of `align + size` bytes takes a texture of the descriptor at
//!    offset `align`;
//! 3. there, where the runtime moves the format's bytes, level 0's first slice written from a
//!    buffer reads back unchanged.
//!
//! A Shared, Managed or Memoryless query is held to the first alone: the runtime places a heap
//! texture in a Private block only (`Gap::HeapTexture`), and placing the descriptor in a Private
//! heap would answer another query. A query whose bytes the runtime does not move, which its copy
//! answers `Gap::TextureBytes` (depth and stencil both, or multisample) or `Gap::FloatDepthUpload`
//! (`Depth32Float`) for, is held to the first two.

use super::{waited_for, work, RtDevice, WAIT};
use crate::case::{Answer, Failure};
use crate::texture_query::{HeapTextureQuery, Storage};
use metal2vulkan_rt as rt;

/// Hold `sized`, the runtime's answer for `descriptor` (the descriptor of `query`), to the
/// module's property, on `queue`.
pub(super) fn hold(
    on: &RtDevice,
    queue: &rt::CommandQueue,
    query: &HeapTextureQuery,
    descriptor: &rt::TextureDescriptor,
    sized: rt::SizeAndAlign,
) -> Answer<()> {
    let rt::SizeAndAlign { size, align } = sized;
    if !align.is_power_of_two() {
        return Err(Failure::named(
            "AlignNotAPowerOfTwo",
            format!("step 1: the alignment {align} is not a power of two"),
        ));
    }
    if query.storage != Storage::Private {
        return Ok(());
    }
    let heap_size = align.checked_add(size).ok_or_else(|| {
        Failure::named(
            "HeapPastU64",
            format!("step 2: a heap of {align} + {size} bytes"),
        )
    })?;
    let (_block, heap) = work::heap_of(on, heap_size, Storage::Private)?;
    let index = on.texture_index(descriptor.pixel_format, descriptor.usage);
    let texture = (on.device.new_heap_texture(descriptor, index, heap, align))
        .map_err(|error| on.failure(&error))?;
    round_trip(on, queue, descriptor, &texture)
}

/// Level 0's first slice of `texture`, written from a buffer and read back into another, at
/// Metal's layout of the level (`Device::texture_level_info`); `Ok` where the runtime moves none
/// of the format's bytes.
fn round_trip(
    on: &RtDevice,
    queue: &rt::CommandQueue,
    descriptor: &rt::TextureDescriptor,
    texture: &rt::Texture,
) -> Answer<()> {
    use rt::TextureType as T;
    let height = match descriptor.texture_type {
        T::Type1D | T::Type1DArray => 1,
        _ => descriptor.height,
    };
    let depth = match descriptor.texture_type {
        T::Type3D => descriptor.depth,
        _ => 1,
    };
    let level = on.device.texture_level_info(&rt::TextureLevelQuery {
        pixel_format: u64::from(descriptor.pixel_format.as_raw()),
        width: u64::from(descriptor.width),
        height: u64::from(height),
        depth: u64::from(depth),
        array_length: 1,
        level: 0,
        planes: 0,
    });
    let written: Vec<u8> = (0..level.bytes).map(|at| (at % 251) as u8 ^ 0x5a).collect();
    let (source, _source_blocks) = work::buffer_of(on, level.bytes, Storage::Shared)?;
    work::write(&source, &written)?;
    let (read, _read_blocks) = work::buffer_of(on, level.bytes, Storage::Shared)?;
    let layout = rt::BufferLayout {
        offset: 0,
        bytes_per_row: level.bytes_per_row,
        bytes_per_image: level.bytes_per_image,
    };
    let place = rt::TexturePlace {
        slice: 0,
        level: 0,
        origin: [0, 0, 0],
    };
    let extent = [descriptor.width, height, depth];
    let failed = |error: rt::Error| on.failure(&error);
    let command_buffer = queue.command_buffer().map_err(failed)?;
    let mut encoder = command_buffer.blit_command_encoder().map_err(failed)?;
    match encoder.copy_from_buffer(&source, layout, extent, texture, place) {
        Ok(()) => {}
        Err(rt::Error::Unsupported(
            rt::Gap::TextureBytes { .. } | rt::Gap::FloatDepthUpload { .. },
        )) => {
            encoder.end_encoding();
            return Ok(());
        }
        Err(error) => return Err(failed(error)),
    }
    (encoder.copy_from_texture_to_buffer(texture, place, extent, &read, layout)).map_err(failed)?;
    encoder.end_encoding();
    command_buffer.commit().map_err(failed)?;
    match waited_for(&command_buffer, WAIT) {
        Some(word) if word == "Completed" => {}
        Some(word) => {
            return Err(Failure::named(
                "RoundTripNotCompleted",
                format!("step 3: the round trip's command buffer ended {word}"),
            ))
        }
        None => {
            return Err(Failure::named(
                "WaitRanOut",
                "step 3: the round trip's command buffer did not complete inside the bound",
            ))
        }
    }
    let bytes = work::read_buffer(&read, 0, written.len())?;
    let differ = (bytes.iter().zip(&written)).filter(|(a, b)| a != b).count();
    if differ == 0 {
        return Ok(());
    }
    let first = (bytes.iter().zip(&written))
        .position(|(a, b)| a != b)
        .unwrap_or_default();
    Err(Failure::named(
        "RoundTripDiffers",
        format!(
            "step 3: {differ} of {} bytes read back otherwise than written, the first at {first}",
            written.len()
        ),
    ))
}
