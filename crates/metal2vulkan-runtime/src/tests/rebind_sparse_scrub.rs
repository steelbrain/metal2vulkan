//! A blit encoder that writes a sparse texture and fills a no-copy buffer, its command buffer
//! committed behind a shared-event wait only the host signals, and the buffer rebound before the
//! signal. The blit's write leaves a scrub of the unmapped tiles it wrote for its batch's
//! submission (`sparse::emulated::Scrub`, `Deferred::SparseScrub`), which reads no buffer: the
//! commit holds none of the buffer's pages until completion (`Deferred::resolved_at_its_calls`),
//! so the rebind encodes the encoder again (`command_buffer/reencode.rs`), the fill writing the
//! new pages, and the encoding again's scrub takes the first's place
//! (`Deferred::left_again_by_an_encoding_again`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. It prints `REBIND-SCRUB` lines, and returns where the device makes no
//! sparse texture or imports no host memory.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::words_of;
use super::rebind_behind_a_wait::{Page, BYTES};
use crate::error::RuntimeGap;
use crate::{
    BlitOption, CommandBufferStatus, Error, EventRef, HeapDescriptor, HeapType, Origin, Region,
    ResourceOptions, Size, TextureDescriptor,
};
use metal2vulkan::bind::TileMapping;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The sparse texture's side, in `R32Float` texels.
const SIDE: u64 = 256;

/// The byte the fill writes.
const FILL: u8 = 0x5a;

/// A rebind before the first submission of a buffer an encoder filled, the encoder's blit into a
/// sparse texture leaving a scrub, encodes the encoder again: the command buffer completes with no
/// error, the fill reaches the pages the buffer lies in and not those it left, and the texture's
/// mapped tiles hold what the blit encoded again wrote.
#[test]
fn a_rebind_before_the_first_submission_encodes_a_sparse_scrub_s_encoder_again() {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new()];
    let scratch = Scratch::new("rebind-scrub");
    let device = device(&scratch.0);
    let lacks = |error: &Error| {
        matches!(error, Error::Unsupported(u) if matches!(u.feature, Feature::SparseTextures { .. }))
            || *error == Error::NotImplemented(RuntimeGap::SparseTileMaps)
    };
    let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
    let heap = match device.new_heap(&descriptor) {
        Ok(heap) => heap,
        Err(error) if lacks(&error) => {
            eprintln!("REBIND-SCRUB: no sparse heap: {error:?}");
            return;
        }
        Err(error) => panic!("a sparse heap: {error:?}"),
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = match heap.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(error) if lacks(&error) => {
            eprintln!("REBIND-SCRUB: no sparse texture: {error:?}");
            return;
        }
        Err(error) => panic!("a sparse texture: {error:?}"),
    };
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let buffer = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("REBIND-SCRUB: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let queue = device.new_command_queue().unwrap();
    // The texture's left tiles mapped, in a command buffer of their own: the blit below writes
    // them and the unmapped ones, which its scrub zeroes where the texture holds their memory.
    let tile = device.sparse_tile_size(TextureType::Type2D, PixelFormat::R32Float, 1);
    let (across, down) = (SIDE / tile.width, SIDE / tile.height);
    let mapped = (across / 2).max(1);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    let region = Region::new_2d(0, 0, mapped, down);
    (encoder.update_texture_mapping(&texture, TileMapping::Map, region, 0, 0)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "the mapping");
    let texels: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let source = device
        .new_buffer_with_bytes(&texels, ResourceOptions::shared())
        .unwrap();
    // Its command buffer waits for an event only the host signals, below: the rebind comes before
    // its first submission.
    let gate = device.new_shared_event().unwrap();
    let encoding = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        Size::new(SIDE, SIDE, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.fill_buffer(&buffer, 0..BYTES as u64, FILL).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(buffer.detach_backing(), Ok(true), "the detach");
    // SAFETY: as above.
    let rebound = unsafe { buffer.rebind_bytes_no_copy(pages[1].memory(), None, None) };
    // Read before the submission: the rebind encodes again in its own call.
    let counts = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let reencoded = counts.encoders_reencoded - encoding.encoders_reencoded;
    let (old, new) = (pages[0].written(), pages[1].written());
    println!(
        "REBIND-SCRUB {}: tile {}x{}, {mapped} of {across} columns mapped; rebind {rebound:?}; \
         {status:?} {error:?}; encoders encoded again {reencoded}, refused {}; old pages written \
         {old}, new {new}",
        device.name(),
        tile.width,
        tile.height,
        counts.reencodes_refused - encoding.reencodes_refused,
    );
    rebound.unwrap();
    assert_eq!((status, error), (CommandBufferStatus::Completed, None));
    assert_eq!(reencoded, 1, "the rebind encoded the encoder again");
    let filled = u32::from_ne_bytes([FILL; 4]);
    assert!(
        pages[1].words().iter().all(|&word| word == filled),
        "the fill wrote every byte of the pages the buffer lies in"
    );
    assert!(
        !old,
        "the fill wrote the pages the buffer left before the submission"
    );
    // The mapped tiles hold what the blit encoded again wrote.
    let width = mapped * tile.width;
    let read = device
        .new_buffer(width * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, SIDE, 1),
        &read,
        0,
        width * 4,
        width * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "the read");
    let wanted: Vec<u32> = (0..SIDE)
        .flat_map(|row| (0..width).map(move |column| ((row * SIDE + column) as f32).to_bits()))
        .collect();
    assert_eq!(words_of(&read), wanted, "the mapped tiles' texels");
}
