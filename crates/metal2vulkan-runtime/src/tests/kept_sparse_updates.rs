//! A completed command buffer the program keeps, as a Metal program keeps a completed
//! `MTLCommandBuffer` whose status, error and GPU times it reads later, holds none of the textures
//! its resource state encoders' updates map (`sparse/encoder.rs`'s `SparseUpdate`) past its
//! completion: Metal lets go of what a command buffer used as it completes, whether the program
//! keeps the command buffer or not.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. It prints a `KEPT-SPARSE` line, and returns where the device makes no
//! sparse texture.

use super::blit_holds::{device, Scratch, WAIT};
use crate::error::RuntimeGap;
use crate::{
    CommandBufferStatus, Error, HeapDescriptor, HeapType, Region, ResourceOptions,
    TextureDescriptor,
};
use metal2vulkan::bind::TileMapping;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};

/// The sparse texture's side, in `R32Float` texels.
const SIDE: u64 = 256;

/// A mapping of every tile of a sparse texture the program let go of once it encoded the
/// mapping: the command buffer, kept by the program, holds none of the texture's handles once it
/// completed, and the mapping ran.
#[test]
fn a_kept_completed_command_buffer_holds_no_texture_its_mapping_update_names() {
    let scratch = Scratch::new("kept-sparse");
    let device = device(&scratch.0);
    let lacks = |error: &Error| {
        matches!(error, Error::Unsupported(u) if matches!(u.feature, Feature::SparseTextures { .. }))
            || *error == Error::NotImplemented(RuntimeGap::SparseTileMaps)
    };
    let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
    let heap = match device.new_heap(&descriptor) {
        Ok(heap) => heap,
        Err(error) if lacks(&error) => {
            eprintln!("KEPT-SPARSE: no sparse heap: {error:?}");
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
            eprintln!("KEPT-SPARSE: no sparse texture: {error:?}");
            return;
        }
        Err(error) => panic!("a sparse texture: {error:?}"),
    };
    let queue = device.new_command_queue().unwrap();
    let tile = device.sparse_tile_size(TextureType::Type2D, PixelFormat::R32Float, 1);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.resource_state_command_encoder().unwrap();
    let region = Region::new_2d(0, 0, SIDE / tile.width, SIDE / tile.height);
    (encoder.update_texture_mapping(&texture, TileMapping::Map, region, 0, 0)).unwrap();
    encoder.end_encoding().unwrap();
    // The handles' retain, which every handle of the program's holds, and the texture object.
    let (handles, object) = (texture.retained(), texture.downgrade());
    drop(texture);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let kept = (handles.strong_count(), object.strong_count());
    let finished = (command_buffer.status(), command_buffer.error());
    drop(command_buffer);
    println!(
        "KEPT-SPARSE {}: {finished:?}; the texture's handles' retain and object held after its \
         completion {kept:?}, after its drop {:?}",
        device.name(),
        (handles.strong_count(), object.strong_count()),
    );
    assert_eq!(finished, (CommandBufferStatus::Completed, None));
    assert_eq!(
        kept.0, 0,
        "the kept command buffer still held the mapped texture's handle after its completion"
    );
    drop(heap);
}
