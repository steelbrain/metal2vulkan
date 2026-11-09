//! A device the program let go of is torn down as its last handle goes, whatever work its
//! resources staged for a next submission it never made: a resource's staged work goes as the
//! resource goes (`FirstMoves::forget`), with the staging buffer it copies from, which holds the
//! device. Left for a submission, a `Private` buffer made with bytes, or a sparse texture's tile
//! map, kept its device alive for good once nothing more was submitted (a conformance cost run
//! alone: every sparse texture's case held its device, and its slot of the recorder's, to the
//! process's end).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::Scratch;
use super::draw_again::{device, Heard};
use crate::error::RuntimeGap;
use crate::{Error, HeapDescriptor, HeapType, ResourceOptions, TextureDescriptor};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureUsage};

#[test]
fn a_device_whose_private_buffer_with_bytes_went_unsubmitted_is_torn_down() {
    let scratch = Scratch::new("staged-teardown-bytes");
    let device = device(&scratch.0, &[], &Heard::default());
    let weak = device.downgrade();
    let buffer = device
        .new_buffer_with_bytes(&[7; 256], ResourceOptions::private())
        .unwrap();
    let raw = buffer.at(0).0;
    let firsts = |device: &crate::Device| device.shared().textures.first_moves.fills_wait(raw);
    assert!(firsts(&device), "the bytes' copy waits for a submission");
    drop(buffer);
    assert!(!firsts(&device), "the bytes' copy went with its buffer");
    drop(device);
    assert!(
        weak.upgrade().is_none(),
        "the device outlived its program's last handle"
    );
}

#[test]
fn a_device_whose_sparse_texture_went_unsubmitted_is_torn_down() {
    let scratch = Scratch::new("staged-teardown-sparse");
    let device = device(&scratch.0, &[], &Heard::default());
    let weak = device.downgrade();
    let lacks = |error: &Error| {
        matches!(error, Error::Unsupported(u) if matches!(u.feature, Feature::SparseTextures { .. }))
            || *error == Error::NotImplemented(RuntimeGap::SparseTileMaps)
    };
    let descriptor = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
    let heap = match device.new_heap(&descriptor) {
        Ok(heap) => heap,
        Err(error) if lacks(&error) => {
            eprintln!("STAGED-WORK-TEARDOWN: no sparse heap: {error:?}");
            return;
        }
        Err(error) => panic!("a sparse heap: {error:?}"),
    };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 256, 256, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = match heap.new_texture(&descriptor) {
        Ok(texture) => texture,
        Err(error) if lacks(&error) => {
            eprintln!("STAGED-WORK-TEARDOWN: no sparse texture: {error:?}");
            return;
        }
        Err(error) => panic!("a sparse texture: {error:?}"),
    };
    let tile_map = texture
        .sparse_image()
        .and_then(|image| image.tile_map())
        .map(|tile_map| tile_map.buffer.at(0).0)
        .expect("a 2D sparse texture's tile map");
    let firsts = |device: &crate::Device| device.shared().textures.first_moves.fills_wait(tile_map);
    assert!(
        firsts(&device),
        "the tile map's bytes wait for a submission"
    );
    drop(texture);
    drop(heap);
    assert!(!firsts(&device), "the tile map's bytes went with it");
    drop(device);
    assert!(
        weak.upgrade().is_none(),
        "the device outlived its program's last handle"
    );
}
