//! A bind state's shape (`BindState::shape`) over a buffer's bind offsets.

use crate::bind::{
    ArgumentSampler, BindState, BoundBuffer, BoundSampler, BoundTexture, SamplerDescriptor,
    StageTable, TextureFacts,
};
use crate::metal::{CompareFunction, PixelFormat, SamplerMinMagFilter, TextureType};
use crate::pipeline::sym::Stage;

/// A buffer of `length` bytes bound at `offset`.
fn buffer(length: u64, offset: u64) -> BoundBuffer {
    BoundBuffer::Buffer {
        length,
        offset,
        gpu_address: 0x1000,
        device_address: None,
        attribute_stride: None,
    }
}

/// A 2D texture of `format`.
fn texture(format: PixelFormat) -> BoundTexture {
    BoundTexture::Texture {
        facts: TextureFacts::new(TextureType::Type2D, format),
        resource_id: Some(7),
    }
}

/// A sampler that filters linearly where `linear`, and compares where `compares`.
fn sampler(linear: bool, compares: bool) -> SamplerDescriptor {
    let mut desc = SamplerDescriptor::new();
    if linear {
        desc.min_filter = SamplerMinMagFilter::Linear;
        desc.mag_filter = SamplerMinMagFilter::Linear;
    }
    if compares {
        desc.compare_function = CompareFunction::Less;
    }
    desc
}

/// The base: a vertex buffer, a fragment texture and sampler, and one argument sampler.
fn base() -> BindState {
    let mut state = BindState::default();
    let mut vertex = StageTable::default();
    vertex.buffers.insert(0, buffer(256, 0));
    state.stages.insert(Stage::Vertex, vertex);
    let mut fragment = StageTable::default();
    fragment
        .textures
        .insert(0, texture(PixelFormat::RGBA8Unorm));
    fragment
        .samplers
        .insert(0, BoundSampler::new(sampler(false, false)));
    state.stages.insert(Stage::Fragment, fragment);
    state
        .argument_samplers
        .push(ArgumentSampler::new(sampler(false, false), 1));
    state
}

/// The base changed by `change`.
fn with(change: impl FnOnce(&mut BindState)) -> BindState {
    let mut state = base();
    change(&mut state);
    state
}

/// `state`'s vertex table.
fn vertex(state: &mut BindState) -> &mut StageTable {
    state.stages.get_mut(&Stage::Vertex).unwrap()
}

/// A buffer's bind offset cycling through the multiples of 16 up to 256 keeps one shape: no
/// alignment past 4 is told apart, and no shape is new on a second cycle.
#[test]
fn a_slot_s_offsets_moving_among_multiples_of_16_keep_their_shapes() {
    let mut seen = Vec::new();
    for cycle in 0..2 {
        for offset in (16..=256).step_by(16) {
            let state = with(|s| {
                vertex(s).buffers.insert(0, buffer(4096, offset));
            });
            let shape = state.shape();
            if !seen.contains(&shape) {
                assert_eq!(
                    cycle, 0,
                    "a new shape at offset {offset} on the second cycle"
                );
                seen.push(shape);
            }
        }
    }
    assert_eq!(seen.len(), 1);
}
