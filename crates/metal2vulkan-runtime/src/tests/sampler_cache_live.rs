//! A sampler state's cache of Vulkan samplers on a device (`sampler::KeptSamplers`): the
//! descriptions a texture-streaming workload binds one state with, asked of the state itself
//! (`SamplerState::vk_sampler`), are made once and found at every bind after, and the device holds
//! one `VkSampler` per description rather than making and destroying one at each bind.
//!
//! The assertions are counts of what the cache answered and of the device's live samplers, not
//! times: facts of the structure, so they hold on any host under any load (`tests/sampler_cache.rs`
//! says why at length, and measures the same rules without a device).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use super::sampler_cache::{bordered, streaming_descriptions};
use crate::sampler::KeptCounts;

/// One state bound with each of its streaming descriptions in turn, four times over: the first
/// round makes a `VkSampler` for each, and every bind after finds the one it asks for. Nothing is
/// forgotten, so no `vkDestroySampler` runs, and the device's live samplers are the descriptions.
#[test]
fn a_streaming_states_samplers_are_made_once_and_found_at_every_bind_after() {
    let scratch = Scratch::new("sampler-cache");
    let device = device(&scratch.0);
    let before = device.shared().sampler_count.live();
    let state = device.new_sampler_state(&bordered()).expect("a state");
    let stream = streaming_descriptions(device.compile_device(), state.descriptor(), 16);
    assert_eq!(stream.len(), 128, "the state's streaming descriptions");
    for _ in 0..4 {
        for desc in &stream {
            (state.vk_sampler(&device, desc).expect("a Vulkan sampler"))
                .expect("of the state's generation");
        }
    }
    assert_eq!(
        state.kept_counts(),
        (
            KeptCounts {
                found: 3 * 128,
                made: 128,
                forgot: 0,
            },
            128
        ),
        "512 binds, 128 samplers made, none forgotten, 128 kept"
    );
    assert_eq!(
        device.shared().sampler_count.live() - before,
        128,
        "the device holds one sampler per description the state was bound with"
    );
    drop(state);
    assert_eq!(
        device.shared().sampler_count.live(),
        before,
        "the state going destroyed every one of them"
    );
}
