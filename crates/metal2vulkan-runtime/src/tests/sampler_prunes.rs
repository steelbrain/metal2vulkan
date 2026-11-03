//! The device's live sampler states (`sampler::LiveSamplers`): a creation prunes the map of the
//! states gone only where it has grown to twice what the last prune left, so creations look at a
//! few entries each however many states live, an equal request shares the live state, and the
//! map holds at most about twice the live states.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::SamplerState;
use metal2vulkan::bind::SamplerDescriptor;

/// The states kept live at once.
const LIVE: usize = 4096;

/// A state of the default descriptor but its upper LOD clamp, `max`: another state for each.
fn clamped(device: &crate::Device, max: f32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.lod_max_clamp = max;
    device
        .new_sampler_state(&descriptor)
        .expect("a sampler state")
}

/// [`LIVE`] states made and kept live, each another: their creations' prunes look at fewer
/// entries than twice the states, where a prune at every creation looks at them all each time
/// (about `LIVE * LIVE / 2`). A request equal to a live one's gets that state. With every state
/// gone, as many more made and each dropped at once leave the map at its least bound.
#[test]
fn creations_among_thousands_of_live_states_prune_a_few_entries_each() {
    let scratch = Scratch::new("sampler-prunes");
    let device = device(&scratch.0);
    let (_, before) = device.live_sampler_entries();
    let states: Vec<SamplerState> = (0..LIVE)
        .map(|k| clamped(&device, 1.0 + k as f32 / 4.0))
        .collect();
    let (entries, looked_at) = device.live_sampler_entries();
    assert!(entries >= LIVE, "{entries} entries for {LIVE} live states");
    assert!(
        looked_at - before < 2 * LIVE,
        "the prunes looked at {} entries over {LIVE} creations",
        looked_at - before
    );
    let again = clamped(&device, 1.0);
    assert_eq!(
        again.gpu_resource_id(),
        states[0].gpu_resource_id(),
        "an equal request got another state"
    );
    drop((states, again));
    for k in 0..LIVE {
        drop(clamped(&device, -1.0 - k as f32 / 4.0));
    }
    let (entries, _) = device.live_sampler_entries();
    assert!(
        entries <= 65,
        "{entries} entries with no state live past the creation"
    );
}
