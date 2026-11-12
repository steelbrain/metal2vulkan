//! What a sampler state's cache of Vulkan samplers (`sampler::KeptSamplers`) is asked for, and
//! what it does with it: how many distinct Vulkan sampler descriptions one Metal sampler state
//! has, and the binds that then find a sampler kept, make one, or forget one.
//!
//! # Why these are counts
//!
//! A bind that finds its sampler kept costs a hash of the description; one that does not costs a
//! `vkCreateSampler` and, where the cache was full, a `vkDestroySampler` with it --- a driver's
//! object making and release, the work the counts of `churn_counts.rs` exist to name. So what the
//! cache is worth is how many binds it answers, and the assertions here are counts of those: facts
//! of the structure that hold on any host under any load, that need no quiet host, and that depend
//! on nothing that ran before them in the process.
//!
//! The descriptions are the real ones, read through `bind::sampler_desc` over every committed
//! device profile (`validation/profiles`), so the count of a state's descriptions is a fact of the
//! translator's sampler rules and not of a case written here.

use super::super::sampler::{HashedDesc, KeptCounts, KeptSamplers};
use super::capabilities::committed_descriptions;
use metal2vulkan::bind::{sampler_desc, SamplerDescriptor};
use metal2vulkan::device::{Device, DevicePolicy};
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse};
use metal2vulkan::metal::{
    SamplerAddressMode as A, SamplerBorderColor as B, SamplerMinMagFilter as F,
    SamplerMipFilter as M,
};
use std::collections::HashSet;
use std::sync::Arc;

/// The cache the state keeps was sized at this, so a workload cycling the descriptions a state has
/// finds every one of them (`sampler::KEPT`, which is private).
const KEPT: usize = 128;

/// Every way a function can sample through one sampler state: a depth comparison and an integer
/// texture, the four uses of the facts Metal keeps out of the descriptor that the heap's sampler
/// classes name (`pipeline::sym::HandleClass::Sampler`).
fn uses() -> Vec<SamplerUse> {
    let mut out = Vec::new();
    for compare in [false, true] {
        for integer in [false, true] {
            out.push(SamplerUse::new(compare, integer));
        }
    }
    out
}

/// A trilinear sampler whose border is opaque black on every axis that clamps to it: the shape
/// with the most distinct Vulkan descriptions, since the border is read one way through a view the
/// device reads Metal's texel through and another way through one it does not
/// (`metal::sampler::opaque_black_border`).
pub(super) fn bordered() -> SamplerDescriptor {
    let mut d = SamplerDescriptor::new();
    d.min_filter = F::Linear;
    d.mag_filter = F::Linear;
    d.mip_filter = M::Linear;
    d.border_color = B::OpaqueBlack;
    d.s_address_mode = A::ClampToBorderColor;
    d.t_address_mode = A::ClampToBorderColor;
    d
}

/// The Vulkan sampler descriptions one state of `descriptor` has on `device`, over every use and
/// every way a view's opaque-black border is read, with `levels` lower LOD clamps replacing the
/// descriptor's as a bind's do (`set_sampler_state_with_lod_clamps`): a texture-streaming workload
/// clamps to the lowest resident mip level, so `levels` of them is one state through a texture of
/// that many levels.
pub(super) fn streaming_descriptions(
    device: &Device,
    descriptor: &SamplerDescriptor,
    levels: u32,
) -> Vec<HashedDesc> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for level in 0..levels.max(1) {
        let mut d = *descriptor;
        if levels > 0 {
            d.lod_min_clamp = level as f32;
            d.lod_max_clamp = 15.0;
        }
        for use_ in uses() {
            for black in [OpaqueBlackBorder::Stated, OpaqueBlackBorder::Transparent] {
                let Ok(desc) = sampler_desc(device, &d, use_, black, None) else {
                    continue;
                };
                if seen.insert(desc) {
                    out.push(HashedDesc::new(desc));
                }
            }
        }
    }
    out
}

/// Every committed profile's device.
fn devices() -> Vec<(String, Device)> {
    committed_descriptions()
        .into_iter()
        .map(|(stem, desc)| {
            let device = Device::new(desc, DevicePolicy::default())
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            (stem, device)
        })
        .collect()
}

/// A state bound without a bind's LOD clamps has a Vulkan sampler per use a function samples it
/// through and per form of its opaque-black border, eight at the most and four for Metal's default
/// descriptor (whose nearest filters and unmipmapped levels leave a comparison and an integer
/// texture the only facts that change it), on every committed profile: well inside what the state
/// keeps, so the cache answers every bind of such a state after the first of each use.
#[test]
fn a_state_has_at_most_eight_descriptions_without_a_binds_lod_clamps() {
    for (stem, device) in devices() {
        let plain = streaming_descriptions(&device, &SamplerDescriptor::new(), 0);
        let bordered = streaming_descriptions(&device, &bordered(), 0);
        assert_eq!(plain.len(), 4, "{stem}: Metal's default sampler state");
        assert_eq!(bordered.len(), 8, "{stem}: an opaque-black border");
        assert!(
            bordered.len() <= KEPT,
            "{stem}: a state's descriptions fit in what it keeps"
        );
    }
}

/// A state a texture-streaming workload binds with the lower LOD clamp of each resident mip level
/// has eight Vulkan samplers per level: 128 over the sixteen levels of the largest mip chain
/// Metal's level grid reaches, on every committed profile. That is what the cache is sized at, so
/// such a workload finds every one of them.
#[test]
fn a_streaming_state_has_a_hundred_and_twenty_eight() {
    for (stem, device) in devices() {
        let streaming = streaming_descriptions(&device, &bordered(), 16);
        assert_eq!(
            streaming.len(),
            128,
            "{stem}: sixteen clamp levels, eight descriptions each"
        );
        assert_eq!(streaming.len(), KEPT, "{stem}: what the state keeps");
    }
}

/// A bind's LOD clamps are handed to the device as given, so clamps off the integer levels are
/// distinct descriptions and a workload can ask one state for more than it keeps: a thousand lower
/// clamps across one level are a thousand of them. The cache is bounded whatever a workload asks,
/// which is why its lookup may not be a walk.
#[test]
fn a_binds_clamps_are_handed_to_the_device_as_given() {
    let mut d = SamplerDescriptor::new();
    d.min_filter = F::Linear;
    d.mag_filter = F::Linear;
    d.mip_filter = M::Linear;
    for (stem, device) in devices() {
        let mut seen = HashSet::new();
        for step in 0..1000u32 {
            let mut d = d;
            d.lod_min_clamp = step as f32 / 1000.0;
            d.lod_max_clamp = 15.0;
            let desc = sampler_desc(
                &device,
                &d,
                SamplerUse::default(),
                OpaqueBlackBorder::Stated,
                None,
            )
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            seen.insert(desc);
        }
        assert_eq!(seen.len(), 1000, "{stem}: every clamp its own description");
    }
}

/// Bind every description of `stream` in turn, `rounds` times, over a cache of `limit`: what the
/// cache did. Each sampler the cache makes is a value of its own that nothing else holds, as a
/// `VkSampler` of a state no command buffer has bound is.
fn cycled(stream: &[HashedDesc], limit: usize, rounds: usize) -> KeptCounts {
    let mut cache: KeptSamplers<()> = KeptSamplers::new();
    for _ in 0..rounds {
        for desc in stream {
            if cache.found(desc).is_some() {
                continue;
            }
            cache.make_room(limit);
            cache.keep(*desc, &Arc::new(()));
        }
    }
    cache.counts()
}

/// A workload cycling the 128 descriptions of a streaming sampler state finds every one of them
/// after the first round, and forgets none: the cache makes 128 Vulkan samplers over the whole
/// run, however many binds it serves.
#[test]
fn the_cache_answers_every_bind_of_a_streaming_state_after_the_first_round() {
    let (stem, device) = devices().into_iter().next().expect("a committed profile");
    let stream = streaming_descriptions(&device, &bordered(), 16);
    assert_eq!(stream.len(), 128, "{stem}");
    let counts = cycled(&stream, KEPT, 4);
    assert_eq!(
        counts,
        KeptCounts {
            found: 3 * 128,
            made: 128,
            forgot: 0,
        },
        "{stem}: 512 binds, 128 samplers made, none forgotten"
    );
}

/// The same workload over a cache of sixteen finds **nothing**: a cycle longer than the cache is
/// the worst case for forgetting the least recently bound, so every one of the 512 binds makes a
/// `VkSampler` and destroys the one it forgets to make room. This is what a cache of sixteen cost,
/// and why the bound is [`KEPT`].
#[test]
fn a_cache_of_sixteen_answers_no_bind_of_a_streaming_state() {
    let (stem, device) = devices().into_iter().next().expect("a committed profile");
    let stream = streaming_descriptions(&device, &bordered(), 16);
    let counts = cycled(&stream, 16, 4);
    assert_eq!(
        counts,
        KeptCounts {
            found: 0,
            made: 4 * 128,
            forgot: 4 * 128 - 16,
        },
        "{stem}: every bind made a sampler, and all but the sixteen kept were forgotten"
    );
}

/// A state bound without a bind's LOD clamps is answered by a cache of sixteen as well as by the
/// one the state keeps: its eight descriptions fit in both. Raising the bound took nothing from
/// such a state and gave it nothing.
#[test]
fn a_state_without_a_binds_clamps_is_answered_by_either_bound() {
    let (stem, device) = devices().into_iter().next().expect("a committed profile");
    let stream = streaming_descriptions(&device, &bordered(), 0);
    let want = KeptCounts {
        found: 3 * 8,
        made: 8,
        forgot: 0,
    };
    assert_eq!(cycled(&stream, KEPT, 4), want, "{stem}: what it keeps");
    assert_eq!(cycled(&stream, 16, 4), want, "{stem}: sixteen");
}

/// A full cache forgets the least recently bound sampler that only the cache holds: a sampler a
/// pending command buffer still holds stays, and once every one kept is held the least recently
/// bound of those leaves the cache, its holders keeping its `VkSampler` until they complete.
#[test]
fn a_full_cache_forgets_the_least_recently_bound_sampler_nothing_else_holds() {
    let (stem, device) = devices().into_iter().next().expect("a committed profile");
    let stream = streaming_descriptions(&device, &bordered(), 0);
    /// One bind of the sampler `at` of the stream, as `SamplerState::vk_sampler` binds one, over a
    /// cache of three.
    fn bind(cache: &mut KeptSamplers<u32>, desc: &HashedDesc, sampler: &Arc<u32>) {
        if cache.found(desc).is_none() {
            cache.make_room(3);
            cache.keep(*desc, sampler);
        }
    }
    // Two samplers a pending command buffer holds, bound before a third the cache alone holds.
    let held = Arc::new(0);
    let mut cache = KeptSamplers::new();
    bind(&mut cache, &stream[0], &held);
    bind(&mut cache, &stream[1], &held);
    bind(&mut cache, &stream[2], &Arc::new(2));
    assert_eq!(cache.samplers_kept(), 3, "{stem}: three kept");
    cache.make_room(3);
    assert_eq!(cache.samplers_kept(), 2, "{stem}: one went");
    assert!(
        cache.found(&stream[2]).is_none(),
        "{stem}: the one only the cache held went, though it was bound last"
    );
    assert!(
        cache.found(&stream[0]).is_some(),
        "{stem}: the least recently bound stayed, a command buffer holding it"
    );
    // Every one kept held: nothing is the cache's alone to forget, and the least recently bound
    // leaves the cache instead.
    let mut cache = KeptSamplers::new();
    for desc in &stream[..3] {
        bind(&mut cache, desc, &held);
    }
    assert!(!cache.forget_spare(), "{stem}: none is the cache's alone");
    cache.make_room(3);
    assert_eq!(cache.samplers_kept(), 2, "{stem}: the first left the cache");
    assert!(
        cache.found(&stream[0]).is_none(),
        "{stem}: the least recently bound is the one that left"
    );
}
