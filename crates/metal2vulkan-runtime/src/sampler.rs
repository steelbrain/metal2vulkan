//! `SamplerState` (`MTLSamplerState`) and `Device::new_sampler_state`.
//!
//! A sampler state is its descriptor (the compile layer's [`SamplerDescriptor`], Metal's
//! properties) and the `VkSampler`s its binds need. One Metal sampler is several Vulkan ones: how
//! a function samples through it (with depth comparison or not) and the texture it meets (an
//! integer one is never filtered) decide fields Metal keeps out of the descriptor
//! (`bind::sampler_desc`), and a bind's LOD clamps (`set_sampler_state_with_lod_clamps`) replace
//! the descriptor's. Each distinct Vulkan sampler is made at its first bind and kept by the state
//! ([`KeptSamplers`]), at most [`KEPT`] of them and at most as many as the device allows live:
//! past that the least recently bound one is forgotten, and the command buffers that bound it keep
//! it until they complete.
//!
//! **Metal's rules** (probe 7: an Apple M3 Max on macOS 26.7 and an M4 on 27.0.1, each case run
//! without and with API validation). Metal's API validation stops, at
//! `newSamplerStateWithDescriptor:`:
//!
//! - a `maxAnisotropy` outside 1 to 16;
//! - a pixel-coordinate sampler (`normalizedCoordinates` false) with a `maxAnisotropy` other than
//!   1, a mip filter other than `NotMipmapped`, or an address mode other than `ClampToEdge`,
//!   `ClampToZero` and `ClampToBorderColor` on any axis. Its two filters may differ, and its LOD
//!   clamps, comparison and `lodAverage` are free.
//!
//! Plain Metal makes both, and so does the runtime ([`taken_descriptor`]). A `maxAnisotropy` of 1
//! to 16 samples as that value rounded up to a power of two: 3 as 4, 5 to 7 as 8, 9 to 15 as 16
//! (an M4 on macOS 27.0.1: every value through runtime and constexpr samplers at 1, 268 footprints
//! of ratios 1 to 32, each answer bit for bit the power of two's, the powers told apart at 388 to
//! 1, 191 of them). A `maxAnisotropy` outside 1 to 16 samples as the low five bits of the value
//! taken through a 32-bit float, saturated to 32 bits, 0 as 1 and 17 to 31 as 32, a value of 1 to
//! 16 rounded up so ([`metal_anisotropy`]; an M4 on macOS 27.0.1: 0, 17 to 2^64 - 1, each sampling
//! as one of 1, 2, 4, 8, 16 and 32 at gradients that tell those apart, footprints up to 256 to 1,
//! two runs alike). An anisotropy past the device's largest (`maxSamplerAnisotropy`, 16 on every
//! venue) filters with that largest, a near-equivalent announced once
//! (`Degradation::CappedAnisotropy`). A pixel-coordinate sampler Metal makes and samples its repeat
//! and mirrored-repeat axes as `ClampToEdge`, at level 0 whatever its mip filter, LOD or gradients,
//! and without anisotropy (an M4 on macOS 27.0.1: nearest and linear filters at 22 points inside,
//! on and past the edges, every answer the `ClampToEdge` sampler's): so does the runtime
//! ([`taken_descriptor`]). A `MirrorClampToEdge` `s` or `t` axis it mirrors once about 0 and then
//! clamps, by texel index: each tap's index `i`, a constant texel offset added first (but dropped
//! where the base level is 3, 5, 6 or 7 texels wide, as on a repeating axis), reads texel `-1 - i`
//! where it is negative, then clamped into the level, as a normalized sampler's axis does (an M4 on
//! macOS 27.0.1: nearest and linear samples, gathers and depth comparisons of textures 1 to 8
//! texels wide, offsets -7 to 5, two runs alike): so does the runtime, keeping the mode, whose
//! Vulkan sampler mirrors once and clamps (`MIRROR_CLAMP_TO_EDGE`) at the coordinate the function
//! divides by the texture's size, the function placing the offset and filtering linearly as Metal
//! does. Address modes that hold both `ClampToZero` and `ClampToBorderColor`, which Metal's
//! validation layer stops where the border color is not transparent black, plain Metal makes, and
//! it reads the border color on every border axis: so does the runtime, as it reads a `ClampToZero`
//! axis's border on any sampler. Metal takes any LOD clamps, at creation and in a bind (inverted,
//! NaN and infinite ones included): so does the runtime. Metal raises a sample's level to the lower
//! clamp and then lowers it to the upper one, so inverted clamps read the upper clamp's level,
//! filtering as clamps of the upper clamp alone do, a descriptor's and a bind's alike: so does the
//! runtime ([`metal_lod_clamps`]; an M4 on macOS 27.0.1, and the M3 Max on macOS 26.7). A sampler
//! whose clamps where it is bound are NaN is refused where it is used, by the binding plan, since
//! what Metal samples then is not measured. Metal takes any raw border color and keeps its low two
//! bits, sampling 3 as transparent black on every format measured ([`sampler_border_color`], the
//! same probes). What the device cannot do with the state on any use (a mirror-clamp address mode
//! it lacks) is refused here, `Unsupported` ([`check_sampler`]). Anisotropy on a device without
//! `samplerAnisotropy` is made: its Vulkan samplers are isotropic, and a function samples through
//! the state with Metal's anisotropic filter emulated where the emulation reaches the sample
//! (`Degradation::AnisotropicFilter`) and the device's isotropic filter elsewhere
//! (`Degradation::DeviceAnisotropicFilter`), each announced; an argument buffer that lists it is
//! refused where it is used, `Unsupported` (`samplerAnisotropy`), since a function reading it
//! there knows not its anisotropy. A minimum or maximum reduction reads the weighted average, as
//! on an Apple M4 without Metal's validation layer (which refuses one), and
//! `DeviceProperties::supports_sampler_reduction_mode` answers no.
//!
//! A min or mag filter `Cubic` (`MTLSamplerMinMagFilterCubic`, raw value 2) is made. Without a
//! mip filter, Metal samples through such a state as through a constexpr `filter::bicubic`
//! sampler of its address modes, border color and coordinates, whatever its other filter (our
//! probe, the M3 Max and an M4 alike): a pipeline samples through it on its cubic-sampler variant,
//! which emulates that filter. With a mip filter, `MirrorClampToEdge`, a minimum or maximum
//! reduction, anisotropy, or through an argument buffer, the dispatch or draw that samples through
//! it is refused `NotImplemented(SamplerBicubicFilter)`: Metal's level rule through it is not
//! modelled, and the others are not measured.

use super::degradation::Degradation;
use super::device::lock;
use super::error::{Error, SamplerLimit};
use super::exit_guard;
use super::key_hash::KeyMap;
use super::resource_ids::{ElementHold, Holder, IdHold, IdPlacement, IdSpace, ResourceId};
use super::retained::HeldBy;
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::sampler::{check_sampler, SamplerUse};
use metal2vulkan::metal::{
    CompareFunction, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    SamplerReductionMode,
};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// How a sampler state is made beyond its descriptor: the rest of `MTLSamplerDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerStateOptions {
    /// `supportArgumentBuffers`. It decides only whether two requests share a state (probe 5):
    /// every state has a `gpuResourceID`, which an argument buffer can hold to name it, and
    /// Metal samples through the ID of a state without the support too (probe 7).
    pub support_argument_buffers: bool,
    /// `resourceIndex` with `forceResourceIndex` (Apple's private properties, a runtime
    /// extension): where the state goes in the space of sampler resource IDs
    /// (`resource_ids.rs`). A request equal to a live state's gets that state, its ID whatever
    /// this asks, as Metal's does (probe 5).
    pub placement: IdPlacement,
}

/// What decides whether two requests share one state (probe 5, an M3 Max and an M4: each
/// descriptor field changed alone, and `supportArgumentBuffers`, gave another state; the
/// resource index never did): every field of the descriptor, its LOD clamps by their bits, and
/// argument-buffer support. Metal's label decides too; the runtime's states have none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SamplerKey {
    filters: (SamplerMinMagFilter, SamplerMinMagFilter, SamplerMipFilter),
    address_modes: [SamplerAddressMode; 3],
    border_color: SamplerBorderColor,
    normalized_coordinates: bool,
    lod_clamps: [u32; 2],
    lod_average: bool,
    max_anisotropy: u32,
    compare_function: CompareFunction,
    reduction_mode: SamplerReductionMode,
    support_argument_buffers: bool,
}

impl SamplerKey {
    fn of(descriptor: &SamplerDescriptor, options: SamplerStateOptions) -> SamplerKey {
        let d = descriptor;
        SamplerKey {
            filters: (d.min_filter, d.mag_filter, d.mip_filter),
            address_modes: [d.s_address_mode, d.t_address_mode, d.r_address_mode],
            border_color: d.border_color,
            normalized_coordinates: d.normalized_coordinates,
            lod_clamps: [d.lod_min_clamp.to_bits(), d.lod_max_clamp.to_bits()],
            lod_average: d.lod_average,
            max_anisotropy: d.max_anisotropy,
            compare_function: d.compare_function,
            reduction_mode: d.reduction_mode,
            support_argument_buffers: options.support_argument_buffers,
        }
    }
}

/// The device's live sampler states, by what decides whether a request shares one
/// ([`SamplerKey`]): an equal request gets the live state, as Metal's does, and an entry whose
/// state is gone answers none. Such an entry stays until a creation prunes the map, which one
/// does where the map has grown to twice what the last prune left, and at least to
/// [`PRUNE_FROM`]: so each creation's share of the prunes is a few entries however many states
/// live, and the map holds at most about twice the live states.
#[derive(Default)]
pub(super) struct LiveSamplers {
    states: HashMap<SamplerKey, Weak<SamplerShared>>,
    /// The length at which a creation prunes first.
    prune_at: usize,
    /// The entries the prunes have looked at (the device tests: `tests/sampler_prunes.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    looked_at: usize,
}

/// The least length at which a creation prunes the live states' map.
const PRUNE_FROM: usize = 64;

impl LiveSamplers {
    /// `key`'s live state, where one lives.
    fn live(&self, key: &SamplerKey) -> Option<Arc<SamplerShared>> {
        self.states.get(key).and_then(Weak::upgrade)
    }

    /// Keep `state` as `key`'s, first dropping the entries of the states gone where the map has
    /// grown to its bound. Dropping one lets go of a `Weak` alone: no state is dropped under the
    /// map's lock.
    fn keep(&mut self, key: SamplerKey, state: &Arc<SamplerShared>) {
        if self.states.len() >= self.prune_at.max(PRUNE_FROM) {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            {
                self.looked_at += self.states.len();
            }
            self.states.retain(|_, state| state.strong_count() > 0);
            self.prune_at = 2 * self.states.len();
        }
        self.states.insert(key, Arc::downgrade(state));
    }
}

impl Device {
    /// The length of the device's map of live sampler states and the entries its prunes have
    /// looked at (the device tests).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn live_sampler_entries(&self) -> (usize, usize) {
        let live = lock(&self.shared().samplers);
        (live.states.len(), live.looked_at)
    }
}

/// The most Vulkan samplers one sampler state keeps for its binds, the device's
/// `maxSamplerAllocationCount` being the other ceiling: a state keeps at most the smaller of the
/// two, and the device's live samplers are held to its limit on every creation ([`SamplerCount`]).
///
/// A state's distinct Vulkan samplers are its descriptor's, one per fact Metal keeps out of the
/// descriptor: how a function samples through it (`SamplerUse`, four in all: a depth comparison
/// and an integer texture), the opaque-black border a view needs, and the LOD clamps a bind
/// replaces the descriptor's with. Without a bind's clamps a state has at most **8** descriptions
/// on every committed device profile, and over the sixteen mip levels a texture-streaming
/// workload clamps to it has **128** (`tests/sampler_cache.rs`, which enumerates
/// them through `bind::sampler_desc` over every profile in `validation/profiles`). A bind's clamps
/// are handed to the device as given, so a workload clamping off the integer levels can ask one
/// state for more than this; it then forgets and makes Vulkan samplers, which is why the lookup is
/// a map and not a walk of what the state keeps ([`KeptSamplers`]).
const KEPT: usize = 128;

/// The device's live Vulkan samplers, at most `maxSamplerAllocationCount`: every sampler the
/// runtime makes (a state's, a pipeline layout's immutable one, a neutral one) is counted from its
/// creation to its destruction, and one past the limit is refused, `Error::SamplerLimit`, rather
/// than made, which Vulkan does not allow. A state's samplers are at most [`KEPT`] in its cache
/// plus those the command buffers that bound them still hold; one per view mapping (an opaque-black
/// border on a device without `borderColorSwizzleFromImage`) is one more such sampler, so a
/// workload cycling views is held to the device's limit too.
pub(super) struct SamplerCount {
    live: AtomicU32,
    limit: u32,
}

impl SamplerCount {
    /// No sampler live, of `limit`.
    pub(super) fn new(limit: u32) -> SamplerCount {
        SamplerCount {
            live: AtomicU32::new(0),
            limit,
        }
    }

    /// Count one more live sampler, or refuse it where `limit` are live.
    pub(super) fn take(&self) -> Result<(), Error> {
        self.live
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                (live < self.limit).then_some(live + 1)
            })
            .map(|_| ())
            .map_err(|_| Error::SamplerLimit(SamplerLimit { limit: self.limit }))
    }

    /// One counted sampler was destroyed.
    pub(super) fn give(&self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }

    /// The samplers live now (the device tests).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn live(&self) -> u32 {
        self.live.load(Ordering::Acquire)
    }

    /// The device's `maxSamplerAllocationCount`: the samplers it allows live at once, which is
    /// also a ceiling on what one sampler state keeps ([`KEPT`]).
    pub(super) fn limit(&self) -> u32 {
        self.limit
    }
}

/// The Vulkan samplers a sampler state made for its binds, each by its description.
///
/// Every bind of the state asks it for the sampler of one description --- a separate sampler
/// slot's, a heap set's sampler array's, an argument buffer's ([`SamplerState::vk_sampler`]) ---
/// so the encoding thread reads this at every draw and dispatch that samples through the state.
/// It is a map ([`KeyMap`], whose keys are descriptions of Vulkan objects as its docs describe),
/// not a list walked from the least recently bound: a bind pays one hash of the description
/// however many samplers the state keeps, so the state can keep enough of them for a workload
/// cycling its descriptions to find them all rather than make and destroy a `VkSampler` at every
/// bind. Making room is the only walk, on a bind that makes a sampler with the cache full.
///
/// `T` is [`OwnedSampler`]; the tests drive these rules over a value of their own.
pub(super) struct KeptSamplers<T> {
    /// The samplers kept, each with the bind that last read it.
    kept: KeyMap<metal2vulkan::vk::SamplerDesc, Kept<T>>,
    /// The binds that read the cache so far: what stamps the sampler a bind reads or makes.
    binds: u64,
    /// What the cache has done ([`KeptCounts`]). Only the tests that measure the cache read it, so
    /// a build that runs none counts nothing.
    #[cfg(test)]
    counts: KeptCounts,
}

/// What a sampler state's cache of Vulkan samplers has done since the state was made
/// ([`KeptSamplers`]). A `found` of nothing over many binds is a workload cycling more
/// descriptions than the state keeps, which makes and destroys a `VkSampler` at each of them.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct KeptCounts {
    /// The binds that found the sampler they asked for kept.
    pub(super) found: u64,
    /// The binds that made the sampler they asked for.
    pub(super) made: u64,
    /// The kept samplers forgotten, to make room for another or to leave the device a slot.
    pub(super) forgot: u64,
}

/// A Vulkan sampler a state keeps ([`KeptSamplers`]).
struct Kept<T> {
    sampler: Arc<T>,
    /// The count of binds at the one that last read it.
    read_at: u64,
}

impl<T> KeptSamplers<T> {
    /// No sampler kept.
    pub(super) fn new() -> KeptSamplers<T> {
        KeptSamplers {
            kept: KeyMap::default(),
            binds: 0,
            #[cfg(test)]
            counts: KeptCounts::default(),
        }
    }

    /// The sampler of `desc` this bind asks for, where the cache keeps it, read now so that it is
    /// the last the cache forgets. Counts the bind, found or not.
    pub(super) fn found(&mut self, desc: &metal2vulkan::vk::SamplerDesc) -> Option<Arc<T>> {
        self.binds += 1;
        let read_at = self.binds;
        let kept = self.kept.get_mut(desc)?;
        kept.read_at = read_at;
        let sampler = Arc::clone(&kept.sampler);
        #[cfg(test)]
        {
            self.counts.found += 1;
        }
        Some(sampler)
    }

    /// Forget samplers until fewer than `limit` are kept.
    pub(super) fn make_room(&mut self, limit: usize) {
        while self.kept.len() >= limit.max(1) {
            self.forget(false);
        }
    }

    /// Forget the least recently bound sampler that only the cache holds, whose `VkSampler` is
    /// destroyed as it goes and its slot of the device's samplers given back
    /// ([`SamplerCount::give`]), so that a device holding as many as it allows has room for one
    /// more: whether one went.
    pub(super) fn forget_spare(&mut self) -> bool {
        self.forget(true)
    }

    /// Forget the least recently bound sampler, one only the cache holds first: its `VkSampler` is
    /// destroyed as it goes, where one a pending command buffer holds stays until the buffer
    /// completes. `spare_only` forgets none where every one kept is held. Whether one went.
    fn forget(&mut self, spare_only: bool) -> bool {
        let least = |spare: bool| {
            self.kept
                .iter()
                .filter(|(_, kept)| !spare || Arc::strong_count(&kept.sampler) == 1)
                .min_by_key(|(_, kept)| kept.read_at)
                .map(|(desc, _)| *desc)
        };
        let desc = match least(true) {
            Some(desc) => desc,
            None if spare_only => return false,
            None => match least(false) {
                Some(desc) => desc,
                None => return false,
            },
        };
        self.kept.remove(&desc);
        #[cfg(test)]
        {
            self.counts.forgot += 1;
        }
        true
    }

    /// Keep `sampler`, the one this bind made of `desc`.
    pub(super) fn keep(&mut self, desc: metal2vulkan::vk::SamplerDesc, sampler: &Arc<T>) {
        #[cfg(test)]
        {
            self.counts.made += 1;
        }
        self.kept.insert(
            desc,
            Kept {
                sampler: Arc::clone(sampler),
                read_at: self.binds,
            },
        );
    }

    /// The samplers kept now.
    #[cfg(test)]
    pub(super) fn samplers_kept(&self) -> usize {
        self.kept.len()
    }

    /// What the cache has done ([`KeptCounts`]).
    #[cfg(test)]
    pub(super) fn counts(&self) -> KeptCounts {
        self.counts
    }
}

/// Metal's largest `maxAnisotropy`.
const MAX_ANISOTROPY: u32 = 16;

/// The anisotropy plain Metal samples a `maxAnisotropy` of 17 to 31 with (at
/// footprints of 32, 64 and 256 to 1 each reads the level an anisotropy of 32 reads, exactly).
const PAST_RANGE_ANISOTROPY: u32 = 32;

/// The anisotropy plain Metal samples a `maxAnisotropy` of `value` with: from 1 to 16, `value`
/// rounded up to a power of two (3 as 4, 5 to 7 as 8, 9 to 15 as 16); outside, the low five bits of
/// `value` taken through a 32-bit float (rounded to nearest) and saturated to 32 bits, 0 as 1 and
/// 17 to 31 as 32, and 1 to 16 rounded up as before (the module's rules). A Metal value past
/// `u32::MAX` takes the float's saturation too, as `u32::MAX` does: a caller may saturate it.
pub(super) fn metal_anisotropy(value: u32) -> u32 {
    // Rust's `as` rounds to nearest and saturates, as the measured conversion does.
    let low = ((value as f32) as u32) & 31;
    match low {
        0 => 1,
        1..=MAX_ANISOTROPY => low.next_power_of_two(),
        _ => PAST_RANGE_ANISOTROPY,
    }
}

/// The LOD clamps plain Metal samples with for clamps `[min, max]`, a descriptor's or a bind's
/// (the module's rules): Metal raises a sample's level to `min` and then lowers it to `max`, so
/// where `min` is past `max` every sample reads `max`'s level, and magnifies or minifies as at
/// `max`: the clamps `[max, max]`, which Vulkan takes (its `minLod` may not pass its `maxLod`).
/// NaN clamps are kept, for the binding plan to refuse where they are used.
pub(super) fn metal_lod_clamps([min, max]: [f32; 2]) -> [f32; 2] {
    if min > max {
        [max, max]
    } else {
        [min, max]
    }
}

/// The border color plain Metal samples through a sampler state whose `MTLSamplerDescriptor`'s
/// `borderColor` is Metal's raw value `raw`. Metal takes any value, its validation layer
/// included, and keeps its low two bits (the descriptor reads back 0 for 4 and 3 for 7); it
/// samples 3, which names no color, as transparent black (samples, linear blends at the edge and
/// depth comparisons of eight formats, unorm, float, integer and depth, every answer border 0's).
/// `SamplerBorderColor`'s `TryFrom<u32>` takes Metal's three named values alone.
pub fn sampler_border_color(raw: u64) -> SamplerBorderColor {
    match raw & 3 {
        1 => SamplerBorderColor::OpaqueBlack,
        2 => SamplerBorderColor::OpaqueWhite,
        _ => SamplerBorderColor::TransparentBlack,
    }
}

/// The descriptor a state of `descriptor` samples with (the module's rules): `descriptor` itself
/// with plain Metal's anisotropy ([`metal_anisotropy`]) and LOD clamps ([`metal_lod_clamps`]),
/// and for a pixel-coordinate one its repeat and mirrored-repeat axes `ClampToEdge`, its `r`
/// axis's `MirrorClampToEdge` too (`r` addresses nothing such a sampler samples), mip filter
/// `NotMipmapped` and anisotropy 1, as plain Metal samples it. A pixel-coordinate
/// `MirrorClampToEdge` `s` or `t` axis keeps its mode, which Metal mirrors once by texel index and
/// then clamps.
pub(super) fn taken_descriptor(descriptor: &SamplerDescriptor) -> SamplerDescriptor {
    use SamplerAddressMode as A;
    let mut taken = *descriptor;
    taken.max_anisotropy = metal_anisotropy(descriptor.max_anisotropy);
    [taken.lod_min_clamp, taken.lod_max_clamp] =
        metal_lod_clamps([descriptor.lod_min_clamp, descriptor.lod_max_clamp]);
    if !descriptor.normalized_coordinates {
        for mode in [&mut taken.s_address_mode, &mut taken.t_address_mode] {
            if matches!(*mode, A::Repeat | A::MirrorRepeat) {
                *mode = A::ClampToEdge;
            }
        }
        // `r` addresses nothing a pixel-coordinate sampler samples (1D and 2D textures).
        if matches!(
            taken.r_address_mode,
            A::Repeat | A::MirrorRepeat | A::MirrorClampToEdge
        ) {
            taken.r_address_mode = A::ClampToEdge;
        }
        taken.mip_filter = SamplerMipFilter::NotMipmapped;
        taken.max_anisotropy = 1;
    }
    taken
}

/// A `VkSampler` the runtime made, destroyed when the last holder lets it go.
pub(super) struct OwnedSampler {
    device: Device,
    raw: vk::Sampler,
}

impl OwnedSampler {
    /// The `VkSampler`.
    pub(super) fn raw(&self) -> vk::Sampler {
        self.raw
    }
}

impl Drop for OwnedSampler {
    fn drop(&mut self) {
        // Once the process is exiting, the sampler is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command buffer uses the sampler: each one that bound it holds it
        // until it completes.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
        self.device.shared().sampler_destroyed();
    }
}

/// A Metal sampler state (`MTLSamplerState`). Cheap to clone (a retain); `Send + Sync`. Two
/// handles are equal when they are the same state.
#[derive(Clone)]
pub struct SamplerState(Arc<SamplerShared>);

pub(super) struct SamplerShared {
    device: Device,
    /// The descriptor the state samples with ([`taken_descriptor`]).
    descriptor: SamplerDescriptor,
    /// The state's `gpuResourceID`.
    id: IdHold,
    /// The state's element of the set-1 sampler arrays (`resource_ids.rs`).
    element: ElementHold,
    /// The Vulkan samplers made for binds, by their description ([`KeptSamplers`]).
    kept: Mutex<KeptSamplers<OwnedSampler>>,
    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`).
    held_by: HeldBy,
}

impl Drop for SamplerShared {
    /// A sampler state going: the device's sampler tables no longer map its ID to it where it
    /// held the ID's slot, and its elements of the heap sets' sampler arrays wait for the command
    /// buffers that may still read them (`heap_sets::sampler_gone`), before its element is given
    /// back.
    fn drop(&mut self) {
        let id = self.id.id().0;
        if id != 0 {
            super::heap_sets::sampler_gone(
                &self.device,
                id,
                self.id.serial(),
                self.element.element(),
            );
        }
    }
}

impl std::fmt::Debug for SamplerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerState")
            .field("descriptor", &self.0.descriptor)
            .finish()
    }
}

impl PartialEq for SamplerState {
    fn eq(&self, other: &SamplerState) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SamplerState {}

impl std::hash::Hash for SamplerState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Device {
    /// `newSamplerStateWithDescriptor:`. A request equal to a live state's (every field of its
    /// descriptor, and its options but the placement) gets that state, as Metal's does.
    ///
    /// A descriptor Metal's validation stops, of a `max_anisotropy` outside 1 to 16 or of pixel
    /// coordinates, is made, and samples as plain Metal samples it (the module's rules); an
    /// anisotropy past the largest of a device with `samplerAnisotropy` filters with that
    /// largest, announced once (`Degradation::CappedAnisotropy`). An anisotropic state without a
    /// mip filter whose two filters differ, on a device that samples through Metal's own sampler
    /// (MoltenVK), filters another footprint than Metal's, announced once
    /// (`Degradation::NotMipmappedAnisotropicSampler`). Anisotropy on a device without it is made,
    /// and samples as the module's docs say. Refused: what the device cannot do on
    /// any use of the state (`Unsupported`, naming the feature: a mirror-clamp address mode, a
    /// pixel-coordinate one's included). A minimum or maximum reduction reads the weighted
    /// average.
    pub fn new_sampler_state(&self, descriptor: &SamplerDescriptor) -> Result<SamplerState, Error> {
        self.sampler_state(
            "new_sampler_state",
            descriptor,
            SamplerStateOptions::default(),
        )
    }

    /// `newSamplerStateWithDescriptor:` with the rest of Metal's descriptor in `options`. Every
    /// state has a [`gpu_resource_id`](SamplerState::gpu_resource_id), which an argument buffer
    /// holds to name it, and every function an encoder runs can name it so while it lives,
    /// whenever the state was made (`heap_sets.rs`). A state no set-1 sampler array of the device
    /// can hold (more states live than it holds), or at an ID neither window of the device's
    /// sampler tables reaches, is made, and announced: a function reading its ID reads the
    /// neutral sampler (`resource_ids.rs`). Refused as
    /// [`new_sampler_state`](Self::new_sampler_state) is, and a placement at an ID of 16381 or
    /// more (`Invalid`, `Problem::ResourceId`), which Metal ends the process for.
    pub fn new_sampler_state_with_options(
        &self,
        descriptor: &SamplerDescriptor,
        options: SamplerStateOptions,
    ) -> Result<SamplerState, Error> {
        self.sampler_state("new_sampler_state_with_options", descriptor, options)
    }

    fn sampler_state(
        &self,
        call: &'static str,
        descriptor: &SamplerDescriptor,
        options: SamplerStateOptions,
    ) -> Result<SamplerState, Error> {
        let taken = taken_descriptor(descriptor);
        // The LOD clamps are checked where they are used: a bind's clamps replace them.
        let mut shape_of = taken;
        shape_of.lod_min_clamp = 0.0;
        shape_of.lod_max_clamp = f32::MAX;
        let profile = self.compile_device().profile();
        check_sampler(&shape_of.shape(), SamplerUse::default(), profile).map_err(|error| {
            match error {
                metal2vulkan::error::Error::Unsupported(unsupported) => {
                    Error::Unsupported(unsupported)
                }
                other => super::error::internal(&format!("a sampler check refused {other:?}")),
            }
        })?;
        self.check_sampler_placement(call, options.placement)?;
        // Metal samples an anisotropy past 16 with 32 (`metal_anisotropy`): a device whose
        // samplers take less filters with its largest (`bind::sampler_desc` bounds it so). A
        // device without `samplerAnisotropy` filters with none, Metal's filter emulated where the
        // function reaches the sample and the isotropic one recorded elsewhere.
        let largest = f32::from_bits(profile.limits().max_sampler_anisotropy_bits);
        let anisotropy = (profile.enabled().features).contains(&VkFeature::SamplerAnisotropy);
        if anisotropy && taken.max_anisotropy as f32 > largest {
            self.degrade(Degradation::CappedAnisotropy, || {
                (
                    format!(
                        "a sampler state of maxAnisotropy {}",
                        descriptor.max_anisotropy
                    ),
                    format!(
                        "plain Metal samples it with an anisotropy of {}, and the device's \
                         samplers take at most {largest} (maxSamplerAnisotropy)",
                        taken.max_anisotropy
                    ),
                )
            });
        }
        // Metal's own sampler handed anisotropy reads no level clamp of a sampler without a mip
        // filter whose filters differ as Metal's `NotMipmapped` (`metal::sampler::vk_sampler`).
        // A pixel-coordinate state has no anisotropy (`taken_descriptor`).
        if anisotropy
            && taken.max_anisotropy > 1
            && taken.mip_filter == SamplerMipFilter::NotMipmapped
            && taken.min_filter != taken.mag_filter
            && profile.takes_driver_deviation(DriverDeviation::MetalSampler)
        {
            self.degrade(Degradation::NotMipmappedAnisotropicSampler, || {
                (
                    format!(
                        "a sampler state of maxAnisotropy {} without a mip filter, minification \
                         {:?} and magnification {:?}",
                        descriptor.max_anisotropy, taken.min_filter, taken.mag_filter
                    ),
                    "the device samples through Metal's own sampler, whose anisotropic filter \
                     reads no level clamp as Metal's NotMipmapped where the filters differ"
                        .to_owned(),
                )
            });
        }
        let shared = self.shared();
        let key = SamplerKey::of(descriptor, options);
        // Under the map's lock, so equal requests on two threads get one state.
        let mut live = lock(&shared.samplers);
        if let Some(state) = live.live(&key) {
            return Ok(SamplerState(state));
        }
        let id = IdHold::take(&shared.resource_ids, IdSpace::Sampler, options.placement);
        let element = ElementHold::take(&shared.resource_ids);
        let state = SamplerState(Arc::new(SamplerShared {
            device: self.clone(),
            descriptor: taken,
            id,
            element,
            kept: Mutex::new(KeptSamplers::new()),
            held_by: HeldBy::default(),
        }));
        live.keep(key, &state.0);
        let hold = &state.0.id;
        // Sampler slots have no readers: the device counts the command buffers that list them.
        hold.hold(Holder::Sampler(Arc::downgrade(&state.0)));
        drop(live);
        // Every heap set holds it, and the sampler tables map its ID to it (`heap_sets.rs`).
        super::heap_sets::place_sampler(self, &state);
        let id = hold.id().0;
        let element = state.heap_element();
        let (available, limit) = self.sampler_array();
        if element >= available {
            self.degrade(Degradation::UnlistedArgumentSampler, || {
                (
                    format!("sampler state at resource ID {id}"),
                    format!(
                        "more sampler states live than a set-1 sampler array of the device \
                         holds ({limit:?} is {available}), so none holds its element {element}, \
                         and a function reading it through an argument buffer while it lives \
                         reads the neutral sampler"
                    ),
                )
            });
        }
        Ok(state)
    }

    /// The live sampler states, in resource-ID order, each the holder of its ID
    /// (`resource_ids.rs`): what an argument buffer can name (`BindState::argument_samplers`).
    pub(super) fn argument_samplers(&self) -> Vec<SamplerState> {
        self.argument_samplers_at().1
    }

    /// [`argument_samplers`](Self::argument_samplers), with the count of the sampler IDs' slots'
    /// changes they are live at ([`sampler_slots_changed`](Self::sampler_slots_changed)).
    pub(super) fn argument_samplers_at(&self) -> (u64, Vec<SamplerState>) {
        let (changed, states) = lock(&self.shared().resource_ids).samplers();
        (changed, states.into_iter().map(SamplerState).collect())
    }

    /// [`argument_samplers_at`](Self::argument_samplers_at), for a draw's listing: counted
    /// (`WorkCounts::argument_samplers_taken`).
    fn argument_samplers_for_draw(&self) -> (u64, Vec<SamplerState>) {
        self.shared().counts.argument_samplers_taken();
        self.argument_samplers_at()
    }

    /// The count of the changes of the sampler IDs' slots so far: while it holds, the live
    /// sampler states are the ones [`argument_samplers`](Self::argument_samplers) answered at it.
    pub(super) fn sampler_slots_changed(&self) -> u64 {
        self.shared()
            .sampler_ids
            .slots_changed
            .load(Ordering::Acquire)
    }
}

/// The device's live sampler states a draw lists (`BindState::argument_samplers`), each at the
/// heap element of its resource ID, taken from the device only where the draw needs them
/// (`RenderCommandEncoder::settle_draw`): a kept bind state listed at the count of the sampler
/// IDs' slots' changes that still holds lists them already (`LiveBind::listed_at`), and a draw
/// that writes no heap element and names no state in a notice takes none.
pub(super) struct ArgumentSamplers {
    /// The states and the count of slot changes they were live at, once taken.
    taken: Option<(u64, Vec<SamplerState>)>,
    /// Those the draw's variant's set-1 sampler arrays leave out were dropped
    /// (`heap_sets::fit_samplers`): the states are no longer every live one.
    trimmed: bool,
}

impl ArgumentSamplers {
    /// None taken yet.
    pub(super) fn untaken() -> ArgumentSamplers {
        ArgumentSamplers {
            taken: None,
            trimmed: false,
        }
    }

    /// `device`'s, taken now.
    pub(super) fn taken(device: &Device) -> ArgumentSamplers {
        ArgumentSamplers {
            taken: Some(device.argument_samplers_for_draw()),
            trimmed: false,
        }
    }

    /// The states, taken from `device` now where none were yet.
    pub(super) fn take(&mut self, device: &Device) -> &mut Vec<SamplerState> {
        let (_, states) = self
            .taken
            .get_or_insert_with(|| device.argument_samplers_for_draw());
        states
    }

    /// The states, where they were taken.
    pub(super) fn states(&self) -> Option<&[SamplerState]> {
        self.taken.as_ref().map(|(_, states)| &states[..])
    }

    /// The count of slot changes every live state is listed at: where the states were taken
    /// and not trimmed.
    pub(super) fn listed_at(&self) -> Option<u64> {
        match (&self.taken, self.trimmed) {
            (Some((changed, _)), false) => Some(*changed),
            _ => None,
        }
    }

    /// Keep those of the taken states that `set`, the variant's set 1, holds
    /// (`heap_sets::fit_samplers`, `unlisted` saying so): whether every one fits.
    pub(super) fn fit(
        &mut self,
        set: Option<&metal2vulkan::pipeline::DescriptorSetLayout>,
        unlisted: &mut Option<(u64, u64)>,
    ) -> bool {
        let Some((_, states)) = &mut self.taken else {
            return set.is_none();
        };
        let fits = super::heap_sets::fit_samplers(set, states, unlisted);
        self.trimmed |= !fits;
        fits
    }
}

impl SamplerState {
    /// Another reference to the sampler state, which a command buffer holds it by
    /// (`EncoderCore::retain_shared`).
    pub(super) fn held(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`).
    pub(super) fn held_by(&self) -> &HeldBy {
        &self.0.held_by
    }

    /// `device`: the device that made the state.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `gpuResourceID`: what an argument buffer holds to name the state (`resource_ids.rs`).
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.0.id.id()
    }

    /// The state's element of the set-1 sampler arrays, which a sampler table maps its ID to
    /// (`resource_ids.rs`).
    pub(super) fn heap_element(&self) -> u64 {
        self.0.element.element()
    }

    /// The serial of its hold of its ID (`IdHold::serial`): what the ID's slot names it by while
    /// it holds it.
    pub(super) fn id_serial(&self) -> u64 {
        self.0.id.serial()
    }

    /// The state, held weakly.
    pub(super) fn downgrade(&self) -> Weak<SamplerShared> {
        Arc::downgrade(&self.0)
    }

    /// The state `weak` holds, where it lives.
    pub(super) fn upgrade(weak: &Weak<SamplerShared>) -> Option<SamplerState> {
        weak.upgrade().map(SamplerState)
    }

    /// The descriptor the state samples with: its request's as plain Metal takes it
    /// ([`taken_descriptor`]).
    pub(super) fn descriptor(&self) -> &SamplerDescriptor {
        &self.0.descriptor
    }

    /// Whether an access through an argument buffer that names this state may hand the device a
    /// pixel coordinate scaled by the texture's size (`Device::degrade_argument_pixel_samplers`):
    /// its coordinates are pixels. Every gather, comparison and sample through such a state but a
    /// sample its flagged fetch answers (nearest, clamped to the edge) reads the scaled coordinate.
    pub(super) fn scales_in_an_argument_buffer(&self) -> bool {
        !self.0.descriptor.normalized_coordinates
    }

    /// The Vulkan sampler of `desc` (one of this state's binds, its view mapping included): the
    /// one the state keeps, else a new one, kept once the cache has room for it ([`KeptSamplers`],
    /// at most [`KEPT`] and at most as many as the device allows live). Where the device already
    /// holds as many samplers as it allows, one this state keeps that no command buffer holds goes
    /// to make room; `Error::SamplerLimit` is the answer only where every one it keeps is held
    /// ([`SamplerCount`]).
    pub(super) fn vk_sampler(
        &self,
        desc: &metal2vulkan::vk::SamplerDesc,
    ) -> Result<Arc<OwnedSampler>, Error> {
        let device = &self.0.device;
        let mut kept = lock(&self.0.kept);
        if let Some(sampler) = kept.found(desc) {
            return Ok(sampler);
        }
        kept.make_room(KEPT.min(device.shared().sampler_count.limit() as usize));
        let mut reduction = vk::SamplerReductionModeCreateInfo::default();
        let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
        let info =
            metal2vulkan::ash_interop::sampler_create_info(desc, &mut reduction, &mut border);
        let raw = loop {
            // SAFETY: the create info is valid for the device: `check_sampler` admitted its
            // features, its LOD range is ordered (`bind::sampler_desc`), and a border mapping is
            // chained only where the device has `borderColorSwizzle`
            // (`metal::sampler::opaque_black_border`).
            match unsafe { device.shared().create_sampler(&info) } {
                Ok(raw) => break raw,
                Err(limit @ Error::SamplerLimit(_)) if !kept.forget_spare() => return Err(limit),
                Err(Error::SamplerLimit(_)) => {}
                Err(error) => return Err(error),
            }
        };
        let sampler = Arc::new(OwnedSampler {
            device: device.clone(),
            raw,
        });
        kept.keep(*desc, &sampler);
        Ok(sampler)
    }

    /// What its cache of Vulkan samplers has done ([`KeptCounts`]), and how many it keeps now: the
    /// device tests that measure the cache read them.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn kept_counts(&self) -> (KeptCounts, usize) {
        let kept = lock(&self.0.kept);
        (kept.counts(), kept.samplers_kept())
    }
}
