//! Sampler states: device samplers shared by create info, each state written into set 1's three
//! sampler arrays at the index the consumer names.
//!
//! A sampler state takes the device sampler of each use it can be used in: the plain one always,
//! the comparison one where its descriptor sets a compare function, and the integer-texture one,
//! nearest with the integer border colour. Equal create infos share one `VkSampler` across uses
//! and states, and a border colour no address mode reads is not part of a create info, so a
//! state's integer-texture use takes its plain sampler wherever the two read alike. The device's
//! `maxSamplerAllocationCount` bounds the samplers, the arrays' two defaults among them; past it a
//! state is refused.
//!
//! A state bound with LOD clamps of its own (Metal's `lodMinClamp:lodMaxClamp:` bind forms) samples
//! as Metal answers on an Apple M3 Max (conformance cases `sampler-bind-lod-clamps-kernel` and
//! `-fragment`): the bind's pair replaces the state's clamps, a lower bound above the upper reads
//! every sample at the upper's level, as the clamps `(max, max)`, and a plain bind afterwards
//! reads the state's own clamps again. Where the pair reads alike with the state's own the state is
//! bound as it is. Otherwise the bind takes the state's variant of the pair: the state's
//! descriptor with the pair as its clamps, its device samplers from the same shared samplers, at a
//! variant element of the sampler arrays (`crate::arrays`), and its own `gpuResourceID`, which the
//! slot pushes. A state keeps its [`KEPT_VARIANTS`] most recently bound variants for its later
//! binds; a new one past them takes the place of the least recently bound one no work holds, and
//! where work holds every one, the new one is held by its command buffer alone. A NaN bound,
//! which Metal documents no answer for, binds the state as it is.

use crate::arrays::{Owner, ResourceId};
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit};
use crate::reaper::Free;
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::metal::sampler::{lod_clamps_read_alike, vk_sampler, SamplerUse, SamplerWord};
use metal2vulkan::metal::{CompareFunction, SamplerMinMagFilter};
use metal2vulkan::pipeline::{CombinedSampler, PrecisionNote};
use metal2vulkan::vk::{BorderColor, SamplerAddressMode, SamplerDesc};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// The variants a state keeps for its later binds: the most recently bound ones.
pub(crate) const KEPT_VARIANTS: usize = 16;

/// The device's samplers: one `VkSampler` for each create info `K`, with how many sampler-array
/// elements hold it.
pub(crate) struct SamplerTable<K = SamplerDesc> {
    entries: HashMap<K, (vk::Sampler, u32)>,
    /// The most samplers the table holds.
    cap: u32,
}

impl<K: Clone + Eq + Hash> SamplerTable<K> {
    /// A table of at most `cap` samplers.
    pub(crate) fn new(cap: u32) -> SamplerTable<K> {
        SamplerTable {
            entries: HashMap::new(),
            cap,
        }
    }

    /// One more holder of `desc`'s sampler, which `make` makes where the table has none.
    ///
    /// Err: [`Limit::DeviceSamplers`] where the table holds `cap` samplers already; what `make`
    /// answers.
    pub(crate) fn take(
        &mut self,
        desc: &K,
        make: impl FnOnce() -> Result<vk::Sampler, Error>,
    ) -> Result<vk::Sampler, Error> {
        if let Some((sampler, holders)) = self.entries.get_mut(desc) {
            *holders += 1;
            return Ok(*sampler);
        }
        if self.entries.len() as u64 >= u64::from(self.cap) {
            return Err(Error::Limit(Limit::DeviceSamplers { max: self.cap }));
        }
        let sampler = make()?;
        self.entries.insert(desc.clone(), (sampler, 1));
        Ok(sampler)
    }

    /// Sets the most samplers the table holds to `cap`, for a test: the samplers it holds, and
    /// the cap it had.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn set_cap(&mut self, cap: impl FnOnce(u32) -> u32) -> u32 {
        let had = self.cap;
        self.cap = cap(self.entries.len() as u32);
        had
    }

    /// One holder fewer of `desc`'s sampler: the sampler, for the reaper, where it was the last.
    pub(crate) fn give(&mut self, desc: &K) -> Option<vk::Sampler> {
        let (sampler, holders) = self.entries.get_mut(desc)?;
        *holders -= 1;
        let sampler = *sampler;
        match *holders {
            0 => self.entries.remove(desc).map(|_| sampler),
            _ => None,
        }
    }
}

/// The LOD clamps a bind's pair `lod_min_clamp` and `lod_max_clamp` samples with, as Metal answers
/// (the module docs): the pair, and the upper bound twice where the lower is above it. `None` for
/// a NaN bound, which binds the state as it is.
fn bound_clamps(lod_min_clamp: f32, lod_max_clamp: f32) -> Option<(f32, f32)> {
    match (lod_min_clamp, lod_max_clamp) {
        (lo, hi) if lo.is_nan() || hi.is_nan() => None,
        (lo, hi) if lo > hi => Some((hi, hi)),
        pair => Some(pair),
    }
}

/// `desc` as the device reads it: a border colour no address mode reads is the default one, so
/// create infos that differ only there are one.
pub(crate) fn canonical(mut desc: SamplerDesc) -> SamplerDesc {
    let modes = [
        desc.address_mode_u,
        desc.address_mode_v,
        desc.address_mode_w,
    ];
    if !modes.contains(&SamplerAddressMode::CLAMP_TO_BORDER) {
        desc.border_color = BorderColor::FLOAT_TRANSPARENT_BLACK;
        desc.border_components = None;
    }
    desc
}

/// The notes of the constexpr samplers of `uses`, each decided by the sampler's own constants:
/// where an address mode reads the border and the border colour is opaque black, the note a
/// sampler state of the same descriptor carries ([`SamplerState::precision_note`]). A use whose
/// sampler the translator cannot state has none: every state that numbers it is refused.
pub(crate) fn constexpr_notes<'a>(
    profile: &'a DeviceProfile,
    uses: &'a [(usize, CombinedSampler)],
) -> impl Iterator<Item = PrecisionNote> + 'a {
    uses.iter().filter_map(move |(_, sampler)| {
        let desc = sampler.constexpr_sampler(profile).ok().flatten()?;
        let modes = [
            desc.address_mode_u,
            desc.address_mode_v,
            desc.address_mode_w,
        ];
        let opaque_black = matches!(
            desc.border_color,
            BorderColor::FLOAT_OPAQUE_BLACK | BorderColor::INT_OPAQUE_BLACK
        );
        (modes.contains(&SamplerAddressMode::CLAMP_TO_BORDER) && opaque_black)
            .then_some(PrecisionNote::SwizzledOpaqueBlackBorder)
    })
}

/// `MTLSamplerState`: a sampler descriptor's device samplers at an index of the sampler arrays.
///
/// A cheap handle: clones share one state. A handle holds the state and the device; what a
/// command buffer keeps of it is the state alone, which holds no handle of the device. Once its
/// last handle is gone, and with it the last command buffer that used it, its elements read the
/// arrays' defaults again and its index is free, where it still owns the index; a device sampler
/// no other state holds goes to the reaper.
#[derive(Clone)]
pub struct SamplerState {
    inner: Arc<SamplerInner>,
    /// The handle's count of the device, which only its drop reads.
    _device: Device,
}

/// The state itself, or a variant of it: what a handle and a command buffer that uses it hold.
pub(crate) struct SamplerInner {
    core: Arc<Core>,
    descriptor: SamplerDescriptor,
    index: u32,
    word: u32,
    /// The create info of each array's element at the index, plain, comparison and integer
    /// texture; `None` where the array keeps its default.
    uses: [Option<SamplerDesc>; 3],
    kind: Kind,
}

/// What a [`SamplerInner`] is.
enum Kind {
    /// A sampler state, at the consumer's index: its owner word there, and the variants it keeps
    /// ([`KEPT_VARIANTS`] at most), each by its clamps' bits, the least recently bound first.
    State {
        owner: Owner,
        variants: Mutex<Vec<(u64, Arc<SamplerInner>)>>,
    },
    /// A variant of a state, bound with LOD clamps of its own, at a variant element. No handle
    /// and no resource ID of the program's names it: work reaches it only through a slot that
    /// binds it, whose command buffer holds it until it retires.
    Variant,
}

impl SamplerInner {
    /// [`SamplerState::gpu_resource_id`], of the state itself.
    pub(crate) fn gpu_resource_id(&self) -> ResourceId {
        ResourceId {
            index: self.index,
            word: self.word,
        }
    }
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        let core = &self.core;
        match self.kind {
            Kind::State { owner, .. } => {
                (core.pool).release(core, Free::SamplerIndex(self.index, owner));
                core.counters.fall(Gauge::Samplers);
            }
            Kind::Variant => {
                // SAFETY: no work in flight reads the variant's elements: each command buffer
                // that bound it held it until it retired, and no resource ID names it.
                unsafe { (core.arrays).vacate_variant(&core.device, &core.counters, self.index) };
                core.counters.fall(Gauge::LodClampVariants);
            }
        }
        give_back(core, &self.uses);
    }
}

impl SamplerInner {
    /// The variant of a state whose descriptor with a bind's clamps is `descriptor`: its device
    /// samplers taken from the device's samplers, its elements at a variant element.
    ///
    /// Err: as [`uses_of`] and [`take_samplers`]; [`Limit::LodClampVariants`] where every
    /// variant element is held.
    fn variant(core: &Arc<Core>, descriptor: &SamplerDescriptor) -> Result<SamplerInner, Error> {
        let uses = uses_of(core, descriptor)?;
        let samplers = take_samplers(core, &uses)?;
        // SAFETY: each sampler is live while the variant holds it.
        let placed =
            unsafe { (core.arrays).place_variant(&core.device, &core.counters, &every(samplers)) };
        let index = match placed {
            Ok(index) => index,
            Err(error) => {
                give_back(core, &uses);
                return Err(error);
            }
        };
        core.counters.rise(Gauge::LodClampVariants);
        Ok(SamplerInner {
            core: Arc::clone(core),
            descriptor: *descriptor,
            index,
            word: SamplerWord::of(descriptor).0,
            uses,
            kind: Kind::Variant,
        })
    }
}

/// The translator's create info of each use a state of `descriptor` can be used in, plain,
/// comparison and integer texture (`metal::sampler::vk_sampler`): the comparison one where its
/// compare function is not `Never`, `None` otherwise.
///
/// Err: [`Error::Translation`] where the translator makes no device sampler of the descriptor.
fn uses_of(core: &Core, descriptor: &SamplerDescriptor) -> Result<[Option<SamplerDesc>; 3], Error> {
    let profile = core.translator.profile();
    let desc = |compare, integer| {
        vk_sampler(descriptor, SamplerUse::new(compare, integer), profile)
            .map(canonical)
            .map_err(Error::translation)
    };
    let compare = descriptor.compare_function != CompareFunction::Never;
    Ok([
        Some(desc(false, false)?),
        match compare {
            true => Some(desc(true, false)?),
            false => None,
        },
        Some(desc(false, true)?),
    ])
}

/// One more hold of the device sampler of each of `uses`, made where the device has none: each
/// array's, `None` where the use is.
///
/// Err: [`Limit::DeviceSamplers`]; [`Error::Vulkan`] where `vkCreateSampler` fails. Nothing is
/// held then.
fn take_samplers(
    core: &Core,
    uses: &[Option<SamplerDesc>; 3],
) -> Result<[Option<vk::Sampler>; 3], Error> {
    let mut taken = Vec::with_capacity(3);
    let mut samplers = [None; 3];
    for (slot, use_) in uses.iter().enumerate() {
        let Some(use_) = use_ else { continue };
        let sampler = lock(&core.samplers).take(use_, || make(core, use_));
        match sampler {
            Ok(sampler) => {
                taken.push(Some(*use_));
                samplers[slot] = Some(sampler);
            }
            Err(error) => {
                give_back(core, &taken);
                return Err(error);
            }
        }
    }
    Ok(samplers)
}

/// Each sampler array's element at an index, plain, comparison and integer texture, from
/// `samplers`.
fn every(samplers: [Option<vk::Sampler>; 3]) -> [(usize, Option<vk::Sampler>); 3] {
    [0, 1, 2].map(|array| (array, samplers[array]))
}

/// A constexpr sampler at an index of the sampler arrays, by its index and state words: the
/// create info and sampler of each array's element it holds there (plain, comparison, integer
/// texture), how many pipeline states number it there, and its owner word at the index.
pub(crate) struct Constexpr {
    uses: [Option<(SamplerDesc, vk::Sampler)>; 3],
    holders: u32,
    owner: Owner,
}

/// One more pipeline state numbers the constexpr sampler of state words `words` at `index`, with
/// the create info of each array's element it uses there (`None` for an array it does not use).
/// Each numbering is a placement, as any is
/// ([`Arrays::place_samplers`](crate::arrays::Arrays::place_samplers)): the first holder places it
/// at the index and writes its elements, taking the index from a sampler state or a constexpr
/// sampler of other words there. A later one of the same words writes the elements of the arrays
/// it adds and leaves the others, which pending work may read, where the constexpr sampler still
/// owns the index; where another object took the index since, it places the constexpr sampler
/// there again with every element it holds
/// ([`Arrays::hold_samplers`](crate::arrays::Arrays::hold_samplers)).
///
/// Err: [`Refusal::IndicesNotGranted`](crate::Refusal::IndicesNotGranted);
/// [`Limit::DeviceSamplers`]; [`Error::Vulkan`].
pub(crate) fn hold_constexpr(
    core: &Core,
    index: u32,
    words: [u64; 2],
    uses: [Option<SamplerDesc>; 3],
) -> Result<(), Error> {
    let mut held = lock(&core.constexprs);
    let present = held
        .get(&(index, words))
        .map(|entry| (entry.uses, entry.owner));
    let mut added: Vec<(usize, SamplerDesc, vk::Sampler)> = Vec::new();
    for (array, desc) in uses.iter().enumerate() {
        let Some(desc) = desc else { continue };
        if present.is_some_and(|(present, _)| present[array].is_some()) {
            continue;
        }
        // Taken in a statement of its own, so the table's lock is let go before the arms: the
        // refusal's `give_back` takes it again.
        let taken = lock(&core.samplers).take(desc, || make(core, desc));
        match taken {
            Ok(sampler) => added.push((array, *desc, sampler)),
            Err(error) => {
                let taken: Vec<Option<SamplerDesc>> =
                    added.iter().map(|&(_, desc, _)| Some(desc)).collect();
                give_back(core, &taken);
                return Err(error);
            }
        }
    }
    let elements: Vec<(usize, Option<vk::Sampler>)> = added
        .iter()
        .map(|&(array, _, sampler)| (array, Some(sampler)))
        .collect();
    let placed = match present {
        Some((present, owner)) => {
            let held = present
                .iter()
                .enumerate()
                .filter_map(|(array, used)| used.map(|(_, sampler)| (array, Some(sampler))));
            let all: Vec<(usize, Option<vk::Sampler>)> = held.chain(elements.clone()).collect();
            // SAFETY: no work reads an element of `elements` (no state numbering it there uses
            // that array), each sampler is held while its element holds it, and no work in
            // flight reads the object it takes the index from (the consumer's duty).
            unsafe {
                (core.arrays).hold_samplers(
                    &core.device,
                    &core.counters,
                    (index, owner),
                    &elements,
                    &all,
                )
            }
        }
        // SAFETY: each sampler is held while its element holds it, and no work in flight reads
        // the object it takes the index from (the consumer's duty).
        None => unsafe {
            (core.arrays).place_samplers(&core.device, &core.counters, index, &elements)
        },
    };
    let owner = match placed {
        Ok(owner) => owner,
        Err(error) => {
            let taken: Vec<Option<SamplerDesc>> =
                added.iter().map(|&(_, desc, _)| Some(desc)).collect();
            give_back(core, &taken);
            return Err(error);
        }
    };
    let entry = held.entry((index, words)).or_insert(Constexpr {
        uses: [None; 3],
        holders: 0,
        owner,
    });
    entry.owner = owner;
    for (array, desc, sampler) in added {
        entry.uses[array] = Some((desc, sampler));
    }
    entry.holders += 1;
    Ok(())
}

/// One pipeline state fewer numbers the constexpr sampler of state words `words` at `index`: the
/// last one's going gives back its samplers and, where it still owns the index, writes the arrays'
/// defaults there and frees the index.
pub(crate) fn release_constexpr(core: &Core, index: u32, words: [u64; 2]) {
    let mut held = lock(&core.constexprs);
    let Some(entry) = held.get_mut(&(index, words)) else {
        return;
    };
    entry.holders -= 1;
    if entry.holders > 0 {
        return;
    }
    let Some(entry) = held.remove(&(index, words)) else {
        return;
    };
    // SAFETY: no pending work reads the elements: each command buffer that used a state
    // numbering it held the state until it retired.
    unsafe { (core.arrays).vacate_samplers(&core.device, &core.counters, index, entry.owner) };
    drop(held);
    give_back(core, &entry.uses.map(|used| used.map(|(desc, _)| desc)));
}

/// Gives up one hold of each sampler of `uses`, the reaper freeing each one no element holds.
fn give_back(core: &Core, uses: &[Option<SamplerDesc>]) {
    let freed: Vec<vk::Sampler> = {
        let mut table = lock(&core.samplers);
        uses.iter()
            .flatten()
            .filter_map(|desc| table.give(desc))
            .collect()
    };
    for sampler in freed {
        core.pool.release(core, Free::Sampler(sampler));
    }
}

impl Device {
    /// `newSamplerStateWithDescriptor:`, at `index` of a live
    /// [`IndexClass::Sampler`](crate::IndexClass::Sampler) grant: the translator's device sampler
    /// of each use the state can be used in (`metal::sampler::vk_sampler`), taken from the
    /// device's samplers, which share one `VkSampler` per create info, and written into the three
    /// sampler arrays at `index` before the call returns. The comparison array takes the state's
    /// sampler where its compare function is not `Never`, and keeps its default otherwise. An
    /// index another state or a constexpr sampler holds, live or released and held by a command
    /// buffer not yet retired, is taken at once, as a texture's is
    /// ([`Device::new_texture`](crate::Device::new_texture)): the earlier one's going leaves it.
    ///
    /// Err: [`Refusal::IndicesNotGranted`](crate::Refusal::IndicesNotGranted);
    /// [`Gap::CubicSampler`];
    /// [`Error::Translation`] where the translator makes no device sampler of the descriptor;
    /// [`Limit::DeviceSamplers`] past the device's `maxSamplerAllocationCount`; [`Error::Vulkan`]
    /// where `vkCreateSampler` fails.
    pub fn new_sampler_state(
        &self,
        descriptor: &SamplerDescriptor,
        index: u32,
    ) -> Result<SamplerState, Error> {
        if [descriptor.min_filter, descriptor.mag_filter].contains(&SamplerMinMagFilter::Cubic) {
            return Err(Error::Unsupported(Gap::CubicSampler));
        }
        let core = self.core();
        let uses = uses_of(core, descriptor)?;
        let samplers = take_samplers(core, &uses)?;
        // SAFETY: each sampler is live while the state holds it.
        let placed = unsafe {
            (core.arrays).place_samplers(&core.device, &core.counters, index, &every(samplers))
        };
        let owner = match placed {
            Ok(owner) => owner,
            Err(error) => {
                give_back(core, &uses);
                return Err(error);
            }
        };
        core.counters.rise(Gauge::Samplers);
        Ok(SamplerState {
            inner: Arc::new(SamplerInner {
                core: Arc::clone(core),
                descriptor: *descriptor,
                index,
                word: SamplerWord::of(descriptor).0,
                uses,
                kind: Kind::State {
                    owner,
                    variants: Mutex::new(Vec::with_capacity(KEPT_VARIANTS)),
                },
            }),
            _device: self.clone(),
        })
    }
}

/// Makes the `VkSampler` of `desc`.
fn make(core: &Core, desc: &SamplerDesc) -> Result<vk::Sampler, Error> {
    let mut reduction = vk::SamplerReductionModeCreateInfo::default();
    let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
    let info = metal2vulkan::ash_interop::sampler_create_info(desc, &mut reduction, &mut border);
    // SAFETY: the translator's create info of the device's profile, whose features device
    // creation enabled.
    let sampler = unsafe { core.device.create_sampler(&info, None) }
        .map_err(Error::vulkan("vkCreateSampler"))?;
    core.counters.rise(Gauge::DeviceSamplers);
    Ok(sampler)
}

impl SamplerState {
    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.inner.core, core)
    }

    /// `gpuResourceID`: its index and the translator's `SamplerWord` of its descriptor.
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.inner.gpu_resource_id()
    }

    /// The descriptor it was made with.
    pub fn descriptor(&self) -> &SamplerDescriptor {
        &self.inner.descriptor
    }

    /// The state as it samples with the LOD clamps `lod_min_clamp` and `lod_max_clamp` (Metal's
    /// `lodMinClamp:lodMaxClamp:` bind forms, the module docs), given to `lend`: the state itself
    /// where the clamps read alike with its own ([`lod_clamps_read_alike`]) or a bound is NaN, and
    /// otherwise its variant of them, which it keeps where it can. `lend` runs while the state's
    /// variants are locked, so it takes its hold of the variant before any later bind can let it
    /// go.
    ///
    /// Err, binding nothing: as [`Device::new_sampler_state`]'s device samplers;
    /// [`Limit::LodClampVariants`] where every variant element is held.
    pub(crate) fn with_lod_clamps<R>(
        &self,
        lod_min_clamp: f32,
        lod_max_clamp: f32,
        lend: impl FnOnce(&Arc<SamplerInner>) -> R,
    ) -> Result<R, Error> {
        let state = &self.inner;
        let pair = bound_clamps(lod_min_clamp, lod_max_clamp);
        let (lo, hi) = match pair {
            Some((lo, hi)) if !lod_clamps_read_alike(&state.descriptor, lo, hi) => (lo, hi),
            _ => return Ok(lend(state)),
        };
        // A handle holds a state; a variant has no variants of its own.
        let Kind::State { variants, .. } = &state.kind else {
            return Ok(lend(state));
        };
        let key = u64::from(lo.to_bits()) | u64::from(hi.to_bits()) << 32;
        let mut kept = lock(variants);
        if let Some(at) = kept.iter().position(|(held, _)| *held == key) {
            let entry = kept.remove(at);
            let lent = lend(&entry.1);
            kept.push(entry);
            return Ok(lent);
        }
        // The least recently bound variant no command buffer holds gives its place, and its
        // element, to the new one.
        let evicted = match kept.len() < KEPT_VARIANTS {
            true => None,
            false => (kept.iter())
                .position(|(_, variant)| Arc::strong_count(variant) == 1)
                .map(|at| kept.remove(at)),
        };
        drop(evicted);
        let mut descriptor = state.descriptor;
        (descriptor.lod_min_clamp, descriptor.lod_max_clamp) = (lo, hi);
        let variant = Arc::new(SamplerInner::variant(&state.core, &descriptor)?);
        let lent = lend(&variant);
        if kept.len() < KEPT_VARIANTS {
            kept.push((key, variant));
        }
        Ok(lent)
    }

    /// No Metal counterpart: the static note of a sample of its border, where an address mode
    /// reads the border (`ClampToZero` or `ClampToBorderColor`) and the border colour is opaque
    /// black ([`PrecisionNote::SwizzledOpaqueBlackBorder`]: through a view whose component
    /// mapping is not the identity, on a device without `borderColorSwizzleFromImage`, the
    /// border texel is the device's); `None` for any other state. Decided by the descriptor
    /// alone, on every device: a state meets every view, so none of its device samplers names a
    /// view's mapping.
    pub fn precision_note(&self) -> Option<PrecisionNote> {
        let opaque_black = self.inner.descriptor.shape().opaque_black_border;
        opaque_black.then_some(PrecisionNote::SwizzledOpaqueBlackBorder)
    }
}

impl crate::kept::Handle for SamplerState {
    type Kept = Arc<SamplerInner>;

    fn object(&self) -> &Arc<SamplerInner> {
        &self.inner
    }

    fn address_of(object: &Arc<SamplerInner>) -> usize {
        Arc::as_ptr(object) as *const () as usize
    }
}
