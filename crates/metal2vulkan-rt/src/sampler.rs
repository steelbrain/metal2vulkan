//! Sampler states: device samplers shared by create info, each state written into set 1's three
//! sampler arrays at the index the consumer names.
//!
//! A sampler state takes the device sampler of each use it can be used in: the plain one always,
//! the comparison one where its descriptor sets a compare function, and the integer-texture one,
//! nearest with the integer border colour. Equal create infos share one `VkSampler` across uses
//! and states, and a border colour no address mode reads is not part of a create info, so a
//! state's integer-texture use takes its plain sampler wherever the two read alike. The device's
//! `maxSamplerAllocationCount` bounds the samplers, the arrays' two defaults among them; past it a
//! state is refused. Nothing is made at bind time.

use crate::arrays::{Owner, ResourceId};
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit};
use crate::reaper::{Free, Job};
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::sampler::{lod_clamps_read_alike, vk_sampler, SamplerUse, SamplerWord};
use metal2vulkan::metal::{CompareFunction, SamplerMinMagFilter};
use metal2vulkan::vk::{BorderColor, SamplerAddressMode, SamplerDesc};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

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

/// The LOD clamps a sampler state is bound with (Metal's `lodMinClamp:lodMaxClamp:` bind forms):
/// the state is bound as it is where its samples read alike with its clamps replaced by these
/// ([`lod_clamps_read_alike`]: each bound held to the levels a sample can read is the state's
/// own). `None` binds no state and reads no clamps.
///
/// Err: [`Gap::LodClamps`] where they read otherwise.
pub(crate) fn clamps_at_bind(
    sampler: Option<&SamplerState>,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
) -> Result<(), Error> {
    match sampler {
        Some(sampler)
            if !lod_clamps_read_alike(sampler.descriptor(), lod_min_clamp, lod_max_clamp) =>
        {
            Err(Error::Unsupported(Gap::LodClamps))
        }
        _ => Ok(()),
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
    device: Device,
}

/// The state itself: what a handle and a command buffer that uses it hold.
pub(crate) struct SamplerInner {
    core: Arc<Core>,
    descriptor: SamplerDescriptor,
    index: u32,
    /// Its owner word at the index.
    owner: Owner,
    word: u32,
    /// The create info of each array's element at the index, plain, comparison and integer
    /// texture; `None` where the array keeps its default.
    uses: [Option<SamplerDesc>; 3],
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
        // SAFETY: no work that reads its elements is pending: each command buffer that used the
        // state held it until it retired.
        unsafe {
            (core.arrays).vacate_samplers(&core.device, &core.counters, self.index, self.owner)
        };
        core.counters.fall(Gauge::Samplers);
        give_back(core, &self.uses);
    }
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
        core.reaper.give(Job::Free(Free::Sampler(sampler)));
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
        let profile = core.translator.profile();
        let desc = |compare, integer| {
            vk_sampler(descriptor, SamplerUse::new(compare, integer), profile)
                .map(canonical)
                .map_err(Error::Translation)
        };
        let compare = descriptor.compare_function != CompareFunction::Never;
        let uses = [
            Some(desc(false, false)?),
            match compare {
                true => Some(desc(true, false)?),
                false => None,
            },
            Some(desc(false, true)?),
        ];
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
        // SAFETY: each sampler is live while the state holds it.
        let placed = unsafe {
            (core.arrays).place_samplers(&core.device, &core.counters, index, &every(samplers))
        };
        let owner = match placed {
            Ok(owner) => owner,
            Err(error) => {
                give_back(core, &taken);
                return Err(error);
            }
        };
        core.counters.rise(Gauge::Samplers);
        Ok(SamplerState {
            inner: Arc::new(SamplerInner {
                core: Arc::clone(core),
                descriptor: *descriptor,
                index,
                owner,
                word: SamplerWord::of(descriptor).0,
                uses,
            }),
            device: self.clone(),
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
