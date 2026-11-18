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

use crate::arrays::{IndexClass, ResourceId};
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit};
use crate::reaper::{Free, Job};
use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::sampler::{vk_sampler, SamplerUse, SamplerWord};
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
/// A cheap handle: clones share one state. The state holds the device. Once its last handle is
/// gone, and with it the last command buffer that used it, its elements read the arrays' defaults
/// again and its index is free; a device sampler no other state holds goes to the reaper.
#[derive(Clone)]
pub struct SamplerState {
    inner: Arc<SamplerInner>,
}

struct SamplerInner {
    device: Device,
    descriptor: SamplerDescriptor,
    index: u32,
    word: u32,
    /// The create info of each array's element at the index, plain, comparison and integer
    /// texture; `None` where the array keeps its default.
    uses: [Option<SamplerDesc>; 3],
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        let core = self.device.core();
        // SAFETY: the state owns its elements until it vacates the index below, and no work that
        // reads them is pending: each command buffer that used the state held it until it
        // retired.
        unsafe {
            core.arrays
                .write_samplers(&core.device, self.index, &every([None; 3]))
        };
        lock(&core.arrays.grants).vacate(IndexClass::Sampler, self.index);
        core.counters.fall(Gauge::DescriptorSlots);
        core.counters.fall(Gauge::Samplers);
        give_back(core, &self.uses);
    }
}

/// Each sampler array's element at an index, plain, comparison and integer texture, from
/// `samplers`.
fn every(samplers: [Option<vk::Sampler>; 3]) -> [(usize, Option<vk::Sampler>); 3] {
    [0, 1, 2].map(|array| (array, samplers[array]))
}

/// A constexpr sampler at an index of the sampler arrays: its state words, the create info of
/// each array's element it holds there (plain, comparison, integer texture), and how many
/// pipeline states number it there.
pub(crate) struct Constexpr {
    words: [u64; 2],
    uses: [Option<SamplerDesc>; 3],
    holders: u32,
}

/// One more pipeline state numbers the constexpr sampler of state words `words` at `index`, with
/// the create info of each array's element it uses there (`None` for an array it does not use).
/// The first holder occupies the index and writes its elements; a later one of the same words
/// writes the elements of the arrays it adds and leaves the others, which pending work may read.
///
/// Err: [`Refusal::IndicesNotGranted`](crate::Refusal::IndicesNotGranted);
/// [`Refusal::IndexInUse`](crate::Refusal::IndexInUse) where a sampler state or another
/// constexpr sampler lives at the index; [`Limit::DeviceSamplers`]; [`Error::Vulkan`].
pub(crate) fn hold_constexpr(
    core: &Core,
    index: u32,
    words: [u64; 2],
    uses: [Option<SamplerDesc>; 3],
) -> Result<(), Error> {
    let mut held = lock(&core.constexprs);
    let present = match held.get(&index) {
        Some(entry) if entry.words == words => Some(entry.uses),
        Some(_) => {
            return Err(Error::Refused(crate::Refusal::IndexInUse {
                class: IndexClass::Sampler,
                index,
            }))
        }
        None => None,
    };
    if present.is_none() {
        lock(&core.arrays.grants).occupy(IndexClass::Sampler, index)?;
    }
    let mut added: Vec<(usize, SamplerDesc, vk::Sampler)> = Vec::new();
    for (array, desc) in uses.iter().enumerate() {
        let Some(desc) = desc else { continue };
        if present.is_some_and(|present| present[array].is_some()) {
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
                if present.is_none() {
                    lock(&core.arrays.grants).vacate(IndexClass::Sampler, index);
                }
                return Err(error);
            }
        }
    }
    let elements: Vec<(usize, Option<vk::Sampler>)> = added
        .iter()
        .map(|&(array, _, sampler)| (array, Some(sampler)))
        .collect();
    // SAFETY: the index is occupied for the constexpr sampler, no work reads an element it had
    // not written (no state numbering it there uses that array), and each sampler is held while
    // its element holds it.
    unsafe { core.arrays.write_samplers(&core.device, index, &elements) };
    let entry = held.entry(index).or_insert_with(|| {
        core.counters.rise(Gauge::DescriptorSlots);
        Constexpr {
            words,
            uses: [None; 3],
            holders: 0,
        }
    });
    for (array, desc, _) in added {
        entry.uses[array] = Some(desc);
    }
    entry.holders += 1;
    Ok(())
}

/// One pipeline state fewer numbers the constexpr sampler at `index`: the last one's going writes
/// the arrays' defaults there, frees the index and gives back its samplers.
pub(crate) fn release_constexpr(core: &Core, index: u32) {
    let mut held = lock(&core.constexprs);
    let Some(entry) = held.get_mut(&index) else {
        return;
    };
    entry.holders -= 1;
    if entry.holders > 0 {
        return;
    }
    let Some(entry) = held.remove(&index) else {
        return;
    };
    // SAFETY: no pending work reads the elements: each command buffer that used a state
    // numbering it held the state until it retired.
    unsafe {
        core.arrays
            .write_samplers(&core.device, index, &every([None; 3]))
    };
    lock(&core.arrays.grants).vacate(IndexClass::Sampler, index);
    core.counters.fall(Gauge::DescriptorSlots);
    drop(held);
    give_back(core, &entry.uses);
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
    /// `newSamplerStateWithDescriptor:`, at `index` of a live [`IndexClass::Sampler`] grant: the
    /// translator's device sampler of each use the state can be used in
    /// (`metal::sampler::vk_sampler`), taken from the device's samplers, which share one
    /// `VkSampler` per create info, and written into the three sampler arrays at `index` before
    /// the call returns. The comparison array takes the state's sampler where its compare
    /// function is not `Never`, and keeps its default otherwise.
    ///
    /// Err: [`Refusal::IndicesNotGranted`](crate::Refusal::IndicesNotGranted) and
    /// [`Refusal::IndexInUse`](crate::Refusal::IndexInUse); [`Gap::CubicSampler`];
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
        lock(&core.arrays.grants).occupy(IndexClass::Sampler, index)?;
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
                    lock(&core.arrays.grants).vacate(IndexClass::Sampler, index);
                    return Err(error);
                }
            }
        }
        core.counters.rise(Gauge::DescriptorSlots);
        core.counters.rise(Gauge::Samplers);
        // SAFETY: the index is occupied for the state, and each sampler is live while the state
        // holds it.
        unsafe {
            core.arrays
                .write_samplers(&core.device, index, &every(samplers))
        };
        Ok(SamplerState {
            inner: Arc::new(SamplerInner {
                device: self.clone(),
                descriptor: *descriptor,
                index,
                word: SamplerWord::of(descriptor).0,
                uses,
            }),
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
    /// Whether two handles are one state.
    pub(crate) fn is(&self, other: &SamplerState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.inner.device.core(), core)
    }

    /// `gpuResourceID`: its index and the translator's `SamplerWord` of its descriptor.
    pub fn gpu_resource_id(&self) -> ResourceId {
        ResourceId {
            index: self.inner.index,
            word: self.inner.word,
        }
    }

    /// The descriptor it was made with.
    pub fn descriptor(&self) -> &SamplerDescriptor {
        &self.inner.descriptor
    }
}
