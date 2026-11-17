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
                .write_samplers(&core.device, self.index, [None; 3])
        };
        lock(&core.arrays.grants).vacate(IndexClass::Sampler, self.index);
        core.counters.fall(Gauge::DescriptorSlots);
        core.counters.fall(Gauge::Samplers);
        give_back(core, &self.uses);
    }
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
        unsafe { core.arrays.write_samplers(&core.device, index, samplers) };
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
