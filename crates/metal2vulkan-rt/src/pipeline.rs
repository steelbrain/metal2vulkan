//! Compute pipeline states: a kernel's interface, which the translator states, and its
//! `VkPipeline`s, one per threadgroup size and threadgroup memory lengths, each made by the
//! encoding call that first needs it.

use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::hazard::Access;
use crate::reaper::{Free, Job};
use ash::vk;
use std::sync::{Arc, Mutex};

/// The push window: the parameters a kernel reads from push constants, `[0, 128)` on every device,
/// as 128 bytes is what Vulkan guarantees. Fields past it go to a block in the parameter ring,
/// whose address is a field in the window.
pub(crate) const PUSH_BYTES: u32 = 128;

/// `MTLComputePipelineState`.
///
/// A cheap handle: clones share one state. A command buffer that dispatches with it keeps it until
/// it retires.
#[derive(Clone)]
pub struct ComputePipelineState {
    pub(crate) inner: Arc<PipelineInner>,
}

pub(crate) struct PipelineInner {
    core: Arc<Core>,
    module: vk::ShaderModule,
    pub(crate) interface: Interface,
    /// The `VkPipeline`s made so far, by their specialization.
    variants: Mutex<Vec<(Variant, vk::Pipeline)>>,
}

/// What the runtime needs of a kernel, from the translator.
pub(crate) struct Interface {
    /// Each buffer slot the kernel uses, with its access.
    pub(crate) buffers: Vec<(u32, Access)>,
    /// Whether the kernel reads through an address that is not one of its bound slots.
    pub(crate) reads_unbound: bool,
    /// Whether the kernel writes through an address that is not one of its bound slots.
    pub(crate) writes_unbound: bool,
    /// Bytes of the push window the fields use.
    pub(crate) push_len: u32,
    /// The length of the ring block of the fields past the window, a multiple of 16; 0 where
    /// every field is in the window.
    pub(crate) block_len: u32,
    /// The offset in the window of the ring block's address, a `u64`, where there is a block.
    pub(crate) block_address: u32,
    /// The parameters.
    pub(crate) fields: Vec<Field>,
    /// The specialization constant ids of the threadgroup size's three dimensions.
    pub(crate) size_ids: [u32; 3],
    /// The specialization constant id of each threadgroup memory length the kernel takes, by its
    /// index.
    pub(crate) threadgroup_ids: Vec<(u32, u32)>,
    /// `maxTotalThreadsPerThreadgroup`.
    pub(crate) max_total_threads: u32,
}

/// A parameter and where it goes.
pub(crate) struct Field {
    pub(crate) place: Place,
    pub(crate) value: FieldValue,
}

/// Where a parameter goes: a byte offset of the push window, or of the ring block.
#[derive(Clone, Copy)]
pub(crate) enum Place {
    Push(u32),
    Block(u32),
}

/// What a parameter holds.
#[derive(Clone, Copy)]
pub(crate) enum FieldValue {
    /// A `u64`: the address of the buffer bound at the slot plus its offset, the address of the
    /// slot's `set_bytes` copy, or 0 where nothing is bound.
    BufferAddress(u32),
    /// The dispatch's region, 32 bytes read as two `uvec4`: the first threadgroup of the dispatch
    /// in Metal's grid (0 for a whole dispatch) with the threadgroup size packed as
    /// `(x-1) | (y-1) << 10 | (z-1) << 20`, then Metal's threads per grid (exact for
    /// `dispatch_threads`, threadgroups times their size for `dispatch_threadgroups`, and zeros
    /// for an indirect dispatch, whose kernel takes the grid from its workgroup count) and 0.
    DispatchRegion,
}

/// A `VkPipeline`'s specialization: the threadgroup size and the threadgroup memory lengths in the
/// order of [`Interface::threadgroup_ids`].
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) size: [u32; 3],
    pub(crate) threadgroup_lengths: Vec<u32>,
}

impl ComputePipelineState {
    /// `maxTotalThreadsPerThreadgroup`.
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.inner.interface.max_total_threads
    }

    /// For the device tests: a state of a hand-made kernel and its interface.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn for_test(
        device: &crate::Device,
        words: &[u32],
        interface: Interface,
    ) -> Result<ComputePipelineState, Error> {
        let core = Arc::clone(device.core());
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: the words are a valid SPIR-V module.
        let module = unsafe { core.device.create_shader_module(&info, None) }
            .map_err(Error::vulkan("vkCreateShaderModule"))?;
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner {
                core,
                module,
                interface,
                variants: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Whether two handles are one state.
    pub(crate) fn is(&self, other: &ComputePipelineState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PipelineInner {
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.core
    }

    /// The `VkPipeline` of `variant`, made now where it is not yet: the calling thread waits for
    /// it, so what is recorded never depends on when a compile finishes.
    pub(crate) fn pipeline(&self, variant: &Variant) -> Result<vk::Pipeline, Error> {
        let mut variants = lock(&self.variants);
        if let Some((_, pipeline)) = variants.iter().find(|(made, _)| made == variant) {
            return Ok(*pipeline);
        }
        let pipeline = self.make(variant)?;
        variants.push((variant.clone(), pipeline));
        Ok(pipeline)
    }

    fn make(&self, variant: &Variant) -> Result<vk::Pipeline, Error> {
        let interface = &self.interface;
        let ids = interface
            .size_ids
            .iter()
            .chain(interface.threadgroup_ids.iter().map(|(_, id)| id));
        let entries: Vec<vk::SpecializationMapEntry> = ids
            .enumerate()
            .map(|(index, &id)| vk::SpecializationMapEntry {
                constant_id: id,
                offset: (index * 4) as u32,
                size: 4,
            })
            .collect();
        let values: Vec<u8> = variant
            .size
            .iter()
            .chain(&variant.threadgroup_lengths)
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&values);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.module)
            .name(c"main")
            .specialization_info(&specialization);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.core.layout);
        let device = &self.core.device;
        // SAFETY: the module and the layout are the device's, and the specialization matches the
        // kernel's constants.
        let made =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        let pipeline = made.map_err(|(_, result)| Error::Vulkan {
            call: "vkCreateComputePipelines",
            result,
        })?[0];
        self.core.counters.count(Metric::PipelinesCreated);
        self.core.counters.rise(Gauge::Pipelines);
        Ok(pipeline)
    }
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        // No command buffer that used it is left: each kept the state until it retired.
        let variants = self
            .variants
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, pipeline) in variants.drain(..) {
            self.core.reaper.give(Job::Free(Free::Pipeline(pipeline)));
        }
        self.core
            .reaper
            .give(Job::Free(Free::ShaderModule(self.module)));
    }
}
