//! The runtime's own kernels (`metal2vulkan::pipeline::OwnKernel`): each made once a device, at its
//! first use, from the translator's module for the device's profile, through the device's one
//! pipeline cache and its pipeline layout (counted as a pipeline made, in
//! `Metrics::pipelines_created` and the `pipelines` gauge), and destroyed with the device.

use crate::counters::Gauge;
use crate::device::{lock, Core};
use crate::error::Error;
use crate::pipeline::made_through_cache;
use ash::vk;
use metal2vulkan::pipeline::{own_kernel, OwnKernel};
use std::sync::Mutex;

/// A device's own kernels made so far: each with its module and its pipeline.
#[derive(Default)]
pub(crate) struct OwnKernels {
    made: Mutex<Vec<(OwnKernel, vk::ShaderModule, vk::Pipeline)>>,
}

impl OwnKernels {
    /// `kernel`'s pipeline on `core`'s device, made now where it is not yet: the calling thread
    /// waits for it, a first-use wait, as another that asks meanwhile does.
    ///
    /// Err: [`Error::Translation`] where the translator makes no module of it for the device's
    /// profile; [`Error::Vulkan`] where the module or the pipeline cannot be made.
    pub(crate) fn pipeline(&self, core: &Core, kernel: OwnKernel) -> Result<vk::Pipeline, Error> {
        let mut made = lock(&self.made);
        if let Some(&(_, _, pipeline)) = made.iter().find(|(held, ..)| *held == kernel) {
            return Ok(pipeline);
        }
        let words = own_kernel(kernel, core.translator.profile()).map_err(Error::translation)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: the words are the translator's SPIR-V for the device's profile.
        let module =
            unsafe { core.device.create_shader_module(&info, None) }.map_err(|result| {
                Error::Vulkan {
                    call: "vkCreateShaderModule",
                    result,
                }
            })?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(core.layout);
        let pipeline = made_through_cache(core, "vkCreateComputePipelines", |cache| {
            // SAFETY: the module, the layout and the cache are the device's.
            unsafe { core.device.create_compute_pipelines(cache, &[info], None) }
        });
        match pipeline {
            Ok(pipeline) => {
                made.push((kernel, module, pipeline));
                Ok(pipeline)
            }
            Err(error) => {
                // SAFETY: no pipeline was made of the module.
                unsafe { core.device.destroy_shader_module(module, None) };
                Err(error)
            }
        }
    }

    /// Destroys every kernel made.
    ///
    /// # Safety
    ///
    /// No command buffer that used one is pending, and none is used again.
    pub(crate) unsafe fn destroy(&self, core: &Core) {
        for (_, module, pipeline) in lock(&self.made).drain(..) {
            // SAFETY: the caller's contract.
            unsafe {
                core.device.destroy_pipeline(pipeline, None);
                core.device.destroy_shader_module(module, None);
            }
            core.counters.fall(Gauge::Pipelines);
        }
    }
}
