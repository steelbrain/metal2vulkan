//! A multisampled stencil copied without colour copies: where maintenance8's depth-stencil/colour
//! `vkCmdCopyImage` is missing or runs wrong
//! (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), the runtime moves a multisampled
//! stencil by `bind::depth_stencil_copy`'s modules.
//!
//! - **The plane draws** ([`StencilPlanes`]) write a source's stencil into a depth/stencil
//!   target's: a render pass of the target alone, its depth loaded and stored and its stencil
//!   cleared to 0 over the render area, of eight sample-rate draws, the plane of each
//!   [`StencilBit`]: under write mask [`StencilBit::mask`], reference 0xFF, compare `ALWAYS` and
//!   pass op `REPLACE`, each draw sets its bit in the samples whose source has it, so every
//!   sample's stencil ends as the source's byte. The depth test is off, so the depth stays the
//!   target's. The source is another image's stencil aspect through a stencil-aspect sampled
//!   view ([`CopySource::Aspect`]: the separate textures' copies through their scratch image,
//!   `depth_stencil_scratch.rs`) or the shader-side stencil's shadow ([`CopySource::Shadow`]:
//!   its copy back, `shadow_stencil.rs`).
//! - **The split** ([`ShadowPlanes`], the shadow's copy in): a dispatch of
//!   `bind::stencil_split_module`, which fetches each sample of the attachment's stencil aspect
//!   through a sampled view and stores it into the shadow's multisampled `R8_UINT` storage image.
//!
//! An aspect read through a sampled view needs its image to be sampled (`SAMPLED` usage, which a
//! texture made for shader reads has, [`sampled`]), and the device to run the modules
//! (`sampleRateShading`; the shadow's, multisampled storage images); their consumers refuse,
//! typed, where either is missing. The kernel, each source's plane shaders, and each source,
//! format and sample count's render pass and pipelines are made once per device, at first use,
//! and destroyed with the device; a copy's descriptor set and framebuffer are its own, held by its
//! command buffer until it completes.

use super::binding_view::VkView;
use super::device::{device_error, lock};
use super::error::{internal, vulkan, Error};
use super::exit_guard;
use super::release;
use super::render_pass::Target;
use super::resource::ResourceKey;
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::texture::image_aspects;
use super::tracker::{Access, Use};
use super::Device;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    depth_stencil_copy_modules, stencil_split_module, CopySource, DepthStencilCopy, StencilBit,
    StencilSplitModule,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::error::Feature;
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageUsageFlags, ImageViewType,
};
use std::sync::Arc;

/// The split kernel, the device's.
#[derive(Clone, Copy, Debug)]
struct SplitKernel {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// The plane draws' shaders, reading the shadow, the device's.
#[derive(Clone, Copy, Debug)]
struct PlaneShaders {
    vertex: vk::ShaderModule,
    /// One per [`StencilBit`], in order.
    fragments: [vk::ShaderModule; 8],
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// The digest of its modules' SPIR-V words, which its pipelines' records are named by
    /// (`shader_cache/driver.rs`).
    words: Hash256,
}

/// The plane draws' render pass and pipelines for one attachment format and sample count.
#[derive(Clone, Copy, Debug)]
struct PlanePipelines {
    render_pass: vk::RenderPass,
    /// One per [`StencilBit`], in order.
    pipelines: [vk::Pipeline; 8],
}

/// The stencil copies' objects one device has made (the module docs).
#[derive(Default)]
pub(super) struct StencilCopyObjects {
    split: Option<SplitKernel>,
    /// The plane shaders of each source.
    shaders: Vec<(CopySource, PlaneShaders)>,
    /// The plane pipelines of each source, attachment format and sample count.
    pipelines: Vec<((CopySource, Format, u32), PlanePipelines)>,
}

impl StencilCopyObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the device is idle, so no pending command uses the objects; destroying a null
        // handle is a no-op.
        unsafe {
            for (_, planes) in &self.pipelines {
                for &pipeline in &planes.pipelines {
                    release::object(raw, pipeline);
                }
                release::object(raw, planes.render_pass);
            }
            for (_, shaders) in &self.shaders {
                release::object(raw, shaders.layout);
                release::object(raw, shaders.set_layout);
                for &fragment in &shaders.fragments {
                    release::object(raw, fragment);
                }
                release::object(raw, shaders.vertex);
            }
            if let Some(kernel) = self.split {
                release::object(raw, kernel.pipeline);
                release::object(raw, kernel.layout);
                release::object(raw, kernel.set_layout);
                release::object(raw, kernel.module);
            }
        }
    }
}

/// The descriptor types of the split's set, in binding order.
const SPLIT_SET: [vk::DescriptorType; 2] = [
    vk::DescriptorType::SAMPLED_IMAGE,
    vk::DescriptorType::STORAGE_IMAGE,
];

/// The descriptor type of the plane draws' source.
fn source_descriptor(source: CopySource) -> vk::DescriptorType {
    match source {
        CopySource::Aspect => vk::DescriptorType::SAMPLED_IMAGE,
        CopySource::Shadow => vk::DescriptorType::STORAGE_IMAGE,
    }
}

impl Device {
    /// The split kernel, made on first use; refused naming `feature` where the device lacks what
    /// it needs.
    fn stencil_split(&self, feature: &Feature) -> Result<SplitKernel, Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn =
            crate::pipeline::budget::helper_turn(&self.shared().stencil_copies, |objects| {
                objects.split.is_some()
            });
        let mut objects = lock(&self.shared().stencil_copies);
        if let Some(kernel) = objects.split {
            return Ok(kernel);
        }
        let module =
            stencil_split_module(self.compile_device(), feature.clone()).map_err(device_error)?;
        if StencilSplitModule::SOURCE_BINDING != 0 || StencilSplitModule::DESTINATION_BINDING != 1 {
            return Err(internal(
                "the stencil split's bindings out of their set layout's order",
            ));
        }
        let shared = self.shared();
        let raw = shared.vk();
        let mut kernel = SplitKernel {
            module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        let made = (|| {
            kernel.module = shader_module(raw, &module.spirv)?;
            kernel.set_layout = set_layout(raw, &SPLIT_SET, vk::ShaderStageFlags::COMPUTE)?;
            kernel.layout = pipeline_layout(raw, kernel.set_layout)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(kernel.module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(kernel.layout);
            kernel.pipeline = shared
                .make_compute_pipeline(Making::new("stencil split").words(&module.spirv), info)?;
            Ok(())
        })();
        match made {
            Ok(()) => {
                objects.split = Some(kernel);
                Ok(kernel)
            }
            Err(error) => {
                // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                unsafe {
                    release::object(raw, kernel.pipeline);
                    release::object(raw, kernel.layout);
                    release::object(raw, kernel.set_layout);
                    release::object(raw, kernel.module);
                }
                Err(error)
            }
        }
    }

    /// The plane draws' shaders reading `source` and their pipelines for an attachment of
    /// `format` at `samples`, made on first use; refused naming `feature` where the device lacks
    /// what they need.
    fn stencil_planes(
        &self,
        source: CopySource,
        format: Format,
        samples: u32,
        feature: &Feature,
    ) -> Result<(PlaneShaders, PlanePipelines), Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn =
            crate::pipeline::budget::helper_turn(&self.shared().stencil_copies, |objects| {
                objects
                    .pipelines
                    .iter()
                    .any(|(key, _)| *key == (source, format, samples))
            });
        let mut objects = lock(&self.shared().stencil_copies);
        let shared = self.shared();
        let raw = shared.vk();
        let made_shaders = objects
            .shaders
            .iter()
            .find(|(of, _)| *of == source)
            .map(|(_, shaders)| *shaders);
        let shaders = match made_shaders {
            Some(shaders) => shaders,
            None => {
                let mut modules = Vec::with_capacity(StencilBit::ALL.len());
                for bit in StencilBit::ALL {
                    let copy = DepthStencilCopy::StencilPlane { source, bit };
                    modules.push(
                        depth_stencil_copy_modules(self.compile_device(), copy, feature.clone())
                            .map_err(device_error)?,
                    );
                }
                let vertex_words = modules
                    .first()
                    .map(|m| m.vertex.clone())
                    .ok_or_else(|| internal("no stencil plane"))?;
                let words = modules
                    .iter()
                    .fold(
                        Making::new("stencil plane modules").words(&vertex_words),
                        |making, module| making.words(&module.fragment),
                    )
                    .finish();
                let mut shaders = PlaneShaders {
                    vertex: vk::ShaderModule::null(),
                    fragments: [vk::ShaderModule::null(); 8],
                    set_layout: vk::DescriptorSetLayout::null(),
                    layout: vk::PipelineLayout::null(),
                    words,
                };
                let made = (|| {
                    shaders.vertex = shader_module(raw, &vertex_words)?;
                    for (fragment, module) in shaders.fragments.iter_mut().zip(&modules) {
                        *fragment = shader_module(raw, &module.fragment)?;
                    }
                    shaders.set_layout = set_layout(
                        raw,
                        &[source_descriptor(source)],
                        vk::ShaderStageFlags::FRAGMENT,
                    )?;
                    shaders.layout = pipeline_layout(raw, shaders.set_layout)?;
                    Ok(())
                })();
                if let Err(error) = made {
                    // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                    unsafe {
                        release::object(raw, shaders.layout);
                        release::object(raw, shaders.set_layout);
                        for &fragment in &shaders.fragments {
                            release::object(raw, fragment);
                        }
                        release::object(raw, shaders.vertex);
                    }
                    return Err(error);
                }
                objects.shaders.push((source, shaders));
                shaders
            }
        };
        if let Some((_, planes)) = objects
            .pipelines
            .iter()
            .find(|(key, _)| *key == (source, format, samples))
        {
            return Ok((shaders, *planes));
        }
        let mut planes = PlanePipelines {
            render_pass: vk::RenderPass::null(),
            pipelines: [vk::Pipeline::null(); 8],
        };
        let made = (|| {
            planes.render_pass = planes_pass(raw, format, samples)?;
            let draws = shaders.fragments.iter().zip(StencilBit::ALL);
            for (pipeline, (&fragment, bit)) in planes.pipelines.iter_mut().zip(draws) {
                let making = Making::new("stencil plane")
                    .digest(&shaders.words)
                    .text(&(source, format, samples, bit));
                *pipeline = plane_pipeline(
                    self,
                    &shaders,
                    fragment,
                    planes.render_pass,
                    (samples, bit, making),
                )?;
            }
            Ok(())
        })();
        match made {
            Ok(()) => {
                objects.pipelines.push(((source, format, samples), planes));
                Ok((shaders, planes))
            }
            Err(error) => {
                // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                unsafe {
                    for &pipeline in &planes.pipelines {
                        release::object(raw, pipeline);
                    }
                    release::object(raw, planes.render_pass);
                }
                Err(error)
            }
        }
    }
}

/// The planes' render pass for an attachment of `format` at `samples`: the attachment alone,
/// `GENERAL` throughout, its depth loaded and stored, its stencil cleared over the render area and
/// stored.
pub(super) fn planes_pass(
    raw: &ash::Device,
    format: Format,
    samples: u32,
) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let attachments = [vk::AttachmentDescription::default()
        .format(format.into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(general)
        .final_layout(general)];
    let depth_stencil = vk::AttachmentReference::default()
        .attachment(0)
        .layout(general);
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_stencil)];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// The plane draw of `bit`, whose fragment stage is `fragment`, in `render_pass` at `samples`:
/// the triangle over the viewport at sample rate, the depth test off, the stencil set to the
/// reference where the fragment lives, under the bit's write mask.
/// Its record is the one `making` names.
fn plane_pipeline(
    device: &Device,
    shaders: &PlaneShaders,
    fragment: vk::ShaderModule,
    render_pass: vk::RenderPass,
    (samples, bit, making): (u32, StencilBit, Making),
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shaders.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let mask = [u32::MAX];
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::from_raw(samples))
        .sample_shading_enable(true)
        .min_sample_shading(1.0)
        .sample_mask(&mask);
    let face = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::REPLACE)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS)
        .compare_mask(0xff)
        .write_mask(bit.mask())
        .reference(0xff);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .stencil_test_enable(true)
        .front(face)
        .back(face);
    let blend = vk::PipelineColorBlendStateCreateInfo::default();
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(shaders.layout)
        .render_pass(render_pass)
        .subpass(0);
    shared.make_graphics_pipeline(making, info)
}

/// The objects one copy holds until its command buffer completes: its descriptor set's pool and,
/// for a draw, its framebuffer (also `depth_copy.rs`'s).
pub(super) struct PassObjects {
    device: Device,
    pool: vk::DescriptorPool,
    pub(super) framebuffer: vk::Framebuffer,
}

impl Drop for PassObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them (a command buffer retains them until it completes);
        // the set is freed with its pool, and a null handle is a no-op.
        unsafe {
            release::object(raw, self.framebuffer);
            release::object(raw, self.pool);
        }
    }
}

impl PassObjects {
    /// A pool of one set and the set, of `set_layout`, whose binding `i` is the view of
    /// `bindings[i]`, of its descriptor type, in `GENERAL`; no framebuffer yet.
    pub(super) fn with_set(
        device: &Device,
        set_layout: vk::DescriptorSetLayout,
        bindings: &[(vk::DescriptorType, vk::ImageView)],
    ) -> Result<(PassObjects, vk::DescriptorSet), Error> {
        let raw = device.shared().vk();
        let mut objects = PassObjects {
            device: device.clone(),
            pool: vk::DescriptorPool::null(),
            framebuffer: vk::Framebuffer::null(),
        };
        let sizes: Vec<_> = bindings
            .iter()
            .map(|&(ty, _)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(1))
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `info` and its array are live for the call.
        objects.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let layouts = [set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(objects.pool)
            .set_layouts(&layouts);
        // SAFETY: the pool holds the set (counted above).
        let sets = unsafe { raw.allocate_descriptor_sets(&info) }.map_err(vulkan)?;
        let set = sets
            .first()
            .copied()
            .ok_or_else(|| internal("the driver allocated no set"))?;
        for (binding, &(ty, view)) in (0u32..).zip(bindings) {
            let info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // SAFETY: the set was just allocated and no command uses it; the view is live (the
            // command buffer retains its texture).
            unsafe {
                raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(ty)
                        .image_info(&info)],
                    &[],
                )
            };
        }
        Ok((objects, set))
    }
}

/// Whether `target`'s image is sampled, as the copies read a source aspect (the module docs).
pub(super) fn sampled(target: &Target) -> Result<bool, Error> {
    Ok(target
        .texture
        .image_plan()?
        .usage
        .contains(ImageUsageFlags::SAMPLED))
}

/// Make, where they are not yet made, the plane draws reading `source` into an attachment of
/// `format` at `samples`: refused naming `feature` where the device lacks what they need.
pub(super) fn plane_draws_run(
    device: &Device,
    source: CopySource,
    format: Format,
    samples: u32,
    feature: &Feature,
) -> Result<(), Error> {
    device
        .stencil_planes(source, format, samples, feature)
        .map(|_| ())
}

/// A copy draw's use of its destination `destination`: its attachment's tests read and write it.
pub(super) fn attachment_use(destination: &Target) -> Result<Use<ResourceKey>, Error> {
    let tests = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    let access = Access::new(
        tests,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    );
    destination.texture.image_use(
        destination.level..destination.level + 1,
        destination.layer..destination.layer + 1,
        access,
        true,
    )
}

/// A copy draw's use of a source aspect `source` it samples.
pub(super) fn sampled_use(source: &Target) -> Result<Use<ResourceKey>, Error> {
    let access = Access::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    source.texture.image_use(
        source.level..source.level + 1,
        source.layer..source.layer + 1,
        access,
        false,
    )
}

/// The stages a copy draw runs in.
pub(super) const DRAW_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
);

/// A copy draw's framebuffer of `destination` alone, as the depth/stencil attachment of
/// `render_pass` (or, of a colour format, its colour attachment: the bits draws, `depth_copy.rs`),
/// `width` by `height`; the objects hold it from here.
pub(super) fn draw_framebuffer(
    objects: &mut PassObjects,
    render_pass: vk::RenderPass,
    destination: &Target,
    [width, height]: [u32; 2],
) -> Result<(), Error> {
    let aspect = image_aspects(destination.format);
    let usage = match aspect == ImageAspectFlags::COLOR {
        true => ImageUsageFlags::COLOR_ATTACHMENT,
        false => ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    };
    let attachment_view = destination.texture.image_view(&VkView {
        view_type: ImageViewType::TYPE_2D,
        format: destination.format,
        components: ComponentMapping::IDENTITY,
        aspect,
        base_level: destination.level,
        level_count: 1,
        base_layer: destination.layer,
        layer_count: 1,
        usage,
    })?;
    let views = [attachment_view];
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&views)
        .width(width)
        .height(height)
        .layers(1);
    let raw = objects.device.shared().vk();
    // SAFETY: `info` and its array are live for the call; the view is live.
    objects.framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
    Ok(())
}

/// The view of `target`'s `aspect` a copy samples.
pub(super) fn aspect_view(
    target: &Target,
    aspect: ImageAspectFlags,
) -> Result<vk::ImageView, Error> {
    target.texture.image_view(&VkView {
        view_type: ImageViewType::TYPE_2D,
        format: target.format,
        components: ComponentMapping::IDENTITY,
        aspect,
        base_level: target.level,
        level_count: 1,
        base_layer: target.layer,
        layer_count: 1,
        usage: ImageUsageFlags::SAMPLED,
    })
}

/// Record, outside a render pass, a copy draw's render pass over `area` from the origin
/// ([`record_draws_in`]).
///
/// # Safety
///
/// As [`record_draws_in`]'s.
pub(super) unsafe fn record_draws(
    device: &ash::Device,
    raw: vk::CommandBuffer,
    pass: (vk::RenderPass, vk::Framebuffer),
    bound: (vk::PipelineLayout, vk::DescriptorSet),
    pipelines: &[vk::Pipeline],
    area: [u32; 2],
) {
    // SAFETY: the caller's contract.
    unsafe { record_draws_in(device, raw, pass, bound, pipelines, DrawArea::whole(area)) }
}

/// A copy draw's render area: `size` pixels from `origin`. Its framebuffer and viewport reach
/// from `(0, 0)` to its end, so a fragment's `FragCoord` is its pixel's in the attachment, and
/// the scissor is the area: a region's copy reads and writes it where it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DrawArea {
    pub(super) origin: [u32; 2],
    pub(super) size: [u32; 2],
}

impl DrawArea {
    /// The area of `size` from the origin.
    pub(super) fn whole(size: [u32; 2]) -> DrawArea {
        DrawArea {
            origin: [0, 0],
            size,
        }
    }

    /// Where the area ends: the framebuffer's and the viewport's size.
    pub(super) fn end(self) -> [u32; 2] {
        [self.origin[0] + self.size[0], self.origin[1] + self.size[1]]
    }
}

/// Record, outside a render pass, a copy draw's render pass over `area`: the framebuffer of
/// `objects` begun in `render_pass` (its stencil cleared to 0 over the area where it clears),
/// `set` bound to `layout`, then each of `pipelines` drawing the triangle over the viewport from
/// `(0, 0)` to the area's end, scissored to the area.
///
/// # Safety
///
/// `raw` is the device's command buffer, recording outside a render pass, which only the caller
/// records; the source, the destination and the objects are live until it completes and rest
/// in `GENERAL`; the framebuffer reaches the area's end.
pub(super) unsafe fn record_draws_in(
    device: &ash::Device,
    raw: vk::CommandBuffer,
    (render_pass, framebuffer): (vk::RenderPass, vk::Framebuffer),
    (layout, set): (vk::PipelineLayout, vk::DescriptorSet),
    pipelines: &[vk::Pipeline],
    drawn: DrawArea,
) {
    let [x, y] = drawn.origin;
    let [width, height] = drawn.size;
    let area = vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D { width, height },
    };
    let [end_x, end_y] = drawn.end();
    let clear = [vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 0.0,
            stencil: 0,
        },
    }];
    let begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(area)
        .clear_values(&clear);
    device.cmd_begin_render_pass(raw, &begin, vk::SubpassContents::INLINE);
    device.cmd_bind_descriptor_sets(raw, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set], &[]);
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: end_x as f32,
        height: end_y as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    device.cmd_set_viewport(raw, 0, &[viewport]);
    device.cmd_set_scissor(raw, 0, &[area]);
    for &pipeline in pipelines {
        device.cmd_bind_pipeline(raw, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(raw, 3, 1, 0, 0);
    }
    device.cmd_end_render_pass(raw);
}

/// One copy into a depth/stencil target's stencil by the plane draws (the module docs).
#[derive(Clone)]
pub(super) struct StencilPlanes {
    objects: Arc<PassObjects>,
    shaders: PlaneShaders,
    planes: PlanePipelines,
    set: vk::DescriptorSet,
    /// The render area.
    area: DrawArea,
}

impl StencilPlanes {
    /// The draws that write the stencil `view` holds, a view of `source`'s kind, into
    /// `destination`'s stencil over the render area `area`, which `destination` covers.
    /// `destination` has a stencil aspect and is a depth/stencil attachment. Refused naming
    /// `feature` where the device lacks what the draws need.
    fn new(
        device: &Device,
        source: CopySource,
        view: vk::ImageView,
        destination: &Target,
        area: DrawArea,
        feature: &Feature,
    ) -> Result<StencilPlanes, Error> {
        if !image_aspects(destination.format).contains(ImageAspectFlags::STENCIL) {
            return Err(internal("a stencil copy into a target without stencil"));
        }
        let (shaders, planes) =
            device.stencil_planes(source, destination.format, destination.samples, feature)?;
        let (mut objects, set) = PassObjects::with_set(
            device,
            shaders.set_layout,
            &[(source_descriptor(source), view)],
        )?;
        draw_framebuffer(&mut objects, planes.render_pass, destination, area.end())?;
        Ok(StencilPlanes {
            objects: Arc::new(objects),
            shaders,
            planes,
            set,
            area,
        })
    }

    /// The draws reading `source`'s stencil aspect, sampled ([`sampled`]), into `destination`'s.
    pub(super) fn from_aspect(
        device: &Device,
        source: &Target,
        destination: &Target,
        area: [u32; 2],
        feature: &Feature,
    ) -> Result<StencilPlanes, Error> {
        if source.samples != destination.samples
            || !image_aspects(source.format).contains(ImageAspectFlags::STENCIL)
        {
            return Err(internal(
                "a stencil copy between targets of other samples or without stencil",
            ));
        }
        let view = aspect_view(source, ImageAspectFlags::STENCIL)?;
        let area = DrawArea::whole(area);
        StencilPlanes::new(device, CopySource::Aspect, view, destination, area, feature)
    }

    /// The draws reading the stencil values `source`, a multisampled `R8_UINT` colour target,
    /// holds at each pixel of `area` into `destination`'s stencil there.
    pub(super) fn from_colour(
        device: &Device,
        source: &Target,
        destination: &Target,
        area: DrawArea,
        feature: &Feature,
    ) -> Result<StencilPlanes, Error> {
        if source.samples != destination.samples || source.format != Format::R8_UINT {
            return Err(internal(
                "a stencil copy from a colour target of other samples or another format",
            ));
        }
        let view = aspect_view(source, ImageAspectFlags::COLOR)?;
        StencilPlanes::new(device, CopySource::Aspect, view, destination, area, feature)
    }

    /// Record, outside a render pass, the draws' render pass.
    ///
    /// # Safety
    ///
    /// `raw` is the device's command buffer, recording outside a render pass, which only the
    /// caller records; the source, the destination and these draws' objects are live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        record_draws_in(
            device,
            raw,
            (self.planes.render_pass, self.objects.framebuffer),
            (self.shaders.layout, self.set),
            &self.planes.pipelines,
            self.area,
        );
    }
}

/// One shadow's copies between its multisampled `R8_UINT` storage image and the stencil of a
/// combined depth/stencil attachment (the module docs), planned with the shadow.
#[derive(Clone)]
pub(super) struct ShadowPlanes {
    /// The split's descriptor pool, held until the command buffer completes (dropping the last
    /// handle frees it).
    _split_pool: Arc<PassObjects>,
    split: SplitKernel,
    split_set: vk::DescriptorSet,
    /// The copy back.
    planes: StencilPlanes,
    /// The render area, from the origin.
    area: [u32; 2],
    /// The attachment's and the shadow's samples: the split dispatches one workgroup layer each.
    samples: u32,
}

impl ShadowPlanes {
    /// The copies between `shadow`, the shadow's storage target, and `stencil`, the pass's
    /// depth/stencil target, over the render area `area`. `stencil`'s image must be sampled
    /// ([`sampled`]). Refused (`Unsupported`, naming `feature`: `StencilExport` for the shader-side
    /// stencil's shadow, `FramebufferFetch` for the in-shader stencil test's stand-in) where the
    /// device lacks what the modules need.
    pub(super) fn new(
        device: &Device,
        stencil: &Target,
        shadow: &Target,
        area: [u32; 2],
        feature: Feature,
    ) -> Result<ShadowPlanes, Error> {
        if !image_aspects(stencil.format).contains(ImageAspectFlags::STENCIL) {
            return Err(internal(
                "a shadow's copies of an attachment without stencil",
            ));
        }
        let split = device.stencil_split(&feature)?;
        let shadow_view = super::render_target::storage_view(shadow)?;
        let planes = StencilPlanes::new(
            device,
            CopySource::Shadow,
            shadow_view,
            stencil,
            DrawArea::whole(area),
            &feature,
        )?;
        let (objects, split_set) = PassObjects::with_set(
            device,
            split.set_layout,
            &[
                (
                    SPLIT_SET[0],
                    aspect_view(stencil, ImageAspectFlags::STENCIL)?,
                ),
                (SPLIT_SET[1], shadow_view),
            ],
        )?;
        Ok(ShadowPlanes {
            _split_pool: Arc::new(objects),
            split,
            split_set,
            planes,
            area,
            samples: stencil.samples,
        })
    }

    /// The copies' uses, both ways, of the shadow and `stencil`.
    pub(super) fn uses(
        &self,
        stencil: &Target,
        shadow: &Target,
    ) -> Result<Vec<Use<ResourceKey>>, Error> {
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let levels = stencil.level..stencil.level + 1;
        let layers = stencil.layer..stencil.layer + 1;
        let sampled = Access::new(compute, vk::AccessFlags2::SHADER_SAMPLED_READ);
        let stored = Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let read = Access::new(fragment, vk::AccessFlags2::SHADER_STORAGE_READ);
        Ok(vec![
            stencil.texture.image_use(levels, layers, sampled, false)?,
            shadow.texture.image_use(0..1, 0..1, stored, true)?,
            shadow.texture.image_use(0..1, 0..1, read, false)?,
            attachment_use(stencil)?,
        ])
    }

    /// Record, outside a render pass, the split of the attachment's stencil into the shadow.
    ///
    /// # Safety
    ///
    /// `raw` is the device's command buffer, recording outside a render pass, which only the
    /// caller records; the shadow, the attachment and these copies' objects are live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record_in(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        let [width, height] = self.area;
        let local = StencilSplitModule::LOCAL_SIZE;
        device.cmd_bind_pipeline(raw, vk::PipelineBindPoint::COMPUTE, self.split.pipeline);
        device.cmd_bind_descriptor_sets(
            raw,
            vk::PipelineBindPoint::COMPUTE,
            self.split.layout,
            0,
            &[self.split_set],
            &[],
        );
        let groups = [width.div_ceil(local), height.div_ceil(local), self.samples];
        device.cmd_dispatch(raw, groups[0], groups[1], groups[2]);
    }

    /// Record, outside a render pass, the plane draws that write the shadow back into the
    /// attachment's stencil.
    ///
    /// # Safety
    ///
    /// As for [`ShadowPlanes::record_in`].
    pub(super) unsafe fn record_out(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        self.planes.record(device, raw);
    }
}
