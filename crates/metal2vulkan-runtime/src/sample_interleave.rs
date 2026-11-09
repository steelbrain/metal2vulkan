//! The exact emulation of a blit copy between a buffer and a multisampled color texture, through a
//! single-sample "wide" image and a multisampled uint scratch image, by the crate-owned interleave
//! modules (`bind::sample_interleave`).
//!
//! Vulkan copies no multisampled image to or from a buffer. Metal copies such a blit per sample,
//! the pixel's samples consecutive in the buffer: texel `(x, y)` sample `s` of an `n`-sample
//! texture lies at `offset + y * bytesPerRow + (x * n + s) * bpp` where the offset and
//! `bytesPerRow` are multiples of the pixel's `n * bpp` bytes, and where it rounds them to
//! otherwise ([`placed_rows`], measured on an Apple M3 Max under macOS 27.0.1). The runtime moves
//! the samples the only way Vulkan can, through
//! two images in the unsigned-integer copy format of the texel size ([`PlanesCopy`]), so a
//! texel's bits cross unchanged:
//!
//! - a **wide** single-sample 2D image, `band_w * n` texels wide and `band_h` tall, whose column
//!   `x * n + s` holds sample `s` of pixel `x`. It is filled from, or emptied into, the buffer by
//!   ordinary buffer-image copies, one per row, so any `bytesPerRow` (overlapping rows included)
//!   is exact. A buffer-image copy's offset is a multiple of the copy format's texel, so where a
//!   row's offset in the buffer's `VkBuffer` is not, the rows go through a tightly packed staging
//!   buffer by buffer copies (announced as `staged-texture-copy`);
//! - a **scratch** multisampled uint image of the band, the same sample count as the texture, in
//!   which each pixel's samples sit in their sample slots, rendered from or split into the wide
//!   image by the interleave modules, and moved to or from the real texture's band sub-rect by a
//!   compatible-format `vkCmdCopyImage` (same sample count, same texel-size class), so the real
//!   texture is only ever transferred, never viewed in another format.
//!
//! **Into the texture** (buffer to texture), per band: fill the wide image from the buffer; the
//! merge sample-rate draw reads wide texel `(x * n + s, y)` and writes sample `s` of scratch
//! texel `(x, y)`; `vkCmdCopyImage` places the scratch band at the texture's band origin.
//! **Out of the texture** (texture to buffer), per band: `vkCmdCopyImage` copies the texture's
//! band into the scratch; the split dispatch reads sample `s` of scratch texel `(x, y)` and
//! writes wide texel `(x * n + s, y)`; empty the wide image into the buffer.
//!
//! The copy moves the **logical** sample index `s` to `s`: the modules use `texelFetch` with
//! `sample = s`, the same logical index Metal's `copyFromBuffer` uses and the sample planes' bit
//! copies already use for multisampled color moves. Physical sample positions do not enter a buffer
//! copy (they affect rasterization and resolve), so no position remap applies.
//!
//! **Banding.** The wide image is `band_w * n` wide, which can exceed `maxImageDimension2D`, so
//! the region is tiled into bands small enough that the wide and scratch images fit the device's
//! 2D limit on each axis and a memory cap; each band is an independent, origin-zero sub-copy.
//!
//! **Depth and stencil**. A multisampled depth or stencil texture's samples cross the same wide
//! image and scratch, in the unsigned format of the aspect's bits (`R32` for 32-bit float depth,
//! `R16` for 16-bit normalized depth, `R8` for stencil); between the scratch and the texture's
//! aspect, where Vulkan copies no colour image into a depth or stencil one without maintenance8,
//! they move by the multisampled blit copies' own move (`sample_copy.rs`): maintenance8's
//! `vkCmdCopyImage` where the driver copies such images right, else the sample-rate draws (the
//! stencil's plane draws, exact; a 32-bit float depth's `FragDepth` draw, exact or the announced
//! `drawn-depth-copy`; a 16-bit normalized depth's code draw, exact where the driver keeps every
//! code; out of the texture, the bits draws). Whatever that move refuses, this copy refuses.
//!
//! A device that cannot run the modules (no `sampleRateShading`, no integer multisample color
//! attachment at the count, no storage/sampled/transfer of the copy format, or a texture without
//! transfer usage), or the depth or stencil move, refuses the copy typed
//! (`RuntimeGap::MultisampledBufferCopy`), the caller's honest fallback; a layout whose rows Metal
//! places by a rule the runtime does not hold, typed too (`RuntimeGap::MultisampledBufferLayout`).

use super::binding_view::VkView;
use super::blit_plan::BufferLayout;
use super::copy_plan::{depth_bytes, CopyProblem, ImageRegion, TextureRegion};
use super::degradation::Degradation;
use super::device::{device_error, lock};
use super::error::{internal, invalid, vulkan, Error, Problem, RuntimeGap};
use super::exit_guard;
use super::image_plan::{sample_count, ImagePlan};
use super::release;
use super::resource::ResourceKey;
use super::sample_copy::{self, SampleCopy, Side};
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::texture::copy_images::Reused;
use super::texture::{ImageMaker, Texture};
use super::texture_descriptor::{largest_side, TextureDescriptor};
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::{Buffer, Device};
use crate::device::functions::Vk;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    interleave_merge_modules, interleave_split_module, InterleaveSplitModule, PlanesCopy,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
    ImageViewType,
};

/// The stages an interleave copy runs in, but for a depth or stencil move's own (`sample_copy`).
const INTERLEAVE_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
        | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
);

/// A cap on the bytes a band's wide plus scratch images hold, so the copy stays bounded: every
/// band of the copy goes through the images of its first of its extent (`texture/copy_images.rs`).
const BAND_MEMORY_CAP: u64 = 64 << 20;

/// The Metal uint pixel format whose Vulkan format is the copy format (for the scratch and wide
/// descriptors).
fn metal_format(copy: PlanesCopy) -> Result<PixelFormat, Error> {
    Ok(match copy {
        PlanesCopy::R8 => PixelFormat::R8Uint,
        PlanesCopy::R16 => PixelFormat::R16Uint,
        PlanesCopy::R32 => PixelFormat::R32Uint,
        PlanesCopy::Rg32 => PixelFormat::RG32Uint,
        PlanesCopy::Rgba32 => PixelFormat::RGBA32Uint,
        _ => return Err(internal("an interleave copy of an unknown copy format")),
    })
}

/// The bytes of one texel of the copy format.
fn texel_bytes(copy: PlanesCopy) -> u64 {
    match copy {
        PlanesCopy::R8 => 1,
        PlanesCopy::R16 => 2,
        PlanesCopy::R32 => 4,
        PlanesCopy::Rg32 => 8,
        PlanesCopy::Rgba32 => 16,
        _ => 4,
    }
}

/// The copy format of a texture's texels in `aspect` of its Vulkan `format` (the module docs):
/// a colour format's of its texel size; a depth or stencil aspect's of its bits (`R32` for
/// 32-bit float depth, `R16` for 16-bit normalized depth, `R8` for stencil). `None` for a
/// format of no copy.
fn copy_format(format: Format, aspect: ImageAspectFlags) -> Option<PlanesCopy> {
    if aspect == ImageAspectFlags::COLOR {
        return PlanesCopy::for_format(format);
    }
    if aspect == ImageAspectFlags::STENCIL {
        return Some(PlanesCopy::R8);
    }
    if aspect != ImageAspectFlags::DEPTH {
        return None;
    }
    match depth_bytes(format)? {
        4 => Some(PlanesCopy::R32),
        2 => Some(PlanesCopy::R16),
        _ => None,
    }
}

/// A compute kernel of the split, the device's.
#[derive(Clone, Copy, Debug)]
struct SplitKernel {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// The merge's shader modules and layouts for one sample count, the device's: the fragment
/// module bakes the count into its wide column `x * n + s`, so they are one per count.
#[derive(Clone, Copy, Debug)]
struct MergeShaders {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// The digest of its modules' SPIR-V words, which its pipelines' records are named by
    /// (`shader_cache/driver.rs`).
    words: Hash256,
}

/// The merge's render pass and pipeline for one copy format and sample count, the device's.
#[derive(Clone, Copy, Debug)]
struct MergePipeline {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

/// The interleave copies' objects one device has made (the module docs).
#[derive(Default)]
pub(super) struct InterleaveObjects {
    split: Vec<((PlanesCopy, u32), SplitKernel)>,
    merge: Vec<(u32, MergeShaders)>,
    merge_pipelines: Vec<((PlanesCopy, u32), MergePipeline)>,
}

impl InterleaveObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the device is idle, so no pending command uses the objects; destroying a null
        // handle is a no-op.
        unsafe {
            for (_, merge) in &self.merge_pipelines {
                release::object(raw, merge.pipeline);
                release::object(raw, merge.render_pass);
            }
            for (_, shaders) in &self.merge {
                release::object(raw, shaders.layout);
                release::object(raw, shaders.set_layout);
                release::object(raw, shaders.fragment);
                release::object(raw, shaders.vertex);
            }
            for (_, kernel) in &self.split {
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

/// The descriptor types of the merge's set.
const MERGE_SET: [vk::DescriptorType; 1] = [vk::DescriptorType::SAMPLED_IMAGE];

impl Device {
    /// The split kernel of `copy` at `samples`, made on first use.
    fn interleave_split(&self, copy: PlanesCopy, samples: u32) -> Result<SplitKernel, Error> {
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().interleave, |objects| {
            objects.split.iter().any(|(k, _)| *k == (copy, samples))
        });
        let mut objects = lock(&self.shared().interleave);
        if let Some((_, kernel)) = objects.split.iter().find(|(k, _)| *k == (copy, samples)) {
            return Ok(*kernel);
        }
        let module =
            interleave_split_module(self.compile_device(), copy, samples).map_err(device_error)?;
        if InterleaveSplitModule::SOURCE_BINDING != 0 || InterleaveSplitModule::WIDE_BINDING != 1 {
            return Err(internal(
                "the split's bindings out of their set layout's order",
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
            kernel.pipeline = shared.make_compute_pipeline(
                Making::new("interleave split").words(&module.spirv),
                info,
            )?;
            Ok(())
        })();
        if let Err(error) = made {
            // SAFETY: nothing used the objects; destroying a null handle is a no-op.
            unsafe {
                release::object(raw, kernel.pipeline);
                release::object(raw, kernel.layout);
                release::object(raw, kernel.set_layout);
                release::object(raw, kernel.module);
            }
            return Err(error);
        }
        objects.split.push(((copy, samples), kernel));
        Ok(kernel)
    }

    /// The merge's shaders and its pipeline for `copy` at `samples`, made on first use.
    fn interleave_merge(
        &self,
        copy: PlanesCopy,
        samples: u32,
    ) -> Result<(MergeShaders, MergePipeline), Error> {
        // The merge renders the scratch's view in the copy format, an integer one, at the
        // texture's samples: a count every integer color attachment takes, or the copy format's
        // own.
        let alternatives = [
            VkProperty::FramebufferIntegerColorSampleCount(samples),
            VkProperty::ColorAttachmentSampleCount(copy.format(), samples),
        ];
        let enabled = &self.compile_device().profile().enabled().properties;
        if !alternatives.iter().any(|p| enabled.contains(p)) {
            return Err(Error::Unsupported(Unsupported::needs_any_of(
                Feature::Format,
                alternatives
                    .iter()
                    .map(|&p| vec![Capability::Property(p)])
                    .collect(),
            )));
        }
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().interleave, |objects| {
            objects
                .merge_pipelines
                .iter()
                .any(|(key, _)| *key == (copy, samples))
        });
        let mut objects = lock(&self.shared().interleave);
        let shared = self.shared();
        let raw = shared.vk();
        let known = objects
            .merge
            .iter()
            .find(|(count, _)| *count == samples)
            .map(|(_, shaders)| *shaders);
        let shaders = match known {
            Some(shaders) => shaders,
            None => {
                let modules = interleave_merge_modules(self.compile_device(), samples)
                    .map_err(device_error)?;
                let mut shaders = MergeShaders {
                    vertex: vk::ShaderModule::null(),
                    fragment: vk::ShaderModule::null(),
                    set_layout: vk::DescriptorSetLayout::null(),
                    layout: vk::PipelineLayout::null(),
                    words: Making::new("interleave merge modules")
                        .words(&modules.vertex)
                        .words(&modules.fragment)
                        .finish(),
                };
                let made = (|| {
                    shaders.vertex = shader_module(raw, &modules.vertex)?;
                    shaders.fragment = shader_module(raw, &modules.fragment)?;
                    shaders.set_layout =
                        set_layout(raw, &MERGE_SET, vk::ShaderStageFlags::FRAGMENT)?;
                    shaders.layout = pipeline_layout(raw, shaders.set_layout)?;
                    Ok(())
                })();
                if let Err(error) = made {
                    // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                    unsafe {
                        release::object(raw, shaders.layout);
                        release::object(raw, shaders.set_layout);
                        release::object(raw, shaders.fragment);
                        release::object(raw, shaders.vertex);
                    }
                    return Err(error);
                }
                objects.merge.push((samples, shaders));
                shaders
            }
        };
        if let Some((_, merge)) = objects
            .merge_pipelines
            .iter()
            .find(|(key, _)| *key == (copy, samples))
        {
            return Ok((shaders, *merge));
        }
        let mut merge = MergePipeline {
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
        };
        let made = (|| {
            merge.render_pass = merge_pass(raw, copy, samples)?;
            let making = Making::new("interleave merge")
                .digest(&shaders.words)
                .text(&(copy, samples));
            merge.pipeline = merge_pipeline(self, &shaders, merge.render_pass, samples, making)?;
            Ok(())
        })();
        match made {
            Ok(()) => {
                objects.merge_pipelines.push(((copy, samples), merge));
                Ok((shaders, merge))
            }
            Err(error) => {
                // SAFETY: nothing used the objects; destroying a null handle is a no-op.
                unsafe {
                    release::object(raw, merge.pipeline);
                    release::object(raw, merge.render_pass);
                }
                Err(error)
            }
        }
    }
}

/// The merge's render pass for `copy` at `samples`: one color attachment in the copy format,
/// `GENERAL` throughout, its render area written whole (`DONT_CARE`, then `STORE`).
fn merge_pass(raw: &Vk, copy: PlanesCopy, samples: u32) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let attachments = [vk::AttachmentDescription::default()
        .format(copy.format().into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general)];
    let colors = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(general)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// The merge's pipeline in `render_pass` at `samples`: the triangle over the viewport at sample
/// rate, without blending, writing every component. Its record is the one `making` names.
fn merge_pipeline(
    device: &Device,
    shaders: &MergeShaders,
    render_pass: vk::RenderPass,
    samples: u32,
    making: Making,
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shaders.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shaders.fragment)
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
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(shaders.layout)
        .render_pass(render_pass)
        .subpass(0);
    shared.make_graphics_pipeline(making, info)
}

/// A band's descriptor pool and framebuffer, held until the command buffer completes.
struct Holder {
    device: Device,
    pool: vk::DescriptorPool,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for Holder {
    fn drop(&mut self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them (the command buffer retains them until it
        // completes); sets are freed with their pool, and a null handle is a no-op.
        unsafe {
            for &framebuffer in &self.framebuffers {
                release::object(raw, framebuffer);
            }
            release::object(raw, self.pool);
        }
    }
}

impl Holder {
    /// A pool of `splits` split sets and `merges` merge sets on `device`.
    fn new(device: &Device, splits: u32, merges: u32) -> Result<Holder, Error> {
        let mut holder = Holder {
            device: device.clone(),
            pool: vk::DescriptorPool::null(),
            framebuffers: Vec::new(),
        };
        let sets = splits + merges;
        if sets > 0 {
            let mut sizes = vec![vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(splits + merges)];
            if splits > 0 {
                sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(splits),
                );
            }
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(sets)
                .pool_sizes(&sizes);
            let raw = device.shared().vk();
            // SAFETY: `info` and its array are live for the call.
            holder.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        }
        Ok(holder)
    }

    /// A set of `set_layout` with each binding's image written, of its type in `types` and its
    /// view in `views`.
    fn set(
        &self,
        set_layout: vk::DescriptorSetLayout,
        types: &[vk::DescriptorType],
        views: &[vk::ImageView],
    ) -> Result<vk::DescriptorSet, Error> {
        let raw = self.device.shared().vk();
        let layouts = [set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: the pool holds a set of every band's layout (counted by `new`'s caller).
        let sets = unsafe { raw.allocate_descriptor_sets(&info) }.map_err(vulkan)?;
        let set = sets
            .first()
            .copied()
            .ok_or_else(|| internal("the driver allocated no set"))?;
        for (binding, (&ty, &view)) in types.iter().zip(views).enumerate() {
            let image = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // SAFETY: the set was just allocated and no command uses it; the view is live.
            unsafe {
                raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(ty)
                        .image_info(&image)],
                    &[],
                )
            };
        }
        Ok(set)
    }
}

/// One recorded step of an interleave copy.
enum Recorded {
    /// Buffer-image copies on the buffer and the wide image (filling the wide image from the
    /// buffer, or emptying it, by `into_wide`).
    BufferRows {
        buffer: vk::Buffer,
        wide: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
        into_wide: bool,
    },
    /// The split dispatch: scratch -> wide.
    Split {
        kernel: SplitKernel,
        set: vk::DescriptorSet,
        groups: [u32; 3],
    },
    /// The merge draw: wide -> scratch.
    Merge {
        shaders: MergeShaders,
        merge: MergePipeline,
        framebuffer: vk::Framebuffer,
        set: vk::DescriptorSet,
        extent: [u32; 2],
    },
    /// A `vkCmdCopyImage` between the scratch and the real texture (same sample count).
    CopyImage {
        src: vk::Image,
        dst: vk::Image,
        region: vk::ImageCopy,
    },
    /// The move between the scratch and a depth or stencil texture's aspect (`sample_copy.rs`).
    Move(SampleCopy),
    /// Buffer copies between the buffer's rows and the staging buffer's (`Staging`).
    Rows {
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
}

/// The tightly packed buffer a copy's rows go through where a row's offset in the buffer's
/// `VkBuffer` is off the copy format's texel, which a buffer-image copy's offset must be a
/// multiple of (VUID-VkBufferImageCopy-bufferOffset-00193): its `VkBuffer`, where its byte 0
/// lies in it, and the buffer, held until the command buffer completes.
struct Staging {
    raw: vk::Buffer,
    at: u64,
    _buffer: Buffer,
}

/// The emulated copy: its steps, the resource uses the command barriers over, the stages it
/// runs in, the band objects, the images it goes through (its bands' scratch and wide images, the
/// bands of one extent through the first's, `texture/copy_images.rs`), held as long as the copy,
/// and the staging buffer its rows go through, if they do (with the first row's offset in the
/// buffer). The texture it copies into or out of is its caller's to hold while a command buffer
/// records the copy (a blit's encoder retains it, and an encoder a heap texture's moves'
/// texture, `texture/heap_tiled.rs`), and so is the buffer it copies with
/// (`EncoderCore::blit_at`).
pub(super) struct InterleaveCopy {
    steps: Vec<Recorded>,
    pub(super) uses: Vec<Use<ResourceKey>>,
    pub(super) stages: vk::PipelineStageFlags2,
    _holder: Holder,
    _textures: Vec<Texture>,
    staging: Option<(Staging, u64)>,
}

/// A 2D view of `texture`'s only level and layer in its own format, serving `usage`.
fn whole_view(
    texture: &Texture,
    format: Format,
    usage: ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    texture.image_view(&VkView {
        view_type: ImageViewType::TYPE_2D,
        format,
        components: ComponentMapping::IDENTITY,
        aspect: ImageAspectFlags::COLOR,
        base_level: 0,
        level_count: 1,
        base_layer: 0,
        layer_count: 1,
        usage,
    })
}

/// A single-sample wide image of `width` by `height` in the copy format, with `usage`.
fn wide_texture(
    maker: &dyn ImageMaker,
    copy: PlanesCopy,
    [width, height]: [u32; 2],
    usage: ImageUsageFlags,
) -> Result<Texture, Error> {
    let pixel_format = metal_format(copy)?;
    let descriptor = TextureDescriptor {
        texture_type: TextureType::Type2D,
        pixel_format,
        width: width.into(),
        height: height.into(),
        usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format: copy.format(),
        extent: [width, height, 1],
        levels: 1,
        layers: 1,
        samples: sample_count(1).ok_or_else(|| internal("one sample"))?,
        usage,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(copy.format(), ImageAspectFlags::COLOR),
        unavailable: TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    Device::new_planned_texture(maker, &descriptor, plan)
}

/// A multisampled uint scratch image of `width` by `height` at `samples` in the copy format,
/// with `usage`.
fn scratch_texture(
    maker: &dyn ImageMaker,
    copy: PlanesCopy,
    [width, height]: [u32; 2],
    samples: u32,
    usage: ImageUsageFlags,
) -> Result<Texture, Error> {
    let pixel_format = metal_format(copy)?;
    let descriptor = TextureDescriptor {
        texture_type: TextureType::Type2DMultisample,
        pixel_format,
        width: width.into(),
        height: height.into(),
        sample_count: samples.into(),
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format: copy.format(),
        extent: [width, height, 1],
        levels: 1,
        layers: 1,
        samples: sample_count(samples)
            .ok_or_else(|| internal("an interleave copy of a sample count Vulkan lacks"))?,
        usage,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(copy.format(), ImageAspectFlags::COLOR),
        unavailable: TextureUsage::SHADER_WRITE | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    Device::new_planned_texture(maker, &descriptor, plan)
}

/// The bands of a region of `width` by `height` texels at `samples`, each `band_w` by `band_h`
/// so the wide (`band_w * n` wide) and scratch images fit `limit` on each axis and the memory
/// cap: `(bx, by, band_w, band_h)` tiles covering the region.
fn bands(width: u32, height: u32, samples: u32, bpp: u64, limit: u32) -> Vec<(u32, u32, u32, u32)> {
    let n = samples.max(1);
    // Wide width is `band_w * n`, so cap `band_w` by `limit / n`; scratch width is `band_w`.
    let mut max_w = (limit / n).max(1);
    // Keep the wide plus scratch bytes of a band under the cap: both are `band_w * n * band_h *
    // bpp`, so `2 * band_w * n * band_h * bpp <= CAP`. Hold `band_h` at the row count first.
    let per_texel = 2 * u64::from(n) * bpp;
    // Columns a full-height band may hold under the cap (at least one).
    if height > 0 {
        let by_memory = (BAND_MEMORY_CAP / (per_texel * u64::from(height))).max(1);
        if let Ok(by_memory) = u32::try_from(by_memory) {
            max_w = max_w.min(by_memory.max(1));
        }
    }
    let band_w = width.min(max_w).max(1);
    // Rows a band of `band_w` columns may hold under the cap and the 2D limit.
    let by_memory = (BAND_MEMORY_CAP / (u64::from(band_w) * per_texel)).max(1);
    let max_h = u32::try_from(by_memory)
        .unwrap_or(u32::MAX)
        .min(limit)
        .max(1);
    let band_h = height.min(max_h).max(1);
    let mut tiles = Vec::new();
    let mut by = 0;
    while by < height {
        let bh = band_h.min(height - by);
        let mut bx = 0;
        while bx < width {
            let bw = band_w.min(width - bx);
            tiles.push((bx, by, bw, bh));
            bx += bw;
        }
        by += bh;
    }
    tiles
}

/// Why plain Metal's placement of a multisampled copy's rows ([`placed_rows`]) gives no rows the
/// runtime copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unplaced {
    /// A layout plain Metal copies by a rule not yet reduced, which Metal's API validation
    /// rejects (`RuntimeGap::MultisampledBufferLayout`).
    Gap,
    /// A pitch below a row out of the texture, where plain Metal's row writes race, or into it
    /// one below the pixel's placement unit from an offset that is a multiple of 16, from which
    /// it reads bytes the buffer does not hold (`CopyProblem::BytesPerRow`).
    BytesPerRow,
    /// A row past `length`, or past `u64::MAX` (`CopyProblem::BufferRange`).
    BufferRange,
}

/// Where plain Metal places the buffer's rows of a blit between a buffer of `length` bytes laid
/// out as `layout` and `rows` rows of a multisampled texture whose pixels (a texel's samples,
/// consecutive) are `pixel` bytes, `row_bytes` a row (`into_texture` from the buffer): each row's
/// offset from the buffer's byte 0, in order. Measured on an Apple M3 Max under macOS 27.0.1 (its
/// API validation stops every offset off the pixel), for 1-, 2-, 4-, 8- and 16-byte colour, depth
/// and stencil samples at 2 and 4 samples, both ways, 1 to 8 texels wide, and nine of those
/// layouts measured again at render-target usage with and without `PixelFormatView`
/// (`metal2vulkan-conformance/probes/multisample_buffer_copies.m`, each copy in a spawned process:
/// in a child that forked and has not exec'd, Metal's blit aborts making its program):
///
/// - the first row lies at the offset rounded down to 4 bytes, or to the pixel where that is
///   smaller, and to 2 bytes for a 32-byte pixel;
/// - each next row lies `bytesPerRow` rounded down to the pixel, or to 16 bytes where that is
///   smaller, further on: into the texture at a pitch below a row too, rows that overlap
///   included, and every row at the first where the pitch rounds to 0 from an offset off 16.
///
/// Refused (`Unplaced::BytesPerRow`): out of the texture, a pitch below a row of more than one
/// row; into it, one between 0 and the unit it rounds to from an offset that is a multiple of 16,
/// from which plain Metal reads bytes the buffer does not hold. The runtime's gap
/// (`Unplaced::Gap`), layouts plain Metal copies by rules not yet reduced and its API validation
/// rejects: a pitch of 0 from an offset that is a multiple of 16 (but for a single row from 0),
/// where plain Metal copies the texture's samples in a block order of its own (2 by 2 pixels,
/// from byte 0 for an offset below 64); and a pixel of 32 bytes or more at a pitch below a row,
/// or off the pixel for more than one row, whose rows plain Metal places by the offset's and the
/// pitch's bits together. A row past `length`
/// (`Unplaced::BufferRange`) is refused as a single-sample copy's layout past its buffer is. An
/// empty copy places no rows.
pub(super) fn placed_rows(
    layout: &BufferLayout,
    pixel: u64,
    row_bytes: u64,
    rows: u64,
    length: u64,
    into_texture: bool,
) -> Result<Vec<u64>, Unplaced> {
    if rows == 0 || row_bytes == 0 {
        return Ok(Vec::new());
    }
    let (offset, bpr) = (layout.offset, layout.bytes_per_row);
    let block_order = offset.is_multiple_of(16);
    if bpr == 0 && block_order && (offset != 0 || rows > 1) {
        return Err(Unplaced::Gap);
    }
    if pixel >= 32 && (bpr < row_bytes || (rows > 1 && !bpr.is_multiple_of(pixel))) {
        return Err(Unplaced::Gap);
    }
    let align = if pixel == 32 { 2 } else { pixel.min(4) };
    let first = offset - offset % align;
    let unit = pixel.min(16);
    if rows > 1 && bpr < row_bytes && (!into_texture || (bpr < unit && block_order)) {
        return Err(Unplaced::BytesPerRow);
    }
    let pitch = bpr - bpr % unit;
    let starts = (0..rows)
        .map(|y| (y.checked_mul(pitch)).and_then(|row| first.checked_add(row)))
        .collect::<Option<Vec<u64>>>()
        .ok_or(Unplaced::BufferRange)?;
    // The rows follow one another (a pitch is never negative), so the last ends furthest.
    let past = (starts.last())
        .is_some_and(|&start| start.checked_add(row_bytes).is_none_or(|end| end > length));
    match past {
        true => Err(Unplaced::BufferRange),
        false => Ok(starts),
    }
}

/// The buffer-image copy regions for a band of `bw` by `bh` texels at `samples`, `bpp` bytes a
/// sample, the band's column base `bx` texels and row base `by`, between the wide image's rows
/// `0..bh` and the rows of the buffer that start `starts` bytes from `base` ([`placed_rows`]), or
/// (`staged`) a tightly packed scratch's from `base`: one region per row, so rows that overlap
/// are exact.
fn buffer_rows(
    base: u64,
    starts: &[u64],
    staged: bool,
    (bx, by): (u32, u32),
    (bw, bh): (u32, u32),
    samples: u32,
    bpp: u64,
) -> Vec<vk::BufferImageCopy> {
    let n = u64::from(samples);
    let row_texels = u64::from(bw) * n;
    (0..bh)
        .map(|ry| {
            let offset = match staged {
                true => base + u64::from(ry) * row_texels * bpp,
                false => base + starts[(by + ry) as usize] + u64::from(bx) * n * bpp,
            };
            vk::BufferImageCopy {
                buffer_offset: offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: 0,
                    y: ry as i32,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: row_texels as u32,
                    height: 1,
                    depth: 1,
                },
            }
        })
        .collect()
}

/// The copies of a band's rows between the buffer (its rows from `base`, at `starts`) and the
/// staging scratch (tightly packed from `at`), buffer to scratch: a band of `bw` by `bh` texels
/// at `samples`, `bpp` bytes a sample, from column `bx` and row `by`.
fn staged_rows(
    (base, starts): (u64, &[u64]),
    at: u64,
    (bx, by): (u32, u32),
    (bw, bh): (u32, u32),
    samples: u32,
    bpp: u64,
) -> Vec<vk::BufferCopy> {
    let row = u64::from(bw) * u64::from(samples) * bpp;
    (0..bh)
        .map(|ry| vk::BufferCopy {
            src_offset: base
                + starts[(by + ry) as usize]
                + u64::from(bx) * u64::from(samples) * bpp,
            dst_offset: at + u64::from(ry) * row,
            size: row,
        })
        .collect()
}

/// The `vkCmdCopyImage` region between a band's scratch (origin 0) and the texture's band
/// sub-rect at `(level, layer)` origin `(tx, ty)`, of `bw` by `bh` texels at `layers` 1.
fn scratch_copy(
    (tx, ty): (u32, u32),
    (bw, bh): (u32, u32),
    level: u32,
    layer: u32,
    scratch_to_texture: bool,
) -> vk::ImageCopy {
    let scratch = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let texture = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: level,
        base_array_layer: layer,
        layer_count: 1,
    };
    let zero = vk::Offset3D { x: 0, y: 0, z: 0 };
    let at = vk::Offset3D {
        x: tx as i32,
        y: ty as i32,
        z: 0,
    };
    let extent = vk::Extent3D {
        width: bw,
        height: bh,
        depth: 1,
    };
    if scratch_to_texture {
        vk::ImageCopy {
            src_subresource: scratch,
            src_offset: zero,
            dst_subresource: texture,
            dst_offset: at,
            extent,
        }
    } else {
        vk::ImageCopy {
            src_subresource: texture,
            src_offset: at,
            dst_subresource: scratch,
            dst_offset: zero,
            extent,
        }
    }
}

impl InterleaveCopy {
    /// Plan the copy of `region` of `texture` (at `samples`) in `aspect` (Metal's copy unit's)
    /// and the buffer `buffer` laid out as `layout` (its bytes `base` into its `VkBuffer`, `length`
    /// of them the copy may reach), `into_texture` from the buffer, for `call`. `None` where the
    /// device cannot run it (the caller refuses typed). The texture's Vulkan format and the aspect
    /// decide the copy format ([`copy_format`]); a format of no copy (not a single-texel color
    /// format of 1, 2, 4, 8 or 16 bytes, nor a 16- or 32-bit depth or a stencil) gives `None`.
    /// The buffer's rows lie where plain Metal places them ([`placed_rows`]), through a staging
    /// buffer where a row's offset in the `VkBuffer` is off the copy format's texel. Refused
    /// (`Invalid`): a row's samples past `length` ([`CopyProblem::BufferRange`]), as a
    /// single-sample copy's layout past its buffer is, and a pitch [`placed_rows`] refuses
    /// ([`CopyProblem::BytesPerRow`]); a layout it places by a rule the runtime does not hold
    /// is the gap `RuntimeGap::MultisampledBufferLayout`.
    pub(super) fn plan(
        call: &'static str,
        maker: &dyn ImageMaker,
        buffer: vk::Buffer,
        base: u64,
        length: u64,
        layout: &BufferLayout,
        texture: &Texture,
        region: &TextureRegion,
        aspect: ImageAspectFlags,
        samples: u32,
        into_texture: bool,
    ) -> Result<Option<InterleaveCopy>, Error> {
        let device = maker.device().clone();
        // Each band takes the images an earlier band of its extent made: the bands are recorded
        // one after another, each command after a barrier from everything
        // (`texture/copy_images.rs`).
        let reused = Reused::new(maker);
        let maker: &dyn ImageMaker = &reused;
        let plan = texture.image_plan()?;
        let format = plan.format;
        let Some(copy) = copy_format(format, aspect) else {
            return Ok(None);
        };
        let bpp = texel_bytes(copy);
        let samples = samples.max(1);
        let [width, height, _] = region.size;
        let pixel = u64::from(samples) * bpp;
        let row_bytes = u64::from(width) * pixel;
        let rows = u64::from(height);
        let starts = match placed_rows(layout, pixel, row_bytes, rows, length, into_texture) {
            Ok(starts) => starts,
            Err(Unplaced::Gap) => {
                return Err(Error::NotImplemented(RuntimeGap::MultisampledBufferLayout))
            }
            Err(Unplaced::BytesPerRow) => {
                return Err(invalid(call, Problem::Copy(CopyProblem::BytesPerRow)))
            }
            Err(Unplaced::BufferRange) => {
                return Err(invalid(call, Problem::Copy(CopyProblem::BufferRange)))
            }
        };
        // The real texture must take the transfer the `vkCmdCopyImage` needs.
        let transfer = if into_texture {
            ImageUsageFlags::TRANSFER_DST
        } else {
            ImageUsageFlags::TRANSFER_SRC
        };
        if !plan.usage.contains(transfer) {
            return Ok(None);
        }
        // The device's modules and pipelines; `Unsupported` means the device cannot run them.
        let made = if into_texture {
            device.interleave_merge(copy, samples).map(|_| ())
        } else {
            device.interleave_split(copy, samples).map(|_| ())
        };
        match made {
            Ok(()) => {}
            Err(Error::Unsupported(_)) => return Ok(None),
            Err(error) => return Err(error),
        }
        if width == 0 || height == 0 {
            return Ok(Some(InterleaveCopy {
                steps: Vec::new(),
                uses: Vec::new(),
                stages: INTERLEAVE_STAGES,
                _holder: Holder::new(&device, 0, 0)?,
                _textures: Vec::new(),
                staging: None,
            }));
        }
        let [ox, oy, _] = region.origin;
        let level = region.level;
        let layer = region.slice;
        // The wide and scratch images are the runtime's own textures, made through Metal's
        // descriptor check, so a side is held to the smaller of the device's 2D limit and the
        // largest side Metal makes a 2D texture with (16384).
        let metal_side = largest_side(TextureType::Type2D)
            .and_then(|side| u32::try_from(side).ok())
            .unwrap_or(u32::MAX);
        let limit = device
            .compile_device()
            .profile()
            .limits()
            .max_image_dimension_2d
            .min(metal_side);
        // The last band first: a depth or stencil aspect's draws go through images at the
        // texture's coordinates, from the origin to the band's end (`sample_copy.rs`), so the
        // first band's cover every later band's (`texture/copy_images.rs`). The bands write
        // disjoint texels and rows (`placed_rows` refuses rows out of the texture that overlap),
        // so their order is free.
        let mut tiles = bands(width, height, samples, bpp, limit);
        tiles.reverse();
        // A buffer-image copy's offset is a multiple of the copy format's texel; the pitch is a
        // multiple of the pixel or of 16 bytes ([`placed_rows`]), so of the texel too, and only
        // the first row's offset (and a no-copy buffer's byte 0 in its `VkBuffer`) may miss it.
        let staged = (starts.iter()).any(|&start| !(base + start).is_multiple_of(bpp));
        let staging = match staged {
            true => {
                // One band's rows at a time, every band through the same buffer.
                let band = (tiles.iter())
                    .map(|&(_, _, bw, bh)| u64::from(bw) * pixel * u64::from(bh))
                    .max()
                    .unwrap_or(0);
                // A buffer the runtime makes for its own use starts at its `VkBuffer`'s start
                // (`Buffer::base`), so its rows' offsets are multiples of the texel.
                let buffer = device.scratch_buffer(band)?;
                let (raw, at) = buffer.at(0);
                Some(Staging {
                    raw,
                    at,
                    _buffer: buffer,
                })
            }
            false => None,
        };
        let count = tiles.len() as u32;
        let (splits, merges) = if into_texture { (0, count) } else { (count, 0) };
        let mut holder = Holder::new(&device, splits, merges)?;
        let mut steps = Vec::new();
        let mut uses = Vec::new();
        let mut stages = INTERLEAVE_STAGES;
        let mut textures = Vec::new();
        // The real texture's one band-covering use (one level, one layer).
        let stage = if into_texture {
            Access::new(
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            )
        } else {
            Access::new(
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )
        };
        uses.push(texture.image_use(level..level + 1, layer..layer + 1, stage, into_texture)?);
        let texture_image = texture.raw_image()?;
        for (bx, by, bw, bh) in tiles {
            let wide_usage = if into_texture {
                ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST
            } else {
                ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC
            };
            let wide = match wide_texture(maker, copy, [bw * samples, bh], wide_usage) {
                Ok(t) => t,
                Err(Error::Unsupported(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
            let scratch_usage = if into_texture {
                ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC
            } else {
                ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST
            };
            let scratch = match scratch_texture(maker, copy, [bw, bh], samples, scratch_usage) {
                Ok(t) => t,
                Err(Error::Unsupported(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
            let wide_image = wide.raw_image()?;
            let scratch_image = scratch.raw_image()?;
            // Into the texture, the buffer's rows move into the staging buffer first; out of it,
            // they move out of it last.
            let (rows_buffer, regions, staged_step) = match &staging {
                Some(staging) => {
                    let rows = staged_rows(
                        (base, &starts),
                        staging.at,
                        (bx, by),
                        (bw, bh),
                        samples,
                        bpp,
                    );
                    let rows = match into_texture {
                        true => Recorded::Rows {
                            src: buffer,
                            dst: staging.raw,
                            regions: rows,
                        },
                        false => Recorded::Rows {
                            src: staging.raw,
                            dst: buffer,
                            regions: (rows.into_iter())
                                .map(|r| vk::BufferCopy {
                                    src_offset: r.dst_offset,
                                    dst_offset: r.src_offset,
                                    size: r.size,
                                })
                                .collect(),
                        },
                    };
                    let regions =
                        buffer_rows(staging.at, &starts, true, (bx, by), (bw, bh), samples, bpp);
                    (staging.raw, regions, Some(rows))
                }
                None => {
                    let regions =
                        buffer_rows(base, &starts, false, (bx, by), (bw, bh), samples, bpp);
                    (buffer, regions, None)
                }
            };
            let (tx, ty) = (ox + bx, oy + by);
            // The scratch and the texture's band: a colour texture's by one `vkCmdCopyImage`; a
            // depth or stencil aspect's by the multisampled blit copies' move, refused as it is.
            let into_place = match aspect == ImageAspectFlags::COLOR {
                true => Recorded::CopyImage {
                    src: if into_texture {
                        scratch_image
                    } else {
                        texture_image
                    },
                    dst: if into_texture {
                        texture_image
                    } else {
                        scratch_image
                    },
                    region: scratch_copy((tx, ty), (bw, bh), level, layer, into_texture),
                },
                false => {
                    let scratch_side = Side {
                        texture: &scratch,
                        image: scratch_image,
                        aspect: ImageAspectFlags::COLOR,
                        format: copy.format(),
                    };
                    let texture_side = Side {
                        texture,
                        image: texture_image,
                        aspect,
                        format,
                    };
                    let in_scratch = ImageRegion {
                        aspect: ImageAspectFlags::COLOR,
                        level: 0,
                        layer: 0,
                        layers: 1,
                        origin: [0; 3],
                        size: [bw, bh, 1],
                    };
                    let in_texture = ImageRegion {
                        aspect,
                        level,
                        layer,
                        layers: 1,
                        origin: [tx, ty, 0],
                        size: [bw, bh, 1],
                    };
                    let planned = match into_texture {
                        true => sample_copy::plan(
                            maker,
                            &[(in_scratch, in_texture)],
                            scratch_side,
                            texture_side,
                            samples,
                        ),
                        false => sample_copy::plan(
                            maker,
                            &[(in_texture, in_scratch)],
                            texture_side,
                            scratch_side,
                            samples,
                        ),
                    };
                    let moved = match planned {
                        Ok(moved) => moved,
                        Err(Error::Unsupported(_)) => return Ok(None),
                        Err(error) => return Err(error),
                    };
                    uses.extend(moved.uses.iter().cloned());
                    stages |= moved.stages;
                    Recorded::Move(moved)
                }
            };
            if into_texture {
                let (merge_shaders, merge) = device.interleave_merge(copy, samples)?;
                let from = whole_view(&wide, copy.format(), ImageUsageFlags::SAMPLED)?;
                let set = holder.set(merge_shaders.set_layout, &MERGE_SET, &[from])?;
                let into = whole_view(&scratch, copy.format(), ImageUsageFlags::COLOR_ATTACHMENT)?;
                let views = [into];
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(merge.render_pass)
                    .attachments(&views)
                    .width(bw)
                    .height(bh)
                    .layers(1);
                let raw = device.shared().vk();
                // SAFETY: `info` and its array are live for the call; the view is live.
                let framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
                holder.framebuffers.push(framebuffer);
                steps.extend(staged_step);
                steps.push(Recorded::BufferRows {
                    buffer: rows_buffer,
                    wide: wide_image,
                    regions,
                    into_wide: true,
                });
                steps.push(Recorded::Merge {
                    shaders: merge_shaders,
                    merge,
                    framebuffer,
                    set,
                    extent: [bw, bh],
                });
                steps.push(into_place);
            } else {
                let kernel = device.interleave_split(copy, samples)?;
                let source = whole_view(&scratch, copy.format(), ImageUsageFlags::SAMPLED)?;
                let into = whole_view(&wide, copy.format(), ImageUsageFlags::STORAGE)?;
                let set = holder.set(kernel.set_layout, &SPLIT_SET, &[source, into])?;
                let local = InterleaveSplitModule::LOCAL_SIZE;
                steps.push(into_place);
                steps.push(Recorded::Split {
                    kernel,
                    set,
                    groups: [bw.div_ceil(local), bh.div_ceil(local), samples],
                });
                steps.push(Recorded::BufferRows {
                    buffer: rows_buffer,
                    wide: wide_image,
                    regions,
                    into_wide: false,
                });
                steps.extend(staged_step);
            }
            let write = Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            );
            let read = Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            );
            uses.push(wide.image_use(0..1, 0..1, read, true)?);
            uses.push(scratch.image_use(0..1, 0..1, write, true)?);
            for texture in [wide, scratch] {
                if !textures.contains(&texture) {
                    textures.push(texture);
                }
            }
            reused.next();
        }
        let first = starts.first().copied().unwrap_or(0);
        Ok(Some(InterleaveCopy {
            steps,
            uses,
            stages,
            _holder: holder,
            _textures: textures,
            staging: staging.map(|staging| (staging, first)),
        }))
    }

    /// What its depth move's `FragDepth` draw changes where it writes a 32-bit float depth
    /// inexactly (the announced `drawn-depth-copy`); `None` where every step is exact.
    pub(super) fn inexact_depth(&self) -> Option<&'static str> {
        self.steps.iter().find_map(|step| match step {
            Recorded::Move(moved) => moved.inexact_depth(),
            _ => None,
        })
    }

    /// Announce, once the copy is recorded, what its depth move announces for `call` into
    /// `destination` (`drawn-depth-copy`, where the `FragDepth` draw writes a depth inexactly).
    pub(super) fn announce(&self, device: &Device, call: &'static str, destination: &Texture) {
        for step in &self.steps {
            if let Recorded::Move(moved) = step {
                moved.announce(device, call, destination);
            }
        }
    }

    /// Announce, once the copy is recorded, that its rows went through a staging buffer
    /// (`staged-texture-copy`), with a texture of `format`, `into_texture` from the buffer.
    pub(super) fn announce_staging(
        &self,
        device: &Device,
        format: PixelFormat,
        into_texture: bool,
    ) {
        let Some((_, offset)) = self.staging else {
            return;
        };
        device.degrade(Degradation::StagedTextureCopy, || {
            let object = match into_texture {
                true => format!(
                    "a copy from a buffer at offset {offset} into a multisampled {format:?} \
                     texture"
                ),
                false => format!(
                    "a copy from a multisampled {format:?} texture into a buffer at offset \
                     {offset}"
                ),
            };
            let reason = "Vulkan cannot state the buffer's layout: a row's offset in the VkBuffer \
                          is off a sample's bytes";
            (object, reason.to_owned())
        });
    }

    /// Record the copy into `command_buffer`, each step between barriers from everything before
    /// and to everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// buffer, the texture, and this copy's images and objects are live until it completes.
    pub(super) unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        for step in &self.steps {
            // SAFETY: the caller's contract; each command's objects were made for this device.
            unsafe {
                Barrier::everything().record(raw, command_buffer);
                match step {
                    Recorded::BufferRows {
                        buffer,
                        wide,
                        regions,
                        into_wide,
                    } => {
                        if *into_wide {
                            raw.cmd_copy_buffer_to_image(
                                command_buffer,
                                *buffer,
                                *wide,
                                vk::ImageLayout::GENERAL,
                                regions,
                            );
                        } else {
                            raw.cmd_copy_image_to_buffer(
                                command_buffer,
                                *wide,
                                vk::ImageLayout::GENERAL,
                                *buffer,
                                regions,
                            );
                        }
                    }
                    Recorded::Split {
                        kernel,
                        set,
                        groups,
                    } => {
                        raw.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.pipeline,
                        );
                        raw.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.layout,
                            0,
                            &[*set],
                            &[],
                        );
                        let [x, y, z] = *groups;
                        raw.cmd_dispatch(command_buffer, x, y, z);
                    }
                    Recorded::Merge {
                        shaders,
                        merge,
                        framebuffer,
                        set,
                        extent,
                    } => {
                        let [width, height] = *extent;
                        let area = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D { width, height },
                        };
                        let begin = vk::RenderPassBeginInfo::default()
                            .render_pass(merge.render_pass)
                            .framebuffer(*framebuffer)
                            .render_area(area);
                        raw.cmd_begin_render_pass(
                            command_buffer,
                            &begin,
                            vk::SubpassContents::INLINE,
                        );
                        raw.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            merge.pipeline,
                        );
                        raw.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            shaders.layout,
                            0,
                            &[*set],
                            &[],
                        );
                        let viewport = vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: width as f32,
                            height: height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        };
                        raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        raw.cmd_set_scissor(command_buffer, 0, &[area]);
                        raw.cmd_draw(command_buffer, 3, 1, 0, 0);
                        raw.cmd_end_render_pass(command_buffer);
                    }
                    Recorded::CopyImage { src, dst, region } => {
                        raw.cmd_copy_image(
                            command_buffer,
                            *src,
                            vk::ImageLayout::GENERAL,
                            *dst,
                            vk::ImageLayout::GENERAL,
                            &[*region],
                        );
                    }
                    // The move's images rest in `GENERAL`, live until the copy completes, and
                    // `sample_copy::plan` checked its regions, formats and targets.
                    Recorded::Move(moved) => moved.record(raw, command_buffer),
                    Recorded::Rows { src, dst, regions } => {
                        raw.cmd_copy_buffer(command_buffer, *src, *dst, regions)
                    }
                }
                Barrier::everything().record(raw, command_buffer);
            }
        }
    }
}
