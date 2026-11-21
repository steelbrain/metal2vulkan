//! Multisample image allocation and per-sample literal initialization.

use super::buffers::create_host_buffer;
use super::device::VulkanContext;
use super::pipeline::create_shader_module;
use super::METAL_DEFAULT_FRONT_FACE;
use ash::{vk, Device};
use metal2vulkan::device::VkFeature;
use std::ffi::CString;

fn vulkan_sample_count(count: u32) -> Result<vk::SampleCountFlags, String> {
    match count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        8 => Ok(vk::SampleCountFlags::TYPE_8),
        _ => Err(format!("unsupported Vulkan sample count {count}")),
    }
}

/// The SPIR-V words of the executor's own shader `assembly` (`spirv-as` for Vulkan 1.2),
/// assembled in a scratch directory named for `what`.
pub(super) fn assemble_spirv(what: &str, assembly: &str) -> Result<Vec<u8>, String> {
    Ok(crate::spirv_tools::assemble(assembly, what)?)
}

struct MultisampleInitObjects {
    device: Device,
    staging_image: vk::Image,
    staging_memory: vk::DeviceMemory,
    layer_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

impl MultisampleInitObjects {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            staging_image: vk::Image::null(),
            staging_memory: vk::DeviceMemory::null(),
            layer_views: Vec::new(),
            framebuffers: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            shader: vk::ShaderModule::null(),
            command_pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
        }
    }
}

impl Drop for MultisampleInitObjects {
    fn drop(&mut self) {
        unsafe {
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
            for view in self.layer_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            if self.shader != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader, None);
            }
            if self.staging_image != vk::Image::null() {
                self.device.destroy_image(self.staging_image, None);
            }
            if self.staging_memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.staging_memory, None);
            }
        }
    }
}

/// Every aspect of `format` an image of it holds, given the one aspect (`aspect`) a case's
/// literal fills: a combined depth/stencil format's depth comes with its stencil, which a view
/// that attaches the image and a barrier that moves its layout must name together.
fn format_aspects(format: vk::Format, aspect: vk::ImageAspectFlags) -> vk::ImageAspectFlags {
    if aspect == vk::ImageAspectFlags::DEPTH && super::graphics_pipeline::has_stencil(format) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        aspect
    }
}

pub(super) fn initialize_multisample_image(
    context: &VulkanContext,
    destination: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    layout: crate::literal::TextureLayout,
    literal_format: crate::case::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    if literal_format.sampled_only() {
        return Err(format!(
            "{literal_format:?} is a sampled-only format, which no case multisamples"
        ));
    }
    if !context
        .enabled
        .features
        .contains(&VkFeature::SampleRateShading)
    {
        return Err("Vulkan device does not support per-sample literal initialization".into());
    }
    let mut objects = MultisampleInitObjects::new(&context.device);
    let staging_layers = layout
        .array_layers
        .checked_mul(layout.sample_count)
        .ok_or_else(|| "multisample staging layer count overflows".to_string())?;
    // The per-sample source is a 2D ARRAY image with one layer per (array layer, sample), and
    // Vulkan requires `arrayLayers == 1` for LINEAR tiling -- so it cannot be filled by mapping
    // its memory. Stage it the way every other image here is staged: an OPTIMAL image filled by
    // one `vkCmdCopyBufferToImage` from a host buffer holding the layers tightly packed.
    let staging_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: layout.width,
            height: layout.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(staging_layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    objects.staging_image = unsafe { context.device.create_image(&staging_info, None) }
        .map_err(|error| format!("create multisample staging image: {error}"))?;
    let requirements = unsafe {
        context
            .device
            .get_image_memory_requirements(objects.staging_image)
    };
    let memory_type = context.memory_type(
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    objects.staging_memory = unsafe { context.device.allocate_memory(&allocation, None) }
        .map_err(|error| format!("allocate multisample staging memory: {error}"))?;
    unsafe {
        context
            .device
            .bind_image_memory(objects.staging_image, objects.staging_memory, 0)
    }
    .map_err(|error| format!("bind multisample staging image: {error}"))?;
    let pixel_size = literal_format.bytes_per_pixel();
    // Authored order is texel-major, sample-minor; the copy wants layer-major, tightly packed.
    let mut packed = vec![0u8; bytes.len()];
    for layer in 0..layout.array_layers as usize {
        for sample in 0..layout.sample_count as usize {
            let staging_layer = layer * layout.sample_count as usize + sample;
            for y in 0..layout.height as usize {
                for x in 0..layout.width as usize {
                    let source = (((layer * layout.height as usize + y) * layout.width as usize
                        + x)
                        * layout.sample_count as usize
                        + sample)
                        * pixel_size;
                    let destination =
                        ((staging_layer * layout.height as usize + y) * layout.width as usize + x)
                            * pixel_size;
                    packed[destination..destination + pixel_size]
                        .copy_from_slice(&bytes[source..source + pixel_size]);
                }
            }
        }
    }
    let staging_transfer = create_host_buffer(
        context,
        "multisample staging",
        &packed,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    // The pass attaches every aspect of the format: a depth/stencil attachment's view covers
    // both of a combined format's aspects, whose stencil the initializer leaves undefined (a
    // multisampled stencil is cleared after it).
    let attached = format_aspects(format, aspect);
    for layer in 0..layout.array_layers {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(destination)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: attached,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            });
        objects.layer_views.push(
            unsafe { context.device.create_image_view(&view_info, None) }
                .map_err(|error| format!("create multisample layer {layer} view: {error}"))?,
        );
    }

    let descriptor_binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    objects.descriptor_layout = unsafe {
        context.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_binding),
            None,
        )
    }
    .map_err(|error| format!("create multisample initializer descriptor layout: {error}"))?;
    let layouts = [objects.descriptor_layout];
    objects.pipeline_layout = unsafe {
        context.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
            None,
        )
    }
    .map_err(|error| format!("create multisample initializer pipeline layout: {error}"))?;

    let samples = vulkan_sample_count(layout.sample_count)?;
    let attachment_layout = if aspect == vk::ImageAspectFlags::DEPTH {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    };
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(samples)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::GENERAL)];
    let reference = vk::AttachmentReference::default()
        .attachment(0)
        .layout(attachment_layout);
    let color_references = [reference];
    let mut subpass =
        vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
    if aspect == vk::ImageAspectFlags::DEPTH {
        subpass = subpass.depth_stencil_attachment(&reference);
    } else {
        subpass = subpass.color_attachments(&color_references);
    }
    let subpasses = [subpass];
    objects.render_pass = unsafe {
        context.device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses),
            None,
        )
    }
    .map_err(|error| format!("create multisample initializer render pass: {error}"))?;
    for view in &objects.layer_views {
        let framebuffer_attachments = [*view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(objects.render_pass)
            .attachments(&framebuffer_attachments)
            .width(layout.width)
            .height(layout.height)
            .layers(1);
        objects.framebuffers.push(
            unsafe { context.device.create_framebuffer(&info, None) }
                .map_err(|error| format!("create multisample initializer framebuffer: {error}"))?,
        );
    }

    let spv = assemble_spirv(
        "multisample-initializer",
        &multisample_initializer_spvasm(literal_format),
    )?;
    objects.shader = create_shader_module(context, &spv, "multisample initializer")?;
    let vertex_name = CString::new("vertex").expect("static vertex entry");
    let fragment_name = CString::new("fragment").expect("static fragment entry");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(objects.shader)
            .name(&vertex_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(objects.shader)
            .name(&fragment_name),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: layout.width as f32,
        height: layout.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: layout.width,
            height: layout.height,
        },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(METAL_DEFAULT_FRONT_FACE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .sample_shading_enable(true)
        .min_sample_shading(1.0);
    let color_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(
        if aspect == vk::ImageAspectFlags::DEPTH {
            &[]
        } else {
            &color_attachments
        },
    );
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(aspect == vk::ImageAspectFlags::DEPTH)
        .depth_write_enable(aspect == vk::ImageAspectFlags::DEPTH)
        .depth_compare_op(vk::CompareOp::ALWAYS);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(objects.pipeline_layout)
        .render_pass(objects.render_pass)
        .subpass(0)];
    objects.pipeline = unsafe {
        context
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
    }
    .map_err(|(_, error)| format!("create multisample initializer pipeline: {error}"))?[0];

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: layout.array_layers,
    }];
    objects.descriptor_pool = unsafe {
        context.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(layout.array_layers)
                .pool_sizes(&pool_sizes),
            None,
        )
    }
    .map_err(|error| format!("create multisample initializer descriptor pool: {error}"))?;
    let set_layouts = vec![objects.descriptor_layout; layout.array_layers as usize];
    let descriptor_sets = unsafe {
        context.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(objects.descriptor_pool)
                .set_layouts(&set_layouts),
        )
    }
    .map_err(|error| format!("allocate multisample initializer descriptors: {error}"))?;
    let mut source_views = Vec::with_capacity(layout.array_layers as usize);
    for layer in 0..layout.array_layers {
        let info = vk::ImageViewCreateInfo::default()
            .image(objects.staging_image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer * layout.sample_count,
                layer_count: layout.sample_count,
            });
        source_views.push(
            unsafe { context.device.create_image_view(&info, None) }
                .map_err(|error| format!("create multisample source layer view: {error}"))?,
        );
    }
    objects.layer_views.extend(source_views.iter().copied());
    let image_infos = source_views
        .iter()
        .map(|view| {
            [vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        })
        .collect::<Vec<_>>();
    let writes = descriptor_sets
        .iter()
        .zip(&image_infos)
        .map(|(set, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(info)
        })
        .collect::<Vec<_>>();
    unsafe { context.device.update_descriptor_sets(&writes, &[]) };

    objects.command_pool = unsafe {
        context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family),
            None,
        )
    }
    .map_err(|error| format!("create multisample initializer command pool: {error}"))?;
    let command = context
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(objects.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .map_err(|error| format!("allocate multisample initializer command: {error}"))?[0];
    unsafe {
        context.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }
    .map_err(|error| format!("begin multisample initializer command: {error}"))?;
    // The staging image has the destination's format, so a combined format's layout moves
    // name both aspects: without `separateDepthStencilLayouts` a barrier of one aspect of a
    // combined depth/stencil image is invalid (VUID-VkImageMemoryBarrier-image-03320). Its
    // stencil is never copied or read.
    let staging_range = vk::ImageSubresourceRange {
        aspect_mask: attached,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: staging_layers,
    };
    let to_transfer = [vk::ImageMemoryBarrier::default()
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(objects.staging_image)
        .subresource_range(staging_range)];
    let staging_barrier = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(objects.staging_image)
        .subresource_range(staging_range)];
    let staging_copy = [vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: staging_layers,
        })
        .image_extent(vk::Extent3D {
            width: layout.width,
            height: layout.height,
            depth: 1,
        })];
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_transfer,
        );
        context.device.cmd_copy_buffer_to_image(
            command,
            staging_transfer.buffer,
            objects.staging_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &staging_copy,
        );
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &staging_barrier,
        );
        context.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::GRAPHICS,
            objects.pipeline,
        );
        for (framebuffer, descriptor_set) in objects.framebuffers.iter().zip(&descriptor_sets) {
            context.device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(objects.render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(scissors[0]),
                vk::SubpassContents::INLINE,
            );
            context.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                objects.pipeline_layout,
                0,
                &[*descriptor_set],
                &[],
            );
            context.device.cmd_draw(command, 3, 1, 0, 0);
            context.device.cmd_end_render_pass(command);
        }
        context.device.end_command_buffer(command)
    }
    .map_err(|error| format!("end multisample initializer command: {error}"))?;
    objects.fence = unsafe {
        context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|error| format!("create multisample initializer fence: {error}"))?;
    let command_buffers = [command];
    let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
    unsafe {
        context
            .device
            .queue_submit(context.queue, &submits, objects.fence)
            .map_err(|error| format!("submit multisample initializer: {error}"))?;
    }
    context.wait_for_fence(objects.fence, "Vulkan multisample initializer")
}

/// The initializer's shader for a texel of `format`'s class: a full-screen triangle whose
/// fragment, run per sample, fetches layer `SampleId` of a 2D-array source at its pixel and writes
/// it to its sample.
pub(super) fn multisample_initializer_spvasm(format: crate::case::TextureFormat) -> String {
    use crate::case::TextureFormat as F;
    let (component, four, output_pointer, output) = match format {
        // `Rg32Uint`: the case refuses a multisampled `ulong` texture before this.
        F::Rgba8Uint | F::R16Uint | F::Rgba16Uint | F::R32Uint | F::Rgba32Uint | F::Rg32Uint => {
            ("uint", "%v4uint", "%ptr_output_v4uint", "%color")
        }
        F::Rgba8Sint | F::R32Sint | F::Rgba16Sint | F::Rgba32Sint => {
            ("int", "%v4int", "%ptr_output_v4int", "%color")
        }
        F::R8Unorm
        | F::Rgba8Unorm
        | F::R16Float
        | F::Rg16Float
        | F::Rg32Float
        | F::Rgba16Float
        | F::R32Float
        | F::Rgba32Float
        | F::Depth32Float
        // Sampled-only: `initialize_multisample_image` refuses them before this.
        | F::R8Snorm
        | F::Rgba8Snorm
        | F::R16Unorm
        | F::Rgba16Unorm
        | F::R16Snorm
        | F::Rg16Snorm
        | F::Rgba16Snorm
        | F::Rgba8UnormSrgb
        | F::Bgra8Unorm
        | F::Bgra8UnormSrgb
        | F::R8UnormSrgb
        | F::Rg8Unorm
        | F::Rg8UnormSrgb
        | F::Rg8Snorm
        | F::Rg16Unorm
        | F::A8Unorm
        | F::B5g6r5Unorm
        | F::A1bgr5Unorm
        | F::Abgr4Unorm
        | F::Bgr5a1Unorm
        | F::Rgb10a2Unorm
        | F::Bgr10a2Unorm
        | F::Rg11b10Float
        | F::Rgb9e5Float => ("float", "%v4float", "%ptr_output_v4float", "%color"),
    };
    let depth = format == F::Depth32Float;
    let depth_execution = if depth {
        "OpExecutionMode %fragment DepthReplacing"
    } else {
        ""
    };
    let fragment_interfaces = if depth {
        "%frag_coord %sample_id %source %frag_depth"
    } else {
        "%frag_coord %sample_id %source %color"
    };
    let output_decorations = if depth {
        "OpDecorate %frag_depth BuiltIn FragDepth".to_string()
    } else {
        format!("OpDecorate {output} Location 0")
    };
    let output_variable = if depth {
        "%frag_depth = OpVariable %ptr_output_float Output".to_string()
    } else {
        format!("{output} = OpVariable {output_pointer} Output")
    };
    let fragment_store = if depth {
        "%red = OpCompositeExtract %float %texel 0\nOpStore %frag_depth %red".to_string()
    } else {
        format!("OpStore {output} %texel")
    };
    format!(
        r#"OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %vertex "vertex" %vertex_index %position
OpEntryPoint Fragment %fragment "fragment" {fragment_interfaces}
OpExecutionMode %fragment OriginUpperLeft
{depth_execution}
OpDecorate %vertex_index BuiltIn VertexIndex
OpDecorate %position BuiltIn Position
OpDecorate %frag_coord BuiltIn FragCoord
OpDecorate %sample_id BuiltIn SampleId
OpDecorate %sample_id Flat
OpDecorate %source DescriptorSet 0
OpDecorate %source Binding 0
{output_decorations}
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3int = OpTypeVector %int 3
%v4int = OpTypeVector %int 4
%v4uint = OpTypeVector %uint 4
%v4float = OpTypeVector %float 4
%fn = OpTypeFunction %void
%ptr_input_int = OpTypePointer Input %int
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_output_v4int = OpTypePointer Output %v4int
%ptr_output_v4uint = OpTypePointer Output %v4uint
%ptr_output_float = OpTypePointer Output %float
%image = OpTypeImage %{component} 2D 0 1 0 1 Unknown
%ptr_uniform_image = OpTypePointer UniformConstant %image
%vertex_index = OpVariable %ptr_input_int Input
%position = OpVariable %ptr_output_v4float Output
%frag_coord = OpVariable %ptr_input_v4float Input
%sample_id = OpVariable %ptr_input_int Input
%source = OpVariable %ptr_uniform_image UniformConstant
{output_variable}
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%float_n1 = OpConstant %float -1
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%float_3 = OpConstant %float 3
%vertex = OpFunction %void None %fn
%vertex_entry = OpLabel
%vertex_id = OpLoad %int %vertex_index
%is_one = OpIEqual %bool %vertex_id %int_1
%is_two = OpIEqual %bool %vertex_id %int_2
%x = OpSelect %float %is_one %float_3 %float_n1
%y = OpSelect %float %is_two %float_3 %float_n1
%vertex_position = OpCompositeConstruct %v4float %x %y %float_0 %float_1
OpStore %position %vertex_position
OpReturn
OpFunctionEnd
%fragment = OpFunction %void None %fn
%fragment_entry = OpLabel
%position_value = OpLoad %v4float %frag_coord
%fx = OpCompositeExtract %float %position_value 0
%fy = OpCompositeExtract %float %position_value 1
%ix = OpConvertFToS %int %fx
%iy = OpConvertFToS %int %fy
%sample = OpLoad %int %sample_id
%coordinate = OpCompositeConstruct %v3int %ix %iy %sample
%source_image = OpLoad %image %source
%texel = OpImageFetch {four} %source_image %coordinate
{fragment_store}
OpReturn
OpFunctionEnd
"#
    )
}

#[test]
fn a_combined_depth_stencil_format_moves_both_aspects_together() {
    let aspects = |format, aspect| format_aspects(format, aspect).as_raw();
    let depth = vk::ImageAspectFlags::DEPTH;
    let both = (vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL).as_raw();
    assert_eq!(aspects(vk::Format::D32_SFLOAT_S8_UINT, depth), both);
    assert_eq!(aspects(vk::Format::D24_UNORM_S8_UINT, depth), both);
    assert_eq!(aspects(vk::Format::D32_SFLOAT, depth), depth.as_raw());
    let color = vk::ImageAspectFlags::COLOR;
    assert_eq!(aspects(vk::Format::R32_SFLOAT, color), color.as_raw());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn multisample_initializers_are_vulkan_legal_for_each_component_class() {
    use crate::case::TextureFormat;

    for format in [
        TextureFormat::Rg32Float,
        TextureFormat::Rgba32Float,
        TextureFormat::Rgba32Uint,
        TextureFormat::Rgba32Sint,
        TextureFormat::Depth32Float,
    ] {
        let bytes = assemble_spirv(
            "multisample-initializer",
            &multisample_initializer_spvasm(format),
        )
        .unwrap();
        crate::spirv_tools::validate(&bytes, "multisample-initializer-test").unwrap();
    }
}
