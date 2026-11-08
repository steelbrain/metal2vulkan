//! The draw of the device's in-place check of render targets (`in_place_check.rs`): the crate's
//! own modules (`bind::check_draw_modules`, one triangle writing zero), and for each color format
//! a render pass that loads and stores its one color in `GENERAL` and the pipeline in it, made at
//! the format's first draw and kept for the check's class; drawn into an image in place through a
//! scissor of its last column ([`Texture::draw_in_place_column`](crate::Texture)), so that a
//! device whose draws write back the bytes past a row's last texels fails the check.

use crate::device::device_error;
use crate::error::{internal, vulkan, Error};
use crate::{exit_guard, release, Device};
use ash::vk;
use metal2vulkan::bind::check_draw_modules;

/// The check draw's objects for one class's rounds: its shader modules and pipeline layout, and
/// each color format's render pass and pipeline. Released as it goes.
pub(crate) struct CheckDraw {
    device: Device,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    layout: vk::PipelineLayout,
    formats: Vec<(vk::Format, vk::RenderPass, vk::Pipeline)>,
}

/// One draw's target: an image's view and its framebuffer in the format's pass, released as it
/// goes, once its submission completed.
pub(crate) struct CheckTarget<'a> {
    draw: &'a CheckDraw,
    pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    extent: [u32; 2],
}

impl CheckDraw {
    /// The draw's modules and layout on `device`.
    pub(crate) fn new(device: &Device) -> Result<CheckDraw, Error> {
        let modules = check_draw_modules(device.compile_device()).map_err(device_error)?;
        let raw: &ash::Device = device.shared().vk();
        let mut draw = CheckDraw {
            device: device.clone(),
            vertex: vk::ShaderModule::null(),
            fragment: vk::ShaderModule::null(),
            layout: vk::PipelineLayout::null(),
            formats: Vec::new(),
        };
        let module = |words: &[u32]| {
            let info = vk::ShaderModuleCreateInfo::default().code(words);
            // SAFETY: the code is live SPIR-V words for the call.
            unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)
        };
        draw.vertex = module(&modules.vertex)?;
        draw.fragment = module(&modules.fragment)?;
        let info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: `info` is live for the call.
        draw.layout = unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)?;
        Ok(draw)
    }

    /// The render pass and pipeline of `format`, made at its first draw.
    fn of_format(&mut self, format: vk::Format) -> Result<(vk::RenderPass, vk::Pipeline), Error> {
        if let Some(&(_, pass, pipeline)) = self.formats.iter().find(|(f, _, _)| *f == format) {
            return Ok((pass, pipeline));
        }
        let raw: &ash::Device = self.device.shared().vk();
        let general = vk::ImageLayout::GENERAL;
        let attachments = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
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
        let pass = unsafe { crate::creation::render_pass(raw, &info) }?;
        let pipeline = self.pipeline(pass);
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(error) => {
                // SAFETY: nothing used the pass.
                unsafe { release::object(raw, pass) };
                return Err(error);
            }
        };
        self.formats.push((format, pass, pipeline));
        Ok((pass, pipeline))
    }

    /// The draw's pipeline in `pass`: the triangle over the viewport, without blending, writing
    /// every component, the viewport and scissor dynamic.
    fn pipeline(&self, pass: vk::RenderPass) -> Result<vk::Pipeline, Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment)
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
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
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
            .layout(self.layout)
            .render_pass(pass)
            .subpass(0);
        // SAFETY: every pointer in `info` outlives the call; no pipeline cache.
        let made =
            unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        let made = made.map_err(|(_, error)| vulkan(error))?;
        made.first()
            .copied()
            .ok_or_else(|| internal("the driver made no pipeline"))
    }

    /// The target of a draw into `image`, a 2D color image of `format` and `extent` with color
    /// attachment usage: its view and framebuffer in the format's pass.
    pub(crate) fn target(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        extent: [u32; 2],
    ) -> Result<CheckTarget<'_>, Error> {
        let (pass, pipeline) = self.of_format(format)?;
        let raw: &ash::Device = self.device.shared().vk();
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: the image is live, 2D, of `format`, one level and layer at least.
        let view = unsafe { crate::creation::image_view(raw, &info) }.map_err(vulkan)?;
        let views = [view];
        let [width, height] = extent;
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        // SAFETY: `info` and its array are live for the call; the view is of the pass's format.
        let framebuffer = match unsafe { crate::creation::framebuffer(raw, &info) } {
            Ok(framebuffer) => framebuffer,
            Err(error) => {
                // SAFETY: nothing used the view.
                unsafe { release::object(raw, view) };
                return Err(error);
            }
        };
        Ok(CheckTarget {
            draw: self,
            pass,
            pipeline,
            view,
            framebuffer,
            extent,
        })
    }
}

impl Drop for CheckDraw {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard: the check's thread drops it after
        // its rounds' sections.
        let Some(_busy) = exit_guard::enter() else {
            // The process is exiting: the objects are left to its end.
            return;
        };
        let raw: &ash::Device = self.device.shared().vk();
        // SAFETY: every draw's submission completed (the check waits for each); a null handle is
        // a no-op.
        unsafe {
            for &(_, pass, pipeline) in &self.formats {
                release::object(raw, pipeline);
                release::object(raw, pass);
            }
            release::object(raw, self.layout);
            release::object(raw, self.fragment);
            release::object(raw, self.vertex);
        }
    }
}

impl CheckTarget<'_> {
    /// Record the draw: the pass over the whole image, loading and storing it, the triangle
    /// through a scissor of its last column.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass and only the caller records it; the
    /// image is live until it completes, and rests in `GENERAL`.
    pub(crate) unsafe fn record(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        let [width, height] = self.extent;
        let whole = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let column = vk::Rect2D {
            offset: vk::Offset2D {
                x: width.saturating_sub(1) as i32,
                y: 0,
            },
            extent: vk::Extent2D { width: 1, height },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.pass)
            .framebuffer(self.framebuffer)
            .render_area(whole);
        // SAFETY: the caller's contract; the objects are this draw's, live until it is dropped.
        unsafe {
            raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
            raw.cmd_set_scissor(command_buffer, 0, &[column]);
            raw.cmd_draw(command_buffer, 3, 1, 0, 0);
            raw.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for CheckTarget<'_> {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(_busy) = exit_guard::enter() else {
            // The process is exiting: the objects are left to its end.
            return;
        };
        let raw: &ash::Device = self.draw.device.shared().vk();
        // SAFETY: the draw's submission completed (the check waits for it).
        unsafe {
            release::object(raw, self.framebuffer);
            release::object(raw, self.view);
        }
    }
}
