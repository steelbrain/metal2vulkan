//! The tiling-cost check's draws (`tiling_check.rs`): the crate's own modules
//! (`bind::check_draw_modules`, one triangle over the viewport), a render pass that loads and
//! stores its one color in `GENERAL`, and two pipelines in it, each over the whole target: one
//! whose fragments fetch the texel at their own place of a sampled source of the target's size,
//! and one whose fragments write zero. Made once for the check's run, each arm's target and
//! source viewed for it, and released as they go.

use crate::device::Device;
use crate::error::{internal, vulkan, Error};
use crate::{exit_guard, release};
use ash::vk;
use metal2vulkan::bind::{check_draw_modules, CheckDrawModules};

/// The draws' objects for one run of the check: the modules, the source's set layout and pool,
/// the pipeline layout, the pass of the check's format and its two pipelines. Released as it goes.
pub(crate) struct TilingDraw {
    device: Device,
    vertex: vk::ShaderModule,
    zero: vk::ShaderModule,
    fetch: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    fetching: vk::Pipeline,
    zeroing: vk::Pipeline,
}

/// One arm's draws: its source's sampled view and set, its target's view and framebuffer in the
/// draws' pass. Released as it goes, once its submissions completed.
pub(crate) struct TilingTarget<'a> {
    draw: &'a TilingDraw,
    source: vk::ImageView,
    target: vk::ImageView,
    framebuffer: vk::Framebuffer,
    set: vk::DescriptorSet,
    extent: [u32; 2],
}

/// The arms a run draws in: the most sets its pool holds.
const ARMS: u32 = 2;

impl TilingDraw {
    /// The draws on `device` for 2D color images of `format`; inside a section of the exit guard.
    pub(crate) fn new(device: &Device, format: vk::Format) -> Result<TilingDraw, Error> {
        let modules =
            check_draw_modules(device.compile_device()).map_err(crate::device::device_error)?;
        let mut draw = TilingDraw {
            device: device.clone(),
            vertex: vk::ShaderModule::null(),
            zero: vk::ShaderModule::null(),
            fetch: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            layout: vk::PipelineLayout::null(),
            pass: vk::RenderPass::null(),
            fetching: vk::Pipeline::null(),
            zeroing: vk::Pipeline::null(),
        };
        draw.make(&modules, format)?;
        Ok(draw)
    }

    /// Make every object of the draws; those made before a failure go with the draws' drop.
    fn make(&mut self, modules: &CheckDrawModules, format: vk::Format) -> Result<(), Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let module = |words: &[u32]| {
            let info = vk::ShaderModuleCreateInfo::default().code(words);
            // SAFETY: the code is live SPIR-V words for the call.
            unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)
        };
        self.vertex = module(&modules.vertex)?;
        self.zero = module(&modules.fragment)?;
        self.fetch = module(&modules.fetch)?;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(CheckDrawModules::FETCH_BINDING)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `info` and its array are live for the call.
        self.set_layout =
            unsafe { raw.create_descriptor_set_layout(&info, None) }.map_err(vulkan)?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: ARMS,
        }];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(ARMS)
            .pool_sizes(&sizes);
        // SAFETY: `info` and its array are live for the call.
        self.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let layouts = [self.set_layout];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        // SAFETY: `info` and its array are live for the call.
        self.layout = unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)?;
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
        self.pass = unsafe { crate::creation::render_pass(raw, &info) }?;
        self.fetching = self.pipeline(self.fetch, "fetch")?;
        self.zeroing = self.pipeline(self.zero, "zero")?;
        Ok(())
    }

    /// A pipeline in the draws' pass whose fragment stage is `fragment`: the triangle over the
    /// viewport, without blending, writing every component, the viewport and scissor dynamic.
    fn pipeline(&self, fragment: vk::ShaderModule, what: &str) -> Result<vk::Pipeline, Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex)
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
            .render_pass(self.pass)
            .subpass(0);
        // SAFETY: every pointer in `info` outlives the call; no pipeline cache.
        let made =
            unsafe { raw.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        let made = made.map_err(|(_, error)| vulkan(error))?;
        let made =
            (made.first().copied()).ok_or_else(|| internal("the driver made no pipeline"))?;
        (self.device.shared()).name_pipeline(made, || {
            format!("metal2vulkan: tiling-cost check {what} draw")
        });
        Ok(made)
    }

    /// An arm's draws: from `source` into `target`, 2D color images of the draws' format and
    /// `extent`, `source` with sampled usage and `target` with color attachment usage, each of
    /// one level and layer.
    pub(crate) fn target(
        &self,
        (source, target): (vk::Image, vk::Image),
        format: vk::Format,
        extent: [u32; 2],
    ) -> Result<TilingTarget<'_>, Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let mut made = TilingTarget {
            draw: self,
            source: vk::ImageView::null(),
            target: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            set: vk::DescriptorSet::null(),
            extent,
        };
        let view = |image: vk::Image| {
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
            // SAFETY: the image is live, 2D, of `format`, of one level and layer.
            unsafe { crate::creation::image_view(raw, &info) }.map_err(vulkan)
        };
        made.source = view(source)?;
        made.target = view(target)?;
        let views = [made.target];
        let [width, height] = extent;
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(self.pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        // SAFETY: `info` and its array are live for the call; the view is of the pass's format.
        made.framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
        let layouts = [self.set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: `info` and its array are live for the call; the pool holds a set for each arm.
        let sets = unsafe { raw.allocate_descriptor_sets(&info) }.map_err(vulkan)?;
        made.set = (sets.first().copied()).ok_or_else(|| internal("the pool gave no set"))?;
        let images = [vk::DescriptorImageInfo::default()
            .image_view(made.source)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(made.set)
            .dst_binding(CheckDrawModules::FETCH_BINDING)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&images)];
        // SAFETY: the set is this arm's, used by no submission yet; the view is live.
        unsafe { raw.update_descriptor_sets(&writes, &[]) };
        Ok(made)
    }
}

impl Drop for TilingDraw {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard: the check's thread drops it after
        // its rounds' sections.
        let Some(_busy) = exit_guard::enter() else {
            // The process is exiting: the objects are left to its end.
            return;
        };
        let raw: &ash::Device = self.device.shared().vk();
        // SAFETY: every draw's submission completed (the check waits for each); a null handle is
        // a no-op; the pool's sets go with it.
        unsafe {
            release::object(raw, self.zeroing);
            release::object(raw, self.fetching);
            release::object(raw, self.pass);
            release::object(raw, self.layout);
            release::object(raw, self.pool);
            release::object(raw, self.set_layout);
            release::object(raw, self.fetch);
            release::object(raw, self.zero);
            release::object(raw, self.vertex);
        }
    }
}

impl TilingTarget<'_> {
    /// Record the arm's draws: a pass over the whole target that loads and stores it, each
    /// fragment fetching the source's texel at its place; the pass's writes before the next
    /// pass's loads; and a second such pass, each fragment writing zero.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass and only the caller records it; both
    /// images are live until it completes, and rest in `GENERAL`, the source written by nothing
    /// in it.
    pub(crate) unsafe fn record(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer) {
        let [width, height] = self.extent;
        let whole = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
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
            .render_pass(self.draw.pass)
            .framebuffer(self.framebuffer)
            .render_area(whole);
        let color = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let between = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )];
        let graphics = vk::PipelineBindPoint::GRAPHICS;
        // SAFETY: the caller's contract; the objects are this arm's and the draws', live until
        // they are dropped.
        unsafe {
            raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            raw.cmd_bind_pipeline(command_buffer, graphics, self.draw.fetching);
            raw.cmd_bind_descriptor_sets(
                command_buffer,
                graphics,
                self.draw.layout,
                0,
                &[self.set],
                &[],
            );
            raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
            raw.cmd_set_scissor(command_buffer, 0, &[whole]);
            raw.cmd_draw(command_buffer, 3, 1, 0, 0);
            raw.cmd_end_render_pass(command_buffer);
            raw.cmd_pipeline_barrier(
                command_buffer,
                color,
                color,
                vk::DependencyFlags::empty(),
                &between,
                &[],
                &[],
            );
            raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            raw.cmd_bind_pipeline(command_buffer, graphics, self.draw.zeroing);
            raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
            raw.cmd_set_scissor(command_buffer, 0, &[whole]);
            raw.cmd_draw(command_buffer, 3, 1, 0, 0);
            raw.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for TilingTarget<'_> {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(_busy) = exit_guard::enter() else {
            // The process is exiting: the objects are left to its end.
            return;
        };
        let raw: &ash::Device = self.draw.device.shared().vk();
        // SAFETY: the arm's submissions completed (the check waits for each); its set goes with
        // the draws' pool.
        unsafe {
            release::object(raw, self.framebuffer);
            release::object(raw, self.target);
            release::object(raw, self.source);
        }
    }
}
