//! The composite's frame in plain Vulkan on the runtime's own device (`composite_costs`' module
//! docs, `bare sets` and `bare pushed`): its own target, textures, uniform buffer, render passes,
//! pipelines and command buffer, recorded through the driver's commands as the device's
//! `vkGetDeviceProcAddr` gives them, and submitted to the queue's `VkQueue`
//! (`CommandQueue::with_vk_queue`).

use super::super::frame_costs::{instructions, step, thread_cpu, TEXTURES};
use super::{summarize, Composite, Frame, Measured, CHUNK, COUNTED, DRAWS, PASSES, RESET, SLOT};
use super::{TARGET_SIDE, TIMED, WARM};
use ash::vk;
use metal2vulkan_devtools::{fixtures_dir, spirv_assemble};
use std::ffi::c_void;
use std::time::Instant;

/// A shape's label: its descriptors bound as sets made once, or pushed.
pub(super) fn label(pushed: bool) -> &'static str {
    match pushed {
        false => "bare sets",
        true => "bare pushed",
    }
}

/// The side of each sampled texture.
const TEXTURE_SIDE: u32 = 64;
const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// What the bare frames draw with, all made once on the composite's device.
pub(super) struct Bare<'a> {
    composite: &'a Composite,
    device: ash::Device,
    /// `vkCmdPushDescriptorSetKHR`, where the device has it.
    push: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    memory: Vec<vk::DeviceMemory>,
    /// The target, then the textures; their views in the same order.
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    buffer: vk::Buffer,
    /// The render pass that clears the target, and the one that loads it.
    passes: [vk::RenderPass; 2],
    framebuffer: vk::Framebuffer,
    pool: vk::DescriptorPool,
    /// One set a texture, its buffers' ranges bound with dynamic offsets.
    sets: Vec<vk::DescriptorSet>,
    /// The sets' layout, pipeline layout and pipeline; then the pushed ones', where the device
    /// pushes.
    set_layouts: Vec<vk::DescriptorSetLayout>,
    layouts: Vec<vk::PipelineLayout>,
    pipelines: Vec<vk::Pipeline>,
    modules: [vk::ShaderModule; 2],
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'a> Bare<'a> {
    pub(super) fn new(composite: &'a Composite) -> Bare<'a> {
        let runtime = &composite.device;
        // SAFETY: the objects stay the runtime's; this destroys none of them, uses them while the
        // device lives, and submits only through `CommandQueue::with_vk_queue`.
        let handles = unsafe { runtime.raw() };
        let load = |name: &std::ffi::CStr| -> *const c_void {
            runtime
                .device_proc_addr(name)
                .map_or(std::ptr::null(), |function| function as *const c_void)
        };
        // SAFETY: every command loads from the device's own `vkGetDeviceProcAddr`.
        let device = unsafe { ash::Device::load_with(load, handles.device.handle()) };
        let push = (runtime.device_proc_addr(c"vkCmdPushDescriptorSetKHR"))
            .or_else(|| runtime.device_proc_addr(c"vkCmdPushDescriptorSet"))
            .map(|function| {
                // SAFETY: the device gave the command of this name, whose signature this is.
                unsafe {
                    std::mem::transmute::<
                        unsafe extern "system" fn(),
                        vk::PFN_vkCmdPushDescriptorSetKHR,
                    >(function)
                }
            });
        // SAFETY: the instance and physical device are the device's own.
        let properties = unsafe {
            (handles.instance).get_physical_device_memory_properties(handles.physical_device)
        };
        // SAFETY: from here on every call makes, binds or fills an object of this lane's own,
        // with valid create infos, on the device's live handle.
        unsafe {
            let mut memory = Vec::new();
            let mut allocate = |requirements: vk::MemoryRequirements,
                                flags: vk::MemoryPropertyFlags| {
                let kind = (0..properties.memory_type_count)
                    .find(|&kind| {
                        requirements.memory_type_bits & (1 << kind) != 0
                            && (properties.memory_types[kind as usize].property_flags)
                                .contains(flags)
                    })
                    .expect("a memory type");
                let info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(kind);
                let made = device.allocate_memory(&info, None).unwrap();
                memory.push(made);
                made
            };
            let image = |side: u32, usage: vk::ImageUsageFlags| {
                let info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D {
                        width: side,
                        height: side,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                device.create_image(&info, None).unwrap()
            };
            let mut images = vec![image(
                TARGET_SIDE as u32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )];
            for _ in 0..TEXTURES {
                images.push(image(
                    TEXTURE_SIDE,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ));
            }
            for &image in &images {
                let made = allocate(
                    device.get_image_memory_requirements(image),
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                device.bind_image_memory(image, made, 0).unwrap();
            }
            let color = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let views: Vec<vk::ImageView> = (images.iter())
                .map(|&image| {
                    let info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(FORMAT)
                        .subresource_range(color);
                    device.create_image_view(&info, None).unwrap()
                })
                .collect();
            let sampler = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.25);
            let sampler = device.create_sampler(&sampler, None).unwrap();
            // The uniform buffer: each slot's four corners (clip-space position, texture
            // coordinates) of a strip, and a tint 256 bytes past them.
            let info = vk::BufferCreateInfo::default()
                .size(CHUNK as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);
            let buffer = device.create_buffer(&info, None).unwrap();
            let host = allocate(
                device.get_buffer_memory_requirements(buffer),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            device.bind_buffer_memory(buffer, host, 0).unwrap();
            let mapped = device
                .map_memory(host, 0, CHUNK as u64, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<u8>();
            for slot in 0..CHUNK / SLOT {
                let (x, y) = (
                    (slot % 8) as f32 * 0.25 - 1.0,
                    (slot / 8 % 8) as f32 * 0.25 - 1.0,
                );
                let words: [f32; 16] = [
                    x,
                    y,
                    0.0,
                    0.0,
                    x + 0.2,
                    y,
                    1.0,
                    0.0,
                    x,
                    y + 0.2,
                    0.0,
                    1.0,
                    x + 0.2,
                    y + 0.2,
                    1.0,
                    1.0,
                ];
                let tint = [0.5f32; 4];
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                let tint: Vec<u8> = tint.iter().flat_map(|word| word.to_le_bytes()).collect();
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.add(slot * SLOT), 64);
                std::ptr::copy_nonoverlapping(tint.as_ptr(), mapped.add(slot * SLOT + 256), 16);
            }
            device.unmap_memory(host);
            let passes = [vk::AttachmentLoadOp::CLEAR, vk::AttachmentLoadOp::LOAD].map(|load| {
                let attachment = [vk::AttachmentDescription::default()
                    .format(FORMAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(load)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(match load {
                        vk::AttachmentLoadOp::CLEAR => vk::ImageLayout::UNDEFINED,
                        _ => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    })
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
                let reference = [vk::AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }];
                let subpass = [vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&reference)];
                // The earlier passes' color writes before this one's loads and writes.
                let dependency = [vk::SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dependency_flags: vk::DependencyFlags::empty(),
                }];
                let info = vk::RenderPassCreateInfo::default()
                    .attachments(&attachment)
                    .subpasses(&subpass)
                    .dependencies(&dependency);
                device.create_render_pass(&info, None).unwrap()
            });
            let attachments = [views[0]];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(passes[0])
                .attachments(&attachments)
                .width(TARGET_SIDE as u32)
                .height(TARGET_SIDE as u32)
                .layers(1);
            let framebuffer = device.create_framebuffer(&info, None).unwrap();
            let bindings = |uniform: vk::DescriptorType| {
                [
                    (0, uniform, vk::ShaderStageFlags::VERTEX),
                    (1, uniform, vk::ShaderStageFlags::FRAGMENT),
                    (
                        2,
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        vk::ShaderStageFlags::FRAGMENT,
                    ),
                ]
                .map(|(binding, kind, stages)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(kind)
                        .descriptor_count(1)
                        .stage_flags(stages)
                })
            };
            let dynamic = bindings(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC);
            let pushed = bindings(vk::DescriptorType::UNIFORM_BUFFER);
            let mut set_layouts = vec![device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&dynamic),
                    None,
                )
                .unwrap()];
            if push.is_some() {
                set_layouts.push(
                    device
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default()
                                .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                                .bindings(&pushed),
                            None,
                        )
                        .unwrap(),
                );
            }
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 2 * TEXTURES as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: TEXTURES as u32,
                },
            ];
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(TEXTURES as u32)
                .pool_sizes(&sizes);
            let pool = device.create_descriptor_pool(&info, None).unwrap();
            let layouts_of_sets = vec![set_layouts[0]; TEXTURES];
            let sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts_of_sets),
                )
                .unwrap();
            for (n, &set) in sets.iter().enumerate() {
                let corners = [vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: 64,
                }];
                let tint = [vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: 16,
                }];
                let texture = [vk::DescriptorImageInfo {
                    sampler,
                    image_view: views[1 + n],
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&corners),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(&tint),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&texture),
                ];
                device.update_descriptor_sets(&writes, &[]);
            }
            let layouts: Vec<vk::PipelineLayout> = (set_layouts.iter())
                .map(|set_layout| {
                    let set_layout = [*set_layout];
                    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layout);
                    device.create_pipeline_layout(&info, None).unwrap()
                })
                .collect();
            let module = |name: &str| {
                let words = spirv_assemble(&fixtures_dir().join("bare").join(name)).unwrap();
                let info = vk::ShaderModuleCreateInfo::default().code(&words);
                device.create_shader_module(&info, None).unwrap()
            };
            let modules = [
                module("composite_quad.vert.spvasm"),
                module("composite_layer.frag.spvasm"),
            ];
            let pipelines = (layouts.iter())
                .map(|&layout| pipeline(&device, modules, layout, passes[0]))
                .collect();
            let (command_pool, command_buffer, fence) =
                composite.queue.with_vk_queue(|_, family| {
                    let info = vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(family);
                    let command_pool = device.create_command_pool(&info, None).unwrap();
                    let info = vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    // Through the loaded table: no `VulkanContext` holds this device (the runtime's).
                    let mut command_buffer = vk::CommandBuffer::null();
                    let allocate = device.fp_v1_0().allocate_command_buffers;
                    allocate(device.handle(), &info, &mut command_buffer)
                        .result()
                        .unwrap();
                    let fence = device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .unwrap();
                    (command_pool, command_buffer, fence)
                });
            let bare = Bare {
                composite,
                device,
                push,
                memory,
                images,
                views,
                sampler,
                buffer,
                passes,
                framebuffer,
                pool,
                sets,
                set_layouts,
                layouts,
                pipelines,
                modules,
                command_pool,
                command_buffer,
                fence,
            };
            bare.clear_textures(color);
            bare
        }
    }

    /// Clear every texture and move it to `SHADER_READ_ONLY_OPTIMAL`, waited for.
    ///
    /// # Safety
    ///
    /// The command buffer and fence are idle.
    unsafe fn clear_textures(&self, color: vk::ImageSubresourceRange) {
        let device = &self.device;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let barrier = |image, from, to, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color)
        };
        // SAFETY: the caller's contract; every image is this lane's.
        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin)
                .unwrap();
            let textures = &self.images[1..];
            let to_transfer: Vec<_> = (textures.iter())
                .map(|&image| {
                    barrier(
                        image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            for &image in textures {
                let value = vk::ClearColorValue {
                    float32: [0.5, 0.25, 0.75, 1.0],
                };
                device.cmd_clear_color_image(
                    self.command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &value,
                    &[color],
                );
            }
            let to_read: Vec<_> = (textures.iter())
                .map(|&image| {
                    barrier(
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_read,
            );
            device.end_command_buffer(self.command_buffer).unwrap();
            self.submit();
        }
    }

    /// Submit the command buffer to the queue's `VkQueue` and wait for it.
    ///
    /// # Safety
    ///
    /// The command buffer is ended, and the fence idle.
    unsafe fn submit(&self) {
        let device = &self.device;
        let buffers = [self.command_buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
        // SAFETY: the caller's contract; the queue is the runtime's, submitted to through
        // `with_vk_queue`.
        unsafe {
            self.composite.queue.with_vk_queue(|queue, _| {
                device.queue_submit(queue, &submit, self.fence).unwrap();
            });
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();
            device.reset_fences(&[self.fence]).unwrap();
        }
    }

    /// One pass's binds and draws, the frame's draws so far `n`: the Vulkan commands recorded.
    fn pass_ops(&self, pushed: bool, n: &mut usize) -> usize {
        let (device, command_buffer) = (&self.device, self.command_buffer);
        let (layout, pipeline) = (
            self.layouts[usize::from(pushed)],
            self.pipelines[usize::from(pushed)],
        );
        let side = TARGET_SIDE as f32;
        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: side,
            height: side,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: TARGET_SIDE as u32,
                height: TARGET_SIDE as u32,
            },
        }];
        let mut commands = 3;
        // SAFETY: the command buffer records, inside a pass compatible with the pipeline's; every
        // object is this lane's, live.
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &viewport);
            device.cmd_set_scissor(command_buffer, 0, &scissor);
            for draw in 0..DRAWS {
                if draw > 0 && (*n).is_multiple_of(RESET) {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                    commands += 1;
                }
                let slot = (*n % (CHUNK / SLOT)) * SLOT;
                match self.push.filter(|_| pushed) {
                    Some(push) => {
                        let corners = [vk::DescriptorBufferInfo {
                            buffer: self.buffer,
                            offset: slot as u64,
                            range: 64,
                        }];
                        let tint = [vk::DescriptorBufferInfo {
                            buffer: self.buffer,
                            offset: slot as u64 + 256,
                            range: 16,
                        }];
                        let texture = [vk::DescriptorImageInfo {
                            sampler: self.sampler,
                            image_view: self.views[1 + *n % TEXTURES],
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        }];
                        let writes = [
                            vk::WriteDescriptorSet::default()
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&corners),
                            vk::WriteDescriptorSet::default()
                                .dst_binding(1)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&tint),
                            vk::WriteDescriptorSet::default()
                                .dst_binding(2)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(&texture),
                        ];
                        push(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            layout,
                            0,
                            writes.len() as u32,
                            writes.as_ptr(),
                        );
                    }
                    None => device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        0,
                        &[self.sets[*n % TEXTURES]],
                        &[slot as u32, slot as u32 + 256],
                    ),
                }
                device.cmd_draw(command_buffer, 4, 1, 0, 0);
                commands += 2;
                *n += 1;
            }
        }
        commands
    }

    /// Record one frame (descriptors pushed where `pushed`), counting the encoding thread's
    /// instructions in each step where `counted`, then submit it and wait for it.
    pub(super) fn frame(&self, pushed: bool, counted: bool) -> Frame {
        let (device, command_buffer) = (&self.device, self.command_buffer);
        let mut frame = Frame::default();
        let cpu = thread_cpu();
        let start = Instant::now();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is idle (the last frame was waited for).
        unsafe { device.begin_command_buffer(command_buffer, &begin) }.unwrap();
        let clear = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        }];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: TARGET_SIDE as u32,
                height: TARGET_SIDE as u32,
            },
        };
        let mut n = 0;
        for pass in 0..PASSES {
            let info = vk::RenderPassBeginInfo::default()
                .render_pass(self.passes[usize::from(pass > 0)])
                .framebuffer(self.framebuffer)
                .render_area(area)
                .clear_values(&clear[..usize::from(pass == 0)]);
            // SAFETY: the command buffer records outside a pass; the pass and framebuffer are
            // compatible.
            let ((), open) = step(counted, || unsafe {
                device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE)
            });
            frame.open.push(open);
            let (commands, ops) = step(counted, || self.pass_ops(pushed, &mut n));
            frame.ops.push((ops, commands));
            // SAFETY: inside the pass begun above.
            let ((), end) = step(counted, || unsafe {
                device.cmd_end_render_pass(command_buffer)
            });
            frame.end.push(end);
        }
        // SAFETY: recording, outside a pass.
        unsafe { device.end_command_buffer(command_buffer) }.unwrap();
        frame.wall = start.elapsed();
        frame.cpu = thread_cpu().saturating_sub(cpu);
        // SAFETY: ended, and the fence idle.
        unsafe { self.submit() };
        frame
    }

    /// The bare frames' measurement, descriptors pushed where `pushed`: `None` where the device
    /// has no push command.
    pub(super) fn measure(&self, pushed: bool) -> Option<Measured> {
        if pushed && self.push.is_none() {
            return None;
        }
        for _ in 0..WARM {
            self.frame(pushed, false);
        }
        let frames: Vec<Frame> = (0..TIMED).map(|_| self.frame(pushed, false)).collect();
        let counted = instructions::read().is_some().then(|| {
            (0..COUNTED)
                .map(|_| self.frame(pushed, true))
                .collect::<Vec<_>>()
        });
        Some(summarize(&frames, counted.as_deref()))
    }
}

/// The graphics pipeline of `modules` and `layout`, a 4-vertex strip blending premultiplied
/// colors into the target, its viewport and scissor dynamic, compatible with `pass`.
///
/// # Safety
///
/// Every handle is live and `device`'s.
unsafe fn pipeline(
    device: &ash::Device,
    modules: [vk::ShaderModule; 2],
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
) -> vk::Pipeline {
    let stages = [
        (vk::ShaderStageFlags::VERTEX, modules[0]),
        (vk::ShaderStageFlags::FRAGMENT, modules[1]),
    ]
    .map(|(stage, module)| {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(module)
            .name(c"main")
    });
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    // SAFETY: the caller's contract; the create info's pointers are live for the call.
    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
        .map_err(|(_, error)| error)
        .unwrap()[0]
}

impl Drop for Bare<'_> {
    fn drop(&mut self) {
        let device = &self.device;
        // SAFETY: every object is this lane's, and idle: each frame's submission was waited for.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            for &pipeline in &self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            for &module in &self.modules {
                device.destroy_shader_module(module, None);
            }
            for &layout in &self.layouts {
                device.destroy_pipeline_layout(layout, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
            for &layout in &self.set_layouts {
                device.destroy_descriptor_set_layout(layout, None);
            }
            device.destroy_framebuffer(self.framebuffer, None);
            for &pass in &self.passes {
                device.destroy_render_pass(pass, None);
            }
            device.destroy_buffer(self.buffer, None);
            device.destroy_sampler(self.sampler, None);
            for &view in &self.views {
                device.destroy_image_view(view, None);
            }
            for &image in &self.images {
                device.destroy_image(image, None);
            }
            for &memory in &self.memory {
                device.free_memory(memory, None);
            }
        }
    }
}
