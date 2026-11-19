//! Presenting: a drawable's texture over a swapchain's image, the consumer's acquire of it, and the
//! command buffer that presents it.
//!
//! No test has a window, so an image made as a swapchain makes its images stands in for one: a 2D
//! `B8G8R8A8_UNORM` image of one level and layer, optimal, of the usage a consumer's swapchain asks
//! (colour attachment, transfer source and destination, sampled), in the undefined layout, with
//! memory the test allocates and frees. A submission of the test's own on the presenting queue
//! stands in for the present: it waits on the command buffer's semaphore, as a present does, copies
//! the image out of the present layout, and leaves it in that layout, as the presentation engine
//! leaves an image it showed.
//!
//! - Under a device cfg, on every device of the cfg's kind, under the validation layer, for a
//!   render-target drawable (home: the attachment layout) and a shader-read one at an index (home:
//!   shader-read-only): three frames of acquire, a pass, present and the stand-in present. The
//!   first clears, the second loads and stores what the first left (the move home after a present
//!   starts in the present layout and keeps the texels), the third clears again. The stand-in
//!   reads each frame's texels in the present layout, which the layer checks against what the
//!   runtime recorded. A presenting command buffer is one submission, and its scheduled handler,
//!   which sleeps, has returned before its `wait_until_scheduled` does, so the present waits on a
//!   signal already submitted. The runtime never destroys the image: the test does, once the
//!   texture's views are gone.
//! - What a drawable's texture is not is refused: another descriptor, format or usage than the
//!   swapchain's images, an acquire or a present of a texture that is not a drawable's, a present
//!   of a drawable not acquired, a second acquire, and a second present; a command buffer dropped
//!   uncommitted leaves its drawable acquired for another present.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, within, Arena, PATIENCE};
    use crate::{
        Buffer, ClearColor, CommandBuffer, CommandBufferStatus, Device, Error, IndexClass,
        LoadAction, PixelFormat, Refusal, RenderPassDescriptor, StorageMode, StoreAction, Texture,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// The stand-in swapchain's image extent.
    const EXTENT: [u32; 2] = [8, 4];

    /// The usage a consumer's swapchain asks its images for.
    const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
            | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
            | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
            | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );

    /// A swapchain's create info, of `B8G8R8A8_UNORM` images of `usage`, as far as the runtime
    /// reads it: no surface is named.
    fn swapchain(usage: vk::ImageUsageFlags) -> vk::SwapchainCreateInfoKHR<'static> {
        vk::SwapchainCreateInfoKHR::default()
            .min_image_count(1)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D {
                width: EXTENT[0],
                height: EXTENT[1],
            })
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
    }

    /// The drawable's descriptor: a Private `BGRA8Unorm` 2D texture of the swapchain's extent.
    fn descriptor(usage: TextureUsage) -> TextureDescriptor {
        let mut d = TextureDescriptor::new(
            TextureType::Type2D,
            PixelFormat::BGRA8Unorm,
            EXTENT[0],
            EXTENT[1],
        );
        d.usage = usage;
        d.storage_mode = StorageMode::Private;
        d
    }

    /// An image made as a swapchain makes its images, of `info`, with memory of its own.
    struct StandIn {
        image: vk::Image,
        memory: vk::DeviceMemory,
    }

    impl StandIn {
        fn new(device: &Device, info: &vk::SwapchainCreateInfoKHR<'_>) -> StandIn {
            let raw = device.raw();
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(info.image_format)
                .extent(vk::Extent3D {
                    width: info.image_extent.width,
                    height: info.image_extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(info.image_array_layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(info.image_usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: a valid create info on the live device; the image is bound once to memory
            // of a type its requirements allow, and destroyed by `destroy`.
            unsafe {
                let image = (raw.device.create_image(&image_info, None)).expect("an image");
                let needs = raw.device.get_image_memory_requirements(image);
                let types =
                    (raw.instance).get_physical_device_memory_properties(raw.physical_device);
                let allowed = |i: &u32| needs.memory_type_bits & (1 << i) != 0;
                let local = |i: &u32| {
                    types.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                };
                let index = (0..types.memory_type_count)
                    .find(|i| allowed(i) && local(i))
                    .or_else(|| (0..types.memory_type_count).find(allowed))
                    .expect("a memory type for the image");
                let allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(needs.size)
                    .memory_type_index(index);
                let memory = (raw.device.allocate_memory(&allocate, None)).expect("memory");
                (raw.device.bind_image_memory(image, memory, 0)).expect("a bind");
                StandIn { image, memory }
            }
        }

        /// Destroys the image and frees its memory.
        ///
        /// # Safety
        ///
        /// Nothing uses the image: its texture's views are gone, and every submission that used
        /// it has completed.
        unsafe fn destroy(self, device: &Device) {
            let raw = device.raw();
            // SAFETY: the caller's contract.
            unsafe {
                raw.device.destroy_image(self.image, None);
                raw.device.free_memory(self.memory, None);
            }
        }
    }

    /// The consumer's present, stood in for on the presenting queue: a submission that waits on
    /// `ready`, as a present does, copies the stand-in's image out of the present layout into
    /// `out`, and moves it back to that layout, as the presentation engine leaves an image it
    /// showed; its fence waited for.
    fn shown(name: &str, device: &Device, stand_in: &StandIn, ready: vk::Semaphore, out: &Buffer) {
        let raw = device.raw();
        let d = raw.device;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |old, new, src: vk::PipelineStageFlags2, dst, access| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src)
                .dst_stage_mask(dst)
                .dst_access_mask(access)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(stand_in.image)
                .subresource_range(range)
        };
        // After the semaphore wait, which waits in every stage.
        let away = [barrier(
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        )];
        let back = [barrier(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COPY,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        )];
        let host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        let (target, offset) = out.device_range();
        let copy = [vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: EXTENT[0],
                height: EXTENT[1],
                depth: 1,
            })];
        // SAFETY: the pool, the command buffer and the fence are this call's, used by this thread
        // alone and destroyed once the fence is waited for; the image is the stand-in's, in the
        // present layout once the submission's wait is met; `out` holds the copy's bytes.
        unsafe {
            let pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(raw.queue_family_index);
            let pool = d.create_command_pool(&pool_info, None).expect("a pool");
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command = d
                .allocate_command_buffers(&allocate)
                .expect("a command buffer")[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            d.begin_command_buffer(command, &begin).expect("a begin");
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&away);
            d.cmd_pipeline_barrier2(command, &dependency);
            d.cmd_copy_image_to_buffer(
                command,
                stand_in.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                &copy,
            );
            let dependency = vk::DependencyInfo::default()
                .memory_barriers(&host)
                .image_memory_barriers(&back);
            d.cmd_pipeline_barrier2(command, &dependency);
            d.end_command_buffer(command).expect("an end");
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("a fence");
            let waits = [vk::SemaphoreSubmitInfo::default()
                .semaphore(ready)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            let submit = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&waits)
                .command_buffer_infos(&commands)];
            device
                .with_present_queue(|queue| d.queue_submit2(queue, &submit, fence))
                .expect("the stand-in present's submission");
            let waited = d.wait_for_fences(&[fence], true, PATIENCE.as_nanos() as u64);
            assert_eq!(
                waited,
                Ok(()),
                "{name}: the stand-in present waited on the presenting command buffer's semaphore"
            );
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
        }
    }

    /// The bytes of the image's texels as the stand-in present copied them into `out`.
    fn texels(out: &Buffer) -> Vec<u8> {
        let len = (EXTENT[0] * EXTENT[1] * 4) as usize;
        let contents = out.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive and at least `len` bytes long, and the copy into it is
        // complete and visible to the host.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec()
    }

    /// What a frame presents through: the drawable over the stand-in, the consumer's semaphore,
    /// and the buffer the stand-in present copies the texels into.
    struct Presenting<'a> {
        name: &'a str,
        device: &'a Device,
        drawable: &'a Texture,
        stand_in: &'a StandIn,
        ready: vk::Semaphore,
        out: &'a Buffer,
    }

    impl Presenting<'_> {
        /// One frame, the drawable acquired: a pass into it in `command_buffer` that clears it to
        /// `clear` or loads it, its present, with a scheduled handler that sleeps first, and the
        /// stand-in present, all complete.
        fn frame(&self, command_buffer: CommandBuffer, clear: Option<ClearColor>) {
            let name = self.name;
            let mut d = RenderPassDescriptor::default();
            let color = &mut d.color_attachments[0];
            color.texture = Some(self.drawable.clone());
            color.store_action = StoreAction::Store;
            match clear {
                Some(clear) => {
                    color.load_action = LoadAction::Clear;
                    color.clear_color = clear;
                }
                None => color.load_action = LoadAction::Load,
            }
            command_buffer
                .render_command_encoder(&d)
                .unwrap_or_else(|error| panic!("{name}: a pass into the drawable: {error}"))
                .end_encoding();
            let handled = Arc::new(AtomicBool::new(false));
            let mark = Arc::clone(&handled);
            command_buffer
                .add_scheduled_handler(move |_| {
                    thread::sleep(Duration::from_millis(100));
                    mark.store(true, Ordering::SeqCst);
                })
                .expect("a scheduled handler");
            // SAFETY: `ready` is a binary semaphore of the device, unsignalled with no signal
            // pending (the last stand-in present waited on it), and lives past the command buffer.
            unsafe { command_buffer.present_drawable(self.drawable, self.ready) }
                .unwrap_or_else(|error| panic!("{name}: a present: {error}"));
            command_buffer.commit().expect("a commit");
            command_buffer
                .wait_until_scheduled()
                .expect("a wait outside a handler");
            assert!(
                handled.load(Ordering::SeqCst),
                "{name}: wait_until_scheduled returned before the scheduled handler, which runs \
                 before the presenting command buffer's submission"
            );
            assert_ne!(
                command_buffer.status(),
                CommandBufferStatus::Error,
                "{name}: the presenting command buffer ran"
            );
            shown(name, self.device, self.stand_in, self.ready, self.out);
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
        }
    }

    /// Three frames into each of a render-target drawable and a shader-read one: each texel the
    /// stand-in present reads is the frame's, the second frame's those the first left; a
    /// presenting command buffer is one submission and one pass, with two barriers (its upload's,
    /// which moves the image home, and its end's, which moves it to the present layout) where the
    /// pass renders at home and four where it moves the image to the attachment layout and back.
    /// Its wait until scheduled returns after its scheduled handler. The image outlives its texture.
    #[test]
    fn every_device_renders_into_a_drawable_and_signals_its_present() {
        each_device("present", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let counters = device.counters();
            let raw = device.raw();
            // SAFETY: a valid create info on the live device; destroyed below, after every
            // submission that used it has completed.
            let ready = unsafe {
                raw.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .expect("a binary semaphore");
            let mut arena = Arena::new(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let first = queue.command_buffer().expect("a command buffer");
            first
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            first.commit().expect("a commit");
            finished(name, &first);
            let read_render = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            let first_clear = ClearColor::new(0.2, 0.6, 0.0, 1.0);
            let third_clear = ClearColor::new(1.0, 0.0, 0.4, 1.0);
            // `BGRA8Unorm`'s bytes: blue, green, red, alpha.
            let first_texels = [0u8, 153, 51, 255].repeat(32);
            let third_texels = [102u8, 0, 255, 255].repeat(32);
            for (usage, index, barriers) in [
                (TextureUsage::RENDER_TARGET, None, 2),
                (read_render, Some(4096), 4),
            ] {
                let info = swapchain(USAGE);
                let stand_in = StandIn::new(&device, &info);
                let textures = counters.gauges().textures;
                // SAFETY: the stand-in is made with `info` on the device, in the undefined
                // layout, and outlives the texture and its command buffers.
                let drawable = unsafe {
                    device.new_drawable_texture(&descriptor(usage), index, stand_in.image, &info)
                }
                .unwrap_or_else(|error| panic!("{name}: a drawable's texture: {error}"));
                assert_eq!(counters.gauges().textures, textures + 1, "{name}: counted");
                let out = arena.buffer();
                let presenting = Presenting {
                    name,
                    device: &device,
                    drawable: &drawable,
                    stand_in: &stand_in,
                    ready,
                    out: &out,
                };
                for (clear, expected) in [
                    (Some(first_clear), &first_texels),
                    (None, &first_texels),
                    (Some(third_clear), &third_texels),
                ] {
                    // SAFETY: the stand-in is the test's, in the present layout (or undefined
                    // before the first frame), and nothing reads it.
                    unsafe { drawable.drawable_acquired() }
                        .unwrap_or_else(|error| panic!("{name}: an acquire: {error}"));
                    settled(&device);
                    let before = counters.metrics();
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    presenting.frame(command_buffer, clear);
                    let after = counters.metrics();
                    assert_eq!(
                        (
                            after.submissions - before.submissions,
                            after.barriers - before.barriers,
                            after.passes - before.passes,
                        ),
                        (1, barriers, 1),
                        "{name} {usage:?}: submissions, barriers, passes of a presenting command \
                         buffer"
                    );
                    assert_eq!(
                        &texels(&out),
                        expected,
                        "{name} {usage:?}: the presented texels"
                    );
                }
                drop(drawable);
                within("the drawable's views to go", || {
                    counters.gauges().textures == textures
                });
                // SAFETY: the texture's views are gone, and its command buffers and the stand-in
                // presents have completed.
                unsafe { stand_in.destroy(&device) };
            }
            // SAFETY: every submission that signalled or waited on it has completed.
            unsafe { raw.device.destroy_semaphore(ready, None) };
            println!("{name}: three frames presented from each drawable");
        });
    }

    /// A refusal of `made`, which is to be refused.
    fn refusal(name: &str, made: Result<impl Sized, Error>) -> Refusal {
        match made {
            Err(Error::Refused(refusal)) => refusal,
            Err(error) => panic!("{name}: another error than a refusal: {error}"),
            Ok(_) => panic!("{name}: made, where it was to be refused"),
        }
    }

    /// Another descriptor, format or usage than the swapchain's images is refused, and so is an
    /// acquire or a present out of turn or of a texture that is not a drawable's; a command buffer
    /// dropped uncommitted leaves its drawable acquired.
    #[test]
    fn every_device_refuses_what_a_drawable_is_not() {
        each_device("present_refusals", |name, _, device| {
            let counters = device.counters();
            let info = swapchain(USAGE);
            let stand_in = StandIn::new(&device, &info);
            let render = TextureUsage::RENDER_TARGET;
            let made = |d: &TextureDescriptor, index, info: &vk::SwapchainCreateInfoKHR<'_>| {
                // SAFETY: the stand-in is made with `USAGE`'s info on the device, in the undefined
                // layout, and outlives every texture over it.
                unsafe { device.new_drawable_texture(d, index, stand_in.image, info) }
            };
            let mut levels = descriptor(render);
            levels.mipmap_level_count = 2;
            let mut wide = descriptor(render);
            wide.width = EXTENT[0] * 2;
            let mut array = descriptor(render);
            array.texture_type = TextureType::Type2DArray;
            for (what, d, info) in [
                ("two levels", levels, info),
                ("another extent", wide, info),
                ("a 2D array", array, info),
                ("two layers", descriptor(render), info.image_array_layers(2)),
            ] {
                assert_eq!(
                    refusal(name, made(&d, None, &info)),
                    Refusal::DrawableDescriptor,
                    "{name}: {what}"
                );
            }
            let mut rgba = descriptor(render);
            rgba.pixel_format = PixelFormat::RGBA8Unorm;
            assert_eq!(
                refusal(name, made(&rgba, None, &info)),
                Refusal::DrawableFormat {
                    pixel_format: PixelFormat::RGBA8Unorm,
                    held: vk::Format::R8G8B8A8_UNORM.as_raw(),
                    swapchain: vk::Format::B8G8R8A8_UNORM.as_raw(),
                },
                "{name}: another format"
            );
            let transfer = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
            let attachment = info.image_usage(USAGE & !transfer);
            assert_eq!(
                refusal(name, made(&descriptor(render), None, &attachment)),
                Refusal::DrawableUsage {
                    missing: transfer.as_raw()
                },
                "{name}: no transfer usage"
            );
            let unsampled = info.image_usage(USAGE & !vk::ImageUsageFlags::SAMPLED);
            let read_render = TextureUsage::SHADER_READ | render;
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .expect("a grant");
            assert_eq!(
                refusal(name, made(&descriptor(read_render), Some(4096), &unsampled)),
                Refusal::DrawableUsage {
                    missing: vk::ImageUsageFlags::SAMPLED.as_raw()
                },
                "{name}: shader-read usage over unsampled images"
            );
            assert_eq!(
                refusal(name, made(&descriptor(read_render), None, &info)),
                Refusal::TextureIndex { index: None },
                "{name}: shader-read usage without an index"
            );

            let drawable = made(&descriptor(render), None, &info)
                .unwrap_or_else(|error| panic!("{name}: a drawable's texture: {error}"));
            let plain = device
                .new_texture(&descriptor(render), None)
                .expect("a texture");
            let queue = device.new_command_queue(4).expect("a queue");
            let presenting = queue.command_buffer().expect("a command buffer");
            let no_semaphore = vk::Semaphore::null();
            // SAFETY: each present is refused, or its command buffer is dropped uncommitted: no
            // semaphore is signalled.
            unsafe {
                assert_eq!(
                    refusal(name, plain.drawable_acquired()),
                    Refusal::NotDrawable,
                    "{name}: an acquire of a texture that is not a drawable's"
                );
                assert_eq!(
                    refusal(name, presenting.present_drawable(&plain, no_semaphore)),
                    Refusal::NotDrawable,
                    "{name}: a present of a texture that is not a drawable's"
                );
                assert_eq!(
                    refusal(name, presenting.present_drawable(&drawable, no_semaphore)),
                    Refusal::DrawableNotAcquired,
                    "{name}: a present before the acquire"
                );
                drawable.drawable_acquired().expect("an acquire");
                assert_eq!(
                    refusal(name, drawable.drawable_acquired()),
                    Refusal::DrawableAcquired,
                    "{name}: a second acquire"
                );
                presenting
                    .present_drawable(&drawable, no_semaphore)
                    .expect("a present");
                let second = queue.command_buffer().expect("a command buffer");
                assert_eq!(
                    refusal(name, second.present_drawable(&drawable, no_semaphore)),
                    Refusal::DrawableNotAcquired,
                    "{name}: a second present"
                );
                drop(presenting);
                second
                    .present_drawable(&drawable, no_semaphore)
                    .expect("a present once the first command buffer is dropped uncommitted");
            }
            // The acquire's move home goes with the next submission, before the stand-in goes.
            let flush = queue.command_buffer().expect("a command buffer");
            flush
                .record_for_test(&[], |_, _| {})
                .expect("a command buffer's commands");
            flush.commit().expect("a commit");
            finished(name, &flush);
            drop((drawable, plain, flush));
            within("the textures' views to go", || {
                counters.gauges().textures == 0
            });
            // SAFETY: the texture's views are gone, and the submission that moved the image has
            // completed.
            unsafe { stand_in.destroy(&device) };
        });
    }
}
