//! A multisampled render pass's attachments on the engine render candidate
//! (`executor_contract::multisample`).
//!
//! An attachment of more than one sample is an image of the pass's sample count that the pass
//! attaches and a compute dispatch samples (`SAMPLED` usage; a color one `MUTABLE_FORMAT` too, so
//! an unsigned-integer view of its texel size can read its bits). It is filled sample by sample
//! by the multisample-literal initializer (`multisample::initialize_multisample_image`, its own
//! submission, which leaves the image in `GENERAL`), moved to its attachment layout before the
//! case's pass loads it, and read after the pass by a dispatch of one invocation per sample:
//! each fetches its sample (`OpImageFetch` with `Sample`) through the unsigned-integer view of
//! the texel size, through the depth view whose `f32` it bit-casts, or through the stencil view
//! whose unsigned value is the byte, and stores whole words at its place in the sample-minor
//! attachment layout ([`unpack_words`]). Nothing is resolved and nothing is copied: Vulkan copies
//! no multisampled image to a buffer. A stencil aspect, which no initializer writes sample by
//! sample, holds one value in every sample ([`uniform_stencil`]) and is cleared to it in a
//! submission of its own, which also leaves the image in `GENERAL`. A device that cannot fill the
//! samples (`sampleRateShading`, the initializer's formats) or fetch them
//! (`sampledImageIntegerSampleCounts`, `sampledImageDepthSampleCounts`,
//! `sampledImageStencilSampleCounts`, the view's format) is `NotSupported::NotEnabled`, never a
//! guess.
//!
//! A pass of one sample reaches none of this: its attachments are uploaded and read back by
//! copies as before ([`super::Transfer::Copy`]).

use super::{not_implemented, Aspect, Bound, DepthStencilTarget, Target, Targets, Transfer};
use crate::candidate::engine::graphics::ColorTarget;
use crate::candidate::platform::buffers::{create_host_buffer, HostBuffer};
use crate::candidate::platform::device::VulkanContext;
use crate::candidate::platform::graphics_pipeline;
use crate::candidate::platform::multisample::{assemble_spirv, initialize_multisample_image};
use crate::candidate::platform::pipeline::create_shader_module;
use crate::candidate::platform::plan_resources::{
    barrier, create_image, multisample_initializer_support,
};
use crate::case::{TextureFormat, TextureType, RASTER_SAMPLE_COUNTS};
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::multisample::{
    readback_bytes, sample_read, uniform_stencil, unpack_words, ReadbackView, SampleRead,
};
use crate::literal::{texture_layout, LiteralDepthStencil, TextureLayout};
use ash::vk;
use std::collections::BTreeSet;
use std::ffi::CString;

/// The readback of one multisampled attachment: the view the dispatch fetches its samples
/// through, and the host-visible words the dispatch writes.
pub(super) struct SampleReadback {
    /// The unsigned-integer view of a color attachment's texel size, or the depth view of a
    /// depth attachment; destroyed with the attachment ([`Targets`]'s `Drop`).
    pub(super) view: vk::ImageView,
    words: HostBuffer,
    read: SampleRead,
}

impl SampleReadback {
    /// The attachment's bytes, `dimensions` pixels of `samples` samples each, in the attachment
    /// layout.
    ///
    /// # Safety
    ///
    /// The submission that read the samples has completed.
    pub(super) unsafe fn bytes(
        &self,
        dimensions: [u32; 2],
        samples: u32,
    ) -> Result<Vec<u8>, String> {
        // SAFETY: the caller's contract.
        let mapped = unsafe { self.words.map()? };
        unpack_words(mapped.bytes, dimensions, samples, self.read)
    }
}

/// The Vulkan format of `view`.
pub(super) fn view_format(view: ReadbackView) -> vk::Format {
    match view {
        ReadbackView::R8Uint => vk::Format::R8_UINT,
        ReadbackView::R16Uint => vk::Format::R16_UINT,
        ReadbackView::R32Uint => vk::Format::R32_UINT,
        ReadbackView::Rg32Uint => vk::Format::R32G32_UINT,
        ReadbackView::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
    }
}

/// The Vulkan sample count of a pass of `samples` samples: one of the counts a case authors
/// ([`RASTER_SAMPLE_COUNTS`]). Any other is refused typed, as the case schema refuses it: the
/// Metal reference device rasterizes 1, 2, and 4 samples, so no other count has a Metal answer,
/// whatever this device offers.
fn sample_flags(samples: u32) -> Result<vk::SampleCountFlags, ExecutorError> {
    match samples {
        2 => Ok(vk::SampleCountFlags::TYPE_2),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        other => Err(not_implemented(format!(
            "a pass of {other} samples: a case authors {RASTER_SAMPLE_COUNTS:?} (the Metal \
             reference device rasterizes 1, 2, and 4)"
        ))),
    }
}

/// Whether the device can fill and read an attachment of `format` (its `aspect`), laid out as
/// `layout`, through a view of `fetched` of the same aspect: the initializer's needs
/// ([`multisample_initializer_support`]) and [`fetch_support`].
fn fill_and_fetch_support(
    context: &VulkanContext,
    label: &str,
    (format, aspect): (vk::Format, vk::ImageAspectFlags),
    fetched: vk::Format,
    layout: &TextureLayout,
) -> Result<(), ExecutorError> {
    multisample_initializer_support(context, label, format, aspect, layout)?;
    fetch_support(context, label, aspect, fetched, layout.sample_count)
}

/// Whether a shader can fetch every one of `samples` samples of an attachment's `aspect` through
/// a view of `fetched`: sampling of the view's format with optimal tiling, and the device's count
/// of samples a shader fetches from an image of its kind.
fn fetch_support(
    context: &VulkanContext,
    label: &str,
    aspect: vk::ImageAspectFlags,
    fetched: vk::Format,
    samples: u32,
) -> Result<(), ExecutorError> {
    let mut missing = Vec::new();
    // SAFETY: the physical device belongs to the live instance.
    let features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, fetched)
    }
    .optimal_tiling_features;
    if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
        missing.push(format!(
            "{label}: its readback view's format {} sampled with optimal tiling",
            fetched.as_raw()
        ));
    }
    // SAFETY: as above.
    let limits = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
    }
    .limits;
    let (counts, limit) = if aspect.contains(vk::ImageAspectFlags::DEPTH) {
        (
            limits.sampled_image_depth_sample_counts,
            "sampledImageDepthSampleCounts",
        )
    } else if aspect.contains(vk::ImageAspectFlags::STENCIL) {
        (
            limits.sampled_image_stencil_sample_counts,
            "sampledImageStencilSampleCounts",
        )
    } else {
        (
            limits.sampled_image_integer_sample_counts,
            "sampledImageIntegerSampleCounts",
        )
    };
    if !counts.contains(sample_flags(samples)?) {
        missing.push(format!(
            "{label}: {limit} with {samples} samples (the device's: {:#b}, bit n: 2^n samples)",
            counts.as_raw()
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(missing).into())
    }
}

/// A view of `image` in `format` over `aspect`, usable as `usage` (a readback view only samples).
fn create_view(
    context: &VulkanContext,
    label: &str,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    usage: Option<vk::ImageUsageFlags>,
) -> Result<vk::ImageView, ExecutorError> {
    let mut restricted = usage.map(|usage| vk::ImageViewUsageCreateInfo::default().usage(usage));
    let mut info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    if let Some(restricted) = restricted.as_mut() {
        info = info.push_next(restricted);
    }
    // SAFETY: `info` names the live image.
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutorError::Failed(format!("create {label} view: {error}")))
}

/// The image, memory, attachment view, and readback of one multisampled attachment.
struct Attachment {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback: SampleReadback,
}

/// A multisampled attachment of `format` (its `aspect`) holding `bytes` (each pixel's samples in
/// order, as `literal_format`), laid out as `layout`, read through a view of `fetched`, with
/// `usage` beside an attachment's own (a fetched color target's, [`Bound`]):
/// created, filled by the initializer, and given its readback words. Everything made is
/// destroyed on failure.
fn create_attachment(
    context: &VulkanContext,
    label: &str,
    (format, aspect): (vk::Format, vk::ImageAspectFlags),
    (literal_format, read, fetched): (TextureFormat, SampleRead, vk::Format),
    layout: TextureLayout,
    bytes: &[u8],
    usage: vk::ImageUsageFlags,
) -> Result<Attachment, ExecutorError> {
    fill_and_fetch_support(context, label, (format, aspect), fetched, &layout)?;
    let color = !aspect.contains(vk::ImageAspectFlags::DEPTH);
    let (flags, attachment) = if color {
        (
            vk::ImageCreateFlags::MUTABLE_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
    } else {
        (
            vk::ImageCreateFlags::empty(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    };
    let (image, memory) = create_image(
        context,
        label,
        &vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: layout.width,
                height: layout.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_flags(layout.sample_count)?)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(attachment | vk::ImageUsageFlags::SAMPLED | usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let mut views = Vec::new();
    let made = (|| {
        let view = create_view(context, label, image, format, aspect, None)?;
        views.push(view);
        let readback = create_view(
            context,
            &format!("{label} readback"),
            image,
            fetched,
            aspect,
            Some(vk::ImageUsageFlags::SAMPLED),
        )?;
        views.push(readback);
        initialize_multisample_image(
            context,
            image,
            format,
            aspect,
            layout,
            literal_format,
            bytes,
        )
        .map_err(ExecutorError::Failed)?;
        let length = readback_bytes([layout.width, layout.height], layout.sample_count, read)
            .map_err(ExecutorError::Failed)?;
        let words = create_host_buffer(
            context,
            &format!("{label} readback words"),
            &vec![0; length],
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .map_err(ExecutorError::Failed)?;
        Ok::<_, ExecutorError>((view, readback, words))
    })();
    match made {
        Ok((view, readback, words)) => Ok(Attachment {
            image,
            memory,
            view,
            readback: SampleReadback {
                view: readback,
                words,
                read,
            },
        }),
        Err(error) => {
            // SAFETY: every view was created above on the image, which nothing has used since
            // the initializer's submission completed.
            unsafe {
                for view in views {
                    context.device.destroy_image_view(view, None);
                }
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            Err(error)
        }
    }
}

/// Color target `index` of `format` holding `authored`, of `samples` samples per pixel, used as
/// `bound`: a color attachment, or one a framebuffer-fetch input also reads (`INPUT_ATTACHMENT`
/// usage) at the invocation's own sample.
pub(super) fn create_target(
    context: &VulkanContext,
    index: u32,
    format: vk::Format,
    authored: &ColorTarget,
    samples: u32,
    bound: Bound,
) -> Result<Target, ExecutorError> {
    let label = format!("multisampled color target {index}");
    if authored.samples != samples {
        return Err(ExecutorError::Failed(format!(
            "{label} holds {} samples per pixel in a pass of {samples}",
            authored.samples
        )));
    }
    let [width, height] = authored.dimensions;
    let layout = texture_layout(TextureType::D2Multisample, [width, height, 1], samples)
        .map_err(ExecutorError::Failed)?;
    let read = sample_read(authored.format).map_err(ExecutorError::Failed)?;
    let SampleRead::Color(view) = read else {
        return Err(ExecutorError::Failed(format!(
            "{label} has the depth format {:?}",
            authored.format
        )));
    };
    let expected = layout.width as usize
        * layout.height as usize
        * samples as usize
        * authored.format.bytes_per_pixel();
    if authored.bytes.len() != expected {
        return Err(ExecutorError::Failed(format!(
            "{label} authors {} bytes, not the {expected} of {width}x{height} pixels of {samples} \
             samples",
            authored.bytes.len()
        )));
    }
    let attachment = create_attachment(
        context,
        &label,
        (format, vk::ImageAspectFlags::COLOR),
        (authored.format, read, view_format(view)),
        layout,
        &authored.bytes,
        bound.usage(),
    )?;
    Ok(Target {
        image: attachment.image,
        memory: attachment.memory,
        view: attachment.view,
        transfer: Transfer::PerSample(Box::new(attachment.readback)),
        bound,
    })
}

/// The depth/stencil attachment of `format` holding each aspect `authored` gives, of `samples`
/// samples per pixel: the depth aspect filled sample by sample by the initializer, the stencil
/// aspect cleared to the one value its samples hold ([`clear_stencil`]), each read back sample by
/// sample through a view of its aspect. The image is a transfer destination too where it holds a
/// stencil, for the clear.
/// Everything made is destroyed on failure.
pub(super) fn create_depth_stencil(
    context: &VulkanContext,
    format: vk::Format,
    authored: &LiteralDepthStencil,
    samples: u32,
) -> Result<DepthStencilTarget, ExecutorError> {
    let label = "the multisampled depth/stencil attachment";
    if authored.samples != samples {
        return Err(ExecutorError::Failed(format!(
            "{label} holds {} samples per pixel in a pass of {samples}",
            authored.samples
        )));
    }
    let [width, height] = authored.dimensions;
    let layout = texture_layout(TextureType::D2Multisample, [width, height, 1], samples)
        .map_err(ExecutorError::Failed)?;
    let format_aspects = (if graphics_pipeline::has_depth(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::empty()
    }) | if graphics_pipeline::has_stencil(format) {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::empty()
    };
    let samples_of = width as usize * height as usize * samples as usize;
    for (aspect, bytes, texel, name) in [
        (vk::ImageAspectFlags::DEPTH, &authored.depth, 4, "depth"),
        (
            vk::ImageAspectFlags::STENCIL,
            &authored.stencil,
            1,
            "stencil",
        ),
    ] {
        let Some(bytes) = bytes else { continue };
        if !format_aspects.contains(aspect) {
            return Err(format!(
                "the case authors {name}, which VkFormat {} of the pass does not hold",
                format.as_raw()
            )
            .into());
        }
        if bytes.len() != samples_of * texel {
            return Err(ExecutorError::Failed(format!(
                "{label}'s {name} authors {} bytes, not the {} of {width}x{height} pixels of \
                 {samples} samples",
                bytes.len(),
                samples_of * texel
            )));
        }
    }
    let stencil = authored
        .stencil
        .as_deref()
        .map(uniform_stencil)
        .transpose()
        .map_err(ExecutorError::Failed)?;
    if authored.depth.is_none() && stencil.is_none() {
        return Err(ExecutorError::Failed(format!("{label} authors no aspect")));
    }
    if authored.depth.is_some() {
        fill_and_fetch_support(
            context,
            label,
            (format, vk::ImageAspectFlags::DEPTH),
            format,
            &layout,
        )?;
    }
    if stencil.is_some() {
        fetch_support(
            context,
            label,
            vk::ImageAspectFlags::STENCIL,
            format,
            samples,
        )?;
    }
    let transfers = if stencil.is_some() {
        vk::ImageUsageFlags::TRANSFER_DST
    } else {
        vk::ImageUsageFlags::empty()
    };
    let (image, memory) = create_image(
        context,
        label,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_flags(samples)?)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | transfers,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let mut views = Vec::new();
    let made = (|| {
        let view = create_view(context, label, image, format, format_aspects, None)?;
        views.push(view);
        let mut aspects = Vec::new();
        let mut layout_now = vk::ImageLayout::UNDEFINED;
        if let Some(depth) = &authored.depth {
            initialize_multisample_image(
                context,
                image,
                format,
                vk::ImageAspectFlags::DEPTH,
                layout,
                TextureFormat::Depth32Float,
                depth,
            )
            .map_err(ExecutorError::Failed)?;
            layout_now = vk::ImageLayout::GENERAL;
            aspects.push((vk::ImageAspectFlags::DEPTH, SampleRead::Depth));
        }
        if let Some(value) = stencil {
            clear_stencil(context, image, format_aspects, layout_now, value)
                .map_err(ExecutorError::Failed)?;
            aspects.push((vk::ImageAspectFlags::STENCIL, SampleRead::Stencil));
        }
        let mut read = Vec::new();
        for (aspect, how) in aspects {
            let readback = create_view(
                context,
                &format!("{label} readback"),
                image,
                format,
                aspect,
                Some(vk::ImageUsageFlags::SAMPLED),
            )?;
            views.push(readback);
            let length =
                readback_bytes([width, height], samples, how).map_err(ExecutorError::Failed)?;
            let words = create_host_buffer(
                context,
                &format!("{label} readback words"),
                &vec![0; length],
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .map_err(ExecutorError::Failed)?;
            read.push(Aspect {
                aspect,
                transfer: Transfer::PerSample(Box::new(SampleReadback {
                    view: readback,
                    words,
                    read: how,
                })),
            });
        }
        Ok::<_, ExecutorError>((view, read))
    })();
    match made {
        Ok((view, aspects)) => Ok(DepthStencilTarget {
            image,
            memory,
            view,
            format_aspects,
            aspects,
            dimensions: authored.dimensions,
            samples,
        }),
        Err(error) => {
            // SAFETY: every view was created above on the image, which nothing has used since
            // the last fill's submission completed.
            unsafe {
                for view in views {
                    context.device.destroy_image_view(view, None);
                }
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            Err(error)
        }
    }
}

/// Clear the stencil aspect of `image` (a depth/stencil image of `format_aspects`, now in
/// `from`) to `value` in a submission of its own, waited for, leaving every aspect in `GENERAL`
/// as the initializer leaves a filled one (every sample of a multisampled stencil holds one value,
/// [`uniform_stencil`]). A depth aspect the initializer filled keeps its samples: the transition
/// from `GENERAL` preserves them, and the clear touches the stencil alone.
fn clear_stencil(
    context: &VulkanContext,
    image: vk::Image,
    format_aspects: vk::ImageAspectFlags,
    from: vk::ImageLayout,
    value: u8,
) -> Result<(), String> {
    let device = &context.device;
    // SAFETY: the info is live for the call; the pool is destroyed below.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family),
            None,
        )
    }
    .map_err(|error| format!("create the stencil clear's command pool: {error}"))?;
    let mut fence = vk::Fence::null();
    let recorded = (|| {
        let command = context
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|error| format!("allocate the stencil clear's command buffer: {error}"))?[0];
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::STENCIL,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        // SAFETY: the command buffer is recording on the live device; the image is live, and
        // the earlier submission that filled its depth completed.
        unsafe {
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| format!("begin the stencil clear: {error}"))?;
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    image,
                    format_aspects,
                    1,
                    (vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE, from),
                    (vk::AccessFlags::TRANSFER_WRITE, vk::ImageLayout::GENERAL),
                )],
            );
            device.cmd_clear_depth_stencil_image(
                command,
                image,
                vk::ImageLayout::GENERAL,
                &vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: u32::from(value),
                },
                &[range],
            );
            device
                .end_command_buffer(command)
                .map_err(|error| format!("end the stencil clear: {error}"))?;
            fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| format!("create the stencil clear's fence: {error}"))?;
            let commands = [command];
            device
                .queue_submit(
                    context.queue,
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    fence,
                )
                .map_err(|error| format!("submit the stencil clear: {error}"))?;
        }
        context.wait_for_fence(fence, "Vulkan multisampled stencil clear")
    })();
    // SAFETY: the submission completed (or never happened); the pool frees its buffer.
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
    }
    recorded
}

/// Whether the device rasterizes `samples` samples in a pass without attachments
/// (`framebufferNoAttachmentsSampleCounts`).
pub(super) fn attachmentless_support(
    context: &VulkanContext,
    samples: u32,
) -> Result<(), ExecutorError> {
    // SAFETY: the physical device belongs to the live instance.
    let counts = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
    }
    .limits
    .framebuffer_no_attachments_sample_counts;
    if counts.contains(sample_flags(samples)?) {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(vec![format!(
            "framebufferNoAttachmentsSampleCounts with {samples} samples (the device's: {:#b})",
            counts.as_raw()
        )])
        .into())
    }
}

/// One multisampled attachment image's moves around the pass: every aspect of its format (a
/// depth/stencil image's layout covers both), the writes that filled it, and the access and
/// layout the pass keeps it in.
struct Moved {
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
    filled: vk::AccessFlags,
    attachment: (vk::AccessFlags, vk::ImageLayout),
}

/// One attachment aspect's dispatch: the set and pipeline that read it.
struct Read {
    set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
}

/// The per-sample readback of every multisampled attachment of a pass: one compute pipeline per
/// attachment over one layout (a sampled image at binding 0, the words at binding 1), destroyed
/// before the attachments.
pub(super) struct SampleReaders<'a> {
    device: &'a ash::Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    modules: Vec<vk::ShaderModule>,
    moved: Vec<Moved>,
    reads: Vec<Read>,
    dimensions: [u32; 2],
    samples: u32,
}

impl Drop for SampleReaders<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run (or is null, which each
        // destroy ignores), is destroyed once, and the submission that used it has completed (or
        // never happened). The pool frees its sets.
        unsafe {
            for read in &self.reads {
                self.device.destroy_pipeline(read.pipeline, None);
            }
            for module in &self.modules {
                self.device.destroy_shader_module(*module, None);
            }
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl<'a> SampleReaders<'a> {
    /// The readers of `targets`' multisampled attachments; `None` for a pass of one sample,
    /// whose attachments are read back by copies.
    pub(super) fn create(
        context: &'a VulkanContext,
        targets: &Targets<'_>,
    ) -> Result<Option<SampleReaders<'a>>, ExecutorError> {
        let depth = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let mut moved = Vec::new();
        let mut attachments = Vec::new();
        for target in targets.targets.values() {
            if let Transfer::PerSample(readback) = &target.transfer {
                let (layout, access, _) = target.attachment_use();
                moved.push(Moved {
                    image: target.image,
                    aspects: vk::ImageAspectFlags::COLOR,
                    filled: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    attachment: (access, layout),
                });
                attachments.push(readback);
            }
        }
        if let Some(target) = &targets.depth_stencil {
            let read: Vec<_> = target
                .aspects
                .iter()
                .filter_map(|aspect| match &aspect.transfer {
                    Transfer::PerSample(readback) => Some(readback),
                    Transfer::Copy(_) => None,
                })
                .collect();
            if !read.is_empty() {
                // The initializer's depth writes and the stencil's clear.
                moved.push(Moved {
                    image: target.image,
                    aspects: target.format_aspects,
                    filled: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE,
                    attachment: (depth, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                });
                attachments.extend(read);
            }
        }
        if attachments.is_empty() {
            return Ok(None);
        }
        let count = attachments.len() as u32;
        let mut readers = SampleReaders {
            device: &context.device,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pool: vk::DescriptorPool::null(),
            modules: Vec::new(),
            moved,
            reads: Vec::new(),
            dimensions: [targets.extent.width, targets.extent.height],
            samples: targets.samples,
        };
        let failed = |what: &str| {
            let what = what.to_owned();
            move |error: vk::Result| {
                ExecutorError::Failed(format!("create the sample readback's {what}: {error}"))
            }
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        // SAFETY: each info is live for its call; every object is destroyed by `Drop`.
        unsafe {
            readers.set_layout = context
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(failed("descriptor set layout"))?;
            let set_layouts = [readers.set_layout];
            readers.pipeline_layout = context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(failed("pipeline layout"))?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: count,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: count,
                },
            ];
            readers.pool = context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(count)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(failed("descriptor pool"))?;
        }
        let entry = CString::new("main").expect("a static entry name");
        for readback in attachments {
            let spv = assemble_spirv(
                "sample-readback",
                &readback_spvasm(readback.read, readers.dimensions, readers.samples),
            )
            .map_err(ExecutorError::Failed)?;
            let module = create_shader_module(context, &spv, "sample readback")
                .map_err(ExecutorError::Failed)?;
            readers.modules.push(module);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(readers.pipeline_layout);
            let set_layouts = [readers.set_layout];
            // SAFETY: each info is live for its call; the set is freed with the pool.
            let set = unsafe {
                context.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(readers.pool)
                        .set_layouts(&set_layouts),
                )
            }
            .map_err(failed("descriptor set"))?[0];
            let images = [vk::DescriptorImageInfo::default()
                .image_view(readback.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let buffers = [vk::DescriptorBufferInfo::default()
                .buffer(readback.words.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&images),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers),
            ];
            // SAFETY: the set, view, and buffer are live and no submission uses the set.
            unsafe { context.device.update_descriptor_sets(&writes, &[]) };
            // SAFETY: `info` is live for the call; the pipeline is destroyed by `Drop`.
            let pipeline = unsafe {
                context
                    .device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            }
            .map_err(|(_, error)| failed("pipeline")(error))?[0];
            readers.reads.push(Read { set, pipeline });
        }
        Ok(Some(readers))
    }

    /// Before the pass: move each attachment from the initializer's `GENERAL` to the layout the
    /// pass keeps it in, after the initializer's writes (an earlier submission on this queue).
    ///
    /// # Safety
    ///
    /// `command` is recording on the readers' device, outside a render pass.
    pub(super) unsafe fn record_initialized(
        &self,
        command: vk::CommandBuffer,
        attachment_stages: vk::PipelineStageFlags,
    ) {
        let barriers: Vec<vk::ImageMemoryBarrier<'_>> = self
            .moved
            .iter()
            .map(|moved| {
                barrier(
                    moved.image,
                    moved.aspects,
                    1,
                    (moved.filled, vk::ImageLayout::GENERAL),
                    moved.attachment,
                )
            })
            .collect();
        // SAFETY: the caller's contract; every image is live.
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                attachment_stages,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    /// After the pass: read every sample of every attachment into its words, and make the words
    /// visible to the host.
    ///
    /// # Safety
    ///
    /// `command` is recording on the readers' device, outside a render pass, after the pass that
    /// left each attachment in its attachment layout.
    pub(super) unsafe fn record_reads(
        &self,
        command: vk::CommandBuffer,
        attachment_stages: vk::PipelineStageFlags,
    ) {
        let to_shader: Vec<vk::ImageMemoryBarrier<'_>> = self
            .moved
            .iter()
            .map(|moved| {
                barrier(
                    moved.image,
                    moved.aspects,
                    1,
                    moved.attachment,
                    (
                        vk::AccessFlags::SHADER_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ),
                )
            })
            .collect();
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        let [width, height] = self.dimensions;
        // SAFETY: the caller's contract; every handle is live.
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                attachment_stages,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_shader,
            );
            for read in &self.reads {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    read.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[read.set],
                    &[],
                );
                // One invocation per sample, one per workgroup: no grid edge to guard.
                self.device
                    .cmd_dispatch(command, width, height, self.samples);
            }
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
        }
    }
}

/// The compute shader that writes every sample of a `width`-wide attachment of `samples` samples,
/// read as `read`, as words at its place in the attachment layout: workgroup `(x, y, s)` fetches
/// sample `s` of pixel `(x, y)` and stores its words from `((y * width + x) * samples + s) *
/// words`. A depth sample's `f32` is stored as its bits; a stencil sample, fetched through a view
/// of the stencil aspect as an unsigned integer, as its value.
fn readback_spvasm(read: SampleRead, [width, _]: [u32; 2], samples: u32) -> String {
    let words = read.words() as u32;
    // Each integer constant once.
    let constants: BTreeSet<u32> = [0, width, samples, words]
        .into_iter()
        .chain(0..words)
        .collect();
    let declarations = constants
        .iter()
        .map(|value| format!("%c{value} = OpConstant %uint {value}\n"))
        .collect::<String>();
    let (image, fetch) = match read {
        SampleRead::Color(_) | SampleRead::Stencil => (
            "%image = OpTypeImage %uint 2D 0 0 1 1 Unknown",
            "%texel = OpImageFetch %v4uint %loaded %xy Sample %s\n".to_string(),
        ),
        SampleRead::Depth => (
            "%image = OpTypeImage %float 2D 1 0 1 1 Unknown",
            "%fetched = OpImageFetch %v4float %loaded %xy Sample %s\n\
             %depth = OpCompositeExtract %float %fetched 0\n\
             %bits = OpBitcast %uint %depth\n\
             %texel = OpCompositeConstruct %v4uint %bits %bits %bits %bits\n"
                .to_string(),
        ),
    };
    let stores = (0..words)
        .map(|word| {
            format!(
                "%index{word} = OpIAdd %uint %base %c{word}\n\
                 %word{word} = OpCompositeExtract %uint %texel {word}\n\
                 %at{word} = OpAccessChain %ptr_word %words %c0 %index{word}\n\
                 OpStore %at{word} %word{word}\n"
            )
        })
        .collect::<String>();
    format!(
        r#"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %workgroup %source %words
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %workgroup BuiltIn WorkgroupId
OpDecorate %source DescriptorSet 0
OpDecorate %source Binding 0
OpDecorate %words DescriptorSet 0
OpDecorate %words Binding 1
OpDecorate %word_array ArrayStride 4
OpMemberDecorate %block 0 Offset 0
OpDecorate %block Block
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v3uint = OpTypeVector %uint 3
%v4uint = OpTypeVector %uint 4
%v4float = OpTypeVector %float 4
{image}
%ptr_image = OpTypePointer UniformConstant %image
%word_array = OpTypeRuntimeArray %uint
%block = OpTypeStruct %word_array
%ptr_block = OpTypePointer StorageBuffer %block
%ptr_word = OpTypePointer StorageBuffer %uint
%ptr_v3uint = OpTypePointer Input %v3uint
%workgroup = OpVariable %ptr_v3uint Input
%source = OpVariable %ptr_image UniformConstant
%words = OpVariable %ptr_block StorageBuffer
{declarations}%main = OpFunction %void None %fn
%entry = OpLabel
%id = OpLoad %v3uint %workgroup
%x = OpCompositeExtract %uint %id 0
%y = OpCompositeExtract %uint %id 1
%s = OpCompositeExtract %uint %id 2
%xy = OpCompositeConstruct %v2uint %x %y
%loaded = OpLoad %image %source
{fetch}%row = OpIMul %uint %y %c{width}
%pixel = OpIAdd %uint %row %x
%first = OpIMul %uint %pixel %c{samples}
%sample = OpIAdd %uint %first %s
%base = OpIMul %uint %sample %c{words}
{stores}OpReturn
OpFunctionEnd
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The readback shader of each kind of attachment is valid Vulkan SPIR-V, and stores each
    /// word of a texel at its place in the sample-minor layout.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn each_readback_shader_is_vulkan_legal_and_stores_the_attachment_layout() {
        for (read, words) in [
            (SampleRead::Color(ReadbackView::R8Uint), 1),
            (SampleRead::Color(ReadbackView::Rg32Uint), 2),
            (SampleRead::Color(ReadbackView::Rgba32Uint), 4),
            (SampleRead::Depth, 1),
            (SampleRead::Stencil, 1),
        ] {
            // Width 2 and 4 samples collide with word constants; each is declared once.
            let assembly = readback_spvasm(read, [2, 3], 4);
            assert_eq!(
                assembly.matches("%c2 = OpConstant").count(),
                1,
                "{assembly}"
            );
            assert_eq!(assembly.matches("OpStore %at").count(), words, "{assembly}");
            let bytes = assemble_spirv("sample-readback-test", &assembly).unwrap();
            crate::spirv_tools::validate(&bytes, "sample-readback-test")
                .unwrap_or_else(|error| panic!("{read:?}: {error}\n{assembly}"));
        }
    }

    /// Only the counts a case authors reach a Vulkan sample count; eight, which the Metal
    /// reference device does not rasterize, is refused typed like any other.
    #[test]
    fn only_an_authored_sample_count_is_drawn() {
        for samples in RASTER_SAMPLE_COUNTS {
            assert_eq!(sample_flags(samples).unwrap().as_raw(), samples);
        }
        for samples in [0, 1, 3, 8, 16] {
            assert!(
                matches!(
                    sample_flags(samples),
                    Err(ExecutorError::NotSupported(NotSupported::NotImplemented(_)))
                ),
                "{samples}"
            );
        }
    }

    /// Each color view is an unsigned-integer format of the texel size the readback unpacks.
    #[test]
    fn each_readback_view_is_the_unsigned_format_of_its_texel_size() {
        for (view, format) in [
            (ReadbackView::R8Uint, vk::Format::R8_UINT),
            (ReadbackView::R16Uint, vk::Format::R16_UINT),
            (ReadbackView::R32Uint, vk::Format::R32_UINT),
            (ReadbackView::Rg32Uint, vk::Format::R32G32_UINT),
            (ReadbackView::Rgba32Uint, vk::Format::R32G32B32A32_UINT),
        ] {
            assert_eq!(view_format(view).as_raw(), format.as_raw(), "{view:?}");
        }
    }
}
