//! Sample planes (its fourth part): the storage form of framebuffer fetch in a multisampled pass on
//! a device without multisampled storage images (ANV, MoltenVK).
//!
//! The compile layer reads and writes such a pass's fetched attachment as its sample planes
//! (`StorageAttachmentView::SamplePlanes`): a single-sample 2D-array image of one layer per sample,
//! addressed `(x, y, SampleId)` inside the pixel interlock. The render encoder keeps one per
//! fetched attachment for the pass ([`Planes`], made at the first draw that reads it), binds its
//! 2D-array storage view of every layer in the attachment's format (an 8-bit sRGB one's in the
//! unsigned-integer format of its texel size, a BGRA-order one's in its RGBA-order twin on ANV,
//! `framebuffer_fetch::storage_view_format`, to which the planes are made mutable) to each fetching
//! draw, and
//! moves the samples between it and the attachment around the draws that read it
//! (`render.rs`, [`PlanesStep`]):
//!
//! - **Into the planes** before a run of fetching draws: the split, a compute dispatch of
//!   `bind::planes_split_module` (one invocation per pixel and sample); at the pass's first
//!   instance the attachment's `Load` action is the split, its `Clear` a clear of the planes to
//!   the clear value (`vkCmdClearColorImage`, which converts it as the load operation would), and
//!   `DontCare` nothing: the planes start undefined, as Metal's texels do.
//! - **Out of the planes** where the attachment needs its samples again: before a later instance
//!   whose draws render it, and after the pass where Metal stores or resolves it. The merge, a
//!   sample-rate pass of `bind::planes_merge_modules` into the attachment, whose invocations each
//!   write their own sample. A pass whose last instance leaves out an attachment that resolves
//!   runs one more instance, of no draws, after the merge: the pass's resolve (`render.rs`).
//!   Where Metal stores the attachment `DontCare`, nothing merges after the pass, nor where it
//!   only resolves an extended-range attachment, which the crate-owned resolve reads from the
//!   planes (`color_resolve.rs`); a resolve into the attachment's sRGB twin reads the attachment,
//!   after the merge.
//!
//! The fetching draws' subpass leaves the attachment out; where it would then have no
//! attachment, it keeps the first such one, `DONT_CARE` and written by no pipeline, so that it
//! rasterizes the pass's samples on every device (`render_target.rs`, the sample planes' anchor):
//! its samples are the planes' there.
//!
//! Both copies read and write the bits of each texel through a view in the unsigned-integer
//! format of its size (`bind::PlanesCopy`), so every format crosses unchanged; `A8_UNORM`, whose
//! class no unsigned format shares, crosses as the `float`s of its own views
//! (`bind::PlanesCopy::Alpha8`), which keep each code. The image the
//! copies run on is the attachment's own where it is sampled and may be viewed so
//! (`framebuffer_fetch::planes_copies`, which a multisampled color render target the runtime
//! makes takes where it costs the image nothing), else a stand-in that is
//! (`attachment_stand_in.rs`): a scratch copied from and to the texture at the pass's ends, or a
//! memoryless texture's backing. The planes cover the render area, are the pass's own (held by
//! the command buffer until it completes), and every step runs between barriers from everything
//! before and to everything after, outside the pass's instances.
//!
//! A device that cannot run a copy refuses the first draw that fetches the attachment, typed
//! `Unsupported` (feature `FramebufferFetch`) naming what it lacks: a storage view of an extended
//! copy format (`shaderStorageImageExtendedFormats`), `sampleRateShading` for the merge, the
//! pass's sample count among `framebufferIntegerColorSampleCounts` (the merge renders the copy
//! format, an integer one; `framebufferColorSampleCounts` for `A8_UNORM`'s), or a format usage of
//! the attachment's or the copy format's (`A8_UNORM`'s storage write without a format among them,
//! which its split's declaration needs). A device
//! whose portability subset withholds `imageViewFormatReinterpretation` refuses it too, typed
//! `Unsupported` (feature `ViewFormatReinterpretation`, `view_plan.rs`), where the copy format's
//! components differ from the attachment's in count or width: every format but those of one 8-, 16-
//! or 32-bit component, or of two or four 32-bit ones. The kernels and the merge's pipelines are
//! made once per device and copy format (and sample count), at that first draw, and destroyed with
//! the device.
//!
//! The same two copies move a blit copy's samples between two multisampled color images where the
//! device's own copy converts float NaNs ([`BitsCopy`]): the split out of the source's level and
//! layer, the merge into the destination's over the region, through planes of their own. And
//! the merge alone writes a multisampled fill's samples (`blit_fill.rs`, [`BitsCopy::fill`]):
//! planes of the unsigned-integer format of the texel's size, each layer cleared to its sample's
//! bits, merged over the region into a multisampled image of that format.

use super::binding_view::VkView;
use super::command_buffer::CommandBuffer;
use super::copy_plan::ImageRegion;
use super::device::{device_error, lock};
use super::error::{internal, vulkan, Error};
use super::exit_guard;
use super::framebuffer_fetch;
use super::image_plan::{sample_count, ImagePlan};
use super::release;
use super::render_pass::{PassLayout, Target};
use super::render_target::PassShape;
use super::render_values::{LoadAction, StoreAction};
use super::resource::ResourceKey;
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::{ResourceOptions, StorageMode};
use super::view_plan::{keeps_components, reinterpretation_refused};
use super::Device;
use crate::device::functions::Vk;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{
    planes_merge_modules, planes_split_module, PlanesCopy, PlanesSplitModule,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::{FormatUsage, VkProperty};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType,
    ImageUsageFlags, ImageViewType,
};
use std::ops::Range;

/// One fetched attachment's sample planes for a pass (the module docs).
#[derive(Clone, Debug)]
pub(super) struct Planes {
    pub(super) texture: Texture,
    pub(super) copy: PlanesCopy,
    /// One layer per sample of the pass.
    pub(super) layers: u32,
    /// The format of the draws' storage view (`framebuffer_fetch::storage_view_format`): the
    /// attachment's, the planes' own, or an 8-bit sRGB one's unsigned-integer view or a BGRA-order
    /// one's RGBA-order twin, to which the planes are mutable.
    pub(super) view: Format,
    /// The render area's extent, which the planes cover.
    pub(super) extent: [u32; 2],
    /// The planes take transfer writes: a `Clear` load action clears them.
    pub(super) clearable: bool,
}

/// The refusal of a planes copy the device cannot run: it lacks `capability`.
fn lacks(capability: Capability) -> Error {
    Error::Unsupported(Unsupported::needs(Feature::FramebufferFetch, capability))
}

/// The refusal where the device lacks `usage` of `format`.
fn lacks_format(format: Format, usage: FormatUsage) -> Error {
    lacks(Capability::Format { format, usage })
}

impl Planes {
    /// The planes of `target`, a color attachment of `samples` samples, over the render area
    /// `extent`: its format, one layer per sample, storage (in the draws' storage view's format,
    /// `framebuffer_fetch::storage_view_format`, and the copy format) and sampled (in the copy
    /// format) usage, and transfer writes where the format takes them.
    /// Refused, `Unsupported`, where the device cannot run the copies on them or view the image
    /// in the copy format (the module docs); the copies' kernel and pipeline are made here, once
    /// per device.
    pub(super) fn new(
        maker: &CommandBuffer,
        target: &Target,
        samples: u32,
        extent: [u32; 2],
    ) -> Result<Planes, Error> {
        let device = maker.device();
        let format = target.format;
        // A format of a compatibility class of its own has no view whose texels the copies move
        // as bits; `A8_UNORM`'s move as its own views' floats (`PlanesCopy::for_planes`).
        let copy = PlanesCopy::for_planes(format).ok_or_else(|| {
            Error::Unsupported(Unsupported::not_implemented(Feature::FramebufferFetch))
        })?;
        // The copies view the planes and the attachment's image in the copy format, whose
        // components differ from the attachment's in count or width for most formats (`RG8Unorm`
        // as `R16_UINT`): a device that reinterprets no image's components makes no such view
        // (`view_plan.rs`), so it refuses the draw here.
        if !device.shared().view_reinterpretation && !keeps_components(format, copy.format()) {
            return Err(Error::Unsupported(reinterpretation_refused()));
        }
        // The copies' objects, made now: a device that cannot run them refuses the draw.
        device.planes_split(copy)?;
        device.planes_merge(copy, samples)?;
        let profile = device.compile_device().profile();
        // The copies read and write the attachment's image through its copy-format views. A
        // memoryless attachment's run on its backing, which the pass makes where it splits into
        // instances or merges the planes after its last (`render.rs`), or never run.
        let memoryless = target.texture.storage_mode() == StorageMode::Memoryless;
        if !memoryless && !framebuffer_fetch::planes_copies(target.texture.image_plan()?) {
            return Err(
                match framebuffer_fetch::planes_copies_gap(format, profile) {
                    Some(capability) => lacks(capability),
                    None => internal("sample planes of an image the copies cannot view"),
                },
            );
        }
        // The draws' storage view in its format (the attachment's, an 8-bit sRGB one's
        // unsigned-integer view, or a BGRA-order one's RGBA-order twin), the split's storage view
        // and the merge's sampled view in the copy format.
        let view = framebuffer_fetch::storage_view_format(profile, format);
        // The split writes `A8_UNORM` planes through a declaration without a format
        // (`PlanesSplitModule::requirements`), which the format's own bit must allow.
        let formatless = (!copy.moves_bits())
            .then_some((copy.format(), FormatUsage::StorageImageWriteWithoutFormat));
        for (format, usage) in [
            (view, FormatUsage::StorageImage),
            (copy.format(), FormatUsage::StorageImage),
            (copy.format(), FormatUsage::SampledImage),
        ]
        .into_iter()
        .chain(formatless)
        {
            if !profile.supports_format(format, usage) {
                return Err(lacks_format(format, usage));
            }
        }
        let mut usage = ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED;
        let transfer = device.shared().textures.facts(profile).optimal(format);
        let clearable = transfer.contains(FormatFeatureFlags2::TRANSFER_DST);
        if clearable {
            usage |= ImageUsageFlags::TRANSFER_DST;
        }
        // Viewed in its format, the copy format and the draws' view's; a usage the planes' own
        // format lacks (sampled, or storage where the draws view them in another format), which
        // those formats have, needs extended usage.
        let mut view_formats = vec![format];
        for other in [copy.format(), view] {
            if !view_formats.contains(&other) {
                view_formats.push(other);
            }
        }
        let mut flags = ImageCreateFlags::empty();
        if view_formats.len() > 1 {
            flags |= ImageCreateFlags::MUTABLE_FORMAT;
            let own = |usage| profile.supports_format(format, usage);
            if !own(FormatUsage::SampledImage) || !own(FormatUsage::StorageImage) {
                flags |= ImageCreateFlags::EXTENDED_USAGE;
            }
        } else {
            view_formats.clear();
        }
        let pixel_format = target.texture.pixel_format();
        let [width, height] = extent;
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            pixel_format,
            width: width.into(),
            height: height.into(),
            array_length: samples.into(),
            usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format,
            flags,
            image_type: ImageType::TYPE_2D,
            format,
            extent: [width, height, 1],
            levels: 1,
            layers: samples,
            samples: sample_count(1).ok_or_else(|| internal("one sample"))?,
            usage,
            view_formats,
            mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
        Ok(Planes {
            texture,
            copy,
            layers: samples,
            view,
            extent,
            clearable,
        })
    }

    /// A 2D-array view of every layer in `format`, serving `usage`.
    pub(super) fn view(
        &self,
        format: Format,
        usage: ImageUsageFlags,
    ) -> Result<vk::ImageView, Error> {
        self.texture.image_view(&VkView {
            view_type: ImageViewType::TYPE_2D_ARRAY,
            format,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: self.layers,
            usage,
        })
    }

    /// The storage view a fetching draw binds (`WriteSource::TileAttachment`'s `SamplePlanes`):
    /// every layer in the draws' storage view's format.
    pub(super) fn draw_view(&self) -> Result<vk::ImageView, Error> {
        self.view(self.view, ImageUsageFlags::STORAGE)
    }

    /// The use of every layer of the planes with `access`, a write where `write`.
    pub(super) fn image_use(&self, access: Access, write: bool) -> Result<Use<ResourceKey>, Error> {
        self.texture.image_use(0..1, 0..self.layers, access, write)
    }
}

/// The copy-format view of `target`'s level and layer, serving `usage`: the split's sampled view
/// of the multisampled image, the merge's color attachment view.
fn target_view(
    target: &Target,
    copy: PlanesCopy,
    usage: ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    target.texture.image_view(&VkView {
        view_type: ImageViewType::TYPE_2D,
        format: copy.format(),
        components: ComponentMapping::IDENTITY,
        aspect: ImageAspectFlags::COLOR,
        base_level: target.level,
        level_count: 1,
        base_layer: target.layer,
        layer_count: 1,
        usage,
    })
}

/// One step of a pass's sample planes, of the planes of a color attachment by its Metal index,
/// recorded before a render pass instance or after the pass's last (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlanesStep {
    /// Fill the planes from the attachment.
    Split(u32),
    /// Write the planes back into the attachment.
    Merge(u32),
    /// Clear the planes to the attachment's clear value (its `Clear` load action).
    Clear(u32),
}

/// A compute kernel of the split, the device's (the module docs).
#[derive(Clone, Copy, Debug)]
struct SplitKernel {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// The merge's shader modules and layouts, the device's.
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

/// The planes' objects one device has made (the module docs). The merge's shaders are one pair
/// for every bits copy and one for the copy of floats (`PlanesCopy::moves_bits`).
#[derive(Default)]
pub(super) struct PlanesObjects {
    split: Vec<(PlanesCopy, SplitKernel)>,
    merge: Vec<(bool, MergeShaders)>,
    merge_pipelines: Vec<((PlanesCopy, u32), MergePipeline)>,
}

impl PlanesObjects {
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

/// A shader module of `spirv`.
pub(super) fn shader_module(raw: &ash::Device, spirv: &[u32]) -> Result<vk::ShaderModule, Error> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: the code is live SPIR-V words for the call.
    unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)
}

/// A set layout of one image binding of `ty` per entry of `types`, numbered in order, for `stage`.
pub(super) fn set_layout(
    raw: &ash::Device,
    types: &[vk::DescriptorType],
    stage: vk::ShaderStageFlags,
) -> Result<vk::DescriptorSetLayout, Error> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = types
        .iter()
        .enumerate()
        .map(|(binding, &ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stage)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `info` and its array are live for the call.
    unsafe { raw.create_descriptor_set_layout(&info, None) }.map_err(vulkan)
}

/// A pipeline layout of the one set `set_layout`.
pub(super) fn pipeline_layout(
    raw: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, Error> {
    let set_layouts = [set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    // SAFETY: `info` and its array are live for the call.
    unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)
}

/// The descriptor types of the split's set, in binding order.
const SPLIT_SET: [vk::DescriptorType; 2] = [
    vk::DescriptorType::SAMPLED_IMAGE,
    vk::DescriptorType::STORAGE_IMAGE,
];

/// The descriptor types of the merge's set.
const MERGE_SET: [vk::DescriptorType; 1] = [vk::DescriptorType::SAMPLED_IMAGE];

impl Device {
    /// The split kernel of `copy`, made on first use.
    fn planes_split(&self, copy: PlanesCopy) -> Result<SplitKernel, Error> {
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().planes, |objects| {
            objects.split.iter().any(|(c, _)| *c == copy)
        });
        let mut objects = lock(&self.shared().planes);
        if let Some((_, kernel)) = objects.split.iter().find(|(c, _)| *c == copy) {
            return Ok(*kernel);
        }
        let module = planes_split_module(self.compile_device(), copy).map_err(device_error)?;
        if PlanesSplitModule::SOURCE_BINDING != 0 || PlanesSplitModule::PLANES_BINDING != 1 {
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
            kernel.pipeline = shared
                .make_compute_pipeline(Making::new("planes split").words(&module.spirv), info)?;
            Ok(())
        })();
        match made {
            Ok(()) => {
                objects.split.push((copy, kernel));
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

    /// The merge's shaders and its pipeline for `copy` at `samples`, made on first use.
    fn planes_merge(
        &self,
        copy: PlanesCopy,
        samples: u32,
    ) -> Result<(MergeShaders, MergePipeline), Error> {
        // The merge renders the attachment's view in the copy format, an integer one (or
        // `A8_UNORM` itself), at the pass's samples: a count every integer (or every) color
        // attachment takes, or the copy format's own, the alternatives link asks of a pipeline's
        // color attachment (`pipeline::graphics`, which the runtime does not import), in its order.
        let every = match copy.moves_bits() {
            true => VkProperty::FramebufferIntegerColorSampleCount(samples),
            false => VkProperty::FramebufferColorSampleCount(samples),
        };
        let alternatives = [
            every,
            VkProperty::ColorAttachmentSampleCount(copy.format(), samples),
        ];
        let enabled = &self.compile_device().profile().enabled().properties;
        if !alternatives.iter().any(|p| enabled.contains(p)) {
            return Err(Error::Unsupported(Unsupported::needs_any_of(
                Feature::FramebufferFetch,
                alternatives
                    .iter()
                    .map(|&p| vec![Capability::Property(p)])
                    .collect(),
            )));
        }
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn = crate::pipeline::budget::helper_turn(&self.shared().planes, |objects| {
            objects
                .merge_pipelines
                .iter()
                .any(|(key, _)| *key == (copy, samples))
        });
        let mut objects = lock(&self.shared().planes);
        let shared = self.shared();
        let raw = shared.vk();
        let bits = copy.moves_bits();
        let made = objects
            .merge
            .iter()
            .find(|(b, _)| *b == bits)
            .map(|(_, s)| *s);
        let shaders = match made {
            Some(shaders) => shaders,
            None => {
                let modules =
                    planes_merge_modules(self.compile_device(), copy).map_err(device_error)?;
                let mut shaders = MergeShaders {
                    vertex: vk::ShaderModule::null(),
                    fragment: vk::ShaderModule::null(),
                    set_layout: vk::DescriptorSetLayout::null(),
                    layout: vk::PipelineLayout::null(),
                    words: Making::new("planes merge modules")
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
                objects.merge.push((bits, shaders));
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
            let making = Making::new("planes merge")
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
fn merge_pass(raw: &ash::Device, copy: PlanesCopy, samples: u32) -> Result<vk::RenderPass, Error> {
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

/// The objects one pass's planes steps hold until its command buffer completes: their
/// descriptor sets' pool and the merges' framebuffers.
struct PassObjects {
    device: Device,
    pool: vk::DescriptorPool,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for PassObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them (a command buffer retains them until it completes);
        // the sets are freed with their pool, and a null handle is a no-op.
        unsafe {
            for &framebuffer in &self.framebuffers {
                release::object(raw, framebuffer);
            }
            release::object(raw, self.pool);
        }
    }
}

impl PassObjects {
    /// The objects of `splits` splits and `merges` merges on `device`: a pool of their sets.
    fn new(device: &Device, splits: u32, merges: u32) -> Result<PassObjects, Error> {
        let mut objects = PassObjects {
            device: device.clone(),
            pool: vk::DescriptorPool::null(),
            framebuffers: Vec::new(),
        };
        let sets = splits + merges;
        if sets > 0 {
            let mut sizes = vec![vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(sets)];
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
            objects.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        }
        Ok(objects)
    }

    /// A set of `set_layout` from the pool, its images written: each binding's in order, of
    /// its type in `types`, its view in `views`.
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
        // SAFETY: the pool holds a set of every step's layout (counted by `new`'s caller).
        let sets = unsafe { raw.allocate_descriptor_sets(&info) }.map_err(vulkan)?;
        let set = sets
            .first()
            .copied()
            .ok_or_else(|| internal("the driver allocated no set"))?;
        for (binding, (&ty, &view)) in types.iter().zip(views).enumerate() {
            let info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // SAFETY: the set was just allocated and no command uses it; the view is live (the
            // command buffer retains its texture).
            unsafe {
                raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(ty)
                        .image_info(&info)],
                    &[],
                )
            };
        }
        Ok(set)
    }

    /// The split of `target`'s samples into `planes`, its uses pushed onto `uses`.
    fn split(
        &self,
        target: &Target,
        planes: &Planes,
        uses: &mut Vec<Use<ResourceKey>>,
    ) -> Result<Recorded, Error> {
        let kernel = self.device.planes_split(planes.copy)?;
        let source = target_view(target, planes.copy, ImageUsageFlags::SAMPLED)?;
        let into = planes.view(planes.copy.format(), ImageUsageFlags::STORAGE)?;
        let set = self.set(kernel.set_layout, &SPLIT_SET, &[source, into])?;
        let stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let read = Access::new(stage, vk::AccessFlags2::SHADER_SAMPLED_READ);
        let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        uses.push(target.texture.image_use(
            target.level..target.level + 1,
            target.layer..target.layer + 1,
            read,
            false,
        )?);
        uses.push(planes.image_use(write, true)?);
        let local = PlanesSplitModule::LOCAL_SIZE;
        let [width, height] = planes.extent;
        Ok(Recorded::Split {
            kernel,
            set,
            groups: [width.div_ceil(local), height.div_ceil(local), planes.layers],
        })
    }

    /// The merge of `planes` into `target` over `area`, its uses pushed onto `uses`.
    fn merge(
        &mut self,
        planes: &Planes,
        target: &Target,
        area: vk::Rect2D,
        uses: &mut Vec<Use<ResourceKey>>,
    ) -> Result<Recorded, Error> {
        let (shaders, merge) = self.device.planes_merge(planes.copy, planes.layers)?;
        let from = planes.view(planes.copy.format(), ImageUsageFlags::SAMPLED)?;
        let set = self.set(shaders.set_layout, &MERGE_SET, &[from])?;
        let into = target_view(target, planes.copy, ImageUsageFlags::COLOR_ATTACHMENT)?;
        let [width, height] = planes.extent;
        let views = [into];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(merge.render_pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        let raw = self.device.shared().vk();
        // SAFETY: `info` and its array are live for the call; the view is live.
        let framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
        self.framebuffers.push(framebuffer);
        let read = Access::new(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let write = Access::new(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        uses.push(planes.image_use(read, false)?);
        uses.push(target.texture.image_use(
            target.level..target.level + 1,
            target.layer..target.layer + 1,
            write,
            true,
        )?);
        Ok(Recorded::Merge {
            shaders,
            merge,
            framebuffer,
            set,
            extent: planes.extent,
            area,
        })
    }
}

/// One recorded step (the module docs).
enum Recorded {
    Split {
        kernel: SplitKernel,
        set: vk::DescriptorSet,
        groups: [u32; 3],
    },
    Merge {
        shaders: MergeShaders,
        merge: MergePipeline,
        framebuffer: vk::Framebuffer,
        set: vk::DescriptorSet,
        extent: [u32; 2],
        /// The render area: the whole extent in a pass, a copy's region in a [`BitsCopy`].
        area: vk::Rect2D,
    },
    Clear {
        image: vk::Image,
        value: vk::ClearColorValue,
        layers: Range<u32>,
    },
}

/// A pass's planes steps, planned before the pass records: the steps of each position (one
/// before each render pass instance, then one after the last), their uses, and the objects
/// they hold, which go with them (the pass's steps hold them until its command buffer
/// completes, `render/pass_steps.rs`).
pub(super) struct PlanesCommands {
    positions: Vec<Vec<Recorded>>,
    pub(super) uses: Vec<Use<ResourceKey>>,
    _objects: PassObjects,
}

/// The planes steps of a pass of `layout`, its final layout, whose render pass instances have
/// `shapes` in order: [`ordered_steps`] of its attachments with planes.
pub(super) fn steps(layout: &PassLayout, shapes: &[&PassShape]) -> Vec<Vec<PlanesStep>> {
    let attachments: Vec<PlanesAttachment> = layout
        .colors
        .iter()
        .enumerate()
        .filter_map(|(index, color)| {
            let color = color.as_ref().filter(|c| c.planes.is_some())?;
            Some(PlanesAttachment {
                index: index as u32,
                load: color.load,
                store: color.store,
                planes_resolve: color
                    .resolve
                    .as_ref()
                    .is_some_and(|r| r.shader.is_some() && !r.twin),
            })
        })
        .collect();
    let storages: Vec<&[u32]> = shapes.iter().map(|shape| &shape.storage[..]).collect();
    ordered_steps(&attachments, &storages)
}

/// A color attachment with sample planes, as [`ordered_steps`] reads it.
#[derive(Clone, Copy, Debug)]
pub(super) struct PlanesAttachment {
    /// Its Metal index.
    pub(super) index: u32,
    pub(super) load: LoadAction,
    pub(super) store: StoreAction,
    /// It resolves by the crate-owned resolve from the planes where they hold its samples as the
    /// pass ends (`color_resolve.rs`): an extended-range attachment's. A resolve into the
    /// attachment's sRGB twin reads the attachment itself, so its planes merge back first.
    pub(super) planes_resolve: bool,
}

/// The planes steps of `attachments` in a pass whose render pass instances leave out of the
/// draws' subpass the color indices `storages`, in order: those before each instance, then those
/// after the last (the module docs). An attachment's planes hold its samples while an instance
/// leaves it out: before the first instance that does, the split (at the pass's first instance,
/// its load action on the planes); before the next that includes it, the merge; and after the
/// last instance, the merge where Metal keeps the samples (`Store`, `StoreAndMultisampleResolve`)
/// or a resolve reads the attachment (`MultisampleResolve` without the crate-owned resolve from
/// the planes, `color_resolve.rs`).
pub(super) fn ordered_steps(
    attachments: &[PlanesAttachment],
    storages: &[&[u32]],
) -> Vec<Vec<PlanesStep>> {
    let mut positions = Vec::with_capacity(storages.len() + 1);
    let mut before: Option<&[u32]> = None;
    for &storage in storages {
        let mut steps = Vec::new();
        for attachment in attachments {
            let index = attachment.index;
            let now = storage.contains(&index);
            match (before.map(|b| b.contains(&index)), now) {
                (None, true) => match attachment.load {
                    LoadAction::Load => steps.push(PlanesStep::Split(index)),
                    LoadAction::Clear => steps.push(PlanesStep::Clear(index)),
                    LoadAction::DontCare => {}
                },
                (Some(false), true) => steps.push(PlanesStep::Split(index)),
                (Some(true), false) => steps.push(PlanesStep::Merge(index)),
                _ => {}
            }
        }
        positions.push(steps);
        before = Some(storage);
    }
    let after = attachments
        .iter()
        .filter(|attachment| {
            let merged = match attachment.store {
                StoreAction::DontCare => false,
                StoreAction::MultisampleResolve => !attachment.planes_resolve,
                _ => true,
            };
            before.is_some_and(|last| last.contains(&attachment.index)) && merged
        })
        .map(|attachment| PlanesStep::Merge(attachment.index))
        .collect();
    positions.push(after);
    positions
}

/// The color attachment of index `index` of `layout`, and its planes.
fn planes_of(layout: &PassLayout, index: u32) -> Result<(&Target, &Planes), Error> {
    let color = layout
        .colors
        .get(index as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| internal("sample planes of an attachment the pass does not have"))?;
    let planes = color
        .planes
        .as_ref()
        .ok_or_else(|| internal("a planes step of an attachment without planes"))?;
    Ok((&color.target, planes))
}

/// Refuse (internal) a copy on `target`'s image where the copies cannot view it: a memoryless
/// attachment's planes copy only into and out of its backing (`Planes::new`).
fn copies_run_on(target: &Target) -> Result<(), Error> {
    if framebuffer_fetch::planes_copies(target.texture.image_plan()?) {
        Ok(())
    } else {
        Err(internal(
            "a sample planes copy on an image the copies cannot view",
        ))
    }
}

impl PlanesCommands {
    /// The commands of `positions`' steps (each instance's, then the pass's after the last) over
    /// `layout`, the pass's final layout; `None` where there is none.
    pub(super) fn plan(
        device: &Device,
        layout: &PassLayout,
        positions: &[Vec<PlanesStep>],
    ) -> Result<Option<PlanesCommands>, Error> {
        let steps = positions.iter().flat_map(|steps| steps.iter());
        let (mut splits, mut merges) = (0u32, 0u32);
        for step in steps.clone() {
            match step {
                PlanesStep::Split(_) => splits += 1,
                PlanesStep::Merge(_) => merges += 1,
                PlanesStep::Clear(_) => {}
            }
        }
        if steps.clone().next().is_none() {
            return Ok(None);
        }
        let mut objects = PassObjects::new(device, splits, merges)?;
        let mut uses = Vec::new();
        let mut planned = Vec::with_capacity(positions.len());
        for steps in positions {
            let mut recorded = Vec::with_capacity(steps.len());
            for &step in steps.iter() {
                match step {
                    PlanesStep::Split(index) => {
                        let (target, planes) = planes_of(layout, index)?;
                        copies_run_on(target)?;
                        recorded.push(objects.split(target, planes, &mut uses)?);
                    }
                    PlanesStep::Merge(index) => {
                        let (target, planes) = planes_of(layout, index)?;
                        copies_run_on(target)?;
                        let area = whole(planes.extent);
                        recorded.push(objects.merge(planes, target, area, &mut uses)?);
                    }
                    PlanesStep::Clear(index) => {
                        let (_, planes) = planes_of(layout, index)?;
                        if !planes.clearable {
                            return Err(internal("a clear of planes without transfer writes"));
                        }
                        let color = layout
                            .colors
                            .get(index as usize)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| internal("a clear of an attachment the pass lacks"))?;
                        let write = Access::new(
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                        );
                        uses.push(planes.image_use(write, true)?);
                        recorded.push(Recorded::Clear {
                            image: planes.texture.raw_image()?,
                            value: color.clear,
                            layers: 0..planes.layers,
                        });
                    }
                }
            }
            planned.push(recorded);
        }
        Ok(Some(PlanesCommands {
            positions: planned,
            uses,
            _objects: objects,
        }))
    }

    /// Record the steps of `position` (an instance's index, or the instances' count for the pass's
    /// after) into `command_buffer`, each between barriers from everything before and to
    /// everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// planes, the attachments and these commands' objects are live until it completes.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        position: usize,
    ) {
        let Some(steps) = self.positions.get(position) else {
            return;
        };
        // SAFETY: the caller's contract.
        unsafe { record_steps(raw, command_buffer, steps) };
    }
}

/// Copies of regions between two multisampled color images by the sample planes' two copies,
/// the split out of the source and the merge into the destination, through the unsigned-integer
/// view of the texel size of both, so every sample's bits cross unchanged: where the device's own
/// multisampled copy converts float NaNs (`nan_clear::converting_copy`); or the merge alone, a
/// multisampled fill's ([`BitsCopy::fill`]). Held until the command buffer completes: the
/// planes, the steps, their uses, and the objects the steps hold.
pub(super) struct BitsCopy {
    _planes: Vec<Planes>,
    steps: Vec<Recorded>,
    pub(super) uses: Vec<Use<ResourceKey>>,
    _objects: PassObjects,
}

/// The stages a [`BitsCopy`] runs in: a fill's clears, the split's compute, and the merge's
/// draw.
pub(super) const BITS_COPY_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::ALL_TRANSFER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
);

impl BitsCopy {
    /// The copies of `regions` (each a source and a destination region in the images' own
    /// levels and layers, of one size) from `source` into `destination`, whose image is of
    /// `format`, both of `samples` samples: for each region a new set of planes over the
    /// destination's level from its origin to the region's far corner, and for each of its
    /// layers the split of the source's layer into them and their merge into the destination's
    /// over the region. `None` where the copies cannot run: a region at another origin in each
    /// image (the planes hold each sample where the source has it, the merge writes it where
    /// the destination has it), an image they cannot view in the copy format
    /// (`framebuffer_fetch::planes_copies`; the destination's a color attachment), formats of
    /// two copy formats, or a device that cannot run them (an `Unsupported` refusal of the
    /// planes, `Planes::new`).
    pub(super) fn plan(
        maker: &CommandBuffer,
        source: &Texture,
        destination: &Texture,
        format: Format,
        regions: &[(ImageRegion, ImageRegion)],
        samples: u32,
    ) -> Result<Option<BitsCopy>, Error> {
        let (from, into) = (source.image_plan()?, destination.image_plan()?);
        let copy = PlanesCopy::for_format(format);
        let runs = copy.is_some()
            && copy == PlanesCopy::for_format(from.format)
            && framebuffer_fetch::planes_copies(from)
            && framebuffer_fetch::planes_copies(into)
            && into.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
            && regions.iter().all(|(src, dst)| src.origin == dst.origin);
        if !runs {
            return Ok(None);
        }
        let layers: u32 = regions.iter().map(|(src, _)| src.layers).sum();
        let mut objects = PassObjects::new(maker.device(), layers, layers)?;
        let (mut made, mut steps, mut uses) = (Vec::new(), Vec::new(), Vec::new());
        for (src, dst) in regions {
            let [x, y, _] = dst.origin;
            let [width, height, _] = dst.size;
            let extent = [x + width, y + height];
            let target = |texture: &Texture, format, region: &ImageRegion, layer| Target {
                texture: texture.clone(),
                level: region.level,
                layer: region.layer + layer,
                layers: 1,
                format,
                samples,
                extent,
            };
            let planes =
                match Planes::new(maker, &target(destination, format, dst, 0), samples, extent) {
                    Ok(planes) => planes,
                    Err(Error::Unsupported(_)) => return Ok(None),
                    Err(error) => return Err(error),
                };
            let area = vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D { width, height },
            };
            for layer in 0..src.layers {
                let from = target(source, from.format, src, layer);
                steps.push(objects.split(&from, &planes, &mut uses)?);
                let into = target(destination, format, dst, layer);
                steps.push(objects.merge(&planes, &into, area, &mut uses)?);
            }
            made.push(planes);
        }
        Ok(Some(BitsCopy {
            _planes: made,
            steps,
            uses,
            _objects: objects,
        }))
    }

    /// A fill of the region `origin` by `size` of a new multisampled image of `copy`'s format
    /// (an unsigned-integer one) over the region's far corner, whose sample `s` of each texel there
    /// takes the bits `values[s]` (the format's words): the image, a texture the command buffer
    /// holds, and the steps that fill it, the sample planes each cleared to its sample's value
    /// (`vkCmdClearColorImage` in the format itself, which takes the value's bits) and merged
    /// into the image over the region. Refused, `Unsupported`, where the device cannot run the
    /// merge (`Planes::new`) or make the image.
    pub(super) fn fill(
        maker: &CommandBuffer,
        copy: PlanesCopy,
        values: &[[u32; 4]],
        [x, y]: [u32; 2],
        [width, height]: [u32; 2],
    ) -> Result<(Texture, BitsCopy), Error> {
        let pixel_format = match copy {
            PlanesCopy::R8 => PixelFormat::R8Uint,
            PlanesCopy::R16 => PixelFormat::R16Uint,
            PlanesCopy::R32 => PixelFormat::R32Uint,
            PlanesCopy::Rg32 => PixelFormat::RG32Uint,
            PlanesCopy::Rgba32 => PixelFormat::RGBA32Uint,
            _ => return Err(internal("a multisampled fill of no unsigned format")),
        };
        let samples = u32::try_from(values.len()).map_err(|_| internal("a sample count"))?;
        let format = copy.format();
        let extent = [x + width, y + height];
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2DMultisample,
            pixel_format,
            width: extent[0].into(),
            height: extent[1].into(),
            sample_count: samples.into(),
            usage: TextureUsage::RENDER_TARGET
                | TextureUsage::SHADER_READ
                | TextureUsage::PIXEL_FORMAT_VIEW,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        // Mutable, as the merge's views and the copies out of it (into another format of the
        // size, `blit_texture.rs`) view it.
        let plan = ImagePlan {
            pixel_format,
            flags: ImageCreateFlags::MUTABLE_FORMAT,
            image_type: ImageType::TYPE_2D,
            format,
            extent: [extent[0], extent[1], 1],
            levels: 1,
            layers: 1,
            samples: sample_count(samples)
                .ok_or_else(|| internal("a fill of a sample count Vulkan lacks"))?,
            usage: ImageUsageFlags::COLOR_ATTACHMENT
                | ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::TRANSFER_DST,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC,
        };
        let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
        let target = Target {
            texture: texture.clone(),
            level: 0,
            layer: 0,
            layers: 1,
            format,
            samples,
            extent,
        };
        let planes = Planes::new(maker, &target, samples, extent)?;
        if !planes.clearable {
            return Err(internal(
                "planes of an unsigned format without transfer writes",
            ));
        }
        let mut objects = PassObjects::new(maker.device(), 0, 1)?;
        let write = Access::new(
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let mut uses = vec![planes.image_use(write, true)?];
        let image = planes.texture.raw_image()?;
        let mut steps: Vec<Recorded> = (0..samples)
            .zip(values)
            .map(|(layer, &uint32)| Recorded::Clear {
                image,
                value: vk::ClearColorValue { uint32 },
                layers: layer..layer + 1,
            })
            .collect();
        let area = vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D { width, height },
        };
        steps.push(objects.merge(&planes, &target, area, &mut uses)?);
        Ok((
            texture,
            BitsCopy {
                _planes: vec![planes],
                steps,
                uses,
                _objects: objects,
            },
        ))
    }

    /// Record the copies into `command_buffer`, each step between barriers from everything
    /// before and to everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// images and these copies are live until it completes.
    pub(super) unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe { record_steps(raw, command_buffer, &self.steps) };
    }
}

/// The rectangle of `extent` from the origin.
fn whole([width, height]: [u32; 2]) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    }
}

/// Record `steps` into `command_buffer`, each between barriers from everything before and to
/// everything after.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass, and only the caller records it; the
/// images the steps run on and their objects are live until it completes.
unsafe fn record_steps(raw: &Vk, command_buffer: vk::CommandBuffer, steps: &[Recorded]) {
    for step in steps {
        // SAFETY: the caller's contract; each command's objects were made for this device.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            match step {
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
                    area,
                } => {
                    let [width, height] = *extent;
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(merge.render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(*area);
                    raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
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
                    // The triangle covers the whole extent, so each fragment's `FragCoord` is
                    // its texel in both the planes and the image; the scissor keeps the area.
                    let viewport = vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    };
                    raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
                    raw.cmd_set_scissor(command_buffer, 0, &[*area]);
                    raw.cmd_draw(command_buffer, 3, 1, 0, 0);
                    raw.cmd_end_render_pass(command_buffer);
                }
                Recorded::Clear {
                    image,
                    value,
                    layers,
                } => {
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: layers.start,
                        layer_count: layers.end - layers.start,
                    };
                    raw.cmd_clear_color_image(
                        command_buffer,
                        *image,
                        vk::ImageLayout::GENERAL,
                        value,
                        &[range],
                    );
                }
            }
            Barrier::everything().record(raw, command_buffer);
        }
    }
}
