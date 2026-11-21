//! The Vulkan objects behind a binder plan's descriptor writes, shared by the engine compute
//! and render executors.
//!
//! Every write of the plan is resolved against the run's guest resources, keyed by slot, and
//! nothing about the layout is re-derived:
//!
//! - buffer writes bind the guest buffer over exactly the plan's `DescriptorRange`; the argument
//!   table, each stage's constant blob, and the zero and sink neutrals are buffers this module
//!   allocates;
//! - a texture write views the slot's image with the plan's `ImageViewDesc` (type, format,
//!   component mapping, aspect, layout, and levels: a whole-texture view holds every level, an
//!   element of a storage slot's level views its one level, the plan's neutrals past the
//!   texture's last). Each bound texture is one image of its authored type, format, extent, and
//!   levels, every level uploaded before the work and left in the one layout its views name (a
//!   plan whose views of one texture name two layouts is not implemented);
//! - a heap entry views its member's image the same way, at its element of the heap array: a
//!   used texture's (by consumer token, an argument-buffer texture) or a bound one's (by slot), one
//!   image per texture whatever writes view it (every view of one member takes one layout); each
//!   stage's handle table is a buffer of exactly the bytes instantiation gave it, bound over
//!   exactly its planned length;
//! - a sampler (plain, or the sampler half of a combined write) is `bind::sampler_desc` of the
//!   bound descriptor for the write's use, and the image format for a combined write; an
//!   immutable one is the layout's;
//! - a neutral image is created once per kind, cleared to zero, and left in the layout its
//!   descriptor type reads (`GENERAL` for storage, `SHADER_READ_ONLY_OPTIMAL` otherwise); a
//!   neutral texel buffer is 16 zero bytes; the neutral sampler is nearest and clamped;
//! - a tessellation-factor write binds the draw's factor buffer over exactly the plan's range;
//! - an input-attachment write views the draw's color attachment of its index in `GENERAL`, the
//!   layout the render pass keeps a framebuffer-fetch attachment in.
//!
//! The run keeps no imageblock storage: an imageblock write fails. A write of a source the runtime
//! does not build (a texture buffer, an acceleration structure, a tile pass's attachment, a sparse
//! texture's tile map), which the case's requirements refuse first, and inline bytes are typed
//! `NotImplemented`. Descriptor sets come from one
//! pool sized from the layout's bindings, update-after-bind when the layout has the heap's set
//! ([`PlanResources::write_sets`]).

use super::acceptance::LayoutObjects;
use super::buffers::{create_host_buffer, HostBuffer};
use super::device::VulkanContext;
use super::multisample::initialize_multisample_image;
use crate::candidate::engine::bind_refusal;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::literal::{texture_layout, LiteralTexture, TextureLayout};
use ash::vk;
use metal2vulkan::bind::{
    sampler_desc, ArgumentSampler, DescriptorBytes, DescriptorRange, DescriptorWrite,
    ExecutionStatus, HeapSource, ImageSource, ImageViewDesc, NeutralResource, SamplerDescriptor,
    SamplerSource, WriteSource,
};
use metal2vulkan::device::Device;
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse};
use metal2vulkan::pipeline::{SlotRef, Stage};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Bytes of a neutral texel buffer: one texel of any format the core declares.
const NEUTRAL_TEXEL_BYTES: usize = 16;

fn not_implemented(what: impl Into<String>) -> ExecutorError {
    NotSupported::NotImplemented(format!("engine candidate: {}", what.into())).into()
}

/// A texture a plan write views: the texture bound at a slot, or the used texture with a consumer
/// token (a heap member; an argument-buffer texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TextureKey {
    Slot(SlotRef),
    Used(u32),
}

impl TextureKey {
    /// The key of a heap entry's `source`.
    fn of_heap(source: HeapSource) -> Result<TextureKey, ExecutorError> {
        match source {
            HeapSource::Bound(slot) => Ok(TextureKey::Slot(slot)),
            HeapSource::Used(token) => Ok(TextureKey::Used(token)),
        }
    }

    /// A label of the texture, for errors and object names.
    fn label(self) -> String {
        match self {
            TextureKey::Slot(slot) => format!("texture {}", slot.index),
            TextureKey::Used(token) => format!("used texture {token}"),
        }
    }
}

/// The run's guest resources a plan's writes resolve against.
pub(super) struct Guest<'a> {
    /// Each bound buffer's allocation, by slot.
    pub(super) buffers: &'a BTreeMap<SlotRef, vk::Buffer>,
    /// Each bound texture with its initial contents, by slot.
    pub(super) textures: &'a BTreeMap<SlotRef, &'a LiteralTexture>,
    /// Each used texture (an argument-buffer texture) with its initial contents, by consumer
    /// token.
    pub(super) used_textures: &'a BTreeMap<u32, &'a LiteralTexture>,
    /// Each bound sampler, by slot.
    pub(super) samplers: &'a BTreeMap<SlotRef, SamplerDescriptor>,
    /// The bind state's argument samplers, which a sampler array's `WriteSource::ArgumentSampler`
    /// element names by index.
    pub(super) argument_samplers: &'a [ArgumentSampler],
    /// Each stage's constant blob.
    pub(super) constant_blobs: &'a BTreeMap<Stage, Arc<[u8]>>,
    pub(super) descriptor_ranges: &'a [DescriptorRange],
    pub(super) descriptor_bytes: &'a [DescriptorBytes],
    /// The encoder's tessellation factor buffer of a patch draw.
    pub(super) tessellation_factors: Option<vk::Buffer>,
    /// Each color target's view of a draw's pass, by Metal index: what an input-attachment write
    /// reads (framebuffer fetch), in `GENERAL`. A dispatch has none.
    pub(super) attachments: &'a BTreeMap<u32, vk::ImageView>,
    /// The run's `ExecutionStatus` buffer, which a stage binds on a device that does not address
    /// buffers (`WriteSource::ExecutionStatus`). A device replay, which needs the addresses,
    /// has none.
    pub(super) status: Option<vk::Buffer>,
}

/// What one descriptor write points at.
enum Info {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
    Texel(vk::BufferView),
}

/// One resolved descriptor write.
struct Write {
    set: u32,
    binding: u32,
    element: u32,
    ty: vk::DescriptorType,
    info: Info,
}

/// The image of one texture.
struct SlotImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    aspect: vk::ImageAspectFlags,
    extent: vk::Extent3D,
    layers: u32,
    /// Its mipmap levels, the texture's.
    levels: u32,
    /// The layout every view of the image names.
    layout: vk::ImageLayout,
    /// The authored contents, uploaded before the work; `None` for a multisampled image, which
    /// its own initializer submission wrote per sample and left in `GENERAL`.
    staging: Option<HostBuffer>,
    /// The copies from `staging` that upload the image: each level's whole extent.
    copies: Vec<vk::BufferImageCopy>,
    /// The final contents, when the texture is the observed output.
    readback: Option<HostBuffer>,
}

impl SlotImage {
    /// Every level and layer of the image.
    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            level_count: self.levels,
            ..range(self.aspect, self.layers)
        }
    }
}

/// One neutral image.
struct NeutralImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    aspect: vk::ImageAspectFlags,
    layers: u32,
    layout: vk::ImageLayout,
}

/// Every object behind one plan's descriptor writes, destroyed on every exit path. The caller
/// waits for its submission before dropping it.
pub(super) struct PlanResources<'a> {
    device: &'a ash::Device,
    writes: Vec<Write>,
    images: BTreeMap<TextureKey, SlotImage>,
    neutral_images: BTreeMap<NeutralResource, NeutralImage>,
    views: Vec<vk::ImageView>,
    buffer_views: Vec<vk::BufferView>,
    samplers: Vec<vk::Sampler>,
    descriptor_pool: vk::DescriptorPool,
    /// Buffers this module allocated (neutrals, tables, blobs).
    buffers: Vec<HostBuffer>,
}

impl Drop for PlanResources<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run and is destroyed once, after
        // the submission that used it completed (or before any submission).
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for sampler in &self.samplers {
                self.device.destroy_sampler(*sampler, None);
            }
            for view in &self.buffer_views {
                self.device.destroy_buffer_view(*view, None);
            }
            for view in &self.views {
                self.device.destroy_image_view(*view, None);
            }
            for image in self.images.values() {
                self.device.destroy_image(image.image, None);
                self.device.free_memory(image.memory, None);
            }
            for image in self.neutral_images.values() {
                self.device.destroy_image_view(image.view, None);
                self.device.destroy_image(image.image, None);
                self.device.free_memory(image.memory, None);
            }
        }
    }
}

/// The layout and format of one view a plan asks of an image.
#[derive(Clone, Copy)]
struct ViewUse {
    layout: vk::ImageLayout,
    format: vk::Format,
}

impl From<&ImageViewDesc> for ViewUse {
    fn from(view: &ImageViewDesc) -> ViewUse {
        ViewUse {
            layout: view.layout.into(),
            format: view.format.into(),
        }
    }
}

/// The views a plan asks of one bound texture, and how its descriptors use it.
#[derive(Default)]
struct TextureUse {
    views: Vec<ViewUse>,
    sampled: bool,
    storage: bool,
}

/// Whether a descriptor type reads an image through a sampler or `OpImageFetch` (sampled) rather
/// than as a storage image.
fn is_storage(ty: vk::DescriptorType) -> bool {
    ty == vk::DescriptorType::STORAGE_IMAGE || ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER
}

impl<'a> PlanResources<'a> {
    /// Create every object `descriptors` needs from `guest` on `device` (the product device the
    /// plan was made on), with a host readback of `observed` when it is a texture the plan views.
    pub(super) fn create(
        context: &'a VulkanContext,
        device: &Device,
        descriptors: &[DescriptorWrite],
        guest: &Guest<'_>,
        observed: Option<TextureKey>,
    ) -> Result<PlanResources<'a>, ExecutorError> {
        let mut resources = PlanResources {
            device: &context.device,
            writes: Vec::new(),
            images: BTreeMap::new(),
            neutral_images: BTreeMap::new(),
            views: Vec::new(),
            buffer_views: Vec::new(),
            samplers: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            buffers: Vec::new(),
        };
        // Every view of each texture first: its image serves all of them.
        let mut uses: BTreeMap<TextureKey, TextureUse> = BTreeMap::new();
        for write in descriptors {
            let ty: vk::DescriptorType = write.ty.into();
            let (key, view) = match &write.source {
                WriteSource::Texture { slot, view }
                | WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { slot, view },
                    ..
                } => (TextureKey::Slot(*slot), ViewUse::from(view)),
                WriteSource::HeapEntry { source, view, .. } => {
                    (TextureKey::of_heap(*source)?, ViewUse::from(view))
                }
                _ => continue,
            };
            let entry = uses.entry(key).or_default();
            entry.views.push(view);
            if is_storage(ty) {
                entry.storage = true;
            } else {
                entry.sampled = true;
            }
        }
        for (&key, texture_use) in &uses {
            let texture: &LiteralTexture = match key {
                TextureKey::Slot(slot) => guest.textures.get(&slot).copied(),
                TextureKey::Used(token) => guest.used_textures.get(&token).copied(),
            }
            .ok_or_else(|| format!("the plan views {} that is not bound", key.label()))?;
            let image = create_slot_image(
                context,
                &key.label(),
                texture,
                texture_use,
                observed == Some(key),
            )?;
            resources.images.insert(key, image);
        }
        for write in descriptors {
            let info = resources.resolve(context, device, write, guest)?;
            resources.writes.push(Write {
                set: write.set,
                binding: write.binding,
                element: write.array_element,
                ty: write.ty.into(),
                info,
            });
        }
        Ok(resources)
    }

    fn resolve(
        &mut self,
        context: &VulkanContext,
        device: &Device,
        write: &DescriptorWrite,
        guest: &Guest<'_>,
    ) -> Result<Info, ExecutorError> {
        let key = (write.set, write.binding);
        let ty: vk::DescriptorType = write.ty.into();
        let whole = |buffer: &HostBuffer| {
            Info::Buffer(
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.len),
            )
        };
        let range = |key: (u32, u32)| {
            guest
                .descriptor_ranges
                .iter()
                .find(|r| (r.set, r.binding) == key)
                .map(|r| r.range)
                .ok_or_else(|| format!("no descriptor range for {key:?}"))
        };
        Ok(match &write.source {
            WriteSource::Buffer { slot } => {
                let buffer = guest
                    .buffers
                    .get(slot)
                    .ok_or_else(|| format!("the plan binds unbound buffer {slot:?}"))?;
                let range = range(key)?;
                Info::Buffer(
                    vk::DescriptorBufferInfo::default()
                        .buffer(*buffer)
                        .offset(range.offset)
                        .range(range.range),
                )
            }
            WriteSource::Neutral(
                NeutralResource::ZeroBuffer { min_bytes } | NeutralResource::Sink { min_bytes },
            ) => {
                let zeros = vec![0u8; (*min_bytes as usize).max(16)];
                let buffer = self.buffer(context, "neutral buffer", &zeros)?;
                whole(buffer)
            }
            WriteSource::ArgumentTable { .. } => {
                let bytes = guest
                    .descriptor_bytes
                    .iter()
                    .find(|d| (d.set, d.binding) == key)
                    .ok_or_else(|| format!("no argument-table bytes for {key:?}"))?;
                let buffer = self.buffer(context, "argument table", &bytes.bytes)?;
                whole(buffer)
            }
            // The shader counts the table's entries from the bound range, so it is bound over
            // exactly its planned length.
            WriteSource::HandleTable { stage, bytes_len } => {
                let bytes = guest
                    .descriptor_bytes
                    .iter()
                    .find(|d| (d.set, d.binding) == key)
                    .ok_or_else(|| format!("no {stage:?} handle-table bytes for {key:?}"))?;
                if bytes.bytes.len() != *bytes_len as usize {
                    return Err(format!(
                        "the {stage:?} handle table holds {} bytes, not its planned {bytes_len}",
                        bytes.bytes.len()
                    )
                    .into());
                }
                let buffer = self.buffer(context, "handle table", &bytes.bytes)?;
                Info::Buffer(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer.buffer)
                        .offset(0)
                        .range(u64::from(*bytes_len)),
                )
            }
            WriteSource::ConstantBlob { stage } => {
                let blob = guest.constant_blobs.get(stage).ok_or_else(|| {
                    format!("the plan writes a {stage:?} constant blob the pipeline does not carry")
                })?;
                let blob = blob.clone();
                let buffer = self.buffer(context, "constant blob", &blob)?;
                whole(buffer)
            }
            // The run's status words, the ones every command of it writes.
            WriteSource::ExecutionStatus => {
                let status = guest
                    .status
                    .ok_or("the plan binds an execution status the run has none of")?;
                Info::Buffer(
                    vk::DescriptorBufferInfo::default()
                        .buffer(status)
                        .offset(0)
                        .range(ExecutionStatus::BYTES as u64),
                )
            }
            WriteSource::Texture { slot, view } => {
                let (image, layout) = self.image(TextureKey::Slot(*slot))?;
                let view = self.view(context, image, view)?;
                Info::Image(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(layout),
                )
            }
            WriteSource::HeapEntry { source, view, .. } => {
                let (image, layout) = self.image(TextureKey::of_heap(*source)?)?;
                let view = self.view(context, image, view)?;
                Info::Image(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(layout),
                )
            }
            WriteSource::CombinedImageSampler { image, sampler } => {
                let (view, layout, format) = match image {
                    ImageSource::Texture { slot, view } => {
                        let (image, layout) = self.image(TextureKey::Slot(*slot))?;
                        (self.view(context, image, view)?, layout, view.format)
                    }
                    ImageSource::Neutral(neutral) => {
                        let image = self.neutral_image(context, *neutral, ty)?;
                        let format = neutral.format().ok_or("a neutral image without a format")?;
                        (image.0, image.1, format)
                    }
                };
                let sampler = match sampler {
                    SamplerSource::Slot {
                        slot,
                        compare,
                        integer,
                        opaque_black,
                    } => {
                        let desc = bound_sampler(guest, *slot)?;
                        let use_ = SamplerUse::new(*compare, *integer);
                        self.slot_sampler(
                            context,
                            device,
                            desc,
                            (use_, *opaque_black),
                            Some(format),
                        )?
                    }
                    SamplerSource::Neutral(NeutralResource::Sampler { compare }) => {
                        self.neutral_sampler(context, *compare)?
                    }
                    // The binding's immutable sampler is the layout's; the write carries only
                    // the image.
                    SamplerSource::Immutable => vk::Sampler::null(),
                    other => return Err(not_implemented(format!("the sampler source {other:?}"))),
                };
                Info::Image(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(layout)
                        .sampler(sampler),
                )
            }
            WriteSource::Sampler {
                slot,
                compare,
                integer,
                opaque_black,
            } => {
                let desc = bound_sampler(guest, *slot)?;
                let use_ = SamplerUse::new(*compare, *integer);
                let sampler =
                    self.slot_sampler(context, device, desc, (use_, *opaque_black), None)?;
                Info::Image(vk::DescriptorImageInfo::default().sampler(sampler))
            }
            // An element past the list is the neutral sampler (the plan's array covers its
            // class's entries).
            WriteSource::ArgumentSampler {
                index,
                compare,
                integer,
            } => {
                let use_ = SamplerUse::new(*compare, *integer);
                let sampler = match guest.argument_samplers.get(*index as usize) {
                    Some(listed) => self.slot_sampler(
                        context,
                        device,
                        &listed.desc,
                        (use_, OpaqueBlackBorder::Stated),
                        None,
                    )?,
                    None => self.neutral_sampler(context, *compare)?,
                };
                Info::Image(vk::DescriptorImageInfo::default().sampler(sampler))
            }
            WriteSource::Neutral(NeutralResource::Sampler { compare }) => Info::Image(
                vk::DescriptorImageInfo::default()
                    .sampler(self.neutral_sampler(context, *compare)?),
            ),
            WriteSource::Neutral(neutral @ NeutralResource::Image { .. }) => {
                let (view, layout) = self.neutral_image(context, *neutral, ty)?;
                Info::Image(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(layout),
                )
            }
            WriteSource::Neutral(NeutralResource::TexelBuffer { format }) => {
                let usage = if is_storage(ty) {
                    vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                } else {
                    vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                };
                let buffer = create_host_buffer(
                    context,
                    "neutral texel buffer",
                    &[0; NEUTRAL_TEXEL_BYTES],
                    usage,
                )?;
                let view =
                    self.buffer_view(context, buffer.buffer, (*format).into(), (0, buffer.len))?;
                self.buffers.push(buffer);
                Info::Texel(view)
            }
            // The control stage reads the draw's factors over exactly the plan's range.
            WriteSource::TessellationFactors => {
                let buffer = guest
                    .tessellation_factors
                    .ok_or("the plan binds a tessellation factor buffer the draw does not set")?;
                let range = range(key)?;
                Info::Buffer(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(range.offset)
                        .range(range.range),
                )
            }
            // The executor draws patches without a patch index buffer: the zero buffer, whose
            // words the control stage discards.
            WriteSource::PatchIndices { min_bytes } => {
                if range(key).is_ok() {
                    let why = "the draw binds a patch index buffer the executor lacks";
                    return Err(why.to_owned().into());
                }
                let zeros = vec![0u8; (*min_bytes as usize).max(16)];
                let buffer = self.buffer(context, "zero patch index buffer", &zeros)?;
                whole(buffer)
            }
            // The executor's pipelines have no control-point index type.
            WriteSource::ControlPointIndices => {
                let why = "the plan binds a control-point index buffer the executor lacks";
                return Err(why.to_owned().into());
            }
            // Framebuffer fetch reads the color attachment of its index, which the render pass
            // keeps in `GENERAL` while the subpass writes and reads it.
            WriteSource::Attachment { color_index } => {
                let view = guest.attachments.get(color_index).ok_or_else(|| {
                    format!("the plan reads color attachment {color_index}, which the draw lacks")
                })?;
                Info::Image(
                    vk::DescriptorImageInfo::default()
                        .image_view(*view)
                        .image_layout(vk::ImageLayout::GENERAL),
                )
            }
            WriteSource::Imageblock => {
                return Err("the plan binds an imageblock storage the run does not keep".into())
            }
            other => return Err(not_implemented(format!("the descriptor source {other:?}"))),
        })
    }

    /// A buffer holding `bytes` (padded to 16), owned by the run.
    fn buffer(
        &mut self,
        context: &VulkanContext,
        label: &str,
        bytes: &[u8],
    ) -> Result<&HostBuffer, ExecutorError> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer = create_host_buffer(context, label, &padded(bytes), usage)?;
        self.buffers.push(buffer);
        Ok(self.buffers.last().expect("just pushed"))
    }

    fn view(
        &mut self,
        context: &VulkanContext,
        image: vk::Image,
        desc: &ImageViewDesc,
    ) -> Result<vk::ImageView, ExecutorError> {
        let view_type: vk::ImageViewType = desc.view_type.into();
        // A level view holds its one level, a whole-texture view every level of the image.
        let range = vk::ImageSubresourceRange {
            base_mip_level: desc.level.unwrap_or(0),
            level_count: match desc.level {
                Some(_) => 1,
                None => vk::REMAINING_MIP_LEVELS,
            },
            ..range(desc.aspect.into(), layer_count(view_type))
        };
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(desc.format.into())
            .components(desc.components.into())
            .subresource_range(range);
        // SAFETY: `info` names a live image of this run.
        let view = unsafe { context.device.create_image_view(&info, None) }
            .map_err(|error| format!("create image view {desc:?}: {error}"))?;
        self.views.push(view);
        Ok(view)
    }

    /// A texel view of `buffer` over `(offset, range)`.
    fn buffer_view(
        &mut self,
        context: &VulkanContext,
        buffer: vk::Buffer,
        format: vk::Format,
        (offset, range): (u64, u64),
    ) -> Result<vk::BufferView, ExecutorError> {
        let info = vk::BufferViewCreateInfo::default()
            .buffer(buffer)
            .format(format)
            .offset(offset)
            .range(range);
        // SAFETY: `info` names a live buffer of this run.
        let view = unsafe { context.device.create_buffer_view(&info, None) }
            .map_err(|error| format!("create texel buffer view: {error}"))?;
        self.buffer_views.push(view);
        Ok(view)
    }

    /// The image and layout of the texture `key`.
    fn image(&self, key: TextureKey) -> Result<(vk::Image, vk::ImageLayout), ExecutorError> {
        let image = self
            .images
            .get(&key)
            .ok_or_else(|| format!("no image for {}", key.label()))?;
        Ok((image.image, image.layout))
    }

    /// The `VkSampler` of a bound sampler's descriptor `desc` for a use and opaque black border
    /// reading `image_format`.
    fn slot_sampler(
        &mut self,
        context: &VulkanContext,
        device: &Device,
        desc: &SamplerDescriptor,
        (use_, opaque_black): (SamplerUse, OpaqueBlackBorder),
        image_format: Option<metal2vulkan::vk::Format>,
    ) -> Result<vk::Sampler, ExecutorError> {
        let mapped = sampler_desc(device, desc, use_, opaque_black, image_format)
            .map_err(|e| bind_refusal(&e))?;
        let mut reduction = vk::SamplerReductionModeCreateInfo::default();
        let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
        let info =
            metal2vulkan::ash_interop::sampler_create_info(&mapped, &mut reduction, &mut border);
        // SAFETY: `info` and its chain are live for the call.
        let sampler = unsafe { context.device.create_sampler(&info, None) }
            .map_err(|error| format!("create sampler {mapped:?}: {error}"))?;
        self.samplers.push(sampler);
        Ok(sampler)
    }

    /// The neutral sampler: nearest, clamped to the edge, level 0; with depth comparison when the
    /// use compares.
    fn neutral_sampler(
        &mut self,
        context: &VulkanContext,
        compare: bool,
    ) -> Result<vk::Sampler, ExecutorError> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .compare_enable(compare)
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
        // SAFETY: `info` is live for the call.
        let sampler = unsafe { context.device.create_sampler(&info, None) }
            .map_err(|error| format!("create neutral sampler: {error}"))?;
        self.samplers.push(sampler);
        Ok(sampler)
    }

    /// The view and layout of the neutral image `neutral`, created (and cleared on upload) once
    /// per kind.
    fn neutral_image(
        &mut self,
        context: &VulkanContext,
        neutral: NeutralResource,
        ty: vk::DescriptorType,
    ) -> Result<(vk::ImageView, vk::ImageLayout), ExecutorError> {
        if let Some(image) = self.neutral_images.get(&neutral) {
            return Ok((image.view, image.layout));
        }
        let NeutralResource::Image {
            view_type,
            depth_compare,
            samples,
            storage_format,
            ..
        } = neutral
        else {
            return Err(format!("{neutral:?} is not a neutral image").into());
        };
        let format: vk::Format = neutral
            .format()
            .ok_or("a neutral image without a format")?
            .into();
        let view_type: vk::ImageViewType = view_type.into();
        let storage = storage_format.is_some();
        if storage != is_storage(ty) {
            return Err(format!(
                "the neutral {neutral:?} bound as descriptor type {}",
                ty.as_raw()
            )
            .into());
        }
        let aspect = if depth_compare {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let (image_type, layers, flags) = match view_type {
            vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => {
                (vk::ImageType::TYPE_1D, 1, vk::ImageCreateFlags::empty())
            }
            vk::ImageViewType::TYPE_3D => {
                (vk::ImageType::TYPE_3D, 1, vk::ImageCreateFlags::empty())
            }
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => (
                vk::ImageType::TYPE_2D,
                6,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
            _ => (vk::ImageType::TYPE_2D, 1, vk::ImageCreateFlags::empty()),
        };
        let usage = vk::ImageUsageFlags::TRANSFER_DST
            | if storage {
                vk::ImageUsageFlags::STORAGE
            } else {
                vk::ImageUsageFlags::SAMPLED
            };
        let (image, memory) = create_image(
            context,
            "neutral image",
            &vk::ImageCreateInfo::default()
                .flags(flags)
                .image_type(image_type)
                .format(format)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(layers)
                .samples(sample_count(samples)?)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )?;
        let layout = if storage {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(range(aspect, layer_count(view_type)));
        // SAFETY: `info` names the image just created.
        let view = match unsafe { context.device.create_image_view(&info, None) } {
            Ok(view) => view,
            Err(error) => {
                // SAFETY: the image and memory were created above and are unused.
                unsafe {
                    context.device.destroy_image(image, None);
                    context.device.free_memory(memory, None);
                }
                return Err(format!("create neutral view: {error}").into());
            }
        };
        self.neutral_images.insert(
            neutral,
            NeutralImage {
                image,
                memory,
                view,
                aspect,
                layers,
                layout,
            },
        );
        Ok((view, layout))
    }

    /// Record, before the work: every bound texture's upload and every neutral image's clear, each
    /// left in the layout its descriptors name.
    ///
    /// # Safety
    ///
    /// `command` is in the recording state on this run's device, and every object stays alive
    /// until its submission completes.
    pub(super) unsafe fn record_uploads(&self, command: vk::CommandBuffer) {
        let device = self.device;
        let to_transfer: Vec<vk::ImageMemoryBarrier<'_>> = self
            .images
            .values()
            .filter(|image| image.staging.is_some())
            .map(|image| (image.image, image.range()))
            .chain(
                self.neutral_images
                    .values()
                    .map(|image| (image.image, range(image.aspect, image.layers))),
            )
            .map(|(image, range)| {
                range_barrier(
                    image,
                    range,
                    (vk::AccessFlags::empty(), vk::ImageLayout::UNDEFINED),
                    (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                )
            })
            .collect();
        // A multisampled image's initializer completed in an earlier submission on this queue.
        let initialized: Vec<vk::ImageMemoryBarrier<'_>> = self
            .images
            .values()
            .filter(|image| image.staging.is_none())
            .map(|image| {
                range_barrier(
                    image.image,
                    image.range(),
                    (
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::ImageLayout::GENERAL,
                    ),
                    (vk::AccessFlags::SHADER_READ, image.layout),
                )
            })
            .collect();
        if !initialized.is_empty() {
            // SAFETY: the caller's contract; every handle is live.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &initialized,
                );
            }
        }
        if to_transfer.is_empty() {
            return;
        }
        let to_shader: Vec<vk::ImageMemoryBarrier<'_>> = self
            .images
            .values()
            .filter(|image| image.staging.is_some())
            .map(|image| (image.image, image.range(), image.layout))
            .chain(
                self.neutral_images
                    .values()
                    .map(|image| (image.image, range(image.aspect, image.layers), image.layout)),
            )
            .map(|(image, range, layout)| {
                range_barrier(
                    image,
                    range,
                    (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ),
                    (
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        layout,
                    ),
                )
            })
            .collect();
        // SAFETY: the caller's contract; every handle is live.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            for image in self.images.values() {
                let Some(staging) = &image.staging else {
                    continue;
                };
                if image.copies.is_empty() {
                    continue;
                }
                device.cmd_copy_buffer_to_image(
                    command,
                    staging.buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &image.copies,
                );
            }
            for image in self.neutral_images.values() {
                let range = [range(image.aspect, image.layers)];
                if image.aspect == vk::ImageAspectFlags::DEPTH {
                    device.cmd_clear_depth_stencil_image(
                        command,
                        image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearDepthStencilValue::default(),
                        &range,
                    );
                } else {
                    device.cmd_clear_color_image(
                        command,
                        image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue::default(),
                        &range,
                    );
                }
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_shader,
            );
        }
    }

    /// Record, after the work: a shader-to-host dependency, and the observed texture's copy into its
    /// host readback.
    ///
    /// # Safety
    ///
    /// As [`PlanResources::record_uploads`].
    pub(super) unsafe fn record_readback(&self, command: vk::CommandBuffer) {
        let device = self.device;
        let shader_to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        let transfer_to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: the caller's contract; every handle is live.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &shader_to_host,
                &[],
                &[],
            );
            for image in self.images.values() {
                let Some(readback) = &image.readback else {
                    continue;
                };
                let to_source = [range_barrier(
                    image.image,
                    image.range(),
                    (vk::AccessFlags::SHADER_WRITE, image.layout),
                    (
                        vk::AccessFlags::TRANSFER_READ,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                )];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_source,
                );
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(layers(image.aspect, image.layers))
                    .image_extent(image.extent);
                device.cmd_copy_image_to_buffer(
                    command,
                    image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    &[copy],
                );
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &transfer_to_host,
                    &[],
                    &[],
                );
            }
        }
    }

    /// `texture` (the texture `key`) with its final contents: the observed image's readback, or its
    /// authored bytes when no descriptor of the plan reaches it.
    ///
    /// # Safety
    ///
    /// The submission that used the resources has completed.
    pub(super) unsafe fn final_texture(
        &self,
        key: TextureKey,
        texture: &LiteralTexture,
    ) -> Result<LiteralTexture, String> {
        let bytes = if let Some(image) = self.images.get(&key) {
            let readback = image
                .readback
                .as_ref()
                .ok_or_else(|| format!("{} was not read back", key.label()))?;
            // SAFETY: the caller's contract.
            unsafe { readback.map()? }.bytes[..texture.bytes.len()].to_vec()
        } else {
            texture.bytes.clone()
        };
        Ok(LiteralTexture {
            bytes,
            ..texture.clone()
        })
    }

    /// Allocate one descriptor set per layout set, from one pool sized from the layout's
    /// bindings (each binding's whole count, written or not: a heap array's capacity, an
    /// immutable heap sampler the plan never writes) and update-after-bind when the layout has
    /// the heap's set ([`LayoutObjects::pool_sizes`]), and write every resolved descriptor.
    pub(super) fn write_sets(
        &mut self,
        context: &VulkanContext,
        layout: &LayoutObjects<'_>,
    ) -> Result<Vec<vk::DescriptorSet>, ExecutorError> {
        if layout.set_layouts.is_empty() {
            return Ok(Vec::new());
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(layout.pool_flags)
            .max_sets(layout.set_layouts.len() as u32)
            .pool_sizes(&layout.pool_sizes);
        // SAFETY: `pool_info` and its sizes are live for the call.
        self.descriptor_pool = unsafe { context.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|error| format!("create descriptor pool: {error}"))?;
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layout.set_layouts);
        // SAFETY: the pool and the layouts are live; the sets are freed with the pool.
        let sets = unsafe { context.device.allocate_descriptor_sets(&allocate_info) }
            .map_err(|error| format!("allocate descriptor sets: {error}"))?;
        let mut vk_writes = Vec::with_capacity(self.writes.len());
        for write in &self.writes {
            let set = *sets
                .get(write.set as usize)
                .ok_or_else(|| format!("a write to set {} beyond the layout", write.set))?;
            let base = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(write.binding)
                .dst_array_element(write.element)
                .descriptor_type(write.ty);
            vk_writes.push(match &write.info {
                Info::Buffer(info) => base.buffer_info(std::slice::from_ref(info)),
                Info::Image(info) => base.image_info(std::slice::from_ref(info)),
                Info::Texel(view) => base.texel_buffer_view(std::slice::from_ref(view)),
            });
        }
        // SAFETY: every set, buffer, view, sampler, and info array is live for the call.
        unsafe { context.device.update_descriptor_sets(&vk_writes, &[]) };
        Ok(sets)
    }
}

/// The descriptor of the sampler bound at `slot`.
fn bound_sampler<'g>(
    guest: &'g Guest<'_>,
    slot: SlotRef,
) -> Result<&'g SamplerDescriptor, ExecutorError> {
    guest
        .samplers
        .get(&slot)
        .ok_or_else(|| format!("the plan samples through unbound sampler {slot:?}").into())
}

/// `bytes` zero-padded to a multiple of 16 and at least 16 bytes: the consumer pads every
/// backing buffer to 16.
pub(super) fn padded(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out.resize(bytes.len().next_multiple_of(16).max(16), 0);
    out
}

fn sample_count(samples: u32) -> Result<vk::SampleCountFlags, ExecutorError> {
    Ok(match samples {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        other => return Err(format!("sample count {other}").into()),
    })
}

/// The layer count of a view of `view_type`: every layer of an arrayed or cube view, one
/// otherwise.
fn layer_count(view_type: vk::ImageViewType) -> u32 {
    match view_type {
        vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_3D => 1,
        _ => vk::REMAINING_ARRAY_LAYERS,
    }
}

fn range(aspect: vk::ImageAspectFlags, layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: layers,
    }
}

pub(super) fn layers(aspect: vk::ImageAspectFlags, layers: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: aspect,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: layers,
    }
}

pub(super) fn barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    layers: u32,
    src: (vk::AccessFlags, vk::ImageLayout),
    dst: (vk::AccessFlags, vk::ImageLayout),
) -> vk::ImageMemoryBarrier<'static> {
    range_barrier(image, range(aspect, layers), src, dst)
}

/// The layout transition of the subresources `range` of `image`.
fn range_barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    (src_access, old): (vk::AccessFlags, vk::ImageLayout),
    (dst_access, new): (vk::AccessFlags, vk::ImageLayout),
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}

/// A device-local image of `info` with its memory bound.
pub(super) fn create_image(
    context: &VulkanContext,
    label: &str,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<(vk::Image, vk::DeviceMemory), ExecutorError> {
    image_support(context, label, info)?;
    // SAFETY: `info` is live for the call.
    let image = unsafe { context.device.create_image(info, None) }
        .map_err(|error| format!("create {label}: {error}"))?;
    // SAFETY: the image was just created.
    let requirements = unsafe { context.device.get_image_memory_requirements(image) };
    let allocated = context
        .memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .and_then(|memory_type| {
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            // SAFETY: `allocation` is live for the call.
            unsafe { context.device.allocate_memory(&allocation, None) }
                .map_err(|error| format!("allocate {label}: {error}"))
        })
        .and_then(|memory| {
            // SAFETY: both handles are live and unbound.
            match unsafe { context.device.bind_image_memory(image, memory, 0) } {
                Ok(()) => Ok(memory),
                Err(error) => {
                    // SAFETY: the memory was allocated above and is unused.
                    unsafe { context.device.free_memory(memory, None) };
                    Err(format!("bind {label}: {error}"))
                }
            }
        });
    match allocated {
        Ok(memory) => Ok((image, memory)),
        Err(error) => {
            // SAFETY: the image was created above and is unused.
            unsafe { context.device.destroy_image(image, None) };
            Err(error.into())
        }
    }
}

/// Whether the device can create an image of `info`, checked before `vkCreateImage` (which is
/// invalid usage for a combination the device does not support): the device's
/// `vkGetPhysicalDeviceImageFormatProperties` for the format, type, tiling, usage, and flags must
/// exist and allow the sample count, extent, layers, and levels. A missing combination is
/// `NotSupported::NotEnabled` naming the image, its format, and its sample count.
pub(super) fn image_support(
    context: &VulkanContext,
    label: &str,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<(), ExecutorError> {
    // Raw values: this build of `ash` has no `Debug` for its enums.
    let image = || {
        format!(
            "{label}: an image of VkFormat {} with {} sample(s), VkImageType {}, {}x{}x{}, {} \
             layer(s), {} level(s), VkImageTiling {}, usage {:#x}, flags {:#x}",
            info.format.as_raw(),
            info.samples.as_raw(),
            info.image_type.as_raw(),
            info.extent.width,
            info.extent.height,
            info.extent.depth,
            info.array_layers,
            info.mip_levels,
            info.tiling.as_raw(),
            info.usage.as_raw(),
            info.flags.as_raw()
        )
    };
    // SAFETY: the physical device belongs to the live instance.
    let queried = unsafe {
        context
            .instance
            .get_physical_device_image_format_properties(
                context.physical,
                info.format,
                info.image_type,
                info.tiling,
                info.usage,
                info.flags,
            )
    };
    let limits = match queried {
        Ok(limits) => limits,
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
            return Err(NotSupported::NotEnabled(vec![format!(
                "{} (vkGetPhysicalDeviceImageFormatProperties: not supported)",
                image()
            )])
            .into())
        }
        Err(error) => {
            return Err(format!(
                "{}: vkGetPhysicalDeviceImageFormatProperties: {error}",
                image()
            )
            .into())
        }
    };
    let mut lacks = Vec::new();
    if !limits.sample_counts.contains(info.samples) {
        lacks.push(format!(
            "sample counts {:#b} (bit n: 2^n samples)",
            limits.sample_counts.as_raw()
        ));
    }
    let max = limits.max_extent;
    if info.extent.width > max.width
        || info.extent.height > max.height
        || info.extent.depth > max.depth
    {
        lacks.push(format!(
            "an extent of at most {}x{}x{}",
            max.width, max.height, max.depth
        ));
    }
    if info.array_layers > limits.max_array_layers {
        lacks.push(format!("at most {} layer(s)", limits.max_array_layers));
    }
    if info.mip_levels > limits.max_mip_levels {
        lacks.push(format!("at most {} level(s)", limits.max_mip_levels));
    }
    if lacks.is_empty() {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(vec![format!(
            "{}: the device allows {}",
            image(),
            lacks.join(", ")
        )])
        .into())
    }
}

/// What this executor's multisample initializer (a per-sample render into the texture) needs of
/// the device beyond what the shader needs: sample-rate shading, a format that optimal tiling can
/// both attach and sample, and its one-sample staging image (`multisample.rs`: a 2D array of
/// `layers × samples` layers, sampled and a transfer destination) of that format and extent.
pub(super) fn multisample_initializer_support(
    context: &VulkanContext,
    label: &str,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    literal: &crate::literal::TextureLayout,
) -> Result<(), ExecutorError> {
    let mut missing = Vec::new();
    if !context
        .enabled
        .features
        .contains(&metal2vulkan::device::VkFeature::SampleRateShading)
    {
        missing.push("sampleRateShading (multisample texture initializer)".to_owned());
    }
    // SAFETY: the physical device belongs to the live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, format)
    };
    let attachment = if aspect.contains(vk::ImageAspectFlags::DEPTH) {
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
    } else {
        vk::FormatFeatureFlags::COLOR_ATTACHMENT
    };
    if !properties
        .optimal_tiling_features
        .contains(attachment | vk::FormatFeatureFlags::SAMPLED_IMAGE)
    {
        missing.push(format!(
            "{label}: format {} attachable and sampled with optimal tiling (multisample texture \
             initializer)",
            format.as_raw()
        ));
    }
    if !missing.is_empty() {
        return Err(NotSupported::NotEnabled(missing).into());
    }
    let staging_layers = literal
        .array_layers
        .checked_mul(literal.sample_count)
        .ok_or_else(|| format!("{label}: the multisample staging layer count overflows"))?;
    image_support(
        context,
        &format!("{label} (multisample initializer staging)"),
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: literal.width,
                height: literal.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(staging_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
    )
}

/// The image of the texture `label`: its authored type, format, extent, and levels, the usages
/// its descriptors need plus transfers, mutable-format when a view's format differs from the
/// texture's own (a storage alias), the one layout every view names, and a host readback when it
/// is the `observed` output. Each level is uploaded whole from its bytes ([`level_uploads`]). A
/// mipmapped texture is one the case checker admits (`case::mips`): single-sample, not sparse,
/// and an input, which no executor reads back.
fn create_slot_image(
    context: &VulkanContext,
    label: &str,
    texture: &LiteralTexture,
    texture_use: &TextureUse,
    observed: bool,
) -> Result<SlotImage, ExecutorError> {
    use crate::case::TextureType as T;
    let pixel_format = crate::candidate::engine::resources::pixel_format(texture.format);
    let format: vk::Format = metal2vulkan::metal::format::direct_format(pixel_format)
        .ok_or_else(|| format!("{label}: no Vulkan format for {pixel_format:?}"))?
        .into();
    let aspect: vk::ImageAspectFlags = metal2vulkan::metal::format::aspects(pixel_format)
        .ok_or_else(|| format!("{label}: no aspects for {pixel_format:?}"))?
        .into();
    let layout = texture_use.views[0].layout;
    if texture_use.views.iter().any(|view| view.layout != layout) {
        return Err(not_implemented(format!(
            "{label} viewed in two layouts by one plan"
        )));
    }
    let literal = texture.layout()?;
    let (image_type, extent) = image_shape(texture.texture_type, &literal)
        .ok_or_else(|| format!("{label} is a texture buffer, not an image"))?;
    let multisampled = literal.sample_count != 1;
    // Levels only where the checker admits them: a multisampled image is made and uploaded one
    // level, and a readback copies one.
    if texture.mip_levels != 1 && (multisampled || observed) {
        return Err(not_implemented(format!(
            "{label}: a mipmapped texture ({} levels) that is multisampled or read back",
            texture.mip_levels
        )));
    }
    if multisampled {
        multisample_initializer_support(context, label, format, aspect, &literal)?;
    }
    let mut flags = vk::ImageCreateFlags::empty();
    if matches!(texture.texture_type, T::Cube | T::CubeArray) {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    if texture_use.views.iter().any(|view| view.format != format) {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
    }
    // A multisampled image cannot be a copy destination: its initializer renders each sample.
    let mut usage = if !multisampled {
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
    } else if aspect.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    } else {
        vk::ImageUsageFlags::COLOR_ATTACHMENT
    };
    if texture_use.sampled {
        usage |= vk::ImageUsageFlags::SAMPLED;
    }
    if texture_use.storage {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    let info = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(image_type)
        .format(format)
        .extent(extent)
        .mip_levels(texture.mip_levels)
        .array_layers(literal.array_layers)
        .samples(sample_count(literal.sample_count)?)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let (upload, copies) = level_uploads(texture, aspect)?;
    let (image, memory) = create_image(context, label, &info)?;
    let destroy = || {
        // SAFETY: the image and memory were created above and are unused.
        unsafe {
            context.device.destroy_image(image, None);
            context.device.free_memory(memory, None);
        }
    };
    let staging = if multisampled {
        if let Err(error) = initialize_multisample_image(
            context,
            image,
            format,
            aspect,
            literal,
            texture.format,
            &texture.bytes,
        ) {
            destroy();
            return Err(error.into());
        }
        None
    } else {
        match create_host_buffer(
            context,
            &format!("{label} staging"),
            &upload,
            vk::BufferUsageFlags::TRANSFER_SRC,
        ) {
            Ok(staging) => Some(staging),
            Err(error) => {
                destroy();
                return Err(error.into());
            }
        }
    };
    let readback = if observed && !multisampled {
        match create_host_buffer(
            context,
            &format!("{label} readback"),
            &texture.bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
        ) {
            Ok(readback) => Some(readback),
            Err(error) => {
                destroy();
                return Err(error.into());
            }
        }
    } else {
        None
    };
    Ok(SlotImage {
        image,
        memory,
        aspect,
        extent,
        layers: literal.array_layers,
        levels: texture.mip_levels,
        layout,
        staging,
        copies,
        readback,
    })
}

/// The image type and extent of a texture of `texture_type` laid out as `layout`; `None` for a
/// texture buffer, which is no image.
fn image_shape(
    texture_type: crate::case::TextureType,
    layout: &TextureLayout,
) -> Option<(vk::ImageType, vk::Extent3D)> {
    use crate::case::TextureType as T;
    let (image_type, [width, height, depth]) = match texture_type {
        T::Buffer => return None,
        T::D1 | T::D1Array => (vk::ImageType::TYPE_1D, [layout.width, 1, 1]),
        T::D3 => (
            vk::ImageType::TYPE_3D,
            [layout.width, layout.height, layout.depth],
        ),
        _ => (vk::ImageType::TYPE_2D, [layout.width, layout.height, 1]),
    };
    Some((
        image_type,
        vk::Extent3D {
            width,
            height,
            depth,
        },
    ))
}

/// The bytes and copies that upload every level of `texture`, whose image has `aspect`: one copy
/// per level, of its whole extent and every layer, from that level's bytes
/// ([`LiteralTexture::level`]). Each level starts at a multiple of the texel size and of 4, as a
/// copy's `bufferOffset` must (`VUID-vkCmdCopyBufferToImage-dstImage-07975` and `-07978`), so
/// the bytes are the literal's own unless a level ends off that alignment, where the next starts
/// past zero padding.
fn level_uploads(
    texture: &LiteralTexture,
    aspect: vk::ImageAspectFlags,
) -> Result<(Cow<'_, [u8]>, Vec<vk::BufferImageCopy>), String> {
    let texel = texture.format.bytes_per_pixel() * texture.sample_count as usize;
    // The least common multiple of the texel size and 4.
    let align = [1, 2, 4]
        .into_iter()
        .map(|texels| texels * texel)
        .find(|bytes| bytes % 4 == 0)
        .expect("four texels' bytes are a multiple of 4");
    let levels = (0..texture.mip_levels)
        .map(|level| texture.level(level))
        .collect::<Result<Vec<_>, _>>()?;
    let (mut copies, mut starts) = (Vec::with_capacity(levels.len()), Vec::new());
    let (mut end, mut padded) = (0usize, false);
    for (level, (dimensions, bytes)) in (0..).zip(&levels) {
        let start = end.next_multiple_of(align);
        padded |= start != end;
        let layout = texture_layout(texture.texture_type, *dimensions, texture.sample_count)?;
        let (_, extent) =
            image_shape(texture.texture_type, &layout).ok_or("a texture buffer is no image")?;
        copies.push(
            vk::BufferImageCopy::default()
                .buffer_offset(start as vk::DeviceSize)
                .image_subresource(vk::ImageSubresourceLayers {
                    mip_level: level,
                    ..layers(aspect, layout.array_layers)
                })
                .image_extent(extent),
        );
        starts.push(start);
        end = start + bytes.len();
    }
    if !padded {
        return Ok((Cow::Borrowed(&texture.bytes), copies));
    }
    let mut staged = vec![0; end];
    for ((_, bytes), start) in levels.iter().zip(starts) {
        staged[start..start + bytes.len()].copy_from_slice(bytes);
    }
    Ok((Cow::Owned(staged), copies))
}
