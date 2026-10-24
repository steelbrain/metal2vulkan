//! A device's neutral images and samplers: what a binding plan writes where a function statically
//! uses a texture or sampler slot that nothing is bound to (`NeutralResource`).
//!
//! The device makes each kind at its first use and keeps it for its lifetime; the kinds are a
//! finite set (the compile layer's `Device::neutral_kinds`). Whatever may draw in a host step makes
//! the neutrals its plans may bind before, on the program's thread (`binding::make_neutrals`: a
//! compute execution's commit, and a render state's creation where indirect command buffers may
//! draw it, and an execution's commit), so a step finds them; one no earlier settle made a host
//! step makes itself, inside its section, whose table holds the entry points (`creation.rs`,
//! `exit_guard::HOST_STEP`), its first work taken by the step's batch's submission, after the
//! step. A neutral image is one texel (six layers for a cube type)
//! of the kind's format and sample count, cleared to zero, in the `GENERAL` layout like every
//! runtime image, so a read returns zero and a store reaches nothing another command reads. A
//! neutral sampler is the default one: nearest filtering, clamped to the edge, depth comparison
//! where the kind asks for it. A neutral texel buffer is a view of one texel of zero bytes in a
//! buffer of its own, made for the one kind of texel buffer (uniform or storage) its binding is.
//!
//! **No wait**. Making a neutral waits for no GPU work: a neutral image's move to `GENERAL` and its
//! clear, and a neutral texel buffer's fill with zeros, are first work the device's next submission
//! records before any of its commands (`texture/first_move.rs`), as a new texture's and buffer's
//! are, where they were an internal submission the making waited for, behind every command buffer
//! submitted before it on the device's first `VkQueue`. Every submission that may bind a neutral is
//! made after it was made, so its first work, or a batch it waits for, does the neutral's first. A
//! neutral image takes memory of its own where binds wait: bound into a block beside another
//! neutral that queued work samples, its bind waited for that work on NVIDIA's driver; ANV and
//! RADV bind at once, so there it shares a pooled block (`memory.rs`'s `Fit`).

use super::device::lock;
use super::error::{vulkan, Error};
use super::memory::{Memory, MemoryClass, Placement};
use super::resource::ResourceKey;
use super::texture::first_move::{Filled, FirstMove, Fresh};
use super::Device;
use crate::{creation, release};
use ash::vk;
use metal2vulkan::bind::NeutralResource;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::vk::Format;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// One neutral image, its view and its memory.
struct NeutralImage {
    image: vk::Image,
    view: vk::ImageView,
    placement: Placement,
    /// Its first move, with its clear, until a submission takes it (the module docs).
    first: Mutex<Option<FirstMove>>,
}

impl Fresh for NeutralImage {
    fn take(&self) -> Vec<FirstMove> {
        lock(&self.first).take().into_iter().collect()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        let mut first = lock(&self.first);
        if first.is_none() {
            *first = moves.into_iter().next();
        }
    }

    fn rebindable(&self) -> Option<ResourceKey> {
        None
    }
}

/// One neutral texel buffer: its buffer, its view and its memory.
struct NeutralTexels {
    buffer: vk::Buffer,
    view: vk::BufferView,
    placement: Placement,
}

impl Filled for NeutralTexels {
    /// Nothing to hold: the device keeps a neutral's memory for its lifetime and frees it once
    /// idle.
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(()))
    }
}

/// One neutral zero buffer and its memory: the buffer a plan binds where a function reads a
/// **read-only** buffer binding that nothing is bound to
/// ([`NeutralResource::ZeroBuffer`]).
struct NeutralZeros {
    buffer: vk::Buffer,
    placement: Placement,
}

impl Filled for NeutralZeros {
    /// Nothing to hold, as [`NeutralTexels`]: the device keeps it for its lifetime.
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>> {
        Some(Box::new(()))
    }
}

/// A device's neutral images, texel buffers, zero buffers and samplers, made at first use.
#[derive(Default)]
pub(super) struct Neutrals {
    images: HashMap<NeutralResource, Arc<NeutralImage>>,
    /// By format and whether the view is a storage texel buffer's.
    texels: HashMap<(Format, bool), Arc<NeutralTexels>>,
    /// By their length in bytes, which the plan asks for (`zero_buffer_bytes`, fixed for the
    /// device): the read-only zero buffers, and how many the device has made.
    ///
    /// Before, a read-only buffer binding that nothing is bound to took a fresh zero-filled
    /// region from the command buffer's upload arena at **every draw and dispatch** that bound
    /// one, and a region of `maxUniformBufferRange` capped at 64 KiB is past half the arena's
    /// chunk, so each took a dedicated 64 KiB `VkBuffer` of its own, allocated and written to
    /// zero. A triangle drawn with neither of its two `setBytes` slots set cost 128 KiB of
    /// allocate-and-zero a draw. Zeros of a length the device's own limits fix cannot have
    /// changed since the device was made, so the device makes each length once and keeps it, as
    /// it keeps its neutral images, texel buffers and samplers.
    zeros: HashMap<u64, Arc<NeutralZeros>>,
    /// How many zero buffers the device has made, for the check that draws share one
    /// (`tests::neutral_zeros_shared`).
    zeros_made: u64,
    /// By whether depth comparison is enabled.
    samplers: HashMap<bool, vk::Sampler>,
}

impl Neutrals {
    /// Destroy every neutral object of `device`, freeing their memory into `memory`. The device
    /// is idle.
    pub(super) fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        for image in self.images.values() {
            // SAFETY: the device is idle, so nothing uses the view or the image.
            unsafe {
                release::object(device, image.view);
                release::object(device, image.image);
            }
            memory.free(device, &image.placement);
        }
        for zeros in self.zeros.values() {
            // SAFETY: the device is idle, so nothing uses the buffer.
            unsafe { release::object(device, zeros.buffer) };
            memory.free(device, &zeros.placement);
        }
        for texels in self.texels.values() {
            // SAFETY: as above.
            unsafe {
                release::object(device, texels.view);
                release::object(device, texels.buffer);
            }
            memory.free(device, &texels.placement);
        }
        for &sampler in self.samplers.values() {
            // SAFETY: as above.
            unsafe { release::object(device, sampler) };
        }
    }
}

/// The image type, layer count and create flags of a one-texel image viewed as `view_type`.
fn image_shape(view_type: vk::ImageViewType) -> (vk::ImageType, u32, vk::ImageCreateFlags) {
    use vk::ImageViewType as V;
    match view_type {
        V::TYPE_1D | V::TYPE_1D_ARRAY => (vk::ImageType::TYPE_1D, 1, vk::ImageCreateFlags::empty()),
        V::TYPE_3D => (vk::ImageType::TYPE_3D, 1, vk::ImageCreateFlags::empty()),
        V::CUBE | V::CUBE_ARRAY => (
            vk::ImageType::TYPE_2D,
            6,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        ),
        _ => (vk::ImageType::TYPE_2D, 1, vk::ImageCreateFlags::empty()),
    }
}

impl Device {
    /// The view of the neutral image of `kind`, made at its first use.
    pub(super) fn neutral_view(&self, kind: NeutralResource) -> Result<vk::ImageView, Error> {
        let shared = self.shared();
        let mut neutrals = lock(&shared.textures.neutrals);
        if let Some(image) = neutrals.images.get(&kind) {
            return Ok(image.view);
        }
        let image = self.new_neutral_image(kind)?;
        let view = image.view;
        neutrals.images.insert(kind, image);
        Ok(view)
    }

    /// The view of the neutral texel buffer of `format`, a storage texel buffer's (`storage`) or
    /// a uniform one's, made at its first use.
    pub(super) fn neutral_texel_view(
        &self,
        format: Format,
        storage: bool,
    ) -> Result<vk::BufferView, Error> {
        let shared = self.shared();
        let mut neutrals = lock(&shared.textures.neutrals);
        if let Some(texels) = neutrals.texels.get(&(format, storage)) {
            return Ok(texels.view);
        }
        let texels = self.new_neutral_texels(format, storage)?;
        let view = texels.view;
        neutrals.texels.insert((format, storage), texels);
        Ok(view)
    }

    /// A new neutral texel buffer of `format`: one texel of zero bytes, viewed as a storage
    /// texel buffer (`storage`) or a uniform one.
    fn new_neutral_texels(
        &self,
        format: Format,
        storage: bool,
    ) -> Result<Arc<NeutralTexels>, Error> {
        let (usage, needed) = if storage {
            (
                vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                FormatUsage::StorageTexelBuffer,
            )
        } else {
            (
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                FormatUsage::UniformTexelBuffer,
            )
        };
        let bytes = format
            .block()
            .map(|block| u64::from(block.size))
            .filter(|_| {
                self.compile_device()
                    .profile()
                    .supports_format(format, needed)
            })
            .ok_or_else(|| super::error::internal("a neutral texel buffer of a format it lacks"))?;
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(usage | vk::BufferUsageFlags::TRANSFER_DST),
        );
        // SAFETY: the device is live and `info` valid.
        let buffer = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(buffer) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, buffer) };
                return Err(error);
            }
        };
        let release = |placement: &Placement| {
            lock(&shared.memory).free(raw_device, placement);
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, buffer) };
        };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        if let Err(error) = unsafe {
            crate::creation::bind_buffer(raw_device, buffer, placement.memory, placement.offset)
        } {
            release(&placement);
            return Err(vulkan(error));
        }
        let view_info = vk::BufferViewCreateInfo::default()
            .buffer(buffer)
            .format(format.into())
            .offset(0)
            .range(bytes);
        // SAFETY: the buffer is live and bound, made for the one texel buffer usage `format`
        // serves, and holds one texel of it.
        let view = match unsafe { raw_device.create_buffer_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                release(&placement);
                return Err(vulkan(error));
            }
        };
        let texels = Arc::new(NeutralTexels {
            buffer,
            view,
            placement,
        });
        let weak: Weak<NeutralTexels> = Arc::downgrade(&texels);
        let keeper: Weak<dyn Filled> = weak;
        let filled = (shared.textures.first_moves).register_fill(self, keeper, buffer, 0..bytes);
        if let Err(error) = filled {
            // SAFETY: the view and the buffer are unused: no submission took the fill.
            unsafe {
                release::object(raw_device, texels.view);
                release::object(raw_device, texels.buffer);
            }
            lock(&shared.memory).free(raw_device, &texels.placement);
            return Err(error);
        }
        Ok(texels)
    }

    /// The device's read-only zero buffer of `bytes` bytes, made at its first use and kept for
    /// the device's lifetime ([`Neutrals::zeros`]).
    ///
    /// Only a **read-only** binding may be served from it: every draw and dispatch of the device
    /// binds the one buffer, so a shader that could store through the binding would be writing
    /// zeros another command reads. A Vulkan uniform-buffer descriptor is read-only by
    /// construction, which is the binding `binding::descriptor` serves from here; a writable one
    /// keeps its own region, as before.
    pub(super) fn neutral_zero_buffer(&self, bytes: u64) -> Result<vk::Buffer, Error> {
        let shared = self.shared();
        if let Some(zeros) = lock(&shared.textures.neutrals).zeros.get(&bytes) {
            return Ok(zeros.buffer);
        }
        let zeros = self.new_neutral_zeros(bytes)?;
        let mut neutrals = lock(&shared.textures.neutrals);
        neutrals.zeros_made += 1;
        Ok(neutrals.zeros.entry(bytes).or_insert(zeros).buffer)
    }

    /// How many zero buffers the device has made: one per length its plans ask for, however many
    /// draws and dispatches bind one (`tests::neutral_zeros_shared`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn neutral_zeros_made(&self) -> u64 {
        lock(&self.shared().textures.neutrals).zeros_made
    }

    /// A new zero buffer of `bytes` bytes: a `Private` buffer filled with zeros, as
    /// [`Device::new_neutral_texels`] makes its one texel of zeros, so the fill is the device's
    /// next submission's first work and waits for nothing.
    fn new_neutral_zeros(&self, bytes: u64) -> Result<Arc<NeutralZeros>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
        );
        // SAFETY: the device is live and `info` valid.
        let buffer = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(buffer) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, buffer) };
                return Err(error);
            }
        };
        let release = |placement: &Placement| {
            lock(&shared.memory).free(raw_device, placement);
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, buffer) };
        };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        if let Err(error) = unsafe {
            crate::creation::bind_buffer(raw_device, buffer, placement.memory, placement.offset)
        } {
            release(&placement);
            return Err(vulkan(error));
        }
        let zeros = Arc::new(NeutralZeros { buffer, placement });
        let weak: Weak<NeutralZeros> = Arc::downgrade(&zeros);
        let keeper: Weak<dyn Filled> = weak;
        let filled = (shared.textures.first_moves).register_fill(self, keeper, buffer, 0..bytes);
        if let Err(error) = filled {
            // SAFETY: the buffer is unused: no submission took the fill.
            release(&zeros.placement);
            return Err(error);
        }
        Ok(zeros)
    }

    /// The neutral sampler, with depth comparison when `compare` is set.
    pub(super) fn neutral_sampler(&self, compare: bool) -> Result<vk::Sampler, Error> {
        let shared = self.shared();
        let mut neutrals = lock(&shared.textures.neutrals);
        if let Some(&sampler) = neutrals.samplers.get(&compare) {
            return Ok(sampler);
        }
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        if compare {
            info = info.compare_enable(true).compare_op(vk::CompareOp::NEVER);
        }
        // SAFETY: the device is live and `info` valid. Counted against the device's limit; it
        // lives as long as the device.
        let sampler = unsafe { shared.create_sampler(&info) }?;
        neutrals.samplers.insert(compare, sampler);
        Ok(sampler)
    }

    /// A new neutral image of `kind`, zero and in the `GENERAL` layout.
    fn new_neutral_image(&self, kind: NeutralResource) -> Result<Arc<NeutralImage>, Error> {
        let NeutralResource::Image {
            view_type,
            depth_compare,
            samples,
            storage_format,
            ..
        } = kind
        else {
            return Err(super::error::internal(
                "a neutral image of a kind that is no image",
            ));
        };
        let kind_format = kind
            .format()
            .ok_or_else(|| super::error::internal("a neutral image without a format"))?;
        let format: vk::Format = kind_format.into();
        let view_type: vk::ImageViewType = view_type.into();
        let (image_type, layers, flags) = image_shape(view_type);
        let usage = if storage_format.is_some() {
            vk::ImageUsageFlags::STORAGE
        } else {
            vk::ImageUsageFlags::SAMPLED
        } | vk::ImageUsageFlags::TRANSFER_DST;
        let aspect = if depth_compare {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
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
                .samples(vk::SampleCountFlags::from_raw(samples))
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        );
        // SAFETY: the device is live; the kind's format has the usage at the kind's sample count
        // on every device that asks for it (the mandatory formats and sample counts, or a storage
        // format the profile lists, `NeutralResource`).
        let image = unsafe { creation::image(raw_device, &info) }?;
        // SAFETY: `image` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
        let requirements = shared.textures.placement().place(requirements);
        // Bound while work that samples another neutral may be queued: memory of its own, which
        // no submission used, where such a bind waits for that work (`memory.rs`'s `Fit`).
        let placement = match crate::memory::allocate_image_while_queued(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw_device, image) };
                return Err(error);
            }
        };
        let release = |placement: &Placement| {
            lock(&shared.memory).free(raw_device, placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw_device, image) };
        };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) = unsafe {
            crate::creation::bind_image(raw_device, image, placement.memory, placement.offset)
        } {
            release(&placement);
            return Err(vulkan(error));
        }
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(range);
        // SAFETY: the image is live and bound; the view type takes its layers.
        let view = match unsafe { crate::creation::image_view(raw_device, &view_info) } {
            Ok(view) => view,
            Err(error) => {
                release(&placement);
                return Err(vulkan(error));
            }
        };
        let first = FirstMove::new(image, kind_format, vk::ImageLayout::UNDEFINED, None).cleared();
        let neutral = Arc::new(NeutralImage {
            image,
            view,
            placement,
            first: Mutex::new(Some(first)),
        });
        let weak: Weak<NeutralImage> = Arc::downgrade(&neutral);
        let fresh: Weak<dyn Fresh> = weak;
        shared.textures.first_moves.register(fresh);
        Ok(neutral)
    }
}
