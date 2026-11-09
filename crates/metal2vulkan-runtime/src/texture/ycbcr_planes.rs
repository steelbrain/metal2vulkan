//! A two-plane YCbCr texture's planes read in place through the device's own sampler Y′CbCr
//! conversion (`ycbcr_image.rs`'s planes in place, `metal2vulkan::bind::YCbCrPlanesSampler`): a
//! linear image of the multi-planar 4:2:0 format bound over the planes' own bytes, which a slot of
//! a function whose every sample the device's sampler takes samples through an immutable sampler
//! of the device's conversion, where the device converts the format with Metal's reconstruction.
//!
//! - **Which.** A texture whose planes a function reads in place (`PlanesInPlace`), on a device
//!   whose linear tiling of `G8_B8R8_2PLANE_420_UNORM` (8-bit codes) or
//!   `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` (10-bit) reports `SAMPLED_IMAGE`,
//!   `MIDPOINT_CHROMA_SAMPLES` and a linear chroma filter, with no defect in the format
//!   (`RuntimeDriverDefect::TwoPlaneChromaMisplaced`, `TenBitCodesReadAsWords`;
//!   `ycbcr_device.rs`), of an even width and height within the format's linear limits, whose
//!   luma plane comes first.
//! - **Its memory.** The planes' bytes in memory of the runtime's own that it allocated (a
//!   buffer's pooled memory or a heap's): the image starts `PREINITIALIZED`, whose move to
//!   `GENERAL` keeps its bytes (the first move, `first_move.rs`). Over fresh pages
//!   (`fresh_pages.rs`), a no-copy buffer's pages (imported by their pointer or as dma-bufs) or the
//!   caller's `VkBuffer`, none: an image bound to imported memory starts `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`), whose move the device's check covers for
//!   single-plane images alone (`in_place_check.rs`), so the texel buffer reads the planes there.
//! - **Its layout** is the driver's: the image is taken where its planes lie at the surface's
//!   offsets and row pitches from the luma plane's first byte, at an offset of the memory that
//!   meets its alignment, within the buffer, in a memory type it takes; elsewhere the texel buffer
//!   reads the planes.
//! - **What a sample answers** is the device's: its conversion of the codes (the surface's matrix
//!   and range, Metal's reconstruction) and its filter, announced `ycbcr-filter-not-exact` where
//!   the image is made.
//!
//! Each texture whose planes are read in place and take no such image is counted by the reason
//! (`WorkCounts::ycbcr_planes_images_kept_*`).

use super::first_move::{FirstMove, Fresh};
use crate::buffer_texture::Texels;
use crate::creation;
use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::resource::ResourceKey;
use crate::ycbcr_device::YCbCrSampler;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{YCbCrInPlace, YCbCrPlanes, YCbCrPlanesSampler};
use std::sync::{Arc, Mutex};

/// Why a texture whose planes a function reads in place samples them through no image of the
/// device's conversion (the module docs), in `WorkCounts::ycbcr_planes_images_kept_*`'s order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeptTexels {
    /// The device does not convert the format with Metal's reconstruction as a linear image of
    /// the texture's size.
    Conversion,
    /// The planes lie in memory no such image is made over: fresh pages, a no-copy buffer's
    /// pages, the caller's `VkBuffer`, or memory the runtime exports (`exported_memory.rs`), whose
    /// image would name its handle type (`VUID-vkBindImageMemory-memory-02728`) and so start
    /// `UNDEFINED`, which no check covers for such an image; or a buffer's dedicated allocation
    /// (`VUID-vkBindImageMemory-memory-10926`). Its memory's record decides
    /// (`memory.rs`'s `Memory::bound_memory`): only memory of no export or import takes it.
    Memory,
    /// The driver's image lays the planes out otherwise, or does not fit the buffer there.
    Layout,
    /// The planes lie in a block of the runtime's buffers allocated exportable and never
    /// exported (`memory.rs`'s "Buffers in exportable blocks"), whose image would name one of its
    /// handle types (`VUID-vkBindImageMemory-memory-02728`) and so start `UNDEFINED`, which the
    /// device's check of such blocks covers for single-plane images alone.
    ExportableBlock,
}

impl KeptTexels {
    /// The reasons, in their counters' order.
    pub(crate) const ALL: [KeptTexels; 4] = [
        KeptTexels::Conversion,
        KeptTexels::Memory,
        KeptTexels::Layout,
        KeptTexels::ExportableBlock,
    ];
}

/// The image of a texture's planes the device's conversion samples in place (the module docs):
/// its view, made with the conversion every immutable sampler of its sampler bakes, and its first
/// move until a submission takes it. It holds the texels, and so the memory it is bound to.
pub(crate) struct PlanesImage {
    device: Device,
    image: vk::Image,
    pub(crate) view: vk::ImageView,
    _texels: Arc<Texels>,
    first: Mutex<Option<FirstMove>>,
}

impl Fresh for PlanesImage {
    fn take(&self) -> Vec<FirstMove> {
        lock(&self.first).take().into_iter().collect()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        let mut first = lock(&self.first);
        if first.is_none() {
            *first = moves.into_iter().next();
        }
    }

    /// Bound for good: memory of the runtime's own never moves.
    fn rebindable(&self) -> Option<ResourceKey> {
        None
    }
}

impl Drop for PlanesImage {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them: the commands that bind the view hold the texture,
        // and a submission that takes the move holds this image, until they complete.
        unsafe {
            release::object(raw, self.view);
            release::object(raw, self.image);
        }
    }
}

impl Device {
    /// The image of `planes` (read in place as `words`, from byte `start` of `texels`' buffer)
    /// that the device's conversion samples (the module docs), or why there is none.
    pub(super) fn planes_image(
        &self,
        texels: &Arc<Texels>,
        (planes, words): (&YCbCrPlanes, &YCbCrInPlace),
        start: u64,
    ) -> Result<Result<Arc<PlanesImage>, KeptTexels>, Error> {
        let sampler = words.sampler();
        let [width, height] = planes.size;
        let Some(ycbcr) = YCbCrSampler::planes(self, sampler)
            .filter(|ycbcr| ycbcr.holds_planes(self, [width, height]))
        else {
            return Ok(Err(KeptTexels::Conversion));
        };
        let bound = texels.buffer.bound_memory();
        if bound == crate::memory::BoundMemory::Blocks {
            return Ok(Err(KeptTexels::ExportableBlock));
        }
        if texels.buffer.caller_memory().is_some() || bound != crate::memory::BoundMemory::Own {
            return Ok(Err(KeptTexels::Memory));
        }
        let [luma_at, luma_row] = planes.luma.map(u64::from);
        let [chroma_at, chroma_row] = planes.chroma.map(u64::from);
        if chroma_at <= luma_at {
            return Ok(Err(KeptTexels::Layout));
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        let format: vk::Format = sampler.format().into();
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::PREINITIALIZED),
        );
        let limits = self.image_limits(&info, &[])?;
        if width > limits.max_extent.width || height > limits.max_extent.height {
            return Ok(Err(KeptTexels::Conversion));
        }
        // SAFETY: the device is live and `info` valid and within its limits.
        let image = unsafe { creation::image(raw_device, &info) }?;
        let unwind = |image| {
            // SAFETY: the image is live, and no command uses it.
            unsafe { release::object(raw_device, image) };
        };
        let (requirements, dedicated) = super::in_place_image::requirements_of(raw_device, image);
        let plane = |aspect| vk::ImageSubresource {
            aspect_mask: aspect,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: `image` is a live linear image of a two-plane format, whose planes are its
        // subresources' aspects (`VUID-vkGetImageSubresourceLayout-format-08886`).
        let [luma, chroma] = [vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1]
            .map(|aspect| unsafe { raw_device.get_image_subresource_layout(image, plane(aspect)) });
        exit_guard::called(Call::GetImageSubresourceLayout);
        let (memory, buffer_offset, type_index) = texels.buffer.memory();
        let at = buffer_offset + start + luma_at;
        let room = texels
            .buffer
            .padded_length()
            .saturating_sub(start + luma_at);
        let lies = luma.offset == 0
            && luma.row_pitch == luma_row
            && chroma.offset == chroma_at - luma_at
            && chroma.row_pitch == chroma_row;
        if dedicated
            || !lies
            || !at.is_multiple_of(requirements.alignment.max(1))
            || requirements.size > room
            || requirements.memory_type_bits & (1 << type_index) == 0
        {
            unwind(image);
            return Ok(Err(KeptTexels::Layout));
        }
        // SAFETY: the image is live and unbound; the memory is the buffer's, which the image
        // holds through the texels, in a type the image takes, at an offset that meets its
        // alignment, its size within the buffer.
        if let Err(error) = unsafe { creation::bind_image(raw_device, image, memory, at) } {
            unwind(image);
            return Err(vulkan(error));
        }
        let conversion = match self.made_ycbcr_conversion(ycbcr) {
            Ok(conversion) => conversion,
            Err(error) => {
                unwind(image);
                return Err(error);
            }
        };
        let mut chained = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .push_next(&mut chained);
        // SAFETY: the image is live and bound; the conversion is of its format.
        let view = match unsafe { creation::image_view(raw_device, &view_info) } {
            Ok(view) => view,
            Err(error) => {
                unwind(image);
                return Err(vulkan(error));
            }
        };
        // Leaving `PREINITIALIZED` keeps the bytes the image holds: the planes'. Its colour
        // aspect is every plane of the image, which is not disjoint.
        let first = FirstMove::new(
            image,
            sampler.format(),
            vk::ImageLayout::PREINITIALIZED,
            None,
        );
        let made = Arc::new(PlanesImage {
            device: self.clone(),
            image,
            view,
            _texels: Arc::clone(texels),
            first: Mutex::new(Some(first)),
        });
        let fresh: Arc<dyn Fresh> = made.clone();
        shared.textures.first_moves.register(Arc::downgrade(&fresh));
        shared.counts.ycbcr_planes_image();
        announce(self, sampler);
        Ok(Ok(made))
    }
}

/// The notice of a planes image of `sampler` (the module docs).
fn announce(device: &Device, sampler: YCbCrPlanesSampler) {
    device.degrade(Degradation::YCbCrFilterNotExact, || {
        (
            "texture".to_owned(),
            format!(
                "a two-plane YCbCr texture of {}-bit 4:2:0 codes ({:?}, {:?}) whose planes a \
                 function samples in place reads the device's own sampler Y′CbCr conversion and \
                 filter of them, not Metal's fixed-point conversion and exact filter",
                sampler.bits(),
                sampler.range(),
                sampler.matrix()
            ),
        )
    });
}
