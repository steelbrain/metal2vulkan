//! A sparse heap's texture: Metal's answers of it (its tile, `firstMipmapInTail`,
//! `tailSizeInBytes`), and its `VkImage`, created with `SPARSE_BINDING | SPARSE_RESIDENCY`, laid
//! out in the device's sparse units: blocks of `imageGranularity` texels (`alignment` bytes each)
//! and the mip tail from `imageMipTailFirstLod`, one per layer or one for all (`SINGLE_MIPTAIL`);
//! or an ordinary image (`emulated.rs`). Either has a tile map where its type has one.

use super::emulated::{Backing, TileMapBuffer};
use super::mapping::{MetalUnit, SparseState, UnitGrid};
use super::{first_mipmap_in_tail, tail_size_in_bytes, tile_texels, SparsePageSize};
use crate::error::{internal, Error, RuntimeGap};
use crate::heap::{Heap, WeakHeap};
use metal2vulkan::error::{Feature, SparseNeed, Unsupported};
use metal2vulkan::metal::{PixelFormat, TextureType};

use crate::release;
use crate::Device;
use ash::vk;
use std::sync::Mutex;

/// The refusal of a sparse heap or texture on a device that lacks `need`, where a function may
/// read the image's own residency (the sparse arm of a slot without a tile map; elsewhere the
/// runtime emulates the texture).
///
/// Two kinds of need, kept apart, because a consumer reads the difference and acts on it.
///
/// **Ours.** A device that binds no sparse memory (`sparseBinding`, or the queue family's bit)
/// makes no sparse resident image, but an ordinary image whose unmapped texels a tile map answers
/// is an exact path, and it is not built: the runtime's gap, `NotImplemented(SparseTileMaps)`. A
/// sparse image whose memory needs a metadata aspect bound is the same shape — the binding exists
/// and the runtime does not do it.
///
/// **The device's.** Residency of the texture's type or sample count, and the sparse image formats
/// the device lists, are Vulkan 1.0 capabilities a device either has or does not. They are outside
/// the compile layer's capability vocabulary on purpose (`super::device`), so the refusal names
/// them with `Needs::DeviceCapability`. It used to say `NotRepresentable`, which claims no Vulkan
/// mechanism exists on any device: a device without `sparseResidency2Samples` therefore made our
/// refusal read as unbuilt work of ours, and the conformance suite scored it as a refusal rather
/// than as the venue's own gap.
pub(crate) fn sparse_refusal(need: SparseNeed) -> Error {
    let capability = match need {
        // The exact path exists and the runtime has not built it: ours to answer for.
        SparseNeed::SparseBinding | SparseNeed::SparseBindingQueue | SparseNeed::SparseMetadata => {
            return Error::NotImplemented(RuntimeGap::SparseTileMaps)
        }
        SparseNeed::SparseResidencyImage2D => "sparseResidencyImage2D",
        SparseNeed::SparseResidencyImage3D => "sparseResidencyImage3D",
        SparseNeed::SparseResidencySamples(2) => "sparseResidency2Samples",
        SparseNeed::SparseResidencySamples(4) => "sparseResidency4Samples",
        SparseNeed::SparseResidencySamples(8) => "sparseResidency8Samples",
        SparseNeed::SparseResidencySamples(16) => "sparseResidency16Samples",
        // A sample count Vulkan has no `sparseResidencyNSamples` member for: still the device's
        // own gap, named for what it is rather than by a member that does not exist.
        SparseNeed::SparseResidencySamples(_) => "sparse residency of the texture's sample count",
        SparseNeed::SparseImageFormat => {
            "sparse residency of the texture's format, type, sample count and usage \
             (vkGetPhysicalDeviceSparseImageFormatProperties lists none)"
        }
        // A need the vocabulary gained since: not claimed to be the device's without knowing.
        _ => {
            return Error::Unsupported(Unsupported::not_implemented(Feature::SparseTextures {
                missing: need,
            }))
        }
    };
    Error::Unsupported(Unsupported::device_capability(
        Feature::SparseTextures { missing: need },
        capability,
    ))
}

/// How a sparse image's memory is laid out in the device's sparse units.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct VkSparseLayout {
    /// The aspect the units bind: color, depth or stencil.
    pub(crate) aspect: vk::ImageAspectFlags,
    /// A block's texels (`imageGranularity`).
    pub(crate) granularity: [u32; 3],
    /// A block's bytes, and the alignment of every bind's memory offset (`alignment`).
    pub(crate) block_bytes: u64,
    /// The memory types the image's memory may be in.
    pub(crate) memory_type_bits: u32,
    /// The first level in the mip tail (`imageMipTailFirstLod`): the level count without one.
    pub(crate) mip_tail_first_lod: u32,
    /// The tail's bytes (`imageMipTailSize`), at `imageMipTailOffset` plus a layer's
    /// `imageMipTailStride` in the image's opaque space.
    pub(crate) mip_tail_size: u64,
    pub(crate) mip_tail_offset: u64,
    pub(crate) mip_tail_stride: u64,
    /// One tail for all layers (`SINGLE_MIPTAIL`).
    pub(crate) single_mip_tail: bool,
}

/// A sparse texture's image beyond the `VkImage`: its heap, Metal's answers of it, its units,
/// how its texels are held, and its tile map.
pub(crate) struct SparseImage {
    /// The heap whose tiles its mappings take, held weakly: the texture objects over the image
    /// hold the heap (`Texture::heap`), as Metal's texture keeps its heap, and an image the
    /// runtime keeps after them (a gone texture's, retired for the command buffers that may
    /// reach it by its resource ID) keeps no heap, so the heap goes as Metal frees it.
    pub(crate) heap: WeakHeap,
    /// Its device, for its teardown.
    pub(crate) device: Device,
    pub(crate) page: SparsePageSize,
    pub(crate) tail_size_in_bytes: u64,
    /// Its units, Metal's and the device's: its Metal shape and layers (Metal's slices, a cube's
    /// faces each, 1 for a 3D texture), Metal's tile and `firstMipmapInTail`, and its device
    /// layout.
    pub(crate) grid: UnitGrid,
    /// What it has mapped and bound, as the scheduler applies its updates in submission order.
    pub(crate) state: Mutex<SparseState>,
    /// How its texels are held: a sparse resident image, or an ordinary one (`emulated.rs`).
    pub(crate) backing: Backing,
    /// Zeros its unmapped texels are copied from: its heap's, more than any tile's bytes.
    pub(crate) zeros: crate::buffer::Buffer,
    /// How the runtime zeroes its unmapped texels, so a device unit binds while any of its Metal
    /// units is mapped.
    pub(crate) zeroing: Zeroing,
    /// The buffer a function reads its residency from, where its type has a tile map: every
    /// single-sampled 2D, 2D array and 3D texture.
    pub(crate) tile_map: Option<TileMapBuffer>,
}

/// How the runtime writes zero into a sparse texture's texels where it keeps them at zero
/// (`emulated.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Zeroing {
    /// Copies of zeros, into a single-sampled image that takes transfers.
    Copies,
    /// Clears (`vkCmdClearAttachments`) in a render pass of the image's level and layer, of a
    /// multisampled image, which no copy writes, made an attachment of its format.
    Clears,
}

/// The Metal shape of a sparse texture: its type, format, sample count, level 0's extent and
/// its levels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseShape {
    pub(crate) texture_type: TextureType,
    pub(crate) pixel_format: PixelFormat,
    pub(crate) samples: u64,
    pub(crate) extent: [u64; 3],
    pub(crate) levels: u64,
}

impl SparseShape {
    /// Metal's tile of this shape in pages of `page`.
    pub(crate) fn tile(&self, page: SparsePageSize) -> [u64; 3] {
        tile_texels(self.pixel_format, self.samples, page.bytes())
    }
}

impl Device {
    /// What the device lacks to make a sparse resident image of `info` (its type, sample count,
    /// format and usage), if anything. `aspects` are the image's.
    pub(crate) fn check_sparse_image(
        &self,
        info: &vk::ImageCreateInfo<'_>,
        aspects: vk::ImageAspectFlags,
    ) -> Result<(), SparseNeed> {
        let shared = self.shared();
        shared.sparse.image(info.image_type, info.samples)?;
        // SAFETY: the physical device belongs to the live instance.
        let formats = unsafe {
            shared
                .instance
                .raw
                .get_physical_device_sparse_image_format_properties(
                    shared.physical,
                    info.format,
                    info.image_type,
                    info.samples,
                    info.usage,
                    info.tiling,
                )
        };
        let listed = formats
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |all, format| {
                all | format.aspect_mask
            });
        if !listed.contains(aspects) {
            return Err(SparseNeed::SparseImageFormat);
        }
        Ok(())
    }
}

/// The sparse layout of `raw`, an image of `levels` levels with `aspects`, made with the sparse
/// flags. Refused where its memory needs a metadata aspect bound, which the runtime does not
/// bind (`SparseNeed::SparseMetadata`), or the requirements name no aspect of the image's.
///
/// # Safety
///
/// `raw` is a live image of `device`, created with `SPARSE_BINDING | SPARSE_RESIDENCY`.
pub(crate) unsafe fn sparse_layout(
    device: &ash::Device,
    raw: vk::Image,
    levels: u32,
    aspects: vk::ImageAspectFlags,
) -> Result<VkSparseLayout, Error> {
    // SAFETY: the caller's.
    let (requirements, memory) = unsafe {
        (
            device.get_image_sparse_memory_requirements(raw),
            device.get_image_memory_requirements(raw),
        )
    };
    if requirements.iter().any(|requirement| {
        requirement
            .format_properties
            .aspect_mask
            .contains(vk::ImageAspectFlags::METADATA)
    }) {
        return Err(sparse_refusal(SparseNeed::SparseMetadata));
    }
    let requirement = requirements
        .iter()
        .find(|requirement| {
            requirement
                .format_properties
                .aspect_mask
                .intersects(aspects)
        })
        .ok_or(sparse_refusal(SparseNeed::SparseImageFormat))?;
    let properties = requirement.format_properties;
    let granularity = properties.image_granularity;
    let has_tail =
        requirement.image_mip_tail_size != 0 && requirement.image_mip_tail_first_lod < levels;
    Ok(VkSparseLayout {
        aspect: properties.aspect_mask & aspects,
        granularity: [granularity.width, granularity.height, granularity.depth],
        block_bytes: memory.alignment.max(1),
        memory_type_bits: memory.memory_type_bits,
        mip_tail_first_lod: if has_tail {
            requirement.image_mip_tail_first_lod
        } else {
            levels
        },
        mip_tail_size: requirement.image_mip_tail_size,
        mip_tail_offset: requirement.image_mip_tail_offset,
        mip_tail_stride: requirement.image_mip_tail_stride,
        single_mip_tail: properties
            .flags
            .contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL),
    })
}

/// What a sparse texture's image holds beside its texels: its heap's zeros, how the runtime
/// zeroes its texels, and its tile map where its type has one.
pub(crate) struct SparseExtras {
    pub(crate) zeros: crate::buffer::Buffer,
    pub(crate) zeroing: Zeroing,
    pub(crate) tile_map: Option<TileMapBuffer>,
}

impl SparseImage {
    /// The sparse state of a bound texture of `shape` made from `heap`, whose image `raw` was
    /// just made with the sparse flags, with `extras`; the image is released when this fails.
    ///
    /// # Safety
    ///
    /// `raw` is a live image of `heap`'s device, created with `SPARSE_BINDING |
    /// SPARSE_RESIDENCY`, that nothing uses yet.
    pub(crate) unsafe fn new(
        heap: &Heap,
        raw: vk::Image,
        (shape, layers): (SparseShape, u32),
        aspects: vk::ImageAspectFlags,
        extras: SparseExtras,
    ) -> Result<SparseImage, Error> {
        let device = heap.generation().shared().vk();
        let levels = u32::try_from(shape.levels).unwrap_or(u32::MAX);
        // SAFETY: the caller's.
        let made = unsafe { sparse_layout(device, raw, levels, aspects) }.and_then(|layout| {
            SparseImage::with(heap, (shape, layers), layout, Backing::Bound, extras)
        });
        if made.is_err() {
            // SAFETY: the image is unused and unbound.
            unsafe { release::object(device, raw) };
        }
        made
    }

    /// The sparse state of an emulated texture of `shape` made from `heap`: its device units
    /// Metal's own (a tile, a slice's tail), so every update is exact, and its texels held by an
    /// ordinary image in `memory`, whose aspects are `aspects`.
    pub(crate) fn over_image(
        heap: &Heap,
        (shape, layers): (SparseShape, u32),
        aspects: vk::ImageAspectFlags,
        memory: crate::memory::Placement,
        extras: SparseExtras,
    ) -> Result<SparseImage, Error> {
        let Some(sparse) = heap.sparse() else {
            return Err(internal("a sparse image of a heap that is not sparse"));
        };
        let tile = shape.tile(sparse.page);
        let first = first_mipmap_in_tail(shape.extent, shape.levels, tile);
        let n = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
        // One device unit per Metal unit: a tile's block, and a slice's tail from Metal's first
        // tail level.
        let layout = VkSparseLayout {
            aspect: aspects,
            granularity: tile.map(n),
            block_bytes: sparse.page.bytes(),
            memory_type_bits: 0,
            mip_tail_first_lod: n(first.min(shape.levels)),
            mip_tail_size: 0,
            mip_tail_offset: 0,
            mip_tail_stride: 0,
            single_mip_tail: false,
        };
        let backing = Backing::Emulated { memory };
        SparseImage::with(heap, (shape, layers), layout, backing, extras)
    }

    /// The sparse state of a texture of `shape` and `layers` from `heap`, laid out in `layout`
    /// and held by `backing`, with `extras`, with nothing mapped.
    fn with(
        heap: &Heap,
        (shape, layers): (SparseShape, u32),
        layout: VkSparseLayout,
        backing: Backing,
        extras: SparseExtras,
    ) -> Result<SparseImage, Error> {
        let Some(sparse) = heap.sparse() else {
            return Err(internal("a sparse image of a heap that is not sparse"));
        };
        let page = sparse.page;
        let tile = shape.tile(page);
        let first = first_mipmap_in_tail(shape.extent, shape.levels, tile);
        let tail = tail_size_in_bytes(
            shape.texture_type,
            shape.pixel_format,
            shape.samples,
            shape.extent,
            shape.levels,
            first,
            page.bytes(),
        );
        let SparseExtras {
            zeros,
            zeroing,
            tile_map,
        } = extras;
        Ok(SparseImage {
            heap: heap.downgrade(),
            device: heap.generation().clone(),
            page,
            tail_size_in_bytes: tail,
            grid: UnitGrid {
                shape,
                layers,
                tile,
                first_mipmap_in_tail: first,
                layout,
            },
            state: Mutex::new(SparseState::default()),
            backing,
            zeros,
            zeroing,
            tile_map,
        })
    }

    /// The tiles of its heap that `unit` takes: one for a tile, the tail's pages for a tail.
    pub(crate) fn heap_tiles(&self, unit: MetalUnit) -> u64 {
        match unit {
            MetalUnit::Tile { .. } => 1,
            MetalUnit::Tail { .. } => (self.tail_size_in_bytes / self.page.bytes()).max(1),
        }
    }
}
