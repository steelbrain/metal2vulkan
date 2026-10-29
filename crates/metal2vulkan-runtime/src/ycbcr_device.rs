//! The device's own sampler Y′CbCr conversion of a two-plane YCbCr texture's planes
//! (`ycbcr.rs`): where the device converts the planes' format itself, the texture's image holds
//! the device's answer, and Metal's fixed-point conversion (`bind::ycbcr_convert_module`) runs
//! only where it does not.
//!
//! - **Which textures.** A texture whose planes form one image of a multi-planar format: 8-bit
//!   or 10-bit codes unpacked (a byte, or the high 10 bits of a 16-bit word), the luma plane
//!   and a chroma plane of Cb and Cr pairs at their own rows (no chroma codes 0, no rows past the
//!   buffer; each plane at a multiple of its texel's bytes, its rows whole texels that do not
//!   overlap), 4:2:0, 4:2:2 or 4:4:4: `G8_B8R8_2PLANE_420_UNORM`,
//!   `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` and their 4:2:2 and 4:4:4 formats. The device
//!   converts the format where it enables `samplerYcbcrConversion` (a device the runtime makes
//!   enables it wherever it has it, outside the vocabulary: `device/create.rs`) and its optimal
//!   tiling reports `SAMPLED_IMAGE`, `TRANSFER_DST` and a chroma location (Metal's midpoint,
//!   `MIDPOINT_CHROMA_SAMPLES`, for a 4:2:0 or 4:2:2 format; either for a 4:4:4 one, which
//!   reconstructs nothing), a 4:4:4 format also Vulkan 1.3 or `VK_EXT_ycbcr_2plane_444_formats`,
//!   and its driver has no defect in it (`RuntimeDriverDefect::TwoPlaneChromaMisplaced`, MoltenVK,
//!   for the subsampled formats; `RuntimeDriverDefect::TenBitCodesReadAsWords`, lavapipe, RADV and
//!   NVIDIA, for the 10-bit ones). A 4:4:4 format read as codes (`RGB8_444_2P`, `RGB10_444_2P`) converts
//!   nothing there (the identity model, which ignores the range, and no chroma to reconstruct),
//!   so it reads its codes through ours, without a planes' image. Every other texture, and every
//!   texture of another device, converts through Metal's arithmetic as before
//!   (`WorkCounts::ycbcr_device_conversions` counts the device's).
//! - **The conversion** (`VkSamplerYcbcrConversion`): the surface's matrix (BT.601, BT.709 or
//!   BT.2020 model) and range (`ITU_NARROW` video, `ITU_FULL` full), or the identity model of a
//!   format read as codes; and Metal's reconstruction of a 4:2:0 or 4:2:2 format's chroma, its
//!   midpoint (`MIDPOINT_CHROMA_SAMPLES`) through a linear chroma filter
//!   (`SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER`, and a linear sampler filter or a separate
//!   one). Vulkan's implicit reconstruction weighs a midpoint format's chroma 3 and 1 along a
//!   halved axis, as Metal does; cosited, its chroma lies half a luma texel off, and nearest, it
//!   takes one pair: what the surface's codes mean to Metal moves (up to 1340/2048 of random
//!   planes, lavapipe, cosited), a capability the device lacks rather than a precision, so such
//!   a format converts through Metal's arithmetic. A 4:4:4 format reconstructs nothing, so its
//!   location and chroma filter are any the device has (linear where it has it). The sampler's
//!   own filter is nearest where the format has `SEPARATE_RECONSTRUCTION_FILTER`, else the
//!   chroma's, as Vulkan requires: "If sampler Y′CBCR
//!   conversion is enabled and the potential format features of the sampler Y′CBCR conversion do
//!   not support VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
//!   minFilter and magFilter must be equal to the sampler Y′CBCR conversion’s chromaFilter"
//!   (`VUID-VkSamplerCreateInfo-minFilter-01645`); each sample lies at a luma texel's centre, where
//!   either filter reads that texel's luma.
//! - **The image and the dispatch.** The texture's conversion copies the planes from its staging
//!   copy (`ycbcr.rs`'s refresh) into an optimal image of the format (its size rounded up to the
//!   chroma block) and samples it once per texel into the texture's image
//!   (`bind::ycbcr_sample_module`), through the refresh's indirect arguments: unchanged planes
//!   copy their bytes and sample no workgroup. A render pass's load of the texture stays Metal's
//!   arithmetic (`ycbcr.rs`): Metal's `MTLLoadActionLoad` keeps the attachment's contents, and the
//!   store, ours (`bind::ycbcr_store`), inverts Metal's read, not the device's, so the device's
//!   load would move the codes of texels the pass does not draw (lavapipe: by a code, over an
//!   8-bit 4:4:4 surface).
//! - **What it reads** is the device's answer, announced where the texture is made
//!   (`ycbcr-filter-not-exact`): Vulkan states the reconstruction's weights and the model's
//!   matrices, and leaves each step's precision at least the format's and the clamp of the result
//!   into `[0, 1]` the implementation's ("An implementation may clamp the R′G′B′ results of these
//!   conversions to the range \[0,1\]."). Measured at every texel of random planes
//!   (`tests/ycbcr_device.rs`, and the validation crate's YCbCr lanes) on lavapipe (Mesa 26.0.8,
//!   LLVM 21.1.8), which converts only the 8-bit 4:4:4 layout here (its 4:2:0 and 4:2:2
//!   conversions site the chroma cosited alone, and its 10-bit ones read words,
//!   `TenBitCodesReadAsWords`): each channel within 0.04/2048 of Vulkan's conversion past the
//!   `half` the texture's image stores (0.038 at most), and of Metal's read within 6/2048 where
//!   the result lies in `[0, 1]`; outside it, which lavapipe does not clamp, by what Metal's clamp
//!   takes off. On NVIDIA's 615.71.09 (RTX 5080 Laptop GPU), which converts the 8-bit 4:2:0,
//!   4:2:2 and 4:4:4 layouts (its 10-bit ones read the codes' unused bits,
//!   `TenBitCodesReadAsWords`): within 0.034/2048 of Vulkan's conversion, and of Metal's read within
//!   12.4/2048 where it lies in `[0, 1]` (19.1 through an `_sRGB` twin's curve), Metal rounding each
//!   reconstructed chroma to a code where the device does not; a code-reading format's codes are
//!   Metal's. MoltenVK 1.4.2 converts no two-plane format here: its 4:2:0 and 4:2:2
//!   reconstruction is misplaced (`TwoPlaneChromaMisplaced`), and it reports no 4:4:4 two-plane
//!   format.
//! - **4:2:2 pairs** (`GBGR422`, `BGRG422`, `chroma_pairs.rs`). The device converts a 4:2:2
//!   texture's pairs itself where its optimal tiling of `G8B8G8R8_422_UNORM` or
//!   `B8G8R8G8_422_UNORM` reports `SAMPLED_IMAGE`, `TRANSFER_DST` and Metal's reconstruction
//!   (`MIDPOINT_CHROMA_SAMPLES` through a linear chroma filter, as above): the texture's decoded
//!   twin then holds the device's conversion of its pairs (the identity model, full range),
//!   sampled once per texel by each decode pass ([`PairsSampling`], `bind::chroma_sample_module`),
//!   where elsewhere it holds Metal's (`bind::chroma_decode_module`). MoltenVK 1.4.2 converts both
//!   (its single-plane 4:2:2 reconstruction is Vulkan's, `TwoPlaneChromaMisplaced`); lavapipe
//!   filters their chroma nearest alone (no `SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER`) and
//!   RADV has neither format, so their twins hold Metal's decode.
//! - **Planes images** (`texture/ycbcr_planes.rs`). A texture whose planes a function reads in
//!   place is also a linear image of its 4:2:0 format over their bytes where the device's linear
//!   tiling of `G8_B8R8_2PLANE_420_UNORM` or `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` reports
//!   `SAMPLED_IMAGE` and Metal's reconstruction (as above) and its driver has no defect in the
//!   format ([`YCbCrSampler::planes`]): the conversion is the surface's matrix and range, and the
//!   view and every immutable sampler of it in a pipeline's layout share one conversion object
//!   ([`Device::made_ycbcr_conversion`]). A pool sets aside `combinedImageSamplerDescriptorCount`
//!   descriptors for each such binding ([`YCbCrFormats::planes_descriptors`]).

use super::device::lock;
use super::error::{vulkan, Error};
use super::memory::{MemoryClass, Placement};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use super::Device;
use crate::device::functions::Vk;
use crate::{creation, release};
use ash::vk;
use metal2vulkan::bind::{
    ChromaDecode, ChromaSampleModule, YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrPlanesSampler,
    YCbCrRange, YCbCrSampleModule,
};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::format::ChromaSubsampling;

/// The formats the device is asked to convert, each of its codes' bits and subsampling.
const FORMATS: [(vk::Format, u32, ChromaSubsampling); 6] = [
    (
        vk::Format::G8_B8R8_2PLANE_420_UNORM,
        8,
        ChromaSubsampling::Chroma420,
    ),
    (
        vk::Format::G8_B8R8_2PLANE_422_UNORM,
        8,
        ChromaSubsampling::Chroma422,
    ),
    (
        vk::Format::G8_B8R8_2PLANE_444_UNORM,
        8,
        ChromaSubsampling::Chroma444,
    ),
    (
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        10,
        ChromaSubsampling::Chroma420,
    ),
    (
        vk::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
        10,
        ChromaSubsampling::Chroma422,
    ),
    (
        vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
        10,
        ChromaSubsampling::Chroma444,
    ),
];

/// The 4:2:2 formats of a single plane the device is asked to convert (the module docs).
const PAIR_FORMATS: [vk::Format; 2] = [
    vk::Format::G8B8G8R8_422_UNORM,
    vk::Format::B8G8R8G8_422_UNORM,
];

/// One format the device converts: its optimal tiling's features, the largest image of it the
/// device makes, and the descriptors a combined image sampler of it takes.
#[derive(Clone, Copy)]
struct Converts {
    format: vk::Format,
    features: vk::FormatFeatureFlags,
    extent: [u32; 2],
    descriptors: u32,
}

/// The formats of [`FORMATS`] a device converts (the module docs): none where it does not enable
/// `samplerYcbcrConversion`.
#[derive(Clone, Default)]
pub(super) struct YCbCrFormats {
    converts: Vec<Converts>,
    /// The 4:2:0 formats it converts as a linear image that is sampled, with Metal's
    /// reconstruction: the planes images a function samples in place
    /// (`texture/ycbcr_planes.rs`).
    linear: Vec<Converts>,
}

/// The 4:2:0 formats a planes image is of (`texture/ycbcr_planes.rs`).
const PLANES_FORMATS: [vk::Format; 2] = [
    vk::Format::G8_B8R8_2PLANE_420_UNORM,
    vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
];

impl YCbCrFormats {
    /// The formats `physical` converts, as an image that is sampled and copied into; the
    /// two-plane 4:4:4 ones only where its device takes them (`four_four_four`: Vulkan 1.3, or
    /// `VK_EXT_ycbcr_2plane_444_formats` enabled).
    ///
    /// # Safety
    ///
    /// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and its device
    /// enables `samplerYcbcrConversion`.
    pub(super) unsafe fn of(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        four_four_four: bool,
    ) -> YCbCrFormats {
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let pairs = PAIR_FORMATS
            .iter()
            .map(|&format| (format, 8, ChromaSubsampling::Chroma422));
        let converts = FORMATS
            .iter()
            .copied()
            .filter(|&(_, _, subsampling)| {
                four_four_four || subsampling != ChromaSubsampling::Chroma444
            })
            .chain(pairs)
            .filter_map(|(format, _, subsampling)| {
                // SAFETY: the caller's.
                let properties =
                    unsafe { instance.get_physical_device_format_properties(physical, format) };
                let features = properties.optimal_tiling_features;
                let needed =
                    vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
                // Metal's reconstruction of a subsampled format: its midpoint, through a linear
                // chroma filter (the module docs); a 4:4:4 one reconstructs nothing.
                let reconstructs = match subsampling {
                    ChromaSubsampling::Chroma444 => features.intersects(
                        vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES
                            | vk::FormatFeatureFlags::COSITED_CHROMA_SAMPLES,
                    ),
                    _ => {
                        features.contains(vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES)
                            && linear_chroma(features)
                    }
                };
                if !features.contains(needed) || !reconstructs {
                    return None;
                }
                let info = vk::PhysicalDeviceImageFormatInfo2::default()
                    .format(format)
                    .ty(vk::ImageType::TYPE_2D)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage);
                let mut ycbcr = vk::SamplerYcbcrConversionImageFormatProperties::default();
                let mut out = vk::ImageFormatProperties2::default().push_next(&mut ycbcr);
                // SAFETY: the caller's; the chain lives for the call.
                unsafe {
                    instance.get_physical_device_image_format_properties2(physical, &info, &mut out)
                }
                .ok()?;
                let extent = out.image_format_properties.max_extent;
                Some(Converts {
                    format,
                    features,
                    extent: [extent.width, extent.height],
                    descriptors: ycbcr.combined_image_sampler_descriptor_count.max(1),
                })
            })
            .collect();
        // As a linear image that is sampled: its linear tiling's features and limits.
        let linear = PLANES_FORMATS
            .iter()
            .filter_map(|&format| {
                // SAFETY: the caller's.
                let properties =
                    unsafe { instance.get_physical_device_format_properties(physical, format) };
                let features = properties.linear_tiling_features;
                if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
                    || !features.contains(vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES)
                    || !linear_chroma(features)
                {
                    return None;
                }
                let info = vk::PhysicalDeviceImageFormatInfo2::default()
                    .format(format)
                    .ty(vk::ImageType::TYPE_2D)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::SAMPLED);
                let mut ycbcr = vk::SamplerYcbcrConversionImageFormatProperties::default();
                let mut out = vk::ImageFormatProperties2::default().push_next(&mut ycbcr);
                // SAFETY: the caller's; the chain lives for the call.
                unsafe {
                    instance.get_physical_device_image_format_properties2(physical, &info, &mut out)
                }
                .ok()?;
                let extent = out.image_format_properties.max_extent;
                Some(Converts {
                    format,
                    features,
                    extent: [extent.width, extent.height],
                    descriptors: ycbcr.combined_image_sampler_descriptor_count.max(1),
                })
            })
            .collect();
        YCbCrFormats { converts, linear }
    }

    /// A conversion of the first format the device converts, which its priming makes and
    /// releases (`device/prime.rs`); `None` where it converts none.
    pub(super) fn primer(&self) -> Option<vk::SamplerYcbcrConversionCreateInfo<'static>> {
        let first = self.converts.first()?;
        Some(conversion_info(&YCbCrSampler {
            format: first.format,
            model: vk::SamplerYcbcrModelConversion::RGB_IDENTITY,
            range: vk::SamplerYcbcrRange::ITU_FULL,
            location: location(first.features),
            chroma_filter: vk::Filter::NEAREST,
            filter: vk::Filter::NEAREST,
            descriptors: first.descriptors,
        }))
    }

    /// The format's entry, where the device converts it.
    fn of_format(&self, format: vk::Format) -> Option<Converts> {
        self.converts.iter().copied().find(|c| c.format == format)
    }

    /// `format` as a linear image the device samples, where it converts it so.
    fn linear_of(&self, format: vk::Format) -> Option<Converts> {
        self.linear.iter().copied().find(|c| c.format == format)
    }

    /// The descriptors a combined image sampler of a planes image of `sampler` takes from a pool
    /// (`combinedImageSamplerDescriptorCount`): Vulkan's "each combined image sampler descriptor
    /// binding with a layout that includes at least one immutable Y′CBCR conversion sampler will
    /// internally use a number of descriptors equal to the maximum
    /// combinedImageSamplerDescriptorCount of the formats", 1 where the device converts no such
    /// image.
    pub(super) fn planes_descriptors(&self, sampler: YCbCrPlanesSampler) -> u32 {
        self.linear_of(sampler.format().into())
            .map_or(1, |c| c.descriptors)
    }
}

/// Whether a format of `features` filters its chroma linearly: the conversion's linear filter,
/// and either a separate reconstruction filter or a linear sampler filter, which then is the
/// chroma's (`VUID-VkSamplerCreateInfo-minFilter-01645`).
fn linear_chroma(features: vk::FormatFeatureFlags) -> bool {
    let separate = features.contains(
        vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER,
    );
    features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER)
        && (separate || features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
}

/// The chroma location of a format of `features`: Metal's midpoint where it has it (every
/// subsampled format [`YCbCrFormats`] holds has it).
fn location(features: vk::FormatFeatureFlags) -> vk::ChromaLocation {
    match features.contains(vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES) {
        true => vk::ChromaLocation::MIDPOINT,
        false => vk::ChromaLocation::COSITED_EVEN,
    }
}

/// The device's conversion of one texture's planes (the module docs): the sampler a sample
/// kernel's layout bakes (`owned_kernel.rs`), its conversion's, and the planes' image's format.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct YCbCrSampler {
    pub(super) format: vk::Format,
    model: vk::SamplerYcbcrModelConversion,
    range: vk::SamplerYcbcrRange,
    location: vk::ChromaLocation,
    chroma_filter: vk::Filter,
    filter: vk::Filter,
    /// The descriptors a combined image sampler of the format takes
    /// (`combinedImageSamplerDescriptorCount`).
    pub(super) descriptors: u32,
}

impl std::fmt::Debug for YCbCrSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ash` names no Vulkan value here: the raw numbers.
        f.debug_struct("YCbCrSampler")
            .field("format", &self.format.as_raw())
            .field("model", &self.model.as_raw())
            .field("range", &self.range.as_raw())
            .field("location", &self.location.as_raw())
            .field("chroma_filter", &self.chroma_filter.as_raw())
            .field("filter", &self.filter.as_raw())
            .field("descriptors", &self.descriptors)
            .finish()
    }
}

/// The conversion's creation info of `sampler`.
fn conversion_info(sampler: &YCbCrSampler) -> vk::SamplerYcbcrConversionCreateInfo<'static> {
    vk::SamplerYcbcrConversionCreateInfo::default()
        .format(sampler.format)
        .ycbcr_model(sampler.model)
        .ycbcr_range(sampler.range)
        .components(vk::ComponentMapping::default())
        .x_chroma_offset(sampler.location)
        .y_chroma_offset(sampler.location)
        .chroma_filter(sampler.chroma_filter)
        .force_explicit_reconstruction(false)
}

impl YCbCrSampler {
    /// The device's conversion of `planes`, where it converts them (the module docs); `None`
    /// where Metal's arithmetic does. `zeros` says whether the texture reads zeros past its
    /// planes' bytes (chroma codes 0, or rows past its buffer), which no image of the planes
    /// holds.
    pub(super) fn of(device: &Device, planes: &YCbCrPlanes, zeros: bool) -> Option<YCbCrSampler> {
        let codes = planes.codes;
        if zeros || codes.packed {
            return None;
        }
        // The planes form one image: each plane's rows whole texels that do not overlap, at a
        // multiple of its texel's bytes ("bufferRowLength must be 0, or greater than or equal
        // to the width member of imageExtent", `VUID-VkBufferImageCopy-bufferRowLength-09101`;
        // a plane's `bufferOffset` a multiple of its element's size,
        // `VUID-vkCmdCopyBufferToImage-dstImage-07976`).
        let [chroma_width, _] = planes.chroma_size();
        let rows = [
            (planes.luma, codes.bytes, planes.size[0]),
            (planes.chroma, 2 * codes.bytes, chroma_width),
        ];
        if !rows.iter().all(|&([at, row], texel, width)| {
            at % texel == 0 && row % texel == 0 && row / texel >= width
        }) {
            return None;
        }
        let (format, _, subsampling) =
            FORMATS.iter().copied().find(|&(_, bits, subsampling)| {
                bits == codes.bits && subsampling == codes.subsampling
            })?;
        let defects = &device.properties().driver_defects;
        let subsampled = subsampling != ChromaSubsampling::Chroma444;
        if (subsampled && defects.contains(&RuntimeDriverDefect::TwoPlaneChromaMisplaced))
            || (codes.bits == 10 && defects.contains(&RuntimeDriverDefect::TenBitCodesReadAsWords))
        {
            return None;
        }
        // A 4:4:4 format's codes convert through nothing: the identity model, no range
        // (`RGB_IDENTITY` ignores it) and no chroma to reconstruct, so the device's answer is the
        // codes ours reads, without a planes' image.
        if codes.output == YCbCrOutput::Codes && !subsampled {
            return None;
        }
        let converts = device.shared().ycbcr_formats.of_format(format)?;
        let extent = extent(planes.size, subsampling);
        if extent[0] > converts.extent[0] || extent[1] > converts.extent[1] {
            return None;
        }
        let features = converts.features;
        let separate = features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER,
        );
        let chroma_filter = match linear_chroma(features) {
            true => vk::Filter::LINEAR,
            false => vk::Filter::NEAREST,
        };
        // An output, matrix or range not named here converts through Metal's arithmetic.
        let (model, range) = match codes.output {
            YCbCrOutput::Codes => (
                vk::SamplerYcbcrModelConversion::RGB_IDENTITY,
                vk::SamplerYcbcrRange::ITU_FULL,
            ),
            YCbCrOutput::Converted | YCbCrOutput::Srgb => (
                match planes.matrix {
                    YCbCrMatrix::Bt601 => vk::SamplerYcbcrModelConversion::YCBCR_601,
                    YCbCrMatrix::Bt709 => vk::SamplerYcbcrModelConversion::YCBCR_709,
                    YCbCrMatrix::Bt2020 => vk::SamplerYcbcrModelConversion::YCBCR_2020,
                    _ => return None,
                },
                match planes.range {
                    YCbCrRange::Video => vk::SamplerYcbcrRange::ITU_NARROW,
                    YCbCrRange::Full => vk::SamplerYcbcrRange::ITU_FULL,
                    _ => return None,
                },
            ),
            _ => return None,
        };
        Some(YCbCrSampler {
            format,
            model,
            range,
            location: location(features),
            chroma_filter,
            filter: match separate {
                true => vk::Filter::NEAREST,
                false => chroma_filter,
            },
            descriptors: converts.descriptors,
        })
    }

    /// The device's conversion of a planes image of `sampler` (`texture/ycbcr_planes.rs`): its
    /// 4:2:0 format as a linear image the device samples with Metal's reconstruction (the module
    /// docs), the surface's matrix and range, and a linear filter, which every immutable sampler of
    /// it in a pipeline's layout bakes; `None` where the device does not convert the format so (a
    /// defect in it among the reasons).
    pub(super) fn planes(device: &Device, sampler: YCbCrPlanesSampler) -> Option<YCbCrSampler> {
        let format: vk::Format = sampler.format().into();
        let converts = device.shared().ycbcr_formats.linear_of(format)?;
        let defects = &device.properties().driver_defects;
        if defects.contains(&RuntimeDriverDefect::TwoPlaneChromaMisplaced)
            || (sampler.bits() == 10
                && defects.contains(&RuntimeDriverDefect::TenBitCodesReadAsWords))
        {
            return None;
        }
        let model = match sampler.matrix() {
            YCbCrMatrix::Bt601 => vk::SamplerYcbcrModelConversion::YCBCR_601,
            YCbCrMatrix::Bt709 => vk::SamplerYcbcrModelConversion::YCBCR_709,
            YCbCrMatrix::Bt2020 => vk::SamplerYcbcrModelConversion::YCBCR_2020,
            _ => return None,
        };
        let range = match sampler.range() {
            YCbCrRange::Video => vk::SamplerYcbcrRange::ITU_NARROW,
            YCbCrRange::Full => vk::SamplerYcbcrRange::ITU_FULL,
            _ => return None,
        };
        Some(YCbCrSampler {
            format,
            model,
            range,
            location: vk::ChromaLocation::MIDPOINT,
            chroma_filter: vk::Filter::LINEAR,
            // The chroma's filter, as Vulkan requires without a separate reconstruction filter
            // (`VUID-VkSamplerCreateInfo-minFilter-01645`), and the filter a linear sample takes.
            filter: vk::Filter::LINEAR,
            descriptors: converts.descriptors,
        })
    }

    /// Whether a planes image of [`planes`](Self::planes)'s format may be `[width, height]`: a
    /// 4:2:0 image's width and height are even ("If format has a _420 suffix, extent.width must
    /// be a multiple of 2", `VUID-VkImageCreateInfo-format-04712`, and its height,
    /// `-04713`), within the format's linear limits.
    pub(super) fn holds_planes(&self, device: &Device, [width, height]: [u32; 2]) -> bool {
        let Some(converts) = device.shared().ycbcr_formats.linear_of(self.format) else {
            return false;
        };
        width > 0
            && height > 0
            && width.is_multiple_of(2)
            && height.is_multiple_of(2)
            && width <= converts.extent[0]
            && height <= converts.extent[1]
    }

    /// The device's conversion of a 4:2:2 texture's pairs of `decode` over an image of `extent`,
    /// where it converts them with Metal's reconstruction (the module docs): its midpoint through
    /// a linear chroma filter, the identity model at full range (its channels are the codes);
    /// `None` where Metal's decode holds them.
    pub(super) fn pairs(
        device: &Device,
        decode: ChromaDecode,
        extent: [u32; 2],
    ) -> Option<YCbCrSampler> {
        let format: vk::Format = decode.format().into();
        let converts = device.shared().ycbcr_formats.of_format(format)?;
        // A 4:2:2 image's width is even (`VUID-VkImageCreateInfo-format-04712`).
        if !extent[0].is_multiple_of(2)
            || extent[0] > converts.extent[0]
            || extent[1] > converts.extent[1]
        {
            return None;
        }
        // Its midpoint and linear chroma filter: `YCbCrFormats::of` holds no subsampled format
        // without them.
        let separate = converts.features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER,
        );
        Some(YCbCrSampler {
            format,
            model: vk::SamplerYcbcrModelConversion::RGB_IDENTITY,
            range: vk::SamplerYcbcrRange::ITU_FULL,
            location: vk::ChromaLocation::MIDPOINT,
            chroma_filter: vk::Filter::LINEAR,
            filter: match separate {
                true => vk::Filter::NEAREST,
                false => vk::Filter::LINEAR,
            },
            descriptors: converts.descriptors,
        })
    }

    /// Whether its chroma lies at Metal's midpoint (else cosited), and whether its chroma filter
    /// is linear (else nearest).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn chroma(&self) -> (bool, bool) {
        (
            self.location == vk::ChromaLocation::MIDPOINT,
            self.chroma_filter == vk::Filter::LINEAR,
        )
    }

    /// The conversion and the sampler of it, made by `device`; the sampler counted against the
    /// device's samplers (`DeviceShared::create_sampler`).
    pub(super) fn make(
        &self,
        device: &Device,
    ) -> Result<(vk::SamplerYcbcrConversion, vk::Sampler), Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let info = conversion_info(self);
        // SAFETY: the device enables `samplerYcbcrConversion` and converts the format with the
        // location and filter chosen from its features (`of`).
        let conversion =
            unsafe { raw.create_sampler_ycbcr_conversion(&info, None) }.map_err(vulkan)?;
        let mut chained = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(self.filter)
            .min_filter(self.filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0)
            .push_next(&mut chained);
        // SAFETY: `info` and its chain live for the call; the conversion is live.
        match unsafe { shared.create_sampler(&info) } {
            Ok(sampler) => Ok((conversion, sampler)),
            Err(error) => {
                // SAFETY: the conversion is unused.
                unsafe { release::object(raw, conversion) };
                Err(error)
            }
        }
    }
}

/// The planes' image of a texture of `size` luma texels at `subsampling`: its size rounded up
/// to the chroma block, which Vulkan's 4:2:0 and 4:2:2 formats require.
fn extent(size: [u32; 2], subsampling: ChromaSubsampling) -> [u32; 2] {
    let halved = subsampling.halved();
    [0, 1].map(|axis| match halved[axis] {
        true => size[axis].next_multiple_of(2),
        false => size[axis],
    })
}

/// The planes' image of the device's conversion of one texture, and the dispatch that samples
/// it into the texture's image (the module docs). Its objects
/// are released with it; no pending command uses them then (the texture's command buffers hold
/// the texture until they complete).
pub(super) struct YCbCrDeviceSampling {
    device: Device,
    image: vk::Image,
    placement: Placement,
    view: vk::ImageView,
    kernel: OwnedKernel,
    sets: OwnedSets,
    push: [u8; YCbCrSampleModule::PUSH_BYTES as usize],
    /// The planes' bytes from the staging buffer's copy into each plane, from the copy's first
    /// byte.
    regions: [vk::BufferImageCopy; 2],
    groups: [u32; 2],
}

impl YCbCrDeviceSampling {
    /// The image of `planes` through `sampler`, and the sampling of it into `target`, a storage
    /// view of the texture's image, on `device`. Refused as the image, its memory, its view, the
    /// kernel's objects and its set are refused.
    pub(super) fn new(
        device: &Device,
        (planes, sampler): (&YCbCrPlanes, YCbCrSampler),
        target: vk::ImageView,
    ) -> Result<YCbCrDeviceSampling, Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let [width, height] = extent(planes.size, planes.codes.subsampling);
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(sampler.format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        );
        // SAFETY: the device converts the format as an optimal image that is sampled and
        // copied into, of this extent (`YCbCrSampler::of`).
        let image = unsafe { creation::image(raw, &info) }?;
        // SAFETY: `image` is a live image of this device.
        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        let requirements = shared.textures.placement().place(requirements);
        let placement = match crate::memory::allocate_image(
            &shared.memory,
            raw,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw, image) };
                return Err(error);
            }
        };
        let unwind = |view: Option<vk::ImageView>| {
            if let Some(view) = view {
                // SAFETY: the view is unused.
                unsafe { release::object(raw, view) };
            }
            lock(&shared.memory).free(raw, &placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw, image) };
        };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) =
            unsafe { creation::bind_image(raw, image, placement.memory, placement.offset) }
        {
            unwind(None);
            return Err(vulkan(error));
        }
        let kernel = match device.owned_kernel(OwnedModule::YCbCrSample(sampler, planes.codes)) {
            Ok(kernel) => kernel,
            Err(error) => {
                unwind(None);
                return Err(error);
            }
        };
        let Some(conversion) = device.ycbcr_conversion(sampler) else {
            unwind(None);
            return Err(super::error::internal(
                "a YCbCr sample kernel without its conversion",
            ));
        };
        let mut chained = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(sampler.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .push_next(&mut chained);
        // SAFETY: the image is live and bound; the conversion is of its format.
        let view = match unsafe { creation::image_view(raw, &info) } {
            Ok(view) => view,
            Err(error) => {
                unwind(None);
                return Err(vulkan(error));
            }
        };
        let set = OwnedSet {
            images: vec![view],
            storage_images: vec![target],
            buffers: Vec::new(),
        };
        let sets = match OwnedSets::write_each(device, &[kernel], &[set]) {
            Ok(sets) => sets,
            Err(error) => {
                unwind(Some(view));
                return Err(error);
            }
        };
        let bytes = planes.codes.bytes;
        let [chroma_width, chroma_height] = planes.chroma_size();
        let region =
            |(at, row): (u32, u32), aspect, texel: u32, [w, h]: [u32; 2]| vk::BufferImageCopy {
                buffer_offset: u64::from(at),
                buffer_row_length: row / texel,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                },
            };
        let regions = [
            region(
                (planes.luma[0], planes.luma[1]),
                vk::ImageAspectFlags::PLANE_0,
                bytes,
                planes.size,
            ),
            region(
                (planes.chroma[0], planes.chroma[1]),
                vk::ImageAspectFlags::PLANE_1,
                2 * bytes,
                [chroma_width, chroma_height],
            ),
        ];
        device.shared().counts.ycbcr_device_conversion();
        Ok(YCbCrDeviceSampling {
            device: device.clone(),
            image,
            placement,
            view,
            kernel,
            sets,
            push: YCbCrSampleModule::push(planes.size, [width, height]),
            regions,
            groups: planes
                .size
                .map(|n| n.div_ceil(YCbCrSampleModule::LOCAL_SIZE)),
        })
    }

    /// The workgroups of its dispatch, which the conversion's refresh writes into its indirect
    /// arguments where the planes changed (`ycbcr.rs`).
    pub(super) fn groups(&self) -> [u32; 2] {
        self.groups
    }

    /// Record the copy of the planes from `bytes_at` of `staging` into the image, and the
    /// sampling of it into the target: dispatched through the indirect arguments at `indirect`
    /// of `staging` where given, else over every texel.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `staging` holds the planes' bytes
    /// from `bytes_at`, as the regions read them, written before (a barrier recorded), and it,
    /// the image and the target are live until the command buffer completes; the target is in
    /// the `GENERAL` layout.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (staging, bytes_at): (vk::Buffer, u64),
        indirect: Option<u64>,
    ) {
        let regions = self.regions.map(|region| vk::BufferImageCopy {
            buffer_offset: bytes_at + region.buffer_offset,
            ..region
        });
        let whole = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        // Every earlier use of the image (an earlier conversion's sample) before its texels are
        // dropped and copied in whole.
        let into = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(whole);
        let sampled = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(whole);
        // SAFETY: the caller's contract; the regions lie in the staging copy's planes, and the
        // set binds the image's view and the target.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[into],
            );
            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging,
                self.image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[sampled],
            );
            let point = vk::PipelineBindPoint::COMPUTE;
            device.cmd_bind_pipeline(command_buffer, point, self.kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                point,
                self.kernel.layout,
                0,
                &self.sets.sets,
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.push,
            );
            match indirect {
                Some(at) => device.cmd_dispatch_indirect(command_buffer, staging, at),
                None => device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1),
            }
        }
        self.device.shared().counts.ycbcr_device_sampled();
    }
}

impl Drop for YCbCrDeviceSampling {
    fn drop(&mut self) {
        let shared = self.device.shared();
        let raw = shared.vk();
        // SAFETY: no pending command uses them (the type's docs).
        unsafe {
            release::object(raw, self.view);
            release::object(raw, self.image);
        }
        lock(&shared.memory).free(raw, &self.placement);
    }
}

/// The pairs' image of a 4:2:2 texture whose decoded twin holds the device's own conversion of
/// its pairs (the module docs, `texture/decoded_twin.rs`): an optimal image of the pairs' format
/// of the texture's extent, which each decode pass's step copies the band's pairs into and
/// samples through the device's conversion into the band's texels (`bind::chroma_sample_module`).
/// Its objects are released with it; no pending command uses them then (the texture's command
/// buffers hold the texture, and its twin with it, until they complete).
pub(crate) struct PairsSampling {
    device: Device,
    pub(crate) image: vk::Image,
    placement: Placement,
    pub(crate) view: vk::ImageView,
    pub(crate) kernel: OwnedKernel,
    pub(crate) extent: [u32; 2],
}

impl PairsSampling {
    /// The pairs' image of a texture of `extent` (an even width) through `sampler`
    /// ([`YCbCrSampler::pairs`]), on `device`. Refused as the image, its memory, the kernel and
    /// the view are refused.
    pub(crate) fn new(
        device: &Device,
        sampler: YCbCrSampler,
        extent: [u32; 2],
    ) -> Result<PairsSampling, Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let [width, height] = extent;
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(sampler.format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        );
        // SAFETY: the device converts the format as an optimal image that is sampled and
        // copied into, of this extent (`YCbCrSampler::pairs`).
        let image = unsafe { creation::image(raw, &info) }?;
        // SAFETY: `image` is a live image of this device.
        let requirements = unsafe { raw.get_image_memory_requirements(image) };
        let requirements = shared.textures.placement().place(requirements);
        let placement = match crate::memory::allocate_image(
            &shared.memory,
            raw,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw, image) };
                return Err(error);
            }
        };
        let unwind = || {
            lock(&shared.memory).free(raw, &placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw, image) };
        };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) =
            unsafe { creation::bind_image(raw, image, placement.memory, placement.offset) }
        {
            unwind();
            return Err(vulkan(error));
        }
        let kernel = match device.owned_kernel(OwnedModule::ChromaSample(sampler)) {
            Ok(kernel) => kernel,
            Err(error) => {
                unwind();
                return Err(error);
            }
        };
        let Some(conversion) = device.ycbcr_conversion(sampler) else {
            unwind();
            return Err(super::error::internal(
                "a 4:2:2 sample kernel without its conversion",
            ));
        };
        let mut chained = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(sampler.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .push_next(&mut chained);
        // SAFETY: the image is live and bound; the conversion is of its format.
        let view = match unsafe { creation::image_view(raw, &info) } {
            Ok(view) => view,
            Err(error) => {
                unwind();
                return Err(vulkan(error));
            }
        };
        shared.counts.chroma_device_twin();
        Ok(PairsSampling {
            device: device.clone(),
            image,
            placement,
            view,
            kernel,
            extent,
        })
    }

    /// The shape of a step that samples `rows` rows from row `first` of the image
    /// (`ChromaSampleModule::shape`), and its workgroups.
    pub(crate) fn step(
        &self,
        first: u32,
        rows: u32,
    ) -> ([u8; ChromaSampleModule::SHAPE_WORDS as usize * 4], [u32; 3]) {
        let [width, _] = self.extent;
        let local = ChromaSampleModule::LOCAL_SIZE;
        (
            ChromaSampleModule::shape([width, rows], first, self.extent),
            [width.div_ceil(local), rows.div_ceil(local), 1],
        )
    }
}

impl Drop for PairsSampling {
    fn drop(&mut self) {
        let shared = self.device.shared();
        let raw = shared.vk();
        // SAFETY: no pending command uses them (the type's docs).
        unsafe {
            release::object(raw, self.view);
            release::object(raw, self.image);
        }
        lock(&shared.memory).free(raw, &self.placement);
    }
}
