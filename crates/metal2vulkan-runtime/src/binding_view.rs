//! The Vulkan image views of a texture: what a binding write needs of a texture presenting a
//! [`ViewRecord`], from the binding plan's view.
//!
//! The view composes the record's Metal swizzle on the component mapping that presents the view
//! format's channels (`TextureSwizzleChannels::compose`): Metal applies a swizzle to the channels
//! the view's format gives, border texels included (measured on an Apple M3 Max and M4). A texture
//! whose alpha reads 1 (`TextureDescriptor::alpha_reads_one`) composes the swizzle that reads it so
//! ([`ViewRecord::sampled_swizzle`]), and is, to the binding plan, a texture of that swizzle: a
//! storage slot's function writes it unswizzled, which keeps the alpha it writes. Its usage is
//! restricted to what it serves (`VkImageViewUsageCreateInfo`), so a view format lacking the
//! image's other usages is valid.

use super::image_plan::{block_family, compatible, ImagePlan};
use super::view_plan::{ViewProblem, ViewRecord, ViewRefusal};
use ash::vk;
use metal2vulkan::bind::{ImageViewDesc, TextureFacts};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageCreateFlags, ImageUsageFlags, ImageViewType,
};

/// The facts a binding plan reads of a texture presenting `record`.
pub(super) fn texture_facts(record: &ViewRecord) -> TextureFacts {
    let mut facts = TextureFacts::new(record.texture_type, record.pixel_format);
    facts.sample_count = record.samples;
    facts.usage = record.usage;
    facts.array_length = record.array_length();
    facts.mipmap_level_count = record.level_count;
    // A texture whose alpha reads 1 is read through the swizzle that reads it so.
    facts.swizzle = record.sampled_swizzle();
    // A slot a constexpr sampler samples compiles under the map (`VariantKey::address_maps`).
    facts.address_modes = record.address_mode_map;
    // Metal's texel-offset rule reads the base-level size of the texture whose levels a view
    // presents (`SpecKey::OffsetBaseAxes`): the image's.
    facts.parent_relative_level = record.base_level;
    facts.parent_base_size = record.image_extent;
    facts
}

/// Whether a view of `image` may have format `view`: the image's own format, a listed one, or,
/// for an image that takes any compatible format, one of its compatibility class (`family`: the
/// two Metal formats are of one compressed family, which block-compressed formats need).
pub(super) fn views_as(image: &ImagePlan, view: Format, family: bool) -> bool {
    if view == image.format {
        return true;
    }
    if !image.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) {
        return false;
    }
    if image.view_formats.is_empty() {
        compatible(image.format, view, family)
    } else {
        image.view_formats.contains(&view)
    }
}

/// A Vulkan image view of a texture: the key of the texture's view cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct VkView {
    pub(super) view_type: ImageViewType,
    pub(super) format: Format,
    pub(super) components: ComponentMapping,
    pub(super) aspect: ImageAspectFlags,
    pub(super) base_level: u32,
    pub(super) level_count: u32,
    pub(super) base_layer: u32,
    pub(super) layer_count: u32,
    /// What the view serves (`VkImageViewUsageCreateInfo`), so a view format lacking the image's
    /// other usages is valid.
    pub(super) usage: ImageUsageFlags,
}

impl VkView {
    /// The `VkImageViewCreateInfo` of this view of `image`, with its usage chained through
    /// `usage` (caller storage).
    pub(super) fn create_info<'a, 'b>(
        &self,
        image: vk::Image,
        usage: &'a mut vk::ImageViewUsageCreateInfo<'b>,
    ) -> vk::ImageViewCreateInfo<'a> {
        *usage = vk::ImageViewUsageCreateInfo::default().usage(self.usage.into());
        vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(self.view_type.into())
            .format(self.format.into())
            .components(self.components.into())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspect.into(),
                base_mip_level: self.base_level,
                level_count: self.level_count,
                base_array_layer: self.base_layer,
                layer_count: self.layer_count,
            })
            .push_next(usage)
    }
}

/// The view a binding write needs of a texture whose image is `image` and which presents
/// `record`: `desc` is the binding plan's view of the record's facts, and `storage` says the
/// write is a storage image's.
///
/// A record whose usage lacks what the binding needs (`ShaderWrite` for a storage image,
/// `ShaderRead` for a sampled one) is bound where the image has the usage, as plain Metal writes
/// and reads such a texture (an M4 under macOS 27.0.1: a write to a `ShaderRead` texture stores, a
/// read of a `ShaderWrite` one reads, and a swizzled view, whose usage Metal strips of
/// `ShaderWrite`, is written through; its validation stops each): the image has it where the
/// texture's usage asked for it, or where adding it cost the image nothing
/// (`Device::with_undeclared_usage`).
///
/// Refusals: where the image lacks the usage, the texture's usage lacking it too
/// (`NotImplemented(UndeclaredTextureUsage)`, the runtime's gap) or the device lacking the usage
/// for the format (`Unsupported{Format}`); a view format the image cannot take
/// (`NotImplemented(TextureView)`); a swizzle on a storage image, whose view Vulkan cannot swizzle
/// (`NotImplemented(TextureSwizzle)`), but where the binding plan's function applies it
/// (`ImageViewDesc::shader_swizzle`: a swizzled texture or view at a storage slot, whose reads the
/// function swizzles and whose writes it unswizzles, or one the function only queries, which no
/// swizzle reaches; and the sampled view of a swizzled extended-range texture, whose decoded texels
/// the function swizzles), which leaves a texture made resident for a function reading handles, or
/// a `ulong` atomic's slot.
pub(super) fn binding_view(
    image: &ImagePlan,
    record: &ViewRecord,
    desc: &ImageViewDesc,
    storage: bool,
) -> Result<VkView, ViewRefusal> {
    let (usage, metal, format_usage) = if storage {
        (
            ImageUsageFlags::STORAGE,
            TextureUsage::SHADER_WRITE,
            FormatUsage::StorageImage,
        )
    } else {
        (
            ImageUsageFlags::SAMPLED,
            TextureUsage::SHADER_READ,
            FormatUsage::SampledImage,
        )
    };
    if !image.usage.contains(usage) && !record.usage.contains(metal) {
        return Err(ViewRefusal::Unsupported(Unsupported::not_implemented(
            Feature::UndeclaredTextureUsage,
        )));
    }
    if !image.usage.contains(usage) {
        return Err(ViewRefusal::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: desc.format,
                usage: format_usage,
            },
        )));
    }
    let family = block_family(record.pixel_format) == block_family(image.pixel_format);
    if !views_as(image, desc.format, family) {
        return Err(ViewRefusal::Unsupported(Unsupported::not_implemented(
            Feature::TextureView,
        )));
    }
    // A view whose function applies the texture's swizzle takes none (a storage view; an
    // extended-range texture's sampled view).
    let components = match desc.shader_swizzle {
        true => desc.components,
        false => record.sampled_swizzle().compose(desc.components),
    };
    if storage && components != ComponentMapping::IDENTITY {
        return Err(ViewRefusal::Unsupported(Unsupported::not_implemented(
            Feature::TextureSwizzle,
        )));
    }
    let (base_level, level_count) = match desc.level {
        Some(level) if level < record.level_count => (record.base_level + level, 1),
        Some(_) => return Err(ViewProblem::LevelRange.into()),
        None => (record.base_level, record.level_count),
    };
    Ok(VkView {
        view_type: desc.view_type,
        format: desc.format,
        components,
        aspect: desc.aspect,
        base_level,
        level_count,
        base_layer: record.base_layer,
        layer_count: record.layer_count,
        usage,
    })
}
